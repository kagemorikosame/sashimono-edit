"""書き出しを別のプロセスで走らせる（#453）

子を本当に起こして書き出させるのは 1 本だけ（GL と FFmpeg を子で読むので数秒かかる）
戻る道（子を起こせない・子で GL を作れない・子が落ちた）は、子を起こす所を差し替えて確かめる
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sashimono.core.commands import Document, insert_media
from sashimono.core.model import Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import probe_media
from sashimono.engine.encode import ExportError, ExportSettings
from sashimono.engine.encode import export_job as job_module
from sashimono.engine.encode.export_job import (
    GL_UNAVAILABLE_MARK,
    ExportOutcome,
    export_anywhere,
)
from sashimono.engine.worker import (
    WorkerCrashedError,
    WorkerFailedError,
    WorkerUnavailableError,
)
from tests.media_fixtures import SampleMedia

pytestmark = pytest.mark.usefixtures("gpu")


@pytest.fixture
def project(sample_av: SampleMedia) -> Project:
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(sample_av.path)):
        document.execute(command)
    return document.project


def test_a_child_writes_the_file(project: Project, tmp_path: Path) -> None:
    """壊れると、設定で別のプロセスを選んだ書き出しが何も書かずに終わるか、進み具合が出ない"""
    seen: list[float] = []
    target = tmp_path / "out.mp4"
    written = export_anywhere(
        project, ExportSettings(path=target), separate_process=True, progress=seen.append
    )
    assert written == target and target.stat().st_size > 0
    assert seen and seen[-1] == pytest.approx(1.0)


def _fake_run(outcome: object, *, write: bytes | None = None) -> object:
    """子の代わり ``write`` を渡すと、子が書いたように渡された書き出し先へ書く"""

    def run(_job: object, options: dict[str, Any], **_kwargs: object) -> object:
        if write is not None:
            Path(options["settings"].path).write_bytes(write)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return run


@pytest.mark.parametrize(
    "failure",
    [
        WorkerUnavailableError("起こせない"),
        WorkerFailedError(f"RuntimeError: {GL_UNAVAILABLE_MARK}: 版が足りない"),
    ],
)
def test_it_falls_back_to_this_process(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """壊れると、子を起こせない機械や子で GL を作れない機械（リモート接続など）で書き出せない"""
    monkeypatch.setattr(job_module, "run_in_process", _fake_run(failure))
    target = tmp_path / "out.mp4"
    assert export_anywhere(project, ExportSettings(path=target), separate_process=True) == target
    assert target.stat().st_size > 0


def test_an_export_error_in_the_child_reads_as_is(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、書き出しの失敗の文の頭に型の名前が付いて利用者に出るか、スレッドで書き直して倍待たせる"""
    monkeypatch.setattr(
        job_module, "run_in_process", _fake_run(WorkerFailedError("ExportError: 書けない"))
    )
    called: list[object] = []
    monkeypatch.setattr(job_module, "export_project", lambda *a, **k: called.append(a))
    with pytest.raises(ExportError, match=r"^書けない$"):
        export_anywhere(project, ExportSettings(path=tmp_path / "x.mp4"), separate_process=True)
    assert called == []


def test_a_crash_is_not_retried_here(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、素材のせいで子が落ちたとき、同じ素材をアプリの中で読み直してアプリごと落ちる"""
    monkeypatch.setattr(job_module, "run_in_process", _fake_run(WorkerCrashedError("落ちた")))
    called: list[object] = []
    monkeypatch.setattr(job_module, "export_project", lambda *a, **k: called.append(a))
    with pytest.raises(ExportError, match="落ちた"):
        export_anywhere(project, ExportSettings(path=tmp_path / "x.mp4"), separate_process=True)
    assert called == []


def test_notes_from_the_child_reach_the_caller(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、子が別のエンコーダで書き直したことが利用者に伝わらない（#365）"""
    monkeypatch.setattr(
        job_module,
        "run_in_process",
        _fake_run(ExportOutcome(path="", notes=("CPU で書いた",)), write=b"done"),
    )
    notes: list[str] = []
    export_anywhere(
        project, ExportSettings(path=tmp_path / "x.mp4"), separate_process=True, notice=notes.append
    )
    assert notes == ["CPU で書いた"]


def test_cancelling_in_the_child_reads_as_cancelled(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、取り消した書き出しが「書き出しました」と出て、無いファイルを指す"""
    monkeypatch.setattr(job_module, "run_in_process", _fake_run(None))
    with pytest.raises(ExportError, match="中止"):
        export_anywhere(project, ExportSettings(path=tmp_path / "x.mp4"), separate_process=True)


@pytest.mark.parametrize(
    "result",
    [WorkerCrashedError("落ちた"), None],
    ids=["落ちた子", "止めた子"],
)
def test_a_child_that_did_not_finish_leaves_the_target_as_it_was(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    """壊れると、子が落ちた・猶予を過ぎて止めたとき、途中で切れた動画が残るか、前からあった
    ファイルまで消える"""
    target = tmp_path / "x.mp4"
    target.write_bytes(b"before")
    monkeypatch.setattr(job_module, "run_in_process", _fake_run(result, write=b"half"))
    with pytest.raises(ExportError):
        export_anywhere(project, ExportSettings(path=target), separate_process=True)
    assert target.read_bytes() == b"before"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["x.mp4"]


def test_a_finished_child_replaces_the_target(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、子が書き終えても書き出し先が前のままか、仮のファイルが横に残る"""
    target = tmp_path / "x.mp4"
    target.write_bytes(b"before")
    monkeypatch.setattr(
        job_module, "run_in_process", _fake_run(ExportOutcome(path="", notes=()), write=b"new")
    )
    assert export_anywhere(project, ExportSettings(path=target), separate_process=True) == target
    assert target.read_bytes() == b"new"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["x.mp4"]
