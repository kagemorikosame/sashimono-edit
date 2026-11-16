"""評価の台本に足した課題（#414 の 4 言いよどみを見つける・5 定点の映像のつなぎ目）

実物の素材と字幕起こしの部品は CI に無いので、合成の音と偽の字幕起こしで 1 周させる
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sashimono.core.commands import AddMarkers, Document, RemoveMarkers
from sashimono.core.io import load_project, save_project
from sashimono.core.model import Project, Transcript, TranscriptSegment, Word
from sashimono.core.model.timeline import Marker
from tests.check_media import write_video
from tests.test_eval_tasks_tool import _write_talk, tool

__all__ = ["tool"]

F = Fraction


def _words(path: Path, params: object) -> Transcript:
    """「えーと今日は」と「あのー外へ」を 1 字ずつの語で返す（Whisper の日本語の形）"""

    def line(start: float, end: float, *items: tuple[str, float, float]) -> TranscriptSegment:
        words = tuple(
            Word(F(a).limit_denominator(100), F(b).limit_denominator(100), text)
            for text, a, b in items
        )
        return TranscriptSegment(
            F(start).limit_denominator(100),
            F(end).limit_denominator(100),
            "".join(word.text for word in words),
            words=words,
        )

    return Transcript(
        segments=(
            line(0.2, 1.6, ("え", 0.2, 0.4), ("ー", 0.4, 0.6), ("と", 0.6, 0.8), ("今", 1.0, 1.6)),
            line(4.5, 6.0, ("あ", 4.5, 4.7), ("の", 4.7, 4.9), ("ー", 4.9, 5.1), ("外", 5.4, 6.0)),
        ),
        language="ja",
    )


@pytest.fixture
def folder(tool: ModuleType, tmp_path: Path) -> tuple[ModuleType, Any, Any]:
    media = tmp_path / "ひみつの素材置き場"
    media.mkdir()
    audio = _write_talk(media / "しゃべりの録音.wav")
    # 絵と音のある素材 真ん中に無音があるもの（つなぎ目 1 つ）と、終わりだけが無音のもの
    middle = write_video(media / "まんなか.mp4", 8.0, tone_db=-12.0, silent=(3.0, 5.0))
    tail = write_video(media / "おわり.mp4", 8.0, tone_db=-12.0, silent=(6.0, 8.0))
    root = tmp_path / "evaldir"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"E02": str(audio), "E03": str(middle), "E04": str(tail)}, ensure_ascii=False),
        encoding="utf-8",
    )
    book_path = tmp_path / "tasks.json"
    book_path.write_text(
        json.dumps(
            {
                "materials": [
                    {"id": "E02", "tasks": ["fillers"]},
                    {"id": "E03", "tasks": ["seams"]},
                    {"id": "E04", "tasks": ["seams"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    return tool, tool.EvalDir(root), tool.load_tasks(book_path)


def test_the_filler_draft_marks_what_the_tool_found(folder: tuple[ModuleType, Any, Any]) -> None:
    """言いよどみの下書きは、道具が見つけた頭に目印を置き、探す手がかりの字幕も付ける

    壊れると、本人が直す .sme に目印か字幕が無く、53 分を一から聞き直すことになる
    """
    tool, eval_dir, book = folder
    made = tool.command_draft(eval_dir, book, ["E02"], word_transcriber=_words)
    assert made == ["E02 fillers"]
    answer = eval_dir.read_answer("E02")["tasks"]["fillers"]
    assert answer["provisional"] is True
    assert answer["points"] == [0.2, 4.5]
    project = load_project(eval_dir.draft_path("E02", "fillers"))
    assert [marker.label for marker in project.timeline.markers] == ["言いよどみ"] * 2
    # 目印は見つけた頭のフレームに置く 違う所に置くと、本人が違う所を聞きに行く
    fps = float(project.rate.fps)
    assert [marker.frame for marker in project.timeline.markers] == [
        round(0.2 * fps),
        round(4.5 * fps),
    ]
    (media,) = project.media
    transcript = media.transcript_for(None)
    assert transcript is not None
    assert any(segment.words for segment in transcript.segments)


def test_the_filler_answer_comes_from_the_fixed_markers(
    folder: tuple[ModuleType, Any, Any],
    tmp_path: Path,
) -> None:
    """本人が目印を消し・足した .sme を正解にして、道具の頭と比べる

    壊れると、直した目印が正解に入らず、道具の下書きと道具を比べた 1.0 が精度に見え続ける
    """
    tool, eval_dir, book = folder
    tool.command_draft(eval_dir, book, ["E02"], word_transcriber=_words)
    sheet = eval_dir.draft_path("E02", "fillers")
    document = Document(load_project(sheet))
    project: Project = document.project
    # 2 つ目（あのー）は言いよどみではなかったので消し、道具が見落とした 7 秒の「えー」を足した
    second = project.timeline.markers[1]
    document.execute(RemoveMarkers((second.frame,)))
    document.execute(AddMarkers((Marker(frame=round(7.0 * float(project.rate.fps))),)))
    save_project(document.project, sheet)
    tool.command_adopt(eval_dir, "E02", "fillers")
    answer = eval_dir.read_answer("E02")["tasks"]["fillers"]
    assert answer["provisional"] is False
    assert answer["points"] == pytest.approx([0.2, 7.0], abs=0.05)

    [record] = tool.command_run(
        eval_dir,
        book,
        ["E02"],
        results=tmp_path / "r.jsonl",
        commit="",
        word_transcriber=_words,
    )
    assert record["status"] == "ok", record
    metrics = record["metrics"]
    assert (metrics["hits"], metrics["precision"], metrics["recall"]) == (1, 0.5, 0.5)
    assert metrics["f1"] == 0.5
    assert tool.HEADLINE[tool.FILLERS] == "f1"


def _clips(project: Project) -> int:
    return sum(len(track.clips) for track in project.timeline.tracks)


def _seam_run(
    tool: ModuleType, eval_dir: Any, book: Any, tmp_path: Path, material: str
) -> tuple[dict[str, Any], dict[str, Project]]:
    """つなぎ目の課題を 1 本回し、書き出しへ渡った作品を処理ごとに返す"""
    written: dict[str, Project] = {}

    def export(project: Project, output: Path) -> None:
        written[output.name] = project
        output.write_bytes(b"")

    [record] = tool.command_run(
        eval_dir, book, [material], results=tmp_path / "r.jsonl", commit="", seam_exporter=export
    )
    assert record["status"] == "ok", record
    return record, written


def test_the_seams_are_written_once_per_style(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """無音を詰めたつなぎ目を、何もしない・クロスフェード・寄りの 3 通りで書き出す

    壊れると、本人が見比べる物の 1 つが無いか、どれも同じ処理のまま書き出される
    """
    tool, eval_dir, book = folder
    record, written = _seam_run(tool, eval_dir, book, tmp_path, "E03")
    names = [f"E03_seams_{video}.partial.mp4" for video in ("none", "crossfade", "zoom")]
    assert list(written) == names
    plain, faded, zoomed = (written[name] for name in names)
    # クロスフェードは場面切り替えのクリップを足し、寄りはクリップの形を変える 何もしないとは違う
    assert _clips(faded) > _clips(plain)
    assert zoomed != plain and _clips(zoomed) == _clips(plain)
    # 書き終えてから付け替える 本人が開くのは .partial の付かない名前
    assert sorted(path.name for path in eval_dir.exports.iterdir()) == sorted(
        name.replace(".partial", "") for name in names
    )
    metrics = record["metrics"]
    assert metrics["seams"] == 1
    assert metrics["removed_seconds"] > 0
    assert (metrics["handled_crossfade"], metrics["handled_zoom"]) == (1, 1)
    assert metrics["skipped_none"] == 0
    assert "provisional" not in record
    params = record["params"]
    assert params["threshold_db"] == -40.0
    assert params["keep_speech"] is False
    assert "audio_fade_ms" in params


def test_the_end_of_the_work_is_not_a_seam(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """終わりだけの無音を詰めた作品 壊れると、作品の終わりをつなぎ目と数え、処理した数と食い違う"""
    tool, eval_dir, book = folder
    # 無音の検出は終わりに余白を残す（素材の終わりまで削らない）ので、終わりまで削る無音を渡す
    monkeypatch.setattr(tool, "find_silences", lambda _path, _params: [(6.0, 8.0)])
    record, _written = _seam_run(tool, eval_dir, book, tmp_path, "E04")
    assert record["metrics"]["seams"] == 0
    assert record["metrics"]["removed_seconds"] == 2.0


def test_a_failed_export_leaves_no_half_set(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """2 本目の書き出しで落ちた 壊れると、前の回の書き出しと新しい 1 本が混ざって残る"""
    tool, eval_dir, book = folder
    eval_dir.exports.mkdir(parents=True, exist_ok=True)
    old = eval_dir.exports / "E03_seams_none.mp4"
    old.write_bytes(b"old")

    def export(project: Project, output: Path) -> None:
        output.write_bytes(b"new")
        if "crossfade" in output.name:
            # 途中まで書いてから落ちた 半端な物も残さない
            raise RuntimeError("書き出しに失敗した")

    [record] = tool.command_run(
        eval_dir, book, ["E03"], results=tmp_path / "r.jsonl", commit="", seam_exporter=export
    )
    assert (record["status"], record["reason"]) == ("failed", "export-error")
    assert old.read_bytes() == b"old"
    assert [path.name for path in eval_dir.exports.iterdir()] == ["E03_seams_none.mp4"]


def test_a_failed_swap_brings_back_the_previous_set(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 本は書けたが、2 本目の付け替えで落ちた（開いたままの書き出しへのアクセス拒否など）

    壊れると、今回の 1 本と前の回の 2 本が並び、本人が違う回の処理を見比べる
    """
    tool, eval_dir, book = folder
    eval_dir.exports.mkdir(parents=True, exist_ok=True)
    styles = ("none", "crossfade", "zoom")
    for video in styles:
        (eval_dir.exports / f"E03_seams_{video}.mp4").write_bytes(b"old")

    def export(project: Project, output: Path) -> None:
        output.write_bytes(b"new")

    real_replace = Path.replace

    def replace(self: Path, target: Path) -> Path:
        if self.name == "E03_seams_crossfade.partial.mp4":
            raise PermissionError("開いたままの書き出しは置き換えられない")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    [record] = tool.command_run(
        eval_dir, book, ["E03"], results=tmp_path / "r.jsonl", commit="", seam_exporter=export
    )
    assert (record["status"], record["reason"]) == ("failed", "export-error")
    assert sorted(path.name for path in eval_dir.exports.iterdir()) == sorted(
        f"E03_seams_{video}.mp4" for video in styles
    )
    for video in styles:
        assert (eval_dir.exports / f"E03_seams_{video}.mp4").read_bytes() == b"old"


def test_a_failed_restore_does_not_stop_the_rest(
    folder: tuple[ModuleType, Any, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """付け替えで落ち、戻す側でも 1 本目が落ちた

    壊れると、戻せる 2 本まで退避した名前のまま残り、本人が開く名前から消える
    """
    tool, eval_dir, book = folder
    eval_dir.exports.mkdir(parents=True, exist_ok=True)
    styles = ("none", "crossfade", "zoom")
    for video in styles:
        (eval_dir.exports / f"E03_seams_{video}.mp4").write_bytes(b"old")

    def export(project: Project, output: Path) -> None:
        output.write_bytes(b"new")

    real_replace = Path.replace

    def replace(self: Path, target: Path) -> Path:
        if self.name in ("E03_seams_crossfade.partial.mp4", "E03_seams_none.previous.mp4"):
            raise PermissionError("開いたままの書き出しは置き換えられない")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    [record] = tool.command_run(
        eval_dir, book, ["E03"], results=tmp_path / "r.jsonl", commit="", seam_exporter=export
    )
    assert (record["status"], record["reason"]) == ("failed", "export-error")
    for video in ("crossfade", "zoom"):
        assert (eval_dir.exports / f"E03_seams_{video}.mp4").read_bytes() == b"old"
    # 戻せなかった物は退避した名前のまま残す 今回の物は残さない
    assert (eval_dir.exports / "E03_seams_none.previous.mp4").read_bytes() == b"old"
    assert not (eval_dir.exports / "E03_seams_none.mp4").exists()
    assert not list(eval_dir.exports.glob("*.partial.mp4"))
    # 戻せなかった名前と戻し方を本人に見せる 結果の行には失敗の型だけを残す
    shown = capsys.readouterr().out
    assert "E03_seams_none.previous.mp4" in shown
    assert ".previous を外せば戻る" in shown


@pytest.mark.parametrize("bad", [-0.5, float("nan")])
def test_a_broken_tolerance_is_refused(tool: ModuleType, tmp_path: Path, bad: float) -> None:
    """負か有限でない許す幅 壊れると、当たりを 1 つも数えない誤った f1 が ok のまま残る"""
    book = tmp_path / "tasks.json"
    book.write_text(
        json.dumps({"tolerances": {"filler_seconds": bad}, "materials": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="filler_seconds"):
        tool.load_tasks(book)
