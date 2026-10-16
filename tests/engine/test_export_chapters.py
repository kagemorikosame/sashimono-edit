"""書き出す動画に、名前の付いた目印をチャプターとして埋める（#330）

2026-10-17 に利用者が決めた 既定は入 名前の付いた目印があり、mp4・mov・mkv のときだけ効く
書いたファイルを PyAV で読み戻し、プレーヤーが章送りに使う所に入ったかを見る
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from sashimono.core.commands import Document, insert_media
from sashimono.core.model import Marker, Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import probe_media
from sashimono.engine.encode import ExportSettings, export_project
from sashimono.engine.encode.exporter import export_chapters, write_chapters
from tests.media_fixtures import SampleMedia

#: 2 秒の素材に置く目印 名前の無い 1 つは目次に入らない
_MARKERS = (Marker(0, "はじめに"), Marker(15), Marker(30, "本編"), Marker(45, "まとめ"))


def _read_back(path: Path) -> list[tuple[Fraction, Fraction, str]]:
    """ファイルのチャプターを（始まりの秒, 終わりの秒, 名前）で読む"""
    with av.open(str(path)) as container:
        rows = []
        for chapter in container.chapters():
            base = chapter["time_base"] or Fraction(1, 1_000_000)
            rows.append(
                (
                    chapter["start"] * base,
                    chapter["end"] * base,
                    chapter["metadata"].get("title", ""),
                )
            )
        return rows


def _project_with_markers(sample_av: SampleMedia) -> Project:
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(sample_av.path)):
        document.execute(command)
    project = document.project
    return project.with_timeline(replace(project.timeline, markers=_MARKERS))


class TestList:
    """埋める物の一覧 GPU の無い機械でも見る"""

    def test_named_markers_become_entries_in_frames(self) -> None:
        # 壊れると、名前の無い覚え書きの目印までチャプターに出る
        project = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
        project = project.with_timeline(replace(project.timeline, markers=_MARKERS))
        entries = export_chapters(project, (0, 60), Path("出力.mp4"))
        assert [(e["start"], e["end"], e["metadata"]["title"]) for e in entries] == [
            (0, 30, "はじめに"),
            (30, 45, "本編"),
            (45, 60, "まとめ"),
        ]
        assert entries[0]["time_base"] == Fraction(1, 30)

    def test_the_range_head_is_zero(self) -> None:
        # 範囲の頭を引かないと、範囲を決めて書き出した動画で章の位置が範囲の頭の分だけ遅れる
        project = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
        project = project.with_timeline(replace(project.timeline, markers=_MARKERS))
        entries = export_chapters(project, (30, 60), Path("出力.mkv"))
        assert [(e["start"], e["metadata"]["title"]) for e in entries] == [
            (0, "本編"),
            (15, "まとめ"),
        ]

    @pytest.mark.parametrize("name", ["出力.webm", "出力.avi", "出力"])
    def test_other_containers_get_nothing(self, name: str) -> None:
        # 章を持てない入れ物へ渡すと、書き出しの最後で多重化に断られて動画ごと失敗しうる
        project = Project.create(ProjectSettings(frame_rate=FrameRate(30)))
        project = project.with_timeline(replace(project.timeline, markers=_MARKERS))
        assert export_chapters(project, (0, 60), Path(name)) == []

    @pytest.mark.parametrize("suffix", [".mp4", ".mov", ".mkv"])
    def test_written_chapters_read_back(self, tmp_path: Path, suffix: str) -> None:
        # 壊れると、書いたつもりのチャプターがファイルに入らず、プレーヤーで章送りできない
        project = Project.create(ProjectSettings(width=64, height=64, frame_rate=FrameRate(30)))
        project = project.with_timeline(replace(project.timeline, markers=_MARKERS))
        path = tmp_path / f"小さい{suffix}"
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            write_chapters(container, export_chapters(project, (0, 60), path))
            for index in range(60):
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24"
                )
                frame.pts = index
                container.mux(stream.encode(frame.reformat(format="yuv420p")))
            container.mux(stream.encode(None))
        assert [(start, title) for start, _end, title in _read_back(path)] == [
            (Fraction(0), "はじめに"),
            (Fraction(1), "本編"),
            (Fraction(3, 2), "まとめ"),
        ]


@pytest.mark.usefixtures("gpu")
class TestExport:
    @pytest.mark.parametrize("suffix", [".mp4", ".mkv"])
    def test_the_export_carries_the_chapters(
        self, sample_av: SampleMedia, tmp_path: Path, suffix: str
    ) -> None:
        # 壊れると、AI と人が付けた目次が書き出した動画に入らない
        output = tmp_path / f"章{suffix}"
        export_project(_project_with_markers(sample_av), ExportSettings(path=output))
        assert [(start, title) for start, _end, title in _read_back(output)] == [
            (Fraction(0), "はじめに"),
            (Fraction(1), "本編"),
            (Fraction(3, 2), "まとめ"),
        ]

    def test_turned_off_writes_none(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        # 切ったのに入ると、要らない人の動画に章が付く
        output = tmp_path / "章なし.mp4"
        export_project(
            _project_with_markers(sample_av), ExportSettings(path=output, chapters=False)
        )
        assert _read_back(output) == []
