"""字幕パネルの焼き込みと〔選んだ行を置く〕、AI の place_subtitles

焼き込みは話し手（素材と音声）を窓で選び、選んだ行だけを置く入口もある 見た目は
タイムラインで選んだテキストを写す どれも 1 回の取り消しで全部戻る
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip, AddTrack, Command, Voice
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import Clip, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.subtitle.panel import SubtitlePanel
from tests.ai.conftest import FakeHost
from tests.core.test_burn_voices import _project


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[tuple[SubtitlePanel, list[list[Command]]]]:
    del qt_application
    project, _media = _project()
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    created = SubtitlePanel(project, analyzer)
    sent: list[list[Command]] = []
    created.commands_requested.connect(lambda commands, _label: sent.append(list(commands)))
    yield created, sent
    created.close()
    analyzer.close()


def _texts(commands: list[Command]) -> list[str]:
    return [
        str(c.clip.source.params["text"])
        for c in commands
        if isinstance(c, AddClip) and c.clip.source is not None
    ]


class TestBurning:
    def test_the_window_lists_each_voice_and_the_look(
        self, panel: tuple[SubtitlePanel, list[list[Command]]]
    ) -> None:
        widget, sent = panel
        asked: list[tuple[list[tuple[Voice, str]], str]] = []

        def only_the_second(voices: list[tuple[Voice, str]], note: str) -> list[Voice]:
            asked.append((voices, note))
            return [voices[1][0]]

        widget.ask_burn = only_the_second
        widget.burn()
        ((voices, note),) = asked
        assert [label for _, label in voices] == ["録画.mp4 音声 1", "録画.mp4 音声 2"]
        assert "既定" in note
        # 選んだ話し手だけ 1 回の命令の並びで出す（取り消しは 1 回）
        (commands,) = sent
        assert _texts(commands) == ["マイクの声"]

    def test_a_selected_text_is_copied(
        self, panel: tuple[SubtitlePanel, list[list[Command]]]
    ) -> None:
        widget, sent = panel
        styled = with_fixed_items(
            Clip(timeline_start=0, duration=30, source=TEXT.create(text="見本", size=90.0)),
            picture=True,
        )
        widget.template_provider = lambda: styled
        notes: list[str] = []

        def everyone(voices: list[tuple[Voice, str]], note: str) -> list[Voice]:
            notes.append(note)
            return [voice for voice, _ in voices]

        widget.ask_burn = everyone
        widget.burn()
        assert "見本" in notes[0]
        placed = [c.clip for c in sent[0] if isinstance(c, AddClip)]
        assert placed and all(
            getattr(c.source.params["size"], "static", None) == 90.0
            for c in placed
            if c.source is not None
        )

    def test_the_selected_rows_are_placed(
        self, panel: tuple[SubtitlePanel, list[list[Command]]]
    ) -> None:
        widget, sent = panel
        widget.select_stream(1)
        selection = widget._table.selectionModel()
        selection.clearSelection()
        index = widget._table.model().index(1, 0)
        selection.select(
            index,
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
        widget.place_selected_rows()
        (commands,) = sent
        assert _texts(commands) == ["二つ目"]
        assert any(isinstance(c, AddTrack) for c in commands)


class TestTheAssistant:
    def test_it_can_place_subtitles_as_text(self) -> None:
        # 前は道具が無く、AI は字幕をテキストオブジェクトとして置けなかった
        project, media = _project()
        host = FakeHost(project)
        operation = find_operation("place_subtitles")
        assert operation is not None
        result = operation(host, {"media_id": str(media.id), "audio": 2})
        assert result == {"placed": 1, "tracks": ["字幕"]}
        placed = [
            c
            for t in host.document.project.timeline.tracks
            if t.kind in (TrackKind.MIXED, TrackKind.VIDEO)
            for c in t.clips
        ]
        assert [str(c.source.params["text"]) for c in placed if c.source is not None] == [
            "マイクの声"
        ]
        host.document.undo()
        assert host.document.project == project

    def test_it_uses_a_template_clip(self) -> None:
        project, _media = _project()
        styled = Clip(timeline_start=0, duration=30, source=TEXT.create(text="見本", size=77.0))
        track = Track(TrackKind.VIDEO, "テロップ", (styled,))
        project = AddTrack(track).apply(project)
        host = FakeHost(project)
        operation = find_operation("place_subtitles")
        assert operation is not None
        operation(host, {"template_clip_id": str(styled.id)})
        sizes = {
            getattr(c.source.params.get("size"), "static", None)
            for t in host.document.project.timeline.tracks
            if t.name.startswith("字幕")
            for c in t.clips
            if c.source is not None
        }
        assert sizes == {77.0}
