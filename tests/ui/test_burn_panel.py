"""字幕パネルの焼き込みと〔選んだ行を置く〕、AI の place_subtitles

焼き込みは話し手（素材と音声）と見た目を窓で選び、選んだ行だけを置く入口もある 見た目は
タイムラインで選んだテキスト・既定・保存したプリセットとエイリアス・テンプレートの棚から
見本の絵つきの一覧で選ぶ（#308） 話し手ごとに別の見た目にでき、どれも 1 回の取り消しで全部戻る

窓は試験の中で組んで操作する 押す人を待つ窓（``exec`` や確かめの窓）が開いたら、その場で落とす
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QListWidget, QMessageBox

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.compat.catalog import TemplateCatalog
from sashimono.compat.subtitle_looks import (
    DEFAULT,
    SELECTED,
    LookChoice,
    alias_choice,
    default_choice,
    selected_choice,
    template_choice,
)
from sashimono.core.commands import AddClip, AddTrack, Command, Voice
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.io.aliases import Alias
from sashimono.core.io.presets import Preset
from sashimono.core.model import AnimatedValue, Clip, Project, Track, TrackKind
from sashimono.effects.sources import SHAPE, TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.library_thumbnails import THUMBNAILS_SIMPLE, LookThumbnails
from sashimono.ui.library_view import LibraryOptions
from sashimono.ui.main_window import MainWindow
from sashimono.ui.subtitle.dialogs import BurnAnswer, BurnDialog, BurnRequest
from sashimono.ui.subtitle.panel import SubtitlePanel
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import FakeHost
from tests.compat.test_subtitle_looks import _styled_alias
from tests.core.test_burn_voices import _project


def accept_everyone(request: BurnRequest) -> BurnAnswer:
    """焼き込みの窓の代わり 話し手を全部選び、最初に選ばれている見た目のまま進める"""
    selected = request.sources.selected
    look: LookChoice = (
        selected_choice(selected)
        if request.initial == SELECTED and selected is not None
        else default_choice()
    )
    return BurnAnswer([voice for voice, _ in request.voices], look)


@pytest.fixture(autouse=True)
def no_window_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """押す人を待つ窓が開いたら、その場で落とす 開くと試験が止まったまま終わらない"""

    def opened(*_args: object, **_kwargs: object) -> object:
        pytest.fail("押す人を待つ窓が開いた")

    monkeypatch.setattr(QDialog, "exec", opened)
    for name in ("question", "information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(opened))
    for name in ("getSaveFileName", "getOpenFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(opened))


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


@pytest.fixture
def thumbnails(tmp_path: Path) -> Iterator[LookThumbnails]:
    created = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_SIMPLE)
    yield created
    created.release()


@pytest.fixture
def window(qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    del qt_application
    project, _media = _project()
    created = MainWindow(project, confirm_unsaved=False)
    # 素材は本物のファイルではない 裏で読みに行かせない
    monkeypatch.setattr(created, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(created._analyzer, "request", lambda _media, **_kwargs: None)
    # 棚は試験の外（本人の AviUtl2 の置き場）を読まない
    created._subtitles.shelf_provider = list
    yield created
    created.close()


def _texts(commands: list[Command]) -> list[str]:
    return [
        str(c.clip.source.params["text"])
        for c in commands
        if isinstance(c, AddClip) and c.clip.source is not None
    ]


def _wait(condition: Callable[[], bool], seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def _subtitle_clips(project: Project) -> dict[str, list[Clip]]:
    return {t.name: list(t.clips) for t in project.timeline.tracks if t.name.startswith("字幕")}


def _param(clip: Clip, name: str) -> object:
    assert clip.source is not None
    return clip.source.params[name]


def _shape_alias() -> Alias:
    shape = with_fixed_items(
        Clip(timeline_start=0, duration=30, source=SHAPE.create()), picture=True
    )
    return Alias.of("丸", shape)


def _mincho() -> Preset:
    saved = Clip(
        timeline_start=0, duration=30, source=TEXT.create(text="保存", font="Yu Mincho", size=70.0)
    )
    return Preset.capture("明朝", with_fixed_items(saved, picture=True))


class TestBurning:
    def test_the_window_lists_each_voice_and_the_look(
        self, panel: tuple[SubtitlePanel, list[list[Command]]]
    ) -> None:
        widget, sent = panel
        asked: list[BurnRequest] = []

        def only_the_second(request: BurnRequest) -> BurnAnswer:
            asked.append(request)
            return BurnAnswer([request.voices[1][0]], default_choice())

        widget.ask_burn = only_the_second
        widget.burn()
        (request,) = asked
        assert [label for _, label in request.voices] == ["録画.mp4 音声 1", "録画.mp4 音声 2"]
        # 何も選んでいなければ、今まで通り既定の見た目が最初に選ばれている
        assert request.initial == DEFAULT
        assert "既定" in request.describe(default_choice())
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
        asked: list[BurnRequest] = []

        def everyone(request: BurnRequest) -> BurnAnswer:
            asked.append(request)
            return accept_everyone(request)

        widget.ask_burn = everyone
        widget.burn()
        # 選んでいるテキストがあれば、今まで通りそれが最初に選ばれている
        assert asked[0].initial == SELECTED
        assert "見本" in asked[0].describe(selected_choice(styled))
        placed = [c.clip for c in sent[0] if isinstance(c, AddClip)]
        assert placed and all(
            getattr(c.source.params["size"], "static", None) == 90.0
            for c in placed
            if c.source is not None
        )

    def test_the_selected_rows_are_placed_with_the_chosen_look(
        self, panel: tuple[SubtitlePanel, list[list[Command]]]
    ) -> None:
        # 壊れると、〔選んだ行を置く〕だけは見た目を選べず、既定の見た目で置かれる
        widget, sent = panel
        widget.select_stream(1)
        selection = widget._table.selectionModel()
        selection.clearSelection()
        index = widget._table.model().index(1, 0)
        selection.select(
            index,
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
        asked: list[BurnRequest] = []

        def with_alias(request: BurnRequest) -> BurnAnswer:
            asked.append(request)
            return BurnAnswer([v for v, _ in request.voices], alias_choice(_styled_alias()))

        widget.ask_burn = with_alias
        widget.place_selected_rows()
        # 見ている 1 本だけ 話し手は選ばせない
        assert not asked[0].choose_voices and len(asked[0].voices) == 1
        (commands,) = sent
        assert _texts(commands) == ["二つ目"]
        assert any(isinstance(c, AddTrack) for c in commands)
        clip = next(c.clip for c in commands if isinstance(c, AddClip))
        assert _param(clip, "font") == "Meiryo"


class TestTheWindow:
    def test_a_remembered_shelf_look_that_no_longer_has_text_falls_back(
        self,
        panel: tuple[SubtitlePanel, list[list[Command]]],
        thumbnails: LookThumbnails,
        tmp_path: Path,
    ) -> None:
        # 壊れると、覚えた棚のテンプレートが書き換わって文字を持たなくなったとき、窓が断られた
        # 選択のまま開き、〔焼き込む〕が押せない（既定の見た目へ戻らない CodeRabbit の指摘）
        widget, _sent = panel
        shelf = tmp_path / "棚"
        shelf.mkdir()
        (shelf / "丸だけ.object").write_text(
            "\n".join(["[Object]", "frame=0,89", "[Object.0]", "effect.name=図形", ""]),
            "utf-8",
        )
        (entry,) = TemplateCatalog().scan((shelf,))
        widget.shelf_provider = lambda: (entry,)
        widget.remember_look = True
        widget.last_look = template_choice(entry).key
        request = widget._request([(None, 0)], title="置く", accept="置く", choose=False)
        dialog = BurnDialog(
            request, thumbnails=thumbnails, options=LibraryOptions(thumbnails=THUMBNAILS_SIMPLE)
        )
        try:
            answer = dialog.answer()
            assert answer is not None and answer.look.key == DEFAULT
        finally:
            dialog.reject()
            dialog.deleteLater()

    def test_text_less_aliases_are_greyed_and_thumbnails_are_drawn(
        self,
        panel: tuple[SubtitlePanel, list[list[Command]]],
        thumbnails: LookThumbnails,
    ) -> None:
        # 壊れると、図形のエイリアスを選べてしまうか、一覧に見本の絵が出ない（#277 の描画）
        widget, _sent = panel
        library = widget.library_provider()
        library.aliases.save(_styled_alias("黄色"))
        library.aliases.save(_shape_alias())
        request = widget._request([(None, 0)], title="置く", accept="置く", choose=False)
        dialog = BurnDialog(
            request, thumbnails=thumbnails, options=LibraryOptions(thumbnails=THUMBNAILS_SIMPLE)
        )
        try:
            dialog.show()
            picker = dialog.picker
            picker.show_tab("alias")
            shown = {c.label: c for c in picker.shown()}
            assert shown["丸"].refusal and not shown["黄色"].refusal
            listing = picker.findChild(QListWidget, "subtitle_looks")
            assert listing is not None
            items = {listing.item(row).text(): listing.item(row) for row in range(listing.count())}
            assert not items["丸"].flags() & Qt.ItemFlag.ItemIsEnabled
            assert not dialog.choose("alias:丸")
            answer = dialog.answer()
            assert answer is not None and answer.look.key == DEFAULT
            # 見本の地だけの絵から、描いた見本の絵に貼り替わる
            yellow = items["黄色"]
            backdrop = yellow.icon().cacheKey()
            _wait(lambda: yellow.icon().cacheKey() != backdrop)
            assert yellow.icon().cacheKey() != backdrop
            saved = next(e.item for e in library.entries("alias") if e.name == "黄色")
            assert thumbnails.cached(saved) is not None
        finally:
            dialog.reject()
            dialog.deleteLater()


class TestInTheEditor:
    def test_choose_looks_per_voice_place_and_undo_once(
        self, window: MainWindow, thumbnails: LookThumbnails
    ) -> None:
        """字幕を起こした作品で、焼き込みの窓から全員と話し手 2 の見た目を選んで置く

        壊れると、選んだ見た目が置いたクリップに写らない・話し手ごとに分かれない・
        取り消し 1 回で置いた字幕が戻らない、のどれかになる
        """
        project = window.document.project
        library = window.look_library()
        library.aliases.save(_styled_alias("黄色"))
        library.presets.save(_mincho())
        media = project.media[0]
        second: Voice = (media.id, 2)
        panel = window._subtitles

        def through_the_window(request: BurnRequest) -> BurnAnswer | None:
            dialog = BurnDialog(
                request,
                thumbnails=thumbnails,
                options=LibraryOptions(thumbnails=THUMBNAILS_SIMPLE),
                parent=window,
            )
            try:
                dialog.show()
                QApplication.processEvents()
                assert dialog.choose("alias:黄色")
                preset_key = next(c.key for c in _all(dialog, "preset") if c.label == "明朝")
                assert dialog.choose(preset_key, voice=second)
                return dialog.answer()
            finally:
                dialog.reject()
                dialog.deleteLater()

        panel.ask_burn = through_the_window
        panel.burn()
        placed = _subtitle_clips(window.document.project)
        first = placed["字幕 音声 1"]
        assert [_param(c, "text") for c in first] == ["ゲームの声", "二つ目"]
        assert {_param(c, "font") for c in first} == {"Meiryo"}
        assert all(c.source is not None and c.source.strokes for c in first)
        assert [c.duration for c in first] == [60, 30]
        (mine,) = placed["字幕 音声 2"]
        assert _param(mine, "text") == "マイクの声"
        assert _param(mine, "font") == "Yu Mincho"
        assert _param(mine, "size") == AnimatedValue(70.0)
        window.undo()
        assert window.document.project == project

    def test_the_last_look_is_remembered_only_when_asked(self, window: MainWindow) -> None:
        # 壊れると、覚える設定を入れても次の窓が既定の見た目から始まる（切っていても覚える）
        window.look_library().aliases.save(_styled_alias("黄色"))
        panel = window._subtitles
        asked: list[BurnRequest] = []

        def yellow(request: BurnRequest) -> BurnAnswer:
            asked.append(request)
            return BurnAnswer([v for v, _ in request.voices], alias_choice(_styled_alias("黄色")))

        panel.ask_burn = yellow
        panel.burn()
        window.undo()
        panel.burn()
        assert [r.initial for r in asked] == [DEFAULT, DEFAULT]
        assert PreferenceStore().load().subtitle_look_last == ""

        # 設定画面で OK を押したのと同じ道で入れる
        window._apply_preferences(Preferences(subtitle_look_remember=True))
        window.undo()
        panel.burn()
        window.undo()
        panel.burn()
        assert asked[-1].initial == "alias:黄色"
        assert PreferenceStore().load().subtitle_look_last == "alias:黄色"


def _all(dialog: BurnDialog, tab: str) -> list[LookChoice]:
    dialog.picker.show_tab(tab)
    return dialog.picker.shown()


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

    def test_it_chooses_a_saved_look_by_name(self) -> None:
        # 壊れると、AI に「黄色テロップで字幕を置いて」と頼んでも既定の見た目で置かれる
        project, _media = _project()
        host = FakeHost(project)
        host.library.aliases.save(_styled_alias("黄色"))
        host.library.presets.save(_mincho())
        operation = find_operation("place_subtitles")
        assert operation is not None
        operation(host, {"look_name": "黄色"})
        fonts = {
            _param(c, "font")
            for clips in _subtitle_clips(host.document.project).values()
            for c in clips
        }
        assert fonts == {"Meiryo"}
        host.document.undo()
        operation(host, {"look_kind": "preset", "look_name": "明朝"})
        texts = {
            (_param(c, "text"), _param(c, "font"))
            for clips in _subtitle_clips(host.document.project).values()
            for c in clips
        }
        assert ("マイクの声", "Yu Mincho") in texts

    def test_an_unknown_or_text_less_look_is_refused(self) -> None:
        # 壊れると、名前を間違えたときに黙って既定の見た目で置くか、図形を字幕にする
        project, _media = _project()
        host = FakeHost(project)
        host.library.aliases.save(_shape_alias())
        operation = find_operation("place_subtitles")
        assert operation is not None
        with pytest.raises(ToolError, match="無い"):
            operation(host, {"look_name": "黄色"})
        with pytest.raises(ToolError, match="文字を持たない"):
            operation(host, {"look_name": "丸"})
        assert host.document.project == project
