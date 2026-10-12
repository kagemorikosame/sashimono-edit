"""トラックを足すボタンと、右クリックから物を足すメニュー（Issue #27）

入口が正しいコマンドを出し、右クリックした所（フレームとトラック）へ置くかを見る
置き先の決まりそのもの（空いたトラックを探すなど）は core 側の試験で見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPoint, Qt
from PySide6.QtGui import QAction, QDragMoveEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

from sashimono.compat.aviutl.catalog import ScriptCatalog, ScriptEntry
from sashimono.compat.catalog import TemplateEntry
from sashimono.core.commands import AddEffect, Command
from sashimono.core.io.aliases import AliasStore
from sashimono.core.model import (
    Clip,
    LayerMode,
    MediaItem,
    Project,
    Scene,
    Timeline,
    Track,
    TrackId,
    TrackKind,
)
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.library_save import SaveChoice
from sashimono.ui.main_window import MainWindow
from sashimono.ui.media_pool import MEDIA_MIME
from sashimono.ui.theme import Colors
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.add_menu import LAYER_ADD_TEXT, AddSources
from tests.conftest import make_clip

#: 置き場を持たない出どころ 試験から本人の AviUtl2 やテンプレートの置き場を読まない
_NOTHING: tuple[ScriptEntry, ...] = ()


def _project() -> Project:
    base = Project.create()
    text = Clip(timeline_start=0, duration=60, source=TEXT.create(text="見出し"))
    tracks = (
        Track(TrackKind.VIDEO, "V1", (text,)),
        Track(TrackKind.VIDEO, "V2"),
        Track(TrackKind.AUDIO, "A1"),
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _take_suggestion(
    *_args: object, suggestion: str, category: str, **_kwargs: object
) -> SaveChoice:
    """保存の窓の代わり 勧めた名前のまま、上書きの指定はせずに保存する"""
    return SaveChoice(name=suggestion, category=category, replacing=None)


def _sources(tmp_path: Path, **overrides: object) -> AddSources:
    sources = AddSources(
        custom_objects=lambda: _NOTHING,
        templates=lambda: (),
        aliases=AliasStore(tmp_path / "aliases"),
        ask_save=_take_suggestion,
    )
    for name, value in overrides.items():
        setattr(sources, name, value)
    return sources


def _wire(view: TimelineView) -> list[str]:
    """窓の代わりに、出たコマンドをその場で当てる 断られたら何も変えない"""
    labels: list[str] = []

    def run(commands: list[Command], label: str) -> None:
        project = view.project
        try:
            for command in commands:
                project = command.apply(project)
        except (ValueError, KeyError):
            return
        labels.append(label)
        view.set_project(project)

    view.commands_requested.connect(run)
    return labels


@pytest.fixture
def view(qt_application: QApplication, tmp_path: Path) -> Iterator[TimelineView]:
    del qt_application
    analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
    created = TimelineView(_project(), analyzer)
    created.add_sources = _sources(tmp_path)
    created.resize(900, 400)
    yield created
    analyzer.close()


def _track(view: TimelineView, name: str) -> Track:
    return next(t for t in view.project.timeline.tracks if t.name == name)


def _point(view: TimelineView, track: str, frame: int) -> QPoint:
    band = next(b for b in view._layout.bands(view.project.timeline) if b.track.name == track)
    return QPoint(int(view._layout.frame_to_x(frame)) + 1, band.top + band.height // 2)


def _find(menu: QMenu, *path: str) -> QAction:
    """見出しをたどって項目を探す サブメニューは開いたときと同じく中身を作らせる"""
    current = menu
    for index, text in enumerate(path):
        action = next((a for a in current.actions() if a.text() == text), None)
        assert action is not None, f"{text} が無い: {[a.text() for a in current.actions()]}"
        if index == len(path) - 1:
            return action
        submenu = action.menu()
        assert isinstance(submenu, QMenu), f"{text} はサブメニューではない"
        submenu.aboutToShow.emit()
        current = submenu
    raise AssertionError("見出しが空")


def _texts(menu: QMenu) -> list[str]:
    menu.aboutToShow.emit()
    return [a.text() for a in menu.actions() if a.text()]


class TestTrackAddButton:
    def test_the_button_sits_under_the_last_track_in_the_header(self, view: TimelineView) -> None:
        # トラックの並びから離れた所にあると、足すたびに探すことになる
        rect = view.track_add_button()
        assert rect is not None
        last = view._layout.bands(view.project.timeline)[-1]
        assert last.bottom < rect.top() < last.bottom + 20
        assert rect.right() < view._layout.frame_to_x(0)

    def test_the_button_follows_a_new_track(self, view: TimelineView) -> None:
        # 足したトラックの上に重なって残ると、次のトラックの名前が隠れる
        _wire(view)
        before = view.track_add_button()
        _find(view.build_track_add_menu(), "音声トラック").trigger()
        after = view.track_add_button()
        assert before is not None and after is not None
        assert after.top() > before.top()

    def test_pressing_the_button_opens_the_choices_without_moving_the_playhead(
        self, view: TimelineView, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ヘッダの空いた所の扱い（再生ヘッドを動かす）に落ちると、押しても何も出ない
        opened: list[QPoint] = []

        class Recorder(QMenu):
            def exec(self, position: QPoint) -> None:  # type: ignore[override]
                opened.append(position)

        monkeypatch.setattr(view, "build_track_add_menu", lambda: Recorder())
        view.set_playhead(40)
        rect = view.track_add_button()
        assert rect is not None
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=rect.center())
        assert len(opened) == 1
        assert view.playhead == 40

    def test_the_menu_offers_video_audio_and_effect_tracks(self, view: TimelineView) -> None:
        # 欠けると、その種類のトラックはボタンから足せず、読み込みやフィルタの置き場任せになる
        assert _texts(view.build_track_add_menu()) == [
            "映像トラック",
            "音声トラック",
            "エフェクトトラック（フィルタ用）",
        ]

    def test_a_mixed_project_offers_layers(self, view: TimelineView) -> None:
        # 混合の作品で映像・音声のトラックを足せると、方式を混合にしたのに分けたトラックが増える
        settings = replace(view.project.settings, layer_mode=LayerMode.MIXED)
        view.set_project(replace(view.project, settings=settings))
        _wire(view)
        # エフェクトのレイヤーも出さない（利用者の要望 分けない方式では普通のレイヤーに何でも置く）
        assert _texts(view.build_track_add_menu()) == ["レイヤー"]
        _find(view.build_track_add_menu(), "レイヤー").trigger()
        added = view.project.timeline.tracks[-1]
        assert (added.name, added.kind) == ("レイヤー 1", TrackKind.MIXED)

    def test_a_mixed_project_adds_a_layer_without_a_menu(
        self, view: TimelineView, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 選べるのがレイヤーだけなのにメニューを出すと、押すたびに 1 手余計に掛かる（利用者の要望）
        settings = replace(view.project.settings, layer_mode=LayerMode.MIXED)
        view.set_project(replace(view.project, settings=settings))
        labels = _wire(view)
        opened: list[QPoint] = []

        class Recorder(QMenu):
            def exec(self, position: QPoint) -> None:  # type: ignore[override]
                opened.append(position)

        monkeypatch.setattr(view, "build_track_add_menu", lambda: Recorder())
        before = len(view.project.timeline.tracks)
        rect = view.track_add_button()
        assert rect is not None
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=rect.center())
        assert opened == []
        assert len(view.project.timeline.tracks) == before + 1
        added = view.project.timeline.tracks[-1]
        assert added.kind is TrackKind.MIXED
        assert len(labels) == 1, "取り消し 1 回で戻る 1 つの操作にする"

    def test_a_video_track_goes_on_top_and_audio_at_the_bottom(self, view: TimelineView) -> None:
        # 映像は並びの末尾ほど手前 足したトラックが間に挟まると、重なり順が変わる
        _wire(view)
        _find(view.build_track_add_menu(), "映像トラック").trigger()
        _find(view.build_track_add_menu(), "音声トラック").trigger()
        shown = [b.track.name for b in view._layout.bands(view.project.timeline)]
        assert shown == ["V3", "V2", "V1", "A1", "A2"]

    def test_an_effect_track_is_a_named_video_track_on_top(self, view: TimelineView) -> None:
        # 種類を増やさず名前で見分ける 形式を変えると古い版で開けなくなる
        _wire(view)
        _find(view.build_track_add_menu(), "エフェクトトラック（フィルタ用）").trigger()
        top = view._layout.bands(view.project.timeline)[0].track
        assert (top.name, top.kind) == ("FX1", TrackKind.VIDEO)

    def test_the_button_is_drawn(self, view: TimelineView) -> None:
        # 当たり判定だけあって描かれていないと、押せる所があると気付けない
        rect = view.track_add_button()
        assert rect is not None
        image = view.grab().toImage()
        # 枠は矩形の縁の画素に乗るので、縁まで含めて拾う 内側だけを見ると、文字を
        # なめらかに描かない環境（CI）では下地と文字の 2 色しか拾えない
        inside = {
            image.pixelColor(x, y).name()
            for x in range(rect.left(), rect.right() + 1)
            for y in range(rect.top(), rect.bottom() + 1)
        }
        # 枠と文字と下地で 3 色以上 何も描かなければタイムラインの下地 1 色になる
        assert Colors.BORDER.name() in inside
        assert len(inside) >= 3


class TestWhileDragging:
    def test_the_button_moves_below_the_new_track_rows(
        self, qt_application: QApplication, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 素材を引いている間は、新しく作るトラックの仮の行が末尾（音声の下）に並ぶ
        # 本物の並びでボタンを置くと、その仮の行の上に重なって「新しく作る」が読めない
        del qt_application
        base = Project.create(media=(video_media, audio_media))
        tracks = (
            Track(TrackKind.VIDEO, "V1", (make_clip(0, 300, video_media),)),
            Track(TrackKind.AUDIO, "A1", (make_clip(0, 300, audio_media),)),
        )
        analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
        view = TimelineView(base.with_timeline(replace(base.timeline, tracks=tracks)), analyzer)
        try:
            view.resize(900, 400)
            band = view._layout.bands(view.project.timeline)[0]
            mime = QMimeData()
            mime.setData(MEDIA_MIME, str(video_media.id).encode("utf-8"))
            view.dragMoveEvent(
                QDragMoveEvent(
                    QPoint(int(view._layout.frame_to_x(30)), band.top + band.height // 2),
                    Qt.DropAction.CopyAction,
                    mime,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
            )
            preview = view.drop_preview
            assert preview is not None and preview.new_tracks
            shown = view._layout.bands(preview.timeline)
            assert shown[-1].track.id in preview.new_tracks
            rect = view.track_add_button()
            assert rect is not None
            assert rect.top() >= shown[-1].bottom
        finally:
            analyzer.close()


class TestTrackMenu:
    def test_a_track_can_be_added_from_its_menu(self, view: TimelineView) -> None:
        # 右クリックから足せないと、トラックが多くて下のボタンが隠れているとき、
        # 一番下まで送ってボタンを探すことになる
        _wire(view)
        menu = view.build_context_menu(_point(view, "A1", 100))
        _find(menu, "トラックを追加", "音声トラック").trigger()
        assert [t.name for t in view.project.timeline.audio_tracks()] == ["A1", "A2"]

    def test_a_mixed_project_adds_a_layer_straight_from_the_menu(self, view: TimelineView) -> None:
        # 中身が 1 つのサブメニューを開かせると、右クリックから足すのに 1 手余計に掛かる
        settings = replace(view.project.settings, layer_mode=LayerMode.MIXED)
        view.set_project(replace(view.project, settings=settings))
        _wire(view)
        menu = view.build_context_menu(_point(view, "A1", 100))
        action = _find(menu, LAYER_ADD_TEXT)
        assert action.menu() is None, "サブメニューではなく直に足す項目"
        assert "トラックを追加" not in _texts(menu)
        action.trigger()
        added = view.project.timeline.tracks[-1]
        assert added.kind is TrackKind.MIXED

    def test_only_an_empty_track_can_be_removed(self, view: TimelineView) -> None:
        # クリップごと消せると、見えていない所のクリップまで黙って消える
        _wire(view)
        busy = view.build_context_menu(_point(view, "V1", 100))
        assert not _find(busy, "トラックを削除（V1）").isEnabled()
        empty = view.build_context_menu(_point(view, "V2", 100))
        removal = _find(empty, "トラックを削除（V2）")
        assert removal.isEnabled()
        removal.trigger()
        assert [t.name for t in view.project.timeline.tracks] == ["V1", "A1"]

    def test_below_the_last_track_still_offers_adding(self, view: TimelineView) -> None:
        # トラックの無い所で何も出ないと、全部消した後に足す入口が右クリックに無い
        rect = view.track_add_button()
        assert rect is not None
        menu = view.build_context_menu(QPoint(400, rect.bottom() + 10))
        assert "トラックを追加" in _texts(menu)


class TestNextToTheWorkArea:
    def test_clearing_the_range_comes_before_the_track_items(self, view: TimelineView) -> None:
        # トラックを足す・消すは最後のまとまり 範囲の項目が後ろに来ると、トラックの項目の
        # 間に時間の操作が挟まって見える
        view.set_project(
            view.project.with_timeline(replace(view.project.timeline, work_area=(100, 300)))
        )
        texts = _texts(view.build_context_menu(_point(view, "V2", 200)))
        assert texts.index("追加") < texts.index("範囲を解除") < texts.index("トラックを追加")
        assert texts[-1] == "トラックを削除（V2）"

    def test_the_ruler_offers_no_track_items(self, view: TimelineView) -> None:
        # 目盛りはトラックの欄ではない ここでトラックを消す項目が出ると、何を消すのか分からない
        texts = _texts(view.build_context_menu(QPoint(400, 5)))
        assert "トラックを追加" not in texts
        assert "追加" not in texts


class TestAddingFromTheEmptySpace:
    def test_the_add_menu_lists_every_kind(self, view: TimelineView) -> None:
        # 欠けた種類は右クリックから置けず、メニューバーから再生ヘッドの位置へ置き直すことになる
        menu = view.build_context_menu(_point(view, "V2", 200))
        texts = _texts(_menu(menu, "追加"))
        for expected in (
            "テキスト",
            "図形",
            "場面切り替え",
            "フィルタ",
            "カスタムオブジェクト",
            "エイリアス",
            "シーン",
        ):
            assert expected in texts
        # 既にあった項目は残す
        assert "貼り付け（再生ヘッドの位置）" in _texts(menu)

    def test_text_lands_where_it_was_asked_and_is_selected(self, view: TimelineView) -> None:
        # 再生ヘッドの位置へ置くと、置きたい所まで再生ヘッドを動かし直すことになる
        _wire(view)
        view.set_playhead(0)
        _find(view.build_context_menu(_point(view, "V2", 200)), "追加", "テキスト").trigger()
        v2 = _track(view, "V2")
        assert len(v2.clips) == 1
        assert v2.clips[0].timeline_start == pytest.approx(200, abs=1)
        assert view.selected_clips == (v2.clips[0].id,)

    def test_the_add_menu_is_not_offered_on_a_clip(self, view: TimelineView) -> None:
        # クリップの上は「掛ける」 置く物を出すと、そのクリップのある所へ置けずに別へ行く
        assert "追加" not in _texts(view.build_context_menu(_point(view, "V1", 10)))

    def test_a_scene_other_than_the_open_one_can_be_placed(self, view: TimelineView) -> None:
        # 開いているシーンを出すと、選んだとたんに入れ子が自分へ戻って断られる
        _wire(view)
        opening = Scene(name="オープニング", timeline=Timeline(rate=view.project.rate))
        ending = Scene(name="エンディング", timeline=Timeline(rate=view.project.rate))
        view.set_project(replace(view.project, scenes=(opening, ending)))
        view.set_open_scene(opening.id)
        menu = view.build_context_menu(_point(view, "V2", 90))
        assert _texts(_menu(menu, "追加", "シーン")) == ["エンディング"]
        _find(menu, "追加", "シーン", "エンディング").trigger()
        placed = _track(view, "V2").clips
        assert [c.scene_id for c in placed] == [ending.id]

    def test_a_custom_object_is_a_script_on_an_empty_object(
        self, view: TimelineView, tmp_path: Path
    ) -> None:
        # 読み込めたカスタムオブジェクトが並ばないと、配布スクリプトを置く入口が無い
        _wire(view)
        catalog = ScriptCatalog(roots=())
        entry = catalog.add_text(
            "aviutl:試験.obj:円", "@円\n--track0:大きさ,0,100,50\n", kind="obj"
        )
        view.add_sources = _sources(tmp_path, custom_objects=lambda: (entry,))
        _find(
            view.build_context_menu(_point(view, "V2", 120)), "追加", "カスタムオブジェクト", "円"
        ).trigger()
        (clip,) = _track(view, "V2").clips
        # スクリプトの後ろに描画の欄（反転・配置）が付く（#27 P2）
        assert [e.kind for e in clip.effects] == [entry.identifier, "flip", "transform"]
        assert clip.source is not None and clip.source.params.get("text") == ""

    def test_a_custom_object_is_called_by_its_script(
        self, view: TimelineView, tmp_path: Path
    ) -> None:
        """置いたカスタムオブジェクトは、タイムラインでも設定パネルでもスクリプトの名前で出る（#147）

        土台が空のテキストなので、種類の名前のまま出すと「テキスト」になり、何を置いたのか
        分からない エイリアスとして保存するときの名前の既定も同じ
        """
        from sashimono.ui.inspector.header import identify_clip
        from sashimono.ui.timeline.add_menu import _suggested_name
        from sashimono.ui.timeline.painter import _clip_name

        _wire(view)
        catalog = ScriptCatalog(roots=())
        entry = catalog.add_text(
            "aviutl:試験.obj:円", "@円\n--track0:大きさ,0,100,50\n", kind="obj"
        )
        view.add_sources = _sources(tmp_path, custom_objects=lambda: (entry,))
        _find(
            view.build_context_menu(_point(view, "V2", 120)), "追加", "カスタムオブジェクト", "円"
        ).trigger()
        (clip,) = _track(view, "V2").clips
        assert _clip_name(clip, None) == "カスタムオブジェクト: 円"
        identity = identify_clip(view.project, clip.id)
        assert identity is not None and identity.title == "カスタムオブジェクト（円）"
        assert _suggested_name(clip) == "カスタムオブジェクト 円"

    def test_the_script_catalog_is_not_read_until_the_submenu_opens(
        self, view: TimelineView, tmp_path: Path
    ) -> None:
        # 右クリックのたびに何百本ものスクリプトを読むと、メニューが出るまで待たされる
        asked: list[int] = []

        def entries() -> tuple[ScriptEntry, ...]:
            asked.append(1)
            return ()

        view.add_sources = _sources(tmp_path, custom_objects=entries)
        menu = view.build_context_menu(_point(view, "V2", 120))
        assert asked == []
        _menu(menu, "追加", "カスタムオブジェクト")
        assert asked == [1]

    def test_a_shelf_template_is_handed_to_the_window(
        self, view: TimelineView, tmp_path: Path
    ) -> None:
        # 素材の読み込みと登録は窓が持っている ビューが自分で置くと素材が結ばれない
        entry = TemplateEntry(name="強調", path=tmp_path / "強調.object", folder="字幕")
        view.add_sources = _sources(tmp_path, templates=lambda: (entry,))
        asked: list[tuple[object, int, str]] = []
        view.template_requested.connect(lambda e, f, t: asked.append((e, f, t)))
        point = _point(view, "V2", 150)
        _find(view.build_context_menu(point), "追加", "エイリアス", "字幕", "強調").trigger()
        v2 = _track(view, "V2")
        assert asked == [(entry, view._layout.frame_at(point.x()), v2.id)]

    def test_an_effect_track_offers_the_filter_first(self, view: TimelineView) -> None:
        # フィルタを置くために足したトラックで、毎回サブメニューを探すのは遠い
        _wire(view)
        _find(view.build_track_add_menu(), "エフェクトトラック（フィルタ用）").trigger()
        menu = view.build_context_menu(_point(view, "FX1", 30))
        assert _texts(menu)[0] == "フィルタを置く"
        _find(menu, "フィルタを置く").trigger()
        (clip,) = _track(view, "FX1").clips
        assert clip.is_filter

    def test_an_old_fx_layer_is_an_ordinary_layer(self, view: TimelineView) -> None:
        # 利用者の要望 分けない方式ではエフェクトのレイヤーを持たない 前の版で作った
        # FX のレイヤーだけ右クリックの頭が変わると、同じレイヤーなのに扱いが違って見える
        settings = replace(view.project.settings, layer_mode=LayerMode.MIXED)
        tracks = (Track(TrackKind.MIXED, "レイヤー 1"), Track(TrackKind.MIXED, "FX1"))
        view.set_project(
            replace(
                view.project,
                settings=settings,
                timeline=replace(view.project.timeline, tracks=tracks),
            )
        )
        _wire(view)
        menu = view.build_context_menu(_point(view, "FX1", 30))
        assert "フィルタを置く" not in _texts(menu)
        # フィルタは〔追加〕から、ほかのレイヤーと同じく置ける
        _find(view.build_context_menu(_point(view, "FX1", 30)), "追加", "フィルタ").trigger()
        (clip,) = _track(view, "FX1").clips
        assert clip.is_filter
        assert [t.name for t in view.project.timeline.tracks] == ["レイヤー 1", "FX1"]


class TestEffectsOnAClip:
    def test_only_effects_of_the_clip_kind_are_offered(self, view: TimelineView) -> None:
        # 映像のクリップへ音のエフェクトを積んでも何も起きない
        menu = view.build_context_menu(_point(view, "V1", 10))
        effects = _menu(menu, "エフェクトを追加")
        offered = {a.text() for sub in _submenus(effects) for a in sub.actions()}
        audio = {d.label for d in registry.all() if d.audio_process is not None}
        video = {d.label for d in registry.all() if d.audio_process is None}
        assert offered & video
        assert not offered & (audio - video)

    def test_an_effect_goes_to_every_selected_clip_at_once(self, view: TimelineView) -> None:
        # 1 本ずつ別の取り消しになると、戻すのに本数ぶん押すことになる
        second = Clip(timeline_start=100, duration=30, source=TEXT.create())
        v1 = _track(view, "V1")
        view.set_project(
            view.project.with_timeline(
                view.project.timeline.replace_track(v1.with_clips((*v1.clips, second)))
            )
        )
        received: list[list[Command]] = []
        view.commands_requested.connect(lambda commands, _label: received.append(commands))
        first = _track(view, "V1").clips[0]
        view.set_selection((first.id, second.id))
        blur = registry.get("blur")
        assert blur is not None
        menu = view.build_context_menu(_point(view, "V1", 10))
        category = blur.category
        _find(menu, "エフェクトを追加", category, blur.label).trigger()
        (commands,) = received
        assert all(isinstance(c, AddEffect) for c in commands)
        assert {c.clip_id for c in commands if isinstance(c, AddEffect)} == {first.id, second.id}


class TestEffectsOnALayerClip:
    """レイヤー（混合）の 1 本のクリップは、描く・鳴らすに合わせてエフェクトを選ぶ"""

    def _layered(
        self, view: TimelineView, video_media: MediaItem, audio_media: MediaItem
    ) -> tuple[Clip, Clip]:
        movie = Clip(0, 60, media_id=video_media.id, audio_stream=1)
        bgm = Clip(0, 60, media_id=audio_media.id, audio_stream=0, show_picture=False)
        base = Project.create(media=(video_media, audio_media))
        tracks = (
            Track(TrackKind.MIXED, "レイヤー 1", (movie,)),
            Track(TrackKind.MIXED, "レイヤー 2", (bgm,)),
        )
        view.set_project(base.with_timeline(replace(base.timeline, tracks=tracks)))
        return movie, bgm

    def _offered(self, view: TimelineView, track: str) -> set[str]:
        # 画面の並べ方がまだレイヤーを出さない（P4b）ので、クリップの右クリックの中身を
        # 作る所を直に呼ぶ
        layer = next(t for t in view.project.timeline.tracks if t.name == track)
        menu = QMenu()
        view._add_menus.add_clip_items(menu, layer, layer.clips[0])
        effects = _menu(menu, "エフェクトを追加")
        return {a.text() for sub in _submenus(effects) for a in sub.actions()}

    def test_the_offer_follows_what_the_clip_plays(
        self, view: TimelineView, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 種類だけで見ると、レイヤーの BGM に音のエフェクトが出ず、効かない絵のエフェクトが並ぶ
        self._layered(view, video_media, audio_media)
        audio = {d.label for d in registry.all() if d.audio_process is not None}
        video = {d.label for d in registry.all() if d.audio_process is None}
        movie = self._offered(view, "レイヤー 1")
        bgm = self._offered(view, "レイヤー 2")
        assert movie & audio and movie & video
        assert bgm & audio
        assert not bgm & (video - audio)

    def test_a_sound_effect_reaches_layer_clips(
        self, view: TimelineView, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 音声トラックのクリップにしか掛けないと、レイヤーでは音のエフェクトが何も起きない
        movie, bgm = self._layered(view, video_media, audio_media)
        received: list[list[Command]] = []
        view.commands_requested.connect(lambda commands, _label: received.append(commands))
        view.set_selection((movie.id, bgm.id))
        sound = next(d for d in registry.all() if d.audio_process is not None)
        view._add_menus.add_effect(sound.kind)
        (commands,) = received
        assert {c.clip_id for c in commands if isinstance(c, AddEffect)} == {movie.id, bgm.id}


class TestAliases:
    def test_a_saved_text_comes_back_from_the_add_menu(self, view: TimelineView) -> None:
        # 保存したのに一覧に出ないと、同じテロップを毎回作り直すことになる
        _wire(view)
        _find(view.build_context_menu(_point(view, "V1", 10)), "エイリアスとして保存…").trigger()
        menu = view.build_context_menu(_point(view, "V2", 300))
        saved = _texts(_menu(menu, "追加", "エイリアス", "保存したもの"))
        assert saved == ["テキスト 見出し"]
        _find(menu, "追加", "エイリアス", "保存したもの", "テキスト 見出し").trigger()
        (placed,) = _track(view, "V2").clips
        original = _track(view, "V1").clips[0]
        assert placed.source == original.source
        assert placed.duration == original.duration
        assert placed.id != original.id
        assert view.selected_clips == (placed.id,)

    def test_a_clip_with_media_cannot_be_saved(
        self, view: TimelineView, video_media: MediaItem
    ) -> None:
        # 素材の道は本人の機械にしか無い 別のプロジェクトで置くと何も映らない
        movie = Clip(timeline_start=0, duration=30, media_id=video_media.id)
        v2 = _track(view, "V2")
        project = view.project.with_media((video_media,))
        view.set_project(
            project.with_timeline(project.timeline.replace_track(v2.with_clips((movie,))))
        )
        menu = view.build_context_menu(_point(view, "V2", 10))
        assert not _find(menu, "エイリアスとして保存…").isEnabled()

    def test_saving_can_be_cancelled(self, view: TimelineView, tmp_path: Path) -> None:
        # 保存の窓でやめても保存されると、要らないエイリアスが一覧に増える
        view.add_sources = _sources(tmp_path, ask_save=lambda *_a, **_k: None)
        _find(view.build_context_menu(_point(view, "V1", 10)), "エイリアスとして保存…").trigger()
        assert view.add_sources.aliases.all() == ()

    def test_the_same_name_is_not_overwritten_without_asking(
        self, view: TimelineView, tmp_path: Path
    ) -> None:
        # 既定の名前はテキストの頭から作る 同じ文言の別の見た目を保存すると、黙って前の物が消える
        # 上書きは保存の窓で指定する 指定せずに同じ名前で保存しても上書きしない
        messages: list[str] = []
        view.status_message.connect(messages.append)
        view.add_sources = _sources(tmp_path)
        save = ("エイリアスとして保存…",)
        _find(view.build_context_menu(_point(view, "V1", 10)), *save).trigger()
        first = view.add_sources.aliases.all()
        v1 = _track(view, "V1")
        blurred = replace(v1.clips[0], blend_mode="add")
        view.set_project(
            view.project.with_timeline(
                view.project.timeline.replace_track(v1.with_clips((blurred,)))
            )
        )
        _find(view.build_context_menu(_point(view, "V1", 10)), *save).trigger()
        assert view.add_sources.aliases.all() == first
        assert any("もうある" in message for message in messages)


class TestInTheWindow:
    @pytest.fixture
    def window(self, qt_application: QApplication, tmp_path: Path) -> Iterator[MainWindow]:
        del qt_application
        created = MainWindow(_project(), confirm_unsaved=False)
        created._timeline.add_sources = _sources(tmp_path)
        created._timeline.resize(900, 400)
        yield created
        created.close()

    def test_an_added_track_can_be_undone(self, window: MainWindow) -> None:
        # 取り消せないと、押し間違えて足したトラックを右クリックから 1 本ずつ消すことになる
        _find(window._timeline.build_track_add_menu(), "映像トラック").trigger()
        assert len(window.document.project.timeline.tracks) == 4
        window.undo()
        assert len(window.document.project.timeline.tracks) == 3

    def test_a_shelf_alias_lands_on_the_clicked_track(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        # 棚から置いた物が元のレイヤーの番号へ行くと、右クリックした所と違うトラックに入る
        path = tmp_path / "図形.exa"
        path.write_text(
            "[vo]\nlength=30\n[vo.0]\n_name=図形\nサイズ=100\n[vo.1]\n_name=標準描画\n", "utf-8"
        )
        entry = TemplateEntry(name="図形", path=path)
        v2 = TrackId(_track(window._timeline, "V2").id)
        assert window.place_template_entry(entry, 200, v2)
        (placed,) = _track(window._timeline, "V2").clips
        assert placed.timeline_start == 200
        assert window._timeline.selected_clips == (placed.id,)

    def test_the_open_scene_reaches_the_timeline(self, window: MainWindow) -> None:
        # 届かないと〔追加〕→〔シーン〕に開いているシーン自身が並び、選ぶと入れ子が
        # 自分へ戻って断られる
        scene_id = window.create_scene("中")
        assert window._timeline.open_scene == scene_id


def _menu(menu: QMenu, *path: str) -> QMenu:
    submenu = _find(menu, *path).menu()
    assert isinstance(submenu, QMenu)
    submenu.aboutToShow.emit()
    return submenu


def _submenus(menu: QMenu) -> list[QMenu]:
    found: list[QMenu] = []
    for action in menu.actions():
        submenu = action.menu()
        if isinstance(submenu, QMenu):
            found.append(submenu)
    return found
