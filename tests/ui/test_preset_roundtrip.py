"""設定パネルの〔プリセット…〕で保存して、別のクリップへ当てる往復（#275）

前は保存するのが足したエフェクトの列だけで、テキストの文字の色・大きさ・縁取り、
最初から持つ欄の値、不透明度・合成モード・切り抜き、場面切り替えの後の場面の
エフェクトが落ちた エフェクトを足していないテキストでは保存の項目が押せなかった
どれも画面からの保存の道（メニューの項目 → 名前の窓 → 置き場）を通して確かめる
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu, QMessageBox

from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddMedia,
    Command,
    ParamPath,
    SetClipProperty,
    SetParam,
)
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, with_fixed_items
from sashimono.core.commands.preset import PresetOptions
from sashimono.core.io import Preset, PresetStore
from sashimono.core.io.presets import FORMAT_NAME, SUFFIX
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    Keyframe,
    MediaItem,
    Project,
    Track,
    TrackKind,
)
from sashimono.effects import registry
from sashimono.effects.sources import TEXT, TRANSITION
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.conftest import make_clip

RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


def _text(start: int, duration: int, words: str) -> Clip:
    clip = Clip(timeline_start=start, duration=duration, source=TEXT.create(text=words))
    return with_fixed_items(clip, picture=True)


def _project() -> Project:
    """V1 に保存する元（0〜60）と当てる先（100〜220） V2 にもう 1 本の当てる先（100〜160）"""
    base = Project.create()
    tracks = (
        Track(
            TrackKind.VIDEO,
            "V1",
            (_text(0, 60, "保存したい文字"), _text(100, 120, "残す文字")),
        ),
        Track(TrackKind.VIDEO, "V2", (_text(100, 60, "もう 1 本"),)),
        Track(TrackKind.AUDIO, "A1"),
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _flush() -> None:
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


@pytest.fixture(autouse=True)
def no_modal_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """知らせと問いの小窓を開かせない 開くと試験が止まったまま返らない

    当てて何も変わらなかったときの知らせや、上書きの問いが思わぬ所で出ると、CI は落ちずに
    時間切れまで待つ 出たら試験を落とす 問いに答える試験は、自分で差し替える
    """

    def refuse(*args: object, **_kwargs: object) -> QMessageBox.StandardButton:
        pytest.fail(f"思わぬ小窓が出た: {args[2] if len(args) > 2 else args}")

    for name in ("information", "warning", "question"):
        monkeypatch.setattr(QMessageBox, name, refuse)


@pytest.fixture
def store(tmp_path: Path) -> PresetStore:
    return PresetStore(tmp_path / "presets")


@pytest.fixture
def window(qt_application: QApplication, store: PresetStore) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_project(), confirm_unsaved=False)
    created._inspector.set_preset_store(store)
    yield created
    created.close()


def _clips(window: MainWindow) -> tuple[Clip, Clip, Clip]:
    v1, v2 = window.document.project.timeline.tracks[:2]
    return v1.clips[0], v1.clips[1], v2.clips[0]


def _select(window: MainWindow, *clip_ids: ClipId) -> None:
    window._timeline.set_selection(clip_ids)
    _flush()


def _menu(window: MainWindow) -> QMenu:
    menu = window._inspector.preset_menu()
    assert menu is not None
    return menu


def _actions(menu: QMenu) -> list[QAction]:
    """サブメニューの中まで、項目を並びのとおりに"""
    found: list[QAction] = []
    for action in menu.actions():
        submenu = action.menu()
        if isinstance(submenu, QMenu):
            found.extend(_actions(submenu))
        else:
            found.append(action)
    return found


def _save_action(window: MainWindow) -> QAction:
    return next(a for a in _actions(_menu(window)) if a.data() == "save_preset")


def _save(window: MainWindow, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """メニューの「保存」を押し、名前の窓に ``name`` を打って決める"""
    monkeypatch.setattr(QInputDialog, "getText", lambda *_args, **_kwargs: (name, True))
    action = _save_action(window)
    assert action.isEnabled()
    window._inspector.run_preset_action(action)


def _apply(window: MainWindow, category: str, name: str) -> None:
    """メニューから ``category`` の ``name`` を選ぶ 分類の見出しの下の項目を押す"""
    menu = _menu(window)
    submenu = next(
        a.menu() for a in menu.actions() if isinstance(a.menu(), QMenu) and a.text() == category
    )
    assert isinstance(submenu, QMenu)
    action = next(a for a in submenu.actions() if a.text() == name)
    window._inspector.run_preset_action(action)
    _flush()


def _run(window: MainWindow, *commands: Command) -> None:
    assert window.execute_all(list(commands), "下ごしらえ")
    _flush()


def _fixed(clip: Clip, kind: str) -> Effect:
    return next(e for e in clip.effects if e.fixed and e.kind == kind)


def _dress_up(window: MainWindow) -> None:
    """保存する元のクリップを作り込む 色・大きさ・縁取り・グロー・欄の値・不透明度など"""
    source_clip, _, _ = _clips(window)
    path = ParamPath.of_source
    transform = _fixed(source_clip, TRANSFORM_EFFECT_KIND)
    glow = registry.require("glow").create()
    moving = AnimatedValue(0.0, (Keyframe(0, 0.0), Keyframe(59, 50.0)))
    _run(
        window,
        SetParam(path(source_clip.id, "color"), RED),
        SetParam(path(source_clip.id, "size"), AnimatedValue(120.0)),
        SetParam(path(source_clip.id, "border_width"), AnimatedValue(6.0)),
        SetParam(path(source_clip.id, "pos_x"), AnimatedValue(300.0)),
        AddEffect(source_clip.id, glow),
        SetParam(ParamPath.of_effect(source_clip.id, glow.id, "radius"), moving),
        SetParam(ParamPath.of_effect(source_clip.id, transform.id, "scale"), AnimatedValue(150.0)),
        SetParam(ParamPath.of_effect(source_clip.id, transform.id, "pos_x"), AnimatedValue(-200.0)),
        SetParam(ParamPath.of_clip(source_clip.id, "opacity"), AnimatedValue(0.5)),
        SetClipProperty(source_clip.id, "blend_mode", "add"),
        SetClipProperty(source_clip.id, "clip_to_below", True),
    )


class TestSaveThenApply:
    def test_the_look_reaches_another_clip(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _dress_up(window)
        source_clip, target, _ = _clips(window)
        _select(window, source_clip.id)
        _save(window, monkeypatch, "見出し")

        _select(window, target.id)
        _apply(window, "ユーザー", "見出し")
        _, after, _ = _clips(window)
        assert after.source is not None
        params = after.source.params
        # 前はエフェクトの列だけが入り、文字の見た目が 1 つも当たらなかった
        assert params["color"] == RED
        assert params["size"] == AnimatedValue(120.0)
        assert params["border_width"] == AnimatedValue(6.0)
        # 既定では文字と位置は当てる先のまま（見た目だけを当てる）
        assert params["text"] == "残す文字"
        assert params["pos_x"] == AnimatedValue(0.0)
        # 最初から持つ欄の値も当たる 位置だけは残る 欄は増やさず同じ欄へ写す
        transform = _fixed(after, TRANSFORM_EFFECT_KIND)
        assert transform.params["scale"] == AnimatedValue(150.0)
        assert transform.params["pos_x"] == AnimatedValue(0.0)
        assert sum(1 for e in after.effects if e.kind == TRANSFORM_EFFECT_KIND) == 1
        # 不透明度・合成モード・切り抜きはクリップの欄で、前は入れる場所が無かった
        assert after.opacity == AnimatedValue(0.5)
        assert after.blend_mode == "add"
        assert after.clip_to_below
        # 長さ 60 で作った (0, 59) の動きを、長さ 120 のクリップの両端へ伸ばす
        (glow,) = [e for e in after.effects if e.kind == "glow"]
        radius = glow.params["radius"]
        assert isinstance(radius, AnimatedValue)
        assert [k.frame for k in radius.keyframes] == [0, 119]
        # 保存した元のクリップは変わらない（ID も別の物を振る）
        original, _, _ = _clips(window)
        assert glow.id not in {e.id for e in original.effects}

    def test_one_undo_takes_it_all_back(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _dress_up(window)
        source_clip, target, _ = _clips(window)
        _select(window, source_clip.id)
        _save(window, monkeypatch, "見出し")
        _select(window, target.id)
        _apply(window, "ユーザー", "見出し")

        window.undo()
        _, restored, _ = _clips(window)
        assert restored == target

    def test_a_plain_text_without_effects_can_be_saved(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, store: PresetStore
    ) -> None:
        # 前は足したエフェクトが無いと「保存」が灰色だった 文字の見た目は作り込める
        source_clip, _, _ = _clips(window)
        _run(window, SetParam(ParamPath.of_source(source_clip.id, "color"), RED))
        _select(window, source_clip.id)
        assert _save_action(window).isEnabled()
        _save(window, monkeypatch, "赤い字")
        (saved,) = store.all()
        assert saved.source is not None
        assert saved.source.params["color"] == RED
        assert saved.effects == ()

    def test_the_after_scene_effects_of_a_transition_travel_too(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 場面切り替えの後の場面の列（after_effects）は、前は見もしなかった
        first = Clip(timeline_start=300, duration=30, source=TRANSITION.create())
        second = Clip(timeline_start=400, duration=30, source=TRANSITION.create())
        glow = registry.require("glow").create()
        track = window.document.project.timeline.tracks[1]
        _run(
            window,
            AddClip(track.id, first),
            AddClip(track.id, second),
            AddEffect(first.id, glow, after=True),
        )
        _select(window, first.id)
        _save(window, monkeypatch, "光る切り替え")
        _select(window, second.id)
        _apply(window, "ユーザー", "光る切り替え")

        located = window.document.project.timeline.locate_clip(second.id)
        assert located is not None
        assert [e.kind for e in located[1].after_effects] == ["glow"]
        assert [e.kind for e in located[1].effects] == []


class TestChoosingAndSaving:
    def test_the_same_name_in_another_category_is_not_mixed_up(
        self, window: MainWindow, store: PresetStore
    ) -> None:
        # 前はメニューの項目が名前しか持たず、「ユーザー/見本」を選んでも「あ分類/見本」が当たった
        blur = registry.require("blur").create()
        glow = registry.require("glow").create()
        store.save(Preset(name="見本", effects=(blur,), category="あ分類"))
        store.save(Preset(name="見本", effects=(glow,), category="ユーザー"))
        _, target, _ = _clips(window)
        _select(window, target.id)
        _apply(window, "ユーザー", "見本")
        _, after, _ = _clips(window)
        assert [e.kind for e in after.effects if not e.fixed] == ["glow"]

    def test_saving_over_the_same_name_asks_first(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, store: PresetStore
    ) -> None:
        source_clip, _, _ = _clips(window)
        _select(window, source_clip.id)
        _save(window, monkeypatch, "見出し")
        _run(window, SetParam(ParamPath.of_source(source_clip.id, "color"), BLUE))

        asked: list[str] = []

        def decline(*args: object, **_kwargs: object) -> QMessageBox.StandardButton:
            asked.append(str(args[2]))
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question", decline)
        _save(window, monkeypatch, "見出し")
        # 断ったら前の物が残る 前は確かめずに書き換え、作り直せないプリセットが消えた
        assert asked
        (kept,) = store.all()
        assert kept.source is not None
        assert kept.source.params["color"] != BLUE

        monkeypatch.setattr(
            QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
        )
        _save(window, monkeypatch, "見出し")
        (replaced,) = store.all()
        assert replaced.source is not None
        assert replaced.source.params["color"] == BLUE

    def test_an_old_preset_is_listed_and_applied(
        self, window: MainWindow, store: PresetStore
    ) -> None:
        # 前の版が書いた形（エフェクトの列と空の中身だけ） 項目を足しただけなので読める
        folder = store.root / "ユーザー"
        folder.mkdir(parents=True)
        old = {
            "format": FORMAT_NAME,
            "version": 1,
            "name": "昔のぼかし",
            "category": "ユーザー",
            "effects": [
                {"id": "x", "kind": "blur", "enabled": True, "params": {"radius": {"static": 5}}}
            ],
            "source": None,
        }
        (folder / f"昔のぼかし{SUFFIX}").write_text(json.dumps(old), encoding="utf-8")
        _, target, _ = _clips(window)
        _run(window, AddEffect(target.id, registry.require("glow").create()))
        _select(window, target.id)
        _apply(window, "ユーザー", "昔のぼかし")

        _, after, _ = _clips(window)
        # 前の版の当て方（足すだけ）のまま 先に足してあったグローは消さない
        assert [e.kind for e in after.effects if not e.fixed] == ["glow", "blur"]
        assert after.source == target.source


class TestManyClips:
    def test_every_selected_clip_gets_it_and_one_undo_returns_all(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 前は設定パネルに出ている 1 本だけに当たった
        _dress_up(window)
        source_clip, target, other = _clips(window)
        _select(window, source_clip.id)
        _save(window, monkeypatch, "見出し")

        _select(window, target.id, other.id)
        _apply(window, "ユーザー", "見出し")
        _, first, second = _clips(window)
        for clip in (first, second):
            assert clip.source is not None
            assert clip.source.params["color"] == RED
            assert clip.blend_mode == "add"
        assert "2 本" in str(window.document.undo_label)

        window.undo()
        _, first, second = _clips(window)
        assert (first, second) == (target, other)

    def test_a_sound_clip_in_the_mix_keeps_its_own_effects(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, audio_media: MediaItem
    ) -> None:
        # 映像のプリセット（グロー）は音のクリップへ足せない 前は入れ替えのために音の
        # クリップのエフェクトを先に消し、何も足さないので、音の残響だけが消えた
        _dress_up(window)
        source_clip, target, _ = _clips(window)
        audio_track = window.document.project.timeline.tracks[2]
        sound = make_clip(100, 60, audio_media)
        _run(
            window,
            AddMedia(audio_media),
            AddClip(audio_track.id, sound),
            AddEffect(sound.id, registry.require("audio_reverb").create()),
            AddEffect(target.id, registry.require("blur").create()),
        )
        _select(window, source_clip.id)
        _save(window, monkeypatch, "見出し")

        _select(window, target.id, sound.id)
        _apply(window, "ユーザー", "見出し")
        _, text, _ = _clips(window)
        located = window.document.project.timeline.locate_clip(sound.id)
        assert located is not None
        # 映像のクリップは入れ替わり、音のクリップは自分のエフェクトのまま
        assert [e.kind for e in text.effects if not e.fixed] == ["glow"]
        assert [e.kind for e in located[1].effects if not e.fixed] == ["audio_reverb"]


class TestOptions:
    def test_text_and_position_follow_the_preferences(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _dress_up(window)
        source_clip, target, _ = _clips(window)
        _select(window, source_clip.id)
        _save(window, monkeypatch, "見出し")

        window._inspector.set_preset_options(PresetOptions(with_text=True, with_position=True))
        _select(window, target.id)
        _apply(window, "ユーザー", "見出し")
        _, after, _ = _clips(window)
        assert after.source is not None
        assert after.source.params["text"] == "保存したい文字"
        assert after.source.params["pos_x"] == AnimatedValue(300.0)
        assert _fixed(after, TRANSFORM_EFFECT_KIND).params["pos_x"] == AnimatedValue(-200.0)

    def test_keeping_effects_adds_after_the_ones_already_there(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _dress_up(window)
        source_clip, target, _ = _clips(window)
        _run(window, AddEffect(target.id, registry.require("blur").create()))
        _select(window, source_clip.id)
        _save(window, monkeypatch, "見出し")
        _select(window, target.id)

        # 既定は入れ替える 試しに当て比べても前のエフェクトが重ならない
        _apply(window, "ユーザー", "見出し")
        _, replaced, _ = _clips(window)
        assert [e.kind for e in replaced.effects if not e.fixed] == ["glow"]
        window.undo()

        window._inspector.set_preset_options(PresetOptions(keep_effects=True))
        _apply(window, "ユーザー", "見出し")
        _, kept, _ = _clips(window)
        assert [e.kind for e in kept.effects if not e.fixed] == ["blur", "glow"]

    def test_the_preferences_reach_the_panel_and_survive_a_restart(
        self, window: MainWindow, qt_application: QApplication
    ) -> None:
        del qt_application
        # 既定は見た目だけ 知らない人が当てて文字が消えると困る
        assert Preferences().preset_options == PresetOptions()
        store = PreferenceStore()
        store.save(Preferences(preset_with_text=True, preset_keep_effects=True))
        loaded = store.load()
        assert loaded.preset_options == PresetOptions(with_text=True, keep_effects=True)

        dialog = PreferencesDialog(loaded)
        try:
            assert dialog.preferences().preset_options == loaded.preset_options
        finally:
            dialog.deleteLater()

        window._apply_preferences(loaded)
        assert window._inspector._preset_options == loaded.preset_options
