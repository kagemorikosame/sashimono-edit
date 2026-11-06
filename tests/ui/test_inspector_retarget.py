"""選び替えで設定パネルを作り直さず、同じ構成の欄を今のクリップへ向け直す（#431）

選ぶたびに欄を全部作り直すと、テキストだけのクリップでも 1 回 50 ms、エフェクトが 15 個で
200 ms ほど掛かっていた 構成（エフェクトの種類と並び）が同じなら欄はそのままにして、
欄が覚えている ID（元のクリップの物）を今のクリップの ID と入れ替える

ここでは、向け直した欄が「今のクリップの値を出す」「触ると今のクリップへ当たる」
「前のクリップへは何も当たらない」かを見る 入れ替えの漏れがあると、選んでいない
クリップの値が変わる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QWidget

from sashimono.core.commands import (
    AddEffect,
    Command,
    MoveClip,
    ParamPath,
    SetKeyframe,
    SetParam,
)
from sashimono.core.commands.base import Command as BaseCommand
from sashimono.core.model import AnimatedValue, Clip, ClipId, Effect, Project, Track, TrackKind
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.ui.inspector.alias import IdSwap
from sashimono.ui.inspector.panel import InspectorPanel
from sashimono.ui.inspector.widgets import TextEditor, TrackEditor


def _texts(*effects: tuple[str, ...]) -> Project:
    """テキストクリップを並べる ``effects`` の 1 つずつが 1 本のクリップに積むエフェクトの種類"""
    base = Project.create()
    clips = tuple(
        Clip(
            timeline_start=40 * index,
            duration=30,
            source=TEXT.create(text=f"本文{index}"),
            opacity=AnimatedValue(1.0 - 0.1 * index),
            effects=tuple(registry.require(kind).create() for kind in kinds),
        )
        for index, kinds in enumerate(effects)
    )
    tracks = (Track(TrackKind.VIDEO, "V1", clips),)
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _clips(project: Project) -> list[Clip]:
    return list(project.timeline.tracks[0].clips)


def _flush() -> None:
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


class _Sent:
    """設定パネルが出した命令と途中経過"""

    def __init__(self, panel: InspectorPanel) -> None:
        self.commands: list[Command] = []
        self.previews: list[Command] = []
        self.focused: list[str] = []
        self.paths: list[ParamPath] = []
        panel.commands_requested.connect(lambda commands, _label: self.commands.extend(commands))
        panel.commands_continued.connect(lambda commands, _label: self.commands.extend(commands))
        panel.preview_requested.connect(self.previews.append)
        panel.effect_focused.connect(self.focused.append)
        panel.param_focused.connect(self.paths.append)


def _body(panel: InspectorPanel) -> QWidget:
    return panel._body


def _number(panel: InspectorPanel, owner: str, name: str) -> TrackEditor:
    editor = panel._editors[(owner, name)]
    assert isinstance(editor, TrackEditor)
    return editor


def _shown_number(editor: TrackEditor) -> float:
    box = editor.findChild(QDoubleSpinBox)
    assert box is not None
    return box.value()


def _opacity(panel: InspectorPanel) -> float:
    return _shown_number(_number(panel, "clip", "opacity"))


def _ids(command: Any) -> set[str]:
    """命令の中に出てくるクリップとエフェクトの ID"""
    found: set[str] = set()
    for name in ("clip_id", "effect_id"):
        value = getattr(command, name, None)
        if isinstance(value, str):
            found.add(value)
    path = getattr(command, "path", None)
    if isinstance(path, ParamPath):
        found.add(path.clip_id)
        if path.effect_id is not None:
            found.add(path.effect_id)
    return found


class TestSameLayout:
    def test_the_panel_is_not_rebuilt(self, panel: InspectorPanel) -> None:
        """壊れると、構成の同じテキストを選び替えるたびに欄を全部作り直し、1 回 50 ms 以上
        （エフェクトが多いと 200 ms）かかって選ぶだけで引っ掛かる"""
        project = _texts((), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        body = _body(panel)
        panel.set_clip(second.id)
        assert _body(panel) is body

    def test_the_values_are_the_new_clip(self, panel: InspectorPanel) -> None:
        """壊れると、欄を使い回した先で前のクリップの値（不透明度・本文・見出し）が残る"""
        project = _texts(("blur",), ("blur",))
        first, second = _clips(project)
        second = replace(
            second,
            effects=(second.effects[0].with_param("radius", AnimatedValue(7.0)),),
        )
        project = project.with_timeline(
            project.timeline.replace_track(project.timeline.tracks[0].with_clips((first, second)))
        )
        panel.set_project(project)
        panel.set_clip(first.id)
        panel.set_clip(second.id)
        assert _opacity(panel) == pytest.approx(0.9)
        text = panel._editors[("source", "text")].findChild(QPlainTextEdit)
        assert text is not None and text.toPlainText() == "本文1"
        blur = _number(panel, str(first.effects[0].id), "radius")
        assert _shown_number(blur) == pytest.approx(7.0)

    def test_editing_reaches_only_the_new_clip(self, panel: InspectorPanel) -> None:
        """壊れると、向け直した欄を触ったときに前に選んでいたクリップの値が変わる"""
        project = _texts(("blur",), ("blur",))
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        panel.set_clip(second.id)
        sent = _Sent(panel)
        _number(panel, "clip", "opacity").value_changed.emit(AnimatedValue(0.5))
        _number(panel, str(first.effects[0].id), "radius").value_changed.emit(AnimatedValue(3.0))
        assert sent.commands == [
            SetParam(ParamPath.of_clip(second.id, "opacity"), AnimatedValue(0.5)),
            SetParam(
                ParamPath.of_effect(second.id, second.effects[0].id, "radius"),
                AnimatedValue(3.0),
            ),
        ]
        assert sent.focused == [str(second.effects[0].id)]
        touched = {i for command in sent.commands for i in _ids(command)}
        assert not touched & {first.id, first.effects[0].id}

    def test_a_drag_preview_reaches_the_new_clip(self, panel: InspectorPanel) -> None:
        """壊れると、ドラッグ中のプレビューだけ前のクリップが動いて見える"""
        project = _texts((), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        panel.set_clip(second.id)
        sent = _Sent(panel)
        _number(panel, "clip", "opacity").value_previewed.emit(AnimatedValue(0.4))
        assert sent.previews == [
            SetParam(ParamPath.of_clip(second.id, "opacity"), AnimatedValue(0.4))
        ]

    def test_a_keyframe_reaches_the_new_clip(self, panel: InspectorPanel) -> None:
        """壊れると、◆ を押したときに前のクリップへキーが打たれ、グラフエディタも前の値を開く"""
        project = _texts((), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        panel.set_clip(second.id)
        panel.set_frame(second.timeline_start + 5)
        sent = _Sent(panel)
        panel._key_controls[("clip", "opacity")].toggled.emit()
        path = ParamPath.of_clip(second.id, "opacity")
        assert sent.commands == [SetKeyframe(path, 5, 0.9)]
        assert sent.paths == [path]

    def test_an_edit_of_the_new_clip_refreshes_the_values(self, panel: InspectorPanel) -> None:
        """壊れると、向け直したクリップを取り消しや AI で変えても欄が前の値のまま残る"""
        project = _texts((), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        panel.set_clip(second.id)
        changed = SetParam(ParamPath.of_clip(second.id, "opacity"), AnimatedValue(0.25)).apply(
            project
        )
        panel.set_project(changed)
        assert _opacity(panel) == pytest.approx(0.25)
        # 前のクリップを変えても、出している値は今のクリップのまま
        other = SetParam(ParamPath.of_clip(first.id, "opacity"), AnimatedValue(0.75)).apply(changed)
        panel.set_project(other)
        assert _opacity(panel) == pytest.approx(0.25)

    def test_an_added_effect_rebuilds_for_the_new_clip(self, panel: InspectorPanel) -> None:
        """壊れると、向け直した後にエフェクトを足しても欄が増えないか、前のクリップの欄が出る"""
        project = _texts((), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        panel.set_clip(second.id)
        blur = registry.require("blur").create()
        panel.set_project(AddEffect(second.id, blur).apply(project))
        assert (str(blur.id), "radius") in panel._editors
        sent = _Sent(panel)
        _number(panel, "clip", "opacity").value_changed.emit(AnimatedValue(0.5))
        assert sent.commands == [
            SetParam(ParamPath.of_clip(second.id, "opacity"), AnimatedValue(0.5))
        ]

    def test_the_old_clip_removed_keeps_working(self, panel: InspectorPanel) -> None:
        """壊れると、欄を作ったクリップを消した後に向け直した欄を触ると、何も起きないか落ちる"""
        project = _texts((), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        panel.set_clip(second.id)
        track = project.timeline.tracks[0]
        removed = project.with_timeline(project.timeline.replace_track(track.with_clips((second,))))
        panel.set_project(removed)
        sent = _Sent(panel)
        _number(panel, "clip", "opacity").value_changed.emit(AnimatedValue(0.5))
        assert sent.commands == [
            SetParam(ParamPath.of_clip(second.id, "opacity"), AnimatedValue(0.5))
        ]

    def test_typing_history_does_not_follow(self, panel: InspectorPanel) -> None:
        """壊れると、前のクリップで打った分を欄の中の取り消し（Ctrl+Z）で戻す操作が、
        向け直した先のクリップへの編集になる"""
        project = _texts((), ())
        first, second = _clips(project)
        same = replace(second, source=first.source)
        project = project.with_timeline(
            project.timeline.replace_track(project.timeline.tracks[0].with_clips((first, same)))
        )
        panel.set_project(project)
        panel.set_clip(first.id)
        editor = panel._editors[("source", "text")]
        assert isinstance(editor, TextEditor)
        area = editor.findChild(QPlainTextEdit)
        assert area is not None
        area.insertPlainText("足した")
        assert area.document().isUndoAvailable()
        # 本文が同じでも欄の中の取り消しを残さない（本文が違えば入れ直しで消える）
        panel.set_project(project)
        panel.set_clip(same.id)
        assert not area.document().isUndoAvailable()


class TestDifferentLayout:
    def test_a_different_effect_list_rebuilds(self, panel: InspectorPanel) -> None:
        """壊れると、エフェクトの違うクリップへ選び替えても前のクリップの欄が並んだまま残る"""
        project = _texts(("blur",), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        body = _body(panel)
        panel.set_clip(second.id)
        _flush()
        assert _body(panel) is not body
        assert (str(first.effects[0].id), "radius") not in panel._editors

    def test_a_different_kind_rebuilds(self, panel: InspectorPanel) -> None:
        """壊れると、同じ数でも種類の違うエフェクトの欄へ値を入れようとして、違う欄が並ぶ"""
        project = _texts(("blur",), ("mosaic",))
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        body = _body(panel)
        panel.set_clip(second.id)
        assert _body(panel) is not body
        assert (str(second.effects[0].id), "radius") not in panel._editors


class TestUnrelatedEdits:
    def test_an_edit_elsewhere_does_not_rewrite_the_fields(self, panel: InspectorPanel) -> None:
        """壊れると、別のクリップを動かすだけでも、選んだクリップのエフェクトの数だけ欄を
        入れ直し、エフェクトが 15 個で 1 回 50 ms ほど引っ掛かる"""
        project = _texts((), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        calls: list[None] = []
        original = panel._refresh_values

        def counted() -> None:
            calls.append(None)
            original()

        panel._refresh_values = counted  # type: ignore[method-assign]  # 呼ばれた数を見る
        moved = MoveClip(second.id, 200, project.timeline.tracks[0].id).apply(project)
        panel.set_project(moved)
        assert calls == []
        panel.set_project(
            SetParam(ParamPath.of_clip(first.id, "opacity"), AnimatedValue(0.3)).apply(moved)
        )
        assert calls == [None]


class TestOthersOnly:
    def test_adding_other_clips_keeps_the_fields(self, panel: InspectorPanel) -> None:
        """壊れると、Ctrl で足したり囲んだりするたびに主のクリップの欄を作り直し、1 回
        50 ms 以上かかる"""
        project = _texts((), (), ())
        first, second, third = _clips(project)
        panel.set_project(project)
        panel.set_selection((first.id,))
        body = _body(panel)
        panel.set_selection((first.id, second.id))
        panel.set_selection((first.id, second.id, third.id))
        assert _body(panel) is body
        sent = _Sent(panel)
        _number(panel, "clip", "opacity").value_changed.emit(AnimatedValue(0.5))
        assert {c.path.clip_id for c in sent.commands if isinstance(c, SetParam)} == {
            first.id,
            second.id,
            third.id,
        }


class TestBusy:
    def test_a_selection_while_dragging_waits(self, panel: InspectorPanel) -> None:
        """壊れると、つまみを押している間に選びが変わったとき、掴んだまま別のクリップの
        値を動かすか、欄が消えてドラッグが切れる"""
        project = _texts((), ())
        first, second = _clips(project)
        panel.set_project(project)
        panel.set_clip(first.id)
        editor = _number(panel, "clip", "opacity")
        editor.is_busy = lambda: True  # type: ignore[method-assign]  # 押している最中にする
        panel.set_clip(second.id)
        sent = _Sent(panel)
        editor.value_changed.emit(AnimatedValue(0.5))
        assert sent.commands == [
            SetParam(ParamPath.of_clip(first.id, "opacity"), AnimatedValue(0.5))
        ]
        editor.is_busy = lambda: False  # type: ignore[method-assign]
        panel._after_interaction()
        _flush()
        sent.commands.clear()
        _number(panel, "clip", "opacity").value_changed.emit(AnimatedValue(0.6))
        assert sent.commands == [
            SetParam(ParamPath.of_clip(second.id, "opacity"), AnimatedValue(0.6))
        ]


class TestIdSwap:
    def _pair(self) -> tuple[Clip, Clip]:
        first, second = _clips(_texts(("blur", "mosaic"), ("blur", "mosaic")))
        return first, second

    def test_it_swaps_both_ways(self) -> None:
        """壊れると、欄から出る命令を今の ID へ戻せず、選んでいないクリップへ値が入る"""
        first, second = self._pair()
        swap = IdSwap.between(first, second)
        assert swap is not None
        path = ParamPath.of_effect(first.id, first.effects[1].id, "size")
        moved = swap.apply(SetParam(path, AnimatedValue(2.0)))
        assert moved == SetParam(
            ParamPath.of_effect(second.id, second.effects[1].id, "size"), AnimatedValue(2.0)
        )
        assert swap.apply(moved) == SetParam(path, AnimatedValue(2.0))

    def test_an_added_effect_is_carried_into_the_new_clip(self) -> None:
        """壊れると、まだクリップに無い欄（前の版のファイル）を触ったときに前のクリップへ足す"""
        first, second = self._pair()
        swap = IdSwap.between(first, second)
        assert swap is not None
        effect = Effect("blur")
        assert swap.apply(AddEffect(first.id, effect)) == AddEffect(second.id, effect)

    def test_a_different_count_is_not_paired(self) -> None:
        """壊れると、数の違う並びを何番目どうしで組み、違うエフェクトへ値が入る"""
        first, _ = self._pair()
        (other,) = _clips(_texts(("blur",)))
        assert IdSwap.between(first, other) is None

    def test_the_project_swaps_the_two_clips(self) -> None:
        """壊れると、欄が見るプロジェクトで今のクリップを元の ID で引けず、欄が空になる"""
        project = _texts(("blur",), ("blur",))
        first, second = _clips(project)
        swap = IdSwap.between(first, second)
        assert swap is not None
        swapped = swap.project(project)
        located = swapped.timeline.locate_clip(first.id)
        assert located is not None
        assert located[1].source == second.source
        assert located[1].effects[0].id == first.effects[0].id
        other = swapped.timeline.locate_clip(second.id)
        assert other is not None and other[1].source == first.source

    def test_an_unreadable_command_is_refused(self) -> None:
        """壊れると、中を辿れない命令が元の ID のまま出て、選んでいないクリップが変わる"""
        first, second = self._pair()
        swap = IdSwap.between(first, second)
        assert swap is not None

        class Opaque(BaseCommand):
            @property
            def label(self) -> str:
                return ""

            def apply(self, project: Project) -> Project:
                return project

        with pytest.raises(TypeError):
            swap.apply(Opaque())

    def test_the_same_clip_changes_nothing(self) -> None:
        """壊れると、同じクリップを選び直しただけで命令を作り直して重くなる"""
        first, _ = self._pair()
        swap = IdSwap.between(first, first)
        assert swap is not None and not swap.active
        command = SetParam(ParamPath.of_clip(ClipId(first.id), "opacity"), AnimatedValue(1.0))
        assert swap.apply(command) is command


def test_a_single_line_field_forgets_its_history(qt_application: QApplication) -> None:
    """壊れると、1 行の欄では前のクリップで打った分の取り消しが向け直した先に残る"""
    del qt_application
    from sashimono.effects.spec import TextSpec

    editor = TextEditor(TextSpec("name", "名前", "", multiline=False))
    line = editor.findChild(QLineEdit)
    assert line is not None
    line.insert("足した")
    assert line.isUndoAvailable()
    editor.forget_typing()
    assert not line.isUndoAvailable()
    assert line.text() == "足した"
