"""設定パネルでの固定の項目の見え方

固定の項目（クリップが最初から持つ欄）は外すことも並べ替えることもできない
押しても断られるボタンを並べると、押して何も起きない理由が分からない
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import shiboken6
from PySide6.QtWidgets import QApplication, QLabel, QToolButton

from sashimono.core.commands import AddEffect, insert_media
from sashimono.core.model import Clip, MediaItem, Project, TrackKind
from sashimono.effects import registry
from sashimono.ui.inspector.panel import InspectorPanel, _Section, lock_pixmap
from sashimono.ui.theme import Colors


def _placed(media: MediaItem) -> Project:
    project = Project.create()
    for command in insert_media(project, media):
        project = command.apply(project)
    return project


def _sound(project: Project) -> Clip:
    (clip,) = [c for t in project.timeline.tracks if t.kind is TrackKind.AUDIO for c in t.clips]
    return clip


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


def _sections(panel: InspectorPanel) -> list[_Section]:
    # 音声のクリップは 音声の組（固定の欄）→ 足したエフェクト の順に並ぶ
    return panel._body.findChildren(_Section)


def _buttons(section: _Section) -> dict[str, QToolButton]:
    return {button.text(): button for button in section.findChildren(QToolButton)}


def test_the_fixed_effect_shows_a_lock_instead_of_remove_and_move(
    panel: InspectorPanel, audio_media: MediaItem
) -> None:
    # 固定のエフェクトに ✕ や並べ替えが出ると、押しても外せず動かせないボタンで迷う
    project = _placed(audio_media)
    clip = _sound(project)
    blur = registry.require("blur").create()
    project = AddEffect(clip.id, blur).apply(project)
    panel.set_project(project)
    panel.set_clip(clip.id)

    fixed, loose = _sections(panel)
    assert fixed.heading == "音声"
    assert fixed.findChild(QLabel, "fixed_lock") is not None
    assert not {"✕", "▲", "▼"} & set(_buttons(fixed))
    # 無効にはできる 切り替えまで消すと、効かせたくないときの逃げ道が無い
    assert _buttons(fixed)["有効"].isEnabled()

    # 鍵は文字ではなく自前で描いた絵 絵文字の書体で出すと、Windows ではカラーの
    # オレンジの鍵になり、ほかのボタンと揃わない（再生ボタンの ⏸ と同じ）
    lock = fixed.findChild(QLabel, "fixed_lock")
    assert lock is not None
    assert lock.text() == ""
    assert not lock.pixmap().isNull()
    assert lock.toolTip()

    # ふつうのエフェクトは外せるが、固定の物をまたいで下へは動かせない
    loose_buttons = _buttons(loose)
    assert loose.findChild(QLabel, "fixed_lock") is None
    assert loose_buttons["✕"].isEnabled()
    assert not loose_buttons["▼"].isEnabled()


def test_the_lock_is_drawn_in_the_button_colour(qt_application: QApplication) -> None:
    # ほかのボタンの文字と同じ色だけで描く 色の付いた画素が混ざると、絵文字と同じく
    # 見出しの中で浮いて見える
    del qt_application
    image = lock_pixmap().toImage()
    colours = {
        (
            image.pixelColor(x, y).red(),
            image.pixelColor(x, y).green(),
            image.pixelColor(x, y).blue(),
        )
        for x in range(image.width())
        for y in range(image.height())
        if image.pixelColor(x, y).alpha() == 255
    }
    assert colours == {(Colors.TEXT.red(), Colors.TEXT.green(), Colors.TEXT.blue())}
