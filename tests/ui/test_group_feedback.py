"""利用者の確認で出た 4 件（磁石の印・リンクの解除・グループのカット・1 枚の絵の切り替え）

窓は表示しない（オフスクリーン）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
import shiboken6
from PySide6.QtWidgets import QApplication, QCheckBox

from sashimono.core.commands import (
    Command,
    Document,
    GroupClips,
    MoveClip,
    SplitClip,
    UngroupClips,
    insert_media,
)
from sashimono.core.model import (
    GROUP_KIND,
    GROUP_LAYERS,
    Clip,
    GeneratedSource,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    group_as_one,
)
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.resources import BUNDLED_FILES, MAGNET_ICONS, path_to
from sashimono.selfcheck import _bundled_files
from sashimono.ui.inspector.panel import InspectorPanel
from sashimono.ui.scene_bar import SceneBar
from sashimono.ui.timeline import TimelineView


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _clips(project: Project) -> list[Clip]:
    return [clip for track in project.timeline.tracks for clip in track.clips]


class TestMagnetButton:
    def test_the_button_shows_its_state_with_an_icon(self, qt_application: QApplication) -> None:
        # 文字だけでは入と切が見分けにくかった 印を描き分け、補足に状態と一時的に切るキーを書く
        del qt_application
        bar = SceneBar()
        try:
            button = bar.snap_button
            assert button.text() == ""
            assert not button.icon().isNull()
            assert button.isChecked()
            assert "入" in button.toolTip() and "Shift" in button.toolTip()
            on = button.icon().pixmap(18, 18).toImage()
            button.click()
            assert "切" in button.toolTip()
            off = button.icon().pixmap(18, 18).toImage()
            assert on != off, "入と切で同じ印だと、どちらなのか分からない"
        finally:
            bar.deleteLater()

    def test_the_icons_are_bundled_and_checked(self) -> None:
        # 配る版に積み忘れると印が黙って消える 自己診断が揃っているかを見る
        assert set(MAGNET_ICONS) <= set(BUNDLED_FILES)
        assert all(path_to(name).is_file() for name in MAGNET_ICONS)
        assert _bundled_files() == f"{len(BUNDLED_FILES)} 個"


def _placed(video_media: MediaItem) -> Project:
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED), media=(video_media,))
    return _apply(base, insert_media(base, video_media, split_audio=True))


class TestUngroupingLinks:
    def test_ungroup_separates_the_placed_picture_and_sound(self, video_media: MediaItem) -> None:
        # 置いた映像と音はリンクで一緒に動く グループ解除で外れないと、別々に動かせない
        project = _placed(video_media)
        picture, sound = _clips(project)
        assert picture.link_group is not None and picture.link_group == sound.link_group
        freed = UngroupClips((picture.id,)).apply(project)
        assert all(clip.link_group is None for clip in _clips(freed))
        # 外した後は映像だけを別のレイヤーへ動かせ、音は元の所に残る
        third = Track(TrackKind.MIXED, "レイヤー 3")
        freed = freed.with_timeline(replace(freed.timeline, tracks=(*freed.timeline.tracks, third)))
        moved = MoveClip(picture.id, 90, third.id).apply(freed)
        located = moved.timeline.locate_clip(sound.id)
        assert located is not None and located[1].timeline_start == 0
        assert moved.timeline.tracks[2].clips[0].id == picture.id

    def test_the_menu_offers_ungroup_for_a_link(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 映像と音のつながりを解けないと、音だけずらしたり消したりするたびに絵まで一緒に動く
        del qt_application
        project = _placed(video_media)
        picture, _ = _clips(project)
        view = TimelineView(project, MediaAnalyzer(sample_rate=48000, channels=2))
        try:
            received: list[list[Command]] = []
            view.commands_requested.connect(lambda commands, _label: received.append(commands))
            view.select(picture.id)
            assert view.ungroup_selected()
            assert isinstance(received[0][0], UngroupClips)
        finally:
            view.deleteLater()


class TestCuttingGroups:
    def _grouped(self) -> tuple[Project, Clip, Clip]:
        base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
        first = Clip(timeline_start=0, duration=100, source=TEXT.create())
        second = Clip(timeline_start=0, duration=100, source=TEXT.create())
        tracks = (
            Track(TrackKind.MIXED, "レイヤー 1", (first,)),
            Track(TrackKind.MIXED, "レイヤー 2", (second,)),
        )
        project = GroupClips((first.id, second.id)).apply(
            base.with_timeline(replace(base.timeline, tracks=tracks))
        )
        return project, first, second

    def test_the_back_halves_become_their_own_group(self, qt_application: QApplication) -> None:
        # 割った後も全体が 1 つのグループのままだと、前と後ろを別々に選べない（利用者の報告）
        del qt_application
        project, first, second = self._grouped()
        view = TimelineView(project, MediaAnalyzer(sample_rate=48000, channels=2))
        document = Document(project)
        try:

            def run(commands: list[Command], label: str) -> None:
                with document.checkpoint(label):
                    for command in commands:
                        document.execute(command)
                view.set_project(document.project)

            view.commands_requested.connect(run)
            view.set_selection((first.id, second.id))
            view.set_playhead(40, follow=False)
            view.split_at_playhead()
            clips = _clips(document.project)
            front = {c.group_id for c in clips if c.timeline_start == 0}
            back = {c.group_id for c in clips if c.timeline_start == 40}
            assert len(front) == 1 and len(back) == 1
            assert front != back and None not in front | back
            document.undo()
            assert document.project == project
        finally:
            view.deleteLater()

    def test_a_single_split_keeps_the_group(self) -> None:
        # 1 本だけ割ったとき（相手を渡さない）は今までどおり同じグループに残す
        project, first, _ = self._grouped()
        split = SplitClip(first.id, 40).apply(project)
        assert len({c.group_id for c in _clips(split)}) == 1

    def test_linked_halves_split_the_same_way(self, video_media: MediaItem) -> None:
        # リンクも前と後ろで別の組になる（前の片割れどうし・後ろの片割れどうし）
        project = _placed(video_media)
        picture, _ = _clips(project)
        split = SplitClip(picture.id, 60).apply(project)
        front = {c.link_group for c in _clips(split) if c.timeline_start == 0}
        back = {c.link_group for c in _clips(split) if c.timeline_start == 60}
        assert len(front) == 1 and len(back) == 1 and front != back


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


def test_the_panel_has_a_readable_as_one_switch(panel: InspectorPanel) -> None:
    # 前は名前の列に長い名前が切れて四角だけが並び、設定パネルで見つけられなかった
    group = Clip(
        timeline_start=0,
        duration=60,
        source=GeneratedSource(kind=GROUP_KIND, params={GROUP_LAYERS: 1}),
    )
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    project = base.with_timeline(
        replace(base.timeline, tracks=(Track(TrackKind.MIXED, "レイヤー 1", (group,)),))
    )
    document = Document(project)
    labels: list[str] = []

    def run(commands: list[Command], label: str) -> None:
        with document.checkpoint(label):
            for command in commands:
                document.execute(command)
        labels.append(label)
        panel.set_project(document.project)

    panel.commands_requested.connect(run)
    panel.set_project(project)
    panel.set_selection((group.id,))
    boxes = [box for box in panel.findChildren(QCheckBox) if "1 枚の絵として扱う" in box.text()]
    assert len(boxes) == 1
    boxes[0].setChecked(True)
    located = document.project.timeline.locate_clip(group.id)
    assert located is not None and group_as_one(located[1])
    assert len(labels) == 1
    document.undo()
    located = document.project.timeline.locate_clip(group.id)
    assert located is not None and not group_as_one(located[1])
