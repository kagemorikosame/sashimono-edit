"""設定パネルを YMM4 のアイテムの並びで見せる（#27 YMM4 型アイテムの設計 P2）

描画（X・Y・不透明度・拡大率・回転角・合成モード・左右反転・クリッピング）→ 中身 →
動画・音声（音量・パン・再生速度・再生開始位置・フェード）→ 足したエフェクト
最初から持つ欄は組の中に出し、足したエフェクトだけを下の一覧に出す
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction

import pytest
import shiboken6
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QToolButton

from sashimono.core.commands import (
    AddEffect,
    Command,
    SetClipProperty,
    SetParam,
    insert_generated,
    insert_media,
)
from sashimono.core.model import AnimatedValue, Clip, MediaItem, Project, TrackKind
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.ui.inspector.panel import InspectorPanel, _Section
from sashimono.ui.inspector.widgets import CheckEditor, TrackEditor


@pytest.fixture
def panel(qt_application: QApplication) -> Iterator[InspectorPanel]:
    del qt_application
    created = InspectorPanel()
    yield created
    created.close()
    shiboken6.delete(created)


def _placed(media: MediaItem) -> Project:
    project = Project.create()
    for command in insert_media(project, media):
        project = command.apply(project)
    return project


def _placed_again(project: Project, media: MediaItem) -> Project:
    """同じ素材をもう 1 本、後ろへ置く"""
    for command in insert_media(project, media):
        project = command.apply(project)
    return project


def _clip(project: Project, kind: TrackKind) -> Clip:
    return next(c for t in project.timeline.tracks if t.kind is kind for c in t.clips)


def _headings(panel: InspectorPanel) -> list[str]:
    return [section.heading for section in panel._body.findChildren(_Section)]


def _rows(section: _Section) -> list[str]:
    """組の行の見出し 見出しの列（0 列目）に並ぶ言葉を上から"""
    grid = section._grid
    names = []
    for row in range(1, grid.rowCount()):
        item = grid.itemAtPosition(row, 0)
        widget = item.widget() if item is not None else None
        if isinstance(widget, QLabel):
            names.append(widget.text())
    return names


def _group(panel: InspectorPanel, heading: str) -> _Section:
    return next(s for s in panel._body.findChildren(_Section) if s.heading == heading)


def _requests(panel: InspectorPanel) -> list[tuple[list[Command], str]]:
    sent: list[tuple[list[Command], str]] = []
    panel.commands_requested.connect(lambda commands, label: sent.append((commands, label)))
    return sent


class TestOrder:
    def test_a_video_clip_shows_picture_then_movie_then_effects(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # YMM4 の動画アイテムと同じ並び 足したエフェクトは組の後ろの一覧に出る
        project = _placed(video_media)
        picture = _clip(project, TrackKind.VIDEO)
        project = AddEffect(picture.id, registry.require("blur").create()).apply(project)
        panel.set_project(project)
        panel.set_clip(picture.id)
        assert _headings(panel) == ["描画", "動画", "ぼかし"]
        assert _rows(_group(panel, "描画")) == [
            "X",
            "Y",
            "不透明度",
            "拡大率",
            "回転角",
            "揃える",
            "合成モード",
            "左右反転",
            "クリッピング",
            "画素で置く",
        ]
        assert _rows(_group(panel, "動画")) == ["再生速度", "再生開始位置"]
        headings = [label.text() for label in panel._body.findChildren(QLabel, "effects_heading")]
        assert headings == ["映像エフェクト"]

    def test_a_sound_clip_shows_no_picture_group(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 音声トラックのクリップに合成モードや不透明度を出すと、動かしても何も変わらない
        # 欄を触らせることになる（前の版では出ていた）
        project = _placed(video_media)
        sound = _clip(project, TrackKind.AUDIO)
        panel.set_project(project)
        panel.set_clip(sound.id)
        assert _headings(panel) == ["音声"]
        assert _rows(_group(panel, "音声")) == [
            "音量",
            "パン",
            "再生速度",
            "再生開始位置",
            "フェードイン",
            "フェードアウト",
        ]
        assert not panel._body.findChildren(QComboBox)
        headings = [label.text() for label in panel._body.findChildren(QLabel, "effects_heading")]
        assert headings == ["音声エフェクト"]

    def test_a_picture_clip_shows_no_sound_group(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 映像トラックのクリップは鳴らない 音量を出すと、動かしても音が変わらない
        project = _placed(video_media)
        panel.set_project(project)
        panel.set_clip(_clip(project, TrackKind.VIDEO).id)
        assert "音声" not in _headings(panel)
        assert "音量" not in _rows(_group(panel, "描画"))

    def test_text_shows_its_content_after_the_picture_group(self, panel: InspectorPanel) -> None:
        project = Project.create()
        for command in insert_generated(project, TEXT.create()):
            project = command.apply(project)
        panel.set_project(project)
        panel.set_clip(_clip(project, TrackKind.VIDEO).id)
        # 素材の画素で置く欄はテキストには出ない（画面の大きさで作るので効かない）
        assert _headings(panel)[:2] == ["描画", TEXT.label]
        assert "画素で置く" not in _rows(_group(panel, "描画"))

    def test_the_fixed_items_are_not_listed_again(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 組と一覧の両方に出すと、同じ値の欄が 2 か所に並び、どちらを触ればよいか分からない
        project = _placed(video_media)
        panel.set_project(project)
        panel.set_clip(_clip(project, TrackKind.VIDEO).id)
        assert "変形" not in _headings(panel)
        assert "反転" not in _headings(panel)


class TestMixedTrack:
    def test_a_movie_on_a_layer_shows_both_picture_and_sound(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 混合トラックの動画は 1 本で絵と音を持つ（#27 P3） トラックの種類で決めると、
        # 音量の欄が出ないか、描画の組が消える
        from sashimono.core.commands import AddClip, AddMedia, AddTrack
        from sashimono.core.commands.fixed import with_fixed_items
        from sashimono.core.model import Track

        track = Track(kind=TrackKind.MIXED, name="レイヤー 1")
        clip = with_fixed_items(
            Clip(
                timeline_start=0,
                duration=30,
                media_id=video_media.id,
                stream_index=video_media.video_streams[0].index,
                audio_stream=video_media.audio_streams[0].index,
            ),
            picture=True,
            sound=True,
        )
        project = Project.create()
        for command in (AddMedia(video_media), AddTrack(track), AddClip(track.id, clip)):
            project = command.apply(project)
        panel.set_project(project)
        panel.set_clip(clip.id)
        assert _headings(panel) == ["描画", "動画"]
        assert _rows(_group(panel, "動画")) == [
            "音量",
            "パン",
            "再生速度",
            "再生開始位置",
            "フェードイン",
            "フェードアウト",
        ]


class TestOldFiles:
    def _old(self, project: Project, clip: Clip) -> Project:
        """前の版のファイルと同じく、欄を持たないクリップにする"""
        located = project.timeline.locate_clip(clip.id)
        assert located is not None
        track, _ = located
        bare = replace(clip, effects=())
        return project.with_timeline(
            project.timeline.replace_track(
                track.with_clips(tuple(bare if c.id == clip.id else c for c in track.clips))
            )
        )

    def test_touching_a_missing_item_adds_it_in_one_step(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 開いただけで足すと、見ただけのクリップに変更が入る 触ったときに足し、値を入れるのと
        # 同じ 1 段にまとめる 分けると、1 回戻しただけでは既定の値の欄が残る
        project = _placed(video_media)
        picture = _clip(project, TrackKind.VIDEO)
        project = self._old(project, picture)
        panel.set_project(project)
        panel.set_clip(picture.id)
        assert _rows(_group(panel, "描画"))[:2] == ["X", "Y"]
        sent = _requests(panel)

        x = _group(panel, "描画").findChildren(TrackEditor)[0]
        x.value_changed.emit(AnimatedValue(120.0))

        ((commands, _),) = sent
        add, change = commands
        assert isinstance(add, AddEffect) and add.effect.fixed
        assert add.effect.kind == "transform"
        assert isinstance(change, SetParam) and change.path.effect_id == add.effect.id
        for command in commands:
            project = command.apply(project)
        (placed,) = [e for e in _clip(project, TrackKind.VIDEO).effects if e.fixed]
        assert placed.params["pos_x"] == AnimatedValue(120.0)

    def test_an_old_clip_shows_the_default_values(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        project = _placed(video_media)
        picture = _clip(project, TrackKind.VIDEO)
        panel.set_project(self._old(project, picture))
        panel.set_clip(picture.id)
        scale = _group(panel, "描画").findChildren(TrackEditor)[3]
        assert scale._number.value() == pytest.approx(100.0)


class TestClipFields:
    def test_the_start_position_moves_the_linked_sound_too(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 片方だけ動かすと、絵と音がずれて鳴る
        project = _placed(video_media)
        picture = _clip(project, TrackKind.VIDEO)
        sound = _clip(project, TrackKind.AUDIO)
        panel.set_project(project)
        panel.set_clip(picture.id)
        sent = _requests(panel)
        start = _group(panel, "動画").findChild(TrackEditor, "clip_source_in")
        assert start is not None
        start.value_changed.emit(AnimatedValue(1.5))
        ((commands, _),) = sent
        assert {(c.clip_id, c.value) for c in commands if isinstance(c, SetClipProperty)} == {
            (picture.id, Fraction(3, 2)),
            (sound.id, Fraction(3, 2)),
        }

    def test_the_group_toggle_reaches_every_selected_clip(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 主のクリップだけ切り替わると、一緒に値を変えたほかのクリップと欄の効き方が食い違う

        from sashimono.core.commands import SetEffectEnabled

        project = _placed(video_media)
        project = _placed_again(project, video_media)
        first, second = [
            c for t in project.timeline.tracks if t.kind is TrackKind.VIDEO for c in t.clips
        ]
        panel.set_project(project)
        panel.set_selection((first.id, second.id))
        sent = _requests(panel)
        toggle = _group(panel, "描画").findChild(QToolButton, "fixed_toggle")
        assert toggle is not None
        toggle.setChecked(False)
        ((commands, _),) = sent
        # 切った値まで見る 全部へ「有効」を送る作りでも、相手の顔ぶれだけなら合ってしまう
        switched = {
            (c.clip_id, c.effect_id, c.enabled) for c in commands if isinstance(c, SetEffectEnabled)
        }
        expected = {
            (clip.id, e.id, False) for clip in (first, second) for e in clip.effects if e.fixed
        }
        assert switched == expected

    def test_a_missing_item_on_the_primary_still_reaches_the_others(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 主のクリップが前の版のファイルで欄を持たないと、仮の欄の ID は相手に見つからず、
        # 相手が実在の欄を持っていても値が当たらなかった 相手に無い欄は作らない
        from sashimono.core.commands import SetEffectEnabled

        project = _placed_again(_placed(video_media), video_media)
        first, second = [
            c for t in project.timeline.tracks if t.kind is TrackKind.VIDEO for c in t.clips
        ]
        project = TestOldFiles()._old(project, first)
        panel.set_project(project)
        panel.set_selection((first.id, second.id))
        sent = _requests(panel)

        x = _group(panel, "描画").findChildren(TrackEditor)[0]
        x.value_changed.emit(AnimatedValue(80.0))
        ((commands, _),) = sent
        placed = next(e for e in second.effects if e.fixed and e.kind == "transform")
        assert any(
            isinstance(c, SetParam)
            and c.path.clip_id == second.id
            and c.path.effect_id == placed.id
            for c in commands
        )
        # 足すのは主のクリップの欄だけ
        assert [c.clip_id for c in commands if isinstance(c, AddEffect)] == [first.id]

        sent.clear()
        toggle = _group(panel, "描画").findChild(QToolButton, "fixed_toggle")
        assert toggle is not None
        toggle.setChecked(False)
        ((commands, _),) = sent
        others = {
            (c.effect_id, c.enabled)
            for c in commands
            if isinstance(c, SetEffectEnabled) and c.clip_id == second.id
        }
        assert others == {(e.id, False) for e in second.effects if e.fixed}

    def test_the_native_size_can_be_switched(
        self, panel: InspectorPanel, video_media: MediaItem
    ) -> None:
        # 前の版で置いた物（画面に収めた物）を、素材の画素の大きさへ揃える道
        project = _placed(video_media)
        picture = _clip(project, TrackKind.VIDEO)
        panel.set_project(project)
        panel.set_clip(picture.id)
        sent = _requests(panel)
        check = _group(panel, "描画").findChild(CheckEditor, "clip_native_size")
        assert check is not None
        check.value_changed.emit(False)
        ((commands, _),) = sent
        assert commands == [SetClipProperty(picture.id, "native_size", False)]
