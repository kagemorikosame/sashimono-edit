"""素材を置いた音声のクリップは、最初から音量調整を持つ

置いた直後に設定パネルで音量を動かせないと、下げたいだけでもエフェクトの一覧から
探して足すことになる（Issue #27） 付けるのは素材を置く共通の入口（:func:`insert_media`）
で、画面・AI・貼り付けのどこから置いても同じになる
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import shiboken6
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddClip, insert_generated, insert_media
from sashimono.core.commands.insert import VOLUME_EFFECT_KIND, default_volume_effect
from sashimono.core.model import AnimatedValue, Clip, MediaItem, Project, TrackKind
from sashimono.effects import registry
from sashimono.effects.audio import AudioContext
from sashimono.effects.sources import TEXT
from sashimono.ui.inspector.panel import InspectorPanel


def _placed(media: MediaItem) -> Project:
    project = Project.create()
    for command in insert_media(project, media):
        project = command.apply(project)
    return project


def _clips(project: Project, kind: TrackKind) -> list[Clip]:
    return [c for t in project.timeline.tracks if t.kind is kind for c in t.clips]


class TestPlacedSound:
    def test_the_sound_of_a_movie_has_a_volume(self, video_media: MediaItem) -> None:
        # 付いていないと、置いた動画の音量を下げたいだけでもエフェクトの一覧から探して
        # 足すことになる 映像つきの素材でも、音声の側に付く
        # 後ろのフェードも音声の欄（#27 P2）
        (audio,) = _clips(_placed(video_media), TrackKind.AUDIO)
        assert [effect.kind for effect in audio.effects] == [VOLUME_EFFECT_KIND, "audio_fade"]

    def test_a_sound_file_has_a_volume(self, audio_media: MediaItem) -> None:
        # BGM のような音声だけの素材も同じ 付かないと、置いてすぐ音量を変えられない
        (audio,) = _clips(_placed(audio_media), TrackKind.AUDIO)
        assert [effect.kind for effect in audio.effects] == [VOLUME_EFFECT_KIND, "audio_fade"]

    def test_the_picture_gets_no_volume(self, video_media: MediaItem) -> None:
        # 映像のクリップに音量を付けても鳴らない 設定パネルに効かない項目が並ぶだけ
        # 付くのは描画の欄（反転・配置）だけ
        (video,) = _clips(_placed(video_media), TrackKind.VIDEO)
        assert [effect.kind for effect in video.effects] == ["flip", "transform"]

    def test_each_clip_gets_its_own_effect(self) -> None:
        # 同じ ID のエフェクトを 2 本のクリップが持つと、片方の音量を変えたつもりで
        # もう片方の設定を指すことがある
        first, second = default_volume_effect(), default_volume_effect()
        assert first.id != second.id

    def test_it_matches_the_effect_definition(self) -> None:
        # コア層はエフェクトの定義を読めないので、同じ値を別に書いている 定義の既定と
        # 食い違うと、置いただけで音量が変わる
        definition = registry.get(VOLUME_EFFECT_KIND)
        assert definition is not None
        effect = default_volume_effect()
        assert effect.params == definition.default_params()
        assert definition.normalize(effect.params) == effect.params

    def test_it_leaves_the_sound_alone(self) -> None:
        # 付けただけで音が変わると、今まで置いた素材と聞こえ方が変わる
        definition = registry.get(VOLUME_EFFECT_KIND)
        assert definition is not None and definition.audio_process is not None
        samples = np.random.default_rng(1).uniform(-1, 1, (256, 2)).astype(np.float32)
        values = {
            name: value.static
            for name, value in default_volume_effect().params.items()
            if isinstance(value, AnimatedValue)
        }
        assert set(values) == {"volume", "pan"}
        out = definition.audio_process(samples, values, AudioContext(0, 48000, 256))
        np.testing.assert_array_equal(out, samples)

    def test_generated_objects_get_no_volume(self) -> None:
        # 音声のトラックへ置くもの以外（テキストなど）には付けない 一覧に出る
        # 「音量」は鳴るものにだけ意味がある
        project = Project.create()
        commands = insert_generated(project, TEXT.create())
        added = [c.clip for c in commands if isinstance(c, AddClip)]
        assert added
        assert all(e.kind != VOLUME_EFFECT_KIND for clip in added for e in clip.effects)


class TestInspector:
    @pytest.fixture
    def panel(self, qt_application: QApplication) -> Iterator[InspectorPanel]:
        del qt_application
        created = InspectorPanel()
        yield created
        created.close()
        shiboken6.delete(created)

    def test_the_volume_is_ready_to_change(
        self, panel: InspectorPanel, audio_media: MediaItem
    ) -> None:
        # 置いて選んだだけで、設定パネルに音量の欄が出る 出ないと、音量を変えるのに
        # エフェクトの一覧を開いて音量調整を探し、足してからになる
        project = _placed(audio_media)
        (audio,) = _clips(project, TrackKind.AUDIO)
        panel.set_project(project)
        panel.set_clip(audio.id)
        names = {name for _, name in panel._editors}
        assert {"volume", "pan"} <= names


def _menu_categories(panel: InspectorPanel) -> set[str]:
    menu = panel.effect_menu()
    assert menu is not None
    return {action.text() for action in menu.actions() if action.menu() is not None}


class TestEffectMenu:
    """〔＋ エフェクト〕に並ぶのは、選んだクリップに効く物だけ

    前は音だけのクリップにも映像のエフェクト（ぼかし・色など）が並び、絵だけのクリップにも
    「音」が並んだ 積めても何も起きない
    """

    @pytest.fixture
    def panel(self, qt_application: QApplication) -> Iterator[InspectorPanel]:
        del qt_application
        created = InspectorPanel()
        yield created
        created.close()
        shiboken6.delete(created)

    def test_a_sound_clip_lists_only_sound_effects(
        self, panel: InspectorPanel, audio_media: MediaItem
    ) -> None:
        project = _placed(audio_media)
        (audio,) = _clips(project, TrackKind.AUDIO)
        panel.set_project(project)
        panel.set_clip(audio.id)
        assert _menu_categories(panel) == {"音"}

    def test_a_text_lists_no_sound_effects(self, panel: InspectorPanel) -> None:
        project = Project.create()
        for command in insert_generated(project, TEXT.create()):
            project = command.apply(project)
        (text,) = [c for t in project.timeline.tracks for c in t.clips]
        panel.set_project(project)
        panel.set_clip(text.id)
        categories = _menu_categories(panel)
        assert "音" not in categories
        assert "ぼかし" in categories
