"""プリセットとエイリアスの見本の絵（#277）

見本は保存した中身から置いて描く 前は名前だけのメニューで、当ててから見た目を確かめていた
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.io.aliases import Alias
from sashimono.core.io.presets import Preset
from sashimono.core.model import AnimatedValue, Clip, Effect
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.engine.render.look_preview import (
    SAMPLE_TEXT,
    LookPicture,
    LookRenderer,
    LookWorker,
    crop_to_content,
    sample_project,
)


def _clip(text: str = "見出し", duration: int = 60) -> Clip:
    return with_fixed_items(
        Clip(timeline_start=0, duration=duration, source=TEXT.create(text=text)), picture=True
    )


class TestSampleProject:
    def test_an_effect_only_preset_is_shown_on_sample_text(self) -> None:
        # 文字を持たないプリセット（前の版の物・エフェクトだけの物）は、見本の文字に当てる
        # 何も無い所へ当てると、グローだけのプリセットの見本が空になる
        preset = Preset(name="光", effects=(Effect(kind="glow"),))
        project, frame = sample_project(preset)
        ((clip,),) = [track.clips for track in project.timeline.tracks if track.clips]
        assert clip.source is not None
        assert clip.source.params["text"] == SAMPLE_TEXT
        assert [e.kind for e in clip.effects if not e.fixed] == ["glow"]
        assert frame == clip.duration // 2

    def test_a_preset_shows_its_own_text_and_position(self) -> None:
        # 見本は当て方の設定に関係なく、保存した見た目そのもの（文字も位置も）
        source = TEXT.create(text="保存した文字", pos_x=AnimatedValue(300.0))
        clip = with_fixed_items(Clip(timeline_start=0, duration=90, source=source), picture=True)
        project, frame = sample_project(Preset.capture("見出し", clip))
        drawn = next(track.clips[0] for track in project.timeline.tracks if track.clips)
        assert drawn.source is not None
        assert drawn.source.params["text"] == "保存した文字"
        assert drawn.source.params["pos_x"] == AnimatedValue(300.0)
        assert frame == 45

    def test_an_alias_is_placed_as_it_is(self) -> None:
        alias = Alias.of("テロップ", _clip("置く文字", duration=40))
        project, frame = sample_project(alias)
        drawn = next(track.clips[0] for track in project.timeline.tracks if track.clips)
        assert drawn.source == alias.clip.source
        assert frame == 20


class TestCrop:
    def test_nothing_visible_keeps_the_whole_frame(self) -> None:
        image = np.zeros((1080, 1920, 4), dtype=np.uint8)
        cropped, empty = crop_to_content(image)
        assert empty
        assert cropped.shape == image.shape

    def test_a_small_mark_is_not_blown_up_to_the_whole_tile(self) -> None:
        # 字幕のような小さな物を見本いっぱいに広げると、実際の大きさの感じが分からない
        image = np.zeros((1080, 1920, 4), dtype=np.uint8)
        image[900:920, 950:970] = 255
        cropped, empty = crop_to_content(image)
        assert not empty
        height, width = cropped.shape[:2]
        assert width >= 1920 // 3
        assert width / height == pytest.approx(16 / 9, abs=0.02)
        # 切り出しの中に物が入っている
        assert cropped[:, :, 3].max() == 255

    def test_the_crop_stays_inside_the_frame(self) -> None:
        image = np.zeros((1080, 1920, 4), dtype=np.uint8)
        image[0:10, 1900:1920] = 255
        cropped, _ = crop_to_content(image)
        assert cropped[:, :, 3].max() == 255
        assert cropped.shape[0] <= 1080 and cropped.shape[1] <= 1920


class TestSimple:
    def test_text_is_drawn_without_a_gpu(self) -> None:
        picture = LookRenderer(gpu=False).render(Preset.capture("見出し", _clip()))
        assert picture.simple
        assert not picture.empty
        assert picture.image[:, :, 3].max() > 0


@pytest.mark.usefixtures("gpu")
class TestGpu:
    def test_effects_reach_the_picture(self) -> None:
        # 簡易の描き方ではグローが出ない GPU の見本はグローの分だけ見える所が広がる
        glow = registry.require("glow").create()
        plain = Preset.capture("素", _clip())
        glowing = Preset.capture("光", _clip(), category="ユーザー")
        glowing = Preset(
            name="光",
            source=glowing.source,
            effects=(glow,),
            fixed=glowing.fixed,
            span=glowing.span,
        )
        renderer = LookRenderer(gpu=True)
        try:
            without = renderer.render(plain)
            with_glow = renderer.render(glowing)
        finally:
            renderer.close()
        assert not without.simple and not with_glow.simple
        visible = (without.image[:, :, 3] > 8).sum() / without.image[:, :, 3].size
        visible_glow = (with_glow.image[:, :, 3] > 8).sum() / with_glow.image[:, :, 3].size
        assert visible_glow > visible

    def test_the_backdrop_is_left_transparent(self) -> None:
        # 黒を焼き込むと、明るい地や市松で見たときに黒い四角が出る
        renderer = LookRenderer(gpu=True)
        try:
            picture = renderer.render(Preset.capture("見出し", _clip()))
        finally:
            renderer.close()
        assert picture.image[0, 0, 3] == 0

    def test_the_worker_draws_off_the_screen_thread(self, qt_application: QApplication) -> None:
        worker = LookWorker(gpu=True)
        received: list[tuple[str, object]] = []
        worker.drawn.connect(lambda key, result: received.append((key, result)))
        worker.start()
        try:
            worker.submit("a", Preset.capture("見出し", _clip()))
            deadline = time.monotonic() + 30
            while not received and time.monotonic() < deadline:
                qt_application.processEvents()
                time.sleep(0.01)
        finally:
            worker.stop()
        ((key, result),) = received
        assert key == "a"
        assert isinstance(result, LookPicture)
        assert not result.simple
