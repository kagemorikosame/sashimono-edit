"""GPU 合成とフレームレンダラ

色の扱いを厚めに確認する 符号化されたままの値を混ぜると半透明やフェードが
暗く沈むが、絵としては「なんとなく変」に見えるだけで気づきにくい 数値で押さえる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from sashimono.core.commands import AddClip, AddMedia, AddTrack
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Keyframe,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.decode import probe_media
from sashimono.engine.gpu import (
    Compositor,
    GLContextError,
    OffscreenGLContext,
    Placement,
    Texture,
    fit_placement,
)
from sashimono.engine.render import FrameRenderer, RenderQuality
from tests.media_fixtures import SampleMedia


@pytest.fixture(scope="session")
def gl_context() -> Iterator[OffscreenGLContext]:
    """オフスクリーンの GL コンテキスト 作れない環境ではテストを飛ばす"""
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def solid(width: int, height: int, rgb: tuple[int, int, int], alpha: int = 255) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = rgb
    image[..., 3] = alpha
    return image


def srgb_to_linear(value: float) -> float:
    value /= 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def linear_to_srgb(value: float) -> float:
    encoded = value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
    return encoded * 255.0


class TestPlacement:
    def test_fits_inside_a_wider_frame(self) -> None:
        # 正方形を横長のフレームへ 左右に余白ができる
        assert fit_placement(100, 100, 200, 100) == Placement(50.0, 0.0, 100.0, 100.0)

    def test_fits_inside_a_taller_frame(self) -> None:
        # 縦長の画面で上下に余白を取らないと、正方形の素材が縦に引き伸ばされて歪む
        assert fit_placement(100, 100, 100, 200) == Placement(0.0, 50.0, 100.0, 100.0)

    def test_exact_match_fills_the_frame(self) -> None:
        # 画面と同じ大きさの素材が縮んだりずれたりすると、縁に背景の黒い筋が覗く
        assert fit_placement(1920, 1080, 1920, 1080) == Placement(0.0, 0.0, 1920.0, 1080.0)

    def test_upscales_small_sources(self) -> None:
        # 小さい素材を拡大しないと、640x480 の作品に 320x240 の素材が小さく浮いて映る
        assert fit_placement(320, 240, 640, 480) == Placement(0.0, 0.0, 640.0, 480.0)

    def test_degenerate_source_fills_the_frame(self) -> None:
        # 幅や高さが 0 と読めた素材で 0 で割ると、そのフレームの描画ごと例外で止まる
        assert fit_placement(0, 0, 200, 100) == Placement(0.0, 0.0, 200.0, 100.0)


class TestCompositor:
    def test_opaque_layer_round_trips_exactly(self, gl_context: OffscreenGLContext) -> None:
        # sRGB で入れた値がリニアを経由して同じ値で戻ること ここがずれると
        # 何も加工していない素材の色が変わってしまう
        with gl_context:
            compositor = Compositor(32, 16)
            texture = Texture.from_array(solid(32, 16, (128, 64, 200)))
            compositor.begin()
            compositor.draw(texture)
            result = compositor.read()
            compositor.release()
            texture.release()

        assert result.shape == (16, 32, 4)
        assert result[8, 16, :3].tolist() == [128, 64, 200]

    def test_blending_happens_in_linear_space(self, gl_context: OffscreenGLContext) -> None:
        # sRGB 128 を 50% で黒に重ねる 符号化されたまま混ぜれば 64、
        # リニアで混ぜれば 92 前後 ここが 64 なら色管理が壊れている
        with gl_context:
            compositor = Compositor(16, 16)
            texture = Texture.from_array(solid(16, 16, (128, 128, 128)))
            compositor.begin((0.0, 0.0, 0.0, 1.0))
            compositor.draw(texture, opacity=0.5)
            result = compositor.read()
            compositor.release()
            texture.release()

        expected = linear_to_srgb(srgb_to_linear(128) * 0.5)
        assert result[8, 8, 0] == pytest.approx(expected, abs=2)
        assert result[8, 8, 0] > 80, "符号化されたまま混ざっている"

    def test_reading_back_keeps_the_top_row_on_top(self, gl_context: OffscreenGLContext) -> None:
        # GL は左下から行を返す 上下を返し忘れると、書き出した動画が丸ごと逆さまになる
        # 読み戻しは GPU 側で返しているので、ここが唯一それを押さえる所
        with gl_context:
            compositor = Compositor(8, 8)
            texture = Texture.from_array(solid(8, 8, (255, 0, 0)))
            compositor.begin((0.0, 0.0, 1.0, 1.0))
            # 上半分だけ赤で塗る 画面の Y は下が正なので、上端が 0
            compositor.draw(texture, placement=Placement(0.0, 0.0, 8.0, 4.0))
            result = compositor.read()
            compositor.release()
            texture.release()

        assert result[1, 4, :3].tolist() == [255, 0, 0], "上下が逆さまになっている"
        assert result[6, 4, 0] < 128, "上下が逆さまになっている"

    def test_layers_stack_in_draw_order(self, gl_context: OffscreenGLContext) -> None:
        # 描いた順と重なりが逆になると、上に置いた字幕や画像が下の映像に隠れる
        with gl_context:
            compositor = Compositor(16, 16)
            back = Texture.from_array(solid(16, 16, (255, 0, 0)))
            front = Texture.from_array(solid(16, 16, (0, 255, 0)))
            compositor.begin()
            compositor.draw(back)
            compositor.draw(front)
            result = compositor.read()
            compositor.release()
            back.release()
            front.release()

        # 後に描いた方が手前
        assert result[8, 8, :3].tolist() == [0, 255, 0]

    def test_transparent_layer_leaves_the_background(self, gl_context: OffscreenGLContext) -> None:
        # 透明な画素が色を塗ると、透過 PNG の周りが四角い板になって下の映像を隠す
        with gl_context:
            compositor = Compositor(16, 16)
            back = Texture.from_array(solid(16, 16, (255, 0, 0)))
            front = Texture.from_array(solid(16, 16, (0, 255, 0), alpha=0))
            compositor.begin()
            compositor.draw(back)
            compositor.draw(front)
            result = compositor.read()
            compositor.release()
            back.release()
            front.release()

        assert result[8, 8, :3].tolist() == [255, 0, 0]

    def test_letterboxing_leaves_the_background_visible(
        self, gl_context: OffscreenGLContext
    ) -> None:
        # 正方形の素材を横長のフレームへ 左右に背景が残る
        with gl_context:
            compositor = Compositor(64, 32)
            texture = Texture.from_array(solid(32, 32, (255, 255, 255)))
            compositor.begin((0.0, 0.0, 0.0, 1.0))
            compositor.draw(texture)
            result = compositor.read()
            compositor.release()
            texture.release()

        assert result[16, 32, :3].tolist() == [255, 255, 255], "中央に素材が無い"
        assert result[16, 2, :3].tolist() == [0, 0, 0], "左端に余白が無い"

    def test_orientation_is_preserved(self, gl_context: OffscreenGLContext) -> None:
        # GL は左下原点、画像は左上原点 上下が入れ替わっていないか確かめる
        image = np.zeros((16, 16, 4), dtype=np.uint8)
        image[..., 3] = 255
        image[:4, :, 0] = 255  # 上端だけ赤

        with gl_context:
            compositor = Compositor(16, 16)
            texture = Texture.from_array(image)
            compositor.begin()
            compositor.draw(texture)
            result = compositor.read()
            compositor.release()
            texture.release()

        assert result[1, 8, 0] == 255, "上端が赤くない"
        assert result[14, 8, 0] == 0, "下端が赤い"

    def test_resize(self, gl_context: OffscreenGLContext) -> None:
        # 大きさを作り直せないと、作品の解像度を変えた後のプレビューや書き出しが古い大きさのまま出る
        with gl_context:
            compositor = Compositor(16, 16)
            compositor.resize(32, 8)
            compositor.begin()
            result = compositor.read()
            compositor.release()
        assert result.shape == (8, 32, 4)

    def test_rejects_bad_size(self, gl_context: OffscreenGLContext) -> None:
        # 解像度 0 を通すと、GL の奥で原因の分からない失敗になる 理由の書かれた誤りで止める
        with gl_context, pytest.raises(ValueError, match="解像度"):
            Compositor(0, 16)

    def test_texture_rejects_wrong_dtype(self, gl_context: OffscreenGLContext) -> None:
        # RGBA の 4 つ組でない配列を上げると、画素の区切りがずれて絵が崩れるか、配列の外を読む
        with gl_context, pytest.raises(ValueError, match="RGBA uint8"):
            Texture.from_array(np.zeros((4, 4, 3), dtype=np.uint8))


@pytest.fixture
def rendered_project(sample_av: SampleMedia) -> Project:
    """320x240 の素材を 1 本置いた 320x240 のプロジェクト"""
    media = probe_media(sample_av.path)
    project = Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    project = AddMedia(media).apply(project)
    track = Track(kind=TrackKind.VIDEO, name="V1")
    project = AddTrack(track).apply(project)
    clip = Clip(timeline_start=0, duration=60, media_id=media.id)
    return AddClip(track.id, clip).apply(project)


class TestFrameRenderer:
    def test_renders_at_project_resolution(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 作品の解像度と違う大きさで返すと、書き出した動画が設定と違う大きさになる
        renderer = FrameRenderer(rendered_project, context=gl_context)
        try:
            frame = renderer.render(0)
        finally:
            renderer.close()
        assert frame.shape == (240, 320, 4)
        assert frame.dtype == np.uint8

    def test_empty_timeline_renders_black(self, gl_context: OffscreenGLContext) -> None:
        # 何も置いていない作品が黒でないと、空の所のプレビューや書き出しにごみの色が映る
        project = Project.create(ProjectSettings(width=32, height=32))
        renderer = FrameRenderer(project, context=gl_context)
        try:
            frame = renderer.render(0)
        finally:
            renderer.close()
        assert frame[..., :3].max() == 0

    def test_gap_between_clips_renders_black(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # クリップの外側は何も無い 前のフレームが残ると「消したのに映る」になる
        renderer = FrameRenderer(rendered_project, context=gl_context)
        try:
            renderer.render(0)
            beyond = renderer.render(120)
        finally:
            renderer.close()
        assert beyond[..., :3].max() == 0

    def test_different_frames_differ(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 時刻を進めても同じ絵だと、再生しても書き出しても動画が最初のフレームで止まって見える
        renderer = FrameRenderer(rendered_project, context=gl_context)
        try:
            first = renderer.render(0)
            later = renderer.render(30)
        finally:
            renderer.close()
        assert not np.array_equal(first, later), "時間が進んでも絵が変わっていない"

    def test_opacity_keyframes_are_applied(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 不透明度のキーフレームが効かないと、フェードインを付けたクリップが頭からくっきり映る
        track = rendered_project.timeline.tracks[0]
        faded = replace(
            track.clips[0],
            opacity=AnimatedValue(
                keyframes=(Keyframe(frame=0, value=0.0), Keyframe(frame=30, value=1.0))
            ),
        )
        project = rendered_project.with_timeline(
            rendered_project.timeline.replace_track(track.with_clips((faded,)))
        )

        renderer = FrameRenderer(project, context=gl_context)
        try:
            transparent = renderer.render(0)
            opaque = renderer.render(30)
        finally:
            renderer.close()

        assert transparent[..., :3].max() == 0, "不透明度 0 なのに映っている"
        assert opaque[..., :3].max() > 0

    def test_disabled_clip_is_skipped(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 無効にしたクリップが描かれると、一時的に外したつもりの映像が書き出しに映る
        track = rendered_project.timeline.tracks[0]
        project = rendered_project.with_timeline(
            rendered_project.timeline.replace_track(
                track.with_clips((replace(track.clips[0], enabled=False),))
            )
        )
        renderer = FrameRenderer(project, context=gl_context)
        try:
            frame = renderer.render(0)
        finally:
            renderer.close()
        assert frame[..., :3].max() == 0

    def test_muted_track_is_skipped(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 非表示にしたトラックが描かれると、隠したつもりのトラックの映像が書き出しに映る
        track = rendered_project.timeline.tracks[0]
        project = rendered_project.with_timeline(
            rendered_project.timeline.replace_track(replace(track, muted=True))
        )
        renderer = FrameRenderer(project, context=gl_context)
        try:
            frame = renderer.render(0)
        finally:
            renderer.close()
        assert frame[..., :3].max() == 0

    def test_reduced_quality_shrinks_the_output(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 画質を落としても縮まないと、プレビューの画質を下げても再生が軽くならない
        renderer = FrameRenderer(rendered_project, context=gl_context, quality=RenderQuality(2))
        try:
            frame = renderer.render(0)
        finally:
            renderer.close()
        assert frame.shape == (120, 160, 4)

    def test_quality_can_change_at_runtime(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 途中で画質を変えても大きさが変わらないと、再生中に画質の欄を切り替えても効かない
        renderer = FrameRenderer(rendered_project, context=gl_context)
        try:
            assert renderer.render(0).shape == (240, 320, 4)
            renderer.set_quality(RenderQuality(4))
            assert renderer.render(0).shape == (60, 80, 4)
        finally:
            renderer.close()

    def test_offline_media_renders_black_instead_of_crashing(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 素材が 1 本行方不明でも、プロジェクト全体が開けなくなってはいけない
        original = rendered_project.media[0]
        broken = replace(original, path=original.path.parent / "行方不明.mp4")
        project = rendered_project.replace_media(broken)
        renderer = FrameRenderer(project, context=gl_context)
        try:
            frame = renderer.render(0)
        finally:
            renderer.close()
        assert frame[..., :3].max() == 0

    def test_set_project_closes_removed_media(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 外した素材のデコーダを閉じないと、消した後もファイルを掴み、上書きも削除もできない
        renderer = FrameRenderer(rendered_project, context=gl_context)
        try:
            renderer.render(0)
            emptied = Project.create(rendered_project.settings)
            renderer.set_project(emptied)
            assert renderer.render(0)[..., :3].max() == 0
        finally:
            renderer.close()

    def test_speed_changes_which_source_frame_is_shown(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 2 倍速のクリップは、15 フレーム目でソースの 30 フレーム目を映す
        track = rendered_project.timeline.tracks[0]
        fast = replace(track.clips[0], speed=Fraction(2), duration=30)
        project = rendered_project.with_timeline(
            rendered_project.timeline.replace_track(track.with_clips((fast,)))
        )

        renderer = FrameRenderer(project, context=gl_context)
        try:
            at_15 = renderer.render(15)
        finally:
            renderer.close()

        normal = FrameRenderer(rendered_project, context=gl_context)
        try:
            source_30 = normal.render(30)
        finally:
            normal.close()

        assert np.array_equal(at_15, source_30)

    def test_a_held_clip_shows_the_last_picture_past_the_end_of_its_source(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        """素材より長いクリップは、素材の終わりの後も最後の絵を出す（Issue #115）

        素材は 2 秒（60 フレーム） 止めないと、素材の終わりから後は何も映らない
        YMM4 は素材の最後のフレームを枠の終わりまで出し続ける
        """
        media = rendered_project.media[0]
        track = rendered_project.timeline.tracks[0]
        held = replace(track.clips[0], duration=90, hold_at=media.duration - Fraction(1, 30))
        project = rendered_project.with_timeline(
            rendered_project.timeline.replace_track(track.with_clips((held,)))
        )

        renderer = FrameRenderer(project, context=gl_context)
        try:
            moving = renderer.render(30)
            past = [renderer.render(number) for number in (60, 75, 89)]
        finally:
            renderer.close()

        normal = FrameRenderer(rendered_project, context=gl_context)
        try:
            source_30 = normal.render(30)
            source_last = normal.render(59)
        finally:
            normal.close()

        # 止める前は素のクリップと同じ絵 止める所を早めに効かせると、動くはずの所が止まる
        assert np.array_equal(moving, source_30)
        assert source_last[..., :3].max() > 0
        for frame in past:
            assert np.array_equal(frame, source_last)

    def test_a_clip_held_at_its_start_never_moves(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # YMM4 の再生速度 0 は素材の頭（ContentOffset の位置）の絵のまま
        # 素材の 1 秒目から読むクリップは、どのフレームでも素材の 30 フレーム目を出す
        track = rendered_project.timeline.tracks[0]
        held = replace(track.clips[0], source_in=Fraction(1), hold_at=Fraction(1), duration=60)
        project = rendered_project.with_timeline(
            rendered_project.timeline.replace_track(track.with_clips((held,)))
        )

        renderer = FrameRenderer(project, context=gl_context)
        try:
            produced = [renderer.render(number) for number in (0, 20, 59)]
        finally:
            renderer.close()

        normal = FrameRenderer(rendered_project, context=gl_context)
        try:
            source_30 = normal.render(30)
        finally:
            normal.close()

        for frame in produced:
            assert np.array_equal(frame, source_30)

    def test_closed_renderer_refuses_to_render(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 閉じた後に描けてしまうと、解放した GPU の資源を触ってアプリごと落ちる
        renderer = FrameRenderer(rendered_project, context=gl_context)
        renderer.close()
        with pytest.raises(RuntimeError, match="閉じたレンダラ"):
            renderer.render(0)

    def test_close_is_idempotent(
        self, rendered_project: Project, gl_context: OffscreenGLContext
    ) -> None:
        # 2 度目の後始末で例外になると、窓を閉じる途中でアプリが落ちる
        renderer = FrameRenderer(rendered_project, context=gl_context)
        renderer.close()
        renderer.close()


class TestRenderQuality:
    def test_rejects_zero(self) -> None:
        # 0 の画質を受け付けると、プレビューの大きさを割り出す所が 0 除算で落ちる
        with pytest.raises(ValueError, match="1 以上"):
            RenderQuality(0)

    def test_never_shrinks_below_one_pixel(self) -> None:
        # 小さい作品を低い画質にしたとき 0 画素になると、合成先を作れずプレビューが止まる
        assert RenderQuality(100).apply(32, 32) == (1, 1)


class TestGeneratedCache:
    """作った絵を使い回すこと 動かない字幕を何本も重ねると、作り直しが再生の足を引く"""

    def _project(self, text: str = "字幕") -> Project:
        settings = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30))
        base = Project.create(settings)
        clip = Clip(timeline_start=0, duration=30, source=TEXT.create(text=text, size=16))
        track = Track(TrackKind.VIDEO, "V1", (clip,))
        return base.with_timeline(replace(base.timeline, tracks=(track,)))

    def test_a_still_text_is_drawn_once(self, gl_context: OffscreenGLContext) -> None:
        # 動かない字幕を毎フレーム作り直すと、字幕を何本も重ねたときに再生がもたつく
        project = self._project()
        renderer = FrameRenderer(project, context=gl_context)
        try:
            clip = project.timeline.tracks[0].clips[0]
            first = renderer._generate(clip, 0, project.rate)
            second = renderer._generate(clip, 7, project.rate)
            assert first is not None
            # 別のフレームでも同じ絵をそのまま返す（作り直していない）
            assert second is first
        finally:
            renderer.close()

    def test_an_animated_value_is_drawn_again(self, gl_context: OffscreenGLContext) -> None:
        # キーフレームで動く字幕まで使い回すと、文字が最初の位置に止まったまま動かない
        settings = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30))
        base = Project.create(settings)
        moving = AnimatedValue(0.0, (Keyframe(0, 0.0), Keyframe(29, 40.0)))
        source = TEXT.create(text="字幕", size=16)
        source = source.with_param("pos_x", moving)
        clip = Clip(timeline_start=0, duration=30, source=source)
        track = Track(TrackKind.VIDEO, "V1", (clip,))
        project = base.with_timeline(replace(base.timeline, tracks=(track,)))
        renderer = FrameRenderer(project, context=gl_context)
        try:
            first = renderer._generate(clip, 0, project.rate)
            second = renderer._generate(clip, 20, project.rate)
            assert first is not None and second is not None
            assert second is not first
        finally:
            renderer.close()
