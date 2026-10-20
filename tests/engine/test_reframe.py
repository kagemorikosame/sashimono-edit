"""縦動画への再構図（#334）の命令を組む所

被写体の並び（:class:`~sashimono.engine.subject.SubjectTrack`）を手で組んで渡し、組んだ命令を
作品へ当てて、背景の動画の枠・重ねた文字・シーン・ロック・前からの動きの扱いを見る
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from sashimono.core.commands import AddScene, Command, Document
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, with_fixed_items
from sashimono.core.emphasis import fixed_transform
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    MediaItem,
    ParamValue,
    Project,
    ProjectSettings,
    Scene,
    Timeline,
    Track,
    TrackKind,
)
from sashimono.core.reframe import AUTO, CENTER, FOLLOW, STEADY, ReframeError
from sashimono.core.timebase import FrameRate
from sashimono.engine.reframe import ReframeRequest, plan_reframe, tracked_media
from sashimono.engine.scenes import SceneSignal
from sashimono.engine.subject import SUBJECT_BINS, SUBJECT_SECONDS, SubjectTrack

RATE = FrameRate(30)
REQUEST = ReframeRequest(1080, 1920, AUTO, 1000, 90)


def _subject(where: list[tuple[float, float]], *, until: float = 10.0) -> SubjectTrack:
    """``where`` は（この秒から, 被写体の横の位置 0〜1） 縦は真ん中"""
    times = np.arange(SUBJECT_SECONDS / 2, until, SUBJECT_SECONDS)
    columns = np.zeros((times.shape[0], SUBJECT_BINS), np.float32)
    rows = np.zeros((times.shape[0], SUBJECT_BINS), np.float32)
    for index, moment in enumerate(times):
        place = [x for start, x in where if moment >= start][-1]
        columns[index, min(SUBJECT_BINS - 1, int(place * SUBJECT_BINS))] = 1.0
        rows[index, SUBJECT_BINS // 2] = 1.0
    strength = np.full(times.shape, 0.02, np.float32)
    return SubjectTrack(times=times, columns=columns, rows=rows, strength=strength)


def _signal(where: list[tuple[float, float]], cuts: tuple[float, ...] = ()) -> SceneSignal:
    times = np.arange(1, 301, dtype=np.float64) / 30
    pixel = np.full(times.shape, 0.01, np.float32)
    histogram = np.full(times.shape, 0.01, np.float32)
    for moment in cuts:
        index = round(moment * 30) - 1
        pixel[index], histogram[index] = 0.3, 0.8
    return SceneSignal(
        times, pixel, histogram, analyzed=10.0, complete=True, stream=0, subject=_subject(where)
    )


def _video(video_media: MediaItem, *, native: bool = True, start: int = 0) -> Clip:
    clip = Clip(
        timeline_start=start,
        duration=300,
        media_id=video_media.id,
        native_size=native,
    )
    return with_fixed_items(clip, picture=True)


def _text(text: str, *, pos_y: float = -380.0, wrap: float = 0.0, layout: str = "native") -> Clip:
    params: dict[str, ParamValue] = {
        "text": text,
        "size": AnimatedValue(48.0),
        "pos_y": AnimatedValue(pos_y),
        "wrap_width": AnimatedValue(wrap),
        "layout": layout,
        "align": "left",
    }
    clip = Clip(timeline_start=0, duration=300, source=GeneratedSource(kind="text", params=params))
    return with_fixed_items(clip, picture=True)


def _project(video_media: MediaItem, *clips_on_tracks: tuple[Clip, ...]) -> Project:
    tracks = tuple(
        Track(TrackKind.VIDEO, f"V{index + 1}", clips)
        for index, clips in enumerate(clips_on_tracks)
    )
    return Project(
        settings=ProjectSettings(width=1920, height=1080, frame_rate=RATE),
        timeline=Timeline(rate=RATE, tracks=tracks),
        media=(video_media,),
    )


def _applied(project: Project, commands: list[Command]) -> Project:
    document = Document(project)
    with document.checkpoint("作り替え"):
        for command in commands:
            document.execute(command)
    return document.project


def _transform(project: Project, clip: Clip) -> Effect:
    located = project.timeline.locate_clip(clip.id)
    if located is None:
        for scene in project.scenes:
            located = scene.timeline.locate_clip(clip.id)
            if located is not None:
                break
    assert located is not None
    effect = fixed_transform(located[1])
    assert effect is not None
    return effect


def _value(effect: Effect, name: str, default: float) -> AnimatedValue:
    value = effect.params.get(name)
    return value if isinstance(value, AnimatedValue) else AnimatedValue(default)


def _subject_on_screen(project: Project, clip: Clip, frame: int, across: float) -> float:
    """絵の左から ``across`` の所が、縦の画面の中央から何画素の所に描かれるか（画素で置く動画）"""
    effect = _transform(project, clip)
    pos = _value(effect, "pos_x", 0.0).at(frame)
    scale = _value(effect, "scale", 100.0).at(frame) / 100.0
    return pos + scale * (across - 0.5) * 1920


class TestTheBackground:
    def test_the_subject_on_the_right_comes_to_the_middle(self, video_media: MediaItem) -> None:
        # 壊れると、右に座った話し手の縦の作品で、人が画面の外に切れて背景だけが映る
        video = _video(video_media)
        project = _project(video_media, (video,))
        plan = plan_reframe(project, REQUEST, {video_media.id: _signal([(0.0, 0.78)])})
        done = _applied(project, plan.commands)
        assert done.settings.resolution == (1080, 1920)
        assert abs(_subject_on_screen(done, video, 100, 0.78)) < 1920 / 32 * 2
        assert plan.shots[0].mode == STEADY
        assert _value(_transform(done, video), "scale", 100.0).static == pytest.approx(
            100 * 1920 / 1080
        )

    def test_a_cut_moves_the_window_to_the_new_place(self, video_media: MediaItem) -> None:
        # 場面の切り替わりで枠を変えないと、左右に座った 2 人の掛け合いで片方が映らない
        video = _video(video_media)
        project = _project(video_media, (video,))
        signal = _signal([(0.0, 0.78), (5.0, 0.22)], cuts=(5.0,))
        done = _applied(project, plan_reframe(project, REQUEST, {video_media.id: signal}).commands)
        assert abs(_subject_on_screen(done, video, 60, 0.78)) < 120
        assert abs(_subject_on_screen(done, video, 200, 0.22)) < 120
        # 切り替わりの直前までは前の位置のまま（枠が流れない）
        assert abs(_subject_on_screen(done, video, 149, 0.78)) < 120

    def test_a_walking_subject_is_followed(self, video_media: MediaItem) -> None:
        # 止めたままにすると、画面を横切る人が途中で枠から出ていく
        video = _video(video_media)
        project = _project(video_media, (video,))
        walk = [(t / 4, 0.1 + 0.8 * t / 40) for t in range(40)]
        plan = plan_reframe(project, REQUEST, {video_media.id: _signal(walk)})
        done = _applied(project, plan.commands)
        assert plan.shots[0].mode == FOLLOW
        for frame in (60, 150, 240):
            across = 0.1 + 0.8 * (frame / 30) / 10
            assert abs(_subject_on_screen(done, video, frame, across)) < 1080 / 2

    def test_center_ignores_the_subject(self, video_media: MediaItem) -> None:
        # 真ん中を頼まれて寄せると、解析を待たない約束と違う絵になる
        video = _video(video_media)
        project = _project(video_media, (video,))
        request = replace(REQUEST, follow=CENTER)
        done = _applied(
            project, plan_reframe(project, request, {video_media.id: _signal([(0, 0.9)])}).commands
        )
        assert _subject_on_screen(done, video, 0, 0.5) == pytest.approx(0.0)

    def test_a_fitted_clip_is_scaled_for_the_new_screen(self, video_media: MediaItem) -> None:
        # 画面に収めて置く古いクリップは縦の画面で幅に合わせて縮む 戻さないと上下に黒い帯が出る
        video = _video(video_media, native=False)
        project = _project(video_media, (video,))
        done = _applied(project, plan_reframe(project, REQUEST, {}).commands)
        scale = _value(_transform(done, video), "scale", 100.0).static / 100.0
        # 縦の画面では幅 1080 に収まる（1080x607.5） それを縦 1920 まで拡げる
        assert 607.5 * scale == pytest.approx(1920.0)

    def test_a_flipped_clip_looks_at_the_mirrored_side(self, video_media: MediaItem) -> None:
        # 左右反転を数えないと、反転した素材で被写体と逆の側を切り出す
        video = _video(video_media)
        flipped = replace(
            video,
            effects=tuple(
                e.with_param("horizontal", True) if e.kind == "flip" else e for e in video.effects
            ),
        )
        project = _project(video_media, (flipped,))
        done = _applied(
            project,
            plan_reframe(project, REQUEST, {video_media.id: _signal([(0, 0.8)])}).commands,
        )
        # 素材の右（0.8）は反転して画面の左（0.2）に描かれる
        assert abs(_subject_on_screen(done, flipped, 100, 0.2)) < 120

    def test_an_own_zoom_is_kept_and_the_window_held(self, video_media: MediaItem) -> None:
        # 前からの寄り（位置の点）を捨てると、人が付けた強調の動きが作り替えで消える
        video = _video(video_media)
        moving = AnimatedValue(0.0, (Keyframe(0, 0.0), Keyframe(100, 200.0)))
        effects = tuple(
            e.with_param("pos_x", moving) if e.kind == TRANSFORM_EFFECT_KIND else e
            for e in video.effects
        )
        video = replace(video, effects=effects)
        project = _project(video_media, (video,))
        walk = [(t / 4, 0.1 + 0.8 * t / 40) for t in range(40)]
        plan = plan_reframe(project, REQUEST, {video_media.id: _signal(walk)})
        done = _applied(project, plan.commands)
        assert str(video.id) in plan.held
        pos = _value(_transform(done, video), "pos_x", 0.0)
        assert len(pos.keyframes) == 2
        assert pos.at(100) - pos.at(0) == pytest.approx(200.0 * 1920 / 1080)

    def test_only_moving_backgrounds_wait_for_the_analysis(self, video_media: MediaItem) -> None:
        # 文字や小さい画像の素材まで待つと、待たなくてよい作品で解析の終わりを待たせる
        video = _video(video_media)
        project = _project(video_media, (video,), (_text("字幕"),))
        assert [media.id for media in tracked_media(project)] == [video_media.id]


def _with_transform(clip: Clip, **values: float) -> Clip:
    effects = tuple(
        _set_all(e, values) if e.kind == TRANSFORM_EFFECT_KIND else e for e in clip.effects
    )
    return replace(clip, effects=effects)


def _set_all(effect: Effect, values: dict[str, float]) -> Effect:
    for name, value in values.items():
        effect = effect.with_param(name, AnimatedValue(value))
    return effect


def _turned_on_screen(
    project: Project, clip: Clip, frame: int, point: tuple[float, float]
) -> float:
    """絵の中の点（素材の中央からの画素 Y は上が正）が縦の画面の中央から横に何画素の所か

    配置の式（シェーダの _TRANSFORM と同じ） 位置 + 回転(-角度) x 拡大 x 点（支点は画面の中央）
    """
    effect = _transform(project, clip)
    pos = _value(effect, "pos_x", 0.0).at(frame)
    scale = _value(effect, "scale", 100.0).at(frame) / 100.0
    turn = -math.radians(_value(effect, "rotation", 0.0).at(frame))
    return pos + scale * (math.cos(turn) * point[0] - math.sin(turn) * point[1])


class TestTheSpreadMotion:
    def test_a_spread_motion_is_no_clue(self, video_media: MediaItem) -> None:
        # 画面じゅうに散った動きを手がかりにすると、たまたまの偏りへ枠が寄って跳ね回る
        video = _video(video_media)
        project = _project(video_media, (video,))
        signal = _signal([(0.0, 0.8)])
        flat = replace(
            signal.subject, columns=np.full_like(signal.subject.columns, 1.0 / SUBJECT_BINS)
        )
        plan = plan_reframe(project, REQUEST, {video_media.id: replace(signal, subject=flat)})
        assert plan.shots[0].no_clue


class TestTheTurnAndPixels:
    def test_a_turned_clip_counts_the_turn(self, video_media: MediaItem) -> None:
        # 回転を数えないと、傾けて置いた動画で、傾ける前の位置へ枠を合わせて人が外れる
        video = _with_transform(_video(video_media), rotation=30.0, scale=130.0)
        project = _project(video_media, (video,))
        done = _applied(
            project,
            plan_reframe(project, REQUEST, {video_media.id: _signal([(0.0, 0.62)])}).commands,
        )
        across = (int(0.62 * SUBJECT_BINS) + 0.5) / SUBJECT_BINS
        point = ((across - 0.5) * 1920, 0.0)
        assert abs(_turned_on_screen(done, video, 100, point)) < 20

    def test_a_turned_window_stays_on_the_picture(self, video_media: MediaItem) -> None:
        # 回した絵の外接の箱で範囲を取ると、枠の角が絵の外へ出て縦の画面の隅が黒くなる
        video = _with_transform(_video(video_media), rotation=10.0, scale=150.0)
        project = _project(video_media, (video,))
        plan = plan_reframe(project, REQUEST, {video_media.id: _signal([(0.0, 0.99)])})
        focus = plan.shots[0].focus
        crop = plan.window.size
        turn = math.radians(10.0)
        half = (1.5 * 1920 / 2, 1.5 * 1080 / 2)
        for corner in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            x = focus[0] + corner[0] * crop[0] / 2
            y = focus[1] + corner[1] * crop[1] / 2
            # 枠の角を絵の向きへ戻して、絵の中にあるかを見る
            local = (
                math.cos(turn) * x - math.sin(turn) * y,
                math.sin(turn) * x + math.cos(turn) * y,
            )
            assert abs(local[0]) <= half[0] + 1e-6 and abs(local[1]) <= half[1] + 1e-6

    def test_a_blur_on_a_fitted_clip_keeps_its_look(self, video_media: MediaItem) -> None:
        # 画面に収めて置く古いクリップで直さないと、縦の画面で拡げ直したときにぼかしだけが太る
        blur = Effect(kind="blur", params={"radius": AnimatedValue(16.0)})
        video = _video(video_media, native=False)
        video = replace(video, effects=(blur, *video.effects))
        project = _project(video_media, (video,))
        done = _applied(project, plan_reframe(project, REQUEST, {}).commands)
        located = done.timeline.locate_clip(video.id)
        assert located is not None
        radius = next(e for e in located[1].effects if e.kind == "blur").params["radius"]
        # 縦の画面で置いた絵は幅 1080 に収まって 0.5625 倍 ぼかしも同じ比にする
        assert radius == AnimatedValue(16.0 * 1080 / 1920)

    def test_a_blur_on_a_pixel_clip_is_left_alone(self, video_media: MediaItem) -> None:
        # 画素で置くクリップのぼかしを直すと、拡大と二重に掛かってぼけ方が変わる
        blur = Effect(kind="blur", params={"radius": AnimatedValue(16.0)})
        video = _video(video_media)
        video = replace(video, effects=(blur, *video.effects))
        project = _project(video_media, (video,))
        done = _applied(project, plan_reframe(project, REQUEST, {}).commands)
        located = done.timeline.locate_clip(video.id)
        assert located is not None
        assert next(e for e in located[1].effects if e.kind == "blur").params["radius"] == (
            AnimatedValue(16.0)
        )

    def test_an_effect_after_the_frame_grows_with_the_window(self, video_media: MediaItem) -> None:
        # 配置より後ろのエフェクトは画面の上で掛かる 拡げないと縦の画面でだけぼかしが細くなる
        blur = Effect(kind="blur")
        video = _video(video_media)
        video = replace(video, effects=(*video.effects, blur))
        project = _project(video_media, (video,))
        done = _applied(project, plan_reframe(project, REQUEST, {}).commands)
        located = done.timeline.locate_clip(video.id)
        assert located is not None
        radius = next(e for e in located[1].effects if e.kind == "blur").params["radius"]
        assert radius == AnimatedValue(8.0 * 1920 / 1080)


class TestTheOverlays:
    def test_a_caption_stays_at_the_bottom_and_wraps(self, video_media: MediaItem) -> None:
        # 壊れると、横の画面の下の字幕が縦の画面の中ほどに浮き、長い字幕は縦の画面の端で切れる
        caption = _text("長い字幕の文がここに入ります")
        project = _project(video_media, (_video(video_media),), (caption,))
        plan = plan_reframe(project, REQUEST, {})
        done = _applied(project, plan.commands)
        located = done.timeline.locate_clip(caption.id)
        assert located is not None
        source = located[1].source
        assert source is not None
        assert source.params["pos_y"] == AnimatedValue(-380.0 * 1920 / 1080)
        wrap = source.params["wrap_width"]
        assert isinstance(wrap, AnimatedValue)
        assert wrap.static == pytest.approx(1080 * 0.9, abs=1)
        assert str(caption.id) in plan.wrapped

    def test_an_aviutl_layout_switches_to_wrap(self, video_media: MediaItem) -> None:
        # AviUtl2 の組み方のまま幅を入れても折り返さず、縦の画面で字幕が切れる
        caption = _text("字幕", layout="aviutl")
        project = _project(video_media, (caption,))
        done = _applied(project, plan_reframe(project, REQUEST, {}).commands)
        located = done.timeline.locate_clip(caption.id)
        assert located is not None and located[1].source is not None
        assert located[1].source.params["layout"] == "native"

    def test_turning_wrap_off_leaves_the_width(self, video_media: MediaItem) -> None:
        # 折り返しを切った人の字幕まで折り返すと、設定が効かない
        caption = _text("字幕")
        project = _project(video_media, (caption,))
        plan = plan_reframe(project, replace(REQUEST, wrap_share=0), {})
        done = _applied(project, plan.commands)
        located = done.timeline.locate_clip(caption.id)
        assert located is not None and located[1].source is not None
        assert located[1].source.params["wrap_width"] == AnimatedValue(0.0)
        assert not plan.wrapped

    def test_a_left_title_is_pulled_inside(self, video_media: MediaItem) -> None:
        # 寄せないと、左寄せのタイトルが縦の画面の右へはみ出す
        caption = _text("見出し")
        effects = tuple(
            e.with_param("pos_x", AnimatedValue(500.0)) if e.kind == TRANSFORM_EFFECT_KIND else e
            for e in caption.effects
        )
        caption = replace(caption, effects=effects)
        caption = replace(
            caption,
            source=caption.source.with_param("anchor", "left") if caption.source else None,
        )
        project = _project(video_media, (caption,))
        done = _applied(project, plan_reframe(project, REQUEST, {}).commands)
        x = _value(_transform(done, caption), "pos_x", 0.0).static
        assert x + 1080 * 0.9 <= 540 + 1e-6


class TestTheWholeWork:
    def test_scenes_are_reframed_too(self, video_media: MediaItem) -> None:
        # シーンを残すと、シーンを置いた所だけ縦の画面で横長の絵が小さく潰れる
        inner = _video(video_media)
        project = _project(video_media, (_video(video_media),))
        scene = Scene("中", Timeline(rate=RATE, tracks=(Track(TrackKind.VIDEO, "V1", (inner,)),)))
        project = AddScene(scene).apply(project)
        done = _applied(project, plan_reframe(project, REQUEST, {}).commands)
        assert _value(_transform(done, inner), "scale", 100.0).static == pytest.approx(
            100 * 1920 / 1080
        )

    def test_a_locked_track_stops_everything(self, video_media: MediaItem) -> None:
        # ロックを飛ばして続けると、そのトラックだけ横の配置のまま縦の画面に残る
        project = _project(video_media, (_video(video_media),))
        track = replace(project.timeline.tracks[0], locked=True)
        project = project.with_timeline(project.timeline.replace_track(track))
        with pytest.raises(ReframeError, match="ロック"):
            plan_reframe(project, REQUEST, {})

    def test_one_undo_brings_the_wide_work_back(self, video_media: MediaItem) -> None:
        # 何段にも分かれると、人が Ctrl+Z を何度も押さないと横の作品へ戻れない
        project = _project(video_media, (_video(video_media),), (_text("字幕"),))
        document = Document(project)
        with document.checkpoint("作り替え"):
            for command in plan_reframe(project, REQUEST, {}).commands:
                document.execute(command)
        document.undo()
        assert document.project == project

    def test_a_late_clip_counts_its_source_time(self, video_media: MediaItem) -> None:
        # 素材の頭を切ったクリップで時刻をずらし損ねると、別の場面の被写体へ枠を合わせる
        video = replace(_video(video_media), source_in=Fraction(5))
        video = replace(video, duration=150)
        project = _project(video_media, (video,))
        signal = _signal([(0.0, 0.2), (5.0, 0.8)])
        done = _applied(project, plan_reframe(project, REQUEST, {video_media.id: signal}).commands)
        assert abs(_subject_on_screen(done, video, 60, 0.8)) < 120
