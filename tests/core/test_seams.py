"""ジャンプカットのつなぎ目の処理（#332 B6） core/seams.py

つなぎ目を探す決まり（同じ素材を飛び飛びにつないだ所だけ）と、つなぎ目へ付ける物
（場面切り替えのクロスフェード・1 本おきの寄り・音のフェード）を、人が手で付ける物と
同じ欄と命令で作ることを見る AI の道具からの呼び方は tests/ai/test_speed_seam_tools.py
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands import AddClip, Document, RippleCut, SetTrackState
from sashimono.core.commands.fixed import (
    FADE_EFFECT_KIND,
    TRANSFORM_EFFECT_KIND,
    fixed_effect,
)
from sashimono.core.commands.insert import insert_generated
from sashimono.core.emphasis import fixed_transform
from sashimono.core.model import (
    TRANSITION_KIND,
    AnimatedValue,
    AudioStreamInfo,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    VideoStreamInfo,
    new_clip_id,
    new_group_id,
)
from sashimono.core.seams import (
    SEAM_CROSSFADE,
    SEAM_NONE,
    SEAM_ZOOM,
    SeamStyle,
    SmoothCuts,
    find_seams,
    plan_smoothing,
    seams_after_cuts,
)
from sashimono.core.timebase import FrameRate

RATE = FrameRate(30)
#: 場面切り替えの中身 コアはエフェクトの定義を読めないので、試験でも手で作る
FADE = GeneratedSource(kind=TRANSITION_KIND, params={"style": "fade"})
#: 音だけ処理する
AUDIO_ONLY = SeamStyle(video=SEAM_NONE)


def talking_media() -> MediaItem:
    """30 秒の話す動画 映像 1 本と音声 1 本"""
    return MediaItem(
        path=Path("C:/素材/話.mp4"),
        duration=Fraction(30),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1920,
                height=1080,
                frame_rate=RATE,
                time_base=Fraction(1, 15360),
                codec="h264",
                pixel_format="yuv420p",
            ),
        ),
        audio_streams=(AudioStreamInfo(1, 48000, 2, Fraction(1, 48000), "aac"),),
    )


class JumpCut:
    """V1 と A1 に、同じ素材を 3 本ずつ飛び飛びにつないだ作品（ジャンプカット）

    0〜60 は素材の 0 秒から、60〜120 は 5 秒から、120〜180 は 10 秒から つなぎ目は 60 と 120
    映像と音はリンクしている 映像は最初から配置の欄を持つ（置いたときと同じ）
    """

    def __init__(self, *, durations: tuple[int, int, int] = (60, 60, 60)) -> None:
        self.media = talking_media()
        base = Project.create(ProjectSettings(frame_rate=RATE), media=(self.media,))
        self.video = Track(TrackKind.VIDEO, "V1")
        self.audio = Track(TrackKind.AUDIO, "A1")
        project = base.with_timeline(replace(base.timeline, tracks=(self.video, self.audio)))
        self.pictures: list[Clip] = []
        self.sounds: list[Clip] = []
        start = 0
        for index, duration in enumerate(durations):
            sound = Clip(
                timeline_start=start,
                duration=duration,
                media_id=self.media.id,
                source_in=Fraction(index * 5),
                link_group=new_group_id(),
            )
            picture = replace(sound, effects=(fixed_effect(TRANSFORM_EFFECT_KIND),))
            sound = replace(sound, id=new_clip_id())
            project = AddClip(self.video.id, picture).apply(project)
            project = AddClip(self.audio.id, sound).apply(project)
            self.pictures.append(picture)
            self.sounds.append(sound)
            start += duration
        self.project = project
        self.seams = (durations[0], durations[0] + durations[1])


def _clip(project: Project, clip: Clip) -> Clip:
    located = project.timeline.locate_clip(clip.id)
    assert located is not None
    return located[1]


def _fade(project: Project, clip: Clip) -> dict[str, float]:
    found = _clip(project, clip)
    effect = next(e for e in found.effects if e.kind == FADE_EFFECT_KIND)
    values = {}
    for name in ("fade_in", "fade_out"):
        value = effect.params[name]
        assert isinstance(value, AnimatedValue)
        values[name] = value.static
    return values


def _scale(project: Project, clip: Clip) -> float:
    transform = fixed_transform(_clip(project, clip))
    assert transform is not None
    value = transform.params["scale"]
    assert isinstance(value, AnimatedValue)
    return value.static


def _transitions(project: Project) -> list[tuple[Track, Clip]]:
    return [
        (track, clip)
        for track in project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == TRANSITION_KIND
    ]


class TestFindSeams:
    def test_jumps_in_the_same_media_are_seams(self) -> None:
        # 壊れると、無音を詰めた所を AI が探せず、つなぎ目の位置を 1 つずつ数えて渡すことになる
        work = JumpCut()
        assert find_seams(work.project) == (60, 120)

    def test_a_plain_split_is_not_a_seam(self) -> None:
        # 壊れると、分割しただけで絵の続いている所にまでクロスフェードが掛かり、止めた絵が挟まる
        work = JumpCut()
        project = work.project
        for track in (work.video, work.audio):
            current = project.timeline.find_track(track.id)
            assert current is not None
            clips = sorted(current.clips, key=lambda c: c.timeline_start)
            # 60 からの 2 本目を、1 本目の続き（素材の 2 秒）から流す
            joined = replace(clips[1], source_in=Fraction(2))
            others = tuple(c for c in current.clips if c.id != joined.id)
            project = project.with_timeline(
                project.timeline.replace_track(current.with_clips((*others, joined)))
            )
        assert find_seams(project) == (120,)

    def test_a_cut_to_another_media_is_not_a_jump(self) -> None:
        # 壊れると、別の素材へ切り替えた（場面を替えた）所まで、ジャンプカットとして溶かす
        work = JumpCut()
        other = replace(talking_media(), path=Path("C:/素材/別.mp4"))
        project = replace(work.project, media=(*work.project.media, other))
        track = project.timeline.find_track(work.video.id)
        assert track is not None
        swapped = replace(work.pictures[2], media_id=other.id)
        others = tuple(c for c in track.clips if c.id != swapped.id)
        project = project.with_timeline(
            project.timeline.replace_track(track.with_clips((*others, swapped)))
        )
        assert find_seams(project, track_ids=(work.video.id,)) == (60,)

    def test_the_range_and_locked_tracks_are_respected(self) -> None:
        # 壊れると、範囲の外やロックしたトラックのつなぎ目まで変わり、人が守った所に手が入る
        work = JumpCut()
        assert find_seams(work.project, start=61) == (120,)
        assert find_seams(work.project, end=120) == (60,)
        locked = SetTrackState(work.video.id, locked=True).apply(work.project)
        locked = SetTrackState(work.audio.id, locked=True).apply(locked)
        assert find_seams(locked) == ()

    def test_seams_after_cuts_follow_the_closed_gaps(self) -> None:
        # 壊れると、無音を詰めた後のつなぎ目の位置が、前で削った長さぶんずれて別の所を溶かす
        assert seams_after_cuts(((30, 40), (100, 130))) == (30, 90)
        # 頭から削った所は前にクリップが無いので数えない
        assert seams_after_cuts(((0, 10), (50, 60))) == (40,)


class TestAudio:
    def test_both_sides_of_a_seam_get_a_short_fade(self) -> None:
        # 壊れると、波形が途中で切れた所で「プツッ」と鳴ったまま残る
        work = JumpCut()
        plan = plan_smoothing(work.project, work.seams, AUDIO_ONLY)
        assert plan.audio == (60, 120)
        assert _fade(plan.project, work.sounds[0]) == {"fade_in": 0.0, "fade_out": 0.02}
        assert _fade(plan.project, work.sounds[1]) == {"fade_in": 0.02, "fade_out": 0.02}
        assert _fade(plan.project, work.sounds[2]) == {"fade_in": 0.02, "fade_out": 0.0}

    def test_a_longer_fade_is_kept_and_a_moving_one_is_left_alone(self) -> None:
        # 壊れると、人が付けた長いフェードを短く縮めるか、キーフレームで動かした長さを潰す
        work = JumpCut()
        long_fade = fixed_effect(FADE_EFFECT_KIND).with_param("fade_out", AnimatedValue(static=0.5))
        moving = fixed_effect(FADE_EFFECT_KIND).with_param(
            "fade_in", AnimatedValue(keyframes=(Keyframe(0, 0.1), Keyframe(10, 0.3)))
        )
        project = work.project
        track = project.timeline.find_track(work.audio.id)
        assert track is not None
        first = replace(work.sounds[0], effects=(long_fade,))
        second = replace(work.sounds[1], effects=(moving,))
        keep = tuple(c for c in track.clips if c.id not in (first.id, second.id))
        project = project.with_timeline(
            project.timeline.replace_track(track.with_clips((*keep, first, second)))
        )
        plan = plan_smoothing(project, (60,), AUDIO_ONLY)
        assert _fade(plan.project, work.sounds[0])["fade_out"] == 0.5
        found = _clip(plan.project, work.sounds[1])
        fade = next(e for e in found.effects if e.kind == FADE_EFFECT_KIND)
        assert fade.params["fade_in"] == moving.params["fade_in"]

    def test_a_short_clip_fades_at_most_half_of_itself(self) -> None:
        # 壊れると、1 フレームのクリップで頭と終わりのフェードが重なり、音がほとんど消える
        work = JumpCut(durations=(60, 1, 60))
        plan = plan_smoothing(
            work.project, work.seams, SeamStyle(video=SEAM_NONE, audio_fade_ms=200)
        )
        fades = _fade(plan.project, work.sounds[1])
        assert fades["fade_in"] == pytest.approx(1 / 60)
        assert fades["fade_out"] == pytest.approx(1 / 60)


class TestCrossfade:
    def test_a_fade_starts_one_frame_before_the_seam_above_the_picture(self) -> None:
        # 壊れると、場面切り替えがつなぎ目の絵より下に入って何も溶けないか、真ん中に置かれて
        # つなぎ目で急に半分まで溶けた絵が出る
        work = JumpCut()
        plan = plan_smoothing(work.project, work.seams, SeamStyle(audio_fade_ms=0), FADE)
        # 100 ms は 30 fps で 3 フレーム
        assert plan.crossfades == ((60, 3), (120, 3))
        placed = _transitions(plan.project)
        assert [(c.timeline_start, c.duration) for _, c in placed] == [(59, 4), (119, 4)]
        order = [t.id for t in plan.project.timeline.tracks]
        for track, _clip_placed in placed:
            assert order.index(track.id) > order.index(work.video.id)

    def test_the_fade_ends_inside_the_next_clip(self) -> None:
        # 壊れると、短いクリップの後ろで溶かす範囲が次のつなぎ目の切り替えと重なり、別の
        # トラックへ逃げて二重に溶ける
        work = JumpCut(durations=(60, 3, 60))
        style = SeamStyle(crossfade_ms=1000, audio_fade_ms=0)
        plan = plan_smoothing(work.project, work.seams, style, FADE)
        assert plan.crossfades == ((60, 2), (63, 30))
        tracks = {track.id for track, _ in _transitions(plan.project)}
        assert len(tracks) == 1

    def test_running_it_again_adds_nothing(self) -> None:
        # 壊れると、AI が同じ頼みを 2 回送ったとき、つなぎ目に切り替えが二重に重なる
        work = JumpCut()
        style = SeamStyle(audio_fade_ms=0)
        first = plan_smoothing(work.project, work.seams, style, FADE)
        again = plan_smoothing(first.project, work.seams, style, FADE)
        assert again.commands == ()
        assert {reason for _, reason in again.skipped} == {"もう場面切り替えが掛かっている"}

    def test_a_one_frame_clip_before_the_seam_is_skipped(self) -> None:
        # 壊れると、前の場面に入るクリップが無いまま切り替えが置かれ、黒から溶けて出る
        work = JumpCut(durations=(1, 60, 60))
        plan = plan_smoothing(work.project, work.seams, SeamStyle(audio_fade_ms=0), FADE)
        assert plan.crossfades == ((61, 3),)
        assert (1, "前のクリップが 1 フレームしかない") in plan.skipped


class TestZoom:
    def test_every_other_clip_is_pushed_in(self) -> None:
        # 壊れると、全部のクリップが寄って跳びが残るか、拡大が積み重なって画面の外へ出る
        work = JumpCut()
        style = SeamStyle(video=SEAM_ZOOM, audio_fade_ms=0)
        plan = plan_smoothing(work.project, work.seams, style)
        assert [_scale(plan.project, clip) for clip in work.pictures] == [100.0, 110.0, 100.0]
        assert plan.zoomed == (work.pictures[1].id,)
        assert plan.handled == (60, 120)
        again = plan_smoothing(plan.project, work.seams, style)
        assert again.commands == ()

    def test_a_moving_scale_is_not_touched(self) -> None:
        # 壊れると、人が付けた寄りの動き（キーフレーム）を 1 つの値で潰す
        work = JumpCut()
        project = work.project
        track = project.timeline.find_track(work.video.id)
        assert track is not None
        moving = AnimatedValue(keyframes=(Keyframe(0, 100.0), Keyframe(30, 130.0)))
        transform = fixed_effect(TRANSFORM_EFFECT_KIND).with_param("scale", moving)
        second = replace(work.pictures[1], effects=(transform,))
        keep = tuple(c for c in track.clips if c.id != second.id)
        project = project.with_timeline(
            project.timeline.replace_track(track.with_clips((*keep, second)))
        )
        plan = plan_smoothing(project, (60,), SeamStyle(video=SEAM_ZOOM, audio_fade_ms=0))
        assert plan.commands == ()
        assert plan.skipped == ((60, f"{second.id} の拡大率がキーフレームで動いている"),)


class TestOneUndo:
    def test_cutting_and_smoothing_go_back_in_one_undo(self) -> None:
        # 壊れると、無音を詰めた後のつなぎ目の処理を戻すのに取り消しが 2 回要るか、詰めた後の
        # クリップの ID が当てるまで決まらないので、つなぎ目へ何も付かない
        work = JumpCut(durations=(180, 1, 1))
        document = Document(work.project)
        ranges = ((50, 70),)
        style = SeamStyle()
        with document.checkpoint("無音カット"):
            document.execute(RippleCut(ranges))
            document.execute(SmoothCuts(seams_after_cuts(ranges), style, FADE))
        cut = document.project
        assert [(c.timeline_start, c.duration) for _, c in _transitions(cut)] == [(49, 4)]
        track = cut.timeline.find_track(work.audio.id)
        assert track is not None
        right = next(c for c in track.clips if c.timeline_start == 50)
        fade = next(e for e in right.effects if e.kind == FADE_EFFECT_KIND)
        assert fade.params["fade_in"] == AnimatedValue(static=0.02)
        document.undo()
        assert document.project == work.project

    def test_nothing_to_do_changes_nothing(self) -> None:
        # 壊れると、つなぎ目の無い所へ掛けたときに空の 1 段が履歴に積まれ、AI の取り消しを止める
        work = JumpCut()
        command = SmoothCuts((30,), SeamStyle(), FADE)
        assert command.apply(work.project) is work.project

    def test_a_crossfade_needs_the_transition(self) -> None:
        # 壊れると、中身の無い場面切り替えが置かれ、設定パネルに欄が出ない
        work = JumpCut()
        with pytest.raises(ValueError, match="場面切り替え"):
            plan_smoothing(work.project, work.seams, SeamStyle(video=SEAM_CROSSFADE))


def test_a_transition_from_the_menu_goes_above_the_picture() -> None:
    # 壊れると、切れ目の絵より下に空いたトラックがあると場面切り替えがそこへ入り、置いたのに
    # 何も変わらない（場面切り替えは下のトラックにしか効かない）
    work = JumpCut()
    empty = Track(TrackKind.VIDEO, "V0")
    tracks = (empty, *work.project.timeline.tracks)
    project = work.project.with_timeline(replace(work.project.timeline, tracks=tracks))
    commands = insert_generated(project, FADE, at_frame=55, duration=10)
    added = [c for c in commands if isinstance(c, AddClip)]
    assert added
    after = project
    for command in commands:
        after = command.apply(after)
    order = [t.id for t in after.timeline.tracks]
    assert order.index(added[0].track_id) > order.index(work.video.id)


def test_effects_other_than_fade_are_untouched() -> None:
    # 壊れると、つなぎ目のフェードを足すときに、クリップに積んだほかのエフェクトの並びが変わる
    work = JumpCut()
    blur = Effect(kind="blur")
    project = work.project
    track = project.timeline.find_track(work.audio.id)
    assert track is not None
    first = replace(work.sounds[0], effects=(blur,))
    keep = tuple(c for c in track.clips if c.id != first.id)
    project = project.with_timeline(
        project.timeline.replace_track(track.with_clips((*keep, first)))
    )
    plan = plan_smoothing(project, (60,), AUDIO_ONLY)
    kinds = [e.kind for e in _clip(plan.project, first).effects]
    assert kinds == ["blur", FADE_EFFECT_KIND]
