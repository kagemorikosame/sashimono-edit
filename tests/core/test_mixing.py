"""音量の線を組む計算（ダッキングと全体の上げ下げ #329）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.commands import AddEffect, SetParam
from sashimono.core.commands.fixed import FADE_EFFECT_KIND, VOLUME_EFFECT_KIND, fixed_effect
from sashimono.core.commands.volume import clip_volume, volume_command, volume_commands
from sashimono.core.mixing import (
    VOLUME_LIMIT,
    DuckShape,
    audible_clips,
    db_to_ratio,
    duck_curve,
    loudest,
    scaled_volume,
    speech_frames,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Keyframe,
    MediaId,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate

RATE = FrameRate(30)
#: 300 ms で 9 フレーム 800 ms で 24 フレーム（30 fps）
SHAPE = DuckShape(depth_db=12, attack_ms=300, release_ms=800)
DUCKED = 100.0 * db_to_ratio(-12)


def _points(value: AnimatedValue | None) -> list[tuple[int, float]]:
    assert value is not None
    return [(k.frame, round(k.value, 3)) for k in value.keyframes]


class TestDuckCurve:
    def test_speech_in_the_middle_dips_and_comes_back(self) -> None:
        # 壊れると、話している間に BGM が下がらないか、話し終えても下がったまま戻らない
        curve = duck_curve(300, [(90, 150)], base=100.0, shape=SHAPE, rate=RATE)
        assert _points(curve) == [
            (81, 100.0),
            (90, round(DUCKED, 3)),
            (150, round(DUCKED, 3)),
            (174, 100.0),
        ]
        assert curve is not None
        # 下げ幅は dB で効く 12 dB は 4 分の 1 ほど
        assert curve.at(120) == pytest.approx(25.12, abs=0.01)
        assert curve.at(0) == 100.0 and curve.at(299) == 100.0

    def test_short_pauses_stay_ducked(self) -> None:
        # 壊れると、文と文の短い間で BGM が上がり下がりして落ち着かない（ポンピング）
        curve = duck_curve(300, [(90, 120), (140, 150)], base=100.0, shape=SHAPE, rate=RATE)
        assert [frame for frame, _ in _points(curve)] == [81, 90, 150, 174]

    def test_long_pauses_come_back_up(self) -> None:
        # 壊れると、長い間でも BGM が下がったままで、間が寂しく聞こえる
        curve = duck_curve(400, [(30, 60), (200, 230)], base=100.0, shape=SHAPE, rate=RATE)
        assert curve is not None
        assert len(curve.keyframes) == 8
        assert curve.at(130) == 100.0

    def test_speech_at_the_head_starts_ducked(self) -> None:
        # 壊れると、クリップの頭から話しているのに頭で BGM が大きいまま鳴る
        # （線は最初の点より前を最初の点の値で止めるので、頭に点が無いと下がらない）
        curve = duck_curve(300, [(0, 30)], base=100.0, shape=SHAPE, rate=RATE)
        assert _points(curve) == [(0, round(DUCKED, 3)), (30, round(DUCKED, 3)), (54, 100.0)]

    def test_speech_past_the_tail_stays_ducked_to_the_end(self) -> None:
        # 壊れると、クリップの終わりまで話しているのに、終わり際で BGM が戻る
        curve = duck_curve(100, [(80, 200)], base=100.0, shape=SHAPE, rate=RATE)
        assert _points(curve) == [(71, 100.0), (80, round(DUCKED, 3)), (99, round(DUCKED, 3))]

    def test_a_ramp_cut_by_the_tail_keeps_its_slope(self) -> None:
        # 壊れると、下げ始めの途中でクリップが終わったとき、端の値が下がり切った値に化ける
        curve = duck_curve(85, [(90, 120)], base=100.0, shape=SHAPE, rate=RATE)
        assert curve is not None
        (start, end) = curve.keyframes
        assert (start.frame, start.value) == (81, 100.0)
        assert end.frame == 84
        assert end.value == pytest.approx(100.0 + (DUCKED - 100.0) * 3 / 9)

    def test_no_speech_inside_the_clip_changes_nothing(self) -> None:
        # 壊れると、話していない BGM にも点が打たれ、人の線を消してしまう
        assert duck_curve(100, [(300, 400)], base=100.0, shape=SHAPE, rate=RATE) is None
        assert duck_curve(100, [], base=100.0, shape=SHAPE, rate=RATE) is None

    def test_the_base_is_kept(self) -> None:
        # 壊れると、前に 60% へ下げておいた BGM が、ダッキングで 100% へ戻される
        curve = duck_curve(300, [(90, 150)], base=60.0, shape=SHAPE, rate=RATE)
        assert curve is not None
        assert curve.at(0) == 60.0
        assert curve.at(120) == pytest.approx(60.0 * db_to_ratio(-12))

    def test_a_zero_ramp_still_takes_a_frame(self) -> None:
        # 壊れると、上の点と下の点が同じフレームに重なり、線を作る所で落ちる
        shape = DuckShape(depth_db=12, attack_ms=0, release_ms=0)
        assert shape.frames(RATE) == (1, 1)
        assert duck_curve(300, [(90, 150)], base=100.0, shape=shape, rate=RATE) is not None


class TestSpeechFrames:
    def test_source_seconds_follow_trim_and_speed(self) -> None:
        # 壊れると、頭を切ったクリップや速めたクリップで、話していない所の BGM が下がる
        clip = Clip(
            timeline_start=500,
            duration=300,
            media_id=MediaId("m"),
            source_in=Fraction(2),
            speed=Fraction(2),
        )
        # 素材の 3〜4 秒は、頭から (3-2)/2 = 0.5 秒〜 1 秒
        assert speech_frames(clip, [(Fraction(3), Fraction(4))], RATE) == [(15, 30)]

    def test_rounds_outward(self) -> None:
        # 壊れると、声の頭と終わりの 1 フレームが下がり切っていない BGM に乗る
        clip = Clip(timeline_start=0, duration=300, media_id=MediaId("m"))
        assert speech_frames(clip, [(Fraction(101, 100), Fraction(199, 100))], RATE) == [(30, 60)]

    def test_hidden_parts_are_dropped(self) -> None:
        # 壊れると、トリムで外した所の声で BGM が下がる
        clip = Clip(timeline_start=0, duration=60, media_id=MediaId("m"), source_in=Fraction(5))
        assert speech_frames(clip, [(Fraction(1), Fraction(4))], RATE) == []
        assert speech_frames(clip, [(Fraction(6), Fraction(20))], RATE) == [(30, 60)]


class TestScaling:
    def test_every_point_is_multiplied(self) -> None:
        # 壊れると、全体を上げたときにダッキングの下げ幅（dB）が変わり、声と BGM の釣り合いが崩れる
        value = AnimatedValue(
            static=100.0, keyframes=(Keyframe(frame=0, value=100.0), Keyframe(frame=9, value=25.0))
        )
        louder = scaled_volume(value, 2.0)
        assert [k.value for k in louder.keyframes] == [200.0, 50.0]
        assert loudest(louder) == 200.0

    def test_values_stop_at_the_limit(self) -> None:
        # 壊れると、欄の上限（400%）を超えた値が入り、設定パネルの欄と鳴る音が食い違う
        assert scaled_volume(AnimatedValue(static=300.0), 2.0).static == VOLUME_LIMIT


class TestVolumeCommands:
    def test_an_old_clip_gets_one_fixed_volume(self) -> None:
        # 壊れると、固定の音量を持たない古いクリップで取り消しが 2 段になり、1 回戻すと
        # 100% の音量調整だけが残る
        clip = Clip(
            timeline_start=0,
            duration=30,
            media_id=MediaId("m"),
            effects=(fixed_effect(FADE_EFFECT_KIND),),
        )
        (command,) = volume_commands(clip, AnimatedValue(static=50.0), AnimatedValue(static=-30.0))
        assert isinstance(command, AddEffect)
        assert command.effect.fixed and command.effect.kind == VOLUME_EFFECT_KIND
        assert command.effect.params["volume"] == AnimatedValue(static=50.0)
        assert command.effect.params["pan"] == AnimatedValue(static=-30.0)
        # 固定の項目どうしの並び（音量 → フェード）の所へ入る
        assert command.index == 0

    def test_a_placed_clip_changes_its_own_field(self) -> None:
        # 壊れると、置いた時から持っている音量の欄の横に、もう 1 つ音量調整が積まれる
        volume = fixed_effect(VOLUME_EFFECT_KIND)
        clip = Clip(timeline_start=0, duration=30, media_id=MediaId("m"), effects=(volume,))
        command = volume_command(clip, AnimatedValue(static=50.0))
        assert isinstance(command, SetParam)
        assert command.path.effect_id == volume.id
        assert clip_volume(clip).static == 100.0


class TestAudible:
    def _project(self, *tracks: Track) -> Project:
        base = Project.create(ProjectSettings(frame_rate=RATE))
        return base.with_timeline(replace(base.timeline, tracks=tracks))

    def test_it_follows_the_mixer(self) -> None:
        # 壊れると、ミキサが鳴らさない音（ソロで外れた・ミュート・無効）を測って音量をそろえる
        on = Clip(timeline_start=0, duration=30, media_id=MediaId("a"))
        off = Clip(timeline_start=30, duration=30, media_id=MediaId("a"), enabled=False)
        soloed = Track(kind=TrackKind.AUDIO, name="声", clips=(on, off), solo=True)
        other = Track(
            kind=TrackKind.AUDIO,
            name="BGM",
            clips=(Clip(timeline_start=0, duration=30, media_id=MediaId("b")),),
        )
        assert [c.id for _, c in audible_clips(self._project(soloed, other))] == [on.id]
        muted = replace(soloed, solo=False, muted=True)
        assert [c.id for _, c in audible_clips(self._project(muted, other))] == [other.clips[0].id]
