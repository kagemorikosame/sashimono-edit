"""再生速度を変える命令（#332 B6） core/commands/speed.py

中身を保って長さを合わせる変え方と、設定パネルと同じ長さを保つ変え方の両方で、リンクした
絵と音がそろって変わること、ぶつかる・ロック・時間を持たないクリップを断ることを見る
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.commands import AddClip, SetTrackState
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, fixed_effect
from sashimono.core.commands.speed import ChangeSpeed, sped_duration
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    Project,
)
from tests.core.test_seams import RATE, JumpCut


def _clip(project: Project, clip: Clip) -> Clip:
    located = project.timeline.locate_clip(clip.id)
    assert located is not None
    return located[1]


class TestKeepContent:
    def test_double_speed_halves_the_clip_and_its_partner(self) -> None:
        # 壊れると、「2 倍速にして」で長さが変わらず別の場面まで流れるか、音だけ元の速さで
        # 絵とずれていく
        work = JumpCut()
        after = ChangeSpeed((work.pictures[0].id,), Fraction(2)).apply(work.project)
        for clip in (work.pictures[0], work.sounds[0]):
            sped = _clip(after, clip)
            assert (sped.speed, sped.duration) == (Fraction(2), 30)
            # 素材の同じ範囲（0〜2 秒）を流す
            assert sped.source_out(RATE) == work.pictures[0].source_out(RATE)

    def test_slowing_down_into_the_next_clip_is_refused(self) -> None:
        # 壊れると、遅くして長くなったクリップが後ろのクリップに重なり、どちらかが消えるか
        # 黙って短く切られて頼んだ速度と長さが合わない
        work = JumpCut()
        with pytest.raises(ValueError, match="あと 0 フレーム"):
            ChangeSpeed((work.pictures[0].id,), Fraction(1, 2)).apply(work.project)
        # 最後のクリップは後ろが空いているので伸ばせる
        after = ChangeSpeed((work.pictures[2].id,), Fraction(1, 2)).apply(work.project)
        assert _clip(after, work.pictures[2]).duration == 120

    def test_the_length_rounds_to_the_nearest_frame(self) -> None:
        # 壊れると、切り捨てで掛け直すたびに最後の絵が 1 フレームずつ欠けていく
        clip = Clip(timeline_start=0, duration=10, media_id=JumpCut().media.id)
        assert sped_duration(clip, Fraction(3)) == 3
        assert sped_duration(clip, Fraction(3, 2)) == 7
        assert sped_duration(clip, Fraction(1000)) == 1


def _with_scale_keys(work: JumpCut, keys: tuple[Keyframe, ...]) -> tuple[Project, Clip]:
    """3 本目の絵に、拡大率のキーフレームを打つ（強調の寄りを付けた形）"""
    transform = fixed_effect(TRANSFORM_EFFECT_KIND).with_param(
        "scale", AnimatedValue(keyframes=keys)
    )
    picture = replace(
        work.pictures[2],
        effects=(transform,),
        opacity=AnimatedValue(keyframes=(Keyframe(30, 0.5),)),
    )
    track = work.project.timeline.find_track(work.video.id)
    assert track is not None
    others = tuple(c for c in track.clips if c.id != picture.id)
    project = work.project.with_timeline(
        work.project.timeline.replace_track(track.with_clips((*others, picture)))
    )
    return project, picture


def _scale_keys(project: Project, clip: Clip) -> list[tuple[int, float]]:
    effect = next(e for e in _clip(project, clip).effects if e.kind == TRANSFORM_EFFECT_KIND)
    scale = effect.params["scale"]
    assert isinstance(scale, AnimatedValue)
    return [(key.frame, key.value) for key in scale.keyframes]


class TestKeyframes:
    def test_keeping_content_moves_the_points_with_the_content(self) -> None:
        # 壊れると、2 倍速にした後も寄りの点が元のフレームに残り、話の要所からずれるか、
        # 短くなった後ろに出て見えなくなる
        work = JumpCut()
        keys = (Keyframe(0, 100.0), Keyframe(30, 120.0), Keyframe(50, 100.0))
        project, picture = _with_scale_keys(work, keys)
        after = ChangeSpeed((picture.id,), Fraction(2)).apply(project)
        assert _scale_keys(after, picture) == [(0, 100.0), (15, 120.0), (25, 100.0)]
        # 不透明度や中身の値の点も同じだけ動く
        assert [key.frame for key in _clip(after, picture).opacity.keyframes] == [15]
        slower = ChangeSpeed((picture.id,), Fraction(1, 2)).apply(project)
        assert _scale_keys(slower, picture) == [(0, 100.0), (60, 120.0), (100, 100.0)]

    def test_points_that_meet_keep_the_later_value(self) -> None:
        # 壊れると、速くして同じフレームに重なった点で、寄り切る前の値が残って寄らない
        work = JumpCut()
        keys = (Keyframe(0, 100.0), Keyframe(30, 110.0), Keyframe(31, 130.0))
        project, picture = _with_scale_keys(work, keys)
        after = ChangeSpeed((picture.id,), Fraction(10)).apply(project)
        assert _scale_keys(after, picture) == [(0, 100.0), (3, 130.0)]

    def test_keeping_length_leaves_the_points(self) -> None:
        # 壊れると、設定パネルと同じ長さを保つ変え方で、点だけが動いて動きの時刻が変わる
        work = JumpCut()
        keys = (Keyframe(0, 100.0), Keyframe(30, 120.0))
        project, picture = _with_scale_keys(work, keys)
        after = ChangeSpeed((picture.id,), Fraction(2), keep_length=True).apply(project)
        assert _scale_keys(after, picture) == [(0, 100.0), (30, 120.0)]


class TestKeepLength:
    def test_the_length_stays_like_the_panel(self) -> None:
        # 壊れると、設定パネルの「再生速度」と同じ変え方を頼んだのに長さまで変わる
        work = JumpCut()
        after = ChangeSpeed((work.sounds[1].id,), Fraction(1, 2), keep_length=True).apply(
            work.project
        )
        for clip in (work.pictures[1], work.sounds[1]):
            sped = _clip(after, clip)
            assert (sped.speed, sped.duration) == (Fraction(1, 2), 60)


class TestRefusals:
    def test_a_locked_partner_is_refused(self) -> None:
        # 壊れると、ロックした音のトラックを飛ばして絵だけ速くなり、絵と音がずれていく
        work = JumpCut()
        project = SetTrackState(work.audio.id, locked=True).apply(work.project)
        with pytest.raises(ValueError, match="ロック"):
            ChangeSpeed((work.pictures[0].id,), Fraction(2)).apply(project)

    def test_a_text_clip_is_refused(self) -> None:
        # 壊れると、速度の効かないテキストに速度を入れて「変えた」と答え、何も変わらない
        work = JumpCut()
        text = Clip(
            timeline_start=300,
            duration=30,
            source=GeneratedSource(kind="text", params={"text": "見出し"}),
        )
        project = AddClip(work.video.id, text).apply(work.project)
        with pytest.raises(ValueError, match="素材の時間を持たない"):
            ChangeSpeed((text.id,), Fraction(2)).apply(project)

    def test_the_same_speed_changes_nothing(self) -> None:
        # 壊れると、何も変わらない頼みで空の 1 段が履歴に積まれ、AI の取り消しを止める
        work = JumpCut()
        assert ChangeSpeed((work.pictures[0].id,), Fraction(1)).apply(work.project) is work.project

    def test_a_missing_clip_is_refused(self) -> None:
        # 壊れると、消えたクリップを指した頼みが素の例外で落ち、AI が何を直せばよいか分からない
        work = JumpCut()
        missing = replace(work.pictures[0], id=Clip(timeline_start=0, duration=1).id)
        with pytest.raises(KeyError):
            ChangeSpeed((missing.id,), Fraction(2)).apply(work.project)
