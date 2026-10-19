"""キーフレームを入れたクリップを割っても、動きがタイムライン上で変わらないこと（利用者の報告）

キーはクリップの頭から数えたフレームで持つ 割ったときに後ろのクリップへキーをそのまま
写すと、後ろのクリップにも前と同じ位置付近に同じキーが入り、割っただけで動きが変わる
（割った位置だけずれて、同じフェードがもう 1 度起きる）

割る所をまたぐ区間は、形を変えない点を足して切る（前は最後のフレーム、後ろは頭）
それぞれの範囲の外に出たキーは捨てる 無音のカット（:class:`RippleCut`）も同じ決まり
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.core.commands import Document, RippleCut, SplitClip
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Interpolation,
    Keyframe,
    LayerMode,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects.sources import TEXT

START = 50
DURATION = 120


def _keys(*keys: Keyframe) -> AnimatedValue:
    return AnimatedValue(static=1.0, keyframes=keys)


def _project(opacity: AnimatedValue, **source: AnimatedValue) -> tuple[Project, Clip]:
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    text = TEXT.create()
    for name, value in source.items():
        text = text.with_param(name, value)
    clip = Clip(timeline_start=START, duration=DURATION, source=text, opacity=opacity)
    track = Track(TrackKind.MIXED, "レイヤー 1", (clip,))
    return base.with_timeline(replace(base.timeline, tracks=(track,))), clip


def _pieces(project: Project) -> tuple[Clip, ...]:
    return project.timeline.tracks[0].clips


def _opacity_at(project: Project, frame: int) -> float:
    """タイムラインの ``frame`` での不透明度（そこにあるクリップの値）"""
    clip = next(c for c in _pieces(project) if c.contains(frame))
    return clip.opacity.at(frame - clip.timeline_start)


FADE = _keys(Keyframe(0, 0.0), Keyframe(100, 1.0))


class TestSplit:
    def test_the_motion_stays_where_it_was_on_the_timeline(self) -> None:
        # 後ろへキーをそのまま写すと、後ろのクリップの頭からまたフェードが始まる
        project, clip = _project(FADE)
        split = SplitClip(clip.id, START + 50).apply(project)
        for frame in range(START, START + DURATION):
            assert _opacity_at(split, frame) == pytest.approx(clip.opacity.at(frame - START))

    def test_each_side_gets_a_key_at_the_cut_and_drops_the_rest(self) -> None:
        # 範囲の外に出たキーを残すと、描く所の無い点がグラフだけに残る
        project, clip = _project(FADE)
        before, after = _pieces(SplitClip(clip.id, START + 50).apply(project))
        assert [(k.frame, k.value) for k in before.opacity.keyframes] == [
            (0, 0.0),
            (49, pytest.approx(0.49)),
        ]
        assert [(k.frame, k.value) for k in after.opacity.keyframes] == [
            (0, pytest.approx(0.5)),
            (50, 1.0),
        ]

    def test_keys_only_after_the_cut_move_with_the_second_part(self) -> None:
        # 後ろのキーを割った位置から数え直さないと、後ろのクリップの動きが割った長さだけ遅れ、
        # クリップの外へ出て消える
        project, clip = _project(_keys(Keyframe(80, 0.2), Keyframe(110, 0.9)))
        before, after = _pieces(SplitClip(clip.id, START + 60).apply(project))
        assert not before.opacity.is_animated
        assert before.opacity.static == pytest.approx(0.2)
        assert [k.frame for k in after.opacity.keyframes] == [20, 50]

    def test_keys_only_before_the_cut_leave_the_second_part_still(self) -> None:
        # 前にしか無いキーを後ろへも写すと、後ろのクリップの頭でまた同じ動きが始まる
        project, clip = _project(_keys(Keyframe(0, 0.2), Keyframe(20, 0.9)))
        before, after = _pieces(SplitClip(clip.id, START + 60).apply(project))
        assert [k.frame for k in before.opacity.keyframes] == [0, 20]
        assert not after.opacity.is_animated
        assert after.opacity.static == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "interpolation",
        [Interpolation.EASE_IN, Interpolation.EASE_OUT, Interpolation.EASE_IN_OUT],
    )
    def test_an_eased_curve_keeps_its_shape(self, interpolation: Interpolation) -> None:
        # 割る所で曲線を切らずに直線の点を足すと、イージングの形が崩れる
        eased = _keys(Keyframe(10, 0.0, interpolation=interpolation), Keyframe(110, 1.0))
        project, clip = _project(eased)
        split = SplitClip(clip.id, START + 47).apply(project)
        for frame in range(START, START + DURATION):
            assert _opacity_at(split, frame) == pytest.approx(
                clip.opacity.at(frame - START), abs=1e-6
            )

    def test_effect_and_content_values_are_split_too(self) -> None:
        # 不透明度だけ直すと、文字の大きさや位置のキーは後ろのクリップでずれたまま
        size = _keys(Keyframe(0, 20.0), Keyframe(100, 120.0))
        project, clip = _project(FADE, size=size)
        _, after = _pieces(SplitClip(clip.id, START + 50).apply(project))
        assert after.source is not None
        moved = after.source.params["size"]
        assert isinstance(moved, AnimatedValue)
        assert [(k.frame, k.value) for k in moved.keyframes] == [
            (0, pytest.approx(70.0)),
            (50, 120.0),
        ]

    def test_it_can_be_undone(self) -> None:
        # 取り消しても元の 1 本に戻らないと、キーを入れたクリップを割り間違えたら動きを打ち直す
        project, clip = _project(FADE)
        document = Document(project)
        document.execute(SplitClip(clip.id, START + 50))
        document.undo()
        assert _pieces(document.project) == (clip,)


class TestRippleCut:
    def test_the_part_after_the_cut_keeps_its_motion(self) -> None:
        # 無音のカットで後ろに残った部分も、キーをそのまま写すと動きが削った長さだけ前へずれる
        project, clip = _project(FADE)
        cut = RippleCut(((START + 30, START + 40),)).apply(project)
        for frame in range(START, START + 30):
            assert _opacity_at(cut, frame) == pytest.approx(clip.opacity.at(frame - START))
        for frame in range(START + 30, START + DURATION - 10):
            assert _opacity_at(cut, frame) == pytest.approx(clip.opacity.at(frame + 10 - START))
