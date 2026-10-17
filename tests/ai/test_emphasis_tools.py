"""AI の強調の動きの道具（#331 B5） emphasize

偽のホストで、道具が人の操作と同じ命令（配置の欄の値の線・設定パネルの〔エフェクトを追加…〕の
ランダム移動）を通すこと、取り消し 1 回で戻ること、本人の設定を既定に使うこと、
範囲の外の動きと人の点を壊さないことを見る 描いた絵で確かめる試験は
tests/engine/test_emphasis_render.py、本物の窓と設定画面は tests/ui/test_emphasis_ui.py
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddClip, SetTrackState
from sashimono.core.commands.fixed import (
    TRANSFORM_EFFECT_KIND,
    VOLUME_EFFECT_KIND,
    fixed_effect,
)
from sashimono.core.emphasis import EmphasisStyle, fixed_transform
from sashimono.core.model import (
    AnimatedValue,
    AudioStreamInfo,
    Clip,
    ClipId,
    GeneratedSource,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from tests.ai.conftest import RATE_30, FakeHost


def _run(host: FakeHost, **arguments: Any) -> Any:
    operation = find_operation("emphasize")
    assert operation is not None
    return operation(host, arguments)


def _square(start: int, duration: int, *, transform: bool = True) -> Clip:
    source = GeneratedSource(kind="shape", params={"shape": "rect"})
    effects = (fixed_effect(TRANSFORM_EFFECT_KIND),) if transform else ()
    return Clip(timeline_start=start, duration=duration, source=source, effects=effects)


class _Work:
    """映像トラックに図形を 2 本（配置を持つ物と持たない古い物）、音声トラックに音を 1 本置く"""

    def __init__(self) -> None:
        sound = MediaItem(
            path=Path("C:/素材/効果音.wav"),
            duration=Fraction(10),
            audio_streams=(
                AudioStreamInfo(
                    index=0,
                    sample_rate=48000,
                    channels=2,
                    time_base=Fraction(1, 48000),
                    codec="pcm",
                ),
            ),
        )
        base = Project.create(ProjectSettings(frame_rate=RATE_30), media=(sound,))
        self.video = Track(kind=TrackKind.VIDEO, name="V1")
        self.audio = Track(kind=TrackKind.AUDIO, name="A1")
        project = base.with_timeline(replace(base.timeline, tracks=(self.video, self.audio)))
        self.square = _square(30, 120)
        self.old = _square(200, 60, transform=False)
        self.sound = Clip(
            timeline_start=0,
            duration=300,
            media_id=sound.id,
            effects=(fixed_effect(VOLUME_EFFECT_KIND),),
        )
        for track, clip in (
            (self.video, self.square),
            (self.video, self.old),
            (self.audio, self.sound),
        ):
            project = AddClip(track.id, clip).apply(project)
        self.project = project
        self.host = FakeHost(project)

    def clip(self, clip_id: ClipId) -> Clip:
        located = self.host.document.project.timeline.locate_clip(clip_id)
        assert located is not None
        return located[1]

    def transform(self, clip_id: ClipId) -> dict[str, AnimatedValue]:
        effect = fixed_transform(self.clip(clip_id))
        assert effect is not None
        return {k: v for k, v in effect.params.items() if isinstance(v, AnimatedValue)}


@pytest.fixture
def work() -> _Work:
    return _Work()


class TestDeclaration:
    def test_it_is_an_undoable_edit_that_sends_the_preferences(self) -> None:
        # 壊れると、取り消せる編集なのに毎回確かめが出るか、本人の設定の既定を結果に載せるのに
        # 送る物に書かず、設定画面と個人情報の扱いの一覧に無い物が外へ出る
        operation = find_operation("emphasize")
        assert operation is not None
        assert operation.safety is Safety.UNDOABLE
        assert operation.sends == {Outbound.PROJECT, Outbound.APP_SETTINGS}


class TestZoom:
    def test_zoom_rises_and_returns_in_one_undo(self, work: _Work) -> None:
        # 壊れると、AI が付けた寄りを 1 回の取り消しで戻せず、人が何度も戻すか手で点を消す
        result = _run(
            work.host, clip_id=str(work.square.id), kind="zoom", start_frame=60, end_frame=90
        )
        scale = work.transform(work.square.id)["scale"]
        # クリップの頭（30）から数えた 30〜60 フレームが範囲 既定は 250 ms（8 フレーム）で 120%
        assert scale.at(0) == 100.0
        assert scale.at(30) == pytest.approx(100.0)
        assert scale.at(45) == pytest.approx(120.0)
        assert scale.at(60) == pytest.approx(100.0)
        assert result["keyframes"] == [60, 68, 82, 90]
        assert result["scale_percent"] == 120.0
        work.host.document.undo()
        assert work.host.document.project == work.project

    def test_the_preference_is_the_default(self, work: _Work) -> None:
        # 壊れると、設定画面で選んだ寄る量や動き方が AI の強調の動きに効かない
        work.host.emphasis = EmphasisStyle(zoom_percent=150, ramp_ms=0, easing="linear")
        result = _run(
            work.host, clip_id=str(work.square.id), kind="zoom", start_frame=60, end_frame=90
        )
        scale = work.transform(work.square.id)["scale"]
        assert scale.at(31) == pytest.approx(150.0)
        assert (result["ramp_ms"], result["easing"]) == (0, "linear")

    def test_arguments_beat_the_preference(self, work: _Work) -> None:
        # 壊れると、AI が「少しだけ寄って」と値を渡しても本人の既定の量で寄る
        work.host.emphasis = EmphasisStyle(zoom_percent=150)
        _run(
            work.host,
            clip_id=str(work.square.id),
            kind="zoom",
            start_frame=60,
            end_frame=90,
            scale_percent=110,
        )
        assert work.transform(work.square.id)["scale"].at(45) == pytest.approx(110.0)

    def test_focus_moves_the_position_toward_the_point(self, work: _Work) -> None:
        # 壊れると、話者の顔（画面の右上）へ寄ったつもりが画面の中央へ寄り、顔が画面から外れる
        _run(
            work.host,
            clip_id=str(work.square.id),
            kind="zoom",
            start_frame=60,
            end_frame=90,
            scale_percent=150,
            focus_x=200,
            focus_y=100,
        )
        values = work.transform(work.square.id)
        # 止めておく点が動かない位置 pos + (focus - pos) x (1 - 1.5)
        assert values["pos_x"].at(45) == pytest.approx(-100.0)
        assert values["pos_y"].at(45) == pytest.approx(-50.0)

    def test_an_old_clip_without_a_transform_still_moves(self, work: _Work) -> None:
        # 壊れると、配置の欄を持たない前の版のクリップに寄りが付かないか、取り消しが 2 段になる
        _run(work.host, clip_id=str(work.old.id), kind="zoom", start_frame=210, end_frame=240)
        assert work.transform(work.old.id)["scale"].at(25) == pytest.approx(120.0)
        work.host.document.undo()
        assert work.host.document.project == work.project

    def test_keep_stays_zoomed_to_the_end(self, work: _Work) -> None:
        # 壊れると、「ここから寄ったままにして」がクリップの終わりの前で戻る
        result = _run(
            work.host, clip_id=str(work.square.id), kind="zoom", start_frame=100, keep=True
        )
        scale = work.transform(work.square.id)["scale"]
        assert scale.at(119) == pytest.approx(120.0)
        assert result["end_frame"] == work.square.timeline_end

    def test_existing_animation_outside_is_kept(self, work: _Work) -> None:
        # 壊れると、人が付けたゆっくりした拡大が、強調の範囲の外まで崩れる
        animated = AnimatedValue(static=100.0, keyframes=(Keyframe(0, 100.0), Keyframe(119, 200.0)))
        transform = fixed_transform(work.square)
        assert transform is not None
        changed = replace(
            work.square,
            id=ClipId("animated"),
            timeline_start=400,
            effects=(transform.with_param("scale", animated),),
        )
        host = FakeHost(AddClip(work.video.id, changed).apply(work.project))
        _run(host, clip_id="animated", kind="zoom", start_frame=430, end_frame=460)
        located = host.document.project.timeline.locate_clip(ClipId("animated"))
        assert located is not None
        effect = fixed_transform(located[1])
        assert effect is not None
        scale = effect.params["scale"]
        assert isinstance(scale, AnimatedValue)
        for frame in [*range(0, 31), *range(60, 120)]:
            assert scale.at(frame) == pytest.approx(animated.at(frame)), frame


class TestOtherKinds:
    def test_pan_up_raises_y(self, work: _Work) -> None:
        # 壊れると、「上へずらして」で絵が下へ動く（Y は上が正）
        work.host.emphasis = EmphasisStyle(pan_px=90)
        result = _run(
            work.host,
            clip_id=str(work.square.id),
            kind="pan",
            start_frame=60,
            end_frame=90,
            direction="up",
        )
        values = work.transform(work.square.id)
        assert values["pos_y"].at(45) == pytest.approx(90.0)
        assert not values["pos_x"].keyframes
        assert result["offset_px"] == {"x": 0.0, "y": 90.0}

    def test_pop_swells_and_returns(self, work: _Work) -> None:
        # 壊れると、一瞬ふくらむはずのポップが止まったまま残るか、ふくらまない
        _run(work.host, clip_id=str(work.square.id), kind="pop", start_frame=60, end_frame=75)
        scale = work.transform(work.square.id)["scale"]
        # 15 フレームの範囲なので、動き切るのは半分の 7 フレーム目（既定の 8 フレームより短い）
        assert scale.at(37) == pytest.approx(115.0)
        assert scale.at(45) == pytest.approx(100.0)

    def test_shake_adds_a_random_move_that_is_quiet_outside(self, work: _Work) -> None:
        # 壊れると、揺れを付けたクリップが範囲の外でも揺れ続けるか、人が設定パネルで揺れの幅を
        # 直せない（ランダム移動の欄で揺らしていない）
        _run(
            work.host,
            clip_id=str(work.square.id),
            kind="shake",
            start_frame=60,
            end_frame=90,
            amplitude_px=20,
        )
        clip = work.clip(work.square.id)
        (shake,) = [e for e in clip.effects if e.kind == "random_move"]
        width = shake.params["range_x"]
        assert isinstance(width, AnimatedValue)
        assert width.at(0) == 0.0
        assert width.at(45) == pytest.approx(20.0)
        assert width.at(100) == 0.0
        assert shake.params["range_y"] == width
        # 足したエフェクトは置く前（固定の配置の前）に入る 画面の〔エフェクトを追加…〕と同じ
        kinds = [e.kind for e in clip.effects]
        assert kinds.index("random_move") < kinds.index(TRANSFORM_EFFECT_KIND)
        work.host.document.undo()
        assert work.host.document.project == work.project


class TestRefusals:
    def test_a_sound_clip_is_refused(self, work: _Work) -> None:
        # 壊れると、音だけのクリップに点が打たれ、何も動かないのに AI は付けたと思い込む
        with pytest.raises(ToolError, match="絵を描かない"):
            _run(work.host, clip_id=str(work.sound.id), kind="zoom", start_frame=0, end_frame=30)

    def test_a_locked_track_is_refused(self, work: _Work) -> None:
        # 壊れると、人が触らせないためにロックしたトラックの絵が AI に動かされる
        work.host.apply_commands([SetTrackState(work.video.id, locked=True)], "ロック")
        with pytest.raises(ToolError, match="ロック"):
            _run(work.host, clip_id=str(work.square.id), kind="zoom", start_frame=60, end_frame=90)

    @pytest.mark.parametrize(
        "arguments",
        [
            {"start_frame": 0, "end_frame": 60},
            {"start_frame": 60, "end_frame": 200},
            {"start_frame": 90, "end_frame": 60},
            {"start_frame": 60},
            {"start_frame": 60, "end_frame": 90, "keep": True},
            {"start_frame": 60, "end_frame": 90, "easing": "wobble"},
            {"start_frame": 60, "end_frame": 90, "scale_percent": 100},
            {"start_frame": 60, "end_frame": 90, "ramp_ms": -1},
        ],
    )
    def test_bad_arguments_are_refused_without_change(
        self, work: _Work, arguments: dict[str, Any]
    ) -> None:
        # 壊れると、クリップの外や逆順の範囲が端へ寄せられ、AI が狙った言葉とずれた所で動く
        # どれも断るだけで作品を変えない
        with pytest.raises(ToolError):
            _run(work.host, clip_id=str(work.square.id), kind="zoom", **arguments)
        assert work.host.document.project == work.project

    def test_unknown_kind_and_keep_for_shake_are_refused(self, work: _Work) -> None:
        # 壊れると、綴りの違う種類が別の動きになるか、揺れがクリップの終わりまで続く
        with pytest.raises(ToolError):
            _run(work.host, clip_id=str(work.square.id), kind="spin", start_frame=60, end_frame=90)
        with pytest.raises(ToolError):
            _run(work.host, clip_id=str(work.square.id), kind="shake", start_frame=60, keep=True)

    def test_pan_needs_one_way_to_say_how_far(self, work: _Work) -> None:
        # 壊れると、向きと量の両方を渡したときにどちらかが黙って捨てられる
        with pytest.raises(ToolError):
            _run(
                work.host,
                clip_id=str(work.square.id),
                kind="pan",
                start_frame=60,
                end_frame=90,
                direction="up",
                offset_x=10,
            )
        with pytest.raises(ToolError):
            _run(work.host, clip_id=str(work.square.id), kind="pan", start_frame=60, end_frame=90)

    def test_going_past_the_field_range_is_refused(self, work: _Work) -> None:
        # 壊れると、拡大率が欄の上限（800%）を超えた所で止まり、寄る先がずれる
        transform = fixed_transform(work.square)
        assert transform is not None
        big = replace(work.square, effects=(transform.with_param("scale", AnimatedValue(700.0)),))
        project = AddClip(work.video.id, replace(big, id=ClipId("big"), timeline_start=400)).apply(
            work.project
        )
        host = FakeHost(project)
        with pytest.raises(ToolError, match="範囲"):
            _run(host, clip_id="big", kind="zoom", start_frame=410, end_frame=440)
        assert host.document.project == project

    def test_a_keyframe_inside_the_range_is_refused(self, work: _Work) -> None:
        # 壊れると、人が範囲の中に打った点が消えるか、強調と混ざってどちらの動きでもなくなる
        transform = fixed_transform(work.square)
        assert transform is not None
        animated = AnimatedValue(static=100.0, keyframes=(Keyframe(45, 130.0),))
        keyed = replace(
            work.square,
            id=ClipId("keyed"),
            timeline_start=400,
            effects=(transform.with_param("scale", animated),),
        )
        project = AddClip(work.video.id, keyed).apply(work.project)
        host = FakeHost(project)
        with pytest.raises(ToolError, match="キーフレーム"):
            _run(host, clip_id="keyed", kind="zoom", start_frame=430, end_frame=460)
        assert host.document.project == project
