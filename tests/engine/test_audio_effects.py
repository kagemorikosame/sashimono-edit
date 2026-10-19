"""音を加工するエフェクト（AviUtl の音声フィルタ 3 種）

項目名は AviUtl2 に音声ファイルを置いてフィルタを積み、エイリアスを作らせて
読み取った AviUtl2 v2.1.6a の音声フィルタは 音量フェード・モノラル化・音量調整
の **3 つだけ**（音声波形表示 は音ではなく絵なので映像の側）
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sashimono.core.model import (
    Clip,
    Effect,
    Interpolation,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.audio import AudioContext
from sashimono.effects.spec import TrackSpec

RATE = 48000


def _stereo(count: int, left: float = 1.0, right: float = 1.0) -> np.ndarray:
    return np.stack(
        [np.full(count, left, dtype=np.float32), np.full(count, right, dtype=np.float32)],
        axis=1,
    )


def _run(
    kind: str, samples: np.ndarray, *, offset: int = 0, duration: int = 0, **values: float
) -> np.ndarray:
    """音のエフェクトを 1 つ掛ける 値は既定値に ``values`` を重ねる"""
    definition = registry.require(kind)
    assert definition.audio_process is not None, f"{kind}: 音の処理が無い"
    resolved: dict[str, float] = {}
    for spec in definition.parameters:
        assert isinstance(spec, TrackSpec)
        resolved[spec.name] = float(values.get(spec.name, spec.default))
    out: np.ndarray = definition.audio_process(
        samples,
        resolved,
        AudioContext(offset=offset, sample_rate=RATE, duration=duration or len(samples)),
    )
    return out


class TestVolume:
    def test_it_scales_the_level(self) -> None:
        # 音量 50 で半分にならないと、音量調整で BGM を下げても声の後ろで同じ大きさのまま鳴る
        out = _run("audio_volume", _stereo(4), volume=50.0)
        assert out[0, 0] == pytest.approx(0.5)

    def test_the_default_changes_nothing(self) -> None:
        # 音量 100 が元のまま ここが違うと、置いただけで音量が変わる
        plain = _stereo(4)
        assert np.allclose(_run("audio_volume", plain), plain)

    def test_panning_right_quiets_the_left(self) -> None:
        """左右は**絞る側だけ**を動かす

        反対側を持ち上げると、真ん中へ寄せたときに音が大きくなる
        """
        out = _run("audio_volume", _stereo(4), pan=100.0)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(1.0)

    def test_panning_left_quiets_the_right(self) -> None:
        # 左へ振ったときに右が絞られないと、パンを -100 にしても右のスピーカーから音が出続ける
        out = _run("audio_volume", _stereo(4), pan=-100.0)
        assert out[0, 0] == pytest.approx(1.0)
        assert out[0, 1] == pytest.approx(0.0)


class TestFade:
    def test_it_rises_over_the_in_time(self) -> None:
        # イン 1 秒 先頭が 0 で 1 秒後に 1
        out = _run("audio_fade", _stereo(RATE), duration=RATE * 4, fade_in=1.0)
        assert out[0, 0] == pytest.approx(0.0, abs=0.01)
        assert out[RATE // 2, 0] == pytest.approx(0.5, abs=0.02)
        assert out[-1, 0] == pytest.approx(1.0, abs=0.01)

    def test_it_falls_before_the_end(self) -> None:
        # アウト 1 秒 クリップの終わりへ向かって 0 へ
        total = RATE * 2
        out = _run("audio_fade", _stereo(total), duration=total, fade_out=1.0)
        assert out[0, 0] == pytest.approx(1.0, abs=0.01)
        assert out[-1, 0] == pytest.approx(0.0, abs=0.01)

    def test_it_keeps_shaping_inside_a_block(self) -> None:
        """塊の**中でも**サンプルごとに変わる

        塊の先頭の値だけで掛けると、塊の境目で音が階段状に変わる
        """
        out = _run("audio_fade", _stereo(RATE), duration=RATE * 4, fade_in=1.0)
        assert len(np.unique(np.round(out[:, 0], 3))) > 100

    def test_a_block_in_the_middle_knows_where_it_is(self) -> None:
        # 途中の塊を渡されても、クリップ先頭からの位置で決まる
        out = _run("audio_fade", _stereo(100), offset=RATE, duration=RATE * 4, fade_in=1.0)
        assert out[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_a_long_clip_keeps_the_fade_smooth(self) -> None:
        """長いクリップでもフェードが**階段状にならない**

        位置を float32 で持つと 1677 万サンプル（48 kHz で 6 分弱）を超えた所から
        1 サンプル単位を表せなくなり、隣り合うサンプルの音量が同じ値になる
        （5 分を超える曲では普通に起きる）
        """
        total = 30_000_000
        out = _run("audio_fade", _stereo(4), offset=total - 10_000, duration=total, fade_out=1.0)
        assert len(np.unique(out[:, 0])) == 4, "隣り合うサンプルの音量が同じになっている"

    def test_no_fade_changes_nothing(self) -> None:
        # イン・アウト 0 秒で音が変わると、音量フェードを積んだだけで
        # クリップの音が変わって書き出される
        plain = _stereo(8)
        assert np.allclose(_run("audio_fade", plain, duration=8), plain)


class TestMonaural:
    def test_zero_keeps_the_stereo(self) -> None:
        """比率 0 が**元のまま** 逆に読むと、既定のままでステレオが潰れる"""
        plain = _stereo(4, left=1.0, right=-1.0)
        assert np.allclose(_run("audio_monaural", plain), plain)

    def test_full_ratio_makes_both_sides_the_same(self) -> None:
        # 比率 100 で左右がそろわないと、モノラル化を掛けても片側だけの録音が片耳からしか聞こえない
        out = _run("audio_monaural", _stereo(4, left=1.0, right=-1.0), ratio=100.0)
        assert out[0, 0] == pytest.approx(out[0, 1])

    def test_half_way_is_between(self) -> None:
        # 比率 50 で半分だけ寄せないと、比率を少し上げたつもりで広がりが一気に消えるか変わらない
        out = _run("audio_monaural", _stereo(4, left=1.0, right=0.0), ratio=50.0)
        assert out[0, 0] == pytest.approx(0.75)
        assert out[0, 1] == pytest.approx(0.25)


class TestTheMixerAppliesThem:
    """積んだ音のエフェクトが、実際に混ぜるときに掛かる"""

    def _project(self, *effects: Effect) -> Project:
        project = Project.create(
            ProjectSettings(width=64, height=64, frame_rate=FrameRate(30), sample_rate=RATE)
        )
        track = Track(
            kind=TrackKind.AUDIO,
            clips=(Clip(timeline_start=0, duration=30, effects=effects),),
        )
        return project.with_timeline(project.timeline.__class__(rate=project.rate, tracks=(track,)))

    def test_a_video_effect_on_an_audio_clip_is_skipped(self) -> None:
        """映像のエフェクトは音の側で飛ばす

        AviUtl は音声オブジェクトにも映像フィルタを積めるので、
        種類を見ずに呼ぶと、シェーダしか持たないものを呼んで落ちる
        """
        from sashimono.engine.audio.mixer import _apply_effects

        clip = Clip(timeline_start=0, duration=30, effects=(registry.require("blur").create(),))
        samples = _stereo(16)
        assert np.allclose(_apply_effects(clip, samples, 0, RATE, 16, FrameRate(30)), samples)

    def test_the_effects_run_in_the_order_they_are_stacked(self) -> None:
        """置いた順に掛かる

        同じエフェクトを 2 つ積むだけでは「両方走った」ことしか分からない
        左右へ振ってからモノラル化すると両側が 0.5 になり、
        順番が逆なら左が 0 のまま残る 見分けのつく組にする
        """
        from sashimono.engine.audio.mixer import _apply_effects

        pan = registry.require("audio_volume").create(pan=100.0)
        mono = registry.require("audio_monaural").create(ratio=100.0)

        first = _apply_effects(
            Clip(timeline_start=0, duration=30, effects=(pan, mono)),
            _stereo(4),
            0,
            RATE,
            4,
            FrameRate(30),
        )
        assert first[0, 0] == pytest.approx(0.5)
        assert first[0, 1] == pytest.approx(0.5)

        second = _apply_effects(
            Clip(timeline_start=0, duration=30, effects=(mono, pan)),
            _stereo(4),
            0,
            RATE,
            4,
            FrameRate(30),
        )
        assert second[0, 0] == pytest.approx(0.0)
        assert second[0, 1] == pytest.approx(1.0)

    def test_a_broken_value_falls_back_to_the_default(self) -> None:
        """読めない値は**既定値**へ戻す

        0 にすると、既定が 100 の音量ではクリップが丸ごと無音になる
        （古いファイルや壊れたファイルに文字が入っていることがある）
        """
        from dataclasses import replace as _replace

        from sashimono.engine.audio.mixer import _apply_effects

        broken = registry.require("audio_volume").create()
        broken = _replace(broken, params={**broken.params, "volume": "でたらめ"})
        out = _apply_effects(
            Clip(timeline_start=0, duration=30, effects=(broken,)),
            _stereo(4),
            0,
            RATE,
            4,
            FrameRate(30),
        )
        assert out[0, 0] == pytest.approx(1.0), "無音になっている"

    def test_an_animated_value_changes_at_the_frame_boundary(self) -> None:
        """動く値は**映像のフレームの切れ目**で変わる

        塊の先頭で 1 度だけ解くと、プレビューの細かい塊がフレームを
        またいだときに音量の変わる時刻がずれ、書き出しと合わなくなる
        """
        from sashimono.core.model import AnimatedValue, Keyframe
        from sashimono.engine.audio.mixer import _apply_effects

        fading = registry.require("audio_volume").create(
            volume=AnimatedValue(
                keyframes=(
                    Keyframe(frame=0, value=100.0, interpolation=Interpolation.HOLD),
                    Keyframe(frame=1, value=0.0),
                )
            )
        )
        per_frame = RATE // 30
        out = _apply_effects(
            Clip(timeline_start=0, duration=30, effects=(fading,)),
            _stereo(per_frame * 2),
            0,
            RATE,
            per_frame * 2,
            FrameRate(30),
        )
        assert out[0, 0] == pytest.approx(1.0), "1 フレーム目から下がっている"
        assert out[per_frame + 1, 0] == pytest.approx(0.0), "2 フレーム目で下がっていない"

    def test_the_boundary_is_exact_at_a_fractional_rate(self) -> None:
        """29.97 fps でも切れ目が**1 サンプルもずれない**

        1 フレームを小数のサンプル数で持って割ると、フレーム 1 は
        切り捨てて 1601 サンプル目から始まるのに 1601 / 1601.6 は 0 になり、
        先頭の 1 サンプルだけ前のフレームの音量で鳴る
        """
        from sashimono.core.model import AnimatedValue, Keyframe
        from sashimono.engine.audio.mixer import _apply_effects, _frame_to_sample

        rate = FrameRate(30000, 1001)
        fading = registry.require("audio_volume").create(
            volume=AnimatedValue(
                keyframes=(
                    Keyframe(frame=0, value=100.0, interpolation=Interpolation.HOLD),
                    Keyframe(frame=1, value=0.0),
                )
            )
        )
        boundary = _frame_to_sample(1, rate, RATE)
        assert boundary == 1601, "測り直す ここは 48 kHz / 29.97 fps の実際の値"
        out = _apply_effects(
            Clip(timeline_start=0, duration=30, effects=(fading,)),
            _stereo(boundary + 2),
            0,
            RATE,
            boundary + 2,
            rate,
        )
        assert out[boundary - 1, 0] == pytest.approx(1.0), "切れ目の手前で下がっている"
        assert out[boundary, 0] == pytest.approx(0.0), "切れ目の 1 サンプル目が前のフレームのまま"

    def test_the_boundary_follows_the_timeline_grid(self) -> None:
        """切れ目は**タイムラインの升目**で決める クリップの先頭から数え直さない

        29.97 fps の升目は 1601 と 1602 のサンプルが混ざる（1 フレームは
        1601.6 サンプル） クリップの先頭から数え直すと、置いた場所によって
        最初の升の幅が 1 サンプル狂い、音の変わる所が絵の切り替わる所とずれる
        タイムラインのフレーム 4 は 6406 から、フレーム 5 は 8008 から始まるので、
        ここに置いたクリップの最初の升は 1601 ではなく 1602 サンプル
        """
        from sashimono.core.model import AnimatedValue, Keyframe
        from sashimono.engine.audio.mixer import _apply_effects, _frame_to_sample

        rate = FrameRate(30000, 1001)
        start = 4
        width = _frame_to_sample(start + 1, rate, RATE) - _frame_to_sample(start, rate, RATE)
        assert width == 1602, "測り直す ここは 48 kHz / 29.97 fps の実際の値"

        fading = registry.require("audio_volume").create(
            volume=AnimatedValue(
                keyframes=(
                    Keyframe(frame=0, value=100.0, interpolation=Interpolation.HOLD),
                    Keyframe(frame=1, value=0.0),
                )
            )
        )
        out = _apply_effects(
            Clip(timeline_start=start, duration=30, effects=(fading,)),
            _stereo(width + 2),
            0,
            RATE,
            width + 2,
            rate,
        )
        assert out[width - 1, 0] == pytest.approx(1.0), "升の終わりより早く下がっている"
        assert out[width, 0] == pytest.approx(0.0), "升をまたいでも下がっていない"

    def test_a_disabled_effect_is_skipped(self) -> None:
        # 無効にした音のエフェクトが掛かると、音量 0 のチェックを外しても
        # クリップが無音のまま書き出される
        from sashimono.engine.audio.mixer import _apply_effects

        muted = replace(registry.require("audio_volume").create(volume=0.0), enabled=False)
        clip = Clip(timeline_start=0, duration=30, effects=(muted,))
        samples = _stereo(4)
        assert np.allclose(_apply_effects(clip, samples, 0, RATE, 4, FrameRate(30)), samples)


def test_the_audio_effects_only_take_numbers() -> None:
    """音のエフェクトの項目は :class:`TrackSpec` だけ

    ミキサは数にならない仕様を values へ入れない（0 を渡すと既定値と違う値で
    走るため） 数以外の項目を足すなら、先にミキサの読み方を決める
    """
    for definition in registry.all():
        if definition.audio_process is None:
            continue
        for spec in definition.parameters:
            assert isinstance(spec, TrackSpec), f"{definition.kind}: {spec.name} が数でない"
