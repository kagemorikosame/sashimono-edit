"""音の山をリミッターで抑えて音量を基準まで上げる（#414）

通しの確かめで、一瞬の山が天井に当たって基準に 3.8〜6 LU 届かなかった 山だけを抑えれば上げられる
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sashimono.ai.fixes import ASK, AUTO, fix_for
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddEffect, SetEffectEnabled, SetParam
from sashimono.core.commands.effects import ParamPath
from sashimono.core.commands.fixed import VOLUME_EFFECT_KIND, with_fixed_items
from sashimono.core.mixing import MAX_LIMITER_DB
from sashimono.core.model import AnimatedValue, Clip, Keyframe, MediaItem, Transcript
from sashimono.effects import registry
from sashimono.effects.audio import LIMITER_KIND, AudioContext
from sashimono.effects.limiter import limit
from sashimono.engine.checks.report import HIGH, MEDIUM
from sashimono.engine.checks.scan import LOUDNESS
from tests.ai.test_mix_tools import _measured, _Work

RATE = 48_000


def _run(work: _Work, **arguments: Any) -> Any:
    operation = find_operation("normalize_loudness")
    assert operation is not None
    return operation(work.host, {"wait_seconds": 0, **arguments})


@pytest.fixture
def work(transcript: Transcript, video_media: MediaItem) -> _Work:
    return _Work(transcript, video_media)


def _kinds(clip: Clip) -> list[str]:
    return [effect.kind for effect in clip.effects]


def _tone(seconds: float = 1.0, level: float = 0.3) -> np.ndarray:
    times = np.arange(int(RATE * seconds)) / RATE
    mono = (level * np.sin(2 * np.pi * 220 * times)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


class TestTheEffect:
    def test_no_sample_goes_over_the_ceiling(self) -> None:
        """壊れると、抑えたつもりの山が上限を越え、書き出しで音が割れる"""
        samples = _tone()
        samples[10_000:10_050] *= 4.0
        out = limit(samples, {"ceiling": -1.0, "release": 150}, AudioContext(0, RATE, RATE))
        assert float(np.abs(out).max()) <= 10 ** (-1 / 20) + 1e-6

    def test_quiet_sound_is_left_alone(self) -> None:
        """壊れると、上限より小さい話し声まで小さくなる"""
        samples = _tone(level=0.3)
        out = limit(samples, {"ceiling": -1.0, "release": 150}, AudioContext(0, RATE, RATE))
        assert np.array_equal(out, samples)

    def test_the_gain_comes_back_after_the_peak(self) -> None:
        """壊れると、一瞬の山の後ずっと小さいまま残り、話し声が沈む"""
        samples = _tone()
        samples[10_000:10_050] *= 4.0
        out = limit(samples, {"ceiling": -1.0, "release": 50}, AudioContext(0, RATE, RATE))
        tail = slice(40_000, 48_000)
        assert float(np.abs(out[tail]).max()) == pytest.approx(0.3, abs=1e-3)

    def test_the_cut_of_the_blocks_does_not_change_the_sound(self) -> None:
        """壊れると、プレビューと書き出しで区切り方が違うだけで音が変わり、区切りに段が出る"""
        samples = _tone()
        samples[10_000:10_050] *= 4.0
        values = {"ceiling": -1.0, "release": 150}
        whole = limit(samples, values, AudioContext(0, RATE, RATE))
        later = limit(samples, values, AudioContext(0, RATE, RATE, keep_from=20_000))
        assert np.allclose(later[20_000:], whole[20_000:])

    def test_it_is_placed_after_the_volume(self, work: _Work) -> None:
        """前に積むと、後から掛かる音量の欄で山が上限を越える"""
        clip = with_fixed_items(work.voice, sound=True)
        effect = registry.require(LIMITER_KIND).create()
        project = AddEffect(clip.id, effect).apply(work.host.document.project)
        placed = next(c for t in project.timeline.tracks for c in t.clips if c.id == clip.id)
        assert placed.effects[-1].kind == LIMITER_KIND
        assert VOLUME_EFFECT_KIND in _kinds(placed)


class TestNormalize:
    def test_the_peak_is_limited_and_the_target_reached(self, work: _Work) -> None:
        """通しの確かめの形 -18 LUFS で山が -1 dBFS 前は 0 dB で止まって基準に届かなかった"""
        _run(work)
        _measured(work.host, -18.0, -1.0)
        result = _run(work)
        assert result["reached"] is True
        assert result["gain_db"] == pytest.approx(4.0)
        assert result["limiter_db"] == pytest.approx(4.0)
        assert result["expected_peak_dbfs"] == -1.0
        for clip in (work.clip(work.voice.id), work.clip(work.bgm.id)):
            assert _kinds(clip)[-1] == LIMITER_KIND

    def test_the_limiting_is_capped(self, work: _Work) -> None:
        """壊れると、山を何 dB でも潰して、話し声の立ち上がりまで平たくなる"""
        _run(work)
        _measured(work.host, -24.0, -1.0)
        result = _run(work)
        assert result["gain_db"] == pytest.approx(MAX_LIMITER_DB)
        assert result["reached"] is False
        assert result["limited_by"] == ["peak"]
        assert "リミッターで" in result["note"]

    def test_a_second_call_does_not_stack(self, work: _Work) -> None:
        """抑えた後の音をまた抑えると、上限の 6 dB を越えて重なる"""
        _run(work)
        _measured(work.host, -24.0, -1.0)
        _run(work)
        _run(work)
        _measured(work.host, -18.5, -1.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert result["reached"] is False
        limiters = [k for k in _kinds(work.clip(work.voice.id)) if k == LIMITER_KIND]
        assert limiters == [LIMITER_KIND]

    def test_a_new_clip_does_not_reopen_the_limiting(self, work: _Work) -> None:
        """抑えた作品へクリップを足して呼び直しても、前のクリップを重ねて潰さない（Codex の指摘）"""
        effect = registry.require(LIMITER_KIND).create()
        work.host.apply_commands([AddEffect(work.voice.id, effect)], "前のリミッター")
        _run(work)
        _measured(work.host, -18.0, -1.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert result["reached"] is False
        assert LIMITER_KIND not in _kinds(work.clip(work.bgm.id))

    def test_switched_off_keeps_the_old_way(self, work: _Work) -> None:
        """設定で切った人の作品に、黙ってリミッターを積まない"""
        work.host.limiter = False
        _run(work)
        _measured(work.host, -18.0, -1.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert LIMITER_KIND not in _kinds(work.clip(work.voice.id))

    def test_a_moving_ceiling_is_not_overwritten(self, work: _Work) -> None:
        """人がキーフレームで動かした上限を、1 つの値で黙って消さない

        切ってあるリミッターは効いていないので入れ直す側 そのとき動きを潰さない
        """
        effect = registry.require(LIMITER_KIND).create()
        moving = AnimatedValue(-1.0, (Keyframe(0, -3.0), Keyframe(30, -1.0)))
        work.host.apply_commands(
            [
                AddEffect(work.voice.id, effect),
                SetParam(ParamPath.of_effect(work.voice.id, effect.id, "ceiling"), moving),
                SetEffectEnabled(work.voice.id, effect.id, False),
            ],
            "人のリミッター",
        )
        _run(work)
        _measured(work.host, -18.0, -1.0)
        with pytest.raises(ToolError, match="キーフレーム"):
            _run(work)


class TestCheckOutput:
    @pytest.mark.parametrize("severity", [HIGH, MEDIUM])
    def test_a_limitable_work_is_fixed_by_the_tool(self, severity: str) -> None:
        """壊れると、リミッターで上げられる作品まで人に尋ね、通しの編集が止まる"""
        measure = {"lufs": -20.3, "target_lufs": -14.0, "difference_lu": -6.3}
        fix = fix_for(LOUDNESS, severity, measure, peak_db=-1.0, limiter=True)
        assert fix.action == AUTO
        assert fix.tools == ("normalize_loudness",)

    def test_after_limiting_the_person_is_asked(self) -> None:
        """もう抑えた後なら、同じ道具を勧め続けて直す回数を使い切らない"""
        measure = {"lufs": -20.3, "target_lufs": -14.0, "difference_lu": -6.3}
        assert fix_for(LOUDNESS, HIGH, measure, peak_db=-1.0, limiter=False).action == ASK

    def test_the_check_result_offers_the_limiter(self, work: _Work) -> None:
        """壊れると、表は auto でも check_output が尋ねる側を返し、AI が人を待たせる"""
        check = find_operation("check_output")
        assert check is not None
        check(work.host, {"wait_seconds": 0})
        _measured(work.host, -20.3, -1.0)
        result: Any = check(work.host, {"wait_seconds": 0})
        rows = [row for row in result["problems"] if row["kind"] == LOUDNESS]
        assert rows[0]["fix"]["action"] == AUTO
