"""音の山をリミッターで抑えて音量を基準まで上げる（#414）

通しの確かめで、一瞬の山が天井に当たって基準に 3.8〜6 LU 届かなかった 山だけを抑えれば上げられる
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.ai import loudness_plan
from sashimono.ai.auto_fix import (
    FITTED_CEILING,
    clips_overlap,
    limited_ceiling,
    limiter_ceiling,
    limiter_commands,
    limiters_match,
    loudness_predictor,
    raise_cap,
    redoable_limiters,
)
from sashimono.ai.fixes import ASK, AUTO, fix_for
from sashimono.ai.host import ToolError
from sashimono.ai.loudness_plan import (
    LoudnessPlan,
    PlanAbandonedError,
    PlanJob,
    Stopped,
    plan_loudness,
)
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddEffect, SetEffectEnabled, SetParam
from sashimono.core.commands.effects import ParamPath
from sashimono.core.commands.fixed import (
    VOLUME_EFFECT_KIND,
    fixed_slot,
    loose_slot,
    with_fixed_items,
)
from sashimono.core.io.serialize import effect_from_json, effect_to_json
from sashimono.core.mixing import MAX_LIMITER_DB, LoudnessGain, gain_commands, loudness_gain
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Interpolation,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.audio import LIMITER_KIND, AudioContext
from sashimono.effects.limiter import limit, limiter_history
from sashimono.engine.audio import AudioMixer
from sashimono.engine.audio.mixer import HISTORY_WINDOW
from sashimono.engine.checks.job import CheckRequest, _check_timeline, _Progress
from sashimono.engine.checks.loudness import Dynamics, Headroom, LoudnessLane, LoudnessMeter
from sashimono.engine.checks.report import HIGH, MEDIUM, CheckReport
from sashimono.engine.checks.scan import LOUDNESS, CheckLimits
from sashimono.engine.checks.sound_lanes import LaneMeters
from sashimono.engine.decode.probe import probe_media
from tests.ai.conftest import FakeHost
from tests.ai.pending import settle
from tests.ai.test_mix_tools import _measured, _Work
from tests.check_media import tone, write_tone
from tests.waiting import GUARD_SECONDS, wait_until

RATE = 48_000


def _run(work: _Work, **arguments: Any) -> Any:
    operation = find_operation("normalize_loudness")
    assert operation is not None
    # 抑えた後の見積もりは裏で回る（#497） ブリッジと同じく待ってから辿る
    return settle(operation(work.host, {"wait_seconds": 0, **arguments}))


@pytest.fixture
def work(transcript: Transcript, video_media: MediaItem) -> _Work:
    return _Work(transcript, video_media)


def _kinds(clip: Clip) -> list[str]:
    return [effect.kind for effect in clip.effects]


def _track_volume(work: _Work, db: float) -> None:
    """声のトラックの音量を ``db`` にした作品へ替える（リミッターの後に掛かる音量）"""
    tracks = tuple(
        replace(track, volume_db=db) if track.id == work.voice_track.id else track
        for track in work.project.timeline.tracks
    )
    work.host = FakeHost(work.project.with_timeline(replace(work.project.timeline, tracks=tracks)))


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

    def test_a_slice_with_history_matches_the_whole(self) -> None:
        """区切りの直前の山の戻りを、前の音として読んだ分から引き継ぐ

        同じ配列を渡すだけでは、前の音を読まずに後の区切りで絞りを忘れる壊れ方を見落とす
        （PR #451 の CodeRabbit の指摘）
        """
        samples = _tone()
        samples[29_900:29_950] *= 4.0
        values = {"ceiling": -1.0, "release": 50}
        whole = limit(samples, values, AudioContext(0, RATE, RATE))
        history = round(limiter_history(values) * RATE)
        begin = 30_000 - history
        later = limit(samples[begin:], values, AudioContext(begin, RATE, RATE, keep_from=history))
        assert np.allclose(later[history:], whole[30_000:], atol=1e-6)
        # 区切りの頭はまだ絞っている（引き継がないと元の大きさで鳴る）
        assert float(np.abs(later[history : history + 100]).max()) < 0.29

    def test_the_mixer_carries_the_history_across_windows(self, tmp_path: Path) -> None:
        """ミキサの区切りの直前の山で絞った分は、区切りの後も戻りきるまで続く

        ``limit`` へ履歴を直に渡すだけでは、ミキサが前の音を読み戻さなくなっても通る
        後半だけを新しいミキサで鳴らし、通して鳴らした後半と比べる
        """
        samples = tone(1.0, -20.0)
        boundary = HISTORY_WINDOW
        samples[boundary - 200 : boundary - 100] = 0.9
        path = write_tone(tmp_path / "peak.wav", samples)
        media = probe_media(path)
        limiter = registry.require(LIMITER_KIND).create(ceiling=-10.0, release=150.0)
        clip = Clip(timeline_start=0, duration=30, media_id=media.id, effects=(limiter,))
        base = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(media,))
        track = Track(kind=TrackKind.AUDIO, name="声", clips=(clip,))
        project = base.with_timeline(replace(base.timeline, tracks=(track,)))
        count = len(samples) - boundary
        whole_mixer, later_mixer = AudioMixer(project), AudioMixer(project)
        try:
            whole = whole_mixer.render(0, len(samples))
            later = later_mixer.render(boundary, count)
        finally:
            whole_mixer.close()
            later_mixer.close()
        # 区切りの直後はまだ絞っている（戻る途中） 読み戻さないと素の -20 dBFS の波になる
        assert np.abs(whole[boundary : boundary + 480]).max() < 10 ** (-20 / 20) * 0.9
        np.testing.assert_allclose(later, whole[boundary:], atol=1e-4)

    def test_a_volume_added_later_goes_before_the_limiter(self) -> None:
        """固定の音量を持たない古いクリップへ後から音量を足すと、リミッターの後に掛かって山が越える"""
        effects = (registry.require(LIMITER_KIND).create(),)
        assert fixed_slot(effects, VOLUME_EFFECT_KIND) == 0
        assert loose_slot(effects) == 0

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
        # 抑えた分だけ音量は下がる 測る前の値から届いたとは言い切らない
        assert result["reached"] is None
        assert result["expected_lufs"] is None
        assert result["loudness_rechecked"] is False
        assert "loudness" in result["note"]
        assert result["gain_db"] == pytest.approx(4.0)
        assert result["limiter_db"] == pytest.approx(4.0)
        for clip in (work.clip(work.voice.id), work.clip(work.bgm.id)):
            assert _kinds(clip)[-1] == LIMITER_KIND
        # 声と BGM は重なって鳴る クリップごとのリミッターでは和の山を言い切れない
        assert result["expected_peak_dbfs"] is None
        assert result["mixed_peak_checked"] is False
        assert "clipping" in result["note"]

    def test_a_single_clip_peak_is_promised(self, work: _Work) -> None:
        """重ならない音なら、リミッターの上限がそのまま書き出しの山になる"""
        _run(work, clip_ids=[str(work.voice.id)])
        _measured(work.host, -18.0, -1.0)
        result = _run(work, clip_ids=[str(work.voice.id)])
        assert result["expected_peak_dbfs"] == -1.0
        assert "mixed_peak_checked" not in result

    def test_an_old_low_limiter_caps_the_gain(self, work: _Work) -> None:
        """-10 dBFS で抑えた音を上げても潰れるだけ 届いたと答えると、小さいまま書き出す"""
        effect = registry.require(LIMITER_KIND).create(ceiling=-10.0)
        work.host.apply_commands([AddEffect(work.voice.id, effect)], "前のリミッター")
        _run(work)
        _measured(work.host, -18.0, -10.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert result["reached"] is False
        assert result["limited_by"] == ["peak"]

    def test_a_peak_over_the_ceiling_is_limited_without_gain(self, work: _Work) -> None:
        """基準に合っていても、山が天井を越えたまま残すと書き出しで割れる"""
        _run(work)
        _measured(work.host, -14.0, 2.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert result["reached"] is None
        assert result["limiter_db"] == pytest.approx(3.0)
        assert _kinds(work.clip(work.voice.id))[-1] == LIMITER_KIND

    def test_the_limiting_is_capped(self, work: _Work) -> None:
        """壊れると、山を何 dB でも潰して、話し声の立ち上がりまで平たくなる"""
        _run(work)
        _measured(work.host, -30.0, -1.0)
        result = _run(work)
        assert result["gain_db"] == pytest.approx(MAX_LIMITER_DB)
        assert result["reached"] is False
        assert result["limited_by"] == ["peak"]
        assert "リミッターで" in result["note"]

    def test_a_second_call_does_not_stack(self, work: _Work) -> None:
        """抑えた後の音をまた抑えると、上限を越えて重なる"""
        _run(work)
        _measured(work.host, -30.0, -1.0)
        _run(work)
        _run(work)
        # 測るのはリミッターを外した音 12 dB 上げたので山は +11 dBFS
        _measured(work.host, -18.0, 11.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert result["reached"] is False
        limiters = [k for k in _kinds(work.clip(work.voice.id)) if k == LIMITER_KIND]
        assert limiters == [LIMITER_KIND]

    def test_a_new_clip_does_not_reopen_the_limiting(self, work: _Work) -> None:
        """抑えた作品へクリップを足して呼び直しても、前のクリップを重ねて潰さない（Codex の指摘）"""
        effect = registry.require(LIMITER_KIND).create(release=300.0)  # 人の整えたリミッター
        work.host.apply_commands([AddEffect(work.voice.id, effect)], "前のリミッター")
        _run(work)
        _measured(work.host, -18.0, -1.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert result["reached"] is False
        assert LIMITER_KIND not in _kinds(work.clip(work.bgm.id))

    def test_a_louder_track_lowers_the_new_limiter(self, work: _Work) -> None:
        """+6 dB のトラックに -1 dBFS で積むと、書き出しの山は +5 dBFS になる（CodeRabbit）"""
        _track_volume(work, 6.0)
        ids = [str(work.voice.id)]
        _run(work, clip_ids=ids)
        _measured(work.host, -18.0, -1.0)
        result = _run(work, clip_ids=ids)
        limiter = work.clip(work.voice.id).effects[-1]
        assert limiter.kind == LIMITER_KIND
        assert limiter.params["ceiling"] == AnimatedValue(static=-7.0)
        assert result["expected_peak_dbfs"] == -1.0

    def test_a_track_too_loud_for_the_limiter_is_told(self, work: _Work) -> None:
        """上限の範囲（-20 dBFS まで）で収め切れない分は、守れると言わない"""
        _track_volume(work, 24.0)
        ids = [str(work.voice.id)]
        _run(work, clip_ids=ids)
        _measured(work.host, -18.0, -1.0)
        result = _run(work, clip_ids=ids)
        # 上げた山は -1 + 4 = +3 dBFS リミッターは -20 dBFS（書き出しで +4）までしか下げられない
        assert result["expected_peak_dbfs"] == 3.0
        assert "トラックの音量" in result["note"]

    def test_an_old_limiter_on_a_quieter_track_caps_where_it_is_heard(self, work: _Work) -> None:
        """-6 dB のトラックの -10 dBFS のリミッターは、書き出しで -16 dBFS に山を止める

        測った山は -16 dBFS 上限の -10 と比べると 6 dB 上げられると読んで、潰れた分を届いたと答える
        """
        _track_volume(work, -6.0)
        effect = registry.require(LIMITER_KIND).create(ceiling=-10.0)
        work.host.apply_commands([AddEffect(work.voice.id, effect)], "前のリミッター")
        _run(work)
        _measured(work.host, -24.0, -16.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert result["reached"] is False

    def test_a_quiet_clip_with_a_low_limiter_does_not_lower_the_rest(self, work: _Work) -> None:
        """静かな声の -20 dBFS のリミッターで、山を出す BGM ごと全体を下げない（CodeRabbit）

        壊れると、上げたい作品の音量を -18 dB 下げる 測った山は混ぜた音の物で、声の山ではない
        """
        effect = registry.require(LIMITER_KIND).create(ceiling=-20.0)
        work.host.apply_commands([AddEffect(work.voice.id, effect)], "前のリミッター")
        _run(work)
        _measured(work.host, -18.0, -1.0)
        result = _run(work)
        assert result["gain_db"] == 0.0
        assert result["reached"] is False
        assert result["limited_by"] == ["peak"]

    def test_lowering_through_an_old_limiter_is_not_promised(self, work: _Work) -> None:
        """抑えていたリミッターの前で下げると、抑えていた分が戻って掛けた量ほどは下がらない

        壊れると、基準より大きいまま「届いた」と答えて書き出す
        """
        effect = registry.require(LIMITER_KIND).create(release=300.0)  # 人の整えたリミッター
        work.host.apply_commands([AddEffect(work.voice.id, effect)], "前のリミッター")
        _run(work)
        _measured(work.host, -10.0, -1.0)
        result = _run(work)
        assert result["gain_db"] == pytest.approx(-4.0)
        assert result["reached"] is None
        assert result["expected_lufs"] is None
        assert result["loudness_rechecked"] is False
        assert "loudness" in result["note"]

    def test_a_misplaced_old_limiter_is_put_back_at_the_end(self, work: _Work) -> None:
        """読み込んだファイルで音量の欄より前にある切ったリミッターを、そのまま入れ直さない

        壊れると、上げた音量の山がリミッターの後で上限を越える
        """
        effect = registry.require(LIMITER_KIND).create()
        work.host.apply_commands(
            [
                AddEffect(work.voice.id, effect, index=0),
                SetEffectEnabled(work.voice.id, effect.id, False),
            ],
            "前の並びのリミッター",
        )
        _run(work)
        _measured(work.host, -18.0, -1.0)
        _run(work)
        clip = work.clip(work.voice.id)
        limiters = [e for e in clip.effects if e.kind == LIMITER_KIND]
        assert len(limiters) == 1
        assert clip.effects[-1] is limiters[0]
        assert limiters[0].enabled

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


class TestTheCeilingInUse:
    """前に積んだリミッターが実際に山を止める所 リミッターと同じ読み方をする"""

    def _limited(self, clip: Clip, ceiling: float = 0.0, volume_db: float = 0.0) -> float:
        track = Track(kind=TrackKind.AUDIO, name="声", volume_db=volume_db)
        return limited_ceiling([(track, clip)], ceiling)

    def test_an_omitted_ceiling_is_the_default(self, work: _Work) -> None:
        # 壊れると、上限を省いたリミッター（-1 dBFS で潰す）の上へ上げて、届いたと答える
        effect = registry.require(LIMITER_KIND).create()
        bare = replace(effect, params={k: v for k, v in effect.params.items() if k != "ceiling"})
        clip = replace(work.voice, effects=(*work.voice.effects, bare))
        assert self._limited(clip) == pytest.approx(-1.0)

    def test_an_overshooting_curve_is_seen(self, work: _Work) -> None:
        """back の曲線はキーフレームの値より下へ行き過ぎる その谷でも潰れる"""
        effect = registry.require(LIMITER_KIND).create()
        moving = AnimatedValue(
            -1.0,
            (
                Keyframe(0, -3.0, Interpolation.EASE_IN_OUT, curve="back"),
                Keyframe(30, -1.0),
            ),
        )
        clip = replace(work.voice, effects=(effect.with_param("ceiling", moving),))
        assert self._limited(clip) < -3.0

    def test_a_disabled_track_gain_is_not_applied_to_video(self, work: _Work) -> None:
        """映像トラックの音量はミキサが使わない（置いたシーンの音だけを鳴らす）"""
        effect = registry.require(LIMITER_KIND).create(ceiling=-10.0)
        clip = replace(work.voice, effects=(effect,))
        track = Track(kind=TrackKind.VIDEO, name="V", volume_db=-6.0)
        assert limited_ceiling([(track, clip)], 0.0) == pytest.approx(-10.0)

    def test_no_old_limiter_leaves_the_squeeze_free(self, work: _Work) -> None:
        """壊れると、画面の〔音量を基準にそろえる…〕だけ山を天井で止め、リミッターを入れても届かない"""
        track = Track(kind=TrackKind.AUDIO, name="声")
        assert raise_cap([(track, work.voice)], -1.0) is None

    def test_a_low_old_limiter_still_caps_the_raise(self, work: _Work) -> None:
        # 壊れると、前のリミッターが潰す所より上へ上げて、届いたと答える
        effect = registry.require(LIMITER_KIND).create(ceiling=-10.0)
        clip = replace(work.voice, effects=(*work.voice.effects, effect))
        track = Track(kind=TrackKind.AUDIO, name="声")
        assert raise_cap([(track, clip)], -1.0) == pytest.approx(-10.0)


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


def _peaky(seconds: float = 12.0) -> np.ndarray:
    """平均より 20 dB ほど高い山が時々立つ、話し声に似た音"""
    times = np.arange(int(RATE * seconds)) / RATE
    body = 0.09 * np.sin(2 * np.pi * 220 * times) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * times))
    burst = 0.85 * np.sin(2 * np.pi * 700 * times) * np.exp(-(times % 0.9) / 0.012)
    mono = (body + burst).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def _meter(samples: np.ndarray) -> LoudnessMeter:
    meter = LoudnessMeter(RATE, 2)
    # 検査は塊で渡す 区切りの端数の持ち越しも一緒に通す
    for start in range(0, len(samples), 7001):
        meter.add(samples[start : start + 7001])
    return meter


def _limited(samples: np.ndarray, gain: float) -> float:
    context = AudioContext(offset=0, sample_rate=RATE, duration=len(samples))
    values = {"ceiling": -1.0, "release": 150.0}
    value = _meter(limit(samples * 10 ** (gain / 20), values, context)).integrated()
    assert value is not None
    return value


class TestTheEstimate:
    """抑えた分の音量の低下を見込む（#494）"""

    @pytest.mark.parametrize("gain", [0.0, 4.0, 8.0, 12.0])
    def test_the_estimate_matches_the_real_limiter(self, gain: float) -> None:
        """ずれると、届いたと答えた作品が小さいか大きいまま書き出される"""
        samples = _peaky()
        dynamics = _meter(samples).dynamics()
        assert dynamics is not None
        estimated = dynamics.limited_loudness(gain, -1.0)
        assert estimated is not None
        assert estimated == pytest.approx(_limited(samples, gain), abs=0.3)

    @pytest.mark.parametrize(("depth", "reaches"), [(12.0, True), (4.0, False)])
    def test_the_gain_makes_up_for_the_squashed_peaks(self, depth: float, reaches: bool) -> None:
        """見込まないと基準に届かず、上限を越えて見込むと浅くした人の立ち上がりまで潰す"""
        samples = _peaky()
        meter = _meter(samples)
        measured, dynamics = meter.integrated(), meter.dynamics()
        assert measured is not None
        assert dynamics is not None
        target = measured + 5.0
        decided = loudness_gain(
            [],
            measured=measured,
            peak_db=float(20 * np.log10(meter.peak)),
            target=target,
            ceiling=-1.0,
            squeeze=depth,
            predict=lambda gain: dynamics.limited_loudness(gain, -1.0),
        )
        assert decided.squeeze_db <= depth + 1e-6
        if reaches:
            # 掛ける量は足りない分より多い 抑えて下がる分を見込んでいる
            assert decided.gain_db > 5.0
            assert _limited(samples, decided.gain_db) == pytest.approx(target, abs=0.3)
            assert not decided.limited_by
        else:
            assert decided.limited_by == ("peak",)


class TestAnotherCall:
    """掛け直しは抑える前の音から数える（#494）"""

    def test_the_own_limiter_is_bypassed_to_measure(self, work: _Work) -> None:
        """残して測ると、2 回目から 1 dB も上がらず基準に届かない"""
        effect = registry.require(LIMITER_KIND).create()
        work.host.apply_commands([AddEffect(work.voice.id, effect)], "そろえたときのリミッター")
        _run(work)
        project = work.host.check_requests[-1].project
        assert project is not None
        clips = [c for t in project.timeline.tracks for c in t.clips]
        found = [e for c in clips for e in c.effects if e.kind == LIMITER_KIND]
        assert found
        assert not any(e.enabled for e in found)
        _measured(work.host, -18.0, -1.0)
        assert _run(work)["gain_db"] == pytest.approx(4.0)

    def test_a_tuned_limiter_stays_while_measuring(self, work: _Work) -> None:
        """人の整えたリミッターを外して測ると、その人の音と違う音から掛ける量を決める"""
        effect = registry.require(LIMITER_KIND).create(release=300.0)
        work.host.apply_commands([AddEffect(work.voice.id, effect)], "人のリミッター")
        _run(work)
        project = work.host.check_requests[-1].project
        assert project is not None
        clips = [c for t in project.timeline.tracks for c in t.clips]
        assert all(e.enabled for c in clips for e in c.effects if e.kind == LIMITER_KIND)

    def test_the_depth_follows_the_preference(self, work: _Work) -> None:
        """設定を読まないと、浅くした人の音も 12 dB まで潰す"""
        work.host.depth = 6.0
        _run(work)
        _measured(work.host, -30.0, -1.0)
        assert _run(work)["gain_db"] == pytest.approx(6.0)

    def test_a_settled_call_says_the_loudness_after_limiting(self, work: _Work) -> None:
        """抑える前の値だけ返すと、基準に合った音を AI が大きすぎると読んで下げる"""
        for clip_id in (work.voice.id, work.bgm.id):
            effect = registry.require(LIMITER_KIND).create()
            work.host.apply_commands([AddEffect(clip_id, effect)], "そろえたときのリミッター")
        samples = _peaky() * 10 ** (6.0 / 20)
        meter = _meter(samples)
        loudness, dynamics = meter.integrated(), meter.dynamics()
        assert loudness is not None
        assert dynamics is not None
        target = dynamics.limited_loudness(0.0, -1.0)
        assert target is not None
        assert clips_overlap([work.clip(work.voice.id), work.clip(work.bgm.id)])
        _run(work, target_lufs=loudness)
        _measured(work.host, loudness, float(20 * np.log10(meter.peak)))
        state = work.host.check
        assert state.report is not None
        work.host.check = replace(state, report=replace(state.report, dynamics=dynamics))
        result = _run(work, target_lufs=loudness)
        assert result["gain_db"] == 0.0
        # 声と BGM が重なる 和は抑えた後を見積もれないので、届いたとも合っているとも言い切らない
        assert result["reached"] is None
        assert "expected_lufs" not in result
        assert "すでに基準" not in result["note"]
        alone = [str(work.voice.id)]
        _run(work, clip_ids=alone, target_lufs=target)
        _measured(work.host, loudness, float(20 * np.log10(meter.peak)))
        state = work.host.check
        assert state.report is not None
        work.host.check = replace(state, report=replace(state.report, dynamics=dynamics))
        result = _run(work, clip_ids=alone, target_lufs=target)
        assert result["gain_db"] == 0.0
        assert result["reached"] is True
        assert result["expected_lufs"] == pytest.approx(target, abs=0.15)


class TestWhereTheEstimateHolds:
    """見積もりが実際のミキサと合う所でだけ使う（PR #495 のレビュー）"""

    def test_overlapping_sounds_are_not_predicted(self, work: _Work) -> None:
        """混ぜた山を抑える見込みで上げると、クリップごとには抑えられず基準より大きくなる"""
        _run(work)
        _measured(work.host, -20.0, -6.0)
        state = work.host.check
        assert state.report is not None
        report = replace(state.report, dynamics=_meter(_peaky()).dynamics())
        voice, bgm = work.clip(work.voice.id), work.clip(work.bgm.id)
        assert loudness_predictor(report, -1.0, [voice, bgm]) is None
        assert loudness_predictor(report, -1.0, [voice]) is not None
        work.host.check = replace(state, report=report)
        result = _run(work)
        assert result["gain_db"] == pytest.approx(6.0)
        assert result["reached"] is None
        assert result["expected_lufs"] is None

    def test_the_hold_does_not_cross_a_clip_head(self) -> None:
        """持ち越すと次のクリップの頭を小さく見積もり、上げすぎて基準より大きくなる"""
        times = np.arange(int(RATE * 0.2)) / RATE
        tone_mono = np.sin(2 * np.pi * 440 * times)
        parts, starts, limited = [], [], []
        context = AudioContext(offset=0, sample_rate=RATE, duration=len(times))
        for index in range(30):
            mono = (tone_mono * (2.5 if index % 2 == 0 else 0.6)).astype(np.float32)
            part = np.stack([mono, mono], axis=1)
            parts.append(part)
            starts.append(index * 0.2)
            limited.append(limit(part.copy(), {"ceiling": -1.0, "release": 150.0}, context))
        dynamics = _meter(np.concatenate(parts)).dynamics()
        assert dynamics is not None
        real = _meter(np.concatenate(limited)).integrated()
        assert real is not None
        assert dynamics.limited_loudness(0.0, -1.0, starts) == pytest.approx(real, abs=0.2)
        carried = dynamics.limited_loudness(0.0, -1.0)
        assert carried is not None
        assert carried < real - 0.2

    @pytest.mark.parametrize(("target", "found"), [(-16.4, 4.1), (-16.05, 4.45), (-15.5, 4.5)])
    def test_the_loudness_may_fall_as_the_gain_rises(self, target: float, found: float) -> None:
        """上端だけで諦めると、途中で届くのに潰せるだけ潰して基準より小さくなる"""
        decided = loudness_gain(
            [],
            measured=target - 3.2,
            peak_db=-3.0,
            target=target,
            ceiling=-1.0,
            squeeze=11.0,
            # 4.5 dB で一番大きく、その先は静かな所がゲートを越えて平均が下がる
            predict=lambda gain: -16.0 - abs(gain - 4.5),
        )
        assert decided.gain_db == pytest.approx(found, abs=0.11)
        assert bool(decided.limited_by) is (target > -16.0)

    @pytest.mark.parametrize("peak_at", [3.25, 4.45, 7.0, 9.91, 12.9])
    def test_a_band_narrower_than_the_scan_is_found(self, peak_at: float) -> None:
        """1 dB おきの点の間だけで届く形を飛び越すと、届くのに手前で止めて届かないと答える"""
        decided = loudness_gain(
            [],
            measured=-19.2,
            peak_db=-3.0,
            target=-16.0,
            ceiling=-1.0,
            squeeze=11.0,
            predict=lambda gain: -15.96 - abs(gain - peak_at),
        )
        assert decided.limited_lufs == pytest.approx(-16.0, abs=0.01)
        assert decided.gain_db <= peak_at + 1e-3
        assert not decided.limited_by

    def test_a_rising_end_is_not_searched_again(self) -> None:
        """届かない長い作品で山を探し直すと、そろえる度に画面が数秒止まる"""
        asked: list[float] = []

        def rising(gain: float) -> float:
            asked.append(gain)
            return -20.0 + gain * 0.1

        decided = loudness_gain(
            [],
            measured=-19.2,
            peak_db=-3.0,
            target=-16.0,
            ceiling=-1.0,
            squeeze=11.0,
            predict=rising,
        )
        assert decided.gain_db == pytest.approx(13.0)
        assert len(asked) <= 15

    def test_the_active_limiter_is_updated(self, work: _Work) -> None:
        """前にある無効な物を取ると、積み直した物と効いていた物が重なって潰れる"""
        definition = registry.require(LIMITER_KIND)
        idle, active = definition.create(), definition.create()
        clip = work.clip(work.voice.id)
        clip = replace(clip, effects=(replace(idle, enabled=False), *clip.effects, active))
        (command,) = limiter_commands([(work.voice_track, clip)], -1.0)
        assert isinstance(command, SetParam)
        assert command.path.effect_id == active.id


def _overlapping(tmp_path: Path, music: np.ndarray) -> Project:
    """山の立つ声と、その下に通しで鳴る音"""
    voice = _peaky() * 10 ** (-6.0 / 20)
    media, tracks = [], []
    for name, sound in (("voice", voice), ("music", music)):
        path = tmp_path / f"{name}.wav"
        write_tone(path, sound)
        item = probe_media(path)
        media.append(item)
        clip = Clip(timeline_start=0, duration=360, media_id=item.id)
        tracks.append(Track(kind=TrackKind.AUDIO, name=name, clips=(clip,)))
    project = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=tuple(media))
    return project.with_timeline(replace(project.timeline, tracks=tuple(tracks)))


def _music(level: float) -> np.ndarray:
    times = np.arange(RATE * 12) / RATE
    mono = (level * np.sin(2 * np.pi * 330 * times)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def _mixed(project: Project, frames: int = 360) -> np.ndarray:
    mixer = AudioMixer(project)
    try:
        pieces = [mixer.render_frames(at, min(30, frames - at)) for at in range(0, frames, 30)]
        return np.concatenate(pieces)
    finally:
        mixer.close()


def _checked(project: Project, frames: int = 360) -> CheckReport:
    """実際の検査の道で、組に分けて測る"""
    request = CheckRequest(
        CheckLimits(kinds=frozenset({LOUDNESS})), project=project, end=frames, separate=True
    )
    progress = _Progress(request, lambda: None)
    _check_timeline(request, progress, None, decode_threads=1, smooth_history=True)
    progress.running = False
    report = progress.report()
    assert report is not None
    return report


class TestOverlappingSounds:
    """声と BGM が重なっても基準へ届き、割れない（#494）"""

    def test_the_parts_add_up_to_the_mix(self, tmp_path: Path) -> None:
        """内訳が混ぜた音とずれると、違う音から掛ける量と上限を決める"""
        project = _overlapping(tmp_path, _music(0.05))
        meters = LaneMeters.of(project, 0, 360)
        assert meters is not None
        mixer = AudioMixer(project)
        try:
            mixed, parts, seen = mixer.render_frames_apart(30, 5, meters.lane_of, meters.count)
            assert seen == pytest.approx(np.abs(parts).max(axis=2), abs=1e-6)
            assert meters.count == 2
            assert parts.sum(axis=0) == pytest.approx(mixed, abs=1e-6)
            assert mixed == pytest.approx(mixer.render_frames(30, 5), abs=1e-6)
        finally:
            mixer.close()

    @pytest.mark.parametrize("level", [0.02, 0.05])
    def test_it_reaches_without_clipping(self, tmp_path: Path, level: float) -> None:
        """見込まないと 1 LU 手前で止まり、見込むだけだと抑えた音どうしの和が割れる"""
        project = _overlapping(tmp_path, _music(level))
        report = _checked(project)
        assert report.loudness is not None
        targets = [(track, track.clips[0]) for track in project.timeline.tracks]
        target = report.loudness + 6.0
        plan = plan_loudness(
            report, targets, target=target, ceiling=-1.0, squeeze=12.0, raise_ceiling=None
        )
        assert plan.held_db is not None
        assert plan.decided.squeeze_db <= 12.0 + 1e-6
        clips = [clip for _, clip in targets]
        changed = project
        commands = gain_commands(clips, plan.decided.gain_db)
        for command in commands + limiter_commands(targets, -1.0, plan.lowered):
            changed = command.apply(changed)
        out = _mixed(changed)
        assert _meter(out).integrated() == pytest.approx(target, abs=0.25)
        assert float(np.abs(out).max()) <= 10 ** (plan.held_db / 20) + 1e-3

    def test_too_shallow_a_depth_keeps_the_old_way(self, tmp_path: Path) -> None:
        """余裕を取ると前より小さくなるのに取ると、そろえたせいで声が引っ込む"""
        project = _overlapping(tmp_path, _peaky() * 0.5)
        report = _checked(project)
        assert report.loudness is not None
        assert report.dynamics is not None
        targets = [(track, track.clips[0]) for track in project.timeline.tracks]
        target = report.loudness + 5.0
        plan = plan_loudness(
            report, targets, target=target, ceiling=-1.0, squeeze=3.0, raise_ceiling=None
        )
        assert plan.lowered == 0.0
        assert plan.held_db is None
        assert "peak" in plan.decided.limited_by
        assert plan.decided.limited_lufs is not None

    def test_the_lowered_limiter_is_our_own_until_a_person_moves_it(self, work: _Work) -> None:
        """人の物と読むと掛け直せず、人が動かした物まで外すとその人の音を作り直す"""
        targets = [(work.voice_track, work.clip(work.voice.id))]
        work.host.apply_commands(limiter_commands(targets, -1.0, 2.5), "そろえた")
        clip = work.clip(work.voice.id)
        assert redoable_limiters([(work.voice_track, clip)], -1.0)
        limiter = next(e for e in clip.effects if e.kind == LIMITER_KIND)
        assert limiters_match([(work.voice_track, clip)], -3.5)
        path = ParamPath.of_effect(clip.id, limiter.id, "ceiling")
        work.host.apply_commands([SetParam(path, AnimatedValue(static=-5.0))], "人が直した")
        assert not redoable_limiters([(work.voice_track, work.clip(work.voice.id))], -1.0)

    def test_the_note_of_the_lowered_ceiling_is_saved(self) -> None:
        """保存で落ちると、開き直した作品では掛け直しても基準へ上がらない"""
        effect = registry.require(LIMITER_KIND).create(ceiling=-3.5)
        effect = effect.with_param(FITTED_CEILING, AnimatedValue(static=-3.5))
        assert effect_from_json(effect_to_json(effect)).params == effect.params

    def test_the_request_asks_to_measure_apart(self, work: _Work) -> None:
        """頼まないと組の手がかりが無く、重なる音は 1 LU 手前で止まる"""
        _run(work)
        assert work.host.check_requests[-1].separate
        _run(work, clip_ids=[str(work.voice.id)])
        assert not work.host.check_requests[-1].separate


def _stacked(tmp_path: Path, *sounds: np.ndarray) -> tuple[Project, list[tuple[Track, Clip]]]:
    """同じ時刻に鳴る音を、1 本ずつ別のトラックへ"""
    media, tracks = [], []
    for index, sound in enumerate(sounds):
        path = tmp_path / f"sound{index}.wav"
        write_tone(path, sound.astype(np.float32))
        item = probe_media(path)
        media.append(item)
        clip = Clip(timeline_start=0, duration=len(sound) * 30 // RATE, media_id=item.id)
        tracks.append(Track(kind=TrackKind.AUDIO, name=str(index), clips=(clip,)))
    project = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=tuple(media))
    project = project.with_timeline(replace(project.timeline, tracks=tuple(tracks)))
    return project, [(track, track.clips[0]) for track in tracks]


def _applied(project: Project, targets: list[tuple[Track, Clip]], plan: LoudnessPlan) -> Project:
    clips = [clip for _, clip in targets]
    commands = gain_commands(clips, plan.decided.gain_db)
    for command in commands + limiter_commands(targets, -1.0, plan.lowered):
        project = command.apply(project)
    return project


class TestTheSumOfTheLanes:
    """組の和の見積もり（PR #495 の 2 回目のレビュー）"""

    def test_the_last_part_of_a_second_is_held_too(self, tmp_path: Path) -> None:
        """100 ms に満たない末尾を見ないと、作品の終わりで同時に鳴る音の和が割れる"""
        frames = 362
        times = np.arange(frames * RATE // 30) / RATE
        ending = times >= 12.0
        sounds = []
        for hertz in (330.0, 470.0):
            mono = 0.05 * np.sin(2 * np.pi * hertz * times)
            mono = np.where(ending, 0.4 * np.sin(2 * np.pi * 1000 * times), mono)
            sounds.append(np.stack([mono, mono], axis=1))
        project, targets = _stacked(tmp_path, *sounds)
        report = _checked(project, frames)
        assert report.loudness is not None
        plan = plan_loudness(
            report,
            targets,
            target=report.loudness + 6.0,
            ceiling=-1.0,
            squeeze=12.0,
            raise_ceiling=None,
        )
        assert plan.held_db is not None
        top = float(np.abs(_mixed(_applied(project, targets, plan), frames)).max())
        assert top <= 10 ** (plan.held_db / 20) + 1e-3

    @pytest.mark.parametrize("extra", [False, True])
    @pytest.mark.parametrize("second", [-0.5, 0.5])
    def test_correlated_lanes_are_estimated(
        self, tmp_path: Path, second: float, extra: bool
    ) -> None:
        """組の絞り方が違うと相殺の具合が変わる 比率だけでは基準から外れたのに届いたと答える"""
        base = _peaky() * 0.98
        sounds = [base, base * second]
        if extra:
            sounds.append(_music(0.03))
        project, _targets = _stacked(tmp_path, *sounds)
        report = _checked(project)
        assert report.dynamics is not None
        for gain in (4.0, 8.0):
            context = AudioContext(offset=0, sample_rate=RATE, duration=len(base))
            values = {"ceiling": -1.0, "release": 150.0}
            summed = np.zeros_like(base)
            for sound in sounds:
                part = (sound * 10 ** (gain / 20)).astype(np.float32)
                summed += limit(part, values, context)
            real = _meter(summed).integrated()
            assert real is not None
            guessed = report.dynamics.limited_loudness(gain, -1.0)
            assert guessed == pytest.approx(real, abs=0.35)

    def test_the_depth_counts_each_lane(self, tmp_path: Path) -> None:
        """逆相では混ぜた山が組の山より低い 混ぜた山で数えると本人の深さを越えて潰す"""
        base = _peaky() * 0.98
        project, targets = _stacked(tmp_path, base, base * -0.5)
        report = _checked(project)
        # もう積んである形で見る 積んでいなければ、抑えて相殺するより抑えない方を選ぶ
        limiting = project
        for command in limiter_commands(targets, -1.0):
            limiting = command.apply(limiting)
        targets = [(track, track.clips[0]) for track in limiting.timeline.tracks]
        assert report.loudness is not None
        depth = 9.0
        plan = plan_loudness(
            report,
            targets,
            target=report.loudness + 14.0,
            ceiling=-1.0,
            squeeze=depth,
            raise_ceiling=None,
        )
        loudest = float(20 * np.log10(np.abs(base).max()))
        cut = loudest + plan.decided.gain_db - (-1.0 - plan.lowered)
        assert cut <= depth + 0.05
        assert plan.decided.squeeze_db >= cut - 0.05

    def test_a_soloed_out_mixed_track_is_not_a_lane(self, tmp_path: Path) -> None:
        """鳴らないトラックを数えると組の上限を越え、見積もりが使われない"""
        project, targets = _stacked(tmp_path, *([_music(0.02)] * 7))
        tracks = [
            replace(track, solo=True)
            if index < 2
            else replace(track, kind=TrackKind.MIXED, clips=(replace(clip, audio_stream=0),))
            for index, (track, clip) in enumerate(targets)
        ]
        project = project.with_timeline(replace(project.timeline, tracks=tuple(tracks)))
        meters = LaneMeters.of(project, 0, 360)
        assert meters is not None
        assert meters.count == 2


class TestLimitersFromTheNarrowRange:
    """欄が -20〜0 dB だったときに、0 dB で頭打ちにして置いた自動のリミッター（#499）"""

    @pytest.mark.parametrize("volume", [-6.0, -20.0])
    def test_an_old_capped_limiter_is_redone(self, volume: float) -> None:
        """人の物と読むと外して測れず、下げたトラックが前の止まる所のまま基準に届かない"""
        track = Track(kind=TrackKind.AUDIO, name="声", volume_db=volume)
        old = registry.require(LIMITER_KIND).create(ceiling=0.0)
        clip = Clip(timeline_start=0, duration=30, effects=(old,))
        assert redoable_limiters([(track, clip)], -1.0) == frozenset({old.id})

    def test_a_limiter_set_by_hand_below_the_old_cap_is_kept(self) -> None:
        """前の範囲で止まらない値は、前の道具も置かない 人の物を外して測ると、その人の音と違う"""
        track = Track(kind=TrackKind.AUDIO, name="声", volume_db=-6.0)
        tuned = registry.require(LIMITER_KIND).create(ceiling=-0.5)
        clip = Clip(timeline_start=0, duration=30, effects=(tuned,))
        assert redoable_limiters([(track, clip)], -1.0) == frozenset()

    def test_a_track_that_never_reached_the_cap_is_unchanged(self) -> None:
        """前の範囲に収まっていたトラックの 0 dB は、人が置いた値"""
        track = Track(kind=TrackKind.AUDIO, name="声", volume_db=0.0)
        tuned = registry.require(LIMITER_KIND).create(ceiling=0.0)
        clip = Clip(timeline_start=0, duration=30, effects=(tuned,))
        assert redoable_limiters([(track, clip)], -1.0) == frozenset()


class TestLeftLimiters:
    """外して測ったリミッターを、音に効いたまま残さない"""

    @pytest.mark.parametrize("over", [0.0, 2.0])
    def test_a_lowered_limiter_is_put_back(self, work: _Work, over: float) -> None:
        """残すと、外して測った音で届いたと答えたのに、書き出しは抑えられて基準より小さい"""
        alone = [str(work.voice.id)]
        targets = [(work.voice_track, work.clip(work.voice.id))]
        work.host.apply_commands(limiter_commands(targets, -1.0, 5.0), "そろえた")
        _run(work, clip_ids=alone, target_lufs=-18.0)
        _measured(work.host, -18.0 + over, -3.0)
        result = _run(work, clip_ids=alone, target_lufs=-18.0)
        assert result["reached"] is True
        clip = work.clip(work.voice.id)
        assert limiters_match([(work.voice_track, clip)], -1.0)
        assert len([e for e in clip.effects if e.kind == LIMITER_KIND]) == 1
        # もう同じ形なら積み直さない
        _run(work, clip_ids=alone, target_lufs=-18.0)
        _measured(work.host, -18.0, -3.0)
        before = work.host.document.project
        _run(work, clip_ids=alone, target_lufs=-18.0)
        assert work.host.document.project == before

    def test_the_note_follows_the_ceiling(self, work: _Work) -> None:
        """古い控えが残ると、人が同じ値へ直した上限をそろえた物として外して測る"""
        for lowered in (2.5, 0.0):
            targets = [(work.voice_track, work.clip(work.voice.id))]
            work.host.apply_commands(limiter_commands(targets, -1.0, lowered), "そろえた")
        clip = work.clip(work.voice.id)
        limiter = next(e for e in clip.effects if e.kind == LIMITER_KIND)
        path = ParamPath.of_effect(clip.id, limiter.id, "ceiling")
        work.host.apply_commands([SetParam(path, AnimatedValue(static=-3.5))], "人が直した")
        assert not redoable_limiters([(work.voice_track, work.clip(work.voice.id))], -1.0)


class TestTheLimiterComesBeforeTheFader:
    """リミッターはトラックの音量と定位の前に掛かる（PR #495 の 3 回目のレビュー）"""

    @pytest.mark.parametrize("overlap", [False, True])
    @pytest.mark.parametrize("volume", [-20.0, -6.0, 0.0])
    def test_a_lowered_track_still_reaches_the_target(
        self, tmp_path: Path, volume: float, overlap: bool
    ) -> None:
        """欄が 0 dB で止まると、下げたトラックの声だけが基準に届かないまま書き出す（#499）"""
        voice = _peaky() * 0.98
        sounds = [voice, _music(0.02)] if overlap else [voice]
        project, targets = _stacked(tmp_path, *sounds)
        tracks = tuple(
            replace(track, volume_db=volume) if index == 0 else track
            for index, track in enumerate(project.timeline.tracks)
        )
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        targets = [(track, track.clips[0]) for track in project.timeline.tracks]
        report = _checked(project)
        assert report.loudness is not None
        target = report.loudness + 5.0
        plan = plan_loudness(
            report, targets, target=target, ceiling=-1.0, squeeze=12.0, raise_ceiling=None
        )
        assert "peak" not in plan.decided.limited_by
        changed = _applied(project, targets, plan)
        mixed = _mixed(changed)
        assert _meter(mixed).integrated() == pytest.approx(target, abs=0.25)
        # 書き出しの山は頼んだ上限（重なる分を引いた所）を越えない
        held = -1.0 if plan.held_db is None else plan.held_db
        assert float(np.abs(mixed).max()) <= 10 ** (held / 20) + 1e-3

    @pytest.mark.parametrize("installed", [False, True])
    @pytest.mark.parametrize("overlap", [False, True])
    @pytest.mark.parametrize("pan", [0.0, 0.9])
    @pytest.mark.parametrize("volume", [-20.0, -6.0, 0.0, 3.0])
    def test_the_answer_is_what_is_exported(
        self, tmp_path: Path, volume: float, pan: float, overlap: bool, installed: bool
    ) -> None:
        """後の音で見積もると、届いたと答えた音が小さく書き出され、本人の深さも越えて潰す"""
        voice = _peaky() * np.array([0.98, 0.098])
        sounds = [voice, _music(0.02)] if overlap else [voice]
        project, targets = _stacked(tmp_path, *sounds)
        first = replace(project.timeline.tracks[0], volume_db=volume, pan=pan)
        tracks = (first, *project.timeline.tracks[1:])
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        bare = project
        targets = [(track, track.clips[0]) for track in project.timeline.tracks]
        if installed:
            for command in limiter_commands(targets, -1.0):
                project = command.apply(project)
            targets = [(track, track.clips[0]) for track in project.timeline.tracks]
        report = _checked(bare)
        assert report.loudness is not None
        depth = 12.0
        plan = plan_loudness(
            report,
            targets,
            target=report.loudness + 5.0,
            ceiling=-1.0,
            squeeze=depth,
            raise_ceiling=None,
        )
        assert not plan.unchecked
        gain = plan.decided.gain_db
        commands = gain_commands([clip for _, clip in targets], gain)
        squeezed = plan.decided.squeeze_db >= 0.05
        if squeezed:
            commands += limiter_commands(targets, -1.0, plan.lowered)
        changed = project
        for command in commands:
            changed = command.apply(changed)
        said = plan.decided.limited_lufs
        if said is None:
            said = report.loudness + gain
        assert _meter(_mixed(changed)).integrated() == pytest.approx(said, abs=0.25)
        if not squeezed:
            return
        # 本物のリミッターが潰す量 数えた深さと、本人の設定の深さに収まる
        cuts = [
            float(20 * np.log10(np.abs(sound).max()))
            + gain
            - limiter_ceiling(track, -1.0 - plan.lowered)
            for sound, (track, _clip) in zip(sounds, targets, strict=True)
        ]
        assert max(cuts) <= plan.decided.squeeze_db + 0.1
        assert plan.decided.squeeze_db <= depth + 0.1


class TestPlanJob:
    """抑えた後の見積もりを裏で回す口（#497）"""

    def test_a_failure_is_told_and_raised_again(self) -> None:
        """裏で落ちたのを知らせないと、待つ側は「見積もっています」のまま止まる"""
        told = threading.Event()

        def broken() -> LoudnessPlan:
            raise ValueError("壊れた手がかり")

        job = PlanJob(lambda **_kw: broken(), done=told.set)
        # 終わった印は知らせより先に立つ 印だけを待つと、知らせの前に読んで落ちることがある
        assert told.wait(GUARD_SECONDS)
        assert job.wait(0)
        with pytest.raises(ValueError, match="壊れた手がかり"):
            job.take()

    def test_an_abandoned_job_stops(self) -> None:
        """やめても最後まで回すと、選び直す度に 3 時間分で数秒ずつ裏の仕事が重なる"""
        steps: list[int] = []

        def work(*, stopped: Stopped) -> LoudnessPlan:
            while not stopped():
                steps.append(1)
                time.sleep(0.001)
            raise PlanAbandonedError

        job = PlanJob(work)
        wait_until(lambda: bool(steps), "見積もりの始まり")
        job.abandon()
        assert job.wait(GUARD_SECONDS)
        with pytest.raises(PlanAbandonedError):
            job.take()

    def test_the_estimate_asks_before_each_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """重なって鳴る音の見積もりは 1 回ごとに重い 尋ねずに回すと、やめても数秒止まらない"""
        project, targets = _stacked(tmp_path, _peaky() * 0.98, _music(0.05))
        report = _checked(project)
        assert report.loudness is not None
        loudness = report.loudness
        steps: list[str] = []
        real = Dynamics.limited_loudness

        def counted(self: Dynamics, *args: Any, **kwargs: Any) -> float | None:
            steps.append("見積もり")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Dynamics, "limited_loudness", counted)

        class CountedHeadroom(Headroom):
            def __init__(self, lanes: Sequence[LoudnessLane]) -> None:
                steps.append("和の上限")
                super().__init__(lanes)

        monkeypatch.setattr(loudness_plan, "Headroom", CountedHeadroom)

        def plan(stopped: Stopped) -> LoudnessPlan:
            return plan_loudness(
                report,
                targets,
                target=loudness + 6.0,
                ceiling=-1.0,
                squeeze=12.0,
                raise_ceiling=None,
                stopped=stopped,
            )

        def asked() -> bool:
            steps.append("尋ねた")
            return False

        plan(asked)
        assert "見積もり" in steps
        assert "和の上限" in steps
        # どの重い 1 歩（見積もり 1 回と和の上限の用意）の前にも尋ねている
        for at, step in enumerate(steps):
            if step != "尋ねた":
                assert at > 0 and steps[at - 1] == "尋ねた", steps[max(0, at - 3) : at + 1]
        with pytest.raises(PlanAbandonedError):
            plan(lambda: True)

    def test_nothing_is_taken_before_the_end(self) -> None:
        """終わる前に取って空の計画を当てると、何も見積もらないまま掛ける"""
        gate = threading.Event()
        job = PlanJob(lambda **_kw: (gate.wait(GUARD_SECONDS), _plain())[1])
        try:
            assert job.take() is None
            assert not job.wait(0)
        finally:
            gate.set()
        assert job.wait(GUARD_SECONDS)
        assert job.take() == _plain()


def _plain() -> LoudnessPlan:
    return LoudnessPlan(LoudnessGain(gain_db=1.0))
