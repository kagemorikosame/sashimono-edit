"""音の山をリミッターで抑えて音量を基準まで上げる（#414）

通しの確かめで、一瞬の山が天井に当たって基準に 3.8〜6 LU 届かなかった 山だけを抑えれば上げられる
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.ai.auto_fix import limited_ceiling
from sashimono.ai.fixes import ASK, AUTO, fix_for
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddEffect, SetEffectEnabled, SetParam
from sashimono.core.commands.effects import ParamPath
from sashimono.core.commands.fixed import (
    VOLUME_EFFECT_KIND,
    fixed_slot,
    loose_slot,
    with_fixed_items,
)
from sashimono.core.mixing import MAX_LIMITER_DB
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
from sashimono.engine.checks.report import HIGH, MEDIUM
from sashimono.engine.checks.scan import LOUDNESS
from sashimono.engine.decode.probe import probe_media
from tests.ai.conftest import FakeHost
from tests.ai.test_mix_tools import _measured, _Work
from tests.check_media import tone, write_tone

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
        effect = registry.require(LIMITER_KIND).create()
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
