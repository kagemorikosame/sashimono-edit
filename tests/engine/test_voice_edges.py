"""字幕の行の頭と終わりを声へ寄せる（#426）"""

from __future__ import annotations

import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.commands import SetTranscript
from sashimono.core.model import (
    MediaId,
    Project,
    SegmentId,
    Transcript,
    TranscriptSegment,
    Word,
)
from sashimono.core.subtitle_timing import SubtitleTiming
from sashimono.engine.audio.voice_edges import (
    fit_to_voice,
    loudness_of_samples,
    loudness_of_waveform,
    retime,
    retime_command,
)
from sashimono.engine.audio.waveform import analyze_waveform

RATE = 16000


def _voice(
    spans: list[tuple[float, float, float]], *, seconds: float, noise: float = 0.0
) -> np.ndarray:
    """``(頭, 終わり, 大きさ)`` の所だけ鳴る音 声の代わりの 220 Hz"""
    t = np.arange(int(seconds * RATE)) / RATE
    out = np.zeros_like(t, dtype=np.float32)
    for start, end, amplitude in spans:
        inside = (t >= start) & (t < end)
        out[inside] = amplitude * np.sin(2 * np.pi * 220 * t[inside])
    if noise:
        out += np.random.default_rng(0).uniform(-noise, noise, len(out)).astype(np.float32)
    return out


def _f(value: float) -> Fraction:
    return Fraction(value).limit_denominator(1000)


def _fit(times: list[tuple[float, float]], samples: np.ndarray) -> list[tuple[float, float]]:
    fitted = fit_to_voice([(_f(a), _f(b)) for a, b in times], loudness_of_samples(samples, RATE))
    return [(float(a), float(b)) for a, b in fitted]


class TestFitToVoice:
    def test_an_early_line_moves_to_where_the_voice_starts(self) -> None:
        # 壊れると、字幕が声より 0.3 秒早く出て、遅く消える（利用者の報告 #426）
        samples = _voice([(2.0, 4.0, 0.5)], seconds=7.0)
        ((start, end),) = _fit([(1.7, 4.3)], samples)
        assert start == pytest.approx(2.0, abs=0.02)
        assert end == pytest.approx(4.0, abs=0.02)

    def test_a_line_that_ends_too_soon_reaches_the_end_of_the_voice(self) -> None:
        # 語の時刻から作った行は終わりが早めに寄る 壊れると最後の語の途中で字幕が消える
        samples = _voice([(2.0, 4.0, 0.5)], seconds=7.0)
        ((start, end),) = _fit([(2.25, 3.75)], samples)
        assert start == pytest.approx(2.0, abs=0.02)
        assert end == pytest.approx(4.0, abs=0.02)

    def test_a_weak_tail_counts_as_voice(self) -> None:
        # 文の終わりの「す」は息だけの弱い音 壊れると、聞こえているのに字幕が先に消える
        samples = _voice([(2.0, 3.8, 0.5), (3.8, 4.0, 0.01)], seconds=7.0)
        ((_start, end),) = _fit([(2.0, 3.7)], samples)
        assert end == pytest.approx(4.0, abs=0.02)

    def test_it_does_not_reach_further_than_half_a_second(self) -> None:
        # 認識が別の文と取り違えた行を遠くの声へ持っていかない 壊れると正しい行まで動く
        samples = _voice([(2.0, 4.0, 0.5)], seconds=7.0)
        assert _fit([(1.0, 1.4)], samples) == [(1.0, 1.4)]

    def test_a_click_does_not_start_the_line(self) -> None:
        # 机を叩いた一瞬の音で字幕を早めない
        samples = _voice([(1.75, 1.76, 0.8), (2.0, 4.0, 0.5)], seconds=7.0)
        ((start, _end),) = _fit([(1.7, 4.0)], samples)
        assert start == pytest.approx(2.0, abs=0.02)

    def test_a_breath_inside_a_line_does_not_end_it(self) -> None:
        # 語の間の短い息継ぎ（0.1 秒）で文が終わったと読むと、最後の語の前で字幕が消える
        samples = _voice([(2.0, 3.0, 0.5), (3.1, 3.5, 0.5)], seconds=7.0)
        ((_start, end),) = _fit([(2.0, 3.05)], samples)
        assert end == pytest.approx(3.5, abs=0.02)

    def test_noise_is_not_mistaken_for_a_tail(self) -> None:
        # 雑音の多い素材で深いしきい値が雑音を声と読むと、字幕が無音の中まで残る
        samples = _voice([(2.0, 4.0, 0.5)], seconds=7.0, noise=0.01)
        ((start, end),) = _fit([(1.8, 4.3)], samples)
        assert start == pytest.approx(2.0, abs=0.03)
        assert end == pytest.approx(4.0, abs=0.03)

    def test_a_line_split_inside_speech_keeps_its_seam(self) -> None:
        # 認識が 1 つの話を 2 行に割った所は声が続く 壊れると後の行の頭が前の行の頭まで戻り、
        # 2 行が重なる
        samples = _voice([(2.0, 5.0, 0.5)], seconds=8.0)
        fitted = _fit([(1.8, 3.5), (3.5, 5.2)], samples)
        assert fitted[0] == pytest.approx((2.0, 3.5), abs=0.02)
        assert fitted[1] == pytest.approx((3.5, 5.0), abs=0.02)

    def test_a_short_line_before_a_split_keeps_its_words(self) -> None:
        # 割れ目の近くで話が始まると、後の行の頭が前の行の声の頭まで戻り、前の行が 0 秒に潰れる
        samples = _voice([(3.2, 5.0, 0.5)], seconds=8.0)
        fitted = _fit([(3.0, 3.6), (3.6, 5.2)], samples)
        assert fitted[0] == pytest.approx((3.2, 3.6), abs=0.02)
        assert fitted[1] == pytest.approx((3.6, 5.0), abs=0.02)

    def test_lines_stay_in_order_without_overlapping(self) -> None:
        # 壊れると、字幕の並びの不変条件が崩れて作品へ入れられず、焼き込みで前の字幕が欠ける
        samples = _voice([(1.0, 2.0, 0.5), (2.1, 3.0, 0.5)], seconds=5.0)
        fitted = _fit([(0.8, 1.9), (2.0, 3.2)], samples)
        assert fitted[0][1] <= fitted[1][0]
        assert fitted[0][0] <= fitted[1][0]

    def test_silence_leaves_the_line_alone(self) -> None:
        # 声の大きさを読めない所（無音の素材）で寄せると、行がどこかへ飛ぶ
        samples = np.zeros(5 * RATE, dtype=np.float32)
        assert _fit([(1.0, 2.0)], samples) == [(1.0, 2.0)]

    def test_lines_outside_the_sound_are_left_alone(self) -> None:
        # 素材より後ろを指す行（長さの読み違い）で落ちると、起こし全体が失敗に見える
        samples = _voice([(1.0, 2.0, 0.5)], seconds=3.0)
        assert _fit([(10.0, 11.0)], samples) == [(10.0, 11.0)]


class TestLoudness:
    def test_the_waveform_reads_like_the_samples(self, tmp_path: Path) -> None:
        # 起こした直後（読んだ音）と後から合わせ直す（解析した波形）で寄せる先が違うと、同じ字幕が
        # 合わせ直すたびに動く
        samples = _voice([(2.0, 4.0, 0.5)], seconds=7.0)
        path = tmp_path / "声.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes((samples * 32767).astype(np.int16).tobytes())
        waveform = analyze_waveform(path, sample_rate=48000, channels=1)
        assert waveform is not None
        times = [(_f(1.7), _f(3.75))]
        from_samples = fit_to_voice(times, loudness_of_samples(samples, RATE))
        from_waveform = fit_to_voice(times, loudness_of_waveform(waveform))
        for (a, b), (c, d) in zip(from_samples, from_waveform, strict=True):
            assert float(a) == pytest.approx(float(c), abs=0.02)
            assert float(b) == pytest.approx(float(d), abs=0.02)
        assert float(from_waveform[0][0]) == pytest.approx(2.0, abs=0.02)


def _segment(start: float, end: float, text: str, segment_id: str) -> TranscriptSegment:
    return TranscriptSegment(
        _f(start),
        _f(end),
        text,
        words=(Word(_f(start), _f(end), text),),
        speaker="話者 1",
        edited=True,
        id=SegmentId(segment_id),
    )


class TestRetime:
    def test_text_words_and_speaker_stay(self) -> None:
        # 時刻を合わせただけで文言や語の時刻が変わると、人が直した文と語の境目の確かめが崩れる
        samples = _voice([(2.0, 4.0, 0.5)], seconds=7.0)
        line = _segment(1.7, 4.3, "人が直した文", "a")
        (moved,) = retime([line], SubtitleTiming(), loudness_of_samples(samples, RATE))
        assert (moved.text, moved.words, moved.speaker, moved.edited, moved.id) == (
            line.text,
            line.words,
            line.speaker,
            line.edited,
            line.id,
        )
        assert float(moved.start) == pytest.approx(2.0, abs=0.02)

    def test_offsets_come_after_fitting(self) -> None:
        # 先にずらすと、好みで早めた頭が声の出だしへ戻り、早めた分が消える
        samples = _voice([(2.0, 4.0, 0.5)], seconds=7.0)
        timing = SubtitleTiming(start_offset_ms=-200, end_offset_ms=300)
        (moved,) = retime(
            [_segment(1.7, 4.3, "文", "a")], timing, loudness_of_samples(samples, RATE)
        )
        assert float(moved.start) == pytest.approx(1.8, abs=0.02)
        assert float(moved.end) == pytest.approx(4.3, abs=0.02)

    def test_turned_off_it_does_nothing(self) -> None:
        # 切っても寄せると、設定がある方が質が悪い
        samples = _voice([(2.0, 4.0, 0.5)], seconds=7.0)
        line = _segment(1.7, 4.3, "文", "a")
        timing = SubtitleTiming(fit_to_voice=False)
        assert retime([line], timing, loudness_of_samples(samples, RATE)) == (line,)

    def test_without_sound_it_only_shifts(self) -> None:
        # 長さの分からない素材（音を読めない）で落ちると、起こしそのものが失敗する
        line = _segment(1.7, 4.3, "文", "a")
        (moved,) = retime([line], SubtitleTiming(start_offset_ms=-100), None)
        assert (float(moved.start), float(moved.end)) == pytest.approx((1.6, 4.3))


class TestRetimeCommand:
    def _transcript(self) -> Transcript:
        return Transcript(
            (
                _segment(0.8, 1.9, "一", "a"),
                _segment(2.0, 3.2, "二", "b"),
                _segment(3.4, 4.6, "三", "c"),
            )
        )

    def test_only_the_chosen_lines_move(self) -> None:
        # 壊れると、選んだ行だけ直したつもりが、人が手で合わせた別の行まで動く
        samples = _voice([(1.0, 1.9, 0.5), (2.1, 3.0, 0.5), (3.6, 4.4, 0.5)], seconds=6.0)
        command = retime_command(
            MediaId("m"),
            self._transcript(),
            SubtitleTiming(),
            loudness_of_samples(samples, RATE),
            only={SegmentId("b")},
        )
        assert command is not None
        assert [t[0] for t in command.times] == [SegmentId("b")]

    def test_a_chosen_line_does_not_run_over_its_neighbour(self) -> None:
        # 選ばなかった隣の行は動かない 遅く消す分で隣の行の頭を越えると、焼き込みで欠ける
        command = retime_command(
            MediaId("m"),
            self._transcript(),
            SubtitleTiming(fit_to_voice=False, end_offset_ms=1000),
            None,
            only={SegmentId("a")},
        )
        assert command is not None
        ((_, _start, end),) = command.times
        assert end == _f(2.0)

    def test_a_chosen_line_does_not_jump_over_its_neighbour(self) -> None:
        # 大きくずらした行が選ばなかった隣の行を追い越すと、前後が入れ替わって重なりを止められず、
        # 焼き込みで隣の字幕が欠ける（PR #434 の Codex の指摘）
        transcript = Transcript((_segment(1.0, 1.5, "一", "a"), _segment(1.6, 2.1, "二", "b")))
        command = retime_command(
            MediaId("m"),
            transcript,
            SubtitleTiming(fit_to_voice=False, start_offset_ms=1000, end_offset_ms=1000),
            None,
            only={SegmentId("a")},
        )
        assert command is not None
        ((_, start, end),) = command.times
        assert end <= _f(1.6)
        assert start <= end

    def test_a_line_without_room_stays(self) -> None:
        # 前後の行の間に見える長さが残らないのに動かすと、行が潰れるか隣と重なる
        transcript = Transcript(
            (
                _segment(1.0, 1.5, "一", "a"),
                _segment(1.5, 1.55, "二", "b"),
                _segment(1.55, 2.0, "三", "c"),
            )
        )
        command = retime_command(
            MediaId("m"),
            transcript,
            SubtitleTiming(fit_to_voice=False, start_offset_ms=-300),
            None,
            only={SegmentId("b")},
        )
        assert command is None

    def test_all_lines_moved_early_stay_at_or_after_zero(self, project: Project) -> None:
        # 2 行とも 0 秒で止まると、先頭の行の頭が負になって命令が断られ、何も合わせられない
        media = project.media[0]
        transcript = Transcript((_segment(0.1, 0.5, "一", "a"), _segment(0.6, 1.0, "二", "b")))
        placed = SetTranscript(media.id, transcript).apply(project)
        command = retime_command(
            media.id, transcript, SubtitleTiming(fit_to_voice=False, start_offset_ms=-1000), None
        )
        assert command is not None
        moved = command.apply(placed).require_media(media.id).transcript_for(None)
        assert moved is not None
        assert all(s.start >= 0 and s.end > s.start for s in moved.segments)
        assert moved.segments[0].end <= moved.segments[1].start

    def test_nothing_to_move_gives_no_command(self) -> None:
        # 空の命令を履歴に積むと、何も変わらない「元に戻す」が 1 つ増える
        line = _segment(1.0, 2.0, "文", "a")
        samples = _voice([(1.0, 2.0, 0.5)], seconds=4.0)
        command = retime_command(
            MediaId("m"), Transcript((line,)), SubtitleTiming(), loudness_of_samples(samples, RATE)
        )
        assert command is None
