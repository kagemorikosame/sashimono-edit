"""時間変換のテスト

ここが狂うと長尺で音ズレになり、しかも症状が出るのが編集の終盤なので、
往復の正確さは property-based test で押さえる
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sashimono.core.timebase import (
    FrameRate,
    Rounding,
    format_timecode,
    frame_to_sample,
    frame_to_seconds,
    parse_timecode,
    pts_to_seconds,
    sample_to_frame,
    sample_to_seconds,
    seconds_to_frame,
    seconds_to_pts,
    seconds_to_sample,
)

NTSC_30 = FrameRate(30000, 1001)
NTSC_60 = FrameRate(60000, 1001)
FILM = FrameRate(24)
PAL = FrameRate(25)

RATES = [FILM, PAL, FrameRate(30), NTSC_30, NTSC_60, FrameRate(24000, 1001)]
SAMPLE_RATES = [44100, 48000, 96000]


class TestFrameRate:
    def test_reduces_to_lowest_terms(self) -> None:
        # 既約にしないと 60/2 と 30/1 が別のレートになり、保存した作品がタイムラインと設定の
        # フレームレートの食い違いとして開けなくなる
        assert FrameRate(60, 2) == FrameRate(30, 1)

    def test_rejects_non_positive(self) -> None:
        # 0 のレートを通すと 1 フレームの長さを求める所で 0 で割って落ち、負のレートでは
        # 時刻が逆向きに進む
        with pytest.raises(ValueError, match="正でなければ"):
            FrameRate(0)
        with pytest.raises(ValueError, match="正でなければ"):
            FrameRate(30, -1)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("30", FrameRate(30)),
            ("30000/1001", NTSC_30),
            ("29.97", NTSC_30),
            ("23.976", FrameRate(24000, 1001)),
            ("59.94", NTSC_60),
            ("25", PAL),
        ],
    )
    def test_parse(self, text: str, expected: FrameRate) -> None:
        # 小数を放送規格の分数へ寄せられないと 29.97 が 2997/100 のまま返り、欠番を持てない
        # レートと見なされて、format_timecode がドロップフレームで数えなくなる
        assert FrameRate.parse(text) == expected

    def test_decimal_snaps_to_broadcast_rate(self) -> None:
        # 29.97 と書かれたものは 2997/100 ではなく 30000/1001 を意図している
        # 吸着しないと 1 時間で 3 フレーム以上ずれる
        assert FrameRate.from_decimal(Fraction("29.97")) == NTSC_30
        assert FrameRate.from_decimal(29.97) == NTSC_30

    def test_unusual_rate_is_kept_as_is(self) -> None:
        # GoPro などが吐く半端なレートを勝手に丸めない
        assert FrameRate.from_decimal(Fraction("11.5")) == FrameRate(23, 2)

    @pytest.mark.parametrize(
        ("rate", "nominal"),
        [(NTSC_30, 30), (NTSC_60, 60), (FrameRate(24000, 1001), 24), (PAL, 25)],
    )
    def test_nominal_fps(self, rate: FrameRate, nominal: int) -> None:
        # 29.97 の数え方を 29 にすると、タイムコードのフレーム桁が 29 で繰り上がり、
        # 再生位置の表示が実際の時刻から毎秒ずれていく
        assert rate.nominal_fps == nominal

    def test_drop_frame_capability(self) -> None:
        # 29.97 で無しと答えると表示が 1 時間で 3.6 秒時計から遅れ、30fps で有りと答えると
        # 欠番の表に無いレートを引いてタイムコードの表示が落ちる
        assert NTSC_30.is_drop_frame_capable
        assert NTSC_60.is_drop_frame_capable
        assert not FrameRate(30).is_drop_frame_capable
        assert not FrameRate(24000, 1001).is_drop_frame_capable


class TestRounding:
    def test_nearest_rounds_half_away_from_zero(self) -> None:
        # 組み込み round() の偶数丸めだと、隣り合うフレームで丸め方向が変わる
        assert seconds_to_frame(Fraction(1, 60), FrameRate(30), Rounding.NEAREST) == 1
        assert seconds_to_frame(Fraction(3, 60), FrameRate(30), Rounding.NEAREST) == 2

    def test_floor_and_ceil(self) -> None:
        # 切り捨てと切り上げが入れ替わると、その時刻に映っているコマの代わりに次のコマを
        # 指し、再生位置や切った所が 1 フレームずれる
        just_under_two = Fraction(199, 100) / 30
        assert seconds_to_frame(just_under_two, FrameRate(30), Rounding.FLOOR) == 1
        assert seconds_to_frame(just_under_two, FrameRate(30), Rounding.CEIL) == 2


class TestFrameSecondsRoundTrip:
    @given(
        frame=st.integers(min_value=-(10**7), max_value=10**7),
        rate=st.sampled_from(RATES),
    )
    def test_frame_survives_round_trip(self, frame: int, rate: FrameRate) -> None:
        # フレームを秒にして戻すと別のフレームになるなら、秒を介して換算するたびに
        # クリップの位置が 1 フレームずつ動いていく
        seconds = frame_to_seconds(frame, rate)
        assert seconds_to_frame(seconds, rate, Rounding.FLOOR) == frame

    def test_no_drift_over_long_timeline(self) -> None:
        # 29.97fps で 3 時間 浮動小数だとここで確実にずれる
        frame = 30000 * 3 * 3600 // 1001
        assert seconds_to_frame(frame_to_seconds(frame, NTSC_30), NTSC_30) == frame

    def test_seconds_are_exact_rationals(self) -> None:
        # 秒を浮動小数で持つと 29.97fps の 1 フレームが割り切れず、長尺の終盤で音と絵が
        # 少しずつずれていく
        assert frame_to_seconds(1, NTSC_30) == Fraction(1001, 30000)


class TestAudio:
    @given(
        sample=st.integers(min_value=-(10**9), max_value=10**9),
        sample_rate=st.sampled_from(SAMPLE_RATES),
    )
    def test_sample_survives_round_trip(self, sample: int, sample_rate: int) -> None:
        # サンプル番号を秒にして戻すとずれるなら、音の位置を換算するたびに 1 サンプルずつ
        # 動き、クリップの継ぎ目の音が前後へずれる
        seconds = sample_to_seconds(sample, sample_rate)
        assert seconds_to_sample(seconds, sample_rate, Rounding.NEAREST) == sample

    def test_frame_to_sample_at_48k(self) -> None:
        # 30fps / 48kHz はちょうど 1600 サンプルで割り切れる
        assert frame_to_sample(1, FrameRate(30), 48000) == 1600
        assert frame_to_sample(100, FrameRate(30), 48000) == 160000

    def test_frame_to_sample_at_ntsc(self) -> None:
        # 29.97fps / 48000Hz は割り切れない 1 フレーム = 1601.6 サンプル
        assert frame_to_sample(1, NTSC_30, 48000) == 1602
        assert frame_to_sample(5, NTSC_30, 48000) == 8008

    @given(
        frame=st.integers(min_value=0, max_value=10**6),
        sample_rate=st.sampled_from(SAMPLE_RATES),
        rate=st.sampled_from(RATES),
    )
    def test_sample_maps_back_into_same_frame(
        self, frame: int, sample_rate: int, rate: FrameRate
    ) -> None:
        # フレーム先頭のサンプルは、必ず元のフレームか直前のフレームに落ちる
        # 直前になるのは NEAREST が切り上げた場合で、1 サンプル以内のずれ
        sample = frame_to_sample(frame, rate, sample_rate)
        assert sample_to_frame(sample, sample_rate, rate) in (frame - 1, frame)

    def test_rejects_bad_sample_rate(self) -> None:
        # サンプリングレート 0 を通すと、秒からサンプルへの換算がすべて 0 になり、どこを
        # 再生しても頭の音しか読まない
        with pytest.raises(ValueError, match="正でなければ"):
            seconds_to_sample(Fraction(1), 0)


class TestPts:
    def test_round_trip(self) -> None:
        # PTS と秒の往復がずれると、シークした先で目当てと違うフレームを読み、プレビューに
        # 隣のコマが映る
        time_base = Fraction(1, 90000)
        assert seconds_to_pts(pts_to_seconds(12345, time_base), time_base) == 12345

    def test_seek_target_never_overshoots(self) -> None:
        # シーク用途なので、切り上げて目的フレームを飛び越してはいけない
        time_base = Fraction(1, 1000)
        pts = seconds_to_pts(Fraction(1, 3), time_base)
        assert pts_to_seconds(pts, time_base) <= Fraction(1, 3)

    def test_rejects_bad_time_base(self) -> None:
        # タイムベース 0 を黙って通すと、壊れた素材の時刻がすべて 0 秒になり、どこへ
        # シークしても頭の絵が出る
        with pytest.raises(ValueError, match="正でなければ"):
            pts_to_seconds(1, Fraction(0))


class TestTimecode:
    @pytest.mark.parametrize(
        ("frame", "rate", "expected"),
        [
            (0, FrameRate(30), "00:00:00:00"),
            (29, FrameRate(30), "00:00:00:29"),
            (30, FrameRate(30), "00:00:01:00"),
            (30 * 60, FrameRate(30), "00:01:00:00"),
            (30 * 3600, FrameRate(30), "01:00:00:00"),
            (24, FILM, "00:00:01:00"),
            (25 * 61, PAL, "00:01:01:00"),
            (-30, FrameRate(30), "-00:00:01:00"),
        ],
    )
    def test_non_drop(self, frame: int, rate: FrameRate, expected: str) -> None:
        # ノンドロップの桁の繰り上げを誤ると、再生位置や素材の長さの表示が実際と違う
        # 時・分・秒を示す
        assert format_timecode(frame, rate, drop_frame=False) == expected

    @pytest.mark.parametrize(
        ("frame", "expected"),
        [
            (0, "00:00:00;00"),
            (29, "00:00:00;29"),
            (30, "00:00:01;00"),
            # 1 分の境界で 2 番と 3 番が欠番になる
            (1798, "00:00:59;28"),
            (1799, "00:00:59;29"),
            (1800, "00:01:00;02"),
            (1801, "00:01:00;03"),
            # 10 分目は欠番なし
            (17982, "00:10:00;00"),
            # 1 時間ちょうどは 107892 フレーム
            (107892, "01:00:00;00"),
        ],
    )
    def test_drop_frame_29_97(self, frame: int, expected: str) -> None:
        # 欠番の位置を誤ると、29.97fps の作品の再生位置の表示が分の変わり目で存在しない
        # 番号を出したり、10 分ごとに時計とずれたりする
        assert format_timecode(frame, NTSC_30) == expected

    def test_drop_frame_matches_wall_clock(self) -> None:
        # ドロップフレームの存在理由そのもの 1 時間分のフレームが 01:00:00;00 になる
        one_hour_frames = seconds_to_frame(Fraction(3600), NTSC_30)
        assert format_timecode(one_hour_frames, NTSC_30) == "01:00:00;00"

    def test_non_drop_drifts_from_wall_clock(self) -> None:
        # 対比 ノンドロップだと 1 時間で 3.6 秒ずれる これは仕様どおり
        one_hour_frames = seconds_to_frame(Fraction(3600), NTSC_30)
        assert format_timecode(one_hour_frames, NTSC_30, drop_frame=False) == "00:59:56:12"

    @given(frame=st.integers(min_value=0, max_value=30 * 3600 * 4))
    def test_drop_frame_round_trip(self, frame: int) -> None:
        # parse_timecode と format_timecode の欠番の数え方が食い違うと、format_timecode が
        # 出した 29.97fps の文字列を読み返しても元のフレーム番号に戻らない
        assert parse_timecode(format_timecode(frame, NTSC_30), NTSC_30) == frame

    @given(frame=st.integers(min_value=0, max_value=60 * 3600 * 2))
    def test_drop_frame_round_trip_59_94(self, frame: int) -> None:
        # 59.94fps は 1 分ごとに 4 番を飛ばす 2 番のまま数えると、format_timecode が出した
        # 59.94fps の文字列を読み返しても元のフレーム番号に戻らない
        assert parse_timecode(format_timecode(frame, NTSC_60), NTSC_60) == frame

    @given(
        frame=st.integers(min_value=-(10**6), max_value=10**6),
        rate=st.sampled_from([FILM, PAL, FrameRate(30), FrameRate(60)]),
    )
    def test_non_drop_round_trip(self, frame: int, rate: FrameRate) -> None:
        # 負の時刻や 24・25fps で表示と読みが食い違うと、format_timecode が出した文字列を
        # parse_timecode で読み返しても元のフレーム番号に戻らない
        assert parse_timecode(format_timecode(frame, rate), rate) == frame

    def test_separator_signals_drop_frame(self) -> None:
        # 10 分地点で両者は 18 フレーム分ずれる 同じ数字列でも区切りで意味が変わる
        assert parse_timecode("00:10:00;00", NTSC_30) == 17982
        assert parse_timecode("00:10:00:00", NTSC_30) == 18000

    def test_rejects_drop_frame_on_incompatible_rate(self) -> None:
        # 欠番の無いレートでドロップフレームを受けると、表に無いレートを引いて落ちるか、
        # 30fps の作品の時刻を 29.97 の欠番で数えて分ごとにずらす
        with pytest.raises(ValueError, match="ドロップフレーム"):
            format_timecode(0, FrameRate(30), drop_frame=True)
        with pytest.raises(ValueError, match="ドロップフレーム"):
            parse_timecode("00:00:00;00", FrameRate(30))

    @pytest.mark.parametrize(
        "text", ["", "abc", "1:2", "00:00:00", "00:60:00:00", "00:00:60:00", "00:00:00:30"]
    )
    def test_rejects_malformed(self, text: str) -> None:
        # 崩れたタイムコードを 0 や近い値で受けると、呼び手は ValueError で崩れを知れず、
        # 書かれていないフレーム番号を受け取る
        with pytest.raises(ValueError):
            parse_timecode(text, FrameRate(30))
