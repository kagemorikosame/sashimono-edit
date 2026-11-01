"""語の時刻を波形で補い、切ってよい位置を探す（#420）"""

from __future__ import annotations

from fractions import Fraction

from sashimono.core.model import Word
from sashimono.core.words import phrases
from sashimono.engine.audio.word_cuts import (
    JOINED,
    PAUSE,
    QUIET_DB,
    UNCHECKED,
    VALLEY,
    cut_points,
    level_at,
)
from tests.words_helpers import sample_transcript, wave

_WORDS = phrases(sample_transcript())


class TestCutPoints:
    def test_a_pause(self) -> None:
        """話の間は入りを後ろの語の手前へ、出を前の語の後ろへ寄せる

        間の真ん中で切ると、切り抜きの頭と尻に長い無音が残る
        """
        points = cut_points(_WORDS, wave(), Fraction(3), Fraction(2))
        point = next(p for p in points if p.after == "はい")
        assert point.kind == PAUSE
        assert point.before == "作ります"
        assert Fraction(39, 10) <= point.in_seconds <= Fraction(4)
        assert Fraction(26, 10) <= point.out_seconds <= Fraction(27, 10)
        assert point.quiet_seconds > 1
        assert point.level_db is not None and point.level_db < QUIET_DB

    def test_the_word_time_is_corrected_by_the_valley(self) -> None:
        """語の時刻より 0.07 秒後ろに本当の谷がある形

        語の時刻のまま切ると「作」の頭を前の片に残す
        """
        sound = wave(((0.0, 1.0), (1.86, 1.9), (2.6, 12.0)))
        point = next(
            p
            for p in cut_points(_WORDS, sound, Fraction(18, 10), Fraction(1))
            if p.before == "カメラを"
        )
        assert point.kind == VALLEY
        assert Fraction(185, 100) <= point.in_seconds <= Fraction(195, 100)

    def test_a_boundary_inside_steady_speech(self) -> None:
        # 声が途切れない語の境目も候補として返す 代わりに joined と分かるようにする
        sound = wave(((0.0, 1.0), (2.6, 12.0)))
        point = next(
            p
            for p in cut_points(_WORDS, sound, Fraction(18, 10), Fraction(1))
            if p.before == "カメラを"
        )
        assert point.kind == JOINED
        assert point.quiet_seconds == 0

    def test_a_single_dip_is_not_a_valley(self) -> None:
        """波の 0 交差の 1 点を谷と読むと、声の途中を切り所として勧める"""
        sound = wave(((0.0, 1.0), (1.8, 1.81), (2.6, 12.0)))
        point = next(
            p
            for p in cut_points(_WORDS, sound, Fraction(18, 10), Fraction(1))
            if p.before == "カメラを"
        )
        assert point.kind == JOINED

    def test_without_a_waveform_only_the_word_times(self) -> None:
        # 壊れると、波形の解析が終わるまで切り所を読めない
        points = cut_points(_WORDS, None, Fraction(18, 10), Fraction(1))
        assert {point.kind for point in points} == {UNCHECKED}
        assert points[0].in_seconds == Fraction(9, 5)
        assert points[0].level_db is None

    def test_nearest_first_and_within_the_window(self) -> None:
        # 壊れると、遠い切り所を先に勧め、頼まれた言葉を落とす
        points = cut_points(_WORDS, wave(), Fraction(19, 10), Fraction(1))
        assert points[0].before == "カメラを"
        # 窓の端にかかる話の頭（1.0 秒）と間（2.6〜4.0 秒）も返す
        assert len(points) == 3
        assert cut_points(_WORDS, wave(), Fraction(9), Fraction(1)) == []

    def test_the_points_are_cut_short(self) -> None:
        # 長い作品でも返す数は増えない 長く返すと会話の費用だけが増える
        many = tuple(
            Word(Fraction(i, 2), Fraction(i, 2) + Fraction(2, 5), "語") for i in range(1000)
        )
        assert len(cut_points(many, None, Fraction(100), Fraction(10))) == 6


class TestLevelAt:
    def test_loud_and_quiet(self) -> None:
        # 壊れると、声の途中と静かな所を見分けられない
        sound = wave()
        loud = level_at(sound, Fraction(14, 10))
        quiet = level_at(sound, Fraction(3))
        assert loud is not None and loud > QUIET_DB
        assert quiet is not None and quiet < QUIET_DB

    def test_no_waveform(self) -> None:
        # 壊れると、波形がまだの素材で検査が落ちる
        assert level_at(None, Fraction(1)) is None

    def test_a_silent_material_reads_quiet(self) -> None:
        """全部静かな素材を話の大きさの基準にすると、雑音を声と読む"""
        sound = wave(((0.0, 12.0),))
        level = level_at(sound, Fraction(1))
        assert level is not None
        assert level <= QUIET_DB

    def test_past_the_end(self) -> None:
        # 壊れると、素材の終わりより後を見た所で検査が落ちる
        assert level_at(wave(seconds=2.0), Fraction(9)) is None
