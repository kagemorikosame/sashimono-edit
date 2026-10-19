"""字幕ファイルの書き出し

出すのは投影後の時刻 素材が持っている生の時刻をそのまま出すと、編集前の動画に
しか合わない字幕になる
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands import AddClip, RippleCut, SetTranscript
from sashimono.core.io import save_subtitles, to_srt, to_text, to_vtt
from sashimono.core.model import MediaItem, Project, Transcript
from sashimono.core.projection import project_timeline
from sashimono.core.timebase import FrameRate
from tests.conftest import make_clip


@pytest.fixture
def placed(project: Project, video_media: MediaItem, transcript: Transcript) -> Project:
    with_transcript = SetTranscript(video_media.id, transcript).apply(project)
    track = with_transcript.timeline.tracks[0]
    return AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)


class TestSrt:
    def test_numbering_and_timestamps(self, placed: Project) -> None:
        # SRT の番号や時刻の書き方が崩れると、動画サイトや再生ソフトが字幕を読めないか、
        # 話と違う時刻に出す
        srt = to_srt(project_timeline(placed), placed.rate)
        assert srt.startswith("1\n00:00:01,000 --> 00:00:03,000\n今日は\n")
        assert "3\n00:00:07,000 --> 00:00:09,000\n作ります\n" in srt

    def test_the_cut_moves_the_times(self, placed: Project) -> None:
        # 冒頭 1 秒を切ると、字幕もその分だけ前へ来る 書き出しに追従処理は要らない
        cut = RippleCut(((0, 30),)).apply(placed)
        srt = to_srt(project_timeline(cut), cut.rate)
        assert srt.startswith("1\n00:00:00,000 --> 00:00:02,000\n今日は\n")

    def test_empty_text_is_skipped(self, project: Project, video_media: MediaItem) -> None:
        # 空白だけの字幕まで書くと、字幕の書き出しで出した SRT に本文の無いコマが番号付きで混ざる
        from sashimono.core.model import TranscriptSegment

        blank = Transcript(
            (
                TranscriptSegment(Fraction(0), Fraction(1), "  "),
                TranscriptSegment(Fraction(2), Fraction(3), "本題"),
            )
        )
        placed = SetTranscript(video_media.id, blank).apply(project)
        placed = AddClip(placed.timeline.tracks[0].id, make_clip(0, 300, video_media)).apply(placed)
        assert to_srt(project_timeline(placed), placed.rate).count("-->") == 1

    def test_no_subtitles_gives_an_empty_string(self, project: Project) -> None:
        # 字幕の無い作品で何か返すと、字幕の書き出し（メニュー・字幕パネル・AI）がそのまま書き、
        # 中身の無いはずの SRT に字幕でない行が入る
        assert to_srt(project_timeline(project), project.rate) == ""


class TestVtt:
    def test_header_and_period_separator(self, placed: Project) -> None:
        # WEBVTT の見出しや時刻のピリオドが崩れると、ブラウザの動画再生が VTT として読まず
        # 字幕が 1 枚も出ない
        vtt = to_vtt(project_timeline(placed), placed.rate)
        assert vtt.startswith("WEBVTT\n")
        assert "00:00:01.000 --> 00:00:03.000" in vtt


class TestText:
    def test_only_the_body_survives(self, placed: Project) -> None:
        # 文章として書き出したときに番号や時刻が混ざると、文字起こしをそのまま台本や記事に使えない
        assert to_text(project_timeline(placed)) == "今日は\n編集ソフトを\n作ります\n"

    def test_line_breaks_inside_a_subtitle_become_spaces(
        self, project: Project, video_media: MediaItem
    ) -> None:
        # 2 行に折った字幕が文章の書き出しでも 2 行に割れると、1 行 1 字幕の並びが崩れて
        # 1 文が別々に見える
        from sashimono.core.model import TranscriptSegment

        wrapped = Transcript((TranscriptSegment(Fraction(0), Fraction(2), "上の行\n下の行"),))
        placed = SetTranscript(video_media.id, wrapped).apply(project)
        placed = AddClip(placed.timeline.tracks[0].id, make_clip(0, 300, video_media)).apply(placed)
        assert to_text(project_timeline(placed)) == "上の行 下の行\n"


class TestTimestamps:
    def test_fractional_frame_rates_use_real_time(self, placed: Project) -> None:
        # 29.97fps では、フレーム番号をそのまま秒にするとずれる ミリ秒で出す
        from dataclasses import replace

        ntsc_rate = FrameRate(30000, 1001)
        ntsc = replace(
            placed,
            settings=replace(placed.settings, frame_rate=ntsc_rate),
            timeline=replace(placed.timeline, rate=ntsc_rate),
        )
        srt = to_srt(project_timeline(ntsc), ntsc.rate)
        # 素材の 1 秒は 29 フレーム目（切り捨て） その実時間は 29 x 1001/30000 で
        # 0.968 秒 フレーム番号を 30 で割った 0.967 秒でも、1.000 秒でもない
        assert "00:00:00,968 --> " in srt


class TestSaveSubtitles:
    def test_the_extension_decides_the_format(self, placed: Project, tmp_path: Path) -> None:
        # 選んだ拡張子と中身の形式が食い違うと、.vtt なのに SRT の中身が入り、
        # 再生ソフトが字幕を読めない
        srt = save_subtitles(placed, tmp_path / "a.srt").read_text(encoding="utf-8")
        assert srt.startswith("1\n")
        assert (
            save_subtitles(placed, tmp_path / "a.vtt")
            .read_text(encoding="utf-8")
            .startswith("WEBVTT")
        )
        assert save_subtitles(placed, tmp_path / "a.txt").read_text(encoding="utf-8") == (
            "今日は\n編集ソフトを\n作ります\n"
        )

    def test_an_unknown_extension_falls_back_to_srt(self, placed: Project, tmp_path: Path) -> None:
        # 拡張子を付け間違えただけで保存に失敗すると、文字起こしの結果を外へ出せない
        # 読める SRT で書いておけば名前を直すだけで使える
        written = save_subtitles(placed, tmp_path / "a.unknown")
        assert written.read_text(encoding="utf-8").startswith("1\n")

    def test_written_without_a_byte_order_mark(self, placed: Project, tmp_path: Path) -> None:
        # BOM 付きだと、古い再生機で 1 枚目の番号が読めず字幕全体が出ないことがある
        raw = save_subtitles(placed, tmp_path / "a.srt").read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")


class TestFrameRange:
    """書き出し範囲（#140）で動画を出したときに、字幕を同じ所へ合わせる（#141）

    動画は範囲の頭が 0 秒になる 字幕がタイムラインの時刻のままだと、範囲の頭の分だけ
    字幕が遅れて出る 範囲の外の字幕も、映っていない所の文として付いてくる
    """

    def test_the_head_of_the_range_becomes_zero(self, placed: Project, tmp_path: Path) -> None:
        # 範囲 [90, 300) は 3 秒から 範囲の頭を 0 にずらさないと、4 秒の字幕が 4 秒に出て
        # 範囲だけの動画では 3 秒遅れる
        srt = save_subtitles(placed, tmp_path / "a.srt", frame_range=(90, 300)).read_text(
            encoding="utf-8"
        )
        assert srt.startswith("1\n00:00:01,000 --> 00:00:03,000\n編集ソフトを\n")

    def test_subtitles_outside_the_range_are_left_out(
        self, placed: Project, tmp_path: Path
    ) -> None:
        # 範囲の外の字幕まで書くと、映っていない所の文が 0 秒より前や動画の後ろに付く
        text = save_subtitles(placed, tmp_path / "a.txt", frame_range=(100, 200)).read_text(
            encoding="utf-8"
        )
        assert text == "編集ソフトを\n"

    def test_a_subtitle_crossing_an_edge_is_cut_at_the_edge(
        self, placed: Project, tmp_path: Path
    ) -> None:
        # 「今日は」は 30〜90、「作ります」は 210〜270 範囲 [45, 225) の端で切らないと、
        # 1 枚目が負の時刻から始まり、最後の 1 枚が動画の終わりより後ろまで出る
        srt = save_subtitles(placed, tmp_path / "a.srt", frame_range=(45, 225)).read_text(
            encoding="utf-8"
        )
        assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\n今日は\n")
        assert "3\n00:00:05,500 --> 00:00:06,000\n作ります\n" in srt

    def test_no_range_writes_the_whole_timeline(self, placed: Project, tmp_path: Path) -> None:
        # 範囲を渡さない呼び出し（これまでどおりの書き出し）は時刻を動かさない
        srt = save_subtitles(placed, tmp_path / "a.srt", frame_range=None).read_text(
            encoding="utf-8"
        )
        assert srt.startswith("1\n00:00:01,000 --> 00:00:03,000\n今日は\n")
        assert srt.count("-->") == 3
