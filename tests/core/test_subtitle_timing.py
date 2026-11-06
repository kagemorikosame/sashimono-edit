"""字幕の行の時刻をずらす（#426）と、時刻を付け替える命令"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sashimono.core.commands import RetimeSegments, SetTranscript
from sashimono.core.model import MediaItem, Project, SegmentId, Transcript, TranscriptSegment
from sashimono.core.subtitle_timing import MIN_LENGTH, SubtitleTiming, retimed, shifted

F = Fraction


class TestShifted:
    def test_both_edges_move_by_their_own_amount(self) -> None:
        # 壊れると、早く出す好みが消す時刻まで早め、読み切れなくなる
        assert shifted([(F(2), F(4))], F(-1, 5), F(1, 2)) == [(F(9, 5), F(9, 2))]

    def test_a_line_does_not_run_into_the_next(self) -> None:
        # 遅く消す分で次の行の頭を越えると、焼き込みで前の字幕が切り詰められて欠ける
        out = shifted([(F(1), F(2)), (F(2, 1) + F(1, 10), F(3))], F(0), F(1))
        assert out[0][1] == F(21, 10)

    def test_overlapping_lines_stay_as_they_were(self) -> None:
        # 元から重なっている行（認識が重ねて返した）を縮めると、片方の字幕が消える
        out = shifted([(F(1), F(3)), (F(2), F(4))], F(0), F(1, 2))
        assert out[0] == (F(1), F(7, 2))

    def test_nothing_starts_before_zero(self) -> None:
        # 負の時刻の字幕は作品へ入れられない
        assert shifted([(F(1, 10), F(1))], F(-1), F(0))[0][0] == 0

    def test_a_line_keeps_a_visible_length(self) -> None:
        # 終わりを早めすぎて 0 秒になると、行が見えなくなる
        ((start, end),) = shifted([(F(1), F(2))], F(0), F(-1))
        assert end - start == MIN_LENGTH

    def test_the_order_stays(self) -> None:
        # 並びが崩れると、字幕の不変条件で作品へ入れられない
        out = shifted([(F(1), F(2)), (F(2), F(3)), (F(5), F(6))], F(-1, 2), F(1, 2))
        starts = [start for start, _ in out]
        assert starts == sorted(starts)


class TestRetimed:
    def test_only_times_change(self) -> None:
        # 時刻だけ直したのに編集済みの印や語が変わると、起こし直しで守る行が変わる
        line = TranscriptSegment(F(1), F(2), "文", speaker="話者 1", edited=True)
        (moved,) = retimed([line], [(F(3, 2), F(5, 2))])
        assert (moved.id, moved.text, moved.speaker, moved.edited) == (
            line.id,
            "文",
            "話者 1",
            True,
        )
        assert (moved.start, moved.end) == (F(3, 2), F(5, 2))

    def test_counts_must_match(self) -> None:
        # 数が合わないまま当てると、別の行へ時刻が付く
        with pytest.raises(ValueError, match="数"):
            retimed([TranscriptSegment(F(1), F(2), "文")], [])


def test_timing_offsets_are_seconds() -> None:
    # ミリ秒を秒と取り違えると、200 ms の好みが 200 秒ずれる
    timing = SubtitleTiming(start_offset_ms=-200, end_offset_ms=1500)
    assert (timing.start_offset, timing.end_offset) == (F(-1, 5), F(3, 2))
    assert SubtitleTiming(fit_to_voice=False).changes is False


class TestRetimeSegments:
    def _project(self, project: Project, video_media: MediaItem) -> tuple[Project, Transcript]:
        transcript = Transcript(
            (TranscriptSegment(F(1), F(2), "一"), TranscriptSegment(F(3), F(4), "二"))
        )
        return SetTranscript(video_media.id, transcript).apply(project), transcript

    def test_the_lines_get_their_new_times(self, project: Project, video_media: MediaItem) -> None:
        # 壊れると、窓で合わせた時刻が字幕に入らない
        placed, transcript = self._project(project, video_media)
        first = transcript.segments[0]
        moved = RetimeSegments(video_media.id, ((first.id, F(6, 5), F(19, 10)),)).apply(placed)
        after = moved.require_media(video_media.id).transcript_for(None)
        assert after is not None
        assert (after.segments[0].start, after.segments[0].end) == (F(6, 5), F(19, 10))
        assert after.segments[1] == transcript.segments[1]

    def test_an_unknown_line_is_refused(self, project: Project, video_media: MediaItem) -> None:
        # 黙って飛ばすと、合わせたつもりの行がずれたまま残る
        placed, _ = self._project(project, video_media)
        with pytest.raises(KeyError):
            RetimeSegments(video_media.id, ((SegmentId("無い"), F(1), F(2)),)).apply(placed)

    def test_backwards_times_are_refused(self, project: Project, video_media: MediaItem) -> None:
        # 終わりが頭より前の行を入れると、作品を開き直したときに読めない
        placed, transcript = self._project(project, video_media)
        first = transcript.segments[0]
        with pytest.raises(ValueError, match="逆"):
            RetimeSegments(video_media.id, ((first.id, F(2), F(1)),)).apply(placed)
