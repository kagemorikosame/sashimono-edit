"""字幕の行の時刻をずらす（#426）

字幕が声より少し早く出る方が読みやすい人も、声が終わってすぐ消えると読み切れない人もいる
どれだけ早く出し遅く消すかは好みなので、行の頭と終わりを別々にずらせるようにする
ずらすのは起こした直後（本人の設定）と、起こした後の字幕パネルの〔時刻を合わせる…〕の 2 か所
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from sashimono.core.model import MediaId, Project, SegmentId, TranscriptSegment
from sashimono.core.projection import burned_subtitles

__all__ = [
    "BURNED_NOTE",
    "MIN_LENGTH",
    "OFFSET_MS_RANGE",
    "SubtitleTiming",
    "burned_copies",
    "retimed",
    "shifted",
]

#: 焼き込んだテキストが元の時刻のまま残ることの案内 字幕パネルと AI の返事で同じ文にする
BURNED_NOTE = (
    "焼き込んだテキスト {count} 個は元の時刻のままです"
    " 合わせた時刻で出すには、テキストを消して焼き込み直してください"
)

#: ずらした後の行の長さの下限（秒） 終わりを早めすぎた行が 0 秒になって見えなくなるのを防ぐ
#: 元からこれより短い行は元の長さまで
MIN_LENGTH = Fraction(1, 10)
#: ずらせる幅（ミリ秒） 早めるのは 1 秒まで それより早いと前の文を話している間に出る
#: 遅く消すのは 2 秒まで 読み切れない人が延ばす幅で、それより長いと次の文の字幕とぶつかる
OFFSET_MS_RANGE = (-1000, 2000)


@dataclass(frozen=True, slots=True)
class SubtitleTiming:
    """起こした字幕の時刻の合わせ方 本人の設定（``Preferences``）から作る"""

    #: 行の頭と終わりを近くの声へ寄せる
    fit_to_voice: bool = True
    #: 行の頭をずらす（ミリ秒 負は早く出す）
    start_offset_ms: int = 0
    #: 行の終わりをずらす（ミリ秒 正は遅く消す）
    end_offset_ms: int = 0

    @property
    def changes(self) -> bool:
        """時刻を動かすか 何もしない形なら行をそのまま返せる"""
        return self.fit_to_voice or bool(self.start_offset_ms) or bool(self.end_offset_ms)

    @property
    def start_offset(self) -> Fraction:
        return Fraction(self.start_offset_ms, 1000)

    @property
    def end_offset(self) -> Fraction:
        return Fraction(self.end_offset_ms, 1000)


def shifted(
    times: Sequence[tuple[Fraction, Fraction]], start_offset: Fraction, end_offset: Fraction
) -> list[tuple[Fraction, Fraction]]:
    """行ごとの ``(頭, 終わり)`` を、頭は ``start_offset``、終わりは ``end_offset`` 秒ずらす

    負は早める 頭は 0 より前へ出さない 元は離れていた行どうしは重ねない（終わりを次の行の頭で
    止める） 重ねると焼き込みで前の字幕が切り詰められ、どちらの行が出ているか分からなくなる
    並びは開始の昇順のまま（頭はどの行も同じだけ動く）

    頭を早めて 0 秒で止まった行どうしは同じ頭になる 次の行の頭で止めると前の行が 0 秒に潰れて
    書き出しから消えるので、そのときだけは重ねて残す（見えないより重なる方がまし）
    """
    starts = [max(Fraction(0), start + start_offset) for start, _ in times]
    out: list[tuple[Fraction, Fraction]] = []
    for index, (start, end) in enumerate(times):
        new_start = starts[index]
        least = new_start + min(end - start, MIN_LENGTH)
        new_end = max(end + end_offset, least)
        if index + 1 < len(times) and end <= times[index + 1][0]:
            limit = starts[index + 1]
            if limit >= least:
                new_end = min(new_end, limit)
        out.append((new_start, new_end))
    return out


def retimed(
    segments: Sequence[TranscriptSegment], times: Sequence[tuple[Fraction, Fraction]]
) -> tuple[TranscriptSegment, ...]:
    """行の時刻だけを差し替える 本文・語の時刻・話者・ID・編集済みの印は変えない

    語の時刻は動かさない 語の時刻は語の境目で切る確かめ（#420）が波形で別に補っていて、
    行と一緒に動かすと 2 重に寄る
    """
    if len(times) != len(segments):
        raise ValueError(f"時刻の数が行の数と違う: {len(times)} と {len(segments)}")
    updated = [
        replace(segment, start=start, end=end)
        for segment, (start, end) in zip(segments, times, strict=True)
    ]
    return tuple(sorted(updated, key=lambda s: (s.start, s.end)))


def burned_copies(
    project: Project, media_id: MediaId, stream: int, segment_ids: Collection[SegmentId]
) -> int:
    """時刻を合わせた行から焼き込んだ、映るテキストのクリップの数

    焼き込んだテキストは字幕の行とは別のクリップで、行の時刻を動かしても付いてこない 数を
    知らせないと、合わせたつもりの字幕が書き出しでは元の時刻で出る ``stream`` は
    :meth:`MediaItem.transcript_stream` でそろえた番号
    """
    wanted = set(segment_ids)
    clips = {
        item.clip.id
        for item in burned_subtitles(project)
        if item.origin.segment_id in wanted
        and item.origin.stream == stream
        and item.origin.media_id in (None, media_id)
    }
    return len(clips)
