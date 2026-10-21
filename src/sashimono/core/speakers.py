"""話者の分離（#323）の結果を字幕へ当てる

分離そのもの（どの時刻を誰が話したか）は :mod:`sashimono.asr.speakers` が外の部品で求める
ここは返ってきた区間を、字幕 1 枚ずつの話者（:attr:`TranscriptSegment.speaker`）へ写すだけで、
外の部品も Qt も読まない 試験と AI の道具と字幕パネルが同じ決まりで当てるため

時刻は字幕と同じく素材内のソース秒
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from sashimono.core.model import Transcript

__all__ = [
    "NEAREST_SECONDS",
    "SpeakerTurn",
    "assign_speakers",
    "speaker_name",
    "transcript_speakers",
]

#: 字幕と重なる区間が無いときに、近い区間の話者を付ける幅（秒）
#: 分離は声の弱い頭と終わりを区間から落とすので、短い相づちの字幕が区間の外に出る
#: 付けずに残すと、その 1 枚だけ「話者なし」の別のレイヤーへ焼き込まれる
#: 広げすぎると、無音を挟んだ別の人の字幕を取り違える
NEAREST_SECONDS = Fraction(1)


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """1 人が話している区間 ``speaker`` は分離が振った番号（0 から 意味は回ごとに違う）"""

    start: Fraction
    end: Fraction
    speaker: int


def speaker_name(number: int) -> str:
    """画面と AI に出す話者の名前 番号は 1 から"""
    return f"話者 {number}"


def assign_speakers(transcript: Transcript, turns: Sequence[SpeakerTurn]) -> Transcript:
    """字幕 1 枚ずつへ話者を付けた新しい字幕を返す

    1 枚の途中で話し手が替わるときは、重なりのいちばん長い人を付ける（2026-10-20 本人の決定）
    字幕を割ると枚数と行の ID が変わり、人が直した行や焼き込んだ文字の印が外れる
    重なる区間が無い字幕は :data:`NEAREST_SECONDS` 以内の一番近い区間の人 それも無ければ
    話者を外す（前の回の名前を残すと、新しい回の番号と混ざる）

    名前は字幕に初めて出てきた順に「話者 1」「話者 2」と振り直す 分離の番号は回ごとに
    意味が違い、そのまま使うと 2 人目が「話者 1」になったり番号が飛んだりする
    本文と編集済みの印は変えない（話者は本文ではない）
    """
    chosen: list[int | None] = [
        _best_turn(segment.start, segment.end, turns) for segment in transcript
    ]
    names: dict[int, str] = {}
    for number in chosen:
        if number is not None and number not in names:
            names[number] = speaker_name(len(names) + 1)
    segments = tuple(
        replace(segment, speaker=names[number] if number is not None else None)
        for segment, number in zip(transcript.segments, chosen, strict=True)
    )
    return replace(transcript, segments=segments)


def _best_turn(start: Fraction, end: Fraction, turns: Sequence[SpeakerTurn]) -> int | None:
    """字幕 1 枚に付ける話者の番号 重なりの合計が一番長い人 同じなら先に話し始めた人"""
    spoken: dict[int, Fraction] = {}
    first: dict[int, Fraction] = {}
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > 0:
            spoken[turn.speaker] = spoken.get(turn.speaker, Fraction(0)) + overlap
            first.setdefault(turn.speaker, turn.start)
    if spoken:
        return max(spoken, key=lambda who: (spoken[who], -first[who]))
    nearest: tuple[Fraction, int] | None = None
    for turn in turns:
        gap = turn.start - end if turn.start >= end else start - turn.end
        if gap <= NEAREST_SECONDS and (nearest is None or gap < nearest[0]):
            nearest = (gap, turn.speaker)
    return nearest[1] if nearest is not None else None


def transcript_speakers(transcript: Transcript | None) -> list[str]:
    """字幕に付いている話者の名前 初めて出てきた順"""
    found: list[str] = []
    for segment in transcript or ():
        if segment.speaker and segment.speaker not in found:
            found.append(segment.speaker)
    return found
