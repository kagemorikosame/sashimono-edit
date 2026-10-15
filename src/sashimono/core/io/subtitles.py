"""字幕ファイルの書き出し SRT / WebVTT / 素のテキスト

書き出すのは**投影した結果**（:mod:`sashimono.core.projection`）で、素材が持っている
生の起こしではない カットや並べ替えを終えたタイムラインの見た目どおりの時刻が
要るため 素材の時刻をそのまま出すと、編集前の動画にしか合わない字幕になる
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path

from sashimono.core.io.atomic import Expected, write_atomic
from sashimono.core.model import Project
from sashimono.core.projection import ProjectedSubtitle, project_timeline
from sashimono.core.timebase import FrameRate

__all__ = [
    "SUBTITLE_FILTER",
    "SUBTITLE_SUFFIXES",
    "save_subtitles",
    "subtitles_in_range",
    "to_srt",
    "to_text",
    "to_vtt",
]

#: 保存ダイアログのフィルタ
SUBTITLE_FILTER = "SubRip (*.srt);;WebVTT (*.vtt);;テキスト (*.txt)"

#: 書ける字幕の形式（拡張子） 保存ダイアログのフィルタと同じ 3 つ
#: AI の字幕の書き出し（#327）はこれ以外の名前を断る 知らない拡張子を通すと、作品や
#: 素材の名前で字幕の文字を書けてしまう
SUBTITLE_SUFFIXES = (".srt", ".vtt", ".txt")


def to_srt(subtitles: Iterable[ProjectedSubtitle], rate: FrameRate) -> str:
    """SubRip 形式 時刻の区切りはコンマ"""
    blocks = []
    for number, subtitle in enumerate(_prepared(subtitles), start=1):
        start = _timestamp(subtitle.start_frame, rate, ",")
        end = _timestamp(subtitle.end_frame, rate, ",")
        blocks.append(f"{number}\n{start} --> {end}\n{subtitle.segment.text.strip()}\n")
    return "\n".join(blocks)


def to_vtt(subtitles: Iterable[ProjectedSubtitle], rate: FrameRate) -> str:
    """WebVTT 形式 時刻の区切りはピリオド"""
    blocks = ["WEBVTT\n"]
    for subtitle in _prepared(subtitles):
        start = _timestamp(subtitle.start_frame, rate, ".")
        end = _timestamp(subtitle.end_frame, rate, ".")
        blocks.append(f"{start} --> {end}\n{subtitle.segment.text.strip()}\n")
    return "\n".join(blocks)


def to_text(subtitles: Iterable[ProjectedSubtitle]) -> str:
    """本文だけ 台本として読み返すため"""
    lines = [subtitle.segment.text.strip().replace("\n", " ") for subtitle in _prepared(subtitles)]
    return "\n".join(line for line in lines if line) + "\n"


def subtitles_in_range(
    subtitles: Iterable[ProjectedSubtitle], frame_range: tuple[int, int]
) -> Iterator[ProjectedSubtitle]:
    """範囲 ``[start, end)`` にかかる字幕を、範囲の頭を 0 にずらして返す

    書き出し範囲で出した動画は範囲の頭が 0 秒になる 字幕も同じだけずらさないと、
    範囲の頭の分だけ遅れて出る 範囲の端にかかる字幕は端で切る 切らないと 1 枚目が
    負の時刻から始まり、最後の 1 枚が動画の終わりより後ろまで出る
    """
    start, end = frame_range
    for subtitle in subtitles:
        head = max(subtitle.start_frame, start)
        tail = min(subtitle.end_frame, end)
        if tail <= head:
            continue
        yield replace(
            subtitle,
            start_frame=head - start,
            end_frame=tail - start,
            clipped_head=subtitle.clipped_head or head > subtitle.start_frame,
            clipped_tail=subtitle.clipped_tail or tail < subtitle.end_frame,
        )


def save_subtitles(
    project: Project,
    path: Path,
    *,
    frame_range: tuple[int, int] | None = None,
    expect: Expected | None = None,
) -> Path:
    """タイムラインの字幕をファイルへ 形式は拡張子で決める

    ``frame_range`` を渡すと、その範囲だけを頭を 0 にずらして書く（:func:`subtitles_in_range`）
    書き出し範囲で出した動画に合わせるため ``None`` ならタイムライン全体をその時刻のまま
    """
    target = Path(path)
    subtitles = list(project_timeline(project))
    if frame_range is not None:
        subtitles = list(subtitles_in_range(subtitles, frame_range))
    suffix = target.suffix.lower()

    if suffix == ".vtt":
        body = to_vtt(subtitles, project.rate)
    elif suffix == ".txt":
        body = to_text(subtitles)
    else:
        body = to_srt(subtitles, project.rate)

    target.parent.mkdir(parents=True, exist_ok=True)
    # BOM 無しの UTF-8 で書く BOM 付きで書くと、古い再生機で 1 枚目の番号が
    # 読めずに字幕全体が出ないことがある
    # 書いてから置き換える（#327） 開いて上書きすると、書き出し先が元の素材への
    # ハードリンクだったときに素材の中身を書き換え、途中で失敗すると前の字幕も壊れる
    write_atomic(target, body, expect=expect)
    return target


def _prepared(subtitles: Iterable[ProjectedSubtitle]) -> Sequence[ProjectedSubtitle]:
    """時刻順に並べ、本文が空のものを落とす"""
    return sorted(
        (s for s in subtitles if s.segment.text.strip()),
        key=lambda s: (s.start_frame, s.end_frame),
    )


def _timestamp(frame: int, rate: FrameRate, separator: str) -> str:
    """``HH:MM:SS,mmm`` の形へ

    タイムコードではなくミリ秒を使う 字幕ファイルの時刻は実時間で解釈されるので、
    フレーム番号をそのまま持ち込むと 29.97fps でずれる
    """
    total_ms = round(float(frame * rate.frame_duration) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"
