"""語の時刻で作品を調べる 言葉の途中の切れ目と、字幕と音の食い違い（#420）

絵も混ぜた音も使わない 作品と素材の字幕と波形から数えられるので、検査の走り係ではなく道具の
答えを作る所（``ai/operations.py``）から呼ぶ 走り係は波形の解析の係を持たず、持たせるには
頼み（:class:`~sashimono.engine.checks.job.CheckRequest`）の比べ方まで変えることになる
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

from sashimono.core.model import MediaId, MediaItem, Project, SegmentId, Word
from sashimono.core.projection import burned_subtitles, heard_pieces, heard_timeline
from sashimono.core.timebase import Rounding, seconds_to_frame
from sashimono.core.words import (
    CUT_IN,
    CutEdge,
    HeardWord,
    cut_edges,
    heard_between,
    heard_words,
    phrases,
    unheard_words,
    word_at,
)
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.audio.word_cuts import (
    JOINED,
    QUIET_DB,
    WORD_EDGE,
    CutPoint,
    cut_points,
    level_at,
)
from sashimono.engine.checks.report import HIGH, MEDIUM, Problem
from sashimono.engine.checks.scan import SUBTITLE_WORDS, WORD_CUT, WORD_KINDS

__all__ = ["WordReport", "check_words"]

#: 切り所を探す幅（秒） 端を動かす量はこの内側 遠い所まで動かすと、切り抜きの中身が変わる
_REACH = Fraction(3, 2)
#: 残した音に無いと知らせる最小の字数 1 字の助詞は語の時刻のずれでも落ちる
_MIN_UNHEARD_CHARS = 2
#: 音の残り端に掛かるだけの字幕と読む長さ（秒） 実物の語の束の長さの中央値（0.6〜0.7 秒）より
#: 短く、1 束も言い切らない 起こした字幕の行は切れ目で自動で切り詰められるので、前後の語の時刻の
#: ずれ（0.1 秒）でも数コマ掛かることがあり、これを返すと直しようが無いまま検査に残り続ける
_SLIVER_SECONDS = Fraction(1, 2)
#: 文言と語を比べるときに外す字 句読点は認識の語と字幕の文言で付き方が違う
_MARKS = frozenset(" 　、\u3002，．！？!?,.…・「」『』（）()\n")

type WaveformOf = Callable[[MediaItem, int], Waveform | None]


@dataclass(frozen=True, slots=True)
class WordReport:
    """語の時刻で調べた結果"""

    problems: tuple[Problem, ...] = ()
    notes: tuple[str, ...] = ()
    #: 調べられなかった種類
    skipped: frozenset[str] = field(default_factory=frozenset)


def _target(edge: CutEdge, point: CutPoint, word: Word) -> Fraction:
    """切り所へ動かすときの素材の秒 語の外へ出す

    谷は語の時刻の前後 0.1 秒を探すので、切り所が語の内側に入り、今の切れ目と同じ値になることが
    ある そのまま返すと動かす先として役に立たない 内側なら近い方の語の端まで出す
    """
    target = point.in_seconds if edge.side == CUT_IN else point.out_seconds
    if word.start < target < word.end:
        return word.end if word.end - target <= target - word.start else word.start
    return target


def _extends(edge: CutEdge, target: Fraction) -> bool:
    """動かすと音を残す側へ伸びるか 伸ばす方が頼まれた言葉を欠かさない"""
    return target <= edge.seconds if edge.side == CUT_IN else target >= edge.seconds


def _suggested(edge: CutEdge, points: list[CutPoint], word: Word) -> CutPoint | None:
    """端を動かす先 声が続く境目より音の下がる境目を、縮める向きより伸ばす向きを選ぶ"""

    def moved(point: CutPoint) -> Fraction:
        return abs(_target(edge, point, word) - edge.seconds)

    near = [point for point in points if moved(point) <= _REACH]
    if not near:
        return None
    return min(
        near,
        key=lambda point: (
            point.kind == JOINED,
            moved(point) * (1 if _extends(edge, _target(edge, point, word)) else 2),
        ),
    )


def _at_placed_edge(project: Project, edge: CutEdge) -> bool:
    """シーンの音の切れ目が、シーンを置いたクリップの端か 端なら ``trim_clip`` で動かせる"""
    for track in project.timeline.tracks:
        if track.id != edge.piece.track_id:
            continue
        for clip in track.clips:
            if clip.id == edge.piece.clip_id:
                placed = clip.timeline_start if edge.side == CUT_IN else clip.timeline_end
                return edge.frame == placed
    return False


def _edge_values(
    project: Project, edge: CutEdge, word: Word, level: float | None, point: CutPoint | None
) -> dict[str, float | int | str | bool]:
    values: dict[str, float | int | str | bool] = {
        "side": edge.side,
        "media_id": str(edge.piece.media_id),
        "source_seconds": round(float(edge.seconds), 3),
        "text": word.text,
        "word_start_seconds": round(float(word.start), 3),
        "word_end_seconds": round(float(word.end), 3),
    }
    movable = edge.piece.inner_clip_id is None or _at_placed_edge(project, edge)
    if edge.piece.inner_clip_id is not None:
        values["scene_clip_id"] = str(edge.piece.inner_clip_id)
        values["inside_scene"] = not movable
    if level is not None:
        values["level_db"] = round(level, 1)
    if point is None:
        return values
    target = _target(edge, point, word)
    near = point.in_seconds if edge.side == CUT_IN else point.out_seconds
    values["suggested_source_seconds"] = round(float(target), 3)
    values["suggested_kind"] = point.kind if target == near else WORD_EDGE
    if movable:
        # 動かすのはコマ単位 今の位置から遠い向きへ丸める 近い向きへ丸めると語の中に残る
        moved = (target - edge.seconds) / edge.piece.speed
        rounding = Rounding.FLOOR if moved < 0 else Rounding.CEIL
        delta = seconds_to_frame(moved, project.rate, rounding)
        values["suggested_head_delta" if edge.side == CUT_IN else "suggested_tail_delta"] = delta
    return values


def _word_cuts(
    project: Project, start: int, end: int, waveform_of: WaveformOf
) -> tuple[list[Problem], int, set[tuple[MediaId, int]]]:
    """言葉の途中の切れ目 戻り値は（問題、調べた端の数、語の時刻が無い音）"""
    problems: list[Problem] = []
    checked = 0
    missing: set[tuple[MediaId, int]] = set()
    words: dict[tuple[MediaId, int], tuple[Word, ...]] = {}
    for edge in cut_edges(project, start, end):
        key = (edge.piece.media_id, edge.piece.stream)
        media = project.find_media(key[0])
        if media is None:
            continue
        if key not in words:
            words[key] = phrases(media.transcript_for(key[1]))
        if not words[key]:
            missing.add(key)
            continue
        checked += 1
        word = word_at(words[key], edge.seconds)
        if word is None:
            continue
        waveform = waveform_of(media, key[1])
        level = level_at(waveform, edge.seconds)
        if level is not None and level <= QUIET_DB:
            # 語の時刻の上では途中でも、音が静かなら語は欠けない 語の時刻のずれのことが多い
            continue
        point = _suggested(edge, cut_points(words[key], waveform, edge.seconds, _REACH), word)
        name = "入り" if edge.side == CUT_IN else "出"
        # 出は残った最後のコマで返す 切った位置のコマは次の音か、作品の終わりの外に当たる
        frame = int(edge.frame) if edge.side == CUT_IN else max(0, math.ceil(edge.frame) - 1)
        detail = (
            f"音の{name}が言葉の途中で切れている（切った所に声が続く）"
            if level is not None
            else f"音の{name}が語の時刻の上で言葉の途中（波形の解析がまだで音は確かめていない）"
        )
        problems.append(
            Problem(
                WORD_CUT,
                HIGH if level is not None else MEDIUM,
                frame,
                frame + 1,
                detail,
                _edge_values(project, edge, word, level, point),
                (str(edge.piece.clip_id),),
            )
        )
    return problems, checked, missing


def _plain(text: str) -> str:
    return "".join(letter for letter in text if letter not in _MARKS)


def _claims(shown: str, missing: list[Word], kept: list[HeardWord]) -> bool:
    """字幕に出す文言が、鳴らない語を書いているか 手で短くした字幕は書いていない

    残した語の字を先に文言から引く 同じ字が繰り返す文で、残した語の字を鳴らない語と数えないため
    """
    plain = _plain(shown)
    for item in kept:
        plain = plain.replace(_plain(item.word.text), "", 1)
    written = 0
    for word in missing:
        letters = _plain(word.text)
        if letters and letters in plain:
            plain = plain.replace(letters, "", 1)
            written += len(letters)
    total = sum(len(_plain(word.text)) for word in missing)
    return written >= _MIN_UNHEARD_CHARS and written * 2 >= total


def _subtitle_problem(
    text: str,
    start: int,
    end: int,
    clip_ids: tuple[str, ...],
    segment_id: str,
    media_id: MediaId,
    missing: list[Word],
    heard: list[HeardWord],
    scene_clip: str | None,
) -> Problem:
    values: dict[str, float | int | str | bool] = (
        {} if scene_clip is None else {"scene_clip_id": scene_clip, "inside_scene": True}
    )
    return Problem(
        SUBTITLE_WORDS,
        MEDIUM,
        start,
        end,
        "字幕の文言の一部が残した音に入っていない 言っていない言葉が字幕に出る",
        {
            "text": text,
            "heard": "".join(item.word.text for item in heard),
            "missing": "".join(word.text for word in missing),
            "segment_id": segment_id,
            "media_id": str(media_id),
            **values,
        },
        clip_ids,
    )


def _owner(project: Project, segment_id: SegmentId, stream: int) -> MediaId | None:
    """素材の印の無い焼き込み字幕の、行を持つ素材 前の版の字幕の印は素材を持たないことがある"""
    for media in project.media:
        transcript = media.transcript_for(stream)
        if transcript is not None and any(s.id == segment_id for s in transcript.segments):
            return media.id
    return None


def _subtitle_words(project: Project, start: int, end: int) -> list[Problem]:
    """字幕の文言と残した音の食い違い 字幕の行が音の切れ目をまたぐ形を見る"""
    heard = heard_words(project)
    problems: list[Problem] = []
    done: set[tuple[str, int, int, str]] = set()

    def look(
        segment_id: str,
        media_id: MediaId,
        stream: int,
        text: str,
        low: int,
        high: int,
        clip: str,
        scene_clip: str | None = None,
    ) -> None:
        media = project.find_media(media_id)
        transcript = None if media is None else media.transcript_for(stream)
        # 同じ行を別の長さや文言で置くことがある 頭だけで見ると、短い方が長い方の検査を止める
        key = (segment_id, low, high, text)
        if transcript is None or key in done:
            return
        done.add(key)
        segment = next((s for s in transcript.segments if str(s.id) == segment_id), None)
        if segment is None or not segment.words:
            return
        missing = unheard_words(segment, heard, media_id, stream, low, high)
        kept = [
            item
            for item in heard_between(heard, media_id, stream, low, high)
            if item.word in segment.words
        ]
        if not missing or not _claims(text, missing, kept):
            return
        # 判定は字幕全体で、返す区間だけ調べた範囲へ切る ほかの問題は範囲の中に収まる
        area = (max(low, start), min(high, end))
        problems.append(
            _subtitle_problem(text, *area, (clip,), segment_id, media_id, missing, kept, scene_clip)
        )

    # 焼き込んだ字幕を先に見る 見る人に出るのは焼き込んだ文字で、出す長さも文言も変えられる
    # 焼き込んだ行の文言は焼き込んだ文字で見る 行の文言と比べ直すと、文字を直しても消えない
    burned: set[str] = set()
    # 描かれない字幕は見る人に届かない シーンの中も置いた窓を通して外の時刻で見る
    for item in burned_subtitles(project):
        origin = item.origin
        media_id = origin.media_id or _owner(project, origin.segment_id, origin.stream)
        source = item.clip.source
        if media_id is None or source is None or item.end <= start or item.start >= end:
            continue
        burned.add(str(origin.segment_id))
        look(
            str(origin.segment_id),
            media_id,
            origin.stream,
            str(source.params.get("text", "")),
            # シーンを速度付きで置くと 1 コマの途中に当たる 映っているコマを全部含むよう広げる
            math.floor(item.start),
            math.ceil(item.end),
            str(item.clip_id),
            str(item.clip.id) if item.nested else None,
        )
    for subtitle in heard_timeline(project):
        if subtitle.media_id is None or subtitle.end_frame <= start or subtitle.start_frame >= end:
            continue
        if not (subtitle.clipped_head or subtitle.clipped_tail):
            continue
        if str(subtitle.segment.id) in burned:
            continue
        shown = subtitle.duration * project.rate.frame_duration
        if shown < _SLIVER_SECONDS:
            continue
        look(
            str(subtitle.segment.id),
            subtitle.media_id,
            subtitle.stream,
            subtitle.segment.text,
            subtitle.start_frame,
            subtitle.end_frame,
            str(subtitle.clip_id),
        )
    return problems


def _has_words(project: Project, key: tuple[MediaId, int]) -> bool:
    media = project.find_media(key[0])
    return media is not None and bool(phrases(media.transcript_for(key[1])))


def _has_subtitles(project: Project, key: tuple[MediaId, int]) -> bool:
    media = project.find_media(key[0])
    transcript = None if media is None else media.transcript_for(key[1])
    return transcript is not None and bool(transcript.segments)


def check_words(
    project: Project, start: int, end: int, kinds: frozenset[str], waveform_of: WaveformOf
) -> WordReport:
    """タイムラインの ``[start, end)`` を語の時刻で調べる"""
    wanted = kinds & WORD_KINDS
    if not wanted:
        return WordReport()
    problems: list[Problem] = []
    notes: list[str] = []
    skipped: set[str] = set()
    checked = 0
    missing: set[tuple[MediaId, int]] = set()
    if WORD_CUT in wanted:
        cuts, checked, missing = _word_cuts(project, start, end, waveform_of)
        problems += cuts
    if SUBTITLE_WORDS in wanted:
        problems += _subtitle_words(project, start, end)
    sounding = {
        (piece.media_id, piece.stream)
        for piece in heard_pieces(project)
        if piece.start < end and piece.end > start
    }
    if not sounding:
        # 素材の音を置いていなければ切れ目も無い 調べられなかったと返すと、文字だけの作品の
        # 検査のたびに語の時刻の案内が付く 焼き込んだ字幕だけが残った問題は返す
        return WordReport(tuple(problems))
    timed = {key for key in sounding if _has_words(project, key)}
    untimed = {key for key in sounding - timed if _has_subtitles(project, key)}
    if not timed:
        # 調べたことにしない 語の時刻が無いのに問題 0 件と読むと、切れ目は無事と言ったことになる
        skipped |= wanted
        notes.append(
            "鳴っている素材に語の時刻が無いので、語の境目と字幕の文言は調べていません"
            " transcribe の words_only で今の字幕に語の時刻を足せます"
        )
    elif untimed:
        # 問題 0 件と返すと、語の時刻の無い音の字幕まで無事と読む 見つけた問題は返す
        skipped |= wanted
        notes.append(
            f"字幕に語の時刻が無い音が {len(untimed)} 本あり、その音の切れ目と字幕は調べていません"
            " transcribe の words_only で足せます"
        )
    elif missing:
        notes.append(
            f"字幕の無い音が {len(missing)} 本あり、その切れ目は調べていません"
            f"（調べた切れ目 {checked} か所）"
        )
    return WordReport(tuple(problems), tuple(notes), frozenset(skipped))
