"""語の時刻で作品を調べる 言葉の途中の切れ目と、字幕と音の食い違い（#420）

絵も混ぜた音も使わない 作品と素材の字幕と波形から数えられるので、検査の走り係ではなく道具の
答えを作る所（``ai/operations.py``）から呼ぶ 走り係は波形の解析の係を持たず、持たせるには
頼み（:class:`~sashimono.engine.checks.job.CheckRequest`）の比べ方まで変えることになる
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from sashimono.core.model import MediaId, MediaItem, Project, SegmentId, TranscriptSegment, Word
from sashimono.core.projection import (
    HeardPiece,
    burned_subtitles,
    heard_pieces,
    heard_timeline,
)
from sashimono.core.subtitle_timing import OFFSET_MS_RANGE, SubtitleTiming, shifted
from sashimono.core.timebase import Rounding, seconds_to_frame
from sashimono.core.words import (
    CUT_IN,
    CutEdge,
    HeardWord,
    WordMap,
    cut_edges,
    heard_between,
    heard_words,
    phrases,
    split_seam,
)
from sashimono.engine.audio.voice_edges import (
    REACH,
    fit_to_voice,
    found_edges,
    loudness_of_waveform,
)
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.audio.word_cuts import (
    JOINED,
    QUIET_DB,
    WORD_EDGE,
    CutPoint,
    cut_points,
    level_at,
    read_cut,
)
from sashimono.engine.checks.report import HIGH, MEDIUM, Problem
from sashimono.engine.checks.scan import SUBTITLE_TIMING, SUBTITLE_WORDS, WORD_CUT, WORD_KINDS

__all__ = ["WordReport", "check_words"]

#: 切り所を探す幅（秒） 端を動かす量はこの内側 遠い所まで動かすと、切り抜きの中身が変わる
_REACH = Fraction(3, 2)
#: 残した音に無いと知らせる最小の字数 1 字の助詞は語の時刻のずれでも落ちる
_MIN_UNHEARD_CHARS = 2
#: 音の残り端に掛かるだけの字幕と読む長さ（秒） 実物の語の束の長さの中央値（0.6〜0.7 秒）より
#: 短く、1 束も言い切らない 起こした字幕の行は切れ目で自動で切り詰められるので、前後の語の時刻の
#: ずれ（0.1 秒）でも数コマ掛かることがあり、これを返すと直しようが無いまま検査に残り続ける
_SLIVER_SECONDS = Fraction(1, 2)
#: 字幕の行の前後でも聞こえる語と数える幅（秒） 行の時刻を声へ寄せる（#426 前後
#: :data:`REACH` まで）・好みでずらす（:data:`OFFSET_MS_RANGE` まで）と、語が行の表示の外に出る
#: 行の中だけで数えると、音を切っていないのに聞こえている語を残した音に無いと報告する
#: 語は行ごとに別の物なので、この幅の中で鳴れば音は残っている 切って消した語はどこでも鳴らない
_HEARD_SLACK = REACH + Fraction(max(abs(ms) for ms in OFFSET_MS_RANGE), 1000)
#: 字幕の出る・消える時刻が声からこれだけ離れていたら知らせる（秒 #414） 声へ寄せた後のずれは
#: 合成の声で 9 割が 0.09 秒以内（#426） 読む人が字幕と声のずれに気付くのは 0.2〜0.3 秒から
#: 寄せた後の揺れで知らせないよう、気付く側の端に置いた
TIMING_TOLERANCE = Fraction(1, 4)
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
    """端を動かす先 語を残す向きを先に選び、その中で声が続く境目より音の下がる境目を選ぶ

    伸ばせないときだけ縮め、語の外へ出る最小の量にする 向きを重みで比べると、語の頭の近くの切れ目で
    語を丸ごと落とす提案が勝ち、high の auto の直しとして語が黙って消える 縮める向きで静かな所を
    探すと、間にある別の語まで落とす
    """

    def moved(point: CutPoint) -> Fraction:
        return abs(_target(edge, point, word) - edge.seconds)

    near = [point for point in points if moved(point) <= _REACH]
    if not near:
        return None

    def order(point: CutPoint) -> tuple[bool, bool, Fraction]:
        extends = _extends(edge, _target(edge, point, word))
        return (not extends, extends and point.kind == JOINED, moved(point))

    return min(near, key=order)


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
) -> tuple[list[Problem], int, set[tuple[MediaId, int]], int]:
    """言葉の途中の切れ目

    戻り値は（問題、調べた端の数、字幕の無い音、語で確かめられなかった端の数）
    """
    problems: list[Problem] = []
    checked = 0
    unknown = 0
    missing: set[tuple[MediaId, int]] = set()
    maps: dict[tuple[MediaId, int], WordMap] = {}
    for edge in cut_edges(project, start, end):
        key = (edge.piece.media_id, edge.piece.stream)
        media = project.find_media(key[0])
        if media is None:
            continue
        if key not in maps:
            maps[key] = WordMap(media.transcript_for(key[1]))
        words = maps[key]
        if not words.lines:
            missing.add(key)
            continue
        waveform = waveform_of(media, key[1])
        reading = read_cut(words, waveform, edge.seconds)
        if not reading.known:
            unknown += 1
            continue
        checked += 1
        word = reading.word
        if word is None:
            continue
        level = level_at(waveform, edge.seconds)
        if level is not None and level <= QUIET_DB:
            # 語の時刻の上では途中でも、音が静かなら語は欠けない 語の時刻のずれのことが多い
            continue
        point = _suggested(edge, cut_points(words.phrases, waveform, edge.seconds, _REACH), word)
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
    return problems, checked, missing, unknown


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


def _subtitle_words(project: Project, start: int, end: int) -> tuple[list[Problem], int]:
    """字幕の文言と残した音の食い違い 字幕の行が音の切れ目をまたぐ形を見る

    戻り値は（問題、語の時刻が無く比べられなかった字幕の行の数）
    """
    heard = heard_words(project)
    slack = seconds_to_frame(_HEARD_SLACK, project.rate, Rounding.CEIL)
    problems: list[Problem] = []
    done: set[tuple[str, int, int, str]] = set()
    blind: set[str] = set()

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
        if segment is None:
            return
        if not segment.words:
            blind.add(segment_id)
            return
        # 字幕が出ている間に鳴る音の片だけで数える その片が行の表示の外（寄せた・ずらした分）で
        # 鳴らす語も聞こえる側 別の所に置いた同じ所の音の片まで数えると、そちらで鳴る語のせいで
        # この置き方の欠けを見逃す
        # 分割しただけでつながる片は 1 つの音として数える 割っただけの前の片の語も鳴っている
        near = heard_between(heard, media_id, stream, low - slack, high + slack)
        playing = _joined_pieces(
            [item.piece for item in near], low, high, project.rate.frame_duration
        )
        local = [item for item in near if item.piece in playing]
        sounding = {item.word for item in local}
        missing = [word for word in segment.words if word not in sounding]
        kept = [item for item in local if item.word in segment.words]
        if not missing or not _claims(text, missing, kept):
            return
        # 判定は字幕全体で、返す区間だけ調べた範囲へ切る ほかの問題は範囲の中に収まる
        area = (max(low, start), min(high, end))
        problems.append(
            _subtitle_problem(text, *area, (clip,), segment_id, media_id, missing, kept, scene_clip)
        )

    # 焼き込んだ字幕を先に見る 見る人に出るのは焼き込んだ文字で、出す長さも文言も変えられる
    # 焼き込んだ行の文言は焼き込んだ文字で見る 行の文言と比べ直すと、文字を直しても消えない
    # 行だけで覚えない 同じ行を別の所へ置いた焼き込んでいない字幕まで検査から外れる
    burned: dict[tuple[str, MediaId, int], list[tuple[int, int]]] = {}
    # 描かれない字幕は見る人に届かない シーンの中も置いた窓を通して外の時刻で見る
    for item in burned_subtitles(project):
        origin = item.origin
        media_id = origin.media_id or _owner(project, origin.segment_id, origin.stream)
        source = item.clip.source
        if media_id is None or source is None or item.end <= start or item.start >= end:
            continue
        low, high = math.floor(item.start), math.ceil(item.end)
        burned.setdefault((str(origin.segment_id), media_id, origin.stream), []).append((low, high))
        look(
            str(origin.segment_id),
            media_id,
            origin.stream,
            str(source.params.get("text", "")),
            # シーンを速度付きで置くと 1 コマの途中に当たる 映っているコマを全部含むよう広げる
            low,
            high,
            str(item.clip_id),
            str(item.clip.id) if item.nested else None,
        )
    for subtitle in heard_timeline(project):
        if subtitle.media_id is None or subtitle.end_frame <= start or subtitle.start_frame >= end:
            continue
        if not (subtitle.clipped_head or subtitle.clipped_tail):
            continue
        placed = burned.get((str(subtitle.segment.id), subtitle.media_id, subtitle.stream), [])
        if any(low < subtitle.end_frame and subtitle.start_frame < high for low, high in placed):
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
    return problems, len(blind)


def _joined_pieces(
    pieces: list[HeardPiece], low: int, high: int, frame_seconds: Fraction
) -> set[HeardPiece]:
    """``[low, high)`` に鳴る音の片と、それに分割しただけでつながる片（同じ素材の同じ音）

    片は語の数だけ重ねて渡ってくる 重複を落とさずに総当たりすると、細かく切った作品で
    字幕の行ごとに語の数の 2 乗だけ回る 新しく加わった片の隣だけを探せば、同じ組を見直さない
    """
    rest = set(pieces)
    found = {piece for piece in rest if piece.start < high and piece.end > low}
    rest -= found
    frontier = list(found)
    while frontier and rest:
        joined = {
            piece
            for piece in rest
            if any(
                split_seam(piece, other, frame_seconds) or split_seam(other, piece, frame_seconds)
                for other in frontier
            )
        }
        rest -= joined
        found |= joined
        frontier = list(joined)
    return found


def _has_phrases(project: Project, key: tuple[MediaId, int]) -> bool:
    media = project.find_media(key[0])
    return media is not None and bool(phrases(media.transcript_for(key[1])))


def _has_subtitles(project: Project, key: tuple[MediaId, int]) -> bool:
    media = project.find_media(key[0])
    transcript = None if media is None else media.transcript_for(key[1])
    return transcript is not None and bool(transcript.segments)


@dataclass(slots=True)
class _Shown:
    """タイムラインに映る字幕 1 つ 時刻は実際に映る範囲（タイムラインのフレーム）

    焼き込んでいない行は、同じ行の続けて映る所（クリップを分けただけの所・詰めた所）を
    まとめた物 離れた所に映る同じ行はまとめない（間の関係の無い所まで問題の範囲に入る）
    """

    segment_id: str
    start: Fraction
    end: Fraction
    clip_ids: list[str]
    burned: bool


def _shown_lines(project: Project, start: int, end: int) -> dict[tuple[MediaId, int], list[_Shown]]:
    """``[start, end)`` に映る字幕を、素材と音ごとに集める（焼き込んでいない行と焼き込んだ文字）

    焼き込んだ文字は元の行とは別に動かせる（行を合わせ直しても文字は元の時刻のまま） 元の行で
    見ると、行だけを直して文字が残ったずれを通してしまい、文字だけを動かして合わせた形をずれと読む
    同じ行を焼き込んだ所は、焼き込んでいない行としては数えない（見る人に出るのは文字の方）
    続けて映る所は範囲で絞る前にまとめる 先に絞ると、範囲の頭で切れた後半だけを行の全体と読み、
    声に合った字幕の頭をずれと言う（PR #451 の CodeRabbit の指摘）
    """
    shown: dict[tuple[MediaId, int], list[_Shown]] = {}
    burned: dict[tuple[MediaId, int], list[_Shown]] = {}
    # 素材・音・行ごとの、いちばん後に足した所 長いタイムラインで、行ごとに前の全部を探さない
    latest: dict[tuple[tuple[MediaId, int], str], _Shown] = {}
    #: 焼き込んだ文字を元の行ごとに引く 素材ごとの全部を行ごとに見比べると、行の数の 2 乗になり、
    #: 長い作品で検査の結果を返すのが待たされる（PR #451 の CodeRabbit の指摘）
    by_line: dict[tuple[tuple[MediaId, int], str], list[_Shown]] = {}
    for item in burned_subtitles(project):
        origin = item.origin
        media_id = origin.media_id or _owner(project, origin.segment_id, origin.stream)
        if media_id is None:
            continue
        copy = _Shown(str(origin.segment_id), item.start, item.end, [str(item.clip_id)], True)
        burned.setdefault((media_id, origin.stream), []).append(copy)
        by_line.setdefault(((media_id, origin.stream), copy.segment_id), []).append(copy)
    for subtitle in heard_timeline(project):
        if subtitle.media_id is None:
            continue
        key = (subtitle.media_id, subtitle.stream)
        segment = str(subtitle.segment.id)
        if any(
            copy.start < subtitle.end_frame and subtitle.start_frame < copy.end
            for copy in by_line.get((key, segment), [])
        ):
            continue
        # 開始位置の順に来るので、続けて映る所は直前の同じ行へ足す
        found = latest.get((key, segment))
        if found is None or subtitle.start_frame > found.end:
            found = _Shown(
                segment,
                Fraction(subtitle.start_frame),
                Fraction(subtitle.end_frame),
                [str(subtitle.clip_id)],
                False,
            )
            shown.setdefault(key, []).append(found)
            latest[(key, segment)] = found
            continue
        found.end = max(found.end, Fraction(subtitle.end_frame))
        if str(subtitle.clip_id) not in found.clip_ids:
            found.clip_ids.append(str(subtitle.clip_id))
    for key, copies in burned.items():
        shown.setdefault(key, []).extend(copies)
    return {
        key: inside
        for key, lines in shown.items()
        if (inside := [line for line in lines if line.end > start and line.start < end])
    }


def _placed_with(heard: list[HeardPiece], line: _Shown) -> list[HeardPiece]:
    """字幕 ``line`` の声として比べる配置 同じ素材を離れた所にも置いたとき、そちらで比べない

    焼き込んでいない行は、行を出したクリップの音だけ 焼き込んだ文字は元のクリップと切り離せる
    ので、文字の映る所にいちばん近い配置 どこでも比べると、頭を切った行の頭を離れた配置で
    見つけて、大きなずれと要らない直しを返す（PR #451 の CodeRabbit の指摘）
    """
    if not line.burned:
        return [piece for piece in heard if str(piece.clip_id) in line.clip_ids]
    gaps = [
        (max(Fraction(0), piece.start - line.end, line.start - piece.end), piece) for piece in heard
    ]
    if not gaps:
        return []
    nearest = min(gap for gap, _ in gaps)
    return [piece for gap, piece in gaps if gap == nearest]


def _projected(
    pieces: list[HeardPiece], seconds: Fraction, near: Fraction, project: Project
) -> Fraction | None:
    """素材の秒が鳴るタイムラインのフレーム 同じ所が何度も鳴るときは ``near`` に近い方"""
    frames = [
        piece.frame_at(seconds, project.rate)
        for piece in pieces
        if piece.source_start <= seconds <= piece.source_end
    ]
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame - near))


def _timing_problem(
    low: int,
    high: int,
    early: Fraction | None,
    late: Fraction | None,
    clip_ids: tuple[str, ...],
    values: dict[str, float | int | str | bool],
) -> Problem:
    """比べられた端のずれだけを書いた問題

    比べられなかった端（寄せた所を切った）を 0 と書くと、合っていると読まれる
    """
    # 正は声より遅い 負は声より早い（設定のずらしを足した後の所から 映る時刻の秒）
    # 比べられなかった端（寄せた所を切った）は書かない 0 と書くと、合っていると読む
    offsets: dict[str, float | int | str | bool] = {}
    if early is not None:
        offsets["start_offset_seconds"] = round(float(early), 3)
    if late is not None:
        offsets["end_offset_seconds"] = round(float(late), 3)
    return Problem(
        SUBTITLE_TIMING,
        MEDIUM,
        low,
        high,
        "字幕の出る・消える時刻が声とずれている",
        {**offsets, **values},
        clip_ids,
    )


def _expected_edges(
    segments: Sequence[TranscriptSegment], timing: SubtitleTiming, loudness: np.ndarray
) -> dict[str, tuple[Fraction, Fraction]]:
    """行ごとに、声へ寄せて設定のずらしを足した（あるべき）頭と終わり 行の ID で引く

    行の時刻に設定のずらしが足してあるか（起こしたときや合わせ直したときに足した）どうかは、
    行を見ても分からない 足してあると決めて足したまま寄せると、ずらしが寄せる幅
    （``voice_edges.REACH``）を越えた端は声を見つけられずに残り、もう 1 度ずらしを足して比べる先が
    倍ずれる（PR #451 の Codex の指摘） 足していないと決めて外してから寄せると、足していない行では
    外した先から寄せ直すのでずれの幅が小さく出る
    そこで端ごとに、外した位置と外さない位置の両方から寄せ、声の境目が見つかった方を採る 両方で
    見つかれば動きの小さい方（境目のすぐそばにいた方） どちらも見つからなければ、起こしたときに
    足すのが普通なので外した方を採る
    """
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    raw = [(s.start, s.end) for s in ordered]
    seeds = [shifted(raw, -timing.start_offset, -timing.end_offset), raw]
    if not (timing.start_offset_ms or timing.end_offset_ms):
        seeds = [raw]
    guesses = []
    for times in seeds:
        fitted = fit_to_voice(times, loudness)
        expected = shifted(fitted, timing.start_offset, timing.end_offset)
        found = found_edges(times, loudness)
        guesses.append(list(zip(times, fitted, expected, found, strict=True)))
    out: dict[str, tuple[Fraction, Fraction]] = {}
    for index, segment in enumerate(ordered):
        edges = []
        for side in (0, 1):
            choices = [
                (not found[side], abs(fit[side] - seed[side]), order, expected[side])
                for order, (seed, fit, expected, found) in enumerate(g[index] for g in guesses)
            ]
            edges.append(min(choices)[3])
        out[str(segment.id)] = (edges[0], edges[1])
    return out


def _subtitle_timing(
    project: Project, start: int, end: int, waveform_of: WaveformOf, timing: SubtitleTiming
) -> tuple[list[Problem], int, int]:
    """映る字幕の頭と終わりが、声へ寄せた時刻（#426 と同じ寄せ方）から離れている所

    寄せた時刻は本人の設定のずらし（``timing`` の頭と終わりのミリ秒）も足した物と比べる 足さないと、
    わざと早く出している人の字幕を全部ずれと読む 比べるのはタイムラインの上（映る時刻） 素材の秒で
    比べると、速度を変えたクリップのずれ（半速なら倍）を見落とす
    戻り値は（問題、波形が無く比べられなかった所の数、寄せた頭か終わりを切って比べられなかった行の数）
    """
    problems: list[Problem] = []
    unknown = unplaced = 0
    frame = project.rate.frame_duration
    pieces = heard_pieces(project)
    for key, lines in _shown_lines(project, start, end).items():
        media_id, stream = key
        media = project.find_media(media_id)
        transcript = None if media is None else media.transcript_for(stream)
        if media is None or transcript is None or not transcript.segments:
            continue
        waveform = waveform_of(media, stream)
        if waveform is None:
            unknown += len(lines)
            continue
        fitted = _expected_edges(transcript.segments, timing, loudness_of_waveform(waveform))
        known = [s.index for s in media.audio_streams]
        common: dict[str, float | int | str | bool] = {
            "media_id": str(media_id),
            # 直す道具（retime_subtitles）の audio と同じ 1 から数えた番号 省くと 1 本目の字幕を探す
            "audio": known.index(stream) + 1 if stream in known else 1,
            "start_offset_ms": timing.start_offset_ms,
            "end_offset_ms": timing.end_offset_ms,
        }
        edited = {str(s.id): s.edited for s in transcript.segments}
        heard = [piece for piece in pieces if (piece.media_id, piece.stream) == key]
        for line in lines:
            target = fitted.get(line.segment_id)
            if target is None:
                continue
            placed = _placed_with(heard, line)
            head = _projected(placed, target[0], line.start, project)
            tail = _projected(placed, target[1], line.end, project)
            if head is None or tail is None:
                # 寄せた端を切って鳴らない 比べられない端を、合っていると読まない 片端でも数える
                # （残った端が合っていても、切った側の字幕が声の無い所に映っているかもしれない）
                unplaced += 1
            if head is None and tail is None:
                continue
            early = None if head is None else (line.start - head) * frame
            late = None if tail is None else (line.end - tail) * frame
            if max(abs(x) for x in (early, late) if x is not None) < TIMING_TOLERANCE:
                continue
            values: dict[str, float | int | str | bool] = {
                "segment_id": line.segment_id,
                "burned": line.burned,
                "edited": False if line.burned else edited.get(line.segment_id, False),
            }
            problems.append(
                _timing_problem(
                    max(math.floor(line.start), start),
                    min(math.ceil(line.end), end),
                    early,
                    late,
                    tuple(line.clip_ids),
                    {**common, **values},
                )
            )
    problems.sort(key=lambda problem: (problem.start, problem.end))
    return problems, unknown, unplaced


def check_words(
    project: Project,
    start: int,
    end: int,
    kinds: frozenset[str],
    waveform_of: WaveformOf,
    timing: SubtitleTiming | None = None,
) -> WordReport:
    """タイムラインの ``[start, end)`` を語の時刻で調べる

    ``timing`` は本人の字幕の時刻の設定（#426） 声へ寄せない人の字幕は、時刻のずれを調べない
    """
    wanted = kinds & WORD_KINDS
    if not wanted:
        return WordReport()
    problems: list[Problem] = []
    notes: list[str] = []
    skipped: set[str] = set()
    checked = unknown = 0
    missing: set[tuple[MediaId, int]] = set()
    if WORD_CUT in wanted:
        cuts, checked, missing, unknown = _word_cuts(project, start, end, waveform_of)
        problems += cuts
    blind = 0
    if SUBTITLE_WORDS in wanted:
        lines, blind = _subtitle_words(project, start, end)
        problems += lines
    if SUBTITLE_TIMING in wanted:
        chosen = timing if timing is not None else SubtitleTiming()
        if chosen.fit_to_voice:
            late, deaf, unplaced = _subtitle_timing(project, start, end, waveform_of, chosen)
            problems += late
            if deaf:
                # 比べられなかった行を問題 0 件へ混ぜない
                skipped.add(SUBTITLE_TIMING)
                notes.append(
                    f"波形の解析がまだの音の字幕が {deaf} 行あり、出る時刻と声を比べていません"
                    " 解析が終わってから調べ直してください"
                )
            if unplaced:
                skipped.add(SUBTITLE_TIMING)
                notes.append(
                    f"声の頭か終わりを切った所に映る字幕が {unplaced} 行あり、切った側の端は"
                    "声と比べていません（残った端は比べています） 字幕が声の無い所に映って"
                    "いないかを preview_frame で確かめてください"
                )
        else:
            skipped.add(SUBTITLE_TIMING)
            notes.append(
                "本人の設定で字幕の時刻を声に合わせない（起こした字幕の時刻を声に合わせる が切）"
                "ので、字幕の時刻は調べていません"
            )
    sounding = {
        (piece.media_id, piece.stream)
        for piece in heard_pieces(project)
        if piece.start < end and piece.end > start
    }
    # 語の時刻が 1 つも無い作品だけを作品の単位で断る ほかは切れ目と行を 1 つずつ読んだ数で決める
    # 音の単位で断ると、語の無い所に切れ目が無い作品まで調べられなかった扱いになる
    timed = {key for key in sounding if _has_phrases(project, key)}
    # 語の時刻で調べる種類 字幕の時刻（#414）は語ではなく行と波形で見るので、語の有無で断らない
    worded = wanted & {WORD_CUT, SUBTITLE_WORDS}
    if (
        worded
        and sounding
        and not timed
        and not any(_has_subtitles(project, key) for key in sounding)
    ):
        # 字幕の無い音だけ 話し声が無いとは言えないので調べたことにしない 足す字幕も無い
        skipped |= worded
        notes.append(
            "字幕の無い音だけが鳴っているので、語の境目は調べていません"
            " 話している音なら transcribe で起こすと調べられます"
        )
    elif worded and sounding and not timed:
        # 調べたことにしない 語の時刻が無いのに問題 0 件と読むと、切れ目は無事と言ったことになる
        skipped |= worded
        notes.append(
            "鳴っている素材に語の時刻が無いので、語の境目と字幕の文言は調べていません"
            " transcribe の words_only で今の字幕に語の時刻を足せます"
        )
    elif missing:
        notes.append(
            f"字幕の無い音が {len(missing)} 本あり、その切れ目は調べていません"
            f"（調べた切れ目 {checked} か所）"
        )
    if unknown and WORD_CUT not in skipped:
        # 問題 0 件と返すと、語の途中かを読めなかった切れ目まで無事と読む 見つけた問題は返す
        skipped.add(WORD_CUT)
        notes.append(
            f"語で確かめられない切れ目が {unknown} か所あり、言葉の途中かを見ていません"
            " 語の時刻の無い所（字幕の無い所も）で音が続くか、波形の解析がまだの所です"
            " transcribe の words_only で足すか、その所は人が聞いて確かめてください"
        )
    if blind and SUBTITLE_WORDS not in skipped:
        # 音の無い所の焼き込みも同じ 比べられなかった行を問題 0 件へ混ぜない
        skipped.add(SUBTITLE_WORDS)
        notes.append(
            f"語の時刻が無い字幕の行が {blind} 行あり、文言と音を比べていません"
            " transcribe の words_only で足せます"
        )
    return WordReport(tuple(problems), tuple(notes), frozenset(skipped))
