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
from sashimono.core.projection import (
    HeardPiece,
    burned_subtitles,
    heard_pieces,
    heard_timeline,
)
from sashimono.core.subtitle_timing import OFFSET_MS_RANGE
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
from sashimono.engine.audio.voice_edges import REACH
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
#: 字幕の行の前後でも聞こえる語と数える幅（秒） 行の時刻を声へ寄せる（#426 前後
#: :data:`REACH` まで）・好みでずらす（:data:`OFFSET_MS_RANGE` まで）と、語が行の表示の外に出る
#: 行の中だけで数えると、音を切っていないのに聞こえている語を残した音に無いと報告する
#: 語は行ごとに別の物なので、この幅の中で鳴れば音は残っている 切って消した語はどこでも鳴らない
_HEARD_SLACK = REACH + Fraction(max(abs(ms) for ms in OFFSET_MS_RANGE), 1000)
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
    checked = unknown = 0
    missing: set[tuple[MediaId, int]] = set()
    if WORD_CUT in wanted:
        cuts, checked, missing, unknown = _word_cuts(project, start, end, waveform_of)
        problems += cuts
    blind = 0
    if SUBTITLE_WORDS in wanted:
        lines, blind = _subtitle_words(project, start, end)
        problems += lines
    sounding = {
        (piece.media_id, piece.stream)
        for piece in heard_pieces(project)
        if piece.start < end and piece.end > start
    }
    # 語の時刻が 1 つも無い作品だけを作品の単位で断る ほかは切れ目と行を 1 つずつ読んだ数で決める
    # 音の単位で断ると、語の無い所に切れ目が無い作品まで調べられなかった扱いになる
    timed = {key for key in sounding if _has_phrases(project, key)}
    if sounding and not timed and not any(_has_subtitles(project, key) for key in sounding):
        # 字幕の無い音だけ 話し声が無いとは言えないので調べたことにしない 足す字幕も無い
        skipped |= wanted
        notes.append(
            "字幕の無い音だけが鳴っているので、語の境目は調べていません"
            " 話している音なら transcribe で起こすと調べられます"
        )
    elif sounding and not timed:
        # 調べたことにしない 語の時刻が無いのに問題 0 件と読むと、切れ目は無事と言ったことになる
        skipped |= wanted
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
