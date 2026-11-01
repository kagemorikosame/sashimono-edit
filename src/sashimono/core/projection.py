"""素材に紐付いた字幕を、タイムライン上の位置へ投影する

字幕はソース時刻でしか持っていない タイムライン上のどこに出るかは、その素材を
参照しているクリップごとに毎回ここで計算する

この方式にしているのは「同期を取らない」ため 字幕にタイムライン位置を持たせると、
カット・トリム・移動・速度変更・複製のすべてに追従処理が必要になり、どれか 1 つ
漏れた瞬間にずれる 位置を持たせなければ、ずれようがない
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from fractions import Fraction

from sashimono.core.model import (
    Clip,
    ClipId,
    MediaId,
    MediaItem,
    Project,
    Timeline,
    Track,
    TrackId,
    TrackKind,
    TranscriptSegment,
    heard_stream,
)
from sashimono.core.timebase import FrameRate, Rounding, seconds_to_frame

__all__ = [
    "HeardPiece",
    "ProjectedSubtitle",
    "heard_pieces",
    "heard_timeline",
    "project_clip",
    "project_timeline",
    "sounding",
    "source_seconds_at",
    "subtitle_stream",
    "subtitle_voice",
]


def subtitle_stream(project: Project, track: Track, clip: Clip) -> int | None:
    """クリップに出す字幕の音声ストリーム 音を鳴らすクリップはその音、絵だけのクリップは 1 本目

    絵だけのクリップ（分けて置いた映像）は前と同じく 1 本目の字幕を出す
    """
    if not project.plays_sound(track, clip):
        return None
    return heard_stream(track, clip)


def subtitle_voice(project: Project, track: Track, clip: Clip, media: MediaItem) -> int:
    """クリップに出す字幕の音声を、素材の字幕の番号（:meth:`MediaItem.transcript_stream`）で

    字幕パネル・無音カット・分割の時刻など「どのクリップがこの音のクリップか」を決める所は
    すべてこれで比べる 鳴らす音はトラックの種類で変わる（音声トラックは stream_index、
    混合は audio_stream、映像トラックは鳴らさないので 1 本目） クリップの番号だけを見たり、
    音を見ずに最初のクリップを使ったりすると、別の音の位置で字幕を割り、無音を切る
    """
    return media.transcript_stream(subtitle_stream(project, track, clip))


def source_seconds_at(
    project: Project, media: MediaItem, stream: int | None, frame: int
) -> Fraction | None:
    """タイムラインのフレームを、素材の ``stream`` の音のソース秒へ

    その音のクリップが再生ヘッドの所に無ければ ``None``

    字幕パネルの〔再生ヘッドで分割〕と AI の字幕の分割（#349）が同じ時刻で割るよう、1 か所に置く
    見ている音の字幕を出すクリップから取る 最初に当たるクリップを使うと、リンクを外して
    音声 2 だけを切り詰めたとき、音声 1 のクリップの時刻で音声 2 の字幕を割る（PR #231）
    """
    wanted = media.transcript_stream(stream)
    for track in project.timeline.tracks:
        for clip in track.clips:
            if clip.media_id != media.id or not clip.contains(frame):
                continue
            if subtitle_voice(project, track, clip, media) != wanted:
                continue
            elapsed = (frame - clip.timeline_start) * project.rate.frame_duration
            return clip.source_in + elapsed * clip.speed
    return None


@dataclass(frozen=True, slots=True)
class ProjectedSubtitle:
    """タイムライン上に現れた字幕 1 枚"""

    segment: TranscriptSegment
    clip_id: ClipId
    track_id: TrackId
    #: タイムライン上の表示範囲（フレーム） ``end_frame`` は含まない
    start_frame: int
    end_frame: int
    #: クリップの端で切り詰められたか UI で「続きがある」表示に使う
    clipped_head: bool = False
    clipped_tail: bool = False
    #: 字幕の出どころ（素材と、:meth:`MediaItem.transcript_stream` でそろえた音声の番号）
    #: 置いたシーンの中の字幕は ``clip_id`` が外側のシーンのクリップに書き換わるので、
    #: そこから素材をたどれない 出どころを持たせないと、焼き込みでシーンの中の別々の
    #: 話し手が 1 本にまとまって欠けた（PR #231 の指摘）
    media_id: MediaId | None = None
    stream: int = 0

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame


def project_clip(
    clip: Clip,
    media: MediaItem,
    rate: FrameRate,
    track_id: TrackId,
    stream: int | None = None,
) -> Iterator[ProjectedSubtitle]:
    """1 つのクリップに現れる字幕を返す

    クリップが使っているソース範囲に重なるセグメントだけが対象で、はみ出した分は
    クリップの端で切り詰められる ``stream`` はクリップが鳴らす音声ストリーム
    （``None`` と素材に無い番号は 1 本目） 音ごとに分けて置いたクリップには、その音の
    字幕だけが出る
    """
    transcript = media.transcript_for(stream)
    if transcript is None:
        return

    source_in = clip.source_in
    source_out = clip.source_out(rate)

    for segment in transcript.overlapping(source_in, source_out):
        visible_start = max(segment.start, source_in)
        visible_end = min(segment.end, source_out)

        start_offset = _source_to_clip_frame(visible_start, clip, rate, Rounding.FLOOR)
        end_offset = _source_to_clip_frame(visible_end, clip, rate, Rounding.CEIL)

        start_offset = max(0, min(start_offset, clip.duration - 1))
        # 表示は最低 1 フレーム 丸めの結果 0 フレームになると画面に出ない
        end_offset = max(start_offset + 1, min(end_offset, clip.duration))

        yield ProjectedSubtitle(
            segment=segment,
            clip_id=clip.id,
            track_id=track_id,
            start_frame=clip.timeline_start + start_offset,
            end_frame=clip.timeline_start + end_offset,
            clipped_head=segment.start < source_in,
            clipped_tail=segment.end > source_out,
            media_id=media.id,
            stream=media.transcript_stream(stream),
        )


def project_timeline(project: Project) -> Iterator[ProjectedSubtitle]:
    """タイムライン全体に現れる字幕を、開始位置順に返す

    同じ素材を複数回置けば、字幕もその回数だけ現れる これは意図した挙動で、
    素材を使い回したときに字幕が片方にしか出ないことの方が驚きが大きい

    同じ時刻に同じ字幕が出る物は 1 つにする リンクした映像と音声はどちらも同じ素材の同じ所を
    指すので、そのままだと字幕の一覧も字幕のファイルも同じ行が 2 回ずつ並ぶ（#413）
    """
    projected = list(_project_tracks(project, project.timeline, 0))
    projected.sort(key=lambda p: (p.start_frame, p.end_frame))
    seen: set[tuple[object, ...]] = set()
    for subtitle in projected:
        key = (
            subtitle.media_id,
            subtitle.stream,
            subtitle.segment.id,
            subtitle.start_frame,
            subtitle.end_frame,
        )
        if key not in seen:
            seen.add(key)
            yield subtitle


def heard_timeline(project: Project) -> Iterator[ProjectedSubtitle]:
    """書き出しで声が鳴るクリップの字幕だけを、開始位置順に返す（#322）

    シーンの中も、ミキサと同じ決まり（:func:`sounding`）でトラックとクリップを選ぶ
    投影した後の字幕は ``clip_id`` が外側のシーンのクリップに書き換わるので、後から
    鳴るかを照らすと、シーンの中で消音したトラックや無効のクリップの字幕まで通る
    （#385 の Codex の指摘）
    """
    projected = list(_project_tracks(project, project.timeline, 0, heard=True))
    projected.sort(key=lambda p: (p.start_frame, p.end_frame))
    yield from projected


#: シーンの入れ子をたどる深さの上限 描画と音（MAX_SCENE_DEPTH）に揃える
MAX_SCENE_DEPTH = 8


@dataclass(frozen=True, slots=True)
class HeardPiece:
    """書き出しで鳴る素材の音のひと続き シーンの中も、置いた所の外の時刻まで戻した物

    フレームは分数で持つ シーンを速度付きで置くと 1 コマの途中に当たるので、整数へ丸めると
    元の時刻へ戻すときに速度の分だけ（2 倍速なら 2 コマ分）語の判定がずれる
    """

    media_id: MediaId
    #: :meth:`MediaItem.transcript_stream` でそろえた音声の番号
    stream: int
    #: 鳴る素材の範囲（秒） ``source_end`` は含まない
    source_start: Fraction
    source_end: Fraction
    #: タイムラインの範囲（フレーム） ``end`` は含まない
    start: Fraction
    end: Fraction
    #: 素材の 1 秒がタイムラインの何秒に当たるかの逆（中のクリップと置いたシーンの積）
    speed: Fraction
    #: 開いているタイムラインの上のクリップ シーンの中の物は、シーンを置いたクリップ
    clip_id: ClipId
    track_id: TrackId
    #: シーンの中のクリップなら、その ID ``None`` なら ``clip_id`` が音のクリップそのもの
    inner_clip_id: ClipId | None = None

    def frame_at(self, seconds: Fraction, rate: FrameRate) -> Fraction:
        """素材の秒を、この音の並びのタイムラインのフレームへ 範囲の外も同じ割合で延ばす"""
        return self.start + (seconds - self.source_start) / self.speed / rate.frame_duration

    def seconds_at(self, frame: Fraction | int, rate: FrameRate) -> Fraction:
        """タイムラインのフレームを素材の秒へ 範囲の外も同じ割合で延ばす"""
        return self.source_start + (frame - self.start) * rate.frame_duration * self.speed


def heard_pieces(project: Project) -> list[HeardPiece]:
    """書き出しで鳴る素材の音を、タイムラインの順に返す（#420）

    語の境目の検査はタイムラインのコマではなく素材の秒で見るので、シーンの中と速度変更を
    ここで元の時刻へ戻す 鳴るかはミキサと同じ決まり（:func:`sounding`）で選ぶ 別の決まりで
    選ぶと、消音したトラックの切れ目を問題として返す
    """
    pieces = _heard_pieces(project, project.timeline, 0)
    seen: set[tuple[object, ...]] = set()
    unique: list[HeardPiece] = []
    for piece in sorted(pieces, key=lambda p: (p.start, p.end)):
        # 速度が違えば別の音 頭と長さが同じでも、片方を落とすとその語と切れ目が消える
        key = (
            piece.media_id,
            piece.stream,
            piece.source_start,
            piece.source_end,
            piece.start,
            piece.end,
            piece.speed,
        )
        if key not in seen:
            seen.add(key)
            unique.append(piece)
    return unique


def _heard_pieces(project: Project, timeline: Timeline, depth: int) -> Iterator[HeardPiece]:
    rate = project.rate
    for track, clip in sounding(project, timeline):
        if clip.scene_id is not None:
            scene = project.find_scene(clip.scene_id)
            if scene is not None and depth < MAX_SCENE_DEPTH:
                inner = _heard_pieces(project, scene.timeline, depth + 1)
                yield from _placed_pieces(inner, clip, rate, track.id)
            continue
        if clip.media_id is None:
            continue
        media = project.find_media(clip.media_id)
        if media is None:
            continue
        yield HeardPiece(
            media_id=media.id,
            stream=subtitle_voice(project, track, clip, media),
            source_start=clip.source_in,
            source_end=clip.source_out(rate),
            start=Fraction(clip.timeline_start),
            end=Fraction(clip.timeline_end),
            speed=clip.speed,
            clip_id=clip.id,
            track_id=track.id,
        )


def _placed_pieces(
    inner: Iterator[HeardPiece], clip: Clip, rate: FrameRate, track_id: TrackId
) -> Iterator[HeardPiece]:
    """シーンの中の音を、シーンを置いたクリップの窓で切り、外の時刻へ写す

    窓の端で切れた所も素材の秒へ戻す シーンを置いたクリップの端が語の途中に当たる形を
    見逃さないため 写し方は描画と同じ（:func:`_place_scene`）
    """
    scene_start = clip.source_in / rate.frame_duration
    scene_end = scene_start + clip.duration * clip.speed
    for piece in inner:
        low = max(piece.start, scene_start)
        high = min(piece.end, scene_end)
        if low >= high:
            continue
        yield replace(
            piece,
            source_start=piece.seconds_at(low, rate),
            source_end=piece.seconds_at(high, rate),
            start=clip.timeline_start + (low - scene_start) / clip.speed,
            end=clip.timeline_start + (high - scene_start) / clip.speed,
            speed=piece.speed * clip.speed,
            clip_id=clip.id,
            track_id=track_id,
            inner_clip_id=piece.inner_clip_id or piece.clip_id,
        )


def _project_tracks(
    project: Project, timeline: Timeline, depth: int, *, heard: bool = False
) -> Iterator[ProjectedSubtitle]:
    placed = (
        sounding(project, timeline)
        if heard
        else [(track, clip) for track in timeline.tracks for clip in track.clips]
    )
    for track, clip in placed:
        if clip.scene_id is not None:
            # シーンの中の字幕も、置いた場所に出す 見ないと、シーンにまとめた
            # 途端に字幕パネル・焼き込み・字幕ファイルから消える
            # 描画と同じ深さまで見る（レンダラは深さ 7 に置いたシーンの中身も描く）
            if depth >= MAX_SCENE_DEPTH:
                continue
            scene = project.find_scene(clip.scene_id)
            if scene is None:
                continue
            inner = _project_tracks(project, scene.timeline, depth + 1, heard=heard)
            yield from _place_scene(inner, clip, project.rate, track.id)
            continue
        if clip.media_id is None:
            continue
        media = project.find_media(clip.media_id)
        if media is None or not media.transcripts:
            continue
        yield from project_clip(
            clip, media, project.rate, track.id, subtitle_stream(project, track, clip)
        )


def sounding(project: Project, timeline: Timeline) -> list[tuple[Track, Clip]]:
    """``timeline`` の中で、書き出しで音が鳴るクリップ（ミキサと同じ決まり）

    音声トラックと混合トラックは聞こえるトラック（ソロとミュート）、映像トラックは映る
    トラックの置いたシーンだけ どれも無効のクリップは鳴らさない 入れ子のシーンの中も
    同じ決まりで選ぶ（ミキサはシーンの中のタイムラインも同じ道で混ぜる）
    """
    tracks = list(timeline.active_sound_tracks())
    tracks += [t for t in timeline.active_picture_tracks() if t.kind is TrackKind.VIDEO]
    return [
        (track, clip)
        for track in tracks
        for clip in track.clips
        if clip.enabled and project.plays_sound(track, clip)
    ]


def _place_scene(
    inner: Iterator[ProjectedSubtitle], clip: Clip, rate: FrameRate, track_id: TrackId
) -> Iterator[ProjectedSubtitle]:
    """シーンの中の位置を、シーンを置いたクリップの上の位置へ写す

    シーンの中の時刻は ``source_in``（秒）と速度で決まる（描画と同じ決まり）
    クリップの範囲の外に出る分は切り詰める 字幕は置いたクリップの持ち物として
    返す 中のクリップはメインのタイムラインに無いので、画面で選べない
    """
    # 端数を先にフレームへ落とすと、速度で割ったあとに 1 フレームずれる
    # 描画（renderer._draw_scene）は秒のまま足してから 1 回だけフレームへ直している
    scene_start = clip.source_in / rate.frame_duration
    for subtitle in inner:
        start = _scene_to_clip_frame(subtitle.start_frame - scene_start, clip, Rounding.FLOOR)
        end = _scene_to_clip_frame(subtitle.end_frame - scene_start, clip, Rounding.CEIL)
        if end <= 0 or start >= clip.duration:
            continue
        clipped_start = max(0, start)
        clipped_end = min(clip.duration, max(end, clipped_start + 1))
        yield replace(
            subtitle,
            clip_id=clip.id,
            track_id=track_id,
            start_frame=clip.timeline_start + clipped_start,
            end_frame=clip.timeline_start + clipped_end,
            clipped_head=subtitle.clipped_head or start < 0,
            clipped_tail=subtitle.clipped_tail or end > clip.duration,
        )


def _scene_to_clip_frame(scene_frames: Fraction | int, clip: Clip, rounding: Rounding) -> int:
    elapsed = Fraction(scene_frames) / clip.speed
    if rounding is Rounding.CEIL:
        return -((-elapsed.numerator) // elapsed.denominator)
    return elapsed.numerator // elapsed.denominator


def _source_to_clip_frame(
    source_time: Fraction, clip: Clip, rate: FrameRate, rounding: Rounding
) -> int:
    """ソース秒を、クリップ先頭からの相対フレームへ

    速度変更を掛けたクリップでは、ソース時間の進みとタイムライン時間の進みが
    ``clip.speed`` 倍だけ違う
    """
    elapsed = (source_time - clip.source_in) / clip.speed
    return seconds_to_frame(elapsed, rate, rounding)
