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
    TranscriptSegment,
    heard_stream,
)
from sashimono.core.timebase import FrameRate, Rounding, seconds_to_frame

__all__ = [
    "ProjectedSubtitle",
    "project_clip",
    "project_timeline",
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
    """
    projected = list(_project_tracks(project, project.timeline, 0))
    projected.sort(key=lambda p: (p.start_frame, p.end_frame))
    yield from projected


#: シーンの入れ子をたどる深さの上限 描画と音（MAX_SCENE_DEPTH）に揃える
MAX_SCENE_DEPTH = 8


def _project_tracks(
    project: Project, timeline: Timeline, depth: int
) -> Iterator[ProjectedSubtitle]:
    for track in timeline.tracks:
        for clip in track.clips:
            if clip.scene_id is not None:
                # シーンの中の字幕も、置いた場所に出す 見ないと、シーンにまとめた
                # 途端に字幕パネル・焼き込み・字幕ファイルから消える
                # 描画と同じ深さまで見る（レンダラは深さ 7 に置いたシーンの中身も描く）
                if depth >= MAX_SCENE_DEPTH:
                    continue
                scene = project.find_scene(clip.scene_id)
                if scene is None:
                    continue
                inner = _project_tracks(project, scene.timeline, depth + 1)
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
