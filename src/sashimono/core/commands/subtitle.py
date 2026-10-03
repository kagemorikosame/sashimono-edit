"""字幕の編集コマンドと、タイムラインへの焼き込み

字幕は素材（:class:`~sashimono.core.model.MediaItem`）に属するので、ここの操作は
どれもタイムラインを触らない 1 か所直せば、その素材を使っているすべての箇所に
同時に反映される 素材を 3 回置いていても、直すのは 1 回で済む

例外は :func:`burn_subtitles` で、これだけはタイムラインへテキストを並べる
書き出し先が字幕に対応していない場合や、装飾を凝りたい場合の逃げ道
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from fractions import Fraction

from sashimono.core.commands.base import Command
from sashimono.core.commands.edit import AddClip, AddTrack
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.commands.layers import new_layer, places_mixed
from sashimono.core.model import (
    Clip,
    GeneratedSource,
    MediaId,
    Project,
    SegmentId,
    SubtitleOrigin,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    new_clip_id,
    new_effect_id,
)
from sashimono.core.projection import ProjectedSubtitle, project_timeline

__all__ = [
    "AddSegment",
    "MergeWithNext",
    "RemoveSegment",
    "RetimeSegment",
    "SetSegmentText",
    "SplitSegment",
    "Voice",
    "burn_defaults",
    "burn_subtitles",
    "subtitle_voices",
    "voice_label",
]

#: 焼き込むテキストオブジェクトで、本文を入れるパラメータ名
TEXT_PARAM = "text"

#: 焼き込むテキストの既定の見た目（1080p のとき） 大きさ・画面の中央からの縦位置
#: （上が正）・縁取りの太さ 画面の下から 160px に、縁取りを付けて読めるように置く
_BURN_AT_1080 = {"size": 48.0, "pos_y": -380.0, "border_width": 4.0}


def burn_defaults(height: int) -> dict[str, float]:
    """焼き込むテキストの既定の見た目 画面の高さ ``height`` に合わせて縮める

    画素の値のまま使うと、720p の作品では縦位置 -380 が画面の下端（-360）より下になり、
    焼き込んだ字幕がプレビューにも書き出しにも出ない 1080p の値を高さの比で縮めて、
    どの解像度でも画面の同じ所に同じ大きさで出す
    """
    scale = max(1, height) / 1080
    return {name: value * scale for name, value in _BURN_AT_1080.items()}


@dataclass(frozen=True, slots=True)
class SetSegmentText(Command):
    """字幕 1 枚の本文を書き換える"""

    media_id: MediaId
    segment_id: SegmentId
    text: str
    #: 音声ストリームの番号 ``None`` なら 1 本目（下のコマンドもすべて同じ）
    stream: int | None = None

    @property
    def label(self) -> str:
        return "字幕を編集"

    def apply(self, project: Project) -> Project:
        transcript, segment = _locate(project, self.media_id, self.segment_id, self.stream)
        updated = transcript.replace_segment(segment.with_text(self.text))
        return _store(project, self.media_id, updated, self.stream)


@dataclass(frozen=True, slots=True)
class RetimeSegment(Command):
    """字幕 1 枚の時刻を動かす 値はソース秒"""

    media_id: MediaId
    segment_id: SegmentId
    start: Fraction
    end: Fraction
    stream: int | None = None

    @property
    def label(self) -> str:
        return "字幕の時刻を変更"

    def apply(self, project: Project) -> Project:
        transcript, segment = _locate(project, self.media_id, self.segment_id, self.stream)
        if self.end <= self.start:
            raise ValueError(f"終了が開始以前: {self.start} .. {self.end}")
        moved = replace(segment, start=self.start, end=self.end, edited=True)
        return _store(
            project, self.media_id, _rebuilt(transcript, _swap(transcript, moved)), self.stream
        )


@dataclass(frozen=True, slots=True)
class RemoveSegment(Command):
    """字幕 1 枚を消す"""

    media_id: MediaId
    segment_id: SegmentId
    stream: int | None = None

    @property
    def label(self) -> str:
        return "字幕を削除"

    def apply(self, project: Project) -> Project:
        transcript, segment = _locate(project, self.media_id, self.segment_id, self.stream)
        remaining = [s for s in transcript.segments if s.id != segment.id]
        return _store(project, self.media_id, _rebuilt(transcript, remaining), self.stream)


@dataclass(frozen=True, slots=True)
class AddSegment(Command):
    """字幕を 1 枚足す 起こしを使わずに手で入れる場合に使う"""

    media_id: MediaId
    start: Fraction
    end: Fraction
    text: str = ""
    stream: int | None = None

    @property
    def label(self) -> str:
        return "字幕を追加"

    def apply(self, project: Project) -> Project:
        item = project.require_media(self.media_id)
        found = item.transcript_for(self.stream)
        transcript = found if found is not None else Transcript()
        added = TranscriptSegment(start=self.start, end=self.end, text=self.text, edited=True)
        return _store(
            project,
            self.media_id,
            _rebuilt(transcript, [*transcript.segments, added]),
            self.stream,
        )


@dataclass(frozen=True, slots=True)
class SplitSegment(Command):
    """字幕 1 枚を、ソース時刻 ``at`` で 2 枚に割る

    単語タイムスタンプがあればその境界で本文を分ける 無ければ時間の比で分ける
    どちらにしても文の途中で切れることはあるので、割ったあとに手で直せるよう、
    分けた両方に編集済みの印は付けない
    """

    media_id: MediaId
    segment_id: SegmentId
    at: Fraction
    stream: int | None = None

    @property
    def label(self) -> str:
        return "字幕を分割"

    def apply(self, project: Project) -> Project:
        transcript, segment = _locate(project, self.media_id, self.segment_id, self.stream)
        if not (segment.start < self.at < segment.end):
            raise ValueError(f"分割位置が字幕の内側にない: {self.at}")

        head_text, tail_text = _split_text(segment, self.at)
        head = TranscriptSegment(
            start=segment.start,
            end=self.at,
            text=head_text,
            words=tuple(w for w in segment.words if w.start < self.at),
            speaker=segment.speaker,
            id=segment.id,
        )
        tail = TranscriptSegment(
            start=self.at,
            end=segment.end,
            text=tail_text,
            words=tuple(w for w in segment.words if w.start >= self.at),
            speaker=segment.speaker,
        )
        others = [s for s in transcript.segments if s.id != segment.id]
        return _store(
            project, self.media_id, _rebuilt(transcript, [*others, head, tail]), self.stream
        )


@dataclass(frozen=True, slots=True)
class MergeWithNext(Command):
    """字幕を次の 1 枚と繋げる 認識が細かく割れすぎたときに使う"""

    media_id: MediaId
    segment_id: SegmentId
    stream: int | None = None

    @property
    def label(self) -> str:
        return "字幕を結合"

    def apply(self, project: Project) -> Project:
        transcript, segment = _locate(project, self.media_id, self.segment_id, self.stream)
        index = transcript.segments.index(segment)
        if index + 1 >= len(transcript.segments):
            raise ValueError("次の字幕が無い")
        following = transcript.segments[index + 1]

        parts = [part for part in (segment.text.strip(), following.text.strip()) if part]
        merged = TranscriptSegment(
            start=segment.start,
            end=following.end,
            # 改行ではなく空白で繋ぐ 折り返しは整形が決めるもので、ここで行を
            # 確定させると、整形を掛け直したときに二重に折り返される
            text=" ".join(parts),
            words=segment.words + following.words,
            speaker=segment.speaker or following.speaker,
            edited=True,
            id=segment.id,
        )
        remaining = [s for s in transcript.segments if s.id not in (segment.id, following.id)]
        return _store(
            project, self.media_id, _rebuilt(transcript, [*remaining, merged]), self.stream
        )


#: 焼き込む字幕の話し手（素材と音声ストリームの番号） 素材を持たない（シーンの中の）字幕は
#: ``(None, 0)`` にまとめる
Voice = tuple[MediaId | None, int]


def subtitle_voices(project: Project) -> list[Voice]:
    """タイムラインに字幕が出ている話し手 素材の並び・音声の番号の順"""
    found = {voice for voice, _ in _voiced(project)}
    order = {item.id: index for index, item in enumerate(project.media)}
    return sorted(found, key=lambda v: (order.get(v[0], -1) if v[0] else -1, v[1]))


def voice_label(project: Project, voice: Voice, *, with_media: bool = True) -> str:
    """話し手の名前 音声が何本もある素材は「音声 N」を添える"""
    media_id, stream = voice
    media = project.find_media(media_id) if media_id is not None else None
    if media is None:
        return "シーン"
    known = [s.index for s in media.audio_streams]
    number = known.index(stream) + 1 if stream in known else 1
    voice_part = f"音声 {number}" if len(known) > 1 else ""
    if not with_media:
        return voice_part or media.name
    return f"{media.name} {voice_part}".strip()


def _voiced(project: Project) -> list[tuple[Voice, ProjectedSubtitle]]:
    """タイムラインに出る字幕と、その話し手

    話し手は投影した字幕の出どころ（素材と音声）から取る 置いたクリップから素材を
    たどると、置いたシーンの中の字幕はシーンのクリップ（素材を持たない）に当たり、
    シーンの中の別々の話し手が (None, 0) の 1 本にまとまって欠けた（PR #231 の指摘）
    """
    result: list[tuple[Voice, ProjectedSubtitle]] = []
    for subtitle in project_timeline(project):
        voice: Voice = (None, 0)
        if subtitle.media_id is not None and project.find_media(subtitle.media_id) is not None:
            voice = (subtitle.media_id, subtitle.stream)
        result.append((voice, subtitle))
    return result


def burn_subtitles(
    project: Project,
    template: GeneratedSource | Clip,
    *,
    track_name: str = "字幕",
    text_param: str = TEXT_PARAM,
    voices: Collection[Voice] | None = None,
    segments: Collection[SegmentId] | None = None,
) -> list[Command]:
    """いま画面に出る字幕を、テキストオブジェクトとしてタイムラインへ並べる

    投影した結果をそのまま置くので、この時点のカット状態が固定される あとから
    素材を切っても焼き込んだテキストは動かない だから仕上げの最後に使う

    話し手（素材と音声 :data:`Voice`）ごとに別のレイヤー（分ける方式では映像トラック）へ
    入れる 1 本にまとめていたときは、音声 1 と 2 が同時に話している所で前の字幕が
    切り詰められて欠けた 同じ話し手の中で重なる字幕は、前の方を切り詰める
    （1 本のトラックにクリップを重ねられないため）

    ``template`` はテキストの生成物か、ひな形にするテキストのクリップ クリップなら
    フォント・色・位置・縁取り・エフェクトまでそのまま写し、本文だけを差し替える
    ``voices`` は焼き込む話し手（``None`` なら全部）、``segments`` は焼き込む字幕の ID
    （``None`` なら全部 字幕パネルで選んだ行だけを置くときに渡す）
    コマンドは 1 回の取り消しで全部戻る並び（呼ぶ側がまとめて出す）
    """
    grouped: dict[Voice, list[ProjectedSubtitle]] = {}
    for voice, subtitle in _voiced(project):
        if voices is not None and voice not in voices:
            continue
        if segments is not None and subtitle.segment.id not in segments:
            continue
        if not subtitle.segment.text.strip():
            continue
        grouped.setdefault(voice, []).append(subtitle)
    if not grouped:
        return []

    order = subtitle_voices(project)
    keys = sorted(grouped, key=lambda v: order.index(v) if v in order else len(order))
    same_media = len({media for media, _ in keys}) == 1
    commands: list[Command] = []
    for voice in keys:
        if len(keys) == 1:
            name = track_name
        else:
            name = f"{track_name} {voice_label(project, voice, with_media=not same_media)}"
        if places_mixed(project):
            # 混合の方式ではレイヤーにする 並びの末尾（一番手前）に入るので、動画の上に出る
            # 映像トラックにすると、方式を混合にしたのに字幕だけ別の種類のトラックへ入る
            track = new_layer(project, commands, name=name)
        else:
            track = Track(kind=TrackKind.VIDEO, name=name)
            commands.append(AddTrack(track))
        for start, end, text, segment_id in _laid_out(grouped[voice]):
            clip = _text_clip(template, text_param, text, start, end)
            # 出どころの印を付ける 字幕の誤植を直すとき、印のあるクリップだけを一緒に直す
            # （本文の一致で探すと、手で書いた同じ本文のタイトルまで書き換わる）
            origin = SubtitleOrigin(media_id=voice[0], stream=voice[1], segment_id=segment_id)
            commands.append(AddClip(track.id, replace(clip, subtitle_origin=origin)))
    return commands


def _laid_out(subtitles: list[ProjectedSubtitle]) -> list[tuple[int, int, str, SegmentId]]:
    """1 本のトラックへ並べる 重なる字幕は前の方を切り詰める 字幕の行の ID を添える"""
    placed: list[tuple[int, int, str, SegmentId]] = []
    for subtitle in sorted(subtitles, key=lambda s: (s.start_frame, s.end_frame)):
        text = subtitle.segment.text.strip()
        start, end = subtitle.start_frame, subtitle.end_frame
        if placed and start < placed[-1][1]:
            previous_start, _, previous_text, previous_id = placed[-1]
            if start <= previous_start:
                continue
            placed[-1] = (previous_start, start, previous_text, previous_id)
        placed.append((start, end, text, subtitle.segment.id))
    return placed


def _text_clip(
    template: GeneratedSource | Clip, text_param: str, text: str, start: int, end: int
) -> Clip:
    """字幕 1 枚のテキストのクリップ"""
    if isinstance(template, Clip) and template.source is not None:
        # ひな形のクリップを写す ID は振り直す（同じ ID が 2 本あるとプロジェクトの検査に
        # 断られ、片方を直したつもりで両方を探し当てる） リンクとグループは外す
        return replace(
            template,
            id=new_clip_id(),
            timeline_start=start,
            duration=end - start,
            source=template.source.with_param(text_param, text),
            effects=tuple(replace(e, id=new_effect_id()) for e in template.effects),
            after_effects=tuple(replace(e, id=new_effect_id()) for e in template.after_effects),
            link_group=None,
            group_id=None,
        )
    source = template.source if isinstance(template, Clip) else template
    assert source is not None
    # 置いたテキストと同じく描画の欄を持たせる 焼き込んだ字幕だけ欄が無いと、
    # 位置を直すのに変形をエフェクトの一覧から探して足すことになる
    return with_fixed_items(
        Clip(
            timeline_start=start,
            duration=end - start,
            source=source.with_param(text_param, text),
        ),
        picture=True,
    )


def _locate(
    project: Project, media_id: MediaId, segment_id: SegmentId, stream: int | None = None
) -> tuple[Transcript, TranscriptSegment]:
    item = project.require_media(media_id)
    transcript = item.transcript_for(stream)
    if transcript is None:
        raise KeyError(f"素材に字幕が無い: {item.name}")
    for segment in transcript.segments:
        if segment.id == segment_id:
            return transcript, segment
    raise KeyError(f"字幕が見つからない: {segment_id}")


def _swap(transcript: Transcript, segment: TranscriptSegment) -> list[TranscriptSegment]:
    return [segment if s.id == segment.id else s for s in transcript.segments]


def _rebuilt(transcript: Transcript, segments: list[TranscriptSegment]) -> Transcript:
    """順序を整えて作り直す

    :class:`Transcript` は開始時刻の昇順を不変条件にしている 時刻を動かす操作で
    並びが崩れるので、呼び出し側が気にせずに済むようここで整える
    """
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    return Transcript(tuple(ordered), language=transcript.language, model=transcript.model)


def _store(
    project: Project, media_id: MediaId, transcript: Transcript, stream: int | None = None
) -> Project:
    item = project.require_media(media_id)
    return project.replace_media(item.with_transcript(transcript, stream))


def _split_text(segment: TranscriptSegment, at: Fraction) -> tuple[str, str]:
    """本文を分割位置で分ける"""
    text = segment.text.strip()
    if segment.words:
        head = "".join(w.text for w in segment.words if w.start < at).strip()
        tail = "".join(w.text for w in segment.words if w.start >= at).strip()
        if head or tail:
            return head, tail

    if segment.duration <= 0:
        return text, ""
    ratio = (at - segment.start) / segment.duration
    cut = max(0, min(len(text), round(len(text) * float(ratio))))
    return text[:cut].strip(), text[cut:].strip()
