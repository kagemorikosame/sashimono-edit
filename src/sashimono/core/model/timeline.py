"""タイムライン、トラック、クリップ

クリップの位置と長さは**フレーム単位の整数**で持つ ここを秒にすると、隣接クリップの
境界で丸め方向が食い違って 1 フレームの隙間や重なりが生まれる 素材のどこを使うかを
示す ``source_in`` だけは秒（:class:`~fractions.Fraction`）で持つ 素材のフレームレートが
プロジェクトと異なることがあるため
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction

from sashimono.core.model.effect import AnimatedValue, Effect, ParamValue
from sashimono.core.model.ids import (
    ClipId,
    GroupId,
    MediaId,
    SceneId,
    SegmentId,
    StrokeId,
    TrackId,
    new_clip_id,
    new_track_id,
)
from sashimono.core.model.media import MediaItem
from sashimono.core.model.stroke import Stroke
from sashimono.core.timebase import FrameRate

__all__ = [
    "FILTER_KIND",
    "GROUP_AS_ONE",
    "GROUP_KIND",
    "GROUP_LAYERS",
    "Clip",
    "GeneratedSource",
    "Marker",
    "Timeline",
    "Track",
    "TrackKind",
    "controlling_groups",
    "default_track_name",
    "draws_picture",
    "group_as_one",
    "group_layers",
    "group_reaches",
    "plays_sound",
]

#: 下のトラックを重ね終えた絵にエフェクトを掛ける生成オブジェクトの種類（AviUtl の
#: フィルタオブジェクト Issue #27） 掛けるエフェクトはクリップの :attr:`Clip.effects`
#:
#: トラックの種類（``TrackKind.EFFECT``）ではなく、映像トラックに置く生成オブジェクトに
#: した 理由は 2 つ
#:
#: - 効くのは「重ね順でそれより下にある絵」 映像トラックと同じ並びの中に居ないと
#:   上下が決まらない 種類を分けると、映像を重ねる所（描画・先読みの捨てる範囲・
#:   音のシーン・タイムラインの並び・互換の読み込みなど 17 ファイル 36 か所）が
#:   どれも「映像か効果か」を見分け直すことになり、1 か所でも忘れるとフィルタの
#:   トラックだけ重ね順から抜ける
#: - AviUtl も YMM4 も、フィルタ（エフェクトアイテム）は他のオブジェクトと同じ
#:   レイヤーに並ぶ トラックの種類で分けると、同じレイヤーにテキストとフィルタが
#:   交互に並ぶ作品を 1 本のトラックへ写せない
#:
#: 「エフェクトのトラック」が欲しいときは、映像トラックにフィルタのクリップだけを
#: 並べればよい 音声トラックに置いて下の音全体に掛ける使い方も、同じ種類のまま
#: ミキサが読めば足せる（いまは映像だけ）
FILTER_KIND = "filter"

#: 下のレイヤーのオブジェクトをまとめて動かす生成オブジェクトの種類（AviUtl の拡張編集の
#: グループ制御） 自分では何も描かず、置いたトラックより手前（並びの後ろ）の
#: :data:`GROUP_LAYERS` 本のトラックの、同じ時刻に描くクリップ 1 本ずつへ、自分の配置
#: （位置・拡大・回転）と不透明度とエフェクトを掛ける 掛け方は :func:`controlling_groups`
#: と :mod:`sashimono.engine.render.groups`
#:
#: 混合の方式ではレイヤーの番号が大きい側（画面では下に並ぶ 描くと手前）を受け持つ
#: AviUtl と同じ向き 分ける方式の映像トラックでも同じく並びの後ろ（画面では上の V）を持つ
#: どちらも「自分より手前に描くトラック」で、描く順の決まりが 1 つで済む
GROUP_KIND = "group_control"

#: グループ制御の「対象レイヤー数」の項目の名前 0 なら手前のすべて
GROUP_LAYERS = "layers"

#: グループ制御の「1 枚の絵として扱う」の項目の名前 真なら受け持つトラックのその時刻の絵を
#: 1 枚に重ねてから、グループの配置・不透明度・エフェクトを掛ける（重なった半透明が透けない）
#: 偽（既定）なら 1 本ずつに掛ける
GROUP_AS_ONE = "as_one"


@dataclass(frozen=True, slots=True)
class GeneratedSource:
    """素材を持たないクリップの中身 テキストや図形

    :class:`~sashimono.core.model.Effect` と同じく ``kind`` と ``params`` だけを持つ
    フィルタと生成物は役割が違うので型は分けるが、パラメータの仕組みは共有する
    設定 UI もプリセットも 1 つの実装で済ませるため
    """

    kind: str
    params: dict[str, ParamValue] = field(default_factory=dict)
    #: テキストの縁取りの層（#272） 並びの頭が一番上 空なら ``params`` の縁取りの項目で描く
    #: （:mod:`sashimono.core.model.stroke`） テキスト以外の種類は持たない
    strokes: tuple[Stroke, ...] = ()

    def with_param(self, name: str, value: ParamValue) -> GeneratedSource:
        # ``replace`` で作る 欄を並べて作り直すと、縁取りの層のような後で足した欄が
        # 値を 1 つ触っただけで落ちる
        return replace(self, params={**self.params, name: value})

    def with_strokes(self, strokes: Sequence[Stroke]) -> GeneratedSource:
        return replace(self, strokes=tuple(strokes))

    def find_stroke(self, stroke_id: StrokeId) -> Stroke | None:
        return next((stroke for stroke in self.strokes if stroke.id == stroke_id), None)


class TrackKind(Enum):
    """トラックの種類

    ``MIXED`` は映像・音声・テキストを何でも置ける 1 本のレイヤー（YMM4・AviUtl の
    レイヤー Issue #27） 音付きの動画は絵と音を 1 本のクリップで持ち、どちらを
    出すかはクリップの :attr:`Clip.show_picture` と :attr:`Clip.audio_stream` が決める
    映像と音声を分ける今の方式（``VIDEO`` と ``AUDIO``）はそのまま残る

    トラックが絵を描くか・音を鳴らすかは、種類を直に見て決めない
    :meth:`Timeline.picture_tracks` などと :func:`draws_picture` :func:`plays_sound` を通す
    種類の分岐を描画・音の合成・書き出し・先読みの捨て方へ散らすと、どれか 1 か所で
    混合を忘れたときに「プレビューには出るのに書き出すと消える」が起きる
    """

    VIDEO = "video"
    AUDIO = "audio"
    MIXED = "mixed"


#: 種類ごとのトラック名の頭 混合は利用者の決定で「レイヤー 1」（番号の前に空白）
_NAME_PREFIX = {TrackKind.VIDEO: "V", TrackKind.AUDIO: "A", TrackKind.MIXED: "レイヤー "}


def default_track_name(kind: TrackKind, number: int, taken: Collection[str] = ()) -> str:
    """``number`` 本目の ``kind`` のトラックに付ける名前（``V1`` ``A1`` ``レイヤー 1``）

    トラックを作る所が同じ名前を付けるため 映像か音声かの 2 択で頭の文字を
    書き分けると、混合トラックだけ「A3」のような名前が付く

    ``taken`` にある名前は飛ばして番号を進める 消したトラックの名前が残っていると、
    本数を数えただけでは同じ名前が 2 本並ぶ（レイヤー 1 を消してレイヤー 2 が残ると、
    もう 1 本のレイヤー 2 ができる）
    """
    while f"{_NAME_PREFIX[kind]}{number}" in taken:
        number += 1
    return f"{_NAME_PREFIX[kind]}{number}"


@dataclass(frozen=True, slots=True)
class SubtitleOrigin:
    """字幕から置いた（焼き込んだ）テキストのクリップが、どの字幕の行から来たか

    字幕の誤植をまとめて直すとき、焼き込んだ文字も一緒に直すのに使う 本文の一致だけで
    探すと、手で書いたタイトルがたまたま直す前の字幕と同じ本文だと書き換えてしまう
    """

    #: 字幕を持つ素材 シーンの中の字幕（素材を持たない）は ``None``
    media_id: MediaId | None
    #: 字幕の音声ストリームの番号（:meth:`MediaItem.transcript_stream` でそろえた物）
    stream: int
    segment_id: SegmentId


@dataclass(frozen=True, slots=True)
class Clip:
    """タイムライン上に置かれた 1 つのクリップ

    ``media_id`` が ``None`` のクリップは、素材を持たない生成オブジェクト
    （テキスト、図形など） その場合の見た目は :attr:`effects` が決める
    ``scene_id`` を持つクリップは、別のシーン（タイムライン）を 1 本の絵と音として
    入れ子に置いたもの（AviUtl のシーンオブジェクト）
    """

    #: タイムライン上の開始位置（フレーム）
    timeline_start: int
    #: タイムライン上の長さ（フレーム） 1 以上
    duration: int
    media_id: MediaId | None = None
    #: 素材を持たないクリップの中身（テキスト・図形） ``media_id`` が
    #: ``None`` のときだけ意味を持つ
    source: GeneratedSource | None = None
    #: 素材内の開始位置（秒）
    source_in: Fraction = Fraction(0)
    #: 使用する素材内のストリーム番号 多言語音声などで意味を持つ
    stream_index: int = 0
    #: 再生速度 2 なら 2 倍速で、同じ長さに 2 倍のソース範囲が入る
    speed: Fraction = Fraction(1)
    #: 絵を止める素材の時刻（秒） 読む時刻がここを越えたら、ここの絵を出し続ける
    #: ``None`` なら止めない（素材の終わりを越えた所は何も映らない）
    #:
    #: 素材の長さを越えた所で最後の絵を出し続ける（YMM4 の素材より長い動画アイテム）なら
    #: 最後のフレームの時刻、頭の絵で止める（YMM4 の再生速度 0）なら ``source_in`` を持つ
    #: 「止める時刻」ではなく**素材の中の時刻の上限**で持つのは、分割・トリム・速さの
    #: 変更で ``source_in`` と ``speed`` が変わっても、そのまま写すだけで同じ絵が出るため
    #: 分割した後半が上限より後から始まれば、後半はずっと止まった絵になる
    #: クリップの中の経過で持つと、編集の命令がそれぞれ計算し直す必要があり、1 つでも
    #: 忘れると止まる位置がずれる
    #:
    #: **絵だけに効く** 音は止めない（止めた絵の間も素材は進む） YMM4 でも、素材の
    #: 終わりを越えた所は無音で、再生速度 0 の音は鳴らない（音量 0 で写している）
    hold_at: Fraction | None = None
    effects: tuple[Effect, ...] = ()
    #: 混合トラックで鳴らす音声ストリームの番号（:attr:`AudioStreamInfo.index`）
    #: ``None`` なら鳴らさない 映像・音声のトラックでは読まない（音声トラックは
    #: :attr:`stream_index` を鳴らす）
    #:
    #: 絵の :attr:`stream_index` とは別に持つ 混合トラックのクリップは 1 本で絵と音の
    #: 両方を出すので、1 つの番号では映像と音声のストリームを同時に指せない
    #: 真偽ではなく番号にしたのは、多言語音声の素材でどの音を鳴らすかを選ぶため
    audio_stream: int | None = None
    #: 混合トラックで絵を描くか 偽なら重ねから外す（:attr:`clip_to_below` の相手にもならない）
    #: 映像・音声のトラックでは読まない 音だけ使いたい動画を、クリップを分けずに置くため
    show_picture: bool = True
    #: 場面切り替え（生成オブジェクト ``transition``）で、後の場面に掛けるエフェクト
    #: 前の場面には :attr:`effects` が掛かる ほかのクリップでは使わない
    after_effects: tuple[Effect, ...] = ()
    opacity: AnimatedValue = field(default_factory=lambda: AnimatedValue(1.0))
    #: 下のトラックとの重ね方 値は :class:`~sashimono.engine.gpu.BlendMode` の定数
    #: 文字列で持つのは、プロジェクトファイルに出るものを列挙型に縛らないため
    blend_mode: str = "normal"
    #: すぐ下に重なっているクリップの形（不透明度）で切り抜く YMM4 の「上のオブジェクトで
    #: クリッピング」 背景の模様を吹き出しの形だけに見せる、といった使い方をする
    clip_to_below: bool = False
    #: 映像と音声を連動させるためのグループ 同じ値を持つクリップは一緒に動く
    link_group: GroupId | None = None
    #: 入れ子にしたシーン 素材（``media_id``）とは同時に持てない
    scene_id: SceneId | None = None
    #: 束ねたグループ 同じ値を持つクリップは、クリック 1 回でまとめて選ばれる
    #: ``link_group`` とは別物 リンクは映像と音声の同期、グループは編集の手間を
    #: 省くための束ねで、解除しても同期は崩れない
    group_id: GroupId | None = None
    enabled: bool = True
    id: ClipId = field(default_factory=new_clip_id)
    #: 素材の絵を素材の画素の大きさで置くか 偽なら縦横比を保って画面に収める
    #:
    #: YMM4 と AviUtl は拡大率 100% を素材の画素で置く 素材を置くときと互換の読み込みは
    #: 真にする 既定を偽にしてあるのは、項目を持たない前の版のファイルを、そのころの
    #: 描き方（画面に収める）のまま開くため 既定で真にすると、開いただけで小さい素材が
    #: 縮み、大きい素材が画面からはみ出す 素材を持たないクリップ（テキスト・図形）は
    #: 画面の大きさで作るので、どちらでも同じ
    native_size: bool = False
    #: 字幕から置いたテキストなら、その字幕の行（:class:`SubtitleOrigin`） 手で置いた物と
    #: 前の版で焼き込んだ物は ``None``
    subtitle_origin: SubtitleOrigin | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"クリップの長さは 1 フレーム以上必要: {self.duration}")
        if self.source_in < 0:
            raise ValueError(f"素材内の開始位置が負: {self.source_in}")
        if self.speed <= 0:
            raise ValueError(f"再生速度は正でなければならない: {self.speed}")
        if self.hold_at is not None and self.hold_at < 0:
            raise ValueError(f"絵を止める時刻が負: {self.hold_at}")
        if self.audio_stream is not None and self.audio_stream < 0:
            raise ValueError(f"音声ストリームの番号が負: {self.audio_stream}")
        if self.scene_id is not None and (self.media_id is not None or self.source is not None):
            # 両方を持つと、どちらを描くのかが決まらない
            raise ValueError("シーンを置いたクリップは素材や生成オブジェクトを持てない")

    @property
    def timeline_end(self) -> int:
        """タイムライン上の終了位置（フレーム、この位置は含まない）"""
        return self.timeline_start + self.duration

    def source_duration(self, rate: FrameRate) -> Fraction:
        """このクリップが素材から消費するソース時間の長さ（秒）"""
        return self.duration * rate.frame_duration * self.speed

    def source_out(self, rate: FrameRate) -> Fraction:
        """素材内の終了位置（秒、この位置は含まない）"""
        return self.source_in + self.source_duration(rate)

    def picture_time(self, local_frame: int, rate: FrameRate) -> Fraction:
        """クリップの頭から ``local_frame`` 進んだ所で、絵を素材のどの時刻から取るか

        絵を止めていれば :attr:`hold_at` を越えない 音はこれを使わない（止めない）
        描画・先読み・タイムラインの絵の並びが同じ式を使う ずれると先読みが当たらず、
        タイムラインに並ぶ絵とプレビューが食い違う
        """
        seconds = self.source_in + local_frame * rate.frame_duration * self.speed
        if self.hold_at is not None and seconds > self.hold_at:
            return self.hold_at
        return seconds

    @property
    def is_filter(self) -> bool:
        """下のトラックの絵へエフェクトを掛けるクリップか（:data:`FILTER_KIND`）"""
        return self.source is not None and self.source.kind == FILTER_KIND

    @property
    def is_group(self) -> bool:
        """下のレイヤーのオブジェクトを動かすグループ制御か（:data:`GROUP_KIND`）"""
        return self.source is not None and self.source.kind == GROUP_KIND

    def contains(self, frame: int) -> bool:
        return self.timeline_start <= frame < self.timeline_end

    def overlaps(self, start: int, end: int) -> bool:
        """タイムライン範囲 ``[start, end)`` と重なるか"""
        return self.timeline_start < end and start < self.timeline_end

    def moved_to(self, timeline_start: int) -> Clip:
        """ソース範囲を保ったまま、タイムライン上の位置だけ変えた複製を返す"""
        return replace(self, timeline_start=timeline_start)


@dataclass(frozen=True, slots=True)
class Marker:
    """タイムライン上の目印"""

    frame: int
    label: str = ""
    color: str = "#ffcc00"


@dataclass(frozen=True, slots=True)
class Track:
    """クリップを並べる 1 本のトラック

    同一トラック内でクリップが重なることは許さない 重なりを許すと「どちらが上か」の
    規則が必要になり、リップル編集の意味も定義できなくなる 重ねたい場合は
    トラックを分ける
    """

    kind: TrackKind
    name: str = ""
    clips: tuple[Clip, ...] = ()
    #: トラック全体に掛かるフィルタ いまのレンダラは読まない AviUtl のフィルタオブジェクトは
    #: ここではなく、映像トラックに置くフィルタのクリップ（:data:`FILTER_KIND`）で表す
    #: 効く時間をクリップの長さで決められ、同じトラックに他のクリップとも並べられるため
    effects: tuple[Effect, ...] = ()
    locked: bool = False
    muted: bool = False
    solo: bool = False
    #: UI 上の表示高さ（ピクセル）
    height: int = 60
    #: 音声トラックと混合トラックの音量（dB） 映像トラックでは無視される
    volume_db: float = 0.0
    #: 音声トラックと混合トラックの定位 -1 が左、+1 が右
    pan: float = 0.0
    id: TrackId = field(default_factory=new_track_id)

    def __post_init__(self) -> None:
        starts = [c.timeline_start for c in self.clips]
        if starts != sorted(starts):
            raise ValueError(f"トラック {self.name!r} のクリップが開始位置順に並んでいない")
        for left, right in zip(self.clips, self.clips[1:], strict=False):
            if left.timeline_end > right.timeline_start:
                raise ValueError(
                    f"トラック {self.name!r} でクリップが重なっている: "
                    f"{left.id} [{left.timeline_start}, {left.timeline_end}) と "
                    f"{right.id} [{right.timeline_start}, {right.timeline_end})"
                )

    @property
    def end_frame(self) -> int:
        """最後のクリップの終端 空トラックなら 0"""
        return self.clips[-1].timeline_end if self.clips else 0

    def clip_at(self, frame: int) -> Clip | None:
        """``frame`` にあるクリップ 無ければ ``None``"""
        for clip in self.clips:
            if clip.contains(frame):
                return clip
        return None

    def find(self, clip_id: ClipId) -> Clip | None:
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        return None

    def with_clips(self, clips: tuple[Clip, ...]) -> Track:
        """クリップ列を差し替えた複製を返す 開始位置順に整列してから渡す"""
        return replace(self, clips=tuple(sorted(clips, key=lambda c: c.timeline_start)))


@dataclass(frozen=True, slots=True)
class Timeline:
    """トラックの集合

    **重なり順** :attr:`tracks` の並びがそのまま描く順で、先頭が一番奥 後ろのトラックほど
    手前に重なる 映像トラックと混合トラックは同じ並びの中で重なる（種類ごとに別の
    重なりを持たない） 混ざっていても、並びで前にある方が奥

    混合トラックの「レイヤー n」は、混合トラックの中で並びの n 番目 YMM4・AviUtl と
    同じく、レイヤー 1 が一番奥で、番号が大きいレイヤーほど手前に描く
    モデルの並びの意味は分ける方式と同じなので、描く側は種類を問わず並びの順に重ねる
    違うのは画面の並べ方だけで、映像トラックは V1 を一番下に置き番号が大きいほど上へ
    （手前が上）、混合トラックはレイヤー 1 を一番上に置き番号が大きいほど下へ（手前が下）
    並べる（:meth:`~sashimono.ui.timeline.layout.TimelineLayout.bands`）
    並びを種類ごとに逆さに持つと、変換（映像トラック ⇔ 混合トラック）のたびに重なりが
    裏返り、描く側も種類で順を分けることになる

    音声トラックは並びの中のどこにあっても重なりに加わらない（絵を描かない）
    """

    rate: FrameRate
    tracks: tuple[Track, ...] = ()
    markers: tuple[Marker, ...] = ()
    #: 書き出し範囲 ``None`` なら全体
    work_area: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        ids = [t.id for t in self.tracks]
        if len(set(ids)) != len(ids):
            raise ValueError("トラック ID が重複している")
        if self.work_area is not None:
            start, end = self.work_area
            if end <= start:
                raise ValueError(f"書き出し範囲が不正: {self.work_area}")

    @property
    def duration(self) -> int:
        """全トラックを通した長さ（フレーム）"""
        return max((t.end_frame for t in self.tracks), default=0)

    def video_tracks(self) -> Iterator[Track]:
        return (t for t in self.tracks if t.kind is TrackKind.VIDEO)

    def audio_tracks(self) -> Iterator[Track]:
        return (t for t in self.tracks if t.kind is TrackKind.AUDIO)

    def mixed_tracks(self) -> Iterator[Track]:
        return (t for t in self.tracks if t.kind is TrackKind.MIXED)

    def picture_tracks(self) -> tuple[Track, ...]:
        """絵を描きうるトラック（映像と混合） 並びは :attr:`tracks` のまま（奥から手前）"""
        return tuple(t for t in self.tracks if t.kind is not TrackKind.AUDIO)

    def sound_tracks(self) -> tuple[Track, ...]:
        """クリップの音を鳴らしうるトラック（音声と混合） 並びは :attr:`tracks` のまま

        映像トラックは入れない 映像トラックが鳴らすのは置いたシーンの音だけで、
        トラックの音量・定位も掛けない（ミキサの決まり）
        """
        return tuple(t for t in self.tracks if t.kind is not TrackKind.VIDEO)

    def active_picture_tracks(self) -> tuple[Track, ...]:
        """実際に映るトラック（映像と混合） 並びは :attr:`tracks` のまま"""
        return _audible_or_visible(self.picture_tracks())

    def active_sound_tracks(self) -> tuple[Track, ...]:
        """実際に聞こえるトラック（音声と混合） 並びは :attr:`tracks` のまま"""
        return _audible_or_visible(self.sound_tracks())

    def active_tracks(self, kind: TrackKind) -> tuple[Track, ...]:
        """``kind`` のトラックのうち実際に映る／聞こえるもの 並びは :attr:`tracks` のまま

        映像は :meth:`active_picture_tracks`、音声は :meth:`active_sound_tracks` から
        その種類だけを抜く ソロの決まりをここで別に持たないため 混合トラックは
        絵の側（映るもの）を返す 混合トラックは絵と音でソロの効き方が違うことがあるので、
        役割の分かっている呼び手は種類ではなく役割のメソッドを使う
        """
        if kind is TrackKind.AUDIO:
            return tuple(t for t in self.active_sound_tracks() if t.kind is kind)
        return tuple(t for t in self.active_picture_tracks() if t.kind is kind)

    def find_track(self, track_id: TrackId) -> Track | None:
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None

    def locate_clip(self, clip_id: ClipId) -> tuple[Track, Clip] | None:
        """クリップとその所属トラックを探す"""
        for track in self.tracks:
            clip = track.find(clip_id)
            if clip is not None:
                return track, clip
        return None

    def replace_track(self, track: Track) -> Timeline:
        """同じ ID のトラックを差し替えた新しい :class:`Timeline` を返す"""
        for index, existing in enumerate(self.tracks):
            if existing.id == track.id:
                tracks = (*self.tracks[:index], track, *self.tracks[index + 1 :])
                return replace(self, tracks=tracks)
        raise KeyError(f"トラックが見つからない: {track.id}")

    def linked_clips(self, group: GroupId) -> Iterator[tuple[Track, Clip]]:
        """同じリンクグループに属するクリップをすべて返す"""
        for track in self.tracks:
            for clip in track.clips:
                if clip.link_group == group:
                    yield track, clip

    def grouped_clips(self, group: GroupId) -> Iterator[tuple[Track, Clip]]:
        """同じグループ（束ね）に属するクリップをすべて返す"""
        for track in self.tracks:
            for clip in track.clips:
                if clip.group_id == group:
                    yield track, clip

    def scene_references(self) -> set[SceneId]:
        """このタイムラインに置かれているシーン"""
        return {
            clip.scene_id
            for track in self.tracks
            for clip in track.clips
            if clip.scene_id is not None
        }


def _audible_or_visible(tracks: tuple[Track, ...]) -> tuple[Track, ...]:
    """``tracks`` のうちミュートとソロで残るもの

    ミュートを除き、``tracks`` にソロが 1 本でもあればソロのものだけを残す
    ミュートとソロが両方付いていればミュートが勝ち、そのソロはほかを止めない
    止めると、ソロを外し忘れたトラックをミュートしただけで全部が無音になる

    **ソロは役割（絵か音か）の中で決まる** 絵の側は映像と混合、音の側は音声と混合の中で
    見る 混合トラックは両方に入るので、混合トラックをソロにすると、ほかの映像トラックは
    映らず、ほかの音声トラックも鳴らない（そのレイヤーだけが見えて聞こえる）
    音声トラックをソロにしても映像トラックと混合トラックの絵は消えず、混合トラックの
    音だけが止まる 分ける方式で音声トラックのソロが絵を消さないのと揃えた
    種類ごと（映像の中・音声の中・混合の中）に決めると、混合トラックをソロにしても
    映像トラックが映り続け、音付きの動画を 1 本だけ確かめる、というソロの使い道が無くなる
    混合と分ける方式のトラックが並ぶのは移り変わりの間だけで、そこでも同じ役割の中で
    比べておけば、変換の前後でソロの意味が変わらない

    プレビュー・ミキサ・書き出しの 3 か所が必ずここを通る 判断が分かれると
    「プレビューでは消えているのに書き出すと出る」が起きる 実際に書き出しだけ
    ソロを見ていなかった
    """
    soloed = any(t.solo and not t.muted for t in tracks)
    return tuple(t for t in tracks if not t.muted and (t.solo or not soloed))


def draws_picture(track: Track, clip: Clip, media: MediaItem | None) -> bool:
    """``track`` に置いた ``clip`` が絵を描くか ``media`` はクリップの素材（無ければ ``None``）

    重ねに入るかどうかをここで決める 偽のクリップは重ねから外すので、上のクリップの
    :attr:`Clip.clip_to_below` の相手にもならない 無効（:attr:`Clip.enabled`）かどうかは
    見ない 編集で切り替える状態で、種類の決まりとは別に呼び手が見る

    - 映像トラック 常に描く（音だけの素材は置く時点で断っている）
    - 音声トラック 描かない
    - 混合トラック :attr:`Clip.show_picture` が真で、素材に絵がある（映像か静止画）か
      素材を持たないクリップ（テキスト・図形・シーン・フィルタ・場面切り替え）なら描く
      音だけの素材は描かない 描く物の無いクリップを重ねに残すと、上のクリップが
      それで切り抜いて何も映らなくなる

    素材を探せなかった（消えた）ときは描く側に数える 今の映像トラックと同じく、
    描けずに空いた所になる 描かない側に数えると、素材を戻しただけで切り抜きの相手が
    入れ替わる
    """
    if track.kind is TrackKind.AUDIO:
        return False
    if track.kind is TrackKind.VIDEO:
        return True
    if not clip.show_picture:
        return False
    if clip.media_id is None or media is None:
        return True
    return media.has_video or media.is_still


def plays_sound(track: Track, clip: Clip, media: MediaItem | None) -> bool:
    """``track`` に置いた ``clip`` の音を鳴らすか ``media`` はクリップの素材（無ければ ``None``）

    - 音声トラック 素材を持つクリップと、置いたシーン
    - 映像トラック 置いたシーンだけ（シーンの中の BGM やナレーションを消さないため）
    - 混合トラック 置いたシーンと、素材にある :attr:`Clip.audio_stream` を持つクリップ
      番号が ``None`` か素材に無い番号なら鳴らさない

    素材を探せなかったときは鳴らす側に数える（ミキサが開けずに無音になる）
    無効かどうかは :func:`draws_picture` と同じく見ない
    """
    if clip.scene_id is not None:
        return True
    if clip.media_id is None or track.kind is TrackKind.VIDEO:
        return False
    if track.kind is TrackKind.AUDIO:
        return True
    if clip.audio_stream is None:
        return False
    # 素材に無い番号は鳴らさない デコーダは無い番号を頼まれると先頭の音へ逃げるので、
    # 手で直したファイルや差し替えた素材で、選んでいない言語が鳴る
    return media is None or any(s.index == clip.audio_stream for s in media.audio_streams)


def group_layers(clip: Clip) -> int:
    """グループ制御が受け持つトラックの本数 0 なら手前のすべて 壊れた値も 0 より小さくしない"""
    value = clip.source.params.get(GROUP_LAYERS) if clip.source is not None else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 1
    return max(0, int(value))


def group_as_one(clip: Clip) -> bool:
    """グループ制御が受け持つ物を 1 枚の絵にしてから掛けるか（:data:GROUP_AS_ONE）"""
    return clip.source is not None and clip.source.params.get(GROUP_AS_ONE) is True


def group_reaches(
    tracks: Sequence[Track], group_track: TrackId, group: Clip, track_id: TrackId
) -> bool:
    """`group_track` に置いた `group` が `track_id` のトラックを受け持つか

    `tracks` は :func:controlling_groups と同じ並び（描かない物も含む）
    """
    ids = [track.id for track in tracks]
    if group_track not in ids or track_id not in ids:
        return False
    distance = ids.index(track_id) - ids.index(group_track)
    reach = group_layers(group)
    return distance > 0 and (reach == 0 or distance <= reach)


def controlling_groups(
    tracks: Sequence[Track], track_id: TrackId, frame: int
) -> list[tuple[Track, Clip]]:
    """``track_id`` のトラックの ``frame`` の絵を動かすグループ制御 近い物から

    ``tracks`` は描くトラックの並び（奥から手前 :meth:`Timeline.picture_tracks`）
    数えるのはこの並びの本数 描いていない（ミュートした）トラックの本数も数えるよう、
    呼ぶ側は描くかどうかで間引く前の並びを渡し、描かないグループ制御は ``enabled`` と
    同じく呼ぶ側が外す（AviUtl は非表示のレイヤーも番号として数える）
    グループ制御がグループ制御を受け持つこともある（入れ子） 近い物から並べて返すので、
    呼ぶ側は近い物から順に当てる（内側のグループの動きを、外側のグループがさらに動かす）
    """
    position = next((index for index, track in enumerate(tracks) if track.id == track_id), None)
    if position is None:
        return []
    found: list[tuple[Track, Clip]] = []
    for index in range(position - 1, -1, -1):
        track = tracks[index]
        clip = track.clip_at(frame)
        if clip is None or not clip.enabled or not clip.is_group:
            continue
        reach = group_layers(clip)
        if reach == 0 or position - index <= reach:
            found.append((track, clip))
    return found


def heard_stream(track: Track, clip: Clip) -> int | None:
    """``track`` に置いた ``clip`` が鳴らす音声ストリームの番号 音を持たなければ ``None``

    音声トラックは :attr:`Clip.stream_index`、混合トラックは :attr:`Clip.audio_stream`
    混合トラックの ``stream_index`` は絵のストリームを指す 音を鳴らす所（ミキサ）と
    波形を描く所が同じ決まりで番号を選ぶため、ここ 1 か所に置く 別々に選ぶと、
    鳴っている音と違う音の波形が出る
    """
    if track.kind is TrackKind.MIXED:
        return clip.audio_stream
    return clip.stream_index
