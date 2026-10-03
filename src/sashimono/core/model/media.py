"""メディアプールの素材

素材はタイムラインとは独立に存在する 同じ素材を何度タイムラインに置いても
実体は 1 つで、字幕もサムネイルも波形もこちらに紐付く
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

from sashimono.core.model.ids import MediaId, new_media_id
from sashimono.core.model.transcript import Transcript
from sashimono.core.timebase import FrameRate

__all__ = [
    "HDR_TRANSFERS",
    "WIDE_GAMUT_PRIMARIES",
    "AudioStreamInfo",
    "MediaItem",
    "VideoStreamInfo",
]

#: HDR の伝達特性（FFmpeg の名前） → 呼び名 PQ は HDR10 と Dolby Vision、HLG は放送と
#: スマホの HDR 動画
HDR_TRANSFERS = {"smpte2084": "PQ", "arib-std-b67": "HLG"}

#: SDR より広い色域の原色（FFmpeg の名前） → 呼び名 HDR の印が無くても、BT.2020 の原色の
#: まま Rec.709 として出すと、色が淡く褪せる
WIDE_GAMUT_PRIMARIES = {"bt2020": "広色域（BT.2020）"}


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    """素材に含まれる映像ストリーム 1 本の情報"""

    index: int
    width: int
    height: int
    frame_rate: FrameRate
    time_base: Fraction
    codec: str
    pixel_format: str = ""
    #: コンテナに記録された回転角（0 / 90 / 180 / 270） スマホ撮影で頻出する
    rotation: int = 0
    #: 映像の道の終わりの時刻（秒 フレームの表示時刻と同じく素材の頭から数える Issue #123）
    #: 分からなければ ``None``
    #:
    #: :attr:`MediaItem.duration` はコンテナ全体の長さで、音の方が長い素材では映像の
    #: 終わりより後ろを指す 最後の絵で止める時刻（``Clip.hold_at``）をそこから決めると、
    #: 映像の最後のフレームより後ろを読みに行き、止めた後も毎フレームデコーダを動かす
    end_time: Fraction | None = None
    #: 伝達特性の印（FFmpeg の名前 ``bt709`` ``smpte2084`` ``arib-std-b67`` など）
    #: 印が無い・分からなければ空
    color_transfer: str = ""
    #: 原色の印（FFmpeg の名前 ``bt709`` ``bt2020`` など） 印が無い・分からなければ空
    color_primaries: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"解像度が不正: {self.width}x{self.height}")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError(f"回転角が不正: {self.rotation}")
        if self.end_time is not None and self.end_time < 0:
            raise ValueError(f"映像の終わりの時刻が負: {self.end_time}")

    @property
    def color_outside_sdr(self) -> str:
        """SDR（Rec.709）の範囲の外の色なら、その呼び名（``HDR（PQ）`` など） 範囲の中なら空

        Sashimono は HDR と広い色域を変換せずに SDR として描くので、この素材は白っぽく
        褪せて出る（docs/development.md「色の範囲」） 見分けは印だけで行う 印の無い
        素材の中身を推し量ると、SDR の素材まで HDR と言い当ててしまう
        """
        transfer = HDR_TRANSFERS.get(self.color_transfer)
        if transfer is not None:
            return f"HDR（{transfer}）"
        return WIDE_GAMUT_PRIMARIES.get(self.color_primaries, "")

    @property
    def display_size(self) -> tuple[int, int]:
        """回転を適用した後の表示サイズ"""
        if self.rotation in (90, 270):
            return self.height, self.width
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """素材に含まれる音声ストリーム 1 本の情報

    多言語音声や 5.1ch の素材では複数本あり、読み込み時にそれぞれ別トラックへ
    展開する
    """

    index: int
    sample_rate: int
    channels: int
    time_base: Fraction
    codec: str
    language: str | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"サンプリングレートが不正: {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"チャンネル数が不正: {self.channels}")


@dataclass(frozen=True, slots=True)
class MediaItem:
    """メディアプールに登録された 1 つの素材"""

    path: Path
    #: 素材全体の長さ（秒） 静止画では 0
    duration: Fraction = Fraction(0)
    video_streams: tuple[VideoStreamInfo, ...] = ()
    audio_streams: tuple[AudioStreamInfo, ...] = ()
    #: 字幕起こしの結果 トラックではなくここに持たせるのが設計の要
    #: 音声ストリームごとに持つ（ストリームの番号 :attr:`AudioStreamInfo.index`, 字幕）
    #: ゲームの録画のように音声が何本もある素材では、音ごとに別の字幕になる 前は素材に
    #: 1 つだけで、音声 2 を起こすと音声 1 の字幕が置き換わった 並びは番号の昇順
    transcripts: tuple[tuple[int, Transcript], ...] = ()
    #: 空ならファイル名を表示名として使う
    display_name: str = ""
    id: MediaId = field(default_factory=new_media_id)

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError(f"長さが負: {self.duration}")

    @property
    def name(self) -> str:
        return self.display_name or self.path.name

    @property
    def has_video(self) -> bool:
        return len(self.video_streams) > 0

    @property
    def has_audio(self) -> bool:
        return len(self.audio_streams) > 0

    @property
    def color_outside_sdr(self) -> str:
        """映像のどれかが SDR の範囲の外なら、その呼び名 無ければ空

        映像を何本も持つ素材は、最初に見つかった物で呼ぶ どれか 1 本でも外なら、
        その映像を置いたときに褪せて出る
        """
        return next(
            (name for stream in self.video_streams if (name := stream.color_outside_sdr)), ""
        )

    @property
    def is_still(self) -> bool:
        """静止画のように、任意の長さで使える素材か"""
        return self.duration == 0 and not self.has_audio

    def transcript_stream(self, stream: int | None = None) -> int:
        """字幕を引く鍵にする音声ストリームの番号

        ``None`` と素材に無い番号は 1 本目 デコーダも素材に無い番号では 1 本目を開くので、
        鳴っている音と字幕の音が食い違わない 音声の無い素材は 0
        """
        known = [s.index for s in self.audio_streams]
        if stream is not None and stream in known:
            return stream
        return known[0] if known else 0

    def transcript_for(self, stream: int | None = None) -> Transcript | None:
        """その音声ストリームの字幕 ``None`` なら 1 本目"""
        key = self.transcript_stream(stream)
        return next((t for index, t in self.transcripts if index == key), None)

    @property
    def transcript(self) -> Transcript | None:
        """1 本目の音声の字幕 音声が 1 本の素材ではこれがその素材の字幕"""
        return self.transcript_for(None)

    def with_transcript(
        self, transcript: Transcript | None, stream: int | None = None
    ) -> MediaItem:
        """その音声ストリーム（``None`` なら 1 本目）の字幕を差し替えた新しい素材

        ほかの音の字幕は残す
        """
        key = self.transcript_stream(stream)
        kept = [(index, t) for index, t in self.transcripts if index != key]
        if transcript is not None:
            kept.append((key, transcript))
        return replace(self, transcripts=tuple(sorted(kept, key=lambda pair: pair[0])))
