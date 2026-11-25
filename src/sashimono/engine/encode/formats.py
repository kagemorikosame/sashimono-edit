"""書き出しの形式（入れ物）と、形式ごとに使える物（#527）

入れ物は名前の拡張子で FFmpeg が選ぶ 画面の窓と AI の書き出しと書き出しの係が、ここから同じ
形式の表を引く 別々に持つと、窓では選べるのに AI では断られる・書き出しの係だけ別の音を
入れようとして断られる、が起きる 拡張子で形式を決めるのは、別のプロセスの書き出しや AI の
書き出しが仮の名前で書くときにも、拡張子だけは残して渡しているため（:mod:`export_job`
:mod:`sashimono.ui.ai_export`）

Qt を使わない AI の道具の確かめ（編集画面のスレッドの外でも呼ぶ）からも使う
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AUDIO_KBPS_CHOICES",
    "CHAPTER_SUFFIXES",
    "DEFAULT_AUDIO_KBPS",
    "DEFAULT_FORMAT",
    "FORMATS",
    "FORMAT_KEYS",
    "VIDEO_STANDARD",
    "VIDEO_WEB",
    "ExportFormat",
    "format_named",
    "format_of",
    "with_format_suffix",
]

#: 映像を H.264・H.265 など今までの候補で書く形式（MP4・MOV・MKV）
VIDEO_STANDARD = "standard"
#: 映像を VP9 か AV1 で書く形式（WebM） WebM の入れ物は H.264 を受けない
VIDEO_WEB = "web"

#: チャプターを埋める入れ物 mp4・mov は章の文字トラック、mkv は Chapters の要素で持つ
#: ほかの入れ物（webm・m4a・音声だけの物）は章を持てないか、プレーヤーが読まない 渡すと多重化に
#: 断られて書き出しごと失敗しうるので、埋めない
CHAPTER_SUFFIXES = frozenset({".mp4", ".mov", ".mkv"})

#: 音声だけの書き出しで選べる音質（kbps） MP3 と M4A（AAC）だけが使う
#: 320 は MP3 の上限 256 より上は聞き分けにくいが、配る先が求めることがあるので並べる
AUDIO_KBPS_CHOICES = (128, 192, 256, 320)
#: 窓の初めの値 知らない人が選ばずに書き出しても、BGM やナレーションで粗さの分からない所
DEFAULT_AUDIO_KBPS = 192


@dataclass(frozen=True, slots=True)
class ExportFormat:
    """書き出しの形式 1 つ"""

    #: 設定と AI に渡す名前（拡張子から点を除いた物）
    key: str
    #: 窓に出す名前
    label: str
    #: 名前の終わり 入れ物はこれで FFmpeg が選ぶ
    suffix: str
    #: 映像の作り方 :data:`VIDEO_STANDARD` か :data:`VIDEO_WEB` 空なら音声だけ
    video: str
    #: 音声のコーデック
    audio_codec: str
    #: 音質（:data:`AUDIO_KBPS_CHOICES`）を選べるか 非可逆の音声だけの形式だけ
    audio_quality: bool = False

    @property
    def audio_only(self) -> bool:
        """映像を作らず、音だけを書く形式か"""
        return not self.video

    @property
    def chapters(self) -> bool:
        """チャプターを埋められる入れ物か"""
        return self.suffix in CHAPTER_SUFFIXES


#: 窓に並べる順 映像の形式を先に、音声だけの物を後ろにまとめる
#: 映像の形式の音は今までどおり AAC WebM は入れ物が AAC を受けないので Opus
#: 音声だけの WAV は 16 bit（配る先で読めない所が無い） FLAC は可逆で WAV より小さい
FORMATS = (
    ExportFormat("mp4", "MP4", ".mp4", VIDEO_STANDARD, "aac"),
    ExportFormat("mov", "MOV", ".mov", VIDEO_STANDARD, "aac"),
    ExportFormat("mkv", "MKV", ".mkv", VIDEO_STANDARD, "aac"),
    ExportFormat("webm", "WebM", ".webm", VIDEO_WEB, "libopus"),
    ExportFormat("wav", "WAV（音声だけ）", ".wav", "", "pcm_s16le"),
    ExportFormat("mp3", "MP3（音声だけ）", ".mp3", "", "libmp3lame", audio_quality=True),
    ExportFormat("m4a", "M4A（音声だけ）", ".m4a", "", "aac", audio_quality=True),
    ExportFormat("flac", "FLAC（音声だけ）", ".flac", "", "flac"),
)

#: 形式の名前の一覧 設定の値と AI の引数を確かめる
FORMAT_KEYS = tuple(item.key for item in FORMATS)

#: 窓の初めの形式 前に選んだ物を覚えていなければこれ
DEFAULT_FORMAT = "mp4"

_BY_KEY = {item.key: item for item in FORMATS}
_BY_SUFFIX = {item.suffix: item for item in FORMATS}


def format_named(key: str) -> ExportFormat:
    """名前から形式を引く 知らない名前は ``ValueError``"""
    found = _BY_KEY.get(key)
    if found is None:
        raise ValueError(f"書き出しの形式は {', '.join(FORMAT_KEYS)} のどれかです: {key}")
    return found


def format_of(path: Path) -> ExportFormat | None:
    """名前の拡張子に当たる形式 知らない拡張子なら ``None``

    大文字の拡張子（``.MP4``）も同じ形式に読む FFmpeg も大きさを問わずに入れ物を選ぶ
    """
    return _BY_SUFFIX.get(path.suffix.lower())


def with_format_suffix(path: Path, chosen: ExportFormat) -> Path:
    """``path`` の名前の終わりを ``chosen`` の拡張子にする

    知っている形式の拡張子なら付け替え、知らない物（``作品.v2`` の ``.v2`` など）なら後ろへ足す
    付け替えると、名前の一部として付けた点の後ろが消える
    """
    # 名前の無い場所（空の欄やドライブだけ）は付ける先が無い そのまま返し、書き出す前の
    # 拡張子の確かめで断らせる
    if not path.name or path.suffix.lower() == chosen.suffix:
        return path
    if format_of(path) is not None:
        return path.with_suffix(chosen.suffix)
    return path.with_name(path.name + chosen.suffix)
