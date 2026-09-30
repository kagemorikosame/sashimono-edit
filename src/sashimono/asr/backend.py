"""字幕起こしのバックエンド抽象

実装を差し替えられるようにしてあるのは、音声認識の実装が数年で入れ替わるため
初期実装は faster-whisper（CTranslate2）だが、この層より上は「音声ファイルを渡すと
:class:`~sashimono.core.model.Transcript` が返る」という約束しか知らない

返す時刻はすべて**素材内のソース秒** タイムライン上の位置は決して持たせない
（:mod:`sashimono.core.projection` を参照）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from sashimono.core.model import Transcript

__all__ = [
    "MODELS",
    "AsrError",
    "ModelInfo",
    "Progress",
    "ShouldCancel",
    "TranscribeOptions",
    "TranscriptionBackend",
    "to_source_time",
]

#: 進捗の通知 第 1 引数は 0..1 の割合、第 2 引数は画面に出す短い文言
type Progress = Callable[[float, str], None]

#: 中断の問い合わせ 真を返したらバックエンドは速やかに ``None`` を返す
type ShouldCancel = Callable[[], bool]

#: 時刻の分解能（1 秒あたり） 字幕はミリ秒より細かくしても意味が無く、
#: 有理数の分母を揃えておくと分割や結合を繰り返しても値が暴れない
TIME_RESOLUTION = 1000


class AsrError(RuntimeError):
    """起こしに失敗した 導入不足・モデルの取得失敗・素材の異常をまとめて表す"""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """選べるモデル 1 つ"""

    name: str
    label: str
    #: おおよそのディスク使用量（MB） 初回はここに書いた分だけ取得が走る
    size_mb: int
    note: str = ""

    def describe(self) -> str:
        size = f"{self.size_mb / 1000:.1f} GB" if self.size_mb >= 1000 else f"{self.size_mb} MB"
        return (
            f"{self.label}（{size}）" if not self.note else f"{self.label}（{size}・{self.note}）"
        )


#: 既定で選べるモデル 日本語は large 系でないと実用にならないので、既定は
#: large-v3 にしてある turbo は速度重視で、下書き用
MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("large-v3", "large-v3", 3100, "精度重視"),
    ModelInfo("large-v3-turbo", "large-v3-turbo", 1600, "速度重視"),
    ModelInfo("medium", "medium", 1500),
    ModelInfo("small", "small", 480),
    ModelInfo("base", "base", 150, "確認用"),
    ModelInfo("tiny", "tiny", 75, "確認用"),
)

DEFAULT_MODEL = MODELS[0].name


@dataclass(frozen=True, slots=True)
class TranscribeOptions:
    """1 回の起こしの条件"""

    model: str = DEFAULT_MODEL
    #: ``None`` なら自動判定 決め打ちできるなら指定した方が精度が上がる
    language: str | None = "ja"
    #: ``"auto"`` / ``"cuda"`` / ``"cpu"``
    device: str = "auto"
    #: ``"auto"`` / ``"float16"`` / ``"int8_float16"`` / ``"int8"``
    compute_type: str = "auto"
    beam_size: int = 5
    #: 無音区間を先に落としてから認識に掛ける 長い無音での幻聴が減る
    vad_filter: bool = True
    #: 単語単位のタイムスタンプ 既定は切る 取得コストが上がる割に、使うのは
    #: クリップ分割時の境界決定だけなので、必要な素材でだけ入れる
    word_timestamps: bool = False
    #: 固有名詞などを与えると認識が寄る
    initial_prompt: str = ""
    #: 起こす音声ストリームの番号（:attr:`AudioStreamInfo.index`） ``None`` なら 1 本目
    #: ゲームの録画（ゲームの音とマイクの声）のように音声が何本もある素材で、声の方を選ぶ
    audio_stream: int | None = None

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise ValueError(f"ビーム幅は 1 以上必要: {self.beam_size}")


def to_source_time(seconds: float) -> Fraction:
    """バックエンドが返す秒（float）を、モデルが使う有理数へ

    float のまま持つと、分割・結合を繰り返すうちに端数が積もって字幕の境界が
    1 フレーム単位で揺れる ここで 1 度だけミリ秒へ落とす
    """
    return Fraction(round(max(0.0, seconds) * TIME_RESOLUTION), TIME_RESOLUTION)


class TranscriptionBackend(Protocol):
    """音声を文字に起こすもの"""

    @property
    def name(self) -> str:
        """画面に出す実装名"""
        ...

    def is_available(self) -> bool:
        """いま実行できるか 導入されていなければ偽"""
        ...

    def transcribe(
        self,
        path: Path,
        options: TranscribeOptions,
        *,
        progress: Progress | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> Transcript | None:
        """起こす 中断された場合は ``None`` を返す

        失敗は :class:`AsrError` を投げる
        """
        ...
