r"""編集の正しさを、人の編集（正解）と比べて数える（#345 D7 道具の段）

    .venv\Scripts\python.exe tools\eval_tasks.py draft              # 仮の正解と直す用の .sme を作る
    .venv\Scripts\python.exe tools\eval_tasks.py adopt E04 silence  # 直した .sme を正解にする
    .venv\Scripts\python.exe tools\eval_tasks.py run                # 道具を走らせて正解と比べる
    .venv\Scripts\python.exe tools\eval_tasks.py report             # 前回との差を出す

素材は**リポジトリの外**のフォルダ（環境変数 ``SASHIMONO_EVAL_DIR`` か ``--dir``）に置く
番号（E04 など）と実物の対応表 ``manifest.json``・正解・直す用の .sme もそこに置く
リポジトリに入れるのは、この台本と課題の定義（``eval/tasks.json``）と、数えた結果
（``eval/results/tools.jsonl``）だけ 結果を書く前に、素材の名前や置き場が混じって
いないかを確かめ、混じっていれば 1 行も書かずに止まる（:func:`guard_record`）

道具の段は AI を通さず、編集の道具が使う処理（無音の検出・場面の検出・字幕起こし）を
直に呼ぶ 決まった結果が出て無料なので、既定値や感度を変えた前と後を比べるのに使う
「話の区切り」は道具が無い（AI が字幕を読んで決める）ので、道具の段では数えない
「書き出し」は正解と比べず、字幕を焼き込んで書き出した動画を ``check_output`` と同じ検査に
掛け、見つかった問題の数と書き出しの秒を数える（短い素材 2 本だけ GPU が要る）

字幕起こしは開発の環境（.venv）へ入れず、アプリが入れた部品（LOCALAPPDATA の Sashimono の
runtime）を読む（:func:`load_app_runtime`） 部品の無い機械では字幕の課題を飛ばす

正解は、本人が直すまでは道具の下書き（仮の正解）で、結果の行に ``provisional`` の印を
付ける 仮の正解と同じ道具の結果を比べると 1.0 に近くなるのは当たり前で、その数は
「前と変わったか」の目安にしかならない 本人が直す手順は docs/evaluation.md に書いた
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sashimono import __version__  # noqa: E402
from sashimono.core.commands import (  # noqa: E402
    AddMarkers,
    AddTrack,
    Document,
    RippleCut,
    SetTranscript,
)
from sashimono.core.commands.insert import insert_media  # noqa: E402
from sashimono.core.io import load_project, save_project  # noqa: E402
from sashimono.core.io.atomic import write_atomic  # noqa: E402
from sashimono.core.jetcut import plan_cuts  # noqa: E402
from sashimono.core.model import (  # noqa: E402
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Transcript,
    TranscriptSegment,
)
from sashimono.core.model.timeline import Marker  # noqa: E402
from sashimono.core.timebase import FrameRate  # noqa: E402

#: 課題の名前 結果の行と正解の JSON の鍵に使うので ASCII にしておく
SILENCE = "silence"
TALK = "talk"
SCENES = "scenes"
SUBTITLES = "subtitles"
#: 字幕を焼き込んで書き出し、書き出した動画を検査する 正解と比べる課題ではなく、
#: 書き出せたかと、検査（``check_output`` と同じ）で見つかった問題の数を数える
EXPORT = "export"
#: 見どころを選ぶ（#414） 人が素材の全体を見て残した区間を正解にし、``find_highlights`` と同じ
#: 処理の区間と比べる
HIGHLIGHTS = "highlights"
#: グループを保ったままカットする（#437） 正解は持たず、カットの後にグループが残ったかと、
#: AI の段では ``ungroup_clips`` を呼んだ・尋ねた回数を数える
GROUPS = "groups"
TASKS = (SILENCE, TALK, SCENES, SUBTITLES, EXPORT, HIGHLIGHTS, GROUPS)
#: 正解を持つ課題 下書きと取り込みはこれだけ
ANSWERED = (SILENCE, TALK, SCENES, SUBTITLES, HIGHLIGHTS)
#: 道具の下書きを仮の正解にせず、人が素材の全体を見て一から付ける課題（#345 の決まり）
#: 下書きから直すと、直し忘れがそのまま正解になって正解が道具の癖に寄る
BLANK_SHEETS = (SCENES, HIGHLIGHTS)

#: 結果の行の段 道具の段（AI を通さない）と、AI に指示を出して測る段（#437）
STAGES = ("tools", "ai")
#: 結果の行の状態
STATUSES = ("ok", "skipped", "failed")
#: 数えなかった理由 結果の行の ``reason`` はこの中の物だけを書ける
#: 段・課題・状態・理由は名前の照らしを外している（年のフォルダや export のフォルダに当たるため）
#: 決まった値だけに絞らないと、将来ここへ例外の文や素材から来る文字を入れたときに、
#: 名前の照らしをすり抜けて素材の名前が公開のリポジトリへ載る
#: docs/evaluation.md の「状態と理由」の表と同じ物を並べる（試験が食い違いを見る）
REASONS = (
    "no-material",
    "no-answer",
    "no-tool",
    "asr-missing",
    "asr-error",
    "export-error",
    "tool-error",
    "answer-error",
    "no-consent",
    "ai-error",
    "ai-timeout",
    "ai-no-cut",
)

#: 課題の定義と、結果を足していく所（どちらもリポジトリの中）
TASKS_FILE = ROOT / "eval" / "tasks.json"
RESULTS_FILE = ROOT / "eval" / "results" / "tools.jsonl"
#: AI の段の結果 道具の段と分けるのは、AI の段は本人が同意して回したときだけ増え、回すたびに
#: 費用がかかるので、道具の段の行と混ぜると前回との差の読み方を取り違えやすいため
AI_RESULTS_FILE = ROOT / "eval" / "results" / "ai.jsonl"

#: 外のフォルダを指す環境変数
EVAL_DIR_VARIABLE = "SASHIMONO_EVAL_DIR"

#: 無音の一致を比べる刻み（秒） 素材のフレームの刻みに合わせると、fps の違う素材同士で
#: 一致率の重みが変わる 10ms なら 60fps の 1 コマより細かい
GRID_SECONDS = 0.01

#: 話の区切りの下書きに使う間の長さ（秒） これより長い無音の真ん中を区切りの候補にする
#: 息継ぎ（0.5 秒前後）で区切ると候補が多すぎて、直す手間が一から付けるのと変わらない
TALK_PAUSE_SECONDS = 1.5

#: 結果の行に書いてよい文字 日本語と区切り記号（/ \）を通さないので、素材の名前・置き場・
#: 字幕の文は形の上で入らない 版（0.2.3）・コミット・日時・番号（E04）はこれで書ける
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9._:+-]{0,64}$")
#: 字幕起こしのモデルの名前のうち、Hugging Face の形（組織/名前 例 Systran/faster-whisper-large-v3）
#: 結果の ``params.model`` にだけ許す 手元の置き場は :data:`LOCAL_MODEL` に置き換えて書く
_MODEL_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}/[A-Za-z0-9._-]{1,96}$")
#: 手元のモデルの置き場を渡した回に、結果へ書く印 場所は書かない（本人の置き場が公開される）
LOCAL_MODEL = "local-model"
#: 結果の行の一番上の鍵 増やすときはここにも足す 知らない鍵は書かずに止まる
_RECORD_KEYS = frozenset(
    {
        "date",
        "version",
        "commit",
        "stage",
        "material",
        "task",
        "status",
        "reason",
        "provisional",
        "seconds",
        "metrics",
        "params",
    }
)
#: 道具が自分で作る値の鍵 素材から来ないので、名前の照らしから外す（形の決まりは見る）
#: 日時・版・コミット・段・番号（課題の定義から）・課題・状態・理由（固定の文字）のどれも、
#: 素材の名前や置き場を入れる道が無い 照らすと、年ごとのフォルダ（2026）が日時に、
#: export や tools という名のフォルダが課題や段の名前に当たり、毎回 1 行も書けずに止まる
_OWN_KEYS = frozenset(
    {"date", "version", "commit", "stage", "material", "task", "status", "reason", "provisional"}
)
#: 名前の照らしで、これより短い名前は見ない 2 文字の名前（例 R）は版やコミットの中にも
#: 現れるので、照らすと正しい行まで止めてしまう 短い名前は形の決まり（_SAFE_TEXT）で守る
_MIN_NAME_LENGTH = 3


class LeakError(RuntimeError):
    """結果の行に、リポジトリへ入れてはいけない物が混じっていた"""


# --- 課題の定義と外のフォルダ ---


@dataclass(frozen=True, slots=True)
class Tolerances:
    """一致とみなす幅"""

    #: 場面の切り替わり（フレーム） 素材の fps で秒に直す
    scene_frames: int = 2
    #: 話の区切り（秒） 人によって答えが割れるので広めにする
    talk_seconds: float = 1.0
    #: 字幕の切れ目（秒）
    subtitle_seconds: float = 0.5


@dataclass(frozen=True, slots=True)
class Material:
    """課題を当てる素材 1 本 リポジトリでは番号だけで呼ぶ"""

    id: str
    tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskBook:
    """課題の定義（``eval/tasks.json``）"""

    tolerances: Tolerances
    materials: tuple[Material, ...]


_MATERIAL_ID = re.compile(r"^E\d{2}$")


def load_tasks(path: Path = TASKS_FILE) -> TaskBook:
    """課題の定義を読む 形が違えばその場で止める

    番号の形（E と 2 桁）と課題の名前を確かめる 定義に素材の名前を書いてしまっても、
    ここで止まってリポジトリへ入る前に気付ける
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("tolerances", {})
    default = Tolerances()
    tolerances = Tolerances(
        scene_frames=int(raw.get("scene_frames", default.scene_frames)),
        talk_seconds=float(raw.get("talk_seconds", default.talk_seconds)),
        subtitle_seconds=float(raw.get("subtitle_seconds", default.subtitle_seconds)),
    )
    materials: list[Material] = []
    for entry in data.get("materials", []):
        identifier = str(entry.get("id", ""))
        if not _MATERIAL_ID.match(identifier):
            raise ValueError(f"素材の番号は E と 2 桁の数字にする: {identifier!r}")
        if set(entry) - {"id", "tasks"}:
            raise ValueError(f"{identifier} に知らない鍵がある（番号と課題だけを書く）")
        tasks = tuple(str(task) for task in entry.get("tasks", []))
        unknown = [task for task in tasks if task not in TASKS]
        if unknown:
            raise ValueError(f"{identifier} の課題が分からない: {unknown}")
        materials.append(Material(identifier, tasks))
    return TaskBook(tolerances, tuple(materials))


@dataclass(frozen=True, slots=True)
class EvalDir:
    """リポジトリの外の評価のフォルダ

    ``manifest.json`` は ``{"E04": "D:/どこか/素材.mp4", ...}`` の形 相対の場所は
    このフォルダから読む
    """

    root: Path

    @property
    def answers(self) -> Path:
        return self.root / "answers"

    @property
    def drafts(self) -> Path:
        return self.root / "drafts"

    def manifest(self) -> dict[str, Path]:
        path = self.root / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"対応表がありません: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(key): (self.root / str(value)).resolve() for key, value in dict(data).items()}

    def answer_path(self, material: str) -> Path:
        return self.answers / f"{material}.json"

    @property
    def exports(self) -> Path:
        """書き出しの課題が書く動画 回すたびに上書きする（見返せるように残す）"""
        return self.root / "exports"

    def draft_path(self, material: str, task: str) -> Path:
        return self.drafts / f"{material}_{task}.sme"

    def read_answer(self, material: str) -> dict[str, Any]:
        path = self.answer_path(material)
        if not path.exists():
            return {"material": material, "tasks": {}}
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def write_answer(self, material: str, answer: Mapping[str, Any]) -> None:
        # その場で上書きすると、書く途中で落ちたときに切れたファイルが残り、本人が数時間かけて
        # 直した正解（同じファイルに入っている）まで読めなくなる 作り直せないので fsync もする
        self.answers.mkdir(parents=True, exist_ok=True)
        # 裸の NaN を書くと、次に読むときに厳格な読み手が正解のファイルを読めない
        text = json.dumps(answer, ensure_ascii=False, indent=1, allow_nan=False)
        write_atomic(self.answer_path(material), text + "\n", durable=True)


def eval_dir(argument: str | None) -> EvalDir:
    """``--dir`` か環境変数から外のフォルダを決める どちらも無ければ止める

    既定の置き場を決め打ちしない リポジトリの中を既定にすると、素材の対応表や
    正解（字幕の文を含む）をうっかり入れてしまう
    """
    chosen = argument or os.environ.get(EVAL_DIR_VARIABLE, "")
    if not chosen:
        raise SystemExit(f"評価のフォルダを --dir か {EVAL_DIR_VARIABLE} で指してください")
    root = Path(chosen).resolve()
    if root == ROOT or ROOT in root.parents:
        raise SystemExit("評価のフォルダはリポジトリの外に置いてください（素材の名前が入るため）")
    return EvalDir(root)


def forbidden_names(folder: EvalDir, manifest: Mapping[str, Path]) -> list[str]:
    """結果の行に現れてはいけない名前 素材のファイル名・名前の幹・置き場のフォルダ名

    置き場はフォルダ名を 1 つずつ照らす 場所をまるごと照らすだけだと、区切りの書き方
    （/ と \\）を変えただけで素通りする
    """
    names: set[str] = set()
    for path in [*manifest.values(), folder.root]:
        names.update({path.name, path.stem})
        names.update(part for part in path.parent.parts if part not in {path.anchor, ""})
    return sorted(name for name in names if len(name) >= _MIN_NAME_LENGTH)


# --- 道具の段: 道具が使う処理を直に呼ぶ ---


def _is_local_model(value: str) -> bool:
    """字幕起こしのモデルの指定が、手元の置き場か

    名前（large-v3）と Hugging Face の名前（組織/名前）は名前として扱う ただし手元に同じ名前の
    フォルダがあれば、字幕起こしもそちらを読むので置き場として扱う
    """
    path = Path(value)
    if "\\" in value or path.is_absolute() or path.drive or value.startswith((".", "~")):
        return True
    if path.exists():
        return True
    return "/" in value and not _MODEL_NAME.match(value)


class ParamsError(ValueError):
    """道具の値が使えない 走らせる前に、何がいけないかを人に分かる文で出す"""


@dataclass(frozen=True, slots=True)
class ToolParams:
    """道具の既定値 変える前と後を比べるために、引数で差し替えられるようにしておく"""

    silence_threshold_db: float = -40.0
    min_silence: float = 0.5
    padding: float = 0.1
    scene_threshold: float | None = None
    min_scene: float | None = None
    asr_model: str | None = None

    def __post_init__(self) -> None:
        # nan や inf は結果の行に書けない（標準の JSON に無い値で、厳格な読み手が行を読めない）
        # 道具を走らせてから書く段で止まると数分が無駄になるので、作る所で断る
        for name in (
            "silence_threshold_db",
            "min_silence",
            "padding",
            "scene_threshold",
            "min_scene",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ParamsError(f"{name} に有限の数でない値（{value}）は使えない")
        self._check_scene_ranges()
        if self.asr_model is not None and not self.asr_model.strip():
            raise ParamsError("asr_model に空の名前は使えない（省くと既定のモデル）")

    def _check_scene_ranges(self) -> None:
        """場面の値は、アプリの AI の道具が受け付ける範囲だけを許す

        範囲の外は、アプリでは選べない値なので測っても使い道が無い 感度 0 は手持ちの揺れや
        圧縮のノイズまで切り替わりに数え、一覧が素材のフレーム数に近づく
        最短の場面の 0 はアプリでも選べる（寄せずにすべて残す）ので許す
        """
        if self.scene_threshold is None and self.min_scene is None:
            return
        # 重いので、場面の値を変えたときだけ読む
        from sashimono.ai.operations import MAX_MIN_SCENE_SECONDS, SCENE_THRESHOLD_RANGE

        low, high = SCENE_THRESHOLD_RANGE
        if self.scene_threshold is not None and not low <= self.scene_threshold <= high:
            raise ParamsError(
                f"scene_threshold は {low} から {high} の間にする"
                "（アプリの場面検出が受け付ける範囲）"
            )
        if self.min_scene is not None and not 0.0 <= self.min_scene <= MAX_MIN_SCENE_SECONDS:
            raise ParamsError(f"min_scene は 0 から {MAX_MIN_SCENE_SECONDS} 秒の間にする")

    @property
    def scene_values(self) -> tuple[float, float]:
        """場面検出に渡す感度と最短の場面 省いた物は既定値

        ``or`` で既定値を選ぶと 0.0 を偽と見て、最短の場面 0 が既定値の 0.5 に置き換わり、
        結果の行にも渡していない値が書かれる
        """
        from sashimono.engine.scenes import DEFAULT_MIN_SCENE, DEFAULT_THRESHOLD

        threshold = DEFAULT_THRESHOLD if self.scene_threshold is None else self.scene_threshold
        min_scene = DEFAULT_MIN_SCENE if self.min_scene is None else self.min_scene
        return threshold, min_scene

    def record(self, task: str) -> dict[str, float | str]:
        """結果の行に残す値 課題に関係する物だけ"""
        if task == SILENCE:
            return {
                "threshold_db": self.silence_threshold_db,
                "min_silence": self.min_silence,
                "padding": self.padding,
            }
        if task == SCENES:
            threshold, min_scene = self.scene_values
            return {"threshold": threshold, "min_scene": min_scene}
        if task == SUBTITLES:
            from sashimono.asr.backend import DEFAULT_MODEL

            if self.asr_model is None:
                return {"model": DEFAULT_MODEL}
            # 手元の置き場は場所を伏せて印だけ書く 書くと本人のフォルダ名が公開の結果に載る
            return {"model": LOCAL_MODEL if _is_local_model(self.asr_model) else self.asr_model}
        return {}


@dataclass(frozen=True, slots=True)
class MediaFacts:
    """正解に一緒に残す素材の長さと fps 場面の許す幅を秒に直すのに使う"""

    duration: float
    fps: float
    #: 音を持つか 無い素材を字幕起こしへ渡すと失敗し、見どころまで asr-error になる
    has_audio: bool = True


def media_facts(path: Path) -> MediaFacts:
    from sashimono.engine.decode.probe import probe_media

    media = probe_media(path)
    fps = float(media.video_streams[0].frame_rate.fps) if media.video_streams else 0.0
    return MediaFacts(float(media.duration), fps, media.has_audio)


def find_silences(path: Path, params: ToolParams) -> list[tuple[float, float]]:
    """無音の区間（素材の秒） ジェットカット（``jet_cut``）と同じ処理"""
    from sashimono.engine.audio.silence import SilenceOptions, detect_silence
    from sashimono.engine.audio.waveform import analyze_waveform

    waveform = analyze_waveform(path)
    if waveform is None:
        return []
    options = SilenceOptions(
        threshold_db=params.silence_threshold_db,
        min_silence=Fraction(params.min_silence).limit_denominator(1000),
        padding=Fraction(params.padding).limit_denominator(1000),
    )
    return [(float(start), float(end)) for start, end in detect_silence(waveform, options)]


def kept_ranges(silences: Sequence[tuple[float, float]], duration: float) -> list[list[float]]:
    """無音を除いて残る区間 正解と比べるのは残した方（人が編集で残した物と同じ向き）"""
    kept: list[list[float]] = []
    cursor = 0.0
    for start, end in sorted(silences):
        if start > cursor:
            kept.append([cursor, min(start, duration)])
        cursor = max(cursor, end)
    if cursor < duration:
        kept.append([cursor, duration])
    return [pair for pair in kept if pair[1] > pair[0]]


def find_scenes(path: Path, params: ToolParams) -> list[float]:
    """場面の切り替わり（新しい場面の最初の絵の秒） ``detect_scenes`` と同じ処理"""
    from sashimono.engine.scenes import analyze_scenes

    # 0 は機械のコアをすべて使う 評価は手元で 1 本ずつ回すので、ほかの仕事と取り合わない
    signal = analyze_scenes(path, threads=0)
    if signal is None:
        return []
    cuts = signal.cuts(*params.scene_values)
    return [cut.seconds for cut in cuts]


def find_highlight_ranges(
    path: Path, duration: float, transcript: Transcript | None
) -> list[list[float]]:
    """見どころの区間（素材の秒） ``find_highlights`` と同じ手がかりと既定のしきい値

    字幕があれば笑いの印を主な手がかりにする（道具と同じ） 字幕起こしの部品が無ければ音と動きだけ
    """
    from sashimono.engine.audio.waveform import analyze_waveform
    from sashimono.engine.decode.probe import probe_media
    from sashimono.engine.highlights import highlight_curves
    from sashimono.engine.scenes import analyze_scenes

    # 音の無い素材の波形・絵の無い素材の場面は読めない（読むと失敗する） 道具と同じく、
    # 無い手がかりは調べずに返す
    media = probe_media(path)
    curves = highlight_curves(
        duration,
        waveform=analyze_waveform(path) if media.has_audio else None,
        scenes=analyze_scenes(path, threads=0) if media.has_video else None,
        transcript=transcript,
    )
    return [[found.start, found.end] for found in curves.highlights()]


#: 字幕起こしを返す物 試験では偽物に差し替える 入っていなければ None
type Transcriber = Callable[[Path, ToolParams], Transcript | None]


def transcribe_with_app(path: Path, params: ToolParams) -> Transcript | None:
    """アプリと同じ字幕起こし（整形まで） 実行環境が入っていなければ None"""
    from sashimono.asr import TranscribeOptions, activate_runtime, clean_transcript
    from sashimono.asr import default_backend as asr_backend

    activate_runtime()
    load_app_runtime()
    backend = asr_backend()
    if not backend.is_available():
        return None
    options = (
        TranscribeOptions()
        if params.asr_model is None
        else TranscribeOptions(model=params.asr_model)
    )
    transcript = backend.transcribe(path, options)
    return None if transcript is None else clean_transcript(transcript)


def app_runtime_dir() -> Path:
    """配布版のアプリが画面のボタンから入れた部品の置き場（LOCALAPPDATA の Sashimono の runtime）"""
    from sashimono.core import userdirs
    from sashimono.runtime import _RUNTIME_DIR

    return userdirs.data_root() / _RUNTIME_DIR


def load_app_runtime(target: Path | None = None) -> Path | None:
    """アプリが入れた字幕起こしの部品を、評価の台本から読めるようにする 読めた置き場を返す

    開発の環境（.venv）へは字幕起こしを入れない（2026-10-20 本人が決めた 2 GB 近くあり、
    アプリが入れた物と同じ物を二重に持つことになる） 開発の実行ではアプリの
    ``activate_runtime`` が何もしない（配布版だけの決まり）ので、ここで同じ置き場を読む

    探す道の**後ろ**へ足す 前へ足すと、置き場にある numpy などの別の版が .venv の物より
    先に読まれ、本体の試験と違う部品で数えることになる 別の Python 向けに入れた置き場は
    読まない（拡張モジュールが読めず、import した所で落ちる）
    """
    from sashimono.runtime import python_abi, read_path_files, runtime_abi

    place = target if target is not None else app_runtime_dir()
    if not place.is_dir():
        return None
    found = runtime_abi(place, "asr")
    if found is not None and found != python_abi():
        return None
    if str(place) not in sys.path:
        read_path_files(str(place))
    return place


def segments_of(transcript: Transcript) -> list[list[Any]]:
    return [[float(s.start), float(s.end), s.text] for s in transcript.segments]


def talk_from_silences(silences: Sequence[tuple[float, float]]) -> list[float]:
    """話の区切りの下書き 長い間の真ん中 道具の段では数えず、仮の正解を作るのにだけ使う"""
    return [
        round((start + end) / 2, 3) for start, end in silences if end - start >= TALK_PAUSE_SECONDS
    ]


# --- 書き出しと検査 ---

#: 書き出しの課題を行う物 試験では GPU の無い CI でも回るよう偽物に差し替えられる
type Exporter = Callable[[Path, Transcript | None, Path], dict[str, float | int | bool | None]]

#: 書き出した動画の検査を待つ上限（秒） 短い素材 2 本だけに当てる課題なので、数分で終わる
CHECK_TIMEOUT_SECONDS = 1800.0


def _application() -> object:
    """書き出し（GPU の合成）と検査（Qt のスレッド）に要る Qt の入れ物 無ければ作る"""
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    from sashimono.engine.gpu import preferred_surface_format

    existing = QApplication.instance()
    if existing is not None:
        return existing
    QSurfaceFormat.setDefaultFormat(preferred_surface_format())
    return QApplication([])


def burned_project(media_path: Path, transcript: Transcript | None) -> Project:
    """素材を置き、字幕があれば画面の〔焼き込み〕と同じ既定の見た目で焼き込んだ作品"""
    from sashimono.core.commands.subtitle import (
        burn_defaults,
        burn_subtitles,
        subtitle_wrap_width,
    )
    from sashimono.effects.sources import TEXT
    from sashimono.engine.decode.probe import probe_media
    from sashimono.ui.workspace import Preferences

    media = probe_media(media_path)
    document = _project_for(media, "export")
    if transcript is not None and transcript.segments:
        document.execute(SetTranscript(media.id, transcript))
        settings = document.project.settings
        # 折り返しの幅は本人の設定の既定（知らない人が困らない側）に合わせる
        wrap = subtitle_wrap_width(settings.width, Preferences().subtitle_wrap_share)
        base = TEXT.create(**burn_defaults(settings.height, wrap))
        for command in burn_subtitles(document.project, base):
            document.execute(command)
    return document.project


def export_and_check(
    media_path: Path, transcript: Transcript | None, output: Path
) -> dict[str, float | int | bool | None]:
    """字幕を焼き込んで書き出し、書き出した動画を ``check_output`` と同じ検査に掛ける

    数えるのは、問題の数（種類ごと）・音量（LUFS）・長さのずれ・書き出しと検査の秒
    字幕のはみ出しは書き出した動画では調べられない（検査の決まり）ので数に入らない
    """
    from sashimono.engine.checks import CheckJob, CheckLimits, CheckRequest
    from sashimono.engine.checks.scan import KINDS
    from sashimono.engine.encode.exporter import ExportSettings, export_project

    _application()
    project = burned_project(media_path, transcript)
    expected = float(project.duration * project.rate.frame_duration)
    began = time.perf_counter()
    export_project(project, ExportSettings(path=output))
    exported = time.perf_counter()
    job = CheckJob()
    try:
        stat = output.stat()
        job.start(
            CheckRequest(
                CheckLimits(expected_seconds=expected),
                path=output,
                stamp=(stat.st_size, stat.st_mtime_ns),
            )
        )
        if not job.waiter()(CHECK_TIMEOUT_SECONDS):
            raise RuntimeError("書き出した動画の検査が時間内に終わらなかった")
        state = job.state()
    finally:
        job.close()
    if state.failure is not None or state.report is None:
        raise RuntimeError(f"書き出した動画を検査できなかった: {state.failure}")
    report = state.report
    counts = report.counts()
    metrics: dict[str, float | int | bool | None] = {
        "problems": len(report.problems),
        "complete": report.complete,
        "loudness": None if report.loudness is None else round(report.loudness, 2),
        "length_error": round(report.seconds - expected, 3),
        # 焼き込んだ字幕はテキストのクリップ トラックの名前で数えると、方式や話し手の数で
        # 名前が変わったときに 0 と数える
        "subtitle_clips": sum(
            1
            for track in project.timeline.tracks
            for clip in track.clips
            if clip.source is not None and clip.source.kind == "text"
        ),
        "export_seconds": round(exported - began, 2),
        "check_seconds": round(time.perf_counter() - exported, 2),
    }
    for kind in KINDS:
        metrics[f"count_{kind}"] = counts.get(kind, 0)
    return metrics


# --- グループを保つ（#437） ---

#: AI の段でグループの課題に出す指示 素材の名前も中身も言わない（どの素材でも同じ文で比べる）
#: 保存と書き出しを止めるのは、評価の実行が本人のフォルダへ書かないため（確かめの窓でも断る）
GROUP_INSTRUCTION = (
    "話していない間を詰めて、テンポよく見られるようにしてください 保存と書き出しはしないでください"
)


#: グループの課題の前半と後半の見出し 2 つのグループがまとまったかを見分けるのに使う
_GROUP_TITLES = ("前半", "後半")


def grouped_project(media_path: Path, material: str) -> Document:
    """素材を置き、前半と後半に分けて、それぞれ映像・音声・テロップを 1 つのグループにした作品

    人がテロップと話し手を束ねてからカットを頼む形 前半と後半で別のグループにするのは、
    カットでグループがまとめて 1 つになる崩れ（#427）も数えられるようにするため
    """
    from sashimono.core.commands import GroupClips, SplitClips
    from sashimono.core.model import Track, TrackKind
    from sashimono.effects.sources import TEXT
    from sashimono.engine.decode.probe import probe_media

    media = probe_media(media_path)
    document = _project_for(media, f"{material} {GROUPS}")
    project = document.project
    middle = project.duration // 2
    if middle <= 0:
        raise ValueError("素材が短すぎてグループを 2 つに分けられない")
    placed = [clip.id for track in project.timeline.tracks for clip in track.clips]
    document.execute(SplitClips(tuple(placed), middle))
    titles = (
        Clip(timeline_start=0, duration=middle, source=TEXT.create(text=_GROUP_TITLES[0])),
        Clip(
            timeline_start=middle,
            duration=project.duration - middle,
            source=TEXT.create(text=_GROUP_TITLES[1]),
        ),
    )
    document.execute(AddTrack(Track(TrackKind.VIDEO, "テロップ", titles)))
    for low, high in ((0, middle), (middle, document.project.duration)):
        members = tuple(
            clip.id
            for track in document.project.timeline.tracks
            for clip in track.clips
            if clip.timeline_start >= low and clip.timeline_end <= high
        )
        document.execute(GroupClips(members))
    return document


def group_metrics(project: Project) -> dict[str, float | int]:
    """カットの後にグループが残っているか

    ``kept_share`` は、時刻の重なる別のトラックのクリップの組のうち、同じグループに居る組の割合
    前半と後半のグループは時刻で分かれているので、重なる組はどれも同じグループの仲間だった物
    解かれると 1.0 より下がる 割った後ろの片割れが 1 本だけになった物（#427 の決まりで
    グループから外れる）は、重なる相手が居ないので数に入らない
    """
    clips = [
        (index, clip) for index, track in enumerate(project.timeline.tracks) for clip in track.clips
    ]
    pairs = same = 0
    for position, (track, clip) in enumerate(clips):
        for other_track, other in clips[position + 1 :]:
            if other_track == track:
                continue
            if (
                clip.timeline_start < other.timeline_end
                and other.timeline_start < clip.timeline_end
            ):
                pairs += 1
                same += int(clip.group_id is not None and clip.group_id == other.group_id)
    groups = {clip.group_id for _, clip in clips if clip.group_id is not None}
    # 前半と後半の見出しが同じグループに入ったら、2 つのグループが 1 つにまとまった 重なる組は
    # どれも同じグループのままなので割合は 1.0 のまま残る 別々に動かせないのは束ねが崩れた形
    titles = {
        str(clip.source.params.get("text")): clip.group_id
        for _, clip in clips
        if clip.source is not None and clip.source.params.get("text") in _GROUP_TITLES
    }
    merged = int(
        len(titles) == len(_GROUP_TITLES)
        and None not in titles.values()
        and len(set(titles.values())) == 1
    )
    return {
        "kept_share": 0.0 if merged else (round(same / pairs, 4) if pairs else 1.0),
        "merged": merged,
        "pairs": pairs,
        "groups": len(groups),
        "loose_clips": sum(1 for _, clip in clips if clip.group_id is None),
        "clips": len(clips),
    }


def cut_groups_with_tools(path: Path, params: ToolParams) -> dict[str, float | int]:
    """道具の段 無音カット（``jet_cut`` と同じ命令）をグループの作品に掛けて数える"""
    document = grouped_project(path, "tools")
    media = document.project.media[0]
    silences = [
        (Fraction(a).limit_denominator(1000), Fraction(b).limit_denominator(1000))
        for a, b in find_silences(path, params)
    ]
    ranges = plan_cuts(document.project, media.id, silences)
    if ranges:
        document.execute(RippleCut(ranges))
    return {**group_metrics(document.project), "cuts": len(ranges)}


@dataclass(slots=True)
class AiRun:
    """AI の段の 1 回で数えた物"""

    tool_calls: int = 0
    ungroup_calls: int = 0
    ungroup_questions: int = 0
    questions: int = 0
    refused_irreversible: int = 0
    finished: bool = False
    failed: bool = False

    def metrics(self) -> dict[str, int]:
        return {
            "tool_calls": self.tool_calls,
            "ungroup_calls": self.ungroup_calls,
            "ungroup_questions": self.ungroup_questions,
            "questions": self.questions,
            "refused_irreversible": self.refused_irreversible,
        }


class SessionLike(Protocol):
    """AI の段が使う会話の形（:class:`~sashimono.ai.session.AgentSession` と同じ口）"""

    def send(self, prompt: str) -> None: ...

    def poll(self) -> list[Any]: ...

    def close(self, *, wait: bool = True) -> None: ...


#: 会話を作る物（ブリッジと作品の置き方の方式から） 試験では偽の会話へ差し替える
type SessionFactory = Callable[[Any, str], SessionLike]

#: AI の段の 1 つの指示を待つ上限（秒） 長い素材の無音カットと確かめで 10 分ほどかかる
AI_TIMEOUT_SECONDS = 1800.0
#: 〔グループを解く前に聞く〕の問いに返す選択肢の位置 0 が〔解除する〕、1 が〔解除しない〕
#: 評価ではグループを残す答えを返し、残したまま AI がカットを終えられるかを見る
_KEEP_GROUP = 1


def real_session(model: str | None) -> SessionFactory:
    """本物の AI（Claude Code）との会話を作る物 指示の文は画面の AI のパネルと同じ"""

    def make(bridge: Any, layer_mode: str) -> SessionLike:
        from sashimono.ai.session import AgentSession, system_prompt

        return AgentSession(
            bridge,
            model=model,
            system_prompt=system_prompt(layer_mode, ask=True, plan=False),
            ask=True,
            plan=False,
        )

    return make


def drive_ai(
    window: Any,
    instruction: str,
    session_factory: SessionFactory,
    *,
    timeout: float = AI_TIMEOUT_SECONDS,
) -> AiRun:
    """本物の編集画面で AI に指示を 1 つ出し、終わるまで確かめの窓と問いに答える

    取り消せる編集は許し（変更を自動で承認と同じ）、取り消せない操作（保存・書き出し）は断る
    グループを解く前の問いには〔解除しない〕、ほかの問いには決めずに進めるよう書いて返す
    外へ送る物はどれも許す（呼ぶ側が本人の同意を取ってから呼ぶ）
    """
    from PySide6.QtWidgets import QApplication

    from sashimono.ai.bridge import EditorBridge
    from sashimono.ai.outbound import ASKED, SendChoice
    from sashimono.ai.session import EventKind

    bridge = EditorBridge(window, auto_approve=True)
    bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
    bridge.ask_first = True
    run = AiRun()
    session = session_factory(bridge, window.document.project.settings.layer_mode)
    bridge.begin_turn()
    session.send(instruction)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline and not (run.finished or run.failed):
            _answer_once(bridge, run)
            for event in session.poll():
                if event.kind is EventKind.TOOL_USE:
                    run.tool_calls += 1
                    run.ungroup_calls += int(event.tool == "ungroup_clips")
                elif event.kind is EventKind.TURN_DONE:
                    run.finished = True
                elif event.kind in (EventKind.ERROR, EventKind.CLOSED):
                    run.failed = True
            QApplication.processEvents()
            time.sleep(0.01)
    finally:
        bridge.end_turn()
        bridge.cancel()
        session.close(wait=False)
    return run


def _answer_once(bridge: Any, run: AiRun) -> None:
    """画面が出すはずの確かめと問いに 1 回ずつ答える（AI のパネルの代わり）"""
    bridge.pump()
    approval = bridge.take_approval()
    if approval is not None:
        if approval.irreversible:
            run.refused_irreversible += 1
            approval.deny("評価の実行では保存と書き出しをしません")
        else:
            approval.allow()
    question = bridge.take_question()
    if question is not None:
        run.questions += 1
        if question.gate == "ungroup_clips":
            run.ungroup_questions += 1
            question.reply((question.spec.choices[_KEEP_GROUP].label,))
        else:
            question.reply(
                free_text="評価の実行なので答えられません 決めずに進められる所まで進めてください"
            )
    budget = bridge.take_budget_request()
    if budget is not None:
        budget.reply(False)


def run_ai_groups(
    path: Path,
    sheet: Path,
    material: str,
    session_factory: SessionFactory,
    *,
    timeout: float = AI_TIMEOUT_SECONDS,
) -> tuple[AiRun, dict[str, float | int]]:
    """AI の段のグループの課題 グループの作品を本物の編集画面で開き、AI にカットを頼む

    作品は外のフォルダの ``drafts`` に置く（AI が書ける場所を作品のフォルダに限るため）
    """
    from sashimono.ui.main_window import MainWindow

    _application()
    document = grouped_project(path, material)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    save_project(document.project, sheet)
    window = MainWindow(load_project(sheet), path=sheet, confirm_unsaved=False)
    before = document.project.duration
    try:
        run = drive_ai(window, GROUP_INSTRUCTION, session_factory, timeout=timeout)
        after = window.document.project
        metrics = group_metrics(after)
    finally:
        window.close()
    # 切らずに終えた回は、グループが残っていて当たり前 数えないと切れなかった回まで成功に見える
    removed = (before - after.duration) * after.rate.frame_duration
    metrics["removed_seconds"] = round(float(removed), 2)
    return run, metrics


# --- 一致率 ---


def _grid(ranges: Sequence[Sequence[float]], cells: int) -> list[bool]:
    marks = [False] * cells
    for start, end in ranges:
        first = max(0, round(start / GRID_SECONDS))
        last = min(cells, round(end / GRID_SECONDS))
        for index in range(first, last):
            marks[index] = True
    return marks


def _edges(ranges: Sequence[Sequence[float]], duration: float) -> list[float]:
    """区間の端 素材の頭と終わりは、残す・削るの判断ではないので数えない"""
    edges = {float(value) for pair in ranges for value in pair}
    return sorted(edge for edge in edges if 0.0 < edge < duration)


def interval_scores(
    predicted: Sequence[Sequence[float]], answer: Sequence[Sequence[float]], duration: float
) -> dict[str, float | None]:
    """残した区間の一致 iou は残した所の重なり agreement は残す・削るが合った割合

    境界のずれは、正解の端ごとに一番近い道具の端までの秒の中央値 一致率だけだと、
    長い素材で端が全部 0.3 秒ずれていても 0.99 になり、聞いて分かるずれが数に出ない
    """
    cells = max(1, math.ceil(duration / GRID_SECONDS))
    mine = _grid(predicted, cells)
    truth = _grid(answer, cells)
    both = sum(1 for a, b in zip(mine, truth, strict=True) if a and b)
    either = sum(1 for a, b in zip(mine, truth, strict=True) if a or b)
    same = sum(1 for a, b in zip(mine, truth, strict=True) if a == b)
    answer_edges = _edges(answer, duration)
    mine_edges = _edges(predicted, duration)
    # 道具の端が無いまま min() に渡すと空の列で止まり、同じ回のほかの結果まで書かれない
    offsets = (
        [min(abs(edge - other) for other in mine_edges) for edge in answer_edges]
        if mine_edges
        else []
    )
    # 正解に端があるのに道具に端が無いときは測れない（None） 0.0 にすると「ずれ無し」と読めて、
    # 何も切らなかった道具を良いと取り違える 正解にも端が無ければ、ずれは無い（0.0）
    unmeasured = None if answer_edges else 0.0
    edge_median = round(statistics.median(offsets), 3) if offsets else unmeasured
    return {
        "iou": round(both / either, 4) if either else 1.0,
        "agreement": round(same / cells, 4),
        "edge_median": edge_median,
        "kept_seconds": round(sum(mine) * GRID_SECONDS, 2),
        "answer_kept_seconds": round(sum(truth) * GRID_SECONDS, 2),
    }


def range_scores(
    predicted: Sequence[Sequence[float]], answer: Sequence[Sequence[float]], duration: float
) -> dict[str, float | None]:
    """見どころの区間の一致 重なりの一致（:func:`interval_scores`）に、区間の当たりを足す

    見どころは区間の数が少なく、端の位置より「その場面を選んだか」が大事 正解の区間に 1 つでも
    重なる道具の区間があれば当たり（再現率）、道具の区間が正解のどれかに重なれば当たり（適合率）
    """
    scores: dict[str, float | None] = dict(interval_scores(predicted, answer, duration))

    def overlaps(item: Sequence[float], others: Sequence[Sequence[float]]) -> bool:
        return any(float(item[0]) < float(b) and float(a) < float(item[1]) for a, b in others)

    found = sum(1 for item in answer if overlaps(item, predicted))
    right = sum(1 for item in predicted if overlaps(item, answer))
    precision = right / len(predicted) if predicted else (1.0 if not answer else 0.0)
    recall = found / len(answer) if answer else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    scores |= {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted": len(predicted),
        "answer": len(answer),
    }
    return scores


def _span(item: float | Sequence[float]) -> tuple[float, float]:
    """正解の 1 つ 数なら点、[頭, 終わり] ならゆっくりした場面転換の区間"""
    if isinstance(item, int | float):
        return float(item), float(item)
    return float(item[0]), float(item[1])


def point_scores(
    predicted: Sequence[float], answer: Sequence[float | Sequence[float]], tolerance: float
) -> dict[str, float]:
    """位置の一致 許す幅の中で正解と 1 対 1 に組み、適合率・再現率・F1 を出す

    組める数が一番多くなる組み方をする 正解を受け付ける範囲の終わりが早い順に、まだ
    使っていない一番早い道具の位置を当てる（直線上の区間と点の組では、これで最多になる）
    近い組から順に決めると、2 つの正解の間にある位置が近い方に取られ、遠い方と組めた
    はずの隣の位置が余って、実際より低い数が出る
    """
    spans = [_span(item) for item in answer]
    windows = sorted((end + tolerance + 1e-9, start - tolerance - 1e-9) for start, end in spans)
    moments = sorted(float(moment) for moment in predicted)
    used = [False] * len(moments)
    hits = 0
    for right, left in windows:
        for index, moment in enumerate(moments):
            if used[index] or moment < left:
                continue
            if moment <= right:
                used[index] = True
                hits += 1
            break
    precision = hits / len(predicted) if predicted else (1.0 if not spans else 0.0)
    recall = hits / len(spans) if spans else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hits": hits,
        "predicted": len(predicted),
        "answer": len(spans),
    }


#: 文字の誤り率で比べない文字 字幕の整形（句点の付け方など）の好みで数が動くと、
#: 聞き取りの良し悪しが見えなくなる
#: 長音符（ー）は外さない 語の一部で、落としたり足したりするのは聞き取りの誤り（ビール と ビル）
#: 外すと、その誤りが 0 に見え、ほかの誤りの割合も分母が縮んで変わる
_IGNORED_CHARACTERS = re.compile(r"[\s、\N{IDEOGRAPHIC FULL STOP}，．,.!?！？「」『』（）()…・〜~]")


def character_error_rate(predicted: str, answer: str) -> float:
    """文字の誤り率（編集距離 ÷ 正解の文字数） 日本語は語の区切りが無いので文字で数える"""
    mine = _IGNORED_CHARACTERS.sub("", predicted)
    truth = _IGNORED_CHARACTERS.sub("", answer)
    if not truth:
        return 0.0 if not mine else 1.0
    previous = list(range(len(truth) + 1))
    for i, letter in enumerate(mine, start=1):
        current = [i]
        for j, other in enumerate(truth, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (letter != other))
            )
        previous = current
    return round(previous[-1] / len(truth), 4)


def subtitle_scores(
    predicted: Sequence[Sequence[Any]], answer: Sequence[Sequence[Any]], tolerance: float
) -> dict[str, float]:
    """字幕の文字の誤り率と、切れ目（各字幕の頭と終わり）の一致"""
    cer = character_error_rate(
        "".join(str(item[2]) for item in predicted), "".join(str(item[2]) for item in answer)
    )
    edges = point_scores(
        sorted({float(item[k]) for item in predicted for k in (0, 1)}),
        sorted({float(item[k]) for item in answer for k in (0, 1)}),
        tolerance,
    )
    return {
        "cer": cer,
        "edge_f1": edges["f1"],
        "segments": len(predicted),
        "answer_segments": len(answer),
    }


# --- 結果の行 ---


def finite_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """数えた値のうち有限でない数（無音の音量の -inf など）を ``None``（測れない）にする

    そのまま渡すと書く前の確かめで止まり、その回のほかの課題の結果まで 1 行も残らない
    数えた値は道具の都合で測れないことがあるので、止めずに「測れない」と書く
    引数から来る値（params）は人が直せるので、ここでは直さず確かめで断る
    """
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in metrics.items()
    }


def finite_float(text: str) -> float:
    """引数の数を読む nan や inf は、何がいけないかを書いて断る（結果の行に書けないため）"""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数ではない: {text}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            f"有限の数を入れる（nan や inf は結果の行に書けないので使えない）: {text}"
        )
    return value


def _check_names(value: str, where: str, names: Sequence[str]) -> None:
    lowered = value.lower()
    for name in names:
        if name.lower() in lowered:
            raise LeakError(f"{where} に素材の名前か置き場が混じっている")


def _check_value(value: object, where: str, names: Sequence[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        # 書くと標準の JSON に無い裸の NaN や Infinity になり、厳格な読み手が結果の行を読めない
        raise LeakError(f"{where} が有限の数でない")
    if value is None or isinstance(value, bool | int | float):
        return
    if isinstance(value, str):
        if not _SAFE_TEXT.match(value):
            raise LeakError(f"{where} に書けない文字がある（英数字と . _ : + - だけ）")
        _check_names(value, where, names)
        return
    raise LeakError(f"{where} の値の形が書けない物（{type(value).__name__}）")


def _own_models() -> tuple[str, str]:
    from sashimono.asr.backend import DEFAULT_MODEL

    return DEFAULT_MODEL, LOCAL_MODEL


def check_params(params: ToolParams, names: Sequence[str]) -> None:
    """結果の行に残す道具の値を、道具を走らせる前に確かめる だめなら :class:`ParamsError`

    書く段で初めて止まると、字幕起こしや書き出しを数分走らせた後でその回が 1 行も残らない
    確かめは書く段と同じ物を通す（別に書くと、片方だけ通る値ができる）
    値そのものは文に出さない 置き場が混じっているかもしれない
    """
    for task in TASKS:
        try:
            guard_record({"params": params.record(task)}, names)
        except LeakError as error:
            raise ParamsError(
                f"{error}（結果の行に書けない値なので、走らせる前に断った）"
            ) from None


def guard_record(record: Mapping[str, object], names: Sequence[str]) -> None:
    """結果の 1 行を、リポジトリへ書いてよいか確かめる だめなら :class:`LeakError`

    形の決まり（知っている鍵・英数字だけの文字）と、素材の名前の照らしの 2 段にする
    形だけだと英数字の素材の名前（録画の日時の名前など）が通り、照らすだけだと
    知らない名前（対応表に無い物）や字幕の文が通る
    """
    unknown = set(record) - _RECORD_KEYS
    if unknown:
        raise LeakError(f"知らない鍵がある: {sorted(unknown)}")
    # 名前を照らさない鍵のうち、台本が決まった値しか書かない物は、その一覧で絞る
    # 値そのものは出さない 素材の名前が混じっていたら、それを画面に出すことになる
    for key, allowed in (
        ("stage", STAGES),
        ("task", TASKS),
        ("status", STATUSES),
        ("reason", REASONS),
    ):
        if key in record and record[key] not in allowed:
            raise LeakError(f"{key} が決まった値の一覧に無い")
    for key, value in record.items():
        # 道具が自分で作る値は、形の決まりだけを見て名前は照らさない（下の _OWN_KEYS）
        # 中の鍵（metrics と params の数の名前）もコードが決めた名前なので同じ扱いにする
        # 素材や引数から来うる値（params と metrics の値）には名前の照らしを残す
        own = () if key in _OWN_KEYS else names
        if isinstance(value, Mapping):
            for inner, item in value.items():
                _check_value(str(inner), f"{key} の鍵", ())
                if key == "params" and inner == "model" and item in _own_models():
                    # 台本が決めたモデルの名前（既定のモデルと手元のモデルの印）は素材から来ない
                    # 照らすと、Local や large という名のフォルダの下に素材を置いただけで止まる
                    _check_value(item, f"{key}.{inner}", ())
                    continue
                if (
                    key == "params"
                    and inner == "model"
                    and isinstance(item, str)
                    and _MODEL_NAME.match(item)
                ):
                    # Hugging Face のモデルの名前（組織/名前）は / を持つが、場所ではないので許す
                    _check_names(item, f"{key}.{inner}", names)
                    continue
                _check_value(item, f"{key}.{inner}", names)
        else:
            _check_value(value, key, own)


def append_results(
    path: Path, records: Sequence[Mapping[str, object]], names: Sequence[str]
) -> None:
    """結果を足す 全部を確かめてから書く 1 行でもだめなら 1 行も書かない

    確かめながら書くと、だめな行の手前までが書かれ、同じ回の数が半端に残る
    """
    for record in records:
        guard_record(record, names)
    path.parent.mkdir(parents=True, exist_ok=True)
    _mend_tail(path)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            # allow_nan=False は守りの 2 段目 確かめを抜けて来ても裸の NaN を書かずに止まる
            file.write(
                json.dumps(record, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n"
            )


class BrokenResultsError(RuntimeError):
    """結果のファイルの途中の行が読めない 書く途中で切れたのではないので、直すまで読まない"""


def _parse_results(text: str) -> tuple[list[dict[str, Any]], int | None]:
    """結果の行を読む 最後の行が切れていれば、その行の頭の位置（文字）を添えて返す

    足して書く途中で落ちると、最後の行だけが切れて残る その 1 行のために全部を読めなくすると、
    report も次の run も止まり、それまでの回の数が見られなくなる 途中の行が壊れているのは
    切れたのではない（手で書き換えた・ほかの物が書いた）ので、黙って飛ばさずに止める
    行の中身は出さない 素材の名前や置き場が混じっているかもしれない
    """
    rows: list[dict[str, Any]] = []
    lines = text.splitlines(keepends=True)
    filled = [index for index, line in enumerate(lines) if line.strip()]
    for index in filled:
        try:
            rows.append(json.loads(lines[index]))
        except json.JSONDecodeError:
            if index != filled[-1]:
                raise BrokenResultsError(
                    f"結果のファイルの {index + 1} 行目が読めない（途中の行が壊れている）"
                ) from None
            return rows, sum(len(line) for line in lines[:index])
    return rows, None


def _mend_tail(path: Path) -> None:
    """足して書く前に、切れた最後の行を外し、改行で終わっていなければ改行を足す

    改行を足さずに書くと、切れた行と新しい行が 1 行につながって新しい行まで読めなくなる
    切れた行は、改行を足すだけだと次からは途中の壊れた行になって読めなくなるので外す
    （切れた行は JSON として読めず、数を取り出せないので、外しても失う物は無い）
    途中の行が壊れていれば、その上に足さずに止める
    """
    if not path.exists():
        return
    text = _raw_text(path)
    _, torn = _parse_results(text)
    if torn is not None:
        print("結果のファイルの最後の切れた行を外してから足す")
        with path.open("r+b") as file:
            file.truncate(len(text[:torn].encode("utf-8")))
    elif text and not text.endswith("\n"):
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write("\n")


def _raw_text(path: Path) -> str:
    """改行を読み替えずに読む 読み替えると、切れた行の頭の位置がファイルの位置とずれる"""
    with path.open(encoding="utf-8", newline="") as file:
        return file.read()


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows, torn = _parse_results(_raw_text(path))
    if torn is not None:
        print("結果のファイルの最後の行が切れていたので 1 行飛ばした")
    return rows


def current_commit(root: Path = ROOT, results: Path = RESULTS_FILE) -> str:
    """今のコミット 書き換えたまま回したときは印を付ける（どの形で測ったかが曖昧になるため）

    台本が自分で足す結果のファイルは、書き換えの判定から外す 外さないと、結果をコミットせずに
    2 回目を回すだけで必ず印が付き、コードを変えた回と変えていない回を見分けられない
    外すのは結果のフォルダごと（道具の段の ``tools.jsonl`` と AI の段の ``ai.jsonl``）
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ".",
                f":(exclude){results.parent.relative_to(root).as_posix()}",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


# --- 直す用の .sme と正解の取り込み ---


def _project_for(media: MediaItem, name: str) -> Document:
    """素材を 1 本だけ頭から置いた作品 解像度と fps は素材に合わせる"""
    if media.video_streams:
        stream = media.video_streams[0]
        settings = ProjectSettings(
            width=stream.width, height=stream.height, frame_rate=stream.frame_rate
        )
    else:
        settings = ProjectSettings()
    document = Document(Project.create(settings, name=name))
    for command in insert_media(document.project, media, at_frame=0):
        document.execute(command)
    return document


def draft_project(media_path: Path, material: str, task: str, answer: Mapping[str, Any]) -> Project:
    """正解を直すための作品 課題ごとに 1 つ作る（1 つにまとめると直す所が混ざる）

    - 無音: 無音を詰めた後のタイムライン 人はクリップの端を直す
    - 話の区切り: 素材をそのまま置き、区切りに目印を置く 人は目印を足す・消す・動かす
    - 場面: 素材をそのまま置くだけで目印は置かない 人が素材の全体を見て一から目印を置く
      （#345 の決まり 道具の下書きから直すと、直し忘れがそのまま正解になって正解が道具の癖に寄り、
      道具を良くしても数が上がらない）
    - 見どころ: 場面と同じ決まりで素材をそのまま置くだけ 人が見どころでない所を消して残す
    - 字幕: 素材に字幕を付ける 人は字幕のパネルで文と時刻を直す
    """
    from sashimono.engine.decode.probe import probe_media

    media = probe_media(media_path)
    document = _project_for(media, f"{material} {task}")
    rate = document.project.rate
    if task == SILENCE:
        kept = answer["kept"]
        duration = float(media.duration)
        silences = [
            (Fraction(a).limit_denominator(1000), Fraction(b).limit_denominator(1000))
            for a, b in _gaps(kept, duration)
        ]
        ranges = plan_cuts(document.project, media.id, silences)
        if ranges:
            document.execute(RippleCut(ranges))
    elif task == TALK:
        points = [_span(item)[0] for item in answer["boundaries"]]
        frames = sorted({round(moment * float(rate.fps)) for moment in points})
        markers = tuple(Marker(frame=frame, label="区切り") for frame in frames if frame > 0)
        if markers:
            document.execute(AddMarkers(markers))
    elif task == SUBTITLES:
        segments = tuple(
            TranscriptSegment(
                start=Fraction(start).limit_denominator(1000),
                end=Fraction(end).limit_denominator(1000),
                text=str(text),
            )
            for start, end, text in answer["segments"]
        )
        document.execute(SetTranscript(media.id, Transcript(segments=segments, language="ja")))
    return document.project


def _gaps(kept: Sequence[Sequence[float]], duration: float) -> list[tuple[float, float]]:
    """残す区間の隙間（＝削る区間）"""
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted((float(a), float(b)) for a, b in kept):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def _clips_of(project: Project, media_path: Path) -> tuple[MediaItem, list[Clip]]:
    """正解にする素材と、それを使っているクリップ 素材は置き場で探す（ID は作り直しで変わる）"""
    wanted = media_path.resolve()
    for media in project.media:
        if Path(media.path).resolve() == wanted:
            clips = [
                clip
                for track in project.timeline.tracks
                for clip in track.clips
                if clip.media_id == media.id
            ]
            return media, clips
    raise ValueError("作品の中に、その番号の素材がありません")


def _source_seconds(clips: Sequence[Clip], frame: int, rate: FrameRate) -> float | None:
    """タイムラインのフレームが、素材の何秒に当たるか クリップを動かされても合うように"""
    for clip in clips:
        if clip.timeline_start <= frame < clip.timeline_end:
            elapsed = (frame - clip.timeline_start) * rate.frame_duration * clip.speed
            return round(float(clip.source_in + elapsed), 3)
    return None


def answer_from_project(project: Project, media_path: Path, task: str) -> dict[str, Any]:
    """直した作品から正解を取り出す :func:`draft_project` の逆"""
    media, clips = _clips_of(project, media_path)
    rate = project.rate
    if task in (SILENCE, HIGHLIGHTS):
        ranges = sorted((float(clip.source_in), float(clip.source_out(rate))) for clip in clips)
        merged: list[list[float]] = []
        for start, end in ranges:
            # 絵と音の組は同じ区間を 2 回持つ 重ねてまとめる
            if merged and start <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        rounded = [[round(a, 3), round(b, 3)] for a, b in merged]
        return {"kept" if task == SILENCE else "ranges": rounded}
    if task in (SCENES, TALK):
        moments = [
            moment
            for marker in project.timeline.markers
            if (moment := _source_seconds(clips, marker.frame, rate)) is not None
        ]
        return {"cuts" if task == SCENES else "boundaries": sorted(moments)}
    if task == SUBTITLES:
        transcript = media.transcript_for(None)
        return {"segments": [] if transcript is None else segments_of(transcript)}
    raise ValueError(f"課題が分からない: {task}")


# --- 命令 ---


def _materials(book: TaskBook, chosen: Sequence[str]) -> list[Material]:
    if not chosen:
        return list(book.materials)
    known = {material.id: material for material in book.materials}
    missing = [identifier for identifier in chosen if identifier not in known]
    if missing:
        raise SystemExit(f"課題の定義に無い番号: {missing}")
    return [known[identifier] for identifier in chosen]


@dataclass(frozen=True, slots=True)
class Prediction:
    """道具の段の答え 1 つ ``value`` は正解と同じ形 ``None`` は数えられなかった"""

    value: dict[str, Any] | None
    seconds: float
    reason: str = ""


def predict(
    task: str, path: Path, params: ToolParams, transcriber: Transcriber, facts: MediaFacts
) -> Prediction:
    """道具を走らせて、正解と同じ形の答えを作る かかった秒も測る"""
    began = time.perf_counter()
    if task == SILENCE:
        value: dict[str, Any] | None = {
            "kept": kept_ranges(find_silences(path, params), facts.duration)
        }
    elif task == SCENES:
        value = {"cuts": find_scenes(path, params)}
    elif task == SUBTITLES:
        transcript = transcriber(path, params)
        if transcript is None:
            return Prediction(None, 0.0, "asr-missing")
        value = {"segments": segments_of(transcript)}
    elif task == HIGHLIGHTS:
        # 字幕起こしの部品が無ければ音と動きだけで選ぶ（道具の段の find_highlights と同じ）
        # 音の無い素材は起こさない 起こすと失敗して、動きで選べる見どころまで数えられない
        words = transcriber(path, params) if facts.has_audio else None
        value = {"ranges": find_highlight_ranges(path, facts.duration, words)}
    else:
        # 話の区切りの道具は無い AI の段（後で作る）で数える
        return Prediction(None, 0.0, "no-tool")
    return Prediction(value, round(time.perf_counter() - began, 2))


def with_facts(truth: Mapping[str, Any], facts: MediaFacts) -> dict[str, Any]:
    """課題 1 つの正解に、素材の長さと fps を添える 比べるときと .sme を作るときに要る

    正解のファイルに残った長さではなく、その回に素材から測った値を使う ``adopt --from`` で
    下書きを通さずに取り込んだ正解には長さが無く、0 のまま数えると無音の一致率が意味を失い、
    場面の許す幅も 30fps の決め打ちになって、誤った数が ok のまま残る
    """
    return {**truth, "duration": facts.duration, "fps": facts.fps}


def score(
    task: str, value: Mapping[str, Any], answer: Mapping[str, Any], book: TaskBook
) -> Mapping[str, float | None]:
    tolerances = book.tolerances
    if task == EXPORT:
        raise ValueError("書き出しの課題は正解と比べない")
    if task == SILENCE:
        return interval_scores(value["kept"], answer["kept"], float(answer["duration"]))
    if task == SCENES:
        fps = float(answer.get("fps") or 30.0)
        return point_scores(value["cuts"], answer["cuts"], tolerances.scene_frames / fps)
    if task == TALK:
        return point_scores(value["boundaries"], answer["boundaries"], tolerances.talk_seconds)
    if task == HIGHLIGHTS:
        return range_scores(value["ranges"], answer["ranges"], float(answer["duration"]))
    if task == GROUPS:
        raise ValueError("グループの課題は正解と比べない")
    return subtitle_scores(value["segments"], answer["segments"], tolerances.subtitle_seconds)


@dataclass(frozen=True, slots=True)
class Opened:
    """素材 1 本の下ごしらえ ``reason`` が空でなければ、その素材の課題はどれも数えない"""

    answer: dict[str, Any]
    facts: MediaFacts
    reason: str = ""


def open_material(folder: EvalDir, material: str, path: Path | None) -> Opened:
    """正解を読み、素材の長さと fps を測る 失敗しても止めず、固定の理由を返す

    ここで止めると、外付けのドライブを外した・素材の名前を変えた・正解のファイルが壊れた
    といった 1 本の事情で、同じ回のほかの素材の結果まで 1 行も残らない
    例外の文には置き場が混じりうるので、画面には型の名前だけを出す
    """
    empty = MediaFacts(0.0, 0.0)
    if path is None or not path.is_file():
        return Opened({"material": material, "tasks": {}}, empty, "no-material")
    try:
        answer = folder.read_answer(material)
    except Exception as error:
        print(f"{material}: 正解を読めなかった（{type(error).__name__}）")
        return Opened({"material": material, "tasks": {}}, empty, "answer-error")
    try:
        facts = media_facts(path)
    except Exception as error:
        print(f"{material}: 素材を読めなかった（{type(error).__name__}）")
        return Opened(answer, empty, "tool-error")
    return Opened(answer, facts)


#: 白紙の下書きを作ったときに出す、人がすること
_BLANK_GUIDES = {
    SCENES: "目印の無い .sme を作った（素材の全体を見て一から目印を置く）",
    HIGHLIGHTS: (
        "素材をそのまま置いた .sme を作った（見どころでない所を消して、見どころだけを残す）"
    ),
}


def _blank_sheet(
    folder: EvalDir, material: str, path: Path, tasks: dict[str, Any], task: str, *, force: bool
) -> bool:
    """人が一から付ける課題（場面・見どころ）は、素材をそのまま置いた .sme だけを作る 作ったら真

    #345 の決まりで、場面（T3）と見どころは人が素材の全体を見て一から付ける 道具の下書きを
    仮の正解にすると、人が直す前の回は道具と道具を比べて数が実際より高く出る
    前の版が置いた道具の仮の正解は、ここで外す 残すと run がそれと比べてしまう
    人が付けている途中の .sme は、``force`` でなければ上書きしない（付けた手間が消える）
    """
    if tasks.pop(task, None) is not None:
        print(f"{material} {task}: 道具の下書きの仮の正解を外した（人が一から付ける）")
    sheet = folder.draft_path(material, task)
    if sheet.exists() and not force:
        print(f"{material} {task}: 正解を付ける .sme があるので飛ばす（作り直すなら --force）")
        return False
    try:
        folder.drafts.mkdir(parents=True, exist_ok=True)
        save_project(draft_project(path, material, task, {}), sheet)
    except Exception as error:
        print(f"{material} {task}: .sme を作れなかった（{type(error).__name__}）")
        return False
    print(f"{material} {task}: {_BLANK_GUIDES[task]}")
    return True


def command_draft(
    folder: EvalDir,
    book: TaskBook,
    chosen: Sequence[str],
    *,
    force: bool = False,
    params: ToolParams | None = None,
    transcriber: Transcriber = transcribe_with_app,
) -> list[str]:
    """道具の下書きから仮の正解と直す用の .sme を作る 作った物の一覧（番号と課題）を返す

    人が直した正解（provisional が偽）は ``force`` でも書き換えない 直した手間が消える
    仮の正解は、前に作った物があれば ``force`` のときだけ作り直す
    """
    resolved = params or ToolParams()
    manifest = folder.manifest()
    made: list[str] = []
    for material in _materials(book, chosen):
        path = manifest.get(material.id)
        if path is None:
            print(f"{material.id}: 対応表に無いので飛ばす")
            continue
        opened = open_material(folder, material.id, path)
        if opened.reason:
            # 正解が読めないまま書くと、本人が直した正解を空の正解で上書きしてしまう
            print(f"{material.id}: 下書きを作らずに飛ばす（{opened.reason}）")
            continue
        answer, facts = opened.answer, opened.facts
        answer.update({"material": material.id, "duration": facts.duration, "fps": facts.fps})
        tasks: dict[str, Any] = answer.setdefault("tasks", {})
        silences: list[tuple[float, float]] | None = None
        for task in material.tasks:
            if task not in ANSWERED:
                continue
            existing = tasks.get(task)
            if existing is not None and not existing.get("provisional", True):
                print(f"{material.id} {task}: 人が直した正解があるので触らない")
                continue
            if task in BLANK_SHEETS:
                if _blank_sheet(folder, material.id, path, tasks, task, force=force):
                    made.append(f"{material.id} {task}")
                continue
            if existing is not None and not force:
                print(f"{material.id} {task}: 仮の正解があるので飛ばす（作り直すなら --force）")
                continue
            # 1 つの課題の道具が落ちても、同じ素材のほかの課題とあとの素材の下書きは作る
            # 止めると、それまでに作った仮の正解もこの素材の分は書かれずに消える
            try:
                if task == TALK:
                    if silences is None:
                        silences = find_silences(path, resolved)
                    value: dict[str, Any] | None = {"boundaries": talk_from_silences(silences)}
                else:
                    value = predict(task, path, resolved, transcriber, facts).value
                if value is None:
                    print(f"{material.id} {task}: 道具が使えないので下書きを作れない（字幕起こし）")
                    continue
                draft = {"provisional": True, "source": "draft", **value}
                folder.drafts.mkdir(parents=True, exist_ok=True)
                project = draft_project(path, material.id, task, with_facts(draft, facts))
                save_project(project, folder.draft_path(material.id, task))
            except Exception as error:
                print(f"{material.id} {task}: 下書きを作れなかった（{type(error).__name__}）")
                continue
            # 直す用の .sme を置けてから正解へ足す 先に足すと、直す物の無い仮の正解が残る
            tasks[task] = draft
            made.append(f"{material.id} {task}")
            print(f"{material.id} {task}: 仮の正解を作った")
        folder.write_answer(material.id, answer)
    return made


def command_adopt(folder: EvalDir, material: str, task: str, source: Path | None = None) -> None:
    """直した .sme を正解にする 仮の印を外す"""
    if task not in ANSWERED:
        raise SystemExit(f"正解を持つ課題ではない: {task}")
    manifest = folder.manifest()
    path = manifest.get(material)
    if path is None:
        raise SystemExit(f"対応表に無い番号: {material}")
    project = load_project(source or folder.draft_path(material, task))
    answer = folder.read_answer(material)
    # 下書きを通さずに取り込んだ正解にも長さと fps を残す 正解のファイルだけを見る人
    # （report を読む人や後で作る AI の段）が、素材の長さを 0 と取り違えない
    facts = media_facts(path)
    answer.update({"material": material, "duration": facts.duration, "fps": facts.fps})
    answer.setdefault("tasks", {})[task] = {
        "provisional": False,
        "source": "human",
        **answer_from_project(project, path, task),
    }
    folder.write_answer(material, answer)
    print(f"{material} {task}: 直した正解を取り込んだ")


class TranscribeError(RuntimeError):
    """字幕起こしが例外を投げた 同じ素材のあとの課題でも、やり直さずにこれを投げる"""


def _remembered(transcriber: Transcriber) -> Transcriber:
    """1 本の素材の中で、字幕起こしの結果を覚えて使い回す包み

    失敗も覚える 覚えないと、字幕の課題で落ちた字幕起こしを書き出しの課題でもう 1 度走らせ、
    長い素材では同じ失敗を待つだけで数分を失う
    """
    memo: dict[str, Transcript | None] = {}
    failed: list[TranscribeError] = []

    def once(media: Path, chosen: ToolParams) -> Transcript | None:
        if failed:
            raise failed[0]
        if "value" not in memo:
            try:
                memo["value"] = transcriber(media, chosen)
            except Exception as error:
                failed.append(TranscribeError(type(error).__name__))
                raise failed[0] from error
        return memo["value"]

    return once


def command_run(
    folder: EvalDir,
    book: TaskBook,
    chosen: Sequence[str],
    *,
    results: Path = RESULTS_FILE,
    params: ToolParams | None = None,
    transcriber: Transcriber = transcribe_with_app,
    exporter: Exporter | None = None,
    commit: str | None = None,
) -> list[dict[str, object]]:
    """道具の段を回して正解と比べ、結果を足す 足した行を返す"""
    resolved = params or ToolParams()
    export = exporter if exporter is not None else export_and_check
    # 途中の行が壊れた結果のファイルには足せない 道具を数分走らせてから止まらないように先に見る
    if results.exists():
        _parse_results(_raw_text(results))
    manifest = folder.manifest()
    names = forbidden_names(folder, manifest)
    check_params(resolved, names)
    stamp = {
        "date": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "version": __version__,
        "commit": commit if commit is not None else current_commit(),
        "stage": "tools",
    }
    records: list[dict[str, object]] = []
    for material in _materials(book, chosen):
        path = manifest.get(material.id)
        opened = open_material(folder, material.id, path)
        answer, facts = opened.answer, opened.facts
        # 字幕と書き出しの両方を持つ素材で、字幕起こしを 2 回走らせない（長い素材で数分かかる）
        once = _remembered(transcriber)
        for task in material.tasks:
            record: dict[str, object] = {**stamp, "material": material.id, "task": task}
            truth = answer.get("tasks", {}).get(task)
            if task in BLANK_SHEETS and truth is not None and truth.get("provisional", True):
                # 前の版の draft が残した道具の仮の正解とは比べない 道具と道具を比べた F1 が
                # 本当の精度に見える 場面と見どころは人が一から付けた正解ができるまで数えない
                truth = None
            if path is None or opened.reason:
                # 素材が無いのは数える前の事情なので飛ばした扱い 読めなかったのは失敗
                status = "skipped" if opened.reason == "no-material" else "failed"
                record.update(status=status, reason=opened.reason)
                records.append(record)
                continue
            # 1 つの課題の失敗（GPU の無い機械の書き出し・検査の時間切れ・字幕起こしの部品の
            # 不具合）で止めると、数分かけて数えた同じ回のほかの結果が 1 行も残らない
            # 例外の文には素材の置き場が混じりうるので、結果には固定の理由だけを書き、
            # 画面にも例外の型の名前だけを出す
            try:
                if task == EXPORT:
                    began = time.perf_counter()
                    transcript = once(path, resolved)
                    folder.exports.mkdir(parents=True, exist_ok=True)
                    metrics = export(path, transcript, folder.exports / f"{material.id}.mp4")
                    record.update(
                        status="ok",
                        seconds=round(time.perf_counter() - began, 2),
                        metrics=finite_metrics(metrics),
                        params={"subtitles": transcript is not None},
                    )
                elif task == GROUPS:
                    # 正解は持たない 無音カットの後にグループが残ったかを数える
                    began = time.perf_counter()
                    kept = cut_groups_with_tools(path, resolved)
                    record.update(
                        status="ok",
                        seconds=round(time.perf_counter() - began, 2),
                        metrics=finite_metrics(kept),
                        params=resolved.record(SILENCE),
                    )
                elif truth is None:
                    record.update(status="skipped", reason="no-answer")
                else:
                    prediction = predict(task, path, resolved, once, facts)
                    if prediction.value is None:
                        record.update(status="skipped", reason=prediction.reason)
                    else:
                        record.update(
                            status="ok",
                            provisional=bool(truth.get("provisional", True)),
                            seconds=prediction.seconds,
                            metrics=finite_metrics(
                                score(task, prediction.value, with_facts(truth, facts), book)
                            ),
                            params=resolved.record(task),
                        )
            except TranscribeError as error:
                # 字幕起こしの失敗は、それを使う課題のどれでも同じ理由にする 書き出しの失敗と
                # 取り違えると、GPU を疑って字幕起こしの部品の不具合を見落とす
                print(f"{material.id} {task}: 字幕起こしに失敗した（{error}）")
                record.update(status="failed", reason="asr-error")
            except Exception as error:
                print(f"{material.id} {task}: 失敗した（{type(error).__name__}）")
                record.update(
                    status="failed", reason="export-error" if task == EXPORT else "tool-error"
                )
            records.append(record)
    append_results(results, records, names)
    return records


#: AI の段の同意が無いときの断り 送る物を先に言う（#342 の同意の表と同じ物）
NO_CONSENT = (
    "AI の段は、素材の名前と場所・字幕の文・映像の絵を Claude Code を通して Anthropic へ送ります"
    " 送ってよいと確かめたら --allow-send を付けて回してください"
)


def command_ai_run(
    folder: EvalDir,
    book: TaskBook,
    chosen: Sequence[str],
    *,
    allow_send: bool,
    results: Path = AI_RESULTS_FILE,
    session_factory: SessionFactory | None = None,
    model: str | None = None,
    timeout: float = AI_TIMEOUT_SECONDS,
    commit: str | None = None,
    runner: Callable[..., tuple[AiRun, dict[str, float | int]]] | None = None,
) -> list[dict[str, object]]:
    """AI の段を回して結果を足す 今あるのはグループの課題だけ（#437）

    同意が無ければ 1 行も書かずに止まる 1 つの素材で会話が失敗しても、ほかの素材は回す
    """
    if not allow_send:
        raise SystemExit(NO_CONSENT)
    if model is not None and not _SAFE_TEXT.match(model):
        raise ParamsError("--model には英数字と . _ : + - だけの名前を渡す")
    factory = session_factory if session_factory is not None else real_session(model)
    drive = runner if runner is not None else run_ai_groups
    if results.exists():
        _parse_results(_raw_text(results))
    manifest = folder.manifest()
    names = forbidden_names(folder, manifest)
    stamp = {
        "date": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "version": __version__,
        "commit": commit if commit is not None else current_commit(),
        "stage": "ai",
    }
    records: list[dict[str, object]] = []
    for material in _materials(book, chosen):
        for task in material.tasks:
            if task != GROUPS:
                # AI の段で数えるのはグループの課題だけ ほかは道具の段で数える
                continue
            record: dict[str, object] = {**stamp, "material": material.id, "task": task}
            path = manifest.get(material.id)
            if path is None or not path.is_file():
                record.update(status="skipped", reason="no-material")
                records.append(record)
                continue
            began = time.perf_counter()
            try:
                sheet = folder.drafts / f"{material.id}_{GROUPS}_ai.sme"
                run, metrics = drive(path, sheet, material.id, factory, timeout=timeout)
            except Exception as error:
                # 例外の文には素材の置き場が混じりうる 型の名前だけを出す
                print(f"{material.id} {task}: AI の段が失敗した（{type(error).__name__}）")
                record.update(status="failed", reason="ai-error")
                records.append(record)
                continue
            if run.failed or not run.finished:
                record.update(status="failed", reason="ai-error" if run.failed else "ai-timeout")
            elif float(metrics.get("removed_seconds", 0.0)) <= 0.0:
                # 切らずに終えた グループが残ったのは切っていないから 成功と並べない
                record.update(
                    status="failed",
                    reason="ai-no-cut",
                    metrics=finite_metrics({**metrics, **run.metrics()}),
                )
            else:
                record.update(
                    status="ok",
                    seconds=round(time.perf_counter() - began, 2),
                    metrics=finite_metrics({**metrics, **run.metrics()}),
                    params={"model": model or "default"},
                )
            records.append(record)
    append_results(results, records, names)
    return records


#: 表に出す主な数 課題ごとに 1 つ
HEADLINE = {
    SILENCE: "iou",
    SCENES: "f1",
    TALK: "f1",
    SUBTITLES: "cer",
    EXPORT: "problems",
    HIGHLIGHTS: "f1",
    GROUPS: "kept_share",
}


def report_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """課題ごとに、最新の回と 1 つ前の回の主な数と秒を並べる"""
    latest: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["stage"]), str(row["material"]), str(row["task"]))
        if row["task"] in BLANK_SHEETS:
            # 場面と見どころは人が一から付けた正解だけで数える（#345） 前の版が道具の下書きを
            # 仮の正解にして数えた行は記録に残すが、道具と道具を比べた数なので表の数には使わない
            # 使うと、人の正解で数えるまで 1.0 に近い数が今の精度に見え続ける
            latest.setdefault(key, [])
            if row.get("provisional") is not False:
                continue
        if row.get("status") != "ok":
            continue
        latest.setdefault(key, []).append(row)
    lines = ["段     番号 課題       数      前回    差      秒     正解"]
    for (stage, material, task), history in sorted(latest.items()):
        if not history:
            lines.append(f"{stage:6} {material:4} {task:10} まだ人の正解が無い")
            continue
        name = HEADLINE[task]
        now = history[-1]
        value = float(now["metrics"][name])
        previous = history[-2] if len(history) > 1 else None
        before = None if previous is None else float(previous["metrics"][name])
        change = "" if before is None else f"{value - before:+.4f}"
        # 書き出しの課題は正解を持たないので、仮も人も付けない
        mark = "-" if "provisional" not in now else ("仮" if now["provisional"] else "人")
        # 道具の値を変えて回した差と、コードを変えた差を取り違えないように印を付ける
        # 正解を直した回（仮から人へ）も、数の動きが道具の変化ではないので印を付ける
        notes = []
        if previous is not None and previous.get("params") != now.get("params"):
            notes.append("値が違う")
        if previous is not None and previous.get("provisional") != now.get("provisional"):
            notes.append("正解が変わった")
        lines.append(
            f"{stage:6} {material:4} {task:10} {name}={value:<6.4f} "
            f"{'' if before is None else f'{before:.4f}':7} {change:7} "
            f"{float(now.get('seconds', 0)):6.1f} {mark} {' '.join(notes)}".rstrip()
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    folder_help = f"評価のフォルダ（省くと {EVAL_DIR_VARIABLE}）"
    parser.add_argument("--dir", help=folder_help)
    sub = parser.add_subparsers(dest="command", required=True)

    def command_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        """命令ごとの parser 命令の後ろに置いた --dir も受ける

        一番上にだけ足すと、手順書どおり命令の後ろに置いたとき unrecognized arguments で止まる
        既定を SUPPRESS にして、命令の前に置いた値を空で上書きしない（前でも後ろでも動く
        両方に置いたら後ろが勝つ）
        """
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--dir", default=argparse.SUPPRESS, help=folder_help)
        return command

    def tool_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("ids", nargs="*", help="素材の番号（省くと全部）")
        command.add_argument("--silence-threshold-db", type=finite_float, default=-40.0)
        command.add_argument("--min-silence", type=finite_float, default=0.5)
        command.add_argument("--padding", type=finite_float, default=0.1)
        command.add_argument("--scene-threshold", type=finite_float)
        command.add_argument("--min-scene", type=finite_float)
        command.add_argument("--asr-model")

    draft = command_parser("draft", "仮の正解と直す用の .sme を作る")
    tool_options(draft)
    draft.add_argument("--force", action="store_true", help="仮の正解を作り直す")
    adopt = command_parser("adopt", "直した .sme を正解にする")
    adopt.add_argument("id")
    adopt.add_argument("task", choices=ANSWERED)
    adopt.add_argument("--from", dest="source", type=Path, help="直した .sme（省くと下書きの所）")
    run = command_parser("run", "道具の段を回して結果を足す")
    tool_options(run)
    ai_run = command_parser("ai-run", "AI の段（グループを保つ）を回して結果を足す")
    ai_run.add_argument("ids", nargs="*", help="素材の番号（省くと全部）")
    ai_run.add_argument(
        "--allow-send",
        action="store_true",
        help="素材の名前・字幕・絵を Claude（Anthropic）へ送ってよいと確かめた",
    )
    ai_run.add_argument("--model", help="AI のモデル（省くと Claude Code の既定）")
    ai_run.add_argument(
        "--timeout", type=finite_float, default=AI_TIMEOUT_SECONDS, help="1 つの指示を待つ秒"
    )
    # report はリポジトリの中の結果だけを読むので外のフォルダは使わないが、ほかの命令と同じ
    # 打ち方（--dir 付き）で止まらないように受ける
    command_parser("report", "前回との差を出す")
    arguments = parser.parse_args(argv)
    try:
        return _dispatch(arguments)
    except BrokenResultsError as error:
        # 行の番号だけを出す 例外の流れ（Traceback）を見せても直す手がかりは増えない
        print(error)
        return 1
    except ParamsError as error:
        # 引数の読み違いと同じ扱い（argparse も 2 で返す） 何がいけないかだけを出す
        print(error, file=sys.stderr)
        return 2


def _all_results() -> list[dict[str, Any]]:
    """道具の段と AI の段の結果をまとめて読む 無い方は空"""
    rows: list[dict[str, Any]] = []
    for path in (RESULTS_FILE, AI_RESULTS_FILE):
        if path.exists():
            rows.extend(read_results(path))
    return rows


def _dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command == "report":
        print("\n".join(report_lines(_all_results())))
        return 0
    folder = eval_dir(arguments.dir)
    if arguments.command == "ai-run":
        records = command_ai_run(
            folder,
            load_tasks(),
            arguments.ids,
            allow_send=arguments.allow_send,
            model=arguments.model,
            timeout=arguments.timeout,
        )
        for record in records:
            if record["status"] != "ok":
                print(f"{record['material']} {record['task']}: 数えなかった（{record['reason']}）")
        print("\n".join(report_lines(_all_results())))
        return 0
    if arguments.command == "adopt":
        command_adopt(folder, arguments.id, arguments.task, arguments.source)
        return 0
    params = ToolParams(
        silence_threshold_db=arguments.silence_threshold_db,
        min_silence=arguments.min_silence,
        padding=arguments.padding,
        scene_threshold=arguments.scene_threshold,
        min_scene=arguments.min_scene,
        asr_model=arguments.asr_model,
    )
    book = load_tasks()
    if arguments.command == "draft":
        command_draft(folder, book, arguments.ids, force=arguments.force, params=params)
        return 0
    records = command_run(folder, book, arguments.ids, params=params)
    for record in records:
        if record["status"] != "ok":
            print(f"{record['material']} {record['task']}: 数えなかった（{record['reason']}）")
    print("\n".join(report_lines(_all_results())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
