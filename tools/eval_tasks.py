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
from typing import Any

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sashimono import __version__  # noqa: E402
from sashimono.core.commands import AddMarkers, Document, RippleCut, SetTranscript  # noqa: E402
from sashimono.core.commands.insert import insert_media  # noqa: E402
from sashimono.core.io import load_project, save_project  # noqa: E402
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
TASKS = (SILENCE, TALK, SCENES, SUBTITLES, EXPORT)
#: 正解を持つ課題 下書きと取り込みはこれだけ
ANSWERED = (SILENCE, TALK, SCENES, SUBTITLES)

#: 課題の定義と、結果を足していく所（どちらもリポジトリの中）
TASKS_FILE = ROOT / "eval" / "tasks.json"
RESULTS_FILE = ROOT / "eval" / "results" / "tools.jsonl"

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
        self.answers.mkdir(parents=True, exist_ok=True)
        text = json.dumps(answer, ensure_ascii=False, indent=1)
        self.answer_path(material).write_text(text + "\n", encoding="utf-8")


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


@dataclass(frozen=True, slots=True)
class ToolParams:
    """道具の既定値 変える前と後を比べるために、引数で差し替えられるようにしておく"""

    silence_threshold_db: float = -40.0
    min_silence: float = 0.5
    padding: float = 0.1
    scene_threshold: float | None = None
    min_scene: float | None = None
    asr_model: str | None = None

    def record(self, task: str) -> dict[str, float | str]:
        """結果の行に残す値 課題に関係する物だけ"""
        if task == SILENCE:
            return {
                "threshold_db": self.silence_threshold_db,
                "min_silence": self.min_silence,
                "padding": self.padding,
            }
        if task == SCENES:
            from sashimono.engine.scenes import DEFAULT_MIN_SCENE, DEFAULT_THRESHOLD

            return {
                "threshold": self.scene_threshold or DEFAULT_THRESHOLD,
                "min_scene": self.min_scene or DEFAULT_MIN_SCENE,
            }
        if task == SUBTITLES:
            from sashimono.asr.backend import DEFAULT_MODEL

            return {"model": self.asr_model or DEFAULT_MODEL}
        return {}


@dataclass(frozen=True, slots=True)
class MediaFacts:
    """正解に一緒に残す素材の長さと fps 場面の許す幅を秒に直すのに使う"""

    duration: float
    fps: float


def media_facts(path: Path) -> MediaFacts:
    from sashimono.engine.decode.probe import probe_media

    media = probe_media(path)
    fps = float(media.video_streams[0].frame_rate.fps) if media.video_streams else 0.0
    return MediaFacts(float(media.duration), fps)


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
    from sashimono.engine.scenes import DEFAULT_MIN_SCENE, DEFAULT_THRESHOLD, analyze_scenes

    # 0 は機械のコアをすべて使う 評価は手元で 1 本ずつ回すので、ほかの仕事と取り合わない
    signal = analyze_scenes(path, threads=0)
    if signal is None:
        return []
    cuts = signal.cuts(
        params.scene_threshold or DEFAULT_THRESHOLD, params.min_scene or DEFAULT_MIN_SCENE
    )
    return [cut.seconds for cut in cuts]


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
    options = TranscribeOptions(model=params.asr_model) if params.asr_model else TranscribeOptions()
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
) -> dict[str, float]:
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
    offsets = [min(abs(edge - other) for other in mine_edges) for edge in answer_edges]
    return {
        "iou": round(both / either, 4) if either else 1.0,
        "agreement": round(same / cells, 4),
        "edge_median": round(statistics.median(offsets), 3) if offsets and mine_edges else 0.0,
        "kept_seconds": round(sum(mine) * GRID_SECONDS, 2),
        "answer_kept_seconds": round(sum(truth) * GRID_SECONDS, 2),
    }


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
_IGNORED_CHARACTERS = re.compile(
    r"[\s、\N{IDEOGRAPHIC FULL STOP}，．,.!?！？「」『』（）()…・ー〜~]"
)


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


def _check_value(value: object, where: str, names: Sequence[str]) -> None:
    if value is None or isinstance(value, bool | int | float):
        return
    if isinstance(value, str):
        if not _SAFE_TEXT.match(value):
            raise LeakError(f"{where} に書けない文字がある（英数字と . _ : + - だけ）")
        lowered = value.lower()
        for name in names:
            if name.lower() in lowered:
                raise LeakError(f"{where} に素材の名前か置き場が混じっている")
        return
    raise LeakError(f"{where} の値の形が書けない物（{type(value).__name__}）")


def guard_record(record: Mapping[str, object], names: Sequence[str]) -> None:
    """結果の 1 行を、リポジトリへ書いてよいか確かめる だめなら :class:`LeakError`

    形の決まり（知っている鍵・英数字だけの文字）と、素材の名前の照らしの 2 段にする
    形だけだと英数字の素材の名前（録画の日時の名前など）が通り、照らすだけだと
    知らない名前（対応表に無い物）や字幕の文が通る
    """
    unknown = set(record) - _RECORD_KEYS
    if unknown:
        raise LeakError(f"知らない鍵がある: {sorted(unknown)}")
    for key, value in record.items():
        if isinstance(value, Mapping):
            for inner, item in value.items():
                _check_value(str(inner), f"{key} の鍵", names)
                _check_value(item, f"{key}.{inner}", names)
        else:
            _check_value(value, key, names)


def append_results(
    path: Path, records: Sequence[Mapping[str, object]], names: Sequence[str]
) -> None:
    """結果を足す 全部を確かめてから書く 1 行でもだめなら 1 行も書かない

    確かめながら書くと、だめな行の手前までが書かれ、同じ回の数が半端に残る
    """
    for record in records:
        guard_record(record, names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def current_commit() -> str:
    """今のコミット 書き換えたまま回したときは印を付ける（どの形で測ったかが曖昧になるため）"""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
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
    - 場面と話の区切り: 素材をそのまま置き、区切りに目印を置く 人は目印を足す・消す・動かす
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
    elif task in (SCENES, TALK):
        points = [_span(item)[0] for item in answer["cuts" if task == SCENES else "boundaries"]]
        frames = sorted({round(moment * float(rate.fps)) for moment in points})
        label = "場面" if task == SCENES else "区切り"
        markers = tuple(Marker(frame=frame, label=label) for frame in frames if frame > 0)
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
    if task == SILENCE:
        ranges = sorted((float(clip.source_in), float(clip.source_out(rate))) for clip in clips)
        merged: list[list[float]] = []
        for start, end in ranges:
            # 絵と音の組は同じ区間を 2 回持つ 重ねてまとめる
            if merged and start <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return {"kept": [[round(a, 3), round(b, 3)] for a, b in merged]}
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
    else:
        # 話の区切りの道具は無い AI の段（後で作る）で数える
        return Prediction(None, 0.0, "no-tool")
    return Prediction(value, round(time.perf_counter() - began, 2))


def with_facts(truth: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """課題 1 つの正解に、素材の長さと fps を添える 比べるときと .sme を作るときに要る"""
    return {**truth, "duration": answer.get("duration", 0.0), "fps": answer.get("fps", 0.0)}


def score(
    task: str, value: Mapping[str, Any], answer: Mapping[str, Any], book: TaskBook
) -> dict[str, float]:
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
    return subtitle_scores(value["segments"], answer["segments"], tolerances.subtitle_seconds)


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
        answer = folder.read_answer(material.id)
        facts = media_facts(path)
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
            if existing is not None and not force:
                print(f"{material.id} {task}: 仮の正解があるので飛ばす（作り直すなら --force）")
                continue
            if task == TALK:
                silences = silences if silences is not None else find_silences(path, resolved)
                value: dict[str, Any] | None = {"boundaries": talk_from_silences(silences)}
            else:
                value = predict(task, path, resolved, transcriber, facts).value
            if value is None:
                print(f"{material.id} {task}: 道具が使えないので下書きを作れない（字幕起こし）")
                continue
            tasks[task] = {"provisional": True, "source": "draft", **value}
            folder.drafts.mkdir(parents=True, exist_ok=True)
            project = draft_project(path, material.id, task, with_facts(tasks[task], answer))
            save_project(project, folder.draft_path(material.id, task))
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
    answer.setdefault("tasks", {})[task] = {
        "provisional": False,
        "source": "human",
        **answer_from_project(project, path, task),
    }
    folder.write_answer(material, answer)
    print(f"{material} {task}: 直した正解を取り込んだ")


def _remembered(transcriber: Transcriber) -> Transcriber:
    """1 本の素材の中で、字幕起こしの結果を覚えて使い回す包み"""
    memo: dict[str, Transcript | None] = {}

    def once(media: Path, chosen: ToolParams) -> Transcript | None:
        if "value" not in memo:
            memo["value"] = transcriber(media, chosen)
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
    manifest = folder.manifest()
    names = forbidden_names(folder, manifest)
    stamp = {
        "date": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "version": __version__,
        "commit": commit if commit is not None else current_commit(),
        "stage": "tools",
    }
    records: list[dict[str, object]] = []
    for material in _materials(book, chosen):
        path = manifest.get(material.id)
        answer = folder.read_answer(material.id)
        facts = media_facts(path) if path is not None else MediaFacts(0.0, 0.0)
        # 字幕と書き出しの両方を持つ素材で、字幕起こしを 2 回走らせない（長い素材で数分かかる）
        once = _remembered(transcriber)
        for task in material.tasks:
            record: dict[str, object] = {**stamp, "material": material.id, "task": task}
            truth = answer.get("tasks", {}).get(task)
            if path is None:
                record.update(status="skipped", reason="no-material")
            elif task == EXPORT:
                began = time.perf_counter()
                transcript = once(path, resolved)
                folder.exports.mkdir(parents=True, exist_ok=True)
                metrics = export(path, transcript, folder.exports / f"{material.id}.mp4")
                record.update(
                    status="ok",
                    seconds=round(time.perf_counter() - began, 2),
                    metrics=metrics,
                    params={"subtitles": transcript is not None},
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
                        metrics=score(task, prediction.value, with_facts(truth, answer), book),
                        params=resolved.record(task),
                    )
            records.append(record)
    append_results(results, records, names)
    return records


#: 表に出す主な数 課題ごとに 1 つ
HEADLINE = {SILENCE: "iou", SCENES: "f1", TALK: "f1", SUBTITLES: "cer", EXPORT: "problems"}


def report_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """課題ごとに、最新の回と 1 つ前の回の主な数と秒を並べる"""
    latest: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["stage"]), str(row["material"]), str(row["task"]))
        latest.setdefault(key, []).append(row)
    lines = ["段     番号 課題       数      前回    差      秒     正解"]
    for (stage, material, task), history in sorted(latest.items()):
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
    parser.add_argument("--dir", help=f"評価のフォルダ（省くと {EVAL_DIR_VARIABLE}）")
    sub = parser.add_subparsers(dest="command", required=True)

    def tool_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("ids", nargs="*", help="素材の番号（省くと全部）")
        command.add_argument("--silence-threshold-db", type=float, default=-40.0)
        command.add_argument("--min-silence", type=float, default=0.5)
        command.add_argument("--padding", type=float, default=0.1)
        command.add_argument("--scene-threshold", type=float)
        command.add_argument("--min-scene", type=float)
        command.add_argument("--asr-model")

    draft = sub.add_parser("draft", help="仮の正解と直す用の .sme を作る")
    tool_options(draft)
    draft.add_argument("--force", action="store_true", help="仮の正解を作り直す")
    adopt = sub.add_parser("adopt", help="直した .sme を正解にする")
    adopt.add_argument("id")
    adopt.add_argument("task", choices=ANSWERED)
    adopt.add_argument("--from", dest="source", type=Path, help="直した .sme（省くと下書きの所）")
    run = sub.add_parser("run", help="道具の段を回して結果を足す")
    tool_options(run)
    sub.add_parser("report", help="前回との差を出す")
    arguments = parser.parse_args(argv)

    if arguments.command == "report":
        print("\n".join(report_lines(read_results(RESULTS_FILE))))
        return 0
    folder = eval_dir(arguments.dir)
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
    print("\n".join(report_lines(read_results(RESULTS_FILE))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
