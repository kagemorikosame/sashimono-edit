"""画面のスレッドの止まりを見張って記録する（#442 の 0 番 #443 の段階 1）

画面のスレッドが決まった間隔で :meth:`StallWatch.beat` を打ち、見張りのスレッドが
打たれていない長さを見る 止まりが :attr:`StallWatch.threshold_ms` を超えたら、その場で
全スレッドの Python の呼び出しの並び（:func:`sys._current_frames`）を写し、動き出した
ところで、止まりの長さと、その間に走ったごみ集め（:data:`gc.callbacks`）の時間を添えて
1 行ずつ書く

止まった**最中に**写すのが要点 画面のスレッドのタイマーだけで測ると、遅れに気付くのは
動き出した後で、そのときにはもう止めていた物が呼び出しの並びに残っていない

Qt に依存しない 画面の側（:mod:`sashimono.ui.stall_monitor`）がタイマーで打つ
普段は動かさない 見張りのスレッドが 1 秒に何十回も起きるので、診断を入れた人だけが払う
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any

from sashimono.core.userdirs import state_root

__all__ = [
    "DEFAULT_THRESHOLD_MS",
    "ENV_FLAG",
    "LOG_NAME",
    "StallWatch",
    "default_log_path",
    "threshold_from_env",
]

#: これより長く画面のスレッドが止まったら記録する（ミリ秒）
#: 60fps の 2 コマ 1 コマを少し超えるだけの揺れまで書くと、記録が埋もれて読めない
DEFAULT_THRESHOLD_MS = 30.0

#: 設定画面を開かずに入れる環境変数 ``1`` で既定の線 数を書けばその線（ミリ秒）
#: 起動の直後（設定を読む前・窓を出す前）の止まりも測れるようにする
ENV_FLAG = "SASHIMONO_STALL_WATCH"

#: 記録のファイルの名前 1 行が 1 回の止まり（JSON）
LOG_NAME = "stalls.jsonl"

#: 記録がこの大きさを超えたら 1 世代前へ回す 診断を入れたまま忘れても、置き場を食い続けない
MAX_LOG_BYTES = 4 * 1024 * 1024

#: 1 回の止まりで何回まで呼び出しの並びを写すか 長く止まったときに途中で何が変わったかを
#: 見たいが、写すたびに見張りのスレッドが GIL を取るので、写しすぎると止まりを伸ばす
MAX_SAMPLES = 3

#: 1 スレッドあたり何段まで写すか 深い所（今動いている所）から数える
STACK_DEPTH = 25

#: 覚えておくごみ集めの回数 止まりの間に走った物を探すのに使う
GC_HISTORY = 64


def default_log_path() -> Path:
    """記録の置き場 退避と同じ側（大きくなりうるので移動プロファイルに載せない）"""
    return state_root() / "diagnostics" / LOG_NAME


def threshold_from_env(environ: Mapping[str, str] | None = None) -> float | None:
    """環境変数で入れたときの線（ミリ秒） 入っていなければ ``None``

    ``0`` と空と数でない値は入っていない扱い 打ち間違いで毎コマ記録されるより、
    動かない方が困らない ``inf`` のような限りの無い値も同じ 通すと見張りのスレッドが
    待つ長さを求める所で落ち、入っているように見えて何も記録しない
    """
    value = (os.environ if environ is None else environ).get(ENV_FLAG, "").strip()
    if not value or value == "0":
        return None
    if value == "1":
        return DEFAULT_THRESHOLD_MS
    try:
        threshold = float(value)
    except ValueError:
        return None
    return threshold if math.isfinite(threshold) and threshold > 0 else None


@dataclass(slots=True)
class _Stall:
    """止まっている最中の 1 回 動き出したら書く"""

    began: float
    samples: list[dict[str, Any]] = field(default_factory=list)


class StallWatch:
    """画面のスレッドの止まりを見張る

    ``clock`` と ``frames`` は試験で差し替える 本物の時計では止まりを作るのに本当に
    待つことになり、試験が遅く揺れる
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        threshold_ms: float = DEFAULT_THRESHOLD_MS,
        clock: Callable[[], float] = time.perf_counter,
        frames: Callable[[], Mapping[int, FrameType]] = sys._current_frames,
    ) -> None:
        self.path = path if path is not None else default_log_path()
        self.threshold_ms = threshold_ms
        self._clock = clock
        self._frames = frames
        self._lock = threading.Lock()
        self._last_beat: float | None = None
        #: 見張りの側が止まりに気付いて写している最中の 1 回
        self._stall: _Stall | None = None
        #: ごみ集めの記録（終わった時刻・世代・長さ ミリ秒）
        self._gc: deque[tuple[float, int, float]] = deque(maxlen=GC_HISTORY)
        self._gc_began: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        #: 動き出した止まり（止まり、動き出した時刻） 画面のスレッドが積み、見張りが書く
        #: deque の両端の出し入れは 1 つずつ不可分に行われるので、錠を取らずに渡せる
        self._pending: deque[tuple[_Stall, float]] = deque()
        #: 書いた回数 試験と、診断の欄に何件あるかを出すのに使う
        self.recorded = 0

    # --- 画面のスレッドから ---

    def beat(self) -> None:
        """画面のスレッドが動いている印 タイマーで決まった間隔ごとに打つ

        ここでは書かない 書くのは見張りのスレッド 画面のスレッドでファイルを書くと、
        記録そのものが次の止まりになる
        """
        now = self._clock()
        finished: tuple[_Stall, float] | None = None
        with self._lock:
            last = self._last_beat
            self._last_beat = now
            stall = self._stall
            if stall is not None:
                self._stall = None
                finished = (stall, now)
            elif last is not None and (now - last) * 1000 > self.threshold_ms:
                # 見張りが写す前に動き出した（線を少し超えただけ） 長さだけでも残す
                finished = (_Stall(began=last), now)
        if finished is not None:
            self._pending.append(finished)

    # --- 見張りのスレッドから ---

    def check(self) -> None:
        """止まっていれば呼び出しの並びを写す 見張りのスレッドが決まった間隔で呼ぶ

        試験からも呼ぶ（スレッドを立てずに、時計を進めてから呼ぶ）
        """
        now = self._clock()
        with self._lock:
            last = self._last_beat
            if last is None:
                return
            waited = (now - last) * 1000
            if waited <= self.threshold_ms:
                return
            stall = self._stall
            if stall is None:
                stall = self._stall = _Stall(began=last)
            # 線を超えるたびに 1 回写す（線の 1 倍・2 倍・3 倍の所） 毎回写すと、
            # 見張りのスレッドが GIL を取り続けて止まりを伸ばす
            if len(stall.samples) >= MAX_SAMPLES or waited < self.threshold_ms * (
                len(stall.samples) + 1
            ):
                return
        sample = {"at_ms": round(waited, 1), "threads": self._stacks()}
        with self._lock:
            if self._stall is stall:
                stall.samples.append(sample)

    def flush(self) -> None:
        """動き出した止まりを書く 見張りのスレッドが呼ぶ"""
        while self._pending:
            stall, ended = self._pending.popleft()
            self._write(stall, ended)

    def _stacks(self) -> dict[str, list[str]]:
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        own = threading.get_ident()
        found: dict[str, list[str]] = {}
        for ident, frame in self._frames().items():
            if ident == own:
                # 見張り自身の並びは毎回同じで、読む人の邪魔になる
                continue
            lines = [
                f"{Path(entry.filename).name}:{entry.lineno} {entry.name}"
                for entry in traceback.extract_stack(frame, limit=STACK_DEPTH)
            ]
            # 名前で引く 同じ名前のスレッド（取り消して頼み直した読み込みの係など）が
            # 並ぶと片方が消えるので、2 本目からは番号を添える ``MainThread`` は 1 本だけなので
            # 名前のまま残り、手引きの読み方（MainThread の並びを見る）は変わらない
            name = names.get(ident, str(ident))
            found[name if name not in found else f"{name} #{ident}"] = lines
        return found

    def _write(self, stall: _Stall, ended: float) -> None:
        history = self._gc_history()
        pauses = [
            {"generation": generation, "ms": round(length, 1)}
            for finished, generation, length in history
            if stall.began <= finished <= ended
        ]
        record = {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "stall_ms": round((ended - stall.began) * 1000, 1),
            "threshold_ms": self.threshold_ms,
            "gc": pauses,
            "samples": stall.samples,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > MAX_LOG_BYTES:
                self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # 書けない置き場（容量が尽きた・読み取り専用）でも編集は止めない 診断のために
            # 本人の作業を落とすのは本末転倒
            return
        self.recorded += 1

    # --- ごみ集め ---

    def _gc_history(self) -> list[tuple[float, int, float]]:
        """ごみ集めの記録を写す 写している最中に :meth:`_on_gc` が足すと写しが失敗する

        ごみ集めはこのスレッドの写しの中でも走りうるので、錠では防げない（錠を待つ
        知らせが同じスレッドで錠を取りに行き、止まる） 何度か写し直し、だめなら空にする
        ごみ集めの欄が空になるだけで、見張りのスレッドは落ちない
        """
        for _ in range(5):
            try:
                return list(self._gc)
            except RuntimeError:
                continue
        return []

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        now = self._clock()
        if phase == "start":
            self._gc_began = now
            return
        began, self._gc_began = self._gc_began, None
        if began is not None:
            self._gc.append((now, int(info.get("generation", -1)), (now - began) * 1000))

    # --- 動かす・止める ---

    def start(self) -> None:
        """見張りのスレッドを立て、ごみ集めの知らせを受け始める"""
        if self._thread is not None:
            return
        self._stop.clear()
        gc.callbacks.append(self._on_gc)
        # 線の 1/3 ごとに起きる 線ちょうどで起きると、線を少し超えた止まりを写し損ねる
        interval = max(0.002, self.threshold_ms / 3000)

        def run() -> None:
            while not self._stop.wait(interval):
                self.check()
                self.flush()
            self.flush()

        # daemon にする 閉じるときに止め忘れても、見張りのせいでプロセスが残らない
        self._thread = threading.Thread(target=run, name="sashimono-stall-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread, self._thread = self._thread, None
        if thread is None:
            return
        # 止まっている最中に止められた分（閉じる処理そのものが重かった等）も書く
        # 書かないと、写したはずの並びが次の打ちを待たずに捨てられる 止める合図より前に
        # 待ちの列へ入れ、見張りのスレッドが最後に書く ここで書くと、書き込みが 1 秒を
        # 超えて見張りが残ったとき、2 つのスレッドが同じ列から取り合って記録の順が入れ替わる
        with self._lock:
            stall, self._stall = self._stall, None
        if stall is not None:
            self._pending.append((stall, self._clock()))
        self._stop.set()
        thread.join(timeout=1.0)
        with contextlib.suppress(ValueError):
            gc.callbacks.remove(self._on_gc)

    @property
    def running(self) -> bool:
        return self._thread is not None
