"""画面の中身（文字と顔の位置 #324）を裏で調べて覚える

場面の切り替わり（#319 #321）を先に待ち、場面ごとに数枚の絵を取り出して調べる 絵を取り出す
（シークしてデコードする）のも、文字を読むのも 1 枚ずつ時間がかかるので、波形やサムネイルとは
別の 1 本のスレッドで順に走らせる 同じ置き場だと、長い素材を調べる間、後から読み込んだ素材の
サムネイルが出ない

部品が入っていない機械（CI を含む）では、調べ始めずに理由を失敗として残す
（:class:`~sashimono.engine.vision.parts.PartsMissingError`） AI の道具はその文をそのまま返す
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sashimono.core.model import MediaId, MediaItem
from sashimono.engine.scenes import SceneSignal
from sashimono.engine.vision import (
    Detector,
    PartsMissingError,
    ScreenKind,
    ScreenReport,
    ScreenScan,
    fallback_cuts,
    make_detector,
    sample_times,
    scan_screen,
)

from .analyzer import MediaAnalyzer
from .screen_cache import load_screen, save_screen, screen_key
from .store import CacheStore, media_key

__all__ = [
    "DEFAULT_SAMPLES_PER_SCENE",
    "SAMPLES_PER_SCENE_CHOICES",
    "ScenesFor",
    "ScreenAnalyzer",
    "ScreenState",
    "scenes_from",
]

#: 場面ごとに調べる枚数の既定 本人の設定（``Preferences.screen_samples``）で変える
#: 縦への作り替え（#334）は場面ごとに枠を決めるので、場面の中の動きを少し拾えれば足りる
DEFAULT_SAMPLES_PER_SCENE = 4
#: 設定で選べる枚数 多いほど場面の中の動き（歩く人・流れるテロップ）を拾うが、その分だけ待つ
SAMPLES_PER_SCENE_CHOICES = (2, 4, 8)

#: 場面の解析を待つ間に、止める頼みを見る間隔（秒）
_POLL_SECONDS = 0.2

#: 素材と、場面の解析が止まったか（真で止める）を受け取り、読み終えた場面の並びを返す関数
#: 場面が分からない（解析に失敗した・止めた）なら ``None``
ScenesFor = Callable[[MediaItem, Callable[[], bool]], SceneSignal | None]

_Key = tuple[MediaId, ScreenKind]


@dataclass(frozen=True, slots=True)
class ScreenState:
    """調べた様子（:meth:`ScreenAnalyzer.state`）"""

    #: 調べ終えた結果、調べている途中の写し、どちらも無ければ ``None``
    report: ScreenReport | None
    #: 調べているか（場面の解析を待っている間と、順番待ちも含む）
    running: bool
    #: 最後に失敗した理由（部品が無い・素材が開けない） 頼み直すか調べ終えるまで残す
    failure: str | None = None

    @property
    def complete(self) -> bool:
        return self.report is not None and self.report.complete


def scenes_from(
    analyzer: MediaAnalyzer, on_ready: Callable[[MediaId], None] | None = None
) -> ScenesFor:
    """場面の解析（:class:`MediaAnalyzer`）に頼んで、読み終えるまで待つ関数を作る

    待つのは画面の解析の係のスレッド 画面のスレッドでは呼ばない
    """

    def scenes(media: MediaItem, cancelled: Callable[[], bool]) -> SceneSignal | None:
        analyzer.request_scenes(media, on_ready=on_ready)
        while not analyzer.wait_scenes(media.id, _POLL_SECONDS):
            if cancelled():
                return None
        signal = analyzer.scene_state(media).signal
        if signal is None or not (signal.complete or signal.broken):
            return None
        return signal

    return scenes


def _stamp(media: MediaItem) -> str:
    """素材が差し替わったかを見る印（大きさと更新時刻）"""
    return media_key(media.path, extra="screen")


class ScreenAnalyzer:
    """素材の画面の中身を、頼まれた物から順に裏で調べる 結果はメモリとディスクに覚える"""

    def __init__(
        self,
        store: CacheStore | None = None,
        *,
        scenes: ScenesFor,
        detectors: Callable[[ScreenKind], Detector] = make_detector,
        per_scene: int = DEFAULT_SAMPLES_PER_SCENE,
    ) -> None:
        self._store = store if store is not None else CacheStore()
        self._scenes = scenes
        self._make_detector = detectors
        self._per_scene = max(1, per_scene)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sashimono-screen")
        self._lock = threading.Lock()
        #: 調べ終える（止まる・落ちるも）たびに起こす AI の道具が待つ
        self._changed = threading.Condition(self._lock)
        #: 調べ終えた結果 （素材, 種類）→（素材の印, 結果）
        self._done: dict[_Key, tuple[str, ScreenReport]] = {}
        self._scans: dict[_Key, ScreenScan] = {}
        self._failures: dict[_Key, str] = {}
        self._running: set[_Key] = set()
        self._cancelled: set[_Key] = set()
        #: 作った調べる係 係のスレッドだけが触る onnxruntime の読み込みは 1 秒ほどかかるので、
        #: 素材ごとに作り直さない
        self._detectors: dict[ScreenKind, Detector] = {}
        self._closed = False

    @property
    def per_scene(self) -> int:
        return self._per_scene

    def set_per_scene(self, count: int) -> None:
        """場面ごとに調べる枚数 次に頼まれた物から効く

        覚えた結果は捨てる 前の枚数で調べた結果を返し続けると、増やしても何も変わらない
        （ディスクの控えは枚数ごとに別の鍵なので、戻せば読み直さずに済む）
        """
        with self._lock:
            if count == self._per_scene:
                return
            self._per_scene = max(1, count)
            self._done.clear()

    def report(self, media: MediaItem, kind: ScreenKind) -> ScreenReport | None:
        """調べ終えた結果 無いか、素材が変わっていれば ``None``（覚えた物は捨てる）"""
        stamp = _stamp(media)
        with self._lock:
            held = self._done.get((media.id, kind))
            if held is None:
                return None
            if held[0] != stamp:
                del self._done[(media.id, kind)]
                return None
            return held[1]

    def state(self, media: MediaItem, kind: ScreenKind) -> ScreenState:
        """調べた様子 調べ終えていればその結果、途中なら調べた所までの写し"""
        done = self.report(media, kind)
        if done is not None:
            return ScreenState(done, running=False)
        key = (media.id, kind)
        with self._lock:
            running = key in self._running
            failure = self._failures.get(key)
            scan = self._scans.get(key)
        return ScreenState(
            scan.snapshot() if scan is not None else None, running=running, failure=failure
        )

    def request(self, media: MediaItem, kind: ScreenKind) -> None:
        """調べるよう頼む 調べ終えた物・調べている物があれば何もしない

        失敗した物は頼み直す 部品を入れてから AI がもう 1 度呼べば、そのまま調べ始める
        """
        if not media.has_video:
            return
        self.report(media, kind)
        key = (media.id, kind)
        with self._lock:
            if self._closed or key in self._done:
                return
            if key in self._running:
                # 止める頼み（設定を切った・素材を外した）の後にまた頼まれたら、走っている物を
                # 生かし直す 止まるのを待ってから頼み直すと、AI が「止まった」を受け取って呼び直す
                self._cancelled.discard(key)
                return
            self._cancelled.discard(key)
            self._failures.pop(key, None)
            self._running.add(key)
            per_scene = self._per_scene
            self._executor.submit(self._run, media, kind, per_scene)

    def wait(self, media_id: MediaId, kind: ScreenKind, timeout: float) -> bool:
        """調べ終える（止まる・落ちるも）まで待つ 終わっていれば真 AI のスレッドから呼ぶ"""
        deadline = time.monotonic() + max(0.0, timeout)
        key = (media_id, kind)
        with self._changed:
            while key in self._running and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._changed.wait(remaining)
            return True

    def cancel(self) -> None:
        """調べている・待っている物をすべて止める 調べ終えた物は残す

        設定で読み込みのときに調べるのを切ったときに呼ぶ AI が頼んだ物も止まるが、
        次に呼ばれたら頼み直す（場面の解析の :meth:`MediaAnalyzer.cancel_scenes` と同じ）
        """
        with self._lock:
            self._cancelled.update(self._running)

    def forget(self, media_id: MediaId) -> None:
        """素材を外したときに、結果と調べている物を捨てる"""
        with self._lock:
            for key in [key for key in self._done if key[0] == media_id]:
                del self._done[key]
            for key in [key for key in self._failures if key[0] == media_id]:
                del self._failures[key]
            self._cancelled.update(key for key in self._running if key[0] == media_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled.update(self._running)
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._changed.notify_all()

    def _is_cancelled(self, key: _Key) -> bool:
        with self._lock:
            return self._closed or key in self._cancelled

    def _detector(self, kind: ScreenKind) -> Detector:
        found = self._detectors.get(kind)
        if found is None:
            found = self._make_detector(kind)
            self._detectors[kind] = found
        return found

    def _run(self, media: MediaItem, kind: ScreenKind, per_scene: int) -> None:
        key = (media.id, kind)
        failure: str | None = None
        try:
            self._examine(media, kind, per_scene)
        except PartsMissingError as exc:
            failure = str(exc)
        except Exception as exc:  # 裏のスレッドの例外は誰にも見えずに消える 理由として残す
            failure = f"{kind.label}を調べられませんでした: {type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                stopped = self._closed or key in self._cancelled
                self._cancelled.discard(key)
                self._running.discard(key)
                self._scans.pop(key, None)
                if failure is not None and not stopped:
                    self._failures[key] = failure
                self._changed.notify_all()

    def _examine(self, media: MediaItem, kind: ScreenKind, per_scene: int) -> None:
        key = (media.id, kind)

        def cancelled() -> bool:
            return self._is_cancelled(key)

        # 先に係を作る 部品が無ければ、場面の解析（全フレームを読む）を待たずに断れる
        detector = self._detector(kind)
        stamp = _stamp(media)
        signal = self._scenes(media, cancelled)
        if cancelled():
            return
        duration = float(media.duration)
        if signal is not None and signal.broken:
            # 途中で壊れた素材は、読めた所までしか絵を取り出せない その先を予定に数えると、
            # 取り出せない絵の分だけ「調べ終えた」のに予定より少ない結果になる（実素材の 362 秒の
            # 動画は 214 秒で壊れていて、9 枚の予定のうち 2 枚を取り出せなかった）
            duration = min(duration, signal.analyzed)
        by_scene = signal is not None
        stream = signal.stream if signal is not None else None
        if signal is not None:
            cuts: tuple[float, ...] = tuple(cut.seconds for cut in signal.cuts())
            count = per_scene
            cache_key: str | None = screen_key(media.path, kind, detector.engine, per_scene, stream)
        else:
            # 場面が分からなければ等間隔で調べる 控えには書かない 場面の解析を直した後も
            # 等間隔の結果を返し続けないため
            cuts = fallback_cuts(duration)
            count = 1
            cache_key = None
        if cache_key is not None:
            cached = load_screen(self._store, cache_key)
            if cached is not None:
                self._publish(key, stamp, cached, per_scene)
                return
        scan = ScreenScan()
        with self._lock:
            self._scans[key] = scan
        report = scan_screen(
            media.path,
            sample_times(cuts, duration, count),
            detector,
            stream_index=stream,
            by_scene=by_scene,
            scan=scan,
            should_cancel=cancelled,
        )
        if report is None:
            return
        if cache_key is not None:
            save_screen(self._store, cache_key, report)
        self._publish(key, stamp, report, per_scene)

    def _publish(self, key: _Key, stamp: str, report: ScreenReport, per_scene: int) -> None:
        """結果を載せる 止められていたら載せない（外した素材の結果を残さないため）

        調べている間に場面ごとの枚数を変えられたら載せない 載せると、変えた後も前の枚数の
        結果を返し続ける（次に頼まれたときに今の枚数で調べ直す）
        """
        with self._lock:
            if self._closed or key in self._cancelled or per_scene != self._per_scene:
                return
            self._done[key] = (stamp, report)
