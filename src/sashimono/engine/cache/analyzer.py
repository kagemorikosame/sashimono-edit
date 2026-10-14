"""波形・サムネイル・場面の一覧をバックグラウンドで用意する

素材を読み込んだ直後に UI が固まるのが一番まずい 解析は必ず別スレッドで走らせ、
できたものから順に通知する

キャッシュがあれば解析せずに即返す プロジェクトを開き直すたびに数十秒待つのは
実用に耐えない

場面の一覧（#321）は素材を全フレーム読むので、波形やサムネイルよりずっと重い
別の 1 本のスレッドで順に走らせる 同じ置き場で走らせると、長い動画の場面を読む間、
後から読み込んだ素材のサムネイルが出ない
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction

from sashimono.core.model import MediaId, MediaItem
from sashimono.engine.audio.waveform import Waveform, analyze_waveform
from sashimono.engine.scenes import SceneScan, SceneSignal, analyze_scenes

from .progress import JobBoard, ProgressSnapshot
from .scene_cache import load_scenes, save_scenes, scenes_key
from .store import CacheStore
from .thumbnails import (
    DEFAULT_INTERVAL,
    THUMBNAIL_HEIGHT,
    Filmstrip,
    build_filmstrip,
    filmstrip_key,
    load_filmstrip,
    save_filmstrip,
)
from .waveform_cache import load_waveform, save_waveform, waveform_key

__all__ = ["DEFAULT_SCENE_THREADS", "MediaAnalyzer", "SceneState", "waveform_stream"]

#: 失敗したときに画面へ出す名前
_KIND_NAMES = {"waveform": "波形", "filmstrip": "サムネイル", "scenes": "場面の一覧"}

#: 同時に走らせる解析の数 増やしすぎるとディスクの取り合いで全体が遅くなる
MAX_WORKERS = 2

#: 場面の解析に使うデコードのスレッドの既定 本人の設定（``Preferences.scene_threads``）で変える
#: 少なめにするのは、裏で読む間も編集と再生に CPU を残すため 実測は
#: :data:`sashimono.engine.scenes.MEASURED_SCENE_SECONDS`
DEFAULT_SCENE_THREADS = 2


@dataclass(frozen=True, slots=True)
class SceneState:
    """場面の解析の今の様子（:meth:`MediaAnalyzer.scene_state`）"""

    #: 読み終えた並び、読んでいる途中の写し、どちらも無ければ ``None``
    signal: SceneSignal | None
    #: 解析が走っているか（順番待ちも含む）
    running: bool
    #: 最後に失敗した理由 頼み直すか読み終えるまで残す
    failure: str | None = None

    @property
    def complete(self) -> bool:
        return self.signal is not None and self.signal.complete


#: 解析の仕事の鍵（種類, 素材, 音声ストリーム） ストリームは 2 本目以降の音の波形だけが持ち、
#: 1 本目の音とサムネイルは ``None`` 素材だけを鍵にすると、音声が何本もある動画の
#: どの音のクリップにも 1 本目の波形が出る（利用者の画面で 4 本とも同じ波形だった）
_JobKey = tuple[str, MediaId, int | None]


def waveform_stream(media: MediaItem, stream: int | None) -> int | None:
    """波形を引く鍵にする音声ストリームの番号 1 本目の音と、素材に無い番号は ``None``

    デコーダ（:class:`~sashimono.engine.decode.AudioDecoder`）は素材に無い番号を渡されると
    1 本目を開く 鍵も同じく 1 本目へ寄せないと、同じ音を 2 回解析する 1 本目を ``None`` に
    するのは、ストリームを区別する前に作った控え（1 本目の波形）をそのまま使うため
    """
    if stream is None or not media.audio_streams:
        return None
    if stream == media.audio_streams[0].index:
        return None
    return stream if any(s.index == stream for s in media.audio_streams) else None


class MediaAnalyzer:
    """素材の波形とサムネイルを非同期に用意する

    結果はメモリにも保持するので、2 度目以降はディスクも読まない
    """

    def __init__(
        self,
        store: CacheStore | None = None,
        *,
        sample_rate: int = 48000,
        channels: int = 2,
        max_workers: int = MAX_WORKERS,
        scene_threads: int = DEFAULT_SCENE_THREADS,
    ) -> None:
        self._store = store if store is not None else CacheStore()
        self._sample_rate = sample_rate
        self._channels = channels
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="sashimono-analyze"
        )
        #: 場面の解析だけを順に走らせる置き場（このファイルの頭の説明）
        self._scene_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sashimono-scenes"
        )
        self._scene_threads = scene_threads
        self._lock = threading.Lock()
        #: 場面の解析が終わる（止まる・落ちるも）たびに起こす AI の道具が待つ
        #: 錠は上の物を共にする 別の錠にすると、終わりを書く所と待つ所の間で見落とす
        self._changed = threading.Condition(self._lock)
        #: 場面の一覧 素材 → （控えの鍵, 並び） 鍵を一緒に持つのは、素材を差し替えた
        #: （大きさか更新時刻が変わった）ことに気付いて読み直すため（:meth:`scenes`）
        self._scenes: dict[MediaId, tuple[str, SceneSignal]] = {}
        #: 読んでいる途中の場面の並び 途中までを AI に渡すため
        self._scans: dict[MediaId, SceneScan] = {}
        #: 場面の解析が失敗した理由 AI に「待っても出ない」ことを伝えるため
        self._scene_failures: dict[MediaId, str] = {}
        #: 波形は音声ストリームごとに持つ（:func:`waveform_stream`）
        self._waveforms: dict[tuple[MediaId, int | None], Waveform] = {}
        self._filmstrips: dict[MediaId, Filmstrip] = {}
        #: 実行中の解析 Future ではなく鍵の集合で持つ Future を辞書に
        #: 入れ直す形にすると、投入前に完了した場合に消し損ねる
        self._running: set[_JobKey] = set()
        self._cancelled: set[_JobKey] = set()
        #: 取り消し印の付いた仕事が走っている間に、同じ鍵でまた頼まれた物 走っている仕事は
        #: 結果を載せずに終わるので、終わってから走らせ直す 捨てると、素材一覧から消して
        #: すぐ取り消したときに、名前は出るのにサムネイルと波形がいつまでも出なかった
        self._again: dict[
            _JobKey,
            tuple[
                MediaItem,
                Callable[[MediaItem, _JobKey, Callable[[float], None]], bool],
                Callable[[MediaId], None] | None,
            ],
        ] = {}
        self._closed = False
        #: 画面へ出す進み具合 数えるのは裏のスレッド、読むのは画面のタイマー
        self._board = JobBoard()

    def poll(self) -> ProgressSnapshot:
        """画面へ出す進み具合 画面のスレッドのタイマーから呼ぶ（:meth:`JobBoard.poll`）"""
        return self._board.poll()

    def settle(self, seen: ProgressSnapshot) -> bool:
        """画面が終わりを見届けた ひと続きの数を戻す（:meth:`JobBoard.settle`）"""
        return self._board.settle(seen)

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None:
        """すでに用意できていれば返す 無ければ ``None``

        ``stream`` は鳴らす音声ストリームの番号（:func:`~sashimono.core.model.heard_stream`）
        省けば 1 本目の音 描画のたびに呼ばれるので、ここでは決してブロックしない
        """
        with self._lock:
            return self._waveforms.get((media.id, waveform_stream(media, stream)))

    def filmstrip(self, media: MediaItem) -> Filmstrip | None:
        with self._lock:
            return self._filmstrips.get(media.id)

    def scenes(self, media: MediaItem) -> SceneSignal | None:
        """読み終えた場面の並び 無いか、素材が変わっていれば ``None``

        素材のファイルの大きさと更新時刻を見るので、描画のたびには呼ばない（AI の道具から呼ぶ）
        変わっていたら覚えた物を捨てる 捨てないと、差し替えた素材に前の素材の切り替わりを返す
        """
        key = scenes_key(media.path)
        with self._lock:
            held = self._scenes.get(media.id)
            if held is None:
                return None
            if held[0] != key:
                del self._scenes[media.id]
                return None
            return held[1]

    def scene_state(self, media: MediaItem) -> SceneState:
        """場面の解析の今 読み終えていればその並び、途中なら読んだ所までの写し"""
        done = self.scenes(media)
        if done is not None:
            return SceneState(done, running=False)
        with self._lock:
            running = ("scenes", media.id, None) in self._running
            failure = self._scene_failures.get(media.id)
            scan = self._scans.get(media.id)
        partial = scan.snapshot() if scan is not None else None
        return SceneState(partial, running=running, failure=failure)

    def request_scenes(
        self, media: MediaItem, *, on_ready: Callable[[MediaId], None] | None = None
    ) -> None:
        """場面の解析を予約する 読み終えた物・走っている物があれば何もしない

        覚えた物が古ければ（素材が変わった）捨ててから頼む（:meth:`scenes`）
        """
        if not media.has_video:
            return
        self.scenes(media)
        self._submit(("scenes", media.id, None), media, self._analyze_scenes, on_ready)

    def wait_scenes(self, media_id: MediaId, timeout: float) -> bool:
        """場面の解析が終わる（止まる・落ちるも）まで待つ 終わっていれば真

        AI のスレッドから呼ぶ 画面のスレッドで待つと、待つ間は画面が固まる
        頼んでいない素材は待たずに真を返す
        """
        deadline = time.monotonic() + max(0.0, timeout)
        key: _JobKey = ("scenes", media_id, None)
        with self._changed:
            while key in self._running and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._changed.wait(remaining)
            return True

    def set_scene_threads(self, threads: int) -> None:
        """場面の解析に使うデコードのスレッドの数 次に始める解析から効く 0 はすべて"""
        self._scene_threads = max(0, threads)

    def cancel_scenes(self) -> None:
        """走っている・待っている場面の解析をすべて止める 読み終えた物は残す

        設定で取り込み時の解析を切ったときに呼ぶ 切っても裏で読み続けるなら、
        切った意味が無い
        """
        with self._lock:
            self._cancelled.update(key for key in self._running if key[0] == "scenes")
            for job in [key for key in self._again if key[0] == "scenes"]:
                del self._again[job]

    def request(
        self,
        media: MediaItem,
        *,
        on_ready: Callable[[MediaId], None] | None = None,
        scenes: bool = False,
    ) -> None:
        """素材の解析を予約する すでにあるもの・処理中のものは無視する

        ``scenes`` が真なら場面の一覧も（本人の設定 ``Preferences.scene_analysis``）
        """
        # 音声は全部のストリームを解析する 置いた後で鳴らす音を切り替えたクリップにも、
        # 切り替えた先の波形を出すため
        for audio in media.audio_streams:
            self._submit(
                ("waveform", media.id, waveform_stream(media, audio.index)),
                media,
                self._analyze_waveform,
                on_ready,
            )
        if media.has_video:
            self._submit(("filmstrip", media.id, None), media, self._analyze_filmstrip, on_ready)
        if scenes:
            self.request_scenes(media, on_ready=on_ready)

    def forget(self, media_id: MediaId) -> None:
        """素材を外したときに、結果と進行中の解析を捨てる"""
        with self._lock:
            for owned in [key for key in self._waveforms if key[0] == media_id]:
                del self._waveforms[owned]
            self._filmstrips.pop(media_id, None)
            self._scenes.pop(media_id, None)
            self._scene_failures.pop(media_id, None)
            self._cancelled.update(key for key in self._running if key[1] == media_id)
            # 走らせ直す約束も捨てる 外した素材を、取り消し中の仕事が終わった後に解析し直さない
            for job in [key for key in self._again if key[1] == media_id]:
                del self._again[job]
        self._board.forget(media_id)

    def close(self) -> None:
        # 投入（_submit）と同じロックの中で止める ロックの外で止めると、投入側が
        # 「まだ止まっていない」と見た直後に止まり、停止済みの executor へ投げて
        # RuntimeError になる 投入したキーも _running に残ったままになる
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled.update(self._running)
            self._again.clear()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._scene_executor.shutdown(wait=False, cancel_futures=True)
            # 待っている AI の道具を起こす 閉じた後は終わらないので、待たせても無駄
            self._changed.notify_all()

    def _submit(
        self,
        key: _JobKey,
        media: MediaItem,
        work: Callable[[MediaItem, _JobKey, Callable[[float], None]], bool],
        on_ready: Callable[[MediaId], None] | None,
    ) -> None:
        kind = key[0]
        with self._lock:
            if self._closed:
                return
            if kind == "waveform":
                done = (media.id, key[2]) in self._waveforms
            elif kind == "scenes":
                done = media.id in self._scenes
            else:
                done = media.id in self._filmstrips
            if done:
                return
            if key in self._running:
                if key in self._cancelled:
                    # 取り消した仕事はまだ走っているが、結果を載せない 終わってから頼み直す
                    self._again[key] = (media, work, on_ready)
                return
            self._cancelled.discard(key)
            self._running.add(key)
            if kind == "scenes":
                # 頼み直したら前の失敗は消す 残すと、読み直している間も「出せない」と返す
                self._scene_failures.pop(media.id, None)
            self._board.start(key, media.id)
        self._launch(key, media, work, on_ready)

    def _launch(
        self,
        key: _JobKey,
        media: MediaItem,
        work: Callable[[MediaItem, _JobKey, Callable[[float], None]], bool],
        on_ready: Callable[[MediaId], None] | None,
    ) -> None:
        """``_running`` に載せた仕事を投げる 載せるのは呼ぶ側（:meth:`_submit` と走らせ直し）"""
        kind = key[0]

        def report(value: float) -> None:
            self._board.report(key, value)

        def run() -> None:
            produced = False
            failure: str | None = None
            try:
                produced = work(media, key, report)
            except Exception as exc:  # 裏のスレッドの例外は誰にも見えずに消える
                # 投げ直さずに失敗として数える 前は executor の中で黙って消え、
                # 波形が出ないまま理由も分からなかった
                failure = f"{_KIND_NAMES[kind]}を作れなかった: {exc}"
            finally:
                with self._lock:
                    stopped = self._closed or key in self._cancelled
                    self._cancelled.discard(key)
                    again = self._again.pop(key, None)
                    # 走らせ直すなら、鍵を ``_running`` に載せたまま次へ渡す 1 度外してから
                    # 投げ直すと、その間に素材を外された（forget）とき、forget は旧い仕事も
                    # 次の仕事も見つけられず、外した素材の解析が始まって結果が載る
                    if again is None:
                        self._running.discard(key)
                    if kind == "scenes":
                        if not produced and not stopped:
                            self._scene_failures[media.id] = (
                                failure or f"{_KIND_NAMES[kind]}を作れなかった"
                            )
                        self._changed.notify_all()
            if stopped:
                self._board.drop(key)
            elif produced:
                self._board.finish(key)
            else:
                self._board.finish(key, failure or f"{_KIND_NAMES[kind]}を作れなかった")
            # 取り消された仕事の結果は載っていない（:meth:`_publish`） 載せていないのに
            # 「できた」と伝えると、描き直しても空のまま
            if produced and not stopped and on_ready is not None:
                on_ready(media.id)
            if again is not None:
                with self._lock:
                    # 間に外されていれば ``_cancelled`` に入っている 次の仕事はすぐ止まり、
                    # 結果も載せない（:meth:`_publish`）
                    self._board.start(key, again[0].id)
                self._launch(key, *again)

        with self._lock:
            # close と同じロックの中で投げる（close の説明を参照）
            if self._closed:
                self._running.discard(key)
                self._cancelled.discard(key)
                self._board.drop(key)
                self._changed.notify_all()
                return
            executor = self._scene_executor if kind == "scenes" else self._executor
            executor.submit(run)

    def _publish(
        self,
        kind: str,
        media_id: MediaId,
        result: Waveform | Filmstrip,
        stream: int | None = None,
    ) -> bool:
        """結果を登録する 取り消されていたら登録しない

        解析は時間が掛かるので、走っている間に素材が外される（forget）ことがある
        確かめずに登録すると、外した素材の波形やサムネイルが復活する
        確かめるのと登録するのを同じロックの中で行う
        """
        with self._lock:
            if self._closed or (kind, media_id, stream) in self._cancelled:
                return False
            if isinstance(result, Waveform):
                self._waveforms[(media_id, stream)] = result
            else:
                self._filmstrips[media_id] = result
            return True

    def _publish_scenes(
        self, job: _JobKey, media_id: MediaId, key: str, signal: SceneSignal
    ) -> bool:
        """場面の並びを登録する 取り消されていたら登録しない（:meth:`_publish` と同じ理由）"""
        with self._lock:
            if self._closed or job in self._cancelled:
                return False
            self._scenes[media_id] = (key, signal)
            return True

    def _is_cancelled(self, key: _JobKey) -> bool:
        with self._lock:
            return self._closed or key in self._cancelled

    def _analyze_waveform(
        self, media: MediaItem, job: _JobKey, report: Callable[[float], None]
    ) -> bool:
        stream = job[2]
        key = waveform_key(media.path, self._sample_rate, self._channels, stream=stream)
        waveform = load_waveform(self._store, key)

        if waveform is None:
            waveform = analyze_waveform(
                media.path,
                sample_rate=self._sample_rate,
                channels=self._channels,
                stream_index=stream,
                progress=report,
                should_cancel=lambda: self._is_cancelled(job),
            )
            if waveform is None:
                return False
            save_waveform(self._store, key, waveform)

        return self._publish("waveform", media.id, waveform, stream)

    def _analyze_filmstrip(
        self, media: MediaItem, job: _JobKey, report: Callable[[float], None]
    ) -> bool:
        interval = _interval_for(media.duration)
        key = filmstrip_key(media.path, interval, THUMBNAIL_HEIGHT)
        filmstrip = load_filmstrip(self._store, key)

        if filmstrip is None:
            filmstrip = build_filmstrip(
                media.path,
                interval=interval,
                height=THUMBNAIL_HEIGHT,
                progress=report,
                should_cancel=lambda: self._is_cancelled(job),
            )
            if filmstrip is None:
                return False
            save_filmstrip(self._store, key, filmstrip)

        return self._publish("filmstrip", media.id, filmstrip)

    def _analyze_scenes(
        self, media: MediaItem, job: _JobKey, report: Callable[[float], None]
    ) -> bool:
        key = scenes_key(media.path)
        signal = load_scenes(self._store, key)

        if signal is None:
            scan = SceneScan()
            with self._lock:
                self._scans[media.id] = scan
            try:
                signal = analyze_scenes(
                    media.path,
                    threads=self._scene_threads,
                    scan=scan,
                    progress=report,
                    should_cancel=lambda: self._is_cancelled(job),
                )
            finally:
                with self._lock:
                    if self._scans.get(media.id) is scan:
                        del self._scans[media.id]
            if signal is None:
                return False
            save_scenes(self._store, key, signal)

        return self._publish_scenes(job, media.id, key, signal)


def _interval_for(duration: Fraction) -> Fraction:
    """素材の長さに応じたサムネイル間隔

    短い素材は細かく、長い素材は粗く 一定にすると、1 時間の素材で
    7200 枚を作ることになる
    """
    if duration <= 0:
        return DEFAULT_INTERVAL
    if duration <= 60:
        return DEFAULT_INTERVAL
    if duration <= 600:
        return Fraction(2)
    return Fraction(5)
