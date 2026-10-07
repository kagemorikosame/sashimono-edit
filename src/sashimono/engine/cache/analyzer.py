"""波形とサムネイルをバックグラウンドで用意する

素材を読み込んだ直後に UI が固まるのが一番まずい 解析は必ず別スレッドで走らせ、
できたものから順に通知する

キャッシュがあれば解析せずに即返す プロジェクトを開き直すたびに数十秒待つのは
実用に耐えない
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from sashimono.core.model import MediaId, MediaItem
from sashimono.engine.audio.waveform import Waveform, analyze_waveform

from .progress import JobBoard, ProgressSnapshot
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

__all__ = ["MediaAnalyzer", "waveform_stream"]

#: 失敗したときに画面へ出す名前
_KIND_NAMES = {"waveform": "波形", "filmstrip": "サムネイル"}

#: 同時に走らせる解析の数 増やしすぎるとディスクの取り合いで全体が遅くなる
MAX_WORKERS = 2

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
    ) -> None:
        self._store = store if store is not None else CacheStore()
        self._sample_rate = sample_rate
        self._channels = channels
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="sashimono-analyze"
        )
        self._lock = threading.Lock()
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

    def request(
        self, media: MediaItem, *, on_ready: Callable[[MediaId], None] | None = None
    ) -> None:
        """素材の解析を予約する すでにあるもの・処理中のものは無視する"""
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

    def forget(self, media_id: MediaId) -> None:
        """素材を外したときに、結果と進行中の解析を捨てる"""
        with self._lock:
            for owned in [key for key in self._waveforms if key[0] == media_id]:
                del self._waveforms[owned]
            self._filmstrips.pop(media_id, None)
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
            done = (
                (media.id, key[2]) in self._waveforms
                if kind == "waveform"
                else media.id in self._filmstrips
            )
            if done:
                return
            if key in self._running:
                if key in self._cancelled:
                    # 取り消した仕事はまだ走っているが、結果を載せない 終わってから頼み直す
                    self._again[key] = (media, work, on_ready)
                return
            self._cancelled.discard(key)
            self._running.add(key)
            self._board.start(key, media.id)

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
                    self._running.discard(key)
                    self._cancelled.discard(key)
                    again = self._again.pop(key, None)
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
                self._submit(key, *again)

        with self._lock:
            # close と同じロックの中で投げる（close の説明を参照）
            if self._closed:
                self._running.discard(key)
                self._board.drop(key)
                return
            self._executor.submit(run)

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
