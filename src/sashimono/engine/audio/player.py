"""音声の再生と、再生位置の時計

A/V 同期はオーディオを基準にする 映像を基準にすると、映像の遅れを取り戻すために
音を飛ばすことになり、途切れが即座に耳につく 音は途切れさせず、映像側が
追いつけない分をコマ落としで吸収する

そのため、このクラスは「音を出す」だけでなく「今どこを再生しているか」を答える
時計でもある
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from sashimono.engine.audio.mixer import AudioMixer

__all__ = ["AudioPlayer", "PlaybackError"]

#: 出力に一度に書き込むサンプル数 小さいほど遅延は減るが、書き込み回数が増える
BLOCK_SAMPLES = 1024

#: 出力デバイスに要求する遅延 小さくしすぎると音が途切れる
LATENCY = "low"

#: 書き終えたあと、残りが鳴り終わるまで時計を進める間隔（秒）
#: 再生の側が位置を見に行く間隔（8ms）より細かくして、最後の数フレームも飛ばさず出す
DRAIN_POLL_SECONDS = 0.004

#: 書いた時刻からの経過で位置を補う量の上限（サンプル 最後に書いたかたまりから数える）
#: 位置はかたまりを 1 つ書くたびにしか進まず、補わないと 48kHz で約 21 ms 刻みになる
#: 60fps（16.7 ms ごと）の素材では 1 コマ飛ばしが規則的に起きた（#463）
#: 1 かたまりちょうどにすると、書き込みが戻るのが数 ms 遅れただけで補いが上限で止まり、
#: 次の書き込みで 2 コマ進む 手元の MME では戻る間隔の 95% が 30 ms で、1 かたまりの
#: 上限では 60fps のコマの 2 割ほどが 25 ms を越えて出た 2 かたまり（約 43 ms）なら
#: 揺れを越えて進め、デバイスが詰まって書けない間はここで止まり、映像だけが音より
#: 先へ走らない
EXTRAPOLATE_LIMIT = 2 * BLOCK_SAMPLES

#: 位置を見積もるのに使う、最近書いたかたまりの数
#: 書き込みが戻るのは、デバイスが鳴らし終えた後で、前には戻らない（遅れることだけがある）
#: なので「書いた時刻からの経過で進めた位置」はどれも本当の位置より手前か同じで、
#: その中の一番先が本当の位置に一番近い 古いかたまりまで見ると、デバイスと
#: パソコンの時計の速さの差（ppm）がたまって先へ出るので、最近の分だけを見る
ANCHOR_WINDOW = 8

#: 前の書き込みからこのかたまりの数の時間より空いたら、覚えた時刻を捨てて補いを始め直す
#: 書く側が遅れてデバイスが空になった（アンダーラン）後は、鳴っていなかった間も
#: 古い時刻から経過で進めると、詰め直している途中の音より映像が先へ出る
#: 手元の MME では書き込みが戻る間隔の 95% が 30 ms なので、2 かたまり（約 43 ms）で分ける
UNDERRUN_BLOCKS = 2


class PlaybackError(RuntimeError):
    """音声出力を開けない"""


class AudioPlayer:
    """ミキサの出力をサウンドデバイスへ流す

    再生は専用スレッドで行う デコードを含むミックスをオーディオコールバックの
    中で行うと、ディスクが詰まった瞬間に音が途切れる
    """

    def __init__(
        self,
        mixer: AudioMixer,
        *,
        on_finished: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._mixer = mixer
        self._on_finished = on_finished
        # 経過を測る時計 ``time.monotonic`` は Windows では約 16 ms 刻みで、補う幅
        # （21 ms）と同じくらい粗く、補っても刻みが残る 試験では作り物を渡す
        self._clock = clock
        self._stream: sd.OutputStream | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._position = 0
        self._written = 0
        self._start_sample = 0
        self._end_sample: int | None = None
        #: 最近書いたかたまりの（書き終えた時刻, そのときの位置） 補ってよいときだけ持つ
        #: （:meth:`_advance`）
        self._anchors: deque[tuple[float, int]] = deque(maxlen=ANCHOR_WINDOW)
        #: 最後に答えた位置 補った位置より手前を答えると、映像が 1 コマ戻って見える
        self._reported = 0

    @property
    def is_playing(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def position_sample(self) -> int:
        """今スピーカーから出ているおおよそのサンプル位置

        デバイスのバッファに積んだ分だけ、書き込み位置は実際の再生位置より先を
        行っている その差を引いて返す

        かたまりを書いた後は、書いた時刻からの経過で位置を補う（:data:`ANCHOR_WINDOW`
        上限は :data:`EXTRAPOLATE_LIMIT`） 音を時計にする作りは変えず、時計の針を
        かたまりの間でも進めるだけ 補った位置は書き込み済みを越えず、
        一度答えた位置より手前も答えない
        """
        with self._lock:
            position = self._position
            if self._anchors and not self._stop.is_set():
                now = self._clock()
                rate = self._mixer.sample_rate
                estimate = max(base + int(max(0.0, now - at) * rate) for at, base in self._anchors)
                position = max(
                    position,
                    min(
                        estimate,
                        position + EXTRAPOLATE_LIMIT,
                        self._start_sample + self._written,
                    ),
                )
            position = max(position, self._reported)
            self._reported = position
            return position

    def start(self, from_sample: int, *, end_sample: int | None = None) -> None:
        """``from_sample`` から再生を始める すでに再生中なら一度止める"""
        self.stop()
        self._stop.clear()
        self._reset(from_sample, end_sample)

        try:
            self._stream = sd.OutputStream(
                samplerate=self._mixer.sample_rate,
                channels=self._mixer.channels,
                dtype="float32",
                blocksize=BLOCK_SAMPLES,
                latency=LATENCY,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            self._stream = None
            raise PlaybackError(f"音声出力を開けない: {exc}") from exc

        self._thread = threading.Thread(target=self._run, name="sashimono-audio", daemon=True)
        self._thread.start()

    def _reset(self, from_sample: int, end_sample: int | None) -> None:
        """時計を ``from_sample`` に置き直す

        前の回の補いと答えた位置も捨てる 残すと、手前へ飛んだのに前の位置を答え続ける
        """
        self._start_sample = max(0, from_sample)
        self._end_sample = end_sample
        with self._lock:
            self._position = self._start_sample
            self._written = 0
            self._anchors.clear()
            self._reported = self._start_sample

    def stop(self) -> None:
        """再生を止める 止まるまで待つ"""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        # 止めた後は補わない 止めた後も経過で進めると、止めた所より先を答える
        with self._lock:
            self._anchors.clear()
        self._close_stream()

    def close(self) -> None:
        self.stop()

    def _run(self) -> None:
        stream = self._stream
        if stream is None:
            return

        cursor = self._start_sample
        buffered = 0
        try:
            while not self._stop.is_set():
                count = BLOCK_SAMPLES
                if self._end_sample is not None:
                    count = min(count, self._end_sample - cursor)
                    if count <= 0:
                        self._drain(stream, cursor, buffered)
                        break

                block = self._mixer.render(cursor, count)
                # 出力段で 1 度だけ頭打ちにする 途中で潰すと、後段の調整で
                # 潰れた音しか扱えなくなる
                # 戻り値はデバイスが空になったか（アンダーラン） 書き込みの間が短くても、
                # 出口のバッファが小さければ空になることがある
                underflowed = stream.write(np.clip(block, -1.0, 1.0))
                cursor += count
                buffered = int(stream.latency * self._mixer.sample_rate)
                self._advance(count, buffered, underflowed=bool(underflowed))
        except (sd.PortAudioError, RuntimeError):
            # デバイスが抜かれた等 再生を諦めるが、アプリは落とさない
            pass
        finally:
            # 鳴らなくなった後も経過で進めると、止まった所より先を答える
            with self._lock:
                self._anchors.clear()
            if not self._stop.is_set() and self._on_finished is not None:
                self._on_finished()

    def _advance(self, count: int, buffered: int, *, underflowed: bool = False) -> None:
        """かたまりを 1 つ書き終えた 位置を進め、この時刻を補いの起点に足す

        書き込みはデバイスに空きができるまで戻らないので、戻った時刻はおおよそ
        バッファの頭のかたまりが鳴り終えた時刻 そこからの経過で位置を補える

        書いた量がまだバッファに収まっている（鳴り始める前の）間は補わない
        鳴っていない所を経過で進めると、音より先に映像が動き出す

        デバイスが空になった後（``underflowed`` か、前の書き込みから :data:`UNDERRUN_BLOCKS`
        より空いた）は覚えた時刻を捨てる 鳴っていなかった間も古い時刻から進めると、
        詰め直している途中の音より映像が先へ出る 空になった直後の位置は、バッファが
        満ちている前提で引くので本当の位置より手前で、そこから補っても先へは出ない
        """
        with self._lock:
            self._written += count
            # 書き込んだ分から、まだデバイスのバッファに残っている分を引く
            heard = self._written - buffered
            self._position = max(self._start_sample, self._start_sample + heard)
            now = self._clock()
            gap = UNDERRUN_BLOCKS * BLOCK_SAMPLES / self._mixer.sample_rate
            if underflowed or (self._anchors and now - self._anchors[-1][0] > gap):
                self._anchors.clear()
            if heard > 0:
                self._anchors.append((now, self._position))
            else:
                self._anchors.clear()

    def _drain(self, stream: sd.OutputStream, end: int, buffered: int) -> None:
        """終わりまで書き込んだあと、デバイスに残った分が鳴り終わるまで時計を進める

        書き込みを終えた時点の位置は、終わりからデバイスのバッファの分（数フレーム）
        手前 ここで抜けると時計がそこで止まり、終わりまで再生したのに再生ヘッドが
        終わりの少し前で止まった（Issue #27 13:26:01 の所が 13:25:27）
        鳴っている間は経過時間で進め、鳴り終えたら終わりちょうどに置く

        ``latency`` は遅れの見積もりで、残っている量そのものではない 見積もりの時間が
        過ぎても鳴り終わっているとは限らないので、終わりに置く前に出口を止めて
        （``stop`` は積んだ音を鳴らし切るまで戻らない）本当に鳴り終えるのを待つ
        待たないと、音がまだ出ている間に再生ヘッドが終わりへ着いて止まる
        """
        rate = self._mixer.sample_rate
        # 書いた量よりバッファの方が大きい（短い区間だけ鳴らした）ときは、書いた分しか残っていない
        buffered = min(buffered, end - self._start_sample)
        # ここからは下の繰り返しが細かく進めるので、かたまりの間の補いは要らない
        with self._lock:
            self._anchors.clear()
        started = time.monotonic()
        while buffered > 0 and not self._stop.is_set():
            played = int((time.monotonic() - started) * rate)
            if played >= buffered:
                break
            with self._lock:
                self._position = max(self._position, end - buffered + played)
            self._stop.wait(DRAIN_POLL_SECONDS)
        if self._stop.is_set():
            return
        # 止め方の失敗は握る 鳴り終わりを待てなかっただけで、終わりに置けないと
        # 再生ヘッドが手前で止まったままになる（外から閉じるときにもう 1 度止めに行く）
        with contextlib.suppress(sd.PortAudioError):
            stream.stop()
        if not self._stop.is_set():
            with self._lock:
                self._position = end

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except sd.PortAudioError:
            pass
