"""重い裏の処理を別のプロセスで走らせる共通の仕組み（#440 #443 の段階 2）

同じプロセスのスレッドで 1 コマごとに PyAV を呼ぶ処理は、画面や先読みのスレッドと GIL を
取り合う 控えの作成は単体なら約 25 倍速なのに、アプリの中では約 3 倍速だった（#440）
別のプロセスにすれば GIL は別々になり、画面のスレッドも裏の処理に足止めされない

作り

- 子は ``multiprocessing`` の spawn で起こす 配布版（PyInstaller）では自分の exe が
  ``--multiprocessing-fork`` 付きで起き、:func:`sashimono.app.main` の頭の
  :func:`multiprocessing.freeze_support` が子の役へ回す 専用の小さな exe を同梱する案より、
  組み立てと署名・Defender の確かめ（配る zip の自己診断）が 1 本で済む
- 子に渡すのは**名前で import できる関数**と、pickle できる引数だけ 関数は
  ``progress`` と ``should_cancel`` を受ける（スレッドで呼ぶときと同じ形）ので、同じ関数を
  スレッドでもプロセスでも走らせられる
- 子から親へは進み具合・できた・失敗の 3 つを、親から子へは「やめて」だけを、別々の
  管で送る 1 本の管を 2 つのスレッドで読み書きすると、Windows の管では取り違えが起きうる
- 親が落ちたら、子は「やめて」の管の向こうが閉じたことで気付いてやめる 書きかけを
  消して終わるので、親の無いまま控えを作り続ける子は残らない
- やめると言ってから :data:`CANCEL_GRACE_S` 待っても終わらなければ、子を止める
  （書きかけの ``.part`` は親の側の呼ぶ所で消す）

子が import の途中で落ちた（起きる前に Defender が止めた・exe が無い）ときは
:class:`WorkerUnavailableError` を投げる 呼ぶ側はスレッドへ戻せる 起きた後に落ちた
（素材のせいで FFmpeg が落ちた）ときは :class:`WorkerCrashedError` スレッドで同じ素材を
読み直すと、今度はアプリごと落ちるので戻さない
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import threading
import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

__all__ = [
    "CANCEL_GRACE_S",
    "START_TIMEOUT_S",
    "WorkerCrashedError",
    "WorkerError",
    "WorkerFailedError",
    "WorkerUnavailableError",
    "probe_job",
    "run_in_process",
    "stop_all",
]

#: やめると言ってから子が自分で終わるのを待つ長さ（秒）
#: 控えの作成は 1 コマごとに確かめるので普通は 0.1 秒もかからない 待たずに止めると、
#: 書きかけの消し損ねと、エンコーダ（NVENC）の後片付けの飛ばしが起きる
CANCEL_GRACE_S = 3.0

#: 子が「起きた」を送ってくるまで待つ長さ（秒） 起きた後の仕事の長さには掛けない
#: 配布版は exe ごと読み直すので、冷えた機械で見張りのソフトが中を調べると 10 秒を超える
#: ことがある 短くするとそういう機械で毎回スレッドへ戻り、無いと子が起動の途中で
#: 固まったときに控えの係と自己診断が止まったままになる
START_TIMEOUT_S = 60.0

#: 親が子からの知らせを待つ 1 回の長さ（秒） やめるかどうかもこの間隔で見る
POLL_S = 0.1

#: 子が進み具合を送る最短の間隔（秒） 1 コマごとに送ると、管の読み書きが変換より重くなる
PROGRESS_INTERVAL_S = 0.1

#: 子へ「途中の値を送る口を渡す」と伝える引数の名前 ``job`` の引数と重ならない名前にする
_PUBLISH = "__sashimono_publish__"

#: 起こした子を、閉じるときにまとめて止めるために覚えておく
_LIVE: weakref.WeakSet[BaseProcess] = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


class WorkerError(RuntimeError):
    """別のプロセスでの処理がうまくいかなかった"""


class WorkerUnavailableError(WorkerError):
    """子のプロセスを起こせなかった（起きる前に終わった） スレッドへ戻してよい"""


class WorkerCrashedError(WorkerError):
    """子が仕事の途中で落ちた 同じ仕事をこのプロセスで繰り返すと、こちらも落ちうる"""


class WorkerFailedError(WorkerError):
    """子の中で仕事が例外を投げた 理由の文は子の側の物"""


class _Channel(Protocol):
    """管の端 Windows と ほかの OS で型の名前が違う（PipeConnection と Connection）ので形で受ける"""

    def send(self, obj: Any, /) -> None: ...
    def recv(self) -> Any: ...
    def poll(self, timeout: float | None = 0.0, /) -> bool: ...
    def close(self) -> None: ...


def _context() -> multiprocessing.context.SpawnContext:
    # fork は使わない 画面のスレッドや GL を抱えたまま複製すると、子の中で錠が
    # 取られたままになる Windows にはそもそも spawn しかなく、どの OS でも同じ道を通す
    return multiprocessing.get_context("spawn")


def run_in_process[T](
    job: Callable[..., T],
    kwargs: dict[str, Any],
    *,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    partial: Callable[[Any], None] | None = None,
    name: str = "sashimono-worker",
    start_timeout_s: float = START_TIMEOUT_S,
) -> T | None:
    """``job(progress=..., should_cancel=..., **kwargs)`` を別のプロセスで走らせて結果を返す

    やめたときは ``None`` 呼ぶ側のスレッドはほとんど眠って待つ（GIL を手放す）
    ``partial`` を渡すと、``job`` は ``publish=`` も受け取る 子が ``publish(値)`` で送った
    途中の値（pickle できる物）が、呼ぶ側のスレッドで ``partial(値)`` として届く
    場面の解析で、読み終えた所までの並びを AI へ先に渡すのに使う（#321 #453）
    送る間隔は ``job`` の側で決める 管の読み書きは仕事より重くなりうる
    """
    if partial is not None:
        kwargs = {**kwargs, _PUBLISH: True}
    context = _context()
    ends: list[_Channel] = []
    try:
        # 子 → 親（知らせ）と 親 → 子（やめて）を別の管にする
        # 管を作る所から包む ハンドルが尽きた機械で管を作れないのも「起こせない」のうちで、
        # 包まないとスレッドへ戻れば作れる控えまで失敗になる
        news_out, news_in = context.Pipe(duplex=False)
        ends += (news_out, news_in)
        stop_out, stop_in = context.Pipe(duplex=False)
        ends += (stop_out, stop_in)
        process = context.Process(
            target=_child_main, args=(news_in, stop_out, job, kwargs), name=name, daemon=True
        )
        process.start()
    except Exception as exc:  # 起こせない理由は環境ごとに違う（OSError・pickle の失敗など）
        for end in ends:
            end.close()
        raise WorkerUnavailableError(f"別のプロセスを起こせなかった: {exc}") from exc
    # 子に渡した端は親の側で閉じる 閉じないと、子が落ちても管の向こうが開いたままに見え、
    # 親が終わりの知らせを待ち続ける
    news_in.close()
    stop_out.close()
    with _LIVE_LOCK:
        _LIVE.add(process)
    try:
        return _wait(process, news_out, stop_in, progress, should_cancel, start_timeout_s, partial)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=CANCEL_GRACE_S)
        with _LIVE_LOCK:
            _LIVE.discard(process)
        news_out.close()
        stop_in.close()


def _wait[T](
    process: BaseProcess,
    news: _Channel,
    stop: _Channel,
    progress: Callable[[float], None] | None,
    should_cancel: Callable[[], bool] | None,
    start_timeout_s: float = START_TIMEOUT_S,
    partial: Callable[[Any], None] | None = None,
) -> T | None:
    started = False
    stopping_since: float | None = None
    began = time.monotonic()
    while True:
        if not started and stopping_since is None and time.monotonic() - began >= start_timeout_s:
            process.terminate()
            raise WorkerUnavailableError(
                f"別のプロセスが {start_timeout_s:g} 秒たっても起きなかった"
            )
        if stopping_since is None and should_cancel is not None and should_cancel():
            stopping_since = time.monotonic()
            # 子がもう終わっていれば送れない 下の読みで終わりを拾う
            with contextlib.suppress(OSError):
                stop.send(("cancel",))
        if stopping_since is not None and time.monotonic() - stopping_since > CANCEL_GRACE_S:
            # 自分で終わらない子は止める 書きかけは呼ぶ側が消す
            process.terminate()
            return None
        try:
            ready = news.poll(POLL_S)
        except (EOFError, OSError):
            ready = False
        if not ready:
            if not process.is_alive():
                # 知らせが残っていないか最後に 1 度見る 終わる直前に送った「できた」を落とさない
                try:
                    if news.poll(0):
                        continue
                except (EOFError, OSError):
                    pass
                _ended_without_answer(process, started, stopping_since)
                return None
            continue
        try:
            message = news.recv()
        except (EOFError, OSError):
            _ended_without_answer(process, started, stopping_since)
            return None
        kind = message[0]
        if kind == "ready":
            started = True
        elif kind == "progress":
            if progress is not None:
                progress(float(message[1]))
        elif kind == "partial":
            # やめると言った後に届いた途中の値は渡さない 呼ぶ側はもう結果を捨てている
            if partial is not None and stopping_since is None:
                partial(message[1])
        elif kind == "done":
            return None if stopping_since is not None else message[1]
        elif kind == "failed":
            raise WorkerFailedError(str(message[1]))


def _ended_without_answer(
    process: BaseProcess, started: bool, stopping_since: float | None
) -> None:
    """子が答えを返さずに終わった やめさせたのなら ``None`` 違えば理由を投げる"""
    process.join(timeout=CANCEL_GRACE_S)
    if stopping_since is not None:
        return None
    code = process.exitcode
    if not started:
        raise WorkerUnavailableError(f"別のプロセスが起動の途中で終わった（終了コード {code}）")
    raise WorkerCrashedError(f"別のプロセスが途中で終わった（終了コード {code}）")


def _child_main(
    news: _Channel,
    stop: _Channel,
    job: Callable[..., object],
    kwargs: dict[str, Any],
) -> None:
    """子のプロセスの入口 ``job`` を走らせ、結果を親へ送る"""
    cancelled = threading.Event()

    def listen() -> None:
        # 親が落ちると管の向こうが閉じて EOF になる そこでやめる
        try:
            while True:
                if stop.recv()[0] == "cancel":
                    cancelled.set()
                    return
        except (EOFError, OSError):
            cancelled.set()

    threading.Thread(target=listen, name="sashimono-worker-stop", daemon=True).start()
    send_lock = threading.Lock()

    def send(message: tuple[object, ...]) -> None:
        with send_lock:
            try:
                news.send(message)
            except OSError:
                # 親がいない 送り先が無いので、やめる印だけ立てる
                cancelled.set()

    sent_at = 0.0

    def report(value: float) -> None:
        nonlocal sent_at
        now = time.monotonic()
        if value < 1.0 and now - sent_at < PROGRESS_INTERVAL_S:
            return
        sent_at = now
        send(("progress", value))

    extra: dict[str, Any] = {}
    if kwargs.pop(_PUBLISH, False):
        extra["publish"] = lambda value: send(("partial", value))
    send(("ready",))
    try:
        result = job(progress=report, should_cancel=cancelled.is_set, **kwargs, **extra)
    except BaseException as exc:  # 子の例外は親へ文で渡す 例外そのものは pickle できないことがある
        send(("failed", f"{type(exc).__name__}: {exc}"))
        return
    send(("done", result))


def stop_all() -> None:
    """起こしている子をすべて止める 窓を閉じるときに呼ぶ

    普通は呼ぶ側の取り消しで子は自分で終わる ここは取り消しが届かない所（呼ぶ側の
    スレッドが待っている途中でアプリが閉じる）の後始末
    """
    with _LIVE_LOCK:
        processes = list(_LIVE)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=CANCEL_GRACE_S)


def probe_job(
    *,
    progress: Callable[[float], None],
    should_cancel: Callable[[], bool],
    value: str = "ok",
    steps: int = 3,
    fail: bool = False,
    wait_s: float = 0.0,
    crash: bool = False,
    publish: Callable[[object], None] | None = None,
) -> str | None:
    """子が起きて仕事を受け、知らせを返せるかを確かめる小さな仕事

    配る zip の自己診断（:mod:`sashimono.selfcheck`）が、配布版の exe で子を起こせるかを
    見るのに使う 控えのような本物の仕事は素材が要り、自己診断の機械では作れない
    ``wait_s`` の間はやめると言われるまで待つ（取り消しの確かめ用）
    ``crash`` は例外を通さずにプロセスごと終わる（素材のせいで FFmpeg が落ちたときの真似）
    """
    if crash:
        # 起きた後に落ちたことを親が見分けられるか確かめるため、「起きた」の後で終わる
        os._exit(3)
    for step in range(1, steps + 1):
        progress(step / steps)
        if publish is not None:
            # 途中の値の口（``partial``）を確かめる 進み具合と同じ数だけ送る
            publish(f"{value} {step}")
    if fail:
        raise RuntimeError("頼まれた失敗")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if should_cancel():
            return None
        time.sleep(0.01)
    return value
