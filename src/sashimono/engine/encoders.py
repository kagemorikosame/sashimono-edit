"""映像のエンコーダの選び方 書き出し（:mod:`sashimono.engine.encode`）と控えの作成
（:mod:`sashimono.engine.cache.proxy`）が同じ物を使う

ここを分けたのは、控えの側が書き出しの側を読み込めないため（書き出しは描画を読み、描画は控えを読む）
選び方を 2 か所に書くと、片方だけ AMD の GPU を知らない、のような食い違いが起きる
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

import av
import av.video.codeccontext
import av.video.frame

from sashimono.engine.worker import WorkerError, WorkerUnavailableError, run_in_process

__all__ = [
    "AMF_PEAK_RATIO",
    "AMF_QUALITY",
    "ENCODER_PRIORITIES",
    "PRIORITY_AMD",
    "PRIORITY_AUTO",
    "PRIORITY_CPU",
    "PRIORITY_INTEL",
    "PROBE_BIT_RATE",
    "PROBE_TIMEOUT_S",
    "MonotonicProgress",
    "ProbeMemory",
    "encoder_options",
    "encoder_vendor",
    "forget_probes",
    "order_by_priority",
    "probe_and_remember",
    "probe_encoder",
    "probe_in_children",
    "probe_troubles",
    "remember",
    "remembered",
    "trouble_notice",
    "usable",
    "usable_encoders",
    "usable_encoders_now",
    "verdict",
]

_log = logging.getLogger(__name__)

#: 優先の選び方 既定（自動）は渡された並びのまま（NVIDIA → AMD → Intel → CPU）
PRIORITY_AUTO = "auto"
#: AMD の GPU（AMF）を先に NVIDIA と AMD を両方挿した機械で、AMD の側を使いたい人向け
PRIORITY_AMD = "amd"
#: Intel の GPU（QSV）を先に
PRIORITY_INTEL = "intel"
#: CPU（libx264）を先に 同じビットレートなら GPU のエンコーダより画質が良いことが多い
PRIORITY_CPU = "cpu"
ENCODER_PRIORITIES = (PRIORITY_AUTO, PRIORITY_AMD, PRIORITY_INTEL, PRIORITY_CPU)

#: 名前の末尾と、作り手の対応 FFmpeg の GPU のエンコーダは末尾で作り手が分かる
_VENDOR_SUFFIXES = (("_nvenc", "nvidia"), ("_amf", PRIORITY_AMD), ("_qsv", PRIORITY_INTEL))

#: 開けるかを試すときの大きさ NVENC は小さすぎる画を断る（64x64 では開けない）ので、
#: 書き出しでよく使う大きさに近い 16:9 で試す 小さくすると、使える NVENC まで外れる
PROBE_SIZE = (640, 360)
#: 試すときのビットレート（bps） 書き出しの既定（12 Mbps）と同じにする AMF は上限の
#: ビットレートをこれから組むので、書き出しと同じ選択肢で開けるかを試せる
PROBE_BIT_RATE = 12_000_000

#: AMF の品質の選び（FFmpeg の amfenc の ``quality`` speed / balanced / quality / high_quality）
#: 2026-10-20 に利用者が決めた 省くと AMF の駆動の既定任せになり、機械ごとに画が変わる
#: AMD の実機で確かめていない 外れたら直す（#365）
AMF_QUALITY = "balanced"
#: AMF の上限のビットレート（FFmpeg の汎用の ``maxrate``）を、指定のビットレートの何倍にするか
#: 2026-10-20 に利用者が決めた 省くと amfenc は peak constrained VBR を上限なしで選び、
#: 駆動の既定の上限になる 指定と同じにすると CBR に切り替わる（amfenc_h264.c の選び方）
#: NVENC の VBR に近い振る舞いを狙う AMD の実機で確かめていない 外れたら直す（#365）
AMF_PEAK_RATIO = (3, 2)


class MonotonicProgress:
    """次の候補のエンコーダで書き直しても、進み具合を後ろへ戻さない

    書き直しは 1 コマ目から 0..1 を送り直す そのまま流すと、書き出しの窓のバーや AI の
    export_status、素材の一覧のプロキシの進み具合が 80% から 1% へ下がり、壊れたのか
    最初に戻ったのか分からない 出した値で止めておく形は選ばない 書き直しの間ずっと動かず、
    固まったように見える 書き直しの進み具合を、出した値から 1 までの残りへ割り当てて、
    動きながら戻らない形にする 何で書き直したかは書き上げた後に知らせる
    """

    def __init__(self, forward: Callable[[float], None]) -> None:
        self._forward = forward
        #: 今の試みの 0 を、どの値に置くか 書き直すたびに、それまでに出した値へ上げる
        self._base = 0.0
        self._last = 0.0

    def report(self, value: float) -> None:
        """今の試みの進み具合（0..1）を受けて、戻らない値にして渡す"""
        shown = self._base + (1.0 - self._base) * value
        # 割り当てても丸めで前より小さくなることはあり得る 戻る値は出さない
        if shown < self._last:
            return
        self._last = shown
        self._forward(shown)

    def start_again(self) -> None:
        """次の候補で頭から書き直す"""
        self._base = self._last


def encoder_options(name: str, bit_rate: int | None) -> dict[str, str]:
    """``name`` のエンコーダへ既定で渡す選択肢 AMF のほかは空

    名前と値は FFmpeg の文書と amfenc_h264.c / amfenc_hevc.c の AVOption に合わせた
    ``maxrate`` は全エンコーダに共通の選択肢（``rc_max_rate``）で、AMF は peak bitrate として読む
    """
    if encoder_vendor(name) != PRIORITY_AMD:
        return {}
    options = {"quality": AMF_QUALITY}
    if bit_rate:
        numerator, denominator = AMF_PEAK_RATIO
        options["maxrate"] = str(bit_rate * numerator // denominator)
    return options


def encoder_vendor(name: str) -> str:
    """``name`` のエンコーダを動かす物 GPU の作り手か :data:`PRIORITY_CPU`"""
    for suffix, vendor in _VENDOR_SUFFIXES:
        if name.endswith(suffix):
            return vendor
    return PRIORITY_CPU


def order_by_priority(names: list[str], priority: str) -> list[str]:
    """``names`` を ``priority`` の作り手が先に来るよう並べ替える 同じ側どうしの順は保つ

    知らない ``priority`` は ``ValueError`` 黙って自動に戻すと、選んだ物が効かない理由が分からない
    """
    if priority not in ENCODER_PRIORITIES:
        raise ValueError(f"エンコーダの優先は {', '.join(ENCODER_PRIORITIES)} のどれか: {priority}")
    if priority == PRIORITY_AUTO:
        return list(names)
    # sorted は安定なので、先に出す側どうし・後ろの側どうしの順は元のまま
    return sorted(names, key=lambda name: encoder_vendor(name) != priority)


def probe_encoder(name: str) -> None:
    """``name`` のエンコーダを yuv420p で開き、1 コマ書いてみる 書けなければ投げる

    開けるだけでは足りない GPU のエンコーダは、駆動の版や GPU の種類によって、開けても
    最初のコマで断ることがある（開くのは FFmpeg の側、書くのは駆動の側の仕事のため）
    開けただけで選ぶと、そういう機械では書き出しの途中で失敗する 1 コマ書ければ、
    書き出しや控えの作成でも同じ道を通る

    画素形式は書き出しの既定と同じ yuv420p に固定する 答えはコーデックごとに 1 つだけ覚えるので、
    書き出しごとの設定には追従できない 別の画素形式で開けないときは、コーデックを指定しない
    書き出しなら始めた時点で次の候補へ移る
    """
    context = _open_context(name)
    width, height = PROBE_SIZE
    frame = av.video.frame.VideoFrame(width, height, "yuv420p")
    frame.pts = 0
    # 1 コマ目では何も出さず貯めるエンコーダが多い 最後まで流し切って、1 つでも出たかを見る
    packets = [*context.encode(frame), *context.encode(None)]
    if not packets:
        raise RuntimeError(f"{name} は開けたが 1 コマも書けなかった")
    # 開いた文脈は捨てるだけでよい PyAV は参照が切れたときに閉じる


def _open_context(name: str) -> av.video.codeccontext.VideoCodecContext:
    """試しに書くための文脈を開く 開けなければ投げる

    ビットレートと既定の選択肢（:func:`encoder_options`）は書き出しと同じ組み方で渡す
    渡さずに試すと、AMF の駆動が書き出しの選択肢だけを断る機械で、試しには通って
    書き出しで落ちる
    """
    context = _new_context(name)
    context.width, context.height = PROBE_SIZE
    context.pix_fmt = "yuv420p"
    context.time_base = Fraction(1, 30)
    context.bit_rate = PROBE_BIT_RATE
    options = encoder_options(name, PROBE_BIT_RATE)
    if options:
        context.options = options
    context.open()
    return context


def _new_context(name: str) -> av.video.codeccontext.VideoCodecContext:
    """まだ開いていない書き込みの文脈 試験が偽のエンコーダに差し替える所"""
    # create は種類の union を返す 映像の属性を触るので、ここで型を確定させる
    return cast("av.video.codeccontext.VideoCodecContext", av.CodecContext.create(name, "w"))


#: 1 つのエンコーダの確かめを待つ長さ（秒） 子が起きた後、そのエンコーダを試し始めてから数える
#: NVENC を開くのは普通 0.1 秒から 0.4 秒 駆動を初めて読む冷えた機械でも数秒で済む
#: 壊れた駆動は落ちずに固まることもあり、待ち続けると書き出しの窓が開かないまま止まる
PROBE_TIMEOUT_S = 15.0

#: 確かめの答え 名前ごとに書けるか 起動中は覚え、:func:`forget_probes` で忘れる
_verdicts: dict[str, bool] = {}
#: 子ごと落ちたか固まったエンコーダと、その理由 利用者に知らせる文の元
_troubles: dict[str, str] = {}
#: 確かめは 1 度に 1 組だけ 書き出しの窓と控えの係が同時に問うても、子を 2 つ起こさない
_probe_lock = threading.Lock()


def usable(name: str) -> bool:
    """``name`` で書けるか 1 つにつき 1 度だけ試して覚える（:func:`usable_encoders`）"""
    return bool(usable_encoders([name]))


def usable_encoders(names: Iterable[str]) -> list[str]:
    """``names`` のうち書ける物を、渡された順に返す まだ試していない物だけを試して覚える

    試すのは別のプロセス（#464） 駆動が壊れた機械では、NVENC を開くだけで例外ではなく
    access violation でプロセスごと落ちる アプリの中で試すと、書き出しを押しただけで
    保存していない編集ごと消える 子が落ちたか固まったら、そのエンコーダを使えない物にして
    次の候補へ進む 理由は :func:`probe_troubles` で拾える

    NVENC を開くのは 1 回 100 ms ほど、子を起こすのは 200 ms ほどかかる 書き出しの窓を開くたび・
    控えを作るたびに払わないよう、答えは起動中ずっと覚える 機械の GPU は起動中に変わらない
    駆動が戻ったとき（PC を再起動せずに入れ直したなど）は :func:`forget_probes` で忘れる
    """
    with _probe_lock:
        return _usable_locked(list(dict.fromkeys(names)))


def usable_encoders_now(names: Iterable[str]) -> list[str] | None:
    """覚えた答えだけで足りれば :func:`usable_encoders` と同じ 足りなければ待たずに ``None``

    書き出しの窓は画面のスレッドで候補を並べる ほかの確かめが走っている所で待つか、
    ここで子を起こして試すと、固まった駆動を待ち切るまで（候補ごとに 15 秒）窓が開かない
    ``None`` なら呼ぶ側は裏で :func:`usable_encoders` を呼んで、終わってから並べる
    """
    if not _probe_lock.acquire(blocking=False):
        return None
    try:
        wanted = list(dict.fromkeys(names))
        if any(name not in _verdicts for name in wanted):
            return None
        return [name for name in wanted if _verdicts[name]]
    finally:
        _probe_lock.release()


def probe_and_remember(names: Iterable[str]) -> ProbeMemory:
    """``names`` を確かめてから、覚えている答えの写しを返す 確かめと写しは同じ錠の中で

    書き出しの子へ渡す 確かめてから写すまでの間に確かめ直し（:func:`forget_probes`）が
    割り込むと、空の写しが子に渡り、子が自分の中で GPU を試して落ちる
    （PR #492 の CodeRabbit の指摘）
    """
    with _probe_lock:
        _usable_locked(list(dict.fromkeys(names)))
        return ProbeMemory(tuple(_verdicts.items()), tuple(_troubles.items()))


def verdict(name: str) -> tuple[bool, str | None]:
    """``name`` で書けるかと、確かめで落ちたか固まったならその理由 同じ時点の答えを返す

    書けるかと理由を別々に問うと、その間に確かめ直し（:func:`forget_probes`）が割り込んで
    理由だけ消え、落ちた GPU の物をこのプロセスで開きにいく（PR #492 の CodeRabbit の指摘）
    """
    with _probe_lock:
        ok = bool(_usable_locked([name]))
        return ok, _troubles.get(name)


def probe_in_children(names: Iterable[str]) -> None:
    """``names`` を子のプロセスだけで確かめて覚える 子を起こせなければ何もしない

    起動の後に裏で確かめておくときに使う 子を起こせない機械でこのプロセスへ戻すと、
    書き出しを押してもいないのに、壊れた駆動で編集画面ごと落ちる 書き出しと控えの作成は
    :func:`usable_encoders`（子を起こせなければこの中で試す）を通るので、そちらに任せる
    """
    with _probe_lock:
        unknown = [name for name in dict.fromkeys(names) if name not in _verdicts]
        if not unknown:
            return
        try:
            _verdicts.update(_probe_in_child(unknown))
        except WorkerUnavailableError as exc:
            _log.warning("起動の後のエンコーダの確かめで別のプロセスを起こせなかった: %s", exc)


def _usable_locked(wanted: list[str]) -> list[str]:
    """錠を取った後の :func:`usable_encoders` 知らない物を試して覚え、書ける物を返す"""
    unknown = [name for name in wanted if name not in _verdicts]
    if unknown:
        _verdicts.update(_probe(unknown))
    return [name for name in wanted if _verdicts.get(name, False)]


def probe_troubles() -> dict[str, str]:
    """確かめで子ごと落ちたか固まったエンコーダと、その理由

    入っていないだけ・開けないだけ（AMD の無い機械の AMF）の物は含めない それは普通の機械の
    姿で、知らせると書き出しのたびに要らない注意が出る
    """
    with _probe_lock:
        return dict(_troubles)


def trouble_notice(names: Iterable[str]) -> str | None:
    """``names`` のうち確かめで落ちたか固まった物があれば、利用者への知らせの文 無ければ ``None``

    書き出しと控えで同じ文にする 片方だけ「再起動で直る」を書き忘れると、利用者は
    GPU が壊れたのか設定の誤りなのか分からない
    """
    troubles = probe_troubles()
    hit = [name for name in dict.fromkeys(names) if name in troubles]
    if not hit:
        return None
    reasons = " / ".join(f"{name} は{troubles[name]}" for name in hit)
    return (
        f"GPU のエンコーダが使えなかったので、ほかのエンコーダで書いた（{reasons}）"
        " GPU の駆動の不調のことが多く、PC を再起動すると直ることがある"
        " 直ったら 表示 → 設定… の「エンコーダを確かめ直す」で GPU へ戻せる"
    )


@dataclass(frozen=True, slots=True)
class ProbeMemory:
    """覚えた確かめの答えの写し 書き出しの子へ渡す

    書き出しは別のプロセスで走り（:mod:`sashimono.engine.encode.export_job`）、子は spawn で
    起きるので覚えた答えを持たない 渡さないと、子が書き出しのたびに孫を起こして確かめ直し、
    始まりが 0.5 秒ほど遅れる 落ちた物の理由も子に無く、知らせから「再起動で直る」が消える
    """

    verdicts: tuple[tuple[str, bool], ...] = ()
    troubles: tuple[tuple[str, str], ...] = ()


def remembered() -> ProbeMemory:
    """このプロセスで覚えている答えの写し"""
    with _probe_lock:
        return ProbeMemory(tuple(_verdicts.items()), tuple(_troubles.items()))


def remember(memory: ProbeMemory) -> None:
    """別のプロセスで覚えた答えを、このプロセスへ足す 知らない名前は次に問われたときに確かめる"""
    with _probe_lock:
        _verdicts.update(memory.verdicts)
        _troubles.update(memory.troubles)


def forget_probes() -> None:
    """覚えている確かめの答えを捨てる 次に問われたときに確かめ直す

    設定画面の「エンコーダを確かめ直す」から呼ぶ 駆動を入れ直して直ったのに、
    アプリを閉じるまで CPU で書き出し続けるのを避ける
    """
    with _probe_lock:
        _verdicts.clear()
        _troubles.clear()


def _probe(names: list[str]) -> dict[str, bool]:
    """``names`` を確かめる 別のプロセスで 起こせない機械ではこのプロセスで

    子を起こせない（見張りのソフトが exe を止めた・管を作れない）機械で GPU を
    すべて使えない物にすると、駆動の壊れていない普通の機械まで CPU の書き出しに落ちる
    この道は #464 の前の作りと同じで、子を起こせないうえに駆動も壊れている機械は少ない
    """
    try:
        return _probe_in_child(names)
    except WorkerUnavailableError as exc:
        _log.warning("エンコーダの確かめで別のプロセスを起こせず、この中で確かめる: %s", exc)
    return {name: _probe_here(name) for name in names}


def _probe_in_child(names: list[str]) -> dict[str, bool]:
    """``names`` を子のプロセスで順に確かめる 子が落ちたら、落とした物を外して残りを次の子で

    1 つずつ子を起こすと、候補 4 つで 1 秒近くかかる 1 つの子で順に試し、試し始める前に
    名前を送らせておけば、落ちたときにどれが落としたかが分かる
    最初の子を起こせないときは :class:`WorkerUnavailableError` を投げる（呼ぶ側がこの中で試す）
    """
    verdicts: dict[str, bool] = {}
    remaining = list(names)
    while remaining:
        trouble, culprits = _probe_batch(remaining, verdicts)
        if trouble is not None:
            # 何を試していたか分からないまま落ちた（試す前・答えた直後）ときは、
            # 残りの先頭のせいにする そうしないと、同じ所で落ちる子を起こし続ける
            left = [name for name in remaining if name not in verdicts]
            for culprit in culprits or left[:1]:
                verdicts[culprit] = False
                _troubles[culprit] = trouble
                _log.warning("エンコーダ %s の確かめで%s", culprit, trouble)
        remaining = [name for name in remaining if name not in verdicts]
    return verdicts


def _probe_batch(names: list[str], verdicts: dict[str, bool]) -> tuple[str | None, list[str]]:
    """子を 1 つ起こして ``names`` を順に試させ、答えを ``verdicts`` へ足す

    子が落ちたか固まったら、その理由と、使えない物にする名前（分からなければ空）を返す
    無事に終われば ``(None, [])``
    """
    current: str | None = None
    began = 0.0

    def heard(message: Any) -> None:
        nonlocal current, began
        if message[0] == "begin":
            current = str(message[1])
            began = time.monotonic()
        else:
            verdicts[str(message[1])] = bool(message[2])
            current = None

    def overdue() -> bool:
        return current is not None and time.monotonic() - began > PROBE_TIMEOUT_S

    try:
        run_in_process(
            _probe_job,
            {"names": names},
            should_cancel=overdue,
            partial=heard,
            name="sashimono-encoder-probe",
        )
    except WorkerUnavailableError:
        if not verdicts and not _troubles:
            raise
        # 1 度は起こせたのに、次の子を起こせなかった 残りの GPU の物はまとめて使えない物にする
        # この中で試し直すと、さっき子を落とした駆動を読んで今度はアプリごと落ちうる
        # 1 つずつ起こし直すと、起きない子を 1 つにつき 60 秒待つ
        # CPU の物は駆動を読まないのでこの中で試す 外すと書き出せる物が 1 つも残らない
        left = [name for name in names if name not in verdicts]
        for name in left:
            if encoder_vendor(name) == PRIORITY_CPU:
                verdicts[name] = _probe_here(name)
        gpu = [name for name in left if name not in verdicts]
        return "落ちた後の確かめ直しで別のプロセスを起こせなかった", gpu
    except WorkerError as exc:
        return f"確かめの途中でプロセスごと落ちた（{exc}）", _named(current)
    if current is not None:
        # やめさせて返った（:func:`overdue`） 試していた物が固まった
        return f"{PROBE_TIMEOUT_S:g} 秒たっても確かめが終わらなかった", [current]
    return None, []


def _named(name: str | None) -> list[str]:
    return [] if name is None else [name]


def _probe_job(
    *,
    progress: Callable[[float], None],
    should_cancel: Callable[[], bool],
    publish: Callable[[object], None],
    names: list[str],
) -> None:
    """子の側 ``names`` を順に試し、試し始める前と答えを親へ送る

    答えをまとめて最後に返す形にはしない 途中で落ちると、それまでの答えも
    どれが落としたかも親に届かない
    """
    del progress  # 進み具合は送らない 1 つずつの答えが進み具合の代わり
    for name in names:
        if should_cancel():
            return
        publish(("begin", name))
        publish(("verdict", name, _probe_here(name)))


def _probe_here(name: str) -> bool:
    """このプロセスで ``name`` を試す 書ければ真"""
    try:
        probe_encoder(name)
    except Exception:
        # 入っていない（UnknownCodecError）・開けない（AMD の GPU が無い機械の AMF は
        # UnknownError、Intel の無い機械の QSV は ArgumentError）と、種類は環境によって違う
        # 名前で拾うと、知らない種類の失敗で候補の列挙ごと落ちる
        return False
    return True
