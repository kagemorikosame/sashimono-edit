"""映像のエンコーダの選び方 書き出し（:mod:`sashimono.engine.encode`）と控えの作成
（:mod:`sashimono.engine.cache.proxy`）が同じ物を使う

ここを分けたのは、控えの側が書き出しの側を読み込めないため（書き出しは描画を読み、描画は控えを読む）
選び方を 2 か所に書くと、片方だけ AMD の GPU を知らない、のような食い違いが起きる
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from typing import cast

import av
import av.video.codeccontext
import av.video.frame

__all__ = [
    "ENCODER_PRIORITIES",
    "PRIORITY_AMD",
    "PRIORITY_AUTO",
    "PRIORITY_CPU",
    "PRIORITY_INTEL",
    "encoder_vendor",
    "order_by_priority",
    "probe_encoder",
    "usable",
]

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
    """試しに書くための文脈を開く 開けなければ投げる"""
    # create は種類の union を返す 映像の属性を触るので、ここで型を確定させる
    context = cast("av.video.codeccontext.VideoCodecContext", av.CodecContext.create(name, "w"))
    context.width, context.height = PROBE_SIZE
    context.pix_fmt = "yuv420p"
    context.time_base = Fraction(1, 30)
    context.open()
    return context


@cache
def usable(name: str) -> bool:
    """``name`` で書けるか 1 つにつき 1 度だけ試して覚える

    NVENC を開くのは 1 回 100 ms ほどかかる 控えを作るたびに試すと、そのたびに待たされる
    機械の GPU は起動中に変わらないので、覚えておいてよい
    """
    try:
        probe_encoder(name)
    except Exception:
        # 入っていない（UnknownCodecError）・開けない（AMD の GPU が無い機械の AMF は
        # UnknownError、Intel の無い機械の QSV は ArgumentError）と、種類は環境によって違う
        # 名前で拾うと、知らない種類の失敗で候補の列挙ごと落ちる
        return False
    return True
