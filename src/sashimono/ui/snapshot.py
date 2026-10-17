"""再生位置の静止画（スクリーンショット）を作る

プレビューの GL の絵をそのまま読まない プレビューは窓の大きさに合わせた表示倍率と、
再生品質（1/2・1/4）と控え（プロキシ）で描いているので、読んだ絵は書き出しと
大きさも細かさも違う ここでは書き出しと同じく、控えを使わない描画係で
プロジェクトの解像度のまま 1 枚描く
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QSaveFile
from PySide6.QtGui import QImage, QImageWriter

from sashimono.core.model import Project
from sashimono.core.timebase import format_timecode
from sashimono.engine.render import FrameRenderer
from sashimono.ui.system_clipboard import clipboard

__all__ = [
    "SNAPSHOT_FILTER",
    "copy_to_clipboard",
    "encode_png",
    "render_snapshot",
    "snapshot_frame",
    "snapshot_name",
    "write_png",
]

#: 保存の窓の絞り込み
SNAPSHOT_FILTER = "PNG 画像 (*.png)"

#: Windows のファイル名に使えない文字 タイムコードの ``:`` と ``;`` もここに入る
_UNSAFE = re.compile(r'[\\/:*?"<>|;\x00-\x1f]')


def snapshot_frame(project: Project, frame: int) -> int:
    """静止画にするフレーム 終わり（``duration``）にいるときは最後のフレームにする

    再生ヘッドは最後まで再生すると ``duration``（最後のフレームの 1 つ後ろ）で止まる
    そこを描くと何も無い黒の絵になり、最後の場面を撮ったつもりが撮れていない
    """
    return max(0, min(frame, project.duration - 1))


def snapshot_name(project: Project, frame: int) -> str:
    """保存の既定の名前 ``プロジェクト名_タイムコード.png``

    タイムコードの区切り（``:``）は Windows のファイル名に使えないので ``-`` にする
    プロジェクト名に使えない文字が入っていても同じく置き換える
    """
    timecode = format_timecode(frame, project.rate)
    stem = f"{project.name}_{timecode}"
    return _UNSAFE.sub("-", stem).strip(" .") + ".png"


def render_snapshot(project: Project, frame: int) -> QImage:
    """``frame`` を書き出しと同じ描き方で 1 枚描く 大きさはプロジェクトの解像度

    描画係は毎回作って閉じる 撮るのは時々で、持ち続けると GPU のメモリを
    プロジェクトの解像度 1 枚ぶん（4K で 33MB）握ったままになる
    """
    renderer = FrameRenderer(project)
    try:
        image = renderer.render(frame)
    finally:
        renderer.close()
    return _to_qimage(image)


def _to_qimage(image: np.ndarray) -> QImage:
    """描いた RGBA を画像にする 不透明度は捨てる

    書き出し（:mod:`sashimono.engine.encode`）も RGBA の A を捨てて RGB だけを符号化する
    A を残すと、何も置いていない所が画像を開くソフトで透けて見え、書き出した動画と
    絵が変わる
    """
    height, width = image.shape[0], image.shape[1]
    contiguous = np.ascontiguousarray(image[..., :4], dtype=np.uint8)
    # 元の配列を離れても使えるよう写し取る QImage は渡した bytes を指したままになる
    return QImage(
        contiguous.tobytes(), width, height, width * 4, QImage.Format.Format_RGBX8888
    ).copy()


def encode_png(image: QImage) -> bytes | None:
    """:func:`write_png` と同じ書き方（``QImageWriter`` の PNG）でバイト列にする 失敗なら ``None``

    AI の静止画の保存（#327）が使う 書く所を手元で決めたいので（書いてから置き換える・
    許した後に書く先が変わっていないかを照らす）、ファイルではなくバイト列で受け取る
    """
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not QImageWriter(buffer, b"PNG").write(image):
        return None
    return bytes(buffer.data().data())


def write_png(image: QImage, path: Path) -> bool:
    """PNG で書く 書けなければ偽

    ``QImage.save`` の書式引数は、この PySide6 では str しか受け取らない
    （型情報は bytes だと言う） 食い違いを避けるため ``QImageWriter`` を使う

    ``QSaveFile`` で一時ファイルへ書き、書き終えてから置き換える 前の静止画へ
    上書きするとき、途中で失敗（容量が足りないなど）すると前の絵まで壊れるため
    """
    output = QSaveFile(str(path))
    if not output.open(QIODevice.OpenModeFlag.WriteOnly):
        return False
    if not QImageWriter(output, b"PNG").write(image):
        output.cancelWriting()
        return False
    return output.commit()


def copy_to_clipboard(image: QImage) -> None:
    """クリップボードへ画像として置く 貼り付け先（ペイント・チャットなど）がそのまま受け取る"""
    clipboard().setImage(image)
