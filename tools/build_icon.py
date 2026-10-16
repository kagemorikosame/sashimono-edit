"""ロゴの SVG から、配布用の PNG と Windows の .ico を作る

.venv\\Scripts\\python.exe tools/build_icon.py

SVG が唯一の原本 ここで作るものは全部その派生なので、形を直したいときは
``src/sashimono/resources/icon.svg`` だけを触る
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QImageWriter, QPainter
from PySide6.QtSvg import QSvgRenderer

#: .ico に入れる大きさ Windows は場面ごとに違う大きさを選ぶので、
#: 小さいほうを縮小任せにすると、タスクバーで潰れる
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

ROOT = Path(__file__).resolve().parent.parent
RESOURCES = ROOT / "src" / "sashimono" / "resources"
DOCS = ROOT / "docs"


def render(source: Path, size: int) -> QImage:
    """SVG を指定の大きさで描く"""
    image = QImage(size, size, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    QSvgRenderer(str(source)).render(painter)
    painter.end()
    return image


def as_png(image: QImage) -> bytes:
    """PNG のバイト列にする

    ``QImage.save`` にファイル名ではなくバッファを渡す場合、書式は文字列では
    なくバイト列で指定する（型注釈と実装が食い違っている）
    """
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    QImageWriter(buffer, b"PNG").write(image)
    return bytes(buffer.data().data())


def build_ico(source: Path, target: Path, sizes: tuple[int, ...] = ICO_SIZES) -> None:
    """複数の大きさを収めた .ico を書く

    中身は PNG Vista 以降はこれを読めるし、生のビットマップより小さい
    """
    payloads = [as_png(render(source, size)) for size in sizes]

    # ICONDIR: 予約(0), 種別(1=アイコン), 枚数
    parts = [struct.pack("<HHH", 0, 1, len(payloads))]
    offset = 6 + 16 * len(payloads)
    for size, payload in zip(sizes, payloads, strict=True):
        # 256 は 0 として書く決まり（1 バイトに収まらないため）
        stored = 0 if size >= 256 else size
        parts.append(struct.pack("<BBBBHHII", stored, stored, 0, 0, 1, 32, len(payload), offset))
        offset += len(payload)
    parts.extend(payloads)

    target.write_bytes(b"".join(parts))


def main() -> int:
    QGuiApplication(sys.argv)
    DOCS.mkdir(exist_ok=True)

    icon = RESOURCES / "icon.svg"
    logo = RESOURCES / "logo.svg"

    build_ico(icon, RESOURCES / "sashimono.ico")
    print(f"作成: {RESOURCES / 'sashimono.ico'}（{len(ICO_SIZES)} サイズ）")

    for size in (256, 512):
        render(icon, size).save(str(DOCS / f"icon-{size}.png"))
    render(logo, 512).save(str(DOCS / "logo.png"))
    print(f"作成: {DOCS / 'logo.png'}, icon-256.png, icon-512.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
