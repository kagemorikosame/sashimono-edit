"""文字の影は影の周りだけでぼかす（#312 の続き）

前は画面に収まる字幕の影を画面 1 枚ぶんの面でぼかしていて、大きさが動いて毎コマ描き直す
字幕 20 本で 1 コマ約 2.7 秒かかった 影の周りだけにしても絵は 1 画素も変わらない（周りの外は
0 なので、ぼかしの累積和は 0 を足すだけ）
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QImage

from sashimono.core.model import GeneratedSource
from sashimono.effects.sources import TEXT
from sashimono.engine import sources as engine_sources
from sashimono.engine.sources import render_source


def shadowed(text: str, size: float) -> GeneratedSource:
    return TEXT.create(
        text=text,
        size=size,
        shadow_x=8.0,
        shadow_y=-6.0,
        shadow_blur=5.0,
        shadow_color=(0.0, 0.0, 0.0, 1.0),
    )


def test_the_blur_covers_only_the_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    # 画面 1 枚ぶんをぼかすと、影のある字幕を毎コマ描き直すたびに 1 本 100 ms 余り掛かる
    sizes: list[tuple[int, int]] = []
    blur = engine_sources._blur_alpha

    def measured(image: QImage, radius: float) -> QImage:
        sizes.append((image.width(), image.height()))
        return blur(image, radius)

    monkeypatch.setattr(engine_sources, "_blur_alpha", measured)
    image = render_source(shadowed("字幕", 64.0), 1920, 1080)
    assert image is not None
    ys, xs = np.nonzero(image[..., 3] > 0)
    (width, height) = sizes[0]
    # 描いた物の広がり（字と影）より、ぼかしの広がりの分（2 x 5 + 余り 4）までしか大きくない
    assert width <= int(xs.max() - xs.min()) + 1 + 2 * 14
    assert height <= int(ys.max() - ys.min()) + 1 + 2 * 14


def test_the_shadow_does_not_depend_on_the_picture_size() -> None:
    # 影の濃さが絵の大きさや置き場所で変わると、同じ字幕が画面ごとに違って見える
    source = shadowed("影の字", 80.0)
    large = render_source(source, 1920, 1080)
    small = render_source(source, 800, 450)
    assert large is not None and small is not None
    dx, dy = (1920 - 800) // 2, (1080 - 450) // 2
    assert np.array_equal(large[dy : dy + 450, dx : dx + 800], small)
