"""テキストの折り返しの幅を、字の実寸で描く所に効かせる（#249）

位置の決まり（禁則・英単語）は test_text_wrap が書体抜きで見る ここは描いた絵で、
幅に収まること・0 なら今と同じこと・縦書き・AviUtl2 の組み方・プレビューと書き出しを見る
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.core.model import AnimatedValue, GeneratedSource, ParamValue
from sashimono.effects.sources import AVIUTL_NO_WRAP, TEXT
from sashimono.engine.sources import render_source, render_source_framed, source_canvas

SCREEN = (1920, 1080)
LONG = "今日はとてもいい天気なので、みんなで近くの公園まで歩いて出かけることにしました"


def text(**params: ParamValue | float | str | bool) -> GeneratedSource:
    base: dict[str, ParamValue] = {"text": LONG, "size": AnimatedValue(64.0)}
    for name, value in params.items():
        base[name] = (
            AnimatedValue(float(value))
            if isinstance(value, int | float) and not isinstance(value, bool)
            else value
        )
    return GeneratedSource(kind="text", params=base)


def ink(image: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(image[..., 3] > 0)
    assert len(xs), "何も描かれていない"
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def drawn(source: GeneratedSource, scale: float = 1.0) -> np.ndarray:
    width, height = round(SCREEN[0] * scale), round(SCREEN[1] * scale)
    canvas = source_canvas(source, width, height, scale=(scale, scale))
    image = render_source_framed(source, *canvas, scale=(scale, scale), screen=(width, height))[0]
    assert image is not None
    return image


class TestSpec:
    def test_the_width_is_the_last_item_and_off_by_default(self) -> None:
        # 既定 0 は折り返さない 古いファイルは項目が無く、既定で開くので見た目が変わらない
        assert TEXT.parameters[-1].name == "wrap_width"
        width = TEXT.create().params["wrap_width"]
        assert isinstance(width, AnimatedValue) and width.static == 0

    def test_the_aviutl_layout_locks_the_width(self) -> None:
        # AviUtl2 のテキストに折り返しは無い 欄は灰色にして理由を出す
        assert TEXT.locked_reasons(TEXT.create(layout="aviutl").params) == {
            "wrap_width": AVIUTL_NO_WRAP
        }
        assert TEXT.locked_reasons(TEXT.create().params) == {}
        # 縦書きは AviUtl2 の組み方でも標準の縦書きで描くので折り返せる
        assert TEXT.locked_reasons(TEXT.create(layout="aviutl", vertical=True).params) == {}


class TestDrawing:
    def test_zero_keeps_the_old_picture(self) -> None:
        # 0 で絵が変わると、既にある作品の字幕が全部動く
        plain = render_source(text(), *SCREEN)
        zero = render_source(text(wrap_width=0), *SCREEN)
        assert plain is not None and zero is not None
        np.testing.assert_array_equal(plain, zero)

    def test_a_long_line_fits_the_width(self) -> None:
        # 1 行では画面の幅を超える文が、800 の幅に収まって何行にもなる
        single = ink(drawn(text()))
        wrapped = ink(drawn(text(wrap_width=800)))
        assert single[2] - single[0] > 1600
        assert wrapped[2] - wrapped[0] <= 800
        assert wrapped[3] - wrapped[1] > (single[3] - single[1]) * 2.5

    def test_the_wrapped_text_stays_centred(self) -> None:
        # 行揃えと基準はそのまま 中央揃えなら折り返した塊の真ん中が位置に来る
        left, top, right, bottom = ink(drawn(text(wrap_width=800)))
        assert abs((left + right) / 2 - SCREEN[0] / 2) < 40
        assert abs((top + bottom) / 2 - SCREEN[1] / 2) < 40

    def test_a_bigger_font_wraps_again(self) -> None:
        # 文字数ではなく字の実寸で折り返す 大きさを変えたら折り返し直す
        small = ink(drawn(text(wrap_width=800)))
        big = ink(drawn(text(wrap_width=800, size=96)))
        assert big[2] - big[0] <= 800
        assert (big[3] - big[1]) > (small[3] - small[1]) * 1.6

    def test_vertical_text_wraps_by_height(self) -> None:
        # 縦書きは列の高さで折り返す 幅で数えると、縦に画面を突き抜ける
        single = ink(drawn(text(vertical=True)))
        wrapped = ink(drawn(text(vertical=True, wrap_width=600)))
        assert single[3] - single[1] > 1600
        assert wrapped[3] - wrapped[1] <= 600
        assert wrapped[2] - wrapped[0] > (single[2] - single[0]) * 2.5

    def test_the_aviutl_layout_does_not_wrap(self) -> None:
        # 合わせる相手（AviUtl2）に無い動きは足さない 読み込んだ作品の行が変わる
        plain = render_source(text(layout="aviutl"), *SCREEN)
        wrapped = render_source(text(layout="aviutl", wrap_width=800), *SCREEN)
        assert plain is not None and wrapped is not None
        np.testing.assert_array_equal(plain, wrapped)

    @pytest.mark.parametrize("divisor", [2, 4])
    def test_the_preview_breaks_at_the_same_places(self, divisor: int) -> None:
        # 画質を落としたプレビューでも同じ所で折り返す 幅は画面の画素で数える
        full = ink(drawn(text(wrap_width=800)))
        light = ink(drawn(text(wrap_width=800), scale=1 / divisor))
        for edge in range(4):
            assert abs(light[edge] * divisor - full[edge]) <= divisor * 3

    def test_revealing_does_not_move_a_word(self) -> None:
        # 文字送りの途中で英単語が行をまたいで飛ばない 全部を出した文で折り返してから送る
        source: dict[str, ParamValue | float | str | bool] = {
            "text": "aaaa bbbbbbbb",
            "wrap_width": 400,
            "align": "left",
            "anchor": "left",
        }
        full = ink(drawn(text(**source)))
        half = ink(drawn(text(**source, reveal=60)))
        # 6 割で出ているのは 1 行目と 2 行目の頭 2 行目が出ていれば高さは全部と同じ
        assert half[3] - half[1] == pytest.approx(full[3] - full[1], abs=2)
        assert half[0] == full[0]

    def test_manual_breaks_stay(self) -> None:
        # 手で入れた改行は残し、その間の行だけを幅で折り返す
        two = ink(drawn(text(text="あいう\nえお", wrap_width=800)))
        none = ink(drawn(text(text="あいう\nえお")))
        assert two == none

    def test_the_wrapped_text_needs_no_wider_picture(self) -> None:
        # 折り返して画面に収まる字は、画面の大きさの絵で描く（#256 の広げる道を通らない）
        assert source_canvas(text(wrap_width=1600), *SCREEN) == SCREEN
        assert source_canvas(text(), *SCREEN)[0] > SCREEN[0]
