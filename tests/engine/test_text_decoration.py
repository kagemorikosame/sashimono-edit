"""文字装飾が実際に絵になるか

対応表を通っただけでは意味がない 影は落ちているか、縁は付いているか、
縦の基準はどちらへ効くのかを、描いた画素で確かめる
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimono.core.model import AnimatedValue, GeneratedSource
from sashimono.engine.sources import render_source

SIZE = (320, 240)


def draw(**params: object) -> np.ndarray:
    base: dict[str, object] = {
        "text": "あ",
        "size": AnimatedValue(96.0),
        "color": (1.0, 1.0, 1.0, 1.0),
    }
    base.update(params)
    image = render_source(GeneratedSource(kind="text", params=base), *SIZE)  # type: ignore[arg-type]
    assert image is not None
    return image


def opaque(image: np.ndarray) -> int:
    """不透明な画素の数"""
    return int((image[:, :, 3] > 8).sum())


def centre_of_mass(image: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(image[:, :, 3] > 8)
    return float(ys.mean()), float(xs.mean())


class TestShadow:
    def test_a_shadow_adds_ink(self) -> None:
        # 影を設定しても何も描かれないと、白い字幕が明るい背景に溶けて読めなくなる
        plain = draw()
        shadowed = draw(
            shadow_x=AnimatedValue(8.0),
            shadow_y=AnimatedValue(-8.0),
            shadow_color=(0.0, 0.0, 0.0, 1.0),
        )
        assert opaque(shadowed) > opaque(plain)

    def test_it_falls_down_and_to_the_right(self) -> None:
        # 設定の Y は上向き 負の値を渡したら画面では下へ行く
        shadowed = draw(
            shadow_x=AnimatedValue(12.0),
            shadow_y=AnimatedValue(-12.0),
            shadow_color=(0.0, 0.0, 0.0, 1.0),
        )
        plain_y, plain_x = centre_of_mass(draw())
        shadow_y, shadow_x = centre_of_mass(shadowed)
        assert shadow_y > plain_y
        assert shadow_x > plain_x

    def test_a_transparent_shadow_changes_nothing(self) -> None:
        # 影の色を透明にしても何か描かれると、影を消したつもりの字幕の周りに跡が残る
        assert opaque(draw(shadow_x=AnimatedValue(8.0), shadow_color=(0, 0, 0, 0.0))) == opaque(
            draw()
        )

    def test_blurring_spreads_it_without_losing_the_letter(self) -> None:
        # 影のぼかしの値を上げても影が広がらないと、柔らかい影が作れずぼかしの欄が効かない
        sharp = draw(
            shadow_x=AnimatedValue(10.0),
            shadow_y=AnimatedValue(-10.0),
            shadow_color=(0.0, 0.0, 0.0, 1.0),
        )
        soft = draw(
            shadow_x=AnimatedValue(10.0),
            shadow_y=AnimatedValue(-10.0),
            shadow_blur=AnimatedValue(6.0),
            shadow_color=(0.0, 0.0, 0.0, 1.0),
        )
        # ぼかすと薄く広がる 触れている画素は増える
        assert int((soft[:, :, 3] > 0).sum()) > int((sharp[:, :, 3] > 0).sum())

    def test_the_shadow_follows_the_outline_not_just_the_fill(self) -> None:
        # 縁取りがあるときは、その外形の影が落ちる 塗りだけの影にすると
        # 縁の分だけ影が細く見える
        with_border = draw(
            border_width=AnimatedValue(6.0),
            border_color=(0.0, 0.0, 1.0, 1.0),
            shadow_x=AnimatedValue(20.0),
            shadow_y=AnimatedValue(-20.0),
            shadow_color=(1.0, 0.0, 0.0, 1.0),
        )
        without_border = draw(
            shadow_x=AnimatedValue(20.0),
            shadow_y=AnimatedValue(-20.0),
            shadow_color=(1.0, 0.0, 0.0, 1.0),
        )
        assert opaque(with_border) > opaque(without_border)


class TestOutline:
    def test_an_outline_adds_ink(self) -> None:
        # 縁取りの太さを上げても縁が描かれないと、字幕が背景に埋もれて読めない
        assert opaque(draw(border_width=AnimatedValue(8.0))) > opaque(draw())

    def test_the_outline_sits_under_the_fill(self) -> None:
        # 文字の中心は塗りの色のまま 縁が上に来ていたら文字が潰れる
        image = draw(
            color=(1.0, 1.0, 1.0, 1.0),
            border_width=AnimatedValue(10.0),
            border_color=(1.0, 0.0, 0.0, 1.0),
        )
        white = ((image[:, :, 0] > 200) & (image[:, :, 1] > 200) & (image[:, :, 3] > 200)).sum()
        assert white > 0


class TestVerticalAnchor:
    def test_the_bottom_anchor_puts_the_text_above_the_point(self) -> None:
        # 縦の基準を下にしても字が点の下へ垂れると、画面の下端に置いた字幕が画面の外へはみ出す
        top = centre_of_mass(draw(valign="top"))[0]
        middle = centre_of_mass(draw(valign="middle"))[0]
        bottom = centre_of_mass(draw(valign="bottom"))[0]
        assert bottom < middle < top

    def test_two_lines_move_further(self) -> None:
        # 下基準では、行が増えるぶんだけ上へ伸びる 下端は動かない
        # 画面からはみ出すと測れないので、ここだけ小さい文字で描く
        small = AnimatedValue(28.0)
        one = centre_of_mass(draw(text="あ", size=small, valign="bottom"))[0]
        two = centre_of_mass(draw(text="あ" + chr(10) + "い", size=small, valign="bottom"))[0]
        assert two < one


class TestVerticalWriting:
    def test_vertical_text_is_taller_than_it_is_wide(self) -> None:
        # 縦書きの受け皿は前からあったが、呼ばれていなかった
        image = draw(text="あいう", vertical=True)
        ys, xs = np.nonzero(image[:, :, 3] > 8)
        assert np.ptp(ys) > np.ptp(xs)

    def test_horizontal_text_is_wider_than_it_is_tall(self) -> None:
        # 横書きのテキストが縦に並んで描かれると、ふつうの字幕が縦書きになる
        image = draw(text="あいう", vertical=False)
        ys, xs = np.nonzero(image[:, :, 3] > 8)
        assert np.ptp(xs) > np.ptp(ys)

    def test_vertical_text_takes_decorations_too(self) -> None:
        # 縦書きにすると縁取りが消えると、縦書きのテロップだけ縁の無い字になる
        plain = draw(text="あい", vertical=True)
        decorated = draw(text="あい", vertical=True, border_width=AnimatedValue(8.0))
        assert opaque(decorated) > opaque(plain)


class TestReveal:
    def test_the_reveal_still_works_with_decorations(self) -> None:
        # 縁取りを付けた字で文字送りが効かないと、1 字ずつ出るはずの字幕が初めから全部出る
        half = draw(text="あいうえお", reveal=AnimatedValue(40.0), border_width=AnimatedValue(4.0))
        full = draw(text="あいうえお", reveal=AnimatedValue(100.0), border_width=AnimatedValue(4.0))
        assert opaque(half) < opaque(full)

    def test_nothing_is_drawn_at_zero(self) -> None:
        # 文字送り 0% のコマで字が少しでも描かれると、出てくる前の字がちらつく
        assert opaque(draw(text="あいう", reveal=AnimatedValue(0.0))) == 0


@pytest.mark.parametrize("blur", [1.0, 4.0, 16.0])
def test_the_blur_keeps_the_alpha_in_range(blur: float) -> None:
    # 平均を取るだけなので 255 を超えないはずだが、超えると桁が回って
    # 影に穴が空く
    image = draw(
        shadow_x=AnimatedValue(4.0),
        shadow_y=AnimatedValue(-4.0),
        shadow_blur=AnimatedValue(blur),
        shadow_color=(0.0, 0.0, 0.0, 1.0),
    )
    assert image[:, :, 3].max() <= 255
