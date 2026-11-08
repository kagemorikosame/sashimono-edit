"""プレビューで何も映らない所に敷く地（#456）

何も映らない所は 2 種類ある 画面（作品の枠）の外の余白と、画面の中で何のクリップも描いて
いない所 どちらも黒で塗っていたので、黒い映像と見分けがつかなかった

ここで決めるのは**プレビューにだけ**敷く地 書き出しとスナップショットは今まで通り黒を敷く
（YMM4 と見え方を合わせる作りを変えない） なので :meth:`FrameRenderer.compose` の既定は
黒のままにして、プレビューの道だけが地を渡す

色は見本の絵の地（``ui/library_view.py``）と同じ値を使う 同じ「市松」「暗い地」を選んで、
所によって色が違うと、同じ設定に見えない
"""

from __future__ import annotations

from dataclasses import dataclass

from sashimono.engine.gpu.effects import srgb_to_linear

__all__ = [
    "BLACK",
    "CHECKER_COLORS",
    "CHECKER_ROWS",
    "DARK_COLOR",
    "FILL_BLACK",
    "FILL_CHECKER",
    "FILL_DARK",
    "FILL_LIGHT",
    "FILL_MODES",
    "LIGHT_COLOR",
    "MARGIN_BLACK",
    "MARGIN_GRAY",
    "MARGIN_GRAY_COLOR",
    "MARGIN_MODES",
    "MARGIN_THEME",
    "Backdrop",
    "fill_backdrop",
    "hex_rgb",
]

#: 画面の中で何も描いていない所 :attr:`Preferences.preview_fill` の値
FILL_BLACK = "black"
FILL_CHECKER = "checker"
FILL_DARK = "dark"
FILL_LIGHT = "light"
FILL_MODES = (FILL_BLACK, FILL_CHECKER, FILL_DARK, FILL_LIGHT)

#: 画面の外の余白 :attr:`Preferences.preview_margin` の値
MARGIN_GRAY = "gray"
MARGIN_BLACK = "black"
MARGIN_THEME = "theme"
MARGIN_MODES = (MARGIN_GRAY, MARGIN_BLACK, MARGIN_THEME)

#: 市松の 2 色・暗い地・明るい地（sRGB の 16 進） 見本の絵の地と同じ値
#: 画面のテーマでは変えない 地は映像の代わりで、テーマに合わせると明るいテーマで
#: 白い字幕が見えなくなる
CHECKER_COLORS = ("#9a9a9a", "#6e6e6e")
DARK_COLOR = "#202020"
LIGHT_COLOR = "#f2f2f2"
#: 余白の暗い灰色 黒い映像（0）とも暗い地（#202020）とも見分けられ、
#: 暗いテーマの窓の地より少し明るいので、画面の枠の境が見える
MARGIN_GRAY_COLOR = "#383838"

#: 市松の 1 目を、画面の高さの何分の 1 にするか
#: 画素で決めると、プレビューの画質（1/2 や 1/4 で描く）を変えるたびに目の大きさが
#: 変わって見える 高さで割れば、画質を変えても画面の上で同じ大きさに見える
CHECKER_ROWS = 18


def hex_rgb(value: str) -> tuple[float, float, float]:
    """``#rrggbb`` を sRGB の 0..1 へ"""
    text = value.lstrip("#")
    return (int(text[0:2], 16) / 255.0, int(text[2:4], 16) / 255.0, int(text[4:6], 16) / 255.0)


def _linear(value: str) -> tuple[float, float, float, float]:
    red, green, blue = hex_rgb(value)
    return (srgb_to_linear(red), srgb_to_linear(green), srgb_to_linear(blue), 1.0)


@dataclass(frozen=True, slots=True)
class Backdrop:
    """何も描いていない所に敷く地（リニア値） 合成の最後に、透けた所だけへ敷く

    ``second`` があれば ``color`` と ``second`` の市松にする 値で比べられるので、
    先読みの貯めは前と違う地が来たら絵を捨てる鍵に使う
    """

    color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    second: tuple[float, float, float, float] | None = None

    @property
    def black(self) -> bool:
        """書き出しと同じ黒か プレビューに「書き出しでは黒」の印を出すかを決める"""
        return self.second is None and self.color[:3] == (0.0, 0.0, 0.0)


#: 書き出しとスナップショットが敷く地 プレビューの既定もこれ
BLACK = Backdrop()


def fill_backdrop(mode: str) -> Backdrop:
    """設定の値から地を作る 知らない値は黒（今までの見え方）"""
    if mode == FILL_CHECKER:
        return Backdrop(_linear(CHECKER_COLORS[0]), _linear(CHECKER_COLORS[1]))
    if mode == FILL_DARK:
        return Backdrop(_linear(DARK_COLOR))
    if mode == FILL_LIGHT:
        return Backdrop(_linear(LIGHT_COLOR))
    return BLACK
