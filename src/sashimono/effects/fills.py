"""テキストの塗り・縁取りの層・影の色を、単色か複数色のグラデーションで塗る項目（#312）

Premiere Pro のテキストの「塗り」と同じく、塗りの種類（単色・線形・円形）を選び、
グラデーションは色の止まりの並び（:class:`~sashimono.effects.spec.GradientSpec`）で持つ
同じ項目の組を、字の塗り（``fill``）・影（``shadow_fill``）・縁取りの層（層の中の ``fill``）に
名前の頭だけを変えて持たせる 3 つを別々に書くと、範囲や既定が少しずつずれていく

グラデーションの基準は字の枠（字の輪郭を囲む四角 AviUtl2 の組み方なら文字の枠も含める）
1 行ごと・1 文字ごとを選ぶと、行や字の輪郭を囲む四角ごとに掛け直す
エフェクトの「グラデーション」「模様で塗る」は絵の全体（縁取りと影も）に掛かるが、こちらは
塗る所ごとに掛かるので、縁取りと影を巻き込まない

- 角度は「グラデーション」エフェクト（AviUtl2 で測った決まり）と同じ 0 で上から下、90 で
  右から左（止まりの位置 0 の色の側から 1 の色の側への向き） 画面の上で時計回りに増える
  テロップの縦のグラデーションが既定の 0 で作れる
- 色の止まりの間は sRGB の値のまま混ぜ、色と不透明度を別々に混ぜる（YMM4・AviUtl2 と同じ）
- 長さ・ずらし・中心・半径は字の枠に対する割合 画質を落としたプレビューでも字と一緒に縮む
"""

from __future__ import annotations

from collections.abc import Mapping

from sashimono.core.model import ParamValue
from sashimono.effects.spec import GradientSpec, ParameterSpec, SelectSpec, TrackSpec

__all__ = [
    "FILL_BASES",
    "FILL_KINDS",
    "FILL_SPANS",
    "GRADIENT_SUFFIXES",
    "fill_names",
    "fill_specs",
    "fill_unused",
    "is_gradient",
]

#: 塗りの種類 単色は前からの色の項目（``color`` など）で塗る
FILL_KINDS = (("solid", "単色"), ("linear", "線形グラデーション"), ("radial", "円形グラデーション"))

#: 線形のグラデーションの長さの基準
#: 枠の向きの長さは、字の枠を角度の向きに測った長さ（100% で枠の端から端まで Premiere と同じ）
#: 枠の幅は角度に関わらず字の枠の横の幅（YMM4 のブラシの割合の指定と同じ 読み込みで使う）
#: 画素は長さとずらしを画面の画素で数える（AviUtl2 の「グラデーション」の幅と同じ 読み込みで使う）
FILL_SPANS = (("extent", "枠の向きの長さ"), ("width", "枠の幅"), ("pixels", "画素"))

#: グラデーションを掛け直す単位 字の枠全体は Premiere と同じ 1 行ごと・1 文字ごとは日本語の
#: テロップで使う（行ごとに上から下へ明るくする、字ごとに同じ光り方をさせる）
FILL_BASES = (("box", "字の枠全体"), ("line", "1 行ごと"), ("char", "1 文字ごと"))

#: 頭の後ろに付ける項目の名前 並びは設定パネルに出す順
#: 基準（``basis``）は後から足したので末尾に置く（前に保存した値の並びを動かさない）
GRADIENT_SUFFIXES = (
    "type",
    "gradient",
    "angle",
    "length",
    "shift",
    "span",
    "center_x",
    "center_y",
    "radius",
    "basis",
)

#: 線形のときだけ使う項目と、円形のときだけ使う項目
_LINEAR_ONLY = frozenset({"angle", "length", "shift", "span"})
_RADIAL_ONLY = frozenset({"center_x", "center_y", "radius"})


def fill_names(prefix: str) -> tuple[str, ...]:
    """``prefix`` の塗りの項目の名前 並びは :data:`GRADIENT_SUFFIXES` と同じ"""
    return tuple(f"{prefix}_{suffix}" for suffix in GRADIENT_SUFFIXES)


def fill_specs(prefix: str, owner: str = "") -> tuple[ParameterSpec, ...]:
    """``prefix`` の塗りの項目 ``owner`` は表示名の頭（「影」なら「影の塗り」）

    範囲はどれも字の枠に対する割合 画素で持つと、字の大きさを変えるたびに作り直すことになる
    角度は ±360 度 「グラデーション」エフェクトと同じ範囲にして、写すときに切れないようにする
    """
    head = f"{owner}の" if owner else ""
    return (
        SelectSpec(f"{prefix}_type", f"{head}塗り", FILL_KINDS, "solid"),
        GradientSpec(f"{prefix}_gradient", f"{head}グラデーション"),
        TrackSpec(f"{prefix}_angle", f"{head}グラデーションの角度", -360, 360, 0, unit="度"),
        # 長さとずらしは基準が画素なら画素の数 AviUtl2 の幅（4000 まで）を切らずに写せる範囲にする
        # 画質を落としたプレビューでも描く座標ごと縮むので、画素の印（px）は付けない
        TrackSpec(
            f"{prefix}_length", f"{head}グラデーションの長さ", 1, 10000, 100, unit="%", pixels=False
        ),
        TrackSpec(
            f"{prefix}_shift",
            f"{head}グラデーションのずらし",
            -10000,
            10000,
            0,
            unit="%",
            pixels=False,
        ),
        SelectSpec(f"{prefix}_span", f"{head}長さの基準", FILL_SPANS, "extent"),
        # 中心の Y は上が正（テキストの位置と同じ）
        TrackSpec(f"{prefix}_center_x", f"{head}グラデーションの中心 X", -500, 500, 0, unit="%"),
        TrackSpec(f"{prefix}_center_y", f"{head}グラデーションの中心 Y", -500, 500, 0, unit="%"),
        TrackSpec(f"{prefix}_radius", f"{head}グラデーションの半径", 1, 1000, 100, unit="%"),
        SelectSpec(f"{prefix}_basis", f"{head}グラデーションの基準", FILL_BASES, "box"),
    )


def is_gradient(params: Mapping[str, ParamValue], prefix: str) -> bool:
    """``prefix`` の塗りがグラデーションか 項目が無い（前の版の作品）ときは単色"""
    return params.get(f"{prefix}_type") in ("linear", "radial")


def fill_unused(params: Mapping[str, ParamValue], prefix: str, colour: str) -> frozenset[str]:
    """``prefix`` の塗りで、今の値では読まない項目 ``colour`` は単色のときに塗る色の項目

    単色ならグラデーションの項目を、グラデーションなら単色の色を隠す 線形と円形は
    それぞれ相手の形だけが読む項目を隠す 値は消さずに残す（種類を戻せば前の設定で出る）
    """
    kind = params.get(f"{prefix}_type")
    if kind not in ("linear", "radial"):
        return frozenset(f"{prefix}_{suffix}" for suffix in GRADIENT_SUFFIXES[1:])
    hidden = _RADIAL_ONLY if kind == "linear" else _LINEAR_ONLY
    return frozenset({colour, *(f"{prefix}_{suffix}" for suffix in hidden)})
