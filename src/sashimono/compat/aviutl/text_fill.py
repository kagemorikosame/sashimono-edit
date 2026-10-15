"""AviUtl2 のテキストのすぐ後ろの「グラデーション」を、テキストの塗りにする（#312）

写すのは、「グラデーション」エフェクトで描いた絵と同じになると言える物だけ 同じにならない物は
これまでどおり「グラデーション」エフェクトのまま残す

決まりは AviUtl2 v2.1.6a に描かせて確かめた（2026-10-15 ``tools/aviutl_compare.py``
手元のエイリアス ``Premiere自作字幕`` の 8 本と、文字揃え・幅・角度・中心・字の大きさを
変えた探り 9 本）

- 中心は文字の枠（AviUtl2 のテキストの絵）の真ん中 中央揃え[下] の字を縦に塗ると、枠の真ん中で
  色が切り替わった（エフェクトのまま絵の真ん中を中心にすると、枠が中心の上にあるので全部が開始色に
  なり、差が 8.8 あった） 左揃え[上] の字を横に塗っても枠の真ん中だった
- 幅は画面の画素 字の大きさ（50 と 100）を変えても色の帯の幅は同じだった
- 角度は 0 で上から下、90 で右から左（「グラデーション」エフェクトと同じ） 45 度も合った
- 混ぜ方は S 字（smoothstep） 幅 800 で帯の中ほどの色も合った 塗りの止まりは直線で混ぜるので、
  S 字を 16 個の止まりで刻む（8 bit で 1 段に満たないずれ）
- 中心 X・Y は帯の向きへのずらしになる（線形なので帯と直角の向きのずらしは絵を変えない）
"""

from __future__ import annotations

import math

from sashimono.core.model import AnimatedValue, Effect, GeneratedSource, ParamValue
from sashimono.effects.spec import MAX_GRADIENT_STOPS

__all__ = ["text_gradient_fill"]


def _static(value: ParamValue | None, default: float) -> float | None:
    """動かない数 動く値や数でない物は ``None``（写さない）"""
    if value is None:
        return default
    if isinstance(value, AnimatedValue):
        return None if value.is_animated else value.static
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _colour(value: ParamValue | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, tuple) and len(value) >= 3:
        return (*value[:3], value[3] if len(value) > 3 else 1.0)
    return default


def _plain_text(source: GeneratedSource) -> bool:
    """縁や影の無い字か、字ごとの制御文字を持たないか

    AviUtl2 のグラデーションは字の絵の全体（文字装飾の縁と影も）を塗る 字の塗りへ移すと縁と影が
    塗られなくなるので、縁や影のある字は写さない 制御文字（``<#色>`` など）で色を書いた字は、
    塗りでは書いた色が勝つので写さない
    """
    if source.kind != "text":
        return False
    params = source.params
    text = params.get("text")
    if not isinstance(text, str) or "<" in text:
        return False
    for name in ("border_width", "shadow_x", "shadow_y", "shadow_blur"):
        amount = _static(params.get(name), 0.0)
        if amount is None or amount != 0.0:
            return False
    return not source.strokes


def text_gradient_fill(effect: Effect, source: GeneratedSource) -> dict[str, ParamValue] | None:
    """テキスト ``source`` のすぐ後ろの ``effect``（グラデーション）を塗りの項目にする

    写せなければ ``None``（呼ぶ側はエフェクトのまま残す）
    """
    if effect.kind != "gradient" or not effect.enabled or not _plain_text(source):
        return None
    params = effect.params
    if params.get("shape", "linear") != "linear" or params.get("blend", "normal") != "normal":
        return None
    numbers = {
        name: _static(params.get(name), default)
        for name, default in (
            ("strength", 100.0),
            ("angle", 90.0),
            ("span", 100.0),
            ("center_x", 0.0),
            ("center_y", 0.0),
        )
    }
    if any(value is None or not math.isfinite(value) for value in numbers.values()):
        return None
    strength = min(max((numbers["strength"] or 0.0) / 100.0, 0.0), 1.0)
    start = _colour(params.get("start_color"), (1.0, 1.0, 1.0, 1.0))
    end = _colour(params.get("end_color"), (0.0, 0.0, 0.0, 1.0))
    if start[3] != end[3]:
        # 色と不透明度が両方変わると、字の色に重ねた色が止まりの間で直線に並ばない
        return None
    under = _colour(source.params.get("color"), (1.0, 1.0, 1.0, 1.0))
    amount = strength * start[3]
    flat: list[float] = []
    last = MAX_GRADIENT_STOPS - 1
    for index in range(MAX_GRADIENT_STOPS):
        position = index / last
        along = position * position * (3.0 - 2.0 * position)
        ramp = [a + (b - a) * along for a, b in zip(start[:3], end[:3], strict=True)]
        mixed = [u + (r - u) * amount for u, r in zip(under[:3], ramp, strict=True)]
        flat.extend((position, *mixed, under[3]))
    angle = numbers["angle"] or 0.0
    radians = math.radians(angle)
    # 帯の向き（画面の Y は下向き）と中心のずらし（設定の Y は上向き 写す所で反転してある）
    dx, dy = -math.sin(radians), math.cos(radians)
    shift = (numbers["center_x"] or 0.0) * dx + (-(numbers["center_y"] or 0.0)) * dy
    return {
        "fill_type": "linear",
        "fill_gradient": tuple(flat),
        "fill_angle": AnimatedValue(angle),
        "fill_length": AnimatedValue(max(numbers["span"] or 1.0, 1.0)),
        "fill_shift": AnimatedValue(shift),
        "fill_span": "pixels",
    }
