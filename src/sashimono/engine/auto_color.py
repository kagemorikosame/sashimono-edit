"""クリップの色を測り、拡張色調補正の値を決める（#336）

AI の道具（``auto_color``）が使う 新しい種類の補正は作らず、人が設定パネルで触る
拡張色調補正（``color_grade`` :mod:`sashimono.effects.grading`）の値を決めるだけにする
付けた後は人が同じ欄で直せ、取り消し 1 回で戻る

外の機械学習のモデルは使わない（#298 の「決めること」） 測るのは取り込みのときに作る
サムネイルのシート（:mod:`sashimono.engine.cache.thumbnails` 0.5 秒ごと 高さ 72）だけで、
素材を読み直さない 縮めた絵でも明るさの分布と色の平均はほとんど変わらない

値は画面に出る値（sRGB）で測る 拡張色調補正も sRGB の値を動かす（AviUtl と同じ）ので、
測った値と掛けた後の値を同じ物差しで比べられる 掛けた後の絵は :func:`graded` で
シェーダと同じ式を numpy で辿って求める（GPU を通さずに確かめる 試験で GPU の絵と比べる）

決めること

- 明るさ（露出とコントラスト） 暗い方から 0.5% の所を黒へ、明るい方から 0.5% の所を白へ
  広げる 広げすぎると平らな絵（霧・曇り空）で雑音まで目立つので倍率に上限を置く
  真ん中の明るさが暗すぎる・明るすぎるときだけガンマで寄せる
- 白のつり合い 色の薄い所（白・灰色の物）の赤・緑・青の平均をそろえる（灰色の仮定）
  色の薄い所が少ない絵は全体の平均でそろえ、強さを控える
- ほかのクリップに合わせる（``reference``） 明るさの幅と真ん中、色の平均の比、彩度を
  相手の絵へ寄せる
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "AUTO_COLOR_STRENGTH_RANGE",
    "DEFAULT_AUTO_COLOR_STRENGTH",
    "GRADE_KIND",
    "ColorPlan",
    "ColorStats",
    "graded",
    "measure",
    "pixels_of",
    "plan_auto",
    "plan_match",
]

#: 決めた値を入れるエフェクト（拡張色調補正）
GRADE_KIND = "color_grade"
#: 強さの既定と範囲（%） 決めた値にこれを掛ける 100 で測ったとおりに直す
#: 既定は控えめにする 夕焼けの赤み・夜の暗さのように、わざとそう撮った絵まで
#: 真っ平らな絵へ寄せてしまうことがあるため（知らない人が困らない側）
DEFAULT_AUTO_COLOR_STRENGTH = 70
AUTO_COLOR_STRENGTH_RANGE = (10, 100)

#: 明るさの重み（Rec.601） 拡張色調補正のシェーダと同じ
_LUMA = np.array([0.299, 0.587, 0.114])
#: 測る画素の上限 シートは長い素材で横に 600 枚並ぶので、間引いてから測る
_MAX_PIXELS = 200_000
#: 黒と白とみなす所（明るさの分布の端から %）
_CLIP_PERCENT = 0.5
#: 明るさを広げる倍率の上限 暗い夜の絵を昼のように持ち上げない
_MAX_STRETCH = 1.8
#: 明るさの幅がこれより狭い絵は広げない（一色に近い絵 黒い画面・白い画面）
_MIN_SPAN = 0.08
#: 黒と白がもうこれだけ出ている絵は広げない（暗い方の端・明るい方の端） 画面の録画のように
#: 白が 95% ほどで止まる絵を毎回少しだけ広げると、整っている絵まで変わる（本人の手元の実素材）
_LEVELS_DONE = (0.04, 0.9)
#: 真ん中の明るさをこの間に収める（sRGB） 外れたときだけガンマで寄せる
#: 下を 0.33 にしていたら、暗めに作られた画面の録画（真ん中 0.29）まで持ち上げた
_MIDDLE_BAND = (0.25, 0.65)
#: ガンマの乗数の範囲 大きく曲げると肌の階調が潰れる
_POWER_RANGE = (0.75, 1.33)
#: 色の薄い所とみなす彩度（最大と最小の差 / 最大）
_NEUTRAL_SATURATION = 0.3
#: 色の薄い所を測る明るさの範囲 黒に近い所は雑音、白に近い所は飛んでいて色が当てにならない
_NEUTRAL_LUMA = (0.12, 0.92)
#: 色の薄い所（灰色の明るさの範囲で彩度の低い画素）が全体の画素のうちこれより少なければ、
#: 白のつり合いは直さない 本人の手元の実素材で決めた 実写は 0.28〜0.78、画面の録画は 0.14、
#: 暗い所の多い実写は 0.08 で、録画の色の偏りを色かぶりと見て直していた（前は 0.03）
_NEUTRAL_SHARE = 0.2
#: 赤・緑・青の倍率の範囲 外れるほどの色かぶりは、測り違い（一面の緑の芝など）の方が多い
_CHANNEL_RANGE = (0.8, 1.25)
#: これより小さい色かぶりは直さない（倍率 1 からの差）
_MIN_CAST = 0.015
#: 合わせるときの倍率の範囲（明るさの幅・彩度）
_MATCH_RANGE = (0.5, 2.0)


@dataclass(frozen=True, slots=True)
class ColorStats:
    """絵の明るさの分布と色の平均（どれも sRGB 0..1）"""

    #: 暗い方から 0.5% の明るさ
    black: float
    #: 明るい方から 0.5% の明るさ
    white: float
    #: 真ん中の明るさ
    middle: float
    #: 色の薄い所の赤・緑・青の平均 色の薄い所が少なければ全体の平均
    neutral: tuple[float, float, float]
    #: 色の薄い所の割合
    neutral_share: float
    #: 全体の赤・緑・青の平均（明るさの端を除く）
    average: tuple[float, float, float]
    #: 色の濃さ（色差の長さ）の平均
    chroma: float

    def summary(self) -> dict[str, object]:
        """AI へ返す形（%）"""
        red, green, blue = self.neutral
        mean = (red + green + blue) / 3.0 or 1.0
        return {
            "black_percent": round(self.black * 100, 1),
            "white_percent": round(self.white * 100, 1),
            "middle_percent": round(self.middle * 100, 1),
            # 灰色の所の色の偏り 0 ならかぶっていない 赤が多ければ赤が正
            "cast_percent": {
                "red": round((red / mean - 1.0) * 100, 1),
                "green": round((green / mean - 1.0) * 100, 1),
                "blue": round((blue / mean - 1.0) * 100, 1),
            },
            "saturation_percent": round(self.chroma * 100, 1),
            # 色の薄い所の割合 少ない絵の cast_percent は全体の平均で、色かぶりではなく
            # 映っている物の色のことが多い（読み違えないよう一緒に返す）
            "neutral_percent": round(self.neutral_share * 100, 1),
        }


@dataclass(frozen=True, slots=True)
class ColorPlan:
    """拡張色調補正へ入れる値と、掛ける前と後の様子"""

    params: dict[str, float]
    before: ColorStats
    after: ColorStats
    notes: tuple[str, ...] = field(default=())

    @property
    def changes(self) -> bool:
        return any(abs(value) >= 0.05 for value in self.params.values())


def pixels_of(images: Iterable[np.ndarray]) -> np.ndarray:
    """RGBA の絵（uint8）から、透けていない画素の sRGB（0..1 ``(数, 3)``）を集める"""
    picked: list[np.ndarray] = []
    for image in images:
        flat = np.asarray(image).reshape(-1, image.shape[-1])
        if flat.shape[1] >= 4:
            flat = flat[flat[:, 3] > 0]
        picked.append(flat[:, :3])
    if not picked:
        return np.zeros((0, 3))
    joined = np.concatenate(picked).astype(np.float64) / 255.0
    if len(joined) > _MAX_PIXELS:
        # 等間隔に間引く 乱数で選ぶと、同じクリップで呼ぶたびに値が変わる
        joined = joined[:: math.ceil(len(joined) / _MAX_PIXELS)]
    return joined


def measure(rgb: np.ndarray) -> ColorStats:
    """画素（sRGB ``(数, 3)``）の明るさの分布と色の平均"""
    if len(rgb) == 0:
        raise ValueError("測る画素がありません")
    luma = rgb @ _LUMA
    black, middle, white = np.percentile(luma, [_CLIP_PERCENT, 50.0, 100.0 - _CLIP_PERCENT])
    inside = (luma >= _NEUTRAL_LUMA[0]) & (luma <= _NEUTRAL_LUMA[1])
    body = rgb[inside] if inside.any() else rgb
    high = rgb.max(axis=1)
    saturation = (high - rgb.min(axis=1)) / np.maximum(high, 1e-6)
    neutral_mask = inside & (saturation <= _NEUTRAL_SATURATION)
    share = float(neutral_mask.mean())
    neutral = rgb[neutral_mask] if share >= _NEUTRAL_SHARE else body
    u = 0.492 * (rgb[:, 2] - luma)
    v = 0.877 * (rgb[:, 0] - luma)
    return ColorStats(
        black=float(black),
        white=float(white),
        middle=float(middle),
        neutral=_triple(neutral.mean(axis=0)),
        neutral_share=share,
        average=_triple(body.mean(axis=0)),
        chroma=float(np.hypot(u, v).mean()),
    )


def graded(rgb: np.ndarray, params: Mapping[str, float | bool]) -> np.ndarray:
    """拡張色調補正（:data:`~sashimono.effects.grading._COLOR_GRADE`）を掛けた後の sRGB

    シェーダと同じ式を同じ順で辿る（明るさ・彩度・色相を YUV で動かしてから、赤・緑・青）
    シェーダを直したらここも直す（``tests/engine/test_auto_color.py`` が GPU の絵と比べる）
    """

    def value(name: str) -> float:
        raw = params.get(name, 0.0)
        return float(raw) if not isinstance(raw, bool) else 0.0

    def curve(channel: np.ndarray, prefix: str) -> np.ndarray:
        out = channel * (1.0 + value(f"{prefix}_gain") * 0.01) + value(f"{prefix}_offset") * 0.01
        low = value(f"{prefix}_lift") * 0.01
        out = out * (1.0 - low) + low
        bent: np.ndarray = np.maximum(out, 0.0) ** (2.0 ** (-value(f"{prefix}_gamma") * 0.01))
        return bent

    clipped = np.clip(rgb, 0.0, 1.0)
    y = clipped @ _LUMA
    u = 0.492 * (clipped[:, 2] - y)
    v = 0.877 * (clipped[:, 0] - y)
    y = curve(y, "luma")
    chroma = np.hypot(u, v)
    coloured = chroma > 1e-6
    if coloured.any():
        scaled = curve(chroma[coloured], "sat")
        turn = math.radians(value("hue_offset"))
        cos, sin = math.cos(turn), math.sin(turn)
        ratio = scaled / chroma[coloured]
        cu, cv = u[coloured], v[coloured]
        u = u.copy()
        v = v.copy()
        u[coloured] = (cu * cos - cv * sin) * ratio
        v[coloured] = (cu * sin + cv * cos) * ratio
    out = np.stack([y + 1.140 * v, y - 0.395 * u - 0.581 * v, y + 2.032 * u], axis=1)
    for index, prefix in enumerate(("red", "green", "blue")):
        out[:, index] = curve(out[:, index], prefix)
    if params.get("clamped", True) is False:
        high = out.max(axis=1, keepdims=True)
        out = np.where(high > 1.0, out / np.maximum(high, 1e-6), out)
    clamped: np.ndarray = np.clip(out, 0.0, 1.0)
    return clamped


def plan_auto(
    rgb: np.ndarray, *, strength: float, exposure: bool = True, white_balance: bool = True
) -> ColorPlan:
    """測った絵を、黒から白まで使い、灰色が灰色に見えるように直す値"""
    before = measure(rgb)
    params: dict[str, float] = {}
    notes: list[str] = []
    if exposure:
        params |= _levels(before, notes)
    if white_balance:
        middle = measure(graded(rgb, params)) if params else before
        gains = _balance(middle, notes)
        params |= gains
    return _finish(rgb, before, params, strength, notes)


def plan_match(
    rgb: np.ndarray,
    reference: np.ndarray,
    *,
    strength: float,
    exposure: bool = True,
    white_balance: bool = True,
) -> ColorPlan:
    """``rgb`` の絵を ``reference`` の絵の明るさと色合いへ寄せる値"""
    before = measure(rgb)
    target = measure(reference)
    params: dict[str, float] = {}
    notes: list[str] = []
    if exposure:
        span = before.white - before.black
        if span < _MIN_SPAN:
            notes.append("明るさの幅がほとんど無い絵なので、明るさは合わせませんでした")
        else:
            scale = _within((target.white - target.black) / span, _MATCH_RANGE)
            offset = target.black - before.black * scale
            params["luma_gain"] = (scale - 1.0) * 100.0
            params["luma_offset"] = offset * 100.0
            moved = float(np.clip(before.middle * scale + offset, 0.02, 0.98))
            power = _within(math.log(max(target.middle, 0.02)) / math.log(moved), _POWER_RANGE)
            if abs(power - 1.0) > 0.02:
                params["luma_gamma"] = -100.0 * math.log2(power)
    if white_balance:
        middle = measure(graded(rgb, params)) if params else before
        ratio = np.array(target.average) / max(sum(target.average) / 3.0, 1e-6)
        mine = np.array(middle.average) / max(sum(middle.average) / 3.0, 1e-6)
        gains = np.clip(ratio / np.maximum(mine, 1e-6), *_CHANNEL_RANGE)
        if np.abs(gains - 1.0).max() >= _MIN_CAST:
            for name, gain in zip(("red", "green", "blue"), gains, strict=True):
                params[f"{name}_gain"] = (float(gain) - 1.0) * 100.0
        if middle.chroma > 1e-3 and target.chroma > 1e-3:
            saturation = _within(target.chroma / middle.chroma, _MATCH_RANGE)
            if abs(saturation - 1.0) > 0.03:
                params["sat_gain"] = (saturation - 1.0) * 100.0
    return _finish(rgb, before, params, strength, notes)


def _levels(stats: ColorStats, notes: list[str]) -> dict[str, float]:
    """暗い端を黒へ、明るい端を白へ 倍率は上限まで 真ん中が外れていればガンマで寄せる"""
    span = stats.white - stats.black
    if span < _MIN_SPAN:
        notes.append("明るさの幅がほとんど無い絵（一色に近い）なので、明るさは直しませんでした")
        return {}
    params: dict[str, float] = {}
    if stats.black <= _LEVELS_DONE[0] and stats.white >= _LEVELS_DONE[1]:
        # 黒も白ももう出ている 広げずに、真ん中だけを見る
        scale = 1.0
    else:
        scale = min(1.0 / span, _MAX_STRETCH)
    if scale > 1.0 + 1e-3:
        params["luma_gain"] = (scale - 1.0) * 100.0
        params["luma_offset"] = -stats.black * scale * 100.0
    if scale >= _MAX_STRETCH:
        notes.append(
            f"明るさの幅が狭い絵なので、広げるのを {_MAX_STRETCH:g} 倍で止めました"
            "（暗い場面をわざと暗く撮ったときに昼の絵へ寄せないため）"
        )
    shift = stats.black if scale > 1.0 + 1e-3 else 0.0
    moved = float(np.clip((stats.middle - shift) * scale, 0.02, 0.98))
    low, high = _MIDDLE_BAND
    if not low <= moved <= high:
        aim = low if moved < low else high
        power = _within(math.log(aim) / math.log(moved), _POWER_RANGE)
        params["luma_gamma"] = -100.0 * math.log2(power)
    return params


def _balance(stats: ColorStats, notes: list[str]) -> dict[str, float]:
    """色の薄い所の赤・緑・青の平均をそろえる倍率"""
    means = np.array(stats.neutral)
    if means.max() <= 1e-3:
        return {}
    if stats.neutral_share < _NEUTRAL_SHARE:
        # 灰色の物が少ない絵（色の付いた物で埋まった録画・暗い所の多い絵）では、灰色の所の偏りが
        # 映っている物の色か色かぶりかを見分けられない 直すと、わざとの色まで消す
        notes.append(
            "白や灰色の物が少ない絵なので、偏りが映っている物の色か色かぶりかを見分けられず、"
            "白のつり合いは直しませんでした"
        )
        return {}
    gains = np.clip(means.mean() / np.maximum(means, 1e-6), *_CHANNEL_RANGE)
    if np.abs(gains - 1.0).max() < _MIN_CAST:
        return {}
    return {
        f"{name}_gain": (float(gain) - 1.0) * 100.0
        for name, gain in zip(("red", "green", "blue"), gains, strict=True)
    }


def _finish(
    rgb: np.ndarray,
    before: ColorStats,
    params: dict[str, float],
    strength: float,
    notes: list[str],
) -> ColorPlan:
    """強さを掛けて欄の範囲へ収め、掛けた後の様子を測る"""
    share = float(np.clip(strength, 0.0, 100.0)) / 100.0
    fitted = {
        name: round(float(np.clip(value * share, -100.0, 100.0)), 1)
        for name, value in params.items()
    }
    fitted = {name: value for name, value in fitted.items() if value != 0.0}
    after = measure(graded(rgb, fitted)) if fitted else before
    return ColorPlan(params=fitted, before=before, after=after, notes=tuple(notes))


def _within(value: float, bounds: tuple[float, float]) -> float:
    return float(min(max(value, bounds[0]), bounds[1]))


def _triple(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])
