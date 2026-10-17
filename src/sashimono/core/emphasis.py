"""強調の動き（#331 B5）の線を組む計算

話の要所を目立たせる動き（ズーム・パン・揺れ・ポップ）を、人がキーフレームを打って作るのと
同じ物にする 動きは配置（固定の項目 ``transform``）の値の線と、動きのエフェクト
（``random_move`` 揺れ）の幅の線で表す 専用のエフェクトを新しく作らないのは、人が設定パネルと
グラフエディタで点を直せるようにするため 新しい種類にすると、人は AI が付けた動きを
欄の値でしか直せない

ここは Qt もエフェクトの定義も知らない 今の値の線（:class:`AnimatedValue`）と、範囲と強さを
受け取り、新しい線を返すだけ 範囲の外の値は変えない（範囲の頭と終わりに、形を変えない点を
足してから中を組む :meth:`AnimatedValue.with_keyframe_at`）

Y は上が正（配置の ``pos_y`` と同じ） パンの「上」は ``pos_y`` を増やす
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction

from sashimono.core.commands.base import Command
from sashimono.core.commands.effects import AddEffect, ParamPath, SetParam
from sashimono.core.commands.fixed import (
    TRANSFORM_EFFECT_KIND,
    fixed_effect,
    fixed_slot,
    takes_picture_items,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    Interpolation,
    Keyframe,
    Project,
    Track,
)
from sashimono.core.model.easing import CURVES
from sashimono.core.timebase import FrameRate, Rounding, seconds_to_frame

__all__ = [
    "DEFAULT_EMPHASIS_EASING",
    "DEFAULT_EMPHASIS_PAN_PX",
    "DEFAULT_EMPHASIS_POP_PERCENT",
    "DEFAULT_EMPHASIS_RAMP_MS",
    "DEFAULT_EMPHASIS_SHAKE_PX",
    "DEFAULT_EMPHASIS_ZOOM_PERCENT",
    "EMPHASIS_EASINGS",
    "EMPHASIS_EASING_NAMES",
    "EMPHASIS_KINDS",
    "EMPHASIS_KIND_NAMES",
    "EMPHASIS_PAN_PX_RANGE",
    "EMPHASIS_POP_PERCENT_RANGE",
    "EMPHASIS_RAMP_MS_RANGE",
    "EMPHASIS_SHAKE_PX_RANGE",
    "EMPHASIS_ZOOM_PERCENT_RANGE",
    "PAN",
    "PAN_DIRECTIONS",
    "PAN_DIRECTION_NAMES",
    "POP",
    "SHAKE",
    "ZOOM",
    "EmphasisError",
    "EmphasisStyle",
    "Envelope",
    "can_emphasize",
    "easing_point",
    "envelope",
    "fixed_transform",
    "pan_curves",
    "ramp_frames",
    "scaled_curve",
    "shake_curve",
    "shaped",
    "transform_commands",
    "zoom_curves",
]

ZOOM = "zoom"
PAN = "pan"
SHAKE = "shake"
POP = "pop"
#: 動きの種類 ズームとパンは戻さずに残せる（``keep``） 揺れとポップは一瞬の動きなので残さない
EMPHASIS_KINDS = (ZOOM, PAN, SHAKE, POP)
#: 画面と取り消しの段の名前に出す種類の名前
EMPHASIS_KIND_NAMES: dict[str, str] = {
    ZOOM: "寄る（ズーム）",
    PAN: "ずらす（パン）",
    SHAKE: "揺らす",
    POP: "一瞬ふくらむ（ポップ）",
}
#: パンの向き（Y は上が正 配置の pos_y と同じ）
PAN_DIRECTIONS: dict[str, tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
}
PAN_DIRECTION_NAMES: dict[str, str] = {"left": "左", "right": "右", "up": "上", "down": "下"}

#: ズームの既定の拡大率（今の大きさからの %） 120% は話者の顔へ寄ったと分かり、画面の端が
#: 大きく欠けない量 130% を超えると、横並びの 2 人の片方が画面から外れやすい
DEFAULT_EMPHASIS_ZOOM_PERCENT = 120
EMPHASIS_ZOOM_PERCENT_RANGE = (101, 400)
#: ポップ（一瞬ふくらんで戻る）の既定の拡大率 ズームより小さくする テロップや絵が跳ねたと
#: 分かればよく、大きいと画面の外へはみ出す
DEFAULT_EMPHASIS_POP_PERCENT = 115
EMPHASIS_POP_PERCENT_RANGE = (101, 200)
#: 揺れの既定の幅（画素 上下左右それぞれ） 1920x1080 で 12 画素は、驚きやツッコミの揺れと
#: 分かり、字幕が読めなくなるほどではない量
DEFAULT_EMPHASIS_SHAKE_PX = 12
EMPHASIS_SHAKE_PX_RANGE = (1, 200)
#: パンで向きだけを言われたときに動かす長さ（画素） 1920 の幅のおよそ 1/16
DEFAULT_EMPHASIS_PAN_PX = 120
EMPHASIS_PAN_PX_RANGE = (1, 2000)
#: 動き始めてから動き切るまで（と戻るまで）の長さ（ミリ秒） 短いと急に寄ったように見え、
#: 長いと要所を過ぎてから寄り切る 0 は 1 フレームで切り替える（ジャンプカットの寄り）
DEFAULT_EMPHASIS_RAMP_MS = 250
EMPHASIS_RAMP_MS_RANGE = (0, 3000)

#: 動き方の名前 ``smooth`` は CSS の ease-in-out（出だしと終わりがなめらか） ほかは
#: キーフレームの曲線の名前（:data:`~sashimono.core.model.easing.CURVES`）
#: 行き過ぎて戻る曲線（back・elastic・bounce）は Out で使う 着いた所で跳ねるのが強調の見え方で、
#: InOut にすると動き出しで逆へ振れる
SMOOTH = "smooth"
LINEAR = "linear"
EMPHASIS_EASINGS: tuple[str, ...] = (
    SMOOTH,
    LINEAR,
    "sine",
    "quad",
    "cubic",
    "expo",
    "circ",
    "back",
    "elastic",
    "bounce",
)
#: 設定画面に出す動き方の名前 曲線の英語名だけでは、知らない人が選べない
EMPHASIS_EASING_NAMES: dict[str, str] = {
    SMOOTH: "なめらか（ゆっくり動き出して止まる）",
    LINEAR: "一定の速さ",
    "sine": "ややなめらか（サイン）",
    "quad": "なめらか・弱（2 次）",
    "cubic": "なめらか・中（3 次）",
    "expo": "鋭い（指数）",
    "circ": "鋭い・円（円）",
    "back": "少し行き過ぎて戻る（バック）",
    "elastic": "ばねのように揺れて止まる（エラスティック）",
    "bounce": "弾んで止まる（バウンス）",
}
_OVERSHOOT = frozenset({"back", "elastic", "bounce"})
DEFAULT_EMPHASIS_EASING = SMOOTH


class EmphasisError(ValueError):
    """強調の動きを組めない（範囲が短い・範囲の中に人が打った点がある など）"""


@dataclass(frozen=True, slots=True)
class EmphasisStyle:
    """強調の動きの既定の強さ・長さ・動き方（本人の設定 AI が値を言わなければこれを使う）"""

    zoom_percent: int = DEFAULT_EMPHASIS_ZOOM_PERCENT
    pop_percent: int = DEFAULT_EMPHASIS_POP_PERCENT
    shake_px: int = DEFAULT_EMPHASIS_SHAKE_PX
    pan_px: int = DEFAULT_EMPHASIS_PAN_PX
    ramp_ms: int = DEFAULT_EMPHASIS_RAMP_MS
    easing: str = DEFAULT_EMPHASIS_EASING


def easing_point(name: str) -> tuple[Interpolation, str]:
    """動き方の名前を、キーフレームの補間と曲線の名前にする"""
    if name == LINEAR:
        return Interpolation.LINEAR, ""
    if name == SMOOTH:
        return Interpolation.EASE_IN_OUT, ""
    if name not in CURVES:
        raise EmphasisError(
            f"知らない動き方です: {name}（選べるのは {'・'.join(EMPHASIS_EASINGS)}）"
        )
    if name in _OVERSHOOT:
        return Interpolation.EASE_OUT, name
    return Interpolation.EASE_IN_OUT, name


def ramp_frames(ramp_ms: int, rate: FrameRate) -> int:
    """動き切るまでのフレーム 1 以上

    0 フレームにすると、動く前の点と動き切った点が同じフレームに重なり、線の点として
    持てない（同じフレームに 2 つの点は置けない） 0 ms は 1 フレームで切り替える
    """
    return max(1, seconds_to_frame(Fraction(ramp_ms, 1000), rate, Rounding.NEAREST))


@dataclass(frozen=True, slots=True)
class Envelope:
    """動きの強さの形 ``points`` は（クリップの頭からのフレーム, 強さ 0〜1, そこから次の点への出方）

    最後の点の出方は使わない（範囲の終わりの点は、範囲の後ろの線の出方を引き継ぐ）
    ``keep`` は範囲の終わりで戻さず、クリップの終わりまで動き切ったまま残す
    """

    points: tuple[tuple[int, float, Interpolation, str], ...]
    keep: bool = False

    @property
    def start(self) -> int:
        return self.points[0][0]

    @property
    def end(self) -> int:
        return self.points[-1][0]


def envelope(
    kind: str, start: int, end: int, ramp: int, easing: str, *, keep: bool = False
) -> Envelope:
    """``start``〜``end``（クリップの頭からのフレーム）の強さの形

    - ズーム・パン・揺れ 動き始めて ``ramp`` で動き切り、止まったまま、終わりの ``ramp`` で戻る
      範囲が ``ramp`` の 2 倍より短ければ、真ん中で動き切ってすぐ戻る
    - ポップ ``ramp`` で動き切り、残りで戻る（止まらない）
    - ``keep`` 動き切ったら戻さない（ズームとパンだけ） 範囲の終わりは見ない
    """
    if kind not in EMPHASIS_KINDS:
        raise EmphasisError(f"知らない動きです: {kind}（選べるのは {'・'.join(EMPHASIS_KINDS)}）")
    if keep and kind not in (ZOOM, PAN):
        raise EmphasisError("戻さずに残せる（keep）のはズームとパンだけです")
    interpolation, curve = easing_point(easing)
    if keep:
        # 1 フレームあれば動き切れる 範囲の終わりはクリップの終わりなので見ない
        return Envelope(
            (
                (start, 0.0, interpolation, curve),
                (start + ramp, 1.0, Interpolation.LINEAR, ""),
            ),
            keep=True,
        )
    if end - start < 2:
        # 動き出しと動き切りと戻りの 3 点を置けない 2 点だと動き切った所が範囲の終わりになり、
        # 戻らずに線の端の値（動いていない値）へ飛ぶ
        raise EmphasisError("範囲が短すぎます（2 フレーム以上にしてください）")
    if kind == POP:
        peak = start + min(ramp, (end - start) // 2)
        return Envelope(
            (
                (start, 0.0, interpolation, curve),
                (peak, 1.0, interpolation, curve),
                (end, 0.0, Interpolation.LINEAR, ""),
            )
        )
    reach = min(ramp, (end - start) // 2)
    rise, fall = start + reach, end - reach
    if rise >= fall:
        return Envelope(
            (
                (start, 0.0, interpolation, curve),
                (rise, 1.0, interpolation, curve),
                (end, 0.0, Interpolation.LINEAR, ""),
            )
        )
    return Envelope(
        (
            (start, 0.0, interpolation, curve),
            (rise, 1.0, Interpolation.LINEAR, ""),
            (fall, 1.0, interpolation, curve),
            (end, 0.0, Interpolation.LINEAR, ""),
        )
    )


def _as_curve(value: object, default: float) -> AnimatedValue:
    if isinstance(value, AnimatedValue):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return AnimatedValue(static=float(value))
    return AnimatedValue(static=default)


def shaped(
    base: AnimatedValue,
    shape: Envelope,
    value_at: Callable[[float, float], float],
) -> AnimatedValue:
    """``base`` の範囲の中を ``value_at(今の値, 強さ)`` で置き換えた線

    範囲の外は形を変えない 範囲の頭と終わりに形を変えない点を足してから、中の点を置く
    範囲の中に人が打った点があれば断る 中の点を消すと人が付けた動きが消え、残すと
    強調の形と混ざってどちらの動きでもなくなる
    """
    if shape.keep:
        # 残す動きは範囲の終わりがクリップの終わり 後ろに点があれば、動き切った後に
        # 人が付けた動きへ向かって動き、残したはずの寄りがずれていく
        inside = [k.frame for k in base.keyframes if k.frame > shape.start]
    else:
        inside = [k.frame for k in base.keyframes if shape.start < k.frame < shape.end]
    if inside:
        raise EmphasisError(
            f"範囲の中にキーフレームがあります（クリップの頭から {inside[0]} フレーム目など）"
            " 範囲をずらすか、その点を消してから呼んでください"
        )
    cut = base.with_keyframe_at(shape.start)
    if not shape.keep:
        cut = cut.with_keyframe_at(shape.end)
    kept = {k.frame: k for k in cut.keyframes}
    points: list[Keyframe] = [k for k in cut.keyframes if not shape.start <= k.frame <= shape.end]
    for index, (frame, amount, interpolation, curve) in enumerate(shape.points):
        level = value_at(base.at(frame), amount)
        if index == len(shape.points) - 1 and not shape.keep:
            # 範囲の終わりの点は、範囲の後ろへの出方（もとの線の形）を引き継ぐ 直線に
            # すると、範囲の後ろにあった曲線の動きが直線に変わる
            points.append(replace(kept[frame], value=level))
            continue
        points.append(Keyframe(frame=frame, value=level, interpolation=interpolation, curve=curve))
    return AnimatedValue(static=base.static, keyframes=tuple(sorted(points, key=lambda k: k.frame)))


def scaled_curve(base: AnimatedValue, shape: Envelope, ratio: float) -> AnimatedValue:
    """拡大率の線 動き切った所で今の値の ``ratio`` 倍"""
    return shaped(base, shape, lambda now, amount: now * (1.0 + amount * (ratio - 1.0)))


def zoom_curves(
    transform: Effect,
    shape: Envelope,
    ratio: float,
    focus: tuple[float, float] | None,
) -> dict[str, AnimatedValue]:
    """ズームとポップの線 ``focus`` は動かさずに残す点（配置の支点から 画素 Y は上が正）

    支点から ``Q`` の所にある絵の点は ``pos + 拡大率 x Q`` に描かれる（回転も同じ比で
    掛かるので向きに関係ない） 拡大率を ``k`` 倍にしたとき ``focus`` を動かさないには、
    位置を ``pos + (focus - pos) x (1 - k)`` にする 拡大率と位置の線は同じ強さで同じ
    曲線を辿るので、動きの途中でも ``focus`` は止まって見える
    ``focus`` を省けば位置は変えない（絵の支点を中心に拡がる）
    """
    scale = _as_curve(transform.params.get("scale"), 100.0)
    curves = {"scale": scaled_curve(scale, shape, ratio)}
    if focus is None:
        return curves
    for name, target in (("pos_x", focus[0]), ("pos_y", focus[1])):
        curves[name] = shaped(
            _as_curve(transform.params.get(name), 0.0), shape, _toward(target, ratio)
        )
    return curves


def _toward(target: float, ratio: float) -> Callable[[float, float], float]:
    """拡大率を ``1 + 強さ x (ratio - 1)`` 倍にしたとき ``target`` を動かさない位置"""

    def moved(now: float, amount: float) -> float:
        k = 1.0 + amount * (ratio - 1.0)
        return now + (target - now) * (1.0 - k)

    return moved


def _shifted(shift: float) -> Callable[[float, float], float]:
    return lambda now, amount: now + amount * shift


def pan_curves(
    transform: Effect, shape: Envelope, offset: tuple[float, float]
) -> dict[str, AnimatedValue]:
    """パンの線 動き切った所で今の位置から ``offset``（画素 Y は上が正）だけずらす

    ずらさない向きの線は返さない 返すと、動いていない欄にまで点が並び、人がグラフで
    見たときにどの欄を動かしたのかが分からない
    """
    return {
        name: shaped(_as_curve(transform.params.get(name), 0.0), shape, _shifted(shift))
        for name, shift in (("pos_x", offset[0]), ("pos_y", offset[1]))
        if shift != 0.0
    }


def shake_curve(shape: Envelope, amplitude: float) -> AnimatedValue:
    """揺れの幅の線（範囲の外は 0） ``random_move`` の X と Y の幅に入れる"""
    return shaped(AnimatedValue(static=0.0), shape, lambda _now, amount: amount * amplitude)


def can_emphasize(project: Project, track: Track, clip: Clip) -> bool:
    """強調の動きを付けられるクリップか（AI の道具と右クリックで同じ決まりにする）

    音だけのクリップや場面切り替え・フィルタは配置の欄を持たないので、点を打っても何も
    動かない ロックは人が触らせないために掛けた物なので、右クリックからも AI からも変えない
    """
    return not track.locked and takes_picture_items(clip) and project.draws_picture(track, clip)


def fixed_transform(clip: Clip) -> Effect | None:
    """クリップの固定の配置 まだ持たない古いクリップは ``None``"""
    return next((e for e in clip.effects if e.fixed and e.kind == TRANSFORM_EFFECT_KIND), None)


def transform_commands(clip: Clip, changes: dict[str, AnimatedValue]) -> list[Command]:
    """配置の値の線を ``changes`` に差し替える命令

    固定の配置をまだ持たない古いクリップは、値を入れた固定の配置を足す 1 つの命令にする
    足すのと値を変えるのを分けると、1 回戻すと動かない配置だけが残る
    """
    if not changes:
        return []
    effect = fixed_transform(clip)
    if effect is None:
        added = fixed_effect(TRANSFORM_EFFECT_KIND)
        for name, value in changes.items():
            added = added.with_param(name, value)
        return [AddEffect(clip.id, added, index=fixed_slot(clip.effects, TRANSFORM_EFFECT_KIND))]
    return [
        SetParam(ParamPath.of_effect(clip.id, effect.id, name), value)
        for name, value in changes.items()
    ]
