"""縦動画への再構図（#334 B8）の計算

横の作品を縦（9:16 など）の作品に作り替えるとき、横の画面の上に縦の枠を置き、枠の中だけを
縦の画面いっぱいに映す 枠は被写体を追って動かす（:mod:`sashimono.engine.subject` が見積もった
被写体のいそうな所から、場面ごとに位置を決める）

作り替えは、人が設定パネルで打つのと同じ配置（固定の項目 ``transform``）の値の線と、解像度の
変更だけで表す 専用のエフェクトを新しく作らないのは、人が設定パネルとグラフエディタで枠の
動きを直せるようにするため

- 背景（画面をほぼ覆う動画と画像） 枠の中を縦の画面へ拡げる 拡大率を ``k`` 倍にし、位置を
  ``k x (位置 - 枠の中心)`` にする 回転・左右反転・支点の位置に関係なく、横の画面の点
  ``p`` が縦の画面の ``k x (p - 枠の中心)`` へ行く（:func:`background_changes`）
- 上に重ねた物（テキスト・図形・小さい画像・シーン） 枠と一緒には切らず、縦の画面の同じ
  割合の所へ置き直す 画面の外へはみ出す物は内側へ寄せる（:func:`overlay_changes`）

値の線は 1 次の式（``a x 値 + b``）でだけ書き換える キーフレームの値を同じ式で変えれば、
イージングやベジェの曲がり方は変わらない（曲がり方は 0〜1 の割合で持っている） 前からの
動きと枠の動きの両方が動く所は 1 次の式で書けないので、そのクリップでは枠を止める
（クリップの間の枠の中心の平均）

ここは Qt もエフェクトの定義も素材の読み方も知らない 被写体の位置は呼ぶ側が
（:mod:`sashimono.engine.reframe`）横の画面の画素（中央が 0、Y は上が正）にして渡す
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from sashimono.core.model import AnimatedValue, Interpolation, Keyframe, ParamValue

__all__ = [
    "AUTO",
    "CENTER",
    "DEFAULT_REFRAME_FACES",
    "DEFAULT_REFRAME_FOLLOW",
    "DEFAULT_REFRAME_SIZE",
    "DEFAULT_REFRAME_SMOOTH_MS",
    "FOLLOW",
    "REFRAME_FOLLOWS",
    "REFRAME_FOLLOW_NAMES",
    "REFRAME_SIZES",
    "REFRAME_SIZE_NAMES",
    "REFRAME_SMOOTH_MS_RANGE",
    "STEADY",
    "Area",
    "CropWindow",
    "ReframeError",
    "ReframeStyle",
    "Sample",
    "Shot",
    "ShotPlan",
    "affine",
    "background_changes",
    "clamp_span",
    "crop_window",
    "focus_curves",
    "overlay_changes",
    "parse_size",
    "plan_shot",
    "window_of",
]

#: 作る縦の作品の大きさ（横x縦） 1080x1920 はショート動画（YouTube Shorts・TikTok・
#: Instagram のリール）の大きさ 4:5 はフィードの縦長の投稿、1:1 は正方形の投稿
#: 720x1280 は書き出しを軽くしたい人向け
REFRAME_SIZES: tuple[str, ...] = ("1080x1920", "1080x1350", "1080x1080", "720x1280")
REFRAME_SIZE_NAMES: dict[str, str] = {
    "1080x1920": "縦 9:16（1080x1920 ショート動画）",
    "1080x1350": "縦 4:5（1080x1350 フィードの縦長）",
    "1080x1080": "正方形 1:1（1080x1080）",
    "720x1280": "縦 9:16・軽め（720x1280）",
}
DEFAULT_REFRAME_SIZE = "1080x1920"

#: 枠の動かし方
#: ``auto`` 場面の中で被写体が枠に収まる範囲で動くなら止め、収まらなければ追う
#: ``steady`` 場面ごとに 1 か所で止める（三脚で撮ったように見える 被写体が外れることがある）
#: ``follow`` 被写体を追ってなめらかに動く（手持ちのカメラのように見える）
#: ``center`` 追わずに真ん中を切り出す（被写体を見積もらない 解析を待たない）
AUTO = "auto"
STEADY = "steady"
FOLLOW = "follow"
CENTER = "center"
REFRAME_FOLLOWS: tuple[str, ...] = (AUTO, STEADY, FOLLOW, CENTER)
REFRAME_FOLLOW_NAMES: dict[str, str] = {
    AUTO: "おまかせ（収まる場面は止め、外れる場面は追う）",
    STEADY: "場面ごとに止める",
    FOLLOW: "被写体を追ってなめらかに動く",
    CENTER: "追わずに真ん中を切り出す",
}
#: 既定は止めるのを先にする 動き続ける枠は、知らない人が見ると酔いやすい
DEFAULT_REFRAME_FOLLOW = AUTO

#: 追うときにならす長さ（ミリ秒） 長いほどゆっくり動き、短いほど被写体にぴったり付く
#: 1 秒は話しながら少し動く人に付いていけて、手振りでは揺れない長さ
REFRAME_SMOOTH_MS_RANGE = (200, 5000)
DEFAULT_REFRAME_SMOOTH_MS = 1000
#: 被写体の見積もりに顔の位置（#324 の画面の解析 #407）を使うか 既定は使う
#: 使うのは画面の解析の部品が入っているときだけで、入れた人は人の映る素材を扱うことが
#: 多い 動きと肌色だけでは肌色に近い絵や壁へ枠が寄る（実素材の配信で 12 コマのどれにも
#: アバターが入らなかった） 切るのは、顔を調べる間（場面ごとに数枚 1080p で 1 枚
#: 0.1 秒ほど）を待ちたくない人と、顔でない物を映したい人
DEFAULT_REFRAME_FACES = True

#: 「収まる」と見なす被写体の動く幅（枠の幅に対する割合） 動く幅が枠の 6 割までなら、
#: 止めた枠の真ん中に置けば両端にまだ余白が残る
FIT_SHARE = 0.6
#: 追う線を点にするときに許す、線からのずれ（枠の幅に対する割合） 小さいと点が増え、人が
#: グラフで直すときに直す点が多くなる 4% は 1080 の縦の画面で 43 画素
PATH_TOLERANCE = 0.04
#: 被写体の分布の、外れた値を除くときの割合 上下 10% ずつ（手が一瞬伸びたなど）は数えない
SPREAD_QUANTILES = (0.1, 0.9)

#: 配置の欄の範囲（:mod:`sashimono.effects.builtin` の ``transform``） 越える値は設定パネルで
#: 直せなくなるので、越えるクリップは作り替えずに理由を返す
POSITION_RANGE = (-4000.0, 4000.0)
SCALE_RANGE = (1.0, 800.0)


class ReframeError(ValueError):
    """作り替えられない（ロックしたトラック・同じ大きさ・欄の範囲を越える など）"""


@dataclass(frozen=True, slots=True)
class ReframeStyle:
    """作り替えの既定（本人の設定 AI が値を言わなければこれを使う）"""

    size: str = DEFAULT_REFRAME_SIZE
    follow: str = DEFAULT_REFRAME_FOLLOW
    smooth_ms: int = DEFAULT_REFRAME_SMOOTH_MS
    #: 顔の位置も手がかりにするか（画面の解析の部品が入っているときだけ効く）
    faces: bool = DEFAULT_REFRAME_FACES

    @property
    def resolution(self) -> tuple[int, int]:
        return parse_size(self.size)


def parse_size(text: str) -> tuple[int, int]:
    """``"1080x1920"`` を（横, 縦）にする"""
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError as exc:
        raise ReframeError(f"大きさは 横x縦 で書いてください（{text!r} が来ました）") from exc
    return width, height


@dataclass(frozen=True, slots=True)
class CropWindow:
    """横の画面に置く枠（横の画面の画素）と、枠を縦の画面へ拡げる倍率"""

    old: tuple[int, int]
    new: tuple[int, int]
    #: 枠を縦の画面へ拡げる倍率 枠の大きさは ``new / scale``
    scale: float

    @property
    def size(self) -> tuple[float, float]:
        return self.new[0] / self.scale, self.new[1] / self.scale


def crop_window(old: tuple[int, int], new: tuple[int, int]) -> CropWindow:
    """横の画面に収まるいちばん大きい、縦の画面と同じ縦横比の枠

    枠を大きく取るほど拡げる倍率が小さく、絵が粗くならない 1920x1080 から 1080x1920 なら
    枠は 607.5x1080 で、1.78 倍に拡げる
    """
    if old == new:
        raise ReframeError(f"作品はもう {new[0]}x{new[1]} です")
    return CropWindow(old=old, new=new, scale=max(new[0] / old[0], new[1] / old[1]))


@dataclass(frozen=True, slots=True)
class Sample:
    """被写体のいそうな所 1 つ（タイムラインのフレーム 横の画面の画素 中央が 0 Y は上が正）

    ``weight`` は手がかりの強さ 0 は手がかりが無い（動きも肌色も無い）
    """

    frame: int
    x: float
    y: float
    weight: float


@dataclass(frozen=True, slots=True)
class Area:
    """回した絵の中で枠の中心を置ける範囲 絵の向きに傾いた四角

    ``center`` は四角の中心（横の画面の画素） ``reach`` は絵の向きで見た半分の幅と高さ
    ``turn`` は絵を回した角度（ラジアン Y は上が正で反時計回りが正 配置の回転の符号を返した物）
    枠は画面の向きのまま動くので、枠の角が回した絵からはみ出さない所は傾いた四角になる
    軸にそろえた箱で内側から取ると、傾けた絵ではほとんど動けなくなる
    """

    center: tuple[float, float]
    reach: tuple[float, float]
    turn: float

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """範囲の中のいちばん近い所（絵の向きで、横と縦を別々に寄せる）"""
        cos, sin = math.cos(self.turn), math.sin(self.turn)
        dx, dy = x - self.center[0], y - self.center[1]
        across = _clamp(cos * dx + sin * dy, -self.reach[0], self.reach[0])
        down = _clamp(-sin * dx + cos * dy, -self.reach[1], self.reach[1])
        return (
            self.center[0] + cos * across - sin * down,
            self.center[1] + sin * across + cos * down,
        )


@dataclass(frozen=True, slots=True)
class Shot:
    """1 つの場面（タイムラインのフレーム ``start`` 以上 ``end`` 未満）

    ``bounds`` は枠の中心を置ける範囲（左・右・下・上） 背景の絵の外を映さない範囲
    ``area`` があれば、傾けた絵の中の範囲としてこちらを使う（``bounds`` は見ない）
    ``home`` は手がかりが無いときに置く所（背景の絵の中心）
    """

    start: int
    end: int
    samples: tuple[Sample, ...]
    bounds: tuple[float, float, float, float]
    home: tuple[float, float]
    area: Area | None = None


@dataclass(frozen=True, slots=True)
class ShotPlan:
    """場面ごとに決めた枠の中心 ``mode`` は実際に使った動かし方（auto は止めたか追ったかに直す）"""

    start: int
    end: int
    mode: str
    points: tuple[tuple[int, float, float], ...]
    #: 手がかりが無くて真ん中にした
    no_clue: bool = False


def _clamp(value: float, low: float, high: float) -> float:
    if low > high:
        return (low + high) / 2.0
    return min(max(value, low), high)


def _quantile(values: Sequence[float], weights: Sequence[float], share: float) -> float:
    """重み付きの分位 ``share``（0〜1）"""
    ordered = sorted(zip(values, weights, strict=True))
    total = sum(weight for _value, weight in ordered)
    reach = total * share
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= reach:
            return value
    return ordered[-1][0]


def _smoothed(samples: Sequence[Sample], reach: int) -> list[tuple[int, float, float]]:
    """手がかりの無い所を前の値で埋め、前後 ``reach`` フレームの重み付きの平均でならす"""
    filled: list[tuple[int, float, float, float]] = []
    last: tuple[float, float] | None = None
    first = next((s for s in samples if s.weight > 0), None)
    if first is None:
        return []
    for sample in samples:
        if sample.weight > 0:
            last = (sample.x, sample.y)
            filled.append((sample.frame, sample.x, sample.y, sample.weight))
        else:
            hold = last if last is not None else (first.x, first.y)
            # 埋めた所の重みは小さく取る 手がかりのある所が近くにあれば、そちらに従う
            filled.append((sample.frame, hold[0], hold[1], 1e-6))
    # 累積和で前後の和を引く 1 本の長回し（10 分で 2400 行）でも、行ごとに数え直さない
    frames = [entry[0] for entry in filled]
    sums_w, sums_x, sums_y = [0.0], [0.0], [0.0]
    for _frame, x, y, weight in filled:
        sums_w.append(sums_w[-1] + weight)
        sums_x.append(sums_x[-1] + x * weight)
        sums_y.append(sums_y[-1] + y * weight)
    path: list[tuple[int, float, float]] = []
    for frame in frames:
        low = bisect.bisect_left(frames, frame - reach)
        high = bisect.bisect_right(frames, frame + reach)
        total = sums_w[high] - sums_w[low]
        path.append(
            (
                frame,
                (sums_x[high] - sums_x[low]) / total,
                (sums_y[high] - sums_y[low]) / total,
            )
        )
    return path


def _limited(
    path: list[tuple[int, float, float]], max_step: float
) -> list[tuple[int, float, float]]:
    """枠の動く速さを ``max_step``（1 フレームの画素）までに抑えた線

    ならしても、散った動きの偏りや速い被写体で枠が 1 秒に画面の幅ほど動くことがある 速すぎる枠は
    手で振り回したカメラのように見える 抑えた分は遅れて追い付く
    """
    if not path or math.isinf(max_step):
        return path
    limited = [path[0]]
    for frame, x, y in path[1:]:
        last_frame, last_x, last_y = limited[-1]
        reach = max_step * max(1, frame - last_frame)
        limited.append(
            (
                frame,
                last_x + max(-reach, min(reach, x - last_x)),
                last_y + max(-reach, min(reach, y - last_y)),
            )
        )
    return limited


def _simplified(
    path: list[tuple[int, float, float]], tolerance: tuple[float, float]
) -> list[tuple[int, float, float]]:
    """線を、ずれが ``tolerance`` に収まる少ない点にする（Ramer-Douglas-Peucker）

    人がグラフで直すとき、0.25 秒ごとの点が何百も並ぶと手で直せない
    """
    if len(path) <= 2:
        return list(path)
    # 再帰で書くと、長回しのゆっくりした動き（ずれが端から順に出る）で深さが行の数まで届く
    kept = {0, len(path) - 1}
    pending = [(0, len(path) - 1)]
    while pending:
        begin, finish = pending.pop()
        first, last = path[begin], path[finish]
        span = last[0] - first[0]
        worst, index = 0.0, begin
        for position in range(begin + 1, finish):
            frame, x, y = path[position]
            progress = (frame - first[0]) / span if span else 0.0
            line_x = first[1] + (last[1] - first[1]) * progress
            line_y = first[2] + (last[2] - first[2]) * progress
            error = max(
                abs(x - line_x) / max(tolerance[0], 1e-9),
                abs(y - line_y) / max(tolerance[1], 1e-9),
            )
            if error > worst:
                worst, index = error, position
        if worst > 1.0:
            kept.add(index)
            pending.extend(((begin, index), (index, finish)))
    return [path[position] for position in sorted(kept)]


def plan_shot(
    shot: Shot,
    follow: str,
    window: tuple[float, float],
    smooth_frames: int,
    *,
    max_step: float = math.inf,
) -> ShotPlan:
    """場面 1 つの枠の中心を決める

    ``window`` は枠の大きさ（横の画面の画素） ``smooth_frames`` は追うときにならす長さ
    ``max_step`` は追うときに枠が 1 フレームで動ける長さ（画素 横と縦それぞれ）
    """
    if follow not in REFRAME_FOLLOWS:
        raise ReframeError(
            f"枠の動かし方は {'・'.join(REFRAME_FOLLOWS)} のどれかです（{follow!r} が来ました）"
        )
    left, right, bottom, top = shot.bounds
    area = shot.area

    def placed(x: float, y: float) -> tuple[float, float]:
        if area is not None:
            return area.clamp(x, y)
        return _clamp(x, left, right), _clamp(y, bottom, top)

    home = placed(*shot.home)
    inside = [s for s in shot.samples if shot.start <= s.frame < shot.end]
    strong = [s for s in inside if s.weight > 0]
    if follow == CENTER or not strong:
        return ShotPlan(
            shot.start,
            shot.end,
            CENTER if follow == CENTER else STEADY,
            ((shot.start, *home),),
            no_clue=follow != CENTER,
        )
    weights = [s.weight for s in strong]
    low_x, high_x = (_quantile([s.x for s in strong], weights, q) for q in SPREAD_QUANTILES)
    low_y, high_y = (_quantile([s.y for s in strong], weights, q) for q in SPREAD_QUANTILES)
    fits = high_x - low_x <= window[0] * FIT_SHARE and high_y - low_y <= window[1] * FIT_SHARE
    if follow == STEADY or (follow == AUTO and fits):
        if fits:
            # 動く幅の真ん中に止める 両端の動きがどちらも枠に入る
            center = ((low_x + high_x) / 2.0, (low_y + high_y) / 2.0)
        else:
            # 収まらないのに止めると頼まれた 多くいる所（中央値）に止める
            center = (
                _quantile([s.x for s in strong], weights, 0.5),
                _quantile([s.y for s in strong], weights, 0.5),
            )
        return ShotPlan(shot.start, shot.end, STEADY, ((shot.start, *placed(*center)),))
    path = _limited(
        [(frame, *placed(x, y)) for frame, x, y in _smoothed(inside, smooth_frames // 2)],
        max_step,
    )
    points = _simplified(path, (window[0] * PATH_TOLERANCE, window[1] * PATH_TOLERANCE))
    # 場面の頭から動いた先を映す 最初の手がかりが頭より後ろでも、頭の絵は最初の位置で切る
    points[0] = (shot.start, points[0][1], points[0][2])
    if len(points) >= 2 and points[1][0] <= shot.start:
        points = [points[0], *points[2:]]
    if all(
        math.isclose(p[1], points[0][1], abs_tol=0.5)
        and math.isclose(p[2], points[0][2], abs_tol=0.5)
        for p in points
    ):
        return ShotPlan(shot.start, shot.end, STEADY, (points[0],))
    return ShotPlan(shot.start, shot.end, FOLLOW, tuple(points))


def focus_curves(plans: Sequence[ShotPlan]) -> tuple[AnimatedValue, AnimatedValue]:
    """場面ごとの枠の中心を、タイムラインのフレームの値の線（横, 縦）にする

    場面の中はなめらかに動き、場面の最後の点は次の場面の頭まで止めて、頭で切り替える
    （切り替わりの間に枠が流れると、前の場面の絵が横へ滑って見える）
    """
    keys_x: list[Keyframe] = []
    keys_y: list[Keyframe] = []
    for plan in sorted(plans, key=lambda p: p.start):
        for index, (frame, x, y) in enumerate(plan.points):
            if keys_x and frame <= keys_x[-1].frame:
                continue
            last = index == len(plan.points) - 1
            mode = Interpolation.HOLD if last else Interpolation.EASE_IN_OUT
            keys_x.append(Keyframe(frame=frame, value=x, interpolation=mode))
            keys_y.append(Keyframe(frame=frame, value=y, interpolation=mode))
    return _curve(keys_x), _curve(keys_y)


def _curve(keys: list[Keyframe]) -> AnimatedValue:
    if not keys:
        return AnimatedValue(static=0.0)
    if all(math.isclose(k.value, keys[0].value, abs_tol=1e-6) for k in keys):
        return AnimatedValue(static=keys[0].value)
    return AnimatedValue(static=keys[0].value, keyframes=tuple(keys))


def window_of(curve: AnimatedValue, start: int, end: int) -> AnimatedValue:
    """タイムラインのフレームの線から、``start``〜``end``（含まない）を切り出したクリップの線

    クリップのキーフレームはクリップの頭から数える 範囲の頭と最後のフレームに形を変えない点を
    足してから切るので、範囲の中の値は元の線と同じ
    """
    if not curve.keyframes:
        return curve
    last = max(start, end - 1)
    cut = curve.with_keyframe_at(start).with_keyframe_at(last)
    keys = tuple(
        replace(k, frame=k.frame - start) for k in cut.keyframes if start <= k.frame <= last
    )
    if all(math.isclose(k.value, keys[0].value, abs_tol=1e-6) for k in keys):
        return AnimatedValue(static=keys[0].value)
    return AnimatedValue(static=keys[0].value, keyframes=keys)


def _as_curve(value: ParamValue | None, default: float) -> AnimatedValue:
    if isinstance(value, AnimatedValue):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return AnimatedValue(static=float(value))
    return AnimatedValue(static=default)


def affine(value: ParamValue | None, scale: float, offset: float, default: float) -> AnimatedValue:
    """値の線の、どのフレームの値も ``scale x 値 + offset`` にした線

    キーフレームの値だけを書き換える 曲がり方（イージング・ベジェ・名前付きの曲線）は
    0〜1 の割合で持っているので、1 次の式なら形はそのまま移る
    """
    base = _as_curve(value, default)
    return AnimatedValue(
        static=base.static * scale + offset,
        keyframes=tuple(replace(k, value=k.value * scale + offset) for k in base.keyframes),
    )


def _values(curve: AnimatedValue) -> list[float]:
    return [k.value for k in curve.keyframes] if curve.keyframes else [curve.static]


def _within(name: str, curve: AnimatedValue, bounds: tuple[float, float]) -> None:
    worst = max(_values(curve), key=lambda v: max(bounds[0] - v, v - bounds[1], 0.0))
    if not bounds[0] - 1e-6 <= worst <= bounds[1] + 1e-6:
        raise ReframeError(
            f"{name} が欄の範囲（{bounds[0]:g}〜{bounds[1]:g}）を越えます（{worst:.1f}）"
        )


def background_changes(
    params: Mapping[str, ParamValue],
    focus: tuple[AnimatedValue, AnimatedValue],
    window: CropWindow,
    placement: float,
    pivot: tuple[float, float],
) -> dict[str, AnimatedValue]:
    """背景のクリップの配置の線 枠の中を縦の画面いっぱいに映す

    横の画面の点 ``p``（中央が 0）が縦の画面の ``k x (p - 枠の中心)`` に行くように、
    拡大率を ``k / placement`` 倍、位置を ``k x (位置 - 枠の中心) + (k - placement) x 支点``
    にする ``placement`` は置き方の大きさの比（縦の画面での置いた大きさ / 横の画面での大きさ
    画素で置くクリップは 1、画面に収めるクリップは画面の形で変わる） ``pivot`` は支点の、
    画面の中央からのずれ（横の画面の画素）

    ``focus`` は枠の中心の線（クリップの頭からのフレーム） 位置か枠のどちらかが動かない
    ときだけ 1 次の式で書ける 両方が動くなら呼ぶ側が枠を止める
    """
    k = window.scale
    changes: dict[str, AnimatedValue] = {}
    for name, target, shift in (
        ("pos_x", focus[0], pivot[0]),
        ("pos_y", focus[1], pivot[1]),
    ):
        position = _as_curve(params.get(name), 0.0)
        if not target.keyframes:
            changed = affine(position, k, -k * target.static + (k - placement) * shift, 0.0)
        elif not position.keyframes:
            changed = affine(target, -k, k * position.static + (k - placement) * shift, 0.0)
        else:
            raise ReframeError("位置と枠の両方が動く所は 1 次の式で書けません 枠を止めてください")
        _within("位置", changed, POSITION_RANGE)
        changes[name] = changed
    scale = affine(params.get("scale"), k / placement, 0.0, 100.0)
    _within("拡大率", scale, SCALE_RANGE)
    changes["scale"] = scale
    return changes


def clamp_span(center: float, span: tuple[float, float], size: float) -> float:
    """``center`` から ``span``（左・右の端までのずれ）だけ広がる物が、幅 ``size`` の画面
    （中央が 0）に収まる中心 収まらないほど広い物は、画面の真ん中に広がりの真ん中を置く
    """
    low = -size / 2.0 - span[0]
    high = size / 2.0 - span[1]
    if low > high:
        return -(span[0] + span[1]) / 2.0
    return min(max(center, low), high)


def overlay_changes(
    params: Mapping[str, ParamValue],
    ratio: tuple[float, float],
    placement: float,
) -> dict[str, AnimatedValue]:
    """上に重ねた物の配置の線 縦の画面の同じ割合の所へ置き直し、大きさは画素のまま保つ

    ``ratio`` は（縦の画面の幅 / 横の画面の幅, 高さの比） 位置は画面の中央からの画素なので、
    比を掛けると画面の同じ割合の所になる（下の字幕は下のまま、右上のロゴは右上のまま）
    ``placement`` で割るのは、画面に収めて置く画像が縦の画面では小さく収まる分を戻すため
    """
    changes = {
        "pos_x": affine(params.get("pos_x"), ratio[0], 0.0, 0.0),
        "pos_y": affine(params.get("pos_y"), ratio[1], 0.0, 0.0),
    }
    if not math.isclose(placement, 1.0):
        changes["scale"] = affine(params.get("scale"), 1.0 / placement, 0.0, 100.0)
    return changes
