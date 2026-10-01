"""プレビューで絵を直接動かすときの、掴む所の決め方と命令の作り方

画面（ウィジェット）から切り離してある 掴んだ所から値を出す計算と、値から命令を作る
決まりは、窓を作らずに試験で押さえたい 画面の側（:mod:`sashimono.ui.preview`）は
マウスの座標を渡して、返った命令を流すだけにする

座標は合成の画素（左上が原点、Y は下が正 :class:`~sashimono.engine.render.outline.Outline`
と同じ向き） 設定の Y は上が正なので、値へ直すところで向きを返す
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from sashimono.core.commands import AddEffect, Command, ParamPath, SetKeyframe, SetParam
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, fixed_effect
from sashimono.core.model import AnimatedValue, Clip, ClipId, Effect, Project
from sashimono.effects import registry
from sashimono.effects.spec import TrackSpec
from sashimono.engine.render.outline import Outline, Point, has_outline, transform_values

__all__ = [
    "KEYFRAME_DRAG_AT_PLAYHEAD",
    "KEYFRAME_DRAG_CHOICES",
    "KEYFRAME_DRAG_MODES",
    "KEYFRAME_DRAG_SHIFT_ALL",
    "ROTATION_STEP",
    "Grip",
    "Hit",
    "clamp_value",
    "fixed_transform",
    "hit_test",
    "inside",
    "moved_values",
    "pick_clip",
    "rotated_values",
    "rotation_knob",
    "scaled_values",
    "start_values",
    "transform_commands",
    "turn_between",
    "value_commands",
]

#: キーフレームのある値を動かしたら、再生ヘッドの所へ点を打つ（利用者の決定の既定）
KEYFRAME_DRAG_AT_PLAYHEAD = "playhead"
#: キーフレームのある値を動かしたら、全部の点を同じだけずらす
KEYFRAME_DRAG_SHIFT_ALL = "shift"
KEYFRAME_DRAG_MODES = (KEYFRAME_DRAG_AT_PLAYHEAD, KEYFRAME_DRAG_SHIFT_ALL)
#: 設定の画面に出す言葉
KEYFRAME_DRAG_CHOICES: tuple[tuple[str, str], ...] = (
    (KEYFRAME_DRAG_AT_PLAYHEAD, "再生ヘッドの所へ点を打つ（既定）"),
    (KEYFRAME_DRAG_SHIFT_ALL, "全部の点を同じだけずらす"),
)

#: Shift を押して回すときの刻み（度）
ROTATION_STEP = 15.0


class Grip(Enum):
    """掴んだ所"""

    #: 枠の中 位置を動かす
    MOVE = "move"
    #: 角 拡大率を変える
    SCALE = "scale"
    #: 枠の外の角の近くと、回転の掴み所 回す
    ROTATE = "rotate"
    #: 辺の真ん中（部分フィルタの範囲だけ） その向きの大きさだけを変える
    EDGE = "edge"


@dataclass(frozen=True, slots=True)
class Hit:
    grip: Grip
    #: 角を掴んだときの角の番号（左上・右上・右下・左下） 辺なら辺の番号（上・右・下・左）
    #: それ以外は -1
    corner: int = -1


def inside(corners: Sequence[Point], point: Point) -> bool:
    """``point`` が四隅の囲む範囲の中か 配置は平行四辺形にしか写さないので凸で数える"""
    sign = 0.0
    for index in range(4):
        ax, ay = corners[index]
        bx, by = corners[(index + 1) % 4]
        cross = (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
        if abs(cross) < 1e-9:
            continue
        if sign == 0.0:
            sign = cross
        elif (cross > 0) != (sign > 0):
            return False
    return sign != 0.0


def rotation_knob(corners: Sequence[Point], reach: float) -> tuple[Point, Point] | None:
    """回転の掴み所 上の辺の中点と、そこから外へ ``reach`` 離した所 形が潰れていれば ``None``"""
    center_x = sum(x for x, _ in corners) / 4.0
    center_y = sum(y for _, y in corners) / 4.0
    mid = ((corners[0][0] + corners[1][0]) / 2.0, (corners[0][1] + corners[1][1]) / 2.0)
    dx, dy = mid[0] - center_x, mid[1] - center_y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    return mid, (mid[0] + dx / length * reach, mid[1] + dy / length * reach)


def hit_test(
    corners: Sequence[Point], point: Point, *, grip: float, reach: float, knob: float
) -> Hit | None:
    """``point`` で掴むのは何か 何も無ければ ``None``

    座標は画面の上の同じ物差し（ウィジェットの座標）で渡す 掴める幅を合成の画素で
    数えると、プレビューを小さくしたときに角がほとんど掴めなくなる
    ``grip`` は角を掴める半径、``reach`` は角の外で回せる半径、``knob`` は回転の掴み所の
    長さ 角 → 回転の掴み所 → 枠の中 → 角の外 の順に見る 角は枠の中と重なるので先に見る
    """
    for index, (x, y) in enumerate(corners):
        if math.hypot(point[0] - x, point[1] - y) <= grip:
            return Hit(Grip.SCALE, index)
    found = rotation_knob(corners, knob)
    if found is not None:
        tip = found[1]
        if math.hypot(point[0] - tip[0], point[1] - tip[1]) <= grip:
            return Hit(Grip.ROTATE)
    if inside(corners, point):
        return Hit(Grip.MOVE)
    for x, y in corners:
        if math.hypot(point[0] - x, point[1] - y) <= reach:
            return Hit(Grip.ROTATE)
    return None


def fixed_transform(clip: Clip) -> Effect | None:
    """クリップが最初から持つ配置 前の版のクリップには無い"""
    return next((e for e in clip.effects if e.fixed and e.kind == TRANSFORM_EFFECT_KIND), None)


def start_values(clip: Clip, local_frame: int) -> dict[str, float]:
    """掴んだ時点の配置の数の値 配置が無ければ既定の値"""
    effect = fixed_transform(clip) or fixed_effect(TRANSFORM_EFFECT_KIND)
    return {
        name: float(value)
        for name, value in transform_values(effect, local_frame).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


#: 範囲（左・上・右・下 合成の画素、Y は下が正）
Box = tuple[float, float, float, float]

#: 吸い付いた先 ``("x", 位置)`` は縦の線、``("y", 位置)`` は横の線
Guide = tuple[str, float]


def bounding_box(corners: Sequence[Point]) -> Box:
    """四隅を囲む軸にそろった範囲 回した絵でも見えている端で揃える"""
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    return min(xs), min(ys), max(xs), max(ys)


def snap_targets(canvas: tuple[int, int], others: Sequence[Box]) -> tuple[list[float], list[float]]:
    """吸い付く先 画面の端と中央、ほかの物の端と中央（横の位置の並び, 縦の位置の並び）"""
    width, height = canvas
    xs = [0.0, width / 2.0, float(width)]
    ys = [0.0, height / 2.0, float(height)]
    for left, top, right, bottom in others:
        xs.extend((left, (left + right) / 2.0, right))
        ys.extend((top, (top + bottom) / 2.0, bottom))
    return xs, ys


def snap_offset(
    moving: Box, targets: tuple[Sequence[float], Sequence[float]], reach: float
) -> tuple[float, float, list[Guide]]:
    """``moving`` の端か中央を、``reach`` 以内で一番近い先へ寄せるずれ（横, 縦, 引く線）

    横と縦は別々に決める 横だけ近ければ横だけ吸い付く 同じ近さなら先に並べた物（画面）を取る
    """
    left, top, right, bottom = moving
    guides: list[Guide] = []

    def nearest(own: Sequence[float], candidates: Sequence[float], axis: str) -> float:
        best: tuple[float, float] | None = None
        for target in candidates:
            for edge in own:
                distance = abs(target - edge)
                if distance <= reach and (best is None or distance < abs(best[0])):
                    best = (target - edge, target)
        if best is None:
            return 0.0
        guides.append((axis, best[1]))
        return best[0]

    dx = nearest((left, (left + right) / 2.0, right), targets[0], "x")
    dy = nearest((top, (top + bottom) / 2.0, bottom), targets[1], "y")
    return dx, dy, guides


#: 配置のテンプレート 名前と、画面の横・縦のどこへ寄せるか（0 が左・上、1 が右・下）
ALIGNMENTS: tuple[tuple[str, str, float, float], ...] = (
    ("top_left", "左上", 0.0, 0.0),
    ("top", "上の中央", 0.5, 0.0),
    ("top_right", "右上", 1.0, 0.0),
    ("left", "左の中央", 0.0, 0.5),
    ("center", "中央", 0.5, 0.5),
    ("right", "右の中央", 1.0, 0.5),
    ("bottom_left", "左下", 0.0, 1.0),
    ("bottom", "下の中央", 0.5, 1.0),
    ("bottom_right", "右下", 1.0, 1.0),
)


def aligned_values(
    start: dict[str, float],
    box: Box,
    canvas: tuple[int, int],
    anchor: str,
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    """見えている範囲 ``box`` を画面の ``anchor`` の所へ寄せる X と Y

    範囲の端を画面の端へ付ける（中央なら範囲の中央を画面の中央へ） 回した絵は見えている
    範囲で揃える ``scale`` は :func:`moved_values` と同じ（合成の画素を画面の画素へ直す）
    """
    found = next((a for a in ALIGNMENTS if a[0] == anchor), None)
    if found is None:
        raise ValueError(f"知らない配置: {anchor}")
    _, _, across, down = found
    left, top, right, bottom = box
    width, height = canvas
    want_x = across * (width - (right - left)) + (right - left) / 2.0
    want_y = down * (height - (bottom - top)) + (bottom - top) / 2.0
    dx = (want_x - (left + right) / 2.0) / scale
    # 画面の Y は下が正、設定の Y は上が正
    dy = ((top + bottom) / 2.0 - want_y) / scale
    return {"pos_x": start.get("pos_x", 0.0) + dx, "pos_y": start.get("pos_y", 0.0) + dy}


def moved_values(
    start: dict[str, float],
    press: Point,
    current: Point,
    *,
    one_axis: bool,
    scale: float = 1.0,
) -> dict[str, float]:
    """枠の中を掴んで動かしたときの X と Y

    X・Y は画面（プロジェクトの解像度）の画素 配置は拡大と回転の後で位置を足すので、
    画面の画素 1 つが X・Y の 1 そのまま ``scale`` は合成の画素 1 つが画面の画素いくつ分かの
    逆数（:func:`~sashimono.engine.render.outline.canvas_scale`） 画質を落としたプレビューの
    合成の画素のまま足すと、マウスの半分・4 分の 1 しか絵が動かない
    ``one_axis`` なら動きの大きい方だけ（Shift） 画面の Y は下が正、設定の Y は上が正
    """
    dx = (current[0] - press[0]) / scale
    dy = (press[1] - current[1]) / scale
    if one_axis:
        if abs(dx) >= abs(dy):
            dy = 0.0
        else:
            dx = 0.0
    return {"pos_x": start.get("pos_x", 0.0) + dx, "pos_y": start.get("pos_y", 0.0) + dy}


def scaled_values(
    start: dict[str, float],
    corners: Sequence[Point],
    pivot: Point,
    press: Point,
    current: Point,
    *,
    separate: bool,
) -> dict[str, float]:
    """角を掴んで動かしたときの拡大率 ``corners`` は掴んだ時点の枠

    中心（``pivot``）から角までの距離の比で拡げる 縦の拡大率は横に掛ける比なので、
    縦横をそろえて拡げるときは触らない ``separate`` なら（Alt）絵の横と縦の向きへ
    分けて比を取る 回した絵でも絵の向きで分かれるよう、軸は枠の辺から取る
    """
    before = (press[0] - pivot[0], press[1] - pivot[1])
    after = (current[0] - pivot[0], current[1] - pivot[1])
    scale = start.get("scale", 100.0)
    scale_y = start.get("scale_y", 100.0)
    if not separate:
        ratio = _ratio(math.hypot(*after), math.hypot(*before))
        return {"scale": _clamp("scale", scale * ratio)}
    across = _unit(corners[0], corners[1])
    down = _unit(corners[0], corners[3])
    ratio_x = _ratio(abs(_dot(after, across)), abs(_dot(before, across)))
    ratio_y = _ratio(abs(_dot(after, down)), abs(_dot(before, down)))
    new_scale = _clamp("scale", scale * ratio_x)
    # 縦は「横に掛ける比」で持つ 横を変えたぶんを割り戻しておかないと、横だけ
    # 引いたのに縦まで伸びる
    height = scale * scale_y / 100.0 * ratio_y
    return {"scale": new_scale, "scale_y": _clamp("scale_y", height / new_scale * 100.0)}


def rotated_values(start: dict[str, float], turned: float, *, snap: bool) -> dict[str, float]:
    """回した角度（度 時計回りが正）を足した回転 ``snap`` なら 15 度刻みへ（Shift）"""
    rotation = start.get("rotation", 0.0) + turned
    if snap:
        rotation = round(rotation / ROTATION_STEP) * ROTATION_STEP
    return {"rotation": rotation}


def turn_between(pivot: Point, before: Point, after: Point) -> float:
    """``pivot`` の周りで ``before`` から ``after`` へ回った角度（度 時計回りが正 -180 から 180）

    画面の Y は下が正なので、atan2 の増える向きがそのまま時計回り 1 回の動きの差だけを
    返し、呼ぶ側で足していく 押した所との差で数えると、半周を越えた所で逆回りに跳ぶ
    """
    first = math.atan2(before[1] - pivot[1], before[0] - pivot[0])
    second = math.atan2(after[1] - pivot[1], after[0] - pivot[0])
    step = math.degrees(second - first)
    return (step + 180.0) % 360.0 - 180.0


def transform_commands(
    project: Project,
    clip_id: ClipId,
    changes: dict[str, float],
    frame: int,
    *,
    keyframes: str = KEYFRAME_DRAG_AT_PLAYHEAD,
) -> list[Command]:
    """配置の値を ``changes`` にする命令の一覧 1 回の取り消しで戻せるよう、まとめて流す

    - 配置を持たない前の版のクリップには、先に固定の配置を足す
    - 動かない値は :class:`SetParam` で差し替える
    - キーフレームのある値は :class:`SetParam` を使わない（点を全部捨てて止まった値になる）
      既定は再生ヘッドの所（クリップの中の時刻に収める）へ点を打つ 設定で「全部の点を
      同じだけずらす」を選んでいれば、その時刻の値との差を全部の点へ足す
    """
    located = project.timeline.locate_clip(clip_id)
    if located is None:
        return []
    _, clip = located
    local = min(max(frame - clip.timeline_start, 0), max(clip.duration - 1, 0))
    commands: list[Command] = []
    effect = fixed_transform(clip)
    if effect is None:
        effect = fixed_effect(TRANSFORM_EFFECT_KIND)
        commands.append(AddEffect(clip.id, effect))
    commands.extend(value_commands(clip.id, effect, changes, local, keyframes=keyframes))
    if len(commands) == 1 and isinstance(commands[0], AddEffect):
        # 値が何も変わらないなら、配置を足すだけの段を積まない
        return []
    return commands


def value_commands(
    clip_id: ClipId,
    effect: Effect,
    changes: dict[str, float],
    local: int,
    *,
    keyframes: str = KEYFRAME_DRAG_AT_PLAYHEAD,
) -> list[Command]:
    """``effect`` の数の値を ``changes`` にする命令 ``local`` はクリップの頭から数えた時刻

    配置（:func:`transform_commands`）と部分フィルタの範囲で同じ決まりを使う
    キーフレームのある値は、既定では ``local`` へ点を打つ（設定パネルの ◆ と同じ時刻）
    """
    commands: list[Command] = []
    for name, wanted in changes.items():
        # 設定パネルと同じ範囲に収める 範囲の外の値は、パネルで開いたときに端へ丸められる
        value = clamp_value(effect.kind, name, wanted)
        path = ParamPath.of_effect(clip_id, effect.id, name)
        current = effect.params.get(name)
        if isinstance(current, AnimatedValue) and current.is_animated:
            delta = value - current.at(local)
            if abs(delta) < 1e-9:
                # 元の所へ戻した 点を打つと、その時刻に無かった直線の点が入って、
                # 前後の曲線の出方まで変わる
                continue
            if keyframes == KEYFRAME_DRAG_SHIFT_ALL:
                commands.extend(
                    SetKeyframe(
                        path, point.frame, clamp_value(effect.kind, name, point.value + delta)
                    )
                    for point in current.keyframes
                )
            else:
                commands.append(SetKeyframe(path, local, value))
            continue
        if isinstance(current, AnimatedValue) and current.static == value:
            continue
        if current is None and value == _default(effect.kind, name):
            continue
        commands.append(SetParam(path, AnimatedValue(value)))
    return commands


def pick_clip(
    project: Project, frame: int, point: Point, outline_of: Callable[[Clip], Outline | None]
) -> ClipId | None:
    """``point`` に描かれている一番手前のクリップ 無ければ ``None``

    重なり順は描く側と同じ（トラックの並びの後ろほど手前） 絵を描かないクリップ（音だけ・
    無効）は外す 枠の中かどうかで見る 透けた所を掴んでも選べるのは、YMM4 と同じ
    """
    for track in reversed(project.timeline.active_picture_tracks()):
        clip = track.clip_at(frame)
        if clip is None or not clip.enabled or not has_outline(clip):
            continue
        if not project.draws_picture(track, clip):
            continue
        outline = outline_of(clip)
        if outline is not None and inside(outline.corners, point):
            return clip.id
    return None


def _ratio(after: float, before: float) -> float:
    # 中心のすぐそばを掴むと、わずかに動かしただけで何倍にもなる 1 画素より近ければ動かさない
    return after / before if before >= 1.0 else 1.0


def _unit(start: Point, end: Point) -> Point:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > 1e-9 else (0.0, 0.0)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _clamp(name: str, value: float) -> float:
    """配置の定義の範囲に収める 0 や負の拡大率は絵を消し、戻す手掛かりも無くなる"""
    return clamp_value(TRANSFORM_EFFECT_KIND, name, value)


def clamp_value(kind: str, name: str, value: float) -> float:
    """``kind`` のエフェクトの ``name`` の定義の範囲に収める"""
    definition = registry.get(kind)
    spec = definition.spec(name) if definition is not None else None
    if isinstance(spec, TrackSpec):
        return min(max(value, spec.minimum), spec.maximum)
    return value


def _default(kind: str, name: str) -> float | None:
    definition = registry.get(kind)
    spec = definition.spec(name) if definition is not None else None
    return spec.default if isinstance(spec, TrackSpec) else None
