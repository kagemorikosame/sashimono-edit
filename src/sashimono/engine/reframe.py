"""縦動画への再構図（#334）を命令にする所

線の形は :mod:`sashimono.core.reframe` が組む ここは素材の置き方（画素で置く・画面に収める）・
配置の値の読み方・被写体の並び（:mod:`sashimono.engine.subject`）を知っている側の仕事を足して、
人の操作と同じ命令（解像度の変更と、配置とテキストの欄の値の差し替え）にする AI の道具も、
人のメニューの〔縦の作品に作り替える…〕も、ここを通す 入口ごとに書くと、同じ作品から違う縦の
作品ができる

作り替えるのはメインとすべてのシーン 解像度は作品で 1 つなので、シーンだけ横のまま残すと、
シーンを置いた所が縦の画面の真ん中に横長の絵で潰れる
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import pairwise

import numpy as np

from sashimono.core.commands.base import Command
from sashimono.core.commands.edit import SetResolution
from sashimono.core.commands.effects import ParamPath, SetParam
from sashimono.core.commands.fixed import FLIP_EFFECT_KIND
from sashimono.core.commands.scene import InScene
from sashimono.core.emphasis import fixed_transform, transform_commands
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Interpolation,
    Keyframe,
    MediaId,
    MediaItem,
    ParamValue,
    Project,
    SceneId,
    Timeline,
    Track,
    TrackKind,
    group_reaches,
)
from sashimono.core.reframe import (
    Area,
    CropWindow,
    ReframeError,
    Sample,
    Shot,
    ShotPlan,
    affine,
    background_changes,
    clamp_span,
    crop_window,
    focus_curves,
    overlay_changes,
    plan_shot,
    window_of,
)
from sashimono.core.timebase import Rounding, seconds_to_frame
from sashimono.effects import registry
from sashimono.effects.sources import TO_NATIVE_LAYOUT
from sashimono.effects.spec import TrackSpec
from sashimono.engine.cache.analyzer import SceneState
from sashimono.engine.gpu import fit_placement
from sashimono.engine.render.outline import has_outline, media_pixel_size, transform_values
from sashimono.engine.scenes import SceneSignal
from sashimono.engine.subject import SUBJECT_SECONDS

__all__ = [
    "BACKGROUND_COVER",
    "MAX_PAN_WINDOWS",
    "MIN_GATHERED",
    "MIN_STRENGTH",
    "Gathered",
    "NoPlaceError",
    "ReframePlan",
    "ReframeRequest",
    "gather_signals",
    "plan_reframe",
    "tracked_media",
]

#: 背景と見なす、絵が画面を覆う割合（横と縦のどちらも） 9 割を覆う絵は、上に重ねた物ではなく
#: 画面そのもの 少し縮めて縁を見せた動画も背景に数える
BACKGROUND_COVER = 0.9

#: 手がかりと見なす濃さの下限（:attr:`~sashimono.engine.subject.SubjectTrack.strength`）
#: 動きも肌色も無い絵（止まった風景）は 0.002 を下回る ここより薄い行で枠を動かすと、
#: 圧縮のざらつきを追って枠が揺れる
MIN_STRENGTH = 0.002

#: 手がかりと見なす、枠に集まっている度合いの下限（:func:`~sashimono.engine.subject.gathering`）
#: 散った動き（ゲームの画面の録画・画面じゅうが揺れる手持ち）の偏りを追うと、枠が跳ね回る
#: 人の映る実物でも中央値は 0.13〜0.3 ほどしかないので、外すのは 0.1 未満だけにし、残りは
#: 度合いを手がかりの重みに掛ける（散った行ほど枠を動かさない）
MIN_GATHERED = 0.1

#: 追うときに枠が動ける速さ（1 秒に枠の幅の何倍か） 速いと被写体の動きに付いていけるが、
#: 速すぎる枠は手で振り回したカメラのように見える 1 は横切って歩く人（画面の幅を 3 秒ほど）に
#: 付いていけ、実物のゲームの画面の録画で跳ね回る速さを半分ほどに抑えた
MAX_PAN_WINDOWS = 1.0

#: 被写体の並びの行と、クリップの時刻の差の上限（秒） 解析の届いていない所は手がかりに数えない
_NEAREST = SUBJECT_SECONDS

_TEXT = "text"


@dataclass(frozen=True, slots=True)
class ReframeRequest:
    """作り替えの頼み ``wrap_share`` は文字を折り返す幅（縦の画面の幅の % 0 は折り返さない）"""

    width: int
    height: int
    follow: str
    smooth_ms: int
    wrap_share: int


@dataclass(frozen=True, slots=True)
class ShotNote:
    """決めた場面 1 つ（返事に載せる） 位置は横の画面の画素（中央が 0 Y は上が正）"""

    scene_id: SceneId | None
    start: int
    end: int
    mode: str
    focus: tuple[float, float]
    points: int
    no_clue: bool
    clip_id: str


@dataclass(slots=True)
class ReframePlan:
    """組んだ命令と、何をどう変えたかの控え"""

    window: CropWindow
    commands: list[Command] = field(default_factory=list)
    shots: list[ShotNote] = field(default_factory=list)
    backgrounds: list[str] = field(default_factory=list)
    overlays: list[str] = field(default_factory=list)
    wrapped: list[str] = field(default_factory=list)
    #: 前からの位置の動きを残すため、そのクリップの間は枠を止めた
    held: list[str] = field(default_factory=list)
    #: 作り替えられなかったクリップと理由
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _timelines(project: Project) -> list[tuple[SceneId | None, Timeline]]:
    return [(None, project.timeline), *((scene.id, scene.timeline) for scene in project.scenes)]


def _placed(size: tuple[int, int], clip: Clip, canvas: tuple[int, int]) -> tuple[float, float]:
    """素材を置いた大きさ（配置を掛ける前 画面の画素） 描く側（``_media_placement``）と同じ"""
    if clip.native_size:
        return float(size[0]), float(size[1])
    placement = fit_placement(size[0], size[1], canvas[0], canvas[1])
    return placement.width, placement.height


def _transform_at(clip: Clip, local: int) -> dict[str, float | str | bool]:
    effect = fixed_transform(clip)
    if effect is None:
        return {}
    return transform_values(effect, local)


def _number(values: Mapping[str, float | str | bool], name: str, default: float) -> float:
    value = values.get(name, default)
    return float(value) if isinstance(value, int | float) else default


def _drawn_scale(values: Mapping[str, float | str | bool]) -> tuple[float, float]:
    whole = _number(values, "scale", 100.0) / 100.0
    return (
        whole * _number(values, "scale_x", 100.0) / 100.0,
        whole * _number(values, "scale_y", 100.0) / 100.0,
    )


def _pivot(
    values: Mapping[str, float | str | bool], placed: tuple[float, float]
) -> tuple[float, float]:
    """支点の、画面の中央からのずれ（画素 Y は上が正） シェーダの ``_TRANSFORM`` と同じ決まり"""
    horizontal = {"left": -placed[0] / 2.0, "right": placed[0] / 2.0}
    vertical = {"top": placed[1] / 2.0, "bottom": -placed[1] / 2.0}
    return (
        horizontal.get(str(values.get("pivot_h", "screen")), 0.0)
        + _number(values, "anchor_x", 0.0),
        vertical.get(str(values.get("pivot_v", "screen")), 0.0) + _number(values, "anchor_y", 0.0),
    )


def _flipped(clip: Clip) -> tuple[bool, bool]:
    for effect in clip.effects:
        if effect.kind == FLIP_EFFECT_KIND and effect.enabled:
            values = transform_values(effect, 0)
            return bool(values.get("horizontal", False)), bool(values.get("vertical", False))
    return False, False


@dataclass(frozen=True, slots=True)
class _Picture:
    """背景のクリップ 1 本の置き方"""

    scene_id: SceneId | None
    track: Track
    clip: Clip
    media: MediaItem
    placed_old: tuple[float, float]
    placed_new: tuple[float, float]

    @property
    def moving(self) -> bool:
        return self.media.has_video and self.media.duration > 0


def _is_background(
    project: Project, clip: Clip, canvas: tuple[int, int]
) -> tuple[MediaItem, tuple[int, int]] | None:
    if clip.media_id is None or clip.scene_id is not None:
        return None
    size = media_pixel_size(project, clip)
    media = project.find_media(clip.media_id)
    if size is None or media is None:
        return None
    placed = _placed(size, clip, canvas)
    scale_x, scale_y = _drawn_scale(_transform_at(clip, 0))
    covers = (
        placed[0] * abs(scale_x) >= canvas[0] * BACKGROUND_COVER
        and placed[1] * abs(scale_y) >= canvas[1] * BACKGROUND_COVER
    )
    return (media, size) if covers else None


def tracked_media(project: Project) -> list[MediaItem]:
    """被写体を追う素材（背景に置いた動画） 場面の解析を待つ相手"""
    canvas = project.settings.resolution
    found: dict[MediaId, MediaItem] = {}
    for _scene_id, timeline in _timelines(project):
        for track in timeline.tracks:
            for clip in track.clips:
                if not project.draws_picture(track, clip) or not has_outline(clip):
                    continue
                background = _is_background(project, clip, canvas)
                if background is None:
                    continue
                media = background[0]
                if media.has_video and media.duration > 0:
                    found.setdefault(media.id, media)
    return list(found.values())


def _stream_of(media: MediaItem, wanted: int) -> int | None:
    """デコーダと同じ選び方（番号が合う物、無ければ最初の映像）"""
    if not media.video_streams:
        return None
    if any(stream.index == wanted for stream in media.video_streams):
        return wanted
    return media.video_streams[0].index


def _inside(
    center: tuple[float, float],
    half: tuple[float, float],
    crop: tuple[float, float],
    turn: float,
) -> Area:
    """枠の中心を置ける範囲 枠の角が回した絵（中心 ``center`` 半分の大きさ ``half``）から
    はみ出さない所

    枠は画面の向きのまま 絵の向きで見ると、枠を回した四角になり、その外接の箱がはみ出さなければ
    枠の角もはみ出さない（絵は絵の向きで箱なので） 収まらなければ絵の中心だけにする
    """
    cos, sin = abs(math.cos(turn)), abs(math.sin(turn))
    need = (cos * crop[0] / 2.0 + sin * crop[1] / 2.0, sin * crop[0] / 2.0 + cos * crop[1] / 2.0)
    room = (max(0.0, half[0] - need[0]), max(0.0, half[1] - need[1]))
    return Area(center=center, reach=room, turn=turn)


class _Tracker:
    """背景のクリップ 1 本の、タイムラインのフレームごとの被写体と枠を置ける範囲"""

    def __init__(
        self,
        project: Project,
        picture: _Picture,
        signal: SceneSignal | None,
        window: CropWindow,
    ) -> None:
        self._project = project
        self.picture = picture
        self._window = window
        self._flip = _flipped(picture.clip)
        subject = signal.subject if signal is not None else None
        usable = (
            subject is not None
            and len(subject) > 0
            and picture.moving
            and (
                signal is None
                or signal.stream in (None, _stream_of(picture.media, picture.clip.stream_index))
            )
        )
        self._subject = subject if usable else None
        self._centers: tuple[np.ndarray, np.ndarray] | None = None
        self._gathered: np.ndarray | None = None
        if self._subject is not None:
            scale_x, scale_y = _drawn_scale(_transform_at(picture.clip, 0))
            crop = window.size
            shares = (
                min(1.0, crop[0] / max(1e-6, picture.placed_old[0] * abs(scale_x))),
                min(1.0, crop[1] / max(1e-6, picture.placed_old[1] * abs(scale_y))),
            )
            self._centers = self._subject.centers(shares)
            self._gathered = self._subject.gathered(shares, self._centers)

    def sample(self, frame: int) -> tuple[Sample, Area]:
        """タイムラインの ``frame`` の被写体と、枠の中心を置ける範囲"""
        clip = self.picture.clip
        local = frame - clip.timeline_start
        values = _transform_at(clip, local)
        scale_x, scale_y = _drawn_scale(values)
        placed = self.picture.placed_old
        pivot = _pivot(values, placed)
        position = (_number(values, "pos_x", 0.0), _number(values, "pos_y", 0.0))
        # 配置の式（シェーダの _TRANSFORM）は 支点 + 位置 + 回転(-角度) x 拡大 x (絵の点 - 支点)
        # （Y は上が正 正の角度で時計回りに回る） 回転を数えないと、傾けて置いた動画で、
        # 被写体のいる所を傾ける前の位置で数えて枠がずれる
        turn = -math.radians(_number(values, "rotation", 0.0))
        cos, sin = math.cos(turn), math.sin(turn)

        def placed_at(point: tuple[float, float]) -> tuple[float, float]:
            x, y = scale_x * (point[0] - pivot[0]), scale_y * (point[1] - pivot[1])
            return (
                pivot[0] + position[0] + cos * x - sin * y,
                pivot[1] + position[1] + sin * x + cos * y,
            )

        center = placed_at((0.0, 0.0))
        bounds = _inside(
            center,
            (abs(scale_x) * placed[0] / 2.0, abs(scale_y) * placed[1] / 2.0),
            self._window.size,
            turn,
        )
        weight = 0.0
        across, down = 0.5, 0.5
        if self._subject is not None and self._centers is not None:
            seconds = float(clip.picture_time(local, self._project.rate))
            times = self._subject.times
            index = int(np.clip(np.searchsorted(times, seconds), 0, len(times) - 1))
            if index > 0 and abs(times[index - 1] - seconds) < abs(times[index] - seconds):
                index -= 1
            if abs(float(times[index]) - seconds) <= _NEAREST:
                strength = float(self._subject.strength[index])
                gathered = 1.0 if self._gathered is None else float(self._gathered[index])
                if strength >= MIN_STRENGTH and gathered >= MIN_GATHERED:
                    weight = strength * gathered
                    across = float(self._centers[0][index])
                    down = float(self._centers[1][index])
        if self._flip[0]:
            across = 1.0 - across
        if self._flip[1]:
            down = 1.0 - down
        # 絵の中の点（左から across・上から down）の、支点からの画素（Y は上が正）
        x, y = placed_at(((across - 0.5) * placed[0], (0.5 - down) * placed[1]))
        return Sample(frame=frame, x=x, y=y, weight=weight), bounds

    def area_frames(self, start: int, end: int) -> list[int]:
        """枠を置ける範囲を数えるコマ（タイムラインのフレーム ``start`` 以上 ``end`` 未満）

        範囲の端は、絵の向きで見た位置と大きさの 1 次の式（回さなければ 位置 + 拡大 x 定数）
        なので、すべての欄のキーフレームのコマで区切った区間ごとに、次のどちらかなら端は区間の
        両端のコマで決まる 両端（とキーフレームの前後のコマ 止まる補間はキーフレームの 1 つ前の
        コマで切り替わる）だけを数えれば、どのコマの範囲の重なりとも同じになる

        - 区間で動く欄が直線（``LINEAR``）だけ 直線どうしを足しても直線
        - 区間で動く欄が 1 つで、その補間が両端の値の間から出ない（:func:`_stays_between`）
          イージング 3 種と、サイン・2 次〜5 次・指数・円・バウンス（止まる前に跳ねるが両端の間に
          収まる）の曲線と、制御点が 0〜1 のベジェ

        それ以外（行き過ぎて戻るバック・エラスティック、外へ出る制御点のベジェ、別々の曲線で
        動く 2 つ以上の欄）は区間の中に山ができうるので、その区間はどのコマも数える
        回転が動くと範囲の端が角度の三角関数で曲がるので、動く間はどのコマも数える
        配置が動かなければ頭の 1 コマで足りる 前はどのコマも数えていて、60fps の 10 分の場面で
        1.2 秒かかった（docs の表）
        """
        clip = self.picture.clip
        effect = fixed_transform(clip)
        moving = (
            {}
            if effect is None
            else {
                name: value
                for name, value in effect.params.items()
                if isinstance(value, AnimatedValue) and value.keyframes
            }
        )
        if not moving:
            return [start]
        if moving.keys() & {"rotation", "rotation_x", "rotation_y"}:
            return list(range(start, end))
        offset = clip.timeline_start
        curves = list(moving.values())
        edges = sorted({k.frame for curve in curves for k in curve.keyframes})
        wanted = {start, end - 1}
        for frame in edges:
            wanted.update((frame + offset - 1, frame + offset, frame + offset + 1))
        for left, right in pairwise(edges):
            if right + offset < start or left + offset >= end:
                continue
            active = [key for curve in curves if (key := _moving_key(curve, left, right))]
            if not active:
                continue
            if all(key.interpolation is Interpolation.LINEAR for key in active):
                continue
            if len(active) == 1 and _stays_between(active[0]):
                continue
            wanted.update(range(left + offset, right + offset + 1))
        return sorted(frame for frame in wanted if start <= frame < end)

    def home(self, frame: int) -> tuple[float, float]:
        return self.sample(frame)[1].center

    def cuts(self, signal: SceneSignal | None) -> set[int]:
        """素材の場面の切り替わりが、このクリップで出るタイムラインのフレーム"""
        clip = self.picture.clip
        if signal is None or self._subject is None:
            return set()
        frames: set[int] = set()
        rate = self._project.rate
        for cut in signal.cuts():
            moment = cut.seconds
            if moment <= clip.source_in or (clip.hold_at is not None and moment > clip.hold_at):
                continue
            offset = seconds_to_frame(
                (Fraction(str(moment)) - clip.source_in) / clip.speed, rate, Rounding.CEIL
            )
            if 0 < offset < clip.duration:
                frames.add(clip.timeline_start + offset)
        return frames


class NoPlaceError(ReframeError):
    """どこに枠を止めても、どこかのコマで背景の絵の外が映る

    そのクリップだけ作り替えずに残すと縦の画面で配置が崩れるので、作品ごと断る（ほかの
    クリップごとの断り :attr:`ReframePlan.skipped` とは分ける）
    """


def _common(areas: list[Area], clip: Clip) -> Area:
    """場面の間ずっと枠を置ける範囲 どの時刻の範囲にも入る所（重なり）

    寄りやパンで絵が動くと範囲も動く 絵の向きで見た横と縦の区間の重なりを取り、その真ん中を
    中心にする 頭の範囲の中心のまま幅だけ縮めると、大きくパンする絵で頭の中心が後の時刻の
    範囲から外れ、縦の画面の端に黒い帯が出た（PR #388 の Codex の指摘） 回す角度は頭の物を使う
    （回しながら動く絵では近い値） 重ならなければ、どこに止めてもどこかの時刻で絵の外が映るので断る
    """
    first = areas[0]
    cos, sin = math.cos(first.turn), math.sin(first.turn)
    low = [-math.inf, -math.inf]
    high = [math.inf, math.inf]
    for area in areas:
        across = cos * area.center[0] + sin * area.center[1]
        down = -sin * area.center[0] + cos * area.center[1]
        for axis, middle in enumerate((across, down)):
            low[axis] = max(low[axis], middle - area.reach[axis])
            high[axis] = min(high[axis], middle + area.reach[axis])
    if low[0] > high[0] + 1e-6 or low[1] > high[1] + 1e-6:
        raise NoPlaceError(
            f"背景のクリップ {clip.id} は位置や拡大の動きが大きく、場面の間ずっと絵の中に収まる"
            "枠の位置がありません どこに止めても縦の画面の端に絵の外が映ります 動きを小さくするか、"
            "クリップを分けてから呼んでください"
        )
    across = (low[0] + high[0]) / 2.0
    down = (low[1] + high[1]) / 2.0
    return Area(
        center=(cos * across - sin * down, sin * across + cos * down),
        reach=(max(0.0, (high[0] - low[0]) / 2.0), max(0.0, (high[1] - low[1]) / 2.0)),
        turn=first.turn,
    )


def _moving_key(curve: AnimatedValue, left: int, right: int) -> Keyframe | None:
    """``left``〜``right`` の区間で値を動かしているキーフレーム（区間の頭か前の点）

    区間で止まっていれば ``None``
    """
    keys = curve.keyframes
    before = [index for index, key in enumerate(keys) if key.frame <= left]
    if not before or before[-1] + 1 >= len(keys):
        return None
    key, following = keys[before[-1]], keys[before[-1] + 1]
    if following.frame < right or key.interpolation is Interpolation.HOLD:
        return None
    if math.isclose(key.value, following.value):
        return None
    return key


@functools.cache
def _stays_between(key: Keyframe) -> bool:
    """この点から次の点への補間が、両端の値の間から出ないか

    名前付きの曲線は形を式から読まず、細かく引いて確かめる（曲線を足しても、ここが自動で合う）
    """
    if key.interpolation in (Interpolation.LINEAR, Interpolation.HOLD):
        return True
    if key.interpolation is Interpolation.BEZIER:
        points = key.control_points or (0.0, 0.0, 1.0, 1.0)
        # 制御点が 0〜1 なら曲線は両端と制御点の囲む中にある
        return 0.0 <= points[1] <= 1.0 and 0.0 <= points[3] <= 1.0
    shape = AnimatedValue(0.0, (replace(key, frame=0, value=0.0), Keyframe(frame=1000, value=1.0)))
    values = [shape.at(frame) for frame in range(1001)]
    return min(values) >= -1e-9 and max(values) <= 1.0 + 1e-9


#: 透けるかもしれない合成の仕方以外の名前（ふつうに上へ重ねる）
_OPAQUE_BLEND = "normal"
#: 透明の欄（アルファ）を持つ画素の形式の目印 画像（PNG）や透過の動画は、絵の中に透ける所がある
_ALPHA_FORMATS = ("rgba", "bgra", "argb", "abgr", "yuva", "gbrap", "ya8", "ya16", "pal8")


def _see_through(project: Project, clip: Clip) -> bool:
    """手前の背景のクリップが、どこか少しでも透けるかもしれないか（奥の背景が映るか）

    どこが透けるかを正しく出すには絵を描くしかないので、透けうる物はどれも透けると数える
    （数えすぎると枠を置ける範囲が狭まるだけで、透けた所に絵の外が映るよりよい）

    - 不透明度が 1 でない時がある（フェードのキーフレームも）
    - 合成の仕方がふつうでない・下のクリップで切り抜く
    - 固定の欄でないエフェクトを積んでいる（マスク・クロマキー・透明度・切り抜きなど 種類を
      選り分けずに数える）
    - 素材が透明の欄を持つ（PNG の画像・透過の動画）
    """
    opacity = clip.opacity
    if opacity.keyframes or opacity.static < 1.0:
        return True
    if clip.blend_mode != _OPAQUE_BLEND or clip.clip_to_below:
        return True
    if any(effect.enabled and not effect.fixed for effect in clip.effects):
        return True
    media = project.find_media(clip.media_id) if clip.media_id is not None else None
    if media is None:
        return False
    stream = next(
        (s for s in media.video_streams if s.index == clip.stream_index),
        media.video_streams[0] if media.video_streams else None,
    )
    pixel_format = (stream.pixel_format or "").lower() if stream is not None else ""
    return any(pixel_format.startswith(mark) for mark in _ALPHA_FORMATS)


def _visible(timeline: Timeline) -> set[int]:
    return {id(track) for track in timeline.active_picture_tracks()}


def _shots(
    project: Project,
    timeline: Timeline,
    trackers: list[_Tracker],
    signals: Mapping[MediaId, SceneSignal],
) -> list[tuple[Shot, _Tracker]]:
    """背景の絵がいちばん手前の物の切り替わりと、素材の場面の切り替わりで区切った場面"""
    visible = _visible(timeline)
    order = {id(track): index for index, track in enumerate(timeline.tracks)}
    candidates = [t for t in trackers if id(t.picture.track) in visible and t.picture.clip.enabled]
    boundaries: set[int] = set()
    cut_frames: set[int] = set()
    for tracker in candidates:
        clip = tracker.picture.clip
        boundaries.update((clip.timeline_start, clip.timeline_end))
        cut_frames.update(tracker.cuts(signals.get(tracker.picture.media.id)))
    events = sorted(boundaries | cut_frames)

    def front(frame: int) -> _Tracker | None:
        showing = [t for t in candidates if t.picture.clip.contains(frame)]
        if not showing:
            return None
        return max(showing, key=lambda t: order[id(t.picture.track)])

    def through(frame: int) -> list[_Tracker]:
        """いちばん手前の背景が透けるとき、奥に見える背景（手前から順に 透けない物で止める）"""
        showing = sorted(
            (t for t in candidates if t.picture.clip.contains(frame)),
            key=lambda t: order[id(t.picture.track)],
            reverse=True,
        )
        seen: list[_Tracker] = []
        for upper, lower in pairwise(showing):
            if not _see_through(project, upper.picture.clip):
                break
            seen.append(lower)
        return seen

    spans: list[tuple[int, int, _Tracker]] = []
    for start, end in pairwise(events):
        if end <= start:
            continue
        owner = front(start)
        if owner is None:
            continue
        if spans and spans[-1][2] is owner and spans[-1][1] == start and start not in cut_frames:
            spans[-1] = (spans[-1][0], end, owner)
        else:
            spans.append((start, end, owner))
    step = max(1, round(float(project.rate.fps) * SUBJECT_SECONDS))
    shots: list[tuple[Shot, _Tracker]] = []
    for start, end, owner in spans:
        samples = [owner.sample(frame)[0] for frame in range(start, end, step)]
        # 枠を置ける範囲は、範囲がいちばん狭まりうるコマを全部数える（:meth:`_Tracker.area_frames`）
        # 被写体と同じ 0.25 秒ごとだけだと、場面の最後のコマやキーフレームの折れ目を数えず、
        # ぎりぎり収まるパンでその数コマだけ縦の画面の端に絵の外が映った
        # （PR #388 の CodeRabbit の指摘）
        frames = owner.area_frames(start, end)
        areas = [owner.sample(frame)[1] for frame in frames]
        # 手前の背景が透ける間は奥の背景も映る 奥の背景の枠も同じ枠なので、奥の絵の範囲にも収める
        # 収めないと、半透明の手前の絵の向こうに奥の絵の外（黒）が透けて見える 範囲の分かれ目は
        # 場面の中の背景の出入り（:data:`boundaries`）で、その間は奥に見える物が変わらない
        cuts_inside = [frame for frame in boundaries if start < frame < end]
        for head, tail in pairwise(sorted({start, end, *cuts_inside})):
            for lower in through(head):
                clip = lower.picture.clip
                low = max(head, clip.timeline_start)
                high = min(tail, clip.timeline_end)
                if low < high:
                    areas.extend(lower.sample(frame)[1] for frame in lower.area_frames(low, high))
        area = _common(areas, owner.picture.clip)
        left, right = area.center[0] - area.reach[0], area.center[0] + area.reach[0]
        bottom, top = area.center[1] - area.reach[1], area.center[1] + area.reach[1]
        shots.append(
            (
                Shot(
                    start=start,
                    end=end,
                    samples=tuple(samples),
                    bounds=(left, right, bottom, top),
                    home=owner.home(start),
                    area=area,
                ),
                owner,
            )
        )
    return shots


def _animated(value: ParamValue | None) -> bool:
    return isinstance(value, AnimatedValue) and bool(value.keyframes)


def _changed(
    params: Mapping[str, ParamValue], name: str, value: AnimatedValue, default: float
) -> bool:
    before = params.get(name)
    if isinstance(before, AnimatedValue):
        return before != value
    plain = (
        float(before)
        if isinstance(before, int | float) and not isinstance(before, bool)
        else default
    )
    return value.keyframes != () or not math.isclose(value.static, plain, abs_tol=1e-9)


_DEFAULTS = {"pos_x": 0.0, "pos_y": 0.0, "scale": 100.0, "anchor_x": 0.0, "anchor_y": 0.0}


def _transform_edit(clip: Clip, changes: dict[str, AnimatedValue]) -> list[Command]:
    effect = fixed_transform(clip)
    params = dict(effect.params) if effect is not None else {}
    kept = {
        name: value
        for name, value in changes.items()
        if _changed(params, name, value, _DEFAULTS.get(name, 0.0))
    }
    return transform_commands(clip, kept)


def _mean_focus(curve: AnimatedValue, start: int, end: int) -> AnimatedValue:
    if not curve.keyframes:
        return curve
    frames = range(start, max(start + 1, end))
    return AnimatedValue(static=sum(curve.at(frame) for frame in frames) / len(frames))


def _background_commands(
    tracker: _Tracker,
    focus: tuple[AnimatedValue, AnimatedValue],
    window: CropWindow,
    plan: ReframePlan,
) -> list[Command]:
    picture = tracker.picture
    clip = picture.clip
    effect = fixed_transform(clip)
    params: dict[str, ParamValue] = dict(effect.params) if effect is not None else {}
    if _animated(params.get("anchor_x")) or _animated(params.get("anchor_y")):
        raise ReframeError("配置の中心（中心 X・Y）がキーフレームで動いています")
    local = (
        window_of(focus[0], clip.timeline_start, clip.timeline_end),
        window_of(focus[1], clip.timeline_start, clip.timeline_end),
    )
    targets = list(local)
    held = [
        axis
        for axis, name in enumerate(("pos_x", "pos_y"))
        if _animated(params.get(name)) and targets[axis].keyframes
    ]
    if held:
        # 前からの位置の動きと枠の動きは 1 次の式で重ねられない 位置の動き（人が付けた寄りや
        # パン）を残し、このクリップの間は枠を止める 止める所は場面ごとの枠の平均を、クリップの
        # 全体で枠を置ける範囲（:func:`_common`）へ寄せた所 場面ごとには収まる枠でも、平均は
        # パンする絵ではほかの場面の範囲の外になりうり、その場面で縦の画面の端に絵の外が映った
        # （PR #388 の CodeRabbit の指摘）
        # 数えるコマは場面と同じ決まり（:meth:`_Tracker.area_frames`）
        start, end = clip.timeline_start, clip.timeline_end
        area = _common([tracker.sample(f)[1] for f in tracker.area_frames(start, end)], clip)
        mean = area.clamp(
            _mean_focus(focus[0], start, end).static, _mean_focus(focus[1], start, end).static
        )
        for axis in held:
            targets[axis] = AnimatedValue(static=mean[axis])
        if str(clip.id) not in plan.held:
            plan.held.append(str(clip.id))
    values = _transform_at(clip, 0)
    placement = picture.placed_new[0] / max(picture.placed_old[0], 1e-9)
    pivot = _pivot(values, picture.placed_old)
    changes = background_changes(params, (targets[0], targets[1]), window, placement, pivot)
    if not math.isclose(placement, 1.0):
        # 支点も置いた絵と同じ比で縮む（絵の端を選んだ支点は自動で付いてくるので、ずらし分だけ）
        for name in ("anchor_x", "anchor_y"):
            if params.get(name) is not None:
                changes[name] = affine(params.get(name), placement, 0.0, 0.0)
    return [
        *_transform_edit(clip, changes),
        *_pixel_commands(clip, placement, window.scale),
    ]


def _pixel_commands(clip: Clip, before: float, after: float) -> list[Command]:
    """エフェクトの画素で決める値（ぼかしの強さ・影のずれ・揺れの幅など）を直す命令

    固定の配置より前のエフェクトは置いた絵の上で掛かり、配置の拡大と一緒に拡がる 画面に収めて
    置く古いクリップは、縦の画面で置いた絵が ``before``（置き方の比）倍に縮むので、同じ比で
    縮めないと、拡大し直したときにぼかしや影のずれだけが 1 / 置き方の比 倍に太る
    固定の配置より後ろのエフェクトは画面の上で掛かるので、背景なら枠と一緒に ``after``（枠を
    拡げる倍率）倍にする 倍率が 1 の所は触らない 書いていない値は定義の既定の画素で数える
    """
    transform = fixed_transform(clip)
    order = next((i for i, e in enumerate(clip.effects) if e is transform), len(clip.effects))
    commands: list[Command] = []
    for index, effect in enumerate(clip.effects):
        if effect.fixed:
            continue
        factor = before if index < order else after
        definition = registry.get(effect.kind)
        if definition is None or math.isclose(factor, 1.0):
            continue
        for spec in definition.parameters:
            if not isinstance(spec, TrackSpec) or not spec.in_pixels:
                continue
            raw = effect.params.get(spec.name)
            default = float(spec.default)
            if raw is not None and not isinstance(raw, AnimatedValue | int | float):
                continue
            commands.append(
                SetParam(
                    ParamPath.of_effect(clip.id, effect.id, spec.name),
                    affine(raw, factor, 0.0, default),
                )
            )
    return commands


def _static(value: ParamValue | None, default: float) -> float | None:
    """動かない値ならその数 動く値は ``None``"""
    if isinstance(value, AnimatedValue):
        return None if value.keyframes else value.static
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default if value is None else None


def _overlay_media_commands(
    project: Project,
    clip: Clip,
    window: CropWindow,
) -> list[Command]:
    """上に重ねた画像・動画 同じ割合の所へ置き、縦の画面からはみ出すほど大きければ縮めて収める"""
    old, new = window.old, window.new
    ratio = (new[0] / old[0], new[1] / old[1])
    effect = fixed_transform(clip)
    params: dict[str, ParamValue] = dict(effect.params) if effect is not None else {}
    size = media_pixel_size(project, clip)
    placement = 1.0
    placed_new: tuple[float, float] | None = None
    if size is not None:
        placed_old = _placed(size, clip, old)
        placed_new = _placed(size, clip, new)
        placement = placed_new[0] / max(placed_old[0], 1e-9)
    changes = overlay_changes(params, ratio, placement)
    scale = _static(changes.get("scale", params.get("scale")), 100.0)
    x, y = _static(changes["pos_x"], 0.0), _static(changes["pos_y"], 0.0)
    if placed_new is not None and scale is not None and x is not None and y is not None:
        values = _transform_at(clip, 0)
        unit_x = _number(values, "scale_x", 100.0) / 100.0
        unit_y = _number(values, "scale_y", 100.0) / 100.0
        width = placed_new[0] * scale / 100.0 * abs(unit_x)
        height = placed_new[1] * scale / 100.0 * abs(unit_y)
        shrink = min(1.0, new[0] / max(width, 1e-9), new[1] / max(height, 1e-9))
        if shrink < 1.0:
            scale *= shrink
            width, height = width * shrink, height * shrink
            changes["scale"] = AnimatedValue(static=scale)
        changes["pos_x"] = AnimatedValue(static=clamp_span(x, (-width / 2, width / 2), new[0]))
        changes["pos_y"] = AnimatedValue(static=clamp_span(y, (-height / 2, height / 2), new[1]))
    return [*_transform_edit(clip, changes), *_pixel_commands(clip, placement, 1.0)]


def _widening(clip: Clip) -> float:
    """配置が文字を横へ広げる倍率 キーフレームで動く物はいちばん大きい値"""
    effect = fixed_transform(clip)
    total = 1.0
    if effect is None:
        return total
    for name in ("scale", "scale_x"):
        value = effect.params.get(name)
        if isinstance(value, AnimatedValue):
            largest = max((abs(k.value) for k in value.keyframes), default=abs(value.static))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            largest = abs(float(value))
        else:
            largest = 100.0
        total *= largest / 100.0
    return total


def _text_commands(
    clip: Clip, window: CropWindow, wrap_share: int, plan: ReframePlan
) -> list[Command]:
    """テキスト 同じ割合の所へ置き、縦の画面の幅で折り返し、はみ出す位置なら内側へ寄せる

    折り返しの幅は字を組むときの画素で、配置の拡大は後から掛かる 拡大の分だけ幅を縮める
    （字幕の見た目を着せるときの決まり :mod:`sashimono.compat.subtitle_looks` と同じ）
    AviUtl2 の組み方は折り返さないので、設定パネルの〔組み方を「標準」にする〕と同じ値で替える
    """
    old, new = window.old, window.new
    ratio = (new[0] / old[0], new[1] / old[1])
    source = clip.source
    assert source is not None
    effect = fixed_transform(clip)
    params: dict[str, ParamValue] = dict(effect.params) if effect is not None else {}
    changes = overlay_changes(params, ratio, 1.0)
    source_changes: dict[str, ParamValue] = {}
    own = dict(source.params)
    for name, share in (("pos_x", ratio[0]), ("pos_y", ratio[1])):
        moved = affine(own.get(name), share, 0.0, 0.0)
        if _changed(own, name, moved, 0.0):
            source_changes[name] = moved
    vertical = own.get("vertical")
    upright = vertical is True or (
        isinstance(vertical, int) and not isinstance(vertical, bool) and vertical != 0
    )
    widening = max(_widening(clip), 1e-6)
    limit = new[0] * wrap_share / 100.0 / widening
    wrap = own.get("wrap_width")
    current = _static(wrap, 0.0)
    span_width: float | None = None
    if not upright and wrap_share > 0 and current is not None:
        if current <= 0 or current > limit:
            source_changes["wrap_width"] = AnimatedValue(static=round(limit))
            if own.get("layout") == "aviutl":
                source_changes.update(TO_NATIVE_LAYOUT.switch(own))
            plan.wrapped.append(str(clip.id))
            span_width = round(limit) * widening
        else:
            span_width = current * widening
    commands = _transform_edit(clip, changes)
    if span_width is not None:
        anchor = str(source_changes.get("anchor", own.get("anchor", "center")))
        span = {"left": (0.0, span_width), "right": (-span_width, 0.0)}.get(
            anchor, (-span_width / 2, span_width / 2)
        )
        own_x = _static(source_changes.get("pos_x", own.get("pos_x")), 0.0)
        frame_x = _static(changes["pos_x"], 0.0)
        if own_x is not None and frame_x is not None:
            placed = clamp_span(own_x + frame_x, span, new[0])
            if not math.isclose(placed, own_x + frame_x):
                commands = _transform_edit(
                    clip, {**changes, "pos_x": AnimatedValue(static=placed - own_x)}
                )
    commands.extend(
        SetParam(ParamPath.of_source(clip.id, name), value)
        for name, value in source_changes.items()
    )
    return commands


#: グループ制御の配置の欄と、動かしていないときの値
_GROUP_MOVES = {"pos_x": 0.0, "pos_y": 0.0, "rotation": 0.0, "rotation_x": 0.0, "rotation_y": 0.0}


def _moved(value: ParamValue | None, still: float) -> bool:
    return _static(value, still) != still


def _group_commands(
    timeline: Timeline,
    track: Track,
    group: Clip,
    backgrounds: set[ClipId],
    window: CropWindow,
) -> list[Command]:
    """グループ制御の配置 受け持つ物と同じく、縦の画面の同じ割合の所へ置き直す

    グループ制御の配置は受け持つクリップの配置の後ろへ足される（:mod:`sashimono.engine.render.groups`）
    重ねた物は位置を画面の比で置き直すので、グループの位置も同じ比にすれば、合わせた位置も同じ
    割合になる（拡大は画面の中央を中心に掛かるので、比で置き直した位置にもそのまま効く）
    外すと、受け持つ物だけが新しい画面の比へ移り、グループの位置は元の画素のまま残ってずれる
    （PR #388 の Codex の指摘）

    次の 2 つは 1 次の式で置き直せないので、作品を変える前に断る
    - 背景の動画や画像を受け持つ 背景は枠の中心をクリップの位置から引くが、グループの位置は
      その後ろで足されるので、枠の動きをグループの位置へ移さないと枠が合わない
    - 回す 回した後の位置を横と縦で別の比にすると、重ねた物どうしの並びが崩れる
    """
    effect = fixed_transform(group)
    if effect is None or not effect.enabled:
        return []
    params = effect.params
    if not any(_moved(params.get(name), still) for name, still in _GROUP_MOVES.items()):
        return []
    pictures = timeline.picture_tracks()
    reached = [
        clip
        for other in pictures
        if group_reaches(pictures, track.id, group, other.id)
        for clip in other.clips
        if clip.overlaps(group.timeline_start, group.timeline_end)
    ]
    if any(clip.id in backgrounds for clip in reached):
        raise ReframeError(
            f"グループ制御 {group.id} が背景の動画や画像を位置や回転で動かしています 縦の枠は"
            "背景のクリップの位置で合わせるので、グループの動きと重ねると枠がずれます"
            " グループ制御の位置と回転を既定に戻すか、グループ制御から背景を外してから"
            "呼んでください"
        )
    if any(_moved(params.get(name), 0.0) for name in ("rotation", "rotation_x", "rotation_y")):
        raise ReframeError(
            f"グループ制御 {group.id} が回っています 回した後の位置を横と縦で別の比に置き直すと、"
            "受け持つ物どうしの並びが崩れます 回転を既定に戻してから呼んでください"
        )
    old, new = window.old, window.new
    changes = {
        "pos_x": affine(params.get("pos_x"), new[0] / old[0], 0.0, 0.0),
        "pos_y": affine(params.get("pos_y"), new[1] / old[1], 0.0, 0.0),
    }
    return _transform_edit(group, changes)


def _in(scene_id: SceneId | None, commands: Iterable[Command]) -> list[Command]:
    if scene_id is None:
        return list(commands)
    return [InScene(scene_id, command) for command in commands]


def plan_reframe(
    project: Project, request: ReframeRequest, signals: Mapping[MediaId, SceneSignal]
) -> ReframePlan:
    """横の作品を ``request`` の大きさに作り替える命令を組む

    ``signals`` は背景の動画の場面の解析（被写体の並びを含む） 無い素材は真ん中を切り出す
    ロックしたトラックで変える物があれば、何も組まずに断る（人が触らせないために掛けた物）
    """
    old = project.settings.resolution
    new = (request.width, request.height)
    window = crop_window(old, new)
    plan = ReframePlan(window=window)
    plan.commands.append(SetResolution(request.width, request.height))
    smooth_frames = max(1, round(float(project.rate.fps) * request.smooth_ms / 1000))
    max_step = window.size[0] * MAX_PAN_WINDOWS / float(project.rate.fps)
    locked: list[str] = []
    for scene_id, timeline in _timelines(project):
        trackers: list[_Tracker] = []
        overlays: list[tuple[Track, Clip]] = []
        for track in timeline.tracks:
            if track.kind is TrackKind.AUDIO:
                continue
            for clip in track.clips:
                if not project.draws_picture(track, clip):
                    continue
                if clip.scene_id is None and not has_outline(clip):
                    continue
                background = _is_background(project, clip, old)
                if background is not None:
                    media, size = background
                    picture = _Picture(
                        scene_id,
                        track,
                        clip,
                        media,
                        _placed(size, clip, old),
                        _placed(size, clip, new),
                    )
                    trackers.append(_Tracker(project, picture, signals.get(media.id), window))
                else:
                    overlays.append((track, clip))
        plans: list[ShotPlan] = []
        for shot, owner in _shots(project, timeline, trackers, signals):
            shot_plan = plan_shot(
                shot, request.follow, window.size, smooth_frames, max_step=max_step
            )
            plans.append(shot_plan)
            plan.shots.append(
                ShotNote(
                    scene_id=scene_id,
                    start=shot_plan.start,
                    end=shot_plan.end,
                    mode=shot_plan.mode,
                    focus=(shot_plan.points[0][1], shot_plan.points[0][2]),
                    points=len(shot_plan.points),
                    no_clue=shot_plan.no_clue,
                    clip_id=str(owner.picture.clip.id),
                )
            )
        focus = focus_curves(plans) if plans else (AnimatedValue(), AnimatedValue())
        for tracker in trackers:
            picture = tracker.picture
            try:
                commands = _background_commands(tracker, focus, window, plan)
            except NoPlaceError:
                raise
            except ReframeError as exc:
                plan.skipped.append((str(picture.clip.id), str(exc)))
                continue
            if commands and picture.track.locked:
                locked.append(picture.track.name)
            plan.commands.extend(_in(scene_id, commands))
            plan.backgrounds.append(str(picture.clip.id))
        for track, clip in overlays:
            if clip.source is not None and clip.source.kind == _TEXT:
                commands = _text_commands(clip, window, request.wrap_share, plan)
            else:
                commands = _overlay_media_commands(project, clip, window)
            if not commands:
                continue
            if track.locked:
                locked.append(track.name)
            plan.commands.extend(_in(scene_id, commands))
            plan.overlays.append(str(clip.id))
        background_ids = {tracker.picture.clip.id for tracker in trackers}
        for track in timeline.tracks:
            for clip in track.clips:
                if not clip.is_group:
                    continue
                commands = _group_commands(timeline, track, clip, background_ids, window)
                if not commands:
                    continue
                if track.locked:
                    locked.append(track.name)
                plan.commands.extend(_in(scene_id, commands))
                plan.overlays.append(str(clip.id))
    if locked:
        names = "・".join(dict.fromkeys(locked))
        raise ReframeError(
            f"ロックされたトラック（{names}）のクリップも動かす必要があります"
            " 解像度は作品全体で 1 つなので、残すと縦の画面で位置がずれます"
            " ロックを解いてから呼んでください"
        )
    return plan


@dataclass(slots=True)
class Gathered:
    """背景の動画ごとの解析の様子 AI の道具と人のメニューが同じ分け方をする"""

    #: 読み終えた（か、壊れていて読めた所までの）並び
    signals: dict[MediaId, SceneSignal] = field(default_factory=dict)
    #: まだ読んでいる 途中までの並びで作ると、読んでいない所の枠が真ん中に止まる
    waiting: list[MediaId] = field(default_factory=list)
    #: 読めなかった（失敗した） 被写体を追わずに真ん中を切り出す
    untracked: list[MediaId] = field(default_factory=list)


def gather_signals(
    media: Iterable[MediaItem], state_of: Callable[[MediaItem], SceneState]
) -> Gathered:
    """場面の解析の様子を、使える・待つ・追わないに分ける"""
    gathered = Gathered()
    for item in media:
        state = state_of(item)
        signal = state.signal
        if signal is not None and (signal.complete or signal.broken):
            gathered.signals[item.id] = signal
        elif state.running:
            gathered.waiting.append(item.id)
        else:
            gathered.untracked.append(item.id)
    return gathered
