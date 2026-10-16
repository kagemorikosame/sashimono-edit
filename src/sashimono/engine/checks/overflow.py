"""字幕（テキストのクリップ）が画面からはみ出す所を拾う（#337）

絵の画素からは、どの字が切れたのかが分からない（字が画面の外にあれば絵に出ないので）
プレビューの外枠と同じ計算（:func:`~sashimono.engine.render.outline.clip_outline`）で、
見えている字の範囲が画面のどこへ来るかを求め、画面（と余白）の外へ出る分を測る

1 本のクリップを全部のコマで測ると、長い字幕の多い作品で遅い 頭と終わりのコマと、
その間を :data:`SAMPLE_SECONDS` ごとに測る 位置が動く字幕でも 0.5 秒より短い間だけ
はみ出すことは少なく、はみ出す字幕はたいてい置いた位置か長さのせいで、どのコマでも同じ
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from sashimono.core.model import Clip, Project, Timeline, Track, controlling_groups
from sashimono.core.timebase import seconds_to_frame
from sashimono.engine.checks.scan import TEXT_OVERFLOW, CheckLimits, Run
from sashimono.engine.render.groups import grouped
from sashimono.engine.render.outline import Point, clip_outline
from sashimono.engine.render.renderer import MAX_SCENE_DEPTH, FrameRenderer

__all__ = ["SAMPLE_SECONDS", "OverflowScan"]

#: 1 本の字幕を測る間（秒）
SAMPLE_SECONDS = 0.5
#: はみ出しとみなす長さ（合成の画素） 外枠の計算の端数で、ちょうど端に届く字を拾わない
_SLACK = 0.5
#: 動く字幕とみなす、測った所の間の位置の違い（合成の画素） 流れるテロップのように、
#: わざと画面の外から入ってくる物は、はみ出しても直す物ではないことが多い
_MOVING = 2.0
#: 返事に載せる字の長さ 長い字幕を丸ごと載せると、一覧が字で埋まる
_TEXT_EXCERPT = 24


@dataclass(slots=True)
class _Seen:
    """1 本の字幕で見たはみ出し"""

    clip: Clip
    #: 字幕を描く、置いたシーンのクリップ（外から順） 今のタイムラインの字幕なら空
    scenes: tuple[str, ...] = ()
    first: int | None = None
    last: int = 0
    #: 一番大きなはみ出し（画面の画素）
    worst: float = 0.0
    sides: set[str] = field(default_factory=set)
    #: 画面の端の外へ出た（余白の中だけなら偽）
    beyond: bool = False
    #: 見えている割合の一番小さい値（0〜1）
    visible: float = 1.0
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


def _is_text(clip: Clip) -> bool:
    return clip.source is not None and clip.source.kind == "text" and clip.enabled


def _same(point: Point) -> Point:
    return point


def _bilinear(
    corners: tuple[Point, Point, Point, Point],
    canvas: tuple[int, int],
    place: Callable[[Point], Point],
) -> Callable[[Point], Point]:
    """シーンの絵の中の点を、外側の画面の点へ写す関数

    ``corners`` はシーンの絵の四隅（左上・右上・右下・左下）が行き着いた所 配置（移動・拡大・
    回転）は形を保つので、左上からの 2 辺の割合で写せば中の点も同じ所へ行く
    外側がさらにシーンの中なら ``place`` でもう 1 段外へ写す
    """
    width, height = max(canvas[0], 1), max(canvas[1], 1)
    (x0, y0), (x1, y1), _, (x3, y3) = corners

    def mapped(point: Point) -> Point:
        u, v = point[0] / width, point[1] / height
        return place((x0 + u * (x1 - x0) + v * (x3 - x0), y0 + u * (y1 - y0) + v * (y3 - y0)))

    return mapped


def _text_of(clip: Clip) -> str:
    source = clip.source
    if source is None:
        return ""
    value = source.params.get("text", "")
    text = " ".join(str(getattr(value, "static", value)).split())
    return text if len(text) <= _TEXT_EXCERPT else text[: _TEXT_EXCERPT - 1] + "…"


class OverflowScan:
    """字幕のはみ出しを、測るコマが来るたびに測る

    レンダラは検査の走り係の物を借りる（同じスレッドでだけ使う） 絵を作るのは CPU で、
    同じ設定の字幕なら描いたときに覚えた絵を使う
    """

    def __init__(self, project: Project, start: int, end: int, limits: CheckLimits) -> None:
        self._project = project
        self._limits = limits
        self._seen: dict[str, _Seen] = {}
        #: 測るコマ → そのコマで測る字幕と、置いたシーン（中の字幕はそのコマで探す）
        self._plan: dict[int, list[tuple[Track, Clip]]] = {}
        step = max(1, round(float(project.rate.fps) * SAMPLE_SECONDS))
        timeline = project.timeline
        for track in timeline.active_picture_tracks():
            for clip in track.clips:
                if not clip.enabled or not project.draws_picture(track, clip):
                    continue
                if not (_is_text(clip) or self._holds_text(clip, 0)):
                    continue
                first = max(start, clip.timeline_start)
                last = min(end, clip.timeline_end) - 1
                if last < first:
                    continue
                frames = set(range(first, last + 1, step))
                frames.add(last)
                for frame in frames:
                    self._plan.setdefault(frame, []).append((track, clip))

    def _holds_text(self, clip: Clip, depth: int) -> bool:
        """置いたシーンの中（入れ子も）に字幕があるか 無いシーンは測るコマに入れない"""
        if clip.scene_id is None or depth >= MAX_SCENE_DEPTH:
            return False
        scene = self._project.find_scene(clip.scene_id)
        if scene is None:
            return False
        return any(
            _is_text(inner) or self._holds_text(inner, depth + 1)
            for track in scene.timeline.active_picture_tracks()
            for inner in track.clips
        )

    @property
    def frames(self) -> frozenset[int]:
        """測るコマ"""
        return frozenset(self._plan)

    def measure(self, renderer: FrameRenderer, frame: int, canvas: tuple[int, int]) -> None:
        """``frame`` で測る字幕を測る ``canvas`` はレンダラの合成の大きさ"""
        for track, clip in self._plan.get(frame, ()):
            self._measure(
                renderer, self._project.timeline, track, clip, frame, frame, canvas, _same, (), 0
            )

    def _measure(
        self,
        renderer: FrameRenderer,
        timeline: Timeline,
        track: Track,
        clip: Clip,
        local: int,
        frame: int,
        canvas: tuple[int, int],
        place: Callable[[Point], Point],
        scenes: tuple[str, ...],
        depth: int,
    ) -> None:
        """``timeline`` の ``local`` のコマに描く ``clip`` を測る ``place`` は外側の配置

        置いたシーンは、中の時刻へ直して中の字幕を探す 中の字幕の位置は、シーンの絵を
        外側のクリップの配置（位置・大きさ・回転・グループ制御）で動かした所で測る
        レンダラと同じく、シーンは画面と同じ大きさの絵に描いてから 1 本のクリップとして置く
        """
        shown = self._grouped(timeline, track, clip, local)
        if clip.scene_id is not None:
            scene = self._project.find_scene(clip.scene_id)
            if scene is None or depth >= MAX_SCENE_DEPTH:
                return
            corners = self._scene_corners(shown, local, canvas)
            if corners is None:
                return
            inner_frame = seconds_to_frame(
                clip.picture_time(local - clip.timeline_start, self._project.rate),
                self._project.rate,
            )
            outer = _bilinear(corners, canvas, place)
            for inner_track in scene.timeline.active_picture_tracks():
                inner = inner_track.clip_at(inner_frame)
                if inner is None or not inner.enabled:
                    continue
                if not self._project.draws_picture(inner_track, inner):
                    continue
                if _is_text(inner) or inner.scene_id is not None:
                    self._measure(
                        renderer,
                        scene.timeline,
                        inner_track,
                        inner,
                        inner_frame,
                        frame,
                        canvas,
                        outer,
                        (*scenes, str(clip.id)),
                        depth + 1,
                    )
            return
        extent = renderer.content_extent(clip, local)
        if extent is None:
            return
        outline = clip_outline(self._project, shown, local, canvas=canvas, extent=extent)
        if outline is None:
            return
        points = [place(point) for point in outline.corners]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        self._note(clip, frame, (min(xs), min(ys), max(xs), max(ys)), canvas, scenes)

    def _scene_corners(
        self, clip: Clip, local: int, canvas: tuple[int, int]
    ) -> tuple[Point, Point, Point, Point] | None:
        """置いたシーンの絵（画面いっぱい）の四隅が、外側のどこへ来るか

        外枠の計算はシーンのクリップには枠を出さない（掴んで動かさない）ので、同じ配置を持つ
        画面いっぱいの絵として測る レンダラの ``_draw_nested`` も画面いっぱいに置いてから掛ける
        """
        width, height = canvas
        stand_in = replace(clip, scene_id=None, media_id=None)
        box = (0.0, 0.0, float(width), float(height))
        outline = clip_outline(
            self._project, stand_in, local, canvas=canvas, extent=(box, (width, height))
        )
        return None if outline is None else outline.corners

    def _grouped(self, timeline: Timeline, track: Track, clip: Clip, frame: int) -> Clip:
        """グループ制御で動かした後のクリップ プレビューの外枠と同じ当て方"""
        drawn = {item.id for item in timeline.active_picture_tracks()}
        groups = [
            (group_track, group)
            for group_track, group in controlling_groups(timeline.picture_tracks(), track.id, frame)
            if group_track.id in drawn
        ]
        return grouped(clip, groups, frame)

    def _note(
        self,
        clip: Clip,
        frame: int,
        box: tuple[float, float, float, float],
        canvas: tuple[int, int],
        scenes: tuple[str, ...] = (),
    ) -> None:
        width, height = canvas
        left, top, right, bottom = box
        margin_x = width * self._limits.safe_margin
        margin_y = height * self._limits.safe_margin
        # 画面の Y は下が正（合成の画素） 上へはみ出すと top が小さくなる
        overs = {
            "左": margin_x - left,
            "右": right - (width - margin_x),
            "上": margin_y - top,
            "下": bottom - (height - margin_y),
        }
        beyond = max(-left, right - width, -top, bottom - height) > _SLACK
        sides = {side for side, amount in overs.items() if amount > _SLACK}
        # 同じシーンを 2 か所に置けば、中の同じ字幕も 2 か所に出る 置いた所ごとに分けて数える
        seen = self._seen.setdefault("/".join((*scenes, str(clip.id))), _Seen(clip, scenes))
        seen.boxes.append(box)
        if not sides:
            return
        # 合成の画素を画面（プロジェクトの解像度）の画素へ戻して返す 画質を落として測っても、
        # 人が設定の欄で直す値と同じ尺度で読めるように
        scale = self._project.settings.width / max(width, 1)
        area = max((right - left) * (bottom - top), 1e-9)
        inside_w = max(0.0, min(right, width) - max(left, 0.0))
        inside_h = max(0.0, min(bottom, height) - max(top, 0.0))
        if seen.first is None:
            seen.first = frame
        seen.last = max(seen.last, frame)
        seen.worst = max(seen.worst, max(overs.values()) * scale)
        seen.sides |= sides
        seen.beyond = seen.beyond or beyond
        seen.visible = min(seen.visible, inside_w * inside_h / area)

    def runs(self) -> list[Run]:
        found: list[Run] = []
        for seen in self._seen.values():
            if seen.first is None:
                continue
            values: dict[str, float | int | str | bool] = {}
            if seen.scenes:
                # 字幕を直すには、そのシーンを開く（set_active_scene） 外側の配置で切れているなら
                # 置いたシーンのクリップを動かす どちらでも直せるよう、置いた所も添える
                values["in_scene_clip"] = seen.scenes[0]
            spread = max(
                (
                    max(abs(a[i] - b[i]) for i in range(4))
                    for a, b in zip(seen.boxes, seen.boxes[1:], strict=False)
                ),
                default=0.0,
            )
            found.append(
                Run(
                    TEXT_OVERFLOW,
                    seen.first,
                    seen.last + 1,
                    {
                        "overflow_px": math.ceil(seen.worst),
                        "sides": "・".join(side for side in "左右上下" if side in seen.sides),
                        "visible_percent": round(seen.visible * 100.0, 1),
                        "beyond_screen": seen.beyond,
                        "moving": spread > _MOVING,
                        "text": _text_of(seen.clip),
                        **values,
                    },
                    clip_ids=(str(seen.clip.id),),
                )
            )
        return found
