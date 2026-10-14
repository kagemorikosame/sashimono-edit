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
from dataclasses import dataclass, field

from sashimono.core.model import Clip, Project, Track, controlling_groups
from sashimono.engine.checks.scan import TEXT_OVERFLOW, CheckLimits, Run
from sashimono.engine.render.groups import grouped
from sashimono.engine.render.outline import clip_outline
from sashimono.engine.render.renderer import FrameRenderer

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
        #: 測るコマ → そのコマで測る字幕
        self._plan: dict[int, list[tuple[Track, Clip]]] = {}
        step = max(1, round(float(project.rate.fps) * SAMPLE_SECONDS))
        timeline = project.timeline
        for track in timeline.active_picture_tracks():
            for clip in track.clips:
                source = clip.source
                if source is None or source.kind != "text" or not clip.enabled:
                    continue
                if not project.draws_picture(track, clip):
                    continue
                first = max(start, clip.timeline_start)
                last = min(end, clip.timeline_end) - 1
                if last < first:
                    continue
                frames = set(range(first, last + 1, step))
                frames.add(last)
                for frame in frames:
                    self._plan.setdefault(frame, []).append((track, clip))

    @property
    def frames(self) -> frozenset[int]:
        """測るコマ"""
        return frozenset(self._plan)

    def measure(self, renderer: FrameRenderer, frame: int, canvas: tuple[int, int]) -> None:
        """``frame`` で測る字幕を測る ``canvas`` はレンダラの合成の大きさ"""
        for track, clip in self._plan.get(frame, ()):
            extent = renderer.content_extent(clip, frame)
            if extent is None:
                continue
            shown = self._grouped(track, clip, frame)
            outline = clip_outline(self._project, shown, frame, canvas=canvas, extent=extent)
            if outline is None:
                continue
            xs = [point[0] for point in outline.corners]
            ys = [point[1] for point in outline.corners]
            self._note(clip, frame, (min(xs), min(ys), max(xs), max(ys)), canvas)

    def _grouped(self, track: Track, clip: Clip, frame: int) -> Clip:
        """グループ制御で動かした後のクリップ プレビューの外枠と同じ当て方"""
        timeline = self._project.timeline
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
        seen = self._seen.setdefault(str(clip.id), _Seen(clip))
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
        for clip_id, seen in self._seen.items():
            if seen.first is None:
                continue
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
                    },
                    clip_ids=(clip_id,),
                )
            )
        return found
