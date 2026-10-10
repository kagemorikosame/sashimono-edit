"""クリップの上に、キーフレームの位置を小さなひし形で描く

キーフレームは設定パネルとグラフでしか見えなかった 打ったのかどうか、どこに打ったのかを
タイムラインで確かめられないと、動かしたつもりの無い所で値が変わる理由が分からない

描く位置と押せる位置は :func:`keyframe_marks` の 1 か所で決める 別々に求めると、
間引いて描かなかったひし形を押せたり、描いたひし形が押せなかったりする
"""

from __future__ import annotations

from collections.abc import Iterator

from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QPainter, QPen, QPolygonF

from sashimono.core.model import AnimatedValue, Clip, Effect
from sashimono.ui.theme import Colors
from sashimono.ui.timeline.layout import TimelineLayout

__all__ = ["KEYFRAME_SIZE", "draw_keyframes", "keyframe_at", "keyframe_frames", "keyframe_marks"]

#: ひし形の対角線の半分（画素）
KEYFRAME_SIZE = 4

#: 隣のひし形とこれより近ければ描かない（画素） 縮小して何十個も重なると、
#: 帯のように潰れて 1 つずつを見分けられず、押してもどれに当たるか分からない
MIN_KEYFRAME_GAP = 9

#: 押したとみなす距離（画素） ひし形より少し広くして、小さな印でも押しやすくする
_HIT_SLOP = 2


def _animated(values: Iterator[object]) -> Iterator[AnimatedValue]:
    for value in values:
        if isinstance(value, AnimatedValue) and value.keyframes:
            yield value


def _effect_values(effects: tuple[Effect, ...]) -> Iterator[object]:
    for effect in effects:
        yield from effect.params.values()


def keyframe_frames(clip: Clip) -> tuple[int, ...]:
    """クリップに打ったキーフレームの位置（クリップの頭から数えたフレーム） 重なりは 1 つ

    集めるのは、値を時間で変えられる所すべて 不透明度・エフェクト（場面切り替えの後ろの
    場面に掛けるものも）・テキストや図形の中身 1 か所でも数え漏らすと、その値だけ
    キーフレームを打っても印が出ない
    クリップの外（トリムで外れた所）のキーフレームは描く所が無いので外す
    """
    values: list[object] = [clip.opacity]
    values.extend(_effect_values(clip.effects))
    values.extend(_effect_values(clip.after_effects))
    if clip.source is not None:
        values.extend(clip.source.params.values())
        # テキストの縁取りの層の値と、層に掛けたエフェクトの値（#272 #273） 数え漏らすと、
        # 縁の太さや色にキーを打ってもタイムラインに印が出ない
        for stroke in clip.source.strokes:
            values.extend(stroke.params.values())
            values.extend(_effect_values(stroke.effects))
    frames = {
        keyframe.frame
        for value in _animated(iter(values))
        for keyframe in value.keyframes
        if 0 <= keyframe.frame < clip.duration
    }
    return tuple(sorted(frames))


def keyframe_marks(
    clip: Clip, layout: TimelineLayout, clip_rect: QRect
) -> list[tuple[int, QPointF]]:
    """描くひし形 ``(タイムラインのフレーム, 中心)`` の並び 込み合う所は間引く

    ``clip_rect`` は画面に見えている部分へ切り詰めた矩形 外のひし形は描かない
    """
    marks: list[tuple[int, QPointF]] = []
    last = float("-inf")
    y = clip_rect.bottom() - KEYFRAME_SIZE - 2
    for relative in keyframe_frames(clip):
        frame = clip.timeline_start + relative
        x = layout.frame_to_x(frame)
        if x < clip_rect.left() or x > clip_rect.right():
            continue
        if x - last < MIN_KEYFRAME_GAP:
            continue
        marks.append((frame, QPointF(x, y)))
        last = x
    return marks


def draw_keyframes(
    painter: QPainter, clip: Clip, layout: TimelineLayout, clip_rect: QRect, *, selected: bool
) -> None:
    """クリップの下端に、キーフレームの位置のひし形を描く 選んだクリップの印は明るく大きく"""
    marks = keyframe_marks(clip, layout, clip_rect)
    if not marks:
        return
    size = KEYFRAME_SIZE + (1 if selected else 0)
    painter.save()
    painter.setClipRect(clip_rect)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(Colors.KEYFRAME_OUTLINE, 1))
    painter.setBrush(Colors.KEYFRAME_SELECTED if selected else Colors.KEYFRAME)
    for _, centre in marks:
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(centre.x(), centre.y() - size),
                    QPointF(centre.x() + size, centre.y()),
                    QPointF(centre.x(), centre.y() + size),
                    QPointF(centre.x() - size, centre.y()),
                ]
            ),
            Qt.FillRule.OddEvenFill,
        )
    painter.restore()


def keyframe_at(
    clip: Clip, layout: TimelineLayout, clip_rect: QRect, position: QPoint
) -> int | None:
    """``position`` にあるひし形のフレーム（タイムラインの位置） 無ければ ``None``"""
    reach = KEYFRAME_SIZE + _HIT_SLOP
    best: tuple[float, int] | None = None
    for frame, centre in keyframe_marks(clip, layout, clip_rect):
        dx = abs(position.x() - centre.x())
        dy = abs(position.y() - centre.y())
        if dx <= reach and dy <= reach and (best is None or dx < best[0]):
            best = (dx, frame)
    return best[1] if best is not None else None
