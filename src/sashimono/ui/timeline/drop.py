"""タイムラインへの落とし込み（エクスプローラーのファイルと、素材一覧の素材）

ビュー（:class:`~sashimono.ui.timeline.view.TimelineView`）は落とされた物と位置を
信号で外へ出すだけで、置くのは窓の仕事 ファイルは調べてからでないと長さも種類も
分からず、調べるのは窓の読み込みの流れ（裏のスレッド）が持っている
ここには、落とす位置の求め方と、ドラッグ中に落ちる所の目安を描く所をまとめる
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QMimeData, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from sashimono.core.commands import AddClip, AddTrack, Command, place_media
from sashimono.core.model import MediaId, Project, Timeline, TrackId
from sashimono.core.timebase import format_timecode
from sashimono.ui.media_pool import media_ids_in
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline.layout import TimelineLayout

__all__ = [
    "DropGuide",
    "DropPreview",
    "DropSpot",
    "accepts",
    "local_paths",
    "paint_drop_guide",
    "preview_drop",
    "spot_at",
]


@dataclass(frozen=True, slots=True)
class DropSpot:
    """落とした位置 ``track_id`` はトラックの無い所（下の空き）へ落としたら ``None``"""

    frame: int
    track_id: TrackId | None = None

    def after(self, commands: Sequence[Command]) -> DropSpot:
        """置いたクリップの後ろ 何本かを落としたとき、次の素材はここから並べる

        置けなかった（長さの無い素材だった）ときは動かさない
        """
        ends = [c.clip.timeline_end for c in commands if isinstance(c, AddClip)]
        return replace(self, frame=max([self.frame, *ends]))


@dataclass(frozen=True, slots=True)
class DropGuide:
    """ドラッグ中の目安 描くためだけに持つ"""

    spot: DropSpot
    #: 素材一覧から引いてきた素材 ファイルを引いてきたときは空
    #: ファイルは調べるまで長さが分からないので、落ちる位置の線だけを出す
    media_ids: tuple[MediaId, ...] = ()


def local_paths(mime: QMimeData) -> list[Path]:
    """落とされたファイル ネット上の URL は読み込めないので外す"""
    if not mime.hasUrls():
        return []
    return [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]


def accepts(mime: QMimeData) -> bool:
    """タイムラインが受け取れる物か 受け取れない物は、落とせない印（禁止の指）にする"""
    return bool(media_ids_in(mime)) or bool(local_paths(mime))


@dataclass(frozen=True, slots=True)
class DropPreview:
    """ドラッグ中に見せる、落としたときの姿

    素材一覧から引いてきた素材は、窓が置くのと同じ決め方（:func:`place_media`）で
    置く先を求める 新しくトラックを作ることになるなら、そのトラックを実際に足される
    位置（映像なら上、音声なら下）へ空のまま並べた :attr:`timeline` を描く
    並べた最後の下に仮の行を出す形だと、落とした後に出てくる位置と違って迷う
    """

    guide: DropGuide
    #: 足すトラックを空のまま並べたタイムライン 足さないなら元のまま
    timeline: Timeline
    #: 置くときのコマンド ファイルを引いてきたときは空（調べるまで長さが分からない）
    commands: tuple[Command, ...] = ()

    @property
    def new_tracks(self) -> frozenset[TrackId]:
        return frozenset(c.track.id for c in self.commands if isinstance(c, AddTrack))


def preview_drop(project: Project, guide: DropGuide, *, split_audio: bool = False) -> DropPreview:
    """``split_audio`` は窓が置くときと同じ値を渡す（:func:`place_media`）"""
    media = [
        item for media_id in guide.media_ids if (item := project.find_media(media_id)) is not None
    ]
    if not media:
        return DropPreview(guide, project.timeline)
    commands = place_media(
        project,
        media,
        at_frame=guide.spot.frame,
        track_id=guide.spot.track_id,
        split_audio=split_audio,
    )
    # 足すトラックだけを当てる クリップまで当てると、枠の点線ではなく本物のクリップに見える
    shown = project
    for command in commands:
        if isinstance(command, AddTrack):
            shown = command.apply(shown)
    return DropPreview(guide, shown.timeline, tuple(commands))


def spot_at(
    layout: TimelineLayout,
    timeline: Timeline,
    position: QPointF,
    *,
    real: Timeline | None = None,
) -> DropSpot:
    """画面の位置を、落とす先のフレームとトラックへ直す

    トラックの名前の欄（左端）へ落としたら、その時点で見えている左端のフレームに置く
    名前の欄はフレームを持たないが、そこへ落とした人は「このトラックへ」と思っている

    ``timeline`` は画面に出しているもの（仮のトラックを並べたもの）で、``real`` は
    本物 仮のトラックの行はまだ無いトラックなので、トラックの無い所と同じに扱う
    """
    x = max(float(Metrics.TRACK_HEADER_WIDTH), position.x())
    track_id = layout.track_at(timeline, int(position.y()))
    if track_id is not None and real is not None and real.find_track(track_id) is None:
        track_id = None
    return DropSpot(frame=layout.frame_at(x), track_id=track_id)


def paint_drop_guide(
    painter: QPainter,
    layout: TimelineLayout,
    project: Project,
    preview: DropPreview,
    size: tuple[int, int],
) -> None:
    """落ちる所の目安を描く 落とすトラックを薄く塗り、落ちるフレームに縦線を引く

    素材一覧から引いてきた素材は、置かれるクリップの枠を点線で出す 落としたトラックが
    埋まっていて別のトラックへ回る、を離す前に分かるようにする
    ``project`` は本物 トラックの並びは ``preview`` の方を使う
    """
    guide = preview.guide
    width, height = size
    bands = {band.track.id: band for band in layout.bands(preview.timeline)}
    painter.save()
    target = bands.get(guide.spot.track_id) if guide.spot.track_id is not None else None
    if target is not None:
        fill = QColor(Colors.ACCENT)
        fill.setAlpha(36)
        painter.fillRect(
            QRectF(
                Metrics.TRACK_HEADER_WIDTH,
                target.top,
                width - Metrics.TRACK_HEADER_WIDTH,
                target.height,
            ),
            fill,
        )

    new_tracks = preview.new_tracks
    for track_id in new_tracks:
        band = bands.get(track_id)
        if band is None:
            continue
        # まだ無いトラックだと分かるように地を明るくし、名前の欄の下に添える
        row = QRectF(0, band.top, width, band.height)
        painter.fillRect(row.adjusted(0, 1, 0, -1), Colors.GHOST_ROW)
        painter.setPen(Colors.TEXT_MUTED)
        painter.drawText(
            QRectF(8, band.top, Metrics.TRACK_HEADER_WIDTH - 8, band.height - 4),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            "新しく作る",
        )

    painter.setPen(QPen(Colors.ACCENT, 2, Qt.PenStyle.DashLine))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for command in preview.commands:
        if not isinstance(command, AddClip) or (band := bands.get(command.track_id)) is None:
            continue
        left = layout.frame_to_x(command.clip.timeline_start)
        right = layout.frame_to_x(command.clip.timeline_end)
        painter.drawRect(QRectF(left, band.top + 1, max(2.0, right - left), band.height - 3))

    x = layout.frame_to_x(guide.spot.frame)
    painter.setPen(QPen(Colors.ACCENT, 2))
    painter.drawLine(QPointF(x, Metrics.RULER_HEIGHT), QPointF(x, height))
    # 目盛りの上に落ちる時刻を添える 線だけだと、何秒の所へ落ちるのかを目盛りから読むことになる
    label = format_timecode(guide.spot.frame, project.rate)
    metrics = painter.fontMetrics()
    box = QRectF(x + 4, 2, metrics.horizontalAdvance(label) + 8, Metrics.RULER_HEIGHT - 4)
    if box.right() > width:
        # 右端では線の左へ回す はみ出すと、右端へ落とすときほど時刻が読めない
        box.moveRight(x - 4)
    painter.fillRect(box, Colors.ACCENT)
    painter.setPen(Colors.WINDOW)
    painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)
    painter.restore()
