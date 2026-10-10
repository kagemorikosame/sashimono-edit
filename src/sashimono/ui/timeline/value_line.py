"""クリップの上の値の線 不透明度と音量をタイムラインで直接動かす（#27 P8）

絵を持つクリップは不透明度、音を持つクリップは固定の音量（``audio_volume`` の ``volume``）
の折れ線を引く 音付きの動画（混合トラックの 1 本のクリップ）は右クリックでどちらを出すか
切り替える 設定パネルを開かずに、フェードや音量の上げ下げを目で見ながら決められる

:class:`~sashimono.ui.timeline.view.TimelineView` からは入口を数行呼ぶだけにして、
描き方・掴み方・命令の作り方はここにまとめる（書き出し範囲の :mod:`work_area` と同じ作り）

線の点と #155 のひし形（:mod:`sashimono.ui.timeline.keyframes`）の関係
- ひし形はクリップのどの値のキーフレームも出す、時刻だけの印 下の端に並ぶ
- 線の点は線に出している値のキーフレームだけ 線の上の、値の高さに丸で描く
- 線はひし形の列より上だけを使う 値が 0 でも点がひし形に重ならず、押し分けられる
- 押したときはひし形が先（再生ヘッドを動かす今までの動き） 線はその後で見る
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QMenu

from sashimono.core.commands import (
    AddEffect,
    Command,
    MoveKeyframe,
    ParamPath,
    RemoveKeyframe,
    SetParam,
)
from sashimono.core.commands.fixed import VOLUME_EFFECT_KIND, fixed_effect, fixed_slot
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    MediaItem,
    Project,
    Track,
    TrackKind,
)
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline.keyframes import KEYFRAME_SIZE, MIN_KEYFRAME_GAP
from sashimono.ui.timeline.layout import TimelineLayout
from sashimono.ui.timeline.painter import DETAIL_MIN_WIDTH, cached_pen, clip_rect_for

__all__ = [
    "LINE_GRAB",
    "MIN_LINE_TRACK_HEIGHT",
    "SHOW_OPACITY_TEXT",
    "SHOW_VOLUME_TEXT",
    "ValueGrab",
    "ValueKind",
    "ValueLineEditor",
    "line_area",
    "value_kinds",
    "value_of",
    "value_to_y",
]

#: 線を引くトラックの高さの下限（画素） 低い帯では名前とひし形の間に数画素しか残らず、
#: 線が名前の帯やひし形に重なって、どれを掴んだのか分からない 既定の高さ（60）では出す
MIN_LINE_TRACK_HEIGHT = 48

#: 線の上下で掴める幅（画素） 狭いと 1 画素の線を押し損ね、広いとクリップの移動を掴めない
LINE_GRAB = 4

#: 線の点の半径（画素） ひし形（:data:`KEYFRAME_SIZE`）と見分けが付くよう丸にし、少し小さくする
_POINT_RADIUS = 3

#: 名前の帯の下に空ける隙間（画素） 不透明度 100% の線が名前の帯の縁に張り付くと見えない
_TOP_GAP = 3

#: 下の端はひし形の列（中心が下端から KEYFRAME_SIZE + 2、選ぶと 1 画素大きい）の上で止める
#: 値が 0 の線がひし形に重なると、どちらを押したのか分からない
_BOTTOM_GAP = 2 * KEYFRAME_SIZE + 5

#: 線の種類を貯めるクリップの数の上限 超えたら全部捨てて貯め直す
_SHAPES_LIMIT = 8192

#: 右クリックに出す名前 試験もこの名前で探す
SHOW_OPACITY_TEXT = "線に不透明度を出す"
SHOW_VOLUME_TEXT = "線に音量を出す"


class ValueKind(Enum):
    """線に出す値"""

    OPACITY = "opacity"
    VOLUME = "volume"

    @property
    def title(self) -> str:
        return "不透明度" if self is ValueKind.OPACITY else "音量"

    @property
    def maximum(self) -> float:
        """値の上限 下限はどちらも 0

        音量は ``audio_volume`` の定義の上限（400%）と同じ 目盛りは % のまま比例で並べる
        （利用者の決定） 100% は下から 4 分の 1 の高さになる
        """
        return 1.0 if self is ValueKind.OPACITY else 400.0

    def percent(self, value: float) -> float:
        """画面に出す % 不透明度は 0〜1 で持っているので 100 倍する"""
        return value * 100.0 if self is ValueKind.OPACITY else value

    def clamp(self, value: float) -> float:
        return min(max(value, 0.0), self.maximum)


class ValueGrab(Enum):
    """押した所で始まった操作"""

    #: 線を掴んだ 上下で値を動かす
    LINE = "line"
    #: 線の点を掴んだ 時刻と値を動かす
    KEY = "key"
    #: 押しただけで済んだ（Ctrl+クリックで点を打った） ドラッグは始めない
    DONE = "done"


def value_kinds(track: Track, clip: Clip, project: Project) -> tuple[ValueKind, ...]:
    """クリップに出せる値 先頭が既定

    映像トラックは不透明度だけ 置いたシーンは音も鳴るが、映像トラックの音量は
    クリップの音量調整を通らない（シーンの中のミキサで決まる）ので、線を出しても効かない
    音声トラックは音量だけ 混合トラックは、絵を描くなら不透明度、音を鳴らすなら音量
    """
    # 素材は混合トラックのときだけ引く（Project の draws_picture と plays_sound） 描くたびに
    # クリップごとに素材の一覧をなめると、細い帯との境の辺りで拡大したときの 400 本を超える
    # 線で、素材の多い作品ほど重くなる（#260）
    kinds: list[ValueKind] = []
    if track.kind is not TrackKind.AUDIO and project.draws_picture(track, clip):
        kinds.append(ValueKind.OPACITY)
    if track.kind is not TrackKind.VIDEO and project.plays_sound(track, clip):
        kinds.append(ValueKind.VOLUME)
    return tuple(kinds)


def _fixed_volume(clip: Clip) -> Effect | None:
    return next((e for e in clip.effects if e.fixed and e.kind == VOLUME_EFFECT_KIND), None)


def _animated(value: object, default: float) -> AnimatedValue:
    if isinstance(value, AnimatedValue):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return AnimatedValue(static=float(value))
    return AnimatedValue(static=default)


def value_of(clip: Clip, kind: ValueKind) -> AnimatedValue:
    """線に出す値 固定の音量をまだ持たない古いクリップは 100%（音を変えない値）として描く"""
    if kind is ValueKind.OPACITY:
        return clip.opacity
    effect = _fixed_volume(clip)
    return _animated(effect.params.get("volume") if effect is not None else None, 100.0)


def line_area(rect: QRect) -> QRect | None:
    """線を引く範囲 クリップの名前の帯の下から、ひし形の列の上まで 狭すぎれば ``None``"""
    top = rect.top() + Metrics.CLIP_LABEL_HEIGHT + _TOP_GAP
    bottom = rect.bottom() - _BOTTOM_GAP
    if bottom - top < 8:
        return None
    return QRect(rect.left(), top, rect.width(), bottom - top)


def value_to_y(area: QRect, kind: ValueKind, value: float) -> float:
    return area.bottom() - kind.clamp(value) / kind.maximum * area.height()


def _points(
    clip: Clip, kind: ValueKind, value: AnimatedValue, layout: TimelineLayout, area: QRect
) -> list[tuple[int, QPointF]]:
    """描く点 ``(クリップの頭からのフレーム, 中心)`` ひし形と同じく込み合う所は間引く

    描く点と押せる点をここ 1 か所で決める（ひし形の :func:`keyframe_marks` と同じ理由）
    """
    points: list[tuple[int, QPointF]] = []
    last = float("-inf")
    for keyframe in value.keyframes:
        if not 0 <= keyframe.frame < clip.duration:
            continue
        x = layout.frame_to_x(clip.timeline_start + keyframe.frame)
        if x < area.left() or x > area.right() or x - last < MIN_KEYFRAME_GAP:
            continue
        points.append((keyframe.frame, QPointF(x, value_to_y(area, kind, keyframe.value))))
        last = x
    return points


def _shifted(value: AnimatedValue, kind: ValueKind, delta: float) -> AnimatedValue:
    """全部の点（点が無ければ値そのもの）を同じだけ動かす 範囲の外へ出る点は端で止める

    点ごとに止める 一番低い点が 0 に着いた所で全体を止めると、ほかの点を下げ切れない
    """
    return AnimatedValue(
        static=kind.clamp(value.static + delta),
        keyframes=tuple(
            replace(keyframe, value=kind.clamp(keyframe.value + delta))
            for keyframe in value.keyframes
        ),
    )


def _path(clip: Clip, kind: ValueKind) -> ParamPath | None:
    """値の在りか 固定の音量をまだ持たないクリップは ``None``"""
    if kind is ValueKind.OPACITY:
        return ParamPath.of_clip(clip.id, "opacity")
    effect = _fixed_volume(clip)
    return None if effect is None else ParamPath.of_effect(clip.id, effect.id, "volume")


def _set_value(clip: Clip, kind: ValueKind, value: AnimatedValue) -> Command:
    """値を丸ごと差し替える命令

    固定の音量をまだ持たない古いクリップは、値を入れた固定の音量を足す 1 つの命令にする
    足すのと値を変えるのを分けると、取り消しが 2 段になり、1 回戻すと 100% の音量調整だけが残る
    """
    path = _path(clip, kind)
    if path is not None:
        return SetParam(path, value)
    # 差し込む位置は置いたときと同じ決まり（fixed_slot） 自前で決めると、固定の項目どうしの
    # 並び（反転 → 配置 → 音量 → フェード）が置いたクリップと食い違う
    effect = fixed_effect(VOLUME_EFFECT_KIND).with_param("volume", value)
    return AddEffect(clip.id, effect, index=fixed_slot(clip.effects, VOLUME_EFFECT_KIND))


@dataclass(slots=True)
class _Drag:
    """掴んでいる線か点 途中は掴む前のプロジェクトへ命令を当てて描くだけ"""

    clip_id: ClipId
    kind: ValueKind
    base: Project
    #: 線を引いた範囲 掴んだ時点で決める 途中で高さが変わっても、動かす量を変えない
    area: QRect
    grab_y: int
    #: 線を掴んだ所の値 点を掴んだときは点の値
    grab_value: float
    #: 掴んだ点（クリップの頭からのフレーム） 線を掴んだときは ``None``
    key: int | None = None
    command: Command | None = None
    #: いまの値と、それを描く位置（値の札を出す）
    shown: float = 0.0
    at: QPointF | None = None


class ValueLineEditor:
    """値の線の描き方と、掴んで動かす操作

    ドラッグの途中はプロジェクトを書き換えず、掴む前のプロジェクトに命令を当てた物を
    ビューに描かせる（トラックの高さのドラッグと同じ作り） プレビューにも途中の値を
    出せるよう ``preview`` へ命令を渡す 離したときに 1 つの命令を出すので、取り消しは 1 段
    """

    def __init__(
        self,
        request: Callable[[list[Command], str], None],
        preview: Callable[[Command], None],
        show: Callable[[Project], None],
        redraw: Callable[[], None],
    ) -> None:
        self._request = request
        self._preview = preview
        #: ビューに描かせるプロジェクトを差し替える口 ドラッグの途中と、離したときに使う
        self._show = show
        self._redraw = redraw
        #: 線を出すか 設定（:attr:`Preferences.value_lines`）で切れる
        self.enabled = True
        #: これより細いクリップには線を描かず、掴ませもしない ビューが設定から合わせる
        #: （:meth:`TimelineView.set_detail_min_width`）
        self.detail_min_width = DETAIL_MIN_WIDTH
        #: 音量の線に切り替えたクリップ 画面の都合なのでプロジェクトには残さない
        self._volume_shown: set[ClipId] = set()
        self._drag: _Drag | None = None
        #: 線の種類と平らな線の高さの割合を、クリップごとに貯める（:meth:`_shape`）
        self._shapes: dict[
            ClipId,
            tuple[Clip, Track, tuple[MediaItem, ...], bool, tuple[ValueKind, float | None] | None],
        ] = {}

    @property
    def dragging(self) -> bool:
        return self._drag is not None

    # --- どの値を出すか ---

    def kind_for(self, project: Project, track: Track, clip: Clip) -> ValueKind | None:
        """``clip`` の線に出す値 出さなければ ``None``"""
        if not self.enabled:
            return None
        kinds = value_kinds(track, clip, project)
        if not kinds:
            return None
        if ValueKind.VOLUME in kinds and clip.id in self._volume_shown:
            return ValueKind.VOLUME
        return kinds[0]

    def add_menu_items(
        self, menu: QMenu, project: Project, track: Track, clip: Clip, selection: Iterable[ClipId]
    ) -> None:
        """右クリックに、線に出す値の切り替えを足す 両方を出せるクリップのときだけ

        選んでいる何本かのうち、両方を出せる物をまとめて切り替える 1 本ずつ右クリックし直すと、
        並べた音付きの動画の音量を見比べるまでが長い
        """
        if not self.enabled or len(value_kinds(track, clip, project)) < 2:
            return
        targets = [clip.id]
        for clip_id in selection:
            located = project.timeline.locate_clip(clip_id)
            if (
                clip_id != clip.id
                and located is not None
                and len(value_kinds(located[0], located[1], project)) > 1
            ):
                targets.append(clip_id)
        showing = self.kind_for(project, track, clip)
        menu.addSeparator()
        for kind, text in (
            (ValueKind.OPACITY, SHOW_OPACITY_TEXT),
            (ValueKind.VOLUME, SHOW_VOLUME_TEXT),
        ):
            action = menu.addAction(text)
            if action is None:  # pragma: no cover - Qt が None を返すのは異常系のみ
                continue
            action.setCheckable(True)
            action.setChecked(kind is showing)
            action.triggered.connect(
                lambda _checked=False, kind=kind: self.show_kind(targets, kind)
            )

    def show_kind(self, clip_ids: Iterable[ClipId], kind: ValueKind) -> None:
        for clip_id in clip_ids:
            if kind is ValueKind.VOLUME:
                self._volume_shown.add(clip_id)
            else:
                self._volume_shown.discard(clip_id)
        self._redraw()

    # --- 描く ---

    def paint(
        self,
        painter: QPainter,
        project: Project,
        track: Track,
        clip: Clip,
        layout: TimelineLayout,
        rect: QRect,
        band_height: int,
        *,
        selected: bool,
    ) -> None:
        """``clip`` の上に線と点を描く ``rect`` は見えている部分へ切り詰めた矩形"""
        if band_height < MIN_LINE_TRACK_HEIGHT:
            return
        kind = self.kind_for(project, track, clip)
        area = line_area(rect)
        if kind is None or area is None:
            return
        value = value_of(clip, kind)
        color = Colors.OPACITY_LINE if kind is ValueKind.OPACITY else Colors.VOLUME_LINE
        if value.is_animated:
            # 列ごとに値を引く 点と点の間の出方（イージング）も、再生したときの値と同じ形で見える
            line = QPolygonF(
                [
                    QPointF(
                        column,
                        value_to_y(
                            area, kind, value.at(layout.x_to_frame(column) - clip.timeline_start)
                        ),
                    )
                    for column in range(area.left(), area.right() + 2)
                ]
            )
        else:
            y = value_to_y(area, kind, value.static)
            line = QPolygonF([QPointF(area.left(), y), QPointF(area.right() + 1, y)])

        painter.save()
        painter.setClipRect(rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(cached_pen(Colors.VALUE_LINE_SHADOW, 3.5))
        painter.drawPolyline(line)
        # 線は 2 画素で引く 1 画素半にすると、どの行にも半分ずつしか乗らず、画素の上では
        # 背景と混ざった灰色になって、サムネイルの上で見分けにくい
        painter.setPen(cached_pen(color, 2.0))
        painter.drawPolyline(line)
        painter.setPen(cached_pen(Colors.VALUE_LINE_SHADOW, 1))
        painter.setBrush(color)
        radius = _POINT_RADIUS + (0.5 if selected else 0.0)
        for _, centre in _points(clip, kind, value, layout, area):
            painter.drawEllipse(centre, radius, radius)
        drag = self._drag
        if drag is not None and drag.clip_id == clip.id and drag.at is not None:
            _draw_tag(painter, rect, drag.at, f"{kind.title} {kind.percent(drag.shown):.0f}%")
        painter.restore()

    def paint_many(
        self,
        painter: QPainter,
        project: Project,
        track: Track,
        clips: Sequence[tuple[Clip, QRect, bool]],
        layout: TimelineLayout,
        band_height: int,
    ) -> None:
        """1 本のトラックの ``(クリップ, 見えている矩形, 選んだか)`` に線をまとめて描く

        描く絵は :meth:`paint` を 1 本ずつ呼んだときと同じ 境を越えて拡大すると線を
        描くクリップが 400 本を超え、1 本ずつ状態を保存して戻すと 14ms ほど掛かった（#260）
        点の無い平らな線（ほとんど）は、影と線をそれぞれまとめて引く クリップは重ならず、
        線はクリップの矩形で切り落とすので、影と線の順がクリップごとに守られれば同じ絵になる
        点のある線・動かしている線は今までの描き方で描く
        """
        if band_height < MIN_LINE_TRACK_HEIGHT or not self.enabled:
            return
        # 平らな線（左端, 右端の次の画素, 高さ, 色） 隙間なく続き、高さと色が同じ物は 1 本に繋ぐ
        flat: list[tuple[int, int, float, QColor]] = []
        drag = self._drag
        colours = {ValueKind.OPACITY: Colors.OPACITY_LINE, ValueKind.VOLUME: Colors.VOLUME_LINE}
        top_gap = Metrics.CLIP_LABEL_HEIGHT + _TOP_GAP
        for clip, rect, selected in clips:
            shape = self._shape(project, track, clip)
            if shape is None:
                continue
            kind, share = shape
            # line_area と value_to_y と同じ式 矩形を作らずに求める（area の下端は
            # bottom - 1、高さは bottom - top）
            top = rect.top() + top_gap
            bottom = rect.bottom() - _BOTTOM_GAP
            if bottom - top < 8:
                continue
            if share is None or (drag is not None and drag.clip_id == clip.id):
                self.paint(
                    painter, project, track, clip, layout, rect, band_height, selected=selected
                )
                continue
            y = (bottom - 1) - share * (bottom - top)
            colour = colours[kind]
            left, right = rect.left(), rect.right() + 1
            last = flat[-1] if flat else None
            if last is not None and last[1] == left and last[2] == y and last[3] is colour:
                flat[-1] = (last[0], right, y, colour)
            else:
                flat.append((left, right, y, colour))
        if not flat:
            return
        # paint は四角い端のペンで矩形の外まで引き、クリップの矩形で切り落とす 端を切りっぱなしに
        # して矩形の左右ちょうどで止めれば、切り落とさなくても同じ画素になる 切り落とさなければ、
        # 隣り合うクリップの同じ高さの線を 1 本にまとめて引ける
        lines = [
            (QPolygonF([QPointF(left, y), QPointF(right, y)]), colour)
            for left, right, y, colour in flat
        ]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(cached_pen(Colors.VALUE_LINE_SHADOW, 3.5, flat=True))
        for line, _ in lines:
            painter.drawPolyline(line)
        # 線は 2 画素で引く（paint と同じ）
        for line, colour in lines:
            painter.setPen(cached_pen(colour, 2.0, flat=True))
            painter.drawPolyline(line)
        painter.restore()

    def forget_shapes(self) -> None:
        """貯めた線の種類と高さを捨てる クリップとトラックと素材の一覧を強く持つので、
        別のプロジェクトを開いたときに捨てる（:meth:`TimelineView.forget_drawing`）"""
        self._shapes.clear()

    def _shape(
        self, project: Project, track: Track, clip: Clip
    ) -> tuple[ValueKind, float | None] | None:
        """線に出す値と、平らな線ならその高さの割合（``clamp(値) / 上限``） 点のある線は
        割合を ``None`` にする 線を出さなければ ``None``

        クリップ・トラック・素材の一覧・音量の線に切り替えたかが同じ間は前に求めた物を使う
        1 本ごとに値の種類と値を求め直すと、境を越えて拡大したときの 400 本で数 ms になる
        """
        shown = clip.id in self._volume_shown
        entry = self._shapes.get(clip.id)
        if (
            entry is not None
            and entry[0] is clip
            and entry[1] is track
            and entry[2] is project.media
            and entry[3] == shown
        ):
            return entry[4]
        kind = self.kind_for(project, track, clip)
        found: tuple[ValueKind, float | None] | None = None
        if kind is not None:
            value = value_of(clip, kind)
            flat = not (value.is_animated or value.keyframes)
            found = (kind, kind.clamp(value.static) / kind.maximum if flat else None)
        if len(self._shapes) >= _SHAPES_LIMIT:
            self._shapes.clear()
        self._shapes[clip.id] = (clip, track, project.media, shown, found)
        return found

    # --- 押す・動かす・離す ---

    def _geometry(
        self, project: Project, layout: TimelineLayout, width: int, clip: Clip, position: QPoint
    ) -> tuple[ValueKind, QRect] | None:
        """押せる線があれば、出している値と線の範囲"""
        if clip.duration * layout.pixels_per_frame < self.detail_min_width:
            # 細いクリップは名前も線も描かない（:meth:`TimelineView.paintEvent`）
            return None
        band = layout.band_at(project.timeline, position.y())
        if band is None or band.height < MIN_LINE_TRACK_HEIGHT or band.track.locked:
            # ロックしたトラックは移動もトリムも断る 値だけ動くと、ロックの意味が崩れる
            return None
        kind = self.kind_for(project, band.track, clip)
        rect = clip_rect_for(clip, band, layout, width)
        area = line_area(rect) if rect is not None else None
        if kind is None or area is None:
            return None
        return kind, area

    def grab_at(
        self, project: Project, layout: TimelineLayout, width: int, clip: Clip, position: QPoint
    ) -> ValueGrab | None:
        """``position`` で掴めるもの（点か線） 掴めなければ ``None`` 押さずに調べる（形を変える）"""
        found = self._grab_at(project, layout, width, clip, position)
        return None if found is None else found[0]

    def _grab_at(
        self, project: Project, layout: TimelineLayout, width: int, clip: Clip, position: QPoint
    ) -> tuple[ValueGrab, ValueKind, QRect, int | None] | None:
        geometry = self._geometry(project, layout, width, clip, position)
        if geometry is None:
            return None
        kind, area = geometry
        value = value_of(clip, kind)
        reach = _POINT_RADIUS + 2
        for frame, centre in _points(clip, kind, value, layout, area):
            if abs(position.x() - centre.x()) <= reach and abs(position.y() - centre.y()) <= reach:
                return ValueGrab.KEY, kind, area, frame
        at = layout.x_to_frame(position.x()) - clip.timeline_start
        if abs(position.y() - value_to_y(area, kind, value.at(at))) <= LINE_GRAB:
            return ValueGrab.LINE, kind, area, None
        return None

    def press(
        self,
        project: Project,
        layout: TimelineLayout,
        width: int,
        clip: Clip,
        position: QPoint,
        modifiers: Qt.KeyboardModifier,
    ) -> ValueGrab | None:
        """線か点の上なら操作を始める 何も掴まなければ ``None``（呼んだ側が今までどおり続ける）

        トリムの端は呼んだ側が先に見る 端の近くの線を取ると、短いクリップの端を掴めない
        """
        found = self._grab_at(project, layout, width, clip, position)
        if found is None:
            return None
        grab, kind, area, key = found
        value = value_of(clip, kind)
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if grab is ValueGrab.LINE:
                self._add_key(clip, kind, value, layout, position)
            # 点の上の Ctrl+クリックは何もしない 選択の足し引きにすると、点を打つつもりで
            # 少しずれただけでクリップの選択が変わる
            return ValueGrab.DONE
        if key is not None:
            grab_value = next(k.value for k in value.keyframes if k.frame == key)
        else:
            grab_value = value.at(layout.x_to_frame(position.x()) - clip.timeline_start)
        self._drag = _Drag(
            clip_id=clip.id,
            kind=kind,
            base=project,
            area=area,
            grab_y=position.y(),
            grab_value=grab_value,
            key=key,
            shown=grab_value,
            at=QPointF(position),
        )
        return grab

    def _add_key(
        self,
        clip: Clip,
        kind: ValueKind,
        value: AnimatedValue,
        layout: TimelineLayout,
        position: QPoint,
    ) -> None:
        """Ctrl+クリックの所へ、今の線の値のまま点を打つ 線の形は変えない"""
        frame = round(layout.x_to_frame(position.x())) - clip.timeline_start
        frame = min(max(frame, 0), clip.duration - 1)
        # SetKeyframe で足すと新しい点は直線になり、イージングや瞬間移動の区間の途中へ
        # 打つと形が変わる 左の点の出方まで直した値を丸ごと入れる
        added = value.with_keyframe_at(frame)
        if added != value:
            self._request([_set_value(clip, kind, added)], f"{kind.title}にキーフレーム")

    def move(self, layout: TimelineLayout, position: QPoint) -> None:
        drag = self._drag
        if drag is None:
            return
        located = drag.base.timeline.locate_clip(drag.clip_id)
        if located is None:
            return
        clip = located[1]
        value = value_of(clip, drag.kind)
        command: Command | None
        if drag.key is None:
            span = drag.kind.maximum / max(1, drag.area.height())
            # 差は止めずに渡し、止めるのは点ごと（_shifted） 掴んだ所の値で先に止めると、
            # そこが端に着いた時点で、まだ動けるほかの点まで止まる
            delta = (drag.grab_y - position.y()) * span
            changed = _shifted(value, drag.kind, delta)
            command = None if changed == value else _set_value(clip, drag.kind, changed)
            drag.shown = drag.kind.clamp(drag.grab_value + delta)
        else:
            frame = self._key_frame(clip, value, drag.key, layout, position)
            # 掴んだ位置からの差で決める 指の位置をそのまま値にすると、点の中心から外れて
            # 掴んだだけで、動かし始めに値が飛ぶ（既定の高さで 1 画素が不透明度の約 4%）
            span = drag.kind.maximum / max(1, drag.area.height())
            level = drag.kind.clamp(drag.grab_value + (drag.grab_y - position.y()) * span)
            path = _path(clip, drag.kind)
            same = frame == drag.key and level == drag.grab_value
            command = None if path is None or same else MoveKeyframe(path, drag.key, frame, level)
            drag.shown = level
        drag.command = command
        drag.at = QPointF(position)
        # 元の値へ戻したときも知らせる 知らせないと、プレビューだけ途中の値のまま残る
        # 戻すのは何も変えない命令（今の不透明度をそのまま入れる）で足りる プレビューは
        # 本物のプロジェクトへ当て直して描くので、途中の値が消える
        unchanged = SetParam(ParamPath.of_clip(clip.id, "opacity"), clip.opacity)
        self._preview(command if command is not None else unchanged)
        self._show(command.apply(drag.base) if command is not None else drag.base)

    @staticmethod
    def _key_frame(
        clip: Clip, value: AnimatedValue, key: int, layout: TimelineLayout, position: QPoint
    ) -> int:
        """点を動かす先のフレーム 隣の点を越えず、クリップの外へも出さない

        越えさせると並びが入れ替わり、隣の点と重なったときはその点を消してしまう
        （:class:`MoveKeyframe` は行き先の点を置き換える）
        """
        frames = [k.frame for k in value.keyframes]
        index = frames.index(key)
        low = frames[index - 1] + 1 if index > 0 else 0
        high = frames[index + 1] - 1 if index + 1 < len(frames) else clip.duration - 1
        frame = round(layout.x_to_frame(position.x())) - clip.timeline_start
        return min(max(frame, low), max(low, high))

    def release(self) -> None:
        """離した 描いていた途中の値を捨て、決まった値を 1 つの命令にして出す"""
        drag, self._drag = self._drag, None
        if drag is None:
            return
        # 先に掴む前へ戻す 受け手は今のプロジェクトへ命令を当てるので、途中の値に
        # 当てると、動かした量が 2 重に掛かる
        self._show(drag.base)
        if drag.command is not None:
            done = "のキーフレームを移動" if drag.key is not None else "を変更"
            self._request([drag.command], f"{drag.kind.title}{done}")

    def remove_at(
        self, project: Project, layout: TimelineLayout, width: int, clip: Clip, position: QPoint
    ) -> bool:
        """点の上の右クリックなら、その点を消して真を返す"""
        found = self._grab_at(project, layout, width, clip, position)
        if found is None or found[0] is not ValueGrab.KEY or found[3] is None:
            return False
        path = _path(clip, found[1])
        if path is None:
            return False
        self._request([RemoveKeyframe(path, found[3])], f"{found[1].title}のキーフレームを削除")
        return True


def _draw_tag(painter: QPainter, rect: QRect, at: QPointF, text: str) -> None:
    """ドラッグ中の値の札 指の右上に出す はみ出すならクリップの中へ寄せる"""
    font = QFont(painter.font())
    font.setPointSizeF(8.0)
    painter.setFont(font)
    width = painter.fontMetrics().horizontalAdvance(text) + 8
    height = painter.fontMetrics().height() + 2
    left = min(at.x() + 8, rect.right() - width)
    top = max(at.y() - height - 4, rect.top())
    box = QRectF(max(rect.left(), left), top, width, height)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 190))
    painter.drawRoundedRect(box, 3, 3)
    painter.setPen(QPen(QColor("#ffffff"), 1))
    painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
