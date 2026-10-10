"""グラフエディタ キーフレームの値と補間曲線を直接いじる

数値欄だけでも打点はできるが、「じわっと始めて最後に止める」ような動きは
曲線を見ないと調整できない 値の時間変化を線として見せ、点をつまんで動かせる
ようにする

選んだクリップに付いてくる（:meth:`GraphEditor.set_clip`） 前は設定パネルの ◆ の
右クリックの「グラフエディタで開く」でしか値を渡せず、キーフレームを入れたクリップを
選んでグラフエディタを開いても「パラメータを選んでください」のままで何も触れなかった
いまはクリップを選ぶとキーフレームのある最初の値を出し、上の欄でほかの値へ切り替えられる

自分ではプロジェクトを書き換えない 操作はコマンドとして外へ出す
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from sashimono.core.commands import (
    Command,
    MoveKeyframe,
    ParamPath,
    ParamTarget,
    RemoveKeyframe,
    SetKeyframe,
    resolve_param,
)
from sashimono.core.model import AnimatedValue, Clip, ClipId, Interpolation, Keyframe, Project
from sashimono.effects import ParameterSpec, TrackSpec, registry
from sashimono.effects.sources import source_registry
from sashimono.effects.strokes import STROKE
from sashimono.ui.theme import Colors, themed_style

__all__ = ["GraphEditor", "curve_choices"]

#: クリップ自身の値で、グラフにできる物（不透明度だけ）
_OPACITY = TrackSpec("opacity", "不透明度", 0, 1, 1, step=0.01)


def curve_choices(clip: Clip) -> list[tuple[str, ParamPath, bool]]:
    """``clip`` のグラフにできる値 ``(表示名, 在りか, キーフレームがあるか)`` の並び

    数のスライダー（:class:`TrackSpec`）の値だけ 並びは設定パネルと同じ
    （不透明度 → 中身 → エフェクト → 後の場面のエフェクト）
    """

    def animated(value: object) -> bool:
        return isinstance(value, AnimatedValue) and value.is_animated

    choices: list[tuple[str, ParamPath, bool]] = [
        (
            f"クリップ: {_OPACITY.label}",
            ParamPath.of_clip(clip.id, "opacity"),
            animated(clip.opacity),
        )
    ]
    source = source_registry.get(clip.source.kind) if clip.source is not None else None
    if source is not None and clip.source is not None:
        for spec in source.parameters:
            if isinstance(spec, TrackSpec):
                choices.append(
                    (
                        f"{source.label}: {spec.label}",
                        ParamPath.of_source(clip.id, spec.name),
                        animated(clip.source.params.get(spec.name)),
                    )
                )
    # テキストの縁取りの層（#272）と層のエフェクト（#273） 設定パネルと同じく中身のすぐ後ろ
    strokes = clip.source.strokes if clip.source is not None else ()
    for number, stroke in enumerate(strokes, start=1):
        for spec in STROKE.parameters:
            if isinstance(spec, TrackSpec):
                choices.append(
                    (
                        f"縁取り {number}: {spec.label}",
                        ParamPath.of_stroke(clip.id, stroke.id, spec.name),
                        animated(stroke.params.get(spec.name)),
                    )
                )
        for effect in stroke.effects:
            definition = registry.get(effect.kind)
            if definition is None:
                continue
            for spec in definition.parameters:
                if isinstance(spec, TrackSpec):
                    choices.append(
                        (
                            f"{definition.label}（縁取り {number}）: {spec.label}",
                            ParamPath.of_stroke_effect(clip.id, stroke.id, effect.id, spec.name),
                            animated(effect.params.get(spec.name)),
                        )
                    )
    for after, effects in ((False, clip.effects), (True, clip.after_effects)):
        for effect in effects:
            definition = registry.get(effect.kind)
            if definition is None:
                continue
            owner = f"{definition.label}（後の場面）" if after else definition.label
            for spec in definition.parameters:
                if isinstance(spec, TrackSpec):
                    choices.append(
                        (
                            f"{owner}: {spec.label}",
                            ParamPath.of_effect(clip.id, effect.id, spec.name, after=after),
                            animated(effect.params.get(spec.name)),
                        )
                    )
    return choices


#: 補間方法の表示名
INTERPOLATION_LABELS: dict[Interpolation, str] = {
    Interpolation.HOLD: "瞬間移動",
    Interpolation.LINEAR: "直線",
    Interpolation.EASE_IN: "加速",
    Interpolation.EASE_OUT: "減速",
    Interpolation.EASE_IN_OUT: "加減速",
    Interpolation.BEZIER: "曲線",
}

#: 点をつかめる距離（ピクセル）
_GRAB_RADIUS = 8

#: グラフの余白 値の上下端が枠に張り付くと、つまみにくい
_MARGIN = 18


@dataclass(frozen=True, slots=True)
class _Plot:
    """グラフの座標変換 値とフレームを画面座標へ"""

    width: int
    height: int
    start_frame: int
    end_frame: int
    minimum: float
    maximum: float

    def to_x(self, frame: float) -> float:
        span = max(1, self.end_frame - self.start_frame)
        usable = self.width - _MARGIN * 2
        return _MARGIN + (frame - self.start_frame) / span * usable

    def to_y(self, value: float) -> float:
        span = self.maximum - self.minimum or 1.0
        usable = self.height - _MARGIN * 2
        # 値が大きいほど上 グラフとしての向き
        return self.height - _MARGIN - (value - self.minimum) / span * usable

    def to_frame(self, x: float) -> int:
        span = max(1, self.end_frame - self.start_frame)
        usable = max(1, self.width - _MARGIN * 2)
        return round(self.start_frame + (x - _MARGIN) / usable * span)

    def to_value(self, y: float) -> float:
        span = self.maximum - self.minimum or 1.0
        usable = max(1, self.height - _MARGIN * 2)
        return self.minimum + (self.height - _MARGIN - y) / usable * span


class GraphEditor(QWidget):
    """1 つのパラメータの時間変化を編集する"""

    commands_requested = Signal(list, str)
    #: 再生ヘッドを動かしたい グラフ上をクリックしたとき
    seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._path: ParamPath | None = None
        #: 選んでいるクリップ（:meth:`set_clip`） 値の選び直しはこのクリップの中で行う
        self._clip_id: ClipId | None = None
        self._frame = 0
        self._dragging: int | None = None

        self._title = QLabel("パラメータを選んでください", self)
        themed_style(self._title, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        #: 曲線を出す値を選ぶ欄 並びは :func:`curve_choices` キーフレームのある値に ◆ を付ける
        self._params = QComboBox(self)
        self._params.setToolTip("曲線を出す値 ◆ はキーフレームのある値")
        self._params.currentIndexChanged.connect(self._on_param_chosen)
        #: 欄の項目ごとの在りか（先頭は「選んでいない」の ``None``）
        self._choice_paths: list[ParamPath | None] = []

        self._interpolation = QComboBox(self)
        for kind, label in INTERPOLATION_LABELS.items():
            self._interpolation.addItem(label, kind)
        self._interpolation.setEnabled(False)
        self._interpolation.currentIndexChanged.connect(self._on_interpolation)

        header = QHBoxLayout()
        header.setContentsMargins(8, 4, 8, 4)
        header.addWidget(self._title)
        header.addWidget(self._params, 1)
        header.addWidget(QLabel("補間", self))
        header.addWidget(self._interpolation)

        self._canvas = _Canvas(self)
        self._canvas.changed.connect(self._on_canvas_command)
        self._canvas.seek_requested.connect(self.seek_requested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self._canvas, 1)
        self.setMinimumHeight(160)

    def set_project(self, project: Project) -> None:
        self._project = project
        if self._path is not None and self._spec() is None:
            # 出していた値が無くなった（エフェクトを外した・クリップを消した・取り消した）
            # 残すと、無い値の空の曲線のまま触れなくなる
            self._path = self._first_animated()
        self._refresh()

    def set_path(self, path: ParamPath | None) -> None:
        self._path = path
        if path is not None:
            self._clip_id = path.clip_id
        self._refresh()

    def set_clip(self, clip_id: ClipId | None) -> None:
        """選んだクリップ（設定パネルが出している 1 本） キーフレームのある最初の値を出す

        同じクリップの値を出していれば、それを続けて出す 選び直すたびに先頭の値へ戻ると、
        2 つ目の値の曲線を直している途中で、タイムラインを押すたびに別の値へ飛ぶ
        """
        self._clip_id = clip_id
        if clip_id is None:
            self._path = None
        elif self._path is None or self._path.clip_id != clip_id or self._spec() is None:
            self._path = self._first_animated()
        self._refresh()

    @property
    def path(self) -> ParamPath | None:
        """いま曲線を出している値"""
        return self._path

    def choices(self) -> list[str]:
        """値を選ぶ欄に並んでいる表示名（先頭の「選んでいない」を除く）"""
        return [self._params.itemText(i) for i in range(1, self._params.count())]

    def choose(self, index: int) -> None:
        """値を選ぶ欄の ``index`` 番目（:meth:`choices` の並び）を選ぶ"""
        self._params.setCurrentIndex(index + 1)

    def _clip(self) -> Clip | None:
        if self._project is None or self._clip_id is None:
            return None
        located = self._project.timeline.locate_clip(self._clip_id)
        return located[1] if located is not None else None

    def _first_animated(self) -> ParamPath | None:
        clip = self._clip()
        if clip is None:
            return None
        return next((path for _, path, animated in curve_choices(clip) if animated), None)

    def _fill_choices(self) -> None:
        """値を選ぶ欄を今のクリップで作り直す 選んでいる値を選んだ形にする"""
        clip = self._clip()
        choices = curve_choices(clip) if clip is not None else []
        self._params.blockSignals(True)
        try:
            self._params.clear()
            self._choice_paths = [None]
            self._params.addItem("値を選ぶ" if choices else "クリップを選んでください")
            for label, path, animated in choices:
                self._params.addItem(f"◆ {label}" if animated else label)
                self._choice_paths.append(path)
            index = self._choice_paths.index(self._path) if self._path in self._choice_paths else 0
            self._params.setCurrentIndex(index)
            self._params.setEnabled(bool(choices))
        finally:
            self._params.blockSignals(False)

    def _on_param_chosen(self, index: int) -> None:
        if 0 <= index < len(self._choice_paths):
            self._path = self._choice_paths[index]
            self._refresh()

    def set_frame(self, frame: int) -> None:
        self._frame = frame
        self._canvas.set_frame(frame)
        value = self._value()
        if isinstance(value, AnimatedValue) and value.is_animated:
            self._show_interpolation(value)

    def _refresh(self) -> None:
        self._fill_choices()
        spec = self._spec()
        value = self._value()
        described = self._describe(spec)
        self._title.setText(described)
        self._title.setVisible(bool(described))

        animated = isinstance(value, AnimatedValue) and value.is_animated
        self._interpolation.setEnabled(animated)
        if animated:
            assert value is not None
            self._show_interpolation(value)
        self._canvas.set_curve(self._path, spec, value, self._clip_start())

    def _show_interpolation(self, value: AnimatedValue) -> None:
        """再生ヘッドの区間の補間方法を選択欄へ反映する

        常に先頭の項目を出していると、実際は直線なのに「瞬間移動」と
        表示され続けることになる
        """
        current = self._active_keyframe(value)
        if current is None:
            return
        index = self._interpolation.findData(current.interpolation)
        if index < 0:
            return
        self._interpolation.blockSignals(True)
        try:
            self._interpolation.setCurrentIndex(index)
        finally:
            self._interpolation.blockSignals(False)

    def _active_keyframe(self, value: AnimatedValue) -> Keyframe | None:
        """再生ヘッドが乗っている区間の始点 区間の性質はここが持つ"""
        local = self._frame - self._clip_start()
        found = None
        for keyframe in value.keyframes:
            if keyframe.frame <= local:
                found = keyframe
        return found or (value.keyframes[0] if value.keyframes else None)

    def _describe(self, spec: TrackSpec | None) -> str:
        if self._path is None or spec is None:
            if self._clip() is not None:
                # キーフレームが 1 つも無いクリップ 打ち方を言わないと、なぜ曲線が出ないのか
                # 分からない
                return "キーフレームは設定パネルの ◆ で打てます"
            return "パラメータを選んでください"
        # どの値かは隣の欄が出している 同じ名前を 2 度並べると、幅の狭いドックで欄が潰れる
        return ""

    def _clip_start(self) -> int:
        """クリップ先頭のフレーム キーフレームはここからの相対で持つ"""
        if self._project is None or self._path is None:
            return 0
        located = self._project.timeline.locate_clip(self._path.clip_id)
        return located[1].timeline_start if located is not None else 0

    def _value(self) -> AnimatedValue | None:
        if self._project is None or self._path is None:
            return None
        value = resolve_param(self._project, self._path)
        return value if isinstance(value, AnimatedValue) else None

    def _spec(self) -> TrackSpec | None:
        """編集対象の仕様 数値スライダー以外はグラフにできない"""
        if self._project is None or self._path is None:
            return None
        located = self._project.timeline.locate_clip(self._path.clip_id)
        if located is None:
            return None
        _, clip = located
        if self._path.target is ParamTarget.CLIP:
            return _OPACITY if self._path.name == _OPACITY.name else None

        # エフェクトと生成オブジェクトは別の型だが、spec() の形は同じ
        # 欲しいのはパラメータ仕様だけなので、ここで 1 本にまとめる
        spec: ParameterSpec | None = None
        stroke = (
            clip.source.find_stroke(self._path.stroke_id)
            if self._path.stroke_id is not None and clip.source is not None
            else None
        )
        if self._path.stroke_id is not None and stroke is None:
            return None
        if self._path.target is ParamTarget.SOURCE:
            if stroke is not None:
                spec = STROKE.spec(self._path.name)
            elif clip.source is not None:
                source = source_registry.get(clip.source.kind)
                spec = source.spec(self._path.name) if source is not None else None
        else:
            # 場面切り替えの後の場面のエフェクトは別の列にある 前の列だけを探すと、
            # 後の場面の値を選んでも曲線が出ない 縁取りの層のエフェクトは層の列にある
            if stroke is not None:
                stack = stroke.effects
            else:
                stack = clip.after_effects if self._path.after else clip.effects
            effect = next((e for e in stack if e.id == self._path.effect_id), None)
            definition = registry.get(effect.kind) if effect is not None else None
            spec = definition.spec(self._path.name) if definition is not None else None

        return spec if isinstance(spec, TrackSpec) else None

    def _on_canvas_command(self, command: Command) -> None:
        self.commands_requested.emit([command], command.label)

    def _on_interpolation(self, index: int) -> None:
        value = self._value()
        if self._path is None or value is None or not value.is_animated:
            return
        kind = self._interpolation.itemData(index)
        # 再生ヘッドの手前にあるキーフレームの出方を変える 区間の性質は
        # 「その区間の始点」が持っているため
        target = self._active_keyframe(value)
        if target is None:
            return

        self.commands_requested.emit(
            [
                SetKeyframe(
                    self._path,
                    target.frame,
                    target.value,
                    interpolation=kind,
                    control_points=(0.42, 0.0, 0.58, 1.0) if kind is Interpolation.BEZIER else None,
                )
            ],
            "補間方法を変更",
        )


class _Canvas(QWidget):
    """グラフの描画と、点の操作"""

    changed = Signal(object)
    seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path: ParamPath | None = None
        self._spec: TrackSpec | None = None
        self._value: AnimatedValue | None = None
        self._clip_start = 0
        self._frame = 0
        self._dragging: int | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_curve(
        self,
        path: ParamPath | None,
        spec: TrackSpec | None,
        value: AnimatedValue | None,
        clip_start: int,
    ) -> None:
        self._path = path
        self._spec = spec
        self._value = value
        self._clip_start = clip_start
        self.update()

    def set_frame(self, frame: int) -> None:
        self._frame = frame
        self.update()

    def _plot(self) -> _Plot | None:
        if self._spec is None or self._value is None:
            return None
        frames = [k.frame for k in self._value.keyframes]
        start = min([*frames, 0])
        end = max([*frames, start + 30])
        # 値の範囲は仕様の全域ではなく、実際に使っている範囲に合わせる
        # 0..4000 の仕様で 0..100 しか使っていないと、線がほぼ平らに見える
        values = [k.value for k in self._value.keyframes] or [self._spec.default]
        low, high = min(values), max(values)
        if high - low < 1e-6:
            low, high = low - 1.0, high + 1.0
        pad = (high - low) * 0.15
        return _Plot(self.width(), self.height(), start, end + 1, low - pad, high + pad)

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt の命名規約
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Colors.TIMELINE_BACKGROUND)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        plot = self._plot()
        if plot is None or self._value is None:
            painter.setPen(QPen(Colors.TEXT_MUTED, 1))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "キーフレームのあるパラメータを選ぶと、ここに曲線が出ます",
            )
            return

        self._draw_grid(painter, plot)
        self._draw_curve(painter, plot)
        self._draw_keyframes(painter, plot)
        self._draw_playhead(painter, plot)

    def _draw_grid(self, painter: QPainter, plot: _Plot) -> None:
        painter.setPen(QPen(Colors.TRACK_SEPARATOR, 1))
        for step in range(5):
            y = _MARGIN + (self.height() - _MARGIN * 2) * step / 4
            painter.drawLine(QPointF(_MARGIN, y), QPointF(self.width() - _MARGIN, y))

        painter.setPen(QPen(Colors.TEXT_MUTED, 1))
        painter.drawText(QPointF(4, _MARGIN + 4), f"{plot.maximum:.4g}")
        painter.drawText(QPointF(4, self.height() - _MARGIN + 4), f"{plot.minimum:.4g}")

    def _draw_curve(self, painter: QPainter, plot: _Plot) -> None:
        assert self._value is not None
        path = QPainterPath()
        # 実際の評価関数を 1 ピクセルずつ引く 曲線の式を描画側で作り直すと、
        # 表示と実際の動きがずれる
        for x in range(_MARGIN, self.width() - _MARGIN + 1):
            frame = plot.to_frame(x)
            y = plot.to_y(self._value.at(frame))
            if x == _MARGIN:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(Colors.ACCENT, 2))
        painter.drawPath(path)

    def _draw_keyframes(self, painter: QPainter, plot: _Plot) -> None:
        assert self._value is not None
        painter.setPen(QPen(Colors.SELECTION, 1))
        for index, keyframe in enumerate(self._value.keyframes):
            centre = QPointF(plot.to_x(keyframe.frame), plot.to_y(keyframe.value))
            painter.setBrush(Colors.PLAYHEAD if index == self._dragging else Colors.SELECTION)
            painter.drawRect(QRectF(centre.x() - 4, centre.y() - 4, 8, 8))

    def _draw_playhead(self, painter: QPainter, plot: _Plot) -> None:
        x = plot.to_x(self._frame - self._clip_start)
        if _MARGIN <= x <= self.width() - _MARGIN:
            painter.setPen(QPen(Colors.PLAYHEAD, 1))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))

    # --- 入力 ---

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        plot = self._plot()
        if plot is None or self._value is None or self._path is None:
            return
        position = event.position()

        index = self._hit(plot, position)
        if event.button() == Qt.MouseButton.RightButton:
            if index is not None:
                self.changed.emit(RemoveKeyframe(self._path, self._value.keyframes[index].frame))
            return

        if index is not None:
            self._dragging = index
            self.update()
            return

        # 何も無い場所を押したら再生ヘッドを動かす 曲線と再生位置を
        # 見比べながら調整できる
        self.seek_requested.emit(self._clip_start + plot.to_frame(position.x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self._dragging is None or self._value is None or self._path is None:
            return
        plot = self._plot()
        if plot is None or self._spec is None:
            return

        keyframe = self._value.keyframes[self._dragging]
        value = self._spec.clamp(plot.to_value(event.position().y()))
        frame = max(0, plot.to_frame(event.position().x()))

        # 隣のキーフレームを追い越させない 追い越すと順序が崩れ、
        # モデル側の検査で弾かれる
        others = [k.frame for i, k in enumerate(self._value.keyframes) if i != self._dragging]
        lower = max([f for f in others if f < keyframe.frame], default=-1)
        upper = min([f for f in others if f > keyframe.frame], default=10**9)
        frame = min(max(frame, lower + 1), upper - 1)

        self.changed.emit(MoveKeyframe(self._path, keyframe.frame, frame, value))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        del event
        self._dragging = None
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt の命名規約
        deleting = event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
        selected = self._path is not None and self._value is not None and self._dragging is not None
        if deleting and selected:
            assert self._path is not None
            assert self._value is not None
            assert self._dragging is not None
            self.changed.emit(
                RemoveKeyframe(self._path, self._value.keyframes[self._dragging].frame)
            )
            return
        super().keyPressEvent(event)

    def _hit(self, plot: _Plot, position: QPointF) -> int | None:
        assert self._value is not None
        for index, keyframe in enumerate(self._value.keyframes):
            centre = QPointF(plot.to_x(keyframe.frame), plot.to_y(keyframe.value))
            if (centre - position).manhattanLength() <= _GRAB_RADIUS * 2:
                return index
        return None
