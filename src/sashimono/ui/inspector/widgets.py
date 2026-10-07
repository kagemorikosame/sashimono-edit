"""パラメータ 1 つ分の入力欄

種類ごとにウィジェットを 1 つずつ用意し、:class:`~sashimono.effects.ParameterSpec`
から自動で選ぶ エフェクトを増やしても UI を書き足す必要は無く、AviUtl の
スクリプトを読み込んだとき（P5）も同じ経路で設定欄が出る
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QWidget,
)

from sashimono.core.model import AnimatedValue, ParamValue
from sashimono.effects import (
    CheckSpec,
    ColorSpec,
    FileSpec,
    FontSpec,
    FontStyleSpec,
    GridSpec,
    ParameterSpec,
    SelectSpec,
    TextSpec,
    TrackSpec,
    ValueSpec,
)
from sashimono.ui.flow_layout import narrow_combo
from sashimono.ui.theme import Colors, themed_style

__all__ = [
    "FONT_STYLE_DEFAULT_LABEL",
    "NUMBER_WIDTH",
    "TYPING_MERGE_SECONDS",
    "FontStyleEditor",
    "ParameterEditor",
    "TextEditor",
    "TrackEditor",
    "create_editor",
]

#: 数値欄の幅の下限（画素） 中身がこれより狭くても、行ごとに欄の幅が揺れないようにそろえる
NUMBER_WIDTH = 96

#: 文字の欄で、打ち始めからこの秒数までの打鍵を取り消しの 1 段にまとめる
#: 1 文字ごとに段を積むと、打った言葉を戻すのに文字の数だけ取り消すことになる
#: 欄を離れたら（フォーカスが外れたら）秒数に関わらず区切る 長い文を打ち続けた
#: ときに、1 回の取り消しで全部が消えないよう、時間でも区切る
TYPING_MERGE_SECONDS = 3.0

#: 打鍵の時刻を測る時計 試験で時間を進めるために差し替えられるようにしておく
_clock: Callable[[], float] = time.monotonic

#: スライダーは整数しか扱えないので、この倍率で小数を載せる
_SLIDER_SCALE = 1000
#: Qt の整数の欄（``QSpinBox`` ``QSlider``）が持てる範囲 C++ の ``int`` は 4 バイト
_INT_MIN = -(2**31)
_INT_MAX = 2**31 - 1


def _qt_int(number: float) -> int:
    """Qt の整数の欄へ渡せる値へ丸める

    収まらない値をそのまま渡すと、shiboken が警告を出したうえで Qt 側の値が化け、
    入れられる範囲が意図と違ってしまう（#148） 仕様は AviUtl のスクリプトからも
    来るので、どれほど大きな範囲が書かれていてもここで収める
    """
    return min(max(int(number), _INT_MIN), _INT_MAX)


#: 数値欄に出す小数の桁の上限 これより細かい刻みは画面で読めない
_MAX_DECIMALS = 6


def _decimals_for(spec: TrackSpec) -> int:
    """刻みと範囲の両端をそのまま表せる小数の桁数

    刻みが 1 以上なら整数、それより細かければ小数第 2 位までをまず取り、刻みや端が
    それより細かい桁を持つなら、その桁まで広げる 足りないと端が丸められ、数値欄が
    仕様の範囲の外の値を返す
    """
    decimals = 0 if spec.step >= 1 else 2
    for number in (spec.step, spec.minimum, spec.maximum):
        needed = next(
            (
                places
                for places in range(_MAX_DECIMALS + 1)
                if abs(round(number, places) - number) <= 1e-9 * max(1.0, abs(number))
            ),
            _MAX_DECIMALS,
        )
        decimals = max(decimals, needed)
    return decimals


class ParameterEditor(QWidget):
    """パラメータ入力欄の共通の親

    値が確定したら :attr:`value_changed` を出す ドラッグ中の途中経過は
    :attr:`value_previewed` で、こちらは履歴に残さない前提
    """

    #: 値が確定した 履歴に残る変更
    value_changed = Signal(object)
    #: ドラッグ中の途中経過 プレビューだけ更新する
    value_previewed = Signal(object)
    #: 初期値へ戻したい（数値のスライダーのダブルクリック） 戻し方はパネルが決める
    #: （キーフレームのある値は再生位置のキーだけを戻す）
    reset_requested = Signal()
    #: 値が確定した 直前の確定の続き（文字の欄で続けて打った分）で、履歴の同じ段へ
    #: まとめてよい :attr:`value_changed` と分けるのは、まとめてよいかを欄しか知らないから
    value_continued = Signal(object)
    #: 押している最中の操作（スライダーのドラッグ・増減のボタンの長押し）が終わった
    #: 設定パネルは、操作の最中に来た作り直しをここまで待つ
    interaction_finished = Signal()

    def __init__(self, spec: ParameterSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._updating = False

    def set_value(self, value: ParamValue | None) -> None:
        """外から値を入れ直す 信号は出さない

        入れ直しで信号を出すと、プロジェクトの更新 → UI 更新 → 変更通知 →
        プロジェクトの更新、と回り続ける
        """
        raise NotImplementedError

    def is_busy(self) -> bool:
        """マウスで押している最中か 最中に作り直すと、掴んでいた部品が消えて操作が切れる"""
        return False

    def _emit(self, value: ParamValue) -> None:
        if not self._updating:
            self.value_changed.emit(value)

    def _preview(self, value: ParamValue) -> None:
        if not self._updating:
            self.value_previewed.emit(value)


class TrackEditor(ParameterEditor):
    """数値スライダーと数値欄の組

    スライダーだけだと細かい値を入れられず、数値欄だけだと感覚的に動かせない
    両方を出して同期させる
    """

    def __init__(self, spec: TrackSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec

        # 1000 倍すると int に収まらない広い範囲は、スライダーの端から端を仕様の最小から
        # 最大へ割り当て直す 端だけを int の上限で切ると、スライダーの端が仕様の端を
        # 表さず、範囲の外の値まで数値欄と保存へ流れる
        low = spec.minimum * _SLIDER_SCALE
        high = spec.maximum * _SLIDER_SCALE
        self._stretched = not (low >= _INT_MIN and high <= _INT_MAX)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        if self._stretched:
            self._slider.setRange(0, _INT_MAX)
        else:
            self._slider.setRange(int(low), int(high))
        # 矢印キーとページのキーで動く幅を、仕様の刻みにする Qt の既定（1 目盛り）は
        # 1000 倍した 1 で、仕様では 0.001 にしかならない 整数の欄では表示の桁で 0 へ
        # 丸められ、矢印キーを何度押しても値が動かなかった
        step = self._to_slider(spec.minimum + spec.step) - self._to_slider(spec.minimum)
        self._slider.setSingleStep(max(1, step))
        self._slider.setPageStep(max(1, _qt_int(step * 10)))
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderPressed.connect(self._on_press)
        self._slider.sliderReleased.connect(self._on_release)
        # スライダーのダブルクリックで初期値に戻す 数値欄のダブルクリックは今までどおり
        # 数字を選ぶ（打ち直す）ために残す
        self._slider.installEventFilter(self)
        #: 押した**前**の値 離したときに比べ、変わっていなければ履歴へ積まない
        #: 押した瞬間につまみが押した所へ飛ぶ見た目（Windows）があるので、スライダーが
        #: 知らせる前（:meth:`eventFilter` の押下）に取る 知らせを受けてから取ると、飛んだ後の
        #: 値と比べることになり、押して離しただけで段が積まれる
        self._pressed_at: float | None = None
        #: マウスを押している間 押している間は、つまみを掴んでいなくても（溝の長押しで
        #: 値が進む見た目がある）プレビューだけにし、離したときに 1 度だけ確定する
        self._held = False
        #: 押した所 動かしたかを見る
        self._press_position: QPointF | None = None
        #: つまみの上を押したか つまみを押して動かさずに離したら、値は変えない（押した所へ
        #: 数画素飛んだ分も戻す） 溝を押して離したら、飛んだ所で確定する
        self._on_handle = False
        #: 押してから、ドラッグと見なす距離より動かしたか（溝を押したときだけに使う）
        self._moved = False
        #: スライダーが押下を受け終えた時点の値（押した所へ飛んだ後） 離したときにこれと
        #: 同じなら「動かしていない」 動かしたかをマウスの距離で決めると、つまみを掴んで
        #: 数画素だけ動かす細かい合わせ（初期値の近くでよくやる）が、動かしていない扱いに
        #: なって捨てられた
        self._after_press: float | None = None
        #: 2 回目の押下（ダブルクリック）で押している 動かさずに離したら初期値へ戻し、
        #: 動かしたらふつうのドラッグにする
        self._double = False
        #: 離したときのスライダーの知らせ（:meth:`_on_release`）で確定しない
        #: （初期値へ戻す・つまみを押して離しただけ） :meth:`eventFilter` が決める
        self._skip_release = False

        self._number = QDoubleSpinBox(self)
        # 桁数は範囲より先に決める Qt は範囲も値も今の桁数へ丸めるので、後から決めると
        # 下限 0.004 が 0.00 になり、仕様の範囲の外の値が数値欄からプレビューと保存へ流れる
        self._number.setDecimals(_decimals_for(spec))
        self._number.setRange(spec.minimum, spec.maximum)
        self._number.setSingleStep(spec.step)
        self._number.setSuffix(f" {spec.unit}" if spec.unit else "")
        self._number.setFixedWidth(self.number_width())
        self._number.setKeyboardTracking(False)
        self._number.valueChanged.connect(self._on_number)
        self._number.installEventFilter(self)
        #: 数値欄の増減のボタンを押した前の値 押していなければ ``None``
        self._arrow_from: float | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._number)

        self.set_value(spec.default_value())

    def number_width(self) -> int:
        """数値欄に範囲の端の値と単位が欠けずに入る幅

        決め打ちの 96 画素では、書体の幅が広い所（Windows の表示の拡大・別の書体）で
        「100.00 %」の後ろが切れた 範囲の端の値で測るのは、動かしても欄の幅が変わらない
        ようにするため（今の値で測ると、桁が増えるたびに隣のスライダーが縮む）
        """
        return max(NUMBER_WIDTH, self._number.sizeHint().width())

    def set_number_width(self, width: int) -> None:
        """数値欄の幅をそろえる 設定パネルが組の中で一番広い欄に合わせる

        行ごとに幅が違うと、スライダーの右端が行ごとにずれて読みにくい
        """
        self._number.setFixedWidth(max(width, self.number_width()))

    def set_value(self, value: ParamValue | None) -> None:
        animated = self._spec.coerce(value)
        self._apply(animated.static if not animated.is_animated else animated.at(0))

    def set_animated_value(self, value: float) -> None:
        """キーフレームで決まった現在値を表示に反映する"""
        self._apply(value)

    def is_busy(self) -> bool:
        return self._held or self._slider.isSliderDown() or self._arrow_from is not None

    def _apply(self, number: float) -> None:
        self._updating = True
        try:
            self._number.setValue(number)
            self._slider.setValue(self._to_slider(number))
        finally:
            self._updating = False

    def _on_slider(self, raw: int) -> None:
        if self._updating:
            return
        number = self._from_slider(raw)
        self._updating = True
        try:
            self._number.setValue(number)
        finally:
            self._updating = False
        # 数値欄が表示の桁へ丸めた値を使う 広い範囲ではスライダー 1 目盛りが表示の桁より
        # 細かく、元の値を流すと、画面の数字とプレビュー・保存の値が食い違う
        # 桁が上限で足りないほど細かい端でも範囲の外へ出さない
        number = self._spec.clamp(self._number.value())
        if not self._slider.isSliderDown() and not self._held:
            # 矢印キーやホイールで動かした 離したときの知らせが来ない プレビューだけに
            # すると、絵は変わったのに履歴にも保存にも残らず、次に選び直したときに元へ戻る
            self._emit(AnimatedValue(static=number))
            return
        # ドラッグ中は履歴に残さない 1 回のドラッグで数十の取り消し段ができると
        # 元の値まで戻すのに数十回押すことになる
        self._preview(AnimatedValue(static=number))

    def _on_press(self) -> None:
        if not self._held:
            # マウス以外（キーボードや試験）で掴んだ マウスなら押下で取ってある
            self._pressed_at = self._number.value()
            return
        # 押した所へ飛ぶ見た目では、この知らせは飛んだ後に来る
        self._after_press = self._number.value()

    def _on_release(self) -> None:
        if self._skip_release:
            self._skip_release = False
            self._pressed_at = None
        else:
            self._commit_if_changed()
        self.interaction_finished.emit()

    def _commit_if_changed(self) -> None:
        """押す前の値から変わっていれば確定する 同じ値を確定すると、戻しても何も変わらない
        取り消しの段が積まれる 確定もプレビューと同じく数値欄に出ている値にそろえる"""
        number = self._spec.clamp(self._number.value())
        pressed, self._pressed_at = self._pressed_at, None
        if pressed is not None and number == self._spec.clamp(pressed):
            return
        self._emit(AnimatedValue(static=number))

    def _restore(self, before: float) -> None:
        """押した所へ動いた表示とプレビューを、押す前の値へ戻す（確定はしない）"""
        if self._number.value() == before:
            return
        self._apply(before)
        self._preview(AnimatedValue(static=self._spec.clamp(before)))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の命名規約
        """スライダーのダブルクリックで初期値へ戻す 押したまま動かせばふつうのドラッグ

        2 回目の押下はダブルクリックとして届く ここで握りつぶすと、クリックの直後に
        押したまま動かして値を合わせる操作（利用者の言う長押しの調整）で、つまみが
        掴めずに値が初期値へ飛んだ 押下はスライダーへ通し、動かさずに離したときだけ戻す
        """
        if watched is self._number and isinstance(event, QMouseEvent):
            self._watch_arrows(event)
            return super().eventFilter(watched, event)
        if watched is not self._slider or not isinstance(event, QMouseEvent):
            return super().eventFilter(watched, event)
        kind = event.type()
        left = event.button() == Qt.MouseButton.LeftButton
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick) and left:
            # スライダーが値を動かす前に、押す前の値と押した所を取る
            self._held = True
            self._pressed_at = self._number.value()
            self._press_position = event.position()
            self._on_handle = self._handle_at(event.position())
            self._moved = False
            self._after_press = None
            self._double = kind == QEvent.Type.MouseButtonDblClick
            self._skip_release = False
        elif kind == QEvent.Type.MouseMove and self._held and self._press_position is not None:
            distance = (event.position() - self._press_position).manhattanLength()
            if distance >= QApplication.startDragDistance():
                self._moved = True
        elif kind == QEvent.Type.MouseButtonRelease and left and self._held:
            self._held = False
            before = self._pressed_at if self._pressed_at is not None else self._number.value()
            still = self._still()
            if self._double and still:
                # 動かさないダブルクリック 押した所へ動いた分は確定せず、初期値を頼む
                self._finish_without_commit(before)
                self.reset_requested.emit()
            elif self._on_handle and still:
                # つまみを押して離しただけ 押した瞬間に数画素飛んだ分も戻す
                self._finish_without_commit(before)
            elif not self._slider.isSliderDown():
                # 溝を押した（長押しで進んだ分も含む） 掴んでいないので、スライダーからの
                # 離した知らせは来ない ここで 1 度だけ確定する
                self._commit_if_changed()
            self._double = False
            self.interaction_finished.emit()
        return super().eventFilter(watched, event)

    def _watch_arrows(self, event: QMouseEvent) -> None:
        """数値欄の増減のボタンを押した・離した 文字の所の押下は数値欄の中の部品へ届くので、
        ここへ来るのはボタンの所だけ"""
        kind = event.type()
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            self._arrow_from = self._number.value()
        elif kind == QEvent.Type.MouseButtonRelease and self._arrow_from is not None:
            before, self._arrow_from = self._arrow_from, None
            number = self._spec.clamp(self._number.value())
            if number != self._spec.clamp(before):
                self._emit(AnimatedValue(static=number))
            self.interaction_finished.emit()

    def _still(self) -> bool:
        """押してから離すまで値を動かしていないか

        つまみを掴んでいれば、押下を受け終えた時の値（押した所へ飛んだ後）と今の値で比べる
        1 画素でも値が動けば動かした扱い 溝を押した見た目（掴んでいない）では、押している間に
        値が進むので、マウスを動かしたかで見る
        """
        if self._after_press is not None:
            return self._spec.clamp(self._number.value()) == self._spec.clamp(self._after_press)
        return not self._moved

    def _finish_without_commit(self, before: float) -> None:
        """確定せずに押す前の値へ戻す 掴んだ扱いを解いたときの知らせでも確定させない"""
        self._skip_release = self._slider.isSliderDown()
        self._pressed_at = None
        self._restore(before)

    def _handle_at(self, position: QPointF) -> bool:
        """``position`` がつまみの上か 見た目ごとの形はスタイルに尋ねる"""
        option = QStyleOptionSlider()
        self._slider.initStyleOption(option)
        control = self._slider.style().hitTestComplexControl(
            QStyle.ComplexControl.CC_Slider, option, position.toPoint(), self._slider
        )
        return bool(control == QStyle.SubControl.SC_SliderHandle)

    def _to_slider(self, number: float) -> int:
        """仕様の値をスライダーの位置へ 範囲の外は端へ寄せる"""
        number = self._spec.clamp(number)
        if not self._stretched:
            return _qt_int(number * _SLIDER_SCALE)
        span = self._spec.maximum - self._spec.minimum
        if span <= 0:
            return 0
        return _qt_int(round((number - self._spec.minimum) / span * _INT_MAX))

    def _from_slider(self, raw: int) -> float:
        """スライダーの位置を仕様の値へ 必ず仕様の範囲に収める"""
        if not self._stretched:
            return self._spec.clamp(raw / _SLIDER_SCALE)
        span = self._spec.maximum - self._spec.minimum
        return self._spec.clamp(self._spec.minimum + raw / _INT_MAX * span)

    def _on_number(self, number: float) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._slider.setValue(self._to_slider(number))
        finally:
            self._updating = False
        if self._arrow_from is not None:
            # 増減のボタンを押している間（長押しで何段も進む）はプレビューだけにし、離したときに
            # 1 度だけ確定する 1 段ずつ確定すると、長押し 1 回で取り消しが何段も積まれる
            self._preview(AnimatedValue(static=number))
            return
        self._emit(AnimatedValue(static=number))


class CheckEditor(ParameterEditor):
    def __init__(self, spec: CheckSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._box = QCheckBox(self)
        self._box.toggled.connect(lambda state: self._emit(bool(state)))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._box)
        layout.addStretch(1)
        self.set_value(spec.default_value())

    def set_value(self, value: ParamValue | None) -> None:
        self._updating = True
        try:
            self._box.setChecked(self._spec.coerce(value))
        finally:
            self._updating = False


class ColorEditor(ParameterEditor):
    """色見本のボタン 押すと色選択ダイアログが出る"""

    def __init__(self, spec: ColorSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._value: tuple[float, ...] = spec.default_value()

        self._button = QPushButton(self)
        self._button.setFixedHeight(24)
        self._button.clicked.connect(self._choose)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._button, 1)
        self.set_value(spec.default_value())

    def set_value(self, value: ParamValue | None) -> None:
        self._value = self._spec.coerce(value)
        red, green, blue, alpha = (round(c * 255) for c in self._value)
        # 明るい色の上に黒、暗い色の上に白を置く どちらか一方だと必ず読めなくなる
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        text = "#000000" if luminance > 140 else "#ffffff"
        # 地と文字は選んだ色で決まり、枠だけテーマで変わる
        swatch = f"background-color: rgba({red}, {green}, {blue}, {alpha}); color: {text};"
        themed_style(self._button, lambda: f"{swatch} border: 1px solid {Colors.BORDER.name()};")
        self._button.setText(f"#{red:02X}{green:02X}{blue:02X}")

    def _choose(self) -> None:
        from PySide6.QtGui import QColor

        red, green, blue, alpha = (round(c * 255) for c in self._value)
        options = (
            QColorDialog.ColorDialogOption.ShowAlphaChannel
            if self._spec.with_alpha
            else QColorDialog.ColorDialogOption(0)
        )
        chosen = QColorDialog.getColor(QColor(red, green, blue, alpha), self, "色を選ぶ", options)
        if not chosen.isValid():
            return
        value = (
            chosen.red() / 255.0,
            chosen.green() / 255.0,
            chosen.blue() / 255.0,
            chosen.alpha() / 255.0,
        )
        self.set_value(value)
        self._emit(value)


class SelectEditor(ParameterEditor):
    def __init__(self, spec: SelectSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._box = QComboBox(self)
        for identifier, label in spec.choices:
            self._box.addItem(label, identifier)
        self._box.currentIndexChanged.connect(
            lambda index: self._emit(str(self._box.itemData(index)))
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._box, 1)
        self.set_value(spec.default_value())

    def set_value(self, value: ParamValue | None) -> None:
        self._updating = True
        try:
            index = self._box.findData(self._spec.coerce(value))
            self._box.setCurrentIndex(max(0, index))
        finally:
            self._updating = False


class TextEditor(ParameterEditor):
    """文字の欄 打つたびに確定する（打った結果をすぐプレビューで見たい）

    続けて打った分は :attr:`value_continued` で知らせ、取り消しの 1 段にまとめさせる
    区切るのは、欄を離れたときと、打ち始めから :data:`TYPING_MERGE_SECONDS` 経ったとき
    """

    def __init__(self, spec: TextSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._multiline = spec.multiline
        #: 今の打ち続けの始まり（:data:`_clock` の時刻） 打っていなければ ``None``
        self._typing_since: float | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if spec.multiline:
            self._area = QPlainTextEdit(self)
            self._area.setFixedHeight(72)
            self._area.textChanged.connect(lambda: self._on_text(self._area.toPlainText()))
            self._area.installEventFilter(self)
            layout.addWidget(self._area, 1)
        else:
            self._line = QLineEdit(self)
            self._line.textChanged.connect(self._on_text)
            self._line.installEventFilter(self)
            layout.addWidget(self._line, 1)

        self.set_value(spec.default_value())

    @property
    def typing_since(self) -> float | None:
        """今の打ち続けの始まり 設定パネルが作り直すときに新しい欄へ引き継ぐ"""
        return self._typing_since

    def continue_typing(self, since: float | None) -> None:
        """作り直す前の欄の打ち続けを引き継ぐ 引き継がないと、欄の構成が変わる 1 文字
        （タイマーの書式の 1 文字目など）の前後で取り消しの段が分かれる"""
        self._typing_since = since

    def _on_text(self, text: str) -> None:
        if self._updating:
            return
        now = _clock()
        since = self._typing_since
        if since is not None and now - since < TYPING_MERGE_SECONDS:
            self.value_continued.emit(text)
            return
        self._typing_since = now
        self.value_changed.emit(text)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の命名規約
        """欄を離れたら打ち続けを区切る 戻ってきて打った分まで前の段へまとめると、
        別々に直した物が 1 回の取り消しで一緒に戻る"""
        if event.type() == QEvent.Type.FocusOut:
            self._typing_since = None
        return super().eventFilter(watched, event)

    def set_value(self, value: ParamValue | None) -> None:
        """本文が同じなら触らない 打っている欄へ同じ本文を入れ直すと、カーソルが頭へ
        飛び、日本語入力の変換中の文字と欄の中の取り消しが消える

        違う本文（取り消しなど外から来た物）は入れ直し、カーソルは同じ位置
        （本文が短くなったら末尾）に置く
        """
        text = self._spec.coerce(value)
        self._updating = True
        try:
            if self._multiline:
                if self._area.toPlainText() != text:
                    cursor = self._area.textCursor()
                    anchor, position = cursor.anchor(), cursor.position()
                    self._area.setPlainText(text)
                    restored = self._area.textCursor()
                    length = len(self._area.toPlainText())
                    restored.setPosition(min(anchor, length))
                    restored.setPosition(min(position, length), QTextCursor.MoveMode.KeepAnchor)
                    self._area.setTextCursor(restored)
            elif self._line.text() != text:
                position = self._line.cursorPosition()
                self._line.setText(text)
                self._line.setCursorPosition(min(position, len(text)))
        finally:
            self._updating = False


class FontEditor(ParameterEditor):
    def __init__(self, spec: FontSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._box = QFontComboBox(self)
        # 入っている書体の中でいちばん長い名前の幅が、設定パネルの最小の幅になっていた
        # （Windows で 247 画素） テキストを選んだだけで右の列が広がり、窓が画面からはみ出す
        narrow_combo(self._box, 10)
        self._box.currentFontChanged.connect(lambda font: self._emit(font.family()))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._box, 1)
        self.set_value(spec.default_value())

    def set_value(self, value: ParamValue | None) -> None:
        from PySide6.QtGui import QFont

        self._updating = True
        try:
            self._box.setCurrentFont(QFont(self._spec.coerce(value)))
        finally:
            self._updating = False


#: スタイルの欄の先頭 空文字（ファミリ名だけで選ぶ今までの描き方）に当たる
FONT_STYLE_DEFAULT_LABEL = "既定"
#: 選んでいるスタイルが今のファミリに無いときに名前へ添える
_MISSING_STYLE_SUFFIX = "（このフォントに無い）"


class FontStyleEditor(ParameterEditor):
    """フォントのファミリの中のスタイル（太さの段階など）を選ぶ欄 先頭は「既定」

    選べるスタイルはファミリで決まるので、設定パネルが :meth:`set_family` で今の
    ファミリを渡す 値だけを持つ欄にすると、ファミリを替えたときに一覧が古いまま残る
    """

    def __init__(self, spec: FontStyleSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._family = ""
        self._value = spec.default_value()
        self._box = QComboBox(self)
        # フォントの欄と同じく、長いスタイル名（Sitka の「Subheading Semibold Italic」など）で
        # 設定パネルの最小の幅を広げない
        narrow_combo(self._box, 10)
        self._box.currentIndexChanged.connect(
            lambda index: self._emit(str(self._box.itemData(index)))
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._box, 1)
        self._fill()

    def set_family(self, family: str) -> None:
        """一覧を ``family`` のスタイルで作り直す 同じファミリなら触らない（開いた一覧が閉じる）"""
        if family == self._family:
            return
        self._family = family
        self._fill()

    def set_value(self, value: ParamValue | None) -> None:
        text = self._spec.coerce(value)
        if text == self._value:
            return
        self._value = text
        self._fill()

    def _fill(self) -> None:
        from PySide6.QtGui import QFontDatabase

        # 可変フォント（源ノ角ゴシック VF など）は同じ名前のスタイルを 2 度返す
        styles = list(dict.fromkeys(QFontDatabase.styles(self._family))) if self._family else []
        self._updating = True
        try:
            self._box.clear()
            self._box.addItem(FONT_STYLE_DEFAULT_LABEL, "")
            for style in styles:
                self._box.addItem(style, style)
            if self._value and self._value not in styles:
                # 消すと、開いただけで選んでいたスタイルが分からなくなる 描く所は既定で描く
                self._box.addItem(self._value + _MISSING_STYLE_SUFFIX, self._value)
            self._box.setCurrentIndex(max(0, self._box.findData(self._value)))
        finally:
            self._updating = False


class FileEditor(ParameterEditor):
    def __init__(self, spec: FileSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._line = QLineEdit(self)
        self._line.editingFinished.connect(lambda: self._emit(self._line.text()))

        browse = QPushButton("…", self)
        browse.setFixedWidth(32)
        browse.clicked.connect(self._choose)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._line, 1)
        layout.addWidget(browse)
        self.set_value(spec.default_value())

    def set_value(self, value: ParamValue | None) -> None:
        """同じ場所なら触らない 入れ直すとカーソルが末尾へ飛び、直している途中の欄が乱れる"""
        text = self._spec.coerce(value)
        if self._line.text() == text:
            return
        self._updating = True
        try:
            self._line.setText(text)
        finally:
            self._updating = False

    def _choose(self) -> None:
        if self._spec.directory:
            chosen = QFileDialog.getExistingDirectory(self, self._spec.label, self._line.text())
        else:
            chosen, _ = QFileDialog.getOpenFileName(
                self, self._spec.label, self._line.text(), self._spec.filter
            )
        if chosen:
            self.set_value(chosen)
            self._emit(chosen)


class ValueEditor(ParameterEditor):
    def __init__(self, spec: ValueSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._box = QSpinBox(self)
        self._box.setRange(_qt_int(spec.minimum), _qt_int(spec.maximum))
        self._box.setKeyboardTracking(False)
        self._box.valueChanged.connect(self._emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._box)
        layout.addStretch(1)
        self.set_value(spec.default_value())

    def set_value(self, value: ParamValue | None) -> None:
        self._updating = True
        try:
            self._box.setValue(_qt_int(self._spec.coerce(value)))
        finally:
            self._updating = False


class GridEditor(ParameterEditor):
    """格子の大きさを見せるだけの欄 触らせない

    点は 5x5 で 50 個あり、並べても読めないし、ずらして直したいときに
    欲しいのは数値欄ではなく画面上の掴み手 互換で読み込んだ格子が
    「消えた」と思われないよう、大きさだけは出しておく
    """

    def __init__(self, spec: GridSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spec = spec
        self._label = QLabel(self)
        self._label.setEnabled(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addStretch(1)
        self.set_value(spec.default_value())

    def set_value(self, value: ParamValue | None) -> None:
        columns, rows = self._spec.size(self._spec.coerce(value))
        if columns < 2 or rows < 2:
            self._label.setText("なし（四隅で変形）")
            return
        self._label.setText(f"{columns}x{rows} の格子")


def create_editor(spec: ParameterSpec, parent: QWidget | None = None) -> ParameterEditor:
    """仕様に合う入力欄を作る"""
    if isinstance(spec, TrackSpec):
        return TrackEditor(spec, parent)
    if isinstance(spec, CheckSpec):
        return CheckEditor(spec, parent)
    if isinstance(spec, ColorSpec):
        return ColorEditor(spec, parent)
    if isinstance(spec, SelectSpec):
        return SelectEditor(spec, parent)
    if isinstance(spec, TextSpec):
        return TextEditor(spec, parent)
    if isinstance(spec, FontSpec):
        return FontEditor(spec, parent)
    if isinstance(spec, FontStyleSpec):
        return FontStyleEditor(spec, parent)
    if isinstance(spec, FileSpec):
        return FileEditor(spec, parent)
    if isinstance(spec, GridSpec):
        return GridEditor(spec, parent)
    return ValueEditor(spec, parent)
