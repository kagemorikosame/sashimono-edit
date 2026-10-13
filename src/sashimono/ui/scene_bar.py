"""タイムラインの上に置く、シーンの切り替えバー

どのシーンを編集しているかは窓が持つ ここは選ぶ・作る・名前を変える・消す・置く、の
入口を並べて、押されたことを信号で知らせるだけ（名前を尋ねるダイアログも窓が出す）
右端には〔磁石〕と再生ヘッドの吸い付きの入り切りのボタンを並べる
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QSize, Signal, SignalInstance
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from sashimono.core.model import Project, SceneId
from sashimono.resources import path_to
from sashimono.ui.theme import (
    Colors,
    magnet_icons,
    playhead_snap_icons,
    theme_signals,
    themed_style,
)

__all__ = ["MAIN_SCENE_LABEL", "SceneBar"]

#: メインのタイムラインの表示名 シーンと同じ並びで選べるようにする
MAIN_SCENE_LABEL = "メイン"


def _snap_tip(state: str) -> str:
    return (
        f"磁石（吸着）: {state} 押すと切り替わる\n"
        "入のときは、クリップを動かす・伸び縮みさせる・置くときに、ほかのクリップの端・"
        "再生位置・キーフレーム・書き出し範囲の端へ吸い付く\n"
        "動かしている途中で Shift を押している間は一時的に吸い付かない\n"
        "目盛りで再生ヘッドを動かすときは、隣の再生ヘッドの吸い付きのボタンに従う"
    )


def _playhead_snap_tip(state: str) -> str:
    # 名前の後ろの括弧に、いつ効くかと外し方を書く ボタンが〔磁石〕の隣に並ぶので、
    # 名前だけでは磁石と何が違うのか分からない Shift の押す順も書く 押してから目盛りを
    # 押すと書き出し範囲の指定になり、吸い付かせたつもりで範囲が変わる
    return (
        "再生ヘッドの吸い付き（目盛りで動かすとき 掴んだ後に Shift で一時的に外す）: "
        f"{state} 押すと切り替わる\n"
        "入のときは、目盛りで再生ヘッドを動かすと、近くのクリップの端・キーフレーム・"
        "書き出し範囲の端へ吸い付く（吸い付く先は〔表示〕→〔設定…〕で選ぶ）\n"
        "Shift を押してから目盛りを押すと、書き出し範囲の指定になる\n"
        "〔磁石〕とは別に入り切りする"
    )


@dataclass(frozen=True, slots=True)
class _Toggle:
    """入り切りのボタン 1 つぶんの、印と補足の作り方と、押されたときに出す知らせ"""

    button: QPushButton
    icons: Callable[[], tuple[str, str]]
    tip: Callable[[str], str]
    signal: SignalInstance


class SceneBar(QWidget):
    #: 編集するシーンを選んだ 引数はシーンの ID、メインなら空文字列
    scene_selected = Signal(str)
    add_requested = Signal()
    rename_requested = Signal()
    remove_requested = Signal()
    place_requested = Signal()
    #: 磁石（タイムラインの吸着）を入れた・切った
    snap_toggled = Signal(bool)
    #: 再生ヘッドの吸い付き（目盛りで動かすとき）を入れた・切った
    playhead_snap_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(QLabel("シーン", self))

        self._combo = QComboBox(self)
        self._combo.setMinimumWidth(160)
        self._combo.setAccessibleName("編集するシーン")
        self._combo.currentIndexChanged.connect(self._on_index_changed)
        layout.addWidget(self._combo)

        self._add_button = self._button("新規", "空のシーンを作って開く", self.add_requested)
        self._rename_button = self._button(
            "名前", "開いているシーンの名前を変える", self.rename_requested
        )
        self._remove_button = self._button(
            "削除",
            "開いているシーンを消す（どこにも置かれていないときだけ）",
            self.remove_requested,
        )
        self._place_button = self._button(
            "置く",
            "ほかのシーンを、再生ヘッドの位置へ 1 本のクリップとして置く",
            self.place_requested,
        )
        for button in (
            self._add_button,
            self._rename_button,
            self._remove_button,
            self._place_button,
        ):
            layout.addWidget(button)
        layout.addStretch(1)
        # 磁石はタイムライン全体の操作の癖 シーンの操作から離して右端に置く
        # 文字だけのボタンは入と切が見分けにくかった（利用者の報告） 印を入と切で描き分け、
        # 入は押し込まれた地とアクセント色の縁、切は薄い灰色にする
        # 再生ヘッドの吸い付き（#313）は〔磁石〕の隣に同じ見た目で並べ、印の形だけを変える
        self._snap = self._toggle(
            "snap_button", "磁石（吸着）", magnet_icons, _snap_tip, self.snap_toggled
        )
        self._playhead_snap = self._toggle(
            "playhead_snap_button",
            "再生ヘッドの吸い付き",
            playhead_snap_icons,
            _playhead_snap_tip,
            self.playhead_snap_toggled,
        )
        layout.addWidget(self._snap.button)
        layout.addWidget(self._playhead_snap.button)
        # 印はテーマごとに別の絵 暗いテーマの印の白に近い色は、明るい地に溶ける
        # 部品の関数（lambda ではなく）でつなぐ 部品が消えたときに Qt がつなぎを外す
        theme_signals().changed.connect(self._redraw_toggle_icons)

    def _button(self, text: str, tip: str, signal: SignalInstance) -> QPushButton:
        button = QPushButton(text, self)
        button.setToolTip(tip)
        button.clicked.connect(lambda _checked=False: signal.emit())
        return button

    def _toggle(
        self,
        name: str,
        accessible: str,
        icons: Callable[[], tuple[str, str]],
        tip: Callable[[str], str],
        signal: SignalInstance,
    ) -> _Toggle:
        """入り切りのボタンを作る 既定は入（設定の既定と同じ）"""
        button = QPushButton(self)
        button.setObjectName(name)
        button.setAccessibleName(accessible)
        button.setCheckable(True)
        button.setIconSize(QSize(18, 18))
        themed_style(
            button,
            lambda: (
                f"QPushButton#{name} {{ padding: 2px 6px; }}"
                f"QPushButton#{name}:checked {{"
                f" background-color: {Colors.TAB_SELECTED.name()};"
                f" border: 1px solid {Colors.ACCENT.name()}; }}"
            ),
        )
        toggle = _Toggle(button, icons, tip, signal)
        button.setChecked(True)
        self._show_state(toggle)
        # ボタンは自分の子なので、lambda でつないでも一緒に消える
        button.toggled.connect(lambda _checked: self._on_toggled(toggle))
        return toggle

    def set_project(self, project: Project, active: SceneId | None) -> None:
        """一覧を作り直す 選び直しの信号は出さない（窓が決めた状態を映すだけ）"""
        self._updating = True
        try:
            self._combo.clear()
            self._combo.addItem(MAIN_SCENE_LABEL, "")
            for scene in project.scenes:
                self._combo.addItem(scene.name, str(scene.id))
            index = self._combo.findData(str(active) if active is not None else "")
            self._combo.setCurrentIndex(max(0, index))
        finally:
            self._updating = False
        # メインは名前を変えたり消したりできない 押せるのにエラーになると壊れて見える
        editing_scene = active is not None
        self._rename_button.setEnabled(editing_scene)
        self._remove_button.setEnabled(editing_scene)
        # 開いているシーン自身は置けない 置ける候補が無いのに押せると、押してから断られる
        self._place_button.setEnabled(any(scene.id != active for scene in project.scenes))

    @property
    def snap_button(self) -> QPushButton:
        return self._snap.button

    @property
    def playhead_snap_button(self) -> QPushButton:
        return self._playhead_snap.button

    def set_snap(self, enabled: bool) -> None:
        """磁石のボタンの状態を合わせる 知らせは出さない（窓が決めた状態を映すだけ）"""
        self._set_checked(self._snap, enabled)

    def set_playhead_snap(self, enabled: bool) -> None:
        """再生ヘッドの吸い付きのボタンの状態を合わせる 知らせは出さない（設定画面で変えたとき）"""
        self._set_checked(self._playhead_snap, enabled)

    def _set_checked(self, toggle: _Toggle, enabled: bool) -> None:
        self._updating = True
        try:
            toggle.button.setChecked(enabled)
        finally:
            self._updating = False

    def _on_toggled(self, toggle: _Toggle) -> None:
        self._show_state(toggle)
        if not self._updating:
            toggle.signal.emit(toggle.button.isChecked())

    def _redraw_toggle_icons(self) -> None:
        self._show_state(self._snap)
        self._show_state(self._playhead_snap)

    @staticmethod
    def _show_state(toggle: _Toggle) -> None:
        """印と補足を入・切に合わせる 補足に今の状態と一時的に切るキーを書く"""
        enabled = toggle.button.isChecked()
        on, off = toggle.icons()
        toggle.button.setIcon(QIcon(str(path_to(on if enabled else off))))
        toggle.button.setToolTip(toggle.tip("入" if enabled else "切"))

    def _on_index_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self.scene_selected.emit(str(self._combo.itemData(index) or ""))
