"""タイムラインの上に置く、シーンの切り替えバー

どのシーンを編集しているかは窓が持つ ここは選ぶ・作る・名前を変える・消す・置く、の
入口を並べて、押されたことを信号で知らせるだけ（名前を尋ねるダイアログも窓が出す）
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Signal, SignalInstance
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from sashimono.core.model import Project, SceneId
from sashimono.resources import path_to
from sashimono.ui.theme import Colors, magnet_icons, theme_signals, themed_style

__all__ = ["MAIN_SCENE_LABEL", "SceneBar"]

#: メインのタイムラインの表示名 シーンと同じ並びで選べるようにする
MAIN_SCENE_LABEL = "メイン"


class SceneBar(QWidget):
    #: 編集するシーンを選んだ 引数はシーンの ID、メインなら空文字列
    scene_selected = Signal(str)
    add_requested = Signal()
    rename_requested = Signal()
    remove_requested = Signal()
    place_requested = Signal()
    #: 磁石（タイムラインの吸着）を入れた・切った
    snap_toggled = Signal(bool)

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
        self._snap_button = QPushButton(self)
        self._snap_button.setObjectName("snap_button")
        self._snap_button.setAccessibleName("磁石（吸着）")
        self._snap_button.setCheckable(True)
        self._snap_button.setIconSize(QSize(18, 18))
        themed_style(
            self._snap_button,
            lambda: (
                "QPushButton#snap_button { padding: 2px 6px; }"
                "QPushButton#snap_button:checked {"
                f" background-color: {Colors.TAB_SELECTED.name()};"
                f" border: 1px solid {Colors.ACCENT.name()}; }}"
            ),
        )
        self._snap_button.setChecked(True)
        self._show_snap_state(True)
        self._snap_button.toggled.connect(self._on_snap_toggled)
        # 印はテーマごとに別の絵 暗いテーマの印の白に近い極は、明るい地に溶ける
        # 部品の関数（lambda ではなく）でつなぐ 部品が消えたときに Qt がつなぎを外す
        theme_signals().changed.connect(self._redraw_snap_icon)
        layout.addWidget(self._snap_button)

    def _button(self, text: str, tip: str, signal: SignalInstance) -> QPushButton:
        button = QPushButton(text, self)
        button.setToolTip(tip)
        button.clicked.connect(lambda _checked=False: signal.emit())
        return button

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
        return self._snap_button

    def set_snap(self, enabled: bool) -> None:
        """磁石のボタンの状態を合わせる 知らせは出さない（窓が決めた状態を映すだけ）"""
        self._updating = True
        try:
            self._snap_button.setChecked(enabled)
        finally:
            self._updating = False

    def _on_snap_toggled(self, checked: bool) -> None:
        self._show_snap_state(checked)
        if not self._updating:
            self.snap_toggled.emit(checked)

    def _redraw_snap_icon(self) -> None:
        self._show_snap_state(self._snap_button.isChecked())

    def _show_snap_state(self, enabled: bool) -> None:
        """印と補足を入・切に合わせる 補足に今の状態と一時的に切るキーを書く"""
        on, off = magnet_icons()
        self._snap_button.setIcon(QIcon(str(path_to(on if enabled else off))))
        state = "入" if enabled else "切"
        self._snap_button.setToolTip(
            f"磁石（吸着）: {state} 押すと切り替わる\n"
            "入のときは、クリップを動かす・伸び縮みさせる・置くときに、ほかのクリップの端・"
            "再生位置・キーフレーム・書き出し範囲の端へ吸い付く\n"
            "動かしている途中で Shift を押している間は一時的に吸い付かない"
        )

    def _on_index_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self.scene_selected.emit(str(self._combo.itemData(index) or ""))
