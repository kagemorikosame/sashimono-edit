"""押すと下に欄が開き、もう一度押すと閉じるボタン

開け閉めのボタンは、ふつうのボタンと見分けが付かないと「押したら何が起きたのか」
「もう一度押すと閉じるのか」が分からない（AI パネルの〔AI の部品…〕で利用者が迷った）
そこで次の 3 つで今の状態を見せる どれか 1 つに頼ると、見落とす人が出る

- 開いている間は押し込まれた見た目（地の色をアクセントの色に）
- 向きの印（閉じている間 ▸ 開いている間 ▾） 色の見分けが付きにくい人にも形で分かる
- ツールチップで開く・閉じるを言う

アプリの中で同じ作りのボタンを足すときは、これを使って見た目を揃える
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QWidget

from sashimono.ui.theme import Colors, themed_style

__all__ = ["CLOSED_MARK", "OPEN_MARK", "DisclosureButton"]

#: 閉じている間の印 右向きの三角は「この先に続きがある」と読まれる
CLOSED_MARK = "▸"
#: 開いている間の印 下向きの三角は「下に開いている」と読まれる
OPEN_MARK = "▾"


class DisclosureButton(QPushButton):
    """開け閉めのボタン 開いているかは ``isChecked()`` で、``toggled`` で知らせる"""

    def __init__(self, text: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = text
        self.setCheckable(True)
        self.setToolTip(tooltip)
        self.setAccessibleName(text)
        # 押下の見た目はここで決める アプリ全体の見た目（theme）は押している最中しか
        # 色を変えないので、そのままでは開いている間も押していない見た目に戻る
        # 色は描くたびにテーマから読み、明るいテーマでも暗いテーマでも地と分かれるようにする
        themed_style(
            self,
            lambda: (
                f"QPushButton {{ border: 1px solid {Colors.BORDER.name()};"
                " border-radius: 3px; padding: 4px 10px; }"
                f"QPushButton:checked {{ background-color: {Colors.ACCENT.name()};"
                f" color: {Colors.ACCENT_TEXT.name()}; border-color: {Colors.ACCENT.name()}; }}"
            ),
        )
        self.toggled.connect(self._show_mark)
        self._show_mark(False)
        # 見た目を今ここで当て、大きさを決めておく 並べる側（FlowLayout）が当てる前の
        # 大きさで行の高さを決めると、当てた後に 1 画素はみ出して下の欄に重なる
        self.ensurePolished()

    def set_open(self, opened: bool) -> None:
        """押されたことにせずに、開いた・閉じた見た目へ合わせる（``toggled`` を出さない）

        欄の側の都合（使えない間は開いておく など）で開け閉めしたときに使う 知らせを
        出すと、受け手がもう一度欄を開け閉めし直す
        """
        if self.isChecked() != opened:
            self.blockSignals(True)
            self.setChecked(opened)
            self.blockSignals(False)
        self._show_mark(opened)

    def _show_mark(self, opened: bool) -> None:
        self.setText(f"{OPEN_MARK if opened else CLOSED_MARK} {self._label}")
