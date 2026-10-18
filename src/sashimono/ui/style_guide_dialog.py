"""編集の方針（スタイルガイド #343）を書く窓と、設定の窓に置く 1 行

方針は作品に付く（:class:`~sashimono.core.model.StyleGuide`） 書くのは ファイル →
プロジェクト設定… の「編集の方針」 表示 → 設定… の「新しい作品の編集の方針」は、新しく作る
作品の初めの方針（置き方の方式の初めの値と同じ扱い 開いている作品の方針は変えない）
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFontComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.model import StyleGuide
from sashimono.effects import registry
from sashimono.ui.theme import Colors, themed_style

__all__ = ["StyleGuideDialog", "StyleGuideRow", "effect_choices", "guide_summary"]

#: 文の欄の高さ（行） 3 つの文の欄と 2 つの一覧を縦に並べても、1280x720 の画面に収まる高さ
_TEXT_LINES = 3


def effect_choices() -> list[tuple[str, str]]:
    """選べる効果（種類, 表示名） 分類 → 名前の順 一覧はエフェクトの追加の一覧と同じ定義から作る"""
    return [(d.kind, f"{d.category} / {d.label}") for d in registry.all()]


def guide_summary(guide: StyleGuide) -> str:
    """設定の窓の 1 行に出す要約 中身を開かずに、何を決めてあるかが分かるように"""
    if guide.empty:
        return "決めていない（AI は好きに選ぶ）"
    parts: list[str] = []
    if guide.fonts:
        parts.append(f"書体 {len(guide.fonts)}")
    if guide.colors:
        parts.append(f"色 {len(guide.colors)}")
    if guide.effects:
        parts.append(f"効果 {len(guide.effects)}")
    for label, text in (
        ("字幕の見た目", guide.subtitles),
        ("テンポ", guide.tempo),
        ("そのほか", guide.notes),
    ):
        if text:
            parts.append(label)
    return "・".join(parts)


def _lines(edit: QPlainTextEdit) -> tuple[str, ...]:
    return tuple(edit.toPlainText().splitlines())


def _text_box(text: str, placeholder: str, parent: QWidget) -> QPlainTextEdit:
    box = QPlainTextEdit(text, parent)
    box.setPlaceholderText(placeholder)
    box.setTabChangesFocus(True)
    # 行の高さから決める 決め打ちの画素にすると、書体の大きい環境で 1 行しか見えない
    rows = box.fontMetrics().lineSpacing() * _TEXT_LINES
    box.setFixedHeight(rows + 2 * box.frameWidth() + 8)
    return box


class StyleGuideDialog(QDialog):
    """方針を書く 空の欄は「決めていない」 ``effects`` は選べる効果（試験で絞る）"""

    def __init__(
        self,
        guide: StyleGuide,
        parent: QWidget | None = None,
        *,
        title: str = "編集の方針",
        effects: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        intro = QLabel(
            "AI アシスタントが従う、この作品の決まり 空の欄は決めていない（AI が好きに選ぶ）\n"
            "AI は読むだけで変えない 指示と食い違うときは指示が勝ち、AI が食い違いを伝える",
            self,
        )
        intro.setWordWrap(True)

        self._fonts = _text_box("\n".join(guide.fonts), "1 行に 1 つ 例: 源ノ角ゴシック", self)
        self._font_pick = QFontComboBox(self)
        add_font = QPushButton("書体を足す", self)
        add_font.clicked.connect(self._add_font)
        font_row = QHBoxLayout()
        font_row.addWidget(self._font_pick, 1)
        font_row.addWidget(add_font)

        self._colors = _text_box("\n".join(guide.colors), "1 行に 1 つ #RRGGBB 例: #ffffff", self)
        add_color = QPushButton("色を選んで足す…", self)
        add_color.clicked.connect(self._ask_color)

        self._subtitles = _text_box(
            guide.subtitles, "例: 下の中央に白の字・黒の縁 保存したプリセット「〜」を使う", self
        )
        self._tempo = _text_box(
            guide.tempo, "例: 言葉の間は 0.3 秒まで詰める 1 カットは 5 秒まで", self
        )

        self._effects = QListWidget(self)
        self._effects.setToolTip(
            "使ってよい効果に印を付ける 1 つも付けなければ決めていない（どれでも使ってよい）"
        )
        chosen = set(guide.effects)
        known: set[str] = set()
        for kind, label in effects if effects is not None else effect_choices():
            known.add(kind)
            self._add_effect_item(kind, label, kind in chosen)
        for kind in guide.effects:
            if kind not in known:
                # 今の環境に無い種類（読み込んでいない AviUtl のスクリプトなど）も消さずに残す
                # 一覧に出さないと、OK を押しただけで本人の方針から黙って消える
                self._add_effect_item(kind, f"（今は無い効果）{kind}", True)
        rows = self._effects.fontMetrics().lineSpacing() * 6
        self._effects.setFixedHeight(rows + 2 * self._effects.frameWidth() + 8)

        self._notes = _text_box(guide.notes, "そのほかの決まり 例: 絵文字は使わない", self)

        self._warning = QLabel(self)
        self._warning.setWordWrap(True)
        themed_style(self._warning, lambda: f"color: {Colors.WARNING.name()};")

        form = QFormLayout()
        form.addRow("使う書体", self._fonts)
        form.addRow("", font_row)
        form.addRow("使う色", self._colors)
        form.addRow("", add_color)
        form.addRow("字幕の見た目", self._subtitles)
        form.addRow("テンポ", self._tempo)
        form.addRow("使ってよい効果", self._effects)
        form.addRow("そのほか", self._notes)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self._warning)
        layout.addWidget(self._buttons)

        for box in (self._fonts, self._colors, self._subtitles, self._tempo, self._notes):
            box.textChanged.connect(self._sync)
        self._sync()

    @property
    def effects_list(self) -> QListWidget:
        return self._effects

    def guide(self) -> StyleGuide:
        """書いた方針 前後の空白と空の行は落とし、色は小文字にする"""
        chosen = tuple(
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in (self._effects.item(i) for i in range(self._effects.count()))
            if item is not None and item.checkState() == Qt.CheckState.Checked
        )
        return StyleGuide(
            fonts=_lines(self._fonts),
            colors=_lines(self._colors),
            subtitles=self._subtitles.toPlainText(),
            tempo=self._tempo.toPlainText(),
            effects=chosen,
            notes=self._notes.toPlainText(),
        ).cleaned()

    def set_fields(
        self,
        *,
        fonts: str | None = None,
        colors: str | None = None,
        subtitles: str | None = None,
        tempo: str | None = None,
        notes: str | None = None,
    ) -> None:
        """欄へ書く（試験と、窓の外から下書きを入れる所）"""
        for box, text in (
            (self._fonts, fonts),
            (self._colors, colors),
            (self._subtitles, subtitles),
            (self._tempo, tempo),
            (self._notes, notes),
        ):
            if text is not None:
                box.setPlainText(text)

    def add_color(self, color: QColor) -> None:
        """色を 1 行足す 色の窓で選んだ色を #rrggbb で書く（書き方を 1 つにそろえる）"""
        if not color.isValid():
            return
        lines = [line for line in _lines(self._colors) if line.strip()]
        lines.append(color.name())
        self._colors.setPlainText("\n".join(lines))

    def _add_effect_item(self, kind: str, label: str, checked: bool) -> None:
        item = QListWidgetItem(f"{label}（{kind}）", self._effects)
        item.setData(Qt.ItemDataRole.UserRole, kind)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _add_font(self) -> None:
        name = self._font_pick.currentFont().family()
        lines = [line for line in _lines(self._fonts) if line.strip()]
        if name and name not in lines:
            lines.append(name)
            self._fonts.setPlainText("\n".join(lines))

    def _ask_color(self) -> None:
        self.add_color(QColorDialog.getColor(parent=self, title="使う色を足す"))

    def _sync(self) -> None:
        """持てない方針（色の書き方・長さ）なら理由を出して OK を押せなくする

        閉じてから断ると、書いた文が窓ごと消える ここで止めて直してもらう
        """
        problems = self.guide().problems()
        self._warning.setText("\n".join(problems))
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(not problems)


class StyleGuideRow(QWidget):
    """設定の窓の 1 行 要約と〔編集…〕 押すと :class:`StyleGuideDialog` を開く"""

    def __init__(self, guide: StyleGuide, parent: QWidget | None = None, *, title: str) -> None:
        super().__init__(parent)
        self._guide = guide
        self._title = title
        self._summary = QLabel(self)
        self._summary.setWordWrap(True)
        self._edit = QPushButton("編集…", self)
        self._edit.clicked.connect(self.edit)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._summary, 1)
        row.addWidget(self._edit)
        self._show()

    @property
    def guide(self) -> StyleGuide:
        return self._guide

    @property
    def summary(self) -> str:
        return self._summary.text()

    def edit(self) -> bool:
        """方針の窓を開く OK なら方針を書き換えて真"""
        dialog = StyleGuideDialog(self._guide, self, title=self._title)
        answer = dialog.exec()
        # 開くたびに作る窓 閉じたら捨てる 消えるのは呼んだイベントループへ戻ったとき
        dialog.deleteLater()
        if answer != QDialog.DialogCode.Accepted:
            return False
        self._guide = dialog.guide()
        self._show()
        return True

    def _show(self) -> None:
        self._summary.setText(guide_summary(self._guide))
