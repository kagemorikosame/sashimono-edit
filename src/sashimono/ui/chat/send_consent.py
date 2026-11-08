"""AI が外（Anthropic）へ送る前に、送ってよいかを尋ねる窓（#342 D4）

何がどこへ送られるかを、送る前に言葉で出す 答えは 3 つ

- 送る 次からは尋ねない（設定の「AI へ送る物」に覚える）
- アプリを閉じるまで送る（設定には書かない 次に起動したらまた尋ねる）
  呼び出しごとに尋ねる設定なら「今回は送る」（この 1 回だけ）
- 送らない（アプリを閉じるまで尋ねない 設定で変えればまた尋ねる）

既定のボタンは「送らない」にする Enter を押しただけで絵が外へ出ると、読まずに
押した人が取り返せない ✕ と Esc は送らず、答えも覚えない（次に送るときにまた尋ねる）

送る物の中身の下見（#425）は種類ごとの件数をいつも出し、行と絵は「詳細を見る」で開く
字幕の多い作品で全部を開いて出すと、窓が画面に収まらず、出すのも重くなる 中身は開いた
ときに初めて作る
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.ai.outbound import (
    SAMPLE_HEAD,
    Outbound,
    Sample,
    SendAnswer,
    SendRequest,
    clip_line,
)

__all__ = [
    "SEND_ALWAYS_TEXT",
    "SEND_DENY_TEXT",
    "SEND_ONCE_TEXT",
    "SEND_THIS_TIME_TEXT",
    "THUMBNAIL_WIDTH",
    "SampleView",
    "SendConsentDialog",
    "ask_send",
]

SEND_ALWAYS_TEXT = "送る（次からは尋ねない）"
SEND_ONCE_TEXT = "アプリを閉じるまで送る"
SEND_THIS_TIME_TEXT = "今回は送る"
SEND_DENY_TEXT = "送らない"
#: 詳細を開くボタンの文
SHOW_DETAILS_TEXT = "詳細を見る"
HIDE_DETAILS_TEXT = "詳細を閉じる"

#: 絵の下見の大きさ（画素） 何枚並んでも窓の幅に収まり、映っている物は見分けられる大きさ
THUMBNAIL_WIDTH = 240
THUMBNAIL_HEIGHT = 160
#: 窓が使う画面の高さの割合 これより高い中身は巻物で読む
SCREEN_SHARE = 0.85
#: 窓の幅の下限（画素） 狭い画面でも収まり、説明の文が 1 行に 30 文字ほど入る幅
MINIMUM_WIDTH = 560
#: 行の下見の高さ（行） これより長い分は欄の中で巻いて読む 窓が画面からはみ出さないように
LINES_VISIBLE = 8


def sample_heading(sample: Sample) -> str:
    """種類 1 つの見出し 件数と、詳細を開かずに見せる頭の数行"""
    heading = f"{sample.kind.label}: {sample.amount}"
    # 絵の説明はコマの位置だけなので、頭の行には出さず詳細で絵と並べる
    head = list(sample.lines[:SAMPLE_HEAD]) if sample.kind is not Outbound.FRAMES else []
    if head:
        more = len(sample.lines) - len(head) + sample.omitted
        heading += "\n" + "\n".join(f"  {clip_line(line)}" for line in head)
        if more > 0:
            heading += f"\n  …ほか {more} 件"
    if sample.note:
        heading += f"\n  {sample.note}"
    return heading


class SampleView(QWidget):
    """種類 1 つの下見の中身 行の一覧と絵（設定画面の「最後に送った物」でも使う）"""

    def __init__(self, sample: Sample, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(f"{sample.kind.label}: {sample.amount}", self)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        if sample.note:
            note = QLabel(sample.note, self)
            note.setWordWrap(True)
            layout.addWidget(note)
        if sample.images:
            layout.addWidget(_thumbnails(sample.images, self))
        lines = list(sample.lines)
        if sample.omitted:
            lines.append(
                f"…ほか {sample.omitted} 件（下見に出すのは先頭の {len(sample.lines)} 件）"
            )
        if lines:
            text = QPlainTextEdit(self)
            text.setReadOnly(True)
            text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            text.setPlainText("\n".join(lines))
            height = text.fontMetrics().lineSpacing() * min(len(lines), LINES_VISIBLE) + 12
            text.setFixedHeight(height)
            layout.addWidget(text)
        if not lines and not sample.images:
            layout.addWidget(QLabel("中身はありません", self))


def _thumbnails(images: Iterable[bytes], parent: QWidget) -> QWidget:
    """絵の縮小を横に並べる 並びきらない分は横に巻いて見る"""
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for png in images:
        label = QLabel(row)
        pixmap = QPixmap()
        if pixmap.loadFromData(png):
            label.setPixmap(
                pixmap.scaled(
                    THUMBNAIL_WIDTH,
                    THUMBNAIL_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            label.setText("（絵を読めませんでした）")
        layout.addWidget(label)
    layout.addStretch(1)
    scroll = QScrollArea(parent)
    scroll.setWidget(row)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setFixedHeight(THUMBNAIL_HEIGHT + 24)
    return scroll


class SendConsentDialog(QDialog):
    """送ってよいかの窓 答えは :attr:`answer`（✕・Esc なら ``None``）"""

    def __init__(self, request: SendRequest, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._request = request
        self.answer: SendAnswer | None = None
        self.setWindowTitle("AI へ送る物の確かめ")
        # 説明の文が細く折れて縦に伸びると、詳細を開いたときに画面に収まらない
        self.setMinimumWidth(MINIMUM_WIDTH)
        outer = QVBoxLayout(self)
        self._outer = outer
        # ボタンのほかは全部を 1 つの巻物に入れる 大きな字や長い見出し、開いた詳細で中身が
        # 画面より高くなっても、窓は画面に収め、ボタンはいつも見えるようにする
        body = QWidget(self)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        question = QLabel("AI アシスタントへ、次の物を送ってよいですか？", body)
        font = question.font()
        font.setBold(True)
        question.setFont(font)
        layout.addWidget(question)
        self.info = QLabel(request.text(), body)
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info)
        self.headings: list[QLabel] = []
        for sample in request.samples:
            heading = QLabel(sample_heading(sample), body)
            heading.setWordWrap(True)
            heading.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.headings.append(heading)
            layout.addWidget(heading)
        self.toggle = QToolButton(body)
        self.toggle.setText(SHOW_DETAILS_TEXT)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setVisible(bool(request.samples))
        self.toggle.toggled.connect(self._show_details)
        layout.addWidget(self.toggle)
        layout.addStretch(1)
        #: 開いたときに作る 開かない人の所で絵を読み込まない
        self.details: QWidget | None = None
        self._layout = layout
        self._body = body
        self._scroller = QScrollArea(self)
        self._scroller.setWidget(body)
        self._scroller.setWidgetResizable(True)
        self._scroller.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(self._scroller)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.always = QPushButton(SEND_ALWAYS_TEXT, self)
        self.once = QPushButton(SEND_THIS_TIME_TEXT if request.each_call else SEND_ONCE_TEXT, self)
        self.deny = QPushButton(SEND_DENY_TEXT, self)
        for button, answer in (
            (self.always, SendAnswer.ALWAYS),
            (self.once, SendAnswer.UNTIL_CLOSE),
            (self.deny, SendAnswer.DENY),
        ):
            button.setAutoDefault(False)
            button.clicked.connect(lambda _checked=False, chosen=answer: self._choose(chosen))
            buttons.addWidget(button)
        self.deny.setDefault(True)
        self.deny.setFocus()
        outer.addLayout(buttons)
        self._fit_to_screen()

    def _choose(self, answer: SendAnswer) -> None:
        self.answer = answer
        self.accept()

    def _show_details(self, shown: bool) -> None:
        self.toggle.setText(HIDE_DETAILS_TEXT if shown else SHOW_DETAILS_TEXT)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)
        if self.details is None and shown:
            self.details = self._build_details()
            self._layout.insertWidget(self._layout.indexOf(self.toggle) + 1, self.details)
        if self.details is not None:
            self.details.setVisible(shown)
        self._fit_to_screen()

    def _build_details(self) -> QWidget:
        details = QWidget(self._body)
        column = QVBoxLayout(details)
        column.setContentsMargins(0, 0, 0, 0)
        for sample in self._request.samples:
            column.addWidget(SampleView(sample, details))
        return details

    def _fit_to_screen(self) -> None:
        """中身の大きさで開く ただし画面に収める（残りは巻物で読む ボタンは巻物の外）"""
        wanted = self._body.sizeHint()
        bar = self._scroller.verticalScrollBar().sizeHint().width()
        margins = self._outer.contentsMargins()
        extra = margins.top() + margins.bottom()
        buttons = self.deny.sizeHint().height() + extra + 16
        width = max(MINIMUM_WIDTH, wanted.width() + bar + 24)
        height = wanted.height() + buttons + 8
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, int(available.width() * 0.9))
            height = min(height, int(available.height() * SCREEN_SHARE))
        self.resize(width, height)


def ask_send(parent: QWidget | None, request: SendRequest) -> SendAnswer | None:
    """窓を出して答えを返す 閉じるボタン（✕・Esc）で閉じたら ``None``（送らない 覚えない）"""
    dialog = SendConsentDialog(request, parent)
    dialog.exec()
    answer = dialog.answer
    dialog.deleteLater()
    return answer
