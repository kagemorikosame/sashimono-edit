"""AI のパネルの計画の欄（#340）

AI が長い作業の前に書いた目標と手順（:mod:`sashimono.ai.plan`）と、手順ごとの進み具合を
見せる 人はここで〔区切りで止める〕（今の手順を終えた所で止める）と〔続ける〕（止めた所から
続きを頼む）を押す すぐに止めるのは今までどおりの〔中断〕

欄の中身は計画の写し（:class:`~sashimono.ai.plan.PlanView`）から毎回作り直す パネルの見張りの
時計（80 ミリ秒）から呼ばれるので、写しの版と動いているかが変わったときだけ書き換える
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.ai.plan import PlanView, StepState, Stopped
from sashimono.ui.flow_layout import FlowLayout
from sashimono.ui.theme import Colors, themed_style

__all__ = ["PAUSE_TEXT", "PAUSING_TEXT", "RESUME_TEXT", "PlanBox", "plan_progress"]

PAUSE_TEXT = "区切りで止める"
#: 押した後 区切りに来るまでは今の手順が続くので、止まったと見えないようにする
PAUSING_TEXT = "区切りで止めます…"
RESUME_TEXT = "続ける"

#: 止まった訳ごとに、次に何をすればよいかを添える 訳だけだと、続けてよいのかが分からない
_STOPPED_HINTS = {
    Stopped.PAUSED: "〔続ける〕で止めた所から続きを頼めます",
    Stopped.INTERRUPTED: (
        "手順の途中で止まったかもしれません 〔続ける〕で、AI が作品を確かめてから続けます"
    ),
    Stopped.LIMIT: (
        "手順の途中で止まったかもしれません 上限は 表示 → 設定… の「AI の上限」で変えられます"
        " 〔続ける〕で、AI が作品を確かめてから続けます"
    ),
    Stopped.ENDED: "〔続ける〕で残りの手順を頼めます",
}


def plan_progress(view: PlanView) -> str:
    """状態の行と欄の見出しに添える進み具合（「手順 3/5」など）"""
    running = view.running
    if running is not None:
        return f"手順 {running}/{len(view.steps)}"
    return f"{view.done}/{len(view.steps)} 済み"


class PlanBox(QScrollArea):
    """計画の欄 計画が無い間は隠す

    欄ごと巻物にする 手順を 30 個並べると 30 行を超え、そのまま足すと 1280x720 の画面で
    会話の欄と入力欄が潰れる 広い所では上限（:data:`MAX_LINES` 行）まで見え、狭い所では
    2 行まで縮んで巻ける（質問の箱と同じ）
    """

    pause_requested = Signal()
    resume_requested = Signal()
    close_requested = Signal()

    #: 欄の高さの上限（行） 決めないと、手順が多いと欄が会話の欄の高さを奪う
    MAX_LINES = 10

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 巻物の既定は縦にも広がりたがり、会話の欄から高さを奪う 中身の高さまでにする
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        lines = self.fontMetrics().lineSpacing()
        frame = 2 * self.frameWidth()
        self.setMinimumHeight(2 * lines + frame)
        self.setMaximumHeight(self.MAX_LINES * lines + frame)
        self.setVisible(False)
        #: 最後に描いた中身（版・動いているか・送れるか） 同じなら書き換えない
        #: 同じ文を書き直すだけでも、文字の欄は大きさを測り直して窓の配置をやり直す
        self._drawn: tuple[int, bool, bool] | None = None

        inner = QWidget(self)
        self._title = QLabel(inner)
        bold = self._title.font()
        bold.setBold(True)
        self._title.setFont(bold)
        # 目標の長さで AI のパネルの最小の幅を広げない 収まらない分は切れ、全部は補足で読む
        self._title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._title.setTextFormat(Qt.TextFormat.PlainText)
        self._close = QToolButton(inner)
        self._close.setText("×")
        self._close.setToolTip("計画を閉じる（AI に頼んだ作業は止めない）")
        self._close.setAccessibleName("計画を閉じる")
        self._close.setAutoRaise(True)
        self._close.clicked.connect(self.close_requested)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._title, 1)
        header.addWidget(self._close)

        # 手順の文は AI が書いた物 ``<b>`` などを装飾として読ませない
        self._steps = QLabel(inner)
        self._steps.setTextFormat(Qt.TextFormat.PlainText)
        self._steps.setWordWrap(True)

        self._stopped = QLabel(inner)
        self._stopped.setWordWrap(True)
        themed_style(self._stopped, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._pause = QPushButton(PAUSE_TEXT, inner)
        self._pause.setToolTip(
            "今の手順を終えた所で止める すぐに止めるなら〔中断〕 止めた後は〔続ける〕で続きを頼める"
        )
        self._pause.clicked.connect(self.pause_requested)
        self._resume = QPushButton(RESUME_TEXT, inner)
        self._resume.setToolTip("計画の止めた所から、残りの手順を AI に頼む")
        self._resume.clicked.connect(self.resume_requested)
        # 狭いパネルでもボタンが欠けないように折り返す
        buttons = QWidget(inner)
        buttons_layout = FlowLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addWidget(self._pause)
        buttons_layout.addWidget(self._resume)

        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self._steps)
        layout.addWidget(self._stopped)
        layout.addWidget(buttons)
        self.setWidget(inner)

    @property
    def pause_button(self) -> QPushButton:
        return self._pause

    @property
    def resume_button(self) -> QPushButton:
        return self._resume

    @property
    def close_button(self) -> QToolButton:
        return self._close

    def title(self) -> str:
        return self._title.text()

    def steps_text(self) -> str:
        return self._steps.text()

    def stopped_text(self) -> str:
        return self._stopped.text()

    def show_plan(self, view: PlanView | None, *, working: bool, ready: bool) -> bool:
        """計画を見せる 描き直したら真 ``working`` は AI が応えている最中か

        ``ready`` は指示を送れる（AI の部品が入っている）か 送れない間に〔続ける〕を
        押せると、押しても何も起きない
        """
        if view is None:
            changed = self._drawn is not None
            self._drawn = None
            self.setVisible(False)
            return changed
        key = (view.version, working, ready)
        if key == self._drawn:
            return False
        self._drawn = key
        progress = plan_progress(view)
        title = f"計画（{progress}）: {view.goal}"
        self._title.setText(title)
        self._title.setToolTip(title)
        self._steps.setText(_step_lines(view))
        if view.stopped is not None and not working:
            self._stopped.setText(f"{view.stopped.value} {_STOPPED_HINTS[view.stopped]}")
            self._stopped.setVisible(True)
        else:
            self._stopped.setText("")
            self._stopped.setVisible(False)
        unfinished = not view.finished
        # 止めるのは応えている間だけ 続けるのは応えていない間だけ（応えている間に押すと、
        # 今の指示の後ろへ「続けて」が並び、止めた所が分からなくなる）
        self._pause.setVisible(working)
        self._pause.setEnabled(working and unfinished and not view.pause_requested)
        self._pause.setText(PAUSING_TEXT if view.pause_requested else PAUSE_TEXT)
        self._resume.setVisible(not working)
        self._resume.setEnabled(not working and unfinished and ready)
        # 応えている間に閉じると、AI が書く先の計画が消え、手順の印を書けずに失敗が並ぶ
        self._close.setEnabled(not working)
        self.setVisible(True)
        return True


def _step_lines(view: PlanView) -> str:
    """手順の行 印で進み具合が分かり、取りかかり中の手順には言葉も添える"""
    lines = []
    for number, step in enumerate(view.steps, start=1):
        line = f"{step.state.mark} {number}. {step.title}"
        if step.state is not StepState.PENDING:
            line += f"（{step.state.label}）"
        lines.append(line)
        if step.note:
            lines.append(f"　　{step.note}")
    return "\n".join(lines)
