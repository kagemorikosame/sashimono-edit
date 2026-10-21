"""AI チャットパネル

編集ソフトの中で Claude と話し、実際の編集をさせる 会話の見た目より、
**何をされているかが分かること**を優先している ツールの呼び出しは 1 行ずつ
出し、変更系は許可を求めてから実行する

1 つの指示で行われた編集は、まとめて 1 回の Undo で戻せる AI は 1 つの指示で
何十回も操作するので、これが無いと取り消しに同じ回数が要る
"""

from __future__ import annotations

import html
import re
import time
from collections import deque
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFocusEvent,
    QInputMethodEvent,
    QKeyEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.ai import AI_PACK, Approval, EditorBridge, EditorHost, Question
from sashimono.ai.bridge import BudgetRequest
from sashimono.ai.budget import (
    DailyUsageStore,
    Limits,
    Usage,
    UsageMeter,
    format_minutes,
    format_tokens,
)
from sashimono.ai.environment import claude_cli, credentials_found, open_login_window
from sashimono.ai.models import EFFORTS, MODELS, effort_for, find_model
from sashimono.ai.operations import ASK_USER
from sashimono.ai.outbound import SendAnswer, SendChoice, SendRequest
from sashimono.ai.plan import RESUME_PROMPT, Stopped
from sashimono.ai.session import (
    REINSTALL_HINT,
    UPDATE_PARTS_LABEL,
    AgentEvent,
    AgentSession,
    EventKind,
    OutdatedClaudeCode,
    outdated_claude_code,
    system_prompt,
)
from sashimono.core.commands import Author
from sashimono.ui.chat.parts_updater import PartsUpdater
from sashimono.ui.chat.plan_box import PlanBox, plan_progress
from sashimono.ui.chat.send_consent import ask_send
from sashimono.ui.disclosure import DisclosureButton
from sashimono.ui.flow_layout import FlowLayout
from sashimono.ui.setup import SetupSection
from sashimono.ui.theme import Colors, theme_signals, themed_style
from sashimono.ui.workspace import Preferences, config_root

__all__ = ["ChatPanel"]

#: 1 日に使った量を残すファイル（本人の設定の置き場 #344） 外へは送らない
USAGE_FILE = "ai_usage.json"

#: 上限に当たったときの箱の頭と、ボタンの文言
BUDGET_HEAD = "AI が使う量の上限に達しました 続けますか？"
BUDGET_STOP_TEXT = "ここで止める"

#: 太字と等幅だけを拾う Claude の返事は素の Markdown で来るので、
#: そのまま出すと ** が本文に混ざる 見出しや表まで組む必要は無い
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_CODE = re.compile(r"`([^`]+)`")

#: 確かめの許すボタンの文言
ALLOW_TEXT = "許可"
#: 取り消せない操作（#350）の確かめの頭と、許すボタンの文言 ふだんの確かめと見分けが付き、
#: 押すと戻せないことが押す前に読めるようにする
IRREVERSIBLE_HEAD = "取り消せない操作です 「元に戻す」では戻りません 実行しますか？"
IRREVERSIBLE_ALLOW_TEXT = "実行する"

#: ワーカーからの知らせを拾う間隔（ミリ秒）
#: ここを長くすると、AI の操作が画面へ反映されるまでの間が空く
POLL_MS = 80

#: 使った量を 1 日の置き場へ足す間隔（秒 #344） 道具の前と指示の終わりにも足すので、
#: これは道具を使わない長い応答の間と、落ちたときのための物
USAGE_FLUSH_SECONDS = 10.0

#: 実行中にドックのタブの名前の頭へ付ける印 パネルが別のタブの裏にあっても動いていると分かる
RUNNING_MARK = "● "

#: 会話の欄の区切りの行（1 回の応答の終わり）を表す、言った人の代わりの印
#: 本人や Claude の発言と取り違えないよう、名前に使わない制御文字で始める
_DIVIDER = "\0divider"


class _BusyMark(QWidget):
    """実行中に回る印

    色は描くたびにテーマから読む 作ったときの色を持つと、テーマを替えたとき
    地の色に溶けて見えなくなる 回すのは見張りの時計（``POLL_MS``）に任せ、
    この部品のための時計は持たない（閉じるときに止め忘れる時計を増やさない）
    """

    #: 1 回の進みで回す角度（度） 80 ミリ秒ごとに 30 度でおよそ 1 秒に 1 周
    STEP = 30

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        side = self.fontMetrics().height()
        self.setFixedSize(side, side)
        self._angle = 0
        # 読み上げでは色も動きも伝わらないので、名前で状態を言う
        self.setAccessibleName("実行中")

    def advance(self) -> None:
        self._angle = (self._angle + self.STEP) % 360
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt の命名規約
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 線は太めにする 細いと縁がぼけて、明るいテーマの地では色が薄く見える
        width = max(3, self.width() // 5)
        half = (width + 1) // 2
        ring = self.rect().adjusted(half, half, -half, -half)
        track = QColor(Colors.TEXT_MUTED)
        track.setAlpha(80)
        painter.setPen(QPen(track, width))
        painter.drawEllipse(ring)
        arc = QPen(Colors.ACCENT, width)
        arc.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(arc)
        # Qt の角度は 1/16 度で、正が反時計回り 時計回りに進めたいので負にする
        painter.drawArc(ring, -self._angle * 16, 120 * 16)
        painter.end()


class _Input(QPlainTextEdit):
    """指示の入力欄 既定は Enter で送り、Shift+Enter で改行する

    Ctrl+Enter はどちらの設定でも送る 前の版で覚えた人の手を裏切らないため
    """

    submitted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: Enter だけで送るか 切ると Enter は改行になる
        self.enter_sends = True
        #: 日本語の変換の途中か 変換を確定する Enter で送らないために持つ
        self._composing = False
        #: 案内の文（変換中は隠すので、Qt の欄とは別に本来の文面を持つ #297）
        self._placeholder = ""

    @property
    def placeholder(self) -> str:
        """本来の案内の文 変換中で欄から外している間も、この文面を返す"""
        return self._placeholder

    def set_placeholder(self, text: str) -> None:
        """案内の文を変える 欄の案内の文はここからだけ変える

        ``setPlaceholderText`` を直に呼ぶと、変換中に Enter の設定を変えたとき
        隠していた案内の文が変換中の文字の上へ戻り、また重なる
        """
        self._placeholder = text
        self._show_placeholder(not self._composing)

    def _show_placeholder(self, shown: bool) -> None:
        wanted = self._placeholder if shown else ""
        if self.placeholderText() != wanted:
            self.setPlaceholderText(wanted)

    def inputMethodEvent(self, event: QInputMethodEvent) -> None:  # noqa: N802 - Qt の命名規約
        # 変換中の文字（まだ確定していない読み）があるかどうかで判断する
        # 確定した瞬間は preedit が空になって届くので、そこで変換の終わりと見る
        self._composing = bool(event.preeditString())
        # Qt は文書が空なら案内の文を描く 変換中の文字は文書の外の層で描かれるので、
        # 隠さないと読みと変換の候補の窓に案内の文が重なって読めない（#297）
        self._show_placeholder(not self._composing)
        super().inputMethodEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802 - Qt の命名規約
        # 変換中に別の所を押すと、IME によっては preedit を空にする知らせが来ないまま
        # 欄を離れる 畳まないと、案内の文が消えたままになり、戻ってきた次の Enter も
        # 変換の確定と取られて送れない
        if self._composing:
            self._composing = False
            self._show_placeholder(True)
        super().focusOutEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt の命名規約
        is_enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if not is_enter or self._composing:
            # 変換中の Enter は変換の確定 ここで送ると、確定したつもりの文が
            # 書きかけのまま飛んでいく（IME によっては keyPress も届く）
            super().keyPressEvent(event)
            return
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.submitted.emit()
            return
        if self.enter_sends and modifiers == Qt.KeyboardModifier.NoModifier:
            self.submitted.emit()
            return
        if self.enter_sends and modifiers & Qt.KeyboardModifier.ShiftModifier:
            # Shift+Enter をそのまま渡すと、QPlainTextEdit は段落ではなく行の
            # 区切り（U+2028）を入れる 見た目は同じでも、Enter で書いた改行と
            # 違う文字になるので、ふつうの改行に揃える
            self.insertPlainText("\n")
            return
        super().keyPressEvent(event)


class ChatPanel(QWidget):
    """Claude と話しながら編集する"""

    #: ステータスバーへ出す文言
    status_message = Signal(str)
    #: 欄の上でモデルか考える深さを選び直した 引数はモデル ID とエフォート
    #: 本人の好みとして覚えるのは受け取る側（設定の置き場を 1 か所にするため）
    choices_changed = Signal(str, str)
    #: 送ってよいかの窓で「送る（次からは尋ねない）」と答えた 引数は送る物（:class:`Outbound` の値）
    #: 設定に覚えるのは受け取る側（設定の置き場を 1 か所にするため）
    send_allowed = Signal(str)

    def __init__(
        self,
        host: EditorHost,
        parent: QWidget | None = None,
        *,
        updater: PartsUpdater | None = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        # 1 日の量はアプリを閉じても残す（#344） 置き場は設定と同じ（試験は一時フォルダへ向く）
        self._bridge = EditorBridge(
            host, usage=UsageMeter(DailyUsageStore(config_root() / USAGE_FILE))
        )
        self._session: AgentSession | None = None
        #: 画面に出している「上限に達しました 続けますか」（#344）
        self._budget: BudgetRequest | None = None
        #: 止めると決まった後も道具を呼び続けたので、会話を止めた（同じ指示で 2 度止めない）
        self._halted = False
        #: 使った量を最後に 1 日の置き場へ足した時刻（:data:`USAGE_FLUSH_SECONDS`）
        self._flushed_at = 0.0
        #: 会話を作ったときの置き方の方式 指示の文をこの方式で書いたので、変わったら作り直す
        self._session_mode = ""
        self._approval: Approval | None = None
        #: 画面に出している AI からの質問（#296） 出ている間は、入力欄から送った文を答えにする
        self._question: Question | None = None
        #: あいまいなときに AI に質問させるか（設定） 会話を作るときに指示の文と道具へ渡す
        self._ask_questions = True
        #: 長い作業は計画を先に示させるか（設定 #340） 質問と同じく会話を作るときに渡す
        self._plan_first = True
        #: 開いている作品の計画（#340） ブリッジと同じ物を見る
        self._plan = self._bridge.plan
        self._checkpoint_open = False
        #: 外へ送ってよいかを尋ねる窓（#342） 試験はここを差し替えて答える
        self.ask_send: Callable[[QWidget | None, SendRequest], SendAnswer | None] = ask_send
        #: 送ってよいかの窓を出している間は真 窓の間も見張りの時計は回るので、2 つ目を出さない
        self._asking_send = False
        #: 応答が 1 往復終わった回数 無人での確認に使う
        self._turns_done = 0
        #: 送ってまだ応答が終わっていない指示（送った順） 会話を繋ぎ直してよいかの
        #: 判断と、続けて送った指示の取り消しの段を、その指示の名前で開くのに使う
        self._queued: deque[str] = deque()
        #: 出した会話（言った人、文） 補足は言った人が ``None`` テーマを切り替えたときに
        #: 今の色で書き直すために持つ
        self._log: list[tuple[str | None, str]] = []
        #: 1 度でも画面に出たか 出る前の幅は置き方が決める前の仮の値で、その幅で折り返した
        #: 高さを最小に入れると、窓が出るときにその高さまで縦に伸びる（#316 :meth:`_fit_wrapped`）
        self._shown = False

        # --- いま応えている指示（状態の行と区切りの行に使う） ---
        #: 経過秒を測る時計 試験で差し替える（本物の時計だと秒の境目で揺れる）
        self._clock: Callable[[], float] = time.monotonic
        #: いまの指示に取りかかった時刻 応えている指示が無ければ ``None``
        self._turn_started: float | None = None
        #: 指示を部品の入れ替えの間持っていて、使った量（#344）をまだ数え始めていないか
        #: 数え始めるのは会話へ渡す所（:meth:`_pass_held`）
        self._usage_waiting = False
        #: いまの指示で AI が呼んだ道具の数と、そのうち失敗した数
        self._turn_tools = 0
        self._turn_failures = 0
        #: いまの指示でエラーが出たか 区切りの行を「失敗」にする
        self._turn_error = False
        #: 中断を頼んだか 区切りの行を「中断」にし、状態の行で中断の途中だと見せる
        self._stopping = False
        #: いま使っている道具の名前
        self._tool = ""
        #: 会話が Claude Code に繋がったか 繋がるまでは「接続しています…」と出す
        self._connected = False
        #: 実行中に回る印を出すか（設定） 切っても文字の状態は出す
        self._animate = True
        #: 指示がすべて終わったらタスクバーで知らせるか（設定）
        self._alert_when_done = False

        # --- 部品が古くて断られたとき ---
        #: いまの指示で「Claude Code が古くて、このモデルを使えない」と断られた
        self._outdated: OutdatedClaudeCode | None = None
        #: 部品を新しくしてから、いまの指示を送り直すところ
        self._retrying = False
        #: いまの指示は 1 度送り直した 送り直しても断られたら、もう送り直さず案内を出す
        self._retried = False
        #: 部品を裏で新しくする物 自動の更新（設定で切れる）と、断られたときの立て直しに使う
        self._updater = updater if updater is not None else PartsUpdater(AI_PACK, self)
        self._updater.prepare = self._release_session
        self._updater.finished.connect(self._on_parts_updated)

        self._build()
        # 導入の欄から入れるときも、会話を先に畳ませて畳み終わるのを裏で待つ
        self._setup.prepare = self._release_session
        theme_signals().changed.connect(self._redraw_log)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._refresh_availability()
        self._refresh_status()
        self._updater.schedule()

    # --- 組み立て ---

    def _build(self) -> None:
        self._setup = SetupSection(AI_PACK, self)
        self._setup.finished.connect(self._on_setup_finished)

        # 今どのモデルで話しているかが、いつも見えるように欄の一番上へ置く
        self._model = QComboBox(self)
        for model in MODELS:
            self._model.addItem(model.label, model.id)
        self._model.setToolTip("「既定」は Claude Code がアカウントに合わせて選ぶモデル")
        self._effort = QComboBox(self)
        for effort in EFFORTS:
            self._effort.addItem(effort.label, effort.value)
        self._effort.setToolTip(
            "考える深さ 高くするほどよく考えてから答える代わりに、遅く、使う量も増える"
        )
        self._model.currentIndexChanged.connect(self._on_choice_changed)
        self._effort.currentIndexChanged.connect(self._on_choice_changed)

        # 名前と欄の組ごとに折り返す 1 列に並べると、モデルの名前の欄と深さの欄の幅の和が
        # AI のパネルの最小の幅になり、設定パネルと重ねた右の列が 1366 の画面で広がりすぎた
        # 組を崩さないのは、欄だけが次の行へ落ちると、どの名前の欄か読めなくなるため
        # 並びは部品に入れて、AI の部品が入っていない間は隠す 入れる前はモデルを選んでも
        # 送れず、導入の欄と並びの両方が開いていると、窓の最小の高さが 1280x720 の画面で
        # 大きい書体の余白を食う（並びは狭い所で折り返し、その行の分まで高さを取る）
        self._choices = QWidget(self)
        choices = FlowLayout(self._choices)
        choices.setContentsMargins(0, 0, 0, 0)
        for text, box in (("モデル", self._model), ("考える深さ", self._effort)):
            pair = QWidget(self)
            pair_layout = QHBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.addWidget(QLabel(text, pair))
            pair_layout.addWidget(box)
            choices.addWidget(pair)
        # 使える間は導入の欄を隠しているので、手で更新を確かめる入口をここに置く
        # 組と同じく折り返すので、狭い窓でもパネルの最小の幅は広がらない
        # 開け閉めのボタンだと分かる見た目にする（押し込まれた見た目・▸ ▾ の印）
        self._parts_button = DisclosureButton(
            "AI の部品", "AI の部品の版と更新を開く / 閉じる", self
        )
        self._parts_button.toggled.connect(self._on_parts_toggled)
        choices.addWidget(self._parts_button)

        # 導入の欄を「AI の部品」のまとまりとして枠で囲み、見出しと閉じる印を付ける
        # 枠が無いと、開いた欄がどのボタンの物か・どこまでが欄かが分からない
        self._parts_box = QFrame(self)
        self._parts_box.setObjectName("ai_parts_box")
        themed_style(
            self._parts_box,
            lambda: (
                f"QFrame#ai_parts_box {{ border: 1px solid {Colors.BORDER.name()};"
                " border-radius: 3px; }"
            ),
        )
        parts_title = QLabel("AI の部品", self._parts_box)
        title_font = parts_title.font()
        title_font.setBold(True)
        parts_title.setFont(title_font)
        self._parts_close = QToolButton(self._parts_box)
        self._parts_close.setText("×")
        self._parts_close.setToolTip("閉じる")
        self._parts_close.setAccessibleName("AI の部品を閉じる")
        self._parts_close.setAutoRaise(True)
        self._parts_close.clicked.connect(lambda: self._show_parts(False))
        parts_header = QHBoxLayout()
        parts_header.setContentsMargins(0, 0, 0, 0)
        parts_header.addWidget(parts_title)
        parts_header.addStretch(1)
        parts_header.addWidget(self._parts_close)
        # 導入の欄は巻物に入れる 未導入の間は欄が開いたままで、説明の行とボタンの全部の
        # 高さが窓の最小の高さに足され、1280x720 の画面で大きい書体の余白を食った（CI で
        # 窓の最小が 556 画素） 広い所では全部が見え、狭い所ではボタン 1 行分まで縮んで巻ける
        self._parts_scroll = QScrollArea(self._parts_box)
        self._parts_scroll.setWidgetResizable(True)
        self._parts_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._parts_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._parts_scroll.setWidget(self._setup)
        self._parts_scroll.setMinimumHeight(self._parts_close.sizeHint().height())
        parts_layout = QVBoxLayout(self._parts_box)
        parts_layout.setContentsMargins(6, 2, 6, 4)
        parts_layout.setSpacing(2)
        parts_layout.addLayout(parts_header)
        parts_layout.addWidget(self._parts_scroll)
        # 開け閉めは覚えない 毎回閉じた状態で始める（使えない間だけは開いたまま）
        self._parts_box.setVisible(False)

        # ログインは Claude Code 自身の画面で済ませてもらう 鍵やパスワードを
        # このソフトの入力欄で受け取らない
        self._login_box = QFrame(self)
        self._login_text = QLabel(
            "Claude へのログインがまだのようです 「ログイン…」で Claude Code を開き、"
            "案内に従ってログインしてください 済んだらそのまま指示を送れます"
            "（API キーを使う場合は環境変数 ANTHROPIC_API_KEY を設定し、ソフトを再起動します）",
            self._login_box,
        )
        self._login_text.setWordWrap(True)
        themed_style(self._login_text, lambda: f"color: {Colors.TEXT_MUTED.name()};")
        self._login_button = QPushButton("ログイン…", self._login_box)
        self._login_button.clicked.connect(self.open_login)
        login_layout = QHBoxLayout(self._login_box)
        login_layout.setContentsMargins(0, 0, 0, 0)
        login_layout.addWidget(self._login_text, 1)
        login_layout.addWidget(self._login_button)

        self._view = QTextBrowser(self)
        self._view.setOpenExternalLinks(False)
        themed_style(
            self._view,
            lambda: (
                f"background-color: {Colors.PANEL_ALT.name()};"
                f"border: 1px solid {Colors.BORDER.name()};"
            ),
        )

        self._approval_box = QFrame(self)
        self._approval_box.setFrameShape(QFrame.Shape.StyledPanel)
        self._approval_box.setVisible(False)
        self._approval_text = QLabel(self._approval_box)
        self._approval_text.setWordWrap(True)

        self._approval_allow = QPushButton(ALLOW_TEXT, self._approval_box)
        self._approval_allow.clicked.connect(lambda: self._answer(True))
        self._approval_always = QPushButton("以降は確認しない", self._approval_box)
        self._approval_always.clicked.connect(lambda: self._answer(True, always=True))
        deny = QPushButton("拒否", self._approval_box)
        deny.clicked.connect(lambda: self._answer(False))

        approval_buttons = QHBoxLayout()
        approval_buttons.setContentsMargins(0, 0, 0, 0)
        approval_buttons.addStretch(1)
        approval_buttons.addWidget(deny)
        approval_buttons.addWidget(self._approval_always)
        approval_buttons.addWidget(self._approval_allow)

        approval_layout = QVBoxLayout(self._approval_box)
        approval_layout.setContentsMargins(8, 6, 8, 6)
        approval_layout.addWidget(self._approval_text)
        approval_layout.addLayout(approval_buttons)

        # 上限に当たったとき（#344） 許可の箱と同じ形で、パネルの中に出す 窓にすると、
        # 任せて別の作業をしている人の前に急に割り込む
        self._budget_box = QFrame(self)
        self._budget_box.setFrameShape(QFrame.Shape.StyledPanel)
        self._budget_box.setVisible(False)
        self._budget_text = QLabel(self._budget_box)
        self._budget_text.setWordWrap(True)
        self._budget_go = QPushButton(self._budget_box)
        self._budget_go.clicked.connect(lambda: self._reply_budget(True))
        self._budget_stop = QPushButton(BUDGET_STOP_TEXT, self._budget_box)
        self._budget_stop.clicked.connect(lambda: self._reply_budget(False))
        budget_buttons = QHBoxLayout()
        budget_buttons.setContentsMargins(0, 0, 0, 0)
        budget_buttons.addStretch(1)
        budget_buttons.addWidget(self._budget_stop)
        budget_buttons.addWidget(self._budget_go)
        budget_layout = QVBoxLayout(self._budget_box)
        budget_layout.setContentsMargins(8, 6, 8, 6)
        budget_layout.addWidget(self._budget_text)
        budget_layout.addLayout(budget_buttons)

        self._build_question_box()

        # 長い作業の計画と進み具合（#340） 会話の欄の真下に置く 確かめの箱や質問の箱より上に
        # 置くのは、それらが出ている間も、どの手順の中で止まっているかが読めるように
        self._plan_box = PlanBox(self)
        self._plan_box.pause_requested.connect(self.pause_plan)
        self._plan_box.resume_requested.connect(self.resume_plan)
        self._plan_box.close_requested.connect(self.close_plan)

        # 部品が古くて断られたときの案内 その場で押せる更新のボタンを添える
        # 案内の文だけだと、どこで更新するのかを探させることになる
        self._update_box = QFrame(self)
        self._update_box.setFrameShape(QFrame.Shape.StyledPanel)
        self._update_box.setVisible(False)
        self._update_text = QLabel(self._update_box)
        self._update_text.setWordWrap(True)
        self._update_button = QPushButton(UPDATE_PARTS_LABEL, self._update_box)
        self._update_button.clicked.connect(self.update_parts)
        update_layout = QHBoxLayout(self._update_box)
        update_layout.setContentsMargins(8, 6, 8, 6)
        update_layout.addWidget(self._update_text, 1)
        update_layout.addWidget(self._update_button)

        self._input = _Input(self)
        self._describe_send_key()
        self._input.setMaximumHeight(96)
        # 最小は行の数で決める Qt の既定（巻物の欄の最小）は会話の欄と合わせて 140 画素あり、
        # 設定パネルと重ねた右の列の最小の高さになって、1280x720 の画面に窓が収まらなかった
        # 書体で行の高さが変わっても、会話は 2 行・入力は 1 行が必ず見える
        areas: tuple[tuple[QTextBrowser | QPlainTextEdit, int], ...] = (
            (self._view, 2),
            (self._input, 1),
        )
        for area, lines in areas:
            frame = 2 * area.frameWidth()
            margin = round(2 * area.document().documentMargin())
            area.setMinimumHeight(lines * area.fontMetrics().lineSpacing() + frame + margin)
        self._input.submitted.connect(self.send)

        # 状態の行 文字で出し、色だけに頼らない 秒が進むたびに書き換えるのはこの
        # 行の文字だけで、ほかの部品は作り直さない（#251 のように入力の途中で
        # 打っている位置が飛ばないように）
        self._busy_mark = _BusyMark(self)
        self._busy_mark.setVisible(False)
        self._status_text = QLabel(self)
        # 幅は文字に合わせて広げない 長い道具の名前で右の列の最小の幅が広がり、
        # 1366 の画面で窓に収まらなくなる 収まらない分は切れて見えるだけにする
        self._status_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        themed_style(self._status_text, lambda: f"color: {Colors.TEXT_MUTED.name()};")
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)
        status_row.addWidget(self._busy_mark)
        status_row.addWidget(self._status_text, 1)

        # 使った量（#344） 状態の行と同じく幅を文字に合わせて広げない 収まらない分は切れる
        # ので、全部はツールチップで読めるようにする
        self._usage_text = QLabel(self)
        self._usage_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        themed_style(self._usage_text, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._auto = QCheckBox("変更を自動で承認", self)
        # 入れても止まる所があると先に分かるようにする 知らないと、任せて席を外した後に
        # 確かめの窓が出たまま待っていたことに、戻ってから気付く
        self._auto.setToolTip(
            "取り消せる編集を確かめずに進めます 保存・書き出しなど取り消せない操作は、"
            "入れていても毎回確かめます"
        )
        self._auto.toggled.connect(self._set_auto_approve)
        self._send_button = QPushButton("送信", self)
        self._send_button.clicked.connect(self.send)
        self._stop_button = QPushButton("中断", self)
        self._stop_button.clicked.connect(self.interrupt)
        self._stop_button.setEnabled(False)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self._auto)
        controls.addStretch(1)
        controls.addWidget(self._stop_button)
        controls.addWidget(self._send_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self._choices)
        layout.addWidget(self._parts_box)
        layout.addWidget(self._login_box)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._plan_box)
        layout.addWidget(self._approval_box)
        layout.addWidget(self._budget_box)
        layout.addWidget(self._question_box)
        layout.addWidget(self._update_box)
        layout.addLayout(status_row)
        layout.addWidget(self._usage_text)
        layout.addWidget(self._input)
        layout.addLayout(controls)

    def _build_question_box(self) -> None:
        """AI からの質問の箱（#296） 承認の箱のすぐ下に出す

        箱ごと巻物にする 選択肢 4 つに説明を付けると 10 行を超え、そのまま足すと
        1280x720 の画面で入力欄と会話の欄が潰れる 広い所では全部が見え、狭い所では
        2 行まで縮んで巻ける
        """
        self._question_box = QScrollArea(self)
        self._question_box.setFrameShape(QFrame.Shape.StyledPanel)
        self._question_box.setWidgetResizable(True)
        self._question_box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 巻物の既定は縦にも広がりたがり、会話の欄から高さを奪う 中身の高さまでにする
        self._question_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        # 最小は 2 行 3 行にすると、狭い窓では AI のパネルの最小の高さが窓の最小の高さを
        # 決めるようになり、行の高い書体の機械で 1280x720 の画面の中身を超えうる
        lines = 2 * self.fontMetrics().lineSpacing()
        self._question_box.setMinimumHeight(lines + 2 * self._question_box.frameWidth())
        self._question_box.setVisible(False)

        inner = QWidget(self._question_box)
        self._question_text = QLabel(inner)
        self._question_text.setWordWrap(True)
        bold = self._question_text.font()
        bold.setBold(True)
        self._question_text.setFont(bold)
        # 選択肢は質問ごとに作り直す 入れ物だけを持つ
        self._question_choices = QWidget(inner)
        #: 複数選べる質問の、選択肢の名前とチェック 名前はボタンの文字から読み戻さない
        #: （& を印の文字に変えて出すので、読み戻すと AI が書いた名前と食い違う）
        self._question_checks: list[tuple[str, QCheckBox]] = []
        choices_layout = QVBoxLayout(self._question_choices)
        choices_layout.setContentsMargins(0, 0, 0, 0)
        choices_layout.setSpacing(4)
        self._question_note = QLabel(inner)
        self._question_note.setWordWrap(True)
        themed_style(self._question_note, lambda: f"color: {Colors.TEXT_MUTED.name()};")

        self._question_other = QPushButton("その他（入力欄に書く）", inner)
        self._question_other.setToolTip("選択肢に無い答えを、下の入力欄に書いて送る")
        self._question_other.clicked.connect(lambda: self._input.setFocus())
        self._question_send = QPushButton("選んだ物で答える", inner)
        self._question_send.clicked.connect(self._reply_checked)
        # 狭いパネルでも 2 つのボタンが欠けないように折り返す
        buttons = QWidget(inner)
        buttons_layout = FlowLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addWidget(self._question_other)
        buttons_layout.addWidget(self._question_send)

        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 6, 8, 6)
        inner_layout.addWidget(self._question_text)
        inner_layout.addWidget(self._question_choices)
        inner_layout.addWidget(self._question_note)
        inner_layout.addWidget(buttons)
        self._question_box.setWidget(inner)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt の命名規約
        super().resizeEvent(event)
        self._fit_wrapped()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt の命名規約
        super().showEvent(event)
        # 出る前に溜まっていた大きさの知らせは、出る印が付く前に配られて数えずに終わる
        # ここで置き方が決めた幅で数え直さないと、狭いパネルの折り返しが最小の高さに入らない
        self._shown = True
        self._fit_wrapped()

    def _fit_wrapped(self) -> None:
        """折り返す部品の、今の幅での高さを、パネルの最小の高さに入れる

        モデルの並び（FlowLayout）と、文を折り返す箱（ログインの案内・承認・部品の更新）は、
        幅が狭いと行が増えるのに、最小の高さは 1 行のまま数えられる パネルが最小の高さ
        まで縮められると（質問の箱を出した狭い窓など）、増えた行の分だけ下の部品が重なる
        並びだけを見ていたときは、ログインの済んでいない機械（CI）で案内の箱が並びに重なった
        箱の文は出し入れのたびに変わるので、状態を当て直す所（_refresh_status）からも呼ぶ

        1 度も出ていない間は数えない 出る前の幅は作ったときの仮の幅（100 画素など）で、
        そこで折り返すとログインの案内だけで 460 画素を超える その最小が窓の最小に足され、
        窓が出るときに頼んだ大きさ（覚えている大きさ・既定の大きさ）より縦に伸びて、
        落ち着いて最小が下がっても伸びたまま残った（#316 タイムラインが狭くなる）
        """
        if not self._shown:
            return
        width = self.width()
        layout = self.layout()
        if layout is not None:
            margins = layout.contentsMargins()
            width -= margins.left() + margins.right()
        for part in (self._choices, self._login_box, self._approval_box, self._update_box):
            # isVisible ではなく isHidden で見る 別のタブの裏にいる間に窓を縮められても、
            # 前に出したときの高さが合っているように
            if part.isHidden():
                continue
            need = part.heightForWidth(width)
            if need > 0 and part.minimumHeight() != need:
                part.setMinimumHeight(need)

    # --- 状態 ---

    def _refresh_availability(self) -> None:
        status = self._setup.status
        ready = status.ready
        # 使えない間は導入の欄が入口なので開いたままにし、閉じられないようにする
        self._parts_close.setVisible(ready)
        self._parts_button.setEnabled(ready)
        self._choices.setVisible(ready)
        self._show_parts(not ready)
        self._input.setEnabled(ready)
        self._send_button.setEnabled(ready)
        # 入っていない間はログインの話をしない 導入の案内と重なって読みにくい
        self._login_box.setVisible(ready and not credentials_found())
        if not ready:
            self._say("案内", status.summary())

    def _on_setup_finished(self, succeeded: bool) -> None:
        self._refresh_availability()
        if not succeeded:
            # 使える状態のまま失敗すると欄ごと隠れ、何が起きたかのログが読めなくなる
            self._show_parts(True)
        elif self._setup.note:
            # 使える状態になると導入欄ごと隠れるので、再起動の要る・要らないの
            # 案内は会話の欄へ写す 写さないと、読めないまま消える
            self._say("案内", self._setup.note)
        if self._setup.status.ready:
            # 入れている間に送って持っておいた指示を、入れ終えた部品の会話へ渡す
            self._pass_held()
        elif self._queued:
            # 入れ直しに失敗して使えなくなった 持っておいた指示は渡せないので、送れなかったと
            # 言って終える 持ったままだと、状態の行が実行中のまま待ち続ける
            self._note("AI 連携の環境が使えないため、待っていた指示は送りませんでした")
            self._finish_turn()
            self._close_checkpoint()
            self._queued.clear()
            self._stop_button.setEnabled(False)
        self._refresh_status()

    def _on_parts_toggled(self, opened: bool) -> None:
        """〔AI の部品〕を押した 使える間も、版を見て更新を確かめられるように欄を出し入れする"""
        if not opened and not self._setup.status.ready:
            # 使えない間は導入の欄が入口なので閉じさせない
            self._show_parts(True)
            return
        self._parts_box.setVisible(opened)

    def _show_parts(self, opened: bool) -> None:
        """「AI の部品」の欄を開く・閉じる ボタンの押下の見た目と印も合わせる"""
        self._parts_box.setVisible(opened)
        self._parts_button.set_open(opened)

    def update_parts(self) -> None:
        """AI の部品を新しい版へ入れ替える（手で押した〔AI の部品を更新〕）

        導入の欄の pip（配布版では同じプロセスの pip）で入れ、ログを見せる 走っている
        会話は先に畳む Windows では動いている claude.exe を書き換えられず、pip が失敗する
        """
        if self._hold_reason():
            # 自動の入れ替えの最中・AI が応えている最中は入れない ボタンも押せなくしてある
            # （同じ導入先へ同時に書く・動いている Claude Code を置き換えられない）
            return
        self._retrying = False
        self._stop_button.setEnabled(False)
        self._update_box.setVisible(False)
        self._show_parts(True)
        # 会話は導入の欄が畳む（prepare = _release_session） 畳み終わるのを待つのは裏のスレッド
        # 入れている間に送った指示は会話へ渡さずに持ち、終わってから渡す（_holding）
        # 入力欄は閉じない 導入の欄の〔環境を更新〕を直に押したときと同じにする
        self._setup.start(upgrade=True)
        self._refresh_status()

    def open_login(self) -> None:
        """Claude Code を別の窓で開き、そこでログインしてもらう"""
        cli = claude_cli()
        if cli is None:
            self.status_message.emit("Claude Code が見つかりません 先に環境を導入してください")
            return
        try:
            open_login_window(cli)
        except OSError as exc:
            self._say("エラー", f"Claude Code を開けませんでした: {exc}")
            return
        self._note("Claude Code を別の窓で開きました 案内に従ってログインしてください")

    # --- モデルと考える深さ ---

    @property
    def model(self) -> str:
        return str(self._model.currentData())

    @property
    def effort(self) -> str:
        return str(self._effort.currentData())

    def apply_preferences(self, preferences: Preferences) -> None:
        """本人の好み（設定画面か、前に欄の上で選んだもの）を当てる"""
        self._input.enter_sends = preferences.chat_enter_sends
        self._describe_send_key()
        # 選び直しの当て直し（_select_choices の中の _restart_when_idle）より先に持つ
        # 後にすると、切り替えた設定で会話を作り直す機会を逃す
        self._ask_questions = preferences.ai_ask_questions
        self._plan_first = preferences.ai_plan_first
        self._select_choices(preferences.ai_model, preferences.ai_effort)
        self._animate = preferences.ai_busy_animation
        self._alert_when_done = preferences.ai_done_alert
        self._updater.set_enabled(preferences.ai_auto_update)
        self._updater.schedule()
        self._bridge.set_send_choices(preferences.send_choices())
        # 上限は次の道具の前から効く 応えている途中で変えても、会話を作り直さずに済む
        self._bridge.usage.limits = preferences.budget_limits()
        self._refresh_status()

    def _select_choices(self, model: str, effort: str) -> None:
        # 当てるだけで「選び直した」と知らせない 知らせると、設定を読んだだけで
        # 保存と会話の張り直しが走る
        for box, value in ((self._model, model), (self._effort, effort)):
            box.blockSignals(True)
            box.setCurrentIndex(max(0, box.findData(value)))
            box.blockSignals(False)
        self._on_choice_changed(notify=False)

    def _on_choice_changed(self, _index: int = -1, *, notify: bool = True) -> None:
        choice = find_model(self.model)
        # 受け付けないモデルでは選べなくする 選べたままだと、選んだのに効いていない
        self._effort.setEnabled(choice is None or choice.effort)
        self._restart_when_idle()
        if notify:
            self.choices_changed.emit(self.model, self.effort)

    def _restart_when_idle(self) -> None:
        """選び直した組を、送った指示がすべて終わっていれば当てる

        SDK は会話の途中でエフォートを変えられないので、繋ぎ直す 指示が残って
        いる間は待ち、最後の応答が終わった所（_handle）でもう一度呼ばれる
        ``session.busy`` だけでは見ない 送った直後（Claude Code の起動と接続の
        数秒）は busy が立っておらず、その間に畳むと送った指示が消える
        """
        session = self._session
        if session is None or self._queued or session.busy:
            return
        if (session.model, session.effort) != self._wanted():
            self._restart_session()
        elif self._session_mode != self._host.project.settings.layer_mode:
            # 置き方の方式が会話の途中で変わった 指示の文（system_prompt）は会話を
            # 作るときにしか渡せないので、繋ぎ直して今の方式の説明を当てる 古い説明の
            # ままだと、混合にした作品で AI が無いはずの組の片方を探し回る
            self._restart_session(
                "置き方の方式が変わったので、次の指示から新しい会話を始めます"
                "（それまでのやり取りは引き継ぎません）"
            )
        elif session.ask != self._ask_questions:
            # 聞き返す基準と質問の道具も、会話を作るときにしか渡せない 繋ぎ直さないと、
            # 切った人がまだ質問され、入れた人は聞き返されない
            self._restart_session(
                "AI が質問するかの設定が変わったので、次の指示から新しい会話を始めます"
                "（それまでのやり取りは引き継ぎません）"
            )
        elif session.plan != self._plan_first:
            # 計画の決まりと道具も同じ（#340） 繋ぎ直さないと、切った人の会話でまだ計画を書く
            # 計画そのものは作品に付けて持つので、繋ぎ直しても〔続ける〕で続きから頼める
            self._restart_session(
                "AI が計画を示すかの設定が変わったので、次の指示から新しい会話を始めます"
                "（それまでのやり取りは引き継ぎません）"
            )

    def _settings_changed(self, session: AgentSession) -> bool:
        """会話を作るときにしか渡せない設定（質問と計画）が、会話を作った後に変わったか"""
        return session.ask != self._ask_questions or session.plan != self._plan_first

    def _wanted(self) -> tuple[str | None, str | None]:
        """いま選んでいる組を、会話が持つ形（空は None）で"""
        return self.model or None, effort_for(self.model, self.effort)

    def _restart_session(self, note: str = "") -> None:
        session = self._session
        if session is None:
            return
        self._session = None
        self._queued.clear()
        self._turn_started = None
        self._refresh_status()
        session.close(wait=False)
        label = self._model.currentText()
        self._note(
            note
            or f"次の指示から {label} で新しい会話を始めます（それまでのやり取りは引き継ぎません）"
        )

    def _carry_over_queue(self) -> None:
        """待ちの指示を残したまま会話を作り直し、待ちを新しい会話へ渡す

        :meth:`_restart_session` は待ちを消すので使わない 今の会話は区切りを付け終える
        知らせ（acknowledge_turn）を待って止まっている 畳むときに、受け取ったまま始めて
        いない指示は捨てられる（AgentSession.close）ので、同じ指示が 2 回走ることは無い
        """
        session = self._session
        if session is None:
            return
        self._session = None
        session.close(wait=False)
        self._note(
            "AI の進め方の設定（質問・計画）が変わったので、残りの指示は新しい会話で始めます"
            "（それまでのやり取りは引き継ぎません）"
        )
        self._pass_held()

    def _describe_send_key(self) -> None:
        key = "Enter で送信 Shift+Enter で改行" if self._input.enter_sends else "Ctrl+Enter で送信"
        self._input.set_placeholder(f"編集の指示を書いて {key}（例: 冒頭 10 秒を切って）")

    def _set_auto_approve(self, enabled: bool) -> None:
        self._bridge.auto_approve = enabled
        if not enabled:
            self._bridge.forget_always()

    # --- 送受信 ---

    def send(self) -> None:
        """入力欄の内容を送る"""
        prompt = self._input.toPlainText().strip()
        if not prompt:
            return
        if self._question is not None:
            # 質問が出ている間に送った文は答え（自由な記述）にする 新しい指示として並べると、
            # 返事を待って止まっている今の指示が終わらず、並べた指示もいつまでも始まらない
            # 複数選べる質問なら、チェックした物も一緒に渡す
            if not self._question.spec.free_text:
                # 選択肢に無い答えを断った質問 渡すと、AI が決めた答えの形の外の文が届く
                # 書いた文は消さずに残す 答えた後に新しい指示として送れるように
                self._note("この質問には選択肢から答えてください（入力欄の文は送っていません）")
                return
            self._input.clear()
            self._reply_question(self._checked_labels(), prompt)
            return
        if not self._setup.status.ready:
            self.status_message.emit("AI 連携の環境が入っていません")
            return
        self._input.clear()
        self._submit(prompt)

    def _submit(self, prompt: str) -> None:
        """指示を並べて会話へ渡す 入力欄から送った指示も、計画の〔続ける〕も同じ道を通る"""
        # 「ログイン…」から済ませたかもしれない 済んでいれば案内を下げる
        self._login_box.setVisible(not credentials_found())
        # 方式を変えても知らせは来ない 送る前に確かめ、前の指示がすべて終わっていれば
        # 今の方式の説明で会話を作り直す（指示が残っている間は、終わった所で当てる）
        self._restart_when_idle()
        self._say("あなた", prompt)
        self._queued.append(prompt)
        if len(self._queued) == 1:
            # 応答待ちの指示が他に無いときだけ段を開く 残っているときは、前の
            # 指示が終わった所（_handle）で、この指示の段を開く
            self._open_checkpoint(prompt)
            self._begin_turn()

        self._stop_button.setEnabled(True)
        if not self._holding:
            self._dispatch(prompt)
        # 送った直後に「接続しています…」へ変える Claude Code の起動に数秒かかり、
        # その間に何も変わらないと、送れたのか分からない 部品を入れ替えている間は
        # 「更新しています…」と待ちの件数を出す
        self._refresh_status()

    @property
    def _holding(self) -> bool:
        """指示を会話へ渡さずに持っておく間か（部品を入れ替えている・入れ替えを待っている）

        入れ替えの最中に会話を始めると、入れ替える途中の Claude Code を起動してしまう
        導入の欄の〔環境を更新〕も同じ 数分かかる pip の間に会話を始めると、同じ導入先から
        SDK と claude.exe を読み始め、使用中の exe を置き換えられずに pip が失敗するか、
        書き換えている途中の部品を import する（PR #265 の Codex の指摘）
        持っておいた指示は、入れ替えが終わった所（:meth:`_on_parts_updated` と
        :meth:`_on_setup_finished`）で順に渡す（:meth:`_pass_held`）
        """
        return self._retrying or self._updater.installing or self._setup.busy

    def _pass_held(self) -> None:
        """持っておいた指示を会話へ渡す まだ持っておく間なら何もしない

        自動の入れ替えと導入の欄の終わりはどちらが先に来るか分からない 先に終わった方で
        渡すと、まだ入れている方の途中で会話を始めてしまう
        """
        if self._holding:
            return
        if self._usage_waiting and self._queued:
            # 持っておいた先頭の指示は、ここで初めて AI へ渡る 時間もここから数える
            self._usage_waiting = False
            self._bridge.begin_turn()
        for prompt in self._queued:
            self._dispatch(prompt)

    def _dispatch(self, prompt: str) -> None:
        """指示を会話へ渡す 会話がまだ無ければ作る"""
        if self._session is None:
            self._connected = False
            model, effort = self._wanted()
            # 会話を始めた時点の方式で指示を書く 分ける方式の説明のまま混合の作品を
            # 触らせると、リンクした音声クリップを探し回る
            layer_mode = self._host.project.settings.layer_mode
            self._session_mode = layer_mode
            self._session = AgentSession(
                self._bridge,
                model=model,
                effort=effort,
                system_prompt=system_prompt(
                    layer_mode, ask=self._ask_questions, plan=self._plan_first
                ),
                ask=self._ask_questions,
                plan=self._plan_first,
            )
        self._session.send(prompt)

    def _release_session(self) -> Callable[[], None]:
        """部品を入れ替える前に会話を畳む 戻り値は、畳み終わるのを待つ物（裏のスレッドで呼ぶ）

        ここで待つと画面が固まる（Claude Code が終わるまで数秒かかる）ので、待つのは
        入れ替えの直前、裏のスレッドで行う
        """
        session = self._session
        self._session = None
        if session is None:
            return _nothing
        session.close(wait=False)
        return session.wait_closed

    def _on_parts_updated(self, succeeded: bool, installed: str) -> None:
        """部品の入れ替えが終わった 持っておいた指示を新しい会話へ渡す"""
        retrying = self._retrying
        self._retrying = False
        if succeeded and installed:
            self.status_message.emit(f"AI の部品を新しくしました（{installed}）")
        if retrying:
            if succeeded:
                self._note("AI の部品を更新して送り直しました")
                # 送り直しの応答で、もう一度断られたかどうかを見分けられるように
                # 断られた分の「失敗」も持ち越さない 送り直しが通れば「完了」と出す
                self._outdated = None
                self._turn_error = False
            else:
                # 自動では直せなかった ここで初めて、何が起きたかと手での直し方を見せる
                # 入れ替えをやめた（設定で切った）ときも同じ 送り直しはやめ、続けて送って
                # 持っていた指示は今の版の会話へ渡す（持ったままだと、いつまでも応えない）
                self._show_outdated(self._outdated or OutdatedClaudeCode())
                self._give_up_turn()
                self._refresh_status()
                return
        self._pass_held()
        self._refresh_status()

    def _show_outdated(self, found: OutdatedClaudeCode) -> None:
        message = found.message()
        self._say("エラー", message)
        self._update_text.setText(message)
        self._update_box.setVisible(True)

    def interrupt(self) -> None:
        if self._session is not None:
            self._session.interrupt()
            self._note("中断しています…")
            if self._queued:
                self._stopping = True
                self._refresh_status()

    @property
    def working(self) -> bool:
        """指示に応えている最中か、AI 連携の環境を入れている最中か

        更新のための再起動を断るのに使う 送った直後（Claude Code の起動と接続の数秒）は
        ``session.busy`` が立っていないので、残っている指示も見る
        """
        session = self._session
        return (
            bool(self._queued)
            or (session is not None and session.busy)
            or self._setup.busy
            or self._updater.installing
        )

    def close_session(self) -> None:
        """会話を畳む ウィンドウを閉じるときに呼ぶ"""
        self._timer.stop()
        self._updater.stop()
        self._setup.cancel()
        if self._session is not None:
            self._session.close()
            self._session = None
        # 応えている途中で閉じた指示も、使った量を 1 日の置き場へ足す（#344） 足さないと、
        # 起動し直した後にその分だけ 1 日の上限を越えて使える 会話を畳んだ後に締めるのは、
        # 畳むまでに届いた知らせも数えるため
        self._bridge.end_turn()
        self._close_checkpoint()

    # --- 1 プロンプト = 1 Undo ---

    def _open_checkpoint(self, prompt: str) -> None:
        """この指示による編集をまとめて戻せるようにする"""
        if self._checkpoint_open:
            return
        label = prompt.strip().splitlines()[0]
        if len(label) > 24:
            label = label[:23] + "…"
        self._host.document.begin_checkpoint(f"AI: {label}", by=Author.AI)
        self._checkpoint_open = True

    def _close_checkpoint(self) -> None:
        if not self._checkpoint_open:
            return
        self._checkpoint_open = False
        self._host.document.end_checkpoint()

    # --- ワーカーの見張り ---

    def _poll(self) -> None:
        # まず AI からの依頼を実行する ここが UI スレッド
        self._bridge.pump()
        self._check_approval()
        self._check_question()
        self._check_send()
        self._check_budget()
        now = self._clock()
        if now - self._flushed_at >= USAGE_FLUSH_SECONDS:
            # 道具を使わずに長く書いている間の量も時々足す アプリが落ちても失うのはこの間の分だけ
            self._flushed_at = now
            self._bridge.usage.flush()

        session = self._session
        if session is not None:
            for event in session.poll():
                self._handle(event)
        # 部品の入れ替えは AI が応えていない間だけ 送り直しを待っている間は、前の会話は
        # 次の指示を始めずに止まっている（区切りを付け終えるのを待っている）ので暇と見る
        session = self._session
        idle = self._retrying or session is None or not (self._queued or session.busy)
        # 導入の欄が入れている間も入れ替えない 同じ導入先へ同時に書くと新旧が混ざる
        # （導入先へ書く所は runtime.writing_runtime の錠でも並べてある）
        self._updater.tick(idle and not self._setup.busy)
        # 自動の入れ替えの間と AI が応えている間は、導入の欄からも〔AI の部品を更新〕からも
        # 入れられないようにする わけはツールチップで言う
        reason = self._hold_reason()
        self._setup.hold(reason)
        if self._update_button.isEnabled() == bool(reason):
            self._update_button.setEnabled(not reason)
            self._update_button.setToolTip(reason)
        # 秒の進みと承認の箱の出入りを拾う 変わった所だけを書き換えるので、毎回呼んでも軽い
        self._refresh_status()
        if not self._busy_mark.isHidden():
            self._busy_mark.advance()

    def _handle(self, event: AgentEvent) -> None:
        was_working = bool(self._queued)
        if event.kind is EventKind.CLOSED:
            # 次に送るときは Claude Code の起動からやり直す
            self._connected = False
        elif event.kind is not EventKind.ERROR:
            # 何か返ってきたら繋がっている 起動の失敗はエラーで来るので数えない
            self._connected = True

        if event.kind in (EventKind.TEXT, EventKind.ERROR):
            found = outdated_claude_code(event.text)
            if found is not None:
                # 部品が古くて選んだモデルを使えない 英語の文面は出さない（「claude update を
                # 打て」と言うが、同梱の物はそれでは上がらない） 自動で直せるなら黙って直し、
                # 直せないときだけ案内とボタンを出す
                self._turn_error = True
                if self._outdated is None and not self._can_retry():
                    self._show_outdated(found)
                self._outdated = found
                self._refresh_status()
                return

        if event.kind is EventKind.TURN_DONE and self._outdated is not None and self._can_retry():
            self._retry_after_update()
        elif event.kind is EventKind.TEXT:
            self._say("Claude", event.text)
        elif event.kind is EventKind.TOOL_USE:
            self._turn_tools += 1
            self._tool = event.tool
            if event.tool != ASK_USER:
                # 質問は箱と会話の欄の「質問」で見せる 引数の 1 行（choices=[{...}]）を
                # 並べても読めず、同じ質問が 2 回出るように見える
                self._note(f"▸ {event.tool} {event.detail}")
        elif event.kind is EventKind.TOOL_RESULT:
            if event.text == "失敗":
                self._turn_failures += 1
                self._note(f"　× {event.detail}")
        elif event.kind is EventKind.ERROR:
            self._turn_error = True
            self._say("エラー", event.text)
            if REINSTALL_HINT in event.text:
                # 入れ直す所（環境の導入の欄）は、導入済みのときは隠している 出さないと、
                # 案内された「環境を更新」がどこにも無い
                self._show_parts(True)
        elif event.kind is EventKind.TURN_DONE or event.kind is EventKind.CLOSED:
            self._turns_done += 1
            self._close_checkpoint()
            # 終わった印を会話の欄に残す 中断ボタンが灰色に戻るだけだと気付かない
            self._finish_turn()
            if event.kind is EventKind.TURN_DONE:
                if self._queued:
                    self._queued.popleft()
                if self._queued:
                    # 続けて送った指示の段を、その指示が始まる前に開く
                    self._open_checkpoint(self._queued[0])
                    self._begin_turn()
                session = self._session
                if session is not None and self._queued and self._settings_changed(session):
                    # 応えている間に「AI が質問する」か「計画を示す」を変えた 残りの指示を今の
                    # 会話で続けると、切った後も質問され（計画を書かれ）、入れた後も聞き返されない
                    # 会話を作り直して渡し直す
                    self._carry_over_queue()
                elif session is not None:
                    # 段を付け替え終えてから次の指示を始めさせる 先に始めると、
                    # 次の指示の編集が前の段へ混ざる
                    session.acknowledge_turn()
            else:
                # 会話が終わった 残っていた指示はもう返ってこない
                self._queued.clear()
            if not self._queued:
                # 続けて送った指示がまだ残っているなら、中断ボタンは生かしておく
                self._stop_button.setEnabled(False)
            # 応答の途中で選び直した分を、送った指示が全部終わった所で当てる
            self._restart_when_idle()
        self._refresh_status()
        if was_working and not self._queued:
            self._alert_done()

    def _hold_reason(self) -> str:
        """今は部品を手で入れ替えさせないわけ 入れ替えてよければ空

        応えている間に入れると、会話を畳む所で画面が止まり、終わりきらない Claude Code の
        claude.exe を置き換えられずに pip が失敗する（PR #265 の Codex の指摘） 自動の更新と
        同じく、応え終わるのを待ってもらう
        """
        if self._updater.installing:
            return "AI の部品を入れ替えています 終わると押せます"
        session = self._session
        if self._queued or (session is not None and session.busy):
            return "AI が応えている間は更新できません 応え終わると押せます"
        return ""

    def _can_retry(self) -> bool:
        """部品を自動で新しくして、いまの指示を送り直せるか 送り直すのは 1 度だけ"""
        return not self._retried and self._updater.available

    def _retry_after_update(self) -> None:
        """部品が古くて断られた指示を、部品を新しくしてから送り直す

        区切りの行も段（取り消しの 1 回分）も閉じない 送り直した応答までを 1 つの
        指示として数える 前の会話は区切りを付け終える知らせ（acknowledge_turn）を
        待って止まっているので、続けて送った指示を先に始めてしまうことも無い
        """
        self._retried = True
        self._retrying = True
        if not self._updater.request_now():
            self._retrying = False
            self._show_outdated(self._outdated or OutdatedClaudeCode())
            self._give_up_turn()

    def _give_up_turn(self) -> None:
        """断られた指示を失敗として終え、続けて送ってあった指示は今の版の会話へ渡し直す

        前の会話は区切りを付け終える知らせを待って止まっているので畳み、新しい会話を作る
        続けて送った指示まで捨てると、送ったのに何も起きない指示が残る
        """
        self._release_session()
        self._finish_turn()
        self._close_checkpoint()
        if self._queued:
            self._queued.popleft()
        if not self._queued:
            self._stop_button.setEnabled(False)
            return
        self._open_checkpoint(self._queued[0])
        self._begin_turn()
        self._pass_held()

    def _alert_done(self) -> None:
        """送った指示がすべて終わった 頼まれていれば、別の窓を触っている人に知らせる"""
        if not self._alert_when_done:
            return
        window = self.window()
        if window.isActiveWindow():
            # 見ている人のタスクバーを光らせても、気が散るだけ
            return
        QApplication.alert(window)

    def _check_approval(self) -> None:
        showing = self._approval
        if showing is not None:
            # 中断でブリッジ側が畳んだ確認は、画面からも下げる
            if showing.done.is_set():
                self._approval = None
                self._approval_box.setVisible(False)
            return
        approval = self._bridge.take_approval()
        if approval is None:
            return
        self._approval = approval
        irreversible = approval.irreversible
        head = IRREVERSIBLE_HEAD if irreversible else "この操作を許可しますか？"
        self._approval_text.setText(f"{head}\n{approval.summary}")
        # 取り消せない操作は毎回確かめる（2026-10-14 利用者の決定） 「以降は確認しない」を
        # 出すと、押しても効かないボタンになる 許すボタンも、何が起きるかの言葉にする
        self._approval_always.setVisible(not irreversible)
        self._approval_allow.setText(IRREVERSIBLE_ALLOW_TEXT if irreversible else ALLOW_TEXT)
        self._approval_box.setVisible(True)
        if irreversible:
            # 「変更を自動で承認」に任せて別の窓にいる人は、ここで止まっていることに
            # 気付かない ステータスバーとタスクバーで知らせる
            self.status_message.emit("取り消せない操作の確かめを待っています（AI アシスタント）")
            window = self.window()
            if not window.isActiveWindow():
                QApplication.alert(window)

    def _answer(self, allowed: bool, *, always: bool = False) -> None:
        approval = self._approval
        if approval is None:
            return
        self._approval = None
        self._approval_box.setVisible(False)
        if allowed:
            if always and not approval.irreversible:
                self._bridge.allow_always(approval.tool)
            approval.allow()
        else:
            approval.deny()
        self._refresh_status()

    # --- 外へ送ってよいか（#342） ---

    def _check_send(self) -> None:
        """送ってよいかの確かめがあれば、窓を出して答える

        窓は閉じるまで待つ（モーダル） 待つ間も見張りの時計は回るので、2 つ目を出さない
        AI のスレッドは答えが出るまで止まっているので、窓の間に新しい仕事は来ない
        """
        if self._asking_send:
            return
        request = self._bridge.take_send_request()
        if request is None or request.done.is_set():
            # 画面が取り出す前に、中断で畳まれていた 出すと押しても何も起きない窓が残る
            return
        self._asking_send = True
        try:
            window = self.window()
            if not window.isActiveWindow():
                # 別の窓を触っている人は、AI が止まっている訳に気付かない
                QApplication.alert(window)
            answer = self.ask_send(self, request)
        finally:
            self._asking_send = False
        names = "・".join(kind.label for kind in request.kinds)
        if answer is None:
            request.give_up("ユーザーが送らずに窓を閉じました")
            self._note(f"{names} を送りませんでした（窓を閉じました 次に送るときにまた尋ねます）")
            return
        if not request.reply(answer):
            # 窓を出している間に、中断か時間切れで畳まれた 答えは AI へ届かない
            return
        if answer is SendAnswer.ALWAYS:
            choices = dict(self._bridge.send_choices)
            for kind in request.kinds:
                choices[kind] = SendChoice.ALLOW
                self.send_allowed.emit(kind.value)
            self._bridge.set_send_choices(choices)
            self._note(f"{names} を送ります（次からは尋ねません 表示 → 設定… で変えられます）")
        elif answer is SendAnswer.UNTIL_CLOSE:
            self._note(f"{names} をアプリを閉じるまで送ります")
        else:
            self._note(f"{names} を送りません（アプリを閉じるまで 表示 → 設定… で変えられます）")

    # --- 使う量の上限（#344） ---

    def _check_budget(self) -> None:
        """上限に当たった知らせと「続けますか」を拾う"""
        notice = self._bridge.take_budget_notice()
        if notice is not None:
            # 設定が「尋ねずに止める」 止めた訳を残さないと、AI が途中で投げ出したように見える
            self._say("上限", f"{notice.text()} ここで止めました {_CHANGE_LIMITS}")
            self._alert_when_away()
        if self._bridge.wants_halt and not self._halted and self._session is not None:
            # 断った訳を返しても道具を呼び続けている 呼ぶたびに API を使うので、会話ごと止める
            self._halted = True
            self._note("上限で止めた後も道具を呼び続けたので、会話を止めました")
            self.interrupt()
        showing = self._budget
        if showing is not None:
            if showing.done.is_set():
                # 時間切れか中断でブリッジ側が畳んだ 画面からも下げ、止めたことを残す
                self._hide_budget()
                if showing.reason:
                    self._note(f"上限の確かめを閉じました（{showing.reason}）")
            return
        request = self._bridge.take_budget_request()
        if request is None:
            return
        if request.done.is_set():
            # 取り出す前に畳まれていた 出すと、押しても何も起きない箱が残る
            return
        self._budget = request
        overrun = request.overrun
        self._budget_text.setText(f"{BUDGET_HEAD}\n{overrun.text()}")
        self._budget_go.setText(f"続ける（{overrun.extended()} まで）")
        self._budget_box.setVisible(True)
        self._alert_when_away()

    def _alert_when_away(self) -> None:
        """別の窓を触っている人に、AI が止まっていることをタスクバーで知らせる"""
        window = self.window()
        if not window.isActiveWindow():
            QApplication.alert(window)

    def _reply_budget(self, proceed: bool) -> None:
        request = self._budget
        if request is None:
            return
        self._hide_budget()
        if not request.reply(proceed):
            # 押す前に中断か時間切れで畳まれていた 答えは AI へ届かない
            return
        overrun = request.overrun
        if proceed:
            self._note(
                f"続けます（{overrun.kind.label}の上限を {overrun.extended()} まで広げました）"
            )
        else:
            self._say("上限", f"{overrun.text()} ここで止めました {_CHANGE_LIMITS}")
        self._refresh_status()

    def _hide_budget(self) -> None:
        self._budget = None
        self._budget_box.setVisible(False)

    def _usage_line(self) -> tuple[str, str]:
        """使った量の行（短い文, ツールチップの全文）

        応えている間はこの指示の量、待機中は今日の量を出す 両方を 1 行に並べると、AI の
        パネルの幅に収まらず肝心の所が切れる 全文はツールチップで読める
        """
        meter = self._bridge.usage
        limits = meter.limits
        turn = meter.turn_usage()
        today = _usage_text(meter.day_usage(), limits, daily=True)
        if turn is None:
            return f"今日: {today}", f"今日 AI が使った量: {today}"
        now = _usage_text(turn, limits, daily=False)
        return f"この指示: {now}", f"この指示: {now}\n今日: {today}"

    # --- AI からの質問（#296） ---

    def _check_question(self) -> None:
        showing = self._question
        if showing is not None:
            if showing.done.is_set():
                # 時間切れか中断でブリッジ側が畳んだ 画面からも下げ、答えずに終わったと残す
                self._hide_question()
                if not showing.answered:
                    self._note_folded(showing)
            return
        question = self._bridge.take_question()
        if question is None:
            return
        if question.done.is_set():
            # 画面が取り出す前に畳まれていた（中断の直後に積まれた・画面が塞がって時間が
            # 切れた） 出すと、押しても「もう閉じていた」になる箱が残る 出さずに訳だけ残す
            self._note_folded(question)
            return
        self._show_question(question)

    def _note_folded(self, question: Question) -> None:
        """答えずに畳まれた質問の訳を、会話の欄に残す"""
        reason = question.reason.split(" ")[0] or "答えずに終わりました"
        self._note(f"質問を閉じました（{reason}）")

    def _show_question(self, question: Question) -> None:
        self._question = question
        spec = question.spec
        self._question_text.setText(spec.text)
        layout = self._question_choices.layout()
        assert layout is not None  # 組み立てで必ず付けてある
        while layout.count():
            item = layout.takeAt(0)
            old = item.widget() if item is not None else None
            if old is not None:
                old.deleteLater()
        self._question_checks = []
        for choice in spec.choices:
            if spec.multiple:
                check = QCheckBox(_button_text(choice.label), self._question_choices)
                check.toggled.connect(self._refresh_question_send)
                self._question_checks.append((choice.label, check))
                picker: QWidget = check
            else:
                button = QPushButton(_button_text(choice.label), self._question_choices)
                button.clicked.connect(
                    lambda _checked=False, label=choice.label: self._reply_question((label,))
                )
                picker = button
            picker.setToolTip(choice.description)
            layout.addWidget(picker)
            if choice.description:
                # 説明はツールチップだけにしない 選択肢の違いは、押す前に読めないと選べない
                detail = QLabel(choice.description, self._question_choices)
                detail.setWordWrap(True)
                detail.setContentsMargins(16, 0, 0, 0)
                themed_style(detail, lambda: f"color: {Colors.TEXT_MUTED.name()};")
                layout.addWidget(detail)
        # 入力欄から送った文は、新しい指示ではなくこの質問への答えにする 新しい指示として
        # 扱うと、前の指示が終わるまで待たされ、その前の指示は返事を待ったまま止まる
        # 選択肢だけで答える質問では、入力欄の文を答えにしない（send で断る）
        if spec.free_text:
            note = (
                "選択肢に無い答えは、下の入力欄に書いて送ってください"
                " 質問が出ている間に入力欄から送った文は、この質問への答えになります"
            )
        else:
            note = "選択肢から選んでください この質問には入力欄からは答えられません"
        self._question_note.setText(note)
        self._question_other.setVisible(spec.free_text)
        self._question_send.setVisible(spec.multiple)
        self._refresh_question_send()
        self._question_box.setVisible(True)
        lines = [spec.text] + [
            f"・{choice.label}" + (f": {choice.description}" if choice.description else "")
            for choice in spec.choices
        ]
        self._say("質問", "\n".join(lines))
        self._refresh_status()

    def _refresh_question_send(self, _checked: bool = False) -> None:
        """複数選べる質問は、1 つも選んでいない間は答えられなくする"""
        self._question_send.setEnabled(any(check.isChecked() for _, check in self._question_checks))

    def _checked_labels(self) -> tuple[str, ...]:
        if self._question is None or not self._question.spec.multiple:
            return ()
        return tuple(label for label, check in self._question_checks if check.isChecked())

    def _reply_checked(self) -> None:
        self._reply_question(self._checked_labels())

    def _reply_question(self, selected: tuple[str, ...], free_text: str = "") -> None:
        """質問に答える 押したボタン・選んだ物・入力欄から送った文のどれからも来る"""
        question = self._question
        if question is None:
            return
        self._hide_question()
        if not question.reply(selected, free_text):
            # 押すのと同時に時間切れか中断で畳まれた AI にはもう届かない
            self._note("質問はもう閉じていたので、答えは届きませんでした")
            self._refresh_status()
            return
        parts = list(question.selected)
        if question.free_text:
            parts.append(question.free_text)
        self._say("答え", "\n".join(parts))
        self._refresh_status()

    def _hide_question(self) -> None:
        self._question = None
        self._question_box.setVisible(False)

    # --- 長い作業の計画（#340） ---

    def _refresh_plan(self) -> None:
        """計画の欄を今の計画に合わせる 別の作品を開いていたら、前の作品の計画を閉じる"""
        if self._plan.follow(self._host.document.generation):
            # 前の作品の計画を次の作品で続けると、別の作品のクリップを手順どおりに変えかねない
            self._note("別の作品を開いたので、前の作品の計画を閉じました")
        self._plan_box.show_plan(
            self._plan.view(),
            working=bool(self._queued),
            ready=self._setup.status.ready,
        )

    def pause_plan(self) -> None:
        """〔区切りで止める〕 今の手順を終えた所で止める（すぐ止めるのは〔中断〕）"""
        if not self._queued or not self._plan.request_pause():
            return
        self._note(
            "今の手順を終えた所で止めます（すぐに止めるなら〔中断〕 止めた後は〔続ける〕で"
            "続きを頼めます）"
        )
        self._refresh_status()

    def resume_plan(self) -> None:
        """〔続ける〕 止めた所から残りの手順を頼む 応えている間は並べない"""
        view = self._plan.view()
        if self._queued or view is None or view.finished or not self._setup.status.ready:
            # 応えている間に並べると、今の指示の後ろで「続けて」が走り、止めた所が分からなくなる
            return
        self._submit(RESUME_PROMPT)

    def close_plan(self) -> None:
        """計画の欄の〔×〕 応えている間は閉じない（AI が手順の印を書けずに失敗が並ぶ）"""
        if self._queued:
            return
        self._plan.discard()
        self._refresh_status()

    # --- 状態の行 ---

    def _begin_turn(self) -> None:
        """次の指示に取りかかった 秒と数を数え直す 計画を止めていたら外す（#340）"""
        self._plan.begin_turn(self._queued[0] if self._queued else "")
        self._turn_started = self._clock()
        # 使った量（#344）も指示ごとに数え直す 前の指示の量で次の指示が止まらないように
        # 部品を入れ替えている間は、渡す所まで数え始めない 先に時計を回すと、更新に数分
        # かかったとき、AI が指示を受け取る前の待ちだけで 1 回の時間の上限に当たって止まる
        self._usage_waiting = self._holding
        if not self._usage_waiting:
            self._bridge.begin_turn()
        self._halted = False
        self._turn_tools = 0
        self._turn_failures = 0
        self._turn_error = False
        self._stopping = False
        self._tool = ""
        self._outdated = None
        self._retried = False

    def _finish_turn(self) -> None:
        """1 回の応答が終わった 会話の欄へ区切りの行を足す

        送っていないのに終わりの知らせが来たとき（会話だけが閉じたときなど）は足さない
        何かが終わったように見えて紛らわしい
        """
        started = self._turn_started
        if started is None:
            return
        self._turn_started = None
        if self._stopping:
            outcome = "中断"
        elif self._turn_error:
            outcome = "失敗"
        else:
            outcome = "完了"
        self._bridge.end_turn()
        used = self._bridge.usage.last
        stopped = self._bridge.usage.last_stop
        if stopped is not None:
            # 上限で止めた指示を「完了」と書くと、頼んだことが全部済んだと読める
            # 止めた後も道具を呼び続けて会話ごと止めた（中断）ときも、止めた訳は上限
            outcome = "上限で停止"
        counts = f"操作 {self._turn_tools} 件"
        if self._turn_failures:
            counts += f"・失敗 {self._turn_failures} 件"
        elapsed = _duration(self._clock() - started)
        # 使った量も残す（#344） 終わった後に、どの指示が重かったかを見返せる
        # 知らせが来なかった物（繋がる前に落ちた指示など）は 0 と書かない 使わなかったと読める
        if used.tokens:
            counts += f"・{format_tokens(used.tokens)} トークン"
        if used.images:
            counts += f"・絵 {used.images} 枚"
        self._append(_DIVIDER, f"{outcome}（{elapsed}・{counts}）")
        # 手順が残っていれば、止まった訳を計画に残す（#340） 欄が〔続ける〕を出す
        self._plan.end_turn(interrupted=self._stopping, limited=stopped is not None)
        view = self._plan.view()
        if view is not None and view.stopped is Stopped.PAUSED:
            # 欄だけだと、区切りで止まったのか自分で終えたのかが会話の欄から読めない
            self._note(f"計画を区切りで止めました（{plan_progress(view)}）")
        self._stopping = False

    @property
    def _updating(self) -> bool:
        """部品を入れ替えている（手で押した更新・自動の更新・断られた後の立て直し）"""
        return self._setup.busy or self._updater.installing or self._retrying

    def _status_line(self) -> str:
        """状態の行の文 色だけに頼らず、文字で今の状態を言う"""
        if self._updating:
            # 送った指示は入れ替えが終わってから渡す 待たされている理由をここで言う
            line = "AI の部品を更新しています…"
            if self._queued:
                line += f"・待ち {len(self._queued)} 件"
            return line
        approval = self._approval
        question = self._question
        if not self._queued and approval is None and question is None and self._budget is None:
            if self._updater.struggling:
                # 1 回ごとには言わない（たまたま繋がらなかっただけの人を驚かせない）
                return (
                    "待機中（AI の部品の自動の更新がうまくいっていません"
                    " 〔AI の部品…〕から手で更新できます）"
                )
            return "待機中"
        session = self._session
        if self._stopping:
            head = "中断しています…"
        elif question is not None:
            # 答えるまで AI は先へ進まない 「実行中」のままだと、待たされているのが
            # こちらの返事だと気付かない
            head = "質問待ち"
        elif self._budget is not None:
            # 上限で止まっている 「実行中」のままだと、まだ動いていると思って待ち続ける
            head = "上限の確かめ待ち"
        elif approval is not None:
            head = f"許可待ち: {approval.tool}"
        elif not self._connected and not (session is not None and session.busy):
            head = "接続しています…"
        elif self._tool:
            head = f"実行中: {self._tool}"
        else:
            head = "実行中"
        started = self._turn_started
        line = head if started is None else f"{head}（{_duration(self._clock() - started)}）"
        plan = self._plan.view()
        if plan is not None and not plan.finished:
            # 長い作業のどこにいるかを、欄を巻き戻さなくても読めるようにする（#340）
            line += f"・{plan_progress(plan)}"
            if plan.pause_requested:
                line += "・区切りで止めます"
        waiting = len(self._queued) - 1
        if waiting > 0:
            line += f"・待ち {waiting} 件"
        return line

    def _refresh_status(self) -> None:
        """状態の行・回る印・送信ボタン・ドックの名前を今の状態に合わせる

        見張りの時計から 80 ミリ秒ごとに呼ばれるので、変わった所だけを書き換える
        同じ文を書き直すだけでも、文字の欄は大きさを測り直して窓の配置をやり直す
        計画の欄（#340）もここで合わせる AI が道具で書いた進み具合も、指示の始めと終わりも、
        この後の状態の行と同じ回で欄に出る
        """
        self._refresh_plan()
        line = self._status_line()
        if self._status_text.text() != line:
            self._status_text.setText(line)
        usage, full = self._usage_line()
        if self._usage_text.text() != usage:
            self._usage_text.setText(usage)
            self._usage_text.setToolTip(full)
        working = bool(self._queued)
        show_mark = (working or self._updating) and self._animate
        if self._busy_mark.isHidden() == show_mark:
            self._busy_mark.setVisible(show_mark)
        # 前の指示が終わるまで待たされることを、押す前に分かるようにする
        # 質問が出ている間は、送った文が答えになることもボタンで言う
        label = "追加で送る" if working else "送信"
        if self._question is not None:
            # 選択肢だけで答える質問では、押しても送らない それをボタンの名前でも言う
            label = "答えを送る" if self._question.spec.free_text else "選択肢から答える"
        if self._send_button.text() != label:
            self._send_button.setText(label)
        self._mark_dock(working)
        # 承認の箱やログインの案内を出し入れした後も、折り返した高さを最小に入れておく
        self._fit_wrapped()

    def _mark_dock(self, working: bool) -> None:
        """入れてあるドックのタブの名前に、実行中だけ印を付ける

        パネルが別のタブの裏に隠れていても、動いているかどうかが分かる ドックは
        窓の側が作るので、親をたどって探す（パネルの外に作り方を知らせずに済む）
        """
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QDockWidget):
            parent = parent.parentWidget()
        if parent is None:
            return
        title = parent.windowTitle()
        plain = title.removeprefix(RUNNING_MARK)
        wanted = RUNNING_MARK + plain if working else plain
        if title != wanted:
            parent.setWindowTitle(wanted)

    # --- 表示 ---

    def _say(self, who: str, text: str) -> None:
        self._append(who, text)

    def _note(self, text: str) -> None:
        """ツールの呼び出しなど、会話の本体ではないもの"""
        self._append(None, text)

    def _append(self, who: str | None, text: str) -> None:
        self._log.append((who, text))
        self._view.append(_message_html(who, text))
        self._scroll_to_end()

    def _redraw_log(self) -> None:
        """テーマが変わった 会話を今の色で書き直す

        色は HTML に焼き込んであるので、描き直すだけでは前のテーマの色のまま残る
        暗いテーマの白に近い文字が、明るい地の上で読めなくなる
        """
        self._view.clear()
        for who, text in self._log:
            self._view.append(_message_html(who, text))
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        bar = self._view.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())


def _message_html(who: str | None, text: str) -> str:
    """会話の 1 件を HTML へ ``who`` が ``None`` なら会話の本体ではない補足

    Claude の返事だけ太字と等幅を組む 本人の指示やエラーの文に ``**`` が
    あっても、書いたとおりに見せる
    """
    if who is None:
        return f'<span style="color:{Colors.TEXT_MUTED.name()}">{html.escape(text)}</span>'
    if who == _DIVIDER:
        # 線の文字で挟み、色を落としても区切りだと読めるようにする
        muted = Colors.TEXT_MUTED.name()
        return f'<p align="center" style="color:{muted}">──── {html.escape(text)} ────</p>'
    color = {
        "あなた": Colors.TEXT.name(),
        "Claude": Colors.ACCENT.name(),
        # 質問は Claude から、答えは本人から 言った人と同じ色にする
        "質問": Colors.ACCENT.name(),
        "答え": Colors.TEXT.name(),
        "エラー": Colors.PLAYHEAD.name(),
    }.get(who, Colors.TEXT_MUTED.name())
    body = _to_html(text) if who == "Claude" else html.escape(text).replace("\n", "<br>")
    return f'<b style="color:{color}">{html.escape(who)}</b><br>{body}<br>'


def _button_text(label: str) -> str:
    """ボタンに出す文字 Qt は & の次の文字を押し下げの印（Alt+文字）にして & を消す

    AI が「BGM & 効果音」と書いた選択肢が「BGM  効果音」に見えないように重ねて逃がす
    """
    return label.replace("&", "&&")


def _nothing() -> None:
    """待つ物が無いときの、待つ物"""


#: 上限で止めたときに、続け方と変え方を添える文
_CHANGE_LIMITS = (
    "続けるなら、もう 1 度指示を送ってください（上限は 表示 → 設定… の「AI の上限」で変えられます）"
)


def _usage_text(usage: Usage, limits: Limits, *, daily: bool) -> str:
    """使った量を「3.2 万 / 100 万トークン・絵 2 / 30 枚・1 分 5 秒 / 30 分」の形で

    上限が 0（無し）の物は量だけを出す 1 日の量は時間を出さない（時間の上限は指示ごと）
    """
    tokens_limit = limits.day_tokens if daily else limits.turn_tokens
    images_limit = limits.day_images if daily else limits.turn_images
    tokens = _with_limit(format_tokens(usage.tokens), tokens_limit, format_tokens(tokens_limit))
    images = _with_limit(str(usage.images), images_limit, str(images_limit))
    parts = [f"{tokens} トークン", f"絵 {images} 枚"]
    if not daily:
        seconds = limits.turn_seconds
        parts.append(_with_limit(format_minutes(usage.seconds), seconds, format_minutes(seconds)))
    return "・".join(parts)


def _with_limit(used: str, limit: int, shown: str) -> str:
    return f"{used} / {shown}" if limit > 0 else used


def _duration(seconds: float) -> str:
    """経過した時間を読みやすく 1 分を超えたら分も出す（125 秒より 2 分 5 秒の方が掴みやすい）"""
    whole = max(0, int(seconds))
    minutes, rest = divmod(whole, 60)
    return f"{minutes} 分 {rest} 秒" if minutes else f"{rest} 秒"


def _to_html(text: str) -> str:
    """本文を表示用の HTML へ

    先に文字実体へ逃がしてから太字と等幅を当てる 順番を逆にすると、本文に
    書かれた ``<b>`` がそのまま効いてしまう
    """
    escaped = html.escape(text)
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    code = rf'<code style="color:{Colors.WAVEFORM.name()}">\1</code>'
    escaped = _CODE.sub(code, escaped)
    return escaped.replace("\n", "<br>")
