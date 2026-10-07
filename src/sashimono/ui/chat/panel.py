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
from PySide6.QtGui import QColor, QInputMethodEvent, QKeyEvent, QPainter, QPaintEvent, QPen
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
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.ai import AI_PACK, Approval, EditorBridge, EditorHost
from sashimono.ai.environment import claude_cli, credentials_found, open_login_window
from sashimono.ai.models import EFFORTS, MODELS, effort_for, find_model
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
from sashimono.ui.chat.parts_updater import PartsUpdater
from sashimono.ui.disclosure import DisclosureButton
from sashimono.ui.flow_layout import FlowLayout
from sashimono.ui.setup import SetupSection
from sashimono.ui.theme import Colors, theme_signals, themed_style
from sashimono.ui.workspace import Preferences

__all__ = ["ChatPanel"]

#: 太字と等幅だけを拾う Claude の返事は素の Markdown で来るので、
#: そのまま出すと ** が本文に混ざる 見出しや表まで組む必要は無い
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_CODE = re.compile(r"`([^`]+)`")

#: ワーカーからの知らせを拾う間隔（ミリ秒）
#: ここを長くすると、AI の操作が画面へ反映されるまでの間が空く
POLL_MS = 80

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

    def inputMethodEvent(self, event: QInputMethodEvent) -> None:  # noqa: N802 - Qt の命名規約
        # 変換中の文字（まだ確定していない読み）があるかどうかで判断する
        # 確定した瞬間は preedit が空になって届くので、そこで変換の終わりと見る
        self._composing = bool(event.preeditString())
        super().inputMethodEvent(event)

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

    def __init__(
        self,
        host: EditorHost,
        parent: QWidget | None = None,
        *,
        updater: PartsUpdater | None = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._bridge = EditorBridge(host)
        self._session: AgentSession | None = None
        #: 会話を作ったときの置き方の方式 指示の文をこの方式で書いたので、変わったら作り直す
        self._session_mode = ""
        self._approval: Approval | None = None
        self._checkpoint_open = False
        #: 応答が 1 往復終わった回数 無人での確認に使う
        self._turns_done = 0
        #: 送ってまだ応答が終わっていない指示（送った順） 会話を繋ぎ直してよいかの
        #: 判断と、続けて送った指示の取り消しの段を、その指示の名前で開くのに使う
        self._queued: deque[str] = deque()
        #: 出した会話（言った人、文） 補足は言った人が ``None`` テーマを切り替えたときに
        #: 今の色で書き直すために持つ
        self._log: list[tuple[str | None, str]] = []

        # --- いま応えている指示（状態の行と区切りの行に使う） ---
        #: 経過秒を測る時計 試験で差し替える（本物の時計だと秒の境目で揺れる）
        self._clock: Callable[[], float] = time.monotonic
        #: いまの指示に取りかかった時刻 応えている指示が無ければ ``None``
        self._turn_started: float | None = None
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
        choices = FlowLayout()
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
        parts_layout = QVBoxLayout(self._parts_box)
        parts_layout.setContentsMargins(8, 4, 8, 8)
        parts_layout.setSpacing(4)
        parts_layout.addLayout(parts_header)
        parts_layout.addWidget(self._setup)
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

        allow = QPushButton("許可", self._approval_box)
        allow.clicked.connect(lambda: self._answer(True))
        always = QPushButton("以降は確認しない", self._approval_box)
        always.clicked.connect(lambda: self._answer(True, always=True))
        deny = QPushButton("拒否", self._approval_box)
        deny.clicked.connect(lambda: self._answer(False))

        approval_buttons = QHBoxLayout()
        approval_buttons.setContentsMargins(0, 0, 0, 0)
        approval_buttons.addStretch(1)
        approval_buttons.addWidget(deny)
        approval_buttons.addWidget(always)
        approval_buttons.addWidget(allow)

        approval_layout = QVBoxLayout(self._approval_box)
        approval_layout.setContentsMargins(8, 6, 8, 6)
        approval_layout.addWidget(self._approval_text)
        approval_layout.addLayout(approval_buttons)

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

        self._auto = QCheckBox("変更を自動で承認", self)
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
        layout.addLayout(choices)
        layout.addWidget(self._parts_box)
        layout.addWidget(self._login_box)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._approval_box)
        layout.addWidget(self._update_box)
        layout.addLayout(status_row)
        layout.addWidget(self._input)
        layout.addLayout(controls)

    # --- 状態 ---

    def _refresh_availability(self) -> None:
        status = self._setup.status
        ready = status.ready
        # 使えない間は導入の欄が入口なので開いたままにし、閉じられないようにする
        self._parts_close.setVisible(ready)
        self._parts_button.setEnabled(ready)
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
        if self._setup.busy:
            return
        session = self._session
        self._session = None
        if session is not None:
            session.close()
        if self._queued:
            # 待っていた指示はもう返ってこない 区切りを付けて終わらせる
            self._finish_turn()
            self._queued.clear()
            self._close_checkpoint()
        self._retrying = False
        self._stop_button.setEnabled(False)
        self._update_box.setVisible(False)
        self._show_parts(True)
        self._setup.start(upgrade=True)
        if self._setup.busy:
            self._input.setEnabled(False)
            self._send_button.setEnabled(False)
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
        self._select_choices(preferences.ai_model, preferences.ai_effort)
        self._animate = preferences.ai_busy_animation
        self._alert_when_done = preferences.ai_done_alert
        self._updater.set_enabled(preferences.ai_auto_update)
        self._updater.schedule()
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

    def _describe_send_key(self) -> None:
        key = "Enter で送信 Shift+Enter で改行" if self._input.enter_sends else "Ctrl+Enter で送信"
        self._input.setPlaceholderText(f"編集の指示を書いて {key}（例: 冒頭 10 秒を切って）")

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
        if not self._setup.status.ready:
            self.status_message.emit("AI 連携の環境が入っていません")
            return

        # 「ログイン…」から済ませたかもしれない 済んでいれば案内を下げる
        self._login_box.setVisible(not credentials_found())
        # 方式を変えても知らせは来ない 送る前に確かめ、前の指示がすべて終わっていれば
        # 今の方式の説明で会話を作り直す（指示が残っている間は、終わった所で当てる）
        self._restart_when_idle()
        self._input.clear()
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
        持っておいた指示は、入れ替えが終わった所（:meth:`_on_parts_updated`）で順に渡す
        """
        return self._retrying or self._updater.installing

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
                system_prompt=system_prompt(layer_mode),
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
                self._show_outdated(self._outdated or OutdatedClaudeCode())
                self._give_up_turn()
                self._refresh_status()
                return
        for prompt in self._queued:
            self._dispatch(prompt)
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
        self._close_checkpoint()

    # --- 1 プロンプト = 1 Undo ---

    def _open_checkpoint(self, prompt: str) -> None:
        """この指示による編集をまとめて戻せるようにする"""
        if self._checkpoint_open:
            return
        label = prompt.strip().splitlines()[0]
        if len(label) > 24:
            label = label[:23] + "…"
        self._host.document.begin_checkpoint(f"AI: {label}")
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

        session = self._session
        if session is not None:
            for event in session.poll():
                self._handle(event)
        # 部品の入れ替えは AI が応えていない間だけ 送り直しを待っている間は、前の会話は
        # 次の指示を始めずに止まっている（区切りを付け終えるのを待っている）ので暇と見る
        session = self._session
        idle = self._retrying or session is None or not (self._queued or session.busy)
        self._updater.tick(idle)
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
                if self._session is not None:
                    # 段を付け替え終えてから次の指示を始めさせる 先に始めると、
                    # 次の指示の編集が前の段へ混ざる
                    self._session.acknowledge_turn()
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
        """いまの指示を失敗として終える 待っていた指示も前の会話と一緒に畳む"""
        self._release_session()
        self._finish_turn()
        self._queued.clear()
        self._close_checkpoint()
        self._stop_button.setEnabled(False)

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
        self._approval_text.setText(f"この操作を許可しますか？\n{approval.summary}")
        self._approval_box.setVisible(True)

    def _answer(self, allowed: bool, *, always: bool = False) -> None:
        approval = self._approval
        if approval is None:
            return
        self._approval = None
        self._approval_box.setVisible(False)
        if allowed:
            if always:
                self._bridge.allow_always(approval.tool)
            approval.allow()
        else:
            approval.deny()
        self._refresh_status()

    # --- 状態の行 ---

    def _begin_turn(self) -> None:
        """次の指示に取りかかった 秒と数を数え直す"""
        self._turn_started = self._clock()
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
        counts = f"操作 {self._turn_tools} 件"
        if self._turn_failures:
            counts += f"・失敗 {self._turn_failures} 件"
        elapsed = _duration(self._clock() - started)
        self._append(_DIVIDER, f"{outcome}（{elapsed}・{counts}）")
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
        if not self._queued and approval is None:
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
        waiting = len(self._queued) - 1
        if waiting > 0:
            line += f"・待ち {waiting} 件"
        return line

    def _refresh_status(self) -> None:
        """状態の行・回る印・送信ボタン・ドックの名前を今の状態に合わせる

        見張りの時計から 80 ミリ秒ごとに呼ばれるので、変わった所だけを書き換える
        同じ文を書き直すだけでも、文字の欄は大きさを測り直して窓の配置をやり直す
        """
        line = self._status_line()
        if self._status_text.text() != line:
            self._status_text.setText(line)
        working = bool(self._queued)
        show_mark = (working or self._updating) and self._animate
        if self._busy_mark.isHidden() == show_mark:
            self._busy_mark.setVisible(show_mark)
        # 前の指示が終わるまで待たされることを、押す前に分かるようにする
        label = "追加で送る" if working else "送信"
        if self._send_button.text() != label:
            self._send_button.setText(label)
        self._mark_dock(working)

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
        "エラー": Colors.PLAYHEAD.name(),
    }.get(who, Colors.TEXT_MUTED.name())
    body = _to_html(text) if who == "Claude" else html.escape(text).replace("\n", "<br>")
    return f'<b style="color:{color}">{html.escape(who)}</b><br>{body}<br>'


def _nothing() -> None:
    """待つ物が無いときの、待つ物"""


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
