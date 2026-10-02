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
from collections import deque

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QInputMethodEvent, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from sashimono.ai import AI_PACK, Approval, EditorBridge, EditorHost
from sashimono.ai.environment import claude_cli, credentials_found, open_login_window
from sashimono.ai.models import EFFORTS, MODELS, effort_for, find_model
from sashimono.ai.session import (
    REINSTALL_HINT,
    AgentEvent,
    AgentSession,
    EventKind,
    system_prompt,
)
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

    def __init__(self, host: EditorHost, parent: QWidget | None = None) -> None:
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

        self._build()
        theme_signals().changed.connect(self._redraw_log)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._refresh_availability()

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

        choices = QHBoxLayout()
        choices.setContentsMargins(0, 0, 0, 0)
        choices.addWidget(QLabel("モデル", self))
        choices.addWidget(self._model, 1)
        choices.addWidget(QLabel("考える深さ", self))
        choices.addWidget(self._effort)

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

        self._input = _Input(self)
        self._describe_send_key()
        self._input.setMaximumHeight(96)
        self._input.submitted.connect(self.send)

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
        layout.addWidget(self._setup)
        layout.addWidget(self._login_box)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._approval_box)
        layout.addWidget(self._input)
        layout.addLayout(controls)

    # --- 状態 ---

    def _refresh_availability(self) -> None:
        status = self._setup.status
        ready = status.ready
        self._setup.setVisible(not ready)
        self._input.setEnabled(ready)
        self._send_button.setEnabled(ready)
        # 入っていない間はログインの話をしない 導入の案内と重なって読みにくい
        self._login_box.setVisible(ready and not credentials_found())
        if not ready:
            self._say("案内", status.summary())

    def _on_setup_finished(self, succeeded: bool) -> None:
        self._refresh_availability()
        if succeeded and self._setup.note:
            # 使える状態になると導入欄ごと隠れるので、再起動の要る・要らないの
            # 案内は会話の欄へ写す 写さないと、読めないまま消える
            self._say("案内", self._setup.note)

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

        if self._session is None:
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
        self._stop_button.setEnabled(True)

    def interrupt(self) -> None:
        if self._session is not None:
            self._session.interrupt()
            self._note("中断しています…")

    def close_session(self) -> None:
        """会話を畳む ウィンドウを閉じるときに呼ぶ"""
        self._timer.stop()
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
        if session is None:
            return
        for event in session.poll():
            self._handle(event)

    def _handle(self, event: AgentEvent) -> None:
        if event.kind is EventKind.TEXT:
            self._say("Claude", event.text)
        elif event.kind is EventKind.TOOL_USE:
            self._note(f"▸ {event.tool} {event.detail}")
        elif event.kind is EventKind.TOOL_RESULT:
            if event.text == "失敗":
                self._note(f"　× {event.detail}")
        elif event.kind is EventKind.ERROR:
            self._say("エラー", event.text)
            if REINSTALL_HINT in event.text:
                # 入れ直す所（環境の導入の欄）は、導入済みのときは隠している 出さないと、
                # 案内された「環境を更新」がどこにも無い
                self._setup.setVisible(True)
        elif event.kind is EventKind.TURN_DONE or event.kind is EventKind.CLOSED:
            self._turns_done += 1
            self._close_checkpoint()
            if event.kind is EventKind.TURN_DONE:
                if self._queued:
                    self._queued.popleft()
                if self._queued:
                    # 続けて送った指示の段を、その指示が始まる前に開く
                    self._open_checkpoint(self._queued[0])
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

    # --- 表示 ---

    def _say(self, who: str, text: str) -> None:
        self._log.append((who, text))
        self._view.append(_message_html(who, text))
        self._scroll_to_end()

    def _note(self, text: str) -> None:
        """ツールの呼び出しなど、会話の本体ではないもの"""
        self._log.append((None, text))
        self._view.append(_message_html(None, text))
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
    color = {
        "あなた": Colors.TEXT.name(),
        "Claude": Colors.ACCENT.name(),
        "エラー": Colors.PLAYHEAD.name(),
    }.get(who, Colors.TEXT_MUTED.name())
    body = _to_html(text) if who == "Claude" else html.escape(text).replace("\n", "<br>")
    return f'<b style="color:{color}">{html.escape(who)}</b><br>{body}<br>'


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
