"""AI スレッドと UI スレッドの橋渡し

Claude Agent SDK は自前の asyncio ループを別スレッドで回す そこからウィジェットや
:class:`~sashimono.core.commands.Document` を直に触ると Qt が落ちるので、**やることを
キューへ積んで UI スレッドに実行させる**

同じ仕組みで確認（承認）も運ぶ 変更系のツールは、UI 側が「許可」を返すまでここで
止まる 止まっている間も UI は動き続けるので、内容を見てから決められる
確かめは二段（2026-10-14 利用者の決定 #350） 取り消せる編集は「変更を自動で承認」で
省けるが、取り消せない操作（:attr:`Operation.irreversible`）は毎回確かめる
AI からの質問（ask_user #296）も同じ形で運び、答えが返るまで止まる

:class:`~sashimono.engine.cache.MediaAnalyzer` や字幕起こしと同じ「キューに積んで
タイマーで拾う」形にしてある Qt のシグナルをスレッドまたぎで使うより、いつ実行
されるかが読みやすい
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sashimono.ai.host import EditorHost, ToolError
from sashimono.ai.operations import Operation, QuestionSpec, Waiting, find_operation
from sashimono.ai.safety import Consent

__all__ = ["Approval", "EditorBridge", "Question"]

#: UI スレッドが 1 件の仕事を拾って返すまで待つ上限（秒）
#: これを超えるのは、書き出しなど UI 側が長く塞がっているとき
CALL_TIMEOUT = 60.0

#: 承認の待ち時間 人が席を外していることもあるので長めに取る
APPROVAL_TIMEOUT = 600.0

#: 質問の返事の待ち時間 承認と同じく、人が席を外していることもあるので長めに取る
#: 待つ間は AI のスレッドが止まるだけで、画面は動き続ける
QUESTION_TIMEOUT = APPROVAL_TIMEOUT

#: 長い仕事（書き出し）を待つ間、様子を読み直す間隔（秒） 短くすると編集画面のスレッドへ
#: 頼む仕事が増え、長くすると終わってから AI が気付くまでが延びる
WAIT_INTERVAL = 0.5


@dataclass(slots=True)
class _Work:
    """UI スレッドにやってもらう 1 件"""

    run: Callable[[], object]
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None


@dataclass(slots=True)
class Approval:
    """確認待ちの 1 件"""

    tool: str
    summary: str
    arguments: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    allowed: bool = False
    reason: str = ""
    #: 取り消せない操作なら、何をどこへするか 画面はこれがあると「以降は確認しない」を出さない
    consent: Consent | None = None

    @property
    def irreversible(self) -> bool:
        return self.consent is not None

    def allow(self) -> None:
        self.allowed = True
        self.done.set()

    def deny(self, reason: str = "ユーザーが許可しませんでした") -> None:
        self.allowed = False
        self.reason = reason
        self.done.set()


@dataclass(slots=True)
class Question:
    """返事待ちの質問 1 件（#296）

    答えるのは画面のスレッド、時間切れと中断で畳むのは AI のスレッドと中断のボタン
    先に決まった方だけを残す 時間切れの直後に押された答えで、畳んだ質問が
    「答えた」に書き換わると、AI には返事が無かったと伝えたのに画面は答えたことになる
    """

    spec: QuestionSpec
    done: threading.Event = field(default_factory=threading.Event)
    answered: bool = False
    #: 選んだ選択肢の名前（選んだ順ではなく、選択肢の並び順）
    selected: tuple[str, ...] = ()
    #: 選択肢に無い答えとして書かれた文
    free_text: str = ""
    #: 答えずに畳んだわけ（時間切れ・中断）
    reason: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reply(self, selected: tuple[str, ...] = (), free_text: str = "") -> bool:
        """答える すでに畳まれていたら何もせず偽を返す"""
        with self._lock:
            if self.done.is_set():
                return False
            order = [choice.label for choice in self.spec.choices]
            self.selected = tuple(label for label in order if label in selected)
            self.free_text = free_text.strip()
            self.answered = True
            self.done.set()
            return True

    def give_up(self, reason: str) -> bool:
        """答えずに畳む すでに答えが決まっていたら何もせず偽を返す"""
        with self._lock:
            if self.done.is_set():
                return False
            self.reason = reason
            self.done.set()
            return True

    def answer(self) -> dict[str, object]:
        """AI へ返す答え 選んだ物と書いた物を分けて渡す（混ぜると選択肢の名前と区別できない）"""
        return {"selected": list(self.selected), "free_text": self.free_text}


class EditorBridge:
    """AI からの操作を、UI スレッドで実行する"""

    def __init__(self, host: EditorHost, *, auto_approve: bool = False) -> None:
        self._host = host
        self._work: queue.Queue[_Work] = queue.Queue()
        self._approvals: queue.Queue[Approval] = queue.Queue()
        self._cancel = threading.Event()
        #: 変更系を確認なしで通すか 既定は偽
        self.auto_approve = auto_approve
        #: 「このツールは以降ずっと許可」と言われたもの
        self._always: set[str] = set()
        #: UI が取り出して、まだ返事をもらっていない確認 中断のときに畳む
        self._outstanding: Approval | None = None
        self._questions: queue.Queue[Question] = queue.Queue()
        #: UI が取り出して画面に出している質問 中断のときに畳む
        self._asked: Question | None = None

    # --- AI スレッドから ---

    def invoke(self, operation: Operation, arguments: dict[str, Any]) -> object:
        """ツールを 1 つ実行する 確認が要るものは、許可を待ってから動かす"""
        if operation.asks:
            # 質問は編集を変えないので許可は挟まない 「変更を自動で承認」でも聞く
            # （承認は操作の許可で、質問はどの操作をするかを決める物 別物として扱う）
            spec = operation(self._host, arguments)
            if not isinstance(spec, QuestionSpec):
                raise ToolError(f"{operation.name} が質問の形を返しませんでした")
            return self.ask(spec)
        if operation.irreversible:
            return self._run_irreversible(operation, arguments)
        if operation.writes and not self._approved(operation, arguments):
            raise ToolError("ユーザーがこの操作を許可しませんでした")
        return self._settle(self.call(lambda: operation(self._host, arguments)))

    def _settle(self, result: object) -> object:
        """道具が待つように頼んだ（:class:`Waiting`）なら、ここ（AI のスレッド）で待つ

        待つ間も編集画面は動く 様子は少しずつ編集画面のスレッドで読む 中断のボタンで
        待つのをやめる（書き出しそのものは止めない 止めるのは cancel_export か画面の〔中止〕）
        """
        if not isinstance(result, Waiting):
            return result
        poll = result.poll
        deadline = time.monotonic() + result.seconds
        while True:
            answer = self.call(lambda: poll(self._host))
            if not isinstance(answer, tuple) or len(answer) != 2:
                raise ToolError("待っている仕事の様子を読めませんでした")
            finished, value = answer
            remaining = deadline - time.monotonic()
            if finished or remaining <= 0:
                return value
            if self._cancel.wait(min(WAIT_INTERVAL, remaining)):
                raise ToolError(
                    "中断されたので待つのをやめました 書き出しは止めていません"
                    "（止めるのは cancel_export か画面の〔中止〕）"
                )

    def _run_irreversible(self, operation: Operation, arguments: dict[str, Any]) -> object:
        """取り消せない操作 「変更を自動で承認」も「以降は確認しない」も見ずに、毎回確かめる

        何をどこへするか（:class:`Consent`）は編集画面のスレッドで作る 書けない場所なら
        そこで断られ、窓は出ない 出してから断ると、人が許したのに何も起きない
        """
        describe = operation.consent
        if describe is None:
            # Operation の作りで弾いてある ここへ来たら、確かめずに走らせるより止める
            raise ToolError(f"{operation.name} は何をするかを示せないので実行しません")
        consent = self.call(lambda: describe(self._host, arguments))
        if not isinstance(consent, Consent):
            raise ToolError(f"{operation.name} が確かめの中身を返しませんでした")
        approval = Approval(
            tool=operation.name,
            summary=f"{consent.text()}\n{describe_call(operation, arguments)}",
            arguments=arguments,
            consent=consent,
        )
        if not self._wait_for(approval):
            raise ToolError(
                "ユーザーがこの操作を許可しませんでした（取り消せない操作は毎回確かめます）"
            )
        return self._settle(self.call(lambda: operation(self._host, arguments)))

    def call(self, work: Callable[[], object]) -> object:
        """UI スレッドで実行して、結果を返す"""
        if self._cancel.is_set():
            raise ToolError("中断されました")

        item = _Work(run=work)
        self._work.put(item)
        if not item.done.wait(CALL_TIMEOUT):
            raise ToolError("編集画面が応答しません（重い処理の最中かもしれません）")
        if item.error is not None:
            raise item.error
        return item.result

    def _approved(self, operation: Operation, arguments: dict[str, Any]) -> bool:
        if self.auto_approve or operation.name in self._always:
            return True

        approval = Approval(
            tool=operation.name,
            summary=describe_call(operation, arguments),
            arguments=arguments,
        )
        return self._wait_for(approval)

    def _wait_for(self, approval: Approval) -> bool:
        """確かめを画面へ積み、返事を待つ"""
        self._approvals.put(approval)
        if self._cancel.is_set():
            # 積んだ後に確かめる（:meth:`ask` と同じ） 積む前の中断を見落とすと 10 分待つ
            approval.deny("中断されました")
        if not approval.done.wait(APPROVAL_TIMEOUT):
            approval.deny("確認の返事がありませんでした")
        return approval.allowed

    def ask(self, spec: QuestionSpec) -> dict[str, object]:
        """画面に質問を出し、答えを待つ 答えなければ失敗にする

        失敗で返すのは、返事が無いのに推測で進めさせないため 答えの代わりに空の
        結果を返すと、AI は「どれでもよい」と読んで、聞いた意味が無くなる
        """
        question = Question(spec)
        self._questions.put(question)
        if self._cancel.is_set():
            # 積んだ後に確かめる 積む前に確かめると、確かめてから積むまでの間に中断された
            # とき、中断の側は積む前のキューを畳み終えていて、誰も畳まずに 10 分待つ
            question.give_up("中断されました")
        if not question.done.wait(QUESTION_TIMEOUT):
            minutes = round(QUESTION_TIMEOUT / 60)
            question.give_up(
                f"質問への返事がありませんでした（{minutes} 分待ちました）"
                " 推測で進めず、ここで作業を止めて、返事が無かったことを伝えてください"
            )
        if not question.answered:
            raise ToolError(question.reason)
        return question.answer()

    def cancel(self) -> None:
        """走っているツールを諦めさせる 中断ボタンから呼ぶ"""
        self._cancel.set()
        # 待っている承認は拒否として畳む 放っておくと AI スレッドが
        # 10 分間止まったままになる UI がすでに取り出して画面に出している分も
        # 対象にする 取り出した時点で手が届かなくなると、中断が効かない
        outstanding = self._outstanding
        if outstanding is not None and not outstanding.done.is_set():
            outstanding.deny("中断されました")
        self._outstanding = None
        while True:
            try:
                self._approvals.get_nowait().deny("中断されました")
            except queue.Empty:
                break
        # 質問も同じ 畳まないと、中断したのに AI のスレッドが返事を 10 分待ち続ける
        asked = self._asked
        if asked is not None:
            asked.give_up("中断されました")
        self._asked = None
        while True:
            try:
                self._questions.get_nowait().give_up("中断されました")
            except queue.Empty:
                break

    def resume(self) -> None:
        self._cancel.clear()

    # --- UI スレッドから ---

    def pump(self) -> None:
        """溜まった仕事を実行する タイマーから定期的に呼ぶ"""
        while True:
            try:
                item = self._work.get_nowait()
            except queue.Empty:
                return
            try:
                item.result = item.run()
            except (ToolError, ValueError, KeyError, OSError, RuntimeError) as exc:
                item.error = exc
            finally:
                item.done.set()

    def take_approval(self) -> Approval | None:
        """確認待ちが 1 件あれば取り出す 無ければ ``None``"""
        try:
            approval = self._approvals.get_nowait()
        except queue.Empty:
            return None
        self._outstanding = approval
        return approval

    def take_question(self) -> Question | None:
        """返事待ちの質問が 1 件あれば取り出す 無ければ ``None``"""
        try:
            question = self._questions.get_nowait()
        except queue.Empty:
            return None
        self._asked = question
        return question

    def allow_always(self, tool: str) -> None:
        """このツールは以降ずっと許可する この会話の間だけ有効

        取り消せない操作は覚えない 覚えると、1 度許した上書き保存が、次から黙って走る
        """
        operation = find_operation(tool)
        if operation is not None and operation.irreversible:
            return
        self._always.add(tool)

    def forget_always(self) -> None:
        self._always.clear()


def describe_call(operation: Operation, arguments: dict[str, Any]) -> str:
    """確認ダイアログに出す 1 行

    ツール名だけでは何が起きるか分からない 引数まで見せて、押す前に判断できる
    ようにする
    """
    if not arguments:
        return operation.description
    parts = []
    for name, value in arguments.items():
        text = str(value)
        if len(text) > 40:
            text = text[:39] + "…"
        parts.append(f"{name}={text}")
    return f"{operation.description}\n{', '.join(parts)}"
