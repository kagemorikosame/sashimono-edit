"""AI スレッドと UI スレッドの橋渡し

Claude Agent SDK は自前の asyncio ループを別スレッドで回す そこからウィジェットや
:class:`~sashimono.core.commands.Document` を直に触ると Qt が落ちるので、**やることを
キューへ積んで UI スレッドに実行させる**

同じ仕組みで確認（承認）も運ぶ 変更系のツールは、UI 側が「許可」を返すまでここで
止まる 止まっている間も UI は動き続けるので、内容を見てから決められる
確かめは二段（2026-10-14 利用者の決定 #350） 取り消せる編集は「変更を自動で承認」で
省けるが、取り消せない操作（:attr:`Operation.irreversible`）は毎回確かめる
AI からの質問（ask_user #296）も同じ形で運び、答えが返るまで止まる
外へ送る物の同意（#342 :mod:`sashimono.ai.outbound`）も同じ形で運ぶ 道具を動かす前に、
結果に入りうる物（:attr:`Operation.sends`）のうちまだ許されていない物を尋ねる

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
from sashimono.ai.operations import (
    ImageResult,
    Operation,
    Pending,
    QuestionSpec,
    find_operation,
)
from sashimono.ai.outbound import (
    ASKED,
    Outbound,
    SendAnswer,
    SendChoice,
    SendRequest,
    file_secrets,
    hide_text,
    redact,
    secret_pairs,
    text_secrets,
)
from sashimono.ai.safety import Consent

__all__ = ["Approval", "EditorBridge", "Question"]

#: 送る前の同意の返事の待ち時間 確かめと同じく、人が席を外していることもあるので長めに取る
SEND_TIMEOUT = 600.0

#: UI スレッドが 1 件の仕事を拾って返すまで待つ上限（秒）
#: これを超えるのは、書き出しなど UI 側が長く塞がっているとき
CALL_TIMEOUT = 60.0

#: 承認の待ち時間 人が席を外していることもあるので長めに取る
APPROVAL_TIMEOUT = 600.0

#: 質問の返事の待ち時間 承認と同じく、人が席を外していることもあるので長めに取る
#: 待つ間は AI のスレッドが止まるだけで、画面は動き続ける
QUESTION_TIMEOUT = APPROVAL_TIMEOUT

#: 用意を待つ道具（:class:`~sashimono.ai.operations.Pending`）が 1 回に待つ長さ（秒）
#: 中断のボタンが効くまでの遅れの上限になる
WAIT_SLICE = 0.2


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
        #: 外へ送ってよいか（設定の「AI へ送る物」） 画面が設定を読んだときに入れ替える
        #: 既定は尋ねる 画面の無い所（試験・CLI）で作ったブリッジも、黙って送らない
        self.send_choices: dict[Outbound, SendChoice] = dict.fromkeys(ASKED, SendChoice.ASK)
        #: 確かめの窓で「アプリを閉じるまで」送る・送らないと答えた物 設定には書かない
        self._session_sends: dict[Outbound, bool] = {}
        self._sends_lock = threading.Lock()
        self._send_requests: queue.Queue[SendRequest] = queue.Queue()
        #: 画面が取り出して窓に出している確かめ 中断のときに畳む
        self._sending: SendRequest | None = None
        #: これまでに見た素材・作品の名前と場所（伏せる文字 → 印） :meth:`_learn_names` が足す
        self._known_names: dict[str, str] = {}

    # --- AI スレッドから ---

    def invoke(self, operation: Operation, arguments: dict[str, Any]) -> object:
        """ツールを 1 つ実行する 確認が要るものは、許可を待ってから動かす

        外へ送る物の同意（#342）は、許可より先に取る 送らないと決まった道具の許可を
        尋ねても、許した後に断るだけになる
        """
        self._agree_to_send(operation)
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
        return self._execute(operation, arguments)

    def _execute(self, operation: Operation, arguments: dict[str, Any]) -> object:
        """道具を画面のスレッドで動かし、待つ物（:class:`Pending`）なら待ってから仕上げる

        仕上げ（``then``）の戻り値も、外へ送ってよい形にする（:meth:`_run`） 待つ前の
        戻り値だけを伏せると、待った後に返る名前や本文がそのまま送られる
        同意（:meth:`_agree_to_send`）は :meth:`invoke` の頭で取り終えている 待つ前に効く
        """
        result = self.call(lambda: self._run(operation, lambda: operation(self._host, arguments)))
        if isinstance(result, Pending):
            # 待つのはこの（AI の）スレッド 画面のスレッドは待つ間も動き続ける
            self._wait(result)
            finish = result.then
            return self.call(lambda: self._run(operation, finish))
        return result

    def _wait(self, pending: Pending) -> None:
        """用意できるか、上限の秒が過ぎるか、中断されるまで待つ

        少しずつ区切って待つのは、中断のボタンを待つ間にも効かせるため 区切らずに
        上限（最長 2 分）まで待つと、押しても止まらない
        """
        deadline = time.monotonic() + max(0.0, pending.seconds)
        while not self._cancel.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or pending.wait(min(WAIT_SLICE, remaining)):
                return

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
        return self._execute(operation, arguments)

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

    # --- 外へ送る物（#342） ---

    def may_send(self, kind: Outbound) -> bool:
        """今それを外へ送ってよいか 確かめの窓の答え（アプリを閉じるまで）が設定より先"""
        if not kind.asked:
            return True
        with self._sends_lock:
            answered = self._session_sends.get(kind)
        if answered is not None:
            return answered
        return self.send_choices.get(kind, SendChoice.ASK) is SendChoice.ALLOW

    def set_send_choices(self, choices: dict[Outbound, SendChoice]) -> None:
        """設定の「AI へ送る物」を当てる 画面のスレッドから呼ぶ

        変わった物は、確かめの窓で答えた分（アプリを閉じるまで）を忘れる 設定を
        「送る前に尋ねる」へ戻した人には、次に送るときにまた尋ねる 忘れないと、
        前に窓で「送らない」と答えた物を設定で「尋ねずに送る」にしても送らない
        """
        before = self.send_choices
        self.send_choices = dict(choices)
        with self._sends_lock:
            for kind in ASKED:
                if before.get(kind) != choices.get(kind):
                    self._session_sends.pop(kind, None)

    def _agree_to_send(self, operation: Operation) -> None:
        """道具の結果に入りうる物のうち、同意の要る物を確かめる 断られたら道具を動かさない

        素材の名前と場所は断られても動かす 結果の中を伏せ字にする（:meth:`_run`）
        """
        pending: list[Outbound] = []
        for kind in ASKED:
            if kind not in operation.sends or kind in operation.hidden or self.may_send(kind):
                continue
            with self._sends_lock:
                answered = kind in self._session_sends
            if answered or self.send_choices.get(kind) is SendChoice.DENY:
                continue
            pending.append(kind)
        if pending:
            request = SendRequest(tool=operation.name, kinds=tuple(pending))
            self._send_requests.put(request)
            if self._cancel.is_set():
                # 積んだ後に確かめる（:meth:`ask` と同じ） 積む前の中断を見落とすと 10 分待つ
                request.give_up("中断されました")
            if not request.done.wait(SEND_TIMEOUT):
                request.give_up("送ってよいかの返事がありませんでした")
            if request.answer is None:
                # 答えていないので覚えない 次に送るときにまた尋ねる
                raise ToolError(
                    f"{request.reason} 送っていません 推測で進めず、ここで作業を止めてください"
                )
            allowed = request.answer is not SendAnswer.DENY
            with self._sends_lock:
                for kind in pending:
                    self._session_sends[kind] = allowed
        refused = [
            kind
            for kind in ASKED
            if kind in operation.sends
            and kind is not Outbound.FILES
            and kind not in operation.hidden
            and not self.may_send(kind)
        ]
        if refused:
            names = "・".join(kind.label for kind in refused)
            raise ToolError(
                f"ユーザーは「{names}」を外へ送ることを許していないので、{operation.name} は"
                "使えません この道具を使わずに進めるか、要るならユーザーに 表示 → 設定… の"
                "「AI へ送る物」で変えてもらうよう伝えてください 同じ道具を何度も呼ばないでください"
            )

    def _run(self, operation: Operation, work: Callable[[], object]) -> object:
        """道具を動かし、外へ送ってよい形にして返す 編集画面のスレッドで呼ぶ

        名前と場所を送らないなら、結果とエラーの文の中を伏せ字にする 宣言していない
        道具の結果も伏せる 書き漏らしても外へは出ない
        字幕と文字を伏せて送る道具（:attr:`Operation.hidden` 取り消しと履歴）は、送ってよいと
        言われていなければ、結果の中の文字も伏せる
        """
        texts: list[str] | None = None
        hide_body = Outbound.TEXT in operation.hidden and not self.may_send(Outbound.TEXT)
        if hide_body:
            # 動かす前の文も表に入れる 取り消すと、取り消した操作が足した文は作品から消える
            texts = text_secrets(self._host.document.project)
        elif operation.labels and not self.may_send(Outbound.FILES):
            # 名前と場所だけを断った人 操作の名前の対象を形で伏せる（表は使わない）
            # 置いた後に消した素材の名前は今の作品に無く、表からは伏せられない
            texts = []
        try:
            result = self._guarded(operation, work)
        except ToolError as exc:
            if texts is None:
                raise
            raise ToolError(str(self._hide_text(str(exc), texts, hide_body))) from None
        return result if texts is None else self._hide_text(result, texts, hide_body)

    def _hide_text(self, value: object, before: list[str], body: bool) -> object:
        after = text_secrets(self._host.document.project) if body else []
        return hide_text(value, sorted({*before, *after}, key=len, reverse=True))

    def _guarded(self, operation: Operation, work: Callable[[], object]) -> object:
        """道具を動かし、名前と場所を送らないなら伏せる（:meth:`_run` の前半）"""
        # 動かす前の名前も覚える 道具が素材を消すと、動かした後の作品からは名前が拾えない
        self._learn_names()
        hide = not self.may_send(Outbound.FILES)
        try:
            result = work()
        except (ToolError, ValueError, KeyError) as exc:
            if not hide:
                raise
            # 型をそろえる 文面を伏せた後は、元の型で投げ直しても中身は同じ扱いになる
            # （サーバはどれも同じく AI へのエラーとして返す）
            raise ToolError(str(redact(str(exc), self._file_secrets()))) from None
        if isinstance(result, ImageResult):
            if Outbound.FRAMES not in operation.sends:
                # 絵は伏せられない 宣言せずに絵を返す道具は、同意を取らずに送ることになる
                raise ToolError(f"{operation.name} は絵を返すと書いていないので、送りません")
            if hide:
                return ImageResult(
                    png=result.png, caption=str(redact(result.caption, self._file_secrets()))
                )
            return result
        return redact(result, self._file_secrets()) if hide else result

    def _learn_names(self) -> None:
        """今の作品と履歴に残った作品から、名前と場所を覚える 画面のスレッドで呼ぶ

        覚えた物はアプリを閉じるまで忘れない 履歴の上限で古い段が落ちたり、別の作品を
        開いて履歴が消えたりしても、字幕起こしの記録や AI の会話の中には前の名前が残る
        名前と場所を許している人にも覚えておく 後から断ったときに、前の名前も伏せられるように
        """
        host = self._host
        document = host.document
        learned = file_secrets(
            (document.project, *document.past_projects),
            project_path=host.project_path,
            export_folder=host.export_folder,
        )
        for secret, mark in learned.items():
            self._known_names.setdefault(secret, mark)

    def _file_secrets(self) -> list[tuple[str, str]]:
        self._learn_names()
        return secret_pairs(self._known_names)

    def take_send_request(self) -> SendRequest | None:
        """送ってよいかの確かめが 1 件あれば取り出す 無ければ ``None`` 画面のスレッドから呼ぶ"""
        try:
            request = self._send_requests.get_nowait()
        except queue.Empty:
            return None
        self._sending = request
        return request

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
        # 送ってよいかの確かめも同じ 畳まないと、AI のスレッドが返事を 10 分待ち続ける
        sending = self._sending
        if sending is not None:
            sending.give_up("中断されました")
        self._sending = None
        while True:
            try:
                self._send_requests.get_nowait().give_up("中断されました")
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
