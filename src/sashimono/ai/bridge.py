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
長い作業の計画（#340 :mod:`sashimono.ai.plan`）を人が区切りで止めた後は、計画の道具の
ほかは動かさずに断る
使う量の上限（#344 :mod:`sashimono.ai.budget`）も、道具を動かす前に確かめる 当たったら
設定に従い、続けるかを尋ねるか、尋ねずに断る

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
from functools import partial
from typing import Any

from sashimono.ai.budget import LimitAction, Overrun, UsageMeter
from sashimono.ai.host import EditorHost, ToolError
from sashimono.ai.operations import (
    Continue,
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
from sashimono.ai.plan import WorkPlan, plan_for
from sashimono.ai.safety import Consent

__all__ = ["Approval", "BudgetRequest", "EditorBridge", "Question"]

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

#: 上限に当たったときの「続けますか」の返事の待ち時間 返事が無ければ止める
#: 席を外している間に、上限を越えて使い続けないように
BUDGET_TIMEOUT = APPROVAL_TIMEOUT

#: 止めると決まった後に、道具を何回断ったら会話そのものを止めるか 断った訳は AI へ返して
#: あるので、ふつうは 1 回でまとめに移る それでも道具を呼び続けると、呼ぶたびに API を使う
HALT_AFTER_REFUSALS = 2

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


@dataclass(slots=True)
class BudgetRequest:
    """上限に当たった 続けるかの返事待ち 1 件（#344）

    答えるのは画面のスレッド、時間切れと中断で畳むのは AI のスレッドと中断のボタン
    先に決まった方だけを残す（:class:`Question` と同じ）
    """

    overrun: Overrun
    done: threading.Event = field(default_factory=threading.Event)
    #: 続けると答えたか 畳んだときと止めると答えたときは偽
    proceed: bool = False
    #: 答えずに畳んだわけ（時間切れ・中断）
    reason: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reply(self, proceed: bool) -> bool:
        """答える すでに畳まれていたら何もせず偽を返す"""
        with self._lock:
            if self.done.is_set():
                return False
            self.proceed = proceed
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


class EditorBridge:
    """AI からの操作を、UI スレッドで実行する"""

    def __init__(
        self,
        host: EditorHost,
        *,
        auto_approve: bool = False,
        usage: UsageMeter | None = None,
    ) -> None:
        self._host = host
        #: 開いている作品の計画（#340） 文書は作品を開き直しても同じ物なので、作ったときに
        #: 引いておけば AI のスレッドから編集画面を通さずに止めたかを見られる
        self.plan: WorkPlan = plan_for(host.document)
        #: 画面のスレッドが 1 件を拾うのを待つ上限（秒） ``None`` は :data:`CALL_TIMEOUT`
        #: 本物の窓を並べて走らせる試験だけが長くする（画面のスレッドが他の試験の描画と
        #: GPU を取り合って止まっても、試験の結果は変わらないため）
        self.call_timeout: float | None = None
        #: 使った量と上限（#344） 画面は 1 日の量を設定の置き場へ残す物を渡す
        #: 渡さなければ覚えない物（試験・CLI） 上限は既定の値で効く
        self.usage = usage if usage is not None else UsageMeter()
        self._budget_requests: queue.Queue[BudgetRequest] = queue.Queue()
        #: 画面が取り出して出している「続けますか」 中断のときに畳む
        self._budget_asking: BudgetRequest | None = None
        #: 尋ねずに止めたときの知らせ（設定が「尋ねずに止める」） 画面が拾って会話の欄に出す
        self._budget_notices: queue.Queue[Overrun] = queue.Queue()
        #: 止めると決まった後に断った道具の数 :data:`HALT_AFTER_REFUSALS` で会話を止めさせる
        self._refused_after_stop = 0
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
        区切りで止めた後（#340）は、同意も許可も尋ねる前に断る 止めた人に、止めた後の
        道具の確かめの窓を出さない
        使う量の上限（#344）は区切りの後、同意の前に見る 止めた計画の道具に「上限を広げて
        続けますか」を尋ねると、広げた後に区切りで断るだけになる
        """
        paused = self.plan.refusal(operation.name)
        if paused:
            raise ToolError(paused)
        # 量の上限は同意より先に見る 上限で止まる道具の同意を尋ねても、答えた後に断るだけになる
        self._keep_within_budget(operation)
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

    def _execute(
        self,
        operation: Operation,
        arguments: dict[str, Any],
        body: Callable[[], object] | None = None,
    ) -> object:
        """道具を編集画面のスレッドで動かし、続きがあれば終わりまで辿る

        道具の続きは 2 通りあり、どちらもここで辿る（取り消せない操作も同じ道を通る）

        - :class:`Continue` 描き続ける間に画面へ手を返す（何コマも描く道具 #320）
          続きを 1 つずつ別の仕事として積むので、その間に画面は描き直しと人の操作を受け付ける
        - :class:`Pending` 用意ができるまで、この（AI の）スレッドで待つ（場面の解析 #321）
          待つ間も画面は動き続ける

        どちらも 1 回ごとに :meth:`call` を通すので、中断が回ごとに効き、待つ上限
        （:data:`CALL_TIMEOUT`）も 1 回ずつに掛かる 続きの中からもう一方の続きを返してもよい
        続きの戻り値も 1 回ずつ外へ送ってよい形にする（:meth:`_run`） 初めの戻り値だけを
        伏せると、待った後や描き終えた後に返る名前・本文・絵がそのまま送られる
        同意（:meth:`_agree_to_send`）は :meth:`invoke` の頭で取り終えている 続きの前に効く
        ``body`` を渡すと道具の代わりにそれを動かす（取り消せない操作で、許した中身と照らす #327）
        場面の解析（#321）と書き出し（#327）は同じ :class:`Pending` で待つ
        """
        work = body if body is not None else partial(operation.start, self._host, arguments)
        result = self.call(partial(self._run, operation, work))
        while True:
            if isinstance(result, Pending):
                # 待つのはこの（AI の）スレッド 画面のスレッドは待つ間も動き続ける
                self._wait(result)
                result = self.call(partial(self._run, operation, result.then))
            elif isinstance(result, Continue):
                result = self.call(partial(self._run, operation, result.step))
            else:
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

        def run_where_approved() -> object:
            # 確かめの箱はモードレス 出ている間に人が〔名前を付けて保存〕などで作品の場所を
            # 変えると、道具が書く先を解き直したとき、箱に出して許された場所と別のファイルに
            # なる（上書き保存なら移した先の作品を潰す） 許した中身を実行まで固定し、
            # 作り直した中身と照らして違えば断る 照らすのと実行を同じ 1 件の仕事にして、
            # 間に画面の操作が挟まらないようにする
            # 照らすのは箱に見せた中身すべて（何を・どこへ・添えた注意）と、その元の身元
            # 場所が同じでも、箱が出ている間に同じ名前のファイルができる・描くコマや書き出す
            # 範囲が変わると、許していない物を書く
            again = describe(self._host, arguments)
            if again != consent:
                if isinstance(again, Consent) and again.target != consent.target:
                    what = (
                        f"書き込み先が変わりました（許した場所: {consent.target} "
                        f"今の場所: {again.target}）"
                    )
                elif isinstance(again, Consent):
                    what = f"中身が変わりました（許した時: {consent.detail} 今: {again.detail}）"
                else:
                    what = "中身が変わりました"
                raise ToolError(
                    f"確かめた後に{what} 何も書いていません 今の中身でもう 1 度頼み直してください"
                )
            # 書く処理には人が許した確かめの中身を渡す 書く先の身元はそこから取り、採り直さない
            # 採り直すと、上の照らし合わせの後に書き換えられた中身を新しい基準として受け入れる
            approved = operation.approved
            if approved is not None:
                return approved(self._host, arguments, consent)
            # 続き（Continue）は :meth:`_execute` が 1 回ずつ辿る ここで辿り切らない
            return operation.start(self._host, arguments)

        return self._execute(operation, arguments, run_where_approved)

    def call(self, work: Callable[[], object]) -> object:
        """UI スレッドで実行して、結果を返す"""
        if self._cancel.is_set():
            raise ToolError("中断されました")

        item = _Work(run=work)
        self._work.put(item)
        timeout = CALL_TIMEOUT if self.call_timeout is None else self.call_timeout
        if not item.done.wait(timeout):
            raise ToolError("編集画面が応答しません（重い処理の最中かもしれません）")
        if item.error is not None:
            raise item.error
        return item.result

    # --- 使う量の上限（#344） ---

    def _keep_within_budget(self, operation: Operation) -> None:
        """上限に当たっていれば、設定に従って尋ねるか断る 続けてよければ戻る

        絵を返す道具は、これから送る 1 枚を足して確かめる 送ってから数えると、上限の 1 枚
        先まで描いて送ってしまう
        """
        adding = 1 if Outbound.FRAMES in operation.sends else 0
        while (overrun := self.usage.overrun(adding_images=adding)) is not None:
            if self.usage.stopped is not None:
                self._refused_after_stop += 1
                raise ToolError(_stop_message(overrun))
            if self.usage.limits.action is LimitAction.STOP:
                self._stop(overrun)
                self._budget_notices.put(overrun)
                raise ToolError(_stop_message(overrun))
            request = BudgetRequest(overrun)
            self._budget_requests.put(request)
            if self._cancel.is_set():
                # 積んだ後に確かめる（:meth:`ask` と同じ） 積む前の中断を見落とすと 10 分待つ
                request.give_up("中断されました")
            if not request.done.wait(BUDGET_TIMEOUT):
                request.give_up("続けるかの返事がありませんでした")
            if not request.proceed:
                self._stop(overrun)
                reason = f"{request.reason} " if request.reason else ""
                raise ToolError(reason + _stop_message(overrun))
            self.usage.extend(overrun.kind)

    def _stop(self, overrun: Overrun) -> None:
        self.usage.stop(overrun)
        self._refused_after_stop = 1

    @property
    def wants_halt(self) -> bool:
        """止めると決まった後も道具を呼び続けている 画面は会話を止める（中断と同じ）"""
        return self._refused_after_stop >= HALT_AFTER_REFUSALS

    def begin_turn(self) -> None:
        """指示に取りかかった 量を数え直す 画面のスレッドから呼ぶ"""
        self._refused_after_stop = 0
        self.usage.begin_turn()

    def end_turn(self) -> None:
        """指示が終わった 量を 1 日の分へ足す 画面のスレッドから呼ぶ"""
        self._refused_after_stop = 0
        self.usage.end_turn()

    def take_budget_request(self) -> BudgetRequest | None:
        """「続けますか」が 1 件あれば取り出す 無ければ ``None`` 画面のスレッドから呼ぶ"""
        try:
            request = self._budget_requests.get_nowait()
        except queue.Empty:
            return None
        self._budget_asking = request
        return request

    def take_budget_notice(self) -> Overrun | None:
        """尋ねずに止めた知らせが 1 件あれば取り出す 無ければ ``None``"""
        try:
            return self._budget_notices.get_nowait()
        except queue.Empty:
            return None

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
            # 送る絵として数える（#344） 描けなかった・断った絵は送らないので数えない
            self.usage.record_image()
            if hide:
                return ImageResult(
                    png=result.png, caption=str(redact(result.caption, self._file_secrets()))
                )
            return result
        return redact(result, self._file_secrets(), operation.name_keys) if hide else result

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
            media=document.seen_media,
            names=document.seen_project_names,
            remembered=host.remembered_names(),
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
        # 上限の「続けますか」も同じ 畳まないと、AI のスレッドが返事を 10 分待ち続ける
        asking = self._budget_asking
        if asking is not None:
            asking.give_up("中断されました")
        self._budget_asking = None
        while True:
            try:
                self._budget_requests.get_nowait().give_up("中断されました")
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
            if self._cancel.is_set():
                # 積んだ時は中断の前でも、拾う時に中断されていれば動かさない 動かすと、
                # 描きかけの続き（最後の 1 コマと PNG）が中断の後に描き上がり、AI へ返る
                item.error = ToolError("中断されました")
                item.done.set()
                continue
            try:
                item.result = item.run()
            except (ToolError, ValueError, KeyError, OSError, RuntimeError) as exc:
                item.error = exc
            finally:
                item.done.set()
            if isinstance(item.result, Continue):
                # 続きのある道具は、ここで一度タイマーへ返す 続きを同じ回で拾うと、
                # 全部を描き終えるまで画面が描き直されず、分けた意味が無くなる
                return

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


def _stop_message(overrun: Overrun) -> str:
    """上限で止めたときに AI へ返す文 まとめて終えるよう頼む（道具を呼び直させない）"""
    return (
        f"{overrun.text()} ユーザーが設定した上限なので、ここで作業を止めます"
        " これ以上道具を使わず、ここまでにしたことと残っていることを短く伝えて終えてください"
        "（続けるかはユーザーが決めます 上限は 表示 → 設定… の「AI の上限」で変えられます）"
    )


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
