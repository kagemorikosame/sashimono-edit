"""AI の長い作業の計画と進み具合（#340 D2）

AI は何手もかかる頼みを受けると、始める前に目標と手順を書き（``set_plan``）、手順に
取りかかる時と終えた時に進み具合を書く（``update_plan_step``） 画面（AI のパネルの
計画の欄）はそれを見せ、人は〔区切りで止める〕で今の手順を終えた所で止め、〔続ける〕で
止めた所から再開を頼める 止めた所は ``get_plan`` で AI が読む 会話を作り直した後
（モデルを選び直した・置き方の方式を変えた）でも、計画から続きを始められる

計画は開いている作品に付けて、アプリを閉じるまで持つ（作品のファイルにも設定の置き場にも
書かない） 別の作品を開くと閉じる 前の作品の計画を次の作品で続けると、別の作品の
クリップを手順どおりに切り詰めかねない

ここは Qt を知らない 道具（編集画面のスレッド）・ブリッジの止め所（AI のスレッド）・
画面（編集画面のスレッド）から触るので、中身は錠で守る
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sashimono.ai.handoff import GET_HANDOFF, StopReason
from sashimono.ai.host import ToolError
from sashimono.core.commands import Document

__all__ = [
    "GET_PLAN",
    "GOAL_LIMIT",
    "INSTRUCTION_LIMIT",
    "NOTE_LIMIT",
    "PASSES_AFTER_PAUSE",
    "PAUSED_MESSAGE",
    "PLAN_TOOLS",
    "RESUME_PROMPT",
    "SET_PLAN",
    "STEPS_RANGE",
    "STEP_LIMIT",
    "UPDATE_PLAN_STEP",
    "PlanStep",
    "PlanView",
    "StepState",
    "Stopped",
    "WorkPlan",
    "plan_for",
]

SET_PLAN = "set_plan"
UPDATE_PLAN_STEP = "update_plan_step"
GET_PLAN = "get_plan"
#: 計画の道具 設定で計画を切った人の会話には渡さない 止めた後も計画の道具だけは通す
#: （止めた所を書き終え、読めるように）
PLAN_TOOLS = frozenset({SET_PLAN, UPDATE_PLAN_STEP, GET_PLAN})
#: 区切りで止めた後も通す道具 計画の道具と、止まった指示の引き継ぎを読む道具（#346）
#: 引き継ぎは読むだけで作品を変えず、続きを頼む前に前の指示で何が残ったかを知るのに要る
#: 断ると、止めた後の AI が引き継ぎを確かめられず、残りを言葉だけで伝えることになる
PASSES_AFTER_PAUSE = PLAN_TOOLS | {GET_HANDOFF}

#: 手順の数 30 を超えると欄で追えない 超えるなら手順をまとめさせる 下限は 1 にしておく
#: 続きの指示の中で、残り 1 手順の計画を作り直せるように 2 に上げると、残りが 1 つの続きで
#: set_plan が断られ、作り直した計画が欄に出ない
STEPS_RANGE = (1, 30)
#: 長さの上限（文字） 狭いパネル（300 画素ほど）で目標が 2 行、手順が 1〜2 行に収まる長さ
#: 超えたら切らずに断る 切ると手順の違いを書いた所が消え、どこまで済んだか読めなくなる
GOAL_LIMIT = 200
STEP_LIMIT = 80
NOTE_LIMIT = 120
#: get_plan で返す指示の長さ 手順 30 個と合わせても道具の返事 1 回（12000 字）に収まる
INSTRUCTION_LIMIT = 1000

#: 〔続ける〕で送る指示 計画を作り直したときに、元の指示をこれで上書きしない目印にもする
RESUME_PROMPT = "計画の続きから進めてください（get_plan で済んだ手順と止めた所を確かめてから）"

#: 区切りで止めた後に、道具を断るときの文 AI が次に何をすればよいかを書く
PAUSED_MESSAGE = (
    "ユーザーが計画を区切りで止めました これ以上道具を使わず、済んだ手順と残りの手順を"
    "短く伝えて終えてください 続きはユーザーが〔続ける〕で頼みます"
)


class StepState(Enum):
    """手順 1 つの進み具合"""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        """もう取りかからない手順か できなかった手順も、続きの対象から外す"""
        return self in (StepState.DONE, StepState.SKIPPED, StepState.FAILED)

    @property
    def label(self) -> str:
        return _STATE_LABELS[self]

    @property
    def mark(self) -> str:
        """欄の行の頭の印 色だけに頼らず、印の形で進み具合が読める"""
        return _STATE_MARKS[self]


_STATE_LABELS = {
    StepState.PENDING: "まだ",
    StepState.RUNNING: "取りかかり中",
    StepState.DONE: "済み",
    StepState.SKIPPED: "飛ばした",
    StepState.FAILED: "できなかった",
}
_STATE_MARKS = {
    StepState.PENDING: "・",
    StepState.RUNNING: "▸",
    StepState.DONE: "✓",
    StepState.SKIPPED: "－",
    StepState.FAILED: "×",
}


class Stopped(Enum):
    """手順が残ったまま止まった訳 欄と get_plan に出す

    止まった指示の引き継ぎ（#346 :class:`~sashimono.ai.handoff.StopReason`）の訳のどれにも
    当たる物を持つ（:meth:`WorkPlan.end_turn` が引き継ぎの訳から決める） 上限だけは、
    次にすることが違うので使う量・Claude Code・直しの回数の 3 つに分ける 引き継ぎの訳が
    無い（止まらずに終えた）ときだけ、区切りと応え終えたを使う
    """

    #: 〔区切りで止める〕 今の手順を終えた所で止まった
    PAUSED = "区切りで止めました"
    #: 〔中断〕 手順の途中で止まったかもしれない
    INTERRUPTED = "中断しました"
    #: 使う量の上限（#344）で止めた 手順の途中で止まったかもしれない
    #: 応え終えた・中断と書くと、上限を広げれば続けられることが読めない
    LIMIT = "使う量の上限で止めました"
    #: 検査の直しの回数の上限（#339）に達し、ブリッジが残りの編集を断った 手順の途中で
    #: 止まったかもしれない 応え終えたと書くと、設定で回数を増やせば続けられることが読めない
    FIX_LIMIT = "検査の直しの回数の上限で止めました"
    #: Claude Code の上限（手数・費用）で止まった 使う量の上限と分ける 設定の「AI の上限」を
    #: 広げても変わらず、送り直せば続く
    AGENT_LIMIT = "Claude Code の上限で止まりました"
    #: 応えている途中で Claude Code との会話が閉じた 応え終えたと書くと、途中の手順を
    #: 確かめずに続ける
    DISCONNECTED = "接続が切れて止まりました"
    #: エラーで止まった（例外・Claude Code の失敗・部品が古い）
    ERROR = "エラーで止まりました"
    #: 許されなかった操作・答えなかった質問・送るのを断った物があった
    DENIED = "許されなかった操作があって止まりました"
    #: 頼まれずに応え終えた（質問した・できずに諦めたなど）
    ENDED = "手順を残して応え終えました"


#: 引き継ぎの訳から計画の訳へ 上限（``LIMIT``）は 3 つに分けるので
#: :meth:`WorkPlan._stopped_for` で決める
_FROM_HANDOFF = {
    StopReason.INTERRUPTED: Stopped.INTERRUPTED,
    StopReason.DISCONNECTED: Stopped.DISCONNECTED,
    StopReason.ERROR: Stopped.ERROR,
    StopReason.DENIED: Stopped.DENIED,
}


@dataclass(frozen=True, slots=True)
class PlanStep:
    title: str
    state: StepState = StepState.PENDING
    note: str = ""


@dataclass(frozen=True, slots=True)
class PlanView:
    """画面に見せる計画の写し 錠の外で読めるように、その時の中身を写して渡す"""

    goal: str
    steps: tuple[PlanStep, ...]
    #: 〔区切りで止める〕を押した 区切りに来るまで今の手順は続く
    pause_requested: bool
    stopped: Stopped | None
    #: 中身が変わるたびに増える 画面は変わったときだけ描き直す
    version: int

    @property
    def done(self) -> int:
        return sum(1 for step in self.steps if step.state.finished)

    @property
    def finished(self) -> bool:
        return all(step.state.finished for step in self.steps)

    @property
    def running(self) -> int | None:
        """取りかかっている手順の番号（1 から） 無ければ ``None``"""
        return _first(self.steps, StepState.RUNNING)

    @property
    def next_step(self) -> int | None:
        """次に取りかかる手順の番号（1 から） 無ければ ``None``"""
        return _first(self.steps, StepState.PENDING)


def _first(steps: tuple[PlanStep, ...], state: StepState) -> int | None:
    for number, step in enumerate(steps, start=1):
        if step.state is state:
            return number
    return None


class WorkPlan:
    """開いている作品の、AI の長い作業の計画 1 つ

    作り直すと前の計画を捨てる 長い作業は 1 つずつ進めるので、計画も 1 つで足りる
    """

    def __init__(self, generation: int = 0) -> None:
        self._lock = threading.Lock()
        self._goal = ""
        self._steps: list[PlanStep] = []
        #: 計画を作ったときの人の指示 会話を作り直した後の続きで、元の頼みを読めるように
        self._instruction = ""
        #: 今の指示 計画を作るときに :attr:`_instruction` へ写す
        self._turn_instruction = ""
        self._pause_requested = False
        #: 区切りに来て止めた 止めた後は計画の道具しか通さない
        self._paused = False
        self._stopped: Stopped | None = None
        self._version = 0
        #: 計画を付けた作品（:attr:`Document.generation`）
        self._generation = generation

    # --- AI の道具から（編集画面のスレッド） ---

    def make(self, goal: object, titles: object) -> dict[str, Any]:
        """計画を作る 前の計画は捨てる 止めた後は作らせない（作り直しで止めを外せないように）"""
        text = _text(goal, "goal", GOAL_LIMIT)
        if not text:
            raise ToolError("goal（目標）が空です 何を仕上げる計画かを 1 文で書いてください")
        low, high = STEPS_RANGE
        if not isinstance(titles, list) or not low <= len(titles) <= high:
            raise ToolError(
                f"steps（手順）は {low}〜{high} 個の文字列の配列にしてください"
                " 多すぎるときは手順をまとめます"
            )
        steps = [PlanStep(_step_title(title)) for title in titles]
        with self._lock:
            if self._paused:
                raise ToolError(PAUSED_MESSAGE)
            left = sum(1 for step in self._steps if not step.state.finished)
            self._goal = text
            self._steps = steps
            self._instruction = self._turn_instruction
            self._stopped = None
            self._version += 1
        result: dict[str, Any] = {
            "steps": len(steps),
            "next_step": 1,
            "note": (
                "計画を画面に出しました 手順に取りかかる時と終えた時に update_plan_step で"
                "進み具合を書いてください"
            ),
        }
        if left:
            # 黙って捨てると、前の計画の残りをやり終えたつもりで忘れる
            result["replaced_unfinished"] = left
        return result

    def mark(self, number: object, state: object, note: object = "") -> dict[str, Any]:
        """手順の進み具合を書く 区切りで止めると頼まれていれば、ここで止める"""
        if isinstance(number, bool) or not isinstance(number, int):
            raise ToolError("step は手順の番号（1 から数える整数）で渡してください")
        try:
            wanted = StepState(state)
        except ValueError:
            names = " / ".join(item.value for item in StepState)
            raise ToolError(f"state は {names} のどれかにしてください（{state!r}）") from None
        remark = _text(note, "note", NOTE_LIMIT)
        with self._lock:
            if not self._steps:
                raise ToolError("計画がありません 先に set_plan で目標と手順を書いてください")
            if not 1 <= number <= len(self._steps):
                raise ToolError(f"step は 1〜{len(self._steps)} にしてください（{number}）")
            if self._pause_requested and not self._paused:
                # 区切りに来た 取りかかる前（running）なら手順を始めずに止める 終えた印は
                # 書いてから止める（書かずに止めると、済んだ手順を続きでやり直す）
                self._paused = True
                self._stopped = Stopped.PAUSED
                self._version += 1
            if not (self._paused and wanted is StepState.RUNNING):
                step = self._steps[number - 1]
                self._steps[number - 1] = PlanStep(step.title, wanted, remark)
                self._version += 1
            if all(item.state.finished for item in self._steps):
                # 最後の手順の途中で止めを頼まれ、そのまま終えた 止める所が残っていないので
                # 止めを外す 残すと、終えた計画に「区切りで止めました」と押せない〔続ける〕が出る
                self._pause_requested = False
                self._paused = False
                self._stopped = None
            view = self._view()
            paused = self._paused
        result: dict[str, Any] = {
            "step": number,
            "state": view.steps[number - 1].state.value,
            "done": view.done,
            "total": len(view.steps),
            "next_step": view.next_step,
            "finished": view.finished,
        }
        if paused:
            result["paused"] = True
            result["message"] = PAUSED_MESSAGE
        return result

    def read(self) -> dict[str, Any]:
        """AI が読む計画 止めた所から続けるのに使う"""
        with self._lock:
            if not self._steps:
                return {"plan": None, "note": "計画はありません"}
            view = self._view()
            instruction = self._instruction
        if len(instruction) > INSTRUCTION_LIMIT:
            instruction = instruction[: INSTRUCTION_LIMIT - 1] + "…"
        return {
            "goal": view.goal,
            "instruction": instruction,
            "steps": [
                {"step": number, "title": step.title, "state": step.state.value, "note": step.note}
                for number, step in enumerate(view.steps, start=1)
            ],
            "done": view.done,
            "total": len(view.steps),
            "running": view.running,
            "next_step": view.next_step,
            "finished": view.finished,
            "stopped": view.stopped.value if view.stopped is not None else None,
        }

    # --- ブリッジから（AI のスレッド） ---

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def refusal(self, tool: str) -> str:
        """区切りで止めた後に断る文 断らない道具なら空

        計画の道具は通す 止めた後に済んだ印を書き終え、止めた所を読めるように
        引き継ぎを読む道具（#346）も通す（:data:`PASSES_AFTER_PAUSE`）
        """
        if tool in PASSES_AFTER_PAUSE or not self.paused:
            return ""
        return PAUSED_MESSAGE

    # --- 画面から（編集画面のスレッド） ---

    def view(self) -> PlanView | None:
        """今の計画の写し 無ければ ``None``"""
        with self._lock:
            return self._view() if self._steps else None

    def _view(self) -> PlanView:
        return PlanView(
            goal=self._goal,
            steps=tuple(self._steps),
            pause_requested=self._pause_requested,
            stopped=self._stopped,
            version=self._version,
        )

    def begin_turn(self, instruction: str) -> None:
        """人の指示に取りかかった 前に止めたのを外す（新しい指示も〔続ける〕も同じ）

        続きを頼んだ指示（:data:`RESUME_PROMPT`）は、計画を作った元の指示として覚えない
        続きの中で計画を作り直したときに、元の頼みが「続けて」で上書きされる
        """
        with self._lock:
            if instruction != RESUME_PROMPT:
                self._turn_instruction = instruction
            if self._pause_requested or self._paused or self._stopped is not None:
                self._version += 1
            self._pause_requested = False
            self._paused = False
            self._stopped = None

    def end_turn(
        self,
        *,
        reason: StopReason | None = None,
        usage_limited: bool = False,
        agent_limited: bool = False,
        fixed: bool = False,
    ) -> None:
        """指示が終わった 手順が残っていれば、止まった訳を覚える

        ``reason`` は引き継ぎ（#346）に書いた止まった訳 画面は同じ訳を両方へ渡す 別々に決めると、
        計画の欄は「応え終えた」なのに引き継ぎは「エラー」のように食い違い、どちらを信じて
        続ければよいか分からない 訳があれば計画もその訳にし、無いときだけ区切りか応え終えた
        （直しの回数の上限で引き継ぎが無いときは、画面がこの後に引き継ぎへ上限の訳を書く）
        上限の訳（``LIMIT``）は ``usage_limited``（使う量の上限 #344）・``agent_limited``
        （Claude Code の上限）・``fixed``（検査の直しの回数の上限 #339）で分ける 次にすることが違う
        ``fixed`` は区切りまで来て止めた後なら、ブリッジは区切りの方を先に断るので、引き継ぎの訳が
        無ければ訳は区切り まだ区切りに来ていなければ、残りの編集を断ったのは回数の上限
        """
        if reason is None and (usage_limited or agent_limited):
            reason = StopReason.LIMIT
        with self._lock:
            unfinished = any(not step.state.finished for step in self._steps)
            if unfinished:
                self._stopped = self._stopped_for(reason, usage_limited, agent_limited, fixed)
            else:
                # 終えた計画に止まった訳を残さない 残すと会話の欄が「区切りで止めました」と出す
                self._stopped = None
            self._pause_requested = False
            self._paused = False
            self._version += 1

    def _stopped_for(
        self,
        reason: StopReason | None,
        usage_limited: bool,
        agent_limited: bool,
        fixed: bool,
    ) -> Stopped:
        """止まった訳を引き継ぎの訳から決める 錠の中で呼ぶ"""
        if reason is StopReason.LIMIT:
            if usage_limited:
                return Stopped.LIMIT
            if agent_limited:
                return Stopped.AGENT_LIMIT
            return Stopped.FIX_LIMIT if fixed else Stopped.LIMIT
        if reason is not None:
            return _FROM_HANDOFF[reason]
        if self._paused:
            return Stopped.PAUSED
        if fixed:
            return Stopped.FIX_LIMIT
        if self._pause_requested:
            return Stopped.PAUSED
        return Stopped.ENDED

    def request_pause(self) -> bool:
        """〔区切りで止める〕 残りの手順が無い・もう頼んであるときは偽"""
        with self._lock:
            if self._pause_requested or self._paused:
                return False
            if all(step.state.finished for step in self._steps):
                return False
            self._pause_requested = True
            self._version += 1
            return True

    def discard(self) -> None:
        """計画を閉じる（人の〔×〕・別の作品を開いた）"""
        with self._lock:
            self._goal = ""
            self._steps = []
            self._instruction = ""
            self._pause_requested = False
            self._paused = False
            self._stopped = None
            self._version += 1

    def follow(self, generation: int) -> bool:
        """開いている作品が変わっていれば計画を閉じる 閉じた（計画があった）なら真"""
        with self._lock:
            if generation == self._generation:
                return False
            self._generation = generation
            had = bool(self._steps)
        self.discard()
        return had


#: 文書ごとの計画 道具はホスト（編集画面）しか受け取らないので、文書から引く
#: ホストに口を足さないのは、ホストの口がアプリの設定を変えないことを試験が見張っていて
#: （tests/ai/test_safety.py）、計画は作品にもアプリの設定にも触らない物だから
#: 文書が消えたら計画も消えるよう、弱い参照で持つ
_PLANS: weakref.WeakKeyDictionary[Document, WorkPlan] = weakref.WeakKeyDictionary()
_PLANS_LOCK = threading.Lock()


def plan_for(document: Document) -> WorkPlan:
    """``document`` の計画 開いている作品が変わっていれば、前の計画は閉じてある"""
    with _PLANS_LOCK:
        plan = _PLANS.get(document)
        if plan is None:
            plan = WorkPlan(document.generation)
            _PLANS[document] = plan
    plan.follow(document.generation)
    return plan


def _text(value: object, name: str, limit: int) -> str:
    """文字列の引数 省いたら空 長すぎれば切らずに断る（ask_user と同じ）"""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ToolError(f"{name} は文字列で渡してください（{type(value).__name__} が来ました）")
    text = value.strip()
    if len(text) > limit:
        raise ToolError(f"{name} は {limit} 文字までにしてください（{len(text)} 文字）")
    return text


def _step_title(value: object) -> str:
    title = _text(value, "steps の各項目", STEP_LIMIT)
    if not title:
        raise ToolError("steps に空の手順があります 手順ごとに何をするかを短く書いてください")
    return title
