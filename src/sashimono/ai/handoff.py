"""AI が途中で止まったときの引き継ぎの要約（#346 D8）

AI の指示はいろいろな所で止まる 例外・人の中断・使う量や手数の上限・許されなかった操作・
Claude Code との接続が切れた どこで止まっても、人が次に何をすればよいかを読めるように、
何をやって・どこで止まったかを 1 つの形（:class:`Handoff`）にまとめる

中身は AI の言葉に頼らない 変えた物は作品の履歴（取り消しの段の中の小さな段）から、
止まった訳と許されなかった物は画面が見た出来事から組む AI の最後の言葉も添えるが、
AI の言葉だと分かる欄に分けて置く AI が「全部終えました」と言ったまま止まっても、
履歴から組んだ欄は嘘をつかない

集めるのは :class:`TurnRecord`（画面が 1 つの指示の間に書く） 止まった指示の要約は
文書ごとの棚（:func:`shelf_for`）に置き、AI は ``get_handoff`` の道具で読む 道具を通すので、
外へ送る物の宣言と同意（#342）にそのまま乗る（素材の名前と字幕の文は許すまで伏せる）

止まり方を足すときは :class:`StopReason` に足し、止めた所で :meth:`TurnRecord.stop` を呼ぶ
（使う量の上限 #344・計画を区切りで止めた #340 など） 残った手順や人へ渡す問題（#339）は
:meth:`TurnRecord.remain` で足す 要約の形を所ごとに作らない

ここは Qt を知らない 画面（編集画面のスレッド）と道具（同じく編集画面のスレッド）から触る
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sashimono.core.commands import Document
from sashimono.core.model import Project

__all__ = [
    "AI_WORDS_LIMIT",
    "CONTINUE_PROMPT",
    "GET_HANDOFF",
    "ITEMS_LIMIT",
    "Handoff",
    "HandoffShelf",
    "StopReason",
    "TurnRecord",
    "shelf_for",
]

GET_HANDOFF = "get_handoff"

#: 〔続きを頼む〕で入力欄に入れる指示 要約の中身は入れない 中身は道具で読ませ、外へ送る物の
#: 宣言と同意（#342）を通す 入力欄へ写すと、伏せるはずの素材の名前が指示の文として出る
CONTINUE_PROMPT = (
    "前の指示は途中で止まりました get_handoff で引き継ぎの要約を読み、"
    "変えた物を確かめてから、残りを進めてください"
)

#: 1 つの欄に並べる数 長い指示で失敗が何十件も続くと、要約が読めない長さになる
#: 新しい方から残す（止まる直前の物ほど、どこで止まったかを言う）
ITEMS_LIMIT = 8
#: AI の最後の言葉の長さ 長い返事を丸ごと載せると、履歴から組んだ欄が埋もれる
AI_WORDS_LIMIT = 400
#: 指示の長さ 要約の頭に載せる 長い指示は道具の返事 1 回に収まる長さで切る
INSTRUCTION_LIMIT = 400


class StopReason(Enum):
    """指示が止まった訳 値は道具の返事に載せる名前"""

    #: 人が〔中断〕を押した
    INTERRUPTED = "interrupted"
    #: 使う量・手数の上限に当たった（Claude Code の上限・使う量の上限 #344）
    LIMIT = "limit"
    #: 応えている途中で Claude Code との会話が閉じた
    DISCONNECTED = "disconnected"
    #: エラーで止まった（例外・Claude Code の失敗・部品が古い）
    ERROR = "error"
    #: 許されなかった操作・答えなかった質問・送るのを断った物がある
    DENIED = "denied"

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def rank(self) -> int:
        """重なったときに勝つ順（小さいほど強い）

        人の中断は一番強い 中断の後に来るエラー（「中断されました」）を訳にすると、
        人が止めたのに AI が失敗したように読める 上限で止めた後の接続の切れも同じ
        許されなかった物は、ほかの訳が無いときだけ訳にする（ほかの訳のときも欄には残す）
        """
        return _ORDER.index(self)


_LABELS = {
    StopReason.INTERRUPTED: "中断しました",
    StopReason.LIMIT: "上限で止まりました",
    StopReason.DISCONNECTED: "接続が切れて止まりました",
    StopReason.ERROR: "エラーで止まりました",
    StopReason.DENIED: "許されなかった操作があります",
}
_ORDER = (
    StopReason.INTERRUPTED,
    StopReason.LIMIT,
    StopReason.DISCONNECTED,
    StopReason.ERROR,
    StopReason.DENIED,
)


@dataclass(frozen=True, slots=True)
class Handoff:
    """止まった指示 1 つの引き継ぎ

    ``changes`` と ``checkpoint`` は作品の履歴から、``reason`` と ``refused`` は画面が見た
    出来事から取る ``ai_words`` だけが AI の言葉
    """

    instruction: str
    reason: StopReason
    #: 止まった訳の文（エラーの文・上限の文など） 無ければ空
    detail: str
    #: 指示の取り消しの段の名前（「AI: 〜」）
    checkpoint: str
    #: 指示の中で変えた手（古い順） 作品の履歴の、段の中の小さな段の名前
    changes: tuple[str, ...]
    #: 指示の途中に人が変えたので、指示の段が 2 つ以上に割れた
    split: bool
    #: 呼んだ道具の数と、最後に呼んだ道具
    tools: int
    last_tool: str
    #: 失敗した道具の文（古い順 新しい方から :data:`ITEMS_LIMIT` 件）
    failures: tuple[str, ...]
    #: 許されなかった・答えなかった・送るのを断った物
    refused: tuple[str, ...]
    #: 残り（送れずに捨てた指示・計画の残りの手順・人へ渡す問題など）
    remaining: tuple[str, ...]
    #: AI の最後の言葉 AI の言葉なので、確かめてから使う
    ai_words: str
    #: 止まった時の作品 別の作品を開いた後に、前の作品の引き継ぎを返さないために持つ
    project: Project = field(repr=False, compare=False)

    @property
    def changed(self) -> bool:
        return bool(self.changes) or self.split

    def undo_note(self) -> str:
        """取り消しの段の案内"""
        if not self.changed:
            return "作品は変えていません"
        note = f"Ctrl+Z で「{self.checkpoint}」を戻すと、この指示の前へ戻ります"
        if self.split:
            note += "（途中で人が変えたので、指示の段が 2 つ以上に分かれています）"
        return note

    def headline(self) -> str:
        """欄の頭の 1 行 止まった訳と、作品を変えたか"""
        what = f"変えた手 {len(self.changes)} 件" if self.changed else "作品は変えていません"
        # どの指示の要約かも書く 続けて送った指示が走っている間も欄は残るので、取り違えないように
        return f"{self.reason.label}・{what}（{self.checkpoint}）"

    def text(self) -> str:
        """人が読み、コピーする全文"""
        lines = [f"引き継ぎ: {self.reason.label}", f"指示: {self.instruction}"]
        if self.detail:
            lines.append(f"止まった訳: {self.detail}")
        where = f"呼んだ道具 {self.tools} 件"
        if self.last_tool:
            where += f"・最後に呼んだ道具 {self.last_tool}"
        lines.append(where)
        lines.append("変えた物（作品の履歴から）:")
        lines.extend(f"・{change}" for change in self.changes)
        if not self.changes:
            lines.append("・無し")
        lines.append(f"取り消し: {self.undo_note()}")
        for head, items in (
            ("失敗した操作", self.failures),
            ("許されなかった物", self.refused),
            ("残り", self.remaining),
        ):
            if items:
                lines.append(f"{head}:")
                lines.extend(f"・{item}" for item in items)
        if self.ai_words:
            lines.append(f"AI の最後の言葉（AI の言葉なので確かめてから使う）: {self.ai_words}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """AI へ返す形 履歴から組んだ欄と、AI の言葉を分けて渡す"""
        return {
            "instruction": self.instruction,
            "stopped": self.reason.value,
            "stopped_label": self.reason.label,
            "detail": self.detail,
            "tools_called": self.tools,
            "last_tool": self.last_tool,
            "changes": list(self.changes),
            "checkpoint": self.checkpoint if self.changed else None,
            "split_by_person": self.split,
            "undo": self.undo_note(),
            "failures": list(self.failures),
            "refused": list(self.refused),
            "remaining": list(self.remaining),
            "ai_words": self.ai_words,
            "note": (
                "changes と checkpoint は作品の履歴から、stopped と refused は画面が見た出来事から"
                "組んだ ai_words は前の AI の言葉なので、get_history や list_clips で確かめてから"
                "使う 今の形は人がその後に変えたかもしれない"
            ),
        }


class TurnRecord:
    """1 つの指示の間に起きたことを集める 画面のスレッドで書く

    指示ごとに :meth:`begin` で空にする 止まらずに終えた指示は :meth:`handoff` が
    ``None`` を返す（引き継ぐ物が無い）
    """

    def __init__(self) -> None:
        self._instruction = ""
        self._reasons: dict[StopReason, str] = {}
        self._tools = 0
        self._last_tool = ""
        self._failures: list[str] = []
        self._refused: list[str] = []
        self._remaining: list[str] = []
        self._ai_words = ""
        self._changes: tuple[str, ...] = ()
        self._split = False

    def begin(self, instruction: str) -> None:
        """次の指示に取りかかった 前の指示で集めた物を全部捨てる

        1 つでも残すと、前の指示の失敗や許されなかった物が次の指示の要約に混ざる
        """
        self._reasons = {}
        self._tools = 0
        self._last_tool = ""
        self._failures = []
        self._refused = []
        self._remaining = []
        self._ai_words = ""
        self._changes = ()
        self._split = False
        self._instruction = _cut(instruction.strip(), INSTRUCTION_LIMIT)

    @property
    def instruction(self) -> str:
        return self._instruction

    def tool(self, name: str) -> None:
        self._tools += 1
        self._last_tool = name

    def failed(self, detail: str) -> None:
        _push(self._failures, detail)

    def refused(self, what: str) -> None:
        _push(self._refused, what)

    def remain(self, what: str) -> None:
        _push(self._remaining, what)

    def said(self, text: str) -> None:
        """AI の言葉 最後の物だけを残す（止まる直前の言葉が、どこで止まったかに近い）"""
        if text.strip():
            self._ai_words = _cut(" ".join(text.split()), AI_WORDS_LIMIT)

    def stop(self, reason: StopReason, detail: str = "") -> None:
        """止まった訳を書く 同じ訳は最初の文を残す（後から来るのは巻き添えの文が多い）"""
        self._reasons.setdefault(reason, detail.strip())

    def forget(self, reason: StopReason) -> None:
        """書いた訳を取り下げる 部品を新しくして送り直しが通ったときなど、立て直した訳に使う"""
        self._reasons.pop(reason, None)

    @property
    def reason(self) -> StopReason | None:
        found = set(self._reasons)
        if self._refused:
            found.add(StopReason.DENIED)
        if not found:
            return None
        return min(found, key=lambda reason: reason.rank)

    def capture(self, document: Document) -> None:
        """指示の段を閉じる前に、段の中で変えた手を履歴から写す

        閉じると段の中の小さな段は 1 段にまとまり、名前が読めなくなる 何度呼んでも、
        段が開いている間の最後の形を残す
        """
        if not document.in_checkpoint:
            return
        self._changes = document.steps_in_checkpoint
        self._split = document.parts_left_in_checkpoint

    def handoff(self, document: Document, checkpoint: str) -> Handoff | None:
        """止まった指示の引き継ぎ 止まらずに終えたなら ``None``"""
        reason = self.reason
        if reason is None:
            return None
        # 訳の文は強い順に並べてつなぐ 接続が切れる前のエラーの文のように、負けた訳の文の方が
        # 何が起きたかを言っていることがある
        texts: list[str] = []
        for found in sorted(self._reasons, key=lambda item: item.rank):
            text = " ".join(self._reasons[found].split())
            if text and text not in texts:
                texts.append(text)
        return Handoff(
            instruction=self._instruction,
            reason=reason,
            detail=_cut(" / ".join(texts), AI_WORDS_LIMIT),
            checkpoint=checkpoint,
            changes=self._changes,
            split=self._split,
            tools=self._tools,
            last_tool=self._last_tool,
            failures=tuple(self._failures),
            refused=tuple(self._refused),
            remaining=tuple(self._remaining),
            ai_words=self._ai_words,
            project=document.project,
        )


class HandoffShelf:
    """文書ごとの、最後に止まった指示の引き継ぎ 1 つ

    1 つだけ持つ 前の止まった指示の続きを頼んで、それもまた止まったなら、新しい方が
    今の作品の形に近い 古い物を並べると、どれが今の形か AI が取り違える
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Handoff | None = None
        #: 置いた後に止まらずに終えた指示の数（今の指示は数えない） AI が、その後の指示で
        #: 形が変わった後の要約を、今の形と取り違えないように
        self._later = 0

    def put(self, handoff: Handoff) -> None:
        with self._lock:
            self._latest = handoff
            self._later = 0

    def turn_finished(self) -> None:
        """止まらずに終えた指示があった"""
        with self._lock:
            if self._latest is not None:
                self._later += 1

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._later = 0

    @property
    def later(self) -> int:
        with self._lock:
            return self._later

    def current(self, document: Document) -> Handoff | None:
        """今の作品の引き継ぎ 別の作品を開いた後なら ``None``

        止まった時の作品が、今の作品か履歴（取り消し・やり直し）に残っていれば同じ作品と
        見る 別の作品を開くと履歴は消えるので、前の作品の引き継ぎは返さない 返すと、
        別の作品のクリップを前の指示の続きで切り詰めかねない
        """
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        project = latest.project
        if project is document.project or any(project is past for past in document.past_projects):
            return latest
        return None


#: 文書ごとの棚 道具はホスト（編集画面）しか受け取らないので、文書から引く
#: ホストに口を足さないのは、ホストの口が増えないことを試験が見張っているため
#: （tests/ai/test_safety.py） 文書が消えたら棚も消えるよう、弱い参照で持つ
_SHELVES: weakref.WeakKeyDictionary[Document, HandoffShelf] = weakref.WeakKeyDictionary()
_SHELVES_LOCK = threading.Lock()


def shelf_for(document: Document) -> HandoffShelf:
    with _SHELVES_LOCK:
        shelf = _SHELVES.get(document)
        if shelf is None:
            shelf = HandoffShelf()
            _SHELVES[document] = shelf
        return shelf


def _push(items: list[str], text: str) -> None:
    line = _cut(" ".join(text.split()), AI_WORDS_LIMIT)
    if not line:
        return
    items.append(line)
    del items[:-ITEMS_LIMIT]


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
