"""検査で見つけた問題を AI が直して調べ直す繰り返し（#339 C3）

2026-10-20 に本人が決めた作り

- 直すのは今ある道具のまま（:data:`FIXES` の表が、問題ごとにどの道具で直すかを持つ）
- 回数を数えて上限で止めるのはアプリ（``check_output`` が数え、ブリッジが上限の後の編集を断る）
  AI への説明（``ai/session.py``）は手順を言うだけで、回数を守るかを AI 任せにしない
- 直してから調べ直すと 1 回 上限は本人の設定（``Preferences.ai_fix_rounds`` 既定 3 0 は調べるだけ）
- 直す回ごとに「AI: 検査の直し（n 回目）」の段へ分け、人が Ctrl+Z 1 回でその回だけを戻せる

表の扱いは 4 つ 自動で直す物でも「変更を自動で承認」が切れていれば今までどおり 1 回ずつ確かめる

| 扱い | 値 | AI がすること |
|---|---|---|
| 自動で直す | ``auto`` | 道具で直し、絵か検査で確かめる |
| 尋ねる | ``ask`` | 直す前に人に尋ねる（詰める・削るなど、作品の時刻や中身が変わる） |
| 人へ渡す | ``handoff`` | 直さずに、何がどこにあるかを人へ伝える（直す道具が無い・中身の判断） |
| 直さない | ``leave`` | わざとのことが多い（頭と終わりのフェード・流れるテロップ） |

Qt を知らない（試験から偽のホストで動かす）
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass

from sashimono.core.commands import Author, Document
from sashimono.engine.checks.report import HIGH, LOW, MEDIUM
from sashimono.engine.checks.scan import (
    BLACK,
    CLIPPING,
    FREEZE,
    LENGTH,
    LOUDNESS,
    SILENCE,
    TEXT_OVERFLOW,
)

__all__ = [
    "ASK",
    "AUTO",
    "FIXES",
    "FIX_ACTIONS",
    "FIX_ROUNDS_RANGE",
    "HANDOFF",
    "LEAVE",
    "PASSES_AFTER_LIMIT",
    "Fix",
    "FixRound",
    "fix_for",
    "fix_label",
    "fixes_closed",
    "start_check",
]

AUTO = "auto"
ASK = "ask"
HANDOFF = "handoff"
LEAVE = "leave"
FIX_ACTIONS = (AUTO, ASK, HANDOFF, LEAVE)

#: 直す回数の上限の範囲（本人の設定） 10 を超えて回す訳は無く、止まらない繰り返しは費用だけが増える
FIX_ROUNDS_RANGE = (0, 10)

#: 音割れのうち、混ぜた音が最大を超えた物（下げれば消える）と、素材の音が割れている物（下げても
#: 割れは残る）を分ける印 ``engine/checks/scan.py`` が程度の数（measure）に入れる
_MIXED = "mixed"
_SOURCE = "source"


@dataclass(frozen=True, slots=True)
class Fix:
    """問題 1 つの直し方 ``tools`` は使う道具の名前（``ai/operations.py`` の ``Operation.name``）"""

    action: str
    tools: tuple[str, ...]
    how: str

    def as_dict(self) -> dict[str, object]:
        return {"action": self.action, "tools": list(self.tools), "how": self.how}


#: 種類と程度ごとの直し方 2026-10-20 に本人が決めた表（#339 の Issue の本文）
#: 音割れだけは程度が 1 つなので、混ぜた音か素材かで分ける（3 つ目の値）
#: 種類か程度を足したらここにも足す 足さないと :func:`fix_for` が人へ渡す側に倒すが、
#: 試験（``tests/ai/test_fix_loop.py``）が表の抜けとして落とす
FIXES: Mapping[tuple[str, str, str | None], Fix] = {
    (LOUDNESS, HIGH, None): Fix(
        AUTO,
        ("normalize_loudness",),
        "作品全体を基準の音量へそろえる 声と BGM の釣り合いは変わらない",
    ),
    (LOUDNESS, MEDIUM, None): Fix(
        AUTO,
        ("normalize_loudness",),
        "作品全体を基準の音量へそろえる 声と BGM の釣り合いは変わらない",
    ),
    (CLIPPING, HIGH, _MIXED): Fix(
        AUTO,
        ("normalize_loudness", "set_volume"),
        "混ぜた音が最大を超えている normalize_loudness の peak_ceiling_db か、"
        "その区間の音のクリップを set_volume の gain_db で下げる",
    ),
    (CLIPPING, HIGH, _SOURCE): Fix(
        HANDOFF,
        (),
        "素材の音が割れている 下げても割れは消えないので、録り直すか素材を替えるかは人が決める",
    ),
    (TEXT_OVERFLOW, HIGH, None): Fix(
        AUTO,
        ("set_param", "clean_subtitles"),
        "文字を小さくするか位置を内側へ寄せる（set_param） 字幕から置いた文なら clean_subtitles の"
        " max_line_chars で折り返す 直したら preview_frame で見る",
    ),
    (TEXT_OVERFLOW, MEDIUM, None): Fix(
        AUTO,
        ("set_param", "clean_subtitles"),
        "余白（安全域）の内側へ寄せる 文字を小さくするか位置を内側へ寄せる（set_param）"
        " 直したら preview_frame で見る",
    ),
    (TEXT_OVERFLOW, LOW, None): Fix(
        LEAVE, (), "動く文字（流れるテロップなど）はわざと外へ出すことが多い"
    ),
    (BLACK, HIGH, None): Fix(
        ASK,
        ("move_clips", "delete_clips"),
        "何も置いていない隙間 詰めると後ろの時刻がすべて動き、間をわざと空けたこともあるので、"
        "詰めるかを人に尋ねてから move_clips で後ろを前へ寄せる",
    ),
    (BLACK, MEDIUM, None): Fix(
        HANDOFF, (), "置いた絵が暗いか透明になっている 演出か素材の問題かは人が見て決める"
    ),
    (BLACK, LOW, None): Fix(LEAVE, (), "頭か終わりの黒 フェードの演出ならそのままでよい"),
    (FREEZE, MEDIUM, None): Fix(
        HANDOFF, (), "動画の絵が止まっている 止め絵の演出か素材の終わりかは人が見て決める"
    ),
    (FREEZE, LOW, None): Fix(LEAVE, (), "静止画・文字だけの区間 意図した止め絵ならそのままでよい"),
    (SILENCE, HIGH, None): Fix(
        HANDOFF, (), "全体が無音 音を置き忘れたか、すべて消音になっている 人に伝える"
    ),
    (SILENCE, MEDIUM, None): Fix(
        ASK,
        ("jet_cut", "place_sound"),
        "途中の無音 間の演出か録り漏れかで直し方が違うので、詰めるか（jet_cut）音を置くか"
        "（place_sound）を人に尋ねてから直す",
    ),
    (SILENCE, LOW, None): Fix(LEAVE, (), "頭か終わりの無音 フェードの演出ならそのままでよい"),
    (LENGTH, MEDIUM, None): Fix(HANDOFF, (), "どこを削るか・足すかは中身の判断なので、人に伝える"),
}

#: 表に無い組み合わせ 黙って直すより、人へ渡す側に倒す
_UNKNOWN = Fix(HANDOFF, (), "直し方を決めていない種類 人に伝える")

#: 直す回数の上限に達した後も通す、取り消せる編集の区分の道具（ブリッジが見る）
#: どれも作品を新しく変えない 取り消しとやり直しは前の形へ辿るだけで、人に頼まれて直しを戻すことが
#: ある 書き出しの中止は人が許した書き出しを止めるだけで、断ると AI が始めた書き出しを止められない
#: 保存と書き出しは取り消せない操作の道を通るので、ここに書かなくても断られない
#: 足したら docs/development.md の「検査で見つけた問題を AI が直す繰り返し」にも書く（試験が照らす）
PASSES_AFTER_LIMIT = frozenset({"undo", "redo", "cancel_export"})

#: 上限に達した後の直し方 直さずに人へ渡す
_CLOSED = "直す回数の上限に達したので、直さずに人へ伝える"

#: 書き出した動画のファイルを調べたときの直し方 今の作品と同じ物かは分からないので直さない
_FILE = (
    "書き出した動画のファイルの問題 今の作品から書き出した物かは確かめていないので、今の作品は"
    "直さない 直すなら path を省いてタイムラインを調べ直す"
)


def _variant(kind: str, measure: Mapping[str, object]) -> str | None:
    if kind != CLIPPING:
        return None
    return _MIXED if measure.get("over_full_scale") else _SOURCE


def fix_for(
    kind: str,
    severity: str,
    measure: Mapping[str, object],
    *,
    closed: bool = False,
    timeline: bool = True,
) -> Fix:
    """問題 1 つの直し方 ``closed`` なら（上限に達した後）直す物も人へ渡す

    ``timeline`` が偽（書き出した動画のファイルを調べた）なら、直す物も人へ渡す ファイルが
    今の作品から書き出した物かは確かめていない（古い書き出しや別の作品の動画も、作品のフォルダか
    書き出しのフォルダにあれば調べられる） 直す道具を返すと、AI が関係の無い今の作品を変える
    """
    fix = FIXES.get((kind, severity, _variant(kind, measure)), _UNKNOWN)
    if not timeline and fix.action in (AUTO, ASK):
        return Fix(HANDOFF, (), f"{_FILE}（{fix.how}）")
    if closed and fix.action in (AUTO, ASK):
        return Fix(HANDOFF, (), f"{_CLOSED}（{fix.how}）")
    return fix


def fix_label(round_number: int) -> str:
    """直す回の段の名前 人が履歴と〔元に戻す〕の名前で、どの回の直しかを見分ける"""
    return f"AI: 検査の直し（{round_number} 回目）"


@dataclass(frozen=True, slots=True)
class FixRound:
    """今の指示の直しの回数 ``check_output`` が結果に添える"""

    #: 直してから調べ直した回数（直した回の数）
    used: int
    limit: int
    #: これから直す回の番号 上限に達していれば ``None``
    next_round: int | None

    @property
    def closed(self) -> bool:
        return self.next_round is None

    def as_dict(self) -> dict[str, object]:
        return {
            "used": self.used,
            "limit": self.limit,
            "next_round": self.next_round,
            "closed": self.closed,
        }


@dataclass(slots=True)
class _Loop:
    """1 つの指示の数え 指示（段）が替わると作り直す"""

    serial: int
    #: 前に調べたときに AI が作品を変えた回数（:meth:`Document.changes_by`） 同じなら、その間に
    #: AI は何も直していない 人が手で変えただけの回は数えない
    checked: int | None = None
    used: int = 0
    closed: bool = False


#: 文書ごとの数え 文書を閉じたら一緒に消える（弱い参照） ホストに持たせないのは、
#: アプリの設定を変える口と同じ並びに、AI が数えを書き換えられる口を増やさないため
_LOOPS: weakref.WeakKeyDictionary[Document, _Loop] = weakref.WeakKeyDictionary()


def _loop(document: Document) -> _Loop:
    loop = _LOOPS.get(document)
    if loop is None or loop.serial != document.checkpoint_serial:
        loop = _Loop(document.checkpoint_serial)
        _LOOPS[document] = loop
    return loop


def start_check(document: Document, limit: int) -> FixRound:
    """``check_output`` が調べ始めるたびに呼ぶ（編集画面のスレッド）

    前に調べた後に AI の段で作品が変わっていれば、直した回を 1 つ数える（人の段だけで変わったなら
    数えない 数えると、人が手で直した分だけ AI の直せる回が減る） 上限に達していなければ、
    ここで指示の段を区切り、この後の直しを「AI: 検査の直し（n 回目）」の段にする
    上限に達したら、この指示の残りの編集をブリッジが断る（:func:`fixes_closed`）
    同じ作品のまま呼び直した（途中までの結果の続きを読む）だけなら数えない
    """
    loop = _loop(document)
    changed = document.changes_by(Author.AI)
    if loop.checked is not None and changed != loop.checked and not loop.closed:
        loop.used += 1
    loop.checked = changed
    if loop.closed or loop.used >= max(0, limit):
        loop.closed = True
        return FixRound(loop.used, limit, None)
    next_round = loop.used + 1
    document.split_checkpoint(fix_label(next_round))
    return FixRound(loop.used, limit, next_round)


def fixes_closed(document: Document) -> bool:
    """今の指示が直す回数の上限に達したか 達したら、ブリッジは取り消せる編集を断る"""
    loop = _LOOPS.get(document)
    return loop is not None and loop.serial == document.checkpoint_serial and loop.closed
