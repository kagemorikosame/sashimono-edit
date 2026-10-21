"""AI の道具が使わないコマンドと、その理由（#351 E5）

画面の操作はどれも ``core`` の :class:`~sashimono.core.commands.Command` を通る AI の道具も
同じコマンドを通すので、コマンドの一覧と道具の対応を見れば、人にできて AI にできない操作が
数えられる（#298 の操作の同等性）

道具をコマンドの一覧から自動で作ることはしない（#351 で比べて決めた） 道具には引数の説明・
安全の段（``Operation.safety``）・外へ送る物（``Operation.sends``）・確かめ（``consent``）が要り、
どれもコマンドの形からは決まらない 自動で作っても手で書く所は残り、AI に出してはいけない
コマンド（作品の方針を変える・素材を外すなど）まで出てしまう

代わりに、AI の道具が 1 つも通さないコマンドをここへ理由つきで並べ、
``tests/ai/test_command_coverage.py`` が見張る 道具が通すかは、道具を実際に動かして数える

- コマンドを足して、道具が通さず、ここにも無いと落ちる 道具を足すか、理由を書く
- 道具が通すようになったコマンドがここに残っていても落ちる（理由が古くなる）
- 1 行 1 コマンド 並行して進む PR がコマンドや道具を足したときは、取り込む側がここを直す
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["COMMANDS_WITHOUT_TOOLS", "Note", "Why"]

#: 抜けている道具を足す Issue（E3） 人の操作だけのコマンドの多くはここで埋める
_E3 = 349


class Why(Enum):
    """AI の道具が通さない訳の分け方"""

    #: 画面の操作は使う AI の道具がまだ無い（``issue`` で足す）か、足さないと決めた
    PERSON_ONLY = "人の操作だけ"
    #: 画面も AI も使わない（試験や道具の台本の中だけ） 残す訳を ``reason`` に書く
    #: どこからも使われない物は残さず外す（#351 で AddSegment と RetimeSegment を外した）
    NEITHER = "どちらでもない"


@dataclass(frozen=True, slots=True)
class Note:
    """AI の道具が通さないコマンド 1 つの理由"""

    why: Why
    reason: str
    #: 道具を足す先の Issue（#298 の子） 足さないと決めた物は ``None`` にし、``reason`` に
    #: 「足さない」と決めた訳を書く
    issue: int | None = None


def _person(reason: str, issue: int | None = _E3) -> Note:
    return Note(Why.PERSON_ONLY, reason, issue)


def _neither(reason: str) -> Note:
    return Note(Why.NEITHER, reason)


#: AI の道具が 1 つも通さないコマンド 名前はクラスの名前 道具を足したら行を消す
COMMANDS_WITHOUT_TOOLS: dict[str, Note] = {
    "AdoptLegacyStroke": _person(
        "設定パネルが前の版の縁取りを層へ移す下ごしらえ AI の縁取りの道具は層の ID を指すので、"
        "前の版の縁は指せない"
    ),
    "MoveKeyframe": _person("キーフレームを動かす（設定パネル・グラフエディタ）"),
    "RenameProject": _neither(
        "作品の名前は保存したファイルの名前から付く 名前だけを変える入口は画面にも AI にも無い "
        "残すのは、作品を 1 段だけ確実に変える一番小さい命令として、履歴・取り消し・控え・"
        "自動保存の試験（tests の 11 ファイル）と字幕の速さを測る tools/bench_subtitles.py "
        "が使うため 外すと、それらが中身を変える命令を別に持つことになる"
    ),
    "SetFrameRate": _person("フレームレートの変更（プロジェクト設定・動画に合わせる）"),
    "SetStyleGuide": _person(
        "編集の方針は本人が AI に渡す決まりで、AI が緩められると意味が無いので"
        "道具を足さない（#343）",
        issue=None,
    ),
    "TrimClips": _person(
        "選んだクリップをまとめて伸び縮み（タイムラインのドラッグ） AI は trim_clip を 1 本ずつ"
        "呼べば同じ形になるが、取り消しが本数だけ要る"
    ),
}
