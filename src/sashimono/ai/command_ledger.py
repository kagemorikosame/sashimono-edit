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
    #: 画面も AI も使わない（読み込みや試験の中だけ） 入口を作るときに道具も足す
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
    "AddSegment": _neither("字幕を手で 1 枚足す入口は、字幕パネルにも AI にもまだ無い"),
    "AdoptLegacyStroke": _person(
        "設定パネルが前の版の縁取りを層へ移す下ごしらえ AI の縁取りの道具は層の ID を指すので、"
        "前の版の縁は指せない"
    ),
    "ClearKeyframes": _person("キーフレームをまとめて消す（設定パネル）"),
    "ConvertLayers": _person("置き方の方式の切り替え（ファイルメニュー）で置いた物を移す"),
    "InsertGap": _person("挿入貼り付け（編集メニュー）が後ろを押して空ける"),
    "MergeWithNext": _person("字幕を次の字幕とつなぐ（字幕パネル）"),
    "MoveEffect": _person("エフェクトの並べ替え（設定パネル）"),
    "MoveKeyframe": _person("キーフレームを動かす（設定パネル・グラフエディタ）"),
    "MoveTrack": _person("トラックの並べ替え（トラックの見出しのドラッグ）"),
    "RemoveKeyframe": _person("キーフレームの削除（設定パネル・グラフエディタ）"),
    "RemoveMedia": _person(
        "素材一覧から素材を外す AI が消せるのは作品の中のクリップだけ（#341 の試験が止める） "
        "足すときは取り消せない操作と同じく毎回確かめる（2026-10-14 の利用者の決定）"
    ),
    "RemoveScene": _person("シーンの削除（シーンメニュー）"),
    "RemoveSegment": _person("字幕 1 枚の削除（字幕パネル）"),
    "RemoveTrack": _person("トラックの削除（タイムラインの右クリック）"),
    "RenameProject": _neither(
        "作品の名前は保存したファイルの名前から付く 名前だけを変える入口は無い"
    ),
    "RenameScene": _person("シーンの名前の変更（シーンメニュー）"),
    "RenameTrack": _person("トラックの名前の変更（トラックの見出し）"),
    "RetimeSegment": _neither("字幕の時刻を動かす入口は、字幕パネルにも AI にもまだ無い"),
    "SetBlending": _person("半透明の重ね合わせの方法（プロジェクト設定）"),
    "SetFrameRate": _person("フレームレートの変更（プロジェクト設定・動画に合わせる）"),
    "SetLayerMode": _person("置き方の方式の切り替え（ファイルメニュー・プロジェクト設定）"),
    "SetStyleGuide": _person(
        "編集の方針は本人が AI に渡す決まりで、AI が緩められると意味が無いので"
        "道具を足さない（#343）",
        issue=None,
    ),
    "SetWorkArea": _person("書き出し範囲の設定と解除（編集メニュー・目盛りのドラッグ）"),
    "SplitSegment": _person("字幕 1 枚を 2 つに割る（字幕パネル）"),
    "TrimClips": _person(
        "選んだクリップをまとめて伸び縮み（タイムラインのドラッグ） AI は trim_clip を 1 本ずつ"
        "呼べば同じ形になるが、取り消しが本数だけ要る"
    ),
}
