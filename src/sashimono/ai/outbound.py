"""AI の道具が外（Anthropic）へ送る物と、送る前の同意（#342 D4）

AI アシスタントの道具の結果は、そのまま Claude Code を通して Anthropic へ渡る
プレビューの絵・字幕の文・素材の名前と場所は、作品の中身そのもので、人によっては
外へ出したくない 何がどこへ送られるかを先に見せ、送ってよいかを本人に決めてもらう

道具はどれも、結果に何が入りうるかを :attr:`~sashimono.ai.operations.Operation.sends` で
宣言する（:mod:`sashimono.ai.safety` の区分と同じ考え方） 宣言は省けない 省くと
:class:`~sashimono.ai.operations.Operation` を作れず、後から足した道具の書き忘れは
読み込んだ時点で落ちる

同意を取るのは :data:`ASKED` の 3 つだけ 指示の文・編集の構成（トラックやクリップの
位置）・アプリの設定は、AI に編集を頼む以上いつも送る物で、設定画面と
``docs/code-signing-policy.md`` の通信の一覧に書いてある

断られたときの扱いは種類で違う

- 絵と文 結果そのものが送る物なので、道具を動かさずに断る ただし取り消しと履歴の道具は
  断らず、操作の名前の中の文字を伏せ字にして送る（:attr:`Operation.hidden` 名前に文の頭が
  混ざるだけで、断ると AI に取り消しを頼めなくなる 2026-10-15 利用者の決定）
- 素材の名前と場所 一覧のあちこちに混ざる 断っても道具は動かし、結果の中の名前と場所を
  伏せ字（〔素材 m_…〕）に置き換えて送る 素材は ID で指せるので、名前が無くても編集できる
  宣言していない道具の結果も同じく伏せる（書き漏らしても外へは出ない）
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from sashimono.core.model import Project

__all__ = [
    "ASKED",
    "DESTINATION",
    "TEXT_MARK",
    "Outbound",
    "SendAnswer",
    "SendChoice",
    "SendRequest",
    "file_secrets",
    "hide_text",
    "redact",
    "text_secrets",
]

#: 送り先 確かめの窓と設定画面に出す
DESTINATION = "Anthropic（Claude Code を通して Claude のモデルへ）"


class Outbound(StrEnum):
    """道具の結果に入りうる物 値は設定ファイルと台帳（#347）にそのまま載せられる文字"""

    #: 本人の指示と、AI の質問への答え
    CONVERSATION = "conversation"
    #: 編集の構成 プロジェクトの設定・トラック・クリップの位置と長さ・エフェクト・シーン・ID
    PROJECT = "project"
    #: アプリ自身の設定（好み・ショートカット・画面配置・版）
    APP_SETTINGS = "app_settings"
    #: 素材の名前と、ファイルの場所（作品の場所・書き出しの場所を含む）
    FILES = "files"
    #: 字幕の文（起こした結果）と、テキストのクリップの文
    TEXT = "text"
    #: 映像の絵（プレビューを描いた画像）
    FRAMES = "frames"

    @property
    def label(self) -> str:
        """設定画面と確かめの窓に出す名前"""
        return _LABELS[self]

    @property
    def detail(self) -> str:
        """何が入るかの説明 確かめの窓で、名前の下に出す"""
        return _DETAILS[self]

    @property
    def asked(self) -> bool:
        """送る前に本人の同意が要るか"""
        return self in ASKED


_LABELS: dict[Outbound, str] = {
    Outbound.CONVERSATION: "指示と答え",
    Outbound.PROJECT: "編集の構成",
    Outbound.APP_SETTINGS: "アプリの設定",
    Outbound.FILES: "素材の名前と場所",
    Outbound.TEXT: "字幕と文字",
    Outbound.FRAMES: "映像の絵",
}

_DETAILS: dict[Outbound, str] = {
    Outbound.CONVERSATION: "AI の欄に書いた指示と、AI の質問への答え",
    Outbound.PROJECT: (
        "解像度・長さ・トラック・クリップの位置と長さ・エフェクトの種類と値・シーン・操作の履歴"
    ),
    Outbound.APP_SETTINGS: "本人の好み・ショートカット・画面配置・版",
    Outbound.FILES: (
        "素材のファイル名と置き場のパス（ユーザー名を含むことがある）、作品と書き出しの場所"
        " 断ると伏せ字にして送る"
    ),
    Outbound.TEXT: (
        "字幕起こしの結果の文と、テキストのクリップの文"
        " 取り消しと操作の履歴では、名前に入った文を伏せ字にして送る"
    ),
    Outbound.FRAMES: "プレビューを描いた画像（指定したコマの絵 映っている人や画面もそのまま）",
}

#: 送る前に同意を取る物 確かめの窓と設定画面はこの順に並べる
#: 絵を先にするのは、外へ出たときに一番取り返しが付かないため
ASKED: tuple[Outbound, ...] = (Outbound.FRAMES, Outbound.TEXT, Outbound.FILES)


class SendChoice(StrEnum):
    """設定画面で選ぶ、送るかどうか（:class:`~sashimono.ui.workspace.Preferences` に持つ）"""

    #: 初めて送るときに尋ねる（既定）
    ASK = "ask"
    #: 尋ねずに送る
    ALLOW = "allow"
    #: 送らない
    DENY = "deny"

    @property
    def label(self) -> str:
        if self is SendChoice.ASK:
            return "送る前に尋ねる"
        if self is SendChoice.ALLOW:
            return "尋ねずに送る"
        return "送らない"


class SendAnswer(StrEnum):
    """確かめの窓での答え"""

    #: 送る 次からは尋ねない（設定に覚える）
    ALWAYS = "always"
    #: アプリを閉じるまでは送る（設定には書かない）
    UNTIL_CLOSE = "until_close"
    #: 送らない アプリを閉じるまで、もう尋ねない
    DENY = "deny"


@dataclass(slots=True)
class SendRequest:
    """送ってよいかの確かめ 1 件

    答えるのは画面のスレッド、時間切れと中断で畳むのは AI のスレッドと中断のボタン
    先に決まった方だけを残す（:class:`~sashimono.ai.bridge.Question` と同じ）
    """

    #: 送ろうとしている道具
    tool: str
    #: まだ同意を取っていない物（:data:`ASKED` の順）
    kinds: tuple[Outbound, ...]
    done: threading.Event = field(default_factory=threading.Event)
    answer: SendAnswer | None = None
    #: 答えずに畳んだわけ
    reason: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reply(self, answer: SendAnswer) -> bool:
        """答える すでに畳まれていたら何もせず偽を返す"""
        with self._lock:
            if self.done.is_set():
                return False
            self.answer = answer
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

    def text(self) -> str:
        """確かめの窓に出す文"""
        lines = [f"AI アシスタントが、次の物を {DESTINATION} へ送ろうとしています"]
        lines += [f"・{kind.label}: {kind.detail}" for kind in self.kinds]
        lines.append(f"使う道具: {self.tool}")
        lines.append(
            "送った物の扱いは Anthropic の規約と個人情報の方針によります"
            " 選んだ答えは 表示 → 設定… の「AI へ送る物」でいつでも変えられます"
        )
        return "\n".join(lines)


#: 伏せ字にしない短さ 1 文字の名前まで伏せると、関係の無い文字が消えて結果が読めなくなる
_SHORTEST_SECRET = 2

#: 表に無いファイルの場所 表は素材と作品と書き出しの場所だけ アプリの設定に入った置き場
#: （退避のフォルダなど）や、エラーの文に混ざった場所（AI が渡した場所が無いと断った文）もここで拾う
#: 拾う形は ``file://`` の URL・ドライブ名・UNC・``~`` で始まる場所・POSIX の絶対パス
#: AI は Windows 以外の書き方（``/home/…``）でも場所を渡してくるので、ドライブ名だけでは漏れる
#: POSIX の形は区切りを 2 つ以上持つ物だけにし、直前が語・区切り・``:`` でないことを見る
#: 区切りは ``\\`` も見る Windows で ``Path("/home/…")`` を文字にすると ``\\home\\…`` になる
#: ``30/1``（フレームレート）・「色/明るさ」・``https://…`` の中の区切りを場所と見なさないため
#: ドライブ名の頭は英数字の続きでないことだけを見る ``\b`` は日本語の文字も語の一部と見るので、
#: 「場所C:/…」のように文に続けて書かれた場所を拾えない 終わりは空白と、文の区切りの記号
_PATH_TAIL = r"[^\s\"'<>|*?\u3001\u3002\uff0c\uff08\uff09()]*"
_ABSOLUTE_PATH = re.compile(
    r"file://"
    + _PATH_TAIL
    + r"|(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\)"
    + _PATH_TAIL
    + r"|(?<![\w/.:~\\])~[^\s\"'<>|*?/\\]*[/\\]"
    + _PATH_TAIL
    + r"|(?<![\w/.:~\\])[/\\][^\s\"'<>|*?/\\]+[/\\]"
    + _PATH_TAIL
)
_PATH_MARK = "〔ファイルの場所〕"


def file_secrets(
    project: Project, *, project_path: Path | None, export_folder: Path | None
) -> list[tuple[str, str]]:
    """素材の名前と場所を伏せる表（伏せる文字, 置き換える文字） 長い順

    長い順にするのは、場所の中の名前を先に置き換えると、場所の残りが伏せずに残るため
    """
    table: dict[str, str] = {}

    def add(values: Iterable[str], mark: str) -> None:
        for value in values:
            if len(value) >= _SHORTEST_SECRET and value not in table:
                table[value] = mark

    for media in project.media:
        path = media.path
        add((str(path), path.as_posix(), media.name, path.name, path.stem), f"〔素材 {media.id}〕")
    if project_path is not None:
        add(
            (str(project_path), project_path.as_posix(), project_path.name, project_path.stem),
            "〔作品〕",
        )
        add((str(project_path.parent), project_path.parent.as_posix()), "〔作品のフォルダ〕")
    if export_folder is not None:
        add((str(export_folder), export_folder.as_posix()), "〔書き出しのフォルダ〕")
    return sorted(table.items(), key=lambda item: len(item[0]), reverse=True)


#: 字幕と文字を伏せたときの印
TEXT_MARK = "〔伏せ字〕"

#: 操作の名前の「動き: 対象」の対象のうち、伏せずに残す物 数と数え方だけ（「無音カット: 3 か所」）
#: 中身の文字ではないので、残しても字幕や本文は外へ出ない
_COUNT_ONLY = re.compile(r"[\d\s.,/]*(?:か所|件|本|個|コマ|フレーム|秒)?")
#: 名前の中のかぎ括弧 中は本文の引用（「選んでいるテキスト「…」を写す」など）
_QUOTED = re.compile(r"「[^」]*」")


def text_secrets(project: Project) -> list[str]:
    """字幕と文字を伏せる表 字幕起こしの文と、テキストのクリップの文（メインとシーン） 長い順"""
    found: set[str] = set()
    for media in project.media:
        for _stream, transcript in media.transcripts:
            found.update(segment.text for segment in transcript.segments)
    for timeline in (project.timeline, *(scene.timeline for scene in project.scenes)):
        for track in timeline.tracks:
            for clip in track.clips:
                if clip.source is not None and "text" in clip.source.params:
                    found.add(str(clip.source.params["text"]))
    lines = {line for text in found for line in (text, *text.splitlines())}
    return sorted(
        (line for line in lines if len(line.strip()) >= _SHORTEST_SECRET),
        key=len,
        reverse=True,
    )


def hide_text(value: Any, secrets: Iterable[str]) -> Any:
    """操作の名前（取り消しと履歴の道具の結果）の中の文字を伏せる 元の値は変えない

    表（今ある字幕と本文）に載った文字を伏せたうえで、「動き: 対象」の対象と、かぎ括弧の中を
    伏せる 表だけでは足りない 名前は文の頭（12 文字）で切ってあり、消したテキストや
    直す前の字幕の文も、表から消えた後まで履歴の名前に残る 動き（「テキストを追加」）は
    残すので、AI はどの操作を取り消したかを読める
    """
    pairs = list(secrets)

    def one(text: str) -> str:
        for secret in pairs:
            if secret in text:
                text = text.replace(secret, TEXT_MARK)
        text = _QUOTED.sub(f"「{TEXT_MARK}」", text)
        for separator in (": ", "："):
            head, found, tail = text.partition(separator)
            if found:
                if _COUNT_ONLY.fullmatch(tail.strip()) is None:
                    return f"{head}{found}{TEXT_MARK}"
                return text
        return text

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return one(item)
        if isinstance(item, Mapping):
            return {key: walk(entry) for key, entry in item.items()}
        if isinstance(item, list | tuple):
            return [walk(entry) for entry in item]
        return item

    return walk(value)


def redact(value: Any, table: Iterable[tuple[str, str]]) -> Any:
    """結果の中の文字を、表に従って伏せ字にする 辞書と並びは中まで辿る 元の値は変えない"""
    pairs = list(table)

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            for secret, mark in pairs:
                if secret in item:
                    item = item.replace(secret, mark)
            return _ABSOLUTE_PATH.sub(_PATH_MARK, item)
        if isinstance(item, Mapping):
            return {walk(key): walk(entry) for key, entry in item.items()}
        if isinstance(item, list | tuple):
            return [walk(entry) for entry in item]
        return item

    return walk(value)
