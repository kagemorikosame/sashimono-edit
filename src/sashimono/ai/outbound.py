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
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from sashimono.core.model import Effect, MediaId, Project
from sashimono.effects import registry
from sashimono.effects.sources import source_registry
from sashimono.effects.spec import FileSpec, TextSpec
from sashimono.effects.strokes import STROKE

__all__ = [
    "ASKED",
    "DESTINATION",
    "FIELD_SENDS",
    "NAME_MARK",
    "PATH_MARK",
    "PROJECT_MARK",
    "REMEMBERED_MARK",
    "SAMPLE_HEAD",
    "SAMPLE_LINES",
    "SAMPLE_LINE_CHARS",
    "TEXT_MARK",
    "UNKNOWN_FIELD_SENDS",
    "NamedMedia",
    "Outbound",
    "Preview",
    "PreviewSource",
    "Sample",
    "SendAnswer",
    "SendChoice",
    "SendRequest",
    "build_samples",
    "clip_line",
    "field_sends",
    "file_sample",
    "file_secrets",
    "frame_sample",
    "from_arguments",
    "from_result",
    "hide_text",
    "masked_value",
    "redact",
    "secret_pairs",
    "sendable_now",
    "sending",
    "text_sample",
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
    #: 字幕の文（起こした結果）と、テキストのクリップの文 本人の書いた文（目印の名前・
    #: 編集の方針 #330 #343）と、画面の中から読み取った文字（OCR #324）もここに数える
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
        "解像度・長さ・トラック・クリップの位置と長さ・エフェクトの種類と値・シーン・操作の履歴・"
        "画面の表示（見えている範囲・再生・選んでいる物・進み具合）"
    ),
    Outbound.APP_SETTINGS: (
        "本人の好み・ショートカット・画面配置・版・更新の様子・保存したプリセットとエイリアスと"
        "テンプレートの棚の名前・互換性レポートの記録"
    ),
    Outbound.FILES: (
        "素材のファイル名と置き場のパス（ユーザー名を含むことがある）、作品と書き出しの場所、"
        "テンプレートの棚のフォルダの名前、エフェクトが読む画像とスクリプトの置き場の場所"
        " 断ると伏せ字にして送る"
    ),
    Outbound.TEXT: (
        "字幕起こしの結果の文と、テキストのクリップの文、目印の名前、作品の編集の方針の文、"
        "画面の中から読み取った文字（テロップ・看板・資料 手元の OCR で読み、絵は送らない）"
        " 取り消しと操作の履歴とクリップの値では、入った文を伏せ字にして送る"
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


class PreviewSource(StrEnum):
    """下見を何から作るか（#425）"""

    #: 先に道具を動かし、結果を保留して、その結果から作る 読み取りだけの道具に限る
    #: 送ると答えたら、保留した結果をそのまま AI へ返す 下見と送る物が食い違わない
    RESULT = "result"
    #: 道具の引数から作る 作品を変える道具・アプリの外へ何かを出す道具（クリップボード）
    #: 確かめる前に動かすと、断っても作品や外が変わってしまう
    ARGUMENTS = "arguments"


@dataclass(frozen=True, slots=True)
class Preview:
    """道具ごとの下見の作り方（:attr:`~sashimono.ai.operations.Operation.preview` #425）

    ``sends`` と同じく道具ごとに書く 書かないと道具を作れない 読み取りだけに見える道具でも、
    クリップボードへ写す物は先に動かすと断っても写ってしまう 動かしてよいかは道具を書いた人
    にしか分からないので、``writes`` から決めずに書かせる
    """

    source: PreviewSource
    #: 字幕と文字として下見に出す欄の名前 結果（か引数）の中で、この名前の欄の文字を数える
    #: 欄の名前で見るのは、ID や秒や種類の名前まで「文字」と数えると、送る文の量が読めないため
    text_keys: frozenset[str] = frozenset({"text"})
    #: 引数から作る道具で、引数に無い送る物を先に読む口（作品を変えず、外へも出さない読み取り）
    #: クリップボードへ字幕を写す道具は、引数が字幕の ID だけなので、本文を読まないと何を
    #: 写すのかが窓に出ない 読めなければ引数から作る 編集画面のスレッドで呼ぶ
    reader: Callable[[Any, dict[str, Any]], object] | None = None


def from_result(*text_keys: str) -> Preview:
    """先に動かして結果から下見を作る 欄を書かなければ ``text`` だけを数える"""
    return Preview(PreviewSource.RESULT, frozenset(text_keys or ("text",)))


def from_arguments(
    *text_keys: str, reader: Callable[[Any, dict[str, Any]], object] | None = None
) -> Preview:
    """引数から下見を作る 欄を書かなければ ``text`` だけを数える

    ``reader`` は :attr:`Preview.reader`
    """
    return Preview(PreviewSource.ARGUMENTS, frozenset(text_keys or ("text",)), reader)


#: 下見に持つ行の上限 字幕の多い作品で全部を持つと、窓を開くのが重くなる
#: 件数（:attr:`Sample.count`）は上限に関係なく全部を数える
SAMPLE_LINES = 200
#: 見出しの 1 行の長さの上限（文字） 長い本文 1 つで窓の幅が広がらないように
#: 詳細の行は切らない（:func:`clip_line`）
SAMPLE_LINE_CHARS = 200
#: 窓を開いたときに、詳細を開かずに見せる行の数
SAMPLE_HEAD = 3


@dataclass(frozen=True, slots=True)
class Sample:
    """送る物の種類 1 つの下見（#425）"""

    kind: Outbound
    #: 件数 字幕と文字は行、絵は枚、名前と場所は名前の数
    count: int
    #: 字幕と文字の文字数 ほかの種類は 0
    chars: int = 0
    #: 下見の行（上限 :data:`SAMPLE_LINES` まで） 名前と場所は「元 → 伏せた後」
    lines: tuple[str, ...] = ()
    #: 絵（PNG） 映像の絵だけ
    images: tuple[bytes, ...] = ()
    #: 添える注意（引数から作った・ユーザー名を含む）
    note: str = ""

    @property
    def amount(self) -> str:
        """件数の短い文 記録と窓の見出しに出す"""
        if self.kind is Outbound.TEXT:
            return f"{self.count} 行 {self.chars} 文字"
        if self.kind is Outbound.FRAMES:
            return f"{self.count} 枚"
        return f"{self.count} 件"

    @property
    def omitted(self) -> int:
        """上限で下見から落とした行の数 絵は 1 枚ずつが行の代わりなので数えない"""
        if self.kind is Outbound.FRAMES:
            return 0
        return max(0, self.count - len(self.lines))


def clip_line(text: str) -> str:
    """見出しに出す 1 行 長い本文 1 つで窓の幅が広がらないように切る

    下見の行そのものは切らずに持つ 詳細と「最後に送った物」では、送った文を全部読めるように
    """
    line = " ".join(text.split())
    if len(line) <= SAMPLE_LINE_CHARS:
        return line
    return line[: SAMPLE_LINE_CHARS - 1] + "…"


def _texts_under(value: Any, keys: frozenset[str]) -> Iterator[str]:
    """``keys`` の欄に入った文字を順に返す 欄の中の並び・辞書もたどる"""
    if isinstance(value, Mapping):
        for key, entry in value.items():
            if key in keys:
                yield from _all_texts(entry)
            else:
                yield from _texts_under(entry, keys)
    elif isinstance(value, list | tuple):
        for entry in value:
            yield from _texts_under(entry, keys)


def _all_texts(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for entry in value.values():
            yield from _all_texts(entry)
    elif isinstance(value, list | tuple):
        for entry in value:
            yield from _all_texts(entry)


def text_sample(value: Any, keys: frozenset[str]) -> Sample:
    """字幕と文字の下見 行数・文字数と、先頭から :data:`SAMPLE_LINES` 行"""
    lines = [line for text in _texts_under(value, keys) for line in text.splitlines() if line]
    return Sample(
        Outbound.TEXT,
        count=len(lines),
        chars=sum(len(line) for line in lines),
        lines=tuple(lines[:SAMPLE_LINES]),
    )


def _images(value: Any) -> Iterator[Any]:
    """結果の中の絵（:class:`~sashimono.ai.operations.ImageResult`） 型は形で見分ける

    ここから operations を読むと、operations がこのモジュールを読むので輪になる
    """
    if isinstance(value, bytes):
        return
    png = getattr(value, "png", None)
    if isinstance(png, bytes):
        yield value
    elif isinstance(value, Mapping):
        for entry in value.values():
            yield from _images(entry)
    elif isinstance(value, list | tuple):
        for entry in value:
            yield from _images(entry)


def frame_sample(value: Any) -> Sample:
    """映像の絵の下見 送る絵そのものと、添える説明（コマの位置）"""
    found = list(_images(value))
    captions = tuple(str(getattr(image, "caption", "")) for image in found if image.caption)
    return Sample(
        Outbound.FRAMES,
        count=len(found),
        lines=captions,
        images=tuple(image.png for image in found),
    )


def _user_name() -> str:
    """場所に混ざるユーザー名 分からなければ空"""
    try:
        return Path.home().name
    except (OSError, RuntimeError):
        return ""


def file_sample(
    value: Any, pairs: Iterable[tuple[str, str]], name_keys: Iterable[str] = ()
) -> Sample:
    """素材の名前と場所の下見 名前ごとに「元 → 伏せた後」

    伏せた後の形は :func:`redact` そのもので作る 断ったときに実際に送る形と食い違わないように
    """
    hide = _redactor(pairs, name_keys)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def look(key: object, text: str) -> None:
        if not text or text in seen:
            return
        after = hide(key, text)
        if after != text:
            seen.add(text)
            found.append((text, str(after)))

    def walk(key: object, item: Any) -> None:
        if isinstance(item, str):
            look(key, item)
        elif isinstance(item, Mapping):
            for name, entry in item.items():
                walk(name, entry)
        elif isinstance(item, list | tuple):
            for entry in item:
                walk(key, entry)
        else:
            caption = getattr(item, "caption", None)
            if isinstance(caption, str):
                look(None, caption)

    walk(None, value)
    user = _user_name()
    with_user = sum(1 for before, _after in found if len(user) >= 2 and user in before)
    note = f"ユーザー名「{user}」を含む物が {with_user} 件あります" if with_user else ""
    return Sample(
        Outbound.FILES,
        count=len(found),
        lines=tuple(f"{before} → {after}" for before, after in found[:SAMPLE_LINES]),
        note=note,
    )


def build_samples(
    kinds: Iterable[Outbound],
    value: Any,
    preview: Preview,
    *,
    pairs: Iterable[tuple[str, str]] = (),
    name_keys: Iterable[str] = (),
) -> tuple[Sample, ...]:
    """送る物の種類ごとの下見 ``value`` は道具の結果か引数（``preview.source`` で決まる）"""
    table = list(pairs)
    samples: list[Sample] = []
    for kind in kinds:
        if kind is Outbound.TEXT:
            sample = text_sample(value, preview.text_keys)
        elif kind is Outbound.FRAMES:
            sample = frame_sample(value)
        elif kind is Outbound.FILES:
            sample = file_sample(value, table, name_keys)
        else:
            continue
        if preview.source is PreviewSource.ARGUMENTS:
            # 結果はまだ無い 引数に無い物（道具が読んで返す名前など）も送りうると添える
            extra = "道具の引数から作った下見です 結果に入る物はここに出ません"
            sample = Sample(
                sample.kind,
                count=sample.count,
                chars=sample.chars,
                lines=sample.lines,
                images=sample.images,
                note=f"{sample.note} {extra}".strip(),
            )
        samples.append(sample)
    return tuple(samples)


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
    #: 送る物の下見（#425） 種類ごとに 1 つ 作れなかった種類は無い
    samples: tuple[Sample, ...] = ()
    #: 道具の呼び出しごとに尋ねているか（設定） 窓は「アプリを閉じるまで送る」の代わりに
    #: 「今回は送る」を出す
    each_call: bool = False
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
        amounts = {sample.kind: sample.amount for sample in self.samples}
        for kind in self.kinds:
            amount = amounts.get(kind)
            lines.append(
                f"・{kind.label}（{amount}）: {kind.detail}"
                if amount
                else f"・{kind.label}: {kind.detail}"
            )
        lines.append(f"使う道具: {self.tool}")
        lines.append(
            "送った物の扱いは Anthropic の規約と個人情報の方針によります"
            " 選んだ答えは 表示 → 設定… の「AI へ送る物」でいつでも変えられます"
        )
        return "\n".join(lines)


#: 文の中で、どこに混ざっていても伏せる長さ これより短い名前（1 文字）は、前後が語の続きで
#: ない所（文の頭・空白・記号の隣）だけで伏せる どこでも伏せると、関係の無い文字が消えて
#: 結果が読めなくなる 名前そのものを返す欄は長さに関係なく値ごと伏せる（``name_keys``）
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
    # 根の直下の 1 段だけの場所（``/backup``） 区切りが 1 つしか無いので上には当たらない
    # 文字の頭か空白の後に始まり、空白か終わりで閉じる物だけ ``1/2 画質`` の ``/2`` は
    # 数字の後なので当たらない
    + r"|(?:^|(?<=\s))[/\\][^\s\"'<>|*?/\\]+(?=$|\s)"
)
_PATH_MARK = "〔ファイルの場所〕"
#: 場所の欄（:class:`FileSpec`）を値ごと伏せた印 文の中の場所を形で伏せた印と同じ
PATH_MARK = _PATH_MARK

#: 値ごと伏せる項目の名前 設定の中のフォルダ（``state_folder``・``export_folder``）や
#: 一覧の ``path`` など 中身の形（正規表現）で見分けると、``/backup`` のような短い場所や、
#: 区切りの無い書き方で漏れる 名前で分かる物は中身を見ずに伏せる
_PLACE_KEY = re.compile(r"(?:^|_)(?:path|folder|dir|directory|file)s?$")


class NamedMedia(Protocol):
    """名前と場所を持つ素材（:class:`MediaItem` か、文書が覚えた軽い記録 ``SeenMedia``）"""

    @property
    def id(self) -> MediaId: ...

    @property
    def name(self) -> str: ...

    @property
    def path(self) -> Path: ...


#: 作品の名前を伏せた印
PROJECT_MARK = "〔作品〕"
#: 名前そのものを返す欄（``name_keys``）を値ごと伏せた印
NAME_MARK = "〔名前〕"
#: アプリが作品の外で覚えている名前と場所を伏せた印
REMEMBERED_MARK = "〔ファイルの名前〕"


def file_secrets(
    projects: Project | Iterable[Project],
    *,
    project_path: Path | None,
    export_folder: Path | None,
    media: Iterable[NamedMedia] = (),
    names: Iterable[str] = (),
    remembered: Iterable[str] = (),
) -> dict[str, str]:
    """素材の名前と場所を伏せる表（伏せる文字 → 置き換える文字）

    名前の出どころはここに集める 今の作品だけでなく、履歴に残った前の作品も渡す
    （:attr:`~sashimono.core.commands.Document.past_projects`） 置いた後に消した素材や、
    名前を変える前の作品の名前は、履歴の名前や字幕起こしの記録に残ったまま外へ出るため
    作品の名前（:attr:`Project.name`）も入れる 未保存の作品や、保存した後に名前を変えた作品は、
    保存先の名前と違う 並べる順は :func:`secret_pairs` で決める

    ``media`` と ``names`` は文書が一度でも持った素材と作品の名前
    （:attr:`~sashimono.core.commands.Document.seen_media`） 履歴は上限で落ち、別の作品を
    開くと消えるので、履歴だけでは拾い切れない ``remembered`` はアプリが作品の外で覚えている
    名前と場所（:meth:`~sashimono.ai.host.EditorHost.remembered_names` 字幕起こしの記録・
    前に開いた作品の場所など）
    """
    table: dict[str, str] = {}

    def add(values: Iterable[str], mark: str) -> None:
        for value in values:
            # 短い名前も表に入れる 伏せ方は :func:`redact` が長さで変える
            if value.strip() and value not in table:
                table[value] = mark

    def add_media(item: NamedMedia) -> None:
        path = item.path
        add((str(path), path.as_posix(), item.name, path.name, path.stem), f"〔素材 {item.id}〕")

    for project in (projects,) if isinstance(projects, Project) else projects:
        for owned in project.media:
            add_media(owned)
        add((project.name,), PROJECT_MARK)
    for item in media:
        add_media(item)
    add(names, PROJECT_MARK)
    for value in remembered:
        place = Path(value)
        add((value, place.as_posix(), place.name, place.stem), REMEMBERED_MARK)
    if project_path is not None:
        add(
            (str(project_path), project_path.as_posix(), project_path.name, project_path.stem),
            PROJECT_MARK,
        )
        add((str(project_path.parent), project_path.parent.as_posix()), "〔作品のフォルダ〕")
    if export_folder is not None:
        add((str(export_folder), export_folder.as_posix()), "〔書き出しのフォルダ〕")
    return table


def secret_pairs(table: Mapping[str, str]) -> list[tuple[str, str]]:
    """伏せる順に並べた表 長い順

    長い順にするのは、場所の中の名前を先に置き換えると、場所の残りが伏せずに残るため
    """
    return sorted(table.items(), key=lambda item: len(item[0]), reverse=True)


#: 字幕と文字を伏せたときの印
TEXT_MARK = "〔伏せ字〕"


# --- 欄の型で伏せる（#391 の Codex の指摘）---
#
# 欄の値（テキストの文字・スクリプトの文字の欄・画像の場所など）は、文字列を表に集めて後から
# 照らす伏せ方では漏れる 名前の text だけを集めると別の名前の文字の欄が、2 文字より短い値は
# 表に入らず 1 文字の文が、場所の形と名前の形で見分けると「模様.png」のような相対の名前が、
# どれも伏せずに送られた（Codex の指摘が 2 回続いた） 値を返す道具は結果を組む所で欄の型を
# 引き、送らない設定ならその値そのものを印へ置き換える 長さや名前の形には頼らない
# 表の照らし（:func:`text_secrets` :func:`redact`）は、型の無い所（字幕・エラーの文）の補いに残す

#: 欄の型ごとに、値を送るのに要る許し この表が 1 か所の決まり（docs/development.md の
#: 「人が画面で見ている物を AI も読む」と ``tests/ai/test_screen_reading.py`` が同じ表を持つ）
#: 型を足すときは、人が書いた文か場所かを考えてここへ足す 書かない型は編集の構成として送る
FIELD_SENDS: dict[type, frozenset[Outbound]] = {
    TextSpec: frozenset({Outbound.TEXT}),
    FileSpec: frozenset({Outbound.FILES}),
}
#: 定義の分からない種類（読み込んでいないスクリプト）の文字の値 文か場所か分からないので、
#: 両方を許したときだけ送る
UNKNOWN_FIELD_SENDS = frozenset({Outbound.TEXT, Outbound.FILES})

#: 今動いている道具の結果にそのまま載せてよい物 ブリッジが道具を動かす間だけ立てる
#: 立っていないのは編集画面の中から直に呼んだときで、結果は外へ出ないので全部を載せる
_SENDABLE: ContextVar[frozenset[Outbound] | None] = ContextVar("sendable", default=None)


@contextmanager
def sending(kinds: frozenset[Outbound]) -> Iterator[None]:
    """この間に動く道具の結果へ、``kinds`` だけをそのまま載せる（ブリッジが道具を包む）"""
    token = _SENDABLE.set(kinds)
    try:
        yield
    finally:
        _SENDABLE.reset(token)


def sendable_now(kind: Outbound) -> bool:
    """今動いている道具の結果へ ``kind`` を載せてよいか 編集画面から直に呼んだときは真

    型の付いた欄（:func:`masked_value`）でも、表と形（:func:`hide_text`）でも伏せられない物を返す
    道具が、断られていれば欄ごと返さないために使う（計画の自由記述 #340）
    """
    allowed = _SENDABLE.get()
    return allowed is None or kind in allowed


def field_sends(spec: object | None, *, known: bool = True) -> frozenset[Outbound]:
    """欄の値を送るのに要る許し ``known`` が偽なら定義の分からない種類の欄"""
    if not known:
        return UNKNOWN_FIELD_SENDS
    for kind, needs in FIELD_SENDS.items():
        if isinstance(spec, kind):
            return needs
    return frozenset()


def masked_value(value: Any, needs: frozenset[Outbound]) -> Any:
    """送ってよくなければ、欄の値そのものを印へ置き換える 長さや形を問わない

    空の文字は何も載っていないので残す（印にすると、書いてあるように読める）
    文字でない値（数・色・印）は伏せない 文字の欄と場所の欄が持つのは文字だけ
    """
    if not needs or not isinstance(value, str) or value == "":
        return value
    allowed = _SENDABLE.get()
    if allowed is None or needs <= allowed:
        return value
    return TEXT_MARK if Outbound.TEXT in needs - allowed else PATH_MARK


#: 操作の名前の「動き: 対象」の対象のうち、伏せずに残す物 数と数え方だけ（「無音カット: 3 か所」）
#: 中身の文字ではないので、残しても字幕や本文は外へ出ない
_COUNT_ONLY = re.compile(r"[\d\s.,/]*(?:か所|件|本|個|コマ|フレーム|秒)?")
#: 名前の中のかぎ括弧 中は本文の引用（「選んでいるテキスト「…」を写す」など）
_QUOTED = re.compile(r"「[^」]*」")


def _typed_texts(params: Mapping[str, object], specs: Iterable[object] | None) -> Iterator[str]:
    """欄のうち、人が文字を書く欄（:class:`TextSpec`）の今の値

    名前ではなく欄の型で見分ける AviUtl のスクリプトは文字の欄に好きな名前を付け、テキストにも
    text のほかに文字の欄（タイマーの書式など）がある 名前の text だけを見ると、それらに本人が
    書いた文を、字幕と文字を送らないと決めた人の所でも送る（#391 の Codex の指摘）
    定義の分からない種類（読み込んでいないスクリプト）は、どの欄が文字か分からないので文字の値を
    全部数える 伏せすぎて困るのは値が読めないことだけで、外へ出た文は取り戻せない
    """
    if specs is None:
        yield from (value for value in params.values() if isinstance(value, str))
        return
    names = {spec.name for spec in specs if isinstance(spec, TextSpec)} | {"text"}
    yield from (value for name, value in params.items() if name in names and isinstance(value, str))


def _effect_texts(effects: Iterable[Effect]) -> Iterator[str]:
    for effect in effects:
        definition = registry.get(effect.kind)
        yield from _typed_texts(
            effect.params, definition.parameters if definition is not None else None
        )


def text_secrets(project: Project) -> list[str]:
    """字幕と文字を伏せる表 字幕起こしの文と、クリップの文字の欄の文（メインとシーン） 長い順

    文字の欄は中身・縁取りの層・エフェクト（場面切り替えの後の場面と、層に掛けた物も）のどれでも
    欄の型で拾う（:func:`_typed_texts`）
    """
    found: set[str] = set()
    for media in project.media:
        for _stream, transcript in media.transcripts:
            found.update(segment.text for segment in transcript.segments)
    for timeline in (project.timeline, *(scene.timeline for scene in project.scenes)):
        for track in timeline.tracks:
            for clip in track.clips:
                source = clip.source
                if source is not None:
                    definition = source_registry.get(source.kind)
                    found.update(
                        _typed_texts(
                            source.params,
                            definition.parameters if definition is not None else None,
                        )
                    )
                    for stroke in source.strokes:
                        found.update(_typed_texts(stroke.params, STROKE.parameters))
                        found.update(_effect_texts(stroke.effects))
                found.update(_effect_texts(clip.effects))
                found.update(_effect_texts(clip.after_effects))
    lines = {line for text in found for line in (text, *text.splitlines())}
    return sorted(
        (line for line in lines if len(line.strip()) >= _SHORTEST_SECRET),
        key=len,
        reverse=True,
    )


def hide_text(value: Any, secrets: Iterable[str], *, labels: bool = True) -> Any:
    """操作の名前（取り消しと履歴の道具の結果）の中の文字を伏せる 元の値は変えない

    表（今ある字幕と本文）に載った文字を伏せたうえで、「動き: 対象」の対象と、かぎ括弧の中を
    伏せる 表だけでは足りない 名前は文の頭（12 文字）で切ってあり、消したテキストや
    直す前の字幕の文も、表から消えた後まで履歴の名前に残る 動き（「テキストを追加」）は
    残すので、AI はどの操作を取り消したかを読める

    ``labels`` が偽なら表で伏せるだけにする 操作の名前でない結果（欄の値・検査の一覧・互換性
    レポート #391）は、値そのものを型で伏せてあり（:func:`masked_value`）、「: 」の後や括弧の中を
    伏せると、検査の説明や AviUtl の API の名前まで読めなくなる
    """
    pairs = list(secrets)

    def one(text: str) -> str:
        for secret in pairs:
            if secret in text:
                text = text.replace(secret, TEXT_MARK)
        if not labels:
            return text
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


def redact(value: Any, table: Iterable[tuple[str, str]], name_keys: Iterable[str] = ()) -> Any:
    """結果の中の文字を、表に従って伏せ字にする 辞書と並びは中まで辿る 元の値は変えない

    ``name_keys`` は名前そのものを返す欄（作品や素材の名前 道具ごとに
    :attr:`~sashimono.ai.operations.Operation.name_keys` で決める） 表に載っていなくても、
    短くても、値ごと伏せる
    """
    return _redactor(table, name_keys)(None, value)


def _redactor(
    table: Iterable[tuple[str, str]], name_keys: Iterable[str] = ()
) -> Callable[[object, Any], Any]:
    """:func:`redact` の伏せ方 欄の名前と値を受けて伏せた値を返す（欄が無ければ ``None``）

    下見（:func:`file_sample`）は欄ごとに伏せる前と後を並べる 欄ごとに :func:`redact` を
    呼ぶと、短い名前の正規表現を欄の数だけ作り直し、長い一覧で窓が出るまで待たせる
    """
    pairs = list(table)
    names = frozenset(name_keys)
    # 短い名前は、前後が語の続きでない所だけで伏せる（:data:`_SHORTEST_SECRET`）
    short = [
        (re.compile(rf"(?<!\w){re.escape(secret)}(?!\w)"), mark)
        for secret, mark in pairs
        if len(secret) < _SHORTEST_SECRET
    ]
    long = [(secret, mark) for secret, mark in pairs if len(secret) >= _SHORTEST_SECRET]

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            for secret, mark in long:
                if secret in item:
                    item = item.replace(secret, mark)
            for pattern, mark in short:
                item = pattern.sub(mark, item)
            return _ABSOLUTE_PATH.sub(_PATH_MARK, item)
        if isinstance(item, Mapping):
            return {walk(key): value_of(key, entry) for key, entry in item.items()}
        if isinstance(item, list | tuple):
            return [walk(entry) for entry in item]
        return item

    def value_of(key: object, entry: Any) -> Any:
        if not isinstance(key, str):
            return walk(entry)
        if key in names:
            return whole(walk(entry), NAME_MARK)
        if _PLACE_KEY.search(key):
            return whole(walk(entry), _PATH_MARK)
        return walk(entry)

    def whole(item: Any, mark: str) -> Any:
        """名前か場所の欄の値 表で伏せ字になった物（〔素材 …〕）は残し、ほかは値ごと伏せる"""
        if isinstance(item, str):
            return item if not item or item.startswith("〔") else mark
        if isinstance(item, list):
            return [whole(entry, mark) for entry in item]
        return item

    return value_of
