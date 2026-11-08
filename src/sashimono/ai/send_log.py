"""AI へ送った物の記録（#425）

確かめの窓で「送る」と答えた後や、設定で「尋ねずに送る」にした後は、何が・いつ・どれだけ
外へ出たかを見直す手段が無かった 道具の結果を AI へ返すたびに、日時・道具・種類・件数を
手元に残す

記録に持つのは種類と件数（行数・文字数・枚数・名前の数）だけで、中身は持たない
（2026-11 #425 で決めた） 中身を残すと、外へ出したくない人の字幕や場所の一覧が、記録として
この機械に積もり続ける 中身は「最後に送った物」（:class:`LastSent`）として、アプリを閉じるまで
だけ持つ 尋ねない設定にした人も、何が出ているかを設定画面で確かめられる

記録は本人の設定の置き場の ``ai_sends.json`` に置く（:class:`SendLogStore`） 外へは送らない
読めない・書けないときは残さないだけで、AI の作業は止めない（使った量の置き場と同じ考え方）

このモジュールは Qt を知らない 試験と CLI からも同じ物を使う
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from sashimono.ai.outbound import ASKED, Outbound, Sample
from sashimono.core.io.atomic import held_for_update, write_atomic

__all__ = [
    "KEEP_DAYS",
    "KEEP_ENTRIES",
    "LastSent",
    "MemoryLogStore",
    "SendLog",
    "SendLogStore",
    "SentEntry",
    "SentKind",
]

#: 記録を残す日数 これより古い物は書くときに落とす 見直したいのは最近の物で、
#: 古い記録を持ち続けても、いつ何を使ったかの履歴が積もるだけになる
KEEP_DAYS = 90
#: 記録の件数の上限 道具を何百回も呼ぶ日が続いても、ファイルが膨らみ続けないように
KEEP_ENTRIES = 2000


@dataclass(frozen=True, slots=True)
class SentKind:
    """送った種類 1 つの件数"""

    kind: Outbound
    count: int
    chars: int = 0

    @property
    def amount(self) -> str:
        return Sample(self.kind, count=self.count, chars=self.chars).amount


@dataclass(frozen=True, slots=True)
class SentEntry:
    """送った 1 回 中身は持たない"""

    at: datetime
    tool: str
    kinds: tuple[SentKind, ...]

    def text(self) -> str:
        """一覧に出す 1 行"""
        what = " ".join(f"{sent.kind.label} {sent.amount}" for sent in self.kinds)
        return f"{self.at:%Y-%m-%d %H:%M:%S} {self.tool} {what}"


@dataclass(frozen=True, slots=True)
class LastSent:
    """種類ごとに、最後に送った物の下見 アプリを閉じるまでだけ持つ"""

    at: datetime
    tool: str
    sample: Sample


class _Store(Protocol):
    def load(self) -> list[SentEntry]: ...

    def add(self, entry: SentEntry) -> bool: ...

    def clear(self) -> bool: ...


def _entry_json(entry: SentEntry) -> dict[str, Any]:
    return {
        "at": entry.at.isoformat(timespec="seconds"),
        "tool": entry.tool,
        "kinds": {
            sent.kind.value: {"count": sent.count, "chars": sent.chars} for sent in entry.kinds
        },
    }


def _entry_of(value: object) -> SentEntry | None:
    """ファイルの 1 件 壊れた物は落とす（手で書き換えられても起動と AI を止めない）"""
    if not isinstance(value, dict):
        return None
    at, tool, kinds = value.get("at"), value.get("tool"), value.get("kinds")
    if not isinstance(at, str) or not isinstance(tool, str) or not isinstance(kinds, dict):
        return None
    try:
        when = datetime.fromisoformat(at)
    except ValueError:
        return None
    found: list[SentKind] = []
    for kind in ASKED:
        raw = kinds.get(kind.value)
        if not isinstance(raw, dict):
            continue
        count, chars = raw.get("count"), raw.get("chars", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            continue
        if isinstance(chars, bool) or not isinstance(chars, int) or chars < 0:
            chars = 0
        found.append(SentKind(kind, count, chars))
    if not found:
        return None
    return SentEntry(when, tool, tuple(found))


class SendLogStore:
    """記録のファイル 形は ``{"entries": [{"at": …, "tool": …, "kinds": {…}}]}``

    書くのは足す 1 件だけ 読んで足して書き戻すまでを錠で守る 2 つの窓が自分の記録を
    書き戻すと、後から書いた方が先の窓の分を消す
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[SentEntry]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        raw = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        return [entry for item in raw if (entry := _entry_of(item)) is not None]

    def add(self, entry: SentEntry) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with held_for_update(self._lock_path()):
                kept = _trimmed([*self.load(), entry], entry.at)
                self._write(kept)
        except OSError:
            return False
        return True

    def clear(self) -> bool:
        try:
            with held_for_update(self._lock_path()):
                self.path.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    def _lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def _write(self, entries: Iterable[SentEntry]) -> None:
        data = {"entries": [_entry_json(entry) for entry in entries]}
        write_atomic(self.path, json.dumps(data, ensure_ascii=False, indent=1))


def _trimmed(entries: list[SentEntry], now: datetime) -> list[SentEntry]:
    oldest = now - timedelta(days=KEEP_DAYS)
    return [entry for entry in entries if entry.at > oldest][-KEEP_ENTRIES:]


class MemoryLogStore:
    """覚えない置き場（試験・CLI） 記録はこの置き場が生きている間だけ持つ"""

    def __init__(self) -> None:
        self._entries: list[SentEntry] = []

    def load(self) -> list[SentEntry]:
        return list(self._entries)

    def add(self, entry: SentEntry) -> bool:
        self._entries = _trimmed([*self._entries, entry], entry.at)
        return True

    def clear(self) -> bool:
        self._entries = []
        return True


class SendLog:
    """送った物の記録と、最後に送った物の下見

    足すのは AI のスレッド、読むのは画面のスレッド 最後に送った物は錠で守る
    """

    def __init__(self, store: _Store | None = None) -> None:
        self.store: _Store = store if store is not None else MemoryLogStore()
        #: 記録を残すか（設定） 切っても最後に送った物は持つ 画面で確かめる手段は残す
        self.keep = True
        self._last: dict[Outbound, LastSent] = {}
        self._lock = threading.Lock()

    def add(self, tool: str, samples: Iterable[Sample], at: datetime | None = None) -> None:
        """送った 1 回を残す 件数が 0 の種類（何も入っていなかった）は残さない"""
        when = at if at is not None else datetime.now().astimezone()
        sent = [sample for sample in samples if sample.count > 0]
        if not sent:
            return
        with self._lock:
            for sample in sent:
                self._last[sample.kind] = LastSent(when, tool, sample)
        if self.keep:
            self.store.add(
                SentEntry(
                    when,
                    tool,
                    tuple(SentKind(sample.kind, sample.count, sample.chars) for sample in sent),
                )
            )

    def last(self) -> dict[Outbound, LastSent]:
        """種類ごとの最後に送った物（:data:`ASKED` の順）"""
        with self._lock:
            return {kind: self._last[kind] for kind in ASKED if kind in self._last}

    def entries(self) -> list[SentEntry]:
        """残した記録 新しい順"""
        return list(reversed(self.store.load()))

    def clear(self) -> bool:
        """記録を消す 最後に送った物も忘れる 消した後に画面に残っていると消えたように見えない"""
        with self._lock:
            self._last.clear()
        return self.store.clear()
