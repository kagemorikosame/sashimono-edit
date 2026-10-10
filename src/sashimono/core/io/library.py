"""保存したプリセットと自作のエイリアスの整理（#276）

名前の変更・分類の変更・複製・削除（ごみ箱）・書き出し・読み込みを、2 つの置き場
（:class:`~sashimono.core.io.presets.PresetStore` と
:class:`~sashimono.core.io.aliases.AliasStore`）へ同じ形で行う 画面は管理の窓
（``ui/library_dialog.py``）1 つで、ここはファイルの扱いだけを持つ

決まり

* **消すときはごみ箱へ移す** 置き場の中の :data:`~sashimono.core.io.presets.TRASH_FOLDER`
  へ、置き場からの相対の道のまま移し、管理の窓の〔元に戻す〕で戻せる プリセットも
  エイリアスも作り直せない物で、確かめの窓だけでは押し間違いを戻せない
  ごみ箱は時間で勝手に空にしない（黙って消さない） 空にするのは本人が窓から選んだときだけ
* **上書きで消える物もごみ箱へ移す** 名前の変更・分類の変更・読み込みで同じ名前の物を
  上書きすると、前の物は戻せないまま消える 上書きを選んだ後でも戻せるようにする
* **ファイルの形は変えない** 書き出しは置き場に書くのと同じ ``.smep`` / ``.smea``
  旧い拡張子（``.kmkp`` ``.nvpreset``）の物も一覧に出し、書き直すときは新しい形で書く
* **ほかの窓が先に変えていたら断る** 同じ置き場を 2 つの窓（2 つの起動）から整理できる
  一覧を出した後で元のファイルが無くなっていたら :class:`LibraryChangedError` を投げ、
  画面は一覧を読み直す 黙って作り直すと、ほかの窓で消した物が戻ってくる
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, Literal

from sashimono.core.io.aliases import SUFFIX as ALIAS_SUFFIX
from sashimono.core.io.aliases import Alias, AliasStore
from sashimono.core.io.presets import LEGACY_SUFFIXES as PRESET_LEGACY_SUFFIXES
from sashimono.core.io.presets import SUFFIX as PRESET_SUFFIX
from sashimono.core.io.presets import TRASH_FOLDER, Preset, PresetStore
from sashimono.core.io.serialize import ProjectFileError, json_text

__all__ = [
    "ALIAS",
    "PRESET",
    "Library",
    "LibraryChangedError",
    "LibraryConflictError",
    "LibraryEntry",
    "LibraryError",
    "LibraryKind",
    "TrashEntry",
    "kind_of_file",
    "look_fingerprint",
]

LibraryKind = Literal["preset", "alias"]
PRESET: Final = "preset"
ALIAS: Final = "alias"

#: ごみ箱の 1 件ごとの覚え書き 何をどこから消したか
_TRASH_NOTE = "entry.json"
#: ごみ箱の 1 件の中で、消したファイルを置く所
_TRASH_FILES = "files"

Item = Preset | Alias


class LibraryError(Exception):
    """整理できなかった理由 画面へそのまま出せる文"""


class LibraryConflictError(LibraryError):
    """同じ名前（同じファイル）の物がもうある 上書きしてよいかは画面が尋ねる"""


class LibraryChangedError(LibraryError):
    """一覧を出した後で元のファイルが無くなった ほかの窓が消したか名前を変えた"""


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    """一覧の 1 件 ファイルの場所を持つので、名前や分類の重なりで別の物を取り違えない"""

    kind: LibraryKind
    item: Item
    #: 中身を読んだファイル
    path: Path
    #: 最後に保存した時刻（秒） 名前や分類を変えても動かさない（並べ替えの「保存した日」）
    modified: float
    #: 同じ名前の旧い拡張子のファイル 一覧には出ないが、同じプリセットとして一緒に扱う
    #: 残すと、名前を変えたり消したりした後に、隠れていた旧い方が一覧に戻ってくる
    shadows: tuple[Path, ...] = ()

    @property
    def name(self) -> str:
        return self.item.name

    @property
    def category(self) -> str:
        return self.item.category

    @property
    def legacy(self) -> bool:
        """改名前の旧い拡張子のファイルか 書き直すと新しい形になる"""
        return self.path.suffix != (PRESET_SUFFIX if self.kind == PRESET else ALIAS_SUFFIX)

    @property
    def files(self) -> tuple[Path, ...]:
        return (self.path, *self.shadows)


@dataclass(frozen=True, slots=True)
class TrashEntry:
    """ごみ箱の 1 件"""

    kind: LibraryKind
    #: ごみ箱の中のこの 1 件のフォルダ
    folder: Path
    name: str
    category: str
    #: 消した時刻（秒）
    deleted_at: float
    #: 置き場からの相対の道 戻すときはここへ戻す
    files: tuple[str, ...]


def kind_of_file(path: Path) -> LibraryKind | None:
    """拡張子から、どちらの置き場の物か 知らない拡張子は ``None``"""
    suffix = path.suffix.lower()
    if suffix in (PRESET_SUFFIX, *PRESET_LEGACY_SUFFIXES):
        return PRESET
    if suffix == ALIAS_SUFFIX:
        return ALIAS
    return None


def look_fingerprint(item: Item) -> str:
    """見た目の中身の指紋 見本の絵（#277）の鍵に使う

    名前と分類は入れない 名前を変えただけで絵を作り直すと、整理するたびに一覧が
    描き直しで重くなる 中身（エフェクト・文字・値）が 1 つでも変われば指紋も変わる
    """
    data = item.to_dict()
    for key in ("name", "category"):
        data.pop(key, None)
    # クリップとエフェクトの ID も外す 見た目に関係が無く、同じ見た目を保存し直すたびに
    # 振り直されるので、入れると同じ絵を何度も描き直す
    text = json_text(_without_ids(data), indent=None)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(slots=True)
class Library:
    """2 つの置き場をまとめて整理する係"""

    presets: PresetStore = field(default_factory=PresetStore)
    aliases: AliasStore = field(default_factory=AliasStore)

    def root(self, kind: LibraryKind) -> Path:
        return self.presets.root if kind == PRESET else self.aliases.root

    # --- 一覧 ---

    def entries(self, kind: LibraryKind) -> tuple[LibraryEntry, ...]:
        """読める物だけを名前の順に 壊れた 1 つで一覧全体を出なくしない（置き場と同じ）"""
        found: list[LibraryEntry] = []
        for path in self._files(kind):
            try:
                item = self._load(kind, path)
                modified = path.stat().st_mtime
            except (LibraryError, OSError):
                continue
            found.append(
                LibraryEntry(kind, item, path, modified, shadows=self._shadows(kind, path))
            )
        return tuple(sorted(found, key=lambda entry: (entry.name, entry.category)))

    def categories(self, kind: LibraryKind) -> tuple[str, ...]:
        return tuple(sorted({entry.category for entry in self.entries(kind)}))

    # --- 名前・分類・複製 ---

    def rename(self, entry: LibraryEntry, name: str, *, overwrite: bool = False) -> LibraryEntry:
        """名前を変える 同じ名前があれば :class:`LibraryConflictError`（``overwrite`` で上書き）"""
        return self._rewrite(entry, _with(entry.item, name=_cleaned(name)), overwrite=overwrite)

    def recategorize(
        self, entry: LibraryEntry, category: str, *, overwrite: bool = False
    ) -> LibraryEntry:
        """分類を変える 新しい分類の名前を渡せば、その分類ができる"""
        changed = _with(entry.item, category=_cleaned(category))
        return self._rewrite(entry, changed, overwrite=overwrite)

    def duplicate(self, entry: LibraryEntry, name: str, *, overwrite: bool = False) -> LibraryEntry:
        """別の名前で写しを作る 元はそのまま残す"""
        self._require(entry)
        copy = _with(entry.item, name=_cleaned(name))
        self._make_room(entry.kind, copy, keep=(), overwrite=overwrite)
        path = self._save(copy)
        return self._entry(entry.kind, copy, path)

    def free_name(self, kind: LibraryKind, item: Item) -> str:
        """同じ名前の物と重ならない名前（``名前 (2)`` のように番号を足す） 読み込みで両方残すとき"""
        if not self._occupants(kind, item, keep=()):
            return item.name
        number = 2
        while True:
            name = f"{item.name} ({number})"
            if not self._occupants(kind, _with(item, name=name), keep=()):
                return name
            number += 1

    # --- 削除とごみ箱 ---

    def delete(self, entry: LibraryEntry) -> TrashEntry:
        """ごみ箱へ移す 管理の窓の〔元に戻す〕（:meth:`restore`）で戻せる"""
        self._require(entry)
        return self._to_trash(entry.kind, entry.item, entry.files)

    def trashed(self, kind: LibraryKind) -> tuple[TrashEntry, ...]:
        """ごみ箱の中身 新しく消した順 覚え書きの読めない物は飛ばす"""
        trash = self.root(kind) / TRASH_FOLDER
        if not trash.is_dir():
            return ()
        found: list[TrashEntry] = []
        for folder in trash.iterdir():
            note = _read_note(kind, folder)
            if note is not None:
                found.append(note)
        return tuple(sorted(found, key=lambda entry: entry.deleted_at, reverse=True))

    def restore(self, trashed: TrashEntry, *, overwrite: bool = False) -> LibraryEntry:
        """ごみ箱から元の場所へ戻す 同じ名前の物が後から作られていれば :class:`LibraryConflictError`

        ``overwrite`` なら、いまある方をごみ箱へ移してから戻す
        """
        root = self.root(trashed.kind)
        sources = [trashed.folder / _TRASH_FILES / relative for relative in trashed.files]
        if not trashed.folder.is_dir() or not all(source.is_file() for source in sources):
            raise LibraryChangedError("ごみ箱の中身が見つからない（ほかの窓で戻したか、空にした）")
        item = self._load(trashed.kind, sources[0])
        targets = [root / relative for relative in trashed.files]
        # 戻す先そのものと、同じ名前で後から保存した物（旧い拡張子なら新しい方）の両方を見る
        occupants = self._occupants(trashed.kind, item, keep=())
        occupants.extend(t for t in targets if t.is_file() and t not in occupants)
        if occupants:
            if not overwrite:
                raise LibraryConflictError(f"「{item.name}」がもうある")
            self._to_trash(trashed.kind, self._load_or(trashed.kind, occupants[0], item), occupants)
        for source, target in zip(sources, targets, strict=True):
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
        shutil.rmtree(trashed.folder, ignore_errors=True)
        return self._entry(trashed.kind, item, targets[0])

    def empty_trash(self, kind: LibraryKind, entries: Iterable[TrashEntry] | None = None) -> int:
        """ごみ箱を空にする（``entries`` を渡せばその物だけ） 戻せなくなる 消した件数を返す"""
        chosen = tuple(entries) if entries is not None else self.trashed(kind)
        count = 0
        for trashed in chosen:
            if trashed.folder.is_dir():
                shutil.rmtree(trashed.folder)
                count += 1
        return count

    # --- 書き出しと読み込み ---

    def export(self, entry: LibraryEntry, folder: Path, *, overwrite: bool = False) -> Path:
        """``folder`` へ置き場と同じ形のファイルを書く 旧い拡張子の物も新しい形で書く"""
        self._require(entry)
        target = Path(folder) / self._target(entry.item).name
        if target.exists() and not overwrite:
            raise LibraryConflictError(f"{target.name} がもうある")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".writing")
        temporary.write_text(json_text(entry.item.to_dict()), encoding="utf-8")
        temporary.replace(target)
        return target

    def read_file(self, path: Path) -> tuple[LibraryKind, Item]:
        """読み込む前に中身を確かめる 形の違う物・新しい版の物は :class:`LibraryError`"""
        kind = kind_of_file(Path(path))
        if kind is None:
            raise LibraryError(f"プリセット（.smep）でもエイリアス（.smea）でもない: {path}")
        return kind, self._load(kind, Path(path))

    def occupied(self, kind: LibraryKind, item: Item) -> bool:
        """同じ名前（同じファイル）の物が置き場にあるか"""
        return bool(self._occupants(kind, item, keep=()))

    def add(self, kind: LibraryKind, item: Item, *, overwrite: bool = False) -> LibraryEntry:
        """置き場へ足す（読み込み） 同じ名前があれば :class:`LibraryConflictError`"""
        self._make_room(kind, item, keep=(), overwrite=overwrite)
        return self._entry(kind, item, self._save(item))

    # --- 中の手順 ---

    def _rewrite(self, entry: LibraryEntry, changed: Item, *, overwrite: bool) -> LibraryEntry:
        """名前や分類を変えて書き直す 保存した日は動かさない"""
        self._require(entry)
        if changed == entry.item and not entry.legacy:
            return entry
        self._make_room(entry.kind, changed, keep=entry.files, overwrite=overwrite)
        stat = entry.path.stat()
        target = self._target(changed)
        same = target.exists() and _same_file(target, entry.path)
        path = self._save(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        if not same:
            entry.path.unlink(missing_ok=True)
        if entry.shadows:
            # 隠れていた旧い拡張子の写し 消すと戻せないので、ごみ箱へ移す
            self._to_trash(entry.kind, entry.item, entry.shadows)
        return self._entry(entry.kind, changed, path)

    def _make_room(
        self, kind: LibraryKind, item: Item, *, keep: tuple[Path, ...], overwrite: bool
    ) -> None:
        occupants = self._occupants(kind, item, keep=keep)
        if not occupants:
            return
        if not overwrite:
            where = f"「{item.category}」に" if kind == PRESET else ""
            raise LibraryConflictError(f"{where}「{item.name}」がもうある")
        # 上書きで消える方もごみ箱へ 上書きを選んだ後でも戻せるようにする
        self._to_trash(kind, self._load_or(kind, occupants[0], item), occupants)

    def _occupants(self, kind: LibraryKind, item: Item, *, keep: tuple[Path, ...]) -> list[Path]:
        """``item`` を書くと重なる、いまある別のファイル ``keep`` は自分自身なので数えない"""
        target = self._target(item)
        candidates = [target]
        if kind == PRESET:
            candidates.extend(target.with_suffix(suffix) for suffix in PRESET_LEGACY_SUFFIXES)
        return [
            path
            for path in candidates
            if path.is_file() and not any(_same_file(path, own) for own in keep)
        ]

    def _to_trash(self, kind: LibraryKind, item: Item, files: Iterable[Path]) -> TrashEntry:
        root = self.root(kind)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = root / TRASH_FOLDER / f"{stamp}-{uuid.uuid4().hex[:8]}"
        moving = [Path(path) for path in files]
        relatives = tuple(path.relative_to(root).as_posix() for path in moving)
        entry = TrashEntry(kind, folder, item.name, item.category, time.time(), relatives)
        folder.mkdir(parents=True)
        # 覚え書きを先に書く 移している途中で落ちても、移した物がごみ箱の一覧に出る
        (folder / _TRASH_NOTE).write_text(
            json_text(
                {
                    "kind": kind,
                    "name": entry.name,
                    "category": entry.category,
                    "deleted_at": entry.deleted_at,
                    "files": list(relatives),
                }
            ),
            encoding="utf-8",
        )
        moved: list[tuple[Path, Path]] = []
        try:
            for path, relative in zip(moving, relatives, strict=True):
                destination = folder / _TRASH_FILES / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
                moved.append((destination, path))
        except OSError as exc:
            # 途中で移せなかった（ほかのソフトが開いている など） 半分だけ消えた形を残さない
            for destination, original in reversed(moved):
                destination.replace(original)
            shutil.rmtree(folder, ignore_errors=True)
            raise LibraryError(f"ごみ箱へ移せなかった: {exc}") from exc
        return entry

    def _require(self, entry: LibraryEntry) -> None:
        if not entry.path.is_file():
            raise LibraryChangedError(
                f"「{entry.name}」が見つからない（ほかの窓で消したか、名前を変えた）"
            )

    def _files(self, kind: LibraryKind) -> list[Path]:
        if kind == PRESET:
            # 旧い拡張子を拾う決まり（新しい方が勝つ）は置き場と同じ物を使う
            return self.presets.files()
        root = self.aliases.root
        return sorted(root.glob(f"*{ALIAS_SUFFIX}")) if root.is_dir() else []

    def _shadows(self, kind: LibraryKind, path: Path) -> tuple[Path, ...]:
        if kind != PRESET or path.suffix != PRESET_SUFFIX:
            return ()
        return tuple(
            sibling
            for sibling in (path.with_suffix(suffix) for suffix in PRESET_LEGACY_SUFFIXES)
            if sibling.is_file()
        )

    def _target(self, item: Item) -> Path:
        if isinstance(item, Preset):
            return self.presets.path_for(item)
        return self.aliases.path_for(item.name)

    def _save(self, item: Item) -> Path:
        if isinstance(item, Preset):
            return self.presets.save(item)
        return self.aliases.save(item)

    def _load(self, kind: LibraryKind, path: Path) -> Item:
        try:
            if kind == PRESET:
                return self.presets.load(path)
            return self.aliases.load(path)
        except ProjectFileError as exc:
            raise LibraryError(str(exc)) from exc
        except (ValueError, TypeError, KeyError) as exc:
            # 中の値の形が違う（手で書き換えた・別のソフトのファイル） 読めない物として扱う
            raise LibraryError(f"読めない: {path} ({exc})") from exc

    def _load_or(self, kind: LibraryKind, path: Path, fallback: Item) -> Item:
        """ごみ箱の覚え書きに使う名前 読めない物でも移せるよう、読めなければ代わりを使う"""
        try:
            return self._load(kind, path)
        except LibraryError:
            return fallback

    def _entry(self, kind: LibraryKind, item: Item, path: Path) -> LibraryEntry:
        return LibraryEntry(
            kind, item, path, path.stat().st_mtime, shadows=self._shadows(kind, path)
        )


def _without_ids(value: object) -> object:
    if isinstance(value, dict):
        return {key: _without_ids(inner) for key, inner in value.items() if key != "id"}
    if isinstance(value, list):
        return [_without_ids(inner) for inner in value]
    return value


def _with(item: Item, *, name: str | None = None, category: str | None = None) -> Item:
    if isinstance(item, Preset):
        return replace(
            item,
            name=item.name if name is None else name,
            category=item.category if category is None else category,
        )
    return replace(
        item,
        name=item.name if name is None else name,
        category=item.category if category is None else category,
    )


def _cleaned(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise LibraryError("名前が空")
    return cleaned


def _same_file(first: Path, second: Path) -> bool:
    """同じファイルか Windows は大文字と小文字を区別しないので、道の文字では比べない"""
    try:
        return first.samefile(second)
    except OSError:
        return first == second


def _read_note(kind: LibraryKind, folder: Path) -> TrashEntry | None:
    try:
        data = json.loads((folder / _TRASH_NOTE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("kind") != kind:
        return None
    files = data.get("files")
    deleted_at = data.get("deleted_at")
    if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
        return None
    if not isinstance(deleted_at, int | float):
        return None
    return TrashEntry(
        kind,
        folder,
        str(data.get("name", "")),
        str(data.get("category", "")),
        float(deleted_at),
        tuple(files),
    )
