"""登録したフォルダの素材（エクスプローラー #520 #521 #522）

利用者が効果音・BGM・動画・画像などの分類を付けてフォルダを登録し、その中の素材を
アプリのエクスプローラーで探して置く AI アシスタントは、利用者が AI に使わせると選んだ分類の
フォルダからだけ効果音と BGM を選ぶ

素材はアプリに同梱せず、ネットからも取らない（2026-11-24 本人の決定） 権利の確かめは
利用者が登録するときに済ませる形にし、配る物に他人の素材を入れない

分類はフォルダごとに 1 つ（同じく本人の決定） ファイルごとのタグにすると、数千の効果音に
1 つずつ付ける手間が要り、付け忘れた物を AI が見つけられない フォルダの並びは利用者が
もう整えていることが多いので、そこへ分類を 1 つ付けるだけにする

AI が選ぶ手がかりは、ファイルとフォルダの名前と、測って覚えた長さと音の大きさ（:class:`Measure`）
雰囲気（明るい・緊張など）の見積もりはしない（同じく本人の決定 見積もりの確かめ方が無い）

ここは Qt もデコーダも知らない 測る関数は呼び手（画面と AI の口）が渡す
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sashimono.core.io.atomic import write_atomic

__all__ = [
    "AUDIO_SUFFIXES",
    "CATEGORIES",
    "CATEGORY_IMAGE",
    "CATEGORY_MUSIC",
    "CATEGORY_OTHER",
    "CATEGORY_SOUND_EFFECT",
    "CATEGORY_VIDEO",
    "DEFAULT_AI_CATEGORIES",
    "IMAGE_SUFFIXES",
    "KIND_IMAGE",
    "KIND_SOUND",
    "KIND_VIDEO",
    "SCAN_LIMIT",
    "VIDEO_SUFFIXES",
    "FolderListing",
    "FolderScan",
    "LibraryFile",
    "LibraryFolder",
    "MaterialLibrary",
    "Measure",
    "MeasureBook",
    "add_folder",
    "file_id",
    "kind_of",
    "list_folder",
    "match_files",
    "parse_categories",
    "parse_folders",
    "remove_folder",
    "scan_folder",
    "set_category",
]

CATEGORY_SOUND_EFFECT = "sound_effect"
CATEGORY_MUSIC = "music"
CATEGORY_VIDEO = "video"
CATEGORY_IMAGE = "image"
CATEGORY_OTHER = "other"

#: 分類と表示名 並びは画面の選択肢の並び（よく使う音を先に）
CATEGORIES: dict[str, str] = {
    CATEGORY_SOUND_EFFECT: "効果音",
    CATEGORY_MUSIC: "BGM",
    CATEGORY_VIDEO: "動画",
    CATEGORY_IMAGE: "画像",
    CATEGORY_OTHER: "その他",
}

#: AI が使ってよい分類の既定 効果音と BGM だけ（2026-11-24 本人の決定） 動画と画像は
#: 作品の絵そのものを変えるので、利用者が選んだときだけ使わせる
DEFAULT_AI_CATEGORIES: tuple[str, ...] = (CATEGORY_SOUND_EFFECT, CATEGORY_MUSIC)

AUDIO_SUFFIXES = (".wav", ".mp3", ".aac", ".flac", ".m4a", ".ogg", ".opus")
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

#: 分類ごとに一覧へ出すファイルの後ろ 効果音のフォルダに混ざった説明の画像や
#: 利用規約の文を出すと、AI が音でない物を選んで置こうとする
_SUFFIXES: dict[str, frozenset[str]] = {
    CATEGORY_SOUND_EFFECT: frozenset(AUDIO_SUFFIXES),
    CATEGORY_MUSIC: frozenset(AUDIO_SUFFIXES),
    CATEGORY_VIDEO: frozenset(VIDEO_SUFFIXES),
    CATEGORY_IMAGE: frozenset(IMAGE_SUFFIXES),
    CATEGORY_OTHER: frozenset(AUDIO_SUFFIXES + VIDEO_SUFFIXES + IMAGE_SUFFIXES),
}

#: 1 つのフォルダから拾うファイルの上限 ドライブの頭のような大きすぎる所を登録されても、
#: 一覧を作るたびに画面が止まらないようにする 市販の効果音集でも 1 万を超える物は少ない
SCAN_LIMIT = 20000


@dataclass(frozen=True, slots=True)
class LibraryFolder:
    """登録したフォルダ 1 つ 設定（``Preferences.library_folders``）にそのまま書く

    欄の名前を ``path`` にしているのは、AI へ送る設定の写しで名前と場所として伏せる
    （:mod:`sashimono.ai.outbound` が ``path`` の値を値ごと伏せる）ため
    """

    path: str
    category: str

    @property
    def category_name(self) -> str:
        return CATEGORIES.get(self.category, self.category)

    @property
    def label(self) -> str:
        """画面の選択肢に出す名前 分類と、フォルダの名前（場所の最後）"""
        return f"{self.category_name}｜{Path(self.path).name or self.path}"


def file_id(path: Path) -> str:
    """登録したフォルダのファイルを AI が指す印

    場所そのものを渡させないのは、名前と場所を送らないと決めた人の所で AI が伏せ字の
    場所を読めず、選んだファイルを指せなくなるため 大文字と小文字を区別しない Windows で
    同じファイルが別の印にならないよう、場所はそろえてから数える
    """
    key = os.path.normcase(os.path.normpath(Path(path).absolute()))
    return "lib_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class LibraryFile:
    """登録したフォルダの中のファイル 1 つ"""

    path: Path
    folder: LibraryFolder
    size: int
    modified_ns: int

    @property
    def id(self) -> str:
        return file_id(self.path)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def category(self) -> str:
        return self.folder.category

    @property
    def relative(self) -> str:
        """登録したフォルダから見た場所 名前で絞るときに、下のフォルダの名前（「爆発」など）
        も手がかりにするため
        """
        try:
            return str(self.path.relative_to(self.folder.path))
        except ValueError:
            return self.path.name

    @property
    def subfolder(self) -> str:
        """ファイルが入っているフォルダの名前 返事に「どのフォルダの物か」を書くのに使う"""
        return self.path.parent.name


@dataclass(frozen=True, slots=True)
class FolderScan:
    """フォルダ 1 つを見て回った結果

    ``missing`` は登録した場所が無い（消えた・移った・外付けのドライブを外した）
    落とさずに知らせ、登録は残す ドライブを挿し直せば戻る物を、勝手に外さない
    """

    folder: LibraryFolder
    files: tuple[LibraryFile, ...] = ()
    missing: bool = False
    #: :data:`SCAN_LIMIT` で打ち切った
    truncated: bool = False
    #: 読めなかった訳（権利が無い など） 無ければ空
    error: str = ""


def scan_folder(folder: LibraryFolder, limit: int = SCAN_LIMIT) -> FolderScan:
    """``folder`` の中（下のフォルダも）を見て回り、分類に合うファイルを拾う

    ``os.scandir`` で 1 段ずつ見る ``Path.rglob`` は 1 ファイルごとに ``stat`` を呼び直し、
    数千のファイルで目に見えて遅い リンクを辿った先のフォルダには入らない 自分を指す
    リンクがあると、見て回るのが終わらない
    """
    root = Path(folder.path)
    if not root.is_dir():
        return FolderScan(folder, missing=True)
    suffixes = _SUFFIXES.get(folder.category, _SUFFIXES[CATEGORY_OTHER])
    found: list[LibraryFile] = []
    pending = [root]
    error = ""
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                listed = sorted(entries, key=lambda entry: entry.name.casefold())
        except OSError as exc:
            # 下のフォルダの 1 つが読めないだけで全体を空にすると、ほかの素材まで見えなくなる
            error = error or f"読めないフォルダがあります: {exc.strerror or exc}"
            continue
        folders: list[Path] = []
        for entry in listed:
            try:
                if entry.is_dir(follow_symlinks=False):
                    folders.append(Path(entry.path))
                    continue
                if Path(entry.name).suffix.lower() not in suffixes:
                    continue
                if not entry.is_file():
                    continue
                stat = entry.stat()
            except OSError:
                continue
            found.append(LibraryFile(Path(entry.path), folder, stat.st_size, stat.st_mtime_ns))
            if len(found) >= limit:
                return FolderScan(folder, tuple(found), truncated=True, error=error)
        # 後で取り出す順が名前の順になるよう逆に積む
        pending.extend(reversed(folders))
    return FolderScan(folder, tuple(found), error=error)


@dataclass(frozen=True, slots=True)
class FolderListing:
    """フォルダ 1 段の中身（エクスプローラーの木で開いたフォルダ） 下のフォルダには入らない"""

    files: tuple[LibraryFile, ...] = ()
    folders: tuple[Path, ...] = ()
    missing: bool = False


def list_folder(folder: LibraryFolder, directory: Path, limit: int = SCAN_LIMIT) -> FolderListing:
    """登録した ``folder`` の中の ``directory`` 1 段の、分類に合うファイルと下のフォルダ

    木で開いたフォルダの中身を出すのに使う 下まで全部を見て回ると、根を開くたびに
    数万のファイルを待つ 名前で絞るときだけ :func:`scan_folder` で下まで見る
    """
    if not directory.is_dir():
        return FolderListing(missing=True)
    suffixes = _SUFFIXES.get(folder.category, _SUFFIXES[CATEGORY_OTHER])
    files: list[LibraryFile] = []
    folders: list[Path] = []
    try:
        with os.scandir(directory) as entries:
            listed = sorted(entries, key=lambda entry: entry.name.casefold())
    except OSError:
        return FolderListing(missing=True)
    for entry in listed:
        try:
            if entry.is_dir(follow_symlinks=False):
                folders.append(Path(entry.path))
                continue
            if Path(entry.name).suffix.lower() not in suffixes or not entry.is_file():
                continue
            stat = entry.stat()
        except OSError:
            continue
        if len(files) < limit:
            files.append(LibraryFile(Path(entry.path), folder, stat.st_size, stat.st_mtime_ns))
    return FolderListing(tuple(files), tuple(folders))


#: ファイルの種類（エクスプローラーの種類の絞り込みと列）
KIND_SOUND = "sound"
KIND_VIDEO = "video"
KIND_IMAGE = "image"


def kind_of(path: Path) -> str:
    """後ろから見た種類 開かずに決める（数千の行を並べるたびに開くと待たせる）"""
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return KIND_VIDEO
    if suffix in IMAGE_SUFFIXES:
        return KIND_IMAGE
    return KIND_SOUND


def match_files(files: Iterable[LibraryFile], query: str) -> list[LibraryFile]:
    """名前で絞る 空白で区切った語をどれも含む物（大文字と小文字・全角と半角の英字は区別しない）

    下のフォルダの名前も見る（「爆発/ドカン.wav」は「爆発」で引ける）
    """
    words = [_normalized(word) for word in query.split()]
    if not words:
        return list(files)
    return [file for file in files if all(word in _normalized(file.relative) for word in words)]


def _normalized(text: str) -> str:
    # 全角の英数字を半角へそろえる 効果音の名前は全角の英字で書かれていることがある
    return text.translate(_WIDE).casefold()


_WIDE = {code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)}


def add_folder(
    folders: Sequence[LibraryFolder], path: str, category: str
) -> tuple[LibraryFolder, ...]:
    """フォルダを登録した後の一覧 同じ場所をもう一度登録したら、分類だけを変える

    同じフォルダを 2 つの分類で登録させない 中のファイルが一覧に 2 度出て、
    AI に使わせない分類として登録し直したつもりでも、もう片方の登録から選ばれる
    """
    if category not in CATEGORIES:
        raise ValueError(f"分類は {'・'.join(CATEGORIES.values())} のどれかです")
    if not path.strip():
        raise ValueError("フォルダが選ばれていません")
    key = _folder_key(path)
    kept = [folder for folder in folders if _folder_key(folder.path) != key]
    return (*kept, LibraryFolder(path, category))


def remove_folder(folders: Sequence[LibraryFolder], path: str) -> tuple[LibraryFolder, ...]:
    """登録を外した後の一覧 ファイルは消さない"""
    key = _folder_key(path)
    return tuple(folder for folder in folders if _folder_key(folder.path) != key)


def set_category(
    folders: Sequence[LibraryFolder], path: str, category: str
) -> tuple[LibraryFolder, ...]:
    """登録したフォルダの分類を変えた後の一覧 並びは変えない"""
    if category not in CATEGORIES:
        raise ValueError(f"分類は {'・'.join(CATEGORIES.values())} のどれかです")
    key = _folder_key(path)
    return tuple(
        LibraryFolder(folder.path, category) if _folder_key(folder.path) == key else folder
        for folder in folders
    )


def _folder_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(Path(path).absolute()))


def parse_folders(value: object) -> tuple[LibraryFolder, ...]:
    """設定のファイルから読んだ登録の一覧 壊れた行は飛ばす（起動を止めない）"""
    if not isinstance(value, list | tuple):
        return ()
    folders: list[LibraryFolder] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict):
            continue
        path, category = row.get("path"), row.get("category")
        if not isinstance(path, str) or not path.strip() or category not in CATEGORIES:
            continue
        key = _folder_key(path)
        if key in seen:
            continue
        seen.add(key)
        folders.append(LibraryFolder(path, str(category)))
    return tuple(folders)


def parse_categories(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    """設定のファイルから読んだ AI に使わせる分類 知らない分類は落とす 形が違えば既定"""
    if not isinstance(value, list | tuple):
        return default
    chosen = {str(item) for item in value if item in CATEGORIES}
    # 並びは分類の表の並びにそろえる 保存するたびに並びが変わると、設定の差が読みにくい
    return tuple(key for key in CATEGORIES if key in chosen)


# --- 測って覚えた手がかり ---


@dataclass(frozen=True, slots=True)
class Measure:
    """素材 1 つを測った手がかり

    ``loudness`` は頭から :data:`MEASURE_SECONDS` 秒ぶんの積分ラウドネス（LUFS）
    ``peak`` は同じ範囲の最大のサンプル（dBFS） 音が無い・測れないときは ``None``
    """

    seconds: float
    has_audio: bool
    has_video: bool
    loudness: float | None = None
    peak: float | None = None
    #: 絵の大きさ（表示の向き）と 1 秒のコマ数 エクスプローラーの列に出す 絵が無ければ 0
    width: int = 0
    height: int = 0
    fps: float = 0.0


#: 測る関数 測れない（開けない・壊れた）ときは ``None``
Measurer = Callable[[Path], Measure | None]

#: 覚えた手がかりの形の版 形を変えたら上げる（前の版の物は読まずに測り直す）
_BOOK_VERSION = 2


@dataclass
class MeasureBook:
    """測った手がかりの控え ファイルの場所・大きさ・更新時刻で引く

    同じファイルを毎回デコードし直すと、数百の効果音から選ぶたびに数十秒待つ
    大きさか更新時刻が変わったら測り直す（同じ名前で中身を差し替えた素材）
    ``path`` が ``None`` なら覚えるのはアプリを閉じるまで（試験と、置き場の無い所）
    """

    path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _entries: dict[str, dict[str, object]] | None = field(default=None, repr=False)
    _dirty: bool = field(default=False, repr=False)

    def get(self, file: LibraryFile) -> Measure | None:
        with self._lock:
            row = self._load().get(file.id)
        if row is None or row.get("size") != file.size or row.get("mtime") != file.modified_ns:
            return None
        return _measure_from(row)

    def put(self, file: LibraryFile, measure: Measure) -> None:
        row: dict[str, object] = {
            "size": file.size,
            "mtime": file.modified_ns,
            "seconds": measure.seconds,
            "audio": measure.has_audio,
            "video": measure.has_video,
            "loudness": measure.loudness,
            "peak": measure.peak,
            "width": measure.width,
            "height": measure.height,
            "fps": measure.fps,
        }
        with self._lock:
            self._load()[file.id] = row
            self._dirty = True

    def save(self) -> None:
        """覚えた物を書く 書けなくても投げない（測り直せば済む物で、編集を止める訳ではない）

        2 つの窓が同時に書くと後の方が残る 消し合っても測り直すだけなので錠では守らない
        """
        with self._lock:
            if self.path is None or not self._dirty or self._entries is None:
                return
            text = json.dumps(
                {"version": _BOOK_VERSION, "files": self._entries}, ensure_ascii=False
            )
            self._dirty = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(self.path, text)
        except OSError:
            return

    def _load(self) -> dict[str, dict[str, object]]:
        if self._entries is not None:
            return self._entries
        entries: dict[str, dict[str, object]] = {}
        if self.path is not None:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict) and data.get("version") == _BOOK_VERSION:
                files = data.get("files")
                if isinstance(files, dict):
                    entries = {str(key): row for key, row in files.items() if isinstance(row, dict)}
        self._entries = entries
        return entries


def _measure_from(row: dict[str, object]) -> Measure | None:
    seconds = row.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, int | float):
        return None
    return Measure(
        seconds=float(seconds),
        has_audio=bool(row.get("audio")),
        has_video=bool(row.get("video")),
        loudness=_number(row.get("loudness")),
        peak=_number(row.get("peak")),
        width=int(_number(row.get("width")) or 0),
        height=int(_number(row.get("height")) or 0),
        fps=_number(row.get("fps")) or 0.0,
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


# --- AI が使う口 ---


@dataclass(frozen=True)
class MaterialLibrary:
    """本人の登録と設定と、測った手がかりの控えをまとめた物 AI の道具と画面が使う

    ``ai_categories`` は AI に使わせると利用者が選んだ分類（既定は効果音と BGM）
    ``auto_bgm`` は頼まれなくても BGM を付けてよいか（既定は頼まれたときだけ）
    """

    folders: tuple[LibraryFolder, ...] = ()
    ai_categories: tuple[str, ...] = DEFAULT_AI_CATEGORIES
    auto_bgm: bool = False
    book: MeasureBook = field(default_factory=MeasureBook)
    measure: Measurer | None = None

    @property
    def ai_folders(self) -> tuple[LibraryFolder, ...]:
        """AI が見てよいフォルダ 選ばれていない分類のフォルダは名前も見せない"""
        return tuple(folder for folder in self.folders if folder.category in self.ai_categories)

    def scan_for_ai(self) -> list[FolderScan]:
        return [scan_folder(folder) for folder in self.ai_folders]

    def find_for_ai(self, wanted: str) -> LibraryFile | None:
        """AI が渡した印のファイル AI に使わせない分類のフォルダからは探さない"""
        for scan in self.scan_for_ai():
            for file in scan.files:
                if file.id == wanted:
                    return file
        return None

    def measured(self, file: LibraryFile, *, allow_measure: bool) -> Measure | None:
        """覚えた手がかり 無ければ ``allow_measure`` のときだけ測って覚える"""
        known = self.book.get(file)
        if known is not None or not allow_measure or self.measure is None:
            return known
        fresh = self.measure(file.path)
        if fresh is not None:
            self.book.put(file, fresh)
        return fresh
