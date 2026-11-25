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
import time
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
    "is_favorite",
    "kind_of",
    "list_folder",
    "match_files",
    "parse_categories",
    "parse_folders",
    "parse_paths",
    "reachable_dir",
    "remove_folder",
    "same_place",
    "scan_folder",
    "set_category",
    "toggle_favorite",
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
    # 対にならないサロゲートを含む Windows のファイル名でも投げない（その 1 つで一覧と道具が落ちる）
    return "lib_" + hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()[:16]


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


def reachable_dir(path: Path) -> tuple[bool, str]:
    """フォルダがあって中を見られるか と、見られない訳（無ければ空）

    登録先の親へのアクセス権を失うと ``Path.is_dir`` が ``PermissionError`` を投げることがある
    1 つの登録先が見られないだけで、ほかの登録先の一覧や起動まで止めない
    """
    try:
        return path.is_dir(), ""
    except OSError as exc:
        return False, f"見られません: {exc.strerror or exc}"


def _linked_inside(path: Path, root: Path, others: Sequence[Path]) -> bool:
    """リンクの実体が ``root`` の中で、別に登録したフォルダ（``others``）の中でないか

    別の登録の中を指すリンクを親の分類で拾うと、その登録を AI に使わせない設定にしていても、
    親の分類の素材として選ばれる 別の登録の物は、その登録自身を見て回るときだけ扱う
    """
    return _inside(path, root) and not any(_inside(path, other) for other in others)


def _inside(path: Path, root: Path) -> bool:
    """``path`` の実体が ``root`` の中か（リンクを解いて比べる）

    解けないリンク（自分を指す循環のリンク）は中でないと扱う ``Path.resolve`` は循環で
    ``RuntimeError`` を投げ、捕まえないと 1 つの登録の壊れたリンクでほかの登録まで見て回れない
    """
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _real_key(path: str) -> str:
    """実体の場所の鍵（リンクとジャンクションを解く） 解けなければ書いたままの場所の鍵"""
    try:
        return os.path.normcase(os.path.realpath(path))
    except (OSError, RuntimeError, ValueError):
        return _folder_key(path)


def _junction(entry: os.DirEntry[str]) -> bool:
    """Windows のジャンクションか（``is_dir(follow_symlinks=False)`` はジャンクションにも真を返す）

    辿ると登録したフォルダの外の普通のファイルに届き、リンクの実体の照らしも通らない
    """
    check = getattr(entry, "is_junction", None)
    return bool(check()) if callable(check) else False


def scan_folder(
    folder: LibraryFolder,
    limit: int = SCAN_LIMIT,
    *,
    skip: Sequence[str] = (),
    accept: Callable[[LibraryFile], bool] | None = None,
) -> FolderScan:
    """``folder`` の中（下のフォルダも）を見て回り、分類に合うファイルを拾う

    ``os.scandir`` で 1 段ずつ見る ``Path.rglob`` は 1 ファイルごとに ``stat`` を呼び直し、
    数千のファイルで目に見えて遅い リンクを辿った先のフォルダには入らない 自分を指す
    リンクがあると、見て回るのが終わらない ファイルのリンクも、実体が登録したフォルダの外なら
    拾わない（登録したフォルダだけを素材の出どころにする決まりと、フォルダのリンクの扱いをそろえる）

    ``skip`` は入らないフォルダ（別に登録したフォルダ） 親を効果音・子を動画として登録した
    ときに、親から子の音を効果音として拾うと、子の分類の決まり（AI に使わせるか）が効かない

    ``accept`` を渡すと、合う物だけを拾い、``limit`` も合う物で数える 名前で絞るときに
    頭の 2 万個で打ち切ってから絞ると、2 万個より後ろにしか無い名前がいつまでも出ない
    """
    root = Path(folder.path)
    present, problem = reachable_dir(root)
    if not present:
        return FolderScan(folder, missing=True, error=problem)
    skipped = {_folder_key(entry) for entry in skip} | {_real_key(entry) for entry in skip}
    # リンクの実体の照らしに使うのは、自分の中にある別の登録だけ（実体を含む登録のうち一番深い
    # 物で分類を決める） 自分を含む親の登録まで見ると、子の登録が持つリンクを子自身の走査でも捨てる
    others = [Path(entry) for entry in skip if _inside(Path(entry), root)]
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
                    # 別の登録は書き方（ジャンクション・別のドライブの割り当て）が違っても
                    # 実体で見分ける 見分けないと、子の分類の物を親の分類で拾う
                    if (
                        _folder_key(entry.path) not in skipped
                        and _real_key(entry.path) not in skipped
                        and not _junction(entry)
                    ):
                        folders.append(Path(entry.path))
                    continue
                if Path(entry.name).suffix.lower() not in suffixes:
                    continue
                if not entry.is_file():
                    continue
                if entry.is_symlink() and not _linked_inside(Path(entry.path), root, others):
                    continue
                stat = entry.stat()
            except OSError:
                continue
            file = LibraryFile(Path(entry.path), folder, stat.st_size, stat.st_mtime_ns)
            if accept is not None and not accept(file):
                continue
            found.append(file)
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


def list_folder(
    folder: LibraryFolder,
    directory: Path,
    limit: int = SCAN_LIMIT,
    *,
    accept: Callable[[LibraryFile], bool] | None = None,
) -> FolderListing:
    """登録した ``folder`` の中の ``directory`` 1 段の、分類に合うファイルと下のフォルダ

    木で開いたフォルダの中身を出すのに使う 下まで全部を見て回ると、根を開くたびに
    数万のファイルを待つ 名前で絞るときだけ :func:`scan_folder` で下まで見る
    """
    present, _problem = reachable_dir(directory)
    if not present:
        return FolderListing(missing=True)
    root = Path(folder.path)
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
                if not _junction(entry):
                    folders.append(Path(entry.path))
                continue
            if Path(entry.name).suffix.lower() not in suffixes or not entry.is_file():
                continue
            if entry.is_symlink() and not _inside(Path(entry.path), root):
                continue
            stat = entry.stat()
        except OSError:
            continue
        file = LibraryFile(Path(entry.path), folder, stat.st_size, stat.st_mtime_ns)
        # 絞り込み（``accept``）に合う物だけを上限に数える 先に上限で打ち切ると、上限より後ろに
        # しか無い種類の物が出ない
        if len(files) < limit and (accept is None or accept(file)):
            files.append(file)
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
        # 分類を文字か確かめてから表で引く 壊れた行（分類が配列など）で引くと型の例外で起動が止まる
        if not isinstance(path, str) or not path.strip():
            continue
        if not isinstance(category, str) or category not in CATEGORIES:
            continue
        key = _folder_key(path)
        if key in seen:
            continue
        seen.add(key)
        folders.append(LibraryFolder(path, str(category)))
    return tuple(folders)


# --- お気に入り（#521 本人の要望 2026-11-24） ---


def same_place(a: str | Path, b: str | Path) -> bool:
    """同じ場所か 大文字と小文字・区切りの書き方の違いを同じと見る（Windows の場所の決まり）"""
    return _folder_key(str(a)) == _folder_key(str(b))


def toggle_favorite(favorites: Sequence[str], path: str | Path, wanted: bool) -> tuple[str, ...]:
    """お気に入りに入れた（``wanted`` が偽なら外した）後の一覧 並びは入れた順

    場所で覚える 消えた・移った物も外さずに残す（外付けのドライブを挿し直せば戻る物を、
    勝手に忘れない 登録したフォルダと同じ扱い）
    """
    kept = [entry for entry in favorites if not same_place(entry, path)]
    if wanted:
        kept.append(str(path))
    return tuple(kept)


def is_favorite(favorites: Sequence[str], path: str | Path) -> bool:
    """``path`` がお気に入りか、お気に入りのフォルダの中の物か"""
    key = _folder_key(str(path))
    for entry in favorites:
        mark = _folder_key(entry)
        if key == mark or key.startswith(mark.rstrip(os.sep) + os.sep):
            return True
    return False


def parse_paths(value: object) -> tuple[str, ...]:
    """設定のファイルから読んだ場所の一覧 壊れた物と重なりは落とす（起動を止めない）"""
    if not isinstance(value, list | tuple):
        return ()
    kept: list[str] = []
    for entry in value:
        if isinstance(entry, str) and entry.strip() and not any(same_place(entry, k) for k in kept):
            kept.append(entry)
    return tuple(kept)


def parse_categories(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    """設定のファイルから読んだ AI に使わせる分類 知らない分類は落とす 形が違えば既定"""
    if not isinstance(value, list | tuple):
        return default
    # 文字でない要素（{} など）を表で引くと型の例外で起動が止まる 文字だけを見る
    chosen = {item for item in value if isinstance(item, str) and item in CATEGORIES}
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
    #: 音を読めずに大きさを測れなかった（音の無い素材・本当に無音の音とは別） 長さは使えるが、
    #: 控えのファイルには書かず、しばらくしたら測り直す（ドライブが一時だけ応えなかった物を戻す）
    audio_failed: bool = False


#: 測る関数 測れない（開けない・壊れた）ときは ``None``
Measurer = Callable[[Path], Measure | None]

#: 覚えた手がかりの形の版 形を変えたら上げる（前の版の物は読まずに測り直す）
_BOOK_VERSION = 2


# 同じ控えかは中身でなく物そのもので見る（AI の道具が走っている仕事を控えごとに引くため）
@dataclass(eq=False)
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
    #: 測れなかったファイル 印 → （大きさ, 更新時刻, 覚えた時計） 控えのファイルには書かない
    #: ドライブが一時だけ応えなかった物まで書くと、挿し直して起動し直しても測り直さない
    _failures: dict[str, tuple[int, int, float]] = field(default_factory=dict, repr=False)
    #: 見て回って見つけたファイル 印 → 場所（:meth:`remember_found`）
    _found: dict[str, Path] = field(default_factory=dict, repr=False)
    #: 音を読めなかった測った物（:attr:`Measure.audio_failed`） 失敗と同じく記憶の中だけに持つ
    _passing: dict[str, tuple[int, int, float, Measure]] = field(default_factory=dict, repr=False)

    def get(self, file: LibraryFile) -> Measure | None:
        """覚えた物 同じファイルの完全に測れた物を、音を読めなかった一時の物より先に返す

        画面と AI が同じ素材を同時に測って片方だけ音を読めなかったとき、一時の物を先に見ると
        完全に測れた物があっても大きさの無い物を返し続ける
        """
        with self._lock:
            passing = self._passing.get(file.id)
            row = self._load().get(file.id)
        if (
            row is not None
            and row.get("size") == file.size
            and row.get("mtime") == file.modified_ns
        ):
            whole = _measure_from(row)
            if whole is not None:
                return whole
        if passing is not None:
            size, mtime, at, measure = passing
            if (size, mtime) == (file.size, file.modified_ns) and (
                time.monotonic() - at < FAILURE_SECONDS
            ):
                return measure
        return None

    def remember_found(self, files: Iterable[LibraryFile]) -> None:
        """見て回って見つけたファイルの印と場所を覚える（記憶の中だけ）

        AI が印で指したファイルを、登録したフォルダを全部見て回り直さずに引くため 置く道具は
        画面のスレッドで動くので、見て回り直すと大きな登録先で画面が止まる
        """
        with self._lock:
            for file in files:
                self._found[file.id] = file.path

    def found(self, wanted: str) -> Path | None:
        with self._lock:
            return self._found.get(wanted)

    def preload(self) -> None:
        """控えのファイルを先に読んでおく（裏のスレッドから呼ぶ） 初めて一覧を並べるときに、
        画面のスレッドで数万行の控えを読まない
        """
        # 読むのは錠の外 錠の内で読むと、初めての先読みの間に窓を閉じたときに、終わりの保存
        # （:meth:`save`）が読み終えるまで待つ 読み終えたときに、その間に測って入れた物が
        # あれば（:meth:`put` が先に読んだ）読んだ物は捨てる（新しい測った値を上書きしない）
        if self._entries is not None:
            return
        entries = _read_book(self.path)
        with self._lock:
            if self._entries is None:
                self._entries = entries

    def failed(self, file: LibraryFile) -> bool:
        """少し前に測れなかったファイルか（変わっておらず :data:`FAILURE_SECONDS` の内）"""
        with self._lock:
            known = self._failures.get(file.id)
        if known is None:
            return False
        size, mtime, at = known
        return (size, mtime) == (file.size, file.modified_ns) and (
            time.monotonic() - at < FAILURE_SECONDS
        )

    def put_failure(self, file: LibraryFile) -> None:
        """測れなかったことをしばらく覚える 覚えないと、読めないファイルが頭に並んだときに、
        呼び直すたびに同じ物で測る数を使い切り、後ろの読めるファイルがいつまでも測られない
        控えのファイルには書かず、時間が経てば測り直す（一時だけ読めなかった物を戻すため）
        """
        with self._lock:
            self._failures[file.id] = (file.size, file.modified_ns, time.monotonic())

    def put(self, file: LibraryFile, measure: Measure) -> None:
        if measure.audio_failed:
            # 長さは使うが控えのファイルには書かない 書くと、一時だけ読めなかった音の大きさを
            # 大きさと更新時刻が同じ限り測り直さない
            with self._lock:
                self._passing[file.id] = (
                    file.size,
                    file.modified_ns,
                    time.monotonic(),
                    measure,
                )
            return
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
        # 何も測っていなければ錠を待たずに抜ける（先読みの途中でも窓をすぐ閉じられる）
        if self.path is None or not self._dirty:
            return
        with self._lock:
            if not self._dirty or self._entries is None:
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
        if self._entries is None:
            self._entries = _read_book(self.path)
        return self._entries


def _read_book(path: Path | None) -> dict[str, dict[str, object]]:
    """控えのファイルを読む 無い・壊れている・版が違うときは空"""
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        # 深く入れ子にした配列は json が RecursionError を投げる 壊れた控えと同じく空にする
        # （投げると、一覧を読む裏の仕事ごと落ちて「この場所を読めません」と出る）
        return {}
    if not isinstance(data, dict) or data.get("version") != _BOOK_VERSION:
        return {}
    files = data.get("files")
    if not isinstance(files, dict):
        return {}
    return {str(key): row for key, row in files.items() if isinstance(row, dict)}


#: 測れなかったファイルを測り直さない間（秒） 呼び直すたびに同じ物で測る数を使い切らず、
#: ドライブを挿し直した後などは、しばらくすれば測り直す
FAILURE_SECONDS = 600

#: 控えから読む数の上限 これを越える値（手で書き換えた 1e999 など）の行は測り直す
_LIMIT = 1e9


def _measure_from(row: dict[str, object]) -> Measure | None:
    """控えの行を測った物に戻す 壊れた行は ``None``（測り直す） 一覧を作る途中で例外にしない"""
    if row.get("failed"):
        return None
    numbers = ("seconds", "width", "height", "fps", "loudness", "peak")
    if any(row.get(key) is not None and _number(row.get(key)) is None for key in numbers):
        # 読めない数（文字・無限・桁違い）のある行は丸ごと捨てて測り直す 一部だけ 0 にすると、
        # 壊れた値を正しい測った値として使い続ける
        return None
    seconds = _number(row.get("seconds"))
    width, height = _number(row.get("width")), _number(row.get("height"))
    fps = _number(row.get("fps"))
    if seconds is None or seconds < 0:
        return None
    if any(value is not None and value < 0 for value in (width, height, fps)):
        return None
    return Measure(
        seconds=seconds,
        has_audio=bool(row.get("audio")),
        has_video=bool(row.get("video")),
        loudness=_number(row.get("loudness")),
        peak=_number(row.get("peak")),
        width=int(width or 0),
        height=int(height or 0),
        fps=fps or 0.0,
    )


def _number(value: object) -> float | None:
    """有限で上限より小さい数 そうでなければ ``None``"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, int) and abs(value) > _LIMIT:
        # 桁違いの整数は float にすると OverflowError を投げる 変える前に断る
        return None
    number = float(value)
    if number != number or abs(number) > _LIMIT:
        return None
    return number


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
    #: お気に入りの場所（ファイルかフォルダ） 本人の設定 :attr:`Preferences.explorer_favorite_paths`
    favorites: tuple[str, ...] = ()

    def is_favorite(self, file: LibraryFile) -> bool:
        return is_favorite(self.favorites, file.path)

    @property
    def ai_folders(self) -> tuple[LibraryFolder, ...]:
        """AI が見てよいフォルダ 選ばれていない分類のフォルダは名前も見せない"""
        return tuple(folder for folder in self.folders if folder.category in self.ai_categories)

    def scan_for_ai(self) -> list[FolderScan]:
        """AI が見てよいフォルダを見て回る 中に別に登録したフォルダがあれば、そこには入らない
        （その登録の分類が AI に使わせる物なら、それ自身として別に見て回る）
        """
        others = [folder.path for folder in self.folders]
        scans = [
            scan_folder(
                folder,
                skip=[path for path in others if not same_place(path, folder.path)],
            )
            for folder in self.ai_folders
        ]
        for scan in scans:
            self.book.remember_found(scan.files)
        return scans

    def owner_of(self, path: str | Path) -> LibraryFolder | None:
        """``path`` を含む登録のうち一番深い物（そろえた場所の深さで比べる） 無ければ ``None``"""
        owners = [folder for folder in self.folders if is_favorite([folder.path], path)]
        if not owners:
            return None
        return max(owners, key=lambda folder: len(Path(_folder_key(folder.path)).parts))

    def find_for_ai(self, wanted: str, *, scan: bool = True) -> LibraryFile | None:
        """AI が渡した印のファイル AI に使わせない分類のフォルダからは探さない

        前に見て回って覚えた場所があれば、それが今も AI に使わせる登録の物かだけを確かめて返す
        （見て回り直さない） 覚えていなければ見て回る
        """
        known = self.book.found(wanted)
        if known is not None:
            # 覚えている印は、今の登録と分類で通らなければそこで断る（見て回り直さない）
            # 断るだけの道具で大きな登録先を見て回ると、画面のスレッドが止まる
            quick = self._check(known)
            return quick if quick is not None and quick.id == wanted else None
        if not scan:
            # 覚えていない印は見て回らずに断る（画面のスレッドから呼ぶ道具のため） 呼び手は
            # search_library で探し直すよう伝える
            return None
        for found in self.scan_for_ai():
            for file in found.files:
                if file.id == wanted:
                    return file
        return None

    def _check(self, path: Path) -> LibraryFile | None:
        """覚えた場所が今も AI に使わせる登録の、分類に合うファイルか（見て回るのと同じ決まり）"""
        owner = self.owner_of(path)
        if owner is None or owner.category not in self.ai_categories:
            return None
        if path.suffix.lower() not in _SUFFIXES.get(owner.category, _SUFFIXES[CATEGORY_OTHER]):
            return None
        root = Path(owner.path)
        try:
            if path.is_symlink():
                # 実体を解くのはリンクのときだけ（画面のスレッドから呼ぶので、応えないネットワークの
                # 登録先を含む全部の登録を解きに行かない） 中の別の登録は文字の場所で先に絞る
                others = [
                    Path(folder.path)
                    for folder in self.folders
                    if not same_place(folder.path, owner.path)
                    and is_favorite([owner.path], folder.path)
                ]
                if not _linked_inside(path, root, others):
                    return None
            # 途中のフォルダがジャンクションやリンクなら、見て回るときは入らない
            for parent in path.parents:
                if same_place(parent, root):
                    break
                if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                    return None
            stat = path.stat()
        except OSError:
            return None
        if not path.is_file():
            return None
        return LibraryFile(path, owner, stat.st_size, stat.st_mtime_ns)

    def measured(self, file: LibraryFile, *, allow_measure: bool) -> Measure | None:
        """覚えた手がかり 無ければ ``allow_measure`` のときだけ測って覚える"""
        known = self.book.get(file)
        if known is not None or not allow_measure or self.measure is None:
            return known
        if self.book.failed(file):
            return None
        fresh = self.measure(file.path)
        if not _unchanged(file):
            # 測っている間に同じ場所のファイルが差し替わった 新しいファイルの測った物を前の
            # ファイルの物として覚えず、返しもしない（成功も失敗も覚えない 次に測り直す）
            return None
        if fresh is None:
            self.book.put_failure(file)
        else:
            self.book.put(file, fresh)
        return fresh

    def tried(self, file: LibraryFile) -> bool:
        """前に測ろうとして測れなかったか（測る数を使わずに飛ばす）"""
        return self.book.failed(file)


def _unchanged(file: LibraryFile) -> bool:
    """見て回ったときから大きさと更新時刻が変わっていないか"""
    try:
        stat = file.path.stat()
    except OSError:
        return False
    return (stat.st_size, stat.st_mtime_ns) == (file.size, file.modified_ns)
