"""持ち出し用のパッケージの書き出しと展開（#423）

作品（``.sme``）は素材をパスで指すだけなので、素材の無い PC では絵も音も出ない
パッケージは作品と素材を 1 つの ZIP（拡張子 :data:`PACKAGE_SUFFIX`）にまとめた物

- 素材は「無圧縮で格納」する 動画と音声はもう圧縮してあり、さらに縮めても小さくならず、
  書き出しと展開が遅くなるだけ
- 素材は既定で使っている範囲だけを再エンコードせずに写す（:mod:`.remux` 画質はそのまま）
- 中の場所は番号だけ（``media/0001.mp4``） 元のフォルダ（ユーザー名が入る）を入れない
- 目録（``manifest.json``）に入れた物ごとの SHA-256 を書き、開く前に全部を照らす
- 普段の保存（``.sme``）と自動の退避は今のまま パッケージは持ち出すときだけ書く

開くと作業用のフォルダへ展開し、素材を指す場所を展開した先へ付け替えた ``.sme`` を置く
その後は普段の作品と同じ（編集・保存・書き出し・バックアップ）
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, BinaryIO

from sashimono import __version__
from sashimono.core.io.atomic import Expected, fill_atomic
from sashimono.core.io.serialize import (
    SUFFIX,
    ProjectFileError,
    json_text,
    project_from_dict,
    project_to_dict,
    save_project,
)
from sashimono.core.model import MediaId, MediaItem, Project
from sashimono.engine.package.plan import (
    Usage,
    file_fields,
    folder_kinds,
    map_paths,
    unknown_kinds,
    usage,
)
from sashimono.engine.package.remux import Cancelled, Span, cut_media, output_format
from sashimono.engine.package.scrub import scrub_image, scrubbable

__all__ = [
    "MANIFEST",
    "PACKAGE_FORMAT",
    "PACKAGE_SUFFIX",
    "PACKAGE_VERSION",
    "RIGHTS_NOTE",
    "Cancelled",
    "OpenedPackage",
    "PackOptions",
    "PackResult",
    "PackageError",
    "open_package",
    "write_package",
]

#: パッケージの拡張子 プリセット（``.smep``）とエイリアス（``.smea``）とは別にする
PACKAGE_SUFFIX = ".smex"
#: 目録の ``format`` 別の ZIP を取り違えて開かないための印
PACKAGE_FORMAT = "sashimono-package"
#: 目録の形の版 形を変えたら上げる
#:
#: 目録には ``minimum`` も書く（読むのに要る本体の版の下限 自動更新の目録と同じ考え）
#: 前の本体でも読める足し方（項目を足すだけ）なら ``minimum`` は上げずに版だけ上げ、
#: 前の本体が読み違える変え方をしたときだけ ``minimum`` も上げる 読む側は ``minimum`` が
#: 自分の版より大きいときだけ断る
PACKAGE_VERSION = 1
#: 読むのに要る版の下限（今の形を読むのに要るのは 1）
PACKAGE_MINIMUM = 1
#: 素材の権利の断り 書き出しの窓と AI の確かめの箱の両方に出す
RIGHTS_NOTE = (
    "素材をそのまま入れます 渡す相手に素材を使わせてよいか（権利）は、持ち出す人が確かめてください"
)
#: 目録の名前
MANIFEST = "manifest.json"
#: 作品の名前（パッケージの中）
PROJECT_ENTRY = "project.sme"

#: 中の素材とファイルの場所の形 番号と短い拡張子だけ 展開するときにこれ以外の名前を断る
#: （``..`` や絶対の場所で、展開先の外へ書かせない）
_ENTRY_NAME = re.compile(r"(media|files)/\d{4,6}(\.[A-Za-z0-9]{1,10})?")

#: 1 度に読む長さ 手元に動画を丸ごと抱えない
_CHUNK = 1 << 20

#: 空きの見積もりに足す余裕（割合とバイト） 入れ物の頭や目録、書き手の癖の分
_SPACE_RATIO = Fraction(11, 10)
_SPACE_EXTRA = 64 << 20

#: 範囲だけ入れる素材の見積もりで、区間ごとに足す長さ（秒） キーフレームまで戻る分
#: 長い GOP の素材ではもっと増えるが、見積もりは足りないときに先に断るための物で、
#: 書いている途中で足りなくなっても前のファイルは無傷で残る
_GOP_GUESS = Fraction(10)

#: 付け足しの情報を取り除くとき、手元に読み込む静止画の大きさの上限
_SCRUB_LIMIT = 256 << 20

#: 目録と作品を手元へ読み込むときの大きさの上限 どちらもまるごと読んで JSON にするので、
#: 細工したパッケージが巨大な物を置くと、開くだけで編集画面がメモリを使い切って落ちる
#: （#449 の CodeRabbit の指摘） 目録は名前と数の一覧、作品は長い物でも数十 MB に収まる
_MANIFEST_LIMIT = 16 << 20
_PROJECT_LIMIT = 512 << 20


class PackageError(Exception):
    """パッケージを書けない・開けない 文はそのまま人へ見せる"""


@dataclass(frozen=True, slots=True)
class PackOptions:
    """書き出しの選び方（書き出しの窓と AI の道具が同じ物を作る）"""

    #: 素材をまるごと入れる（偽なら使っている範囲だけ） まるごとなら範囲の外へもトリムを
    #: 伸ばせるが、大きくなる
    whole: bool = False
    #: 使う範囲の前後に足す余裕（秒）
    margin: Fraction = Fraction(1)
    #: 撮った場所などの付け足しの情報を残す（既定は取り除く 人へ渡す物なので）
    keep_metadata: bool = False


@dataclass(frozen=True, slots=True)
class PackResult:
    """書き出した結果 画面と AI が知らせる物"""

    path: Path
    size: int
    media_count: int
    file_count: int
    #: 入れなかった書体 開く人の PC に無ければ別の書体で描かれる
    fonts: tuple[str, ...]
    #: 入れなかった AviUtl のスクリプト
    scripts: tuple[str, ...]
    #: 付け足しの情報を取り除けずにそのまま入れた物（元の名前）
    metadata_kept: tuple[str, ...]
    #: 範囲だけを切り出せず、まるごと入れた素材（元の名前）
    whole_instead: tuple[str, ...]
    #: 設定が指していたのに見つからなかったファイル（元の名前）
    missing_files: tuple[str, ...]
    #: 素材一覧にあってもタイムラインで使っていないので入れなかった素材（元の名前）
    unused_media: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OpenedPackage:
    """展開した結果"""

    #: 展開した先に置いた作品のファイル
    project_path: Path
    project: Project
    folder: Path
    fonts: tuple[str, ...]
    scripts: tuple[str, ...]
    #: 書き出した本体の版
    made_by: str


Progress = Callable[[float, str], None]


@dataclass(slots=True)
class _Entry:
    """パッケージに入れる 1 つ"""

    name: str
    role: str
    source: Path
    original: str
    media_id: MediaId | None = None
    spans: list[Span] | None = None
    weight: int = 0
    #: 書いた後に埋める
    size: int = 0
    sha256: str = ""
    whole: bool = True
    metadata_removed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _media_name(number: int, source: Path, suffix: str | None = None) -> str:
    """中の素材の名前（番号と拡張子だけ 元のフォルダと名前は入れない）"""
    extension = suffix if suffix is not None else source.suffix.lower()
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", extension):
        extension = ""
    return f"media/{number:04d}{extension}"


def _file_name(number: int, source: Path) -> str:
    """設定が指すファイルの中の名前（番号と拡張子だけ）"""
    extension = source.suffix.lower()
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", extension):
        extension = ""
    return f"files/{number:04d}{extension}"


def _estimate(item: MediaItem, spans: list[Span] | None, size: int) -> int:
    """範囲だけ入れる素材の大きさの見積もり（バイト） まるごとなら元の大きさ"""
    if spans is None or item.duration <= 0:
        return size
    covered = sum((span.end - span.start + _GOP_GUESS for span in spans), Fraction(0))
    share = min(Fraction(1), covered / item.duration)
    return int(size * share)


def _free_space(folder: Path) -> int:
    """``folder`` のあるディスクの空き（バイト） 試験で差し替える"""
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _require_space(folder: Path, needed: int, what: str) -> None:
    """空きが ``size`` より少なければ、書く前に断る"""
    free = _free_space(folder)
    wanted = int(needed * _SPACE_RATIO) + _SPACE_EXTRA
    if free < wanted:
        raise PackageError(
            f"{what}の空きが足りません（要る見込み {_megabytes(wanted)}、空き {_megabytes(free)}）"
            " 空きを作るか、別の場所を選んでください"
        )


def _megabytes(size: int) -> str:
    """大きさを人が読む形にする（知らせの文に使う）"""
    if size >= 1 << 30:
        return f"{size / (1 << 30):.1f} GB"
    return f"{size / (1 << 20):.0f} MB"


def write_package(
    project: Project,
    target: Path,
    options: PackOptions,
    *,
    scratch: Path,
    progress: Progress = lambda _fraction, _text: None,
    cancelled: Callable[[], bool] = lambda: False,
    expect: Expected | None = None,
) -> PackResult:
    """作品と素材を ``target`` へまとめる 書き終えるまで前のファイルは無傷で残る

    ``scratch`` は範囲を切り出した素材を一時に置くフォルダ（1 本ずつ書いては消す）
    止められたら :class:`Cancelled` 書けなければ :class:`PackageError`
    """
    target = Path(target)
    found = usage(project, options.margin, whole=options.whole)
    by_id = {item.id: item for item in project.media}
    missing = [
        by_id[key].display_name or Path(by_id[key].path).name
        for key in found.spans
        if not Path(by_id[key].path).is_file()
    ]
    if missing:
        raise PackageError(
            "素材のファイルが見つからないので、パッケージにできません: " + "、".join(missing)
        )
    unknown = unknown_kinds(project)
    if unknown:
        # どの欄がファイルかを定義から読めないので、元の PC の場所が欄に残ったまま人へ渡り、
        # 指すファイルも入らない 読み込んでから書き出してもらう（#449 の CodeRabbit の指摘）
        raise PackageError(
            "読み込んでいないスクリプトのエフェクトがあるので、パッケージにできません"
            " スクリプトを読み込むか、そのエフェクトを外してから書き出してください: "
            + "、".join(unknown[:10])
        )
    folders = folder_kinds(project)
    if folders:
        # フォルダはまだ入れられない 欄に元の PC の場所（ユーザー名が入る）が残ったまま人へ渡る
        # （#449 の CodeRabbit の指摘）
        raise PackageError(
            "フォルダを指すエフェクトがあるので、パッケージにできません"
            " フォルダの欄を空にするか、そのエフェクトを外してから書き出してください: "
            + "、".join(folders[:10])
        )

    entries: list[_Entry] = []

    for number, (key, spans) in enumerate(found.spans.items(), start=1):
        item = by_id[key]
        source = Path(item.path)
        size = source.stat().st_size
        entry = _Entry(
            name=_media_name(number, source),
            role="media",
            source=source,
            original=item.display_name or source.name,
            media_id=key,
            spans=spans,
            weight=_estimate(item, spans, size),
        )
        entries.append(entry)

    file_names: dict[str, str] = {}
    missing_files: list[str] = []
    for number, value in enumerate(found.files, start=1):
        source = Path(value)
        if not source.is_file():
            missing_files.append(source.name)
            continue
        entry = _Entry(
            name=_file_name(number, source),
            role="file",
            source=source,
            original=source.name,
            weight=source.stat().st_size,
        )
        entries.append(entry)
        file_names[value] = entry.name

    total = sum(entry.weight for entry in entries) or 1
    _require_space(target.parent, total, "書き出し先")
    largest = max((entry.weight for entry in entries if entry.spans is not None), default=0)
    if largest:
        _require_space(scratch, largest, "作業用のフォルダ（アプリのキャッシュ）")

    def missing_name(value: str) -> str:
        """設定が指すファイルの中の名前 入れなかった物は名前だけ"""
        # 見つからないファイルは場所を捨てて名前だけにする 元のフォルダ（ユーザー名が入る）を
        # 人へ渡すパッケージに残さないため 開いた先でも見つからないのは元の PC と同じ
        return file_names.get(value, Path(value).name)

    metadata_kept: list[str] = []
    whole_instead: list[str] = []
    done = 0

    def fill(handle: BinaryIO) -> None:
        """パッケージの中身を書く（書き終えるまで前のファイルは無傷）"""
        nonlocal done
        with zipfile.ZipFile(
            handle, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zipped:
            for entry in entries:
                if cancelled():
                    raise Cancelled
                progress(done / total, f"{entry.original} を入れています")
                _put_entry(zipped, entry, options, scratch, cancelled, metadata_kept, whole_instead)
                done += entry.weight
            # 中の名前は書いてから決まる（切り出すと入れ物が変わり、拡張子が変わることがある）
            names = {entry.media_id: entry.name for entry in entries if entry.media_id is not None}
            # 中の場所は番号だけなので、表示の名前が空の素材は元のファイルの名前を残す
            # 残さないと、開いた先の素材一覧とクリップに 0001.mp4 と出て、どの素材か分からない
            named = replace(
                project,
                media=tuple(
                    replace(item, display_name=item.display_name or Path(item.path).name)
                    for item in project.media
                ),
            )
            packed = map_paths(
                named,
                lambda item: Path(names[item.id]) if item.id in names else _unused_place(item),
                missing_name,
            )
            project_bytes = json_text(project_to_dict(packed)).encode("utf-8")
            project_entry = _Entry(PROJECT_ENTRY, "project", Path(), "")
            project_entry.size = len(project_bytes)
            project_entry.sha256 = hashlib.sha256(project_bytes).hexdigest()
            zipped.writestr(PROJECT_ENTRY, project_bytes)
            manifest = _manifest(project, entries, project_entry, found)
            zipped.writestr(MANIFEST, json_text(manifest).encode("utf-8"))
        progress(1.0, "書き終えました")

    try:
        written = fill_atomic(target, fill, durable=True, expect=expect)
    except Cancelled:
        raise
    except PackageError:
        raise
    except OSError as exc:
        raise PackageError(
            f"パッケージを書けませんでした（書き出し先は前のままです）: {exc}"
        ) from exc
    return PackResult(
        path=target,
        size=written.st_size,
        media_count=sum(1 for entry in entries if entry.role == "media"),
        file_count=sum(1 for entry in entries if entry.role == "file"),
        fonts=found.fonts,
        scripts=found.scripts,
        metadata_kept=tuple(metadata_kept),
        whole_instead=tuple(whole_instead),
        missing_files=tuple(missing_files),
        unused_media=tuple(
            item.display_name or Path(item.path).name
            for item in project.media
            if item.id not in found.spans
        ),
    )


#: 使っていない素材を指す場所の頭（パッケージの中には無い） 素材一覧の行と字幕の起こし結果を
#: 残すため、素材一覧からは消さずに、無いと分かる場所を指させる（開いた先では見つからない素材）
UNUSED_PREFIX = "unused"


#: 展開した先で、パッケージに無いファイルを指させる場所（作らない）
MISSING_FOLDER = "missing"
#: 使っていない素材の場所の形 名前 1 つだけ（``..`` や区切りを含めて外を指させない）
_UNUSED_NAME = re.compile(rf"{UNUSED_PREFIX}/[^/\\:]+")


def _safe_file_name(value: str) -> str:
    """場所の最後の名前だけを、ファイルの名前に使えない文字を除いて取り出す"""
    last = re.split(r"[/\\]", value)[-1]
    return re.sub(r'[<>:"|?*\x00-\x1f]', "_", last).strip(" .") or "ファイル"


def _outside_place(value: str) -> bool:
    """パッケージの外を指す文字か 共有のフォルダ（UNC）かドライブから始まる場所

    共有のフォルダは開くと別の機械へつなぐ ドライブから始まる場所は開いた PC の手元の
    ファイル（人が選んでいない物）を読む ``/字幕`` のような斜線 1 つで始まる文は Windows では
    場所にならないので外す
    """
    text = value.strip()
    head = text[:2]
    if len(head) == 2 and all(char in "/\\" for char in head):
        return True
    return re.match(r"[A-Za-z]:[/\\]", text) is not None


def _unused_place(item: MediaItem) -> Path:
    """使っていない素材の場所 元のフォルダ（ユーザー名が入る）は捨てて名前だけ残す"""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(item.path).name).strip(" .") or "素材"
    return Path(UNUSED_PREFIX) / name


def _put_entry(
    zipped: zipfile.ZipFile,
    entry: _Entry,
    options: PackOptions,
    scratch: Path,
    cancelled: Callable[[], bool],
    metadata_kept: list[str],
    whole_instead: list[str],
) -> None:
    """1 つを入れる 素材は切り出すか、取り除くか、そのまま写す"""
    source = entry.source
    if entry.role == "media" and (entry.spans is not None or not options.keep_metadata):
        if _put_cut(zipped, entry, options, scratch, cancelled):
            return
        if entry.spans is not None:
            whole_instead.append(entry.original)
    if (
        not options.keep_metadata
        and scrubbable(source.suffix)
        and source.stat().st_size <= _SCRUB_LIMIT
    ):
        cleaned = scrub_image(source.read_bytes(), source.suffix)
        if cleaned is not None:
            entry.metadata_removed = True
            _put_bytes(zipped, entry, cleaned)
            return
    if not options.keep_metadata:
        # 形で決め打ちせず、取り除けずにそのまま入れた物は全部知らせる WAV の BWF のように、
        # 音のファイルも作った人の名前などを持ちうる（#449 の CodeRabbit の指摘）
        metadata_kept.append(entry.original)
    with source.open("rb") as reader:
        _put_stream(zipped, entry, reader, source.stat().st_size, cancelled)


def _put_cut(
    zipped: zipfile.ZipFile,
    entry: _Entry,
    options: PackOptions,
    scratch: Path,
    cancelled: Callable[[], bool],
) -> bool:
    """切り出して（まるごとなら付け足しの情報だけ落として）入れる できなければ偽"""
    chosen = output_format(entry.source)
    if chosen is None:
        return False
    muxer, extension = chosen
    spans = entry.spans if entry.spans is not None else [Span(Fraction(0), Fraction(10**9))]
    scratch.mkdir(parents=True, exist_ok=True)
    cut = scratch / f"{uuid.uuid4().hex}{extension}"
    try:
        if not cut_media(
            entry.source,
            cut,
            spans,
            muxer=muxer,
            keep_metadata=options.keep_metadata,
            cancelled=cancelled,
        ):
            return False
        entry.name = f"{entry.name.split('.', 1)[0]}{extension}"
        entry.whole = entry.spans is None
        entry.metadata_removed = not options.keep_metadata
        with cut.open("rb") as reader:
            _put_stream(zipped, entry, reader, cut.stat().st_size, cancelled)
        return True
    finally:
        with contextlib.suppress(OSError):
            cut.unlink(missing_ok=True)


def _put_bytes(zipped: zipfile.ZipFile, entry: _Entry, data: bytes) -> None:
    """中身をそのまま入れ、大きさと SHA-256 を記す"""
    entry.size = len(data)
    entry.sha256 = hashlib.sha256(data).hexdigest()
    zipped.writestr(entry.name, data)


def _put_stream(
    zipped: zipfile.ZipFile,
    entry: _Entry,
    reader: BinaryIO,
    size: int,
    cancelled: Callable[[], bool],
) -> None:
    """ファイルを少しずつ写して入れ、大きさと SHA-256 を記す（大きな素材を丸ごと読まない）"""
    digest = hashlib.sha256()
    written = 0
    info = zipfile.ZipInfo(entry.name)
    info.compress_type = zipfile.ZIP_STORED
    with zipped.open(info, "w", force_zip64=size >= (1 << 31) - _CHUNK) as out:
        while chunk := reader.read(_CHUNK):
            if cancelled():
                raise Cancelled
            digest.update(chunk)
            out.write(chunk)
            written += len(chunk)
    entry.size = written
    entry.sha256 = digest.hexdigest()


def _manifest(
    project: Project, entries: list[_Entry], project_entry: _Entry, found: Usage
) -> dict[str, Any]:
    """目録（``manifest.json``）の中身"""
    return {
        "format": PACKAGE_FORMAT,
        "version": PACKAGE_VERSION,
        "minimum": PACKAGE_MINIMUM,
        "app_version": __version__,
        "project": {
            "path": PROJECT_ENTRY,
            "size": project_entry.size,
            "sha256": project_entry.sha256,
        },
        "name": project.name,
        # ファイルを指す欄の在りか 開く PC にスクリプトが無くても、この欄だけを付け替える
        "file_fields": list(file_fields(project)),
        "entries": [
            {
                "path": entry.name,
                "role": entry.role,
                "name": entry.original,
                "size": entry.size,
                "sha256": entry.sha256,
                "whole": entry.whole,
                "metadata_removed": entry.metadata_removed,
                **({"media_id": entry.media_id} if entry.media_id is not None else {}),
                **(
                    {
                        "spans": [
                            [
                                f"{span.start.numerator}/{span.start.denominator}",
                                f"{span.end.numerator}/{span.end.denominator}",
                            ]
                            for span in entry.spans
                        ]
                    }
                    if entry.spans is not None and not entry.whole
                    else {}
                ),
            }
            for entry in entries
        ],
        "fonts": list(found.fonts),
        "scripts": list(found.scripts),
    }


# --- 開く ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Listed:
    """目録に書かれた中身 1 つ（場所・大きさ・SHA-256）"""

    path: str
    size: int
    sha256: str


def _read_manifest(zipped: zipfile.ZipFile) -> dict[str, Any]:
    """目録を読み、形と版を確かめる 読めなければ断る"""
    try:
        raw = _bounded_read(zipped, MANIFEST, _MANIFEST_LIMIT, "目録")
        manifest = json.loads(raw.decode("utf-8"))
    except KeyError as exc:
        raise PackageError(
            "Sashimono Edit の持ち出し用パッケージではありません（目録が無い）"
        ) from exc
    except (ValueError, zipfile.BadZipFile, OSError) as exc:
        raise PackageError(f"パッケージの目録が壊れています: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != PACKAGE_FORMAT:
        raise PackageError("Sashimono Edit の持ち出し用パッケージではありません")
    minimum = manifest.get("minimum")
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise PackageError("パッケージの目録が壊れています（minimum）")
    if minimum > PACKAGE_VERSION:
        raise PackageError(
            "新しい版の Sashimono Edit で作ったパッケージです 本体を更新してから開いてください"
            f"（パッケージの形 {minimum}、この版が読めるのは {PACKAGE_VERSION} まで）"
        )
    return manifest


def _bounded_read(zipped: zipfile.ZipFile, path: str, limit: int, label: str) -> bytes:
    """中身を ``limit`` バイトまで読む 超えれば断る

    中の見出しの大きさは細工できるので、先に見出しで断り、読むときも上限の 1 つ先までしか
    手元に置かない（見出しが小さく偽っていても、それ以上は読まない）
    """
    if zipped.getinfo(path).file_size > limit:
        raise PackageError(f"パッケージの{label}が大きすぎます（{limit >> 20} MB まで）")
    with zipped.open(path) as reader:
        data = reader.read(limit + 1)
    if len(data) > limit:
        raise PackageError(f"パッケージの{label}が大きすぎます（{limit >> 20} MB まで）")
    return data


def _listed(raw: object, *, project: bool = False) -> _Listed:
    """目録の 1 項目を形で照らして読む 展開先の外を指す名前は断る"""
    if not isinstance(raw, dict):
        raise PackageError("パッケージの目録が壊れています（項目）")
    path, size, sha256 = raw.get("path"), raw.get("size"), raw.get("sha256")
    if not isinstance(path, str) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise PackageError("パッケージの目録が壊れています（項目の形）")
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise PackageError("パッケージの目録が壊れています（SHA-256）")
    if project:
        if path != PROJECT_ENTRY:
            raise PackageError("パッケージの目録が壊れています（作品の場所）")
    elif not _ENTRY_NAME.fullmatch(path):
        # 展開先の外（``..`` や絶対の場所）へ書かせない
        raise PackageError(f"パッケージに扱えない名前があります: {path!r}")
    return _Listed(path, size, sha256)


def _safe_folder_name(name: str) -> str:
    """作品の名前から、フォルダの名前に使える物を作る"""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return cleaned[:60] or "作品"


def _unique_folder(root: Path, name: str) -> Path:
    """``root`` の下の、まだ無いフォルダの場所（同じ名前なら番号を付ける）"""
    candidate = root / name
    number = 2
    while candidate.exists():
        candidate = root / f"{name} ({number})"
        number += 1
    return candidate


def open_package(
    package: Path,
    root: Path,
    *,
    progress: Progress = lambda _fraction, _text: None,
    cancelled: Callable[[], bool] = lambda: False,
    expect: Expected | None = None,
) -> OpenedPackage:
    """``package`` を ``root`` の下の新しいフォルダへ展開し、展開した先を指す作品を置く

    中身は SHA-256 で照らし、1 つでも違えば展開した物を消して断る（壊れた・途中で切れた
    パッケージを開かない） 空きが足りなければ書き始める前に断る
    ``expect`` を渡すと、開いたファイルが人の選んだときのままかを照らし、違えば断る
    """
    handle, zipped = _open_checked(Path(package), expect)
    with handle, zipped:
        manifest = _read_manifest(zipped)
        project_listed = _listed(manifest.get("project"), project=True)
        raw_entries = manifest.get("entries")
        if not isinstance(raw_entries, list):
            raise PackageError("パッケージの目録が壊れています（entries）")
        listed = [_listed(raw) for raw in raw_entries]
        names = [item.path for item in listed]
        if len(set(names)) != len(names):
            raise PackageError("パッケージの目録が壊れています（同じ名前が 2 つ）")
        for item in (*listed, project_listed):
            try:
                info = zipped.getinfo(item.path)
            except KeyError as exc:
                raise PackageError(
                    f"パッケージの中身が足りません（途中で切れています）: {item.path}"
                ) from exc
            if info.file_size != item.size:
                raise PackageError(f"パッケージの中身の大きさが目録と違います: {item.path}")

        # 一覧の形も展開の前に確かめる 展開の後に読み違えると、取り出したフォルダが残ったまま
        # 失敗し、開き直すたびにフォルダが増える（#449 の CodeRabbit の指摘）
        fonts = _names_list(manifest, "fonts")
        scripts = _names_list(manifest, "scripts")
        # 記していなければ空 欄の在りかを記さない形は公開していない
        fields = set(_names_list(manifest, "file_fields"))
        if project_listed.size > _PROJECT_LIMIT:
            # 作品はまるごと読んで JSON にする 取り出し始める前に断り、仮のフォルダも作らない
            raise PackageError(f"パッケージの作品が大きすぎます（{_PROJECT_LIMIT >> 20} MB まで）")
        total = sum(item.size for item in listed) + project_listed.size
        root.mkdir(parents=True, exist_ok=True)
        _require_space(root, total, "展開先")

        name = manifest.get("name") if isinstance(manifest.get("name"), str) else ""
        folder = _unique_folder(root, _safe_folder_name(str(name)))
        work = root / f".{folder.name}.{uuid.uuid4().hex[:8]}.writing"
        work.mkdir()
        try:
            project_bytes = _verified_bytes(zipped, project_listed)
            done = 0
            for item in listed:
                if cancelled():
                    raise Cancelled
                progress(done / max(total, 1), f"{item.path} を取り出しています")
                destination = work / item.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                _extract(zipped, item, destination, cancelled)
                done += item.size
            project = _relocated(project_bytes, folder, set(names), str(name), fields)
            folders = folder_kinds(project)
            if folders:
                # フォルダの欄は書き出しで断っているので、値があるのは細工したパッケージ 置いた
                # 場所（共有のフォルダなど）を描くたびに読みに行くので開かない（#449 の CodeRabbit
                # の指摘）
                raise PackageError(
                    "フォルダを指すエフェクトに値があるので、このパッケージは開けません: "
                    + "、".join(folders[:10])
                )
            project_path = work / f"{_safe_folder_name(project.name)}{SUFFIX}"
            # 置き換えた後の場所で書く（展開した素材を指す） 書くのは仮のフォルダの中
            save_project(project, project_path)
            work.rename(folder)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    progress(1.0, "取り出しました")
    made_by = manifest.get("app_version")
    return OpenedPackage(
        project_path=folder / project_path.name,
        project=project,
        folder=folder,
        fonts=fonts,
        scripts=scripts,
        made_by=made_by if isinstance(made_by, str) else "",
    )


def _open_checked(package: Path, expect: Expected | None) -> tuple[IO[bytes], zipfile.ZipFile]:
    """パッケージを 1 度だけ開き、その開いた物の身元を ``expect`` と照らしてから ZIP として読む

    選んでから展開を始めるまでの間に、同じ名前へ別のファイルを置かれると、人が選んでいない物を
    展開する 名前で照らしてから開き直すと、その間にも差し替えられるので、開いた物そのもので照らし、
    照らした物だけを読む（#449 の CodeRabbit の指摘）
    """
    try:
        handle: IO[bytes] = package.open("rb")
    except OSError as exc:
        raise PackageError(f"パッケージを開けません: {exc}") from exc
    try:
        if expect is not None:
            status = os.fstat(handle.fileno())
            identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
            if identity != expect.identity:
                raise PackageError("パッケージが選んだ後に変わったので、開きませんでした")
        try:
            zipped = zipfile.ZipFile(handle)
        except (zipfile.BadZipFile, OSError) as exc:
            raise PackageError(
                f"パッケージを開けません（壊れているか、途中で切れています）: {exc}"
            ) from exc
    except BaseException:
        handle.close()
        raise
    return handle, zipped


def _names_list(manifest: dict[str, Any], key: str) -> tuple[str, ...]:
    """目録の名前の一覧（書体・スクリプト・欄）を形で照らして読む"""
    raw = manifest.get(key, [])
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise PackageError(f"パッケージの目録が壊れています（{key}）")
    return tuple(raw)


def _verified_bytes(zipped: zipfile.ZipFile, item: _Listed) -> bytes:
    """中身を読み、大きさと SHA-256 を目録と照らす"""
    try:
        data = _bounded_read(zipped, item.path, _PROJECT_LIMIT, "作品")
    except (zipfile.BadZipFile, OSError) as exc:
        raise PackageError(f"パッケージの中身が壊れています: {item.path} ({exc})") from exc
    if len(data) != item.size or hashlib.sha256(data).hexdigest() != item.sha256:
        raise PackageError(f"パッケージの中身が壊れています（SHA-256 が合わない）: {item.path}")
    return data


def _extract(
    zipped: zipfile.ZipFile, item: _Listed, destination: Path, cancelled: Callable[[], bool]
) -> None:
    """中身を 1 つ取り出しながら SHA-256 を照らす 違えば断る"""
    digest = hashlib.sha256()
    written = 0
    try:
        with zipped.open(item.path) as reader, destination.open("xb") as writer:
            source: IO[bytes] = reader
            while chunk := source.read(_CHUNK):
                if cancelled():
                    raise Cancelled
                written += len(chunk)
                if written > item.size:
                    break
                digest.update(chunk)
                writer.write(chunk)
    except (zipfile.BadZipFile, OSError) as exc:
        raise PackageError(f"パッケージの中身を取り出せません: {item.path} ({exc})") from exc
    if written != item.size or digest.hexdigest() != item.sha256:
        raise PackageError(f"パッケージの中身が壊れています（SHA-256 が合わない）: {item.path}")


def _relocated(
    project_bytes: bytes,
    folder: Path,
    names: set[str],
    name: str,
    fields: set[str],
) -> Project:
    """中の相対の場所を、展開した先の場所へ付け替えた作品"""
    try:
        project = project_from_dict(json.loads(project_bytes.decode("utf-8")))
    except (ValueError, ProjectFileError) as exc:
        raise PackageError(f"パッケージの中の作品が読めません: {exc}") from exc
    for item in project.media:
        place = item.path.as_posix()
        if place not in names and not _UNUSED_NAME.fullmatch(place):
            raise PackageError(f"作品が指す素材がパッケージに入っていません: {item.display_name}")

    def file(value: str) -> str:
        """ファイルの欄の値を、展開した先の場所へ付け替える"""
        # 目録に無い場所は、パッケージの外を指す 細工したパッケージが共有のフォルダ（UNC）を
        # 指させると、描くたびにその機械へつなぎ、Windows の認証を試みる 展開した先の無い場所へ
        # 置き換え、外へ読みに行かせない（#449 の CodeRabbit の指摘）
        if value in names:
            return str(folder / value)
        return str(folder / MISSING_FOLDER / _safe_file_name(value))

    def unknown(where: str, value: str) -> str:
        """この PC に無いスクリプトの欄の値を付け替える"""
        # この PC に無いスクリプトの欄 書き出した側が記したファイルの欄（fields）は、ファイルの
        # 欄と同じに付け替える（後でスクリプトを入れても画像が見つかり、外も指さない）
        # 記していない欄は文字の欄なので触らない 場所に見えるだけの文（/字幕 など）を書き換えると、
        # スクリプトを入れた後の出力が変わる（#449 の CodeRabbit の指摘）
        # 名前が同じだけの文は付け替えない たまたま中の名前と同じ文の欄まで書き換わる
        # 外を指す文字（共有のフォルダ・ドライブから始まる場所）は、記していなくても展開した先の
        # 無い場所へ置き換える 細工したパッケージが欄を記さずに置くと、後でスクリプトを入れた
        # ときに、共有のフォルダなら描くたびその機械へつないで Windows の認証を試み、手元の場所
        # なら人が選んでいないファイルが出力に混ざる（#449 の CodeRabbit の指摘） 正しく書き出した
        # パッケージでは、外を指す値はファイルの欄（記してある）にしか入らない
        if where in fields:
            return file(value)
        if _outside_place(value):
            return str(folder / MISSING_FOLDER / _safe_file_name(value))
        return value

    relocated = map_paths(
        project,
        lambda item: (
            folder / item.path.as_posix()
            if item.path.as_posix() in names
            else folder / UNUSED_PREFIX / _safe_file_name(item.path.as_posix())
        ),
        file,
        unknown=unknown,
    )
    if name and (not relocated.name or relocated.name == "無題"):
        relocated = replace(relocated, name=name)
    return relocated
