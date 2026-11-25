r"""配る zip の部品の一覧（SBOM）を、zip そのものから作る・照らす・比べる（#517）

    .venv\Scripts\python.exe tools\sbom.py write                 （dist の今の版の zip から作る）
    .venv\Scripts\python.exe tools\sbom.py write --archive 別の.zip --out 一覧.cdx.json
    .venv\Scripts\python.exe tools\sbom.py check 配る.zip 一覧.cdx.json   （作り直して同じか）
    .venv\Scripts\python.exe tools\sbom.py diff 前の.cdx.json 今の.cdx.json

形式は CycloneDX 1.6 の JSON（https://cyclonedx.org/specification/overview/） 部品ごとに
版・使用許諾・purl（Python の包み）・CPE（脆弱性の照合に使う名前が決まっている物だけ）を書き、
DLL には SHA-256 を付ける zip そのものの SHA-256 は一覧の頭（``metadata.component``）に書く

**読むのは配る zip だけ**（と、どの DLL がどの部品かの表 ``build_package.NATIVE_LICENSES``）
組み立てた機械の Python 環境は見ない 開発用の .venv と配る zip の部品は版が違うことがある
（0.2.3 で collect_sources.py が正しい zip を NG にした）

何を積んだかは、組み立てが zip に入れた出どころの記録（``bundled-files.json``
``build_package.bundle_record``）から読む 積んだファイル 1 つずつに、どの包みから来たか・
Python 本体か・本体（sashimono）か・組み立てが作った物かが書いてある zip の中身からフォルダの
名前や拡張子で推すと当てはめに漏れが出た（PR #548 のレビューで 3 度続いた）ので、推さない
部品の名前・版・使用許諾は zip に入れた使用許諾の一覧（``THIRD_PARTY_NOTICES.txt``）の表から引く

次のどれかがあれば一覧を書かずに止まる（誤った一覧を配らない）

- zip のファイルが出どころの記録に無い・記録の出どころが分からない・記録にあるが zip に無い
- 積んだ包みが一覧の表に無い（一覧への足し忘れ） 包みの版と一覧の表の版が違う（版違い）
- av.libs・lupa の DLL が、どの部品の物か決まっていない
- 写しのフォルダが、積んだ包みにもリポジトリの写しにも当たらない
- 書き付け（``build-info.json``）・出どころの記録・使用許諾の一覧・Python の DLL が無い

止めないのは「一覧の表にあるが zip に入っていない」物だけ（表は積む候補を広めに書いてある）
表にあっても積んでいない物は一覧に載せず、何を外したかを出す

ネットへは出ない 上流の置き場が消えても作れる（0.2.4 は上流の取得元が消えてリリースが
止まった 一覧の生成まで同じ理由で止めない）
同じ zip からは同じ一覧ができる（時刻を入れず、通し番号も zip の SHA-256 から決める）
release.py はこれを使って、下書きに上がった一覧が zip と合うかを作り直して確かめる
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import re
import sys
import uuid
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# DLL と部品の対応・zip の置き場の名前は組み立ての道具の物を使う 書き写すと片方だけ直して食い違う
# （置き場の名前は build_package の頭で決まる）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_package  # noqa: E402
import collect_sources  # noqa: E402
from package_notices import canonical_name  # noqa: E402

from sashimono import __version__  # noqa: E402
from sashimono.update.package import BUILD_INFO_NAME  # noqa: E402

#: 一覧の資産の名前 版を入れない 前に公開した版の一覧を ``releases/latest/download/`` の
#: 決まった URL で読めるようにする（目録 update.json と同じ置き方）
BOM_NAME = f"{build_package.ARCHIVE_PREFIX}-windows-x64.cdx.json"
#: CycloneDX の版 1.6 は 2024 に出た版で、主な読み手（Dependency-Track・Trivy・Grype）が読める
SPEC_VERSION = "1.6"
#: 一覧を作った道具の名前（``metadata.tools``）
TOOL_NAME = "tools/sbom.py"
#: 通し番号を zip の SHA-256 から決めるための名前空間（同じ zip なら同じ番号）
_SERIAL_NAMESPACE = uuid.UUID("6f1c3c64-1bb0-4f1e-9a0e-6b4c2a8d5e17")

_APP = build_package.APP_NAME
_LICENSES = f"{_APP}/{build_package.LICENSES_DIR}/"
_INTERNAL = f"{_APP}/_internal/"

#: 組み立ての記録に載らないが :func:`build_package.assemble` などが置く物
#: （``Sashimono`` からの相対）
ASSEMBLED_FILES = frozenset(
    (
        *build_package.ASSEMBLED,
        *build_package.ASSEMBLED_INFO,
    )
)
ASSEMBLED_FOLDERS = tuple(f"{folder}/" for folder in build_package.ASSEMBLED_FOLDERS)

#: 脆弱性の照合に使う名前（CPE の作り手と製品） NVD で名前が決まっていると確かめた物だけ
#: 推測で書くと、照合で別の製品に当たるか、当たるはずの物を外す 部品の名前の頭の語で引く
CPE_PRODUCTS: dict[str, tuple[str, str]] = {
    "python": ("python", "python"),
    "qt": ("qt", "qt"),
    "ffmpeg": ("ffmpeg", "ffmpeg"),
    "zlib": ("zlib", "zlib"),
    "libwebp": ("webmproject", "libwebp"),
    "libvpx": ("webmproject", "libvpx"),
    "libiconv": ("gnu", "libiconv"),
    "dav1d": ("videolan", "dav1d"),
    "opus": ("opus-codec", "opus"),
    "openssl": ("openssl", "openssl"),
    "sqlite": ("sqlite", "sqlite"),
}

#: Python 本体と一緒に入る DLL を、部品ごとに分ける（DLL の名前の頭 → 部品の名前）
#: 一覧の表は「Python と同梱の OpenSSL・libffi など」の 1 行にまとめているが、脆弱性は部品ごとの版で
#: 決まる まとめたまま Python の版を付けると、OpenSSL や SQLite の照合が外れる（PR #548 の 4 回目）
PYTHON_PARTS: tuple[tuple[str, str], ...] = (
    ("python3", "Python"),
    ("libcrypto", "OpenSSL"),
    ("libssl", "OpenSSL"),
    ("sqlite3", "SQLite"),
    ("libffi", "libffi"),
)
#: Python 本体（CPython）の使用許諾 同梱の部品の分は Python の LICENSE.txt の中に並ぶ
PYTHON_LICENSE = [{"expression": "PSF-2.0"}]
BUNDLED_WITH_PYTHON = [
    {
        "license": {
            "name": "licenses\\Python\\LICENSE.txt に Python に同梱の部品として書かれた使用許諾"
        }
    }
]

#: SPDX の式として読める使用許諾の欄 これに当たらない書き方（文章）は名前として残す
#: 識別子は版の付いた形（``GPL-2.0-or-later``）か、版の無い決まった名前だけにする 「LGPL」の
#: ような版の無い略し方を式として書くと、SPDX に無い識別子になり読み手が捨てる
_SPDX_ID = r"(?:[A-Za-z0-9.+]+-[A-Za-z0-9.+-]+|MIT|Zlib|ISC|0BSD|Unlicense)"
_SPDX = re.compile(rf"{_SPDX_ID}(?: (?:AND|OR|WITH) {_SPDX_ID})*")
#: 版として読める欄 「numpy に同梱の版」のような文章は版にしない
_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]*")


class BomError(Exception):
    """一覧を作れない 中身は止めた理由の並び"""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


# --- 使用許諾の一覧の表を読む -------------------------------------------------------


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Markdown の表を（見出し, 行の並び）で返す 区切りの行（``|---|``）は外す"""
    found: list[tuple[list[str], list[list[str]]]] = []
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.strip().startswith("|"):
            cells = _cells(line)
            if header is None:
                header = cells
            elif not all(re.fullmatch(r":?-+:?", cell) for cell in cells if cell):
                rows.append(cells)
            continue
        if header is not None:
            found.append((header, rows))
        header, rows = None, []
    return found


def _ticks(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def _head(cell: str) -> str:
    """括弧の前（「16.1.0（MSYS2）」→「16.1.0」）"""
    return cell.split("（", 1)[0].strip()


def license_entry(text: str) -> list[dict[str, Any]]:
    """使用許諾の欄を CycloneDX の ``licenses`` へ SPDX の式なら式、文章なら名前で残す"""
    head = _head(text)
    if head and _SPDX.fullmatch(head):
        return [{"expression": head}]
    return [{"license": {"name": text.strip()}}] if text.strip() else []


def version_of(cell: str) -> str | None:
    head = _head(cell).removeprefix("コミット ").strip()
    return head if _VERSION.fullmatch(head) else None


@dataclass
class PackageRow:
    """一覧の表の Python の包み 1 行"""

    name: str
    version: str
    license: str
    source: str


@dataclass
class PartRow:
    """一覧の表の DLL・同梱の部品 1 行"""

    name: str
    version_cell: str
    place: str
    license: str
    #: 写しの置き場（``licenses`` の下のフォルダの名前）
    copies: set[str]
    #: 部品のファイルの名前の型（``VCRUNTIME140*.dll`` など）
    patterns: list[str]


def notice_rows(text: str) -> tuple[dict[str, PackageRow], list[PartRow]]:
    """使用許諾の一覧から、Python の包みの表と DLL の表を読む

    「同上」は上の行の値を引き継ぐ 表の書き方が変わってどちらの表も見つからなければ止める
    （空の一覧を書くより、止めて表の読み方を直す方がよい）
    """
    packages: dict[str, PackageRow] = {}
    parts: list[PartRow] = []
    for header, rows in tables(text):
        if header[:2] == ["配布名", "版"]:
            previous = PackageRow("", "", "", "")
            for cells in rows:
                ticks = _ticks(cells[0])
                if not ticks or len(cells) < 4:
                    continue
                license_text = previous.license if cells[2] == "同上" else cells[2]
                source = previous.source if cells[3] == "同上" else cells[3]
                row = PackageRow(ticks[0], cells[1], license_text, source)
                packages[canonical_name(row.name)] = row
                previous = row
        elif header[:3] == ["部品", "版", "置き場"]:
            for cells in rows:
                if len(cells) < 5:
                    continue
                copies = {tick.split("\\", 1)[0] for tick in _ticks(cells[4]) if "\\" in tick}
                patterns = [tick for tick in _ticks(cells[0]) if "*" in tick]
                places = _ticks(cells[2])
                place = places[0].replace("\\", "/") if places else ""
                parts.append(PartRow(_head(cells[0]), cells[1], place, cells[3], copies, patterns))
    if not packages or not parts:
        raise BomError(["使用許諾の一覧に、Python の包みの表か DLL の表が見つからない"])
    return packages, parts


# --- zip から一覧を作る -----------------------------------------------------------


@dataclass
class Found:
    """zip の中身から作った部品 1 つ"""

    order: int
    name: str
    version: str | None
    licenses: list[dict[str, Any]]
    purl: str | None = None
    cpe: str | None = None
    source: str = ""
    evidence: str = ""
    note: str = ""
    files: dict[str, str] = field(default_factory=dict)

    def component(self) -> dict[str, Any]:
        ref = f"{self.name}@{self.version or '-'}"
        data: dict[str, Any] = {"type": "library", "bom-ref": ref, "name": self.name}
        if self.version:
            data["version"] = self.version
        if self.licenses:
            data["licenses"] = self.licenses
        if self.purl:
            data["purl"] = self.purl
        if self.cpe:
            data["cpe"] = self.cpe
        if self.source.startswith("http"):
            data["externalReferences"] = [{"type": "vcs", "url": self.source}]
        if self.note:
            data["properties"] = [{"name": "sashimono:version-note", "value": self.note}]
        if self.evidence:
            data["evidence"] = {"occurrences": [{"location": self.evidence}]}
        if self.files:
            data["components"] = [
                {
                    "type": "file",
                    "bom-ref": f"{ref}#{name}",
                    "name": name,
                    "hashes": [{"alg": "SHA-256", "content": digest}],
                }
                for name, digest in sorted(self.files.items())
            ]
        return data


def _cpe(name: str, version: str | None) -> str | None:
    words = name.split()
    if not version or not words:
        return None
    product = CPE_PRODUCTS.get(words[0].lower())
    if product is None:
        return None
    return f"cpe:2.3:a:{product[0]}:{product[1]}:{version}:*:*:*:*:*:*:*"


def _digest(archive: zipfile.ZipFile, name: str) -> str:
    return hashlib.sha256(archive.read(name)).hexdigest()


@dataclass
class Record:
    """組み立てが書いた出どころの記録（``bundled-files.json``）"""

    #: 配布名（PEP 503 でそろえた物）→（表記どおりの名前, 版）
    packages: dict[str, tuple[str, str]]
    #: ``Sashimono`` からの相対の場所 → 出どころ（包みの配布名か python・sashimono・build）
    files: dict[str, str]


def read_record(archive: zipfile.ZipFile) -> Record:
    """出どころの記録を読む 無い・形が違うなら :class:`BomError`

    記録の無い zip（一覧を作る前の組み立て）からは作らない 中身から推して作ると、
    推し方の漏れがそのまま一覧の欠けになる
    """
    name = f"{_APP}/{build_package.BUNDLED_FILES_NAME}"
    try:
        data = json.loads(archive.read(name))
        packages = {
            str(key): (str(value["name"]), str(value["version"]))
            for key, value in data["packages"].items()
        }
        files = {str(key): str(value) for key, value in data["files"].items()}
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise BomError(
            [f"zip に読める {build_package.BUNDLED_FILES_NAME}（出どころの記録）が無い（{exc!r}）"]
        ) from exc
    return Record(packages, files)


def coverage(names: list[str], record: Record) -> list[str]:
    """zip のファイルと出どころの記録が 1 対 1 で合うか 合わない所を返す"""
    problems: list[str] = []
    inside = {name[len(_APP) + 1 :] for name in names if name.startswith(f"{_APP}/")}
    for name in names:
        if not name.startswith(f"{_APP}/") and name != build_package.INSTALL_NOTE_NAME:
            problems.append(f"{name} は組み立ての記録に無い")
    for relative in sorted(inside):
        if relative in record.files:
            if record.files[relative] == build_package.OWNER_UNKNOWN:
                problems.append(f"{relative} の出どころが分からない")
            continue
        if relative in ASSEMBLED_FILES or relative.startswith(ASSEMBLED_FOLDERS):
            continue
        problems.append(f"{relative} は組み立ての記録に無い（どの部品の物か決まっていない）")
    owners = set(record.packages) | {
        build_package.OWNER_OWN,
        build_package.OWNER_BUILD,
        build_package.OWNER_PYTHON,
    }
    for relative, owner in sorted(record.files.items()):
        if relative not in inside:
            problems.append(f"{relative} は組み立ての記録にあるが zip に無い")
        elif owner not in owners and owner != build_package.OWNER_UNKNOWN:
            problems.append(f"{relative} の出どころ {owner} が積んだ包みの記録に無い")
    return problems


def _python_packages(
    names: list[str],
    record: Record,
    rows: dict[str, PackageRow],
    copies: set[str],
    problems: list[str],
) -> list[Found]:
    """積んだ Python の包み 出どころの記録の包みを、一覧の表と写しのフォルダに照らす"""
    folders = {
        PurePosixPath(name[len(_LICENSES) :]).parts[0]
        for name in names
        if name.startswith(_LICENSES) and name.count("/") >= 3
    }
    expected: set[str] = set()
    found: list[Found] = []
    for key, (name, version) in sorted(record.packages.items()):
        row = rows.get(key)
        if row is None:
            problems.append(f"{name} {version} を積んでいるが、使用許諾の一覧の表に無い")
            continue
        if row.version != version:
            problems.append(
                f"{name} は {version} を積んでいるが、使用許諾の一覧の表は {row.version}"
            )
            continue
        folder = f"{name}-{version}"
        expected.add(folder)
        if folder in folders:
            evidence = f"{_LICENSES}{folder}"
        else:
            placed = sorted(path for path, owner in record.files.items() if owner == key)
            evidence = f"{_APP}/{PurePosixPath(placed[0]).parent}" if placed else ""
        found.append(_package(row, evidence))
    for folder in sorted(folders - expected - copies - {"Python"}):
        # 前の組み立ての写しや、積んでいない包みの写しを配らない
        problems.append(f"licenses/{folder} が積んだ包みにもリポジトリの写しにも当たらない")
    return found


def _package(row: PackageRow, evidence: str) -> Found:
    key = canonical_name(row.name)
    return Found(
        0,
        row.name,
        row.version,
        license_entry(row.license),
        purl=f"pkg:pypi/{key}@{row.version}",
        source=row.source,
        evidence=evidence,
    )


#: lupa の中の Lua 5.x の拡張 lupa の包みの物で、使用許諾も lupa の dist-info にある
_LUPA_OWN = re.compile(r"lua5\d\.[^/]*\.pyd", re.IGNORECASE)


def _parts(
    archive: zipfile.ZipFile,
    names: list[str],
    record: Record,
    rows: list[PartRow],
    python: str | None,
    problems: list[str],
) -> tuple[list[Found], list[str]]:
    """DLL の表から、積んだ部品を数える 戻り値の 2 つ目は表にあるが積んでいない物"""
    native_copies = set(build_package.NATIVE_LICENSES.values())
    by_copy = {copy: row for row in rows for copy in row.copies}
    reached: dict[int, Found] = {}

    def entry(row: PartRow) -> Found:
        index = rows.index(row)
        if index not in reached:
            version = version_of(row.version_cell)
            note = "" if version else row.version_cell
            reached[index] = Found(
                1 + index,
                row.name,
                version,
                license_entry(row.license),
                cpe=_cpe(row.name, version),
                evidence=f"{_APP}/{row.place}" if row.place else "",
                note=note,
            )
        return reached[index]

    # av.libs と lupa の DLL は 1 つずつ部品へ当てる 下のフォルダの物も含め、当たらない物は止める
    for relative in sorted(record.files):
        parts = PurePosixPath(relative).parts
        if (
            len(parts) < 3
            or parts[0] != "_internal"
            or parts[1] not in build_package.NATIVE_FOLDERS
        ):
            continue
        if PurePosixPath(relative).suffix.lower() not in (".dll", ".pyd"):
            continue
        filename = parts[-1]
        if parts[1] == "lupa" and len(parts) == 3 and _LUPA_OWN.fullmatch(filename):
            continue
        copy = build_package.NATIVE_LICENSES.get(build_package.native_base_name(filename))
        if copy is None or len(parts) != 3:
            problems.append(f"{'/'.join(parts[1:])} がどの部品の物か決まっていない")
            continue
        row = by_copy.get(copy)
        if row is None:
            problems.append(f"{parts[1]}/{filename} の部品（{copy}）が使用許諾の一覧の表に無い")
            continue
        if not any(other.startswith(f"{_LICENSES}{copy}/") for other in names):
            problems.append(f"{parts[1]}/{filename} の写し licenses/{copy} が zip に無い")
        entry(row).files[relative] = _digest(archive, f"{_APP}/{relative}")

    left: list[str] = []
    split: list[Found] = []
    for row in rows:
        if row.copies & native_copies:
            # DLL を数えて当てる部品 当たらなければ積んでいない
            if rows.index(row) not in reached:
                left.append(row.name)
            continue
        inside = [path for path in record.files if row.place and path.startswith(row.place)]
        if not inside:
            left.append(row.name)
            continue
        if row.name.lower().startswith("python"):
            # 1 行にまとめた Python と同梱の部品は、部品ごとに分けて載せる
            split += _python_parts(archive, record, inside, rows, row, python)
            continue
        found = entry(row)
        taken = {path for part in reached.values() for path in part.files}
        for relative in inside:
            filename = PurePosixPath(relative).name.lower()
            if row.patterns:
                hit = any(fnmatch.fnmatch(filename, pattern.lower()) for pattern in row.patterns)
            else:
                # 名前の型の無い行（OpenBLAS・PortAudio・freeglut など）は、置き場の DLL と拡張を
                # その部品の物と見る 型が無いからと外すと、ハッシュが載らず差し替えに気付けない
                # ほかの部品へ当てた物（lupa の LuaJIT など）は取らない
                hit = _is_binary(relative) and relative not in taken
            if hit:
                found.files[relative] = _digest(archive, f"{_APP}/{relative}")
    every = [*reached.values(), *split]
    for part in every:
        _version_from_files(archive, part)
    return every, left


#: ハッシュを付けるファイル（実行されるもの） .pyc やデータはハッシュの比べに入れない
_BINARIES = (".dll", ".pyd", ".exe")


def _is_binary(relative: str) -> bool:
    return PurePosixPath(relative).suffix.lower() in _BINARIES


def place_leftovers(
    archive: zipfile.ZipFile, record: Record, packages: list[Found], parts: list[Found]
) -> dict[str, str]:
    """どの部品にもハッシュの載っていない DLL・拡張・exe を、出どころの部品へ載せる

    表の部品へ当たらない物（PySide6 の拡張・numpy の拡張・標準ライブラリの拡張など）を残すと、
    中身の差し替えに気付けない 包みの物はその包みへ、Python 本体の物は Python へ載せ、本体
    （sashimono）と組み立てが作った物（Sashimono.exe など）は戻り値で返して、一覧の頭に載せる
    """
    taken = {path for part in [*packages, *parts] for path in part.files}
    by_key = {canonical_name(part.name): part for part in packages}
    python = next((part for part in parts if part.name == "Python"), None)
    own: dict[str, str] = {}
    for relative, owner in sorted(record.files.items()):
        if relative in taken or not _is_binary(relative):
            continue
        digest = _digest(archive, f"{_APP}/{relative}")
        if owner in by_key:
            by_key[owner].files[relative] = digest
        elif owner == build_package.OWNER_PYTHON and python is not None:
            python.files[relative] = digest
        else:
            own[relative] = digest
    return own


def _dll_version(archive: zipfile.ZipFile, relative: str) -> str | None:
    """DLL の版情報の ProductVersion 4 つ目の数が 0 なら落とす（3.50.4.0 → 3.50.4）"""
    found = collect_sources.dll_version_text(archive.read(f"{_APP}/{relative}"), "ProductVersion")
    if not found or not _VERSION.fullmatch(found.strip()):
        return None
    text = found.strip()
    return text[:-2] if text.count(".") == 3 and text.endswith(".0") else text


def _version_from_files(archive: zipfile.ZipFile, part: Found) -> None:
    """表に版の書けない部品（VC++ の実行時の部品など）は、DLL の版情報がそろっていればその版にする

    版の無い部品は脆弱性の照合で当たらない 版情報の違う DLL が混ざる部品は決められないので付けない
    """
    if part.version or not part.files:
        return
    versions = {_dll_version(archive, relative) for relative in part.files}
    if len(versions) == 1 and None not in versions:
        part.version = versions.pop()
        part.cpe = _cpe(part.name, part.version)
        part.note = "DLL の版情報から読んだ版"


def _python_parts(
    archive: zipfile.ZipFile,
    record: Record,
    inside: list[str],
    rows: list[PartRow],
    row: PartRow,
    python: str | None,
) -> list[Found]:
    """Python 本体と一緒に入った DLL を部品ごとに分ける

    Python（CPython）の版は python3XX.dll から読む（表は「3.14」までしか書かない） OpenSSL・SQLite
    などは各 DLL の版情報から読む ほかの行の型に当たる DLL（VC++ の実行時の部品）はその行に任せる
    """
    others = [pattern.lower() for other in rows if other is not row for pattern in other.patterns]
    grouped: dict[str, Found] = {}
    order = 1 + rows.index(row)
    for relative in sorted(inside):
        filename = PurePosixPath(relative).name.lower()
        if record.files[relative] != build_package.OWNER_PYTHON or not filename.endswith(".dll"):
            continue
        if any(fnmatch.fnmatch(filename, pattern) for pattern in others):
            continue
        base = build_package.native_base_name(filename)
        name = next((part for head, part in PYTHON_PARTS if base.startswith(head)), base)
        if name not in grouped:
            is_python = name == "Python"
            grouped[name] = Found(
                order,
                name,
                python if is_python else None,
                PYTHON_LICENSE if is_python else BUNDLED_WITH_PYTHON,
                cpe=_cpe(name, python) if is_python else None,
                evidence=f"{_APP}/{row.place}",
                note="" if is_python else "Python に同梱の版",
            )
        grouped[name].files[relative] = _digest(archive, f"{_APP}/{relative}")
    return [grouped[name] for name in sorted(grouped, key=lambda name: (name != "Python", name))]


@dataclass
class Bom:
    """作った一覧と、表にあるが積んでいないので外した物"""

    data: dict[str, Any]
    left_out: list[str]

    def text(self) -> str:
        return dumps(self.data)


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def generate(path: Path) -> Bom:
    """配る zip から一覧を作る 誤った一覧になる理由があれば :class:`BomError`"""
    problems: list[str] = []
    zip_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        try:
            info = json.loads(archive.read(f"{_APP}/{BUILD_INFO_NAME}"))
            version = str(info["version"])
            abi = str(info["python_abi"])
        except (KeyError, ValueError, TypeError) as exc:
            raise BomError([f"zip に読める {BUILD_INFO_NAME} が無い（{exc}）"]) from exc
        try:
            notices = archive.read(f"{_APP}/{build_package.NOTICES_NAME}").decode("utf-8")
        except KeyError as exc:
            raise BomError([f"zip に {build_package.NOTICES_NAME} が無い"]) from exc
        packages, parts = notice_rows(notices)
        record = read_record(archive)
        problems += coverage(names, record)
        python = build_package.zip_python_version(path)
        if python is None:
            problems.append("zip に Python の DLL が無いか、版を読めない")
        copies = {
            PurePosixPath(name).parts[0]
            for name in build_package.repository_license_files()
            if len(PurePosixPath(name).parts) > 1
        }
        found = _python_packages(names, record, packages, copies, problems)
        native, left = _parts(
            archive,
            names,
            record,
            parts,
            build_package.version_text(python) if python else None,
            problems,
        )
        own_files = place_leftovers(archive, record, found, native) if not problems else {}
    if problems:
        raise BomError(problems)
    left_out = [packages[key].name for key in sorted(set(packages) - set(record.packages))] + left
    components = [
        item.component()
        for item in sorted(found, key=lambda item: canonical_name(item.name))
        + sorted(native, key=lambda item: item.order)
    ]
    app_ref = f"{build_package.PRODUCT_NAME}@{version}"
    data: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid5(_SERIAL_NAMESPACE, zip_sha256)}",
        "version": 1,
        "metadata": {
            "tools": {"components": [{"type": "application", "name": TOOL_NAME}]},
            "component": {
                "type": "application",
                "bom-ref": app_ref,
                "name": build_package.PRODUCT_NAME,
                "version": version,
                # 本体のソースは MIT 配る zip は GPL の部品（x264・x265）を積むので全体を GPL で配る
                "licenses": [{"expression": "MIT"}],
                "hashes": [{"alg": "SHA-256", "content": zip_sha256}],
                # 本体と組み立てが作った物（exe など）のハッシュ 部品の側と同じ形で持つ
                "components": [
                    {
                        "type": "file",
                        "bom-ref": f"{app_ref}#{name}",
                        "name": name,
                        "hashes": [{"alg": "SHA-256", "content": digest}],
                    }
                    for name, digest in own_files.items()
                ],
                "properties": [
                    {"name": "sashimono:archive", "value": path.name},
                    {"name": "sashimono:python-abi", "value": abi},
                    {
                        "name": "sashimono:distributed-under",
                        "value": "GPL（同梱の部品の使用許諾は THIRD_PARTY_NOTICES.txt）",
                    },
                ],
            },
        },
        "components": components,
        "dependencies": [
            {"ref": app_ref, "dependsOn": [item["bom-ref"] for item in components]},
        ],
    }
    return Bom(data, left_out)


# --- 照らす・比べる -----------------------------------------------------------------


def _table(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _sha256(hashes: Any) -> str:
    """ハッシュの並びから SHA-256 の値 無い・空・形が違うなら空

    一覧はほかの人や前の版が書いた物も読む（release.py の 4 と 8） 欄が欠けた一覧で落ちると、
    4 では「違う」と言えず、8 では参考に見せるだけの物のせいで公開まで進めない 空は
    「ハッシュ無し」として比べる（ハッシュの付いた一覧とは違うと出る）
    """
    for entry in _list(hashes):
        entry = _table(entry)
        if entry.get("alg") == "SHA-256" and isinstance(entry.get("content"), str):
            return str(entry["content"])
    return ""


def zip_digest(data: dict[str, Any]) -> str:
    """一覧の頭に書いた zip の SHA-256 無ければ空"""
    return _sha256(_table(_table(_table(data).get("metadata")).get("component")).get("hashes"))


def _index(data: dict[str, Any]) -> dict[str, tuple[str, dict[str, str]]]:
    found: dict[str, tuple[str, dict[str, str]]] = {}
    for item in _list(_table(data).get("components")):
        item = _table(item)
        files = {
            str(_table(inner).get("name")): _sha256(_table(inner).get("hashes"))
            for inner in _list(item.get("components"))
        }
        found[str(item.get("name"))] = (str(item.get("version", "")), files)
    return found


def differences(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """2 つの一覧の違い 足した・外した・版が変わった・版が同じなのに中身が違う

    版が同じなのに DLL の中身が違うのは、上流が同じ版の名前で組み直したか、組み立てで別の物を
    拾った印 版の差よりも先に見る物なので分けて出す
    """
    before, after = _index(old), _index(new)
    lines: list[str] = []
    for name in sorted(set(after) - set(before)):
        lines.append(f"足した: {name} {after[name][0]}".rstrip())
    for name in sorted(set(before) - set(after)):
        lines.append(f"外した: {name} {before[name][0]}".rstrip())
    for name in sorted(set(before) & set(after)):
        (old_version, old_files), (new_version, new_files) = before[name], after[name]
        if old_version != new_version:
            lines.append(f"版が変わった: {name} {old_version} → {new_version}")
        elif old_files != new_files:
            # 名前ごとに比べる 値だけを並べ替えて比べると、2 つの DLL の中身が
            # 入れ替わっても同じに見える
            lines.append(f"版が同じで中身が違う: {name} {new_version}（DLL {len(new_files)} 個）")
    return lines


def mismatches(archive: Path, published: dict[str, Any]) -> list[str]:
    """上げた一覧が、zip から作り直した物と違う所 空なら同じ

    作り直せないとき（zip が誤った一覧になる中身）は :class:`BomError` をそのまま上げる
    """
    expected = generate(archive).data
    if published == expected:
        return []
    found = differences(published, expected)
    if zip_digest(published) != zip_digest(expected):
        found.insert(0, f"zip の SHA-256 が違う（一覧 {zip_digest(published)[:12]}…）")
    return found or ["中身の並びや書き方が、作り直した物と違う"]


def hash_summary(archive: Path, data: dict[str, Any]) -> list[str]:
    """zip の DLL・拡張・exe のうちハッシュを載せた数と、部品ごとの載せた数

    一覧を作り直さずに、ハッシュの載らない物が出ていないかを人とログで見られるようにする
    """
    with zipfile.ZipFile(archive) as opened:
        binaries = {
            name[len(_APP) + 1 :]
            for name in opened.namelist()
            if name.startswith(f"{_APP}/") and _is_binary(name)
        }
    holders = [
        *_list(data.get("components")),
        _table(_table(data.get("metadata")).get("component")),
    ]
    counts = {
        str(_table(item).get("name")): {
            str(_table(inner).get("name")) for inner in _list(_table(item).get("components"))
        }
        for item in holders
    }
    hashed = set().union(*counts.values()) if counts else set()
    done = len(binaries & hashed)
    lines = [f"ハッシュを載せた DLL・拡張・exe {done} 個（zip の中 {len(binaries)} 個）"]
    missing = sorted(binaries - hashed)
    if missing:
        lines.append(f"ハッシュの載っていない物: {'、'.join(missing)}")
    shown = "、".join(f"{name} {len(files)}" for name, files in counts.items() if files)
    lines.append(f"部品ごとのハッシュの数: {shown}")
    return lines


def _print(lines: Iterable[str], lead: str = "") -> None:
    for line in lines:
        print(f"{lead}{line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    writing = commands.add_parser("write", help="配る zip から一覧を作る")
    writing.add_argument("--archive", type=Path, default=None, help="既定は dist の今の版の zip")
    writing.add_argument("--out", type=Path, default=None, help=f"既定は zip の隣の {BOM_NAME}")
    checking = commands.add_parser("check", help="一覧が zip から作り直した物と同じか")
    checking.add_argument("archive", type=Path)
    checking.add_argument("bom", type=Path)
    diffing = commands.add_parser("diff", help="2 つの一覧の違い")
    diffing.add_argument("old", type=Path)
    diffing.add_argument("new", type=Path)
    args = parser.parse_args(argv)

    if args.command == "write":
        archive = args.archive or (
            ROOT / "dist" / f"{build_package.ARCHIVE_PREFIX}-{__version__}-windows-x64.zip"
        )
        out = args.out or archive.with_name(BOM_NAME)
        if out.resolve() == archive.resolve():
            # 下で前の一覧を消すので、同じ場所を渡されると配る zip を消してしまう
            parser.error("--out に --archive と同じ場所は渡せない")
        # 前の一覧を先に消す 今回作れずに終わったとき、前の版の一覧が今の zip の隣に残って
        # 一緒に上げられると、誤った一覧を配ることになる
        out.unlink(missing_ok=True)
        if not archive.is_file():
            print(f"[NG] {archive} が無い（tools/build_package.py で組み立ててから）")
            return 1
        try:
            bom = generate(archive)
        except BomError as exc:
            _print(exc.problems, "[NG] ")
            print("一覧（SBOM）は書かなかった")
            return 1
        temporary = out.with_name(out.name + ".writing")
        temporary.write_text(bom.text(), encoding="utf-8")
        temporary.replace(out)
        count = len(bom.data["components"])
        print(f"できた: {out}（部品 {count} 個 zip の SHA-256 {zip_digest(bom.data)}）")
        if bom.left_out:
            print(f"表にあるが積んでいないので載せていない: {'、'.join(bom.left_out)}")
        # 組み立ての記録（CI のログ）だけで、ハッシュの載り方を確かめられるようにする
        for line in hash_summary(archive, bom.data):
            print(line)
        return 0

    if args.command == "check":
        try:
            found = mismatches(args.archive, json.loads(args.bom.read_text(encoding="utf-8")))
        except BomError as exc:
            _print(exc.problems, "[NG] ")
            return 1
        except (OSError, ValueError) as exc:
            print(f"[NG] {args.bom} を読めない: {exc}")
            return 1
        if found:
            _print(found, "[NG] ")
            return 1
        print(f"[ok] {args.bom.name} は {args.archive.name} から作り直した物と同じ")
        return 0

    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    lines = differences(old, new)
    _print(lines or ["違いは無い"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
