r"""配る zip を作る

    .venv\Scripts\python.exe tools\build_package.py
    .venv\Scripts\python.exe tools\build_package.py --skip-build   （組み立て済みを zip にし直す）

やること

1. PyInstaller で ``Sashimono.exe`` と部品一式（``_internal``）を組み立てる
2. PyInstaller がプラグインごと積んだ Qt の部品のうち、使わない物を外す
   （``UNUSED_QT_PARTS`` 外した物を読む物が残っていれば止まる）
3. 組み立ての記録から、積んだファイルがどの包みから来たかを辿り、包みごとの
   使用許諾の写しを ``licenses`` へ集める 出どころの分からないファイルがあれば止まる
4. 隣にスクリプト置き場（``scripts``）と説明書き・使用許諾の一覧を置く
5. ``dist\SashimonoEdit-<版>-windows-x64.zip`` にまとめる
6. **できた zip を別の場所へ展開し、中の exe で ``--self-check`` を走らせる**
   組み立てた直後のフォルダで確かめると、開発環境の DLL や Python を
   拾って通ってしまう 配るのは zip なので、zip から確かめる

確かめるときの環境変数は最小にする（``PATH`` は Windows の分だけ）
開発機の ``PATH`` に FFmpeg や Python が載っていると、積み忘れがあっても通る

**依存が何も入っていない機械での確認**は、ここではできない（VC++ ランタイムなど
Windows 側の部品は開発機に入っている） CI のまっさらな Windows で、zip だけを持って
確かめる（.github/workflows/package.yml と tools/check_clean_machine.ps1 Issue #33）
CI には GPU が無いので、GL で描く確かめはここ（開発機）で行う
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import locale
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sashimono import __version__  # noqa: E402
from sashimono.app import IMPORT_CHECK_FLAG, SELF_CHECK_FLAG  # noqa: E402
from sashimono.compat.aviutl.catalog import PORTABLE_SCRIPTS_DIR  # noqa: E402
from sashimono.links import REPORT_URL  # noqa: E402
from sashimono.runtime import python_abi  # noqa: E402
from sashimono.selfcheck import UPDATE_CHECK_NAME  # noqa: E402
from sashimono.update.package import BUILD_INFO_NAME, write_build_info  # noqa: E402

#: exe と、zip を展開したときのフォルダの名前
#: 短い名前にする 空白を含むとコマンドから ``--self-check`` を打つときに括りが要る
APP_NAME = "Sashimono"

#: 配る zip の名前の頭 ダウンロードのフォルダで見つけやすいよう製品名（Sashimono Edit）
#: から付ける 展開したフォルダ（APP_NAME）とは別に持つ
ARCHIVE_PREFIX = "SashimonoEdit"

#: 同梱しないもの 追加機能（字幕起こし・AI 連携）は画面のボタンから後で入れる
#: 開発機に入っていると PyInstaller が拾ってしまい、zip が 2 GB を超える
EXCLUDED_MODULES = (
    "faster_whisper",
    "ctranslate2",
    "torch",
    "nvidia",
    "claude_agent_sdk",
    # 開発の道具 動かすのに要らない
    "pytest",
    "hypothesis",
    "mypy",
    "PyInstaller",
)

#: 後から画面のボタンで入れる部品（:data:`EXCLUDED_MODULES` のうち、使う人の手元で pip が入れる物）
#: 配布版に Python の本体は無いので、これらとその依存が使う標準ライブラリは配布版が持って
#: いなければならない PyInstaller は本体が import する物しか積まないため、AI 連携を入れて
#: 送った途端に ``No module named 'zoneinfo'`` で止まった（pydantic が読む 利用者の画面）
RUNTIME_PACKAGES = ("claude-agent-sdk", "faster-whisper", "ctranslate2")
#: 上の包みの import する名前（配布版の exe の中で import してみる）
RUNTIME_MODULES = ("claude_agent_sdk", "faster_whisper", "ctranslate2")

#: 本体も一部を使うので配布版に入り、後から入れる部品も使う包み **下の部品まで全部積む**
#: 配布版に入った包みは、後から入れた置き場の同じ包みより先に読まれる（PyInstaller の
#: 読み込み方が探す道より前に立つ） 本体が使う部品だけを積むと、mcp が読む
#: ``cryptography.hazmat.primitives.ciphers.aead`` が無く、zoneinfo を足した次にそこで
#: 落ちた（組み立ての確かめの exe の中での import で分かった）
SHARED_WITH_ADD_ONS = ("cryptography", "attr", "attrs")

#: 標準ライブラリで積まない物 画面の部品（Tk）・Python 自身の試験・開発の道具
#: どれも Sashimono も後から入れる部品も使わない Tk は DLL と Tcl の書庫で 10 MB を超える
STDLIB_LEFT_OUT = frozenset(
    {
        "tkinter",
        "_tkinter",
        "turtle",
        "turtledemo",
        "idlelib",
        "test",
        "lib2to3",
        "pydoc_data",
        "ensurepip",
        "antigravity",
        "this",
    }
)


def stdlib_names() -> list[str]:
    """配布版に積む標準ライブラリの名前（最上位） この機械で import できる物だけ

    Windows に無い物（``fcntl`` ``curses`` など）は積めないので外す 後から入れる部品も
    Windows では読まない（読む所は ``if sys.platform`` の向こう側）
    """
    names = []
    for name in sorted(sys.stdlib_module_names):
        if name in STDLIB_LEFT_OUT or name.startswith(("_test", "__")):
            continue
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if found is not None:
            names.append(name)
    return names


def stdlib_arguments() -> list[str]:
    """標準ライブラリを丸ごと積む PyInstaller の引数

    1 つずつ足すと、足していない次の物で落ちる（zoneinfo の次は何か、を実物で探すことに
    なる） 包み（``email`` ``xml`` など）は下の部品まで積む 包みの頭だけを積むと、
    ``email.mime.text`` のような下の部品が無い
    """
    arguments: list[str] = []
    for name in stdlib_names():
        found = importlib.util.find_spec(name)
        if found is not None and found.submodule_search_locations is not None:
            arguments += ["--collect-submodules", name]
        else:
            arguments += ["--hidden-import", name]
    return arguments


def runtime_distributions() -> list[importlib.metadata.Distribution]:
    """後から入れる部品と、その依存（この機械に入っている物だけ） 依存は辿れる所まで辿る"""
    found: dict[str, importlib.metadata.Distribution] = {}
    #: 包みと、頼まれた追加の組（``pyjwt[crypto]`` の crypto）
    pending: list[tuple[str, frozenset[str]]] = [(name, frozenset()) for name in RUNTIME_PACKAGES]
    seen: set[tuple[str, frozenset[str]]] = set()
    while pending:
        raw, extras = pending.pop()
        name = canonical_name(raw)
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        found[name] = distribution
        for requirement in distribution.requires or ():
            # 追加の組の依存（``; extra == "crypto"``）は、その組を頼まれたときだけ辿る
            # 頼まれた組を見ないと、mcp が頼む pyjwt[crypto] の cryptography を落とす
            marker = re.search(r"extra\s*==\s*['\"]([^'\"]+)['\"]", requirement)
            if marker is not None and marker.group(1) not in extras:
                continue
            head = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*(\[([^\]]*)\])?", requirement)
            if head is None:
                continue
            wanted = frozenset(e.strip() for e in (head.group(3) or "").split(",") if e.strip())
            pending.append((head.group(1), wanted))
    return list(found.values())


def imported_stdlib(distributions: Iterable[importlib.metadata.Distribution]) -> set[str]:
    """包みの .py が import する標準ライブラリの名前（最上位） 字面で集める（読み込まない）"""
    needed: set[str] = set()
    for distribution in distributions:
        for file in distribution.files or ():
            if file.suffix != ".py":
                continue
            try:
                tree = ast.parse(Path(str(distribution.locate_file(file))).read_bytes())
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    heads = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    heads = [node.module.split(".")[0]]
                else:
                    continue
                needed.update(head for head in heads if head in sys.stdlib_module_names)
    return needed


def bundled_modules(record: Path) -> set[str]:
    """組み立てた配布版にある最上位のモジュールの名前（書庫の .pyc と、隣の .pyd）"""
    collected = ast.literal_eval((record / "COLLECT-00.toc").read_text(encoding="utf-8"))[0]
    _, modules = ast.literal_eval((record / "PYZ-00.toc").read_text(encoding="utf-8"))
    names = {str(entry[0]).split(".")[0] for entry in modules}
    # 書庫を開く前に要る物（struct）は exe の中の荷物（PKG）へ直に入る
    package = record / "PKG-00.toc"
    if package.exists():
        entries = ast.literal_eval(package.read_text(encoding="utf-8"))[2]
        names.update(str(e[0]).split(".")[0] for e in entries if e[2] == "PYMODULE")
    # 起動に要る物（os re codecs など）は書庫ではなく base_library.zip に入る
    base = record / "base_library.zip"
    if base.exists():
        with zipfile.ZipFile(base) as library:
            names.update(PurePosixPath(item).parts[0].split(".")[0] for item in library.namelist())
    for entry in collected:
        name = PurePosixPath(str(entry[0]).replace("\\", "/"))
        if name.suffix == ".pyd" and len(name.parts) == 1:
            names.add(name.name.split(".")[0])
    return names | set(sys.builtin_module_names)


def missing_stdlib(record: Path) -> list[str]:
    """後から入れる部品が import するのに、配布版に無い標準ライブラリ

    この機械に部品が入っていなければ何も見ない（確かめる元が無い 試験のために落とさない）
    Windows で import できない物は数えない（積めず、Windows では読まれない）
    """
    have = bundled_modules(record)
    wanted = sorted(set(stdlib_names()) & imported_stdlib(runtime_distributions()))
    return [name for name in wanted if name not in have]


#: 読み込み方が動的で、PyInstaller が辿れないもの **まとめて積む**
#: lupa は AviUtl に近い Lua を名前で選んで読む（``import_module("lupa.lua51")``）
#: pip は導入ボタンが使う 中身を名前で引くうえ、HTTPS の証明書（データ）も要る
COLLECTED_PACKAGES = ("lupa", "pip")

#: exe の隣に置く説明書き
README_TEXT = f"""Sashimono Edit {__version__}

起動: Sashimono.exe

置き場: %LOCALAPPDATA%\\Programs\\Sashimono へ展開するのがおすすめです 新しい版が出ると、
ソフトが自分で落として確かめ、尋ねてから入れ替えます（〔表示〕→〔設定…〕で切れます）
Program Files のように管理者でないと書けない場所では、自動では入れ替えられません
入れ替えはこのフォルダを丸ごと替えるので、このフォルダの中に自分の物を置かないでください
（設定・スクリプト・テンプレート・入れた実行環境は、別の場所に置かれていて消えません）

AviUtl のスクリプト（.anm2 .obj2 など）は、ソフトの〔互換〕→〔スクリプトフォルダを開く〕で
開くフォルダ（%APPDATA%\\Sashimono\\scripts）へ置けば読み込まれます AviUtl2 が入って
いれば、そちらの Script フォルダも読みます

この隣の {PORTABLE_SCRIPTS_DIR} フォルダも読みますが、新しい版へ入れ替えるときにこの
フォルダごと差し替えると中身が消えます 自分で足すスクリプトは上のフォルダへ置いてください

動かないときは、このフォルダでコマンドを開いて次を打つと、どの部品が
動いていないかが 1 行ずつ出ます（そのまま打つと、結果は窓で出ます）

    Sashimono.exe {SELF_CHECK_FLAG} | more

字幕起こしと AI 連携は、ソフトの中のボタンから必要になったときに入れます
（最初から入れると 2 GB を超えるため）

不具合・要望・使い方の質問は、ソフトの〔ヘルプ〕→〔不具合・要望を送る〕か次から送れます
上の自己診断の結果も一緒に貼ってもらえると助かります

    {REPORT_URL}

使用許諾: Sashimono 本体は MIT（LICENSE.txt） 一緒に入れている部品に GPL の物が
あるため、この zip は全体として GPL の条件で配っています 部品ごとの使用許諾と
ソースの入手先は THIRD_PARTY_NOTICES.txt と licenses フォルダにあります
"""

SCRIPTS_README = f"""AviUtl のスクリプトの置き場です

ここへ .anm2 .obj2 .cam2 .scn2 .tra2（と AviUtl1 世代の .anm .obj）を置くと、
Sashimono を起動し直したときに読み込まれます フォルダに分けて置いても読みます

AviUtl2 が入っている機械では、AviUtl2 の Script フォルダも同じように読みます
（こちらへ写す必要はありません）

動いているか確かめるには Sashimono.exe {SELF_CHECK_FLAG}
"""

#: 使用許諾の写しを置くフォルダ exe の隣とリポジトリの直下で同じ名前にする
#: リポジトリの方には、どの包みも写しを持っていない物（GNU の全文、PyAV の wheel に
#: 入っている DLL と LuaJIT の写し）を置いてある 出どころは THIRD_PARTY_NOTICES.md
LICENSES_DIR = "licenses"

#: 同梱した部品の一覧 リポジトリの THIRD_PARTY_NOTICES.md を、Windows で開きやすい名前で置く
NOTICES_SOURCE = ROOT / "THIRD_PARTY_NOTICES.md"
NOTICES_NAME = "THIRD_PARTY_NOTICES.txt"

#: 使用許諾の写しを dist-info に持っていない包み（配布名を小文字で）
#: 写しが無いことは一覧（THIRD_PARTY_NOTICES.md）に書いてある 足すときも、先に一覧へ
#: 書いてからここへ足す 黙って足すと、写しの無い物を無いまま配ることになる
WITHOUT_LICENSE_FILES = frozenset({"pyopengl"})

#: いつも積む包み ``Sashimono.exe`` の起動部そのものが PyInstaller の物で、組み立ての記録では
#: 作業フォルダの exe として出てくるため、ファイルを辿っても包みに行き着かない
ALWAYS_BUNDLED = ("pyinstaller",)


def pyinstaller_arguments(work: Path, dist: Path) -> list[str]:
    """PyInstaller へ渡す引数

    画面のアプリなので窓を出さない（``--windowed``） 1 ファイルにまとめる形
    （``--onefile``）は使わない 起動のたびに一時フォルダへ全部を展開するので、
    300 MB 近い部品だと起動に数秒余計に掛かり、ウイルス対策にも引っかかりやすい
    もう 1 つ、Qt を LGPL-3.0 で使う以上、使う人が Qt の DLL を差し替えられなければ
    ならない onedir なら DLL は別のファイルのまま置かれる
    """
    arguments = [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name",
        APP_NAME,
        "--icon",
        str(ROOT / "src" / "sashimono" / "resources" / "sashimono.ico"),
        "--paths",
        str(ROOT / "src"),
        "--workpath",
        str(work),
        "--distpath",
        str(dist),
        "--specpath",
        str(work),
        # アイコンやロゴ（.ico .svg） importlib.resources で引くので、
        # データとして積まないと窓のアイコンが出ない
        "--collect-data",
        "sashimono.resources",
    ]
    for package in COLLECTED_PACKAGES:
        arguments += ["--collect-all", package]
    arguments += stdlib_arguments()
    for package in SHARED_WITH_ADD_ONS:
        if importlib.util.find_spec(package) is not None:
            arguments += ["--collect-submodules", package]
    for module in EXCLUDED_MODULES:
        arguments += ["--exclude-module", module]
    arguments.append(str(ROOT / "src" / "sashimono" / "__main__.py"))
    return arguments


def assemble(bundle: Path) -> None:
    """組み立てたフォルダへ、使う人が触る物を足す"""
    scripts = bundle / PORTABLE_SCRIPTS_DIR
    scripts.mkdir(exist_ok=True)
    (scripts / "README.txt").write_text(SCRIPTS_README, encoding="utf-8")
    (bundle / "README.txt").write_text(README_TEXT, encoding="utf-8")
    shutil.copyfile(ROOT / "LICENSE", bundle / "LICENSE.txt")
    shutil.copyfile(NOTICES_SOURCE, bundle / NOTICES_NAME)
    # 自動更新が、落として展開した物が目録の言う版と Python かを確かめるのに読む
    # 組み立てた Python の印を書く 配った版の中の Python と同じ物
    write_build_info(bundle, __version__, python_abi())
    for relative in repository_license_files():
        destination = bundle / LICENSES_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / LICENSES_DIR / relative, destination)


def repository_license_files() -> list[str]:
    """リポジトリの ``licenses`` に置いた写し（``licenses`` からの相対の綴り）

    GNU の全文と、wheel が写しを持っていない部品（av.libs の DLL・LuaJIT）の写し
    zip から確かめるときの見本にもする
    """
    root = ROOT / LICENSES_DIR
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def bundled_sources(record: Path) -> list[Path]:
    """組み立ての記録（PyInstaller の TOC）から、積んだファイルの元の場所を読む

    exe の隣に並ぶファイル（``COLLECT``）だけでは足りない 純 Python の包みは
    exe の中の書庫（``PYZ``）に入り、フォルダには現れない
    """
    collected = ast.literal_eval((record / "COLLECT-00.toc").read_text(encoding="utf-8"))[0]
    _, modules = ast.literal_eval((record / "PYZ-00.toc").read_text(encoding="utf-8"))
    sources: list[Path] = []
    for entry in [*collected, *modules]:
        source = entry[1]
        # 名前空間の包みはファイルを持たず "-" で記録される 辿る物が無い
        if source and source != "-":
            sources.append(Path(source))
    return sources


def _key(path: Path) -> str:
    # 記録の綴りと dist-info の綴りで大文字小文字や区切りが揺れる 揃えないと同じ
    # ファイルを別物と見て、出どころが分からないと言って止まる
    # resolve は使わない 数千のファイルごとに実体を辿ると遅く、辿らなくても揃う
    return os.path.normcase(os.path.normpath(path.absolute()))


def _inside(path: Path, root: Path) -> bool:
    return _key(path).startswith(_key(root) + os.sep)


def distribution_owners() -> dict[str, importlib.metadata.Distribution]:
    """入っているファイル 1 つずつから、それを入れた包みを引く表"""
    owners: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions():
        for file in distribution.files or ():
            owners[_key(Path(str(file.locate())))] = distribution
    return owners


def license_files(
    distribution: importlib.metadata.Distribution,
) -> tuple[list[tuple[str, Path]], list[str]]:
    """包みが dist-info に持っている使用許諾の写し（置く名前、元の場所）と、見つからない物

    名前は METADATA の ``License-File`` から引く PEP 639 以降は ``licenses`` の下、
    それより前は dist-info の直下に置かれる 書いていない古い包みは名前で拾う
    書いてあるのに無い物は 2 つ目に返す 1 つでも見つかれば良しとすると、写しの
    欠けた一式を黙って配る
    """
    inside: dict[PurePosixPath, Path] = {}
    for file in distribution.files or ():
        if len(file.parts) > 1 and file.parts[0].endswith(".dist-info"):
            inside[PurePosixPath(*file.parts[1:])] = Path(str(file.locate()))
    declared = distribution.metadata.get_all("License-File") or []
    found: list[tuple[str, Path]] = []
    missing: list[str] = []
    for name in declared:
        for candidate in (PurePosixPath("licenses", name), PurePosixPath(name)):
            if candidate in inside:
                found.append((name, inside[candidate]))
                break
        else:
            missing.append(name)
    if not declared:
        found = [
            (str(relative), path)
            for relative, path in inside.items()
            if relative.name.upper().startswith(("LICEN", "COPYING", "NOTICE"))
        ]
    return found, missing


def canonical_name(name: str) -> str:
    """配布名の表記揺れをそろえる（PEP 503）

    ``PySide6_Essentials`` と ``PySide6-Essentials`` は同じ包み
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def listed_versions(notices: str) -> dict[str, str]:
    """一覧（THIRD_PARTY_NOTICES.md）の Python の包みの表から、配布名と版を読む

    表の行は ``| `配布名` | 版 | ...`` の形 1 列目の最初の ``` `...` ``` を配布名とする
    """
    versions: dict[str, str] = {}
    for line in notices.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].startswith("`"):
            continue
        name = cells[0].split("`")[1]
        versions[canonical_name(name)] = cells[1]
    return versions


def constraints(notices: str) -> list[str]:
    """一覧（THIRD_PARTY_NOTICES.md）の包みを、一覧の版に留める制約（``uv pip install -c`` の形）

    依存は下限だけで書いてあるので、CI のまっさらな機械で入れるとその日の最新が入り、
    一覧と版が食い違って組み立てが止まる（av 19 で DLL が増え、写しの無い部品になった）
    一覧の版は、使用許諾の写しとソースの添付を揃えた版 組み立てる機械が変わっても、
    この版で組む 上げるときは一覧と写しを先に直す
    版の列が版の形でない行（同梱のファイルの表）は外す
    """
    return [
        f"{name}=={version}"
        for name, version in sorted(listed_versions(notices).items())
        if re.fullmatch(r"\d[0-9A-Za-z.!+]*", version)
    ]


def collect_licenses(
    bundle: Path,
    sources: Iterable[Path],
    own_roots: Sequence[Path],
    *,
    owners: Mapping[str, importlib.metadata.Distribution] | None = None,
    notices: str | None = None,
) -> list[str]:
    """積んだファイルの出どころの包みを数え、使用許諾の写しを ``licenses`` へ集める

    戻り値は配れない理由の一覧 空なら配ってよい 包みの名前を決め打ちの一覧で持つと、
    組み立てる機械に入っている包みが変わったとき（PyInstaller は入っていれば拾う）に
    写しの無い物を黙って配る そのため毎回、実際に積んだ物から数える

    ``own_roots`` は Sashimono 自身のファイルの置き場（ソースと組み立ての作業フォルダ）
    Python 本体のファイルは ``sys.base_prefix`` の下にあり、使用許諾はそこの
    ``LICENSE.txt`` を写す
    """
    owners = distribution_owners() if owners is None else owners
    notices = NOTICES_SOURCE.read_text(encoding="utf-8") if notices is None else notices
    python_root = Path(sys.base_prefix)
    found: dict[str, importlib.metadata.Distribution] = {}
    problems: list[str] = []
    for source in sources:
        distribution = owners.get(_key(source))
        if distribution is not None:
            found.setdefault(distribution.metadata["Name"].lower(), distribution)
        elif not any(_inside(source, root) for root in (*own_roots, python_root)):
            # 開発機の PATH から拾った DLL などがここに来る どの使用許諾で配るのか
            # 決められない物は配らない
            problems.append(f"出どころの分からないファイルを積んでいる: {source}")
    for name in ALWAYS_BUNDLED:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            found.setdefault(name, importlib.metadata.distribution(name))

    listed = listed_versions(notices)
    target = bundle / LICENSES_DIR
    # 前の組み立ての写し（上げる前の版のフォルダ）を残すと、今は積んでいない版の
    # 写しまで zip に入る
    shutil.rmtree(target, ignore_errors=True)
    for key, distribution in sorted(found.items()):
        name = distribution.metadata["Name"]
        # 一覧に書いていない包みは、ソースの入手先も書いていない 版が違えば、書いてある
        # 使用許諾も違うかもしれない どちらも一覧を直してから配る
        version = listed.get(canonical_name(name))
        if version is None:
            problems.append(f"{name} が {NOTICES_SOURCE.name} の一覧に無い")
        elif version != distribution.version:
            problems.append(
                f"{name} は {distribution.version} を積んでいるが、"
                f"{NOTICES_SOURCE.name} の一覧は {version}"
            )
        files, missing = license_files(distribution)
        if not files and key not in WITHOUT_LICENSE_FILES:
            problems.append(f"{name} {distribution.version} の使用許諾の写しが見つからない")
        problems += [
            f"{name} {distribution.version} が書いている使用許諾の写しが無い: {relative}"
            for relative in missing
        ]
        folder = target / f"{name}-{distribution.version}"
        for relative, path in files:
            destination = folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)

    python_license = python_root / "LICENSE.txt"
    if python_license.exists():
        (target / "Python").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(python_license, target / "Python" / "LICENSE.txt")
    else:
        problems.append(f"Python 本体の使用許諾が無い: {python_license}")
    return problems


def missing_notices(home: Path) -> list[str]:
    """展開した zip に、使用許諾の一覧と全文がそろっているか

    GPL の部品を積んで配る以上、使用許諾の全文を一緒に渡さなければならない
    組み立ての途中の段を飛ばしても zip は作れてしまうので、配る物の側で確かめる
    """
    expected = [
        "LICENSE.txt",
        NOTICES_NAME,
        f"{LICENSES_DIR}/Python/LICENSE.txt",
        *(f"{LICENSES_DIR}/{name}" for name in repository_license_files()),
    ]
    return [name for name in expected if not (home / name).is_file()]


#: wheel の中の DLL と、その使用許諾の写しを置いたリポジトリの ``licenses`` のフォルダ
#: DLL の名前（版と、delvewheel が付ける印を外した物）で引く PyAV を上げて DLL が
#: 増えたり版が変わったりしたら、ここと写しと一覧を一緒に直す
NATIVE_LICENSES: dict[str, str] = {
    **dict.fromkeys(
        ("avcodec", "avdevice", "avfilter", "avformat", "avutil", "swresample", "swscale"),
        "ffmpeg-8.1.2",
    ),
    "libx264": "x264-b35605ac",
    "libx265": "x265-4.2",
    "libdav1d": "dav1d-1.5.3",
    "libmp3lame": "lame-3.100",
    "libopencore-amrnb": "opencore-amr-0.1.6",
    "libopencore-amrwb": "opencore-amr-0.1.6",
    "libopus": "opus-1.6.1",
    "libsvtav1enc": "SVT-AV1-4.1.0",
    "libvpx": "libvpx-1.16.0",
    "libwebp": "libwebp-1.6.0",
    "libwebpmux": "libwebp-1.6.0",
    "libsharpyuv": "libwebp-1.6.0",
    "libvpl": "libvpl-2.16.0",
    "libiconv": "libiconv-1.19",
    "zlib1": "zlib-1.3.2",
    "libgcc_s_seh": "gcc-16.1.0",
    "libstdc++": "gcc-16.1.0",
    "libwinpthread": "winpthreads-mingw-w64-14.0.0",
    # lupa の中の LuaJIT（Lua 5.x の分は lupa 自身の写しにある）
    "luajit20": "LuaJIT-2.0-e4c7d8b3",
    "luajit21": "LuaJIT-2.1-18b087cd",
}

#: 写しを見張る DLL の置き場（``_internal`` からの相対） どれも wheel が写しを持たない
NATIVE_FOLDERS = ("av.libs", "lupa")


def native_base_name(filename: str) -> str:
    """``libx264-165-f3a9....dll`` → ``libx264`` 版の番号と delvewheel の印を外す"""
    stem = filename.lower().split(".", 1)[0]
    stem = re.sub(r"-[0-9a-f]{32}$", "", stem)
    return re.sub(r"(-\d+)+$", "", stem)


def native_license_problems(internal: Path) -> list[str]:
    """wheel の中の DLL に、リポジトリに置いた使用許諾の写しが対応しているか

    DLL の使用許諾は wheel が持っていないので、dist-info から集めても入らない
    表に無い DLL が増えていたら止める 黙って通すと、写しの無い部品を配る
    """
    problems = []
    for folder in NATIVE_FOLDERS:
        for path in sorted((internal / folder).glob("*")):
            if path.suffix.lower() not in (".dll", ".pyd"):
                continue
            base = native_base_name(path.name)
            if folder == "lupa" and not base.startswith("luajit"):
                continue
            copy = NATIVE_LICENSES.get(base)
            if copy is None:
                problems.append(f"{folder}/{path.name} の使用許諾の写しが決まっていない")
            elif not (ROOT / LICENSES_DIR / copy).is_dir():
                problems.append(f"{folder}/{path.name} の写し licenses/{copy} が無い")
    return problems


#: 組み立ての記録に載らないが、zip に入れてよい物（:func:`assemble` と
#: :func:`collect_licenses` が置く物）
ASSEMBLED = ("README.txt", "LICENSE.txt", NOTICES_NAME)
#: 使用許諾ではないが :func:`assemble` が置く物（使用許諾の照合には混ぜない）
ASSEMBLED_INFO = (BUILD_INFO_NAME,)
ASSEMBLED_FOLDERS = (PORTABLE_SCRIPTS_DIR, LICENSES_DIR)


def untracked_files(bundle: Path, record: Path) -> list[str]:
    """組み立てたフォルダにあるのに、組み立ての記録に無いファイル

    ``--skip-build`` で前の組み立てを使うと、手で足した DLL や前の組み立ての残りが
    フォルダに混ざっていても zip に入る 記録に無い物は出どころを辿れず、使用許諾を
    そろえられない
    """
    collected = ast.literal_eval((record / "COLLECT-00.toc").read_text(encoding="utf-8"))[0]
    # 記録の置き先は exe だけがフォルダの直下で、ほかは ``_internal`` の下
    known = {
        PurePosixPath(PureWindowsPath(entry[0]).as_posix()).as_posix().lower()
        for entry in collected
    }
    found = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle)
        if (
            relative.parts[0] in ASSEMBLED_FOLDERS
            or relative.as_posix() in ASSEMBLED + ASSEMBLED_INFO
        ):
            continue
        inner = (
            PurePosixPath(*relative.parts[1:]).as_posix()
            if relative.parts[0] == "_internal"
            else relative.as_posix()
        )
        if inner.lower() not in known:
            found.append(relative.as_posix())
    return found


def notice_digests(bundle: Path) -> dict[str, str]:
    """zip に入れる使用許諾の一覧・写しと、その sha256（``Sashimono`` からの相対）

    zip から確かめる段で、全部そろっていて中身も同じかを見る 名前だけ見ると、
    途中で消えた写しや壊れた写しに気付けない
    """
    paths = [bundle / name for name in ASSEMBLED if (bundle / name).is_file()]
    paths += [path for path in (bundle / LICENSES_DIR).rglob("*") if path.is_file()]
    return {
        path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def changed_notices(home: Path, expected: Mapping[str, str]) -> list[str]:
    """展開した zip の使用許諾が、組み立てたときの物と違う・無い物"""
    changed = []
    for relative, digest in expected.items():
        path = home / relative
        if not path.is_file():
            changed.append(f"無い: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(f"中身が違う: {relative}")
    return changed


def make_zip(bundle: Path, target: Path) -> Path:
    """フォルダごと zip にする 展開すると ``Sashimono\\`` が 1 つできる形

    中身をばらで入れると、展開した場所に部品が散らばる
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # 前の組み立ての zip を先に消す 残したまま今回が途中で落ちると、
    # 前の物が今回の版の名前で残り、新しい物と取り違えて配ることになる
    # （消しても組み立て直せば戻る） 開いたままで消せなければ、ここで止まる
    target.unlink(missing_ok=True)
    temporary = target.with_name(target.name + ".writing")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(APP_NAME) / path.relative_to(bundle))
        # 途中で止まった zip を完成品と取り違えないよう、書き終えてから名前を付ける
        # 名前を付ける所も後始末の中に入れる 前の zip を開いたままだと Windows は
        # 置き換えを断り、書き終えた 100 MB がそのまま残る
        temporary.replace(target)
    except BaseException:
        # 書きかけも残さない 名前が違うので完成品とは取り違えないが、
        # 100 MB ずつ溜まるうえ、手で配るときに紛れる
        temporary.unlink(missing_ok=True)
        raise
    return target


def minimal_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """確かめるときの環境変数 Windows が動くのに要る分だけ残す

    開発機の ``PATH`` には Python や FFmpeg が載っている 残したまま確かめると、
    zip に積み忘れた DLL をそちらから拾って通ってしまう
    """
    keep = (
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "USERNAME",
        "COMPUTERNAME",
        # 素の Windows の利用者なら誰でも持っている変数 中身は Windows の物で開発の道具を
        # 指さない 外すと PowerShell 5.1 の起動が遅くなり（CI で 22 秒が 34 秒）、
        # 使う人の機械では起きない遅さで入れ替え係の確かめが落ちる（Issue #33）
        "ALLUSERSPROFILE",
        "PUBLIC",
        "COMSPEC",
        "PATHEXT",
        "OS",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
    )
    upper = {key.upper(): value for key, value in environ.items()}
    minimal = {key: upper[key] for key in keep if key in upper}
    system_root = minimal.get("SYSTEMROOT", r"C:\Windows")
    minimal["PATH"] = os.pathsep.join(
        [
            str(Path(system_root) / "System32"),
            system_root,
            str(Path(system_root) / "System32" / "Wbem"),
        ]
    )
    return minimal


#: zip から確かめるときに置き場へ置く見本 読まれた本数で確かめる
SAMPLE_SCRIPT_NAME = "確かめる用.anm2"
SAMPLE_SCRIPT = """--track@amount:量,0,100,50
obj.ox = amount
"""

#: pip で入れてみる小さな包み ネットにつながずに入れられるよう、ここで作る
SAMPLE_PACKAGE = "sashimono_check_sample"


def write_sample_wheel(folder: Path) -> Path:
    """中身が 1 行だけの wheel を作る

    導入ボタンと同じ ``--target`` の入れ方を、ネットにつながずに通すため
    PyPI から落とすと、確かめるたびに外へ取りに行くことになる
    """
    name = f"{SAMPLE_PACKAGE}-0.1-py3-none-any.whl"
    info = f"{SAMPLE_PACKAGE}-0.1.dist-info"
    files = {
        f"{SAMPLE_PACKAGE}/__init__.py": "VALUE = 1\n",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {SAMPLE_PACKAGE}\nVersion: 0.1\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: sashimono\nRoot-Is-Purelib: true\n"
        "Tag: py3-none-any\n",
    }
    record = "".join(f"{path},,\n" for path in files) + f"{info}/RECORD,,\n"
    target = folder / name
    with zipfile.ZipFile(target, "w") as wheel:
        for path, text in files.items():
            wheel.writestr(path, text)
        wheel.writestr(f"{info}/RECORD", record)
    return target


def _run(executable: Path, arguments: list[str], folder: str) -> subprocess.CompletedProcess[str]:
    """展開した exe を走らせる

    渡すのは、この道具が展開した exe の場所と、この道具の中で決めた引数だけ
    外から来た文字列は混ざらず、shell も通さない（引数は並びのまま渡す）
    """
    # 監査済み 引数の出どころは上のとおりで、外から来た文字列は混ざらない
    return subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        [str(executable), *arguments],
        cwd=folder,
        env=minimal_environment(os.environ),
        capture_output=True,
        text=True,
        # exe はこの機械の既定の文字コード（日本語の Windows なら cp932）で書く
        # 配布版は ``PYTHONIOENCODING`` を見ない（PyInstaller が環境変数から
        # 切り離している 実際に渡して確かめた）ので、読む側を合わせる
        # 使う人がコマンドで結果を見るときも、この既定の文字コードで正しく出る
        # getpreferredencoding は使わない この道具を UTF-8 モード（PYTHONUTF8=1）で
        # 動かすと UTF-8 を返し、exe の書いた cp932 を読み違える
        encoding=locale.getencoding(),
        errors="replace",
        timeout=600,
        check=False,
    )


def smoke_test(archive: Path, notices: Mapping[str, str]) -> int:
    """zip を別の場所へ展開し、中の exe で確かめる

    ``notices`` は組み立てたときの使用許諾の一覧と写しの sha256（:func:`notice_digests`）

    1. 自己診断（描く・書き出す・Lua・pip の有無）
    2. exe の隣の置き場へ見本を置き、**読まれた**こと
    3. exe に pip を走らせ、導入ボタンと同じ入れ方で**実際に入る**こと
    """
    with tempfile.TemporaryDirectory(prefix="sashimono-zip-check-") as folder:
        with zipfile.ZipFile(archive) as opened:
            opened.extractall(folder)
        home = Path(folder) / APP_NAME
        executable = home / f"{APP_NAME}.exe"
        print(f"展開した先で確かめる: {executable}")

        (home / PORTABLE_SCRIPTS_DIR / SAMPLE_SCRIPT_NAME).write_text(
            SAMPLE_SCRIPT, encoding="utf-8"
        )
        checked = _run(executable, [SELF_CHECK_FLAG], folder)
        print(checked.stdout.rstrip())
        if checked.stderr.strip():
            print(checked.stderr.rstrip())
        failures = [] if checked.returncode == 0 else ["自己診断"]
        failures += [f"使用許諾が入っていない: {name}" for name in missing_notices(home)]
        failures += [f"使用許諾の写しが{change}" for change in changed_notices(home, notices)]

        beside = f"{home / PORTABLE_SCRIPTS_DIR}（1 本）"
        if beside not in checked.stdout:
            failures.append("exe の隣の置き場に置いたスクリプトが読まれていない")
        failures += update_failures(home, checked.stdout)

        failures += runtime_import_failures(executable, folder)

        wheel = write_sample_wheel(Path(folder))
        target = Path(folder) / "runtime"
        installed = _run(
            executable,
            ["-m", "pip", "install", "--no-index", "--target", str(target), str(wheel)],
            folder,
        )
        # 終了コードも見る ファイルを置いたあとで落ちた pip を「入った」と数えない
        if installed.returncode == 0 and (target / SAMPLE_PACKAGE / "__init__.py").exists():
            print(f"[ok] exe の pip で入れられた: {target}")
        else:
            print(installed.stdout.rstrip())
            print(installed.stderr.rstrip())
            failures.append("exe の pip で入れられない（導入ボタンが動かない）")

        for failure in failures:
            print(f"[NG] {failure}")
        return 1 if failures else 0


def update_failures(home: Path, self_check_output: str) -> list[str]:
    """展開した zip で、自動更新の部品が動くか ネットワークへは出ない

    - 自己診断の「自動更新」の項目が通った（使い捨ての鍵と見本のリリースで、署名・照合・
      展開・PowerShell の入れ替えまでを配布版の exe の中で通す）
    - zip に版と Python の書き付けが入っていて、この版と同じ 無いと、配った版がこの zip を
      新しい版として受け取れない（展開した物を確かめる所で止まる）
    """
    failures = []
    if f"[ok] {UPDATE_CHECK_NAME}:" not in self_check_output:
        failures.append("自動更新の部品が配布版の中で動かない（自己診断の項目が通らない）")
    try:
        written = json.loads((home / BUILD_INFO_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        written = None
    expected = {"version": __version__, "python_abi": python_abi()}
    if written != expected:
        failures.append(f"{BUILD_INFO_NAME} が無いか違う（{written} / {expected}）")
    return failures


def runtime_import_failures(executable: Path, folder: str) -> list[str]:
    """後から入れる部品を、配布版の exe の中で import してみる この機械に入っている物だけ

    部品は開発の環境の置き場から読ませる 標準ライブラリは exe の持ち物しか見えないので、
    配布版に無い物があればここで落ちる（利用者の手元で送った途端に落ちるのと同じ）
    置き場は前へ足す 配布版は入れた部品の置き場を前へ足して読む（:func:`activate_runtime`）
    後ろへ足すと、配布版が一部だけ積んでいる包み（cryptography）が先に読まれ、使う人の
    手元では起きない失敗になる
    """
    found = {
        name: spec
        for name in RUNTIME_MODULES
        if (spec := importlib.util.find_spec(name)) is not None and spec.origin is not None
    }
    if not found:
        print("[--] 後から入れる部品がこの機械に無いので、exe の中での import は確かめない")
        return []
    places = sorted({str(Path(str(spec.origin)).parent.parent) for spec in found.values()})
    result = _run(executable, [IMPORT_CHECK_FLAG, os.pathsep.join(places), *found], folder)
    print(result.stdout.rstrip())
    if result.returncode == 0:
        return []
    if result.stderr.strip():
        print(result.stderr.rstrip())
    return ["後から入れる部品を exe の中で import できない（標準ライブラリが足りない など）"]


def package(bundle: Path, target: Path, *, check: bool = True) -> int:
    """組み立てたフォルダを zip にし、zip から確かめる 戻り値は終了コード"""
    assemble(bundle)
    notices = notice_digests(bundle)
    archive = make_zip(bundle, target)
    print(f"できた: {archive}（{_size(archive)} 展開すると {_size(bundle)}）")
    if not check:
        return 0
    try:
        result = smoke_test(archive, notices)
    except BaseException:
        # 確かめる途中で落ちた（展開できない・exe が返ってこない）ときも同じ
        # 確かめ終えていない zip を完成品の名前で残さない
        archive.unlink(missing_ok=True)
        raise
    if result != 0:
        # 確かめて落ちた zip は残さない **dist に zip がある＝確かめ済み** に
        # そろえる 残すと、動かない物を完成品と取り違えて配る 中身を調べたい
        # ときは、組み立てたフォルダがそのまま残っている
        archive.unlink(missing_ok=True)
        print(f"確かめて落ちたので {archive.name} を消した（{bundle} は残してある）")
    return result


def _size(path: Path) -> str:
    total = (
        path.stat().st_size
        if path.is_file()
        else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    )
    return f"{total / (1024 * 1024):.0f} MB"


def main(argv: list[str] | None = None, *, dist: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="組み立て済みを使う")
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="zip からの確認を省く（できた zip は確かめていない物になる）",
    )
    parser.add_argument(
        "--write-constraints",
        type=Path,
        metavar="PATH",
        help="積む包みを一覧の版に留める制約を書いて終わる（組み立てない CI で使う）",
    )
    args = parser.parse_args(argv)

    if args.write_constraints is not None:
        lines = constraints(NOTICES_SOURCE.read_text(encoding="utf-8"))
        args.write_constraints.parent.mkdir(parents=True, exist_ok=True)
        args.write_constraints.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0

    work = ROOT / "build" / "pyinstaller"
    dist = dist if dist is not None else ROOT / "dist"
    bundle = dist / APP_NAME
    target = dist / f"{ARCHIVE_PREFIX}-{__version__}-windows-x64.zip"

    # **組み立てる前に**前の zip を消す 組み立て（PyInstaller）で落ちると
    # zip を作る所まで進まないので、そこで消していては前の物が残り、
    # 今回の完成品に見えてしまう
    target.unlink(missing_ok=True)

    if not args.skip_build:
        import PyInstaller.__main__

        with _without_developer_path():
            PyInstaller.__main__.run(pyinstaller_arguments(work, dist))
    if not (bundle / f"{APP_NAME}.exe").exists():
        print(f"{bundle} に {APP_NAME}.exe が無い 組み立てに失敗している")
        return 1

    drop_unused_qt(bundle)
    dangling = dangling_imports(bundle / "_internal", UNUSED_QT_PARTS)
    if dangling:
        # 外した DLL を読む物が残っていると、それを使った時点で落ちる 外した判断が誤り
        for user, name in dangling:
            print(f"[NG] 外した {name} を {user} が読む")
        return 1

    record = work / APP_NAME
    try:
        sources = bundled_sources(record)
        untracked = untracked_files(bundle, record)
    except FileNotFoundError:
        # 記録が無いと、何を積んだかを数えられず使用許諾をそろえられない
        print(f"{record} に組み立ての記録が無い --skip-build を外して組み立て直す")
        return 1
    problems = [f"組み立ての記録に無いファイルがある: {name}" for name in untracked]
    problems += [
        f"後から入れる部品が使う標準ライブラリが無い: {name}" for name in missing_stdlib(record)
    ]
    problems += native_license_problems(bundle / "_internal")
    problems += collect_licenses(bundle, sources, (ROOT / "src", record))
    if problems:
        for problem in problems:
            print(f"[NG] {problem}")
        return 1

    return package(bundle, target, check=not args.skip_check)


#: 積まない Qt の部品（``_internal`` からの相対の綴り）と、外してよいと言える理由
#: PyInstaller の PySide6 の差し込みはプラグインをまとめて積み、プラグインが読む DLL も
#: 付いてくる どの DLL を誰が読むかは、組み立てた zip の DLL の import 表で確かめた
#: 外した物を読む物が残っていないかは、組み立てのたびに :func:`dangling_imports` が見る
UNUSED_QT_PARTS: dict[str, str] = {
    "PySide6/plugins/imageformats/qpdf.dll": (
        "PDF を絵として読むプラグイン Sashimono は PDF を素材にしない"
        "（読み込める拡張子にも、読み込みの窓の絞り込みにも無い）"
    ),
    "PySide6/Qt6Pdf.dll": "qpdf.dll だけが読む",
    "PySide6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll": (
        "画面に出す仮想キーボード 環境変数 QT_IM_MODULE で選んだときだけ読まれ、"
        "Sashimono は選ばない（日本語の入力は Windows の IME が受ける）"
    ),
    "PySide6/Qt6VirtualKeyboard.dll": "仮想キーボードのプラグインだけが読む",
    "PySide6/Qt6Quick.dll": "仮想キーボードだけが読む（QML の画面部品）",
    "PySide6/Qt6Qml.dll": "仮想キーボードと Qt6Quick だけが読む",
    "PySide6/Qt6QmlMeta.dll": "Qt6Quick だけが読む",
    "PySide6/Qt6QmlModels.dll": "Qt6Quick と Qt6QmlMeta だけが読む",
    "PySide6/Qt6QmlWorkerScript.dll": "Qt6QmlMeta だけが読む",
    "PySide6/opengl32sw.dll": (
        "Qt が GPU の無い機械で使うソフトウェアの GL（Mesa llvmpipe 20 MB） Sashimono の"
        "描画には使えない 取れるのは OpenGL 3.0 までで、描画に要る 4.3 に届かない うえに、"
        "描く関数は PyOpenGL が Windows の opengl32.dll から引くので、Qt がこちらで作った"
        "コンテキストへは届かない（2026-10 に手元で AA_UseSoftwareOpenGL を立てて確かめた）"
        " GPU の無い機械では積んでいても描けず、プレビューの所に理由を出す（Issue #33）"
    ),
}


def drop_unused_qt(bundle: Path) -> list[str]:
    """使わない Qt の部品を組み立てたフォルダから外す 戻り値は外した物

    外さないと 18 MB ほど zip が膨らむうえ、LGPL の部品として対応するソースを
    添付しなければならない（Qt6Pdf の入っている qtwebengine のソースは 580 MB ある）
    """
    removed = []
    for relative in UNUSED_QT_PARTS:
        path = bundle / "_internal" / relative
        if path.exists():
            path.unlink()
            removed.append(relative)
    return removed


def _imported_dlls(path: Path) -> list[str]:
    """DLL / pyd が読む DLL の名前（小文字）"""
    import pefile

    image = pefile.PE(str(path), fast_load=True)
    try:
        image.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        )
        return [
            entry.dll.decode().lower() for entry in getattr(image, "DIRECTORY_ENTRY_IMPORT", [])
        ]
    finally:
        image.close()


def dangling_imports(
    internal: Path,
    removed: Iterable[str],
    *,
    reader: Callable[[Path], list[str]] = _imported_dlls,
) -> list[tuple[str, str]]:
    """外した DLL を、まだ残っている DLL / pyd が読んでいないか（読む物、読まれる DLL）"""
    names = {Path(relative).name.lower() for relative in removed}
    found = []
    for path in sorted([*internal.rglob("*.dll"), *internal.rglob("*.pyd")]):
        for name in reader(path):
            if name in names:
                found.append((path.relative_to(internal).as_posix(), name))
    return found


@contextlib.contextmanager
def _without_developer_path() -> Iterator[None]:
    """組み立てる間だけ ``PATH`` を Windows の分にする

    PyInstaller は Qt の通信部品のために OpenSSL の DLL を ``PATH`` から探す
    開発機では Git for Windows の物（``C:\\Program Files\\Git\\mingw64\\bin``）が
    見つかって積まれていた Sashimono は Qt で通信しないので要らないうえ、出どころが
    組み立てる機械で変わる物は使用許諾をそろえられない
    """
    before = os.environ.get("PATH")
    os.environ["PATH"] = minimal_environment(os.environ)["PATH"]
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = before


if __name__ == "__main__":
    raise SystemExit(main())
