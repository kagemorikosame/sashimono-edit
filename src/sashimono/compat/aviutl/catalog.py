"""スクリプトのフォルダを走査して、使える形に並べる

AviUtl はスクリプトを拡張子で区別する

===========  ==========================================================
``.anm``     アニメーション効果 オブジェクトの動きを作る 一番多い
``.obj``     カスタムオブジェクト 中身そのものを作る
``.scn``     シーンチェンジ
``.cam``     カメラ効果
``.tra``     トラックバー変化（移動方法）
===========  ==========================================================

AviUtl2 世代では末尾に ``2`` が付く（``.anm2`` など） 読み方は同じなので、
拡張子から種類だけを取り出して同じように扱う

見つけたスクリプトは :class:`~sashimono.effects.EffectDefinition` として
エフェクトの一覧へ登録する **そうすると設定 UI もプリセットもキーフレームも
自前のエフェクトとまったく同じ経路に乗る** これが制御文字を
``ParameterSpec`` へ写しておいた狙い
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from sashimono.compat.aviutl.control import ScriptHeader, ScriptSection, split_scripts
from sashimono.compat.aviutl.encoding import read_text
from sashimono.compat.aviutl.report import CompatibilityReport, global_report
from sashimono.core import userdirs
from sashimono.effects.definition import EffectDefinition, registry
from sashimono.runtime import app_dir

__all__ = [
    "KIND_LABELS",
    "PORTABLE_SCRIPTS_DIR",
    "SCRIPT_SUFFIXES",
    "ScriptCatalog",
    "ScriptEntry",
    "default_script_roots",
    "script_catalog",
    "set_script_catalog",
]

#: 配った zip で、``Sashimono.exe`` の隣に置くスクリプト置き場の名前
#: パッケージを作る側（tools/build_package.py）も同じ名前で空のフォルダを作る
PORTABLE_SCRIPTS_DIR = "scripts"

#: 読み込む拡張子と、その種類
SCRIPT_SUFFIXES: dict[str, str] = {
    ".anm": "anm",
    ".obj": "obj",
    ".scn": "scn",
    ".cam": "cam",
    ".tra": "tra",
    ".anm2": "anm",
    ".obj2": "obj",
    ".scn2": "scn",
    ".cam2": "cam",
    ".tra2": "tra",
}

KIND_LABELS: dict[str, str] = {
    "anm": "アニメーション効果",
    "obj": "カスタムオブジェクト",
    "scn": "シーンチェンジ",
    "cam": "カメラ効果",
    "tra": "トラックバー変化",
}

#: エフェクト種別の接頭辞 プロジェクトファイルにそのまま出るので変えない
PREFIX = "aviutl:"


@dataclass(frozen=True, slots=True)
class ScriptEntry:
    """使えるスクリプト 1 つ"""

    #: エフェクト種別 ``aviutl:フォルダ/ファイル.anm:名前``
    identifier: str
    kind: str
    name: str
    path: Path
    source: str
    header: ScriptHeader
    #: スクリプトが置かれている実際のフォルダ 共通処理（.mod2）を探すのに使う
    folder: Path | None = None

    @property
    def label(self) -> str:
        return self.name or self.path.stem

    @property
    def category(self) -> str:
        return KIND_LABELS.get(self.kind, "AviUtl")

    def definition(self) -> EffectDefinition:
        """エフェクトの定義として見せる

        ``fragment_shader`` は持たない GPU のシェーダではなく Lua で動くので、
        描画側（:mod:`sashimono.engine.render`）が種別を見て振り分ける
        """
        return EffectDefinition(
            kind=self.identifier,
            label=self.label,
            category=self.category,
            parameters=self.header.parameters,
        )


def default_script_roots() -> tuple[Path, ...]:
    """既定で見に行くフォルダ

    アプリ自身のフォルダに加えて、AviUtl2 が入っていればその ``Script`` も見る
    すでに持っている資産を、わざわざコピーしなくても使えるようにするため

    配った zip では **``Sashimono.exe`` の隣の ``scripts``** も読む 同梱の見本の置き場で、
    前の版の案内どおりそこへ置いている人もいる ただし新しい版の zip を手で展開し直して
    フォルダごと入れ替えると中身が消えるので、置き場として案内するのは ``%APPDATA%`` の側
    （〔互換〕→〔スクリプトフォルダを開く〕が開く Issue #138） 自動更新は新しい版へ写し、
    起動したときに ``%APPDATA%`` の側へ移す（:mod:`sashimono.update.portable` Issue #244）

    **並びを変えない** 同じ名前は後に読んだ方が勝つ exe の隣をいちばん先（いちばん負ける）に
    置いているので、そこから ``%APPDATA%`` へ移しても、どれが使われるかは変わらない
    """
    roots: list[Path] = []
    beside = app_dir()
    if beside is not None:
        roots.append(beside / PORTABLE_SCRIPTS_DIR)
    # APPDATA の有無で分けない 置き場は userdirs が XDG まで見て決めるので、ここで
    # APPDATA だけを条件にすると、引き継ぎで写した先とここで探す先が食い違う
    roots.append(userdirs.config_root() / "scripts")

    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        roots.append(Path(program_data) / "aviutl2" / "Script")
    return tuple(roots)


class ScriptCatalog:
    """スクリプトを探して覚えておく"""

    def __init__(
        self,
        roots: tuple[Path, ...] | None = None,
        *,
        report: CompatibilityReport | None = None,
    ) -> None:
        self.roots = roots if roots is not None else default_script_roots()
        self._report = report if report is not None else global_report
        self._entries: dict[str, ScriptEntry] = {}

    def scan(self) -> tuple[ScriptEntry, ...]:
        """フォルダを走査して一覧を作り直す

        読めないファイルがあっても止まらない 1 つのスクリプトのせいで
        他の数百本が使えなくなる方が困る
        """
        self._entries.clear()
        for root in self.roots:
            if not root.exists():
                continue
            for path in sorted(_script_files(root)):
                for entry in self._read(root, path):
                    self._entries[entry.identifier] = entry
        return self.all()

    def register_all(self) -> int:
        """一覧をエフェクトの登録簿へ入れる 戻り値は登録した数

        すでにある同名の定義は差し替える スクリプトは編集されうるもので、
        走査し直したときに古い設定欄が残る方が困る
        """
        count = 0
        for entry in self._entries.values():
            registry.register(entry.definition(), replace=True)
            count += 1
        return count

    def all(self) -> tuple[ScriptEntry, ...]:
        return tuple(sorted(self._entries.values(), key=lambda e: (e.kind, e.label)))

    def of_kind(self, kind: str) -> tuple[ScriptEntry, ...]:
        return tuple(entry for entry in self.all() if entry.kind == kind)

    def get(self, identifier: str) -> ScriptEntry | None:
        return self._entries.get(identifier)

    def add_text(self, identifier: str, text: str, *, kind: str = "anm") -> ScriptEntry:
        """ファイルを介さずに 1 本足す 試験と、貼り付けからの取り込み用"""
        section = split_scripts(text)[0]
        entry = ScriptEntry(
            identifier=identifier,
            kind=kind,
            name=section.header.name or identifier,
            path=Path(identifier),
            source=section.source,
            header=section.header,
        )
        self._entries[identifier] = entry
        return entry

    def _read(self, root: Path, path: Path) -> Iterator[ScriptEntry]:
        kind = SCRIPT_SUFFIXES.get(path.suffix.lower())
        if kind is None:
            return
        try:
            text, _ = read_text(path)
        except OSError as exc:
            self._report.note_failure(path.name, f"開けない: {exc}")
            return

        relative = _relative(root, path)

        def broken(title: str, exc: ValueError) -> None:
            # 制御文字から設定欄を作れない節（範囲の崩れた値など）だけを飛ばす
            # ファイルごと飛ばすと、同じファイルの読めていた節まで使えなくなる
            where = f"{path.name}@{title}" if title else path.name
            self._report.note_failure(where, f"制御文字を読めない: {exc}")

        sections = split_scripts(text, on_error=broken)
        for index, section in enumerate(sections):
            yield _entry(relative, kind, index, section, path.parent)


def _entry(
    relative: str, kind: str, index: int, section: ScriptSection, folder: Path | None = None
) -> ScriptEntry:
    name = section.header.name
    identifier = f"{PREFIX}{relative}:{name or index}"
    for line in section.header.unknown:
        global_report.note_control(line)
    return ScriptEntry(
        identifier=identifier,
        kind=kind,
        name=name,
        path=Path(relative),
        source=section.source,
        header=section.header,
        folder=folder,
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - 走査の結果なので通常は起きない
        return path.name


def _script_files(root: Path) -> Iterator[Path]:
    for suffix in SCRIPT_SUFFIXES:
        yield from root.rglob(f"*{suffix}")


#: アプリ全体で 1 つ 描画側も UI も同じ一覧を見る
_catalog: ScriptCatalog | None = None


def script_catalog() -> ScriptCatalog:
    """共有のスクリプト一覧 初めて呼ばれたときに走査する"""
    global _catalog
    if _catalog is None:
        _catalog = ScriptCatalog()
        _catalog.scan()
        _catalog.register_all()
    return _catalog


def set_script_catalog(catalog: ScriptCatalog) -> None:
    """一覧を差し替える 試験と、フォルダを変えたときに使う"""
    global _catalog
    _catalog = catalog
    catalog.register_all()
