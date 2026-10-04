r"""配る zip に積む包みの一覧（THIRD_PARTY_NOTICES.md）を読む

    python tools\package_notices.py --write-constraints build\constraints.txt

**標準ライブラリだけで動かす** CI はこれで書いた制約を、依存を入れる最初の
``uv pip install`` から使う 依存を読む道具（build_package.py）で書くと、制約の無い
導入を 1 度先に通さなければならず、その日の新しい依存が壊れていると制約付きの導入まで
届かずに止まる（PR #241 のレビュー）
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 同梱した部品の一覧
NOTICES_SOURCE = ROOT / "THIRD_PARTY_NOTICES.md"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-constraints",
        type=Path,
        metavar="PATH",
        required=True,
        help="積む包みを一覧の版に留める制約を書く",
    )
    args = parser.parse_args(argv)
    lines = constraints(NOTICES_SOURCE.read_text(encoding="utf-8"))
    args.write_constraints.parent.mkdir(parents=True, exist_ok=True)
    args.write_constraints.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
