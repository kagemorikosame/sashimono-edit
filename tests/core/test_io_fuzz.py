"""壊れたプロジェクトファイルを開いても、ProjectFileError 以外で落ちないこと

配布されたプロジェクトや、途中で書き込みが止まったファイルは、どの値がどう壊れて
いるか分からない 素の例外が漏れると、開く側は ProjectFileError しか受けないので
起動ごと落ちる 往復テストの題材を、決まった種で何百通りにも壊して確かめる
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from sashimono.core.io import ProjectFileError, project_from_dict, project_to_dict
from sashimono.core.model import Project

#: 値を差し替えるときの候補 型違い、極端な数、壊れた文字、入れ子の取り違え
_BROKEN_VALUES: tuple[Any, ...] = (
    None,
    True,
    -1,
    0,
    2**63,
    -(2**63),
    1e308,
    float("inf"),
    float("nan"),
    "",
    "abc",
    "1/0",
    "-1/3",
    "９",
    [],
    [None],
    {},
    {"a": 1},
)

#: 1 つのファイルに加える壊し方の数 少ないと、検査を抜けた値が次の段で使われる経路を踏めない
_MUTATIONS_PER_FILE = 3
_FILES = 400


def _paths(node: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """辞書と配列の中の、差し替えられる場所をすべて並べる"""
    found: list[tuple[Any, ...]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append((*prefix, key))
            found.extend(_paths(value, (*prefix, key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.append((*prefix, index))
            found.extend(_paths(value, (*prefix, index)))
    return found


def _mutate(data: Any, rng: random.Random) -> None:
    paths = _paths(data)
    path = rng.choice(paths)
    parent = data
    for step in path[:-1]:
        parent = parent[step]
    key = path[-1]
    action = rng.random()
    if action < 0.15 and isinstance(parent, dict):
        del parent[key]
    elif action < 0.25 and isinstance(parent, list):
        parent.append(copy.deepcopy(parent[key]))
    else:
        parent[key] = copy.deepcopy(rng.choice(_BROKEN_VALUES))


@pytest.fixture
def source(rich_project: Project) -> dict[str, Any]:
    data = project_to_dict(rich_project)
    assert isinstance(data, dict)
    return data


def test_broken_files_fail_only_as_file_errors(source: dict[str, Any]) -> None:
    # 書き込みが途中で止まった作品や配布された作品を開くと、「開けない」と出ずにソフトごと落ちる
    rng = random.Random(20260917)
    leaks: list[str] = []
    for _ in range(_FILES):
        data = copy.deepcopy(source)
        for _ in range(_MUTATIONS_PER_FILE):
            _mutate(data, rng)
        try:
            project_from_dict(data)
        except ProjectFileError:
            continue
        except Exception as exc:
            leaks.append(f"{type(exc).__name__}: {exc}")
    assert leaks == [], f"{len(leaks)} 件漏れた 例: {sorted(set(leaks))[:5]}"


def test_non_finite_control_points_are_refused(source: dict[str, Any]) -> None:
    """ベジェの制御点の NaN や無限大を通すと、補間を経て描画へ非有限の値が流れる"""
    data = copy.deepcopy(source)
    found = False
    for track in data["timeline"]["tracks"]:
        for clip in track["clips"]:
            for effect in clip["effects"]:
                for value in effect["params"].values():
                    if isinstance(value, dict) and value.get("keyframes"):
                        value["keyframes"][0]["control_points"] = [0.1, float("nan"), 0.3, 0.4]
                        found = True
    assert found, "キーフレームを持つ値が試験のプロジェクトに無い"
    with pytest.raises(ProjectFileError, match="control_points"):
        project_from_dict(data)
