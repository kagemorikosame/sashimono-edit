"""後から入れる部品の、PyPI にある新しい版を尋ねる

尋ねるのは、利用者が導入の欄の〔更新を確かめる〕を押したときと、AI の部品の自動の更新
（設定で切れる 1 日に 1 回まで）のときだけ どちらも docs/code-signing-policy.md の
通信の表に書いてある

選ぶのは、この機械で pip が実際に入れられる版 claude-agent-sdk は Windows の wheel を
出さない版（0.2.157・0.2.160 から 0.2.163）があり、配布版の pip は wheel だけを選ぶ
（:func:`sashimono.runtime.install_arguments`） 一覧の一番新しい版を見せると、
〔環境を更新〕を押しても入らない版を「更新があります」と言ってしまう
"""

from __future__ import annotations

import http.client
import json
import sys
import sysconfig
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import IO, Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

__all__ = [
    "PYPI_JSON",
    "Latest",
    "Opener",
    "latest_release",
    "newest_installable",
    "platform_tag",
]

#: 版の一覧を返す PyPI の口 ``{name}`` に配布名が入る
PYPI_JSON = "https://pypi.org/pypi/{name}/json"

#: 待つ長さ（秒） 繋がらないときは早めに諦める 自動の更新は裏で待つだけだが、
#: 手で押したときに長く待たせると、押せたのかどうか分からない
TIMEOUT_SECONDS = 10.0

#: URL の頼みを開く物 試験で差し替える（本物の PyPI へ出ないように）
Opener = Callable[[urllib.request.Request], IO[bytes]]


@dataclass(frozen=True, slots=True)
class Latest:
    """尋ねた答え"""

    #: 指定の範囲（``>=0.2.158,<0.3`` など）の中で、入れられる一番新しい版
    allowed: str | None
    #: 範囲を問わず、入れられる一番新しい版 範囲の外の大きな版上げを知らせるのに使う
    newest: str | None


def _open(request: urllib.request.Request) -> IO[bytes]:
    response: IO[bytes] = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
    return response


def latest_release(requirement: str, *, opener: Opener | None = None) -> Latest | None:
    """pip の指定（``claude-agent-sdk>=0.2.158,<0.3`` など）の部品の新しい版
    尋ねられなければ ``None``
    """
    from sashimono import __version__

    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return None
    request = urllib.request.Request(
        PYPI_JSON.format(name=parsed.name),
        headers={"User-Agent": f"SashimonoEdit/{__version__} (component-check)"},
    )
    try:
        with (opener or _open)(request) as response:
            data = json.load(response)
    except (OSError, ValueError, http.client.HTTPException):
        # 繋がらない・返事が壊れている・途中で切れた（IncompleteRead など HTTPException は
        # OSError ではない） どれも「確かめられなかった」と扱うだけにする
        return None
    tag = platform_tag()
    return Latest(
        allowed=newest_installable(data, tag, parsed.specifier),
        newest=newest_installable(data, tag),
    )


def platform_tag() -> str:
    """wheel の名前に入る、この機械の印（``win_amd64``） Windows の外では空

    Windows の外（試験の CI など）では wheel の印の付け方が多く（manylinux など）、
    合っているかを見分ける手間に見合わないので、どの wheel でも入れられると見る
    """
    if sys.platform != "win32":
        return ""
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def newest_installable(
    data: object, tag: str = "", specifier: SpecifierSet | None = None
) -> str | None:
    """PyPI の返事から、wheel で入れられる一番新しい正式な版を選ぶ

    ``tag`` が空でなければ、その印の wheel か、どの機械でも入る wheel（``none-any``）が
    ある版だけを見る ``specifier`` を渡せば、その範囲の中だけを見る 前触れの版
    （``rc`` など）と、取り下げられた（yanked）物は選ばない pip も既定では選ばないので、
    選ばない物を見せると入らない版を勧めることになる
    """
    if not isinstance(data, Mapping):
        return None
    releases = data.get("releases")
    if not isinstance(releases, Mapping):
        return None
    best: Version | None = None
    for text, files in releases.items():
        try:
            version = Version(str(text))
        except InvalidVersion:
            continue
        if version.is_prerelease or not isinstance(files, list):
            continue
        if specifier is not None and version not in specifier:
            continue
        if not any(_installable(entry, tag) for entry in files):
            continue
        if best is None or version > best:
            best = version
    return None if best is None else str(best)


def _installable(entry: Any, tag: str) -> bool:
    if not isinstance(entry, Mapping) or entry.get("yanked"):
        return False
    if entry.get("packagetype") != "bdist_wheel":
        return False
    filename = str(entry.get("filename", ""))
    if not tag or filename.endswith("-none-any.whl"):
        return True
    return filename.endswith(f"-{tag}.whl")
