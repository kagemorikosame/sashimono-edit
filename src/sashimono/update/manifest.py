"""更新の目録（``update.json``）

**配った版が読み続ける契約** キーを消したり意味を変えたりしない 足すのは自由
（知らないキーは読み飛ばす） 形を大きく変えるときは ``minimum`` を上げ、古い版には
手で入れ直すよう案内させる

.. code-block:: json

    {
      "schema": 1,
      "version": "1.2.3",
      "minimum": "1.0.0",
      "python_abi": "cp314",
      "notes_url": "https://github.com/<owner>/<repo>/releases/tag/v1.2.3",
      "package": {
        "url": "https://github.com/<owner>/<repo>/releases/download/v1.2.3/SashimonoEdit-1.2.3-windows-x64.zip",
        "size": 258000000,
        "sha256": "…"
      }
    }

ここで読むのは**署名を確かめた後の**中身だけ 確かめる前の中身で何かを決めない
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from packaging.version import InvalidVersion, Version

from sashimono.update.fetch import FetchError, check_url

__all__ = [
    "MAX_MANIFEST_BYTES",
    "MAX_PACKAGE_BYTES",
    "SCHEMA",
    "Manifest",
    "ManifestError",
    "PackageInfo",
    "is_newer",
    "manifest_to_json",
    "parse_manifest",
    "parse_version",
]

#: この版が書く目録の形の版
SCHEMA = 1

#: 目録の大きさの上限 中身は数百バイト 上限を置かないと、すり替わった返事を丸ごと抱える
MAX_MANIFEST_BYTES = 64 * 1024

#: zip の大きさの上限 今は 100 MB ほど 字幕起こしを積んでも 2 GB は超えない
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ABI = re.compile(r"cp3\d{1,2}")


class ManifestError(ValueError):
    """目録として読めない 署名が通っても、読めない物で何かを決めない"""


@dataclass(frozen=True, slots=True)
class PackageInfo:
    """落とす zip"""

    url: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """読んだ目録"""

    schema: int
    version: str
    minimum: str
    python_abi: str
    notes_url: str
    package: PackageInfo


def parse_version(text: str) -> Version:
    """版の決まり（PEP 440）で読む 数字を拾って比べると ``1.2.0b1`` と ``1.2.0`` を取り違える"""
    try:
        return Version(text)
    except InvalidVersion as exc:
        raise ManifestError(f"版として読めない: {text!r}") from exc


def is_newer(candidate: str, current: str) -> bool:
    """``candidate`` が ``current`` より新しいか 読めない版は新しくないとする"""
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return False


def _text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} が無いか、文字でない")
    return value


def _https(url: str, key: str, *, fixed_host: bool) -> str:
    """URL を確かめる zip は決めたホストからしか落とさない 案内のページは HTTPS だけを見る"""
    if fixed_host:
        try:
            check_url(url)
        except FetchError as exc:
            raise ManifestError(f"{key}: {exc}") from exc
    elif not url.startswith("https://"):
        raise ManifestError(f"{key} が HTTPS でない: {url}")
    return url


def parse_manifest(data: bytes) -> Manifest:
    """目録を読む 1 つでも欠けた・型の違う値があれば :class:`ManifestError`

    ``schema`` は 1 以上なら読む 形を上げても、キーを消したり意味を変えたりは
    しない約束なので、1 の形のキーはそのまま読める
    """
    if len(data) > MAX_MANIFEST_BYTES:
        raise ManifestError("目録が大きすぎる")
    try:
        root = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestError(f"JSON として読めない: {exc}") from exc
    if not isinstance(root, dict):
        raise ManifestError("目録の一番上が表でない")

    schema = root.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
        raise ManifestError(f"schema が読めない: {schema!r}")
    version = _text(root, "version")
    minimum = _text(root, "minimum")
    parse_version(version)
    parse_version(minimum)
    abi = _text(root, "python_abi")
    if _ABI.fullmatch(abi) is None:
        raise ManifestError(f"python_abi が読めない: {abi!r}")
    notes_url = _https(_text(root, "notes_url"), "notes_url", fixed_host=False)

    package = root.get("package")
    if not isinstance(package, dict):
        raise ManifestError("package が無い")
    url = _https(_text(package, "url"), "package.url", fixed_host=True)
    size = package.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_PACKAGE_BYTES:
        raise ManifestError(f"package.size が読めない: {size!r}")
    sha256 = _text(package, "sha256").lower()
    if _SHA256.fullmatch(sha256) is None:
        raise ManifestError("package.sha256 が 64 桁の 16 進でない")
    return Manifest(schema, version, minimum, abi, notes_url, PackageInfo(url, size, sha256))


def manifest_to_json(manifest: Manifest) -> bytes:
    """目録を書く（署名する道具が使う） 読み戻して同じ物になる形"""
    data = {
        "schema": manifest.schema,
        "version": manifest.version,
        "minimum": manifest.minimum,
        "python_abi": manifest.python_abi,
        "notes_url": manifest.notes_url,
        "package": {
            "url": manifest.package.url,
            "size": manifest.package.size,
            "sha256": manifest.package.sha256,
        },
    }
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
