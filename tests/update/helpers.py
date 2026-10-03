"""自動更新の試験で使う、使い捨ての鍵と見本のリリース

本物の公開鍵と GitHub には触らない 鍵は試験の中で作って捨てる
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sashimono.links import BETA_MANIFEST_URL, STABLE_MANIFEST_URL
from sashimono.runtime import python_abi
from sashimono.update.fetch import MemoryTransport
from sashimono.update.manifest import Manifest, PackageInfo, manifest_to_json
from sashimono.update.package import APP_EXE, BUILD_INFO_NAME
from sashimono.update.signing import SIGNATURE_SUFFIX, sign_manifest

#: 落とす zip の置き場 GitHub と同じく、github.com から CDN へ転送される
ZIP_URL = "https://github.com/owner/repo/releases/download/v{version}/SashimonoEdit.zip"
CDN_URL = "https://release-assets.githubusercontent.com/assets/{version}.zip"


def package_bytes(
    version: str, *, abi: str | None = None, extra: dict[str, bytes] | None = None
) -> bytes:
    """配る zip の形（``Sashimono\\`` の下に exe と書き付け）"""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(f"Sashimono/{APP_EXE}", f"MZ {version}".encode())
        archive.writestr(
            f"Sashimono/{BUILD_INFO_NAME}",
            json.dumps({"version": version, "python_abi": abi or python_abi()}),
        )
        archive.writestr("Sashimono/scripts/README.txt", "同梱の説明")
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return payload.getvalue()


def manifest_for(version: str, body: bytes, *, minimum: str = "0.0.0") -> Manifest:
    return Manifest(
        schema=1,
        version=version,
        minimum=minimum,
        python_abi=python_abi(),
        notes_url=f"https://github.com/owner/repo/releases/tag/v{version}",
        package=PackageInfo(
            ZIP_URL.format(version=version), len(body), hashlib.sha256(body).hexdigest()
        ),
    )


def publish(
    transport: MemoryTransport,
    key: Ed25519PrivateKey,
    manifest: Manifest,
    body: bytes,
    *,
    url: str = STABLE_MANIFEST_URL,
    data: bytes | None = None,
) -> bytes:
    """目録・署名・zip を取り口に載せる 目録の固定の URL は、本物と同じく転送で届く"""
    data = data if data is not None else manifest_to_json(manifest)
    tagged = f"https://github.com/owner/repo/releases/download/v{manifest.version}/update.json"
    transport.redirects[url] = tagged
    transport.redirects[url + SIGNATURE_SUFFIX] = tagged + SIGNATURE_SUFFIX
    transport.pages[tagged] = data
    transport.pages[tagged + SIGNATURE_SUFFIX] = sign_manifest(data, key)
    transport.redirects[manifest.package.url] = CDN_URL.format(version=manifest.version)
    transport.pages[CDN_URL.format(version=manifest.version)] = body
    return data


def release(
    key: Ed25519PrivateKey,
    version: str,
    *,
    transport: MemoryTransport | None = None,
    beta: bool = False,
    minimum: str = "0.0.0",
) -> tuple[MemoryTransport, Manifest]:
    """署名した見本のリリースを 1 つ載せた取り口"""
    transport = transport if transport is not None else MemoryTransport()
    body = package_bytes(version)
    manifest = manifest_for(version, body, minimum=minimum)
    publish(transport, key, manifest, body, url=BETA_MANIFEST_URL if beta else STABLE_MANIFEST_URL)
    return transport, manifest


__all__ = [
    "CDN_URL",
    "ZIP_URL",
    "manifest_for",
    "package_bytes",
    "publish",
    "release",
]
