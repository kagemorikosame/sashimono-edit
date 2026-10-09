"""更新の道を、ネットワークへ出ずに端から端まで通す（自己診断が使う）

配布版では、暗号の部品（cryptography の DLL）・zip の展開・PowerShell の入れ替え係が、
開発機とは違う持ち物で動く どれかを積み忘れても起動はするので、更新を出した日に
初めて「受け取れない」と分かる それを配る前に見つけるため、その場で作った使い捨ての鍵と
見本のリリースで、本物と同じ道を通す（本物の公開鍵と GitHub には触らない）
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import ssl
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sashimono.links import STABLE_MANIFEST_URL
from sashimono.update.check import Outcome, check_for_update
from sashimono.update.fetch import MemoryTransport
from sashimono.update.manifest import Manifest, PackageInfo, manifest_to_json
from sashimono.update.package import APP_EXE, BUILD_INFO_NAME, Layout, stage
from sashimono.update.signing import SIGNATURE_SUFFIX, sign_manifest
from sashimono.update.swap import (
    BUNDLED_SCRIPT,
    HELPER_SCRIPT,
    SWAP_CONTRACT,
    SwapPlan,
    launch,
    wait_started,
)

__all__ = ["REHEARSAL_VERSION", "build_release", "rehearse"]

#: 見本のリリースの版 本物の版と取り違えない大きな数
REHEARSAL_VERSION = "9999.0.0"

#: 見本の zip の置き場（GitHub の形をまねる 決めたホストの中なので転送を追える）
_ZIP_URL = "https://github.com/sashimono-check/release/releases/download/v9999.0.0/check.zip"
_CDN_URL = "https://release-assets.githubusercontent.com/sashimono-check/check.zip"


def build_release(
    key: Ed25519PrivateKey, *, version: str = REHEARSAL_VERSION, python_abi: str = ""
) -> tuple[MemoryTransport, Manifest]:
    """見本のリリース（目録・署名・zip）を、転送つきで返す取り口"""
    abi = python_abi or f"cp{sys.version_info.major}{sys.version_info.minor}"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(f"Sashimono/{APP_EXE}", b"MZ rehearsal")
        # 本物の zip と同じく、入れ替え係の台本と受け渡しの版を持たせる 入れ替えの予行で、
        # 新しい版の台本を使う道（swap.helper_script）も通す
        archive.writestr(f"Sashimono/{BUNDLED_SCRIPT}", HELPER_SCRIPT.encode("utf-8-sig"))
        archive.writestr(
            f"Sashimono/{BUILD_INFO_NAME}",
            json.dumps(
                {
                    "version": version,
                    "python_abi": abi,
                    "swap_contract": SWAP_CONTRACT,
                    "swap_script": BUNDLED_SCRIPT,
                }
            ),
        )
    body = payload.getvalue()

    manifest = Manifest(
        schema=1,
        version=version,
        minimum="0.0.0",
        python_abi=abi,
        notes_url="https://github.com/sashimono-check/release/releases/tag/v9999.0.0",
        package=PackageInfo(_ZIP_URL, len(body), hashlib.sha256(body).hexdigest()),
    )
    data = manifest_to_json(manifest)
    transport = MemoryTransport()
    # 本物と同じく、目録の固定の URL は別の所へ転送される
    tagged = "https://github.com/sashimono-check/release/releases/download/v9999.0.0/update.json"
    transport.redirects[STABLE_MANIFEST_URL] = tagged
    transport.redirects[STABLE_MANIFEST_URL + SIGNATURE_SUFFIX] = tagged + SIGNATURE_SUFFIX
    transport.pages[tagged] = data
    transport.pages[tagged + SIGNATURE_SUFFIX] = sign_manifest(data, key)
    transport.redirects[_ZIP_URL] = _CDN_URL
    transport.pages[_CDN_URL] = body
    return transport, manifest


def rehearse(folder: Path, *, swap: bool = sys.platform == "win32") -> str:
    """見本のリリースで、確かめる → 落とす → 展開する →（Windows では）入れ替える を通す

    通らなければ例外 通れば 1 行の説明
    """
    context = ssl.create_default_context()
    authorities = context.cert_store_stats().get("x509_ca", 0)
    if authorities == 0:
        # 証明書を 1 枚も読めないと、本物の GitHub へ HTTPS で繋げない
        raise RuntimeError("HTTPS の証明書を読めない（Windows の証明書の置き場が見えない）")

    key = Ed25519PrivateKey.generate()
    transport, manifest = build_release(key)
    result = check_for_update("0.0.1", transport=transport, keys=(key.public_key(),))
    if result.outcome is not Outcome.AVAILABLE:
        raise RuntimeError(f"見本の目録を受け取れない（{result.outcome.value} {result.reason}）")
    tampered = MemoryTransport()
    tampered.pages = dict(transport.pages)
    tampered.redirects = dict(transport.redirects)
    for url, page in transport.pages.items():
        if url.endswith("update.json"):
            tampered.pages[url] = page.replace(b"9999.0.0", b"9999.0.1")
    if check_for_update("0.0.1", transport=tampered, keys=(key.public_key(),)).outcome is not (
        Outcome.FAILED
    ):
        raise RuntimeError("書き換えた目録の署名が通ってしまう")

    install = folder / "Sashimono"
    install.mkdir(parents=True)
    (install / APP_EXE).write_bytes(b"MZ current")
    layout = Layout(install)
    stage(manifest, transport, layout)
    if layout.staged_version() != manifest.version:
        raise RuntimeError("展開した版が目録と違う")
    if not swap:
        return f"署名・照合・展開を確かめた（証明書 {authorities} 枚）"

    plan = SwapPlan("apply", layout, relaunch=False, check_start=False, wait_seconds=10)
    # 本体の作業場所をインストール先にして起こす Explorer やショートカットから起こした本体は
    # こうなる 入れ替え係がそれを受け継ぐと、自分で改名を断らせて入れ替えられない（#279）
    # 予行では本体（このプロセス）が終わらないので、起こした直後に作業場所を戻す 戻さないと
    # このプロセスが改名を断らせる 入れ替え係が改名に掛かるのは PowerShell が立ち上がって
    # 錠を取った後なので、戻すのはそれより十分に早い
    here = Path.cwd()
    os.chdir(install)
    try:
        launched = launch(plan, folder / "update")
    finally:
        os.chdir(here)
    began = time.monotonic()
    if not wait_started(launched):
        # 待ちきれなかったのか、すぐ終わったのかで原因が違う（遅い機械か、PowerShell が
        # 台本を読めない・走らせてもらえないか） 結果のファイルに書けた所まで添える
        # 理由が無いと、使う人の機械で落ちたときに貼ってもらっても直す所が分からない
        elapsed = time.monotonic() - began
        # wait_started は見切った入れ替え係を kill するが、終わるのは待たない Windows の
        # kill は終わらせる指示を出すだけなので、すぐ読むと終了コードが None になる
        # 待ってから読む（待つのはここだけ 本体の更新の道は終わりを待たずに知らせる）
        with contextlib.suppress(subprocess.TimeoutExpired):
            launched.process.wait(timeout=10)
        written = " ".join(launched.lines()) or "結果のファイルに何も無い"
        raise RuntimeError(
            f"入れ替え係（PowerShell）が走らない（{elapsed:.0f} 秒 終了コード "
            f"{launched.process.poll()} {written}）"
        )
    launched.process.wait(timeout=120)
    swapped = (install / BUILD_INFO_NAME).is_file() and (layout.previous / APP_EXE).is_file()
    if not swapped:
        lines = launched.result.read_text(encoding="utf-8-sig", errors="replace").split()
        raise RuntimeError(f"入れ替えられない（{' '.join(lines)}）")
    if not launched.from_staged:
        # 入れ替えられても、新しい版の台本を使わない版は、台本の直しを 1 つ前の版からの更新に
        # 効かせられない
        raise RuntimeError("新しい版の入れ替え係の台本を使わなかった（書き付けか台本が読めない）")
    return (
        "署名・照合・展開・入れ替え（PowerShell）を確かめた"
        f"（本体の作業場所はインストール先 新しい版の台本 証明書 {authorities} 枚）"
    )
