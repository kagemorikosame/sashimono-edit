"""新しい版があるかを確かめる

目録と署名を固定の URL から読み、埋め込んだ公開鍵で確かめてから中身を信じる
ベータを受け取る人は、正式版とベータの両方を読み、新しい方を採る（ベータの目録を
正式版の後に直し忘れても、正式版を受け取り損ねない）

ここは決めるだけで、落とさない 起動時の確認は裏で走り、失敗は黙る（呼ぶ側が決める）
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.version import Version

from sashimono.links import BETA_MANIFEST_URL, STABLE_MANIFEST_URL
from sashimono.update.fetch import FetchError, Transport, fetch_bytes
from sashimono.update.manifest import (
    MAX_MANIFEST_BYTES,
    Manifest,
    ManifestError,
    is_newer,
    parse_manifest,
)
from sashimono.update.signing import MAX_SIGNATURE_BYTES, SIGNATURE_SUFFIX, verify_manifest

__all__ = [
    "CHECK_INTERVAL_SECONDS",
    "CheckResult",
    "Outcome",
    "check_for_update",
    "is_due",
    "manifest_urls",
]

#: 起動時に確かめる間隔の下限（秒） 数時間に 1 回まで 起動のたびに取りに行くと、
#: 何度も開き直す人の分だけ GitHub へ行く
CHECK_INTERVAL_SECONDS = 6 * 60 * 60


class Outcome(enum.Enum):
    """確かめた結果"""

    #: 公開鍵が埋め込まれていない版 確かめない
    NOT_CONFIGURED = "not-configured"
    #: 今の版が最新
    UP_TO_DATE = "up-to-date"
    #: 入れられる新しい版がある
    AVAILABLE = "available"
    #: 新しい版はあるが、この版からは自動では入れられない（``minimum`` より古い）
    MANUAL = "manual"
    #: 取れない・署名が通らない・読めない
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CheckResult:
    outcome: Outcome
    manifest: Manifest | None = None
    #: 失敗したわけ 起動時の確認では見せない（手で確かめたときだけ出す）
    reason: str = ""


def manifest_urls(*, beta: bool) -> tuple[str, ...]:
    """読む目録 正式版を先に読む"""
    return (STABLE_MANIFEST_URL, BETA_MANIFEST_URL) if beta else (STABLE_MANIFEST_URL,)


def is_due(last_checked: float, now: float, interval: float = CHECK_INTERVAL_SECONDS) -> bool:
    """起動時に確かめに行ってよいか 時計が戻った（前の時刻が未来）ときも行く

    戻ったときに行かないと、時計を合わせ直すまで何日でも確かめなくなる
    """
    return last_checked > now or now - last_checked >= interval


def _read_one(url: str, transport: Transport, keys: Sequence[Ed25519PublicKey]) -> Manifest:
    """目録 1 つを読み、署名を確かめてから中身を読む"""
    data = fetch_bytes(url, transport, limit=MAX_MANIFEST_BYTES)
    signature = fetch_bytes(url + SIGNATURE_SUFFIX, transport, limit=MAX_SIGNATURE_BYTES)
    # 確かめる前に中身を読まない 読む所（JSON の解釈）そのものも攻め口になる
    if not verify_manifest(data, signature, keys):
        raise ManifestError("署名が通らない")
    return parse_manifest(data)


def check_for_update(
    current: str,
    *,
    transport: Transport,
    keys: Sequence[Ed25519PublicKey],
    beta: bool = False,
    skipped: Sequence[str] = (),
) -> CheckResult:
    """``current`` より新しい版があるか

    ``skipped`` の版は「新しい版は無い」と同じに扱う（本人が飛ばした・入れたら起動
    できずに戻した版） それより新しい版が出れば、また知らせる
    """
    if not keys:
        return CheckResult(Outcome.NOT_CONFIGURED)
    found: list[Manifest] = []
    reasons: list[str] = []
    for url in manifest_urls(beta=beta):
        try:
            found.append(_read_one(url, transport, keys))
        except (FetchError, ManifestError) as exc:
            reasons.append(f"{url}: {exc}")
    if not found:
        return CheckResult(Outcome.FAILED, reason=" / ".join(reasons))
    # 飛ばした版を除いてから新しい方を採る ベータの 1 本を飛ばしても、正式版の新しい版は受け取る
    offered = [m for m in found if m.version not in skipped and is_newer(m.version, current)]
    if not offered:
        return CheckResult(Outcome.UP_TO_DATE, max(found, key=lambda m: Version(m.version)))
    best = max(offered, key=lambda manifest: Version(manifest.version))
    if Version(current) < Version(best.minimum):
        return CheckResult(Outcome.MANUAL, best, f"この版（{current}）は {best.minimum} より古い")
    return CheckResult(Outcome.AVAILABLE, best)
