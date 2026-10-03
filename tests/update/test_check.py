"""新しい版があるかの判断 署名・最低版・ベータ・飛ばした版・間隔"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sashimono.links import STABLE_MANIFEST_URL
from sashimono.update.check import (
    CHECK_INTERVAL_SECONDS,
    Outcome,
    check_for_update,
    is_due,
)
from sashimono.update.fetch import MemoryTransport
from sashimono.update.signing import SIGNATURE_SUFFIX
from tests.update.helpers import manifest_for, package_bytes, publish, release


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


class TestTheAnswer:
    def test_a_newer_version_is_offered(self, key: Ed25519PrivateKey) -> None:
        transport, manifest = release(key, "1.2.0")
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.AVAILABLE
        assert result.manifest == manifest

    def test_the_same_version_is_up_to_date(self, key: Ed25519PrivateKey) -> None:
        transport, _ = release(key, "1.2.0")
        result = check_for_update("1.2.0", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.UP_TO_DATE

    def test_an_older_release_is_not_offered(self, key: Ed25519PrivateKey) -> None:
        """古い目録を差し出されても戻らない（署名の通った古い目録の使い回しに乗らない）"""
        transport, _ = release(key, "1.0.0")
        result = check_for_update("1.2.0", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.UP_TO_DATE

    def test_below_the_minimum_is_sent_to_the_page(self, key: Ed25519PrivateKey) -> None:
        """minimum より古い版は自動では入れない 目録の形を変えたときの逃げ道"""
        transport, _ = release(key, "2.0.0", minimum="1.5.0")
        result = check_for_update("1.2.0", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.MANUAL

    def test_without_keys_nothing_is_asked(self, key: Ed25519PrivateKey) -> None:
        """鍵の入っていない版は確かめない（黙って何もしない） 取りにも行かない"""
        transport, _ = release(key, "1.2.0")
        result = check_for_update("1.1.0", transport=transport, keys=[])
        assert result.outcome is Outcome.NOT_CONFIGURED
        assert transport.requested == []


class TestTheSignatureGate:
    def test_a_rewritten_manifest_is_ignored(self, key: Ed25519PrivateKey) -> None:
        """署名の後で SHA-256 を書き換えた目録を信じると、偽の zip を入れることになる"""
        transport, manifest = release(key, "1.2.0")
        for url, page in list(transport.pages.items()):
            if url.endswith("update.json"):
                transport.pages[url] = page.replace(manifest.package.sha256.encode(), b"0" * 64)
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.FAILED

    def test_a_stranger_key_is_ignored(self, key: Ed25519PrivateKey) -> None:
        stranger = Ed25519PrivateKey.generate()
        transport, _ = release(stranger, "1.2.0")
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.FAILED

    def test_the_spare_key_is_accepted(self, key: Ed25519PrivateKey) -> None:
        """今使う鍵を失くした後、予備の鍵で署名した版を受け取れる"""
        spare = Ed25519PrivateKey.generate()
        transport, _ = release(spare, "1.2.0")
        result = check_for_update(
            "1.1.0", transport=transport, keys=[key.public_key(), spare.public_key()]
        )
        assert result.outcome is Outcome.AVAILABLE

    def test_a_missing_signature_is_ignored(self, key: Ed25519PrivateKey) -> None:
        transport, _ = release(key, "1.2.0")
        del transport.redirects[STABLE_MANIFEST_URL + SIGNATURE_SUFFIX]
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.FAILED

    def test_a_broken_but_signed_manifest_is_ignored(self, key: Ed25519PrivateKey) -> None:
        """署名が通っても、読めない目録で何かを決めない"""
        transport = MemoryTransport()
        body = package_bytes("1.2.0")
        publish(transport, key, manifest_for("1.2.0", body), body, data=b'{"schema": 1}')
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.FAILED

    def test_offline_is_a_quiet_failure(self, key: Ed25519PrivateKey) -> None:
        """繋がらないときは例外にせず FAILED を返す（起動時の確認はこれを黙って流す）"""
        result = check_for_update("1.1.0", transport=MemoryTransport(), keys=[key.public_key()])
        assert result.outcome is Outcome.FAILED
        assert result.reason


class TestTheBetaChannel:
    def test_beta_is_not_read_unless_asked(self, key: Ed25519PrivateKey) -> None:
        transport, _ = release(key, "1.2.0")
        release(key, "1.3.0b1", transport=transport, beta=True)
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()])
        assert result.manifest is not None and result.manifest.version == "1.2.0"

    def test_the_newer_of_the_two_is_taken(self, key: Ed25519PrivateKey) -> None:
        transport, _ = release(key, "1.2.0")
        release(key, "1.3.0b1", transport=transport, beta=True)
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()], beta=True)
        assert result.manifest is not None and result.manifest.version == "1.3.0b1"

    def test_a_stale_beta_does_not_hide_the_release(self, key: Ed25519PrivateKey) -> None:
        """ベータの目録を直し忘れても、正式版の新しい版を受け取る"""
        transport, _ = release(key, "1.3.0")
        release(key, "1.3.0b1", transport=transport, beta=True)
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()], beta=True)
        assert result.manifest is not None and result.manifest.version == "1.3.0"

    def test_a_missing_beta_does_not_fail(self, key: Ed25519PrivateKey) -> None:
        """beta のタグがまだ無い間も、正式版は受け取れる"""
        transport, _ = release(key, "1.2.0")
        result = check_for_update("1.1.0", transport=transport, keys=[key.public_key()], beta=True)
        assert result.outcome is Outcome.AVAILABLE


class TestSkippedVersions:
    def test_a_skipped_version_is_not_offered(self, key: Ed25519PrivateKey) -> None:
        transport, _ = release(key, "1.2.0")
        result = check_for_update(
            "1.1.0", transport=transport, keys=[key.public_key()], skipped=["1.2.0"]
        )
        assert result.outcome is Outcome.UP_TO_DATE

    def test_a_skipped_beta_leaves_the_release(self, key: Ed25519PrivateKey) -> None:
        transport, _ = release(key, "1.2.0")
        release(key, "1.3.0b1", transport=transport, beta=True)
        result = check_for_update(
            "1.1.0",
            transport=transport,
            keys=[key.public_key()],
            beta=True,
            skipped=["1.3.0b1"],
        )
        assert result.manifest is not None and result.manifest.version == "1.2.0"


class TestHowOften:
    def test_not_again_within_the_interval(self) -> None:
        """起動のたびに取りに行かない 何度も開き直す人の分だけ GitHub へ行く"""
        assert not is_due(1000.0, 1000.0 + CHECK_INTERVAL_SECONDS - 1)

    def test_again_after_it(self) -> None:
        assert is_due(1000.0, 1000.0 + CHECK_INTERVAL_SECONDS)

    def test_the_first_time(self) -> None:
        assert is_due(0.0, 1000.0 + CHECK_INTERVAL_SECONDS)

    def test_a_clock_moved_back(self) -> None:
        """時計が戻ったときに待ち続けない（前の時刻が未来になる）"""
        assert is_due(5000.0, 1000.0)

    def test_hours_not_minutes(self) -> None:
        assert 3 * 60 * 60 <= CHECK_INTERVAL_SECONDS <= 12 * 60 * 60
