"""目録の署名と鍵のファイル 鍵はどれも試験の中で作る使い捨て"""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sashimono.update import signing
from sashimono.update.signing import (
    KEY_PREFIX,
    TRUSTED_PUBLIC_KEYS,
    KeyFileError,
    decrypt_private_key,
    encrypt_private_key,
    key_file_public_key,
    parse_public_key,
    public_key_text,
    sign_manifest,
    trusted_keys,
    verify_manifest,
)

MANIFEST = b'{"schema": 1, "version": "1.2.3"}\n'

#: 試験では鍵のファイルを軽く包む 本物の重さ（2**17）だと 1 本ごとに 128 MB と 1 秒かかる
LIGHT = 2**10


@pytest.fixture
def current() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def spare() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


class TestTheSignature:
    def test_a_signed_manifest_is_trusted(self, current: Ed25519PrivateKey) -> None:
        signature = sign_manifest(MANIFEST, current)
        assert verify_manifest(MANIFEST, signature, [current.public_key()])

    def test_a_rewritten_manifest_is_not(self, current: Ed25519PrivateKey) -> None:
        """版や SHA-256 を 1 文字書き換えた目録を通すと、偽の zip を入れることになる"""
        signature = sign_manifest(MANIFEST, current)
        assert not verify_manifest(
            MANIFEST.replace(b"1.2.3", b"1.2.4"), signature, [current.public_key()]
        )

    def test_a_reformatted_manifest_is_not(self, current: Ed25519PrivateKey) -> None:
        """署名はバイト列そのものに付く 読み直して書き直した目録は通らない（手順書の注意）"""
        signature = sign_manifest(MANIFEST, current)
        assert not verify_manifest(MANIFEST.rstrip(b"\n"), signature, [current.public_key()])

    def test_a_different_key_is_not(
        self, current: Ed25519PrivateKey, spare: Ed25519PrivateKey
    ) -> None:
        """埋め込んでいない鍵の署名を通すと、誰の鍵でも更新を配れる"""
        signature = sign_manifest(MANIFEST, spare)
        assert not verify_manifest(MANIFEST, signature, [current.public_key()])

    def test_the_spare_key_is_trusted(
        self, current: Ed25519PrivateKey, spare: Ed25519PrivateKey
    ) -> None:
        """今使う鍵を失くしたとき、予備の鍵で署名した版を配った版が受け取れる"""
        signature = sign_manifest(MANIFEST, spare)
        assert verify_manifest(MANIFEST, signature, [current.public_key(), spare.public_key()])

    def test_no_keys_trust_nothing(self, current: Ed25519PrivateKey) -> None:
        """鍵の入っていない版は、どの署名も信じない（その版は更新を確かめない）"""
        assert not verify_manifest(MANIFEST, sign_manifest(MANIFEST, current), [])

    def test_a_signature_made_for_something_else_is_not(self, current: Ed25519PrivateKey) -> None:
        """同じ鍵で別の物に付けた署名を、目録の署名として通さない（頭に目録の印を混ぜる）"""
        bare = base64.b64encode(current.sign(MANIFEST))
        assert not verify_manifest(MANIFEST, bare, [current.public_key()])

    @pytest.mark.parametrize("signature", [b"", b"not base64!", b"QUJD", b"A" * 5000])
    def test_garbage_is_not(self, current: Ed25519PrivateKey, signature: bytes) -> None:
        assert not verify_manifest(MANIFEST, signature, [current.public_key()])


class TestThePublicKey:
    def test_it_round_trips(self, current: Ed25519PrivateKey) -> None:
        text = public_key_text(current.public_key())
        assert text.startswith(KEY_PREFIX)
        signature = sign_manifest(MANIFEST, current)
        assert verify_manifest(MANIFEST, signature, [parse_public_key(text)])

    @pytest.mark.parametrize(
        "text",
        [
            "AAAA",
            KEY_PREFIX + "not base64!",
            KEY_PREFIX + base64.b64encode(b"x" * 31).decode(),
            KEY_PREFIX + base64.b64encode(b"x" * 64).decode(),
        ],
    )
    def test_a_wrong_paste_is_refused(self, text: str) -> None:
        """頭の無い物・長さの違う物（秘密鍵の中身を貼った、など）は公開鍵として読まない"""
        with pytest.raises(ValueError):
            parse_public_key(text)

    def test_unreadable_keys_are_skipped(self, current: Ed25519PrivateKey) -> None:
        """貼り間違いで起動を止めない 読める鍵だけを使う"""
        found = trusted_keys(["broken", public_key_text(current.public_key())])
        assert len(found) == 1

    def test_the_embedded_keys_are_readable(self) -> None:
        """埋め込んだ公開鍵は全部読めて、多くても 2 本（今使う鍵と予備の鍵）

        読めない物が混ざると、その鍵で署名した版を配った版が受け取れない
        """
        assert len(trusted_keys(TRUSTED_PUBLIC_KEYS)) == len(TRUSTED_PUBLIC_KEYS)
        assert len(TRUSTED_PUBLIC_KEYS) <= 2


class TestTheKeyFile:
    def test_it_opens_with_the_passphrase(self, current: Ed25519PrivateKey) -> None:
        blob = encrypt_private_key(current, "正しい合言葉です長め", n=LIGHT)
        opened = decrypt_private_key(blob, "正しい合言葉です長め")
        assert opened.private_bytes_raw() == current.private_bytes_raw()

    def test_a_wrong_passphrase_does_not(self, current: Ed25519PrivateKey) -> None:
        blob = encrypt_private_key(current, "正しい合言葉です長め", n=LIGHT)
        with pytest.raises(KeyFileError):
            decrypt_private_key(blob, "違う合言葉です長めの")

    def test_the_secret_is_not_in_the_file(self, current: Ed25519PrivateKey) -> None:
        """ファイルが漏れても、合言葉が無ければ秘密鍵は読めない"""
        blob = encrypt_private_key(current, "正しい合言葉です長め", n=LIGHT)
        raw = current.private_bytes_raw()
        assert raw not in blob
        assert base64.b64encode(raw) not in blob
        assert raw.hex().encode() not in blob

    def test_the_public_key_is_shown_without_the_passphrase(
        self, current: Ed25519PrivateKey
    ) -> None:
        blob = encrypt_private_key(current, "正しい合言葉です長め", n=LIGHT)
        assert key_file_public_key(blob) == public_key_text(current.public_key())

    def test_a_swapped_public_key_is_caught(
        self, current: Ed25519PrivateKey, spare: Ed25519PrivateKey
    ) -> None:
        """添えた公開鍵を書き換えたファイルは開けない 別の鍵だと思って署名しない"""
        blob = encrypt_private_key(current, "正しい合言葉です長め", n=LIGHT)
        forged = blob.replace(
            public_key_text(current.public_key()).encode(),
            public_key_text(spare.public_key()).encode(),
        )
        with pytest.raises(KeyFileError):
            decrypt_private_key(forged, "正しい合言葉です長め")

    def test_an_empty_passphrase_is_refused(self, current: Ed25519PrivateKey) -> None:
        with pytest.raises(KeyFileError):
            encrypt_private_key(current, "", n=LIGHT)

    @pytest.mark.parametrize("blob", [b"", b"{}", b"not json", b'{"format": "other"}'])
    def test_other_files_are_refused(self, blob: bytes) -> None:
        with pytest.raises(KeyFileError):
            decrypt_private_key(blob, "正しい合言葉です長め")

    def test_the_real_weight_is_heavy(self) -> None:
        """本物の鍵のファイルは重く包む 軽いと、漏れたファイルの合言葉を総当たりで試せる"""
        assert signing.SCRYPT_N >= 2**17
