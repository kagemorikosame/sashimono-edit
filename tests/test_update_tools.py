"""鍵を作る道具（tools/update_keys.py）と、目録を作って署名する道具（tools/update_sign.py）

鍵はどれも試験の中で作る使い捨て 本物の鍵は作らない・読まない
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sashimono import __version__
from sashimono.update.manifest import parse_manifest
from sashimono.update.signing import (
    encrypt_private_key,
    parse_public_key,
    public_key_text,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parent.parent

PASSPHRASE = "試験の合言葉はこれですよね"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def keys_tool() -> ModuleType:
    return _load("update_keys")


@pytest.fixture(scope="module")
def sign_tool() -> ModuleType:
    return _load("update_sign")


def _answers(*values: str) -> Callable[[str], str]:
    queue = list(values)

    def ask(_prompt: str) -> str:
        return queue.pop(0)

    return ask


class TestMakingKeys:
    @pytest.fixture
    def light(self, keys_tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """試験では鍵のファイルを軽く包む（本物は 1 本 128 MB と 1 秒）"""
        original = keys_tool.generate

        def generate(out: Path, ask: Callable[[str], str], *, n: int = 2**10) -> list[str]:
            return list(original(out, ask, n=n))

        monkeypatch.setattr(keys_tool, "generate", generate)
        yield

    @pytest.mark.usefixtures("light")
    def test_two_keys_and_only_public_keys_on_screen(
        self, keys_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """秘密鍵と合言葉は画面にも記録にも出さない 出すのは貼る公開鍵だけ"""
        code = keys_tool.main(
            ["generate", "--out", str(tmp_path)],
            ask=_answers(PASSPHRASE, PASSPHRASE, PASSPHRASE + "予備", PASSPHRASE + "予備"),
        )
        assert code == 0
        out = capsys.readouterr().out
        files = sorted(tmp_path.iterdir())
        assert [f.name for f in files] == [
            "sashimono-update-current.key",
            "sashimono-update-spare.key",
        ]
        printed = [line.split('"')[1] for line in out.splitlines() if "sashimono-ed25519:" in line]
        assert len(printed) == 2 and printed[0] != printed[1]
        for key in printed:
            parse_public_key(key)
        assert PASSPHRASE not in out
        for path in files:
            blob = path.read_bytes()
            assert keys_tool.key_file_public_key(blob) in printed
            secret = json.loads(blob)["ciphertext"]
            assert secret not in out

    def test_existing_keys_are_not_overwritten(self, keys_tool: ModuleType, tmp_path: Path) -> None:
        """前の鍵を上書きで失くすと、その鍵で署名した版を出せなくなる"""
        (tmp_path / "sashimono-update-spare.key").write_text("前の鍵", encoding="utf-8")
        code = keys_tool.main(["generate", "--out", str(tmp_path)], ask=_answers())
        assert code == 1
        assert (tmp_path / "sashimono-update-spare.key").read_text(encoding="utf-8") == "前の鍵"
        assert not (tmp_path / "sashimono-update-current.key").exists()

    def test_a_short_or_mistyped_passphrase_is_asked_again(
        self, keys_tool: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """打ち間違えたまま包むと、その鍵は二度と開けない"""
        ask = _answers("短い", PASSPHRASE, PASSPHRASE + "x", PASSPHRASE, PASSPHRASE)
        assert keys_tool.ask_passphrase("今使う鍵", ask) == PASSPHRASE
        out = capsys.readouterr().out
        assert "文字以上" in out and "1 回目と違います" in out

    def test_show_prints_the_public_key(
        self, keys_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        key = Ed25519PrivateKey.generate()
        path = tmp_path / "k.key"
        path.write_bytes(encrypt_private_key(key, PASSPHRASE, n=2**10))
        assert keys_tool.main(["show", str(path)]) == 0
        assert capsys.readouterr().out.strip() == public_key_text(key.public_key())


def _zip(path: Path, *, version: str = __version__, info: bool = True) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Sashimono/Sashimono.exe", b"MZ")
        if info:
            archive.writestr(
                "Sashimono/build-info.json", json.dumps({"version": version, "python_abi": "cp314"})
            )
    return path


class TestTheManifest:
    def test_it_is_made_from_the_zip(self, sign_tool: ModuleType, tmp_path: Path) -> None:
        """版と Python は zip の中の書き付けから読む 手で打つと食い違う"""
        archive = _zip(tmp_path / "SashimonoEdit-1.2.3-windows-x64.zip", version="1.2.3")
        manifest = parse_manifest(sign_tool.build_manifest(archive, minimum="1.0.0"))
        assert (manifest.version, manifest.minimum, manifest.python_abi) == (
            "1.2.3",
            "1.0.0",
            "cp314",
        )
        assert manifest.package.url.endswith(
            "/releases/download/v1.2.3/SashimonoEdit-1.2.3-windows-x64.zip"
        )
        assert manifest.notes_url.endswith("/releases/tag/v1.2.3")
        assert manifest.package.size == archive.stat().st_size

    def test_a_zip_without_build_info_is_refused(
        self, sign_tool: ModuleType, tmp_path: Path
    ) -> None:
        with pytest.raises(sign_tool.ReleaseError):
            sign_tool.build_manifest(_zip(tmp_path / "x.zip", info=False))

    def test_a_minimum_above_the_version_is_refused(
        self, sign_tool: ModuleType, tmp_path: Path
    ) -> None:
        """自分自身も受け取れない目録になる"""
        with pytest.raises(sign_tool.ReleaseError):
            sign_tool.build_manifest(_zip(tmp_path / "x.zip", version="1.0.0"), minimum="2.0.0")


class TestSigning:
    @pytest.fixture
    def key(self) -> Ed25519PrivateKey:
        return Ed25519PrivateKey.generate()

    @pytest.fixture
    def key_file(self, key: Ed25519PrivateKey, tmp_path: Path) -> Path:
        path = tmp_path / "current.key"
        path.write_bytes(encrypt_private_key(key, PASSPHRASE, n=2**10))
        return path

    @pytest.fixture
    def manifest(self, sign_tool: ModuleType, tmp_path: Path) -> Path:
        path = tmp_path / "update.json"
        path.write_bytes(sign_tool.build_manifest(_zip(tmp_path / "x.zip")))
        return path

    def test_the_signature_is_written_beside(
        self,
        sign_tool: ModuleType,
        key: Ed25519PrivateKey,
        key_file: Path,
        manifest: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sign_tool, "TRUSTED_PUBLIC_KEYS", (public_key_text(key.public_key()),))
        code = sign_tool.main(
            ["sign", str(manifest), "--key", str(key_file)], ask=_answers(PASSPHRASE)
        )
        assert code == 0
        signature = manifest.with_name("update.json.sig").read_bytes()
        assert verify_manifest(manifest.read_bytes(), signature, [key.public_key()])

    def test_a_key_not_in_the_source_is_warned(
        self,
        sign_tool: ModuleType,
        key_file: Path,
        manifest: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """埋め込んでいない鍵の署名は、配った版が信じない 上げる前に気付けるように"""
        monkeypatch.setattr(sign_tool, "TRUSTED_PUBLIC_KEYS", ())
        code = sign_tool.main(
            ["sign", str(manifest), "--key", str(key_file)], ask=_answers(PASSPHRASE)
        )
        assert code == 2
        assert "埋め込まれていない" in capsys.readouterr().out

    def test_a_wrong_passphrase_writes_nothing(
        self, sign_tool: ModuleType, key_file: Path, manifest: Path
    ) -> None:
        code = sign_tool.main(
            ["sign", str(manifest), "--key", str(key_file)], ask=_answers("違う合言葉ですけれど")
        )
        assert code == 1
        assert not manifest.with_name("update.json.sig").exists()

    def test_an_unreadable_manifest_is_not_signed(
        self, sign_tool: ModuleType, key_file: Path, tmp_path: Path
    ) -> None:
        """ソフトが読めない目録に署名すると、上げても誰も受け取れない"""
        broken = tmp_path / "update.json"
        broken.write_text('{"schema": 1}', encoding="utf-8")
        code = sign_tool.main(
            ["sign", str(broken), "--key", str(key_file)], ask=_answers(PASSPHRASE)
        )
        assert code == 1
        assert not broken.with_name("update.json.sig").exists()

    def test_verify_uses_the_embedded_keys(
        self,
        sign_tool: ModuleType,
        key: Ed25519PrivateKey,
        manifest: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sashimono.update.signing import sign_manifest

        manifest.with_name("update.json.sig").write_bytes(sign_manifest(manifest.read_bytes(), key))
        monkeypatch.setattr(sign_tool, "trusted_keys", lambda: (key.public_key(),))
        assert sign_tool.main(["verify", str(manifest)]) == 0
        manifest.write_bytes(manifest.read_bytes().replace(b'"minimum"', b'"minimum" '))
        assert sign_tool.main(["verify", str(manifest)]) == 1

    def test_verify_without_keys_fails(
        self,
        sign_tool: ModuleType,
        key: Ed25519PrivateKey,
        manifest: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """鍵を貼り忘れた版を配ると、誰も更新を受け取れない 上げる前に止める"""
        from sashimono.update.signing import sign_manifest

        manifest.with_name("update.json.sig").write_bytes(sign_manifest(manifest.read_bytes(), key))
        monkeypatch.setattr(sign_tool, "trusted_keys", lambda: ())
        assert sign_tool.main(["verify", str(manifest)]) == 1
        assert "公開鍵が入っていない" in capsys.readouterr().out

    def test_a_throwaway_key_does_not_pass_the_real_keys(
        self, sign_tool: ModuleType, key: Ed25519PrivateKey, manifest: Path
    ) -> None:
        """埋め込んだ本物の公開鍵は、試験で作った使い捨ての鍵の署名を信じない"""
        from sashimono.update.signing import sign_manifest

        manifest.with_name("update.json.sig").write_bytes(sign_manifest(manifest.read_bytes(), key))
        assert sign_tool.main(["verify", str(manifest)]) == 1


class TestTheTag:
    def test_a_matching_tag_passes(self, sign_tool: ModuleType) -> None:
        assert sign_tool.main(["check-tag", f"v{__version__}"]) == 0

    @pytest.mark.parametrize("tag", ["v9.9.9", __version__, f"v{__version__}-fix"])
    def test_a_different_tag_fails(self, sign_tool: ModuleType, tag: str) -> None:
        """食い違ったまま配ると、更新したのに「最新です」と言い続ける版ができる"""
        assert sign_tool.main(["check-tag", tag]) == 1


def test_nothing_secret_is_in_the_repository() -> None:
    """鍵のファイルをリポジトリへ入れない（.gitignore で外し、入っていたら落とす）"""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.key" in ignore.split()
    for folder in ("src", "tests", "tools", "docs"):
        assert list((ROOT / folder).rglob("*.key")) == []
