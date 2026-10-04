"""リリースを 1 コマンドで終える道具（tools/release.py）

git と gh は偽物に差し替える 本物の GitHub へは何も書かない 鍵は試験の中で作る使い捨て
確かめるのは、済んだ段を飛ばすこと・落ちたら公開しないこと・人が y と答えなければ
公開しないこと・公開済みや別の commit を指すタグで止まること
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sashimono.update.signing import encrypt_private_key, public_key_text

ROOT = Path(__file__).resolve().parent.parent

VERSION = "1.2.3"
TAG = f"v{VERSION}"
ZIP = f"SashimonoEdit-{VERSION}-windows-x64.zip"
HEAD = "a" * 40
OLD = "b" * 40
STRANGER = "c" * 40
PASSPHRASE = "試験の合言葉はこれですよね"
SOURCES = ("sources-manifest.json", "sources-SHA256SUMS.txt", "ffmpeg-8.1.2.tar.xz")


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release", ROOT / "tools" / "release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _zip_bytes(version: str = VERSION) -> bytes:
    path_like = io.BytesIO()
    with zipfile.ZipFile(path_like, "w") as archive:
        archive.writestr("Sashimono/Sashimono.exe", b"MZ")
        archive.writestr(
            "Sashimono/build-info.json", json.dumps({"version": version, "python_abi": "cp314"})
        )
        archive.writestr("Sashimono/_internal/PySide6/Qt6Core.dll", b"dll")
    return path_like.getvalue()


def _asset(name: str, data: bytes) -> dict[str, Any]:
    return {
        "name": name,
        "size": len(data),
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
    }


@dataclass
class FakeGitHub:
    """git と gh の偽物 書き込みは ``writes`` に残し、資産は ``files`` に持つ"""

    branch: str = "main"
    dirty: str = ""
    head: str = HEAD
    remote_main: str = HEAD
    remote_tag: str | None = OLD
    local_tag: str | None = None
    ancestors: set[str] = field(default_factory=lambda: {OLD})
    tag_version: str = VERSION
    auth: bool = True
    runs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    draft: bool | None = True
    prerelease: bool = False
    beta_release: bool = False
    files: dict[str, bytes] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)
    writes: list[list[str]] = field(default_factory=list)
    #: 何回目の ``gh run list`` で run を終えたことにするか（待つ試験）
    finish_after: int = 0
    listed: int = 0

    def __post_init__(self) -> None:
        if not self.runs:
            commit = self.remote_tag or self.head
            self.runs = {name: [_run(commit, name)] for name in ("CI", "Package", "release.yml")}

    def __call__(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(arguments)
        self.calls.append(args)
        out, code = self.answer(args)
        return subprocess.CompletedProcess(args, code, out, "" if code == 0 else "偽物の失敗")

    def answer(self, args: list[str]) -> tuple[str, int]:
        match args:
            case ["gh", "auth", "status"]:
                return "", 0 if self.auth else 1
            case ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
                return self.branch + "\n", 0
            case ["git", "status", *_]:
                return self.dirty, 0
            case ["git", "rev-parse", "HEAD"]:
                return self.head + "\n", 0
            case ["git", "ls-remote", *_]:
                lines = [f"{self.remote_main}\trefs/heads/main"]
                if self.remote_tag:
                    lines += [
                        f"{'d' * 40}\trefs/tags/{TAG}",
                        f"{self.remote_tag}\trefs/tags/{TAG}^{{}}",
                    ]
                return "\n".join(lines) + "\n", 0
            case ["git", "rev-parse", "-q", "--verify", _]:
                return (self.local_tag + "\n", 0) if self.local_tag else ("", 1)
            case ["git", "merge-base", "--is-ancestor", commit, "HEAD"]:
                return "", 0 if commit in self.ancestors else 1
            case ["git", "show", _]:
                return f'__version__ = "{self.tag_version}"\n', 0
            case ["git", "tag", *_]:
                self.writes.append(args)
                self.local_tag = self.head
                return "", 0
            case ["git", "push", *_]:
                self.writes.append(args)
                self.remote_tag = self.local_tag
                return "", 0
            case ["gh", "run", "list", *rest]:
                self.listed += 1
                name = rest[rest.index("--workflow") + 1]
                runs = self.runs.get(name, [])
                if self.listed <= self.finish_after:
                    runs = [{**run, "status": "in_progress", "conclusion": ""} for run in runs]
                return json.dumps(runs), 0
            case ["gh", "release", "view", "beta"]:
                return "", 0 if self.beta_release else 1
            case ["gh", "release", "view", tag, *_] if tag == TAG:
                if self.draft is None:
                    return "", 1
                assets = [_asset(name, data) for name, data in self.files.items()]
                info = {
                    "isDraft": self.draft,
                    "isPrerelease": self.prerelease,
                    "assets": assets,
                    "url": "https://example.invalid/release",
                }
                return json.dumps(info), 0
            case ["gh", "release", "download", _, "--pattern", name, "--dir", folder, *_]:
                Path(folder).mkdir(parents=True, exist_ok=True)
                (Path(folder) / name).write_bytes(self.files[name])
                return "", 0
            case ["gh", "release", "upload", tag, *paths]:
                self.writes.append(args)
                if tag == TAG:
                    for path in paths:
                        if not path.startswith("--"):
                            self.files[Path(path).name] = Path(path).read_bytes()
                return "", 0
            case ["gh", "release", "edit", *_]:
                self.writes.append(args)
                self.draft = False
                return "", 0
        raise AssertionError(f"偽物が知らない呼び方: {args}")


def _run(commit: str, name: str, conclusion: str = "success") -> dict[str, Any]:
    return {
        "databaseId": 1,
        "workflowName": name,
        "status": "completed",
        "conclusion": conclusion,
        "headSha": commit,
        "url": f"https://example.invalid/{name}",
        "event": "push",
    }


@dataclass
class World:
    """1 回のリリースの周り 偽の GitHub・手元の木・鍵・人の答え"""

    root: Path
    github: FakeGitHub
    key: Path
    public: str
    answers: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    smoked: list[Path] = field(default_factory=list)
    launched: list[Path] = field(default_factory=list)
    collected: list[Path] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    smoke_result: int = 0
    latest: bytes = b""
    passphrase: str = PASSPHRASE

    def ask(self, prompt: str) -> str:
        return self.answers.pop(0) if self.answers else ""

    def secret(self, prompt: str) -> str:
        self.secrets.append(prompt)
        return self.passphrase

    def smoke(self, archive: Path, home: Path) -> int:
        self.smoked.append(archive)
        return self.smoke_result

    def launch(self, executable: Path) -> int:
        self.launched.append(executable)
        return 0

    def collect(self, dist: Path) -> int:
        self.collected.append(dist)
        folder = dist / "sources"
        folder.mkdir(parents=True, exist_ok=True)
        for name in SOURCES:
            (folder / name).write_bytes(name.encode())
        return 0

    def fetch(self, url: str) -> bytes:
        self.fetched.append(url)
        return self.latest or self.github.files["update.json"]

    def main(self, tool: ModuleType, *extra: str, key: bool = True) -> int:
        argv = [VERSION, *extra]
        if key:
            argv += ["--key", str(self.key)]
        code: int = tool.main(
            argv,
            root=self.root,
            run=self.github,
            ask=self.ask,
            secret=self.secret,
            sleep=lambda _seconds: None,
            fetch=self.fetch,
            smoke=self.smoke,
            launch=self.launch,
            collect=self.collect,
        )
        return code

    @property
    def published(self) -> bool:
        return any(args[:3] == ["gh", "release", "edit"] for args in self.github.writes)


@pytest.fixture
def world(tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    """0.1.0 と同じ形 タグと下書き（zip とソース）はあり、目録と署名はまだ"""
    root = tmp_path / "木"
    (root / "src" / "sashimono").mkdir(parents=True)
    (root / "src" / "sashimono" / "__init__.py").write_text(
        f'__version__ = "{VERSION}"\n', encoding="utf-8"
    )
    secret = Ed25519PrivateKey.generate()
    key = tmp_path / "鍵" / "sashimono-update-current.key"
    key.parent.mkdir()
    key.write_bytes(encrypt_private_key(secret, PASSPHRASE, n=2**10))
    public = public_key_text(secret.public_key())
    spare = public_key_text(Ed25519PrivateKey.generate().public_key())
    monkeypatch.setattr(tool, "TRUSTED_PUBLIC_KEYS", (public, spare))
    monkeypatch.setattr(tool.update_sign, "trusted_keys", lambda: (secret.public_key(),))
    files = {ZIP: _zip_bytes(), **{name: name.encode() for name in SOURCES}}
    github = FakeGitHub(head=HEAD, remote_main=HEAD, remote_tag=HEAD, ancestors=set(), files=files)
    yield World(root=root, github=github, key=key, public=public)


def _signed(world: World, tool: ModuleType, tmp_path: Path) -> None:
    """下書きに目録と署名を上げた後の形にする"""
    archive = tmp_path / ZIP
    archive.write_bytes(world.github.files[ZIP])
    manifest = tmp_path / "update.json"
    manifest.write_bytes(tool.update_sign.build_manifest(archive))
    tool.update_sign.sign_file(manifest, world.key, lambda _prompt: PASSPHRASE)
    world.github.files["update.json"] = manifest.read_bytes()
    world.github.files["update.json.sig"] = (tmp_path / "update.json.sig").read_bytes()


class TestTheWholeRun:
    def test_from_the_draft_it_signs_and_publishes_only_after_yes(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """0.1.0 の形 タグ・組み立て・ソースは飛ばし、確かめて署名し、y で公開する"""
        world.answers = ["y", "y"]
        assert world.main(tool) == 0
        out = capsys.readouterr().out
        writes = world.github.writes
        assert not any(args[:2] == ["git", "tag"] for args in writes)
        assert not any(args[:2] == ["git", "push"] for args in writes)
        uploads = [args for args in writes if args[:3] == ["gh", "release", "upload"]]
        assert len(uploads) == 1
        assert [Path(p).name for p in uploads[0][4:6]] == ["update.json", "update.json.sig"]
        assert writes[-1] == ["gh", "release", "edit", TAG, "--draft=false", "--latest"]
        assert world.collected == []
        assert world.smoked and world.launched
        assert len(world.secrets) == 1
        assert "TRUSTED_PUBLIC_KEYS の 1 本目" in out
        assert "資産      6 個" in out
        assert world.fetched == [tool.STABLE_MANIFEST_URL]
        # 合言葉は画面に出さない
        assert PASSPHRASE not in out

    def test_anything_but_yes_keeps_the_draft(self, tool: ModuleType, world: World) -> None:
        """公開は人が y と答えたときだけ 空・n・ほかの文字では下書きのまま"""
        # 2 回目からは GL を見終えた印があるので、尋ねるのは公開だけ
        for answers in (["y", ""], ["n"], ["はい"], ["yes please"]):
            world.answers = answers
            assert world.main(tool) == 0
            assert world.answers == []
            assert not world.published

    def test_a_second_run_skips_what_is_done(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """2 回目は確かめと署名を飛ばし、合言葉も尋ねず、公開の所から尋ねる"""
        world.answers = ["y", "n"]
        assert world.main(tool) == 0
        capsys.readouterr()
        smoked, launched = len(world.smoked), len(world.launched)
        world.answers = ["n"]
        assert world.main(tool) == 0
        out = capsys.readouterr().out
        assert len(world.smoked) == smoked and len(world.launched) == launched
        assert len(world.secrets) == 1
        assert "下書きの目録と署名が通った" in out
        uploads = [a for a in world.github.writes if a[:3] == ["gh", "release", "upload"]]
        assert len(uploads) == 1

    def test_a_folder_given_to_key_uses_the_current_key_inside(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world.key = world.key.parent
        world.answers = ["y", "n"]
        assert world.main(tool) == 0
        assert "中の今使う鍵を使う" in capsys.readouterr().out

    def test_a_folder_without_the_current_key_stops_before_anything(
        self, tool: ModuleType, world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = tmp_path / "空"
        empty.mkdir()
        world.key = empty
        assert world.main(tool) == 1
        assert "sashimono-update-current.key が無い" in capsys.readouterr().out
        assert world.github.writes == []

    def test_an_untrusted_key_stops_before_tagging(
        self, tool: ModuleType, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """埋め込んでいない鍵で署名しても配った版は信じない タグを打つ前に止める"""
        monkeypatch.setattr(tool, "TRUSTED_PUBLIC_KEYS", ("sashimono-ed25519:別",))
        world.github.remote_tag = None
        assert world.main(tool) == 1
        assert world.github.writes == []


class TestPrerequisites:
    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"branch": "phase/x"}, "main でない"),
            ({"dirty": " M src/x.py\n"}, "commit していない変更"),
            ({"remote_main": "e" * 40}, "origin/main"),
            ({"auth": False}, "gh が使えない"),
        ],
    )
    def test_a_missing_prerequisite_stops_without_writing(
        self,
        tool: ModuleType,
        world: World,
        capsys: pytest.CaptureFixture[str],
        change: dict[str, Any],
        message: str,
    ) -> None:
        for name, value in change.items():
            setattr(world.github, name, value)
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        out = capsys.readouterr().out
        assert message in out and "公開はしていない" in out
        assert world.github.writes == []

    def test_the_version_must_match_the_source(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (world.root / "src" / "sashimono" / "__init__.py").write_text(
            '__version__ = "9.9.9"\n', encoding="utf-8"
        )
        assert world.main(tool) == 1
        assert "__version__ 9.9.9" in capsys.readouterr().out
        assert world.github.writes == []

    @pytest.mark.parametrize("workflow", ["CI", "Package"])
    def test_a_failed_ci_run_stops(self, tool: ModuleType, world: World, workflow: str) -> None:
        world.github.runs[workflow] = [_run(HEAD, workflow, "failure")]
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert world.github.writes == []


class TestTheTag:
    def test_a_missing_tag_is_made_and_pushed(self, tool: ModuleType, world: World) -> None:
        world.github.remote_tag = None
        world.answers = ["y", "n"]
        assert world.main(tool) == 0
        writes = world.github.writes
        assert writes[0][:4] == ["git", "tag", "-a", TAG] and writes[0][-1] == HEAD
        assert writes[1] == ["git", "push", "origin", f"refs/tags/{TAG}"]

    def test_a_tag_on_a_commit_outside_main_stops(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """打ち間違えたタグの zip は、出したい物と違う 公開せず、何も上げない"""
        world.github.remote_tag = STRANGER
        world.github.runs = {n: [_run(STRANGER, n)] for n in ("CI", "Package", "release.yml")}
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert "main に無い commit" in capsys.readouterr().out
        assert world.github.writes == []

    def test_a_tag_whose_version_differs_stops(self, tool: ModuleType, world: World) -> None:
        world.github.remote_tag = OLD
        world.github.ancestors = {OLD}
        world.github.tag_version = "1.2.2"
        world.github.runs = {n: [_run(OLD, n)] for n in ("CI", "Package", "release.yml")}
        assert world.main(tool) == 1
        assert world.github.writes == []

    def test_a_tag_behind_main_with_the_same_version_is_used(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """タグの後に main へ文書の直しが入るのは普通 組み立てたのはタグの commit"""
        world.github.remote_tag = OLD
        world.github.ancestors = {OLD}
        world.github.runs = {n: [_run(OLD, n)] for n in ("CI", "Package", "release.yml")}
        world.answers = ["y", "n"]
        assert world.main(tool) == 0
        assert "手元の HEAD より前の bbbbbbb" in capsys.readouterr().out

    def test_a_local_tag_on_another_commit_stops(self, tool: ModuleType, world: World) -> None:
        world.github.remote_tag = None
        world.github.local_tag = STRANGER
        assert world.main(tool) == 1
        assert world.github.writes == []


class TestTheReleaseWorkflow:
    def test_it_waits_until_the_run_finishes(self, tool: ModuleType, world: World) -> None:
        world.github.finish_after = 4
        world.answers = ["y", "n"]
        assert world.main(tool) == 0
        assert world.github.listed > 4

    def test_a_failed_run_stops(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world.github.runs["release.yml"] = [_run(HEAD, "release.yml", "failure")]
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert "Release（v1.2.3） が failure" in capsys.readouterr().out
        assert not world.published


class TestTheDraft:
    def test_a_published_release_stops(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """公開した後の zip や目録は差し替えない 直すなら版を上げる"""
        world.github.draft = False
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert "公開済み" in capsys.readouterr().out
        assert world.github.writes == []

    def test_a_zip_that_differs_from_github_stops(self, tool: ModuleType, world: World) -> None:
        """落とした物が GitHub の digest と合わなければ、それを確かめて署名しない"""

        original = world.github.answer

        def corrupt(args: list[str]) -> tuple[str, int]:
            out, code = original(args)
            if args[:3] == ["gh", "release", "download"] and args[5] == ZIP:
                (Path(args[7]) / ZIP).write_bytes(b"broken")
            return out, code

        world.github.answer = corrupt  # type: ignore[method-assign]
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert not world.published and world.smoked == []

    def test_missing_sources_are_collected_and_uploaded(
        self, tool: ModuleType, world: World
    ) -> None:
        for name in SOURCES:
            del world.github.files[name]
        world.answers = ["y", "n"]
        assert world.main(tool) == 0
        assert len(world.collected) == 1
        uploaded = [a for a in world.github.writes if a[:3] == ["gh", "release", "upload"]]
        assert sorted(Path(p).name for p in uploaded[0][4:-1]) == sorted(SOURCES)


class TestTheCheckAndSignature:
    def test_a_failed_self_check_stops_before_signing(self, tool: ModuleType, world: World) -> None:
        world.smoke_result = 1
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert world.secrets == [] and not world.published

    def test_no_to_the_gl_question_stops(self, tool: ModuleType, world: World) -> None:
        world.answers = ["n", "y"]
        assert world.main(tool) == 1
        assert world.secrets == [] and not world.published

    def test_skip_launch_does_not_start_the_app(self, tool: ModuleType, world: World) -> None:
        world.answers = ["n"]
        assert world.main(tool, "--skip-launch") == 0
        assert world.launched == [] and world.smoked

    def test_a_signed_draft_is_verified_and_not_signed_again(
        self, tool: ModuleType, world: World, tmp_path: Path
    ) -> None:
        _signed(world, tool, tmp_path)
        world.answers = ["y", "y"]
        assert world.main(tool, key=False) == 0
        assert world.secrets == []
        assert world.published

    def test_a_bad_signature_on_the_draft_stops(
        self, tool: ModuleType, world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _signed(world, tool, tmp_path)
        world.github.files["update.json.sig"] = b"AAAA\n"
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert "下書きの目録が通らない" in capsys.readouterr().out
        assert not world.published

    def test_a_wrong_passphrase_stops_without_uploading(
        self, tool: ModuleType, world: World
    ) -> None:
        world.passphrase = "違う合言葉"
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert world.github.writes == []

    def test_publish_checks_the_stable_url(
        self, tool: ModuleType, world: World, tmp_path: Path
    ) -> None:
        """公開しても固定の URL が古い版のままなら、知らせて非 0 で終わる"""
        old = tmp_path / "old"
        old.mkdir()
        archive = old / "SashimonoEdit-1.0.0-windows-x64.zip"
        archive.write_bytes(_zip_bytes("1.0.0"))
        world.latest = tool.update_sign.build_manifest(archive)
        world.answers = ["y", "y"]
        assert world.main(tool) == 1
        assert world.published
        assert len(world.fetched) == tool.LATEST_TRIES


class TestDryRun:
    def test_it_writes_nothing_and_lists_what_is_left(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world.answers = ["y", "y"]
        assert world.main(tool, "--dry-run") == 0
        out = capsys.readouterr().out
        assert world.github.writes == []
        assert world.smoked == [] and world.launched == [] and world.secrets == []
        downloads = [a for a in world.github.calls if a[:3] == ["gh", "release", "download"]]
        assert downloads == []
        assert not (world.root / "dist").exists()
        assert f"[済] {TAG} は push 済み" in out
        assert "[済] Release（v1.2.3） が success" in out
        assert "[済] 下書きに sources-manifest.json" in out
        assert "[残] zip を落とす" in out
        assert "[残] 目録を作り" in out
        assert "[残] 要約を出して" in out
        assert world.answers == ["y", "y"]

    def test_it_reports_a_missing_prerequisite_and_goes_on(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world.github.branch = "phase/x"
        assert world.main(tool, "--dry-run") == 1
        out = capsys.readouterr().out
        assert "[NG] 手元が main でない" in out and "[済] 下書きに" in out
        assert world.github.writes == []

    def test_without_a_tag_it_says_the_tag_would_be_made(
        self, tool: ModuleType, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world.github.remote_tag = None
        world.github.draft = None
        world.github.runs["release.yml"] = []
        assert world.main(tool, "--dry-run") == 0
        out = capsys.readouterr().out
        assert f"[残] {TAG} を aaaaaaa に打って push する" in out
        assert "[残] 下書き" in out
        assert world.github.writes == []


def test_the_main_entry_has_no_flag_that_skips_the_question(tool: ModuleType) -> None:
    """確かめを全部飛ばす旗を足すと、公開を機械に任せられてしまう"""
    with pytest.raises(SystemExit):
        tool.main([VERSION, "--yes"])
