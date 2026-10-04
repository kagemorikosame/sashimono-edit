r"""リリースを 1 コマンドで終える（タグ → 下書きの確かめ → ソース → 自己診断 → 署名 → 公開）

    .venv\Scripts\python.exe tools\release.py 0.1.0 --key <鍵のファイルか、それを入れたフォルダ>
    .venv\Scripts\python.exe tools\release.py 0.1.0 --key <鍵> --dry-run  （何も変えない）

段ごとに、済んでいれば飛ばして続きから進む 途中で落ちたら公開せずに止まり、次に何をするかを出す
直してからもう一度同じコマンドを打てば、済んだ段は飛ばして続きから進む

1. 前提 引数の版と ``__version__`` が同じ 手元が main で汚れていない origin/main と同じ
   出す commit の CI と Package の run が success gh が使える
2. タグ ``v<版>`` 無ければ注釈付きで打って push する あれば、指す commit を確かめて飛ばす
3. Actions の Release（release.yml）がそのタグで終わるのを待つ
4. 下書き リリースが下書きであること zip を ``dist\release`` へ落とし、GitHub の digest と照らす
   公開済みなら差し替えず、目録が下書きのときと同じことを確かめて、公開した後の残りだけ続ける
5. ソース 索引（``sources-manifest.json``）に載った物が全部下書きにあるか照らし、足りなければ
   ``collect_sources.py`` で集めて、本体 → 索引の順に上げる
6. 自己診断 展開した zip を ``build_package.py`` と同じ確かめに掛け、アプリを起こして
   GL で描けているかを人に尋ねる（``--skip-launch`` で起こさない）
7. 目録と署名 ``update_sign.py`` の manifest → sign → verify で作って上げる
   上げてあれば落として確かめ、通れば飛ばす
8. 要約を出し、人が y と答えたときだけ公開する 公開した後、固定の URL が新しい版を指すかを見る

公開は必ず人が y と答える 確かめを全部飛ばす旗は作らない（取り返しが付かないのは公開だけで、
そこを機械に任せると、署名の通らない版や描けない版を配って全員の自動更新を止める）

秘密鍵と合言葉は画面にもファイルにも出さない 合言葉は ``update_sign.py sign`` と同じく getpass で
尋ね、鍵を開くのは署名する一瞬だけ 手順と中身は docs/development.md の「自動更新のリリース」
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# 中の処理は隣の道具をそのまま呼ぶ 同じ処理を書き直すと、片方だけ直して食い違う
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_package  # noqa: E402
import collect_sources  # noqa: E402
import update_sign  # noqa: E402
from update_keys import KEY_FILES  # noqa: E402

from sashimono.links import BETA_MANIFEST_URL, STABLE_MANIFEST_URL  # noqa: E402
from sashimono.update.manifest import (  # noqa: E402
    Manifest,
    ManifestError,
    PackageInfo,
    is_prerelease,
    parse_manifest,
)
from sashimono.update.package import APP_EXE  # noqa: E402
from sashimono.update.signing import (  # noqa: E402
    SIGNATURE_SUFFIX,
    TRUSTED_PUBLIC_KEYS,
    KeyFileError,
    key_file_public_key,
    parse_public_key,
    verify_manifest,
)

#: 出す commit で success になっていなければならない workflow（main への push で走る物）
REQUIRED_WORKFLOWS = ("CI", "Package")
#: タグの push で zip を組み立てて下書きへ上げる workflow
RELEASE_WORKFLOW = "release.yml"
#: 目録と署名の資産の名前
MANIFEST_NAME = "update.json"
SIGNATURE_NAME = MANIFEST_NAME + SIGNATURE_SUFFIX
#: ベータの目録を置く、動かすリリース（docs/development.md の「ベータ」）
BETA_TAG = "beta"
#: 自己診断が通った印 zip の sha256 と一緒に書く 別の zip を確かめ済みと取り違えない
CHECKED_NAME = "release-checked.json"
#: 展開した印 これが今の zip の sha256 と同じなら展開し直さない
EXTRACTED_NAME = "extracted-from.txt"

#: Actions の run を見に行く間隔と、諦めるまでの長さ 組み立てと、まっさらな Windows での
#: 確かめで 30 分ほど掛かる 待つのを短くすると、通る run を落ちたと取り違える
POLL_SECONDS = 30.0
WAIT_LIMIT_SECONDS = 2 * 60 * 60
#: タグを push してから run が現れるまで GitHub は数秒〜数十秒遅れる
APPEAR_LIMIT_SECONDS = 10 * 60
#: 公開した後、固定の URL が新しい目録を返すまで CDN が古い物を返すことがある
LATEST_TRIES = 6
LATEST_INTERVAL_SECONDS = 10.0

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


class StopError(Exception):
    """ここで止める 公開はしていない 文言には次に何をするかを書く"""


def run_command(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """git と gh を走らせる 出力は UTF-8 で読む（どちらも UTF-8 で書く）

    渡すのはこの道具の中で組んだ引数だけ shell は通さない
    読めない文字があっても置き換えて続ける cp932 の機械で落ちて、途中で止まらないように
    """
    # 監査済み 引数はこの道具が組んだ物で、外から来た文字列は版の数字と道具の中のパスだけ
    return subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        list(arguments),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def launch_app(executable: Path) -> int:
    """展開したアプリを起こし、閉じられるまで待つ

    環境変数は ``build_package.py`` の確かめと同じく Windows の分だけにする 開発機の ``PATH``
    から DLL を拾って描けてしまうと、使う人の手元で描けない zip を見逃す
    """
    # 監査済み 起こすのはこの道具が展開した exe だけ 引数は無い
    process = (
        subprocess.Popen(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            [str(executable)],
            cwd=executable.parent,
            env=build_package.minimal_environment(os.environ),
        )
    )
    return process.wait()


def smoke_test(archive: Path, home: Path) -> int:
    """``build_package.py`` が組み立てた直後に掛けるのと同じ、zip からの確かめ

    自己診断（GL で描く・書き出す・Lua・自動更新）・置き場のスクリプト・pip・後から入れる部品の
    import まで 使用許諾は展開した物を手本にする（組み立てた時の物はこの機械に無い）
    """
    return build_package.smoke_test(archive, build_package.notice_digests(home))


def fetch_url(url: str) -> bytes:
    """固定の URL から目録を読む 転送は urllib が追う"""
    request = urllib.request.Request(url, headers={"User-Agent": "sashimono-release"})
    # 開く先は links.py の固定の URL だけ 外から来た文字列は混ざらない
    with urllib.request.urlopen(request, timeout=60) as response:
        data: bytes = response.read()
    return data


def ask_line(prompt: str) -> str:
    """y/N を尋ねる 入力が閉じていれば「いいえ」"""
    try:
        return input(prompt)
    except EOFError:
        return ""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_version(root: Path) -> str:
    """``src/sashimono/__init__.py`` の ``__version__`` 版の出どころはここ 1 か所"""
    return _version_in((root / "src" / "sashimono" / "__init__.py").read_text(encoding="utf-8"))


def _version_in(text: str) -> str:
    found = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if found is None:
        raise StopError("__version__ が読めない（src/sashimono/__init__.py）")
    return found.group(1)


def resolve_key(path: Path) -> Path:
    """``--key`` に渡された物から鍵のファイルを決める

    フォルダなら、``update_keys.py generate`` が書く今使う鍵の名前を中から使う
    予備の鍵は差し替えのときだけ使う物なので、黙って選ばない（ファイルで渡してもらう）
    """
    if path.is_dir():
        current = path / KEY_FILES[0][1]
        if not current.is_file():
            raise StopError(
                f"{path} はフォルダで、中に {KEY_FILES[0][1]} が無い 鍵のファイルを直に渡す"
            )
        print(f"--key はフォルダなので、中の今使う鍵を使う: {current}")
        return current
    if not path.is_file():
        raise StopError(
            f"鍵のファイルが無い: {path}（鍵を入れた USB などをつないでから、もう一度）"
        )
    return path


def trusted_index(public: str) -> int | None:
    """公開鍵が ``TRUSTED_PUBLIC_KEYS`` の何番目か（1 から） 無ければ None"""
    for index, text in enumerate(TRUSTED_PUBLIC_KEYS, start=1):
        if text.strip() == public.strip():
            return index
    return None


def signer_index(data: bytes, signature: bytes) -> int | None:
    """署名がどの埋め込んだ鍵で通るか（1 から） どれでも通らなければ None"""
    for index, text in enumerate(TRUSTED_PUBLIC_KEYS, start=1):
        try:
            key = parse_public_key(text)
        except ValueError:
            continue
        if verify_manifest(data, signature, (key,)):
            return index
    return None


def key_label(index: int | None) -> str:
    if index is None:
        return "埋め込んだ鍵のどれでもない"
    names = {1: "今使う鍵", 2: "予備の鍵"}
    return f"TRUSTED_PUBLIC_KEYS の {index} 本目（{names.get(index, '鍵')}）"


def manifest_differences(found: Manifest, expected: Manifest) -> list[str]:
    """2 つの目録で違う項目 ``項目 が 値（今は 値）`` の形 項目は目録の型から数える

    名前を手で並べると、目録に項目を足したときに比べ漏れる
    """
    differences = []
    for item in fields(Manifest):
        have, want = getattr(found, item.name), getattr(expected, item.name)
        if item.name == "package":
            for part in fields(PackageInfo):
                if getattr(have, part.name) != getattr(want, part.name):
                    differences.append(
                        f"package.{part.name} が {getattr(have, part.name)}"
                        f"（今は {getattr(want, part.name)}）"
                    )
        elif have != want:
            differences.append(f"{item.name} が {have}（今は {want}）")
    return differences


def zip_name(version: str) -> str:
    return f"{build_package.ARCHIVE_PREFIX}-{version}-windows-x64.zip"


def _size(value: int) -> str:
    return f"{value / 1_000_000:.1f} MB（{value:,} バイト）"


@dataclass
class Release:
    """1 回のリリースの作業 外とのやり取り（git・gh・人・時計）は差し替えられる"""

    version: str
    key: Path | None = None
    dry_run: bool = False
    skip_launch: bool = False
    minimum: str = "0.0.0"
    root: Path = ROOT
    run: Runner = run_command
    ask: Callable[[str], str] = ask_line
    secret: Callable[[str], str] = getpass.getpass
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    fetch: Callable[[str], bytes] = fetch_url
    smoke: Callable[[Path, Path], int] = smoke_test
    launch: Callable[[Path], int] = launch_app
    collect: Callable[[Path], int] = field(default=lambda dist: collect_sources.main([], dist=dist))
    #: 試しに見た（--dry-run）ときの、済んでいない前提
    problems: list[str] = field(default_factory=list)
    #: 出す commit（タグが指す物 タグが無ければ手元の HEAD）
    commit: str = ""
    release: dict[str, Any] = field(default_factory=dict)
    zip_sha256: str = ""
    signer: int | None = None
    #: 確かめた目録の場所（署名はその隣） ベータの置き場へ上げ直すときに同じ物を使う
    manifest: Path | None = None
    #: 照らす zip の大きさ（下書きなら落とした物、公開済みなら資産の値）
    zip_size: int = -1
    #: 公開まで済んだ 後で落ちても「公開はしていない」と出さない
    published: bool = False
    #: 公開した後に beta へも目録を上げ直すか
    has_beta: bool = False
    #: 公開済みで止まったとき、手で打つ残りを出してよいか
    #: 公開された目録が確かめられないときは出さない（違う目録を beta へ広げない）
    hint_after_stop: bool = True
    #: 公開済みで止まったとき、打ち直せば続きから進めるか
    rerun_helps: bool = True
    #: 今の引数で作り直した目録（:meth:`expected_manifest` が 1 度だけ作る）
    expected: Manifest | None = None

    # --- 小さな道具 ---------------------------------------------------------------

    @property
    def tag(self) -> str:
        return f"v{self.version}"

    @property
    def folder(self) -> Path:
        """落とした物と作った物を置く所 dist は .gitignore 済みで、コミットに混ざらない"""
        return self.root / "dist" / "release"

    @property
    def archive(self) -> Path:
        return self.folder / zip_name(self.version)

    @property
    def extracted(self) -> Path:
        return self.folder / f"{build_package.ARCHIVE_PREFIX}-{self.version}"

    @property
    def beta(self) -> bool:
        return is_prerelease(self.version)

    def call(self, *arguments: str, what: str) -> str:
        """走らせて標準出力を返す 落ちたら止める"""
        result = self.run(list(arguments))
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise StopError(f"{what}に失敗した: {' '.join(arguments[:3])} …\n{detail}")
        return result.stdout

    def problem(self, message: str) -> None:
        """前提が満たされていない 試しに見ているときは数えて続け、本番は止める"""
        if self.dry_run:
            print(f"[NG] {message}")
            self.problems.append(message)
            return
        raise StopError(message)

    def done(self, message: str) -> None:
        print(f"[済] {message}")

    def todo(self, message: str) -> None:
        print(f"[残] {message}")

    # --- 1. 前提 -----------------------------------------------------------------

    def check_prerequisites(self) -> None:
        print("== 1. 前提 ==")
        source_version = read_version(self.root)
        if source_version != self.version:
            self.problem(
                f"引数の版 {self.version} と __version__ {source_version} が違う"
                "（版を上げた PR を main へ入れてから、その版で打つ）"
            )
        if self.run(["gh", "auth", "status"]).returncode != 0:
            # 以降の読み取りも書き込みも gh を通す 使えないまま進むと、何も分からない
            raise StopError("gh が使えない（gh auth login を済ませる）")
        branch = self.call("git", "rev-parse", "--abbrev-ref", "HEAD", what="枝の確かめ").strip()
        if branch != "main":
            self.problem(f"手元が main でない（{branch}） git switch main してから")
        # 追跡していないファイルは見ない タグは commit を指すので、置いてあるだけの物は混ざらない
        dirty = self.call(
            "git", "status", "--porcelain", "--untracked-files=no", what="手元の確かめ"
        ).strip()
        if dirty:
            self.problem("手元に commit していない変更がある（片付けてから）")
        head = self.call("git", "rev-parse", "HEAD", what="手元の確かめ").strip()
        remote = self.remote_refs()
        main = remote.get("refs/heads/main", "")
        if head != main:
            self.problem(
                f"手元の HEAD（{head[:7]}）が origin/main（{main[:7]}）と違う（git pull してから）"
            )
        tag_commit = remote.get(f"refs/tags/{self.tag}^{{}}") or remote.get(f"refs/tags/{self.tag}")
        # タグがあれば、それが出す commit main がその後に進んでいても、組み立てたのはタグの commit
        self.commit = tag_commit or head
        self.check_runs()
        if self.key is not None:
            self.key = resolve_key(self.key)
            try:
                public = key_file_public_key(self.key.read_bytes())
            except (OSError, KeyFileError, ValueError) as exc:
                raise StopError(f"鍵のファイルが読めない: {exc}") from exc
            index = trusted_index(public)
            if index is None:
                # 埋め込んでいない鍵で署名しても、配った版はその署名を信じない
                self.problem(
                    "鍵が TRUSTED_PUBLIC_KEYS に無い（別の鍵を渡していないか、"
                    "tools/update_keys.py show で確かめる）"
                )
            else:
                print(f"鍵: {key_label(index)}")

    def remote_refs(self) -> dict[str, str]:
        """origin の main とタグ 取ってくる（fetch）と手元を書き換えるので、読むだけにする"""
        out = self.call(
            "git",
            "ls-remote",
            "origin",
            "refs/heads/main",
            f"refs/tags/{self.tag}",
            f"refs/tags/{self.tag}^{{}}",
            what="origin の確かめ",
        )
        refs: dict[str, str] = {}
        for line in out.splitlines():
            sha, _, ref = line.partition("\t")
            if ref:
                refs[ref.strip()] = sha.strip()
        return refs

    def check_runs(self) -> None:
        """出す commit の CI と Package が success か 走っている途中なら終わるまで待つ"""
        for name in REQUIRED_WORKFLOWS:
            run = self.wait_run(
                ["--commit", self.commit, "--workflow", name, "--event", "push"],
                f"{name}（{self.commit[:7]}）",
                appear=False,
            )
            if run is None:
                self.problem(
                    f"{self.commit[:7]} の {name} の run が無い（main へ入れた物か確かめる）"
                )
            elif run.get("conclusion") != "success":
                self.problem(
                    f"{self.commit[:7]} の {name} が {run.get('conclusion') or run.get('status')}"
                    f"（{run.get('url', '')} 直して main へ入れ、その commit で出す）"
                )
            else:
                self.done(f"{name} が success（{self.commit[:7]}）")

    def latest_run(self, filters: Sequence[str]) -> dict[str, Any] | None:
        out = self.call(
            "gh",
            "run",
            "list",
            *filters,
            "--limit",
            "20",
            "--json",
            "databaseId,workflowName,status,conclusion,headSha,url,event",
            what="Actions の確かめ",
        )
        runs = [run for run in json.loads(out or "[]") if run.get("headSha") == self.commit]
        # gh は新しい順に返す 走らせ直した物があれば、いちばん新しい結果を信じる
        return runs[0] if runs else None

    def wait_run(
        self, filters: Sequence[str], label: str, *, appear: bool
    ) -> dict[str, Any] | None:
        """run が終わるまで待つ ``appear`` なら、まだ現れていない run も現れるまで待つ"""
        started = self.clock()
        last = ""
        while True:
            run = self.latest_run(filters)
            if run is not None and run.get("status") == "completed":
                return run
            if self.dry_run:
                # 試しに見るときは待たない 今の様子を返す
                return run
            elapsed = self.clock() - started
            if run is None and (not appear or elapsed > APPEAR_LIMIT_SECONDS):
                return None
            if elapsed > WAIT_LIMIT_SECONDS:
                raise StopError(f"{label} が {WAIT_LIMIT_SECONDS // 60:.0f} 分たっても終わらない")
            status = run.get("status", "") if run is not None else "まだ無い"
            if status != last:
                print(f"待つ: {label} {status}")
                last = str(status)
            self.sleep(POLL_SECONDS)

    # --- 2. タグ ------------------------------------------------------------------

    def ensure_tag(self) -> None:
        print(f"== 2. タグ {self.tag} ==")
        remote = self.remote_refs()
        pushed = remote.get(f"refs/tags/{self.tag}^{{}}") or remote.get(f"refs/tags/{self.tag}")
        if pushed:
            self.check_tag_commit(pushed)
            self.done(f"{self.tag} は push 済み（{pushed[:7]}）")
            return
        local = self.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{self.tag}^{{commit}}"])
        head = self.call("git", "rev-parse", "HEAD", what="手元の確かめ").strip()
        if local.returncode == 0 and local.stdout.strip() != head:
            raise StopError(
                f"手元のタグ {self.tag} が HEAD でない commit（{local.stdout.strip()[:7]}）を指す"
                f"（打ち間違いなら git tag -d {self.tag} で消してから）"
            )
        if self.dry_run:
            self.todo(f"{self.tag} を {head[:7]} に打って push する")
            return
        if local.returncode != 0:
            self.call(
                "git",
                "tag",
                "-a",
                self.tag,
                "-m",
                f"Sashimono Edit {self.version}",
                head,
                what="タグを打つの",
            )
        self.call("git", "push", "origin", f"refs/tags/{self.tag}", what="タグの push")
        self.commit = head
        self.done(f"{self.tag} を {head[:7]} に打って push した")

    def check_tag_commit(self, commit: str) -> None:
        """push 済みのタグが、出してよい commit を指しているか

        main の先頭か、main の歴史の中にあって ``__version__`` が引数の版の commit
        タグの後に main へ文書の直しなどが入るのは普通なので、先頭だけには絞らない
        それ以外（main に無い commit・版の違う commit）は打ち間違い 組み立てた zip が
        出したい物と違う
        """
        head = self.call("git", "rev-parse", "HEAD", what="手元の確かめ").strip()
        if commit == head:
            return
        if self.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"]).returncode != 0:
            raise StopError(
                f"タグ {self.tag} が main に無い commit（{commit[:7]}）を指す"
                "（打ち間違い 公開していないなら、下書きとタグを消して打ち直す"
                " 消せないなら版を上げる）"
            )
        shown = self.run(["git", "show", f"{commit}:src/sashimono/__init__.py"])
        if shown.returncode != 0 or _version_in(shown.stdout) != self.version:
            raise StopError(
                f"タグ {self.tag} の commit（{commit[:7]}）の __version__ が {self.version} でない"
            )
        print(f"タグは手元の HEAD より前の {commit[:7]} を指す（組み立てたのはこの commit）")

    # --- 3. 組み立て ---------------------------------------------------------------

    def wait_release_workflow(self) -> None:
        print("== 3. Actions の Release ==")
        label = f"Release（{self.tag}）"
        run = self.wait_run(
            ["--workflow", RELEASE_WORKFLOW, "--branch", self.tag], label, appear=True
        )
        if run is None:
            if self.dry_run:
                self.todo(f"{label} の終わりを待つ（まだ run が無い）")
                return
            raise StopError(
                f"{label} の run が見つからない（Actions の画面で、タグの push で走ったかを見る"
                f" 走っていなければ gh workflow run {RELEASE_WORKFLOW} --ref {self.tag}）"
            )
        if run.get("status") != "completed":
            self.todo(f"{label} が {run.get('status')} 終わるのを待つ")
            return
        if run.get("conclusion") != "success":
            raise StopError(
                f"{label} が {run.get('conclusion')}（{run.get('url', '')}）"
                " 落ちた所を直す zip が組めない・まっさらな Windows で起きないなら"
                "、直した commit で版を上げて打ち直す"
            )
        self.done(f"{label} が success")

    # --- 4. 下書き -----------------------------------------------------------------

    def load_release(self) -> dict[str, Any] | None:
        """リリースの様子 まだ無ければ None"""
        result = self.run(
            ["gh", "release", "view", self.tag, "--json", "isDraft,isPrerelease,assets,url"]
        )
        if result.returncode != 0:
            return None
        data: dict[str, Any] = json.loads(result.stdout)
        self.release = data
        return data

    def assets(self) -> dict[str, dict[str, Any]]:
        return {str(asset["name"]): asset for asset in self.release.get("assets", [])}

    def check_draft(self) -> bool:
        """下書きを確かめて zip を手元にそろえる 試しに見ていて先へ進めないときは False"""
        print("== 4. 下書き ==")
        release = self.load_release()
        if release is None:
            if self.dry_run:
                self.todo(f"下書き {self.tag} はまだ無い（3 の Release が作る）")
                return False
            raise StopError(
                f"リリース {self.tag} が無い（Release の workflow が下書きを作る 3 のログを見る）"
            )
        if not release.get("isDraft"):
            # 通しでは公開済みを先に resume_published へ回す ここへ来るのは、見た直後に誰かが
            # 公開したときだけ 何を公開したかが分からないので、手で打つ残りは出さない
            self.published = True
            self.hint_after_stop = False
            raise StopError(
                f"{self.tag} は公開済み 公開した後の zip や目録は差し替えない"
                "（直すなら版を上げて新しいタグで出す）"
            )
        if release.get("isPrerelease") != self.beta:
            raise StopError(
                f"{self.tag} のプレリリースの印が版と合わない（{release.get('isPrerelease')}）"
                " gh release edit で直してから"
            )
        asset = self.assets().get(zip_name(self.version))
        if asset is None:
            raise StopError(
                f"下書きに {zip_name(self.version)} が無い（Release の run のログを見る）"
            )
        self.done(f"{self.tag} は下書き 資産 {len(self.assets())} 個")
        if self.matches(self.archive, asset):
            self.done(f"zip は落としてあり、GitHub の物と同じ: {self.archive}")
        elif self.dry_run:
            self.todo(f"zip を落とす（{_size(int(asset['size']))}）: {self.archive}")
            return False
        else:
            self.folder.mkdir(parents=True, exist_ok=True)
            print(f"落とす: {zip_name(self.version)}（{_size(int(asset['size']))}）")
            self.call(
                "gh",
                "release",
                "download",
                self.tag,
                "--pattern",
                zip_name(self.version),
                "--dir",
                str(self.folder),
                "--clobber",
                what="zip を落とすの",
            )
            if not self.matches(self.archive, asset):
                raise StopError(
                    "落とした zip が GitHub の digest・大きさと合わない（もう一度打つ）"
                )
            self.done(f"zip を落とし、GitHub の物と照らした: {self.archive}")
        self.zip_sha256 = sha256_of(self.archive)
        self.zip_size = self.archive.stat().st_size
        self.extract()
        return True

    def matches(self, path: Path, asset: dict[str, Any]) -> bool:
        """手元の物が資産と同じか digest があれば sha256 で、無ければ大きさで"""
        if not path.is_file() or path.stat().st_size != int(asset.get("size", -1)):
            return False
        digest = str(asset.get("digest") or "")
        if digest.startswith("sha256:"):
            return sha256_of(path) == digest.removeprefix("sha256:")
        return True

    def extract(self) -> None:
        """自己診断とソースを数えるために展開する 同じ zip から展開済みなら飛ばす"""
        marker = self.extracted / EXTRACTED_NAME
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == self.zip_sha256:
            return
        if self.dry_run:
            self.todo(f"zip を展開する: {self.extracted}")
            return
        # 前の zip から展開した物の上に重ねない 消えたファイルが残って、確かめが通ってしまう
        shutil.rmtree(self.extracted, ignore_errors=True)
        with zipfile.ZipFile(self.archive) as opened:
            opened.extractall(self.extracted)
        marker.write_text(self.zip_sha256 + "\n", encoding="utf-8")
        print(f"展開した: {self.extracted}")

    @property
    def home(self) -> Path:
        return self.extracted / build_package.APP_NAME

    # --- 5. ソース -----------------------------------------------------------------

    def ensure_sources(self) -> None:
        print("== 5. ソース ==")
        missing = self.missing_sources()
        if not missing:
            self.done(f"下書きに索引（{collect_sources.MANIFEST_NAME}）の載せたソースが全部ある")
            return
        if self.dry_run:
            self.todo(
                f"ソースを集めて下書きへ上げる（足りない物: {'、'.join(missing)}）"
                "（collect_sources.py 落とすのは足りない分だけ 全部なら 100 MB ほど）"
            )
            return
        print(f"足りない物: {'、'.join(missing)}")
        if self.collect(self.extracted) != 0:
            raise StopError("ソースを集められなかった（上の [NG] を直してから、もう一度打つ）")
        folder = self.extracted / "sources"
        indexes = (collect_sources.MANIFEST_NAME, collect_sources.SUMS_NAME)
        assets = self.assets()
        bodies = [
            path
            for path in sorted(folder.iterdir())
            if path.is_file()
            and path.name not in indexes
            and not self.matches(path, assets.get(path.name, {}))
        ]
        # 本体を上げ終えてから索引を上げる 1 回で並べて上げると、途中で落ちたときに索引だけが
        # 上がり、次に打ったときに揃ったと見て、GPL のソースが欠けたまま公開まで進む
        if bodies:
            self.call(
                "gh",
                "release",
                "upload",
                self.tag,
                *(str(path) for path in bodies),
                "--clobber",
                what="ソースを上げるの",
            )
        self.call(
            "gh",
            "release",
            "upload",
            self.tag,
            *(str(folder / name) for name in indexes),
            "--clobber",
            what="ソースの索引を上げるの",
        )
        self.load_release()
        left = self.missing_sources()
        if left:
            raise StopError(f"上げた後もソースが欠けている: {'、'.join(left)}（もう一度打つ）")
        self.done(f"ソース {len(bodies)} 個と索引を上げた")

    def missing_sources(self) -> list[str]:
        """下書きに無い・中身の違うソース 索引に載った物を、名前と大きさと sha256 で照らす

        索引が 2 つあるだけでは揃ったと見ない 前に上げる途中で落ちていると、索引だけが
        あって本体が欠けている
        """
        assets = self.assets()
        indexes = (collect_sources.MANIFEST_NAME, collect_sources.SUMS_NAME)
        missing = [name for name in indexes if name not in assets]
        if collect_sources.MANIFEST_NAME in missing:
            return missing
        # 標準出力へ落とす 読むだけなので手元にファイルを作らない（--dry-run でも使える）
        out = self.call(
            "gh",
            "release",
            "download",
            self.tag,
            "--pattern",
            collect_sources.MANIFEST_NAME,
            "--output",
            "-",
            what="ソースの索引を読むの",
        )
        try:
            entries = json.loads(out)
            listed = [(str(e["file"]), int(e["size"]), str(e["sha256"])) for e in entries]
        except (ValueError, TypeError, KeyError) as exc:
            return [f"{collect_sources.MANIFEST_NAME}（読めない: {exc}）"]
        if not listed:
            return [f"{collect_sources.MANIFEST_NAME}（空）"]
        for name, size, sha256 in listed:
            asset = assets.get(name)
            digest = str((asset or {}).get("digest") or "")
            if (
                asset is None
                or int(asset.get("size", -1)) != size
                or (digest.startswith("sha256:") and digest.removeprefix("sha256:") != sha256)
            ):
                missing.append(name)
        return missing

    # --- 6. 自己診断 ----------------------------------------------------------------

    def self_check(self) -> None:
        print("== 6. 自己診断 ==")
        marker = self.folder / CHECKED_NAME
        if self.checked(marker):
            self.done("この zip は確かめ済み（自己診断と GL）")
            return
        if self.dry_run:
            self.todo("展開した zip で自己診断を走らせ、アプリを起こして GL で描けるかを見る")
            return
        if self.smoke(self.archive, self.home) != 0:
            raise StopError(
                "zip からの確かめが通らない（上の [NG] を見る この zip は出せない"
                " 直して版を上げるか、下書きとタグを消して打ち直す）"
            )
        gl = "飛ばした"
        if not self.skip_launch:
            executable = self.home / APP_EXE
            print(f"アプリを起こす: {executable}")
            print("プレビューに絵が出るか（GL で描けているか）を見て、見終えたらアプリを閉じる")
            self.launch(executable)
            answer = self.ask("GL で描けていましたか y/N: ").strip().lower()
            if answer not in {"y", "yes"}:
                raise StopError(
                    "GL で描けていない（--self-check の GL の行と、プレビューの所の理由を見る）"
                )
            gl = "見た"
        marker.write_text(
            json.dumps({"zip_sha256": self.zip_sha256, "gl": gl}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.done(f"自己診断が通った（GL は{gl}）")

    def checked(self, marker: Path) -> bool:
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict) or data.get("zip_sha256") != self.zip_sha256:
            return False
        # 前は起こさずに通した印なら、起こして見るよう言われたときに見直す
        return bool(self.skip_launch or data.get("gl") == "見た")

    # --- 7. 目録と署名 ---------------------------------------------------------------

    def ensure_signature(self) -> None:
        print("== 7. 目録と署名 ==")
        names = self.assets()
        local = self.folder / MANIFEST_NAME
        if MANIFEST_NAME in names and SIGNATURE_NAME in names:
            if self.dry_run:
                self.done("下書きに目録と署名がある（本番では落として確かめる）")
                return
            remote = self.folder / "remote"
            remote.mkdir(parents=True, exist_ok=True)
            for name in (MANIFEST_NAME, SIGNATURE_NAME):
                self.call(
                    "gh",
                    "release",
                    "download",
                    self.tag,
                    "--pattern",
                    name,
                    "--dir",
                    str(remote),
                    "--clobber",
                    what="目録を落とすの",
                )
            problem = self.manifest_problem(remote / MANIFEST_NAME)
            if problem:
                # 道具が消して作り直すことはしない --minimum の打ち間違いのこともあり、消すと
                # 戻せない（もう一度署名するしかない） どちらが正しいかは人が決める
                local_copies = " ".join(
                    str(path) for path in (local, local.with_name(SIGNATURE_NAME)) if path.exists()
                )
                tidy = f" と手元の写し（{local_copies}）" if local_copies else ""
                raise StopError(
                    f"下書きの目録が通らない: {problem}\n  引数が正しいなら、下書きの "
                    f"{MANIFEST_NAME} と {SIGNATURE_NAME}{tidy}を消して、もう一度打つ\n"
                    f"  gh release delete-asset {self.tag} {MANIFEST_NAME} -y\n"
                    f"  gh release delete-asset {self.tag} {SIGNATURE_NAME} -y\n"
                    "  引数を間違えたなら、前と同じ引数で打ち直す"
                )
            self.done(f"下書きの目録と署名が通った（{key_label(self.signer)}）")
            return
        if self.manifest_problem(local) is None:
            # 前に署名したが上げる所で落ちた 合言葉を尋ね直さずに上げる
            self.done(f"手元に署名済みの目録がある（{key_label(self.signer)}）")
        elif self.dry_run:
            self.todo("目録を作り、合言葉を尋ねて署名し、確かめて下書きへ上げる")
            return
        else:
            self.sign(local)
        if self.dry_run:
            self.todo("手元の目録と署名を下書きへ上げる")
            return
        self.call(
            "gh",
            "release",
            "upload",
            self.tag,
            str(local),
            str(local.with_name(SIGNATURE_NAME)),
            "--clobber",
            what="目録と署名を上げるの",
        )
        self.load_release()
        self.done("目録と署名を下書きへ上げた")

    def sign(self, local: Path) -> None:
        if self.key is None:
            raise StopError("署名する鍵が要る（--key に鍵のファイルを渡す）")
        local.write_bytes(update_sign.build_manifest(self.archive, minimum=self.minimum))
        local.with_name(SIGNATURE_NAME).unlink(missing_ok=True)
        print(f"目録を作った: {local}")
        try:
            _, public = update_sign.sign_file(local, self.key, self.secret)
        except update_sign.ReleaseError as exc:
            raise StopError(f"署名できない: {exc}（合言葉の打ち間違いなら、もう一度打つ）") from exc
        if trusted_index(public) is None:
            local.with_name(SIGNATURE_NAME).unlink(missing_ok=True)
            raise StopError(
                "この鍵は埋め込まれていない 配った版はこの署名を信じない（鍵を確かめる）"
            )
        problem = self.manifest_problem(local)
        if problem:
            raise StopError(f"署名した目録が通らない: {problem}")
        self.done(f"署名して確かめた（{key_label(self.signer)}）")

    def expected_manifest(self) -> Manifest | None:
        """手元の zip と今の引数で作り直した目録 zip が手元に無い・照らす zip と違えば None

        署名する物と同じ ``update_sign.build_manifest`` で作る 比べる側だけ別に組むと、
        目録に項目を足したときに比べ漏れる
        """
        if self.expected is not None:
            return self.expected
        if not self.archive.is_file() or self.archive.stat().st_size != self.zip_size:
            return None
        if sha256_of(self.archive) != self.zip_sha256:
            return None
        built = update_sign.build_manifest(self.archive, minimum=self.minimum)
        self.expected = parse_manifest(built)
        return self.expected

    def manifest_problem(self, manifest: Path) -> str | None:
        """目録と署名が、埋め込んだ鍵で通り、この版とこの zip を指しているか 通れば None"""
        signature = manifest.with_name(manifest.name + SIGNATURE_SUFFIX)
        if not self.zip_sha256:
            return "zip がまだ手元に無い"
        if not manifest.is_file() or not signature.is_file():
            return "目録か署名が無い"
        try:
            version = update_sign.verify_file(manifest, signature)
            parsed = parse_manifest(manifest.read_bytes())
        except (OSError, update_sign.ReleaseError, ManifestError) as exc:
            return str(exc)
        if version != self.version:
            return f"目録の版が {version}"
        if parsed.package.sha256 != self.zip_sha256 or parsed.package.size != self.zip_size:
            return "目録の zip の sha256 か大きさが、下書きの zip と違う"
        # 今の引数（--minimum など）で作り直した目録と、全部の項目を比べる 版と zip だけを
        # 見ると、--minimum を変えて打ち直したときに、古い minimum で署名した目録を通してしまう
        expected = self.expected_manifest()
        if expected is None:
            return "zip が手元に無く、目録を作り直して比べられない"
        differences = manifest_differences(parsed, expected)
        if differences:
            return "今の引数で作り直した目録と違う（" + "、".join(differences) + "）"
        self.signer = signer_index(manifest.read_bytes(), signature.read_bytes())
        self.manifest = manifest
        return None

    # --- 8. 公開 -------------------------------------------------------------------

    def publish(self) -> int:
        print("== 8. 公開 ==")
        if self.beta and self.run(["gh", "release", "view", BETA_TAG]).returncode != 0:
            raise StopError(
                f"ベータの目録を置くリリース {BETA_TAG} が無い（docs/development.md の「ベータ」"
                " プレリリースで 1 つ作ってから）"
            )
        has_beta = self.beta or self.run(["gh", "release", "view", BETA_TAG]).returncode == 0
        size = self.archive.stat().st_size
        print()
        print(f"  版        {self.version}{'（ベータ）' if self.beta else ''}")
        print(f"  タグ      {self.tag}（{self.commit[:7]}）")
        print(f"  zip       {zip_name(self.version)} {_size(size)}")
        print(f"  sha256    {self.zip_sha256}")
        print(f"  資産      {len(self.assets())} 個")
        print(f"  署名      {key_label(self.signer)}")
        print(f"  公開先    {self.release.get('url', '')}")
        if self.beta:
            print(f"  ベータ    目録と署名を {BETA_TAG} へ上げ直す（{BETA_MANIFEST_URL}）")
        else:
            print(f"  固定の URL {STABLE_MANIFEST_URL} がこの版を指すようになる")
            if has_beta:
                print(f"  ベータ    正式版の目録を {BETA_TAG} にも上げ直す")
        print()
        answer = self.ask("公開しますか y/N: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("公開しなかった 下書きのまま（もう一度打てば、ここから尋ねる）")
            return 0
        self.has_beta = has_beta
        flag = "--prerelease" if self.beta else "--latest"
        try:
            self.call("gh", "release", "edit", self.tag, "--draft=false", flag, what="公開")
        except StopError:
            # 通信が切れただけで、GitHub の側では公開まで済んでいることがある 見直さずに
            # 「公開はしていない」と出すと、公開された版の残り（beta の目録）が放っておかれる
            after = self.load_release()
            if after is not None and not after.get("isDraft"):
                self.published = True
            raise
        self.published = True
        print(f"公開した: {self.tag}")
        return self.after_publish()

    def after_publish(self) -> int:
        """公開した後の残り beta の目録を上げ直し、固定の URL がこの版を指すかを見る

        ここで落ちても公開は済んでいる 呼び手は :meth:`after_publish_hint` で残りを出す
        """
        manifest = self.manifest
        if self.has_beta and manifest is not None:
            self.call(
                "gh",
                "release",
                "upload",
                BETA_TAG,
                str(manifest),
                str(manifest.with_name(SIGNATURE_NAME)),
                "--clobber",
                what=f"{BETA_TAG} へ目録を上げるの",
            )
            print(f"{BETA_TAG} へ目録と署名を上げた")
        return self.check_latest(BETA_MANIFEST_URL if self.beta else STABLE_MANIFEST_URL)

    def after_publish_hint(self) -> list[str]:
        """公開した後で止まったときに、手で済ませる残り"""
        lines = []
        manifest = self.manifest or self.folder / MANIFEST_NAME
        if self.has_beta:
            lines.append(
                f"gh release upload {BETA_TAG} {manifest} "
                f"{manifest.with_name(SIGNATURE_NAME)} --clobber"
            )
        url = BETA_MANIFEST_URL if self.beta else STABLE_MANIFEST_URL
        lines.append(f"{url} を開き、version が {self.version} かを見る")
        return lines

    def resume_published(self) -> int:
        """公開済みのリリースに打ち直したとき、公開した後の残りだけを続ける

        zip や目録は差し替えない 続けるのは、公開されている目録と署名が、下書きのときに
        手元で確かめた物とバイト列まで同じときだけ 違えば、公開した後に誰かが差し替えた
        """
        print("== 4. 公開済み ==")
        # 公開済みと分かった時点で記録する この後どこで止まっても「公開はしていない」と出さない
        self.published = True
        assets = self.assets()
        asset = assets.get(zip_name(self.version))
        stop = (
            f"{self.tag} は公開済み 公開した後の zip や目録は差し替えない"
            "（直すなら版を上げて新しいタグで出す）"
        )
        if asset is None or MANIFEST_NAME not in assets or SIGNATURE_NAME not in assets:
            # 署名した目録の無い公開 beta へ上げる手順を出しても、上げる物が無い
            self.hint_after_stop = False
            raise StopError(stop + " 公開されているリリースに zip か目録か署名が無い")
        digest = str(asset.get("digest") or "")
        if not digest.startswith("sha256:"):
            self.hint_after_stop = False
            raise StopError(stop + " zip の digest が無く、目録と照らせない")
        self.zip_sha256 = digest.removeprefix("sha256:")
        self.zip_size = int(asset.get("size", -1))
        self.has_beta = self.beta or self.run(["gh", "release", "view", BETA_TAG]).returncode == 0
        if self.dry_run:
            self.done(f"{self.tag} は公開済み")
            self.todo("公開した後の残り（beta の目録・固定の URL の確かめ）を続けられるかを見る")
            return 1 if self.problems else 0
        # 下書きのときに手元で確かめた写し（落とした物か、署名した物）
        # 目録を今の引数で作り直して比べるのに、公開された zip そのものが要る
        if not self.matches(self.archive, asset):
            self.folder.mkdir(parents=True, exist_ok=True)
            self.call(
                "gh",
                "release",
                "download",
                self.tag,
                "--pattern",
                zip_name(self.version),
                "--dir",
                str(self.folder),
                "--clobber",
                what="zip を落とすの",
            )
            if not self.matches(self.archive, asset):
                raise StopError(
                    "落とした zip が GitHub の digest・大きさと合わない（もう一度打つ）"
                )
        known: list[Path] = []
        reasons: list[str] = []
        for path in (self.folder / "remote" / MANIFEST_NAME, self.folder / MANIFEST_NAME):
            if not path.is_file():
                continue
            problem = self.manifest_problem(path)
            if problem is None:
                known.append(path)
            else:
                reasons.append(f"{path}: {problem}")
        if not known and reasons:
            # 写しはあるが今の引数と合わない --minimum を変えて打ち直したなど 公開した物は
            # 前の引数で作ってあるので、前と同じ引数で打ち直せば続けられる
            self.hint_after_stop = False
            raise StopError(
                stop
                + " 手元の写しが今の引数と合わない（前と同じ引数で打ち直す）\n  "
                + "\n  ".join(reasons)
            )
        if not known:
            # 打ち直しても同じ所で止まる 手で済ませる残りは main が並べる
            self.rerun_helps = False
            raise StopError(stop + " 下書きのときに確かめた目録が手元に無い")
        published = self.folder / "published"
        published.mkdir(parents=True, exist_ok=True)
        for name in (MANIFEST_NAME, SIGNATURE_NAME):
            self.call(
                "gh",
                "release",
                "download",
                self.tag,
                "--pattern",
                name,
                "--dir",
                str(published),
                "--clobber",
                what="公開した目録を落とすの",
            )
        problem = self.manifest_problem(published / MANIFEST_NAME)
        same = any(
            (published / MANIFEST_NAME).read_bytes() == path.read_bytes()
            and (published / SIGNATURE_NAME).read_bytes()
            == path.with_name(SIGNATURE_NAME).read_bytes()
            for path in known
        )
        if problem or not same:
            # 違う目録を beta へ広げる手順は出さない 先に誰が差し替えたかを確かめる
            self.hint_after_stop = False
            raise StopError(
                f"公開されている目録が、下書きのときに確かめた物と違う（{problem or '中身が違う'}）"
                " 誰が差し替えたかを確かめる（直すなら版を上げて新しいタグで出す）"
            )
        self.done(
            f"{self.tag} は公開済み 目録と署名は下書きのときと同じ（{key_label(self.signer)}）"
        )
        rest = f"{BETA_TAG} へ目録を上げ直し、" if self.has_beta else ""
        answer = self.ask(f"公開した後の残り（{rest}固定の URL の確かめ）を続けますか y/N: ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("続けなかった")
            return 0
        return self.after_publish()

    def check_latest(self, url: str) -> int:
        """公開した後、配った版が読む固定の URL がこの版を指すか"""
        seen = ""
        for attempt in range(LATEST_TRIES):
            if attempt:
                self.sleep(LATEST_INTERVAL_SECONDS)
            try:
                seen = parse_manifest(self.fetch(url)).version
            except (OSError, ManifestError) as exc:
                seen = f"読めない（{exc}）"
                continue
            if seen == self.version:
                print(f"[ok] {url} が {self.version} を指す")
                return 0
        print(
            f"[NG] {url} が {seen} のまま（CDN の遅れなら数分後に開いて確かめる"
            " 変わらなければ、リリースが Latest になっているかを GitHub の画面で見る）"
        )
        return 1

    # --- 通し -----------------------------------------------------------------------

    def run_all(self) -> int:
        if self.dry_run:
            print("--dry-run 何も書き換えない（タグ・上げる・署名・公開はしない）")
        self.check_prerequisites()
        if self.problems:
            # 前提が欠けたままタグを打つ話はしない 残りは今の GitHub の様子だけ見る
            print("（前提が欠けているので、以下は今の様子だけ）")
        self.ensure_tag()
        self.wait_release_workflow()
        current = self.load_release()
        if current is not None and not current.get("isDraft"):
            # 公開済み 差し替えはしない 公開した後の残りだけを、確かめた上で続けられる
            return self.resume_published()
        if not self.check_draft() and not self.release:
            # 試しに見ていて、下書きがまだ無い 残りは全部これから
            self.todo("5 から先（ソース・自己診断・目録と署名・公開）は下書きができてから")
            return 1 if self.problems else 0
        self.ensure_sources()
        self.self_check()
        self.ensure_signature()
        if self.dry_run:
            self.todo("要約を出して「公開しますか y/N」と尋ねる")
            return 1 if self.problems else 0
        return self.publish()


def main(argv: list[str] | None = None, **overrides: Any) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="出す版（__version__ と同じ 例 0.1.0）")
    parser.add_argument("--key", type=Path, help="署名する鍵のファイルか、それを入れたフォルダ")
    parser.add_argument(
        "--minimum", default="0.0.0", help="これより古い版は自動では上げられない（目録の minimum）"
    )
    parser.add_argument("--dry-run", action="store_true", help="何も変えずに、済んだ段と残りを出す")
    parser.add_argument(
        "--skip-launch", action="store_true", help="自己診断の後にアプリを起こして見る所を飛ばす"
    )
    args = parser.parse_args(argv)

    release = Release(
        version=args.version,
        key=args.key,
        dry_run=args.dry_run,
        skip_launch=args.skip_launch,
        minimum=args.minimum,
        **overrides,
    )
    try:
        return release.run_all()
    except StopError as exc:
        print(f"[止めた] {exc}")
    except (OSError, ValueError, update_sign.ReleaseError) as exc:
        print(f"[止めた] {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("[止めた] 中断した")
    if release.published:
        # 公開した後で落ちた 「公開はしていない」と出すと、事実と違ううえに、打ち直しても
        # 公開済みで止まると思って残り（beta の目録）を放っておかれる
        if not release.hint_after_stop:
            print(f"{release.tag} の公開は済んでいる 上の理由を確かめるまで、残りは進めない")
            return 1
        lead = "同じコマンドを打ち直すか、手で次を行う" if release.rerun_helps else "手で次を行う"
        print(f"{release.tag} の公開は済んでいる 残りは{lead}")
        for line in release.after_publish_hint():
            print(f"  {line}")
        return 1
    print("公開はしていない 直してから同じコマンドを打てば、済んだ段は飛ばして続きから進む")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
