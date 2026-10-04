"""リリースと配る zip の workflow（.github/workflows/release.yml・package.yml）

組み立ては PyPI から依存を入れて走らせる 依存のどれかが悪さをしても、リリースを書き換え
られるトークンに届かないようにする（書ける権限は、依存を入れないジョブだけに渡す）

zip の確かめ（Issue #33）は、依存を入れた機械とは別の機械で、zip だけを持って行う
同じ機械で確かめると、その機械の Python や DLL を拾って通ってしまう
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
PACKAGE = (WORKFLOWS / "package.yml").read_text(encoding="utf-8")
CHECK_SCRIPT = ROOT / "tools" / "check_clean_machine.ps1"


def _jobs(workflow: str) -> dict[str, str]:
    """ジョブの名前 → そのジョブの本文（字面で切る 字下げ 2 つのキーがジョブの頭）"""
    body = workflow.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _all_jobs() -> dict[str, str]:
    return {
        **{f"release/{name}": text for name, text in _jobs(RELEASE).items()},
        **{f"package/{name}": text for name, text in _jobs(PACKAGE).items()},
    }


def test_nothing_is_granted_by_default() -> None:
    """既定で書く権限を持たせると、権限を書き忘れたジョブがそれを受け継ぐ"""
    for workflow in (RELEASE, PACKAGE):
        assert re.search(r"^permissions: \{\}$", workflow, flags=re.M)


def test_installing_jobs_cannot_write() -> None:
    jobs = _all_jobs()
    installing = [name for name, text in jobs.items() if "pip install" in text]
    assert installing
    for name in installing:
        assert "contents: write" not in jobs[name], name


def test_writing_jobs_install_nothing() -> None:
    jobs = _all_jobs()
    writing = [name for name, text in jobs.items() if "contents: write" in text]
    assert writing
    for name in writing:
        text = jobs[name]
        assert "pip install" not in text and "actions/checkout" not in text, name


def test_the_called_build_can_only_read() -> None:
    """呼んだ先（package.yml）の権限は、呼ぶ側のジョブが渡した物が上限になる

    release.yml の package に書く権限を渡すと、依存を入れて走らせる組み立てまで書ける
    """
    package = _jobs(RELEASE)["package"]
    assert "uses: ./.github/workflows/package.yml" in package
    assert "contents: read" in package
    assert "contents: write" not in package


def test_a_release_waits_for_the_clean_machine() -> None:
    """まっさらな Windows で zip から起動できなければ、下書きへ上げない

    release.yml は package.yml を呼び、publish はその終わりを待つ 呼んだ先のジョブが
    1 つでも落ちれば package が落ち、publish は走らない clean-machine を build の後に
    置かないと、確かめる前に上げることになる
    """
    assert re.search(r"^    needs: package$", _jobs(RELEASE)["publish"], flags=re.M)
    assert "workflow_call:" in PACKAGE
    assert re.search(r"^    needs: build$", _jobs(PACKAGE)["clean-machine"], flags=re.M)


def test_the_clean_machine_brings_no_developer_tools() -> None:
    """確かめる機械には Python も uv も依存も入れない 渡すのは zip だけ

    入れると、zip に積み忘れた物をそちらから拾って通ってしまう
    """
    clean = _jobs(PACKAGE)["clean-machine"]
    for tool in ("setup-python", "setup-uv", "pip install", "python "):
        assert tool not in clean, tool
    assert "tools/check_clean_machine.ps1" in clean
    assert "download-artifact" in clean


def test_the_tag_and_version_are_compared_before_a_release() -> None:
    """タグから呼ばれたときは、組み立てる前に版とタグを照らす（F-12-8）"""
    build = _jobs(PACKAGE)["build"]
    check = build.index("check-tag")
    assert "startsWith(github.ref, 'refs/tags/')" in build[:check]
    assert check < build.index("build_package.py --skip-check")


def test_main_and_packaging_changes_run_it() -> None:
    """main への push と、組み立てと確かめに関わるファイルを変える PR で走る

    ほかの PR でも走らせると、組み立てだけで 10 分ほど待たされる
    """
    head = PACKAGE.split("\njobs:\n", 1)[0]
    assert re.search(r"^    branches: \[main\]$", head, flags=re.M)
    assert "workflow_dispatch:" in head
    paths = re.findall(r'^      - "([^"]+)"$', head, flags=re.M)
    for path in (
        "tools/build_package.py",
        "tools/check_clean_machine.ps1",
        "src/sashimono/selfcheck.py",
        ".github/workflows/package.yml",
    ):
        assert path in paths, path


def test_the_check_script_has_a_bom() -> None:
    """BOM が無いと Windows PowerShell 5.1 は Shift_JIS として読み、日本語が全部化ける

    CI は pwsh 7 で走らせるが、手で 5.1 から開いた人に、走らない理由を読めるようにする
    """
    assert CHECK_SCRIPT.read_bytes().startswith(b"\xef\xbb\xbf")


def test_the_check_script_knows_the_self_check_names() -> None:
    """確かめる道具は自己診断の項目を名前で見る 名前を変えたら道具も直さないと、

    GL の無い機械で落ちてよい項目を見分けられず、要る項目が無いことにも気付けない
    """
    from sashimono.selfcheck import (
        ENCODE_CHECK_NAME,
        EXPORT_CHECK_NAME,
        RENDER_CHECK_NAME,
        UPDATE_CHECK_NAME,
        VC_RUNTIME_CHECK_NAME,
    )

    script = CHECK_SCRIPT.read_text(encoding="utf-8-sig")
    for name in (
        ENCODE_CHECK_NAME,
        EXPORT_CHECK_NAME,
        RENDER_CHECK_NAME,
        UPDATE_CHECK_NAME,
        VC_RUNTIME_CHECK_NAME,
    ):
        assert f"'{name}'" in script, name


def test_a_published_or_signed_release_is_not_replaced() -> None:
    """同じタグを走らせ直して、公開済み・署名済みの zip を差し替えない

    組み立て直した zip は SHA-256 が変わり、署名した update.json と食い違って、全員の
    自動更新が照合で止まる 上げる（``gh release upload``）より前で止める
    """
    publish = _jobs(RELEASE)["publish"]
    upload = publish.index("gh release upload")
    draft = publish.find("isDraft")
    signed = publish.find('"update.json"')
    assert 0 <= draft < upload
    assert 0 <= signed < upload
    assert '.name == "update.json.sig"' in publish
    # どちらも止める（exit 1）
    guard = publish[draft:upload]
    assert guard.count("exit 1") >= 2


def test_checkout_leaves_no_token_behind() -> None:
    """checkout は既定でトークンを .git/config に残し、後で走る依存のコードが読める"""
    for name, text in _all_jobs().items():
        for step in text.split("- uses: ")[1:]:
            if step.startswith("actions/checkout"):
                assert "persist-credentials: false" in step.split("\n      - ")[0], name
