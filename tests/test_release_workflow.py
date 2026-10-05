"""リリースと配る zip の workflow（.github/workflows/release.yml・package.yml）

組み立ては PyPI から依存を入れて走らせる 依存のどれかが悪さをしても、リリースを書き換え
られるトークンに届かないようにする（書ける権限は、依存を入れないジョブだけに渡す）

zip の確かめ（Issue #33）は、依存を入れた機械とは別の機械で、zip だけを持って行う
同じ機械で確かめると、その機械の Python や DLL を拾って通ってしまう
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

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


def test_every_install_is_pinned_from_the_first() -> None:
    """依存を入れる最初の導入から、一覧の版に留める制約を使う

    制約の無い導入を先に通すと、その日の新しい依存が壊れていたときに制約付きの
    導入まで届かずに止まる 制約は依存を入れる前に、標準ライブラリだけの道具で書く
    """
    build = _jobs(PACKAGE)["build"]
    installs = [m.start() for m in re.finditer(r"uv pip install", build)]
    assert installs
    written = build.index("tools/package_notices.py --write-constraints build/constraints.txt")
    assert written < installs[0]
    for start in installs:
        line = build[start : build.index("\n", start)]
        assert "-c build/constraints.txt" in line, line


def test_a_release_is_verified_before_the_zip_is_built() -> None:
    """タグのとき（release.yml から呼ばれたとき）は、組み立てる前に verify を走らせる

    タグの push では ci.yml が走らないので、ruff・mypy・pytest が落ちる状態のまま下書きが
    作られうる（PR #241 のレビュー） main と PR では ci.yml が走らせるので、タグに絞って
    二重にしない
    """
    build = _jobs(PACKAGE)["build"]
    verify = build.index("python tools/verify.py")
    step = build[build.rindex("- name:", 0, verify) : verify]
    assert RELEASE_OR_MANUAL in step
    assert build.index("uv pip install") < verify < build.index("build_package.py --skip-check")
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "python tools/verify.py" in ci
    assert "tags:" not in ci.split("\njobs:\n", 1)[0]


#: package.yml の検証を走らせる条件 タグ（release.yml から呼ばれたとき github.ref と
#: github.event_name は呼び元の物）と、前もって確かめる手動の実行
RELEASE_OR_MANUAL = (
    "if: startsWith(github.ref, 'refs/tags/') || github.event_name == 'workflow_dispatch'"
)
FFMPEG_ACTION = ROOT / ".github" / "actions" / "ffmpeg" / "action.yml"


def test_the_release_verify_has_ffmpeg() -> None:
    """タグのときの検証も ffmpeg を入れてから走らせる 無いと素材を使う試験が黙って飛ぶ

    ci.yml と違い、入れられなければ止める（continue-on-error にしない）
    """
    build = _jobs(PACKAGE)["build"]
    install = build.index("uses: ./.github/actions/ffmpeg")
    step = build[build.rindex("- name:", 0, install) : install + 40]
    assert RELEASE_OR_MANUAL in step
    assert "continue-on-error" not in step
    assert install < build.index("python tools/verify.py")


def test_ffmpeg_is_pinned_in_one_place() -> None:
    """ffmpeg の版とハッシュは .github/actions/ffmpeg の 1 か所に書く

    ci.yml と package.yml に別々に書くと、片方だけ上げたときにリリースの前の検証だけが
    別の ffmpeg で走る 中で 2 度書いている所（置き場の鍵と照らす値）もそろえる
    版は配る zip の FFmpeg（collect_sources.FFMPEG_VERSION）と同じ
    """
    action = FFMPEG_ACTION.read_text(encoding="utf-8")
    version = re.search(r"FFMPEG_VERSION: (\S+)", action)
    digest = re.search(r"FFMPEG_SHA256: ([0-9a-f]{64})", action)
    assert version is not None and digest is not None
    assert f"-{version.group(1)}-essentials-{digest.group(1)}" in action
    collect = (ROOT / "tools" / "collect_sources.py").read_text(encoding="utf-8")
    assert f'FFMPEG_VERSION = "{version.group(1)}"' in collect
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    for workflow in (ci, PACKAGE):
        assert "uses: ./.github/actions/ffmpeg" in workflow
        assert "FFMPEG_SHA256" not in workflow and "codexffmpeg" not in workflow


def test_packages_outside_the_dependencies_are_installed_at_the_listed_version() -> None:
    """依存では入らないが積む包み（pip）を、一覧の版で明示して入れる（PR #241 のレビュー P1）

    pip は Python に最初から入っていて、制約（-c）は版を縛るだけで入れ直さない
    setup-python の同梱する pip が上がると、一覧の版と食い違って組み立てが止まる
    """
    build = _jobs(PACKAGE)["build"]
    written = build.index("--write-requirements build/bundled-beside.txt")
    install = build.index("uv pip install")
    assert written < install
    line = build[install : build.index("\n", install)]
    assert "-r build/bundled-beside.txt" in line


def test_pyinstaller_bootloader_is_rebuilt_and_checked() -> None:
    """PyInstaller の起動部を sdist から組み直し、既成の物と違うことを組み立ての前に確かめる

    既成の起動部のまま配った 0.1.1 は、Windows Defender の機械学習の推測
    （Trojan:Win32/Bearfoos.A!ml）で exe ごと消された 組み直しを外す変更や、
    制約の外で別に入れ直す変更を止める
    """
    build = _jobs(PACKAGE)["build"]
    install = build.index("uv pip install")
    step = build[build.rindex("- name:", 0, install) : build.index("\n", install)]
    assert 'PYINSTALLER_COMPILE_BOOTLOADER: "1"' in step
    assert "--no-binary pyinstaller" in step
    assert len(re.findall(r"uv pip install", build)) == 1
    check = build.index("python tools/check_bootloader.py")
    assert install < check < build.index("build_package.py --skip-check")


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
        "tools/package_notices.py",
        "tools/check_bootloader.py",
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


def _run_script_functions(
    names: tuple[str, ...],
    body: str,
    environment: dict[str, str],
    variables: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """道具（check_clean_machine.ps1）から関数だけを取り出し、Windows の PowerShell 5.1 で走らせる

    道具を丸ごと走らせると、zip の展開や CI の機械の設定まで行う 確かめたい関数だけを読み、
    ``body`` で呼ぶ ``variables`` は道具の中で値を入れている変数（入れる文をそのまま走らせる
    並べた順に） 関数か変数が無ければ終了コード 3
    """
    listed = ", ".join(repr(name) for name in variables) or "@()"
    code = (
        "$ErrorActionPreference = 'Stop'\n"
        "$language = 'System.Management.Automation.Language'\n"
        '$parser = "$language.Parser" -as [type]\n'
        '$definition = "$language.FunctionDefinitionAst" -as [type]\n'
        '$assignment = "$language.AssignmentStatementAst" -as [type]\n'
        "$ast = $parser::ParseFile($env:SCRIPT, [ref]$null, [ref]$null)\n"
        f"foreach ($name in @({listed})) {{\n"
        "    $found = $ast.Find({ param($node)\n"
        "        $node -is $assignment -and $node.Left.Extent.Text -eq ('$' + $name) }, $true)\n"
        "    if (-not $found) { exit 3 }\n"
        "    Invoke-Expression $found.Extent.Text\n"
        "}\n"
        f"foreach ($name in {', '.join(repr(name) for name in names)}) {{\n"
        "    $found = $ast.Find({ param($node)\n"
        "        $node -is $definition -and $node.Name -eq $name }, $true)\n"
        "    if (-not $found) { exit 3 }\n"
        "    Invoke-Expression $found.Extent.Text\n"
        "}\n" + body
    )
    powershell = (
        Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    return subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", code],
        env={**os.environ, "SCRIPT": str(CHECK_SCRIPT), **environment},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows の PowerShell 5.1 で走らせる")
def test_a_write_inside_an_existing_user_folder_is_caught(tmp_path: Path) -> None:
    """実の置き場（%APPDATA%\\Sashimono）が前からあっても、中へ書いたら見つける

    手元で走らせると実の置き場があることが多い フォルダがあるかだけを見ると、その中へ
    設定を書いても「書いていない」で通ってしまう 道具の関数だけを取り出して走らせる
    """
    place = tmp_path / "Sashimono"
    place.mkdir()
    (place / "前から.txt").write_text("before", encoding="utf-8")
    body = (
        "$before = @(Get-RealSnapshot @($env:PLACE))\n"
        "Set-Content -LiteralPath (Join-Path $env:PLACE 'settings.json') -Value '{}'\n"
        "$written = @(Get-RealWrites $before @(Get-RealSnapshot @($env:PLACE)))\n"
        "[Console]::Out.Write(($written | ForEach-Object { Split-Path -Leaf $_ }) -join ',')\n"
    )
    completed = _run_script_functions(
        ("Get-RealSnapshot", "Get-RealWrites"), body, {"PLACE": str(place)}
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "settings.json"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows の PowerShell 5.1 で走らせる")
def test_a_hung_powershell_probe_does_not_stop_the_check(tmp_path: Path) -> None:
    """手掛かりの PowerShell が時間内に終わらなくても、止めて書き残し、確かめを続ける

    終わっていないプロセスの終了コードを読むと例外になり、確かめ全体が要約を書かずに
    止まる（PR #241 のレビュー） 30 秒眠る台本を 2 秒で見切らせる
    """
    script = tmp_path / "眠る 台本.ps1"
    script.write_text("Start-Sleep -Seconds 30\n", encoding="utf-8-sig")
    body = (
        "$script:notes = [System.Collections.Generic.List[string]]::new()\n"
        "function Note([string]$Message) { $script:notes.Add($Message) }\n"
        "$Windows = $env:SystemRoot\n"
        "$variables = [ordered]@{ SYSTEMROOT = $env:SystemRoot }\n"
        "Measure-PowerShell '眠る' $variables $env:PROBE 2\n"
        # 5.1 の標準出力は日本語を化かすので、見分けた結果だけを ASCII で返す
        "$stopped = @($script:notes | Where-Object { $_ -like '*終わらないので止めた*' }).Count\n"
        "[Console]::Out.Write('stopped=' + $stopped)\n"
    )
    started = time.monotonic()
    completed = _run_script_functions(("Measure-PowerShell",), body, {"PROBE": str(script)})
    assert completed.returncode == 0, completed.stderr
    assert time.monotonic() - started < 25
    assert completed.stdout.strip() == "stopped=1"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows の PowerShell 5.1 で走らせる")
@pytest.mark.parametrize("dropped", ["GL で描く", "書き出す（FFmpeg）", "自動更新"])
def test_a_self_check_item_that_went_missing_fails(dropped: str) -> None:
    """自己診断から必須の項目が抜けたら落とす GL の 2 項目も必須（PR #241 のレビュー）

    GL の 2 項目が抜けると「GL で落ちた項目が 0」と見分けがつかず、GL の確かめも
    窓の知らせの確かめも黙って飛び、確かめは通ってしまう
    """
    from sashimono.selfcheck import (
        ENCODE_CHECK_NAME,
        EXPORT_CHECK_NAME,
        RENDER_CHECK_NAME,
        UPDATE_CHECK_NAME,
        VC_RUNTIME_CHECK_NAME,
    )

    names = [
        "版",
        "スクリプト置き場",
        ENCODE_CHECK_NAME,
        EXPORT_CHECK_NAME,
        RENDER_CHECK_NAME,
        UPDATE_CHECK_NAME,
        VC_RUNTIME_CHECK_NAME,
    ]
    present = [name for name in names if name != dropped]
    listed = ", ".join(f"'{name}'" for name in present)
    every = ", ".join(f"'{name}'" for name in names)
    body = (
        "$items = [ordered]@{}\n"
        f"foreach ($name in @({listed})) {{ $items[$name] = 'ok' }}\n"
        "$all = [ordered]@{}\n"
        f"foreach ($name in @({every})) {{ $all[$name] = 'ok' }}\n"
        # 5.1 の標準出力は日本語を化かすので、数だけを ASCII で返す
        "$missing = @(Get-MissingItems $items $RequiredItems)\n"
        "$none = @(Get-MissingItems $all $RequiredItems)\n"
        "$hit = @($missing | Where-Object { $_ -eq $env:DROPPED }).Count\n"
        '[Console]::Out.Write("missing=$($missing.Count) hit=$hit none=$($none.Count)")\n'
    )
    completed = _run_script_functions(
        ("Get-MissingItems",),
        body,
        {"DROPPED": dropped},
        variables=("GLItems", "RequiredItems"),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "missing=1 hit=1 none=0"


#: 機械の設定を触るコマンドの身代わり 呼ばれた引数を控えるだけで、何も変えない
#: 関数はコマンドレットや exe より先に引かれるので、本物には届かない
_MACHINE_STUBS = (
    "$script:calls = [System.Collections.Generic.List[string]]::new()\n"
    "function Get-NetFirewallProfile {\n"
    "    [pscustomobject]@{ Name = 'Domain'; Enabled = 'False' }\n"
    "    [pscustomobject]@{ Name = 'Public'; Enabled = 'True' }\n"
    "}\n"
    "function Set-NetFirewallProfile { $script:calls.Add('Set ' + ($args -join ' ')) }\n"
    "function New-NetFirewallRule { $script:calls.Add('New ' + ($args -join ' ')) }\n"
    "function Remove-NetFirewallRule { $script:calls.Add('Remove ' + ($args -join ' ')) }\n"
    "function auditpol {\n"
    "    $script:calls.Add('auditpol ' + ($args -join ' '))\n"
    "    if ($args[0] -eq '/get') {\n"
    "        'Machine Name,Policy Target,Subcategory,Subcategory GUID,'"
    " + 'Inclusion Setting,Exclusion Setting'\n"
    "        'PC,System,Filtering Platform Connection,{X},No Auditing,'\n"
    "    }\n"
    "    $global:LASTEXITCODE = 0\n"
    "}\n"
    "$RuleGroup = 'G'\n"
    "$AuditGuid = '{X}'\n"
)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows の PowerShell 5.1 で走らせる")
def test_the_firewall_is_left_alone_outside_ci() -> None:
    """手元（CI の runner でない機械）では、ファイアウォールと監査の設定に触らない

    触ると、開発者の機械のファイアウォールの有効・無効と監査の設定を書き換えたまま残す
    （PR #241 のレビュー P1）
    """
    body = (
        _MACHINE_STUBS
        + "$guarded = Enable-NetworkGuard @('C:\\a.exe')\n"
        + "[Console]::Out.Write(\"$guarded|\" + ($script:calls -join ';'))\n"
    )
    environment = {key: value for key, value in os.environ.items() if key != "GITHUB_ACTIONS"}
    environment["GITHUB_ACTIONS"] = ""
    completed = _run_script_functions(
        ("Enable-NetworkGuard", "Restore-NetworkGuard"), body, environment
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False|"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows の PowerShell 5.1 で走らせる")
def test_ci_puts_the_firewall_back() -> None:
    """CI でも、触る前の設定を控えて戻し、足した規則を消す 送り出しの既定には触らない

    既定が Block の機械で Allow へ書き換えると、守りを弱めたまま残す
    """
    body = (
        _MACHINE_STUBS
        + "$guarded = Enable-NetworkGuard @('C:\\a.exe', 'C:\\b.exe')\n"
        + "Restore-NetworkGuard\n"
        + "[Console]::Out.Write(\"$guarded|\" + ($script:calls -join ';'))\n"
    )
    completed = _run_script_functions(
        ("Enable-NetworkGuard", "Restore-NetworkGuard"), body, {"GITHUB_ACTIONS": "true"}
    )
    assert completed.returncode == 0, completed.stderr
    guarded, calls_text = completed.stdout.strip().split("|", 1)
    calls = calls_text.split(";")
    assert guarded == "True"
    assert "DefaultOutboundAction" not in calls_text
    # 触る前に控える
    assert calls[0].startswith("auditpol /get")
    assert "Set -All -Enabled True" in calls
    rules = [call for call in calls if call.startswith("New ")]
    assert len(rules) == 2 and all("-Group G" in rule for rule in rules)
    assert "auditpol /set /subcategory:{X} /failure:enable" in calls
    # 戻す 規則を消し、控えた有効・無効と監査の設定へ
    restored = calls[calls.index("auditpol /set /subcategory:{X} /failure:enable") + 1 :]
    assert "Remove -Group G -ErrorAction SilentlyContinue" in restored
    assert "Set -Name Domain -Enabled False" in restored
    assert "Set -Name Public -Enabled True" in restored
    assert "auditpol /set /subcategory:{X} /success:disable /failure:disable" in restored


@pytest.mark.skipif(sys.platform != "win32", reason="Windows の PowerShell 5.1 で走らせる")
def test_a_folder_the_tool_did_not_make_is_not_deleted(tmp_path: Path) -> None:
    """-Root に前からあるフォルダを渡しても、中身を消さずに止まる（PR #241 のレビュー P2）

    この道具が作った印のあるフォルダだけを消して作り直す
    """
    mine = tmp_path / "人のフォルダ"
    mine.mkdir()
    (mine / "大事.txt").write_text("keep", encoding="utf-8")
    refused = _run_script_functions(
        ("Initialize-Root",), "Initialize-Root $env:TARGET\n", {"TARGET": str(mine)}
    )
    assert refused.returncode != 0
    assert (mine / "大事.txt").read_text(encoding="utf-8") == "keep"

    made = tmp_path / "確かめ 用"
    script = "Initialize-Root $env:TARGET\n"
    first = _run_script_functions(("Initialize-Root",), script, {"TARGET": str(made)})
    assert first.returncode == 0, first.stderr
    (made / "前の確かめの残り.txt").write_text("old", encoding="utf-8")
    again = _run_script_functions(("Initialize-Root",), script, {"TARGET": str(made)})
    assert again.returncode == 0, again.stderr
    assert not (made / "前の確かめの残り.txt").exists()
    assert (made / ".sashimono-clean-check").is_file()


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
