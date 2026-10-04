r"""フォルダの入れ替えと、入れた版が起動できなかったときの巻き戻し

**動いている自分自身は置き換えられない** 本体（``Sashimono.exe``）と ``_internal`` の DLL は
動いている間ずっと開かれていて、入っているフォルダごと名前を変えられない そこで、入れ替えは
本体の外のプログラムに任せる 本体は落として確かめ、入れ替え係を起こして終わる

入れ替え係は Windows に最初から入っている PowerShell 5.1 に、ここで書いた台本を渡して走らせる
別の小さな実行ファイルを作らないのは、Python を積んだ実行ファイルをもう 1 つ配ることになり
（PyInstaller で 1 ファイルにしても 10 MB 近い）、それ自身も入れ替えの対象になるため
台本は ASCII の外の文字を含まない場所を環境変数で受け取る（本人の名前が日本語でも
コマンドの行の文字コードで化けない）

入れ替え係のすること

1. 本体と、同じフォルダから動いているほかの窓がすべて終わるのを待つ
2. 前の世代（``.previous``）を消し、今の版 → ``.previous``、新しい版 → 今の版 と改名する
   改名に失敗したら元へ戻し、前の版を起こし直す
3. 新しい版を起こし、起動できたかを確かめる（本体が窓を出した所で印のファイルを書く
   :func:`mark_started`） 2 回続けて起動できなければ、新しい版を ``.failed`` へよけて
   前の版へ戻し、前の版を起こす
4. 何をしたかを結果のファイルへ書く 次に起動した本体が読んで本人に知らせる（:func:`take_result`）
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sashimono.core.io.locks import is_held
from sashimono.update.package import APP_EXE, Layout
from sashimono.update.state import SWAP_LOCK, lock_path, update_dir

__all__ = [
    "HEALTH_ENV",
    "HELPER_SCRIPT",
    "SwapPlan",
    "launch",
    "mark_started",
    "powershell_path",
    "take_result",
    "wait_started",
]

#: 入れ替え係が新しい版に渡す、起動できた印のファイルの場所
HEALTH_ENV = "SASHIMONO_UPDATE_HEALTH_FILE"

#: 入れ替え係の台本 ここに書いた物しか走らない（落とした物は走らせない）
#: 文字列は単引用符だけで書く 二重引用符は PowerShell の中で展開の意味を持つ
HELPER_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$mode = $env:SASHIMONO_UPDATE_MODE
$install = $env:SASHIMONO_UPDATE_INSTALL
$staged = $env:SASHIMONO_UPDATE_STAGED
$previous = $env:SASHIMONO_UPDATE_PREVIOUS
$failed = $env:SASHIMONO_UPDATE_FAILED
$exeName = $env:SASHIMONO_UPDATE_EXE
$argsText = $env:SASHIMONO_UPDATE_ARGS
$waitPid = [int]$env:SASHIMONO_UPDATE_PID
$result = $env:SASHIMONO_UPDATE_RESULT
$health = $env:SASHIMONO_UPDATE_HEALTH
$relaunch = $env:SASHIMONO_UPDATE_RELAUNCH -eq '1'
$hidden = $env:SASHIMONO_UPDATE_HIDDEN -eq '1'
$waitSeconds = [int]$env:SASHIMONO_UPDATE_WAIT_SECONDS
$healthSeconds = [int]$env:SASHIMONO_UPDATE_HEALTH_SECONDS
$lock = $env:SASHIMONO_UPDATE_LOCK

function Write-Result([string]$text) {
    # 本体が結果を読んでいる間は、書き足しが断られることがある（ほかのプロセスが使用中）
    # 書けずに止まると、走り始めたのに本体は走らないと取り違える 少し待って書き直す
    for ($i = 0; $i -lt 50; $i++) {
        try {
            Add-Content -LiteralPath $result -Value $text -Encoding UTF8 -ErrorAction Stop
            return
        } catch {
            Start-Sleep -Milliseconds 100
        }
    }
    Add-Content -LiteralPath $result -Value $text -Encoding UTF8
}

# 本体と、同じフォルダから動いているほかの窓が終わるのを待つ
# 1 つでも残っていると、フォルダの名前を変えられない
function Wait-Exit {
    $deadline = (Get-Date).AddSeconds($waitSeconds)
    if ($waitPid -gt 0) {
        while ((Get-Date) -lt $deadline) {
            if (-not (Get-Process -Id $waitPid -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 200
        }
    }
    $prefix = $install.TrimEnd('\') + '\'
    $ignoreCase = [System.StringComparison]::OrdinalIgnoreCase
    while ((Get-Date) -lt $deadline) {
        $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
            try { $_.Path -and $_.Path.StartsWith($prefix, $ignoreCase) } catch { $false }
        })
        if ($running.Count -eq 0) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# 終わった直後は、ウイルス対策が exe を読んでいて改名を断ることがあるので、しばらく試し続ける
function Move-Folder([string]$from, [string]$to) {
    for ($i = 0; $i -lt 20; $i++) {
        try {
            [System.IO.Directory]::Move($from, $to)
            return $true
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

function Remove-Folder([string]$path) {
    for ($i = 0; $i -lt 10; $i++) {
        if (-not (Test-Path -LiteralPath $path)) { return $true }
        try {
            Remove-Item -LiteralPath $path -Recurse -Force
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return -not (Test-Path -LiteralPath $path)
}

function Start-App([string]$folder = $install) {
    $exe = Join-Path $folder $exeName
    $list = @()
    if ($argsText) {
        foreach ($item in ($argsText -split [char]10)) {
            if ($item) { $list += ([char]34 + $item + [char]34) }
        }
    }
    $params = @{ FilePath = $exe; PassThru = $true; WorkingDirectory = $folder }
    if ($list.Count -gt 0) { $params.ArgumentList = $list }
    if ($hidden) { $params.WindowStyle = 'Hidden' }
    return Start-Process @params
}

# 新しい版を起こし、窓を出した印を待つ 2 回続けて印を書かずに終わったら起動できないとする
# 印を書かないまま動き続けているものは、遅いだけかもしれないので起動できたとする
function Test-Started {
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        Remove-Item -LiteralPath $health -Force -ErrorAction SilentlyContinue
        $env:SASHIMONO_UPDATE_HEALTH_FILE = $health
        $process = $null
        try { $process = Start-App } catch { $process = $null }
        if ($process) {
            $deadline = (Get-Date).AddSeconds($healthSeconds)
            while ((Get-Date) -lt $deadline) {
                if (Test-Path -LiteralPath $health) { return $true }
                if ($process.HasExited) { break }
                Start-Sleep -Milliseconds 250
            }
            if (Test-Path -LiteralPath $health) { return $true }
            if (-not $process.HasExited) { return $true }
        }
        Write-Result ('start-failed ' + $attempt)
    }
    return $false
}

# 前の版を起こし直す 渡した順に、本体の exe が在るフォルダを探して起こす
# 改名の戻しまで失敗すると今の版の場所が空になる 空の場所を起こして何も出ないままにしない
function Start-Old([string[]]$folders = @($install, $previous)) {
    Remove-Item Env:SASHIMONO_UPDATE_HEALTH_FILE -ErrorAction SilentlyContinue
    if (-not $relaunch) { return }
    foreach ($folder in $folders) {
        if ($folder -and (Test-Path -LiteralPath (Join-Path $folder $exeName))) {
            if ($folder -ne $install) { Write-Result 'started-from-aside' }
            try { Start-App $folder | Out-Null } catch { Write-Result 'relaunch-failed' }
            return
        }
    }
    Write-Result 'relaunch-failed'
}

try {
    Write-Result 'started'
    # 入れ替え係は 1 つだけ 誰にも開かせずに錠を開いたまま持ち、終わるまで離さない
    # 取れなければ、ほかの入れ替え係が同じフォルダを触っている 何もせずに終わる
    # （起こした窓は閉じるだけでよい 先の入れ替え係が窓の終わるのを待って入れ替える）
    $held = $null
    try {
        $held = [System.IO.File]::Open($lock, 'OpenOrCreate', 'ReadWrite', 'None')
    } catch {
        Write-Result 'already-running'
        exit 8
    }
    Write-Result 'holding'
    if (-not (Wait-Exit)) {
        Write-Result 'busy'
        Start-Old
        exit 2
    }
    if ($mode -eq 'apply') {
        if (-not (Remove-Folder $previous)) {
            Write-Result 'previous-locked'
            Start-Old
            exit 3
        }
        if (-not (Move-Folder $install $previous)) {
            Write-Result 'install-locked'
            Start-Old
            exit 4
        }
        if (-not (Move-Folder $staged $install)) {
            Write-Result 'staged-locked'
            if (-not (Move-Folder $previous $install)) {
                # 戻しも断られた（ウイルス対策・一時的な錠） 今の版は previous に在るので、
                # そこから直に起こす 次の起動は previous の名前のままでも動く
                Write-Result 'restore-failed'
            }
            Start-Old
            exit 5
        }
        Write-Result 'swapped'
        if (-not $relaunch) { exit 0 }
        if (-not $health) { Start-App | Out-Null; exit 0 }
        if (Test-Started) { Write-Result 'healthy'; exit 0 }
        Remove-Folder $failed | Out-Null
        if ((Move-Folder $install $failed) -and (Move-Folder $previous $install)) {
            Write-Result 'rolled-back'
            Start-Old
        } else {
            # 戻し切れない 今の場所には起動できない新しい版が残っているか、空になっている
            # 前の版は previous に在るので、そこから直に起こす
            Write-Result 'rollback-failed'
            Start-Old @($previous)
        }
        exit 6
    }
    if ($mode -eq 'rollback') {
        if (-not (Remove-Folder $staged)) {
            Write-Result 'aside-locked'
            Start-Old
            exit 3
        }
        if (-not (Move-Folder $install $staged)) {
            Write-Result 'install-locked'
            Start-Old
            exit 4
        }
        if (-not (Move-Folder $previous $install)) {
            Write-Result 'previous-missing'
            if (-not (Move-Folder $staged $install)) { Write-Result 'restore-failed' }
            # 戻せなければ、よけておいた今の版から直に起こす
            Start-Old @($install, $staged, $previous)
            exit 5
        }
        if (-not (Move-Folder $staged $previous)) { Write-Result 'aside-kept' }
        Write-Result 'reverted'
        Start-Old
        exit 0
    }
    Write-Result ('unknown-mode ' + $mode)
    exit 7
} catch {
    Write-Result ('error ' + $_.Exception.Message)
    exit 9
}
"""

#: 本体が終わるのを待つ秒数 保存を尋ねる窓が出ていても、ふつうはこの間に閉じ終わる
WAIT_SECONDS = 120

#: 新しい版が窓を出すのを待つ秒数 初めての起動はウイルス対策の検査で遅いことがある
HEALTH_SECONDS = 90

#: 入れ替え係が走り始めたことを待つ秒数
#: その利用者が初めて PowerShell 5.1 を起こすときは「Preparing modules for first use」で遅い
#: CI のまっさらな Windows で 11〜22 秒、変数を削った環境では 20〜34 秒かかった（Issue #33）
#: PowerShell を使わない人は多く、更新のときが初めての起動になる 20 秒では遅い機械で
#: 「PowerShell が動かない」と取り違える 走り始めればすぐ返るので、長めに待っても速い機械は待たない
START_SECONDS = 60.0


def powershell_path() -> Path:
    """Windows に最初から入っている PowerShell 5.1 PATH は見ない（別の物にすり替えられない）"""
    # Windows の環境変数は大文字小文字を区別しない（os.environ も同じに扱う）
    root = os.environ.get("SYSTEMROOT") or r"C:\Windows"
    return Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


@dataclass(frozen=True, slots=True)
class SwapPlan:
    """入れ替え係に頼むこと"""

    #: ``apply``（新しい版を入れる）か ``rollback``（前の版へ戻す）
    mode: str
    layout: Layout
    #: 終わるのを待つ本体の番号 0 なら待たない
    pid: int = 0
    #: 入れ替えた後に起こすか（自己診断の試しでは起こさない）
    relaunch: bool = True
    #: 起こすときに渡す引数（開いていたプロジェクト）
    arguments: Sequence[str] = field(default_factory=tuple)
    #: 新しい版が起動できたかを確かめるか（``apply`` で起こすときだけ意味がある）
    check_start: bool = True
    exe_name: str = APP_EXE
    wait_seconds: int = WAIT_SECONDS
    health_seconds: int = HEALTH_SECONDS
    #: 起こす物の窓を隠す（試験で .cmd を起こすとき） 本物の exe には使わない
    #: 隠すと本体の最初の窓まで隠れたままになる
    hidden: bool = False


@dataclass
class Launched:
    """起こした入れ替え係"""

    process: subprocess.Popen[bytes]
    result: Path

    def lines(self) -> list[str]:
        return _lines(self.result)

    def lost_to_another(self) -> bool:
        """ほかの入れ替え係が先に錠を取っていた（2 つの窓で入れ替えを選んだ）"""
        return "already-running" in self.lines()


def _aside(layout: Layout) -> Path:
    """前の版へ戻すとき、今の版を一時的によけておく場所"""
    return layout.install.with_name(layout.install.name + ".rolling")


def launch(plan: SwapPlan, folder: Path | None = None) -> Launched:
    """入れ替え係を起こす 本体はこの後に終わる（待つのは入れ替え係の側）

    台本は毎回書き直す 置いてある台本を走らせると、誰かが書き換えた物を走らせることになる
    """
    folder = folder if folder is not None else update_dir()
    folder.mkdir(parents=True, exist_ok=True)
    # 台本と結果は起こすたびに別の名前にする 2 つの窓がほぼ同時に起こしたとき、片方の
    # 結果をもう片方が消したり、片方の「走り始めた」をもう片方が自分の物と読んだりしない
    token = uuid.uuid4().hex
    script = folder / f"swap-{token}.ps1"
    # BOM 付きで書く PowerShell 5.1 は BOM の無い台本を本人の文字コード（cp932）で読む
    script.write_text(HELPER_SCRIPT, encoding="utf-8-sig")
    result = folder / f"result-{token}.txt"
    health = folder / f"started-{token}.txt"
    layout = plan.layout
    staged = layout.staged if plan.mode == "apply" else _aside(layout)
    environment = {
        **os.environ,
        "SASHIMONO_UPDATE_MODE": plan.mode,
        "SASHIMONO_UPDATE_INSTALL": str(layout.install),
        "SASHIMONO_UPDATE_STAGED": str(staged),
        "SASHIMONO_UPDATE_PREVIOUS": str(layout.previous),
        "SASHIMONO_UPDATE_FAILED": str(layout.failed),
        "SASHIMONO_UPDATE_EXE": plan.exe_name,
        "SASHIMONO_UPDATE_ARGS": "\n".join(plan.arguments),
        "SASHIMONO_UPDATE_PID": str(plan.pid),
        "SASHIMONO_UPDATE_RESULT": str(result),
        "SASHIMONO_UPDATE_HEALTH": str(health) if plan.check_start else "",
        "SASHIMONO_UPDATE_RELAUNCH": "1" if plan.relaunch else "0",
        "SASHIMONO_UPDATE_HIDDEN": "1" if plan.hidden else "0",
        "SASHIMONO_UPDATE_WAIT_SECONDS": str(plan.wait_seconds),
        "SASHIMONO_UPDATE_HEALTH_SECONDS": str(plan.health_seconds),
        "SASHIMONO_UPDATE_LOCK": str(lock_path(SWAP_LOCK, folder)),
    }
    environment.pop(HEALTH_ENV, None)
    command = [
        str(powershell_path()),
        "-NoProfile",
        "-NonInteractive",
        # 会社の機械で台本の実行を止める設定（グループポリシー）はこれでも越えられない
        # そのときは走り始めた印が来ないので、本体は終わらずに知らせる（wait_started）
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    ]
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    # 本体が終わっても入れ替え係は残す 本体を起こした物（端末など）がジョブで子を
    # まとめて止める作りのときは、抜けないと本体と一緒に止められる 抜けるのを許さない
    # ジョブでは起動そのものが断られるので、抜けずに起こし直す
    for flags in (no_window | group | breakaway, no_window | group):
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
                close_fds=True,
            )
        except PermissionError:
            continue
        return Launched(process, result)
    raise OSError("入れ替え係を起こせない")


def _lines(result: Path) -> list[str]:
    try:
        text = result.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def wait_started(launched: Launched, timeout: float = START_SECONDS) -> bool:
    """入れ替え係が走り始め、錠を取れたか 取れなければ止めて偽を返す

    走り始めたのを見てから本体を終える 見ずに終えると、台本の実行が止められている
    機械では、本体が消えたまま誰も起こし直さない ほかの入れ替え係が錠を持っていれば
    偽（:meth:`Launched.lost_to_another` が真）
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lines = launched.lines()
        if "holding" in lines:
            return True
        if "already-running" in lines or launched.process.poll() is not None:
            break
        time.sleep(0.1)
    if "holding" in launched.lines():
        return True
    if launched.process.poll() is None:
        launched.process.kill()
    return False


def mark_started() -> None:
    """新しい版が窓を出せたことを入れ替え係へ知らせる 入れ替え係が起こしたときだけ"""
    path = os.environ.pop(HEALTH_ENV, None)
    if not path:
        return
    # 書けなければ、入れ替え係は動き続けている本体を見て起動できたとする
    with contextlib.suppress(OSError):
        Path(path).write_text("ok", encoding="utf-8")


def take_result(folder: Path | None = None) -> list[str]:
    """前の入れ替えの結果を読んで消す 1 回だけ知らせるため

    入れ替え係がまだ走っている（起こした新しい版が窓を出すのを待っている）間は読まない
    消すと、その後に書かれる結果が次の起動まで残り、片付けも途中の物を消すことになる
    """
    folder = folder if folder is not None else update_dir()
    if is_held(lock_path(SWAP_LOCK, folder)):
        return []
    results = sorted(folder.glob("result-*.txt"), key=lambda path: path.stat().st_mtime)
    lines = [line for result in results for line in _lines(result)]
    for leftover in (*results, *folder.glob("started-*.txt"), *folder.glob("swap-*.ps1")):
        leftover.unlink(missing_ok=True)
    return lines
