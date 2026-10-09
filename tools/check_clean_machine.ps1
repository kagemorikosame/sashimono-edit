<#
配る zip を、開発の道具が何も無い状態で展開して確かめる（Issue #33）

CI のまっさらな Windows（.github/workflows/package.yml の clean-machine）で走らせる
ファイアウォールと監査の設定を触る 3 は CI の runner（GITHUB_ACTIONS が true）でだけ行い、
触る前の設定を控えて最後に戻し、足した規則も消す 手元で走らせると 3 は理由を出して飛ばす
（使う人や開発者の機械のファイアウォールを書き換えない）

    pwsh tools/check_clean_machine.ps1 -Zip dist\SashimonoEdit-<版>-windows-x64.zip

確かめること
  1. 展開先と使う人の置き場（APPDATA・LOCALAPPDATA・USERPROFILE・TEMP）を日本語と空白を
     含むフォルダにする ユーザー名が日本語の人の機械はこうなる
  2. 環境変数は Windows の分と上の置き場だけにして走らせる PATH から Python・uv・FFmpeg を
     外し、PYTHONHOME や PYTHONPATH も渡さない runner に元からある開発の道具を拾って
     通ってしまうと、zip の積み忘れに気付けない
  3. （CI だけ）exe が外へ出られないようにファイアウォールで止める 自己診断（自動更新の項目を含む）は
     外へ出ずに通らなければならない 止めた接続を監査の記録（5157）で数える
  4. Sashimono.exe --self-check の結果を 1 項目ずつ見る GPU が無いので GL の 2 項目
     （描く・書き出す）は落ちてよいが、落ちた理由が GL であること（GLContextError）を確かめる
     それ以外の項目は全部通らなければならない
  5. exe の pip で、導入ボタンと同じ入れ方の見本の wheel を日本語のフォルダへ入れる
  6. 引数無しで起動して窓が出ること GL の無い機械で描けない理由を窓に出すこと
     閉じて終了コード 0 で終わること（#149 GL の無い機械で閉じた後に落ちた）を見る
  7. 実の利用者の置き場（runner の APPDATA など）に何も書いていないことを見る
  8. ファイアウォールを戻した後で、後から入れる部品（AI 連携・字幕起こし）を exe の pip で
     日本語のフォルダへ入れて import する（Sashimono.exe --add-on-check ネットにつなぐ）
     導入ボタンと同じく exe は自分を子として起こさず、自分の中で走らせる 0.1.3 は自分を子として
     起こす流れの途中で、利用者の機械の Defender に消された
  9. exe のプロパティ（版情報）に製品名と版が入っていること 版は zip の build-info.json と同じ
     （コード署名の条件で、署名する exe の製品名と版をそろえる Issue #255）
 10. 自動更新の入れ替えを本物の exe で通す 作業場所をインストール先にして（Explorer から起こしたのと
     同じ）起こし、展開済みの新しい版（今組んだ zip）へ入れ替わって窓を出すまでを見る（#279）
     今組んだ版からは必ず通す 1 つ前に公開した版（GitHub Releases の latest CI でだけ落とす）からは、
     その版が入れ替え係の台本を持つ（0.2.1 から）なら必ず通し、持たなければ失敗を記録するだけ
     （切り替えは Test-PreviousMustPass の 1 か所）

結果は -Report のフォルダ（ログと窓の写真）と、GITHUB_STEP_SUMMARY（あれば）へ書く
1 つでも落ちたら終了コード 1

Windows PowerShell 5.1 では走らない（ProcessStartInfo.ArgumentList と Encoding.Latin1 を使う）
ファイルは BOM 付きの UTF-8 で置く BOM が無いと 5.1 で開いたときに日本語が化けて、
走らない理由が読めない
#>
param(
    [Parameter(Mandatory = $true)][string]$Zip,
    # 日本語と空白を含める ドライブの直下に置くのは、パスの長さで落ちる別の失敗と混ぜないため
    # 前からあるフォルダは、この道具が作った印（.sashimono-clean-check）が無ければ断る
    [string]$Root = 'C:\テスト 利用者',
    [string]$Report = (Join-Path ([IO.Path]::GetTempPath()) 'sashimono-clean-check'),
    # 窓が出てから閉じるまで待つ秒数 起動の 3 秒後に更新を確かめに行く所まで通したい
    [int]$WindowSeconds = 10,
    # GL の 2 項目が GL の理由で落ちるのを許す GPU のある機械で走らせるなら外す
    [switch]$RequireGL,
    # 1 つ前に公開した版の zip 渡さなければ、CI の runner でだけ GitHub Releases の latest から落とす
    [string]$PreviousZip = ''
)

$ErrorActionPreference = 'Stop'

$script:Failures = [System.Collections.Generic.List[string]]::new()
$script:Notes = [System.Collections.Generic.List[string]]::new()

function Fail([string]$Message) {
    $script:Failures.Add($Message)
    Write-Host "::error::$Message"
}

function Note([string]$Message) {
    $script:Notes.Add($Message)
    Write-Host $Message
}

# 自己診断が GL の無い機械で落としてよい項目 名前は src/sashimono/selfcheck.py と揃える
$GLItems = @('GL で描く', '書き出す（FFmpeg）')
# 自己診断に必ずある項目 GL の 2 項目も含める 抜けると「GL で落ちた項目が 0」と
# 見分けがつかず、GL の確かめも窓の知らせの確かめも黙って飛ぶ
$RequiredItems = @('版', 'FFmpeg で符号化（日本語のパス）', '自動更新', 'Visual C++ の実行時の部品',
    'スクリプト置き場') + $GLItems

function Get-MissingItems($Items, [string[]]$Required) {
    # 自己診断の結果（名前 → 結果）に無い必須の項目の名前
    $Required | Where-Object { -not $Items.Contains($_) }
}

# 自己診断の最後の行 これが無ければ途中で落ちている
$Passed = 'すべて動いた'

New-Item -ItemType Directory -Force $Report | Out-Null
function Initialize-Root([string]$Path) {
    # 前の確かめで作ったフォルダだけを消す 印の無いフォルダを消すと、-Root を書き違えた
    # ときに人のファイルを丸ごと消す 印はフォルダの直下に、この道具が作ったときにだけ置く
    $mark = Join-Path $Path '.sashimono-clean-check'
    if (Test-Path -LiteralPath $Path) {
        if (-not (Test-Path -LiteralPath $mark -PathType Leaf)) {
            throw "$Path は前からあるフォルダ（この道具が作った印が無い） 消さずに止める 無いフォルダを -Root に渡す"
        }
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Path | Out-Null
    Set-Content -LiteralPath $mark -Value 'tools/check_clean_machine.ps1 が作った 次に走らせると消える' -Encoding utf8
}
Initialize-Root $Root
$Roaming = Join-Path $Root 'AppData\Roaming'
$Local = Join-Path $Root 'AppData\Local'
$Temp = Join-Path $Local 'Temp'
$Place = Join-Path $Root '展開 先'
foreach ($folder in @($Roaming, $Local, $Temp, $Place)) {
    New-Item -ItemType Directory -Force $folder | Out-Null
}

Expand-Archive -LiteralPath $Zip -DestinationPath $Place
$AppHome = Join-Path $Place 'Sashimono'
$Exe = Join-Path $AppHome 'Sashimono.exe'
if (-not (Test-Path -LiteralPath $Exe)) { throw "展開した zip に Sashimono.exe が無い: $Exe" }
Note "展開した先: $Exe"

# --- exe のプロパティ（版情報） ---
# コード署名（SignPath Issue #255）の条件で、署名する exe の製品名と版をそろえる 組み立てで
# 入れ忘れても exe は動くので、配る zip の exe を読んで確かめる
# 版は zip の build-info.json（組み立てが __version__ から書いた物）と比べる 製品名は
# tools/build_package.py の PRODUCT_NAME と同じ（揃っているかは tests/test_packaging.py が見る）
$ExpectedProduct = 'Sashimono Edit'
$BuildInfo = Get-Content -LiteralPath (Join-Path $AppHome 'build-info.json') -Raw -Encoding utf8 | ConvertFrom-Json
$VersionInfo = (Get-Item -LiteralPath $Exe).VersionInfo
$ExpectedVersionInfo = [ordered]@{
    ProductName      = $ExpectedProduct
    ProductVersion   = $BuildInfo.version
    FileVersion      = $BuildInfo.version
    OriginalFilename = 'Sashimono.exe'
    InternalName     = 'Sashimono'
}
foreach ($entry in $ExpectedVersionInfo.GetEnumerator()) {
    $actual = $VersionInfo.($entry.Key)
    if ($actual -cne $entry.Value) { Fail "exe の版情報の $($entry.Key) が '$actual'（'$($entry.Value)' のはず）" }
}
foreach ($name in @('FileDescription', 'CompanyName', 'LegalCopyright')) {
    if (-not $VersionInfo.$name) { Fail "exe の版情報の $name が空" }
}
Note ('exe の版情報: ' + ((@('ProductName', 'ProductVersion', 'FileVersion', 'FileDescription', 'CompanyName',
                'LegalCopyright', 'OriginalFilename', 'InternalName') |
            ForEach-Object { "$_=$($VersionInfo.$_)" }) -join ' / '))

# 走らせる前の実の置き場 終わった後に Sashimono の物が増えていないかを見る
$RealPlaces = @($env:APPDATA, $env:LOCALAPPDATA) | Where-Object { $_ } |
    ForEach-Object { Join-Path $_ 'Sashimono' }
function Get-RealSnapshot([string[]]$Places) {
    # フォルダとその中の 1 つずつを、場所・大きさ・書いた時刻で控える 手元で走らせると
    # 実の置き場が前からあることがあり、フォルダがあるかだけを見ると、その中へ書いても通ってしまう
    foreach ($place in $Places) {
        if (-not (Test-Path -LiteralPath $place)) { continue }
        $place
        Get-ChildItem -LiteralPath $place -Force -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object { '{0}|{1}|{2}' -f $_.FullName, $(if ($_.PSIsContainer) { '' } else { $_.Length }), $_.LastWriteTimeUtc.Ticks }
    }
}
function Get-RealWrites([string[]]$Before, [string[]]$After) {
    # 増えた物と書き換わった物（大きさか時刻が変わった）の場所 消えた物は数えない
    $After | Where-Object { $Before -notcontains $_ } | ForEach-Object { ($_ -split '\|')[0] }
}
$RealBefore = @(Get-RealSnapshot $RealPlaces)

# 環境変数 Windows が動くのに要る分と、日本語の置き場だけ
# PATH は Windows の分だけ（tools/build_package.py の minimal_environment と同じ）
$Windows = $env:SystemRoot
$Environment = [ordered]@{
    SYSTEMROOT   = $Windows
    WINDIR       = $Windows
    SYSTEMDRIVE  = $env:SystemDrive
    COMPUTERNAME = $env:COMPUTERNAME
    USERNAME     = $env:USERNAME
    PROGRAMDATA  = $env:ProgramData
    USERPROFILE  = $Root
    HOMEDRIVE    = Split-Path -Qualifier $Root
    HOMEPATH     = Split-Path -NoQualifier $Root
    APPDATA      = $Roaming
    LOCALAPPDATA = $Local
    TEMP         = $Temp
    TMP          = $Temp
    PATH         = @("$Windows\System32", $Windows, "$Windows\System32\Wbem") -join ';'
}
# 素の Windows の利用者なら誰でも持っている変数 中身は Windows の物で、開発の道具は指さない
# 無いと PowerShell 5.1 の起動に 30 秒かかった（入れ替え係が走り始めるのを待ちきれない）
# 使う人の機械には必ずあるので、外したまま確かめると使う人の機械で起きない失敗になる
$WindowsDefaults = [ordered]@{}
foreach ($name in @(
        'ALLUSERSPROFILE', 'PUBLIC', 'ComSpec', 'PATHEXT', 'OS', 'NUMBER_OF_PROCESSORS',
        'PROCESSOR_ARCHITECTURE', 'PROCESSOR_IDENTIFIER', 'PROCESSOR_LEVEL', 'PROCESSOR_REVISION',
        'ProgramFiles', 'ProgramFiles(x86)', 'ProgramW6432',
        'CommonProgramFiles', 'CommonProgramFiles(x86)', 'CommonProgramW6432', 'DriverData'
    )) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($value) { $WindowsDefaults[$name] = $value }
}
# runner の PSModulePath には pwsh や Azure のモジュールが載っている Windows の既定の並びにする
$WindowsDefaults['PSModulePath'] = @(
    "$env:ProgramFiles\WindowsPowerShell\Modules", "$Windows\system32\WindowsPowerShell\v1.0\Modules"
) -join ';'
$Minimal = [ordered]@{}
foreach ($entry in $Environment.GetEnumerator()) { $Minimal[$entry.Key] = $entry.Value }
foreach ($entry in $WindowsDefaults.GetEnumerator()) { $Environment[$entry.Key] = $entry.Value }

function Read-Output([byte[]]$Bytes) {
    # 自己診断は、出口の文字コードで日本語を書けないとき UTF-8 へ切り替える
    # （英語の Windows の CI がそう） 日本語の Windows なら cp932 のまま書くので、UTF-8 として
    # 読めなければ Windows の既定の文字コードで読む
    try {
        return [System.Text.UTF8Encoding]::new($false, $true).GetString($Bytes)
    } catch {
        $codePage = [Globalization.CultureInfo]::CurrentCulture.TextInfo.ANSICodePage
        try {
            [System.Text.Encoding]::RegisterProvider([System.Text.CodePagesEncodingProvider]::Instance)
            return [System.Text.Encoding]::GetEncoding($codePage).GetString($Bytes)
        } catch {
            return [System.Text.Encoding]::Latin1.GetString($Bytes)
        }
    }
}

function New-StartInfo([string[]]$Arguments, [bool]$Capture) {
    $info = [System.Diagnostics.ProcessStartInfo]::new($Exe)
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    $info.WorkingDirectory = $Place
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $Capture
    $info.RedirectStandardError = $Capture
    # 受け継いだ環境変数を全部消してから足す 消さないと runner の PATH・PYTHONHOME・
    # pythonLocation（setup-python が立てる）がそのまま渡る
    $info.Environment.Clear()
    foreach ($entry in $Environment.GetEnumerator()) { $info.Environment[$entry.Key] = $entry.Value }
    return $info
}

function Invoke-Exe([string]$Name, [string[]]$Arguments, [int]$TimeoutSeconds = 900) {
    $process = [System.Diagnostics.Process]::Start((New-StartInfo $Arguments $true))
    $out = [IO.MemoryStream]::new()
    $err = [IO.MemoryStream]::new()
    # 2 本を同時に読む 片方だけ読むと、もう片方の管が詰まった所で exe が止まる
    $copyOut = $process.StandardOutput.BaseStream.CopyToAsync($out)
    $copyErr = $process.StandardError.BaseStream.CopyToAsync($err)
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill($true)
        throw "$Name が $TimeoutSeconds 秒で返ってこない"
    }
    [void][Threading.Tasks.Task]::WaitAll(@($copyOut, $copyErr), 30000)
    $stdout = Read-Output $out.ToArray()
    $stderr = Read-Output $err.ToArray()
    $log = Join-Path $Report "$Name.txt"
    Set-Content -LiteralPath $log -Encoding utf8 -Value @(
        "終了コード: $($process.ExitCode)", '--- 標準出力 ---', $stdout, '--- 標準エラー ---', $stderr
    )
    return [pscustomobject]@{ ExitCode = $process.ExitCode; Out = $stdout; Err = $stderr; Id = $process.Id }
}

Add-Type -AssemblyName System.Drawing
Add-Type -Namespace Sashimono -Name Native -MemberDefinition @'
[System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
public struct Rect { public int Left; public int Top; public int Right; public int Bottom; }
[System.Runtime.InteropServices.DllImport("user32.dll")]
public static extern bool GetWindowRect(System.IntPtr window, out Rect rect);
'@

function Test-GLNotice([long]$Handle) {
    # 窓の中の文字を UI Automation で読み、GL を使えない理由が出ていれば 0 を返す
    # pwsh 7 から使えるとは限らないので、Windows に入っている PowerShell 5.1 で読む
    # 型は Add-Type の後で文字列から引く（5.1 は型を書いた所を読む時点で解決しようとする）
    # 進み具合の表示を止める 出力が端末でないと CLIXML でログに混ざる
    $find = @"
`$ProgressPreference = 'SilentlyContinue'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
`$element = 'System.Windows.Automation.AutomationElement' -as [type]
`$scope = 'System.Windows.Automation.TreeScope' -as [type]
`$condition = 'System.Windows.Automation.Condition' -as [type]
`$root = `$element::FromHandle([IntPtr]::new($Handle))
foreach (`$item in `$root.FindAll(`$scope::Descendants, `$condition::TrueCondition)) {
    `$name = `$item.Current.Name
    if (`$name -and `$name.Contains('OpenGL 4.3')) { exit 0 }
}
exit 1
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($find))
    & "$Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive `
        -EncodedCommand $encoded 2>$null | Out-Null
    return $LASTEXITCODE
}

# --- ファイアウォール 外へ出られないようにする（CI だけ） ---
# 触るのは CI の runner（使い捨ての機械）でだけ 手元で走らせて、開発者の機械の
# ファイアウォールの有効・無効や監査の設定を書き換えたまま残さない（PR #241 のレビュー）
# CI でも、触る前の設定を控えて最後（finally）に戻し、足した規則を消す
$RuleGroup = 'Sashimono clean-machine check'
# Windows Filtering Platform の監査（接続を止めた 5157） 名前は言語で変わるので GUID で指す
$AuditGuid = '{0CCE9226-69AE-11D9-BED3-505054503030}'
$script:Guard = $null

function Enable-NetworkGuard([string[]]$Programs) {
    # 触ってよい機械でなければ何もしない 偽を返す
    if ($env:GITHUB_ACTIONS -ne 'true') { return $false }
    # 触る前に控える 控えられなければ（権限が無いなど）触らずに止まる
    $profiles = @(Get-NetFirewallProfile | ForEach-Object { [pscustomobject]@{ Name = $_.Name; Enabled = $_.Enabled } })
    $audit = @(auditpol /get /subcategory:$AuditGuid /r | ConvertFrom-Csv)
    if ($LASTEXITCODE -ne 0 -or $audit.Count -eq 0) { throw '監査の設定を読めない（触らずに止める）' }
    $script:Guard = [pscustomobject]@{ Profiles = $profiles; Audit = [string]$audit[0].'Inclusion Setting' }
    # runner は既定でファイアウォールを切っていることがある 切れていると規則が効かない
    # 送り出しの既定（DefaultOutboundAction）には触らない 止めるのは渡した exe だけ
    Set-NetFirewallProfile -All -Enabled True
    foreach ($program in $Programs) {
        New-NetFirewallRule -DisplayName "Sashimono の確かめで外へ出さない $([IO.Path]::GetFileName($program))" `
            -Group $RuleGroup -Direction Outbound -Program $program -Action Block | Out-Null
    }
    # ファイアウォールの記録（pfirewall.log）は runner では止めた送り出しを 1 件も書かなかった
    # 監査の記録はどの exe が出ようとしたかまで残る
    auditpol /set /subcategory:$AuditGuid /failure:enable | Out-Null
    return $true
}

function Restore-NetworkGuard {
    # 控えた設定へ戻し、足した規則を消す 控えが無ければ（触っていない）何もしない
    if ($null -eq $script:Guard) { return }
    Remove-NetFirewallRule -Group $RuleGroup -ErrorAction SilentlyContinue
    foreach ($saved in $script:Guard.Profiles) {
        Set-NetFirewallProfile -Name $saved.Name -Enabled $saved.Enabled
    }
    $success = if ($script:Guard.Audit -match 'Success') { 'enable' } else { 'disable' }
    $failure = if ($script:Guard.Audit -match 'Failure') { 'enable' } else { 'disable' }
    auditpol /set /subcategory:$AuditGuid /success:$success /failure:$failure | Out-Null
    $script:Guard = $null
}

function Get-Blocked([datetime]$Since, [string]$Program) {
    if (-not $script:Guarded) { return @() }
    # 記録はすぐには書かれないことがある 少し待ってから読む
    Start-Sleep -Seconds 5
    $found = @(Get-WinEvent -FilterHashtable @{ LogName = 'Security'; Id = 5157; StartTime = $Since } `
            -ErrorAction SilentlyContinue)
    # 1 番目が exe の場所（\device\harddiskvolume…\…\sashimono.exe） 2 番目が向き（%%14593 が外向き）
    return @($found | Where-Object {
            ([string]$_.Properties[1].Value) -like "*\$Program" -and ([string]$_.Properties[2].Value) -match '14593'
        })
}

function Measure-PowerShell([string]$Label, $Variables, [string]$Script, [int]$TimeoutSeconds = 120) {
    # 手掛かりを書くだけ 時間切れでも止めて Note を書き、確かめの残りと要約へ進む
    # 終わっていないプロセスの ExitCode は例外を投げ、確かめ全体が要約を書かずに止まる
    # Windows PowerShell 5.1 からも呼べる形で書く（試験が 5.1 で取り出して走らせる）
    $probeResult = Join-Path (Split-Path -Parent $Script) "result-$([guid]::NewGuid().ToString('N')).txt"
    $probeInfo = New-Object System.Diagnostics.ProcessStartInfo "$Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    $probeInfo.Arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $Script + '"'
    $probeInfo.UseShellExecute = $false
    $probeInfo.RedirectStandardOutput = $true
    $probeInfo.RedirectStandardError = $true
    $probeInfo.Environment.Clear()
    foreach ($entry in $Variables.GetEnumerator()) { $probeInfo.Environment[$entry.Key] = $entry.Value }
    $probeInfo.Environment['PROBE_RESULT'] = $probeResult
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $probe = [System.Diagnostics.Process]::Start($probeInfo)
    $probeOut = $probe.StandardOutput.ReadToEndAsync()
    $probeErr = $probe.StandardError.ReadToEndAsync()
    if (-not $probe.WaitForExit($TimeoutSeconds * 1000)) {
        # 止めれば管が閉じて、標準出力・標準エラーの読み取りも終わる それでも待ち続けない
        try { $probe.Kill() } catch { }
        [void]$probe.WaitForExit(10000)
        Note ('PowerShell 5.1 の台本（{0}）: {1} 秒で終わらないので止めた' -f $Label, $TimeoutSeconds)
        return
    }
    $probeText = if (Test-Path -LiteralPath $probeResult) { (Get-Content -LiteralPath $probeResult -Raw).Trim() } else { '（無い）' }
    Note ('PowerShell 5.1 の台本（{0}）: {1:N1} 秒 終了コード {2} 結果 {3}' -f
        $Label, $watch.Elapsed.TotalSeconds, $probe.ExitCode, $probeText)
    # 終わった後でも読み取りが残ることがある 待つのは少しだけ
    if ($probeErr.Wait(5000) -and $probeErr.Result.Trim()) { Note "PowerShell 5.1 の標準エラー: $($probeErr.Result.Trim())" }
    if ($probeOut.Wait(5000) -and $probeOut.Result.Trim()) { Note "PowerShell 5.1 の標準出力: $($probeOut.Result.Trim())" }
}

try {
    # 止める規則と記録が本当に効いているかを、先に別の exe（curl の写し）で確かめる
    # 効いていないと、自己診断が外へ出ても「止めた 0 件」で通ってしまう
    $curlCopy = Join-Path $Temp 'firewall-check.exe'
    Copy-Item -LiteralPath "$Windows\System32\curl.exe" -Destination $curlCopy
    $script:Guarded = Enable-NetworkGuard @($Exe, $curlCopy)
    if (-not $script:Guarded) {
        Note 'CI の runner ではないので、ファイアウォールと監査の設定に触らず、外へ出ないことの確かめを飛ばした'
    } else {
        $since = Get-Date
        & $curlCopy --silent --output NUL --max-time 15 https://api.github.com/zen
        $curlExit = $LASTEXITCODE
        $controlDrops = @(Get-Blocked $since 'firewall-check.exe')
        if ($curlExit -eq 0) {
            Fail 'ファイアウォールの規則で exe を止められない（外へ出ないことを確かめられない）'
        } elseif ($controlDrops.Count -eq 0) {
            Fail "止めた接続が監査の記録に残らない（curl は止まった 終了コード $curlExit） 自己診断が外へ出ようとしたかを数えられない"
        } else {
            Note "ファイアウォールが効いている（見本の exe の外向きの接続を $($controlDrops.Count) 件止めて記録した）"
        }
    }

    # --- 手掛かり 同じ環境で PowerShell 5.1 の台本が走るか ---
    # 自動更新の入れ替え係は Windows の PowerShell 5.1 で台本を走らせる 自己診断の自動更新の
    # 項目が落ちたときに、PowerShell そのものが走らないのか、入れ替え係の中で落ちたのかを分ける
    $probeFolder = Join-Path $Temp '台本 確かめ'
    New-Item -ItemType Directory -Force $probeFolder | Out-Null
    $probeScript = Join-Path $probeFolder 'probe.ps1'
    [IO.File]::WriteAllText($probeScript,
        "Add-Content -LiteralPath `$env:PROBE_RESULT -Value 'holding' -Encoding UTF8`n",
        [Text.UTF8Encoding]::new($true))

    # 変数を削りすぎた環境と、確かめに使う環境（素の Windows の利用者と同じ変数）を比べる
    Measure-PowerShell '削りすぎた環境変数' $Minimal $probeScript
    Measure-PowerShell '確かめに使う環境変数' $Environment $probeScript

    # --- 自己診断 ---
    $Scripts = Join-Path $AppHome 'scripts'
    New-Item -ItemType Directory -Force $Scripts | Out-Null
    # 置き場に置いた見本が読まれることも見る（tools/build_package.py の smoke_test と同じ見本）
    Set-Content -LiteralPath (Join-Path $Scripts '確かめる用.anm2') -Encoding utf8NoBOM -Value @(
        '--track@amount:量,0,100,50', 'obj.ox = amount'
    )

    $since = Get-Date
    # CI で 30 秒余り 待ちの合計がジョブの制限（package.yml の timeout-minutes）に収まるよう絞る
    # 収まらないと、要約を書く前に Actions がジョブを取り消す（試験が合計を見張る）
    $check = Invoke-Exe 'self-check' @('--self-check') 300
    $checkDrops = @(Get-Blocked $since 'sashimono.exe')
    Write-Host $check.Out
    if ($check.Err.Trim()) { Write-Host $check.Err }

    $items = [ordered]@{}
    foreach ($line in ($check.Out -split "`r?`n")) {
        if ($line -match '^\[(ok|NG|--)\] (.+?): (.*)$') {
            $items[$Matches[2]] = [pscustomobject]@{ Mark = $Matches[1]; Detail = $Matches[3] }
        }
    }
    $last = ($check.Out.Trim() -split "`r?`n")[-1]
    if ($items.Count -eq 0) {
        Fail "自己診断が 1 項目も出さずに終わった（終了コード $($check.ExitCode)）"
    } elseif ($last -ne $Passed -and $last -notmatch '^動かない項目が \d+ 個$') {
        Fail "自己診断が最後まで走っていない（最後の行: $last）"
    }

    $glFailures = 0
    foreach ($entry in $items.GetEnumerator()) {
        $name = $entry.Key
        $item = $entry.Value
        if ($item.Mark -ne 'NG') { continue }
        if (-not $RequireGL -and $GLItems -contains $name -and $item.Detail.StartsWith('GLContextError:')) {
            $glFailures += 1
            Note "GL が無いので落ちた（理由を確かめた）: $name — $($item.Detail)"
            continue
        }
        Fail "自己診断の $name が動かない: $($item.Detail)"
    }
    # 落ちた項目が GL の分だけなら、終了コードは 1 になる それ以外で 0 でなければ落ちている
    $expectedExit = if ($glFailures -gt 0) { 1 } else { 0 }
    if ($items.Count -gt 0 -and $check.ExitCode -ne $expectedExit) {
        Fail "自己診断の終了コードが $($check.ExitCode)（$expectedExit のはず）"
    }

    foreach ($missing in @(Get-MissingItems $items $RequiredItems)) {
        Fail "自己診断に $missing の項目が無い"
    }
    if ($items.Contains('版') -and $items['版'].Detail -notmatch '配布版') {
        Fail "配布版として動いていない: $($items['版'].Detail)"
    }
    if ($items.Contains('スクリプト置き場') -and -not $items['スクリプト置き場'].Detail.Contains("$Scripts（1 本）")) {
        Fail "exe の隣の置き場に置いたスクリプトが読まれていない: $($items['スクリプト置き場'].Detail)"
    }
    if ($checkDrops.Count -gt 0) {
        Fail "自己診断が外へ出ようとした（止めた接続 $($checkDrops.Count) 件）"
        $checkDrops | ForEach-Object { Write-Host $_.Message }
    }

    # --- 導入ボタンの pip 見本の wheel を日本語のフォルダへ入れる ---
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $sample = 'sashimono_check_sample'
    $info = "$sample-0.1.dist-info"
    $wheel = Join-Path $Place "$sample-0.1-py3-none-any.whl"
    $files = [ordered]@{
        "$sample/__init__.py" = "VALUE = 1`n"
        "$info/METADATA"      = "Metadata-Version: 2.1`nName: $sample`nVersion: 0.1`n"
        "$info/WHEEL"         = "Wheel-Version: 1.0`nGenerator: sashimono`nRoot-Is-Purelib: true`nTag: py3-none-any`n"
    }
    $files["$info/RECORD"] = (($files.Keys | ForEach-Object { "$_,," }) + "$info/RECORD,,") -join "`n"
    $archive = [System.IO.Compression.ZipFile]::Open($wheel, 'Create')
    try {
        foreach ($entry in $files.GetEnumerator()) {
            $writer = [IO.StreamWriter]::new($archive.CreateEntry($entry.Key).Open(), [Text.UTF8Encoding]::new($false))
            $writer.Write($entry.Value)
            $writer.Dispose()
        }
    } finally {
        $archive.Dispose()
    }
    $target = Join-Path $Local 'Sashimono\入れた 部品'
    # ネットにつながない 1 行の見本 CI で 1 秒ほど
    $pip = Invoke-Exe 'pip' @('-m', 'pip', 'install', '--no-index', '--target', $target, $wheel) 120
    if ($pip.ExitCode -eq 0 -and (Test-Path -LiteralPath (Join-Path $target "$sample\__init__.py"))) {
        Note "exe の pip で日本語のフォルダへ入れられた: $target"
    } else {
        Write-Host $pip.Out
        Write-Host $pip.Err
        Fail "exe の pip で入れられない（終了コード $($pip.ExitCode)）"
    }

    # --- 起動して窓を出し、閉じる ---

    $since = Get-Date
    $window =[System.Diagnostics.Process]::Start((New-StartInfo @() $false))
    $deadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $deadline -and -not $window.HasExited) {
        $window.Refresh()
        if ($window.MainWindowHandle -ne [IntPtr]::Zero) { break }
        Start-Sleep -Milliseconds 500
    }
    if ($window.HasExited) {
        Fail ('起動した exe が窓を出す前に終わった（終了コード {0:X8}）' -f $window.ExitCode)
    } elseif ($window.MainWindowHandle -eq [IntPtr]::Zero) {
        Fail '起動して 120 秒たっても窓が出ない'
        $window.Kill($true)
    } else {
        Start-Sleep -Seconds $WindowSeconds
        $window.Refresh()
        if ($window.HasExited) {
            Fail ('窓を出した後に落ちた（終了コード {0:X8}）' -f $window.ExitCode)
        } else {
            Note "窓が出た: $($window.MainWindowTitle)"
            if ($glFailures -gt 0) {
                # GL の無い機械では、プレビューの所に描けない理由を出す（main_window.py）
                $found = Test-GLNotice $window.MainWindowHandle.ToInt64()
                if ($found -eq 0) {
                    Note '窓に GL を使えない理由（OpenGL 4.3 を使えないため…）が出ている'
                } else {
                    Fail "GL の無い機械なのに、窓に描けない理由が出ていない（UI Automation の終了コード $found）"
                }
            }
            # 写真は手掛かりとして残すだけ 撮れなくても（画面の無い runner）落とさない
            try {
                $rect = [Sashimono.Native+Rect]::new()
                [void][Sashimono.Native]::GetWindowRect($window.MainWindowHandle, [ref]$rect)
                $width = $rect.Right - $rect.Left
                $height = $rect.Bottom - $rect.Top
                $bitmap = [System.Drawing.Bitmap]::new($width, $height)
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
                $bitmap.Save((Join-Path $Report 'window.png'), [System.Drawing.Imaging.ImageFormat]::Png)
                $graphics.Dispose()
                $bitmap.Dispose()
            } catch {
                Note "窓の写真を撮れない: $_"
            }
            if (-not $window.CloseMainWindow()) { Fail '窓へ閉じる知らせを送れない' }
            if (-not $window.WaitForExit(60000)) {
                Fail '閉じる知らせを送って 60 秒たっても終わらない（保存の確認などで止まっている）'
                $window.Kill($true)
            } elseif ($window.ExitCode -ne 0) {
                # 0xC0000005 のような値なら、閉じた後の片付けで落ちている（#149）
                Fail ('閉じた後の終了コードが {0:X8}（0 のはず）' -f $window.ExitCode)
            } else {
                Note '閉じて終了コード 0 で終わった'
            }
        }
    }
    $windowDrops = @(Get-Blocked $since 'sashimono.exe')
    # 起動の後に更新を確かめに行くのは正しい動き 止めても落ちずに動き続けたことを上で見ている
    # 起動の 3 秒後に新しい版を確かめに外へ出る 止められても落ちずに閉じられたことは上で見ている
    if ($script:Guarded) {
        Note "起動から閉じるまでに止めた外向きの接続: $($windowDrops.Count) 件（更新の確認）"
    }

    # --- exe の隣の scripts に置いた見本が、起動したときに %APPDATA% 側へ移ったか（Issue #244） ---
    # 既定は尋ねずに移す 移さないままだと、zip を手で展開し直した人の物が消える
    # 移した知らせは窓を塞がない（塞ぐと上の「閉じる知らせ」が届かずに落ちる）
    $movedSample = Join-Path $Roaming 'Sashimono\scripts\確かめる用.anm2'
    $leftSample = Join-Path $Scripts '確かめる用.anm2'
    if ((Test-Path -LiteralPath $movedSample) -and -not (Test-Path -LiteralPath $leftSample)) {
        Note "exe の隣の見本が %APPDATA% 側へ移った: $movedSample"
    } else {
        Fail 'exe の隣の scripts に置いた見本が、起動しても %APPDATA% 側へ移っていない'
    }

    # --- 実の置き場に書いていないか ---
    $written = @(Get-RealWrites $RealBefore @(Get-RealSnapshot $RealPlaces))
    if ($written.Count -gt 0) {
        Fail "渡した置き場ではなく、実の利用者の置き場に書いた: $(($written | Select-Object -First 10) -join ', ')"
    }
    $own = @(Get-ChildItem -LiteralPath $Roaming, $Local -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq 'Sashimono' })
    if ($own.Count -eq 0) {
        Fail '渡した日本語の置き場（APPDATA・LOCALAPPDATA）に設定のフォルダができていない'
    } else {
        Note "日本語の置き場に書いた: $(($own | ForEach-Object { $_.FullName }) -join ', ')"
    }
} finally {
    # 落ちても（throw でも）設定を戻し、規則を消す
    Restore-NetworkGuard
}

# --- 後から入れる部品（AI 連携・字幕起こし）を exe の pip で入れて読む ネットにつなぐ ---
# ファイアウォールの規則を戻した後で走らせる 前で走らせると pip が止められて落ち、外へ出ないことを
# 数える所にも pip の接続が混ざる 0.1.0 は CI でここを見ておらず、AI 連携が読めない zip を組んだ
# 入れ方と読み方は exe 自身が持つ（Sashimono.exe --add-on-check sashimono/addon_check.py）
# ここへ書き写すと、導入ボタンの入れ方が変わったときに確かめだけが古いまま残る
$addOns = Join-Path $Local 'Sashimono\後から 入れた 部品'
# exe の側が段（AI 連携の pip・字幕起こしの pip・import の 3 つ どれも exe の中で走る）を 1 つずつ待つ秒数
# CI では合わせて 40 秒ほど 時間切れなら exe が [NG] を書いて終わり、下で要約まで書ける
$AddOnStepSeconds = 300
# exe 全体を待つ秒数 段 3 つ分と、exe の起動の分
$AddOnSeconds = $AddOnStepSeconds * 3 + 60
try {
    $addOnCheck = Invoke-Exe 'add-on-check' @('--add-on-check', $addOns, "$AddOnStepSeconds") $AddOnSeconds
    foreach ($line in ($addOnCheck.Out -split "`r?`n")) {
        if ($line -match '^\[(ok|NG)\] ') { Note "後から入れる部品: $line" }
    }
    if ($addOnCheck.ExitCode -ne 0) {
        Write-Host $addOnCheck.Out
        Write-Host $addOnCheck.Err
        Fail "後から入れる部品を exe の pip で入れて読めない（終了コード $($addOnCheck.ExitCode)）"
    }
} catch {
    Fail "後から入れる部品の確かめが終わらない: $_"
}

# --- 自動更新の入れ替えを、本物の exe と Explorer から起こしたのと同じ作業場所で通す（#279） ---
# 自己診断の予行は使い捨ての見本で入れ替えるだけで、本物の exe が入れ替え係を起こし、入れた版が
# 窓を出すまでは通らない 0.2.0 までの版は、作業場所（インストール先）を受け継いだ入れ替え係が
# 自分で改名を断らせ、使う人の機械で毎回失敗した 試験と予行はその条件を作っていなかった
# 目録の署名は CI では本物の鍵が無いので、落として確かめ終えた後の形（.new と覚え書き）を
# 置いてから起こす 本人が〔次の起動で入れる〕を選んで起こし直したのと同じ道（apply_on_start）
# 書き出しの後に走らせる ファイアウォールの規則を戻した後なので、入れた版の更新の確認が外へ出ても
# 数えない（自己診断が外へ出ないことは上で見てある）
function Test-Swap([string]$Label, [string]$FromZip, [int]$TimeoutSeconds = 300) {
    # 返すのは [pscustomobject]@{ Passed; Detail } 落ちても止めずに理由を返す
    # Windows PowerShell 5.1 からも呼べる形で書く（手元で zip を組んで確かめるときに 5.1 で走らせる）
    $base = Join-Path $Root "入れ替え $Label"
    $swapLocal = Join-Path $base 'AppData\Local'
    $swapRoaming = Join-Path $base 'AppData\Roaming'
    $swapTemp = Join-Path $swapLocal 'Temp'
    foreach ($folder in @($swapLocal, $swapRoaming, $swapTemp)) {
        New-Item -ItemType Directory -Force $folder | Out-Null
    }
    Expand-Archive -LiteralPath $FromZip -DestinationPath $base
    $install = Join-Path $base 'Sashimono'
    $staged = "$install.new"
    $unpack = Join-Path $base '新しい版'
    Expand-Archive -LiteralPath $Zip -DestinationPath $unpack
    Move-Item -LiteralPath (Join-Path $unpack 'Sashimono') -Destination $staged
    # 入れるのは今の版より新しい版だけなので、新しい版の書き付けの版を上げる（中身は今組んだ物）
    $infoPath = Join-Path $staged 'build-info.json'
    $info = Get-Content -LiteralPath $infoPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $info.version = $SwapVersion
    # BOM を付けない 本体は書き付けを BOM の無い UTF-8 として読む
    $noBom = New-Object System.Text.UTF8Encoding $false
    [IO.File]::WriteAllText($infoPath, ($info | ConvertTo-Json), $noBom)
    $updateDir = Join-Path $swapLocal 'Sashimono\update'
    New-Item -ItemType Directory -Force $updateDir | Out-Null
    $state = [ordered]@{ ready_version = $SwapVersion; apply_on_start = $true; apply_chosen = $true }
    [IO.File]::WriteAllText((Join-Path $updateDir 'state.json'), ($state | ConvertTo-Json), $noBom)

    $start = New-Object System.Diagnostics.ProcessStartInfo (Join-Path $install 'Sashimono.exe')
    # Explorer やショートカットから起こしたのと同じく、作業場所をインストール先にする
    $start.WorkingDirectory = $install
    $start.UseShellExecute = $false
    $start.Environment.Clear()
    foreach ($entry in $Environment.GetEnumerator()) { $start.Environment[$entry.Key] = $entry.Value }
    $start.Environment['APPDATA'] = $swapRoaming
    $start.Environment['LOCALAPPDATA'] = $swapLocal
    $start.Environment['TEMP'] = $swapTemp
    $start.Environment['TMP'] = $swapTemp
    $lines = New-Object 'System.Collections.Generic.List[string]'
    $detail = ''
    $passed = $false
    try {
        $first = [System.Diagnostics.Process]::Start($start)
        # 起こした版は入れ替え係を起こし、走り始めたのを見て終わる（swap.START_SECONDS まで待つ）
        if (-not $first.WaitForExit(120000)) {
            $detail = '起こした版が入れ替え係を起こさずに 120 秒動き続けた（覚え書きを読んでいない）'
        } else {
            # 入れ替え係が終わる（錠が空く）まで待つ 結果のファイルは、入れ替え係が起こし直した
            # 版が読んで消すことがあるので、待つ間に読み続けて控える
            $lock = Join-Path $updateDir 'swap.lock'
            $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
            $finished = $false
            while ((Get-Date) -lt $deadline) {
                foreach ($file in @(Get-ChildItem -LiteralPath $updateDir -Filter 'result-*.txt' -ErrorAction SilentlyContinue)) {
                    try {
                        foreach ($line in [IO.File]::ReadAllLines($file.FullName)) {
                            $text = $line.Trim()
                            if ($text -and -not $lines.Contains($text)) { $lines.Add($text) }
                        }
                    } catch { }
                }
                if (-not (Test-Path -LiteralPath $lock)) { $finished = $true; break }
                try {
                    $held = [IO.File]::Open($lock, 'Open', 'ReadWrite', 'None')
                    $held.Dispose()
                    $finished = $true
                    break
                } catch {
                    Start-Sleep -Milliseconds 500
                }
            }
            if (-not $finished) { $detail = "入れ替え係が $TimeoutSeconds 秒で終わらない" }
        }
        $now = Get-Content -LiteralPath (Join-Path $install 'build-info.json') -Raw -Encoding UTF8 -ErrorAction SilentlyContinue | ConvertFrom-Json
        $passed = (-not $detail) -and $lines.Contains('healthy') -and $now -and ($now.version -eq $SwapVersion)
        if (-not $detail) {
            $detail = '結果: ' + ($lines -join ' ') + ' / 入れた後のインストール先の版: ' + $(if ($now) { $now.version } else { '読めない' })
        }
    } catch {
        $passed = $false
        $detail = "確かめの途中で止まった: $_"
    } finally {
        # 起こし直された版（入れた版か、失敗して起こし直された前の版）を閉じる 残すと、後の
        # 片付けと次の確かめが、そのフォルダを消せない
        $prefix = $base.TrimEnd('\') + '\'
        Get-Process -ErrorAction SilentlyContinue | Where-Object {
            try { $_.Path -and $_.Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) } catch { $false }
        } | ForEach-Object { try { $_.Kill(); [void]$_.WaitForExit(30000) } catch { } }
    }
    return [pscustomobject]@{ Passed = [bool]$passed; Detail = $detail }
}

# 新しい版の書き付けに書く版 本物の版と取り違えない大きな数（update/rehearsal.py の見本と同じ）
$SwapVersion = '9999.0.0'

# 1 つ前に公開した版 → 今組んだ版 を必ず通す扱いにするかを決める所（ここだけ）
# 0.2.0 までの版は、作業場所を受け継いだ入れ替え係が自分で改名を断らせるので、この条件では
# 必ず失敗する（#279） 失敗を記録して知らせるだけにする 入れ替え係の台本を持つ版（書き付けに
# swap_contract がある 0.2.1 から）が 1 つ前の公開版になったら、必ず通す扱いへ自動で切り替わる
# 書き付けを読めなくても、zip の名前の版が 0.2.1 以上なら必ず通す（読めないだけで警告に下げない）
$ScriptSince = [version]'0.2.1'
function Test-PreviousMustPass($PreviousInfo, [string]$ZipVersion) {
    if ($null -ne $PreviousInfo -and $null -ne $PreviousInfo.swap_contract) { return $true }
    if ($ZipVersion -match '^(\d+)\.(\d+)\.(\d+)') {
        return [version]"$($Matches[1]).$($Matches[2]).$($Matches[3])" -ge $ScriptSince
    }
    # 名前からも版が分からない 分からない物を警告に下げると、必須の確かめが黙って抜ける
    return $true
}

function Read-ZipBuildInfo([string]$Path) {
    # zip の中の書き付けを読む 無い・読めない・版が無いときは例外（呼ぶ側が理由を残して落とす）
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $opened = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $entry = $opened.GetEntry('Sashimono/build-info.json')
        if ($null -eq $entry) { throw 'zip に Sashimono/build-info.json が無い' }
        $reader = New-Object IO.StreamReader ($entry.Open())
        try { $info = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    } finally {
        $opened.Dispose()
    }
    if ($null -eq $info -or -not $info.version) { throw 'build-info.json に版が無い' }
    return $info
}

function Get-ZipVersion([string]$Path) {
    # 配る zip の名前（SashimonoEdit-<版>-windows-x64.zip tools/build_package.py）から版を読む
    $name = [IO.Path]::GetFileName($Path)
    if ($name -match '^SashimonoEdit-(.+)-windows-x64\.zip$') { return $Matches[1] }
    return ''
}

function Get-PreviousZip {
    # 1 つ前に公開した版（GitHub Releases の latest）の zip CI の runner でだけ落とす
    # 手元では -PreviousZip に渡した物を使い、渡さなければ飛ばす
    if ($PreviousZip) { return $PreviousZip }
    if ($env:GITHUB_ACTIONS -ne 'true' -or -not $env:GH_TOKEN -or -not $env:GITHUB_REPOSITORY) { return $null }
    $folder = Join-Path ([IO.Path]::GetTempPath()) 'sashimono-previous-release'
    if (Test-Path -LiteralPath $folder) { Remove-Item -LiteralPath $folder -Recurse -Force }
    New-Item -ItemType Directory -Path $folder | Out-Null
    & gh release download --repo $env:GITHUB_REPOSITORY --pattern 'SashimonoEdit-*-windows-x64.zip' --dir $folder 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) { return $null }
    $found = @(Get-ChildItem -LiteralPath $folder -Filter 'SashimonoEdit-*-windows-x64.zip')
    if ($found.Count -ne 1) { return $null }
    return $found[0].FullName
}

try {
    # 今組んだ版 → 今組んだ版 必ず通す この版の入れ替え係と、入れ替え係に起こされた版の起動を見る
    $selfSwap = Test-Swap '今の版から' $Zip
    if ($selfSwap.Passed) {
        Note "作業場所をインストール先にして、今組んだ版から今組んだ版へ入れ替わった（$($selfSwap.Detail)）"
    } else {
        Fail "作業場所をインストール先にした入れ替えが通らない（今組んだ版から）: $($selfSwap.Detail)"
    }

    $previous = Get-PreviousZip
    if (-not $previous) {
        Note '1 つ前の公開版の zip が無いので、公開版からの入れ替えを飛ばした（CI の runner でだけ落とす）'
    } else {
        $previousInfo = $null
        $zipVersion = Get-ZipVersion $previous
        try {
            $previousInfo = Read-ZipBuildInfo $previous
        } catch {
            # 0.1.0 から配った zip はどれも書き付けを持つ 読めないのは zip か落とし方の異常で、
            # 黙ると必須の確かめが警告に下がりうる 理由を残して落とす
            Fail "公開版の zip（$([IO.Path]::GetFileName($previous))）の build-info.json を読めない: $_"
        }
        $previousVersion = if ($previousInfo) { $previousInfo.version } elseif ($zipVersion) { $zipVersion } else { '版が読めない' }
        $previousSwap = Test-Swap '公開版から' $previous
        if ($previousSwap.Passed) {
            Note "作業場所をインストール先にして、公開版 $previousVersion から今組んだ版へ入れ替わった（$($previousSwap.Detail)）"
        } elseif (Test-PreviousMustPass $previousInfo $zipVersion) {
            Fail "公開版 $previousVersion から今組んだ版へ入れ替わらない（作業場所はインストール先）: $($previousSwap.Detail)"
        } else {
            Write-Host "::warning::公開版 $previousVersion から今組んだ版へ入れ替わらない（この版の入れ替え係の不具合 #279 で分かっている）: $($previousSwap.Detail)"
            Note "公開版 $previousVersion からの入れ替えは失敗した（#279 で分かっている古い版の不具合 落とさずに記録だけ）: $($previousSwap.Detail)"
        }
    }
} catch {
    Fail "自動更新の入れ替えの確かめが終わらない: $_"
}

# --- 結果 ---
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add('## まっさらな Windows で zip から起動する（Issue #33）')
$summary.Add('')
$summary.Add("展開先: ``$AppHome``")
$summary.Add('')
$summary.Add('| 結果 | 項目 | 詳しく |')
$summary.Add('|---|---|---|')
foreach ($entry in $items.GetEnumerator()) {
    $detail = $entry.Value.Detail -replace '\|', '\|'
    $summary.Add("| $($entry.Value.Mark) | $($entry.Key) | $detail |")
}
$summary.Add('')
foreach ($note in $script:Notes) { $summary.Add("- $note") }
if ($script:Failures.Count -gt 0) {
    $summary.Add('')
    $summary.Add("### 落ちた所 $($script:Failures.Count) 件")
    foreach ($failure in $script:Failures) { $summary.Add("- $failure") }
} else {
    $summary.Add('')
    $summary.Add('すべて確かめた')
}
Set-Content -LiteralPath (Join-Path $Report 'summary.md') -Encoding utf8 -Value $summary
if ($env:GITHUB_STEP_SUMMARY) {
    Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Encoding utf8 -Value $summary
}

if ($script:Failures.Count -gt 0) { exit 1 }
exit 0
