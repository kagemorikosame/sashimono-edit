"""AviUtl2 と比べる道具（tools/aviutl_compare.py）の、合成フォントの下ごしらえ

合成フォント（``comfont.aux2``）は設定（``compositefont\\profiles.json``）が無いと
本文をそのまま返し、書体の組み替えが起きない 道具は見本の設定を作業フォルダへ書き、
AviUtl2 の置き場へ写す手順を出す **利用者の設定へは書きに行かない**

実物のプラグインを使う試験は、入っている機械でだけ走る（CI では飛ぶ）
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sashimono.compat.aviutl import plugin

ROOT = Path(__file__).resolve().parent.parent

#: 実物（comfont.aux2 v0.2.0）が受け付けた 1 文字種ぶんの項目
#: 臨時の置き場に置いたとき ``resolve("あ", "default")`` が書いた書体を返した
_ACCEPTED_ADJUSTMENT = {
    "font_family",
    "fallback_font_families",
    "size_ratio",
    "baseline_shift_em",
    "tracking_adjust_em",
    "metric_unit",
    "size_px",
    "baseline_shift_px",
    "tracking_adjust_px",
    "vertical_scale_ratio",
    "horizontal_scale_ratio",
}
_ACCEPTED_KINDS = ("western", "hiragana", "katakana", "kanji", "digit", "symbol", "other")


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aviutl_compare", ROOT / "tools" / "aviutl_compare.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile(document: dict[str, Any]) -> dict[str, Any]:
    profiles = document["profiles"]
    assert isinstance(profiles, list) and len(profiles) == 1
    profile: dict[str, Any] = profiles[0]
    return profile


class TestSampleProfile:
    def test_has_the_fields_the_real_plugin_accepted(self, tool: ModuleType) -> None:
        """項目が欠けたり名前が違ったりすると、実物は設定を読まずに本文をそのまま返し、
        組み替えの起きない絵を AviUtl2 の絵と比べることになる
        """
        document = tool.composite_profiles()
        assert document["schema_version"] == 1
        profile = _profile(document)
        expected = {"name", *_ACCEPTED_KINDS, *(f"{kind}_fallbacks" for kind in _ACCEPTED_KINDS)}
        assert set(profile) == expected
        for kind in _ACCEPTED_KINDS:
            assert set(profile[kind]) == _ACCEPTED_ADJUSTMENT

    def test_uses_the_profile_the_aliases_ask_for(self, tool: ModuleType) -> None:
        """配布エイリアスは ``profile="default"`` を引く 名前が違うと組み替えが起きない"""
        assert _profile(tool.composite_profiles())["name"] == "default"

    def test_neighbouring_kinds_get_different_fonts(self, tool: ModuleType) -> None:
        """同じ書体が並ぶと、組み替えが起きたかどうかを絵で見分けられない"""
        profile = _profile(tool.composite_profiles())
        japanese = {profile[kind]["font_family"] for kind in ("hiragana", "katakana", "kanji")}
        assert len(japanese) == 3
        assert profile["western"]["font_family"] not in japanese

    def test_avoids_the_default_font(self, tool: ModuleType) -> None:
        """Sashimono の既定（Yu Gothic UI）を混ぜると、組み替えずに既定で描いた絵と
        区別が付かない
        """
        profile = _profile(tool.composite_profiles())
        assert all(profile[kind]["font_family"] != "Yu Gothic UI" for kind in _ACCEPTED_KINDS)

    def test_only_the_font_changes(self, tool: ModuleType) -> None:
        """大きさや位置まで動かすと、絵の違いが書体から来たのか見分けられない"""
        profile = _profile(tool.composite_profiles())
        for kind in _ACCEPTED_KINDS:
            adjustment = profile[kind]
            assert adjustment["size_ratio"] == 1.0
            assert adjustment["baseline_shift_em"] == 0.0
            assert adjustment["tracking_adjust_em"] == 0.0

    def test_missing_fonts_are_named(
        self, tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """無い書体を指すと、AviUtl2 も Sashimono も既定の書体で描き、組み替えが見えない

        その PC に書体が入っているかは機械ごとに違う（CI の英語版 Windows には
        游明朝もメイリオも無い） 入っているかそのものではなく、足りない物を
        言い当てられるかを偽の Fonts フォルダで確かめる 言い当てられれば、
        `profile` がその場で警告を出す
        """
        fonts = tmp_path / "Fonts"
        fonts.mkdir()
        (_key, first, file), *rest = tool.COMPOSITE_FONTS
        (fonts / file).write_bytes(b"")
        monkeypatch.setenv("WINDIR", str(tmp_path))
        assert tool.missing_fonts() == [family for _k, family, _f in rest]
        assert first not in tool.missing_fonts()


class TestProfileCommand:
    def _run(self, tool: ModuleType, work: Path) -> None:
        assert tool.command_profile(SimpleNamespace(work=work)) == 0

    def test_writes_only_into_the_work_folder(
        self, tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """利用者の置き場へ書くと、合成フォントのエディターで作った設定を黙って潰す"""
        program_data = tmp_path / "ProgramData"
        (program_data / "aviutl2").mkdir(parents=True)
        monkeypatch.setenv("PROGRAMDATA", str(program_data))
        work = tmp_path / "work"
        self._run(tool, work)
        sample = work / "appdata" / "compositefont" / "profiles.json"
        assert json.loads(sample.read_text(encoding="utf-8")) == tool.composite_profiles()
        assert not (program_data / "aviutl2" / "compositefont").exists()

    def test_an_existing_user_setting_is_left_alone(
        self, tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """控えも取らずに上書きすると、本人の設定が戻らなくなる"""
        program_data = tmp_path / "ProgramData"
        target = program_data / "aviutl2" / "compositefont" / "profiles.json"
        target.parent.mkdir(parents=True)
        target.write_text('{"本人": 1}', encoding="utf-8")
        monkeypatch.setenv("PROGRAMDATA", str(program_data))
        self._run(tool, tmp_path / "work")
        assert target.read_text(encoding="utf-8") == '{"本人": 1}'
        assert not tool.backup_of(target).exists()

    def test_tells_where_to_put_it(
        self,
        tool: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """置き場を示さないと、本人は AviUtl2 のどこへ置けば効くのか分からない"""
        program_data = tmp_path / "ProgramData"
        monkeypatch.setenv("PROGRAMDATA", str(program_data))
        self._run(tool, tmp_path / "work")
        out = capsys.readouterr().out
        target = program_data / "aviutl2" / "compositefont" / "profiles.json"
        assert str(target) in out
        # 元の設定が無いのに控えを取らせると、無いファイルを写す命令で止まる
        assert f'Copy-Item "{target}"' not in out

    def test_asks_for_a_backup_when_one_exists(self, tool: ModuleType, tmp_path: Path) -> None:
        """既にある設定へそのまま写させると、本人の組み合わせが消える"""
        target = tmp_path / "compositefont" / "profiles.json"
        target.parent.mkdir()
        target.write_text("{}", encoding="utf-8")
        sample = tmp_path / "見本.json"
        lines = tool.profile_guide(sample, target)
        assert any("控え" in line for line in lines)
        backup = next(i for i, line in enumerate(lines) if str(tool.backup_of(target)) in line)
        copy = next(
            i for i, line in enumerate(lines) if f"Copy-Item -LiteralPath '{sample}'" in line
        )
        # 控えを取る行が、写す行より先に来ること 後だと上書きした物の控えになる
        assert backup < copy


#: 手順のうち、本人がそのまま打つ命令の行
_COMMANDS = ("Copy-Item", "New-Item", "Move-Item", "Remove-Item")

#: 試験の間ずっと開いておく PowerShell が回す輪 1 行に 1 つ、UTF-16 の base64 で命令を受け、
#: 打てたら ``ok``、打てなければ ``error`` と理由（UTF-8 の base64）を返す
#: 行き来を ASCII だけにするのは、日本語のパスを英語版 Windows（CI）の OEM の符号でも
#: 化けさせずに渡すため ``Stop`` にするのは、途中の 1 つが打てずに後ろだけが通っても
#: 「なぞれた」と読まないため（前は ``;`` でつないだ最後の命令の成否しか見ていなかった）
_LOOP = """
$ErrorActionPreference = 'Stop'
[Console]::Out.WriteLine('ready')
[Console]::Out.Flush()
while ($null -ne ($line = [Console]::In.ReadLine())) {
    try {
        $text = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($line))
        Invoke-Expression $text | Out-Null
        [Console]::Out.WriteLine('ok')
    } catch {
        $reason = [Text.Encoding]::UTF8.GetBytes($_.ToString())
        [Console]::Out.WriteLine('error ' + [Convert]::ToBase64String($reason))
    }
    [Console]::Out.Flush()
}
"""

#: PowerShell が立ち上がって輪に入るまでの上限（秒） 手元では 0.2 秒ほどだが、CI の遅い機械で
#: 並べて走らせると、1 本を立ち上げるだけで 60 秒を超えたことがある（#406） 立ち上げは
#: 試験のファイルで 1 回だけなので、長く取っても速い機械の待ちは変わらない
_START_TIMEOUT = 300.0
#: 立ち上がった後の 1 回の命令（小さなファイルの写しと消し）の上限（秒） 返ればすぐ戻る
_COMMAND_TIMEOUT = 60.0


class _PowerShell:
    """命令を打つ PowerShell 1 本 試験のファイルの間ずっと使い回す

    前は手順を 1 回なぞるたびに powershell.exe を立ち上げていた 試験 5 本で 13 回になり、
    命令そのもの（小さなファイルの写し）より立ち上げの方がずっと重い 遅い機械ではその
    立ち上げが 60 秒の上限を超えて落ちた（#406） 1 本を使い回せば立ち上げは 1 回で済む

    標準入力は自分の管にする 渡さないと xdist の働き手の標準入力（試験を配る管）を
    PowerShell が受け継ぐ
    """

    def __init__(self) -> None:
        script = base64.b64encode(_LOOP.encode("utf-16-le")).decode("ascii")
        self._process = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="ascii",
            errors="replace",
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        #: 輪の決まりに無い行（PowerShell が自分で出した警告など） 落ちたときに添える
        self._noise: list[str] = []
        #: 上限で諦めたか、途中で終わった 立ったら打たない（:meth:`_give_up`）
        self._broken = False
        # 読むのは別のスレッド 読みながら上限で諦めるには、塞がる読みを試験の外へ出すしかない
        threading.Thread(target=self._read, daemon=True).start()
        # 立ち上げで諦めたときも _expect が止める 落ちると fixture の close まで届かない
        self._expect("ready", _START_TIMEOUT)

    def _read(self) -> None:
        stdout = self._process.stdout
        assert stdout is not None
        for line in stdout:
            self._lines.put(line.rstrip("\r\n"))
        self._lines.put(None)

    def _expect(self, what: str, timeout: float) -> str:
        """``what`` で始まる行か ``error`` の行を待つ"""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                line = self._lines.get(timeout=max(0.0, remaining))
            except queue.Empty:
                self._give_up()
                pytest.fail(f"PowerShell が {timeout:g} 秒で {what} を返さなかった {self._noise}")
            if line is None:
                self._give_up()
                pytest.fail(f"PowerShell が途中で終わった {self._noise}")
            if line.startswith((what, "error")):
                return line
            self._noise.append(line)

    def _give_up(self) -> None:
        """この 1 本をもう使わない 止めておく

        1 本を試験のファイルで使い回すので、諦めた命令の遅れた ``ok`` が残ると、次の命令の
        答えとして受け取り、打てていない手順を「なぞれた」と読む 止まったまま残すと、
        次の試験が上限まで待つ
        """
        self._broken = True
        self._process.kill()

    def run(self, command: str, timeout: float = _COMMAND_TIMEOUT) -> None:
        """命令を 1 つ打ち、打てたことを確かめる"""
        if self._broken:
            pytest.fail(f"前の命令で PowerShell が応答しなくなったので打たない {self._noise}")
        stdin = self._process.stdin
        assert stdin is not None
        stdin.write(base64.b64encode(command.encode("utf-16-le")).decode("ascii") + "\n")
        stdin.flush()
        answer = self._expect("ok", timeout)
        if answer.startswith("error"):
            reason = base64.b64decode(answer.removeprefix("error ")).decode("utf-8")
            pytest.fail(f"打てなかった: {command}\n{reason}")

    def close(self) -> None:
        stdin = self._process.stdin
        if stdin is not None:
            # 諦めて止めた後は管がもう切れている 閉じるときの書き損じは捨てる
            with contextlib.suppress(OSError):
                stdin.close()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


@pytest.fixture(scope="module")
def powershell() -> Iterator[_PowerShell]:
    session = _PowerShell()
    yield session
    session.close()


def _follow(shell: _PowerShell, lines: list[str], part: str) -> None:
    """手順の命令を、書かれた順に PowerShell で打つ ``part`` は置く側か戻す側か

    文言を見るだけでは、打てない命令や順番の誤りを見逃す 実際に打って結果を見る
    """
    split = next(i for i, line in enumerate(lines) if line.startswith("比べ終わったら"))
    chosen = lines[:split] if part == "place" else lines[split:]
    for line in chosen:
        if line.strip().startswith(_COMMANDS):
            shell.run(line.strip())


@pytest.mark.skipif(sys.platform != "win32", reason="手順は PowerShell の命令で出す")
def test_a_session_that_timed_out_is_not_reused() -> None:
    """壊れると、上限で諦めた命令の遅れた ``ok`` を次の命令の答えとして受け取り、打てて
    いない手順を「なぞれた」と読む 止まった PowerShell も残り、次の試験が上限まで待つ

    試験のファイルで使い回す 1 本とは別に立ち上げ、上限より長くかかる命令で順番を決め打ちにする
    """
    session = _PowerShell()
    try:
        with pytest.raises(pytest.fail.Exception, match="返さなかった"):
            session.run("Start-Sleep -Seconds 3", timeout=0.5)
        # 諦めた時に止めてあること 止めないと、遅れた ok が次の命令の答えになる
        assert session._process.wait(timeout=30) is not None
        started = time.monotonic()
        with pytest.raises(pytest.fail.Exception, match="応答しなくなった"):
            session.run("New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null")
        # 次の命令は上限（60 秒）まで待たずに断られる
        assert time.monotonic() - started < 10.0
    finally:
        session.close()


@pytest.mark.skipif(sys.platform != "win32", reason="手順は PowerShell の命令で出す")
class TestFollowingTheGuide:
    """出た手順をそのままなぞって、置いて戻せるか"""

    @pytest.fixture
    def sample(self, tmp_path: Path) -> Path:
        path = tmp_path / "work" / "profiles.json"
        path.parent.mkdir()
        path.write_text('{"見本": 1}', encoding="utf-8")
        return path

    @pytest.fixture
    def target(self, tmp_path: Path) -> Path:
        return tmp_path / "aviutl2" / "compositefont" / "profiles.json"

    def test_a_folder_with_brackets_quotes_and_dollars_still_works(
        self, tool: ModuleType, powershell: _PowerShell, tmp_path: Path
    ) -> None:
        """二重引用符と ``-Path`` では、``[`` ``]`` がワイルドカード、``$`` が変数として読まれ、
        見本も控えも見つからないまま置く手順も戻す手順も失敗する
        """
        odd = tmp_path / "作業[2] $HOME 'x'"
        sample = odd / "work" / "profiles.json"
        sample.parent.mkdir(parents=True)
        sample.write_text('{"見本": 1}', encoding="utf-8")
        target = odd / "aviutl2" / "compositefont" / "profiles.json"
        target.parent.mkdir(parents=True)
        target.write_text('{"元": 1}', encoding="utf-8")
        lines = tool.profile_guide(sample, target)
        _follow(powershell, lines, "place")
        assert target.read_text(encoding="utf-8") == '{"見本": 1}'
        _follow(powershell, lines, "restore")
        assert target.read_text(encoding="utf-8") == '{"元": 1}'

    def test_restoring_brings_back_the_original(
        self, tool: ModuleType, powershell: _PowerShell, sample: Path, target: Path
    ) -> None:
        """戻す手順が無いと、比べ終わったあとも見本の組み替えが普段の AviUtl2 に残る"""
        target.parent.mkdir(parents=True)
        target.write_text('{"本人": 1}', encoding="utf-8")
        lines = tool.profile_guide(sample, target)
        _follow(powershell, lines, "place")
        assert target.read_text(encoding="utf-8") == '{"見本": 1}'
        _follow(powershell, lines, "restore")
        assert target.read_text(encoding="utf-8") == '{"本人": 1}'
        assert not tool.backup_of(target).exists()

    def test_restoring_removes_the_sample_when_there_was_nothing(
        self, tool: ModuleType, powershell: _PowerShell, sample: Path, target: Path
    ) -> None:
        """元が無かったのに見本を残すと、置く前に無かった組み替えが居座る"""
        lines = tool.profile_guide(sample, target)
        _follow(powershell, lines, "place")
        assert target.read_text(encoding="utf-8") == '{"見本": 1}'
        _follow(powershell, lines, "restore")
        assert not target.exists()

    def test_following_it_twice_keeps_the_original_backup(
        self, tool: ModuleType, powershell: _PowerShell, sample: Path, target: Path
    ) -> None:
        """もう一度なぞると、置いたままの見本を「元の設定」として控えに上書きし、
        本人の設定へ戻せなくなる
        """
        target.parent.mkdir(parents=True)
        target.write_text('{"本人": 1}', encoding="utf-8")
        _follow(powershell, tool.profile_guide(sample, target), "place")
        again = tool.profile_guide(sample, target)
        assert any("見本が置かれたまま" in line for line in again)
        _follow(powershell, again, "place")
        assert tool.backup_of(target).read_text(encoding="utf-8") == '{"本人": 1}'
        _follow(powershell, again, "restore")
        assert target.read_text(encoding="utf-8") == '{"本人": 1}'

    def test_following_it_twice_without_an_original_still_cleans_up(
        self, tool: ModuleType, powershell: _PowerShell, sample: Path, target: Path
    ) -> None:
        """2 度目の手順が控えを戻そうとすると、無い控えを探して止まり見本が残る"""
        _follow(powershell, tool.profile_guide(sample, target), "place")
        again = tool.profile_guide(sample, target)
        _follow(powershell, again, "place")
        _follow(powershell, again, "restore")
        assert not target.exists()


def test_previews_of_aliases_with_the_same_name_do_not_overwrite(
    tool: ModuleType, tmp_path: Path
) -> None:
    """名前だけで絵を書くと、別のフォルダの同じ名前のエイリアスが先の絵を上書きし、
    描いたはずの 1 本が黙って消える
    """
    alias = "[Object]\nframe=0,29\n[Object.0]\neffect.name=図形\n"
    files = []
    for folder in ("甲", "乙"):
        path = tmp_path / folder / "同じ名前.object"
        path.parent.mkdir()
        path.write_text(alias, encoding="utf-8")
        files.append(path)
    cases, _ = tool.build_cases(files)
    assert len(cases) == 2
    assert len({tool.preview_name(case) for case in cases}) == 2


def test_an_alias_whose_picture_is_missing_is_left_out(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """素材の見つからないエイリアスは描かずに外し、どの素材が無いかを出す（#183 のレビュー）

    素材の無い絵を見本に残すと、描けたものと取り違える 比べる側では差が素材の欠けを
    測っただけの数になり、平均を汚す
    """
    lost = tmp_path / "無い画像.png"
    path = tmp_path / "画像.object"
    path.write_text(
        f"[Object]\nframe=0,29\n[Object.0]\neffect.name=画像ファイル\nファイル={lost}\n",
        encoding="utf-8",
    )
    arguments = SimpleNamespace(work=tmp_path / "work", files=[str(path)], app_data=None)
    assert tool.command_preview(arguments) == 0
    assert not any((tmp_path / "work" / "preview").iterdir())
    output = capsys.readouterr().out
    assert "素材が見つからないので比べない" in output
    assert str(lost) in output


class TestAppDataPath:
    @pytest.fixture(autouse=True)
    def reset(self) -> Any:
        yield
        plugin.set_app_data_path(None)

    def test_override_reaches_the_plugin(self, tmp_path: Path) -> None:
        """差し替えが届かないと、見本を試すには利用者の設定を書き換えるしかなくなる"""
        plugin.set_app_data_path(tmp_path)
        assert plugin._config().app_data_path == str(tmp_path)

    def test_without_override_the_users_folder_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """既定が変わると、利用者が AviUtl2 で作った書体の組み合わせが見えなくなる"""
        monkeypatch.setenv("PROGRAMDATA", str(tmp_path))
        plugin.set_app_data_path(tmp_path / "一時")
        plugin.set_app_data_path(None)
        assert plugin._config().app_data_path == str(tmp_path / "aviutl2")


def test_the_real_plugin_composes_with_the_sample(
    tool: ModuleType, tmp_path: Path, real_comfont: Path
) -> None:
    """見本の設定を実物が読まないと、組み替えの無い絵を AviUtl2 と比べることになる

    プラグインは読んだときの置き場を持ち続けるので、別のプロセスで読ませる
    同じプロセスで先に読まれていると、見本の置き場が届かない 渡すのは実物だけを
    写した置き場 本人の ``Plugin`` フォルダを渡すと、別のプロセスの中でほかの
    汎用プラグインまで初期化して、本人の置き場へ書かせる（Issue #135）
    """
    tool.command_profile(SimpleNamespace(work=tmp_path))
    script = (
        "import sys, json\n"
        "from pathlib import Path\n"
        "from sashimono.compat.aviutl import plugin\n"
        "plugin.set_app_data_path(Path(sys.argv[1]))\n"
        "module = plugin.script_modules((Path(sys.argv[2]),))['compositefont']\n"
        "text = module.call('decorate_layout', ['あア亜A1', 'default', 64.0, 0.0, ''])\n"
        "print(json.dumps(text, ensure_ascii=False))\n"
    )
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "appdata"), str(real_comfont)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=60,
        check=True,
    )
    decorated = json.loads(result.stdout.strip().splitlines()[-1])[0]
    for _kind, family, _file in tool.COMPOSITE_FONTS[:5]:
        assert f"<@{family}>" in decorated


class TestSavingPictures:
    def test_a_failed_save_is_reported(self, tool: ModuleType, tmp_path: Path) -> None:
        """Qt は保存に失敗しても例外を投げない 見ないと「書いた」と出したのに絵が無い"""
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        assert tool._save_png(image, tmp_path / "書ける.png") is True
        assert tool._save_png(image, tmp_path / "無いフォルダ" / "書けない.png") is False

    def test_a_long_alias_name_still_fits(self, tool: ModuleType) -> None:
        """番号と拡張子を足すとファイル名の上限を超え、絵が作られない"""
        case = SimpleNamespace(start=12, name="長" * 300)
        name = tool.preview_name(case)
        assert len(name) <= 255
        assert name.startswith("000012_")

    def test_a_name_of_wide_characters_still_fits(self, tool: ModuleType) -> None:
        """絵文字は UTF-16 で 2 単位 文字数で切ると NTFS の上限を超えて絵が作られない"""
        case = SimpleNamespace(start=12, name="😀" * 200)
        name = tool.preview_name(case)
        assert len(name.encode("utf-16-le")) // 2 <= 255
        # 文字の途中（サロゲートの片割れ）で切ると、名前として書けない
        name.encode("utf-8")


def test_an_alias_with_intermediate_points_keeps_its_own_length(tool: ModuleType) -> None:
    """中間点のある ``frame=244,333,423`` を読めずに既定の長さへ倒すと、AviUtl2 側と
    違う長さの区間に並べ、終わり際の比べるフレームが区間の外へ出る
    """
    assert tool._own_length(["frame=244,333,423"]) == 180
    assert tool._own_length(["frame=0,179"]) == 180
