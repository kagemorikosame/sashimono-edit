"""AviUtl2 に書き出させる道具（tools/aviutl2_export.py と aviutl2_export.ps1）と見本を作る道具

AviUtl2 そのものは起動しない（本人の画面を取り合うので、実機で確かめるのは人が頼んだときだけ）
ここで見るのは、走らせてよいかの判断・書き出される枚数の数え方・設定の控えと戻し・
PowerShell へ渡す命令・スクリプトが読めること・見本の形・連番の PNG の読み方
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "aviutl2_export.ps1"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load("aviutl2_export")


@pytest.fixture(scope="module")
def probes() -> ModuleType:
    return _load("aviutl_tag_probes")


@pytest.fixture(scope="module")
def compare() -> ModuleType:
    return _load("aviutl_compare")


class TestFrames:
    def test_counts_to_the_end_of_the_last_object(self, tool: ModuleType) -> None:
        # 数え違うと、揃うのを待ち続けるか、書き終える前に止めてしまう
        # 並べた 354 フレームのプロジェクトで frame000〜frame353 が出た
        text = "[project]\n[scene.0]\nvideo.rate=60\n[0]\nframe=0,5\n[0.0]\nframe=900,999\n"
        text += "[1]\nlayer=0\nframe=342,353\n"
        assert tool.project_frames(text) == 354

    def test_filter_sections_do_not_count(self, tool: ModuleType) -> None:
        # [0.1] のようなフィルタの節の値まで拾うと、書き出されない枚数を待つ
        assert tool.project_frames("[0]\nframe=0,9\n[0.1]\nframe=0,500\n") == 10

    def test_intermediate_points_are_read_to_the_end(self, tool: ModuleType) -> None:
        # 中間点のあるオブジェクトは frame=開始,中間点,終了 と並ぶ（AviUtl2 に作らせた
        # kumiki_motion_probe.object は frame=244,333,423） 2 つ目を終わりと読むと
        # 334 枚を待ち、424 枚書き出されたところで多すぎるとして失敗する
        text = "[0]\nframe=0,100\n[1]\nframe=244,333,423\n"
        assert tool.project_frames(text) == 424

    def test_the_real_alias_with_intermediate_points_is_counted_to_its_end(
        self, tool: ModuleType
    ) -> None:
        # 見本は手元にだけある（tests/fixtures/aviutl は .object を入れない） 無い機械では
        # 上の試験が同じ並びを受け持つ
        probe = ROOT / "tests" / "fixtures" / "aviutl" / "probes" / "kumiki_motion_probe.object"
        if not probe.is_file():
            pytest.skip("AviUtl2 に作らせた中間点の見本が手元に無い")
        header = probe.read_text(encoding="utf-8-sig").replace("[Object]", "[0]", 1)
        assert tool.project_frames(header) == 424

    def test_an_empty_project_has_no_frames(self, tool: ModuleType) -> None:
        # 1 枚以上と数えると、何も書き出されないのに AviUtl2 を起こし、来ない PNG を
        # 待ち時間いっぱいまで待つ 0 枚なら起こす前に断る
        assert tool.project_frames("[project]\nversion=2010601\n") == 0


class TestSettings:
    def test_restores_what_aviutl2_rewrote(self, tool: ModuleType, tmp_path: Path) -> None:
        # AviUtl2 は閉じるときに保存の窓の置き場と開いた履歴を書き直す 戻さないと
        # 本人の履歴に見本のプロジェクトが残り、保存の窓が作業フォルダで開く
        (tmp_path / "aviutl2.ini").write_bytes(b"[Dialog.InitialDir]\r\n")
        (tmp_path / "history.ini").write_bytes(b"[project]\r\n1=mine.aup2\r\n")
        saved = tool.save_settings(tmp_path)
        (tmp_path / "aviutl2.ini").write_bytes(b"rewritten")
        (tmp_path / "history.ini").write_bytes(b"[project]\r\n1=probe.aup2\r\n")
        (tmp_path / "explorer.ini").write_bytes(b"created by aviutl2")
        assert tool.restore_settings(saved) == []
        assert (tmp_path / "aviutl2.ini").read_bytes() == b"[Dialog.InitialDir]\r\n"
        assert (tmp_path / "history.ini").read_bytes() == b"[project]\r\n1=mine.aup2\r\n"
        # 控えを取ったときに無かった物は、AviUtl2 が作った物なので取り除く
        assert not (tmp_path / "explorer.ini").exists()

    def test_a_setting_that_cannot_be_written_is_reported(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        # 戻せなかったのに黙って終えると、本人の設定が変わったまま残る
        target = tmp_path / "aviutl2.ini"
        target.write_bytes(b"original")
        saved = tool.save_settings(tmp_path)
        target.unlink()
        target.mkdir()
        problems = tool.restore_settings(saved)
        assert problems and "aviutl2.ini" in problems[0]

    def test_the_data_folder_beside_the_exe_wins(self, tool: ModuleType, tmp_path: Path) -> None:
        # aviutl2.txt の決まり 違う方の控えを取ると、書き直された設定が戻らない
        exe = tmp_path / "aviutl2" / "aviutl2.exe"
        exe.parent.mkdir()
        environ = {"PROGRAMDATA": str(tmp_path / "pd")}
        assert tool.data_folder(exe, environ) == tmp_path / "pd" / "aviutl2"
        (exe.parent / "data").mkdir()
        assert tool.data_folder(exe, environ) == exe.parent / "data"


class TestOutput:
    def test_a_folder_with_something_in_it_is_refused(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        # 入れ替えると前の書き出しを消すことになる
        output = tmp_path / "aviutl"
        assert tool.output_ready(output) is None
        output.mkdir()
        assert tool.output_ready(output) is None
        (output / "frame000.png").write_bytes(b"")
        assert tool.output_ready(output) is not None

    def test_the_partial_folder_steps_around_existing_names(
        self, tool: ModuleType, tmp_path: Path
    ) -> None:
        # 前の回に失敗して残った絵と混ざると、枚数が揃ったように見える
        output = tmp_path / "aviutl"
        (tmp_path / "aviutl.sashimono-aaaaaaaa.part").mkdir()
        marks = iter(["aaaaaaaa", "bbbbbbbb"])
        chosen = tool.partial_folder(output, lambda: next(marks))
        assert chosen.name == "aviutl.sashimono-bbbbbbbb.part"

    def test_publish_renames_the_partial_folder(self, tool: ModuleType, tmp_path: Path) -> None:
        # 名前を変えられないと、書き終えた PNG が一時のフォルダに残り、compare が
        # aviutl フォルダに絵を見つけられない 空の出力のフォルダが邪魔をしてもいけない
        partial = tmp_path / "aviutl.sashimono-0123abcd.part"
        partial.mkdir()
        (partial / "frame000.png").write_bytes(b"png")
        output = tmp_path / "aviutl"
        output.mkdir()
        tool.publish(partial, output)
        assert (output / "frame000.png").read_bytes() == b"png"
        assert not partial.exists()


@pytest.fixture
def ready(tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """AviUtl2 が見つかり、何も開いていない状態 走らせた命令を記録する"""
    exe = tmp_path / "AviUtl2 v2" / "aviutl2.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    data = exe.parent / "data"
    data.mkdir()
    (data / "aviutl2.ini").write_bytes(b"original ini")
    project = tmp_path / "probe.aup2"
    project.write_text("[0]\nframe=0,2\n", encoding="utf-8")
    state: dict[str, Any] = {
        "exe": exe,
        "data": data,
        "project": project,
        "output": tmp_path / "out" / "aviutl",
        "commands": [],
        "running": set(),
        "code": 0,
    }

    def run_script(command: list[str], limit: int) -> tuple[int, bool]:
        state["commands"].append(command)
        folder = Path(command[command.index("-Folder") + 1])
        for index in range(3):
            (folder / f"frame{index}.png").write_bytes(b"png")
        (data / "aviutl2.ini").write_bytes(b"rewritten by aviutl2")
        return state["code"], False

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tool, "find_aviutl2", lambda explicit, environ: (exe, []))
    monkeypatch.setattr(tool, "ymm4_running", lambda image: image in state["running"])
    monkeypatch.setattr(tool, "run_script", run_script)
    return state


def _arguments(state: dict[str, Any]) -> list[str]:
    return ["--project", str(state["project"]), "--output", str(state["output"])]


@pytest.mark.parametrize("image", ["aviutl2.exe", "YukkuriMovieMaker.exe"])
def test_an_open_program_stops_the_tool_before_it_touches_anything(
    tool: ModuleType, ready: dict[str, Any], image: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # 開いているプロジェクトの上から別の物を開くと、作業中の物を壊しかねない
    ready["running"].add(image)
    assert tool.main(_arguments(ready)) == 1
    assert ready["commands"] == []
    assert image in capsys.readouterr().out


def test_a_finished_export_is_published_and_the_settings_are_restored(
    tool: ModuleType, ready: dict[str, Any]
) -> None:
    # 書き終えた連番が --output に移らないと見本の絵が手に入らず、aviutl2.ini を戻さないと
    # 道具を走らせた開発者の AviUtl2 が道具の書き換えた設定のまま残る
    assert tool.main(_arguments(ready)) == 0
    assert sorted(path.name for path in ready["output"].iterdir()) == [
        "frame0.png",
        "frame1.png",
        "frame2.png",
    ]
    assert (ready["data"] / "aviutl2.ini").read_bytes() == b"original ini"
    command = ready["commands"][0]
    assert command[command.index("-Frames") + 1] == "3"


def test_a_failed_export_leaves_nothing_behind_but_restores_the_settings(
    tool: ModuleType, ready: dict[str, Any]
) -> None:
    # 書き終えなかった絵が残ると、次の回に枚数の揃った物と取り違える
    ready["code"] = 1
    assert tool.main(_arguments(ready)) == 1
    assert not ready["output"].exists()
    assert list(ready["output"].parent.iterdir()) == []
    assert (ready["data"] / "aviutl2.ini").read_bytes() == b"original ini"


def test_too_few_pictures_are_a_failure_even_when_the_script_says_done(
    tool: ModuleType,
    ready: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """入口が PowerShell の終了コードだけを信じると、途中で切れた連番を出力として出す（Issue #210）

    YMM4 の道具では、書き出しが途中で止まっても PowerShell が書き終わりを待てたと言った
    この道具も枚数を数えるのは PowerShell の側だけだったので、入口でも数え直す
    """

    def run_script(command: list[str], limit: int) -> tuple[int, bool]:
        folder = Path(command[command.index("-Folder") + 1])
        for index in range(2):
            (folder / f"frame{index}.png").write_bytes(b"png")
        # 書き直しておかないと、戻す所を通らなくても下の確かめが通る
        (ready["data"] / "aviutl2.ini").write_bytes(b"rewritten by aviutl2")
        return 0, False

    monkeypatch.setattr(tool, "run_script", run_script)
    assert tool.main(_arguments(ready)) == 1
    assert "2 / 3 枚" in capsys.readouterr().out
    assert not ready["output"].exists()
    assert list(ready["output"].parent.iterdir()) == []
    assert (ready["data"] / "aviutl2.ini").read_bytes() == b"original ini"


def test_aviutl2_still_open_afterwards_is_shouted_without_restoring(
    tool: ModuleType,
    ready: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 開いたまま戻すと、閉じるときにまた書き直されて戻したことにならない
    def run_script(command: list[str], limit: int) -> tuple[int, bool]:
        ready["running"].add("aviutl2.exe")
        return 1, False

    monkeypatch.setattr(tool, "run_script", run_script)
    assert tool.main(_arguments(ready)) == tool.RESTORE_FAILED
    assert "!!!" in capsys.readouterr().out


def test_every_flag_the_entry_passes_is_a_script_parameter(tool: ModuleType) -> None:
    """名前が食い違うと PowerShell は「そんな引数は無い」で落ち、AviUtl2 は開きもしない"""
    text = SCRIPT.read_text(encoding="utf-8-sig")
    block = text[text.index("param(") : text.index(")\n\n")]
    parameters = {name.lower() for name in re.findall(r"\$(\w+)", block)}
    command = tool.powershell_command(
        Path("a.exe"), Path("a.aup2"), Path("out"), frames=3, timeout=5
    )
    flags = command[command.index("-File") + 2 :]
    passed = {flag[1:].lower() for flag in flags if flag.startswith("-")}
    assert passed <= parameters
    assert passed == {"aviutl2", "project", "folder", "frames", "timeoutseconds"}


def test_the_script_is_utf8_with_a_bom() -> None:
    """BOM が無いと Windows PowerShell 5.1 は Shift_JIS として読み、メニューの名前が全部化ける"""
    raw = SCRIPT.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    raw[3:].decode("utf-8")


_POWERSHELL = shutil.which("powershell.exe")
_WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None, reason="Windows PowerShell が要る"
)


@_WINDOWS_ONLY
def test_the_script_parses_in_windows_powershell() -> None:
    """構文が壊れると、AviUtl2 を開く前に PowerShell が落ちる"""
    command = (
        "$errors = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SCRIPT}', [ref]$null, [ref]$errors); "
        "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout.decode("cp932", errors="replace")


@_WINDOWS_ONLY
def test_the_script_loads_its_types_and_refuses_a_missing_folder(
    tool: ModuleType, tmp_path: Path
) -> None:
    """型を引く所まで走り、書き出す先が無ければ AviUtl2 を起こす前に断る

    名前の違う exe を渡すので、万一先へ進んでも AviUtl2 は起動しない
    """
    command = tool.powershell_command(
        tmp_path / "NotAviutl2AtAll.exe",
        tmp_path / "a.aup2",
        tmp_path / "missing",
        frames=3,
        timeout=5,
    )
    completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
    lines = [raw.decode("utf-8", errors="replace") for raw in completed.stdout.splitlines()]
    assert completed.returncode == 1, lines
    assert any("フォルダ" in line and "ありません" in line for line in lines), lines


class TestProbes:
    def test_every_probe_is_a_text_with_the_frame_painted_blue(
        self, probes: ModuleType, tmp_path: Path
    ) -> None:
        # 枠を塗らないと、行の高さと枠の上下の余白を分けて読めない
        written = probes.write_probes(tmp_path)
        assert len(written) == len(probes.PROBES)
        for path in written:
            text = path.read_bytes().decode("utf-8")
            assert "\r\n" in text
            assert "effect.name=テキスト" in text
            assert "合成モード=後方から合成" in text
            assert f"画像={(tmp_path / 'probes' / 'blue.png').resolve()}" in text

    def test_the_probes_can_be_read_back(self, probes: ModuleType, tmp_path: Path) -> None:
        # 読めない見本を並べると、AviUtl2 の絵と比べる相手が無くなる
        from sashimono.compat.aviutl.exo import load_exo

        for path in probes.write_probes(tmp_path):
            document = load_exo(path)
            assert document.generation >= 2
            assert document.objects

    def test_the_blue_picture_is_one_blue_pixel(self, probes: ModuleType, tmp_path: Path) -> None:
        # 青の 1 画素でないと、字（赤の成分が 255）と文字の枠（赤の成分が 0）を赤の成分で
        # 分けられず、行の高さと枠の余白を AviUtl2 の絵から読めない
        from PySide6.QtGui import QImage

        target = tmp_path / "blue.png"
        target.write_bytes(probes.blue_png())
        image = QImage(str(target))
        assert (image.width(), image.height()) == (1, 1)
        colour = image.pixelColor(0, 0)
        assert (colour.red(), colour.green(), colour.blue()) == (0, 0, 255)


def test_png_frames_are_read_by_their_number_over_black(
    compare: ModuleType, tmp_path: Path
) -> None:
    """連番の PNG は番号の値で引き、黒の上に重ねて返す

    名前の並びで引くと、桁の数が変わったときに別のフレームと比べる 透明のまま比べると、
    同じ絵でも Sashimono（黒の背景）との差が出る
    """
    from PySide6.QtGui import QColor, QImage

    for number, alpha in ((7, 128), (12, 255)):
        image = QImage(2, 2, QImage.Format.Format_RGBA8888)
        image.fill(QColor(255, 255, 255, alpha))
        image.save(str(tmp_path / f"frame{number:03d}.png"))
    found = compare.png_frames(tmp_path, {7, 12})
    assert set(found) == {7, 12}
    assert int(found[12][0, 0, 0]) == 255
    assert int(found[7][0, 0, 0]) == pytest.approx(128, abs=1)
    assert found[7].shape == (2, 2, 3)
    assert isinstance(found[7], np.ndarray)
