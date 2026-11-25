r"""利用者が自分で確かめて送る診断情報（#516）

報告は公開の Discussions に添えられる 伏せ損ねると、利用者名・素材の名前・字幕・API の鍵が
誰でも読める所に出る 伏せすぎると、どのファイルの何行目で落ちたかが読めず報告の意味が無い
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

from sashimono import diagnostic_report
from sashimono.ai.outbound import PATH_MARK, TEXT_MARK, file_secrets, secret_pairs
from sashimono.core.model import Project, StyleGuide
from sashimono.diagnostic_report import (
    Environment,
    Gathered,
    build_report,
    gather,
    mask,
    preference_lines,
    program_paths,
    suggested_name,
)
from sashimono.fault_log import FaultLog, LostSession
from sashimono.ui.report_masking import (
    HOME_PLACEHOLDER,
    SECRET_MARK,
    mask_error_record,
    mask_fault_record,
    mask_places,
    mask_secrets,
)
from sashimono.ui.workspace import Preferences

#: 空白を含む利用者名 形で場所を拾う伏せ方は空白で切れるので、名前の後ろが残りやすい
HOME = r"C:\Users\Kage Mori"
FOLDERS = [(HOME, HOME_PLACEHOLDER)]

#: 並びの場所の削り方（``Path`` で部品に分ける）は Windows の区切りを前提にする 配る版は
#: Windows だけなので、ほかの OS では Windows の形の入力を使う試験を飛ばす
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="Windows の場所の形を使う")

#: 本体の置き場 並びの中の本体の場所は、ここの下にある物だけを本体の中の場所にする
PACKAGE = Path(diagnostic_report.__file__).absolute().parent

TRACE = rf"""Traceback (most recent call last):
  File "{PACKAGE}\ui\main_window.py", line 12, in save
  File "C:\Python313\Lib\pathlib.py", line 3, in open
FileNotFoundError: [Errno 2] No such file or directory: '{HOME}\Videos\旅行 1 日目.mp4'
"""


class TestMask:
    @WINDOWS_ONLY
    def test_the_trace_keeps_where_it_failed(self) -> None:
        # 伏せすぎると、どのファイルの何行目で落ちたかが読めない
        masked = mask(TRACE, folders=FOLDERS)
        assert 'File "sashimono/ui/main_window.py", line 12, in save' in masked
        assert 'File "pathlib.py", line 3, in open' in masked

    def test_the_user_name_and_the_file_name_are_hidden(self) -> None:
        # 利用者名は空白を含んでも残さない 置き場の下のファイルの名前も伏せる
        masked = mask(TRACE, folders=FOLDERS)
        assert "Kage" not in masked and "Mori" not in masked
        assert "旅行" not in masked and "日目" not in masked
        assert f"{HOME_PLACEHOLDER}\\{PATH_MARK}" in masked

    @pytest.mark.parametrize(
        "text",
        [
            r"'%USERPROFILE%\Videos\社外秘 顧客名簿.csv' を読めない",
            r"読めない: %USERPROFILE%\Videos\社外秘 顧客名簿.csv",
            r"読めない: D:\動画\社外秘 顧客名簿.csv",
            r"読めない: \\server\共有\社外秘 顧客名簿.csv",
        ],
    )
    def test_a_name_with_a_space_is_hidden_to_the_end(self, text: str) -> None:
        # 壊れると、空白の後ろ（顧客名簿.csv）が報告に残る（PR #550 の指摘）
        masked = mask_places(text, PATH_MARK)
        assert "顧客名簿" not in masked and "社外秘" not in masked
        assert PATH_MARK in masked

    def test_material_names_and_subtitles_are_hidden(self, tmp_path: Path) -> None:
        # 素材の名前は場所の形でなくても（エラーの文の中の名前だけでも）伏せる
        media = tmp_path / "社員旅行の記録.mp4"
        media.write_bytes(b"")
        project = Project.create()
        names = secret_pairs(
            file_secrets(
                project,
                project_path=None,
                export_folder=None,
                remembered=[str(media)],
            )
        )
        text = "素材を読めない: 社員旅行の記録.mp4\n書体に無い字: 社長のあいさつ\n"
        masked = mask(text, names=names, texts=["社長のあいさつ"], folders=FOLDERS)
        assert "社員旅行" not in masked
        assert "社長のあいさつ" not in masked
        assert TEXT_MARK in masked

    def test_other_absolute_places_are_hidden(self) -> None:
        # 壊れると、ホームの外に置いた素材の場所と名前がそのまま報告に出る
        masked = mask(r"書けない: D:\動画\案件\納品.mp4 を見る", folders=FOLDERS)
        assert "案件" not in masked and "納品" not in masked
        assert PATH_MARK in masked

    @pytest.mark.parametrize(
        ("text", "kept"),
        [
            ("key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123 で断られた", "sk-"),
            ("ghp_abcdefghijklmnopqrstuvwxyz012345", "ghp_"),
            ("Authorization: Bearer abc.def-ghi", "Bearer "),
            ('{"api_key": "abcd1234"}', '"api_key": "'),
            ("token=abcd1234&x=1", "token="),
            ("password: hunter2", "password: "),
            ('{"password": "abcd1234 hunter2"}', '"password": "'),
            ("password = 'abcd1234 \\' hunter2'", "password = '"),
        ],
    )
    def test_secrets_are_hidden(self, text: str, kept: str) -> None:
        # 鍵は 1 つ漏れるだけで他人に使われる 何の鍵だったかは残す
        masked = mask_secrets(text)
        assert kept + SECRET_MARK in masked
        assert "abcd1234" not in masked and "hunter2" not in masked
        assert "abcdefghijklmnop" not in masked

    @WINDOWS_ONLY
    def test_program_paths_only_touch_traces(self) -> None:
        # 並びの外の場所は削らない（伏せる側が伏せる） 削ると素材の名前だけが残る
        text = f'File "{PACKAGE / "app.py"}", line 1' + "\n" + r"C:\x\旅行.mp4"
        assert program_paths(text) == 'File "sashimono/app.py", line 1' + "\n" + r"C:\x\旅行.mp4"


class TestPastRecords:
    def test_names_from_an_earlier_run_are_hidden_without_the_table(self) -> None:
        # 壊れると、前の起動で開いた素材の名前と字幕が、今の作品の表に無いので報告に残る
        record = (
            "=== 2026-11-24T10:00:00 版 0.2.7\n"
            "Traceback (most recent call last):\n"
            '  File "sashimono/engine/decode/probe.py", line 3, in probe_media\n'
            "OSError: 社員旅行の記録.mp4 を読めない\n"
            "ValueError: 書体に無い字 「社長のあいさつ」 と 'secret plan.txt'\n"
        )
        masked = mask_error_record(record, TEXT_MARK)
        assert "社員旅行" not in masked and "社長" not in masked and "secret plan" not in masked
        # 並びの行・版・部品の名前は残す
        assert '  File "sashimono/engine/decode/probe.py", line 3, in probe_media' in masked
        assert "版 0.2.7" in masked
        assert "OSError:" in masked

    def test_escaped_quotes_and_spaced_names_do_not_leave_pieces(self, tmp_path: Path) -> None:
        # 壊れると、囲みの中の \' で囲みが終わったと見て後ろ半分（omega）が残るか、空白を含む
        # 名前の頭（社外秘）が残る（PR #550 の指摘） 集める所から最後の伏せ方まで通して見る
        errors = tmp_path / "errors.log"
        errors.write_text(
            "=== 2026-11-24T10:00:00 版 0.2.7\n"
            "Traceback (most recent call last):\n"
            '  File "sashimono/ui/main_window.py", line 3, in save\n'
            "ValueError: password = 'alpha\\' omega'\n"
            "OSError: 社外秘 顧客名簿.csv を読めない\n"
            "ImportError: sashimono_native.pyd を読めない\n",
            encoding="utf-8",
        )
        gathered = gather(FaultLog(tmp_path, catch_faults=False))
        masked = mask(build_report(ENV, None, gathered), folders=FOLDERS)
        assert "omega" not in masked and "alpha" not in masked
        assert "社外秘" not in masked and "顧客名簿" not in masked
        # 並びの行と例外の種類は読める 本文は丸ごと伏せる
        assert '  File "sashimono/ui/main_window.py", line 3, in save' in masked
        assert f"OSError: {TEXT_MARK}\n" in masked
        assert f"ImportError: {TEXT_MARK}\n" in masked

    def test_an_error_record_keeps_only_the_fixed_shapes(self) -> None:
        # 壊れると、囲みも拡張子も無い本文（顧客名 田中）・例外の形でない行の字幕・ソースの行の
        # 文字・決まった見出しで始まる本文の行が残る 名前の形を探す作りは 3 回続けて漏れた
        # （PR #550） 決まった形から要る所だけを取り出して組み直す
        record = (
            "=== 2026-11-24T10:00:00 版 0.2.7 スレッド sashimono-encoder-probe\n"
            "Traceback (most recent call last):\n"
            '  File "sashimono/core/io/project.py", line 9, in load\n'
            '    raise ValueError("顧客 山田様")\n'
            "           ^^^^^^^^^^^^^^^^\n"
            "ValueError: 顧客名 田中\n"
            "社長のあいさつ\n"
            "Extension modules: 顧客名簿の続き\n"
            "社長: こんにちは\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'x'\n"
            "Windows fatal exception: 顧客\n"
        )
        masked = mask_error_record(record, TEXT_MARK)
        for gone in ("田中", "山田", "社長", "顧客", "^^^^"):
            assert gone not in masked
        assert masked.splitlines() == [
            "=== 2026-11-24T10:00:00 版 0.2.7 スレッド sashimono-encoder-probe",
            "Traceback (most recent call last):",
            '  File "sashimono/core/io/project.py", line 9, in load',
            # ソースの行と印は写さない 続けて伏せた行は 1 つにまとめる
            f"    {TEXT_MARK}",
            f"           {TEXT_MARK}",
            f"ValueError: {TEXT_MARK}",
            TEXT_MARK,
            f"FileNotFoundError: [Errno 2] {TEXT_MARK}",
            TEXT_MARK,
        ]

    def test_a_fault_record_keeps_only_the_fixed_shapes(self) -> None:
        # 壊れると、落ちた時の記録の行の後ろに続いた文や、例外の本文に似た行が残る
        record = (
            "Windows fatal exception: access violation\n"
            "Windows fatal exception: code 0xc0000005\n"
            "Fatal Python error: Segmentation fault 顧客名\n"
            "\n"
            "Current thread 0x00001a2b (most recent call first):\n"
            '  File "sashimono/engine/gpu/context.py", line 40 in make_current 顧客\n'
            "  <no Python frame>\n"
            "Extension modules: numpy.core._multiarray_umath, PySide6.QtCore 顧客 (total: 2)\n"
            "ValueError: 社長\n"
        )
        masked = mask_fault_record(record, TEXT_MARK)
        assert "顧客" not in masked and "社長" not in masked
        assert masked.splitlines() == [
            "Windows fatal exception: access violation",
            "Windows fatal exception: code 0xc0000005",
            f"Fatal Python error: Segmentation fault {TEXT_MARK}",
            "",
            "Current thread 0x00001a2b (most recent call first):",
            '  File "sashimono/engine/gpu/context.py", line 40 in make_current',
            "  <no Python frame>",
            "Extension modules: numpy.core._multiarray_umath, PySide6.QtCore",
            TEXT_MARK,
        ]

    def test_a_secret_is_hidden_even_when_a_subtitle_matches_its_key(self) -> None:
        # 壊れると、字幕に password という文があると、字幕の伏せ方が鍵の名前を印に変え、
        # 秘密の伏せ方が値を見つけられずに残す（PR #550 の指摘）
        masked = mask("password: hunter2", texts=["password"], folders=FOLDERS)
        assert "hunter2" not in masked

    def test_a_secret_is_hidden_even_when_a_name_matches_its_key(self) -> None:
        # 壊れると、作品の名前が password のとき、名前の表が鍵の名前を消し、値が残る
        masked = mask("password = hunter2", names=[("password", "〔作品〕")], folders=FOLDERS)
        assert "hunter2" not in masked

    def test_gathered_records_are_masked(self, tmp_path: Path) -> None:
        # 集めた所で伏せる 窓へ出る前に、表に無い名前が消えている
        errors = tmp_path / "errors.log"
        errors.write_text(
            "=== 2026-11-24T10:00:00 版 0.2.7\nOSError: 社員旅行の記録.mp4 を読めない\n",
            encoding="utf-8",
        )
        gathered = gather(FaultLog(tmp_path, catch_faults=False))
        assert "社員旅行" not in "\n".join(gathered.errors)


ENV = Environment(
    version="0.2.7",
    python="3.13.1",
    qt="6.12.0",
    system="Windows-11",
    frozen=True,
    screens=("1920x1080 150%",),
)


class TestReport:
    def test_an_empty_machine_says_there_is_nothing(self) -> None:
        # 壊れると、記録が無かったのか載せ忘れたのかが報告から読めない
        text = build_report(ENV, None, Gathered(), now=datetime(2026, 11, 24, 12, 0))
        assert "作った時刻: 2026-11-24T12:00:00" in text
        assert "版: 0.2.7（配布版）" in text
        assert "画面: 1920x1080 150%" in text
        assert "自動では送られません" in text
        assert "## 拾われなかったエラー（新しい 5 件まで）\nなし" in text
        assert "## 正しく終わらなかった起動（新しい 3 件まで）\nなし" in text
        assert "設定\n  （読めなかった）" in text

    def test_errors_and_lost_sessions_are_listed(self) -> None:
        # 壊れると、落ちた起動やエラーが報告に載らず、調べる手掛かりが無い
        gathered = Gathered(
            stalls="止まり 3 回",
            errors=("=== 2026-11-24T10:00:00 版 0.2.7\nValueError: x",),
            lost=(LostSession("2026-11-24T09:00:00", "0.2.6", ""),),
            problems=("エラーの記録を読めなかった: OSError",),
        )
        text = build_report(ENV, None, gathered)
        assert "止まり 3 回" in text
        assert "ValueError: x" in text
        assert "起動 2026-11-24T09:00:00 版 0.2.6" in text
        assert "強制終了" in text
        assert "## 読めなかった記録\nエラーの記録を読めなかった: OSError" in text

    def test_long_traces_keep_the_end(self) -> None:
        # 落ちた所に近い行（後ろ）ほど要る
        long = "\n".join(f"行 {number}" for number in range(200))
        text = build_report(ENV, None, Gathered(errors=(long,)))
        assert "行 199" in text and "行 0\n" not in text
        assert "前の 140 行を省いた" in text

    def test_preferences_hide_folders_and_the_style_guide(self) -> None:
        # 置き場の値は利用者名を含む 編集の方針は本人の文
        from dataclasses import asdict

        preferences = asdict(
            Preferences(
                package_folder=r"D:\山田\packages",
                report_folder="",
                theme="light",
                new_project_style_guide=StyleGuide(notes="社外秘の方針 顧客 A 社向け"),
                subtitle_look_last="社長の字幕",
            )
        )
        lines = "\n".join(preference_lines(preferences))
        for gone in ("山田", "社外秘", "顧客", "社長"):
            assert gone not in lines
        assert f"package_folder: {PATH_MARK}" in lines
        assert "report_folder: （空 既定）" in lines
        assert "theme: light" in lines
        # 方針は中身がある印だけ 本人が名付けた物は形が決まらないので「設定あり」
        assert "new_project_style_guide: （有り）" in lines
        assert "subtitle_look_last: （設定あり）" in lines
        empty = "\n".join(preference_lines(asdict(Preferences())))
        assert "new_project_style_guide: （無し）" in empty

    def test_the_file_name_has_the_time(self) -> None:
        # 壊れると、2 度作ったときに前の報告を上書きする
        name = suggested_name(datetime(2026, 11, 24, 9, 5, 7))
        assert name == "sashimono-診断情報-20261124-090507.txt"


class TestGather:
    def test_a_broken_stall_log_does_not_stop_the_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 1 つ読めないだけで報告を作れないと、調べる手掛かりが何も残らない
        def broken(*_args: object, **_kwargs: object) -> object:
            raise OSError("読めない")

        monkeypatch.setattr(diagnostic_report, "read", broken)
        log = FaultLog(tmp_path, catch_faults=False)
        gathered = gather(log)
        assert gathered.stalls == ""
        assert gathered.problems == ("画面の止まりの記録を読めなかった: OSError",)

    def test_a_log_that_cannot_be_read_is_not_called_empty(self, tmp_path: Path) -> None:
        # 壊れると、在るのに読めなかった記録が「なし」と出て、無かったのか読めなかったのか
        # 分からない（PR #550 の指摘） 読めない物の代わりに、ファイルの名前の所をフォルダにする
        (tmp_path / "errors.log").mkdir()
        (tmp_path / "sessions").mkdir()
        (tmp_path / "sessions" / "20261124-100000-1.log").mkdir()
        gathered = gather(FaultLog(tmp_path, catch_faults=False))
        assert "エラーの記録を読めなかった: errors.log" in gathered.problems
        assert "落ちた起動の記録を読めなかった: 20261124-100000-1.log" in gathered.problems

    def test_stalls_are_counted_the_same_way_as_the_tool(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # 報告の表と、開発者が tools/stall_report.py で数えた表が食い違わない 同じ記録を
        # 道具の入口にも渡し、同じ数（場所の件数も報告と同じ）で出た表と比べる
        import importlib.util

        stalls = tmp_path / "stalls.jsonl"
        stalls.write_text(
            '{"time": "2026-11-24T10:00:00", "stall_ms": 120, "gc": [], "samples": []}\n'
            '{"time": "2026-11-24T10:00:01", "stall_ms": 45.5}\n'
            "壊れた行\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(diagnostic_report, "default_paths", lambda: [stalls])
        gathered = gather(FaultLog(tmp_path, catch_faults=False))
        spec = importlib.util.spec_from_file_location(
            "stall_report_for_parity",
            Path(__file__).resolve().parents[1] / "tools" / "stall_report.py",
        )
        assert spec is not None and spec.loader is not None
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        capsys.readouterr()
        assert tool.main([str(stalls), "--top", str(diagnostic_report.STALL_TOP)]) == 0
        assert capsys.readouterr().out.strip() == gathered.stalls.strip()
        assert gathered.stalls.startswith("止まり 2 回")
        assert gathered.problems == ()

    def test_a_stall_line_of_the_wrong_shape_is_skipped_and_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、欄の型が違う 1 行で数える所が落ち、止まりの表もエラーと落ちた起動の記録も
        # 集められない（PR #550 の指摘） 正しい行と、ほかの記録は集める
        stalls = tmp_path / "stalls.jsonl"
        stalls.write_text(
            '{"time": "2026-11-24T10:00:00", "stall_ms": 120}\n'
            '{"time": "2026-11-24T10:00:01", "stall_ms": 80, "gc": [0]}\n'
            '{"time": "2026-11-24T10:00:02", "stall_ms": 80, "samples": "x"}\n'
            '{"time": "2026-11-24T10:00:03", "stall_ms": 80, "memory_mb": 3}\n'
            '{"time": "2026-11-24T10:00:04", "stall_ms": true}\n',
            encoding="utf-8",
        )
        (tmp_path / "errors.log").write_text(
            "=== 2026-11-24T10:00:00 版 0.2.7\nOSError: x\n", encoding="utf-8"
        )
        monkeypatch.setattr(diagnostic_report, "default_paths", lambda: [stalls])
        gathered = gather(FaultLog(tmp_path, catch_faults=False))
        assert gathered.stalls.startswith("止まり 1 回 合計 120 ms")
        assert gathered.problems == ("画面の止まりの記録で形の違う行を 4 行飛ばした",)
        assert len(gathered.errors) == 1
