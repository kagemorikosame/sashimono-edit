r"""利用者が自分で確かめて送る診断情報（#516）

報告は公開の Discussions に添えられる 伏せ損ねると、利用者名・素材の名前・字幕・API の鍵が
誰でも読める所に出る 伏せすぎると、どのファイルの何行目で落ちたかが読めず報告の意味が無い
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from sashimono import diagnostic_report
from sashimono.ai.outbound import PATH_MARK, TEXT_MARK, file_secrets, secret_pairs
from sashimono.core.model import Project
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
from sashimono.ui.report_masking import HOME_PLACEHOLDER, SECRET_MARK, mask_secrets
from sashimono.ui.workspace import Preferences

#: 空白を含む利用者名 形で場所を拾う伏せ方は空白で切れるので、名前の後ろが残りやすい
HOME = r"C:\Users\Kage Mori"
FOLDERS = [(HOME, HOME_PLACEHOLDER)]

TRACE = rf"""Traceback (most recent call last):
  File "{HOME}\Programs\Sashimono\_internal\sashimono\ui\main_window.py", line 12, in save
  File "C:\Python313\Lib\pathlib.py", line 3, in open
FileNotFoundError: [Errno 2] No such file or directory: '{HOME}\Videos\旅行 1 日目.mp4'
"""


class TestMask:
    def test_the_trace_keeps_where_it_failed(self) -> None:
        # 伏せすぎると、どのファイルの何行目で落ちたかが読めない
        masked = mask(TRACE, folders=FOLDERS)
        assert 'File "sashimono/ui/main_window.py", line 12, in save' in masked
        assert 'File "pathlib.py", line 3, in open' in masked

    def test_the_user_name_and_the_file_name_are_hidden(self) -> None:
        # 利用者名は空白を含んでも残さない 置き場の下のファイルの名前も伏せる
        masked = mask(TRACE, folders=FOLDERS)
        assert "Kage" not in masked and "Mori" not in masked
        assert "旅行" not in masked
        assert f"{HOME_PLACEHOLDER}\\{PATH_MARK}" in masked

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
        ],
    )
    def test_secrets_are_hidden(self, text: str, kept: str) -> None:
        # 鍵は 1 つ漏れるだけで他人に使われる 何の鍵だったかは残す
        masked = mask_secrets(text)
        assert kept + SECRET_MARK in masked
        assert "abcd1234" not in masked and "hunter2" not in masked
        assert "abcdefghijklmnop" not in masked

    def test_program_paths_only_touch_traces(self) -> None:
        # 並びの外の場所は削らない（伏せる側が伏せる） 削ると素材の名前だけが残る
        text = r'File "C:\x\sashimono\app.py", line 1' + "\n" + r"C:\x\旅行.mp4"
        assert program_paths(text) == 'File "sashimono/app.py", line 1' + "\n" + r"C:\x\旅行.mp4"


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
            Preferences(package_folder=r"D:\山田\packages", report_folder="", theme="light")
        )
        lines = "\n".join(preference_lines(preferences))
        assert "山田" not in lines
        assert f"package_folder: {PATH_MARK}" in lines
        assert "report_folder: （空 既定）" in lines
        assert "theme: light" in lines
        assert "new_project_style_guide: （" in lines

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

    def test_stalls_are_counted_the_same_way_as_the_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 報告の表と、開発者が tools/stall_report.py で数えた表が食い違わない
        stalls = tmp_path / "stalls.jsonl"
        stalls.write_text(
            '{"time": "2026-11-24T10:00:00", "stall_ms": 120, "gc": [], "samples": []}\n壊れた行\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(diagnostic_report, "default_paths", lambda: [stalls])
        gathered = gather(FaultLog(tmp_path, catch_faults=False))
        assert gathered.stalls.startswith("止まり 1 回 合計 120 ms")
        assert gathered.problems == ()
