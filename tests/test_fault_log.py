"""落ちた起動と拾われなかったエラーの記録（#516）

記録は、利用者の機械でだけ起きた不具合を診断情報に載せるための手掛かり 壊れると、
落ちた後に作った診断情報が空で、何が起きたか分からない
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from sashimono import __version__
from sashimono.core.io.locks import try_hold
from sashimono.fault_log import KEEP_LOST, MAX_LOG_BYTES, FaultLog

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def log(tmp_path: Path) -> Iterator[FaultLog]:
    created = FaultLog(tmp_path / "diagnostics", catch_faults=False)
    assert created.install()
    yield created
    created.uninstall()


def _raise_and_report(message: str) -> None:
    try:
        raise ValueError(message)
    except ValueError:
        sys.excepthook(*sys.exc_info())


class TestErrors:
    def test_an_unhandled_error_is_kept_with_its_trace(
        self, log: FaultLog, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # 画面のボタンから呼んだ所のエラーは Qt が握りつぶす 書き残さないと本人にも分からない
        _raise_and_report("書けなかった")
        (entry,) = log.recent_errors()
        assert f"版 {__version__}" in entry
        assert "ValueError: 書けなかった" in entry
        assert "_raise_and_report" in entry
        # 前の受け手（標準エラーへ書く）にも渡す 開発の環境でコマンドから起こした人が見られる
        assert "書けなかった" in capsys.readouterr().err

    def test_an_error_in_a_worker_thread_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 裏の係（先読み・書き出し）のエラーはスレッドの受け手へ行く 前の受け手は pytest の
        # 物で、渡すと試験の終わりに警告として上がるので、黙る受け手に替えてから掛ける
        handed: list[threading.ExceptHookArgs] = []
        monkeypatch.setattr(threading, "excepthook", handed.append)
        log = FaultLog(tmp_path, catch_faults=False)
        log.install()

        def work() -> None:
            raise OSError("読めない")

        try:
            thread = threading.Thread(target=work, name="sashimono-試し")
            thread.start()
            thread.join()
        finally:
            log.uninstall()
        assert len(handed) == 1
        (entry,) = log.recent_errors()
        assert "スレッド sashimono-試し" in entry
        assert "OSError: 読めない" in entry

    def test_only_the_newest_are_read(self, log: FaultLog) -> None:
        # 壊れると、古いエラーばかり載るか、文の中の === で 1 件が割れる
        for number in range(7):
            _raise_and_report(f"{number} 回目 === の入った文")
        entries = log.recent_errors(5)
        assert [entry.count("回目") for entry in entries] == [1] * 5
        assert "6 回目" in entries[-1]
        assert "2 回目" in entries[0]

    def test_uninstall_gives_the_hooks_back(self, tmp_path: Path) -> None:
        # 壊れると、外した後も受け手が残り、閉じた記録へ書こうとする
        before = (sys.excepthook, threading.excepthook)
        log = FaultLog(tmp_path, catch_faults=False)
        log.install()
        assert sys.excepthook != before[0]
        log.uninstall()
        assert (sys.excepthook, threading.excepthook) == before


class TestLostSessions:
    def test_a_clean_end_leaves_nothing(self, tmp_path: Path) -> None:
        # 正しく終わった起動の記録を残すと、次の診断情報で落ちた起動に数えられる
        log = FaultLog(tmp_path, catch_faults=False)
        log.install()
        path = log.session_path
        assert path is not None and path.is_file()
        log.uninstall()
        assert not path.exists()
        assert FaultLog(tmp_path).lost_sessions() == []

    def test_a_session_left_behind_is_lost(self, log: FaultLog) -> None:
        # 落ちた起動は記録が残り、錠の持ち主がいない
        crashed = log.sessions / "20261124-100000-1234.log"
        crashed.write_text(
            "起動 2026-11-24T10:00:00 版 0.2.6\nWindows fatal exception: access violation\n",
            encoding="utf-8",
        )
        (lost,) = log.lost_sessions()
        assert (lost.started, lost.version) == ("2026-11-24T10:00:00", "0.2.6")
        assert "access violation" in lost.text

    def test_a_running_window_is_not_lost(self, log: FaultLog) -> None:
        # 2 つ目の窓が動いている間に作った診断情報で、その窓を落ちた物に数えない
        other = log.sessions / "20261124-110000-5678.log"
        other.write_text("起動 2026-11-24T11:00:00 版 0.2.6\n", encoding="utf-8")
        lock = try_hold(other.with_suffix(".lock"))
        assert lock is not None
        try:
            assert log.lost_sessions() == []
        finally:
            lock.release()
        assert len(log.lost_sessions()) == 1

    def test_old_lost_sessions_are_forgotten(self, tmp_path: Path) -> None:
        # 毎回落ちる機械でも溜まり続けない
        sessions = tmp_path / "sessions"
        sessions.mkdir(parents=True)
        for number in range(KEEP_LOST + 3):
            (sessions / f"20261124-1000{number:02d}-1.log").write_text(
                f"起動 2026-11-24T10:00:{number:02d} 版 0.2.6\n", encoding="utf-8"
            )
        log = FaultLog(tmp_path, catch_faults=False)
        log.install()
        try:
            lost = log.lost_sessions()
        finally:
            log.uninstall()
        assert len(lost) == KEEP_LOST
        assert lost[-1].started == f"2026-11-24T10:00:{KEEP_LOST + 2:02d}"

    def test_a_process_that_dies_is_found_by_the_next_one(self, tmp_path: Path) -> None:
        # アプリの異常終了の後 atexit を通らずに終わった起動を、次の起動が見つける
        code = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "from sashimono.fault_log import FaultLog\n"
            "log = FaultLog(Path(sys.argv[1]))\n"
            "assert log.install()\n"
            "os._exit(3)\n"
        )
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"}
        completed = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            timeout=300,
            check=False,
        )
        assert completed.returncode == 3, completed.stderr
        (lost,) = FaultLog(tmp_path).lost_sessions()
        assert lost.version == __version__
        # Python の外で止まった（強制終了など）ので、落ちた時の並びは無い
        assert lost.text == ""


def test_an_error_is_kept_even_when_the_log_cannot_be_rotated(tmp_path: Path) -> None:
    # 壊れると、2 つ目の窓が記録を開いていて回せないだけで、エラーを書かずに落とす
    log = FaultLog(tmp_path, catch_faults=False)
    log.errors_path.write_bytes(b"x" * (MAX_LOG_BYTES + 1))
    # 1 世代前の名前をフォルダにして、名前を変えられなくする
    log.errors_path.with_suffix(".log.1").mkdir()
    try:
        raise ValueError("回せない")
    except ValueError:
        kind, error, trace = sys.exc_info()
        assert kind is not None and error is not None
        log.record_error(kind, error, trace)
    assert "ValueError: 回せない" in log.errors_path.read_text(encoding="utf-8")


def test_a_session_head_of_the_wrong_shape_is_not_copied(tmp_path: Path) -> None:
    # 壊れると、頭の行が壊れた・書き換えられた記録の中身が、起動の時刻や版として報告に写る
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "20261124-100000-1.log").write_text("起動 顧客 版 社外秘 続き\n", encoding="utf-8")
    (lost,) = FaultLog(tmp_path).lost_sessions()
    assert (lost.started, lost.version) == ("20261124-100000-1", "不明")


def test_a_folder_that_cannot_be_written_does_not_stop_the_start(tmp_path: Path) -> None:
    # 診断のために本人の作業を落とさない
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    log = FaultLog(blocker / "diagnostics", catch_faults=False)
    before = sys.excepthook
    assert not log.install()
    assert sys.excepthook is before
    log.uninstall()
