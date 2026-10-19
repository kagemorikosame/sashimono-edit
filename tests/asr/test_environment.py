"""起こしの実行環境の検出と導入

実際に 2 GB を落とすわけにはいかないので、コマンドの組み立てと、子プロセスの
出力を 1 行ずつ拾えることを確かめる
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from sashimono import runtime as runtime_module
from sashimono.asr.environment import (
    ASR_PACK,
    CUDA_PACKAGES,
    REQUIRED_PACKAGES,
    install_command,
    install_runtime,
    runtime_status,
)
from sashimono.runtime import PackageStatus, PackStatus


def _status(*, installed: bool, cuda: bool) -> PackStatus:
    return PackStatus(
        pack=ASR_PACK,
        packages=tuple(
            PackageStatus(name, "1.0" if installed else None) for name in REQUIRED_PACKAGES
        ),
        extras=tuple(PackageStatus(name, "1.0" if cuda else None) for name in CUDA_PACKAGES),
    )


class TestRuntimeStatus:
    def test_missing_packages_are_reported(self) -> None:
        # 入っていない物を数えられないと、字幕起こしの窓の「入れるもの」が空になり、
        # 何を落とすか見せないまま導入が始まる
        status = _status(installed=False, cuda=False)
        assert status.installed is False
        assert status.missing(extra=False) == REQUIRED_PACKAGES

    def test_cpu_only_is_ready_but_not_cuda_ready(self) -> None:
        # CUDA ランタイムだけが無い人を未導入と見ると、
        # CPU で起こせるのに字幕起こしの窓の〔起こす〕が押せない
        status = _status(installed=True, cuda=False)
        assert (status.installed, status.extra_installed) == (True, False)
        assert status.missing(extra=True) == CUDA_PACKAGES
        assert status.missing(extra=False) == ()

    def test_summary_distinguishes_the_three_states(self) -> None:
        # 3 つの言い分けが崩れると、CUDA ランタイムだけ足りない人の導入欄に
        # 「未導入」か「導入済み」が出て、GPU 版を後から足せることに気付けない
        assert "未導入" in _status(installed=False, cuda=False).summary()
        assert "CUDA" in _status(installed=True, cuda=False).summary()
        assert _status(installed=True, cuda=True).summary() == "導入済み"

    def test_real_lookup_does_not_raise(self) -> None:
        # 入っていない環境で落ちないことが要点 既定では未導入で配布する
        assert isinstance(runtime_status().installed, bool)


class TestInstallCommand:
    def test_cuda_packages_are_included_only_when_asked(self) -> None:
        # 印に関係なく cuDNN が入ると、
        # 「GPU を使う」を外した人まで 1.7 GB の CUDA ランタイムを落とす
        with_cuda = install_command(cuda=True)
        without = install_command(cuda=False)
        assert any("cudnn" in part for part in with_cuda)
        assert not any("cudnn" in part for part in without)

    def test_it_runs_pip_from_the_current_interpreter(self) -> None:
        # 動いている Python 以外の pip を走らせると別の環境へ入り、
        # 導入が済んでも字幕起こしが faster-whisper を読めない
        command = install_command(cuda=False)
        assert command[:4] == [sys.executable, "-m", "pip", "install"]

    def test_faster_whisper_carries_a_lower_bound(self) -> None:
        # 下限が外れると、古い faster-whisper が入った人も導入済みと見なされ、
        # 導入のボタンを押しても古い版が入れ替わらない
        assert "faster-whisper>=1.1" in install_command(cuda=False)

    def test_frozen_builds_install_into_their_own_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 固めた実行ファイルには書き込めないので、専用フォルダへ入れる
        monkeypatch.setattr(runtime_module, "runtime_target_dir", lambda: tmp_path / "runtime")
        command = install_command(cuda=False)
        assert "--target" in command
        assert str(tmp_path / "runtime") in command


class TestInstallRuntime:
    def test_output_arrives_line_by_line(self) -> None:
        # 出力を 1 行ずつ渡せないと、導入の窓の記録欄が数分間空のままで、
        # 止まったのか進んでいるのか分からない
        lines: list[str] = []
        code = install_runtime(
            command=[sys.executable, "-c", "print('1 行目'); print('2 行目')"],
            on_output=lines.append,
        )
        assert code == 0
        # 1 行目は実行するコマンドそのもの 何が走るのか見せてから始める
        assert lines[0].startswith("> ")
        assert lines[1:] == ["1 行目", "2 行目"]

    def test_failure_is_reported_as_a_non_zero_code(self) -> None:
        # pip が失敗しても 0 が返ると、導入の窓が成功と出し、入っていない環境を読み込みにいく
        assert install_runtime(command=[sys.executable, "-c", "raise SystemExit(3)"]) == 3

    def test_a_silent_install_can_be_cancelled(self) -> None:
        """pip が何も出さずに落としている間でも、中断が効く

        出力の 1 行ごとにしか中断を見ないと、黙って数分落としている間は止まらず、
        導入欄は導入中のまま、ダイアログも閉じられない
        """
        lines: list[str] = []
        codes: list[int] = []
        asked = threading.Event()

        def run() -> None:
            codes.append(
                install_runtime(
                    command=[sys.executable, "-c", "import time; time.sleep(60)"],
                    on_output=lines.append,
                    should_cancel=asked.is_set,
                )
            )

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        threading.Event().wait(0.5)  # 子が黙って動いている所で頼む
        asked.set()
        worker.join(timeout=15.0)

        assert not worker.is_alive(), "中断を頼んでも導入が終わらない"
        assert codes and codes[0] != 0
        assert lines[-1] == "中断した"

    def test_a_missing_executable_does_not_raise(self) -> None:
        # 起動できないことを例外で返すと、導入の窓の記録欄に「pip を起動できない」が出ず、
        # 例外の型の名前だけが並ぶ
        lines: list[str] = []
        code = install_runtime(command=["sashimono-存在しないコマンド"], on_output=lines.append)
        assert code != 0
        assert any("起動できない" in line for line in lines)


class TestActivateRuntime:
    def test_normal_runs_do_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 導入先の無い開発の環境で落ちると、起動の頭で呼ぶ app が編集画面を出す前に止まる
        monkeypatch.setattr(runtime_module, "runtime_target_dir", lambda: None)
        assert runtime_module.activate_runtime() is None

    def test_the_runtime_folder_is_put_on_the_import_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 導入先を import の道の頭へ足せないと、
        # 配布版でソフトの中から入れた faster-whisper が読めず、字幕起こしを始められない
        target = tmp_path / "runtime"
        target.mkdir()
        monkeypatch.setattr(runtime_module, "runtime_target_dir", lambda: target)
        monkeypatch.setattr(sys, "path", list(sys.path))

        assert runtime_module.activate_runtime() == target
        assert sys.path[0] == str(target)
