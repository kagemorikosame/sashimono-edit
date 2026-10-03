"""追加機能を導入した直後に、再起動しなくても使えること（Issue #27）

配布版（PyInstaller で固めた exe）では、導入先が専用フォルダ
（``%LOCALAPPDATA%\\Sashimono\\runtime``）になる 起動時にそこを import の道へ
載せる手当てはあったが、**初めて導入する人は起動時にそのフォルダがまだ無い**
ので載らず、導入が済んでも「未導入」のまま、字幕起こしもアシスタントも
ボタンが押せなかった

本物の pip は走らせない 導入の代わりに、pip が置くのと同じ形（パッケージの
フォルダと ``*.dist-info``）を書き込む偽物に差し替える
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType

import pytest
from PySide6.QtWidgets import QApplication

from sashimono import runtime
from sashimono.runtime import (
    FeaturePack,
    install_runtime,
    refresh_runtime,
    restart_note,
    runtime_target_dir,
)

#: 配布版のふりをする試験は sys.executable を差し替えるので、本物は読み込んだ時点で控える
REAL_PYTHON = sys.executable


def write_distribution(site: Path, dist: str, module: str, version: str = "1.0") -> None:
    """pip が導入先に置くのと同じ形を書く"""
    package = site / module
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    # メタデータは名前で引かれるとき、フォルダ名（配布名の - を _ にした物）で探される
    info = site / f"{dist.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n", encoding="utf-8"
    )


@pytest.fixture
def isolated_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """試験の中で足した import の道と読み込んだモジュールを、終わったら元へ戻す"""
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("sashimono_fake_"):
            del sys.modules[name]
    importlib.invalidate_caches()


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_imports: None) -> Path:
    """配布版として動いている状態 戻り値は導入先の専用フォルダ"""
    del isolated_imports
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Sashimono" / "Sashimono.exe"))
    target = runtime_target_dir()
    assert target is not None
    return target


def _unique() -> tuple[str, str]:
    """試験ごとに別の名前 同じ名前だと、前の試験で読み込んだ物が残って通ってしまう"""
    token = uuid.uuid4().hex[:8]
    return f"sashimono-fake-{token}", f"sashimono_fake_{token}"


class TestFrozenFirstInstall:
    def test_the_runtime_folder_did_not_exist_at_startup(self, frozen: Path) -> None:
        # 前提 初めての人は起動時に専用フォルダが無く、起動時の手当ては何もしない
        assert not frozen.exists()
        assert runtime.activate_runtime() is None
        assert str(frozen) not in sys.path

    def test_a_first_install_is_seen_without_a_restart(self, frozen: Path) -> None:
        """直す前は、導入が済んでも status が未導入のままだった"""
        dist, module = _unique()
        pack = FeaturePack(key="fake", label="偽物", required=(dist,))
        runtime.activate_runtime()  # 起動時（まだフォルダが無い）

        write_distribution(frozen, dist, module)  # pip が済んだ
        assert pack.status().installed is False  # 道に載っていないので見えない

        assert refresh_runtime() == ()
        assert pack.status().installed is True
        assert importlib.import_module(module).VALUE == 42

    def test_a_second_pack_after_the_first_is_seen(self, frozen: Path) -> None:
        # 字幕起こしを入れたあとでアシスタントを入れる場合 フォルダは起動時から
        # 道に載っているが、中身の控えが古いと、あとから入れた物を見落とす
        first_dist, first_module = _unique()
        write_distribution(frozen, first_dist, first_module)
        runtime.activate_runtime()
        importlib.import_module(first_module)

        dist, module = _unique()
        pack = FeaturePack(key="fake", label="偽物", required=(dist,))
        write_distribution(frozen, dist, module)
        refresh_runtime()
        assert pack.status().installed is True
        assert importlib.import_module(module).VALUE == 42


class TestUpgradeInPlace:
    def test_the_old_metadata_left_by_an_upgrade_is_removed(self, frozen: Path) -> None:
        """``pip --target --upgrade`` が残した古い版の ``*.dist-info`` を消す

        残ると、メタデータを引くときに古い方を拾うことがあり、入れ直したのに
        「古い版が入っています」のまま使えない
        """
        dist, module = _unique()
        write_distribution(frozen, dist, module, "0.2.10")  # 前に入れた版
        write_distribution(frozen, dist, module, "0.2.152")  # 入れ直した版
        old = frozen / f"{module}-0.2.10.dist-info"
        pack = FeaturePack(key="fake", label="偽物", required=(f"{dist}>=0.2.152",))

        refresh_runtime()

        assert not old.exists()
        assert (frozen / f"{module}-0.2.152.dist-info").exists()
        assert pack.status().installed is True

    def test_different_packages_are_left_alone(self, frozen: Path) -> None:
        # 名前の違う物まで消すと、別の機能の導入が壊れる
        first_dist, first_module = _unique()
        second_dist, second_module = _unique()
        write_distribution(frozen, first_dist, first_module, "1.0")
        write_distribution(frozen, second_dist, second_module, "2.0")
        assert runtime.remove_stale_metadata(frozen) == ()


class TestDevelopmentInstall:
    def test_packages_added_to_a_known_folder_are_seen(
        self, tmp_path: Path, isolated_imports: None
    ) -> None:
        # 開発環境では pip が動いている環境（site-packages）へ直接入れる
        # そのフォルダはもう道に載っていて、中身の控えを持っている
        del isolated_imports
        assert not runtime.is_frozen()
        site = tmp_path / "site-packages"
        site.mkdir()
        sys.path.insert(0, str(site))
        dist, module = _unique()
        pack = FeaturePack(key="fake", label="偽物", required=(dist,))
        assert pack.status().installed is False

        write_distribution(site, dist, module)
        assert refresh_runtime() == ()  # 通常の実行では専用フォルダを使わない
        assert pack.status().installed is True
        assert importlib.import_module(module).VALUE == 42


class TestRestartNote:
    def test_modules_already_loaded_from_the_bundle_are_reported(
        self, frozen: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """同梱の numpy などを入れ直したときは、今の実行では入れ替わらない

        ここで黙ると、古い方のまま動いて分かりにくい失敗になる 再起動を勧める
        """
        _, module = _unique()
        bundled = ModuleType(module)
        bundled.__file__ = str(tmp_path / "bundle" / module / "__init__.py")
        monkeypatch.setitem(sys.modules, module, bundled)
        write_distribution(frozen, module.replace("_", "-"), module)

        loaded = refresh_runtime()
        assert loaded == (module,)
        note = restart_note(loaded)
        assert module in note
        assert "再起動" in note

    def test_submodules_of_a_namespace_package_are_looked_at(
        self, frozen: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """名前空間パッケージ（``nvidia`` など）は親に ``__file__`` が無い

        親だけを見ると、同梱の方から読み込み済みの子を見落とし、入れ替わって
        いないのに「再起動しなくても使えます」と言ってしまう
        """
        _, namespace = _unique()
        parent = ModuleType(namespace)  # 名前空間パッケージには __file__ が無い
        child = ModuleType(f"{namespace}.cublas")
        child.__file__ = str(tmp_path / "bundle" / namespace / "cublas" / "__init__.py")
        monkeypatch.setitem(sys.modules, namespace, parent)
        monkeypatch.setitem(sys.modules, f"{namespace}.cublas", child)
        (frozen / namespace / "cublas").mkdir(parents=True)

        assert refresh_runtime() == (namespace,)

    def test_modules_loaded_from_the_runtime_folder_are_not_reported(self, frozen: Path) -> None:
        # 入れたばかりの物を読んだだけで再起動を勧めると、毎回の導入で再起動させてしまう
        dist, module = _unique()
        write_distribution(frozen, dist, module)
        refresh_runtime()
        importlib.import_module(module)
        assert refresh_runtime() == ()

    def test_a_module_updated_in_place_is_reported(self, frozen: Path) -> None:
        """専用フォルダから読み込み済みの物を、同じ場所の新しい版で上書きしたとき

        場所だけを比べると「専用フォルダの物だから問題ない」と見て、古い版の
        まま動いているのに「再起動しなくても使えます」と言ってしまう
        """
        dist, module = _unique()
        write_distribution(frozen, dist, module, "1.0")
        refresh_runtime()
        importlib.import_module(module)
        source = frozen / module / "__init__.py"
        before = source.stat().st_mtime_ns

        snapshot = runtime.snapshot_runtime_modules()  # 導入を始める前に控える
        source.write_text("VALUE = 43\n", encoding="utf-8")  # pip が新しい版で上書きした
        os.utime(source, ns=(before + 10**9, before + 10**9))

        assert refresh_runtime(snapshot) == (module,)

    def test_overlapping_installs_keep_their_own_snapshot(self, frozen: Path) -> None:
        """字幕起こしとアシスタントの導入が重なっても、控えが混ざらない

        控えを 1 か所で共有すると、あとから始めた導入の控えで上書きされ、先の
        導入が上書きされたモジュールを見落として再起動を勧めない
        """
        dist, module = _unique()
        write_distribution(frozen, dist, module, "1.0")
        refresh_runtime()
        importlib.import_module(module)
        source = frozen / module / "__init__.py"
        stamp = source.stat().st_mtime_ns

        first = runtime.snapshot_runtime_modules()  # 先に始めた導入の控え
        # あとから始めた導入が、何もしない子プロセスで流れを最後まで通す
        assert install_runtime(command=[REAL_PYTHON, "-c", "pass"]) == 0
        assert refresh_runtime({}) == ()
        source.write_text("VALUE = 43\n", encoding="utf-8")  # 先の導入の pip が上書きした
        os.utime(source, ns=(stamp + 10**9, stamp + 10**9))

        assert refresh_runtime(first) == (module,)

    def test_nothing_to_restart_says_so(self) -> None:
        # 再起動する物が無いのに再起動を勧めると、導入のたびに要らない再起動をさせる
        assert "再起動しなくても" in restart_note(())

    def test_an_install_that_cannot_be_seen_asks_for_a_restart(self) -> None:
        # 再起動を頼まないと、押せないボタンが残ったまま次にすることが分からない
        assert "再起動" in restart_note((), visible=False)


def _fake_installer(write: Callable[[], object], code: int = 0) -> Callable[..., int]:
    """``install_runtime`` の差し替え pip の代わりに、置くはずの物を書く"""

    def install(
        *,
        pack: object = None,
        command: Sequence[str],
        on_output: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> int:
        del pack, command, should_cancel
        if on_output is not None:
            on_output("Successfully installed")
        if code == 0:
            write()
        return code

    return install


def _slow_cancelled_installer() -> Callable[..., int]:
    """中断を頼まれてから、少し遅れて失敗を返す ``install_runtime`` の差し替え

    本物の pip も、止めるよう頼んでから子プロセスが終わるまで間がある その間に
    画面が「終わった」と読むかどうかを見分けるため、わざと遅らせる
    """

    def install(
        *,
        pack: object = None,
        command: Sequence[str],
        on_output: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> int:
        del pack, command, on_output
        assert should_cancel is not None
        for _ in range(500):
            if should_cancel():
                break
            threading.Event().wait(0.01)
        threading.Event().wait(0.3)
        return 1

    return install


def _wait_for(done: Callable[[], bool], app: QApplication) -> None:
    """導入のスレッドが終わり、画面側の見張りが拾うまで待つ"""
    finished = threading.Event()
    for _ in range(500):
        app.processEvents()
        if done():
            finished.set()
            break
        threading.Event().wait(0.01)
    assert finished.is_set(), "導入が終わらなかった"


class TestSetupSection:
    def test_the_section_reports_ready_right_after_installing(
        self, frozen: Path, monkeypatch: pytest.MonkeyPatch, qt_application: QApplication
    ) -> None:
        from sashimono.ui import setup

        dist, module = _unique()
        pack = FeaturePack(key="fake", label="偽物", required=(dist,))
        monkeypatch.setattr(
            setup,
            "install_runtime",
            _fake_installer(lambda: write_distribution(frozen, dist, module)),
        )
        section = setup.SetupSection(pack)
        ready: list[bool] = []
        section.changed.connect(ready.append)
        results: list[bool] = []
        section.finished.connect(results.append)

        section.start()
        _wait_for(lambda: bool(results), qt_application)

        assert results == [True]
        assert ready[-1] is True
        assert section.status.installed is True
        assert "再起動しなくても" in section._status.text()
        section.deleteLater()

    def test_a_missing_command_is_not_called_usable(
        self, frozen: Path, monkeypatch: pytest.MonkeyPatch, qt_application: QApplication
    ) -> None:
        """pip で入る物は揃ったが、別に要るコマンドが無いとき

        「そのまま使えます」と出すと、使えない機能を使えると案内することになる
        """
        from sashimono.ui import setup

        dist, module = _unique()
        pack = FeaturePack(
            key="fake",
            label="偽物",
            required=(dist,),
            commands=("sashimono-fake-command",),
            locate=lambda _name: None,
        )
        monkeypatch.setattr(
            setup,
            "install_runtime",
            _fake_installer(lambda: write_distribution(frozen, dist, module)),
        )
        section = setup.SetupSection(pack)
        results: list[bool] = []
        section.finished.connect(results.append)

        section.start()
        _wait_for(lambda: bool(results), qt_application)

        assert section.status.installed is True
        assert section.note == ""
        assert "使えます" not in section._status.text()
        assert "sashimono-fake-command" in section._status.text()
        section.deleteLater()

    def test_a_cancelled_install_is_not_called_a_success(
        self, frozen: Path, monkeypatch: pytest.MonkeyPatch, qt_application: QApplication
    ) -> None:
        """中断を頼んだ瞬間に「終わった」と読まないこと

        読むと、pip が走っている最中に「再起動しなくても使えます」と出て、
        まだ入っていない機能を使えると案内してしまう
        """
        from sashimono.ui import setup

        dist, _module = _unique()
        pack = FeaturePack(key="fake", label="偽物", required=(dist,))
        monkeypatch.setattr(setup, "install_runtime", _slow_cancelled_installer())
        section = setup.SetupSection(pack)
        results: list[bool] = []
        section.finished.connect(results.append)

        section.start()
        section.cancel()
        _wait_for(lambda: bool(results), qt_application)

        assert results == [False]
        assert "使えます" not in section._status.text()
        section.deleteLater()

    def test_a_failed_install_does_not_claim_success(
        self, frozen: Path, monkeypatch: pytest.MonkeyPatch, qt_application: QApplication
    ) -> None:
        from sashimono.ui import setup

        dist, _module = _unique()
        pack = FeaturePack(key="fake", label="偽物", required=(dist,))
        monkeypatch.setattr(setup, "install_runtime", _fake_installer(lambda: None, code=1))
        section = setup.SetupSection(pack)
        results: list[bool] = []
        section.finished.connect(results.append)

        section.start()
        _wait_for(lambda: bool(results), qt_application)

        assert results == [False]
        assert "導入が終わりました" not in section._status.text()
        section.deleteLater()
