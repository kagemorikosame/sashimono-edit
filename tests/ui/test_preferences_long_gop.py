"""キーフレームの間隔が長い素材にも控えを作るかの設定（#430）

控えを作る間は CPU か GPU を使うので、好みが分かれる 既定は入（飛ぶたびに待たされる
理由は、知らない人には分からない） 切ったら本当に止まることを押さえる
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Project
from sashimono.engine.cache.proxy import KEYFRAME_GAP_LIMIT, MEASURED_KEYFRAME_SEEK_MS
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore


class TestSaving:
    def test_it_is_on_by_default(self) -> None:
        # 既定で切ると、録画を置いた人がなぜ飛ぶたびに待たされるのか分からないまま使う
        assert Preferences().proxy_long_gop is True

    def test_it_comes_back(self, tmp_path: Path) -> None:
        # 読み戻らないと、切った人が起動のたびに切り直すことになる
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(proxy_long_gop=False))
        assert store.load().proxy_long_gop is False

    def test_a_wrong_type_falls_back(self, tmp_path: Path) -> None:
        # 文字を真偽として読むと、"false" と書いた人の設定が入になる
        path = tmp_path / "preferences.json"
        path.write_text('{"proxy_long_gop": "false"}', encoding="utf-8")
        assert PreferenceStore(path).load().proxy_long_gop is True


class TestTheDialog:
    @pytest.fixture
    def dialog(self, qt_application: QApplication) -> PreferencesDialog:
        del qt_application
        return PreferencesDialog(Preferences())

    def test_it_shows_what_is_set(self, qt_application: QApplication) -> None:
        # 映さないと、設定を開いて OK を押しただけで入に戻る
        del qt_application
        chosen = Preferences(proxy_long_gop=False)
        assert PreferencesDialog(chosen).preferences() == chosen

    def test_turning_off_the_proxy_disables_it(self, dialog: PreferencesDialog) -> None:
        # 控えを切っていればどの素材にも作らない 押せると、効いていると思って触る
        dialog._use_proxy.setChecked(False)
        assert not dialog._proxy_long_gop.isEnabled()
        dialog._use_proxy.setChecked(True)
        assert dialog._proxy_long_gop.isEnabled()

    def test_it_names_the_limit_it_really_uses(self, dialog: PreferencesDialog) -> None:
        # 書いた秒数と判定の線が食い違うと、何秒の素材に作られるのか分からない
        found = re.search(r"キーフレームが ([\d.]+) 秒より長く", dialog._proxy_long_gop.text())
        assert found is not None
        assert Fraction(found.group(1)) == KEYFRAME_GAP_LIMIT

    def test_the_numbers_come_from_the_measured_table(self, dialog: PreferencesDialog) -> None:
        # 画面へ直に書くと、測り直したときにここだけ古いまま残る
        tip = dialog._proxy_long_gop.toolTip()
        for value in MEASURED_KEYFRAME_SEEK_MS:
            assert f"{value}ms" in tip, f"{value}ms が出ていない"


class TestTheWindowFollowsIt:
    @pytest.fixture
    def window(self, qt_application: QApplication) -> Iterator[MainWindow]:
        del qt_application
        created = MainWindow(Project.create(), confirm_unsaved=False)
        yield created
        created.close()

    def test_turning_it_off_reaches_the_store(self, window: MainWindow) -> None:
        # 届かないと、切っても作り続け、作った控えも使い続ける
        before = window._proxies
        window._apply_preferences(Preferences(proxy_long_gop=False))
        assert window._proxies is not before, "走っている変換を止めていない"
        assert window._proxies.store.long_gop is False
        assert window._preview._proxies is window._proxies.store

    def test_turning_it_back_on_reaches_the_store(self, window: MainWindow) -> None:
        # 入れ直しても届かないと、1 度切っただけで再起動するまで録画の控えが作られない
        window._apply_preferences(Preferences(proxy_long_gop=False))
        window._apply_preferences(Preferences(proxy_long_gop=True))
        assert window._proxies.store.long_gop is True

    def test_it_starts_from_the_saved_choice(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 起動のときに渡さないと、切った人の機械でも次の起動で作り始める
        del qt_application
        monkeypatch.setattr(PreferenceStore, "load", lambda self: Preferences(proxy_long_gop=False))
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            assert window._proxies.store.long_gop is False
        finally:
            window.close()
