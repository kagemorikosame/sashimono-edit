"""画面の中身（文字と顔の位置 #324）の設定と、部品の導入の窓

いつ調べるか・場面ごとに何枚調べるかは、機械の速さと使い方で答えが分かれる 設定画面から
変えられ、変えたらその場で効き、切ったら本当に止まることを見る 部品は設定画面のボタンから
開く導入の窓で入れる
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QPushButton

from sashimono.core.commands import AddMedia
from sashimono.core.model import MediaItem, Project, VideoStreamInfo
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import ScreenAnalyzer
from sashimono.engine.vision import ScreenKind
from sashimono.engine.vision.sampling import MEASURED_SCREEN_SAMPLE
from sashimono.runtime import FeaturePack
from sashimono.ui import main_window as main_window_module
from sashimono.ui import vision_setup
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import (
    SCREEN_ANALYSIS_CHOICES,
    SCREEN_SAMPLES,
    PreferencesDialog,
)
from sashimono.ui.vision_setup import VisionSetupDialog
from sashimono.ui.workspace import (
    SCREEN_ANALYSIS_MODES,
    SCREEN_ON_DEMAND,
    SCREEN_ON_IMPORT,
    Preferences,
    PreferenceStore,
)


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    開いたまま待つと、試験が人の操作を待って止まる 設定を当てる道は窓を出さずに
    通るはずなので、出たら作りが変わったということ 導入の窓を開く試験は、その試験の中で
    開く所を差し替える
    """

    def refuse(*_arguments: object, **_options: object) -> None:
        raise AssertionError("試験の途中で窓が開いた")

    for owner, name in (
        (QDialog, "exec"),
        (QDialog, "open"),
        (QDialog, "show"),
        (QMainWindow, "show"),
        (QMessageBox, "exec"),
        (QMessageBox, "warning"),
        (QMessageBox, "information"),
        (QMessageBox, "question"),
        (QMessageBox, "critical"),
    ):
        monkeypatch.setattr(owner, name, refuse)


def _video(name: str = "v.mp4") -> MediaItem:
    return MediaItem(
        path=Path(name),
        duration=Fraction(10),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1280,
                height=720,
                frame_rate=FrameRate(30),
                time_base=Fraction(1, 30),
                codec="h264",
            ),
        ),
    )


class _Spy:
    """画面の中身の係に頼まれた物を数える 本当に調べに行かせない（ファイルは無い）"""

    def __init__(self, screen: ScreenAnalyzer, monkeypatch: pytest.MonkeyPatch) -> None:
        self.requests: list[tuple[str, ScreenKind]] = []
        self.cancelled = 0

        def request(media: MediaItem, kind: ScreenKind) -> None:
            self.requests.append((str(media.id), kind))

        def cancel() -> None:
            self.cancelled += 1

        monkeypatch.setattr(screen, "request", request)
        monkeypatch.setattr(screen, "cancel", cancel)


def _parts(monkeypatch: pytest.MonkeyPatch, *, ready: bool) -> None:
    """画面の解析の部品が入っているかを決める（本物の導入先は見ない）"""
    status = SimpleNamespace(ready=ready)
    monkeypatch.setattr(main_window_module, "VISION_PACK", SimpleNamespace(status=lambda: status))


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


class TestTheDefaults:
    def test_only_asked_materials_are_looked_at(self) -> None:
        # 壊れると、部品を入れただけの人の機械で、読み込むたびに裏で CPU を使う
        assert Preferences().screen_analysis == SCREEN_ON_DEMAND
        assert Preferences().screen_samples == 4

    def test_they_come_back(self, tmp_path: Path) -> None:
        # 保存できないと、選び直しても次の起動で既定へ戻る
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(screen_analysis=SCREEN_ON_IMPORT, screen_samples=8))
        loaded = store.load()
        assert loaded.screen_analysis == SCREEN_ON_IMPORT and loaded.screen_samples == 8

    @pytest.mark.parametrize("written", ['"often"', "3", "true", '"8"'], ids=str)
    def test_a_broken_value_falls_back(self, tmp_path: Path, written: str) -> None:
        # 壊れた値で起動が止まるか、知らない値のまま当てると、設定画面に何も選ばれない
        path = tmp_path / "preferences.json"
        path.write_text(
            f'{{"screen_analysis": {written}, "screen_samples": {written}}}', encoding="utf-8"
        )
        loaded = PreferenceStore(path).load()
        assert loaded.screen_analysis == SCREEN_ON_DEMAND
        assert loaded.screen_samples == 4


class TestTheDialog:
    def test_it_shows_and_returns_what_is_set(self, qt_application: QApplication) -> None:
        # 開いて OK を押しただけで値が変わると、選んだ設定が黙って戻る
        del qt_application
        dialog = PreferencesDialog(Preferences(screen_analysis=SCREEN_ON_IMPORT, screen_samples=8))
        returned = dialog.preferences()
        assert returned.screen_analysis == SCREEN_ON_IMPORT
        assert returned.screen_samples == 8

    def test_every_choice_can_be_picked(self) -> None:
        # 選べない値があると、設定ファイルを手で書かないと変えられない
        assert {value for value, _text in SCREEN_ANALYSIS_CHOICES} == set(SCREEN_ANALYSIS_MODES)
        assert Preferences().screen_samples in [count for _label, count in SCREEN_SAMPLES]

    def test_the_measured_time_is_shown(self, qt_application: QApplication) -> None:
        # 測った値が無いと、どの枚数を選べばどれだけ待つのか分からないまま選ぶ
        del qt_application
        dialog = PreferencesDialog(Preferences())
        assert MEASURED_SCREEN_SAMPLE in dialog._screen_samples.toolTip()

    def test_the_parts_button_opens_the_install_window(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、画面の解析の部品をどこから入れればよいか分からず、AI の断りの案内が嘘になる
        del qt_application
        opened: list[QDialog] = []

        def record(self: QDialog) -> int:
            opened.append(self)
            return 0

        monkeypatch.setattr(VisionSetupDialog, "exec", record)
        dialog = PreferencesDialog(Preferences())
        buttons = [b for b in dialog.findChildren(QPushButton) if b.text() == "画面の解析の部品…"]
        assert len(buttons) == 1
        buttons[0].click()
        assert len(opened) == 1


class TestTheInstallWindow:
    def test_closing_while_installing_stops_first(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、導入の途中で窓を閉じたとき、欄が消えた後も裏の導入が錠を持ったまま残る
        del qt_application
        pack = FeaturePack(key="試験", label="試験", required=("pip",))
        dialog = VisionSetupDialog(pack=pack)
        busy = [True]
        cancelled: list[bool] = []
        monkeypatch.setattr(type(dialog.section), "busy", property(lambda _self: busy[0]))
        monkeypatch.setattr(dialog.section, "cancel", lambda: cancelled.append(True))
        closed: list[int] = []
        dialog.finished.connect(closed.append)
        dialog.reject()
        assert cancelled == [True]
        assert closed == []
        busy[0] = False
        vision_setup.VisionSetupDialog._close_when_idle(dialog)
        assert closed == [QDialog.DialogCode.Rejected.value]

    def test_the_pack_shows_what_it_fetches(self, qt_application: QApplication) -> None:
        # 壊れると、pip の外で何をどこから落とすのか見せないまま落とし始める
        del qt_application
        dialog = VisionSetupDialog()
        text = dialog.section._status.text()
        assert "顔の検出のモデル" in text
        assert "media.githubusercontent.com" in text


class TestTheWindowFollowsThem:
    def test_on_import_asks_for_both_when_the_parts_are_in(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 選んでも読み込みで頼まないと、AI が頼んだときに場面ごとに調べるのを待つ
        _parts(monkeypatch, ready=True)
        spy = _Spy(window._screen, monkeypatch)
        window._apply_preferences(Preferences(screen_analysis=SCREEN_ON_IMPORT))
        media = _video()
        window._request_analysis(media)
        assert spy.requests == [(str(media.id), ScreenKind.TEXT), (str(media.id), ScreenKind.FACES)]

    def test_without_the_parts_nothing_is_asked(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、部品の無い機械で読み込むたびに「部品が無い」失敗を積む
        _parts(monkeypatch, ready=False)
        spy = _Spy(window._screen, monkeypatch)
        window._apply_preferences(Preferences(screen_analysis=SCREEN_ON_IMPORT))
        window._request_analysis(_video())
        assert spy.requests == []

    def test_on_demand_does_not(self, window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
        # 壊れると（AI が頼んだときだけを選んでも読み込みで調べると）、選んだ意味が無い
        _parts(monkeypatch, ready=True)
        spy = _Spy(window._screen, monkeypatch)
        window._apply_preferences(Preferences(screen_analysis=SCREEN_ON_DEMAND))
        window._request_analysis(_video())
        assert spy.requests == []

    def test_turning_it_off_stops_the_looking(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 止めないと、切っても裏で調べ続ける
        _parts(monkeypatch, ready=True)
        spy = _Spy(window._screen, monkeypatch)
        window._apply_preferences(Preferences(screen_analysis=SCREEN_ON_IMPORT))
        window._apply_preferences(Preferences(screen_analysis=SCREEN_ON_DEMAND))
        assert spy.cancelled == 1

    def test_turning_it_on_looks_at_what_is_loaded(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 入れても読み込み直すまで効かないと、入れたのに AI が頼んだときに待たされる
        _parts(monkeypatch, ready=True)
        media = _video()
        window.execute(AddMedia(media))
        window._apply_preferences(Preferences(screen_analysis=SCREEN_ON_DEMAND))
        spy = _Spy(window._screen, monkeypatch)
        window._apply_preferences(Preferences(screen_analysis=SCREEN_ON_IMPORT))
        assert {kind for _id, kind in spy.requests} == set(ScreenKind)
        assert spy.cancelled == 0

    def test_the_assistant_can_ask_even_when_off(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 切った人のところで AI の頼みまで断ると、道具が文字も顔も 1 つも返せない
        window._apply_preferences(Preferences(screen_analysis=SCREEN_ON_DEMAND))
        spy = _Spy(window._screen, monkeypatch)
        media = _video()
        window.request_screen(media, ScreenKind.FACES)
        assert spy.requests == [(str(media.id), ScreenKind.FACES)]

    def test_the_count_reaches_the_analyzer(self, window: MainWindow) -> None:
        # 届かないと、8 枚を選んでも 4 枚しか調べない
        window._apply_preferences(Preferences(screen_samples=8))
        assert window._screen.per_scene == 8

    def test_the_window_starts_with_the_saved_count(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 起動した窓が既定の枚数で調べると、保存した設定が次の起動で効かない
        del qt_application
        monkeypatch.setattr(PreferenceStore, "load", lambda _self: Preferences(screen_samples=2))
        created = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            assert created._screen.per_scene == 2
        finally:
            created.close()
