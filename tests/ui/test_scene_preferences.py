"""場面の解析の設定（#321）

読み込んだときに場面を調べるか・何本のスレッドで読むかは、機械と使い方で答えが
分かれる 設定画面から変えられ、変えたらその場で効き、切ったら本当に止まることを見る
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox

from sashimono.core.commands import AddMedia
from sashimono.core.model import MediaItem, Project, VideoStreamInfo
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import MediaAnalyzer
from sashimono.engine.scenes import MEASURED_SCENE_SECONDS
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import (
    SCENE_ANALYSIS_CHOICES,
    SCENE_THREADS,
    PreferencesDialog,
)
from sashimono.ui.workspace import (
    SCENE_ANALYSIS_MODES,
    SCENES_ALWAYS,
    SCENES_ON_DEMAND,
    SCENES_WITH_AI,
    Preferences,
    PreferenceStore,
)


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    開いたまま待つと、試験が人の操作を待って止まる 設定を当てる道は窓を出さずに
    通るはずなので、出たら作りが変わったということ
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
    """解析係に頼まれた物を数える 本当に読みに行かせない（ファイルは無い）"""

    def __init__(self, analyzer: MediaAnalyzer, monkeypatch: pytest.MonkeyPatch) -> None:
        self.requests: list[bool] = []
        self.scene_requests: list[str] = []
        self.cancelled = 0

        def request(media: MediaItem, *, on_ready: object = None, scenes: bool = False) -> None:
            del media, on_ready
            self.requests.append(scenes)

        def request_scenes(media: MediaItem, *, on_ready: object = None) -> None:
            del on_ready
            self.scene_requests.append(str(media.id))

        def cancel() -> None:
            self.cancelled += 1

        monkeypatch.setattr(analyzer, "request", request)
        monkeypatch.setattr(analyzer, "request_scenes", request_scenes)
        monkeypatch.setattr(analyzer, "cancel_scenes", cancel)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


class TestTheDefaults:
    def test_only_assistant_users_pay_for_it(self) -> None:
        # いつも調べる既定にすると、AI を使わない人の機械で読み込むたびに CPU を使う
        assert Preferences().scene_analysis == SCENES_WITH_AI

    def test_two_threads_by_default(self) -> None:
        # すべてのコアを使う既定にすると、裏で読む間に編集と再生が重くなる
        assert Preferences().scene_threads == 2

    def test_they_come_back(self, tmp_path: Path) -> None:
        # 保存できないと、選び直しても次の起動で既定へ戻る
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(scene_analysis=SCENES_ALWAYS, scene_threads=0))
        loaded = store.load()
        assert loaded.scene_analysis == SCENES_ALWAYS and loaded.scene_threads == 0

    @pytest.mark.parametrize(
        "written", ['"often"', "3", "true", '"2"', "-1"], ids=["word", "3", "bool", "text", "neg"]
    )
    def test_a_broken_value_falls_back(self, tmp_path: Path, written: str) -> None:
        # 壊れた値で起動が止まるか、知らない値のまま当てると、設定画面に何も選ばれない
        path = tmp_path / "preferences.json"
        path.write_text(
            f'{{"scene_analysis": {written}, "scene_threads": {written}}}', encoding="utf-8"
        )
        loaded = PreferenceStore(path).load()
        assert loaded.scene_analysis == SCENES_WITH_AI
        assert loaded.scene_threads == 2


class TestTheDialog:
    def test_it_shows_and_returns_what_is_set(self, qt_application: QApplication) -> None:
        # 開いて OK を押しただけで値が変わると、選んだ設定が黙って戻る
        del qt_application
        chosen = Preferences(scene_analysis=SCENES_ON_DEMAND, scene_threads=4)
        dialog = PreferencesDialog(chosen)
        returned = dialog.preferences()
        assert returned.scene_analysis == SCENES_ON_DEMAND
        assert returned.scene_threads == 4

    def test_every_mode_and_count_can_be_chosen(self) -> None:
        # 選べない値があると、設定ファイルを手で書かないと変えられない
        assert {value for value, _text in SCENE_ANALYSIS_CHOICES} == set(SCENE_ANALYSIS_MODES)
        assert Preferences().scene_threads in [threads for _label, threads in SCENE_THREADS]

    def test_the_measured_times_are_shown(self, qt_application: QApplication) -> None:
        # 測った値が無いと、どれを選べば何秒で終わるのか分からないまま選ぶ
        del qt_application
        dialog = PreferencesDialog(Preferences())
        tip = dialog._scene_threads.toolTip()
        for seconds in MEASURED_SCENE_SECONDS.values():
            assert f"{seconds:g} 秒" in tip


class TestTheWindowFollowsThem:
    def test_always_asks_for_scenes_on_import(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 選んでも読み込みで頼まないと、AI が頼んだときに全フレーム読むのを待つ
        spy = _Spy(window._analyzer, monkeypatch)
        window._apply_preferences(Preferences(scene_analysis=SCENES_ALWAYS))
        window._request_analysis(_video())
        assert spy.requests == [True]

    def test_on_demand_does_not(self, window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
        # 切ったのに読み込みで頼むと、切った意味が無い
        spy = _Spy(window._analyzer, monkeypatch)
        window._apply_preferences(Preferences(scene_analysis=SCENES_ON_DEMAND))
        window._request_analysis(_video())
        assert spy.requests == [False]

    @pytest.mark.parametrize("installed", [True, False])
    def test_with_ai_follows_the_assistant(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, installed: bool
    ) -> None:
        # AI の部品が無い人の機械で読むと、使わない一覧のために CPU を使う
        monkeypatch.setattr(window, "_assistant_installed", lambda: installed)
        spy = _Spy(window._analyzer, monkeypatch)
        window._apply_preferences(Preferences(scene_analysis=SCENES_WITH_AI))
        window._request_analysis(_video())
        assert spy.requests == [installed]

    def test_turning_it_off_stops_the_reading(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 止めないと、切っても裏で長い動画を全フレーム読み続ける
        spy = _Spy(window._analyzer, monkeypatch)
        window._apply_preferences(Preferences(scene_analysis=SCENES_ALWAYS))
        window._apply_preferences(Preferences(scene_analysis=SCENES_ON_DEMAND))
        assert spy.cancelled == 1

    def test_turning_it_on_reads_what_is_loaded(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 入れても読み込み直すまで効かないと、入れたのに AI が頼んだときに待たされる
        media = _video()
        window.execute(AddMedia(media))
        window._apply_preferences(Preferences(scene_analysis=SCENES_ON_DEMAND))
        spy = _Spy(window._analyzer, monkeypatch)
        window._apply_preferences(Preferences(scene_analysis=SCENES_ALWAYS))
        assert spy.scene_requests == [str(media.id)]
        assert spy.cancelled == 0

    def test_the_assistant_can_ask_even_when_off(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 切った人のところで AI の頼みまで断ると、道具が一覧を 1 つも返せない
        window._apply_preferences(Preferences(scene_analysis=SCENES_ON_DEMAND))
        spy = _Spy(window._analyzer, monkeypatch)
        media = _video()
        window.request_scenes(media)
        assert spy.scene_requests == [str(media.id)]

    def test_the_thread_count_reaches_the_analyzer(self, window: MainWindow) -> None:
        # 届かないと、1 本に絞っても全部のコアで読む
        window._apply_preferences(Preferences(scene_threads=1))
        assert window._analyzer._scene_threads == 1

    def test_the_window_starts_with_the_saved_count(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 起動した窓が既定の数で読むと、保存した設定が次の起動で効かない
        del qt_application
        monkeypatch.setattr(PreferenceStore, "load", lambda _self: Preferences(scene_threads=4))
        created = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            assert created._analyzer._scene_threads == 4
        finally:
            created.close()
