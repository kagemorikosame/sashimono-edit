"""本物の窓で、人が画面で見ている物を AI も読めるか（読み取りの同等性 #352）

偽のホストの試験（``tests/ai/test_screen_reading.py``）は道具の形を見る ここは本物の
:class:`MainWindow` が、表示・パネルの選択・書き出しの窓・互換性レポート・設定の値を、道具の
読む口へ本当に渡しているかを見る 窓（確かめや知らせ）が開いたらその場で落とす
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.ai.ledger import LEDGER, Status
from sashimono.ai.operations import find_operation
from sashimono.compat.aviutl import report as report_module
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.core.commands import ParamPath
from sashimono.core.model import MediaItem, Transcript
from sashimono.ui.export_dialog import ExportDialog
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import make_loaded


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 読むだけの道具が人に何かを尋ねたり知らせたりしてはいけない"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QMessageBox, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)


@pytest.fixture
def window(
    qt_application: QApplication, video_media: MediaItem, transcript: Transcript
) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(make_loaded(video_media, transcript), confirm_unsaved=False)
    created.resize(1200, 800)
    yield created
    created._chat.close_session()
    created.close()
    created.deleteLater()


def run(window: MainWindow, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, tool
    return operation(window, arguments)


def dig(settings: object, place: str) -> object:
    """``preferences.timeline_snap`` のように点でつないだ在りかを辿る 無ければ ``KeyError``"""
    value = settings
    for part in place.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(place)
        value = value[part]
    return value


class TestReadOnlyRowsAreReallyReadable:
    def test_every_read_only_value_is_in_the_settings(self, window: MainWindow) -> None:
        """壊れると、読むだけと決めた操作（字幕パネル・更新・磁石など）の値が設定の写しに無く、
        AI が「読める」と台帳に数えたまま、人が見ている値を読めない"""
        settings = window.app_settings()
        rows = [entry for entry in LEDGER if entry.status is Status.READ_ONLY]
        assert rows
        missing = []
        for entry in rows:
            for place in entry.reads:
                try:
                    dig(settings, place)
                except KeyError:
                    missing.append(f"{entry.key}: {place}")
        assert not missing

    def test_a_place_that_is_not_there_is_caught(self, window: MainWindow) -> None:
        """壊れると、在りかを書き違えた行（設定の名前を変えた後など）を上の見張りが見逃す"""
        with pytest.raises(KeyError):
            dig(window.app_settings(), "preferences.no_such_setting")

    def test_the_update_state_is_readable(self, window: MainWindow) -> None:
        """壊れると、「更新が来ている？」「前の版に戻せる？」を AI が答えられず、人がヘルプの
        窓を開いて確かめることになる"""
        update = window.app_settings()["update"]
        assert isinstance(update, dict)
        assert set(update) == {"checks", "checking", "ready_version", "can_roll_back"}


class TestWhatIsShown:
    def test_zooming_changes_the_range_the_ai_reads(self, window: MainWindow) -> None:
        """壊れると、人が拡大して見ている所と AI の読む「見えている範囲」が食い違い、
        「ここ」と言われた所と違う所を直す"""
        # 窓を出さない試験ではパネルが並べられず幅が無い 人の画面に近い幅を与える
        window._timeline.resize(1000, 300)
        before = run(window, "get_screen")["timeline"]
        window._timeline.zoom(4.0)
        after = run(window, "get_screen")["timeline"]
        assert after["pixels_per_second"] == pytest.approx(before["pixels_per_second"] * 4, 0.01)
        layout = window._timeline.view_layout
        expected = layout.visible_range(window._timeline.width())
        assert (after["from_frame"], after["to_frame"]) == expected
        assert after["to_frame"] - after["from_frame"] < before["to_frame"] - before["from_frame"]

    def test_the_graph_editor_value_is_read(self, window: MainWindow) -> None:
        """壊れると、「グラフに出ている動きをもっと滑らかに」と言われた AI が、どの値の曲線かを
        読めず、別の値のキーを直す"""
        clip = window._document.project.timeline.tracks[0].clips[0].id
        window._graph.set_path(ParamPath.of_clip(clip, "opacity"))
        window._graph_dock.show()
        graph = run(window, "get_screen")["graph_editor"]
        assert graph == {
            "shown": True,
            "value": {"clip_id": str(clip), "name": "opacity", "target": "clip"},
        }
        assert run(window, "get_clip", clip_id=graph["value"]["clip_id"])["opacity"] == 1.0
        window._graph_dock.hide()
        assert run(window, "get_screen")["graph_editor"]["shown"] is False

    def test_the_person_export_progress_is_read(self, window: MainWindow) -> None:
        """壊れると、人が書き出しの窓で書き出している最中に、AI が終わったと思って次へ進む"""
        assert run(window, "get_screen")["progress"]["person_export"] is None
        dialog = ExportDialog(window._document.project, window)
        try:
            # 書き出しを走らせずに、走っている窓の形だけを作る（中身の書き出しは別の試験が見る）
            dialog._worker = object()  # type: ignore[assignment]  # 走っている印だけが要る
            dialog._on_progress(0.25)
            window._person_export = dialog
            assert run(window, "get_screen")["progress"]["person_export"] == 0.25
        finally:
            dialog._worker = None
            window._person_export = None
            dialog.deleteLater()


class TestWhatIsChosen:
    def test_the_media_and_subtitle_rows_chosen_are_read(
        self, window: MainWindow, transcript: Transcript
    ) -> None:
        """壊れると、素材一覧と字幕パネルで人が選んだ物を AI が読めず、「選んだ行を直して」で
        別の行を直す"""
        media = window._document.project.media[0]
        window._media_pool._list.setCurrentRow(0)
        window._subtitles.select_media(media.id)
        window._subtitles._table.selectRow(1)
        selection = run(window, "get_selection")
        assert selection["media_id"] == str(media.id)
        assert selection["subtitles"]["media_id"] == str(media.id)
        assert selection["subtitles"]["segment_ids"] == [str(transcript.segments[1].id)]


class TestCompatReport:
    def test_the_window_hands_over_the_same_records(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、互換性レポートの窓に並ぶ記録と AI の読む記録が食い違い、AI が「穴は無い」と
        答えたまま、人の窓には読めなかったスクリプトが並ぶ"""
        recorded = CompatibilityReport()
        recorded.note_missing("obj.試しの未実装")
        monkeypatch.setattr(report_module, "global_report", recorded)
        result = run(window, "get_compat_report")
        assert result["summary"] == recorded.summary()
        assert [row["name"] for row in result["records"]] == ["obj.試しの未実装"]
