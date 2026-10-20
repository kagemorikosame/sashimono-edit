"""AI の道具 analyze_audio と audio_graph（#322 #338）を、本物の窓と本物の検査で確かめる

音の場所が分かっている WAV を PyAV で焼いて作品を組み、AI と同じ道（ブリッジ）から呼ぶ
音だけを調べるので GPU は使わない（CI でも走る） グラフは本物の Qt で描いて、PNG を読み戻す
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.operations import ImageResult, find_operation
from sashimono.ai.sound_graph import Band, SoundGraph
from sashimono.core.commands import AddClip
from sashimono.core.model import Clip, MediaItem, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.engine.checks.profile import SPEECH
from sashimono.engine.decode.probe import probe_media
from sashimono.ui.ai_sound_graph import BAND_COLORS, UNANALYZED, draw_sound_graph
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import allow_sending
from tests.check_media import HEIGHT, WIDTH, tone, write_tone


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    音の道具は確かめを挟まない読むだけの道具 窓が出たら作りが変わったということで、
    開いたまま待つと試験が人の操作を待って止まる
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
        (QFileDialog, "getSaveFileName"),
        (QFileDialog, "getOpenFileName"),
    ):
        monkeypatch.setattr(owner, name, refuse)


@pytest.fixture(scope="module")
def sound(tmp_path_factory: pytest.TempPathFactory) -> MediaItem:
    """-14 dBFS の 1 kHz（-14 LUFS）が 12 秒"""
    folder = tmp_path_factory.mktemp("audio_tools")
    return probe_media(write_tone(folder / "音.wav", tone(12.0, -14.0)))


def _project(sound: MediaItem) -> Project:
    """0〜3 秒と 6〜9 秒に音を置き、間の 3 秒は何も置かない作品（30fps）"""
    settings = ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
    base = Project.create(settings, media=(sound,))
    track = Track(kind=TrackKind.AUDIO, name="A1")
    project = base.with_timeline(replace(base.timeline, tracks=(track,)))
    for start, source in ((0, 0), (180, 6)):
        clip = Clip(
            timeline_start=start, duration=90, media_id=sound.id, source_in=Fraction(source)
        )
        project = AddClip(track.id, clip).apply(project)
    return project


@pytest.fixture
def window(qt_application: QApplication, sound: MediaItem) -> Iterator[MainWindow]:
    del qt_application
    opened = MainWindow(_project(sound), confirm_unsaved=False)
    yield opened
    opened.close()
    opened.deleteLater()


def _call(window: MainWindow, name: str, arguments: dict[str, Any]) -> Any:
    """AI のスレッドからブリッジ越しに道具を呼ぶ 画面のスレッドは仕事を拾い続ける"""
    operation = find_operation(name)
    assert operation is not None
    bridge = allow_sending(EditorBridge(window, auto_approve=False))
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    deadline = time.monotonic() + 120.0
    while thread.is_alive():
        bridge.pump()
        QApplication.processEvents()
        thread.join(0.01)
        if time.monotonic() > deadline:
            bridge.cancel()
            thread.join(5.0)
            pytest.fail("道具が返らなかった")
    if "error" in box:
        raise box["error"]
    return box["result"]


class TestTimeline:
    def test_the_mixed_sound_is_read_as_numbers(self, window: MainWindow) -> None:
        # 壊れると、書き出す音（混ぜた音）の推移を AI が読めず、置き忘れた 3 秒の無音や
        # 音量の基準からのずれを、再生せずに見つけられない
        result = _call(window, "analyze_audio", {"wait_seconds": 120, "points": 9})
        assert result["complete"] is True
        assert result["summary"]["integrated_lufs"] == pytest.approx(-14.0, abs=0.3)
        silences = [row for row in result["segments"] if row["kind"] == "silence"]
        assert [(row["start_frame"], row["end_frame"]) for row in silences] == [(90, 180)]
        assert [row[0] for row in result["levels"]["rows"]] == list(range(0, 270, 30))

    def test_a_finished_check_is_not_mixed_again(self, window: MainWindow) -> None:
        # 壊れると、check_output の直後に音の道具を呼ぶたびに検査を頼み直し、同じ音を混ぜ直す
        _call(window, "check_output", {"wait_seconds": 120, "checks": ["silence", "loudness"]})
        before = window.check_state().request
        _call(window, "analyze_audio", {"wait_seconds": 120})
        assert window.check_state().request is before

    def test_the_graph_comes_back_as_a_real_picture(self, window: MainWindow) -> None:
        # 壊れると、グラフの絵が作れず（または大きさが頼みと違い）、AI が音の推移を目で
        # 確かめられない 帯の色が説明と合わないと、無音を話し声と読み違える
        result = _call(window, "audio_graph", {"wait_seconds": 120, "width": 640, "marks": [120]})
        assert isinstance(result, ImageResult)
        image = QImage.fromData(result.png)
        assert (image.width(), image.height()) == (640, 320)
        assert "1) 00:00:04:00（120）" in result.caption


class TestDrawing:
    def test_the_bands_and_the_unanalyzed_part_use_their_colours(
        self, qt_application: QApplication
    ) -> None:
        # 壊れると、帯の色が返事の説明（青=話し声）と違い、まだ測っていない所が無音の白に
        # 見え、AI が絵から逆の判断をする
        del qt_application
        graph = SoundGraph(
            width=600,
            height=240,
            momentary=(None,) * 544,
            short_term=(None,) * 544,
            clipped=(),
            bands=(Band(44, 300, SPEECH),),
            subtitles=(),
            target=-14.0,
            time_ticks=((44, "00:00:00:00"),),
            marks=(),
            analyzed_until=300,
        )
        image = QImage.fromData(draw_sound_graph(graph))
        band_y = graph.bottom + 2 + 6
        assert image.pixelColor(150, band_y) == BAND_COLORS[SPEECH]
        assert image.pixelColor(400, (graph.top + graph.bottom) // 2) == UNANALYZED
        assert image.pixelColor(150, (graph.top + graph.bottom) // 2 + 3) == QColor(255, 255, 255)
