"""空のプロジェクトへ最初の動画を置くとき、プロジェクトを動画に合わせるか（Issue #27）

60fps の動画を 30fps のプロジェクトへ黙って置くと、書き出しで半分のコマが捨てられる
尋ねる・常に合わせる・合わせないを、設定で選べることを確かめる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddMedia, VideoFormat
from sashimono.core.model import (
    MediaItem,
    Project,
    ProjectSettings,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.ui import media_match
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.media_fixtures import make_sample

_SIXTY = FrameRate(60)


def _video(rate: FrameRate = _SIXTY, size: tuple[int, int] = (1280, 720)) -> MediaItem:
    return MediaItem(
        path=Path("C:/素材/合成の動画.mp4"),
        duration=Fraction(5),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=size[0],
                height=size[1],
                frame_rate=rate,
                time_base=Fraction(1, 90000),
                codec="h264",
            ),
        ),
    )


@pytest.fixture
def window(qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    # 置いた素材は本物のファイルではない 解析と控えを頼まない
    monkeypatch.setattr(created, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(created._analyzer, "request", lambda _media, **_kwargs: None)
    yield created
    created.close()


def _answer(monkeypatch: pytest.MonkeyPatch, answer: bool) -> list[VideoFormat]:
    asked: list[VideoFormat] = []

    def ask(_parent: object, _current: ProjectSettings, wanted: VideoFormat) -> bool:
        asked.append(wanted)
        return answer

    monkeypatch.setattr(media_match, "ask_to_match", ask)
    return asked


def _pool(window: MainWindow, media: MediaItem) -> None:
    window.execute(AddMedia(media))


def _placed_frames(window: MainWindow) -> int:
    return window._document.project.timeline.duration


class TestMatchingTheFirstVideo:
    def test_yes_takes_the_rate_and_size_of_the_video(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 「はい」でレートが変わらないと、60fps の動画が 30fps で書き出されて半分のコマが捨てられる
        asked = _answer(monkeypatch, True)
        media = _video()
        _pool(window, media)
        window._on_media_dropped([str(media.id)], 0, "")
        settings = window._document.project.settings
        assert len(asked) == 1
        assert (settings.width, settings.height, settings.frame_rate) == (1280, 720, FrameRate(60))
        # 置く長さは合わせた後のレートで数える 5 秒は 60fps で 300 フレーム
        assert _placed_frames(window) == 300

    def test_the_drop_spot_keeps_its_time(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 落とした位置は合わせる前のレートで数えてある 数のまま使うと、30fps の 1 秒
        # （フレーム 30）へ落とした 60fps の動画が 0.5 秒の所に置かれた（PR #155 の指摘）
        _answer(monkeypatch, True)
        media = _video()
        _pool(window, media)
        window._on_media_dropped([str(media.id)], 30, "")
        clip = next(iter(window._document.project.timeline.video_tracks())).clips[0]
        assert clip.timeline_start == 60

    def test_the_playhead_keeps_its_time(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 数のまま残すと、30fps の 1 秒にあった再生ヘッドが 60fps では 0.5 秒を指し、
        # 表示の時刻も再生を始める位置も変わる（PR #155 の指摘） 取り消しでも同じ
        _answer(monkeypatch, True)
        media = _video()
        _pool(window, media)
        window._timeline.set_playhead(30)
        window._on_media_dropped([str(media.id)], 30, "")
        assert window._timeline.playhead == 60
        assert window._playback.frame == 60
        window.undo()
        window.undo()
        assert window._document.project.rate == FrameRate(30)
        assert window._timeline.playhead == 30

    def test_a_dropped_file_keeps_its_time(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, media_dir: Path
    ) -> None:
        # ファイルは裏で調べてから置くので、落とした時点のレートを覚えておく
        _answer(monkeypatch, True)
        sample = make_sample(media_dir, "match_60_drop.mp4", fps="60", duration=1.0)
        window._on_files_dropped([sample.path], 30, "")
        assert window.wait_for_imports()
        clip = next(iter(window._document.project.timeline.video_tracks())).clips[0]
        assert window._document.project.rate == FrameRate(60)
        assert clip.timeline_start == 60

    def test_matching_is_its_own_undo_step(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 置き方だけを戻したいときに形まで戻ると、次に置いたときにまた尋ねられる
        _answer(monkeypatch, True)
        media = _video()
        _pool(window, media)
        window._on_media_dropped([str(media.id)], 0, "")
        window.undo()
        assert window._document.project.rate == FrameRate(60)
        assert _placed_frames(window) == 0
        window.undo()
        assert window._document.project.rate == FrameRate(30)

    def test_no_keeps_the_project(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 「いいえ」でも形が変わると、決めておいた解像度とレートが黙って書き換わる
        _answer(monkeypatch, False)
        media = _video()
        _pool(window, media)
        window._on_media_dropped([str(media.id)], 0, "")
        assert window._document.project.settings.resolution == (1920, 1080)
        assert window._document.project.rate == FrameRate(30)
        assert _placed_frames(window) == 150

    def test_the_second_video_is_not_asked_about(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 置いた後に変えると、それまでのクリップの長さが変わる
        asked = _answer(monkeypatch, False)
        first, second = _video(), _video(FrameRate(24))
        _pool(window, first)
        _pool(window, second)
        window._on_media_dropped([str(first.id)], 0, "")
        window._on_media_dropped([str(second.id)], 0, "")
        assert len(asked) == 1

    def test_never_does_not_ask(self, window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
        # 「合わせない」を選んだ人に尋ねると、毎回同じ窓を閉じる手間が残る
        asked = _answer(monkeypatch, True)
        window._preferences = replace(window._preferences, match_video=media_match.MATCH_NEVER)
        media = _video()
        _pool(window, media)
        window._on_media_dropped([str(media.id)], 0, "")
        assert asked == []
        assert window._document.project.rate == FrameRate(30)

    def test_always_matches_without_asking(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 「常に合わせる」で尋ねると窓が出て、29.97 を丸めると音と絵が少しずつずれる
        asked = _answer(monkeypatch, False)
        window._preferences = replace(window._preferences, match_video=media_match.MATCH_ALWAYS)
        media = _video(FrameRate(30000, 1001))
        _pool(window, media)
        window._insert_media_by_id(str(media.id))
        assert asked == []
        # 29.97 は分数のまま
        assert window._document.project.rate.fps == Fraction(30000, 1001)

    def test_a_matching_video_is_not_asked_about(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 同じ形なのに尋ねると、毎回「はい」を押すだけの窓になる
        asked = _answer(monkeypatch, True)
        media = _video(FrameRate(30), (1920, 1080))
        _pool(window, media)
        window._on_media_dropped([str(media.id)], 0, "")
        assert asked == []

    def test_importing_a_file_asks_too(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch, media_dir: Path
    ) -> None:
        # ドラッグ＆ドロップも読み込みボタンも、同じ読み込みの流れを通る
        asked = _answer(monkeypatch, True)
        sample = make_sample(media_dir, "match_60.mp4", fps="60", duration=1.0)
        window.import_media([sample.path])
        assert window.wait_for_imports()
        assert len(asked) == 1
        settings = window._document.project.settings
        assert (settings.width, settings.height, settings.frame_rate) == (320, 240, FrameRate(60))
        assert _placed_frames(window) == 60


class TestThePreference:
    def test_asking_is_the_default(self) -> None:
        # 黙って変えても、黙って変えなくても、知らない人は気付けない
        assert Preferences().match_video == media_match.MATCH_ASK

    def test_it_comes_back(self, tmp_path: Path) -> None:
        # 選んだ合わせ方が残らないと、起動のたびに尋ねる窓へ戻される
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(match_video=media_match.MATCH_NEVER))
        assert store.load().match_video == media_match.MATCH_NEVER

    def test_an_unknown_word_falls_back(self, tmp_path: Path) -> None:
        # 知らない値で尋ねずに進むと、古い版で新しい設定を開いた人のプロジェクトが
        # 黙って動画に合わせて変わるか、合わないまま置かれる
        path = tmp_path / "preferences.json"
        path.write_text('{"match_video": "sometimes"}', encoding="utf-8")
        assert PreferenceStore(path).load().match_video == media_match.MATCH_ASK

    def test_the_dialog_shows_and_returns_it(self, qt_application: QApplication) -> None:
        # 設定の画面が今の値を返さないと、OK を押しただけで「常に合わせる」が外れる
        del qt_application
        dialog = PreferencesDialog(Preferences(match_video=media_match.MATCH_ALWAYS))
        try:
            assert dialog.preferences().match_video == media_match.MATCH_ALWAYS
        finally:
            dialog.deleteLater()
