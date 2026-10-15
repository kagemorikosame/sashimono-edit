"""AI の道具 check_output（完成物の検査 #337）を、本物の窓と本物の検査で確かめる

問題の位置が分かっている素材を PyAV で焼いて作品を組み、AI と同じ道（ブリッジ）から道具を
呼ぶ 黒いコマ・止まった絵・無音・音割れ・はみ出す字幕をわざと入れた作品でそれぞれを拾い、
問題の無い作品では何も拾わないことを見る 絵を描く試験は GPU が要る（無ければ飛ばす）
音だけの試験は GPU を使わないので、CI でも走る
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip, insert_generated
from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.checks import job as check_job
from sashimono.engine.decode.probe import probe_media
from sashimono.engine.gpu import GLContextError
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.check_media import HEIGHT, WIDTH, tone, write_tone, write_video


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓や問いかけが開いたら、その場で落とす

    検査は確かめを挟まない読むだけの道具 窓が出たら作りが変わったということで、
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


@dataclass(frozen=True)
class _Media:
    video: MediaItem
    tone: MediaItem
    clipped: MediaItem


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> _Media:
    folder = tmp_path_factory.mktemp("check")
    # 6〜9 秒だけ同じ絵が続く動画（音は無し）
    video = write_video(folder / "絵.mp4", 12.0, frozen=(6.0, 9.0))
    # -14 dBFS の 1 kHz（両方のチャンネル）は -14 LUFS 配信の基準ちょうど
    level = write_tone(folder / "音.wav", tone(12.0, -14.0))
    # 頭打ちが続く 1 秒の音
    clipped = write_tone(folder / "割れ.wav", tone(1.0, -14.0, clipped=(0.0, 1.0)))
    return _Media(probe_media(video), probe_media(level), probe_media(clipped))


def _clip(media: MediaItem, start: int, duration: int, source_in: float = 0.0) -> Clip:
    return Clip(
        timeline_start=start,
        duration=duration,
        media_id=media.id,
        source_in=Fraction(source_in).limit_denominator(1000),
    )


def _project(
    media: _Media,
    *,
    pictures: tuple[Clip, ...],
    sounds: tuple[Clip, ...],
    text: str = "こんにちは",
    text_x: float = 0.0,
) -> Project:
    """絵 1 本・音 1 本・字幕 1 本の作品（320x180 30fps） 字幕は画面の下寄り"""
    settings = ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30))
    base = Project.create(settings, media=(media.video, media.tone, media.clipped))
    picture = Track(kind=TrackKind.VIDEO, name="V1")
    sound = Track(kind=TrackKind.AUDIO, name="A1")
    project = base.with_timeline(replace(base.timeline, tracks=(picture, sound)))
    for clip in pictures:
        project = AddClip(picture.id, clip).apply(project)
    for clip in sounds:
        project = AddClip(sound.id, clip).apply(project)
    end = max(clip.timeline_end for clip in (*pictures, *sounds))
    for command in insert_generated(
        project,
        TEXT.create(text=text, size=20.0, pos_x=text_x, pos_y=-60.0),
        at_frame=0,
        duration=end,
    ):
        project = command.apply(project)
    return project


def _clean(media: _Media) -> Project:
    """問題の無い作品 止まった所（6〜9 秒）は使わず、動く所だけをつなぐ"""
    return _project(
        media,
        pictures=(_clip(media.video, 0, 180, 0.0), _clip(media.video, 180, 90, 9.0)),
        sounds=(_clip(media.tone, 0, 270),),
    )


@pytest.fixture
def opened(qt_application: QApplication) -> Iterator[list[MainWindow]]:
    del qt_application
    windows: list[MainWindow] = []
    yield windows
    for window in windows:
        window.close()
        window.deleteLater()


def _window(opened: list[MainWindow], project: Project) -> MainWindow:
    window = MainWindow(project, confirm_unsaved=False)
    opened.append(window)
    return window


def _call(window: MainWindow, arguments: dict[str, Any], *, timeout: float = 120.0) -> Any:
    """AI のスレッドからブリッジ越しに道具を呼ぶ 画面のスレッドは仕事を拾い続ける

    本物の AI と同じ道を通す 道具は画面のスレッドで動き、待つのは AI のスレッド
    """
    operation = find_operation("check_output")
    assert operation is not None
    bridge = EditorBridge(window, auto_approve=False)
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = bridge.invoke(operation, arguments)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
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


def _kinds(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for row in result["problems"]:
        found.setdefault(row["kind"], []).append(row)
    return found


@pytest.mark.usefixtures("gpu")
class TestTimeline:
    def test_a_clean_work_has_no_problems(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、仕上がった作品にまで問題を並べ、AI が直さなくてよい所を直しに行く
        window = _window(opened, _clean(media))
        result = _call(window, {"wait_seconds": 120})
        assert result["complete"] is True
        assert result["problems"] == [], result["problems"]
        assert result["summary"]["integrated_lufs"] == pytest.approx(-14.0, abs=0.5)
        assert result["length_seconds"] == pytest.approx(9.0)
        assert "問題は見つかりませんでした" in result["note"]

    def test_a_gap_between_clips_is_a_black_frame(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、クリップの間に空いた 10 コマの黒（つなぎ目のちらつき）を見落とす
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 180, 0.0), _clip(media.video, 190, 90, 9.0)),
            sounds=(_clip(media.tone, 0, 280),),
        )
        result = _call(_window(opened, project), {"wait_seconds": 120})
        (black,) = _kinds(result)["black"]
        assert (black["start_frame"], black["end_frame"]) == (180, 190)
        assert black["severity"] == "high"
        assert black["timecode"] == "00:00:06:00"

    def test_a_frozen_video_is_found(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、素材の止まった所をそのまま使った作品を見落とす
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 360, 0.0),),
            sounds=(_clip(media.tone, 0, 360),),
        )
        result = _call(_window(opened, project), {"wait_seconds": 120})
        (freeze,) = _kinds(result)["freeze"]
        assert freeze["start_frame"] == pytest.approx(180, abs=2)
        assert freeze["end_frame"] == pytest.approx(270, abs=2)
        assert freeze["severity"] == "medium"
        assert freeze["clip_ids"]

    def test_overflowing_subtitles_are_found(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、画面の端で切れた字幕（#256 の長い文）がそのまま書き出される
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 180, 0.0),),
            sounds=(_clip(media.tone, 0, 180),),
            text="とても長い字幕が画面の端からはみ出して切れてしまう例",
        )
        result = _call(_window(opened, project), {"checks": ["text_overflow"]})
        (overflow,) = _kinds(result)["text_overflow"]
        assert overflow["severity"] == "high"
        assert "左" in overflow["sides"] and "右" in overflow["sides"]
        assert overflow["beyond_screen"] is True
        text_clip = next(
            clip
            for track in project.timeline.tracks
            for clip in track.clips
            if clip.source is not None and clip.source.kind == "text"
        )
        assert overflow["clip_ids"] == [str(text_clip.id)]

    def test_a_subtitle_pushed_off_to_one_side_is_found(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、位置をずらしすぎて片側が切れた字幕を見落とす
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 180, 0.0),),
            sounds=(_clip(media.tone, 0, 180),),
            text_x=150.0,
        )
        result = _call(_window(opened, project), {"checks": ["text_overflow"]})
        (overflow,) = _kinds(result)["text_overflow"]
        assert overflow["sides"] == "右"
        assert overflow["overflow_px"] > 0

    def test_a_wide_safe_margin_flags_subtitles_near_the_edge(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、安全域の指定（テレビ向け）が効かず、端に寄った字幕を知らせない
        result = _call(
            _window(opened, _clean(media)),
            {"checks": ["text_overflow"], "safe_margin_percent": 20},
        )
        (overflow,) = _kinds(result)["text_overflow"]
        assert overflow["severity"] == "medium"
        assert overflow["beyond_screen"] is False

    def test_the_text_check_alone_does_not_render_every_frame(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、字幕だけを頼んでも全部のコマを描き、長い作品で待たされる
        window = _window(opened, _clean(media))
        result = _call(window, {"checks": ["text_overflow"]})
        assert result["complete"] is True
        assert window._checks.state().progress == 1.0
        assert result["problems"] == []


class TestSound:
    """音だけを調べる 絵を描かないので GPU が無くても走る"""

    def test_a_gap_in_the_sound_is_silence(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、音を置き忘れた 3 秒の無音を見落とす
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 270, 0.0),),
            sounds=(_clip(media.tone, 0, 90), _clip(media.tone, 180, 90, 6.0)),
        )
        result = _call(_window(opened, project), {"checks": ["silence", "clipping", "loudness"]})
        (silence,) = _kinds(result)["silence"]
        assert silence["start_frame"] == pytest.approx(90, abs=2)
        assert silence["end_frame"] == pytest.approx(180, abs=2)
        assert silence["severity"] == "medium"
        assert "置いていない" in silence["detail"]

    def test_a_clipped_sound_is_found(self, opened: list[MainWindow], media: _Media) -> None:
        # 壊れると、頭打ちの続く音（音割れ）を混ぜた作品をそのまま渡す
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 180, 0.0),),
            sounds=(
                _clip(media.tone, 0, 60),
                _clip(media.clipped, 60, 30),
                _clip(media.tone, 90, 90, 3.0),
            ),
        )
        result = _call(_window(opened, project), {"checks": ["clipping"]})
        (clipping,) = _kinds(result)["clipping"]
        assert clipping["start_frame"] == pytest.approx(60, abs=1)
        assert clipping["end_frame"] == pytest.approx(90, abs=1)
        assert clipping["severity"] == "high"

    def test_the_loudness_target_follows_the_preference(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、放送向けに -23 を選んだ人の作品も配信の -14 で比べ、外れていると言わない
        PreferenceStore().save(Preferences(check_loudness_target=-23))
        window = _window(opened, _clean(media))
        result = _call(window, {"checks": ["loudness"]})
        (loudness,) = _kinds(result)["loudness"]
        assert loudness["target_lufs"] == -23
        assert loudness["difference_lu"] == pytest.approx(9.0, abs=0.5)
        assert loudness["severity"] == "high"
        # AI が引数で渡した基準は設定より優先する
        result = _call(window, {"checks": ["loudness"], "loudness_target": -14})
        assert result["problems"] == []

    def test_the_stop_button_stops_a_running_check(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、ステータスバーの〔止める〕を押しても長い検査が走り続ける
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 360, 0.0),),
            sounds=tuple(_clip(media.tone, start, 360) for start in range(0, 36000, 360)),
        )
        window = _window(opened, project)
        first = _call(window, {"checks": ["silence"], "wait_seconds": 0})
        assert first["complete"] is False
        stop = window._check_indicator.cancel_button
        assert stop is not None
        stop.click()
        assert window._checks.waiter()(30.0)
        assert window._checks.state().stopped
        result = _call(window, {"checks": ["silence"], "wait_seconds": 0})
        assert "検査の途中です" in result["note"]

    def test_partial_results_come_back_when_waiting_runs_out(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、長い作品で AI が何分も返事を待たされるか、途中の結果を全部と読み違える
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 360, 0.0),),
            sounds=tuple(_clip(media.tone, start, 360) for start in range(0, 36000, 360)),
        )
        window = _window(opened, project)
        started = time.monotonic()
        result = _call(window, {"checks": ["silence"], "wait_seconds": 0.5})
        assert time.monotonic() - started < 10.0
        assert result["complete"] is False
        assert "同じ引数で呼ぶと続きを返します" in result["note"]
        window._checks.stop()
        assert window._checks.waiter()(30.0)

    def test_closing_the_window_stops_the_check(
        self, qt_application: QApplication, media: _Media
    ) -> None:
        # 壊れると、窓を閉じても検査の走り係が残り、閉じた窓の作品を読み続ける
        del qt_application
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 360, 0.0),),
            sounds=tuple(_clip(media.tone, start, 360) for start in range(0, 36000, 360)),
        )
        window = MainWindow(project, confirm_unsaved=False)
        try:
            _call(window, {"checks": ["silence"], "wait_seconds": 0})
            assert window._checks.running
            window.close()
            assert not window._checks.running
        finally:
            window.deleteLater()

    def test_editing_after_a_check_checks_again(
        self, opened: list[MainWindow], media: _Media
    ) -> None:
        # 壊れると、AI が直した後に呼んでも前の作品の結果が返り、直ったか確かめられない
        project = _project(
            media,
            pictures=(_clip(media.video, 0, 270, 0.0),),
            sounds=(_clip(media.tone, 0, 90), _clip(media.tone, 180, 90, 6.0)),
        )
        window = _window(opened, project)
        result = _call(window, {"checks": ["silence"]})
        assert _kinds(result)["silence"]
        sound = next(t for t in window.project.timeline.tracks if t.kind is TrackKind.AUDIO)
        window.apply_commands([AddClip(sound.id, _clip(media.tone, 90, 90, 3.0))], "音を足す")
        result = _call(window, {"checks": ["silence"]})
        assert result["problems"] == []

    def test_a_file_outside_the_allowed_folders_is_refused(
        self, opened: list[MainWindow], media: _Media, tmp_path: Path
    ) -> None:
        # 壊れると、作品と関係の無い場所の動画まで AI が読みに行ける
        outside = write_video(tmp_path / "よそ.mp4", 1.0)
        window = _window(opened, _clean(media))
        with pytest.raises(ToolError, match="調べられる場所"):
            _call(window, {"path": str(outside)})


class _NoGpuContext:
    """GPU の無い機械のコンテキスト 作るのはできるが、走り係で作り終える所で断られる"""

    def __init__(self, *_arguments: object, **_options: object) -> None:
        pass

    def complete(self) -> None:
        raise GLContextError("試験: OpenGL 4.3 が使えない")

    def move_to_thread(self, _thread: object) -> None:
        pass

    def release(self) -> None:
        pass


def _refuse_gpu(*_arguments: object, **_options: object) -> None:
    raise GLContextError("試験: サーフェスを作れない")


class TestWithoutGpu:
    """GPU が使えず絵を調べられなかった種類を、調べて問題が無かったと言わない"""

    @pytest.mark.parametrize("missing", [_refuse_gpu, _NoGpuContext])
    @pytest.mark.parametrize("checks", [["black"], ["text_overflow"], ["black", "freeze"]])
    def test_picture_checks_are_reported_as_skipped(
        self,
        opened: list[MainWindow],
        media: _Media,
        monkeypatch: pytest.MonkeyPatch,
        missing: object,
        checks: list[str],
    ) -> None:
        # 壊れると、GPU の無い機械で黒いコマを頼んだ AI が「黒いコマは無い」と伝える
        monkeypatch.setattr(check_job, "OffscreenGLContext", missing)
        result = _call(_window(opened, _clean(media)), {"checks": checks})
        assert result["checks"] == []
        assert result["skipped_checks"] == checks
        assert "問題は見つかりませんでした" not in result["note"]
        assert "調べていません" in result["note"]

    def test_sound_is_still_checked_and_only_pictures_are_skipped(
        self, opened: list[MainWindow], media: _Media, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、GPU が無いだけで音の検査まで外れるか、絵の検査を調べた物として並べる
        monkeypatch.setattr(check_job, "OffscreenGLContext", _NoGpuContext)
        result = _call(_window(opened, _clean(media)), {"checks": ["black", "silence"]})
        assert result["checks"] == ["silence"]
        assert result["skipped_checks"] == ["black"]
        assert "問題は見つかりませんでした" not in result["note"]


class TestPreference:
    def test_the_default_target_is_for_streaming(self) -> None:
        # 壊れると、何も選んでいない人の作品を放送の基準で比べ、配信用の作品を大きすぎると言う
        assert Preferences().check_loudness_target == -14

    def test_a_broken_value_falls_back_to_the_default(self) -> None:
        # 壊れると、手で書き換えた値のまま設定画面が開き、OK で別の基準に化ける
        path = PreferenceStore().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"check_loudness_target": -99}', encoding="utf-8")
        assert PreferenceStore().load().check_loudness_target == -14
        path.write_text('{"check_loudness_target": "大きめ"}', encoding="utf-8")
        assert PreferenceStore().load().check_loudness_target == -14

    def test_the_dialog_offers_and_returns_the_choice(self, qt_application: QApplication) -> None:
        # 壊れると、設定画面で放送の基準を選んでも保存されない
        del qt_application
        dialog = PreferencesDialog(Preferences(check_loudness_target=-24))
        try:
            assert dialog.preferences().check_loudness_target == -24
            choice = dialog._check_loudness
            choice.setCurrentIndex(choice.findData(-23))
            assert dialog.preferences().check_loudness_target == -23
        finally:
            dialog.deleteLater()
