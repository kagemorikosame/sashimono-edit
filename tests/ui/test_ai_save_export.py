"""AI の保存と書き出し（B1 #327）を、本物の窓と本物の書き出しで確かめる

窓を組み、AI パネルの「変更を自動で承認」を入れ、AI のスレッドのつもりで道具を呼ぶ
確かめは AI パネルの欄に出たものを読み、〔許す〕のボタンを押して答える（人の代わり）

- 取り消せない操作の確かめは、自動で承認でも出て、取り消せないと書いてある
- 許した場所（作品のフォルダ）に書け、許していない場所は確かめの前に断る
- 書き出しは画面を固めずに裏で進み、小さい作品で実際に動画ができる 止められる
- 書き出し先へ置くときに場所を差し替えられても、確かめていない所や元の素材へは書かない
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import av
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.ai import operations, safety
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import EXPORT_RUNNING, ToolError
from sashimono.ai.operations import Operation, find_operation
from sashimono.core.commands import AddClip, AddTrack, SetTrackState
from sashimono.core.io import subtitles as subtitles_module
from sashimono.core.io.atomic import copy_atomic
from sashimono.core.io.serialize import (
    FORMAT_VERSION,
    PRE_UPGRADE_SUFFIX,
    load_project,
    save_project,
)
from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.core.projection import project_timeline
from sashimono.core.timebase import FrameRate
from sashimono.engine.encode import ExportSettings, available_video_codecs, export_project
from sashimono.ui import ai_export, main_window
from sashimono.ui.ai_export import scratch_folder
from sashimono.ui.backup_settings import state_root_for
from sashimono.ui.chat.panel import IRREVERSIBLE_ALLOW_TEXT, IRREVERSIBLE_HEAD
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import allow_sending, make_loaded
from tests.ai.test_bridge import call_in_thread

#: AI がファイルを書けるのは、書く先のフォルダを書き終えるまで持てる Windows だけ
#: （:func:`~sashimono.ai.safety.pin_holds_the_folder` ほかの OS では確かめの箱を出す前に断る）
#: 書けること・確かめの箱が出た後の振る舞いを見る試験はこの印を付ける 断る試験には付けない
#: （ほかの OS で断ることは test_save_and_export.py の TestOtherSystemsAreRefused が見る）
WINDOWS_WRITES = pytest.mark.skipif(
    sys.platform != "win32", reason="AI がファイルを書くのは Windows だけ（pin_holds_the_folder）"
)


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす AI の保存と書き出しは、確かめの欄のほかに窓を出さない"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


def _small_project(frames: int = 30, media: tuple[MediaItem, ...] = ()) -> Project:
    """320x240 の作品に、frames コマのクリップを 1 本置く 書き出しを速く終えるため"""
    track = Track(TrackKind.VIDEO, "V1")
    project = Project.create(
        ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)), media=media
    )
    project = AddTrack(track).apply(project)
    return AddClip(track.id, Clip(timeline_start=0, duration=frames)).apply(project)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    created = tmp_path / "作品"
    created.mkdir()
    return created


def _open(project: Project, folder: Path) -> MainWindow:
    window = MainWindow(project, path=folder / "作品.sme", confirm_unsaved=False)
    # 人が〔自動で承認〕を入れた（取り消せる編集は確かめずに進む）
    window._chat._auto.setChecked(True)
    # 外へ送る物（#342）は尋ねずに送る 送る前の確かめは tests/ai/test_outbound.py と
    # tests/ui/test_send_consent.py が見る ここで見たいのは保存と書き出しの確かめ
    allow_sending(window._chat._bridge)
    return window


@pytest.fixture
def window(qt_application: QApplication, folder: Path) -> Iterator[MainWindow]:
    del qt_application
    created = _open(_small_project(), folder)
    yield created
    created.close()
    created.deleteLater()


class _Asked:
    """AI パネルの確かめの欄に出た物"""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.always_hidden: list[bool] = []
        self.allow_texts: list[str] = []


def _drive(
    window: MainWindow,
    operation: Operation,
    arguments: dict[str, Any],
    *,
    allow: bool = True,
    timeout: float = 60.0,
    meanwhile: Callable[[], None] | None = None,
) -> tuple[_Asked, dict[str, object]]:
    """AI のスレッドのつもりで道具を呼ぶ 画面のイベントを回し、確かめの欄が出たら答える

    確かめを出すのも答えを受け取るのも AI パネルの本物の仕組み（見張りのタイマー）
    """
    bridge: EditorBridge = window._chat._bridge
    asked = _Asked()
    done, box = call_in_thread(bridge, operation, arguments)
    deadline = time.monotonic() + timeout
    panel = window._chat
    while not done.is_set() and time.monotonic() < deadline:
        QApplication.processEvents()
        if not panel._approval_box.isHidden() and panel._approval is not None:
            asked.texts.append(panel._approval_text.text())
            asked.always_hidden.append(panel._approval_always.isHidden())
            asked.allow_texts.append(panel._approval_allow.text())
            if allow:
                panel._approval_allow.click()
            else:
                # 〔拒否〕のボタンは名前で持っていないので、押したときと同じ所を呼ぶ
                panel._answer(False)
        if meanwhile is not None:
            meanwhile()
        time.sleep(0.005)
    assert done.is_set(), "道具が終わらなかった"
    QApplication.processEvents()
    return asked, box


def _assert_irreversible_request(asked: _Asked) -> None:
    assert len(asked.texts) == 1, "取り消せない操作なのに確かめが出なかった（または 2 度出た）"
    assert asked.texts[0].startswith(IRREVERSIBLE_HEAD)
    assert asked.always_hidden == [True]
    assert asked.allow_texts == [IRREVERSIBLE_ALLOW_TEXT]


def _wait_until(condition: Callable[[], bool], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert condition()


# --- 保存 ---


class TestSave:
    @WINDOWS_WRITES
    def test_auto_approve_still_asks_and_the_project_is_saved(
        self, window: MainWindow, folder: Path
    ) -> None:
        # 壊れると、自動で承認の人の作品を AI が黙って上書きするか、許しても保存されない
        track = window._document.project.timeline.tracks[0]
        window.apply_commands([SetTrackState(track.id, muted=True)], "ミュート")
        assert window.is_modified

        asked, box = _drive(window, _operation("save_project"), {})

        _assert_irreversible_request(asked)
        assert str((folder / "作品.sme").resolve()) in asked.texts[0]
        assert box.get("error") is None
        saved = load_project(folder / "作品.sme")
        assert saved.timeline == window._document.project.timeline
        assert not window.is_modified

    @WINDOWS_WRITES
    def test_denying_writes_nothing(self, window: MainWindow, folder: Path) -> None:
        # 壊れると、拒否しても作品が書かれ、前の中身は取り消しでも戻らない
        asked, box = _drive(window, _operation("save_project"), {}, allow=False)
        assert len(asked.texts) == 1
        assert isinstance(box.get("error"), ToolError)
        assert not (folder / "作品.sme").exists()

    @WINDOWS_WRITES
    def test_save_as_in_the_project_folder_switches_the_project(
        self, window: MainWindow, folder: Path
    ) -> None:
        # 壊れると、AI が名前を付けて保存しても、題や次の保存先が前の場所のまま
        target = folder / "第二稿.sme"
        asked, box = _drive(window, _operation("save_project_as"), {"path": str(target)})
        _assert_irreversible_request(asked)
        assert box.get("error") is None
        assert window.project_path == target.resolve()
        assert load_project(target).timeline == window._document.project.timeline
        assert "第二稿" in window.windowTitle()
        # 書いた場所は伏せ字の表へ入る（#342） 作品を開き直した後も、名前と場所を断った人の
        # 所で、知らせに残ったこの場所を伏せる
        assert str(target.resolve()) in window.remembered_names()

    def test_outside_is_refused_without_asking(
        self, window: MainWindow, folder: Path, tmp_path: Path
    ) -> None:
        # 壊れると、AI が人の選んでいない場所（デスクトップなど）へ作品を書く
        outside = tmp_path / "外"
        outside.mkdir()
        asked, box = _drive(window, _operation("save_project_as"), {"path": str(outside / "a.sme")})
        assert asked.texts == []
        assert isinstance(box.get("error"), ToolError)
        assert not (outside / "a.sme").exists()
        assert window.project_path == folder / "作品.sme"


# --- 字幕の書き出しと静止画 ---


class TestSubtitlesAndSnapshot:
    @WINDOWS_WRITES
    def test_subtitles_are_the_same_file_the_panel_writes(
        self,
        qt_application: QApplication,
        folder: Path,
        video_media: MediaItem,
        transcript: Transcript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、AI が書いた字幕が画面の〔書き出し…〕と違う形式になるか、自動で承認の
        # 人の字幕のファイルを確かめずに上書きする
        del qt_application
        window = _open(make_loaded(video_media, transcript), folder)
        try:
            asked, box = _drive(
                window, _operation("export_subtitles"), {"path": str(folder / "AI.srt")}
            )
            _assert_irreversible_request(asked)
            assert "字幕を書き出します" in asked.texts[0]
            assert box.get("error") is None, box.get("error")

            # 人が字幕パネルの〔書き出し…〕で同じ所へ書いた物と比べる
            by_hand = folder / "人.srt"
            monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(by_hand), ""))
            window._subtitles.export_file()
            ai_text = (folder / "AI.srt").read_text(encoding="utf-8")
            assert ai_text == by_hand.read_text(encoding="utf-8")
            assert "今日は" in ai_text
        finally:
            window.close()
            window.deleteLater()

    def test_subtitles_outside_are_refused_without_asking(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        # 壊れると、AI が人の選んでいない場所へ字幕を書く
        outside = tmp_path / "外"
        outside.mkdir()
        asked, box = _drive(
            window, _operation("export_subtitles"), {"path": str(outside / "a.srt")}
        )
        assert asked.texts == []
        assert isinstance(box.get("error"), ToolError)
        assert list(outside.iterdir()) == []

    @WINDOWS_WRITES
    @pytest.mark.usefixtures("gpu")
    def test_a_snapshot_is_drawn_like_the_screen(self, window: MainWindow, folder: Path) -> None:
        # 壊れると、AI の静止画が画面の〔静止画を保存…〕と違う描き方になるか、確かめずに書く
        asked, box = _drive(window, _operation("save_snapshot"), {"path": "AI.png", "frame": 10})
        _assert_irreversible_request(asked)
        assert "静止画を保存します" in asked.texts[0]
        assert box.get("result") == {"saved": str((folder / "AI.png").resolve()), "frame": 10}
        window.seek(10)
        by_hand = window.save_snapshot(folder / "人.png")
        assert by_hand is not None
        ai_image = QImage(str(folder / "AI.png"))
        assert not ai_image.isNull()
        assert (ai_image.width(), ai_image.height()) == (320, 240)
        assert ai_image == QImage(str(by_hand))

    def test_a_snapshot_outside_is_refused_without_asking(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        # 壊れると、AI が人の選んでいない場所へ絵を書く
        outside = tmp_path / "外"
        outside.mkdir()
        asked, box = _drive(window, _operation("save_snapshot"), {"path": str(outside / "a.png")})
        assert asked.texts == []
        assert isinstance(box.get("error"), ToolError)
        assert list(outside.iterdir()) == []


# --- 書き出し ---


@pytest.fixture
def libx264() -> str:
    if "libx264" not in available_video_codecs():
        pytest.skip("libx264 が使えない")
    return "libx264"


def _frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _scratch_files() -> list[Path]:
    folder = scratch_folder()
    return list(folder.iterdir()) if folder.exists() else []


@pytest.mark.usefixtures("gpu")
class TestExport:
    @WINDOWS_WRITES
    def test_a_small_project_becomes_a_video_without_holding_the_editor(
        self, window: MainWindow, folder: Path, libx264: str
    ) -> None:
        # 壊れると、AI が書き出しを頼めないか、確かめずに書き出すか、書き出しの間に画面が固まる
        seen_running: list[str] = []

        def look() -> None:
            # 書き出しの間も画面のイベントが回り、右端に進み具合が出ている
            indicator = window._assistant_export_indicator
            if not indicator.isHidden():
                seen_running.append(indicator.text)

        asked, box = _drive(
            window,
            _operation("export_video"),
            {"path": "完成.mp4", "codec": libx264, "bitrate_mbps": 2, "wait_seconds": 120},
            meanwhile=look,
        )
        _assert_irreversible_request(asked)
        assert "動画を書き出します" in asked.texts[0]
        assert box.get("error") is None, box.get("error")
        result = box.get("result")
        assert isinstance(result, dict)
        assert result["state"] == "done", result
        output = folder / "完成.mp4"
        assert _frames(output) == 30
        assert seen_running and seen_running[0].startswith("AI の")
        # 仮のファイルは残らない 書き出し先のフォルダにも書きかけを残さない
        assert _scratch_files() == []
        assert sorted(path.name for path in folder.iterdir()) == ["完成.mp4"]
        assert window._assistant_export_indicator.isHidden()
        # AI が書き出しても、人が窓で選んだ場所としては覚えない（書ける場所が勝手に増えない）
        assert window.export_folder is None

    def test_outside_is_refused_without_asking(
        self, window: MainWindow, tmp_path: Path, libx264: str
    ) -> None:
        # 壊れると、AI が人の選んでいない場所へ動画を書く
        outside = tmp_path / "外"
        outside.mkdir()
        asked, box = _drive(
            window, _operation("export_video"), {"path": str(outside / "a.mp4"), "codec": libx264}
        )
        assert asked.texts == []
        assert isinstance(box.get("error"), ToolError)
        assert list(outside.iterdir()) == []

    @WINDOWS_WRITES
    def test_the_export_can_be_stopped_and_keeps_the_old_file(
        self,
        qt_application: QApplication,
        folder: Path,
        libx264: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、AI が要らない書き出しを止められないか、止めた後に書き出し先が壊れて残る
        del qt_application
        _slow_export(monkeypatch)
        window = _open(_small_project(frames=300), folder)
        try:
            target = folder / "完成.mp4"
            target.write_bytes(b"previous")
            _asked, box = _drive(
                window, _operation("export_video"), {"path": str(target), "codec": libx264}
            )
            assert box.get("error") is None
            _wait_until(lambda: window.export_status()["progress"] not in (0, 0.0))
            asked, box = _drive(window, _operation("cancel_export"), {})
            # 止めるのは取り消せる側 自動で承認なら確かめない
            assert asked.texts == []
            assert box.get("error") is None
            _wait_until(lambda: window.export_status()["state"] not in EXPORT_RUNNING)
            assert window.export_status()["state"] == "cancelled"
            assert target.read_bytes() == b"previous"
            assert _scratch_files() == []
        finally:
            window.close()
            window.deleteLater()

    @WINDOWS_WRITES
    def test_the_stop_button_in_the_status_bar_stops_it(
        self,
        qt_application: QApplication,
        folder: Path,
        libx264: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、AI が始めた書き出しを人が止められない（書き出しの窓が無いので止め方が無い）
        del qt_application
        _slow_export(monkeypatch)
        window = _open(_small_project(frames=300), folder)
        try:
            _drive(window, _operation("export_video"), {"path": "a.mp4", "codec": libx264})
            indicator = window._assistant_export_indicator
            _wait_until(lambda: not indicator.isHidden())
            button = indicator.cancel_button
            assert button is not None
            button.click()
            _wait_until(lambda: window.export_status()["state"] == "cancelled")
            assert not (folder / "a.mp4").exists()
        finally:
            window.close()
            window.deleteLater()

    @WINDOWS_WRITES
    def test_closing_the_window_stops_the_export(
        self,
        qt_application: QApplication,
        folder: Path,
        libx264: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、書き出しのスレッドを持ったまま窓が壊れて落ちるか、書きかけが残る
        del qt_application
        _slow_export(monkeypatch)
        window = _open(_small_project(frames=300), folder)
        _drive(window, _operation("export_video"), {"path": "a.mp4", "codec": libx264})
        _wait_until(lambda: window.export_status()["state"] == "rendering")
        window.close()
        window.deleteLater()
        assert not (folder / "a.mp4").exists()
        assert _scratch_files() == []


@WINDOWS_WRITES
class TestTheTargetChangingAfterApproval:
    """許した後、書く前に書き出し先の名前が現れたり中身が変わったりしたら、置かずに断る

    書き出しは描くのに数分かかり、その間に別のアプリが同じ名前のファイルを作れる
    保存・静止画・字幕も、書く途中（描く・控えを取る・字幕を並べる）の間に同じことが起こりうる
    許したときに無かった名前は置き換えず、あった名前は許したときの中身のままかを照らす
    （Codex の P1）
    """

    def _fake_export(
        self, monkeypatch: pytest.MonkeyPatch, meanwhile: Callable[[], object]
    ) -> None:
        def fake(project: Project, settings: ExportSettings, **_options: Any) -> Path:
            settings.path.write_bytes(b"new video")
            # 描いている数分の間に、別のアプリが書き出し先へ手を出した
            meanwhile()
            return settings.path

        monkeypatch.setattr(ai_export, "run_export", fake)

    def test_a_name_that_appears_while_rendering_is_kept(
        self, window: MainWindow, folder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、許したときは無かった名前のファイル（別のアプリが作った物）を、確かめずに
        # 書き出しで置き換えて消す
        target = folder / "完成.mp4"
        self._fake_export(monkeypatch, lambda: target.write_bytes(b"other app"))
        asked, box = _drive(
            window, _operation("export_video"), {"path": str(target), "wait_seconds": 60}
        )
        assert "置き換えます" not in asked.texts[0]
        result = box.get("result")
        assert isinstance(result, dict)
        assert result["state"] == "failed", result
        assert "前のファイルを残しました" in str(result["message"])
        assert target.read_bytes() == b"other app"
        assert _scratch_files() == []

    def test_a_file_changed_while_rendering_is_kept(
        self, window: MainWindow, folder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、人が置き換えを許したときの中身ではなく、その後に書き換わった中身を消す
        target = folder / "完成.mp4"
        target.write_bytes(b"previous")
        self._fake_export(monkeypatch, lambda: target.write_bytes(b"changed meanwhile!"))
        asked, box = _drive(
            window, _operation("export_video"), {"path": str(target), "wait_seconds": 60}
        )
        assert "置き換えます" in asked.texts[0]
        result = box.get("result")
        assert isinstance(result, dict)
        assert result["state"] == "failed", result
        assert target.read_bytes() == b"changed meanwhile!"

    def test_an_unchanged_file_is_replaced(
        self, window: MainWindow, folder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると（照らし方が厳しすぎると）、許した置き換えまで毎回断られる
        target = folder / "完成.mp4"
        target.write_bytes(b"previous")
        self._fake_export(monkeypatch, lambda: None)
        _asked, box = _drive(
            window, _operation("export_video"), {"path": str(target), "wait_seconds": 60}
        )
        result = box.get("result")
        assert isinstance(result, dict)
        assert result["state"] == "done", result
        assert target.read_bytes() == b"new video"

    def test_a_refused_save_leaves_the_backups_and_the_pre_upgrade_copy_alone(
        self, window: MainWindow, folder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、最後の照らし合わせの後に外のアプリが古い形式の作品を書き換えたとき、
        # 保存そのものは断られても、許していない中身から控え（.bak と世代のバックアップ）を
        # 作り、前の控えと古い世代を失う（Codex の P2）
        project_file = folder / "作品.sme"
        old_format = json.dumps({"format": "sashimono-project", "version": FORMAT_VERSION - 1})
        project_file.write_text(old_format, encoding="utf-8")
        pre_upgrade = folder / f"作品.sme{PRE_UPGRADE_SUFFIX}"
        pre_upgrade.write_text("前の控え", encoding="utf-8")
        window._back_up(project_file)
        backups = state_root_for(window._preferences) / "backups"

        def snapshot() -> dict[str, bytes]:
            return {str(path): path.read_bytes() for path in backups.rglob("*") if path.is_file()}

        before = snapshot()
        assert before
        real_pin = safety.pinned_place

        @contextlib.contextmanager
        def change_then_pin(host: Any, raw: object) -> Iterator[Path]:
            # 照らし合わせが済み、書く処理へ入る所で、外のアプリが古い形式のまま書き換えた
            outside = json.dumps(
                {
                    "format": "sashimono-project",
                    "version": FORMAT_VERSION - 1,
                    "外": "許していない",
                },
                ensure_ascii=False,
            )
            project_file.write_text(outside, encoding="utf-8")
            with real_pin(host, raw) as path:
                yield path

        monkeypatch.setattr(operations, "pinned_place", change_then_pin)
        _asked, box = _drive(window, _operation("save_project"), {})
        assert isinstance(box.get("error"), ToolError)
        assert "許していない" in project_file.read_text(encoding="utf-8")
        assert pre_upgrade.read_text(encoding="utf-8") == "前の控え"
        assert snapshot() == before

    def test_a_save_as_name_that_appears_while_writing_is_kept(
        self, window: MainWindow, folder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、名前を付けて保存の途中（作品を書く間）に現れた同じ名前のファイルを潰す
        target = folder / "第二稿.sme"
        real = save_project

        def appear_then_save(project: Project, path: Path, **options: Any) -> None:
            if path == target.resolve():
                target.write_text("other app", encoding="utf-8")
            real(project, path, **options)

        monkeypatch.setattr(main_window, "save_project", appear_then_save)
        _asked, box = _drive(window, _operation("save_project_as"), {"path": str(target)})
        assert isinstance(box.get("error"), ToolError)
        assert target.read_text(encoding="utf-8") == "other app"

    def test_a_subtitle_name_that_appears_while_writing_is_kept(
        self,
        qt_application: QApplication,
        folder: Path,
        video_media: MediaItem,
        transcript: Transcript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、字幕を並べている間に現れた同じ名前のファイルを潰す
        del qt_application
        target = folder / "字幕.srt"
        real = project_timeline

        def appear_then_list(project: Project) -> Any:
            target.write_text("other app", encoding="utf-8")
            return real(project)

        window = _open(make_loaded(video_media, transcript), folder)
        try:
            # 字幕の書き出し（save_subtitles）が字幕を並べる所で名前を現す 確かめの中身を
            # 作る所（道具の側）は別の並べ方なので、許したときには名前は無い
            monkeypatch.setattr(subtitles_module, "project_timeline", appear_then_list)
            _asked, box = _drive(window, _operation("export_subtitles"), {"path": str(target)})
            assert isinstance(box.get("error"), ToolError)
            assert target.read_text(encoding="utf-8") == "other app"
        finally:
            window.close()
            window.deleteLater()

    @pytest.mark.usefixtures("gpu")
    def test_a_snapshot_name_that_appears_while_drawing_is_kept(
        self, window: MainWindow, folder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、絵を描いている間に現れた同じ名前のファイルを潰す
        target = folder / "表紙.png"
        real = window.snapshot_image

        def appear_then_draw(frame: int | None = None) -> QImage:
            target.write_bytes(b"other app")
            return real(frame)

        monkeypatch.setattr(window, "snapshot_image", appear_then_draw)
        _asked, box = _drive(window, _operation("save_snapshot"), {"path": str(target)})
        assert isinstance(box.get("error"), ToolError)
        assert target.read_bytes() == b"other app"


@WINDOWS_WRITES
class TestStoppingWhilePlacing:
    """書き出し先へ置き始めた後の中止は断る（#327 CodeRabbit の指摘）

    画面のスレッドの様子（rendering か placing か）は、ワーカーの知らせが届いてから変わる
    その間に止めると、止めると答えたのに置き換えはそのまま終わり、前のファイルが消える
    """

    def test_a_stop_after_placing_began_is_refused_and_the_file_is_placed(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、AI は「止めています」を受け取るのに、書き出し先の前のファイルが置き換わる
        del qt_application
        entered = threading.Event()
        release = threading.Event()

        def fake_export(project: Project, settings: ExportSettings, **_options: Any) -> Path:
            settings.path.write_bytes(b"new video")
            return settings.path

        def held_copy(source: Path, path: Path, **options: Any) -> os.stat_result:
            entered.set()
            release.wait(10)
            return copy_atomic(source, path, **options)

        monkeypatch.setattr(ai_export, "run_export", fake_export)
        monkeypatch.setattr(ai_export, "copy_atomic", held_copy)
        target = tmp_path / "完成.mp4"
        target.write_bytes(b"previous")
        job = ai_export.AssistantExport()
        try:
            job.start(
                _small_project(),
                ExportSettings(path=target),
                lambda: target.resolve(),
            )
            # 画面のイベントは回さない 置き始めた知らせがまだ画面へ届いていない所を作る
            assert entered.wait(10)
            assert job.status()["state"] == "rendering"
            assert job.cancel() is False
            release.set()
            _wait_until(lambda: job.status()["state"] not in EXPORT_RUNNING)
            assert job.status()["state"] == "done"
            assert "止めています" not in str(job.status()["message"])
            assert target.read_bytes() == b"new video"
        finally:
            release.set()
            job.shutdown()
            job.deleteLater()


def _slow_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """本物の書き出しを、1 コマごとに少し待たせる 止める前に終わってしまわないように"""
    real = export_project

    def slow(project: Project, settings: ExportSettings, **options: Any) -> Path:
        should_cancel = options["should_cancel"]

        def gated() -> bool:
            time.sleep(0.02)
            return bool(should_cancel())

        return real(project, settings, progress=options["progress"], should_cancel=gated)

    monkeypatch.setattr(ai_export, "run_export", slow)


@pytest.mark.usefixtures("gpu")
@WINDOWS_WRITES
class TestPlacingIsGuarded:
    def test_the_folder_cannot_be_swapped_while_the_video_is_placed(
        self,
        window: MainWindow,
        folder: Path,
        tmp_path: Path,
        libx264: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、書き出し先へ置く直前にフォルダをリンクへ差し替えられ、確かめていない所へ書く
        outside = tmp_path / "外"
        outside.mkdir()
        refused: list[str] = []
        real_copy = copy_atomic

        def swap_then_copy(source: Path, path: Path, **options: Any) -> os.stat_result:
            try:
                folder.rename(tmp_path / "退けた")
            except OSError as exc:
                refused.append(str(exc))
            else:
                folder.symlink_to(outside, target_is_directory=True)
            return real_copy(source, path, **options)

        monkeypatch.setattr(ai_export, "copy_atomic", swap_then_copy)
        _asked, box = _drive(
            window,
            _operation("export_video"),
            {"path": "完成.mp4", "codec": libx264, "wait_seconds": 120},
        )
        result = box.get("result")
        assert isinstance(result, dict)
        assert list(outside.iterdir()) == []
        if sys.platform == "win32":
            # 持っている間はフォルダの名前を変えられない 書き出しは作品のフォルダへ届く
            assert refused
            assert result["state"] == "done"
            assert (folder / "完成.mp4").exists()
        else:
            assert result["state"] == "failed"

    def test_a_hard_link_to_an_original_made_during_the_export_is_not_replaced(
        self,
        qt_application: QApplication,
        folder: Path,
        video_media: MediaItem,
        libx264: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、書いている数分の間に書き出し先を元の素材へのハードリンクにされ、素材の
        # 名前の 1 つが書き出しに置き換わる（書き出しの出力が素材と同じ実体になる）
        del qt_application
        original = folder.parent / "素材置き場" / "素材.mp4"
        original.parent.mkdir()
        original.write_bytes(b"original")
        media = replace(video_media, path=original)
        target = folder / "完成.mp4"
        # 作れるかを先に試す 書き出しのスレッドの中で飛ばすと、飛ばしではなく失敗に見える
        probe = folder.parent / "試し"
        try:
            os.link(original, probe)
        except OSError as exc:
            pytest.skip(f"ハードリンクを作れない: {exc}")
        probe.unlink()
        real = export_project

        def export_then_link(project: Project, settings: ExportSettings, **options: Any) -> Path:
            written = real(project, settings, **options)
            os.link(original, target)
            return written

        monkeypatch.setattr(ai_export, "run_export", export_then_link)
        window = _open(_small_project(media=(media,)), folder)
        try:
            _asked, box = _drive(
                window,
                _operation("export_video"),
                {"path": str(target), "codec": libx264, "wait_seconds": 120},
            )
            result = box.get("result")
            assert isinstance(result, dict)
            assert result["state"] == "failed"
            assert "元の素材" in str(result["message"])
            assert original.read_bytes() == b"original"
            assert target.read_bytes() == b"original"
            assert _scratch_files() == []
        finally:
            window.close()
            window.deleteLater()
