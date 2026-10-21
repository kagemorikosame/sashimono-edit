"""作品の外で覚えている名前を、AI へ送る結果で伏せる（#342 Codex の指摘）

本物の窓（MainWindow）で、AI に道具を使わせないまま字幕起こしの知らせを残し、別の作品を
開き直す 名前と場所を断った人の所で、どの読む道具の結果にも前の作品の名前が出ないことを見る
窓が開いたらその場で落とす
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import get_type_hints

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.ai import host as host_module
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import EditorHost
from sashimono.ai.operations import OPERATIONS
from sashimono.ai.outbound import ASKED, Outbound, SendChoice
from sashimono.asr.service import Job
from sashimono.core.io.serialize import save_project
from sashimono.core.model import MediaItem, Transcript
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import make_loaded
from tests.ai.test_bridge import call_in_thread

MARK = "ZZSECRET"

#: 作品の外の物を AI へ返しうるホストの口 名前を覚える口（remembered_names）を通すと決めた物
#: 文字や辞書を返す口を EditorHost に足したら、ここへ足し、その口が返す名前を
#: remembered_names からも返す 足さないと下の試験が落ちる
TEXT_SOURCES = {
    "transcription_status",
    "start_transcription",
    "app_settings",
    # 保存と書き出し（#327） 保存の知らせと書き出しの様子に、書いた場所・置き換える前の
    # ファイルを残した場所・控えの置き場が入る 書いた場所と残した場所は remembered_names から返す
    "write_project",
    "export_status",
}
#: 本人の持ち物と設定（保存したプリセット・エイリアスと着せ方の設定 #308、音量の基準 #337、
#: ダッキングの下げ幅と速さ #329） 送る（使う道具は sends に Outbound.APP_SETTINGS を書く）
#: テンプレートの棚（#328）も本人の置き場の物 名前はアプリの設定として送り、フォルダの名前は
#: 名前と場所（一覧の folder の項目で伏せる） ファイルの場所は道具から返さない
#: 強調の動きの強さと長さの既定（#331）も本人の設定
OWN_LIBRARY = {
    "saved_looks",
    "look_rules",
    "loudness_target",
    "duck_shape",
    "emphasis_style",
    "seam_style",
    # 色と雑音の自動補正の強さ・下げる量の既定（#336）
    "auto_color_strength",
    "denoise_reduction",
    "template_shelf",
    # AI に見せる絵の長い辺の上限（#344） 絵を返す道具が縮めた訳と断りの文に載せる
    "image_edge",
}
#: 作品の中身か、作品と書き出しの場所（伏せ字の表に直に入る）か、作品の外の名前を返す口そのもの
PROJECT_SIDE = {
    "document",
    "project",
    "active_scene",
    "set_active_scene",
    "playhead",
    "seek",
    "selected_clip",
    "select_clip",
    "selected_clips",
    "select_clips",
    "apply_commands",
    # 取り消しとやり直し（#333） 作品の履歴を辿るだけで、何も返さない
    "undo",
    "redo",
    "stop_playback",
    "render_png",
    # コマを並べた絵（#320） 描くのは作品の中身だけ 送る道具は sends に絵を書く
    "start_sheet",
    # 音量の推移のグラフ（#338） 描くのは道具が渡した時刻と値だけ 送る道具は sends に絵を書く
    "draw_sound_graph",
    "probe",
    "splits_media",
    "subtitle_wrap_share",
    "analyze",
    "waveform",
    # 色を測るサムネイル（#336） 素材の絵の縮めた物 道具は測った数だけを返し、絵は送らない
    "filmstrip",
    "request_scenes",
    "scene_state",
    "scene_waiter",
    # 仕上がりの検査（#337） 調べるのは作品か、作品と書き出しのフォルダの中の動画だけ
    "request_check",
    "check_state",
    "check_waiter",
    "project_path",
    "export_folder",
    "remembered_names",
    # 保存と書き出し（#327） 書く先は作品と書き出しのフォルダの中だけ 返すのは書いた場所・
    # コマ・止められたか・待つ関数で、書いた場所は remembered_names からも返す
    "write_subtitles",
    "write_snapshot",
    "start_export",
    "export_waiter",
    "cancel_export",
}


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 作品を開く窓は試験の中で差し替える"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QMessageBox, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)


@pytest.fixture
def window(
    qt_application: QApplication,
    video_media: MediaItem,
    transcript: Transcript,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[MainWindow]:
    """目印の名前の作品で字幕起こしの知らせを残し、AI に道具を使わせないまま別の作品を開いた窓"""
    del qt_application
    media = replace(video_media, path=Path(f"C:/Users/{MARK}人/{MARK}素材.mp4"))
    marked = replace(make_loaded(media, transcript), name=f"{MARK}作品")
    created = MainWindow(marked, confirm_unsaved=False)
    created._path = tmp_path / f"{MARK}作品.sme"
    # 起こしが終わったときと同じ道で知らせを書く（poll_transcription の中と同じ 2 行）
    panel = created._subtitles
    name = panel._describe_job(Job(media.id, media.path))
    panel._job_notes.append(f"{name}: 終了した")
    other = tmp_path / "ふつう.sme"
    save_project(replace(make_loaded(video_media, transcript), name="ふつう"), other)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: (str(other), ""))
    created.open_project()
    assert created.project_path == other
    yield created
    created._chat.close_session()
    created.close()
    created.deleteLater()


def _call(window: MainWindow, name: str) -> str:
    bridge: EditorBridge = window._chat._bridge
    bridge.set_send_choices(
        dict.fromkeys(ASKED, SendChoice.ALLOW) | {Outbound.FILES: SendChoice.DENY}
    )
    operation = next(op for op in OPERATIONS if op.name == name)
    done, box = call_in_thread(bridge, operation, {})
    deadline = time.monotonic() + 10
    while not done.is_set() and time.monotonic() < deadline:
        window._chat._poll()
        time.sleep(0.005)
    assert done.is_set()
    return json.dumps(
        {"result": box.get("result"), "error": str(box.get("error") or "")},
        ensure_ascii=False,
        default=str,
    )


class TestAfterOpeningAnotherProject:
    def test_the_old_names_are_still_in_the_records(self, window: MainWindow) -> None:
        # 前提が崩れると下の試験が何も見ていないことになる 知らせには前の素材の名前が残る
        assert MARK in window.transcription_status()

    @pytest.mark.parametrize(
        "name",
        # 絵は描かない（名前は混ざらず、本物の窓で描くと遅い）
        [op.name for op in OPERATIONS if not (op.writes or op.asks or op.name == "preview_frame")],
    )
    def test_reading_tools_do_not_send_the_old_names(self, window: MainWindow, name: str) -> None:
        # 壊れると、AI に道具を使わせないまま別の作品を開いた人の所で、前の作品の素材の名前
        # （ユーザー名入りの場所も）が、字幕起こしの知らせに混ざって外へ出る
        assert MARK not in _call(window, name)

    def test_the_host_returns_what_it_remembers(self, window: MainWindow) -> None:
        # 壊れると、作品の外で覚えている名前の口が空のまま、伏せ字の表へ何も渡らない
        remembered = window.remembered_names()
        assert any(MARK in value for value in remembered)


class TestEveryTextSourceGoesThroughThePort:
    def test_the_list_of_text_sources_matches_the_host(self) -> None:
        # 壊れると、作品の外の物を返す口を EditorHost に足した人が、その口の名前を
        # remembered_names から返すのを忘れ、別の作品を開いた後に伏せずに送る
        hints = {
            name: hint
            for name, value in vars(EditorHost).items()
            if callable(value) and not name.startswith("_")
            for hint in [get_type_hints(value, vars(host_module)).get("return")]
        }
        texts = {name for name, hint in hints.items() if hint in (str, dict[str, object])}
        assert texts == TEXT_SOURCES
        assert "remembered_names" in hints

    def test_every_port_of_the_host_is_sorted(self) -> None:
        # 壊れると、作品の外の物（本人の持ち物の名前・置き場など）を返す口を足した人が、
        # 送り方を決めないまま道具から返し、名前と場所を断った人の所でも伏せずに送る
        # 口を足したら、下の 3 つのどれかに入れる 作品の外の名前を返すなら TEXT_SOURCES に入れて
        # remembered_names からも返す 本人の持ち物なら OWN_LIBRARY に入れ、使う道具の sends に
        # アプリの設定（Outbound.APP_SETTINGS）を書く
        ports = {name for name in vars(EditorHost) if not name.startswith("_")}
        sorted_ports = [TEXT_SOURCES, OWN_LIBRARY, PROJECT_SIDE]
        assert set().union(*sorted_ports) == ports
        assert sum(len(group) for group in sorted_ports) == len(ports)

    def test_tools_that_read_the_own_library_declare_app_settings(self) -> None:
        # 壊れると、プリセットやエイリアスの名前を断りの文に並べる道具が、送る物にそれを書かず、
        # 設定画面と個人情報の扱いの一覧に無い物が外へ出る
        for name in (
            "place_subtitles",
            "check_output",
            "list_looks",
            "apply_preset",
            "place_alias",
            "place_template",
            "apply_template",
            # 強調の動き（#331） 結果に本人の設定の強さ・長さ・動き方の既定が載る
            "emphasize",
            # つなぎ目（#332） 結果に本人の設定の処理の種類と長さの既定が載る
            "smooth_cuts",
            "jet_cut",
            # 色と雑音の自動補正（#336） 結果に本人の設定の強さと下げる量の既定が載る
            "auto_color",
            "reduce_noise",
            # 絵の長い辺の上限（#344） 縮めた訳と断りの文に本人の設定の画素が載る
            "preview_frame",
            "contact_sheet",
            "preview_range",
        ):
            operation = next(op for op in OPERATIONS if op.name == name)
            assert Outbound.APP_SETTINGS in operation.sends, name
