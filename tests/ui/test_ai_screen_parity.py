"""画面と同じ道具（#349）を本物の窓で動かす

人の Ctrl+C・Ctrl+V と AI のコピー・貼り付けが同じ置き場を使うか、AI の挿入貼り付けが人の
設定に従うか、開いたシーンを消したときに窓がメインへ戻るかを見る 道具の決まりそのものは
``tests/ai/test_screen_parity_tools.py`` が偽のホストで見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox

from sashimono.ai.host import ToolError
from sashimono.ai.operations import ALIGN_ANCHORS, find_operation
from sashimono.core.commands import Author
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.io.serialize import save_project
from sashimono.core.model import Clip, LayerMode, MediaItem, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preview_handles import ALIGNMENTS
from tests.ui.test_preview_handles import _clip as _picture_clip
from tests.ui.test_preview_handles import _media as _picture_media
from tests.ui.test_preview_handles import _project as _picture_project


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 道具はどれも取り消せる編集で、人に何も尋ねないはず"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QInputDialog, "getText", popped)
    monkeypatch.setattr(QInputDialog, "getItem", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


def _text(text: str, start: int = 0) -> Clip:
    return with_fixed_items(
        Clip(timeline_start=start, duration=90, source=TEXT.create(text=text)), picture=True
    )


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    base = Project.create()
    track = Track(TrackKind.VIDEO, "V1", (_text("一つ目"), _text("二つ目", start=100)))
    created = MainWindow(
        base.with_timeline(replace(base.timeline, tracks=(track,))), confirm_unsaved=False
    )
    yield created
    created.close()
    created.deleteLater()


def _run(window: MainWindow, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(window, arguments)


def _who(window: MainWindow) -> Author | None:
    """次に戻す段を誰が変えたか 関数にするのは、続けて読むと型の見方が前の値に固まるため"""
    return window.document.undo_by


def _texts(window: MainWindow) -> list[tuple[int, str]]:
    return sorted(
        (clip.timeline_start, str(clip.source.params["text"]))
        for track in window.document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None
    )


class TestTheClipboardIsShared:
    def test_what_the_assistant_copies_is_pasted_by_ctrl_v(self, window: MainWindow) -> None:
        # 壊れると、AI に「コピーしておいて」と頼んだ物を人が Ctrl+V で貼れない
        first = window.document.project.timeline.tracks[0].clips[0]
        _run(window, "copy_clips", clip_ids=[str(first.id)])
        window.seek(150)
        assert window._timeline.paste_at_playhead()
        assert (150, "一つ目") in _texts(window)

    def test_what_the_person_copies_is_pasted_by_the_assistant(self, window: MainWindow) -> None:
        # 壊れると、人がコピーした物を AI に「3 秒の所へ貼って」と頼んでも貼れない
        second = window.document.project.timeline.tracks[0].clips[1]
        window._timeline.set_selection([second.id])
        assert window._timeline.copy_selected()
        pasted = _run(window, "paste_clips", at_frame=400)["pasted"]
        assert (400, "二つ目") in _texts(window)
        assert [str(clip_id) for clip_id in window.selected_clips] == pasted

    def test_a_cut_by_the_assistant_goes_back_with_one_ctrl_z(self, window: MainWindow) -> None:
        # 壊れると、AI が切り取った物を人が 1 回の Ctrl+Z で戻せない
        before = window.document.project
        first = window.document.project.timeline.tracks[0].clips[0]
        _run(window, "cut_clips", clip_ids=[str(first.id)])
        assert (0, "一つ目") not in _texts(window)
        window.undo()
        assert window.document.project == before

    @pytest.mark.parametrize("all_tracks", [True, False])
    def test_insert_paste_reads_the_person_s_setting(
        self, window: MainWindow, all_tracks: bool
    ) -> None:
        # 壊れると、設定の「挿入貼り付けで押し出す」を変えても、AI の挿入貼り付けだけ前のまま
        window._timeline.set_insert_all_tracks(all_tracks)
        assert window.insert_paste_all_tracks is all_tracks


class TestScenesInTheWindow:
    def test_removing_the_open_scene_shows_the_main_again(self, window: MainWindow) -> None:
        # 壊れると、AI が開いているシーンを消した後、窓が消えたシーンを開いたまま何も描けない
        scene_id = window.create_scene("場面")
        assert scene_id is not None and window.active_scene == scene_id
        _run(window, "remove_scene")
        assert window.active_scene is None
        assert window.document.project.scenes == ()
        assert window.view_project.timeline == window.document.project.timeline

    def test_renaming_the_open_scene_keeps_it_open(self, window: MainWindow) -> None:
        # 壊れると、名前を変えただけで開いているシーンが閉じ、AI の次の道具がメインに当たる
        scene_id = window.create_scene("場面")
        _run(window, "rename_scene", name="冒頭")
        assert window.active_scene == scene_id
        assert [scene.name for scene in window.document.project.scenes] == ["冒頭"]


class TestOpeningAndNewInTheWindow:
    @pytest.fixture
    def saved(self, window: MainWindow, tmp_path: Path) -> Path:
        """窓の作品を tmp_path/作品.sme に保存し、隣に 第二稿.sme を置く 返すのは第二稿の場所"""
        window._path = tmp_path / "作品.sme"
        window._write_project_file(window._path)
        other = tmp_path / "第二稿.sme"
        save_project(replace(Project.create(), name="第二稿"), other)
        return other

    def test_opening_swaps_the_window_without_any_popup(
        self, window: MainWindow, saved: Path
    ) -> None:
        # 壊れると、AI の〔開く〕で保存を尋ねる窓や錠の窓が出て、答えるまで AI が止まる
        window.document.begin_checkpoint("AI: 開く", by=Author.AI)
        _run(window, "open_project", path=str(saved))
        assert window.document.project.name == "第二稿"
        assert window.project_path == saved.resolve()
        assert not window.is_modified
        # 同じ指示の中の編集は断る
        with pytest.raises(ToolError, match="次の指示"):
            _run(window, "add_text", text="字")
        window.document.end_checkpoint()
        _run(window, "add_text", text="字")

    def test_unsaved_changes_are_kept(self, window: MainWindow, saved: Path) -> None:
        # 壊れると、人が保存していない編集が、AI の〔開く〕で黙って消える
        _run(window, "add_text", text="保存していない")
        before = window.document.project
        with pytest.raises(ToolError, match="save_project"):
            _run(window, "open_project", path=str(saved))
        assert window.document.project is before

    def test_a_new_project_uses_the_person_s_settings_for_new_projects(
        self, window: MainWindow
    ) -> None:
        # 壊れると、AI が作った作品だけ置き方の方式が本人の「新しい作品」の設定と違う
        window._preferences = replace(window._preferences, new_project_layers=LayerMode.MIXED)
        window._mark_saved()
        _run(window, "new_project", width=1280, height=720)
        settings = window.document.project.settings
        assert settings.resolution == (1280, 720)
        assert settings.layer_mode == LayerMode.MIXED
        assert window.project_path is None


class TestAlignmentInTheWindow:
    def test_the_anchors_are_the_panel_s_anchors(self) -> None:
        # 壊れると、設定パネルに寄せ先を足しても AI の道具が知らず、名前が食い違う
        assert tuple(name for name, *_ in ALIGNMENTS) == ALIGN_ANCHORS

    def test_the_assistant_aligns_like_the_panel_button(self, qt_application: QApplication) -> None:
        # 壊れると、AI の〔左上へ揃える〕と人のボタンで寄せる所が違う
        del qt_application
        clip = _picture_clip(_picture_media())
        project, _ = _picture_project(clip)
        created = MainWindow(project, confirm_unsaved=False)
        try:
            created.seek(30)
            created.select_clip(clip.id)
            before = created.document.project
            assert _run(created, "align_clip", clip_id=str(clip.id), anchor="top_left")["moved"]
            by_ai = created.document.project
            assert by_ai != before
            assert _who(created) is Author.AI
            created.undo()
            created._align_selected("top_left")
            assert created.document.project == by_ai
            assert _who(created) is Author.PERSON
        finally:
            created.close()
            created.deleteLater()

    def test_the_lists_come_from_the_add_menu(self, window: MainWindow) -> None:
        # 壊れると、右クリックの〔カスタムオブジェクト〕に並ぶ物と AI の一覧が食い違う
        window._timeline.add_sources.custom_objects = lambda: ()
        assert window.custom_objects == ()
        assert _run(window, "list_custom_objects")["custom_objects"] == []


class TestRemovingMediaInTheWindow:
    def test_the_analysis_is_dropped_like_the_pool_menu(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 壊れると、AI が素材一覧から外した素材の波形やサムネイルを、アプリがメモリに抱え続ける
        del qt_application
        media = MediaItem(path=tmp_path / "使わない.wav", duration=Fraction(1))
        created = MainWindow(Project.create(media=(media,)), confirm_unsaved=False)
        try:
            _run(created, "remove_media", media_id=str(media.id))
            assert created.document.project.media == ()
            assert media.id in created._forgotten
        finally:
            created.close()
            created.deleteLater()
