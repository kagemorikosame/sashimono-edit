"""書き出しの画面に並ぶコーデック（#67）

何も選ばずに書き出したときの動きがここで決まる 開けない物が既定になると、
書き出しを押しただけで失敗する
"""

from __future__ import annotations

from collections.abc import Iterator

import av.error
import pytest
from PySide6.QtWidgets import QDialogButtonBox

from sashimono.core.commands import AddClip, AddScene, AddTrack
from sashimono.core.model import (
    Clip,
    Project,
    ProjectSettings,
    Scene,
    Timeline,
    Track,
    TrackKind,
)
from sashimono.engine.encode import exporter
from sashimono.ui import export_dialog
from sashimono.ui.export_dialog import AUTO_CODEC, ExportDialog


def _non_empty_project() -> Project:
    """30 コマのクリップを 1 本置いた作品 画面の判定だけを見るので素材は要らない"""
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create())
    return AddClip(track.id, Clip(timeline_start=0, duration=30)).apply(project)


@pytest.fixture
def qsv_does_not_open(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """この開発機と同じく、QSV は入っているが開けない機械を真似る"""

    def fake(name: str) -> None:
        if name == "h264_qsv":
            raise av.error.ArgumentError(22, "Invalid argument")

    # 開けるかの答えはプロセスの中で覚えている 前後で忘れないと、偽の答えが他の試験へ漏れる
    exporter._opens.cache_clear()
    monkeypatch.setattr(exporter, "_open_encoder", fake)
    monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("h264_qsv", "libx264"))
    yield
    exporter._opens.cache_clear()


def test_an_unopenable_qsv_is_hidden_so_export_does_not_fail_and_need_a_reselect(
    qsv_does_not_open: None,
) -> None:
    """開けない QSV を並べない

    並べると、開けない QSV を選んで書き出しを押し、失敗してから別の物を選び直すことになる
    """
    dialog = ExportDialog(Project.create(ProjectSettings()))
    try:
        codecs = [dialog._codec.itemData(index) for index in range(dialog._codec.count())]
        assert codecs == [AUTO_CODEC, "libx264"]
    finally:
        dialog.deleteLater()


def test_the_default_names_no_codec_so_a_refused_size_can_fall_back(
    qsv_does_not_open: None,
) -> None:
    """既定のまま書き出すと、コーデックを名指ししない

    先頭の候補を名指しで渡すと、試しには開けても作品の大きさで断られたとき
    （NVENC は幅 4096 まで）に次の候補へ落ちられず、画面から書き出すときだけ失敗する
    """
    dialog = ExportDialog(Project.create(ProjectSettings()))
    try:
        settings = dialog._settings()
        assert settings is not None
        assert settings.video_codec is None
    finally:
        dialog.deleteLater()


def test_a_codec_picked_by_hand_is_kept_not_swapped_for_another(qsv_does_not_open: None) -> None:
    """手で選んだコーデックは名指しで渡す 書き出し側が勝手に別の物へ変えない"""
    dialog = ExportDialog(Project.create(ProjectSettings()))
    try:
        dialog._codec.setCurrentIndex(dialog._codec.findData("libx264"))
        settings = dialog._settings()
        assert settings is not None
        assert settings.video_codec == "libx264"
    finally:
        dialog.deleteLater()


def test_without_any_codec_the_button_is_disabled_instead_of_doing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """使えるコーデックが 1 つも無ければ、書き出しのボタンを押せなくする

    押せるままだと、押しても黙って何も起きず、なぜ書き出せないのか分からない
    """
    # 文字列の名前で差し替えない 他の試験がモジュールを読み直していると、この試験が使う
    # ExportDialog とは別のモジュールを差し替えてしまい、何も確かめられない
    monkeypatch.setattr(export_dialog, "available_video_codecs", list)
    # 空でない作品にする 空だとそちらの理由でボタンが無効になり、試験が何も確かめない
    dialog = ExportDialog(_non_empty_project())
    try:
        assert not dialog._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    finally:
        dialog.deleteLater()


def _project_with_scene() -> tuple[Project, Scene]:
    """メインに 30 コマ、シーン「OP」に 12 コマ（範囲 2〜8 を指定）を置いた作品"""
    project = _non_empty_project()
    track = Track(TrackKind.VIDEO, "V1", (Clip(timeline_start=0, duration=12),))
    scene = Scene("OP", Timeline(rate=project.rate, tracks=(track,), work_area=(2, 8)))
    return AddScene(scene).apply(project), scene


class TestChoosingAScene:
    """メイン以外のシーンを選んで書き出す

    前はいつもメインを書き出し、シーンだけを動画にする道が無かった
    """

    def test_the_default_is_still_the_main_timeline(self) -> None:
        # 開いているシーンで出る物が変わると、書き出した物を取り違える
        project, _scene = _project_with_scene()
        dialog = ExportDialog(project, scene_name="OP")
        try:
            assert dialog.target.duration == 30
            assert dialog._scene.count() == 2
        finally:
            dialog.deleteLater()

    def test_choosing_a_scene_exports_its_timeline_and_range(self) -> None:
        # 壊れると、シーンを選んでもメインの 30 コマが出る 範囲もメインの物を当ててしまう
        project, scene = _project_with_scene()
        dialog = ExportDialog(project)
        try:
            dialog.select_scene(scene.id)
            assert dialog.target.timeline is scene.timeline
            assert dialog.target.duration == 12
            settings = dialog._settings()
            assert settings is not None
            assert settings.frame_range == (2, 8)
            assert settings.path.name == f"{project.name}_OP.mp4"
        finally:
            dialog.deleteLater()

    def test_an_empty_main_does_not_block_a_scene(self) -> None:
        # メインが空でもシーンに中身があれば書き出せる
        empty = Project.create(ProjectSettings())
        clip = Clip(timeline_start=0, duration=5)
        scene = Scene(
            "OP", Timeline(rate=empty.rate, tracks=(Track(TrackKind.VIDEO, "V1", (clip,)),))
        )
        project = AddScene(scene).apply(empty)
        dialog = ExportDialog(project)
        ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
        try:
            dialog.select_scene(scene.id)
            assert ok.isEnabled() == dialog._codecs_ready
            dialog.select_scene(None)
            assert not ok.isEnabled()
        finally:
            dialog.deleteLater()

    def test_a_path_typed_by_hand_is_kept(self) -> None:
        # 手で決めた出力先をシーンの名前で書き換えると、選び直しただけで行き先が変わる
        project, scene = _project_with_scene()
        dialog = ExportDialog(project)
        try:
            dialog._path.setText("C:/out/mine.mp4")
            dialog.select_scene(scene.id)
            assert dialog._path.text() == "C:/out/mine.mp4"
        finally:
            dialog.deleteLater()
