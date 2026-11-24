"""書き出しの窓の「形式」（#527）

形式を替えると名前の拡張子・コーデックの並び・出す欄が合わせて変わる 窓で最後に選んだ
形式は本人の設定に覚え（設定画面には出さない）、次に開いた窓の初めの形式にする
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from sashimono.core.commands import AddClip, AddTrack, Document, insert_media
from sashimono.core.model import Clip, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import probe_media
from sashimono.ui import export_dialog
from sashimono.ui.export_dialog import AUTO_CODEC, ExportDialog
from sashimono.ui.main_window import MainWindow
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.media_fixtures import SampleMedia
from tests.ui.export_codecs import settled


@pytest.fixture(autouse=True)
def known_codecs(monkeypatch: pytest.MonkeyPatch) -> None:
    """使える候補を決めておく 機械の GPU で並びが変わらないように"""
    monkeypatch.setattr(export_dialog, "available_video_codecs_now", lambda _p: ["libx264"])
    monkeypatch.setattr(
        export_dialog, "available_web_codecs_now", lambda _p: ["libvpx-vp9", "libsvtav1"]
    )


def _picture_only() -> Project:
    """絵だけの 30 コマ 鳴る物が無い"""
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create())
    return AddClip(track.id, Clip(timeline_start=0, duration=30)).apply(project)


@pytest.fixture
def sounding(sample_av: SampleMedia) -> Project:
    """映像 + 音声の素材を 1 本置いた作品"""
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(sample_av.path)):
        document.execute(command)
    return document.project


def _codecs(dialog: ExportDialog) -> list[object]:
    return [dialog._codec.itemData(index) for index in range(dialog._codec.count())]


def _ok(dialog: ExportDialog) -> bool:
    return dialog._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_choosing_webm_changes_the_name_and_the_codecs(qt_application: QApplication) -> None:
    """壊れると、WebM を選んでも MP4 の名前のまま H.264 を並べ、入れ物に断られる"""
    del qt_application
    dialog = settled(ExportDialog(_picture_only(), folder=Path("C:/出力")))
    try:
        assert Path(dialog._path.text()).suffix == ".mp4"
        assert _codecs(dialog) == [AUTO_CODEC, "libx264"]
        dialog.select_format("webm")
        assert Path(dialog._path.text()).suffix == ".webm"
        assert _codecs(dialog) == [AUTO_CODEC, "libvpx-vp9", "libsvtav1"]
        # 章を持てない入れ物では押せない（値は本人の設定なので残す）
        assert not dialog._chapters.isEnabled()
        settings = dialog._settings()
        assert settings is not None and settings.path.suffix == ".webm"
        assert settings.video_codec is None
    finally:
        dialog.deleteLater()


def test_webm_auto_is_only_vp9(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、VP9 が使えない機械で AV1 を自動と出し、自動のまま押して失敗する"""
    del qt_application
    monkeypatch.setattr(export_dialog, "available_web_codecs_now", lambda _p: ["libsvtav1"])
    dialog = settled(ExportDialog(_picture_only(), export_format="webm"))
    try:
        assert _codecs(dialog) == ["libsvtav1"]
        settings = dialog._settings()
        assert settings is not None and settings.video_codec == "libsvtav1"
    finally:
        dialog.deleteLater()


def test_audio_only_hides_the_video_rows(qt_application: QApplication, sounding: Project) -> None:
    """壊れると、音声だけなのに映像のコーデックやビットレートを替えられ、効かない理由を探す"""
    del qt_application
    dialog = settled(ExportDialog(sounding))
    try:
        dialog.select_format("mp3")
        form = dialog._form
        assert not form.isRowVisible(dialog._codec)
        assert not form.isRowVisible(dialog._bitrate)
        assert form.isRowVisible(dialog._audio_quality)
        dialog._audio_quality.setCurrentIndex(dialog._audio_quality.findData(320))
        settings = dialog._settings()
        assert settings is not None
        assert settings.path.suffix == ".mp3"
        assert settings.video_codec is None and settings.audio_bitrate == 320_000
        assert _ok(dialog)
        # WAV は可逆なので音質の欄も出さない
        dialog.select_format("wav")
        assert not form.isRowVisible(dialog._audio_quality)
        dialog.select_format("mp4")
        assert form.isRowVisible(dialog._codec) and form.isRowVisible(dialog._bitrate)
    finally:
        dialog.deleteLater()


def test_audio_only_without_sound_cannot_be_pressed(qt_application: QApplication) -> None:
    """壊れると、鳴る物の無い作品で押してから断られるか、無音のファイルを書く"""
    del qt_application
    dialog = settled(ExportDialog(_picture_only()))
    try:
        assert _ok(dialog)
        dialog.select_format("flac")
        assert not _ok(dialog)
        assert not dialog._silent.isHidden()
        dialog.select_format("mov")
        assert _ok(dialog) and dialog._silent.isHidden()
    finally:
        dialog.deleteLater()


def test_a_typed_suffix_picks_the_format(qt_application: QApplication) -> None:
    """壊れると、名前を .mkv に打ち替えても MP4 のまま書き出され、名前が .mp4 に付け替わる"""
    del qt_application
    dialog = settled(ExportDialog(_picture_only()))
    try:
        dialog._path.setText("C:/出力/作品.mkv")
        dialog._path.textEdited.emit("C:/出力/作品.mkv")
        assert dialog.chosen_format == "mkv"
        assert dialog._path.text() == "C:/出力/作品.mkv"
        # 知らない拡張子は後ろへ足す（名前の一部の点を消さない）
        dialog._path.setText("C:/出力/作品.v2")
        settings = dialog._settings()
        assert settings is not None and settings.path.name == "作品.v2.mkv"
    finally:
        dialog.deleteLater()


def test_the_remembered_format_is_the_first_one(qt_application: QApplication) -> None:
    """壊れると、BGM だけを何本も書き出す人が窓を開くたびに形式を選び直す"""
    del qt_application
    dialog = settled(ExportDialog(_picture_only(), export_format="flac"))
    unknown = settled(ExportDialog(_picture_only(), export_format="新しい版の形式"))
    try:
        assert dialog.chosen_format == "flac"
        assert dialog._path.text().endswith(".flac")
        assert unknown.chosen_format == "mp4"
    finally:
        dialog.deleteLater()
        unknown.deleteLater()


def test_a_broken_remembered_format_falls_back(tmp_path: Path) -> None:
    """壊れると、手で書き換えた設定の知らない形式で窓が何も選ばれずに開く"""
    store = PreferenceStore(tmp_path / "preferences.json")
    store.save(Preferences(export_format="webm"))
    assert store.load().export_format == "webm"
    (tmp_path / "preferences.json").write_text('{"export_format": "gif"}', encoding="utf-8")
    assert store.load().export_format == "mp4"


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


def test_the_window_remembers_the_format(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、窓で選んだ形式を次に開いたときに忘れている"""
    assert window._preferences.export_format == "mp4"

    def choose_mp3(dialog: ExportDialog) -> int:
        dialog.select_format("mp3")
        return 0

    monkeypatch.setattr(ExportDialog, "exec", choose_mp3)
    window.export()
    assert window._preferences.export_format == "mp3"
    assert PreferenceStore().load().export_format == "mp3"

    seen: list[str] = []

    def look(dialog: ExportDialog) -> int:
        seen.append(dialog.chosen_format)
        return 0

    monkeypatch.setattr(ExportDialog, "exec", look)
    window.export()
    assert seen == ["mp3"]
