"""持ち出し用パッケージ（#423）を本物の窓で書き出して開く

人の〔持ち出し用に書き出す…〕と AI の export_package・open_package が、同じ裏の仕事を通り、
開いた後は展開した素材を指す普段の作品になることを見る
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from sashimono.ai.host import UNSAVED_REFUSAL, ToolError
from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.io.serialize import save_project
from sashimono.core.model import Clip, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import probe_media
from sashimono.engine.package.archive import PACKAGE_SUFFIX, OpenedPackage, PackResult
from sashimono.ui.main_window import MainWindow
from sashimono.ui.package_dialog import PackageExportDialog
from tests.ai.conftest import allow_sending
from tests.media_fixtures import SampleMedia
from tests.ui.ai_tool_calls import call_tool
from tests.waiting import wait_until

WINDOWS_WRITES = pytest.mark.skipif(
    sys.platform != "win32", reason="AI がファイルを書くのは Windows だけ（pin_holds_the_folder）"
)


def _project(sample: SampleMedia) -> Project:
    item = probe_media(sample.path)
    track = Track(TrackKind.VIDEO, "V1")
    project = Project.create(
        ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)), media=(item,)
    )
    project = AddTrack(track).apply(project)
    clip = Clip(timeline_start=0, duration=15, media_id=item.id, source_in=Fraction(1, 2))
    return AddClip(track.id, clip).apply(project)


@pytest.fixture
def window(
    qt_application: QApplication, tmp_path: Path, sample_av: SampleMedia
) -> Iterator[MainWindow]:
    del qt_application
    folder = tmp_path / "作品"
    folder.mkdir()
    project = _project(sample_av)
    save_project(project, folder / "作品.sme")
    created = MainWindow(project, path=folder / "作品.sme", confirm_unsaved=False)
    created._chat._auto.setChecked(True)
    allow_sending(created._chat._bridge)
    # 展開する場所は試験ごとの一時フォルダ 本人の置き場を汚さない
    created._preferences = replace(created._preferences, package_folder=str(tmp_path / "展開"))
    yield created
    created.close()
    created.deleteLater()


def _wait_package(window: MainWindow) -> None:
    wait_until(
        lambda: window._package.wait(0.01), "パッケージ", step=QApplication.processEvents, pause=0.0
    )
    QApplication.processEvents()


@WINDOWS_WRITES
def test_the_assistant_takes_a_package_out_and_brings_it_back(window: MainWindow) -> None:
    """AI が書いたパッケージを AI が開けないと、持ち出しの往復を AI に頼めない
    開いた後の作品が元の素材を指したままだと、素材の無い PC では絵が出ない"""
    original = window.document.project.media[0].path
    started = call_tool(
        window, "export_package", {"path": f"持ち出し{PACKAGE_SUFFIX}"}, allow_irreversible=True
    )
    assert "started" in started
    _wait_package(window)
    status = window.package_status()
    assert status["state"] == "done", status
    package = window.project_path.parent / f"持ち出し{PACKAGE_SUFFIX}"  # type: ignore[union-attr]
    assert package.is_file()
    assert isinstance(window._package.result, PackResult)

    call_tool(
        window, "open_package", {"path": f"持ち出し{PACKAGE_SUFFIX}"}, allow_irreversible=True
    )
    _wait_package(window)
    assert window.package_status()["state"] == "done"
    opened = window._package.result
    assert isinstance(opened, OpenedPackage)
    assert window.project_path == opened.project_path
    media = window.document.project.media[0]
    assert media.path != original and media.path.is_file()
    assert media.path.is_relative_to(Path(window._preferences.package_folder))
    clip = window.document.project.timeline.tracks[0].clips[0]
    assert (clip.timeline_start, clip.duration, clip.source_in) == (0, 15, Fraction(1, 2))


def test_the_assistant_does_not_throw_away_unsaved_work(window: MainWindow, tmp_path: Path) -> None:
    """保存していない変更があるのに展開した作品へ差し替えると、その変更が黙って消える"""
    package = window.project_path.parent / f"p{PACKAGE_SUFFIX}"  # type: ignore[union-attr]
    package.write_bytes(b"")
    window.document.execute(AddTrack(Track(TrackKind.AUDIO, "A9")))
    with pytest.raises(ToolError, match=UNSAVED_REFUSAL[:10]):
        call_tool(window, "open_package", {"path": package.name}, allow_irreversible=True)
    assert window._package.status()["state"] == "idle"


def test_the_person_exports_from_the_menu_and_the_choice_is_remembered(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """窓で選んだ物を覚えないと、毎回同じ選び直しをさせる 終わりの知らせが無いと、
    入れなかった書体やスクリプトに気付けない"""
    told: list[str] = []

    def accept(dialog: PackageExportDialog) -> int:
        dialog._whole.setChecked(True)
        dialog._keep.setChecked(True)
        dialog._path.setText(str(tmp_path / "人の持ち出し"))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(PackageExportDialog, "exec", accept)
    monkeypatch.setattr(QMessageBox, "information", lambda _parent, _title, text: told.append(text))
    monkeypatch.setattr(
        window, "_remember_preferences", lambda chosen: setattr(window, "_preferences", chosen)
    )
    window.export_package()
    _wait_package(window)
    assert (tmp_path / f"人の持ち出し{PACKAGE_SUFFIX}").is_file()
    assert window._preferences.package_whole and window._preferences.package_keep_metadata
    assert told and "書き出しました" in told[0]
