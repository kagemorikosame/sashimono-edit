"""作品を開いたときの見つからない素材の知らせと、探し直し（#447）"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from sashimono.core.io.serialize import save_project
from sashimono.core.model import Project
from sashimono.engine.decode import probe_media
from sashimono.engine.relink_check import plan_relink
from sashimono.ui.main_window import MainWindow
from tests.media_fixtures import SampleMedia
from tests.ui.ai_tool_calls import call_tool


@pytest.fixture
def moved(tmp_path: Path, sample_av: SampleMedia, sample_long: SampleMedia) -> Path:
    """素材のフォルダを移して、作品から見つからなくした作品の場所"""
    before = tmp_path / "元の素材"
    before.mkdir()
    a = shutil.copy(sample_av.path, before / "本編.mp4")
    b = shutil.copy(sample_long.path, before / "別の撮影.mp4")
    project = Project.create(media=(probe_media(Path(a)), probe_media(Path(b))), name="移す")
    folder = tmp_path / "作品"
    folder.mkdir()
    save_project(project, folder / "作品.sme")
    shutil.move(before, tmp_path / "移した素材")
    return folder / "作品.sme"


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(None, confirm_unsaved=False)
    yield created
    created.close()
    created.deleteLater()


def _open(window: MainWindow, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(path), ""))
    window.open_project()


def test_opening_lists_the_missing_media(
    window: MainWindow, moved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """知らせが無いと、絵が映らず音が無いだけで、何が起きたのか分からない"""
    _open(window, moved, monkeypatch)
    dialog = window._missing_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog._list.count() == 2
    assert "2 本見つかりません" in dialog._summary.text()


def test_choosing_one_finds_the_rest(
    window: MainWindow, moved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 本を選んだら残りも付け替わらないと、素材の数だけ同じ窓を出させる 取り消しで戻せる"""
    _open(window, moved, monkeypatch)
    first = window.document.project.media[0]
    target = moved.parent.parent / "移した素材" / "本編.mp4"
    text = window._relink_from_dialog(first.id, target)
    assert "2 本を付け替えました" in text
    assert [m.path.parent for m in window.document.project.media] == [target.parent] * 2
    assert window._missing_dialog is not None
    window._missing_dialog.refresh()
    assert window._missing_dialog._list.count() == 0
    window.undo()
    assert all(not m.path.is_file() for m in window.document.project.media)


def test_a_different_file_is_questioned(
    window: MainWindow, moved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """長さの違う別の動画を黙って付け替えると、クリップの時刻が別の絵を指す"""
    _open(window, moved, monkeypatch)
    first = window.document.project.media[0]
    wrong = moved.parent.parent / "移した素材" / "別の撮影.mp4"
    asked: list[str] = []

    def no(*arguments: object, **_keywords: object) -> QMessageBox.StandardButton:
        asked.append(str(arguments[2]))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", no)
    assert window._relink_from_dialog(first.id, wrong) == "付け替えませんでした"
    assert asked and "長さが違います" in asked[0]
    assert not window.document.project.media[0].path.is_file()


def test_a_same_named_but_different_file_is_not_taken(
    window: MainWindow, moved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """名前だけで見つけた別の撮影を付け替えると、気付かないまま別の絵で書き出す"""
    _open(window, moved, monkeypatch)
    folder = moved.parent.parent / "移した素材"
    # 2 本目と同じ名前で、中身は 1 本目（長さが違う）を置く
    (folder / "別の撮影.mp4").unlink()
    shutil.copy(folder / "本編.mp4", folder / "別の撮影.mp4")
    project = window.document.project
    plan = plan_relink(project, project.media[0].id, folder / "本編.mp4")
    assert [media_id for media_id, _ in plan.changes] == [project.media[0].id]
    assert plan.skipped and plan.skipped[0][0] == "別の撮影.mp4"


def test_the_menu_says_when_nothing_is_missing(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """探し直しの窓が空のまま出ると、まだ何かを探さないといけないように見える"""
    told: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda _p, _t, text: told.append(text))
    window.show_missing_media()
    assert told and "見つかっています" in told[0]
    assert window._missing_dialog is None


def test_the_assistant_hears_and_relinks(
    window: MainWindow, moved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AI に見つからない素材が伝わらないと、映らない理由を別の所に探しに行く"""
    _open(window, moved, monkeypatch)
    listed = call_tool(window, "list_media", {})
    assert all(row["missing"] for row in listed["media"])
    first = window.document.project.media[0]
    target = moved.parent.parent / "移した素材" / "本編.mp4"
    result = call_tool(window, "relink_media", {"media_id": str(first.id), "path": str(target)})
    assert len(result["relinked"]) == 2 and result["still_missing"] == 0
