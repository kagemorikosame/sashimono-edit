"""作品を開いたときの見つからない素材の知らせと、探し直し（#447）"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from sashimono.ai.host import ToolError
from sashimono.core.io.serialize import save_project
from sashimono.core.model import Clip, PackedSpan, Project, Track, TrackKind
from sashimono.core.relink import RelinkMedia
from sashimono.engine.audio.mixer import AudioMixer
from sashimono.engine.decode import probe_media
from sashimono.engine.relink_check import differs, plan_relink
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


def test_other_stream_numbers_are_questioned(sample_av: SampleMedia) -> None:
    """数が同じでも道の番号が違うと、クリップの覚えている番号が別の音を指す"""
    item = probe_media(sample_av.path)
    audio = tuple(replace(stream, index=stream.index + 5) for stream in item.audio_streams)
    reason = differs(replace(item, audio_streams=audio), sample_av.path)
    assert reason is not None and "音声の道の番号" in reason


def test_a_picture_of_another_shape_is_questioned(sample_av: SampleMedia) -> None:
    """作品は覚えている大きさのまま描くので、縦の映像へ付け替えると横の枠へ引き伸ばされる"""
    item = probe_media(sample_av.path)
    video = tuple(
        replace(stream, width=stream.height, height=stream.width) for stream in item.video_streams
    )
    reason = differs(replace(item, video_streams=video), sample_av.path)
    assert reason is not None and "大きさか向き" in reason
    assert differs(item, sample_av.path) is None


def test_an_unreadable_file_is_refused_even_when_forced(
    window: MainWindow, moved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """読めないファイルを付け替えても絵も音も出ず、映らない理由が変わるだけ"""
    _open(window, moved, monkeypatch)
    broken = moved.parent / "壊れた.mp4"
    broken.write_bytes(b"not a movie")
    first = window.document.project.media[0]
    with pytest.raises(ToolError, match="付け替えられません"):
        call_tool(
            window,
            "relink_media",
            {"media_id": str(first.id), "path": str(broken), "force": True},
        )
    assert window._relink_from_dialog(first.id, broken).startswith("付け替えませんでした")
    assert window.document.project.media[0].path == first.path


def test_a_packed_media_can_be_relinked_from_the_menu(
    window: MainWindow, sample_av: SampleMedia, tmp_path: Path
) -> None:
    """範囲だけのファイルはあるので見つからない素材に入らない 入口が無いと範囲の外を使えない"""
    item = replace(probe_media(sample_av.path), packed=(PackedSpan(Fraction(0), Fraction(1, 2)),))
    window._adopt_opened(
        Project.create(media=(item,), name="範囲だけ"), tmp_path / "展開.sme", notice=False
    )
    window.show_missing_media()
    dialog = window._missing_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog._list.count() == 1
    assert "範囲だけ" in dialog._summary.text()


def test_an_old_list_does_not_stay_after_opening_another_project(
    window: MainWindow, moved: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """前の作品の一覧が残ると、選んでも今の作品に無い素材で何も起きない"""
    _open(window, moved, monkeypatch)
    assert window._missing_dialog is not None and window._missing_dialog.isVisible()
    window._adopt_opened(Project.create(name="空"), tmp_path / "空.sme")
    assert not window._missing_dialog.isVisible()
    assert window._missing_dialog._list.count() == 0


def test_undo_asks_again_for_the_moved_media(
    window: MainWindow, moved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """取り消しでも場所は変わる 捨てずに残すと、前のファイルの波形と縮小画像が出続ける"""
    _open(window, moved, monkeypatch)
    forgotten: list[object] = []
    monkeypatch.setattr(window._analyzer, "forget", forgotten.append)
    first = window.document.project.media[0]
    target = moved.parent.parent / "移した素材" / "本編.mp4"
    window._relink_from_dialog(first.id, target)
    assert first.id in forgotten
    forgotten.clear()
    window.undo()
    assert first.id in forgotten


def test_the_mixer_lets_go_of_the_old_file(
    sample_av: SampleMedia, sample_long: SampleMedia
) -> None:
    """付け替えは ID もクリップもそのまま 前のデコーダを持ち続けると前のファイルの音が鳴る"""
    item = probe_media(sample_av.path)
    clip = Clip(timeline_start=0, duration=30, media_id=item.id, source_in=Fraction(0))
    project = Project.create(media=(item,), name="音")
    track = Track(kind=TrackKind.AUDIO, name="音声 1", clips=(clip,))
    project = replace(project, timeline=replace(project.timeline, tracks=(track,)))
    mixer = AudioMixer(project)
    try:
        mixer.render(0, 4800)
        assert mixer._decoders
        mixer.set_project(RelinkMedia(((item.id, sample_long.path),)).apply(project))
        assert not mixer._decoders
    finally:
        mixer.close()
