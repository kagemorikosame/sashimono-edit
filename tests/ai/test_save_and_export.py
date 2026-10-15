"""AI の保存と書き出しの道具（B1 #327）

2026-10-14 に利用者が決めたこと（#298 #350 のコメント）に従っているかを、偽のホストで押さえる

- 上書き保存・名前を付けて保存・書き出しは取り消せない 「変更を自動で承認」でも毎回確かめる
- 書けるのは今の作品のフォルダと、書き出しの窓で前に選んだフォルダだけ
- 元の素材（とハードリンク）には書かない
- 確かめてから書くまでに場所を差し替えられても、確かめていない所へは書かない
- 書き出しは長いので裏で進め、AI は進み具合を読み、待つか戻るかを選び、止められる

本物の窓で保存し、本物の動画を書き出す試験は tests/ui/test_ai_save_export.py
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai import bridge as bridge_module
from sashimono.ai import safety
from sashimono.ai.bridge import Approval, EditorBridge
from sashimono.ai.host import ExportRequest, ToolError
from sashimono.ai.operations import MAX_WAIT_SECONDS, Operation, Pending, find_operation
from sashimono.ai.safety import Safety
from sashimono.core.commands import SetWorkArea
from sashimono.core.io.folder_pin import pin_folder
from sashimono.core.io.serialize import load_project
from sashimono.core.io.subtitles import subtitles_in_range, to_srt, to_vtt
from sashimono.core.model import MediaItem, Transcript
from sashimono.core.projection import project_timeline
from sashimono.engine.encode import RANGE_ALL, RANGE_WORK_AREA
from tests.ai.conftest import FakeHost, make_loaded
from tests.ai.test_bridge import call_in_thread
from tests.ai.test_safety import _hard_link, _symlink_dir, answer_approvals


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


SAVE = _operation("save_project")
SAVE_AS = _operation("save_project_as")
EXPORT = _operation("export_video")
STATUS = _operation("export_status")
CANCEL = _operation("cancel_export")


@pytest.fixture
def saved(host: FakeHost, tmp_path: Path) -> FakeHost:
    """作品を tmp_path/作品/作品.sme に保存してある"""
    folder = tmp_path / "作品"
    folder.mkdir()
    host.saved_at = folder / "作品.sme"
    host.write_project(host.saved_at, adopt=False)
    host.written.clear()
    return host


@pytest.fixture
def codecs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """使えるコーデックを決めておく 機械の GPU で確かめの文面が変わらないように"""
    names = ["h264_nvenc", "libx264"]
    from sashimono.ai import operations

    monkeypatch.setattr(operations, "available_video_codecs", lambda: list(names))
    return names


def _run(
    bridge: EditorBridge, operation: Operation, arguments: dict[str, Any]
) -> tuple[list[Approval], dict[str, object]]:
    """AI のスレッドのつもりで道具を呼び、出た確かめに許すで答える"""
    done, box = call_in_thread(bridge, operation, arguments)
    seen = answer_approvals(bridge, done, lambda _approval: True)
    return seen, box


# --- 区分 ---


class TestKinds:
    def test_save_save_as_and_export_are_irreversible(self) -> None:
        # 壊れると、保存や書き出しが「取り消せる編集」と扱われ、自動で承認の人の作品を
        # 黙って上書きする
        for operation in (SAVE, SAVE_AS, EXPORT):
            assert operation.safety is Safety.IRREVERSIBLE, operation.name

    def test_reading_the_progress_is_read_only_and_stopping_is_an_edit(self) -> None:
        # 壊れると、進み具合を読むたびに確かめが出て、書き出しを待つ間ずっと人が押し続ける
        assert STATUS.safety is Safety.READ_ONLY
        assert CANCEL.safety is Safety.UNDOABLE


# --- 上書き保存 ---


class TestSave:
    def test_auto_approve_still_asks_and_then_saves(self, saved: FakeHost) -> None:
        # 壊れると、「変更を自動で承認」を入れた人の作品を、AI が黙って上書き保存する
        assert saved.saved_at is not None
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SAVE, {})

        assert len(seen) == 1
        assert seen[0].irreversible
        assert seen[0].consent is not None
        assert seen[0].consent.target == str(saved.saved_at.resolve())
        assert "上書き保存" in seen[0].summary
        assert box.get("error") is None
        assert saved.written == [saved.saved_at.resolve()]
        assert load_project(saved.saved_at).timeline == saved.document.project.timeline

    def test_denying_keeps_the_file(self, saved: FakeHost) -> None:
        # 壊れると、拒否しても上書きされ、前の中身は取り消しでも戻らない
        assert saved.saved_at is not None
        before = saved.saved_at.read_bytes()
        bridge = EditorBridge(saved, auto_approve=True)
        done, box = call_in_thread(bridge, SAVE, {})
        answer_approvals(bridge, done, lambda _approval: False)
        assert isinstance(box.get("error"), ToolError)
        assert saved.written == []
        assert saved.saved_at.read_bytes() == before

    def test_an_unsaved_project_is_sent_to_save_as(self, host: FakeHost) -> None:
        # 壊れると、保存先の無い作品で窓だけが出て、許しても何も起きないか、決めていない所へ書く
        bridge = EditorBridge(host, auto_approve=True)
        seen, box = _run(bridge, SAVE, {})
        assert seen == []
        assert isinstance(box.get("error"), ToolError)
        assert "save_project_as" in str(box["error"])


# --- 名前を付けて保存 ---


class TestSaveAs:
    def test_a_new_name_in_the_project_folder_is_saved_and_adopted(self, saved: FakeHost) -> None:
        # 壊れると、AI が付けた名前で保存しても、以後の保存が前のファイルへ行く
        assert saved.saved_at is not None
        target = saved.saved_at.parent / "第二稿.sme"
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SAVE_AS, {"path": str(target)})

        assert len(seen) == 1 and seen[0].irreversible
        assert "以後はこの場所" in seen[0].summary
        assert box.get("error") is None
        assert saved.project_path == target.resolve()
        assert target.exists()

    def test_an_existing_file_is_named_in_the_request(self, saved: FakeHost) -> None:
        # 壊れると、同じ名前の作品を置き換えると知らないまま許す
        assert saved.saved_at is not None
        other = saved.saved_at.parent / "別の作品.sme"
        other.write_text("{}", encoding="utf-8")
        bridge = EditorBridge(saved)
        done, _box = call_in_thread(bridge, SAVE_AS, {"path": str(other)})
        seen = answer_approvals(bridge, done, lambda _approval: False)
        assert "置き換えます" in seen[0].summary
        assert other.read_text(encoding="utf-8") == "{}"

    def test_outside_the_allowed_places_is_refused_before_asking(
        self, saved: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、AI がデスクトップなど人の選んでいない所へ作品を書く
        outside = tmp_path / "外"
        outside.mkdir()
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SAVE_AS, {"path": str(outside / "a.sme")})
        assert seen == []
        assert isinstance(box.get("error"), ToolError)
        assert "ファイルの窓" in str(box["error"])
        assert not (outside / "a.sme").exists()

    @pytest.mark.parametrize("name", ["作品.mp4", "作品", "作品.sme.bat"])
    def test_a_name_that_is_not_a_project_is_refused(self, saved: FakeHost, name: str) -> None:
        # 壊れると、作品の中身を動画や実行できる名前で書き、素材や別のファイルを潰す
        assert saved.saved_at is not None
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SAVE_AS, {"path": str(saved.saved_at.parent / name)})
        assert seen == []
        assert isinstance(box.get("error"), ToolError)
        assert ".sme" in str(box["error"])


# --- 確かめた後の差し替え ---


def _can_link(tmp_path: Path, target: Path) -> None:
    """リンクを作れるかを先に試す 確かめに答える途中で飛ばすと、AI 役のスレッドが残る"""
    probe = tmp_path / "試し"
    _symlink_dir(probe, target)
    if probe.is_symlink():
        probe.unlink()
    else:
        probe.rmdir()


class TestSwappedPlaces:
    def test_a_folder_swapped_just_before_writing_is_refused(
        self, saved: FakeHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると（書く直前の確かめと書くまでの間に隙間があると）、その間にフォルダを
        # リンクへ差し替えられ、許した後に作品のフォルダの外へ書く（#354 の報告の隙間）
        assert saved.saved_at is not None
        inside = saved.saved_at.parent / "版"
        inside.mkdir()
        outside = tmp_path / "外"
        outside.mkdir()
        _can_link(tmp_path, outside)
        real_pin = pin_folder

        def swap_then_pin(path: Path) -> Any:
            # 書く直前の確かめが済んだ所で差し替える
            inside.rmdir()
            _symlink_dir(inside, outside)
            return real_pin(path)

        monkeypatch.setattr(safety, "pin_folder", swap_then_pin)
        bridge = EditorBridge(saved, auto_approve=True)
        _seen, box = _run(bridge, SAVE_AS, {"path": str(inside / "a.sme")})
        assert isinstance(box.get("error"), ToolError)
        assert not (outside / "a.sme").exists()
        assert saved.written == []

    def test_a_hard_link_planted_just_before_writing_is_refused(
        self,
        video_media: MediaItem,
        transcript: Transcript,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると（フォルダを持った後に確かめ直さないと）、書く直前に書く先の名前を元の素材への
        # ハードリンクにされたとき、その名前を作品で置き換える（素材の名前の 1 つが消える）
        original = tmp_path / "素材置き場" / "素材.sme"
        original.parent.mkdir()
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        folder = tmp_path / "作品"
        folder.mkdir()
        host.saved_at = folder / "作品.sme"
        target = folder / "第二稿.sme"
        _hard_link(tmp_path / "試し", original)
        real_pin = pin_folder

        def link_then_pin(path: Path) -> Any:
            _hard_link(target, original)
            return real_pin(path)

        monkeypatch.setattr(safety, "pin_folder", link_then_pin)
        bridge = EditorBridge(host, auto_approve=True)
        _seen, box = _run(bridge, SAVE_AS, {"path": str(target)})
        assert "元の素材" in str(box.get("error"))
        assert target.read_bytes() == b"original"
        assert host.written == []

    def test_the_folder_cannot_be_swapped_while_writing(
        self, saved: FakeHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、書いている最中にフォルダをリンクへ差し替えられ、外へ書く
        # Windows では持っている間は差し替えそのものが断られる ほかでは書いた後に気付いて断る
        assert saved.saved_at is not None
        inside = saved.saved_at.parent / "版"
        inside.mkdir()
        outside = tmp_path / "外"
        outside.mkdir()
        _can_link(tmp_path, outside)
        refused: list[str] = []
        real_write = saved.write_project

        def swap_then_write(path: Path, *, adopt: bool) -> str:
            try:
                inside.rmdir()
            except OSError as exc:
                refused.append(str(exc))
            else:
                _symlink_dir(inside, outside)
            return real_write(path, adopt=adopt)

        monkeypatch.setattr(saved, "write_project", swap_then_write)
        bridge = EditorBridge(saved, auto_approve=True)
        _seen, box = _run(bridge, SAVE_AS, {"path": str(inside / "a.sme")})
        if sys.platform == "win32":
            assert refused, "書いている最中にフォルダを消せた（持っていない）"
            assert box.get("error") is None
            assert (inside / "a.sme").exists() and not inside.is_symlink()
            assert not (outside / "a.sme").exists()
        else:
            # 差し替えは止められないので、書いた後に照らして断り、AI へ書けなかったと伝える
            assert isinstance(box.get("error"), ToolError)

    def test_a_hard_link_to_an_original_is_never_saved_over(
        self, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、作品のフォルダに置いた元の素材へのハードリンクへ作品を書き、素材を壊す
        original = tmp_path / "素材置き場" / "素材.sme"
        original.parent.mkdir()
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        folder = tmp_path / "作品"
        folder.mkdir()
        host.saved_at = folder / "作品.sme"
        link = folder / "別名.sme"
        _hard_link(link, original)
        bridge = EditorBridge(host, auto_approve=True)
        seen, box = _run(bridge, SAVE_AS, {"path": str(link)})
        assert seen == []
        assert "元の素材" in str(box.get("error"))
        assert original.read_bytes() == b"original"


# --- 書き出し ---


class TestExport:
    def test_auto_approve_still_asks_and_says_what_is_exported(
        self, saved: FakeHost, codecs: list[str]
    ) -> None:
        # 壊れると、何をどの画質で書くのか分からないまま、または確かめずに書き出しが走る
        del codecs
        assert saved.saved_at is not None
        target = saved.saved_at.parent / "完成.mp4"
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(
            bridge, EXPORT, {"path": str(target), "codec": "libx264", "bitrate_mbps": 8}
        )
        assert len(seen) == 1 and seen[0].irreversible
        summary = seen[0].summary
        assert "動画を書き出します" in summary
        assert str(target.resolve()) in summary
        assert "メイン の全体" in summary and "300 フレーム" in summary
        assert "libx264" in summary and "8 Mbps" in summary and "mp4" in summary
        assert box.get("error") is None
        assert saved.exports == [
            ExportRequest(
                path=target.resolve(),
                scene_id=None,
                range_choice=RANGE_ALL,
                codec="libx264",
                bitrate_mbps=8,
            )
        ]

    def test_the_range_follows_the_export_window(self, saved: FakeHost, codecs: list[str]) -> None:
        # 壊れると、範囲を決めた人の書き出しが AI からだけ全体になり、長く待たされる
        del codecs
        assert saved.saved_at is not None
        saved.apply_commands([SetWorkArea((30, 90))], "範囲")
        bridge = EditorBridge(saved, auto_approve=True)
        seen, _box = _run(bridge, EXPORT, {"path": str(saved.saved_at.parent / "a.mp4")})
        assert saved.exports[-1].range_choice == RANGE_WORK_AREA
        assert "指定した範囲" in seen[0].summary and "60 フレーム" in seen[0].summary
        _run(bridge, EXPORT, {"path": str(saved.saved_at.parent / "b.mp4"), "range": RANGE_ALL})
        # 1 本目が走っている（偽物は終わらない）ので 2 本目は断られる
        assert len(saved.exports) == 1

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"path": "作品.sme"}, ".mp4"),
            ({"path": "a.exe"}, ".mp4"),
            ({"path": "a.mp4", "range": RANGE_WORK_AREA}, "範囲"),
            ({"path": "a.mp4", "scene_id": "無い"}, "シーン"),
            ({"path": "a.mp4", "codec": "prores"}, "codec"),
            ({"path": "a.mp4", "bitrate_mbps": 0}, "bitrate_mbps"),
            ({"path": "a.mp4", "bitrate_mbps": "12"}, "bitrate_mbps"),
            ({"path": "a.mp4", "wait_seconds": "長く"}, "wait_seconds"),
        ],
    )
    def test_wrong_choices_are_refused_before_asking(
        self, saved: FakeHost, codecs: list[str], arguments: dict[str, Any], message: str
    ) -> None:
        # 壊れると、書けない頼みで確かめが出て人が許しても失敗するか、作品のファイルや
        # 実行できる名前で動画を書く
        del codecs
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, EXPORT, arguments)
        assert seen == []
        assert isinstance(box.get("error"), ToolError)
        assert message in str(box["error"])
        assert saved.exports == []

    def test_an_original_material_is_never_the_output(
        self, video_media: MediaItem, transcript: Transcript, tmp_path: Path, codecs: list[str]
    ) -> None:
        # 壊れると、書き出しの出力が元の素材（やそのハードリンク）を置き換え、素材を失う
        del codecs
        original = tmp_path / "作品" / "素材.mp4"
        original.parent.mkdir()
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        host.saved_at = original.parent / "作品.sme"
        link = original.parent / "別名.mp4"
        _hard_link(link, original)
        bridge = EditorBridge(host, auto_approve=True)
        for target in (original, link):
            seen, box = _run(bridge, EXPORT, {"path": str(target)})
            assert seen == []
            assert "元の素材" in str(box.get("error"))
        assert host.exports == []

    def test_an_empty_timeline_is_refused(self, saved: FakeHost, codecs: list[str]) -> None:
        # 壊れると、何も映らない動画を書くために確かめが出る
        del codecs
        assert saved.saved_at is not None
        clip = saved.project.timeline.tracks[0].clips[0]
        _operation("delete_clip")(saved, {"clip_id": str(clip.id)})
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, EXPORT, {"path": "a.mp4"})
        assert seen == []
        assert "空" in str(box.get("error"))


# --- 確かめの箱が出ている間に作品の場所が変わる（Codex の P1） ---


class TestTheApprovedPlaceIsKept:
    """確かめの箱はモードレス 出ている間に人が〔名前を付けて保存〕で作品の場所を変えても、
    許した後に書くのは箱に出した場所だけ 解き直して違えば断る
    """

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("save_project", {}),
            ("save_project_as", {"path": "第二稿.sme"}),
            ("export_video", {"path": "完成.mp4"}),
            ("save_snapshot", {"path": "表紙.png"}),
            ("export_subtitles", {"path": "字幕.srt"}),
        ],
    )
    def test_moving_the_project_while_asking_is_refused(
        self,
        saved: FakeHost,
        codecs: list[str],
        tmp_path: Path,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        # 壊れると、人が箱に出た場所を見て許したのに、移した先の作品や別のフォルダの
        # ファイルを上書きする（上書き保存なら、移した先の作品を前の中身で潰す）
        del codecs
        assert saved.saved_at is not None
        folder = saved.saved_at.parent
        moved_folder = tmp_path / "移した先"
        moved_folder.mkdir()
        moved = moved_folder / "作品.sme"
        moved.write_text("移した先の作品", encoding="utf-8")
        bridge = EditorBridge(saved, auto_approve=True)
        done, box = call_in_thread(bridge, _operation(name), arguments)

        def move_then_allow(approval: Approval) -> bool:
            assert approval.consent is not None
            assert str(folder.resolve()) in approval.consent.target
            # 箱が出ている間に、人が〔名前を付けて保存〕で別の場所へ移した
            saved.saved_at = moved
            return True

        seen = answer_approvals(bridge, done, move_then_allow)
        assert len(seen) == 1
        assert isinstance(box.get("error"), ToolError)
        assert "変わりました" in str(box["error"])
        assert moved.read_text(encoding="utf-8") == "移した先の作品"
        assert sorted(path.name for path in moved_folder.iterdir()) == ["作品.sme"]
        assert saved.written == [] and saved.exports == [] and saved.snapshots == []


# --- 字幕の書き出しと静止画 ---

SUBTITLES = _operation("export_subtitles")
SNAPSHOT = _operation("save_snapshot")


class TestSubtitles:
    def test_auto_approve_still_asks_and_the_same_srt_as_the_panel_is_written(
        self, saved: FakeHost
    ) -> None:
        # 壊れると、自動で承認の人の字幕のファイルを AI が黙って上書きするか、画面の
        # 〔書き出し…〕と違う形式の字幕ができる
        assert saved.saved_at is not None
        target = saved.saved_at.parent / "字幕.srt"
        target.write_text("前の字幕", encoding="utf-8")
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SUBTITLES, {"path": str(target)})

        assert len(seen) == 1 and seen[0].irreversible
        summary = seen[0].summary
        assert "字幕を書き出します" in summary
        assert "srt" in summary and "全体" in summary and "3 枚" in summary
        assert "置き換えます" in summary
        assert box.get("error") is None
        expected = to_srt(list(project_timeline(saved.project)), saved.project.rate)
        assert target.read_text(encoding="utf-8") == expected

    def test_the_range_is_shifted_like_the_panel(self, saved: FakeHost) -> None:
        # 壊れると、範囲で書き出した動画と AI が書いた字幕が、範囲の頭の分だけずれる
        assert saved.saved_at is not None
        saved.apply_commands([SetWorkArea((90, 300))], "範囲")
        target = saved.saved_at.parent / "字幕.vtt"
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SUBTITLES, {"path": str(target)})
        assert "指定した範囲" in seen[0].summary
        assert box.get("result") == {"written": str(target.resolve()), "range": [90, 300]}
        subtitles = list(subtitles_in_range(project_timeline(saved.project), (90, 300)))
        assert target.read_text(encoding="utf-8") == to_vtt(subtitles, saved.project.rate)

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"path": "字幕.sme"}, ".srt"),
            ({"path": "字幕.mp4"}, ".srt"),
            ({"path": "字幕.srt", "range": RANGE_WORK_AREA}, "範囲"),
        ],
    )
    def test_wrong_choices_are_refused_before_asking(
        self, saved: FakeHost, arguments: dict[str, Any], message: str
    ) -> None:
        # 壊れると、作品や素材の名前で字幕の文字を書くか、人が許しても書けない確かめが出る
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SUBTITLES, arguments)
        assert seen == []
        assert message in str(box.get("error"))
        assert saved.written == []

    def test_outside_and_originals_are_refused(
        self, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、AI が人の選んでいない場所や、元の素材（のハードリンク）へ字幕を書く
        original = tmp_path / "作品" / "素材.srt"
        original.parent.mkdir()
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        host.saved_at = original.parent / "作品.sme"
        link = original.parent / "別名.srt"
        _hard_link(link, original)
        outside = tmp_path / "外"
        outside.mkdir()
        bridge = EditorBridge(host, auto_approve=True)
        for target, message in (
            (original, "元の素材"),
            (link, "元の素材"),
            (outside / "a.srt", "書ける場所の外"),
        ):
            seen, box = _run(bridge, SUBTITLES, {"path": str(target)})
            assert seen == []
            assert message in str(box.get("error"))
        assert original.read_bytes() == b"original"
        assert list(outside.iterdir()) == []

    def test_nothing_to_write_is_refused(self, saved: FakeHost) -> None:
        # 壊れると、空の字幕のファイルを書くために取り消せない確かめが出る
        clip = saved.project.timeline.tracks[0].clips[0]
        _operation("delete_clip")(saved, {"clip_id": str(clip.id)})
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SUBTITLES, {"path": "字幕.srt"})
        assert seen == []
        assert "字幕がありません" in str(box.get("error"))

    def test_a_folder_swapped_just_before_writing_is_refused(
        self, saved: FakeHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、確かめてから書くまでの間にフォルダをリンクへ差し替えられ、外へ字幕を書く
        assert saved.saved_at is not None
        inside = saved.saved_at.parent / "字幕"
        inside.mkdir()
        outside = tmp_path / "外"
        outside.mkdir()
        _can_link(tmp_path, outside)

        def swap_then_pin(path: Path) -> Any:
            inside.rmdir()
            _symlink_dir(inside, outside)
            return pin_folder(path)

        monkeypatch.setattr(safety, "pin_folder", swap_then_pin)
        bridge = EditorBridge(saved, auto_approve=True)
        _seen, box = _run(bridge, SUBTITLES, {"path": str(inside / "a.srt")})
        assert isinstance(box.get("error"), ToolError)
        assert list(outside.iterdir()) == []


class TestSnapshot:
    def test_auto_approve_still_asks_and_the_frame_is_saved(self, saved: FakeHost) -> None:
        # 壊れると、自動で承認の人の絵を AI が黙って上書きするか、頼んだのと違うコマを描く
        assert saved.saved_at is not None
        target = saved.saved_at.parent / "表紙.png"
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SNAPSHOT, {"path": str(target), "frame": 45})
        assert len(seen) == 1 and seen[0].irreversible
        assert "静止画を保存します" in seen[0].summary and "45 フレーム" in seen[0].summary
        assert box.get("error") is None
        assert saved.snapshots == [45]
        assert target.exists()

    def test_the_end_draws_the_last_frame_like_the_screen(self, saved: FakeHost) -> None:
        # 壊れると、最後まで再生した所で撮ると何も無い黒の絵になる（画面と食い違う）
        assert saved.saved_at is not None
        saved.frame = saved.project.duration
        bridge = EditorBridge(saved, auto_approve=True)
        _run(bridge, SNAPSHOT, {"path": str(saved.saved_at.parent / "終わり.png")})
        assert saved.snapshots == [saved.project.duration - 1]

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"path": "絵.jpg"}, ".png"),
            ({"path": "作品.sme"}, ".png"),
            ({"path": "絵.png", "frame": "10"}, "frame"),
        ],
    )
    def test_wrong_choices_are_refused_before_asking(
        self, saved: FakeHost, arguments: dict[str, Any], message: str
    ) -> None:
        # 壊れると、作品や素材の名前で絵を書くか、人が許しても書けない確かめが出る
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(bridge, SNAPSHOT, arguments)
        assert seen == []
        assert message in str(box.get("error"))
        assert saved.written == []

    def test_outside_and_originals_are_refused(
        self, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、AI が人の選んでいない場所や、元の素材（のハードリンク）へ絵を書く
        original = tmp_path / "作品" / "素材.png"
        original.parent.mkdir()
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        host.saved_at = original.parent / "作品.sme"
        link = original.parent / "別名.png"
        _hard_link(link, original)
        outside = tmp_path / "外"
        outside.mkdir()
        bridge = EditorBridge(host, auto_approve=True)
        for target, message in (
            (original, "元の素材"),
            (link, "元の素材"),
            (outside / "a.png", "書ける場所の外"),
        ):
            seen, box = _run(bridge, SNAPSHOT, {"path": str(target)})
            assert seen == []
            assert message in str(box.get("error"))
        assert original.read_bytes() == b"original"
        assert host.snapshots == []

    def test_a_hard_link_planted_just_before_writing_is_refused(
        self,
        video_media: MediaItem,
        transcript: Transcript,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 壊れると、書く直前に書く先の名前を元の素材へのハードリンクにされ、その名前を絵で置き換える
        original = tmp_path / "素材置き場" / "素材.png"
        original.parent.mkdir()
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        folder = tmp_path / "作品"
        folder.mkdir()
        host.saved_at = folder / "作品.sme"
        target = folder / "表紙.png"
        _hard_link(tmp_path / "試し", original)

        def link_then_pin(path: Path) -> Any:
            _hard_link(target, original)
            return pin_folder(path)

        monkeypatch.setattr(safety, "pin_folder", link_then_pin)
        bridge = EditorBridge(host, auto_approve=True)
        _seen, box = _run(bridge, SNAPSHOT, {"path": str(target)})
        assert "元の素材" in str(box.get("error"))
        assert target.read_bytes() == b"original"
        assert host.snapshots == []


# --- 長い書き出しを待つ・読む・止める ---


def _finish_later(host: FakeHost, delay: float) -> threading.Timer:
    """走り出した書き出しを ``delay`` 秒後に終わらせる（書き出しのふり）"""

    def finish() -> None:
        while host.export_state["state"] != "rendering":
            time.sleep(0.005)
        host.export_state = {**host.export_state, "state": "done", "progress": 1.0}

    timer = threading.Timer(delay, finish)
    timer.daemon = True
    timer.start()
    return timer


@pytest.fixture
def quick_waits(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(bridge_module, "WAIT_SLICE", 0.01)
    yield


class TestWaiting:
    def test_the_ai_can_wait_for_the_end_without_holding_the_editor(
        self, saved: FakeHost, codecs: list[str], quick_waits: None
    ) -> None:
        # 壊れると、AI が書き出しの終わりを知る手段が無いか、待つ間に編集画面が固まる
        # （待つのは AI のスレッドで、編集画面のスレッドは様子を読むときだけ使う）
        del codecs, quick_waits
        assert saved.saved_at is not None
        timer = _finish_later(saved, 0.2)
        bridge = EditorBridge(saved, auto_approve=True)
        seen, box = _run(
            bridge,
            EXPORT,
            {"path": str(saved.saved_at.parent / "a.mp4"), "wait_seconds": 30},
        )
        timer.join(5)
        assert len(seen) == 1
        result = box.get("result")
        assert isinstance(result, dict)
        assert result["state"] == "done"

    def test_waiting_gives_up_at_the_limit_and_returns_the_progress(
        self, saved: FakeHost, quick_waits: None
    ) -> None:
        # 壊れると、終わらない書き出しを待つ AI が、人の新しい指示に応えないまま止まる
        del quick_waits
        saved.export_state = {"state": "rendering", "progress": 0.25, "path": "a.mp4"}
        bridge = EditorBridge(saved)
        started = time.monotonic()
        _seen, box = _run(bridge, STATUS, {"wait_seconds": 0.2})
        assert time.monotonic() - started < 5
        result = box.get("result")
        assert isinstance(result, dict)
        assert result["state"] == "rendering" and result["progress"] == 0.25

    def test_the_wait_is_capped(self, saved: FakeHost) -> None:
        # 壊れると、AI が何時間も待つように頼んで、その間は人に応えない
        saved.export_state = {"state": "rendering", "progress": 0.0, "path": "a.mp4"}
        result = STATUS(saved, {"wait_seconds": 10_000})
        assert isinstance(result, Pending)
        assert result.seconds == MAX_WAIT_SECONDS

    def test_interrupting_stops_the_wait(self, saved: FakeHost) -> None:
        # 壊れると、中断を押しても AI のスレッドが書き出しの終わりを待ち続ける
        saved.export_state = {"state": "rendering", "progress": 0.0, "path": "a.mp4"}
        bridge = EditorBridge(saved)
        done, box = call_in_thread(bridge, STATUS, {"wait_seconds": 60})
        started = time.monotonic()
        while not done.is_set() and time.monotonic() < started + 5:
            bridge.pump()
            if time.monotonic() > started + 0.1:
                bridge.cancel()
            time.sleep(0.005)
        assert done.is_set()
        assert isinstance(box.get("error"), ToolError)
        assert "中断" in str(box["error"])
        # 待つのをやめただけで、書き出しは止めない（止めるのは cancel_export か画面の〔中止〕）
        assert saved.export_state["state"] == "rendering"

    def test_cancel_stops_a_running_export_and_says_when_none_runs(self, saved: FakeHost) -> None:
        # 壊れると、AI が始めた書き出しを AI が止められず、要らない書き出しが最後まで走る
        with pytest.raises(ToolError, match="走っている書き出しはありません"):
            CANCEL(saved, {})
        saved.export_state = {"state": "rendering", "progress": 0.5, "path": "a.mp4"}
        result = CANCEL(saved, {})
        assert isinstance(result, dict)
        assert saved.export_state["state"] == "cancelled"

    def test_a_stop_while_placing_says_the_file_will_be_replaced(self, saved: FakeHost) -> None:
        # 壊れると、置き始めた後の中止を「走っていない」と答え、AI は前のファイルが残ると思う
        saved.export_state = {"state": "placing", "progress": 1.0, "path": "a.mp4"}
        with pytest.raises(ToolError, match="置き換わります"):
            CANCEL(saved, {})
        assert saved.export_state["state"] == "placing"

    def test_reading_the_status_returns_a_copy(self, saved: FakeHost) -> None:
        # 壊れると、読んだ様子を書き換えただけで、画面の進み具合の表示が狂う
        result = STATUS(saved, {})
        assert isinstance(result, dict)
        result["state"] = "done"
        assert saved.export_state["state"] == "idle"


def test_waiting_runs_on_the_ai_thread(
    saved: FakeHost, quick_waits: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 壊れると、待つのを編集画面のスレッドで行い、書き出しの数分のあいだ画面が固まる
    del quick_waits
    saved.export_state = {"state": "rendering", "progress": 0.0, "path": "a.mp4"}
    bridge = EditorBridge(saved)
    pumped_on: set[int] = set()
    real = saved.export_status

    def recording() -> dict[str, object]:
        pumped_on.add(threading.get_ident())
        return real()

    monkeypatch.setattr(saved, "export_status", recording)
    done, _box = call_in_thread(bridge, STATUS, {"wait_seconds": 0.1})
    deadline = time.monotonic() + 5
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        time.sleep(0.005)
    assert done.is_set()
    # 様子を読むのは画面のスレッドだけ 待つのはそれ以外（AI）のスレッド
    assert pumped_on == {threading.get_ident()}
    assert saved.export_waits
    assert threading.get_ident() not in saved.export_waits
