"""AI の道具の区分と、取り消せない操作の確かめ（#350）

2026-10-14 に利用者が決めた 3 つ（#298 #350 のコメント）を押さえる

1. 確かめは二段 取り消せる編集は「変更を自動で承認」で省けるが、取り消せない操作は毎回確かめる
2. AI が書けるのは、作品のフォルダと書き出しの窓で前に選んだフォルダだけ
3. アプリ自身の設定は、AI は読むだけ

土台そのものを、試験の中の最小の道具（メモを書き出す）で通す 確かめの窓は画面を作らず、
ブリッジから確かめを取り出して試験が答える（人の代わり） 本物の取り消せない道具
（保存・書き出し B1 #327）は test_save_and_export.py と tests/ui/test_ai_save_export.py
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai import operations
from sashimono.ai.bridge import Approval, EditorBridge
from sashimono.ai.host import EditorHost, ToolError
from sashimono.ai.operations import OPERATIONS, Operation, find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Consent, Safety, check_writable, writable_path
from sashimono.core.model import MediaItem, Transcript
from tests.ai.conftest import FakeHost, allow_sending, make_loaded
from tests.ai.test_bridge import call_in_thread
from tests.ai.test_outbound import undeclared

# --- 試験のための最小の取り消せない道具 ---


def _note_consent(host: EditorHost, arguments: dict[str, Any]) -> Consent:
    path = writable_path(host, arguments.get("path"))
    detail = "同じ名前のファイルがあり、中身を置き換えます" if path.exists() else ""
    return Consent(action="メモを書き出します", target=str(path), detail=detail)


def _write_note(host: EditorHost, arguments: dict[str, Any]) -> object:
    # 書く直前にもう 1 度確かめる 窓が出ている間に場所が差し替えられても、ここで断る
    path = writable_path(host, arguments.get("path"))
    path.write_text(str(arguments.get("text", "")), encoding="utf-8")
    return {"written": str(path)}


WRITE_NOTE = Operation(
    name="write_note",
    description="メモをファイルへ書き出す（試験用）",
    schema={"type": "object", "properties": {}},
    handler=_write_note,
    writes=True,
    irreversible=True,
    consent=_note_consent,
    # 書いた場所を返す 作品のフォルダの場所なので名前と場所を送る物に書く（CodeRabbit の指摘 #366）
    sends=frozenset({Outbound.PROJECT, Outbound.FILES}),
)


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


# --- 人の代わりに確かめへ答える ---


def answer_approvals(
    bridge: EditorBridge,
    done: threading.Event,
    reply: Callable[[Approval], bool],
    timeout: float = 5.0,
) -> list[Approval]:
    """画面のスレッドの代わりに、仕事を拾い、出てきた確かめに ``reply`` で答える

    出てきた確かめをすべて返す 窓が出なかったことも、出た中身も、試験から見える
    """
    seen: list[Approval] = []
    deadline = time.monotonic() + timeout
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        approval = bridge.take_approval()
        if approval is not None:
            seen.append(approval)
            if reply(approval):
                approval.allow()
            else:
                approval.deny()
        time.sleep(0.005)
    bridge.pump()
    assert done.is_set(), "道具が終わらなかった（確かめを待ったまま止まっている）"
    return seen


@pytest.fixture
def saved(host: FakeHost, tmp_path: Path) -> FakeHost:
    """作品を tmp_path/作品/作品.sme に保存してある"""
    folder = tmp_path / "作品"
    folder.mkdir()
    host.saved_at = folder / "作品.sme"
    host.saved_at.write_text("{}", encoding="utf-8")
    return host


# --- 区分 ---


class TestKinds:
    def test_every_tool_falls_into_one_of_the_three_kinds(self) -> None:
        # 壊れると、プロジェクトを変える道具が「読むだけ」と扱われ、確かめずに走る
        for operation in OPERATIONS:
            assert operation.safety in set(Safety)
            assert (operation.safety is Safety.READ_ONLY) is (not operation.writes), operation.name

    def test_the_kind_is_a_plain_word_the_ledger_can_hold(self) -> None:
        # 壊れると、操作の台帳（#347）に載せた区分と道具の区分が名前で突き合わせられない
        assert {kind.value for kind in Safety} == {"read_only", "undoable", "irreversible"}
        assert Safety.IRREVERSIBLE.label == "取り消せない操作"
        assert WRITE_NOTE.safety is Safety.IRREVERSIBLE

    @pytest.mark.parametrize(
        "options",
        [
            {"writes": True},
            {"writes": False, "consent": _note_consent},
        ],
    )
    def test_an_irreversible_tool_must_say_what_it_does(self, options: dict[str, Any]) -> None:
        # 壊れると、何をどこへするかの無い空の確かめが出て、人が判断できないまま許す
        with pytest.raises(ValueError, match="取り消せない"):
            Operation(
                name="broken",
                description="",
                schema={},
                handler=_write_note,
                irreversible=True,
                sends=frozenset({Outbound.PROJECT}),
                **options,
            )


# --- アプリ自身の設定は読むだけ ---


class TestAppSettingsAreReadOnly:
    def test_settings_can_be_read(self, host: FakeHost) -> None:
        # 壊れると、AI がショートカットや好みを知らずに、人の操作の癖と違う案内をする
        operation = _operation("get_app_settings")
        assert operation.safety is Safety.READ_ONLY
        result = operation(host, {})
        assert result == host.settings

    def test_changing_what_was_read_does_not_change_the_settings(self, host: FakeHost) -> None:
        # 壊れると、読んだ値を書き換えただけで、本人の設定やショートカットが変わる
        result = _operation("get_app_settings")(host, {})
        assert isinstance(result, dict)
        preferences = result["preferences"]
        assert isinstance(preferences, dict)
        preferences["update_check"] = False
        result["shortcuts"] = {}
        assert host.settings["preferences"] == {"theme": "dark", "update_check": True}
        assert host.settings["shortcuts"] == {"編集/分割": "S"}

    def test_the_host_offers_no_way_to_change_app_settings(self) -> None:
        # 壊れると（設定・更新・画面配置・ショートカットを変える口を足すと）、AI が人の
        # 知らない間にアプリの動きを変えられる 足すなら利用者の決定（読むだけ）を見直してから
        changes = {
            name
            for name, member in inspect.getmembers(EditorHost)
            if not name.startswith("_") and callable(member)
        }
        # 保存・書き出し・字幕の書き出し・静止画（B1 #327）は作品を書く口で、
        # アプリ自身の設定は変えない
        assert changes == {
            "analyze",
            "app_settings",
            "apply_commands",
            "cancel_export",
            # 完成物の検査（#337） 調べるのは作品の写しで、アプリの設定は変えない
            # 音量の基準（Preferences.check_loudness_target）は読むだけ（loudness_target）
            "check_state",
            "check_waiter",
            "export_status",
            "export_waiter",
            "probe",
            # 取り消しとやり直し（#333） 人のメニューと同じく作品の履歴を辿るだけ
            "redo",
            # 作品の外で覚えている名前を読むだけ（#342 伏せ字の表に入れる） 何も変えない
            "remembered_names",
            "render_png",
            "request_check",
            # 場面の解析（#319 #321） 頼むのは analyze と同じく本人のキャッシュへの解析だけで、
            # 取り込み時に調べるかの設定（Preferences.scene_analysis）は変えない
            "request_scenes",
            "scene_state",
            "scene_waiter",
            "seek",
            "select_clip",
            "select_clips",
            "set_active_scene",
            "start_export",
            # コマを並べた絵を描くだけ 作品にもアプリの設定にも触らない（#320）
            "start_sheet",
            "start_transcription",
            "stop_playback",
            "transcription_status",
            "undo",
            "waveform",
            # 色を測るサムネイルを読むだけ（#336） 作れていなければ None を返し、作るのは analyze
            "filmstrip",
            "write_project",
            "write_snapshot",
            "write_subtitles",
        }


# --- 書ける場所 ---


def _symlink_dir(link: Path, target: Path) -> None:
    """フォルダへのリンクを作る Windows で権限が無ければジャンクションにする

    どちらも作れない所（リンクを禁じたファイルシステムなど）では試験を飛ばす 抜け道の
    確かめは作れる所で走らせる ジャンクションは権限なしで作れるので、CI の Windows では走る
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as exc:
        if sys.platform != "win32":
            pytest.skip(f"フォルダへのリンクを作れない: {exc}")
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"フォルダへのリンクもジャンクションも作れない: {exc}")


def _hard_link(link: Path, target: Path) -> None:
    """ハードリンクを作る 作れない所（FAT など）では試験を飛ばす"""
    try:
        os.link(target, link)
    except OSError as exc:
        pytest.skip(f"ハードリンクを作れない: {exc}")


class TestWritablePlaces:
    def test_the_project_folder_is_writable(self, saved: FakeHost) -> None:
        # 壊れると、作品のフォルダへの保存・書き出しまで断られ、AI が完成まで進めない
        assert saved.saved_at is not None
        folder = saved.saved_at.parent
        assert writable_path(saved, str(folder / "出力.mp4")) == (folder / "出力.mp4").resolve()

    def test_a_relative_place_is_read_from_the_project_folder(self, saved: FakeHost) -> None:
        # 壊れると、相対の場所が起動したときの作業フォルダから読まれ、思わぬ所へ書く
        assert saved.saved_at is not None
        expected = (saved.saved_at.parent / "出力.mp4").resolve()
        assert writable_path(saved, "出力.mp4") == expected

    def test_the_folder_chosen_in_the_export_window_is_writable(
        self, host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、人が書き出した所へ AI が書き直せない（利用者の決定の 2 つ目）
        host.exported_to = tmp_path
        assert writable_path(host, str(tmp_path / "完成.mp4")) == (tmp_path / "完成.mp4").resolve()

    def test_anywhere_else_is_refused(self, saved: FakeHost, tmp_path: Path) -> None:
        # 壊れると、AI がデスクトップやシステムのフォルダへ好きに書ける
        outside = tmp_path / "外"
        outside.mkdir()
        with pytest.raises(ToolError, match="書ける場所の外") as caught:
            writable_path(saved, str(outside / "a.mp4"))
        # 断るときは、書ける場所と人に選んでもらう道を示す AI はこの文面で次の手を決める
        assert "ファイルの窓" in str(caught.value)

    def test_a_folder_with_the_same_head_is_not_inside(self, saved: FakeHost) -> None:
        # 壊れると（文字の頭だけで比べると）、「作品2」のような隣のフォルダへ書ける
        assert saved.saved_at is not None
        neighbor = saved.saved_at.parent.with_name("作品2")
        neighbor.mkdir()
        with pytest.raises(ToolError, match="書ける場所の外"):
            writable_path(saved, str(neighbor / "a.mp4"))

    def test_going_up_with_dot_dot_is_refused(self, saved: FakeHost) -> None:
        # 壊れると、「..」で作品のフォルダの外へ抜けられる 中へ戻る道でも、窓に出した場所と
        # AI の思う場所が食い違うので断る
        assert saved.saved_at is not None
        folder = saved.saved_at.parent
        with pytest.raises(ToolError, match=r"\.\."):
            writable_path(saved, str(folder / ".." / "外.mp4"))
        with pytest.raises(ToolError, match=r"\.\."):
            writable_path(saved, str(folder / ".." / folder.name / "中.mp4"))

    def test_a_link_out_of_the_project_folder_is_refused(
        self, saved: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、作品のフォルダに置いたリンク（ジャンクション）を通って外のファイルを上書きする
        assert saved.saved_at is not None
        outside = tmp_path / "外"
        outside.mkdir()
        link = saved.saved_at.parent / "近道"
        _symlink_dir(link, outside)
        with pytest.raises(ToolError, match="書ける場所の外"):
            writable_path(saved, str(link / "a.mp4"))

    def test_an_original_material_inside_the_folder_is_refused(
        self, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、作品のフォルダに置いた元の素材を上書きし、取り消しても戻らない（#341）
        original = tmp_path / "素材.mp4"
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        host.saved_at = tmp_path / "作品.sme"
        with pytest.raises(ToolError, match="元の素材"):
            writable_path(host, str(original))
        # 相対の場所で渡しても同じ素材と見なす
        with pytest.raises(ToolError, match="元の素材"):
            writable_path(host, "素材.mp4")
        assert original.read_bytes() == b"original"

    def test_a_hard_link_to_an_original_is_refused(
        self, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、作品のフォルダに置いた元の素材へのハードリンクへ書き、元の素材の中身を
        # 上書きする（場所の文字は別なので、文字だけ比べると気付けない）
        original = tmp_path / "素材置き場" / "素材.mp4"
        original.parent.mkdir()
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        folder = tmp_path / "作品"
        folder.mkdir()
        host.saved_at = folder / "作品.sme"
        link = folder / "別名.mp4"
        _hard_link(link, original)
        with pytest.raises(ToolError, match="元の素材"):
            writable_path(host, str(link))
        assert original.read_bytes() == b"original"

    def test_the_check_works_without_a_host(self, tmp_path: Path) -> None:
        # 壊れると、ホストを持たない所（保存・書き出しの道具の下ごしらえ）から同じ判定を使えない
        assert (
            check_writable(str(tmp_path / "a.mp4"), project_path=None, export_folder=tmp_path)
            == (tmp_path / "a.mp4").resolve()
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="大文字小文字を区別しないのは Windows")
    def test_a_different_letter_case_is_still_inside(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、Windows で同じフォルダを大文字で書いただけで断られる
        folder = tmp_path / "Export"
        folder.mkdir()
        host.exported_to = folder
        assert writable_path(host, str(tmp_path / "EXPORT" / "a.mp4")).name == "a.mp4"

    def test_a_data_stream_name_is_refused(self, saved: FakeHost) -> None:
        # 壊れると、名前:流れ の形で、見た目は変わらないファイルへ別の中身を足せる
        assert saved.saved_at is not None
        with pytest.raises(ToolError, match=":"):
            writable_path(saved, str(saved.saved_at.parent / "a.mp4:隠し"))

    def test_without_a_saved_project_or_export_folder_nothing_is_writable(
        self, host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、保存していない作品のとき、どこにでも書けてしまう
        with pytest.raises(ToolError, match="書ける場所がありません"):
            writable_path(host, str(tmp_path / "a.mp4"))

    def test_a_relative_place_needs_a_saved_project(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、保存していない作品で相対の場所が作業フォルダから読まれる
        host.exported_to = tmp_path
        with pytest.raises(ToolError, match="相対"):
            writable_path(host, "a.mp4")

    def test_folders_are_not_made_and_a_folder_is_not_a_file(self, saved: FakeHost) -> None:
        # 壊れると、書ける場所の中へ人の知らないフォルダが増えるか、フォルダを潰しに行く
        assert saved.saved_at is not None
        folder = saved.saved_at.parent
        with pytest.raises(ToolError, match="フォルダがありません"):
            writable_path(saved, str(folder / "無い" / "a.mp4"))
        (folder / "中").mkdir()
        with pytest.raises(ToolError, match="ファイルの場所"):
            writable_path(saved, str(folder / "中"))

    @pytest.mark.parametrize("raw", [None, "", "   ", 3])
    def test_a_missing_place_is_refused(self, saved: FakeHost, raw: object) -> None:
        # 壊れると、空の場所が作品のフォルダそのものと読まれる
        with pytest.raises(ToolError, match="path"):
            writable_path(saved, raw)


# --- 確かめは二段 ---


class TestIrreversibleIsAlwaysConfirmed:
    """確かめの二段を見る 送る物の確かめ（#342）は tests/ai/test_outbound.py が見るので、
    ここのブリッジは送る物を許してある（allow_sending）
    """

    def test_the_test_tool_declares_what_it_returns(self, saved: FakeHost) -> None:
        # 壊れると、試験用の道具の宣言が結果と食い違ったまま、確かめの試験が
        # 名前と場所の確かめを通らない道で走り、本物の道具と違う道筋を見る
        assert saved.saved_at is not None
        target = saved.saved_at.parent / "メモ.txt"
        result = WRITE_NOTE(saved, {"path": str(target), "text": "済"})
        assert not undeclared(WRITE_NOTE, result)

    def test_auto_approve_does_not_skip_it(self, saved: FakeHost) -> None:
        # 壊れると、「変更を自動で承認」を入れた人の作品を、AI が黙って上書きする
        assert saved.saved_at is not None
        target = saved.saved_at.parent / "メモ.txt"
        bridge = allow_sending(EditorBridge(saved, auto_approve=True))
        done, box = call_in_thread(bridge, WRITE_NOTE, {"path": str(target), "text": "済"})
        seen = answer_approvals(bridge, done, lambda _approval: True)

        assert len(seen) == 1
        assert box.get("error") is None
        assert target.read_text(encoding="utf-8") == "済"

    def test_the_request_says_what_and_where(self, saved: FakeHost) -> None:
        # 壊れると、道具の名前しか見えず、どのファイルが置き換わるのか分からないまま許す
        assert saved.saved_at is not None
        target = saved.saved_at.parent / "メモ.txt"
        target.write_text("前の中身", encoding="utf-8")
        bridge = allow_sending(EditorBridge(saved))
        done, _box = call_in_thread(bridge, WRITE_NOTE, {"path": str(target)})
        seen = answer_approvals(bridge, done, lambda _approval: False)

        assert len(seen) == 1
        approval = seen[0]
        assert approval.irreversible
        assert approval.consent is not None
        assert approval.consent.action == "メモを書き出します"
        assert approval.consent.target == str(target.resolve())
        assert "置き換えます" in approval.summary
        assert str(target.resolve()) in approval.summary

    def test_allowing_it_always_is_not_remembered(self, saved: FakeHost) -> None:
        # 壊れると、1 度「以降は確認しない」を押した上書きが、次から黙って走る
        assert saved.saved_at is not None
        bridge = allow_sending(EditorBridge(saved))
        bridge.allow_always("write_note")
        for text in ("1 回目", "2 回目"):
            done, _box = call_in_thread(
                bridge, WRITE_NOTE, {"path": str(saved.saved_at.parent / "メモ.txt"), "text": text}
            )
            assert len(answer_approvals(bridge, done, lambda _approval: True)) == 1

    def test_the_bridge_does_not_even_note_always_for_irreversible_tools(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、保存や書き出しの道具（B1 #327）の「以降は確認しない」が覚えられ、
        # 確かめの経路を 1 か所変えただけで黙って走るようになる
        monkeypatch.setattr(operations, "OPERATIONS", (*OPERATIONS, WRITE_NOTE))
        bridge = EditorBridge(host)
        bridge.allow_always("write_note")
        bridge.allow_always("split_clip")
        assert bridge._always == {"split_clip"}

    def test_denying_writes_nothing(self, saved: FakeHost) -> None:
        # 壊れると、拒否しても書かれ、上書きされた中身は取り消しでも戻らない
        assert saved.saved_at is not None
        target = saved.saved_at.parent / "メモ.txt"
        target.write_text("前の中身", encoding="utf-8")
        bridge = allow_sending(EditorBridge(saved, auto_approve=True))
        done, box = call_in_thread(bridge, WRITE_NOTE, {"path": str(target), "text": "新"})
        answer_approvals(bridge, done, lambda _approval: False)

        assert isinstance(box.get("error"), ToolError)
        assert "毎回確かめ" in str(box["error"])
        assert target.read_text(encoding="utf-8") == "前の中身"

    def test_a_place_outside_is_refused_before_asking(
        self, saved: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、書けない場所なのに窓が出て、人が許しても何も起きない（または外へ書く）
        bridge = allow_sending(EditorBridge(saved, auto_approve=True))
        target = tmp_path / "外.txt"
        done, box = call_in_thread(bridge, WRITE_NOTE, {"path": str(target)})
        seen = answer_approvals(bridge, done, lambda _approval: True)

        assert seen == []
        assert isinstance(box.get("error"), ToolError)
        assert not target.exists()

    def test_a_place_swapped_while_asking_is_refused_when_writing(
        self, saved: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、窓が出ている間にフォルダをリンクへ差し替えられ、許した後に外へ書く
        assert saved.saved_at is not None
        inside = saved.saved_at.parent / "出力"
        inside.mkdir()
        outside = tmp_path / "外"
        outside.mkdir()
        # リンクを作れるかを先に試す 確かめに答える途中で飛ばすと、AI 役のスレッドが残る
        probe = tmp_path / "試し"
        _symlink_dir(probe, outside)
        if probe.is_symlink():
            probe.unlink()
        else:
            probe.rmdir()
        bridge = allow_sending(EditorBridge(saved))
        done, box = call_in_thread(bridge, WRITE_NOTE, {"path": str(inside / "メモ.txt")})

        def swap_then_allow(_approval: Approval) -> bool:
            inside.rmdir()
            _symlink_dir(inside, outside)
            return True

        answer_approvals(bridge, done, swap_then_allow)
        assert isinstance(box.get("error"), ToolError)
        assert not (outside / "メモ.txt").exists()

    def test_a_hard_link_made_while_asking_is_refused_when_writing(
        self, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、窓が出ている間に書く先を元の素材へのハードリンクに差し替えられ、
        # 許した後に元の素材の中身を上書きする
        original = tmp_path / "素材置き場" / "素材.mp4"
        original.parent.mkdir()
        original.write_bytes(b"original")
        host = FakeHost(make_loaded(replace(video_media, path=original), transcript))
        folder = tmp_path / "作品"
        folder.mkdir()
        host.saved_at = folder / "作品.sme"
        target = folder / "メモ.txt"
        # 作れるかを先に試す 確かめに答える途中で飛ばすと、AI 役のスレッドが残る
        _hard_link(tmp_path / "試し", original)
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, WRITE_NOTE, {"path": str(target), "text": "壊す"})

        def link_then_allow(_approval: Approval) -> bool:
            _hard_link(target, original)
            return True

        answer_approvals(bridge, done, link_then_allow)
        assert isinstance(box.get("error"), ToolError)
        assert original.read_bytes() == b"original"

    def test_undoable_edits_still_follow_auto_approve(self, host: FakeHost) -> None:
        # 壊れると（全部を毎回確かめると）、自動で承認に任せた人の編集が 1 手ごとに止まる
        bridge = EditorBridge(host, auto_approve=True)
        clip = host.project.timeline.tracks[0].clips[0]
        done, box = call_in_thread(
            bridge, _operation("split_clip"), {"clip_id": str(clip.id), "frame": 30}
        )
        seen = answer_approvals(bridge, done, lambda _approval: True)
        assert seen == []
        assert box.get("error") is None

    def test_an_interrupt_while_asking_folds_it(self, saved: FakeHost) -> None:
        # 壊れると、中断を押しても AI のスレッドが確かめの返事を 10 分待ち続ける
        assert saved.saved_at is not None
        bridge = allow_sending(EditorBridge(saved))
        done, box = call_in_thread(
            bridge, WRITE_NOTE, {"path": str(saved.saved_at.parent / "メモ.txt")}
        )

        def interrupt(_approval: Approval) -> bool:
            bridge.cancel()
            return True

        deadline = time.monotonic() + 5
        while not done.is_set() and time.monotonic() < deadline:
            bridge.pump()
            approval = bridge.take_approval()
            if approval is not None:
                interrupt(approval)
            time.sleep(0.005)
        assert done.is_set()
        assert isinstance(box.get("error"), ToolError)
        assert not (saved.saved_at.parent / "メモ.txt").exists()
