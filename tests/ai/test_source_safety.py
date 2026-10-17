"""AI の道具は元の素材を壊さない（#341 D3）

確かめたこと（2026-10-14 ``ai/operations.py`` を読んで、書き込む道具を全部数えた）

- 変える道具は 35 個 どれも ``host.apply_commands`` か ``Document`` の取り消しを通して
  プロジェクト（メモリの中）だけを変える ファイルへ書く道具は 0 個
- 2026-10-15 に保存と書き出し（B1 #327）を足した ファイルへ書く道具は 5 個（上書き保存・
  名前を付けて保存・書き出し・字幕の書き出し・静止画の保存）になった どれも取り消せない
  操作で、毎回確かめ、書くのはホスト（画面の〔保存〕・書き出しの窓・字幕の〔書き出し…〕・
  〔静止画を保存…〕と同じ処理）に頼む 道具のコードはファイルを書かない
  書き出しを止める道具（cancel_export）は作品もファイルも変えない（書きかけを捨てるだけ）
- ファイルに触るのは読む所だけ ``import_media`` が素材を読み（``host.probe``）、波形と
  控えを予約する（書く先は本人のキャッシュの置き場で、素材の隣ではない）
- 消す道具が消すのは、プロジェクトの中のクリップ（と、クリップが持つ縁取りの層）だけ
  素材を素材一覧から外す・トラックやシーンを消す道具は無い
- 2026-10-17 に目印（マーカー）を置く・直す・消す道具を足した（#330） 消すのは目盛りの目印で、
  これもプロジェクトの中だけを変え、取り消し 1 回で戻る
- 2026-10-17 に保存した見た目と棚を当てる・置く道具を足した（#328） プリセットとエイリアスの
  置き場・棚・``.exo`` は読むだけで、置き場の保存・名前の変更・削除の道具は無い（未決 #350）

ここではそれを試験で押さえる 新しく道具を足して崩れたら落ちる
"""

from __future__ import annotations

import ast
import builtins
import io
import os
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import sashimono.ai.bridge
import sashimono.ai.operations
import sashimono.ai.safety
from sashimono.ai.operations import OPERATIONS, find_operation
from sashimono.ai.safety import Safety
from sashimono.core.model import MediaItem, Transcript
from tests.ai.conftest import FakeHost, make_loaded

#: 変える道具の一覧（数えた物） 足すときは、何を変えるかを確かめてからここへ足す
#: 「元に戻す」で戻らない物は Safety.IRREVERSIBLE にして、ここではなく確かめの側へ入れる
UNDOABLE_TOOLS = frozenset(
    {
        # 素材・トラック・プロジェクトの設定
        "import_media",
        "place_media",
        "add_track",
        "set_track_state",
        "set_track_height",
        "set_resolution",
        # 置く
        "add_text",
        "add_shape",
        "add_transition",
        "add_scene",
        "place_scene",
        "place_subtitles",
        "duplicate_clips",
        # 切る・動かす・消す
        "split_clip",
        "trim_clip",
        "move_clip",
        "move_clips",
        "delete_clip",
        "delete_clips",
        "jet_cut",
        "group_clips",
        "ungroup_clips",
        # 見た目
        "set_clip_property",
        "add_effect",
        "add_stroke",
        "remove_stroke",
        "move_stroke",
        "set_stroke_enabled",
        "set_param",
        "add_keyframe",
        # 字幕
        "set_subtitle_text",
        "replace_subtitle_text",
        "clean_subtitles",
        "transcribe",
        # 目印（#330） プロジェクトの中の目印だけを変える 素材にもファイルにも触らない
        "add_markers",
        "update_marker",
        "delete_markers",
        # 保存した見た目と棚（#328） 置き場と棚のファイルは読むだけ プロジェクトの中のクリップ
        # （と、棚や .exo が指す素材の登録）だけを変え、取り消し 1 回で戻る
        "apply_preset",
        "place_alias",
        "place_template",
        "apply_template",
        "import_objects",
        # 履歴
        "undo",
        # 書き出しを止める 作品もファイルも変えず、書きかけを捨てるだけ もう 1 度書き出せば
        # 同じ物ができるので、取り消せない操作の毎回の確かめは要らない
        "cancel_export",
    }
)

#: ファイルへ書く道具（数えた物） どれも取り消せない操作で、確かめの側（test_safety.py
#: test_save_and_export.py）で押さえる 足すときは、書く場所を pinned_place か書き出しの
#: 片付け（ui/ai_export.py）に通してからここへ足す
FILE_WRITING_TOOLS = frozenset(
    {"save_project", "save_project_as", "export_video", "export_subtitles", "save_snapshot"}
)

#: 道具のコードがホストに書かせる口と、それを呼んでよい道具の関数
#: 道具の関数の外（読む道具・編集の道具）から呼ぶと、確かめを通らずにファイルを書く
_HOST_WRITERS = {
    "_save_project": {"write_project"},
    "_save_project_as": {"write_project"},
    "_export_video": {"start_export"},
    "_export_subtitles": {"write_subtitles"},
    "_save_snapshot": {"write_snapshot"},
}

#: 道具のコードが呼んではいけない、ファイルを書く・消す・動かす呼び出し
_WRITING_CALLS = frozenset(
    {
        "open",
        "write_text",
        "write_bytes",
        "write_atomic",
        "write_atomic_bytes",
        "unlink",
        "rename",
        "rmdir",
        "rmtree",
        "touch",
        "mkdir",
        "remove",
        "removedirs",
        "truncate",
        "copyfile",
        "copy2",
        "move",
        "save",
        "save_project",
        "copy_atomic",
    }
)

#: 道具が使ってよい、消す編集の命令 どれもプロジェクトの中のクリップ（と、その縁取りの層）
#: を消す 素材を外す（RemoveMedia）・トラックやシーンを消す命令はここに無い
#: 目印（RemoveMarkers #330）はプロジェクトの中の目盛りの印で、素材にもクリップにも触らず、
#: 取り消し 1 回で戻る
_ALLOWED_REMOVALS = frozenset(
    {"RemoveClip", "RemoveClips", "RemoveStroke", "RippleCut", "RemoveMarkers"}
)


def _source(module: object) -> ast.Module:
    path = getattr(module, "__file__", None)
    assert isinstance(path, str)
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _called_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
            # os.replace / os.rename は名前だけでは dataclasses.replace と見分けられない
            if isinstance(func.value, ast.Name) and func.value.id in {"os", "shutil"}:
                names.add(f"{func.value.id}.{func.attr}")
    return names


class TestCounted:
    def test_the_tools_that_change_things_are_the_ones_counted(self) -> None:
        # 壊れると、確かめていない変える道具が増え、元の素材やファイルに触るかを誰も見ていない
        changing = {operation.name for operation in OPERATIONS if operation.writes}
        undoable = {
            operation.name for operation in OPERATIONS if operation.safety is Safety.UNDOABLE
        }
        assert undoable == UNDOABLE_TOOLS
        # 取り消せない道具は、確かめの側（test_safety.py）で押さえる
        irreversible = {operation.name for operation in OPERATIONS if operation.irreversible}
        assert changing - undoable == irreversible
        assert irreversible == FILE_WRITING_TOOLS

    def test_only_the_counted_tools_ask_the_host_to_write(self) -> None:
        # 壊れると、確かめを通らない道具（読む道具や編集の道具）から作品の保存や書き出しを
        # 呼べ、「変更を自動で承認」の人の作品や書き出し先を黙って上書きする
        tree = _source(sashimono.ai.operations)
        writers = {name for names in _HOST_WRITERS.values() for name in names}
        found: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                called = _called_names(ast.Module(body=node.body, type_ignores=[])) & writers
                if called:
                    found[node.name] = called
        assert found == _HOST_WRITERS
        # 書く道具は、人が許した確かめの中身を受け取る本体（approved）で書く 受け取らずに
        # 書く先の身元を採り直すと、照らし合わせの後の変更を受け入れる（#327 Codex の P2）
        handlers: dict[str, str] = {}
        for operation in OPERATIONS:
            if operation.name in FILE_WRITING_TOOLS:
                assert operation.approved is not None, operation.name
                handlers[operation.name] = operation.approved.__name__
        assert set(handlers.values()) == set(_HOST_WRITERS)
        # 本体の中で書く先の身元を採り直さない（確かめの中身の身元で照らす）
        tree = _source(sashimono.ai.operations)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in _HOST_WRITERS:
                called = _called_names(ast.Module(body=node.body, type_ignores=[]))
                assert "expect_now" not in called, node.name

    @pytest.mark.parametrize(
        "module", [sashimono.ai.operations, sashimono.ai.bridge, sashimono.ai.safety]
    )
    def test_no_tool_code_writes_files(self, module: object) -> None:
        # 壊れると、AI の道具が元の素材や作品のファイルを書き換え、「元に戻す」で戻らない
        # 書く道具（保存・書き出し #327）も、ここでは書かずにホストへ頼む（数えた口だけ
        # test_only_the_counted_tools_ask_the_host_to_write） 取り消せない操作として確かめを
        # 通し、書く場所は pinned_place を通す
        called = _called_names(_source(module))
        writing = called & (_WRITING_CALLS | {f"os.{name}" for name in ("replace", "rename")})
        writing |= {name for name in called if name.startswith("shutil.")}
        assert writing == set()

    def test_tools_only_remove_clips(self) -> None:
        # 壊れると、AI が素材を素材一覧から外したり、トラックやシーンを丸ごと消したりできる
        tree = _source(sashimono.ai.operations)
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id.startswith(("Remove", "Ripple"))
        }
        assert used <= _ALLOWED_REMOVALS
        assert "RemoveMedia" not in used


# --- 実際に消して、素材のファイルと素材一覧が残ることを見る ---


@pytest.fixture
def original(tmp_path: Path) -> Path:
    path = tmp_path / "元の素材.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)))
    return path


@pytest.fixture
def on_disk(video_media: MediaItem, transcript: Transcript, original: Path) -> FakeHost:
    """素材がディスクにある作品"""
    return FakeHost(make_loaded(replace(video_media, path=original), transcript))


@pytest.fixture
def no_file_writes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """道具を動かす間、ファイルを書く・消す・動かす呼び出しが来たら記録して断る

    断るだけでなく記録もするのは、道具の側が例外を握り潰しても、試験が気付けるように
    """
    attempts: list[str] = []
    real_open = builtins.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in "wax+"):
            attempts.append(f"open({file!r}, {mode!r})")
            raise PermissionError("試験: 道具がファイルを書こうとした")
        return real_open(file, mode, *args, **kwargs)

    def refuse(name: str) -> Any:
        def refused(*arguments: Any, **_options: Any) -> None:
            attempts.append(f"{name}{arguments!r}")
            raise PermissionError(f"試験: 道具が {name} を呼んだ")

        return refused

    # Path.write_bytes などは io.open を呼ぶ 片方だけ見張ると、そちらを通る書き込みを見落とす
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    for name in ("remove", "unlink", "replace", "rename", "rmdir"):
        monkeypatch.setattr(os, name, refuse(f"os.{name}"))
    for name in ("rmtree", "move", "copyfile", "copy2"):
        monkeypatch.setattr(shutil, name, refuse(f"shutil.{name}"))
    yield attempts


def _run(host: FakeHost, name: str, arguments: dict[str, Any]) -> object:
    operation = find_operation(name)
    assert operation is not None
    return operation(host, arguments)


class TestDeletingKeepsTheOriginals:
    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("delete_clip", {}),
            ("delete_clip", {"ripple": True}),
            ("delete_clips", {}),
            ("delete_clips", {"ripple": True}),
        ],
    )
    def test_deleting_a_clip_keeps_the_file_and_the_pool_and_undo_restores_it(
        self,
        on_disk: FakeHost,
        original: Path,
        no_file_writes: list[str],
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        # 壊れると、クリップを消しただけで素材のファイルか素材一覧の行まで消え、
        # 取り消してもクリップが戻らないか、戻っても絵が出ない
        before = on_disk.document.project
        content = original.read_bytes()
        stamp = original.stat().st_mtime_ns
        clip = before.timeline.tracks[0].clips[0]
        on_disk.select_clips([clip.id])

        _run(on_disk, name, {"clip_id": str(clip.id), **arguments})

        after = on_disk.document.project
        assert after.timeline.tracks[0].clips == ()
        assert after.media == before.media
        assert original.read_bytes() == content
        assert original.stat().st_mtime_ns == stamp
        assert no_file_writes == []

        on_disk.document.undo()
        assert on_disk.document.project == before

    def test_the_undo_tool_brings_a_deleted_clip_back_in_one_step(
        self, on_disk: FakeHost, no_file_writes: list[str]
    ) -> None:
        # 壊れると、AI が消したクリップを AI 自身が戻せない（戻すのに何段も要る）
        before = on_disk.document.project
        clip = before.timeline.tracks[0].clips[0]
        _run(on_disk, "split_clip", {"clip_id": str(clip.id), "frame": 90})
        middle = on_disk.document.project
        pieces = [piece.id for piece in middle.timeline.tracks[0].clips]
        _run(on_disk, "delete_clips", {"clip_ids": [str(piece) for piece in pieces]})

        _run(on_disk, "undo", {})
        assert on_disk.document.project == middle
        _run(on_disk, "undo", {})
        assert on_disk.document.project == before
        assert no_file_writes == []

    def test_the_guard_notices_a_write(self, original: Path, no_file_writes: list[str]) -> None:
        # 壊れると（見張りが効いていないと）、上の試験は書き込みがあっても通ってしまう
        with pytest.raises(PermissionError):
            original.write_bytes(b"x")
        with pytest.raises(PermissionError):
            original.unlink()
        assert len(no_file_writes) == 2
