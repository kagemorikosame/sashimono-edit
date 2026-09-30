"""AI からシーンとグループを扱うツール

画面でできることは AI からもできる約束 シーンを開いたあとの編集が、開いた
シーンの中へ入ることが肝（メインに入ると、AI の報告と画面が食い違う）
"""

from __future__ import annotations

from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from tests.ai.conftest import FakeHost


def run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(host, arguments)


class TestScenes:
    def test_a_new_scene_is_opened_and_edited(self, host: FakeHost) -> None:
        # 開いたシーンへ足したテキストがメインに入ると、AI の報告と中身が食い違う
        main_clips = len(run(host, "list_clips")["clips"])
        created = run(host, "add_scene", name="オープニング")
        assert run(host, "get_project")["active_scene"] == created["scene_id"]
        run(host, "add_text", text="はじまり", at_frame=0)
        assert len(run(host, "list_clips")["clips"]) == 1

        run(host, "set_active_scene", scene_id="")
        assert len(run(host, "list_clips")["clips"]) == main_clips

    def test_a_scene_can_be_placed_in_the_main_timeline(self, host: FakeHost) -> None:
        created = run(host, "add_scene", name="挿入", open=False)
        run(host, "place_scene", scene_id=created["scene_id"], at_frame=400, duration=60)
        placed = [clip for clip in run(host, "list_clips")["clips"] if clip["scene_id"]]
        assert [(clip["start"], clip["duration"]) for clip in placed] == [(400, 60)]

    def test_a_zero_length_is_refused(self, host: FakeHost) -> None:
        # 0 を既定の長さと読み替えると、AI が頼んだ長さと違うまま黙って置かれる
        created = run(host, "add_scene", name="挿入", open=False)
        with pytest.raises(ToolError, match="duration"):
            run(host, "place_scene", scene_id=created["scene_id"], duration=0)

    def test_a_scene_cannot_be_placed_inside_itself(self, host: FakeHost) -> None:
        # 置けてしまうと、描くときに無限に潜って固まる
        created = run(host, "add_scene", name="自分")
        with pytest.raises(ToolError, match="入れ子"):
            run(host, "place_scene", scene_id=created["scene_id"])

    def test_unknown_scenes_point_to_the_list(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="list_scenes"):
            run(host, "set_active_scene", scene_id="無い")

    def test_the_list_shows_every_scene(self, host: FakeHost) -> None:
        run(host, "add_scene", name="A", open=False)
        run(host, "add_scene", name="B", open=False)
        assert [scene["name"] for scene in run(host, "list_scenes")["scenes"]] == ["A", "B"]


class TestGroups:
    def test_group_and_ungroup(self, host: FakeHost) -> None:
        # 束ねた結果が list_clips に出ないと、AI はどれが一緒に動くのか分からない
        run(host, "add_text", text="上", at_frame=0)
        clips = [clip["clip_id"] for clip in run(host, "list_clips")["clips"]]
        run(host, "group_clips", clip_ids=clips)
        groups = {clip["group_id"] for clip in run(host, "list_clips")["clips"]}
        assert len(groups) == 1 and None not in groups

        run(host, "ungroup_clips", clip_ids=[clips[0]])
        assert {clip["group_id"] for clip in run(host, "list_clips")["clips"]} == {None}

    def test_moving_one_member_moves_the_group(self, host: FakeHost) -> None:
        # 画面では 1 本つかむと束ごと動く AI だけ 1 本を抜き出せると、束が裂ける
        run(host, "add_text", text="上", at_frame=0)
        clips = [clip["clip_id"] for clip in run(host, "list_clips")["clips"]]
        run(host, "group_clips", clip_ids=clips)
        moved = run(host, "move_clips", clip_ids=[clips[0]], delta=30)
        assert set(moved["moved"]) == set(clips)
        assert {clip["start"] for clip in run(host, "list_clips")["clips"]} == {30}

    def test_move_clip_on_one_member_moves_the_group(self, host: FakeHost) -> None:
        # 単体の移動だけ束を見ないと、AI が 1 本を動かしたときに束が裂ける
        run(host, "add_text", text="上", at_frame=0)
        clips = [clip["clip_id"] for clip in run(host, "list_clips")["clips"]]
        run(host, "group_clips", clip_ids=clips)
        run(host, "move_clip", clip_id=clips[0], timeline_start=45)
        assert {clip["start"] for clip in run(host, "list_clips")["clips"]} == {45}
        with pytest.raises(ToolError, match="グループ"):
            run(host, "move_clip", clip_id=clips[0], track_id="どこか")
        # 今のトラックを指すのは移動ではないので、断ると AI が同じ指定で止まる
        here = run(host, "list_clips")["clips"][0]["track_id"]
        run(host, "move_clip", clip_id=clips[0], timeline_start=60, track_id=here)
        assert {clip["start"] for clip in run(host, "list_clips")["clips"]} == {60}

    def test_delete_clip_on_one_member_leaves_no_orphans(self, host: FakeHost) -> None:
        # 画面で消すと仲間ごと消える AI だけ 1 本残ると、何が残ったのか分かりにくい
        run(host, "add_text", text="上", at_frame=0)
        clips = [clip["clip_id"] for clip in run(host, "list_clips")["clips"]]
        run(host, "group_clips", clip_ids=clips)
        run(host, "delete_clip", clip_id=clips[0])
        assert run(host, "list_clips")["clips"] == []

    def test_a_single_clip_is_refused(self, host: FakeHost) -> None:
        clip = run(host, "list_clips")["clips"][0]["clip_id"]
        with pytest.raises(ToolError, match="2 本"):
            run(host, "group_clips", clip_ids=[clip])
