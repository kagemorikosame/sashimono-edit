"""AI から何本かのクリップをまとめて扱うツールと、トラックの高さ

画面でできることは AI からもできる、という約束を守るためのもの 画面で
選んだものに対して「これをまとめて」と頼めることも見る
"""

from __future__ import annotations

from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.model import Clip, ClipId, Track, TrackKind
from sashimono.effects.sources import TEXT
from tests.ai.conftest import FakeHost


def run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(host, arguments)


@pytest.fixture
def two(host: FakeHost) -> tuple[ClipId, ClipId]:
    """素材のクリップ（V1、0〜300）と、別のトラックのテキスト（0〜30）"""
    track = Track(TrackKind.VIDEO, "V2")
    text = Clip(timeline_start=0, duration=30, source=TEXT.create())
    host.apply_commands([AddTrack(track), AddClip(track.id, text)], "準備")
    return host.document.project.timeline.tracks[0].clips[0].id, text.id


def _starts(host: FakeHost) -> list[int]:
    return [t.clips[0].timeline_start for t in host.document.project.timeline.tracks]


class TestSelection:
    def test_selection_round_trips(self, host: FakeHost, two: tuple[ClipId, ClipId]) -> None:
        # 壊れると、AI は画面で選んだ何本かを知る手段が無い
        run(host, "select_clips", clip_ids=[str(c) for c in two])
        result = run(host, "get_selection")
        assert result["clip_ids"] == [str(c) for c in two]
        assert result["clip_id"] == str(two[1])

    def test_the_last_repeat_decides_the_primary(
        self, host: FakeHost, two: tuple[ClipId, ClipId]
    ) -> None:
        # 先に現れた位置を残すと、[A, B, A] の主が B に化け、設定パネルに別の
        # クリップが出る
        a, b = (str(c) for c in two)
        run(host, "select_clips", clip_ids=[a, b, a])
        assert run(host, "get_selection")["clip_id"] == a

    def test_leaving_out_the_ids_does_not_clear(
        self, host: FakeHost, two: tuple[ClipId, ClipId]
    ) -> None:
        # 空の引数で呼ばれただけで選択が消えると、人が選んでいたものを失う
        host.select_clips(list(two))
        with pytest.raises(ToolError, match="空の配列"):
            run(host, "select_clips")
        assert host.selected_clips == two
        run(host, "select_clips", clip_ids=[])
        assert not host.selected_clips

    def test_an_unknown_id_is_refused(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="list_clips"):
            run(host, "select_clips", clip_ids=["無い"])


class TestMoving:
    def test_the_selection_is_the_default(self, host: FakeHost, two: tuple[ClipId, ClipId]) -> None:
        # 「選んでいるのをまとめて 1 秒後ろへ」が ID 無しで通ること
        host.select_clips(list(two))
        run(host, "move_clips", delta=30)
        assert _starts(host) == [30, 30]
        assert host.document.history_labels[-1] == "2 本を移動"

    def test_one_missing_id_moves_nothing(self, host: FakeHost, two: tuple[ClipId, ClipId]) -> None:
        # 一部だけ動くと、AI も人もどれが動いたか追えない
        with pytest.raises(ToolError):
            run(host, "move_clips", clip_ids=[str(two[0]), "無い"], delta=30)
        assert _starts(host) == [0, 0]

    def test_an_empty_list_is_not_the_selection(
        self, host: FakeHost, two: tuple[ClipId, ClipId]
    ) -> None:
        # 空の一覧を「選択を使う」と取ると、何も指していないつもりの呼び出しで、
        # 選んでいる全部が消える
        host.select_clips(list(two))
        with pytest.raises(ToolError, match="空"):
            run(host, "delete_clips", clip_ids=[])
        assert _starts(host) == [0, 0]

    def test_a_wrong_type_is_a_tool_error(self, host: FakeHost) -> None:
        # 素の TypeError で落ちると、AI には何を直せばよいかが伝わらない
        with pytest.raises(ToolError, match="配列"):
            run(host, "move_clips", clip_ids=3, delta=10)

    def test_repeated_ids_count_once(self, host: FakeHost, two: tuple[ClipId, ClipId]) -> None:
        # 重なったまま渡すと、履歴の「2 本を移動」が実際の本数と食い違う
        run(host, "move_clips", clip_ids=[str(two[1]), str(two[1])], delta=10)
        assert host.document.history_labels[-1] == "1 本を移動"

    def test_zero_is_refused(self, host: FakeHost, two: tuple[ClipId, ClipId]) -> None:
        with pytest.raises(ToolError, match="delta"):
            run(host, "move_clips", clip_ids=[str(two[0])], delta=0)


class TestDeletingAndDuplicating:
    def test_delete_clips(self, host: FakeHost, two: tuple[ClipId, ClipId]) -> None:
        # 壊れて一部が残ると、AI の「消しました」とタイムラインの中身が食い違う
        run(host, "delete_clips", clip_ids=[str(c) for c in two])
        assert all(not t.clips for t in host.document.project.timeline.tracks)

    def test_duplicate_is_one_step_and_selects_the_copies(
        self, host: FakeHost, two: tuple[ClipId, ClipId]
    ) -> None:
        # 貼ったものが選ばれていないと、続けて「それを動かして」が通らない
        result = run(host, "duplicate_clips", clip_ids=[str(two[1])], at_frame=100)
        (pasted,) = result["pasted"]
        assert host.selected_clips == (ClipId(pasted),)
        assert host.document.project.timeline.tracks[1].clips[1].timeline_start == 100
        host.document.undo()
        assert len(host.document.project.timeline.tracks[1].clips) == 1


class TestTrackHeight:
    def test_the_actual_height_is_reported(self, host: FakeHost) -> None:
        # 範囲の外は端へ寄せる 頼んだ値を返すと、AI はそうなったと思い込む
        result = run(host, "set_track_height", height=1000)
        assert set(result["heights"].values()) == {240}
        assert run(host, "list_tracks")["tracks"][0]["height"] == 240

    def test_an_unknown_track_is_refused(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="list_tracks"):
            run(host, "set_track_height", track_id="無い", height=80)


class TestTransition:
    def test_a_transition_is_placed(self, host: FakeHost) -> None:
        result = run(host, "add_transition", style="push", duration=40, angle=90.0)
        assert result == {"added": "push", "duration": 40}
        placed = [
            clip
            for track in host.document.project.timeline.tracks
            for clip in track.clips
            if clip.source is not None and clip.source.kind == "transition"
        ]
        assert len(placed) == 1
        assert placed[0].duration == 40
        assert placed[0].source is not None
        assert placed[0].source.params["style"] == "push"

    def test_an_unknown_style_is_refused(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="style"):
            run(host, "add_transition", style="ワイプ")
