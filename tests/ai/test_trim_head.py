"""AI アシスタントの ``trim_clip`` で頭を前へ伸ばす（#309）

前へ伸ばしすぎた頼みは断らず、伸ばせる所まで伸ばして、止めた理由と量を返す（利用者の決め）
置いたばかりのテキストを前へ伸ばして、プレビューで伸ばした所に字が出るまでを通す

各試験の頭の 1 行は、壊れたときに利用者に何が起きるか
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from sashimono.ai.operations import ImageResult, find_operation
from sashimono.core.commands import SetTrackState
from sashimono.core.model import Clip, Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.render import FrameRenderer
from tests.ai.conftest import FakeHost


def run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(host, arguments)


def _texts(host: FakeHost) -> list[Clip]:
    return [
        clip
        for track in host.document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == "text"
    ]


def _track_of(host: FakeHost, clip: Clip) -> str:
    located = host.document.project.timeline.locate_clip(clip.id)
    assert located is not None
    return str(located[0].id)


def _host() -> FakeHost:
    settings = ProjectSettings(width=320, height=180, frame_rate=FrameRate(30))
    return FakeHost(Project.create(settings))


class TestExtendingText:
    def test_a_new_text_extends_forward(self) -> None:
        # 置いたばかりのテキストを前へ伸ばすよう頼むと「素材の先頭より前」と断られ、AI が詰まる
        host = _host()
        run(host, "add_text", text="テロップ", at_frame=60, duration=60)
        (text,) = _texts(host)
        result = run(host, "trim_clip", clip_id=str(text.id), head_delta=-30)
        assert result["head_delta"] == -30
        assert "note" not in result
        (extended,) = _texts(host)
        assert (extended.timeline_start, extended.duration, extended.source_in) == (
            30,
            90,
            Fraction(0),
        )

    def test_too_far_stops_at_the_start_and_says_so(self) -> None:
        # 断ると AI は量を変えて何度も試し直す 止めた理由と量が無いと、伸びたと思い込む
        host = _host()
        run(host, "add_text", text="テロップ", at_frame=60, duration=60)
        (text,) = _texts(host)
        result = run(host, "trim_clip", clip_id=str(text.id), head_delta=-100)
        assert result["head_delta"] == -60
        assert result["requested_head_delta"] == -100
        assert result["stopped_at"] == "タイムラインの先頭"
        assert "60 フレーム" in result["note"]
        (extended,) = _texts(host)
        assert extended.timeline_start == 0

    def test_it_stops_at_the_clip_before(self) -> None:
        # 前のテキストに重なる所まで伸ばして失敗すると、何も変わらないまま AI が止まる
        host = _host()
        run(host, "add_text", text="前", at_frame=0, duration=40)
        run(host, "add_text", text="後", at_frame=60, duration=60)
        first, second = sorted(_texts(host), key=lambda clip: clip.timeline_start)
        # 同じトラックに並んでいないと、前のクリップで止まる所を確かめられない
        assert _track_of(host, first) == _track_of(host, second)
        result = run(host, "trim_clip", clip_id=str(second.id), head_delta=-50)
        assert result["head_delta"] == -20
        assert result["stopped_at"] == "前のクリップの終わり"

    def test_one_undo_brings_it_back(self) -> None:
        # 取り消し 1 回で戻らないと、AI の手を戻したつもりで本人の手まで消す
        host = _host()
        run(host, "add_text", text="テロップ", at_frame=60, duration=60)
        before = host.document.project
        (text,) = _texts(host)
        run(host, "trim_clip", clip_id=str(text.id), head_delta=-100)
        run(host, "undo")
        assert host.document.project == before

    def test_the_tail_stops_at_the_clip_after(self) -> None:
        # 後ろのテキストに重なる所まで伸ばして断られると、AI は量を変えて何度も試し直す
        host = _host()
        run(host, "add_text", text="前", at_frame=0, duration=40)
        run(host, "add_text", text="後", at_frame=60, duration=60)
        first, second = sorted(_texts(host), key=lambda clip: clip.timeline_start)
        assert _track_of(host, first) == _track_of(host, second)
        result = run(host, "trim_clip", clip_id=str(first.id), tail_delta=50)
        assert result["tail_delta"] == 20
        assert result["requested_tail_delta"] == 50
        assert result["stopped_at"] == "後ろのクリップの頭"
        assert "20 フレーム" in result["note"]
        first_now = min(_texts(host), key=lambda clip: clip.timeline_start)
        assert first_now.timeline_end == 60

    @pytest.mark.parametrize(("head", "tail"), [(-10, 0), (10, 0), (0, 10), (0, -10)])
    def test_a_locked_text_is_left_and_told(self, head: int, tail: int) -> None:
        # 動かないのに頼んだ量を返すと、AI は伸ばせたと思い込み、履歴に空の 1 手が残る
        host = _host()
        run(host, "add_text", text="テロップ", at_frame=60, duration=60)
        (text,) = _texts(host)
        track = host.document.project.timeline.locate_clip(text.id)
        assert track is not None
        host.apply_commands([SetTrackState(track[0].id, locked=True)], "ロック")
        before = host.document.project
        labels = host.document.history_labels
        result = run(host, "trim_clip", clip_id=str(text.id), head_delta=head, tail_delta=tail)
        assert (result["head_delta"], result["tail_delta"]) == (0, 0)
        assert result["stopped_at"] == "ロックしたトラック"
        assert "ロック" in result["note"]
        assert host.document.history_labels == labels
        assert host.document.project == before

    def test_nothing_to_extend_changes_nothing(self) -> None:
        # 0 フレームの命令を履歴へ積むと、取り消しを 1 回むだに使わせる
        host = _host()
        run(host, "add_text", text="テロップ", at_frame=0, duration=60)
        (text,) = _texts(host)
        labels = host.document.history_labels
        result = run(host, "trim_clip", clip_id=str(text.id), head_delta=-10)
        assert result["head_delta"] == 0
        assert result["stopped_at"] == "タイムラインの先頭"
        assert host.document.history_labels == labels


class _RenderingHost(FakeHost):
    """プレビューを本当に描く 描いた絵を残し、PNG の頭だけを返す"""

    def __init__(self, project: Project) -> None:
        super().__init__(project)
        self.pictures: list[np.ndarray] = []

    def render_png(self, frame: int, *, width: int) -> bytes:
        renderer = FrameRenderer(self.document.project)
        try:
            self.pictures.append(renderer.render(frame))
        finally:
            renderer.close()
        return super().render_png(frame, width=width)


@pytest.mark.usefixtures("gpu")
def test_place_extend_and_look() -> None:
    # 伸ばした所に字が出ないと、AI はプレビューを見て伸ばせていないと判断し、やり直し続ける
    host = _RenderingHost(
        Project.create(ProjectSettings(width=320, height=180, frame_rate=FrameRate(30)))
    )
    run(host, "add_text", text="■■■■", at_frame=60, duration=60, size=80)
    (text,) = _texts(host)
    run(host, "preview_frame", frame=40)
    assert host.pictures[-1][..., :3].max() == 0, "伸ばす前から字が出ている"
    run(host, "trim_clip", clip_id=str(text.id), head_delta=-30)
    shown = run(host, "preview_frame", frame=40)
    assert isinstance(shown, ImageResult)
    assert host.pictures[-1][..., :3].max() > 128, "伸ばした所に字が出ていない"
