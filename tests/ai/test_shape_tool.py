"""AI の図形の道具（add_shape）

前は道具が無く、AI はテキストしか置けなかった（AI テスト #3 で分かった）
"""

from __future__ import annotations

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.core.model import AnimatedValue
from tests.ai.conftest import FakeHost


def _add_shape(host: FakeHost, **arguments: object) -> object:
    operation = find_operation("add_shape")
    assert operation is not None and operation.writes
    return operation(host, dict(arguments))


def test_it_places_a_shape_with_the_given_look(host: FakeHost) -> None:
    _add_shape(
        host, shape="ellipse", width=200, height=100, color="#FF0000", at_frame=10, duration=20
    )
    placed = [c for t in host.document.project.timeline.tracks for c in t.clips if c.source]
    (clip,) = [c for c in placed if c.source is not None and c.source.kind == "shape"]
    assert (clip.timeline_start, clip.duration) == (10, 20)
    params = clip.source.params if clip.source is not None else {}
    assert params["shape"] == "ellipse"
    width = params["width"]
    assert isinstance(width, AnimatedValue) and width.static == 200
    assert params["color"] == (1.0, 0.0, 0.0, 1.0)


def test_it_is_one_undo_step(host: FakeHost) -> None:
    before = host.document.project
    _add_shape(host)
    host.document.undo()
    assert host.document.project == before


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"shape": "hexagram"}, "shape は"),
        ({"color": "red"}, "color は"),
        ({"duration": 0}, "duration"),
    ],
)
def test_bad_values_are_refused_with_a_reason(
    host: FakeHost, arguments: dict[str, object], message: str
) -> None:
    # 黙って既定の図形を置くと、AI は頼んだ物が置けたと思い込む
    with pytest.raises(ToolError, match=message):
        _add_shape(host, **arguments)
