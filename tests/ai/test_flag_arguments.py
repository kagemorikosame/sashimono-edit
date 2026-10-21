"""AI の道具の真偽の引数（#323 のレビューで見つけた読み方の穴）

``bool(arguments.get(...))`` で読むと、AI が文字の "false" を渡したときに真と読む
断るはずの上書きを始めたり、切るはずのミュートを入れたりと、頼んだのと逆の編集になる
どの道具も ``_flag_argument`` で読み、true / false 以外は断る
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai import operations
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from tests.ai.conftest import FakeHost


def _run(host: FakeHost, tool: str, arguments: dict[str, Any]) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(host, arguments)


def _media_id(host: FakeHost) -> str:
    return str(host.document.project.media[0].id)


def _clip_id(host: FakeHost) -> str:
    return str(host.document.project.timeline.tracks[0].clips[0].id)


def _stroke(host: FakeHost) -> dict[str, Any]:
    _run(host, "add_text", {"text": "縁", "at_frame": 0, "duration": 60})
    clip = next(
        clip
        for track in host.document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == "text"
    )
    added = _run(host, "add_stroke", {"clip_id": str(clip.id)})
    return {"clip_id": str(clip.id), "stroke_id": added["stroke_id"]}


def _jet_cut(host: FakeHost) -> dict[str, Any]:
    from tests.ui.test_subtitle_panel import make_waveform

    host.stub_waveform = make_waveform([(0.0, 400), (0.5, 1500)])
    return {"media_id": _media_id(host)}


#: 道具・真偽の引数・その道具が真偽を読むところまで進むための引数
CASES: list[tuple[str, str, Callable[[FakeHost], dict[str, Any]]]] = [
    ("get_subtitles", "compact", lambda host: {}),
    ("delete_clip", "ripple", lambda host: {"clip_id": _clip_id(host)}),
    ("delete_clips", "ripple", lambda host: {"clip_ids": [_clip_id(host)]}),
    ("set_stroke_enabled", "enabled", _stroke),
    ("jet_cut", "keep_speech", _jet_cut),
    ("transcribe", "replace", lambda host: {"media_id": _media_id(host)}),
    ("add_scene", "open", lambda host: {"name": "二つ目"}),
    (
        "set_track_state",
        "muted",
        lambda host: {"track_id": str(host.document.project.timeline.tracks[0].id)},
    ),
    (
        "set_track_state",
        "solo",
        lambda host: {"track_id": str(host.document.project.timeline.tracks[0].id)},
    ),
    (
        "set_track_state",
        "locked",
        lambda host: {"track_id": str(host.document.project.timeline.tracks[0].id)},
    ),
]


@pytest.mark.parametrize(("tool", "flag", "prepare"), CASES, ids=[f"{t}-{f}" for t, f, _ in CASES])
@pytest.mark.parametrize("value", ["false", "true", 0, None])
def test_a_flag_that_is_not_true_or_false_is_refused_before_any_change(
    host: FakeHost,
    tool: str,
    flag: str,
    prepare: Callable[[FakeHost], dict[str, Any]],
    value: Any,
) -> None:
    """壊れると、AI が文字の "false" を渡したのを真と読み、頼んだのと逆の編集をする
    起こし直し（transcribe の replace）なら人が直した字幕を新しい起こしで上書きし、
    ripple なら後ろのクリップを詰め、muted なら音を消す 断るときに半分だけ変えて
    おくと（シーンだけ増えるなど）、AI は断られたと思って同じ頼みを繰り返す
    """
    arguments = prepare(host)
    before = host.document.project
    label = host.document.undo_label
    started = host.transcription
    arguments[flag] = value
    with pytest.raises(ToolError, match=flag):
        _run(host, tool, arguments)
    assert host.document.project == before
    assert host.document.undo_label == label
    assert host.transcription == started


def test_transcribe_still_replaces_when_told_true(host: FakeHost) -> None:
    """壊れると、厳しく読むついでに本物の true まで断り、AI が字幕を起こし直せなくなる"""
    _run(host, "transcribe", {"media_id": _media_id(host), "replace": True})
    assert host.transcription.endswith("で開始")


def test_no_tool_reads_a_flag_with_bool() -> None:
    """壊れると、新しく足した道具が bool() で真偽を読み、文字の "false" を真として
    頼んだのと逆の編集をする（この表に道具を足し忘れても、ここで気付ける）
    """
    source = Path(operations.__file__).read_text(encoding="utf-8")
    loose = [
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if re.search(r"\bbool\(\s*arguments\b", line)
    ]
    assert loose == [], f"bool(arguments…) で読んでいる行: {loose} _flag_argument で読む"
