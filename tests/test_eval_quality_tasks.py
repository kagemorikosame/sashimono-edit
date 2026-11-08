"""評価の台本に足した課題（#414 見どころを選ぶ・#437 グループを保つ）と AI の段

実物の素材と本物の AI は CI に無いので、合成の素材と偽の会話で 1 周させる
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.session import AgentEvent, EventKind
from sashimono.core.commands import Document, RemoveClip, SplitClips
from sashimono.core.io import load_project, save_project
from tests.scene_video import write_scene_video
from tests.test_eval_tasks_tool import _write_talk, tool

__all__ = ["tool"]


@pytest.fixture
def folder(tool: ModuleType, tmp_path: Path) -> tuple[ModuleType, Any, Any]:
    media = tmp_path / "ひみつの素材置き場"
    media.mkdir()
    video = write_scene_video(media / "secretclip.mp4")
    audio = _write_talk(media / "しゃべりの録音.wav")
    root = tmp_path / "evaldir"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"E01": str(video), "E02": str(audio)}, ensure_ascii=False), encoding="utf-8"
    )
    book_path = tmp_path / "tasks.json"
    book_path.write_text(
        json.dumps(
            {
                "materials": [
                    {"id": "E01", "tasks": ["highlights"]},
                    {"id": "E02", "tasks": ["groups"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    return tool, tool.EvalDir(root), tool.load_tasks(book_path)


def test_range_scores_count_chosen_scenes(tool: ModuleType) -> None:
    """見どころは端の位置より、その場面を選んだかが大事 重なれば当たり

    壊れると、端が 1 秒ずれただけの正しい選びが外れに数えられ、道具の良し悪しが読めない
    """
    scores = tool.range_scores([[1.0, 4.0], [20.0, 22.0]], [[2.0, 5.0], [10.0, 12.0]], 30.0)
    assert scores["recall"] == 0.5
    assert scores["precision"] == 0.5
    assert scores["f1"] == 0.5
    assert tool.range_scores([], [], 30.0)["f1"] == 1.0


def test_highlights_are_answered_from_scratch(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """見どころは場面と同じく、素材をそのまま置いた .sme だけを作り、人の正解ができてから比べる

    道具の下書きから直すと、直し忘れがそのまま正解になって正解が道具の癖に寄る（#345 の決まり）
    """
    tool, eval_dir, book = folder
    assert tool.command_draft(eval_dir, book, ["E01"]) == ["E01 highlights"]
    assert "highlights" not in eval_dir.read_answer("E01").get("tasks", {})
    results = tmp_path / "r.jsonl"
    [record] = tool.command_run(eval_dir, book, ["E01"], results=results, commit="")
    assert record["reason"] == "no-answer"

    # 人が後半を消して前半だけを見どころとして残した
    sheet = eval_dir.draft_path("E01", "highlights")
    document = Document(load_project(sheet))
    clips = [clip for track in document.project.timeline.tracks for clip in track.clips]
    middle = document.project.duration // 2
    document.execute(SplitClips(tuple(clip.id for clip in clips), middle))
    for track in document.project.timeline.tracks:
        for clip in track.clips:
            if clip.timeline_start >= middle:
                document.execute(RemoveClip(clip.id))
    save_project(document.project, sheet)
    tool.command_adopt(eval_dir, "E01", "highlights")
    answer = eval_dir.read_answer("E01")["tasks"]["highlights"]
    assert answer["provisional"] is False
    assert len(answer["ranges"]) == 1
    assert answer["ranges"][0][1] < eval_dir.read_answer("E01")["duration"]

    def deaf(*_args: object) -> None:
        # 音の無い素材を起こすと失敗して、動きで選べる見どころまで asr-error になる
        raise AssertionError("音の無い素材を字幕起こしへ渡した")

    [record] = tool.command_run(
        eval_dir, book, ["E01"], results=results, commit="", transcriber=deaf
    )
    assert record["status"] == "ok"
    assert {"f1", "precision", "recall", "iou"} <= set(record["metrics"])


def test_groups_survive_the_silence_cut(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """道具の段 無音カットで束ねが解けないこと（#427 の分け方を実物の切れ目で見る）

    壊れると、詰めた後にテロップと話し手が別々に動くか、前半と後半が 1 つのグループになる
    """
    tool, eval_dir, book = folder
    results = tmp_path / "r.jsonl"
    [record] = tool.command_run(eval_dir, book, ["E02"], results=results, commit="")
    assert record["status"] == "ok", record
    metrics = record["metrics"]
    assert metrics["cuts"] >= 1
    assert metrics["pairs"] >= 2
    assert metrics["kept_share"] == 1.0
    assert "provisional" not in record


def test_group_metrics_notice_a_dissolved_group(tool: ModuleType, tmp_path: Path) -> None:
    """壊れると、AI がグループを解いても kept_share が 1.0 のままで、解いたことを数えない"""
    from sashimono.core.commands import UngroupClips

    audio = _write_talk(tmp_path / "talk.wav")
    document = tool.grouped_project(audio, "E02")
    before = tool.group_metrics(document.project)
    assert (before["kept_share"], before["pairs"], before["groups"]) == (1.0, 2, 2)
    first = document.project.timeline.tracks[0].clips[0]
    document.execute(UngroupClips((first.id,)))
    after = tool.group_metrics(document.project)
    # 解いた前半の組だけが別々になる 1.0 より下がるだけを見ると 0.25 などの誤りを通す
    assert (after["kept_share"], after["pairs"], after["groups"]) == (0.5, 2, 1)


def test_group_metrics_notice_merged_groups(tool: ModuleType, tmp_path: Path) -> None:
    """前半と後半を 1 つにまとめても重なる組は同じグループのまま 見分けないと 1.0 と書く"""
    from sashimono.core.commands import GroupClips

    audio = _write_talk(tmp_path / "talk.wav")
    document = tool.grouped_project(audio, "E02")
    every = tuple(clip.id for track in document.project.timeline.tracks for clip in track.clips)
    document.execute(GroupClips(every))
    merged = tool.group_metrics(document.project)
    assert merged["merged"] == 1
    assert merged["kept_share"] == 0.0
    assert tool.group_metrics(tool.grouped_project(audio, "E02").project)["merged"] == 0


def test_group_metrics_see_merged_fragments_after_a_split(tool: ModuleType, tmp_path: Path) -> None:
    """まとめた後に前半を割ると、後ろの片割れは新しいグループへ移る 最後の断片だけで決めると
    前の断片のまとまりを見落とし、merged が 0 になる（PR #451 の CodeRabbit の指摘）
    """
    from sashimono.core.commands import GroupClips, SplitClips

    audio = _write_talk(tmp_path / "talk.wav")
    document = tool.grouped_project(audio, "E02")
    every = tuple(clip.id for track in document.project.timeline.tracks for clip in track.clips)
    document.execute(GroupClips(every))
    heading = next(
        clip
        for track in document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.params.get("text") == "前半"
    )
    middle = (heading.timeline_start + heading.timeline_end) // 2
    document.execute(SplitClips((heading.id,), middle))
    assert tool.group_metrics(document.project)["merged"] == 1


def test_the_ai_stage_needs_consent(folder: tuple[ModuleType, Any, Any], tmp_path: Path) -> None:
    """素材の名前・字幕・絵を外へ送るので、同意の印が無ければ 1 行も書かずに止まる"""
    tool, eval_dir, book = folder
    results = tmp_path / "ai.jsonl"
    with pytest.raises(SystemExit, match="allow-send"):
        tool.command_ai_run(eval_dir, book, [], allow_send=False, results=results)
    assert not results.exists()


def test_the_ai_stage_writes_its_counts(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """数えた回数が結果に残り、会話の失敗は ai-error と書いて次へ進む"""
    tool, eval_dir, book = folder
    results = tmp_path / "ai.jsonl"

    def runner(*_args: object, **_kwargs: object) -> tuple[Any, dict[str, float | int]]:
        run = tool.AiRun(tool_calls=5, ungroup_questions=1, questions=1, finished=True)
        metrics = {"kept_share": 1.0, "pairs": 4, "groups": 4, "loose_clips": 0, "clips": 8}
        return run, {**metrics, "removed_seconds": 3.5}

    [record] = tool.command_ai_run(
        eval_dir, book, [], allow_send=True, results=results, runner=runner, commit=""
    )
    assert record["stage"] == "ai"
    assert record["status"] == "ok"
    assert record["metrics"]["ungroup_questions"] == 1
    assert record["metrics"]["kept_share"] == 1.0

    def broken(*_args: object, **_kwargs: object) -> tuple[Any, dict[str, float | int]]:
        return tool.AiRun(failed=True), {}

    [record] = tool.command_ai_run(
        eval_dir, book, [], allow_send=True, results=results, runner=broken, commit=""
    )
    assert (record["status"], record["reason"]) == ("failed", "ai-error")

    def timed_out(*_args: object, **_kwargs: object) -> tuple[Any, dict[str, float | int]]:
        return tool.AiRun(finished=False, failed=False), {}

    # 時間切れを ai-error と書くと、会話の失敗と見分けられない
    [record] = tool.command_ai_run(
        eval_dir, book, [], allow_send=True, results=results, runner=timed_out, commit=""
    )
    assert record["reason"] == "ai-timeout"

    def untouched(*_args: object, **_kwargs: object) -> tuple[Any, dict[str, float | int]]:
        return tool.AiRun(tool_calls=1, finished=True), {"kept_share": 1.0, "removed_seconds": 0}

    # 切らずに終えた回は、グループが残っていて当たり前 成功と並べると見どころが読めない
    [record] = tool.command_ai_run(
        eval_dir, book, [], allow_send=True, results=results, runner=untouched, commit=""
    )
    assert (record["status"], record["reason"]) == ("failed", "ai-no-cut")
    # モデルごとに切らずに終えた回数を比べるので、成功の行と同じくモデルを書く
    assert record["params"] == {"model": "default"}
    assert "しゃべり" not in results.read_text(encoding="utf-8")


class _ScriptedSession:
    """偽の会話 指示を受けたら、AI と同じくブリッジ越しに無音を切ってグループを解こうとする"""

    def __init__(self, bridge: Any, layer_mode: str) -> None:
        del layer_mode
        self._bridge = bridge
        self._events: list[AgentEvent] = []
        self._lock = threading.Lock()

    def _emit(self, event: AgentEvent) -> None:
        with self._lock:
            self._events.append(event)

    def send(self, prompt: str) -> None:
        del prompt

        def work() -> None:
            project = self._bridge._host.document.project
            first = project.timeline.tracks[0].clips[0]
            for name, arguments in (
                ("cut_range", {"start_frame": 0, "end_frame": 10}),
                ("ungroup_clips", {"clip_ids": [str(first.id)]}),
            ):
                operation = find_operation(name)
                assert operation is not None
                self._emit(AgentEvent(EventKind.TOOL_USE, tool=name))
                # 断られた（グループを解かないと答えた）ときも、AI と同じく次へ進む
                with contextlib.suppress(ToolError):
                    self._bridge.invoke(operation, arguments)
            self._emit(AgentEvent(EventKind.TURN_DONE))

        threading.Thread(target=work, daemon=True).start()

    def poll(self) -> list[AgentEvent]:
        with self._lock:
            events, self._events = self._events, []
        return events

    def close(self, *, wait: bool = True) -> None:
        del wait


def test_the_ai_stage_counts_the_ungroup_question(
    tool: ModuleType, tmp_path: Path, qt_application: QApplication
) -> None:
    """本物の編集画面で、AI がグループを解こうとした回数と問いの回数を数え、解かずに残すこと

    壊れると、AI の段が問いに答えられずに止まるか、解いたのに数えずに 1.0 と書く（#437）
    """
    del qt_application
    audio = _write_talk(tmp_path / "talk.wav")
    run, metrics = tool.run_ai_groups(
        audio, tmp_path / "drafts" / "E02_groups_ai.sme", "E02", _ScriptedSession, timeout=60
    )
    assert run.finished
    assert run.ungroup_calls == 1
    assert run.ungroup_questions == 1
    assert run.tool_calls == 2
    assert metrics["kept_share"] == 1.0
    assert metrics["removed_seconds"] > 0
