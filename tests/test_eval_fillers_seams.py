"""評価の台本に足した課題（#414 の 4 言いよどみを見つける・5 定点の映像のつなぎ目）

実物の素材と字幕起こしの部品は CI に無いので、合成の音と偽の字幕起こしで 1 周させる
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sashimono.core.commands import AddMarkers, Document, RemoveMarkers
from sashimono.core.io import load_project, save_project
from sashimono.core.model import Project, Transcript, TranscriptSegment, Word
from sashimono.core.model.timeline import Marker
from tests.test_eval_tasks_tool import _write_talk, tool

__all__ = ["tool"]

F = Fraction


def _words(path: Path, params: object) -> Transcript:
    """「えーと今日は」と「あのー外へ」を 1 字ずつの語で返す（Whisper の日本語の形）"""

    def line(start: float, end: float, *items: tuple[str, float, float]) -> TranscriptSegment:
        words = tuple(
            Word(F(a).limit_denominator(100), F(b).limit_denominator(100), text)
            for text, a, b in items
        )
        return TranscriptSegment(
            F(start).limit_denominator(100),
            F(end).limit_denominator(100),
            "".join(word.text for word in words),
            words=words,
        )

    return Transcript(
        segments=(
            line(0.2, 1.6, ("え", 0.2, 0.4), ("ー", 0.4, 0.6), ("と", 0.6, 0.8), ("今", 1.0, 1.6)),
            line(4.5, 6.0, ("あ", 4.5, 4.7), ("の", 4.7, 4.9), ("ー", 4.9, 5.1), ("外", 5.4, 6.0)),
        ),
        language="ja",
    )


@pytest.fixture
def folder(tool: ModuleType, tmp_path: Path) -> tuple[ModuleType, Any, Any]:
    media = tmp_path / "ひみつの素材置き場"
    media.mkdir()
    audio = _write_talk(media / "しゃべりの録音.wav")
    root = tmp_path / "evaldir"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"E02": str(audio)}, ensure_ascii=False), encoding="utf-8"
    )
    book_path = tmp_path / "tasks.json"
    book_path.write_text(
        json.dumps({"materials": [{"id": "E02", "tasks": ["fillers", "seams"]}]}),
        encoding="utf-8",
    )
    return tool, tool.EvalDir(root), tool.load_tasks(book_path)


def test_the_filler_draft_marks_what_the_tool_found(folder: tuple[ModuleType, Any, Any]) -> None:
    """言いよどみの下書きは、道具が見つけた頭に目印を置き、探す手がかりの字幕も付ける

    壊れると、本人が直す .sme に目印か字幕が無く、53 分を一から聞き直すことになる
    """
    tool, eval_dir, book = folder
    made = tool.command_draft(eval_dir, book, ["E02"], word_transcriber=_words)
    assert made == ["E02 fillers"]
    answer = eval_dir.read_answer("E02")["tasks"]["fillers"]
    assert answer["provisional"] is True
    assert answer["points"] == [0.2, 4.5]
    project = load_project(eval_dir.draft_path("E02", "fillers"))
    assert [marker.label for marker in project.timeline.markers] == ["言いよどみ"] * 2
    (media,) = project.media
    transcript = media.transcript_for(None)
    assert transcript is not None
    assert any(segment.words for segment in transcript.segments)


def test_the_filler_answer_comes_from_the_fixed_markers(
    folder: tuple[ModuleType, Any, Any],
    tmp_path: Path,
) -> None:
    """本人が目印を消し・足した .sme を正解にして、道具の頭と比べる

    壊れると、直した目印が正解に入らず、道具の下書きと道具を比べた 1.0 が精度に見え続ける
    """
    tool, eval_dir, book = folder
    tool.command_draft(eval_dir, book, ["E02"], word_transcriber=_words)
    sheet = eval_dir.draft_path("E02", "fillers")
    document = Document(load_project(sheet))
    project: Project = document.project
    # 2 つ目（あのー）は言いよどみではなかったので消し、道具が見落とした 7 秒の「えー」を足した
    second = project.timeline.markers[1]
    document.execute(RemoveMarkers((second.frame,)))
    document.execute(AddMarkers((Marker(frame=round(7.0 * float(project.rate.fps))),)))
    save_project(document.project, sheet)
    tool.command_adopt(eval_dir, "E02", "fillers")
    answer = eval_dir.read_answer("E02")["tasks"]["fillers"]
    assert answer["provisional"] is False
    assert answer["points"] == pytest.approx([0.2, 7.0], abs=0.05)

    [record] = tool.command_run(
        eval_dir,
        book,
        ["E02"],
        results=tmp_path / "r.jsonl",
        commit="",
        word_transcriber=_words,
        seam_exporter=lambda _project, _output: None,
    )[:1]
    assert record["status"] == "ok", record
    metrics = record["metrics"]
    assert (metrics["hits"], metrics["precision"], metrics["recall"]) == (1, 0.5, 0.5)
    assert tool.HEADLINE[tool.FILLERS] == "f1"


def test_the_seams_are_written_once_per_style(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """無音を詰めたつなぎ目を、何もしない・クロスフェード・寄りの 3 通りで書き出す

    壊れると、本人が見比べる物の 1 つが無いか、どれも同じ処理のまま書き出される
    """
    tool, eval_dir, book = folder
    written: list[str] = []

    def export(project: Project, output: Path) -> None:
        written.append(output.name)

    records = tool.command_run(
        eval_dir,
        book,
        ["E02"],
        results=tmp_path / "r.jsonl",
        commit="",
        word_transcriber=_words,
        seam_exporter=export,
    )
    record = next(row for row in records if row["task"] == "seams")
    assert record["status"] == "ok", record
    assert written == [f"E02_seams_{video}.mp4" for video in ("none", "crossfade", "zoom")]
    metrics = record["metrics"]
    assert metrics["seams"] >= 1
    assert metrics["removed_seconds"] > 0
    assert "provisional" not in record
    assert record["params"]["threshold_db"] == -40.0
