"""評価の台本（tools/eval_tasks.py #345）の数え方と、結果に名前を混ぜない仕組み

実物の素材は CI に無いので、その場で作る合成の素材（切り替わりの位置が分かっている絵と、
無音の位置が分かっている音）で道具の段を 1 周させる 字幕起こしは偽物に差し替える
"""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from sashimono.core.commands import Document, RemoveMarkers
from sashimono.core.io import load_project, save_project
from sashimono.core.model import Transcript, TranscriptSegment
from tests.scene_video import CUTS, write_scene_video

ROOT = Path(__file__).resolve().parent.parent

#: 合成の音の無音（秒） 0.3 秒の短い間は最短の無音（0.5 秒）より短いので切らない
SILENCE_START, SILENCE_END = 2.0, 4.0
AUDIO_SECONDS = 8.0


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_tasks", ROOT / "tools" / "eval_tasks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_talk(path: Path) -> Path:
    """音・2 秒の無音・音・0.3 秒の間・音 の 8 秒の WAV"""
    rate = 48000
    moments = np.arange(int(AUDIO_SECONDS * rate)) / rate
    signal = 0.5 * np.sin(2 * np.pi * 440 * moments)
    quiet = ((moments >= SILENCE_START) & (moments < SILENCE_END)) | (
        (moments >= 5.0) & (moments < 5.3)
    )
    signal[quiet] = 0.0
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes((signal * 32767).astype("<i2").tobytes())
    return path


def _fake_transcriber(path: Path, params: object) -> Transcript:
    from fractions import Fraction

    return Transcript(
        segments=(
            TranscriptSegment(start=Fraction(0), end=Fraction(2), text="今日は晴れ"),
            TranscriptSegment(start=Fraction(4), end=Fraction(8), text="外へ行きます"),
        )
    )


@pytest.fixture
def folder(tool: ModuleType, tmp_path: Path) -> tuple[ModuleType, Any, Any]:
    """素材 2 本と対応表を置いた評価のフォルダ 素材の名前はわざと照らしやすい名前にする"""
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
                    {"id": "E01", "tasks": ["scenes"]},
                    {"id": "E02", "tasks": ["silence", "talk", "subtitles", "export"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    return tool, tool.EvalDir(root), tool.load_tasks(book_path)


def test_point_scores_drop_by_exactly_one_miss(tool: ModuleType) -> None:
    """場面と区切りの一致率が、ずれた 1 か所の分だけ下がること

    数え方がずれていると、道具を良くしても悪くしても数が動かないか、関係の無い所で動き、
    変更の前後を比べる評価そのものが役に立たない
    """
    answer = [1.0, 2.0, 3.0, 4.0]
    exact = tool.point_scores([1.0, 2.0, 3.0, 4.0], answer, 0.05)
    assert exact["f1"] == 1.0
    shifted = tool.point_scores([1.0, 2.0, 3.0, 4.5], answer, 0.05)
    assert (shifted["hits"], shifted["precision"], shifted["recall"]) == (3, 0.75, 0.75)
    # 許す幅の中のずれは当たり ゆっくりした転換は区間で持ち、区間の中なら当たり
    assert tool.point_scores([1.04], [1.0], 0.05)["f1"] == 1.0
    assert tool.point_scores([5.5], [[5.0, 6.0]], 0.05)["f1"] == 1.0


def test_point_scores_pair_nearest_first(tool: ModuleType) -> None:
    """組める数が一番多くなるように組むこと

    近い組から決めると、正解 2 つに挟まれた位置が近い方に取られ、遠い方と組めたはずの
    隣の位置が外れに数えられて、実際より低い数が出る
    """
    result = tool.point_scores([1.4, 1.9], [1.0, 1.5], 0.5)
    assert result["hits"] == 2


def test_interval_scores_follow_the_shift(tool: ModuleType) -> None:
    """無音の一致率が、残した区間のずれの分だけ下がること

    ずれても 1.0 のままだと、無音の切りすぎ（話の頭が欠ける）が数に出ない
    """
    same = tool.interval_scores([[0.0, 5.0]], [[0.0, 5.0]], 10.0)
    assert (same["iou"], same["agreement"], same["edge_median"]) == (1.0, 1.0, 0.0)
    shifted = tool.interval_scores([[0.0, 4.5]], [[0.0, 5.0]], 10.0)
    assert shifted["iou"] == 0.9
    assert shifted["agreement"] == 0.95
    assert shifted["edge_median"] == 0.5


def test_character_error_rate(tool: ModuleType) -> None:
    """字幕の文字の誤り率が、句読点や空白の違いを数えないこと

    整形の好み（句点の有無）で数が動くと、聞き取りが良くなったのか分からなくなる
    """
    assert tool.character_error_rate("今日は、晴れ。", "今日は晴れ") == 0.0
    assert tool.character_error_rate("今日は雨", "今日は晴れ") == 0.4
    assert tool.character_error_rate("", "") == 0.0


def test_guard_refuses_names_and_free_text(tool: ModuleType, tmp_path: Path) -> None:
    """素材の名前・置き場・日本語の文が混じった結果は、1 行も書かずに止まること

    書けてしまうと、公開のリポジトリに本人の動画の名前や字幕の文が載る
    """
    results = tmp_path / "out.jsonl"
    good = {"material": "E01", "task": "scenes", "status": "ok", "metrics": {"f1": 1.0}}
    named = {**good, "reason": "secretclip"}
    japanese = {**good, "reason": "字幕"}
    pathlike = {**good, "reason": "C:/clips"}
    for bad in (named, japanese, pathlike, {**good, "note": "x"}):
        with pytest.raises(tool.LeakError):
            tool.append_results(results, [good, bad], ["secretclip"])
        assert not results.exists()
    tool.append_results(results, [good], ["secretclip"])
    assert len(tool.read_results(results)) == 1


def test_forbidden_names_cover_folders(tool: ModuleType, tmp_path: Path) -> None:
    """照らす名前に、素材のファイル名と置き場のフォルダ名が入ること

    フォルダ名を照らさないと、置き場の名前（本人の持ち物の分け方）が結果に出ても止まらない
    """
    folder = tool.EvalDir(tmp_path / "evaldir")
    names = tool.forbidden_names(folder, {"E01": tmp_path / "置き場" / "secretclip.mp4"})
    assert {"secretclip.mp4", "secretclip", "置き場", "evaldir"} <= set(names)


def test_tool_stage_round_trip(folder: tuple[ModuleType, Any, Any], tmp_path: Path) -> None:
    """下書き → 道具の段 → 結果 が通り、仮の正解の印が付き、2 回回して同じ数が出ること

    印が無いと、本人が直す前の数（道具と道具を比べた 1.0）を本当の精度と取り違える
    同じ数が出ないと、変更の前後の差が変更によるものか揺れかを見分けられない
    """
    tool, eval_dir, book = folder
    made = tool.command_draft(eval_dir, book, [], transcriber=_fake_transcriber)
    assert set(made) == {"E01 scenes", "E02 silence", "E02 talk", "E02 subtitles"}
    answer = eval_dir.read_answer("E02")
    assert answer["tasks"]["silence"]["provisional"] is True
    assert answer["tasks"]["talk"]["boundaries"] == [3.0]

    results = tmp_path / "results.jsonl"
    heard: list[Path] = []
    exported: list[tuple[Path, int]] = []

    def transcriber(path: Path, params: object) -> Transcript:
        heard.append(path)
        return _fake_transcriber(path, params)

    def exporter(path: Path, transcript: Transcript | None, output: Path) -> dict[str, Any]:
        exported.append((output, 0 if transcript is None else len(transcript.segments)))
        return {"problems": 0, "loudness": None, "complete": True}

    first = tool.command_run(
        eval_dir,
        book,
        [],
        results=results,
        transcriber=transcriber,
        exporter=exporter,
        commit="abc",
    )
    second = tool.command_run(
        eval_dir,
        book,
        [],
        results=results,
        transcriber=transcriber,
        exporter=exporter,
        commit="abc",
    )
    # 字幕と書き出しの両方を持つ素材でも、字幕起こしは 1 回の実行で 1 回だけ
    assert len(heard) == 2
    assert exported[0] == (eval_dir.exports / "E02.mp4", 2)
    by_task = {(r["material"], r["task"]): r for r in first}
    assert by_task["E01", "scenes"]["metrics"]["f1"] == 1.0
    assert by_task["E01", "scenes"]["metrics"]["answer"] == len(CUTS)
    assert by_task["E02", "silence"]["metrics"]["iou"] == 1.0
    assert by_task["E02", "subtitles"]["metrics"]["cer"] == 0.0
    assert by_task["E02", "talk"]["reason"] == "no-tool"
    assert by_task["E02", "export"]["params"] == {"subtitles": True}
    assert "provisional" not in by_task["E02", "export"]
    assert all(r["provisional"] for r in first if r["status"] == "ok" and r["task"] != "export")
    assert [r.get("metrics") for r in first] == [r.get("metrics") for r in second]
    text = results.read_text(encoding="utf-8")
    assert "secretclip" not in text and "evaldir" not in text and "\\u" not in text


def test_adopt_reads_the_corrected_project(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """本人が .sme で直した正解を取り込むと、仮の印が外れ、数がその直しに沿って動くこと

    直した作品を読み違えると、本人の直しの手間が正解に入らず、仮の正解のまま数え続ける
    """
    tool, eval_dir, book = folder
    tool.command_draft(eval_dir, book, ["E01"])
    draft = eval_dir.draft_path("E01", "scenes")
    document = Document(load_project(draft))
    first_marker = document.project.timeline.markers[0].frame
    document.execute(RemoveMarkers((first_marker,)))
    save_project(document.project, draft)

    tool.command_adopt(eval_dir, "E01", "scenes")
    truth = eval_dir.read_answer("E01")["tasks"]["scenes"]
    assert truth["provisional"] is False
    assert truth["cuts"] == pytest.approx(list(CUTS[1:]), abs=0.05)

    [record] = tool.command_run(eval_dir, book, ["E01"], results=tmp_path / "r.jsonl", commit="")
    assert record["provisional"] is False
    assert record["metrics"]["precision"] < 1.0
    assert record["metrics"]["recall"] == 1.0


def test_draft_keeps_human_answers(folder: tuple[ModuleType, Any, Any], tmp_path: Path) -> None:
    """作り直し（--force）でも、本人が直した正解は書き換えないこと

    書き換えると、本人が数時間かけた直しが黙って道具の下書きに戻る
    """
    tool, eval_dir, book = folder
    tool.command_draft(eval_dir, book, ["E01"])
    tool.command_adopt(eval_dir, "E01", "scenes")
    before = eval_dir.read_answer("E01")
    tool.command_draft(eval_dir, book, ["E01"], force=True)
    assert eval_dir.read_answer("E01") == before


def test_report_marks_changed_params(tool: ModuleType) -> None:
    """前回と道具の値や正解の種類が違う回には、表に印が付くこと

    印が無いと、感度を変えて試した回の差を、コードを変えた効果と取り違える
    """
    base = {"stage": "tools", "material": "E01", "task": "scenes", "status": "ok"}
    rows = [
        {**base, "metrics": {"f1": 0.5}, "params": {"threshold": 0.15}, "provisional": True},
        {**base, "metrics": {"f1": 0.75}, "params": {"threshold": 0.25}, "provisional": False},
    ]
    [_, line] = tool.report_lines(rows)
    assert "+0.2500" in line and "値が違う" in line and "正解が変わった" in line
    [_, plain] = tool.report_lines([rows[0], {**rows[0], "metrics": {"f1": 0.5}}])
    assert "値が違う" not in plain and "+0.0000" in plain


def test_eval_dir_must_be_outside_the_repository(tool: ModuleType) -> None:
    """評価のフォルダをリポジトリの中に向けると止まること

    中に置くと、素材の対応表や字幕の文の入った正解をうっかりコミットする
    """
    with pytest.raises(SystemExit):
        tool.eval_dir(str(ROOT / "eval"))


def test_repository_task_book_and_results_carry_no_names(tool: ModuleType) -> None:
    """リポジトリの課題の定義が読めて、足してある結果がどれも形の決まりを守っていること

    定義や結果に素材の名前が入ったままコミットされると、公開の場で本人の動画が分かる
    """
    book = tool.load_tasks()
    assert book.materials
    for row in tool.read_results(tool.RESULTS_FILE):
        tool.guard_record(row, [])


def test_task_book_rejects_names(tool: ModuleType, tmp_path: Path) -> None:
    """課題の定義に番号以外（素材の名前など）を書くと読まずに止まること

    読んでしまうと、名前を書いた定義がそのままリポジトリへ入る
    """
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"materials": [{"id": "録画", "tasks": []}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="番号"):
        tool.load_tasks(path)
    path.write_text(
        json.dumps({"materials": [{"id": "E01", "tasks": [], "name": "x"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="鍵"):
        tool.load_tasks(path)


def test_app_runtime_is_read_from_behind(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """アプリが入れた部品の置き場を、探す道の後ろへ足すこと 別の Python 向けなら足さないこと

    前へ足すと、置き場の別の版の numpy などが .venv の物より先に読まれ、本体の試験と違う
    部品で数える 別の Python 向けの置き場を足すと、字幕起こしを始めた所で import が落ちる
    """
    from sashimono.runtime import ABI_MARKER, python_abi

    monkeypatch.setattr(sys, "path", list(sys.path))
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / ABI_MARKER).write_text(json.dumps({"asr": "cp200"}), encoding="utf-8")
    assert tool.load_app_runtime(stale) is None
    assert str(stale) not in sys.path

    fitting = tmp_path / "fitting"
    fitting.mkdir()
    (fitting / ABI_MARKER).write_text(json.dumps({"asr": python_abi()}), encoding="utf-8")
    first = sys.path[0]
    assert tool.load_app_runtime(fitting) == fitting
    assert sys.path[0] == first
    assert str(fitting) in sys.path
    assert tool.load_app_runtime(tmp_path / "missing") is None


@pytest.mark.usefixtures("gpu", "qt_application")
def test_export_and_check_on_a_short_video(tool: ModuleType, tmp_path: Path) -> None:
    """字幕を焼き込んで書き出し、書き出した動画の検査の数が結果の形で返ること

    書き出しか検査の呼び方が本体とずれると、書き出しの課題が実物の素材で毎回落ち、
    書き出しで起きる問題（黒いコマ・音の欠け）の増減が数に出なくなる
    """
    from tests.check_media import write_video

    source = write_video(tmp_path / "talk.mp4", 8.0, tone_db=-14.0)
    output = tmp_path / "out.mp4"
    metrics = tool.export_and_check(source, _fake_transcriber(source, None), output)
    assert output.exists()
    assert metrics["complete"] is True
    assert metrics["subtitle_clips"] == 2
    assert metrics["count_length"] == 0
    assert abs(float(metrics["length_error"])) < 0.1
    tool.guard_record({"metrics": metrics}, [])
