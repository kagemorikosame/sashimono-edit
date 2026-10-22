"""評価の台本（tools/eval_tasks.py #345）の数え方と、結果に名前を混ぜない仕組み

実物の素材は CI に無いので、その場で作る合成の素材（切り替わりの位置が分かっている絵と、
無音の位置が分かっている音）で道具の段を 1 周させる 字幕起こしは偽物に差し替える
"""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from sashimono.core.commands import AddMarkers, Document
from sashimono.core.io import load_project, save_project
from sashimono.core.model import Transcript, TranscriptSegment
from sashimono.core.model.timeline import Marker
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


def test_interval_scores_without_tool_edges(tool: ModuleType) -> None:
    """道具が無音を 1 つも見つけず端が無くても、数え終わって端のずれを「測れない」にすること

    落ちると、しきい値を変えた回や無音の少ない素材で台本が止まり、同じ回の結果が 1 行も残らない
    0.0 にすると「ずれ無し」と読めてしまい、何も切れていない道具を良いと取り違える
    """
    whole = tool.interval_scores([[0.0, 10.0]], [[0.0, 4.0], [6.0, 10.0]], 10.0)
    assert whole["edge_median"] is None
    assert whole["iou"] == 0.8
    # 正解にも端が無ければ、ずれは無いと言える
    assert tool.interval_scores([[0.0, 10.0]], [[0.0, 10.0]], 10.0)["edge_median"] == 0.0


def test_one_failing_task_keeps_the_other_results(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """書き出しや道具の 1 つが例外を投げても、その課題を failed にして同じ回のほかの行を残すこと

    落ちると、GPU の無い機械や検査の時間切れで、それまで数分かけて数えた結果が全部捨てられる
    例外の文には素材の置き場が混じりうるので、結果には固定の理由だけを書く
    """
    tool, eval_dir, book = folder
    tool.command_draft(eval_dir, book, [], transcriber=_fake_transcriber)
    results = tmp_path / "results.jsonl"
    secret = str(eval_dir.root)

    def exporter(path: Path, transcript: Transcript | None, output: Path) -> dict[str, Any]:
        raise RuntimeError(f"書き出しに失敗 {path} {secret}")

    records = tool.command_run(
        eval_dir,
        book,
        [],
        results=results,
        transcriber=_fake_transcriber,
        exporter=exporter,
        commit="abc",
    )
    by_task = {(r["material"], r["task"]): r for r in records}
    assert by_task["E01", "scenes"]["reason"] == "no-answer"
    assert by_task["E02", "silence"]["status"] == "ok"
    assert by_task["E02", "subtitles"]["status"] == "ok"
    assert (by_task["E02", "export"]["status"], by_task["E02", "export"]["reason"]) == (
        "failed",
        "export-error",
    )
    written = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert len(written) == len(records) == 5
    text = results.read_text(encoding="utf-8")
    assert "secretclip" not in text and "evaldir" not in text and "失敗" not in text


def test_unreadable_material_keeps_the_other_results(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """素材が無い・読めない・正解が壊れていても、その素材を飛ばしてほかの素材の行を残すこと

    落ちると、外付けのドライブを外した・素材の名前を変えたといった 1 本の事情で、
    同じ回のほかの素材の結果まで 1 行も残らず、下書きも途中の素材で止まる
    """
    tool, eval_dir, book = folder
    tool.command_draft(eval_dir, book, [], transcriber=_fake_transcriber)
    manifest = eval_dir.manifest()
    results = tmp_path / "results.jsonl"

    def run() -> dict[tuple[str, str], Any]:
        records = tool.command_run(
            eval_dir,
            book,
            [],
            results=results,
            transcriber=_fake_transcriber,
            exporter=lambda *_: {"problems": 0, "loudness": None, "complete": True},
            commit="abc",
        )
        return {(r["material"], r["task"]): r for r in records}

    # 素材の名前を変えた（対応表には残っている）
    moved = manifest["E01"].with_name("moved.mp4")
    manifest["E01"].rename(moved)
    rows = run()
    assert (rows["E01", "scenes"]["status"], rows["E01", "scenes"]["reason"]) == (
        "skipped",
        "no-material",
    )
    assert rows["E02", "silence"]["status"] == "ok"
    moved.rename(manifest["E01"])

    # 素材はあるが読めない（壊れた動画）
    original = manifest["E01"].read_bytes()
    manifest["E01"].write_bytes(b"not a video")
    rows = run()
    assert (rows["E01", "scenes"]["status"], rows["E01", "scenes"]["reason"]) == (
        "failed",
        "tool-error",
    )
    assert rows["E02", "silence"]["status"] == "ok"
    # 下書きも、読めない素材だけを飛ばしてほかの素材は作る
    made = tool.command_draft(eval_dir, book, [], force=True, transcriber=_fake_transcriber)
    assert "E02 silence" in made and not any(item.startswith("E01") for item in made)
    manifest["E01"].write_bytes(original)

    # 正解のファイルが壊れた 下書きは上書きせずに残す
    broken = eval_dir.answer_path("E01")
    broken.write_text("{", encoding="utf-8")
    rows = run()
    assert (rows["E01", "scenes"]["status"], rows["E01", "scenes"]["reason"]) == (
        "failed",
        "answer-error",
    )
    tool.command_draft(eval_dir, book, [], force=True, transcriber=_fake_transcriber)
    assert broken.read_text(encoding="utf-8") == "{"
    text = results.read_text(encoding="utf-8")
    assert "secretclip" not in text and "moved" not in text


def test_answer_survives_a_crash_while_writing(
    folder: tuple[ModuleType, Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """正解を書く途中で落ちても、前の正解がそのまま読めること

    落ちると、本人が数時間かけて直した正解（同じファイルに入っている）が切れたファイルになり、
    次の回から読めなくなる
    """
    _, eval_dir, _ = folder
    before = {"material": "E01", "tasks": {"scenes": {"provisional": False, "cuts": [1.0]}}}
    eval_dir.write_answer("E01", before)

    def torn_write(self: Path, text: str, *args: Any, **kwargs: Any) -> int:
        # 書き始めて半分で落ちた形 その場で上書きする書き方だと、ここで前の中身が消える
        with self.open("w", encoding="utf-8") as file:
            file.write(text[: len(text) // 2])
        raise OSError("書く途中で落ちた")

    def failed_replace(self: Path, target: Any) -> Path:
        raise OSError("置き換える前に落ちた")

    monkeypatch.setattr(Path, "write_text", torn_write)
    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError):
        eval_dir.write_answer("E01", {"material": "E01", "tasks": {}})
    monkeypatch.undo()
    assert eval_dir.read_answer("E01") == before


def test_failed_transcription_is_not_retried(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """字幕起こしが落ちたら、同じ素材のあとの課題でやり直さず、どれも同じ理由で failed にすること

    落ちると、長い素材で同じ失敗を待つだけの字幕起こしを課題の数だけ走らせて数分を失い、
    書き出しの課題が export-error と出て、GPU を疑って字幕起こしの部品の不具合を見落とす
    """
    tool, eval_dir, book = folder
    tool.command_draft(eval_dir, book, [], transcriber=_fake_transcriber)
    results = tmp_path / "results.jsonl"
    calls: list[Path] = []
    exported: list[Path] = []

    def transcriber(path: Path, params: object) -> Transcript:
        calls.append(path)
        raise OSError(f"字幕起こしに失敗 {path}")

    def exporter(path: Path, transcript: Transcript | None, output: Path) -> dict[str, Any]:
        exported.append(output)
        return {"problems": 0, "loudness": None, "complete": True}

    records = tool.command_run(
        eval_dir,
        book,
        [],
        results=results,
        transcriber=transcriber,
        exporter=exporter,
        commit="abc",
    )
    assert len(calls) == 1
    # 字幕が要る書き出しを、字幕無しで黙って書き出して ok にしない
    assert exported == []
    by_task = {(r["material"], r["task"]): r for r in records}
    assert by_task["E02", "silence"]["status"] == "ok"
    for task in ("subtitles", "export"):
        assert (by_task["E02", task]["status"], by_task["E02", task]["reason"]) == (
            "failed",
            "asr-error",
        )
    text = results.read_text(encoding="utf-8")
    assert "secretclip" not in text and "失敗" not in text


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
    # 名前の照らしは、引数から来うる値（params）と数（metrics）に掛かる
    named = {**good, "params": {"asr_model": "secretclip"}}
    japanese = {**good, "reason": "字幕"}
    pathlike = {**good, "reason": "C:/clips"}
    for bad in (named, japanese, pathlike, {**good, "note": "x"}):
        with pytest.raises(tool.LeakError):
            tool.append_results(results, [good, bad], ["secretclip"])
        assert not results.exists()
    tool.append_results(results, [good], ["secretclip"])
    assert len(tool.read_results(results)) == 1


def test_reason_and_status_outside_the_list_are_refused(tool: ModuleType, tmp_path: Path) -> None:
    """理由・状態・段・課題が台本の決まった値の一覧に無ければ、1 行も書かずに止まること

    これらは名前の照らしを外しているので、一覧で絞らないと、将来ここへ例外の文や素材から
    来る文字を入れたときに素材の名前が公開のリポジトリへ載る
    """
    results = tmp_path / "out.jsonl"
    good = {"material": "E01", "task": "scenes", "status": "ok", "stage": "tools"}
    tool.append_results(results, [{**good, "status": "skipped", "reason": "no-answer"}], [])
    for bad in (
        {"reason": "FileNotFoundError"},
        {"reason": "E01clip"},
        {"status": "done"},
        {"stage": "secret"},
        {"task": "clip2026"},
    ):
        with pytest.raises(tool.LeakError) as caught:
            tool.append_results(results, [good, {**good, **bad}], [])
        assert str(next(iter(bad.values()))) not in str(caught.value)
    assert len(tool.read_results(results)) == 1


def test_reason_list_matches_the_docs(tool: ModuleType) -> None:
    """docs/evaluation.md の「状態と理由」の表と、台本の一覧が同じであること

    食い違うと、結果の行に出た理由を表で引けないか、表にある理由を書こうとして止まる
    """
    text = (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
    section = text.split("### 状態と理由", 1)[1].split("\n#", 1)[0]
    rows = [line.split("|") for line in section.splitlines() if line.startswith("| `")]
    documented = {(cells[2].strip(), cells[1].strip().strip("`")) for cells in rows}
    expected = {("状態", value) for value in tool.STATUSES} | {
        ("理由", value) for value in tool.REASONS
    }
    assert documented == expected


def test_torn_last_result_line_is_skipped_and_mended(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """結果を足す途中で落ちて最後の行が切れても、前の回の行が読め、次に足した行も読めること

    落ちると、切れた 1 行のために report も次の run も止まり、それまでの回の数が見られなくなる
    切れた行に次の行をつなげて書くと、新しく数えた行まで読めなくなる
    """
    results = tmp_path / "out.jsonl"
    good = {"material": "E01", "task": "scenes", "status": "ok", "metrics": {"f1": 1.0}}
    tool.append_results(results, [good, good], [])
    whole = results.read_text(encoding="utf-8")
    results.write_text(whole + whole[:25], encoding="utf-8", newline="\n")
    assert len(tool.read_results(results)) == 2
    assert "1 行飛ばした" in capsys.readouterr().out
    tool.append_results(results, [good], [])
    assert len(tool.read_results(results)) == 3
    assert "飛ばした" not in capsys.readouterr().out
    # 切れてはいないが改行で終わらない最後の行も、次の行とつながらない
    results.write_text(whole.rstrip("\n"), encoding="utf-8", newline="\n")
    tool.append_results(results, [good], [])
    assert len(tool.read_results(results)) == 3


def test_broken_middle_result_line_stops_with_its_number(
    tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """途中の行が壊れた結果は、行の番号だけを出して止まり、上に足さないこと

    黙って飛ばすと、手で書き換えた・ほかの物が書いた壊れに気付かず、前回との差を取り違える
    行の中身を出すと、素材の名前や置き場が画面や記録に出るかもしれない
    """
    results = tmp_path / "out.jsonl"
    good = {"material": "E01", "task": "scenes", "status": "ok", "metrics": {"f1": 1.0}}
    tool.append_results(results, [good], [])
    line = results.read_text(encoding="utf-8")
    broken = line + "{secretclip\n" + line
    results.write_text(broken, encoding="utf-8", newline="\n")
    with pytest.raises(tool.BrokenResultsError, match="2 行目") as caught:
        tool.read_results(results)
    assert "secretclip" not in str(caught.value)
    with pytest.raises(tool.BrokenResultsError):
        tool.append_results(results, [good], [])
    assert results.read_text(encoding="utf-8") == broken
    monkey = pytest.MonkeyPatch()
    monkey.setattr(tool, "RESULTS_FILE", results)
    try:
        assert tool.main(["report"]) == 1
    finally:
        monkey.undo()
    shown = capsys.readouterr().out
    assert "2 行目" in shown and "secretclip" not in shown


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
    # 場面は人が一から付けるまで正解が無い（道具と道具を比べた数を残さない）
    assert by_task["E01", "scenes"]["reason"] == "no-answer"
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
    # 本人が素材を見て、最初の切り替わりを見落として目印を置いた形
    document = Document(load_project(draft))
    fps = float(document.project.rate.fps)
    marks = tuple(Marker(frame=round(cut * fps), label="場面") for cut in CUTS[1:])
    document.execute(AddMarkers(marks))
    save_project(document.project, draft)

    tool.command_adopt(eval_dir, "E01", "scenes")
    truth = eval_dir.read_answer("E01")["tasks"]["scenes"]
    assert truth["provisional"] is False
    assert truth["cuts"] == pytest.approx(list(CUTS[1:]), abs=0.05)

    [record] = tool.command_run(eval_dir, book, ["E01"], results=tmp_path / "r.jsonl", commit="")
    assert record["provisional"] is False
    assert record["metrics"]["precision"] < 1.0
    assert record["metrics"]["recall"] == 1.0


def test_adopt_without_draft_scores_with_the_measured_length(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """下書きを通さずに取り込んだ正解でも、素材から測った長さで数え、正解にも長さを残すこと

    落ちると、長さ 0 の素材として数えて無音の一致率が意味を失い、誤った数が ok のまま残る
    """
    tool, eval_dir, book = folder
    tool.command_draft(eval_dir, book, ["E02"], transcriber=_fake_transcriber)
    # 別の場所で直した .sme だけを持ってきた形 正解のファイルはまだ無い
    elsewhere = tmp_path / "fixed.sme"
    eval_dir.draft_path("E02", "silence").replace(elsewhere)
    eval_dir.answer_path("E02").unlink()

    tool.command_adopt(eval_dir, "E02", "silence", elsewhere)
    answer = eval_dir.read_answer("E02")
    assert answer["duration"] == pytest.approx(AUDIO_SECONDS, abs=0.05)

    # 正解のファイルの長さが無くても（前の版で取り込んだ正解）、その回に測った長さで数える
    del answer["duration"], answer["fps"]
    eval_dir.write_answer("E02", answer)
    records = tool.command_run(
        eval_dir,
        book,
        ["E02"],
        results=tmp_path / "r.jsonl",
        transcriber=_fake_transcriber,
        exporter=lambda *_: {"problems": 0, "loudness": None, "complete": True},
        commit="",
    )
    silence = next(r for r in records if r["task"] == "silence")
    assert silence["metrics"]["answer_kept_seconds"] > AUDIO_SECONDS - 3.0
    assert silence["metrics"]["iou"] == 1.0
    assert silence["metrics"]["edge_median"] == 0.0


def test_results_are_written_when_folders_share_names_with_the_tool(
    tool: ModuleType, tmp_path: Path
) -> None:
    """置き場のフォルダ名が年（2026）や段・課題の名前（tools export talk）でも、結果を書けること

    落ちると、年ごとのフォルダに素材を置いた人は毎回、数分走らせたあとに 1 行も書けずに止まり、
    素材の名前ではない値を「名前が混じった」と言われて直し方も分からない
    """
    year = datetime.now(UTC).strftime("%Y")
    media = tmp_path / year / "export" / "tools"
    media.mkdir(parents=True)
    audio = _write_talk(media / "talk.wav")
    root = tmp_path / "evaldir"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"E02": str(audio)}), encoding="utf-8")
    book_path = tmp_path / "tasks.json"
    book_path.write_text(
        json.dumps(
            {"materials": [{"id": "E02", "tasks": ["silence", "talk", "subtitles", "export"]}]}
        ),
        encoding="utf-8",
    )
    eval_dir, book = tool.EvalDir(root), tool.load_tasks(book_path)
    tool.command_draft(eval_dir, book, [], transcriber=_fake_transcriber)
    results = tmp_path / "results.jsonl"
    records = tool.command_run(
        eval_dir,
        book,
        [],
        results=results,
        transcriber=_fake_transcriber,
        exporter=lambda *_: {"problems": 0, "loudness": None, "complete": True},
    )
    assert {r["task"]: r["status"] for r in records}["silence"] == "ok"
    assert len(tool.read_results(results)) == len(records) == 4
    # 引数から来る値に置き場の名前が混じれば、今までどおり止める
    with pytest.raises(tool.LeakError):
        tool.command_run(
            eval_dir,
            book,
            [],
            results=results,
            transcriber=_fake_transcriber,
            params=tool.ToolParams(asr_model=f"m{year}"),
            exporter=lambda *_: {"problems": 0, "loudness": None, "complete": True},
        )


def test_scenes_are_answered_from_scratch_by_a_person(
    folder: tuple[ModuleType, Any, Any], tmp_path: Path
) -> None:
    """場面の課題は目印の無い .sme だけを作り、仮の正解を持たず、run は no-answer で飛ばすこと

    道具の下書きを仮の正解にすると、人が直す前の回は道具と道具を比べて F1 が実際より高く出て、
    直し忘れがそのまま正解になり、道具を良くしても数が上がらない（#345 の決まり）
    """
    tool, eval_dir, book = folder
    made = tool.command_draft(eval_dir, book, ["E01"])
    assert made == ["E01 scenes"]
    sheet = eval_dir.draft_path("E01", "scenes")
    assert load_project(sheet).timeline.markers == ()
    assert "scenes" not in eval_dir.read_answer("E01").get("tasks", {})
    results = tmp_path / "r.jsonl"
    [record] = tool.command_run(eval_dir, book, ["E01"], results=results, commit="")
    assert (record["status"], record["reason"]) == ("skipped", "no-answer")

    # 前の版の draft が残した道具の仮の正解とも比べず、次の draft で外す
    stale = {"provisional": True, "source": "draft", "cuts": list(CUTS)}
    eval_dir.write_answer("E01", {"material": "E01", "tasks": {"scenes": stale}})
    [record] = tool.command_run(eval_dir, book, ["E01"], results=results, commit="")
    assert record["reason"] == "no-answer"

    # 目印を置いている途中の .sme は、--force でなければ上書きしない
    document = Document(load_project(sheet))
    document.execute(AddMarkers((Marker(frame=30, label="場面"),)))
    save_project(document.project, sheet)
    assert tool.command_draft(eval_dir, book, ["E01"]) == []
    assert "scenes" not in eval_dir.read_answer("E01")["tasks"]
    assert len(load_project(sheet).timeline.markers) == 1


def test_results_file_does_not_mark_the_commit_dirty(tool: ModuleType, tmp_path: Path) -> None:
    """台本が足した結果のファイルだけが変わった回は +dirty にならず、コードが変われば付くこと

    落ちると、結果をコミットせずに 2 回目を回すだけで印が付き、コードを変えた回と
    変えていない回を見分けられなくなる
    """
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    results = tmp_path / "eval" / "results" / "tools.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text("", encoding="utf-8")
    code = tmp_path / "code.py"
    code.write_text("x = 1\n", encoding="utf-8")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "first")

    results.write_text('{"a": 1}\n', encoding="utf-8")
    assert not tool.current_commit(tmp_path, results).endswith("+dirty")
    code.write_text("x = 2\n", encoding="utf-8")
    assert tool.current_commit(tmp_path, results).endswith("+dirty")


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
    base = {"stage": "tools", "material": "E01", "task": "silence", "status": "ok"}
    rows = [
        {**base, "metrics": {"iou": 0.5}, "params": {"padding": 0.1}, "provisional": True},
        {**base, "metrics": {"iou": 0.75}, "params": {"padding": 0.2}, "provisional": False},
    ]
    [_, line] = tool.report_lines(rows)
    assert "+0.2500" in line and "値が違う" in line and "正解が変わった" in line
    [_, plain] = tool.report_lines([rows[0], {**rows[0], "metrics": {"iou": 0.5}}])
    assert "値が違う" not in plain and "+0.0000" in plain and "仮" in plain


def test_report_ignores_provisional_scene_rows(tool: ModuleType) -> None:
    """場面の課題では、道具の下書きを仮の正解にして数えた行を表の数に使わないこと

    使うと、人が一から付けた正解で数えるまで、道具と道具を比べた 1.0 に近い数が今の精度に
    見え続ける 記録の行は消さずに残すので、表の側で外す
    """
    base = {"stage": "tools", "material": "E09", "task": "scenes", "status": "ok"}
    tool_vs_tool = {**base, "metrics": {"f1": 1.0}, "provisional": True}
    [_, waiting] = tool.report_lines([tool_vs_tool, tool_vs_tool])
    assert "まだ人の正解が無い" in waiting and "f1=" not in waiting
    skipped = {**base, "status": "skipped", "reason": "no-answer"}
    [_, still] = tool.report_lines([tool_vs_tool, skipped])
    assert "まだ人の正解が無い" in still
    human = {**base, "metrics": {"f1": 0.8}, "provisional": False}
    [_, line] = tool.report_lines([tool_vs_tool, human, tool_vs_tool])
    # 前回の数も人の正解の行から取る 仮の行との差を出すと、精度が落ちたように見える
    assert "f1=0.8000" in line and "人" in line and "正解が変わった" not in line


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
