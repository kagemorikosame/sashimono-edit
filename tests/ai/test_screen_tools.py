"""AI の道具 detect_text と detect_faces（#324 A6）

画面の中の文字と顔の位置を、タイムラインのフレームと一緒に返すこと・部品が無ければ理由と
入れ方を返して断ること・調べ終えていなければ AI のスレッドで待つこと・外へ送る物の宣言を見る
調べる係は偽物（試験が様子を入れる） 本物の係は ``tests/engine/test_screen_analyzer.py`` が見る
"""

from __future__ import annotations

from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.ledger import AI_ONLY
from sashimono.ai.operations import (
    DEFAULT_FACE_SCORE,
    Operation,
    Pending,
    find_operation,
)
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.model import MediaItem
from sashimono.engine.cache import ScreenState
from sashimono.engine.vision import (
    Box,
    Face,
    ScreenKind,
    ScreenReport,
    ScreenSample,
    TextLine,
    parts_note,
)
from sashimono.engine.vision.parts import rapid_note
from tests.ai.conftest import FakeHost, allow_sending
from tests.ai.test_bridge import call_in_thread, pump_until


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


TEXT = _operation("detect_text")
FACES = _operation("detect_faces")


def _media(host: FakeHost) -> MediaItem:
    return host.project.media[0]


def _report(kind: ScreenKind, *samples: ScreenSample, complete: bool = True) -> ScreenReport:
    return ScreenReport(
        kind,
        "試験",
        1920,
        1080,
        samples,
        planned=len(samples) + (0 if complete else 3),
        complete=complete,
    )


def _set(host: FakeHost, kind: ScreenKind, state: ScreenState) -> None:
    host.screen_states[(_media(host).id, kind)] = state


def _call(operation: Operation, host: FakeHost, **arguments: Any) -> dict[str, Any]:
    result = operation(host, {"media_id": str(_media(host).id), **arguments})
    assert isinstance(result, dict), result
    return result


TELOP = TextLine("本日のおすすめ", Box(-400.0, 450.0, 600.0, 80.0))


class TestDeclarations:
    def test_both_only_read(self) -> None:
        # 壊れると（変える道具にすると）、文字や顔を調べるたびに許可を求められる
        assert TEXT.safety is Safety.READ_ONLY
        assert FACES.safety is Safety.READ_ONLY

    def test_the_read_text_is_asked_before_sending(self) -> None:
        # 壊れると、画面に映った本人の文（資料・看板）を、尋ねずに外へ送る（#342）
        assert Outbound.TEXT in TEXT.sends
        # 絵は手元で調べて送らない 絵を送ると書くと、絵を断った人に道具が動かない
        assert Outbound.FRAMES not in TEXT.sends
        assert Outbound.FRAMES not in FACES.sends
        # 顔の位置は数だけ 字幕と文字を断った人にも動く
        assert Outbound.TEXT not in FACES.sends

    def test_the_ledger_knows_both(self) -> None:
        # 壊れると、操作の台帳の見張り（#347）が道具の足し忘れを見逃す
        assert "detect_text" in AI_ONLY
        assert "detect_faces" in AI_ONLY


class TestText:
    def test_the_lines_come_with_their_place_and_frame(self, host: FakeHost) -> None:
        # 壊れると、AI が画面のどこに文字があるかを知らずに、字幕をテロップの上へ重ねる
        _set(
            host,
            ScreenKind.TEXT,
            ScreenState(
                _report(ScreenKind.TEXT, ScreenSample(2.0, lines=(TELOP,)), ScreenSample(5.0)),
                running=False,
            ),
        )
        result = _call(TEXT, host)
        assert host.screen_requests == [(_media(host).id, ScreenKind.TEXT)]
        assert result["complete"] is True
        # 何も読めなかった絵（5 秒）は返さない
        assert [row["source_seconds"] for row in result["samples"]] == [2.0]
        line = result["samples"][0]["lines"][0]
        assert line == {
            "text": "本日のおすすめ",
            "x": -400.0,
            "y": 450.0,
            "width": 600.0,
            "height": 80.0,
        }
        # 素材の 2 秒は、頭に置いたクリップの 60 フレーム目
        assert result["samples"][0]["timeline"][0]["frame"] == 60
        assert (result["frame_width"], result["frame_height"]) == (1920, 1080)
        assert "上が正" in result["coordinates"]

    def test_the_result_says_which_ocr_read_it(self, host: FakeHost) -> None:
        # 壊れると、飾った文字が読めていないときに、AI が Windows の OCR で読んだからだと分からず、
        # 本人に設定の切り替えを案内できない
        sample = ScreenSample(2.0, lines=(TELOP,))
        for engine, way in (("rapidocr:3.10.0", "rapidocr"), ("windows-ocr:ja", "windows")):
            report = ScreenReport(ScreenKind.TEXT, engine, 640, 360, (sample,), 1, complete=True)
            _set(host, ScreenKind.TEXT, ScreenState(report, running=False))
            assert _call(TEXT, host)["ocr"] == way
        _set(host, ScreenKind.FACES, ScreenState(_report(ScreenKind.FACES), running=False))
        assert "ocr" not in _call(FACES, host)

    def test_a_missing_rapidocr_explains_the_choice(self, host: FakeHost) -> None:
        # 壊れると、RapidOCR を選んで入れていない人の所で、AI が設定を戻すか入れるかを案内できない
        note = rapid_note("飾った文字の読み取り（RapidOCR）の部品が入っていません")
        _set(host, ScreenKind.TEXT, ScreenState(None, running=False, failure=note))
        with pytest.raises(ToolError, match="Windows の OCR"):
            _call(TEXT, host)

    def test_a_range_narrows_the_samples(self, host: FakeHost) -> None:
        # 壊れると、長い素材で一部だけを聞いても全部が返り、返事の上限で切れる
        samples = tuple(ScreenSample(float(s), lines=(TELOP,)) for s in range(1, 9))
        _set(host, ScreenKind.TEXT, ScreenState(_report(ScreenKind.TEXT, *samples), running=False))
        result = _call(TEXT, host, from_seconds=3, to_seconds=5)
        assert [row["source_seconds"] for row in result["samples"]] == [3.0, 4.0]

    def test_missing_parts_explain_how_to_install(self, host: FakeHost) -> None:
        # 壊れると、部品の無い機械で AI が理由を知らず、何度も呼び直す
        note = parts_note("Windows の OCR を呼ぶ部品が入っていません")
        _set(host, ScreenKind.TEXT, ScreenState(None, running=False, failure=note))
        with pytest.raises(ToolError, match="画面の解析の部品"):
            _call(TEXT, host)

    def test_a_picture_less_material_is_refused(self, host: FakeHost) -> None:
        # 壊れると、音だけの素材の文字を頼まれて、調べる係が絵を探して失敗を積む
        from dataclasses import replace

        media = _media(host)
        silent = replace(media, video_streams=())
        host.document.reset(replace(host.document.project, media=(silent,)))
        with pytest.raises(ToolError, match="映像がありません"):
            _call(TEXT, host)
        assert host.screen_requests == []

    def test_an_unknown_material_is_refused(self, host: FakeHost) -> None:
        # 壊れると、存在しない素材の文字を「無い」と答え、AI が取り違えに気付かない
        with pytest.raises(ToolError):
            TEXT(host, {"media_id": "m_無い"})

    def test_no_text_is_said_plainly(self, host: FakeHost) -> None:
        # 壊れると、空の一覧だけが返り、AI が調べ損ねたのか文字が無いのかを見分けられない
        _set(
            host,
            ScreenKind.TEXT,
            ScreenState(_report(ScreenKind.TEXT, ScreenSample(1.0)), running=False),
        )
        result = _call(TEXT, host)
        assert result["samples"] == []
        assert "読めた絵はありません" in result["note"]


class TestFaces:
    def test_faces_below_the_score_are_left_out(self, host: FakeHost) -> None:
        # 壊れると、壁の模様などの確からしさの低い候補まで顔として返し、縦の枠が外れる
        sample = ScreenSample(
            3.0,
            faces=(
                Face(Box(100.0, 50.0, 200.0, 240.0), 0.97),
                Face(Box(-600.0, -300.0, 40.0, 50.0), 0.6),
            ),
        )
        _set(host, ScreenKind.FACES, ScreenState(_report(ScreenKind.FACES, sample), running=False))
        result = _call(FACES, host)
        assert result["min_score"] == DEFAULT_FACE_SCORE
        assert [face["x"] for face in result["samples"][0]["faces"]] == [100.0]
        lowered = _call(FACES, host, min_score=0.5)
        assert len(lowered["samples"][0]["faces"]) == 2

    def test_a_score_out_of_range_is_refused(self, host: FakeHost) -> None:
        # 壊れると、0.1 を黙って 0.5 に寄せ、AI は細かく拾ったつもりで少ない一覧を読む
        with pytest.raises(ToolError, match="min_score"):
            _call(FACES, host, min_score=0.1)

    def test_a_picture_without_faces_is_still_listed(self, host: FakeHost) -> None:
        # 壊れると、顔の無い場面が一覧から消え、AI が「顔が無い」ことと「調べていない」ことを
        # 見分けられない
        _set(
            host,
            ScreenKind.FACES,
            ScreenState(_report(ScreenKind.FACES, ScreenSample(1.0)), running=False),
        )
        result = _call(FACES, host)
        assert result["samples"][0]["faces"] == []

    def test_the_text_and_faces_are_asked_separately(self, host: FakeHost) -> None:
        # 壊れると、顔だけを頼んだのに文字まで調べ、待つ時間が倍になる
        _call(FACES, host, wait_seconds=0)
        assert host.screen_requests == [(_media(host).id, ScreenKind.FACES)]


class TestWaiting:
    def test_it_waits_on_the_ai_thread_and_returns_the_result(self, host: FakeHost) -> None:
        # 壊れると、画面のスレッドで待って編集画面が固まるか、調べ終える前の空の一覧を返す
        media = _media(host)
        _set(host, ScreenKind.TEXT, ScreenState(None, running=True))
        first = TEXT.start(host, {"media_id": str(media.id), "wait_seconds": 5})
        assert isinstance(first, Pending)
        assert first.seconds == 5

        def ready(_media: object, _kind: object) -> bool:
            _set(
                host,
                ScreenKind.TEXT,
                ScreenState(
                    _report(ScreenKind.TEXT, ScreenSample(2.0, lines=(TELOP,))), running=False
                ),
            )
            return True

        host.screen_ready = ready
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, TEXT, {"media_id": str(media.id), "wait_seconds": 5})
        pump_until(bridge, done)
        result = box.get("result")
        assert isinstance(result, dict), box
        assert result["complete"] is True
        assert host.screen_waits and host.screen_waits[0][1] is ScreenKind.TEXT

    def test_a_partial_result_says_how_to_get_the_rest(self, host: FakeHost) -> None:
        # 壊れると、途中までの一覧を全部だと思い、後ろの場面のテロップが無いことにされる
        partial = _report(ScreenKind.TEXT, ScreenSample(1.0, lines=(TELOP,)), complete=False)
        _set(host, ScreenKind.TEXT, ScreenState(partial, running=True))
        result = _call(TEXT, host, wait_seconds=0)
        assert result["complete"] is False
        assert "同じ引数で呼ぶ" in result["note"]
        assert result["examined"] == 1 and result["planned"] == 4

    def test_waiting_for_the_scenes_first_is_explained(self, host: FakeHost) -> None:
        # 壊れると、場面の解析を待っている間に「何も無い」と答え、AI が文字の無い素材と思い込む
        _set(host, ScreenKind.FACES, ScreenState(None, running=True))
        result = _call(FACES, host, wait_seconds=0)
        assert result["complete"] is False
        assert "場面の切り替わり" in result["note"]

    def test_a_removed_material_while_waiting_is_reported(self, host: FakeHost) -> None:
        # 壊れると、待っている間に外した素材の結果を返し、一覧に無い素材の文字を AI が使う
        from sashimono.core.commands import RemoveClip, RemoveMedia

        media = _media(host)
        _set(host, ScreenKind.TEXT, ScreenState(None, running=True))
        pending = TEXT.start(host, {"media_id": str(media.id), "wait_seconds": 5})
        assert isinstance(pending, Pending)
        clips = [clip.id for track in host.project.timeline.tracks for clip in track.clips]
        host.apply_commands([RemoveClip(clip) for clip in clips] + [RemoveMedia(media.id)], "外す")
        with pytest.raises(ToolError, match="見つかりません"):
            pending.then()
