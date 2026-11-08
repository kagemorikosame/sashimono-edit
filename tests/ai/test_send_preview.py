"""送る前の確かめの下見と、送った物の記録（#425）"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import OPERATIONS, ImageResult, Operation
from sashimono.ai.outbound import (
    ASKED,
    SAMPLE_LINE_CHARS,
    SAMPLE_LINES,
    Outbound,
    PreviewSource,
    Sample,
    SendAnswer,
    SendChoice,
    SendRequest,
    build_samples,
    file_sample,
    frame_sample,
    from_arguments,
    from_result,
    sendable_now,
    text_sample,
)
from sashimono.ai.send_log import KEEP_DAYS, SendLog, SendLogStore, SentEntry, SentKind
from sashimono.core.model import MediaItem, Transcript
from tests.ai.conftest import FakeHost, make_loaded
from tests.ai.test_bridge import call_in_thread
from tests.ai.test_outbound import SECRET_PATH, _operation, _run, answer_sends


@pytest.fixture
def secret_host(video_media: MediaItem, transcript: Transcript) -> FakeHost:
    """素材の場所に本人の名前が入った作品"""
    return FakeHost(make_loaded(replace(video_media, path=SECRET_PATH), transcript))


# --- 宣言 ---


class TestDeclaration:
    def test_a_tool_that_sends_asked_kinds_must_say_how_to_preview(self) -> None:
        # 壊れると、字幕や絵を返す道具を足した人が書き忘れても黙って通り、窓に中身が出ない
        with pytest.raises(ValueError, match="preview"):
            Operation(
                name="forgot",
                description="",
                schema={},
                handler=lambda _host, _arguments: None,
                sends=frozenset({Outbound.TEXT}),
            )

    def test_a_writing_tool_cannot_preview_from_its_result(self) -> None:
        # 壊れると、作品を変える道具を確かめの前に動かし、断っても作品が変わったままになる
        with pytest.raises(ValueError, match="読み取りだけ"):
            Operation(
                name="writes",
                description="",
                schema={},
                handler=lambda _host, _arguments: None,
                writes=True,
                sends=frozenset({Outbound.TEXT}),
                preview=from_result(),
            )

    def test_every_tool_with_asked_kinds_declares_it(self) -> None:
        # 作る所で落ちるので念のため 読み取りだけの道具の結果から下見を作れるかも見る
        for operation in OPERATIONS:
            if not operation.asked_kinds:
                continue
            assert operation.preview is not None, operation.name
            if operation.preview.source is PreviewSource.RESULT:
                assert not operation.writes, operation.name

    @pytest.mark.parametrize("name", ["copy_snapshot", "copy_subtitle_text"])
    def test_clipboard_tools_are_not_run_before_asking(self, name: str) -> None:
        # 壊れると、断ってもクリップボードへ絵や字幕が写ってしまう（writes が偽でも外へ出す）
        preview = _operation(name).preview
        assert preview is not None
        assert preview.source is PreviewSource.ARGUMENTS


# --- 下見の作り方 ---


class TestSamples:
    def test_text_counts_lines_and_characters_under_the_named_fields(self) -> None:
        # 壊れると、ID や秒まで文字に数え、送る文の量が読めない
        value = {"rows": [{"id": "s_1", "text": "今日は\n晴れ"}, {"id": "s_2", "text": "です"}]}
        sample = text_sample(value, frozenset({"text"}))
        assert sample.count == 3
        assert sample.chars == len("今日は晴れです")
        assert sample.lines == ("今日は", "晴れ", "です")
        assert sample.amount == "3 行 7 文字"

    def test_long_text_is_cut_but_counted_in_full(self) -> None:
        # 壊れると、字幕の多い作品で窓が重くなるか、数が下見の分だけになり少なく見える
        rows = [{"text": "あ" * (SAMPLE_LINE_CHARS + 50)} for _ in range(SAMPLE_LINES + 10)]
        sample = text_sample(rows, frozenset({"text"}))
        assert sample.count == SAMPLE_LINES + 10
        assert len(sample.lines) == SAMPLE_LINES
        assert sample.omitted == 10
        assert all(len(line) == SAMPLE_LINE_CHARS for line in sample.lines)

    def test_frames_carry_the_picture_that_is_sent(self) -> None:
        # 壊れると、窓に出す絵と送る絵が別になり、どのコマが出るかを見誤る
        image = ImageResult(png=b"png", caption="12.0 秒のコマ")
        sample = frame_sample(image)
        assert sample.images == (b"png",)
        assert sample.lines == ("12.0 秒のコマ",)
        assert sample.amount == "1 枚"

    def test_files_show_before_and_after_hiding(self) -> None:
        # 壊れると、断ったときに送る伏せ字の形が見えず、何が残るかを確かめられない
        pairs = [(str(SECRET_PATH), "〔素材 m_1〕")]
        sample = file_sample({"path": str(SECRET_PATH), "count": 1}, pairs)
        assert sample.count == 1
        (line,) = sample.lines
        assert str(SECRET_PATH) in line
        assert line.endswith("〔ファイルの場所〕") or line.endswith("〔素材 m_1〕")

    def test_files_warn_about_the_user_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 壊れると、パスにユーザー名が入っていることに気付かずに送る
        home = tmp_path / "利用者"
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
        place = str(home / "動画" / "a.mp4")
        sample = file_sample({"path": place}, [])
        assert "利用者" in sample.note

    def test_a_preview_from_the_arguments_says_so(self) -> None:
        # 壊れると、引数だけの下見を、結果まで含めた中身と思って答えてしまう
        (sample,) = build_samples([Outbound.TEXT], {"text": "字幕"}, from_arguments())
        assert sample.lines == ("字幕",)
        assert "引数" in sample.note


# --- 先に動かして保留する ---


class TestHeldResult:
    def test_the_subtitle_preview_matches_what_is_sent(self, secret_host: FakeHost) -> None:
        # 壊れると、窓に出した行と、実際に AI へ返す字幕が食い違う
        bridge = EditorBridge(secret_host)
        seen, box = _run(bridge, "get_subtitles", lambda _request: SendAnswer.UNTIL_CLOSE)
        (request,) = seen
        (sample,) = request.samples
        result = box["result"]
        expected = text_sample(result, frozenset({"text"}))
        assert sample == expected
        assert sample.lines == ("今日は", "編集ソフトを", "作ります")
        assert "字幕と文字（3 行 13 文字）" in request.text()

    def test_denying_subtitles_returns_nothing_and_changes_nothing(
        self, secret_host: FakeHost
    ) -> None:
        # 壊れると、断っても先に読んだ字幕が AI へ返るか、読んだだけで作品が変わる
        bridge = EditorBridge(secret_host)
        before = secret_host.document.project
        _seen, box = _run(bridge, "get_subtitles", lambda _request: SendAnswer.DENY)
        assert isinstance(box.get("error"), ToolError)
        assert "result" not in box
        assert "今日は" not in str(box["error"])
        assert secret_host.document.project is before
        assert not secret_host.document.can_undo

    @pytest.mark.parametrize("name", ["get_subtitles", "list_media", "get_project", "list_clips"])
    def test_running_first_does_not_change_the_project(
        self, secret_host: FakeHost, name: str
    ) -> None:
        # 壊れると、下見のために先に動かしただけで、断った後も作品が変わったままになる
        bridge = EditorBridge(secret_host)
        before = secret_host.document.project
        _run(bridge, name, lambda _request: SendAnswer.DENY)
        assert secret_host.document.project is before

    def test_a_writing_tool_is_previewed_from_its_arguments_before_running(
        self, host: FakeHost
    ) -> None:
        # 壊れると、テキストを置いてから送ってよいかを尋ね、断っても置いたままになる
        bridge = EditorBridge(host)
        bridge.auto_approve = True
        clips_when_asked: list[int] = []

        def deny(request: SendRequest) -> SendAnswer:
            clips_when_asked.append(_clip_count(host))
            return SendAnswer.DENY

        before = _clip_count(host)
        seen, box = _run(bridge, "add_text", deny, {"text": "秘密の台詞", "at_frame": 0})
        assert clips_when_asked == [before]
        assert _clip_count(host) == before
        assert isinstance(box.get("error"), ToolError)
        (sample,) = seen[0].samples
        assert sample.lines == ("秘密の台詞",)


def _clip_count(host: FakeHost) -> int:
    return sum(len(track.clips) for track in host.document.project.timeline.tracks)


# --- 呼び出しごとに尋ねる ---


class TestCallsSideBySide:
    """道具は並んで呼ばれる（呼び出しごとに別のスレッドで invoke する Codex の指摘 #450）"""

    def test_a_preview_grant_does_not_leak_to_another_call(self, secret_host: FakeHost) -> None:
        # 壊れると、字幕の下見のために立てた許しが、返事を待つ間に呼ばれた別の道具にも効き、
        # 尋ねていない本文を伏せずに送る
        probe = Operation(
            name="probe",
            description="",
            schema={},
            handler=lambda _host, _arguments: {"sendable": sendable_now(Outbound.TEXT)},
            sends=frozenset({Outbound.PROJECT, Outbound.TEXT}),
            hidden=frozenset({Outbound.TEXT}),
        )
        bridge = EditorBridge(secret_host)
        seen_elsewhere: list[object] = []

        def answer(_request: SendRequest) -> SendAnswer:
            done, box = call_in_thread(bridge, probe, {})
            while not done.is_set():
                bridge.pump()
            seen_elsewhere.append(box.get("result"))
            seen_elsewhere.append(bridge.may_send(Outbound.TEXT))
            return SendAnswer.DENY

        _seen, box = _run(bridge, "get_subtitles", answer)
        assert seen_elsewhere == [{"sendable": False}, False]
        assert "result" not in box


class TestLogFromTheResult:
    def test_a_tool_previewed_from_its_arguments_is_logged_from_its_result(
        self, host: FakeHost
    ) -> None:
        # 壊れると、引数から下見を作る道具（保存など）の記録が引数で数えられ、実際に返した
        # 場所や本文が記録から抜ける（Codex の指摘 #450）
        writer = Operation(
            name="writer",
            description="",
            schema={},
            handler=lambda _host, _arguments: {"text": "結果に入った文"},
            writes=True,
            sends=frozenset({Outbound.PROJECT, Outbound.TEXT}),
            preview=from_arguments(),
        )
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices({**bridge.send_choices, Outbound.TEXT: SendChoice.ALLOW})
        done, box = call_in_thread(bridge, writer, {"text": "引数の文"})
        answer_sends(bridge, done, lambda _request: SendAnswer.DENY)
        assert box.get("result") == {"text": "結果に入った文"}
        assert bridge.send_log.last()[Outbound.TEXT].sample.lines == ("結果に入った文",)

    def test_a_failure_that_carries_a_place_is_logged(self, secret_host: FakeHost) -> None:
        # 壊れると、失敗の文に混ざって AI へ返った素材の場所が、送った物の記録から抜ける
        # （Sourcery の指摘 #450）
        def fail(_host: object, _arguments: dict[str, Any]) -> object:
            raise ToolError(f"{SECRET_PATH} を開けませんでした")

        failing = Operation(
            name="failing",
            description="",
            schema={},
            handler=fail,
            sends=frozenset({Outbound.PROJECT, Outbound.FILES}),
            preview=from_result(),
        )
        bridge = EditorBridge(secret_host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.ALLOW})
        done, box = call_in_thread(bridge, failing, {})
        answer_sends(bridge, done, lambda _request: SendAnswer.DENY)
        assert isinstance(box.get("error"), ToolError)
        (entry,) = bridge.send_log.entries()
        assert entry.kinds == (SentKind(Outbound.FILES, 1),)


class TestEachCall:
    def test_it_asks_again_on_the_next_call(self, host: FakeHost) -> None:
        # 壊れると、呼び出しごとに確かめたい人の所でも、1 回答えたら尋ねずに送り続ける
        bridge = EditorBridge(host)
        bridge.ask_each_call = True
        first, _box = _run(bridge, "preview_frame", lambda _request: SendAnswer.UNTIL_CLOSE)
        second, again = _run(bridge, "preview_frame", lambda _request: SendAnswer.UNTIL_CLOSE)
        assert len(first) == len(second) == 1
        assert first[0].each_call
        assert isinstance(again.get("result"), ImageResult)
        # 今回だけの許しは呼び出しの外に残らない
        assert not bridge.may_send(Outbound.FRAMES)

    def test_a_denial_is_still_kept_until_close(self, host: FakeHost) -> None:
        # 壊れると、断った人に呼び出しのたびに同じ窓が出て、作業が止まり続ける
        bridge = EditorBridge(host)
        bridge.ask_each_call = True
        _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        seen, box = _run(bridge, "preview_frame", lambda _request: SendAnswer.UNTIL_CLOSE)
        assert seen == []
        assert isinstance(box.get("error"), ToolError)


# --- 送った物の記録 ---


class TestLog:
    def test_each_send_is_recorded_with_counts_only(self, secret_host: FakeHost) -> None:
        # 壊れると、窓の後で何がどれだけ送られたかを見直せない
        bridge = EditorBridge(secret_host)
        _run(bridge, "get_subtitles", lambda _request: SendAnswer.UNTIL_CLOSE)
        _run(bridge, "get_subtitles")
        entries = bridge.send_log.entries()
        assert [entry.tool for entry in entries] == ["get_subtitles", "get_subtitles"]
        assert entries[0].kinds == (SentKind(Outbound.TEXT, 3, 13),)
        last = bridge.send_log.last()[Outbound.TEXT]
        assert last.sample.lines == ("今日は", "編集ソフトを", "作ります")

    def test_hidden_names_are_not_recorded_as_sent(self, secret_host: FakeHost) -> None:
        # 壊れると、伏せ字にして送った名前を「送った」と残し、記録が実際より多く見える
        bridge = EditorBridge(secret_host)
        _run(bridge, "list_media", lambda _request: SendAnswer.DENY)
        assert bridge.send_log.entries() == []

    def test_allowed_in_the_settings_is_recorded_without_asking(self, host: FakeHost) -> None:
        # 尋ねない設定にした人の分こそ、何が出たかを残す
        bridge = EditorBridge(host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FRAMES: SendChoice.ALLOW})
        seen, _box = _run(bridge, "preview_frame")
        assert seen == []
        (entry,) = bridge.send_log.entries()
        assert entry.kinds == (SentKind(Outbound.FRAMES, 1),)
        assert bridge.send_log.last()[Outbound.FRAMES].sample.images

    def test_turning_the_record_off_keeps_only_the_last_preview(self, host: FakeHost) -> None:
        # 壊れると、記録を切った人の所にも記録が積もるか、何が出ているかを確かめる手段まで消える
        bridge = EditorBridge(host)
        bridge.send_log.keep = False
        bridge.set_send_choices({**bridge.send_choices, Outbound.FRAMES: SendChoice.ALLOW})
        _run(bridge, "preview_frame")
        assert bridge.send_log.entries() == []
        assert Outbound.FRAMES in bridge.send_log.last()

    def test_an_interrupt_records_nothing(self, host: FakeHost) -> None:
        # 壊れると、送らずに止めた物まで送ったと残り、記録が実際より多く見える
        bridge = EditorBridge(host)

        def interrupt(_request: SendRequest) -> SendAnswer:
            bridge.cancel()
            return SendAnswer.ALWAYS

        done, _box = call_in_thread(bridge, _operation("preview_frame"), {})
        answer_sends(bridge, done, interrupt)
        assert bridge.send_log.entries() == []


class TestLogStore:
    def _entry(self, at: datetime, tool: str = "get_subtitles") -> SentEntry:
        return SentEntry(at, tool, (SentKind(Outbound.TEXT, 2, 10),))

    def test_it_keeps_kinds_and_counts_but_no_content(self, tmp_path: Path) -> None:
        # 壊れると、記録そのものが字幕や場所の写しになり、手元に個人情報が積もる
        store = SendLogStore(tmp_path / "ai_sends.json")
        log = SendLog(store)
        log.add("get_subtitles", [Sample(Outbound.TEXT, 1, 4, lines=("秘密の話",))])
        text = (tmp_path / "ai_sends.json").read_text(encoding="utf-8")
        assert "秘密の話" not in text
        (entry,) = store.load()
        assert entry.kinds == (SentKind(Outbound.TEXT, 1, 4),)

    def test_old_entries_are_dropped(self, tmp_path: Path) -> None:
        # 壊れると、使った履歴が限りなく積もり続ける
        store = SendLogStore(tmp_path / "ai_sends.json")
        now = datetime.now().astimezone()
        store.add(self._entry(now - timedelta(days=KEEP_DAYS + 1), "old"))
        store.add(self._entry(now, "new"))
        assert [entry.tool for entry in store.load()] == ["new"]

    def test_clearing_forgets_the_file_and_the_last_preview(self, tmp_path: Path) -> None:
        # 壊れると、「記録を消す」を押しても中身が画面とファイルに残る
        log = SendLog(SendLogStore(tmp_path / "ai_sends.json"))
        log.add("preview_frame", [Sample(Outbound.FRAMES, 1, images=(b"png",))])
        assert log.clear()
        assert log.entries() == []
        assert log.last() == {}

    def test_a_broken_file_reads_as_empty(self, tmp_path: Path) -> None:
        # 手で書き換えられても起動と AI を止めない
        path = tmp_path / "ai_sends.json"
        path.write_text("{壊れた", encoding="utf-8")
        assert SendLogStore(path).load() == []
        entries: list[Any] = [{"at": "nope"}, {"at": "2026-11-08T10:00:00", "tool": 1}]
        path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
        assert SendLogStore(path).load() == []

    def test_kinds_are_read_in_the_asked_order(self, tmp_path: Path) -> None:
        # 壊れると、記録の一覧で種類の並びが行ごとに違い、見比べにくい
        path = tmp_path / "ai_sends.json"
        kinds = {kind.value: {"count": 1} for kind in reversed(ASKED)}
        path.write_text(
            json.dumps({"entries": [{"at": "2026-11-08T10:00:00", "tool": "t", "kinds": kinds}]}),
            encoding="utf-8",
        )
        (entry,) = SendLogStore(path).load()
        assert tuple(sent.kind for sent in entry.kinds) == ASKED
