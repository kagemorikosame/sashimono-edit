"""AI の道具が外（Anthropic）へ送る物の宣言と、送る前の同意（#342 D4）

道具の結果はそのまま Anthropic へ渡る 絵・字幕の文・素材の名前と場所は、本人が
許すまで送らない ブリッジ（AI のスレッドと画面のスレッドの間）で確かめるので、
実際にスレッドを 2 本使って見る
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai import bridge as bridge_module
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import EditorHost, ToolError
from sashimono.ai.operations import OPERATIONS, ImageResult, Operation, Pending, find_operation
from sashimono.ai.outbound import (
    ASKED,
    DESTINATION,
    TEXT_MARK,
    Outbound,
    SendAnswer,
    SendChoice,
    SendRequest,
    file_secrets,
    hide_text,
    redact,
    secret_pairs,
)
from sashimono.core.commands import RemoveMedia
from sashimono.core.model import MediaItem, Transcript
from sashimono.core.timebase import format_timecode
from tests.ai.conftest import FakeHost, make_loaded
from tests.ai.test_bridge import call_in_thread

SECRET_PATH = Path("C:/Users/秘密の人/動画/結婚式の撮影.mp4")


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


@pytest.fixture
def secret_host(video_media: MediaItem, transcript: Transcript) -> FakeHost:
    """素材の場所に本人の名前が入った作品"""
    return FakeHost(make_loaded(replace(video_media, path=SECRET_PATH), transcript))


def answer_sends(
    bridge: EditorBridge,
    done: threading.Event,
    reply: Callable[[SendRequest], SendAnswer | None],
    timeout: float = 5.0,
) -> list[SendRequest]:
    """画面のスレッドの代わりに、仕事を拾い、出てきた送る前の確かめに ``reply`` で答える

    ``None`` を返すと答えずに窓を閉じたことにする
    """
    seen: list[SendRequest] = []
    deadline = time.monotonic() + timeout
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        request = bridge.take_send_request()
        if request is not None:
            seen.append(request)
            answer = reply(request)
            if answer is None:
                request.give_up("閉じました")
            else:
                request.reply(answer)
        time.sleep(0.005)
    bridge.pump()
    assert done.is_set(), "道具が終わらなかった（送る前の確かめを待ったまま止まっている）"
    return seen


def _never(request: SendRequest) -> SendAnswer:
    pytest.fail(f"尋ねないはずの所で尋ねた: {request.kinds}")


def _run(
    bridge: EditorBridge,
    name: str,
    reply: Callable[[SendRequest], SendAnswer | None] = _never,
    arguments: dict[str, Any] | None = None,
) -> tuple[list[SendRequest], dict[str, object]]:
    done, box = call_in_thread(bridge, _operation(name), arguments or {})
    return answer_sends(bridge, done, reply), box


# --- 宣言 ---


class TestDeclarations:
    def test_a_tool_without_a_declaration_cannot_be_made(self) -> None:
        # 壊れると、絵や字幕を返す道具を足した人が書き忘れても黙って通り、同意を取らずに送る
        with pytest.raises(TypeError, match="sends"):
            Operation(  # type: ignore[call-arg]  # 書き忘れを再現するため、わざと渡さない
                name="forgot",
                description="",
                schema={},
                handler=lambda _host, _arguments: None,
            )

    def test_an_empty_declaration_is_refused(self) -> None:
        # 壊れると、空の宣言で「何も送らない」ふりをして、結果をそのまま送る道具を作れる
        with pytest.raises(ValueError, match="sends"):
            Operation(
                name="empty",
                description="",
                schema={},
                handler=lambda _host, _arguments: None,
                sends=frozenset(),
            )

    def test_every_tool_declares_what_it_sends(self) -> None:
        # 壊れると、宣言の無い道具が同意の外を通る（作る所で落ちるので、ここは念のため）
        for operation in OPERATIONS:
            assert operation.sends, operation.name
            assert operation.sends <= set(Outbound), operation.name

    def test_the_preview_is_declared_as_sending_frames(self) -> None:
        # 壊れると、プレビューの絵を同意なしで送る（この Issue の中心）
        assert Outbound.FRAMES in _operation("preview_frame").sends
        assert Outbound.TEXT in _operation("get_subtitles").sends
        assert Outbound.FILES in _operation("list_media").sends

    def test_what_each_reading_tool_returns_matches_its_declaration(
        self, secret_host: FakeHost, transcript: Transcript
    ) -> None:
        # 壊れると、宣言に無い物（素材の場所・字幕の文・絵）を返す道具が、尋ねずに送る
        # 読むだけの道具を全部、引数なしで動かして、結果の中身から宣言を確かめる
        secrets = {str(SECRET_PATH), SECRET_PATH.as_posix(), SECRET_PATH.name}
        texts = {segment.text for segment in transcript.segments}
        checked = 0
        for operation in OPERATIONS:
            if operation.writes or operation.asks:
                continue
            try:
                result: object = operation(secret_host, {})
            except ToolError as exc:
                # エラーの文も外へ送る 引数が要る道具は、断った文の中身を見る
                result = str(exc)
            checked += 1
            if isinstance(result, ImageResult):
                assert Outbound.FRAMES in operation.sends, operation.name
                continue
            text = json.dumps(result, ensure_ascii=False, default=str)
            if any(secret in text for secret in secrets):
                assert Outbound.FILES in operation.sends, operation.name
            if any(sentence in text for sentence in texts):
                assert Outbound.TEXT in operation.sends, operation.name
        assert checked >= 10

    def test_only_frames_text_and_files_are_asked(self) -> None:
        # 壊れると、指示の文や編集の構成まで尋ねて、AI に何も頼めなくなる（または逆に絵を尋ねない）
        assert ASKED == (Outbound.FRAMES, Outbound.TEXT, Outbound.FILES)
        assert {kind.value for kind in Outbound} == {
            "conversation",
            "project",
            "app_settings",
            "files",
            "text",
            "frames",
        }


#: 変える道具に渡す本文 結果やエラーの文にこれが出たら、字幕と文字を送ると書くべき道具
ARG_TEXT = "引数で渡した秘密の本文"
#: 前からあるテキストのクリップの本文
BODY_TEXT = "前からある秘密の本文"


def _ids(host: FakeHost) -> dict[str, str]:
    project = host.document.project
    media = project.media[0]
    transcript = media.transcript
    assert transcript is not None
    clips = [clip for track in project.timeline.tracks for clip in track.clips]
    text = next(c for c in clips if c.source is not None and c.source.kind == "text")
    video = next(c for c in clips if c.media_id is not None)
    return {
        "media": str(media.id),
        "segment": str(transcript.segments[0].id),
        "track": str(project.timeline.tracks[0].id),
        "video": str(video.id),
        "text": str(text.id),
    }


#: 変える道具ごとの、ありそうな引数 道具を足したらここにも足す（足さないと落ちる）
#: 引数が合わずに断られる物もあるが、それでよい 断った文も外へ送るので中身を見る
WRITE_EXAMPLES: dict[str, Callable[[dict[str, str]], dict[str, Any]]] = {
    "import_media": lambda _ids: {"paths": [str(SECRET_PATH)]},
    "place_media": lambda ids: {"media_id": ids["media"]},
    "add_track": lambda _ids: {"kind": "video", "name": "V9"},
    "set_track_state": lambda ids: {"track_id": ids["track"], "muted": True},
    "set_resolution": lambda _ids: {"width": 1280, "height": 720},
    "add_text": lambda _ids: {"text": ARG_TEXT},
    "add_shape": lambda _ids: {},
    "add_transition": lambda _ids: {},
    "split_clip": lambda ids: {"clip_id": ids["video"], "frame": 30},
    "trim_clip": lambda ids: {"clip_id": ids["video"], "tail_delta": -10},
    "move_clip": lambda ids: {"clip_id": ids["video"], "timeline_start": 10},
    "delete_clip": lambda ids: {"clip_id": ids["video"]},
    "add_scene": lambda _ids: {"name": "場面"},
    "place_scene": lambda _ids: {"scene_id": "無い"},
    "group_clips": lambda ids: {"clip_ids": [ids["video"], ids["text"]]},
    "ungroup_clips": lambda ids: {"clip_ids": [ids["video"]]},
    "move_clips": lambda ids: {"clip_ids": [ids["video"]], "delta": 5},
    "delete_clips": lambda ids: {"clip_ids": [ids["text"]]},
    "duplicate_clips": lambda ids: {"clip_ids": [ids["text"]]},
    "set_track_height": lambda _ids: {"height": 80},
    "set_clip_property": lambda ids: {"clip_id": ids["text"], "name": "enabled", "value": True},
    "add_effect": lambda ids: {"clip_id": ids["text"], "kind": "blur"},
    "add_stroke": lambda ids: {"clip_id": ids["text"]},
    "remove_stroke": lambda ids: {"clip_id": ids["text"], "stroke_id": "無い"},
    "move_stroke": lambda ids: {"clip_id": ids["text"], "stroke_id": "無い", "index": 0},
    "set_stroke_enabled": lambda ids: {"clip_id": ids["text"], "stroke_id": "無い", "enabled": 1},
    "set_param": lambda ids: {"clip_id": ids["text"], "name": "text", "value": ARG_TEXT},
    "add_keyframe": lambda ids: {"clip_id": ids["text"], "name": "text", "value": ARG_TEXT},
    "set_subtitle_text": lambda ids: {
        "media_id": ids["media"],
        "segment_id": ids["segment"],
        "text": ARG_TEXT,
    },
    "replace_subtitle_text": lambda _ids: {"pairs": [{"from": "今日は", "to": ARG_TEXT}]},
    "clean_subtitles": lambda ids: {"media_id": ids["media"]},
    "place_subtitles": lambda ids: {"media_id": ids["media"]},
    "jet_cut": lambda ids: {"media_id": ids["media"]},
    "transcribe": lambda ids: {"media_id": ids["media"], "replace": True},
    "undo": lambda _ids: {},
}


def _with_text_clip(video_media: MediaItem, transcript: Transcript) -> FakeHost:
    """素材の場所に本人の名前が入り、本文を持つテキストのクリップを 1 本置いた作品"""
    created = FakeHost(make_loaded(replace(video_media, path=SECRET_PATH), transcript))
    _operation("add_text")(created, {"text": BODY_TEXT})
    return created


class TestWritingToolsDeclareWhatTheyReturn:
    def test_every_writing_tool_has_an_example(self) -> None:
        # 壊れると、変える道具を足した人が例を書かずに済み、下の見張りの外を通る
        writing = {operation.name for operation in OPERATIONS if operation.writes}
        assert set(WRITE_EXAMPLES) == writing

    @pytest.mark.parametrize("name", sorted(WRITE_EXAMPLES))
    def test_what_each_writing_tool_returns_matches_its_declaration(
        self, name: str, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、変える道具が本文や素材の場所を結果に載せるのに宣言せず、字幕と文字や
        # 名前と場所を断った人にも、尋ねず伏せずに送る（add_text が本文を返していた）
        host = _with_text_clip(video_media, transcript)
        operation = _operation(name)
        arguments = WRITE_EXAMPLES[name](_ids(host))
        try:
            result: object = operation(host, arguments)
        except (ToolError, ValueError, KeyError) as exc:
            result = str(exc)
        text = json.dumps(result, ensure_ascii=False, default=str)
        sentences = {ARG_TEXT, BODY_TEXT[:12]} | {s.text for s in transcript.segments}
        if any(sentence in text for sentence in sentences):
            assert Outbound.TEXT in operation.sends, f"{name}: {text}"
        places = {str(SECRET_PATH), SECRET_PATH.as_posix(), SECRET_PATH.name, SECRET_PATH.stem}
        if any(place in text for place in places):
            assert Outbound.FILES in operation.sends, f"{name}: {text}"


# --- 送る前に尋ねる ---


class TestAskingBeforeSending:
    def test_the_first_preview_asks_before_drawing(self, host: FakeHost) -> None:
        # 壊れると、本人が知らないうちに映像の絵が外へ出る
        bridge = EditorBridge(host)
        drawn_when_asked: list[int] = []

        def deny(request: SendRequest) -> SendAnswer:
            drawn_when_asked.append(len(host.rendered))
            return SendAnswer.DENY

        seen, _box = _run(bridge, "preview_frame", deny)
        assert [request.kinds for request in seen] == [(Outbound.FRAMES,)]
        assert drawn_when_asked == [0]
        assert DESTINATION in seen[0].text()
        assert "映像の絵" in seen[0].text()

    def test_denying_sends_nothing(self, host: FakeHost) -> None:
        # 壊れると、「送らない」を押しても絵が描かれて外へ出る
        bridge = EditorBridge(host)
        _seen, box = _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        assert isinstance(box.get("error"), ToolError)
        assert "映像の絵" in str(box["error"])
        assert host.rendered == []

    def test_a_denial_is_kept_until_the_app_closes(self, host: FakeHost) -> None:
        # 壊れると、断ったのに AI が呼び直すたびに窓が出て、作業が止まり続ける
        bridge = EditorBridge(host)
        _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        _seen, box = _run(bridge, "preview_frame")
        assert isinstance(box.get("error"), ToolError)
        assert host.rendered == []

    def test_allowing_until_close_does_not_ask_again(self, host: FakeHost) -> None:
        # 壊れると、AI が絵を見るたびに窓が出て、1 本の動画を見るのに何十回も押す
        bridge = EditorBridge(host)
        seen, box = _run(bridge, "preview_frame", lambda _request: SendAnswer.UNTIL_CLOSE)
        assert len(seen) == 1
        assert isinstance(box.get("result"), ImageResult)
        _seen, again = _run(bridge, "preview_frame")
        assert isinstance(again.get("result"), ImageResult)
        assert len(host.rendered) == 2

    def test_closing_the_window_is_not_remembered(self, host: FakeHost) -> None:
        # 壊れると、答えずに閉じただけで「送らない」が残り、次に尋ねてもらえない
        bridge = EditorBridge(host)
        _seen, box = _run(bridge, "preview_frame", lambda _request: None)
        assert isinstance(box.get("error"), ToolError)
        seen, _box = _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        assert len(seen) == 1

    def test_auto_approve_does_not_skip_it(self, host: FakeHost) -> None:
        # 壊れると、「変更を自動で承認」を入れた人の絵が、尋ねずに外へ出る（外へ送るのは
        # 取り消せない 2026-10-14 利用者の決定）
        bridge = EditorBridge(host, auto_approve=True)
        seen, _box = _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        assert len(seen) == 1

    def test_a_setting_of_allow_sends_without_asking(self, host: FakeHost) -> None:
        # 壊れると、設定で「尋ねずに送る」にした人にも毎回窓が出る
        bridge = EditorBridge(host)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
        _seen, box = _run(bridge, "preview_frame")
        assert isinstance(box.get("result"), ImageResult)

    def test_a_setting_of_deny_refuses_without_asking(self, host: FakeHost) -> None:
        # 壊れると、設定で「送らない」にしたのに窓が出る（押し間違えると送る）
        bridge = EditorBridge(host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.TEXT: SendChoice.DENY})
        _seen, box = _run(bridge, "get_subtitles")
        assert isinstance(box.get("error"), ToolError)
        assert "字幕と文字" in str(box["error"])

    def test_changing_the_setting_forgets_the_answer_in_the_window(self, host: FakeHost) -> None:
        # 壊れると、設定で「送る前に尋ねる」に戻しても、前に窓で答えたままになり尋ねない
        bridge = EditorBridge(host)
        _run(bridge, "preview_frame", lambda _request: SendAnswer.UNTIL_CLOSE)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FRAMES: SendChoice.ALLOW})
        bridge.set_send_choices({**bridge.send_choices, Outbound.FRAMES: SendChoice.ASK})
        seen, _box = _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        assert len(seen) == 1

    def test_tools_that_send_only_the_structure_never_ask(self, host: FakeHost) -> None:
        # 壊れると、分割や選択のたびに窓が出て、AI に編集を頼めなくなる
        bridge = EditorBridge(host, auto_approve=True)
        clip = host.project.timeline.tracks[0].clips[0]
        _seen, box = _run(bridge, "split_clip", arguments={"clip_id": str(clip.id), "frame": 30})
        assert box.get("error") is None
        _seen, box = _run(bridge, "get_selection")
        assert box.get("error") is None

    def test_an_interrupt_while_asking_folds_it(self, host: FakeHost) -> None:
        # 壊れると、中断を押しても AI のスレッドが送る前の確かめの返事を 10 分待ち続ける
        bridge = EditorBridge(host)

        def interrupt(_request: SendRequest) -> SendAnswer:
            bridge.cancel()
            return SendAnswer.ALWAYS

        _seen, box = _run(bridge, "preview_frame", interrupt)
        assert isinstance(box.get("error"), ToolError)
        assert host.rendered == []

    def test_no_answer_in_time_sends_nothing(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、席を外している間に時間が切れ、黙って送る
        monkeypatch.setattr(bridge_module, "SEND_TIMEOUT", 0.05)
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, _operation("preview_frame"), {})
        deadline = time.monotonic() + 5
        while not done.is_set() and time.monotonic() < deadline:
            bridge.pump()
            time.sleep(0.005)
        assert isinstance(box.get("error"), ToolError)
        assert host.rendered == []


# --- 素材の名前と場所 ---


class TestFileNames:
    def test_not_allowed_names_are_hidden_but_the_tool_still_works(
        self, secret_host: FakeHost
    ) -> None:
        # 壊れると、名前を送らないと決めた人の素材の場所（ユーザー名入り）が外へ出るか、
        # 一覧の道具ごと断られて AI が素材を指せなくなる
        bridge = EditorBridge(secret_host)
        seen, box = _run(bridge, "list_media", lambda _request: SendAnswer.DENY)
        assert [request.kinds for request in seen] == [(Outbound.FILES,)]
        text = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密の人" not in text
        assert "結婚式" not in text
        media = secret_host.project.media[0]
        assert str(media.id) in text

    def test_errors_are_hidden_too(self, secret_host: FakeHost) -> None:
        # 壊れると、断った人の素材の名前が、エラーの文に混ざって外へ出る
        bridge = EditorBridge(secret_host, auto_approve=True)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.DENY})
        media = secret_host.project.media[0]
        _seen, box = _run(bridge, "jet_cut", arguments={"media_id": str(media.id)})
        error = box.get("error")
        assert isinstance(error, ToolError)
        assert "波形" in str(error)
        assert "結婚式" not in str(error)

    def test_a_tool_that_forgot_to_declare_names_still_hides_them(
        self, secret_host: FakeHost
    ) -> None:
        # 壊れると、宣言を書き漏らした道具が、まだ許していない素材の場所を送る
        leaky = Operation(
            name="leaky",
            description="",
            schema={},
            handler=lambda host, _arguments: {"path": str(host.document.project.media[0].path)},
            sends=frozenset({Outbound.PROJECT}),
        )
        bridge = EditorBridge(secret_host)
        done, box = call_in_thread(bridge, leaky, {})
        answer_sends(bridge, done, _never)
        assert "秘密の人" not in json.dumps(box["result"], ensure_ascii=False)

    def test_allowed_names_pass_through(self, secret_host: FakeHost) -> None:
        # 壊れると、名前を送ってよいと答えた人にも伏せ字が返り、AI が素材を名前で探せない
        bridge = EditorBridge(secret_host)
        _seen, box = _run(bridge, "list_media", lambda _request: SendAnswer.UNTIL_CLOSE)
        assert "結婚式の撮影.mp4" in json.dumps(box["result"], ensure_ascii=False)

    def test_an_image_from_an_undeclared_tool_is_refused(self, host: FakeHost) -> None:
        # 壊れると、絵を返すと宣言していない道具（#320 の見渡しの絵など）が、尋ねずに絵を送る
        def draw(target: EditorHost, _arguments: dict[str, Any]) -> object:
            return ImageResult(png=target.render_png(0, width=160))

        sneaky = Operation(
            name="sneaky",
            description="",
            schema={},
            handler=draw,
            sends=frozenset({Outbound.PROJECT}),
        )
        bridge = EditorBridge(host)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
        done, box = call_in_thread(bridge, sneaky, {})
        answer_sends(bridge, done, _never)
        assert isinstance(box.get("error"), ToolError)


# --- 取り消しと履歴（2026-10-15 利用者の決定 字幕と文字を送らなくても取り消せる） ---

SECRET_TEXT = "秘密の挨拶をここに書きます"


def _add_secret_text(bridge: EditorBridge) -> None:
    _seen, box = _run(bridge, "add_text", arguments={"text": SECRET_TEXT})
    assert box.get("error") is None, box.get("error")


def _type_secret_text(host: FakeHost) -> None:
    """人が画面でテキストを足した、の代わり（ブリッジを通さない 外へは何も送らない）"""
    _operation("add_text")(host, {"text": SECRET_TEXT})


@pytest.fixture
def text_refused(host: FakeHost) -> EditorBridge:
    """字幕と文字を「送らない」にした人 名前と場所は許してある 変更は自動で承認

    名前と場所を許しておくのは、伏せ字が字幕と文字の扱いから来たことを見分けるため
    """
    bridge = EditorBridge(host, auto_approve=True)
    bridge.set_send_choices(
        {Outbound.FRAMES: SendChoice.ASK, Outbound.TEXT: SendChoice.DENY}
        | {Outbound.FILES: SendChoice.ALLOW}
    )
    return bridge


def _text_clips(host: FakeHost) -> int:
    return sum(
        1
        for track in host.document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == "text"
    )


class TestWritingTextWithoutSendingText:
    """本文を引数で渡して結果にも返す道具（CodeRabbit の指摘 #366）"""

    def test_adding_text_is_refused_and_nothing_is_added(
        self, host: FakeHost, text_refused: EditorBridge
    ) -> None:
        # 壊れると、字幕と文字を断った人の所で、本文が要る道具が尋ねずに動き、本文が結果で返る
        _seen, box = _run(text_refused, "add_text", arguments={"text": SECRET_TEXT})
        assert isinstance(box.get("error"), ToolError)
        assert _text_clips(host) == 0

    @pytest.mark.parametrize("name", ["set_subtitle_text", "replace_subtitle_text"])
    def test_subtitle_edits_are_refused(self, name: str, text_refused: EditorBridge) -> None:
        # 壊れると、字幕を断った人の字幕を、AI が尋ねずに書き換え、直した本文が外へ出る
        _seen, box = _run(text_refused, name, arguments={"pairs": [], "text": SECRET_TEXT})
        assert isinstance(box.get("error"), ToolError)
        assert "字幕と文字" in str(box["error"])

    def test_a_text_value_is_set_but_hidden(self, host: FakeHost) -> None:
        # 壊れると、値を変える道具ごと断られて色や位置も変えられないか、本文が外へ出る
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
        _add_secret_text(bridge)
        bridge.set_send_choices(
            {Outbound.FRAMES: SendChoice.ASK, Outbound.TEXT: SendChoice.DENY}
            | {Outbound.FILES: SendChoice.ALLOW}
        )
        clip = next(
            c
            for track in host.document.project.timeline.tracks
            for c in track.clips
            if c.source is not None and c.source.kind == "text"
        )
        changed = "書き換えた秘密の本文"
        arguments = {"clip_id": str(clip.id), "name": "text", "value": changed}
        _seen, box = _run(bridge, "set_param", arguments=arguments)
        assert box.get("error") is None, box.get("error")
        assert "秘密" not in json.dumps(box["result"], ensure_ascii=False)
        bodies = [
            c.source.params["text"]
            for track in host.document.project.timeline.tracks
            for c in track.clips
            if c.id == clip.id and c.source is not None
        ]
        assert bodies == [changed]


class TestUndoWithoutSendingText:
    def test_undo_still_works_and_the_text_is_hidden(
        self, host: FakeHost, text_refused: EditorBridge
    ) -> None:
        # 壊れると、字幕と文字を送らない人は AI に取り消しを頼めないか、取り消した操作の名前に
        # 混ざったテキストの文が外へ出る
        _type_secret_text(host)
        assert _text_clips(host) == 1
        _seen, box = _run(text_refused, "undo")
        assert box.get("error") is None, box.get("error")
        assert _text_clips(host) == 0
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密" not in sent
        assert SECRET_TEXT[:12] not in sent
        # どの操作を取り消したかは読める
        assert "テキストを追加" in sent

    def test_the_history_is_read_with_the_text_hidden(
        self, host: FakeHost, text_refused: EditorBridge
    ) -> None:
        # 壊れると、履歴を読むだけで、送らないと決めた文の頭が外へ出る（または履歴を読めない）
        _type_secret_text(host)
        _seen, box = _run(text_refused, "get_history")
        assert box.get("error") is None, box.get("error")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密" not in sent
        assert "テキストを追加" in sent
        assert box["result"]["can_undo"] is True  # type: ignore[index]  # 結果は辞書で返る

    def test_undo_does_not_ask_about_text(self, host: FakeHost) -> None:
        # 壊れると、まだ答えていない人が取り消しを頼むたびに、字幕と文字の窓が出る
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.ALLOW})
        _type_secret_text(host)
        _seen, box = _run(bridge, "undo")
        assert box.get("error") is None, box.get("error")
        assert "秘密" not in json.dumps(box["result"], ensure_ascii=False)

    def test_allowed_text_passes_through(self, host: FakeHost) -> None:
        # 壊れると、文字を送ってよいと答えた人にも伏せ字が返り、AI がどの文を消したか読めない
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
        _add_secret_text(bridge)
        _seen, box = _run(bridge, "undo")
        assert SECRET_TEXT[:12] in json.dumps(box["result"], ensure_ascii=False)

    def test_subtitles_are_still_refused(self, text_refused: EditorBridge) -> None:
        # 壊れると、伏せて送る扱いが字幕の一覧にまで広がり、送らないと決めた字幕の道具が動く
        _seen, box = _run(text_refused, "get_subtitles")
        assert isinstance(box.get("error"), ToolError)

    def test_only_text_can_be_hidden(self) -> None:
        # 壊れると、絵を「伏せて送る」と書いた道具が、尋ねずに絵を送る
        with pytest.raises(ValueError, match="伏せて"):
            Operation(
                name="hidden_frames",
                description="",
                schema={},
                handler=lambda _host, _arguments: None,
                sends=frozenset({Outbound.FRAMES}),
                hidden=frozenset({Outbound.FRAMES}),
            )

    def test_counts_stay_and_quotes_go(self) -> None:
        # 壊れると、数だけの名前（無音カットの数）まで伏せて読めなくなるか、引用した文が外へ出る
        hidden = hide_text(
            ["無音カット: 3 か所", "見た目: 選んでいるテキスト「秘密」を写します", "配置: 本編"],
            [],
        )
        assert hidden[0] == "無音カット: 3 か所"
        assert "秘密" not in hidden[1]
        assert hidden[2] == f"配置: {TEXT_MARK}"


class TestWaitingTools:
    """待ってから返す道具（:class:`Pending` #321 の detect_scenes など）"""

    @staticmethod
    def _waiting(steps: list[str], answer: Callable[[EditorHost], object]) -> Operation:
        def wait(seconds: float) -> bool:
            del seconds
            steps.append("待った")
            return True

        def start(host: EditorHost, _arguments: dict[str, Any]) -> object:
            steps.append("動いた")
            return Pending(wait=wait, seconds=1.0, then=lambda: answer(host))

        return Operation(
            name="waiting",
            description="",
            schema={},
            handler=start,
            sends=frozenset({Outbound.PROJECT, Outbound.FILES}),
        )

    def test_consent_is_asked_before_running_and_waiting(self, secret_host: FakeHost) -> None:
        # 壊れると、待つ道具が同意の前に動き出して待ち、断っても解析や読み込みが走る
        steps: list[str] = []
        operation = self._waiting(
            steps, lambda host: {"media": host.document.project.media[0].name}
        )
        bridge = EditorBridge(secret_host)

        def deny(_request: SendRequest) -> SendAnswer:
            steps.append("尋ねた")
            return SendAnswer.DENY

        done, box = call_in_thread(bridge, operation, {})
        answer_sends(bridge, done, deny)
        assert steps[0] == "尋ねた"
        assert steps == ["尋ねた", "動いた", "待った"]
        # 名前と場所は断っても動かし、待った後に返る名前も伏せる
        assert "結婚式" not in json.dumps(box["result"], ensure_ascii=False)

    def test_what_comes_back_after_waiting_is_hidden_too(self, secret_host: FakeHost) -> None:
        # 壊れると、待つ前の戻り値だけを伏せ、待った後に返る素材の場所がそのまま外へ出る
        steps: list[str] = []
        operation = self._waiting(steps, lambda host: str(host.document.project.media[0].path))
        bridge = EditorBridge(secret_host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.DENY})
        done, box = call_in_thread(bridge, operation, {})
        answer_sends(bridge, done, _never)
        assert "秘密の人" not in str(box["result"])
        assert steps == ["動いた", "待った"]

    def test_detect_scenes_declares_names_and_places(self) -> None:
        # 壊れると、素材の名前の入った断りの文（映像がありません・調べられませんでした）を、
        # 名前と場所を断った人の所でも尋ねずに送る
        assert _operation("detect_scenes").sends == {Outbound.PROJECT, Outbound.FILES}


class TestPlacesOutsideTheTable:
    """表（今の作品の素材と場所）に無い名前と場所（Codex の指摘 #366）"""

    def test_a_removed_material_name_in_the_history_is_hidden(self, secret_host: FakeHost) -> None:
        # 壊れると、置いてから消した素材の名前（「配置: 結婚式の撮影.mp4」）が履歴に残り、
        # 名前と場所を断った人の所でも外へ出る 今の作品に無い名前は表から伏せられない
        media = secret_host.document.project.media[0]
        _operation("place_media")(secret_host, {"media_id": str(media.id)})
        clips = [
            str(clip.id)
            for track in secret_host.document.project.timeline.tracks
            for clip in track.clips
        ]
        _operation("delete_clips")(secret_host, {"clip_ids": clips})
        secret_host.apply_commands([RemoveMedia(media.id)], "素材を削除")
        assert not secret_host.document.project.media
        bridge = EditorBridge(secret_host)
        bridge.set_send_choices(
            {Outbound.FRAMES: SendChoice.ASK, Outbound.TEXT: SendChoice.ALLOW}
            | {Outbound.FILES: SendChoice.DENY}
        )
        _seen, box = _run(bridge, "get_history")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "結婚式" not in sent
        assert "配置" in sent

    def test_a_folder_at_the_root_is_hidden(self) -> None:
        # 壊れると、根の直下の 1 段だけの場所（/backup）が、区切りが 1 つなので伏せずに送られる
        assert "backup" not in redact("/backup", [])
        assert "backup" not in redact("置き場 \\backup", [])
        assert redact("1/2 画質", []) == "1/2 画質"

    def test_folders_in_the_settings_are_hidden_whole(self, host: FakeHost) -> None:
        # 壊れると、設定の中のフォルダの値が、形（区切りの数など）しだいで伏せずに送られる
        host.settings = {
            "preferences": {
                "state_folder": "backup",
                "export_folder": "/書き出し",
                "theme": "dark",
            },
        }
        bridge = EditorBridge(host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.DENY})
        _seen, box = _run(bridge, "get_app_settings")
        preferences = box["result"]["preferences"]  # type: ignore[index]  # 結果は辞書で返る
        assert preferences["state_folder"] != "backup"
        assert "書き出し" not in preferences["export_folder"]
        assert preferences["theme"] == "dark"


# --- 名前と場所を断った人の所で、目印の名前がどこにも出ないこと（Codex の指摘 #366） ---

#: 目印 素材・作品・置き場の名前に入れ、結果のどこかに出たら漏れ
MARK = "ZZSECRET"


def _marked_host(
    video_media: MediaItem, transcript: Transcript, scene: str, tmp_path: Path
) -> FakeHost:
    """目印の名前を持つ作品を、場面ごとの形にする

    - そのまま: 素材も作品の名前も今の作品にある
    - 消した後: 置いた素材をクリップごと消し、素材も外した 名前は履歴と字幕起こしの記録にだけ残る
    - 名前を変えた後: 作品は別の名前で保存してあり、作品の名前だけが目印を持つ
    - 開き直した後: AI に道具を使わせないまま別の作品を開いた 名前は作品の外にだけ残る
    """
    media = replace(video_media, path=Path(f"C:/Users/{MARK}人/動画/{MARK}素材.mp4"))
    project = replace(make_loaded(media, transcript), name=f"{MARK}作品")
    host = FakeHost(project)
    host.settings = {"preferences": {"state_folder": f"/{MARK}退避", "theme": "dark"}}
    host.transcription = f"{MARK}素材.mp4: 終了した"
    # 変える道具の例（WRITE_EXAMPLES）はテキストのクリップを指す
    _operation("add_text")(host, {"text": BODY_TEXT})
    if scene == "消した後":
        placed = host.document.project.media[0]
        _operation("place_media")(host, {"media_id": str(placed.id)})
        clips = [
            str(clip.id) for track in host.document.project.timeline.tracks for clip in track.clips
        ]
        _operation("delete_clips")(host, {"clip_ids": clips})
        host.apply_commands([RemoveMedia(placed.id)], "素材を削除")
        assert not host.document.project.media
    elif scene == "名前を変えた後":
        host.saved_at = tmp_path / "ふつうの名前.sme"
    elif scene == "開き直した後":
        # AI に一度も道具を使わせないまま、別の作品を開いた 履歴は消え、前の作品の名前は
        # 字幕起こしの記録と、アプリが作品の外で覚えている所（前の作品の場所）にだけ残る
        host.document.reset(replace(make_loaded(video_media, transcript), name="ふつう"))
        _operation("add_text")(host, {"text": BODY_TEXT})
        host.remembered = [f"{MARK}前の作品"]
        host.transcription = f"{MARK}素材.mp4: 終了した（{MARK}前の作品 で始めた起こし）"
    return host


def _every_example(host: FakeHost) -> list[tuple[str, dict[str, Any]]]:
    """読む道具は引数なしで、変える道具は例の引数で、質問の道具を除いて全部"""
    project = host.document.project
    ids = (
        _ids(host)
        if project.media
        else dict.fromkeys(("media", "segment", "track", "video", "text"), "無い")
    )
    calls: list[tuple[str, dict[str, Any]]] = []
    for operation in OPERATIONS:
        if operation.asks:
            continue
        example = WRITE_EXAMPLES.get(operation.name)
        calls.append((operation.name, example(ids) if example is not None else {}))
    return calls


class TestNoNameLeaksWhenNamesAreRefused:
    @pytest.mark.parametrize("scene", ["そのまま", "消した後", "名前を変えた後", "開き直した後"])
    def test_the_mark_never_reaches_the_results(
        self, scene: str, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、名前と場所を断った人の素材や作品の名前（ユーザー名入りの場所も）が、どれかの
        # 道具の結果か断りの文に混ざって外へ出る 今の作品の素材だけを伏せる作りだと、消した
        # 素材の名前（字幕起こしの記録・履歴）や、保存先と違う作品の名前が漏れる
        leaks: list[str] = []
        for name, arguments in _every_example(
            _marked_host(video_media, transcript, scene, tmp_path)
        ):
            # 道具ごとに作り直す 前の道具が素材を消すと、後の道具が名前を出さずに通ってしまう
            host = _marked_host(video_media, transcript, scene, tmp_path)
            bridge = EditorBridge(host, auto_approve=True)
            bridge.set_send_choices(
                {Outbound.FRAMES: SendChoice.ALLOW, Outbound.TEXT: SendChoice.ALLOW}
                | {Outbound.FILES: SendChoice.DENY}
            )
            done, box = call_in_thread(bridge, _operation(name), arguments)
            answer_sends(bridge, done, _never)
            sent = json.dumps(
                {"result": box.get("result"), "error": str(box.get("error") or "")},
                ensure_ascii=False,
                default=str,
            )
            if MARK in sent:
                leaks.append(f"{name}: {sent[:200]}")
        assert not leaks


class TestRedact:
    @pytest.mark.parametrize(
        "place",
        [
            "/home/秘密の人/動画/無い.mp4",
            "/Users/秘密の人/Movies/x.mov",
            "file:///C:/Users/秘密の人/x.mp4",
            "file:///home/秘密の人/x.mp4",
            "~/秘密の人の動画/x.mp4",
            "~秘密の人/動画/x.mp4",
        ],
    )
    def test_other_ways_of_writing_a_place_are_hidden(self, place: str) -> None:
        # 壊れると、AI が Windows 以外の書き方で渡した場所が、断った文に混ざってそのまま外へ出る
        assert "秘密の人" not in redact(f"ファイルがありません: {place}", [])

    @pytest.mark.parametrize(
        "kept",
        [
            "30/1",
            "色/明るさ",
            "https://github.com/kagemorikosame/sashimono-edit/issues",
            "1/2 画質",
        ],
    )
    def test_things_that_only_look_like_places_stay(self, kept: str) -> None:
        # 壊れると、フレームレートや分類の名前、ヘルプの URL まで伏せられ、AI が読めなくなる
        assert redact(kept, []) == kept

    def test_a_missing_posix_place_in_an_import_error_is_hidden(self, host: FakeHost) -> None:
        # 壊れると、名前と場所を断った人の所でも、読み込めなかった場所（ユーザー名入り）が
        # import_media の断りの文に混ざって返る（CodeRabbit の指摘 #366）
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.DENY})
        place = "/home/秘密の人/動画/無い.mp4"
        _seen, box = _run(bridge, "import_media", arguments={"paths": [place]})
        error = box.get("error")
        assert isinstance(error, ToolError)
        assert "ファイルがありません" in str(error)
        assert "秘密の人" not in str(error)

    def test_a_path_inside_a_sentence_is_hidden(self) -> None:
        # 壊れると、表に無い場所（退避のフォルダなど）が文に続けて書かれたとき、伏せずに送る
        text = redact("置き場はC:\\Users\\秘密の人\\退避 です", [])
        assert "秘密の人" not in text
        assert "置き場は" in text

    def test_longer_secrets_are_replaced_first(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、場所の中のファイル名だけが先に伏せられ、フォルダの名前が残る
        project = make_loaded(replace(video_media, path=SECRET_PATH), transcript)
        table = secret_pairs(file_secrets(project, project_path=None, export_folder=None))
        hidden = redact({"path": SECRET_PATH.as_posix(), "list": [SECRET_PATH.name]}, table)
        assert "秘密の人" not in json.dumps(hidden, ensure_ascii=False)
        assert "〔素材" in hidden["path"]

    def test_the_caption_of_a_frame_keeps_the_timecode(self, host: FakeHost) -> None:
        # 壊れると、伏せ字の処理で絵の説明（何フレーム目か）まで消え、AI がどこの絵か分からない
        bridge = EditorBridge(host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FRAMES: SendChoice.ALLOW})
        _seen, box = _run(bridge, "preview_frame", arguments={"frame": 30})
        result = box.get("result")
        assert isinstance(result, ImageResult)
        assert format_timecode(30, host.project.rate) in result.caption
