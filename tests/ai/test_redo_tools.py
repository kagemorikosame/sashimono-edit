"""AI の取り消し・やり直しと、取り消しの範囲（B7 #333）

AI の 1 回の指示は、始めから終わりまで 1 つの段（チェックポイント）を開いたまま進む
（ui/chat/panel.py） ここではその形を ``begin_checkpoint`` で作り、AI の道具の
``undo`` ``redo`` ``get_history`` が、指示の中の 1 手・指示の中の全部・前の指示の 1 段を
人の Ctrl+Z / Ctrl+Y と同じ履歴の上で辿ることを押さえる
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound, SendChoice
from sashimono.core.commands import AddMedia
from sashimono.core.model import ClipId, MediaItem, Transcript, new_media_id
from tests.ai.conftest import FakeHost
from tests.ai.test_bridge import call_in_thread
from tests.ai.test_outbound import (
    BODY_TEXT,
    SECRET_PATH,
    SECRET_TEXT,
    _with_text_clip,
    answer_sends,
    undeclared,
)


def run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんな道具は無い: {tool}"
    return operation(host, arguments)


def _first_clip(host: FakeHost) -> ClipId:
    return host.document.project.timeline.tracks[0].clips[0].id


def _clip_count(host: FakeHost) -> int:
    return len(host.document.project.timeline.tracks[0].clips)


def _split(host: FakeHost, frame: int) -> None:
    run(host, "split_clip", clip_id=str(_first_clip(host)), frame=frame)


class TestRedo:
    def test_redo_brings_back_what_undo_took_away(self, host: FakeHost) -> None:
        """壊れると、AI が取り消しすぎたときに自分で戻せず、人が Ctrl+Y を押すしかない"""
        _split(host, 100)
        run(host, "undo")
        assert _clip_count(host) == 1

        result = run(host, "redo")

        assert result["redone"] == ["クリップを分割"]
        assert result["can_redo"] is False
        assert _clip_count(host) == 2

    def test_redo_with_nothing_to_redo_says_so(self, host: FakeHost) -> None:
        """壊れると、やり直す物が無いのに AI が「やり直した」と思い込んで先へ進む"""
        with pytest.raises(ToolError, match="やり直せる操作がありません"):
            run(host, "redo")

    def test_redo_several_steps_stops_at_the_end(self, host: FakeHost) -> None:
        """壊れると、多めに頼んだやり直しが失敗し、戻せた分まで AI に伝わらない"""
        _split(host, 100)
        _split(host, 50)
        run(host, "undo", steps=2)
        result = run(host, "redo", steps=5)
        assert result["redone"] == ["クリップを分割", "クリップを分割"]
        assert _clip_count(host) == 3

    def test_redo_is_the_same_step_the_person_would_redo(self, host: FakeHost) -> None:
        """壊れると、AI のやり直しと人の Ctrl+Y が別の履歴を辿り、片方で戻した物が片方で消える"""
        _split(host, 100)
        run(host, "undo")
        assert host.document.redo_label == "クリップを分割"
        run(host, "redo")
        assert host.document.can_redo is False
        host.document.undo()
        assert _clip_count(host) == 1


class TestTheScopeOfOneInstruction:
    def test_undo_in_an_instruction_reverts_only_the_last_tool(self, host: FakeHost) -> None:
        """壊れると、AI が直前の 1 手だけ戻すつもりで、前の指示ごと戻してしまう"""
        _split(host, 200)
        host.document.begin_checkpoint("AI: 細かく切って")
        _split(host, 100)
        _split(host, 50)

        result = run(host, "undo")

        assert result["undone"] == ["クリップを分割"]
        assert _clip_count(host) == 3
        host.document.end_checkpoint()
        # 前の分割も、この指示の段も残る
        assert host.document.history_labels == ("クリップを分割", "AI: 細かく切って")

    def test_undo_the_whole_instruction_keeps_what_came_before(self, host: FakeHost) -> None:
        """壊れると、「この指示でやったことを全部戻して」で、前の指示や人の編集まで戻る"""
        _split(host, 200)
        host.document.begin_checkpoint("AI: 細かく切って")
        _split(host, 100)
        _split(host, 50)

        result = run(host, "undo", this_instruction=True)

        assert result["undone"] == ["クリップを分割", "クリップを分割"]
        assert _clip_count(host) == 2
        host.document.end_checkpoint()
        assert host.document.history_labels == ("クリップを分割",)

    def test_undo_the_whole_instruction_with_nothing_done_is_refused(self, host: FakeHost) -> None:
        """壊れると、まだ何もしていない指示で「全部戻す」が前の操作を戻してしまう"""
        _split(host, 200)
        host.document.begin_checkpoint("AI: 何もまだ")
        with pytest.raises(ToolError, match="この指示ではまだ何も変えていません"):
            run(host, "undo", this_instruction=True)
        assert _clip_count(host) == 2
        host.document.end_checkpoint()

    def test_steps_and_the_whole_instruction_cannot_be_mixed(self, host: FakeHost) -> None:
        """壊れると、どちらの頼みを優先したのか AI に分からないまま、片方だけ当たる"""
        host.document.begin_checkpoint("AI: 両方")
        _split(host, 100)
        with pytest.raises(ToolError, match="steps"):
            run(host, "undo", steps=2, this_instruction=True)
        assert _clip_count(host) == 2
        host.document.end_checkpoint()

    def test_the_previous_instruction_is_undone_as_one_step(self, host: FakeHost) -> None:
        """壊れると、「さっきの指示を取り消して」で前の指示が 1 手ずつしか戻らないか、履歴が壊れる

        戻した指示は、人が後から Ctrl+Y でまとめてやり直せる
        """
        host.document.begin_checkpoint("AI: 前の指示")
        _split(host, 100)
        _split(host, 50)
        host.document.end_checkpoint()

        host.document.begin_checkpoint("AI: さっきのを取り消して")
        result = run(host, "undo")
        host.document.end_checkpoint()

        assert result["undone"] == ["AI: 前の指示"]
        assert _clip_count(host) == 1
        # 人は Ctrl+Y で前の指示をまとめてやり直せる
        assert host.document.redo_label == "AI: 前の指示"
        host.document.redo()
        assert _clip_count(host) == 3

    def test_what_the_ai_undid_last_can_be_redone_by_the_person_later(self, host: FakeHost) -> None:
        """壊れると、AI が指示の最後に取り消した 1 手を、人が後から Ctrl+Y で戻せない"""
        host.document.begin_checkpoint("AI: 試して戻す")
        _split(host, 100)
        _split(host, 50)
        run(host, "undo")
        host.document.end_checkpoint()

        assert _clip_count(host) == 2
        host.document.redo()
        assert _clip_count(host) == 3


class TestHistory:
    def test_history_shows_what_can_be_redone_and_this_instructions_steps(
        self, host: FakeHost
    ) -> None:
        """壊れると、AI は何をやり直せるか・この指示で何を変えたかを読めず、推測で取り消す"""
        _split(host, 200)
        host.document.begin_checkpoint("AI: 細かく")
        _split(host, 100)
        _split(host, 50)
        run(host, "undo")

        result = run(host, "get_history")

        assert result["can_redo"] is True
        assert result["redo_label"] == "クリップを分割"
        assert result["this_instruction"] == ["クリップを分割"]
        assert result["undo_label"] == "クリップを分割"
        host.document.end_checkpoint()

    def test_history_outside_an_instruction_has_no_steps_of_its_own(self, host: FakeHost) -> None:
        """壊れると、指示の外で読んだ履歴に、前の指示の手が今の指示の物として出る"""
        _split(host, 100)
        result = run(host, "get_history")
        assert result["this_instruction"] == []
        assert result["can_redo"] is False


class TestWhatRedoReturns:
    def test_redo_declares_the_same_as_undo(self) -> None:
        """壊れると、やり直しだけ名前の伏せ方が弱く、取り消しでは伏せた名前がやり直しの結果で出る

        どちらも履歴の名前を返す 名前と場所だけを断った人には「動き: 対象」の対象を形で伏せ
        （labels）、字幕と文字を断った人には断らずに文字を伏せる（hidden）
        """
        undo = find_operation("undo")
        redo = find_operation("redo")
        assert undo is not None and redo is not None
        assert (redo.sends, redo.hidden, redo.labels, redo.safety) == (
            undo.sends,
            undo.hidden,
            undo.labels,
            undo.safety,
        )

    def test_what_redo_returns_matches_its_declaration(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """壊れると、やり直しの名前に入る文の頭と素材の名前を、断った人にも尋ねず伏せずに送る

        変える道具の例（test_outbound.py の WRITE_EXAMPLES）は、作ったばかりの作品で動かすので
        やり直す物が無く、断りの文しか見ない ここでは取り消した後にやり直して名前を返させる
        """
        host = _with_text_clip(video_media, transcript)
        added = replace(video_media, path=SECRET_PATH.with_name("別の撮影.mp4"), id=new_media_id())
        host.document.execute(AddMedia(added))
        host.document.undo()
        host.document.undo()
        host.document.undo()
        operation = find_operation("redo")
        assert operation is not None

        result = operation(host, {"steps": 3})

        assert isinstance(result, dict) and len(result["redone"]) == 3
        places = {"別の撮影", str(SECRET_PATH)}
        assert not undeclared(operation, result, sentences={BODY_TEXT[:12]}, places=places)


class TestRedoWithoutSendingText:
    def test_redo_still_works_and_the_text_is_hidden(self, host: FakeHost) -> None:
        """壊れると、字幕と文字を送らない人は AI にやり直しを頼めないか、やり直した文が外へ出る"""
        run(host, "add_text", text=SECRET_TEXT)
        host.document.undo()
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices(
            {Outbound.FRAMES: SendChoice.ASK, Outbound.TEXT: SendChoice.DENY}
            | {Outbound.FILES: SendChoice.ALLOW}
        )
        operation = find_operation("redo")
        assert operation is not None

        done, box = call_in_thread(bridge, operation, {})
        asked = answer_sends(bridge, done, lambda _request: None)

        assert asked == []
        assert box.get("error") is None, box.get("error")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密" not in sent
        assert SECRET_TEXT[:12] not in sent
        # どの操作をやり直したかは読める
        assert "テキストを追加" in sent

    def test_history_hides_the_text_waiting_to_be_redone(self, host: FakeHost) -> None:
        """壊れると、取り消した文の頭がやり直しの名前に残り、履歴を読むだけで外へ出る"""
        run(host, "add_text", text=SECRET_TEXT)
        host.document.undo()
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices(
            {Outbound.FRAMES: SendChoice.ASK, Outbound.TEXT: SendChoice.DENY}
            | {Outbound.FILES: SendChoice.ALLOW}
        )
        operation = find_operation("get_history")
        assert operation is not None

        done, box = call_in_thread(bridge, operation, {})
        answer_sends(bridge, done, lambda _request: None)

        assert box.get("error") is None, box.get("error")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密" not in sent
        assert "テキストを追加" in sent
