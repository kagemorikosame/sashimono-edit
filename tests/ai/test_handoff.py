"""AI が途中で止まったときの引き継ぎ（#346 D8）

要約の中身（変えた手・取り消しの段・止まった訳・残り）を、AI の言葉ではなく作品の履歴と
画面が見た出来事から組むこと、AI が道具（get_handoff）で読むときに外へ送る物の宣言と
同意（#342）を通ることを見る 画面の側（止まる道ごとに欄に出ること）は
tests/ui/test_chat_handoff.py で見る
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.fixes import fix_label
from sashimono.ai.handoff import (
    AI_WORDS_LIMIT,
    GET_HANDOFF,
    ITEMS_LIMIT,
    Handoff,
    HandoffShelf,
    StopReason,
    TurnRecord,
    shelf_for,
)
from sashimono.ai.operations import Operation, find_operation
from sashimono.ai.outbound import TEXT_MARK, Outbound, SendChoice
from sashimono.ai.session import LIMIT_DETAIL, LOGIN_DETAIL, EventKind, result_error
from sashimono.core.commands import Author, Document, RenameProject, SplitClip
from sashimono.core.model import MediaItem, Project, Transcript
from tests.ai.conftest import FakeHost, allow_sending, make_loaded
from tests.ai.test_bridge import call_in_thread, pump_until
from tests.ai.test_fix_loop import _MIXED_CLIP, _finish, _fix
from tests.ai.test_outbound import SECRET_PATH, answer_sends, put_handoff, undeclared


def assert_split_note(note: str) -> None:
    """人が割った指示の取り消しの案内 一度には戻せないことと、人の段を巻き込むことを言う

    「Ctrl+Z でこの指示の前へ戻ります」と言い切ったまま注記を足すと、AI の変更だけを
    安全に戻せると読める（Codex の指摘 #396）
    """
    assert "この指示の前へ戻ります" not in note
    assert "一度には戻せません" in note
    assert "人の変更も一緒に戻ります" in note


def _operation() -> Operation:
    found = find_operation(GET_HANDOFF)
    assert found is not None
    return found


def _first_clip(document: Document) -> Any:
    return document.project.timeline.tracks[0].clips[0].id


def _ai_turn(document: Document, label: str = "AI: 冒頭を切って", cuts: int = 2) -> None:
    """AI の指示の段を開き、分割を ``cuts`` 回する 閉じない（指示の途中で止まった形）"""
    document.begin_checkpoint(label, by=Author.AI)
    # 頭のクリップを後ろから切っていく 前から切ると、頭のクリップが切る位置より短くなる
    for frame in (90, 60, 30)[:cuts]:
        with document.checkpoint("分割", by=Author.AI):
            document.execute(SplitClip(_first_clip(document), frame))


class TestRecord:
    def test_a_turn_that_did_not_stop_has_no_handoff(self, loaded: Project) -> None:
        # 壊れると、ふつうに終えた指示のたびに「止まりました」の要約が出て、
        # 本当に止まった時に読まれない
        record = TurnRecord()
        record.begin("切って")
        record.tool("split_clip")
        record.said("切りました")
        assert record.handoff(Document(loaded), "AI: 切って") is None

    @pytest.mark.parametrize("reason", list(StopReason))
    def test_every_way_of_stopping_makes_a_handoff(
        self, reason: StopReason, loaded: Project
    ) -> None:
        # 壊れると、どれかの止まり方（例外・中断・上限・断られた・接続が切れた）で要約が出ず、
        # 人は半端な編集の上に次の指示を重ねる
        record = TurnRecord()
        record.begin("切って")
        if reason is StopReason.DENIED:
            record.refused("split_clip（許可しませんでした）")
        else:
            record.stop(reason, "訳の文")
        document = Document(loaded)
        handoff = record.handoff(document, "AI: 切って")
        assert handoff is not None
        assert handoff.reason is reason
        assert reason.label in handoff.text(document)

    def test_an_interrupt_wins_over_the_error_that_follows_it(self, loaded: Project) -> None:
        # 壊れると、人が〔中断〕を押したのに、巻き添えの「中断されました」のエラーで AI が
        # 失敗したと読める
        record = TurnRecord()
        record.begin("切って")
        record.stop(StopReason.ERROR, "ToolError: 中断されました")
        record.stop(StopReason.INTERRUPTED, "人が〔中断〕を押しました")
        handoff = record.handoff(Document(loaded), "AI: 切って")
        assert handoff is not None
        assert handoff.reason is StopReason.INTERRUPTED
        # 負けた訳の文も残す 何が起きていたかを言っていることがある
        assert "中断されました" in handoff.detail

    def test_refusals_are_kept_even_when_another_reason_wins(self, loaded: Project) -> None:
        # 壊れると、エラーで止まった指示の中で許さなかった操作が要約から消え、人がやり直しを頼むと
        # 許さなかった操作をまた頼まれる
        record = TurnRecord()
        record.begin("保存して")
        record.refused("save_project（許可しませんでした）")
        record.stop(StopReason.ERROR, "失敗")
        handoff = record.handoff(Document(loaded), "AI: 保存して")
        assert handoff is not None
        assert handoff.reason is StopReason.ERROR
        assert handoff.refused == ("save_project（許可しませんでした）",)

    def test_a_retry_that_worked_takes_the_reason_back(self, loaded: Project) -> None:
        # 壊れると、部品を新しくして送り直しが通った指示まで「エラーで止まりました」と出る
        record = TurnRecord()
        record.begin("切って")
        record.stop(StopReason.ERROR, "部品が古い")
        record.forget(StopReason.ERROR)
        assert record.handoff(Document(loaded), "AI: 切って") is None

    def test_beginning_again_forgets_the_previous_instruction(self, loaded: Project) -> None:
        # 壊れると、前の指示の失敗や許さなかった物が、次の指示の要約に混ざる
        record = TurnRecord()
        record.begin("前の指示")
        record.failed("split_clip: 失敗")
        record.refused("save_project（許可しませんでした）")
        record.stop(StopReason.ERROR, "前のエラー")
        record.begin("次の指示")
        record.stop(StopReason.INTERRUPTED)
        handoff = record.handoff(Document(loaded), "AI: 次の指示")
        assert handoff is not None
        assert (handoff.failures, handoff.refused, handoff.detail) == ((), (), "")
        assert handoff.instruction == "次の指示"

    def test_long_lists_keep_the_newest(self, loaded: Project) -> None:
        # 壊れると、何十件も続いた失敗で要約が読めない長さになり、止まる直前の失敗が埋もれる
        record = TurnRecord()
        record.begin("切って")
        for number in range(ITEMS_LIMIT + 5):
            record.failed(f"失敗 {number}")
        record.said("あ" * (AI_WORDS_LIMIT * 2))
        record.stop(StopReason.ERROR)
        handoff = record.handoff(Document(loaded), "AI: 切って")
        assert handoff is not None
        assert len(handoff.failures) == ITEMS_LIMIT
        assert handoff.failures[-1] == f"失敗 {ITEMS_LIMIT + 4}"
        assert len(handoff.ai_words) == AI_WORDS_LIMIT


class TestChangesComeFromTheHistory:
    def test_the_changes_are_the_steps_in_the_instruction(self, loaded: Project) -> None:
        # 壊れると、要約の「変えた物」が AI の言葉だけになり、AI が「何も変えていません」と
        # 言ったまま 2 回切った作品が残っても、人は気付かない
        document = Document(loaded)
        _ai_turn(document)
        record = TurnRecord()
        record.begin("冒頭を切って")
        record.said("何も変えていません")
        record.capture(document)
        document.end_checkpoint()
        record.stop(StopReason.ERROR, "落ちた")
        handoff = record.handoff(document, "AI: 冒頭を切って")
        assert handoff is not None
        assert handoff.changes == ("分割", "分割")
        assert handoff.ai_words == "何も変えていません"
        # 書いた取り消しの段の名前が、本当に次に戻す段の名前と同じ
        assert document.undo_label == handoff.checkpoint
        assert "Ctrl+Z" in handoff.undo_note(document)
        assert "AI の言葉" in handoff.text(document)

    def test_an_instruction_that_changed_nothing_says_so(self, loaded: Project) -> None:
        # 壊れると、何も変えずに止まった指示でも「Ctrl+Z で戻す」と案内し、人の前の編集を戻させる
        document = Document(loaded)
        document.begin_checkpoint("AI: 切って", by=Author.AI)
        record = TurnRecord()
        record.begin("切って")
        record.capture(document)
        document.end_checkpoint()
        record.stop(StopReason.INTERRUPTED)
        handoff = record.handoff(document, "AI: 切って")
        assert handoff is not None
        assert handoff.changed is False
        assert "Ctrl+Z" not in handoff.text(document)
        assert handoff.as_dict(document)["checkpoint"] is None

    def test_a_split_instruction_is_pointed_out(self, loaded: Project) -> None:
        # 壊れると、人が途中で変えて 2 段に割れた指示を、Ctrl+Z 1 回で戻し切れると案内してしまう
        document = Document(loaded)
        _ai_turn(document, cuts=1)
        document.execute(SplitClip(_first_clip(document), 15), by=Author.PERSON)
        with document.checkpoint("分割", by=Author.AI):
            document.execute(SplitClip(_first_clip(document), 5))
        record = TurnRecord()
        record.begin("冒頭を切って")
        record.capture(document)
        document.end_checkpoint()
        record.stop(StopReason.ERROR)
        handoff = record.handoff(document, "AI: 冒頭を切って")
        assert handoff is not None
        assert handoff.split is True
        assert_split_note(handoff.undo_note(document))
        # 人が割る前の手も変えた手に残る 割った所で段の中の名前は消えるので、写しておかないと抜ける
        assert handoff.changes == ("分割", "分割")

    def test_changes_before_a_person_split_survive_an_idle_ai(self, loaded: Project) -> None:
        # 壊れると、AI が 2 回変えた後に人が 1 回変え、AI が何もせずに止まると、引き継ぎに
        # 「変えた手 0 件」「変えた物: 無し」と出て、人は AI の手が残っていることに気付かない
        document = Document(loaded)
        _ai_turn(document, cuts=2)
        document.execute(SplitClip(_first_clip(document), 15), by=Author.PERSON)
        record = TurnRecord()
        record.begin("冒頭を切って")
        record.capture(document)
        document.end_checkpoint()
        record.stop(StopReason.INTERRUPTED)
        handoff = record.handoff(document, "AI: 冒頭を切って")
        assert handoff is not None
        assert handoff.changes == ("分割", "分割")
        assert "変えた手 2 件" in handoff.headline()
        assert "・無し" not in handoff.text(document)

    def test_capture_outside_an_instruction_keeps_what_was_taken(self, loaded: Project) -> None:
        # 壊れると、段を閉じた後にもう 1 度写すと空で上書きし、変えた手が要約から消える
        document = Document(loaded)
        _ai_turn(document, cuts=1)
        record = TurnRecord()
        record.capture(document)
        document.end_checkpoint()
        record.capture(document)
        record.stop(StopReason.ERROR)
        handoff = record.handoff(document, "AI: 冒頭を切って")
        assert handoff is not None
        assert handoff.changes == ("分割",)


def _cut(document: Document, by: Author) -> None:
    """頭のクリップの真ん中で切る AI なら道具 1 回分の段にする"""
    clip = document.project.timeline.tracks[0].clips[0]
    frame = (clip.timeline_start + clip.timeline_end) // 2
    if by is Author.AI:
        with document.checkpoint("分割", by=Author.AI):
            document.execute(SplitClip(clip.id, frame))
    else:
        document.execute(SplitClip(clip.id, frame), by=Author.PERSON)


def _stopped_after(document: Document, moves: str) -> Handoff:
    """指示の段を開いて ``moves`` を当て、止まった引き継ぎを作る

    ``a`` は AI の道具 1 回 ``p`` は人の変更 ``|`` は検査の直しの回で区切る
    """
    document.begin_checkpoint("AI: 冒頭を切って", by=Author.AI)
    for move in moves:
        if move == "|":
            assert document.split_checkpoint(fix_label(1))
        else:
            _cut(document, Author.AI if move == "a" else Author.PERSON)
    record = TurnRecord()
    record.begin("冒頭を切って")
    record.capture(document)
    document.end_checkpoint()
    record.stop(StopReason.ERROR)
    handoff = record.handoff(document, "AI: 冒頭を切って")
    assert handoff is not None
    return handoff


#: 指示の中の手・止まった後の手 → 案内に出る言葉・出ない言葉
#: 取り消しの先頭が AI の段か人の段か、直しの回で区切ったか、割れていないかを組み合わせる
_NOTE_TABLE = [
    # 割れていない AI の段が先頭
    ("a", "", ["「AI: 冒頭を切って」を戻すと、この指示の前へ戻ります"], ["人の変更"]),
    # 割れていないが、止まった後に人が変えて人の段が先頭
    ("a", "p", ["先に戻るのは人の変更", "先にそれを戻す必要"], ["1 回押すと"]),
    # 人が割り、最後に AI が変えた AI の段が先頭
    ("apa", "", ["1 回押すと最後の AI の段", "一度には戻せません"], ["先に戻るのは"]),
    # 人が割り、AI は何も足さずに止まった 人の段が先頭（CodeRabbit の指摘 #396）
    ("ap", "", ["先に戻るのは人の変更", "一度には戻せません"], ["1 回押すと"]),
    # 直しの回で区切り、直しの回の段が先頭
    ("a|a", "", ["検査の直し", "この指示の前へ戻ります"], ["人の変更", "先に戻るのは"]),
    # 直しの回で区切った後に人が変えた
    ("a|a", "p", ["先に戻るのは人の変更"], ["1 回押すと"]),
    # 止まった後に人が人の段を戻した 先頭は AI の前半の段に戻る（読む時の並びで書く）
    ("ap", "u", ["1 回押すと最後の AI の段"], ["先に戻るのは"]),
    # 止まった後に別の指示の AI の段が積まれた
    ("a", "n", ["この指示の後に積んだ段"], ["1 回押すと"]),
]


class TestUndoNoteFollowsTheHistory:
    @pytest.mark.parametrize(("inside", "after", "said", "unsaid"), _NOTE_TABLE)
    def test_the_note_names_what_ctrl_z_undoes_now(
        self,
        loaded: Project,
        inside: str,
        after: str,
        said: list[str],
        unsaid: list[str],
    ) -> None:
        # 壊れると、AI が変えた後に人が変えて AI が何も足さずに止まったとき、「1 回押すと最後の
        # AI の段だけが戻る」と案内し、信じて Ctrl+Z を押した人の変更が消える 引き継ぎを作った
        # 後に人が足したり戻したりしても、読む時の並びで案内する
        document = Document(loaded)
        handoff = _stopped_after(document, inside)
        for move in after:
            if move == "p":
                _cut(document, Author.PERSON)
            elif move == "u":
                document.undo()
            else:
                with document.checkpoint("AI: 別の指示", by=Author.AI):
                    _cut(document, Author.AI)
        note = handoff.undo_note(document)
        for words in said:
            assert words in note
        for words in unsaid:
            assert words not in note

    def test_an_instruction_still_open_is_judged_as_closed(self, loaded: Project) -> None:
        # 壊れると、環境が使えずに段を閉じる前に引き継ぎを作る道で、開いている段の手を
        # 「この指示の後に積んだ段」と取り違える
        document = Document(loaded)
        document.begin_checkpoint("AI: 冒頭を切って", by=Author.AI)
        _cut(document, Author.AI)
        record = TurnRecord()
        record.begin("冒頭を切って")
        record.capture(document)
        record.stop(StopReason.ERROR)
        handoff = record.handoff(document, "AI: 冒頭を切って")
        assert handoff is not None
        assert "この指示の前へ戻ります" in handoff.undo_note(document)


class TestShelf:
    def _put(self, document: Document) -> HandoffShelf:
        record = TurnRecord()
        record.begin("切って")
        record.stop(StopReason.ERROR, "落ちた")
        handoff = record.handoff(document, "AI: 切って")
        assert handoff is not None
        shelf = shelf_for(document)
        shelf.put(handoff)
        return shelf

    def test_more_edits_than_the_history_holds_keep_it(self, loaded: Project) -> None:
        # 壊れると、止まった後に同じ作品で履歴の上限を超えて編集すると、止まった時の形が
        # 履歴から落ち、別の作品を開いていないのに get_handoff が空になる（Codex の指摘 #396）
        document = Document(loaded, history_limit=3)
        shelf = self._put(document)
        for number in range(5):
            document.execute(RenameProject(f"名前 {number}"))
        assert shelf.current(document) is not None

    def test_later_edits_and_undo_keep_it(self, loaded: Project) -> None:
        # 壊れると、止まった後に人が 1 手直しただけで、AI が引き継ぎを読めなくなる
        document = Document(loaded)
        shelf = self._put(document)
        document.execute(SplitClip(_first_clip(document), 30))
        assert shelf.current(document) is not None
        document.undo()
        document.undo()
        assert shelf.current(document) is not None

    def test_opening_another_project_drops_it(self, loaded: Project) -> None:
        # 壊れると、別の作品を開いた後に「続きを」と頼むと、前の作品の手順で今の作品を切り詰める
        document = Document(loaded)
        shelf = self._put(document)
        document.reset(replace(loaded, name="別の作品"))
        assert shelf.current(document) is None

    def test_each_document_has_its_own(self, loaded: Project) -> None:
        # 壊れると、2 つ開いた窓の片方で止まった指示の要約を、もう片方の AI が読む
        first, second = Document(loaded), Document(loaded)
        self._put(first)
        assert shelf_for(second).current(second) is None


class TestTheTool:
    def test_nothing_to_hand_off_is_said_plainly(self, host: FakeHost) -> None:
        # 壊れると、止まった指示が無いのに AI が前の要約を探し回るか、
        # 空の辞書を「変えていない」と読む
        result = _operation()(host, {})
        assert isinstance(result, dict)
        assert result["handoff"] is None

    def test_the_handoff_is_read_with_how_many_turns_came_after(self, host: FakeHost) -> None:
        # 壊れると、止まった後に別の指示で形が変わったのに、AI が要約の形を今の形と思って続ける
        document = host.document
        _ai_turn(document)
        record = TurnRecord()
        record.begin("冒頭を切って")
        record.capture(document)
        document.end_checkpoint()
        record.stop(StopReason.LIMIT, "上限")
        handoff = record.handoff(document, "AI: 冒頭を切って")
        assert handoff is not None
        shelf = shelf_for(document)
        shelf.put(handoff)
        shelf.turn_finished()
        result = _operation()(host, {})
        assert isinstance(result, dict)
        assert result["stopped"] == "limit"
        assert result["changes"] == ["分割", "分割"]
        assert result["checkpoint"] == "AI: 冒頭を切って"
        assert result["later_instructions"] == 1

    def test_it_is_a_reading_tool_on_the_ledger(self) -> None:
        # 壊れると、要約を読むだけの道具に確かめの窓が出るか、台帳の見張りの外に置かれる
        from sashimono.ai.ledger import AI_ONLY

        operation = _operation()
        assert operation.writes is False
        assert operation.irreversible is False
        assert GET_HANDOFF in AI_ONLY


def _secret_handoff(host: FakeHost, sentence: str) -> None:
    """素材の場所と字幕の文を含む引き継ぎを置く（失敗した道具の文に場所、指示に字幕の文）"""
    # 字幕の文はかぎ括弧で囲まない 囲むと形で伏せる処理に当たり、表で伏せる所を見ずに通る
    put_handoff(host, f"{sentence} の所で切って", f"split_clip で {SECRET_PATH} を開けません")


class TestWhatIsSent:
    @pytest.fixture
    def secret_host(self, video_media: MediaItem, transcript: Transcript) -> FakeHost:
        return FakeHost(make_loaded(replace(video_media, path=SECRET_PATH), transcript))

    def test_the_declaration_covers_what_it_returns(
        self, secret_host: FakeHost, transcript: Transcript
    ) -> None:
        # 壊れると（宣言から名前と場所か字幕と文字を外すと）、素材の場所や字幕の文を同意なしで送る
        sentence = transcript.segments[0].text
        _secret_handoff(secret_host, sentence)
        operation = _operation()
        result = operation(secret_host, {})
        places = {str(SECRET_PATH), SECRET_PATH.name}
        assert not undeclared(operation, result, sentences={sentence}, places=places)
        weaker_files = replace(operation, sends=operation.sends - {Outbound.FILES})
        assert undeclared(weaker_files, result, sentences={sentence}, places=places)
        weaker_text = replace(
            operation, sends=operation.sends - {Outbound.TEXT}, hidden=frozenset()
        )
        assert undeclared(weaker_text, result, sentences={sentence}, places=places)

    def test_refused_names_and_text_are_hidden(
        self, secret_host: FakeHost, transcript: Transcript
    ) -> None:
        # 壊れると、名前と場所・字幕と文字を断った人の素材の場所と字幕の文が、続きを頼んだ時に
        # 引き継ぎの要約に乗って外へ出る
        sentence = transcript.segments[0].text
        _secret_handoff(secret_host, sentence)
        bridge = EditorBridge(secret_host)
        bridge.set_send_choices(
            {
                Outbound.FILES: SendChoice.DENY,
                Outbound.TEXT: SendChoice.DENY,
                Outbound.FRAMES: SendChoice.DENY,
            }
        )
        done, box = call_in_thread(bridge, _operation(), {})
        # 断った人には尋ねずに動かして伏せる（履歴の道具と同じ 取り消しを頼めなくならないように）
        answer_sends(bridge, done, lambda request: pytest.fail(f"尋ねた: {request.kinds}"))
        sent = repr(box.get("result"))
        assert box.get("error") is None
        assert str(SECRET_PATH) not in sent
        assert SECRET_PATH.name not in sent
        assert sentence not in sent
        assert TEXT_MARK in sent

    def test_allowed_sends_pass_through(
        self, secret_host: FakeHost, transcript: Transcript
    ) -> None:
        # 壊れると、許した人にまで伏せ字の要約が渡り、AI がどの素材で止まったかを読めない
        sentence = transcript.segments[0].text
        _secret_handoff(secret_host, sentence)
        bridge = EditorBridge(secret_host)
        bridge.set_send_choices(dict.fromkeys(Outbound, SendChoice.ALLOW))
        done, box = call_in_thread(bridge, _operation(), {})
        answer_sends(bridge, done, lambda request: pytest.fail(f"尋ねた: {request.kinds}"))
        sent = repr(box.get("result"))
        assert str(SECRET_PATH.name) in sent


class TestLimitFromClaudeCode:
    @pytest.mark.parametrize("subtype", ["error_max_turns", "error_max_budget_usd"])
    def test_a_limit_is_marked_apart_from_errors(self, subtype: str) -> None:
        # 壊れると、Claude Code の手数や費用の上限で止まった指示を「エラー」と引き継ぎ、
        # 送り直せば続くことに気付かない
        event = result_error(subtype, None)
        assert event.kind is EventKind.ERROR
        assert event.detail == LIMIT_DETAIL
        assert subtype in event.text

    def test_other_failures_stay_errors(self) -> None:
        # 壊れると、ログインが要る失敗まで「上限」と読め、ログインの案内も消える
        event = result_error("error_during_execution", "Not logged in /login")
        assert event.detail == LOGIN_DETAIL
        assert "ログイン" in event.text


def test_the_tool_is_named_in_the_instructions() -> None:
    # 壊れると、続きを頼まれた AI が引き継ぎの道具を知らず、前の言葉だけを頼りに続ける
    from sashimono.ai.session import system_prompt

    assert GET_HANDOFF in system_prompt()


class TestFixLoop:
    """検査の直しの繰り返し（#339）と合わせたときの引き継ぎ"""

    def _record(self, host: FakeHost, *, stop: bool = False) -> TurnRecord:
        record = TurnRecord()
        record.begin("仕上げて")
        record.capture(host.document)
        host.document.end_checkpoint()
        if stop:
            record.stop(StopReason.ERROR, "落ちた")
        return record

    def test_every_round_is_kept_in_the_changes(self, host: FakeHost) -> None:
        # 壊れると、直しの回で段を区切った後に止まった指示の要約から、前の回で直した手が消え、
        # 「途中で人が変えた」と人のせいにした案内が出る
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _fix(host, "編集")
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し 1")
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し 2")
        handoff = self._record(host, stop=True).handoff(host.document, "AI: 仕上げて")
        assert handoff is not None
        assert handoff.changes == ("直す", "直す", "直す")
        assert handoff.split is False
        assert handoff.fix_parts == 2
        note = handoff.undo_note(host.document)
        assert "検査の直し" in note
        assert "AI: 仕上げて" in note
        assert "人が変えた" not in note

    def test_a_person_splitting_a_fixed_instruction_is_still_pointed_out(
        self, host: FakeHost
    ) -> None:
        # 壊れると、直しの回の区切りと人の割り込みの区切りを取り違え、人が割った指示を
        # 「直しの回ごと」とだけ案内する
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _fix(host, "編集")
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し 1")
        host.document.execute(RenameProject("人の直し"), by=Author.PERSON)
        _fix(host, "直し 2")
        handoff = self._record(host, stop=True).handoff(host.document, "AI: 仕上げて")
        assert handoff is not None
        assert handoff.split is True
        assert_split_note(handoff.undo_note(host.document))
        # 直しの回で区切った手と、人が割った前半の手（直し 1）の両方が残る
        assert handoff.changes == ("直す", "直す", "直す")

    def test_the_limit_with_problems_left_is_a_stop(self, host: FakeHost) -> None:
        # 壊れると、直す回数の上限で止めた指示は AI が言葉で締めるだけなので止まらずに
        # 終えた扱いになり、直せずに残った問題が人に届かない
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し 1")
        assert _finish(host, [_MIXED_CLIP])["fix_round"]["closed"] is True
        handoff = self._record(host).handoff(host.document, "AI: 仕上げて")
        assert handoff is not None
        assert handoff.reason is StopReason.LIMIT
        assert "直す回数の上限" in handoff.detail
        (left,) = handoff.remaining
        assert "音割れ" in left
        assert "上限で直していない" in left

    def test_reaching_the_limit_with_nothing_left_is_not_a_stop(self, host: FakeHost) -> None:
        # 壊れると、最後の回で直し切った指示にも「上限で止まりました」の要約が出る
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し 1")
        assert _finish(host, [])["fix_round"]["closed"] is True
        assert self._record(host).handoff(host.document, "AI: 仕上げて") is None

    def test_checking_only_is_not_a_stop(self, host: FakeHost) -> None:
        # 壊れると、上限 0（調べるだけ）と本人が決めた作品で調べるたびに「上限で止まりました」が出る
        host.fix_limit = 0
        host.document.begin_checkpoint("AI: 調べて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        assert self._record(host).handoff(host.document, "AI: 調べて") is None

    def test_an_edit_refused_after_the_limit_is_a_stop(self, host: FakeHost) -> None:
        # 壊れると、上限の後に断った編集があっても、AI が言葉で締めれば「完了」と読め、
        # やろうとしてできなかったことが人に届かない
        host.fix_limit = 0
        host.document.begin_checkpoint("AI: 調べて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        tool = find_operation("add_track")
        assert tool is not None
        done, box = call_in_thread(bridge, tool, {"kind": "video"})
        pump_until(bridge, done)
        assert "上限" in str(box["error"])
        handoff = self._record(host).handoff(host.document, "AI: 調べて")
        assert handoff is not None
        assert handoff.reason is StopReason.LIMIT
        assert "1 回断りました" in handoff.detail

    def test_the_limit_of_a_closed_project_is_not_handed_to_the_reopened_one(
        self, host: FakeHost
    ) -> None:
        # 壊れると、直す回数の上限に達した後に AI が新しい作品へ差し替えた（#349）指示で、
        # 閉じた作品の残った問題が新しい作品の引き継ぎに「残り」として並び、
        # 続きを頼まれた AI が今の作品に無いクリップを直しに行く
        host.fix_limit = 1
        host.document.begin_checkpoint("AI: 仕上げて", by=Author.AI)
        _finish(host, [_MIXED_CLIP])
        _fix(host, "直し 1")
        assert _finish(host, [_MIXED_CLIP])["fix_round"]["closed"] is True
        host.saved_project = host.document.project
        host.start_new_project(resolution=None, frame_rate=None)
        assert self._record(host).handoff(host.document, "AI: 仕上げて") is None
