"""AI のナレーションの道具（list_voices・generate_narration #335）

2026-10-20 に利用者が決めたことに従っているかを、偽のホストと偽のエンジン（127.0.0.1 の
本物の HTTP の相手 tests/fake_voice_engine.py）で押さえる

- 本人が起動した VOICEVOX 互換のエンジンを呼ぶだけ 自分の機械の中の HTTP だけ
- WAV は作品のフォルダの narration へ書く 保存していない作品では断り、保存を案内する
- 書く・読み込む・置くを 1 段にまとめ、取り消し 1 回で戻す（書いた WAV は残す）
- 使った声の表記を作品に覚え、書き出しのときに知らせる
- エンジンが動いていない機械（CI）では断って理由を返す

本物の窓で作って置く試験は tests/ui/test_narration_ui.py
"""

from __future__ import annotations

import json
import struct
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai import operations
from sashimono.ai.bridge import Approval, EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Operation, find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddCredits, Author
from sashimono.core.io.atomic import expect_now
from sashimono.core.model import TrackKind
from sashimono.engine.audio.voicevox import (
    NarrationSettings,
    VoiceEngine,
    check_engine_url,
    wav_seconds,
)
from sashimono.engine.decode import probe_media
from tests.ai.conftest import FakeHost, allow_sending
from tests.ai.test_bridge import call_in_thread
from tests.ai.test_fix_loop import _MIXED_CLIP, _finish
from tests.ai.test_outbound import undeclared
from tests.ai.test_safety import _symlink_dir, answer_approvals
from tests.fake_voice_engine import (
    SECONDS_PER_CHAR,
    FakeVoiceEngine,
    closed_engine_url,
    fake_voice_engine,
)

WINDOWS_WRITES = pytest.mark.skipif(
    sys.platform != "win32", reason="AI がファイルを書くのは Windows だけ（pin_holds_the_folder）"
)

SCRIPT = "今日は動画の切り方を話します。最後まで見てください。"


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


LIST = _operation("list_voices")
GENERATE = _operation("generate_narration")


@pytest.fixture
def engine() -> Iterator[FakeVoiceEngine]:
    with fake_voice_engine() as running:
        yield running


@pytest.fixture
def narrating(
    host: FakeHost, tmp_path: Path, engine: FakeVoiceEngine, monkeypatch: Any
) -> FakeHost:
    """作品を tmp_path/作品/作品.sme に保存し、偽のエンジンへ向けたホスト 素材は本物で読む"""
    folder = tmp_path / "作品"
    folder.mkdir()
    host.saved_at = folder / "作品.sme"
    host.write_project(host.saved_at, adopt=False, expect=expect_now(host.saved_at))
    host.narration = NarrationSettings(url=engine.url)
    # 書いた WAV を本物の読み方で読む 偽の素材を返すと、書いた中身が読める形かを見逃す
    monkeypatch.setattr(host, "probe", probe_media)
    return host


def _run(
    bridge: EditorBridge,
    operation: Operation,
    arguments: dict[str, Any],
    reply: Callable[[Approval], bool] = lambda _approval: True,
) -> tuple[list[Approval], dict[str, object]]:
    done, box = call_in_thread(bridge, operation, arguments)
    seen = answer_approvals(bridge, done, reply, timeout=15.0)
    return seen, box


def _listed(host: FakeHost, arguments: dict[str, Any] | None = None) -> dict[str, object]:
    bridge = allow_sending(EditorBridge(host))
    _seen, box = _run(bridge, LIST, arguments or {})
    if "error" in box:
        raise box["error"]  # type: ignore[misc]  # 試験の中で断りの文を見るため投げ直す
    result = box["result"]
    assert isinstance(result, dict)
    return result


def _narration_files(host: FakeHost) -> list[str]:
    assert host.saved_at is not None
    folder = host.saved_at.parent / "narration"
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


class TestKinds:
    def test_listing_reads_and_making_is_irreversible(self) -> None:
        # 壊れると、ファイルを書く道具が「取り消せる編集」と扱われ、「変更を自動で承認」の人の
        # 作品のフォルダへ黙って WAV を書く（書き出しと同じく毎回確かめる）
        assert LIST.safety is Safety.READ_ONLY
        assert GENERATE.safety is Safety.IRREVERSIBLE


class TestListingVoices:
    def test_voices_the_default_and_its_terms_are_returned(self, narrating: FakeHost) -> None:
        # 壊れると、AI が声の名前と ID を知らずに作るか、既定の声の規約をユーザーに伝えられない
        result = _listed(narrating)
        voices = result["voices"]
        assert isinstance(voices, list)
        assert [row["speaker"] for row in voices] == ["四国めたん", "ずんだもん"]
        assert voices[1]["credit"] == "VOICEVOX:ずんだもん"
        assert voices[1]["styles"] == [{"voice_id": 3, "name": "ノーマル"}]
        assert result["default_voice"] == "四国めたん（ノーマル）"
        assert result["terms"] == "四国めたんの規約"
        assert result["ai_may_choose_voice"] is True

    def test_the_terms_of_a_named_voice_are_returned(self, narrating: FakeHost) -> None:
        # 壊れると、AI が選んだ声の規約を読めず、非商用だけの声を断りなく使わせる
        result = _listed(narrating, {"voice_id": 3})
        assert result["terms_voice"] == "ずんだもん（ノーマル）"
        assert result["terms"] == "ずんだもんの規約 クレジットを書くこと"

    def test_a_stopped_engine_is_refused_with_how_to_fix_it(self, host: FakeHost) -> None:
        # 壊れると、エンジンを起動していない機械（CI もそう）で AI が理由を知らずに頼み続ける
        host.narration = NarrationSettings(url=closed_engine_url())
        with pytest.raises(ToolError, match="起動"):
            _listed(host)

    def test_an_outside_engine_place_is_refused_without_connecting(self, host: FakeHost) -> None:
        # 壊れると、設定ファイルを書き換えた外の場所へ台本や頼みを送る
        host.narration = NarrationSettings(url="http://192.168.1.10:50021")
        with pytest.raises(ToolError, match="自分の機械の中"):
            LIST(host, {})


class TestRefusedBeforeAsking:
    def test_an_unsaved_project_is_told_to_save_first(
        self, host: FakeHost, engine: FakeVoiceEngine
    ) -> None:
        # 壊れると、保存していない作品で WAV を書く場所が無いのに確かめの窓が出るか、
        # どこか知らない場所へ書く（2026-10-20 利用者の決定 保存を案内する）
        host.narration = NarrationSettings(url=engine.url)
        _listed(host)
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        seen, box = _run(bridge, GENERATE, {"text": SCRIPT})
        assert seen == []
        assert "保存" in str(box.get("error"))
        assert engine.texts == []

    def test_the_voice_list_is_read_first(self, narrating: FakeHost) -> None:
        # 壊れると、確かめの窓に声の名前と表記を出せないまま作るか、窓を出す画面のスレッドで
        # 止まったエンジンを待って画面が固まる
        operations._VOICE_CATALOGS.pop(narrating.narration.url, None)
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        seen, box = _run(bridge, GENERATE, {"text": SCRIPT})
        assert seen == []
        assert "list_voices" in str(box.get("error"))

    def test_a_too_long_script_is_refused_without_quoting_it(self, narrating: FakeHost) -> None:
        # 壊れると、長い台本で待ちが何分も続くか、断りの文に台本を書き返して外へ送る
        _listed(narrating)
        secret = "秘密の台本" * 200
        with pytest.raises(ToolError) as caught:
            GENERATE.consent(narrating, {"text": secret})  # type: ignore[misc]
        assert "長すぎ" in str(caught.value)
        assert "秘密" not in str(caught.value)

    def test_the_owner_can_keep_the_ai_to_the_default_voice(self, narrating: FakeHost) -> None:
        # 壊れると、いつも同じ声で読ませたい人の設定を無視して、AI が声を選び直す
        _listed(narrating)
        narrating.narration = NarrationSettings(url=narrating.narration.url, ai_picks_voice=False)
        assert GENERATE.consent is not None
        with pytest.raises(ToolError, match="既定の声だけ"):
            GENERATE.consent(narrating, {"text": SCRIPT, "voice_id": 3})
        # 既定の声のままなら作れる
        consent = GENERATE.consent(narrating, {"text": SCRIPT})
        assert "四国めたん" in consent.detail

    def test_an_unknown_voice_is_refused(self, narrating: FakeHost) -> None:
        # 壊れると、一覧に無い声を頼んだのに確かめの窓が出て、許した後にエンジンに断られる
        _listed(narrating)
        assert GENERATE.consent is not None
        with pytest.raises(ToolError, match="一覧にありません"):
            GENERATE.consent(narrating, {"text": SCRIPT, "voice_id": 999})

    def test_a_narration_folder_linked_outside_is_refused(
        self, narrating: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、作品のフォルダに置いたリンク（ジャンクション）を通って、AI が書ける場所の
        # 外へ WAV を書く（#350 書けるのは作品のフォルダと許した所だけ）
        _listed(narrating)
        assert narrating.saved_at is not None
        outside = tmp_path / "外"
        outside.mkdir()
        _symlink_dir(narrating.saved_at.parent / "narration", outside)
        assert GENERATE.consent is not None
        with pytest.raises(ToolError, match="外"):
            GENERATE.consent(narrating, {"text": SCRIPT})
        assert list(outside.iterdir()) == []

    def test_a_file_named_narration_is_not_replaced(self, narrating: FakeHost) -> None:
        # 壊れると、作品のフォルダにある narration という名前のファイルを潰してフォルダにする
        _listed(narrating)
        assert narrating.saved_at is not None
        blocker = narrating.saved_at.parent / "narration"
        blocker.write_bytes(b"keep")
        assert GENERATE.consent is not None
        with pytest.raises(ToolError, match="作れません"):
            GENERATE.consent(narrating, {"text": SCRIPT})
        assert blocker.read_bytes() == b"keep"


@WINDOWS_WRITES
class TestMakingAndPlacing:
    def test_auto_approve_still_asks_and_shows_the_voice_and_the_text(
        self, narrating: FakeHost, engine: FakeVoiceEngine
    ) -> None:
        # 壊れると、「変更を自動で承認」の人の作品のフォルダへ黙って WAV を書くか、どの声で
        # 何を読ませるのかを見ないまま許すことになる
        _listed(narrating)
        folder_seen: list[bool] = []

        def look(approval: Approval) -> bool:
            assert narrating.saved_at is not None
            # 許す前はフォルダも作らない 断られたら何も残さない
            folder_seen.append((narrating.saved_at.parent / "narration").exists())
            return True

        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        seen, box = _run(bridge, GENERATE, {"text": SCRIPT, "voice_id": 3}, look)
        assert box.get("error") is None, box.get("error")
        assert len(seen) == 1
        assert seen[0].irreversible
        consent = seen[0].consent
        assert consent is not None
        assert consent.target.endswith("narration_001.wav")
        assert "ずんだもん（ノーマル）" in consent.detail
        assert "VOICEVOX:ずんだもん" in consent.detail
        # 台本は箱に出す（人が読んで許す）が、AI へ返る detail には入れない
        assert SCRIPT in seen[0].summary
        assert SCRIPT not in consent.detail
        assert "narration フォルダを作ります" in consent.detail
        assert folder_seen == [False]
        assert engine.texts  # 声はエンジンで作った

    def test_a_change_while_asking_does_not_send_the_script(self, narrating: FakeHost) -> None:
        # 壊れると、確かめの箱が出ている間に人が再生ヘッドを動かしたとき、照らし合わせの断りの
        # 文に台本の全文が 2 回入って AI へ返り、字幕と文字を送らないと決めた人の文まで送る
        # （#401 の CodeRabbit の指摘）
        _listed(narrating)

        def move(_approval: Approval) -> bool:
            narrating.seek(narrating.playhead + 15)
            return True

        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, GENERATE, {"text": SCRIPT}, move)
        error = str(box.get("error"))
        assert "中身が変わりました" in error
        assert "置く所" in error
        assert "動画の切り方" not in error
        assert "最後まで見て" not in error
        assert narrating.narrations == []

    def test_the_wave_is_written_read_and_placed_and_undone_in_one_step(
        self, narrating: FakeHost
    ) -> None:
        # 壊れると、声を置くのに何段も積まれ、Ctrl+Z 1 回で頼む前へ戻らない（素材だけ残る・
        # 表記だけ残る） 戻すと書いた WAV まで消えると、人が作り直す手間が増える
        _listed(narrating)
        before = narrating.document.project
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, GENERATE, {"text": SCRIPT, "voice_id": 3, "at_frame": 45})
        assert box.get("error") is None, box.get("error")
        result = box["result"]
        assert isinstance(result, dict)

        assert _narration_files(narrating) == ["narration_001.wav"]
        written = Path(str(result["file"]))
        assert written.read_bytes() == narrating.narrations[0][1]
        project = narrating.document.project
        media = next(m for m in project.media if m.path == written)
        assert media.has_audio and not media.has_video
        clips = [
            (track, clip)
            for track in project.timeline.tracks
            for clip in track.clips
            if clip.media_id == media.id
        ]
        assert len(clips) == 1
        track, clip = clips[0]
        assert track.kind is not TrackKind.VIDEO
        assert clip.timeline_start == 45
        assert project.credits == ("VOICEVOX:ずんだもん",)
        assert result["credit"] == "VOICEVOX:ずんだもん"
        assert result["seconds"] == pytest.approx(wav_seconds(written.read_bytes()), abs=0.01)
        # 台本の文は結果に載せない（外へ送らない） 載せた物（書いた場所）は宣言どおり
        assert "動画の切り方" not in json.dumps(result, ensure_ascii=False)
        assert not undeclared(GENERATE, result, sentences={SCRIPT}, places={str(written)})
        assert Outbound.FILES in GENERATE.sends

        narrating.document.undo()
        assert narrating.document.project == before
        assert written.exists()

    def test_subtitles_follow_each_sentence_of_the_made_voice(self, narrating: FakeHost) -> None:
        # 壊れると、作った声に字幕が付かないか、文の区切りと声がずれて、2 文目の字幕が
        # 1 文目を読んでいる間に出る（2026-10-20 利用者の決定 文ごと・声の長さに合わせる）
        _listed(narrating)
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, GENERATE, {"text": SCRIPT, "voice_id": 3})
        result = box["result"]
        assert isinstance(result, dict)
        assert result["subtitle_lines"] == 2
        media = next(
            m for m in narrating.document.project.media if m.path == Path(str(result["file"]))
        )
        transcript = media.transcript
        assert transcript is not None
        assert [segment.text for segment in transcript] == [
            "今日は動画の切り方を話します。",
            "最後まで見てください。",
        ]
        first, second = transcript.segments
        assert first.start == 0
        assert float(first.end) == pytest.approx(15 * SECONDS_PER_CHAR, abs=0.001)
        assert second.start == first.end
        assert float(second.end) == pytest.approx(media.duration, abs=0.01)
        assert all(segment.edited for segment in transcript)
        # 字幕も同じ 1 段で戻る
        narrating.document.undo()
        assert all(m.path != media.path for m in narrating.document.project.media)

    def test_the_subtitles_carry_the_voice_as_their_speaker(self, narrating: FakeHost) -> None:
        # 壊れると、話者を分けた起こしの字幕（#323）と並べたときにナレーションの行だけ話者が
        # 無く、AI が get_subtitles の speaker で声の行を絞れず、分け直そうとする
        _listed(narrating)
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, GENERATE, {"text": SCRIPT, "voice_id": 3})
        result = box["result"]
        assert isinstance(result, dict)
        media = next(
            m for m in narrating.document.project.media if m.path == Path(str(result["file"]))
        )
        assert media.transcript is not None
        assert {segment.speaker for segment in media.transcript} == {"ずんだもん"}

    def test_the_owner_can_turn_the_subtitles_off_for_the_ai(self, narrating: FakeHost) -> None:
        # 壊れると、字幕を別に起こす人の作品に、AI が作るたびに台本の字幕が重なる
        _listed(narrating)
        narrating.narration = NarrationSettings(url=narrating.narration.url, subtitles=False)
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        # AI は設定に従う（引数で字幕を頼んでも付けない）
        _seen, box = _run(bridge, GENERATE, {"text": SCRIPT, "subtitles": True})
        result = box["result"]
        assert isinstance(result, dict)
        assert result["subtitle_lines"] == 0
        media = next(
            m for m in narrating.document.project.media if m.path == Path(str(result["file"]))
        )
        assert media.transcript is None

    def test_the_person_path_picks_any_voice_and_the_subtitles(self, narrating: FakeHost) -> None:
        # 壊れると、人が窓で選んだ声まで「AI に声を選ばせない」設定で断るか、窓で切った字幕を
        # 付ける
        _listed(narrating)
        narrating.narration = NarrationSettings(url=narrating.narration.url, ai_picks_voice=False)
        arguments = {"text": SCRIPT, "voice_id": 3, "subtitles": False}
        plan = operations.narration_plan(narrating, arguments, person=True)
        assert plan.speaker.name == "ずんだもん"
        assert plan.subtitles is False
        with pytest.raises(ToolError, match="既定の声だけ"):
            operations.narration_plan(narrating, arguments)

    def test_a_second_voice_gets_a_new_name_and_the_credit_once(self, narrating: FakeHost) -> None:
        # 壊れると、2 本目の声が 1 本目を潰すか、同じ表記が何度も並ぶ
        _listed(narrating)
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        for _ in range(2):
            _seen, box = _run(bridge, GENERATE, {"text": SCRIPT, "voice_id": 3})
            assert box.get("error") is None, box.get("error")
        assert _narration_files(narrating) == ["narration_001.wav", "narration_002.wav"]
        assert narrating.document.project.credits == ("VOICEVOX:ずんだもん",)

    def test_denying_writes_and_makes_nothing(
        self, narrating: FakeHost, engine: FakeVoiceEngine
    ) -> None:
        # 壊れると、断ったのに声を作り、作品のフォルダに WAV が残る
        _listed(narrating)
        before = narrating.document.project
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, GENERATE, {"text": SCRIPT}, lambda _approval: False)
        assert isinstance(box.get("error"), ToolError)
        assert engine.texts == []
        assert _narration_files(narrating) == []
        assert narrating.document.project == before

    def test_an_engine_failure_writes_nothing(
        self, narrating: FakeHost, engine: FakeVoiceEngine
    ) -> None:
        # 壊れると、エンジンが途中で断ったのに半端な WAV を書くか、素材だけを置く
        _listed(narrating)
        engine.synthesis_status = 500
        before = narrating.document.project
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, GENERATE, {"text": SCRIPT})
        assert "HTTP 500" in str(box.get("error"))
        assert SCRIPT not in str(box.get("error"))
        assert narrating.narrations == []
        assert narrating.document.project == before

    def test_an_unexpected_failure_is_told_as_a_refusal(
        self, narrating: FakeHost, monkeypatch: Any
    ) -> None:
        # 壊れると、声を作る裏のスレッドで思っていない例外（壊れた WAV を読む所など）が出たとき、
        # 終わった印だけが立って結果が無く、AI には AssertionError、人の窓では「作っています」の
        # まま止まって見える（#401 の CodeRabbit の指摘）
        _listed(narrating)

        def broken(*_args: object, **_kwargs: object) -> object:
            raise struct.error("壊れた WAV の頭")

        monkeypatch.setattr(VoiceEngine, "synthesize", broken)
        before = narrating.document.project
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, GENERATE, {"text": SCRIPT})
        error = box.get("error")
        assert isinstance(error, ToolError), error
        assert "思っていない失敗" in str(error)
        assert "壊れた WAV の頭" not in str(error)
        assert narrating.narrations == []
        assert narrating.document.project == before

    def test_the_fix_limit_also_stops_the_narration(self, narrating: FakeHost) -> None:
        # 壊れると、検査の直しの回数の上限（#339）に達して「これ以上は編集しません」と伝えた後に、
        # 取り消せない操作の道を通って声と字幕が置かれる
        _listed(narrating)
        narrating.fix_limit = 0
        narrating.document.begin_checkpoint("AI: 調べて", by=Author.AI)
        _finish(narrating, [_MIXED_CLIP])
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        seen, box = _run(bridge, GENERATE, {"text": SCRIPT})
        narrating.document.end_checkpoint()
        assert seen == []  # 確かめの窓を出す前に断る
        assert "上限" in str(box.get("error"))
        assert narrating.narrations == []

    def test_reopening_the_project_in_the_instruction_stops_the_narration(
        self, narrating: FakeHost
    ) -> None:
        # 壊れると、同じ指示の中で作品を開き直した（#349）後に確かめの窓が出て WAV を書き、
        # 置く所で文書に断られて、開き直した作品のフォルダに誰も置かない WAV が残る
        _listed(narrating)
        narrating.document.begin_checkpoint("AI: 開いて読む", by=Author.AI)
        narrating.document.reset(narrating.document.project, within_checkpoint=True)
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        seen, box = _run(bridge, GENERATE, {"text": SCRIPT})
        narrating.document.end_checkpoint()
        assert seen == []  # 確かめの窓を出す前に断る
        assert "開き直した" in str(box.get("error"))
        assert narrating.narrations == []

    def test_moving_the_project_while_the_voice_is_made_is_refused(
        self, narrating: FakeHost, engine: FakeVoiceEngine, tmp_path: Path
    ) -> None:
        # 壊れると、声を作っている間に〔名前を付けて保存〕で別のフォルダへ移した人の、前の
        # 作品のフォルダへ WAV を書き、今の作品からは見えない所に素材が増える
        _listed(narrating)
        engine.synthesis_delay = 0.5
        other = tmp_path / "移した先"
        other.mkdir()
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        done, box = call_in_thread(bridge, GENERATE, {"text": SCRIPT})
        deadline = time.monotonic() + 15
        moved = False
        while not done.is_set() and time.monotonic() < deadline:
            bridge.pump()
            approval = bridge.take_approval()
            if approval is not None:
                approval.allow()
            if not moved and engine.texts:
                narrating.saved_at = other / "作品.sme"
                moved = True
            time.sleep(0.005)
        bridge.pump()
        assert done.is_set()
        assert "作品の場所が変わった" in str(box.get("error"))
        assert narrating.narrations == []


_ROOT = Path(__file__).resolve().parents[2]


class TestTheDocumentsMatchTheCode:
    def test_the_accepted_engine_places_are_described_as_they_are(self) -> None:
        """壊れると、文書には 127.0.0.1・::1・localhost だけと書いてあるのに、実装は 127.0.0.2 など
        ほかのループバックの番地も通し、読んだ人がどこまでつながるかを取り違える
        （#401 の CodeRabbit の指摘）"""
        assert check_engine_url("http://127.0.0.2:50021") == "http://127.0.0.2:50021"
        development = (_ROOT / "docs" / "development.md").read_text(encoding="utf-8")
        policy = (_ROOT / "docs" / "code-signing-policy.md").read_text(encoding="utf-8")
        section = development[development.index("### ナレーションの生成（#335）") :]
        line = next(row for row in section.splitlines() if "check_engine_url" in row)
        assert "127.0.0.0/8" in line
        rows = [
            row
            for row in policy.splitlines()
            if row.startswith(("| Narration voices", "| ナレーションの声"))
        ]
        assert len(rows) == 2
        assert all("127.0.0.0/8" in row for row in rows)

    def test_the_readme_says_list_voices_reads_one_voice_terms(self) -> None:
        """壊れると、README は声ごとに規約を読むと書くのに、list_voices は頼んだ声か既定の声の
        1 つだけを読み、読んだ人が全部の声の規約を確かめたつもりになる
        （#401 の CodeRabbit の指摘）"""
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        start = readme.index("**台本を読み上げたナレーションを作って置けます**")
        paragraph = readme[start : readme.index("\n\n", start)]
        assert "声ごとの規約を読み" not in paragraph
        assert "1 つの規約" in paragraph


class TestCreditsAreTold:
    def test_the_project_reports_its_credits(self, host: FakeHost) -> None:
        # 壊れると、AI が作品の表記を知らずに書き出し、ユーザーに伝えられない
        host.apply_commands([AddCredits(("VOICEVOX:ずんだもん",))], "表記")
        result = _operation("get_project")(host, {})
        assert isinstance(result, dict)
        assert result["credits"] == ["VOICEVOX:ずんだもん"]

    @WINDOWS_WRITES
    def test_the_export_answer_carries_the_credits(
        self, narrating: FakeHost, monkeypatch: Any
    ) -> None:
        # 壊れると、AI が書き出した後に表記を伝えず、本人が概要欄に入れ忘れる（規約に反する）
        monkeypatch.setattr(operations, "available_video_codecs", lambda: ["libx264"])
        narrating.apply_commands([AddCredits(("VOICEVOX:ずんだもん",))], "表記")
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, _operation("export_video"), {"path": "完成.mp4"})
        assert box.get("error") is None, box.get("error")
        result = box["result"]
        assert isinstance(result, dict)
        assert result["credits"] == ["VOICEVOX:ずんだもん"]
        assert "表記" in str(result["credit_note"])

    @WINDOWS_WRITES
    def test_an_export_without_voices_says_nothing_about_credits(
        self, narrating: FakeHost, monkeypatch: Any
    ) -> None:
        # 壊れると、声を使っていない作品でも表記の知らせが出て、知らせを読み流す癖が付く
        monkeypatch.setattr(operations, "available_video_codecs", lambda: ["libx264"])
        bridge = allow_sending(EditorBridge(narrating, auto_approve=True))
        _seen, box = _run(bridge, _operation("export_video"), {"path": "完成.mp4"})
        result = box["result"]
        assert isinstance(result, dict)
        assert "credits" not in result
