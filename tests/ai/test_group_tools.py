"""AI の道具とグループ（#427 #428）

AI がカットの途中でグループを解かないこと、分割で後ろが元のグループに残らないこと、
グループを解く前に人へ尋ねることを見る
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai import operations as operations_module
from sashimono.ai.bridge import EditorBridge, Question
from sashimono.ai.host import ToolError
from sashimono.ai.operations import UNGROUP_CHOICES, Operation, find_operation
from sashimono.ai.session import system_prompt
from sashimono.core.commands import Command, GroupClips, RippleCut
from sashimono.core.model import (
    Clip,
    GroupId,
    LayerMode,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects.sources import TEXT
from tests.ai.conftest import FakeHost
from tests.ai.test_bridge import call_in_thread, pump_until


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None, name
    return found


def _run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    return _operation(tool)(host, arguments)


def _grouped(*spans: tuple[int, int], locked: int | None = None) -> tuple[FakeHost, list[Clip]]:
    """別々のレイヤーのテキストを 1 つのグループにした作品"""
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    clips = [Clip(timeline_start=s, duration=d, source=TEXT.create()) for s, d in spans]
    tracks = tuple(
        Track(TrackKind.MIXED, f"レイヤー {index + 1}", (clip,), locked=index == locked)
        for index, clip in enumerate(clips)
    )
    project = GroupClips(tuple(clip.id for clip in clips)).apply(
        base.with_timeline(replace(base.timeline, tracks=tracks))
    )
    timeline = project.timeline
    return FakeHost(project), [c for t in timeline.tracks for c in t.clips]


def _with_lone_clip() -> tuple[FakeHost, Clip]:
    """2 本のグループと、どこにも入っていない 1 本"""
    host, _clips = _grouped((0, 100), (0, 100))
    lone = Clip(timeline_start=300, duration=50, source=TEXT.create())
    project = host.document.project
    tracks = (*project.timeline.tracks, Track(TrackKind.MIXED, "レイヤー 3", (lone,)))
    return FakeHost(project.with_timeline(replace(project.timeline, tracks=tracks))), lone


def _take(bridge: EditorBridge, done: threading.Event) -> Question:
    """画面のスレッドの代わりに仕事を拾いながら、問いが出るのを待つ

    問いの前に同意や上限を画面のスレッドで確かめるので、拾わないと問いまで進まない
    """
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not done.is_set():
        bridge.pump()
        question = bridge.take_question()
        if question is not None:
            return question
        time.sleep(0.005)
    raise AssertionError("問いが画面へ届かない")


def _all(host: FakeHost) -> list[Clip]:
    return [clip for track in host.document.project.timeline.tracks for clip in track.clips]


def _group(host: FakeHost, clip: Clip) -> GroupId | None:
    return next(c.group_id for c in _all(host) if c.id == clip.id)


class TestSplitClip:
    def test_the_partners_are_split_into_a_new_back_group(self) -> None:
        # 1 本ずつ割ると、割るたびに後ろが元のグループに残り、前と後ろの 4 本が 1 つの束の
        # ままだった（#427） 後ろを選んでも前まで一緒に選ばれて動く
        host, clips = _grouped((0, 100), (0, 100))
        result = _run(host, "split_clip", clip_id=str(clips[0].id), frame=40)
        assert len(result["new_clips"]) == 2
        backs = [c for c in _all(host) if str(c.id) in result["new_clips"]]
        assert len({c.group_id for c in backs}) == 1
        assert backs[0].group_id not in (None, clips[0].group_id)
        assert _group(host, clips[1]) == clips[0].group_id

    def test_a_lone_back_half_leaves_the_group(self) -> None:
        # 仲間が短く、またいだのが 1 本だけのときも、後ろを元のグループに残さない（#427 B）
        host, clips = _grouped((0, 100), (0, 30))
        result = _run(host, "split_clip", clip_id=str(clips[0].id), frame=40)
        (back,) = [c for c in _all(host) if str(c.id) in result["new_clips"]]
        assert back.group_id is None
        assert _group(host, clips[1]) == clips[0].group_id

    def test_one_undo_brings_the_group_back(self) -> None:
        # 仲間まで割った 1 手が 2 段に分かれると、元に戻す 1 回で半分だけ戻る
        host, clips = _grouped((0, 100), (0, 100))
        before = host.document.project
        _run(host, "split_clip", clip_id=str(clips[0].id), frame=40)
        host.document.undo()
        assert host.document.project == before

    def test_a_locked_partner_refuses_without_pointing_at_ungrouping(self) -> None:
        # 断りの文が「グループを解く」を勧めると、AI は勧められたとおりに解いて進む（#428）
        host, clips = _grouped((0, 100), (0, 100), locked=1)
        with pytest.raises(ToolError, match="ロック") as caught:
            _run(host, "split_clip", clip_id=str(clips[0].id), frame=40)
        assert "ungroup_clips" not in str(caught.value)
        assert "ユーザーに確かめ" in str(caught.value)

    def test_a_lone_clip_on_a_locked_layer_does_not_mention_partners(self) -> None:
        # 相手のいない 1 本で相手や束の話をすると、AI はありもしない組や束を探しに行く
        # （CodeRabbit の指摘 #435）
        base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
        lone = Clip(timeline_start=0, duration=100, source=TEXT.create())
        track = Track(TrackKind.MIXED, "レイヤー 1", (lone,), locked=True)
        host = FakeHost(base.with_timeline(replace(base.timeline, tracks=(track,))))
        with pytest.raises(ToolError, match="ロックされているので分割できません") as caught:
            _run(host, "split_clip", clip_id=str(lone.id), frame=40)
        assert "相手" not in str(caught.value)
        assert "グループ" not in str(caught.value)
        assert "ユーザーに確かめ" in str(caught.value)

    def test_a_locked_partner_through_a_link_also_points_at_asking(self) -> None:
        # リンクの相手の側のロックで断られたときに案内が無いと、AI は解いて進む道を探す
        # （CodeRabbit の指摘 #435）
        host, clips = _grouped((0, 100), (0, 100), locked=1)
        link = GroupId("L")
        lone = Clip(timeline_start=0, duration=100, source=TEXT.create(), link_group=link)
        project = host.document.project
        tracks = list(project.timeline.tracks)
        tracks[0] = replace(tracks[0], clips=(replace(clips[0], link_group=link),))
        tracks.append(Track(TrackKind.MIXED, "レイヤー 3", (lone,)))
        host = FakeHost(project.with_timeline(replace(project.timeline, tracks=tuple(tracks))))
        with pytest.raises(ToolError, match="ロック") as caught:
            _run(host, "split_clip", clip_id=str(lone.id), frame=40)
        assert "ungroup_clips" not in str(caught.value)
        assert "ユーザーに確かめ" in str(caught.value)


class TestCuttingKeepsGroups:
    def test_cutting_a_range_keeps_the_front_group(self) -> None:
        # 範囲を切って詰めても、前に残った物は人が付けたグループのまま（#428 で確かめた形）
        host, clips = _grouped((0, 100), (0, 100))
        _run(host, "cut_range", start_frame=40, end_frame=60)
        assert _group(host, clips[0]) == _group(host, clips[1]) == clips[0].group_id

    def test_a_silence_cut_keeps_the_front_group(self) -> None:
        # 無音カット（jet_cut の中身の RippleCut）も前を元のグループに残す
        host, clips = _grouped((0, 100), (0, 100))
        host.apply_commands([RippleCut(((20, 30), (50, 60)))], "無音をカット")
        assert _group(host, clips[0]) == _group(host, clips[1]) == clips[0].group_id

    def test_the_silence_cut_tool_keeps_the_front_group(self, host: FakeHost) -> None:
        # 道具の jet_cut を通しても、前に残った物は人が付けたグループのまま 後ろの欠片は
        # 分割と同じく新しい束になる（#428 で確かめるよう求められた道具の呼び出し）
        from tests.ui.test_subtitle_panel import make_waveform

        project = host.document.project
        video = project.timeline.tracks[0].clips[0]
        caption = Clip(timeline_start=0, duration=video.duration, source=TEXT.create())
        tracks = (*project.timeline.tracks, Track(TrackKind.VIDEO, "V2", (caption,)))
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        grouped = FakeHost(GroupClips((video.id, caption.id)).apply(project))
        grouped.stub_waveform = make_waveform([(0.5, 700), (0.0, 400), (0.5, 800)])
        group = _group(grouped, video)
        result = _run(grouped, "jet_cut", media_id=str(project.media[0].id), keep_speech=False)
        assert result["cuts"] == 1
        assert _group(grouped, video) == _group(grouped, caption) == group
        backs = [c for c in _all(grouped) if c.id not in (video.id, caption.id)]
        assert len(backs) == 2
        assert len({c.group_id for c in backs}) == 1 and backs[0].group_id not in (None, group)

    def test_cutting_clips_out_keeps_other_groups(self) -> None:
        # 切り取りはグループごと切る 残した別のグループまで解けると、束が黙って消える
        host, clips = _grouped((0, 100), (0, 100), (200, 50), (200, 50))
        project = GroupClips((clips[2].id, clips[3].id)).apply(host.document.project)
        host = FakeHost(project)
        _run(host, "cut_clips", clip_ids=[str(clips[2].id)])
        assert {c.id for c in _all(host)} == {clips[0].id, clips[1].id}
        assert _group(host, clips[0]) == _group(host, clips[1]) == clips[0].group_id

    def test_cutting_a_locked_group_does_not_suggest_ungrouping(self) -> None:
        # 断りの文の言い換え（「グループを外してから」など）は文字の見張りをすり抜ける 実際に
        # 断らせて、解除を勧めずに確かめ方を返すことを見る（CodeRabbit の指摘 #435）
        host, clips = _grouped((0, 100), (0, 100), locked=1)
        before = host.document.project
        with pytest.raises(ToolError, match="ロック") as caught:
            _run(host, "cut_clips", clip_ids=[str(clips[0].id)])
        assert "ungroup" not in str(caught.value)
        assert "グループを解" not in str(caught.value) and "グループを外" not in str(caught.value)
        assert "ユーザーに確かめ" in str(caught.value)
        assert host.document.project == before

    def test_moving_to_another_track_does_not_suggest_ungrouping(self) -> None:
        # 断りの文が ungroup_clips を勧めていたので、AI はトラックを移すために束を解いた（#428）
        host, clips = _grouped((0, 100), (0, 100))
        target = host.document.project.timeline.tracks[1].id
        with pytest.raises(ToolError, match="move_clips") as caught:
            _run(host, "move_clip", clip_id=str(clips[0].id), track_id=str(target))
        assert "ungroup_clips" not in str(caught.value)
        assert "ユーザーに確かめ" in str(caught.value)


@dataclass(frozen=True, slots=True)
class _Swap(Command):
    """作品を差し替える 答える間に人が束を組み直したことを、履歴に載せて作る"""

    project: Project

    @property
    def label(self) -> str:
        return "組み直し"

    def apply(self, project: Project) -> Project:
        del project
        return self.project


def test_no_message_advises_ungrouping() -> None:
    """道具の断りの文に、グループの解除を勧める文が残っていないか

    勧める文があると、AI はカットの途中で止まったときにそのとおり ungroup_clips を呼び、
    人が付けたグループを黙って解く（#428） 文言を足した人が勧め直すのを見張る
    """
    source = Path(operations_module.__file__).read_text(encoding="utf-8")
    advice = re.compile(
        r"ungroup_clips\s*で(?:解|外)|グループを解除してください|グループを解いて(?:ください|から)"
    )
    assert advice.findall(source) == []


class TestPrompt:
    @pytest.mark.parametrize("ask", [True, False])
    def test_the_prompt_keeps_groups_unless_asked(self, ask: bool) -> None:
        # 指示の文にグループの決まりが無いと、AI はカットで止まったときに解いて進む（#428）
        # 質問の設定を切った人にも効かせる（解かない決まりは聞き返しとは別）
        prompt = system_prompt(ask=ask)
        assert "頼まれない限り ungroup_clips で解きません" in prompt
        assert "解いたときは、どのグループを解いたか" in prompt


class TestAskingBeforeUngrouping:
    def _ungroup(self) -> Operation:
        return _operation("ungroup_clips")

    def test_auto_approve_still_asks(self) -> None:
        # 「変更を自動で承認」の人には許可の窓が出ない 尋ねなければ、AI が黙って束を解く
        host, clips = _grouped((0, 100), (0, 100))
        bridge = EditorBridge(host, auto_approve=True)
        done, box = call_in_thread(bridge, self._ungroup(), {"clip_ids": [str(clips[0].id)]})
        question = _take(bridge, done)
        assert question.spec.choices == UNGROUP_CHOICES
        question.reply(("解除しない",))
        pump_until(bridge, done)
        assert isinstance(box.get("error"), ToolError)
        assert "解除する" in str(box["error"])
        assert _group(host, clips[0]) == _group(host, clips[1]) == clips[0].group_id

    def test_agreeing_ungroups_without_a_second_approval(self) -> None:
        # 答えた後に許可の窓も出すと、同じことを 2 度聞かれる
        host, clips = _grouped((0, 100), (0, 100))
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, self._ungroup(), {"clip_ids": [str(clips[0].id)]})
        question = _take(bridge, done)
        question.reply(("解除する",))
        pump_until(bridge, done)
        assert bridge.take_approval() is None
        assert "error" not in box
        assert _group(host, clips[0]) is None and _group(host, clips[1]) is None

    def test_nothing_to_ungroup_is_refused_without_asking(self) -> None:
        # 解く物が無いのに尋ねると、人の答えが無駄になる（CodeRabbit の指摘 #435）
        host, lone = _with_lone_clip()
        bridge = EditorBridge(host, auto_approve=True)
        done, box = call_in_thread(bridge, self._ungroup(), {"clip_ids": [str(lone.id)]})
        pump_until(bridge, done)
        assert "解除する対象がありません" in str(box.get("error"))
        assert bridge._questions.empty()

    def test_a_clip_joining_the_group_while_asked_is_left_alone(self) -> None:
        # 答える間に別のクリップが束へ足されたら、人が見て許した本数と違う束を解くことになる
        # ID の集まりだけを比べると、足された 1 本に気付けない（CodeRabbit の指摘 #435）
        host, lone = _with_lone_clip()
        grouped = next(
            c for t in host.document.project.timeline.tracks for c in t.clips if c.group_id
        )
        group = grouped.group_id
        bridge = EditorBridge(host, auto_approve=True)
        done, box = call_in_thread(bridge, self._ungroup(), {"clip_ids": [str(grouped.id)]})
        question = _take(bridge, done)
        project = host.document.project
        tracks = tuple(
            replace(
                track,
                clips=tuple(
                    replace(clip, group_id=group) if clip.id == lone.id else clip
                    for clip in track.clips
                ),
            )
            for track in project.timeline.tracks
        )
        host.document.execute(
            _Swap(project.with_timeline(replace(project.timeline, tracks=tracks)))
        )
        question.reply(("解除する",))
        pump_until(bridge, done)
        assert "変わった" in str(box.get("error"))
        assert _group(host, grouped) == _group(host, lone) == group

    def test_the_question_says_how_much_will_be_ungrouped(self) -> None:
        # 何を解くかが出ないと、思っていたより大きな束を解こうとしていても気付けない
        # （CodeRabbit の指摘 #435）
        host, clips = _grouped((0, 100), (0, 100), (0, 100))
        bridge = EditorBridge(host, auto_approve=True)
        done, _box = call_in_thread(bridge, self._ungroup(), {"clip_ids": [str(clips[0].id)]})
        question = _take(bridge, done)
        assert "グループ 1 個" in question.spec.text
        assert "クリップ 3 本" in question.spec.text
        question.reply(("解除しない",))
        pump_until(bridge, done)

    def test_choosing_again_while_asked_does_not_change_the_target(self) -> None:
        # clip_ids を省くと選択中のクリップを解く 答える間に人が選び直したクリップを解くと、
        # 許したのと違う束が解ける（CodeRabbit の指摘 #435）
        host, clips = _grouped((0, 100), (0, 100), (200, 50), (200, 50))
        project = GroupClips((clips[2].id, clips[3].id)).apply(host.document.project)
        host = FakeHost(project)
        other = _group(host, clips[2])
        host.select_clips([clips[0].id])
        bridge = EditorBridge(host, auto_approve=True)
        done, box = call_in_thread(bridge, self._ungroup(), {})
        question = _take(bridge, done)
        host.select_clips([clips[2].id])
        question.reply(("解除する",))
        pump_until(bridge, done)
        assert "error" not in box
        assert _group(host, clips[0]) is None
        assert _group(host, clips[2]) == _group(host, clips[3]) == other

    def test_a_group_changed_while_asked_is_left_alone(self) -> None:
        # 答える間に束が組み直されたのに解くと、人が見て許したのと違う束が消える
        host, clips = _grouped((0, 100), (0, 100), (200, 50))
        bridge = EditorBridge(host, auto_approve=True)
        done, box = call_in_thread(bridge, self._ungroup(), {"clip_ids": [str(clips[0].id)]})
        question = _take(bridge, done)
        regrouped = GroupClips((clips[0].id, clips[2].id)).apply(host.document.project)
        host.document.execute(_Swap(regrouped))
        question.reply(("解除する",))
        pump_until(bridge, done)
        assert "変わった" in str(box.get("error"))
        assert _group(host, clips[0]) is not None
        assert _group(host, clips[0]) == _group(host, clips[2])

    def test_the_written_reason_reaches_the_assistant(self) -> None:
        # 書いた答えが届かないと、AI は断られた訳が分からず、同じ頼み方でまた尋ねる
        host, clips = _grouped((0, 100), (0, 100))
        bridge = EditorBridge(host, auto_approve=True)
        done, box = call_in_thread(bridge, self._ungroup(), {"clip_ids": [str(clips[0].id)]})
        _take(bridge, done).reply((), "字幕と一緒に動かしたい")
        pump_until(bridge, done)
        assert "字幕と一緒に動かしたい" in str(box["error"])

    def test_turning_it_off_goes_back_to_the_usual_approval(self) -> None:
        # 切ったのに尋ね続けると、任せきりにしたい人の作業がそこで止まる
        host, clips = _grouped((0, 100), (0, 100))
        bridge = EditorBridge(host, auto_approve=True)
        bridge.ask_first = False
        done, box = call_in_thread(bridge, self._ungroup(), {"clip_ids": [str(clips[0].id)]})
        pump_until(bridge, done)
        assert bridge.take_question() is None
        assert "error" not in box
        assert _group(host, clips[0]) is None

    def test_only_editing_tools_can_ask_first(self) -> None:
        # 取り消せない操作に付けると、取り消せない操作の確かめと問いの 2 つが出る
        with pytest.raises(ValueError, match="asks_first"):
            replace(self._ungroup(), writes=False)
