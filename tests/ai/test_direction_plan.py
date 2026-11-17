"""演出の計画を計画の欄に出し、外した物以外を一括で付ける（#470 段 4 #479）

propose_direction・get_direction_plan と、計画（:mod:`sashimono.ai.direction_plan`）の
外す・付ける・やめる・作品が変わったら閉じる 欄は tests/ui/test_direction_box.py
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.direction_plan import (
    APPLY_LABEL,
    SOUND_TRACK_NAME,
    ProposalState,
    proposal_for,
)
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddMedia, RemoveClip
from sashimono.core.direction import SHAKE, SOUND, ZOOM, measure_direction
from sashimono.core.direction_recipes import BOLD, DirectionTaste
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects.emphasis import EmphasisRequest, emphasis_commands
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run
from tests.conftest import RATE_30

#: 2 分の作品 派手なら寄るが 6 回まで入る長さ
_SECONDS = 120


@pytest.fixture
def work(video_media: MediaItem) -> FakeHost:
    media = replace(video_media, duration=Fraction(_SECONDS))
    project = Project.create(ProjectSettings(frame_rate=RATE_30))
    project = AddMedia(media).apply(project)
    clip = Clip(timeline_start=0, duration=_SECONDS * 30, media_id=media.id)
    track = Track(kind=TrackKind.VIDEO, name="V1", clips=(clip,))
    host = FakeHost(project.with_timeline(replace(project.timeline, tracks=(track,))))
    host.taste = DirectionTaste(strength=BOLD)
    return host


def _clip(host: FakeHost) -> Clip:
    return host.document.project.timeline.tracks[0].clips[0]


def _cues(host: FakeHost) -> list[dict[str, Any]]:
    """笑いが 10 秒おき・大声が 25 秒おき 強さを散らす"""
    clip_id = str(_clip(host).id)
    laughs = [
        {
            "signal": "laughter",
            "start_frame": t * 30,
            "end_frame": t * 30 + 30,
            "score": (t * 37 % 100) / 100,
            "clip_id": clip_id,
        }
        for t in range(5, _SECONDS, 10)
    ]
    shouts = [
        {"signal": "loud", "start_frame": t * 30, "end_frame": t * 30 + 15, "score": 0.8}
        for t in range(12, _SECONDS, 25)
    ]
    return laughs + shouts


def _sound(duration: float) -> MediaItem:
    return MediaItem(
        path=Path("C:/素材/効果音.wav"),
        duration=Fraction(duration),
        audio_streams=(
            AudioStreamInfo(
                index=0,
                sample_rate=48000,
                channels=2,
                time_base=Fraction(1, 48000),
                codec="pcm_s16le",
            ),
        ),
    )


class TestPropose:
    def test_is_read_only_and_says_what_it_sends(self) -> None:
        # 計画を出すだけで作品は変えない 付けるのは人の〔付ける〕 本人の設定を使うので
        # アプリの設定を送ると書く
        for name in ("propose_direction", "get_direction_plan"):
            operation = find_operation(name)
            assert operation is not None
            assert operation.safety is Safety.READ_ONLY
        propose = find_operation("propose_direction")
        assert propose is not None
        assert propose.sends == {Outbound.PROJECT, Outbound.APP_SETTINGS}

    def test_proposes_without_changing_the_work(self, work: FakeHost) -> None:
        # 出しただけで付くと、人が外す前に演出が入り、外したい 1 つのために全部を取り消す
        before = work.document.project
        result = run(work, "propose_direction", cues=_cues(work))
        assert work.document.project is before
        assert result["state"] == "proposed"
        kinds = [row["kind"] for row in result["items"]]
        assert ZOOM in kinds and SHAKE in kinds
        assert [row["number"] for row in result["items"]] == list(
            range(1, len(result["items"]) + 1)
        )
        # 拾ったクリップへ付ける
        assert {row["clip_id"] for row in result["items"]} == {str(_clip(work).id)}
        assert "sound_media_id" in result["note"]
        view = proposal_for(work.document).view()
        assert view is not None and view.state is ProposalState.PROPOSED

    def test_the_same_cues_give_the_same_plan(self, work: FakeHost) -> None:
        # 同じ素材・同じ頼みで違う計画になると、人が確かめ直すことになる
        first = run(work, "propose_direction", cues=_cues(work))["items"]
        again = run(work, "propose_direction", cues=list(reversed(_cues(work))))["items"]
        assert first == again

    def test_does_not_overwrite_existing_moves(self, work: FakeHost) -> None:
        # 人が手で整えた寄りや前に付けた寄りを上書きすると、人が作った物を AI が黙って壊す
        clip = _clip(work)
        request = EmphasisRequest(kind=ZOOM, start=0, end=_SECONDS * 30 - 1, strength=110)
        commands, _ = emphasis_commands(clip, request, RATE_30)
        for command in commands:
            work.document.execute(command)
        result = run(work, "propose_direction", cues=_cues(work))
        assert ZOOM not in [row["kind"] for row in result["items"]]
        assert any("上書きしません" in row["reason"] for row in result["dropped"])

    def test_refuses_broken_cues_and_wrong_sounds(self, work: FakeHost) -> None:
        # 黙って捨てると、渡した手がかりと違う計画を正しいと思って使う
        with pytest.raises(ToolError, match="cues"):
            run(work, "propose_direction", cues=[])
        with pytest.raises(ToolError, match="signal"):
            run(work, "propose_direction", cues=[{"signal": "x", "start_frame": 0, "end_frame": 9}])
        with pytest.raises(ToolError, match="start_frame"):
            run(
                work,
                "propose_direction",
                cues=[{"signal": "loud", "start_frame": 9, "end_frame": 3}],
            )
        long = _sound(60)
        work.document.execute(AddMedia(long))
        with pytest.raises(ToolError, match="秒以下"):
            run(work, "propose_direction", cues=_cues(work), sound_media_id=str(long.id))


class TestApply:
    def test_applies_what_is_left_as_one_undo(self, work: FakeHost) -> None:
        # 外した物が付くと、一覧で外した意味が無い 何段にも分かれると全部戻すのに何回も押す
        before = work.document.project
        items = run(work, "propose_direction", cues=_cues(work))["items"]
        proposal = proposal_for(work.document)
        dropped = items[0]["number"]
        proposal.exclude(dropped, True)
        result = proposal.apply(work.document)
        assert result is not None
        assert dropped not in result.applied
        assert len(result.applied) == len(items) - 1
        report = measure_direction(work.document.project, work.document.project.timeline)
        counts = {k.kind.key: k.count for k in report.kinds}
        chosen = [row for row in items if row["number"] != dropped]
        for kind in (ZOOM, SHAKE):
            assert counts[kind] == sum(1 for row in chosen if row["kind"] == kind)
            assert next(k for k in report.kinds if k.kind.key == kind).too_close == ()
        assert work.document.undo_label == APPLY_LABEL
        work.document.undo()
        assert work.document.project == before
        state = run(work, "get_direction_plan")
        assert state["state"] == "applied"
        assert state["applied"] == list(result.applied)

    def test_puts_sound_effects_on_their_own_track(self, work: FakeHost) -> None:
        # 本人の音声トラックへ混ぜると BGM やナレーションと重なる 効果音のトラックへ置く
        sound = _sound(1)
        work.document.execute(AddMedia(sound))
        items = run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))[
            "items"
        ]
        sounds = [row for row in items if row["kind"] == SOUND]
        assert sounds and all(row["media_id"] == str(sound.id) for row in sounds)
        result = proposal_for(work.document).apply(work.document)
        assert result is not None and result.skipped == ()
        project = work.document.project
        tracks = [t for t in project.timeline.tracks if t.name == SOUND_TRACK_NAME]
        assert len(tracks) == 1
        assert len(tracks[0].clips) == len(sounds)
        report = measure_direction(project, project.timeline)
        assert next(k for k in report.kinds if k.kind.key == SOUND).count == len(sounds)

    def test_items_whose_clip_went_away_are_skipped(self, work: FakeHost) -> None:
        # 1 つ付けられないだけで全部をやめると、外さなかった残りまで付かない
        sound = _sound(1)
        work.document.execute(AddMedia(sound))
        items = run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))[
            "items"
        ]
        work.document.execute(RemoveClip(_clip(work).id))
        result = proposal_for(work.document).apply(work.document)
        assert result is not None
        moves = [row["number"] for row in items if row["kind"] in (ZOOM, SHAKE)]
        assert sorted(number for number, _ in result.skipped) == moves
        assert all("見つかりません" in reason for _, reason in result.skipped)
        assert len(result.applied) == len(items) - len(moves)

    def test_reopening_the_work_closes_the_plan(self, work: FakeHost) -> None:
        # 前の作品の計画を次の作品で付けると、別の作品のクリップへ演出を付けかねない
        run(work, "propose_direction", cues=_cues(work))
        work.document.reset(work.document.project)
        assert proposal_for(work.document).view() is None

    def test_cancel_and_close(self, work: FakeHost) -> None:
        # やめた後に〔付ける〕が効くと、やめたはずの演出が入る
        before = work.document.project
        run(work, "propose_direction", cues=_cues(work))
        proposal = proposal_for(work.document)
        proposal.discard()
        assert proposal.view() is not None
        proposal.cancel()
        assert proposal.apply(work.document) is None
        assert work.document.project is before
        assert run(work, "get_direction_plan")["state"] == "cancelled"
        proposal.discard()
        assert run(work, "get_direction_plan")["state"] == "none"
