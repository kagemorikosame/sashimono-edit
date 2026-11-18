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

from sashimono.ai import direction_plan, operations
from sashimono.ai.direction_plan import (
    APPLY_LABEL,
    SOUND_TRACK_NAME,
    ProposalState,
    build_items,
    proposal_for,
)
from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import (
    AddClip,
    AddMedia,
    AddTrack,
    Command,
    RemoveClip,
    history,
)
from sashimono.core.direction import SHAKE, SOUND, ZOOM, measure_direction
from sashimono.core.direction_recipes import ACTION_ZOOM, BOLD, DirectionTaste, PlannedDirection
from sashimono.core.io.serialize import json_text
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    SceneId,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate, frame_to_seconds
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


class _Refused(Command):
    """当てると断る命令"""

    @property
    def label(self) -> str:
        return "断る"

    def apply(self, project: Project) -> Project:
        raise ValueError("この命令は通りません")


def _without_ids(project: Project) -> list[tuple[str, object]]:
    clip = project.timeline.tracks[0].clips[0]
    return [(effect.kind, effect.params) for effect in clip.effects]


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
            # 計画を読む道具も本人の設定から決まった強さを返す
            assert operation.sends == {Outbound.PROJECT, Outbound.APP_SETTINGS}

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
        # 段 1 はナレーションの録音を効果音に数えない 置けても計画と数が食い違う
        voice = replace(_sound(1), path=Path("C:/作品/narration/001.wav"))
        work.document.execute(AddMedia(voice))
        with pytest.raises(ToolError, match="ナレーション"):
            run(work, "propose_direction", cues=_cues(work), sound_media_id=str(voice.id))

    @pytest.mark.parametrize("rate", [RATE_30, FrameRate(30000, 1001)])
    def test_items_keep_the_frame_of_the_cue(self, video_media: MediaItem, rate: FrameRate) -> None:
        # 秒から戻すときに切り捨てると、29.97 fps で手がかりより 1 フレーム早くなる
        project = Project.create(ProjectSettings(frame_rate=rate))
        project = AddMedia(video_media).apply(project)
        clip = Clip(timeline_start=0, duration=600, media_id=video_media.id)
        track = Track(kind=TrackKind.VIDEO, name="V1", clips=(clip,))
        project = project.with_timeline(replace(project.timeline, tracks=(track,)))
        planned = [
            PlannedDirection(
                recipe="r",
                action=ACTION_ZOOM,
                kind=ZOOM,
                start=float(frame_to_seconds(frame, rate)),
                end=float(frame_to_seconds(frame + 30, rate)),
                amount=None,
                reason="",
            )
            for frame in (4, 301)
        ]
        items, dropped = build_items(project, planned, [None, None], None, 600)
        assert dropped == []
        assert [(i.start_frame, i.end_frame) for i in items] == [(4, 34), (301, 331)]

    def test_long_plans_come_in_pages(
        self, work: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 項目が多いと、一覧だけで返事の上限を越えて打ち切られ、後半を読めない
        # 上限を小さくして、件数を指定しなくても上限で切れ、続きを最後まで読める所を作る
        monkeypatch.setattr(operations, "MAX_RESULT_CHARS", 1500)
        sound = _sound(1)
        work.document.execute(AddMedia(sound))
        page = run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))
        view = proposal_for(work.document).view()
        assert view is not None
        total = len(view.items)
        assert page["total"] == total and "next_offset" in page
        numbers: list[int] = []
        while True:
            assert len(json_text(page)) <= 1500
            assert page["state"] == "proposed" and "dropped" in page
            numbers += [row["number"] for row in page["items"]]
            if "next_offset" not in page:
                break
            page = run(work, "get_direction_plan", offset=page["next_offset"])
        assert numbers == list(range(1, total + 1))

    def test_refuses_while_a_scene_is_open(self, work: FakeHost) -> None:
        # シーンの時刻とクリップで計画しても、〔付ける〕はメインへ当たり、別の素材へ寄る
        work.scene = SceneId("開いたシーン")
        with pytest.raises(ToolError, match="シーン"):
            run(work, "propose_direction", cues=_cues(work))
        assert proposal_for(work.document).view() is None


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
        # 付けた後は AI が段 1 の道具と絵と書き出しの検査で確かめ、方針の外ややりすぎを直す
        for tool in ("measure_direction", "preview_range", "check_output"):
            assert tool in state["note"]

    def test_moves_are_the_same_as_emphasize(self, work: FakeHost) -> None:
        # 計画で付けた寄りは emphasize と同じ命令で付く 別の作り方をすると、書き出しで
        # 今までの道を通らない動きが混ざる
        before = work.document.project
        items = run(work, "propose_direction", cues=_cues(work))["items"]
        proposal = proposal_for(work.document)
        zoom = next(row for row in items if row["kind"] == ZOOM)
        for row in items:
            proposal.exclude(row["number"], row["number"] != zoom["number"])
        proposal.apply(work.document)
        clip = before.timeline.tracks[0].clips[0]
        request = EmphasisRequest(
            kind=ZOOM,
            start=zoom["start_frame"],
            end=zoom["end_frame"],
            strength=float(zoom["amount"]),
        )
        commands, _ = emphasis_commands(clip, request, RATE_30)
        expected = before
        for command in commands:
            expected = command.apply(expected)
        # 効果の ID は付けるたびに新しく振られるので、比べるのは中身
        assert _without_ids(work.document.project) == _without_ids(expected)

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

    def test_sounds_are_not_planned_twice(self, work: FakeHost) -> None:
        # 付けた後に同じ頼みで計画し直すと、同じ時刻の効果音が二重に鳴って大きくなる
        sound = _sound(1)
        work.document.execute(AddMedia(sound))
        run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))
        first = proposal_for(work.document).apply(work.document)
        assert first is not None and first.applied
        again = run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))
        assert SOUND not in [row["kind"] for row in again["items"]]
        assert any("もう効果音" in row["reason"] for row in again["dropped"])

    def test_a_sound_placed_after_the_plan_is_not_doubled(self, work: FakeHost) -> None:
        # 計画の後で人が同じ区間へ効果音を置くと、付ける時に見直さないと両方が鳴る
        # 重なる効果音だけを理由付きで飛ばし、残りの項目は付ける
        sound = _sound(1)
        work.document.execute(AddMedia(sound))
        items = run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))[
            "items"
        ]
        sounds = [row for row in items if row["kind"] == SOUND]
        assert sounds and len(items) > len(sounds)
        placed = sounds[0]
        track = Track(kind=TrackKind.AUDIO, name="人の効果音")
        work.document.execute(AddTrack(track))
        work.document.execute(
            AddClip(
                track.id,
                Clip(
                    timeline_start=placed["start_frame"],
                    duration=placed["end_frame"] - placed["start_frame"],
                    media_id=sound.id,
                ),
            )
        )
        result = proposal_for(work.document).apply(work.document)
        assert result is not None
        reasons = dict(result.skipped)
        assert "もう効果音" in reasons[placed["number"]]
        assert set(result.applied) == {row["number"] for row in items} - set(reasons)
        assert len(result.applied) >= len(items) - len(sounds)
        heard = [
            e
            for e in measure_direction(work.document.project, work.document.project.timeline).events
            if e.kind == SOUND and e.start_frame == placed["start_frame"]
        ]
        assert len(heard) == 1

    def test_sounds_stay_inside_the_range(self, work: FakeHost) -> None:
        # 範囲の終わりを越えて置くと作品が延び、書き出しに絵の無い末尾が付く
        sound = _sound(9)
        work.document.execute(AddMedia(sound))
        end = _SECONDS * 30
        last = {"signal": "laughter", "start_frame": end - 60, "end_frame": end - 30, "score": 1.0}
        result = run(
            work,
            "propose_direction",
            cues=[last],
            sound_media_id=str(sound.id),
            strength="bold",
        )
        assert all(row["end_frame"] <= end for row in result["items"])
        assert any("範囲の終わり" in row["reason"] for row in result["dropped"])
        proposal_for(work.document).apply(work.document)
        assert work.document.project.timeline.duration == end

    def test_sounds_do_not_go_to_a_silent_sound_track(self, work: FakeHost) -> None:
        # 消音・鍵のトラックへ置くと、付いたことになっても鳴らない（置けない）
        # 人のトラックがソロなら、足すトラックもソロにしないと鳴らない
        sound = _sound(1)
        work.document.execute(AddMedia(sound))
        project = work.document.project
        tracks = (
            replace(project.timeline.tracks[0], solo=True),
            Track(kind=TrackKind.AUDIO, name=SOUND_TRACK_NAME, muted=True, solo=True),
            Track(kind=TrackKind.AUDIO, name=SOUND_TRACK_NAME, locked=True, solo=True),
        )
        work.document.reset(project.with_timeline(replace(project.timeline, tracks=tracks)))
        run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))
        result = proposal_for(work.document).apply(work.document)
        assert result is not None and result.skipped == ()
        after = work.document.project.timeline.tracks
        assert [len(t.clips) for t in after[1:3]] == [0, 0]
        placed = [t for t in after[3:] if t.name == SOUND_TRACK_NAME]
        assert len(placed) == 1 and placed[0].clips
        assert placed[0].solo is True and placed[0].muted is False

    def test_a_picture_solo_does_not_silence_the_other_sounds(self, work: FakeHost) -> None:
        # 映像トラックのソロは音に効かない それを見て足すトラックをソロにすると、
        # 本人の BGM やナレーションが鳴らなくなる
        sound = _sound(1)
        work.document.execute(AddMedia(sound))
        project = work.document.project
        tracks = (
            replace(project.timeline.tracks[0], solo=True),
            Track(kind=TrackKind.AUDIO, name="BGM"),
        )
        work.document.reset(project.with_timeline(replace(project.timeline, tracks=tracks)))
        run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))
        proposal_for(work.document).apply(work.document)
        timeline = work.document.project.timeline
        placed = [t for t in timeline.tracks if t.name == SOUND_TRACK_NAME]
        assert len(placed) == 1 and placed[0].clips and placed[0].solo is False
        assert "BGM" in {t.name for t in timeline.active_sound_tracks()}

    def test_an_item_the_document_refuses_is_skipped_alone(
        self, work: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 持ち出し用パッケージの範囲の外のように文書の側で断る物は、命令の試しでは通る
        # 実行で断られると段ごと戻り、先に付けた項目まで消える
        sound = _sound(1)
        work.document.execute(AddMedia(sound))
        items = run(work, "propose_direction", cues=_cues(work), sound_media_id=str(sound.id))[
            "items"
        ]

        def refuse_new_tracks(before: Project, after: Project, _sounds: object) -> str | None:
            grew = len(after.timeline.tracks) > len(before.timeline.tracks)
            return "パッケージの範囲の外です" if grew else None

        monkeypatch.setattr(history, "packed_refusal", refuse_new_tracks)
        result = proposal_for(work.document).apply(work.document)
        assert result is not None
        sounds = [row["number"] for row in items if row["kind"] == SOUND]
        assert sounds
        assert [number for number, _ in result.skipped] == sounds
        assert len(result.applied) == len(items) - len(sounds) > 0
        assert work.document.undo_label == APPLY_LABEL

    def test_an_item_that_cannot_run_is_skipped_alone(
        self, work: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 途中で通らない命令を実行すると、その例外で段ごと戻り、先に付けた項目まで消える
        # 組んだ命令の 2 つ目で断られる項目を作る（鍵の掛かったトラックへ置く などの形）
        items = run(work, "propose_direction", cues=_cues(work))["items"]
        broken = items[1]["number"]
        original = direction_plan._commands

        def commands(project: Project, item: Any) -> list[Command]:
            built = original(project, item)
            return [*built, _Refused()] if item.number == broken else built

        monkeypatch.setattr(direction_plan, "_commands", commands)
        result = proposal_for(work.document).apply(work.document)
        assert result is not None
        assert [number for number, _ in result.skipped] == [broken]
        assert len(result.applied) == len(items) - 1
        assert work.document.undo_label == APPLY_LABEL
        # 断られた項目は途中まで付いていない 付いた数は付けた項目の分だけ
        report = measure_direction(work.document.project, work.document.project.timeline)
        counts = {k.kind.key: k.count for k in report.kinds}
        assert counts[ZOOM] + counts[SHAKE] == len(items) - 1

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
