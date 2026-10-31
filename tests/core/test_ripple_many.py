"""何百か所を一度に切る命令が、1 か所ずつ切るのと同じ結果になること（#413）

本物の素材の無音カットは 200 か所を超える 1 か所ごとに全部のクリップを動かすと編集画面が
止まるので、まとめて 1 回の走査にしてある 走査の形を変えても切り方が変わらないことを見る
1 か所ずつの命令を基準にするので、その 1 か所の切り方は tests/core/test_commands.py が押さえる
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.commands import RemoveClips, RippleCut
from sashimono.core.jetcut import cut_targets, plan_cuts
from sashimono.core.model import (
    Clip,
    GroupId,
    Marker,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    new_group_id,
)
from sashimono.effects.sources import TEXT
from tests.conftest import RATE_30, make_clip


def _laid_out(project: Project) -> list[tuple[str, int, int, Fraction]]:
    """ID によらない並び 切ると ID が振り直されるので、位置と中身の位置で比べる"""
    return [
        (track.name, clip.timeline_start, clip.duration, clip.source_in)
        for track in project.timeline.tracks
        for clip in track.clips
    ]


def _links(project: Project) -> set[frozenset[tuple[str, int]]]:
    """リンクの組 グループの ID は振り直されるので、組の顔ぶれで比べる"""
    groups: dict[GroupId, set[tuple[str, int]]] = {}
    for track in project.timeline.tracks:
        for clip in track.clips:
            if clip.link_group is not None:
                groups.setdefault(clip.link_group, set()).add((track.name, clip.timeline_start))
    return {frozenset(members) for members in groups.values()}


def _one_by_one(project: Project, ranges: Iterable[tuple[int, int]], *, keep: bool) -> Project:
    """1 か所ずつ後ろから切る 後ろから切れば前の範囲の位置はずれない"""
    for span in sorted(ranges, reverse=True):
        project = RippleCut((span,), keep_area=keep).apply(project)
    return project


def _work(media: MediaItem, seed: int) -> tuple[Project, list[tuple[int, int]]]:
    """映像と音声のリンクした組・字幕・ロックしたトラック・目印を持つ作品と、切る範囲"""
    rng = random.Random(seed)
    pictures, sounds, words = [], [], []
    frame = 0
    for _ in range(40):
        length = rng.randint(5, 30)
        group = new_group_id()
        picture = replace(make_clip(frame, length, media), link_group=group)
        pictures.append(picture)
        sounds.append(replace(picture, id=make_clip(0, 1, media).id))
        if rng.random() < 0.7:
            words.append(Clip(timeline_start=frame, duration=length, source=TEXT.create(text="札")))
        frame += length + rng.randint(0, 5)
    tracks = (
        Track(kind=TrackKind.VIDEO, name="V1", clips=tuple(pictures)),
        Track(kind=TrackKind.AUDIO, name="A1", clips=tuple(sounds)),
        Track(kind=TrackKind.VIDEO, name="字幕", clips=tuple(words)),
        Track(kind=TrackKind.VIDEO, name="鍵", clips=tuple(words[:3]), locked=True),
    )
    marks = tuple(Marker(frame=rng.randrange(frame)) for _ in range(10))
    project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(media,))
    project = project.with_timeline(
        replace(project.timeline, tracks=tracks, markers=marks, work_area=(10, frame - 10))
    )
    ranges = []
    cursor = 0
    while cursor < frame:
        cursor += rng.randint(1, 40)
        length = rng.randint(1, 35)
        ranges.append((cursor, cursor + length))
        cursor += length + 1
    return project, ranges


class TestManyRangesAtOnce:
    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("keep", [False, True])
    def test_the_same_as_one_at_a_time(self, video_media: MediaItem, seed: int, keep: bool) -> None:
        """壊れると、まとめて切ったときだけ位置・リンク・目印・書き出す範囲が変わる"""
        project, ranges = _work(replace(video_media, duration=Fraction(100)), seed)
        together = RippleCut(tuple(ranges), keep_area=keep).apply(project)
        apart = _one_by_one(project, ranges, keep=keep)
        assert _laid_out(together) == _laid_out(apart)
        assert _links(together) == _links(apart)
        assert together.timeline.markers == apart.timeline.markers
        assert together.timeline.work_area == apart.timeline.work_area

    def test_overlapping_and_empty_ranges(self, video_media: MediaItem) -> None:
        """壊れると、重なった範囲を 2 回切るか、長さ 0 の範囲で落ちる"""
        project, _ranges = _work(replace(video_media, duration=Fraction(100)), 1)
        messy = RippleCut(((10, 20), (15, 30), (40, 40), (50, 45)), keep_area=True)
        tidy = RippleCut(((10, 30),), keep_area=True)
        assert _laid_out(messy.apply(project)) == _laid_out(tidy.apply(project))

    def test_nothing_to_cut_changes_nothing(self, video_media: MediaItem) -> None:
        """壊れると、切る所が無いのに書き出す範囲が外れる"""
        project, _ranges = _work(replace(video_media, duration=Fraction(100)), 1)
        assert RippleCut(()).apply(project) == project

    def test_a_clip_across_several_ranges(self, video_media: MediaItem) -> None:
        """壊れると、1 本のクリップに掛かる範囲が 2 つ以上のとき、片方の範囲を切り忘れる"""
        project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(video_media,))
        track = Track(kind=TrackKind.VIDEO, name="V1", clips=(make_clip(0, 100, video_media),))
        project = project.with_timeline(replace(project.timeline, tracks=(track,)))
        ranges = ((10, 20), (30, 40), (50, 60))
        assert _laid_out(RippleCut(ranges).apply(project)) == _laid_out(
            _one_by_one(project, ranges, keep=False)
        )
        assert [row[1:3] for row in _laid_out(RippleCut(ranges).apply(project))] == [
            (0, 10),
            (10, 10),
            (20, 10),
            (30, 40),
        ]


class TestManySilences:
    @pytest.mark.parametrize("seed", range(6))
    def test_plan_cuts_is_the_same_as_one_at_a_time(
        self, video_media: MediaItem, seed: int
    ) -> None:
        """壊れると、無音を絞って見るせいで掛かっている無音を見落とす"""
        media = replace(video_media, duration=Fraction(100))
        project, _ranges = _work(media, seed)
        rng = random.Random(seed)
        spots = sorted(rng.randrange(0, 2000) for _ in range(60))
        silences = [(Fraction(s, 100), Fraction(s + rng.randint(1, 300), 100)) for s in spots]
        rng.shuffle(silences)
        together = plan_cuts(project, media.id, silences)
        merged: list[tuple[int, int]] = []
        for silence in silences:
            merged.extend(plan_cuts(project, media.id, [silence]))
        apart = RippleCut(tuple(merged))
        assert _laid_out(RippleCut(together).apply(project)) == _laid_out(apart.apply(project))
        assert set(cut_targets(project, media.id, silences)) <= {
            clip.id for track in project.timeline.tracks for clip in track.clips
        }


class TestManyGapsAtOnce:
    @staticmethod
    def _gapped(media: MediaItem, audio: MediaItem) -> tuple[Project, list[Clip]]:
        """映像 30 本 字幕 30 本 全体をまたがない効果音を持つ作品"""
        pictures = [make_clip(start * 20, 20, media) for start in range(30)]
        words = [
            Clip(timeline_start=start * 20 + 2, duration=8, source=TEXT.create(text="札"))
            for start in range(30)
        ]
        effects = [make_clip(start * 20 + 12, 4, audio) for start in range(0, 30, 3)]
        tracks = (
            Track(kind=TrackKind.VIDEO, name="V1", clips=tuple(pictures)),
            Track(kind=TrackKind.VIDEO, name="字幕", clips=tuple(words)),
            Track(kind=TrackKind.AUDIO, name="効果音", clips=tuple(effects)),
        )
        project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(media, audio))
        return project.with_timeline(replace(project.timeline, tracks=tracks)), pictures

    def test_the_same_as_one_clip_at_a_time(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、まとめて消したときだけ字幕や効果音の詰まり方が変わる"""
        media = replace(video_media, duration=Fraction(100))
        project, pictures = self._gapped(media, audio_media)
        chosen = pictures[1::2]
        together = RemoveClips(tuple(c.id for c in chosen), True, True).apply(project)
        apart = project
        for clip in reversed(chosen):
            apart = RemoveClips((clip.id,), True, True).apply(apart)
        assert _laid_out(together) == _laid_out(apart)
