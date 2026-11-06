"""グループに入ったクリップを割ったときのグループの分かれ方（#427）

割ってできた後ろの片割れは、元のグループに残さない 同じ所で割った同じグループの後ろが
2 本以上なら、後ろどうしを 1 つの新しいグループにまとめ、1 本だけならグループから外す
割らなかった仲間（その位置をまたがない短いクリップ）は元のグループのまま前に残る

分割・挿入貼り付け・範囲を切って詰める（無音カット）のどれも同じ決まりで分ける 入口ごとに
決まりが違うと、同じ形のグループが割り方によって前と後ろで束ねられたままになる
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.core.commands import (
    Document,
    GroupClips,
    InsertGap,
    RippleCut,
    SplitClips,
    insert_media,
)
from sashimono.core.model import (
    Clip,
    ClipId,
    GroupId,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects.sources import TEXT


def _grouped(*spans: tuple[int, int], locked: int | None = None) -> tuple[Project, list[Clip]]:
    """``spans`` の長さのテキストを 1 本ずつ別のレイヤーに置いて、全部を 1 つのグループにする

    ``locked`` に番号を渡すと、そのレイヤーをロックする
    """
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    clips = [Clip(timeline_start=s, duration=d, source=TEXT.create()) for s, d in spans]
    tracks = tuple(
        Track(TrackKind.MIXED, f"レイヤー {index + 1}", (clip,), locked=index == locked)
        for index, clip in enumerate(clips)
    )
    project = GroupClips(tuple(clip.id for clip in clips)).apply(
        base.with_timeline(replace(base.timeline, tracks=tracks))
    )
    return project, [_clip(project, clip.id) for clip in clips]


def _clip(project: Project, clip_id: ClipId) -> Clip:
    located = project.timeline.locate_clip(clip_id)
    assert located is not None
    return located[1]


def _all(project: Project) -> list[Clip]:
    return [clip for track in project.timeline.tracks for clip in track.clips]


def _groups_from(project: Project, frame: int) -> set[GroupId | None]:
    return {clip.group_id for clip in _all(project) if clip.timeline_start == frame}


class TestSplitting:
    def test_a_both_straddling_split_into_a_front_and_a_back_group(self) -> None:
        # A: 2 本とも割る位置をまたぐ 前どうし・後ろどうしに分かれないと、後ろだけを選べない
        project, clips = _grouped((0, 100), (0, 100))
        group = clips[0].group_id
        split = SplitClips(tuple(clip.id for clip in clips), 40).apply(project)
        assert _groups_from(split, 0) == {group}
        back = _groups_from(split, 40)
        assert len(back) == 1 and None not in back and group not in back

    def test_b_a_lone_back_half_leaves_the_group(self) -> None:
        # B: 2 本のうち 1 本だけがまたぐ 後ろが元のグループに残ると、後ろを選んだだけで前と
        # もう 1 本まで一緒に選ばれ、動かすとまとめて動く（利用者の報告）
        project, clips = _grouped((0, 100), (0, 30))
        group = clips[0].group_id
        split = SplitClips((clips[0].id,), 40).apply(project)
        assert _groups_from(split, 40) == {None}
        # 前と、割らなかった短い仲間は元のグループのまま
        assert _clip(split, clips[0].id).group_id == group
        assert _clip(split, clips[1].id).group_id == group

    def test_c_two_of_three_straddling_make_a_new_group(self) -> None:
        # C: 3 本のうち 2 本がまたぐ またいだ 2 本の後ろが新しいグループになり、短い 1 本は残る
        project, clips = _grouped((0, 100), (0, 100), (0, 30))
        group = clips[0].group_id
        split = SplitClips((clips[0].id, clips[1].id), 40).apply(project)
        back = _groups_from(split, 40)
        assert len(back) == 1 and None not in back and group not in back
        assert _clip(split, clips[2].id).group_id == group

    def test_d_one_of_three_straddling_leaves_the_group(self) -> None:
        # D: 3 本のうち 1 本だけがまたぐ B と同じく後ろは外れる
        project, clips = _grouped((0, 100), (0, 30), (0, 20))
        split = SplitClips((clips[0].id,), 40).apply(project)
        assert _groups_from(split, 40) == {None}

    def test_splitting_one_member_also_splits_the_straddling_partners(self) -> None:
        # グループは一緒に割る組 1 本だけを渡したときに仲間を割らないと、AI が 1 本ずつ
        # 割るたびに前と後ろが同じグループに残った（#427 AI の split_clip）
        project, clips = _grouped((0, 100), (0, 100))
        split = SplitClips((clips[0].id,), 40).apply(project)
        assert len(_all(split)) == 4
        back = _groups_from(split, 40)
        assert len(back) == 1 and None not in back and clips[0].group_id not in back

    def test_passing_one_member_or_all_gives_the_same_shape(self) -> None:
        # 画面は仲間ごと選んで渡し、AI は 1 本を渡す 渡し方で分かれ方が変わると、同じ頼みでも
        # 画面と AI で束ね方が違う
        project, clips = _grouped((0, 100), (0, 100), (0, 30))
        one = SplitClips((clips[0].id,), 40).apply(project)
        both = SplitClips((clips[0].id, clips[1].id), 40).apply(project)
        assert len(_all(one)) == len(_all(both)) == 5
        assert len(_groups_from(one, 40)) == len(_groups_from(both, 40)) == 1
        assert _clip(one, clips[2].id) == _clip(both, clips[2].id) == clips[2]

    def test_an_ungrouped_clip_stays_ungrouped(self) -> None:
        # グループの無いクリップの後ろへ新しいグループを付けると、割っただけで束ねられる
        # 別のグループの仲間を巻き込んで割ると、頼んでいない物まで切れる
        project, clips = _grouped((0, 100), (0, 100))
        lone = Clip(timeline_start=0, duration=100, source=TEXT.create())
        tracks = (*project.timeline.tracks, Track(TrackKind.MIXED, "レイヤー 3", (lone,)))
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        split = SplitClips((lone.id,), 40).apply(project)
        back = [c for c in _all(split) if c.timeline_start == 40]
        assert [c.group_id for c in back] == [None]
        assert _clip(split, clips[0].id) == clips[0] and _clip(split, clips[1].id) == clips[1]

    def test_a_locked_member_refuses_the_whole_split(self) -> None:
        # ロックした仲間だけを残してほかを割ると、束ねた物の頭や長さが食い違う（画面と同じ決まり）
        project, clips = _grouped((0, 100), (0, 100), locked=1)
        with pytest.raises(ValueError, match="ロック"):
            SplitClips((clips[0].id,), 40).apply(project)

    def test_one_undo_brings_the_group_back(self) -> None:
        # 取り消しで分けたグループが戻らないと、割る前の束ね方を人が組み直すことになる
        project, clips = _grouped((0, 100), (0, 30))
        document = Document(project)
        document.execute(SplitClips((clips[0].id,), 40))
        document.undo()
        assert document.project == project

    def test_linked_picture_and_sound_in_a_group_split_together(
        self, video_media: MediaItem
    ) -> None:
        # リンクした映像と音声は一緒に割れる 後ろの 2 本が 1 つのグループに入らないと、
        # 後ろの映像を選んでも音声が付いてこない
        project = _placed(Project.create(ProjectSettings()), video_media)
        picture, sound = _all(project)
        project = GroupClips((picture.id, sound.id)).apply(project)
        group = _clip(project, picture.id).group_id
        split = SplitClips((picture.id,), 60).apply(project)
        back = [c for c in _all(split) if c.timeline_start == 60]
        assert len(back) == 2
        assert len({c.group_id for c in back}) == 1
        assert back[0].group_id not in (None, group)
        assert len({c.link_group for c in back}) == 1
        assert back[0].link_group != _clip(split, picture.id).link_group


def _placed(base: Project, video_media: MediaItem) -> Project:
    project = replace(base, media=(*base.media, video_media))
    document = Document(project)
    for command in insert_media(project, video_media, split_audio=True):
        document.execute(command)
    return document.project


class TestInserting:
    def test_a_lone_back_half_leaves_the_group(self) -> None:
        # 挿入貼り付けで割った後ろも、分割と同じく元のグループに残さない 残すと、貼った物を
        # またいで前と後ろが一緒に動く
        project, clips = _grouped((0, 100), (0, 30))
        group = clips[0].group_id
        pushed = InsertGap(40, 20).apply(project)
        assert _groups_from(pushed, 60) == {None}
        assert _clip(pushed, clips[1].id).group_id == group

    def test_back_halves_together_make_a_new_group(self) -> None:
        # 2 本とも割ったときは、後ろどうしを束ねる（前からの動き）
        project, clips = _grouped((0, 100), (0, 100))
        pushed = InsertGap(40, 20).apply(project)
        back = _groups_from(pushed, 60)
        assert len(back) == 1 and None not in back and clips[0].group_id not in back


class TestRippleCut:
    def test_a_lone_back_piece_leaves_the_group(self) -> None:
        # 無音カット・範囲を切って詰めるでも、後ろの欠片を元のグループに残さない 残すと、
        # 何か所も切った後でも全部が 1 つのグループで、1 か所の後ろだけを選べない
        project, clips = _grouped((0, 100), (0, 30))
        group = clips[0].group_id
        cut = RippleCut(((40, 60),)).apply(project)
        assert _groups_from(cut, 40) == {None}
        assert _clip(cut, clips[0].id).group_id == group
        assert _clip(cut, clips[1].id).group_id == group

    def test_back_pieces_of_one_range_share_a_new_group(self) -> None:
        # 同じ範囲をまたいだ 2 本の後ろは一緒に束ねる 1 本ずつ外すと、映像とテロップのように
        # 一緒に作った物が切っただけでばらばらになる
        project, clips = _grouped((0, 100), (0, 100))
        group = clips[0].group_id
        cut = RippleCut(((40, 60),)).apply(project)
        back = _groups_from(cut, 40)
        assert len(back) == 1 and None not in back and group not in back

    def test_each_range_makes_its_own_group(self) -> None:
        # 何か所も切ったとき、欠片ごとに別のグループになる 同じ新しいグループにまとめると、
        # 2 か所目より後ろの欠片が 1 か所目の後ろと束ねられたままになる
        project, _ = _grouped((0, 100), (0, 100))
        cut = RippleCut(((20, 30), (50, 60))).apply(project)
        middle = _groups_from(cut, 20)
        last = _groups_from(cut, 40)
        assert len(middle) == 1 and len(last) == 1
        assert middle != last and None not in middle | last
