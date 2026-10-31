"""詰めるときに重ねの層も一緒に動かす（#413 の 8 番）

通しの記録の手順をそのまま起こす 混合の方式で素材を置き、無音を詰めてつなぎ目へ場面切り替えを
付けてから、頭のクリップを詰めて消す 前は消した素材のレイヤーだけが詰まり、場面切り替えの
レイヤーが元の位置に残って、つなぎ目とずれた AI はそこから全部を戻すしかなかった
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.core.commands import (
    AddClip,
    AddMarkers,
    AddTrack,
    RemoveClip,
    RemoveClips,
    RippleCut,
)
from sashimono.core.commands.edit import ripple_casualties, ripple_gaps
from sashimono.core.commands.insert import insert_media
from sashimono.core.model import (
    TRANSITION_KIND,
    Clip,
    LayerMode,
    Marker,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    new_group_id,
)
from sashimono.core.seams import (
    SEAM_CROSSFADE,
    SeamScope,
    SeamStyle,
    SmoothCuts,
    seams_after_cuts,
)
from sashimono.effects.seams import seam_transition
from sashimono.effects.sources import TEXT
from tests.conftest import RATE_30, make_clip

#: 通しの記録と同じ形 詰める所が 2 か所（100〜120 と 200〜230）
_RANGES = ((100, 120), (200, 230))
_STYLE = SeamStyle(video=SEAM_CROSSFADE, crossfade_ms=100, zoom_percent=110, audio_fade_ms=20)


def _smoothed(media: MediaItem) -> Project:
    """素材を置き、無音を詰めて、つなぎ目へ場面切り替えを付けた作品"""
    project = Project.create(ProjectSettings(frame_rate=RATE_30, layer_mode=LayerMode.MIXED))
    for command in insert_media(project, media, at_frame=0, split_audio=True):
        project = command.apply(project)
    project = RippleCut(_RANGES).apply(project)
    frames = seams_after_cuts(_RANGES)
    command = SmoothCuts(frames, _STYLE, seam_transition(_STYLE), SeamScope(jumps_only=True))
    return command.apply(project)


def _transitions(project: Project) -> list[int]:
    return sorted(
        clip.timeline_start
        for track in project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == TRANSITION_KIND
    )


def _pictures(project: Project) -> list[Clip]:
    return sorted(
        (
            clip
            for track in project.timeline.tracks
            for clip in track.clips
            if clip.media_id is not None and clip.audio_stream is None and _pictured(project, clip)
        ),
        key=lambda clip: clip.timeline_start,
    )


def _pictured(project: Project, clip: Clip) -> bool:
    located = project.timeline.locate_clip(clip.id)
    assert located is not None
    return located[0].kind is TrackKind.VIDEO or clip.show_picture


def _seams(project: Project) -> list[int]:
    return [clip.timeline_start for clip in _pictures(project)][1:]


@pytest.fixture
def smoothed(video_media: MediaItem) -> Project:
    project = _smoothed(video_media)
    # 手順の前提 つなぎ目のそれぞれの 1 フレーム前に場面切り替えが付いている
    assert _transitions(project) == [seam - 1 for seam in _seams(project)]
    return project


class TestTheRunFromTheLog:
    def test_the_old_way_leaves_the_transitions_behind(self, smoothed: Project) -> None:
        """通しの記録の再現 ``layers`` を渡さない詰め方は、場面切り替えのレイヤーを動かさない"""
        head = _pictures(smoothed)[0]
        after = RemoveClip(head.id, ripple=True).apply(smoothed)
        assert _transitions(after) != [seam - 1 for seam in _seams(after)]

    def test_cutting_the_head_keeps_the_transitions_on_their_seams(self, smoothed: Project) -> None:
        """壊れると、つなぎ目の処理の後に頭を切っただけで全部の場面切り替えが別の所で溶ける"""
        head = _pictures(smoothed)[0]
        after = RemoveClip(head.id, ripple=True, layers=True).apply(smoothed)
        # 頭のつなぎ目は消えたので、そこの場面切り替えも一緒に消える
        assert _transitions(after) == [seam - 1 for seam in _seams(after)]
        assert len(_transitions(after)) == 1

    def test_a_cut_in_the_middle_drops_both_seams_of_the_gap(self, smoothed: Project) -> None:
        """両隣のつなぎ目の場面切り替えは溶かす相手を失う 欠片を残すと 1 コマの溶けが 2 つ並ぶ"""
        middle = _pictures(smoothed)[1]
        after = RemoveClip(middle.id, ripple=True, layers=True).apply(smoothed)
        assert _transitions(after) == []
        assert len(_seams(after)) == 1

    def test_several_clips_at_once(self, smoothed: Project) -> None:
        """壊れると、まとめて消す道具と 1 本ずつ消す道具で重ねの層の動きが食い違う"""
        pictures = _pictures(smoothed)
        ids = (pictures[0].id, pictures[2].id)
        after = RemoveClips(ids, ripple=True, layers=True).apply(smoothed)
        assert after.timeline.duration == pictures[1].duration
        assert _transitions(after) == []


def _text_track(project: Project, start: int, duration: int) -> tuple[Project, Clip]:
    """テキストを 1 本だけ持つトラックを足す"""
    clip = Clip(timeline_start=start, duration=duration, source=TEXT.create(text="札"))
    return _with_track(project, Track(kind=TrackKind.VIDEO, name="字幕"), clip), clip


def _at(project: Project, clip: Clip) -> tuple[int, int] | None:
    located = project.timeline.locate_clip(clip.id)
    return None if located is None else (located[1].timeline_start, located[1].duration)


class _Plain:
    """映像 V1 に 100 フレームずつ 3 本（消す相手）を並べた作品"""

    def __init__(self, media: MediaItem, music: MediaItem) -> None:
        self.track = Track(kind=TrackKind.VIDEO, name="V1")
        project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(media, music))
        project = project.with_timeline(replace(project.timeline, tracks=(self.track,)))
        self.clips = [make_clip(start, 100, media) for start in (0, 100, 200)]
        for clip in self.clips:
            project = AddClip(self.track.id, clip).apply(project)
        self.project = project

    def cut_middle(self, project: Project) -> Project:
        return RemoveClip(self.clips[1].id, ripple=True, layers=True).apply(project)


@pytest.fixture
def plain(video_media: MediaItem, audio_media: MediaItem) -> _Plain:
    return _Plain(video_media, audio_media)


def _with_track(project: Project, track: Track, *clips: Clip) -> Project:
    project = AddTrack(track).apply(project)
    for clip in clips:
        project = AddClip(track.id, clip).apply(project)
    return project


class TestWhatFollows:
    def test_later_text_and_subtitles_move(self, plain: _Plain) -> None:
        """壊れると、焼き込んだ字幕とテロップが話している絵からずれる"""
        project, text = _text_track(plain.project, 220, 30)
        assert _at(plain.cut_middle(project), text) == (120, 30)

    def test_a_text_inside_the_gap_goes_with_it(self, plain: _Plain) -> None:
        """壊れると、消した話の字幕が次の話の上に残る"""
        project, text = _text_track(plain.project, 120, 30)
        assert _at(plain.cut_middle(project), text) is None

    def test_a_long_title_is_shortened(self, plain: _Plain) -> None:
        """壊れると、詰めた後も長いタイトルが元の長さのまま次の字幕へ掛かる"""
        project, text = _text_track(plain.project, 50, 200)
        after = plain.cut_middle(project)
        # 切った前後の 2 本に分かれる（無音カットと同じ切り方）
        assert _at(after, text) == (50, 50)
        located = after.timeline.locate_clip(text.id)
        assert located is not None
        assert [(c.timeline_start, c.duration) for c in located[0].clips] == [(50, 50), (100, 50)]

    def test_a_later_sound_effect_moves(self, plain: _Plain, audio_media: MediaItem) -> None:
        """壊れると、絵に合わせて置いた効果音だけが元の時刻で鳴る"""
        effect = make_clip(250, 10, audio_media)
        project = _with_track(plain.project, Track(kind=TrackKind.AUDIO, name="効果音"), effect)
        assert _at(plain.cut_middle(project), effect) == (150, 10)

    def test_music_under_the_gap_stays(self, plain: _Plain, audio_media: MediaItem) -> None:
        """壊れると、話を 1 つ消しただけで BGM の途中が切れて音が飛ぶ"""
        music = make_clip(0, 300, audio_media)
        project = _with_track(plain.project, Track(kind=TrackKind.AUDIO, name="BGM"), music)
        assert _at(plain.cut_middle(project), music) == (0, 300)

    def test_the_track_behind_the_music_stays_whole(
        self, plain: _Plain, audio_media: MediaItem
    ) -> None:
        """壊れると、続きの曲だけが前へ動いて前の曲に重なり、削除ごと断られる"""
        first = make_clip(0, 250, audio_media)
        second = make_clip(250, 50, audio_media)
        project = _with_track(plain.project, Track(kind=TrackKind.AUDIO, name="BGM"), first, second)
        assert _at(plain.cut_middle(project), second) == (250, 50)

    def test_a_sound_inside_the_gap_does_not_hold_the_track(
        self, plain: _Plain, audio_media: MediaItem
    ) -> None:
        """壊れると、消す範囲の中で終わる効果音 1 つのせいで、後ろの効果音が絵からずれる"""
        inside = make_clip(150, 20, audio_media)
        later = make_clip(250, 10, audio_media)
        track = Track(kind=TrackKind.AUDIO, name="効果音")
        after = plain.cut_middle(_with_track(plain.project, track, inside, later))
        assert _at(after, inside) is None
        assert _at(after, later) == (150, 10)

    def test_a_sound_ending_in_the_gap_is_shortened(
        self, plain: _Plain, audio_media: MediaItem
    ) -> None:
        """壊れると、消した時間へ掛かった音が残り、後ろの音が動けずにずれる"""
        tail = make_clip(50, 100, audio_media)
        later = make_clip(250, 10, audio_media)
        track = Track(kind=TrackKind.AUDIO, name="効果音")
        after = plain.cut_middle(_with_track(plain.project, track, tail, later))
        assert _at(after, tail) == (50, 50)
        assert _at(after, later) == (150, 10)

    def test_a_linked_partner_of_the_stayed_track_stays(
        self, plain: _Plain, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        """壊れると、後ろの映像だけが動き、リンクした音と絵が黙ってずれる"""
        group = new_group_id()
        picture = replace(make_clip(250, 30, video_media), link_group=group)
        sound = replace(make_clip(250, 30, audio_media), link_group=group)
        music = make_clip(0, 240, audio_media)
        project = _with_track(plain.project, Track(kind=TrackKind.VIDEO, name="V2"), picture)
        project = _with_track(project, Track(kind=TrackKind.AUDIO, name="A1"), music, sound)
        after = plain.cut_middle(project)
        assert _at(after, picture) == (250, 30)
        assert _at(after, sound) == (250, 30)

    def test_a_partner_of_an_earlier_locked_clip_stays(self, plain: _Plain) -> None:
        """壊れると、消す所より前のロックしたクリップの仲間だけが動き、組の並びが崩れる"""
        group = new_group_id()
        fixed = Clip(timeline_start=10, duration=5, source=TEXT.create(text="札"), group_id=group)
        partner = Clip(
            timeline_start=250, duration=5, source=TEXT.create(text="札"), group_id=group
        )
        locked = Track(kind=TrackKind.VIDEO, name="T1", clips=(fixed,), locked=True)
        others = Track(kind=TrackKind.VIDEO, name="T2", clips=(partner,))
        project = plain.project.with_timeline(
            replace(plain.project.timeline, tracks=(*plain.project.timeline.tracks, locked, others))
        )
        assert _at(plain.cut_middle(project), partner) == (250, 5)

    @pytest.mark.parametrize("hidden", ["disabled", "muted"])
    def test_music_that_is_not_playing_does_not_hold_the_track(
        self, plain: _Plain, audio_media: MediaItem, hidden: str
    ) -> None:
        """壊れると、鳴っていない曲のせいで、同じトラックの効果音が絵からずれる"""
        music = replace(make_clip(0, 240, audio_media), enabled=hidden != "disabled")
        effect = make_clip(250, 10, audio_media)
        track = Track(kind=TrackKind.AUDIO, name="BGM", muted=hidden == "muted")
        after = plain.cut_middle(_with_track(plain.project, track, music, effect))
        assert _at(after, effect) == (150, 10)

    def test_a_locked_track_stays(self, plain: _Plain) -> None:
        """壊れると、人が動かさないと決めたトラックが動く"""
        text_clip = Clip(timeline_start=250, duration=10, source=TEXT.create(text="札"))
        locked = Track(kind=TrackKind.VIDEO, name="T1", clips=(text_clip,), locked=True)
        project = plain.project.with_timeline(
            replace(plain.project.timeline, tracks=(*plain.project.timeline.tracks, locked))
        )
        assert _at(plain.cut_middle(project), text_clip) == (250, 10)

    def test_markers_and_the_work_area_follow(self, plain: _Plain) -> None:
        """壊れると、話の区切りの目印と書き出す範囲が中身から外れる"""
        marks = (Marker(frame=250, label="区切り"), Marker(frame=150, label="消える所"))
        project = AddMarkers(marks).apply(plain.project)
        project = project.with_timeline(replace(project.timeline, work_area=(50, 250)))
        after = plain.cut_middle(project)
        assert [(m.frame, m.label) for m in after.timeline.markers] == [(150, "区切り")]
        assert after.timeline.work_area == (50, 150)


class TestWhenTheProgrammeGoesOn:
    def test_an_overlay_does_not_pull_the_text(self, plain: _Plain, video_media: MediaItem) -> None:
        """画面の絵が続くのに字幕まで詰めると、下の映像と字幕がずれる（前からの動きを守る）"""
        over = make_clip(120, 30, video_media)
        project = _with_track(plain.project, Track(kind=TrackKind.VIDEO, name="V2"), over)
        project, text = _text_track(project, 200, 20)
        assert not ripple_gaps(project, (over.id,))
        after = RemoveClip(over.id, ripple=True, layers=True).apply(project)
        assert _at(after, text) == (200, 20)

    def test_a_disabled_picture_is_not_the_programme(
        self, plain: _Plain, video_media: MediaItem
    ) -> None:
        """壊れると、映っていない絵のせいで字幕が付いてこず、消した絵とずれる"""
        over = replace(make_clip(120, 30, video_media), enabled=False)
        project = _with_track(plain.project, Track(kind=TrackKind.VIDEO, name="V2"), over)
        project, text = _text_track(project, 220, 10)
        assert _at(plain.cut_middle(project), text) == (120, 10)

    def test_a_muted_track_is_not_the_programme(
        self, plain: _Plain, video_media: MediaItem
    ) -> None:
        """壊れると、隠したトラックの絵のせいで字幕が付いてこず、消した絵とずれる"""
        over = make_clip(120, 30, video_media)
        track = Track(kind=TrackKind.VIDEO, name="V2", muted=True)
        project, text = _text_track(_with_track(plain.project, track, over), 220, 10)
        assert _at(plain.cut_middle(project), text) == (120, 10)

    def test_a_sound_under_other_sound_does_not_pull(
        self, plain: _Plain, audio_media: MediaItem
    ) -> None:
        """効果音を 1 つ消しただけで映像の上の字幕が動くと、絵と字幕がずれる"""
        effect = make_clip(120, 10, audio_media)
        project = _with_track(plain.project, Track(kind=TrackKind.AUDIO, name="効果音"), effect)
        project, text = _text_track(project, 220, 10)
        assert not ripple_gaps(project, (effect.id,))
        after = RemoveClip(effect.id, ripple=True, layers=True).apply(project)
        assert _at(after, text) == (220, 10)

    def test_without_the_setting_nothing_else_moves(self, plain: _Plain) -> None:
        """壊れると、設定で切った人の「削除して詰める」が重ねの層まで動かす"""
        project, text = _text_track(plain.project, 220, 30)
        after = RemoveClip(plain.clips[1].id, ripple=True, layers=False).apply(project)
        assert _at(after, text) == (220, 30)

    def test_a_delete_without_ripple_moves_nothing(self, plain: _Plain) -> None:
        """壊れると、詰めない削除で後ろの字幕が動く"""
        project, text = _text_track(plain.project, 220, 30)
        after = RemoveClip(plain.clips[1].id, ripple=False, layers=True).apply(project)
        assert _at(after, text) == (220, 30)


class TestSeveralClipsTogether:
    def test_overlapping_clips_are_one_cut(self, plain: _Plain, video_media: MediaItem) -> None:
        """壊れると、重なった 2 本を消しても字幕が付いてこず、トラックごとに詰める量がずれる"""
        over = make_clip(120, 30, video_media)
        later = make_clip(250, 10, video_media)
        project = _with_track(plain.project, Track(kind=TrackKind.VIDEO, name="V2"), over, later)
        project, text = _text_track(project, 220, 10)
        after = RemoveClips((plain.clips[1].id, over.id), ripple=True, layers=True).apply(project)
        assert _at(after, plain.clips[2]) == (100, 100)
        assert _at(after, later) == (150, 10)
        assert _at(after, text) == (120, 10)

    def test_the_order_of_the_selection_does_not_matter(
        self, plain: _Plain, video_media: MediaItem
    ) -> None:
        """壊れると、選んだ順で詰まり方が変わる"""
        over = make_clip(120, 30, video_media)
        project = _with_track(plain.project, Track(kind=TrackKind.VIDEO, name="V2"), over)
        project, text = _text_track(project, 220, 10)
        after = RemoveClips((over.id, plain.clips[1].id), ripple=True, layers=True).apply(project)
        assert _at(after, text) == (120, 10)

    def test_a_cut_with_the_picture_going_on_and_a_gap_are_told_apart(
        self, plain: _Plain, video_media: MediaItem
    ) -> None:
        """壊れると、絵が続く所の 1 本のせいで、番組の時間が抜ける所まで字幕が付いてこない"""
        over = make_clip(20, 10, video_media)
        project = _with_track(plain.project, Track(kind=TrackKind.VIDEO, name="V2"), over)
        project, text = _text_track(project, 220, 10)
        after = RemoveClips((over.id, plain.clips[1].id), ripple=True, layers=True).apply(project)
        # 上の絵は下の絵が続くので字幕を動かさず、真ん中の 100 だけが字幕を動かす
        assert _at(after, text) == (120, 10)

    def test_a_locked_clip_stops_them_all(self, plain: _Plain, video_media: MediaItem) -> None:
        """壊れると、途中まで消えてから断られ、どれが消えたのかを追えない"""
        over = make_clip(120, 30, video_media)
        track = Track(kind=TrackKind.VIDEO, name="V2", clips=(over,), locked=True)
        project = plain.project.with_timeline(
            replace(plain.project.timeline, tracks=(*plain.project.timeline.tracks, track))
        )
        with pytest.raises(ValueError, match="ロック"):
            RemoveClips((plain.clips[1].id, over.id), ripple=True, layers=True).apply(project)

    def test_the_order_of_several_gaps_keeps_the_amount(self, plain: _Plain) -> None:
        """壊れると、前のすき間の分だけ後ろの字幕の詰める量がずれる"""
        project, text = _text_track(plain.project, 250, 10)
        ids = (plain.clips[0].id, plain.clips[1].id)
        after = RemoveClips(ids, ripple=True, layers=True).apply(project)
        assert _at(after, text) == (50, 10)

    def test_the_answer_reads_the_whole_cut(self, plain: _Plain, video_media: MediaItem) -> None:
        """壊れると、まとめて消せば動く層を「動かない」と AI に答える"""
        over = make_clip(120, 30, video_media)
        project = _with_track(plain.project, Track(kind=TrackKind.VIDEO, name="V2"), over)
        assert not ripple_gaps(project, (plain.clips[1].id,))
        assert ripple_gaps(project, (plain.clips[1].id, over.id)) == [(100, 200)]


class TestWhatWentWithIt:
    def test_a_sound_inside_the_gap_is_named(self, plain: _Plain, audio_media: MediaItem) -> None:
        """壊れると、消すと頼んでいない音が黙って消える"""
        inside = make_clip(150, 20, audio_media)
        track = Track(kind=TrackKind.AUDIO, name="効果音")
        project = _with_track(plain.project, track, inside)
        after = plain.cut_middle(project)
        lost = ripple_casualties(project, after, (plain.clips[1].id,))
        assert [(found.id, clip.id) for found, clip in lost] == [(track.id, inside.id)]

    def test_the_clip_asked_for_is_not_counted(self, plain: _Plain) -> None:
        """壊れると、頼んだ 1 本まで「一緒に消えた」と言う"""
        after = plain.cut_middle(plain.project)
        assert ripple_casualties(plain.project, after, (plain.clips[1].id,)) == []

    def test_a_linked_partner_is_not_counted(self, video_media: MediaItem) -> None:
        """壊れると、映像を消しただけでリンクした音が消えたと毎回出る"""
        project = _smoothed(video_media)
        head = _pictures(project)[0]
        after = RemoveClip(head.id, ripple=True, layers=True).apply(project)
        lost = ripple_casualties(project, after, (head.id,))
        assert [clip.source.kind for _, clip in lost if clip.source is not None] == [
            TRANSITION_KIND
        ]
        assert all(clip.media_id is None for _, clip in lost)

    def test_a_shortened_clip_is_not_counted(self, plain: _Plain) -> None:
        """壊れると、短くなっただけの字幕を消えたと言う"""
        project, _text = _text_track(plain.project, 50, 100)
        after = plain.cut_middle(project)
        assert ripple_casualties(project, after, (plain.clips[1].id,)) == []


class TestRippleCutAndTheWorkArea:
    def test_the_silence_cut_still_drops_it(self, plain: _Plain) -> None:
        """壊れると、無音カットの前からの動きが変わる"""
        project = plain.project.with_timeline(replace(plain.project.timeline, work_area=(50, 250)))
        assert RippleCut(((100, 120),)).apply(project).timeline.work_area is None

    def test_keeping_it_follows_the_cut(self, plain: _Plain) -> None:
        """壊れると、範囲を切っただけで書き出す範囲が外れ、次の書き出しが全体になる"""
        project = plain.project.with_timeline(replace(plain.project.timeline, work_area=(50, 250)))
        after = RippleCut(((100, 120),), keep_area=True).apply(project)
        assert after.timeline.work_area == (50, 230)

    def test_a_cut_over_all_of_it_drops_it(self, plain: _Plain) -> None:
        """壊れると、中身の無くなった書き出す範囲が残り、長さ 0 の書き出しになる"""
        project = plain.project.with_timeline(replace(plain.project.timeline, work_area=(100, 120)))
        after = RippleCut(((90, 130),), keep_area=True).apply(project)
        assert after.timeline.work_area is None


class TestRippleCutOnTransitions:
    def test_a_second_cut_drops_the_transition_it_touches(self, smoothed: Project) -> None:
        """壊れると、つなぎ目の場面切り替えが 1 コマの欠片になって残り、そこだけ絵が跳ねる"""
        seam = _seams(smoothed)[0]
        after = RippleCut(((seam, seam + 10),)).apply(smoothed)
        assert seam - 1 not in _transitions(after)
        assert len(_transitions(after)) == 1
        assert _transitions(after) == [_seams(smoothed)[1] - 10 - 1]
