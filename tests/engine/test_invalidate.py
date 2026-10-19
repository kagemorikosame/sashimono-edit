"""編集で、どのフレームの絵が変わったかの求め方

先読みした絵を捨てる範囲がここで決まる 捨て損ねると古い絵が画面に残り、
捨てすぎると先読みが効かなくなる どちらも外から見えにくいので数値で押さえる
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    Effect,
    MediaItem,
    Project,
    ProjectSettings,
    Scene,
    Timeline,
    Track,
    TrackKind,
    Transcript,
    VideoStreamInfo,
)
from sashimono.core.model.ids import new_group_id
from sashimono.core.timebase import FrameRate
from sashimono.engine.render import Invalidation, changed_spans

RATE = FrameRate(30)


def _clip(start: int, duration: int, **extra: Any) -> Clip:
    return Clip(timeline_start=start, duration=duration, **extra)


#: 見本のトラック 組み立てるたびに新しい ID を振ると、クリップを 1 本動かした
#: だけでも「トラックが入れ替わった」と読まれて全部捨てる形になる
VIDEO_TRACK = Track(kind=TrackKind.VIDEO)


def _project(*clips: Clip, scenes: tuple[Scene, ...] = ()) -> Project:
    track = replace(VIDEO_TRACK, clips=tuple(clips))
    return Project(
        settings=ProjectSettings(frame_rate=RATE),
        timeline=Timeline(rate=RATE, tracks=(track,)),
        scenes=scenes,
    )


def _media(path: str = "a.mp4") -> MediaItem:
    return MediaItem(
        path=Path(path),
        duration=Fraction(10),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1920,
                height=1080,
                frame_rate=RATE,
                time_base=Fraction(1, 30),
                codec="h264",
            ),
        ),
    )


class TestTheSpans:
    def test_touching_ranges_merge(self) -> None:
        """隣り合う範囲はまとめる

        まとめないと、1 フレームずつ足した範囲が数千個たまり、1 枚捨てるか
        どうかを決めるのにその全部を見ることになる
        """
        assert Invalidation.over([(0, 10), (10, 20), (5, 7)]).spans == ((0, 20),)

    def test_empty_ranges_drop_out(self) -> None:
        # 長さ 0 の範囲は 1 枚も含まない 残すと、当たらない条件を毎回見ることになる
        assert Invalidation.over([(5, 5)]).spans == ()

    def test_everything_contains_any_frame(self) -> None:
        # 全部捨てる指示が遠いフレームを含まないと、長い動画の後半で変える前の古い絵が映る
        assert Invalidation.all().contains(99999)

    def test_nothing_is_false(self) -> None:
        # 捨てるものが無いときは、呼ぶ側が何もしないで済むように偽になる
        assert not Invalidation.nothing()


class TestWhatDoesNotChangeThePicture:
    def test_the_same_project_changes_nothing(self) -> None:
        # 同じプロジェクトで範囲を返すと、何も変えていない更新のたびに先読みした絵が消え、
        # 再生がそのたびに描き直しで止まる
        project = _project(_clip(0, 30))
        assert not changed_spans(project, project)

    def test_renaming_changes_nothing(self) -> None:
        """名前を変えても絵は変わらない

        これを全部捨てる扱いにすると、プロジェクト名を打つ 1 文字ごとに
        貯めた絵が消える
        """
        before = _project(_clip(0, 30))
        assert not changed_spans(before, replace(before, name="別名"))

    def test_the_sound_does_not_touch_the_picture(self) -> None:
        """音量をいじっても絵は変わらない 音を合わせている間ずっと作り直さない"""
        audio = Track(kind=TrackKind.AUDIO, clips=(_clip(0, 30),))
        before = _project(_clip(0, 30))
        before = replace(
            before, timeline=replace(before.timeline, tracks=(*before.timeline.tracks, audio))
        )
        louder = replace(audio, volume_db=6.0)
        after = replace(before, timeline=before.timeline.replace_track(louder))
        assert not changed_spans(before, after)

    def test_the_work_area_does_not_touch_the_picture(self) -> None:
        """書き出し範囲を変えても絵は変わらない

        捨てる扱いにすると、範囲の端をつまんで動かすたびに貯めた絵が消える
        """
        before = _project(_clip(0, 30))
        after = replace(before, timeline=replace(before.timeline, work_area=(0, 10)))
        assert not changed_spans(before, after)


class TestWhatChangesEverything:
    def test_a_new_resolution_drops_all(self) -> None:
        """解像度が変わったら取ってある絵の大きさが合わない"""
        before = _project(_clip(0, 30))
        after = replace(before, settings=ProjectSettings(width=1280, height=720, frame_rate=RATE))
        assert changed_spans(before, after).everything

    def test_a_new_rate_drops_all(self) -> None:
        """フレームレートが変われば、同じ番号が別の時刻を指す"""
        before = _project(_clip(0, 30))
        after = Project(
            settings=ProjectSettings(frame_rate=FrameRate(60)),
            timeline=Timeline(rate=FrameRate(60), tracks=before.timeline.tracks),
        )
        assert changed_spans(before, after).everything

    def test_a_new_track_drops_all(self) -> None:
        """トラックが増えれば、クリップの無いフレームの絵まで変わりうる"""
        before = _project(_clip(0, 30))
        added = Track(kind=TrackKind.VIDEO, clips=())
        after = replace(
            before, timeline=replace(before.timeline, tracks=(*before.timeline.tracks, added))
        )
        assert changed_spans(before, after).everything

    def test_muting_a_track_drops_all(self) -> None:
        """ミュートは見えるトラックの並びを変える 下の絵が透ける"""
        before = _project(_clip(0, 30))
        muted = replace(before.timeline.tracks[0], muted=True)
        after = replace(before, timeline=before.timeline.replace_track(muted))
        assert changed_spans(before, after).everything


class TestWhatChangesOnePlace:
    def test_moving_a_clip_drops_both_places(self) -> None:
        """動かしたクリップは、消えた場所と現れた場所の両方の絵が変わる"""
        clip = _clip(0, 30)
        before = _project(clip)
        after = _project(clip.moved_to(100))
        assert changed_spans(before, after).spans == ((0, 30), (100, 130))

    def test_an_untouched_clip_keeps_its_frames(self) -> None:
        """離れたクリップを触っても、こちらの範囲は残る 先読みの値打ちはここ"""
        first, second = _clip(0, 30), _clip(100, 30)
        before = _project(first, second)
        after = _project(first, replace(second, enabled=False))
        result = changed_spans(before, after)
        assert not result.contains(10)
        assert result.contains(100)

    def test_removing_a_clip_drops_its_frames(self) -> None:
        # 消したクリップの所を捨てないと、クリップを消してもプレビューにその絵が残って見える
        first, second = _clip(0, 30), _clip(100, 30)
        before = _project(first, second)
        after = _project(first)
        assert changed_spans(before, after).spans == ((100, 130),)

    def test_a_track_filter_drops_only_where_clips_are(self) -> None:
        """トラック全体のフィルタも、掛かるのはクリップのある所だけ"""
        before = _project(_clip(0, 30), _clip(100, 30))
        filtered = replace(before.timeline.tracks[0], effects=(Effect(kind="blur"),))
        after = replace(before, timeline=before.timeline.replace_track(filtered))
        assert changed_spans(before, after).spans == ((0, 30), (100, 130))


class TestWhatTheRendererDoesNotReadOnTracks:
    def test_renaming_a_track_does_not_touch_the_picture(self) -> None:
        """トラックの名前を変えても絵は変わらない

        捨てる扱いにすると、名前を打ち直すたびにその上のクリップの絵が全部消える
        """
        before = _project(_clip(0, 30))
        renamed = replace(before.timeline.tracks[0], name="人物")
        after = replace(before, timeline=before.timeline.replace_track(renamed))
        assert not changed_spans(before, after)

    def test_the_row_height_does_not_touch_the_picture(self) -> None:
        """タイムラインの行の高さは見た目の都合 絵には出ない"""
        before = _project(_clip(0, 30))
        taller = replace(before.timeline.tracks[0], height=120)
        after = replace(before, timeline=before.timeline.replace_track(taller))
        assert not changed_spans(before, after)

    def test_locking_a_track_does_not_touch_the_picture(self) -> None:
        """鍵を掛けても絵は変わらない

        捨てる扱いにすると、間違って動かさないよう鍵を掛け外しするたびに
        その上のクリップの絵が消え、描き直しになる
        """
        before = _project(_clip(0, 30))
        locked = replace(before.timeline.tracks[0], locked=True)
        after = replace(before, timeline=before.timeline.replace_track(locked))
        assert not changed_spans(before, after)


class TestWhatTheRendererDoesNotReadOnClips:
    def test_grouping_does_not_touch_the_picture(self) -> None:
        """束ね直しても絵は変わらない レンダラは束ねを読まない

        並べ終えた後にまとめて束ねる使い方で、貯めた絵が全部消えていた
        """
        clip = _clip(0, 30)
        before = _project(clip)
        after = _project(replace(clip, group_id=new_group_id()))
        assert not changed_spans(before, after)

    def test_linking_does_not_touch_the_picture(self) -> None:
        # 映像と音のリンクで捨てると、リンクを付け外しするたびに貯めた絵が消えて描き直しになる
        clip = _clip(0, 30)
        before = _project(clip)
        after = _project(replace(clip, link_group=new_group_id()))
        assert not changed_spans(before, after)

    def test_turning_a_clip_off_still_does(self) -> None:
        # 外しすぎていないことの裏 表示の切り替えは絵に出る
        clip = _clip(0, 30)
        before = _project(clip)
        after = _project(replace(clip, enabled=False))
        assert changed_spans(before, after).contains(0)


class TestTheMedia:
    def test_replacing_a_source_drops_the_clips_that_read_it(self) -> None:
        """素材を差し替えたら、その素材を読むクリップの絵が変わる"""
        media = _media()
        clip = _clip(0, 30, media_id=media.id)
        before = replace(_project(clip, _clip(100, 30)), media=(media,))
        after = replace(before, media=(replace(media, path=Path("b.mp4")),))
        result = changed_spans(before, after)
        assert result.contains(10)
        assert not result.contains(100)


class TestTheScenes:
    def _with_scene(self, inner: Timeline) -> tuple[Project, Scene]:
        scene = Scene(name="中身", timeline=inner)
        clip = _clip(50, 30, scene_id=scene.id)
        return _project(_clip(0, 30), clip, scenes=(scene,)), scene

    def test_editing_a_scene_drops_the_clip_that_places_it(self) -> None:
        """シーンの中を直したら、置いたクリップの範囲を丸ごと捨てる

        中の 1 フレームが外の何フレーム目に出るかは速度と開始位置で決まる
        そこまで追わない（シーンは繰り返し使う部品で、編集の頻度が低い）
        """
        inner = Timeline(rate=RATE, tracks=(Track(kind=TrackKind.VIDEO, clips=(_clip(0, 30),)),))
        before, scene = self._with_scene(inner)
        edited = replace(
            scene,
            timeline=Timeline(
                rate=RATE, tracks=(Track(kind=TrackKind.VIDEO, clips=(_clip(0, 20),)),)
            ),
        )
        after = replace(before, scenes=(edited,))
        result = changed_spans(before, after)
        assert result.contains(50)
        assert not result.contains(0)

    def test_a_nested_scene_reaches_the_outside(self) -> None:
        """シーンの中に置いたシーンを直しても、外の絵は変わる

        たどらないと、2 段入れ子にした所だけ古い絵が残り続ける
        """
        deep = Scene(
            name="奥",
            timeline=Timeline(
                rate=RATE, tracks=(Track(kind=TrackKind.VIDEO, clips=(_clip(0, 30),)),)
            ),
        )
        middle = Scene(
            name="中",
            timeline=Timeline(
                rate=RATE,
                tracks=(Track(kind=TrackKind.VIDEO, clips=(_clip(0, 30, scene_id=deep.id),)),),
            ),
        )
        outer = _clip(50, 30, scene_id=middle.id)
        before = _project(_clip(0, 30), outer, scenes=(deep, middle))
        changed = replace(
            deep,
            timeline=Timeline(
                rate=RATE, tracks=(Track(kind=TrackKind.VIDEO, clips=(_clip(0, 20),)),)
            ),
        )
        after = replace(before, scenes=(changed, middle))
        result = changed_spans(before, after)
        assert result.contains(50)
        assert not result.contains(0)

    def test_replacing_a_source_used_inside_a_scene_reaches_the_outside(self) -> None:
        """シーンの中で使っている素材を差し替えても、外の絵は変わる

        中のクリップは素材を ID で指しているので、差し替えてもシーンの
        タイムラインは同じまま 見落とすと、外へ置いた所だけ古い絵が残る
        """
        media = _media()
        inner = Timeline(
            rate=RATE,
            tracks=(Track(kind=TrackKind.VIDEO, clips=(_clip(0, 30, media_id=media.id),)),),
        )
        before, _ = self._with_scene(inner)
        before = replace(before, media=(media,))
        after = replace(before, media=(replace(media, path=Path("b.mp4")),))
        result = changed_spans(before, after)
        assert result.contains(50), "シーンの中の素材差し替えが外へ届いていない"
        assert not result.contains(0)

    def test_sound_inside_a_scene_does_not_reach_the_outside(self) -> None:
        """シーンの中で音量を動かしても、外の絵は変わらない

        タイムラインを丸ごと比べると、置いた所の絵を全部捨てることになる
        メインのタイムラインでは同じものを無視しているので、扱いがそろわない
        """
        audio = Track(kind=TrackKind.AUDIO, clips=(_clip(0, 30),))
        inner = Timeline(
            rate=RATE,
            tracks=(Track(kind=TrackKind.VIDEO, clips=(_clip(0, 30),)), audio),
        )
        before, scene = self._with_scene(inner)
        louder = replace(scene, timeline=inner.replace_track(replace(audio, volume_db=6.0)))
        after = replace(before, scenes=(louder,))
        assert not changed_spans(before, after)

    def test_grouping_inside_a_scene_does_not_reach_the_outside(self) -> None:
        """束ね直しも同じ レンダラは束ねを読まない"""
        clip = _clip(0, 30)
        inner = Timeline(rate=RATE, tracks=(Track(kind=TrackKind.VIDEO, clips=(clip,)),))
        before, scene = self._with_scene(inner)
        regrouped = replace(
            scene,
            timeline=Timeline(
                rate=RATE,
                tracks=(replace(inner.tracks[0], clips=(replace(clip, group_id=new_group_id()),)),),
            ),
        )
        after = replace(before, scenes=(regrouped,))
        assert not changed_spans(before, after)


class TestWhatTheRendererDoesNotRead:
    """:class:`MediaItem` は絵に関わらないものも持っている

    丸ごと比べると、字幕を直すたびにその素材を使う所の先読みが全部消える
    字幕を焼き込む使い方（1 時間の動画に 2000 本）では、貯める意味がなくなる
    """

    def test_a_transcript_does_not_touch_the_picture(self) -> None:
        # 字幕を 1 本直すたびにその素材の先読みが消えると、字幕を直しながらの再生がその都度止まる
        media = _media()
        clip = _clip(0, 30, media_id=media.id)
        before = replace(_project(clip), media=(media,))
        after = replace(before, media=(media.with_transcript(Transcript(segments=())),))
        assert not changed_spans(before, after), "字幕を直しただけで先読みが消える"

    def test_a_display_name_does_not_touch_the_picture(self) -> None:
        # 表示名で先読みが消えると、素材の名前を付け直すたびにその素材の所が描き直しになる
        media = _media()
        clip = _clip(0, 30, media_id=media.id)
        before = replace(_project(clip), media=(media,))
        after = replace(before, media=(replace(media, display_name="別名"),))
        assert not changed_spans(before, after)

    def test_a_transcript_inside_a_scene_does_not_reach_the_outside(self) -> None:
        """シーンの中の素材でも同じ 外まで伝わると、まとめて消える"""
        media = _media()
        inner = Timeline(
            rate=RATE,
            tracks=(Track(kind=TrackKind.VIDEO, clips=(_clip(0, 30, media_id=media.id),)),),
        )
        scene = Scene(name="中身", timeline=inner)
        before = replace(
            _project(_clip(50, 30, scene_id=scene.id), scenes=(scene,)), media=(media,)
        )
        after = replace(before, media=(media.with_transcript(Transcript(segments=())),))
        assert not changed_spans(before, after)

    def test_gaining_sound_on_a_still_does_touch_the_picture(self) -> None:
        """長さ 0 の映像素材に音が付いたら、絵は変わる

        is_still は音の有無でも変わり、レンダラはこれを見て
        「常に先頭のコマを返す」へ切り替える 音だから絵に関係ない、とは言えない
        """
        still = replace(_media(), duration=Fraction(0))
        clip = _clip(0, 30, media_id=still.id)
        before = replace(_project(clip), media=(still,))
        sounding = replace(
            still,
            audio_streams=(
                AudioStreamInfo(
                    index=1,
                    sample_rate=48000,
                    channels=2,
                    time_base=Fraction(1, 48000),
                    codec="aac",
                ),
            ),
        )
        after = replace(before, media=(sounding,))
        assert changed_spans(before, after).contains(10)
