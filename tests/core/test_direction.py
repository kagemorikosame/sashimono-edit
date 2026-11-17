"""演出の型と密度の数え方（#470 段 1 #476）

数え違えると、AI が「やりすぎ」を見逃して寄るを 5 秒おきに付けたり、字幕や BGM を演出に数えて
足りない演出を「足りている」と読んだりする どの物を何回と数えるかを、作品を組んで確かめる
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core import direction
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND
from sashimono.core.direction import (
    COLOR,
    CUT,
    DIRECTION_KINDS,
    SHAKE,
    SHAKE_DEFAULTS,
    SOUND,
    TELOP,
    TRANSITION,
    ZOOM,
    DirectionReport,
    KindDensity,
    direction_events,
    measure_direction,
)
from sashimono.core.emphasis import easing_point
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Interpolation,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    SegmentId,
    SubtitleOrigin,
    Track,
    TrackKind,
    new_media_id,
)
from tests.conftest import RATE_30

#: 色に数えるエフェクトの種類 呼び手（AI の層）は定義の分類から作るが、ここは名前だけで足りる
COLOR_KINDS = frozenset({"grade"})


def _curve(*points: tuple[int, float]) -> AnimatedValue:
    return AnimatedValue(keyframes=tuple(Keyframe(frame, value) for frame, value in points))


def _transform(**params: AnimatedValue) -> Effect:
    return Effect(TRANSFORM_EFFECT_KIND, params=dict(params), fixed=True)


def _audio(path: str, seconds: int, audio_media: MediaItem) -> MediaItem:
    return replace(audio_media, path=Path(path), duration=Fraction(seconds), id=new_media_id())


class _Work:
    """映像 1 本・テキスト・音を置く作品 素材はどれも 60 秒（30fps で 1800 フレーム）まで置ける"""

    def __init__(self, video_media: MediaItem, audio_media: MediaItem) -> None:
        self.video_media = replace(video_media, duration=Fraction(60))
        self.effect = _audio("C:/素材/ぽん.wav", 1, audio_media)
        self.bgm = _audio("C:/素材/bgm.wav", 60, audio_media)
        self.voice = _audio("C:/作品/narration/001.wav", 3, audio_media)
        media = (self.video_media, self.effect, self.bgm, self.voice)
        self.project = Project.create(ProjectSettings(frame_rate=RATE_30), media=media)
        self.video = Track(kind=TrackKind.VIDEO, name="V1")
        self.text = Track(kind=TrackKind.VIDEO, name="文字")
        self.sound = Track(kind=TrackKind.AUDIO, name="A1")

    def clip(self, start: int, duration: int, **more: object) -> Clip:
        return Clip(timeline_start=start, duration=duration, media_id=self.video_media.id, **more)  # type: ignore[arg-type]  # 試験の手短な組み立て 欄の名前と型は Clip が確かめる

    def build(
        self,
        video: tuple[Clip, ...] = (),
        text: tuple[Clip, ...] = (),
        sound: tuple[Clip, ...] = (),
    ) -> Project:
        tracks = (
            replace(self.video, clips=video),
            replace(self.text, clips=text),
            replace(self.sound, clips=sound),
        )
        return self.project.with_timeline(replace(self.project.timeline, tracks=tracks))


@pytest.fixture
def work(video_media: MediaItem, audio_media: MediaItem) -> _Work:
    return _Work(video_media, audio_media)


def _kind(report: DirectionReport, key: str) -> KindDensity:
    return next(density for density in report.kinds if density.kind.key == key)


def _measure(project: Project, **more: object) -> DirectionReport:
    return measure_direction(project, project.timeline, color_kinds=COLOR_KINDS, **more)  # type: ignore[arg-type]  # 範囲と下限だけを渡し分ける


class TestZoom:
    def test_zoom_in_and_back_is_one_event(self, work: _Work) -> None:
        # 寄って・止まって・戻すの 4 点を点ごとに数えると、強調 1 回が 3 回に見え、
        # 1 分あたりの回数が 3 倍に出て「やりすぎ」と読み違える
        scale = _curve((30, 100), (38, 120), (90, 120), (98, 100))
        clip = work.clip(0, 600, effects=(_transform(scale=scale),))
        events = direction_events(work.build(video=(clip,)), work.build(video=(clip,)).timeline)
        zooms = [e for e in events if e.kind == ZOOM]
        assert [(e.start_frame, e.end_frame) for e in zooms] == [(30, 98)]
        assert zooms[0].strength == pytest.approx(120)
        assert not zooms[0].pan_only

    def test_two_zooms_with_a_rest_between_are_two(self, work: _Work) -> None:
        # 休みの値へ戻った所で区切らないと、離れた 2 回の強調を 1 回と数え、近すぎるのを見逃す
        scale = _curve((0, 100), (10, 120), (20, 100), (150, 100), (160, 130), (170, 100))
        clip = work.clip(100, 600, effects=(_transform(scale=scale),))
        project = work.build(video=(clip,))
        zooms = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert [(e.start_frame, e.strength) for e in zooms] == [(100, 120), (250, 130)]

    def test_pan_alone_is_marked_and_has_no_scale(self, work: _Work) -> None:
        # 位置だけ動いた物に拡大率 100% を返すと、寄らなかったのに「寄った（強さ 100）」と読める
        moved = _curve((0, 0), (10, 120), (40, 120), (50, 0))
        clip = work.clip(0, 300, effects=(_transform(pos_x=moved),))
        project = work.build(video=(clip,))
        (zoom,) = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert zoom.pan_only
        assert zoom.strength is None

    def test_moving_text_is_not_a_zoom(self, work: _Work) -> None:
        # テロップの登場の動きを寄るに数えると、文字を出すたびに寄るが増えて「やりすぎ」と出る
        moved = _curve((0, 0), (10, 120))
        source = GeneratedSource("text", params={"text": "要点"})
        text = Clip(
            timeline_start=0, duration=60, source=source, effects=(_transform(pos_x=moved),)
        )
        project = work.build(text=(text,))
        kinds = [e.kind for e in direction_events(project, project.timeline)]
        assert kinds == [TELOP]

    def test_returning_to_rest_splits_even_without_a_flat_stretch(self, work: _Work) -> None:
        # 戻った点を挟んでつなぐと、寄って戻ってすぐまた寄る 2 回を 1 回と読み、近すぎるのを見逃す
        scale = _curve((0, 100), (10, 120), (20, 100), (30, 130), (40, 100))
        project = work.build(video=(work.clip(0, 300, effects=(_transform(scale=scale),)),))
        zooms = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert [(e.start_frame, e.end_frame) for e in zooms] == [(0, 20), (20, 40)]

    def test_a_kept_zoom_lasts_to_the_end_of_the_clip(self, work: _Work) -> None:
        # 最後の点より後も寄ったまま映る ここで切ると、寄りっぱなしの長さを短く読む
        scale = _curve((30, 100), (40, 150))
        project = work.build(video=(work.clip(0, 300, effects=(_transform(scale=scale),)),))
        (zoom,) = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert (zoom.start_frame, zoom.end_frame, zoom.strength) == (30, 300, 150)

    def test_points_left_outside_a_trimmed_clip_are_ignored(self, work: _Work) -> None:
        # トリムで外に残った点の動きを数えると、映らない演出が回数と近すぎる所に入る
        scale = _curve((0, 100), (150, 100), (160, 120), (170, 100))
        project = work.build(video=(work.clip(0, 100, effects=(_transform(scale=scale),)),))
        assert not [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]

    def test_scaling_one_axis_is_a_zoom(self, work: _Work) -> None:
        # 縦横別の拡大率だけを動かす作品もある 全体の拡大率だけを見ると寄りを 0 回と読む
        scale_x = _curve((0, 100), (10, 125), (20, 100))
        clip = work.clip(0, 300, effects=(_transform(scale_x=scale_x),))
        project = work.build(video=(clip,))
        (zoom,) = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert zoom.strength == pytest.approx(125)

    def test_overshoot_between_points_is_the_strength(self, work: _Work) -> None:
        # 行き過ぎて戻る曲線は点の間に山がある 点の値だけを見ると強さを小さく読む
        interpolation, curve = easing_point("back")
        scale = AnimatedValue(
            keyframes=(
                Keyframe(0, 100.0),
                Keyframe(30, 100.0, interpolation, curve=curve),
                Keyframe(60, 120.0),
                Keyframe(90, 100.0),
            )
        )
        project = work.build(video=(work.clip(0, 300, effects=(_transform(scale=scale),)),))
        (zoom,) = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert zoom.strength is not None
        assert zoom.strength > 120.5

    def test_a_one_frame_peak_in_a_long_clip_is_the_strength(self, work: _Work) -> None:
        # 長いクリップを同じ刻みで間引くと、キー 1 つだけの山を飛ばして強さを小さく読む
        scale = _curve((0, 100), (30, 120), (31, 180), (32, 120))
        project = work.build(video=(work.clip(0, 6000, effects=(_transform(scale=scale),)),))
        (zoom,) = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert zoom.strength == pytest.approx(180)

    def test_a_hold_switch_starts_the_zoom_at_the_key(self, work: _Work) -> None:
        # 右端の値で前の間を採ると、30 で切り替わる寄りを 0 からと読み、頭を 30 にすると数え落とす
        scale = AnimatedValue(
            keyframes=(
                Keyframe(0, 100.0, Interpolation.HOLD),
                Keyframe(30, 120.0, Interpolation.HOLD),
                Keyframe(60, 100.0, Interpolation.HOLD),
            )
        )
        project = work.build(video=(work.clip(0, 300, effects=(_transform(scale=scale),)),))
        zooms = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert [(e.start_frame, e.end_frame) for e in zooms] == [(30, 60)]

    def test_a_zoom_that_starts_late_in_the_span_is_counted(self, work: _Work) -> None:
        # 左端と真ん中だけを見ると、間の後ろ半分で動き出す寄りを「動いていない」と読み数え落とす
        scale = AnimatedValue(
            keyframes=(
                Keyframe(0, 100.0, Interpolation.BEZIER, control_points=(0.0, 0.0, 1.0, -1 / 3)),
                Keyframe(60, 120.0, Interpolation.HOLD),
                Keyframe(90, 100.0, Interpolation.HOLD),
            )
        )
        project = work.build(video=(work.clip(0, 300, effects=(_transform(scale=scale),)),))
        zooms = [e for e in direction_events(project, project.timeline) if e.kind == ZOOM]
        assert [(e.start_frame, e.end_frame) for e in zooms] == [(0, 90)]


class TestShake:
    def test_keyframed_width_counts_each_burst(self, work: _Work) -> None:
        # 幅が 0 の間も揺れに数えると、強調 2 回がクリップ全体の 1 回にまとまる
        width = _curve((0, 0), (5, 12), (15, 12), (20, 0), (200, 0), (205, 30), (210, 0))
        shake = Effect("random_move", params={"range_x": width, "range_y": width})
        project = work.build(video=(work.clip(0, 600, effects=(shake,)),))
        shakes = [e for e in direction_events(project, project.timeline) if e.kind == SHAKE]
        assert [(e.start_frame, e.strength) for e in shakes] == [(0, 12), (200, 30)]

    def test_still_width_is_one_event_for_the_clip(self, work: _Work) -> None:
        # 幅が動かないランダム移動はクリップの間ずっと揺れる
        # 数えないと揺れ続ける作品を「揺れ 0」と読む
        shake = Effect(
            "random_move", params={"range_x": AnimatedValue(8.0), "range_y": AnimatedValue(0.0)}
        )
        project = work.build(video=(work.clip(60, 90, effects=(shake,)),))
        (event,) = [e for e in direction_events(project, project.timeline) if e.kind == SHAKE]
        assert (event.start_frame, event.end_frame, event.strength) == (60, 150, 8)

    def test_zero_width_and_disabled_effects_do_not_count(self, work: _Work) -> None:
        # 切ったエフェクトや幅 0 を数えると、人が外した揺れがまだ残って見える
        off = Effect("random_move", params={"range_x": AnimatedValue(10.0)}, enabled=False)
        flat = Effect(
            "random_move", params={"range_x": AnimatedValue(0.0), "range_y": AnimatedValue(0.0)}
        )
        project = work.build(video=(work.clip(0, 90, effects=(off, flat)),))
        assert not [e for e in direction_events(project, project.timeline) if e.kind == SHAKE]

    def test_a_still_axis_keeps_shaking_while_another_is_keyed(self, work: _Work) -> None:
        # 動く軸だけを見ると、横はずっと 8px 揺れているのに縦の線が 0 なら「揺れ 0」と読む
        flat = _curve((0, 0), (100, 0))
        shake = Effect("random_move", params={"range_x": AnimatedValue(8.0), "range_y": flat})
        project = work.build(video=(work.clip(0, 300, effects=(shake,)),))
        (event,) = [e for e in direction_events(project, project.timeline) if e.kind == SHAKE]
        assert (event.start_frame, event.end_frame, event.strength) == (0, 300, 8)

    def test_depth_and_default_widths_count(self, work: _Work) -> None:
        # 奥行きだけの揺れや、項目を省いた（定義の既定 10px の）揺れを数えないと揺れを見逃す
        depth = Effect(
            "random_move",
            params={
                "range_x": AnimatedValue(0.0),
                "range_y": AnimatedValue(0.0),
                "range_z": AnimatedValue(5.0),
            },
        )
        plain = Effect("random_move")
        project = work.build(
            video=(work.clip(0, 90, effects=(depth,)), work.clip(90, 90, effects=(plain,)))
        )
        shakes = [e for e in direction_events(project, project.timeline) if e.kind == SHAKE]
        assert [(e.start_frame, e.strength) for e in shakes] == [(0, 5), (90, 10)]


class TestOtherKinds:
    def test_subtitles_are_not_telops(self, work: _Work) -> None:
        # 字幕をテロップに数えると、1 分あたりのテロップが字幕の行数で埋まり、
        # 要点の文字の加減が読めない
        origin = SubtitleOrigin(media_id=work.video_media.id, stream=0, segment_id=SegmentId("s1"))
        subtitle = Clip(
            timeline_start=0,
            duration=30,
            source=GeneratedSource("text", params={"text": "字幕"}),
            subtitle_origin=origin,
        )
        telop = Clip(timeline_start=60, duration=30, source=GeneratedSource("text"))
        project = work.build(text=(subtitle, telop))
        telops = [e for e in direction_events(project, project.timeline) if e.kind == TELOP]
        assert [e.start_frame for e in telops] == [60]

    def test_a_graded_telop_counts_as_color_but_a_subtitle_does_not(self, work: _Work) -> None:
        # テロップの色を数え落とすと、色を整えた作品を色 0 回と読む 字幕まで数えると、
        # 字幕の行数で色の回数が埋まる
        origin = SubtitleOrigin(media_id=work.video_media.id, stream=0, segment_id=SegmentId("s1"))
        subtitle = Clip(
            timeline_start=0,
            duration=30,
            source=GeneratedSource("text", params={"text": "字幕"}),
            subtitle_origin=origin,
            effects=(Effect("grade"),),
        )
        telop = Clip(
            timeline_start=60,
            duration=30,
            source=GeneratedSource("text"),
            effects=(Effect("grade"),),
        )
        project = work.build(text=(subtitle, telop))
        events = direction_events(project, project.timeline, color_kinds=COLOR_KINDS)
        assert [(e.kind, e.start_frame) for e in events if e.kind in (TELOP, COLOR)] == [
            (COLOR, 60),
            (TELOP, 60),
        ]

    def test_only_short_sound_only_clips_are_sound_effects(self, work: _Work) -> None:
        # 映像の音・BGM・ナレーションを効果音に数えると、話す間ずっと効果音が鳴っているように見える
        sounds = (
            Clip(timeline_start=0, duration=30, media_id=work.effect.id),
            Clip(timeline_start=30, duration=1800, media_id=work.bgm.id),
            Clip(timeline_start=1830, duration=90, media_id=work.voice.id),
            Clip(timeline_start=1920, duration=60, media_id=work.video_media.id),
        )
        project = work.build(sound=sounds)
        found = [e for e in direction_events(project, project.timeline) if e.kind == SOUND]
        assert [e.clip_id for e in found] == [str(sounds[0].id)]

    def test_transition_reports_its_length_in_seconds(self, work: _Work) -> None:
        # 長さを返さないと、0.2 秒の切り替えと 2 秒の溶けを同じ強さと読む
        transition = Clip(timeline_start=285, duration=15, source=GeneratedSource("transition"))
        project = work.build(text=(transition,))
        (event,) = direction_events(project, project.timeline)
        assert (event.kind, event.strength) == (TRANSITION, 0.5)

    def test_color_counts_given_kinds_and_its_coverage(self, work: _Work) -> None:
        # 覆う割合を返さないと、1 本だけ色を整えた作品と全部整えた作品の区別が付かない
        graded = work.clip(0, 300, effects=(Effect("grade"),))
        plain = work.clip(300, 300, effects=(Effect("grade", enabled=False),))
        report = _measure(work.build(video=(graded, plain)))
        color = _kind(report, COLOR)
        assert color.count == 1
        assert color.coverage == pytest.approx(0.5)

    def test_disabled_clips_are_not_counted(self, work: _Work) -> None:
        # 切ったクリップは書き出しに出ない 数えると外した演出が残って見える
        telop = Clip(timeline_start=0, duration=30, source=GeneratedSource("text"), enabled=False)
        project = work.build(text=(telop,))
        assert direction_events(project, project.timeline) == []

    def test_muted_and_unsoloed_tracks_are_not_counted(self, work: _Work) -> None:
        # 書き出しに出ないトラックを数えると、外したつもりの演出が残って見える
        telop = Clip(timeline_start=0, duration=30, source=GeneratedSource("text"))
        sound = Clip(timeline_start=0, duration=30, media_id=work.effect.id)
        project = work.build(video=(work.clip(0, 300),), text=(telop,), sound=(sound,))
        video, text, audio = project.timeline.tracks
        muted = (video, replace(text, muted=True), replace(audio, muted=True))
        quiet = project.with_timeline(replace(project.timeline, tracks=muted))
        assert [e.kind for e in direction_events(quiet, quiet.timeline)] == [CUT]
        soloed = (replace(video, solo=True), text, audio)
        alone = project.with_timeline(replace(project.timeline, tracks=soloed))
        assert TELOP not in [e.kind for e in direction_events(alone, alone.timeline)]

    def test_filters_count_their_color(self, work: _Work) -> None:
        # フィルタで全体の色を整えた作品を「色 0」と読むと、AI が同じ補正をクリップごとに積む
        graded = Clip(
            timeline_start=0,
            duration=300,
            source=GeneratedSource("filter"),
            effects=(Effect("grade"),),
        )
        project = work.build(video=(work.clip(0, 300),), text=(graded,))
        assert _kind(_measure(project), COLOR).coverage == pytest.approx(1.0)

    def test_a_filter_with_its_picture_off_is_not_counted(self, work: _Work) -> None:
        # 混合トラックで絵を切ったフィルタは描画が外す 数えると出ない色が割合に入る
        hidden = Clip(
            timeline_start=0,
            duration=300,
            source=GeneratedSource("filter"),
            effects=(Effect("grade"),),
            show_picture=False,
        )
        project = work.build(video=(work.clip(0, 300),))
        mixed = Track(kind=TrackKind.MIXED, name="混合", clips=(hidden,))
        tracks = (*project.timeline.tracks, mixed)
        project = project.with_timeline(replace(project.timeline, tracks=tracks))
        color = _kind(_measure(project), COLOR)
        assert (color.count, color.coverage) == (0, 0)


class TestDensity:
    def test_close_zooms_are_reported_against_the_limit(self, work: _Work) -> None:
        # 寄るのが 5 秒おき（#470 の例）を見逃すと、AI が付けすぎたまま仕上げる
        points: list[tuple[int, float]] = []
        for start in (0, 150, 600):
            points += [(start, 100), (start + 10, 120), (start + 20, 100)]
        clip = work.clip(0, 1800, effects=(_transform(scale=_curve(*points)),))
        report = _measure(work.build(video=(clip,)))
        zoom = _kind(report, ZOOM)
        assert (zoom.count, zoom.per_minute) == (3, 3.0)
        assert zoom.min_gap_seconds == 5.0
        assert [(c.first_frame, c.second_frame) for c in zoom.too_close] == [(0, 150)]
        loose = _kind(_measure(work.build(video=(clip,)), min_gaps={ZOOM: 4}), ZOOM)
        assert loose.too_close == ()

    def test_cuts_measure_shot_lengths_and_skip_the_head(self, work: _Work) -> None:
        # 頭の 0 フレームをカットに数えると、カットの無い 1 本の作品にも 1 回のカットが出る
        clips = (work.clip(0, 300), work.clip(300, 600), work.clip(900, 15))
        report = _measure(work.build(video=clips))
        cut = _kind(report, CUT)
        assert cut.count == 2
        # 長さは 10 秒・20 秒・0.5 秒 0.5 秒は下限（1 秒）より短い
        assert cut.min_gap_seconds == 0.5
        assert cut.median_gap_seconds == 10.0
        assert [(c.first_frame, c.second_frame) for c in cut.too_close] == [(900, 915)]

    def test_range_counts_only_events_that_start_inside(self, work: _Work) -> None:
        # 範囲の外を数えると、一部を測ったつもりで全体の数が出て、1 分あたりの回数も合わない
        telops = tuple(
            Clip(timeline_start=frame, duration=30, source=GeneratedSource("text"))
            for frame in (0, 400, 800, 1200)
        )
        project = work.build(video=(work.clip(0, 1800),), text=telops)
        report = _measure(project, start_frame=300, end_frame=900)
        telop = _kind(report, TELOP)
        assert (report.seconds, telop.count, telop.per_minute) == (20.0, 2, 6.0)

    def test_color_coverage_counts_what_started_before_the_range(self, work: _Work) -> None:
        # 頭の位置だけで選ぶと、全体を整えた作品の後半を測ったときに覆う割合が 0 と出る
        graded = work.clip(0, 600, effects=(Effect("grade"),))
        color = _kind(_measure(work.build(video=(graded,)), start_frame=300), COLOR)
        assert (color.count, color.coverage) == (0, 1.0)

    def test_stacked_cuts_are_one_in_the_list_too(self, work: _Work) -> None:
        # 回数は 1 なのに一覧に 2 行あると、AI は一覧を数えて回数を読み違える
        project = work.build(video=(work.clip(0, 300), work.clip(300, 300)))
        upper = Track(kind=TrackKind.VIDEO, name="V2", clips=(work.clip(300, 60),))
        tracks = (*project.timeline.tracks, upper)
        stacked = project.with_timeline(replace(project.timeline, tracks=tracks))
        report = _measure(stacked)
        cuts = [e for e in report.events if e.kind == CUT]
        assert (_kind(report, CUT).count, len(cuts)) == (1, 1)

    def test_sound_only_range_has_no_shot_lengths(self, work: _Work) -> None:
        # 絵の無い範囲をカットの長さとして測ると、無いカットを短すぎると返す
        sound = Clip(timeline_start=0, duration=15, media_id=work.effect.id)
        cut = _kind(_measure(work.build(sound=(sound,))), CUT)
        assert (cut.min_gap_seconds, cut.too_close) == (None, ())

    def test_bad_requests_are_refused(self, work: _Work) -> None:
        # 黙って全体を測ったり知らない型を捨てたりすると、頼んだ物と違う数を正しいと思って使う
        project = work.build(video=(work.clip(0, 300),))
        with pytest.raises(ValueError, match="範囲が空"):
            _measure(project, start_frame=300, end_frame=300)
        with pytest.raises(ValueError, match="知らない型"):
            _measure(project, min_gaps={"bounce": 3})


def test_names_match_the_layers_that_make_them() -> None:
    """コア層は名前だけを写して持つ 作る側が名前を変えたのに写しが古いと、黙って 0 回と数える"""
    from sashimono.effects import registry
    from sashimono.effects.emphasis import SHAKE_EFFECT_KIND
    from sashimono.effects.spec import TrackSpec
    from sashimono.engine.audio.voicevox import NARRATION_FOLDER

    assert direction.SHAKE_EFFECT_KIND == SHAKE_EFFECT_KIND
    assert direction.NARRATION_FOLDER == NARRATION_FOLDER
    assert len({kind.key for kind in DIRECTION_KINDS}) == len(DIRECTION_KINDS)
    definition = registry.get(SHAKE_EFFECT_KIND)
    assert definition is not None
    defaults = {
        spec.name: spec.default
        for spec in definition.parameters
        if spec.name in SHAKE_DEFAULTS and isinstance(spec, TrackSpec)
    }
    assert defaults == dict(SHAKE_DEFAULTS)
