"""過去の作品から方針の下書きを作る（#470 段 2 #477）"""

from __future__ import annotations

import os
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.ai.style_check import style_problems
from sashimono.ai.style_draft import draft_guide, measure_habits, read_habits
from sashimono.core.commands import new_scene
from sashimono.core.io.serialize import ProjectFileError, save_project
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    SegmentId,
    Stroke,
    SubtitleOrigin,
    Track,
    TrackKind,
)
from sashimono.core.model.style import STYLE_ITEMS_LIMIT
from tests.conftest import RATE_30

_WHITE = (1.0, 1.0, 1.0, 1.0)
_YELLOW = (1.0, 1.0, 0.0, 1.0)
_BLACK = (0.0, 0.0, 0.0, 1.0)


def _text(start: int, font: str, color: tuple[float, ...], **more: object) -> Clip:
    source = GeneratedSource(
        "text",
        params={"text": "秘密の本文", "font": font, "color": color, "size": AnimatedValue(48.0)},
        strokes=(Stroke(params={"color": _BLACK}),),
    )
    return Clip(timeline_start=start, duration=30, source=source, **more)  # type: ignore[arg-type]  # 試験の手短な組み立て


def _subtitle(start: int) -> Clip:
    origin = SubtitleOrigin(media_id=None, stream=0, segment_id=SegmentId(f"s{start}"))
    return _text(start, "字幕の書体", _WHITE, subtitle_origin=origin)


def _zoom() -> Effect:
    points = ((0, 100.0), (10, 120.0), (20, 100.0))
    scale = AnimatedValue(keyframes=tuple(Keyframe(frame, value) for frame, value in points))
    return Effect("transform", params={"scale": scale}, fixed=True)


@pytest.fixture
def work(video_media: MediaItem) -> Project:
    """60 秒の作品 カット 4 本（10・10・10・30 秒） 寄り 2 回 テロップ 3 本と字幕 2 本"""
    media = replace(video_media, duration=Fraction(60))
    project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(media,))
    shots = tuple(
        Clip(
            timeline_start=start,
            duration=length,
            media_id=media.id,
            effects=(_zoom(),) if start in (0, 300) else (),
        )
        for start, length in ((0, 300), (300, 300), (600, 300), (900, 900))
    )
    telops = (
        _text(0, "見出しの書体", _YELLOW, effects=(Effect("glow"),)),
        _text(400, "見出しの書体", _YELLOW),
        _text(800, "注釈の書体", _WHITE),
        _text(1000, "使っていない書体", _YELLOW, enabled=False),
    )
    tracks = (
        Track(kind=TrackKind.VIDEO, name="V1", clips=shots),
        Track(kind=TrackKind.VIDEO, name="文字", clips=telops),
        Track(kind=TrackKind.VIDEO, name="字幕", clips=(_subtitle(100), _subtitle(500))),
    )
    return project.with_timeline(replace(project.timeline, tracks=tracks))


class TestHabits:
    def test_fonts_colors_and_effects_are_ranked(self, work: Project) -> None:
        # 多い順に並べないと、本人がいつも使う書体と 1 回だけ使った書体の見分けが付かない
        habits = measure_habits(work)
        assert [(t.value, t.count) for t in habits.fonts] == [
            ("見出しの書体", 2),
            ("字幕の書体", 2),
            ("注釈の書体", 1),
        ]
        assert [t.value for t in habits.colors] == ["#000000", "#ffffff", "#ffff00"]
        assert [(t.value, t.count) for t in habits.effects] == [("glow", 1)]

    def test_disabled_clips_are_not_habits(self, work: Project) -> None:
        # 本人が切ったクリップは絵に出ない 癖に数えると使っていない書体が方針に入る
        assert "使っていない書体" not in [t.value for t in measure_habits(work).fonts]

    def test_density_and_shot_lengths(self, work: Project) -> None:
        # カットの長さと密度が下書きの文の元 外れると方針のテンポが本人の作品と食い違う
        habits = measure_habits(work)
        assert habits.seconds == pytest.approx(60)
        assert habits.shot_seconds == pytest.approx((10, 10, 15))
        assert habits.per_minute["zoom"] == pytest.approx(2)
        assert habits.per_minute["telop"] == pytest.approx(3)

    def test_the_most_used_subtitle_look(self, work: Project) -> None:
        # 字幕の見た目は文の欄に入る 書体・大きさ・色・縁取りがそろわないと本人が直す所が増える
        look = measure_habits(work).subtitle
        assert look is not None
        assert (look.font, look.size, look.color, look.outline) == (
            "字幕の書体",
            48,
            "#ffffff",
            "#000000",
        )
        assert (look.count, look.total) == (2, 2)

    def test_scenes_count_for_styles(self, work: Project) -> None:
        # シーンの中のテロップの書体を数えないと、確定した方針で照らしたときにシーンの中が外れる
        scene = new_scene(work, "オープニング")
        track = Track(kind=TrackKind.VIDEO, name="文字", clips=(_text(0, "場面の書体", _WHITE),))
        scene = replace(scene, timeline=replace(scene.timeline, tracks=(track,)))
        habits = measure_habits(replace(work, scenes=(scene,)))
        assert "場面の書体" in [t.value for t in habits.fonts]

    def test_a_work_without_footage_has_no_shot_lengths(self, work: Project) -> None:
        # 素材の絵が無い作品にカットの長さを書くと、無いカットの長さが方針に入る
        tracks = work.timeline.tracks[1:]
        words = work.with_timeline(replace(work.timeline, tracks=tracks))
        assert measure_habits(words).shot_seconds is None

    def test_an_empty_work_is_refused(self, video_media: MediaItem) -> None:
        # 長さ 0 の作品は 1 分あたりの回数が決まらない 0 で割って落ちる前に断る
        empty = Project.create(ProjectSettings(frame_rate=RATE_30), media=(video_media,))
        with pytest.raises(ValueError, match="空"):
            measure_habits(empty)


class TestDraft:
    def test_a_confirmed_draft_finds_nothing_outside(self, work: Project) -> None:
        # 下書きのまま確定して照らすと何も外れないこと 外れると本人の作品を本人の方針が咎める
        guide = draft_guide(measure_habits(work))
        assert style_problems(replace(work, style_guide=guide)) == []

    def test_the_text_has_numbers_but_no_body_text(self, work: Project) -> None:
        # 方針は AI へ送る テキストの本文が入ると、字幕と文字を送らないと決めた人の文が漏れる
        guide = draft_guide(measure_habits(work))
        assert "1 カットは中央 10 秒" in guide.tempo
        assert "寄る" in guide.tempo
        assert guide.subtitles.startswith("書体 字幕の書体・大きさ 48px・字の色 #ffffff")
        written = "\n".join((guide.tempo, guide.subtitles, guide.notes))
        assert "秘密の本文" not in written
        assert "。" not in written
        assert guide.problems() == []

    def test_too_many_fonts_keep_the_most_used(self, work: Project) -> None:
        # 上限を越えたまま渡すと窓の OK が押せない 多い順に残せば本人がいつも使う書体が残る
        count = STYLE_ITEMS_LIMIT + 5
        many = tuple(_text(n * 30, f"書体{n:02d}", _WHITE) for n in range(count))
        tracks = (*work.timeline.tracks, Track(kind=TrackKind.VIDEO, name="多い", clips=many))
        crowded = work.with_timeline(replace(work.timeline, tracks=tracks))
        guide = draft_guide(measure_habits(crowded))
        assert len(guide.fonts) == STYLE_ITEMS_LIMIT
        assert guide.fonts[0] == "見出しの書体"

    def test_effects_on_subtitles_are_written(self, work: Project) -> None:
        # 縁取りを効果で付けた字幕を「縁取りなし」と書くと、AI が縁の無い字幕を置く
        origin = SubtitleOrigin(media_id=None, stream=0, segment_id=SegmentId("b"))
        plain = _text(1200, "字幕の書体", _WHITE, subtitle_origin=origin)
        assert plain.source is not None
        source = replace(plain.source, strokes=())
        bordered = replace(plain, source=source, effects=(Effect("border"),))
        track = Track(kind=TrackKind.VIDEO, name="縁", clips=(bordered,))
        alone = work.with_timeline(replace(work.timeline, tracks=(work.timeline.tracks[0], track)))
        guide = draft_guide(measure_habits(alone))
        assert "縁取りなし" not in guide.subtitles
        assert "（border）" in guide.subtitles


class TestRead:
    def test_reading_does_not_touch_the_file(self, work: Project, tmp_path: Path) -> None:
        # 本人の仕上げた作品 更新時刻が変わると、どれをいつ直したかの手がかりが消える
        path = tmp_path / "作品.sme"
        save_project(work, path)
        os.utime(path, (1_000_000_000, 1_000_000_000))
        before = path.read_bytes()
        habits = read_habits(path)
        assert habits.fonts[0].value == "見出しの書体"
        assert path.read_bytes() == before
        assert path.stat().st_mtime == 1_000_000_000
        assert sorted(p.name for p in tmp_path.iterdir()) == ["作品.sme"]

    def test_a_broken_file_is_refused(self, tmp_path: Path) -> None:
        # 読めないファイルで落ちると、方針の窓ごと閉じて書きかけの方針が消える
        path = tmp_path / "壊れた.sme"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(ProjectFileError):
            read_habits(path)
