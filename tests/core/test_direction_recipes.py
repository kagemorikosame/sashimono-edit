"""演出のレシピとジャンルごとのひとそろい、強さの加減（#470 段 3 #478）"""

from __future__ import annotations

import itertools
from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.commands import AddMedia
from sashimono.core.direction import KINDS_BY_KEY, SHAKE, ZOOM, measure_direction
from sashimono.core.direction_recipes import (
    ACTION_SHAKE,
    ACTION_ZOOM,
    ACTIONS,
    BOLD,
    CUE_LAUGHTER,
    CUE_LOUD,
    CUES,
    DEFAULT_GENRE,
    DEFAULT_STRENGTH,
    GENRES,
    LETS_PLAY,
    NORMAL,
    RECIPE_SETS,
    STRENGTHS,
    SUBTLE,
    Cue,
    DirectionTaste,
    plan_direction,
    recipe_book,
)
from sashimono.core.emphasis import EMPHASIS_SHAKE_PX_RANGE, EMPHASIS_ZOOM_PERCENT_RANGE
from sashimono.core.model import Clip, MediaItem, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import seconds_to_frame
from sashimono.effects.emphasis import EmphasisRequest, emphasis_commands
from sashimono.engine.highlights import SIGNALS
from tests.conftest import RATE_30

_ORDER = (SUBTLE, NORMAL, BOLD)
#: 10 分の作品 1 分あたりの回数が端数に引っ張られない長さ
_SECONDS = 600


def _cues() -> list[Cue]:
    """笑いが 5 秒おき・大声が 7 秒おきにある作品 強さは散らして、強い順に選ぶ所を通す"""
    laughs = [
        Cue(CUE_LAUGHTER, float(t), float(t) + 1, (t * 37 % 100) / 100) for t in range(0, 600, 5)
    ]
    shouts = [Cue(CUE_LOUD, float(t), float(t) + 1, (t * 53 % 100) / 100) for t in range(3, 600, 7)]
    return laughs + shouts


class TestRecipes:
    def test_cues_are_the_highlight_signals(self) -> None:
        # 名前がずれると、盛り上がりの道具が返した区間にどのレシピも当たらない
        assert set(CUES) <= set(SIGNALS)
        for recipes in RECIPE_SETS.values():
            assert {recipe.cue for recipe in recipes.recipes} <= set(CUES)

    def test_every_genre_has_a_set_and_the_defaults_exist(self) -> None:
        # ジャンルを選べるのにひとそろいが無いと、設定を変えた所で落ちる
        assert set(RECIPE_SETS) == set(GENRES)
        assert (DEFAULT_GENRE, DEFAULT_STRENGTH) == (LETS_PLAY, SUBTLE)

    def test_stronger_never_means_fewer_or_weaker(self) -> None:
        # 派手を選んで控えめより少なくなると、強さの名前と結果が食い違う
        for recipes in RECIPE_SETS.values():
            for recipe in recipes.recipes:
                counts = [recipe.per_minute.get(level, 0.0) for level in _ORDER]
                assert counts == sorted(counts), recipe.key
                amounts = [recipe.amount[level] for level in _ORDER if level in recipe.amount]
                assert amounts == sorted(amounts), recipe.key

    def test_even_bold_keeps_the_too_close_line(self) -> None:
        # 段 1 のやりすぎの線より詰めると、付けた直後の measure_direction が too_close を返す
        for taste in (DirectionTaste(strength=level) for level in _ORDER):
            for rule in recipe_book(taste).rules:
                limit = KINDS_BY_KEY[rule.recipe.kind].min_gap_seconds or 0.0
                assert rule.min_gap_seconds >= limit
                assert rule.min_gap_seconds >= 60 / rule.per_minute

    def test_amounts_are_within_what_emphasize_accepts(self) -> None:
        # 道具が受け付けない値を渡すと、AI がレシピどおりに付けようとして断られる
        ranges = {ACTION_ZOOM: EMPHASIS_ZOOM_PERCENT_RANGE, ACTION_SHAKE: EMPHASIS_SHAKE_PX_RANGE}
        for recipes in RECIPE_SETS.values():
            for recipe in recipes.recipes:
                assert recipe.action in ACTIONS
                low, high = ranges.get(recipe.action, (0, 0))
                for amount in recipe.amount.values():
                    assert low <= amount <= high, recipe.key

    def test_unknown_tastes_are_refused(self) -> None:
        # 知らない値のまま進むと、レシピを引く所で KeyError になる
        with pytest.raises(ValueError, match="ジャンル"):
            DirectionTaste(genre="ドラマ")
        with pytest.raises(ValueError, match="強さ"):
            DirectionTaste(strength="最強")
        assert set(STRENGTHS) == set(_ORDER)

    def test_recipes_outside_the_guide_are_left_out(self) -> None:
        # 方針の外の効果を使う演出を付けると、照らしで本人に直させる
        open_book = recipe_book(DirectionTaste())
        assert open_book.left_out == ()
        strict = recipe_book(DirectionTaste(), allowed_effects=("glow",))
        assert [recipe.action for recipe, _ in strict.left_out] == [ACTION_SHAKE]
        assert ACTION_SHAKE not in [rule.recipe.action for rule in strict.rules]
        allowed = recipe_book(DirectionTaste(), allowed_effects=("random_move",))
        assert allowed.left_out == ()


class TestPlan:
    def test_the_same_cues_give_the_same_plan(self) -> None:
        # 同じ素材・同じ頼みで違う演出になると、試して比べられない
        book = recipe_book(DirectionTaste(strength=NORMAL))
        first = plan_direction(_cues(), book, 0, _SECONDS)
        assert first == plan_direction(list(reversed(_cues())), book, 0, _SECONDS)
        assert [item.start for item in first] == sorted(item.start for item in first)

    def test_the_count_and_gap_follow_the_strength(self) -> None:
        # 強さを上げても数が増えなければ、設定を変えた意味が無い
        counts = []
        for level in _ORDER:
            book = recipe_book(DirectionTaste(strength=level))
            plan = plan_direction(_cues(), book, 0, _SECONDS)
            for rule in book.rules:
                starts = [i.start for i in plan if i.recipe == rule.recipe.key]
                assert len(starts) <= round(rule.per_minute * _SECONDS / 60)
                kind_starts = sorted(i.start for i in plan if i.kind == rule.recipe.kind)
                gaps = [b - a for a, b in itertools.pairwise(kind_starts)]
                assert all(gap >= rule.min_gap_seconds for gap in gaps)
            counts.append(len(plan))
        assert counts[0] < counts[1] < counts[2]

    def test_strong_cues_are_chosen_first(self) -> None:
        # 弱い所から付けると、一番面白い所に演出が無い
        cues = [Cue(CUE_LAUGHTER, 10, 11, 0.2), Cue(CUE_LAUGHTER, 30, 31, 0.9)]
        plan = plan_direction(cues, recipe_book(DirectionTaste()), 0, 60)
        zooms = [item for item in plan if item.kind == ZOOM]
        assert [item.start for item in zooms] == [30]

    def test_short_ranges_and_empty_ranges(self) -> None:
        # 1 分に満たない範囲でも半分を越えれば 1 回は置ける 空の範囲は断る
        cues = [Cue(CUE_LAUGHTER, 5, 6, 0.5)]
        book = recipe_book(DirectionTaste())
        assert [i.kind for i in plan_direction(cues, book, 0, 40)] == [ZOOM]
        assert plan_direction(cues, book, 0, 20) == []
        with pytest.raises(ValueError, match="空"):
            plan_direction(cues, book, 10, 10)


@pytest.fixture
def long_work(video_media: MediaItem) -> Project:
    media = replace(video_media, duration=Fraction(_SECONDS))
    project = Project.create(ProjectSettings(frame_rate=RATE_30))
    project = AddMedia(media).apply(project)
    clip = Clip(timeline_start=0, duration=_SECONDS * 30, media_id=media.id)
    track = Track(kind=TrackKind.VIDEO, name="V1", clips=(clip,))
    return project.with_timeline(replace(project.timeline, tracks=(track,)))


def _applied(project: Project, level: str) -> tuple[Project, dict[str, int]]:
    """計画の寄ると揺らすを、AI の emphasize と同じ組み立てで付ける"""
    book = recipe_book(DirectionTaste(strength=level))
    plan = plan_direction(_cues(), book, 0, _SECONDS)
    rate = project.timeline.rate
    planned = {ZOOM: 0, SHAKE: 0}
    for item in plan:
        if item.kind not in planned:
            continue
        clip = project.timeline.tracks[0].clips[0]
        request = EmphasisRequest(
            kind=item.action,
            start=seconds_to_frame(Fraction(item.start), rate),
            end=seconds_to_frame(Fraction(item.end), rate),
            strength=float(item.amount or 0),
        )
        commands, _ = emphasis_commands(clip, request, rate)
        for command in commands:
            project = command.apply(project)
        planned[item.kind] += 1
    return project, planned


class TestMeasured:
    def test_measured_density_follows_the_strength(self, long_work: Project) -> None:
        # 付けた後に段 1 の道具で数えた回数が計画と食い違うと、強さを選んでも結果が読めない
        previous = {ZOOM: 0.0, SHAKE: 0.0}
        for level in _ORDER:
            project, planned = _applied(long_work, level)
            report = measure_direction(project, project.timeline)
            book = recipe_book(DirectionTaste(strength=level))
            limits = {rule.recipe.kind: rule.per_minute for rule in book.rules}
            for kind in (ZOOM, SHAKE):
                density = next(k for k in report.kinds if k.kind.key == kind)
                assert density.count == planned[kind]
                assert density.per_minute <= limits[kind]
                assert density.too_close == ()
                assert density.per_minute > previous[kind]
                previous[kind] = density.per_minute
