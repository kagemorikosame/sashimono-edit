"""書体に無い字を知らせ、代わりの字の太さをそろえる（#494）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from PySide6.QtGui import QFontDatabase, QGuiApplication

from sashimono.ai.fixes import HANDOFF, fix_for
from sashimono.core.model import Clip, Project, ProjectSettings, Scene, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.checks import KIND_NAMES
from sashimono.engine.checks.glyphs import glyph_runs
from sashimono.engine.checks.report import MEDIUM
from sashimono.engine.checks.scan import MISSING_GLYPH
from sashimono.engine.sources import missing_glyphs, render_source

#: 日本語の字を持たない、Windows に入っている書体
LATIN_ONLY = "Bahnschrift"


@pytest.fixture(autouse=True)
def _latin_only(qt_application: QGuiApplication) -> None:
    if LATIN_ONLY not in QFontDatabase.families():
        pytest.skip(f"{LATIN_ONLY} の無い機械")


def _project(*texts: str) -> Project:
    clips = tuple(
        Clip(
            timeline_start=index * 30,
            duration=30,
            source=TEXT.create(text=text, font=LATIN_ONLY),
        )
        for index, text in enumerate(texts)
    )
    project = Project.create(ProjectSettings())
    track = Track(kind=TrackKind.VIDEO, name="V1", clips=clips)
    return project.with_timeline(replace(project.timeline, tracks=(track,)))


def _inked(text: str, *, matched: bool) -> np.ndarray:
    source = TEXT.create(text=text, font=LATIN_ONLY, size=96.0, match_fallback=matched)
    return np.asarray(render_source(source, 800, 200))[..., 3].astype(np.float64)


class TestFinding:
    def test_only_the_absent_characters(self) -> None:
        """ある字まで返すと知らせが字幕の全部に出て読まれず、無い字を逃すと気付かず書き出す"""
        source = TEXT.create(text="AB 字幕 CD\n字", font=LATIN_ONLY)
        assert missing_glyphs(source) == "字幕"

    def test_spaces_and_emoji_are_not_counted(self) -> None:
        """絵文字は代わりの書体で出るのがふつう 数えると絵文字の字幕が全部問題になる"""
        assert missing_glyphs(TEXT.create(text="AB 12\n😀", font=LATIN_ONLY)) == ""

    def test_the_check_returns_the_clip(self) -> None:
        """壊れると、AI も人もどのテキストの字が化けたか分からない"""
        project = _project("ABC", "字幕")
        (run,) = glyph_runs(project, 0, 60)
        assert run.kind == MISSING_GLYPH
        assert (run.start, run.end) == (30, 60)
        assert run.values["characters"] == "字幕"
        assert run.values["matched"] is False
        assert MISSING_GLYPH in KIND_NAMES

    def test_outside_the_range_is_not_returned(self) -> None:
        """範囲を絞って調べたのに外の問題を返すと、頼まれていない所を AI が直す"""
        assert glyph_runs(_project("ABC", "字幕"), 0, 30) == []

    def test_supplementary_kanji_are_counted(self) -> None:
        """絵文字と一緒に外すと、人名の「𠮷」が代わりの書体で出ても知らせない"""
        assert missing_glyphs(TEXT.create(text="𠮷😀", font=LATIN_ONLY)) == "𠮷"

    def test_the_timer_shows_its_format(self) -> None:
        """本文だけ見ると、タイマーの「時分秒」が細い字で出ても知らせない"""
        source = TEXT.create(text="", font=LATIN_ONLY, timer_format="hh時mm分ss秒")
        assert missing_glyphs(source) == "時分秒"

    @pytest.mark.parametrize(("at", "shown"), [(1, []), (2, [(1, 2)]), (3, []), (4, [(2, 3)])])
    def test_frames_skipped_by_speed_are_not_returned(
        self, at: int, shown: list[tuple[int, int]]
    ) -> None:
        """倍速で飛ばされる 1 コマの字幕を数えると、映らない字を AI と人に直させる"""
        text = Clip(timeline_start=at, duration=1, source=TEXT.create(text="字", font=LATIN_ONLY))
        project = Project.create(ProjectSettings())
        inner = Track(kind=TrackKind.VIDEO, name="V1", clips=(text,))
        scene = Scene("字幕", replace(project.timeline, tracks=(inner,)))
        placed = Clip(timeline_start=0, duration=3, scene_id=scene.id, speed=Fraction(2))
        outer = Track(kind=TrackKind.VIDEO, name="V1", clips=(placed,))
        project = replace(project, scenes=(scene,)).with_timeline(
            replace(project.timeline, tracks=(outer,))
        )
        runs = glyph_runs(project, 0, 3)
        assert [(run.start, run.end) for run in runs] == shown
        for run in runs:
            assert run.values["in_scene_clip"] == str(placed.id)
            assert fix_for(MISSING_GLYPH, MEDIUM, run.values).action == HANDOFF


class TestMatching:
    def test_the_absent_character_gets_heavier(self) -> None:
        """壊れると、欄を入れても細い字のまま浮く"""
        assert _inked("字幕", matched=True).sum() > _inked("字幕", matched=False).sum() * 1.15

    def test_the_characters_stay_in_place(self) -> None:
        """送り幅が変わると、欄を入れただけで折り返しと位置が崩れる"""
        plain, heavy = _inked("AB字幕CD", matched=False), _inked("AB字幕CD", matched=True)
        left, right = np.flatnonzero(plain.max(axis=0) > 0)[[0, -1]]
        assert np.array_equal(np.flatnonzero(heavy.max(axis=0) > 0)[[0, -1]], (left, right))
        # 両端は書体にある字 輪郭を合わせる丸めで縁のぼかしが少し変わるだけで、1 画素も動かない
        for part in (slice(left, left + 40), slice(right - 40, right + 1)):
            assert np.abs(plain[:, part] - heavy[:, part]).max() < 64

    def test_text_without_absent_characters_is_untouched(self) -> None:
        """ある字まで太らせると、欄を入れた字幕の全部が滲む"""
        assert np.array_equal(_inked("ABCD", matched=False), _inked("ABCD", matched=True))

    def test_bold_text_does_not_shift(self) -> None:
        """太らせた字で中心戻しを測ると、行の端に無い字があるだけで行の全部が横へずれる"""
        lefts = []
        for matched in (False, True):
            source = TEXT.create(
                text="AB字", font=LATIN_ONLY, size=96.0, bold=True, match_fallback=matched
            )
            inked = np.asarray(render_source(source, 800, 200))[..., 3].max(axis=0)
            lefts.append(int(np.flatnonzero(inked > 128)[0]))
        assert lefts[0] == lefts[1]

    def test_off_by_default(self) -> None:
        """入を既定にすると、保存済みの作品と互換の見た目が版を上げただけで変わる"""
        assert TEXT.create().params["match_fallback"] is False
