"""字幕を選んだ見た目で置く（#308）

エイリアス・プリセット・テンプレートの棚の見た目を、着せ替えと同じ決まり（今の文字と長さを
残し、動きと縁取りの層を字幕の尺に合わせる）で字幕に着せる 話し手ごとに別の見た目にでき、
取り消し 1 回で置いた全部が戻る
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sashimono.compat.catalog import TemplateCatalog, TemplateEntry
from sashimono.compat.subtitle_looks import (
    NO_TEXT,
    LookError,
    LookRules,
    alias_choice,
    default_choice,
    find_choice,
    preset_choice,
    subtitle_look,
    template_choice,
)
from sashimono.core.commands import AddClip, Command, Document, burn_defaults, burn_subtitles
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.commands.preset import PresetOptions
from sashimono.core.io.aliases import Alias, AliasStore
from sashimono.core.io.library import Library
from sashimono.core.io.presets import Preset, PresetStore
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    Project,
    Stroke,
)
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE, TEXT
from tests.core.test_burn_voices import _project

#: 1080p で画面の 90% の折り返し 字幕の既定（本人の設定の既定）
WRAP = 1920 * 0.9


def _base(wrap: float = WRAP) -> GeneratedSource:
    return TEXT.create(**burn_defaults(1080, wrap))


def _styled_alias(name: str = "黄色テロップ") -> Alias:
    """書体・色・縁取りの層・ぼかし・120 フレームで大きくなる動きを持つテキストのエイリアス"""
    source = TEXT.create(
        text="見本の文字",
        font="Meiryo",
        color=(1.0, 0.9, 0.0, 1.0),
        size=AnimatedValue(keyframes=(Keyframe(0, 40.0), Keyframe(119, 80.0))),
    ).with_strokes([Stroke(params={"width": AnimatedValue(9.0)})])
    clip = with_fixed_items(
        Clip(
            timeline_start=0,
            duration=120,
            source=source,
            effects=(registry.require("blur").create(radius=3),),
        ),
        picture=True,
    )
    return Alias.of(name, clip)


def _placed(project: Project, commands: list[Command]) -> dict[str, list[Clip]]:
    for command in commands:
        project = command.apply(project)
    return {
        track.name: list(track.clips)
        for track in project.timeline.tracks
        if track.name.startswith("字幕")
    }


def _value(clip: Clip, name: str) -> object:
    assert clip.source is not None
    return clip.source.params[name]


class TestAlias:
    def test_the_look_is_worn_and_the_text_and_length_stay(self) -> None:
        # 壊れると、字幕がエイリアスの見本の文字や 120 フレームの長さで置かれる
        project, _media = _project()
        look = subtitle_look(alias_choice(_styled_alias()), _base())
        tracks = _placed(project, burn_subtitles(project, look))
        first = tracks["字幕 音声 1"][0]
        assert _value(first, "text") == "ゲームの声"
        assert first.duration == 60
        assert _value(first, "font") == "Meiryo"
        assert _value(first, "color") == (1.0, 0.9, 0.0, 1.0)
        assert [e.kind for e in first.effects if not e.fixed] == ["blur"]

    def test_the_stroke_layers_come_along(self) -> None:
        # 壊れると、縁取りの層（#272）を持つエイリアスを選んでも縁の無い字幕になる
        project, _media = _project()
        look = subtitle_look(alias_choice(_styled_alias()), _base())
        first = _placed(project, burn_subtitles(project, look))["字幕 音声 1"][0]
        assert first.source is not None
        assert [s.params["width"] for s in first.source.strokes] == [AnimatedValue(9.0)]

    def test_the_motion_fits_each_subtitle(self) -> None:
        # 壊れると、120 フレームの動きが 60 フレームの字幕の途中で切れ、大きくなりきらない
        project, _media = _project()
        look = subtitle_look(alias_choice(_styled_alias()), _base())
        tracks = _placed(project, burn_subtitles(project, look))
        for clip in tracks["字幕 音声 1"]:
            size = _value(clip, "size")
            assert isinstance(size, AnimatedValue)
            assert [k.frame for k in size.keyframes] == [0, clip.duration - 1]

    def test_an_alias_without_text_cannot_be_chosen(self) -> None:
        # 壊れると、図形のエイリアスを選べてしまい、字幕の文字が消えた図形が並ぶ
        shape = with_fixed_items(
            Clip(timeline_start=0, duration=30, source=SHAPE.create()), picture=True
        )
        choice = alias_choice(Alias.of("丸", shape))
        assert choice.refusal == NO_TEXT
        with pytest.raises(LookError):
            subtitle_look(choice, _base())


class TestPreset:
    def test_a_text_preset_is_applied_without_its_text(self) -> None:
        # 壊れると、字幕がどれもプリセットを保存したときの文字になる
        project, _media = _project()
        saved = Clip(
            timeline_start=0,
            duration=30,
            source=TEXT.create(text="保存した文字", font="Yu Mincho", size=70.0),
        )
        preset = Preset.capture("明朝", with_fixed_items(saved, picture=True))
        rules = LookRules(preset_options=PresetOptions(with_text=True))
        look = subtitle_look(preset_choice(preset), _base(), rules)
        first = _placed(project, burn_subtitles(project, look))["字幕 音声 1"][0]
        assert _value(first, "text") == "ゲームの声"
        assert _value(first, "font") == "Yu Mincho"
        assert _value(first, "size") == AnimatedValue(70.0)

    def test_a_shape_preset_cannot_be_chosen(self) -> None:
        # 壊れると、図形のプリセットを選べてしまい、着せても何も変わらない字幕が並ぶ
        saved = with_fixed_items(
            Clip(timeline_start=0, duration=30, source=SHAPE.create()), picture=True
        )
        assert preset_choice(Preset.capture("四角", saved)).refusal == NO_TEXT


class TestWrap:
    def test_a_look_without_wrap_gets_the_subtitle_wrap(self) -> None:
        # 壊れると、折り返さないエイリアスで置いた長い字幕が画面の端で切れる（#256）
        look = subtitle_look(alias_choice(_styled_alias()), _base())
        assert callable(look)
        made = look("とても長い字幕" * 20, 30)
        assert _value(made, "wrap_width") == AnimatedValue(WRAP)

    def test_an_aviutl_layout_switches_to_one_that_wraps(self) -> None:
        # 壊れると、AviUtl2 の組み方のまま折り返しの幅だけが入り、字幕は 1 行のまま切れる（#301）
        alias = _styled_alias()
        assert alias.clip.source is not None
        aviutl = Alias.of(
            "AviUtl",
            Clip(
                timeline_start=0,
                duration=60,
                source=alias.clip.source.with_param("layout", "aviutl"),
            ),
        )
        look = subtitle_look(alias_choice(aviutl), _base())
        assert callable(look)
        made = look("字幕", 30)
        assert _value(made, "layout") == "native"
        assert _value(made, "wrap_width") == AnimatedValue(WRAP)

    def test_no_wrap_is_added_when_the_setting_is_off(self) -> None:
        # 壊れると、折り返しを切った人の字幕まで折り返される
        look = subtitle_look(alias_choice(_styled_alias()), _base(0.0))
        assert callable(look)
        assert _value(look("字幕", 30), "wrap_width") == AnimatedValue(0.0)


class TestEachVoice:
    def test_each_voice_wears_its_own_look_and_one_undo_takes_all_back(self) -> None:
        # 壊れると、話し手ごとに選んだ見た目が全員同じになるか、取り消しで一部が残る
        project, media = _project()
        voice2 = (media.id, 2)
        everyone = subtitle_look(default_choice(), _base())
        mine = subtitle_look(alias_choice(_styled_alias()), _base())
        document = Document(project)
        commands = burn_subtitles(project, everyone, looks={voice2: mine})
        with document.checkpoint("字幕を焼き込み"):
            for command in commands:
                document.execute(command)
        tracks = {t.name: list(t.clips) for t in document.project.timeline.tracks if t.clips}
        assert _value(tracks["字幕 音声 2"][0], "font") == "Meiryo"
        assert _value(tracks["字幕 音声 1"][0], "font") != "Meiryo"
        assert sum(isinstance(c, AddClip) for c in commands) == 3
        document.undo()
        assert document.project == project


class TestTemplateShelf:
    def test_a_shelf_alias_is_worn_like_restyle(self, tmp_path: Path) -> None:
        # 壊れると、棚のテンプレートを選んでも、字幕が見本の文字・テンプレートの長さで置かれる
        (tmp_path / "字幕").mkdir()
        (tmp_path / "字幕" / "大きい.object").write_text(
            "\n".join(
                [
                    "[Object]",
                    "frame=0,89",
                    "[Object.0]",
                    "effect.name=テキスト",
                    "サイズ=88",
                    "テキスト=見本",
                    "",
                ]
            ),
            "utf-8",
        )
        catalog = TemplateCatalog()
        (entry,) = catalog.scan((tmp_path,))
        assert isinstance(entry, TemplateEntry)
        project, _media = _project()
        look = subtitle_look(template_choice(entry), _base())
        first = _placed(project, burn_subtitles(project, look))["字幕 音声 1"][0]
        assert _value(first, "text") == "ゲームの声"
        assert first.duration == 60
        assert _value(first, "size") == AnimatedValue(88.0)


class TestByName:
    def test_presets_and_aliases_are_found_by_name(self, tmp_path: Path) -> None:
        library = Library(
            presets=PresetStore(tmp_path / "presets"), aliases=AliasStore(tmp_path / "aliases")
        )
        library.aliases.save(_styled_alias("黄色"))
        assert find_choice(library, "alias", "黄色").label == "黄色"
        # 無い名前は、ある物の名前を添えて断る（AI が次に何を頼めばよいか分かるように）
        with pytest.raises(LookError, match="黄色"):
            find_choice(library, "alias", "無い")
