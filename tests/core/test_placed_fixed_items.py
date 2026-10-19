"""置いたときに付く最初から持つ欄と、足したエフェクトの位置（#27 YMM4 型アイテムの設計 P2）

YMM4 のアイテムは置いた時点で 描画（X・Y・拡大率・回転角・左右反転）と 音声（音量・パン・
フェード）の欄を持つ こちらでは印の付いたエフェクト（:attr:`Effect.fixed`）で表し、
並びは YMM4 と同じく足したエフェクトの後ろ（エフェクトを掛けてから置く）
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.compat.catalog import place, restyle
from sashimono.compat.mapped import MappedObject
from sashimono.compat.ymm4.template import map_template
from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddTrack,
    Command,
    SetClipProperty,
    insert_filter,
    insert_generated,
    insert_media,
)
from sashimono.core.commands.fixed import (
    FIXED_ORDER,
    FLIP_EFFECT_KIND,
    TRANSFORM_EFFECT_KIND,
    fixed_effect,
    with_fixed_items,
)
from sashimono.core.io.serialize import clip_from_json, clip_to_json
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    MediaItem,
    Project,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import TEXT, TRANSITION


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _clips(project: Project, kind: TrackKind) -> list[Clip]:
    return [c for t in project.timeline.tracks if t.kind is kind for c in t.clips]


def _still() -> MediaItem:
    return MediaItem(
        path=Path("C:/素材/絵.png"),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=640,
                height=360,
                frame_rate=FrameRate(30),
                time_base=Fraction(1, 30),
                codec="png",
            ),
        ),
    )


class TestPlacing:
    def test_a_placed_picture_gets_flip_then_transform(self, video_media: MediaItem) -> None:
        # 欄が無いと、置いた動画の位置を変えるのに変形をエフェクトの一覧から探して足すことになる
        # 並びが逆だと、裏返してから回す YMM4 と回す向きが食い違う
        project = _apply(Project.create(), insert_media(Project.create(), video_media))
        (picture,) = _clips(project, TrackKind.VIDEO)
        assert [e.kind for e in picture.effects] == [FLIP_EFFECT_KIND, TRANSFORM_EFFECT_KIND]
        assert all(e.fixed for e in picture.effects)

    def test_a_placed_sound_gets_volume_then_fade(self, video_media: MediaItem) -> None:
        # 音声の欄は YMM4 の並び（音量・パン → フェード）
        project = _apply(Project.create(), insert_media(Project.create(), video_media))
        (sound,) = _clips(project, TrackKind.AUDIO)
        assert [e.kind for e in sound.effects] == ["audio_volume", "audio_fade"]
        assert all(e.fixed for e in sound.effects)

    def test_a_placed_still_gets_the_picture_items(self) -> None:
        # 静止画だけ欄が付かないと、置いた画像の位置や反転をパネルの組で変えられない
        project = _apply(Project.create(), insert_media(Project.create(), _still()))
        (picture,) = _clips(project, TrackKind.VIDEO)
        assert [e.kind for e in picture.effects] == [FLIP_EFFECT_KIND, TRANSFORM_EFFECT_KIND]

    def test_placed_text_gets_the_picture_items(self) -> None:
        # テキストも YMM4 では描画の欄を持つ 素材だけに付けると、テキストの位置だけ
        # パネルの組ではなくエフェクトの一覧から探すことになる
        project = Project.create()
        project = _apply(project, insert_generated(project, TEXT.create()))
        (text,) = _clips(project, TrackKind.VIDEO)
        assert [e.kind for e in text.effects] == [FLIP_EFFECT_KIND, TRANSFORM_EFFECT_KIND]

    def test_filters_and_transitions_get_no_items(self) -> None:
        # フィルタと場面切り替えは自分の絵を置かない 欄を付けると動かしても効かない欄が並ぶ
        project = Project.create()
        project = _apply(project, insert_filter(project))
        project = _apply(project, insert_generated(project, TRANSITION.create()))
        for clip in _clips(project, TrackKind.VIDEO):
            assert not any(e.fixed for e in clip.effects)

    def test_a_placed_picture_is_drawn_at_its_own_pixels(self, video_media: MediaItem) -> None:
        # 利用者の決定 拡大率 100% は素材の画素 画面に収めると、YMM4 の拡大率を写しても
        # 素材の解像度ごとに違う大きさになる
        project = _apply(Project.create(), insert_media(Project.create(), video_media))
        (picture,) = _clips(project, TrackKind.VIDEO)
        (sound,) = _clips(project, TrackKind.AUDIO)
        assert picture.native_size
        assert not sound.native_size

    def test_an_old_clip_keeps_fitting(self) -> None:
        # 項目の無いクリップ（前の版のファイル・手で作ったクリップ）は画面に収めたまま
        assert not Clip(timeline_start=0, duration=1).native_size


class TestAddingEffects:
    def test_an_added_effect_goes_before_the_fixed_items(self, video_media: MediaItem) -> None:
        # 末尾へ積むと、足したぼかしが置いた後の絵に掛かり、YMM4 で同じ設定にした絵と違う
        # 設定パネル・右クリック・AI の add_effect はどれも位置を渡さずにこれを使う
        project = _apply(Project.create(), insert_media(Project.create(), video_media))
        (picture,) = _clips(project, TrackKind.VIDEO)
        blur = registry.require("blur").create()
        project = AddEffect(picture.id, blur).apply(project)
        (picture,) = _clips(project, TrackKind.VIDEO)
        assert [e.kind for e in picture.effects] == ["blur", "flip", "transform"]

    def test_a_missing_fixed_item_goes_to_its_own_place(self) -> None:
        # 前の版のファイルで欄を持たないクリップに、触ったときに欄を足す 末尾や先頭へ入ると
        # 反転が配置の後ろに来るなど、パネルの組の並びと掛かる順が食い違う
        blur = registry.require("blur").create()
        clip = Clip(
            timeline_start=0,
            duration=30,
            source=GeneratedSource(kind="text"),
            effects=(blur, fixed_effect(TRANSFORM_EFFECT_KIND)),
        )
        project = _on_a_track(clip)
        project = AddEffect(clip.id, fixed_effect(FLIP_EFFECT_KIND)).apply(project)
        (text,) = _clips(project, TrackKind.VIDEO)
        assert [e.kind for e in text.effects] == ["blur", "flip", "transform"]

    def test_the_fixed_items_are_not_added_twice(self) -> None:
        # 読み込みが写した欄に印が付いていれば、置くときに同じ種類を足さない
        own = fixed_effect(TRANSFORM_EFFECT_KIND)
        clip = with_fixed_items(Clip(timeline_start=0, duration=1, effects=(own,)), picture=True)
        assert [e.kind for e in clip.effects] == ["flip", "transform"]
        assert clip.effects[1] is own


def _on_a_track(clip: Clip) -> Project:
    track = Track(kind=TrackKind.VIDEO)
    return _apply(Project.create(), [AddTrack(track), AddClip(track.id, clip)])


class TestDefaults:
    @pytest.mark.parametrize("kind", FIXED_ORDER)
    def test_the_defaults_match_the_definition(self, kind: str) -> None:
        # コア層は定義を読めないので同じ値を別に書いている 食い違うと、置いただけで
        # 絵か音が変わる 反転だけは定義の既定（左右 真）と違い、裏返さない値で持つ
        definition = registry.require(kind)
        expected = definition.default_params()
        if kind == FLIP_EFFECT_KIND:
            expected["horizontal"] = False
        assert fixed_effect(kind).params == expected

    @pytest.mark.parametrize("kind", [FLIP_EFFECT_KIND, TRANSFORM_EFFECT_KIND])
    def test_the_placed_picture_items_do_nothing(self, kind: str) -> None:
        # 置いた物が何もしない値と見なされないと、全クリップが中間バッファを通って遅くなる
        assert registry.require(kind).is_idle(fixed_effect(kind))

    def test_a_moved_transform_is_not_idle(self) -> None:
        # 何もしない物と見なされると描くときに飛ばされ、パネルで X を動かしても絵が元の位置のまま
        moved = fixed_effect(TRANSFORM_EFFECT_KIND).with_param("pos_x", AnimatedValue(10.0))
        assert not registry.require(TRANSFORM_EFFECT_KIND).is_idle(moved)

    def test_an_animated_transform_is_not_idle(self) -> None:
        # 動く値は途中で動く 頭の値だけ見て飛ばすと、動きが出ない
        from sashimono.core.model import Keyframe

        wobble = AnimatedValue(0.0, keyframes=(Keyframe(0, 0.0), Keyframe(10, 0.0)))
        moving = fixed_effect(TRANSFORM_EFFECT_KIND).with_param("rotation", wobble)
        assert not registry.require(TRANSFORM_EFFECT_KIND).is_idle(moving)


class TestFile:
    def test_native_size_survives_a_round_trip(self) -> None:
        # 落ちると、開き直しただけで素材の画素で置いた物が画面いっぱいへ引き伸ばされる
        clip = Clip(timeline_start=0, duration=1, native_size=True)
        assert clip_from_json(clip_to_json(clip)).native_size

    def test_the_old_way_is_not_written(self) -> None:
        # 画面に収める物は項目を書かない 前の版の本体も同じ形で読める（版を上げない）
        assert "native_size" not in clip_to_json(Clip(timeline_start=0, duration=1))

    def test_a_clip_without_the_field_keeps_fitting(self) -> None:
        # 前の版のファイルを開いて見た目が変わると、作った動画の構図が崩れる
        raw = clip_to_json(Clip(timeline_start=0, duration=1))
        assert not clip_from_json(raw).native_size


class TestClipProperties:
    def test_the_start_position_can_be_changed(self, video_media: MediaItem) -> None:
        # 設定パネルの再生開始位置 断られると動画・音声の組の欄が動かない
        project = _apply(Project.create(), insert_media(Project.create(), video_media))
        (picture,) = _clips(project, TrackKind.VIDEO)
        project = SetClipProperty(picture.id, "source_in", Fraction(3, 2)).apply(project)
        (picture,) = _clips(project, TrackKind.VIDEO)
        assert picture.source_in == Fraction(3, 2)

    def test_a_float_start_position_is_refused(self, video_media: MediaItem) -> None:
        # 小数のまま入ると保存の所で分数に直せずに落ちる
        project = _apply(Project.create(), insert_media(Project.create(), video_media))
        (picture,) = _clips(project, TrackKind.VIDEO)
        with pytest.raises(ValueError, match="分数"):
            SetClipProperty(picture.id, "source_in", 1.5).apply(project)


def _ymm4_item(name: str, **fields: object) -> dict[str, object]:
    return {
        "$type": f"YukkuriMovieMaker.Project.Items.{name}, YukkuriMovieMaker",
        "Frame": 0,
        "Layer": 0,
        "Length": 30,
        **fields,
    }


def _value(number: float) -> dict[str, object]:
    return {"Values": [{"Value": number}], "Span": 0.0, "AnimationType": "なし"}


class TestAviUtl:
    def test_the_placement_goes_after_the_filters(self) -> None:
        # 標準描画 を列の先頭に置くと、読み込んだフィルタが固定の欄の後ろに閉じ込められ、
        # あとで足したエフェクト（欄の前へ入る）と並べ替えられない AviUtl もフィルタを
        # 掛けた絵を最後に置く
        from sashimono.compat.aviutl.exo import parse_exo
        from sashimono.compat.aviutl.mapping import map_object
        from sashimono.compat.aviutl.report import CompatibilityReport

        text = (
            "[0]\nstart=1\nend=60\nlayer=1\n"
            "[0.0]\n_name=図形\n"
            "[0.1]\n_name=ぼかし\n範囲=4\n"
            "[0.2]\n_name=標準描画\nX=120.0\n"
        )
        mapped = map_object(parse_exo(text).objects[0], FrameRate(30), report=CompatibilityReport())
        assert mapped is not None
        kinds = [(e.kind, e.fixed) for e in mapped.clip.effects]
        assert kinds == [("blur", False), ("transform", True)]


class TestYmm4:
    def test_the_placement_of_an_item_becomes_the_fixed_transform(self) -> None:
        # 写した配置に印が無いと、置くときに既定の配置がもう 1 つ足され、パネルの X には
        # 既定の 0 が出て、実際の位置は一覧の中の変形が持つ
        text = _ymm4_item("TextItem", Text="字", X=_value(120.0), Zoom=_value(50.0))
        project = Project.create()
        project = _apply(project, place(map_template([text]), project))
        (clip,) = _clips(project, TrackKind.VIDEO)
        kinds = [(e.kind, e.fixed) for e in clip.effects]
        assert kinds[-2:] == [("flip", True), ("transform", True)]
        assert sum(1 for kind, _ in kinds if kind == "transform") == 1
        assert clip.effects[-1].params["pos_x"] == AnimatedValue(120.0)

    def test_an_image_item_is_drawn_at_its_own_pixels(self) -> None:
        # YMM4 は拡大率 100% で素材の画素に置く 画面に収めると、画面と解像度の違う
        # 画像がテンプレートの拡大率のまま別の大きさになる
        image = _ymm4_item("ImageItem", FilePath="C:/素材/絵.png")
        (mapped,) = map_template([image])
        assert mapped.clip.native_size

    def test_restyle_moves_the_template_placement_into_the_clip(self) -> None:
        # 字幕テンプレートの位置は、着せる先の描画の欄へ入る 足すと、クリップの欄と
        # テンプレートの配置が 2 重に掛かって位置がずれる
        placed = fixed_effect(TRANSFORM_EFFECT_KIND).with_param("pos_y", AnimatedValue(-400.0))
        template = MappedObject(
            clip=Clip(
                timeline_start=0,
                duration=30,
                source=GeneratedSource(kind="text", params={"text": "見本"}),
                effects=(placed,),
            ),
            layer=1,
        )
        own = fixed_effect(TRANSFORM_EFFECT_KIND)
        target = Clip(
            timeline_start=0,
            duration=30,
            source=GeneratedSource(kind="text", params={"text": "字"}),
            effects=(own,),
        )
        commands = restyle([template], target)
        assert not any(isinstance(c, AddEffect) for c in commands)
        project = _apply(_on_a_track(target), commands)
        (clip,) = _clips(project, TrackKind.VIDEO)
        assert clip.effects[0].params["pos_y"] == AnimatedValue(-400.0)

        # 既定の位置のテンプレート（読み込みは既定の配置を写さない）を着せ直すと、欄は既定へ
        # 戻る 残すと、前に着せたテンプレートの位置のまま新しい見た目になる
        plain = MappedObject(
            clip=Clip(
                timeline_start=0,
                duration=30,
                source=GeneratedSource(kind="text", params={"text": "見本"}),
            ),
            layer=1,
        )
        project = _apply(project, restyle([plain], clip))
        (clip,) = _clips(project, TrackKind.VIDEO)
        assert clip.effects[0].params["pos_y"] == AnimatedValue(0.0)
