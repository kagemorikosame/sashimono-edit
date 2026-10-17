"""AI がプリセット・エイリアス・テンプレートの棚を読み、当てる・置く道具（#328 B2）

人の操作（設定パネルの〔プリセット…〕・タイムラインの〔追加〕→〔エイリアス〕・互換の
〔テンプレート…〕と〔オブジェクトを読み込む…〕）と同じ手順（``compat/looks.py``）を通り、
取り消し 1 回で戻ることを見る

各試験の頭の 1 行は、壊れたときに利用者に何が起きるか
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.compat.catalog import TemplateCatalog, TemplateEntry
from sashimono.compat.looks import apply_preset_commands, capture_preset
from sashimono.compat.subtitle_looks import LookRules, template_choice
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.commands.preset import PresetOptions
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE, TEXT
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run
from tests.compat.test_subtitle_looks import _styled_alias

#: 字幕の見本の形（AviUtl2 世代の ``.object``） 実物（%PROGRAMDATA%\aviutl2\Alias の
#: 字幕テンプレート）と同じ書き方で、サイズと位置だけを変えてある
_SUBTITLE_OBJECT = "\n".join(
    [
        "[Object]",
        "frame=0,89",
        "[Object.0]",
        "effect.name=テキスト",
        "サイズ=88.00",
        "文字色=ffff00",
        "テキスト=見本の字幕",
        "[Object.1]",
        "effect.name=標準描画",
        "X=0.00",
        "Y=400.00",
        "拡大率=100.000",
        "透明度=0.00",
        "",
    ]
)

#: 効果だけのエイリアス（AviUtl2 の アニメーション効果 の後ろにぼかしを積んだ物）
#: アニメーション効果のスクリプトは手元に無いので読めず、ぼかしだけが読める
_EFFECT_OBJECT = "\n".join(
    [
        "[Object]",
        "frame=0,59",
        "[Object.0]",
        "effect.name=アニメーション効果",
        "name=手元に無い効果",
        "[Object.1]",
        "effect.name=ぼかし",
        "範囲=5",
        "",
    ]
)

#: レイヤー 1 に小さい字、レイヤー 2 に大きい字を置いた ``.exo``（AviUtl1 の形）
#: 元のレイヤーが埋まっているとき、上下の並び（2 が手前）を保って新しいトラックへ置くかを見る
_EXO_TWO_LAYERS = """[exedit]
width=1920
height=1080
rate=30
scale=1
length=300
audio_rate=44100
audio_ch=2
[0]
start=1
end=60
layer=1
overlay=1
camera=0
[0.0]
_name=テキスト
サイズ=40
text=53004b0055005400
[0.1]
_name=標準描画
X=0.0
Y=0.0
Z=0.0
拡大率=100.00
透明度=0.0
回転=0.00
blend=0
[1]
start=1
end=60
layer=2
overlay=1
camera=0
[1.0]
_name=テキスト
サイズ=80
text=53004b0055005400
[1.1]
_name=標準描画
X=0.0
Y=0.0
Z=0.0
拡大率=100.00
透明度=0.0
回転=0.00
blend=0
"""

#: 画像を 1 つ参照する ``.exo``（AviUtl1 の形） 素材の登録と配置が 1 回の取り消しになるかを見る
_EXO_IMAGE = """[exedit]
width=1920
height=1080
rate=30
scale=1
length=300
audio_rate=44100
audio_ch=2
[0]
start=1
end=60
layer=1
group=1
overlay=1
camera=0
[0.0]
_name=画像ファイル
ファイル={path}
[0.1]
_name=標準描画
X=0.0
Y=0.0
Z=0.0
拡大率=100.00
透明度=0.0
回転=0.00
blend=0
"""


def _text_clip(text: str, start: int = 0, duration: int = 90) -> Clip:
    return with_fixed_items(
        Clip(timeline_start=start, duration=duration, source=TEXT.create(text=text)),
        picture=True,
    )


def _with_texts(host: FakeHost, *texts: str) -> list[Clip]:
    """テキストのクリップを並べた映像トラックを足す 置いたクリップを返す"""
    clips = [_text_clip(text, start=index * 100) for index, text in enumerate(texts)]
    track = Track(TrackKind.VIDEO, "T1", tuple(clips))
    project = host.document.project
    host.document.reset(
        project.with_timeline(replace(project.timeline, tracks=(*project.timeline.tracks, track)))
    )
    return clips


def _clip(host: FakeHost, clip_id: object) -> Clip:
    located = host.document.project.timeline.locate_clip(clip_id)  # type: ignore[arg-type]
    assert located is not None
    return located[1]


def _yellow_preset(host: FakeHost, name: str = "黄色", category: str = "ユーザー") -> None:
    """黄色の字・ぼかしを持つテキストの見た目を、画面と同じ写し方で保存する"""
    look = with_fixed_items(
        Clip(
            timeline_start=0,
            duration=90,
            source=TEXT.create(text="保存した文字", color=(1.0, 1.0, 0.0, 1.0), size=72),
            effects=(registry.require("blur").create(radius=4),),
        ),
        picture=True,
    )
    track = Track(TrackKind.VIDEO, "V", (look,))
    preset = replace(capture_preset(name, None, track, look), category=category)
    host.library.presets.save(preset)


def _shelf_entry(
    tmp_path: Path,
    folder: str = "字幕",
    name: str = "大きい",
    text: str = _SUBTITLE_OBJECT,
    suffix: str = ".object",
) -> TemplateEntry:
    """棚の 1 本 置き場は試験ごとの一時フォルダ（本人の AviUtl2 の置き場は読まない）"""
    shelf = tmp_path / name
    (shelf / folder).mkdir(parents=True, exist_ok=True)
    (shelf / folder / f"{name}{suffix}").write_text(text, encoding="utf-8")
    catalog = TemplateCatalog()
    (entry,) = catalog.scan((shelf,))
    return entry


def _sizes_in_new_tracks(host: FakeHost, before: Project) -> list[float]:
    """足したトラックに置かれたテキストの大きさ トラックの並び（奥から手前）の順"""
    old = {track.id for track in before.timeline.tracks}
    sizes: list[float] = []
    for track in host.document.project.timeline.tracks:
        if track.id in old:
            # 前からあるトラックには何も足さない
            assert track == before.timeline.find_track(track.id)
            continue
        for clip in track.clips:
            assert clip.source is not None
            size = clip.source.params["size"]
            assert isinstance(size, AnimatedValue)
            sizes.append(size.static)
    return sizes


def _template_id(entry: TemplateEntry) -> str:
    return template_choice(entry).key


def _look(clip: Clip) -> tuple[object, ...]:
    """見た目だけを比べる形 当てるたびに振り直すエフェクトの ID は外す"""
    return (
        clip.source,
        tuple((effect.kind, effect.params, effect.enabled) for effect in clip.effects),
        clip.opacity,
        clip.blend_mode,
    )


def _empty_host() -> FakeHost:
    """何も置いていない作品 AviUtl のファイルは元のレイヤーの並びのまま頭から置くので、
    クリップのある作品へ読むと重なって断られる（人の〔オブジェクトを読み込む…〕と同じ）"""
    return FakeHost(Project.create(ProjectSettings(frame_rate=FrameRate(30))))


class TestTheToolsAreDeclared:
    @pytest.mark.parametrize(
        ("name", "safety"),
        [
            ("list_looks", Safety.READ_ONLY),
            ("apply_preset", Safety.UNDOABLE),
            ("place_alias", Safety.UNDOABLE),
            ("place_template", Safety.UNDOABLE),
            ("apply_template", Safety.UNDOABLE),
            ("import_objects", Safety.UNDOABLE),
        ],
    )
    def test_each_tool_has_its_safety(self, name: str, safety: Safety) -> None:
        # 壊れると、取り消せる編集なのに毎回確かめが出るか、読むだけの道具が確かめなしで何かを変える
        operation = find_operation(name)
        assert operation is not None, name
        assert operation.safety is safety

    def test_names_of_saved_looks_are_sent_as_app_settings(self) -> None:
        # 壊れると、本人の持ち物の名前を返す道具が、送る物の一覧（設定の「AI へ送る物」）に出ない
        for name in ("list_looks", "apply_preset", "place_alias", "place_template"):
            operation = find_operation(name)
            assert operation is not None
            assert Outbound.APP_SETTINGS in operation.sends, name

    def test_folders_of_the_shelf_are_names_and_places(self) -> None:
        # 壊れると、名前と場所を断った人の棚のフォルダの名前（本人の名前が入りうる）が伏せずに出る
        for name in ("list_looks", "place_template", "apply_template", "import_objects"):
            operation = find_operation(name)
            assert operation is not None
            assert Outbound.FILES in operation.sends, name


class TestList:
    def test_presets_aliases_and_the_shelf_are_listed(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、AI は保存した見た目の名前を知らず、一から作り直す
        _yellow_preset(host, category="字幕")
        host.library.aliases.save(_styled_alias("テロップ"))
        entry = _shelf_entry(tmp_path / "棚")
        host.shelf = [entry]
        looks = run(host, "list_looks")["looks"]
        kinds = {(row["kind"], row["name"]) for row in looks}
        assert kinds == {("preset", "黄色"), ("alias", "テロップ"), ("template", "大きい")}
        preset = next(row for row in looks if row["kind"] == "preset")
        assert preset["category"] == "字幕"
        assert preset["source"] == "text"
        template = next(row for row in looks if row["kind"] == "template")
        assert template["template_id"] == _template_id(entry)
        assert template["folder"] == "字幕"
        # 中身の場所は返さない 選ぶのは template_id で足り、場所には本人の名前が入りうる
        assert "path" not in template

    def test_it_can_be_narrowed(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、数百本の棚が並び、AI が保存した見た目を探し出せない
        _yellow_preset(host)
        host.shelf = [_shelf_entry(tmp_path)]
        assert [row["kind"] for row in run(host, "list_looks", kind="preset")["looks"]] == [
            "preset"
        ]
        assert run(host, "list_looks", contains="大き")["looks"][0]["name"] == "大きい"
        with pytest.raises(ToolError, match="preset・alias・template"):
            run(host, "list_looks", kind="何か")

    def test_listing_does_not_read_the_shelf(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、一覧を読むだけで棚の配布物を全部開き、数百本の棚で道具が返らない
        entry = _shelf_entry(tmp_path)
        (tmp_path / "大きい" / "字幕" / "大きい.object").write_text("壊れた中身", encoding="utf-8")
        host.shelf = [entry]
        (row,) = run(host, "list_looks")["looks"]
        assert row["name"] == "大きい"


class TestApplyPreset:
    def test_it_matches_the_panel_and_undoes_in_one(self, host: FakeHost) -> None:
        # 壊れると、AI が当てた見た目が設定パネルから当てた物と違い、戻すのに何回も押す
        _yellow_preset(host)
        first, second = _with_texts(host, "一つ目", "二つ目")
        before = host.document.project
        expected = before
        preset = host.saved_looks.presets[0]
        for clip in (first, second):
            located = expected.timeline.locate_clip(clip.id)
            assert located is not None
            for command in apply_preset_commands(
                expected, *located, preset, options=PresetOptions()
            ):
                expected = command.apply(expected)
        result = run(host, "apply_preset", name="黄色", clip_ids=[str(first.id), str(second.id)])
        assert result["clip_ids"] == [str(first.id), str(second.id)]
        for clip in (first, second):
            located = expected.timeline.locate_clip(clip.id)
            assert located is not None
            assert _look(_clip(host, clip.id)) == _look(located[1])
        # 文字は当てる先の物のまま（設定パネルの既定と同じ）
        assert _clip(host, first.id).source.params["text"] == "一つ目"  # type: ignore[union-attr]
        assert host.document.history_labels[-1] == "プリセット: 黄色（2 本）"
        run(host, "undo")
        assert host.document.project == before

    def test_the_selection_is_the_default(self, host: FakeHost) -> None:
        # 壊れると、「選んでいるのに当てて」が通じない
        _yellow_preset(host)
        (clip,) = _with_texts(host, "選んだ")
        host.select_clips([clip.id])
        assert run(host, "apply_preset", name="黄色")["clip_ids"] == [str(clip.id)]

    def test_the_person_s_way_of_applying_is_used(self, host: FakeHost) -> None:
        # 壊れると、設定で「文字も当てる」にしている人の所で、AI だけ文字を当てない（逆も）
        _yellow_preset(host)
        (clip,) = _with_texts(host, "元の文字")
        host.rules = LookRules(preset_options=PresetOptions(with_text=True))
        run(host, "apply_preset", name="黄色", clip_ids=[str(clip.id)])
        assert _clip(host, clip.id).source.params["text"] == "保存した文字"  # type: ignore[union-attr]

    def test_an_unknown_name_lists_the_saved_ones(self, host: FakeHost) -> None:
        # 壊れると、AI は名前を言い間違えたまま、何を選び直せばよいか分からない
        _yellow_preset(host)
        (clip,) = _with_texts(host, "文字")
        with pytest.raises(ToolError, match="黄色"):
            run(host, "apply_preset", name="青", clip_ids=[str(clip.id)])

    def test_the_same_name_in_two_categories_must_be_told_apart(self, host: FakeHost) -> None:
        # 壊れると、分類違いの同じ名前のどちらかを黙って選び、頼んだのと違う見た目を当てる
        _yellow_preset(host, category="字幕")
        _yellow_preset(host, category="見出し")
        (clip,) = _with_texts(host, "文字")
        with pytest.raises(ToolError, match="分類"):
            run(host, "apply_preset", name="黄色", clip_ids=[str(clip.id)])
        run(host, "apply_preset", name="黄色", category="見出し", clip_ids=[str(clip.id)])

    def test_nothing_to_change_is_told(self, host: FakeHost) -> None:
        # 壊れると、図形へ文字だけの見た目を当てて何も変わらないのに「当てた」と答える
        plain = _text_clip("文字だけ")
        track = Track(TrackKind.VIDEO, "V", (plain,))
        host.library.presets.save(capture_preset("素のまま", None, track, plain))
        shape = with_fixed_items(
            Clip(timeline_start=0, duration=60, source=SHAPE.create()), picture=True
        )
        project = host.document.project
        host.document.reset(
            project.with_timeline(
                replace(
                    project.timeline,
                    tracks=(*project.timeline.tracks, Track(TrackKind.VIDEO, "S", (shape,))),
                )
            )
        )
        before = host.document.project
        # 中身の種類の違う物へは中身を当てない（設定パネルと同じ） 描画の欄はどちらも既定のまま
        with pytest.raises(ToolError, match="変わる所"):
            run(host, "apply_preset", name="素のまま", clip_ids=[str(shape.id)])
        assert host.document.project == before


class TestPlaceAlias:
    def test_it_lands_where_asked_and_undoes_in_one(self, host: FakeHost) -> None:
        # 壊れると、AI が置いたエイリアスが頼んだ時刻に無いか、戻すのに何回も押す
        host.library.aliases.save(_styled_alias("テロップ"))
        before = host.document.project
        result = run(host, "place_alias", name="テロップ", at_frame=45)
        clip = _clip(host, result["clip_id"])
        assert clip.timeline_start == 45
        assert clip.duration == 120
        assert clip.source is not None and clip.source.params["text"] == "見本の文字"
        assert host.document.history_labels[-1] == "エイリアスを置く: テロップ"
        run(host, "undo")
        assert host.document.project == before

    def test_the_playhead_is_the_default(self, host: FakeHost) -> None:
        # 壊れると、「ここに置いて」が通じず、末尾や頭へ置かれる
        host.library.aliases.save(_styled_alias("テロップ"))
        host.seek(200)
        result = run(host, "place_alias", name="テロップ")
        assert _clip(host, result["clip_id"]).timeline_start == 200

    def test_a_free_track_is_used_when_given(self, host: FakeHost) -> None:
        # 壊れると、右クリックしたトラックへ置く人の操作と違い、AI は狙ったトラックへ置けない
        host.library.aliases.save(_styled_alias("テロップ"))
        _with_texts(host, "先にある")
        track = host.document.project.timeline.tracks[-1]
        result = run(host, "place_alias", name="テロップ", at_frame=400, track_id=str(track.id))
        assert result["track_id"] == str(track.id)

    def test_an_unknown_name_lists_the_saved_ones(self, host: FakeHost) -> None:
        # 壊れると、AI は名前を言い間違えたまま、何を選び直せばよいか分からない
        host.library.aliases.save(_styled_alias("テロップ"))
        with pytest.raises(ToolError, match="テロップ"):
            run(host, "place_alias", name="無い")


class TestShelf:
    def test_a_template_is_placed_with_one_undo(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、棚のテンプレートを AI が置けないか、戻すのに何回も押す
        entry = _shelf_entry(tmp_path)
        host.shelf = [entry]
        before = host.document.project
        result = run(host, "place_template", template_id=_template_id(entry), at_frame=400)
        assert result["placed"] == 1
        (clip_id,) = result["clip_ids"]
        clip = _clip(host, clip_id)
        assert clip.timeline_start == 400
        assert clip.source is not None and clip.source.params["size"] == AnimatedValue(88.0)
        assert host.document.history_labels[-1] == "テンプレートを配置"
        run(host, "undo")
        assert host.document.project == before

    def test_a_template_is_worn_by_a_text(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、棚の〔選択中のクリップに適用〕と違い、文字と長さまで見本に替わる
        entry = _shelf_entry(tmp_path)
        host.shelf = [entry]
        (clip,) = _with_texts(host, "本人の字幕")
        run(host, "apply_template", template_id=_template_id(entry), clip_id=str(clip.id))
        worn = _clip(host, clip.id)
        assert worn.source is not None
        assert worn.source.params["text"] == "本人の字幕"
        assert worn.source.params["size"] == AnimatedValue(88.0)
        assert worn.duration == clip.duration
        assert host.document.history_labels[-1] == "テンプレートを適用"
        run(host, "undo")
        assert _clip(host, clip.id) == clip

    def test_only_a_text_wears_a_template(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、映像のクリップへ着せたつもりで何も変わらず、AI はできたと答える
        entry = _shelf_entry(tmp_path)
        host.shelf = [entry]
        video = host.document.project.timeline.tracks[0].clips[0]
        with pytest.raises(ToolError, match="テキスト"):
            run(host, "apply_template", template_id=_template_id(entry), clip_id=str(video.id))

    def test_an_effect_only_template_is_worn_by_any_clip(
        self, host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、画面の棚では動画にも着せられる効果だけのテンプレートを、AI だけ着せられない
        entry = _shelf_entry(tmp_path, name="ぼかし", text=_EFFECT_OBJECT)
        host.shelf = [entry]
        video = host.document.project.timeline.tracks[0].clips[0]
        run(host, "apply_template", template_id=_template_id(entry), clip_id=str(video.id))
        assert "blur" in [effect.kind for effect in _clip(host, video.id).effects]

    def test_an_effect_only_template_with_nothing_read_says_why(
        self, host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、スクリプトが置き場に無くて何も読めなかった物を「テキストにしか」と断り、
        # テキストを選び直しても直らない
        unread = _EFFECT_OBJECT.split("[Object.1]")[0]
        entry = _shelf_entry(tmp_path, name="読めない", text=unread)
        host.shelf = [entry]
        (clip,) = _with_texts(host, "本人の字幕")
        with pytest.raises(ToolError, match="読めた効果"):
            run(host, "apply_template", template_id=_template_id(entry), clip_id=str(clip.id))

    def test_a_crowded_layer_goes_to_new_tracks_in_front(
        self, host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、元のレイヤーの所に動画があるだけで棚の配置が断られる（2026-10-17 本人の決定で
        # 一番手前の新しいトラックへ上下の並びのまま置く）
        entry = _shelf_entry(tmp_path, name="二段", text=_EXO_TWO_LAYERS, suffix=".exo")
        host.shelf = [entry]
        before = host.document.project
        result = run(host, "place_template", template_id=_template_id(entry), at_frame=30)
        assert result["placed"] == 2
        assert _sizes_in_new_tracks(host, before) == [40.0, 80.0]
        run(host, "undo")
        assert host.document.project == before

    def test_a_crowded_layer_in_a_mixed_project_goes_to_new_layers(self, tmp_path: Path) -> None:
        # 壊れると、レイヤーの方式の作品で、元のレイヤーが埋まっているだけで棚の配置が断られる
        settings = ProjectSettings(frame_rate=FrameRate(30), layer_mode=LayerMode.MIXED)
        base = Project.create(settings)
        busy = Track(TrackKind.MIXED, "レイヤー 1", (_text_clip("前からある", duration=300),))
        host = FakeHost(base.with_timeline(replace(base.timeline, tracks=(busy,))))
        entry = _shelf_entry(tmp_path, name="二段", text=_EXO_TWO_LAYERS, suffix=".exo")
        host.shelf = [entry]
        before = host.document.project
        assert (
            run(host, "place_template", template_id=_template_id(entry), at_frame=30)["placed"] == 2
        )
        assert _sizes_in_new_tracks(host, before) == [40.0, 80.0]
        run(host, "undo")
        assert host.document.project == before

    def test_an_unknown_id_says_where_to_look(self, host: FakeHost) -> None:
        # 壊れると、AI は名前を ID の代わりに渡したまま、何を直せばよいか分からない
        with pytest.raises(ToolError, match="list_looks"):
            run(host, "place_template", template_id="大きい")

    def test_an_unreadable_template_is_refused_without_changes(
        self, host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、読めない配布物を置いたつもりで、空のクリップや半端な物が残る
        entry = _shelf_entry(tmp_path)
        (tmp_path / "大きい" / "字幕" / "大きい.object").unlink()
        host.shelf = [entry]
        before = host.document.project
        with pytest.raises(ToolError, match="読み込めません"):
            run(host, "place_template", template_id=_template_id(entry))
        assert host.document.project == before

    def test_subtitles_can_wear_a_shelf_template(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、画面の〔焼き込み〕では選べる棚の見た目を、AI の焼き込みでは選べない
        entry = _shelf_entry(tmp_path)
        host.shelf = [entry]
        result = run(host, "place_subtitles", look_kind="template", look_name=_template_id(entry))
        assert result["placed"] > 0
        placed = [
            clip
            for track in host.document.project.timeline.tracks
            if track.name.startswith("字幕")
            for clip in track.clips
        ]
        assert placed
        assert all(
            clip.source is not None and clip.source.params["size"] == AnimatedValue(88.0)
            for clip in placed
        )


class TestImportObjects:
    def test_a_crowded_layer_goes_to_new_tracks_in_front(
        self, host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、動画のある作品へ AviUtl のファイルを読むと、元のレイヤーの所で重なって断られる
        source = tmp_path / "二段.exo"
        source.write_text(_EXO_TWO_LAYERS, encoding="utf-8")
        before = host.document.project
        assert run(host, "import_objects", path=str(source))["placed"] == 2
        assert _sizes_in_new_tracks(host, before) == [40.0, 80.0]
        run(host, "undo")
        assert host.document.project == before

    def test_a_crowded_layer_in_a_mixed_project_goes_to_new_layers(self, tmp_path: Path) -> None:
        # 壊れると、レイヤーの方式の作品でも、元のレイヤーが埋まっていると読み込みが断られる
        settings = ProjectSettings(frame_rate=FrameRate(30), layer_mode=LayerMode.MIXED)
        base = Project.create(settings)
        busy = Track(TrackKind.MIXED, "レイヤー 1", (_text_clip("前からある", duration=300),))
        host = FakeHost(base.with_timeline(replace(base.timeline, tracks=(busy,))))
        source = tmp_path / "二段.exo"
        source.write_text(_EXO_TWO_LAYERS, encoding="utf-8")
        before = host.document.project
        assert run(host, "import_objects", path=str(source))["placed"] == 2
        assert _sizes_in_new_tracks(host, before) == [40.0, 80.0]
        run(host, "undo")
        assert host.document.project == before

    def test_media_and_clips_undo_together(self, tmp_path: Path) -> None:
        # 壊れると、戻したときに使われていない素材だけが素材一覧に残る
        host = _empty_host()
        host.probe_result = MediaItem(path=tmp_path / "仮.png")
        picture = tmp_path / "絵.png"
        picture.write_bytes(b"")
        source = tmp_path / "素材つき.exo"
        source.write_text(_EXO_IMAGE.format(path=picture), encoding="utf-8")
        before = host.document.project
        result = run(host, "import_objects", path=str(source))
        assert result["placed"] == 1
        assert result["missing_media"] == 0
        assert len(host.document.project.media) == len(before.media) + 1
        assert host.probed == [picture]
        assert host.document.history_labels[-1] == f"AviUtl から読み込み: {source.name}"
        run(host, "undo")
        assert host.document.project == before

    @pytest.mark.parametrize("name", ["メモ.txt", "作品.sme", "鍵"])
    def test_only_object_files_are_read(self, host: FakeHost, tmp_path: Path, name: str) -> None:
        # 壊れると、AI が AviUtl のファイルでない物（本人のメモや設定など）まで開いて読もうとする
        # 人は窓の「すべてのファイル」で自分で選べるが、AI には選ぶ人がいない（CodeRabbit の指摘）
        other = tmp_path / name
        other.write_text(_EXO_TWO_LAYERS, encoding="utf-8")
        before = host.document.project
        with pytest.raises(ToolError, match=r"\.exo"):
            run(host, "import_objects", path=str(other))
        assert host.document.project == before

    def test_the_same_media_is_registered_once(self, tmp_path: Path) -> None:
        # 壊れると、同じ .exo を 2 回読むたびに、同じ画像が別の素材として素材一覧に増え、
        # 解析と控えも 2 回走る（棚の配置は前から使い回していた CodeRabbit の指摘）
        host = _empty_host()
        picture = tmp_path / "絵.png"
        picture.write_bytes(b"")
        host.probe_result = MediaItem(path=picture)
        source = tmp_path / "素材つき.exo"
        source.write_text(_EXO_IMAGE.format(path=picture), encoding="utf-8")
        run(host, "import_objects", path=str(source))
        run(host, "import_objects", path=str(source))
        project = host.document.project
        assert len(project.media) == 1
        linked = {clip.media_id for track in project.timeline.tracks for clip in track.clips}
        assert linked == {project.media[0].id}
        assert host.probed == [picture]

    def test_a_relative_media_path_is_read_from_the_exo_s_folder(self, tmp_path: Path) -> None:
        # 壊れると、.exo の隣のフォルダに添えた素材が見つからず、同じ名前の別の絵を結ぶか、
        # 素材の無いクリップを置く（CodeRabbit の指摘）
        host = _empty_host()
        folder = tmp_path / "配布物"
        (folder / "素材").mkdir(parents=True)
        wanted = folder / "素材" / "絵.png"
        wanted.write_bytes(b"wanted")
        # 同じ名前の別の絵 名前だけで探すと、こちらを結んでしまう
        (folder / "絵.png").write_bytes(b"decoy")
        host.probe_result = MediaItem(path=wanted)
        source = folder / "素材つき.exo"
        source.write_text(_EXO_IMAGE.format(path="素材\\絵.png"), encoding="utf-8")
        result = run(host, "import_objects", path=str(source))
        assert result["missing_media"] == 0
        assert host.probed == [wanted]

    def test_a_missing_file_is_refused(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、無いファイルを読んだつもりで「読み込んだ」と答える
        with pytest.raises(ToolError, match="ありません"):
            run(host, "import_objects", path=str(tmp_path / "無い.exo"))

    def test_a_broken_file_is_refused(self, host: FakeHost, tmp_path: Path) -> None:
        # 壊れると、読めないファイルで道具ごと落ち、AI は何が悪いのか分からない
        broken = tmp_path / "壊れた.exo"
        broken.write_text("中身ではない", encoding="utf-8")
        before = host.document.project
        with pytest.raises(ToolError):
            run(host, "import_objects", path=str(broken))
        assert host.document.project == before
