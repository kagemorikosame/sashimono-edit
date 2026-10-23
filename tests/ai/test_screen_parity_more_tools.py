"""画面と同じ道具の続き（#349） 作品の設定・置き方の方式・前後のキー・スクリプト・
カスタムオブジェクト・配置のテンプレート・標準の組み方

どれも画面の同じ操作と同じ命令を通り、取り消し 1 回で戻ることを偽のホストで見る
"""

from __future__ import annotations

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import custom_object_key, find_operation
from sashimono.ai.outbound import ASKED, Outbound, SendChoice
from sashimono.ai.safety import Safety
from sashimono.compat.aviutl.catalog import ScriptCatalog
from sashimono.core.commands import AddClip, ParamPath, SetParam
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import (
    AnimatedValue,
    Blending,
    Clip,
    ClipId,
    LayerMode,
    Project,
    TrackKind,
)
from sashimono.effects.sources import TEXT, TO_NATIVE_LAYOUT
from sashimono.effects.unlock import unlock_commands
from tests.ai.conftest import FakeHost
from tests.ai.test_bridge import call_in_thread
from tests.ai.test_operations import run
from tests.ai.test_safety import answer_approvals


def _text_clip(host: FakeHost, start: int = 400, **params: str) -> Clip:
    clip = with_fixed_items(
        Clip(timeline_start=start, duration=90, source=TEXT.create(text="字", **params)),
        picture=True,
    )
    track = host.document.project.timeline.tracks[0]
    host.apply_commands([AddClip(track.id, clip)], "置く")
    return clip


def _undo_returns(host: FakeHost, before: Project) -> None:
    host.document.undo()
    assert host.document.project == before


class TestLayerMode:
    def test_switching_with_conversion_changes_every_track_in_one_undo(
        self, host: FakeHost
    ) -> None:
        # 壊れると、AI に「レイヤーの方式にして」と頼んでも置いた物が前の方式のトラックに残る
        before = host.document.project
        result = run(host, "switch_layer_mode", convert=True)
        project = host.document.project
        assert result["layer_mode"] == LayerMode.MIXED
        assert project.settings.layer_mode == LayerMode.MIXED
        assert {track.kind for track in project.timeline.tracks} == {TrackKind.MIXED}
        _undo_returns(host, before)

    def test_switching_without_conversion_leaves_the_tracks(self, host: FakeHost) -> None:
        # 壊れると、「これから置く物だけ」と頼んだのに置いてあるトラックまで作り替わる
        tracks = host.document.project.timeline.tracks
        run(host, "switch_layer_mode", convert=False)
        assert host.document.project.settings.layer_mode == LayerMode.MIXED
        assert host.document.project.timeline.tracks == tracks

    def test_an_open_scene_or_an_unanswered_question_is_refused(self, host: FakeHost) -> None:
        # 壊れると、シーンの中だけが変わってメインが前の方式に残るか、変換するかを決めずに進む
        before = host.document.project
        with pytest.raises(ToolError, match="convert"):
            run(host, "switch_layer_mode")
        run(host, "add_scene", name="場面")
        with pytest.raises(ToolError, match="メイン"):
            run(host, "switch_layer_mode", convert=True)
        assert host.document.project.settings == before.settings


class TestProjectSettings:
    def test_size_and_blending_change_in_one_undo(self, host: FakeHost) -> None:
        # 壊れると、プロジェクト設定の窓の OK と違い、取り消しの途中で解像度だけ戻った形を通る
        before = host.document.project
        result = run(host, "set_project_settings", width=1080, height=1920, blending="linear")
        assert result == {"resolution": "1080x1920", "blending": Blending.LINEAR}
        _undo_returns(host, before)

    def test_nothing_to_change_and_unknown_blending_are_refused(self, host: FakeHost) -> None:
        # 壊れると、何も変わらない段が履歴に積まれるか、描き方の分からない値が入る
        with pytest.raises(ToolError, match="変わる所"):
            run(host, "set_project_settings", width=1920, height=1080)
        with pytest.raises(ToolError, match="blending"):
            run(host, "set_project_settings", blending="multiply")
        assert not host.document.can_undo


class TestSeekingKeyframes:
    def test_the_playhead_jumps_to_the_keys_inside_the_clip(self, host: FakeHost) -> None:
        # 壊れると、AI がキーの所の絵を確かめるのに、キーの位置を数えて seek し直すことになる
        clip = _text_clip(host, start=400)
        for frame, value in ((0, 40), (30, 60), (60, 80), (200, 99)):
            run(host, "add_keyframe", clip_id=str(clip.id), name="size", frame=frame, value=value)
        host.seek(440)
        assert run(host, "seek_keyframe", clip_id=str(clip.id), name="size")["playhead"] == 460
        assert host.playhead == 460
        # クリップの外（200 フレーム目）のキーへは飛ばない
        with pytest.raises(ToolError, match="次にキーがありません"):
            run(host, "seek_keyframe", clip_id=str(clip.id), name="size", direction="next")
        host.seek(440)
        back = run(host, "seek_keyframe", clip_id=str(clip.id), name="size", direction="previous")
        assert back["playhead"] == 430
        host.seek(400)
        with pytest.raises(ToolError, match="前にキーがありません"):
            run(host, "seek_keyframe", clip_id=str(clip.id), name="size", direction="previous")


class TestScriptsAndCustomObjects:
    def test_reloading_reads_the_folder_again(self, host: FakeHost) -> None:
        # 壊れると、置き場へ足したスクリプトを使うのに、人がアプリを開き直すことになる
        assert run(host, "reload_scripts") == {"loaded": 0}
        assert host.reloads == 1

    def test_a_custom_object_is_placed_like_the_add_menu(self, host: FakeHost) -> None:
        # 壊れると、AI が配布スクリプトの .obj を置けないか、画面と違う形のクリップになる
        catalog = ScriptCatalog(roots=())
        entry = catalog.add_text(
            "aviutl:試験.obj:円", "@円\n--track0:大きさ,0,100,50\n", kind="obj"
        )
        host.scripts = [entry]
        assert run(host, "list_custom_objects")["custom_objects"] == [
            {"key": custom_object_key(entry), "name": "円", "category": entry.category}
        ]
        before = host.document.project
        placed = run(host, "add_custom_object", name="円", at_frame=400)
        located = host.document.project.timeline.locate_clip(ClipId(placed["clip_id"]))
        assert located is not None
        clip = located[1]
        # スクリプトの後ろに描画の欄（反転・配置）が付く（画面の〔追加〕と同じ）
        assert [e.kind for e in clip.effects] == [entry.identifier, "flip", "transform"]
        assert host.selected_clip == clip.id
        _undo_returns(host, before)

    def test_objects_sharing_a_name_are_picked_by_their_key(self, host: FakeHost) -> None:
        # 壊れると、別の .obj と .obj2 に同じ名前の節があるだけで、画面のメニューでは
        # どちらも置けるのに AI からはどちらも置けない
        catalog = ScriptCatalog(roots=())
        first = catalog.add_text("aviutl:私の素材/一.obj:円", "@円\n", kind="obj")
        second = catalog.add_text("aviutl:私の素材/二.obj2:円", "@円\n", kind="obj")
        host.scripts = [first, second]
        rows = run(host, "list_custom_objects")["custom_objects"]
        keys = [row["key"] for row in rows]
        assert len(set(keys)) == 2
        # 鍵に置き場のフォルダやファイルの名前を出さない（#391 の名前と場所は伏せる側）
        assert not any(part in key for key in keys for part in ("私の素材", "一.obj", "二.obj2"))
        placed = run(host, "add_custom_object", key=keys[1], at_frame=400)
        located = host.document.project.timeline.locate_clip(ClipId(placed["clip_id"]))
        assert located is not None
        assert located[1].effects[0].kind == second.identifier

    def test_a_shared_name_is_refused_with_the_keys_to_choose_from(self, host: FakeHost) -> None:
        # 壊れると、名前で頼んだ AI は断られても次に何を渡せば置けるのか分からない
        catalog = ScriptCatalog(roots=())
        host.scripts = [
            catalog.add_text("aviutl:一.obj:円", "@円\n", kind="obj"),
            catalog.add_text("aviutl:二.obj2:円", "@円\n", kind="obj"),
        ]
        keys = [row["key"] for row in run(host, "list_custom_objects")["custom_objects"]]
        before = host.document.project
        with pytest.raises(ToolError) as caught:
            run(host, "add_custom_object", name="円")
        assert all(key in str(caught.value) for key in keys)
        assert host.document.project is before

    def test_an_unknown_name_lists_what_can_be_placed(self, host: FakeHost) -> None:
        # 壊れると、AI は無い名前で置こうとし続け、何なら置けるのかが分からない
        catalog = ScriptCatalog(roots=())
        host.scripts = [catalog.add_text("aviutl:試験.obj:星", "@星\n", kind="obj")]
        with pytest.raises(ToolError, match="星"):
            run(host, "add_custom_object", name="円")


class TestAlignment:
    def test_the_commands_from_the_preview_are_applied_in_one_undo(self, host: FakeHost) -> None:
        # 壊れると、設定パネルの ↖〜↘ でできる寄せを AI ができず、位置を数で探ることになる
        clip = _text_clip(host)
        moved = SetParam(ParamPath.of_source(clip.id, "size"), AnimatedValue(80.0))
        host.aligned = [moved]
        before = host.document.project
        assert run(host, "align_clip", clip_id=str(clip.id), anchor="top_left")["moved"]
        assert host.alignments == [(clip.id, "top_left")]
        _undo_returns(host, before)

    def test_a_clip_not_on_screen_is_refused(self, host: FakeHost) -> None:
        # 壊れると、映っていない物を揃えたと答え、AI は寄せたつもりで次へ進む
        clip = _text_clip(host)
        with pytest.raises(ToolError, match="映っていない"):
            run(host, "align_clip", clip_id=str(clip.id), anchor="center")
        with pytest.raises(ToolError, match="anchor"):
            run(host, "align_clip", clip_id=str(clip.id), anchor="middle")


class TestStandardLayout:
    def test_only_greyed_texts_switch_like_the_panel_button(self, host: FakeHost) -> None:
        # 壊れると、AI の切り替えと設定パネルのボタンで入る値が違うか、灰色でない字まで動く
        greyed = _text_clip(host, start=400, layout="aviutl", align="right")
        native = _text_clip(host, start=600)
        expected, _ = unlock_commands([greyed], TEXT, TO_NATIVE_LAYOUT)
        before = host.document.project
        result = run(host, "use_standard_layout", clip_ids=[str(greyed.id), str(native.id)])
        assert result == {"switched": 1}
        by_ai = host.document.project
        _undo_returns(host, before)
        host.apply_commands(expected, "ボタン")
        assert host.document.project == by_ai

    def test_nothing_greyed_is_refused(self, host: FakeHost) -> None:
        # 壊れると、何も変わらない段が積まれ、AI は切り替えたつもりで次へ進む
        native = _text_clip(host)
        with pytest.raises(ToolError, match="灰色"):
            run(host, "use_standard_layout", clip_ids=[str(native.id)])


class TestCopyingOutOfTheApp:
    """OS のクリップボードへ写す道具（2026-10-20 利用者の決定 読むだけ 送る前の同意に乗る）"""

    def test_a_snapshot_is_copied_without_changing_the_project(self, host: FakeHost) -> None:
        # 壊れると、AI に「この絵をコピーして」と頼んでも貼れないか、終わりの黒い絵が入る
        before = host.document.project
        result = run(host, "copy_snapshot", frame=10_000)
        assert result["copied_frame"] == 299
        assert host.copied_frames == [299]
        assert host.document.project is before
        assert not host.document.can_undo

    def test_a_subtitle_line_is_copied(self, host: FakeHost) -> None:
        # 壊れると、字幕パネルの〔文字をコピー〕と同じことを AI に頼めない
        media = host.document.project.media[0]
        assert media.transcript is not None
        segment = media.transcript.segments[1]
        result = run(host, "copy_subtitle_text", media_id=str(media.id), segment_id=str(segment.id))
        assert host.copied_texts == [segment.text]
        assert result == {"segment_id": str(segment.id), "copied_chars": len(segment.text)}

    @pytest.mark.parametrize(
        ("tool", "kind"),
        [("copy_snapshot", Outbound.FRAMES), ("copy_subtitle_text", Outbound.TEXT)],
    )
    def test_they_only_read_and_declare_what_leaves_the_app(
        self, tool: str, kind: Outbound
    ) -> None:
        # 壊れると、絵や文字をアプリの外へ出すのに送る前の同意に乗らないか、確かめの窓が出る
        operation = find_operation(tool)
        assert operation is not None
        assert operation.safety is Safety.READ_ONLY
        assert kind in operation.sends

    @pytest.mark.parametrize("kind", [Outbound.FRAMES, Outbound.TEXT])
    def test_a_refused_kind_is_not_copied(self, host: FakeHost, kind: Outbound) -> None:
        # 壊れると、絵や字幕を送らないと決めた人の所でも、AI が黙ってクリップボードへ写す
        bridge = EditorBridge(host)
        bridge.set_send_choices({**dict.fromkeys(ASKED, SendChoice.ALLOW), kind: SendChoice.DENY})
        media = host.document.project.media[0]
        assert media.transcript is not None
        tool = "copy_snapshot" if kind is Outbound.FRAMES else "copy_subtitle_text"
        arguments: dict[str, object] = (
            {}
            if kind is Outbound.FRAMES
            else {"media_id": str(media.id), "segment_id": str(media.transcript.segments[0].id)}
        )
        operation = find_operation(tool)
        assert operation is not None
        done, box = call_in_thread(bridge, operation, arguments)
        answer_approvals(bridge, done, lambda _approval: False)
        assert isinstance(box.get("error"), ToolError)
        assert host.copied_frames == [] and host.copied_texts == []
