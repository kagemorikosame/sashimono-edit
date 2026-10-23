"""画面でできて AI にできなかった操作の道具（#349 E3）

どの道具も画面の同じ操作と同じ命令を通り、取り消し 1 回で戻ることを偽のホストで見る
本物の窓とつないだ所（人のコピーと AI の貼り付けが同じ置き場か など）は
``tests/ui/test_ai_screen_parity.py``
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.core.clipboard import ClipboardContent
from sashimono.core.commands import (
    AddClip,
    AddScene,
    AddTrack,
    ParamPath,
    RemoveClip,
    SetKeyframe,
    insert_scene,
    new_scene,
)
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import (
    FILTER_KIND,
    AnimatedValue,
    Clip,
    ClipId,
    Project,
    Track,
    TrackKind,
)
from sashimono.core.projection import source_seconds_at
from sashimono.effects import registry
from sashimono.effects.sources import GROUP, TEXT, TRANSITION
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run


def _clips(host: FakeHost) -> list[Clip]:
    return [clip for track in host.document.project.timeline.tracks for clip in track.clips]


def _first(host: FakeHost) -> Clip:
    return host.document.project.timeline.tracks[0].clips[0]


def _text_clip(host: FakeHost, start: int = 400, *, effects: tuple[str, ...] = ()) -> Clip:
    """テキストを 1 本、映像トラックの ``start`` へ置く 積むエフェクトの種類を渡せる"""
    clip = with_fixed_items(
        Clip(
            timeline_start=start,
            duration=90,
            source=TEXT.create(text="字"),
            effects=tuple(registry.require(kind).create() for kind in effects),
        ),
        picture=True,
    )
    track = host.document.project.timeline.tracks[0]
    host.apply_commands([AddClip(track.id, clip)], "置く")
    return clip


def _undo_returns(host: FakeHost, before: Project) -> None:
    host.document.undo()
    assert host.document.project == before


class TestSelectAll:
    def test_every_clip_of_every_track_is_selected(self, host: FakeHost) -> None:
        # 壊れると「全部まとめて 2 秒後ろへ」と頼んでも、AI が ID を数え漏らした分が残る
        _text_clip(host)
        audio = Track(TrackKind.AUDIO, "A1")
        host.apply_commands([AddTrack(audio)], "足す")
        run(host, "select_all")
        assert set(host.selected_clips) == {clip.id for clip in _clips(host)}
        assert len(host.selected_clips) == 2


class TestClipboard:
    def test_copy_then_paste_puts_a_copy_and_keeps_the_original(self, host: FakeHost) -> None:
        # 壊れると、AI がコピーした物を貼れないか、貼ると元のクリップが消える
        original = _first(host)
        run(host, "copy_clips", clip_ids=[str(original.id)])
        assert isinstance(host.copied, ClipboardContent)
        before = host.document.project
        result = run(host, "paste_clips", at_frame=600)
        (pasted,) = result["pasted"]
        assert _first(host) == original
        located = host.document.project.timeline.locate_clip(ClipId(pasted))
        assert located is not None and located[1].timeline_start == 600
        assert host.selected_clips == (ClipId(pasted),)
        _undo_returns(host, before)

    def test_copying_changes_nothing_in_the_project(self, host: FakeHost) -> None:
        # 壊れると、コピーしただけで履歴に段が積まれ、人の Ctrl+Z が 1 回空振りする
        before = host.document.project
        run(host, "copy_clips", clip_ids=[str(_first(host).id)])
        assert host.document.project == before
        assert not host.document.can_undo

    def test_cut_removes_and_one_undo_brings_it_back(self, host: FakeHost) -> None:
        # 壊れると、切り取った物が戻らないか、戻すのに何回も取り消すことになる
        before = host.document.project
        clip = _first(host)
        run(host, "cut_clips", clip_ids=[str(clip.id)])
        assert _clips(host) == []
        assert host.copied is not None and host.copied.clips[0].clip == clip
        _undo_returns(host, before)

    def test_a_cut_that_is_refused_keeps_the_person_s_copy(self, host: FakeHost) -> None:
        # 壊れると、ロックで断られたのに置き場だけ書き換わり、人が前にコピーした物が黙って消える
        clip = _first(host)
        run(host, "copy_clips", clip_ids=[str(clip.id)])
        kept = host.copied
        track = host.document.project.timeline.tracks[0]
        run(host, "set_track_state", track_id=str(track.id), locked=True)
        with pytest.raises(ToolError):
            run(host, "cut_clips", clip_ids=[str(clip.id)])
        assert host.copied is kept

    def test_a_group_with_a_locked_member_is_not_cut(self, host: FakeHost) -> None:
        # 壊れると、束ねた物の一部だけが消え、グループの頭や長さが食い違う（画面は断る）
        text = _text_clip(host)
        locked = Track(TrackKind.VIDEO, "V2")
        other = replace(text, id=ClipId("ほか"), timeline_start=0)
        host.apply_commands([AddTrack(locked), AddClip(locked.id, other)], "足す")
        run(host, "group_clips", clip_ids=[str(text.id), str(other.id)])
        run(host, "set_track_state", track_id=str(locked.id), locked=True)
        before = host.document.project
        with pytest.raises(ToolError, match="ロック"):
            run(host, "cut_clips", clip_ids=[str(text.id)])
        assert host.document.project == before

    def test_pasting_with_nothing_copied_says_what_to_do(self, host: FakeHost) -> None:
        # 壊れると、貼る物が無いのに黙って成功を返し、AI は貼ったつもりで次へ進む
        with pytest.raises(ToolError, match="copy_clips"):
            run(host, "paste_clips")

    def test_insert_paste_pushes_what_follows(self, host: FakeHost) -> None:
        # 壊れると〔貼り付け（挿入）〕を頼んでも後ろが押されず、貼る所が塞がって別のトラックへ逃げる
        clip = _first(host)
        run(host, "copy_clips", clip_ids=[str(clip.id)])
        before = host.document.project
        run(host, "paste_clips", at_frame=0, insert=True)
        track = host.document.project.timeline.tracks[0]
        assert [c.timeline_start for c in track.clips] == [0, 300]
        assert len(host.document.project.timeline.tracks) == 1
        _undo_returns(host, before)

    @pytest.mark.parametrize(("all_tracks", "pushed_to"), [(True, 400), (False, 100)])
    def test_insert_paste_follows_the_person_s_setting(
        self, host: FakeHost, all_tracks: bool, pushed_to: int
    ) -> None:
        # 壊れると、設定で「コピー元のトラックだけ押す」にした人の所で、AI だけほかのトラックの
        # テロップまで押し、貼った所の後ろで字幕と絵がずれる（逆の設定でも同じ）
        other = Track(TrackKind.VIDEO, "V2")
        caption = replace(_text_clip(host, start=500), id=ClipId("テロップ"), timeline_start=100)
        host.apply_commands([AddTrack(other), AddClip(other.id, caption)], "足す")
        host.insert_all_tracks = all_tracks
        run(host, "copy_clips", clip_ids=[str(_first(host).id)])
        run(host, "paste_clips", at_frame=0, insert=True)
        located = host.document.project.timeline.locate_clip(caption.id)
        assert located is not None and located[1].timeline_start == pushed_to


class TestTracks:
    def test_an_empty_track_is_removed_and_comes_back(self, host: FakeHost) -> None:
        # 壊れると、AI が足した空のトラックを片付けられず、レイヤーの数が増えたまま残る
        added = run(host, "add_track", kind="audio")["track_id"]
        before = host.document.project
        run(host, "remove_track", track_id=added)
        assert [str(t.id) for t in host.document.project.timeline.tracks] != []
        assert host.document.project.timeline.find_track(added) is None
        _undo_returns(host, before)

    def test_a_track_is_renamed_and_an_empty_name_goes_back_to_the_default(
        self, host: FakeHost
    ) -> None:
        # 壊れると、空の名前を頼んだ AI が名前の無いトラックを作ったと思い込む（実際は既定の名前）
        track = host.document.project.timeline.tracks[0]
        assert run(host, "rename_track", track_id=str(track.id), name="本編")["name"] == "本編"
        assert run(host, "rename_track", track_id=str(track.id), name=" ")["name"] == "V1"

    def test_a_track_moves_within_its_kind(self, host: FakeHost) -> None:
        # 壊れると「字幕のトラックを一番手前へ」と頼んでも重なりの順が変わらない
        first = host.document.project.timeline.tracks[0]
        second = run(host, "add_track", kind="video")["track_id"]
        before = host.document.project
        order = run(host, "move_track", track_id=str(first.id), index=1)["order"]
        assert order == [second, str(first.id)]
        _undo_returns(host, before)

    def test_an_effect_track_is_named_like_the_screen_makes_it(self, host: FakeHost) -> None:
        # 壊れると、AI が足したトラックがフィルタ用と見なされず、フィルタが一部にしか掛からない
        assert run(host, "add_track", kind="video", effect=True)["name"] == "FX1"
        with pytest.raises(ToolError):
            run(host, "add_track", kind="audio", effect=True)


class TestScenes:
    def _scene(self, host: FakeHost, name: str = "場面") -> str:
        return str(run(host, "add_scene", name=name)["scene_id"])

    def test_the_open_scene_is_renamed_when_none_is_named(self, host: FakeHost) -> None:
        # 壊れると「このシーンの名前を変えて」と頼んでも、どのシーンか分からず断られる
        scene_id = self._scene(host)
        before = host.document.project
        run(host, "rename_scene", name="冒頭")
        assert [s.name for s in host.document.project.scenes] == ["冒頭"]
        assert host.active_scene is not None and str(host.active_scene) == scene_id
        _undo_returns(host, before)

    def test_the_main_timeline_cannot_be_renamed(self, host: FakeHost) -> None:
        # 壊れると、メインを開いたまま頼んだ名前の変更が、黙ってどこにも当たらない
        with pytest.raises(ToolError, match="メイン"):
            run(host, "rename_scene", name="冒頭")

    def test_removing_the_open_scene_goes_back_to_the_main(self, host: FakeHost) -> None:
        # 壊れると、開いたシーンを消した後もそのシーンを開いたままになり、次の道具が全部断られる
        self._scene(host)
        before = host.document.project
        result = run(host, "remove_scene")
        assert host.document.project.scenes == ()
        assert host.active_scene is None and result["active_scene"] is None
        _undo_returns(host, before)

    def test_a_scene_placed_on_the_main_is_kept_even_from_another_scene(
        self, host: FakeHost
    ) -> None:
        # 壊れると、別のシーンを開いているときだけメインに置いたシーンを消せ、メインに穴が空く
        placed = new_scene(host.document.project, "置いた")
        host.apply_commands([AddScene(placed)], "足す")
        host.apply_commands(insert_scene(host.document.project, placed.id, at_frame=400), "置く")
        self._scene(host, "開いている")
        before = host.document.project
        with pytest.raises(ToolError, match="メイン"):
            run(host, "remove_scene", scene_id=str(placed.id))
        assert host.document.project == before


class TestWorkArea:
    def test_the_range_is_set_and_cleared_one_undo_each(self, host: FakeHost) -> None:
        # 壊れると、AI が「ここだけ書き出す」範囲を決められず、書き出しの work_area が使えない
        before = host.document.project
        run(host, "set_work_area", start_frame=30, end_frame=90)
        assert host.document.project.timeline.work_area == (30, 90)
        middle = host.document.project
        run(host, "set_work_area", clear=True)
        assert host.document.project.timeline.work_area is None
        _undo_returns(host, middle)
        _undo_returns(host, before)

    def test_clearing_without_a_range_is_refused(self, host: FakeHost) -> None:
        # 壊れると、何も変わらない操作が履歴に積まれ、人の Ctrl+Z が空振りする
        with pytest.raises(ToolError, match="指定されていません"):
            run(host, "set_work_area", clear=True)
        assert not host.document.can_undo


class TestPlacingFilters:
    @pytest.mark.parametrize(
        ("tool", "kind"), [("add_filter", FILTER_KIND), ("add_group_control", GROUP.kind)]
    )
    def test_it_is_placed_at_the_playhead_and_selected(
        self, host: FakeHost, tool: str, kind: str
    ) -> None:
        # 壊れると、置いた物を AI が指せず（ID が返らない）、続けてエフェクトや値を足せない
        host.seek(120)
        before = host.document.project
        result = run(host, tool)
        located = host.document.project.timeline.locate_clip(ClipId(result["clip_id"]))
        assert located is not None
        clip = located[1]
        assert clip.source is not None and clip.source.kind == kind
        assert clip.timeline_start == 120
        assert [e for e in clip.effects if not e.fixed] == []
        assert host.selected_clip == clip.id
        _undo_returns(host, before)


class TestEffects:
    def test_an_effect_is_removed_moved_and_switched(self, host: FakeHost) -> None:
        # 壊れると、AI が掛けたぼかしを外せず、掛ける順も変えられず、見比べるために切れない
        clip = _text_clip(host, effects=("blur", "color_correct"))
        blur, bright = (e for e in clip.effects if not e.fixed)
        target = {"clip_id": str(clip.id)}
        before = host.document.project
        order = run(host, "move_effect", **target, effect_id=str(bright.id), index=0)["order"]
        assert order.index(str(bright.id)) < order.index(str(blur.id))
        _undo_returns(host, before)
        run(host, "set_effect_enabled", **target, effect_ids=[str(blur.id)], enabled=False)
        _, switched = host.document.project.timeline.locate_clip(clip.id) or (None, clip)
        assert [e.enabled for e in switched.effects if e.id == blur.id] == [False]
        _undo_returns(host, before)
        run(host, "remove_effect", **target, effect_id=str(blur.id))
        _, removed = host.document.project.timeline.locate_clip(clip.id) or (None, clip)
        assert blur.id not in {e.id for e in removed.effects}
        _undo_returns(host, before)

    def test_a_fixed_item_is_not_removed(self, host: FakeHost) -> None:
        # 壊れると、クリップが最初から持つ欄が消え、設定パネルの並びがクリップごとに崩れる
        clip = _text_clip(host)
        fixed = next(e for e in clip.effects if e.fixed)
        with pytest.raises(ToolError, match="set_effect_enabled"):
            run(host, "remove_effect", clip_id=str(clip.id), effect_id=str(fixed.id))

    def test_a_whole_group_of_fixed_items_switches_in_one_undo(self, host: FakeHost) -> None:
        # 壊れると、組の〔有効〕と同じことを頼むのに欄の数だけ取り消しが要る
        clip = _text_clip(host)
        fixed = [str(e.id) for e in clip.effects if e.fixed]
        before = host.document.project
        run(host, "set_effect_enabled", clip_id=str(clip.id), effect_ids=fixed, enabled=False)
        _undo_returns(host, before)

    def test_an_unknown_effect_is_refused_rather_than_ignored(self, host: FakeHost) -> None:
        # 壊れると（命令は無い ID を黙って飛ばす）、AI は切ったつもりで何も変わらない
        clip = _text_clip(host)
        with pytest.raises(ToolError, match="list_clips"):
            run(host, "set_effect_enabled", clip_id=str(clip.id), effect_id="無い", enabled=False)

    def test_effects_go_to_the_scene_after_a_transition(self, host: FakeHost) -> None:
        # 壊れると、場面切り替えの後の場面へ掛ける色やぼかしを AI が積めず、前の場面にしか掛からない
        clip = Clip(timeline_start=400, duration=30, source=TRANSITION.create())
        track = host.document.project.timeline.tracks[0]
        host.apply_commands([AddClip(track.id, clip)], "置く")
        added = run(host, "add_effect", clip_id=str(clip.id), kind="blur", after=True)
        _, placed = host.document.project.timeline.locate_clip(clip.id) or (None, clip)
        assert [str(e.id) for e in placed.after_effects] == [added["effect_id"]]
        listed = next(
            row for row in run(host, "list_clips")["clips"] if row["clip_id"] == str(clip.id)
        )
        assert [row["effect_id"] for row in listed["after_effects"]] == [added["effect_id"]]
        run(
            host,
            "set_param",
            clip_id=str(clip.id),
            effect_id=added["effect_id"],
            after=True,
            name="radius",
            value=12,
        )
        _, changed = host.document.project.timeline.locate_clip(clip.id) or (None, clip)
        radius = changed.after_effects[0].params["radius"]
        assert isinstance(radius, AnimatedValue) and radius.static == 12
        run(host, "remove_effect", clip_id=str(clip.id), effect_id=added["effect_id"], after=True)

    def test_only_a_transition_has_a_scene_after(self, host: FakeHost) -> None:
        # 壊れると、ふつうのクリップの描かれない置き場へエフェクトが積まれ、掛けたのに絵が変わらない
        clip = _text_clip(host)
        with pytest.raises(ToolError, match="場面切り替え"):
            run(host, "add_effect", clip_id=str(clip.id), kind="blur", after=True)


class TestKeyframes:
    def test_a_key_at_the_playhead_is_counted_from_the_head_of_the_clip(
        self, host: FakeHost
    ) -> None:
        # 壊れると、頭が 0 でないクリップで位置を省いたキーが、再生ヘッドと別の所に付く
        clip = _text_clip(host, start=400)
        host.seek(430)
        run(host, "add_keyframe", clip_id=str(clip.id), name="size", value=80)
        _, keyed = host.document.project.timeline.locate_clip(clip.id) or (None, clip)
        size = keyed.source.params["size"] if keyed.source is not None else None
        assert isinstance(size, AnimatedValue)
        assert [key.frame for key in size.keyframes] == [30]

    def test_a_key_is_removed_and_a_missing_one_names_where_keys_are(self, host: FakeHost) -> None:
        # 壊れると、AI が打ったキーを消せず、外した所を指すと黙って何も変わらない
        clip = _text_clip(host)
        target = {"clip_id": str(clip.id), "name": "size"}
        run(host, "add_keyframe", **target, frame=0, value=40)
        run(host, "add_keyframe", **target, frame=60, value=80)
        before = host.document.project
        with pytest.raises(ToolError, match=r"\[0, 60\]"):
            run(host, "remove_keyframe", **target, frame=30)
        assert run(host, "remove_keyframe", **target, frame=60)["left"] == [0]
        _undo_returns(host, before)

    def test_clearing_fixes_the_value_where_asked(self, host: FakeHost) -> None:
        # 壊れると、アニメーションを解いた値が頼んだ所の値と違い、絵が跳ぶ
        clip = _text_clip(host)
        target = {"clip_id": str(clip.id), "name": "size"}
        run(host, "add_keyframe", **target, frame=0, value=40)
        run(host, "add_keyframe", **target, frame=60, value=80)
        before = host.document.project
        run(host, "clear_keyframes", **target, frame=30)
        _, cleared = host.document.project.timeline.locate_clip(clip.id) or (None, clip)
        size = cleared.source.params["size"] if cleared.source is not None else None
        assert isinstance(size, AnimatedValue)
        assert size.keyframes == () and size.static == pytest.approx(60)
        _undo_returns(host, before)
        run(host, "clear_keyframes", **target, frame=30)
        with pytest.raises(ToolError, match="ありません"):
            run(host, "clear_keyframes", **target)

    def test_the_keyframe_command_matches_the_panel(self, host: FakeHost) -> None:
        # 壊れると、AI のキーと設定パネルの ◆ のキーが別の形で入り、パネルで消せない
        clip = _text_clip(host)
        run(host, "add_keyframe", clip_id=str(clip.id), name="size", frame=10, value=50)
        by_ai = host.document.project
        host.document.undo()
        path = ParamPath.of_source(clip.id, "size")
        host.apply_commands([SetKeyframe(path, 10, 50.0)], "◆")
        assert host.document.project == by_ai


class TestSubtitles:
    def _target(self, host: FakeHost, index: int = 0) -> dict[str, Any]:
        media = host.document.project.media[0]
        assert media.transcript is not None
        return {"media_id": str(media.id), "segment_id": str(media.transcript.segments[index].id)}

    def _texts(self, host: FakeHost) -> list[str]:
        transcript = host.document.project.media[0].transcript
        assert transcript is not None
        return [segment.text for segment in transcript.segments]

    def test_split_merge_and_delete_each_undo_in_one_step(self, host: FakeHost) -> None:
        # 壊れると、字幕パネルの右クリックでできる直しを AI ができず、人が 1 枚ずつ直す
        before = host.document.project
        split = run(host, "split_subtitle", **self._target(host), at_seconds=2)
        assert len(split["segment_ids"]) == 2 and len(self._texts(host)) == 4
        _undo_returns(host, before)
        merged = run(host, "merge_subtitles", **self._target(host))
        assert len(merged["merged"]) == 1
        assert self._texts(host)[0] == "今日は 編集ソフトを"
        _undo_returns(host, before)
        run(host, "delete_subtitle", **self._target(host, 2))
        assert self._texts(host) == ["今日は", "編集ソフトを"]
        _undo_returns(host, before)

    def test_split_at_the_playhead_uses_the_same_time_as_the_panel(self, host: FakeHost) -> None:
        # 壊れると、AI が割る所と字幕パネルの〔再生ヘッドで分割〕が割る所が食い違う
        track = host.document.project.timeline.tracks[0]
        moved = replace(track.clips[0], timeline_start=0, source_in=Fraction(1, 2))
        host.apply_commands([RemoveClip(track.clips[0].id), AddClip(track.id, moved)], "ずらす")
        host.seek(45)
        media = host.document.project.media[0]
        expected = source_seconds_at(host.document.project, media, None, 45)
        result = run(host, "split_subtitle", **self._target(host))
        assert expected == Fraction(2)
        assert result["at_seconds"] == 2.0

    def test_a_split_outside_the_line_is_refused(self, host: FakeHost) -> None:
        # 壊れると、字幕の外の秒で割ろうとして黙って何も起きず、AI は割れたと思い込む
        with pytest.raises(ToolError, match="source_start_seconds"):
            run(host, "split_subtitle", **self._target(host), at_seconds=5)

    def test_a_media_without_subtitles_is_refused_without_its_name(self, host: FakeHost) -> None:
        # 壊れると、名前と場所を送らないと決めた人の所で、断る文に素材の名前が載って外へ出る
        media = host.document.project.media[0]
        bare = FakeHost(host.document.project.replace_media(media.with_transcript(None)))
        with pytest.raises(ToolError) as caught:
            run(bare, "delete_subtitle", media_id=str(media.id), segment_id="無い")
        assert media.name not in str(caught.value)
