"""エフェクトとパラメータの操作、およびプリセット

キーフレームの編集はここが唯一の入口 UI もグラフエディタも AI もこれを通るので、
ここが正しければ 3 つとも正しい
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sashimono.core.commands import (
    AddEffect,
    ClearKeyframes,
    Document,
    MoveEffect,
    MoveKeyframe,
    ParamPath,
    ParamTarget,
    RemoveEffect,
    RemoveKeyframe,
    SetClipProperty,
    SetEffectEnabled,
    SetKeyframe,
    SetParam,
    SetSource,
    resolve_param,
)
from sashimono.core.io import Preset, PresetStore, ProjectFileError
from sashimono.core.io.presets import SUFFIX
from sashimono.core.model import (
    AnimatedValue,
    ClipId,
    Effect,
    EffectId,
    GeneratedSource,
    Interpolation,
    Keyframe,
    MediaItem,
    Project,
    Track,
    TrackKind,
)
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE, TEXT
from tests.conftest import make_clip


@pytest.fixture
def placed(project: Project, video_media: MediaItem) -> Project:
    """クリップを 1 本置いたプロジェクト"""
    track = project.timeline.tracks[0]
    clip = make_clip(0, 120, video_media)
    return project.with_timeline(project.timeline.replace_track(track.with_clips((clip,))))


def only_clip(project: Project) -> ClipId:
    return project.timeline.tracks[0].clips[0].id


def blur(radius: float = 8.0) -> Effect:
    return registry.require("blur").create(radius=radius)


class TestParamPath:
    def test_effect_path_needs_an_effect_id(self) -> None:
        # エフェクトの ID の無い道を作った所で断らないと、値を書く所まで進んでから、
        # どのエフェクトの値か分からない失敗になる
        with pytest.raises(ValueError, match="effect_id"):
            ParamPath(ClipId("c"), ParamTarget.EFFECT, "radius")

    def test_constructors(self) -> None:
        # 作り口が別の種類の道を返すと、不透明度を動かしたつもりがエフェクトの値を探しに行き失敗する
        clip_id, effect_id = ClipId("c"), EffectId("e")
        assert ParamPath.of_effect(clip_id, effect_id, "x").target is ParamTarget.EFFECT
        assert ParamPath.of_source(clip_id, "x").target is ParamTarget.SOURCE
        assert ParamPath.of_clip(clip_id, "x").target is ParamTarget.CLIP


class TestEffectStack:
    def test_add_and_remove(self, placed: Project) -> None:
        # 足しても積まれない・外しても残ると、〔＋ エフェクト〕や削除を押しても絵が変わらない
        clip_id = only_clip(placed)
        effect = blur()
        added = AddEffect(clip_id, effect).apply(placed)
        assert len(added.timeline.tracks[0].clips[0].effects) == 1

        removed = RemoveEffect(clip_id, effect.id).apply(added)
        assert removed.timeline.tracks[0].clips[0].effects == ()

    def test_add_at_index(self, placed: Project) -> None:
        # 指した位置でなく末尾に入ると、掛ける順が変わり、色を変えてからぼかした絵が逆の順の絵になる
        clip_id = only_clip(placed)
        first, second = blur(), registry.require("color").create()
        project = AddEffect(clip_id, first).apply(placed)
        project = AddEffect(clip_id, second, index=0).apply(project)
        assert [e.kind for e in project.timeline.tracks[0].clips[0].effects] == ["color", "blur"]

    def test_move_changes_the_order(self, placed: Project) -> None:
        # 掛ける順で結果が変わる ぼかしてから色を変えるのと逆は別の絵になる
        clip_id = only_clip(placed)
        first, second = blur(), registry.require("color").create()
        project = AddEffect(clip_id, first).apply(placed)
        project = AddEffect(clip_id, second).apply(project)

        moved = MoveEffect(clip_id, second.id, 0).apply(project)
        assert [e.kind for e in moved.timeline.tracks[0].clips[0].effects] == ["color", "blur"]

    def test_move_clamps_the_index(self, placed: Project) -> None:
        # 列より大きい行き先を断ると、一覧のいちばん下より先へドラッグしただけで並べ替えが失敗する
        clip_id = only_clip(placed)
        effect = blur()
        project = AddEffect(clip_id, effect).apply(placed)
        assert MoveEffect(clip_id, effect.id, 99).apply(project) is not None

    def test_enable_toggle(self, placed: Project) -> None:
        # 無効にしても効いたままだと、消さずに切って掛ける前と後を見比べることができない
        clip_id = only_clip(placed)
        effect = blur()
        project = AddEffect(clip_id, effect).apply(placed)
        disabled = SetEffectEnabled(clip_id, effect.id, False).apply(project)
        assert not disabled.timeline.tracks[0].clips[0].effects[0].enabled

    def test_unknown_effect_is_refused(self, placed: Project) -> None:
        # 無いエフェクトを外す指示が通ると、何も変わらないのに「エフェクトを削除」の段が積まれ、
        # 取り消しても何も起きない
        with pytest.raises(KeyError):
            RemoveEffect(only_clip(placed), EffectId("無い")).apply(placed)


class TestSetParam:
    def test_effect_parameter(self, placed: Project) -> None:
        # 書いた値を読み戻せないと、インスペクターでぼかしの強さを変えても絵も数も変わらない
        clip_id = only_clip(placed)
        effect = blur()
        project = AddEffect(clip_id, effect).apply(placed)
        path = ParamPath.of_effect(clip_id, effect.id, "radius")

        updated = SetParam(path, AnimatedValue(static=30.0)).apply(project)
        value = resolve_param(updated, path)
        assert isinstance(value, AnimatedValue)
        assert value.static == 30.0

    def test_clip_parameter(self, placed: Project) -> None:
        # 元の作品まで書き換わると、取り消しても不透明度が戻らない
        clip_id = only_clip(placed)
        path = ParamPath.of_clip(clip_id, "opacity")
        updated = SetParam(path, AnimatedValue(static=0.5)).apply(placed)
        assert placed.timeline.tracks[0].clips[0].opacity.static == 1.0
        assert updated.timeline.tracks[0].clips[0].opacity.static == 0.5

    def test_source_parameter(self, project: Project) -> None:
        # 文字が中身に入らないと、インスペクターで打った文字が画面のテロップに出ない
        track = project.timeline.tracks[0]
        clip = replace(make_clip(0, 60, _dummy_media()), media_id=None, source=TEXT.create())
        staged = project.with_timeline(project.timeline.replace_track(track.with_clips((clip,))))

        path = ParamPath.of_source(clip.id, "text")
        updated = SetParam(path, "こんにちは").apply(staged)
        assert resolve_param(updated, path) == "こんにちは"

    def test_source_parameter_without_a_source(self, placed: Project) -> None:
        # テキストを持たない動画のクリップへ文字を書く指示をここで断らないと、
        # 無い中身を触って落ち、何が悪いのか知らせに出ない
        path = ParamPath.of_source(only_clip(placed), "text")
        with pytest.raises(KeyError, match="生成オブジェクト"):
            SetParam(path, "x").apply(placed)

    def test_unknown_clip_field(self, placed: Project) -> None:
        # 無い項目名をここで断らないと、差し替えの型の誤りで落ち、どの項目名が悪いのか出ない
        path = ParamPath.of_clip(only_clip(placed), "そんな項目は無い")
        with pytest.raises(KeyError):
            SetParam(path, AnimatedValue()).apply(placed)

    def test_resolve_returns_none_for_missing_clip(self, placed: Project) -> None:
        # グラフエディタは開いた時の道を持ち続ける 指すクリップが消えた後に読んで例外になると、
        # クリップを消しただけでグラフエディタが落ちる
        assert resolve_param(placed, ParamPath.of_clip(ClipId("無い"), "opacity")) is None


class TestKeyframes:
    def _path(self, project: Project) -> tuple[Project, ParamPath]:
        clip_id = only_clip(project)
        effect = blur()
        staged = AddEffect(clip_id, effect).apply(project)
        return staged, ParamPath.of_effect(clip_id, effect.id, "radius")

    def test_set_creates_animation(self, placed: Project) -> None:
        # 2 つ目のキーを打っても値が動かないと、0 から 60 へ強まるはずのぼかしがずっと同じ強さで出る
        project, path = self._path(placed)
        project = SetKeyframe(path, 0, 0.0).apply(project)
        project = SetKeyframe(path, 30, 60.0).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert value.is_animated
        assert value.at(0) == 0.0
        assert value.at(15) == pytest.approx(30.0)
        assert value.at(30) == 60.0

    def test_set_replaces_at_the_same_frame(self, placed: Project) -> None:
        # 同じ位置へ打ち直した値を新しい点として足すと、同じフレームの 2 点がモデルに断られ、
        # 値を打ち直せない
        project, path = self._path(placed)
        project = SetKeyframe(path, 10, 1.0).apply(project)
        project = SetKeyframe(path, 10, 5.0).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert len(value.keyframes) == 1
        assert value.keyframes[0].value == 5.0

    def test_keyframes_stay_sorted(self, placed: Project) -> None:
        # 並べ直さずに足すと、前の時刻へ後から打った点がモデルの昇順の検査で断られ、
        # 戻った位置にキーを打てない
        project, path = self._path(placed)
        for frame in (30, 0, 15):
            project = SetKeyframe(path, frame, float(frame)).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert [k.frame for k in value.keyframes] == [0, 15, 30]

    def test_interpolation_is_kept(self, placed: Project) -> None:
        # 止める出方（HOLD）が落ちると、パッと切り替わるはずの値が次の点へ向けてじわじわ変わる
        project, path = self._path(placed)
        project = SetKeyframe(path, 0, 0.0, interpolation=Interpolation.HOLD).apply(project)
        project = SetKeyframe(path, 20, 10.0).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert value.keyframes[0].interpolation is Interpolation.HOLD
        assert value.at(10) == 0.0

    def test_changing_only_the_value_keeps_the_shape_of_the_point(self, placed: Project) -> None:
        """値だけを打ち直しても、その点の出方（補間方法と曲線の名前）は残る

        インスペクターで数を直すたびに直線へ戻ると、YMM4 から読んだ Back_InOut の点が
        値を触っただけで行き過ぎない動きに変わる
        """
        project, path = self._path(placed)
        project = SetKeyframe(
            path, 0, 0.0, interpolation=Interpolation.EASE_IN_OUT, curve="back"
        ).apply(project)
        project = SetKeyframe(path, 30, 60.0).apply(project)
        project = SetKeyframe(path, 0, 10.0).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        first = value.keyframes[0]
        assert first.value == 10.0
        assert first.interpolation is Interpolation.EASE_IN_OUT
        assert first.curve == "back"

    def test_a_new_point_without_an_interpolation_is_straight(self, placed: Project) -> None:
        # 出方を渡さずに新しい点を置いたら直線 渡された曲線の名前を持たせると、
        # 直線の点に効かない名前が残る
        project, path = self._path(placed)
        project = SetKeyframe(path, 0, 0.0, curve="back").apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert value.keyframes[0].interpolation is Interpolation.LINEAR
        assert value.keyframes[0].curve == ""

    def test_choosing_an_interpolation_replaces_the_curve(self, placed: Project) -> None:
        # 出方を選び直したら、前の曲線の名前は持ち越さない（グラフエディタの切り替え）
        project, path = self._path(placed)
        project = SetKeyframe(
            path, 0, 0.0, interpolation=Interpolation.EASE_IN_OUT, curve="back"
        ).apply(project)
        project = SetKeyframe(path, 0, 0.0, interpolation=Interpolation.EASE_OUT).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert value.keyframes[0].interpolation is Interpolation.EASE_OUT
        assert value.keyframes[0].curve == ""

    def test_remove_one(self, placed: Project) -> None:
        # 消した点が残ると、グラフエディタで消したはずのキーがまだ動きに効き続ける
        project, path = self._path(placed)
        project = SetKeyframe(path, 0, 0.0).apply(project)
        project = SetKeyframe(path, 30, 60.0).apply(project)
        project = RemoveKeyframe(path, 30).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert len(value.keyframes) == 1

    def test_removing_the_last_keeps_the_value(self, placed: Project) -> None:
        # 0 に戻ると、キーフレームを消した瞬間に絵が飛ぶ
        project, path = self._path(placed)
        project = SetKeyframe(path, 12, 42.0).apply(project)
        project = RemoveKeyframe(path, 12).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert not value.is_animated
        assert value.static == 42.0

    def test_move(self, placed: Project) -> None:
        # 動かした点が元の位置に残るか値が変わらないと、点を引きずっても動きの時刻と強さが変わらない
        project, path = self._path(placed)
        project = SetKeyframe(path, 0, 0.0).apply(project)
        project = SetKeyframe(path, 30, 60.0).apply(project)
        project = MoveKeyframe(path, 30, 45, 80.0).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert [(k.frame, k.value) for k in value.keyframes] == [(0, 0.0), (45, 80.0)]

    def test_move_onto_another_replaces_it(self, placed: Project) -> None:
        # 重なった 2 点はモデル側の検査で弾かれる 置き換えとして扱う
        project, path = self._path(placed)
        project = SetKeyframe(path, 0, 0.0).apply(project)
        project = SetKeyframe(path, 30, 60.0).apply(project)
        project = MoveKeyframe(path, 30, 0, 90.0).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert [(k.frame, k.value) for k in value.keyframes] == [(0, 90.0)]

    def test_move_unknown_frame(self, placed: Project) -> None:
        # 動かす点が無いと断らないと、別の操作で消えた点をグラフエディタでつまんで動かしたとき、
        # 何を動かしたのか分からない失敗になる
        project, path = self._path(placed)
        project = SetKeyframe(path, 0, 0.0).apply(project)
        with pytest.raises(KeyError):
            MoveKeyframe(path, 99, 10, 1.0).apply(project)

    def test_clear_freezes_the_current_value(self, placed: Project) -> None:
        # 解除した値が見ていたフレームの値にならないと、アニメーションを解除した瞬間に絵が跳ぶ
        project, path = self._path(placed)
        project = SetKeyframe(path, 0, 0.0).apply(project)
        project = SetKeyframe(path, 30, 60.0).apply(project)
        project = ClearKeyframes(path, frame=15).apply(project)

        value = resolve_param(project, path)
        assert isinstance(value, AnimatedValue)
        assert not value.is_animated
        assert value.static == pytest.approx(30.0)

    def test_works_on_clip_and_source_too(self, project: Project) -> None:
        # 指す先ごとにコマンドが分かれていると、どれかの実装が遅れて
        # 「エフェクトは動くがテキストは動かない」というちぐはぐが生まれる
        track = project.timeline.tracks[0]
        clip = replace(make_clip(0, 60, _dummy_media()), media_id=None, source=SHAPE.create())
        staged = project.with_timeline(project.timeline.replace_track(track.with_clips((clip,))))

        for path in (
            ParamPath.of_clip(clip.id, "opacity"),
            ParamPath.of_source(clip.id, "width"),
        ):
            updated = SetKeyframe(path, 10, 3.0).apply(staged)
            value = resolve_param(updated, path)
            assert isinstance(value, AnimatedValue)
            assert value.is_animated


class TestClipProperties:
    def test_blend_mode(self, placed: Project) -> None:
        # 選んだ合成モードがクリップに入らないと、加算を選んでもふつうに重なった絵のまま書き出される
        updated = SetClipProperty(only_clip(placed), "blend_mode", "add").apply(placed)
        assert updated.timeline.tracks[0].clips[0].blend_mode == "add"

    def test_refuses_unknown_fields(self, placed: Project) -> None:
        # 位置や長さを検査なしで書き換えられると、重なりの不変条件を壊せる
        with pytest.raises(ValueError, match="変更できない"):
            SetClipProperty(only_clip(placed), "timeline_start", 999).apply(placed)

    def test_set_source(self, placed: Project) -> None:
        # 差し替えた中身が入らないと、テロップを置いても画面に文字が出ない
        source = TEXT.create(text="テロップ")
        updated = SetSource(only_clip(placed), source).apply(placed)
        assert updated.timeline.tracks[0].clips[0].source == source


class TestUndo:
    def test_effect_edits_are_undoable(self, placed: Project) -> None:
        # 追加とキーを 2 回の Ctrl+Z で戻しきれないと、試しに掛けたぼかしを元の絵へ戻せない
        document = Document(placed)
        effect = blur()
        document.execute(AddEffect(only_clip(placed), effect))
        path = ParamPath.of_effect(only_clip(placed), effect.id, "radius")
        document.execute(SetKeyframe(path, 0, 0.0))

        document.undo()
        document.undo()
        assert document.project == placed


class TestPresets:
    @pytest.fixture
    def store(self, tmp_path: Path) -> PresetStore:
        return PresetStore(tmp_path / "presets")

    def test_round_trip(self, store: PresetStore) -> None:
        # 読み戻せないか並びが変わると、次に開いたとき縁取りと影の組が消えるか、掛ける順が変わる
        preset = Preset(
            name="縁取り＋影",
            effects=(
                registry.require("border").create(width=6),
                registry.require("shadow").create(blur=10),
            ),
        )
        store.save(preset)

        loaded = store.all()
        assert len(loaded) == 1
        assert loaded[0].name == "縁取り＋影"
        assert [e.kind for e in loaded[0].effects] == ["border", "shadow"]

    def test_keyframes_survive(self, store: PresetStore) -> None:
        # プリセットに入れた動きが保存で消えると、ぼかしが抜ける見た目を当てても止まったぼかしになる
        animated = AnimatedValue(
            keyframes=(Keyframe(frame=0, value=0.0), Keyframe(frame=30, value=40.0))
        )
        preset = Preset(
            name="ぼかし抜け",
            effects=(Effect(kind="blur", params={"radius": animated}),),
        )
        store.save(preset)

        restored = store.all()[0].effects[0].params["radius"]
        assert isinstance(restored, AnimatedValue)
        assert restored.at(15) == pytest.approx(20.0)

    def test_instantiate_gives_fresh_ids(self, store: PresetStore) -> None:
        # 同じプリセットを 2 回適用したとき ID が衝突すると、片方を消したつもりで
        # 両方消える
        preset = Preset(name="ぼかし", effects=(blur(),))
        first, second = preset.instantiate(), preset.instantiate()
        assert first[0].id != second[0].id
        assert first[0].kind == second[0].kind

    def test_source_presets(self, store: PresetStore) -> None:
        # テキストのプリセットに中身が残らないと、見出しのプリセットを当てても文字や大きさが入らない
        preset = Preset(name="見出し", source=TEXT.create(text="見出し", size=96))
        store.save(preset)
        loaded = store.all()[0]
        assert loaded.source is not None
        assert loaded.source.params["text"] == "見出し"

    def test_unsafe_names_are_sanitised(self, store: PresetStore) -> None:
        # 名前の「/」や「:」をそのままファイル名に使うと、Windows で保存できないか、
        # 置き場の中の別のフォルダへ書いてしまう
        path = store.save(Preset(name="a/b:c*d?", effects=(blur(),)))
        assert path.exists()
        assert "/" not in path.name

    def test_empty_name(self, store: PresetStore) -> None:
        # 空白だけの名前をそのまま使うと、名前の無い「.smep」のファイルになり、一覧で見分けられない
        path = store.save(Preset(name="   ", effects=(blur(),)))
        assert path.name == f"無題{SUFFIX}"

    def test_broken_preset_does_not_hide_the_others(self, store: PresetStore) -> None:
        # 壊れた 1 つで一覧全体が出なくなると、他のプリセットまで使えなくなる
        store.save(Preset(name="よい", effects=(blur(),)))
        broken = store.root / "ユーザー" / f"こわれ{SUFFIX}"
        broken.write_text("これは JSON ではない", encoding="utf-8")

        assert [p.name for p in store.all()] == ["よい"]

    def test_missing_directory(self, tmp_path: Path) -> None:
        # 置き場のフォルダがまだ無いときに落ちると、プリセットを作る前の人が一覧を開いただけで止まる
        assert PresetStore(tmp_path / "まだ無い").all() == ()

    def test_foreign_format(self, store: PresetStore) -> None:
        # 形式の名前を見ずに読むと、拡張子だけ同じ別の物を、見た目の無いプリセットとして読んでしまう
        path = store.root / "ユーザー" / f"別物{SUFFIX}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"format": "aviutl"}', encoding="utf-8")
        with pytest.raises(ProjectFileError, match="プリセットではない"):
            store.load(path)

    def test_delete(self, store: PresetStore) -> None:
        # 消したプリセットのファイルが残ると、削除したのに次に一覧を開くとまた並ぶ
        preset = Preset(name="消す", effects=(blur(),))
        store.save(preset)
        store.delete(preset)
        assert store.all() == ()


def _dummy_media() -> MediaItem:
    """``make_clip`` に渡すだけの素材 生成オブジェクトでは参照されない"""
    return MediaItem(path=Path("dummy.mp4"))


class TestGeneratedClipsInTracks:
    def test_generated_clips_live_on_video_tracks(self, project: Project) -> None:
        # 素材を持たないクリップを映像トラックに持てないと、テキストや図形をタイムラインに置けない
        track = Track(kind=TrackKind.VIDEO, name="V1")
        clip = replace(
            make_clip(0, 60, _dummy_media()),
            media_id=None,
            source=GeneratedSource(kind="text", params={"text": "x"}),
        )
        assert track.with_clips((clip,)).clips[0].source is not None
