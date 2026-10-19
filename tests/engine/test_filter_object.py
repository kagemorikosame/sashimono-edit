"""フィルタのクリップ（AviUtl のフィルタオブジェクト Issue #27）

フィルタは、それより下のトラックを重ね終えた絵にエフェクトを掛け、その絵で置き換える
効くのはクリップがある時間だけで、上のトラックには効かない 壊れると、置いたのに
何も変わらない・フィルタより上のテロップまで色が変わる・フィルタの無い時間まで変わる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest
from OpenGL import GL

from sashimono.compat.aviutl.catalog import ScriptCatalog, set_script_catalog
from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddScene,
    AddTrack,
    Document,
    InScene,
    SetClipProperty,
    insert_filter,
    new_scene,
)
from sashimono.core.io import (
    FORMAT_VERSION,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from sashimono.core.model import (
    FILTER_KIND,
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import FILTER, source_registry
from sashimono.engine.encode import ExportSettings, available_video_codecs, export_project
from sashimono.engine.gpu import (
    BlendMode,
    Compositor,
    GLContextError,
    OffscreenGLContext,
    Placement,
)
from sashimono.engine.render import FrameRenderer, changed_spans

SETTINGS = ProjectSettings(width=64, height=36, frame_rate=FrameRate(30))
RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)

#: 見る所 左の四角・右の四角・何も無い所（画素は左上が原点）
LEFT = (18, 16)
RIGHT = (18, 48)
EMPTY = (2, 32)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _box(color: tuple[float, float, float, float], x: float) -> GeneratedSource:
    """左右に置く 20px の四角 上下の端と真ん中は空けて、何も無い所を残す"""
    return GeneratedSource(
        kind="shape",
        params={
            "shape": "rect",
            "color": color,
            "width": AnimatedValue(20.0),
            "height": AnimatedValue(20.0),
            "pos_x": AnimatedValue(x),
        },
    )


def _invert() -> Effect:
    definition = registry.get("invert")
    assert definition is not None
    return definition.create()


def _filter(
    start: int = 10,
    duration: int = 10,
    *,
    opacity: float = 1.0,
    effects: tuple[Effect, ...] | None = None,
) -> Clip:
    return Clip(
        timeline_start=start,
        duration=duration,
        source=FILTER.create(),
        effects=(_invert(),) if effects is None else effects,
        opacity=AnimatedValue(opacity),
    )


def _project(*above: Clip, filter_clip: Clip | None = None) -> Project:
    """赤（左）と緑（右）を別のトラックに 30 フレーム 3 本目にフィルタ、4 本目に ``above``"""
    project = Project.create(SETTINGS)
    tracks = (
        Track(TrackKind.VIDEO, "V1", (Clip(timeline_start=0, duration=30, source=_box(RED, -16)),)),
        Track(
            TrackKind.VIDEO, "V2", (Clip(timeline_start=0, duration=30, source=_box(GREEN, 16)),)
        ),
        Track(TrackKind.VIDEO, "V3", () if filter_clip is None else (filter_clip,)),
        Track(TrackKind.VIDEO, "V4", above),
    )
    return replace(project, timeline=replace(project.timeline, tracks=tracks))


def _render(project: Project, gl_context: OffscreenGLContext, *frames: int) -> list[np.ndarray]:
    renderer = FrameRenderer(project, context=gl_context)
    try:
        return [renderer.render(frame) for frame in frames]
    finally:
        renderer.close()


def _rgb(image: np.ndarray, at: tuple[int, int]) -> tuple[int, int, int]:
    row, column = at
    red, green, blue = (int(value) for value in image[row, column, :3])
    return red, green, blue


def _near(actual: tuple[int, int, int], expected: tuple[int, int, int], tolerance: int = 3) -> bool:
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected, strict=True))


class TestRendering:
    def test_the_filter_changes_everything_below_only_while_it_is_there(
        self, gl_context: OffscreenGLContext
    ) -> None:
        # 下の 2 本（別のトラック）の両方が反転する 片方だけなら、フィルタが下の合成結果では
        # なくすぐ下のクリップ 1 本にしか掛かっていない フィルタの無い 5 と 25 は元のまま
        before, during, after = _render(_project(filter_clip=_filter()), gl_context, 5, 15, 25)
        for image in (before, after):
            assert _near(_rgb(image, LEFT), (255, 0, 0))
            assert _near(_rgb(image, RIGHT), (0, 255, 0))
        assert _near(_rgb(during, LEFT), (0, 255, 255))
        assert _near(_rgb(during, RIGHT), (255, 0, 255))

    def test_nothing_is_drawn_where_nothing_was_below(self, gl_context: OffscreenGLContext) -> None:
        # 黒を敷いてから掛けると、何も無い所が反転で白くなり、入れ子のシーンでは外の絵を隠す
        (image,) = _render(_project(filter_clip=_filter()), gl_context, 15)
        assert _near(_rgb(image, EMPTY), (0, 0, 0))

    def test_tracks_above_the_filter_are_left_alone(self, gl_context: OffscreenGLContext) -> None:
        # フィルタより上のトラックは、フィルタの後から重なる 反転すると上のテロップまで
        # 色が変わる
        above = Clip(timeline_start=0, duration=30, source=_box(BLUE, -16))
        (image,) = _render(_project(above, filter_clip=_filter()), gl_context, 15)
        assert _near(_rgb(image, LEFT), (0, 0, 255))
        assert _near(_rgb(image, RIGHT), (255, 0, 255))

    def test_opacity_mixes_before_and_after(self, gl_context: OffscreenGLContext) -> None:
        # 不透明度は掛ける前と後の混ぜ具合 0.5 なら赤と水色の真ん中 上に重ねる作りだと、
        # 下の赤が透けて残る分だけ赤に寄る
        half = _filter(opacity=0.5)
        (image,) = _render(_project(filter_clip=half), gl_context, 15)
        red, green, blue = _rgb(image, LEFT)
        assert abs(red - green) <= 4
        assert abs(green - blue) <= 4
        assert 100 <= red <= 200

    def test_a_filter_without_effects_changes_nothing(self, gl_context: OffscreenGLContext) -> None:
        # 置いたばかり（エフェクトを積む前）のフィルタで絵が消えたり黒くなったりしない
        empty = _filter(effects=())
        (image,) = _render(_project(filter_clip=empty), gl_context, 15)
        assert _near(_rgb(image, LEFT), (255, 0, 0))
        assert _near(_rgb(image, RIGHT), (0, 255, 0))

    def test_a_disabled_filter_changes_nothing(self, gl_context: OffscreenGLContext) -> None:
        # 切ったフィルタは描かない 壊れると、無効にしたのに下の絵が反転し続ける
        off = replace(_filter(), enabled=False)
        (image,) = _render(_project(filter_clip=off), gl_context, 15)
        assert _near(_rgb(image, LEFT), (255, 0, 0))

    def test_inside_a_scene_only_the_scene_is_filtered(
        self, gl_context: OffscreenGLContext
    ) -> None:
        # シーンの中のフィルタは、シーンの中の下の段にだけ効く シーンを置いた外の
        # 下のトラック（赤）まで反転すると、入れ子にした意味が無くなる
        project = Project.create(SETTINGS)
        outer = Track(TrackKind.VIDEO, "V1")
        project = AddTrack(outer).apply(project)
        project = AddClip(
            outer.id, Clip(timeline_start=0, duration=30, source=_box(RED, -16))
        ).apply(project)
        scene = new_scene(project, "中")
        project = AddScene(scene).apply(project)
        inner_below, inner_filter = Track(TrackKind.VIDEO, "S1"), Track(TrackKind.VIDEO, "S2")
        for command in (
            AddTrack(inner_below),
            AddClip(inner_below.id, Clip(timeline_start=0, duration=30, source=_box(GREEN, 16))),
            AddTrack(inner_filter),
            AddClip(inner_filter.id, _filter(start=0, duration=30)),
        ):
            project = InScene(scene.id, command).apply(project)
        placed = Track(TrackKind.VIDEO, "V2")
        project = AddTrack(placed).apply(project)
        project = AddClip(placed.id, Clip(timeline_start=0, duration=30, scene_id=scene.id)).apply(
            project
        )

        (image,) = _render(project, gl_context, 15)
        assert _near(_rgb(image, LEFT), (255, 0, 0))
        assert _near(_rgb(image, RIGHT), (255, 0, 255))

    def test_a_script_filter_keeps_half_transparent_pictures_as_bright(
        self, gl_context: OffscreenGLContext
    ) -> None:
        # スクリプトへは下の絵をストレートアルファで渡す 合成先の事前乗算のまま渡すと、
        # 何もしないスクリプトでも半透明の所で不透明度が 2 回掛かり、赤が暗く沈む
        catalog = ScriptCatalog(roots=())
        catalog.add_text("aviutl:試験.anm:そのまま", "obj.ox = 0")
        set_script_catalog(catalog)
        definition = registry.get("aviutl:試験.anm:そのまま")
        assert definition is not None
        faint = Clip(
            timeline_start=0, duration=30, source=_box(RED, -16), opacity=AnimatedValue(0.5)
        )
        plain = _project()
        plain = plain.with_timeline(
            plain.timeline.replace_track(replace(plain.timeline.tracks[0], clips=(faint,)))
        )
        filtered = plain.with_timeline(
            plain.timeline.replace_track(
                replace(
                    plain.timeline.tracks[2],
                    clips=(_filter(effects=(definition.create(),)),),
                )
            )
        )
        (expected,) = _render(plain, gl_context, 15)
        (actual,) = _render(filtered, gl_context, 15)
        assert _near(_rgb(actual, LEFT), _rgb(expected, LEFT), 4)
        assert _near(_rgb(actual, RIGHT), _rgb(expected, RIGHT), 4)

    def test_a_script_filter_reads_the_screen_below(self, gl_context: OffscreenGLContext) -> None:
        """フィルタに積んだスクリプトの ``obj.copybuffer("obj", "frm")`` は下の絵を写す（#170）

        フィルタは掛けた結果を空の合成先へ描く その空の方を画面として写すと、黒一色の絵に
        置き換わって下の赤と緑が消える
        """
        catalog = ScriptCatalog(roots=())
        catalog.add_text("aviutl:試験.anm:画面を写す", 'obj.copybuffer("obj", "frm")')
        set_script_catalog(catalog)
        definition = registry.get("aviutl:試験.anm:画面を写す")
        assert definition is not None
        project = _project(filter_clip=_filter(effects=(definition.create(),)))
        (image,) = _render(project, gl_context, 15)
        assert _near(_rgb(image, LEFT), (255, 0, 0))
        assert _near(_rgb(image, RIGHT), (0, 255, 0))

    def test_replacing_keeps_a_very_faint_picture(self, gl_context: OffscreenGLContext) -> None:
        # 置き換えは事前乗算の値をそのまま写す 一度ストレートへ戻そうとすると、戻すのを
        # 飛ばすごく薄い所（不透明度 0.0001 以下）で不透明度が 2 回掛かり、色が消える
        with gl_context:
            source = Compositor(4, 4)
            target = Compositor(4, 4)
            try:
                # 事前乗算で (0.00005, 0, 0, 0.00005) ストレートなら不透明度 0.00005 の赤
                source.begin((0.00005, 0.0, 0.0, 0.00005))
                target.begin((0.0, 0.0, 0.0, 0.0))
                target.draw_handle(
                    source.canvas.color,
                    Placement(0.0, 0.0, 4.0, 4.0),
                    flip=False,
                    blend=BlendMode.REPLACE,
                    premultiplied=True,
                )
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.canvas.handle)
                raw = GL.glReadPixels(0, 0, 1, 1, GL.GL_RGBA, GL.GL_FLOAT)
            finally:
                source.release()
                target.release()
        red, _, _, alpha = np.frombuffer(raw, dtype=np.float32)[:4]
        assert alpha == pytest.approx(0.00005, rel=0.05)
        assert red == pytest.approx(0.00005, rel=0.05)

    def test_export_writes_the_same_picture(
        self, gl_context: OffscreenGLContext, tmp_path: Path
    ) -> None:
        # 書き出しもプレビューと同じ道で描く 書き出しだけフィルタを知らないと、
        # 見て確かめた色と違う動画ができる
        if "libx264" not in available_video_codecs():
            pytest.skip("libx264 が使えない")
        project = _project(filter_clip=_filter())
        output = tmp_path / "filter.mp4"
        try:
            export_project(project, ExportSettings(path=output, video_codec="libx264"))
        except GLContextError as exc:
            pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
        with av.open(str(output)) as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        # 圧縮で色がにじむので、四角の真ん中を広めの幅で見る
        assert _near(_rgb(frames[5], LEFT), (255, 0, 0), 40)
        assert _near(_rgb(frames[15], LEFT), (0, 255, 255), 40)
        assert _near(_rgb(frames[15], RIGHT), (255, 0, 255), 40)
        assert _near(_rgb(frames[25], RIGHT), (0, 255, 0), 40)


class TestModel:
    def test_the_kind_is_registered(self) -> None:
        # 設定パネルとタイムラインの名前は登録から引く 無いと「filter」と生の種類名が出る
        definition = source_registry.get(FILTER_KIND)
        assert definition is not None
        assert definition.label == "フィルタ"
        assert _filter().is_filter
        assert not Clip(timeline_start=0, duration=1, source=_box(RED, 0)).is_filter

    def test_saving_and_loading_keeps_the_filter(self, tmp_path: Path) -> None:
        # 壊れると、開き直したときにフィルタが普通の生成オブジェクトになったりエフェクトや
        # 不透明度が落ちたりして、保存前と違う絵になる
        project = _project(filter_clip=_filter(opacity=0.25))
        path = tmp_path / "filter.sme"
        save_project(project, path)
        loaded = load_project(path)
        located = loaded.timeline.locate_clip(_filter_id(project))
        assert located is not None
        clip = located[1]
        assert clip.is_filter
        assert [effect.kind for effect in clip.effects] == ["invert"]
        assert clip.opacity.static == pytest.approx(0.25)
        # 名前はファイル名から付くので、タイムラインだけを比べる
        assert project_to_dict(loaded)["timeline"] == project_to_dict(project)["timeline"]

    def test_the_format_version_went_up(self) -> None:
        # 5 までの本体はフィルタを知らず、下の絵に掛かるはずのエフェクトを黙って落とす
        # 版を上げておけば「新しい形式」で止まる
        assert FORMAT_VERSION >= 6
        assert project_to_dict(_project(filter_clip=_filter()))["version"] == FORMAT_VERSION

    def test_a_version_5_file_still_opens(self) -> None:
        # 版を上げたせいで、フィルタを持たない前の版のファイルが「新しい形式」や読み違いで
        # 開けなくならないこと
        data = project_to_dict(_project())
        data["version"] = 5
        loaded = project_from_dict(data)
        assert [track.name for track in loaded.timeline.tracks] == ["V1", "V2", "V3", "V4"]
        assert not any(clip.is_filter for track in loaded.timeline.tracks for clip in track.clips)


def _filter_id(project: Project) -> ClipId:
    return next(
        clip.id for track in project.timeline.tracks for clip in track.clips if clip.is_filter
    )


class TestInsert:
    def test_the_filter_goes_above_everything_in_its_range(self) -> None:
        # 下から空きを探すと、範囲の絵より下に入り、置いても何も変わらない
        project = _project()
        commands = insert_filter(project, at_frame=10, duration=10)
        for command in commands:
            project = command.apply(project)
        names = [track.name for track in project.timeline.tracks if track.clips]
        filtered = next(t for t in project.timeline.tracks for c in t.clips if c.is_filter)
        # V3 は空いていて、V1 と V2 より上 そこへ入る
        assert filtered.name == "V3"
        assert names == ["V1", "V2", "V3"]

    def test_a_new_top_track_is_made_when_there_is_no_room_above(self) -> None:
        # 上に空きが無いのに下の空きへ入れると、フィルタより上の絵に掛からないどころか
        # 範囲の絵の一部にしか効かない 一番上に新しいトラックを作って置く
        project = _project(Clip(timeline_start=0, duration=30, source=_box(BLUE, 0)))
        commands = insert_filter(project, at_frame=10, duration=10)
        assert isinstance(commands[0], AddTrack)
        for command in commands:
            project = command.apply(project)
        video = list(project.timeline.video_tracks())
        assert video[-1].clips[0].is_filter

    def test_undo_takes_the_filter_away(self) -> None:
        # 置く操作（トラックの用意とクリップ）が 1 回の取り消しで戻らないと、取り消しても
        # 空のフィルタやトラックが残り、下の絵が変わったままに見える
        document = Document(_project())
        with document.checkpoint("フィルタを追加"):
            for command in insert_filter(document.project, at_frame=10, duration=10):
                document.execute(command)
        clip_id = _filter_id(document.project)
        document.execute(AddEffect(clip_id, _invert()))
        located = document.project.timeline.locate_clip(clip_id)
        assert located is not None
        assert len(located[1].effects) == 1
        document.undo()
        document.undo()
        assert not any(c.is_filter for t in document.project.timeline.tracks for c in t.clips)

    def test_a_muted_track_is_not_used(self) -> None:
        # ミュートした空きトラックへ置くと、置いたのにプレビューにも書き出しにも効かない
        project = _project()
        muted = replace(project.timeline.tracks[2], muted=True)
        project = project.with_timeline(project.timeline.replace_track(muted))
        for command in insert_filter(project, at_frame=10, duration=10):
            project = command.apply(project)
        placed = next(t for t in project.timeline.tracks for c in t.clips if c.is_filter)
        assert placed.name == "V4"
        assert placed.id in {t.id for t in project.timeline.active_tracks(TrackKind.VIDEO)}

    def test_a_new_track_joins_the_solo(self) -> None:
        # ソロで絞っている間に作ったトラックがソロの外になると、置いたフィルタが描かれない
        project = _project()
        soloed = replace(project.timeline.tracks[0], solo=True)
        project = project.with_timeline(project.timeline.replace_track(soloed))
        for command in insert_filter(project, at_frame=10, duration=10):
            project = command.apply(project)
        placed = next(t for t in project.timeline.tracks for c in t.clips if c.is_filter)
        assert placed.solo
        assert placed.id in {t.id for t in project.timeline.active_tracks(TrackKind.VIDEO)}


class TestInvalidation:
    """先読みした絵を捨てる範囲 フィルタは同じ時刻の下の絵だけを読むので、変えた所だけ捨てる"""

    def test_placing_a_filter_drops_its_span(self) -> None:
        # 置いた所を捨てないと、先読みした所だけフィルタの掛からない絵が出る 範囲の外まで
        # 捨てると、置くたびに全体を作り直して先読みが効かなくなる
        before = _project()
        after = before
        for command in insert_filter(before, at_frame=10, duration=10):
            after = command.apply(after)
        found = changed_spans(before, after)
        assert found.contains(10) and found.contains(19)
        assert not found.contains(5) and not found.contains(25)

    def test_changing_the_filter_drops_its_span(self) -> None:
        # フィルタの外まで捨てると、入り切りを試すたびに作品全体を描き直して先読みが追い付かない
        before = _project(filter_clip=_filter())
        clip_id = _filter_id(before)
        # フィルタ自身を切り替えたのに捨て損ねると、先読みした所だけ反転したままの絵が出る
        after = SetClipProperty(clip_id, "enabled", False).apply(before)
        found = changed_spans(before, after)
        assert found.contains(15)
        assert not found.contains(5)

    def test_changing_a_clip_below_drops_the_filtered_frames_too(self) -> None:
        # 下の絵が変われば、その上でフィルタを掛けた絵も変わる 捨て損ねると、
        # 下を直したのにフィルタの掛かった所だけ古い色が残る
        before = _project(filter_clip=_filter())
        below = before.timeline.tracks[0].clips[0]
        after = SetClipProperty(below.id, "blend_mode", "add").apply(before)
        found = changed_spans(before, after)
        assert found.contains(15)
