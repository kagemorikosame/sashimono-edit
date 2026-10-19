"""置いたシーンが描かれ、鳴ること

シーンは「中身を別の画面に描いて 1 本のクリップとして重ねる」 描けなければ置いた
場所が空白になり、鳴らなければシーンの中の BGM やナレーションが消える
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import pytest

from sashimono.compat.aviutl.catalog import ScriptCatalog, set_script_catalog
from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddMedia,
    AddScene,
    AddTrack,
    Command,
    InScene,
    insert_scene,
    new_scene,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.definition import registry
from sashimono.engine.audio import AudioMixer
from sashimono.engine.decode import probe_media
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from tests.media_fixtures import SampleMedia

SETTINGS = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30), sample_rate=48000)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _shape(color: tuple[float, float, float, float], size: float | None = None) -> GeneratedSource:
    if size is None:
        return GeneratedSource(kind="shape", params={"shape": "background", "color": color})
    return GeneratedSource(
        kind="shape",
        params={
            "shape": "rect",
            "color": color,
            "width": AnimatedValue(size),
            "height": AnimatedValue(size),
        },
    )


def _scene_with(project: Project, clip: Clip) -> tuple[Project, Clip]:
    scene = new_scene(project, "中")
    project = AddScene(scene).apply(project)
    track = Track(TrackKind.VIDEO, "S1")
    project = InScene(scene.id, AddTrack(track)).apply(project)
    project = InScene(scene.id, AddClip(track.id, clip)).apply(project)
    project = _apply(project, insert_scene(project, scene.id, at_frame=0, duration=60))
    placed = next(c for t in project.timeline.tracks for c in t.clips if c.scene_id == scene.id)
    return project, placed


def _render(project: Project, context: OffscreenGLContext, frame: int) -> np.ndarray:
    renderer = FrameRenderer(project, context=context)
    try:
        return renderer.render(frame)
    finally:
        renderer.close()


class TestPicture:
    def test_the_scene_is_drawn_where_it_is_placed(self, gl_context: OffscreenGLContext) -> None:
        # シーンの中身が描かれないと、シーンにまとめた途端にその場所が
        # プレビューでも書き出しでも空白になる
        project, _ = _scene_with(
            Project.create(SETTINGS),
            Clip(timeline_start=0, duration=60, source=_shape((1.0, 1.0, 1.0, 1.0))),
        )
        assert int(_render(project, gl_context, 10)[32, 32, 0]) > 250

    def test_what_is_below_shows_through(self, gl_context: OffscreenGLContext) -> None:
        # シーンのキャンバスが黒で始まると、シーンを重ねた下の絵が全部隠れる
        project = Project.create(SETTINGS)
        below = Track(TrackKind.VIDEO, "V0")
        project = AddTrack(below).apply(project)
        red = Clip(timeline_start=0, duration=60, source=_shape((1.0, 0.0, 0.0, 1.0)))
        project = AddClip(below.id, red).apply(project)
        project, _ = _scene_with(
            project, Clip(timeline_start=0, duration=60, source=_shape((1.0, 1.0, 1.0, 1.0), 8))
        )
        image = _render(project, gl_context, 10)
        assert tuple(int(v) for v in image[2, 2, :3]) == (255, 0, 0)
        assert int(image[32, 32, 1]) > 250

    def test_source_in_shifts_the_scene_time(self, gl_context: OffscreenGLContext) -> None:
        # 分割やトリムで後ろ半分だけ置いたシーンが、頭から再生し直されないこと
        project, placed = _scene_with(
            Project.create(SETTINGS),
            Clip(timeline_start=30, duration=30, source=_shape((1.0, 1.0, 1.0, 1.0))),
        )
        assert int(_render(project, gl_context, 0)[32, 32, 0]) < 8
        track = next(t for t in project.timeline.tracks if placed in t.clips)
        shifted = Clip(
            timeline_start=0, duration=30, scene_id=placed.scene_id, source_in=Fraction(1)
        )
        project = project.with_timeline(
            project.timeline.replace_track(track.with_clips((shifted,)))
        )
        assert int(_render(project, gl_context, 0)[32, 32, 0]) > 250


class TestSound:
    def test_the_scene_audio_is_mixed(self, sample_av: SampleMedia) -> None:
        # シーンの中の音を混ぜないと、BGM やナレーションをシーンにまとめた途端に音が消える
        # 置いた位置より前で鳴ると、シーンの音が映像より早く始まる
        media = probe_media(sample_av.path)
        project = AddMedia(media).apply(Project.create(SETTINGS))
        scene = new_scene(project, "音")
        project = AddScene(scene).apply(project)
        track = Track(TrackKind.AUDIO, "SA")
        project = InScene(scene.id, AddTrack(track)).apply(project)
        clip = Clip(timeline_start=0, duration=60, media_id=media.id)
        project = InScene(scene.id, AddClip(track.id, clip)).apply(project)
        project = _apply(project, insert_scene(project, scene.id, at_frame=30, duration=60))

        mixer = AudioMixer(project)
        try:
            before = mixer.render(0, 24000)
            during = mixer.render(48000 + 12000, 24000)
        finally:
            mixer.close()
        assert np.all(before == 0.0)
        assert float(np.sqrt(np.mean(during**2))) > 0.0


class TestScripts:
    """シーンとフレームバッファのクリップにも AviUtl スクリプトを掛けられること

    どちらも絵を GPU の中で作る スクリプトは CPU の画像を書き換える作りなので、
    掛けるときだけ 1 枚読み戻して渡す
    """

    def _moved(self) -> str:
        catalog = ScriptCatalog(roots=())
        catalog.add_text("aviutl:試験.anm:寄せる", "obj.ox = 16")
        set_script_catalog(catalog)
        return "aviutl:試験.anm:寄せる"

    def test_a_script_moves_the_scene(self, gl_context: OffscreenGLContext) -> None:
        # シーンに掛けた AviUtl のスクリプトが効かないと、配布の動きを当てたシーンが止まったまま出る
        identifier = self._moved()
        project, placed = _scene_with(
            Project.create(SETTINGS),
            Clip(timeline_start=0, duration=60, source=_shape((1.0, 1.0, 1.0, 1.0), 8)),
        )
        definition = registry.get(identifier)
        assert definition is not None
        project = AddEffect(placed.id, definition.create()).apply(project)
        image = _render(project, gl_context, 10)
        # 8px の四角が中央から右へ 16px 動く
        assert int(image[32, 48, 0]) > 250
        assert int(image[32, 32, 0]) < 8

    def test_a_script_moves_the_framebuffer(self, gl_context: OffscreenGLContext) -> None:
        # フレームバッファのクリップにスクリプトが効かないと、
        # 画面全体をずらす配布の効果が何も起こさない
        identifier = self._moved()
        project = Project.create(SETTINGS)
        below = Track(TrackKind.VIDEO, "V0")
        project = AddTrack(below).apply(project)
        project = AddClip(
            below.id, Clip(timeline_start=0, duration=60, source=_shape((1.0, 1.0, 1.0, 1.0), 8))
        ).apply(project)
        above = Track(TrackKind.VIDEO, "V1")
        project = AddTrack(above).apply(project)
        grabbed = Clip(
            timeline_start=0, duration=60, source=GeneratedSource(kind="framebuffer", params={})
        )
        project = AddClip(above.id, grabbed).apply(project)
        definition = registry.get(identifier)
        assert definition is not None
        project = AddEffect(grabbed.id, definition.create()).apply(project)
        image = _render(project, gl_context, 10)
        # 写し取った画面ごと右へ動くので、元の四角の右に同じ四角が出る
        assert int(image[32, 48, 0]) > 250
