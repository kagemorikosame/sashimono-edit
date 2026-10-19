"""方式を途中で切り替えてトラックを変換しても、書き出す絵と音が変わらないこと（Issue #27）

変換（:class:`~sashimono.core.commands.ConvertLayers`）は、映像トラックと音声トラックを
レイヤーへまとめ直す 重なり順・音のストリーム・音声トラックの音量と定位・BGM の
どれか 1 つでも取り落とすと、変換しただけで作品の見た目か音が変わる
実素材を置いて、変換の前後で描いた画素とミキサの音を比べる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest

from sashimono.core.commands import (
    Command,
    ConvertLayers,
    insert_generated,
    place_media,
)
from sashimono.core.model import (
    AnimatedValue,
    GeneratedSource,
    LayerMode,
    Project,
    ProjectSettings,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.audio import AudioMixer
from sashimono.engine.decode import probe_media
from sashimono.engine.encode import ExportSettings, export_project
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from tests.media_fixtures import SampleMedia, make_silent_gap

pytestmark = pytest.mark.usefixtures("gpu")

_SQUARE = GeneratedSource(kind="shape", params={"shape": "rect", "color": (0.0, 1.0, 0.0, 1.0)})
_FRAMES = [0, 15, 45]


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


def _to(project: Project, mode: str) -> Project:
    return ConvertLayers(mode, registry.sound_kinds()).apply(project)


def _built(mode: str, sample: SampleMedia, music: Path) -> Project:
    """音付きの動画 2 本（手前は半透明）と BGM を重ね、その上に四角を置いた作品

    分ける方式なら、音声トラックに音量と定位も付ける レイヤーへまとめたときに
    クリップへ写せないと、ここで音が変わる
    """
    settings = ProjectSettings(width=320, height=240, frame_rate=FrameRate(30), layer_mode=mode)
    project = Project.create(settings)
    video = probe_media(sample.path)
    for item in (video, video, probe_media(music)):
        project = _apply(project, place_media(project, [item], at_frame=0))
    project = _apply(project, insert_generated(project, _SQUARE, at_frame=0, duration=60))
    # 手前の動画を半透明にする 重なり順が入れ替わると、奥の動画が手前に出て画素が変わる
    pictures = [t for t in project.timeline.tracks if t.kind is not TrackKind.AUDIO]
    front = pictures[1]
    clips = tuple(
        replace(c, opacity=AnimatedValue(0.5)) if c.media_id == video.id else c for c in front.clips
    )
    project = project.with_timeline(project.timeline.replace_track(front.with_clips(clips)))
    if mode == LayerMode.SEPARATED:
        tracks = list(project.timeline.tracks)
        sounds = [i for i, t in enumerate(tracks) if t.kind is TrackKind.AUDIO]
        tracks[sounds[0]] = replace(tracks[sounds[0]], volume_db=-6.0, pan=0.4)
        tracks[sounds[-1]] = replace(tracks[sounds[-1]], volume_db=-3.0, pan=-0.5)
        project = project.with_timeline(replace(project.timeline, tracks=tuple(tracks)))
    return project


@pytest.fixture(scope="module")
def music(media_dir: Path) -> Path:
    return make_silent_gap(media_dir, "convert_bgm.wav", duration=2.0)


def _frames(project: Project, context: OffscreenGLContext) -> list[np.ndarray]:
    renderer = FrameRenderer(project, context=context)
    try:
        return [renderer.render(frame) for frame in _FRAMES]
    finally:
        renderer.close()


def _sound(project: Project) -> np.ndarray:
    mixer = AudioMixer(project)
    try:
        # 作品の頭から終わりまで（2 秒） BGM の後半（無音の後の 880 Hz）まで比べる
        return mixer.render(0, 96000)
    finally:
        mixer.close()


def _assert_same(before: Project, after: Project, context: OffscreenGLContext) -> None:
    for frame, left, right in zip(
        _FRAMES, _frames(before, context), _frames(after, context), strict=True
    ):
        assert np.array_equal(left, right), f"{frame} フレーム目の絵が違う"
    left_sound, right_sound = _sound(before), _sound(after)
    assert float(np.abs(left_sound).max()) > 0.01, "変換前が無音で、比べる意味が無い"
    # 音量と定位を写した音量調整は、トラックの音量と同じ倍率を浮動小数で求め直すので
    # 最後の桁が揃わないことがある 聞いて分かる差（1e-5 は -100 dB）ではない
    assert np.allclose(left_sound, right_sound, atol=1e-5)


def test_separated_to_mixed_looks_and_sounds_the_same(
    sample_av: SampleMedia, music: Path, gl_context: OffscreenGLContext
) -> None:
    # 音声トラックの音量や定位を写し損ねると、レイヤーへまとめただけで BGM が大きくなり左右も変わる
    before = _built(LayerMode.SEPARATED, sample_av, music)
    after = _to(before, LayerMode.MIXED)
    assert {t.kind for t in after.timeline.tracks} == {TrackKind.MIXED}
    _assert_same(before, after, gl_context)


def test_mixed_to_separated_looks_and_sounds_the_same(
    sample_av: SampleMedia, music: Path, gl_context: OffscreenGLContext
) -> None:
    # 分けるときに重なり順や音のストリームを取り落とすと、半透明の動画の奥と手前が
    # 入れ替わったり声が消えたりする
    before = _built(LayerMode.MIXED, sample_av, music)
    layers = list(before.timeline.tracks)
    # レイヤーの音量と定位も付ける 分けたとき音声トラックへ移さないと消える
    layers[0] = replace(layers[0], volume_db=-4.0, pan=0.3)
    before = before.with_timeline(replace(before.timeline, tracks=tuple(layers)))
    after = _to(before, LayerMode.SEPARATED)
    assert TrackKind.MIXED not in {t.kind for t in after.timeline.tracks}
    _assert_same(before, after, gl_context)


def test_there_and_back_looks_and_sounds_the_same(
    sample_av: SampleMedia, music: Path, gl_context: OffscreenGLContext
) -> None:
    # 音量を写した音量調整が行き帰りで重なって掛かると、戻したときに音が小さくなる
    before = _built(LayerMode.SEPARATED, sample_av, music)
    _assert_same(before, _to(_to(before, LayerMode.MIXED), LayerMode.SEPARATED), gl_context)


def test_the_exports_match(sample_av: SampleMedia, music: Path, tmp_path: Path) -> None:
    # 書き出しは描画と音の合成を別の道（絵を描く判定・音声の有無）で呼ぶ 変換した側だけ
    # 音が付かない、を拾う
    before = _built(LayerMode.SEPARATED, sample_av, music)
    outputs = []
    for name, project in (("before", before), ("after", _to(before, LayerMode.MIXED))):
        # 音は無圧縮で書く AAC は入力の最後の桁の違いでも量子化の選び方が変わり、
        # 聞いて分からない差が 1e-3 ほどに広がって、比べても何も言えなくなる
        path = tmp_path / f"{name}.mkv"
        settings = ExportSettings(path=path, video_codec="libx264", audio_codec="pcm_s16le")
        export_project(project, settings)
        outputs.append(path)

    def decoded(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
        with av.open(str(path)) as container:
            pictures = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        with av.open(str(path)) as container:
            chunks = [f.to_ndarray() for f in container.decode(audio=0)]
        return pictures, np.concatenate(chunks, axis=1)

    (left_pictures, left_sound), (right_pictures, right_sound) = map(decoded, outputs)
    assert len(left_pictures) == len(right_pictures) > 0
    for number, (a, b) in enumerate(zip(left_pictures, right_pictures, strict=True)):
        assert np.array_equal(a, b), f"{number} 枚目の絵が違う"
    assert left_sound.shape == right_sound.shape
    # 16 bit へ丸めるので、丸めの境目にあった値が 1 段ずれることはある
    assert np.abs(left_sound.astype(np.int32) - right_sound.astype(np.int32)).max() <= 1
