"""混合トラック（YMM4 型のレイヤー Issue #27）の絵と音

混合トラックの音付き動画は、1 本のクリップで絵と音を出す 描画・音の合成・書き出しの
どれか 1 つでも混合トラックを忘れると「プレビューには出るのに書き出すと消える」
「絵は出るのに音が鳴らない」になる
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from sashimono.core.commands import AddClip, AddScene, AddTrack, InScene, new_scene
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio import AudioMixer
from sashimono.engine.decode import probe_media
from sashimono.engine.encode import ExportSettings, export_project
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from tests.media_fixtures import libx264_available

SETTINGS = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30), sample_rate=48000)

#: 素材の 2 本の音の高さ どちらのストリームが鳴ったかを周波数で見分ける
LOW_HZ = 440
HIGH_HZ = 1500


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture(scope="session")
def two_voices(media_dir: Path) -> MediaItem:
    """赤一色の 2 秒の動画に、高さの違う音を 2 本持たせた素材（多言語音声のつもり）"""
    path = media_dir / "two_voices.mp4"
    if not path.exists():
        if not libx264_available():
            pytest.skip("ffmpeg に libx264 が無いので実素材のテストを飛ばす")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        command += ["-f", "lavfi", "-i", "color=c=red:size=64x64:rate=30:duration=2"]
        for hz in (LOW_HZ, HIGH_HZ):
            tone = f"sine=frequency={hz}:duration=2:sample_rate=48000,volume=12dB"
            command += ["-f", "lavfi", "-i", tone]
        command += ["-map", "0:v", "-map", "1:a", "-map", "2:a"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2"]
        command.append(str(path))
        subprocess.run(command, check=True, capture_output=True)
    media = probe_media(path)
    assert len(media.audio_streams) == 2
    return media


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


RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


def _with_tracks(*tracks: Track, media: tuple[MediaItem, ...] = ()) -> Project:
    base = Project.create(SETTINGS, media=media)
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _video_clip(media: MediaItem, stream: int | None = None, **changes: Any) -> Clip:
    """音付きの動画を混合トラックに置いたクリップ ``stream`` は何本目の音か"""
    audio = media.audio_streams[stream].index if stream is not None else None
    clip = Clip(
        timeline_start=0,
        duration=60,
        media_id=media.id,
        stream_index=media.video_streams[0].index,
        audio_stream=audio,
    )
    return replace(clip, **changes)


def _layer_project(media: MediaItem, clip: Clip, **track: Any) -> Project:
    layer = replace(Track(TrackKind.MIXED, "レイヤー 1", (clip,)), **track)
    return _with_tracks(layer, media=(media,))


def _render(project: Project, context: OffscreenGLContext, frame: int = 10) -> np.ndarray:
    renderer = FrameRenderer(project, context=context)
    try:
        return renderer.render(frame)
    finally:
        renderer.close()


def _mix(project: Project, start: int = 0, count: int = 24000) -> np.ndarray:
    mixer = AudioMixer(project)
    try:
        return mixer.render(start, count)
    finally:
        mixer.close()


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _pitch(samples: np.ndarray, sample_rate: int = 48000) -> float:
    """一番強い周波数（Hz）"""
    mono = samples.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1 / sample_rate)[int(np.argmax(spectrum))])


class TestPicture:
    def test_a_video_on_a_layer_is_drawn(
        self, gl_context: OffscreenGLContext, two_voices: MediaItem
    ) -> None:
        # 描く側が映像トラックしか見ないと、レイヤーに置いた動画が何も映らない
        image = _render(_layer_project(two_voices, _video_clip(two_voices, 0)), gl_context)
        assert image[32, 32, 0] > 200 and image[32, 32, 1] < 60

    def test_a_hidden_picture_is_not_drawn(
        self, gl_context: OffscreenGLContext, two_voices: MediaItem
    ) -> None:
        # 音だけ使うつもりで絵を隠した動画がレイヤーに映ったままになり、下の映像を覆う
        clip = _video_clip(two_voices, 0, show_picture=False)
        image = _render(_layer_project(two_voices, clip), gl_context)
        assert int(image[32, 32, 0]) < 8

    def test_a_hidden_clip_is_not_the_shape_to_clip_to(
        self, gl_context: OffscreenGLContext
    ) -> None:
        # 絵を描かないクリップが切り抜きの相手になると、画面全体の形で切り抜いて
        # 上の赤が外まで広がる（相手がいなければ何も映らない）
        square = Clip(0, 30, source=_shape(BLUE, 20.0))
        hidden = Clip(0, 30, source=_shape(GREEN), show_picture=False)
        cover = Clip(0, 30, source=_shape(RED), clip_to_below=True)
        project = _with_tracks(
            Track(TrackKind.MIXED, "レイヤー 1", (square,)),
            Track(TrackKind.MIXED, "レイヤー 2", (hidden,)),
            Track(TrackKind.MIXED, "レイヤー 3", (cover,)),
        )
        image = _render(project, gl_context)
        assert image[32, 32, 0] > 200 and image[32, 32, 2] < 50, "四角の中が赤くない"
        assert int(image[2, 2, 0]) < 30, "四角の外まで赤い"
        assert int(image[2, 2, 1]) < 30, "隠したクリップが描かれている"

    def test_higher_layers_are_drawn_in_front(self, gl_context: OffscreenGLContext) -> None:
        # YMM4・AviUtl と同じく、番号の大きい（並びで後ろの）レイヤーほど手前
        project = _with_tracks(
            Track(TrackKind.MIXED, "レイヤー 1", (Clip(0, 30, source=_shape(RED)),)),
            Track(TrackKind.MIXED, "レイヤー 2", (Clip(0, 30, source=_shape(GREEN, 20.0)),)),
        )
        image = _render(project, gl_context)
        assert int(image[32, 32, 1]) > 200 and int(image[32, 32, 0]) < 30
        assert int(image[2, 2, 0]) > 200

    def test_video_tracks_and_layers_share_one_stack(self, gl_context: OffscreenGLContext) -> None:
        # 種類ごとに重ねると、映像トラックがいつもレイヤーの上（か下）に来る
        # 並びの順に重ねるので、奥のレイヤー・映像トラック・手前のレイヤーの順に出る
        project = _with_tracks(
            Track(TrackKind.MIXED, "レイヤー 1", (Clip(0, 30, source=_shape(RED)),)),
            Track(TrackKind.VIDEO, "V1", (Clip(0, 30, source=_shape(BLUE, 40.0)),)),
            Track(TrackKind.AUDIO, "A1"),
            Track(TrackKind.MIXED, "レイヤー 2", (Clip(0, 30, source=_shape(GREEN, 12.0)),)),
        )
        image = _render(project, gl_context)
        assert int(image[32, 32, 1]) > 200, "手前のレイヤーが映像トラックの下にいる"
        assert int(image[32, 16, 2]) > 200, "映像トラックが奥のレイヤーの下にいる"
        assert int(image[2, 2, 0]) > 200

    def test_a_muted_layer_is_not_drawn(self, gl_context: OffscreenGLContext) -> None:
        # ミュートしたレイヤーの絵が映り続けると、その層を一時的に外して見比べられない
        layer = Track(TrackKind.MIXED, "レイヤー 1", (Clip(0, 30, source=_shape(RED)),), muted=True)
        assert int(_render(_with_tracks(layer), gl_context)[32, 32, 0]) < 8

    def test_a_soloed_layer_hides_video_tracks(self, gl_context: OffscreenGLContext) -> None:
        # ソロは絵の側（映像と混合）の中で決まる
        project = _with_tracks(
            Track(
                TrackKind.MIXED, "レイヤー 1", (Clip(0, 30, source=_shape(RED, 20.0)),), solo=True
            ),
            Track(TrackKind.VIDEO, "V1", (Clip(0, 30, source=_shape(BLUE)),)),
        )
        image = _render(project, gl_context)
        assert int(image[32, 32, 0]) > 200
        assert int(image[2, 2, 2]) < 8

    def test_layers_inside_a_scene_are_drawn(self, gl_context: OffscreenGLContext) -> None:
        # 入れ子の中も同じ道を通る 通らないと、シーンの中のレイヤーだけ消える
        project, _ = _scene_on_a_layer(Project.create(SETTINGS), Clip(0, 60, source=_shape(GREEN)))
        assert int(_render(project, gl_context)[32, 32, 1]) > 200


class TestSound:
    def test_the_chosen_stream_is_heard(self, two_voices: MediaItem) -> None:
        # 絵のストリーム番号で音を開くと、どちらを選んでも 1 本目が鳴る
        low = _mix(_layer_project(two_voices, _video_clip(two_voices, 0)))
        high = _mix(_layer_project(two_voices, _video_clip(two_voices, 1)))
        assert _pitch(low) == pytest.approx(LOW_HZ, abs=10)
        assert _pitch(high) == pytest.approx(HIGH_HZ, abs=10)

    def test_no_stream_is_silent(self, two_voices: MediaItem) -> None:
        # 音を選ばずに置いた動画から音が鳴ると、要らない声や BGM が作品に混ざる
        assert _rms(_mix(_layer_project(two_voices, _video_clip(two_voices, None)))) == 0.0

    def test_a_stream_the_media_lacks_is_silent(self, two_voices: MediaItem) -> None:
        # デコーダは無い番号を頼まれると先頭の音へ逃げる 1 本目の言語が鳴ってはいけない
        clip = replace(_video_clip(two_voices, 0), audio_stream=9)
        assert _rms(_mix(_layer_project(two_voices, clip))) == 0.0

    def test_a_hidden_picture_still_plays(self, two_voices: MediaItem) -> None:
        # 音だけ使いたい動画 絵を隠したら音まで消えると、クリップを分けるしかなくなる
        clip = _video_clip(two_voices, 0, show_picture=False)
        assert _rms(_mix(_layer_project(two_voices, clip))) > 0.01

    def test_the_layer_volume_applies(self, two_voices: MediaItem) -> None:
        # レイヤーの音量を下げても音が変わらないと、BGM のレイヤーだけ絞る調整ができない
        clip = _video_clip(two_voices, 0)
        loud = _rms(_mix(_layer_project(two_voices, clip)))
        quiet = _rms(_mix(_layer_project(two_voices, clip, volume_db=-20.0)))
        assert quiet / loud == pytest.approx(0.1, rel=0.05)

    def test_the_layer_pan_applies(self, two_voices: MediaItem) -> None:
        # レイヤーの左右の振り分けが効かないと、左に寄せたはずの音が両方から聞こえる
        mixed = _mix(_layer_project(two_voices, _video_clip(two_voices, 0), pan=-1.0))
        assert _rms(mixed[:, 0]) > 0.01
        assert _rms(mixed[:, 1]) < 1e-4

    def test_a_muted_layer_is_silent(self, two_voices: MediaItem) -> None:
        # ミュートしたレイヤーの音が鳴り続けると、消したはずの声がプレビューに残る
        project = _layer_project(two_voices, _video_clip(two_voices, 0), muted=True)
        assert _rms(_mix(project)) == 0.0

    def test_solo_elsewhere_silences_the_layer(self, two_voices: MediaItem) -> None:
        # 音声トラックのソロは、混合トラックの音も止める（音の側の中で決まる）
        project = _layer_project(two_voices, _video_clip(two_voices, 0))
        project = AddTrack(Track(TrackKind.AUDIO, "A1", solo=True)).apply(project)
        assert _rms(_mix(project)) == 0.0

    def test_a_soloed_layer_is_heard(self, two_voices: MediaItem) -> None:
        # ソロにしたレイヤーまで黙ると、そのレイヤーの音だけを聞き直せない
        project = _layer_project(two_voices, _video_clip(two_voices, 0), solo=True)
        project = AddTrack(Track(TrackKind.AUDIO, "A1")).apply(project)
        assert _rms(_mix(project)) > 0.01

    def test_video_tracks_still_play_only_scenes(self, two_voices: MediaItem) -> None:
        # 分ける方式の映像トラックの動画は鳴らさない（音は組の音声クリップが鳴らす）
        clip = _video_clip(two_voices, None)
        project = _with_tracks(Track(TrackKind.VIDEO, "V1", (clip,)), media=(two_voices,))
        assert _rms(_mix(project)) == 0.0

    def test_layers_inside_a_scene_are_heard(self, two_voices: MediaItem) -> None:
        # シーンの中のレイヤーに置いた動画の音が鳴らないか、選んだのと違う声が鳴る
        base = Project.create(SETTINGS, media=(two_voices,))
        project, _ = _scene_on_a_layer(base, _video_clip(two_voices, 1))
        assert _pitch(_mix(project)) == pytest.approx(HIGH_HZ, abs=10)


class TestExport:
    pytestmark = pytest.mark.usefixtures("gpu")

    def test_the_file_has_the_layer_picture_and_sound(
        self, two_voices: MediaItem, tmp_path: Path
    ) -> None:
        # プレビューと同じ描画と音の合成を通ること 書き出しだけ映像トラックしか見ないと、
        # レイヤーに置いた作品が黒い無音の動画になる
        output = tmp_path / "layer.mp4"
        export_project(
            _layer_project(two_voices, _video_clip(two_voices, 1)),
            ExportSettings(path=output, frame_range=(0, 30)),
        )
        with av.open(str(output)) as container:
            frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
            assert frame[32, 32, 0] > 200 and frame[32, 32, 1] < 60
        with av.open(str(output)) as container:
            chunks = [f.to_ndarray() for f in container.decode(audio=0)]
        samples = np.concatenate(chunks, axis=1).T
        assert _pitch(samples[2400:26400]) == pytest.approx(HIGH_HZ, abs=10)

    def test_a_silent_layer_writes_no_sound_track(
        self, two_voices: MediaItem, tmp_path: Path
    ) -> None:
        # 鳴らすクリップが無いのに音の道を作ると、黙った音声が付いた動画になる
        output = tmp_path / "silent.mp4"
        export_project(
            _layer_project(two_voices, _video_clip(two_voices, None)),
            ExportSettings(path=output, frame_range=(0, 10)),
        )
        with av.open(str(output)) as container:
            assert not container.streams.audio

    def test_a_disabled_clip_writes_no_sound_track(
        self, two_voices: MediaItem, tmp_path: Path
    ) -> None:
        # ミキサは無効にしたクリップを飛ばす 数えると、黙った音声だけが付く
        output = tmp_path / "disabled.mp4"
        export_project(
            _layer_project(two_voices, _video_clip(two_voices, 0, enabled=False)),
            ExportSettings(path=output, frame_range=(0, 10)),
        )
        with av.open(str(output)) as container:
            assert not container.streams.audio


def _scene_on_a_layer(project: Project, clip: Clip) -> tuple[Project, Clip]:
    """``clip`` をシーンの中のレイヤーに置き、そのシーンを外のレイヤーに置く"""
    scene = new_scene(project, "中")
    project = AddScene(scene).apply(project)
    inner = Track(TrackKind.MIXED, "レイヤー 1")
    project = InScene(scene.id, AddTrack(inner)).apply(project)
    project = InScene(scene.id, AddClip(inner.id, clip)).apply(project)
    outer = Track(TrackKind.MIXED, "レイヤー 1")
    project = AddTrack(outer).apply(project)
    placed = Clip(0, 60, scene_id=scene.id)
    return AddClip(outer.id, placed).apply(project), placed
