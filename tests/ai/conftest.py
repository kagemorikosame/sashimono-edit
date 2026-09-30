"""AI 層のテスト用のホスト

:class:`~sashimono.ai.host.EditorHost` を満たす偽物を用意する ウィジェットを一切
作らずにツールの挙動を確かめられるのは、AI 層が Qt を知らない作りにしてあるため
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.ai.host import ToolError
from sashimono.core.commands import AddClip, Command, Document, InScene
from sashimono.core.model import (
    ClipId,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    SceneId,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio.waveform import Waveform
from tests.conftest import make_clip

RATE_30 = FrameRate(30)


class FakeHost:
    """編集ソフトのふりをする"""

    def __init__(self, project: Project) -> None:
        self._document = Document(project)
        self.frame = 0
        self.selection: tuple[ClipId, ...] = ()
        self.scene: SceneId | None = None
        self.stopped = 0
        self.rendered: list[tuple[int, int]] = []
        self.analyzed: list[MediaId] = []
        self.probed: list[Path] = []
        self.stub_waveform: Waveform | None = None
        self.probe_result: MediaItem | None = None
        self.transcription = "起こしは走っていません"
        self.transcribed_stream: int | None = None
        #: 本人の設定の「動画の映像と音声」 既定は設定と同じく分ける
        self.split_audio = True

    @property
    def splits_media(self) -> bool:
        return self.split_audio

    @property
    def document(self) -> Document:
        return self._document

    @property
    def project(self) -> Project:
        project = self._document.project
        scene = project.find_scene(self.scene) if self.scene is not None else None
        return project if scene is None else replace(project, timeline=scene.timeline)

    @property
    def active_scene(self) -> SceneId | None:
        return self.scene

    def set_active_scene(self, scene_id: SceneId | None) -> None:
        self.scene = scene_id

    @property
    def playhead(self) -> int:
        return self.frame

    def seek(self, frame: int) -> None:
        self.frame = frame

    @property
    def selected_clip(self) -> ClipId | None:
        return self.selection[-1] if self.selection else None

    def select_clip(self, clip_id: ClipId | None) -> None:
        self.selection = (clip_id,) if clip_id is not None else ()

    @property
    def selected_clips(self) -> tuple[ClipId, ...]:
        return self.selection

    def select_clips(self, clip_ids: list[ClipId]) -> None:
        self.selection = tuple(clip_ids)

    def apply_commands(self, commands: list[Command], label: str) -> None:
        if not commands:
            return
        try:
            with self._document.checkpoint(label):
                for command in commands:
                    if self.scene is not None and not isinstance(command, InScene):
                        command = InScene(self.scene, command)
                    self._document.execute(command)
        except (ValueError, KeyError) as exc:
            raise ToolError(str(exc)) from exc

    def stop_playback(self) -> None:
        self.stopped += 1

    def render_png(self, frame: int, *, width: int) -> bytes:
        self.rendered.append((frame, width))
        # PNG の識別子だけ本物にしておく 中身は誰も見ない
        return b"\x89PNG\r\n\x1a\n" + f"{frame}".encode()

    def probe(self, path: Path) -> MediaItem:
        self.probed.append(path)
        if self.probe_result is None:
            raise ToolError(f"読み込めません: {path}")
        return replace(self.probe_result, path=path)

    def analyze(self, media: MediaItem) -> None:
        self.analyzed.append(media.id)

    def waveform(self, media: MediaItem) -> Waveform | None:
        del media
        return self.stub_waveform

    def start_transcription(
        self, media_id: MediaId, model: str, *, audio_stream: int | None = None
    ) -> str:
        self.transcription = f"{model} で開始"
        #: 起こすように頼まれた音声ストリームの番号
        self.transcribed_stream = audio_stream
        return f"{media_id} の起こしを始めました"

    def transcription_status(self) -> str:
        return self.transcription


def make_loaded(video_media: MediaItem, transcript: Transcript) -> Project:
    """10 秒の素材を 1 本置き、字幕を付けたプロジェクトを組む

    fixture ではなく関数にしてあるのは、別のフォルダのテストからも使うため
    conftest の fixture は、そのフォルダの下からしか見えない
    """
    with_transcript = replace(video_media, transcript=transcript)
    base = Project.create(ProjectSettings(frame_rate=RATE_30), media=(with_transcript,))
    track = Track(kind=TrackKind.VIDEO, name="V1")
    base = base.with_timeline(replace(base.timeline, tracks=(track,)))
    return AddClip(track.id, make_clip(0, 300, with_transcript)).apply(base)


@pytest.fixture
def loaded(video_media: MediaItem, transcript: Transcript) -> Project:
    return make_loaded(video_media, transcript)


@pytest.fixture
def host(loaded: Project, video_media: MediaItem) -> FakeHost:
    created = FakeHost(loaded)
    created.probe_result = MediaItem(
        path=Path("C:/素材/追加.mp4"),
        duration=Fraction(5),
        video_streams=video_media.video_streams,
        audio_streams=video_media.audio_streams,
    )
    return created
