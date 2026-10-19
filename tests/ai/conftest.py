"""AI 層のテスト用のホスト

:class:`~sashimono.ai.host.EditorHost` を満たす偽物を用意する ウィジェットを一切
作らずにツールの挙動を確かめられるのは、AI 層が Qt を知らない作りにしてあるため
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.budget import DEFAULT_IMAGE_EDGE
from sashimono.ai.host import EXPORT_RUNNING, ExportRequest, ToolError
from sashimono.ai.outbound import ASKED, SendChoice
from sashimono.ai.sheet import SheetLayout
from sashimono.compat.catalog import TemplateEntry
from sashimono.compat.subtitle_looks import LookRules, SavedLooks, saved_looks
from sashimono.core.commands import AddClip, Author, Command, Document, InScene
from sashimono.core.emphasis import EmphasisStyle
from sashimono.core.io.atomic import Expected, write_atomic_bytes
from sashimono.core.io.library import Library
from sashimono.core.io.serialize import save_project
from sashimono.core.io.subtitles import save_subtitles
from sashimono.core.mixing import DuckShape
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
from sashimono.core.seams import SeamStyle
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.cache import SceneState
from sashimono.engine.checks import CheckRequest, CheckState
from sashimono.engine.checks.report import assemble
from tests.conftest import make_clip

RATE_30 = FrameRate(30)


class FakeSheet:
    """コマを並べる 1 枚のふり 描いた升・フレーム・字を順に覚える"""

    def __init__(self, layout: SheetLayout) -> None:
        self.layout = layout
        self.drawn: list[tuple[int, int, str]] = []
        self.finished = False

    def draw(self, index: int, frame: int, label: str) -> None:
        self.drawn.append((index, frame, label))

    def png(self) -> bytes:
        self.finished = True
        return b"\x89PNG\r\n\x1a\n" + f"sheet{len(self.drawn)}".encode()


class FakeHost:
    """編集ソフトのふりをする"""

    def __init__(self, project: Project) -> None:
        self._document = Document(project)
        self.frame = 0
        self.selection: tuple[ClipId, ...] = ()
        self.scene: SceneId | None = None
        self.stopped = 0
        self.rendered: list[tuple[int, int]] = []
        #: コマを並べた 1 枚（contact_sheet・preview_range） 用意した順
        self.sheets: list[FakeSheet] = []
        self.analyzed: list[MediaId] = []
        self.probed: list[Path] = []
        self.stub_waveform: Waveform | None = None
        self.probe_result: MediaItem | None = None
        self.transcription = "起こしは走っていません"
        self.transcribed_stream: int | None = None
        #: 本人の設定の「動画の映像と音声」 既定は設定と同じく分ける
        self.split_audio = True
        #: 本人の設定の「新しい字幕を自動で折り返す」の幅（%） 既定は設定と同じく 90
        self.wrap_share = 90
        #: 完成物の検査（#337） 本人の設定の音量の基準と、頼まれた検査・試験が入れる様子
        self.loudness = -14.0
        #: ダッキング（#329）の本人の設定 既定は設定と同じ
        self.duck = DuckShape()
        #: 強調の動き（#331）の本人の設定 既定は設定と同じ
        self.emphasis = EmphasisStyle()
        #: ジャンプカットのつなぎ目（#332）の本人の設定 既定は設定と同じ
        self.seams = SeamStyle()
        #: AI に見せる絵の長い辺の上限（#344）の本人の設定 既定は設定と同じ
        self.image_edge = DEFAULT_IMAGE_EDGE
        self.check_requests: list[CheckRequest] = []
        self.check: CheckState = CheckState()
        #: 待つ関数が呼ばれた秒と、待った結果
        self.check_waits: list[float] = []
        self.check_ready = True
        #: 場面の解析（#321） 試験が素材ごとの様子を入れる 無ければ頼んでいない様子
        self.scene_states: dict[MediaId, SceneState] = {}
        self.scene_requests: list[MediaId] = []
        #: 待つ関数が呼ばれた記録（素材, 秒） と、待った結果を決める関数
        self.scene_waits: list[tuple[MediaId, float]] = []
        self.scene_ready: Callable[[MediaId], bool] = lambda _media: True
        #: 作品（.sme）の場所 既定はまだ保存していない作品
        self.saved_at: Path | None = None
        #: 書き出しの窓で前に選んだフォルダ 既定はまだ書き出していない
        self.exported_to: Path | None = None
        #: アプリ自身の設定 get_app_settings が写しを返すかを見るため、そのまま持つ
        self.settings: dict[str, object] = {
            "preferences": {"theme": "dark", "update_check": True},
            "shortcuts": {"編集/分割": "S"},
        }
        #: 保存した場所（write_project を呼ばれた順）
        self.written: list[Path] = []
        #: 静止画に頼まれたフレーム
        self.snapshots: list[int] = []
        #: 頼まれた書き出し 偽物なので書き出しそのものは走らせない
        self.exports: list[ExportRequest] = []
        self.export_state: dict[str, object] = {"state": "idle", "progress": 0.0, "path": None}
        #: 書き出しの終わりを待った（export_waiter の関数を呼んだ）スレッド
        self.export_waits: list[int] = []

        #: アプリが作品の外で覚えている名前（本物は字幕起こしの記録・前に開いた作品の場所など）
        self.remembered: list[str] = []
        #: 保存したプリセットとエイリアス 置き場は試験ごとの一時フォルダ（conftest が向ける）
        self.library = Library()
        #: 本人の設定の着せ方 既定は設定と同じ
        self.rules = LookRules()
        #: テンプレートの棚（#328） 既定は空 本人の AviUtl2 の置き場は読まない
        self.shelf: list[TemplateEntry] = []

    @property
    def saved_looks(self) -> SavedLooks:
        return saved_looks(self.library)

    @property
    def template_shelf(self) -> tuple[TemplateEntry, ...]:
        return tuple(self.shelf)

    @property
    def look_rules(self) -> LookRules:
        return self.rules

    @property
    def splits_media(self) -> bool:
        return self.split_audio

    @property
    def subtitle_wrap_share(self) -> int:
        return self.wrap_share

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
            # 編集画面と同じく AI の印を付ける 付けないと、AI の段を人の段と数えて取り消しを断る
            with self._document.checkpoint(label, by=Author.AI):
                for command in commands:
                    if self.scene is not None and not isinstance(command, InScene):
                        command = InScene(self.scene, command)
                    self._document.execute(command)
        except (ValueError, KeyError) as exc:
            raise ToolError(str(exc)) from exc

    def undo(self) -> None:
        self._document.undo()

    def redo(self) -> None:
        self._document.redo()

    def stop_playback(self) -> None:
        self.stopped += 1

    def render_png(self, frame: int, *, width: int) -> bytes:
        self.rendered.append((frame, width))
        # PNG の識別子だけ本物にしておく 中身は誰も見ない
        return b"\x89PNG\r\n\x1a\n" + f"{frame}".encode()

    def start_sheet(self, layout: SheetLayout) -> FakeSheet:
        sheet = FakeSheet(layout)
        self.sheets.append(sheet)
        return sheet

    def probe(self, path: Path) -> MediaItem:
        self.probed.append(path)
        if self.probe_result is None:
            raise ToolError(f"読み込めません: {path}")
        return replace(self.probe_result, path=path)

    def analyze(self, media: MediaItem) -> None:
        self.analyzed.append(media.id)

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None:
        del media, stream
        return self.stub_waveform

    @property
    def loudness_target(self) -> float:
        return self.loudness

    @property
    def duck_shape(self) -> DuckShape:
        return self.duck

    @property
    def emphasis_style(self) -> EmphasisStyle:
        return self.emphasis

    @property
    def seam_style(self) -> SeamStyle:
        return self.seams

    def request_check(self, request: CheckRequest) -> None:
        self.check_requests.append(request)
        if self.check.request != request:
            # 本物と同じく、新しい頼みは何も調べていない走っている様子から始まる
            empty = assemble(
                [],
                start=request.start,
                end=max(request.end, request.start + 1),
                analyzed=request.start,
                fps=30.0,
                complete=False,
                limits=request.limits,
                index=None,
                loudness=None,
                peak=None,
                length_seconds=None,
            )
            self.check = CheckState(request=request, report=empty, running=True)

    def check_state(self) -> CheckState:
        return self.check

    def check_waiter(self) -> Callable[[float], bool]:
        def wait(seconds: float) -> bool:
            self.check_waits.append(seconds)
            return self.check_ready

        return wait

    def request_scenes(self, media: MediaItem) -> None:
        self.scene_requests.append(media.id)

    def scene_state(self, media: MediaItem) -> SceneState:
        return self.scene_states.get(media.id, SceneState(None, running=False))

    def scene_waiter(self, media_id: MediaId) -> Callable[[float], bool]:
        def wait(seconds: float) -> bool:
            self.scene_waits.append((media_id, seconds))
            return self.scene_ready(media_id)

        return wait

    def start_transcription(
        self, media_id: MediaId, model: str, *, audio_stream: int | None = None
    ) -> str:
        self.transcription = f"{model} で開始"
        #: 起こすように頼まれた音声ストリームの番号
        self.transcribed_stream = audio_stream
        return f"{media_id} の起こしを始めました"

    def transcription_status(self) -> str:
        return self.transcription

    @property
    def project_path(self) -> Path | None:
        return self.saved_at

    @property
    def export_folder(self) -> Path | None:
        return self.exported_to

    def write_project(self, path: Path, *, adopt: bool, expect: Expected) -> str:
        # 画面の〔保存〕と同じく、作品のファイルの書き方（書いてから置き換える）で書く
        save_project(self._document.project, path, expect=expect)
        self.written.append(path)
        if adopt:
            self.saved_at = path
        return ""

    def write_subtitles(
        self, path: Path, *, frame_range: tuple[int, int] | None, expect: Expected
    ) -> Path:
        # 画面の字幕の書き出しと同じ処理で書く
        written = save_subtitles(self.project, path, frame_range=frame_range, expect=expect)
        self.written.append(written)
        return written

    def write_snapshot(self, path: Path, frame: int, *, expect: Expected) -> int:
        # 絵は描かない（偽物） 書いた場所と頼まれたフレームだけ残す
        write_atomic_bytes(path, b"\x89PNG\r\n\x1a\n" + f"{frame}".encode(), expect=expect)
        self.written.append(path)
        self.snapshots.append(frame)
        return frame

    def start_export(self, request: ExportRequest) -> None:
        if self.export_state["state"] in EXPORT_RUNNING:
            raise ToolError("書き出しがもう走っています")
        self.exports.append(request)
        self.export_state = {"state": "rendering", "progress": 0.0, "path": str(request.path)}

    def export_status(self) -> dict[str, object]:
        return dict(self.export_state)

    def export_waiter(self) -> Callable[[float], bool]:
        def wait(seconds: float) -> bool:
            # 呼ばれたスレッドを残す 待つのが AI のスレッドかを試験が見る
            self.export_waits.append(threading.get_ident())
            deadline = time.monotonic() + seconds
            while self.export_state["state"] in EXPORT_RUNNING:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.005)
            return True

        return wait

    def cancel_export(self) -> bool:
        if self.export_state["state"] != "rendering":
            return False
        self.export_state = {**self.export_state, "state": "cancelled"}
        return True

    def app_settings(self) -> dict[str, object]:
        return self.settings

    def remembered_names(self) -> tuple[str, ...]:
        return tuple(self.remembered)


def allow_sending(bridge: EditorBridge) -> EditorBridge:
    """外へ送る物（#342）をすべて「尋ねずに送る」にしたブリッジ

    包み方や待ち合わせを見る試験で、送る前の確かめに答える役を置かずに済ませる
    確かめそのものは tests/ai/test_outbound.py が見る
    """
    bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
    return bridge


def make_loaded(video_media: MediaItem, transcript: Transcript) -> Project:
    """10 秒の素材を 1 本置き、字幕を付けたプロジェクトを組む

    fixture ではなく関数にしてあるのは、別のフォルダのテストからも使うため
    conftest の fixture は、そのフォルダの下からしか見えない
    """
    with_transcript = video_media.with_transcript(transcript)
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
