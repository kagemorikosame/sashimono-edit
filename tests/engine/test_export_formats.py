"""書き出しの形式（#527）

形式ごとに 1 秒を書き出し、PyAV で開き直して入れ物・コーデック・長さ・音の有無を見る
名前の拡張子で入れ物が決まるので、拡張子を変えただけで中身が合わないファイルになっていないかを
読み戻して確かめる 音声だけの形式は描く係を動かさずに書けることも見る
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import av
import pytest

from sashimono.core.commands import AddClip, AddScene, AddTrack, Document, insert_media
from sashimono.core.model import (
    Clip,
    MediaId,
    Project,
    ProjectSettings,
    Scene,
    Timeline,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine import gpu
from sashimono.engine.decode import probe_media
from sashimono.engine.encode import (
    CONTAINER_SUFFIXES,
    FORMATS,
    ExportError,
    ExportSettings,
    build_settings,
    export_project,
    exporter,
    format_named,
    format_of,
    has_sound,
    with_format_suffix,
)
from sashimono.engine.encode.export_job import AviUtlRuntime, export_anywhere, export_job
from sashimono.engine.encoders import PRIORITY_AUTO
from tests.media_fixtures import SampleMedia

#: 書き出す長さ（フレーム 30fps で 1 秒）
SECOND = 30


@pytest.fixture
def project(sample_av: SampleMedia) -> Project:
    """映像 + 音声を 1 本置いた 320x240 / 30fps の作品"""
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(sample_av.path)):
        document.execute(command)
    return document.project


def _picture_only() -> Project:
    """絵だけの 30 コマを置いた作品 素材は要らない（鳴る物が無いことだけを見る）"""
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create())
    return AddClip(track.id, Clip(timeline_start=0, duration=SECOND)).apply(project)


def _settings(path: Path, **overrides: object) -> ExportSettings:
    values: dict[str, object] = {
        "codec": None,
        "bitrate_mbps": 4,
        "frame_range": (0, SECOND),
        "pipeline_depth": 0,
        "decode_threads": 1,
        "smooth_history": True,
        "gpu_color": False,
        "chapters": True,
        "encoder_priority": PRIORITY_AUTO,
    }
    values.update(overrides)
    return build_settings(path, **values)  # type: ignore[arg-type]


def _read(path: Path) -> tuple[str, list[str], list[str], float]:
    """入れ物の名前・映像のコーデック・音声のコーデック・長さ（秒）"""
    with av.open(str(path)) as container:
        videos = [stream.codec_context.name for stream in container.streams.video]
        audios = [stream.codec_context.name for stream in container.streams.audio]
        duration = (container.duration or 0) / av.time_base
        return container.format.name, videos, audios, duration


class TestTheTable:
    """形式の表 窓と AI と書き出しの係が同じ表を引く"""

    def test_every_suffix_is_accepted_and_unique(self) -> None:
        """壊れると、窓で選べる形式を AI の書き出しが知らない拡張子として断る"""
        suffixes = [item.suffix for item in FORMATS]
        assert len(set(suffixes)) == len(suffixes)
        assert set(CONTAINER_SUFFIXES) == set(suffixes)

    def test_the_suffix_is_read_without_case(self) -> None:
        """壊れると、大文字の拡張子の名前を知らない形式として断る（FFmpeg は読める）"""
        found = format_of(Path("作品.WEBM"))
        assert found is not None and found.key == "webm"
        assert format_of(Path("作品.sme")) is None

    @pytest.mark.parametrize(
        ("name", "key", "expected"),
        [
            ("作品.mp4", "webm", "作品.webm"),
            ("作品.v2", "mp3", "作品.v2.mp3"),
            ("作品.MOV", "mov", "作品.MOV"),
            ("作品", "flac", "作品.flac"),
        ],
    )
    def test_the_name_follows_the_format(self, name: str, key: str, expected: str) -> None:
        """壊れると、形式を変えても名前の拡張子が残り、MP4 の名前で WebM を書く"""
        assert with_format_suffix(Path(name), format_named(key)).name == expected

    def test_opus_gets_a_rate_it_takes(self) -> None:
        """壊れると、44.1 kHz の作品（YMM4 から読んだ物など）の WebM だけ Opus を開けずに失敗する"""
        assert exporter._audio_rate("libopus", 44100) == 48000
        assert exporter._audio_rate("aac", 44100) == 44100
        assert exporter._audio_rate("pcm_s16le", 44100) == 44100

    def test_chapters_only_where_the_container_holds_them(self) -> None:
        """壊れると、章を持てない入れ物へ渡して多重化に断られ、書き出しごと失敗する"""
        assert {item.key for item in FORMATS if item.chapters} == {"mp4", "mov", "mkv"}


def _with_scene(sound_inside: bool, kind: TrackKind) -> Project:
    """``kind`` のトラックにシーンを 1 つ置いた作品 シーンの中は音のクリップか絵だけ"""
    inner_kind = TrackKind.AUDIO if sound_inside else TrackKind.VIDEO
    inner = Clip(timeline_start=0, duration=SECOND, media_id=MediaId("音"))
    scene = Scene("BGM", Timeline(rate=FrameRate(30), tracks=(Track(inner_kind, "中", (inner,)),)))
    project = AddScene(scene).apply(Project.create())
    track = Track(kind, "外")
    project = AddTrack(track).apply(project)
    return AddClip(track.id, Clip(timeline_start=0, duration=SECOND, scene_id=scene.id)).apply(
        project
    )


class TestHasSound:
    """鳴る物の判定はミキサと同じ物を数える（PR #547 の指摘）"""

    def test_a_scene_on_a_video_track_sounds(self) -> None:
        """壊れると、映像トラックのシーンの BGM だけが鳴る作品を音声だけで書き出せない"""
        assert has_sound(_with_scene(True, TrackKind.VIDEO))

    def test_a_scene_without_sound_is_silent(self) -> None:
        """壊れると、絵だけのシーンを置いた作品で無音のファイルを書く"""
        assert not has_sound(_with_scene(False, TrackKind.VIDEO))
        assert not has_sound(_with_scene(False, TrackKind.AUDIO))

    def test_disabled_clips_are_silent(self) -> None:
        """壊れると、無効のクリップしか無い音声トラックで無音のファイルを書く"""
        clip = Clip(timeline_start=0, duration=SECOND, media_id=MediaId("音"), enabled=False)
        project = AddTrack(Track(TrackKind.AUDIO, "A1", (clip,))).apply(Project.create())
        assert not has_sound(project)
        enabled = AddTrack(Track(TrackKind.AUDIO, "A1", (replace(clip, enabled=True),))).apply(
            Project.create()
        )
        assert has_sound(enabled)


class TestBuildSettings:
    """窓と AI が通る設定の作り方で、書けない組み合わせを先に断る"""

    @pytest.mark.parametrize("name", ["作品.sme", "a.exe", "名前だけ"])
    def test_an_unknown_suffix_is_refused(self, name: str, tmp_path: Path) -> None:
        """壊れると、作品のファイルや実行できるファイルの名前で書き出せてしまう"""
        with pytest.raises(ValueError, match="で終えてください"):
            _settings(tmp_path / name)

    def test_an_audio_only_format_refuses_a_video_codec(self, tmp_path: Path) -> None:
        """壊れると、映像を書かないのに選んだコーデックが効いたと思わせる"""
        with pytest.raises(ValueError, match="音声だけ"):
            _settings(tmp_path / "a.wav", codec="libx264")

    def test_webm_refuses_h264(self, tmp_path: Path) -> None:
        """壊れると、WebM の入れ物が受けない H.264 で書き始めてから失敗する"""
        with pytest.raises(ValueError, match="WebM"):
            _settings(tmp_path / "a.webm", codec="libx264")

    def test_the_audio_quality_is_used_only_where_it_can_be_chosen(self, tmp_path: Path) -> None:
        """壊れると、映像の形式の音まで音声だけの音質の欄で変わる"""
        assert _settings(tmp_path / "a.mp3", audio_kbps=320).audio_bitrate == 320_000
        assert _settings(tmp_path / "a.mp4", audio_kbps=320).audio_bitrate == 192_000

    def test_an_unknown_quality_is_refused(self, tmp_path: Path) -> None:
        """壊れると、AI が選べない音質を渡して、エンコーダに断られるか黙って丸められる"""
        with pytest.raises(ValueError, match="音質"):
            _settings(tmp_path / "a.mp3", audio_kbps=100)


@pytest.mark.usefixtures("gpu")
class TestVideoFormats:
    @pytest.mark.parametrize(
        ("name", "codec", "container", "video", "audio"),
        [
            ("out.mov", "libx264", "mov", "h264", "aac"),
            ("out.mkv", "libx264", "matroska", "h264", "aac"),
            ("out.webm", None, "matroska", "vp9", "opus"),
            ("out.webm", "libsvtav1", "matroska", "av1", "opus"),
        ],
    )
    def test_each_format_reads_back(
        self,
        project: Project,
        tmp_path: Path,
        name: str,
        codec: str | None,
        container: str,
        video: str,
        audio: str,
    ) -> None:
        """壊れると、拡張子だけ変えた中身の合わないファイルや、音の抜けた WebM を書く"""
        path = export_project(project, _settings(tmp_path / name, codec=codec))
        found, videos, audios, seconds = _read(path)
        assert found.split(",")[0] == container
        # 読み戻すデコーダの名前は入っている物で変わる（libdav1d など）ので、頭で見る
        assert len(videos) == 1 and video in videos[0]
        assert len(audios) == 1 and audio in audios[0]
        assert seconds == pytest.approx(1.0, abs=0.1)

    def test_a_webm_from_another_process(self, project: Project, tmp_path: Path) -> None:
        """壊れると、設定で別のプロセスを選んだ人だけ WebM を書けない（候補が H.264 のまま）"""
        target = tmp_path / "out.webm"
        written = export_anywhere(project, _settings(target), separate_process=True)
        _found, videos, audios, _seconds = _read(written)
        assert "vp9" in videos[0] and "opus" in audios[0]


class TestAudioOnly:
    """音声だけの形式 GPU の無い機械でも書ける（描く係も GL も作らない）"""

    @pytest.fixture(autouse=True)
    def no_drawing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """描く係と GL を作ったら落とす 音声だけの書き出しが映像を描くと、長い作品で待たされる"""

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("音声だけの書き出しが描く係を作った")

        monkeypatch.setattr(exporter, "FrameRenderer", refuse)
        monkeypatch.setattr(exporter, "OffscreenGLContext", refuse)

    @pytest.mark.parametrize(
        ("name", "container", "audio"),
        [
            ("out.wav", "wav", "pcm_s16le"),
            ("out.mp3", "mp3", "mp3"),
            ("out.m4a", "mov", "aac"),
            ("out.flac", "flac", "flac"),
        ],
    )
    def test_each_format_reads_back(
        self, project: Project, tmp_path: Path, name: str, container: str, audio: str
    ) -> None:
        """壊れると、音声だけを頼んだのに映像が付くか、長さが作品と合わない"""
        path = export_project(project, _settings(tmp_path / name))
        found, videos, audios, seconds = _read(path)
        assert found.split(",")[0] == container
        assert videos == []
        assert len(audios) == 1 and audio in audios[0]
        assert seconds == pytest.approx(1.0, abs=0.06)

    def test_the_chosen_quality_reaches_the_file(self, project: Project, tmp_path: Path) -> None:
        """壊れると、窓で 320 kbps を選んでも既定の 192 kbps で書く"""
        path = export_project(project, _settings(tmp_path / "out.mp3", audio_kbps=320))
        with av.open(str(path)) as container:
            assert container.streams.audio[0].bit_rate == 320_000

    def test_a_project_without_sound_is_refused(self, tmp_path: Path) -> None:
        """壊れると、無音のファイルを書き上げて、書き出せたと思って配ってしまう"""
        project = _picture_only()
        assert not has_sound(project)
        target = tmp_path / "out.mp3"
        with pytest.raises(ExportError, match="鳴る物が無い"):
            export_project(project, _settings(target))
        assert not target.exists()

    def test_cancel_leaves_no_file(self, project: Project, tmp_path: Path) -> None:
        """壊れると、中止した書きかけの音のファイルが残り、書けたのかが分からない"""
        target = tmp_path / "out.wav"
        with pytest.raises(ExportError, match="中止"):
            export_project(project, _settings(target), should_cancel=lambda: True)
        assert not target.exists()

    def test_progress_reaches_the_end(self, project: Project, tmp_path: Path) -> None:
        """壊れると、音声だけの書き出しの進み具合が出ないか、終わっても 100% にならない"""
        seen: list[float] = []
        export_project(project, _settings(tmp_path / "out.flac"), progress=seen.append)
        assert seen and seen[-1] == pytest.approx(1.0)
        assert seen == sorted(seen)

    def test_another_process_writes_audio(self, project: Project, tmp_path: Path) -> None:
        """壊れると、設定で別のプロセスを選んだ人だけ音声だけの形式を書けない"""
        target = tmp_path / "out.m4a"
        written = export_anywhere(project, _settings(target), separate_process=True)
        _found, videos, audios, _seconds = _read(written)
        assert videos == [] and "aac" in audios[0]

    def test_the_child_does_not_ask_for_gl(
        self, project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """壊れると、GL を作れない機械で音声だけの書き出しまで子が断り、画面のスレッドへ戻る"""
        monkeypatch.setattr(gpu, "OffscreenGLContext", _no_gl)
        target = tmp_path / "out.wav"
        outcome = export_job(
            progress=lambda _value: None,
            should_cancel=lambda: False,
            project=project,
            settings=_settings(target),
            runtime=AviUtlRuntime.capture(),
        )
        assert outcome is not None and outcome.error is None
        assert target.stat().st_size > 0


def _no_gl() -> None:
    raise gpu.GLContextError("この機械では GL を作れない")
