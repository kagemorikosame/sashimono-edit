"""GPU の道具が無いときの起こしと、音声が何本もある素材で起こす音を選ぶ

利用者の画面では large-v3・日本語・GPU を使う で「起こしに失敗した: Library
cublas64_12.dll is not found or cannot be loaded」と出て止まった pip で入れた CUDA
ランタイムは ``nvidia/cublas/bin`` に DLL を置き、CTranslate2 はそこを探さない
読めないときは CPU へ落として起こし、何が足りないかと入れ方を知らせる
faster-whisper も CUDA も入れずに、代わりのモデルで確かめる（試験のために落とさない）
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.asr import environment
from sashimono.asr.backend import AsrError, TranscribeOptions
from sashimono.asr.service import JobKind, TranscriptionService
from sashimono.asr.whisper import FasterWhisperBackend, is_cuda_library_error
from sashimono.core.model import MediaId
from sashimono.engine.decode import probe_media
from tests.media_fixtures import SampleMedia

MISSING = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


class _Segment:
    def __init__(self) -> None:
        self.start, self.end, self.text = 0.5, 1.0, "こんにちは"
        self.words: list[Any] = []


class _Info:
    duration = 1.0
    language = "ja"


class _Model:
    """渡された音を覚える代わりのモデル ``fail`` なら起こし始めてから cuBLAS を読みに行って落ちる"""

    def __init__(self, fail: BaseException | None = None) -> None:
        self.fail = fail
        self.audio: Any = None

    def transcribe(self, audio: Any, **_: Any) -> tuple[Any, _Info]:
        self.audio = audio

        def segments() -> Any:
            if self.fail is not None:
                raise self.fail
            yield _Segment()

        return segments(), _Info()


def _backend(
    monkeypatch: pytest.MonkeyPatch, gpu: _Model | Exception, cpu: _Model
) -> tuple[FasterWhisperBackend, list[str]]:
    """device ごとに別のモデルを返す 読んだ device の並びも返す"""
    backend = FasterWhisperBackend()
    loaded: list[str] = []

    def load(options: TranscribeOptions) -> Any:
        loaded.append(options.device)
        if options.device == "cpu":
            return cpu
        if isinstance(gpu, Exception):
            raise AsrError(f"モデルを読み込めない ({options.model}): {gpu}") from gpu
        return gpu

    monkeypatch.setattr(backend, "_ensure_model", load)
    return backend, loaded


class TestFallingBackToTheCpu:
    def test_a_missing_cublas_while_running_is_redone_on_the_cpu(
        self, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 利用者の画面の形 前はここで「起こしに失敗した」と止まり、何も起こせなかった
        cpu = _Model()
        backend, loaded = _backend(monkeypatch, _Model(fail=MISSING), cpu)
        messages: list[str] = []
        transcript = backend.transcribe(
            sample_av.path,
            TranscribeOptions(device="cuda", compute_type="float16"),
            progress=lambda _ratio, message: messages.append(message),
        )
        assert transcript is not None and len(transcript) == 1
        assert loaded == ["cuda", "cpu"]
        assert cpu.audio is not None
        notice = backend.take_notice()
        assert "cublas64_12.dll" in notice and "CPU" in notice
        assert any("CPU" in m for m in messages)
        # 受け取ったら空になる 次の起こしに前の知らせが付かない
        assert backend.take_notice() == ""

    def test_a_model_that_cannot_load_on_the_gpu_loads_on_the_cpu(
        self, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, loaded = _backend(monkeypatch, MISSING, _Model())
        transcript = backend.transcribe(sample_av.path, TranscribeOptions(device="cuda"))
        assert transcript is not None
        assert loaded == ["cuda", "cpu"]

    def test_the_cpu_is_not_retried(
        self, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CPU で落ちたのを CPU で起こし直しても同じ 同じ失敗を 2 回待たせない
        backend, loaded = _backend(monkeypatch, _Model(), _Model(fail=MISSING))
        with pytest.raises(AsrError, match="起こしに失敗した"):
            backend.transcribe(sample_av.path, TranscribeOptions(device="cpu"))
        assert loaded == ["cpu"]

    def test_other_failures_are_not_hidden(
        self, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # GPU と関係の無い失敗まで CPU で起こし直すと、原因が見えないまま遅くなる
        other = RuntimeError("モデルのファイルが壊れている")
        backend, loaded = _backend(monkeypatch, _Model(fail=other), _Model())
        with pytest.raises(AsrError, match="壊れている"):
            backend.transcribe(sample_av.path, TranscribeOptions(device="cuda"))
        assert loaded == ["cuda"]

    def test_the_notice_says_how_to_install_when_the_runtime_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sashimono.asr import whisper

        status = environment.ASR_PACK.status()
        monkeypatch.setattr(whisper, "runtime_status", lambda: status)
        text = whisper.gpu_fallback_notice(MISSING)
        if not status.extra_installed:
            assert "環境を更新" in text and "GPU を使う" in text
        else:
            assert "入れ直して" in text

    def test_the_finished_message_carries_the_notice(
        self, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 窓は起こし終えたら閉じる 知らせを完了の出来事に載せないと、CPU で遅かった理由が消える
        backend, _ = _backend(monkeypatch, _Model(fail=MISSING), _Model())
        service = TranscriptionService(backend)
        job = service.start(MediaId("m"), sample_av.path, TranscribeOptions(device="cuda"))
        assert job.wait(30)
        done = [e for e in job.poll() if e.kind is JobKind.DONE]
        assert done and "CPU" in done[0].notice and "CPU" in done[0].message

    def test_it_knows_a_cuda_failure(self) -> None:
        assert is_cuda_library_error(MISSING)
        assert is_cuda_library_error(RuntimeError("CUDA driver version is insufficient"))
        assert not is_cuda_library_error(RuntimeError("音声を読めない"))


class TestFindingTheRuntime:
    def test_the_pip_cuda_folders_are_put_on_the_search_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 足さないと、CUDA ランタイムを入れてあっても cublas64_12.dll が見つからない
        folder = tmp_path / "nvidia" / "cublas" / "bin"
        folder.mkdir(parents=True)
        (folder / "cublas64_12.dll").write_bytes(b"")
        (tmp_path / "nvidia" / "cuda_nvrtc").mkdir()  # DLL の無い物は足さない
        added: list[str] = []
        monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
        monkeypatch.setattr(environment, "_REGISTERED", {})
        monkeypatch.setenv("PATH", "C:\\before")
        found = environment.register_cuda_libraries([str(tmp_path)])
        assert found == [folder]
        assert added == [str(folder)]
        assert os.environ["PATH"].split(os.pathsep)[0] == str(folder)
        # 2 回呼んでも PATH に重ねない
        environment.register_cuda_libraries([str(tmp_path)])
        assert os.environ["PATH"].split(os.pathsep).count(str(folder)) == 1

    def test_loading_on_the_gpu_registers_them_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sashimono.asr import whisper

        called: list[bool] = []

        def register() -> list[Path]:
            called.append(True)
            return []

        monkeypatch.setattr(whisper, "register_cuda_libraries", register)
        with pytest.raises(AsrError):
            # faster-whisper は入っていないので読み込みは断られる 足すのはその前
            FasterWhisperBackend()._ensure_model(TranscribeOptions(device="cuda", model="?"))
        assert called


def _two_voices(directory: Path) -> Path:
    """音声 1 に 440Hz、音声 2 に 1000Hz を持つ動画（ゲームの音とマイクの声の形）"""
    path = directory / "two-voices.mkv"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=1:sample_rate=48000",
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-c:v", "libx264", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return path


def _dominant(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio[:8192] * np.hanning(8192)))
    return float(np.argmax(spectrum) * 16000 / 8192)


class TestChoosingTheVoice:
    def test_the_chosen_stream_is_the_one_transcribed(
        self, media_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 前は 1 本目（ゲームの音）しか起こせず、マイクの声を字幕にできなかった
        path = _two_voices(media_dir)
        streams = probe_media(path).audio_streams
        assert len(streams) == 2
        model = _Model()
        backend = FasterWhisperBackend()
        monkeypatch.setattr(backend, "_ensure_model", lambda options: model)
        backend.transcribe(path, TranscribeOptions(device="cpu"))
        assert _dominant(model.audio) == pytest.approx(440, abs=20)
        backend.transcribe(path, TranscribeOptions(device="cpu", audio_stream=streams[1].index))
        assert _dominant(model.audio) == pytest.approx(1000, abs=20)

    def test_a_stream_the_media_lacks_is_refused(
        self, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FasterWhisperBackend()
        monkeypatch.setattr(backend, "_ensure_model", lambda options: _Model())
        with pytest.raises(AsrError, match="素材に無い"):
            backend.transcribe(sample_av.path, TranscribeOptions(device="cpu", audio_stream=9))

    def test_the_dialog_lists_the_voices(self, media_dir: Path, qt_application: object) -> None:
        del qt_application
        from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog

        media = probe_media(_two_voices(media_dir))
        dialog = TranscribeDialog(media, TranscriptionService(FasterWhisperBackend()))
        try:
            assert dialog._stream.count() == 2
            dialog._stream.setCurrentIndex(1)
            assert dialog._stream.currentData() == media.audio_streams[1].index
        finally:
            dialog.deleteLater()


class TestTheDialog:
    def test_the_gpu_can_be_added_after_installing_without_it(
        self, sample_av: SampleMedia, qt_application: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 前は CUDA ランタイム無しで入れた後、「GPU を使う」が押せず、
        # 後から GPU 版にする道が無かった
        del qt_application
        from sashimono.runtime import PackageStatus, PackStatus
        from sashimono.ui.subtitle import transcribe_dialog

        pack = environment.ASR_PACK
        status = PackStatus(
            pack=pack,
            packages=tuple(PackageStatus(name, "9.9") for name in pack.required),
            extras=tuple(PackageStatus(name, None) for name in pack.extra),
        )
        monkeypatch.setattr(transcribe_dialog, "runtime_status", lambda: status)
        dialog = transcribe_dialog.TranscribeDialog(
            probe_media(sample_av.path), TranscriptionService(FasterWhisperBackend())
        )
        try:
            assert dialog._gpu.isEnabled()
            assert not dialog._gpu.isChecked()
            dialog._gpu.setChecked(True)
            assert "環境を更新" in dialog._status.text()
            assert dialog._install_button.text() == "環境を更新"
        finally:
            dialog.deleteLater()
