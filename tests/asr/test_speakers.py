"""話者の分離（#323）の実行係と、外の部品（sherpa-onnx）との継ぎ目

部品は CI に無いので、部品の形をまねた代わりの物で継ぎ目を通す 本物の部品で分けた結果は
PR の本文に書いた（日本語の合成の 3 人の声 59 秒を 4 秒で分け、1 か所の取り違え）
"""

from __future__ import annotations

import sys
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sashimono.asr import speakers as speakers_module
from sashimono.asr.backend import AsrError, Progress, ShouldCancel, TranscribeOptions
from sashimono.asr.service import JobKind, TranscriptionService
from sashimono.asr.speakers import (
    SPEAKER_PACK,
    SherpaSpeakerBackend,
    SpeakerOptions,
    _cluster,
    separate_with_retries,
)
from sashimono.core.model import MediaId
from sashimono.core.speakers import SpeakerTurn
from tests.asr.test_service import FakeBackend


def _turns(*speakers: int) -> tuple[SpeakerTurn, ...]:
    return tuple(SpeakerTurn(Fraction(n), Fraction(n + 1), who) for n, who in enumerate(speakers))


class TestRetries:
    def test_too_few_speakers_ask_again_with_one_more(self) -> None:
        """壊れると、3 人と渡したのに部品が 2 人にまとめ、3 人目の字幕が別の人の名前になる"""
        asked: list[int] = []

        def run(clusters: int) -> tuple[SpeakerTurn, ...]:
            asked.append(clusters)
            # sherpa-onnx の速いまとめは小さな塊を落とし、渡した数より 1 人少なく出る
            return _turns(*range(clusters - 1))

        result = separate_with_retries(run, 3)
        assert asked == [3, 4]
        assert result is not None and len({t.speaker for t in result}) == 3

    def test_more_than_asked_is_not_used(self) -> None:
        """壊れると、2 人と渡したのに 3 人に割った回を使い、1 人が 2 つの名前に分かれる"""

        def run(clusters: int) -> tuple[SpeakerTurn, ...]:
            return _turns(0) if clusters == 2 else _turns(0, 1, 2)

        result = separate_with_retries(run, 2)
        assert result == _turns(0)

    def test_automatic_count_runs_once_and_cancel_stops(self) -> None:
        """壊れると、人数を自動にしたのに何度もまとめ直して待たされるか、止めても走り続ける"""
        asked: list[int] = []

        def run(clusters: int) -> tuple[SpeakerTurn, ...] | None:
            asked.append(clusters)
            return None

        assert separate_with_retries(run, None) is None
        assert separate_with_retries(run, 3) is None
        assert asked == [-1, 3]

    def test_the_count_is_checked(self) -> None:
        """壊れると、0 人や 50 人を渡せて、部品の中で落ちるか意味の無い分け方になる"""
        with pytest.raises(ValueError):
            SpeakerOptions(speakers=0)
        with pytest.raises(ValueError):
            SpeakerOptions(speakers=50)


class _FakeDiarization:
    """sherpa-onnx の OfflineSpeakerDiarization の代わり 区切りの数だけ進み具合を知らせる"""

    def __init__(self, config: Any) -> None:
        self.config = config

    def process(self, samples: Any, callback: Any = None) -> Any:
        del samples
        for done in range(1, 4):
            if callback is not None and callback(done, 3) != 0:
                raise RuntimeError("止めた")
        segments = [
            SimpleNamespace(start=0.0, end=1.25, speaker=1),
            SimpleNamespace(start=1.25, end=1.25, speaker=0),
            SimpleNamespace(start=2.0, end=3.5, speaker=0),
        ]
        return SimpleNamespace(sort_by_start_time=lambda: segments)


def _fake_sherpa(valid: bool = True) -> Any:
    def config(**kwargs: Any) -> Any:
        return SimpleNamespace(validate=lambda: valid, **kwargs)

    return SimpleNamespace(
        OfflineSpeakerDiarizationConfig=config,
        OfflineSpeakerSegmentationModelConfig=lambda **kw: SimpleNamespace(**kw),
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda **kw: SimpleNamespace(**kw),
        SpeakerEmbeddingExtractorConfig=lambda **kw: SimpleNamespace(**kw),
        FastClusteringConfig=lambda **kw: SimpleNamespace(**kw),
        OfflineSpeakerDiarization=_FakeDiarization,
    )


class TestTheSeamWithThePart:
    def test_turns_come_back_in_source_seconds(self) -> None:
        """壊れると、部品の秒（float）のまま入って字幕との重なりが揺れるか、長さ 0 の区間が残る"""
        samples = np.zeros(16000, dtype=np.float32)
        seen: list[float] = []
        result = _cluster(
            _fake_sherpa(), "seg", "emb", samples, 2, lambda r, _m: seen.append(r), None
        )
        assert result == (
            SpeakerTurn(Fraction(0), Fraction(5, 4), 1),
            SpeakerTurn(Fraction(2), Fraction(7, 2), 0),
        )
        assert seen and seen[-1] == pytest.approx(1.0)

    def test_stopping_returns_none_instead_of_failing(self) -> None:
        """壊れると、窓の〔中断〕で止めたのに「話者を分けられない」と失敗の知らせが出る"""
        samples = np.zeros(16000, dtype=np.float32)
        result = _cluster(_fake_sherpa(), "seg", "emb", samples, -1, None, lambda: True)
        assert result is None

    def test_a_rejected_config_is_an_asr_error(self) -> None:
        """壊れると、壊れたモデルで部品が設定を断ったときにアプリごと落ちる"""
        samples = np.zeros(16000, dtype=np.float32)
        with pytest.raises(AsrError, match="受け付けない"):
            _cluster(_fake_sherpa(valid=False), "seg", "emb", samples, -1, None, None)

    def test_without_the_part_the_user_is_told_where_to_install(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """壊れると、部品を入れていない人に import の英語の失敗だけが出て、入れ方が分からない"""
        monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
        source = tmp_path / "対談.wav"
        source.write_bytes(b"")
        with pytest.raises(AsrError, match="話者を分ける"):
            SherpaSpeakerBackend().separate(source, SpeakerOptions())

    def test_the_pack_is_separate_from_transcription(self) -> None:
        """壊れると、字幕起こしだけ使う人まで話者の分離の部品を落とすか、入れる物が分からない"""
        from sashimono.asr.environment import ASR_PACK

        assert SPEAKER_PACK.key != ASR_PACK.key
        assert any(r.startswith("sherpa-onnx") for r in SPEAKER_PACK.required)
        assert not any(r.startswith("sherpa-onnx") for r in ASR_PACK.requirements(extra=True))

    def test_models_go_to_the_transcription_model_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """壊れると、モデルを字幕起こしと別の置き場へ落とし、他の道具で落とし済みの物を二重に持つ"""
        asked: list[tuple[str, str, str]] = []

        def download(repository: str, filename: str, *, cache_dir: str) -> str:
            asked.append((repository, filename, cache_dir))
            return str(tmp_path / filename)

        monkeypatch.setitem(
            sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
        )
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        files = SherpaSpeakerBackend()._models()
        assert [Path(f).name for f in files] == [
            speakers_module.SEGMENTATION_FILE,
            speakers_module.EMBEDDING_FILE,
        ]
        assert {cache for _r, _f, cache in asked} == {str(tmp_path / "hf" / "hub")}


class _FakeSpeakers:
    """代わりの分離係 ``gate`` が開くまで待つ"""

    def __init__(self, turns: tuple[SpeakerTurn, ...] | None = None) -> None:
        self.turns = turns if turns is not None else _turns(0, 1)
        self.gate = threading.Event()
        self.gate.set()
        self.options: list[SpeakerOptions] = []

    def is_available(self) -> bool:
        return True

    def separate(
        self,
        path: Path,
        options: SpeakerOptions,
        *,
        progress: Progress | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> tuple[SpeakerTurn, ...] | None:
        del path
        self.gate.wait(5.0)
        self.options.append(options)
        if progress is not None:
            progress(0.5, "半分")
        if should_cancel is not None and should_cancel():
            return None
        return self.turns


class TestTheQueue:
    def test_separation_waits_behind_transcription_in_one_queue(self) -> None:
        """壊れると、起こしと分離が同時に走って機械を取り合い、どちらも遅くなるか落ちる"""
        gate = threading.Event()
        speakers = _FakeSpeakers()
        service = TranscriptionService(FakeBackend(block=gate), speakers)
        media = MediaId("m")
        first = service.start(media, Path("a.wav"), TranscribeOptions(audio_stream=2))
        second = service.start_speakers(media, Path("a.wav"), SpeakerOptions(audio_stream=2))
        assert second.waiting and second.separating and second.stream == 2
        assert service.find(media, 2, separating=True) is second
        # 起こしと分離は別の依頼として探す 同じ音でも、起こしているのを分けていると取り違えない
        assert service.find(media, 2) is first
        gate.set()
        assert first.wait(5) and second.wait(5)
        events = second.poll()
        done = [e for e in events if e.kind is JobKind.DONE]
        assert done and done[0].speakers == _turns(0, 1) and "2 人" in done[0].message

    def test_a_failure_of_the_part_is_reported_not_raised(self) -> None:
        """壊れると、外の部品の思わぬ失敗で裏の列が止まり、後の起こしも走らなくなる"""

        class Broken(_FakeSpeakers):
            def separate(self, *args: Any, **kwargs: Any) -> tuple[SpeakerTurn, ...] | None:
                raise RuntimeError("DLL")

        service = TranscriptionService(FakeBackend(), Broken())
        job = service.start_speakers(MediaId("m"), Path("a.wav"), SpeakerOptions())
        assert job.wait(5)
        kinds = [e.kind for e in job.poll()]
        assert kinds[-1] is JobKind.FAILED
