"""リバーブ・ディレイ・音程の調整（利用者の要望 AviUtl2 には無い音のエフェクト）

どれも前の音を読む ミキサは塊を細かく切って頼むので、前の塊を覚えずに読み直して
掛ける 読み直しが足りないと、プレビュー（1024 サンプルずつ）と書き出し（1 コマずつ）で
音が変わり、塊の切れ目でやまびこが途切れる 音は読み戻して数で確かめる
"""

from __future__ import annotations

import itertools
import wave
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest

from sashimono.core.commands import AddEffect, Document, insert_media
from sashimono.core.model import Project, ProjectSettings, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.audio import AudioMixer
from sashimono.engine.decode import probe_media

RATE = 48000


def _wav(path: Path, samples: np.ndarray) -> Path:
    """モノラルの 16 bit WAV を書く ffmpeg が無くても作れるように標準の道具で"""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
    return path


def _click(tmp_path: Path) -> Path:
    """頭に 1 つだけ音のある 1 秒 遅れて出てくる音はエフェクトが作った物"""
    samples = np.zeros(RATE, dtype=np.float32)
    samples[0] = 0.5
    return _wav(tmp_path / "click.wav", samples)


def _sine(tmp_path: Path, frequency: float = 440.0) -> Path:
    t = np.arange(RATE) / RATE
    return _wav(tmp_path / "sine.wav", 0.4 * np.sin(2 * np.pi * frequency * t))


def _project(path: Path, kind: str, **values: float) -> Project:
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(path)):
        document.execute(command)
    clip = next(
        c
        for t in document.project.timeline.tracks
        if t.kind in (TrackKind.AUDIO, TrackKind.MIXED)
        for c in t.clips
    )
    document.execute(AddEffect(clip.id, registry.require(kind).create(**values)))
    return document.project


def _render(project: Project, block: int) -> np.ndarray:
    mixer = AudioMixer(project)
    try:
        parts = [mixer.render(start, min(block, RATE - start)) for start in range(0, RATE, block)]
    finally:
        mixer.close()
    return np.concatenate(parts)[:, 0]


class TestDelay:
    def test_the_echo_comes_after_the_time(self, tmp_path: Path) -> None:
        # 100ms ごとに、量 50% から繰り返し 50% ずつ小さくなるやまびこ
        project = _project(_click(tmp_path), "audio_delay", time=100, feedback=50, mix=50)
        out = _render(project, RATE)
        step = RATE // 10
        # モノラルの素材は左右へ分けるときに小さくなるので、元の音の大きさとの比で見る
        first = float(out[0])
        assert first > 0.1
        assert out[step] == pytest.approx(first * 0.5, abs=1e-3)
        assert out[step * 2] == pytest.approx(first * 0.25, abs=1e-3)
        assert abs(out[step // 2]) < 1e-4

    def test_small_blocks_hear_the_same(self, tmp_path: Path) -> None:
        # 壊れると、塊の頭より前の音を読めず、プレビューではやまびこが消える
        project = _project(_click(tmp_path), "audio_delay", time=100, feedback=50, mix=50)
        assert np.allclose(_render(project, 1024), _render(project, RATE), atol=1e-5)


class TestALongEcho:
    """10 秒を超えてまだ聞こえるやまびこを途切れさせない（PR #231 の指摘）

    前は前の音を 10 秒で切って読み、繰り返しも 50 回で切っていた 2000ms・90% の
    やまびこは 6 回目（12 秒後）でもまだ元の 6 割の大きさなのに、そこで消えた
    """

    SECONDS = 30

    @pytest.fixture
    def far(self, tmp_path: Path) -> Project:
        samples = np.zeros(RATE * self.SECONDS, dtype=np.float32)
        samples[0] = 0.5
        path = _wav(tmp_path / "far-click.wav", samples)
        return _project(path, "audio_delay", time=2000, feedback=90, mix=100)

    def test_every_audible_repeat_is_counted(self) -> None:
        from sashimono.effects.audio import _delay_repeats

        # 量 100%・繰り返し 90% は 66 回目まで -60 dB より大きい
        assert _delay_repeats({"time": 2000, "feedback": 90, "mix": 100}) == 66
        # 小さい設定は今までどおり少ない数で済む
        assert _delay_repeats({"time": 250, "feedback": 0, "mix": 50}) == 1

    def test_the_echo_after_ten_seconds_is_still_heard(self, far: Project) -> None:
        mixer = AudioMixer(far)
        try:
            first = float(mixer.render(0, 16)[0, 0])
            at = RATE * 12
            heard = mixer.render(at - 8, 16)[8, 0]
        finally:
            mixer.close()
        assert first > 0.1
        assert heard == pytest.approx(first * 0.9**5, rel=1e-3)

    def test_playing_on_and_jumping_there_hear_the_same(self, far: Project) -> None:
        # 区切りの頭から読むか、続けて再生してきたかで音が変わらない（再生と書き出しが同じ音）
        at = RATE * 14
        jumped_mixer = AudioMixer(far)
        jumped = jumped_mixer.render(at, RATE)
        jumped_mixer.close()
        played_mixer = AudioMixer(far)
        for start in range(RATE * 13, at, 1024):
            played_mixer.render(start, min(1024, at - start))
        played = played_mixer.render(at, RATE)
        played_mixer.close()
        assert np.abs(jumped - played).max() < 1e-5
        assert np.abs(jumped).max() > 0.1


class TestTheTapeGrowsAsNeeded:
    """元の音の入れ物は要る分だけ取る（PR #231 の指摘）

    前は値で決まる最も長い読み戻し（2000ms・90% で 132 秒）で最初から取り、クリップの
    頭の小さな区切りしか鳴らしていなくても 100 MB を持った
    """

    def test_the_head_of_a_long_echo_takes_little(self, tmp_path: Path) -> None:
        samples = np.zeros(RATE * 3, dtype=np.float32)
        samples[0] = 0.5
        project = _project(
            _wav(tmp_path / "head.wav", samples), "audio_delay", time=2000, feedback=90, mix=100
        )
        mixer = AudioMixer(project)
        try:
            for start in range(0, RATE, 1024):
                mixer.render(start, 1024)
            taken = sum(value[1]._ring.nbytes for value in mixer._raw.values())
        finally:
            mixer.close()
        # 1 秒鳴らすのに要るのは 1 秒と区切り数個ぶん 4 MB あれば足りる
        assert 0 < taken < 4 * 1024 * 1024

    def test_it_reads_back_what_was_written(self) -> None:
        from sashimono.engine.audio.mixer import _Tape

        rng = np.random.default_rng(7)
        whole = rng.standard_normal((200_000, 2)).astype(np.float32)
        tape = _Tape(0, whole[:100])
        written = 100
        keep = 40
        first = tape.capacity
        need = 0
        for count in (7, 30_000, 13, 9_999, 1, 64, 20_000, 5, 15_000, 40_000, 333, 25_000):
            need = max(need, len(tape.samples) + count)
            tape.append(whole[written : written + count], longest=60_000)
            written += count
            tape.trim(keep)
            keep = min(keep * 2, 30_000)
            # 輪を回り込んで書いても、切り出しは繋ぎ目の無い元の並び
            assert tape.reach == written
            assert np.array_equal(tape.samples, whole[tape.origin : written])
            # 上限（60000）より広くは取らない 一度に要る量が上限を越えたときだけその分まで
            assert tape.capacity <= max(60_000, need)
        # 足りなくなって広げた
        assert tape.capacity > first

    def test_it_shrinks_when_far_less_is_kept(self) -> None:
        from sashimono.engine.audio.mixer import _TAPE_SLACK, _Tape

        tape = _Tape(0, np.zeros((_TAPE_SLACK * 10, 2), dtype=np.float32))
        big = tape.capacity
        tape.trim(10)
        assert tape.capacity < big // 4
        assert tape.reach == _TAPE_SLACK * 10


class TestReverb:
    def test_it_leaves_a_tail(self, tmp_path: Path) -> None:
        project = _project(_click(tmp_path), "audio_reverb", decay=0.5, mix=50)
        out = _render(project, RATE)
        tail = np.abs(out[RATE // 20 : RATE // 4])
        assert tail.max() > 1e-4
        # 長さを過ぎたら消えている（60 dB 下がる）
        assert np.abs(out[int(RATE * 0.6) :]).max() < 1e-3

    def test_small_blocks_hear_the_same(self, tmp_path: Path) -> None:
        project = _project(_click(tmp_path), "audio_reverb", decay=0.5, mix=50)
        assert np.allclose(_render(project, 1024), _render(project, RATE), atol=1e-4)

    def test_nothing_changes_at_zero(self, tmp_path: Path) -> None:
        plain = _project(_click(tmp_path), "audio_volume")
        dry = _project(_click(tmp_path), "audio_reverb", mix=0)
        assert np.allclose(_render(plain, RATE), _render(dry, RATE))


def _peak_frequency(samples: np.ndarray) -> float:
    part = samples[RATE // 4 : RATE // 4 + 16384] * np.hanning(16384)
    spectrum = np.abs(np.fft.rfft(part))
    return float(np.argmax(spectrum) * RATE / 16384)


class TestPitch:
    def test_twelve_semitones_is_an_octave(self, tmp_path: Path) -> None:
        # 440Hz を 12 半音上げると 880Hz 長さは変えない（テープを速く回すのとは別）
        project = _project(_sine(tmp_path), "audio_pitch", semitones=12)
        out = _render(project, RATE)
        assert len(out) == RATE
        assert _peak_frequency(out) == pytest.approx(880.0, abs=6.0)

    def test_down_as_well(self, tmp_path: Path) -> None:
        project = _project(_sine(tmp_path), "audio_pitch", semitones=-12)
        assert _peak_frequency(_render(project, RATE)) == pytest.approx(220.0, abs=6.0)

    def test_small_blocks_hear_the_same(self, tmp_path: Path) -> None:
        project = _project(_sine(tmp_path), "audio_pitch", semitones=5)
        assert np.allclose(_render(project, 1024), _render(project, RATE), atol=1e-4)


@pytest.mark.usefixtures("gpu")
def test_the_export_carries_the_echo(tmp_path: Path) -> None:
    """書き出した動画の音にもやまびこが入る（ミキサとは別の道で掛け忘れていないか）"""
    from sashimono.engine.encode import ExportSettings, available_video_codecs, export_project

    if not available_video_codecs():
        pytest.skip("映像のコーデックが無い")
    project = _project(_click(tmp_path), "audio_delay", time=200, feedback=0, mix=80)
    output = tmp_path / "echo.mp4"
    export_project(project, ExportSettings(path=output))
    with av.open(str(output)) as container:
        stream = container.streams.audio[0]
        # AAC は左右を分けた形（planar）で出てくる 1 行目が左
        heard = np.concatenate([frame.to_ndarray()[0] for frame in container.decode(stream)])
        rate = stream.rate
    # AAC は頭に詰め物が入るので、一番大きい所（元の音）からの間で見る
    first = int(np.argmax(np.abs(heard[: rate // 10])))
    echo = first + rate // 5
    window = np.abs(heard[echo - rate // 100 : echo + rate // 100])
    quiet = np.abs(heard[first + rate // 20 : first + rate // 10])
    assert window.max() > 0.1
    assert window.max() > quiet.max() * 5


class TestPlaybackBlocks:
    """再生（1024 サンプルずつ）でリバーブの音がブチブチ途切れない（利用者の手元）

    前は塊ごとに、前の音（リバーブなら 1.5 秒）を素材から読み直していた AAC のような
    圧縮した音は読み直すたびに頭から解き直すので、1 塊 19ms（予算 21ms）掛かって再生が
    間に合わず、読み直した所の音も塊ごとにわずかにずれて境目に段差が出た
    """

    @pytest.fixture
    def voice(self, media_dir: Path) -> Path:
        from tests.media_fixtures import make_sample

        return make_sample(
            media_dir, "reverb-voice.mp4", duration=5.0, sample_rate=48000, gain_db=8.0
        ).path

    @pytest.mark.parametrize(
        ("kind", "values"),
        [("audio_reverb", {}), ("audio_delay", {}), ("audio_pitch", {"semitones": 3.0})],
    )
    def test_small_blocks_match_one_long_render(
        self, voice: Path, kind: str, values: dict[str, float]
    ) -> None:
        project = _project(voice, kind, **values)
        whole_mixer = AudioMixer(project)
        whole = whole_mixer.render(RATE, RATE * 2)[:, 0]
        whole_mixer.close()
        block_mixer = AudioMixer(project)
        blocks = np.concatenate(
            [block_mixer.render(RATE + start, 1024) for start in range(0, RATE * 2, 1024)]
        )[: RATE * 2, 0]
        block_mixer.close()
        assert np.abs(blocks - whole).max() < 1e-4

    def test_playing_on_reads_the_source_only_forward(
        self, voice: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 塊ごとに前へ戻って読み直すと、圧縮した音は毎回解き直しになり再生が間に合わない
        from sashimono.engine.decode import AudioDecoder

        reads: list[tuple[int, int]] = []
        original = AudioDecoder.read

        def watching(self: AudioDecoder, start: int, count: int) -> np.ndarray:
            reads.append((start, count))
            return original(self, start, count)

        monkeypatch.setattr(AudioDecoder, "read", watching)
        mixer = AudioMixer(_project(voice, "audio_reverb"))
        for start in range(RATE, RATE * 3, 1024):
            mixer.render(start, 1024)
        mixer.close()
        # 始めの 1 回は前の音ごと読む その後は前に読んだ所の続きだけを読む
        for (before, length), (after, _) in itertools.pairwise(reads):
            assert after == before + length, f"{before + length} の続きではなく {after} を読んだ"


class TestTheCacheIsLetGo:
    """前の音を読むエフェクトの貯めを、使わなくなったら捨てる（PR #231 の指摘）

    捨てないと、消したクリップや値を変えたクリップの分（1 本 4 MB ほど）が、ミキサが
    生きている間ずっと残り、長い編集で増え続ける
    """

    def test_a_removed_clip_is_forgotten(self, tmp_path: Path) -> None:
        project = _project(_click(tmp_path), "audio_reverb")
        mixer = AudioMixer(project)
        try:
            mixer.render(0, RATE // 2)
            assert mixer._windows and mixer._raw
            # 同じクリップのままなら残す（関係の無い編集で作り直さない）
            mixer.set_project(project)
            assert mixer._windows and mixer._raw
            empty = project.with_timeline(replace(project.timeline, tracks=()))
            mixer.set_project(empty)
            assert not mixer._windows and not mixer._raw
        finally:
            mixer.close()

    def test_closing_lets_it_all_go(self, tmp_path: Path) -> None:
        mixer = AudioMixer(_project(_click(tmp_path), "audio_reverb"))
        mixer.render(0, RATE // 2)
        mixer.close()
        assert not mixer._windows and not mixer._raw
