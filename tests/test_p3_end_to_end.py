"""P3 の完了条件

「起こし → 整形 → ジェットカット → 保存・読み直し → 書き出し」が一本通り、
その全部を通して**字幕が素材の同じ場所を指したまま**であること

追従の確認は、投影された字幕の開始フレームを素材のソース秒へ戻して、元の
セグメントの開始時刻と一致するかで見る 位置がずれていればここで必ず落ちる
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.asr import CleanupOptions, clean_transcript
from sashimono.core.commands import RippleCut, SetClipProperty, SetTranscript, SplitClip
from sashimono.core.io import load_project, save_subtitles
from sashimono.core.jetcut import plan_cuts
from sashimono.core.model import (
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Transcript,
    TranscriptSegment,
)
from sashimono.core.projection import project_timeline
from sashimono.core.timebase import FrameRate
from sashimono.engine.audio.silence import SilenceOptions, detect_silence, keep_speech
from sashimono.engine.audio.waveform import analyze_waveform
from sashimono.ui.main_window import MainWindow
from tests.media_fixtures import make_silent_gap

RATE = FrameRate(30)

#: 認識器が返したことにする起こし結果 素材は 1..2 秒が無音なので、その前後に
#: 発話があるという想定にする フィラー語をわざと混ぜてある
FAKE_RESULT = Transcript(
    segments=(
        TranscriptSegment(Fraction(1, 10), Fraction(9, 10), "えーと、前半です"),
        TranscriptSegment(Fraction(21, 10), Fraction(29, 10), "あのー、後半です"),
    ),
    language="ja",
    model="fake",
)


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=RATE)),
        confirm_unsaved=False,
    )
    yield created
    created.close()


@pytest.fixture(scope="session")
def speech(media_dir: Path) -> Path:
    """3 秒の音声 1..2 秒が無音"""
    return make_silent_gap(media_dir, "p3_speech.wav", duration=3.0)


def source_time_at(project: Project, media: MediaItem, frame: int) -> Fraction | None:
    """タイムラインのフレームで鳴っている、素材内の位置"""
    for track in project.timeline.tracks:
        for clip in track.clips:
            if clip.media_id != media.id or not clip.contains(frame):
                continue
            elapsed = (frame - clip.timeline_start) * project.rate.frame_duration
            return clip.source_in + elapsed * clip.speed
    return None


def assert_subtitles_point_at_the_same_audio(project: Project, media: MediaItem) -> None:
    """字幕の開始位置で鳴っている音が、その字幕の元の位置と一致すること

    ずれの許容は 1 フレーム 投影は「その秒を含むフレーム」へ落とすので、
    切り捨てた分だけは必ず手前へ寄る
    """
    tolerance = project.rate.frame_duration
    projected = list(project_timeline(project))
    assert projected, "字幕が 1 枚も出ていない"

    for subtitle in projected:
        if subtitle.clipped_head:
            continue
        actual = source_time_at(project, media, subtitle.start_frame)
        assert actual is not None
        assert abs(actual - subtitle.segment.start) <= tolerance, (
            f"{subtitle.segment.text!r} が指す位置がずれた: "
            f"{float(actual):.3f} 秒 ≠ {float(subtitle.segment.start):.3f} 秒"
        )


class TestSubtitleFlow:
    def test_the_whole_flow_keeps_subtitles_on_their_audio(
        self, window: MainWindow, speech: Path, tmp_path: Path
    ) -> None:
        # --- 読み込み ---
        window.import_media([speech])
        assert window.wait_for_imports()
        project = window._document.project
        media = project.media[0]
        assert media.has_audio

        # --- 起こし（結果だけを差し込む 認識器そのものは tests/asr で見る）---
        window.execute(SetTranscript(media.id, FAKE_RESULT))
        assert_subtitles_point_at_the_same_audio(window._document.project, media)

        # --- 整形 ---
        transcript = window._document.project.require_media(media.id).transcript
        assert transcript is not None
        cleaned = clean_transcript(transcript, CleanupOptions(max_line_chars=0))
        window.execute(SetTranscript(media.id, cleaned))

        texts = [s.text for s in cleaned.segments]
        assert texts == ["前半です", "後半です"]
        assert_subtitles_point_at_the_same_audio(window._document.project, media)

        # --- ジェットカット ---
        waveform = analyze_waveform(media.path, sample_rate=48000, channels=2)
        assert waveform is not None
        silences = detect_silence(
            waveform, SilenceOptions(min_silence=Fraction(1, 4), padding=Fraction(1, 20))
        )
        silences = keep_speech(silences, cleaned)
        ranges = plan_cuts(window._document.project, media.id, silences)
        assert ranges, "無音が見つからなかった"

        before = window._document.project.duration
        window.execute(RippleCut(ranges))
        after = window._document.project
        assert after.duration < before
        assert_subtitles_point_at_the_same_audio(after, media)

        # --- 分割 ---
        video_clip = _first_clip(after)
        window.execute(SplitClip(video_clip.id, video_clip.timeline_start + 10))
        assert_subtitles_point_at_the_same_audio(window._document.project, media)

        # --- 速度変更 ---
        target = _first_clip(window._document.project)
        window.execute(SetClipProperty(target.id, "speed", Fraction(2)))
        assert_subtitles_point_at_the_same_audio(window._document.project, media)

        # --- 保存して読み直す ---
        saved = tmp_path / "字幕.sme"
        from sashimono.core.io import save_project

        save_project(window._document.project, saved)
        reopened = load_project(saved)
        restored = reopened.require_media(media.id).transcript
        assert restored is not None
        assert [s.text for s in restored.segments] == texts
        assert_subtitles_point_at_the_same_audio(reopened, reopened.media[0])

        # --- 字幕ファイルの書き出し ---
        written = save_subtitles(window._document.project, tmp_path / "字幕.srt")
        body = written.read_text(encoding="utf-8")
        assert "前半です" in body
        assert body.count("-->") == len(list(project_timeline(window._document.project)))

    def test_undo_returns_the_cut_and_the_subtitles_together(
        self, window: MainWindow, speech: Path
    ) -> None:
        # カットを取り消しても字幕の表示位置が戻らないと、
        # 取り消した後の字幕が話している所からずれて出る
        window.import_media([speech])
        assert window.wait_for_imports()
        media = window._document.project.media[0]
        window.execute(SetTranscript(media.id, FAKE_RESULT))

        def placement() -> list[tuple[str, int]]:
            return [
                (p.segment.text, p.start_frame) for p in project_timeline(window._document.project)
            ]

        before = placement()
        window.execute(RippleCut(((0, 15),)))
        assert placement() != before

        # 字幕は素材に付いているので、カットを戻せば表示位置も一緒に戻る
        window.undo()
        assert placement() == before


def _first_clip(project: Project) -> Clip:
    track = next(iter(project.timeline.video_tracks()), None)
    if track is None or not track.clips:
        track = next(iter(project.timeline.audio_tracks()))
    return track.clips[0]
