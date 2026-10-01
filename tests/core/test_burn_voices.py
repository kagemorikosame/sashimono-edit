"""字幕の焼き込みを話し手（素材と音声）ごとに分け、選んだ字幕とひな形で置く（利用者の要望）

前は全部の字幕を 1 本のレイヤーへ並べ、重なる行は前を切り詰めた 字幕を音声ごとに
持つようになると、音声 1 と 2 が同時に話している所で前の字幕が欠けた 見た目も既定の
大きさ・位置で決め打ちだった
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from sashimono.core.commands import Command, burn_subtitles
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    MediaItem,
    Project,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import TEXT


def _said(*lines: tuple[int, int, str]) -> Transcript:
    return Transcript(
        tuple(TranscriptSegment(Fraction(a), Fraction(b), text) for a, b, text in lines)
    )


def _project() -> tuple[Project, MediaItem]:
    """ゲームの音（音声 1）とマイクの声（音声 2）が 1〜3 秒で重なって話す録画"""
    media = MediaItem(
        path=Path("C:/素材/録画.mp4"),
        duration=Fraction(10),
        video_streams=(VideoStreamInfo(0, 1920, 1080, FrameRate(30), Fraction(1, 15360), "h264"),),
        audio_streams=(
            AudioStreamInfo(1, 48000, 2, Fraction(1, 48000), "aac"),
            AudioStreamInfo(2, 48000, 2, Fraction(1, 48000), "aac"),
        ),
    )
    media = media.with_transcript(_said((1, 3, "ゲームの声"), (5, 6, "二つ目")), 1)
    media = media.with_transcript(_said((1, 3, "マイクの声")), 2)
    voice1 = Clip(timeline_start=0, duration=300, media_id=media.id, stream_index=1)
    voice2 = Clip(timeline_start=0, duration=300, media_id=media.id, stream_index=2)
    base = Project.create(media=(media,))
    tracks = (
        Track(TrackKind.AUDIO, "A1", (voice1,)),
        Track(TrackKind.AUDIO, "A2", (voice2,)),
    )
    return base.with_timeline(replace(base.timeline, tracks=tracks)), media


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _texts(project: Project, name: str) -> list[str]:
    track = next(t for t in project.timeline.tracks if t.name == name)
    return [str(c.source.params["text"]) for c in track.clips if c.source is not None]


class TestEachVoiceGetsItsOwnTrack:
    def test_overlapping_voices_are_both_kept(self) -> None:
        # 壊れると、同じ 1〜3 秒に話す 2 人のうち前の字幕が切り詰められて消える
        project, _media = _project()
        burned = _apply(project, burn_subtitles(project, TEXT.create()))
        assert _texts(burned, "字幕 音声 1") == ["ゲームの声", "二つ目"]
        assert _texts(burned, "字幕 音声 2") == ["マイクの声"]

    def test_the_voices_can_be_chosen(self) -> None:
        project, media = _project()
        commands = burn_subtitles(project, TEXT.create(), voices=[(media.id, 2)])
        burned = _apply(project, commands)
        # 1 人だけなら名前に音声を添えない（今までの「字幕」）
        assert _texts(burned, "字幕") == ["マイクの声"]

    def test_only_the_chosen_rows(self) -> None:
        project, media = _project()
        transcript = media.transcript_for(1)
        assert transcript is not None
        wanted = {transcript.segments[1].id}
        burned = _apply(project, burn_subtitles(project, TEXT.create(), segments=wanted))
        assert _texts(burned, "字幕") == ["二つ目"]


class TestTheLookIsCopied:
    def test_a_selected_text_is_the_template(self) -> None:
        # タイムラインで選んだテキストの見た目（大きさ・色・エフェクト）を写し、本文だけ変える
        project, _media = _project()
        styled = with_fixed_items(
            Clip(
                timeline_start=0,
                duration=30,
                source=TEXT.create(text="見本", size=90.0, color=(1.0, 0.8, 0.0, 1.0)),
                effects=(registry.require("shadow").create(),),
            ),
            picture=True,
        )
        burned = _apply(project, burn_subtitles(project, styled, voices=None))
        placed = [c for t in burned.timeline.tracks if t.name.startswith("字幕") for c in t.clips]
        assert placed and all(c.source is not None for c in placed)
        first = placed[0]
        assert first.source is not None
        size = first.source.params["size"]
        assert getattr(size, "static", None) == 90.0
        assert first.source.params["color"] == (1.0, 0.8, 0.0, 1.0)
        assert [e.kind for e in first.effects] == [e.kind for e in styled.effects]
        # ID は振り直す 同じ ID が並ぶとプロジェクトの検査に断られる
        ids = [c.id for c in placed] + [e.id for c in placed for e in c.effects]
        assert len(ids) == len(set(ids)) and styled.id not in ids

    def test_one_undo_takes_everything_back(self) -> None:
        # 置いたトラックとクリップは 1 つの命令の並び まとめて出せば 1 回で戻る
        from sashimono.core.commands import Document

        project, _media = _project()
        document = Document(project)
        with document.checkpoint("字幕を焼き込み"):
            for command in burn_subtitles(project, TEXT.create()):
                document.execute(command)
        document.undo()
        assert document.project == project
