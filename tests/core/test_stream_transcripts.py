"""字幕を素材と音声ストリームごとに持つ（利用者の要望）

前は素材に 1 つだけで、同じ動画の音声 2 を起こすと音声 1 の字幕が置き換わり、分けて
置いた音声 1〜4 のクリップすべてに同じ字幕が出た 保存の形式は 8 に上げ、7 までの
ファイルの字幕は 1 本目の音声の物として読む
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from sashimono.core.commands import (
    AddScene,
    SetSegmentText,
    SetTranscript,
    burn_subtitles,
    new_scene,
)
from sashimono.core.io.serialize import FORMAT_VERSION, project_from_dict, project_to_dict
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    GeneratedSource,
    MediaItem,
    Project,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    VideoStreamInfo,
)
from sashimono.core.projection import project_timeline
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT


def _movie(voices: int = 2, name: str = "録画") -> MediaItem:
    return MediaItem(
        path=Path(f"C:/素材/{name}.mp4"),
        duration=Fraction(10),
        video_streams=(VideoStreamInfo(0, 1920, 1080, FrameRate(30), Fraction(1, 15360), "h264"),),
        audio_streams=tuple(
            AudioStreamInfo(index, 48000, 2, Fraction(1, 48000), "aac")
            for index in range(1, voices + 1)
        ),
    )


def _said(text: str) -> Transcript:
    return Transcript((TranscriptSegment(Fraction(1), Fraction(2), text),))


def _texts(transcript: Transcript | None) -> list[str]:
    return [] if transcript is None else [s.text for s in transcript.segments]


class TestKeeping:
    def test_the_second_voice_does_not_replace_the_first(self) -> None:
        # 壊れると、マイクの声（音声 2）を起こした途端にゲームの音（音声 1）の字幕が消える
        media = _movie()
        project = Project.create(media=(media,))
        project = SetTranscript(media.id, _said("ゲーム"), stream=1).apply(project)
        project = SetTranscript(media.id, _said("声"), stream=2).apply(project)
        kept = project.require_media(media.id)
        assert _texts(kept.transcript_for(1)) == ["ゲーム"]
        assert _texts(kept.transcript_for(2)) == ["声"]
        # 番号を省くと 1 本目 前と同じ呼び方がそのまま使える
        assert _texts(kept.transcript) == ["ゲーム"]

    def test_other_media_are_left_alone(self) -> None:
        # 壊れると、素材をまたいで字幕が入れ替わる（2 本目を起こすと 1 本目の字幕になる）
        first, second = _movie(name="一"), _movie(name="二")
        project = Project.create(media=(first, second))
        project = SetTranscript(first.id, _said("一の字幕")).apply(project)
        project = SetTranscript(second.id, _said("二の字幕")).apply(project)
        assert _texts(project.require_media(first.id).transcript) == ["一の字幕"]
        assert _texts(project.require_media(second.id).transcript) == ["二の字幕"]

    def test_editing_one_voice_leaves_the_other(self) -> None:
        # 壊れると、音声 2 の字幕を直しただけで音声 1 の字幕が消える
        media = _movie()
        project = Project.create(media=(media,))
        project = SetTranscript(media.id, _said("ゲーム"), stream=1).apply(project)
        project = SetTranscript(media.id, _said("声"), stream=2).apply(project)
        segment = project.require_media(media.id).transcript_for(2)
        assert segment is not None
        project = SetSegmentText(media.id, segment.segments[0].id, "直した声", stream=2).apply(
            project
        )
        kept = project.require_media(media.id)
        assert _texts(kept.transcript_for(1)) == ["ゲーム"]
        assert _texts(kept.transcript_for(2)) == ["直した声"]


class TestSaving:
    def test_both_voices_survive_saving(self) -> None:
        # 壊れると、保存して開き直したら音声 2 の字幕が消えている
        media = _movie().with_transcript(_said("ゲーム"), 1).with_transcript(_said("声"), 2)
        loaded = project_from_dict(project_to_dict(Project.create(media=(media,))))
        again = loaded.require_media(media.id)
        assert _texts(again.transcript_for(1)) == ["ゲーム"]
        assert _texts(again.transcript_for(2)) == ["声"]
        # 8 で音声ごとにした 次に版を上げても、この試験は落ちないように下限で見る
        assert FORMAT_VERSION >= 8

    def test_an_older_file_reads_its_subtitles_as_the_first_voice(self) -> None:
        # 7 までのファイルは素材に 1 つの ``transcript`` を持つ 捨てると字幕が消える
        media = _movie().with_transcript(_said("前の版"), 1)
        data = project_to_dict(Project.create(media=(media,)))
        data["version"] = 7
        entry = data["media"][0]
        entry["transcript"] = entry.pop("transcripts")[0]["transcript"]
        loaded = project_from_dict(data).require_media(media.id)
        assert _texts(loaded.transcript_for(1)) == ["前の版"]
        assert loaded.transcript_for(2) is None


class TestShowing:
    def test_each_voice_clip_shows_its_own_subtitles(self) -> None:
        # 分けて置いた音声 1 と 2 のクリップに、それぞれの音の字幕だけが出る
        media = _movie().with_transcript(_said("ゲーム"), 1).with_transcript(_said("声"), 2)
        voice1 = Clip(timeline_start=0, duration=150, media_id=media.id, stream_index=1)
        voice2 = Clip(timeline_start=0, duration=150, media_id=media.id, stream_index=2)
        tracks = (
            Track(TrackKind.AUDIO, "A1", (voice1,)),
            Track(TrackKind.AUDIO, "A2", (voice2,)),
        )
        base = Project.create(media=(media,))
        project = base.with_timeline(replace(base.timeline, tracks=tracks))
        shown = {(p.clip_id, p.segment.text) for p in project_timeline(project)}
        assert shown == {(voice1.id, "ゲーム"), (voice2.id, "声")}

    def test_voices_inside_a_placed_scene_burn_apart(self) -> None:
        # 壊れると、シーンの中で同時に話す音声 1 と 2 が 1 本のレイヤーにまとまり、前の字幕が
        # 切り詰められて欠ける（シーンのクリップは素材を持たないので話し手が分からなかった
        # PR #231 の指摘）
        media = _movie().with_transcript(_said("ゲーム"), 1).with_transcript(_said("声"), 2)
        base = Project.create(media=(media,))
        scene = new_scene(base, "中")
        inner = (
            Track(
                TrackKind.AUDIO,
                "A1",
                (Clip(timeline_start=0, duration=150, media_id=media.id, stream_index=1),),
            ),
            Track(
                TrackKind.AUDIO,
                "A2",
                (Clip(timeline_start=0, duration=150, media_id=media.id, stream_index=2),),
            ),
        )
        project = AddScene(replace(scene, timeline=replace(scene.timeline, tracks=inner))).apply(
            base
        )
        placed = Track(
            TrackKind.VIDEO, "V1", (Clip(timeline_start=0, duration=150, scene_id=scene.id),)
        )
        project = project.with_timeline(replace(project.timeline, tracks=(placed,)))
        commands = burn_subtitles(project, GeneratedSource(TEXT.kind, {"text": ""}))
        for command in commands:
            project = command.apply(project)
        burned = {
            (str(c.source.params["text"]), c.subtitle_origin)
            for t in project.timeline.tracks
            for c in t.clips
            if c.source is not None and c.subtitle_origin is not None
        }
        texts = {text for text, _ in burned}
        assert texts == {"ゲーム", "声"}
        origins = {(o.media_id, o.stream) for _, o in burned if o is not None}
        assert origins == {(media.id, 1), (media.id, 2)}
        # 話し手ごとに別のトラック
        assert len([t for t in project.timeline.tracks if t.name.startswith("字幕")]) == 2
