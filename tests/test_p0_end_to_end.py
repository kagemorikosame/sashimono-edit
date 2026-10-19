"""P0 の完了条件そのもの

GUI を一切使わずに「素材追加 → クリップ配置 → 保存 → 読込 → Undo」が通ること
コア層が GUI に依存していないことの証明でもあり、同じ経路を AI エージェントも使う
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sashimono.core.commands import (
    AddClip,
    AddMedia,
    AddTrack,
    Document,
    RemoveClip,
    SetTranscript,
    SplitClip,
)
from sashimono.core.io import load_project, save_project
from sashimono.core.model import (
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.core.projection import project_timeline
from sashimono.core.timebase import FrameRate, format_timecode
from tests.conftest import make_clip


def test_edit_save_reload_and_undo(
    video_media: MediaItem, audio_media: MediaItem, transcript: Transcript, tmp_path: Path
) -> None:
    # 冒頭を切って詰めても字幕が話す所に付いて来ないか、保存した作品を開き直すと形が変わるか、
    # 取り消しが 1 手ずつ戻らないと、コアだけで組む編集（AI の道具も同じ経路）が崩れる
    document = Document(Project.create(ProjectSettings(frame_rate=FrameRate(30))))

    # 素材を登録する
    document.execute(AddMedia(video_media))
    document.execute(AddMedia(audio_media))
    document.execute(SetTranscript(video_media.id, transcript))
    assert len(document.project.media) == 2

    # トラックを用意してクリップを置く
    video_track = Track(kind=TrackKind.VIDEO, name="V1")
    audio_track = Track(kind=TrackKind.AUDIO, name="A1")
    document.execute(AddTrack(video_track))
    document.execute(AddTrack(audio_track))
    document.execute(AddClip(video_track.id, make_clip(0, 300, video_media)))
    document.execute(AddClip(audio_track.id, make_clip(0, 900, audio_media)))

    assert document.project.duration == 900
    assert format_timecode(document.project.duration, document.project.rate) == "00:00:30:00"

    # 字幕が投影されている
    assert [p.segment.text for p in project_timeline(document.project)] == [
        "今日は",
        "編集ソフトを",
        "作ります",
    ]

    # カット編集 冒頭 2 秒を切り落として詰める
    placed_clip = document.project.timeline.tracks[0].clips[0]
    document.execute(SplitClip(placed_clip.id, 60))
    head = document.project.timeline.tracks[0].clips[0]
    document.execute(RemoveClip(head.id, ripple=True))

    after_cut = [(p.segment.text, p.start_frame) for p in project_timeline(document.project)]
    assert after_cut == [("今日は", 0), ("編集ソフトを", 60), ("作ります", 150)]

    # 保存して読み直す
    path = tmp_path / "配信回.sme"
    save_project(document.project, path)
    reloaded = load_project(path)
    assert reloaded == replace(document.project, name=reloaded.name)

    # 読み直したプロジェクトでも字幕の位置は変わらない
    assert [(p.segment.text, p.start_frame) for p in project_timeline(reloaded)] == after_cut

    # Undo で 1 手ずつ戻る
    document.undo()  # リップル削除を取り消す
    assert len(document.project.timeline.tracks[0].clips) == 2
    document.undo()  # 分割を取り消す
    assert len(document.project.timeline.tracks[0].clips) == 1
    assert [p.start_frame for p in project_timeline(document.project)] == [30, 120, 210]


def test_ai_style_batch_edit_is_one_undo(
    video_media: MediaItem, transcript: Transcript, tmp_path: Path
) -> None:
    """AI が 1 つの指示で行った複数の変更が、1 回の Undo で戻ること"""
    document = Document(Project.create(ProjectSettings(frame_rate=FrameRate(30))))
    document.execute(AddMedia(video_media))
    document.execute(SetTranscript(video_media.id, transcript))
    track = Track(kind=TrackKind.VIDEO, name="V1")
    document.execute(AddTrack(track))
    document.execute(AddClip(track.id, make_clip(0, 300, video_media)))

    before = document.project
    steps_before = len(document.history_labels)

    # 「無音を詰めて」に相当する一連の操作
    with document.checkpoint("AI: 無音をカット"):
        for cut_at in (240, 180, 90):
            clip = document.project.timeline.tracks[0].clip_at(cut_at)
            assert clip is not None
            document.execute(SplitClip(clip.id, cut_at))

    assert len(document.project.timeline.tracks[0].clips) == 4
    assert len(document.history_labels) == steps_before + 1
    assert document.undo_label == "AI: 無音をカット"

    document.undo()
    assert document.project == before
