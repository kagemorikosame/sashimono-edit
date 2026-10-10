"""保存する側の項目の一覧と、型の項目がそろっているか（#285）

クリップ・トラック・設定などの型の項目は、``core/io/serialize.py`` の ``clip_to_json`` などに
手で書き並べている 型に項目を足して保存側へ書き忘れると、プロジェクト・自動退避・自作の
エイリアスで、その項目が黙って落ちる 開き直すまで誰も気付かない

ここでは型ごとに次を見る

- 型の項目はどれも書く（書かないと決めた物は :data:`NOT_WRITTEN` に理由と一緒に書く）
- 書いた鍵はどれも読む（書いたのに読まなければ、開き直したときに既定へ戻る）
- 全項目に既定と違う値を入れた物を、書いて読むと同じになる
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sashimono.core.io.serialize import project_from_dict, project_to_dict
from sashimono.core.model import (
    AnimatedValue,
    AudioStreamInfo,
    Blending,
    Clip,
    Effect,
    GeneratedSource,
    GroupId,
    Interpolation,
    Keyframe,
    LayerMode,
    Marker,
    MediaItem,
    Project,
    ProjectSettings,
    Scene,
    SubtitleOrigin,
    Timeline,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    VideoStreamInfo,
    Word,
)
from sashimono.core.timebase import FrameRate

#: 型の項目のうち、わざと保存に書かない物と、その理由
#: 一覧に無いのに書いていない項目があれば落ちる 足すときは、書かないと何が起きるかを添える
NOT_WRITTEN: dict[type, dict[str, str]] = {}

#: 型の項目ではないが書く鍵（ファイルの形式の名前と版）
EXTRA_KEYS: dict[type, frozenset[str]] = {Project: frozenset({"format", "version"})}

RATE = FrameRate(24)


class _Reading(dict[str, Any]):
    """読む側が引いた鍵を覚える辞書

    読む側は ``isinstance(raw, dict)`` で確かめるので、dict を継いで中身はそのまま渡す
    """

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__({key: _watch(value) for key, value in data.items()})
        self.seen: set[str] = set()

    def get(self, key: str, default: Any = None) -> Any:
        self.seen.add(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self.seen.add(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            self.seen.add(key)
        return super().__contains__(key)


def _watch(value: Any) -> Any:
    if isinstance(value, dict):
        return _Reading(value)
    if isinstance(value, list):
        return [_watch(item) for item in value]
    return value


def _at(tree: Any, *path: str | int) -> _Reading:
    """``tree`` の中の辞書を、読んだ印を付けずにたどる"""
    for step in path:
        tree = dict.__getitem__(tree, step) if isinstance(step, str) else tree[step]
    assert isinstance(tree, _Reading), path
    return tree


def _full_project() -> Project:
    """全部の型の全部の項目に、既定と違う値を入れた作品

    クリップの項目は 1 本では埋まらない（シーンを置いたクリップは素材も中身も持てない）
    素材と中身を持つ物と、シーンを置いた物の 2 本で埋める
    """
    curve = Keyframe(0, 0.25, Interpolation.EASE_IN, curve="cubic")
    bezier = Keyframe(12, 0.75, Interpolation.BEZIER, control_points=(0.1, 0.2, 0.3, 0.4))
    word = Word(Fraction(1, 2), Fraction(3, 2), "言葉")
    segment = TranscriptSegment(
        Fraction(1, 2), Fraction(2), "字幕の行", words=(word,), speaker="話し手", edited=True
    )
    media = MediaItem(
        path=Path("C:/素材/本編.mp4"),
        duration=Fraction(10),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1920,
                height=1080,
                frame_rate=RATE,
                time_base=Fraction(1, 12288),
                codec="hevc",
                pixel_format="yuv420p10le",
                rotation=90,
                end_time=Fraction(19, 2),
                color_transfer="smpte2084",
                color_primaries="bt2020",
            ),
        ),
        audio_streams=(
            AudioStreamInfo(
                index=1,
                sample_rate=44100,
                channels=1,
                time_base=Fraction(1, 44100),
                codec="aac",
                language="jpn",
            ),
        ),
        transcripts=((1, Transcript((segment,), language="ja", model="small")),),
        display_name="本編の名前",
    )
    scene = Scene("場面", Timeline(rate=RATE))
    filled = Clip(
        timeline_start=5,
        duration=40,
        media_id=media.id,
        source=GeneratedSource("text", {"text": "見出し", "size": AnimatedValue(48.0)}),
        source_in=Fraction(1, 3),
        stream_index=1,
        speed=Fraction(3, 2),
        hold_at=Fraction(2),
        effects=(Effect("blur", {"radius": AnimatedValue(4.0)}, enabled=False, fixed=True),),
        audio_stream=1,
        show_picture=False,
        after_effects=(Effect("blur", {"radius": AnimatedValue(2.0)}),),
        opacity=AnimatedValue(0.5, (curve, bezier)),
        blend_mode="add",
        clip_to_below=True,
        link_group=GroupId("リンク"),
        group_id=GroupId("グループ"),
        enabled=False,
        native_size=True,
        subtitle_origin=SubtitleOrigin(media.id, 1, segment.id),
    )
    nested = Clip(timeline_start=60, duration=10, scene_id=scene.id)
    track = Track(
        TrackKind.VIDEO,
        "V9",
        (filled, nested),
        effects=(Effect("blur"),),
        locked=True,
        muted=True,
        solo=True,
        height=80,
        volume_db=-6.0,
        pan=0.5,
    )
    timeline = Timeline(
        rate=RATE,
        tracks=(track,),
        markers=(Marker(3, "印", "#00ff00"),),
        work_area=(0, 60),
    )
    settings = ProjectSettings(
        width=1280,
        height=720,
        frame_rate=RATE,
        sample_rate=44100,
        channels=1,
        color_space="rec2020",
        blending=Blending.LINEAR,
        layer_mode=LayerMode.MIXED,
    )
    return Project(settings, timeline, media=(media,), name="試し", scenes=(scene,))


def _instances(project: Project) -> dict[type, list[Any]]:
    """型ごとの見本（:func:`_full_project` の中の物）"""
    track = project.timeline.tracks[0]
    filled = track.clips[0]
    media = project.media[0]
    transcript = media.transcripts[0][1]
    origin = filled.subtitle_origin
    assert filled.source is not None and origin is not None
    return {
        Project: [project],
        ProjectSettings: [project.settings],
        Timeline: [project.timeline],
        Marker: list(project.timeline.markers),
        Track: [track],
        Clip: list(track.clips),
        Effect: [*filled.effects, *filled.after_effects, *track.effects],
        GeneratedSource: [filled.source],
        AnimatedValue: [filled.opacity],
        Keyframe: list(filled.opacity.keyframes),
        SubtitleOrigin: [origin],
        Scene: list(project.scenes),
        MediaItem: [media],
        VideoStreamInfo: list(media.video_streams),
        AudioStreamInfo: list(media.audio_streams),
        Transcript: [transcript],
        TranscriptSegment: list(transcript.segments),
        Word: list(transcript.segments[0].words),
    }


def _written(tree: _Reading) -> dict[type, list[_Reading]]:
    """型ごとの書いた辞書（:func:`_instances` と同じ物を指す）"""
    timeline = _at(tree, "timeline")
    track = _at(timeline, "tracks", 0)
    filled = _at(track, "clips", 0)
    media = _at(tree, "media", 0)
    transcript = _at(media, "transcripts", 0, "transcript")
    segment = _at(transcript, "segments", 0)
    return {
        Project: [tree],
        ProjectSettings: [_at(tree, "settings")],
        Timeline: [timeline],
        Marker: [_at(timeline, "markers", 0)],
        Track: [track],
        Clip: [filled, _at(track, "clips", 1)],
        Effect: [
            _at(filled, "effects", 0),
            _at(filled, "after_effects", 0),
            _at(track, "effects", 0),
        ],
        GeneratedSource: [_at(filled, "source")],
        AnimatedValue: [_at(filled, "opacity")],
        Keyframe: [_at(filled, "opacity", "keyframes", 0), _at(filled, "opacity", "keyframes", 1)],
        SubtitleOrigin: [_at(filled, "subtitle_origin")],
        Scene: [_at(tree, "scenes", 0)],
        MediaItem: [media],
        VideoStreamInfo: [_at(media, "video_streams", 0)],
        AudioStreamInfo: [_at(media, "audio_streams", 0)],
        Transcript: [transcript],
        TranscriptSegment: [segment],
        Word: [_at(segment, "words", 0)],
    }


def _read_tree() -> _Reading:
    """書いた物を、読んだ鍵を覚える辞書にして読ませた後の木"""
    tree = _watch(project_to_dict(_full_project()))
    assert isinstance(tree, _Reading)
    project_from_dict(tree)
    return tree


TYPES = list(_instances(_full_project()))


@pytest.mark.parametrize("kind", TYPES, ids=lambda kind: kind.__name__)
class TestEveryFieldIsSaved:
    def test_every_field_is_written(self, kind: type) -> None:
        # 型に足した項目を保存側へ書き忘れると、保存して開き直すだけで黙って既定へ戻る
        fields = {field.name for field in dataclasses.fields(kind)}
        skipped = NOT_WRITTEN.get(kind, {})
        assert set(skipped) <= fields, f"{kind.__name__} に無い項目を書かない一覧に挙げている"
        written = set().union(*(set(entry) for entry in _written(_read_tree())[kind]))
        missing = fields - set(skipped) - written
        assert not missing, f"{kind.__name__} の {sorted(missing)} を保存に書いていない"
        unknown = written - fields - EXTRA_KEYS.get(kind, frozenset())
        assert not unknown, f"{kind.__name__} に無い鍵 {sorted(unknown)} を書いている"
        # 書かないと決めた物を書いていたら、一覧が古い
        assert not set(skipped) & written

    def test_every_written_key_is_read(self, kind: type) -> None:
        # 書いても読まなければ、開き直したときに既定へ戻る（書き忘れと同じ結果）
        for entry in _written(_read_tree())[kind]:
            unread = set(dict.keys(entry)) - entry.seen
            assert not unread, f"{kind.__name__} の {sorted(unread)} を読んでいない"

    def test_the_sample_fills_every_field(self, kind: type) -> None:
        # 既定のままの項目は、書き忘れても読み忘れても往復で同じになり、試験をすり抜ける
        # 型に項目を足したら :func:`_full_project` にも既定と違う値を入れる
        samples = _instances(_full_project())[kind]
        for field in dataclasses.fields(kind):
            if field.default is not dataclasses.MISSING:
                default: Any = field.default
            elif field.default_factory is not dataclasses.MISSING:
                default = field.default_factory()
            else:
                continue
            assert any(getattr(sample, field.name) != default for sample in samples), (
                f"{kind.__name__}.{field.name} が見本で既定のまま"
            )


def test_a_full_project_comes_back_the_same() -> None:
    # 全項目に既定と違う値を入れた物を書いて読む 1 つでも落ちれば既定に戻って食い違う
    project = _full_project()
    assert project_from_dict(project_to_dict(project)) == project
