"""プロジェクトのデータモデル

すべて frozen dataclass で、変更は :func:`dataclasses.replace` による差し替えで表現する
木の一部だけを作り直し、残りは共有される（構造共有）ので、Undo は「古いルートを持っておく」
だけで済む 逆操作コマンドを一つずつ書く方式に比べて、取り消しの取りこぼしが原理的に起きない

この方針の代償は、1 クリップの変更でそのトラックのクリップ列を作り直す点だが、
トラックあたり数千クリップの規模では問題にならない
"""

from sashimono.core.model.effect import (
    AnimatedValue,
    Effect,
    Interpolation,
    Keyframe,
    ParamValue,
)
from sashimono.core.model.ids import (
    ClipId,
    EffectId,
    GroupId,
    MediaId,
    SceneId,
    SegmentId,
    TrackId,
    new_clip_id,
    new_effect_id,
    new_group_id,
    new_media_id,
    new_scene_id,
    new_segment_id,
    new_track_id,
)
from sashimono.core.model.media import AudioStreamInfo, MediaItem, VideoStreamInfo
from sashimono.core.model.project import Blending, LayerMode, Project, ProjectSettings, Scene
from sashimono.core.model.timeline import (
    FILTER_KIND,
    GROUP_AS_ONE,
    GROUP_KIND,
    GROUP_LAYERS,
    Clip,
    GeneratedSource,
    Marker,
    SubtitleOrigin,
    Timeline,
    Track,
    TrackKind,
    controlling_groups,
    default_track_name,
    draws_picture,
    group_as_one,
    group_layers,
    group_reaches,
    heard_stream,
    plays_sound,
)
from sashimono.core.model.transcript import Transcript, TranscriptSegment, Word

__all__ = [
    "FILTER_KIND",
    "GROUP_AS_ONE",
    "GROUP_KIND",
    "GROUP_LAYERS",
    "AnimatedValue",
    "AudioStreamInfo",
    "Blending",
    "Clip",
    "ClipId",
    "Effect",
    "EffectId",
    "GeneratedSource",
    "GroupId",
    "Interpolation",
    "Keyframe",
    "LayerMode",
    "Marker",
    "MediaId",
    "MediaItem",
    "ParamValue",
    "Project",
    "ProjectSettings",
    "Scene",
    "SceneId",
    "SegmentId",
    "SubtitleOrigin",
    "Timeline",
    "Track",
    "TrackId",
    "TrackKind",
    "Transcript",
    "TranscriptSegment",
    "VideoStreamInfo",
    "Word",
    "controlling_groups",
    "default_track_name",
    "draws_picture",
    "group_as_one",
    "group_layers",
    "group_reaches",
    "heard_stream",
    "new_clip_id",
    "new_effect_id",
    "new_group_id",
    "new_media_id",
    "new_scene_id",
    "new_segment_id",
    "new_track_id",
    "plays_sound",
]
