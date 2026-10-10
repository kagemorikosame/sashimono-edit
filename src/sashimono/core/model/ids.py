"""モデル要素の識別子

型ごとに :func:`typing.NewType` を分けているのは、``TrackId`` を受け取るべき場所に
``ClipId`` を渡す取り違えを mypy に検出させるため 実行時の表現はただの文字列
"""

from __future__ import annotations

import uuid
from typing import NewType

__all__ = [
    "ClipId",
    "EffectId",
    "GroupId",
    "MediaId",
    "SceneId",
    "SegmentId",
    "StrokeId",
    "TrackId",
    "new_clip_id",
    "new_effect_id",
    "new_group_id",
    "new_media_id",
    "new_scene_id",
    "new_segment_id",
    "new_stroke_id",
    "new_track_id",
]

MediaId = NewType("MediaId", str)
TrackId = NewType("TrackId", str)
ClipId = NewType("ClipId", str)
EffectId = NewType("EffectId", str)
GroupId = NewType("GroupId", str)
SegmentId = NewType("SegmentId", str)
SceneId = NewType("SceneId", str)
#: テキストの縁取りの層（:class:`~sashimono.core.model.stroke.Stroke`）
StrokeId = NewType("StrokeId", str)


def _generate() -> str:
    # 16 桁あれば衝突は実用上考えなくてよく、プロジェクトファイルも読みやすい
    return uuid.uuid4().hex[:16]


def new_media_id() -> MediaId:
    return MediaId(_generate())


def new_track_id() -> TrackId:
    return TrackId(_generate())


def new_clip_id() -> ClipId:
    return ClipId(_generate())


def new_effect_id() -> EffectId:
    return EffectId(_generate())


def new_group_id() -> GroupId:
    return GroupId(_generate())


def new_segment_id() -> SegmentId:
    return SegmentId(_generate())


def new_scene_id() -> SceneId:
    return SceneId(_generate())


def new_stroke_id() -> StrokeId:
    return StrokeId(_generate())
