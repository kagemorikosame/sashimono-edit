"""AI に見せる編集操作

読み取りと変更をはっきり分けてある（:attr:`Operation.writes`） 変更系だけに確認を
挟めるようにするためで、この区別が無いと「全部確認する」か「何も確認しない」かの
どちらかになる

ここは Qt も MCP も知らない 素の関数として書いてあるので、テストではホストを
偽物に差し替えるだけで全部のツールを試せる MCP のツールに変換するのは
:mod:`sashimono.ai.server` の仕事
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from sashimono.ai.host import EditorHost, ToolError
from sashimono.core.clipboard import copy_clips, paste_commands
from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddScene,
    AddTrack,
    Command,
    GroupClips,
    MoveClip,
    MoveClips,
    ParamPath,
    ParamTarget,
    RemoveClip,
    RemoveClips,
    RippleCut,
    SetClipProperty,
    SetKeyframe,
    SetParam,
    SetResolution,
    SetSegmentText,
    SetTrackHeights,
    SetTrackState,
    SetTranscript,
    SplitClip,
    TrimClip,
    UngroupClips,
    Voice,
    burn_subtitles,
    insert_generated,
    insert_media,
    insert_scene,
    new_scene,
    subtitle_voices,
)
from sashimono.core.commands.insert import new_track
from sashimono.core.commands.layers import places_mixed
from sashimono.core.jetcut import plan_cuts
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    EffectId,
    GeneratedSource,
    Interpolation,
    MediaId,
    MediaItem,
    ParamValue,
    Project,
    SceneId,
    SegmentId,
    Track,
    TrackId,
    TrackKind,
)
from sashimono.core.projection import project_timeline, subtitle_stream
from sashimono.core.timebase import format_timecode
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE, TEXT, TRANSITION, source_registry
from sashimono.effects.spec import (
    CheckSpec,
    ColorSpec,
    ParameterSpec,
    ParamInput,
    SelectSpec,
    TrackSpec,
)

__all__ = ["OPERATIONS", "ImageResult", "Operation", "find_operation"]

#: プレビュー画像の既定の横幅 小さめにしてあるのは、AI が見るのは
#: 「意図した絵になっているか」であって、画素を数えるわけではないため
DEFAULT_PREVIEW_WIDTH = 640
MAX_PREVIEW_WIDTH = 1280


@dataclass(frozen=True, slots=True)
class ImageResult:
    """画像を返すツールの戻り値"""

    png: bytes
    caption: str = ""


@dataclass(frozen=True, slots=True)
class Operation:
    """AI に見せるツール 1 つ"""

    name: str
    description: str
    #: MCP へ渡す JSON Schema 省略可能な引数を表せるよう、素の辞書で持つ
    schema: dict[str, Any]
    handler: Callable[[EditorHost, dict[str, Any]], object]
    #: プロジェクトを変えるか 確認ダイアログの要否がこれで決まる
    writes: bool = False

    def __call__(self, host: EditorHost, arguments: dict[str, Any]) -> object:
        return self.handler(host, arguments)


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _integer(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def _number(description: str) -> dict[str, Any]:
    return {"type": "number", "description": description}


def _boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


# --- 取り出しの補助 ---


def _project(host: EditorHost) -> Project:
    return host.project


def _require_clip(project: Project, clip_id: str) -> tuple[Track, Clip]:
    located = project.timeline.locate_clip(ClipId(clip_id))
    if located is None:
        raise ToolError(f"クリップが見つかりません: {clip_id}（list_clips で一覧を取れます）")
    return located


def _require_media(project: Project, media_id: str) -> MediaItem:
    item = project.find_media(MediaId(media_id))
    if item is None:
        raise ToolError(f"素材が見つかりません: {media_id}（list_media で一覧を取れます）")
    return item


def _target_clip(host: EditorHost, arguments: dict[str, Any]) -> tuple[Track, Clip]:
    """引数のクリップ、無ければ選択中のクリップ

    「選んでいるやつに掛けて」という指示が通るようにする 毎回 ID を聞き返すのは
    会話として重い
    """
    clip_id = str(arguments.get("clip_id") or "")
    if not clip_id:
        selected = host.selected_clip
        if selected is None:
            raise ToolError("clip_id を指定してください（選択中のクリップもありません）")
        clip_id = str(selected)
    return _require_clip(_project(host), clip_id)


def _target_clips(host: EditorHost, arguments: dict[str, Any]) -> tuple[ClipId, ...]:
    """引数の ``clip_ids``、無ければ選んでいるクリップすべて

    画面で何本か選んでから「これをまとめて 2 秒後ろへ」と頼めるようにする
    見つからない ID が 1 つでもあれば止める 一部だけ動かすと、どれが動いたのかを
    AI も人も追えなくなる
    """
    if arguments.get("clip_ids") is None:
        if not host.selected_clips:
            raise ToolError("clip_ids を指定してください（選択中のクリップもありません）")
        return host.selected_clips
    # 空の一覧は「選択を使う」ではない 省略と同じに扱うと、何も指していない
    # つもりの呼び出しが、選んでいる全部を消したり動かしたりする
    clip_ids = _clip_id_list(host, arguments)
    if not clip_ids:
        raise ToolError("clip_ids が空です 選択中のクリップを対象にするなら省いてください")
    return clip_ids


def _with_groups(host: EditorHost, clip_ids: tuple[ClipId, ...]) -> tuple[ClipId, ...]:
    """グループに入ったクリップは仲間も足す 画面で 1 本つかむと束ごと動くのと揃える

    AI が 1 本だけ指したときに束が裂けると、人が画面で直す手間が増える
    """
    timeline = _project(host).timeline
    expanded: dict[ClipId, None] = {}
    for clip_id in clip_ids:
        located = timeline.locate_clip(clip_id)
        if located is None or located[1].group_id is None:
            expanded[clip_id] = None
            continue
        for _, member in timeline.grouped_clips(located[1].group_id):
            expanded[member.id] = None
    return tuple(expanded)


def _clip_id_list(host: EditorHost, arguments: dict[str, Any]) -> tuple[ClipId, ...]:
    """``clip_ids`` を読んで確かめる 重なった ID は 1 つにする

    重なったまま渡すと「3 本を移動」のような本数が実際と食い違う 型が違えば
    ToolError にする 素の TypeError で落ちると、AI は何を直せばよいか分からない
    """
    raw = arguments.get("clip_ids")
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ToolError("clip_ids はクリップ ID の配列で渡してください")
    project = _project(host)
    for clip_id in raw:
        _require_clip(project, str(clip_id))
    # 最後に現れた位置を残す（画面の set_selection と同じ） 最後の 1 本が主になる
    # 約束なので、先に現れた位置を残すと [A, B, A] の主が B に化ける
    ids = [ClipId(str(clip_id)) for clip_id in raw]
    return tuple(reversed(dict.fromkeys(reversed(ids))))


def _clip_ids_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": "対象のクリップ 省略すると選択中のクリップすべて",
    }


# --- 読み取り ---


def _get_project(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    project = _project(host)
    settings = project.settings
    return {
        "name": project.name,
        "resolution": f"{settings.width}x{settings.height}",
        "frame_rate": f"{settings.frame_rate.num}/{settings.frame_rate.den}",
        "fps": round(float(settings.frame_rate.fps), 3),
        "sample_rate": settings.sample_rate,
        "duration_frames": project.duration,
        "duration_timecode": format_timecode(project.duration, project.rate),
        "media_count": len(project.media),
        "track_count": len(project.timeline.tracks),
        "playhead": host.playhead,
        "can_undo": host.document.can_undo,
        "active_scene": str(host.active_scene) if host.active_scene is not None else None,
        "scene_count": len(project.scenes),
        # 置き方の方式 混合（mixed）なら素材は絵と音の 1 本、分ける（separated）なら
        # 映像と音声の 2 本をリンクで結ぶ 知らないと、AI が組の片方を探しに行く
        "layer_mode": settings.layer_mode,
    }


def _list_media(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    project = _project(host)
    return [
        {
            "media_id": str(item.id),
            "name": item.name,
            "path": str(item.path),
            "duration_seconds": round(float(item.duration), 3),
            "has_video": item.has_video,
            "has_audio": item.has_audio,
            # 音声が何本もある素材（ゲームの音とマイクの声など） transcribe の audio で選ぶ
            "audio_count": len(item.audio_streams),
            "subtitle_count": len(item.transcript) if item.transcript is not None else 0,
            # 字幕は音声ごとに別 音声の番号（1 から）ごとの字幕の数
            "subtitle_counts": {
                str(number): len(item.transcript_for(stream.index) or ())
                for number, stream in enumerate(item.audio_streams, start=1)
            },
        }
        for item in project.media
    ]


def _list_tracks(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    return [
        {
            "track_id": str(track.id),
            "kind": track.kind.value,
            "name": track.name,
            "clip_count": len(track.clips),
            "height": track.height,
            "locked": track.locked,
            "muted": track.muted,
            "solo": track.solo,
        }
        for track in _project(host).timeline.tracks
    ]


def _list_clips(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    wanted = str(arguments.get("track_id") or "")
    clips = []
    for track in project.timeline.tracks:
        if wanted and str(track.id) != wanted:
            continue
        for clip in track.clips:
            media = project.find_media(clip.media_id) if clip.media_id is not None else None
            # 混合トラックのクリップは絵と音を 1 本で出すので、どちらを出すかも見せる
            # ほかの種類では読まない項目なので、出すと AI が効かない値を触りに行く
            mixed: dict[str, object] = (
                {"audio_stream": clip.audio_stream, "show_picture": clip.show_picture}
                if track.kind is TrackKind.MIXED
                else {}
            )
            clips.append(
                {
                    **mixed,
                    "clip_id": str(clip.id),
                    "track_id": str(track.id),
                    "track_kind": track.kind.value,
                    "start": clip.timeline_start,
                    "duration": clip.duration,
                    "start_timecode": format_timecode(clip.timeline_start, project.rate),
                    "source_in_seconds": round(float(clip.source_in), 3),
                    "speed": round(float(clip.speed), 4),
                    "media": media.name if media is not None else None,
                    "source": clip.source.kind if clip.source is not None else None,
                    "scene_id": str(clip.scene_id) if clip.scene_id is not None else None,
                    "group_id": str(clip.group_id) if clip.group_id is not None else None,
                    # 映像と音声の組（リンク） 方式を途中で変えた作品では、組の 2 本と
                    # レイヤーの 1 本が並ぶ 見えないと、AI は組の片方を見落とすか、
                    # 組の両方に同じ操作をして 2 回目で断られる
                    "link_group": (str(clip.link_group) if clip.link_group is not None else None),
                    # fixed はクリップが最初から持つ項目 外すことも並べ替えることもできない
                    # 渡しておかないと、AI が外そうとして断られるまで分からない
                    "effects": [
                        {
                            "effect_id": str(e.id),
                            "kind": e.kind,
                            "enabled": e.enabled,
                            "fixed": e.fixed,
                        }
                        for e in clip.effects
                    ],
                }
            )
    return clips


def _get_selection(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    project = _project(host)
    selected = host.selected_clip
    return {
        "clip_id": str(selected) if selected is not None else None,
        "clip_ids": [str(clip_id) for clip_id in host.selected_clips],
        "playhead": host.playhead,
        "playhead_timecode": format_timecode(host.playhead, project.rate),
    }


def _list_effects(host: EditorHost, arguments: dict[str, Any]) -> object:
    del host, arguments
    definitions = []
    for definition in registry.all():
        definitions.append(
            {
                "kind": definition.kind,
                "label": definition.label,
                "category": definition.category,
                "parameters": [_describe_spec(spec) for spec in definition.parameters],
            }
        )
    definitions.extend(
        {
            "kind": source.kind,
            "label": source.label,
            "category": "オブジェクト",
            "parameters": [_describe_spec(spec) for spec in source.parameters],
        }
        for source in (TEXT, SHAPE)
    )
    return definitions


def _describe_spec(spec: object) -> dict[str, Any]:
    """パラメータ 1 つの説明 AI が値の範囲を外さないよう、上下限まで見せる"""
    described: dict[str, Any] = {
        "name": getattr(spec, "name", ""),
        "label": getattr(spec, "label", ""),
    }
    if isinstance(spec, TrackSpec):
        described.update(
            {"type": "number", "min": spec.minimum, "max": spec.maximum, "unit": spec.unit}
        )
    elif isinstance(spec, SelectSpec):
        described.update({"type": "select", "choices": [value for value, _ in spec.choices]})
    elif isinstance(spec, ColorSpec):
        described["type"] = "color"
        described["format"] = "#RRGGBB か #RRGGBBAA、または [R, G, B, A]（0..1）"
    else:
        described["type"] = type(spec).__name__.replace("Spec", "").lower()
    return described


def _audio_stream(media: MediaItem, arguments: dict[str, Any]) -> int | None:
    """引数 ``audio``（1 から数えた音声の番号）を音声ストリームの番号へ 省けば ``None``（1 本目）

    番号はタイムラインの札（音声 N）と同じく 1 から数える ffprobe の番号は映像を含み、
    AI が素材ごとに数え直すことになる
    """
    if arguments.get("audio") is None:
        return None
    try:
        number = int(arguments["audio"])
    except (TypeError, ValueError) as exc:
        raise ToolError("audio は 1 から数えた音声の番号です") from exc
    if not 1 <= number <= len(media.audio_streams):
        raise ToolError(
            f"{media.name} の音声は {len(media.audio_streams)} 本です（audio は 1 から）"
        )
    return media.audio_streams[number - 1].index


def _audio_number(media: MediaItem | None, stream: int | None) -> int:
    """音声ストリームの番号を 1 から数えた番号へ（:func:`_audio_stream` の逆）"""
    if media is None:
        return 1
    key = media.transcript_stream(stream)
    known = [s.index for s in media.audio_streams]
    return known.index(key) + 1 if key in known else 1


def _get_subtitles(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    wanted = str(arguments.get("media_id") or "")
    wanted_audio = arguments.get("audio")
    rows = []
    for subtitle in project_timeline(project):
        located = project.timeline.locate_clip(subtitle.clip_id)
        media_id = located[1].media_id if located is not None else None
        if wanted and str(media_id) != wanted:
            continue
        media = project.find_media(media_id) if media_id is not None else None
        audio = _audio_number(
            media,
            subtitle_stream(project, located[0], located[1]) if located is not None else None,
        )
        if wanted_audio is not None and audio != int(wanted_audio):
            continue
        rows.append(
            {
                "segment_id": str(subtitle.segment.id),
                "media_id": str(media_id) if media_id is not None else None,
                # 字幕は音声ごとに別 set_subtitle_text と clean_subtitles へ同じ番号を渡す
                "audio": audio,
                "text": subtitle.segment.text,
                "start": subtitle.start_frame,
                "end": subtitle.end_frame,
                "start_timecode": format_timecode(subtitle.start_frame, project.rate),
                "source_start_seconds": round(float(subtitle.segment.start), 3),
            }
        )
    return rows


def _preview_frame(host: EditorHost, arguments: dict[str, Any]) -> object:
    """指定フレームを合成して画像で返す

    これが無いと、AI は自分の編集結果を確かめる手段が無く、当てずっぽうになる
    描く前に再生を止める 再生しながら別のフレームを描くと GL の資源を取り合う
    """
    project = _project(host)
    frame = int(arguments.get("frame", host.playhead))
    frame = max(0, min(frame, max(project.duration - 1, 0)))
    width = int(arguments.get("width", DEFAULT_PREVIEW_WIDTH))
    width = max(160, min(width, MAX_PREVIEW_WIDTH))

    host.stop_playback()
    png = host.render_png(frame, width=width)
    when = format_timecode(frame, project.rate)
    return ImageResult(png=png, caption=f"{when}（{frame} フレーム）")


def _get_history(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    document = host.document
    return {
        "can_undo": document.can_undo,
        "undo_label": document.undo_label,
        "recent": list(document.history_labels[-10:]),
    }


# --- 変更 ---


def _import_media(host: EditorHost, arguments: dict[str, Any]) -> object:
    raw = arguments.get("paths") or []
    if isinstance(raw, str):
        raw = [raw]
    paths = [Path(str(entry)) for entry in raw]
    if not paths:
        raise ToolError("paths が空です")

    project = _project(host)
    commands: list[Command] = []
    added: list[str] = []
    for path in paths:
        if not path.exists():
            raise ToolError(f"ファイルがありません: {path}")
        media = host.probe(path)
        batch = insert_media(project, media, at_frame=None, split_audio=host.splits_media)
        for command in batch:
            project = command.apply(project)
        commands.extend(batch)
        host.analyze(media)
        added.append(f"{media.name} ({media.id})")

    host.apply_commands(commands, f"素材を読み込み: {len(paths)} 件")
    return {"imported": added}


def _add_track(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    # 省いたときは方式に合わせる 混合の作品で映像トラックを足すと、AI が置いた物だけ
    # 分けた方式のトラックへ入り、画面で足した物と並びが食い違う
    fallback = TrackKind.MIXED.value if places_mixed(project) else TrackKind.VIDEO.value
    kind = str(arguments.get("kind") or fallback).lower()
    try:
        track_kind = TrackKind(kind)
    except ValueError:
        raise ToolError("kind は video・audio・mixed のどれかです") from None
    # 画面の「トラックを追加」と同じ所で作る 名前とソロの引き継ぎを 1 か所で決めるため
    track = new_track(project, track_kind).track
    if arguments.get("name"):
        track = replace(track, name=str(arguments["name"]))
    host.apply_commands([AddTrack(track)], f"トラックを追加: {track.name}")
    return {"track_id": str(track.id), "name": track.name}


def _set_track_state(host: EditorHost, arguments: dict[str, Any]) -> object:
    track_id = str(arguments.get("track_id", ""))
    track = next((t for t in _project(host).timeline.tracks if str(t.id) == track_id), None)
    if track is None:
        raise ToolError(f"トラックが見つかりません: {track_id}（list_tracks で確かめてください）")
    changes: dict[str, bool] = {
        name: bool(arguments[name]) for name in ("muted", "solo", "locked") if name in arguments
    }
    if not changes:
        raise ToolError("muted・solo・locked のどれかを指定してください")
    command = SetTrackState(
        track.id,
        muted=changes.get("muted"),
        solo=changes.get("solo"),
        locked=changes.get("locked"),
    )
    host.apply_commands([command], command.label)
    result: dict[str, object] = {"track_id": track_id}
    result.update(changes)
    return result


def _set_resolution(host: EditorHost, arguments: dict[str, Any]) -> object:
    command = SetResolution(int(arguments.get("width", 0)), int(arguments.get("height", 0)))
    host.apply_commands([command], command.label)
    return {"resolution": f"{command.width}x{command.height}"}


def _place_media(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    media = _require_media(project, str(arguments.get("media_id", "")))
    at_frame = arguments.get("at_frame")
    commands = insert_media(
        project,
        media,
        at_frame=int(at_frame) if at_frame is not None else None,
        split_audio=host.splits_media,
    )
    if not commands:
        raise ToolError(f"{media.name} は長さが無いので置けません")
    host.apply_commands(commands, f"配置: {media.name}")
    return {"placed": media.name}


def _add_text(host: EditorHost, arguments: dict[str, Any]) -> object:
    text = str(arguments.get("text", "")).strip()
    if not text:
        raise ToolError("text が空です")
    overrides: dict[str, float | str] = {"text": text}
    for name in ("size", "pos_x", "pos_y", "border_width"):
        if name in arguments:
            overrides[name] = float(arguments[name])

    project = _project(host)
    at_frame = arguments.get("at_frame")
    duration = int(arguments.get("duration", 150))
    if duration < 1:
        raise ToolError("duration は 1 フレーム以上です")
    commands = insert_generated(
        project,
        TEXT.create(**overrides),
        at_frame=int(at_frame) if at_frame is not None else host.playhead,
        duration=duration,
    )
    host.apply_commands(commands, f"テキストを追加: {text[:12]}")
    return {"added": text}


def _add_shape(host: EditorHost, arguments: dict[str, Any]) -> object:
    """図形を置く テロップの下に敷く帯や、目印の丸・矢印に使う

    前は道具が無く、AI はテキストしか置けなかった（AI テスト #3 で分かった）
    """
    kind = str(arguments.get("shape", "rect"))
    spec = SHAPE.spec("shape")
    choices = [value for value, _ in getattr(spec, "choices", ())]
    if kind not in choices:
        raise ToolError(f"shape は {'、'.join(choices)} のどれかです: {kind}")
    overrides: dict[str, ParamInput] = {"shape": kind}
    for name in ("width", "height", "pos_x", "pos_y", "rotation", "line_width", "corner_radius"):
        if name in arguments:
            overrides[name] = float(arguments[name])
    if "color" in arguments:
        color = _parse_color(str(arguments["color"]))
        if color is None:
            raise ToolError(f"color は #RRGGBB か #RRGGBBAA で渡してください: {arguments['color']}")
        overrides["color"] = color

    duration = int(arguments.get("duration", 150))
    if duration < 1:
        raise ToolError("duration は 1 フレーム以上です")
    at_frame = arguments.get("at_frame")
    commands = insert_generated(
        _project(host),
        SHAPE.create(**overrides),
        at_frame=int(at_frame) if at_frame is not None else host.playhead,
        duration=duration,
    )
    host.apply_commands(commands, f"図形を追加: {kind}")
    return {"added": kind}


#: 場面切り替えの切り替え方 生成オブジェクトの選択肢と同じ並び
_TRANSITION_STYLES = ("switch", "fade", "push", "slide", "overlay")


def _add_transition(host: EditorHost, arguments: dict[str, Any]) -> object:
    """下のトラックの切れ目に重ねる場面切り替えを置く"""
    style = str(arguments.get("style", "fade"))
    if style not in _TRANSITION_STYLES:
        raise ToolError(f"style は {'、'.join(_TRANSITION_STYLES)} のどれかです: {style}")
    duration = int(arguments.get("duration", 30))
    if duration < 1:
        raise ToolError("duration は 1 フレーム以上です")
    overrides: dict[str, float | str] = {"style": style}
    if "angle" in arguments:
        overrides["angle"] = float(arguments["angle"])
    if "target" in arguments:
        target = str(arguments["target"])
        if target not in ("before", "after"):
            raise ToolError(f"target は before か after です: {target}")
        overrides["target"] = target

    project = _project(host)
    at_frame = arguments.get("at_frame")
    commands = insert_generated(
        project,
        TRANSITION.create(**overrides),
        at_frame=int(at_frame) if at_frame is not None else host.playhead,
        duration=duration,
    )
    host.apply_commands(commands, f"場面切り替えを追加: {style}")
    return {"added": style, "duration": duration}


def _split_clip(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    frame = int(arguments.get("frame", host.playhead))
    host.apply_commands([SplitClip(clip.id, frame)], "クリップを分割")
    return {"split_at": frame}


def _trim_clip(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    head = int(arguments.get("head_delta", 0))
    tail = int(arguments.get("tail_delta", 0))
    if head == 0 and tail == 0:
        raise ToolError("head_delta か tail_delta のどちらかを指定してください")
    host.apply_commands([TrimClip(clip.id, head_delta=head, tail_delta=tail)], "クリップをトリム")
    return {"head_delta": head, "tail_delta": tail}


def _move_clip(host: EditorHost, arguments: dict[str, Any]) -> object:
    track, clip = _target_clip(host, arguments)
    start = int(arguments.get("timeline_start", clip.timeline_start))
    track_id = str(arguments.get("track_id") or "")
    members = _with_groups(host, (clip.id,))
    if len(members) > 1:
        # 1 本だけ動かすと束が裂ける 時刻だけなら仲間ごと同じだけずらせるが、
        # トラックを移すと仲間の行き先が決まらないので断る 今のトラックの指定は移動ではない
        if track_id and track_id != str(track.id):
            raise ToolError(
                "グループに入ったクリップはトラックを移せません"
                "（ungroup_clips で解くか、時刻だけ move_clips で動かしてください）"
            )
        command = MoveClips(members, start - clip.timeline_start)
        host.apply_commands([command], command.label)
        return {"timeline_start": start, "moved": [str(clip_id) for clip_id in members]}
    host.apply_commands(
        [MoveClip(clip.id, start, TrackId(track_id) if track_id else None)], "クリップを移動"
    )
    return {"timeline_start": start}


def _delete_clip(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    ripple = bool(arguments.get("ripple", False))
    members = _with_groups(host, (clip.id,))
    if len(members) > 1:
        # 1 本だけ消すと、画面で消したときと違って仲間が残る
        command = RemoveClips(members, ripple=ripple)
        host.apply_commands([command], command.label)
        return {"deleted": [str(clip_id) for clip_id in members], "ripple": ripple}
    host.apply_commands([RemoveClip(clip.id, ripple=ripple)], "クリップを削除")
    return {"deleted": str(clip.id), "ripple": ripple}


def _move_clips(host: EditorHost, arguments: dict[str, Any]) -> object:
    clip_ids = _with_groups(host, _target_clips(host, arguments))
    delta = int(arguments.get("delta", 0))
    if delta == 0:
        raise ToolError("delta に動かすフレーム数を指定してください（負で前へ）")
    command = MoveClips(clip_ids, delta)
    host.apply_commands([command], command.label)
    return {"moved": [str(clip_id) for clip_id in clip_ids], "delta": delta}


def _delete_clips(host: EditorHost, arguments: dict[str, Any]) -> object:
    clip_ids = _with_groups(host, _target_clips(host, arguments))
    ripple = bool(arguments.get("ripple", False))
    command = RemoveClips(clip_ids, ripple=ripple)
    host.apply_commands([command], command.label)
    return {"deleted": [str(clip_id) for clip_id in clip_ids], "ripple": ripple}


def _duplicate_clips(host: EditorHost, arguments: dict[str, Any]) -> object:
    """コピーして貼り付ける 画面のコピー・貼り付けと同じ決まりで置く

    AI には「クリップボードに入れておく」段を見せない 2 回に分けると、間に人が
    別のものをコピーしたとき、AI の知らない中身が貼られる
    """
    clip_ids = _with_groups(host, _target_clips(host, arguments))
    project = _project(host)
    content = copy_clips(project, clip_ids)
    at_frame = int(arguments.get("at_frame", host.playhead))
    try:
        commands = paste_commands(project, content, at_frame)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    host.apply_commands(commands, f"複製: {len(content.clips)} 本")
    pasted = [c.clip.id for c in commands if isinstance(c, AddClip)]
    host.select_clips(pasted)
    return {"pasted": [str(clip_id) for clip_id in pasted], "at_frame": max(0, at_frame)}


def _set_track_height(host: EditorHost, arguments: dict[str, Any]) -> object:
    tracks = _project(host).timeline.tracks
    track_id = str(arguments.get("track_id") or "")
    if track_id:
        chosen = [t for t in tracks if str(t.id) == track_id]
        if not chosen:
            raise ToolError(
                f"トラックが見つかりません: {track_id}（list_tracks で確かめてください）"
            )
    else:
        chosen = list(tracks)
    height = int(arguments.get("height", 0))
    command = SetTrackHeights(tuple((t.id, height) for t in chosen))
    host.apply_commands([command], command.label)
    # 範囲の外は端へ寄せられる 実際になった高さを返さないと、AI は頼んだ値に
    # なったと思い込む
    after = {t.id: t.height for t in _project(host).timeline.tracks}
    return {"heights": {str(t.id): after.get(t.id) for t in chosen}}


def _set_clip_property(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    name = str(arguments.get("name", ""))
    if name not in SetClipProperty.ALLOWED:
        raise ToolError(f"変えられるのは {'、'.join(SetClipProperty.ALLOWED)} です")
    value: object = arguments.get("value")
    if name == "speed":
        value = Fraction(str(value)).limit_denominator(1000)
    elif name == "enabled":
        value = bool(value)
    elif name == "stream_index":
        value = int(str(value))
    elif name == "hold_at":
        value = _hold_at(value)
    elif name == "source_in":
        # 読み方は絵を止める時刻と同じ（素材の頭からの秒） 解除の null は無い
        # 誤りの文面には項目名を出す hold_at の名前で返すと、AI が別の項目を直しに行く
        value = _hold_at(value, name)
        if value is None:
            raise ToolError("source_in は 0 以上の秒で渡してください")
    elif name in ("clip_to_below", "native_size"):
        value = bool(value)
    host.apply_commands([SetClipProperty(clip.id, name, value)], f"クリップの{name}を変更")
    return {"name": name, "value": str(value)}


def _hold_at(value: object, name: str = "hold_at") -> Fraction | None:
    """絵を止める素材の時刻（秒）を :class:`~fractions.Fraction` にする ``None`` は解除

    直さずに渡すと、文字列はクリップを作る所で落ち、整数や小数はモデルに入ったまま
    保存の所で落ちる（分数として書けない） 真偽値は数に読めるが（``True`` が 1 秒）、
    時刻として渡されることは無いので断る
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ToolError(f"{name} は秒の数か null で渡してください: {value!r}")
    try:
        seconds = Fraction(str(value)).limit_denominator(1000000)
    except (ValueError, ZeroDivisionError) as exc:
        raise ToolError(f"{name} を秒として読めません: {value!r}") from exc
    if seconds < 0:
        raise ToolError(f"{name} は 0 以上の秒です: {value!r}")
    return seconds


def _add_effect(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    kind = str(arguments.get("kind", ""))
    definition = registry.get(kind)
    if definition is None:
        available = "、".join(d.kind for d in registry.all())
        raise ToolError(f"そのエフェクトはありません: {kind}（使えるのは {available}）")

    raw = arguments.get("params") or {}
    if not isinstance(raw, dict):
        raise ToolError("params はオブジェクトで渡してください")
    effect = definition.create(**{str(k): v for k, v in raw.items()})
    host.apply_commands([AddEffect(clip.id, effect)], f"エフェクトを追加: {definition.label}")
    return {"effect_id": str(effect.id), "kind": kind}


def _param_path(clip_id: ClipId, arguments: dict[str, Any]) -> ParamPath:
    name = str(arguments.get("name", ""))
    if not name:
        raise ToolError("name が空です")
    effect_id = str(arguments.get("effect_id") or "")
    if effect_id:
        return ParamPath.of_effect(clip_id, EffectId(effect_id), name)
    target = str(arguments.get("target", "source")).lower()
    if target == "clip":
        return ParamPath.of_clip(clip_id, name)
    return ParamPath.of_source(clip_id, name)


def _set_param(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    path = _param_path(clip.id, arguments)
    value = arguments.get("value")
    resolved = _coerce_param(_project(host), path, value)
    host.apply_commands([SetParam(path, resolved)], f"{path.name} を変更")
    return {"name": path.name, "value": str(value)}


def _spec_for(project: Project, path: ParamPath) -> ParameterSpec | None:
    """そのパラメータの定義を引く

    UI と同じ定義を通して値を寄せるためにある ここを通さないと、色に
    ``"#FFFFFF"`` という文字列がそのまま入るような食い違いが起きる
    """
    located = project.timeline.locate_clip(path.clip_id)
    if located is None:
        return None
    _, clip = located

    if path.target is ParamTarget.EFFECT and path.effect_id is not None:
        effect = next((e for e in clip.effects if e.id == path.effect_id), None)
        definition = registry.get(effect.kind) if effect is not None else None
        return definition.spec(path.name) if definition is not None else None

    if path.target is ParamTarget.SOURCE and clip.source is not None:
        source = source_registry.get(clip.source.kind)
        return source.spec(path.name) if source is not None else None
    return None


def _coerce_param(project: Project, path: ParamPath, value: object) -> ParamValue:
    """AI が渡した値を、パラメータの型へ寄せる"""
    prepared: ParamInput
    if isinstance(value, str):
        parsed = _parse_color(value)
        prepared = parsed if parsed is not None else value
    elif isinstance(value, list):
        prepared = tuple(float(entry) for entry in value)
    elif isinstance(value, bool):
        prepared = 1.0 if value else 0.0
    elif isinstance(value, int | float):
        prepared = float(value)
    else:
        prepared = str(value)

    spec = _spec_for(project, path)
    if spec is not None:
        if isinstance(spec, CheckSpec):
            return int(spec.coerce(_as_check(value, prepared)))
        return spec.coerce(prepared)

    # 定義が引けないもの（クリップ自身の不透明度など）は数値として扱う
    if isinstance(prepared, float):
        return AnimatedValue(prepared)
    if isinstance(prepared, tuple):
        return prepared
    return str(prepared)


def _as_check(original: object, prepared: ParamInput) -> ParamInput:
    """チェック項目は、真偽値をそのまま渡した方が素直に決まる"""
    return original if isinstance(original, bool | str) else prepared


def _parse_color(text: str) -> tuple[float, ...] | None:
    """``#RRGGBB`` / ``#RRGGBBAA`` を 0..1 の組へ 色でなければ ``None``

    AI は色を 16 進で書いてくる ここで受けないと、色のパラメータに文字列が
    入って描画側で無視される（しかも見た目が変わらないので気付きにくい）
    """
    value = text.strip()
    if not value.startswith("#"):
        return None
    digits = value[1:]
    if len(digits) not in (6, 8) or any(c not in "0123456789abcdefABCDEF" for c in digits):
        return None
    channels = [int(digits[i : i + 2], 16) / 255.0 for i in range(0, len(digits), 2)]
    while len(channels) < 4:
        channels.append(1.0)
    return tuple(channels)


def _add_keyframe(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    path = _param_path(clip.id, arguments)
    frame = int(arguments.get("frame", host.playhead))
    try:
        value = float(arguments["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError("value に数値が要ります") from exc

    name = str(arguments.get("interpolation", "linear")).lower()
    try:
        interpolation = Interpolation(name)
    except ValueError as exc:
        choices = "、".join(i.value for i in Interpolation)
        raise ToolError(f"補間は {choices} のどれかです") from exc

    host.apply_commands(
        [SetKeyframe(path, frame, value, interpolation)], f"{path.name} にキーフレーム"
    )
    return {"name": path.name, "frame": frame, "value": value}


def _set_subtitle_text(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    media = _require_media(project, str(arguments.get("media_id", "")))
    segment_id = str(arguments.get("segment_id", ""))
    text = str(arguments.get("text", ""))
    stream = _audio_stream(media, arguments)
    host.apply_commands(
        [SetSegmentText(media.id, SegmentId(segment_id), text, stream=stream)], "字幕を編集"
    )
    return {"segment_id": segment_id, "text": text}


def _clean_subtitles(host: EditorHost, arguments: dict[str, Any]) -> object:
    from sashimono.asr.cleanup import CleanupOptions, clean_transcript

    project = _project(host)
    media = _require_media(project, str(arguments.get("media_id", "")))
    stream = _audio_stream(media, arguments)
    transcript = media.transcript_for(stream)
    if transcript is None:
        raise ToolError(f"{media.name} にはまだ字幕がありません")

    options = CleanupOptions(
        max_line_chars=int(arguments.get("max_line_chars", 20)),
        max_lines=int(arguments.get("max_lines", 2)),
        punctuation=str(arguments.get("punctuation", "keep")),
    )
    cleaned = clean_transcript(transcript, options)
    changed = sum(
        1
        for before, after in zip(transcript.segments, cleaned.segments, strict=False)
        if before.text != after.text
    )
    host.apply_commands([SetTranscript(media.id, cleaned, stream=stream)], "字幕を整形")
    return {"changed": changed, "remaining": len(cleaned)}


def _place_subtitles(host: EditorHost, arguments: dict[str, Any]) -> object:
    """字幕をテキストのクリップとしてタイムラインへ置く（焼き込み） 話し手ごとに別のレイヤー

    前は道具が無く、「字幕をテキストオブジェクトとして書き出して」と頼まれても AI には
    できなかった 画面の〔焼き込み〕と同じ ``burn_subtitles`` を通す
    """
    project = _project(host)
    voices: list[Voice] | None = None
    if arguments.get("media_id"):
        media = _require_media(project, str(arguments["media_id"]))
        if arguments.get("audio") is not None:
            voices = [(media.id, media.transcript_stream(_audio_stream(media, arguments)))]
        else:
            voices = [v for v in subtitle_voices(project) if v[0] == media.id]
    raw_segments = arguments.get("segment_ids")
    segments = {SegmentId(str(s)) for s in raw_segments} if raw_segments else None

    template: GeneratedSource | Clip = TEXT.create(size=48.0, pos_y=-380.0, border_width=4.0)
    if arguments.get("template_clip_id"):
        located = project.timeline.locate_clip(ClipId(str(arguments["template_clip_id"])))
        if located is None:
            raise ToolError(f"ひな形のクリップが見つかりません: {arguments['template_clip_id']}")
        clip = located[1]
        if clip.source is None or clip.source.kind != "text":
            raise ToolError("ひな形にできるのはテキストのクリップだけです")
        template = clip
    commands = burn_subtitles(project, template, voices=voices, segments=segments)
    if not commands:
        raise ToolError("置ける字幕がありません（タイムラインに出ている字幕が無い）")
    host.apply_commands(commands, "字幕を焼き込み")
    placed = sum(1 for c in commands if isinstance(c, AddClip))
    tracks = [c.track.name for c in commands if isinstance(c, AddTrack)]
    return {"placed": placed, "tracks": tracks}


def _jet_cut(host: EditorHost, arguments: dict[str, Any]) -> object:
    from sashimono.engine.audio.silence import SilenceOptions, detect_silence, keep_speech

    project = _project(host)
    media = _require_media(project, str(arguments.get("media_id", "")))
    stream = _audio_stream(media, arguments)
    waveform = host.waveform(media, stream)
    if waveform is None:
        raise ToolError(f"{media.name} の波形解析がまだ終わっていません 少し待ってください")

    options = SilenceOptions(
        threshold_db=float(arguments.get("threshold_db", -40.0)),
        min_silence=_seconds(arguments.get("min_silence", 0.5)),
        padding=_seconds(arguments.get("padding", 0.1)),
    )
    silences = detect_silence(waveform, options)
    transcript = media.transcript_for(stream)
    if bool(arguments.get("keep_speech", True)) and transcript is not None:
        silences = keep_speech(silences, transcript)

    ranges = plan_cuts(project, media.id, silences)
    if not ranges:
        raise ToolError("切れる無音が見つかりません threshold_db を上げてみてください")

    removed = sum(end - start for start, end in ranges)
    host.apply_commands([RippleCut(ranges)], f"無音カット: {len(ranges)} か所")
    return {
        "cuts": len(ranges),
        "removed_frames": removed,
        "removed_seconds": round(float(removed * project.rate.frame_duration), 2),
    }


def _seconds(value: object) -> Fraction:
    return Fraction(round(float(str(value)) * 100), 100)


def _transcribe(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    media = _require_media(project, str(arguments.get("media_id", "")))
    if not media.has_audio:
        raise ToolError(f"{media.name} に音声がありません")
    stream = _audio_stream(media, arguments)
    if media.transcript_for(stream) is not None and not bool(arguments.get("replace", False)):
        # 黙って置き換えると、人が直した字幕まで消える 置き換えるかは頼む側が決める
        raise ToolError(
            f"{media.name} の音声 {_audio_number(media, stream)} には字幕があります"
            " 置き換えるときは replace を true にしてください"
        )
    try:
        message = host.start_transcription(
            media.id, str(arguments.get("model", "large-v3")), audio_stream=stream
        )
    except RuntimeError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "started": message,
        "next": "しばらく待ってから transcription_status を見てください"
        "終わったら get_subtitles で結果を取れます",
    }


def _transcription_status(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    return {"status": host.transcription_status()}


def _undo(host: EditorHost, arguments: dict[str, Any]) -> object:
    steps = max(1, int(arguments.get("steps", 1)))
    document = host.document
    undone: list[str] = []
    for _ in range(steps):
        label = document.undo_label
        if label is None:
            break
        document.undo()
        undone.append(label)
    if not undone:
        raise ToolError("戻せる操作がありません")
    return {"undone": undone}


def _seek(host: EditorHost, arguments: dict[str, Any]) -> object:
    frame = max(0, int(arguments.get("frame", 0)))
    host.seek(frame)
    return {"playhead": frame}


def _list_scenes(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    project = _project(host)
    return {
        "active_scene": str(host.active_scene) if host.active_scene is not None else None,
        "scenes": [
            {
                "scene_id": str(scene.id),
                "name": scene.name,
                "duration_frames": scene.timeline.duration,
                "track_count": len(scene.timeline.tracks),
            }
            for scene in project.scenes
        ],
    }


def _add_scene(host: EditorHost, arguments: dict[str, Any]) -> object:
    name = str(arguments.get("name") or "").strip()
    if not name:
        raise ToolError("name にシーンの名前を指定してください")
    scene = new_scene(_project(host), name)
    # シーンの追加はどのタイムラインを開いていても同じ プロジェクトそのものへ足す
    host.apply_commands([AddScene(scene)], f"シーンを追加: {name}")
    if bool(arguments.get("open", True)):
        host.set_active_scene(scene.id)
    return {"scene_id": str(scene.id), "name": name, "opened": host.active_scene == scene.id}


def _set_active_scene(host: EditorHost, arguments: dict[str, Any]) -> object:
    raw = str(arguments.get("scene_id") or "")
    if not raw:
        host.set_active_scene(None)
        return {"active_scene": None}
    if _project(host).find_scene(SceneId(raw)) is None:
        raise ToolError(f"シーンが見つかりません: {raw}（list_scenes で一覧を取れます）")
    host.set_active_scene(SceneId(raw))
    return {"active_scene": raw}


def _place_scene(host: EditorHost, arguments: dict[str, Any]) -> object:
    raw = str(arguments.get("scene_id") or "")
    project = _project(host)
    scene = project.find_scene(SceneId(raw)) if raw else None
    if scene is None:
        raise ToolError(f"シーンが見つかりません: {raw}（list_scenes で一覧を取れます）")
    at_frame = int(arguments.get("at_frame", host.playhead))
    duration = arguments.get("duration")
    length = int(duration) if duration is not None else None
    if length is not None and length < 1:
        # 0 を既定の長さと読み替えると、AI が頼んだ長さと違うまま黙って置かれる
        raise ToolError("duration は 1 フレーム以上にしてください（省くとシーンの長さ）")
    commands = insert_scene(project, scene.id, at_frame=at_frame, duration=length)
    host.apply_commands(commands, f"シーンを置く: {scene.name}")
    placed = next((c.clip for c in commands if isinstance(c, AddClip)), None)
    return {
        "placed": scene.name,
        "at_frame": max(0, at_frame),
        "duration": placed.duration if placed is not None else length,
    }


def _group_clips(host: EditorHost, arguments: dict[str, Any]) -> object:
    clip_ids = _target_clips(host, arguments)
    command = GroupClips(clip_ids)
    host.apply_commands([command], command.label)
    return {"grouped": [str(clip_id) for clip_id in clip_ids]}


def _ungroup_clips(host: EditorHost, arguments: dict[str, Any]) -> object:
    clip_ids = _target_clips(host, arguments)
    command = UngroupClips(clip_ids)
    host.apply_commands([command], command.label)
    return {"ungrouped": [str(clip_id) for clip_id in clip_ids]}


def _select_clips(host: EditorHost, arguments: dict[str, Any]) -> object:
    # 省いたら断る 空の引数で呼ばれただけで選択が消えると、人が選んでいたものを
    # 失う 解きたいときは空の配列を明示してもらう
    if arguments.get("clip_ids") is None:
        raise ToolError("clip_ids を指定してください（選択を解くなら空の配列）")
    host.select_clips(list(_clip_id_list(host, arguments)))
    return {"selected": [str(clip_id) for clip_id in host.selected_clips]}


def _select(host: EditorHost, arguments: dict[str, Any]) -> object:
    clip_id = str(arguments.get("clip_id") or "")
    if clip_id:
        _require_clip(_project(host), clip_id)
        host.select_clip(ClipId(clip_id))
    else:
        host.select_clip(None)
    return {"selected": clip_id or None}


OPERATIONS: tuple[Operation, ...] = (
    Operation(
        name="get_project",
        description="プロジェクトの設定（解像度・fps・長さ）と再生ヘッドの位置を返す",
        schema=_schema({}),
        handler=_get_project,
    ),
    Operation(
        name="list_media",
        description="メディアプールの素材を一覧する media_id はここで得る",
        schema=_schema({}),
        handler=_list_media,
    ),
    Operation(
        name="list_tracks",
        description="タイムラインのトラックを一覧する",
        schema=_schema({}),
        handler=_list_tracks,
    ),
    Operation(
        name="list_clips",
        description="クリップを一覧する track_id を省くと全トラックが対象",
        schema=_schema({"track_id": _string("絞り込むトラック")}),
        handler=_list_clips,
    ),
    Operation(
        name="get_selection",
        description="選択中のクリップと再生ヘッドの位置",
        schema=_schema({}),
        handler=_get_selection,
    ),
    Operation(
        name="list_effects",
        description="使えるエフェクトと生成オブジェクト、そのパラメータ名と範囲",
        schema=_schema({}),
        handler=_list_effects,
    ),
    Operation(
        name="get_subtitles",
        description="タイムラインに出る字幕を、表示位置つきで一覧する",
        schema=_schema(
            {
                "media_id": _string("絞り込む素材"),
                "audio": _integer("絞り込む音声の番号（1 から）"),
            }
        ),
        handler=_get_subtitles,
    ),
    Operation(
        name="get_history",
        description="直近の操作履歴と、取り消せるかどうか",
        schema=_schema({}),
        handler=_get_history,
    ),
    Operation(
        name="preview_frame",
        description=(
            "そのフレームを合成して画像で返す 編集した結果を自分の目で確かめるために使う"
            "再生中なら止めてから描く"
        ),
        schema=_schema(
            {
                "frame": _integer("見たいフレーム 省略すると再生ヘッド"),
                "width": _integer("画像の横幅（160〜1280、既定 640）"),
            }
        ),
        handler=_preview_frame,
    ),
    Operation(
        name="seek",
        description="再生ヘッドを動かす",
        schema=_schema({"frame": _integer("移動先のフレーム")}, ["frame"]),
        handler=_seek,
    ),
    Operation(
        name="select_clip",
        description="クリップを選択する clip_id を空にすると選択を解く",
        schema=_schema({"clip_id": _string("選ぶクリップ")}),
        handler=_select,
    ),
    Operation(
        name="list_scenes",
        description="シーン（入れ子のタイムライン）の一覧と、いま開いているシーン",
        schema=_schema({}),
        handler=_list_scenes,
    ),
    Operation(
        name="set_active_scene",
        description=(
            "編集するシーンを切り替える scene_id を空にするとメイン "
            "以後の読み取りと編集は、そのシーンのタイムラインが相手になる"
        ),
        schema=_schema({"scene_id": _string("開くシーン 空ならメイン")}),
        handler=_set_active_scene,
    ),
    Operation(
        name="select_clips",
        description="何本かのクリップをまとめて選ぶ 空の配列で選択を解く 最後の 1 本が主になる",
        schema=_schema(
            {"clip_ids": {"type": "array", "items": {"type": "string"}, "description": "選ぶ"}},
            ["clip_ids"],
        ),
        handler=_select_clips,
    ),
    Operation(
        name="import_media",
        description="ファイルを読み込んでタイムラインの末尾へ置く",
        schema=_schema(
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ファイルパス",
                }
            },
            ["paths"],
        ),
        handler=_import_media,
        writes=True,
    ),
    Operation(
        name="place_media",
        description="読み込み済みの素材をタイムラインへ置く",
        schema=_schema(
            {
                "media_id": _string("置く素材"),
                "at_frame": _integer("置く位置 省略すると末尾"),
            },
            ["media_id"],
        ),
        handler=_place_media,
        writes=True,
    ),
    Operation(
        name="add_track",
        description="トラックを足す",
        schema=_schema(
            {
                "kind": _string(
                    "video・audio・mixed のどれか mixed は映像・音声・テキストを何でも置ける"
                    "レイヤー（YMM4 と同じ 番号が大きいレイヤーほど手前に描く）"
                    " 省くとプロジェクトの方式（get_project の layer_mode）に合わせる"
                ),
                "name": _string("表示名"),
            }
        ),
        handler=_add_track,
        writes=True,
    ),
    Operation(
        name="set_track_state",
        description=(
            "トラックのミュート・ソロ・ロックを切り替える 指定しなかった項目はそのまま"
            "ソロは同じ種類（映像なら映像）のほかのトラックを止める"
        ),
        schema=_schema(
            {
                "track_id": _string("対象のトラック"),
                "muted": _boolean("ミュートするか"),
                "solo": _boolean("ソロにするか"),
                "locked": _boolean("ロックするか"),
            },
            ["track_id"],
        ),
        handler=_set_track_state,
        writes=True,
    ),
    Operation(
        name="set_resolution",
        description=(
            "出力の解像度を変える 縦横とも偶数 クリップの位置は中央からの画素数なので"
            "中央のものは中央に残る 縦動画なら 1080x1920"
        ),
        schema=_schema(
            {"width": _integer("横の画素数"), "height": _integer("縦の画素数")},
            ["width", "height"],
        ),
        handler=_set_resolution,
        writes=True,
    ),
    Operation(
        name="add_text",
        description="テキストオブジェクトを置く テロップや字幕の焼き込みに使う",
        schema=_schema(
            {
                "text": _string("本文 改行を含めてよい"),
                "at_frame": _integer("置く位置 省略すると再生ヘッド"),
                "duration": _integer("長さ（フレーム、既定 150）"),
                "size": _number("文字サイズ"),
                "pos_x": _number("中央からの横位置"),
                "pos_y": _number("中央からの縦位置 正が上"),
                "border_width": _number("縁取りの太さ"),
            },
            ["text"],
        ),
        handler=_add_text,
        writes=True,
    ),
    Operation(
        name="add_shape",
        description=(
            "図形オブジェクトを置く テロップの下の帯や目印に使う shape は rect（矩形）"
            "rounded（角丸）ellipse（楕円）triangle star arrow background（画面全体）など"
        ),
        schema=_schema(
            {
                "shape": _string("図形の種類 既定は rect"),
                "width": _number("幅（画素）"),
                "height": _number("高さ（画素）"),
                "color": _string("色 #RRGGBB か #RRGGBBAA"),
                "at_frame": _integer("置く位置 省略すると再生ヘッド"),
                "duration": _integer("長さ（フレーム、既定 150）"),
                "pos_x": _number("中央からの横位置"),
                "pos_y": _number("中央からの縦位置 正が上"),
                "rotation": _number("回転（度）"),
                "line_width": _number("線の太さ"),
                "corner_radius": _number("角丸の半径"),
            },
            [],
        ),
        handler=_add_shape,
        writes=True,
    ),
    Operation(
        name="add_transition",
        description=(
            "場面切り替えを置く 下のトラックのクリップの切れ目に重ねて使う "
            "切り替え方は switch（切り替え）fade（クロスフェード）push（押し出し）"
            "slide（スライド）overlay（重ねる）"
        ),
        schema=_schema(
            {
                "style": _string("切り替え方 既定は fade"),
                "at_frame": _integer("置く位置 省略すると再生ヘッド"),
                "duration": _integer("長さ（フレーム、既定 30）"),
                "angle": _number("押し出しとスライドの向き（度、0 で右へ）"),
                "target": _string("スライドと重ねるで動かす・手前にする場面 before か after"),
            },
            [],
        ),
        handler=_add_transition,
        writes=True,
    ),
    Operation(
        name="split_clip",
        description="クリップを分割する clip_id を省くと選択中のクリップ",
        schema=_schema(
            {"clip_id": _string("対象"), "frame": _integer("分割位置 省略すると再生ヘッド")}
        ),
        handler=_split_clip,
        writes=True,
    ),
    Operation(
        name="trim_clip",
        description="クリップの端を動かす head_delta は正で短く、tail_delta は正で長くなる",
        schema=_schema(
            {
                "clip_id": _string("対象"),
                "head_delta": _integer("先頭を動かす量"),
                "tail_delta": _integer("末尾を動かす量"),
            }
        ),
        handler=_trim_clip,
        writes=True,
    ),
    Operation(
        name="move_clip",
        description="クリップを別の位置・別のトラックへ動かす",
        schema=_schema(
            {
                "clip_id": _string("対象"),
                "timeline_start": _integer("移動先"),
                "track_id": _string("移動先トラック"),
            },
            ["timeline_start"],
        ),
        handler=_move_clip,
        writes=True,
    ),
    Operation(
        name="delete_clip",
        description="クリップを消す ripple を真にすると後ろを詰める",
        schema=_schema({"clip_id": _string("対象"), "ripple": _boolean("詰めるか")}),
        handler=_delete_clip,
        writes=True,
    ),
    Operation(
        name="add_scene",
        description="空のシーンを作る 既定でそのまま開く（open を偽にすると開かない）",
        schema=_schema(
            {"name": _string("シーンの名前"), "open": _boolean("作ったシーンを開くか")},
            ["name"],
        ),
        handler=_add_scene,
        writes=True,
    ),
    Operation(
        name="place_scene",
        description=(
            "シーンを、いま開いているタイムラインへ 1 本のクリップとして置く "
            "自分自身や、自分を含むシーンは置けない"
        ),
        schema=_schema(
            {
                "scene_id": _string("置くシーン"),
                "at_frame": _integer("置く位置 省略すると再生ヘッド"),
                "duration": _integer("長さ（フレーム） 省略するとシーンの長さ"),
            },
            ["scene_id"],
        ),
        handler=_place_scene,
        writes=True,
    ),
    Operation(
        name="group_clips",
        description="2 本以上のクリップを 1 つのグループに束ねる 束ねたものは一緒に選ばれて動く",
        schema=_schema({"clip_ids": _clip_ids_schema()}),
        handler=_group_clips,
        writes=True,
    ),
    Operation(
        name="ungroup_clips",
        description="クリップが入っているグループを、仲間ごと解く",
        schema=_schema({"clip_ids": _clip_ids_schema()}),
        handler=_ungroup_clips,
        writes=True,
    ),
    Operation(
        name="move_clips",
        description=(
            "何本かのクリップをまとめて前後へずらす トラックは変えない "
            "リンクした映像と音声も一緒に動く 1 本でも動かせなければ何も動かない"
        ),
        schema=_schema(
            {"clip_ids": _clip_ids_schema(), "delta": _integer("ずらすフレーム数 負で前へ")},
            ["delta"],
        ),
        handler=_move_clips,
        writes=True,
    ),
    Operation(
        name="delete_clips",
        description="何本かのクリップをまとめて消す ripple を真にすると消したぶんを詰める",
        schema=_schema({"clip_ids": _clip_ids_schema(), "ripple": _boolean("詰めるか")}),
        handler=_delete_clips,
        writes=True,
    ),
    Operation(
        name="duplicate_clips",
        description=(
            "クリップをコピーして貼り付ける（画面のコピー・貼り付けと同じ） 並びの間隔は保つ "
            "元のトラックが塞がっていれば同じ種類の別のトラック、無ければ新しく作る "
            "貼ったクリップが選ばれた状態になる"
        ),
        schema=_schema(
            {
                "clip_ids": _clip_ids_schema(),
                "at_frame": _integer("貼る先頭の位置 省略すると再生ヘッド"),
            }
        ),
        handler=_duplicate_clips,
        writes=True,
    ),
    Operation(
        name="set_track_height",
        description=(
            "タイムラインでのトラックの高さ（画素）を変える 28〜240 の外は端へ寄せる "
            "track_id を省くと全トラック 既定は 60"
        ),
        schema=_schema(
            {"track_id": _string("対象のトラック"), "height": _integer("高さ（画素）")},
            ["height"],
        ),
        handler=_set_track_height,
        writes=True,
    ),
    Operation(
        name="set_clip_property",
        description=(
            "クリップの blend_mode / speed / enabled / stream_index / hold_at / source_in /"
            " clip_to_below / native_size を変える"
            " hold_at は絵を止める素材の時刻（秒） null で止めない"
            " source_in は素材のどこから再生するか（秒） clip_to_below はすぐ下のクリップの形で"
            "切り抜くか native_size は素材を画素の大きさで置くか（偽なら画面に収める）"
        ),
        schema=_schema(
            {
                "clip_id": _string("対象"),
                "name": _string("項目名"),
                "value": {"description": "新しい値"},
            },
            ["name", "value"],
        ),
        handler=_set_clip_property,
        writes=True,
    ),
    Operation(
        name="add_effect",
        description="クリップにエフェクトを積む 使える kind は list_effects で分かる",
        schema=_schema(
            {
                "clip_id": _string("対象"),
                "kind": _string("エフェクトの種類"),
                "params": {"type": "object", "description": "初期値"},
            },
            ["kind"],
        ),
        handler=_add_effect,
        writes=True,
    ),
    Operation(
        name="set_param",
        description=(
            "パラメータを変える effect_id を渡せばそのエフェクト、"
            "省略すればテキストや図形の中身（target=clip でクリップ自身）"
        ),
        schema=_schema(
            {
                "clip_id": _string("対象"),
                "effect_id": _string("エフェクト"),
                "target": _string("source か clip"),
                "name": _string("パラメータ名"),
                "value": {"description": "新しい値"},
            },
            ["name", "value"],
        ),
        handler=_set_param,
        writes=True,
    ),
    Operation(
        name="add_keyframe",
        description="パラメータにキーフレームを打つ 値は数値のみ",
        schema=_schema(
            {
                "clip_id": _string("対象"),
                "effect_id": _string("エフェクト"),
                "target": _string("source か clip"),
                "name": _string("パラメータ名"),
                "frame": _integer("位置 省略すると再生ヘッド"),
                "value": _number("その位置での値"),
                "interpolation": _string("linear / ease / hold / bezier"),
            },
            ["name", "value"],
        ),
        handler=_add_keyframe,
        writes=True,
    ),
    Operation(
        name="set_subtitle_text",
        description="字幕 1 枚の本文を書き換える 素材に紐付くので全出現箇所に反映される",
        schema=_schema(
            {
                "media_id": _string("素材"),
                "segment_id": _string("字幕"),
                "text": _string("新しい本文"),
                "audio": _integer(
                    "音声の番号（1 から タイムラインの「音声 N」と同じ） 省くと 1 本目"
                    " 字幕は音声ごとに別"
                ),
            },
            ["media_id", "segment_id", "text"],
        ),
        handler=_set_subtitle_text,
        writes=True,
    ),
    Operation(
        name="clean_subtitles",
        description="フィラー語を落とし、改行位置を整える",
        schema=_schema(
            {
                "media_id": _string("素材"),
                "max_line_chars": _integer("1 行の文字数（0 で折り返さない）"),
                "max_lines": _integer("行数の上限"),
                "punctuation": _string("keep / space / strip"),
                "audio": _integer(
                    "音声の番号（1 から タイムラインの「音声 N」と同じ） 省くと 1 本目"
                    " 字幕は音声ごとに別"
                ),
            },
            ["media_id"],
        ),
        handler=_clean_subtitles,
        writes=True,
    ),
    Operation(
        name="place_subtitles",
        description=(
            "字幕をテキストオブジェクトとしてタイムラインへ置く（焼き込み）"
            " 話し手（素材と音声）ごとに別のレイヤーへ入れる 省けば出ている字幕を全部"
            " 見た目はひな形のテキストのクリップを写す（省けば下寄せ・縁取りの既定）"
        ),
        schema=_schema(
            {
                "media_id": _string("絞り込む素材"),
                "audio": _integer("絞り込む音声の番号（1 から） media_id と一緒に"),
                "segment_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "置く字幕の ID（get_subtitles の segment_id） 省けば全部",
                },
                "template_clip_id": _string("見た目を写すテキストのクリップ"),
            }
        ),
        handler=_place_subtitles,
        writes=True,
    ),
    Operation(
        name="jet_cut",
        description="無音区間をタイムラインからまとめて削って詰める",
        schema=_schema(
            {
                "media_id": _string("素材"),
                "threshold_db": _number("無音とみなす音量（既定 -40）"),
                "min_silence": _number("最短の無音（秒、既定 0.5）"),
                "padding": _number("前後に残す余白（秒、既定 0.1）"),
                "keep_speech": _boolean("字幕のある区間は切らない（既定 true）"),
                "audio": _integer(
                    "音声の番号（1 から タイムラインの「音声 N」と同じ） 省くと 1 本目"
                    " 字幕は音声ごとに別"
                ),
            },
            ["media_id"],
        ),
        handler=_jet_cut,
        writes=True,
    ),
    Operation(
        name="transcribe",
        description="素材の字幕起こしを始める 終わるまで数分かかる",
        schema=_schema(
            {
                "media_id": _string("素材"),
                "model": _string("モデル名"),
                "audio": _integer(
                    "起こす音声の番号（1 から タイムラインの「音声 N」と同じ） 省くと 1 本目"
                    " 音声の本数は list_media の audio_count"
                ),
                "replace": _boolean(
                    "その音声に字幕があるとき置き換える（既定 false 既にあれば断る）"
                ),
            },
            ["media_id"],
        ),
        handler=_transcribe,
        writes=True,
    ),
    Operation(
        name="transcription_status",
        description=(
            "字幕起こしの様子を見る 走っている物と順番待ちの物を並べる"
            " 起こしは 1 本ずつ順に走り、終わった物はその素材と音声の字幕へ入る"
        ),
        schema=_schema({}),
        handler=_transcription_status,
    ),
    Operation(
        name="undo",
        description="直前の操作を取り消す",
        schema=_schema({"steps": _integer("戻す段数（既定 1）")}),
        handler=_undo,
        writes=True,
    ),
)


def find_operation(name: str) -> Operation | None:
    for operation in OPERATIONS:
        if operation.name == name:
            return operation
    return None
