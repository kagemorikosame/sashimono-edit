"""AI に見せる編集操作

読み取りと変更をはっきり分けてある（:attr:`Operation.writes`） 変更系だけに確認を
挟めるようにするためで、この区別が無いと「全部確認する」か「何も確認しない」かの
どちらかになる 変更系はさらに、取り消せる編集と取り消せない操作に分ける
（:attr:`Operation.safety` :mod:`sashimono.ai.safety` #350）

ここは Qt も MCP も知らない 素の関数として書いてあるので、テストではホストを
偽物に差し替えるだけで全部のツールを試せる MCP のツールに変換するのは
:mod:`sashimono.ai.server` の仕事
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from sashimono.ai.host import EXPORT_RUNNING, EditorHost, ExportRequest, ToolError
from sashimono.ai.safety import Consent, Safety, pinned_place, writable_path
from sashimono.core.clipboard import copy_clips, paste_commands
from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddScene,
    AddStroke,
    AddTrack,
    Command,
    GroupClips,
    MoveClip,
    MoveClips,
    MoveStroke,
    ParamPath,
    ParamTarget,
    RemoveClip,
    RemoveClips,
    RemoveStroke,
    RippleCut,
    SetClipProperty,
    SetKeyframe,
    SetParam,
    SetResolution,
    SetSegmentText,
    SetStrokeEnabled,
    SetTrackHeights,
    SetTrackState,
    SetTranscript,
    SplitClip,
    TrimClip,
    UngroupClips,
    Voice,
    burn_defaults,
    burn_subtitles,
    export_range,
    head_room,
    insert_generated,
    insert_media,
    insert_scene,
    new_scene,
    subtitle_voices,
    subtitle_wrap_width,
    tail_room,
    trimmable,
)
from sashimono.core.commands.edit import STOP_LOCKED
from sashimono.core.commands.insert import new_track
from sashimono.core.commands.layers import places_mixed
from sashimono.core.io.atomic import Expected, expect_now
from sashimono.core.io.serialize import SUFFIX, json_text
from sashimono.core.io.subtitles import SUBTITLE_SUFFIXES, subtitles_in_range
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
    Stroke,
    StrokeId,
    Track,
    TrackId,
    TrackKind,
)
from sashimono.core.projection import project_timeline
from sashimono.core.timebase import Rounding, format_timecode, seconds_to_frame
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
from sashimono.effects.strokes import (
    STROKE,
    STROKE_EFFECT_KINDS,
    next_stroke,
    takes_stroke_effect,
)
from sashimono.engine.cache import SceneState
from sashimono.engine.encode import (
    BITRATE_RANGE_MBPS,
    CONTAINER_SUFFIXES,
    DEFAULT_BITRATE_MBPS,
    RANGE_ALL,
    RANGE_WORK_AREA,
    available_video_codecs,
    chosen_range,
    export_target,
)
from sashimono.engine.scenes import DEFAULT_MIN_SCENE, DEFAULT_THRESHOLD, SceneSignal

__all__ = [
    "ASK_USER",
    "MAX_WAIT_SECONDS",
    "OPERATIONS",
    "Choice",
    "ImageResult",
    "Operation",
    "Pending",
    "QuestionSpec",
    "find_operation",
    "question_spec",
]

#: プレビュー画像の既定の横幅 小さめにしてあるのは、AI が見るのは
#: 「意図した絵になっているか」であって、画素を数えるわけではないため
DEFAULT_PREVIEW_WIDTH = 640
MAX_PREVIEW_WIDTH = 1280

#: 書き出しの codec の「自動」 画面の書き出しの窓の「自動（使えなければ次の候補）」と同じ
AUTO = "auto"


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
    #: 人に尋ねる道具か（#296） 編集画面では動かさず、ブリッジが答えを待つ
    #: ``handler`` は引数を確かめて :class:`QuestionSpec` を返すだけ
    asks: bool = False
    #: 「元に戻す」で戻らない操作か（上書き保存・書き出し・ファイルを開く・外へ送るなど）
    #: 真なら「変更を自動で承認」を入れていても毎回確かめる（2026-10-14 利用者の決定）
    irreversible: bool = False
    #: 取り消せない操作で、確かめの窓に出す「何をどこへ」を作る 編集画面のスレッドで呼ぶ
    #: 書けない場所ならここで :class:`ToolError` を投げ、窓を出さずに断る
    consent: Callable[[EditorHost, dict[str, Any]], Consent] | None = None
    #: 取り消せない操作の本体 人が許した確かめの中身（:class:`Consent`）を受け取って動く
    #: ブリッジは許した後にこれを呼ぶ 書く処理は確かめの中身の身元（``basis``）で照らす
    #: 書く処理の中で身元を採り直すと、照らし合わせの後の変更を受け入れてしまう（#327）
    #: 無ければ ``handler`` を呼ぶ（書く先の身元を持たない道具）
    approved: Callable[[EditorHost, dict[str, Any], Consent], object] | None = None

    def __post_init__(self) -> None:
        if not self.irreversible:
            return
        # 取り消せない操作は、何をどこへするかを見せずに確かめても判断できない
        # 変えない道具・質問の道具を取り消せないと書くと、確かめの窓だけが出て中身が無い
        if self.consent is None or not self.writes or self.asks:
            raise ValueError(f"{self.name}: 取り消せない操作は writes と consent が要ります")

    @property
    def safety(self) -> Safety:
        """道具の区分 確かめの要否がこれで決まる（:mod:`sashimono.ai.safety`）"""
        if self.irreversible:
            return Safety.IRREVERSIBLE
        return Safety.UNDOABLE if self.writes else Safety.READ_ONLY

    def __call__(self, host: EditorHost, arguments: dict[str, Any]) -> object:
        return self.handler(host, arguments)


#: 質問の道具の名前 チャット欄が呼び出しの行を出し分けるのに使う
ASK_USER = "ask_user"

#: 選択肢の数 1 つは質問ではなく、5 つ以上はチャット欄に収まらず選ぶのも手間になる
CHOICES_RANGE = (2, 4)
#: 長さの上限（文字） チャット欄の幅（狭い画面で 300 画素ほど）でボタンの名前が
#: 1 行に収まり、説明が 2 行ほどで済む長さ 超えたら切らずに断り、AI に縮めさせる
#: 切ると、選択肢の違いを書いた所が消えて選べなくなる
QUESTION_LIMIT = 200
LABEL_LIMIT = 16
DESCRIPTION_LIMIT = 60


@dataclass(frozen=True, slots=True)
class Choice:
    """質問の選択肢 1 つ"""

    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    """AI からの質問 1 つ（ask_user の引数を確かめた物）"""

    text: str
    choices: tuple[Choice, ...]
    #: 複数を選べるか
    multiple: bool = False
    #: 選択肢に無い答えを書いてよいか
    free_text: bool = True


def question_spec(arguments: dict[str, Any]) -> QuestionSpec:
    """ask_user の引数を確かめる 合わなければ、直し方を書いて断る

    型は緩く読まない ``bool("false")`` は真なので、AI が文字列で渡すと逆の設定の
    質問が出る 文字列の欄に数や辞書が来たときも、``str`` で読むと中身の分からない
    選択肢が並ぶ どちらも断って、正しい型で渡し直させる
    """
    text = _text_argument(arguments, "question", "question")
    if not text:
        raise ToolError("question（質問の文）が空です")
    if len(text) > QUESTION_LIMIT:
        raise ToolError(f"question は {QUESTION_LIMIT} 文字までにしてください（{len(text)} 文字）")
    raw = arguments.get("choices")
    low, high = CHOICES_RANGE
    if not isinstance(raw, list) or not low <= len(raw) <= high:
        raise ToolError(f"choices（選択肢）は {low}〜{high} 個にしてください")
    choices: list[Choice] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ToolError("choices の各項目は label と description を持つ辞書にしてください")
        label = _text_argument(entry, "label", "choices の label")
        description = _text_argument(entry, "description", "choices の description")
        if not label:
            raise ToolError("choices の label（選択肢の名前）が空です")
        if len(label) > LABEL_LIMIT:
            raise ToolError(
                f"label は {LABEL_LIMIT} 文字までにしてください 違いは description に書きます"
                f"（{label}）"
            )
        if len(description) > DESCRIPTION_LIMIT:
            raise ToolError(f"description は {DESCRIPTION_LIMIT} 文字までにしてください（{label}）")
        choices.append(Choice(label, description))
    labels = [choice.label for choice in choices]
    if len(set(labels)) != len(labels):
        # 同じ名前が並ぶと、どちらを選んだのか答えから見分けられない
        raise ToolError("choices の label が重なっています 選択肢ごとに違う名前にしてください")
    return QuestionSpec(
        text=text,
        choices=tuple(choices),
        multiple=_flag_argument(arguments, "multiple", False),
        free_text=_flag_argument(arguments, "free_text", True),
    )


def _text_argument(arguments: dict[str, Any], key: str, name: str) -> str:
    """文字列の引数 省いたら空 文字列でなければ断る"""
    value = arguments.get(key, "")
    if not isinstance(value, str):
        raise ToolError(f"{name} は文字列で渡してください（{type(value).__name__} が来ました）")
    return value.strip()


def _flag_argument(arguments: dict[str, Any], key: str, default: bool) -> bool:
    """真偽値の引数 省いたら ``default`` true / false 以外（"false" などの文字列）は断る"""
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ToolError(f"{key} は true か false で渡してください（{value!r} が来ました）")
    return value


def _ask_user(host: EditorHost, arguments: dict[str, Any]) -> object:
    del host  # 編集画面には触らない 答えを待つのはブリッジ
    return question_spec(arguments)


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


#: 1 回の返事の長さの上限（JSON の文字数） 配布版の Claude Code（SDK が同梱する claude.exe）は
#: MCP の道具の返事を 25000 トークン（``MAX_MCP_OUTPUT_TOKENS`` の既定）で打ち切り、残りを
#: ファイルへ退ける アシスタントにはファイルを読む道具を渡していないので、退けた分は読めない
#: （字幕 328 行の get_subtitles がそうなり、誤植を直せなかった 利用者の画面） 日本語は
#: 1 文字が 1〜2 トークンになるので、字数で 12000 に収めれば 25000 トークンの半分ほどで済む
MAX_RESULT_CHARS = 12000
#: 一覧の既定の件数と上限 上限まで頼まれても :data:`MAX_RESULT_CHARS` で切る
DEFAULT_PAGE = 50
MAX_PAGE = 300


def _paged(
    rows: list[Any],
    arguments: dict[str, Any],
    *,
    name: str = "items",
    truncated_note: str = "",
) -> dict[str, Any]:
    """一覧を ``offset`` と ``limit`` で切り出す 返事が長すぎれば件数を減らす

    1 件だけでも長すぎるときは中身を切り詰めて ``truncated`` を付け、``truncated_note``
    （全文の取り方）を添える 返事はいつも :data:`MAX_RESULT_CHARS` に収まる

    続きがあれば ``next_offset`` と、次に呼ぶときの書き方を添える 無いときは添えない
    （添えると、もう読み終えたのに続きを探しに行く）
    """
    total = len(rows)
    try:
        offset = max(0, int(arguments.get("offset", 0) or 0))
        limit = int(arguments.get("limit", DEFAULT_PAGE) or DEFAULT_PAGE)
    except (TypeError, ValueError) as exc:
        raise ToolError("offset と limit は整数で渡してください") from exc
    limit = min(max(limit, 1), MAX_PAGE)
    chosen = rows[offset : offset + limit]
    while len(chosen) > 1 and len(json_text(chosen)) > MAX_RESULT_CHARS - _ENVELOPE_CHARS:
        # 1 件ずつ減らすと長い一覧で遅い 4 分の 3 ずつ減らしてから残りを詰める
        chosen = chosen[: max(1, len(chosen) * 3 // 4)]
    if len(chosen) == 1 and len(json_text(chosen)) > MAX_RESULT_CHARS - _ENVELOPE_CHARS:
        # 1 件だけで上限を越える（長い字幕 1 行など） 件数ではもう減らせないので中身を
        # 切り詰めて印を付ける 切らずに返すと打ち切られ、退けた分は読めない（PR #231 の指摘）
        chosen = [_fit(chosen[0], MAX_RESULT_CHARS - _ENVELOPE_CHARS)]
    result = _page(chosen, name, total, offset, truncated_note)
    # 行だけで見積もると、外側（件数・続き・案内）の分で上限を越えうる（PR #231 の指摘）
    # 組み上がった返事の長さで確かめ、越えれば件数を減らすか 1 件を詰め直す
    budget = MAX_RESULT_CHARS - _ENVELOPE_CHARS
    while len(json_text(result)) > MAX_RESULT_CHARS:
        if len(chosen) > 1:
            chosen = chosen[: max(1, min(len(chosen) - 1, len(chosen) * 3 // 4))]
        else:
            budget //= 2
            if budget < _KEEP_CHARS:
                break
            chosen = [_fit(chosen[0], budget)]
        result = _page(chosen, name, total, offset, truncated_note)
    return result


def _page(
    chosen: list[Any], name: str, total: int, offset: int, truncated_note: str
) -> dict[str, Any]:
    """切り出した行に、件数・続き・案内を添えた返事"""
    notes: list[str] = []
    if any(isinstance(row, dict) and row.get("truncated") for row in chosen):
        # 呼ぶ側が先に切り詰めた行（長い字幕の本文）も、全文の読み方を添える
        notes.append(truncated_note or "長すぎる件は中身を切り詰めた（truncated）")
    end = offset + len(chosen)
    result: dict[str, Any] = {name: chosen, "total": total, "offset": offset, "count": len(chosen)}
    if end < total:
        result["next_offset"] = end
        notes.append(f"続きがあります 次は offset={end} で呼んでください（全 {total} 件）")
    if notes:
        result["note"] = " ".join(notes)
    return result


#: 一覧の外側（件数・続きの案内）に取っておく字数
_ENVELOPE_CHARS = 600
#: 切り詰めても残す文字列の長さ これより短い文字列は切らない（ID や名前を壊さない）
_KEEP_CHARS = 80


def _fit(row: Any, limit: int) -> Any:
    """1 件を ``limit`` 字（JSON）に収める 一番長い文字列から半分に切り、足りなければ
    一番長い配列を半分にする 切った件には ``truncated`` を付ける"""
    fitted = json.loads(json_text(row))
    cut = False
    while len(json_text(fitted)) > limit:
        place = _longest(fitted, str, _KEEP_CHARS) or _longest(fitted, list, 1)
        if place is None:
            break
        holder, key, value = place
        if isinstance(value, str):
            holder[key] = value[: len(value) // 2] + "…"
        else:
            holder[key] = value[: len(value) // 2]
        cut = True
    if cut and isinstance(fitted, dict):
        fitted["truncated"] = True
    return fitted


def _longest(value: Any, kind: type, floor: int) -> tuple[Any, Any, Any] | None:
    """``value`` の中で ``kind``（文字列か配列）の一番長い物と、その入れ物と鍵"""
    best: tuple[Any, Any, Any] | None = None
    stack = [value]
    while stack:
        holder = stack.pop()
        items = holder.items() if isinstance(holder, dict) else enumerate(holder)
        for key, child in items:
            size = len(child) if isinstance(child, (str, list)) and isinstance(child, kind) else -1
            if size > floor and (best is None or size > len(best[2])):
                best = (holder, key, child)
            if isinstance(child, (dict, list)):
                stack.append(child)
    return best


def _paging() -> dict[str, Any]:
    """一覧の道具の引数 何件目から・いくつ"""
    return {
        "offset": _integer("何件目から（0 から 続きは返事の next_offset）"),
        "limit": _integer(f"いくつ（既定 {DEFAULT_PAGE} 返事が長ければ少なく返す）"),
    }


#: 一覧の道具の説明に足す文
PAGING_NOTE = (
    " 返事は長くなりすぎないよう切って返す 続きがあれば next_offset が付くので、"
    "offset にその値を渡して続きを読む 多いときは絞り込みの引数で先に絞る"
)


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
    project = _project(host)
    wanted = str(arguments.get("contains") or "")
    rows = [
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
        if not wanted or wanted in item.name
    ]
    return _paged(rows, arguments, name="media")


def _list_tracks(host: EditorHost, arguments: dict[str, Any]) -> object:
    rows = [
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
    return _paged(rows, arguments, name="tracks")


def _list_clips(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    wanted = str(arguments.get("track_id") or "")
    low, high = _frame_window(project, arguments)
    clips = []
    for track in project.timeline.tracks:
        if wanted and str(track.id) != wanted:
            continue
        for clip in track.clips:
            if clip.timeline_end <= low or (high is not None and clip.timeline_start >= high):
                continue
            media = project.find_media(clip.media_id) if clip.media_id is not None else None
            # 混合トラックのクリップは絵と音を 1 本で出すので、どちらを出すかも見せる
            # ほかの種類では読まない項目なので、出すと AI が効かない値を触りに行く
            mixed: dict[str, object] = (
                {"audio_stream": clip.audio_stream, "show_picture": clip.show_picture}
                if track.kind is TrackKind.MIXED
                else {}
            )
            # テキストの縁取りの層 ID が見えないと、層の値も層のエフェクトも指せない
            strokes = _strokes_of(clip)
            if strokes is not None:
                mixed["strokes"] = strokes
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
    return _paged(clips, arguments, name="clips")


def _frame_window(project: Project, arguments: dict[str, Any]) -> tuple[int, int | None]:
    """``from_seconds`` ``to_seconds``（タイムラインの秒）をフレームの範囲 ``[start, end)`` へ

    説明どおり「from より後に掛かり、to より前に始まる」物を残す 呼ぶ側は
    ``終わり <= start`` と ``始まり >= end`` を外す ``start`` は from の秒を含むフレーム
    （そこで終わる物は外れ、そこを越えて続く物は残る） ``end`` は to の秒以上で最初に
    始まるフレーム 前は to のフレームに 1 を足していて、30fps で 1 秒ちょうどを渡すと
    1 秒から始まる物まで返した（PR #231 の指摘） 秒は分数で掛けて、0.1 秒のような
    値が浮動小数の誤差で 1 フレームずれないようにする 省けば全体
    """
    fps = project.rate.fps
    try:
        low = arguments.get("from_seconds")
        high = arguments.get("to_seconds")
        start = math.floor(Fraction(str(float(low))) * fps) if low is not None else 0
        end = math.ceil(Fraction(str(float(high))) * fps) if high is not None else None
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolError("from_seconds と to_seconds は秒の数で渡してください") from exc
    return max(0, start), end


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
    """エフェクトの一覧 kind を渡したときだけパラメータまで出す

    全部のパラメータを並べると、読み込んだ AviUtl のスクリプトまで含めて返事が長すぎ、
    打ち切られる まず名前で探し、使うエフェクトだけ詳しく読む
    """
    del host
    wanted_kind = str(arguments.get("kind") or "")
    category = str(arguments.get("category") or "")
    contains = str(arguments.get("contains") or "")
    entries: list[tuple[str, str, str, tuple[Any, ...]]] = [
        (d.kind, d.label, d.category, d.parameters) for d in registry.all()
    ]
    entries += [(s.kind, s.label, "オブジェクト", s.parameters) for s in (TEXT, SHAPE, STROKE)]
    rows: list[dict[str, Any]] = []
    for kind, label, group, parameters in entries:
        if wanted_kind and kind != wanted_kind:
            continue
        if category and group != category:
            continue
        if contains and contains not in kind and contains not in label:
            continue
        row: dict[str, Any] = {"kind": kind, "label": label, "category": group}
        if wanted_kind:
            row["parameters"] = [_describe_spec(spec) for spec in parameters]
        else:
            row["parameter_count"] = len(parameters)
        rows.append(row)
    return _paged(rows, arguments, name="effects")


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
    """タイムラインに出る字幕 絞り込み（素材・音声・時刻・言葉）と、続きを辿る切り出し

    ``compact`` なら行の番号・時刻・本文だけ（誤植を探す軽い形） 行の番号は絞り込んだ
    後の通し番号で、offset に使える
    """
    project = _project(host)
    wanted = str(arguments.get("media_id") or "")
    wanted_audio = arguments.get("audio")
    contains = str(arguments.get("contains") or "")
    compact = bool(arguments.get("compact", False))
    one = str(arguments.get("segment_id") or "")
    try:
        text_offset = max(0, int(arguments.get("text_offset", 0) or 0))
    except (TypeError, ValueError) as exc:
        raise ToolError("text_offset は整数で渡してください") from exc
    low, high = _frame_window(project, arguments)
    rows = []
    for subtitle in project_timeline(project):
        if one and str(subtitle.segment.id) != one:
            continue
        if subtitle.end_frame <= low or (high is not None and subtitle.start_frame >= high):
            continue
        if contains and contains not in subtitle.segment.text:
            continue
        # 素材と音声は字幕の出どころから取る 置いたクリップからたどると、置いたシーンの
        # 中の字幕は素材を持たないシーンのクリップに当たり、どの素材の字幕か分からない
        media_id = subtitle.media_id
        if wanted and str(media_id) != wanted:
            continue
        media = project.find_media(media_id) if media_id is not None else None
        audio = _audio_number(media, subtitle.stream)
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
    for row in rows:
        _clip_text(row, text_offset if one else None)
    if compact:
        rows = [
            {"n": index, "time": row["start_timecode"], "text": row["text"]}
            | ({"segment_id": row["segment_id"], "truncated": True} if "truncated" in row else {})
            for index, row in enumerate(rows)
        ]
    return _paged(rows, arguments, name="subtitles", truncated_note=_FULL_TEXT_NOTE)


#: 一覧で 1 行の本文を切り詰める長さ 1 行が長いと、ほかの行が 1 回の返事に入らない
LONG_TEXT_CHARS = 2000
#: segment_id で 1 行を読むときに 1 回で返す本文の長さ 返事の上限（12000）に収める
TEXT_CHUNK_CHARS = 8000
_FULL_TEXT_NOTE = (
    "長い本文は切り詰めた（truncated） 全文は get_subtitles に segment_id を渡して読む"
    "（それでも長ければ next_text_offset を text_offset に渡して続きを読む）"
)


def _clip_text(row: dict[str, Any], text_offset: int | None) -> None:
    """字幕 1 行の本文を返事に収まる長さにする

    ``text_offset`` が ``None``（一覧）なら長い本文の頭だけを残して ``truncated`` を付ける
    数（segment_id で 1 行を読む）なら、そこから :data:`TEXT_CHUNK_CHARS` 字を返し、続きが
    あれば ``next_text_offset`` を添える
    """
    text = str(row["text"])
    if text_offset is None:
        if len(text) > LONG_TEXT_CHARS:
            row["text"] = text[:LONG_TEXT_CHARS] + "…"
            row["truncated"] = True
            row["text_length"] = len(text)
        return
    if text_offset == 0 and len(text) <= TEXT_CHUNK_CHARS:
        return
    row["text"] = text[text_offset : text_offset + TEXT_CHUNK_CHARS]
    row["text_offset"] = text_offset
    row["text_length"] = len(text)
    if text_offset + TEXT_CHUNK_CHARS < len(text):
        row["next_text_offset"] = text_offset + TEXT_CHUNK_CHARS


def _replace_subtitle_text(host: EditorHost, arguments: dict[str, Any]) -> object:
    """字幕の本文の「誤 → 正」をまとめて置き換える 字幕と、字幕から置いたテキストの両方

    字幕から置いたテキスト（焼き込み）は、出どころの印（:class:`SubtitleOrigin`）が直した
    字幕の行を指すクリップだけ 字幕だけを直すと、焼き込んだ文字に誤植が残る 本文の一致で
    探すと、手で書いたタイトルがたまたま同じ本文だと書き換えてしまう 印の無い前の版の
    焼き込みは直さず、数を返事に添える 1 回の取り消しで全部戻る
    """
    pairs_raw = arguments.get("pairs") or []
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise ToolError('pairs に [{"from": 誤, "to": 正}, …] を渡してください')
    pairs: list[tuple[str, str]] = []
    for entry in pairs_raw:
        if not isinstance(entry, dict) or not str(entry.get("from") or ""):
            raise ToolError('pairs の各組は {"from": 誤, "to": 正} です（from は空にできません）')
        pairs.append((str(entry["from"]), str(entry.get("to") or "")))

    project = _project(host)
    targets: list[tuple[MediaItem, int]] = []
    if arguments.get("media_id"):
        media = _require_media(project, str(arguments["media_id"]))
        streams = (
            [media.transcript_stream(_audio_stream(media, arguments))]
            if arguments.get("audio") is not None
            else [index for index, _ in media.transcripts]
        )
        targets = [(media, stream) for stream in streams]
    else:
        targets = [(item, index) for item in project.media for index, _ in item.transcripts]

    counts = [0] * len(pairs)

    def fixed(text: str, *, counted: bool = True) -> str:
        for number, (before, after) in enumerate(pairs):
            found = text.count(before)
            if found:
                if counted:
                    counts[number] += found
                text = text.replace(before, after)
        return text

    commands: list[Command] = []
    # 直した字幕の行 ID から、その字幕の素材と音声へ
    renamed: dict[SegmentId, tuple[MediaId, int]] = {}
    old_texts: set[str] = set()
    for media, stream in targets:
        transcript = media.transcript_for(stream)
        if transcript is None:
            continue
        segments = []
        changed = False
        for segment in transcript.segments:
            text = fixed(segment.text)
            if text != segment.text:
                renamed[segment.id] = (media.id, stream)
                old_texts.add(segment.text.strip())
                segment = segment.with_text(text)
                changed = True
            segments.append(segment)
        if changed:
            updated = replace(transcript, segments=tuple(segments))
            commands.append(SetTranscript(media.id, updated, stream=stream))
    # 焼き込んだテキスト 印が直した字幕の行を指すクリップの本文へ、同じ置き換えを掛ける
    # 焼き込んだ後に手で足した言葉も残る（字幕の新しい本文で丸ごと上書きしない）
    burned = 0
    unmarked = 0
    for track in project.timeline.tracks:
        for clip in track.clips:
            if clip.source is None or clip.source.kind != "text":
                continue
            text = str(clip.source.params.get("text", ""))
            origin = clip.subtitle_origin
            if origin is None:
                # 印の無い前の版の焼き込みかもしれない 手で書いた物と見分けられないので直さない
                if text.strip() in old_texts:
                    unmarked += 1
                continue
            owner = renamed.get(origin.segment_id)
            if owner is None:
                continue
            if origin.media_id is not None and owner != (origin.media_id, origin.stream):
                continue
            new_text = fixed(text, counted=False)
            if new_text != text:
                commands.append(SetParam(ParamPath.of_source(clip.id, "text"), new_text))
                burned += 1
    if commands:
        host.apply_commands(commands, "字幕の誤植を直す")
    return {
        "replaced": sum(counts),
        "per_pair": [
            {"from": before, "to": after, "count": count}
            for (before, after), count in zip(pairs, counts, strict=True)
        ],
        "unmatched": [
            before for (before, _), count in zip(pairs, counts, strict=True) if not count
        ],
        "burned_text_clips": burned,
        # 本文は直した字幕と同じだが、出どころの印が無いので直さなかったテキスト
        # 印を付ける前の版で焼き込んだ物か、手で書いた物 要るなら set_param で直す
        "unmarked_text_clips": unmarked,
    }


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


def _get_app_settings(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    # 写しを返す ホストの持ち物をそのまま渡すと、後の道具の中で書き換えたときに
    # 本人の設定まで変わる（アプリ自身の設定は人だけが変える #350）
    return copy.deepcopy(host.app_settings())


def _get_history(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    document = host.document
    return {
        "can_undo": document.can_undo,
        "undo_label": document.undo_label,
        "recent": list(document.history_labels[-10:]),
    }


# --- 場面の切り替わり（#319 #321）---

#: 感度（点数のしきい値）として受け付ける範囲 下の端より小さいと、手持ちの揺れや
#: 圧縮のノイズまで切り替わりに数え、一覧が素材のフレーム数に近づく
SCENE_THRESHOLD_RANGE = (0.02, 1.0)
#: 最短の場面（秒）の上限 長くすると、1 つの場面の中の切り替わりを拾えなくなる
MAX_MIN_SCENE_SECONDS = 600.0
#: 解析が終わっていないときに待つ秒の既定と上限 待つのは AI のスレッドで、画面は止めない
#: 上限を置くのは、長い素材で会話が何分も止まったように見えないため 待ちきれなければ
#: 読んだ所までを返し、続きはもう 1 度呼んで受け取る
DEFAULT_SCENE_WAIT = 30.0
MAX_SCENE_WAIT = 120.0
#: 1 つの切り替わりについて返すタイムラインの位置の数 同じ素材を何十回も置いた
#: プロジェクトで、1 行が長くなりすぎないため
MAX_PLACES_PER_CUT = 5


@dataclass(frozen=True, slots=True)
class Pending:
    """まだ用意できていない物を待ってから返す道具の戻り値（#321）

    道具（``handler``）は画面のスレッドで動くので、そこで待つと待つ間は画面が固まる
    ブリッジ（:meth:`~sashimono.ai.bridge.EditorBridge.invoke`）が AI のスレッドで
    ``wait`` を呼んで最長 ``seconds`` 秒待ち、終わったら（か待ちきれなければ）
    ``then`` を画面のスレッドで呼んで、その戻り値を道具の答えにする
    """

    #: 用意できるまで最長で引数の秒だけ待ち、できていれば真 どのスレッドから呼んでもよい
    wait: Callable[[float], bool]
    seconds: float
    then: Callable[[], object]


@dataclass(frozen=True, slots=True)
class _SceneQuery:
    threshold: float
    min_scene: float
    start: float | None
    end: float | None
    wait: float


def _optional_number(
    arguments: dict[str, Any], key: str, low: float, high: float = math.inf
) -> float | None:
    """数の引数 省けば ``None`` 範囲の外・数でない物は、受け付ける範囲を書いて断る

    丸めて受け取らない 0.001 の感度を黙って 0.02 にすると、AI は細かく拾ったつもりで
    少ない一覧を読む 文字列の数も断る（ほかの道具の真偽値と同じく、型を緩く読まない）
    """
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolError(f"{key} は数で渡してください（{value!r} が来ました）")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ToolError(f"{key} は {low:g}〜{high:g} で渡してください（{value!r} が来ました）")
    return number


def _number_or(
    arguments: dict[str, Any], key: str, default: float, low: float, high: float
) -> float:
    """省けば ``default`` の数の引数（:func:`_optional_number`）"""
    number = _optional_number(arguments, key, low, high)
    return default if number is None else number


def _scene_query(arguments: dict[str, Any]) -> _SceneQuery:
    threshold = _number_or(arguments, "threshold", DEFAULT_THRESHOLD, *SCENE_THRESHOLD_RANGE)
    min_scene = _number_or(
        arguments, "min_scene_seconds", DEFAULT_MIN_SCENE, 0.0, MAX_MIN_SCENE_SECONDS
    )
    start = _optional_number(arguments, "from_seconds", 0.0)
    end = _optional_number(arguments, "to_seconds", 0.0)
    wait = _number_or(arguments, "wait_seconds", DEFAULT_SCENE_WAIT, 0.0, MAX_SCENE_WAIT)
    if start is not None and end is not None and end <= start:
        raise ToolError("to_seconds は from_seconds より後にしてください")
    return _SceneQuery(threshold, min_scene, start, end, wait)


def _detect_scenes(host: EditorHost, arguments: dict[str, Any]) -> object:
    """素材の場面の切り替わりを一覧する 解析が終わっていなければ待ってから返す"""
    project = _project(host)
    media = _require_media(project, str(arguments.get("media_id", "")))
    if not media.has_video:
        raise ToolError(f"{media.name} には映像がありません（場面は映像の切り替わりから探します）")
    query = _scene_query(arguments)
    # 頼むのは呼ばれた最初だけ 失敗した素材も、もう 1 度呼ばれたら頼み直す（直した素材のため）
    host.request_scenes(media)
    state = host.scene_state(media)
    if state.complete or not state.running or query.wait <= 0:
        return _scene_result(host, media, query, state, arguments)
    media_id = media.id

    def finish() -> object:
        # 待っている間に素材が外されたら、外したことを伝える 前の素材のまま答えると、
        # 一覧に無い素材の場面を AI が切りに行く
        again = _require_media(_project(host), str(media_id))
        return _scene_result(host, again, query, host.scene_state(again), arguments)

    return Pending(wait=host.scene_waiter(media_id), seconds=query.wait, then=finish)


def _scene_result(
    host: EditorHost,
    media: MediaItem,
    query: _SceneQuery,
    state: SceneState,
    arguments: dict[str, Any],
) -> object:
    if state.signal is None and state.failure is not None:
        raise ToolError(f"{media.name} の場面を調べられませんでした: {state.failure}")
    signal = state.signal if state.signal is not None else SceneSignal.empty(complete=False)
    project = _project(host)
    stream = _video_stream(media, signal.stream)
    cuts = signal.cuts(query.threshold, query.min_scene, start=query.start, end=query.end)
    rows = [
        {
            "source_seconds": round(cut.seconds, 3),
            # 切り替わりはこの時刻と source_seconds の間にある preview_frame で前後を見る目安
            "after_seconds": round(cut.before, 3),
            "score": round(cut.score, 3),
            "timeline": _scene_places(project, media, stream, cut.seconds),
        }
        for cut in cuts
    ]
    result = _paged(rows, arguments, name="cuts")
    notes = [str(result.pop("note"))] if "note" in result else []
    duration = float(media.duration)
    result |= {
        "media_id": str(media.id),
        "complete": signal.complete,
        "analyzed_seconds": round(min(signal.analyzed, duration), 3),
        "duration_seconds": round(duration, 3),
        "threshold": query.threshold,
        "min_scene_seconds": query.min_scene,
        # 調べた映像の番号（1 から） 映像が何本もある素材は先頭だけを調べ、timeline には
        # その映像を出すクリップだけを載せる
        "video_stream": _video_number(media, stream),
    }
    if signal.broken:
        # 呼び直しても同じ所で止まる 「待てば続きが出る」と読ませると、AI が呼び直し続ける
        notes.append(
            f"素材が途中で壊れていて {round(signal.analyzed, 1)} 秒までしか読めませんでした"
            " 読めた所までの切り替わりを返しました 呼び直しても続きは出ません"
        )
    elif not signal.complete:
        if state.running:
            notes.append(
                f"解析の途中です（{round(signal.analyzed, 1)} 秒まで）"
                " 読んだ所までの切り替わりを返しました"
                " しばらくしてから同じ引数で呼ぶと続きを返します"
            )
        else:
            notes.append("解析が止まりました もう 1 度呼ぶと解析を頼み直します")
    elif not rows:
        notes.append("切り替わりが見つかりません threshold を下げると細かい変化も拾います")
    if notes:
        result["note"] = " ".join(notes)
    return result


def _video_stream(media: MediaItem, wanted: int | None) -> int | None:
    """クリップが絵を取る映像ストリームの番号（素材の中の ``index``）

    デコーダ（:class:`~sashimono.engine.decode.VideoDecoder`）と同じ決まりにする 素材に無い
    番号（音声の番号や、前の版の既定の 0）は先頭の映像になる 決まりがずれると、実際に
    出ている映像と違う映像の切り替わりをクリップへ当てはめる
    """
    streams = media.video_streams
    if not streams:
        return None
    if wanted is not None and any(stream.index == wanted for stream in streams):
        return wanted
    return streams[0].index


def _video_number(media: MediaItem, stream: int | None) -> int | None:
    """映像の番号（1 から） 音声の番号（``audio``）と同じ数え方"""
    for number, info in enumerate(media.video_streams, start=1):
        if info.index == stream:
            return number
    return None


def _scene_places(
    project: Project, media: MediaItem, stream: int | None, seconds: float
) -> list[dict[str, Any]]:
    """素材の ``seconds`` の絵が出る、今のタイムラインの位置（フレーム）

    その位置をそのまま preview_frame へ渡せるように返す 絵を描くクリップだけを見る
    （音だけのクリップの位置では、その場面は見えない） 調べた映像（``stream``）とは
    別の映像を出すクリップも外す（PR #358 の Codex の指摘 その絵には同じ切り替わりが無い）
    1 枚目を CEIL で取るのは、新しい場面の最初の絵が出るフレームにするため
    """
    moment = Fraction(str(seconds))
    places: list[dict[str, Any]] = []
    for track in project.timeline.tracks:
        for clip in track.clips:
            if clip.media_id != media.id or not project.draws_picture(track, clip):
                continue
            if _video_stream(media, clip.stream_index) != stream:
                continue
            if moment < clip.source_in or (clip.hold_at is not None and moment > clip.hold_at):
                continue
            offset = seconds_to_frame(
                (moment - clip.source_in) / clip.speed, project.rate, Rounding.CEIL
            )
            frame = clip.timeline_start + offset
            if frame >= clip.timeline_end:
                continue
            places.append({"clip_id": str(clip.id), "frame": frame})
    return sorted(places, key=lambda place: int(place["frame"]))[:MAX_PLACES_PER_CUT]


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
    project = _project(host)
    if not trimmable(project, (clip.id,)):
        # リンクの相手まで全部ロックしていると、TrimClip は全部を黙って飛ばす 命令を
        # 積むと、何も変わらないのに頼んだ量が動いたと答え、履歴に空の 1 手が残る
        return {
            "head_delta": 0,
            "tail_delta": 0,
            "requested_head_delta": head,
            "requested_tail_delta": tail,
            "stopped_at": STOP_LOCKED,
            "note": (
                f"{STOP_LOCKED}のクリップなので端を動かさなかった ロックを外してから頼んでください"
            ),
        }
    result: dict[str, object] = {}
    stops: list[str] = []
    notes: list[str] = []
    if head < 0:
        # 端を伸ばしすぎた頼みは断らず、伸ばせる所まで伸ばす（利用者が決めた）
        # 断ると、AI は量を変えて何度も試し直し、そのたびに会話が長くなる 止めた理由と
        # 足りなかった量を返すので、AI は前のクリップを動かすなど次の手を選べる
        room = head_room(project, (clip.id,))
        if -head > room.frames:
            result["requested_head_delta"] = head
            stops.append(room.reason)
            notes.append(
                f"{room.reason}で止めた 頼まれた {-head} フレームのうち "
                f"{room.frames} フレームだけ前へ伸ばした"
            )
            head = -room.frames
    if tail > 0:
        # 後ろも頭と同じく、後ろのクリップの頭で止める
        ahead = tail_room(project, (clip.id,))
        if ahead is not None and tail > ahead.frames:
            result["requested_tail_delta"] = tail
            stops.append(ahead.reason)
            notes.append(
                f"{ahead.reason}で止めた 頼まれた {tail} フレームのうち "
                f"{ahead.frames} フレームだけ後ろへ伸ばした"
            )
            tail = ahead.frames
    if stops:
        result["stopped_at"] = "、".join(stops)
        result["note"] = " ".join(notes)
    if head or tail:
        host.apply_commands(
            [TrimClip(clip.id, head_delta=head, tail_delta=tail)], "クリップをトリム"
        )
    return {"head_delta": head, "tail_delta": tail, **result}


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
    stroke_id = str(arguments.get("stroke_id") or "")
    if stroke_id:
        # 縁取りの層に掛ける（#273） 掛けられる種類は設定パネルのメニューと同じ
        _require_stroke(clip, stroke_id)
        if not takes_stroke_effect(kind):
            allowed = "、".join(STROKE_EFFECT_KINDS)
            raise ToolError(f"{kind} は縁取りの層に掛けられません（使えるのは {allowed}）")
        host.apply_commands(
            [AddEffect(clip.id, effect, stroke_id=StrokeId(stroke_id))],
            f"エフェクトを縁取りに追加: {definition.label}",
        )
        return {"effect_id": str(effect.id), "kind": kind, "stroke_id": stroke_id}
    host.apply_commands([AddEffect(clip.id, effect)], f"エフェクトを追加: {definition.label}")
    return {"effect_id": str(effect.id), "kind": kind}


def _require_stroke(clip: Clip, stroke_id: str) -> Stroke:
    found = clip.source.find_stroke(StrokeId(stroke_id)) if clip.source is not None else None
    if found is None:
        raise ToolError(
            f"縁取りの層が見つかりません: {stroke_id}（list_clips の strokes で ID が分かる）"
        )
    return found


def _add_stroke(host: EditorHost, arguments: dict[str, Any]) -> object:
    """テキストに縁取りの層を足す 省いた項目は今の縁の外側に見える太さと色にする"""
    _, clip = _target_clip(host, arguments)
    if clip.source is None or clip.source.kind != "text":
        raise ToolError("縁取りの層を足せるのはテキストのクリップだけです")
    stroke = next_stroke(clip.source)
    params = dict(stroke.params)
    for name in ("width", "color", "position", "opacity", "join"):
        if name not in arguments:
            continue
        spec = STROKE.spec(name)
        assert spec is not None  # 名前は定義の項目
        value = arguments[name]
        if isinstance(value, list):
            # 色を数の並びで渡されたとき 並びのままでは色の欄が受け取らず、既定の黒になる
            value = tuple(float(entry) for entry in value)
        parsed = _parse_color(value) if isinstance(value, str) and name == "color" else None
        params[name] = spec.coerce(parsed if parsed is not None else value)
    stroke = replace(stroke, params=params)
    index = arguments.get("index")
    try:
        host.apply_commands(
            [AddStroke(clip.id, stroke, int(index) if index is not None else None)],
            "縁取りを追加",
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {"stroke_id": str(stroke.id)}


def _remove_stroke(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    stroke = _require_stroke(clip, str(arguments.get("stroke_id") or ""))
    host.apply_commands([RemoveStroke(clip.id, stroke.id)], "縁取りを削除")
    return {"removed": str(stroke.id)}


def _move_stroke(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    stroke = _require_stroke(clip, str(arguments.get("stroke_id") or ""))
    try:
        index = int(arguments["index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError("index に整数が要ります（0 が一番上）") from exc
    host.apply_commands([MoveStroke(clip.id, stroke.id, index)], "縁取りの順番を変更")
    return {"stroke_id": str(stroke.id), "index": index}


def _set_stroke_enabled(host: EditorHost, arguments: dict[str, Any]) -> object:
    _, clip = _target_clip(host, arguments)
    stroke = _require_stroke(clip, str(arguments.get("stroke_id") or ""))
    enabled = bool(arguments.get("enabled", True))
    host.apply_commands([SetStrokeEnabled(clip.id, stroke.id, enabled)], "縁取りの入り切り")
    return {"stroke_id": str(stroke.id), "enabled": enabled}


def _strokes_of(clip: Clip) -> list[dict[str, Any]] | None:
    """テキストの縁取りの層の一覧 テキスト以外は ``None``（返事に出さない）

    層を持たない字は、前からの縁（``border_width``）を描く そのときは空の一覧を返し、
    層を足すと前からの縁が 1 つ目の層へ移ることを道具の説明に書いておく
    """
    if clip.source is None or clip.source.kind != "text":
        return None
    rows: list[dict[str, Any]] = []
    for stroke in clip.source.strokes:
        width = stroke.params.get("width")
        rows.append(
            {
                "stroke_id": str(stroke.id),
                "enabled": stroke.enabled,
                "width": width.static if isinstance(width, AnimatedValue) else width,
                "position": stroke.params.get("position", "outside"),
                "effects": [
                    {"effect_id": str(e.id), "kind": e.kind, "enabled": e.enabled}
                    for e in stroke.effects
                ],
            }
        )
    return rows


def _param_path(clip_id: ClipId, arguments: dict[str, Any]) -> ParamPath:
    name = str(arguments.get("name", ""))
    if not name:
        raise ToolError("name が空です")
    effect_id = str(arguments.get("effect_id") or "")
    stroke_id = str(arguments.get("stroke_id") or "")
    if stroke_id:
        # テキストの縁取りの層の項目か、層に掛けたエフェクトの項目
        if effect_id:
            return ParamPath.of_stroke_effect(
                clip_id, StrokeId(stroke_id), EffectId(effect_id), name
            )
        return ParamPath.of_stroke(clip_id, StrokeId(stroke_id), name)
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
    stroke = (
        clip.source.find_stroke(path.stroke_id)
        if path.stroke_id is not None and clip.source is not None
        else None
    )

    if path.target is ParamTarget.EFFECT and path.effect_id is not None:
        stack = stroke.effects if stroke is not None else clip.effects
        effect = next((e for e in stack if e.id == path.effect_id), None)
        definition = registry.get(effect.kind) if effect is not None else None
        return definition.spec(path.name) if definition is not None else None

    if path.target is ParamTarget.SOURCE and stroke is not None:
        return STROKE.spec(path.name)

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

    # 画面の〔焼き込み〕と同じ既定 作品の高さに合わせて縮める
    settings = project.settings
    wrap = subtitle_wrap_width(settings.width, host.subtitle_wrap_share)
    template: GeneratedSource | Clip = TEXT.create(**burn_defaults(settings.height, wrap))
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

    # 省いた音（None）は 1 本目 そのまま渡すと plan_cuts は音で絞らず、別の位置へ置いた
    # 音声 2 のクリップでも音声 1 の無音を切る（PR #231 の指摘）
    ranges = plan_cuts(project, media.id, silences, stream=media.transcript_stream(stream))
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


# --- 保存と書き出し（取り消せない操作 B1 #327） ---
#
# どれも「元に戻す」では戻らない 2026-10-14 の利用者の決定（#298 #350）のとおり、
# 「変更を自動で承認」でも毎回確かめ（Operation.irreversible）、書く場所は作品のフォルダと
# 書き出しの窓で前に選んだフォルダだけ（writable_path） 書くのは確かめの後、フォルダを
# 開いたまま持って（pinned_place）から 道具はファイルを自分で書かず、画面の〔保存〕と
# 書き出しの窓と同じ処理をホストに頼む


def _require_suffix(raw: object, suffixes: tuple[str, ...], what: str) -> None:
    """拡張子を確かめる 違う種類のファイルの名前で書かせない

    作品の保存で動画の名前を、書き出しで作品（.sme）の名前を渡されると、作品や素材を
    別の中身で上書きする 拡張子で書く物の種類を決め、合わなければ断る
    """
    suffix = Path(str(raw).strip()).suffix.lower() if isinstance(raw, str) else ""
    if suffix not in suffixes:
        allowed = " / ".join(suffixes)
        raise ToolError(f"{what}の名前は {allowed} で終えてください: {raw}")


def _current_project_file(host: EditorHost) -> Path:
    current = host.project_path
    if current is None:
        raise ToolError(
            "作品はまだ一度も保存されていません save_project_as で保存先（作品のフォルダか"
            "書き出しのフォルダの中）を渡すか、人に〔名前を付けて保存〕で選んでもらってください"
        )
    return current


def _replacing(path: Path, what: str) -> str:
    """書く先に同じ名前の物があれば、置き換えることと、その大きさと更新時刻を言う

    大きさと時刻も見せるのは、許した後に中身が変わったら、実行の前の照らし合わせ
    （:meth:`~sashimono.ai.bridge.EditorBridge._run_irreversible`）で気付けるようにするため
    見せた物だけを比べるので、見せない身元（番号など）はここに入れない
    """
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        return ""
    when = datetime.fromtimestamp(status.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"同じ名前の{what}（{status.st_size:,} バイト {when} 更新）があり、中身を置き換えます"


def _save_consent(host: EditorHost, arguments: dict[str, Any]) -> Consent:
    del arguments
    path = writable_path(host, str(_current_project_file(host)))
    return Consent(
        action="作品を上書き保存します",
        target=str(path),
        # 今のファイルの大きさと時刻も見せる 箱が出ている間に別の窓が同じ作品を保存すると
        # 変わるので、実行の前の照らし合わせで気付き、その保存を潰さずに断れる
        detail=_replacing(path, "作品") or "今の場所に作品のファイルが無いので、新しく作ります",
        basis=expect_now(path),
    )


def _basis(consent: Consent) -> Expected:
    """人が許したときの書く先の身元 書く処理はこれで照らす（Codex の P2）

    書く処理の中で採り直すと、実行の前の照らし合わせ（確かめの中身を作り直して比べる）の後に
    書き換えられた中身を、新しい基準として受け入れてしまう
    """
    if consent.basis is None:
        raise ToolError("許したときの書く先の様子が分からないので、書きません")
    return consent.basis


def _consented(
    approved: Callable[[EditorHost, dict[str, Any], Consent], object],
    describe: Callable[[EditorHost, dict[str, Any]], Consent],
) -> Callable[[EditorHost, dict[str, Any]], object]:
    """確かめを通らずに直に呼ばれたとき（試験など）の入口 その場で確かめの中身を作って渡す

    ブリッジは :attr:`Operation.approved` に、人が許した確かめの中身を渡して呼ぶ
    """

    def handler(host: EditorHost, arguments: dict[str, Any]) -> object:
        return approved(host, arguments, describe(host, arguments))

    handler.__name__ = approved.__name__
    return handler


def _save_project(host: EditorHost, arguments: dict[str, Any], consent: Consent) -> object:
    del arguments
    try:
        with pinned_place(host, str(_current_project_file(host))) as path:
            note = host.write_project(path, adopt=False, expect=_basis(consent))
    except OSError as exc:
        raise ToolError(f"保存できませんでした: {exc}") from exc
    return {"saved": str(path), "note": note}


def _save_as_consent(host: EditorHost, arguments: dict[str, Any]) -> Consent:
    raw = arguments.get("path")
    _require_suffix(raw, (SUFFIX,), "作品")
    path = writable_path(host, raw)
    detail = " ".join(
        part
        for part in (_replacing(path, "ファイル"), "以後はこの場所の作品として編集します")
        if part
    )
    return Consent(
        action="作品に名前を付けて保存します",
        target=str(path),
        detail=detail,
        basis=expect_now(path),
    )


def _save_project_as(host: EditorHost, arguments: dict[str, Any], consent: Consent) -> object:
    raw = arguments.get("path")
    _require_suffix(raw, (SUFFIX,), "作品")
    try:
        with pinned_place(host, raw) as path:
            note = host.write_project(path, adopt=True, expect=_basis(consent))
    except OSError as exc:
        raise ToolError(f"保存できませんでした: {exc}") from exc
    return {"saved": str(path), "project_path": str(host.project_path), "note": note}


def _subtitle_plan(
    host: EditorHost, arguments: dict[str, Any]
) -> tuple[Path, tuple[int, int] | None, str]:
    """字幕の書き出しの引数を、画面の字幕パネルの〔書き出し…〕と同じ選び方で確かめる

    相手は字幕パネルと同じく開いているタイムライン 範囲の既定もパネルと同じ（範囲を
    決めてあれば範囲、無ければ全体） 返すのは書く場所・範囲・確かめに出す中身
    """
    raw = arguments.get("path")
    _require_suffix(raw, SUBTITLE_SUFFIXES, "字幕のファイル")
    path = writable_path(host, raw)
    project = _project(host)
    default_range = RANGE_WORK_AREA if export_range(project.timeline) is not None else RANGE_ALL
    try:
        frame_range = chosen_range(project, str(arguments.get("range") or default_range))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    subtitles = [
        subtitle
        for subtitle in (
            subtitles_in_range(project_timeline(project), frame_range)
            if frame_range is not None
            else project_timeline(project)
        )
        if subtitle.segment.text.strip()
    ]
    if not subtitles:
        # 空のファイルを書くために取り消せない確かめを出しても、人が判断する物が無い
        raise ToolError(
            "書き出す字幕がありません（タイムラインに字幕のクリップが無いか、範囲の外）"
        )
    rate = project.rate
    if frame_range is None:
        span = "全体"
    else:
        start, end = frame_range
        span = (
            f"指定した範囲（{format_timecode(start, rate)} 〜 {format_timecode(end, rate)}）"
            " 範囲の頭を 0 秒にずらす"
        )
    parts = [
        f"形式 {path.suffix.lstrip('.')} / {span} / {len(subtitles)} 枚",
        _replacing(path, "ファイル"),
    ]
    return path, frame_range, "\n".join(part for part in parts if part)


def _subtitles_consent(host: EditorHost, arguments: dict[str, Any]) -> Consent:
    path, _range, detail = _subtitle_plan(host, arguments)
    return Consent(
        action="字幕を書き出します", target=str(path), detail=detail, basis=expect_now(path)
    )


def _export_subtitles(host: EditorHost, arguments: dict[str, Any], consent: Consent) -> object:
    _path, frame_range, _detail = _subtitle_plan(host, arguments)
    try:
        with pinned_place(host, arguments.get("path")) as path:
            written = host.write_subtitles(path, frame_range=frame_range, expect=_basis(consent))
    except OSError as exc:
        raise ToolError(f"字幕を書き出せませんでした: {exc}") from exc
    return {"written": str(written), "range": None if frame_range is None else list(frame_range)}


def _snapshot_plan(host: EditorHost, arguments: dict[str, Any]) -> tuple[Path, int, str]:
    """静止画の引数を確かめる 描くのは画面の〔静止画を保存…〕と同じく開いているタイムライン"""
    raw = arguments.get("path")
    _require_suffix(raw, (".png",), "静止画")
    path = writable_path(host, raw)
    project = _project(host)
    if project.duration <= 0:
        raise ToolError("タイムラインが空なので静止画にできません")
    raw_frame = arguments.get("frame", host.playhead)
    if isinstance(raw_frame, bool) or not isinstance(raw_frame, int):
        raise ToolError("frame はフレーム番号（整数）で渡してください 省くと再生ヘッドの位置")
    # 画面と同じく、終わり（最後のフレームの 1 つ後ろ）を指していれば最後のフレームを描く
    frame = max(0, min(raw_frame, project.duration - 1))
    width, height = project.settings.resolution
    parts = [
        f"{format_timecode(frame, project.rate)}（{frame} フレーム）を {width}x{height} の PNG で",
        _replacing(path, "ファイル"),
    ]
    return path, frame, "\n".join(part for part in parts if part)


def _snapshot_consent(host: EditorHost, arguments: dict[str, Any]) -> Consent:
    path, _frame, detail = _snapshot_plan(host, arguments)
    return Consent(
        action="静止画を保存します", target=str(path), detail=detail, basis=expect_now(path)
    )


def _save_snapshot(host: EditorHost, arguments: dict[str, Any], consent: Consent) -> object:
    _path, frame, _detail = _snapshot_plan(host, arguments)
    try:
        with pinned_place(host, arguments.get("path")) as path:
            drawn = host.write_snapshot(path, frame, expect=_basis(consent))
    except OSError as exc:
        raise ToolError(f"静止画を保存できませんでした: {exc}") from exc
    return {"saved": str(path), "frame": drawn}


#: 書き出しを待つ上限（秒） 長く待たせると、その間は AI が人の新しい指示に応えられない
#: 長い書き出しは、待たずに戻って export_status で何度か見てもらう
MAX_WAIT_SECONDS = 300


def _wait_seconds(arguments: dict[str, Any]) -> float:
    raw = arguments.get("wait_seconds", 0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ToolError("wait_seconds は秒の数で渡してください（0 なら待たずに戻る）")
    return max(0.0, min(float(raw), float(MAX_WAIT_SECONDS)))


@dataclass(frozen=True, slots=True)
class _ExportPlan:
    request: ExportRequest
    #: 確かめの窓に出す、何をどう書き出すか
    detail: str


def _export_plan(host: EditorHost, arguments: dict[str, Any]) -> _ExportPlan:
    """書き出しの引数を、画面の書き出しの窓と同じ選び方で確かめる

    確かめ（consent）と実行の 2 度呼ぶ 窓が出ている間に作品が変わっても（範囲を消した など）、
    実行の時点の作品で確かめ直す
    """
    raw = arguments.get("path")
    _require_suffix(raw, CONTAINER_SUFFIXES, "書き出す動画")
    # 待つ秒数も確かめの前に見る 確かめの後で断ると、人が許したのに書き出しが始まらない
    _wait_seconds(arguments)
    if host.export_status().get("state") in EXPORT_RUNNING:
        raise ToolError(
            "書き出しがもう走っています export_status で終わるのを待つか、"
            "cancel_export で止めてから"
            "始めてください"
        )
    path = writable_path(host, raw)
    whole = host.document.project
    raw_scene = arguments.get("scene_id")
    scene_id = SceneId(str(raw_scene)) if raw_scene else None
    try:
        target = export_target(whole, scene_id)
    except KeyError as exc:
        raise ToolError(
            f"シーンが見つかりません: {raw_scene}（list_scenes で一覧を取れます）"
        ) from exc
    if target.duration <= 0:
        raise ToolError("タイムラインが空なので書き出せません")
    # 範囲の既定は画面の書き出しの窓と同じ 範囲を決めてあればその範囲、無ければ全体
    default_range = RANGE_WORK_AREA if export_range(target.timeline) is not None else RANGE_ALL
    choice = str(arguments.get("range") or default_range)
    try:
        frame_range = chosen_range(target, choice)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    codecs = available_video_codecs()
    if not codecs:
        raise ToolError("この機械には使える映像コーデックがありません 書き出せません")
    raw_codec = str(arguments.get("codec") or AUTO)
    if raw_codec != AUTO and raw_codec not in codecs:
        raise ToolError(f"codec は {AUTO} か、使える物（{', '.join(codecs)}）から選んでください")
    codec = None if raw_codec == AUTO else raw_codec
    raw_bitrate = arguments.get("bitrate_mbps", DEFAULT_BITRATE_MBPS)
    low, high = BITRATE_RANGE_MBPS
    if isinstance(raw_bitrate, bool) or not isinstance(raw_bitrate, int):
        raise ToolError(f"bitrate_mbps は {low}〜{high} の整数で渡してください")
    if not low <= raw_bitrate <= high:
        raise ToolError(f"bitrate_mbps は {low}〜{high} です: {raw_bitrate}")

    start, end = frame_range or (0, target.duration)
    rate = target.rate
    scene = whole.find_scene(scene_id) if scene_id is not None else None
    what = "メイン" if scene is None else f"シーン「{scene.name}」"
    span = "全体" if frame_range is None else "指定した範囲"
    label = "自動" if codec is None else codec
    parts = [
        f"{what} の{span}（{format_timecode(start, rate)} 〜 {format_timecode(end, rate)}、"
        f"{end - start} フレーム）",
        f"形式 {path.suffix.lstrip('.')} / 映像 {label} / {raw_bitrate} Mbps",
        _replacing(path, "ファイル"),
    ]
    request = ExportRequest(
        path=path,
        scene_id=scene_id,
        range_choice=RANGE_ALL if frame_range is None else RANGE_WORK_AREA,
        codec=codec,
        bitrate_mbps=raw_bitrate,
        # 許したときの書き出し先 描く数分の間に名前が現れたり中身が変わったりしたら置かない
        expect=expect_now(path),
    )
    return _ExportPlan(request=request, detail="\n".join(part for part in parts if part))


def _export_consent(host: EditorHost, arguments: dict[str, Any]) -> Consent:
    plan = _export_plan(host, arguments)
    return Consent(
        action="動画を書き出します",
        target=str(plan.request.path),
        detail=plan.detail,
        basis=plan.request.expect,
    )


def _awaiting_export(host: EditorHost, seconds: float) -> Pending:
    """書き出しの終わりを待ってから、その時の様子を返す 場面の検出（#321）と同じ待ち方

    待ちきれなければ、その時の様子（rendering と割合）をそのまま返す
    """
    return Pending(wait=host.export_waiter(), seconds=seconds, then=host.export_status)


def _export_video(host: EditorHost, arguments: dict[str, Any], consent: Consent) -> object:
    wait = _wait_seconds(arguments)
    plan = _export_plan(host, arguments)
    # 置く直前に照らす基準は、人が許したときの身元 ここで採り直すと、照らし合わせの後の
    # 変更を新しい基準として受け入れてしまう
    host.start_export(replace(plan.request, expect=_basis(consent)))
    if wait > 0:
        return _awaiting_export(host, wait)
    return {
        "started": str(plan.request.path),
        "next": "書き出しは裏で進みます export_status で進み具合を読めます"
        "（wait_seconds を付けると、その秒数まで終わるのを待ちます）",
    }


def _export_status(host: EditorHost, arguments: dict[str, Any]) -> object:
    wait = _wait_seconds(arguments)
    if wait > 0:
        return _awaiting_export(host, wait)
    return host.export_status()


def _cancel_export(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    if not host.cancel_export():
        if host.export_status().get("state") in EXPORT_RUNNING:
            # 置き始めた後は止めない 止めると答えて置き換えが終わると、前のファイルが消えた
            # ことを AI も人も知らないまま進む
            raise ToolError(
                "書き出しはもう書き出し先へ置き始めているので止めません 書き出し先のファイルは"
                "置き換わります export_status で終わるのを待ってください"
            )
        raise ToolError("走っている書き出しはありません")
    return {"cancelling": True, "status": host.export_status()}


OPERATIONS: tuple[Operation, ...] = (
    Operation(
        name="get_project",
        description="プロジェクトの設定（解像度・fps・長さ）と再生ヘッドの位置を返す",
        schema=_schema({}),
        handler=_get_project,
    ),
    Operation(
        name="list_media",
        description="メディアプールの素材を一覧する media_id はここで得る" + PAGING_NOTE,
        schema=_schema({"contains": _string("名前に含む言葉で絞る"), **_paging()}),
        handler=_list_media,
    ),
    Operation(
        name="list_tracks",
        description="タイムラインのトラックを一覧する" + PAGING_NOTE,
        schema=_schema(_paging()),
        handler=_list_tracks,
    ),
    Operation(
        name="list_clips",
        description="クリップを一覧する track_id を省くと全トラックが対象" + PAGING_NOTE,
        schema=_schema(
            {
                "track_id": _string("絞り込むトラック"),
                "from_seconds": _number("この秒より後に掛かるクリップだけ（タイムラインの秒）"),
                "to_seconds": _number(
                    "この秒より前に始まるクリップだけ（ちょうどから始まる物は外す）"
                ),
                **_paging(),
            }
        ),
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
        description=(
            "使えるエフェクトと生成オブジェクトの一覧 パラメータ名と範囲は kind を渡したときだけ"
            " 返す（全部を並べると長すぎる）" + PAGING_NOTE
        ),
        schema=_schema(
            {
                "kind": _string("詳しく見るエフェクトの種類（パラメータまで返す）"),
                "category": _string("分類で絞る（色・ぼかし・音 など）"),
                "contains": _string("種類か名前に含む言葉で絞る"),
                **_paging(),
            }
        ),
        handler=_list_effects,
    ),
    Operation(
        name="get_subtitles",
        description=(
            "タイムラインに出る字幕を、表示位置つきで一覧する 誤植を探すときは compact で"
            "行の番号・時刻・本文だけを読み、contains（言葉）や時刻で絞る 直すときは"
            " replace_subtitle_text で「誤 → 正」をまとめて渡す" + PAGING_NOTE
        ),
        schema=_schema(
            {
                "media_id": _string("絞り込む素材"),
                "audio": _integer("絞り込む音声の番号（1 から）"),
                "contains": _string("本文に含む言葉で絞る（部分一致）"),
                "segment_id": _string(
                    "この 1 行だけ 長くて切り詰められた（truncated）行の全文を読む"
                ),
                "text_offset": _integer(
                    "segment_id の行の本文を何文字目から返すか（続きは返事の next_text_offset）"
                ),
                "from_seconds": _number("この秒より後に出る字幕だけ（タイムラインの秒）"),
                "to_seconds": _number("この秒より前に出始める字幕だけ（ちょうどから出る物は外す）"),
                "compact": _boolean("行の番号・時刻・本文だけを返す（軽い一覧）"),
                **_paging(),
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
        name="detect_scenes",
        description=(
            "素材（動画）の場面の切り替わり（カット）の位置を一覧する フレーム間の差から"
            "探すので、映像を 1 枚ずつ見るより速い 見つけた位置の前後だけを preview_frame で"
            "確かめる 位置は素材の秒（source_seconds）で、切り替わりは after_seconds との間に"
            "ある timeline はその場面の最初の絵が出るタイムラインのフレーム（preview_frame へ"
            "そのまま渡せる 今のタイムラインに置いていなければ空） 解析は素材を読み込んだときに"
            "裏で始まり、結果は覚える 終わっていなければ wait_seconds まで待ち、それでも"
            "終わらなければ読んだ所までを返す（complete が false） 映像が何本もある素材は"
            "先頭の映像（video_stream）を調べ、timeline にはその映像を出すクリップだけを載せる"
            + PAGING_NOTE
        ),
        schema=_schema(
            {
                "media_id": _string("素材（list_media の media_id）"),
                "threshold": _number(
                    f"感度 {SCENE_THRESHOLD_RANGE[0]:g}〜{SCENE_THRESHOLD_RANGE[1]:g}"
                    f"（既定 {DEFAULT_THRESHOLD:g}） 小さいほど小さな変化も切り替わりに数える"
                    " 一覧の score がこれ以上の所を返す"
                ),
                "min_scene_seconds": _number(
                    f"最短の場面の秒（既定 {DEFAULT_MIN_SCENE:g}） これより近い切り替わりは"
                    "強い方だけ残す フラッシュなどで 2 回続けて拾うのを防ぐ"
                ),
                "from_seconds": _number("この素材の秒から後の切り替わりだけ"),
                "to_seconds": _number("この素材の秒より前の切り替わりだけ"),
                "wait_seconds": _number(
                    f"解析が終わっていないときに待つ秒（0〜{MAX_SCENE_WAIT:g}、"
                    f"既定 {DEFAULT_SCENE_WAIT:g}） 0 なら待たずに読んだ所までを返す"
                ),
                **_paging(),
            },
            ["media_id"],
        ),
        handler=_detect_scenes,
    ),
    Operation(
        name="get_app_settings",
        description=(
            "アプリ自身の設定を読む（本人の好み・ショートカット・画面配置・版・更新・ヘルプ）"
            " 編集の判断に使う 読むだけで、変えるのは人だけ（表示 → 設定… など）"
        ),
        schema=_schema({}),
        handler=_get_app_settings,
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
        description=(
            "クリップの端を動かす head_delta は正で短く、負で頭を前へ伸ばす tail_delta は"
            "正で長くなる テキストや図形も頭を前へ伸ばせる 伸ばしすぎた分は、素材の先頭・"
            "前のクリップの終わり・タイムラインの先頭・後ろのクリップの頭で止め、結果の note と"
            " stopped_at で知らせる"
        ),
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
        description=(
            "クリップにエフェクトを積む 使える kind は list_effects で分かる"
            " stroke_id を渡すとテキストの縁取りの層だけに掛ける（掛けられる kind は "
            + "、".join(STROKE_EFFECT_KINDS)
            + "）"
        ),
        schema=_schema(
            {
                "clip_id": _string("対象"),
                "kind": _string("エフェクトの種類"),
                "params": {"type": "object", "description": "初期値"},
                "stroke_id": _string("縁取りの層（list_clips の strokes）"),
            },
            ["kind"],
        ),
        handler=_add_effect,
        writes=True,
    ),
    Operation(
        name="add_stroke",
        description=(
            "テキストに縁取りの層を足す（Photoshop の境界線のように何重にも重ねられる）"
            " 並びの頭（index 0）が一番手前 太さは字の輪郭から数える 層の無い字に"
            "前からの縁（border_width）があれば、先にそれが 1 つ目の層になる"
            " 省いた項目は今の縁の外側に見える太さと色になる 層の項目は list_effects kind=stroke"
        ),
        schema=_schema(
            {
                "clip_id": _string("対象のテキスト"),
                "width": _number("太さ（画素 字の輪郭から）"),
                "color": _string("色 #RRGGBB か #RRGGBBAA"),
                "position": _string("outside（外側）/ center（中央）/ inside（内側）"),
                "opacity": _number("不透明度（%）"),
                "join": _string("角 round / miter / bevel"),
                "index": _integer("入れる位置 省略すると一番下（一番外）"),
            },
            [],
        ),
        handler=_add_stroke,
        writes=True,
    ),
    Operation(
        name="remove_stroke",
        description="テキストの縁取りの層を外す 層に掛けたエフェクトも一緒に外れる",
        schema=_schema(
            {"clip_id": _string("対象のテキスト"), "stroke_id": _string("縁取りの層")},
            ["stroke_id"],
        ),
        handler=_remove_stroke,
        writes=True,
    ),
    Operation(
        name="move_stroke",
        description="テキストの縁取りの層の順番を変える 0 が一番手前",
        schema=_schema(
            {
                "clip_id": _string("対象のテキスト"),
                "stroke_id": _string("縁取りの層"),
                "index": _integer("行き先"),
            },
            ["stroke_id", "index"],
        ),
        handler=_move_stroke,
        writes=True,
    ),
    Operation(
        name="set_stroke_enabled",
        description="テキストの縁取りの層を消さずに隠す・出す",
        schema=_schema(
            {
                "clip_id": _string("対象のテキスト"),
                "stroke_id": _string("縁取りの層"),
                "enabled": _boolean("描くか"),
            },
            ["stroke_id", "enabled"],
        ),
        handler=_set_stroke_enabled,
        writes=True,
    ),
    Operation(
        name="set_param",
        description=(
            "パラメータを変える effect_id を渡せばそのエフェクト、"
            "省略すればテキストや図形の中身（target=clip でクリップ自身）"
            " stroke_id を渡せば縁取りの層の項目（effect_id も渡せば層に掛けたエフェクト）"
        ),
        schema=_schema(
            {
                "clip_id": _string("対象"),
                "effect_id": _string("エフェクト"),
                "stroke_id": _string("縁取りの層"),
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
                "stroke_id": _string("縁取りの層"),
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
        name="replace_subtitle_text",
        description=(
            "字幕の本文の「誤 → 正」をまとめて置き換える 字幕と、字幕から置いたテキスト"
            "（焼き込み）の両方を直し、1 回の取り消しで戻る 置き換えた数と、当たらなかった組を返す"
            " 焼き込みは字幕から置いた印のあるテキストだけを直す 印の無い前の版の焼き込みは"
            "直さず unmarked_text_clips に数える"
        ),
        schema=_schema(
            {
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                        "required": ["from", "to"],
                    },
                    "description": "置き換える組 [{from: 誤, to: 正}, …] 前から順に当てる",
                },
                "media_id": _string("絞り込む素材（省けば全部）"),
                "audio": _integer("絞り込む音声の番号（1 から media_id と一緒に）"),
            },
            ["pairs"],
        ),
        handler=_replace_subtitle_text,
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
    Operation(
        name="save_project",
        description=(
            "作品を今の場所へ上書き保存する（画面の〔保存〕と同じ） 取り消せない操作なので、"
            "「変更を自動で承認」でも毎回ユーザーに確かめる まだ保存していない作品は "
            "save_project_as を使う"
        ),
        schema=_schema({}),
        handler=_consented(_save_project, _save_consent),
        approved=_save_project,
        writes=True,
        irreversible=True,
        consent=_save_consent,
    ),
    Operation(
        name="save_project_as",
        description=(
            "作品に名前を付けて保存し、以後はその場所の作品として編集する（画面の〔名前を付けて"
            "保存〕と同じ） 書けるのは今の作品のフォルダと、書き出しの窓で前に選んだフォルダの"
            "中だけ ほかの場所は人がファイルの窓で選ぶ 毎回ユーザーに確かめる"
        ),
        schema=_schema(
            {
                "path": _string(
                    f"保存先（{SUFFIX} で終える） 相対の場所は今の作品のフォルダから読む"
                    " フォルダは作らない"
                )
            },
            ["path"],
        ),
        handler=_consented(_save_project_as, _save_as_consent),
        approved=_save_project_as,
        writes=True,
        irreversible=True,
        consent=_save_as_consent,
    ),
    Operation(
        name="export_video",
        description=(
            "動画を書き出す（画面の書き出しの窓と同じ設定と処理） 毎回ユーザーに確かめる"
            " 書き出しは裏で進み、すぐ戻る 進み具合は export_status で読み、止めるのは "
            "cancel_export 書けるのは今の作品のフォルダと、書き出しの窓で前に選んだフォルダの"
            "中だけ 元の素材と同じファイルには書かない"
        ),
        schema=_schema(
            {
                "path": _string(
                    "書き出し先 形式は名前の終わりで決まる（"
                    + " / ".join(CONTAINER_SUFFIXES)
                    + "） 相対の場所は今の作品のフォルダから読む"
                ),
                "range": {
                    "type": "string",
                    "enum": [RANGE_ALL, RANGE_WORK_AREA],
                    "description": (
                        f"書き出す範囲 {RANGE_ALL} は全体、{RANGE_WORK_AREA} はタイムラインで"
                        "決めた範囲 省くと、範囲があれば範囲、無ければ全体（画面と同じ）"
                    ),
                },
                "scene_id": _string("書き出すシーン 省くとメイン（開いているシーンではない）"),
                "codec": _string(
                    f"映像のコーデック {AUTO}（既定 使えなければ次の候補へ）か、"
                    "h264_nvenc / h264_qsv / libx264 などこの機械で使える物"
                ),
                "bitrate_mbps": _integer(
                    f"映像のビットレート（{BITRATE_RANGE_MBPS[0]}〜{BITRATE_RANGE_MBPS[1]} Mbps、"
                    f"既定 {DEFAULT_BITRATE_MBPS}）"
                ),
                "wait_seconds": _number(
                    f"終わるまで待つ秒数（0〜{MAX_WAIT_SECONDS}、既定 0 は待たずに戻る）"
                    " 待っても終わらなければ、その時の進み具合を返す"
                ),
            },
            ["path"],
        ),
        handler=_consented(_export_video, _export_consent),
        approved=_export_video,
        writes=True,
        irreversible=True,
        consent=_export_consent,
    ),
    Operation(
        name="export_status",
        description=(
            "AI が始めた書き出しの進み具合を読む state は idle / rendering（描いている）/ "
            "placing（書き出し先へ置いている）/ done / failed / cancelled"
            " wait_seconds を付けると、終わるかその秒数まで待ってから返す"
        ),
        schema=_schema(
            {
                "wait_seconds": _number(
                    f"終わるまで待つ秒数（0〜{MAX_WAIT_SECONDS}、既定 0 は今の様子をすぐ返す）"
                )
            }
        ),
        handler=_export_status,
    ),
    Operation(
        name="cancel_export",
        description=(
            "走っている書き出しを止める 書きかけは捨て、書き出し先の前からあるファイルは"
            "そのまま残る 作品は変わらない 書き出し先へ置き始めた後（placing）は止めずに断る"
        ),
        schema=_schema({}),
        handler=_cancel_export,
        writes=True,
    ),
    Operation(
        name="export_subtitles",
        description=(
            "開いているタイムラインの字幕をファイルへ書き出す（画面の字幕の〔書き出し…〕と同じ"
            "形式と処理） 毎回ユーザーに確かめる 書けるのは今の作品のフォルダと、書き出しの"
            "窓で前に選んだフォルダの中だけ"
        ),
        schema=_schema(
            {
                "path": _string(
                    "書き出し先 形式は名前の終わりで決まる（"
                    + " / ".join(SUBTITLE_SUFFIXES)
                    + "） 相対の場所は今の作品のフォルダから読む"
                ),
                "range": {
                    "type": "string",
                    "enum": [RANGE_ALL, RANGE_WORK_AREA],
                    "description": (
                        f"{RANGE_ALL} は全体を時刻のまま、{RANGE_WORK_AREA} はタイムラインで決めた"
                        "範囲だけを頭を 0 秒にずらして書く（範囲で書き出した動画に合う）"
                        " 省くと、範囲があれば範囲、無ければ全体（画面と同じ）"
                    ),
                },
            },
            ["path"],
        ),
        handler=_consented(_export_subtitles, _subtitles_consent),
        approved=_export_subtitles,
        writes=True,
        irreversible=True,
        consent=_subtitles_consent,
    ),
    Operation(
        name="save_snapshot",
        description=(
            "開いているタイムラインの 1 フレームを、プロジェクトの解像度の PNG で保存する"
            "（画面の〔静止画を保存…〕と同じ描き方） 絵を見るだけなら preview_frame を使う"
            " 毎回ユーザーに確かめる 書けるのは今の作品のフォルダと、書き出しの窓で前に"
            "選んだフォルダの中だけ"
        ),
        schema=_schema(
            {
                "path": _string("保存先（.png で終える） 相対の場所は今の作品のフォルダから読む"),
                "frame": _integer("描くフレーム 省くと再生ヘッドの位置"),
            },
            ["path"],
        ),
        handler=_consented(_save_snapshot, _snapshot_consent),
        approved=_save_snapshot,
        writes=True,
        irreversible=True,
        consent=_snapshot_consent,
    ),
    Operation(
        name=ASK_USER,
        description=(
            "ユーザーに質問し、答えを待つ 指示があいまいで、どう受け取るかで結果が大きく"
            "変わるときだけ、作業の前に使う 答えは選んだ選択肢の label（selected）と、"
            "ユーザーが自由に書いた文（free_text）で返る 返事が無ければ失敗で返る"
        ),
        schema=_schema(
            {
                "question": {
                    "type": "string",
                    "description": f"質問の文 1 つの話題だけを聞く（{QUESTION_LIMIT} 文字まで）",
                    "maxLength": QUESTION_LIMIT,
                },
                "choices": {
                    "type": "array",
                    "description": (
                        f"選択肢 {CHOICES_RANGE[0]}〜{CHOICES_RANGE[1]} 個"
                        " 違いが分かるように、それぞれ短い名前と説明を付ける"
                    ),
                    "minItems": CHOICES_RANGE[0],
                    "maxItems": CHOICES_RANGE[1],
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": f"ボタンに出す名前（{LABEL_LIMIT} 文字まで）",
                                "maxLength": LABEL_LIMIT,
                            },
                            "description": {
                                "type": "string",
                                "description": (
                                    f"選ぶと何をするか（{DESCRIPTION_LIMIT} 文字まで）"
                                ),
                                "maxLength": DESCRIPTION_LIMIT,
                            },
                        },
                        "required": ["label", "description"],
                        "additionalProperties": False,
                    },
                },
                "multiple": _boolean("複数を選べるようにする（既定 false）"),
                "free_text": _boolean("選択肢に無い答えを書けるようにする（既定 true）"),
            },
            ["question", "choices"],
        ),
        handler=_ask_user,
        asks=True,
    ),
)


def find_operation(name: str) -> Operation | None:
    for operation in OPERATIONS:
        if operation.name == name:
            return operation
    return None
