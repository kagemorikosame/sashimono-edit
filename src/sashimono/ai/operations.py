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
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from sashimono.ai.host import EXPORT_RUNNING, EditorHost, ExportRequest, ToolError
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import (
    Consent,
    Safety,
    pinned_place,
    readable_output,
    refuse_where_the_folder_cannot_be_held,
    writable_path,
)
from sashimono.ai.sheet import (
    MAX_COLUMNS,
    MAX_SHEET_EDGE,
    MIN_CELL_WIDTH,
    SheetError,
    sheet_layout,
)
from sashimono.ai.sound_graph import (
    FLOOR_LUFS,
    GRAPH_DEFAULT_HEIGHT,
    GRAPH_DEFAULT_WIDTH,
    GRAPH_MAX_HEIGHT,
    GRAPH_MIN_HEIGHT,
    GRAPH_MIN_WIDTH,
    MARGIN_LEFT,
    MAX_MARKS,
    Band,
    SoundGraph,
    clipped_columns,
    graph_columns,
    plot_width,
    spans,
    time_step,
)
from sashimono.ai.style_check import COLOR as STYLE_COLOR
from sashimono.ai.style_check import COLOR_TOLERANCE, checked_guide, style_problems
from sashimono.ai.style_check import EFFECT as STYLE_EFFECT
from sashimono.ai.style_check import FONT as STYLE_FONT
from sashimono.compat.aviutl.exo import ExoParseError, load_exo
from sashimono.compat.catalog import TemplateEntry, TemplateError, restyle
from sashimono.compat.looks import (
    exo_commands,
    preset_commands_for,
    restyle_refusal,
    template_placement,
)
from sashimono.compat.mapped import MappedObject
from sashimono.compat.subtitle_looks import (
    TEMPLATE,
    LookError,
    find_choice,
    subtitle_look,
    template_choice,
)
from sashimono.core.chapters import YOUTUBE_DESCRIPTION_LIMIT, chapters_of, youtube_text
from sashimono.core.clipboard import copy_clips, paste_commands
from sashimono.core.commands import (
    MARKER_LABEL_LIMIT,
    AddClip,
    AddCredits,
    AddEffect,
    AddMarkers,
    AddScene,
    AddStroke,
    AddTrack,
    Author,
    Command,
    GroupClips,
    MoveClip,
    MoveClips,
    MoveStroke,
    ParamPath,
    ParamTarget,
    RemoveClip,
    RemoveClips,
    RemoveMarkers,
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
    SubtitleTemplate,
    TrimClip,
    UngroupClips,
    UpdateMarker,
    Voice,
    burn_defaults,
    burn_subtitles,
    export_range,
    has_line_break,
    head_room,
    insert_clip,
    insert_generated,
    insert_media,
    insert_scene,
    marker_at,
    new_scene,
    subtitle_voices,
    subtitle_wrap_width,
    tail_room,
    trimmable,
    valid_marker_color,
)
from sashimono.core.commands.edit import STOP_LOCKED
from sashimono.core.commands.fixed import FADE_EFFECT_KIND, VOLUME_EFFECT_KIND, with_fixed_items
from sashimono.core.commands.insert import new_track, place_media
from sashimono.core.commands.layers import places_mixed
from sashimono.core.commands.speed import SPEED_PERCENT_RANGE, ChangeSpeed, speed_members
from sashimono.core.commands.volume import clip_volume, volume_commands
from sashimono.core.emphasis import (
    EMPHASIS_EASINGS,
    EMPHASIS_KINDS,
    EMPHASIS_PAN_PX_RANGE,
    EMPHASIS_POP_PERCENT_RANGE,
    EMPHASIS_RAMP_MS_RANGE,
    EMPHASIS_SHAKE_PX_RANGE,
    EMPHASIS_ZOOM_PERCENT_RANGE,
    PAN,
    PAN_DIRECTIONS,
    SHAKE,
    ZOOM,
    EmphasisError,
    can_emphasize,
)
from sashimono.core.io.aliases import Alias
from sashimono.core.io.atomic import Expected, expect_now
from sashimono.core.io.presets import Preset
from sashimono.core.io.serialize import SUFFIX, json_text
from sashimono.core.io.subtitles import SUBTITLE_SUFFIXES, subtitles_in_range
from sashimono.core.jetcut import merge_ranges, plan_cuts
from sashimono.core.mixing import (
    DUCK_ATTACK_MS_RANGE,
    DUCK_DEPTH_DB_RANGE,
    DUCK_RELEASE_MS_RANGE,
    VOLUME_LIMIT,
    DuckShape,
    audible_clips,
    db_to_ratio,
    gain_commands,
    loudest,
    loudness_gain,
    measure_project,
    plan_ducking,
    scaled_volume,
    sounds_now,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    EffectId,
    Interpolation,
    Marker,
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
    Transcript,
    TranscriptSegment,
)
from sashimono.core.projection import heard_timeline, project_timeline
from sashimono.core.seams import (
    SEAM_AUDIO_FADE_MS_RANGE,
    SEAM_CROSSFADE,
    SEAM_CROSSFADE_MS_RANGE,
    SEAM_NONE,
    SEAM_VIDEOS,
    SEAM_ZOOM,
    SEAM_ZOOM_PERCENT_RANGE,
    SeamPlan,
    SeamScope,
    SeamStyle,
    SmoothCuts,
    find_seams,
    plan_smoothing,
    seams_after_cuts,
    touching_scope,
)
from sashimono.core.timebase import (
    FrameRate,
    Rounding,
    format_timecode,
    frame_to_seconds,
    seconds_to_frame,
)
from sashimono.effects import registry
from sashimono.effects.emphasis import EmphasisRequest, emphasis_commands, emphasis_label
from sashimono.effects.seams import seam_transition
from sashimono.effects.sources import SHAPE, TEXT, TRANSITION, source_registry
from sashimono.effects.spec import (
    MAX_GRADIENT_STOPS,
    MIN_GRADIENT_STOPS,
    CheckSpec,
    ColorSpec,
    GradientSpec,
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
from sashimono.engine.audio.speech import (
    SPEECH_SOURCES,
    SpeechError,
    WaveformPendingError,
    speaks,
    voice_speech,
)
from sashimono.engine.audio.voicevox import (
    INTONATION_PERCENT_RANGE,
    NARRATION_FOLDER,
    PITCH_RANGE,
    TEXT_LIMIT,
    VOLUME_PERCENT_RANGE,
    Catalog,
    EngineError,
    NarrationSettings,
    Speaker,
    Spoken,
    Style,
    Tuning,
    VoiceEngine,
    split_script,
    wav_seconds,
)
from sashimono.engine.audio.voicevox import SPEED_PERCENT_RANGE as VOICE_SPEED_RANGE
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.cache import SceneState
from sashimono.engine.checks import KIND_NAMES as CHECK_KIND_NAMES
from sashimono.engine.checks import KINDS as CHECK_KINDS
from sashimono.engine.checks import (
    LOUDNESS_TARGETS,
    CheckLimits,
    CheckRequest,
    CheckState,
)
from sashimono.engine.checks.loudness import ABSOLUTE_GATE
from sashimono.engine.checks.profile import (
    MUSIC,
    NOISE,
    SPEECH,
    SPEECH_AND_MUSIC,
    TICK_SECONDS,
    UNCLEAR,
    SoundProfile,
    SoundSegment,
    classify,
)
from sashimono.engine.checks.profile import SILENCE as SOUND_SILENCE
from sashimono.engine.checks.report import CheckReport, ClipIndex
from sashimono.engine.checks.scan import LOUDNESS, SOUND_KINDS
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
    "NARRATION_MODEL",
    "OPERATIONS",
    "Choice",
    "Continue",
    "ImageResult",
    "NarrationJob",
    "NarrationPlan",
    "Operation",
    "Pending",
    "QuestionSpec",
    "find_operation",
    "finish",
    "narration_consent",
    "narration_plan",
    "question_spec",
    "remember_voice_catalog",
    "write_and_place_narration",
]

#: プレビュー画像の既定の横幅 小さめにしてあるのは、AI が見るのは
#: 「意図した絵になっているか」であって、画素を数えるわけではないため
DEFAULT_PREVIEW_WIDTH = 640
MAX_PREVIEW_WIDTH = 1280

#: コンタクトシート（全体を見渡す）の既定の枚数と升の幅
#: 12 枚 x 幅 240 は 1 枚でおよそ 730x630 画素 preview_frame の既定（640x360）の 2 枚ぶんほどの
#: 費用で 12 コマを見せる 細かく見たい所は、ここで当たりを付けてから preview_frame で見る
SHEET_DEFAULT_COUNT = 12
SHEET_MAX_COUNT = 36
SHEET_DEFAULT_CELL = 240
#: 範囲の連続したコマ（動きの確かめ）の既定の枚数と升の幅 動きは細かい所を見るので
#: 升を広めに取り、枚数を絞る 8 枚 x 幅 320 を 4 列で 1 枚およそ 1300x400 画素
RANGE_DEFAULT_COUNT = 8
RANGE_MAX_COUNT = 16
RANGE_DEFAULT_CELL = 320
#: 範囲のコマの既定の列の数 続いたコマは横へ読む 縦横の釣り合いで列を選ぶと 8 枚が
#: 2 列 x 4 行になり、時間の流れが行ごとに折れて追いにくい
RANGE_DEFAULT_COLUMNS = 4
#: 升の幅の上限 これより広くしたいときは preview_frame で 1 コマを大きく見る
MAX_CELL_WIDTH = 640
#: 1 度に描き続けてよい時間（秒） 超えたら編集画面へ手を返し、次の回に続きを描く
#: 返さないと、大きい作品で何十コマも描く間、画面が止まって押した釦も効かない
SHEET_SLICE_SECONDS = 0.1
#: 書き出しの codec の「自動」 画面の書き出しの窓の「自動（使えなければ次の候補）」と同じ
AUTO = "auto"


@dataclass(frozen=True, slots=True)
class ImageResult:
    """画像を返すツールの戻り値"""

    png: bytes
    caption: str = ""


@dataclass(frozen=True, slots=True)
class Continue:
    """長い道具の続き 道具はこれを返して、いったん編集画面へ手を返す

    ブリッジは編集画面の次の回で ``step`` を呼ぶ（:meth:`EditorBridge.invoke`） 続きもまた
    ``Continue`` を返してよく、最後に本当の結果を返す 1 回で全部を描くと、その間は
    画面が描き直されず、中断の釦も効かない
    """

    step: Callable[[], object]


def finish(result: object) -> object:
    """``Continue`` を終わりまで辿って本当の結果を返す 編集画面へ手を返さない呼び方用"""
    while isinstance(result, Continue):
        result = result.step()
    return result


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

    #: 結果に入りうる物 結果はそのまま外（Anthropic）へ渡る（:mod:`sashimono.ai.outbound` #342）
    #: 既定を置かない 置くと、絵や字幕を返す道具を足した人が書き忘れても黙って通り、
    #: 同意を取らずに外へ送る 書き忘れたら道具を作る所で落ちる
    sends: frozenset[Outbound] = field(kw_only=True)
    #: ``sends`` のうち、許されていなければ尋ねも断りもせず、結果の中を伏せ字にして送る物
    #: 今は取り消しと履歴の道具の字幕と文字だけ（操作の名前に文の頭が混ざるだけで、
    #: 道具の役目は文を送ることではない 2026-10-15 利用者の決定）
    hidden: frozenset[Outbound] = field(default=frozenset(), kw_only=True)
    #: 結果が操作の名前（履歴と取り消し）か 名前の「動き: 対象」の対象には素材の名前や本文が
    #: 入るが、消した素材の名前は今の作品に無いので、表
    #: （:func:`~sashimono.ai.outbound.file_secrets`）では伏せられない
    #: 名前と場所か字幕と文字を許していなければ、対象を形で伏せる
    labels: bool = field(default=False, kw_only=True)
    #: 結果のうち、作品や素材の名前そのものを返す欄の名前（``get_project`` の ``name`` など）
    #: 名前と場所を許していなければ、表（:func:`~sashimono.ai.outbound.file_secrets`）に
    #: 頼らず、長さに関係なく値ごと伏せる 表だけだと 1 文字の名前や、表に入る前の名前が漏れる
    #: トラック・シーン・エフェクトの名前は作品の中身の名前で、ここには書かない
    name_keys: frozenset[str] = field(default=frozenset(), kw_only=True)

    def __post_init__(self) -> None:
        if not self.sends:
            # 結果を返す以上、何かは送る 空にすると、何も送らない道具と見分けが付かない
            raise ValueError(f"{self.name}: 結果に入りうる物（sends）を書いてください")
        if not self.hidden <= self.sends & {Outbound.TEXT}:
            # 伏せ方を持っているのは字幕と文字だけ（名前と場所は道具を問わず伏せる）
            # 絵は伏せられないので、ここに書くと尋ねずに送ることになる
            raise ValueError(f"{self.name}: 伏せて送れるのは sends に書いた字幕と文字だけです")
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

    def start(self, host: EditorHost, arguments: dict[str, Any]) -> object:
        """初めの 1 回だけ動かす 長い道具は :class:`Continue` を返す（ブリッジが続きを呼ぶ）"""
        return self.handler(host, arguments)

    def __call__(self, host: EditorHost, arguments: dict[str, Any]) -> object:
        """終わりまで動かして結果を返す 編集画面のスレッドで、手を返さずに済む所から呼ぶ"""
        return finish(self.handler(host, arguments))


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


def _no_explicit_null(arguments: dict[str, Any], key: str) -> None:
    """配列の引数に null を明示したら断る 省いたときと同じに扱わない

    省いたときの既定は「選択中のクリップすべて」「字幕すべて」「全体の 12 コマ」のように
    広い 明示の null をそちらへ読み替えると、AI が渡したつもりの物とは別の、ずっと
    広い相手に効く
    """
    if key in arguments and arguments[key] is None:
        raise ToolError(f"{key} に null が来ました 省くか、配列で渡してください")


def _target_clips(host: EditorHost, arguments: dict[str, Any]) -> tuple[ClipId, ...]:
    """引数の ``clip_ids``、無ければ選んでいるクリップすべて

    画面で何本か選んでから「これをまとめて 2 秒後ろへ」と頼めるようにする
    見つからない ID が 1 つでもあれば止める 一部だけ動かすと、どれが動いたのかを
    AI も人も追えなくなる
    """
    _no_explicit_null(arguments, "clip_ids")
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


def _clip_id_list(
    host: EditorHost, arguments: dict[str, Any], key: str = "clip_ids"
) -> tuple[ClipId, ...]:
    """``key``（既定 ``clip_ids``）を読んで確かめる 重なった ID は 1 つにする

    重なったまま渡すと「3 本を移動」のような本数が実際と食い違う 型が違えば
    ToolError にする 素の TypeError で落ちると、AI は何を直せばよいか分からない
    """
    _no_explicit_null(arguments, key)
    raw = arguments.get(key)
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ToolError(f"{key} はクリップ ID の配列で渡してください")
    names = _string_items(raw, key)
    project = _project(host)
    for clip_id in names:
        _require_clip(project, clip_id)
    # 最後に現れた位置を残す（画面の set_selection と同じ） 最後の 1 本が主になる
    # 約束なので、先に現れた位置を残すと [A, B, A] の主が B に化ける
    ids = [ClipId(clip_id) for clip_id in names]
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
        # 編集の方針（#343）を決めてあるか 中身は本人の文なので get_style_guide で読む
        # （字幕と文字として同意を取る） ここで有る無しだけを返し、読み忘れを防ぐ
        "has_style_guide": not project.style_guide.empty,
        # 作った声の表記（#335） 書き出した動画の概要欄などに入れてもらう物
        "credits": list(project.credits),
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
    elif isinstance(spec, GradientSpec):
        # 平らな数の並びは形を書かないと AI が色だけの並びを渡し、既定へ戻されて何も変わらない
        described["type"] = "gradient"
        described["format"] = (
            "[位置, R, G, B, A, 位置, R, G, B, A, ...] の平らな並び（どれも 0..1 sRGB）"
            f" 止まりは {MIN_GRADIENT_STOPS}〜{MAX_GRADIENT_STOPS} 個"
        )
        if spec.legacy is not None:
            # エフェクトの並び（#364）は空が既定 空の意味を書かないと、AI が空を「色が無い」と読む
            described["format"] += (
                " 空の並び [] なら前からの色の項目（開始色・終了色、色 1 …）で塗る"
            )
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
    for index, entry in enumerate(pairs_raw):
        if not isinstance(entry, dict) or not isinstance(entry.get("from"), str):
            raise ToolError('pairs の各組は {"from": 誤, "to": 正} です（from は文字で渡します）')
        if not entry["from"]:
            raise ToolError('pairs の各組は {"from": 誤, "to": 正} です（from は空にできません）')
        replacement = entry.get("to", "")
        if not isinstance(replacement, str):
            # null を空の文字と読むと、誤植の言葉を字幕から黙って消す 消したいときは
            # to を省くか空の文字を渡してもらう
            raise ToolError(
                f"pairs の {index + 1} 組目の to は文字で渡してください"
                f"（{type(replacement).__name__} が来ました 消すなら空の文字）"
            )
        pairs.append((entry["from"], replacement))

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


def _whole_number(arguments: dict[str, Any], key: str, default: int) -> int:
    """整数の引数 省けば ``default`` 小数や文字で来たら、直し方を書いて断る

    ``int()`` で読むと 2.7 が黙って 2 になり、AI が頼んだのと違うコマを見て判断する
    """
    value = arguments.get(key)
    if value is None:
        return default
    return _integer_value(value, key)


def _integer_value(value: object, name: str) -> int:
    """整数 1 つ 省いたときの既定は持たない（配列の中の null を既定へ読み替えないため）"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolError(f"{name} は整数で渡してください（{type(value).__name__} が来ました）")
    if isinstance(value, float) and not value.is_integer():
        raise ToolError(f"{name} は整数で渡してください（{value}）")
    return int(value)


def _string_items(raw: list[Any], name: str) -> list[str]:
    """文字の配列の中身を 1 つずつ確かめる null や数は、何番目かを書いて断る

    ``str()`` で読むと null が "None" という ID や名前になり、見当違いの「見つかりません」
    が返るか、何にも当たらずに黙って減る
    """
    for index, entry in enumerate(raw):
        if not isinstance(entry, str):
            raise ToolError(
                f"{name} の {index + 1} 個目は文字で渡してください"
                f"（{type(entry).__name__} が来ました）"
            )
    return list(raw)


def _within(arguments: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    """範囲の決まった整数の引数 外れたら端へ寄せず断る

    黙って寄せると、AI は頼んだ枚数を見たつもりで、見ていないコマについて判断する
    """
    value = _whole_number(arguments, key, default)
    if not low <= value <= high:
        raise ToolError(f"{key} は {low}〜{high} にしてください（{value}）")
    return value


def _sheet_frames_from_range(project: Project, arguments: dict[str, Any]) -> list[int]:
    """全体（または from_frame〜to_frame）を ``count`` 等分し、それぞれの真ん中のコマ

    区切りの頭ではなく真ん中を取るのは、頭だと最初の升が黒い頭の 1 コマや場面切り替えの
    途中になりやすく、その区間の様子を表さないため
    """
    last = project.duration
    if last <= 0:
        raise ToolError("タイムラインが空です 描くコマがありません")
    start = _whole_number(arguments, "from_frame", 0)
    end = _whole_number(arguments, "to_frame", last)
    start, end = max(0, start), min(end, last)
    if end <= start:
        raise ToolError(
            f"from_frame〜to_frame に描けるコマがありません（タイムラインは 0〜{last - 1}）"
        )
    count = _within(arguments, "count", SHEET_DEFAULT_COUNT, 1, SHEET_MAX_COUNT)
    span = end - start
    if span <= count:
        return list(range(start, end))
    return [start + (2 * index + 1) * span // (2 * count) for index in range(count)]


def _sheet_frames_from_list(project: Project, raw: object) -> list[int]:
    """AI が選んだコマ（場面の切り替わりの一覧など）をそのまま並べる"""
    if not isinstance(raw, list) or not raw:
        raise ToolError("frames はフレーム番号の並び（1 つ以上）で渡してください")
    if len(raw) > SHEET_MAX_COUNT:
        raise ToolError(
            f"frames は {SHEET_MAX_COUNT} 個までにしてください（{len(raw)} 個）"
            " 多いときは何回かに分けるか、from_frame と to_frame で区切ってください"
        )
    last = project.duration - 1
    frames: list[int] = []
    for index, value in enumerate(raw):
        # null を既定の 0 に読み替えない 読み替えると頼んでいない頭のコマを描く
        frame = _integer_value(value, f"frames の {index + 1} 個目")
        if not 0 <= frame <= last:
            raise ToolError(
                f"frames の {index + 1} 個目（{frame}）はタイムラインの外です（0〜{last}）"
            )
        frames.append(frame)
    return frames


def _contact_sheet(host: EditorHost, arguments: dict[str, Any]) -> object:
    """全体を見渡す 等分したコマ（または指定したコマ）を 1 枚に並べる"""
    project = _project(host)
    # 鍵の有無で分ける 明示の null を省略と同じに読むと、渡したつもりの場面の一覧の
    # 代わりに全体の 12 コマを描く null は _sheet_frames_from_list が断る
    if "frames" in arguments:
        if any(arguments.get(key) is not None for key in ("from_frame", "to_frame", "count")):
            # どちらを優先したかを AI が知らないまま読むと、見たいコマと違う 1 枚を信じる
            raise ToolError("frames と from_frame・to_frame・count は一緒に使えません")
        frames = _sheet_frames_from_list(project, arguments["frames"])
    else:
        frames = _sheet_frames_from_range(project, arguments)
    cell = _within(arguments, "cell_width", SHEET_DEFAULT_CELL, MIN_CELL_WIDTH, MAX_CELL_WIDTH)
    return _draw_sheet(host, project, frames, cell, arguments, "コンタクトシート")


def _preview_range(host: EditorHost, arguments: dict[str, Any]) -> object:
    """動きを確かめる 続いたコマ（step おき）を 1 枚に並べる"""
    project = _project(host)
    last = project.duration - 1
    if last < 0:
        raise ToolError("タイムラインが空です 描くコマがありません")
    start = _whole_number(arguments, "start_frame", host.playhead)
    if not 0 <= start <= last:
        raise ToolError(f"start_frame（{start}）はタイムラインの外です（0〜{last}）")
    count = _within(arguments, "count", RANGE_DEFAULT_COUNT, 2, RANGE_MAX_COUNT)
    step = _within(arguments, "step", 1, 1, max(1, project.duration))
    # 終わりを越えた分は描かない 越えた所は黒いだけで、動きの確かめにならない
    frames = [frame for frame in range(start, start + count * step, step) if frame <= last]
    if len(frames) < 2:
        # 1 コマだけの 1 枚は動きを示さない 返すと AI は止まった絵だと読み違える
        raise ToolError(
            f"start_frame {start} から step {step} おきでは、終わり（{last}）までに描けるのが"
            f" {len(frames)} コマです 動きを見るには 2 コマ以上要ります start_frame を前へ"
            " 戻すか、step を小さくしてください"
        )
    cell = _within(arguments, "cell_width", RANGE_DEFAULT_CELL, MIN_CELL_WIDTH, MAX_CELL_WIDTH)
    # 終わりで足りなくなった分は黙って減らさず、返事に書く 書かないと、AI は頼んだ枚数を
    # 見たつもりで、その先の動きについて判断する
    note = ""
    if len(frames) < count:
        note = f" 頼んだ {count} コマのうち {len(frames)} コマ（タイムラインの終わり {last} まで）"
    return _draw_sheet(
        host,
        project,
        frames,
        cell,
        arguments,
        "範囲のコマ",
        default_columns=RANGE_DEFAULT_COLUMNS,
        note=note,
    )


def _draw_sheet(
    host: EditorHost,
    project: Project,
    frames: list[int],
    cell_width: int,
    arguments: dict[str, Any],
    title: str,
    *,
    default_columns: int | None = None,
    note: str = "",
) -> object:
    """並べ方を決めて、描き始める 描くのは :class:`Continue` で 1 回ずつ（時間で区切る）"""
    columns = default_columns
    if arguments.get("columns") is not None:
        columns = _whole_number(arguments, "columns", 0)
    try:
        layout = sheet_layout(
            len(frames),
            cell_width=cell_width,
            frame_size=project.settings.resolution,
            columns=columns,
        )
    except SheetError as exc:
        # 並べ方の文面は直し方まで書いてある そのまま AI へ返す
        raise ToolError(str(exc)) from exc
    labels = [f"{format_timecode(frame, project.rate)} {frame}" for frame in frames]
    # 再生しながら別のコマを描くと GL の資源を取り合う（preview_frame と同じ）
    host.stop_playback()
    canvas = host.start_sheet(layout)
    caption = (
        f"{title} {len(frames)} コマ（{layout.columns} 列 x {layout.rows} 行 左上から右へ"
        f" {layout.width}x{layout.height} 画素）{note} 各コマの下に「タイムコード フレーム」: "
        + " / ".join(f"{index + 1}) {label}" for index, label in enumerate(labels))
    )
    position = 0

    def step() -> object:
        nonlocal position
        started = time.monotonic()
        # 1 コマは必ず描く 描く前に時間で止めると、1 コマが重い作品で先へ進まない
        while position < len(frames):
            canvas.draw(position, frames[position], labels[position])
            position += 1
            if time.monotonic() - started >= SHEET_SLICE_SECONDS:
                break
        if position < len(frames):
            return Continue(step)
        return ImageResult(png=canvas.png(), caption=caption)

    return Continue(step)


def _get_app_settings(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    # 写しを返す ホストの持ち物をそのまま渡すと、後の道具の中で書き換えたときに
    # 本人の設定まで変わる（アプリ自身の設定は人だけが変える #350）
    return copy.deepcopy(host.app_settings())


# --- 編集の方針（#343）---

#: 方針の道具が添える断り書き 方針は本人が AI に渡す決まり AI は変えない
_STYLE_NOTE = (
    "本人が ファイル → プロジェクト設定… の「編集の方針」で書いた決まりです AI は読むだけで、"
    "変えるのは本人だけです 空の欄は決めていない（好きに選んでよい）という意味です"
    " 指示と方針が食い違うときは指示に従い、食い違ったことを返事で伝えてください"
)


def _effect_label(kind: str) -> str:
    definition = registry.get(kind)
    # 読み込んでいない AviUtl のスクリプトなど、今の環境に無い種類は名前の代わりに種類を出す
    return definition.label if definition is not None else kind


def _paged_after(
    head: dict[str, Any], rows: list[Any], arguments: dict[str, Any], name: str, note: str
) -> dict[str, Any]:
    """``head`` の後ろに一覧 ``rows`` を切って並べた返事 合わせて上限に収める

    一覧だけを :func:`_paged` で切ると、前に置く欄（方針の文・直す先）の分だけ上限を越え、
    打ち切られて続きが読めない 越えたら一覧の件数を減らす 一覧の続きの案内（note）は消さず、
    ``note`` を後ろへ足す
    """
    page = dict(arguments)
    while True:
        listed = _paged(rows, page, name=name)
        merged = {**head, **listed, "note": " ".join(filter(None, (listed.get("note"), note)))}
        count = int(listed["count"])
        if len(json_text(merged)) <= MAX_RESULT_CHARS:
            return merged
        if count <= 1:
            break
        page["limit"] = max(1, count * 3 // 4)
    # 一覧を 1 件まで減らしても越える 前に置く欄（手で書き換えたファイルの長い方針・クリップの
    # 長い書体の名前）が長すぎる 持てる方針は欄の上限で必ず収まるので、ここへ来るのはそれ以外だけ
    # 一番長い所から切って収め、切ったこと（truncated）を返す 黙って返すと上限を越えて打ち切られ、
    # 後ろの欄も続きの案内も読めない（#382 の CodeRabbit の指摘）
    told = "長すぎる所を切った（truncated）"
    fitted = _fit({**merged, "note": f"{told} {merged['note']}"}, MAX_RESULT_CHARS)
    if len(json_text(fitted)) <= MAX_RESULT_CHARS:
        return dict(fitted)
    # 小さな欄が多すぎて切っても収まらない（名前の短い書体を何万個も手で書いたなど）
    return {"truncated": True, "count": 0, "note": f"{told} 返せる長さに収まらなかった"}


def _get_style_guide(host: EditorHost, arguments: dict[str, Any]) -> object:
    guide = _project(host).style_guide
    head: dict[str, Any] = {
        "set": not guide.empty,
        "fonts": list(guide.fonts),
        "colors": list(guide.colors),
        "subtitles": guide.subtitles,
        "tempo": guide.tempo,
        "notes": guide.notes,
    }
    # 効果は一覧から印を付けて選ぶので、数百になりうる（#382 の Codex の指摘で上限を 300 にした）
    # 全部を 1 回で返すと上限を越えて打ち切られるので、効果だけを切って続きを読ませる
    rows = [{"kind": kind, "label": _effect_label(kind)} for kind in guide.effects]
    return _paged_after(head, rows, arguments, "effects", _STYLE_NOTE)


def _check_style_guide(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    # 照らす所（style_problems）と同じ方針から checked と allowed を作る 読めない色を残すと、
    # 照らしていないのに色を照らしたと返す
    guide = checked_guide(project.style_guide)
    problems = style_problems(project)
    counts = dict.fromkeys((STYLE_FONT, STYLE_COLOR, STYLE_EFFECT), 0)
    for problem in problems:
        counts[problem.kind] += 1
    rows = [problem.row() for problem in problems]
    head: dict[str, Any] = {
        # 照らした欄 空の欄（決めていない物）は照らさない 何も照らしていないのに
        # 「食い違い 0」だけを返すと、方針どおりだと読み違える
        "checked": {
            STYLE_FONT: bool(guide.fonts),
            STYLE_COLOR: bool(guide.colors),
            STYLE_EFFECT: bool(guide.effects),
        },
        "counts": counts,
        # 直す先 書体の名前・色・効果の種類だけで、方針の文（字幕の見た目・テンポ・そのほか）は
        # 返さない 字幕と文字を送らないと決めた人にも、何へ直せばよいかが分かる
        "allowed": {
            STYLE_FONT: list(guide.fonts),
            STYLE_COLOR: list(guide.colors),
            STYLE_EFFECT: list(guide.effects),
        },
    }
    note = (
        "照らすのは書体・色（字と図形の塗り・縁取りの層）・効果の種類だけです 影やエフェクトの"
        "中の色、字幕の見た目・テンポ・そのほかの文は get_style_guide で読んで照らしてください"
        f" 色は各成分 {COLOR_TOLERANCE} までの差を同じ色と見なします"
    )
    if len(json_text(head["allowed"])) > MAX_RESULT_CHARS // 2:
        # 効果を一覧のほとんど（数百）選んだ方針 直す先を全部並べると、食い違いの一覧が
        # 1 件も入らない 効果の直す先は省き、方針の外の種類（problems）だけを返す
        head["allowed"][STYLE_EFFECT] = []
        head["allowed_note"] = (
            "効果の直す先は多いので省いた get_style_guide の effects で読める"
            " 方針の外の種類は problems にある"
        )
    return _paged_after(head, rows, arguments, "problems", note)


def _author(by: Author | None) -> str | None:
    return by.value if by is not None else None


def _get_history(host: EditorHost, arguments: dict[str, Any]) -> object:
    del arguments
    document = host.document
    return {
        "can_undo": document.can_undo,
        "undo_label": document.undo_label,
        "recent": list(document.history_labels[-10:]),
        "can_redo": document.can_redo,
        "redo_label": document.redo_label,
        # 次に戻す・やり直す段を誰が変えたか（person か ai） 人の段は AI は戻さない（#333）
        "undo_by": _author(document.undo_by),
        "redo_by": _author(document.redo_by),
        # 今の指示で変えた手（古い順） 指示の中で undo するとここから戻る（#333）
        "this_instruction": list(document.steps_in_checkpoint),
    }


# --- 目印（マーカー）とチャプター（#330）---

#: 1 回で置ける目印の数 長い動画の場面ごとに置いても足りる数 これより多い頼みは、場面の
#: 一覧をそのまま全部渡したなど取り違えのことが多く、目盛りが旗で埋まって読めなくなる
MAX_MARKERS_PER_CALL = 200


def _marker_row(marker: Marker, rate: FrameRate) -> dict[str, Any]:
    return {
        "frame": marker.frame,
        "seconds": round(float(frame_to_seconds(marker.frame, rate)), 3),
        "timecode": format_timecode(marker.frame, rate),
        "label": marker.label,
        "color": marker.color,
    }


def _marker_frame(value: object, name: str) -> int:
    """目印の位置 0 以上の整数だけを受ける

    小数を切り捨てて受けると、秒をフレームへ直し忘れた AI の頼みが、ずっと頭の方へ黙って
    置かれる（2.5 秒のつもりが 2 フレーム目）
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ToolError(f"{name} は 0 以上の整数（フレーム）で渡してください: {value!r}")
    return value


def _marker_color(value: object) -> str:
    # 呼ぶ側はスキーマを確かめずに渡してくる 文字にしてから見ると、形の違う値の理由が
    # 「色の形」だけになり、何を渡し間違えたかが AI に伝わらない
    if not isinstance(value, str) or not valid_marker_color(value):
        raise ToolError(f"色は #RRGGBB の文字列で渡してください: {value!r}")
    return value


def _marker_label(value: object) -> str:
    # 文字にして受けると、数や null が "123" "None" という名前で目次に入る
    if not isinstance(value, str):
        raise ToolError(f"名前は文字列で渡してください（{type(value).__name__} が来ました）")
    label = value.strip()
    if len(label) > MARKER_LABEL_LIMIT:
        raise ToolError(f"名前は {MARKER_LABEL_LIMIT} 文字までです（{len(label)} 文字）")
    # コマンドでも断るが、ここで先に断って直し方を添える 空白に直して通すと、本人の文が
    # 黙って変わる（人の右クリックも同じく断る）
    if has_line_break(label):
        raise ToolError(
            "名前に改行は入れられません 1 行の見出しにしてください（目次の 1 行が割れます）"
        )
    return label


def _require_marker(host: EditorHost, frame: int) -> Marker:
    marker = marker_at(_project(host).timeline, frame)
    if marker is None:
        raise ToolError(
            f"{frame} フレームに目印がありません（list_markers で位置を確かめてください）"
        )
    return marker


def _list_markers(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    markers = sorted(project.timeline.markers, key=lambda marker: marker.frame)
    rows = [_marker_row(marker, project.rate) for marker in markers]
    return _paged(rows, arguments, name="markers")


def _add_markers(host: EditorHost, arguments: dict[str, Any]) -> object:
    items = arguments.get("markers")
    if not isinstance(items, list) or not items:
        raise ToolError("markers に置く目印を 1 つ以上渡してください")
    if len(items) > MAX_MARKERS_PER_CALL:
        raise ToolError(f"1 回に置ける目印は {MAX_MARKERS_PER_CALL} 個までです（{len(items)} 個）")
    timeline = _project(host).timeline
    markers: list[Marker] = []
    for item in items:
        if not isinstance(item, dict):
            raise ToolError("markers の要素は {frame, label, color} の形で渡してください")
        frame = _marker_frame(item.get("frame"), "frame")
        if marker_at(timeline, frame) is not None:
            raise ToolError(
                f"{frame} フレームにはもう目印があります 名前や色を変えるなら update_marker を"
                "使ってください"
            )
        # 既定へ回すのは省いたときだけ 値の真偽で分けると、渡し間違えた 0 や空の色が
        # 確かめられずに黄色の目印になる
        color = _marker_color(item["color"]) if "color" in item else Marker(frame).color
        label = _marker_label(item["label"]) if "label" in item else ""
        markers.append(Marker(frame, label, color))
    host.apply_commands([AddMarkers(tuple(markers))], "マーカーを置く")
    project = _project(host)
    added = {marker.frame for marker in markers}
    return {
        "added": [
            _marker_row(marker, project.rate)
            for marker in project.timeline.markers
            if marker.frame in added
        ],
        "marker_count": len(project.timeline.markers),
    }


def _update_marker(host: EditorHost, arguments: dict[str, Any]) -> object:
    frame = _marker_frame(arguments.get("frame"), "frame")
    current = _require_marker(host, frame)
    if not any(key in arguments for key in ("to_frame", "label", "color")):
        raise ToolError("to_frame・label・color のどれかを指定してください")
    changed = Marker(
        frame=(
            _marker_frame(arguments["to_frame"], "to_frame")
            if "to_frame" in arguments
            else current.frame
        ),
        label=_marker_label(arguments["label"]) if "label" in arguments else current.label,
        color=_marker_color(arguments["color"]) if "color" in arguments else current.color,
    )
    host.apply_commands([UpdateMarker(frame, changed)], "マーカーを変更")
    return {"marker": _marker_row(changed, _project(host).rate)}


def _delete_markers(host: EditorHost, arguments: dict[str, Any]) -> object:
    timeline = _project(host).timeline
    everything = _flag_argument(arguments, "all", False)
    # 両方来たら断る all を先に見ると、位置を指したつもりの頼みで本人の目印が全部消える
    if everything and arguments.get("frames") is not None:
        raise ToolError("frames と all は一緒に渡せません どちらか 1 つにしてください")
    if everything:
        # 目印の数だけ位置を並べる コマンドは位置 1 つで先頭の 1 つだけを消すので、位置の
        # 重なりをまとめると、手で書いた作品の同じ位置の 2 つ目が残る
        frames = tuple(marker.frame for marker in timeline.markers)
        if not frames:
            raise ToolError("消す目印がありません")
    else:
        # 省いたら断る 空の引数で全部消えると、指定し忘れただけで本人の目印が丸ごと消える
        given = arguments.get("frames")
        if not isinstance(given, list) or not given:
            raise ToolError("frames に消す目印の位置を渡してください（全部なら all を真に）")
        # 同じ位置を 2 度書いた頼みは 1 つにまとめる 一覧の位置を重ねて渡しただけで、
        # 同じ位置に目印が 2 つ無いと断られるのは AI に分かりにくい
        frames = tuple(dict.fromkeys(_marker_frame(value, "frames") for value in given))
        for frame in frames:
            _require_marker(host, frame)
    host.apply_commands([RemoveMarkers(frames)], "マーカーを削除")
    return {"deleted": len(frames), "marker_count": len(_project(host).timeline.markers)}


def _get_chapters(host: EditorHost, arguments: dict[str, Any]) -> object:
    result = chapters_of(_project(host).timeline)
    rows = [
        {
            "frame": chapter.frame,
            "start_seconds": round(float(chapter.start), 3),
            "end_seconds": round(float(chapter.end), 3),
            "title": chapter.title,
        }
        for chapter in result.chapters
    ]
    text = youtube_text(result)
    common: dict[str, Any] = {
        "active_scene": str(host.active_scene) if host.active_scene is not None else None,
        "duration_seconds": round(float(result.duration), 3),
        # 概要欄に入らない長さの文は返さない 貼れない上に、道具の結果の上限を食って一覧が切れる
        # わけは warnings に入る
        "youtube": text if len(text) <= YOUTUBE_DESCRIPTION_LIMIT else None,
        "warnings": list(result.warnings),
        "unnamed_markers": result.unnamed,
        "outside_markers": result.outside,
    }
    # 一覧は切って返す 目次の文と一緒に全部を返すと、長い目次で道具の結果の上限を越え、
    # 途中で切れて後ろの行が読めない 文の分だけ一覧の件数を減らして上限に収める
    page = {**common, **_paged(rows, arguments, name="chapters")}
    while len(json_text(page)) > MAX_RESULT_CHARS and page["count"] > 1:
        smaller = {**arguments, "limit": max(1, page["count"] * 3 // 4)}
        page = {**common, **_paged(rows, smaller, name="chapters")}
    if len(json_text(page)) > MAX_RESULT_CHARS and page["youtube"] is not None:
        # 1 行まで減らしても越える（読み込んだ長い名前が文と一覧の両方に入る） 文を外し、
        # 一覧から読めることを書く 文と一覧の両方を残すと、上限を越えて途中が切れる
        page["youtube"] = None
        page["warnings"] = [
            *page["warnings"],
            "目次の文は長すぎて一緒に返せません 一覧（chapters）の時刻と名前から組んでください",
        ]
    overflow = len(json_text(page)) - MAX_RESULT_CHARS
    if overflow > 0 and page["chapters"]:
        # それでも越えるのは、1 行の名前だけで上限に近いとき 名前を切り詰めて印を付ける
        # （ほかの一覧の道具と同じ :func:`_fit`）
        row = page["chapters"][0]
        page["chapters"] = [_fit(row, max(_KEEP_CHARS, len(json_text(row)) - overflow - 64))]
    return page


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


# --- 完成物の検査（#337）---

#: 検査が終わっていないときに待つ秒の既定と上限 待つのは AI のスレッドで、画面は止めない
#: 上限を置くのは、長い作品で会話が何分も止まったように見えないため 待ちきれなければ
#: 調べた所までを返し、続きはもう 1 度呼んで受け取る（場面の検出 #358 と同じ）
DEFAULT_CHECK_WAIT = 30.0
MAX_CHECK_WAIT = 120.0


def _limit_number(
    arguments: dict[str, Any], key: str, default: float, low: float, high: float
) -> float:
    """しきい値の引数 省けば ``default`` 範囲の外・数でない物は、受け付ける範囲を書いて断る

    丸めて受け取らない 範囲の外を黙って端へ寄せると、AI は頼んだしきい値で調べたと思い込む
    """
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolError(f"{key} は数で渡してください（{value!r} が来ました）")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ToolError(f"{key} は {low:g}〜{high:g} で渡してください（{value!r} が来ました）")
    return number


def _check_kinds(arguments: dict[str, Any]) -> frozenset[str]:
    raw = arguments.get("checks")
    if raw is None:
        return frozenset(CHECK_KINDS)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ToolError(f"checks は {', '.join(CHECK_KINDS)} から選んだ配列で渡してください")
    unknown = [str(kind) for kind in raw if kind not in CHECK_KINDS]
    if unknown:
        raise ToolError(
            f"知らない種類です: {', '.join(unknown)}（選べるのは {', '.join(CHECK_KINDS)}）"
        )
    return frozenset(str(kind) for kind in raw)


def _check_limits(host: EditorHost, arguments: dict[str, Any]) -> CheckLimits:
    plain = CheckLimits()
    expected = arguments.get("expected_seconds")
    clip_samples = arguments.get("clip_samples", plain.clip_samples)
    if isinstance(clip_samples, bool) or not isinstance(clip_samples, int):
        raise ToolError(f"clip_samples は整数で渡してください（{clip_samples!r} が来ました）")
    if not 1 <= clip_samples <= 1000:
        raise ToolError(f"clip_samples は 1〜1000 で渡してください（{clip_samples!r} が来ました）")
    return CheckLimits(
        black_level=_limit_number(arguments, "black_level", plain.black_level, 0.0, 0.5),
        min_black_seconds=_limit_number(
            arguments, "min_black_seconds", plain.min_black_seconds, 0.0, 60.0
        ),
        freeze_noise=_limit_number(arguments, "freeze_noise", plain.freeze_noise, 0.0, 0.1),
        min_freeze_seconds=_limit_number(
            arguments, "min_freeze_seconds", plain.min_freeze_seconds, 0.1, 600.0
        ),
        silence_db=_limit_number(arguments, "silence_db", plain.silence_db, -120.0, -10.0),
        min_silence_seconds=_limit_number(
            arguments, "min_silence_seconds", plain.min_silence_seconds, 0.1, 600.0
        ),
        clip_db=_limit_number(arguments, "clip_db", plain.clip_db, -6.0, 0.0),
        clip_samples=clip_samples,
        loudness_target=_limit_number(
            arguments, "loudness_target", float(host.loudness_target), -40.0, -5.0
        ),
        loudness_tolerance=_limit_number(
            arguments, "loudness_tolerance", plain.loudness_tolerance, 0.1, 10.0
        ),
        safe_margin=_limit_number(arguments, "safe_margin_percent", 0.0, 0.0, 20.0) / 100.0,
        expected_seconds=(
            None
            if expected is None
            else _limit_number(arguments, "expected_seconds", 0.0, 0.1, 86400.0)
        ),
        length_tolerance=_limit_number(
            arguments, "length_tolerance_seconds", plain.length_tolerance, 0.0, 3600.0
        ),
        kinds=_check_kinds(arguments),
    )


def _frame_argument(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{key} は整数のフレームで渡してください（{value!r} が来ました）")
    return value


def _check_request(host: EditorHost, arguments: dict[str, Any]) -> CheckRequest:
    limits = _check_limits(host, arguments)
    raw_path = arguments.get("path")
    if raw_path is not None:
        if arguments.get("from_frame") is not None or arguments.get("to_frame") is not None:
            raise ToolError("from_frame と to_frame はタイムラインを調べるときだけ使えます")
        path = readable_output(host, raw_path)
        stat = path.stat()
        return CheckRequest(limits, path=path, stamp=(stat.st_size, stat.st_mtime_ns))
    project = _project(host)
    if project.duration <= 0:
        raise ToolError("タイムラインが空です クリップを置いてから調べてください")
    start = _frame_argument(arguments, "from_frame", 0)
    end = _frame_argument(arguments, "to_frame", project.duration)
    if not 0 <= start < end <= project.duration:
        raise ToolError(
            f"範囲は 0 ≦ from_frame < to_frame ≦ {project.duration}（タイムラインの長さ）で"
            f"渡してください（{start}〜{end} が来ました）"
        )
    scene = str(host.active_scene) if host.active_scene is not None else None
    return CheckRequest(limits, project=project, start=start, end=end, scene=scene)


def _check_output(host: EditorHost, arguments: dict[str, Any]) -> object:
    """完成物を調べる 終わっていなければ待ってから、待ちきれなければ途中までを返す"""
    request = _check_request(host, arguments)
    wait = _limit_number(arguments, "wait_seconds", DEFAULT_CHECK_WAIT, 0.0, MAX_CHECK_WAIT)
    host.request_check(request)
    state = host.check_state()
    if not state.running or wait <= 0:
        return _check_result(host, request, state, arguments)
    return Pending(
        wait=host.check_waiter(),
        seconds=wait,
        then=lambda: _check_result(host, request, host.check_state(), arguments),
    )


def _check_result(
    host: EditorHost, request: CheckRequest, state: CheckState, arguments: dict[str, Any]
) -> object:
    if state.request != request:
        raise ToolError("調べている間に別の検査に替わりました もう 1 度呼んでください")
    report = state.report
    if state.failure is not None:
        raise ToolError(f"調べられませんでした: {state.failure}")
    if report is None:
        raise ToolError("検査の結果がまだありません もう 1 度呼んでください")
    timeline = request.project is not None
    rate = request.project.rate if request.project is not None else None
    rows: list[dict[str, Any]] = []
    for problem in report.problems:
        row: dict[str, Any] = {
            "kind": problem.kind,
            "name": CHECK_KIND_NAMES[problem.kind],
            "severity": problem.severity,
            "start_frame": problem.start,
            "end_frame": problem.end,
            "start_seconds": round(problem.start / report.fps, 3),
            "end_seconds": round(problem.end / report.fps, 3),
            "detail": problem.detail,
            **problem.measure,
        }
        if rate is not None:
            row["timecode"] = format_timecode(problem.start, rate)
        if problem.clip_ids:
            row["clip_ids"] = list(problem.clip_ids)
        if problem.ongoing:
            row["ongoing"] = True
        rows.append(row)
    result = _paged(rows, arguments, name="problems")
    notes = [str(result.pop("note"))] if "note" in result else []
    severities: dict[str, int] = {}
    for problem in report.problems:
        severities[problem.severity] = severities.get(problem.severity, 0) + 1
    result |= {
        "target": "timeline" if timeline else str(request.path),
        "complete": report.complete,
        "analyzed_seconds": round((report.analyzed - report.start) / report.fps, 3),
        "length_seconds": round(report.seconds, 3),
        # 調べた種類だけを並べる 調べられなかった種類は別に返す 混ぜると、GPU の無い機械で
        # 頼んだ黒いコマを「調べて無かった」と AI が読む
        "checks": [kind for kind in CHECK_KINDS if kind in report.checked],
        "summary": {
            "integrated_lufs": None if report.loudness is None else round(report.loudness, 1),
            "loudness_target_lufs": request.limits.loudness_target,
            "sample_peak_dbfs": None if report.peak_db is None else round(report.peak_db, 2),
            "counts": report.counts(),
            "severities": severities,
        },
    }
    if timeline:
        result["from_frame"] = request.start
        result["to_frame"] = request.end
        # 調べ始めたときのシーン 待つ間に人が別のシーンを開いても、結果はこのシーンの物
        result["scene"] = request.scene
    else:
        result["frame_note"] = "フレームは書き出した動画のコマの番号（0 から）"
    if report.skipped:
        result["skipped_checks"] = [kind for kind in CHECK_KINDS if kind in report.skipped]
    notes.extend(report.notes)
    if report.skipped:
        names = "・".join(CHECK_KIND_NAMES[kind] for kind in result["skipped_checks"])
        notes.append(f"{names}は調べていません 問題が無いとは限りません")
    if state.running:
        notes.append(
            f"検査の途中です（{round((report.analyzed - report.start) / report.fps, 1)} 秒まで）"
            " 調べた所までを返しました しばらくしてから同じ引数で呼ぶと続きを返します"
        )
    elif state.stopped:
        notes.append("検査は途中で止められました もう 1 度呼ぶと最初から調べ直します")
    elif not report.problems and not report.skipped:
        notes.append("問題は見つかりませんでした")
    elif not report.problems and report.checked:
        notes.append("調べた種類（checks）には問題がありませんでした")
    if request.project is not None and _project(host) != request.project:
        notes.append(
            "調べている間に作品が変わりました 結果は調べ始めたときの作品の物です"
            " 今の作品を調べるにはもう 1 度呼んでください"
        )
    if notes:
        result["note"] = " ".join(notes)
    return result


# --- 音の推移と見分け（#322 #338）---

#: 推移の点の数の既定と上限 1 点は 30 字ほどなので、上限でも返事の上限の半分ほどに収まる
DEFAULT_SOUND_POINTS = 60
MAX_SOUND_POINTS = 200
#: 無音とみなす大きさ（dBFS）の既定 話の間（静かな部屋の雑音は -50〜-60）も拾える側
#: 検査（check_output）の既定 -60 は「音が無い」不具合を探す値で、話の間を拾わない
DEFAULT_SOUND_SILENCE_DB = -50.0
#: 返す無音の最短（秒）の既定 話の区切り（0.5 秒ほどの間）を拾い、息継ぎは拾わない
DEFAULT_SOUND_MIN_SILENCE = 0.5
#: 一覧の種類 区間の見分けか、無音か
SOUND_LISTS = ("segments", "silences")
SOUND_LABEL_NAMES = {
    SPEECH: "話し声",
    MUSIC: "音楽",
    SPEECH_AND_MUSIC: "BGM の上の話し声",
    SOUND_SILENCE: "無音",
    NOISE: "雑音",
    UNCLEAR: "見分けられない",
}
#: 区間に添えるクリップの数 長い音楽の下に何十本も並べない
_CLIPS_PER_SOUND = 3
#: グラフに頭打ちの印を付ける大きさ（dBFS） 検査の音割れの既定（-0.01）と同じ
_GRAPH_CLIP_DB = -0.01
#: グラフの見分けの帯の色の説明（描く色は ui/ai_sound_graph.py）
_BAND_LEGEND = "青=話し声 橙=音楽 紫=BGM の上の話し声 灰=無音 茶=雑音 白=見分けられない"
_SOUND_LIMITS_NOTE = (
    " 話し声と音楽の見分けは、字幕（文字起こし）の重なりと、音の大きさの上下（話し声は"
    "音節と息継ぎで上下し、谷が無音近くまで落ちる 音楽は谷も鳴っている）と、話の谷に残る音で"
    "見る 外のモデルは使わない 歌は音楽として返す ラップ・笑い声・拍手・打楽器だけの音楽は"
    "外れやすく、ゲームの効果音や中の声は話し声に寄る（録画の話していない所の半分ほどを"
    "話し声と読む） 1 秒より短い切り替わりは拾わない confidence は high（字幕と合う）・"
    "medium・low 大事な所は preview_range と再生で確かめる"
)


def _same_target(one: CheckRequest, other: CheckRequest) -> bool:
    """同じ物（同じ作品の同じ範囲か、同じ書き出したファイル）を調べる頼みか しきい値は見ない"""
    return (one.path, one.stamp, one.start, one.end, one.scene) == (
        other.path,
        other.stamp,
        other.start,
        other.end,
        other.scene,
    ) and (one.project is other.project or one.project == other.project)


def _sound_target(host: EditorHost, arguments: dict[str, Any]) -> CheckRequest:
    """音だけを調べる頼みを作る（まだ頼まない） 範囲と場所はここで確かめて断る"""
    picked = {
        key: arguments[key]
        for key in ("path", "from_frame", "to_frame")
        if arguments.get(key) is not None
    }
    return _check_request(host, {**picked, "checks": sorted(SOUND_KINDS)})


def _start_sound(host: EditorHost, fresh: CheckRequest) -> CheckRequest:
    """``fresh`` を頼む 同じ物を調べた検査（check_output も）があれば、それを使い回す

    推移は検査が音を混ぜるついでに覚える（:mod:`sashimono.engine.checks.profile`）
    使い回さずに頼み直すと、AI が今頼んだ検査を止めて、同じ音をもう 1 度混ぜ直す
    """
    state = host.check_state()
    current = state.request
    if (
        current is not None
        and current.limits.needs_sound
        and state.failure is None
        and not state.stopped
        and _same_target(current, fresh)
    ):
        return current
    host.request_check(fresh)
    return fresh


def _sound_wait(arguments: dict[str, Any]) -> float:
    return _limit_number(arguments, "wait_seconds", DEFAULT_CHECK_WAIT, 0.0, MAX_CHECK_WAIT)


def _after_sound(host: EditorHost, wait: float, answer: Callable[[CheckState], object]) -> object:
    """解析が終わっていなければ待ってから（待ちきれなければ途中までで）答える"""
    state = host.check_state()
    if not state.running or wait <= 0:
        return answer(state)
    return Pending(wait=host.check_waiter(), seconds=wait, then=lambda: answer(host.check_state()))


@dataclass(frozen=True, slots=True)
class _Heard:
    """答えに使う音の推移と、添える断り書き"""

    report: CheckReport
    profile: SoundProfile
    notes: tuple[str, ...]


def _heard(host: EditorHost, request: CheckRequest, state: CheckState) -> _Heard:
    if state.request != request:
        raise ToolError("調べている間に別の検査に替わりました もう 1 度呼んでください")
    if state.failure is not None:
        raise ToolError(f"音を読めませんでした: {state.failure}")
    report = state.report
    profile = report.sound if report is not None else None
    if report is None or profile is None or profile.ticks == 0:
        if state.running:
            raise ToolError(
                "音を読み始めたところです 少し待ってからもう 1 度呼んでください"
                "（wait_seconds を長くすると、そのまま待てます）"
            )
        if state.stopped:
            raise ToolError("音の解析は途中で止められました もう 1 度呼ぶと最初から調べ直します")
        reason = " ".join(report.notes) if report is not None and report.notes else "音がありません"
        raise ToolError(f"音の推移を出せませんでした（{reason}）")
    notes: list[str] = []
    if state.running:
        notes.append(
            f"解析の途中です（{round(profile.ticks * TICK_SECONDS, 1)} 秒まで） 調べた所までを"
            "返しました しばらくしてから同じ引数で呼ぶと続きを返します"
        )
    elif state.stopped:
        notes.append("解析は途中で止められました もう 1 度呼ぶと最初から調べ直します")
    if request.project is not None and _project(host) != request.project:
        notes.append(
            "調べている間に作品が変わりました 結果は調べ始めたときの作品の物です"
            " 今の作品を調べるにはもう 1 度呼んでください"
        )
    return _Heard(report, profile, tuple(notes))


def _tick_frame(report: CheckReport, tick: int) -> int:
    """推移の区切り（100 ms）の頭が当たるフレーム 範囲の終わりを越えない"""
    return min(report.end, report.start + round(tick * TICK_SECONDS * report.fps))


def _subtitle_ticks(project: Project, start: int, fps: float, ticks: int) -> np.ndarray:
    """区切りごとに、鳴っているクリップの字幕が重なるか

    消音のトラックや無効のクリップの字幕は数えない 聞こえない声の字幕で話し声と読むと、
    BGM だけが鳴っている所を話し声と返す 置いたシーンの中も、外へ写す前に鳴るかを見る
    （:func:`heard_timeline` 写した後はクリップ ID が外側のシーンの物になり、見分けられない）
    """
    mask = np.zeros(ticks, dtype=bool)
    for subtitle in heard_timeline(project):
        low = math.floor((subtitle.start_frame - start) / fps / TICK_SECONDS)
        high = math.ceil((subtitle.end_frame - start) / fps / TICK_SECONDS)
        if high <= 0 or low >= ticks:
            continue
        mask[max(0, low) : min(ticks, high)] = True
    return mask


def _sound_segments(request: CheckRequest, heard: _Heard, silence_db: float) -> list[SoundSegment]:
    report, profile = heard.report, heard.profile
    subtitles = (
        _subtitle_ticks(request.project, report.start, report.fps, profile.ticks)
        if request.project is not None
        else None
    )
    return classify(profile, subtitles=subtitles, silence_db=silence_db)


def _lufs(value: float | None) -> float | None:
    """LUFS の値を返事の形にする 測れない（-70 以下・無音）は ``None``"""
    if value is None or not math.isfinite(value) or value <= ABSOLUTE_GATE:
        return None
    return round(value, 1)


def _sound_levels(report: CheckReport, profile: SoundProfile, points: int) -> dict[str, Any]:
    """推移を ``points`` 点に束ねる 1 点は区間の頭のフレームと、その区間の値"""
    momentary = profile.momentary()
    short = profile.short_term()
    peaks = profile.peak_db()
    count = profile.ticks
    points = max(1, min(points, count))
    rows: list[list[Any]] = []
    for index in range(points):
        low = index * count // points
        high = max(low + 1, (index + 1) * count // points)
        rows.append(
            [
                _tick_frame(report, low),
                _lufs(float(short[high - 1])),
                _lufs(float(momentary[low:high].max())),
                round(float(peaks[low:high].max()), 1),
            ]
        )
    return {
        "seconds_per_point": round(count * TICK_SECONDS / points, 2),
        "columns": ["frame", "short_term_lufs", "max_momentary_lufs", "peak_dbfs"],
        "rows": rows,
    }


def _sound_arguments(arguments: dict[str, Any]) -> tuple[float, float]:
    silence_db = _limit_number(arguments, "silence_db", DEFAULT_SOUND_SILENCE_DB, -120.0, -10.0)
    min_silence = _limit_number(
        arguments, "min_silence_seconds", DEFAULT_SOUND_MIN_SILENCE, 0.1, 600.0
    )
    return silence_db, min_silence


def _analyze_audio(host: EditorHost, arguments: dict[str, Any]) -> object:
    """音の推移・ラウドネス・無音・話し声と音楽の見分けを数で返す"""
    points = _within(arguments, "points", DEFAULT_SOUND_POINTS, 0, MAX_SOUND_POINTS)
    silence_db, min_silence = _sound_arguments(arguments)
    listing = str(arguments.get("list") or SOUND_LISTS[0])
    if listing not in SOUND_LISTS:
        raise ToolError(f"list は {' か '.join(SOUND_LISTS)} で渡してください（{listing!r}）")
    wait = _sound_wait(arguments)
    request = _start_sound(host, _sound_target(host, arguments))

    def answer(state: CheckState) -> object:
        heard = _heard(host, request, state)
        return _sound_numbers(
            request, heard, arguments, points, silence_db, min_silence, listing=listing
        )

    return _after_sound(host, wait, answer)


def _sound_numbers(
    request: CheckRequest,
    heard: _Heard,
    arguments: dict[str, Any],
    points: int,
    silence_db: float,
    min_silence: float,
    *,
    listing: str,
) -> object:
    report, profile = heard.report, heard.profile
    project = request.project
    index = ClipIndex(project) if project is not None else None
    segments = _sound_segments(request, heard, silence_db)
    by_kind: dict[str, float] = {}
    for segment in segments:
        seconds = (segment.end - segment.start) * TICK_SECONDS
        by_kind[segment.label] = round(by_kind.get(segment.label, 0.0) + seconds, 1)
    loudness_range = profile.loudness_range()
    head: dict[str, Any] = {
        "target": "timeline" if project is not None else str(request.path),
        "complete": report.complete,
        "analyzed_seconds": round(profile.ticks * TICK_SECONDS, 1),
        "length_seconds": round(report.seconds, 3),
        "summary": {
            "integrated_lufs": _lufs(profile.integrated()),
            "loudness_range_lu": None if loudness_range is None else round(loudness_range, 1),
            "max_momentary_lufs": _lufs(float(profile.momentary().max())),
            "max_short_term_lufs": _lufs(float(profile.short_term().max())),
            "sample_peak_dbfs": round(float(profile.peak_db().max()), 2),
            "loudness_target_lufs": request.limits.loudness_target,
            "seconds_by_kind": by_kind,
        },
    }
    if project is not None:
        head |= {"from_frame": request.start, "to_frame": request.end, "scene": request.scene}
    else:
        head["frame_note"] = "フレームは書き出した動画のコマの番号（0 から）"

    def clips(start: int, end: int) -> list[str]:
        if index is None:
            return []
        return [item.clip_id for item in index.sounds(start, end)][:_CLIPS_PER_SOUND]

    rows: list[dict[str, Any]] = []
    if listing == "segments":
        peaks = profile.peak_db()
        for segment in segments:
            start, end = _tick_frame(report, segment.start), _tick_frame(report, segment.end)
            row: dict[str, Any] = {
                "kind": segment.label,
                "name": SOUND_LABEL_NAMES[segment.label],
                "confidence": segment.confidence,
                "start_frame": start,
                "end_frame": end,
                "start_seconds": round(start / report.fps, 2),
                "end_seconds": round(end / report.fps, 2),
                "lufs": _lufs(profile.integrated(segment.start, segment.end)),
                "peak_dbfs": round(float(peaks[segment.start : segment.end].max()), 1),
                "speech_score": segment.speech_score,
                "music_score": segment.music_score,
            }
            if segment.subtitles is not None:
                row["subtitles"] = segment.subtitles
            rows.append(row)
    else:
        shortest = max(1, math.ceil(min_silence / TICK_SECONDS - 1e-9))
        for low, high, loudest in profile.silences(silence_db, shortest):
            start, end = _tick_frame(report, low), _tick_frame(report, high)
            rows.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "start_seconds": round(start / report.fps, 2),
                    "seconds": round((high - low) * TICK_SECONDS, 1),
                    "loudest_dbfs": round(loudest, 1),
                }
            )
    for row in rows:
        if project is not None:
            row["timecode"] = format_timecode(int(row["start_frame"]), project.rate)
        placed = clips(int(row["start_frame"]), int(row["end_frame"]))
        if placed:
            row["clip_ids"] = placed
    notes = list(heard.notes)
    if points:
        # 推移と区間を合わせて返事の上限に収める 推移は返事の半分まで 越えたら点を減らす
        wanted = points
        head["levels"] = _sound_levels(report, profile, points)
        while points > 10 and len(json_text(head)) > MAX_RESULT_CHARS // 2:
            points = points * 3 // 4
            head["levels"] = _sound_levels(report, profile, points)
        if points < wanted:
            notes.append(f"推移は返事に収まるよう {points} 点に減らした")
    notes.append(
        "LUFS が null の所は無音か -70 LUFS より静か 区間・無音のフレームはタイムラインの"
        "フレームで、preview_frame・preview_range・audio_graph の marks へそのまま渡せる"
        + _SOUND_LIMITS_NOTE
    )
    return _paged_after(head, rows, arguments, listing, " ".join(notes))


def _graph_marks(arguments: dict[str, Any], low: int, high: int) -> list[int]:
    """縦線を引くフレーム 範囲の外は寄せずに断る"""
    raw = arguments.get("marks")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ToolError("marks はフレームの配列で渡してください")
    if len(raw) > MAX_MARKS:
        raise ToolError(f"marks は {MAX_MARKS} 個までにしてください（{len(raw)} 個）")
    frames = [_integer_value(value, "marks") for value in raw]
    outside = [frame for frame in frames if not low <= frame < high]
    if outside:
        raise ToolError(
            f"marks の {', '.join(map(str, outside))} はグラフの範囲（{low}〜{high - 1}）の外です"
        )
    return frames


def _audio_graph(host: EditorHost, arguments: dict[str, Any]) -> object:
    """音量の推移を 1 枚の絵で返す 動き（preview_range）と並べて確かめる"""
    width = _within(arguments, "width", GRAPH_DEFAULT_WIDTH, GRAPH_MIN_WIDTH, MAX_SHEET_EDGE)
    height = _within(arguments, "height", GRAPH_DEFAULT_HEIGHT, GRAPH_MIN_HEIGHT, GRAPH_MAX_HEIGHT)
    silence_db, _min_silence = _sound_arguments(arguments)
    wait = _sound_wait(arguments)
    fresh = _sound_target(host, arguments)
    # 引数は解析を頼む前に確かめる 外れた引数で長い解析を走らせ、AI が頼み直す間も混ぜ続けない
    # 書き出した動画の長さは読むまで分からない その時は答える所で確かめる
    marks = _graph_marks(arguments, fresh.start, fresh.end) if fresh.path is None else None
    request = _start_sound(host, fresh)

    def answer(state: CheckState) -> object:
        heard = _heard(host, request, state)
        report = heard.report
        chosen = marks if marks is not None else _graph_marks(arguments, report.start, report.end)
        graph = _sound_graph(request, heard, width, height, chosen, silence_db)
        png = host.draw_sound_graph(graph)
        return ImageResult(png=png, caption=_graph_caption(request, heard, chosen))

    return _after_sound(host, wait, answer)


def _sound_graph(
    request: CheckRequest,
    heard: _Heard,
    width: int,
    height: int,
    marks: list[int],
    silence_db: float,
) -> SoundGraph:
    report, profile = heard.report, heard.profile
    columns = plot_width(width)
    # 途中までの解析でも横は範囲の全体に取る 測った所を全体に引き伸ばすと、時刻の目盛りが
    # 実際の位置とずれる まだの所は未解析として描く
    total = max(profile.ticks, math.ceil(report.seconds / TICK_SECONDS - 1e-9))
    missing = total - profile.ticks

    def padded(values: np.ndarray, fill: float) -> np.ndarray:
        return np.concatenate([values, np.full(missing, fill)]) if missing > 0 else values

    def x_of(tick: int) -> int:
        return MARGIN_LEFT + round(tick * columns / total)

    subtitles: tuple[tuple[int, int], ...] = ()
    if request.project is not None:
        mask = _subtitle_ticks(request.project, report.start, report.fps, profile.ticks)
        subtitles = tuple((x_of(low), x_of(high)) for low, high in spans(mask.tolist()))
    seconds = total * TICK_SECONDS
    step = time_step(seconds, columns)
    ticks: list[tuple[int, str]] = []
    moment = 0.0
    while moment <= seconds + 1e-9:
        frame = report.start + round(moment * report.fps)
        label = (
            format_timecode(frame, request.project.rate)
            if request.project is not None
            else _clock(moment)
        )
        ticks.append((MARGIN_LEFT + round(moment / seconds * columns), label))
        moment += step
    span = max(1, report.end - report.start)
    return SoundGraph(
        width=width,
        height=height,
        momentary=graph_columns(padded(profile.momentary(), -math.inf), columns),
        short_term=graph_columns(padded(profile.short_term(), -math.inf), columns, reduce="last"),
        clipped=tuple(
            MARGIN_LEFT + column
            for column in clipped_columns(
                padded(profile.peak_db(), -120.0), columns, _GRAPH_CLIP_DB
            )
        ),
        bands=tuple(
            Band(x_of(segment.start), x_of(segment.end), segment.label)
            for segment in _sound_segments(request, heard, silence_db)
        ),
        subtitles=subtitles,
        target=request.limits.loudness_target,
        time_ticks=tuple(ticks),
        marks=tuple(
            (MARGIN_LEFT + round((frame - report.start) * columns / span), str(number))
            for number, frame in enumerate(marks, start=1)
        ),
        analyzed_until=None if missing <= 0 else x_of(profile.ticks),
    )


def _clock(seconds: float) -> str:
    """書き出した動画の時刻の字 時:分:秒"""
    whole = int(seconds)
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def _graph_caption(request: CheckRequest, heard: _Heard, marks: list[int]) -> str:
    report, profile = heard.report, heard.profile
    project = request.project

    def when(frame: int) -> str:
        if project is not None:
            return f"{format_timecode(frame, project.rate)}（{frame}）"
        return f"{_clock(frame / report.fps)}（{frame}）"

    target = "タイムライン" if project is not None else f"書き出した動画 {request.path}"
    clock = "タイムコード" if project is not None else "時:分:秒"
    # 書き出した動画には字幕の区間が無い 帯の説明を書くと、無い帯を探させる
    lane = " その下の細い帯は字幕のある所" if project is not None else ""
    parts = [
        f"音量の推移 {target} {when(report.start)}〜{when(report.end)} {report.seconds:.1f} 秒",
        f"横は時間（下の字は{clock}） 縦はラウドネス（LUFS 上が大きい 横線は 10 LU ごと"
        f" 下の端は {FLOOR_LUFS:g} これより静かな所は下の端に付く）",
        "太い線は短時間（3 秒）・細い線は瞬時（400 ms）の最大",
        f"点線は音量の基準 {request.limits.loudness_target:g} LUFS",
        "上の端の赤い印は頭打ち（音割れのおそれ）",
        f"線の下の帯は話し声と音楽の見分け（{_BAND_LEGEND}）{lane}",
        f"積分ラウドネス {_lufs(profile.integrated())} LUFS"
        f" 最大のサンプル {round(float(profile.peak_db().max()), 2)} dBFS",
    ]
    if marks:
        parts.append(
            "縦線の番号: "
            + " / ".join(f"{number}) {when(frame)}" for number, frame in enumerate(marks, 1))
        )
    if not report.complete:
        parts.append("右の灰色の所はまだ解析していない")
    parts.append("数で読むなら analyze_audio 同じ範囲の動きは preview_range で見る")
    parts.extend(heard.notes)
    return " ".join(parts)


# --- 変更 ---


def _import_media(host: EditorHost, arguments: dict[str, Any]) -> object:
    raw = arguments.get("paths") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ToolError("paths はファイルパスの配列で渡してください")
    paths = [Path(entry) for entry in _string_items(raw, "paths")]
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


#: 速度の変え方（#332） content は中身（素材の範囲）を保って長さを変え、length は長さを保って
#: 流す素材の範囲を変える（設定パネルの「再生速度」と同じ）
_SPEED_KEEPS = ("content", "length")


def _source_end(project: Project, clip: Clip) -> tuple[str, Fraction] | None:
    """クリップが流す物（素材か入れ子のシーン）の名前と長さ（秒） 終わりの無い物は ``None``

    シーンも描画と音のミキサは終わりを越えた時間を空として扱うので、素材と同じく終わりがある
    見ないと、シーンを長さを保ったまま速くして終わりが黒く無音になっても AI に伝わらない
    静止画は同じ 1 枚を出し続けるので終わりが無い
    """
    if clip.scene_id is not None:
        scene = project.find_scene(clip.scene_id)
        if scene is None or scene.timeline.duration <= 0:
            return None
        return "シーン", scene.timeline.duration * project.rate.frame_duration
    media = project.find_media(clip.media_id) if clip.media_id is not None else None
    if media is None or media.is_still or media.duration <= 0:
        return None
    return "素材", media.duration


def _set_speed(host: EditorHost, arguments: dict[str, Any]) -> object:
    """クリップの再生速度を変える（#332 B6） リンクした絵と音は一緒に変わる"""
    if arguments.get("speed_percent") is None:
        raise ToolError("speed_percent に速度（%）を渡してください（100 で等倍、200 で 2 倍速）")
    percent = _limit_number(arguments, "speed_percent", 100.0, *SPEED_PERCENT_RANGE)
    keep = arguments.get("keep", "content")
    if keep not in _SPEED_KEEPS:
        raise ToolError(f"keep は {'・'.join(_SPEED_KEEPS)} のどれかです（{keep!r} が来ました）")
    clip_ids = _target_clips(host, arguments)
    project = _project(host)
    # 百分率の小数は 1000 分の 1 まで 分母を大きくすると保存の分数が長くなるだけで聞き分けられない
    speed = Fraction(str(percent)).limit_denominator(1000) / 100
    command = ChangeSpeed(clip_ids, speed, keep_length=keep == "length")
    try:
        members = speed_members(project, clip_ids)
        # 先に当ててみて断る理由を拾う ホストに渡してから断られると、ぶつかる相手と
        # 伸ばせる量が AI に届かない
        command.apply(project)
    except (KeyError, ValueError) as exc:
        raise ToolError(str(exc).strip("'\"")) from exc
    before = {clip.id: clip.duration for _, clip in members}
    host.apply_commands([command], command.label)

    after = _project(host)
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for clip_id, duration in before.items():
        located = after.timeline.locate_clip(clip_id)
        if located is None:  # pragma: no cover - 速度の命令はクリップを消さない
            continue
        track, clip = located
        rows.append(
            {
                "clip_id": str(clip.id),
                "duration_before": duration,
                "duration": clip.duration,
                "timeline_start": clip.timeline_start,
                "timeline_end": clip.timeline_end,
            }
        )
        source_end = _source_end(after, clip)
        if source_end is None:
            continue
        what, length = source_end
        over = clip.source_out(after.rate) - length
        # 長さを保つ変え方で速くすると、素材の終わりを越える AI が気付かないまま書き出すと、
        # 絵は終わりが黒くなり、音は無音になる 絵を素材の中で止めて（hold_at）いれば絵は黒く
        # ならないので言わない（言うと直す必要の無い trim_clip を勧める） 音は止めない決まり
        # レイヤー（混合トラック）のクリップは 1 本で絵と音の両方を出すので、両方を見て書く
        if over <= 0:
            continue
        held = clip.hold_at is not None and clip.hold_at <= length
        picture_lost = after.draws_picture(track, clip) and not held
        sound_lost = after.plays_sound(track, clip)
        if picture_lost and sound_lost:
            lost = "越えた所は映らず、音は無音になる"
        elif sound_lost:
            lost = "越えた所は無音になる"
        elif picture_lost:
            lost = "越えた所は何も映らない"
        else:
            continue
        warnings.append(
            f"{clip.id} は{what}の終わりを {float(over):.2f} 秒越えます {lost}ので、"
            "trim_clip で終わりを詰めるか、keep を content にしてください"
        )
    result: dict[str, object] = {
        "speed_percent": float(speed * 100),
        "keep": keep,
        "clips": rows,
        "note": (
            "リンクした絵と音も同じ速度にしました 音は高さも変わります "
            + (
                "キーフレームは中身の同じ時刻へ動かしました"
                if keep == "content"
                else "長さを保ったのでキーフレームは動かしていません"
            )
            + " 取り消し 1 回で元に戻ります"
        ),
    }
    if warnings:
        result["warnings"] = warnings
    return result


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


def _emphasis_target(host: EditorHost, arguments: dict[str, Any]) -> tuple[Track, Clip]:
    """動きを付けるクリップ 絵を描かないクリップとロックしたトラックのクリップを断る

    決まりは右クリックの〔強調の動きを付ける…〕と同じ（:func:`can_emphasize`） 断る文を
    分けるのは、AI がロックを解けば済むのか別のクリップを指すのかを選べるようにするため
    """
    track, clip = _target_clip(host, arguments)
    if track.locked:
        raise ToolError(
            f"{clip.id} のトラック {track.name} はロックされています"
            " 変えるなら set_track_state でロックを解いてから"
        )
    if not can_emphasize(_project(host), track, clip):
        raise ToolError(
            f"{clip.id} は絵を描かないクリップです（音・場面切り替え・フィルタ）"
            " 動画・画像・テキスト・図形のクリップを指してください"
        )
    return track, clip


def _emphasis_range(
    host: EditorHost, arguments: dict[str, Any], clip: Clip, *, keep: bool
) -> tuple[int, int]:
    """範囲（クリップの頭からのフレーム） タイムラインのフレームで受けて、クリップの外は断る

    端へ寄せない 寄せると、AI が狙った言葉の頭とずれた所で動き出し、ずれたことに気付かない
    """
    start = _frame_argument(arguments, "start_frame", host.playhead)
    if keep:
        if arguments.get("end_frame") is not None:
            raise ToolError(
                "keep を入れたら end_frame は渡さないでください（クリップの終わりまで残す）"
            )
        end = clip.timeline_end
    else:
        if arguments.get("end_frame") is None:
            raise ToolError("end_frame（動きを終える位置 タイムラインのフレーム）を渡してください")
        end = _frame_argument(arguments, "end_frame", 0)
    if not clip.timeline_start <= start < end <= clip.timeline_end:
        raise ToolError(
            f"範囲はクリップの中（{clip.timeline_start}〜{clip.timeline_end} フレーム）で、"
            f"start_frame < end_frame にしてください（{start}〜{end} が来ました）"
        )
    if keep and start >= clip.timeline_end - 1:
        # 最後のフレームから残しても、動き切るフレームがクリップの外になり何も動かない
        raise ToolError(
            f"keep の start_frame はクリップの最後のフレーム（{clip.timeline_end - 1}）より前に"
            "してください"
        )
    return start - clip.timeline_start, end - clip.timeline_start


def _emphasis_style(host: EditorHost, arguments: dict[str, Any]) -> tuple[int, str]:
    plain = host.emphasis_style
    ramp_ms = round(_limit_number(arguments, "ramp_ms", plain.ramp_ms, *EMPHASIS_RAMP_MS_RANGE))
    easing = arguments.get("easing", plain.easing)
    if not isinstance(easing, str) or easing not in EMPHASIS_EASINGS:
        raise ToolError(
            f"easing は {'・'.join(EMPHASIS_EASINGS)} のどれかです（{easing!r} が来ました）"
        )
    return ramp_ms, easing


def _pan_offset(host: EditorHost, arguments: dict[str, Any]) -> tuple[float, float]:
    has_offset = arguments.get("offset_x") is not None or arguments.get("offset_y") is not None
    direction = arguments.get("direction")
    if has_offset == (direction is not None):
        raise ToolError("パンは offset_x・offset_y か direction のどちらか一方を渡してください")
    if has_offset:
        offset = (
            _limit_number(arguments, "offset_x", 0.0, -4000.0, 4000.0),
            _limit_number(arguments, "offset_y", 0.0, -4000.0, 4000.0),
        )
        if offset == (0.0, 0.0):
            raise ToolError("offset_x と offset_y がどちらも 0 です")
        return offset
    if direction not in PAN_DIRECTIONS:
        raise ToolError(f"direction は {'・'.join(PAN_DIRECTIONS)} のどれかです")
    distance = _limit_number(
        arguments, "distance_px", float(host.emphasis_style.pan_px), *EMPHASIS_PAN_PX_RANGE
    )
    unit_x, unit_y = PAN_DIRECTIONS[direction]
    return unit_x * distance, unit_y * distance


def _focus(arguments: dict[str, Any]) -> tuple[float, float] | None:
    """ズームで止めておく点 どちらも省けば ``None``（絵の支点を中心に拡がる）"""
    if arguments.get("focus_x") is None and arguments.get("focus_y") is None:
        return None
    return (
        _limit_number(arguments, "focus_x", 0.0, -4000.0, 4000.0),
        _limit_number(arguments, "focus_y", 0.0, -4000.0, 4000.0),
    )


def _emphasize(host: EditorHost, arguments: dict[str, Any]) -> object:
    kind = arguments.get("kind")
    if not isinstance(kind, str) or kind not in EMPHASIS_KINDS:
        raise ToolError(f"kind は {'・'.join(EMPHASIS_KINDS)} のどれかです（{kind!r} が来ました）")
    _track, clip = _emphasis_target(host, arguments)
    keep = _flag_argument(arguments, "keep", False)
    if keep and kind not in (ZOOM, PAN):
        raise ToolError("戻さずに残せる（keep）のは zoom と pan だけです")
    start, end = _emphasis_range(host, arguments, clip, keep=keep)
    ramp_ms, easing = _emphasis_style(host, arguments)
    style = host.emphasis_style
    result: dict[str, Any] = {"clip_id": str(clip.id), "kind": kind}
    request = EmphasisRequest(kind, start, end, ramp_ms=ramp_ms, easing=easing, keep=keep)
    if kind == SHAKE:
        amplitude = _limit_number(
            arguments, "amplitude_px", float(style.shake_px), *EMPHASIS_SHAKE_PX_RANGE
        )
        request = replace(request, strength=amplitude)
        result["amplitude_px"] = amplitude
    elif kind == PAN:
        offset = _pan_offset(host, arguments)
        request = replace(request, offset=offset)
        result["offset_px"] = {"x": offset[0], "y": offset[1]}
    else:
        default, bounds = (
            (style.zoom_percent, EMPHASIS_ZOOM_PERCENT_RANGE)
            if kind == ZOOM
            else (style.pop_percent, EMPHASIS_POP_PERCENT_RANGE)
        )
        percent = _limit_number(arguments, "scale_percent", float(default), *bounds)
        focus = _focus(arguments) if kind == ZOOM else None
        request = replace(request, strength=percent, focus=focus)
        result["scale_percent"] = percent
        if focus is not None:
            result["focus_px"] = {"x": focus[0], "y": focus[1]}
    try:
        commands, shape = emphasis_commands(clip, request, _project(host).rate)
    except EmphasisError as exc:
        raise ToolError(str(exc)) from exc
    host.apply_commands(commands, emphasis_label(kind))
    offset_frames = clip.timeline_start
    result.update(
        {
            "start_frame": shape.start + offset_frames,
            "end_frame": (clip.timeline_end if keep else shape.end + offset_frames),
            "keyframes": [frame + offset_frames for frame, *_rest in shape.points],
            "ramp_ms": ramp_ms,
            "easing": easing,
            "keep": keep,
            "note": (
                "配置の値の線（揺れはランダム移動のエフェクトの幅の線）にキーフレームを打ちました"
                " 人は設定パネルとグラフエディタで点を直せます 取り消し 1 回で元に戻ります"
            ),
        }
    )
    return result


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
    _no_explicit_null(arguments, "segment_ids")
    raw_segments = arguments.get("segment_ids")
    if raw_segments is not None and not isinstance(raw_segments, list):
        raise ToolError(
            "segment_ids は字幕の ID（get_subtitles の segment_id）の配列で渡してください"
        )
    segments = (
        {SegmentId(s) for s in _string_items(raw_segments, "segment_ids")} if raw_segments else None
    )

    # 画面の〔焼き込み〕と同じ既定 作品の高さに合わせて縮める
    settings = project.settings
    wrap = subtitle_wrap_width(settings.width, host.subtitle_wrap_share)
    base = TEXT.create(**burn_defaults(settings.height, wrap))
    template: SubtitleTemplate = base
    look_name = str(arguments.get("look_name", "") or "")
    if look_name and arguments.get("template_clip_id"):
        raise ToolError("見た目は look_name か template_clip_id のどちらか 1 つで選んでください")
    if look_name:
        # 画面の〔焼き込み〕の一覧で選ぶのと同じ決まりで着せる（#308）
        look_kind = str(arguments.get("look_kind", "alias") or "alias")
        try:
            if look_kind == TEMPLATE:
                # 棚の配布物は名前が重なりうる（分類違いの同じ名前が何本もある）ので、
                # list_looks の template_id で選ぶ 画面の〔焼き込み〕の一覧と同じ棚
                choice = template_choice(_shelf_entry(host, {"template_id": look_name}))
            else:
                choice = find_choice(
                    host.saved_looks,
                    look_kind,
                    look_name,
                    str(arguments.get("look_category", "") or ""),
                )
            template = subtitle_look(choice, base, host.look_rules)
        except LookError as exc:
            raise ToolError(str(exc)) from exc
    elif arguments.get("template_clip_id"):
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
    cut = RippleCut(ranges)
    commands: list[Command] = [cut]
    result: dict[str, object] = {
        "cuts": len(ranges),
        "removed_frames": removed,
        "removed_seconds": round(float(removed * project.rate.frame_duration), 2),
    }
    if _flag_argument(arguments, "smooth", False):
        # 詰めてできたつなぎ目の処理を同じ 1 回の取り消しに入れる（#332） 詰めた後のクリップは
        # 当てるまで ID が決まらないので、フレームだけを持つ命令（SmoothCuts）で後から探す
        # 結果に載せる中身は、詰めた後の形を手元で作って同じ決まりで組んだ物
        style = _seam_style(host, arguments)
        frames = seams_after_cuts(ranges)
        transition = seam_transition(style)
        # 詰めてできたつなぎ目は同じ素材を飛び飛びにつないだ対 同じフレームで前から区切れて
        # いた別のトラックの関係の無いクリップは処理しない
        jumps = SeamScope(jumps_only=True)
        plan = plan_smoothing(cut.apply(project), frames, style, transition, jumps)
        commands.append(SmoothCuts(frames, style, transition, jumps))
        result["smoothing"] = _seam_result(plan, style, clip_ids=False)
    host.apply_commands(commands, f"無音カット: {len(ranges)} か所")
    return result


# --- ジャンプカットのつなぎ目（#332 B6） ---
#
# 詰めたつなぎ目へ、人が手で付けるのと同じ物を付ける（core/seams.py） クロスフェードは
# 場面切り替えのクリップ、寄りは配置の拡大率の欄、音は音量フェードの欄 人は設定パネルと
# タイムラインで同じ物を直せる 1 回の呼び出しは 1 回の apply_commands で、取り消し 1 回で戻る

#: 1 回に処理するつなぎ目の上限 数千か所を一度に組むと編集画面が止まる 1 時間の話を
#: 細かく詰めても数百か所に収まる
MAX_SEAMS = 2000
#: 結果に並べる付けなかった所の上限 全部並べると会話が長くなるだけで、理由は数種類しかない
_SKIPPED_SHOWN = 20


def _whole_within(
    arguments: dict[str, Any], key: str, default: int, bounds: tuple[int, int]
) -> int:
    """範囲のある整数の引数 省けば ``default`` 小数や範囲の外は、受け付ける範囲を書いて断る

    丸めて受けると 100.5 ms が黙って 100 ms になり、頼んだのと違う長さで付く
    """
    value = _whole_number(arguments, key, default)
    low, high = bounds
    if not low <= value <= high:
        raise ToolError(f"{key} は {low}〜{high} で渡してください（{value} が来ました）")
    return value


def _seam_style(host: EditorHost, arguments: dict[str, Any]) -> SeamStyle:
    """引数のつなぎ目の処理 省いた物は本人の設定（:attr:`EditorHost.seam_style`）"""
    base = host.seam_style
    video = arguments.get("video", base.video)
    if video not in SEAM_VIDEOS:
        raise ToolError(f"video は {'・'.join(SEAM_VIDEOS)} のどれかです（{video!r} が来ました）")
    style = SeamStyle(
        video=str(video),
        crossfade_ms=_whole_within(
            arguments, "crossfade_ms", base.crossfade_ms, SEAM_CROSSFADE_MS_RANGE
        ),
        zoom_percent=_whole_within(
            arguments, "zoom_percent", base.zoom_percent, SEAM_ZOOM_PERCENT_RANGE
        ),
        audio_fade_ms=_whole_within(
            arguments, "audio_fade_ms", base.audio_fade_ms, SEAM_AUDIO_FADE_MS_RANGE
        ),
    )
    if style.video == SEAM_NONE and style.audio_fade_ms == 0:
        raise ToolError("video が none で audio_fade_ms が 0 なので、付ける物がありません")
    return style


def _seam_result(plan: SeamPlan, style: SeamStyle, *, clip_ids: bool = True) -> dict[str, object]:
    """つなぎ目の処理の結果 ``clip_ids`` が偽なら寄せたクリップの ID を出さない

    無音カットと一緒に組んだときは、手元で作った形の ID と、実際に当てた形の ID が違う
    （詰めて割ったクリップの ID は当てるたびに新しく作る） 違う ID を返すと AI が指し損なう
    """
    result: dict[str, object] = {
        "video": style.video,
        "handled": len(plan.handled),
        "audio_faded": len(plan.audio),
        "audio_fade_ms": style.audio_fade_ms,
    }
    if style.video == SEAM_CROSSFADE:
        result["crossfade_ms"] = style.crossfade_ms
        result["crossfades"] = [
            {"frame": frame, "frames": length} for frame, length in plan.crossfades
        ]
    elif style.video == SEAM_ZOOM:
        result["zoom_percent"] = style.zoom_percent
        result["zoomed"] = len(plan.zoomed)
        if clip_ids:
            result["zoomed_clip_ids"] = [str(clip_id) for clip_id in plan.zoomed]
    if plan.skipped:
        result["skipped"] = [
            {"frame": frame, "reason": reason} for frame, reason in plan.skipped[:_SKIPPED_SHOWN]
        ]
        if len(plan.skipped) > _SKIPPED_SHOWN:
            result["skipped_more"] = len(plan.skipped) - _SKIPPED_SHOWN
    return result


def _seam_scope(project: Project, tracks: tuple[TrackId, ...] | None, named: object) -> SeamScope:
    """処理する対の決まり 探したつなぎ目は同じ素材を飛び飛びにつないだ対だけ、位置を名指し
    （``named``）したら別の素材へ切り替えた対も入れる トラックを渡したら、そのトラックの
    クリップとリンクした相手の対だけ

    フレームだけで処理すると、同じフレームで区切れている別のトラックの関係の無いクリップまで
    寄せたりフェードを入れたりし、そこの短いクリップのせいでクロスフェードごと断られる
    """
    jumps_only = named is None
    if tracks is None:
        return SeamScope(jumps_only=jumps_only)
    on_track = tuple(
        clip.id for track in project.timeline.tracks if track.id in tracks for clip in track.clips
    )
    return replace(touching_scope(project, on_track), jumps_only=jumps_only)


def _smooth_cuts(host: EditorHost, arguments: dict[str, Any]) -> object:
    """つなぎ目を目立たなくする 位置を渡さなければ、同じ素材を飛び飛びにつないだ所を探す"""
    project = _project(host)
    _no_explicit_null(arguments, "frames")
    raw = arguments.get("frames")
    track_id = arguments.get("track_id")
    tracks: tuple[TrackId, ...] | None = None
    if track_id is not None:
        if project.timeline.find_track(TrackId(str(track_id))) is None:
            raise ToolError(
                f"トラックが見つかりません: {track_id}（list_tracks で確かめてください）"
            )
        tracks = (TrackId(str(track_id)),)
    if raw is not None:
        if not isinstance(raw, list) or not raw:
            raise ToolError("frames はつなぎ目のフレームの配列で渡してください（省くと探します）")
        frames = tuple(sorted({_integer_value(value, "frames") for value in raw}))
        if frames[0] < 1:
            raise ToolError("frames は 1 以上です（0 フレームの前にクリップは無い）")
    else:
        # 範囲は整数のフレームで受ける 小数を切り捨てて探すと、頼んだのと違う範囲を黙って見る
        start = _frame_argument(arguments, "start_frame", 0)
        if start < 0:
            raise ToolError(f"start_frame は 0 以上です（{start} が来ました）")
        end = (
            None
            if arguments.get("end_frame") is None
            else _frame_argument(arguments, "end_frame", 0)
        )
        if end is not None and end <= start:
            raise ToolError("end_frame は start_frame より後にしてください")
        frames = find_seams(project, start=start, end=end, track_ids=tracks)
        if not frames:
            raise ToolError(
                "ジャンプカットのつなぎ目（同じ素材を飛び飛びにつないだ所）が見つかりません"
                " 別の素材へ切り替えた所なら frames に位置を渡してください"
            )
    if len(frames) > MAX_SEAMS:
        raise ToolError(
            f"つなぎ目が {len(frames)} か所あります 1 回に {MAX_SEAMS} か所までなので、"
            "start_frame と end_frame で分けて呼んでください"
        )
    style = _seam_style(host, arguments)
    plan = plan_smoothing(
        project, frames, style, seam_transition(style), _seam_scope(project, tracks, raw)
    )
    if not plan.commands:
        reasons = "、".join(dict.fromkeys(reason for _, reason in plan.skipped))
        raise ToolError(
            "どのつなぎ目にも付けられませんでした"
            + (f"（{reasons}）" if reasons else "（前後のクリップが接している所がありません）")
        )
    host.apply_commands(list(plan.commands), f"つなぎ目を目立たなくする: {len(plan.handled)} か所")
    result = {"seams": len(frames), **_seam_result(plan, style)}
    if plan.missing:
        result["not_seams"] = list(plan.missing[:_SKIPPED_SHOWN])
    return result


# --- 音量とミックス（#329 B3） ---
#
# どれも人の操作と同じ命令を通す 音量はクリップの固定の音量の欄（設定パネルの「音量」と
# タイムラインの音量の線と同じ所 :func:`volume_command`）、置くのはドラッグ＆ドロップと同じ
# :func:`place_media` 1 回の呼び出しは 1 回の apply_commands なので、取り消し 1 回で戻る

#: BGM を敷き詰めるときに置く本数の上限 短い効果音を長い範囲へ並べるような頼み違いで、
#: 何千本も置いて画面が固まるのを防ぐ
MAX_LOOP_PIECES = 500


def _sound_clip(project: Project, clip_id: object, *, editing: bool = True) -> tuple[Track, Clip]:
    """音を鳴らすクリップ 絵だけのクリップと、変える（``editing``）ならロックしたトラックの
    クリップを断る

    絵のクリップの音量を変えても何も鳴らないので、AI は効いたと思い込む ロックは人が
    触らせないために掛けた物なので、AI も変えない 聞くだけのクリップ（ダッキングの声）は
    ロックしていても使う 断ると、声のトラックを守るためにロックした人が BGM を下げられない
    """
    track, clip = _require_clip(project, str(clip_id))
    if not project.plays_sound(track, clip):
        raise ToolError(
            f"{clip_id} は音を鳴らさないクリップです（映像トラックの絵など）"
            " 音声のクリップを指してください（list_clips の track_kind）"
        )
    if editing and track.locked:
        raise ToolError(
            f"{clip_id} のトラック {track.name} はロックされています"
            " 変えるなら set_track_state でロックを解いてから"
        )
    return track, clip


def _track_sound_clips(
    project: Project, track_id: object, *, editing: bool = True
) -> list[tuple[Track, Clip]]:
    """トラックの音を鳴らすクリップすべて"""
    track = next((t for t in project.timeline.tracks if str(t.id) == str(track_id)), None)
    if track is None:
        raise ToolError(f"トラックが見つかりません: {track_id}（list_tracks で確かめてください）")
    clips = [
        _sound_clip(project, clip.id, editing=editing)
        for clip in track.clips
        if project.plays_sound(track, clip)
    ]
    if not clips:
        raise ToolError(f"トラック {track.name} に音を鳴らすクリップがありません")
    return clips


def _picked_sound_clips(
    host: EditorHost,
    arguments: dict[str, Any],
    ids_key: str,
    track_key: str,
    *,
    editing: bool = True,
) -> list[tuple[Track, Clip]]:
    """``ids_key`` のクリップか ``track_key`` のトラックのクリップ どちらか 1 つだけを受ける"""
    project = _project(host)
    # 明示の null を「渡していない」と読むと、もう片方の引数（トラックごと）へ黙って回る
    _no_explicit_null(arguments, ids_key)
    has_ids = arguments.get(ids_key) is not None
    has_track = arguments.get(track_key) is not None
    if has_ids == has_track:
        raise ToolError(f"{ids_key} か {track_key} のどちらか 1 つを渡してください")
    if has_track:
        return _track_sound_clips(project, arguments[track_key], editing=editing)
    clip_ids = _clip_id_list(host, arguments, ids_key)
    if not clip_ids:
        raise ToolError(f"{ids_key} が空です")
    return [_sound_clip(project, clip_id, editing=editing) for clip_id in clip_ids]


def _volume_row(clip_id: ClipId, value: AnimatedValue) -> dict[str, Any]:
    """返事に載せる音量の線の要約 点の全部は返さない（BGM 1 本で何百にもなる）"""
    if not value.keyframes:
        return {"clip_id": str(clip_id), "volume_percent": round(value.static, 2)}
    levels = [keyframe.value for keyframe in value.keyframes]
    return {
        "clip_id": str(clip_id),
        "keyframes": len(levels),
        "max_percent": round(max(levels), 2),
        "min_percent": round(min(levels), 2),
    }


def _set_volume(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    targets = [_sound_clip(project, clip_id) for clip_id in _target_clips(host, arguments)]
    has_percent = arguments.get("volume_percent") is not None
    has_gain = arguments.get("gain_db") is not None
    has_pan = arguments.get("pan_percent") is not None
    if has_percent and has_gain:
        raise ToolError("volume_percent と gain_db はどちらか 1 つにしてください")
    if not (has_percent or has_gain or has_pan):
        raise ToolError("volume_percent・gain_db・pan_percent のどれかを渡してください")
    percent = _limit_number(arguments, "volume_percent", 100.0, 0.0, VOLUME_LIMIT)
    gain = _limit_number(arguments, "gain_db", 0.0, -60.0, 24.0)
    pan = (
        AnimatedValue(static=_limit_number(arguments, "pan_percent", 0.0, -100.0, 100.0))
        if has_pan
        else None
    )
    commands: list[Command] = []
    rows: list[dict[str, Any]] = []
    stopped: list[str] = []
    for _track, clip in targets:
        before = clip_volume(clip)
        after: AnimatedValue | None = None
        if has_percent:
            # 決めた値にするときは点を捨てる 点を残すと、次のフレームで線の値へ戻って効かない
            # （設定パネルの欄で値を打ったときと同じ）
            after = AnimatedValue(static=percent)
        elif has_gain:
            after = scaled_volume(before, db_to_ratio(gain))
            wanted = loudest(before) * db_to_ratio(gain)
            if wanted > VOLUME_LIMIT:
                stopped.append(str(clip.id))
        commands.extend(volume_commands(clip, after, pan))
        rows.append(
            {
                **_volume_row(clip.id, after if after is not None else before),
                **({"pan_percent": pan.static} if pan is not None else {}),
            }
        )
    host.apply_commands(commands, f"音量を変更: {len(targets)} 本")
    result: dict[str, Any] = {"clips": rows}
    if stopped:
        result["note"] = (
            f"{', '.join(stopped)} は上限の {VOLUME_LIMIT:g}% で止まりました"
            " それ以上大きくするなら素材の音を上げるか、ほかの音を下げてください"
        )
    return result


def _duck_audio(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    ducked = _picked_sound_clips(host, arguments, "clip_ids", "track_id")
    voices = _picked_sound_clips(host, arguments, "voice_clip_ids", "voice_track_id", editing=False)
    # 鳴らない声（ミュート・ソロで外れたトラック・無効のクリップ）では下げない 書き出しで
    # 声が聞こえないのに BGM だけが下がる
    voices = [(track, clip) for track, clip in voices if sounds_now(project, track, clip)]
    # 置いたシーンは話している所を拾えないので声から外す 外さないと、ナレーションと
    # シーンが並んだトラックを声に選んだだけで全体が断られる
    voices = [(track, clip) for track, clip in voices if speaks(project, clip)]
    if not voices:
        raise ToolError(
            "話している所を拾える声のクリップがありません 置いたシーンは使えず、"
            "書き出しで鳴らない声（ミュート・ソロで外れたトラックか、無効のクリップ）も使えません"
        )
    both = {clip.id for _, clip in ducked} & {clip.id for _, clip in voices}
    if both:
        # 自分の声で自分を下げると、話している所だけ声が小さくなる
        raise ToolError(f"下げるクリップと声のクリップが重なっています: {', '.join(both)}")
    plain = host.duck_shape
    shape = DuckShape(
        depth_db=_limit_number(arguments, "depth_db", plain.depth_db, *DUCK_DEPTH_DB_RANGE),
        attack_ms=round(
            _limit_number(arguments, "attack_ms", plain.attack_ms, *DUCK_ATTACK_MS_RANGE)
        ),
        release_ms=round(
            _limit_number(arguments, "release_ms", plain.release_ms, *DUCK_RELEASE_MS_RANGE)
        ),
    )
    source = str(arguments.get("speech_from") or "auto")
    if source not in SPEECH_SOURCES:
        raise ToolError(f"speech_from は {'・'.join(SPEECH_SOURCES)} のどれかです")
    threshold = _limit_number(arguments, "threshold_db", -40.0, -90.0, -10.0)

    def waveform_of(media: MediaItem, stream: int) -> Waveform | None:
        return host.waveform(media, stream)

    try:
        speech = voice_speech(project, voices, waveform_of, source=source, threshold_db=threshold)
    except WaveformPendingError as exc:
        host.analyze(exc.media)
        raise ToolError(str(exc)) from exc
    except SpeechError as exc:
        raise ToolError(f"{exc}（speech_from を waveform にすると波形から拾います）") from exc
    if not speech:
        raise ToolError(
            "声のクリップに話している所が見つかりません threshold_db を下げるか、"
            "字幕を起こしてから呼んでください"
        )

    base = arguments.get("base_volume_percent")
    plan = plan_ducking(
        [clip for _, clip in ducked],
        speech,
        shape=shape,
        rate=project.rate,
        base=None
        if base is None
        else _limit_number(arguments, "base_volume_percent", 100.0, 0.0, VOLUME_LIMIT),
    )
    commands = list(plan.commands)
    rows = [_volume_row(clip.id, curve) for clip, curve in plan.curves]
    untouched = [str(clip.id) for clip in plan.untouched]
    if not commands:
        raise ToolError("下げるクリップの範囲に、話している所がありません")
    host.apply_commands(commands, f"BGM を声に合わせて下げる: {len(commands)} 本")
    result: dict[str, Any] = {
        "clips": rows,
        "depth_db": shape.depth_db,
        "attack_ms": shape.attack_ms,
        "release_ms": shape.release_ms,
        "speech_spans": len(merge_ranges(speech)),
        "note": (
            "音量の線を置き換えました（前の点は消えます フェードの欄はそのまま）"
            " 人はタイムラインの音量の線で点を直せます 取り消し 1 回で元に戻ります"
        ),
    }
    if untouched:
        result["untouched_clip_ids"] = untouched
    return result


def _normalize_loudness(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    # 明示の null を省いたのと同じに読むと、選んだつもりの 1 本ではなく作品全体が変わる
    _no_explicit_null(arguments, "clip_ids")
    if arguments.get("clip_ids") is None:
        # 省いたら作品全体（書き出しで鳴る音） 選択を使わないのは、「全体の音量をそろえて」で
        # たまたま選んでいた 1 本だけが変わるのを防ぐため
        # 書き出しで鳴る音だけ（ミキサと同じくミュート・ソロ・無効を見る） ソロで外れた
        # トラックの音まで入れると、測るときにソロを外すので、鳴らない音まで混ぜて測る
        targets = [_sound_clip(project, clip.id) for _, clip in audible_clips(project)]
        if not targets:
            raise ToolError("タイムラインに鳴っている音のクリップがありません")
    else:
        clip_ids = _clip_id_list(host, arguments)
        if not clip_ids:
            raise ToolError("clip_ids が空です 作品全体をそろえるなら省いてください")
        targets = [_sound_clip(project, clip_id) for clip_id in clip_ids]
        # 無効にしたクリップはミキサが鳴らさないので測りに入らない 入らない物に同じ dB を
        # 掛けると、有効に戻したときにそろえていない音量で鳴る 上限 400% の計算からも外す
        targets = [(track, clip) for track, clip in targets if clip.enabled]
        if not targets:
            raise ToolError("どのクリップも無効です 有効にしてから呼んでください")
    target = _limit_number(arguments, "target_lufs", float(host.loudness_target), -40.0, -5.0)
    ceiling = _limit_number(arguments, "peak_ceiling_db", -1.0, -20.0, 0.0)
    wait = _limit_number(arguments, "wait_seconds", DEFAULT_CHECK_WAIT, 0.0, MAX_CHECK_WAIT)
    start = min(clip.timeline_start for _, clip in targets)
    end = max(clip.timeline_end for _, clip in targets)
    request = CheckRequest(
        CheckLimits(loudness_target=target, kinds=frozenset({LOUDNESS})),
        project=measure_project(project, (clip.id for _, clip in targets)),
        start=start,
        end=end,
        scene=str(host.active_scene) if host.active_scene is not None else None,
    )
    host.request_check(request)

    def finish() -> object:
        return _apply_loudness(host, project, request, [clip for _, clip in targets], ceiling)

    if not host.check_state().running or wait <= 0:
        return finish()
    return Pending(wait=host.check_waiter(), seconds=wait, then=finish)


def _apply_loudness(
    host: EditorHost,
    project: Project,
    request: CheckRequest,
    clips: list[Clip],
    ceiling: float,
) -> object:
    """測り終えた音量から、全部のクリップへ同じ dB を掛ける 測り終えていなければ何も変えない"""
    state = host.check_state()
    if state.request != request:
        raise ToolError("測っている間に別の検査に替わりました もう 1 度呼んでください")
    if state.failure is not None:
        raise ToolError(f"音量を測れませんでした: {state.failure}")
    if state.running:
        report = state.report
        done = 0.0 if report is None else (report.analyzed - report.start) / report.fps
        return {
            "measuring": True,
            "measured_seconds": round(done, 1),
            "note": (
                "音量を測っている途中です まだ何も変えていません"
                " 同じ引数でもう 1 度呼ぶと、測り終えた所から続けて変えます"
            ),
        }
    if state.stopped:
        raise ToolError("音量を測るのが止められました もう 1 度呼ぶと測り直します")
    report = state.report
    if report is None or report.loudness is None:
        raise ToolError(
            "音量を測れませんでした 静かすぎるか、0.4 秒より短い音です（積分ラウドネスは"
            " -70 LUFS より静かな所を数えない）"
        )
    if _project(host) != project:
        raise ToolError(
            "測っている間に作品が変わりました 何も変えていません もう 1 度呼ぶと測り直します"
        )
    measured = report.loudness
    decided = loudness_gain(
        clips,
        measured=measured,
        peak_db=report.peak_db,
        target=request.limits.loudness_target,
        ceiling=ceiling,
    )
    gain = decided.gain_db
    limited = list(decided.limited_by)
    result: dict[str, Any] = {
        "measured_lufs": round(measured, 1),
        "target_lufs": request.limits.loudness_target,
        "sample_peak_dbfs": None if report.peak_db is None else round(report.peak_db, 2),
        "clips": len(clips),
    }
    if abs(gain) < 0.05:
        result |= {"gain_db": 0.0, "note": "すでに基準に合っています 何も変えていません"}
        return result
    commands = gain_commands(clips, gain)
    host.apply_commands(commands, f"音量を {request.limits.loudness_target:g} LUFS へそろえる")
    result |= {
        "gain_db": round(gain, 2),
        "expected_lufs": round(measured + gain, 1),
        "expected_peak_dbfs": None if report.peak_db is None else round(report.peak_db + gain, 2),
    }
    notes = ["どのクリップにも同じだけ掛けました（声と BGM の釣り合いは変えない）"]
    if limited:
        result["limited_by"] = limited
        if "peak" in limited:
            notes.append(f"音の山が {ceiling:g} dBFS を超えるので、基準の手前で止めました")
        if "volume_limit" in limited:
            notes.append(f"音量の欄の上限 {VOLUME_LIMIT:g}% で止めました")
    result["note"] = " ".join(notes)
    return result


def _fixed_param(clip: Clip, kind: str, name: str, value: float) -> Clip:
    """置く前のクリップの固定の項目の値を 1 つ変える（まだ置いていないので命令にしない）"""
    effects = tuple(
        effect.with_param(name, AnimatedValue(static=value))
        if effect.fixed and effect.kind == kind
        else effect
        for effect in clip.effects
    )
    return replace(clip, effects=effects)


def _place_sound(host: EditorHost, arguments: dict[str, Any]) -> object:
    project = _project(host)
    media = _require_media(project, str(arguments.get("media_id", "")))
    if not media.has_audio:
        raise ToolError(f"{media.name} に音がありません")
    if media.has_video or media.is_still:
        raise ToolError(
            f"{media.name} は動画か画像です 音だけの素材を指すか、place_media で置いてください"
        )
    at = _frame_argument(arguments, "at_frame", host.playhead)
    if at < 0:
        raise ToolError("at_frame は 0 以上です")
    preferred: TrackId | None = None
    if arguments.get("track_id") is not None:
        track = next(
            (t for t in project.timeline.tracks if str(t.id) == str(arguments["track_id"])), None
        )
        if track is None or track.kind is TrackKind.VIDEO:
            raise ToolError("track_id は音声トラックか混合トラックを指してください（list_tracks）")
        preferred = track.id
    until = arguments.get("until_frame")
    if until is not None:
        until = _frame_argument(arguments, "until_frame", 0)
        if until <= at:
            raise ToolError("until_frame は at_frame より後ろにしてください")
    loop = _flag_argument(arguments, "loop", True)
    volume = arguments.get("volume_percent")
    level = _limit_number(arguments, "volume_percent", 100.0, 0.0, VOLUME_LIMIT)
    fade_in = _limit_number(arguments, "fade_in_seconds", 0.0, 0.0, 60.0)
    fade_out = _limit_number(arguments, "fade_out_seconds", 0.0, 0.0, 60.0)

    working = project
    pieces: list[list[Command]] = []
    cursor = at
    while True:
        placed = place_media(
            working, [media], at_frame=cursor, track_id=preferred, split_audio=host.splits_media
        )
        adds = [command for command in placed if isinstance(command, AddClip)]
        if not adds:
            raise ToolError(f"{media.name} は長さが無いので置けません")
        end = max(command.clip.timeline_end for command in adds)
        if until is not None and end > until:
            # 最後の 1 本は範囲の終わりで切る 画面で右端を詰めたのと同じ長さになる
            placed = [
                AddClip(c.track_id, replace(c.clip, duration=until - c.clip.timeline_start))
                if isinstance(c, AddClip)
                else c
                for c in placed
            ]
            end = until
        for command in placed:
            working = command.apply(working)
        pieces.append(placed)
        preferred = adds[0].track_id
        cursor = end
        if until is None or not loop or cursor >= until:
            break
        if len(pieces) >= MAX_LOOP_PIECES:
            raise ToolError(
                f"{MAX_LOOP_PIECES} 本を超えて並べることになります until_frame を確かめてください"
            )

    commands: list[Command] = []
    for index, placed in enumerate(pieces):
        for command in placed:
            if not isinstance(command, AddClip):
                commands.append(command)
                continue
            clip = with_fixed_items(command.clip, sound=True)
            if volume is not None:
                clip = _fixed_param(clip, VOLUME_EFFECT_KIND, "volume", level)
            if index == 0 and fade_in > 0:
                clip = _fixed_param(clip, FADE_EFFECT_KIND, "fade_in", fade_in)
            if index == len(pieces) - 1 and fade_out > 0:
                clip = _fixed_param(clip, FADE_EFFECT_KIND, "fade_out", fade_out)
            commands.append(AddClip(command.track_id, clip))
    host.apply_commands(commands, f"音を置く: {media.name}")
    added = [command for command in commands if isinstance(command, AddClip)]
    return {
        "placed": len(added),
        "clip_ids": [str(command.clip.id) for command in added],
        "track_ids": list(dict.fromkeys(str(command.track_id) for command in added)),
        "from_frame": at,
        "to_frame": max(command.clip.timeline_end for command in added),
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


#: 人が手で変えた段を AI が戻さないときの断り（2026-10-18 利用者の決定 #333）
#: 「動き: 対象」の形の文は伏せ字で後ろが消えるので、案内を先に書き、段の名前は最後に付ける
PERSON_UNDO_REFUSAL = "人が後から手で変えた段があるので戻しません 戻すなら Ctrl+Z で"
PERSON_REDO_REFUSAL = "人が手で変えた段なのでやり直しません やり直すなら Ctrl+Y で"


def _undo(host: EditorHost, arguments: dict[str, Any]) -> object:
    """人の〔元に戻す〕を何回か押すのと同じ 指示の途中なら、この指示の 1 手から戻す（#333）

    指示の中の手は :attr:`Document.steps_in_checkpoint` に積まれていて、戻し切ると前の指示の
    1 段（「AI: 〜」）をまとめて戻す 履歴は人の Ctrl+Z / Ctrl+Y と同じ 1 本なので、AI が
    戻した物は人が Ctrl+Y でやり直せる 人が手で変えた段に当たったら、そこで止める
    """
    document = host.document
    # 文字の "false" を真と読むと、1 手のつもりの頼みで指示の手が全部戻る
    whole = _flag_argument(arguments, "this_instruction", False)
    if whole:
        if "steps" in arguments:
            # 両方を受けると、どちらを守ったのか AI に分からない 片方だけ当てて黙らない
            raise ToolError(
                "steps と this_instruction は一緒に使えません どちらか一方にしてください"
            )
        return _undo_this_instruction(host)
    steps = max(1, int(arguments.get("steps", 1)))
    undone, blocked = _step_history(host, steps, forward=False)
    if not undone:
        if blocked is not None:
            raise ToolError(f"{PERSON_UNDO_REFUSAL} 人の段 {blocked}")
        raise ToolError("戻せる操作がありません")
    result: dict[str, object] = {"undone": undone, "can_redo": document.can_redo}
    if blocked is not None:
        result["stopped"] = f"{PERSON_UNDO_REFUSAL} ここで止めました"
    return result


#: 指示が人の変更で割れ、前半が人の段の下に残ったときの知らせ
PERSON_SPLIT_NOTE = (
    "この指示の前半は、人が後から手で変えた段の下にあるので戻していません 戻すなら Ctrl+Z で"
)


def _undo_this_instruction(host: EditorHost) -> object:
    """今の指示で変えた物を戻す 人の変更で割れた前半の段も、人の段の上にある間は戻す

    数を先に数えて戻すと、人の変更で割れた指示の前半（段の外へ先に積んだ段）を数えず、
    後半だけ戻して「戻した」と言ってしまう 1 段ずつ、指示の物かを見て戻す
    """
    document = host.document
    undone: list[str] = []
    # 戻すたびに 1 段減るので、履歴の段数より多くは回らない 念のため上限を置く
    for _ in range(len(document.history_labels) + 1):
        if not document.undo_in_checkpoint:
            break
        label = document.undo_label
        if label is None:
            break
        host.undo()
        undone.append(label)
    left = document.parts_left_in_checkpoint
    if not undone:
        if left:
            raise ToolError(f"{PERSON_UNDO_REFUSAL} 人の段 {document.undo_label}")
        raise ToolError(
            "この指示ではまだ何も変えていません 前の指示を戻すなら、"
            "this_instruction を付けずに undo を呼んでください"
        )
    result: dict[str, object] = {"undone": undone, "can_redo": document.can_redo}
    if left:
        result["stopped"] = PERSON_SPLIT_NOTE
    return result


def _redo(host: EditorHost, arguments: dict[str, Any]) -> object:
    """人の〔やり直す〕を何回か押すのと同じ 取り消した後に別の編集をすると、やり直せない

    人が手で変えた段（人が Ctrl+Z で戻した自分の変更）はやり直さない
    """
    steps = max(1, int(arguments.get("steps", 1)))
    redone, blocked = _step_history(host, steps, forward=True)
    if not redone:
        if blocked is not None:
            raise ToolError(f"{PERSON_REDO_REFUSAL} 人の段 {blocked}")
        raise ToolError(
            "やり直せる操作がありません 取り消した後に別の編集をすると、やり直しは消えます"
        )
    result: dict[str, object] = {"redone": redone, "can_redo": host.document.can_redo}
    if blocked is not None:
        result["stopped"] = f"{PERSON_REDO_REFUSAL} ここで止めました"
    return result


def _step_history(host: EditorHost, steps: int, *, forward: bool) -> tuple[list[str], str | None]:
    """履歴を ``steps`` 段だけ辿り、辿った段の名前と、止めた人の段の名前を返す

    端に着いたら止める 人が手で変えた段に当たったら、そこで止めて名前を返す
    ホストの口（人のメニューと同じ処理）を通す 文書を直に辿ると画面が前の形のまま残る
    """
    document = host.document
    moved: list[str] = []
    for _ in range(steps):
        label = document.redo_label if forward else document.undo_label
        if label is None:
            break
        by = document.redo_by if forward else document.undo_by
        if by is Author.PERSON:
            return moved, label
        if forward:
            host.redo()
        else:
            host.undo()
        moved.append(label)
    return moved, None


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


def _write_target(host: EditorHost, raw: object) -> Path:
    """ファイルを書く道具の書く先 場所の決まりで確かめる（:func:`writable_path`）

    守りの効かない OS で断るのは、引数の確かめを全部済ませた後（:func:`_refuse_elsewhere`）
    先に断ると、どの OS でも出るはずの断り（場所の外・範囲が無い など）が隠れる
    """
    return writable_path(host, raw)


def _refuse_elsewhere() -> None:
    """守りの効かない OS では、確かめの箱を出す前に断る（#327） 確かめの中身を作る最後に呼ぶ"""
    refuse_where_the_folder_cannot_be_held()


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
    path = _write_target(host, str(_current_project_file(host)))
    _refuse_elsewhere()
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


def _kept(consent: Consent) -> dict[str, str]:
    """置き換えた前のファイルを残した場所 AI が人に伝えられるよう、答えに入れる

    前のファイルは消さずに同じフォルダの決まった名前で残す（:mod:`sashimono.core.io.atomic`）
    """
    basis = consent.basis
    if basis is None or not basis.kept:
        return {}
    return {
        "previous_kept_as": str(basis.kept[-1]),
        "previous_note": "置き換える前のファイルを消さずに残しました 古い物から片付けます"
        "（バックアップの世代数と同じ本数）",
    }


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
    return {"saved": str(path), "note": note, **_kept(consent)}


def _save_as_consent(host: EditorHost, arguments: dict[str, Any]) -> Consent:
    raw = arguments.get("path")
    _require_suffix(raw, (SUFFIX,), "作品")
    path = _write_target(host, raw)
    detail = " ".join(
        part
        for part in (_replacing(path, "ファイル"), "以後はこの場所の作品として編集します")
        if part
    )
    _refuse_elsewhere()
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
    return {
        "saved": str(path),
        "project_path": str(host.project_path),
        "note": note,
        **_kept(consent),
    }


def _subtitle_plan(
    host: EditorHost, arguments: dict[str, Any]
) -> tuple[Path, tuple[int, int] | None, str]:
    """字幕の書き出しの引数を、画面の字幕パネルの〔書き出し…〕と同じ選び方で確かめる

    相手は字幕パネルと同じく開いているタイムライン 範囲の既定もパネルと同じ（範囲を
    決めてあれば範囲、無ければ全体） 返すのは書く場所・範囲・確かめに出す中身
    """
    raw = arguments.get("path")
    _require_suffix(raw, SUBTITLE_SUFFIXES, "字幕のファイル")
    path = _write_target(host, raw)
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
    _refuse_elsewhere()
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
    return {
        "written": str(written),
        "range": None if frame_range is None else list(frame_range),
        **_kept(consent),
    }


def _snapshot_plan(host: EditorHost, arguments: dict[str, Any]) -> tuple[Path, int, str]:
    """静止画の引数を確かめる 描くのは画面の〔静止画を保存…〕と同じく開いているタイムライン"""
    raw = arguments.get("path")
    _require_suffix(raw, (".png",), "静止画")
    path = _write_target(host, raw)
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
    _refuse_elsewhere()
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
    return {"saved": str(path), "frame": drawn, **_kept(consent)}


# --- ナレーション（#335） ---
#
# 2026-10-20 に利用者が決めたこと（#335 の「決めたこと」）
# - 声は本人が入れて起動した VOICEVOX 互換のエンジン（VOICEVOX・AivisSpeech）に作らせる
#   部品は同梱せず、呼ぶのは自分の機械の中の HTTP だけ（engine/audio/voicevox.py）
# - WAV は作品のフォルダの narration\ へ書く 保存していない作品では断り、保存を案内する
# - 書く・読み込む・置くを 1 段にまとめ、取り消し 1 回で戻す（書いた WAV は消さない #341）
# - 使った声の表記（クレジット）を作品に覚え、書き出しのときに知らせる
# エンジンとの話は AI のスレッドで待つ（Pending） 編集画面のスレッドで待つと、声を作る
# 数秒〜数十秒の間、画面が固まる

#: 書く WAV の名前の頭 後ろに 3 桁の番号を付け、空いている一番小さい番号を使う
#: 台本の文を名前に入れない 名前は結果と伏せ字の表を通って外へ出るので、本人の文が
#: 名前と場所として送られてしまう
NARRATION_PREFIX = "narration_"
#: 台本から付けた字幕の印（Transcript.model） 起こした字幕と見分ける
NARRATION_MODEL = "narration-script"
#: 番号の上限 1 つの作品でこれを超えて作ることは無い 超えたら断り、フォルダを片付けてもらう
MAX_NARRATION_FILES = 999
#: 声の一覧を待つ秒 自分の機械の中なので、動いていればすぐ返る
VOICE_LIST_WAIT = 15.0
#: 声を作り終えるのを待つ上限（秒） 遅い機械の CPU で上限の長さの台本を作る時間より長く取る
NARRATION_WAIT = 300.0

#: エンジンの場所ごとの、list_voices で読んだ声の一覧 確かめの窓に声の名前と表記を出すのに使う
#: 確かめの中身は編集画面のスレッドで作るので、そこでエンジンへつながない（動いていない
#: エンジンへつなぐと、Windows では断られるまで 2 秒ほど画面が止まる）
_VOICE_CATALOGS: dict[str, Catalog] = {}
_CATALOG_LOCK = threading.Lock()


def remember_voice_catalog(catalog: Catalog) -> None:
    """読んだ声の一覧を控える list_voices と人の窓（〔ナレーションを作る…〕）が読んだ物"""
    with _CATALOG_LOCK:
        _VOICE_CATALOGS[catalog.url] = catalog


def _known_catalog(url: str) -> Catalog | None:
    with _CATALOG_LOCK:
        return _VOICE_CATALOGS.get(url)


class _Job[T]:
    """エンジンとの話を裏のスレッドで進める 待つのは AI のスレッド（:class:`Pending`）

    ``deadline`` を過ぎたら、塊の間で止まる 待つのをやめた道具（時間切れ・中断）の後で、
    エンジンを使い続けない
    """

    def __init__(self, work: Callable[[Callable[[], bool]], T], seconds: float) -> None:
        self._done = threading.Event()
        self._result: T | None = None
        self._error: EngineError | None = None
        self._deadline = time.monotonic() + seconds + 5.0
        self._abandoned = threading.Event()
        thread = threading.Thread(
            target=self._run, args=(work,), name="sashimono-narration", daemon=True
        )
        thread.start()

    def _stopped(self) -> bool:
        return self._abandoned.is_set() or time.monotonic() > self._deadline

    def _run(self, work: Callable[[Callable[[], bool]], T]) -> None:
        try:
            self._result = work(self._stopped)
        except EngineError as exc:
            self._error = exc
        finally:
            self._done.set()

    def wait(self, seconds: float) -> bool:
        return self._done.wait(seconds)

    def take(self, what: str) -> T:
        """終わっていれば結果を返す 終わっていなければ止めて断る"""
        if not self._done.is_set():
            self._abandoned.set()
            raise ToolError(f"{what}が時間内に終わりませんでした 何も書いていません")
        if self._error is not None:
            raise ToolError(str(self._error))
        assert self._result is not None
        return self._result


def _voice_engine(settings: NarrationSettings) -> VoiceEngine:
    try:
        return VoiceEngine(settings.url)
    except EngineError as exc:
        raise ToolError(f"{exc}（表示 → 設定… の「ナレーション: エンジンの場所」）") from exc


def _voice_label(speaker: Speaker, style: Style) -> str:
    return f"{speaker.name}（{style.name}）"


def _list_voices(host: EditorHost, arguments: dict[str, Any]) -> object:
    """エンジンの声の一覧を読む 声の規約は、頼んだ声か既定の声の 1 つだけ読む

    全部の声の規約を読むと、声の数だけエンジンへ頼み、返事も道具の結果の上限を越える
    """
    settings = host.narration_settings
    engine = _voice_engine(settings)
    wanted = arguments.get("voice_id")
    if wanted is not None and (isinstance(wanted, bool) or not isinstance(wanted, int)):
        raise ToolError("voice_id は声の ID（整数）で渡してください")
    terms_for = settings.voice if wanted is None else wanted

    def work(stopped: Callable[[], bool]) -> tuple[Catalog, str]:
        del stopped
        catalog = engine.catalog()
        found = catalog.find(terms_for)
        policy = ""
        if found is not None:
            try:
                policy = engine.policy(found[0])
            except EngineError:
                # 規約が読めなくても一覧は返す 規約はエンジンのアプリでも読める
                policy = ""
        return catalog, policy

    job = _Job(work, VOICE_LIST_WAIT)

    def then() -> object:
        catalog, policy = job.take("エンジンの声の一覧を読むの")
        remember_voice_catalog(catalog)
        return _voices_result(settings, catalog, policy, terms_for, arguments)

    return Pending(wait=job.wait, seconds=VOICE_LIST_WAIT, then=then)


def _voices_result(
    settings: NarrationSettings,
    catalog: Catalog,
    policy: str,
    terms_for: int,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        {
            "speaker": speaker.name,
            "credit": catalog.credit(speaker),
            "styles": [{"voice_id": style.id, "name": style.name} for style in speaker.styles],
        }
        for speaker in catalog.speakers
    ]
    default = catalog.find(settings.voice)
    found = catalog.find(terms_for)
    common: dict[str, Any] = {
        "engine": catalog.engine_name or catalog.brand,
        "url": catalog.url,
        "default_voice_id": default[1].id if default is not None else None,
        "default_voice": _voice_label(*default) if default is not None else None,
        "ai_may_choose_voice": settings.ai_picks_voice,
        "terms_voice": _voice_label(*found) if found is not None else None,
        # エンジンが返した声の規約の文 アプリは中身を判断しない（読むのは本人） 企業での
        # 利用や非商用だけの声があるので、使う前にユーザーへ伝える
        "terms": policy,
    }
    notes: list[str] = []
    if default is None:
        notes.append(
            f"本人の設定の既定の声（ID {settings.voice}）がこのエンジンの一覧にありません"
            " 表示 → 設定… で選び直してもらうか、voice_id を渡してください"
        )
    if found is None and terms_for != settings.voice:
        notes.append(f"voice_id {terms_for} はこのエンジンの一覧にありません")
    if not settings.ai_picks_voice:
        notes.append("本人の設定で、AI は既定の声だけを使います（voice_id を渡すと断ります）")
    if notes:
        common["notes"] = notes
    page = {**common, **_paged(rows, arguments, name="voices")}
    while len(json_text(page)) > MAX_RESULT_CHARS and page["count"] > 1:
        smaller = {**arguments, "limit": max(1, page["count"] * 3 // 4)}
        page = {**common, **_paged(rows, smaller, name="voices")}
    return page


@dataclass(frozen=True, slots=True)
class NarrationPlan:
    """確かめに出す中身と、作って置くのに要る物 引数と作品から決まり、エンジンへはつながない

    AI の道具（generate_narration）と人の窓（〔字幕〕→〔ナレーションを作る…〕）が同じ物を使う
    """

    text: str
    settings: NarrationSettings
    speaker: Speaker
    style: Style
    credit: str
    tuning: Tuning
    at_frame: int
    track: TrackId | None
    #: 台本から字幕を付けるか（本人の設定 人の窓ではその 1 回だけ変えられる）
    subtitles: bool
    path: Path
    detail: str


def _narration_text(arguments: dict[str, Any]) -> str:
    raw = arguments.get("text")
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError("text（読ませる台本の文）を渡してください")
    text = raw.strip()
    if len(text) > TEXT_LIMIT:
        # 文の中身は断りの文に入れない（外へ返るので、本人の文を送ることになる）
        raise ToolError(
            f"text が長すぎます（{len(text)} 文字 1 回に {TEXT_LIMIT} 文字まで）"
            " 場面ごとに分けて、何回かに分けて頼んでください"
        )
    if not split_script(text):
        raise ToolError("text に読める文がありません")
    return text


def _narration_voice(
    arguments: dict[str, Any], settings: NarrationSettings, catalog: Catalog, *, person: bool
) -> tuple[Speaker, Style]:
    """声を決める 人が窓で選んだ声は「AI が声を選んでよいか」の設定に掛からない"""
    raw = arguments.get("voice_id")
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int) or raw < 0):
        raise ToolError(
            "voice_id は声の ID（0 以上の整数 list_voices の voice_id）で渡してください"
        )
    default = catalog.find(settings.voice)
    chosen_elsewhere = raw is not None and (default is None or raw != default[1].id)
    if chosen_elsewhere and not person and not settings.ai_picks_voice:
        raise ToolError(
            "本人の設定で、AI は既定の声だけを使います voice_id を省いてください"
            "（声を変えるなら、本人に 表示 → 設定… で既定の声を選んでもらいます）"
        )
    found = catalog.find(raw) if raw is not None else default
    if found is None:
        if raw is None:
            raise ToolError(
                f"本人の設定の既定の声（ID {settings.voice}）がエンジンの一覧にありません"
                " list_voices の voice_id を渡すか、本人に 表示 → 設定… で"
                "選び直してもらってください"
            )
        raise ToolError(
            f"voice_id {raw} はエンジンの一覧にありません（list_voices で確かめてください）"
        )
    return found


def _narration_track(project: Project, arguments: dict[str, Any]) -> TrackId | None:
    raw = arguments.get("track_id")
    if raw is None:
        return None
    track = next((t for t in project.timeline.tracks if str(t.id) == str(raw)), None)
    if track is None or track.kind is TrackKind.VIDEO:
        raise ToolError("track_id は音声トラックか混合トラックを指してください（list_tracks）")
    return track.id


def _narration_file(host: EditorHost, project_path: Path) -> tuple[Path, bool]:
    """書く WAV の場所と、narration フォルダをこれから作るか

    名前は空いている一番小さい番号 同じ名前の物を置き換えない（前に作った声を潰さない）
    フォルダがまだ無ければ、フォルダの場所そのものを書ける場所の決まりに通す
    （作品のフォルダの中か・リンクで外へ出ないか）
    """
    folder = project_path.parent / NARRATION_FOLDER
    if folder.exists() and not folder.is_dir():
        raise ToolError(
            f"作品のフォルダに {NARRATION_FOLDER} という名前のファイルがあるので、"
            "ナレーションを書くフォルダを作れません 名前を変えるか動かしてもらってください"
        )
    making = not folder.exists()
    if making:
        resolved = _write_target(host, str(folder))
        return resolved / f"{NARRATION_PREFIX}001.wav", True
    for number in range(1, MAX_NARRATION_FILES + 1):
        candidate = folder / f"{NARRATION_PREFIX}{number:03d}.wav"
        if not candidate.exists():
            return _write_target(host, str(candidate)), False
    raise ToolError(
        f"{folder} に WAV が {MAX_NARRATION_FILES} 本あり、新しい名前を付けられません"
        " 使わない物を片付けてもらってください"
    )


def narration_plan(
    host: EditorHost, arguments: dict[str, Any], *, person: bool = False
) -> NarrationPlan:
    """台本・声・置く所・書く場所を確かめる エンジンへはつながない（声の一覧は控えから）

    ``person`` は人の窓から頼んだか 人は自分で声を選ぶので、AI に声を選ばせない設定に掛けない
    字幕の有る無しは、人の窓だけが ``subtitles`` で 1 回ごとに変えられる（AI は本人の設定に従う
    2026-10-20 利用者の決定）
    """
    text = _narration_text(arguments)
    project_path = host.project_path
    if project_path is None:
        raise ToolError(
            "作品をまだ保存していないので、ナレーションの WAV を書く場所（作品のフォルダの"
            f" {NARRATION_FOLDER}）がありません save_project_as で保存するか、人に"
            "〔名前を付けて保存〕で保存してもらってから、もう 1 度頼んでください"
        )
    settings = host.narration_settings
    engine = _voice_engine(settings)
    catalog = _known_catalog(engine.url)
    if catalog is None:
        raise ToolError(
            "先に list_voices を呼んで、エンジンが動いていることと声の一覧・規約を確かめて"
            "ください（声の名前と表記を確かめの窓に出すのに使います）"
        )
    speaker, style = _narration_voice(arguments, settings, catalog, person=person)
    tuning = Tuning(
        speed_percent=_whole_within(
            arguments, "speed_percent", settings.speed_percent, VOICE_SPEED_RANGE
        ),
        volume_percent=_whole_within(
            arguments, "volume_percent", settings.volume_percent, VOLUME_PERCENT_RANGE
        ),
        pitch=_whole_within(arguments, "pitch", 0, PITCH_RANGE),
        intonation_percent=_whole_within(
            arguments, "intonation_percent", 100, INTONATION_PERCENT_RANGE
        ),
    )
    subtitles = settings.subtitles
    if person and "subtitles" in arguments:
        subtitles = bool(arguments["subtitles"])
    project = _project(host)
    at = _frame_argument(arguments, "at_frame", host.playhead)
    if at < 0:
        raise ToolError("at_frame は 0 以上です")
    track = _narration_track(project, arguments)
    path, making = _narration_file(host, project_path)
    credit = catalog.credit(speaker)
    parts = [
        f"声: {_voice_label(speaker, style)}"
        f"（{catalog.engine_name or catalog.brand} {catalog.url}）",
        f"速さ {tuning.speed_percent}% / 音量 {tuning.volume_percent}%"
        f" / 高さ {tuning.pitch} / 抑揚 {tuning.intonation_percent}%",
        f"置く所: {format_timecode(at, project.rate)}（{at} フレーム）から"
        + (f" トラック {track}" if track is not None else " 空いた音声トラック"),
        "字幕: 台本から文ごとに付けます" if subtitles else "字幕: 付けません",
        f"表記: {credit}（作品に覚え、書き出しのときに知らせます 声ごとの規約は list_voices と"
        "エンジンのアプリで確かめてください）",
        f"作品のフォルダに {NARRATION_FOLDER} フォルダを作ります" if making else "",
        f"読ませる文（{len(text)} 文字 この文は自分の機械の中のエンジンへだけ渡します）:\n{text}",
    ]
    _refuse_elsewhere()
    return NarrationPlan(
        text=text,
        settings=settings,
        speaker=speaker,
        style=style,
        credit=credit,
        tuning=tuning,
        at_frame=at,
        track=track,
        subtitles=subtitles,
        path=path,
        detail="\n".join(part for part in parts if part),
    )


def narration_consent(
    host: EditorHost, arguments: dict[str, Any], *, person: bool = False
) -> Consent:
    """確かめに出す中身 人の窓では窓の OK が確かめに当たり、ここで書く先の身元を採る"""
    plan = narration_plan(host, arguments, person=person)
    return Consent(
        action="ナレーションの声を作り、WAV を書いてタイムラインに置きます",
        target=str(plan.path),
        detail=plan.detail,
        basis=expect_now(plan.path),
    )


def _narration_consent(host: EditorHost, arguments: dict[str, Any]) -> Consent:
    return narration_consent(host, arguments)


class NarrationJob:
    """裏で声を作る 1 回 AI は :class:`Pending` で、人の窓はタイマーで終わりを待つ"""

    def __init__(self, plan: NarrationPlan, seconds: float = NARRATION_WAIT) -> None:
        engine = _voice_engine(plan.settings)

        def work(stopped: Callable[[], bool]) -> Spoken:
            return engine.synthesize(plan.text, plan.style.id, plan.tuning, stopped=stopped)

        self.plan = plan
        self._job = _Job(work, seconds)

    def wait(self, seconds: float) -> bool:
        """終わるまで最長で引数の秒だけ待つ 終わっていれば真"""
        return self._job.wait(seconds)

    def take(self) -> Spoken:
        """作った声 終わっていなければ止めて断り、エンジンの失敗は理由を書いて断る"""
        return self._job.take("声を作るの")


def _generate_narration(host: EditorHost, arguments: dict[str, Any], consent: Consent) -> object:
    plan = narration_plan(host, arguments)
    job = NarrationJob(plan)

    def _place_when_made() -> object:
        # 待った後の続き 書く先は人が許した確かめの場所（consent.target）で、作り直さない
        return write_and_place_narration(host, plan, consent, job.take(), host.apply_commands)

    return Pending(wait=job.wait, seconds=NARRATION_WAIT, then=_place_when_made)


def write_and_place_narration(
    host: EditorHost,
    plan: NarrationPlan,
    consent: Consent,
    spoken: Spoken,
    apply: Callable[[list[Command], str], None],
) -> dict[str, object]:
    """作った声を書き、読み込み、置き、字幕と表記を付ける 1 回の ``apply`` で 1 段にまとめる

    AI の道具は ``host.apply_commands``（AI の段）、人の窓は人の段で当てる関数を渡す
    書く先は許した確かめの場所（``consent.target``） 待つ間に作品の場所が変わっていれば、
    許した場所は今の作品のフォルダではないので断る 書いた WAV は、置くのに失敗しても消さない
    （元の素材を壊さない #341 の考え方 人が素材として読み込み直せる）
    """
    target = Path(consent.target)
    folder = host.make_narration_folder()
    if os.path.normcase(str(folder.resolve())) != os.path.normcase(str(target.parent)):
        raise ToolError(
            "声を作っている間に作品の場所が変わったので、WAV を書いていません"
            " もう 1 度頼んでください"
        )
    try:
        with pinned_place(host, str(target)) as path:
            host.write_narration(path, spoken.wav, expect=_basis(consent))
    except OSError as exc:
        raise ToolError(f"ナレーションの WAV を書けませんでした: {exc}") from exc
    media = host.probe(path)
    if not media.has_audio:
        raise ToolError(f"書いた WAV に音がありません: {path}")
    if plan.subtitles:
        media = media.with_transcript(_script_transcript(spoken))
    project = _project(host)
    placed = place_media(
        project,
        [media],
        at_frame=plan.at_frame,
        track_id=plan.track,
        split_audio=host.splits_media,
    )
    commands: list[Command] = [
        AddClip(command.track_id, with_fixed_items(command.clip, sound=True))
        if isinstance(command, AddClip)
        else command
        for command in placed
    ]
    added = [command for command in commands if isinstance(command, AddClip)]
    if not added:
        raise ToolError(f"作った声の長さが無いので置けません（WAV は残っています: {path}）")
    commands.append(AddCredits((plan.credit,)))
    apply(commands, f"ナレーション: {path.name}")
    host.analyze(media)
    return {
        "file": str(path),
        "media_id": str(media.id),
        "clip_ids": [str(command.clip.id) for command in added],
        "track_ids": list(dict.fromkeys(str(command.track_id) for command in added)),
        "from_frame": plan.at_frame,
        "to_frame": max(command.clip.timeline_end for command in added),
        "seconds": round(wav_seconds(spoken.wav), 3),
        "voice": _voice_label(plan.speaker, plan.style),
        "voice_id": plan.style.id,
        "credit": plan.credit,
        # 字幕の文は台本そのものなので数だけ返す（字幕と文字は get_subtitles で同意を取って読む）
        "subtitle_lines": len(spoken.lines) if plan.subtitles else 0,
        "note": (
            "取り消し 1 回で、読み込み・置いた声・字幕・表記がまとめて戻ります（書いた WAV は"
            "残ります） 表記は書き出しのときに人へ知らせます 動画の概要欄などに入れてもらって"
            "ください"
        ),
    }


#: 字幕の時刻の細かさ（秒） 素材の中の時刻は分数で持つので、浮動小数の長さを丸めて渡す
_SUBTITLE_STEP = 1000


def _script_transcript(spoken: Spoken) -> Transcript:
    """台本の文ごとの字幕 作った声の文ごとの長さに合わせ、素材の頭から並べる

    起こし直さずに台本の文をそのまま使う（読み上げた文なので、起こすより確か） 人が
    書いた文として印を付け、字幕起こしをやり直しても上書きされないようにする
    """
    segments: list[TranscriptSegment] = []
    start = Fraction(0)
    for line in spoken.lines:
        end = start + Fraction(round(line.seconds * _SUBTITLE_STEP), _SUBTITLE_STEP)
        segments.append(TranscriptSegment(start=start, end=end, text=line.text, edited=True))
        start = end
    return Transcript(tuple(segments), language="ja", model=NARRATION_MODEL)


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
    path = _write_target(host, raw)
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
    _refuse_elsewhere()
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
    credits = _credit_note(host)
    if wait > 0:
        pending = _awaiting_export(host, wait)
        if not credits:
            return pending
        return replace(pending, then=lambda: {**host.export_status(), **credits})
    return {
        "started": str(plan.request.path),
        "next": "書き出しは裏で進みます export_status で進み具合を読めます"
        "（wait_seconds を付けると、その秒数まで終わるのを待ちます）",
        **credits,
    }


def _credit_note(host: EditorHost) -> dict[str, object]:
    """作品が覚えている表記（#335）を書き出しの答えに添える 無ければ空

    人には書き出しの窓とステータスバーで知らせる AI にも返し、書き出した後に人へ伝えてもらう
    （声の規約が表記を求める 忘れて困るのは本人）
    """
    credits = host.document.project.credits
    if not credits:
        return {}
    return {
        "credits": list(credits),
        "credit_note": (
            "この作品は音声合成の声を使っています 動画の概要欄などに表記を入れるよう、"
            "ユーザーに伝えてください"
        ),
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


# --- 保存した見た目と棚（#328） ---

#: 一覧に出す種類 list_looks の kind で絞る
_LOOK_KINDS = ("preset", "alias", "template")


def _list_looks(host: EditorHost, arguments: dict[str, Any]) -> object:
    """保存したプリセット・自作のエイリアス・テンプレートの棚を 1 つの一覧で返す

    人は設定パネルの〔プリセット…〕と〔追加〕→〔エイリアス〕のメニューで同じ物を見る
    棚の中身はここでは読まない（数百本の棚で、一覧を出すだけで全部を開くことになる）
    棚のファイルの場所は返さない 選ぶのは ``template_id`` で足り、場所には本人の名前が入りうる
    """
    kind = str(arguments.get("kind") or "")
    if kind and kind not in _LOOK_KINDS:
        raise ToolError(f"kind は {'・'.join(_LOOK_KINDS)} のどれかです: {kind}")
    wanted = str(arguments.get("contains") or "")
    rows: list[dict[str, Any]] = []
    saved = host.saved_looks
    if kind in ("", "preset"):
        rows.extend(
            {
                "kind": "preset",
                "name": preset.name,
                "category": preset.category,
                # 中身の種類 テキストの見た目を図形へ当てても中身は当たらない（足したエフェクト
                # だけが当たる） AI が当てる先を選ぶ手掛かり
                "source": preset.source.kind if preset.source is not None else None,
                "effects": [effect.kind for effect in (*preset.effects, *preset.after_effects)],
            }
            for preset in saved.presets
        )
    if kind in ("", "alias"):
        rows.extend(
            {
                "kind": "alias",
                "name": alias.name,
                "category": alias.category,
                "source": alias.clip.source.kind if alias.clip.source is not None else None,
                "duration": alias.clip.duration,
            }
            for alias in saved.aliases
        )
    if kind in ("", "template"):
        for entry in host.template_shelf:
            row: dict[str, Any] = {
                "kind": "template",
                "template_id": template_choice(entry).key,
                "name": entry.name,
                # 棚のフォルダの名前 本人の置き場の名前なので、名前と場所を断った人には伏せる
                # （項目の名前 folder で伏せる :func:`~sashimono.ai.outbound.redact`）
                "folder": entry.folder,
                "app": entry.source,
            }
            if entry.error:
                row["error"] = entry.error
            rows.append(row)
    if wanted:
        rows = [row for row in rows if wanted in str(row["name"])]
    return _paged(rows, arguments, name="looks")


def _find_look(host: EditorHost, kind: str, name: str, category: str = "") -> Preset | Alias:
    """保存したプリセットかエイリアスを名前で探す 字幕の見た目を選ぶのと同じ探し方（#308）"""
    if not name:
        raise ToolError("name を指定してください（list_looks で名前を確かめられます）")
    try:
        item = find_choice(host.saved_looks, kind, name, category).item
    except LookError as exc:
        raise ToolError(str(exc)) from exc
    assert isinstance(item, Preset | Alias)  # find_choice は preset と alias だけを返す
    return item


def _apply_preset(host: EditorHost, arguments: dict[str, Any]) -> object:
    """保存したプリセットを当てる 設定パネルの〔プリセット…〕と同じ当て方（本人の設定の当て方）"""
    preset = _find_look(
        host,
        "preset",
        str(arguments.get("name") or ""),
        str(arguments.get("category") or ""),
    )
    assert isinstance(preset, Preset)
    targets = _target_clips(host, arguments)
    commands, changed = preset_commands_for(
        _project(host), targets, preset, options=host.look_rules.preset_options
    )
    if not commands:
        raise ToolError(
            f"「{preset.name}」を当てても変わる所がありません"
            "（中身の種類が違うか、もう同じ見た目になっている）"
        )
    label = f"プリセット: {preset.name}"
    if len(changed) > 1:
        label = f"{label}（{len(changed)} 本）"
    host.apply_commands(commands, label)
    return {
        "applied": preset.name,
        "clip_ids": [str(clip_id) for clip_id in changed],
        "unchanged": [str(clip_id) for clip_id in targets if clip_id not in changed],
    }


def _look_track(host: EditorHost, arguments: dict[str, Any]) -> TrackId | None:
    """置く先のトラック 無い ID は断る（人は右クリックしたトラックを取り違えない）"""
    raw = str(arguments.get("track_id") or "")
    if not raw:
        return None
    track = _project(host).timeline.find_track(TrackId(raw))
    if track is None:
        raise ToolError(f"トラックが見つかりません: {raw}（list_tracks で確かめてください）")
    return track.id


def _look_frame(host: EditorHost, arguments: dict[str, Any]) -> int:
    raw = arguments.get("at_frame")
    if raw is None:
        return host.playhead
    return max(0, _integer_value(raw, "at_frame"))


def _place_alias(host: EditorHost, arguments: dict[str, Any]) -> object:
    """保存したエイリアスを置く 右クリックの〔追加〕→〔エイリアス〕と同じ insert_clip を通す"""
    alias = _find_look(host, "alias", str(arguments.get("name") or ""))
    assert isinstance(alias, Alias)
    frame = _look_frame(host, arguments)
    commands = insert_clip(
        _project(host),
        alias.instantiate(),
        at_frame=frame,
        track_id=_look_track(host, arguments),
    )
    host.apply_commands(commands, f"エイリアスを置く: {alias.name}")
    placed = next(c for c in commands if isinstance(c, AddClip))
    return {
        "placed": alias.name,
        "clip_id": str(placed.clip.id),
        "track_id": str(placed.track_id),
        "at_frame": placed.clip.timeline_start,
        "duration": placed.clip.duration,
    }


def _shelf_entry(host: EditorHost, arguments: dict[str, Any]) -> TemplateEntry:
    wanted = str(arguments.get("template_id") or "")
    entry = next(
        (item for item in host.template_shelf if template_choice(item).key == wanted), None
    )
    if entry is None:
        raise ToolError(
            f"棚にそのテンプレートがありません: {wanted}"
            "（list_looks の kind=template で template_id を確かめてください）"
        )
    return entry


def _shelf_objects(entry: TemplateEntry) -> list[MappedObject]:
    """棚の 1 本を読む 読めない理由は画面と同じ文で返す 道具ごと落ちると AI は直し方が分からない"""
    try:
        return entry.load()
    except (*TemplateError, OSError) as exc:
        raise ToolError(f"{entry.label} を読み込めません: {exc}") from exc


def _probe_quietly(host: EditorHost) -> Callable[[Path], MediaItem | None]:
    """棚や .exo が指す素材を開く手 開けない素材が 1 つあるだけで全体を止めない（画面と同じ）"""

    def probe(path: Path) -> MediaItem | None:
        try:
            return host.probe(path)
        except ToolError:
            return None

    return probe


def _place_template(host: EditorHost, arguments: dict[str, Any]) -> object:
    """棚のテンプレートを置く 右クリックの〔追加〕→〔エイリアス〕の棚の項目と同じ手順"""
    entry = _shelf_entry(host, arguments)
    objects = _shelf_objects(entry)
    placement = template_placement(
        objects,
        _project(host),
        _look_frame(host, arguments),
        _probe_quietly(host),
        near=entry.path.parent,
        track_id=_look_track(host, arguments),
    )
    if not placement.commands:
        raise ToolError(
            f"{entry.label} に置けるオブジェクトがありません"
            "（エフェクトだけのテンプレートは apply_template でテキストに着せる）"
        )
    # 元のレイヤーの所が埋まっていれば、置く段取りが一番手前の新しいトラックへ並べ替えてある
    # （2026-10-17 本人の決定 :mod:`sashimono.compat.layers`） 重なって断られることは無い
    host.apply_commands(list(placement.commands), "テンプレートを配置")
    # 解析と控えは取り消しても止まらない 入ったことを確かめてから頼む（画面と同じ）
    for media in placement.plan.added:
        host.analyze(media)
    return {
        "placed": len(placement.placed),
        "clip_ids": [str(clip_id) for clip_id in placement.placed],
        "missing_media": len(placement.plan.missing),
    }


def _apply_template(host: EditorHost, arguments: dict[str, Any]) -> object:
    """棚のテンプレートをテキストに着せる 棚の〔選択中のクリップに適用〕と同じ

    文字と長さは今のクリップのまま残す（着せ替え :func:`~sashimono.compat.catalog.restyle`）
    """
    entry = _shelf_entry(host, arguments)
    _, clip = _target_clip(host, arguments)
    objects = _shelf_objects(entry)
    # 効果だけのテンプレート（YMM4 のアニメーション効果など）はテキスト以外にも着せられる
    # 先にテキストかを見て断ると、画面の棚ではできる着せ方が AI だけできない
    commands = restyle(objects, clip, keep_wrap=host.look_rules.keep_wrap)
    if not commands:
        raise ToolError(f"{entry.label}: {restyle_refusal(objects, clip)}")
    host.apply_commands(commands, "テンプレートを適用")
    return {"applied": entry.label, "clip_id": str(clip.id)}


#: import_objects が読む拡張子 互換の〔オブジェクトを読み込む…〕の窓の種類と、棚が読む AviUtl2 の
#: ``.object``
_OBJECT_SUFFIXES = (".exo", ".exa", ".exo2", ".exa2", ".object")


def _import_objects(host: EditorHost, arguments: dict[str, Any]) -> object:
    """AviUtl のオブジェクトのファイルを読み込む 互換の〔オブジェクトを読み込む…〕と同じ手順

    人は窓でファイルを選ぶ 置き方は画面と同じで、元のレイヤーの並びのまま頭（0 フレーム）から
    """
    raw = str(arguments.get("path") or "")
    if not raw:
        raise ToolError("path を指定してください（.exo .exa .exo2 .exa2 .object）")
    # 読むのは AviUtl のオブジェクトのファイルだけ 人は窓の「すべてのファイル」で自分で選べるが、
    # AI には選ぶ人がいない 拡張子を見ないと、本人のメモや設定まで開いて読もうとする
    _require_suffix(raw, _OBJECT_SUFFIXES, "読み込むオブジェクトのファイル")
    source = Path(raw)
    if not source.is_file():
        raise ToolError(f"ファイルがありません: {source}")
    try:
        exo = load_exo(source)
    except ExoParseError as exc:
        raise ToolError(f"読み込めません: {exc}") from exc
    found, commands = exo_commands(exo, source, _project(host), _probe_quietly(host))
    if not commands:
        raise ToolError("読み込めるオブジェクトがありませんでした")
    host.apply_commands(commands, f"AviUtl から読み込み: {source.name}")
    for media in found.items:
        host.analyze(media)
    placed = [c.clip.id for c in commands if isinstance(c, AddClip)]
    return {
        "placed": len(placed),
        "clip_ids": [str(clip_id) for clip_id in placed],
        "missing_media": len(found.missing),
    }


#: 道具の結果に入りうる物の組（:attr:`Operation.sends`）
#: 位置・長さ・ID・エフェクトの値だけを返す道具 素材の名前が混ざるのはエラーの文くらいで、
#: そこは名前を送らないと決めた人には伏せ字にして送る（:func:`sashimono.ai.outbound.redact`）
_STRUCTURE = frozenset({Outbound.PROJECT})
#: 素材の名前か場所を返す道具
_WITH_FILES = frozenset({Outbound.PROJECT, Outbound.FILES})
#: 字幕の文かテキストの本文を返す道具（引数で受けた本文を結果で返す道具も 断られたら動かさない）
_WITH_TEXT = frozenset({Outbound.PROJECT, Outbound.TEXT})
#: 絵を返す道具 #320 のコンタクトシートと範囲のプレビューもこれにする
_WITH_FRAMES = frozenset({Outbound.PROJECT, Outbound.FRAMES})
#: 操作の履歴の名前を返す道具 名前には素材の名前（「配置: 〜」）と、テキストの文の頭
#: （「テキストを追加: 〜」）が入る
_HISTORY = frozenset({Outbound.PROJECT, Outbound.FILES, Outbound.TEXT})
#: 履歴の名前の中の文字は、送らないと決めた人には伏せて送る 取り消しは断らない
#: 断ると、字幕と文字を送らない人は AI に取り消しを頼めない（2026-10-15 利用者の決定）
_HIDDEN_IN_HISTORY = frozenset({Outbound.TEXT})
#: 値を変える道具（set_param・add_keyframe）は、テキストの本文を変えたときだけ結果に本文が載る
#: 本文のための道具ではないので、字幕と文字を送らない人にも断らず、本文を伏せて返す
#: 断ると、色や位置を変えるだけの頼みまで動かなくなる
_HIDDEN_IN_VALUES = frozenset({Outbound.TEXT})
#: アプリ自身の設定 書き出しのフォルダや退避の置き場が入る
_APP_SETTINGS = frozenset({Outbound.APP_SETTINGS, Outbound.FILES})
#: 保存したプリセット・エイリアスで字幕に見た目を着せる道具（#308） 見つからないと断る文に
#: 保存してある物の名前を並べる（AI が名前を選び直すための唯一の手掛かり） 本人の持ち物の
#: 名前で、作品の中身でも素材でもないので、アプリの設定と同じく送る 読み込めない理由の文には
#: 置き場の場所が入りうるので、名前と場所も書く（断った人には伏せる）
_WITH_LOOKS = frozenset({Outbound.PROJECT, Outbound.APP_SETTINGS, Outbound.FILES})
#: 保存したプリセットとエイリアスを当てる・置く道具（#328） 結果と断る文に名前が載る
#: 名前は本人の持ち物で、作品の中身でも素材でもないので、アプリの設定として送る（#308 と同じ）
_SAVED_LOOKS = frozenset({Outbound.PROJECT, Outbound.APP_SETTINGS})
#: 一覧（#328） 保存した名前と棚の名前に加えて、棚のフォルダの名前（本人の置き場の名前）を返す
#: 作品の中身は返さない
_LOOKS_AND_PLACES = frozenset({Outbound.APP_SETTINGS, Outbound.FILES})
#: 棚の 1 本を置く・着せる道具（#328） 読めない理由の文に棚の場所が、素材の数え方に素材の場所が
#: 入りうる
_LOOKS_AND_PLACES_IN_PROJECT = frozenset({Outbound.PROJECT, Outbound.APP_SETTINGS, Outbound.FILES})
#: 仕上がりの検査（#337） 問題の時刻とクリップ ID に、本人の設定（音量の基準）と、
#: 調べた動画の場所・調べられなかった理由（素材の名前が入りうる）が載る
_CHECKS = frozenset({Outbound.PROJECT, Outbound.APP_SETTINGS, Outbound.FILES})
#: 音量の推移のグラフ（#338） 検査と同じ物に、絵（グラフの画像）が加わる 字幕は重なる時刻だけを
#: 使い、文は返さない
_SOUND_PICTURE = frozenset(
    {Outbound.PROJECT, Outbound.FRAMES, Outbound.APP_SETTINGS, Outbound.FILES}
)
#: ダッキング（#329） クリップ ID と音量の線の要約に、本人の設定（下げ幅と速さの既定）が載る
#: 字幕は話している区間の時刻だけを使い、文は返さない 断る文には素材の名前が入りうる
_DUCKING = frozenset({Outbound.PROJECT, Outbound.APP_SETTINGS, Outbound.FILES})
#: 強調の動き（#331） クリップ ID・フレーム・強さに、本人の設定（強さ・長さ・動き方の既定）が
#: 載る 断る文に出るのはクリップ ID とトラックの名前（作品の中身）だけ
_EMPHASIS = frozenset({Outbound.PROJECT, Outbound.APP_SETTINGS})
#: ジャンプカットのつなぎ目（#332） フレームとクリップ ID に、本人の設定（処理の種類と長さの
#: 既定）が載る 断る文に出るのはクリップ ID だけ
_SEAMS = frozenset({Outbound.PROJECT, Outbound.APP_SETTINGS})
#: 無音カット 断る文に素材の名前が入り、つなぎ目も処理すれば本人の設定が載る（#332）
_JET_CUT = frozenset({Outbound.PROJECT, Outbound.FILES, Outbound.APP_SETTINGS})
#: 声の一覧（#335） エンジンの名前・声の名前と規約の文・本人の設定（エンジンの場所と既定の声）
_VOICES = frozenset({Outbound.APP_SETTINGS})
#: ナレーションを作って置く（#335） 書いた WAV の場所・置いたクリップ・声の名前と表記
_NARRATION = frozenset({Outbound.PROJECT, Outbound.FILES, Outbound.APP_SETTINGS})
#: つなぎ目の処理の引数 smooth_cuts と jet_cut（smooth）で同じ
_SEAM_ARGUMENTS: dict[str, Any] = {
    "video": _string("映像の処理 crossfade・zoom・none 省くと本人の設定"),
    "crossfade_ms": _integer("クロスフェードの長さ（ミリ秒 10〜1000） 省くと本人の設定"),
    "zoom_percent": _integer("寄りの拡大率（% 101〜150） 省くと本人の設定"),
    "audio_fade_ms": _integer(
        "音の継ぎ目のフェード（ミリ秒 0〜200 0 で音は触らない） 省くと本人の設定"
    ),
}
#: 本人の答え
_CONVERSATION = frozenset({Outbound.CONVERSATION})


OPERATIONS: tuple[Operation, ...] = (
    Operation(
        name="get_project",
        description="プロジェクトの設定（解像度・fps・長さ）と再生ヘッドの位置を返す",
        schema=_schema({}),
        handler=_get_project,
        sends=_WITH_FILES,
        name_keys=frozenset({"name"}),
    ),
    Operation(
        name="list_media",
        description="メディアプールの素材を一覧する media_id はここで得る" + PAGING_NOTE,
        schema=_schema({"contains": _string("名前に含む言葉で絞る"), **_paging()}),
        handler=_list_media,
        sends=_WITH_FILES,
        name_keys=frozenset({"name"}),
    ),
    Operation(
        name="list_tracks",
        description="タイムラインのトラックを一覧する" + PAGING_NOTE,
        schema=_schema(_paging()),
        handler=_list_tracks,
        sends=_STRUCTURE,
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
        sends=_WITH_FILES,
        name_keys=frozenset({"media"}),
    ),
    Operation(
        name="get_selection",
        description="選択中のクリップと再生ヘッドの位置",
        schema=_schema({}),
        handler=_get_selection,
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_WITH_TEXT,
    ),
    Operation(
        name="get_history",
        description=(
            "直近の操作履歴と、取り消せるか・やり直せるか this_instruction は今の指示で"
            "変えた手（古い順）で、undo はそこから戻す"
        ),
        schema=_schema({}),
        handler=_get_history,
        sends=_HISTORY,
        hidden=_HIDDEN_IN_HISTORY,
        labels=True,
    ),
    # 目印（マーカー）とチャプター（#330） 人は目盛りの右クリックで同じことをする
    Operation(
        name="list_markers",
        description=(
            "タイムラインの目印（マーカー）を位置の順に一覧する 目印は位置（frame）で指す"
            " シーンを開いていればそのシーンの目印" + PAGING_NOTE
        ),
        schema=_schema(_paging()),
        handler=_list_markers,
        # 目印の名前は本人の文（話の区切りの見出し） 字幕と文字として尋ねる
        sends=_WITH_TEXT,
    ),
    Operation(
        name="add_markers",
        description=(
            "目印（マーカー）をまとめて置く 取り消し 1 回で全部戻る 名前を付けた目印は"
            " get_chapters で目次（チャプター）になる 同じ位置に 2 つは置けない"
            " 1 つでも置けなければ何も置かない"
        ),
        schema=_schema(
            {
                "markers": {
                    "type": "array",
                    "description": f"置く目印（{MAX_MARKERS_PER_CALL} 個まで）",
                    "minItems": 1,
                    "maxItems": MAX_MARKERS_PER_CALL,
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame": _integer("置く位置（タイムラインのフレーム）"),
                            "label": _string(
                                f"名前 チャプターの見出しになる（{MARKER_LABEL_LIMIT} 文字まで）"
                                " 省くと目次に入らない覚え書きの目印"
                            ),
                            "color": _string("色 #RRGGBB（省くと黄色）"),
                        },
                        "required": ["frame"],
                        "additionalProperties": False,
                    },
                }
            },
            ["markers"],
        ),
        handler=_add_markers,
        # 置いた目印を名前ごと返す（前後の空白を落とした後の名前を AI が確かめるため）
        # 字幕と文字を送らない人には、add_text と同じく道具を動かさずに断る
        sends=_WITH_TEXT,
        writes=True,
    ),
    Operation(
        name="update_marker",
        description="目印を動かす・名前や色を変える 変える物だけを渡す",
        schema=_schema(
            {
                "frame": _integer("変える目印の今の位置（list_markers の frame）"),
                "to_frame": _integer("動かす先の位置"),
                "label": _string(f"新しい名前（{MARKER_LABEL_LIMIT} 文字まで 空で名前を外す）"),
                "color": _string("新しい色 #RRGGBB"),
            },
            ["frame"],
        ),
        handler=_update_marker,
        sends=_WITH_TEXT,
        writes=True,
    ),
    Operation(
        name="delete_markers",
        description="目印を消す 位置をまとめて渡す 1 つでも無い位置があれば何も消さない",
        schema=_schema(
            {
                "frames": {
                    "type": "array",
                    "description": "消す目印の位置（list_markers の frame）",
                    "items": {"type": "integer"},
                },
                "all": _boolean("真ならすべての目印を消す"),
            }
        ),
        handler=_delete_markers,
        # 返すのは消した数と残りの数だけで、名前は返さない 字幕と文字を送らない人でも消せる
        sends=_STRUCTURE,
        writes=True,
    ),
    Operation(
        name="get_chapters",
        description=(
            "名前の付いた目印から目次（チャプター）を作って返す 時刻は書き出す範囲の頭から数え、"
            "youtube は YouTube の概要欄に貼る形（0:00 はじめに） YouTube が目次にしない"
            "（先頭が 0:00 でない・3 行より少ない・10 秒より短い行がある）ときは warnings に"
            "理由が入る 書き出すメインの目次は、シーンを開いていればメインに戻ってから読む"
            " 行の一覧（chapters）は" + PAGING_NOTE
        ),
        schema=_schema(_paging()),
        handler=_get_chapters,
        sends=_WITH_TEXT,
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
        sends=_WITH_FRAMES,
    ),
    Operation(
        name="contact_sheet",
        description=(
            "何コマも 1 枚に並べた画像で、全体を見渡す（コンタクトシート） 既定はタイムライン"
            "全体を 12 等分した真ん中のコマ from_frame・to_frame で区切るか、frames で見たい"
            "コマ（場面の切り替わりの一覧など）を直に渡す 各コマの下にタイムコードとフレーム"
            "番号を書く 当たりを付けたら preview_frame で 1 コマを大きく見る"
            f" 1 枚の長い辺は {MAX_SHEET_EDGE} 画素までで、超えるときは升を縮める"
        ),
        schema=_schema(
            {
                "frames": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        f"並べるフレーム（{SHEET_MAX_COUNT} 個まで） 渡すと from_frame・"
                        "to_frame・count は使わない"
                    ),
                },
                "from_frame": _integer("見渡す範囲の頭（既定 0）"),
                "to_frame": _integer("見渡す範囲の終わり このフレームの前まで（既定 終わりまで）"),
                "count": _integer(
                    f"並べる枚数（1〜{SHEET_MAX_COUNT}、既定 {SHEET_DEFAULT_COUNT}）"
                ),
                "cell_width": _integer(
                    f"1 コマの横幅（{MIN_CELL_WIDTH}〜{MAX_CELL_WIDTH}、既定 {SHEET_DEFAULT_CELL}）"
                ),
                "columns": _integer(f"列の数（1〜{MAX_COLUMNS}） 省くと升が大きく取れる数"),
            }
        ),
        handler=_contact_sheet,
        # 絵（何コマも並べた 1 枚）と、説明の時刻・フレーム番号（作品の構成）を返す
        sends=_WITH_FRAMES,
    ),
    Operation(
        name="preview_range",
        description=(
            "続いたコマを 1 枚に並べた画像で、動きを確かめる（場面切り替え・キーフレーム・"
            "字幕の出入りなど） start_frame から step おきに count 枚 各コマの下に"
            "タイムコードとフレーム番号を書く 終わりを越える分は描かず、減らしたことを返事に"
            "書く 2 コマ未満になる頼みは断る 長い範囲を見渡すときは contact_sheet を使う"
            " 同じ範囲の音は audio_graph に並べたフレームを marks で渡すと、縦線で揃えて見られる"
        ),
        schema=_schema(
            {
                "start_frame": _integer("頭のフレーム 省くと再生ヘッド"),
                "count": _integer(
                    f"並べる枚数（2〜{RANGE_MAX_COUNT}、既定 {RANGE_DEFAULT_COUNT}）"
                ),
                "step": _integer("何フレームおきに取るか（既定 1 = 1 コマずつ）"),
                "cell_width": _integer(
                    f"1 コマの横幅（{MIN_CELL_WIDTH}〜{MAX_CELL_WIDTH}、既定 {RANGE_DEFAULT_CELL}）"
                ),
                "columns": _integer(
                    f"列の数（1〜{MAX_COLUMNS}、既定 {RANGE_DEFAULT_COLUMNS}） 左上から右へ時間順"
                ),
            }
        ),
        handler=_preview_range,
        # 絵（続いたコマを並べた 1 枚）と、説明の時刻・フレーム番号（作品の構成）を返す
        sends=_WITH_FRAMES,
    ),
    Operation(
        name="detect_scenes",
        description=(
            "素材（動画）の場面の切り替わり（カット）の位置を一覧する フレーム間の差から"
            "探すので、映像を 1 枚ずつ見るより速い 見つけた位置の前後だけを preview_frame で"
            "確かめる 位置は素材の秒（source_seconds）で、切り替わりは after_seconds との間に"
            "ある timeline はその場面の最初の絵が出るタイムラインのフレーム（preview_frame や"
            "contact_sheet の frames へそのまま渡せる 今のタイムラインに置いていなければ空）"
            " 解析は素材を読み込んだときに"
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
        sends=_WITH_FILES,
    ),
    Operation(
        name="check_output",
        description=(
            "仕上がりを調べて問題の一覧を返す（読むだけ 何も変えない） 黒いコマ（black）・"
            "止まった絵（freeze）・無音（silence）・音割れ（clipping）・音量の基準（loudness "
            "ITU-R BS.1770 の積分ラウドネス）・字幕のはみ出し（text_overflow"
            " 画面の外へ出て切れる字）・"
            "長さ（length expected_seconds を渡したとき）を調べる 既定は今のタイムライン"
            "（書き出す前の、描いた絵と混ぜた音） path を渡すと書き出した動画のファイルを調べる"
            "（作品のフォルダと書き出しのフォルダの中だけ 字幕のはみ出しは調べられない"
            " AAC など圧縮した音では音割れを取りこぼすことがある）"
            " 各問題は kind・severity（high / medium / low）・start_frame・end_frame・秒・"
            "detail と程度の数を持つ タイムラインのフレームは preview_frame へそのまま渡して"
            "確かめる 全部のコマを描くので長い作品では時間がかかる 裏で調べ、終わっていなければ"
            "wait_seconds まで待ち、それでも終わらなければ調べた所までを返す（complete が false"
            " 同じ引数でもう 1 度呼ぶと続きを返す） 人はステータスバーの〔止める〕で止められる"
            " checks で種類を絞ると速い（音だけなら絵を描かない） 作品を変えてから呼ぶと調べ直す"
            + PAGING_NOTE
        ),
        schema=_schema(
            {
                "path": _string(
                    "書き出した動画のファイル 省くと今のタイムライン（開いているシーン）を調べる"
                ),
                "from_frame": _integer("タイムラインのこのフレームから（既定 0）"),
                "to_frame": _integer("タイムラインのこのフレームの手前まで（既定 終わり）"),
                "checks": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(CHECK_KINDS)},
                    "description": "調べる種類 省くと全部",
                },
                "black_level": _number(
                    "黒とみなす明るさ 0〜0.5（既定 0.1 FFmpeg の blackdetect と同じ） 画面の 98% "
                    "以上がこれ以下なら黒いコマ"
                ),
                "min_black_seconds": _number(
                    "これより短い黒は返さない（既定 0 = 1 コマでも返す"
                    " つなぎ目の 1 コマの黒も拾う）"
                ),
                "freeze_noise": _number(
                    "止まりとみなす明るさの差の平均 0〜0.1"
                    "（既定 0.001 FFmpeg の freezedetect と同じ）"
                ),
                "min_freeze_seconds": _number("これより短い止まりは返さない（既定 2）"),
                "silence_db": _number(
                    "無音とみなす大きさ dBFS（既定 -60 FFmpeg の silencedetect と同じ）"
                ),
                "min_silence_seconds": _number("これより短い無音は返さない（既定 2）"),
                "clip_db": _number("音割れとみなす大きさ dBFS -6〜0（既定 -0.01）"),
                "clip_samples": _integer(
                    "続けて頭打ちになったサンプルがこの数以上で音割れ（既定 3 Audacity と同じ）"
                ),
                "loudness_target": _number(
                    "音量の基準 LUFS（既定は本人の設定 "
                    + " / ".join(
                        f"{value} は{name.split(' ')[0]}" for value, name in LOUDNESS_TARGETS
                    )
                    + "）"
                ),
                "loudness_tolerance": _number("基準からの許す差 LU（既定 1）"),
                "safe_margin_percent": _number(
                    "字幕を置いてよい範囲の余白 各辺の % 0〜20（既定 0 = 画面の端 テレビの"
                    "安全域なら 5）"
                ),
                "expected_seconds": _number("長さの目安（秒） 渡すと外れたときに length を返す"),
                "length_tolerance_seconds": _number("長さの目安からの許す差（秒 既定 0.5）"),
                "wait_seconds": _number(
                    f"終わっていないときに待つ秒（0〜{MAX_CHECK_WAIT:g}、"
                    f"既定 {DEFAULT_CHECK_WAIT:g}） 0 なら待たずに調べた所までを返す"
                ),
                **_paging(),
            }
        ),
        handler=_check_output,
        # 結果は時刻とクリップ ID だが、書き出した動画を調べたときは target に場所が入り、
        # 調べられなかった理由の文には素材の名前や場所が入りうる 場所は形で、名前は表で伏せる
        # target は "timeline" も取るので、値ごと伏せる欄（name_keys）には入れない
        # 音量の基準（本人の設定 Preferences.check_loudness_target）も結果に載る
        sends=_CHECKS,
    ),
    Operation(
        name="analyze_audio",
        description=(
            "音を数で読む（読むだけ 何も変えない） 全体の積分ラウドネス・ラウドネスの幅（LRA）・"
            "最大の瞬時と短時間のラウドネス・最大のサンプル（summary）と、音量の推移（levels"
            " points 点に束ねた 各点はフレーム・短時間（3 秒）のラウドネス・瞬時（400 ms）の"
            "最大・最大のサンプル）と、一覧（list）を返す 一覧は segments（話し声 speech・"
            "音楽 music・BGM の上の話し声 speech_and_music・無音 silence・雑音 noise・"
            "見分けられない unclear を 1 秒ごとに見分けた区間）か silences（無音の区間"
            " 100 ms 刻み） 既定は今のタイムライン（書き出す前の混ぜた音） path を渡すと"
            "書き出した動画のファイル（作品のフォルダと書き出しのフォルダの中だけ）"
            " check_output と同じ解析を使い回す 同じ範囲を前に調べていれば読み直さない"
            " 長い作品は裏で調べ、終わっていなければ wait_seconds まで待ち、それでも終わらなければ"
            "調べた所までを返す（complete が false 同じ引数でもう 1 度呼ぶと続きを返す）"
            " 絵で見るなら audio_graph" + _SOUND_LIMITS_NOTE + PAGING_NOTE
        ),
        schema=_schema(
            {
                "path": _string(
                    "書き出した動画のファイル 省くと今のタイムライン（開いているシーン）を調べる"
                ),
                "from_frame": _integer("タイムラインのこのフレームから（既定 0）"),
                "to_frame": _integer("タイムラインのこのフレームの手前まで（既定 終わり）"),
                "list": {
                    "type": "string",
                    "enum": list(SOUND_LISTS),
                    "description": "返す一覧 segments（見分けた区間 既定）か silences（無音）",
                },
                "points": _integer(
                    f"推移の点の数（0〜{MAX_SOUND_POINTS}、既定 {DEFAULT_SOUND_POINTS}） 0 なら"
                    " 推移を返さない 狭い範囲を細かく見たいときは from_frame・to_frame で絞る"
                ),
                "silence_db": _number(
                    f"無音とみなす大きさ dBFS（既定 {DEFAULT_SOUND_SILENCE_DB:g} 話の間も拾う"
                    " check_output の既定 -60 より大きい）"
                ),
                "min_silence_seconds": _number(
                    f"silences で返す無音の最短（秒 既定 {DEFAULT_SOUND_MIN_SILENCE:g}）"
                ),
                "wait_seconds": _number(
                    f"終わっていないときに待つ秒（0〜{MAX_CHECK_WAIT:g}、"
                    f"既定 {DEFAULT_CHECK_WAIT:g}） 0 なら待たずに調べた所までを返す"
                ),
                **_paging(),
            }
        ),
        handler=_analyze_audio,
        # 数の並び（作品の構成）と、音量の基準（本人の設定） 書き出した動画を調べたときは
        # target に場所が入り、調べられなかった理由の文に素材の名前が入りうる（check_output と同じ）
        sends=_CHECKS,
    ),
    Operation(
        name="audio_graph",
        description=(
            "音量の推移をグラフの画像で返す（読むだけ） 横は時間・縦はラウドネス（LUFS） 太い"
            "線は短時間（3 秒）、細い線は瞬時（400 ms）の最大 点線は音量の基準 上の端の赤い印は"
            "頭打ち 線の下の帯は話し声・音楽・無音の見分け（色の意味は返事の説明） その下の"
            "細い帯は字幕のある所 marks に渡したフレームに番号付きの縦線を引く（preview_range"
            " で見るコマのフレームを渡すと、動きと音を並べて確かめられる） 範囲は from_frame・"
            "to_frame（既定 タイムライン全体） path を渡すと書き出した動画 解析は analyze_audio"
            "・check_output と同じ物を使い回し、終わっていなければ wait_seconds まで待つ"
        ),
        schema=_schema(
            {
                "path": _string(
                    "書き出した動画のファイル 省くと今のタイムライン（開いているシーン）を描く"
                ),
                "from_frame": _integer("タイムラインのこのフレームから（既定 0）"),
                "to_frame": _integer("タイムラインのこのフレームの手前まで（既定 終わり）"),
                "marks": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": f"縦線を引くフレーム（{MAX_MARKS} 個まで 範囲の中だけ）",
                },
                "width": _integer(
                    f"画像の横幅（{GRAPH_MIN_WIDTH}〜{MAX_SHEET_EDGE}、既定 {GRAPH_DEFAULT_WIDTH}）"
                ),
                "height": _integer(
                    f"画像の高さ（{GRAPH_MIN_HEIGHT}〜{GRAPH_MAX_HEIGHT}、"
                    f"既定 {GRAPH_DEFAULT_HEIGHT}）"
                ),
                "silence_db": _number(
                    f"見分けの帯で無音とみなす大きさ dBFS（既定 {DEFAULT_SOUND_SILENCE_DB:g}）"
                ),
                "wait_seconds": _number(
                    f"終わっていないときに待つ秒（0〜{MAX_CHECK_WAIT:g}、"
                    f"既定 {DEFAULT_CHECK_WAIT:g}）"
                ),
            }
        ),
        handler=_audio_graph,
        # 絵（グラフ）は同意を取ってから送る（#342） 説明の文は作品の構成と音量の基準
        # 書き出した動画を描いたときは説明に場所が入る
        sends=_SOUND_PICTURE,
    ),
    Operation(
        name="get_app_settings",
        description=(
            "アプリ自身の設定を読む（本人の好み・ショートカット・画面配置・版・更新・ヘルプ）"
            " 編集の判断に使う 読むだけで、変えるのは人だけ（表示 → 設定… など）"
        ),
        schema=_schema({}),
        handler=_get_app_settings,
        sends=_APP_SETTINGS,
    ),
    # 編集の方針（#343） 人は ファイル → プロジェクト設定… で書く AI は読むだけ
    Operation(
        name="get_style_guide",
        description=(
            "作品の編集の方針（スタイルガイド）を読む 使う書体・使う色（#RRGGBB）・字幕の見た目・"
            "テンポ・使ってよい効果（エフェクトの kind）・そのほか 見た目や効果やカットの間を"
            "決める前に読んで従う 空の欄は決めていない 読むだけで、変えるのは本人だけ"
            " 効果の一覧（effects）は" + PAGING_NOTE
        ),
        schema=_schema(_paging()),
        handler=_get_style_guide,
        # 字幕の見た目・テンポ・そのほかは本人の文 字幕と文字として尋ね、断られたら動かさない
        sends=_WITH_TEXT,
    ),
    Operation(
        name="check_style_guide",
        description=(
            "編集の方針に照らして、方針の外の書体・色・効果を使うクリップを返す（メインと"
            "すべてのシーン） 照らすのは決めてある欄だけ（checked） 直す先は allowed"
            " 一覧（problems）は" + PAGING_NOTE
        ),
        schema=_schema(_paging()),
        handler=_check_style_guide,
        # 返すのは書体の名前・色・効果の種類とクリップの在りかだけ 方針の文も本文も返さない
        sends=_STRUCTURE,
    ),
    Operation(
        name="seek",
        description="再生ヘッドを動かす",
        schema=_schema({"frame": _integer("移動先のフレーム")}, ["frame"]),
        handler=_seek,
        sends=_STRUCTURE,
    ),
    Operation(
        name="select_clip",
        description="クリップを選択する clip_id を空にすると選択を解く",
        schema=_schema({"clip_id": _string("選ぶクリップ")}),
        handler=_select,
        sends=_STRUCTURE,
    ),
    Operation(
        name="list_scenes",
        description="シーン（入れ子のタイムライン）の一覧と、いま開いているシーン",
        schema=_schema({}),
        handler=_list_scenes,
        sends=_STRUCTURE,
    ),
    Operation(
        name="set_active_scene",
        description=(
            "編集するシーンを切り替える scene_id を空にするとメイン "
            "以後の読み取りと編集は、そのシーンのタイムラインが相手になる"
        ),
        schema=_schema({"scene_id": _string("開くシーン 空ならメイン")}),
        handler=_set_active_scene,
        sends=_STRUCTURE,
    ),
    Operation(
        name="select_clips",
        description="何本かのクリップをまとめて選ぶ 空の配列で選択を解く 最後の 1 本が主になる",
        schema=_schema(
            {"clip_ids": {"type": "array", "items": {"type": "string"}, "description": "選ぶ"}},
            ["clip_ids"],
        ),
        handler=_select_clips,
        sends=_STRUCTURE,
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
        sends=_WITH_FILES,
        name_keys=frozenset({"imported"}),
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
        sends=_WITH_FILES,
        name_keys=frozenset({"placed"}),
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_WITH_TEXT,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
        writes=True,
    ),
    Operation(
        name="split_clip",
        description="クリップを分割する clip_id を省くと選択中のクリップ",
        schema=_schema(
            {"clip_id": _string("対象"), "frame": _integer("分割位置 省略すると再生ヘッド")}
        ),
        handler=_split_clip,
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
        writes=True,
    ),
    Operation(
        name="delete_clip",
        description="クリップを消す ripple を真にすると後ろを詰める",
        schema=_schema({"clip_id": _string("対象"), "ripple": _boolean("詰めるか")}),
        handler=_delete_clip,
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
        writes=True,
    ),
    Operation(
        name="group_clips",
        description="2 本以上のクリップを 1 つのグループに束ねる 束ねたものは一緒に選ばれて動く",
        schema=_schema({"clip_ids": _clip_ids_schema()}),
        handler=_group_clips,
        sends=_STRUCTURE,
        writes=True,
    ),
    Operation(
        name="ungroup_clips",
        description="クリップが入っているグループを、仲間ごと解く",
        schema=_schema({"clip_ids": _clip_ids_schema()}),
        handler=_ungroup_clips,
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
        writes=True,
    ),
    Operation(
        name="delete_clips",
        description="何本かのクリップをまとめて消す ripple を真にすると消したぶんを詰める",
        schema=_schema({"clip_ids": _clip_ids_schema(), "ripple": _boolean("詰めるか")}),
        handler=_delete_clips,
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
        writes=True,
    ),
    Operation(
        name="set_speed",
        description=(
            "クリップの再生速度を変える（スロー・早送り） リンクした絵と音も同じ速度にする"
            " keep が content（既定）なら素材の同じ範囲を流す長さにする（200% で長さが半分）"
            " length なら長さを変えず流す素材の範囲が変わる（設定パネルの「再生速度」と同じ）"
            " 長くなって後ろのクリップにぶつかるなら断る 音は高さも変わる"
        ),
        schema=_schema(
            {
                "clip_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "対象のクリップ 省くと選んでいるクリップすべて",
                },
                "speed_percent": _number("速度（% 1〜1000 100 で等倍、50 で半分の速さ）"),
                "keep": _string("content（中身を保って長さを変える 既定）か length（長さを保つ）"),
            },
            ["speed_percent"],
        ),
        handler=_set_speed,
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_STRUCTURE,
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
        sends=_WITH_TEXT,
        hidden=_HIDDEN_IN_VALUES,
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
        sends=_WITH_TEXT,
        hidden=_HIDDEN_IN_VALUES,
        writes=True,
    ),
    Operation(
        name="emphasize",
        description=(
            "話の要所を目立たせる動き（強調の動き）を、絵のクリップの範囲に付ける"
            " zoom は寄って止まり戻る（focus_x・focus_y の点を止めたまま寄る 話者の顔など）"
            " pan は位置をずらして戻る ポップ（pop）は一瞬ふくらんで戻る 揺れ（shake）は"
            "細かく揺らす zoom と pan は keep で戻さずクリップの終わりまで残せる"
            " どれも配置の値（揺れはランダム移動のエフェクトの幅）にキーフレームを打つので、"
            "人が設定パネルとグラフエディタで直せる 範囲の外の動きは変えない"
            " 範囲の中に前からのキーフレームがあるか、前からの動きが範囲の中で続いていれば断る"
            "（元の動きの形を保てないため） 強さ・動き切る長さ・動き方を省くと"
            "本人の設定（表示 → 設定… の強調の動き）を使う 取り消し 1 回で戻る"
            " 位置は画素で Y は上が正"
        ),
        schema=_schema(
            {
                "clip_id": _string("対象（省くと選択中のクリップ）"),
                "kind": {
                    "type": "string",
                    "enum": list(EMPHASIS_KINDS),
                    "description": "動きの種類",
                },
                "start_frame": _integer(
                    "動き始める位置（タイムラインのフレーム 省くと再生ヘッド）"
                ),
                "end_frame": _integer(
                    "戻り終える位置（タイムラインのフレーム） keep のときは渡さない"
                ),
                "scale_percent": _number(
                    "zoom と pop の拡大率 今の大きさからの %"
                    f"（zoom は {EMPHASIS_ZOOM_PERCENT_RANGE[0]}"
                    f"〜{EMPHASIS_ZOOM_PERCENT_RANGE[1]}、pop は {EMPHASIS_POP_PERCENT_RANGE[0]}"
                    f"〜{EMPHASIS_POP_PERCENT_RANGE[1]} 省くと本人の設定）"
                ),
                "focus_x": _number(
                    "zoom で止めておく点の X 配置の中心（既定は画面の中央）からの画素"
                    " 作品の画面の左上から x の所なら x - 幅/2 省くと絵の中心へ寄る"
                ),
                "focus_y": _number(
                    "zoom で止めておく点の Y（上が正） 作品の画面の左上から y の所なら 高さ/2 - y"
                ),
                "offset_x": _number("pan でずらす X の画素（右が正） direction とどちらか"),
                "offset_y": _number("pan でずらす Y の画素（上が正）"),
                "direction": {
                    "type": "string",
                    "enum": list(PAN_DIRECTIONS),
                    "description": "pan の向き offset_x・offset_y とどちらか",
                },
                "distance_px": _number(
                    f"direction のときにずらす画素（{EMPHASIS_PAN_PX_RANGE[0]}"
                    f"〜{EMPHASIS_PAN_PX_RANGE[1]} 省くと本人の設定）"
                ),
                "amplitude_px": _number(
                    f"shake の幅 上下左右それぞれの画素（{EMPHASIS_SHAKE_PX_RANGE[0]}"
                    f"〜{EMPHASIS_SHAKE_PX_RANGE[1]} 省くと本人の設定）"
                ),
                "ramp_ms": _number(
                    f"動き始めてから動き切るまで（と戻るまで）のミリ秒（{EMPHASIS_RAMP_MS_RANGE[0]}"
                    f"〜{EMPHASIS_RAMP_MS_RANGE[1]} 0 は 1 フレームで切り替える 省くと本人の設定）"
                ),
                "easing": {
                    "type": "string",
                    "enum": list(EMPHASIS_EASINGS),
                    "description": (
                        "動き方 smooth はなめらかに動き出して止まる back・elastic・bounce は"
                        "着いた所で跳ねる（省くと本人の設定）"
                    ),
                },
                "keep": _boolean("zoom と pan を戻さずクリップの終わりまで残す（既定 false）"),
            },
            ["kind"],
        ),
        handler=_emphasize,
        writes=True,
        sends=_EMPHASIS,
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
        sends=_WITH_TEXT,
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
        sends=_WITH_TEXT,
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
        sends=_STRUCTURE,
        writes=True,
    ),
    Operation(
        name="place_subtitles",
        description=(
            "字幕をテキストオブジェクトとしてタイムラインへ置く（焼き込み）"
            " 話し手（素材と音声）ごとに別のレイヤーへ入れる 省けば出ている字幕を全部"
            " 見た目はひな形のテキストのクリップを写すか、保存したエイリアス・プリセット・"
            "テンプレートの棚から選んで着せる（本文と長さは字幕のまま 動きは字幕の長さに合わせる）"
            " どちらも省けば下寄せ・縁取りの既定 選べる物は list_looks で読める"
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
                "look_kind": _string(
                    "look_name の種類 alias（自作のエイリアス 既定）・preset（プリセット）・"
                    "template（テンプレートの棚）"
                ),
                "look_name": _string(
                    "見た目にする保存したエイリアスかプリセットの名前 棚は list_looks の"
                    "template_id template_clip_id と一緒には使えない 文字を持たない物"
                    "（図形など）は使えない"
                ),
                "look_category": _string(
                    "プリセットの分類 同じ名前のプリセットが分類違いで重なるときに指定する"
                ),
            }
        ),
        handler=_place_subtitles,
        sends=_WITH_LOOKS,
        writes=True,
    ),
    Operation(
        name="list_looks",
        description=(
            "保存したプリセット（クリップに当てる見た目）・自作のエイリアス（置き直すクリップ）・"
            "テンプレートの棚（AviUtl のエイリアスと YMM4 のアイテムテンプレート）を一覧する"
            " 見た目を一から作る前に、ここにある物を apply_preset・place_alias・place_template・"
            "apply_template で使う 棚は template_id で指す" + PAGING_NOTE
        ),
        schema=_schema(
            {
                "kind": _string("preset・alias・template のどれかで絞る 省けば全部"),
                "contains": _string("名前に含む言葉で絞る"),
                **_paging(),
            }
        ),
        handler=_list_looks,
        # 本人が保存した名前（アプリの設定）と、棚のフォルダの名前（本人の置き場の名前）
        sends=_LOOKS_AND_PLACES,
    ),
    Operation(
        name="apply_preset",
        description=(
            "保存したプリセット（見た目ひとそろい）をクリップへ当てる 設定パネルの"
            "〔プリセット…〕と同じ当て方（文字と位置を当てるかは本人の設定のまま）"
            " 取り消し 1 回で全部戻る"
        ),
        schema=_schema(
            {
                "name": _string("プリセットの名前（list_looks の kind=preset）"),
                "category": _string("分類 同じ名前が分類違いで重なるときに指定する"),
                "clip_ids": _clip_ids_schema(),
            },
            ["name"],
        ),
        handler=_apply_preset,
        sends=_SAVED_LOOKS,
        writes=True,
    ),
    Operation(
        name="place_alias",
        description=(
            "保存したエイリアス（作り込んだクリップ）を置く 右クリックの〔追加〕→"
            "〔エイリアス〕と同じ 長さと動きは保存したまま"
        ),
        schema=_schema(
            {
                "name": _string("エイリアスの名前（list_looks の kind=alias）"),
                "at_frame": _integer("置く位置 省けば再生ヘッド"),
                "track_id": _string("置くトラック 空いていなければいつもの決まりで選ぶ"),
            },
            ["name"],
        ),
        handler=_place_alias,
        sends=_SAVED_LOOKS,
        writes=True,
    ),
    Operation(
        name="place_template",
        description=(
            "テンプレートの棚の 1 本（AviUtl のエイリアス・YMM4 のアイテムテンプレート）を置く"
            " 互換の〔テンプレート…〕の〔置く〕と同じ 指している画像や音声も素材として登録する"
            " 元のレイヤーの所が埋まっていれば、全部を一番手前の新しいトラックへ上下の並びのまま"
            "置く 取り消し 1 回で素材ごと戻る"
        ),
        schema=_schema(
            {
                "template_id": _string("棚の 1 本（list_looks の kind=template の template_id）"),
                "at_frame": _integer("置く位置 省けば再生ヘッド"),
                "track_id": _string(
                    "置くトラック 置く物が 1 つでそこが空いていれば使う 省けば元のレイヤーの並び"
                    "（埋まっていれば一番手前の新しいトラック）"
                ),
            },
            ["template_id"],
        ),
        handler=_place_template,
        # 読めない理由の文と素材の数え方に、棚や素材の場所が入りうる
        sends=_LOOKS_AND_PLACES_IN_PROJECT,
        writes=True,
    ),
    Operation(
        name="apply_template",
        description=(
            "テンプレートの棚の 1 本の見た目をテキストのクリップに着せる 互換の"
            "〔テンプレート…〕の〔選択中のクリップに適用〕と同じ 文字と長さはそのまま"
        ),
        schema=_schema(
            {
                "template_id": _string("棚の 1 本（list_looks の kind=template の template_id）"),
                "clip_id": _string("着せるテキストのクリップ 省けば選択中のクリップ"),
            },
            ["template_id"],
        ),
        handler=_apply_template,
        sends=_LOOKS_AND_PLACES_IN_PROJECT,
        writes=True,
    ),
    Operation(
        name="import_objects",
        description=(
            "AviUtl のオブジェクトのファイル（.exo .exa .exo2 .exa2 .object）を読み込んで置く"
            " 互換の〔オブジェクトを読み込む…〕と同じ 元のレイヤーの並びのまま頭から置き、"
            "指している素材も登録する 元のレイヤーの所が埋まっていれば、全部を一番手前の新しい"
            "トラックへ上下の並びのまま置く 取り消し 1 回で素材ごと戻る"
        ),
        schema=_schema({"path": _string("読み込むファイルの場所")}, ["path"]),
        handler=_import_objects,
        sends=_WITH_FILES,
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
                "smooth": _boolean(
                    "詰めたつなぎ目を smooth_cuts と同じく目立たなくする（既定 false）"
                    " 同じ取り消し 1 回に入る video などの引数も smooth_cuts と同じ"
                ),
                **_SEAM_ARGUMENTS,
            },
            ["media_id"],
        ),
        handler=_jet_cut,
        sends=_JET_CUT,
        writes=True,
    ),
    Operation(
        name="smooth_cuts",
        description=(
            "ジャンプカットのつなぎ目を目立たなくする frames を省くと、同じ素材を飛び飛びに"
            "つないだ所（無音を詰めた所など）を探す video は crossfade（場面切り替えの短い"
            "クロスフェードを重ねる）zoom（後ろのクリップを 1 本おきに少し拡大する パンチイン）"
            "none（映像は触らない） 音は継ぎ目の前後に短いフェードを入れて「プツッ」を消す"
            " 省いた値は本人の設定 同じ所へ掛け直しても重ならない"
        ),
        schema=_schema(
            {
                "frames": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "つなぎ目のフレーム（前のクリップの終わり=後ろのクリップの頭）",
                },
                "track_id": _string(
                    "このトラックのクリップ（とリンクした相手）の所だけを処理する 省くと全部"
                ),
                "start_frame": _integer("探す範囲の頭（frames を省いたときだけ）"),
                "end_frame": _integer("探す範囲の終わり 含まない（frames を省いたときだけ）"),
                **_SEAM_ARGUMENTS,
            },
            [],
        ),
        handler=_smooth_cuts,
        sends=_SEAMS,
        writes=True,
    ),
    Operation(
        name="set_volume",
        description=(
            "音のクリップの音量と左右を変える（設定パネルの「音量」「パン」と同じ欄）"
            " volume_percent は決めた値にする（100 が素材のまま 0〜400 点は消える）"
            " gain_db は今の音量から上げ下げする（+6 でおよそ 2 倍 点の付いた線も形を保って"
            "動かす） 何本かまとめて変えられる 取り消し 1 回で戻る"
        ),
        schema=_schema(
            {
                "clip_ids": _clip_ids_schema(),
                "volume_percent": _number("音量 %（0〜400 100 が素材のまま）"),
                "gain_db": _number("今の音量から上げ下げする dB（-60〜+24）"),
                "pan_percent": _number("左右 %（-100 が左 +100 が右 0 が真ん中）"),
            }
        ),
        handler=_set_volume,
        writes=True,
        sends=_STRUCTURE,
    ),
    Operation(
        name="duck_audio",
        description=(
            "BGM などを、声が話している間だけ下げる（ダッキング） 声のクリップの話している所を"
            "字幕（無ければ波形）から拾い、下げるクリップの音量の線に点を打つ"
            "（タイムラインの音量の線と同じ点なので人が後で直せる） 下げ幅と速さを省くと"
            "本人の設定（表示 → 設定… の BGM を声の間に下げる量・速さ）を使う"
            " 掛け直すと前の線を置き換える（基準は線の一番大きい値） 取り消し 1 回で戻る"
        ),
        schema=_schema(
            {
                "clip_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "下げるクリップ（BGM） track_id とどちらか",
                },
                "track_id": _string("下げるトラック（その上の音のクリップすべて）"),
                "voice_clip_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "声のクリップ voice_track_id とどちらか",
                },
                "voice_track_id": _string("声のトラック（その上の音のクリップすべて）"),
                "depth_db": _number(
                    f"下げる量 dB（{DUCK_DEPTH_DB_RANGE[0]}〜{DUCK_DEPTH_DB_RANGE[1]}"
                    " 省くと本人の設定）"
                ),
                "attack_ms": _number(
                    f"話し始めの前から下げ始めて下がり切るまでのミリ秒（{DUCK_ATTACK_MS_RANGE[0]}"
                    f"〜{DUCK_ATTACK_MS_RANGE[1]} 省くと本人の設定）"
                ),
                "release_ms": _number(
                    f"話し終えてから元へ戻るまでのミリ秒（{DUCK_RELEASE_MS_RANGE[0]}"
                    f"〜{DUCK_RELEASE_MS_RANGE[1]} 省くと本人の設定） 間がこれと attack_ms の"
                    "和より短い文どうしは、下げたままつなぐ"
                ),
                "base_volume_percent": _number(
                    "下げていない所の音量 %（省くと今の線の一番大きい値）"
                ),
                "speech_from": {
                    "type": "string",
                    "enum": list(SPEECH_SOURCES),
                    "description": (
                        "話している所を何から拾うか auto は字幕があれば字幕、無ければ波形"
                        "（既定 auto）"
                    ),
                },
                "threshold_db": _number(
                    "波形から拾うとき、これより大きい所を声とみなす dBFS（既定 -40）"
                ),
            }
        ),
        handler=_duck_audio,
        writes=True,
        sends=_DUCKING,
    ),
    Operation(
        name="normalize_loudness",
        description=(
            "音量をラウドネスの基準（ITU-R BS.1770 の積分ラウドネス）へそろえる 測った音へ"
            "同じ dB を掛けるので、声と BGM の釣り合いは変えない clip_ids を省くと作品全体"
            "（ミュートしていないトラックの音すべて）、渡すとそのクリップだけを混ぜて測る"
            " 音の山が peak_ceiling_db を超えるときと、音量の欄の上限 400% に当たるときは、"
            "基準の手前で止めて limited_by に書く 測るのに時間がかかる 終わらなければ"
            "何も変えずに measuring を返す（同じ引数でもう 1 度呼ぶと続きから）"
            " 取り消し 1 回で戻る"
        ),
        schema=_schema(
            {
                "clip_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "そろえるクリップ 省くと作品全体",
                },
                "target_lufs": _number(
                    "基準 LUFS（-40〜-5 省くと本人の設定 "
                    + " / ".join(
                        f"{value} は{name.split(' ')[0]}" for value, name in LOUDNESS_TARGETS
                    )
                    + "）"
                ),
                "peak_ceiling_db": _number("音の山の上限 dBFS（-20〜0 既定 -1）"),
                "wait_seconds": _number(
                    f"測り終えるまで待つ秒（0〜{MAX_CHECK_WAIT:g}、既定 {DEFAULT_CHECK_WAIT:g}）"
                ),
            }
        ),
        handler=_normalize_loudness,
        writes=True,
        sends=_CHECKS,
    ),
    Operation(
        name="place_sound",
        description=(
            "BGM や効果音（音だけの素材）をタイムラインへ置く 音声トラック（混合の作品では"
            "空いたレイヤー）の空いた所へ置く until_frame を渡すと、そこまで素材を繰り返して"
            "敷き詰め（loop が true のとき）、最後の 1 本を until_frame で切る 音量とフェードは"
            "置いたクリップの音量・フェードの欄に入る 動画の素材は place_media で置く"
            " 取り消し 1 回で戻る"
        ),
        schema=_schema(
            {
                "media_id": _string("置く素材（音だけの素材）"),
                "at_frame": _integer("置く位置 省くと再生位置"),
                "track_id": _string("置きたいトラック 埋まっていればほかの空いたトラックへ"),
                "until_frame": _integer("ここで終える（BGM を敷く範囲の終わり）"),
                "loop": _boolean("until_frame まで繰り返す（既定 true）"),
                "volume_percent": _number("音量 %（0〜400 既定 100）"),
                "fade_in_seconds": _number("頭のフェードイン 秒（0〜60）"),
                "fade_out_seconds": _number("終わりのフェードアウト 秒（0〜60）"),
            },
            ["media_id"],
        ),
        handler=_place_sound,
        writes=True,
        sends=_WITH_FILES,
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
        sends=_WITH_FILES,
        name_keys=frozenset({"started"}),
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
        sends=_WITH_FILES,
    ),
    Operation(
        name="undo",
        description=(
            "直前の操作を取り消す（画面の〔元に戻す〕と同じ） 今の指示の途中なら、この指示で"
            "変えた手を 1 つずつ戻す この指示で変えた物を全部戻すなら this_instruction を真に"
            "する この指示でまだ何も変えていなければ、前の指示の変更をまとめて 1 段で戻す"
            " 人が手で変えた段は戻さない（当たったら断るか、そこで止めて stopped に書く"
            " 人に Ctrl+Z を頼む） 次に戻す段が誰の物かは get_history の undo_by で分かる"
            " 戻した物は redo でやり直せる"
        ),
        schema=_schema(
            {
                "steps": _integer("戻す段数（既定 1）"),
                "this_instruction": _boolean(
                    "今の指示で変えた物を全部戻す（steps とは一緒に使わない）"
                ),
            }
        ),
        handler=_undo,
        sends=_HISTORY,
        hidden=_HIDDEN_IN_HISTORY,
        labels=True,
        writes=True,
    ),
    Operation(
        name="redo",
        description=(
            "取り消した操作をやり直す（画面の〔やり直す〕と同じ） 取り消した後に別の編集を"
            "するとやり直しは消える やり直せる物は get_history の redo_label で分かる"
            " 人が手で変えた段（redo_by が person）はやり直さない"
        ),
        schema=_schema({"steps": _integer("やり直す段数（既定 1）")}),
        handler=_redo,
        # 結果はやり直した操作の名前 取り消しと同じく、名前に素材の名前と文の頭が入る
        sends=_HISTORY,
        hidden=_HIDDEN_IN_HISTORY,
        labels=True,
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
        # 結果は保存した場所と、置き換える前の作品を残した場所 知らせの文には控えの置き場が入る
        # 名前と場所を断った人には伏せる（場所の欄は値ごと）
        sends=_WITH_FILES,
        name_keys=frozenset({"saved", "previous_kept_as"}),
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
        # 結果は保存した場所と、置き換える前の作品を残した場所 知らせの文には控えの置き場が入る
        # 名前と場所を断った人には伏せる（場所の欄は値ごと）
        sends=_WITH_FILES,
        name_keys=frozenset({"saved", "previous_kept_as"}),
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
        # 結果は書き出し先の場所 待ったときは書き出しの様子（場所と、終わりの文に場所が入る）
        sends=_WITH_FILES,
        name_keys=frozenset({"started"}),
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
        # 様子には書き出し先の場所と、終わりの文（場所・置き換える前のファイルを残した場所・
        # 失敗の理由に素材の名前）が入る
        sends=_WITH_FILES,
    ),
    Operation(
        name="cancel_export",
        description=(
            "走っている書き出しを止める 書きかけは捨て、書き出し先の前からあるファイルは"
            "そのまま残る 作品は変わらない 書き出し先へ置き始めた後（placing）は止めずに断る"
        ),
        schema=_schema({}),
        handler=_cancel_export,
        # 止めた後の様子に書き出し先の場所が入る
        sends=_WITH_FILES,
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
        # 結果は書いた場所だけ 字幕の文はファイルへ書くだけで、結果には載らない
        sends=_WITH_FILES,
        name_keys=frozenset({"written", "previous_kept_as"}),
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
        # 結果は保存した場所とコマ 絵はファイルへ書くだけで、結果には載らない
        sends=_WITH_FILES,
        name_keys=frozenset({"saved", "previous_kept_as"}),
    ),
    Operation(
        name="list_voices",
        description=(
            "ナレーションの声を作るエンジン（本人が入れて起動した VOICEVOX か AivisSpeech）の"
            "声の一覧を読む 声ごとに voice_id・名前・書き出した動画に添える表記（credit）が並ぶ"
            " 本人の設定の既定の声と、AI が声を選んでよいか（ai_may_choose_voice）も返す"
            " terms は voice_id で頼んだ声（省くと既定の声）の、エンジンが返した規約の文"
            " 企業での利用や非商用だけの声があるので、使う前にユーザーへ伝える"
            " generate_narration の前に必ず 1 度呼ぶ エンジンが動いていなければ理由を書いて断る"
            " 呼ぶのは自分の機械の中のエンジンだけで、外のサービスへはつながない" + PAGING_NOTE
        ),
        schema=_schema(
            {
                "voice_id": _integer("規約の文を読みたい声（省くと既定の声）"),
                **_paging(),
            }
        ),
        handler=_list_voices,
        # 返すのはエンジンの名前と声の一覧・規約の文と、本人の設定（エンジンの場所・既定の声）
        # 作品の中身も素材の名前も返さない
        sends=_VOICES,
    ),
    Operation(
        name="generate_narration",
        description=(
            "台本の文をナレーションの声にして、タイムラインに置く 本人が入れて起動した"
            " VOICEVOX か AivisSpeech のエンジンで作り、WAV を作品のフォルダの narration へ書き、"
            "素材として読み込み、音声トラック（混合の作品では空いたレイヤー）の空いた所へ置く"
            " 読み込み・置く・表記を覚えるのは取り消し 1 回で戻る（書いた WAV は残る）"
            " ファイルを書くので毎回ユーザーに確かめる 先に list_voices を呼ぶこと"
            f" 1 回に {TEXT_LIMIT} 文字まで 長い台本は場面ごとに分けて、at_frame をずらして"
            "何回かに分ける 作品を保存していなければ断るので、先に保存してもらう"
            " 使った声の表記（VOICEVOX:ずんだもん など）は作品に覚え、書き出しのときに"
            "ユーザーへ知らせる 台本の文は自分の機械の中のエンジンへだけ渡し、結果には載らない"
            " 本人の設定が入なら、台本から文ごとに字幕を付ける（声の長さに合わせる 素材の字幕"
            "として字幕パネルに出る subtitle_lines が付けた数）"
            " 人も〔字幕〕→〔ナレーションを作る…〕で同じことができる"
        ),
        schema=_schema(
            {
                "text": _string(f"読ませる台本の文（{TEXT_LIMIT} 文字まで）"),
                "voice_id": _integer(
                    "声（list_voices の voice_id） 省くと本人の設定の既定の声"
                    " 本人が AI に声を選ばせない設定なら、渡すと断る"
                ),
                "at_frame": _integer("置く位置 省くと再生位置"),
                "track_id": _string("置きたいトラック 埋まっていればほかの空いたトラックへ"),
                "speed_percent": _integer(
                    f"話す速さ %（{VOICE_SPEED_RANGE[0]}〜{VOICE_SPEED_RANGE[1]} 省くと本人の設定）"
                ),
                "volume_percent": _integer(
                    f"声の大きさ %（{VOLUME_PERCENT_RANGE[0]}〜{VOLUME_PERCENT_RANGE[1]}"
                    " 省くと本人の設定）"
                ),
                "pitch": _integer(f"声の高さ（{PITCH_RANGE[0]}〜{PITCH_RANGE[1]} 既定 0）"),
                "intonation_percent": _integer(
                    f"抑揚 %（{INTONATION_PERCENT_RANGE[0]}〜{INTONATION_PERCENT_RANGE[1]}"
                    " 既定 100 AivisSpeech では感情の強さ）"
                ),
            },
            ["text"],
        ),
        handler=_consented(_generate_narration, _narration_consent),
        approved=_generate_narration,
        writes=True,
        irreversible=True,
        consent=_narration_consent,
        # 台本の文は本人（AI）の文で字幕と文字に当たるが、結果にもエラーの文にも載せない
        # 渡す先は自分の機械の中のエンジンだけで、外（Anthropic）へは送らない
        # 結果は書いた場所（名前と場所）・置いたクリップ（編集の構成）・声の名前と表記（本人の
        # 機械に入れた声と、本人の設定の既定）
        sends=_NARRATION,
        name_keys=frozenset({"file"}),
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
        sends=_CONVERSATION,
        asks=True,
    ),
)


def find_operation(name: str) -> Operation | None:
    for operation in OPERATIONS:
        if operation.name == name:
            return operation
    return None
