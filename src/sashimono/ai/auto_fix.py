"""色と音の雑音の自動補正の計算（#336） AI の道具と、タイムラインの右クリックが同じ所を通る

AI の ``auto_color`` ``reduce_noise`` と、人の〔色を自動で整える…〕〔ノイズを自動で減らす…〕は、
ここで測って同じ命令を作る 入口ごとに書くと、同じクリップで人と AI の直し方が食い違う
命令を当てるのは呼ぶ側（AI は :meth:`EditorHost.apply_commands`、人は窓の ``execute_all`` で
人の印 :attr:`Author.PERSON`）で、どちらも取り消し 1 回で戻る

ここは Qt を知らない 測る材料（サムネイルと波形）は :class:`FixSource` から受け取る
編集画面（:class:`~sashimono.ui.main_window.MainWindow`）も AI のホストもそれを満たす
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Protocol

from sashimono.ai.host import ToolError
from sashimono.core.commands import (
    AddEffect,
    Command,
    ParamPath,
    SetEffectEnabled,
    SetParam,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    MediaId,
    MediaItem,
    Project,
    Track,
    heard_stream,
)
from sashimono.effects import registry
from sashimono.effects.audio import DENOISE_KIND
from sashimono.engine.audio.noise import QUIET_NOISE_DB, NoiseEstimate, estimate_noise
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.auto_color import (
    GRADE_KIND,
    graded,
    pixels_of,
    plan_auto,
    plan_match,
)
from sashimono.engine.cache.thumbnails import Filmstrip

__all__ = [
    "AUTO_GRADE_PARAMS",
    "COLOR_SAMPLE_TILES",
    "ColorFix",
    "ColorRequest",
    "FixSource",
    "NoiseFix",
    "NoiseRequest",
    "can_auto_color",
    "picture_clip",
    "plan_color_fix",
    "plan_noise_fix",
]

#: 色を測るのに使うサムネイルの枚数の上限 0.5 秒ごとの絵なので、長いクリップでも
#: 等間隔に間引いてこれだけを見る 明るさの分布と色の平均はこの数で落ち着く
COLOR_SAMPLE_TILES = 24
#: 色の自動補正が入れる値（拡張色調補正の欄） 掛け直すときはここだけを置き換え、人が足した
#: ほかの欄（リフト・色相など）は残す
AUTO_GRADE_PARAMS = (
    "luma_gain",
    "luma_offset",
    "luma_gamma",
    "sat_gain",
    "red_gain",
    "green_gain",
    "blue_gain",
)


class FixSource(Protocol):
    """測る材料を出す所（編集画面か AI のホスト）"""

    def filmstrip(self, media: MediaItem, stream: int | None = None) -> Filmstrip | None: ...

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None: ...

    def analyze(self, media: MediaItem) -> None: ...


@dataclass(frozen=True, slots=True)
class ColorRequest:
    """色の直し方 AI の引数と右クリックの窓の欄が同じ形"""

    strength: float
    exposure: bool = True
    white_balance: bool = True
    #: 合わせる相手のクリップ ``None`` なら素材ごとに整える
    match: ClipId | None = None


@dataclass(frozen=True, slots=True)
class ColorFix:
    """色の補正の命令と、クリップごとの様子"""

    commands: tuple[Command, ...]
    rows: tuple[dict[str, Any], ...]
    #: もう整っていて何もしなかったクリップ
    unchanged: tuple[ClipId, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class NoiseRequest:
    """雑音の減らし方"""

    reduction: float
    sensitivity: float
    #: 測って掛けないと決めたクリップにも掛けるか
    force: bool = False


@dataclass(frozen=True, slots=True)
class NoiseFix:
    """ノイズ除去の命令と、クリップごとの様子・掛けなかった訳"""

    commands: tuple[Command, ...]
    rows: tuple[dict[str, Any], ...]
    skipped: tuple[dict[str, str], ...] = field(default=())


def can_auto_color(project: Project, track: Track, clip: Clip, *, editing: bool = True) -> bool:
    """色を測れる（``editing`` なら直せる ロックしていない）クリップか

    右クリックの灰色と、窓の合わせる相手の候補に使う AI の道具と同じ決まり
    """
    try:
        _check_picture(project, track, clip, str(clip.id), editing=editing)
    except ToolError:
        return False
    return True


def picture_clip(project: Project, clip_id: object, *, editing: bool = True) -> tuple[Track, Clip]:
    """素材の絵を描くクリップ テキスト・図形・置いたシーン・音だけのクリップは測れないので断る

    変える（``editing``）ならロックしたトラックのクリップも断る 合わせる相手（見るだけ）は
    ロックしていても使う
    """
    located = project.timeline.locate_clip(ClipId(str(clip_id)))
    if located is None:
        raise ToolError(f"クリップが見つかりません: {clip_id}")
    _check_picture(project, *located, str(clip_id), editing=editing)
    return located


def _check_picture(project: Project, track: Track, clip: Clip, name: str, *, editing: bool) -> None:
    media = project.find_media(clip.media_id) if clip.media_id is not None else None
    if media is None or not project.draws_picture(track, clip) or not media.has_video:
        raise ToolError(
            f"{name} は素材の絵を描くクリップではありません（テキスト・図形・置いたシーン・"
            "音だけのクリップは色を測れない） 動画か画像のクリップを指してください"
        )
    if editing and track.locked:
        raise ToolError(
            f"{name} のトラック {track.name} はロックされています"
            " 変えるなら set_track_state でロックを解いてから"
        )


def plan_color_fix(
    source: FixSource, project: Project, targets: list[Clip], request: ColorRequest
) -> ColorFix:
    """クリップの絵を測り、拡張色調補正を積む（もうあれば置き換える）命令を作る"""
    if not request.exposure and not request.white_balance:
        raise ToolError("exposure と white_balance の両方を false にすると何も直しません")
    model: Clip | None = None
    if request.match is not None:
        _, model = picture_clip(project, request.match, editing=False)
        if any(clip.id == model.id for clip in targets):
            raise ToolError("合わせる相手（match_clip_id）を直すクリップに入れないでください")
    _require_strips(source, project, [*targets, *([model] if model is not None else [])])
    reference: Any = None
    if model is not None:
        reference = _clip_pixels(source, project, model, graded_head=True)
    rows: list[dict[str, Any]] = []
    unchanged: list[ClipId] = []
    commands: list[Command] = []
    for clip in targets:
        pixels = _clip_pixels(source, project, clip, graded_head=False)
        if reference is None:
            plan = plan_auto(
                pixels,
                strength=request.strength,
                exposure=request.exposure,
                white_balance=request.white_balance,
            )
        else:
            plan = plan_match(
                pixels,
                reference,
                strength=request.strength,
                exposure=request.exposure,
                white_balance=request.white_balance,
            )
        if not plan.changes and not _has_auto_values(clip):
            unchanged.append(clip.id)
            continue
        # 直す物が無くても、頭の補正に前の自動の値（色合わせなど）が残っていれば 0 へ戻す
        # 戻さないと「整っています」と返しながら前の補正が効き続ける（#387 の Codex の指摘）
        made, replaced = _grade_commands(clip, plan.params)
        commands.extend(made)
        row: dict[str, Any] = {
            "clip_id": str(clip.id),
            "before": plan.before.summary(),
            "after": plan.after.summary(),
            "params": plan.params,
            "replaced": replaced,
        }
        if plan.notes:
            row["notes"] = list(plan.notes)
        rows.append(row)
    return ColorFix(tuple(commands), tuple(rows), tuple(unchanged))


def _require_strips(source: FixSource, project: Project, clips: list[Clip]) -> None:
    """測るクリップの素材のサムネイルがそろっているかを先に見る 足りなければ全部の用意を頼んで断る

    1 本ずつ測りながら断ると、1 本目のまだの素材で止まって残りの用意を頼まず、素材の数だけ
    呼び直すことになる（#387 の CodeRabbit の指摘） 何も作らずに断るのは今までと同じ
    名前は素材の名前なので、名前と場所を送らない人にはブリッジが伏せる（ほかの断りの文と同じ）
    """
    waiting: dict[MediaId, MediaItem] = {}
    for clip in clips:
        media = project.find_media(clip.media_id) if clip.media_id is not None else None
        if media is None or media.id in waiting:
            continue
        strip = source.filmstrip(media, clip.stream_index)
        if strip is None or strip.count == 0:
            waiting[media.id] = media
    if not waiting:
        return
    for media in waiting.values():
        source.analyze(media)
    names = "、".join(media.name for media in waiting.values())
    raise ToolError(
        f"{names} のサムネイルがまだできていません（取り込みの後に裏で作る） 何も変えていません"
        " 少し待ってからもう 1 度呼んでください"
    )


def _clip_tiles(source: FixSource, project: Project, clip: Clip) -> list[Any]:
    """クリップに映る範囲のサムネイル（RGBA） まだ無ければ用意を頼んで断る"""
    media = project.find_media(clip.media_id) if clip.media_id is not None else None
    if media is None:
        raise ToolError(f"{clip.id} の素材が見つかりません")
    # 映すのはクリップの映像の流れ（映像と混合のトラックのどちらも stream_index が絵の流れ）
    # 1 本目で測ると、2 本目を映すクリップの色を別のカメラの絵で決めてしまう
    strip = source.filmstrip(media, clip.stream_index)
    if strip is None or strip.count == 0:
        source.analyze(media)
        raise ToolError(
            f"{media.name} のサムネイルがまだできていません（取り込みの後に裏で作る）"
            " 少し待ってからもう 1 度呼んでください"
        )
    start = clip.source_in
    end = clip.source_out(project.rate)
    if clip.hold_at is not None:
        end = min(end, max(clip.hold_at, start))
    picks = COLOR_SAMPLE_TILES
    numbers = {
        index
        for step in range(picks)
        if (index := strip.index_at(start + (end - start) * Fraction(step, picks))) is not None
    }
    return [tile for index in sorted(numbers) if (tile := strip.tile(index)) is not None]


def _head_grade(clip: Clip) -> Effect | None:
    """クリップの頭（一番先に掛かる所）にある拡張色調補正 色の自動補正が置く所"""
    if clip.effects and clip.effects[0].kind == GRADE_KIND and not clip.effects[0].fixed:
        return clip.effects[0]
    return None


def _has_auto_values(clip: Clip) -> bool:
    """頭の拡張色調補正の自動補正の欄に、0 へ戻すべき値が入っているか

    キーフレームで動いている欄は人が付けた動きなので戻さない（戻すと動きが消える）
    動きを戻す物に数えると、直す物の無いクリップで断ることになり、一緒に頼んだほかの
    クリップも直らない（#387 の CodeRabbit の指摘） 直す物があるクリップで動きを断る
    :func:`_grade_commands` の守りはそのまま
    """
    head = _head_grade(clip)
    if head is None:
        return False
    values = [head.params.get(name) for name in AUTO_GRADE_PARAMS]
    if any(isinstance(value, AnimatedValue) and value.is_animated for value in values):
        return False
    return any(isinstance(value, AnimatedValue) and value.static != 0.0 for value in values)


def _clip_pixels(source: FixSource, project: Project, clip: Clip, *, graded_head: bool) -> Any:
    """クリップの絵の画素（sRGB） ``graded_head`` なら頭の拡張色調補正を掛けた後の色

    合わせる相手は、色の自動補正を掛けた後の見え方に合わせる 素材のままの色に合わせると、
    整えたクリップに合わせたつもりで整える前の色へ寄る
    """
    pixels = pixels_of(_clip_tiles(source, project, clip))
    if len(pixels) == 0:
        raise ToolError(f"{clip.id} の絵がすべて透けていて色を測れません")
    head = _head_grade(clip)
    if graded_head and head is not None and head.enabled:
        values: dict[str, float | bool] = {
            name: value.static
            for name, value in head.params.items()
            if isinstance(value, AnimatedValue)
        }
        values |= {n: v for n, v in head.params.items() if isinstance(v, bool)}
        pixels = graded(pixels, values)
    return pixels


def _grade_commands(clip: Clip, params: dict[str, float]) -> tuple[list[Command], bool]:
    """拡張色調補正をクリップの頭へ積む 頭にもうあれば自動補正の欄だけを置き換える

    頭に積むのは、測ったのが素材の色だから 後ろに積むと、前に積んだエフェクト（色を変える
    物など）を通した後の絵に、素材の色で決めた値を掛けることになる
    返す真偽は置き換えたか
    """
    head = _head_grade(clip)
    if head is None:
        effect = registry.require(GRADE_KIND).create(**params)
        return [AddEffect(clip.id, effect, index=0)], False
    moving = [
        name
        for name in AUTO_GRADE_PARAMS
        if isinstance(value := head.params.get(name), AnimatedValue) and value.is_animated
    ]
    if moving:
        raise ToolError(
            f"{clip.id} の頭の拡張色調補正は {', '.join(moving)} をキーフレームで動かしています"
            " 置き換えると動きが消えるので触りません 外すか、別のクリップで試してください"
        )
    commands: list[Command] = [
        SetParam(
            ParamPath.of_effect(clip.id, head.id, name),
            AnimatedValue(static=params.get(name, 0.0)),
        )
        for name in AUTO_GRADE_PARAMS
    ]
    if not head.enabled and params:
        # 人が切った補正の値だけを直すと、効いていないのに整えたと返すことになる
        # 0 へ戻すだけ（直す物が無い）なら入れ直さない
        commands.append(SetEffectEnabled(clip.id, head.id, True))
    return commands, True


def plan_noise_fix(
    source: FixSource,
    project: Project,
    targets: list[tuple[Track, Clip]],
    request: NoiseRequest,
) -> NoiseFix:
    """音のクリップの雑音を測り、ノイズ除去を積む（もうあれば置き換える）命令を作る

    波形がまだ無い素材があれば、解析を頼んで何も作らずに断る（一部だけ掛けると、どれに
    掛かったのかを追えない）
    """
    waiting: list[MediaItem] = []
    measured: list[tuple[Clip, NoiseEstimate | None]] = []
    for track, clip in targets:
        media = project.find_media(clip.media_id) if clip.media_id is not None else None
        if media is None:
            # 置いたシーンは素材の波形を持たない 測らずに掛けるのは force のときだけ
            measured.append((clip, None))
            continue
        waveform = source.waveform(media, heard_stream(track, clip))
        if waveform is None:
            source.analyze(media)
            waiting.append(media)
            continue
        estimate = estimate_noise(waveform, clip.source_in, clip.source_out(project.rate))
        measured.append((clip, estimate))
    if waiting:
        names = "、".join(dict.fromkeys(media.name for media in waiting))
        raise ToolError(
            f"{names} の波形解析がまだ終わっていません 何も変えていません"
            " 少し待ってから呼んでください"
        )
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    commands: list[Command] = []
    for clip, estimate in measured:
        reason = _noise_skip_reason(estimate)
        if reason and not request.force:
            if any(effect.kind == DENOISE_KIND for effect in clip.effects):
                # 前に掛けたノイズ除去は外さない（人が足した物かもしれない） 効き続けることは
                # 書いておく 書かないと、掛けないと返したのに音が変わったままに見える
                reason += " 前からのノイズ除去はそのまま残しています"
            skipped.append({"clip_id": str(clip.id), "reason": reason})
            continue
        made, replaced = _denoise_commands(clip, request.reduction, request.sensitivity)
        commands.extend(made)
        row: dict[str, Any] = {"clip_id": str(clip.id), "replaced": replaced}
        if estimate is not None:
            row |= {
                "noise_dbfs": estimate.noise_db,
                "speech_dbfs": estimate.speech_db,
                "margin_db": round(estimate.margin_db, 1),
            }
        if reason:
            row["forced"] = reason
        rows.append(row)
    return NoiseFix(tuple(commands), tuple(rows), tuple(skipped))


def _noise_skip_reason(estimate: NoiseEstimate | None) -> str:
    """見積もりから掛けない訳 掛けてよければ空"""
    if estimate is None:
        return "素材の波形が無いか 1 秒より短く、雑音を測れません（置いたシーンなど）"
    if not estimate.audible:
        return (
            f"雑音は {estimate.noise_db:g} dBFS で聞こえない大きさです"
            f"（{QUIET_NOISE_DB:g} dBFS 以下）"
        )
    if not estimate.has_quiet_parts:
        return (
            f"静かな所と大きい所の差が {estimate.margin_db:.1f} dB しかありません"
            "（音楽か、ずっと続く音か、雑音に埋もれた声） 下げると声や音楽まで削れます"
        )
    return ""


def _denoise_commands(
    clip: Clip, reduction: float, sensitivity: float
) -> tuple[list[Command], bool]:
    """ノイズ除去を積む もうあれば値だけを置き換える（重ねて掛けると声まで削れる）"""
    found = next((effect for effect in clip.effects if effect.kind == DENOISE_KIND), None)
    if found is None:
        effect = registry.require(DENOISE_KIND).create(reduction=reduction, sensitivity=sensitivity)
        # 位置は渡さない 固定の項目（音量・フェード）の前へ入る 雑音を下げてから音量を掛ける
        return [AddEffect(clip.id, effect)], False
    moving = [
        name
        for name in ("reduction", "sensitivity")
        if isinstance(value := found.params.get(name), AnimatedValue) and value.is_animated
    ]
    if moving:
        # 拡張色調補正と同じく、人が付けた動きを 1 つの値で黙って消さない（#387 の Codex の指摘）
        raise ToolError(
            f"{clip.id} のノイズ除去は {', '.join(moving)} をキーフレームで動かしています"
            " 置き換えると動きが消えるので触りません 設定パネルで直してください"
        )
    commands: list[Command] = [
        SetParam(ParamPath.of_effect(clip.id, found.id, name), AnimatedValue(static=value))
        for name, value in (("reduction", reduction), ("sensitivity", sensitivity))
    ]
    if not found.enabled:
        commands.append(SetEffectEnabled(clip.id, found.id, True))
    return commands, True
