""".sme プロジェクトファイルの読み書き

JSON にしているのは、外部ツールと AI エージェントから素直に扱えるようにするため
バイナリにすると、AI がプロジェクトを直接読んで状況を把握することも、ユーザーが
壊れたファイルを手で直すこともできなくなる

秒は必ず ``"1001/30000"`` のような分数文字列で書き出す 浮動小数にすると保存と
読み込みを繰り返すだけで値が動く
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from sashimono.core.model import (
    AnimatedValue,
    AudioStreamInfo,
    Blending,
    Clip,
    ClipId,
    Effect,
    EffectId,
    GeneratedSource,
    GroupId,
    Interpolation,
    Keyframe,
    LayerMode,
    Marker,
    MediaId,
    MediaItem,
    ParamValue,
    Project,
    ProjectSettings,
    Scene,
    SceneId,
    SegmentId,
    Timeline,
    Track,
    TrackId,
    TrackKind,
    Transcript,
    TranscriptSegment,
    VideoStreamInfo,
    Word,
)
from sashimono.core.model.easing import CURVES
from sashimono.core.timebase import FrameRate

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "LEGACY_SUFFIXES",
    "ProjectFileError",
    "clip_from_json",
    "clip_to_json",
    "effect_from_json",
    "effect_to_json",
    "json_text",
    "load_project",
    "project_from_dict",
    "project_to_dict",
    "save_project",
    "source_from_json",
    "source_to_json",
]

#: 対になっていない代用符号 1 文字
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def json_text(value: Any, *, indent: int | None = 2, default: Any = None) -> str:
    r"""日本語を読めるまま JSON の文字にする 対になっていない代用符号だけは ``\uXXXX`` で書く

    AviUtl1 のダイアログの ``"\255"`` のように、UTF-8 で読めないバイトを持つ値は、
    そのバイトを ``surrogateescape`` の形（U+DC80〜U+DCFF の 1 文字）で文字の中に持つ
    ``ensure_ascii=False`` のまま UTF-8 で書くとそこで ``UnicodeEncodeError`` になり、
    プロジェクトごと保存できない ファイル全体を ``ensure_ascii=True`` にすると日本語が
    すべて ``\uXXXX`` になって読めなくなるので、その文字だけを逃がす
    ``json.loads`` は ``\udcff`` を同じ 1 文字へ戻すので、読み直すと元の値のまま
    （JSON の中で代用符号が出るのは文字の中だけなので、置き換えても形は崩れない）
    """
    text = json.dumps(value, ensure_ascii=False, indent=indent, default=default)
    return _LONE_SURROGATE.sub(lambda found: f"\\u{ord(found[0]):04x}", text)


FORMAT_NAME = "sashimono-project"
#: 2 でシーン（``scenes`` と ``Clip.scene_id``）とグループ（``Clip.group_id``）を足した
#: 1 の本体は 2 を開くと「更新してください」と言う（シーンを黙って捨てて開くと、
#: 置いたシーンが何も映らない穴になり、保存し直すとシーンごと消える）
#: 3 で重ね合わせの方法（``settings.blending``）を足した（Issue #65） 2 までの本体は項目を
#: 知らないので、sRGB で混ぜる作品をリニアで描き、保存し直すと項目ごと消えて見た目が変わる
#: 黙って変えるより「更新してください」で止める方がよいので、版を上げた
#: 4 で絵を止める時刻（``Clip.hold_at``）を足した（Issue #115） 3 までの本体は項目を
#: 知らないので、止めた絵が動き出す（素材の終わりの後は何も映らなくなる）うえ、保存し直すと
#: 項目ごと消える 3 と同じ理由で版を上げた 3 までのファイルは止めないクリップとして開く
#: 5 で素材の中の時刻を素材の頭から数えるようにした（Issue #123） 頭が 0 の素材は何も
#: 変わらない 頭が 0 より後ろの素材は、4 までは PTS そのままで数えていて、ふつうに置くと
#: 頭の絵が止まったまま音も鳴らなかった MP3（頭が 0.025 秒）の音は 25ms 早く鳴るようになる
#: 版を上げたのは映像の終わり（``VideoStreamInfo.end_time``）のため 4 のファイルの値は
#: PTS そのままで、頭 5 秒・長さ 2 秒の素材なら 7 秒と書いてある 版を分けないと、読む側が
#: どちらの数え方か見分けられず、テンプレートを置いたときに映像の終わりの後で絵を止めて
#: 何も映らなくなる 4 までのファイルは映像の終わりを持たない素材として開く（絵を止める
#: 所で素材を開き直して取る :func:`sashimono.compat.catalog.gather_media`）
#: ``source_in`` と ``hold_at`` はそのまま読む 直すには素材を開いて頭の時刻を知る必要があり、
#: ここ（コア層）では開けない 頭が 0 の素材ではもともと正しい 頭が 0 より後ろの素材で、
#: 手で素材の中の位置を頭の時刻ぶん後ろへずらして映るようにしていたクリップは、5 で開くと
#: その分だけ後ろを読む（ずらさずに置いたクリップは、ここで初めて正しく映る）
#: 6 でフィルタのクリップ（生成オブジェクト ``filter`` Issue #27）を足した 項目の形は
#: 変わらない（生成オブジェクトの種類の文字列が 1 つ増えただけ）が、5 までの本体は
#: 種類を知らないので、下の絵に掛かるはずのエフェクトを黙って描かず、何も無い所として
#: 扱う 3 と同じく「更新してください」で止める方がよいので版を上げた
#: 5 までのファイルはフィルタを持たないので、何も直さずにそのまま読める
#: 7 で混合トラック（トラックの種類 ``mixed``）と、そこで読むクリップの
#: ``audio_stream`` ``show_picture``、置き方の方式（``settings.layer_mode``）を足した
#: （Issue #27） 6 までの本体は種類を知らないので「未知のトラック種別」で開けないか、
#: 項目を捨てて音の鳴らないクリップにする 3 と同じく「更新してください」で止める
#: 6 までのファイルは混合トラックを持たないので、分ける方式のまま何も直さずに読める
#: 8 で字幕を音声ストリームごとに持つようにした（素材の ``transcripts`` 利用者の要望）
#: 7 までの本体は項目を知らず、字幕の無い素材として開いて、保存し直すと字幕が消える
#: 3 と同じく「更新してください」で止める 7 までのファイルの ``transcript`` は、1 本目の
#: 音声の字幕として読む（前は素材に 1 つで、起こしたのは 1 本目の音だったため）
FORMAT_VERSION = 8

#: 映像の終わり（``end_time``）を素材の頭から数え始めた版 これより前の値は捨てる
MEDIA_CLOCK_VERSION = 5

#: プロジェクトファイルの拡張子
SUFFIX = ".sme"

# 旧名を残す: ここから（名前の一括置換でも書き換えない 古い版のファイルを読むのに要る）
#: 読むときだけ受け付ける、昔の名前と拡張子
#:
#: 公開前に ``NovaEdit`` から、公開後に商標の都合で ``Kumiki`` から改名した
#: 手元に保存済みのものがあるので、**読む側だけ**受ける 書くときは常に新しい名前で
#: 書く ここから消すと、改名前に作った作品が「プロジェクトファイルではない」で開けなくなる
LEGACY_FORMAT_NAMES = ("kumiki-project", "novaedit-project")
LEGACY_SUFFIXES = (".kmk", ".nvep")
# 旧名を残す: ここまで


class ProjectFileError(Exception):
    """プロジェクトファイルが読めない、または想定した形をしていない"""


# --- 基本型 ---------------------------------------------------------------


def _fraction_to_json(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _fraction_from_json(value: object, field: str) -> Fraction:
    """分数の項目を読む 書くのは常に ``"分子/分母"`` の文字、既定値だけ整数

    真偽値は ``int`` の仲間なので、断らないと ``true`` が 1（秒・倍・分の 1 秒）として
    読めてしまう こちらが真偽値を書くことは無く、どの分数の項目（時刻・速さ・長さ・
    時間の単位・フレームレート）でも壊れたファイルなので、読み替えずに断る
    """
    if isinstance(value, bool):
        raise ProjectFileError(f"{field} が分数ではない: {value!r}")
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ProjectFileError(f"{field} が分数として読めない: {value!r}") from exc
    if isinstance(value, int):
        return Fraction(value)
    raise ProjectFileError(f"{field} が分数ではない: {value!r}")


def _rate_to_json(rate: FrameRate) -> str:
    return f"{rate.num}/{rate.den}"


def _rate_from_json(value: object, field: str) -> FrameRate:
    fraction = _fraction_from_json(value, field)
    return FrameRate(fraction.numerator, fraction.denominator)


def _require(data: object, field: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ProjectFileError(f"{field} がオブジェクトではない")
    return data


def _get_int(data: dict[str, Any], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProjectFileError(f"{key} が整数ではない: {value!r}")
    return value


def _get_str(data: dict[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ProjectFileError(f"{key} が文字列ではない: {value!r}")
    return value


def _get_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ProjectFileError(f"{key} が真偽値ではない: {value!r}")
    return value


def _get_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProjectFileError(f"{key} が数値ではない: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        # JSON の読み込みは NaN と Infinity を受けてしまう 通すと描画や音の計算が壊れる
        raise ProjectFileError(f"{key} が有限の数ではない: {value!r}")
    return result


def _get_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ProjectFileError(f"{key} が配列ではない: {value!r}")
    return value


# --- エフェクト -----------------------------------------------------------


#: 曲線の名前（``curve``）を持てる補間方法
_EASINGS = frozenset({Interpolation.EASE_IN, Interpolation.EASE_OUT, Interpolation.EASE_IN_OUT})


def _keyframe_to_json(keyframe: Keyframe) -> dict[str, Any]:
    data: dict[str, Any] = {
        "frame": keyframe.frame,
        "value": keyframe.value,
        "interpolation": keyframe.interpolation.value,
    }
    if keyframe.control_points is not None:
        data["control_points"] = list(keyframe.control_points)
    if keyframe.curve:
        # 空のときは書かない 曲線の無いキーフレームの形を、前の版の保存と同じに保つ
        data["curve"] = keyframe.curve
    return data


def _keyframe_from_json(raw: object) -> Keyframe:
    data = _require(raw, "keyframe")
    name = _get_str(data, "interpolation", Interpolation.LINEAR.value)
    try:
        interpolation = Interpolation(name)
    except ValueError as exc:
        raise ProjectFileError(f"未知の補間方法: {name!r}") from exc

    points_raw = data.get("control_points")
    control_points: tuple[float, float, float, float] | None = None
    if points_raw is not None:
        if not isinstance(points_raw, list) or len(points_raw) != 4:
            raise ProjectFileError(f"control_points は 4 要素の配列: {points_raw!r}")
        a, b, c, d = (float(v) for v in points_raw)
        if not all(math.isfinite(value) for value in (a, b, c, d)):
            # 非有限の制御点は補間を通して描画へ流れ、GL の値が壊れる
            raise ProjectFileError(f"control_points に扱えない数がある: {points_raw!r}")
        control_points = (a, b, c, d)

    curve = _get_str(data, "curve", "")
    if curve and curve not in CURVES:
        raise ProjectFileError(f"未知の曲線: {curve!r}")
    if curve and interpolation not in _EASINGS:
        raise ProjectFileError(f"曲線の名前はイージングの点にだけ付く: {name!r} に {curve!r}")

    return Keyframe(
        frame=_get_int(data, "frame"),
        value=_get_float(data, "value", 0.0),
        interpolation=interpolation,
        control_points=control_points,
        curve=curve,
    )


def _param_to_json(value: ParamValue) -> Any:
    if isinstance(value, AnimatedValue):
        # アニメーションしていない値は素の数値で書く プロジェクトファイルの
        # 大半はこちらなので、これだけでファイルサイズがかなり変わる
        if not value.is_animated:
            return {"static": value.static}
        return {
            "static": value.static,
            "keyframes": [_keyframe_to_json(k) for k in value.keyframes],
        }
    if isinstance(value, tuple):
        return list(value)
    return value


def _param_from_json(raw: object) -> ParamValue:
    if isinstance(raw, dict):
        keyframes = tuple(_keyframe_from_json(k) for k in _get_list(raw, "keyframes"))
        return AnimatedValue(static=_get_float(raw, "static", 0.0), keyframes=keyframes)
    if isinstance(raw, list):
        values = tuple(float(v) for v in raw)
        if not all(math.isfinite(v) for v in values):
            raise ProjectFileError(f"パラメータに有限でない数がある: {raw!r}")
        return values
    if isinstance(raw, bool | int | str):
        return raw
    raise ProjectFileError(f"パラメータとして読めない値: {raw!r}")


def _params_to_json(params: dict[str, ParamValue]) -> dict[str, Any]:
    return {name: _param_to_json(value) for name, value in params.items()}


def _params_from_json(raw: object, field: str) -> dict[str, ParamValue]:
    if not isinstance(raw, dict):
        raise ProjectFileError(f"{field} がオブジェクトではない: {raw!r}")
    return {name: _param_from_json(value) for name, value in raw.items()}


def effect_to_json(effect: Effect) -> dict[str, Any]:
    """エフェクト 1 つを辞書へ プリセットの保存でも使う"""
    return {
        "id": effect.id,
        "kind": effect.kind,
        "enabled": effect.enabled,
        # 偽でも書く 項目が無いことを「印を知らない前の本体が書いた」の目印に使う
        # （:func:`_promote_placed_volume`） 版は上げない 前の本体は知らない項目を捨てて開く
        "fixed": effect.fixed,
        "params": _params_to_json(effect.params),
    }


def effect_from_json(raw: object) -> Effect:
    """:func:`effect_to_json` の逆"""
    data = _require(raw, "effect")
    return Effect(
        kind=_get_str(data, "kind"),
        params=_params_from_json(data.get("params", {}), "params"),
        enabled=_get_bool(data, "enabled", True),
        id=EffectId(_get_str(data, "id")),
        fixed=_get_bool(data, "fixed", False),
    )


#: 素材を置いたときに音声のクリップへ付く音量調整の種類
#: :data:`sashimono.core.commands.insert.VOLUME_EFFECT_KIND` と同じ値 読み書きの層から
#: 命令の層を読まないためにここへも書き、食い違わないことは試験で見る
_PLACED_VOLUME_KIND = "audio_volume"


def _promote_placed_volume(
    effects: tuple[Effect, ...], raw_effects: list[Any], *, media_clip: bool
) -> tuple[Effect, ...]:
    """前の本体が素材を置いたときに付けた音量調整を、固定の項目へ格上げする

    #145 から、素材を置くと音声のクリップの先頭へ音量調整が付く そのころは固定の印が
    無かったので、そのままだと外せるふつうのエフェクトとして開く 後で固定の音量調整が
    足されると同じ物が 2 つ並び、どちらが最初からある欄か分からなくなる

    見分け方は 音声トラックの素材のクリップで、先頭のエフェクトが音量調整で、その書き物に
    印の項目が無い（前の本体が書いた）こと 映像トラックのクリップには置いたときに何も
    付かないので、そこの音量調整は本人が足した物 格上げすると外せなくなる 値は見ない
    置いた後で音量を動かした物も、置いたときに付いた物に変わりはない
    #145 より前に音声のクリップの先頭へ自分で足した音量調整も格上げされるが、
    それもクリップの音量の欄として扱って困らない（無効にはできる）
    """
    if not media_clip or not effects or not raw_effects:
        return effects
    first, raw_first = effects[0], raw_effects[0]
    if first.kind != _PLACED_VOLUME_KIND or not isinstance(raw_first, dict):
        return effects
    if "fixed" in raw_first or any(e.fixed for e in effects):
        return effects
    return (replace(first, fixed=True), *effects[1:])


def source_to_json(source: GeneratedSource) -> dict[str, Any]:
    return {"kind": source.kind, "params": _params_to_json(source.params)}


def source_from_json(raw: object) -> GeneratedSource:
    data = _require(raw, "source")
    return GeneratedSource(
        kind=_get_str(data, "kind"),
        params=_params_from_json(data.get("params", {}), "source.params"),
    )


# --- 字幕 -----------------------------------------------------------------


def _transcript_to_json(transcript: Transcript) -> dict[str, Any]:
    return {
        "language": transcript.language,
        "model": transcript.model,
        "segments": [
            {
                "id": segment.id,
                "start": _fraction_to_json(segment.start),
                "end": _fraction_to_json(segment.end),
                "text": segment.text,
                "speaker": segment.speaker,
                "edited": segment.edited,
                "words": [
                    {
                        "start": _fraction_to_json(word.start),
                        "end": _fraction_to_json(word.end),
                        "text": word.text,
                    }
                    for word in segment.words
                ],
            }
            for segment in transcript.segments
        ],
    }


def _transcript_from_json(raw: object) -> Transcript:
    data = _require(raw, "transcript")
    segments = []
    for entry in _get_list(data, "segments"):
        segment_data = _require(entry, "segment")
        speaker = segment_data.get("speaker")
        if speaker is not None and not isinstance(speaker, str):
            raise ProjectFileError(f"speaker が文字列ではない: {speaker!r}")
        words = tuple(
            Word(
                start=_fraction_from_json(_require(w, "word")["start"], "word.start"),
                end=_fraction_from_json(_require(w, "word")["end"], "word.end"),
                text=_get_str(_require(w, "word"), "text"),
            )
            for w in _get_list(segment_data, "words")
        )
        segments.append(
            TranscriptSegment(
                start=_fraction_from_json(segment_data.get("start"), "segment.start"),
                end=_fraction_from_json(segment_data.get("end"), "segment.end"),
                text=_get_str(segment_data, "text"),
                words=words,
                speaker=speaker,
                edited=_get_bool(segment_data, "edited", False),
                id=SegmentId(_get_str(segment_data, "id")),
            )
        )
    return Transcript(
        segments=tuple(segments),
        language=_get_str(data, "language"),
        model=_get_str(data, "model"),
    )


# --- 素材 -----------------------------------------------------------------


def _media_to_json(item: MediaItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "path": str(item.path),
        "duration": _fraction_to_json(item.duration),
        "display_name": item.display_name,
        "video_streams": [
            {
                "index": stream.index,
                "width": stream.width,
                "height": stream.height,
                "frame_rate": _rate_to_json(stream.frame_rate),
                "time_base": _fraction_to_json(stream.time_base),
                "codec": stream.codec,
                "pixel_format": stream.pixel_format,
                "rotation": stream.rotation,
                "end_time": (
                    _fraction_to_json(stream.end_time) if stream.end_time is not None else None
                ),
            }
            for stream in item.video_streams
        ],
        "audio_streams": [
            {
                "index": stream.index,
                "sample_rate": stream.sample_rate,
                "channels": stream.channels,
                "time_base": _fraction_to_json(stream.time_base),
                "codec": stream.codec,
                "language": stream.language,
            }
            for stream in item.audio_streams
        ],
        "transcripts": [
            {"stream": index, "transcript": _transcript_to_json(transcript)}
            for index, transcript in item.transcripts
        ],
    }


def _media_from_json(raw: object, version: int) -> MediaItem:
    data = _require(raw, "media")
    video_streams = []
    for s in _get_list(data, "video_streams"):
        stream_data = _require(s, "video_stream")
        # 項目が無いのは、映像の終わりを覚える前に取り込んだ素材 コンテナの長さで代わりにする
        # 版 4 の値は PTS そのままの数え方（:data:`MEDIA_CLOCK_VERSION`） 頭が 0 より後ろの
        # 素材では今の数え方と頭の時刻ぶん違い、どの素材がそうかはファイルからは分からないので、
        # 覚える前の素材と同じ扱いにする 使う所（絵を止める時刻を決める所）が開き直して取る
        end_raw = stream_data.get("end_time") if version >= MEDIA_CLOCK_VERSION else None
        video_streams.append(
            VideoStreamInfo(
                index=_get_int(stream_data, "index"),
                width=_get_int(stream_data, "width"),
                height=_get_int(stream_data, "height"),
                frame_rate=_rate_from_json(stream_data.get("frame_rate"), "frame_rate"),
                time_base=_fraction_from_json(stream_data.get("time_base"), "time_base"),
                codec=_get_str(stream_data, "codec"),
                pixel_format=_get_str(stream_data, "pixel_format"),
                rotation=_get_int(stream_data, "rotation", 0),
                end_time=(
                    _fraction_from_json(end_raw, "end_time") if end_raw is not None else None
                ),
            )
        )

    audio_streams = []
    for s in _get_list(data, "audio_streams"):
        stream_data = _require(s, "audio_stream")
        language = stream_data.get("language")
        if language is not None and not isinstance(language, str):
            raise ProjectFileError(f"language が文字列ではない: {language!r}")
        audio_streams.append(
            AudioStreamInfo(
                index=_get_int(stream_data, "index"),
                sample_rate=_get_int(stream_data, "sample_rate"),
                channels=_get_int(stream_data, "channels"),
                time_base=_fraction_from_json(stream_data.get("time_base"), "time_base"),
                codec=_get_str(stream_data, "codec"),
                language=language,
            )
        )

    item = MediaItem(
        path=Path(_get_str(data, "path")),
        duration=_fraction_from_json(data.get("duration", 0), "duration"),
        video_streams=tuple(video_streams),
        audio_streams=tuple(audio_streams),
        display_name=_get_str(data, "display_name"),
        id=MediaId(_get_str(data, "id")),
    )
    # 7 までは素材に 1 つ（1 本目の音声の字幕） 8 からは音声ストリームごと
    legacy = data.get("transcript")
    if legacy is not None:
        item = item.with_transcript(_transcript_from_json(legacy))
    for entry in _get_list(data, "transcripts"):
        pair = _require(entry, "transcript")
        item = item.with_transcript(
            _transcript_from_json(pair.get("transcript")), _get_int(pair, "stream")
        )
    return item


# --- タイムライン ---------------------------------------------------------


def clip_to_json(clip: Clip) -> dict[str, Any]:
    written: dict[str, Any] = {
        "id": clip.id,
        "timeline_start": clip.timeline_start,
        "duration": clip.duration,
        "media_id": clip.media_id,
        "source": source_to_json(clip.source) if clip.source is not None else None,
        "source_in": _fraction_to_json(clip.source_in),
        "stream_index": clip.stream_index,
        "speed": _fraction_to_json(clip.speed),
        "hold_at": _fraction_to_json(clip.hold_at) if clip.hold_at is not None else None,
        "opacity": _param_to_json(clip.opacity),
        "blend_mode": clip.blend_mode,
        "link_group": clip.link_group,
        "scene_id": clip.scene_id,
        "group_id": clip.group_id,
        "clip_to_below": clip.clip_to_below,
        "audio_stream": clip.audio_stream,
        "show_picture": clip.show_picture,
        "enabled": clip.enabled,
        "effects": [effect_to_json(e) for e in clip.effects],
        "after_effects": [effect_to_json(e) for e in clip.after_effects],
    }
    if clip.native_size:
        # 真のときだけ書く 項目が無いことを「画面に収めて描いていた前の版」の目印にする
        # 版は上げない 前の本体は知らない項目を捨てて開き、画面に収めて描く（見た目は
        # 変わるが開ける） 版を上げると、前の本体では開くことさえできなくなる
        written["native_size"] = True
    return written


def clip_from_json(raw: object, *, on_audio_track: bool = False) -> Clip:
    """:func:`clip_to_json` の逆

    ``on_audio_track`` は置かれているトラックが音声トラックか 前の本体が素材を置いたときに
    付けた音量調整を見分けるのに使う（:func:`_promote_placed_volume`） トラックの外で読む
    エイリアスでは偽のまま
    """
    data = _require(raw, "clip")
    media_id = data.get("media_id")
    if media_id is not None and not isinstance(media_id, str):
        raise ProjectFileError(f"media_id が文字列ではない: {media_id!r}")
    link_group = data.get("link_group")
    if link_group is not None and not isinstance(link_group, str):
        raise ProjectFileError(f"link_group が文字列ではない: {link_group!r}")
    scene_id = data.get("scene_id")
    if scene_id is not None and not isinstance(scene_id, str):
        raise ProjectFileError(f"scene_id が文字列ではない: {scene_id!r}")
    group_id = data.get("group_id")
    if group_id is not None and not isinstance(group_id, str):
        raise ProjectFileError(f"group_id が文字列ではない: {group_id!r}")

    opacity = _param_from_json(data.get("opacity", {"static": 1.0}))
    if not isinstance(opacity, AnimatedValue):
        raise ProjectFileError(f"opacity がアニメーション値ではない: {opacity!r}")

    source_raw = data.get("source")
    # 版 3 までは項目が無い そのころは絵を止める仕組みが無かったので、止めないで開く
    hold_raw = data.get("hold_at")
    # 版 6 までは項目が無い 混合トラックが無く、どちらも読まれないので既定で開く
    audio_stream = data.get("audio_stream")
    if audio_stream is not None and (
        not isinstance(audio_stream, int) or isinstance(audio_stream, bool)
    ):
        raise ProjectFileError(f"audio_stream が整数ではない: {audio_stream!r}")
    raw_effects = _get_list(data, "effects")
    effects = _promote_placed_volume(
        tuple(effect_from_json(e) for e in raw_effects),
        raw_effects,
        media_clip=on_audio_track
        and media_id is not None
        and source_raw is None
        and scene_id is None,
    )
    return Clip(
        timeline_start=_get_int(data, "timeline_start"),
        duration=_get_int(data, "duration"),
        media_id=MediaId(media_id) if media_id is not None else None,
        source=source_from_json(source_raw) if source_raw is not None else None,
        source_in=_fraction_from_json(data.get("source_in", 0), "source_in"),
        stream_index=_get_int(data, "stream_index", 0),
        speed=_fraction_from_json(data.get("speed", 1), "speed"),
        hold_at=_fraction_from_json(hold_raw, "hold_at") if hold_raw is not None else None,
        effects=effects,
        after_effects=tuple(effect_from_json(e) for e in _get_list(data, "after_effects")),
        opacity=opacity,
        blend_mode=_get_str(data, "blend_mode", "normal"),
        link_group=GroupId(link_group) if link_group is not None else None,
        scene_id=SceneId(scene_id) if scene_id is not None else None,
        group_id=GroupId(group_id) if group_id is not None else None,
        clip_to_below=_get_bool(data, "clip_to_below", False),
        audio_stream=audio_stream,
        show_picture=_get_bool(data, "show_picture", True),
        enabled=_get_bool(data, "enabled", True),
        id=ClipId(_get_str(data, "id")),
        # 項目が無ければ画面に収める（:func:`clip_to_json`）
        native_size=_get_bool(data, "native_size", False),
    )


def _track_to_json(track: Track) -> dict[str, Any]:
    return {
        "id": track.id,
        "kind": track.kind.value,
        "name": track.name,
        "locked": track.locked,
        "muted": track.muted,
        "solo": track.solo,
        "height": track.height,
        "volume_db": track.volume_db,
        "pan": track.pan,
        "effects": [effect_to_json(e) for e in track.effects],
        "clips": [clip_to_json(c) for c in track.clips],
    }


def _track_from_json(raw: object) -> Track:
    data = _require(raw, "track")
    kind_name = _get_str(data, "kind")
    try:
        kind = TrackKind(kind_name)
    except ValueError as exc:
        raise ProjectFileError(f"未知のトラック種別: {kind_name!r}") from exc

    return Track(
        kind=kind,
        name=_get_str(data, "name"),
        clips=tuple(
            clip_from_json(c, on_audio_track=kind is TrackKind.AUDIO)
            for c in _get_list(data, "clips")
        ),
        effects=tuple(effect_from_json(e) for e in _get_list(data, "effects")),
        locked=_get_bool(data, "locked", False),
        muted=_get_bool(data, "muted", False),
        solo=_get_bool(data, "solo", False),
        height=_get_int(data, "height", 60),
        volume_db=_get_float(data, "volume_db", 0.0),
        pan=_get_float(data, "pan", 0.0),
        id=TrackId(_get_str(data, "id")),
    )


def _timeline_to_json(timeline: Timeline) -> dict[str, Any]:
    return {
        "rate": _rate_to_json(timeline.rate),
        "tracks": [_track_to_json(t) for t in timeline.tracks],
        "markers": [
            {"frame": m.frame, "label": m.label, "color": m.color} for m in timeline.markers
        ],
        "work_area": list(timeline.work_area) if timeline.work_area is not None else None,
    }


def _timeline_from_json(raw: object) -> Timeline:
    data = _require(raw, "timeline")
    work_area_raw = data.get("work_area")
    work_area: tuple[int, int] | None = None
    if work_area_raw is not None:
        if not isinstance(work_area_raw, list) or len(work_area_raw) != 2:
            raise ProjectFileError(f"work_area は 2 要素の配列: {work_area_raw!r}")
        work_area = (int(work_area_raw[0]), int(work_area_raw[1]))

    markers = tuple(
        Marker(
            frame=_get_int(_require(m, "marker"), "frame"),
            label=_get_str(_require(m, "marker"), "label"),
            color=_get_str(_require(m, "marker"), "color", "#ffcc00"),
        )
        for m in _get_list(data, "markers")
    )
    return Timeline(
        rate=_rate_from_json(data.get("rate"), "rate"),
        tracks=tuple(_track_from_json(t) for t in _get_list(data, "tracks")),
        markers=markers,
        work_area=work_area,
    )


# --- プロジェクト ---------------------------------------------------------


def project_to_dict(project: Project) -> dict[str, Any]:
    """プロジェクトを JSON にできる辞書へ"""
    settings = project.settings
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "name": project.name,
        "settings": {
            "width": settings.width,
            "height": settings.height,
            "frame_rate": _rate_to_json(settings.frame_rate),
            "sample_rate": settings.sample_rate,
            "channels": settings.channels,
            "color_space": settings.color_space,
            "blending": settings.blending,
            "layer_mode": settings.layer_mode,
        },
        "media": [_media_to_json(m) for m in project.media],
        "timeline": _timeline_to_json(project.timeline),
        "scenes": [
            {"id": scene.id, "name": scene.name, "timeline": _timeline_to_json(scene.timeline)}
            for scene in project.scenes
        ],
    }


def project_from_dict(data: object) -> Project:
    """:func:`project_to_dict` の出力からプロジェクトを復元する

    壊れた値はどの段で見つかっても :class:`ProjectFileError` にして返す 型を 1 つずつ
    確かめる検査をすり抜けた値は、モデルの検査（ValueError）や数の変換（TypeError
    など）で止まる 開く側は ProjectFileError しか受けないので、素のまま漏らすと
    壊れたファイル 1 つで起動ごと落ちる
    """
    try:
        return _project_from_dict(data)
    except ProjectFileError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        OverflowError,
        ZeroDivisionError,
        RecursionError,
    ) as exc:
        raise ProjectFileError(f"壊れた値がある: {exc}") from exc


def _project_from_dict(data: object) -> Project:
    root = _require(data, "プロジェクト")

    format_name = _get_str(root, "format")
    if format_name not in (FORMAT_NAME, *LEGACY_FORMAT_NAMES):
        raise ProjectFileError(f"Sashimono のプロジェクトファイルではない: format={format_name!r}")
    version = _get_int(root, "version", 0)
    if version > FORMAT_VERSION:
        # 壊れているのではなく、こちらが古い 直す手立てを言う
        # 自動更新を入れたあとは、ここが「更新してください」の入口になる
        raise ProjectFileError(
            f"新しい形式のプロジェクトファイル (version {version})"
            f"このバージョンが対応しているのは {FORMAT_VERSION} までです"
            "Sashimono を新しい版に更新してください"
        )

    settings_data = _require(root.get("settings"), "settings")
    settings = ProjectSettings(
        width=_get_int(settings_data, "width", 1920),
        height=_get_int(settings_data, "height", 1080),
        frame_rate=_rate_from_json(settings_data.get("frame_rate", "30/1"), "frame_rate"),
        sample_rate=_get_int(settings_data, "sample_rate", 48000),
        channels=_get_int(settings_data, "channels", 2),
        color_space=_get_str(settings_data, "color_space", "rec709"),
        # 版 2 までは項目が無い 重ね合わせを選べるようになる前のファイルで、そのころは
        # リニアで混ぜていたので、リニアで開く 新規作成の既定（sRGB）で開くと、
        # 半透明の文字やフェードが保存したときより暗くなる
        # 版 3 は必ず書くので、無ければ壊れたファイル リニアで補うと sRGB の作品が
        # 黙って明るくなるので、空の値として断る（モデルの検査が ProjectFileError にする）
        blending=_get_str(settings_data, "blending", Blending.LINEAR if version <= 2 else ""),
        # 版 6 までは項目が無い そのころは分ける方式しか無かった
        layer_mode=_get_str(settings_data, "layer_mode", LayerMode.SEPARATED),
    )

    timeline = _timeline_from_json(root.get("timeline", {"rate": "30/1"}))
    if timeline.rate != settings.frame_rate:
        raise ProjectFileError(
            "タイムラインとプロジェクト設定のフレームレートが食い違っている: "
            f"{timeline.rate} と {settings.frame_rate}"
        )

    try:
        # シーンの中のクリップも、長さ 0 などで ValueError を投げる try の外に置くと、
        # 開く側が ProjectFileError しか受けないので、壊れたファイルで落ちる
        scenes = tuple(_scene_from_json(raw) for raw in _get_list(root, "scenes"))
        return Project(
            settings=settings,
            timeline=timeline,
            media=tuple(_media_from_json(m, version) for m in _get_list(root, "media")),
            name=_get_str(root, "name", "無題"),
            scenes=scenes,
        )
    except ValueError as exc:
        # シーンの入れ子が自分へ戻っている、など 壊れたファイルとして伝える
        raise ProjectFileError(str(exc)) from exc


def _scene_from_json(raw: object) -> Scene:
    data = _require(raw, "scene")
    return Scene(
        name=_get_str(data, "name", "シーン"),
        timeline=_timeline_from_json(data.get("timeline", {"rate": "30/1"})),
        id=SceneId(_get_str(data, "id")),
    )


def save_project(project: Project, path: Path) -> None:
    """プロジェクトをファイルへ書き出す

    一時ファイルへ書いてから差し替える 書き込み中に落ちても、既存のプロジェクト
    ファイルは無傷で残る 編集作業をまるごと失うのが一番痛い失敗なので、
    ここは常に atomic にする
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    payload = json_text(project_to_dict(project))

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_project(path: Path) -> Project:
    """プロジェクトファイルを読み込む"""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProjectFileError(f"プロジェクトファイルを開けない: {path}") from exc
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ProjectFileError(f"JSON として読めない: {path} ({exc})") from exc

    project = project_from_dict(data)
    # ファイル名をプロジェクト名の既定にする 名前が入っていない古いファイル対策
    if not project.name or project.name == "無題":
        project = replace(project, name=path.stem)
    return project
