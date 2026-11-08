"""作品が素材のどこを使い、どのファイルを指すかを数える（#423）

持ち出し用のパッケージに何を入れるかを決める 作品は書き換えない（読むだけ）
パスの付け替え（:func:`map_paths`）も新しい作品を返すだけで、元の作品はそのまま
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from sashimono.compat.aviutl.catalog import PREFIX as SCRIPT_PREFIX
from sashimono.core.model import (
    Clip,
    Effect,
    GeneratedSource,
    MediaId,
    MediaItem,
    ParamValue,
    Project,
    Stroke,
    Timeline,
)
from sashimono.effects import registry
from sashimono.effects.sources import source_registry
from sashimono.effects.spec import FileSpec, FontSpec
from sashimono.effects.strokes import STROKE
from sashimono.engine.audio.mixer import history_seconds
from sashimono.engine.decode.probe import STILL_SUFFIXES
from sashimono.engine.package.remux import Span, spans_union

__all__ = [
    "Usage",
    "file_paths",
    "fonts_used",
    "map_paths",
    "media_spans",
    "scripts_used",
    "usage",
]


@dataclass(frozen=True, slots=True)
class Usage:
    """作品が外に頼る物の一覧"""

    #: 素材ごとの使う区間 ``None`` はまるごと入れる素材（静止画・範囲を決められない物）
    spans: dict[MediaId, list[Span] | None]
    #: 設定（画像合成・模様・音声波形）が指すファイル 作品に書いてあるままの文字
    files: tuple[str, ...]
    #: テキストが使う書体の名前 パッケージには入れない（配布の許諾が書体ごとに違う）
    fonts: tuple[str, ...]
    #: 使っている AviUtl のスクリプトのファイル（スクリプトの置き場から数えた場所）
    #: 他人の配布物なので入れない（再配布の許諾の問題）
    scripts: tuple[str, ...]


def _timelines(project: Project) -> Iterator[Timeline]:
    yield project.timeline
    for scene in project.scenes:
        yield scene.timeline


def _clips(project: Project) -> Iterator[tuple[Timeline, Clip]]:
    for timeline in _timelines(project):
        for track in timeline.tracks:
            for clip in track.clips:
                yield timeline, clip


def _all_effects(project: Project) -> Iterator[Effect]:
    """作品の中のエフェクトすべて トラック・クリップ・後の場面・縁取りの層に掛けた物まで"""
    for timeline in _timelines(project):
        for track in timeline.tracks:
            yield from track.effects
            for clip in track.clips:
                yield from clip.effects
                yield from clip.after_effects
                if clip.source is not None:
                    for stroke in clip.source.strokes:
                        yield from stroke.effects


def _sources(project: Project) -> Iterator[GeneratedSource]:
    for _timeline, clip in _clips(project):
        if clip.source is not None:
            yield clip.source


def _specs_of(kind: str, *, source: bool) -> tuple[object, ...] | None:
    definition = source_registry.get(kind) if source else registry.get(kind)
    return definition.parameters if definition is not None else None


def _file_names(specs: Iterable[object] | None) -> set[str]:
    """ファイルを指す欄（フォルダを指す欄は除く）の名前 定義が分からなければ空

    読み込んでいないスクリプトの欄はどれがファイルか分からない 推して文字を場所と読むと、
    文に見えて実は場所だった値、またはその逆を取り違えるので、触らない（一覧で知らせる）
    """
    if specs is None:
        return set()
    return {spec.name for spec in specs if isinstance(spec, FileSpec) and not spec.directory}


def media_spans(project: Project, margin: Fraction) -> dict[MediaId, list[Span] | None]:
    """素材ごとの使う区間（秒 素材の原点から） 前後に ``margin`` 秒の余裕を足す

    静止画はどこを使っても同じなので ``None``（まるごと） 使っていない素材は入らない
    前の音を読むエフェクト（リバーブ・ディレイ）を掛けたクリップは、頭より前もその長さだけ入れる
    入れないと別の PC で響きの頭が欠ける 絵を止める時刻（``hold_at``）より後ろは読まない
    """
    by_id = {item.id: item for item in project.media}
    found: dict[MediaId, list[Span] | None] = {}
    for timeline, clip in _clips(project):
        if clip.media_id is None or clip.media_id not in by_id:
            continue
        item = by_id[clip.media_id]
        if _is_still(item):
            found[item.id] = None
            continue
        rate = timeline.rate
        start = clip.source_in
        end = clip.source_out(rate)
        if clip.hold_at is not None:
            # 止めるのは絵だけで、音は source_in から長さの分を読み続ける（AudioMixer）
            # 絵の読む所（止める時刻まで）と音の読む所の和を入れる 終わりを止める時刻へ縮めると、
            # 音を鳴らすクリップの止めた後の音が持ち出した先で欠ける（#449 の Codex の指摘）
            start = min(start, clip.hold_at)
        sample_rate = project.settings.sample_rate
        lookback = Fraction(history_seconds(clip, rate, sample_rate)).limit_denominator(sample_rate)
        start -= lookback * clip.speed
        start = max(Fraction(0), start - margin)
        end = end + margin
        if item.duration > 0:
            end = min(end, item.duration)
        if end < start:
            end = start
        if item.id in found and found[item.id] is None:
            continue
        spans = found.setdefault(item.id, [])
        assert spans is not None
        spans.append(Span(start, end))
    return {
        key: (spans_union(spans) if spans is not None else None) for key, spans in found.items()
    }


def _is_still(item: MediaItem) -> bool:
    """どこを読んでも同じ絵の素材（静止画） 道を持たない物も、範囲を決められないので含める"""
    if Path(item.path).suffix.lower() in STILL_SUFFIXES:
        return True
    return not item.video_streams and not item.audio_streams


def file_paths(project: Project) -> tuple[str, ...]:
    """設定が指すファイル（書いてあるままの文字 重なりを除いて出てきた順）"""
    seen: dict[str, None] = {}

    def note(value: str) -> str:
        if value:
            seen.setdefault(value, None)
        return value

    map_paths(project, lambda item: item.path, note)
    return tuple(seen)


def fonts_used(project: Project) -> tuple[str, ...]:
    """テキストとエフェクトの書体の欄が指す書体の名前（重なりを除いて出てきた順）"""
    seen: dict[str, None] = {}
    for source in _sources(project):
        _note_fonts(source.params, _specs_of(source.kind, source=True), seen)
    for effect in _all_effects(project):
        _note_fonts(effect.params, _specs_of(effect.kind, source=False), seen)
    return tuple(seen)


def _note_fonts(
    params: dict[str, ParamValue], specs: tuple[object, ...] | None, seen: dict[str, None]
) -> None:
    if specs is None:
        return
    for spec in specs:
        if isinstance(spec, FontSpec):
            value = params.get(spec.name, spec.default)
            if isinstance(value, str) and value:
                seen.setdefault(value, None)


def scripts_used(project: Project) -> tuple[str, ...]:
    """使っている AviUtl のスクリプトのファイル（重なりを除いて出てきた順）

    種類の名前は ``aviutl:<置き場から数えた場所>:<名前>`` 同じファイルの中の別の効果は
    1 つに数える（入れる・入れないはファイルごとに決まるため）
    """
    seen: dict[str, None] = {}
    kinds = [effect.kind for effect in _all_effects(project)]
    kinds += [source.kind for source in _sources(project)]
    for kind in kinds:
        if kind.startswith(SCRIPT_PREFIX):
            place = kind[len(SCRIPT_PREFIX) :].rpartition(":")[0] or kind[len(SCRIPT_PREFIX) :]
            seen.setdefault(place, None)
    return tuple(seen)


def usage(project: Project, margin: Fraction, *, whole: bool = False) -> Usage:
    """作品が外に頼る物をまとめて数える ``whole`` なら素材はどれもまるごと"""
    spans = media_spans(project, margin)
    if whole:
        spans = dict.fromkeys(spans)
    return Usage(
        spans=spans,
        files=file_paths(project),
        fonts=fonts_used(project),
        scripts=scripts_used(project),
    )


def _map_params(
    params: dict[str, ParamValue], names: set[str], change: Callable[[str], str]
) -> dict[str, ParamValue]:
    if not names:
        return params
    changed = dict(params)
    for name in names:
        value = params.get(name)
        if isinstance(value, str) and value:
            changed[name] = change(value)
    return changed if changed != params else params


def _map_effect(effect: Effect, change: Callable[[str], str]) -> Effect:
    names = _file_names(_specs_of(effect.kind, source=False))
    params = _map_params(effect.params, names, change)
    return effect if params is effect.params else replace(effect, params=params)


def _map_stroke(stroke: Stroke, change: Callable[[str], str]) -> Stroke:
    params = _map_params(stroke.params, _file_names(STROKE.parameters), change)
    effects = tuple(_map_effect(effect, change) for effect in stroke.effects)
    return replace(stroke, params=params, effects=effects)


def _map_clip(clip: Clip, change: Callable[[str], str]) -> Clip:
    source = clip.source
    if source is not None:
        names = _file_names(_specs_of(source.kind, source=True))
        source = replace(
            source,
            params=_map_params(source.params, names, change),
            strokes=tuple(_map_stroke(stroke, change) for stroke in source.strokes),
        )
    return replace(
        clip,
        source=source,
        effects=tuple(_map_effect(effect, change) for effect in clip.effects),
        after_effects=tuple(_map_effect(effect, change) for effect in clip.after_effects),
    )


def _map_timeline(timeline: Timeline, change: Callable[[str], str]) -> Timeline:
    tracks = tuple(
        replace(
            track,
            effects=tuple(_map_effect(effect, change) for effect in track.effects),
            clips=tuple(_map_clip(clip, change) for clip in track.clips),
        )
        for track in timeline.tracks
    )
    return replace(timeline, tracks=tracks)


def map_paths(
    project: Project,
    media: Callable[[MediaItem], Path],
    files: Callable[[str], str],
) -> Project:
    """素材の場所と、設定が指すファイルの場所を付け替えた作品を返す

    パッケージへ入れるときは中の相対の場所へ、開くときは展開した先の場所へ付け替える
    設定の欄は定義の型（:class:`FileSpec`）で見分ける（名前で決めると、スクリプトの
    好きな名前のファイルの欄を取りこぼす）
    """
    return replace(
        project,
        media=tuple(replace(item, path=media(item)) for item in project.media),
        timeline=_map_timeline(project.timeline, files),
        scenes=tuple(
            replace(scene, timeline=_map_timeline(scene.timeline, files))
            for scene in project.scenes
        ),
    )
