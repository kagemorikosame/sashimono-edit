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
from sashimono.core.packed import read_window
from sashimono.effects import registry
from sashimono.effects.sources import source_registry
from sashimono.effects.spec import FileSpec, FontSpec
from sashimono.effects.strokes import STROKE
from sashimono.engine.audio.mixer import history_seconds
from sashimono.engine.decode.probe import STILL_SUFFIXES
from sashimono.engine.package.remux import (
    EDITING_CODECS,
    REENCODE_CRF,
    REENCODE_LEAST_PSNR,
    Span,
    reencodable,
    spans_union,
)

__all__ = [
    "EDITING_NAMES",
    "REENCODE_NOTE",
    "Usage",
    "editing_media",
    "file_fields",
    "file_paths",
    "folder_kinds",
    "fonts_used",
    "map_paths",
    "media_spans",
    "scripts_used",
    "unknown_kinds",
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
    """作品の時間軸すべて（本編と場面）"""
    yield project.timeline
    for scene in project.scenes:
        yield scene.timeline


def _clips(project: Project) -> Iterator[tuple[Timeline, Clip]]:
    """作品のクリップすべてと、その時間軸"""
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
    """作品の生成物（テキスト・図形など）すべて"""
    for _timeline, clip in _clips(project):
        if clip.source is not None:
            yield clip.source


def _specs_of(kind: str, *, source: bool) -> tuple[object, ...] | None:
    """種類の欄の定義 読み込んでいない種類なら ``None``"""
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
        # 止めるのは絵だけで、音は source_in から長さの分を読み続ける（AudioMixer） 絵の読む所と
        # 音の読む所の和を入れる（#449 の Codex の指摘） 数え方は開いた先で範囲の外を断る確かめ
        # （#445）と 1 つにする ずれると、入れていない所を読む編集が断られずに通る
        start, end = read_window(
            clip, timeline, history=history_seconds, sample_rate=project.settings.sample_rate
        )
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


#: 編集用の大きな形式の呼び名（窓と知らせに出す） 無い物は符号の名前のまま出す
EDITING_NAMES = {
    "prores": "ProRes",
    "dnxhd": "DNxHD・DNxHR",
    "rawvideo": "無圧縮",
    "v210": "無圧縮 10bit",
    "v410": "無圧縮 10bit 4:4:4",
    "cfhd": "CineForm",
    "huffyuv": "HuffYUV",
    "ffvhuff": "HuffYUV",
    "utvideo": "Ut Video",
    "magicyuv": "MagicYUV",
}


#: 符号にし直すと何が変わるか（窓と AI の確かめの箱に出す #446）
#: 大きさと差は、ffmpeg で作った 1080p の試験の素材（3 秒）で測った値（PR に記す）
REENCODE_NOTE = (
    f"映像だけを H.264（CRF {REENCODE_CRF}）にし直します 色の深さと色差の間引き（10bit 4:2:2 など）"
    "と時刻は元のまま、音は元のまま入れます\n"
    "画質は元と完全には同じになりません 元との差は PSNR 50〜55 dB ほど（目では見分けにくい程度）"
    f"で、{REENCODE_LEAST_PSNR:.0f} dB に届かない素材と、小さくならない素材は元のまま入れます\n"
    "大きさの目安は ProRes 422 HQ・DNxHR HQ で元の 1.5〜4 割、無圧縮で数 % です"
    "（絵の細かさで大きく変わります） 符号にし直す分、書き出しに時間が掛かります"
)


def editing_media(project: Project) -> tuple[str, ...]:
    """使っている素材のうち編集用の大きな形式の物（``名前（ProRes）`` の形 #446）

    素材の情報（取り込んだときに読んだ符号の名前）で見る 書き出しの窓を開くたびに
    素材を開き直すと、数十本の素材で窓が出るまで待たされる
    """
    used = media_spans(project, Fraction(0))
    found: list[str] = []
    for item in project.media:
        if item.id not in used:
            continue
        # し直せない形（透明を持つ ProRes 4444・RGB・12bit 以上）だけの素材は並べない 並べると、
        # 選んでも元のまま入って「小さくなる」と読んだ人の見込みが外れる
        codecs = [
            stream.codec
            for stream in item.video_streams
            if stream.codec in EDITING_CODECS and reencodable(stream.pixel_format)
        ]
        if len(codecs) == 1:
            found.append(f"{item.name}（{EDITING_NAMES.get(codecs[0], codecs[0])}）")
    return tuple(found)


def _is_still(item: MediaItem) -> bool:
    """どこを読んでも同じ絵の素材（静止画） 道を持たない物も、範囲を決められないので含める"""
    if Path(item.path).suffix.lower() in STILL_SUFFIXES:
        return True
    return not item.video_streams and not item.audio_streams


def file_paths(project: Project) -> tuple[str, ...]:
    """設定が指すファイル（書いてあるままの文字 重なりを除いて出てきた順）"""
    seen: dict[str, None] = {}

    def note(value: str) -> str:
        """見つけた物を覚え、値はそのまま返す"""
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
    """書体の欄の値を ``seen`` へ足す"""
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


#: 文字を付け替える関数
Change = Callable[[str], str]
#: 欄の在りか（:func:`file_fields` の札）と文字を受け取って付け替える関数
Keyed = Callable[[str, str], str]


def _map_params(
    params: dict[str, ParamValue],
    specs: tuple[object, ...] | None,
    owner: str,
    files: Keyed,
    unknown: Keyed | None,
) -> dict[str, ParamValue]:
    """欄の値を付け替える ファイルの欄は ``files`` で、定義の分からない種類の欄は ``unknown`` で

    定義の分からない種類（読み込んでいないスクリプト）はどの欄がファイルか分からないので、
    ``unknown`` を渡したときだけ文字の欄を全部渡す（何を変えるかは関数が決める） 関数には
    欄の在りか（``owner/欄の名前``）も渡す 書き出したときに記したファイルの欄だけを開く側で
    付け替えられるようにするため
    """
    if specs is None:
        if unknown is None:
            return params
        names, change = {name for name, value in params.items() if isinstance(value, str)}, unknown
    else:
        names, change = _file_names(specs), files
    if not names:
        return params
    changed = dict(params)
    for name in names:
        value = params.get(name)
        if isinstance(value, str) and value:
            changed[name] = change(f"{owner}/{name}", value)
    return changed if changed != params else params


def _map_effect(effect: Effect, files: Keyed, unknown: Keyed | None) -> Effect:
    """エフェクトの欄を付け替える"""
    params = _map_params(
        effect.params,
        _specs_of(effect.kind, source=False),
        f"effect:{effect.kind}:{effect.id}",
        files,
        unknown,
    )
    return effect if params is effect.params else replace(effect, params=params)


def _map_stroke(stroke: Stroke, files: Keyed, unknown: Keyed | None) -> Stroke:
    """縁取りの層と、そこに掛けたエフェクトの欄を付け替える"""
    params = _map_params(
        stroke.params, STROKE.parameters, f"stroke:{STROKE.kind}:{stroke.id}", files, unknown
    )
    effects = tuple(_map_effect(effect, files, unknown) for effect in stroke.effects)
    return replace(stroke, params=params, effects=effects)


def _map_clip(clip: Clip, files: Keyed, unknown: Keyed | None) -> Clip:
    """クリップの生成物とエフェクトの欄を付け替える"""
    source = clip.source
    if source is not None:
        source = replace(
            source,
            params=_map_params(
                source.params,
                _specs_of(source.kind, source=True),
                f"source:{source.kind}:{clip.id}",
                files,
                unknown,
            ),
            strokes=tuple(_map_stroke(stroke, files, unknown) for stroke in source.strokes),
        )
    return replace(
        clip,
        source=source,
        effects=tuple(_map_effect(effect, files, unknown) for effect in clip.effects),
        after_effects=tuple(_map_effect(effect, files, unknown) for effect in clip.after_effects),
    )


def _map_timeline(timeline: Timeline, files: Keyed, unknown: Keyed | None) -> Timeline:
    """時間軸のトラックとクリップの欄を付け替える"""
    tracks = tuple(
        replace(
            track,
            effects=tuple(_map_effect(effect, files, unknown) for effect in track.effects),
            clips=tuple(_map_clip(clip, files, unknown) for clip in track.clips),
        )
        for track in timeline.tracks
    )
    return replace(timeline, tracks=tracks)


def map_paths(
    project: Project,
    media: Callable[[MediaItem], Path],
    files: Change,
    *,
    unknown: Keyed | None = None,
) -> Project:
    """素材の場所と、設定が指すファイルの場所を付け替えた作品を返す

    パッケージへ入れるときは中の相対の場所へ、開くときは展開した先の場所へ付け替える
    設定の欄は定義の型（:class:`FileSpec`）で見分ける（名前で決めると、スクリプトの
    好きな名前のファイルの欄を取りこぼす） 定義の分からない種類の欄は ``unknown`` を渡した
    ときだけ、欄の在りかと文字を渡して付け替える（開く PC に無いスクリプトの欄のため）
    """

    def keyed(_where: str, value: str) -> str:
        """欄の在りかを使わない呼び手の関数へつなぐ"""
        return files(value)

    return replace(
        project,
        media=tuple(replace(item, path=media(item)) for item in project.media),
        timeline=_map_timeline(project.timeline, keyed, unknown),
        scenes=tuple(
            replace(scene, timeline=_map_timeline(scene.timeline, keyed, unknown))
            for scene in project.scenes
        ),
    )


def file_fields(project: Project) -> tuple[str, ...]:
    """ファイルを指す欄の在りか（``effect:<種類>:<id>/欄`` など 値の入っている物だけ）

    種類も札に入れる 別の作品から貼った物などで id が重なっても、種類が違えば別の欄として扱う

    パッケージの目録に記す 開く PC にスクリプトが無いと、どの欄がファイルかを定義から
    読めない 記しておけば、その欄だけを展開した先へ付け替え、ほかの文字（``/字幕`` のような
    場所に見える文）は書き換えない（#449 の CodeRabbit の指摘）
    """
    seen: dict[str, None] = {}

    def note(where: str, value: str) -> str:
        """見つけた物を覚え、値はそのまま返す"""
        seen.setdefault(where, None)
        return value

    replace(
        project,
        timeline=_map_timeline(project.timeline, note, None),
        scenes=tuple(
            replace(scene, timeline=_map_timeline(scene.timeline, note, None))
            for scene in project.scenes
        ),
    )
    return tuple(seen)


def folder_kinds(project: Project) -> tuple[str, ...]:
    """フォルダを指す欄に値の入ったエフェクトと生成物の種類 重なりを除いて出てきた順

    フォルダはまだパッケージに入れられない 欄の値（元の PC の場所 ユーザー名が入る）が
    そのまま人へ渡るので、書き出しを断る（#449 の CodeRabbit の指摘）
    """
    seen: dict[str, None] = {}

    def check(kind: str, params: dict[str, ParamValue], specs: tuple[object, ...] | None) -> None:
        """フォルダの欄に値があれば、その種類を覚える"""
        for spec in specs or ():
            if isinstance(spec, FileSpec) and spec.directory:
                value = params.get(spec.name)
                if isinstance(value, str) and value:
                    seen.setdefault(kind, None)

    for effect in _all_effects(project):
        check(effect.kind, effect.params, _specs_of(effect.kind, source=False))
    for source in _sources(project):
        check(source.kind, source.params, _specs_of(source.kind, source=True))
        for stroke in source.strokes:
            check("stroke", stroke.params, STROKE.parameters)
    return tuple(seen)


def unknown_kinds(project: Project) -> tuple[str, ...]:
    """定義の分からない（読み込んでいない）エフェクトと生成物の種類 重なりを除いて出てきた順

    どの欄がファイルを指すか分からないので、パッケージにはできない 欄に元の PC の場所が
    残ったまま人へ渡り、指すファイルも入らない
    """
    seen: dict[str, None] = {}
    for effect in _all_effects(project):
        if registry.get(effect.kind) is None:
            seen.setdefault(effect.kind, None)
    for source in _sources(project):
        if source_registry.get(source.kind) is None:
            seen.setdefault(source.kind, None)
    return tuple(seen)
