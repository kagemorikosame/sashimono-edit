"""YMM4 のアイテムテンプレート（``.ymmt``）を、こちらのクリップへ写す

**``.ymmt`` は ZIP** 中に ``catalog.json`` が 1 つ入っていて、その中身がこの形

.. code-block:: json

    {"FilePath": "C:\\…\\なにか.ymmt",
     "ItemTemplates": [{"Name": "アニメーション効果/振り子",
                        "Path": ["アニメーション効果", "振り子"],
                        "Items": [ … ]}],
     "VideoEffectTemplates": [],
     "AudioEffectTemplates": []}

**1 ファイルに何本も入っている** 手元で確かめた配布物は 17 本と 106 本だった
だから :func:`load_template` はテンプレートの**列**を返し、棚
（:mod:`sashimono.compat.catalog`）はそれを 1 本ずつ並べる

アイテムの種類は ``$type`` に入る 振り分けは**クラス名だけ**で行う 実物には
``Version=4.32.0.2, Culture=neutral, PublicKeyToken=null`` まで書かれていて、
丸ごと突き合わせると YMM4 が更新されただけで読めなくなる

知らない種類が来たら、その名前を :mod:`~sashimono.compat.aviutl.report` に残して
先へ進む 1 種類読めないだけでテンプレート全体が落ちるのは割に合わない
"""

from __future__ import annotations

import importlib
import json
import lzma
import math
import zipfile
import zlib
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from sashimono.compat.aviutl.report import CompatibilityReport, global_report
from sashimono.compat.mapped import MappedObject, fitted_effect
from sashimono.compat.ymm4.brushes import BLEND_NAMES, brush_effect, is_solid
from sashimono.compat.ymm4.decorations import (
    has_outline,
    map_decorations,
    map_video_effects,
    outlines_fit_text,
    with_pivot,
)
from sashimono.compat.ymm4.effects import CenterPoint
from sashimono.compat.ymm4.values import (
    animated,
    brush_colour,
    colour,
    number,
    reporting,
    timespan,
    type_name,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    ParamValue,
    Stroke,
    legacy_in_use,
)
from sashimono.effects.definition import registry
from sashimono.effects.sources import source_registry

__all__ = [
    "CATALOG_NAME",
    "TEMPLATE_SUFFIXES",
    "ItemTemplate",
    "Ymm4ParseError",
    "load_template",
    "map_template",
]

#: アイテムテンプレートの拡張子
TEMPLATE_SUFFIXES = (".ymmt",)

#: ZIP の中に入っているファイルの名前
CATALOG_NAME = "catalog.json"

#: YMM4 の合成モードと、こちらの呼び名
#: こちらに無いものは通常扱いにして記録に残す
_BLEND_MODES: dict[str, str] = {
    "Normal": "normal",
    "通常": "normal",
    "Add": "add",
    "加算": "add",
    "Multiply": "multiply",
    "乗算": "multiply",
    "Screen": "screen",
    "スクリーン": "screen",
    "Overlay": "overlay",
    "オーバーレイ": "overlay",
    # YMM4 の名前は Lighter と Darker（Lighten と Darken は YMM4 が読み込みで断る）
    "Lighter": "lighten",
    "比較(明)": "lighten",
    "Darker": "darken",
    "比較(暗)": "darken",
    "Subtract": "subtract",
    "減算": "subtract",
    # 残りの名前は塗りのエフェクトと同じ表で読む
    **BLEND_NAMES,
}

#: ``BasePoint`` の横と縦 ``CenterCenter`` ``LeftTop`` のように 2 つ並ぶ
_HORIZONTAL = (("Left", "left"), ("Right", "right"), ("Center", "center"))
_VERTICAL = (("Top", "top"), ("Bottom", "bottom"), ("Center", "middle"))

#: 図形の種類
_SHAPES: dict[str, str] = {
    "Rectangle": "rect",
    "Square": "rect",
    "RoundedRectangle": "rounded",
    "Ellipse": "ellipse",
    "Circle": "ellipse",
    # YMM4 の三角形は円に内接する（試験の絵で確かめた）
    "Triangle": "inscribed_triangle",
    "Star": "star",
    "Background": "background",
    "Hexagon": "hexagon",
    "Fan": "fan",
    "Arrow": "arrow",
    "Superformula": "superformula",
}

#: 素材を参照するアイテム 中身ではなくパスだけを返す
_MEDIA_ITEMS: dict[str, str] = {
    "VideoItem": "動画ファイル",
    "ImageItem": "画像ファイル",
    "AudioItem": "音声ファイル",
    "VoiceItem": "音声ファイル",
}

#: 中身を持たないアイテム 読めないのではなく、それ自体は絵を持たない
#:
#: ``GroupItem`` はまとめた相手に掛かるエフェクトを持つ入れ物 AviUtl の
#: 「フィルタオブジェクト」に近い :func:`map_template` が中身へ移す
_CONTAINER_ITEMS = frozenset({"GroupItem"})


class Ymm4ParseError(ValueError):
    """``.ymmt`` として読めない"""


@dataclass(frozen=True, slots=True)
class ItemTemplate:
    """カタログに入っているテンプレート 1 本"""

    name: str
    #: ``["アニメーション効果", "振り子"]`` のような分類
    path: tuple[str, ...] = ()
    items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    #: ファイルの中のどこにあったか（``ItemTemplates の 2 本目`` など） 空なら
    #: ファイル全体が 1 本（古い形やアイテムだけを書き出したもの）
    #:
    #: 並ぶ順の番号では原本と照らし合わせられない アイテムの入っていない物を
    #: 除いた後の番号で、アイテムとエフェクトの 2 つの一覧を続けて数えているため
    origin: str = ""

    @property
    def folder(self) -> str:
        """分類の先頭 棚の見出しに使う"""
        return self.path[0] if len(self.path) > 1 else ""


def _zstd_errors() -> tuple[type[Exception], ...]:
    """Zstandard の展開の失敗 Python 3.14 から ZIP の中身に使える

    3.14 より前には無いので名前で引く 3.12 や 3.13 では、Zstandard の中身は
    ``zipfile`` が「対応していない圧縮方式」（``NotImplementedError``）として断る
    """
    try:
        module = importlib.import_module("compression.zstd")
    except ImportError:
        return ()
    error = getattr(module, "ZstdError", None)
    return (error,) if isinstance(error, type) and issubclass(error, Exception) else ()


#: ZIP の中身を取り出すときに ``zipfile`` が投げうる物のうち、``OSError`` でない物
#: どれもファイルの側の事情で、ここで「読めない」に変えないと棚の走査ごと落ち、
#: ほかのテンプレートまで並ばない（``zipfile`` の実装を読んで拾った）
#:
#: * ``BadZipFile`` — 目録や見出しが壊れている・CRC が合わない
#: * ``RuntimeError`` — 暗号化されていてパスワードが要る・展開に要るモジュールが無い
#: * ``NotImplementedError`` — 対応していない圧縮方式・強い暗号化・ZIP の版が新しすぎる
#: * ``EOFError`` — 中身が途中で切れている
#: * ``zlib.error`` ``lzma.LZMAError`` と Zstandard の失敗 — 圧縮された中身が壊れている
#:   （bz2 の失敗は ``OSError`` で来る）
#:
#: ``NotImplementedError`` は ``RuntimeError`` の仲間だが、分けて書いておく
#: 暗号化と未知の圧縮方式のどちらを受けるつもりかが、並びから読めるように
_ARCHIVE_ERRORS: tuple[type[Exception], ...] = (
    zipfile.BadZipFile,
    RuntimeError,
    NotImplementedError,
    EOFError,
    zlib.error,
    lzma.LZMAError,
    *_zstd_errors(),
)


def load_template(path: Path) -> list[ItemTemplate]:
    """ファイルを読んで、入っているテンプレートの列を返す

    ファイルの側の事情で読めない物は、どれも :class:`Ymm4ParseError` にして返す
    棚はこれだけを受けて「読めません」の項目にする 棚の側で何でも受けると、
    こちらの誤り（型の取り違えなど）まで「ファイルが壊れている」に見えて気付けない
    """
    target = Path(path)
    try:
        raw = _read_catalog(target)
    except OSError as exc:
        raise Ymm4ParseError(f"開けない: {target} ({exc})") from exc
    except _ARCHIVE_ERRORS as exc:
        raise Ymm4ParseError(f"{target.name}: ZIP として読めない ({exc})") from exc
    # UTF-8 でない物は ``ValueError`` の仲間で来る OSError ではないので、受けないと
    # 棚の走査ごと落ちる
    except UnicodeDecodeError as exc:
        raise Ymm4ParseError(f"{target.name}: UTF-8 の文字として読めない ({exc})") from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Ymm4ParseError(f"{target.name}: JSON として読めない ({exc})") from exc
    return _templates_of(document, target)


def _read_catalog(path: Path) -> str:
    """``.ymmt`` の中身を取り出す

    ZIP なら ``catalog.json`` を、そうでなければファイルそのものを読む
    BOM が付くことがあるので ``utf-8-sig`` で開く
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".json")]
            chosen = CATALOG_NAME if CATALOG_NAME in names else (names[0] if names else "")
            if not chosen:
                raise Ymm4ParseError(f"{path.name}: {CATALOG_NAME} が入っていない")
            return archive.read(chosen).decode("utf-8-sig")
    return path.read_text(encoding="utf-8-sig")


def _templates_of(document: Any, path: Path) -> list[ItemTemplate]:
    """包み方の違いを吸収して、テンプレートの列を取り出す"""
    if isinstance(document, dict):
        catalogued = document.get("ItemTemplates")
        effect_templates = document.get("VideoEffectTemplates")
        if isinstance(catalogued, list) or isinstance(effect_templates, list):
            # 位置は除く前の原本の並びで数える 報告を受けた側が原本で同じ物を探せるように
            built = [
                replace(_template_of(entry), origin=f"ItemTemplates の {number} 本目")
                for number, entry in enumerate(
                    catalogued if isinstance(catalogued, list) else [], start=1
                )
                if isinstance(entry, dict)
            ]
            built.extend(
                replace(_effect_template_of(entry), origin=f"VideoEffectTemplates の {number} 本目")
                for number, entry in enumerate(
                    effect_templates if isinstance(effect_templates, list) else [], start=1
                )
                if isinstance(entry, dict)
            )
            found = [item for item in built if item.items]
            if found:
                return found
            raise Ymm4ParseError(f"{path.name}: アイテムの入ったテンプレートが無い")

    # 古い形、あるいはアイテムだけを書き出したもの
    items = _bare_items(document)
    if items:
        return [ItemTemplate(name=path.stem, items=tuple(items))]
    raise Ymm4ParseError(f"{path.name}: アイテムが見つからない")


def _template_of(entry: dict[str, Any]) -> ItemTemplate:
    raw_path = entry.get("Path")
    parts = tuple(str(part) for part in raw_path) if isinstance(raw_path, list) else ()
    name = str(entry.get("Name") or (parts[-1] if parts else ""))
    items = entry.get("Items")
    return ItemTemplate(
        name=name,
        path=parts,
        items=tuple(item for item in items if isinstance(item, dict))
        if isinstance(items, list)
        else (),
    )


#: 映像エフェクトのテンプレートを包むアイテム エフェクトだけを持つグループと同じ扱いになる
_EFFECT_HOLDER = "YukkuriMovieMaker.Project.Items.GroupItem, YukkuriMovieMaker"


def _effect_template_of(entry: dict[str, Any]) -> ItemTemplate:
    """映像エフェクトのテンプレート（``VideoEffectTemplates``）を 1 本読む

    アイテムを持たず、エフェクトの並びだけが入っている 置くものではなく、既にある
    クリップへ着せて使う エフェクトだけを持つグループとして包めば、アイテムの
    テンプレートと同じ道（:func:`map_template`）で読める 包まずに飛ばすと、
    あおもや式のエフェクト集 15 本が棚に並ばず、読めないことにも気付けなかった
    """
    name = str(entry.get("Name") or "")
    effects = entry.get("Effects")
    if not isinstance(effects, list) or not effects:
        return ItemTemplate(name=name)
    holder = {"$type": _EFFECT_HOLDER, "VideoEffects": effects, "Length": 1}
    return ItemTemplate(name=name, path=("映像エフェクト", name), items=(holder,))


def _bare_items(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)]
    if not isinstance(document, dict):
        return []
    for key in ("Items", "TimelineItems", "Contents"):
        found = document.get(key)
        if isinstance(found, list):
            return [item for item in found if isinstance(item, dict)]
    single = document.get("Item")
    if isinstance(single, dict):
        return [single]
    return [document] if "$type" in document else []


def map_template(
    items: list[dict[str, Any]], *, report: CompatibilityReport | None = None
) -> list[MappedObject]:
    """アイテムの列をクリップへ 写せなかったものは飛ばす

    ``GroupItem`` は中身を持たない入れ物で、まとめた相手に掛かるエフェクトを
    持っている 掛かる相手は、グループのレイヤーのすぐ上から ``GroupRange`` 段
    まで 掛け方は「合成する」（``IsComposite``）で 2 通りに分かれる

    * 合成しない — 範囲の中身 1 つずつにエフェクトを掛ける こちらのモデルに
      入れ子は無いので、**中身へエフェクトを移して**平らにする
    * 合成する — 範囲の中身を 1 枚の絵に重ねてから、その絵にエフェクト・反転・
      拡大・不透明度・合成モード・クリッピングを掛ける こちらでは中身を
      シーンにまとめ、グループをそのシーンのクリップとして置く
      （:attr:`MappedObject.children`）

    中身が無いテンプレート（``アニメーション効果/振り子`` のようなもの）は
    **エフェクトだけ**の結果になり、既にあるクリップへ着せて使う
    """
    log = report if report is not None else global_report
    # 値を読む関数の多くは記録を受け取らない 知らない移動方法の形がこの読み込みの
    # 記録に残るよう、入口で置いておく
    with reporting(log):
        return _map_items(list(items), log)


def _map_items(items: list[dict[str, Any]], log: CompatibilityReport) -> list[MappedObject]:
    composites, consumed = _composite_groups(items, log)

    contents: list[tuple[int, MappedObject]] = []
    grouped: list[Effect] = []
    # 入れ物ごとの (レイヤー, 範囲, 長さ, エフェクト) 動く値のキーフレームはその
    # 入れ物の長さの上に並んでいるので、中身の無いテンプレートでも長さを残す
    # 1 にすると、着せるときに尺を合わせられず、動きが着せた先の途中で止まる
    containers: list[_Container] = []
    for item in items:
        if id(item) in consumed:
            continue
        scene = composites.get(id(item))
        if scene is not None:
            contents.append((_layer_of(item), scene))
            continue
        if type_name(item) in _CONTAINER_ITEMS:
            effects = _group_effects(item, log)
            grouped.extend(effects)
            containers.append(
                _Container(
                    layer=_layer_of(item),
                    reach=_reach_of(item),
                    length=max(1, int(number(item.get("Length"), 1.0))),
                    effects=effects,
                )
            )
            continue
        mapped = _map_item(item, log)
        if mapped is not None:
            contents.append((_layer_of(item), mapped))

    if not grouped:
        return [mapped for _, mapped in contents]
    if not contents:
        # 中身のないテンプレート エフェクトだけを**入れ物ごとに**返す
        # 1 つにまとめると、長さの違う入れ物が混ざったときに短い方の動きが
        # 長い方の尺で伸び縮みする（着せる側は 1 つずつ尺を合わせる）
        return [
            MappedObject(
                clip=Clip(
                    timeline_start=0,
                    duration=container.length,
                    effects=tuple(container.effects),
                ),
                layer=1,
                kind="effects",
                has_span=False,
            )
            for container in containers
            if container.effects
        ]
    # 入れ物のエフェクトを中身へ移す 入れ物と中身で長さが違うことがあるので
    # （手元の配布物 97 本のうち 6 本 例: 入れ物 18 中身 300）、動く値の時刻を
    # 中身の長さへ揃えてから移す 揃えずに移すと、入れ物の終わりに置いた点が
    # 中身の途中に残り、エフェクトの終わりの見た目が出ないまま止まる
    #
    # 移すのは中身を範囲に含む入れ物のものだけ 全部へ配ると、リボンのテロップの
    # 文字（範囲の外）に吹き出しの登場の動きが 2 回掛かり、倍の距離を飛んでくる
    return [
        _with_group_effects(
            item,
            [container for container in containers if container.covers(layer)],
            span=max((container.length for container in containers), default=1),
        )
        for layer, item in contents
    ]


@dataclass(frozen=True, slots=True)
class _Container:
    """合成しないグループ 範囲の中身へエフェクトを配る"""

    layer: int
    #: 掛かる段の数 ``None`` は上の段すべて
    reach: int | None
    length: int
    effects: list[Effect]

    def covers(self, layer: int) -> bool:
        return layer > self.layer and (self.reach is None or layer <= self.layer + self.reach)


def _layer_of(item: dict[str, Any]) -> int:
    return int(number(item.get("Layer"), 0.0))


def _reach_of(item: dict[str, Any]) -> int | None:
    """グループが掛かる段の数

    ``GroupRange`` を持たないグループは、映像エフェクトのテンプレートを包んだ入れ物
    （:func:`_effect_template_of`）と、それだけを書き出した配布物（あおもや式の
    アニメーション効果 18 本）だけだった どれもレイヤーも持たず、着せる先の段が
    決まっていないので、上の段すべてに掛かるものとして読む 1 段とすると、
    比べる道具が下地を置いた段によっては何にも掛からなくなる
    """
    if "GroupRange" not in item:
        return None
    return max(1, int(number(item.get("GroupRange"), 1.0)))


def _composite_groups(
    items: list[dict[str, Any]], log: CompatibilityReport
) -> tuple[dict[int, MappedObject], set[int]]:
    """合成するグループと、その範囲の中身を 1 つのまとめた絵にする

    返すのは「グループの ``id`` → まとめた絵」と、まとめた絵の中へ入った中身の ``id``
    入れ子のグループは、外側（レイヤー番号の小さい方 YMM4 の画面では上の段）から
    順にまとめる 内側のグループは中身と一緒に外側の絵の中へ入り、そこでもう一度
    この関数を通る

    中身に数えるのは、範囲の段にあって**時間もグループと重なる**ものだけ
    グループが終わった後に始まるアイテムまでまとめると、シーンのクリップの長さで
    切られて消える

    範囲に中身が 1 つも無い合成するグループは、ふつうの入れ物として残す
    （ペイントトランジションの「この範囲内に次の場面を置いてください」という空の枠）
    空のシーンを置いても何も映らず、トラックとシーンが増えるだけになる
    """
    made: dict[int, MappedObject] = {}
    consumed: set[int] = set()
    for group in sorted(
        (item for item in items if _is_composite(item)),
        key=_layer_of,
    ):
        if id(group) in consumed:
            continue
        reach = _reach_of(group)
        low = _layer_of(group)
        members = [
            item
            for item in items
            if item is not group
            and id(item) not in consumed
            and _layer_of(item) > low
            and (reach is None or _layer_of(item) <= low + reach)
            and _overlaps(item, group)
        ]
        _note_crossing(members, low, reach, log)
        scene = _composite(group, members, log)
        if scene is None:
            continue
        made[id(group)] = scene
        consumed.update(id(item) for item in members)
    return made, consumed


def _span_of(item: dict[str, Any]) -> tuple[int, int]:
    start = int(number(item.get("Frame"), 0.0))
    return start, start + max(1, int(number(item.get("Length"), 1.0)))


def _overlaps(item: dict[str, Any], group: dict[str, Any]) -> bool:
    start, end = _span_of(item)
    group_start, group_end = _span_of(group)
    return start < group_end and group_start < end


def _note_crossing(
    members: list[dict[str, Any]], low: int, reach: int | None, log: CompatibilityReport
) -> None:
    """合成するグループの中のグループが、外側の範囲を越えて掛かっていれば記録する

    内側のグループはまとめた絵の中でしか働かないので、外側の範囲の外の段には何も
    掛けられない YMM4 がこの形をどう描くかは確かめていない（配布物 230 本に合成する
    グループを越える形は無かった） 黙って捨てず、互換性レポートに残す
    """
    if reach is None:
        return
    for item in members:
        if type_name(item) not in _CONTAINER_ITEMS:
            continue
        # ``GroupRange`` を持たない内側のグループは上の段すべてに掛かる扱いなので、
        # 範囲のある外側の中では必ず越えている
        inner = _reach_of(item)
        if inner is None or _layer_of(item) + inner > low + reach:
            log.note_missing("YMM4 の合成するグループの範囲を越える内側のグループ")


def _is_composite(item: dict[str, Any]) -> bool:
    return type_name(item) in _CONTAINER_ITEMS and item.get("IsComposite") is True


def _composite(
    group: dict[str, Any], members: list[dict[str, Any]], log: CompatibilityReport
) -> MappedObject | None:
    """合成するグループ 1 つを、中身をまとめたシーンのクリップへ

    中身のレイヤーはグループからの相対へ直す シーンのトラックは 1 から始まるので、
    グループのすぐ上の段が 1 本目になる

    時刻は、グループと中身のうち一番早く始まるものからの相対にする グループより
    先に始まった中身は、シーンの中でもその分だけ先に始まり、グループのクリップは
    シーンの途中（:attr:`MappedObject.scene_offset`）から映す グループの頭へ
    詰めると、グループが始まった時点で進んでいるはずの動きが頭から描き直される

    グループ自身の反転・拡大・回転・エフェクトは、まとめた絵（画面の大きさ）に
    掛かるので、画面の中心を軸に効く YMM4 の書き出しと比べて確かめた
    （吹き出し風ワイプの縁取りと影と登場の動きを 1 枚に掛けると、真ん中の
    フレームの差が 2.6 から 0.7 へ、リボンのテロップの入りが 8.4 から 4.8 へ下がる）
    """
    center = str(group.get("CompositeCenter") or "ScreenCenter")
    if center != "ScreenCenter":
        log.note_missing(f"YMM4 のグループの合成の中心: {center}")
    start = int(number(group.get("Frame"), 0.0))
    earliest = min((_span_of(item)[0] for item in members), default=start)
    origin = min(start, earliest)
    base = _layer_of(group) + 1
    rebased = [
        {
            **item,
            "Layer": _layer_of(item) - base,
            "Frame": _span_of(item)[0] - origin,
        }
        for item in members
    ]
    children = tuple(_map_items(rebased, log))
    if not any(child.has_picture for child in children):
        return None

    length = max(1, int(number(group.get("Length"), 1.0)))
    keyframes = group.get("KeyFrames")
    remark = str(group.get("Remark") or "").strip().splitlines()
    return MappedObject(
        clip=Clip(
            timeline_start=max(0, start),
            duration=length,
            effects=tuple(_group_effects(group, log)),
            opacity=animated(
                group.get("Opacity"), 100.0, length=length, keyframes=keyframes, scale=0.01
            ),
            blend_mode=_blend_of(group, log),
            clip_to_below=group.get("IsClippingWithObjectAbove") is True,
        ),
        layer=max(1, base),
        kind="scene",
        children=children,
        label=remark[0] if remark else "合成したグループ",
        scene_offset=start - origin,
    )


def _with_group_effects(
    item: MappedObject, containers: list[_Container], *, span: int
) -> MappedObject:
    """入れ物のエフェクトを 1 つの中身へ移す 動く値の時刻は中身の長さへ揃える

    揃える先は中身の長さちょうど YMM4 は最後の点を長さの位置に置く

    中身が長さを持っていないことがある（``レトロなカウントダウン3秒`` は
    入れ物 90 に対して中身 1） そのまま 1 に揃えると 90 フレームの動きが
    2 フレームに潰れ、置くときに既定の長さまで伸ばされても動きは戻らない
    長さが分かるのは入れ物の側だけなので、そちらを中身の長さとして使う
    ``span`` は、範囲に入っていない入れ物も含めた一番長い入れ物の長さ 中身の長さを
    借りるのは範囲と関係なく、テンプレート全体の尺を決めるため
    """
    known = item.has_span or span <= 1
    duration = item.clip.duration if known else span
    # 入れ物の描画の欄は、中身へ移すとふつうのエフェクトになる 中身は自分の欄を持っているので、
    # 印を残すと同じ種類の欄が 2 つ並び、どちらがパネルの欄か決まらない
    moved = [
        replace(fitted_effect(effect, container.length, duration), fixed=False)
        for container in containers
        for effect in container.effects
    ]
    return replace(
        item,
        clip=replace(item.clip, duration=duration, effects=(*item.clip.effects, *moved)),
        # 入れ物から長さを借りたなら、長さの分かるものとして扱う
        has_span=item.has_span or not known,
    )


def _group_effects(item: dict[str, Any], log: CompatibilityReport) -> list[Effect]:
    """``GroupItem`` が持っているエフェクト 位置の動きも含む

    グループに付いた縁取りは、文字そのものの飾りではなく**まとめた絵の外側**に
    掛かる 映像エフェクトの読み方（:func:`map_video_effects`）がテキスト以外の縁取りを
    並びの位置のまま縁取りエフェクトにするので、ここでは並べるだけでよい
    """
    length = max(1, int(number(item.get("Length"), 1.0)))
    keyframes = item.get("KeyFrames")
    _, chain, final, _ = _video_chain(item, log, length, keyframes)
    return [*chain, *final]


#: 描画を遅らせる印（DrawLazyEffect）の設定と、その場で当てる配置の項目
_LAZY_PARTS = (
    ("IsXYZ", ("X", "Y")),
    ("IsZoom", ("Zoom",)),
    ("IsRotation", ("Rotation",)),
)
_RESTING = {"X": 0.0, "Y": 0.0, "Zoom": 100.0, "Rotation": 0.0}


def _video_chain(
    item: dict[str, Any],
    log: CompatibilityReport,
    length: int,
    keyframes: Any,
    *,
    text: bool = False,
) -> tuple[dict[str, ParamValue], list[Effect], list[Effect], list[Stroke]]:
    """映像エフェクトの並びと、最後に当てる配置（反転と位置・拡大・回転）

    YMM4 はエフェクトを掛けた絵を最後に置く ただし描画を遅らせる印があれば、印の場所で
    印が指す分（位置・拡大・回転）だけを先に当て、残りを最後に当てる（試験で確かめた）
    ``text`` が真なら、縁取りをテキストの設定（1 つ目の戻り値）か、2 つ以上なら縁取りの層
    （4 つ目の戻り値 #272）へ分ける
    """
    raw = item.get("VideoEffects")
    entries = raw if isinstance(raw, list) else []
    lazy = next(
        (
            index
            for index, entry in enumerate(entries)
            if isinstance(entry, dict)
            and entry.get("IsEnabled") is not False
            and type_name(entry) == "DrawLazyEffectEffect"
        ),
        None,
    )
    flip = _flip(item)
    if lazy is None:
        video = map_video_effects(entries, log, length=length, keyframes=keyframes, text=text)
        final = _fixed([*flip, *_placement(item, length, keyframes, video.pivot)])
        return video.params, list(video.effects), final, list(video.strokes)

    marker = entries[lazy]
    # テキストの設定へ縁取りを載せるかは、分ける前の列全体で決める 区間ごとに決めると、
    # 載せられない縁取り（縁だけ・ぼかしなど）が印の反対側にあるとき、こちら側だけが
    # テキストへ移って並びの頭へ動く 両側に縁取りがあるときも載せない 区間ごとに一番太い
    # ものを選ぶので、合わせるときに前の区間の分が落ちる
    head, tail = entries[:lazy], entries[lazy + 1 :]
    text = (
        text
        and outlines_fit_text(entries, length=length, keyframes=keyframes)
        and not (has_outline(head) and has_outline(tail))
    )
    first = map_video_effects(head, log, length=length, keyframes=keyframes, text=text)
    # 印の後ろの縁取りは、印の所で先に当てる配置（``placed_early``）の後に付く 層にすると
    # 字と一緒に配置の前へ動くので、層にせず並びの位置に残す
    rest = map_video_effects(
        tail, log, length=length, keyframes=keyframes, text=text, placed_before=True
    )
    early = {key: _RESTING[key] for key in _RESTING}
    late = dict(item)
    for flag, keys in _LAZY_PARTS:
        if marker.get(flag) is True:
            for key in keys:
                early[key] = item.get(key, _RESTING[key])
                late[key] = _RESTING[key]
    placed_early = _placement(early, length, keyframes, first.pivot)
    final = _fixed([*flip, *_placement(late, length, keyframes, rest.pivot or first.pivot)])
    params = {**first.params, **rest.params}
    # 両側に縁取りがあるときはテキストへ載せない（上の ``text``）ので、層はどちらか片側にしか無い
    strokes = [*first.strokes, *rest.strokes]
    return params, [*first.effects, *placed_early, *rest.effects], final, strokes


def _fixed(final: list[Effect]) -> list[Effect]:
    """アイテムの描画の欄（反転と最後の配置）に、クリップが最初から持つ項目の印を付ける

    YMM4 の描画の X・Y・拡大率・回転角・左右反転は、アイテムが最初から持つ欄 置くとき
    （:func:`sashimono.compat.catalog.place`）に同じ種類をもう 1 つ足さないよう、ここで
    印を付ける 既定のままで写さなかった欄は、置くときに既定の値で足される
    描画を遅らせる印の所で先に当てる配置（``placed_early``）は欄ではなく並びの途中の
    効果なので、印を付けない
    """
    return [replace(effect, fixed=True) for effect in final]


def _flip(item: dict[str, Any]) -> list[Effect]:
    """アイテムの反転 YMM4 は左右に裏返してから回す（回す向きは変わらない）"""
    if item.get("IsInverted") is not True:
        return []
    definition = registry.get("flip")
    return [] if definition is None else [definition.create(horizontal=True, vertical=False)]


#: 線の図形の塗りに模様を置くときの目印の色の候補（RGB の立方体の角）
#: 塗りだけに模様を掛けるには、形（線と塗り）の中で塗りの場所を伝える必要がある
#: 図形は塗りを 1 色でしか描けないので、塗りを目印の色で描き、あとから模様に替える
_FILL_MARKERS = (
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
)


def fill_marker(stroke: tuple[float, ...]) -> tuple[float, float, float, float]:
    """線の色から一番遠い目印の色

    目印をマゼンタに決め打ちすると、マゼンタに近い線まで塗りの模様に替わる（#199）
    立方体の角のうち一番遠い物を選べば、どんな線の色からも 0.87 以上離れる
    """
    red, green, blue = max(
        _FILL_MARKERS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, stroke[:3], strict=True))
    )
    return (red, green, blue, 1.0)


def _patterned_fill(
    source: GeneratedSource,
    fill: Any,
    log: CompatibilityReport,
    length: int,
    keyframes: Any,
    effects: list[Effect],
) -> GeneratedSource:
    """線の図形の塗りだけを模様にする 目印の色で塗った所を置き換える

    線の色と目印の色を両方渡す 絵にはこの 2 色（と縁でその間の色）しか無いので、
    画素の色が 2 色の間のどこにあるかで塗りの割合が決まる 目印からの近さだけで
    決めると、線と塗りの境の中間色に目印が残る
    """
    stroke = source.params.get("color")
    keep = stroke if isinstance(stroke, tuple) else (1.0, 1.0, 1.0, 1.0)
    marker = fill_marker(keep)
    filled = brush_effect(
        fill,
        log,
        length=length,
        keyframes=keyframes,
        key_only=True,
        key_color=marker,
        # 不透明度は「線の色を渡した」印に使う 線の透け方は絵の α にもう入っている
        keep_color=(keep[0], keep[1], keep[2], 1.0),
    )
    if filled is None:
        return source
    effects.append(filled)
    return source.with_param("fill_color", marker)


def _map_item(item: dict[str, Any], log: CompatibilityReport) -> MappedObject | None:
    name = type_name(item)

    length = max(1, int(number(item.get("Length"), 1.0)))
    keyframes = item.get("KeyFrames")

    if _preview_only(item):
        return None
    if name == "TransitionItem":
        return _transition(item, length, keyframes, log)
    if name == "EffectItem":
        item = _without_ignored_moves(item)
    source, media_path, kind = _content(item, name, log)
    if source is None and not media_path:
        return None

    if item.get("IsAlwaysOnTop") is True or item.get("IsZOrderEnabled") is True:
        log.note_missing("YMM4 のアイテムの重なり順の設定（常に手前・Z 順）")
    effects: list[Effect] = []
    if source is not None and source.kind == "shape":
        # 図形のブラシが単色でなければ、白で描いた形を模様で塗る
        parameter = item.get("ShapeParameter")
        brush = parameter.get("Brush") if isinstance(parameter, dict) else None
        fill = parameter.get("FillBrush") if isinstance(parameter, dict) else None
        painted = None
        if not is_solid(brush):
            painted = brush_effect(
                brush, log, length=length, keyframes=keyframes, pattern_only=True
            )
            if painted is not None:
                source = source.with_param("color", (1.0, 1.0, 1.0, 1.0))
                effects.append(painted)
        if not is_solid(fill) and source.params.get("shape") == "polyline":
            if painted is not None:
                # 線の模様は形（不透明度）だけを借りて塗るので、塗りの所も線の模様になる
                # 塗りの模様を先に置いても上から塗り潰され、後に置くと線の模様の色を
                # 目印と見分けられない 線と塗りを別々に描く作りが要る
                log.note_missing("YMM4 の線の図形で線と塗りの両方が模様（塗りも線の模様で描いた）")
            else:
                source = _patterned_fill(source, fill, log, length, keyframes, effects)
    is_text = source is not None and source.kind == "text"
    params, chain, final, strokes = _video_chain(item, log, length, keyframes, text=is_text)
    if source is not None and is_text:
        decorations = map_decorations(
            item.get("Decorations"),
            log,
            size=number(item.get("FontSize"), 64.0),
            style=str(item.get("Style") or ""),
            style_colour=item.get("StyleColor"),
        )
        merged = {**source.params, **decorations.params, **params}
        if strokes and legacy_in_use(merged):
            # 文字装飾（Style）の縁と映像エフェクトの縁取りの層が両方ある 1 つの縁だけを載せて
            # いた前と同じく映像エフェクトの側を描く（層を持つ字は前からの項目を読まない）
            # 実物の 88 本には無い組み合わせなので、重なり方は確かめずに記録へ残す
            log.note_missing("YMM4 の文字装飾の縁と、2 つ以上の縁取りエフェクトの組み合わせ")
        source = GeneratedSource(kind="text", params=merged, strokes=tuple(strokes))
        effects.extend(decorations.effects)
    effects.extend(chain)

    # YMM4 はエフェクトを掛けた絵を、最後に位置・拡大・回転で置く 先に置くと、
    # 画面の中で動かしたあとの絵にエフェクトが掛かり、回した図形が中心点の前で切れる
    effects.extend(final)

    speed, silenced = _playback_rate(item, name, log, length=length, keyframes=keyframes)
    source_in = _content_offset(item, log) if media_path else Fraction(0)
    return MappedObject(
        clip=Clip(
            timeline_start=max(0, int(number(item.get("Frame"), 0.0))),
            duration=length,
            source=source,
            source_in=source_in,
            speed=speed,
            # 再生速度 0 の動画アイテムは、素材の頭（ContentOffset の位置）の絵で止まる
            # （:func:`_playback_rate` の測り） 鳴らさない（``silenced``）のは 0 のときだけ
            hold_at=source_in if silenced and name == "VideoItem" else None,
            effects=tuple(effects),
            opacity=animated(
                item.get("Opacity"), 100.0, length=length, keyframes=keyframes, scale=0.01
            ),
            blend_mode=_blend_of(item, log),
            # 上のオブジェクト（すぐ下に描かれる層）の形で切り抜く
            clip_to_below=item.get("IsClippingWithObjectAbove") is True,
            # YMM4 は画像・動画を拡大率 100% で素材の画素の大きさに置く 画面に収めると、
            # 画面と違う解像度の素材がテンプレートの拡大率のまま別の大きさになる
            native_size=bool(media_path) and source is None,
        ),
        # YMM4 のレイヤーは 0 始まり こちらのトラックは 1 始まり
        layer=max(1, int(number(item.get("Layer"), 0.0)) + 1),
        media_path=media_path,
        kind=kind,
        has_span="Length" in item,
        # 動画アイテムは 1 つで映像と音の両方を持つ 音声トラックへも展開しないと鳴らない
        with_sound=name == "VideoItem",
        audio_effects=_audio_effects(
            item, name, log, length=length, keyframes=keyframes, silenced=silenced
        ),
        # 素材より長い動画アイテムは、素材の最後の絵を枠の終わりまで出し続ける
        # （2026-09-23 YMM4 4.56.1.1 素材 120 フレーム・枠 180 フレームで、0→119 のあと
        # 119 が 60 回 頭へ戻って繰り返さない） 素材の長さは置く側が素材を読んでから
        # 分かるので、印だけ立てる 繰り返し（IsLooped）の物は止めずに、繰り返しを写せない
        # ことを数えて残す（:func:`_audio_effects`） 止めると繰り返すはずの所が止まった絵になる
        hold_last_frame=name == "VideoItem" and item.get("IsLooped") is not True,
        audio_track=_audio_track(item, name, log),
    )


def _audio_track(item: dict[str, Any], name: str, log: CompatibilityReport) -> int:
    """素材の何本目の音を鳴らすか（``AudioTrackIndex`` 0 始まり）

    素材の中の番号（ストリームの番号）ではなく、音の道だけを数えた順番として持つ
    番号に直すのは素材を読んだ置く側（:func:`~sashimono.compat.catalog.place`）
    YMM4 の画面の「音声トラック」は素材の音を 1 本目・2 本目と並べて選ばせるので、
    映像を含めた番号として読むと 1 つずれた音が鳴る

    読めない値や負の値は 1 本目として置き、数えて残す 黙って 1 本目にすると、
    選んだはずの言語と違う音が鳴っても互換性レポートに出ない
    """
    if name not in _SOUND_ITEMS:
        return 0
    raw = item.get("AudioTrackIndex")
    if raw is None or raw == "":
        return 0
    value = number(raw, 0.0) if _readable_number(raw) else -1.0
    if value < 0 or value != int(value):
        log.note_missing(f"YMM4 の音声トラックの選択（AudioTrackIndex）が読めない値: {raw!r}")
        return 0
    return int(value)


def _content_offset(item: dict[str, Any], log: CompatibilityReport) -> Fraction:
    """素材のどこから再生するか（``ContentOffset``）を秒で

    切り出して使っているテンプレートは、ここを落とすと絵も音も違う所から始まる
    実物（この機械の YMM4 プロジェクト 16 本）の動画 214 個・音声 125 個のうち
    240 個が 0 以外だった

    **効かせるのは素材を持つアイテムだけ** 配布物 230 本を数えると、0 以外なのは
    テキスト 33・図形 25・フレームバッファ 6・グループ 4 と、素材を読まない物ばかりで、
    YMM4 でも絵は動かない（書き出しに残っている既定の値） こちらで効かせると、
    グループ（入れ子のシーン）の時刻だけがずれる

    負の値はこちらの :class:`~sashimono.core.model.Clip` が受け取らない（素材の
    手前から再生することになる） 0 として置き、数えて残す 読めない形も同じ
    """
    raw = item.get("ContentOffset")
    if raw is None or raw == "":
        return Fraction(0)
    offset = timespan(raw)
    if offset is None:
        log.note_missing(f"YMM4 の素材の開始位置（ContentOffset）の書き方: {raw!r}")
        return Fraction(0)
    if offset < 0:
        log.note_missing("YMM4 の素材の開始位置（ContentOffset）が負")
        return Fraction(0)
    return offset


#: 音を持つアイテム 映像を持つのは ``VideoItem`` だけ
_SOUND_ITEMS = frozenset({"VideoItem", "AudioItem", "VoiceItem"})


def _audio_effects(
    item: dict[str, Any],
    name: str,
    log: CompatibilityReport,
    *,
    length: int,
    keyframes: Any,
    silenced: bool = False,
) -> tuple[Effect, ...]:
    """音の設定を、音声トラックのクリップへ掛けるエフェクトにする

    項目の並びは YMM4 自身が書いたものを数えて決めた 手元の YMM4（4.48.0.3）が
    ``user/setting/…/ItemSettings.json`` の ``DefaultItems.VideoItem`` へ書き出す
    既定の動画アイテムと、実際のプロジェクト 16 本に入っていた動画アイテム 214 個・
    音声アイテム 125 個が同じ並びだった（``Volume`` ``Pan`` ``PlaybackRate``
    ``ContentOffset`` ``IsLooped`` ``AudioTrackIndex`` ``AudioEffects`` ``Echo*``）
    アイテムテンプレートの中身も同じ形で入る（同じ設定ファイルの ``Templates`` が、
    ``.ymmt`` の ``catalog.json`` と同じ ``Name`` ``Path`` ``Items`` を持つ）

    写し方は YMM4 本体に書き出させて測って決めた（2026-09-23 YMM4 4.56.1.1
    440Hz の正弦波 基準の枠との最大振幅の比 表は docs/development.md の「音を測る」）

    * ``Volume`` は振幅比の百分率 50 で 0.501・25 で 0.250・0 で無音 YMM4 に音を消す
      印は無く、実物でも音を消した 18 個は ``Volume`` が 0 だった こちらの
      ``audio_volume`` の ``volume`` も振幅比の百分率なので、そのまま渡す
    * ``Pan`` は -100〜100 で負が左 -100 で左 1.00・右 0.00、-50 で左 1.00・右 0.50
      近い側はそのままで、遠い側だけ直線で下がる ``audio_volume`` の ``pan`` と同じ作り
      なので、そのまま渡す
    * ``PlaybackRate`` はクリップの ``speed`` へ（:func:`_playback_rate`）
      0 のとき（``silenced``）は鳴らさないので、音量を 0 にする

    写せないものは数えて残す
    """
    if name not in _SOUND_ITEMS:
        return ()

    def read(key: str, default: float) -> AnimatedValue:
        return animated(item.get(key), default, length=length, keyframes=keyframes)

    def differs(value: AnimatedValue, default: float) -> bool:
        """既定と違う値を持つか 動く値は途中の点まで見る

        先頭の値だけを見ると（``number``）、0 から動き出す定位のように、
        始まりが既定と同じものを取りこぼし、左右へ振る定位が真ん中のまま鳴る
        """
        if value.keyframes:
            return any(point.value != default for point in value.keyframes)
        return value.static != default

    if item.get("IsLooped") is True:
        log.note_missing("YMM4 の素材の繰り返し（IsLooped）")
    if item.get("EchoIsEnabled") is True:
        log.note_missing("YMM4 のエコー")
    for entry in item.get("AudioEffects") or []:
        # 切ってあるエフェクトは鳴り方に関わらない 数えると、直す順番を決めるときに
        # 効いていないものが上位に来る 映像エフェクトの読み方（map_video_effects）と同じ
        # 実物の音声エフェクト 3 個はどれも IsEnabled を持っていた
        if isinstance(entry, dict) and entry.get("IsEnabled") is not False:
            log.note_missing(f"YMM4 の音声エフェクト: {type_name(entry) or '種類不明'}")

    volume = AnimatedValue(0.0) if silenced else read("Volume", 100.0)
    pan = read("Pan", 0.0)
    if not differs(volume, 100.0) and not differs(pan, 0.0):
        # 既定のままなら何も掛けない 音量 100% のエフェクトが並ぶと、
        # 何を変えたテンプレートなのかが設定画面から読めなくなる
        return ()
    definition = registry.get("audio_volume")
    # 音量とパンはアイテムの音声の欄 置くときに同じ種類を足さないよう印を付ける
    return (
        ()
        if definition is None
        else (replace(definition.create(volume=volume, pan=pan), fixed=True),)
    )


def _playback_rate(
    item: dict[str, Any],
    name: str,
    log: CompatibilityReport,
    *,
    length: int,
    keyframes: Any,
) -> tuple[Fraction, bool]:
    """再生速度（``PlaybackRate``）を、クリップの ``speed`` と「鳴らさないか」の組にする

    YMM4 に書き出させて測った（2026-09-23 YMM4 4.56.1.1 2 秒の 440Hz の正弦波を
    4 秒の枠へ置いた） 50 で 3.98 秒・220Hz、200 で 1.00 秒・880Hz、最大振幅は
    変わらない テープのように速さと一緒に高さも変わり、こちらの ``speed``
    （ミキサーの線形の並べ直し）と同じ作り ``Length`` はタイムライン上の長さの
    ままで、素材は ``Length × rate`` だけ進む 50 で 2 秒の素材が 4 秒の枠に収まった
    のがそれで、:class:`~sashimono.core.model.Clip` の ``duration`` と ``speed`` の
    決まり（素材を ``duration × speed`` 読む）と合うので、長さは変えずに渡す

    **0 は鳴らない**（測ると無音で、鳴っている長さも 0） こちらの ``speed`` は
    正の数しか取らないので、``speed`` は 1 のまま置き、音量を 0 にして止める
    （2 つ目の値） クリップを置かない形にしないのは、音声アイテムだと置いた物が
    まるごと消え、あとで速さを直そうにも手を掛ける所が無くなるため YMM4 でも
    音を消したアイテムは ``Volume`` 0 で持つので、同じ形になる
    動画アイテムの ``speed`` は映像のクリップにも効く 絵の速さも YMM4 に書き出させて
    測った（2026-09-23 YMM4 4.56.1.1 ``tools/ymm4_compare.py`` の ``video-rate-build``
    書き出しの各フレームを素材のフレームと突き合わせ、経過フレームに対する傾きを取った）
    100 で 1.00 倍・50 で 0.50 倍・200 で 2.00 倍と、こちらの ``speed`` と一致したので
    数えない 映像と音で分けないのは、リンクした 2 本の速さが違うと絵と音がずれていくため

    **動画アイテムの 0 は、素材の頭（``ContentOffset`` の位置）の絵で止まる**（同じ測り
    枠の 180 フレームすべてが素材の 0 フレーム目） 絵はクリップの
    :attr:`~sashimono.core.model.Clip.hold_at` を ``source_in`` にして止める（呼ぶ側
    :func:`_map_item`） ``speed`` は 1 のままなので、音の側は素材を等倍で読むが、
    音量 0 で鳴らない 実物の 0 は 5 個あり、どれも動画アイテム（mp4 4 個・webp 1 個）

    NaN や無限大は分数にできないので、等倍として置き数えて残す

    音を持たないアイテムは見ない 実物ではどれも 100 で、``speed`` を持たせると
    テキストや図形の動きの時刻まで変わる

    手元のファイル 99 本（配布物・この機械のプロジェクト・測るために作った試料）の
    アイテム 1156 個の ``PlaybackRate`` はどれもただの数だった 形が来たら先頭の値を
    使い、動くなら数えて残す（``speed`` は動かせない）

    新しい版の書き出しは、ほかに動く値の ``PlaybackRate2`` と、音の速さの変え方
    ``PlaybackRateAudioProcessingMode`` も持つ（実物の音を持つアイテム 138 個は
    どれも ``Resampling``、``PlaybackRate2`` は動かず ``PlaybackRate`` と同じ値だった）
    2 つをわざと食い違わせて YMM4 に書き出させると（2026-09-23 YMM4 4.56.1.1
    ``tools/ymm4_compare.py`` の ``video-rate-build`` と ``audio-build``）、絵も音も
    **止まった値が食い違うときは ``PlaybackRate`` が効いた**ので、読むのは
    ``PlaybackRate`` のままにして、食い違いは数えない

    ``PlaybackRate2`` が動くと、YMM4 の絵の速さは途中で変わった（頭が ``PlaybackRate``・
    終わりが ``PlaybackRate2`` の最後の値の直線と読める 1 つの枠からの読みなので
    決め打ちしない 詳しくは docs/development.md） こちらの ``speed`` は動かせないので、
    頭の値（``PlaybackRate``）で置いて、動く速さは写せないと数えて残す

    ``Sola`` は高さを保って長さだけを変える変え方だった（50 で 3.97 秒・200 で 1.00 秒、
    どちらも 440Hz のまま） こちらの ``speed`` はミキサーの線形の並べ直しだけで、
    高さを保つ変え方を持たないので、長さは ``speed`` で写し、高さも変わることを
    数えて残す 列挙の値は本体の DLL で ``Resampling`` と ``Sola`` の 2 つだけ
    それ以外の値は、まだ見ていない変え方として数えて残す
    """
    if name not in _SOUND_ITEMS:
        return Fraction(1), False
    raw = item.get("PlaybackRate")
    if raw is not None and not _readable_number(raw):
        # 文字や中身の無い Values は number が既定の 100 へ丸めるので、黙っていると
        # 壊れた値が等倍として写り、互換性レポートにも出ない NaN や無限大は分数にできず、
        # そのまま渡すと ValueError で読み込みごと止まり、同じテンプレートの正常な
        # アイテムまで写せなくなる どれも等倍として置き、数えて残す
        log.note_missing(f"YMM4 の再生速度（PlaybackRate）が読めない値: {raw!r}")
        return Fraction(1), False
    rate = number(raw, 100.0)

    # 動きはほかの項目と同じくアイテムの長さと中間点で読む 既定の長さ 1 で読むと、
    # 3 点目以降が同じフレームに重なって捨てられ、途中の値だけが違う動きを数え落とす
    def read(value: Any, default: float) -> AnimatedValue:
        return animated(value, default, length=length, keyframes=keyframes)

    if any(point.value != rate for point in read(raw, 100.0).keyframes):
        log.note_missing("YMM4 の再生速度（PlaybackRate）の動き（先頭の値で写した）")
    newer = item.get("PlaybackRate2")
    if newer is not None and not _readable_number(newer):
        log.note_missing(f"YMM4 の再生速度（PlaybackRate2）が読めない値: {newer!r}")
    elif newer is not None:
        # 止まった値の食い違いは見ない YMM4 は PlaybackRate を読むと測って確かめた
        moving = read(newer, rate)
        if moving.keyframes and any(point.value != rate for point in moving.keyframes):
            log.note_missing(
                "YMM4 の再生速度（PlaybackRate2）の動き 動く速さは写せない（PlaybackRate で置いた）"
            )
    mode = item.get("PlaybackRateAudioProcessingMode")
    if mode == "Sola":
        # 等倍なら高さを保つかどうかで音は変わらず、0 は鳴らない 数えると、
        # 高さの変わらないアイテムまで直す候補に並ぶ
        if rate not in (0.0, 100.0):
            log.note_missing(
                "YMM4 の再生速度の音の変え方 Sola（高さを保つ）は写せない 高さも変わる"
            )
    elif mode is not None and mode != "Resampling":
        log.note_missing(f"YMM4 の再生速度の音の変え方: {mode}")
    if rate == 0:
        return Fraction(1), True
    if rate < 0:
        # 負の値は実物に無く、YMM4 でどう鳴るかも測っていない
        log.note_missing("YMM4 の再生速度（PlaybackRate）が負")
        return Fraction(1), False
    # 2 進の小数のまま分数にすると 102.1 が長い分母の分数になる 書かれた 10 進で持つ
    return Fraction(repr(rate)) / 100, False


def _readable_number(value: Any) -> bool:
    """数として読める形か ただの数・数の文字・値を 1 つ以上持つ動く値 どれも有限に限る

    ``number`` と ``animated`` は読めない形を既定値へ丸める 丸めた後では、書かれて
    いた値が既定だったのか壊れていたのか見分けられないので、丸める前に見る
    真偽値は数に読めるが（``True`` が 1）、速さとして書かれることは無いので断る

    NaN や無限大も ``animated`` に渡す前の ``Values`` の並びで見る ``animated`` は
    同じフレームに重なる値を捨てるので（長さ 1 のアイテムの 3 点など）、読んだ後の
    キーフレームで見ると、重なって捨てられた NaN を見落とす
    ``PlaybackRate2`` は写す値に使わないので、ここで見ないとどこにも引っ掛からない
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float | str):
        # float に直せない桁の整数や数でない文字は、`number` が既定へ丸める
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False
    if isinstance(value, dict):
        values = value.get("Values")
        return (
            isinstance(values, list)
            and bool(values)
            and all(
                isinstance(entry, dict) and _readable_number(entry.get("Value")) for entry in values
            )
        )
    return False


#: YMM4 の切り替えの種類と、場面切り替えの切り替え方
_TRANSITION_STYLES = {
    "SwitchTransitionPlugin": "switch",
    "FadeTransitionPlugin": "fade",
    "PushTransitionPlugin": "push",
    "SlideTransitionPlugin": "slide",
    "NoneTransitionPlugin": "overlay",
}


def _transition(
    item: dict[str, Any], length: int, keyframes: Any, log: CompatibilityReport
) -> MappedObject | None:
    """場面切り替え 前の場面のエフェクトはクリップ、後の場面のエフェクトは after_effects へ

    位置・拡大・回転・不透明度はアイテムの設定画面に出ない（配布物ではどれも既定のまま）
    """
    plugin = str(item.get("TransitionType") or "").partition(",")[0].rpartition(".")[2]
    style = _TRANSITION_STYLES.get(plugin)
    if style is None:
        log.note_missing(f"YMM4 の場面切り替えの種類: {plugin or '種類不明'}")
        style = "fade"
    raw = item.get("TransitionParameter")
    parameter = raw if isinstance(raw, dict) else {}
    target = str(parameter.get("Target") or parameter.get("OverlayTarget") or "After")
    if target not in ("Before", "After"):
        log.note_missing(f"YMM4 の場面切り替えの対象: {target}")
    easing = str(parameter.get("EasingType") or "Linear")
    if easing not in _EASING_NAMES:
        log.note_missing(f"YMM4 の場面切り替えのイージング: {easing}")
    easing_mode = str(parameter.get("EasingMode") or "In")
    if easing_mode not in _EASING_MODE_NAMES:
        log.note_missing(f"YMM4 の場面切り替えのイージングの向き: {easing_mode}")
    definition = source_registry.get("transition")
    if definition is None:  # pragma: no cover - 標準の生成オブジェクト
        return None
    source = definition.create(
        style=style,
        # 押し出しの角度は YMM4 が見ていない（90 にしても 0 と同じ絵だった）
        angle=0.0 if style == "push" else number(parameter.get("Angle"), 0.0),
        target="before" if target == "Before" else "after",
        easing=_EASING_NAMES.get(easing, "linear"),
        easing_mode=_EASING_MODE_NAMES.get(easing_mode, "in"),
    )
    before = map_video_effects(
        item.get("BeforeVideoEffects"), log, length=length, keyframes=keyframes
    )
    after = map_video_effects(
        item.get("AfterVideoEffects"), log, length=length, keyframes=keyframes
    )
    if map_video_effects(item.get("VideoEffects"), log, length=length, keyframes=keyframes).effects:
        log.note_missing("YMM4 の場面切り替えのアイテム自体に掛けたエフェクト")
    return MappedObject(
        clip=Clip(
            timeline_start=max(0, int(number(item.get("Frame"), 0.0))),
            duration=length,
            source=source,
            effects=tuple(before.effects),
            after_effects=tuple(after.effects),
        ),
        layer=max(1, int(number(item.get("Layer"), 0.0)) + 1),
        media_path="",
        kind="transition",
    )


#: 切り替えのイージングの名前
_EASING_NAMES = {
    name: name.lower()
    for name in ("Linear", "Sine", "Quad", "Cubic", "Quart", "Quint", "Expo", "Circ", "Back")
} | {"Elastic": "elastic", "Bounce": "bounce", "Jump": "jump"}
_EASING_MODE_NAMES = {"In": "in", "Out": "out", "InOut": "inout"}


#: エフェクトアイテムに積んでも YMM4 の絵が変わらなかったエフェクト（2026-09-24 YMM4 4.56.1.1
#: ``tools/ymm4_compare.py`` の ``effectitem-build`` 範囲は画面全体） 画面いっぱいの絵の上で
#: 拡大率 50 も X 300 も、書き出しは下の絵のままだった 写すと、縮めた・ずらした写しが
#: 下の絵の上に重なる（差 16.8 と 83.7 → 1.2 前後）
_EFFECT_ITEM_IGNORED = frozenset({"ZoomEffect", "DrawPositionEffect"})


def _without_ignored_moves(item: dict[str, Any]) -> dict[str, Any]:
    """エフェクトアイテムから、YMM4 が絵に当てなかったエフェクトを除いた写し

    除くのは測った条件（範囲が画面全体の背景）だけ ほかの範囲では変形が範囲を動かす
    かもしれず、測っていない 除くと利用者は変形が消えたことを知る手立てが無いので、
    そのまま写す（範囲そのものは :func:`_content` が未対応として記録する）
    """
    if not _range_plugin(item).startswith("Background"):
        return item
    raw = item.get("VideoEffects")
    if not isinstance(raw, list):
        return item
    kept = [
        entry
        for entry in raw
        if not (isinstance(entry, dict) and type_name(entry) in _EFFECT_ITEM_IGNORED)
    ]
    return item if len(kept) == len(raw) else {**item, "VideoEffects": kept}


def _range_plugin(item: dict[str, Any]) -> str:
    """エフェクトアイテムの範囲の種類 型の名前の最後の部分（``BackgroundShapePlugin`` など）

    ``ShapeType2`` は ``Version=4.32.0.2`` のようなアセンブリの版まで付いた名前で、同じ
    範囲でも書き出した YMM4 の版で文字列が変わる 名前空間とアセンブリを落として比べる
    """
    return str(item.get("ShapeType2") or "").partition(",")[0].rpartition(".")[2]


def _preview_only(item: dict[str, Any]) -> bool:
    """編集中の画面にだけ映すアイテムか YMM4 は書き出した動画に出さない

    目印や下書きに使われる 読み込むと書き出しに映り込む
    """
    effects = item.get("VideoEffects")
    return isinstance(effects, list) and any(
        isinstance(entry, dict)
        and entry.get("IsEnabled") is not False
        and type_name(entry) == "ShowOnlyPreviewEffect"
        for entry in effects
    )


def _blend_of(item: dict[str, Any], log: CompatibilityReport) -> str:
    raw = str(item.get("Blend") or "Normal")
    mode = _BLEND_MODES.get(raw)
    if mode is None:
        log.note_missing(f"YMM4 の合成モード: {raw}")
        return "normal"
    return mode


def _content(
    item: dict[str, Any], name: str, log: CompatibilityReport
) -> tuple[GeneratedSource | None, str, str]:
    if name in ("TextItem", "Text"):
        return _text(item, log), "", "text"
    if name in ("ShapeItem", "Shape"):
        return _shape(item, log), "", "shape"

    if name == "EffectItem":
        # 下のレイヤーの絵にエフェクトを掛けるアイテム（範囲は図形で決める） 図形として
        # 読むと、範囲の図形（多くは画面全体の背景）がそのまま画面を塗りつぶす
        # 写し取った画面にエフェクトを掛けるフレームバッファと同じ形で読む
        # フィルタのクリップ（下の絵に掛けて置き換える 透明は透明のまま）とは読まない
        # YMM4 は下の絵の透明な所も黒として掛けた（2026-09-24 YMM4 4.56.1.1 ``effectitem-build``
        # 周りが透明な図形に反転を掛けると周りが白くなり、前景の塗りつぶしは周りまで塗った
        # フィルタで読むと周りが黒のままで、差が 0.1 → 226.8）
        plugin = _range_plugin(item)
        if plugin and not plugin.startswith("Background"):
            log.note_missing(f"YMM4 のエフェクトアイテムの範囲: {plugin}")
        if number(item.get("Blur"), 0.0) > 0 or item.get("InvertMask") is True:
            log.note_missing("YMM4 のエフェクトアイテムの範囲のぼかしと反転")
        return GeneratedSource(kind="framebuffer"), "", "framebuffer"

    if name == "FrameBufferItem":
        # それまでに重ねた画面を素材にする 中身の設定は持たない
        return GeneratedSource(kind="framebuffer"), "", "framebuffer"

    media = _MEDIA_ITEMS.get(name)
    if media is not None:
        return None, str(item.get("FilePath") or item.get("File") or ""), media

    log.note_missing(f"YMM4 のアイテム: {name or '種類不明'}")
    return None, "", name


def _text(item: dict[str, Any], log: CompatibilityReport) -> GeneratedSource:
    length = max(1, int(number(item.get("Length"), 1.0)))
    keyframes = item.get("KeyFrames")
    size = number(item.get("FontSize"), 64.0)
    align, valign = _base_point(item)

    params: dict[str, ParamValue] = {
        "text": str(item.get("Text") or ""),
        "size": animated(item.get("FontSize"), 64.0, length=length, keyframes=keyframes),
        "color": colour(item.get("FontColor"), (1.0, 1.0, 1.0, 1.0)),
        "bold": bool(item.get("Bold")),
        "italic": bool(item.get("Italic")),
        # ``LineHeight2`` は百分率（100 が標準） 画素数だと思って渡すと、
        # 標準のつもりが 100px の行間になる
        "line_spacing": AnimatedValue(
            size * (number(item.get("LineHeight2"), 100.0) - 100.0) / 100.0
        ),
        "letter_spacing": animated(
            item.get("LetterSpacing2"), 0.0, length=length, keyframes=keyframes
        ),
        "align": align,
        # YMM4 の基準位置の左右は、文字の塊の端を位置に合わせる（行揃えと同じ向き）
        "anchor": align,
        "valign": valign,
        "vertical": _is_vertical(item),
    }
    font = item.get("Font")
    if isinstance(font, str) and font:
        params["font"] = font
    wrap = str(item.get("WordWrap") or "NoWrap")
    if wrap != "NoWrap":
        # 折り返す物だけ MaxWidth を折り返しの幅にする（#249） 実物の 114 本・テキスト 310 個は
        # どれも NoWrap と MaxWidth 1920 で、折り返す見本はまだ無い 位置の決まり（禁則・
        # 英単語）を YMM4 と描き比べていないので、読んだことを互換性レポートに残して数える
        params["wrap_width"] = animated(
            item.get("MaxWidth"), 1920.0, length=length, keyframes=keyframes
        )
        log.note_missing(f"YMM4 のテキストの折り返し（位置の決まりは実物で未確認）: {wrap}")
    return GeneratedSource(kind="text", params=params)


def _base_point(item: dict[str, Any]) -> tuple[str, str]:
    """``BasePoint`` を横と縦に分ける ``CenterCenter`` のように 2 つ並ぶ"""
    raw = str(item.get("BasePoint") or "")
    align = next((value for key, value in _HORIZONTAL if raw.startswith(key)), "center")
    valign = next((value for key, value in _VERTICAL if raw.endswith(key)), "middle")
    return align, valign


def _is_vertical(item: dict[str, Any]) -> bool:
    direction = str(item.get("FontDirection") or item.get("TextDirection") or "")
    return "Vertical" in direction or "縦" in direction


def _shape(item: dict[str, Any], log: CompatibilityReport) -> GeneratedSource:
    parameter = item.get("ShapeParameter")
    parameter = parameter if isinstance(parameter, dict) else {}

    # 種類はプラグイン名に入っている（``BackgroundShapePlugin`` など）
    raw = str(item.get("ShapeType2") or item.get("ShapeType") or item.get("Type") or "")
    plugin = raw.partition(",")[0].rpartition(".")[2]
    if plugin.startswith("LineShape"):
        return _line(parameter, log)
    if plugin.startswith("PenShape"):
        return _pen(parameter, item, log)
    if plugin.startswith("TimerShape"):
        return _timer(parameter, item)
    if plugin.startswith("ConcentrationLineShape"):
        return _concentration(parameter, item)
    shape = next(
        (value for key, value in _SHAPES.items() if plugin.startswith(key)),
        None,
    )
    if shape is None:
        shape = _SHAPES.get(type_name(parameter).removesuffix("ShapeParameter"), "")
    if not shape:
        log.note_missing(f"YMM4 の図形: {plugin or type_name(parameter) or '種類不明'}")
        shape = "rect"

    # 色はブラシ（``Brush.Parameter.Color``）に入っている 古い形だけが直に ``Color`` を持つ
    # ブラシを見ないと、配布物の図形がすべて白で出る
    fallback = colour(parameter.get("Color"), (1.0, 1.0, 1.0, 1.0))
    # 単色以外のブラシは、アイテムを写すとき（_map_item）に模様で塗るエフェクトを足す
    colour_value = brush_colour(parameter.get("Brush"), fallback)

    length = max(1, int(number(item.get("Length"), 1.0)))
    keyframes = item.get("KeyFrames")

    def track(key: str, default: float) -> AnimatedValue:
        # 大きさや線の太さも動く（斜めに伸びる帯のトランジションなど） 先頭の値だけ
        # 読むと、0 から伸びる図形がずっと見えない
        return animated(parameter.get(key), default, length=length, keyframes=keyframes)

    width = track("Width", 400.0)
    height = track("Height", 400.0)
    if str(parameter.get("SizeMode") or "") in ("Size", "SizeAspect"):
        # 大きさ 1 つと縦横比で決める形 縦横比は -100〜100 で、正なら縦長
        size = track("Size", 100.0)
        aspect = number(parameter.get("AspectRate"), 0.0) / 100.0
        width = _scaled(size, 1.0 - max(0.0, aspect))
        height = _scaled(size, 1.0 + min(0.0, aspect))

    params: dict[str, ParamValue] = {
        "shape": shape,
        "width": width,
        "height": height,
        "color": colour_value,
    }
    # YMM4 の線の太さは図形の内側へ描く 半分の大きさを超えれば塗りつぶしと同じ
    # （配布物は塗りつぶしに 4000 や 10000 を入れている） こちらの線は輪郭の上に
    # 中心を置いて描くので、そのまま渡すと外側へ数千画素はみ出して画面を覆う
    # 塗りつぶしなら線を付けず、枠だけなら線の太さの分だけ内側へ縮めて描く
    thickness = max(0.0, number(parameter.get("StrokeThickness"), 0.0))
    if shape != "background" and 0.0 < thickness * 2.0 < min(_peak(width), _peak(height)):
        params["outline_only"] = True
        params["line_width"] = AnimatedValue(thickness)
        params["width"] = _shifted(width, -thickness)
        params["height"] = _shifted(height, -thickness)
        # 線の位置は中央のまま 大きさを縮めて内側に見せる書き方で YMM4 の見本と
        # 合わせてあるので、内側に引く（#87 の既定）と二重に細って合わなくなる
        params["line_align"] = "center"
    if params["shape"] == "fan":
        params["span"] = track("CenterAngle", 360.0)
    elif params["shape"] == "arrow":
        params["bar_length"] = track("BarLength", 50.0)
        params["bar_thickness"] = track("BarThickness", 50.0)
    elif params["shape"] == "superformula":
        params["formula_m"] = track("M", 4.0)
        params["formula_n"] = track("N", 1.0)
    round_value = number(parameter.get("Round"), 0.0)
    if shape == "rect" and round_value > 0:
        params["shape"] = "rounded"
        params["corner_radius"] = track("Round", 0.0)
    return GeneratedSource(kind="shape", params=params)


#: 破線の種類と、線の太さを 1 とした長さの並び（Direct2D の決まった模様）
_DASHES = {
    "Solid": "",
    "Dash": "2,2",
    "Dot": "0,2",
    "DashDot": "2,2,0,2",
    "DashDotDot": "2,2,0,2,0,2",
}


def _pen(
    parameter: dict[str, Any], item: dict[str, Any], log: CompatibilityReport
) -> GeneratedSource:
    """手描きの線（ペン） 点は中心からの画素で Y は下が正

    ``Offset`` と ``Length`` は線のどこからどこまでを描くかの割合 太さは描いたときの
    ペンの幅（``DrawingAttributes.Width``）に ``Thickness`` の割合を掛ける
    """
    length = max(1, int(number(item.get("Length"), 1.0)))
    keyframes = item.get("KeyFrames")

    def track(key: str, default: float) -> AnimatedValue:
        return animated(parameter.get(key), default, length=length, keyframes=keyframes)

    strokes = parameter.get("Strokes")
    strokes = strokes if isinstance(strokes, list) else []
    if len(strokes) > 1:
        log.note_missing(f"YMM4 のペンの図形の線の本数（{len(strokes)} 本のうち 1 本目だけ描いた）")
    first = strokes[0] if strokes and isinstance(strokes[0], dict) else {}
    attributes = first.get("DrawingAttributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    # 点は画面の左上を原点にした画素で Y は下が正 画面の大きさによらず同じ画素の所に出る
    # （1920x1080 と 1280x720 で同じ点を YMM4 に描かせて確かめた #198） 真ん中からの画素へ
    # 直すのは描くとき（``points_from``） 読む所で 1920x1080 を決め打ちで引くと、ほかの
    # 大きさのプロジェクトで線がずれる
    points = [
        f"{number(point.get('X'), 0.0):g},{number(point.get('Y'), 0.0):g}"
        for point in first.get("StylusPoints") or []
        if isinstance(point, dict)
    ]
    offset = track("Offset", 0.0)
    span = track("Length", 100.0)
    return GeneratedSource(
        kind="shape",
        params={
            "shape": "polyline",
            "points": ";".join(points),
            "points_from": "corner",
            "closed": False,
            "color": colour(attributes.get("Color"), (1.0, 1.0, 1.0, 1.0)),
            "line_width": _scaled(
                track("Thickness", 100.0), number(attributes.get("Width"), 10.0) / 100.0
            ),
            "trim_start": offset,
            "trim_end": _summed(offset, span),
            "fill_color": (1.0, 1.0, 1.0, 0.0),
        },
    )


def _timer(parameter: dict[str, Any], item: dict[str, Any]) -> GeneratedSource:
    """時間を数える図形 文字として描く 数え下げはクリップの終わりで初めの値になる"""
    length = max(1, int(number(item.get("Length"), 1.0)))
    keyframes = item.get("KeyFrames")
    size = number(parameter.get("FontSize"), 64.0)
    align, valign = _base_point(parameter)
    params: dict[str, ParamValue] = {
        "timer_format": str(parameter.get("Format") or "s"),
        "timer_start": animated(
            parameter.get("InitialValue"), 0.0, length=length, keyframes=keyframes
        ),
        "timer_rate": animated(
            parameter.get("PlaybackRate"), 100.0, length=length, keyframes=keyframes
        ),
        "timer_countdown": str(parameter.get("Direction") or "") == "CountDown",
        "timer_length": length,
        "size": AnimatedValue(size),
        "color": colour(parameter.get("FontColor"), (1.0, 1.0, 1.0, 1.0)),
        "bold": bool(parameter.get("Bold")),
        "italic": bool(parameter.get("Italic")),
        "letter_spacing": animated(
            parameter.get("LetterSpacing2"), 0.0, length=length, keyframes=keyframes
        ),
        "align": align,
        "anchor": align,
        "valign": valign,
    }
    font = parameter.get("Font")
    if isinstance(font, str) and font:
        params["font"] = font
    return GeneratedSource(kind="text", params=params)


def _concentration(parameter: dict[str, Any], item: dict[str, Any]) -> GeneratedSource:
    """集中線 大きさは線が届く円の直径、中心の幅はぼかし、速さは選び直す回数"""
    length = max(1, int(number(item.get("Length"), 1.0)))
    keyframes = item.get("KeyFrames")

    def track(key: str, default: float) -> AnimatedValue:
        return animated(parameter.get(key), default, length=length, keyframes=keyframes)

    size = track("Size", 1000.0)
    return GeneratedSource(
        kind="shape",
        params={
            "shape": "concentration",
            "width": size,
            "height": size,
            "color": colour(parameter.get("Stroke"), (1.0, 1.0, 1.0, 1.0)),
            "density": track("Density", 80.0),
            "line_thickness": track("Thickness", 50.0),
            "line_length": track("Length", 70.0),
            "softness": track("CenterWidth", 50.0),
            "flicker": track("Speed", 5.0),
        },
    )


def _summed(first: AnimatedValue, second: AnimatedValue) -> AnimatedValue:
    """2 つの動く値を足す 片方だけが動くときは、動く側の各点で足す"""
    if not second.is_animated:
        return AnimatedValue(
            first.static + second.static,
            tuple(replace(k, value=k.value + second.static) for k in first.keyframes),
        )
    return AnimatedValue(
        first.static + second.static,
        tuple(replace(k, value=k.value + first.static) for k in second.keyframes),
    )


def _line(parameter: dict[str, Any], log: CompatibilityReport) -> GeneratedSource:
    """線の図形 点は中心からの画素で Y は下が正 閉じていれば中を塗る"""
    points: list[str] = []
    for point in parameter.get("Points") or []:
        if not isinstance(point, dict):
            continue
        x_value = animated(point.get("X"), 0.0)
        y_value = animated(point.get("Y"), 0.0)
        if x_value.is_animated or y_value.is_animated:
            log.note_missing("YMM4 の線の図形の点の動き（先頭の位置で描いた）")
        x = x_value.keyframes[0].value if x_value.is_animated else x_value.static
        y = y_value.keyframes[0].value if y_value.is_animated else y_value.static
        # YMM4 の点は下が正 こちらは上が正
        points.append(f"{x:g},{-y:g}")
    style = str(parameter.get("DashStyle") or "Solid")
    dash = _DASHES.get(style)
    if dash is None:
        dash = str(parameter.get("DashPattern") or "")
    fill = parameter.get("FillBrush")
    stroke = brush_colour(parameter.get("Brush"), (1.0, 1.0, 1.0, 1.0))
    # 単色以外の塗りは、目印の色で塗っておいて、アイテムを読む所で模様に置き換える
    fill_colour = (
        brush_colour(fill, (1.0, 1.0, 1.0, 0.0)) if is_solid(fill) else fill_marker(stroke)
    )
    return GeneratedSource(
        kind="shape",
        params={
            "shape": "polyline",
            "points": ";".join(points),
            "trim_end": animated(parameter.get("LengthRate"), 100.0),
            "line_type": "quadratic"
            if str(parameter.get("LineType") or "") == "QuadraticBezier"
            else "straight",
            "closed": parameter.get("IsClosed") is True,
            "fill_color": fill_colour,
            "color": stroke,
            "line_width": AnimatedValue(
                number(animated(parameter.get("Thickness"), 1.0).static, 1.0)
            ),
            "dash": dash,
        },
    )


def _peak(value: AnimatedValue) -> float:
    return max((k.value for k in value.keyframes), default=value.static)


def _shifted(value: AnimatedValue, delta: float) -> AnimatedValue:
    return replace(
        value,
        static=value.static + delta,
        keyframes=tuple(replace(k, value=k.value + delta) for k in value.keyframes),
    )


def _scaled(value: AnimatedValue, factor: float) -> AnimatedValue:
    return replace(
        value,
        static=value.static * factor,
        keyframes=tuple(replace(k, value=k.value * factor) for k in value.keyframes),
    )


def _placement(
    item: dict[str, Any], length: int, keyframes: Any, pivot: CenterPoint | None = None
) -> list[Effect]:
    """位置・拡大・回転を変形エフェクトへ

    AviUtl 側（:func:`~sashimono.compat.aviutl.mapping.map_object`）と同じ扱いに
    しておく クリップは配置を持たないので、見た目が同じになるエフェクトへ写す
    """
    definition = registry.get("transform")
    if definition is None:  # pragma: no cover - 標準エフェクトは必ずある
        return []

    pos_x = animated(item.get("X"), 0.0, length=length, keyframes=keyframes)
    # YMM4 の Y は下向き こちらは上向き
    pos_y = _negated(animated(item.get("Y"), 0.0, length=length, keyframes=keyframes))
    zoom = animated(item.get("Zoom"), 100.0, length=length, keyframes=keyframes)
    rotation = animated(item.get("Rotation"), 0.0, length=length, keyframes=keyframes)

    moves = pivot is not None and not pivot.keep
    resting = ((pos_x, 0.0), (pos_y, 0.0), (zoom, 100.0), (rotation, 0.0))
    if not moves and not any(value.is_animated or value.static != rest for value, rest in resting):
        return []

    # 縦の拡大率（``scale_y``）は拡大率に掛ける比なので既定の 100 のまま 拡大率を
    # 両方へ入れると縦だけ 2 回掛かり、拡大率 200 の 640x360 が 1280x1440 になる
    # （2026-09-24 YMM4 4.56.1.1 の書き出しは 1280x720 ``zoom-build`` の拡大率 200 の枠）
    placed = definition.create(
        pos_x=pos_x,
        pos_y=pos_y,
        scale=zoom,
        rotation=rotation,
        # 中心点で「位置を保つ」を切ると、選んだ点がアイテムの位置へ来る
        move_to_pivot=moves,
    )
    # 中心点はアイテム自身の拡大と回転の支点にもなる
    return [placed if pivot is None else with_pivot(placed, pivot)]


def _negated(value: AnimatedValue) -> AnimatedValue:
    return AnimatedValue(
        static=-value.static,
        keyframes=tuple(replace(k, value=-k.value) for k in value.keyframes),
    )
