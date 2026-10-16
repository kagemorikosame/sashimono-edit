"""YMM4 本体の絵と Sashimono の絵を並べて比べる

YMM4 のテンプレートは、値の意味を配布物の並びから読み取って写している 読めて描ける
ことはテストで確かめられるが、**YMM4 と同じ絵になるか**は本体で描いてみないと分からない

使い方（3 段）

1. ``build``   テンプレートを時間をずらして並べた YMM4 のプロジェクト（.ymmp）と、
               どこに何を置いたかの一覧（manifest.json）を作る
2. YMM4 でそのプロジェクトを開き、同じフォルダへ ``ymm4.mp4`` として書き出す（手作業か道具）
3. ``compare`` 書き出した動画と、同じアイテムを Sashimono で描いた絵を並べ、差の大きい順に
               一覧（report.html）と並べた絵（PNG）を作る

音は絵と別の 2 段（Issue #89 の残り）

1. ``audio-build``   正弦波を並べた探り用のプロジェクト（audio-probe.ymmp）を作る
2. YMM4 でそれを開き、同じフォルダへ ``audio-probe.mp4`` として書き出す（手作業か道具）
3. ``audio-measure`` 書き出した音を枠ごとに測り、音量の曲線・定位の向き・
                     再生速度 0 の意味を表にする

格子の点の並びも別の 2 段（Issue #107）

1. ``mesh-build``   格子の点を 1 つずつ動かした探り用のプロジェクト（mesh-probe.ymmp）を作る
2. YMM4 でそれを開き、同じフォルダへ ``mesh-probe.mp4`` として書き出す（手作業か道具）
3. ``mesh-measure`` 枠ごとに、動いた点が画面のどこに出たかを YMM4 と Sashimono で並べ、
                    ``Points`` が行ごとか列ごとかを表にする

動画アイテムの再生速度が絵をどう進めるかも別の 2 段（Issue #89 の残り）

1. ``video-rate-build``   フレームごとに絵が変わる動画を、``PlaybackRate`` を変えて並べた
                          探り用のプロジェクト（video-rate-probe.ymmp）を作る
2. YMM4 でそれを開き、同じフォルダへ ``video-rate-probe.mp4`` として書き出す（手作業か道具）
3. ``video-rate-measure`` 書き出しの各フレームが素材の何フレーム目かを枠ごとに並べ、
                          経過フレームに対する傾き（1 で等倍・0 で止まる）を表にする

拡大率 100% で素材をどの大きさに置くかも別の 2 段（Issue #159）

1. ``zoom-build``   画面と違う大きさの画像と動画を並べた探り用のプロジェクト
                    （zoom-probe.ymmp）を作る
2. YMM4 でそれを開き、同じフォルダへ ``zoom-probe.mp4`` として書き出す（手作業か道具）
3. ``zoom-measure`` 素材の真ん中の印が画面に出た大きさを測り、素材の画素のままか
                    画面に収めたかを表にする

エフェクトアイテムが下の絵にどう掛かるかも別の 2 段（Issue #143）

1. ``effectitem-build``   周りが透明な図形と画面いっぱいの絵に、反転・縮める・ずらすを
                          エフェクトアイテムで掛けた探り用のプロジェクト（effectitem-probe.ymmp）を作る
2. YMM4 でそれを開き、同じフォルダへ ``effectitem-probe.mp4`` として書き出す（手作業か道具）
3. ``effectitem-measure`` 同じ枠を「黒を敷いて上に描く」と「下の絵に掛けて置き換える」の
                          2 通りで描き、YMM4 の書き出しに近い方を表にする

絵と音の速さの探りは、どちらも後ろに ``PlaybackRate`` と ``PlaybackRate2`` を
わざと食い違わせた枠を持つ（Issue #117） 測る側は、どちらの値の予想に近いかを並べる
音はさらに ``PlaybackRateAudioProcessingMode`` を ``Sola`` にした枠を持つ

``tools/ymm4_export.py`` は YMM4 の画面を操作して書き出す 動かしている間はマウスと
キーボードを取り合うので、本人に断ってから走らせる 各 ``*-build`` がその命令を出す

作業フォルダは既定で ``.work/ymm4-compare`` リポジトリには入れない
（配布物の絵が入るため）
"""

from __future__ import annotations

import argparse
import copy
import html
import itertools
import json
import math
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from sashimono.compat.mapped import MappedObject
    from sashimono.core.model import MediaItem

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sashimono.compat.aviutl.report import CompatibilityReport  # noqa: E402
from sashimono.compat.catalog import place  # noqa: E402
from sashimono.compat.ymm4.template import load_template, map_template  # noqa: E402
from sashimono.compat.ymm4.values import number, type_name  # noqa: E402
from sashimono.engine.decode.rational import as_fraction  # noqa: E402

WIDTH, HEIGHT, FPS = 1920, 1080, 30
#: 比べる絵の大きさ YMM4 の書き出しは圧縮されるので、縮めてならしてから比べる
#: 細い縁の違いはならすと埋もれるので、縮めない絵でも別の数（:func:`measure`）を出す
COMPARE_WIDTH, COMPARE_HEIGHT = 480, 270
#: 縮める割合（縦横それぞれ）
SHRINK = WIDTH // COMPARE_WIDTH
#: テンプレート 1 本あたりの最短の枠（フレーム）
MIN_SLOT = 60
#: 枠と枠の間に空ける黒 前のテンプレートの残りが次へ混ざらないように
GAP = 6
#: アイテムが終わった後も YMM4 の絵が残るエフェクト
LINGERING_EFFECTS = frozenset({"AfterImageEffect"})
#: 残る絵のために、アイテムの終わりから後ろへ見込むフレーム YMM4 の残像がどれだけ
#: 残るかは測っていない こちらの残像は強さ 53.6（配布物の値）で 7 フレームほどで
#: 見えなくなる 1 秒あれば足りると見て、空きの 6 フレームより大きく取る
LINGER = 30
DEFAULT_WORK = ROOT / ".work" / "ymm4-compare"
#: 置いても比べられないアイテム 場面切り替えは前後の絵が要る
SKIPPED_ITEMS = frozenset({"AudioItem"})

_BRUSH_PARAMETER = "YukkuriMovieMaker.Plugin.Brush.SolidColorBrushParameter, YukkuriMovieMaker"


def _still(value: float) -> dict[str, Any]:
    return {"Values": [{"Value": value}], "Span": 0.0, "AnimationType": "なし"}


def base_shape(frame: int, layer: int, length: int) -> dict[str, Any]:
    """エフェクトだけのテンプレートを着せる下地 角のある図形の方が縁や影の違いが見える"""
    return {
        "$type": "YukkuriMovieMaker.Project.Items.ShapeItem, YukkuriMovieMaker",
        "ShapeType2": "YukkuriMovieMaker.Shape.QuadrilateralShapePlugin, YukkuriMovieMaker",
        "ShapeParameter": {
            "$type": "YukkuriMovieMaker.Project.Items.RectangleShapeParameter, YukkuriMovieMaker",
            "Round": _still(0.0),
            "SizeMode": "WidthHeight",
            "Size": _still(300.0),
            "AspectRate": _still(0.0),
            "Width": _still(640.0),
            "Height": _still(360.0),
            "StrokeThickness": _still(10000.0),
            "Brush": {
                "Type": "YukkuriMovieMaker.Plugin.Brush.SolidColorBrushPlugin, YukkuriMovieMaker",
                "Parameter": {
                    "$type": _BRUSH_PARAMETER,
                    "Color": "#FFE08A2C",
                },
            },
        },
        "X": _still(0.0),
        "Y": _still(0.0),
        "Z": _still(0.0),
        "Opacity": _still(100.0),
        "Zoom": _still(100.0),
        "Rotation": _still(0.0),
        "FadeIn": 0.0,
        "FadeOut": 0.0,
        "Blend": "Normal",
        "IsInverted": False,
        "IsClippingWithObjectAbove": False,
        "IsAlwaysOnTop": False,
        "IsZOrderEnabled": False,
        "VideoEffects": [],
        "Group": 0,
        "Frame": frame,
        "Layer": layer,
        "KeyFrames": {"Frames": [], "Count": 0},
        "Length": length,
        "PlaybackRate": 100.0,
        "ContentOffset": "00:00:00",
        "Remark": "比較の下地",
        "IsLocked": False,
        "IsHidden": False,
    }


@dataclass
class Case:
    """比べるテンプレート 1 本"""

    name: str
    file: str
    index: int
    start: int
    length: int
    items: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def sample_frames(self, *, every: bool = False, clear_from: int = 0) -> list[int]:
        """比べるフレーム 既定は入りと真ん中と終わりの手前 ``every`` なら枠のすべて

        3 枚だけだと、場面切り替えの切れ目のように一瞬だけずれる所を見落とす
        ``clear_from`` より前は前のテンプレートの絵が残りうる（:func:`shadowed_until`）ので、
        残りの区間から選ぶ 枠のすべてが隠れていれば空
        """
        first = max(self.start, clear_from)
        end = self.start + self.length
        if first >= end:
            return []
        if every:
            return list(range(first, end))
        last = end - first - 1
        picks = {min(2, last), last // 2, max(0, last - 3)}
        return sorted(first + offset for offset in picks)


def _span(items: list[dict[str, Any]]) -> tuple[int, int]:
    starts = [int(number(item.get("Frame"), 0.0)) for item in items]
    ends = [
        int(number(item.get("Frame"), 0.0)) + max(1, int(number(item.get("Length"), 1.0)))
        for item in items
    ]
    return min(starts), max(ends)


def _lingers(value: Any) -> bool:
    """終わった後も絵が残るエフェクトを持つか グループや入れ子の中まで見る"""
    if isinstance(value, dict):
        if type_name(value) in LINGERING_EFFECTS and value.get("IsEnabled", True) is not False:
            return True
        return any(_lingers(inner) for inner in value.values())
    if isinstance(value, list):
        return any(_lingers(inner) for inner in value)
    return False


def reach(items: list[dict[str, Any]]) -> int:
    """アイテムの絵が届く最後の次のフレーム 残像なら終わった後に残る分（:data:`LINGER`）も足す

    残る分は残像を持つアイテムの終わりから数える 一番遅い終わりへ足すと、短い残像と
    長いふつうのアイテムが同居したとき、何も残らない所まで次の枠から外れる
    """
    return max(
        (_span([item])[1] + (LINGER if _lingers(item) else 0) for item in items),
        default=0,
    )


def shadowed_until(cases: list[dict[str, Any]]) -> list[int]:
    """枠ごとに、それより前のテンプレートの絵が残りうる最後の次のフレーム

    前に作った並び（空きが 6 フレームで、残像の後ろを広げていない物や、枠より長い
    アイテムを切り詰める前の物）で書き出した動画でも、YMM4 を起動し直さずに比べられる
    ように、写り込みうる所を比べる所から外す こちらはテンプレートを 1 本ずつ描くので
    前の絵は混ざらず、外さないと YMM4 の絵にだけ前のテンプレートが写って差が大きく出る
    ``cases`` は manifest.json の並びのまま受け、同じ並びで返す
    """
    order = sorted(range(len(cases)), key=lambda position: int(cases[position]["start"]))
    result = [0] * len(cases)
    reached = 0
    for position in order:
        result[position] = reached
        reached = max(reached, reach(cases[position]["items"]))
    return result


#: 書き出しの頭で先に 1 度描いておくエフェクト（#210 の 1）
#: YMM4 4.56.1.1 は、その書き出しで方向で色を抜く（DirectionalColorKey）を初めて描くのが
#: シュバッと演出素材（フレームバッファにリール回転・グループの帯）の中だと、書き出しを黙って止める
#: 頭に素の四角を置いても止まり（37 / 466 コマ）、同じ帯を先に単独で描いておくと最後まで書けた
#: （1833 / 1833 コマ） 並べ方によらず、その前に 1 度でも描いていれば止まらない
WARM_UP_EFFECTS = frozenset({"DirectionalColorKeyEffect"})
#: 準備に使う長さ（フレーム） 比べる枠の前に置き、比べない
WARM_UP_LENGTH = 6


def _warm_up_from(item_lists: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """準備のアイテムを作る 準備の要るエフェクトが無ければ空

    アイテムの ``VideoEffects`` にある、効いている（``IsEnabled`` が偽でない）準備の要る
    エフェクトを探す 見つけたアイテムを写し、そのエフェクトだけを残して頭に置く
    写したアイテムの形（図形とブラシ）はそのままなので、エフェクトが受け取る絵も同じ
    グループに付いていれば、グループだけでは絵が無いので下地の図形を 1 つ下に敷く
    枠をずらすかどうかもこの結果で決める 探し方を 2 つ持つと、準備が無いのに枠だけずれる
    """
    warm: list[dict[str, Any]] = []
    wanted = set(WARM_UP_EFFECTS)
    for items in item_lists:
        for item in items:
            effects = item.get("VideoEffects")
            if not isinstance(effects, list):
                continue
            found = [
                effect
                for effect in effects
                if isinstance(effect, dict)
                and type_name(effect) in wanted
                and effect.get("IsEnabled") is not False
            ]
            if not found:
                continue
            copied = copy.deepcopy(item)
            copied.update(
                {
                    "Frame": 0,
                    "Length": WARM_UP_LENGTH,
                    "Layer": 0,
                    "KeyFrames": {"Frames": [], "Count": 0},
                    "VideoEffects": copy.deepcopy(found),
                    "Remark": "比べる前の準備（#210）",
                }
            )
            warm.append(copied)
            if type_name(item) == "GroupItem":
                copied["GroupRange"] = 1
                warm.append(base_shape(0, 1, WARM_UP_LENGTH))
            wanted -= {type_name(effect) for effect in found}
            if not wanted:
                return warm
    return warm


def warm_up_items(cases: list[Case]) -> list[dict[str, Any]]:
    """比べる枠の前に置く準備のアイテム（:func:`_warm_up_from`）"""
    return _warm_up_from(case.items for case in cases)


def build_cases(files: list[Path]) -> tuple[list[Case], list[str]]:
    skipped: list[str] = []
    templates: list[tuple[str, int, str, list[dict[str, Any]]]] = []
    for path in files:
        for index, template in enumerate(load_template(path)):
            items = [copy.deepcopy(item) for item in template.items]
            kinds = {type_name(item) for item in items}
            if kinds & SKIPPED_ITEMS:
                skipped.append(f"{template.name}（{', '.join(sorted(kinds & SKIPPED_ITEMS))}）")
                continue
            templates.append((path.name, index, template.name, items))
    # 準備のアイテムを作れるときだけ、比べる枠を準備の分だけ後ろから始める
    start = WARM_UP_LENGTH + GAP if _warm_up_from(items for _, _, _, items in templates) else 0
    return place_cases(templates, start=start), skipped


def place_cases(
    templates: Iterable[tuple[str, int, str, list[dict[str, Any]]]], *, start: int = 0
) -> list[Case]:
    """テンプレート（ファイル名・番号・名前・アイテム）を時間をずらして並べる

    アイテムは並べた位置へ書き換える（呼ぶ側で写しておくこと） ``start`` は最初の枠の頭
    """
    cases: list[Case] = []
    cursor = start
    for file_name, index, name, items in templates:
        first, end = _span(items)
        length = max(MIN_SLOT, min(end - first, 300))
        top = min(int(number(item.get("Layer"), 0.0)) for item in items)
        for item in items:
            offset = int(number(item.get("Frame"), 0.0)) - first
            item["Frame"] = offset + cursor
            item["Layer"] = int(number(item.get("Layer"), 0.0)) - top
            # 枠より長いアイテムは枠の終わりで切る 切らないと次のテンプレートの枠へ
            # はみ出し、YMM4 の絵にだけ前のテンプレートが映り込む
            item_length = max(1, int(number(item.get("Length"), 1.0)))
            item["Length"] = max(1, min(item_length, length - offset))
        note = ""
        if all(type_name(item) == "TransitionItem" for item in items):
            # 場面切り替えだけのテンプレート 下に前の場面と後の場面を敷き、切れ目を真ん中に置く
            for item in items:
                item["Layer"] = int(item["Layer"]) + 1
            half = length // 2
            before = base_shape(cursor, 0, half)
            after = base_shape(cursor + half, 0, length - half)
            after["ShapeParameter"]["Brush"]["Parameter"]["Color"] = "#FF2C7AE0"
            after["X"] = _still(200.0)
            before["X"] = _still(-200.0)
            items[:0] = [before, after]
            note = "前後の場面の図形を敷いた"
        has_content = any(type_name(item) not in ("GroupItem",) for item in items)
        if not has_content:
            # エフェクトだけのテンプレート グループの範囲の中（1 つ下）に下地を置く
            # 別のグループがいる段は避ける 同じ段に重ねると YMM4 は下地を空いた段へ
            # ずらして描き、こちらは重ねたまま描くので、掛かるグループが食い違う
            # （オーラはグループが 0・1・3 段にあり、1 段目に置いた下地へ YMM4 は
            # 1 段目のグループのノイズを掛けていた）
            group = items[0]
            taken = {int(item.get("Layer", 0)) for item in items}
            below = int(group.get("Layer", 0)) + 1
            while below in taken:
                below += 1
            items.append(base_shape(cursor, below, length))
            for item in items:
                if type_name(item) == "GroupItem":
                    item["Length"] = length
            note = "下地の図形に着せた"
        cases.append(
            Case(
                name=name,
                file=file_name,
                index=index,
                start=cursor,
                length=length,
                items=items,
                note=note,
            )
        )
        # 残像のように終わった後も YMM4 の絵が残るテンプレートの後ろは、残る分だけ空ける
        # 空きが 6 フレームのままだと、次のテンプレートの頭に YMM4 の絵だけ前の文字が写る
        cursor = max(cursor + length, reach(items)) + GAP
    return cases


def write_project(
    cases: list[Case], target: Path, *, width: int = WIDTH, height: int = HEIGHT
) -> None:
    items = [*warm_up_items(cases), *(item for case in cases for item in case.items)]
    length = max((case.start + case.length for case in cases), default=1) + GAP
    write_document(items, length, target, width=width, height=height)


def write_document(
    items: list[dict[str, Any]],
    length: int,
    target: Path,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> None:
    """アイテムの並びを YMM4 のプロジェクト（.ymmp）として書く

    絵の比較と音の探りで同じ書き方を使う YMM4 は BOM 付きの UTF-8 でないと
    プロジェクトを開けない（``utf-8-sig``）
    """
    max_layer = max((int(item.get("Layer", 0)) for item in items), default=0)
    document = {
        "FilePath": str(target),
        "SelectedTimelineIndex": 0,
        "Timelines": [
            {
                "ID": str(uuid.uuid4()),
                "Name": "メイン",
                "VideoInfo": {"FPS": FPS, "Hz": 48000, "Width": width, "Height": height},
                "Items": items,
                "LayerSettings": {"Items": []},
                "CurrentFrame": 0,
                "Length": length,
                "MaxLayer": max_layer + 1,
            }
        ],
        "Characters": [],
        "CollapsedGroups": [],
    }
    target.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8-sig")


def command_build(arguments: argparse.Namespace) -> int:
    work: Path = arguments.work
    work.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for source in arguments.sources
        for path in (source.rglob("*.ymmt") if source.is_dir() else [source])
    )
    if arguments.exclude:
        # YMM4 自身が書き出しに失敗するテンプレートがある 外して並べ直す
        files = [path for path in files if not any(word in path.name for word in arguments.exclude)]
    files = move_last(files, getattr(arguments, "last", []))
    cases, skipped = build_cases(files)
    width, height = getattr(arguments, "size", (WIDTH, HEIGHT))
    write_project(cases, work / "compare.ymmp", width=width, height=height)
    manifest = {
        "width": width,
        "height": height,
        "fps": FPS,
        "skipped": skipped,
        "cases": [
            {
                "name": case.name,
                "file": case.file,
                "index": case.index,
                "start": case.start,
                "length": case.length,
                "note": case.note,
                "items": case.items,
            }
            for case in cases
        ],
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    total = max((case.start + case.length for case in cases), default=0)
    print(
        f"{len(cases)} 本を並べた（{total} フレーム、{total / FPS:.0f} 秒）"
        f" 飛ばした {len(skipped)} 本"
    )
    print(f"YMM4 で {work / 'compare.ymmp'} を開き、{work / 'ymm4.mp4'} へ書き出してください")
    print_export_hint(work / "compare.ymmp", work / "ymm4.mp4")
    return 0


def screen_size(text: str) -> tuple[int, int]:
    """``1280x720`` の形の画面の大きさ 4 で割り切れない大きさは縮めた比較が端を切るので断る"""
    width, _, height = text.lower().partition("x")
    try:
        size = (int(width), int(height))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text} は 幅x高さ の形でない") from None
    if min(size) <= 0 or any(value % SHRINK for value in size):
        raise argparse.ArgumentTypeError(f"{text} は {SHRINK} で割り切れる正の大きさにする")
    return size


def move_last(files: list[Path], words: list[str]) -> list[Path]:
    """名前に ``words`` のどれかを含むファイルを後ろへ回す 語の順に並べ、最後の語の物が最後

    YMM4 は書き出しの途中で黙って止まることがある（aomoya の書き出しは、最初のフレーム
    バッファのアイテムの頭の 15766 フレームで 2 度とも止まった #177） 止める物を後ろへ
    回せば、前の物は全部書き出され、どれで止まったかも止まった所で分かる
    """
    if not words:
        return files
    rest = [path for path in files if not any(word in path.name for word in words)]
    # 2 つの語に当たるファイルは後ろの語の所へ置く 2 度並べるとテンプレートが重なる
    tail: list[Path] = []
    for word in words:
        for path in files:
            if word in path.name:
                if path in tail:
                    tail.remove(path)
                tail.append(path)
    return rest + tail


def export_arguments(project: Path, video: Path, *, no_compressor: bool = False) -> list[str]:
    """``tools/ymm4_export.py`` へ渡す引数 案内の命令と試験で読ませる引数を 1 か所で作る

    パスは絶対にする 案内はリポジトリの根から走らせる形なので、作業フォルダが相対の
    ままだと、別の所で走らせた道具が違う .ymmp を開き、違う所へ書き出す
    """
    arguments = [
        r"tools\ymm4_export.py",
        "--project",
        str(project.resolve()),
        "--output",
        str(video.resolve()),
    ]
    if no_compressor:
        arguments.append("--no-compressor")
    return arguments


def print_export_hint(project: Path, video: Path, *, no_compressor: bool = False) -> None:
    """YMM4 で書き出す所を道具に任せる命令を並べる"""
    command = subprocess.list2cmdline(export_arguments(project, video, no_compressor=no_compressor))
    print("自動で書き出すなら（YMM4 がマウスとキーボードを取り合うので、本人に断ってから）")
    print(rf"  .venv\Scripts\python.exe {command}")


def _shrink(image: np.ndarray) -> np.ndarray:
    """面積の平均で縦横 4 分の 1 に縮める 1920x1080 なら 480x270（``COMPARE_WIDTH``）

    画面の大きさを変えた探り（``build --size``）も同じ割合で縮める 決まった 480x270 へ
    合わせると、1280x720 は割り切れずに右と下が切れる
    """
    height, width = image.shape[:2]
    rows, columns = height // SHRINK, width // SHRINK
    cropped = image[: rows * SHRINK, : columns * SHRINK, :3].astype(np.float32)
    return cropped.reshape(rows, SHRINK, columns, SHRINK, 3).mean(axis=(1, 3))


#: 縁と見なす明るさの段差（隣の画素との差 0〜255） 圧縮の揺れ（数段）は拾わず、
#: 文字の縁や細い線は拾う大きさ
EDGE_STEP = 24.0
#: 縁の差を平均するときの画素数の下限 縁がほとんど無い絵（黒の中の小さな光の粒など）で
#: 数画素の食い違いを平均すると、ほぼ同じ絵でも 46 のように大きく出て、本当に縁が違う枠と
#: 見分けられない（aomoya のキラリンエフェクトで縮めた平均 0.01 のとき 46.4 だった）
EDGE_MIN_PIXELS = 1000
#: 外れた画素と見なす差（3 色のうち一番大きい差 0〜255） 圧縮で縁が滲む分は下回る
OUTLIER_STEP = 48
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class Measure(NamedTuple):
    """1 枚の比べた数

    ``difference`` は 4 分の 1 に縮めた平均で、上限（:data:`CEILINGS`）はこれで見る
    細い縁の違いは平均に埋もれるので、縮めない絵の縁の差（``edge``）と、差の大きい
    画素の割合（``outliers`` 百分率）を別に出す どちらも上限には混ぜない（#200）
    """

    difference: float
    edge: float
    outliers: float


def _gradient(image: np.ndarray) -> np.ndarray:
    """明るさの、右と下の隣との差の大きい方 縁の強さの絵"""
    luma = image[..., :3].astype(np.float32) @ _LUMA
    gradient = np.zeros_like(luma)
    gradient[:, :-1] = np.abs(np.diff(luma, axis=1))
    np.maximum(gradient[:-1, :], np.abs(np.diff(luma, axis=0)), out=gradient[:-1, :])
    return gradient


def measure(reference: np.ndarray, ours: np.ndarray) -> tuple[Measure, np.ndarray, np.ndarray]:
    """YMM4 の絵とこちらの絵を比べた数と、縮めた 2 枚（並べた絵に使う）"""
    a, b = _shrink(reference), _shrink(ours)
    difference = float(np.abs(a - b).mean())
    ours_edges, reference_edges = _gradient(ours), _gradient(reference)
    near = (reference_edges > EDGE_STEP) | (ours_edges > EDGE_STEP)
    # 縁の両脇 1 画素まで広げる 縁が 1 画素ずれたとき、片方の縁の上だけを見ると
    # もう片方の縁が外れ、ずれが半分しか数に出ない
    grown = near.copy()
    grown[1:] |= near[:-1]
    grown[:-1] |= near[1:]
    grown[:, 1:] |= near[:, :-1]
    grown[:, :-1] |= near[:, 1:]
    # 縁の無い絵（黒一色など）でも 0 で割らない 下限の画素数で割る
    count = max(int(grown.sum()), EDGE_MIN_PIXELS)
    edge = float(np.abs(reference_edges - ours_edges)[grown].sum() / count)
    gap = np.abs(reference[..., :3].astype(np.int16) - ours[..., :3].astype(np.int16)).max(axis=2)
    outliers = float((gap > OUTLIER_STEP).mean() * 100.0)
    return Measure(difference, edge, outliers), a, b


def _save_png(image: np.ndarray, target: Path) -> None:
    """RGB の配列を PNG へ 画像のためだけに Pillow を足さず、入っている Qt で書く"""
    from PySide6.QtGui import QImage

    height, width = image.shape[:2]
    data = np.ascontiguousarray(image)
    QImage(data.data, width, height, width * 3, QImage.Format.Format_RGB888).save(str(target))


def frame_index(pts: int, start: int | None, time_base: Fraction, rate: float) -> int:
    """書き出した動画の 1 枚が、プロジェクトの何フレーム目かを返す

    YMM4 の書き出しは、最初の 1 枚の時刻が 0 ではなく 1 フレームぶん後ろ
    （``start_time`` が 1 フレーム）から始まる 時刻をそのままフレーム番号にすると、
    YMM4 の絵が 1 枚ずつ遅れて並び、動きのある所（場面の切れ目や動き出し）で
    差が 8〜24 跳ねる ストリームの頭の時刻を引いて、最初の 1 枚を 0 にそろえる
    """
    return round(float((pts - (start or 0)) * time_base) * rate)


#: 動画の 1 枚 フレーム番号と、絵を取り出す関数の組
#: 取り出す（RGB の配列へ変換する）のは比べる 1 枚だけ 3 枚だけ比べるときに、
#: そこまで読み進めた全部の絵を変換すると、それだけで数分かかる
Picture = tuple[int, "Callable[[], np.ndarray]"]


def _ymm4_frames(video: Path) -> Iterator[Picture]:
    """書き出した動画を頭から 1 枚ずつ返す"""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or FPS)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            # PyAV 19 から time_base の型が AVRational になった 分子と分母から Fraction にする
            index = frame_index(frame.pts, stream.start_time, as_fraction(frame.time_base), rate)
            yield index, partial(frame.to_ndarray, format="rgb24")


class References:
    """YMM4 の絵を、若い番号から順に 1 枚ずつ引く

    先に全部を読んで持っておくと、全フレームを比べたときに 1920x1080 の絵が
    1 万枚を超え、メモリに載らない 比べる順（番号の若い順）に動画を進めて読む
    """

    def __init__(self, frames: Iterator[Picture]) -> None:
        self._frames = frames
        self._current: Picture | None = None
        self._asked = -1

    def get(self, wanted: int) -> np.ndarray | None:
        """``wanted`` 番の絵 動画に無ければ ``None``

        前に引いた番号より若い番号を引くと例外にする 読み進めた動画は戻せないので、
        黙って ``None`` を返すと、呼ぶ側の並べ間違いが「比べる絵が無い」に化けて気付けない
        """
        if wanted < self._asked:
            raise ValueError(f"{self._asked} 番の後に {wanted} 番は引けない 若い順に引くこと")
        self._asked = wanted
        while self._current is None or self._current[0] < wanted:
            following = next(self._frames, None)
            if following is None:
                return None
            self._current = following
        return self._current[1]() if self._current[0] == wanted else None


class Row(NamedTuple):
    """比べた結果の 1 行 差・名前・ファイル・フレーム・並べた絵の名前・注・縁の差・外れた画素

    並べる順と上限は ``difference``（縮めた平均）だけで決める ``edge`` と ``outliers`` は
    一覧に並べる列（:class:`Measure`） ``--every`` では枠の中の最大
    """

    difference: float
    name: str
    file: str
    frame: int
    stem: str
    note: str
    edge: float = 0.0
    outliers: float = 0.0


#: テンプレートごとの差の上限 描き方の変更で YMM4 の絵から大きく離れたら気付けるように
#: 置く 値は 2026-09-24 の main（#168 の直し込み）で測った差に :data:`CEILING_MARGIN` を
#: 足したもの 良くなったら ``compare --write-ceilings`` で下げる
CEILINGS = ROOT / "tools" / "ymm4_compare_ceilings.json"
#: 上限に足すゆとり GPU や書体の違いで 1 前後は揺れる それより大きく動いたら
#: 描き方が変わったと見る（9-18 からの変化で一番小さい悪化が 3.6 だった）
CEILING_MARGIN = 3.0
#: テンプレートごとの縁の差（縮めない絵 :class:`Measure` の ``edge``）の上限（#205）
#: 縮めた平均は動きの位置が 1〜5 画素ずれても 0.5 ほどにしかならず、キャラクターの動きの
#: テンプレートのずれを見落とした 縁の差はずれると 40 を超えるので、ずれが戻れば気付ける
#: 持つのは動きを合わせたテンプレート（キャラクターの動き）だけ 乱数で揺らす物
#: （震え・呼吸の回転）は YMM4 の乱数を写せず、合わせても 40 前後に残る ここに書くのは
#: 今の値が大きく悪くならないための見張り ``--write-ceilings`` は書いてある物だけを書き換える
EDGE_CEILINGS = ROOT / "tools" / "ymm4_compare_edge_ceilings.json"
#: 縁の差の上限に足すゆとり 同じ絵でも圧縮の揺れで 5 ほど動く（動きを合わせた ぽよ登場 の
#: コマごとの縁の差が 5〜20 で揺れた）
EDGE_CEILING_MARGIN = 5.0


def worst_by_template(rows: list[Row]) -> dict[str, float]:
    """テンプレートごとに、比べたフレームのうち一番大きい差"""
    worst: dict[str, float] = {}
    for difference, name, *_ in rows:
        worst[name] = max(difference, worst.get(name, 0.0))
    return worst


def worst_edge_by_template(rows: list[Row]) -> dict[str, float]:
    """テンプレートごとに、比べたフレームのうち一番大きい縁の差"""
    worst: dict[str, float] = {}
    for row in rows:
        worst[row.name] = max(row.edge, worst.get(row.name, 0.0))
    return worst


def over_ceilings(
    worst: dict[str, float], ceilings: dict[str, float], label: str = "差"
) -> list[str]:
    """上限を超えたテンプレートを、超えた分の大きい順に並べる

    上限の無いテンプレート（あとから ``build`` に足したもの）は見ない 上限を持たない
    ものまで落とすと、テンプレートを足すたびに上限を書き足すまで道具が通らない
    """
    exceeded = [
        (difference - ceilings[name], name, difference)
        for name, difference in worst.items()
        if name in ceilings and difference > ceilings[name]
    ]
    return [
        f"{name} {label} {difference:.1f}（上限 {ceilings[name]:.1f}）"
        for _, name, difference in sorted(exceeded, reverse=True)
    ]


def unmeasured_templates(
    rows: list[Row],
    missing: list[tuple[str, int]],
    ceilings: dict[str, float],
    *,
    writing: bool = False,
) -> tuple[list[str], list[str]]:
    """書き出しに無くて比べられなかったフレームを、困る物と困らない物に分ける

    困る物（1 つ目）
    - 一部のフレームだけ比べたテンプレート 書き出しが途中で切れると前半だけが残り、
      上限を見れば後半を見ずに通り、書き換えれば半端な測りが上限になる
    - 上限があるのに 1 枚も比べられなかったテンプレート（比べるときだけ） 前は
      書き出しに届いていたのに見張れなくなっている 書き換えでは前の上限が残るので困らない

    困らない物（2 つ目）は、上限が無く 1 枚も比べなかったテンプレート 手元の aomoya の
    書き出しは 15766 フレームで切れていて、後ろの 18 本は一度も測っていない 上限を
    持たないので、見張りから外れていることは前と変わらない
    """
    measured = worst_by_template(rows)
    short: dict[str, int] = {}
    for name, _ in missing:
        short[name] = short.get(name, 0) + 1
    problems: list[str] = []
    unmeasured: list[str] = []
    for name, count in short.items():
        if name in measured:
            problems.append(f"{name} 一部だけ比べた（{count} 枚足りない）")
        elif name in ceilings and not writing:
            problems.append(f"{name} 上限があるのに 1 枚も比べられない")
        else:
            unmeasured.append(name)
    return problems, unmeasured


def ceilings_from(worst: dict[str, float], margin: float = CEILING_MARGIN) -> dict[str, float]:
    """測った差にゆとりを足した上限 0.5 刻みへ切り上げて、少しの揺れで書き換えない"""
    return {
        name: math.ceil((difference + margin) * 2.0) / 2.0 for name, difference in worst.items()
    }


def read_ceilings(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    # 形の崩れは ValueError にそろえる 呼ぶ側はそれだけを受けて案内を出すので、
    # null や配列が TypeError のまま抜けると、案内の代わりにトレースバックで止まる
    # JSON の読み違い（json.JSONDecodeError）も ValueError の仲間
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} が名前と上限の組になっていない")
    ceilings: dict[str, float] = {}
    for name, value in raw.items():
        # null や並びは float が TypeError を投げる ValueError にそろえないと、呼ぶ側が
        # 終了コード 1 で知らせずに落ちる 真偽値は float が 0 と 1 にしてしまうので断る
        # 文字列は "71" なら float が読めてしまうが、上限を文字で書くのは書き間違いなので断る
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path} の上限が数でない: {name}")
        try:
            ceilings[str(name)] = float(value)
        except OverflowError:
            # float に収まらない桁の整数 OverflowError は ValueError の仲間でない
            raise ValueError(f"{path} の上限が大きすぎる: {name}") from None
    # NaN や Infinity は float が受け取ってしまう どちらも「超えた」にならないので、
    # 入っていると差がいくら大きくても通る
    broken = [name for name, value in ceilings.items() if not math.isfinite(value)]
    if broken:
        raise ValueError(f"{path} の上限が数でない: {', '.join(broken)}")
    return ceilings


def write_ceilings(
    path: Path,
    worst: dict[str, float],
    *,
    margin: float = CEILING_MARGIN,
    only_listed: bool = False,
) -> None:
    """測った分だけ上限を書き換える ``--only`` で一部を測ったときに残りを消さない

    ``only_listed`` なら、ファイルに書いてあるテンプレートだけを書き換える 縁の差の
    上限（:data:`EDGE_CEILINGS`）は見張ると決めた物だけが持つので、測ったすべてを足すと
    乱数で揺らすテンプレートまで見張りに入る
    """
    current = read_ceilings(path)
    if only_listed:
        worst = {name: value for name, value in worst.items() if name in current}
    merged = current | ceilings_from(worst, margin)
    path.write_text(
        json.dumps(dict(sorted(merged.items())), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def command_compare(arguments: argparse.Namespace) -> int:
    work: Path = arguments.work
    output: Path = arguments.output or work
    if not (work / "ymm4.mp4").exists():
        print(f"{work / 'ymm4.mp4'} がありません YMM4 で書き出してから走らせてください")
        return 1
    # 上限のファイルが無いまま比べて通すと、打ち間違いでどのテンプレートも見張られない
    # 書き換えるときだけは、新しく作れるように無くてもよい
    if not arguments.write_ceilings and not arguments.ceilings.exists():
        print(f"{arguments.ceilings} がありません 作るなら --write-ceilings を付けてください")
        return 1
    # 縁の差の上限は既定のファイルなら無くてもよい 見張ると決めたテンプレートだけが持つ
    # （:data:`EDGE_CEILINGS`） --edge-ceilings で名指ししたのに無いのは打ち間違いなので断る
    # 空として通すと、縁の見張りが黙って外れる（#217 の指摘）
    edge_path: Path = arguments.edge_ceilings or EDGE_CEILINGS
    if arguments.edge_ceilings is not None and not edge_path.exists():
        print(f"{edge_path} がありません 縁の差の上限を見るファイルを確かめてください")
        return 1
    # 比べるのに 1 分ほど掛かる 壊れた上限は比べる前に知らせる
    try:
        ceilings = read_ceilings(arguments.ceilings)
        edge_ceilings = read_ceilings(edge_path)
    except ValueError as error:
        print(f"上限を読めない: {error}")
        return 1
    # 比べられなかった・隠れたテンプレートは、どちらかの上限を持つ物をすべて見る 縁の上限だけを
    # 持つ物を外すと、書き出しに届かなくても縁の見張りを黙って通る（#217 の指摘）
    watched = ceilings | edge_ceilings
    words = [word for word in arguments.only.split(",") if word]
    missing: list[tuple[str, int]] = []
    shadowed: list[str] = []
    rows = compare_work(
        work,
        output,
        only=words,
        every=arguments.every,
        blending=arguments.blending,
        missing=missing,
        shadowed=shadowed,
    )
    if not rows:
        # 0 枚のまま上限を見ると、何も比べていないのに「超えなかった」で通る
        print(
            "比べた絵が 1 枚もない --only の語か、書き出しと manifest.json の食い違いを見てください"
        )
        return 1
    print("差（縮めた平均 上限はこれで見る）の大きい順")
    for row in rows[: arguments.top]:
        print(
            f"{row.difference:6.1f}  縁 {row.edge:5.1f}  外れ {row.outliers:5.2f}%"
            f"  {row.name}  フレーム {row.frame}"
        )
    # 縮めた平均の順だけでは、縁だけが違う枠が下に埋もれて目に入らない
    print("\n縁の差の大きい順")
    for row in sorted(rows, key=lambda row: row.edge, reverse=True)[: arguments.top]:
        print(f"{row.edge:6.1f}  差 {row.difference:5.1f}  {row.name}  フレーム {row.frame}")

    problems, unmeasured = unmeasured_templates(
        rows, missing, watched, writing=arguments.write_ceilings
    )
    if unmeasured:
        print(f"\n書き出しが届いておらず比べなかったテンプレート（上限なし）: {len(unmeasured)} 本")
    if shadowed:
        print(f"\n前の枠の絵が残りうる所に隠れて比べなかったテンプレート: {len(shadowed)} 本")
        # 上限を持つ物が隠れたまま通すと、見張っているはずのテンプレートが外れても気付けない
        # 書き換えでは前の上限が残るので困らない
        if not arguments.write_ceilings:
            problems += [
                f"{name} 上限があるのに前の枠の絵に隠れて比べられない"
                for name in shadowed
                if name in watched
            ]
    if problems:
        print("\n比べられないフレームがあり、測りが足りない")
        for line in problems:
            print(f"  {line}")
        return 1

    worst = worst_by_template(rows)
    worst_edges = worst_edge_by_template(rows)
    if arguments.write_ceilings:
        write_ceilings(arguments.ceilings, worst)
        print(f"上限を書き換えた: {arguments.ceilings}")
        if edge_path.exists():
            write_ceilings(
                edge_path,
                worst_edges,
                margin=EDGE_CEILING_MARGIN,
                only_listed=True,
            )
            print(f"縁の差の上限を書き換えた: {edge_path}")
        return 0
    failed = False
    exceeded = over_ceilings(worst, ceilings)
    if exceeded:
        print(f"\n差の上限を超えた（{arguments.ceilings.name}）")
        for line in exceeded:
            print(f"  {line}")
        failed = True
    # 縁の差は縮めた平均と別に見る 平均が上限の中でも、動きがずれると縁だけが大きく出る
    edges_exceeded = over_ceilings(worst_edges, edge_ceilings, label="縁")
    if edges_exceeded:
        print(f"\n縁の差の上限を超えた（{edge_path.name}）")
        for line in edges_exceeded:
            print(f"  {line}")
        failed = True
    return 1 if failed else 0


def compare_work(
    work: Path,
    output: Path,
    *,
    only: list[str] | None = None,
    every: bool = False,
    blending: str = "srgb",
    missing: list[tuple[str, int]] | None = None,
    shadowed: list[str] | None = None,
) -> list[Row]:
    """``work`` の書き出しと Sashimono の絵を比べ、差の大きい順の行を返す

    一覧（report.html / report.json）と並べた絵は ``output`` へ書く 試験が
    手元の作業フォルダの一覧を書き換えないように、読む所と書く所を分けてある
    書き出しに無くて比べられなかったフレームは ``missing`` へ（名前・フレーム）で足す
    前の枠の絵が残りうる所（:func:`shadowed_until`）は比べず、枠のすべてが隠れた
    テンプレートは ``shadowed`` へ名前を足す
    """
    from sashimono.core.model import Project, ProjectSettings
    from sashimono.core.timebase import FrameRate
    from sashimono.engine.render import FrameRenderer

    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    video = work / "ymm4.mp4"

    # 隠れる所は、--only で外す前の並び全体から決める 外した後で決めると、比べない
    # テンプレートの残りが次の枠へ写り込むのを見落とす
    cases = list(zip(manifest["cases"], shadowed_until(manifest["cases"]), strict=True))
    if only:
        # 名前にどれかの語を含むものを比べる
        cases = [pair for pair in cases if any(word in pair[0]["name"] for word in only)]
    # 動画を頭から順に読むので、枠も頭から順に比べる
    cases = sorted(cases, key=lambda pair: int(pair[0]["start"]))
    references = References(_ymm4_frames(video))

    settings = ProjectSettings(
        width=int(manifest.get("width", WIDTH)),
        height=int(manifest.get("height", HEIGHT)),
        frame_rate=FrameRate(FPS),
        blending=blending,
    )
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    report = CompatibilityReport()
    renderer: FrameRenderer | None = None
    try:
        for raw, clear_from in cases:
            case = Case(**raw)
            frames = case.sample_frames(every=every, clear_from=clear_from)
            if not frames:
                if shadowed is not None:
                    shadowed.append(case.name)
                continue
            objects = map_template(case.items, report=report)
            project = Project.create(settings)
            for command in place(objects, project, at_frame=case.start):
                project = command.apply(project)
            if renderer is None:
                renderer = FrameRenderer(project)
            else:
                renderer.set_project(project)
            # 全フレームを比べるときは、枠ごとに差の一番大きい 1 枚だけを残す
            # 1 枚ずつ絵を書き出すと、77 本で 1 万枚を超える
            # 全フレームを比べるときの縁の差と外れた画素は枠の中の最大 差の一番大きい
            # 1 枚の値だけを残すと、縁だけが違うフレームを見落とす
            worst: tuple[float, int, np.ndarray, np.ndarray] | None = None
            top_edge = top_outliers = 0.0
            for frame in frames:
                reference = references.get(frame)
                if reference is None:
                    if missing is not None:
                        missing.append((case.name, frame))
                    continue
                ours = renderer.render(frame)
                measured, a, b = measure(reference, ours)
                if every:
                    top_edge = max(top_edge, measured.edge)
                    top_outliers = max(top_outliers, measured.outliers)
                    if worst is None or measured.difference > worst[0]:
                        worst = (measured.difference, frame, a, b)
                    continue
                rows.append(_saved_row(images, case, frame, measured, a, b))
            if worst is not None:
                measured = Measure(worst[0], top_edge, top_outliers)
                rows.append(_saved_row(images, case, worst[1], measured, worst[2], worst[3]))
    finally:
        if renderer is not None:
            renderer.close()

    rows.sort(reverse=True)
    write_reports(output, rows, manifest.get("skipped", []), report)
    return rows


def _saved_row(
    images: Path, case: Case, frame: int, measured: Measure, a: np.ndarray, b: np.ndarray
) -> Row:
    """並べた絵を書き出し、一覧の 1 行を返す"""
    stem = f"{case.start:06d}_{frame:06d}"
    side = np.concatenate([a, b, np.abs(a - b) * 3.0], axis=1)
    _save_png(np.clip(side, 0, 255).astype(np.uint8), images / f"{stem}.png")
    return Row(
        measured.difference,
        case.name,
        case.file,
        frame,
        stem,
        case.note,
        edge=measured.edge,
        outliers=measured.outliers,
    )


def write_reports(
    output: Path, rows: list[Row], skipped: list[str], report: CompatibilityReport
) -> None:
    """一覧を ``output`` の report.html（絵と並べる）と report.json（道具で読む）へ書く"""
    _write_report(output / "report.html", rows, skipped, report)
    (output / "report.json").write_text(
        json.dumps(
            [
                {
                    "difference": row.difference,
                    "edge": row.edge,
                    "outliers": row.outliers,
                    "name": row.name,
                    "file": row.file,
                    "frame": row.frame,
                    "image": row.stem,
                    "note": row.note,
                }
                for row in rows
            ],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def _write_report(
    target: Path,
    rows: list[Row],
    skipped: list[str],
    report: CompatibilityReport,
) -> None:
    body = [
        "<!doctype html><meta charset='utf-8'><title>YMM4 と Sashimono の比較</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#ddd}"
        "img{max-width:100%}td{vertical-align:top;padding:4px}</style>",
        "<h1>YMM4（左）と Sashimono（中）と差（右、3 倍）</h1>",
        f"<p>比べた絵 {len(rows)} 枚 差は {COMPARE_WIDTH}x{COMPARE_HEIGHT} に縮めた 0〜255 の"
        "平均（上限はこれで見る） 縁は縮めない絵の縁の近くで、明るさの段差が食い違った"
        f"大きさの平均 外れは 3 色のどれかの差が {OUTLIER_STEP} を超えた画素の割合"
        " 全フレームを比べたときの縁と外れは枠の中の最大</p>",
        "<table>",
    ]
    for row in rows:
        body.append(
            f"<tr><td>{row.difference:.1f}<br>縁 {row.edge:.1f}<br>外れ {row.outliers:.2f}%"
            f"<br>{html.escape(row.name)}<br>{html.escape(row.file)}"
            f"<br>フレーム {row.frame}<br>{html.escape(row.note)}</td>"
            f"<td><img src='images/{row.stem}.png'></td></tr>"
        )
    body.append("</table><h2>飛ばしたテンプレート</h2><ul>")
    body.extend(f"<li>{html.escape(line)}</li>" for line in skipped)
    body.append("</ul><h2>写すときの記録</h2><ul>")
    body.extend(f"<li>{html.escape(line)}</li>" for line in report.lines())
    body.append("</ul>")
    target.write_text("\n".join(body), encoding="utf-8")


#: 探りに使う正弦波 440Hz なら書き出しの圧縮（AAC）でもほとんど痩せない
TONE_HZ = 440.0
#: 正弦波の長さ（秒）
TONE_SECONDS = 2.0
#: 書き出しの標本化周波数 YMM4 のプロジェクトの ``Hz`` と合わせる
AUDIO_RATE = 48000
#: 枠 1 つの長さ（フレーム） 素材（2 秒）より長くして、``PlaybackRate`` を
#: 半分にしたときの 4 秒が枠の中へ収まるようにする 収まらないと、長さで
#: 速度を見分けられない
AUDIO_SLOT = 120
#: 枠と枠の間の無音（フレーム） 音は尾を引く（圧縮の先読みと後引き）ので、
#: 絵の ``GAP`` より広く取る 狭いと隣の枠の残りを測る
AUDIO_GAP = 30
#: 無音とみなす振幅 書き出しの圧縮が無音の所に乗せる雑音より上に置く
SILENCE = 2.0e-3
#: 鳴っている長さを数える窓（秒） 1 サンプルずつ見ると、波が 0 を横切る
#: 瞬間まで無音に数えてしまう
ENVELOPE_SECONDS = 0.005
#: 音量つまみを dB 目盛りと見たときの、0 のところの減衰（dB）
#: 予想値の列を出すためだけの仮定 実測がどちらに近いかは人が読んで決める
DB_SPAN = 60.0


#: 音の速さの変え方 YMM4 本体（4.56.1.1）の ``YukkuriMovieMaker.dll`` の
#: 列挙 ``YukkuriMovieMaker.Project.Items.PlaybackRateAudioProcessingMode`` は
#: ``Resampling = 0`` と ``Sola = 1`` の 2 つだけ（メタデータの表を読んだ）
#: 実物の書き出しはどれも ``Resampling`` で、``Sola`` は名前から高さを保って
#: 長さだけを変える変え方（WSOLA）と見当を付けた 測るまでは見当のまま
RESAMPLING = "Resampling"
SOLA = "Sola"
#: 新しい版の音声アイテムの枠（Issue #117） 前からある枠の後ろに並べる
#: ``(種類, PlaybackRate, PlaybackRate2, 音の変え方)``
#: 食い違いは絵の探りと同じ 2 組 音の長さと高さの両方がどちらかの値に合うかで読む
#: ``Sola`` は速さを 50 と 200 の両方で置く 高さを保つ変え方なら、どちらも 440Hz のまま
#: 長さだけが 4 秒と 1 秒になり、``Resampling`` の 220Hz・880Hz と見分けられる
AUDIO_NEWER_CONDITIONS: tuple[tuple[str, float, float, str], ...] = (
    ("rate2", 100.0, 50.0, RESAMPLING),
    ("rate2", 50.0, 100.0, RESAMPLING),
    ("mode", 50.0, 50.0, SOLA),
    ("mode", 200.0, 200.0, SOLA),
)


@dataclass(frozen=True)
class AudioSlot:
    """探りの枠 1 つ 条件 1 つぶんの音声アイテムに対応する

    ``playback_rate2`` が ``None`` なら前の版の形（``PlaybackRate`` だけ）で書く
    前からある枠はこの形で測ったので、形を変えると前の測り結果と比べられない
    """

    kind: str
    name: str
    start: int
    volume: float
    pan: float
    playback_rate: float
    playback_rate2: float | None = None
    mode: str | None = None


def build_audio_slots() -> list[AudioSlot]:
    """測る条件を、時間軸に重ならないように並べる

    先頭を基準（``Volume`` 100・``Pan`` 0・``PlaybackRate`` 100）にする
    ほかの枠は、この枠との比で読む

    ``Pan`` の値域は -100〜100 とした 配布物と手元のプロジェクトの音声アイテム
    125 個はすべて 0 で、実物からは決められない YMM4 本体（4.48.0.3）の IL を
    読むと、``Animation(0, -100, 100)`` を作る並び（``ldc.r8`` 3 つ）は
    ``YukkuriMovieMaker.dll`` に 18 か所あり、``Animation(0, -1, 1)`` は
    本体・Plugin・Community のどれにも 1 か所も無かった ``Volume`` が
    ``Animation(100, 0, 100)`` の百分率であることとも揃う
    もし値域が -1〜1 だったなら YMM4 が読み込みで丸めるので、-100 の枠と
    -50 の枠が同じ測り値になる 表がそうなっていたら、この判断が外れている
    """
    conditions: list[tuple[str, str, float, float, float]] = []
    # 基準を最初に置く 途中に置くと、読む人が表の 1 行目を基準と取り違える
    conditions.append(("基準", "Volume=100 Pan=0 Rate=100", 100.0, 0.0, 100.0))
    # 0 と 100 の間を細かく取る 振幅比なら 50 で 0.5、dB 目盛りなら 0.03 前後になり、
    # 2 点だけでは曲線の形（真ん中がどちらへ曲がるか）が読めない
    for volume in (0.0, 10.0, 25.0, 50.0, 75.0, 90.0):
        conditions.append(("volume", f"Volume={volume:g}", volume, 0.0, 100.0))
    for pan in (-100.0, -50.0, 50.0, 100.0):
        conditions.append(("pan", f"Pan={pan:g}", 100.0, pan, 100.0))
    for rate in (0.0, 50.0, 200.0):
        conditions.append(("rate", f"PlaybackRate={rate:g}", 100.0, 0.0, rate))
    slots: list[AudioSlot] = []
    cursor = 0
    for kind, name, volume, pan, rate in conditions:
        slots.append(
            AudioSlot(
                kind=kind, name=name, start=cursor, volume=volume, pan=pan, playback_rate=rate
            )
        )
        cursor += AUDIO_SLOT + AUDIO_GAP
    # 新しい版の形の枠は後ろへ足す 間に挟むと、前の測り結果と枠の番号と位置がずれる
    for kind, rate, rate2, mode in AUDIO_NEWER_CONDITIONS:
        name = f"PlaybackRate={rate:g} PlaybackRate2={rate2:g}"
        if mode != RESAMPLING:
            name += f" {mode}"
        slots.append(
            AudioSlot(
                kind=kind,
                name=name,
                start=cursor,
                volume=100.0,
                pan=0.0,
                playback_rate=rate,
                playback_rate2=rate2,
                mode=mode,
            )
        )
        cursor += AUDIO_SLOT + AUDIO_GAP
    return slots


def audio_item(slot: AudioSlot, media: Path) -> dict[str, Any]:
    """探りの枠 1 つを YMM4 の音声アイテムにする

    項目の並びと既定の値は、手元の YMM4 プロジェクト 16 本に入っていた
    ``AudioItem`` 125 個から写した（推測で書くと YMM4 が開けない）
    ``Volume`` と ``Pan`` は動く値、``PlaybackRate`` はただの数という食い違いも実物どおり

    ``playback_rate2`` を持つ枠は新しい版の形で書く 並びはこの機械のプロジェクトの
    新しい版の ``AudioItem`` 42 個から写した（``Pan`` の後に ``PlaybackRate2`` と
    ``PlaybackRateAudioProcessingMode``、``PlaybackRate`` は ``Length`` の後ろ）
    """
    if slot.playback_rate2 is not None:
        return _newer_audio_item(slot, media, slot.playback_rate2)
    return {
        "$type": "YukkuriMovieMaker.Project.Items.AudioItem, YukkuriMovieMaker",
        "IsWaveformEnabled": False,
        "FilePath": str(media),
        "AudioTrackIndex": 0,
        "Volume": _still(slot.volume),
        "Pan": _still(slot.pan),
        "PlaybackRate": slot.playback_rate,
        "ContentOffset": "00:00:00",
        "FadeIn": 0.0,
        "FadeOut": 0.0,
        "IsLooped": False,
        "EchoIsEnabled": False,
        "EchoInterval": 0.1,
        "EchoAttenuation": 40.0,
        "AudioEffects": [],
        "Group": 0,
        "Frame": slot.start,
        "Layer": 0,
        "KeyFrames": {"Frames": [], "Count": 0},
        "Length": AUDIO_SLOT,
        "Remark": slot.name,
        "IsLocked": False,
        "IsHidden": False,
    }


def _newer_audio_item(slot: AudioSlot, media: Path, rate2: float) -> dict[str, Any]:
    """新しい版の形の音声アイテム 並びは実物の新しい版の ``AudioItem`` そのまま"""
    return {
        "$type": "YukkuriMovieMaker.Project.Items.AudioItem, YukkuriMovieMaker",
        "IsWaveformEnabled": False,
        "FilePath": str(media),
        "AudioTrackIndex": 0,
        "Volume": _still(slot.volume),
        "Pan": _still(slot.pan),
        "PlaybackRate2": _still(rate2),
        "PlaybackRateAudioProcessingMode": slot.mode or RESAMPLING,
        "ContentOffset": "00:00:00",
        "FadeIn": 0.0,
        "FadeOut": 0.0,
        "IsLooped": False,
        "EchoIsEnabled": False,
        "EchoInterval": 0.1,
        "EchoAttenuation": 40.0,
        "AudioEffects": [],
        "Group": 0,
        "Frame": slot.start,
        "Layer": 0,
        "KeyFrames": {"Frames": [], "Count": 0},
        "Length": AUDIO_SLOT,
        "PlaybackRate": slot.playback_rate,
        "Remark": slot.name,
        "IsLocked": False,
        "IsHidden": False,
    }


def audio_manifest(slots: list[AudioSlot], media: Path) -> dict[str, Any]:
    """枠の一覧 ``audio-measure`` はこれだけを見て切り出す"""
    return {
        "fps": FPS,
        "rate": AUDIO_RATE,
        "slot": AUDIO_SLOT,
        "gap": AUDIO_GAP,
        "tone_hz": TONE_HZ,
        "tone_seconds": TONE_SECONDS,
        "media": str(media),
        "baseline": 0,
        "slots": [
            {
                "index": index,
                "kind": slot.kind,
                "name": slot.name,
                "start": slot.start,
                "length": AUDIO_SLOT,
                "volume": slot.volume,
                "pan": slot.pan,
                "playback_rate": slot.playback_rate,
                "playback_rate2": slot.playback_rate2,
                "mode": slot.mode,
            }
            for index, slot in enumerate(slots)
        ],
    }


def make_tone(target: Path) -> bool:
    """ffmpeg で正弦波を作る 作れなければ ``False``

    ``.wav``（PCM）にする 外の ffmpeg にどのエンコーダが入っているかに左右されない
    """
    if shutil.which("ffmpeg") is None:
        return False
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={TONE_HZ:g}:sample_rate={AUDIO_RATE}:duration={TONE_SECONDS:g}",
        # 左右で同じ波にする 定位の向きは、鳴らし分けた左右の比でしか読めない
        "-ac",
        "2",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    return completed.returncode == 0 and target.exists()


def command_audio_build(arguments: argparse.Namespace) -> int:
    # 絶対パスにしてから書く YMM4 はこの道具の作業フォルダを知らないので、
    # 相対のまま `.ymmp` へ書くと正弦波を見つけられず、全部の枠が無音になる
    # 無音の書き出しを測ると「再生速度 0 で止まる」「音量の比が 0」と読めてしまう
    work: Path = arguments.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    media = work / "audio-probe-tone.wav"
    if not make_tone(media):
        # 測れない環境で「壊れた」と読まれないように、落とさずに終える
        # （tools/bench_export.py と同じ作法）
        _drop_probe(work, "audio-probe", "audio-report.json")
        print("ffmpeg が無いか正弦波を作れないので、探りのプロジェクトは作れない")
        return 0
    slots = build_audio_slots()
    items = [audio_item(slot, media) for slot in slots]
    length = max(slot.start for slot in slots) + AUDIO_SLOT + AUDIO_GAP
    project = work / "audio-probe.ymmp"
    write_document(items, length, project)
    (work / "audio-probe.json").write_text(
        json.dumps(audio_manifest(slots, media), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # 前の測り結果は捨てる 残すと、新しい枠の一覧に対応しない表を読んでしまう
    (work / "audio-report.json").unlink(missing_ok=True)
    print(f"{len(slots)} 枠を並べた（{length} フレーム、{length / FPS:.0f} 秒）")
    video = work / "audio-probe.mp4"
    if video.exists():
        # 前の書き出しが残っていると、新しい枠の一覧で古い音を切り出してしまう
        # `audio-measure` は書き出しが一覧より古ければ止めるが、ここでも言っておく
        print(f"{video} は前の探りの書き出しです 作り直した方で書き出し直してください")
    print(f"YMM4 で {project} を開き、{video} として書き出してください")
    # 既定のコンプレッサー（自動）は音量の比を潰す 切って書き出さないと測れない
    print_export_hint(project, video, no_compressor=True)
    return 0


def _drop_probe(work: Path, stem: str, report: str) -> None:
    """作り直しに失敗したとき、前の探りの一覧・プロジェクト・測り結果を捨てる

    残すと、前に成功していた作業フォルダでは measure が古い一覧と古い書き出しの組を
    今回の物として測る 一覧が無ければ measure は「先に build」と案内して止まる
    書き出し（mp4）は本人が YMM4 で作った物なので残す
    """
    for name in (f"{stem}.json", f"{stem}.ymmp", report):
        (work / name).unlink(missing_ok=True)


def export_fps(video: Path) -> float | None:
    """書き出しの映像のフレームレート 映像の道が無いか読めなければ ``None``"""
    import av

    with av.open(str(video)) as container:
        if not container.streams.video:
            return None
        rate = container.streams.video[0].average_rate
        return None if rate is None else float(rate)


def _fps_differs(video: Path, fps: int) -> bool:
    """書き出しが一覧と違うフレームレートか 違えば案内する

    一覧の枠の番号は一覧の fps で数えている 書き出しを別の fps で数えた番号に
    そのまま当てると、60fps の書き出しでは枠が前半へずれ、等倍が 0.5 倍と出る
    """
    found = export_fps(video)
    if found is None or abs(found - fps) < 0.01:
        return False
    print(f"一覧は {fps}fps 書き出しは {found:g}fps です {fps}fps で書き出し直してください")
    return True


def sample_index(
    pts: int,
    start: int | None,
    time_base: Fraction,
    rate: int,
    start_base: Fraction | None = None,
) -> int:
    """書き出した音の一切れが、頭から何サンプル目かを返す :func:`frame_index` の音版

    YMM4 の書き出しは最初の一切れの時刻が 0 から始まらない 時刻をそのまま位置に
    すると、枠がまるごとずれて隣の枠の音を測る

    ``start`` はストリームの刻み（``start_base``）、``pts`` は並べ直したあとの
    刻み（``time_base``）で数える 刻みが違うまま引き算すると、頭の位置が
    まるごとずれて、枠に別の条件の音や無音が入る

    分数のまま掛ける 先に ``float`` へ落とすと、長い書き出しの終わりの方で
    丸めが 1 サンプルずれ、そこだけ隣の枠の音が混じる
    """
    seconds = pts * time_base - (start or 0) * (start_base if start_base is not None else time_base)
    return round(seconds * rate)


def unreadable_export_errors() -> tuple[type[Exception], ...]:
    """書き出しを開けない・読み切れないときに PyAV が投げる例外

    YMM4 が書き出している最中や中断した後の mp4 は ``moov`` がまだ無く、
    ``av.open`` が ``InvalidDataError``（``FFmpegError`` の一種）を投げる
    時刻の検査は通ってしまうので、捕まえないと案内の無い traceback で終わる
    ``probe_media`` と同じ組み合わせにする
    """
    import av.error

    return (av.error.FFmpegError, OSError)


def _explain_unreadable(video: Path, project: Path) -> None:
    print(f"{video} を読めません 書き出しが終わっていないか壊れています")
    print(f"YMM4 で {project} の書き出しを終えてから走らせてください")


def decode_audio(video: Path) -> tuple[np.ndarray, int] | None:
    """書き出した動画の音を ``(2, サンプル数)`` の配列と標本化周波数で返す

    頭の時刻のずれを詰めた位置へ置く 欠けた所は 0（無音）のまま残す

    音の道が無い書き出し（映像だけの形式で出した物）は ``None``
    ここで添字を取ると `IndexError` で終わり、測り方の案内を出せない
    """
    import av
    from av.audio.resampler import AudioResampler

    with av.open(str(video)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        rate = int(stream.rate or AUDIO_RATE)
        # 左右 2 本の浮動小数へそろえる 書き出しの形式（s16 か fltp か、
        # 何本の音か）で測り方が変わらないようにする
        resampler = AudioResampler(format="fltp", layout="stereo", rate=rate)
        pieces: list[tuple[int, np.ndarray]] = []
        # 1 枚ずつ流す 全部を先に list へ溜めると、長い書き出しで
        # 復号した音を 2 重に抱えることになる 末尾の None は残りを吐かせる合図
        for frame in itertools.chain(container.decode(stream), [None]):
            for converted in resampler.resample(frame):
                if converted.pts is None or converted.time_base is None:
                    continue
                at = sample_index(
                    converted.pts,
                    stream.start_time,
                    as_fraction(converted.time_base),
                    rate,
                    as_fraction(stream.time_base) if stream.time_base is not None else None,
                )
                pieces.append((at, np.asarray(converted.to_ndarray(), dtype=np.float32)))
    return assemble_audio(pieces), rate


def assemble_audio(pieces: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """``(置く位置, (2, n) の一切れ)`` を 1 本の ``(2, サンプル数)`` へ並べる

    頭より前に来る一切れ（圧縮の先読み分や編集リストの扱い）は、頭より前の分を
    捨ててから置く 負の位置のまま添字にすると末尾から数えられ、先頭の音が
    終わりへ書かれるか、長さが合わずに例外で落ちる 頭より前は測る対象ではない
    """
    total = max((at + block.shape[1] for at, block in pieces), default=0)
    if total <= 0:
        return np.zeros((2, 0), dtype=np.float32)
    buffer = np.zeros((2, total), dtype=np.float32)
    for at, block in pieces:
        if at < 0:
            block = block[:, -at:]
            at = 0
        if block.shape[1] == 0:
            continue
        buffer[:, at : at + block.shape[1]] = block[:, : total - at]
    return buffer


def slot_bounds(start: int, length: int, fps: int, rate: int) -> tuple[int, int]:
    """枠の始まりと終わりを、音のサンプルの番号で返す

    フレームからサンプルへ直すときに切り捨てると、枠の頭が 1 サンプル手前へ寄り、
    前の枠の尾を測る 端は必ず「フレーム番号 × レート ÷ FPS」の丸めでそろえる
    """
    return round(start * rate / fps), round((start + length) * rate / fps)


@dataclass(frozen=True)
class AudioMeasure:
    """枠 1 つの測り値 左右は ``(左, 右)`` の順"""

    rms: tuple[float, float]
    peak: tuple[float, float]
    seconds: float
    hz: float


def measure_block(block: np.ndarray, rate: int) -> AudioMeasure:
    """枠 1 つぶんの波形から、RMS・最大振幅・鳴っている長さ・中心の周波数を測る"""
    if block.shape[1] == 0:
        return AudioMeasure(rms=(0.0, 0.0), peak=(0.0, 0.0), seconds=0.0, hz=0.0)
    rms = tuple(float(np.sqrt(np.mean(np.square(row)))) for row in block[:2])
    peak = tuple(float(np.max(np.abs(row))) for row in block[:2])
    mono = np.mean(block[:2], axis=0)
    window = max(1, int(rate * ENVELOPE_SECONDS))
    usable = (mono.shape[0] // window) * window
    if usable:
        envelope = np.max(np.abs(mono[:usable]).reshape(-1, window), axis=1)
        seconds = float(np.count_nonzero(envelope > SILENCE) * window / rate)
    else:
        seconds = 0.0
    hz = 0.0
    if seconds > 0.0:
        spectrum = np.abs(np.fft.rfft(mono))
        hz = float(np.fft.rfftfreq(mono.shape[0], 1.0 / rate)[int(np.argmax(spectrum))])
    return AudioMeasure(rms=(rms[0], rms[1]), peak=(peak[0], peak[1]), seconds=seconds, hz=hz)


def ratio(value: float, base: float) -> float:
    """基準に対する比 基準が無音なら 0 を返す（0 で割らない）"""
    return value / base if base > 0.0 else 0.0


def volume_guesses(volume: float) -> dict[str, float]:
    """音量つまみの読み方ごとの、基準に対する振幅の予想値"""
    share = volume / 100.0
    return {
        "振幅比": share,
        "二乗": share * share,
        "dB目盛り": 10.0 ** ((volume - 100.0) * DB_SPAN / 100.0 / 20.0),
    }


def command_audio_measure(arguments: argparse.Namespace) -> int:
    work: Path = arguments.work
    manifest_path = work / "audio-probe.json"
    if not manifest_path.exists():
        print(f"{manifest_path} がありません 先に audio-build を走らせてください")
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 測れなかった道で前の表が残ると、新しい結果として開けてしまう
    # 測れたときは最後に書き直すので、先に消しておけばどの道でも残らない
    (work / "audio-report.json").unlink(missing_ok=True)
    video = work / "audio-probe.mp4"
    if not video.exists():
        print(f"{video} がまだ書き出されていません")
        print(f"YMM4 で {work / 'audio-probe.ymmp'} を開き、そこへ書き出してから走らせてください")
        return 0

    if video.stat().st_mtime <= manifest_path.stat().st_mtime:
        # 探りを作り直したのに書き出しが前のままだと、新しい枠の一覧で
        # 古い音を切り出して、まるで別の条件を測ったような表が出る
        # 同じ時刻も断る 置き場によっては時刻が 2 秒刻みでしか残らず、
        # 同じ時刻は「あとで書き出した」証しにならない
        print(f"{video} は探りを作り直す前の書き出しです")
        print(f"YMM4 で {work / 'audio-probe.ymmp'} を開き直し、書き出してから走らせてください")
        return 0

    try:
        decoded = decode_audio(video)
    except unreadable_export_errors():
        _explain_unreadable(video, work / "audio-probe.ymmp")
        return 0
    if decoded is None:
        print(f"{video} に音の道がありません 音が入る形式で書き出してください")
        return 0
    samples, rate = decoded
    if samples.shape[1] == 0:
        # 音の道はあるのに中身が無い書き出し 測ると全部の枠が 0 秒・比 0 になり、
        # 「再生速度 0 で止まる」と読み違える
        print(f"{video} の音が空です 音が入る形式で書き出してください")
        return 0
    # 音は書き出しのフレームレートに左右されない 枠の頭は一覧の fps で秒へ直し、
    # 書き出しの音は自分の時刻で並べるので、60fps で書き出しても同じ秒の音を切り出す
    fps = int(manifest.get("fps", FPS))
    measured: list[tuple[dict[str, Any], AudioMeasure]] = []
    for entry in manifest["slots"]:
        begin, end = slot_bounds(int(entry["start"]), int(entry["length"]), fps, rate)
        measured.append((entry, measure_block(samples[:, begin:end], rate)))
    base = measured[int(manifest.get("baseline", 0))][1]
    rows = [_audio_row(entry, value, base) for entry, value in measured]
    _print_audio_rows(rows)
    (work / "audio-report.json").write_text(
        json.dumps(
            {"rate": rate, "samples": int(samples.shape[1]), "rows": rows},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{work / 'audio-report.json'} へ書いた")
    return 0


def _audio_row(entry: dict[str, Any], value: AudioMeasure, base: AudioMeasure) -> dict[str, Any]:
    both = (value.rms[0] + value.rms[1]) / 2.0
    base_both = (base.rms[0] + base.rms[1]) / 2.0
    row = {
        "index": entry["index"],
        "kind": entry["kind"],
        "name": entry["name"],
        "volume": entry["volume"],
        "pan": entry["pan"],
        "playback_rate": entry["playback_rate"],
        "rms_left": value.rms[0],
        "rms_right": value.rms[1],
        "peak_left": value.peak[0],
        "peak_right": value.peak[1],
        "seconds": value.seconds,
        "hz": value.hz,
        "left_ratio": ratio(value.rms[0], base.rms[0]),
        "right_ratio": ratio(value.rms[1], base.rms[1]),
        "both_ratio": ratio(both, base_both),
        "seconds_ratio": ratio(value.seconds, base.seconds),
    }
    if entry["kind"] in ("volume", "基準"):
        row["guesses"] = volume_guesses(float(entry["volume"]))
    expectations = audio_expectations(entry)
    if expectations:
        row["playback_rate2"] = entry.get("playback_rate2")
        row["mode"] = entry.get("mode")
        row["expected"] = expectations
        row["nearer"], row["gaps"] = nearer_sound(value, expectations)
    return row


#: 近い方の予想でも、長さと高さの比の対数の差の和がこれを超えたら、どちらとも合わないと読む
#: 速さ 50 と 100 の差は長さと高さでそれぞれ 0.69 ある 前に測った 50 の長さ
#: （3.98 秒、予想の 4 秒と 0.005 の差）のような揺れよりは十分に大きく取る
AUDIO_FAR = 0.15


def sound_guess(rate: float, *, keep_pitch: bool = False) -> dict[str, float]:
    """速さ ``rate``（百分率）が効いたときの、鳴っている長さと中心の周波数の予想

    長さは枠（``AUDIO_SLOT``）で頭打ちになる 前に測った ``Resampling`` の作り
    （テープのように高さも変わる）を既定に、``keep_pitch`` なら高さを保つ変え方と読む
    """
    share = rate / 100.0
    slot_seconds = AUDIO_SLOT / FPS
    seconds = min(TONE_SECONDS / share, slot_seconds) if share > 0.0 else 0.0
    return {"seconds": seconds, "hz": TONE_HZ if keep_pitch else TONE_HZ * share}


def audio_expectations(entry: dict[str, Any]) -> dict[str, dict[str, float]]:
    """食い違わせた枠と音の変え方を変えた枠の、読み方ごとの予想 ほかの枠は空"""
    kind = entry.get("kind")
    rate = float(entry["playback_rate"])
    if kind == "rate2":
        return {
            "PlaybackRate": sound_guess(rate),
            "PlaybackRate2": sound_guess(float(entry["playback_rate2"])),
        }
    if kind == "mode":
        return {
            "高さも変わる": sound_guess(rate),
            "高さを保つ": sound_guess(rate, keep_pitch=True),
        }
    return {}


def nearer_sound(
    value: AudioMeasure, expectations: dict[str, dict[str, float]]
) -> tuple[str, dict[str, float]]:
    """測った長さと高さが、どの予想に近いか 比の対数で差を取る

    差をそのまま足すと、Hz（数百）が秒（数秒）を飲み込み、長さが合っているかを見なくなる
    比の対数なら、倍と半分が同じ重さになる
    """
    if value.seconds <= 0.0 or value.hz <= 0.0:
        return "鳴らない", {}
    gaps: dict[str, float] = {}
    for name, guess in expectations.items():
        if guess["seconds"] <= 0.0:
            gaps[name] = math.inf
            continue
        gaps[name] = abs(math.log(value.seconds / guess["seconds"])) + abs(
            math.log(value.hz / guess["hz"])
        )
    best = min(gaps, key=lambda name: gaps[name])
    if gaps[best] > AUDIO_FAR:
        return "どちらとも合わない", gaps
    return best, gaps


def _print_audio_rows(rows: list[dict[str, Any]]) -> None:
    # RMS は枠まるごとの平均 鳴っている長さが違う枠どうしを比べるときは、
    # 長さの列と合わせて読む（速度を半分にすると、枠を鳴り通して比が上がる）
    print(f"{'枠':<26}{'左比':>8}{'右比':>8}{'平均比':>8}{'長さ秒':>8}{'中心Hz':>8}")
    for row in rows:
        print(
            f"{row['name']:<26}{row['left_ratio']:>8.3f}{row['right_ratio']:>8.3f}"
            f"{row['both_ratio']:>8.3f}{row['seconds']:>8.2f}{row['hz']:>8.0f}"
        )
    volumes = [row for row in rows if "guesses" in row]
    if volumes:
        print()
        print("音量つまみの読み方くらべ（平均比が、どの予想値に近いか）")
        print(f"{'枠':<26}{'実測':>8}{'振幅比':>8}{'二乗':>8}{'dB目盛り':>10}")
        for row in volumes:
            guesses = row["guesses"]
            print(
                f"{row['name']:<26}{row['both_ratio']:>8.3f}{guesses['振幅比']:>8.3f}"
                f"{guesses['二乗']:>8.3f}{guesses['dB目盛り']:>10.4f}"
            )
    pans = [row for row in rows if row["kind"] == "pan"]
    if pans:
        print()
        print("定位の向き 左比 > 右比 なら、その値は左へ寄せる向き")
    rates = [row for row in rows if row["kind"] == "rate"]
    if rates:
        print()
        print("再生速度 長さが 0 秒なら止まる、基準と同じ長さなら等倍として扱われている")
        print("         中心Hz が基準の半分・倍なら、速度は音の高さごと変えている")
    newer = [row for row in rows if "expected" in row]
    if newer:
        print()
        print("新しい版の形（Issue #117） 長さと高さが、どちらの予想に近いか")
        print(f"{'枠':<40}{'長さ秒':>8}{'中心Hz':>8}  予想（長さ秒 / Hz）")
        for row in newer:
            guesses = "  ".join(
                f"{name} {guess['seconds']:.2f}/{guess['hz']:.0f}"
                for name, guess in row["expected"].items()
            )
            print(f"{row['name']:<40}{row['seconds']:>8.2f}{row['hz']:>8.0f}  {guesses}")
            print(f"{'':<40}→ {row['nearer']}")
        print()
        print("PlaybackRate と PlaybackRate2 の枠は、近い方の値を YMM4 が読んでいる")
        print("Sola の枠は、高さを保つなら中心Hz が 440 のまま長さだけが変わる")


#: 探りの下地の画像の大きさ 画面とちょうど同じにする
#: 画面より小さい画像は、Sashimono は画面いっぱいへ広げ、YMM4 は原寸で置くので、
#: 格子の点の画面の位置が両者で食い違い、予想の点との距離が比べられなくなる
MESH_IMAGE_WIDTH, MESH_IMAGE_HEIGHT = WIDTH, HEIGHT
#: 下地の市松の 1 マス 格子の区切り（3x3 で 960x540、5x5 で 480x270）より
#: ずっと細かくして、どのセルが歪んだかがマスの崩れで見えるようにする
MESH_CELL = 60
#: 枠 1 つの長さ（フレーム） 絵は止まっているので、真ん中の 1 枚だけを見る
MESH_SLOT = 30
#: 動かす点のずれ YMM4 の値のまま（px、Y は下が正）
#: 5x5 のセル（480x270）の 4 分の 1 前後にして、隣のセルまで崩れが広がらないようにする
#: 右と下へ動かすので、上の辺と左の辺の点を動かすと画面の内側へ入り、
#: 抜けた所（透明）が黒く出る 外へ動かすと画面の外で起きて、画素に何も残らない
MESH_SHIFT = (120.0, 60.0)
#: 変化とみなす差（0〜255、色の平均） 書き出しの圧縮が止まった絵に乗せる揺れより上に置く
#: 低いと、動かしていない所の揺れまで重心に入り、重心が画面の真ん中へ寄る
MESH_THRESHOLD = 24.0


@dataclass(frozen=True)
class MeshSlot:
    """探りの枠 1 つ ``moved`` は動かす点の番号（``Points`` の添字） ``None`` なら基準"""

    name: str
    columns: int
    rows: int
    moved: int | None
    start: int


def build_mesh_slots() -> list[MeshSlot]:
    """確かめる条件を、時間軸に重ならないように並べる

    格子の大きさごとに、動かさない枠（基準）を先に置く 変化の場所は基準との差で
    読むので、同じ格子の基準が要る 3x3 の基準で 5x5 を読むと、何も動かさなくても
    格子の分け方の違い（補間の継ぎ目）が差に出るかもしれない

    1 番と 3 番を分けて動かすのは、行ごとと列ごとを見分けるため 真ん中（4 番、
    5x5 の 12 番）は、どちらの並びでも真ん中に出るので見分けられない
    5x5 の 7 番は、行ごとなら上寄りの真ん中、列ごとなら左寄りの真ん中に出る
    """
    conditions: list[tuple[str, int, int, int | None]] = [
        ("基準 3x3", 3, 3, None),
        ("3x3 の 4 番", 3, 3, 4),
        ("3x3 の 1 番", 3, 3, 1),
        ("3x3 の 3 番", 3, 3, 3),
        ("基準 5x5", 5, 5, None),
        ("5x5 の 12 番", 5, 5, 12),
        ("5x5 の 7 番", 5, 5, 7),
    ]
    slots: list[MeshSlot] = []
    cursor = 0
    for name, columns, rows, moved in conditions:
        slots.append(MeshSlot(name=name, columns=columns, rows=rows, moved=moved, start=cursor))
        cursor += MESH_SLOT + GAP
    return slots


def mesh_pattern() -> np.ndarray:
    """位置が読み取れる下地 ``(高さ, 幅, 3)`` の RGB

    赤は左から右、緑は上から下へ明るくなり、青は市松で入れ替わる
    一色の図形では、内側の点を動かしても輪郭が変わらず、画素に何も出ない
    """
    ys, xs = np.mgrid[0:MESH_IMAGE_HEIGHT, 0:MESH_IMAGE_WIDTH]
    image = np.empty((MESH_IMAGE_HEIGHT, MESH_IMAGE_WIDTH, 3), dtype=np.uint8)
    image[..., 0] = (xs * 255 // (MESH_IMAGE_WIDTH - 1)).astype(np.uint8)
    image[..., 1] = (ys * 255 // (MESH_IMAGE_HEIGHT - 1)).astype(np.uint8)
    image[..., 2] = np.where((xs // MESH_CELL + ys // MESH_CELL) % 2 == 0, 255, 0)
    return image


def mesh_point_entry(x: float, y: float, *, selected: bool) -> dict[str, Any]:
    """``Points`` の 1 点 ``X`` と ``Y`` は動く値、``IsSelected`` は UI の選び状態

    ``.work/probes/samples.json`` の実物の形そのまま
    """
    return {"X": _still(x), "Y": _still(y), "IsSelected": selected}


def mesh_effect_entry(slot: MeshSlot) -> dict[str, Any]:
    """探りの枠の ``MeshDeformationEffect`` 動かすのは ``moved`` 番の点だけ

    実物は最初の点だけが ``IsSelected`` 真なので、それに合わせる
    """
    points = [
        mesh_point_entry(0.0, 0.0, selected=index == 0) for index in range(slot.columns * slot.rows)
    ]
    if slot.moved is not None:
        points[slot.moved] = mesh_point_entry(*MESH_SHIFT, selected=slot.moved == 0)
    return {
        "$type": "YukkuriMovieMaker.Project.Effects.MeshDeformationEffect, YukkuriMovieMaker",
        "HorizontalCount": slot.columns,
        "VerticalCount": slot.rows,
        "Points": points,
        "IsEnabled": True,
        "Remark": "",
    }


def image_item(
    media: Path,
    *,
    frame: int,
    length: int,
    remark: str,
    layer: int = 0,
    zoom: float = 100.0,
    effects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """探りの画像アイテム 1 つ

    項目の並びは ``.work/probes/samples.json`` の実物の ``ImageItem`` から写した
    """
    return {
        "$type": "YukkuriMovieMaker.Project.Items.ImageItem, YukkuriMovieMaker",
        "FilePath": str(media),
        "X": _still(0.0),
        "Y": _still(0.0),
        "Z": _still(0.0),
        "Opacity": _still(100.0),
        "Zoom": _still(zoom),
        "Rotation": _still(0.0),
        "FadeIn": 0.0,
        "FadeOut": 0.0,
        "Blend": "Normal",
        "IsInverted": False,
        "IsClippingWithObjectAbove": False,
        "IsAlwaysOnTop": False,
        "IsZOrderEnabled": False,
        "VideoEffects": list(effects or []),
        "Group": 0,
        "Frame": frame,
        "Layer": layer,
        "KeyFrames": {"Frames": [], "Count": 0},
        "Length": length,
        "PlaybackRate": 100.0,
        "PlaybackRate2": _still(100.0),
        "ContentOffset": "00:00:00",
        "Remark": remark,
        "IsLocked": False,
        "IsHidden": False,
    }


def mesh_image_item(slot: MeshSlot, media: Path) -> dict[str, Any]:
    """探りの枠 1 つを、格子で歪ませた画像アイテムにする"""
    return image_item(
        media,
        frame=slot.start,
        length=MESH_SLOT,
        remark=slot.name,
        effects=[mesh_effect_entry(slot)],
    )


def grid_position(index: int, columns: int, rows: int, *, by_row: bool) -> tuple[float, float]:
    """``Points`` の ``index`` 番が、動かす前に画面のどこにあるか（px、Y は下が正）

    ``by_row`` が真なら左上から行ごと、偽なら左上から列ごとの並びと読む
    下地は画面の真ん中に置くので、格子の四隅は画像の四隅
    """
    if by_row:
        column, row = index % columns, index // columns
    else:
        column, row = index // rows, index % rows
    left = (WIDTH - MESH_IMAGE_WIDTH) / 2.0
    top = (HEIGHT - MESH_IMAGE_HEIGHT) / 2.0
    return (
        left + MESH_IMAGE_WIDTH * column / (columns - 1),
        top + MESH_IMAGE_HEIGHT * row / (rows - 1),
    )


def mesh_manifest(slots: list[MeshSlot], media: Path) -> dict[str, Any]:
    """枠の一覧 ``mesh-measure`` はこれだけを見て切り出し、Sashimono でも描く"""
    baselines = {
        (slot.columns, slot.rows): index
        for index, slot in reversed(list(enumerate(slots)))
        if slot.moved is None
    }
    entries: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        entry: dict[str, Any] = {
            "index": index,
            "name": slot.name,
            "start": slot.start,
            "length": MESH_SLOT,
            "columns": slot.columns,
            "rows": slot.rows,
            "moved": slot.moved,
            "baseline": baselines[(slot.columns, slot.rows)],
            "item": mesh_image_item(slot, media),
        }
        if slot.moved is not None:
            entry["by_row"] = grid_position(slot.moved, slot.columns, slot.rows, by_row=True)
            entry["by_column"] = grid_position(slot.moved, slot.columns, slot.rows, by_row=False)
        entries.append(entry)
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "media": str(media),
        "image": [MESH_IMAGE_WIDTH, MESH_IMAGE_HEIGHT],
        "shift": list(MESH_SHIFT),
        "threshold": MESH_THRESHOLD,
        "slots": entries,
    }


def _clear_mesh_results(work: Path) -> None:
    """前の測り結果を捨てる 残すと、新しい枠の一覧に対応しない表や絵を読んでしまう"""
    (work / "mesh-report.json").unlink(missing_ok=True)
    images = work / "images"
    if images.is_dir():
        for old in images.glob("mesh-*.png"):
            old.unlink()


def command_mesh_build(arguments: argparse.Namespace) -> int:
    # 絶対パスにしてから書く YMM4 はこの道具の作業フォルダを知らないので、
    # 相対のまま `.ymmp` へ書くと下地を見つけられず、全部の枠が空になる
    # 空の書き出しを測ると「どの点を動かしても何も変わらない」と読めてしまう
    work: Path = arguments.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    media = work / "mesh-probe-grid.png"
    _save_png(mesh_pattern(), media)
    slots = build_mesh_slots()
    items = [mesh_image_item(slot, media) for slot in slots]
    length = max(slot.start for slot in slots) + MESH_SLOT + GAP
    project = work / "mesh-probe.ymmp"
    write_document(items, length, project)
    (work / "mesh-probe.json").write_text(
        json.dumps(mesh_manifest(slots, media), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _clear_mesh_results(work)
    print(f"{len(slots)} 枠を並べた（{length} フレーム、{length / FPS:.1f} 秒）")
    video = work / "mesh-probe.mp4"
    if video.exists():
        # 前の書き出しが残っていると、新しい枠の一覧で古い絵を切り出してしまう
        # `mesh-measure` は書き出しが一覧より古ければ止めるが、ここでも言っておく
        print(f"{video} は前の探りの書き出しです 作り直した方で書き出し直してください")
    print(f"YMM4 で {project} を開き、{video} として書き出してください")
    print(f"書き出しは {WIDTH}x{HEIGHT}・{FPS}fps・頭から終わりまで（範囲を絞らない）")
    print_export_hint(project, video)
    return 0


def change_centroid(
    picture: np.ndarray, baseline: np.ndarray, threshold: float = MESH_THRESHOLD
) -> tuple[float, float, int] | None:
    """基準との差が ``threshold`` を超えた所の重心を、画面の px（``WIDTH`` 基準）で返す

    返すのは ``(x, y, 変化した画素の数)`` 何も変わっていなければ ``None``
    差の大きさで重みを付ける 数だけで数えると、圧縮の揺れで閾値を少しだけ超えた
    画素が、崩れの芯と同じ重さで重心を引っ張る
    """
    difference = np.abs(picture[..., :3].astype(np.float32) - baseline[..., :3]).mean(axis=2)
    changed = difference > threshold
    count = int(np.count_nonzero(changed))
    if count == 0:
        return None
    weight = np.where(changed, difference, 0.0)
    ys, xs = np.mgrid[0 : difference.shape[0], 0 : difference.shape[1]]
    total = float(weight.sum())
    # 画素の真ん中を位置とする 左上の角で数えると、縮めた絵で半画素ぶん左上へ寄る
    scale_x = WIDTH / difference.shape[1]
    scale_y = HEIGHT / difference.shape[0]
    x = (float((weight * xs).sum()) / total + 0.5) * scale_x
    y = (float((weight * ys).sum()) / total + 0.5) * scale_y
    return x, y, count


def mesh_reading(
    centroid: tuple[float, float] | None,
    by_row: tuple[float, float] | None,
    by_column: tuple[float, float] | None,
) -> str:
    """重心が、行ごとと列ごとのどちらの読みの点に近いか"""
    if by_row is None or by_column is None:
        return "基準"
    if centroid is None:
        return "変化なし"
    if by_row == by_column:
        # 真ん中の点はどちらの並びでも同じ所 重心がそこへ出たかだけを見る
        return "見分けない"
    near_row = float(np.hypot(centroid[0] - by_row[0], centroid[1] - by_row[1]))
    near_column = float(np.hypot(centroid[0] - by_column[0], centroid[1] - by_column[1]))
    if near_row == near_column:
        return "見分けられない"
    return "行ごと" if near_row < near_column else "列ごと"


def has_video_stream(video: Path) -> bool:
    """映像の道があるか 音だけの形式で書き出した物を、添字の例外で落とさずに断る"""
    import av

    with av.open(str(video)) as container:
        return bool(container.streams.video)


def _probe_or_none(path: Path) -> MediaItem | None:
    from sashimono.engine.decode import ProbeError, probe_media

    try:
        return probe_media(path)
    except ProbeError:
        return None


def _render_mesh_slots(entries: list[dict[str, Any]]) -> list[np.ndarray | None]:
    """同じ ``.ymmp`` の枠を Sashimono で描き、縮めた絵を枠の順に返す"""
    return _render_still_slots(entries)


def _render_still_slots(
    entries: list[dict[str, Any]],
    *,
    convert: Callable[[np.ndarray], np.ndarray] = _shrink,
    adjust: Callable[[list[MappedObject]], list[MappedObject]] | None = None,
) -> list[np.ndarray | None]:
    """止まった絵の枠を Sashimono で描き、真ん中の 1 枚を ``convert`` した物を枠の順に返す

    枠のアイテムは ``items``（並び）か ``item``（1 つ） ``adjust`` は写した結果を描く前に
    差し替える 同じ枠を 2 通りの読み方で描き比べるため

    ``compare`` と同じ道（写す・素材を登録する・置く・描く）を通す 素材の登録を
    飛ばすと、画像のクリップが ``media_id`` を持たず、全部の枠が透明になる
    """
    from sashimono.compat.catalog import gather_media
    from sashimono.core.model import Project, ProjectSettings
    from sashimono.core.timebase import FrameRate
    from sashimono.engine.render import FrameRenderer

    settings = ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(FPS))
    report = CompatibilityReport()
    pictures: list[np.ndarray | None] = []
    renderer: FrameRenderer | None = None
    try:
        for entry in entries:
            raw = entry["items"] if "items" in entry else [entry["item"]]
            objects = map_template(copy.deepcopy(raw), report=report)
            if adjust is not None:
                objects = adjust(objects)
            project = Project.create(settings)
            plan = gather_media(objects, project, _probe_or_none)
            if plan.missing:
                pictures.append(None)
                continue
            start = int(entry["start"])
            commands = [*plan.commands, *place(objects, project, at_frame=start, media=plan.media)]
            for command in commands:
                project = command.apply(project)
            if renderer is None:
                renderer = FrameRenderer(project)
            else:
                renderer.set_project(project)
            pictures.append(convert(renderer.render(start + int(entry["length"]) // 2)))
    finally:
        if renderer is not None:
            renderer.close()
    for line in report.lines():
        print(f"写すときの記録 {line}")
    return pictures


def _read_ymm4_slots(
    video: Path,
    entries: list[dict[str, Any]],
    convert: Callable[[np.ndarray], np.ndarray] = _shrink,
) -> list[np.ndarray | None]:
    """YMM4 の書き出しから、枠ごとの真ん中の 1 枚を ``convert`` して返す

    動画に無い枠は ``None`` 既定は縮める 大きさを画素で測る探りは縮めずに受け取る
    """
    references = References(_ymm4_frames(video))
    pictures: list[np.ndarray | None] = [None] * len(entries)
    # 動画は戻せないので、枠を頭から順に引く 返すのは一覧の順
    for index in sorted(range(len(entries)), key=lambda at: int(entries[at]["start"])):
        entry = entries[index]
        picture = references.get(int(entry["start"]) + int(entry["length"]) // 2)
        pictures[index] = None if picture is None else convert(picture)
    return pictures


def _marked(picture: np.ndarray, centroid: tuple[float, float] | None) -> np.ndarray:
    """重心に赤い十字を描いた写し 人が並べた絵で、測った場所を確かめられるようにする"""
    marked = picture.copy()
    if centroid is None:
        return marked
    height, width = marked.shape[:2]
    x = min(width - 1, max(0, int(centroid[0] * width / WIDTH)))
    y = min(height - 1, max(0, int(centroid[1] * height / HEIGHT)))
    marked[y, max(0, x - 6) : x + 7] = (255.0, 0.0, 0.0)
    marked[max(0, y - 6) : y + 7, x] = (255.0, 0.0, 0.0)
    return marked


def command_mesh_measure(arguments: argparse.Namespace) -> int:
    work: Path = arguments.work
    manifest_path = work / "mesh-probe.json"
    if not manifest_path.exists():
        print(f"{manifest_path} がありません 先に mesh-build を走らせてください")
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 測れなかった道で前の表が残ると、新しい結果として開けてしまう
    # 測れたときは最後に書き直すので、先に消しておけばどの道でも残らない
    _clear_mesh_results(work)
    video = work / "mesh-probe.mp4"
    if not video.exists():
        print(f"{video} がまだ書き出されていません")
        print(f"YMM4 で {work / 'mesh-probe.ymmp'} を開き、そこへ書き出してから走らせてください")
        return 0
    if video.stat().st_mtime <= manifest_path.stat().st_mtime:
        # 探りを作り直したのに書き出しが前のままだと、新しい枠の一覧で古い絵を
        # 切り出して、別の点を動かした絵を測った表が出る 同じ時刻も断る
        # 置き場によっては時刻が 2 秒刻みでしか残らない
        print(f"{video} は探りを作り直す前の書き出しです")
        print(f"YMM4 で {work / 'mesh-probe.ymmp'} を開き直し、書き出してから走らせてください")
        return 0
    entries: list[dict[str, Any]] = manifest["slots"]
    threshold = float(manifest.get("threshold", MESH_THRESHOLD))
    try:
        if not has_video_stream(video):
            print(f"{video} に映像の道がありません 映像が入る形式で書き出してください")
            return 0
        if _fps_differs(video, int(manifest.get("fps", FPS))):
            return 0
        # 頭は開けても途中で切れた書き出しは、読み進めた所で復号が失敗する
        # 開けるかどうかだけを見ても、その穴は塞がらない
        theirs = _read_ymm4_slots(video, entries)
    except unreadable_export_errors():
        _explain_unreadable(video, work / "mesh-probe.ymmp")
        return 0
    ours = _render_mesh_slots(entries)
    images = work / "images"
    images.mkdir(exist_ok=True)
    rows = [_mesh_row(entry, theirs, ours, threshold, images) for entry in entries]
    _print_mesh_rows(rows)
    (work / "mesh-report.json").write_text(
        json.dumps({"threshold": threshold, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"{work / 'mesh-report.json'} へ書いた 並べた絵は {images} の mesh-*.png")
    return 0


def _mesh_row(
    entry: dict[str, Any],
    theirs: list[np.ndarray | None],
    ours: list[np.ndarray | None],
    threshold: float,
    images: Path,
) -> dict[str, Any]:
    """枠 1 つぶんの表の行 並べた絵（YMM4・Sashimono・差 3 倍）も書き出す"""
    index = int(entry["index"])
    baseline = int(entry["baseline"])
    by_row = tuple(entry["by_row"]) if "by_row" in entry else None
    by_column = tuple(entry["by_column"]) if "by_column" in entry else None
    row: dict[str, Any] = {
        "index": index,
        "name": entry["name"],
        "columns": entry["columns"],
        "rows": entry["rows"],
        "moved": entry["moved"],
        "by_row": by_row,
        "by_column": by_column,
    }
    for side, pictures in (("ymm4", theirs), ("sashimono", ours)):
        picture, base = pictures[index], pictures[baseline]
        if picture is None or base is None:
            row[side] = None
            row[f"{side}_reading"] = "絵が無い"
            continue
        found = change_centroid(picture, base, threshold)
        centroid = None if found is None else (found[0], found[1])
        row[side] = None if found is None else {"x": found[0], "y": found[1], "pixels": found[2]}
        row[f"{side}_reading"] = mesh_reading(
            centroid,
            None if by_row is None else (by_row[0], by_row[1]),
            None if by_column is None else (by_column[0], by_column[1]),
        )
    a, b = theirs[index], ours[index]
    if a is not None and b is not None:
        spots = [
            None if row[side] is None else (row[side]["x"], row[side]["y"])
            for side in ("ymm4", "sashimono")
        ]
        side_by_side = np.concatenate(
            [_marked(a, spots[0]), _marked(b, spots[1]), np.abs(a - b) * 3.0], axis=1
        )
        stem = f"mesh-{index:02d}"
        _save_png(np.clip(side_by_side, 0, 255).astype(np.uint8), images / f"{stem}.png")
        row["image"] = stem
    return row


def _print_mesh_rows(rows: list[dict[str, Any]]) -> None:
    def spot(value: dict[str, Any] | None) -> str:
        return "-" if value is None else f"({value['x']:.0f},{value['y']:.0f})"

    def point(value: tuple[float, ...] | None) -> str:
        return "-" if value is None else f"({value[0]:.0f},{value[1]:.0f})"

    # 位置は画面の px、Y は下が正 重心は崩れた範囲の真ん中なので、点そのものより
    # 少し格子の内側へ寄る 行ごとと列ごとの予想のどちらに近いかで読む
    print(
        f"{'枠':<14}{'YMM4 重心':>14}{'Sashimono 重心':>16}{'行ごと予想':>14}{'列ごと予想':>14}"
        f"{'YMM4':>10}{'Sashimono':>11}"
    )
    for row in rows:
        print(
            f"{row['name']:<14}{spot(row['ymm4']):>14}{spot(row['sashimono']):>16}"
            f"{point(row['by_row']):>14}{point(row['by_column']):>14}"
            f"{row['ymm4_reading']:>10}{row['sashimono_reading']:>11}"
        )
    print()
    print("YMM4 の列が「行ごと」なら今の写し方どおり 「列ごと」なら並べ替えが要る")


#: 絵の速さの探りの素材の長さ（秒） 枠より短くして、素材を読み切った後の絵
#: （止まるのか消えるのか）も枠の中に入るようにする
RATE_SOURCE_SECONDS = 4
#: 枠 1 つの長さ（フレーム） 素材の 1.5 倍 50% でも素材の 3 秒ぶんまで読み進み、
#: 200% では素材を 2 秒で読み切った後が 4 秒残る
RATE_SLOT = 180
#: 枠と枠の間の黒（フレーム） 前の枠の絵が次の枠の頭に残って、傾きの頭を狂わせない
RATE_GAP = 30
#: 並べる再生速度 等倍を基準に先頭へ 0 は音では無音・長さ 0 だった（2026-09-23）
RATE_CONDITIONS = (100.0, 50.0, 200.0, 0.0)
#: 絵を突き合わせる大きさ 1920x1080 をちょうど 10 分の 1 に縮める
#: testsrc2 は隣り合うフレームの差がこの大きさで 2.5 以上あり、書き出しの圧縮の
#: 揺れ（0.3 前後）よりずっと大きいので、隣のフレームと取り違えない
TINY_WIDTH, TINY_HEIGHT = 192, 108
#: 素材のどのフレームにも似ていないとみなす差（0〜255、色の平均）
#: 黒の画面と素材の差は 126 前後 圧縮の揺れや色の変換のずれは十数までに収まる
RATE_MATCH_LIMIT = 30.0
#: ``PlaybackRate`` と ``PlaybackRate2`` をわざと食い違わせた枠（Issue #117）
#: ``(PlaybackRate, PlaybackRate2 の値)`` 値が 2 つなら ``PlaybackRate2`` を
#: 枠の頭から終わりへ直線で動かす 前からある枠（``RATE_CONDITIONS``）の後ろに
#: 並べるので、前の測り結果とは同じ番号・同じ位置のまま比べられる
RATE_MISMATCHES: tuple[tuple[float, tuple[float, ...]], ...] = (
    (100.0, (50.0,)),
    (50.0, (100.0,)),
    (100.0, (50.0, 200.0)),
)
#: 傾きを区切って読む幅（フレーム） ``PlaybackRate2`` を動かした枠で、途中から
#: 傾きが変わるかを見る 1 秒ぶんなら、50→200 で隣の区切りとの差が 0.25 倍になり、
#: 突き合わせの揺れ（0.02 前後）よりずっと大きい
RATE_WINDOW = 30
#: 直線で動く値の書き方 実物（この機械のプロジェクトの ``Zoom`` 3 個）の形そのまま
#: ``KeyFrames`` を空のまま ``Values`` を 2 つ持ち、アイテムの頭から終わりへ動く
#: ``Bezier`` は実物がどの動く値にも同じ既定の曲線を書いていたので写す
_LINEAR = "直線移動"
_DEFAULT_BEZIER: dict[str, Any] = {
    "Points": [
        {
            "Point": {"X": 0.0, "Y": 0.0},
            "ControlPoint1": {"X": -0.3, "Y": -0.3},
            "ControlPoint2": {"X": 0.3, "Y": 0.3},
        },
        {
            "Point": {"X": 1.0, "Y": 1.0},
            "ControlPoint1": {"X": -0.3, "Y": -0.3},
            "ControlPoint2": {"X": 0.3, "Y": 0.3},
        },
    ],
    "IsQuadratic": False,
}


def _moving(values: tuple[float, ...]) -> dict[str, Any]:
    """動く値 値が 1 つなら止まった値（``_still`` と同じ）、2 つ以上なら直線で動かす"""
    if len(values) == 1:
        return _still(values[0])
    return {
        "Values": [{"Value": value} for value in values],
        "Span": 0.0,
        "AnimationType": _LINEAR,
        "Bezier": copy.deepcopy(_DEFAULT_BEZIER),
    }


@dataclass(frozen=True)
class RateSlot:
    """絵の速さの探りの枠 1 つ

    ``rate2`` は ``PlaybackRate2`` の値 空なら ``rate`` と同じ（実物どおりそろえる）
    """

    name: str
    rate: float
    start: int
    rate2: tuple[float, ...] = ()

    @property
    def second(self) -> tuple[float, ...]:
        """``PlaybackRate2`` に書く値"""
        return self.rate2 or (self.rate,)


def mismatch_name(rate: float, rate2: tuple[float, ...]) -> str:
    """食い違わせた枠の名前 動く値は矢印でつなぐ"""
    second = "→".join(f"{value:g}" for value in rate2)
    return f"PlaybackRate={rate:g} PlaybackRate2={second}"


def build_rate_slots() -> list[RateSlot]:
    """確かめる速さを、時間軸に重ならないように並べる

    食い違わせた枠は前からある枠の後ろへ足す 間に挟むと、前の測り結果と
    枠の番号と位置がずれて、同じ条件どうしを突き合わせられない
    """
    conditions: list[tuple[str, float, tuple[float, ...]]] = [
        (f"PlaybackRate={rate:g}", rate, ()) for rate in RATE_CONDITIONS
    ]
    conditions.extend((mismatch_name(rate, rate2), rate, rate2) for rate, rate2 in RATE_MISMATCHES)
    slots: list[RateSlot] = []
    cursor = 0
    for name, rate, rate2 in conditions:
        slots.append(RateSlot(name=name, rate=rate, start=cursor, rate2=rate2))
        cursor += RATE_SLOT + RATE_GAP
    return slots


def rate_video_item(slot: RateSlot, media: Path) -> dict[str, Any]:
    """探りの枠 1 つを YMM4 の動画アイテムにする

    項目の並びと既定の値は、この機械の YMM4 プロジェクトにあった新しい版の
    ``VideoItem`` 67 個（``PlaybackRate2`` と ``PlaybackRateAudioProcessingMode`` を
    持つ形）から写した 実物は ``PlaybackRate`` と ``PlaybackRate2`` がどれも同じ値
    だったので、前からある枠は両方に同じ速さを書く 食い違わせた枠
    （``RateSlot.rate2``）だけが、どちらが効いたのかを読むために片方を変える
    ``Zoom`` 100 のまま 素材は画面と同じ大きさなので、画面いっぱいに映る
    """
    return {
        "$type": "YukkuriMovieMaker.Project.Items.VideoItem, YukkuriMovieMaker",
        "IsWaveformEnabled": False,
        "FilePath": str(media),
        "AudioTrackIndex": 0,
        "Volume": _still(100.0),
        "Pan": _still(0.0),
        "PlaybackRate2": _moving(slot.second),
        "PlaybackRateAudioProcessingMode": "Resampling",
        "ContentOffset": "00:00:00",
        "IsLooped": False,
        "EchoIsEnabled": False,
        "EchoInterval": 0.1,
        "EchoAttenuation": 40.0,
        "AudioEffects": [],
        "X": _still(0.0),
        "Y": _still(0.0),
        "Z": _still(0.0),
        "Opacity": _still(100.0),
        "Zoom": _still(100.0),
        "Rotation": _still(0.0),
        "FadeIn": 0.0,
        "FadeOut": 0.0,
        "Blend": "Normal",
        "IsInverted": False,
        "IsClippingWithObjectAbove": False,
        "IsAlwaysOnTop": False,
        "IsZOrderEnabled": False,
        "VideoEffects": [],
        "Group": 0,
        "Frame": slot.start,
        "Layer": 0,
        "KeyFrames": {"Frames": [], "Count": 0},
        "Length": RATE_SLOT,
        "PlaybackRate": slot.rate,
        "Remark": slot.name,
        "IsLocked": False,
        "IsHidden": False,
    }


def rate_manifest(slots: list[RateSlot], media: Path) -> dict[str, Any]:
    """枠の一覧 ``video-rate-measure`` はこれだけを見て切り出し、Sashimono でも描く"""
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "media": str(media),
        "source_frames": RATE_SOURCE_SECONDS * FPS,
        "match_limit": RATE_MATCH_LIMIT,
        "slots": [
            {
                "index": index,
                "name": slot.name,
                "rate": slot.rate,
                "rate2": list(slot.second),
                "start": slot.start,
                "length": RATE_SLOT,
                "item": rate_video_item(slot, media),
            }
            for index, slot in enumerate(slots)
        ],
    }


def make_rate_source(target: Path) -> str:
    """ffmpeg の testsrc2 でフレームごとに絵が変わる動画を作る 作れなければ理由を返す

    すべてのフレームを鍵フレームにする（``-g 1``） YMM4 が速さを変えて飛び飛びに
    読んでも、前のフレームからの差分の崩れが絵に乗らない
    """
    if shutil.which("ffmpeg") is None:
        return "ffmpeg が見つからないので、探りの動画を作れない"
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration={RATE_SOURCE_SECONDS}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "1",
        "-crf",
        "18",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode == 0 and target.exists():
        return ""
    message = completed.stderr.decode("utf-8", "replace").strip().splitlines()
    if any("libx264" in line for line in message):
        return "この ffmpeg には libx264 が無いので、探りの動画を作れない"
    return "ffmpeg が探りの動画を作れなかった " + (message[-1] if message else "")


def _clear_rate_results(work: Path) -> None:
    """前の測り結果を捨てる 残すと、新しい枠の一覧に対応しない表を読んでしまう"""
    (work / "video-rate-report.json").unlink(missing_ok=True)


def command_video_rate_build(arguments: argparse.Namespace) -> int:
    # 絶対パスにしてから書く YMM4 はこの道具の作業フォルダを知らないので、
    # 相対のまま `.ymmp` へ書くと素材を見つけられず、全部の枠が黒になる
    # 黒を測ると「どの速さでも絵が出ない」と読めてしまう
    work: Path = arguments.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    # 作れなかった道でも前の表を残さない 残ると、作り直した探りの結果として開けてしまう
    _clear_rate_results(work)
    media = work / "video-rate-source.mp4"
    problem = make_rate_source(media)
    if problem:
        # 測れない環境で「壊れた」と読まれないように、落とさずに終える
        _drop_probe(work, "video-rate-probe", "video-rate-report.json")
        print(problem)
        return 0
    slots = build_rate_slots()
    items = [rate_video_item(slot, media) for slot in slots]
    length = max(slot.start for slot in slots) + RATE_SLOT + RATE_GAP
    project = work / "video-rate-probe.ymmp"
    write_document(items, length, project)
    (work / "video-rate-probe.json").write_text(
        json.dumps(rate_manifest(slots, media), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{len(slots)} 枠を並べた（{length} フレーム、{length / FPS:.0f} 秒）")
    video = work / "video-rate-probe.mp4"
    if video.exists():
        # 前の書き出しが残っていると、新しい枠の一覧で古い絵を突き合わせてしまう
        # `video-rate-measure` は書き出しが一覧より古ければ止めるが、ここでも言っておく
        print(f"{video} は前の探りの書き出しです 作り直した方で書き出し直してください")
    print(f"YMM4 で {project} を開き、{video} として書き出してください")
    print(f"書き出しは {WIDTH}x{HEIGHT}・{FPS}fps・頭から終わりまで（範囲を絞らない）")
    print_export_hint(project, video)
    return 0


def tiny(image: np.ndarray) -> np.ndarray:
    """突き合わせ用に面積の平均で縮める"""
    height, width = image.shape[:2]
    fy, fx = max(1, height // TINY_HEIGHT), max(1, width // TINY_WIDTH)
    cropped = image[: TINY_HEIGHT * fy, : TINY_WIDTH * fx, :3].astype(np.float32)
    return cropped.reshape(TINY_HEIGHT, fy, TINY_WIDTH, fx, 3).mean(axis=(1, 3))


def nearest_frame(picture: np.ndarray, sources: np.ndarray) -> tuple[int, float]:
    """``picture`` に一番近い素材のフレームの番号と、その差（0〜255、色の平均）"""
    distances = np.abs(sources - picture[None]).mean(axis=(1, 2, 3))
    index = int(np.argmin(distances))
    return index, float(distances[index])


def match_frames(
    pictures: list[np.ndarray | None], sources: np.ndarray, limit: float = RATE_MATCH_LIMIT
) -> list[tuple[int | None, float | None]]:
    """枠の各フレームが素材の何フレーム目か 似た物が無ければ番号は ``None``

    差が ``limit`` を超えたものは素材のどれでもない（黒や別の絵） 一番近い物を
    そのまま番号にすると、黒い画面が素材の暗いフレームとして数えられる
    """
    matched: list[tuple[int | None, float | None]] = []
    for picture in pictures:
        if picture is None:
            matched.append((None, None))
            continue
        index, distance = nearest_frame(picture, sources)
        matched.append((index if distance <= limit else None, distance))
    return matched


def rate_slope(indices: list[int | None], last: int) -> float | None:
    """経過フレームに対する素材のフレーム番号の傾き 求められなければ ``None``

    素材の最後のフレーム（``last``）へ届いた所で切る 読み切った後に最後の絵で
    止まる作りだと、そこが傾き 0 の尾になり、200% の傾きが 2 より小さく出る
    最後のフレームそのものも数えないのは、読み切った後の止まった絵と見分けられないため

    ただし一致した絵がすべて同じフレームなら止まった絵（0） 最後のフレームで止まった枠
    （0 で素材の末尾から映す・最後の絵を出し続ける）を切ると点が残らず、一致数は
    揃っているのに「絵が無い」と読み、表の中で食い違う
    """
    found = [index for index in indices if index is not None]
    if len(found) >= 2 and len(set(found)) == 1:
        return 0.0
    points: list[tuple[int, int]] = []
    for elapsed, index in enumerate(indices):
        if index is None:
            continue
        if index >= last:
            break
        points.append((elapsed, index))
    if len(points) < 2:
        return None
    xs = np.array([point[0] for point in points], dtype=np.float64)
    ys = np.array([point[1] for point in points], dtype=np.float64)
    if float(np.ptp(ys)) == 0.0:
        # 最小二乗でも 0 になるが、丸めの揺れで 1e-17 のような値が出ると読みにくい
        return 0.0
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope


def window_slopes(
    indices: list[int | None], last: int, width: int = RATE_WINDOW
) -> list[float | None]:
    """``width`` フレームごとに区切った傾き 区切りの中で求められなければ ``None``

    素材の最後のフレームに届いた点は区切りごとに外す 外さないと、読み切った後に
    最後の絵で止まった区切りが「止まる（0）」と読め、``PlaybackRate2`` が途中で
    0 へ落ちたように見える
    """
    slopes: list[float | None] = []
    for begin in range(0, len(indices), width):
        window = [
            index if index is not None and index < last else None
            for index in indices[begin : begin + width]
        ]
        slopes.append(rate_slope(window, last))
    return slopes


def expected_windows(values: list[float], length: int, width: int = RATE_WINDOW) -> list[float]:
    """速さの値（百分率）がそのまま効いたときの、区切りごとの傾きの予想

    値が 2 つなら、枠の頭から終わりへ直線で動くと読む（こちらの読み込みの
    ``frame_positions`` と同じく、頭が 0・終わりが ``length``） 区切りの傾きは、
    YMM4 がフレームごとに速さを積み上げて素材を進めるなら、区切りの真ん中の速さになる
    積み上げずに別の進め方をしていれば、実測の区切りの並びがこの予想と違う形に曲がる
    """
    first, final = values[0], values[-1]
    windows: list[float] = []
    for begin in range(0, length, width):
        end = min(begin + width, length)
        middle = (begin + end - 1) / 2.0
        value = first + (final - first) * middle / max(1, length)
        windows.append(value / 100.0)
    return windows


#: 近い方の予想でも区切りの傾きの差の平均がこれを超えたら、どちらとも合わないと読む
#: 突き合わせの揺れ（0.02 前後）より十分に大きく、50 と 100 の差（0.5）より十分に小さい
RATE_FAR = 0.15


#: 動く PlaybackRate2 の 3 つ目の読み 頭が PlaybackRate・終わりが PlaybackRate2 の最後
HEAD_TO_LAST = "PlaybackRate→PlaybackRate2の最後"


def rate_expectations(entry: dict[str, Any]) -> dict[str, list[float]]:
    """``PlaybackRate`` と ``PlaybackRate2`` のそれぞれが効いたときの、区切りごとの傾き

    前の版の一覧には ``rate2`` が無い そのときは ``PlaybackRate`` と同じ値を書いていた
    長さの無い行は枠の既定の長さで読む 読めずに止まると、ほかの枠の表まで出ない
    ``PlaybackRate2`` が動く枠は、測って合った 3 つ目の読み（``HEAD_TO_LAST``）も並べる
    """
    length = int(entry.get("length", RATE_SLOT))
    rate = float(entry["rate"])
    second = [float(value) for value in entry.get("rate2") or [rate]]
    expectations = {
        "PlaybackRate": expected_windows([rate], length),
        "PlaybackRate2": expected_windows(second, length),
    }
    if len(second) >= 2:
        # 動く PlaybackRate2 を 1 枠測ったときの読み（2026-09-23 YMM4 4.56.1.1）
        # 100 と 50→200 で傾きが 1.06・1.25・1.41 と並び、頭が PlaybackRate・終わりが
        # PlaybackRate2 の最後の直線（1.08・1.25・1.42）に合った 1 枠からの読みなので、
        # 次に測るときも並べて、ほかの値でも合うかを見る
        expectations[HEAD_TO_LAST] = expected_windows([rate, second[-1]], length)
    return expectations


def nearer_rate(
    measured: list[float | None], expectations: dict[str, list[float]]
) -> tuple[str, dict[str, float]]:
    """区切りごとの実測の傾きが、どちらの予想に近いか

    返すのは読みと、予想ごとの差の平均（測れた区切りだけで取る）
    予想どうしが同じ枠（前からある枠）は見分けられないので、そう書く
    """
    names = list(expectations)
    if len({tuple(expectations[name]) for name in names}) == 1:
        return "見分けられない（同じ値）", {}
    distances: dict[str, float] = {}
    for name in names:
        pairs = [
            abs(found - guess)
            for found, guess in zip(measured, expectations[name], strict=False)
            if found is not None
        ]
        if not pairs:
            return "測れない", {}
        distances[name] = float(np.mean(pairs))
    best = min(names, key=lambda name: distances[name])
    if distances[best] > RATE_FAR:
        return "どちらとも合わない", distances
    return best, distances


def rate_reading(slope: float | None) -> str:
    """傾きの読み 0 に近ければ止まる"""
    if slope is None:
        return "絵が無い"
    if abs(slope) < 0.05:
        return "止まる"
    return f"{slope:.2f} 倍"


def _read_rate_slots(video: Path, entries: list[dict[str, Any]]) -> list[list[np.ndarray | None]]:
    """YMM4 の書き出しから、枠ごとに全フレームを縮めて返す 動画に無い所は ``None``"""
    references = References(_ymm4_frames(video))
    pictures: list[list[np.ndarray | None]] = [[] for _ in entries]
    # 動画は戻せないので、枠を頭から順に引く 返すのは一覧の順
    for index in sorted(range(len(entries)), key=lambda at: int(entries[at]["start"])):
        entry = entries[index]
        start = int(entry["start"])
        for frame in range(start, start + int(entry["length"])):
            picture = references.get(frame)
            pictures[index].append(None if picture is None else tiny(picture))
    return pictures


def _read_source(media: Path) -> np.ndarray:
    """素材の全フレームを縮めて ``(枚数, 高さ, 幅, 3)`` で返す 絵が無ければ 0 枚"""
    frames = [tiny(convert()) for _, convert in _ymm4_frames(media)]
    if not frames:
        return np.zeros((0, TINY_HEIGHT, TINY_WIDTH, 3), dtype=np.float32)
    return np.stack(frames)


def _render_rate_slots(entries: list[dict[str, Any]]) -> list[list[np.ndarray | None]]:
    """同じ ``.ymmp`` の枠を Sashimono で描き、枠ごとに全フレームを縮めて返す

    ``compare`` と同じ道（写す・置く・描く）に、素材の登録を足して通す 登録を
    飛ばすと、動画のクリップが ``media_id`` を持たず、全部の枠が透明になる
    """
    from sashimono.compat.catalog import gather_media
    from sashimono.core.model import Project, ProjectSettings
    from sashimono.core.timebase import FrameRate
    from sashimono.engine.render import FrameRenderer

    settings = ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(FPS))
    report = CompatibilityReport()
    pictures: list[list[np.ndarray | None]] = []
    renderer: FrameRenderer | None = None
    try:
        for entry in entries:
            objects = map_template([copy.deepcopy(entry["item"])], report=report)
            project = Project.create(settings)
            plan = gather_media(objects, project, _probe_or_none)
            start, length = int(entry["start"]), int(entry["length"])
            if plan.missing:
                pictures.append([None] * length)
                continue
            commands = [*plan.commands, *place(objects, project, at_frame=start, media=plan.media)]
            for command in commands:
                project = command.apply(project)
            if renderer is None:
                renderer = FrameRenderer(project)
            else:
                renderer.set_project(project)
            frames = range(start, start + length)
            pictures.append([tiny(renderer.render(frame)) for frame in frames])
    finally:
        if renderer is not None:
            renderer.close()
    for line in report.lines():
        print(f"写すときの記録 {line}")
    return pictures


def rate_row(
    entry: dict[str, Any],
    sides: list[tuple[str, list[np.ndarray | None]]],
    sources: np.ndarray,
    limit: float,
) -> dict[str, Any]:
    """枠 1 つぶんの表の行 YMM4 と Sashimono を同じ読み方で並べる"""
    last = len(sources) - 1
    expectations = rate_expectations(entry)
    row: dict[str, Any] = {
        "index": entry["index"],
        "name": entry["name"],
        "rate": entry["rate"],
        "rate2": entry.get("rate2") or [entry["rate"]],
        "expected": expectations,
    }
    for side, pictures in sides:
        matched = match_frames(pictures, sources, limit)
        indices = [found for found, _ in matched]
        slope = rate_slope(indices, last)
        windows = window_slopes(indices, last)
        # 予想との差は ``gaps`` に置く ``distances`` は素材との突き合わせの差で、
        # 同じ名前にすると、前の表と比べるときに使う突き合わせの差の列が消える
        nearer, gaps = nearer_rate(windows, expectations)
        row[side] = {
            "slope": slope,
            "reading": rate_reading(slope),
            "windows": windows,
            "nearer": nearer,
            "gaps": gaps,
            "first": next((found for found in indices if found is not None), None),
            "matched": sum(found is not None for found in indices),
            "frames": indices,
            "distances": [distance for _, distance in matched],
        }
    return row


def command_video_rate_measure(arguments: argparse.Namespace) -> int:
    work: Path = arguments.work
    manifest_path = work / "video-rate-probe.json"
    if not manifest_path.exists():
        print(f"{manifest_path} がありません 先に video-rate-build を走らせてください")
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 測れなかった道で前の表が残ると、新しい結果として開けてしまう
    # 測れたときは最後に書き直すので、先に消しておけばどの道でも残らない
    _clear_rate_results(work)
    project = work / "video-rate-probe.ymmp"
    video = work / "video-rate-probe.mp4"
    if not video.exists():
        print(f"{video} がまだ書き出されていません")
        print(f"YMM4 で {project} を開き、そこへ書き出してから走らせてください")
        return 0
    if video.stat().st_mtime <= manifest_path.stat().st_mtime:
        # 探りを作り直したのに書き出しが前のままだと、新しい枠の一覧で古い絵を
        # 突き合わせ、別の速さを測った表が出る 同じ時刻も断る
        # 置き場によっては時刻が 2 秒刻みでしか残らない
        print(f"{video} は探りを作り直す前の書き出しです")
        print(f"YMM4 で {project} を開き直し、書き出してから走らせてください")
        return 0
    media = Path(manifest["media"])
    if not media.exists():
        print(f"素材 {media} がありません video-rate-build を走らせ直してください")
        return 0
    entries: list[dict[str, Any]] = manifest["slots"]
    limit = float(manifest.get("match_limit", RATE_MATCH_LIMIT))
    try:
        sources = _read_source(media)
    except unreadable_export_errors():
        print(f"素材 {media} を読めません video-rate-build を走らせ直してください")
        return 0
    if len(sources) == 0:
        print(f"素材 {media} に絵がありません video-rate-build を走らせ直してください")
        return 0
    try:
        if not has_video_stream(video):
            print(f"{video} に映像の道がありません 映像が入る形式で書き出してください")
            return 0
        if _fps_differs(video, int(manifest.get("fps", FPS))):
            return 0
        # 頭は開けても途中で切れた書き出しは、読み進めた所で復号が失敗する
        theirs = _read_rate_slots(video, entries)
    except unreadable_export_errors():
        _explain_unreadable(video, project)
        return 0
    ours = None if arguments.skip_sashimono else _render_rate_slots(entries)
    rows = []
    for index, entry in enumerate(entries):
        sides = [("ymm4", theirs[index])]
        if ours is not None:
            sides.append(("sashimono", ours[index]))
        rows.append(rate_row(entry, sides, sources, limit))
    _print_rate_rows(rows)
    (work / "video-rate-report.json").write_text(
        json.dumps(
            {"source_frames": len(sources), "match_limit": limit, "rows": rows},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{work / 'video-rate-report.json'} へ書いた")
    return 0


def _print_rate_rows(rows: list[dict[str, Any]]) -> None:
    def cell(value: dict[str, Any] | None) -> str:
        if value is None:
            return f"{'-':>24}"
        first = "-" if value["first"] is None else str(value["first"])
        return f"{value['reading']:>10}{first:>6}{value['matched']:>8}"

    print(
        f"{'枠':<22}{'YMM4 読み':>10}{'頭':>6}{'一致数':>8}"
        f"{'Sashimono 読み':>14}{'頭':>6}{'一致数':>8}"
    )
    for row in rows:
        print(f"{row['name']:<22}{cell(row.get('ymm4'))}{cell(row.get('sashimono'))}")
    print()
    print("読みは「枠の頭からの経過フレーム → 素材の何フレーム目」の傾き 1.00 倍で等倍、")
    print("0.50 倍・2.00 倍なら速さのとおり 止まるなら絵は動かない")
    print("絵が無いなら素材の絵が出ていない（黒や別の絵）")
    print("頭は枠の最初に映った素材のフレーム 一致数は素材のどれかに似ていたフレームの数")
    _print_rate_mismatches(rows)


def _slopes(values: list[float | None] | list[float]) -> str:
    return " ".join("  -  " if value is None else f"{value:5.2f}" for value in values)


def _print_rate_mismatches(rows: list[dict[str, Any]]) -> None:
    """食い違わせた枠だけを、区切りごとの傾きと 2 つの予想で並べる（Issue #117）"""
    mismatched = [
        row for row in rows if row["expected"]["PlaybackRate"] != row["expected"]["PlaybackRate2"]
    ]
    if not mismatched:
        return
    print()
    print(f"PlaybackRate と PlaybackRate2 の食い違い（{RATE_WINDOW} フレームごとの傾き）")
    for row in mismatched:
        print(row["name"])
        for name, guess in row["expected"].items():
            print(f"  {name + ' 予想':<20}{_slopes(guess)}")
        for side in ("ymm4", "sashimono"):
            value = row.get(side)
            if value is None:
                continue
            gaps = " ".join(f"{name} との差 {gap:.3f}" for name, gap in value["gaps"].items())
            print(f"  {side + ' 実測':<20}{_slopes(value['windows'])}  → {value['nearer']}  {gaps}")
    print()
    print("区切りの傾きが近い方の値を YMM4 が読んでいる 動かした枠は、区切りごとに")
    print("傾きが増えていけば PlaybackRate2 の動きが効いている - は素材を読み切った後か絵が無い")
    print(f"{HEAD_TO_LAST} は、1 枠を測って合った読み（docs/development.md）")


#: 大きさの探りの枠 1 つの長さ（フレーム） 絵は止まっているので、真ん中の 1 枚だけを見る
ZOOM_SLOT = 30
#: 並べる素材 ``(名前, 幅, 高さ, 種類, 拡大率)``（Issue #159）
#: 画面と同じ大きさの素材は、素材の画素のままでも画面に収めても同じ絵になって見分けられない
#: 小さい・大きい・縦長を並べる 拡大率 200 の枠は、拡大率がどちらの大きさに掛かるかを見る
ZOOM_CONDITIONS: tuple[tuple[str, int, int, str, float], ...] = (
    ("画像 1920x1080", 1920, 1080, "image", 100.0),
    ("画像 640x360", 640, 360, "image", 100.0),
    ("画像 3840x2160", 3840, 2160, "image", 100.0),
    ("画像 360x640 縦長", 360, 640, "image", 100.0),
    ("画像 640x360 拡大率 200", 640, 360, "image", 200.0),
    ("動画 640x360", 640, 360, "video", 100.0),
)
#: 素材の地の色と、真ん中に置く印の色（RGB） 印は素材の縦横の半分
#: 地の大きさだけを測ると、3840x2160 は画素のままでも収めても画面いっぱいになって
#: 見分けられない 印なら 1920x1080 と 960x540 に分かれる
ZOOM_GROUND = (40, 90, 230)
ZOOM_MARK = (235, 40, 40)
#: 印の縁とみなす行と列の画素の数 圧縮で印の色に寄った点が 1 つ 2 つ混じっても、
#: 矩形を広げない
ZOOM_EDGE_PIXELS = 4
#: 予想と合ったとみなす差（px） 補間と圧縮で縁が 1〜2 画素ぶれる分より広く、
#: 2 つの読みの差（一番近い 3840x2160 でも 960px）よりずっと狭くする
ZOOM_TOLERANCE = 8.0
ZOOM_NATIVE = "素材の画素"
ZOOM_FIT = "画面に収める"


@dataclass(frozen=True)
class ZoomSlot:
    """大きさの探りの枠 1 つ ``kind`` は ``image`` か ``video``"""

    name: str
    width: int
    height: int
    kind: str
    zoom: float
    start: int

    @property
    def media_name(self) -> str:
        """素材のファイル名 同じ大きさの画像と動画は同じ絵から作る"""
        suffix = "png" if self.kind == "image" else "mp4"
        return f"zoom-probe-{self.width}x{self.height}.{suffix}"


def build_zoom_slots() -> list[ZoomSlot]:
    """確かめる素材を、時間軸に重ならないように並べる"""
    slots: list[ZoomSlot] = []
    cursor = 0
    for name, width, height, kind, zoom in ZOOM_CONDITIONS:
        slots.append(ZoomSlot(name, width, height, kind, zoom, cursor))
        cursor += ZOOM_SLOT + GAP
    return slots


def zoom_pattern(width: int, height: int) -> np.ndarray:
    """地の色の真ん中に縦横半分の印を置いた絵 ``(高さ, 幅, 3)`` の RGB"""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = ZOOM_GROUND
    top, left = height // 4, width // 4
    image[top : top + height // 2, left : left + width // 2] = ZOOM_MARK
    return image


def zoom_expectations(width: int, height: int, zoom: float) -> dict[str, tuple[float, float]]:
    """印が画面に出る大きさ（幅・高さ px）の予想 画面からはみ出す分は切れた後の大きさ"""
    scale = zoom / 100.0
    fit = min(WIDTH / width, HEIGHT / height)

    def shown(factor: float) -> tuple[float, float]:
        return min(float(WIDTH), width / 2 * factor), min(float(HEIGHT), height / 2 * factor)

    return {ZOOM_NATIVE: shown(scale), ZOOM_FIT: shown(scale * fit)}


def mark_box(picture: np.ndarray) -> tuple[int, int, int, int] | None:
    """印の色が占める矩形（左・上・幅・高さ px） 印が無ければ ``None``

    印の色・地の色・黒のうち一番近いものを色の読みとする 縁は拡大の補間で 2 色が
    混ざるので、決まった閾値で切ると、どちらの色に寄せたかで 1 画素ずつ食い違う
    """
    rgb = picture[..., :3].astype(np.float32)
    colours = np.asarray([ZOOM_MARK, ZOOM_GROUND, (0, 0, 0)], dtype=np.float32)
    distance = np.abs(rgb[..., None, :] - colours).sum(axis=-1)
    mark = distance.argmin(axis=-1) == 0
    columns = np.flatnonzero(mark.sum(axis=0) >= ZOOM_EDGE_PIXELS)
    rows = np.flatnonzero(mark.sum(axis=1) >= ZOOM_EDGE_PIXELS)
    if columns.size == 0 or rows.size == 0:
        return None
    left, top = int(columns[0]), int(rows[0])
    return left, top, int(columns[-1]) - left + 1, int(rows[-1]) - top + 1


def zoom_reading(
    box: tuple[int, int, int, int] | None, expectations: dict[str, tuple[float, float]]
) -> str:
    """測った印の大きさが、どちらの置き方の予想に合うか"""
    if box is None:
        return "印が無い"
    if len(set(expectations.values())) == 1:
        return "見分けない"
    for name, (width, height) in expectations.items():
        if abs(box[2] - width) <= ZOOM_TOLERANCE and abs(box[3] - height) <= ZOOM_TOLERANCE:
            return name
    return "どちらとも合わない"


def zoom_video_item(slot: ZoomSlot, media: Path) -> dict[str, Any]:
    """動画の枠 ``video-rate-build`` の動画アイテム（実物の形）の長さと拡大率だけを変える"""
    item = rate_video_item(RateSlot(name=slot.name, rate=100.0, start=slot.start), media)
    item["Length"] = ZOOM_SLOT
    item["Zoom"] = _still(slot.zoom)
    return item


def zoom_item(slot: ZoomSlot, media: Path) -> dict[str, Any]:
    if slot.kind == "video":
        return zoom_video_item(slot, media)
    return image_item(media, frame=slot.start, length=ZOOM_SLOT, remark=slot.name, zoom=slot.zoom)


def make_still_video(image: Path, target: Path) -> str:
    """止まった絵の動画を ffmpeg で作る 作れなければ理由を返す"""
    if shutil.which("ffmpeg") is None:
        return "ffmpeg が見つからないので、探りの動画を作れない"
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-i",
        str(image),
        "-t",
        "2",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "12",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode == 0 and target.exists():
        return ""
    message = completed.stderr.decode("utf-8", "replace").strip().splitlines()
    return "ffmpeg が探りの動画を作れなかった " + (message[-1] if message else "")


def zoom_manifest(slots: list[ZoomSlot], work: Path) -> dict[str, Any]:
    """枠の一覧 ``zoom-measure`` はこれだけを見て切り出し、Sashimono でも描く"""
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "slots": [
            {
                "index": index,
                "name": slot.name,
                "media": [slot.width, slot.height],
                "kind": slot.kind,
                "zoom": slot.zoom,
                "start": slot.start,
                "length": ZOOM_SLOT,
                "expect": {
                    name: list(size)
                    for name, size in zoom_expectations(slot.width, slot.height, slot.zoom).items()
                },
                "item": zoom_item(slot, work / slot.media_name),
            }
            for index, slot in enumerate(slots)
        ],
    }


def _clear_zoom_results(work: Path) -> None:
    """前の測り結果を捨てる 残すと、新しい枠の一覧に対応しない表や絵を読んでしまう"""
    (work / "zoom-report.json").unlink(missing_ok=True)
    images = work / "images"
    if images.is_dir():
        for old in images.glob("zoom-*.png"):
            old.unlink()


def command_zoom_build(arguments: argparse.Namespace) -> int:
    # 絶対パスにしてから書く YMM4 はこの道具の作業フォルダを知らないので、相対のまま
    # `.ymmp` へ書くと素材を見つけられず、全部の枠が黒になる
    work: Path = arguments.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    _clear_zoom_results(work)
    slots = build_zoom_slots()
    for slot in slots:
        if slot.kind != "image":
            continue
        _save_png(zoom_pattern(slot.width, slot.height), work / slot.media_name)
    for slot in [slot for slot in slots if slot.kind == "video"]:
        still = work / slot.media_name.replace(".mp4", ".png")
        if not still.exists():
            _save_png(zoom_pattern(slot.width, slot.height), still)
        problem = make_still_video(still, work / slot.media_name)
        if problem:
            # 測れない環境で「壊れた」と読まれないように、落とさずに終える
            _drop_probe(work, "zoom-probe", "zoom-report.json")
            print(problem)
            return 0
    items = [zoom_item(slot, work / slot.media_name) for slot in slots]
    length = max(slot.start for slot in slots) + ZOOM_SLOT + GAP
    project = work / "zoom-probe.ymmp"
    write_document(items, length, project)
    (work / "zoom-probe.json").write_text(
        json.dumps(zoom_manifest(slots, work), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{len(slots)} 枠を並べた（{length} フレーム、{length / FPS:.1f} 秒）")
    video = work / "zoom-probe.mp4"
    if video.exists():
        print(f"{video} は前の探りの書き出しです 作り直した方で書き出し直してください")
    print(f"YMM4 で {project} を開き、{video} として書き出してください")
    print(f"書き出しは {WIDTH}x{HEIGHT}・{FPS}fps・頭から終わりまで（範囲を絞らない）")
    print_export_hint(project, video)
    return 0


def _unshrunk(image: np.ndarray) -> np.ndarray:
    """縮めずに RGB だけを取る 印の大きさは画素で測るので、縮めると 4 画素ずつしか読めない"""
    return np.ascontiguousarray(image[..., :3])


def _stale_export(video: Path, manifest: Path, project: Path, build: str) -> bool:
    """書き出しが無いか、探りを作り直す前の物なら案内して真を返す"""
    if not manifest.exists():
        print(f"{manifest} がありません 先に {build} を走らせてください")
        return True
    if not video.exists():
        print(f"{video} がまだ書き出されていません")
        print(f"YMM4 で {project} を開き、そこへ書き出してから走らせてください")
        return True
    if video.stat().st_mtime <= manifest.stat().st_mtime:
        # 探りを作り直したのに書き出しが前のままだと、新しい枠の一覧で古い絵を
        # 切り出して、別の条件を測った表が出る 置き場によっては時刻が 2 秒刻みなので同じ時刻も断る
        print(f"{video} は探りを作り直す前の書き出しです")
        print(f"YMM4 で {project} を開き直し、書き出してから走らせてください")
        return True
    return False


def _read_still_export(
    video: Path,
    project: Path,
    entries: list[dict[str, Any]],
    fps: int,
    convert: Callable[[np.ndarray], np.ndarray],
) -> list[np.ndarray | None] | None:
    """YMM4 の書き出しから枠ごとの 1 枚を取る 読めなければ案内して ``None``"""
    try:
        if not has_video_stream(video):
            print(f"{video} に映像の道がありません 映像が入る形式で書き出してください")
            return None
        if _fps_differs(video, fps):
            return None
        return _read_ymm4_slots(video, entries, convert)
    except unreadable_export_errors():
        _explain_unreadable(video, project)
        return None


def command_zoom_measure(arguments: argparse.Namespace) -> int:
    work: Path = arguments.work
    manifest_path = work / "zoom-probe.json"
    project = work / "zoom-probe.ymmp"
    video = work / "zoom-probe.mp4"
    _clear_zoom_results(work)
    if _stale_export(video, manifest_path, project, "zoom-build"):
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = manifest["slots"]
    theirs = _read_still_export(video, project, entries, int(manifest["fps"]), _unshrunk)
    if theirs is None:
        return 0
    ours = _render_still_slots(entries, convert=_unshrunk)
    images = work / "images"
    images.mkdir(exist_ok=True)
    rows = [zoom_row(entry, theirs[index], ours[index]) for index, entry in enumerate(entries)]
    for index, (a, b) in enumerate(zip(theirs, ours, strict=True)):
        if a is not None and b is not None:
            side = np.concatenate([_shrink(a), _shrink(b)], axis=1)
            _save_png(np.clip(side, 0, 255).astype(np.uint8), images / f"zoom-{index:02d}.png")
    _print_zoom_rows(rows)
    (work / "zoom-report.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"{work / 'zoom-report.json'} へ書いた 並べた絵（YMM4・Sashimono）は {images} の zoom-*.png"
    )
    return 0


def zoom_row(
    entry: dict[str, Any], theirs: np.ndarray | None, ours: np.ndarray | None
) -> dict[str, Any]:
    """枠 1 つぶんの表の行 YMM4 と Sashimono を同じ読み方で並べる"""
    expect = {name: (float(size[0]), float(size[1])) for name, size in entry["expect"].items()}
    row: dict[str, Any] = {
        "name": entry["name"],
        "media": entry["media"],
        "zoom": entry["zoom"],
        "expect": {name: list(size) for name, size in expect.items()},
    }
    for side, picture in (("ymm4", theirs), ("sashimono", ours)):
        box = None if picture is None else mark_box(picture)
        row[side] = None if box is None else list(box)
        row[f"{side}_reading"] = "絵が無い" if picture is None else zoom_reading(box, expect)
    return row


def _print_zoom_rows(rows: list[dict[str, Any]]) -> None:
    def size(box: list[int] | None) -> str:
        return "-" if box is None else f"{box[2]}x{box[3]}"

    def guess(value: list[float]) -> str:
        return f"{value[0]:.0f}x{value[1]:.0f}"

    print(
        f"{'枠':<24}{'YMM4 の印':>12}{'Sashimono の印':>16}"
        f"{'画素のまま':>12}{'収める':>12}{'YMM4':>12}{'Sashimono':>12}"
    )
    for row in rows:
        print(
            f"{row['name']:<24}{size(row['ymm4']):>12}{size(row['sashimono']):>16}"
            f"{guess(row['expect'][ZOOM_NATIVE]):>12}{guess(row['expect'][ZOOM_FIT]):>12}"
            f"{row['ymm4_reading']:>12}{row['sashimono_reading']:>12}"
        )
    print()
    print("印は素材の縦横の半分 画面からはみ出す分は切れた後の大きさ")


#: エフェクトアイテムの探りの枠 1 つの長さ（フレーム）（Issue #143）
EFFECT_SLOT = 30
#: 並べる条件 ``(名前, 下の絵, エフェクト, 不透明度)``
#: 下の絵の ``shape`` は周りが透明な図形（``base_shape``）、``picture`` は画面いっぱいの絵
#: 2 つの読み方が食い違うのは、下の絵が透明な所と、エフェクトが絵を縮める・動かすとき
#: 基準の枠（エフェクトなし）は、どちらの読みでも同じ絵になり、道具のずれを見る
EFFECT_CONDITIONS: tuple[tuple[str, str, tuple[str, ...], float], ...] = (
    ("基準 図形", "shape", (), 100.0),
    ("反転 図形の周りが透明", "shape", ("invert",), 100.0),
    ("反転 不透明度 50", "shape", ("invert",), 50.0),
    ("前景を塗りつぶし", "shape", ("fill",), 100.0),
    ("基準 画面いっぱいの絵", "picture", (), 100.0),
    ("拡大率 50", "picture", ("zoom",), 100.0),
    ("位置 X 300", "picture", ("move",), 100.0),
)
EFFECT_FRAMEBUFFER = "黒を敷いて上に描く"
EFFECT_FILTER = "下の絵に掛けて置き換える"
#: 2 つの読みの差がこれより小さい枠は見分けない（0〜255、色の平均 480x270 に縮めて）
EFFECT_SAME = 1.0


@dataclass(frozen=True)
class EffectItemSlot:
    """エフェクトアイテムの探りの枠 1 つ ``effects`` が空なら下の絵だけ"""

    name: str
    below: str
    effects: tuple[str, ...]
    opacity: float
    start: int


def build_effect_item_slots() -> list[EffectItemSlot]:
    slots: list[EffectItemSlot] = []
    cursor = 0
    for name, below, effects, opacity in EFFECT_CONDITIONS:
        slots.append(EffectItemSlot(name, below, effects, opacity, cursor))
        cursor += EFFECT_SLOT + GAP
    return slots


def _effect_entry(name: str, **values: Any) -> dict[str, Any]:
    return {
        "$type": f"YukkuriMovieMaker.Project.Effects.{name}, YukkuriMovieMaker",
        **values,
        "IsEnabled": True,
        "Remark": "",
    }


def probe_effect_entry(key: str) -> dict[str, Any]:
    """探りのエフェクト 1 つ 形は ``.work/probes/samples.json`` と配布物の実物から写した

    前景の塗りつぶしは配布物（トーン調整Te）の 13 個が使う形 実物は ``Overlay`` だが、
    黒の上の ``Overlay`` は黒のままで、黒を敷いたかどうかが絵に出ない ``Normal`` にする
    """
    if key == "invert":
        return _effect_entry("InvertEffect")
    if key == "zoom":
        return _effect_entry(
            "ZoomEffect",
            Zoom=_still(50.0),
            ZoomX=_still(100.0),
            ZoomY=_still(100.0),
            IsNearestNeighbor=False,
        )
    if key == "move":
        return _effect_entry("DrawPositionEffect", X=_still(300.0), Y=_still(0.0), Z=_still(0.0))
    if key == "fill":
        return _effect_entry(
            "FillForegroundEffect",
            Opacity=_still(50.0),
            BlendMode="Normal",
            IsBrushOnly=False,
            Brush={
                "Type": "YukkuriMovieMaker.Plugin.Brush.SolidColorBrushPlugin, YukkuriMovieMaker",
                "Parameter": {"$type": _BRUSH_PARAMETER, "Color": "#FF2C7AE0"},
            },
        )
    raise ValueError(f"探りに無いエフェクト {key}")


def effect_item(slot: EffectItemSlot, layer: int) -> dict[str, Any]:
    """探りのエフェクトアイテム 項目の並びは配布物（トーン調整Te）の実物の ``EffectItem`` から写した

    範囲は実物 30 個すべてと同じ画面全体（``BackgroundShapePlugin``）
    """
    return {
        "$type": "YukkuriMovieMaker.Project.Items.EffectItem, YukkuriMovieMaker",
        "ShapeType2": "YukkuriMovieMaker.Shape.BackgroundShapePlugin, YukkuriMovieMaker",
        "ShapeParameter": {
            "$type": "YukkuriMovieMaker.Project.Items.BackgroundShapeParameter, YukkuriMovieMaker",
            "StrokeThickness": _still(4000.0),
            "Brush": {
                "Type": "YukkuriMovieMaker.Plugin.Brush.SolidColorBrushPlugin, YukkuriMovieMaker",
                "Parameter": {"$type": _BRUSH_PARAMETER, "Color": "#FFFFFFFF"},
            },
        },
        "Blur": _still(0.0),
        "InvertMask": False,
        "VideoEffects": [probe_effect_entry(key) for key in slot.effects],
        "X": _still(0.0),
        "Y": _still(0.0),
        "Z": _still(0.0),
        "Opacity": _still(slot.opacity),
        "Rotation": _still(0.0),
        "FadeIn": 0.0,
        "FadeOut": 0.0,
        "Blend": "Normal",
        "IsClippingWithObjectAbove": False,
        "IsAlwaysOnTop": False,
        "IsZOrderEnabled": False,
        "Group": 0,
        "Frame": slot.start,
        "Layer": layer,
        "KeyFrames": {"Frames": [], "Count": 0},
        "Length": EFFECT_SLOT,
        "PlaybackRate": 100.0,
        "ContentOffset": "00:00:00",
        "Remark": slot.name,
        "IsLocked": False,
        "IsHidden": False,
    }


def effect_items(slot: EffectItemSlot, picture: Path) -> list[dict[str, Any]]:
    """枠 1 つのアイテム 下の絵を 0 段、エフェクトアイテムを 1 段に置く"""
    if slot.below == "shape":
        below = base_shape(slot.start, 0, EFFECT_SLOT)
    else:
        below = image_item(picture, frame=slot.start, length=EFFECT_SLOT, remark=slot.name)
    if not slot.effects:
        return [below]
    return [below, effect_item(slot, 1)]


def read_effect_items_as(kind: str, objects: list[MappedObject]) -> list[MappedObject]:
    """エフェクトアイテムを写したクリップを ``kind``（``framebuffer`` か ``filter``）で読み直す

    同じ枠を 2 通りに描いて、YMM4 の書き出しに近い方を選ぶため
    """
    from dataclasses import replace

    from sashimono.core.model import FILTER_KIND, GeneratedSource

    readings = ("framebuffer", FILTER_KIND)
    return [
        replace(item, clip=replace(item.clip, source=GeneratedSource(kind=kind)), kind=kind)
        if item.clip.source is not None and item.clip.source.kind in readings
        else item
        for item in objects
    ]


def effect_item_manifest(slots: list[EffectItemSlot], picture: Path) -> dict[str, Any]:
    """枠の一覧 ``effectitem-measure`` はこれだけを見て切り出し、Sashimono でも描く"""
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "slots": [
            {
                "index": index,
                "name": slot.name,
                "below": slot.below,
                "effects": list(slot.effects),
                "opacity": slot.opacity,
                "start": slot.start,
                "length": EFFECT_SLOT,
                "items": effect_items(slot, picture),
            }
            for index, slot in enumerate(slots)
        ],
    }


def _clear_effect_item_results(work: Path) -> None:
    """前の測り結果を捨てる 残すと、新しい枠の一覧に対応しない表や絵を読んでしまう"""
    (work / "effectitem-report.json").unlink(missing_ok=True)
    images = work / "images"
    if images.is_dir():
        for old in images.glob("effectitem-*.png"):
            old.unlink()


def command_effect_item_build(arguments: argparse.Namespace) -> int:
    # 絶対パスにしてから書く 相対のまま `.ymmp` へ書くと YMM4 が絵を見つけられない
    work: Path = arguments.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    _clear_effect_item_results(work)
    # 画面いっぱいの絵は格子の探りと同じ下地 位置で色が変わるので、縮めた・ずらした
    # 絵が元の絵の上に重なったか（黒を敷いた読み）、黒の上に出たか（置き換える読み）が見える
    picture = work / "effectitem-probe-picture.png"
    _save_png(mesh_pattern(), picture)
    slots = build_effect_item_slots()
    items = [item for slot in slots for item in effect_items(slot, picture)]
    length = max(slot.start for slot in slots) + EFFECT_SLOT + GAP
    project = work / "effectitem-probe.ymmp"
    write_document(items, length, project)
    (work / "effectitem-probe.json").write_text(
        json.dumps(effect_item_manifest(slots, picture), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"{len(slots)} 枠を並べた（{length} フレーム、{length / FPS:.1f} 秒）")
    video = work / "effectitem-probe.mp4"
    if video.exists():
        print(f"{video} は前の探りの書き出しです 作り直した方で書き出し直してください")
    print(f"YMM4 で {project} を開き、{video} として書き出してください")
    print(f"書き出しは {WIDTH}x{HEIGHT}・{FPS}fps・頭から終わりまで（範囲を絞らない）")
    print_export_hint(project, video)
    return 0


def effect_item_reading(differences: dict[str, float]) -> str:
    """YMM4 との差が小さい方の読み 2 つの読みが同じ絵なら見分けない"""
    framebuffer, filtered = differences[EFFECT_FRAMEBUFFER], differences[EFFECT_FILTER]
    if abs(framebuffer - filtered) < EFFECT_SAME:
        return "見分けない"
    return EFFECT_FRAMEBUFFER if framebuffer < filtered else EFFECT_FILTER


def command_effect_item_measure(arguments: argparse.Namespace) -> int:
    from sashimono.core.model import FILTER_KIND

    work: Path = arguments.work
    manifest_path = work / "effectitem-probe.json"
    project = work / "effectitem-probe.ymmp"
    video = work / "effectitem-probe.mp4"
    _clear_effect_item_results(work)
    if _stale_export(video, manifest_path, project, "effectitem-build"):
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = manifest["slots"]
    theirs = _read_still_export(video, project, entries, int(manifest["fps"]), _shrink)
    if theirs is None:
        return 0
    readings = {
        EFFECT_FRAMEBUFFER: _render_still_slots(
            entries, adjust=partial(read_effect_items_as, "framebuffer")
        ),
        EFFECT_FILTER: _render_still_slots(
            entries, adjust=partial(read_effect_items_as, FILTER_KIND)
        ),
    }
    images = work / "images"
    images.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        a = theirs[index]
        drawn = {name: pictures[index] for name, pictures in readings.items()}
        row: dict[str, Any] = {"name": entry["name"]}
        if a is None or any(picture is None for picture in drawn.values()):
            row["reading"] = "絵が無い"
            rows.append(row)
            continue
        differences = {
            name: float(np.abs(a - picture).mean())
            for name, picture in drawn.items()
            if picture is not None
        }
        row["difference"] = differences
        row["reading"] = effect_item_reading(differences)
        side = np.concatenate([a, *[p for p in drawn.values() if p is not None]], axis=1)
        _save_png(np.clip(side, 0, 255).astype(np.uint8), images / f"effectitem-{index:02d}.png")
        rows.append(row)
    print(f"{'枠':<22}{'黒を敷く読みとの差':>18}{'置き換える読みとの差':>20}{'近い方':>22}")
    for row in rows:
        difference = row.get("difference")
        if difference is None:
            print(f"{row['name']:<22}{'-':>18}{'-':>20}{row['reading']:>22}")
            continue
        print(
            f"{row['name']:<22}{difference[EFFECT_FRAMEBUFFER]:>18.1f}"
            f"{difference[EFFECT_FILTER]:>20.1f}{row['reading']:>22}"
        )
    print()
    print("差は 480x270 に縮めた絵の 0〜255 の平均 並べた絵は YMM4・黒を敷く・置き換えるの順")
    (work / "effectitem-report.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{work / 'effectitem-report.json'} へ書いた 並べた絵は {images} の effectitem-*.png")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("sources", type=Path, nargs="+")
    build.add_argument("--exclude", action="append", default=[], help="ファイル名に含む語で外す")
    build.add_argument(
        "--last",
        action="append",
        default=[],
        help="ファイル名に含む語で後ろへ回す（YMM4 が書き出しを止める物を後ろへ）",
    )
    build.add_argument(
        "--size",
        type=screen_size,
        default=(WIDTH, HEIGHT),
        help=f"画面の大きさ（既定 {WIDTH}x{HEIGHT}） ペンの点のように画面で意味の変わる物の探り用",
    )
    compare = commands.add_parser("compare")
    compare.add_argument("--only", default="")
    compare.add_argument("--top", type=int, default=30)
    compare.add_argument(
        "--every",
        action="store_true",
        help="枠のフレームをすべて比べ、枠ごとに一番大きい差を出す（既定は 3 枚だけ）",
    )
    # 既定は新しく作るプロジェクトと同じ sRGB（YMM4 の混ぜ方） リニアを選べば、
    # 設定ができる前に保存したプロジェクトの見え方で比べられる
    compare.add_argument("--blending", choices=("srgb", "linear"), default="srgb")
    compare.add_argument(
        "--output", type=Path, default=None, help="一覧と絵を書く所（既定は --work と同じ）"
    )
    compare.add_argument(
        "--ceilings",
        type=Path,
        default=CEILINGS,
        help="テンプレートごとの差の上限 超えたら終了コード 1",
    )
    compare.add_argument(
        "--edge-ceilings",
        type=Path,
        default=None,
        help=(
            f"テンプレートごとの縁の差の上限（書いてある物だけ見る 既定 {EDGE_CEILINGS.name}）"
            " 名指しして無ければ終了コード 1"
        ),
    )
    compare.add_argument(
        "--write-ceilings",
        action="store_true",
        help="測った差にゆとりを足して上限を書き換える（直して差が減ったときに下げる）",
    )
    commands.add_parser("audio-build")
    commands.add_parser("audio-measure")
    commands.add_parser("mesh-build")
    commands.add_parser("mesh-measure")
    commands.add_parser("video-rate-build")
    rate_measure = commands.add_parser("video-rate-measure")
    rate_measure.add_argument(
        "--skip-sashimono",
        action="store_true",
        help="Sashimono で描いて並べるのを飛ばし、YMM4 の書き出しだけを測る",
    )
    commands.add_parser("zoom-build")
    commands.add_parser("zoom-measure")
    commands.add_parser("effectitem-build")
    commands.add_parser("effectitem-measure")
    arguments = parser.parse_args()
    runners: dict[str, Callable[[argparse.Namespace], int]] = {
        "build": command_build,
        "compare": command_compare,
        "audio-build": command_audio_build,
        "audio-measure": command_audio_measure,
        "mesh-build": command_mesh_build,
        "mesh-measure": command_mesh_measure,
        "video-rate-build": command_video_rate_build,
        "video-rate-measure": command_video_rate_measure,
        "zoom-build": command_zoom_build,
        "zoom-measure": command_zoom_measure,
        "effectitem-build": command_effect_item_build,
        "effectitem-measure": command_effect_item_measure,
    }
    return runners[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
