r"""フィルタと中身の意味を AviUtl2 に描かせて測る（#184・#188・#192・#195）

    .venv\Scripts\python.exe tools\aviutl_filter_probes.py --work .work\aviutl-filters build
    .venv\Scripts\python.exe tools\aviutl2_export.py --project .work\aviutl-filters\compare.aup2 ^
        --output .work\aviutl-filters\aviutl --aviutl2 J:\aviutl2_v2.1.6a\aviutl2.exe
    .venv\Scripts\python.exe tools\aviutl_filter_probes.py --work .work\aviutl-filters measure

測るもの

- 文字装飾の縁と影の太さ（``縁取り文字`` などを Arial 100 の H で） 縦に伸ばした字の切れ方
- 斜めクリッピングの向きと幅、色調補正の各項目、レンズブラーの光の強さ、ぼかしとレンズブラーの
  サイズ固定
- 縁取りのぼかし
- 中身の フレームバッファ と 直前オブジェクト（下のレイヤーに置いた白い四角をどう写すか）

見本は合成の物だけ（白の図形と、その場で作る色の画像） 配布物は使わないので、測った数を試験の
定数へそのまま写せる 1 本の見本が 2 つのレイヤーを使うことがあるので、プロジェクトは
``tools/aviutl_compare.py`` の並べ方を借りずにここで書く（レイヤー 0 と 1 を同じ区間に置く）
AviUtl2 を起こす回数を減らすため、測るものを 1 つのプロジェクトにまとめてある
3 回目と 4 回目の見本は ``--third`` ``--fourth`` で選ぶ（どちらか 1 つ） 4 回目は ``--only`` で
名前の頭（``lb,po`` ``sc``）ごとに分けて書き出す 時間制御（``tc``）の見本は AviUtl2 を落とすので、
``--crashing`` を付けたときだけ並べる
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sashimono.compat.mapped import MappedObject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import aviutl_compare  # noqa: E402

from sashimono.compat.aviutl.report import CompatibilityReport  # noqa: E402
from sashimono.core.timebase import FrameRate  # noqa: E402

#: 見本の長さ（フレーム） 動かないので短くてよい
LENGTH = 6
#: 見本どうしの間
GAP = 4

#: 1 つのオブジェクトの節の並び（中身・フィルタ…・描画） 行は ``項目=値``
Blocks = list[list[str]]


@dataclass(frozen=True)
class Probe:
    """見本 1 本 ``layers`` はレイヤー 0 から順のオブジェクト"""

    name: str
    layers: tuple[Blocks, ...] = field(default_factory=tuple)
    #: 見本の長さ（フレーム） 時間で変わる物（時間制御・シーン・残像）は長くする
    length: int = LENGTH


#: レイヤーの頭に置くと、中身ではなくオブジェクトの見出し（``[N]`` の下）へ書く行の組
#: ``clipping.upper=1`` は 上のオブジェクトでクリッピング（本体の文字列から読んだ名前）
HEADER = "#header"
#: 空いたレイヤー 何も置かない
EMPTY: Blocks = []


def draw(x: float = 0.0, y: float = 0.0, zoom: float = 100.0) -> list[str]:
    """標準描画 AviUtl2 が書いたエイリアスと同じ並び"""
    return [
        "effect.name=標準描画",
        f"X={x:.2f}",
        f"Y={y:.2f}",
        "Z=0.00",
        "Group=1",
        "中心X=0.00",
        "中心Y=0.00",
        "中心Z=0.00",
        "Group3=1",
        "X軸回転=0.00",
        "Y軸回転=0.00",
        "Z軸回転=0.00",
        "Group2=1",
        f"拡大率={zoom:.3f}",
        "縦横比=0.000",
        "透明度=0.00",
        "合成モード=通常",
    ]


def square(size: int = 300, colour: str = "ffffff") -> list[str]:
    return [
        "effect.name=図形",
        "図形の種類=四角形",
        f"サイズ={size}",
        "縦横比=0.00",
        "ライン幅=4000",
        f"色={colour}",
        "角を丸くする=0",
    ]


def image(path: Path) -> list[str]:
    return ["effect.name=画像ファイル", f"ファイル={path}"]


def text(
    content: str, decoration: str = "標準文字", size: float = 100.0, edge: str = "ff0000"
) -> list[str]:
    """``tools/aviutl_tag_probes.py`` の見本と同じ項目（AviUtl2 v2.1.6a に作らせた形）"""
    return [
        "effect.name=テキスト",
        f"サイズ={size:.2f}",
        "字間=0.00",
        "行間=0.00",
        "表示速度=0.00",
        "フォント=Arial",
        "文字色=ffffff",
        f"影・縁色={edge}",
        f"文字装飾={decoration}",
        "文字揃え=中央揃え[中]",
        "B=0",
        "I=0",
        f"テキスト={content}",
        "文字毎に個別オブジェクト=0",
        "自動スクロール=0",
        "移動座標上に表示=0",
        "オブジェクトの長さを自動調節=0",
    ]


def filter_block(name: str, **values: object) -> list[str]:
    return [f"effect.name={name}", *(f"{key}={value}" for key, value in values.items())]


def single(*blocks: list[str]) -> tuple[Blocks, ...]:
    return (list(blocks),)


#: 文字装飾の名前（AviUtl2 の一覧の並び）
DECORATIONS = (
    "標準文字",
    "影付き文字",
    "影付き文字（薄）",
    "縁取り文字",
    "縁取り文字（細）",
    "縁取り文字（太）",
    "縁取り文字（角）",
)


def probes(folder: Path) -> tuple[Probe, ...]:
    patches = (folder / "patches.png").resolve()
    dots = (folder / "dots.png").resolve()
    blue = (folder / "blue.png").resolve()
    found: list[Probe] = []

    # --- #184 文字装飾の太さ Arial 100 の H ---
    for index, decoration in enumerate(DECORATIONS):
        found.append(Probe(f"fp01_deco{index}", single(text("H", decoration), draw())))
    found.append(Probe("fp02_border_size50", single(text("H", "縁取り文字", 50.0), draw())))
    found.append(Probe("fp03_border_size200", single(text("H", "縁取り文字", 200.0), draw())))
    # 縦に伸ばした字 文字の枠を青で塗って、枠のどこで切れるかを見る
    framed = text("H<th2>H<th>H")
    frame_fill = filter_block(
        "画像合成",
        X="0.0",
        Y="0.0",
        Group="1",
        拡大率="100.00",
        合成モード="後方から合成",
        画像=str(blue),
        ループ画像="1",
        ループ再生="1",
    )
    found.append(Probe("fp04_tall_th2", single(framed, frame_fill, draw())))
    # 1 回目は 影付き文字（薄） と 縁取り文字（細）（太）（角） が標準文字のまま描かれた
    # 設定欄の名前を読まなかったので、制御文字の番号（``<@Arial,番号>``）と半角の括弧で測り直す
    for number in range(1, 7):
        found.append(Probe(f"fp05_deco_tag{number}", single(text(f"<@Arial,{number}>H"), draw())))
    for index, decoration in enumerate(DECORATIONS[2:], start=2):
        half = decoration.replace("（", "(").replace("）", ")")
        found.append(Probe(f"fp06_deco_half{index}", single(text("H", half), draw())))

    # --- #188 斜めクリッピング 白い四角 300 の中心を通る線 ---
    def slant(angle: float, width: float = 0.0, blur: float = 0.0) -> list[str]:
        return filter_block(
            "斜めクリッピング",
            中心X="0.0",
            中心Y="0.0",
            角度=f"{angle:.1f}",
            ぼかし=f"{blur:.0f}",
            幅=f"{width:.0f}",
        )

    found.append(Probe("fp10_slant_a0", single(square(), slant(0.0), draw())))
    found.append(Probe("fp11_slant_a30", single(square(), slant(30.0), draw())))
    found.append(Probe("fp12_slant_a-30", single(square(), slant(-30.0), draw())))
    found.append(Probe("fp13_slant_a90", single(square(), slant(90.0), draw())))
    found.append(Probe("fp14_slant_a0_w100", single(square(), slant(0.0, 100.0), draw())))
    found.append(Probe("fp15_slant_a0_wm100", single(square(), slant(0.0, -100.0), draw())))

    # --- #188 色調補正 3 色の升（灰 128・橙・青） ---
    neutral = {
        "明るさ": "100.0",
        "コントラスト": "100.0",
        "色相": "0.0",
        "輝度": "100.0",
        "彩度": "100.0",
        "飽和する": "0",
    }
    for label, change in (
        ("neutral", {}),
        ("bright150", {"明るさ": "150.0"}),
        ("bright50", {"明るさ": "50.0"}),
        ("contrast150", {"コントラスト": "150.0"}),
        ("luma150", {"輝度": "150.0"}),
        ("sat50", {"彩度": "50.0"}),
        ("hue90", {"色相": "90.0"}),
    ):
        values = {**neutral, **change}
        found.append(
            Probe(
                f"fp20_color_{label}",
                single(image(patches), filter_block("色調補正", **values), draw()),
            )
        )
    # 何も掛けない升 色の基準
    found.append(Probe("fp29_color_plain", single(image(patches), draw())))

    # --- #188 レンズブラーの光の強さ 黒地に白い点 ---
    for strength in (0, 10, 25, 50, 100):
        found.append(
            Probe(
                f"fp30_lens_light{strength}",
                single(
                    image(dots),
                    filter_block("レンズブラー", 範囲="20", 光の強さ=str(strength), サイズ固定="0"),
                    draw(),
                ),
            )
        )
    # サイズ固定 白い四角の外へにじむかどうか
    for fixed in (0, 1):
        found.append(
            Probe(
                f"fp33_blur_fixed{fixed}",
                single(
                    square(),
                    filter_block(
                        "ぼかし", 範囲="30", 縦横比="0.0", 光の強さ="0", サイズ固定=str(fixed)
                    ),
                    draw(),
                ),
            )
        )
        found.append(
            Probe(
                f"fp35_lens_fixed{fixed}",
                single(
                    square(),
                    filter_block("レンズブラー", 範囲="30", 光の強さ="0", サイズ固定=str(fixed)),
                    draw(),
                ),
            )
        )

    # --- #192 縁取りのぼかし ---
    for blur in (0, 5, 20):
        found.append(
            Probe(
                f"fp40_border_blur{blur}",
                single(
                    square(),
                    filter_block("縁取り", サイズ="10", ぼかし=str(blur), 縁色="ff0000"),
                    draw(),
                ),
            )
        )

    # --- #195 中身 下のレイヤーの白い四角（左）をどう写すか ---
    below = [square(200), draw(x=-400.0)]
    found.append(
        Probe(
            "fp50_framebuffer",
            (
                below,
                [["effect.name=フレームバッファ", "フレームバッファをクリア=0"], draw(zoom=50.0)],
            ),
        )
    )
    found.append(
        Probe(
            "fp51_framebuffer_clear",
            (
                below,
                [["effect.name=フレームバッファ", "フレームバッファをクリア=1"], draw(zoom=50.0)],
            ),
        )
    )
    found.append(Probe("fp52_previous", (below, [["effect.name=直前オブジェクト"], draw(x=400.0)])))
    # 直前オブジェクト は下のオブジェクトのフィルタまで写すか 下の四角を赤く塗っておく
    painted = [square(200), filter_block("単色化", 強さ="100.0", 色="ff0000", 輝度を保持する="0")]
    found.append(
        Probe(
            "fp53_previous_filtered",
            ([*painted, draw(x=-400.0)], [["effect.name=直前オブジェクト"], draw(x=400.0)]),
        )
    )
    return tuple(found)


#: シーン 1 の長さ（フレーム） 白い四角が左から右へ動く
SCENE_LENGTH = 60
#: シーン 1 を読む見本の名前の頭
SCENE_PROBES = "sc"
#: 上のオブジェクトでクリッピングする見出しの行（本体の文字列から読んだ名前）
CLIP_UPPER = "clipping.upper=1"


def moving_square(colour: str = "ffffff") -> list[list[str]]:
    """左 -400 から右 400 へ直線で動く四角 時間の進み方を位置で読む"""
    moving = draw()
    moving[1] = "X=-400.00,400.00,直線移動,0"
    return [square(100, colour), moving]


def scene_one(first_number: int) -> list[str]:
    """シーン 1（白い四角が 60 フレームで左から右へ動く）の節 オブジェクトの番号は続きから"""
    lines = [
        "[scene.1]",
        "scene=1",
        "name=Probe",
        f"video.width={aviutl_compare.WIDTH}",
        f"video.height={aviutl_compare.HEIGHT}",
        f"video.rate={aviutl_compare.FPS}",
        "video.scale=1",
        f"audio.rate={aviutl_compare.AUDIO_RATE}",
        "cursor.frame=0",
        "cursor.layer=0",
        "preview.frame=0",
        "display.frame=0",
        "display.layer=0",
        "display.zoom=10000",
        # AviUtl2 が保存し直した形では 1 だった 4 回目に 1 にしても、中身はルートのシーンの
        # 頭に置かれた（並びの番号のせいではない）
        "display.order=1",
        "display.camera=",
        f"[{first_number}]",
        "layer=0",
        f"frame=0,{SCENE_LENGTH - 1}",
    ]
    for index, block in enumerate(moving_square()):
        lines.append(f"[{first_number}.{index}]")
        lines.extend(block)
    return lines


def probes_third(folder: Path) -> tuple[Probe, ...]:
    """3 回目（#188 #195 #210） レンズブラーの光の強さ・シーン・時間制御・直前オブジェクト"""
    greys = (folder / "greys.png").resolve()
    dot = (folder / "dot.png").resolve()
    found: list[Probe] = []
    # --- レンズブラーの光の強さ 7 段の灰の升（どの升も一様）と、黒地にひとつの白い点 ---
    for strength in (0, 25, 50, 100):
        found.append(
            Probe(
                f"lb01_greys_light{strength}",
                single(
                    image(greys),
                    filter_block("レンズブラー", 範囲="10", 光の強さ=str(strength), サイズ固定="1"),
                    draw(),
                ),
            )
        )
        found.append(
            Probe(
                f"lb02_dot_light{strength}",
                single(
                    image(dot),
                    filter_block("レンズブラー", 範囲="20", 光の強さ=str(strength), サイズ固定="1"),
                    draw(),
                ),
            )
        )
    # --- シーン シーン 1 を読む 項目の名前が分からないので 2 通り ---
    for label, key in (("ascii", "scene"), ("jp", "シーン")):
        found.append(
            Probe(
                f"sc01_scene_{label}",
                (
                    [
                        ["effect.name=シーン", f"{key}=1", "再生位置=0.000", "再生速度=100.00"],
                        draw(),
                    ],
                ),
                length=SCENE_LENGTH,
            )
        )
    found.append(
        Probe(
            "sc02_scene_half_speed",
            ([["effect.name=シーン", "scene=1", "シーン=1", "再生速度=50.00"], draw()],),
            length=SCENE_LENGTH,
        )
    )

    # --- 直前オブジェクト（#195 PR #219 の決めた点） 下の四角は左 -400、写しは右 400 ---
    def below(**draw_changes: str) -> list[list[str]]:
        placed = draw(x=-400.0)
        for key, value in draw_changes.items():
            placed = [f"{key}={value}" if line.startswith(f"{key}=") else line for line in placed]
        return [square(200), placed]

    previous = [["effect.name=直前オブジェクト"], draw(x=400.0)]
    found.append(Probe("po01_plain", (below(), previous)))
    found.append(Probe("po02_below_opacity50", (below(透明度="50.00"), previous)))
    found.append(Probe("po03_below_add", (below(合成モード="加算"), previous)))
    found.append(Probe("po04_empty_layer_between", (below(), EMPTY, previous)))
    found.append(
        Probe(
            "po05_framebuffer_below",
            (
                below(),
                [
                    ["effect.name=フレームバッファ", "フレームバッファをクリア=0"],
                    draw(y=-250.0, zoom=50.0),
                ],
                previous,
            ),
        )
    )
    # 上のオブジェクトでクリッピング 下の四角の上に縦長の細い四角を置き、その形で切る
    found.append(
        Probe(
            "po06_below_clipped",
            (
                below(),
                [[HEADER, "clipping.upper=1"], square(200), draw(x=-400.0)],
                previous,
            ),
        )
    )
    # 残像（モーションブラーの残像）がある下の四角
    moving = moving_square()
    moving.insert(1, filter_block("モーションブラー", 間隔="1.00", 分解能="10", 残像="1"))
    found.append(Probe("po07_below_afterimage", (moving, previous), length=60))
    # --- 時間制御(オブジェクト) 下の動く四角の時間をどう変えるか 最後に置く ---
    # 1 回目の書き出しは 位置=0.500（動かない値）の頭で AviUtl2 が落ちた（例外の自動控えが残った）
    # 落ちても前の見本が残るよう、ここを最後にし、落ちた並びは一番後ろへ回す
    for label, position in (
        ("linear_double", "0.000,2.000,直線移動,0"),
        ("range", "0.000,1.000,再生範囲,0"),
        ("still_half", "0.500"),
    ):
        found.append(
            Probe(
                f"tc01_{label}",
                (
                    [
                        [
                            "effect.name=時間制御(オブジェクト)",
                            f"位置={position}",
                            "繰り返し=0",
                            "コマ落ち=0",
                            "対象レイヤー数=1",
                        ]
                    ],
                    moving_square(),
                ),
                length=60,
            )
        )
    return tuple(found)


#: シーンの見本を並べるとき、頭に空ける長さ 節の書き方が違ってシーン 1 の中身がルートの頭に
#: 置かれても、ほかの見本と重ならず、ここに四角が出ることで分かる
SCENE_LEAD = SCENE_LENGTH
#: AviUtl2 を落とすと分かっている見本の名前の頭 3 回目も 4 回目も、時間制御(オブジェクト) を
#: 並べたプロジェクトで AviUtl2 が落ちた 原因が分かるまで ``--crashing`` のときだけ並べる
CRASHING_PROBES = ("tc",)


def probes_fourth(folder: Path) -> tuple[Probe, ...]:
    """4 回目（#195 #210） レンズブラーの光の強さ・直前オブジェクトの残り・シーン・時間制御

    名前の頭で分けて書き出す（``--only lb,po`` ``--only sc``）
    時間制御は 3 回目も 4 回目も AviUtl2 が落ちたので、``--crashing`` のときだけ並べる
    """
    dot = (folder / "dot.png").resolve()
    grey_dot = (folder / "grey_dot.png").resolve()
    found: list[Probe] = []
    # --- レンズブラーの光の強さ 黒地にひとつの点 3 回目は光の強さ 0 が別の絵と重なった ---
    for strength in (0, 10, 25, 50, 75, 100):
        found.append(
            Probe(
                f"lb03_dot_r20_light{strength}",
                single(
                    image(dot),
                    filter_block("レンズブラー", 範囲="20", 光の強さ=str(strength), サイズ固定="1"),
                    draw(),
                ),
            )
        )
    for size in (10, 40):
        found.append(
            Probe(
                f"lb04_dot_r{size}_light50",
                single(
                    image(dot),
                    filter_block("レンズブラー", 範囲=str(size), 光の強さ="50", サイズ固定="1"),
                    draw(),
                ),
            )
        )
    # 明るさで効き方が変わるか 点の明るさ 128
    for strength in (0, 50, 100):
        found.append(
            Probe(
                f"lb05_grey_dot_r20_light{strength}",
                single(
                    image(grey_dot),
                    filter_block("レンズブラー", 範囲="20", 光の強さ=str(strength), サイズ固定="1"),
                    draw(),
                ),
            )
        )

    # --- 直前オブジェクトの残り 下の四角は左 -400、写しは右 400 ---
    previous = [["effect.name=直前オブジェクト"], draw(x=400.0)]
    below_add = [square(200, "646464"), draw(x=-400.0)]
    below_add[1] = [line.replace("合成モード=通常", "合成モード=加算") for line in below_add[1]]
    # 灰 128 の下地の上で、下の四角（100）を加算にする 写しが加算なら 228、通常なら 100
    found.append(
        Probe(
            "po08_below_add_on_grey",
            ([square(1080, "808080"), draw()], below_add, previous),
        )
    )
    # 小さな青い四角（60）で下の白い四角（200）を切る 写しが切った絵なら 60、切る前なら 200
    found.append(
        Probe(
            "po09_below_clipped_small",
            (
                [square(60, "0000ff"), draw(x=-400.0)],
                [[HEADER, CLIP_UPPER], square(200), draw(x=-400.0)],
                previous,
            ),
        )
    )

    # --- シーン 項目の名前は 3 回目に AviUtl2 が保存し直した形 ---
    # 4 回目もシーン 1 の中身がルートに置かれ、どの見本にも何も写らなかった（節の書き方が違う）
    # 頭の空き（SCENE_LEAD）は見本にせず write_project が空ける 見本にすると ``--only sc03`` の
    # ように選んだときに外れ、選んだ見本がルートへ置かれた中身と重なる

    def scene(position: str = "0.000", speed: str = "100.00", loop: str = "0") -> Blocks:
        return [
            [
                "effect.name=シーン",
                f"再生位置={position}",
                f"再生速度={speed}",
                "シーン=1",
                f"ループ再生={loop}",
            ],
            draw(),
        ]

    found.append(Probe("sc03_plain", (scene(),), length=SCENE_LENGTH))
    found.append(Probe("sc04_half_speed", (scene(speed="50.00"),), length=SCENE_LENGTH))
    found.append(Probe("sc05_position_half", (scene(position="0.500"),), length=SCENE_LENGTH))
    found.append(Probe("sc06_double_loop", (scene(speed="200.00", loop="1"),), length=SCENE_LENGTH))

    # --- 時間制御(オブジェクト) 位置は AviUtl2 が保存し直した形の値だけ ---
    # 4 回目はこの 2 本だけでも、開いて主の窓を出す前に AviUtl2 が落ちた（3 回目は 位置=0.500）
    # 位置の値のせいではない 原因が分かるまで書き出さない
    for label, position in (
        ("linear_double", "0.000,2.000,直線移動,0"),
        ("range", "0.000,1.000,再生範囲,0"),
    ):
        found.append(
            Probe(
                f"tc02_{label}",
                (
                    [
                        [
                            "effect.name=時間制御(オブジェクト)",
                            f"位置={position}",
                            "繰り返し=0",
                            "コマ落ち=0",
                            "対象レイヤー数=1",
                        ]
                    ],
                    moving_square(),
                ),
                length=60,
            )
        )
    return tuple(found)


def greys_png(levels: tuple[int, ...] = (0, 32, 64, 128, 192, 224, 255), size: int = 60) -> bytes:
    """一様な灰の升を横に並べた絵 レンズブラーで升の真ん中が変わらなければ一様な所は保たれる"""
    row = b"".join(bytes((value, value, value)) * size for value in levels)
    return _png(size * len(levels), size, (b"\x00" + row) * size)


def chosen(found: tuple[Probe, ...], only: str, *, crashing: bool = False) -> tuple[Probe, ...]:
    """``--only`` で選んだ見本 名前の頭で選ぶ（``fp05,fp31`` 空なら全部）

    AviUtl2 を起こすたびに自動控えが回るので、足した見本だけを書き出せるようにする
    AviUtl2 を落とす見本（:data:`CRASHING_PROBES`）は ``crashing`` のときだけ残す
    ``--only tc`` だけで並べると、案内どおりに開いた人の AviUtl2 が落ちる
    """
    if not crashing:
        found = tuple(probe for probe in found if not probe.name.startswith(CRASHING_PROBES))
    heads = tuple(head.strip() for head in only.split(",") if head.strip())
    if not heads:
        return found
    return tuple(probe for probe in found if probe.name.startswith(heads))


def _png(width: int, height: int, pixels: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


#: 色調補正の升の色 灰は真ん中、橙と青は彩度と色相の動きが見える色
PATCH_COLOURS = ((128, 128, 128), (200, 100, 50), (50, 150, 220))
#: 升 1 つの大きさ（画素）
PATCH = 100


def patches_png() -> bytes:
    rows = []
    for _ in range(PATCH):
        row = b"".join(bytes(colour) * PATCH for colour in PATCH_COLOURS)
        rows.append(b"\x00" + row)
    return _png(PATCH * len(PATCH_COLOURS), PATCH, b"".join(rows))


def dots_png(
    width: int = 600, height: int = 300, span: int = 60, radius: int = 6, value: int = 255
) -> bytes:
    ys, xs = np.mgrid[0:height, 0:width]
    local_x = (xs % span) - span / 2
    local_y = (ys % span) - span / 2
    lit = (local_x**2 + local_y**2) <= radius**2
    pixels = np.where(lit[..., None], value, 0).astype(np.uint8).repeat(3, axis=2)
    rows = [b"\x00" + pixels[y].tobytes() for y in range(height)]
    return _png(width, height, b"".join(rows))


def blue_png() -> bytes:
    return _png(1, 1, b"\x00\x00\x00\xff")


def write_project(work: Path, chosen: tuple[Probe, ...]) -> dict[str, object]:
    """見本を並べた .aup2 と、見本の区間の一覧を書く"""
    target = work / "compare.aup2"
    out = [
        aviutl_compare._HEADER.format(
            file=target,
            width=aviutl_compare.WIDTH,
            height=aviutl_compare.HEIGHT,
            rate=aviutl_compare.FPS,
            audio_rate=aviutl_compare.AUDIO_RATE,
        )
    ]
    cases: list[dict[str, object]] = []
    number = 0
    scenes = any(probe.name.startswith(SCENE_PROBES) for probe in chosen)
    # シーン 1 の中身がルートの頭に置かれても重ならないよう、シーンの見本があれば頭を空ける
    cursor = SCENE_LEAD + GAP if scenes else 0
    for probe in chosen:
        end = cursor + probe.length - 1
        for layer, blocks in enumerate(probe.layers):
            if not blocks:
                continue
            out.append(f"[{number}]")
            out.append(f"layer={layer}")
            out.append(f"frame={cursor},{end}")
            body = list(blocks)
            if body[0] and body[0][0] == HEADER:
                out.extend(body[0][1:])
                body = body[1:]
            for index, block in enumerate(body):
                out.append(f"[{number}.{index}]")
                out.extend(block)
            number += 1
        cases.append(
            {"name": probe.name, "start": cursor, "length": probe.length, "layers": probe.layers}
        )
        cursor += probe.length + GAP
    # シーン 1 の節は、シーンを読む見本を並べたときだけ足す 書き方がまだ違っていて、中身が
    # ルートのシーンの頭に置かれる（#221） 足すと頭の 60 フレームの見本に動く四角が重なる
    if scenes:
        out.extend(scene_one(number))
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    manifest: dict[str, object] = {"cases": cases}
    (work / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


def frame_count(manifest: dict[str, Any]) -> int:
    """並べたプロジェクトの長さ（フレーム） 最後の見本の終わりまで

    本数 x 6 で数えると、シーンの頭の空きや 60 フレームの見本の分が抜け、書き出す長さを
    短く案内する
    """
    cases = manifest["cases"]
    return max((int(case["start"]) + int(case["length"]) for case in cases), default=0)


def command_build(
    work: Path,
    only: str = "",
    *,
    third: bool = False,
    fourth: bool = False,
    crashing: bool = False,
) -> int:
    if third and fourth:
        # 片方だけを黙って選ぶと、頼んだ見本の一部が無いプロジェクトを出して正常に終わる
        raise ValueError("--third と --fourth は同時に選べない")
    folder = work / "probes"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "patches.png").write_bytes(patches_png())
    (folder / "dots.png").write_bytes(dots_png())
    (folder / "blue.png").write_bytes(blue_png())
    (folder / "greys.png").write_bytes(greys_png())
    (folder / "dot.png").write_bytes(dots_png(200, 200, 200, 6))
    (folder / "grey_dot.png").write_bytes(dots_png(200, 200, 200, 6, value=128))
    builder = probes_fourth if fourth else probes_third if third else probes
    source = builder(folder)
    picked = chosen(source, only, crashing=crashing)
    if not picked:
        # 前に書いたプロジェクトを消す 残すと、--crashing で並べた落ちるプロジェクトを
        # 選び直しに失敗したあとでも案内どおりに開けてしまう
        for stale in ("compare.aup2", "manifest.json"):
            (work / stale).unlink(missing_ok=True)
        print(f"--only {only} に当たる見本がありません（時間制御は --crashing のときだけ）")
        return 1
    if any(probe.name.startswith(CRASHING_PROBES) for probe in picked):
        print("!!! AviUtl2 を落とすと分かっている見本（時間制御）を並べた 開くと落ちる")
    manifest = write_project(work, picked)
    print(f"{len(picked)} 本を並べた（{frame_count(manifest)} フレーム）")
    print(
        f"AviUtl2 で {work / 'compare.aup2'} を開き、{work / aviutl_compare.PNG_FOLDER} へ書き出す"
    )
    return 0


def object_text(blocks: Blocks, length: int = LENGTH) -> str:
    """1 レイヤーの見本を Sashimono の読み込みへ渡す ``.object`` の形 長さは見本の長さ"""
    lines = ["[Object]", f"frame=0,{length - 1}"]
    if blocks and blocks[0] and blocks[0][0] == HEADER:
        blocks = blocks[1:]
    for index, block in enumerate(blocks):
        lines.append(f"[Object.{index}]")
        lines.extend(block)
    return "\r\n".join(lines) + "\r\n"


def layer_objects(
    case: dict[str, Any], scratch: Path, rate: FrameRate, report: CompatibilityReport
) -> list[MappedObject]:
    """見本 1 本のレイヤーを Sashimono の読み込みで写す レイヤーは下から 1 番

    長さは見本ごとの長さ（時間で変わる見本は 60 フレーム） 決まった 6 フレームで書くと、
    6 フレームより後ろを比べるときに Sashimono の側だけ何も無い
    見出しの ``clipping.upper=1`` は読み込み（エイリアスの形）に無いので、ここでクリップへ移す
    移さないと、AviUtl2 が切り抜いた見本を Sashimono だけ切らずに描く
    """
    from dataclasses import replace

    from sashimono.compat.aviutl.exo import load_exo
    from sashimono.compat.aviutl.mapping import map_object

    length = int(case.get("length", LENGTH))
    objects: list[MappedObject] = []
    for layer, blocks in enumerate(case["layers"]):
        path = scratch / f"{case['name']}_{layer}.object"
        path.write_bytes(object_text(blocks, length).encode("utf-8"))
        clipped = (
            bool(blocks)
            and bool(blocks[0])
            and blocks[0][0] == HEADER
            and (CLIP_UPPER in blocks[0][1:])
        )
        for obj in load_exo(path).objects:
            mapped = map_object(obj, rate, report=report)
            if mapped is None:
                continue
            if clipped:
                mapped = replace(mapped, clip=replace(mapped.clip, clip_to_below=True))
            objects.append(replace(mapped, layer=layer + 1))
    return objects


def command_measure(work: Path) -> int:
    """AviUtl2 の書き出しと Sashimono の絵で、見本ごとの明るい外形と升の色を並べる"""
    from sashimono.core.model import ProjectSettings
    from sashimono.engine.render import FrameRenderer

    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    frames = {
        case["name"]: int(case["start"]) + int(case.get("length", LENGTH)) // 2 for case in cases
    }
    references = aviutl_compare.reference_frames(work, set(frames.values()))
    if references is None:
        print(f"{work / aviutl_compare.PNG_FOLDER} に書き出しがありません 先に書き出してください")
        return 1
    settings = ProjectSettings(
        width=aviutl_compare.WIDTH,
        height=aviutl_compare.HEIGHT,
        frame_rate=FrameRate(aviutl_compare.FPS),
        sample_rate=aviutl_compare.AUDIO_RATE,
    )
    report = CompatibilityReport()
    scratch = work / "objects"
    scratch.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    pictures: dict[str, np.ndarray] = {}
    renderer: FrameRenderer | None = None
    try:
        for case in cases:
            name = str(case["name"])
            frame = frames[name]
            reference = references.get(frame)
            if reference is None:
                print(f"{name}: 書き出しにフレーム {frame} がありません")
                return 1
            objects = layer_objects(case, scratch, settings.frame_rate, report)
            project, _ = aviutl_compare.placed_project(objects, settings, int(case["start"]))
            if renderer is None:
                renderer = FrameRenderer(project)
            else:
                renderer.set_project(project)
            ours = np.asarray(renderer.render(frame))[..., :3]
            theirs = np.asarray(reference)[..., :3]
            pictures[f"{name}_aviutl"] = theirs
            pictures[f"{name}_sashimono"] = ours
            row: dict[str, object] = {
                "name": name,
                "aviutl": describe(theirs),
                "sashimono": describe(ours),
            }
            rows.append(row)
            print(name)
            print(f"  AviUtl2   {row['aviutl']}")
            print(f"  Sashimono {row['sashimono']}")
    finally:
        if renderer is not None:
            renderer.close()
    (work / "measure.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
    # 絵の名前はどれも `<ケース>_aviutl` の形で、savez の引数（allow_pickle）とは重ならない
    # 型の上では重なりうるので、名前と絵の組として渡す
    arrays: dict[str, Any] = dict(pictures)
    np.savez_compressed(work / "measure.npz", **arrays)
    for line in report.lines():
        print(f"  記録: {line}")
    return 0


def _box(mask: np.ndarray) -> list[int] | None:
    rows = np.nonzero(mask.any(axis=1))[0]
    columns = np.nonzero(mask.any(axis=0))[0]
    if rows.size == 0 or columns.size == 0:
        return None
    return [int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1]


def describe(image: np.ndarray) -> dict[str, object]:
    """明るい所・白・赤の外形と、升 3 つの真ん中の色（色調補正の見本用）"""
    pixels = image.astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    height, width = image.shape[:2]
    centre_y = height // 2
    patches = []
    for index in range(len(PATCH_COLOURS)):
        centre_x = width // 2 + (index - 1) * PATCH
        spot = image[centre_y - 10 : centre_y + 10, centre_x - 10 : centre_x + 10]
        patches.append([round(float(v), 1) for v in spot.reshape(-1, 3).mean(axis=0)])
    return {
        "lit": _box(image.max(axis=2) >= 128),
        "white": _box((red > 200) & (green > 200) & (blue > 200)),
        "red": _box((red > 100) & (green < 90) & (blue < 90)),
        "blue": _box((blue > 150) & (red < 80) & (green < 80)),
        "patches": patches,
        "mean": round(float(image.mean()), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True, help="作業フォルダ")
    parser.add_argument("command", choices=("build", "measure"))
    parser.add_argument(
        "--only", default="", help="並べる見本の名前の頭（fp05,fp31 のように , で区切る）"
    )
    # 3 回目と 4 回目は同時に選べない 片方を黙って捨てず、使い方の誤りとして止める
    rounds = parser.add_mutually_exclusive_group()
    rounds.add_argument(
        "--third",
        action="store_true",
        help="3 回目の見本（レンズブラーの光の強さ・シーン・時間制御・直前オブジェクト）を並べる",
    )
    rounds.add_argument(
        "--fourth",
        action="store_true",
        help="4 回目の見本（レンズブラーの光の強さ・直前オブジェクトの残り・シーン）",
    )
    parser.add_argument(
        "--crashing",
        action="store_true",
        help="AviUtl2 を落とすと分かっている見本（時間制御）も並べる 落ちる所を確かめるときだけ",
    )
    arguments = parser.parse_args(argv)
    # 丸ごとの道にする 相対のまま渡すと、プロジェクトに書く自分の道（``file=``）が相対になる
    work: Path = arguments.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if arguments.command == "build":
        return command_build(
            work,
            arguments.only,
            third=arguments.third,
            fourth=arguments.fourth,
            crashing=arguments.crashing,
        )
    return command_measure(work)


if __name__ == "__main__":
    raise SystemExit(main())
