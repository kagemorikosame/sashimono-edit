"""生成オブジェクトの定義 テキストと図形

素材を持たないクリップの中身 エフェクトと同じパラメータ仕様に載せてあるので、
設定 UI もプリセットも同じ実装で扱える

描画はここではなく :mod:`sashimono.engine.sources` が行う テキストの整形と縁取りは
Qt の描画系に任せるのが現実的で、その依存をこの層に持ち込みたくない
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sashimono.core.model import (
    FILTER_KIND,
    GROUP_AS_ONE,
    GROUP_KIND,
    GROUP_LAYERS,
    AnimatedValue,
    GeneratedSource,
    ParamValue,
)
from sashimono.effects.easing import EASING_KINDS, EASING_MODES
from sashimono.effects.spec import (
    CheckSpec,
    ColorSpec,
    FontSpec,
    ParameterSpec,
    ParamInput,
    SelectSpec,
    TextSpec,
    TrackSpec,
    ValueSpec,
)

__all__ = [
    "FILTER",
    "FRAMEBUFFER",
    "GROUP",
    "PREVIOUS_OBJECT",
    "SHAPE",
    "TEXT",
    "TRANSITION",
    "SourceDefinition",
    "source_registry",
]


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    """1 種類の生成オブジェクト"""

    kind: str
    label: str
    parameters: tuple[ParameterSpec, ...] = ()
    #: 今の値で使わない項目の名前を返す 設定パネルはこれを出さない（値は消さずに残す）
    #: 図形の種類や切り替え方で使う項目が変わる物だけが持つ 全部を並べると、図形の
    #: 設定に 60 近い欄が並び、どれを動かせば変わるのかが分からない
    unused: Callable[[Mapping[str, ParamValue]], frozenset[str]] | None = None

    def unused_names(self, params: Mapping[str, ParamValue]) -> frozenset[str]:
        """``params`` のときに使わない項目 :attr:`unused` が無ければ空"""
        return self.unused(params) if self.unused is not None else frozenset()

    def spec(self, name: str) -> ParameterSpec | None:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        return None

    def default_params(self) -> dict[str, ParamValue]:
        return {spec.name: spec.default_value() for spec in self.parameters}

    def create(self, **overrides: ParamInput) -> GeneratedSource:
        params = self.default_params()
        for name, value in overrides.items():
            spec = self.spec(name)
            if spec is not None:
                params[name] = spec.coerce(value)
        return GeneratedSource(kind=self.kind, params=params)


def _still_zero(value: ParamValue | None) -> bool:
    """動かない 0 か（キーフレームで動く値は、途中で 0 でなくなるので使う側に数える）"""
    if value is None:
        return True
    if isinstance(value, AnimatedValue):
        return not value.keyframes and value.static == 0
    return False


def _text_unused(params: Mapping[str, ParamValue]) -> frozenset[str]:
    """テキストの設定で使わない項目

    描く所（:mod:`sashimono.engine.sources` の ``_draw_text``）が読まない物だけ
    - タイマーの書式が空なら、タイマーの初めの値・速さ・数え下げ・長さを読まない
    - タイマーの書式があれば、文字の代わりに時間を出すので文字を読まない
    - 数え下げを切っていれば、数え下げる長さを読まない
    - 縦書きは行揃えと組み方を読まない（縦書きは標準の組み方で描く）
    - 縁取りの太さが 0 なら縁取りの色、影のずらしとぼかしが 0 なら影の色を読まない
      （文字の中の制御文字で縁や影を付けたときは、制御文字の色を使う）
    """
    unused: set[str] = set()
    if not str(params.get("timer_format", "") or ""):
        unused |= {"timer_start", "timer_rate", "timer_countdown", "timer_length"}
    else:
        unused.add("text")
        countdown = params.get("timer_countdown")
        if not (countdown is True or (isinstance(countdown, int) and countdown)):
            unused.add("timer_length")
    vertical = params.get("vertical")
    if vertical is True or (isinstance(vertical, int) and vertical):
        unused |= {"align", "layout"}
    if _still_zero(params.get("border_width")):
        unused.add("border_color")
    if all(_still_zero(params.get(name)) for name in ("shadow_x", "shadow_y", "shadow_blur")):
        unused.add("shadow_color")
    return frozenset(unused)


TEXT = SourceDefinition(
    kind="text",
    label="テキスト",
    unused=_text_unused,
    parameters=(
        TextSpec("text", "文字", "テキスト"),
        FontSpec("font", "フォント"),
        TrackSpec("size", "サイズ", 4, 512, 64, step=1, unit="px"),
        ColorSpec("color", "色", (1.0, 1.0, 1.0, 1.0)),
        CheckSpec("bold", "太字", False),
        CheckSpec("italic", "斜体", False),
        SelectSpec(
            "align",
            "行揃え",
            (("left", "左"), ("center", "中央"), ("right", "右")),
            "center",
        ),
        # AviUtl の「文字揃え」は横と縦の 2 つを 1 つにまとめた呼び方をする
        # （``中央揃え[下]`` など） こちらは別々に持つ まとめると、片方だけを
        # 変えたいときに全部の組み合わせを並べることになる
        SelectSpec(
            "valign",
            "縦の基準",
            (("top", "上"), ("middle", "中"), ("bottom", "下")),
            "middle",
        ),
        # 横の基準 行揃えは行どうしの揃え方で、文字の塊は位置を真ん中にして置く
        # YMM4 の基準位置（BasePoint）の左右は塊の端を位置に合わせる（左上なら左の端が位置）
        # 真ん中に置いたまま読むと、左上の 4 文字の H が 130 画素ほど左へずれた（#198）
        # 既定の ``center`` は今までの置き方のまま 既にある作品の見た目を変えない
        SelectSpec(
            "anchor",
            "横の基準",
            (("left", "左"), ("center", "中"), ("right", "右")),
            "center",
        ),
        TrackSpec("line_spacing", "行間", -50, 200, 0, step=1, unit="px"),
        TrackSpec("letter_spacing", "字間", -20, 100, 0, step=1, unit="px"),
        TrackSpec("border_width", "縁取りの太さ", 0, 64, 0, step=1, unit="px"),
        ColorSpec("border_color", "縁取りの色", (0.0, 0.0, 0.0, 1.0)),
        # 影は「文字の飾り」として文字と一緒に描く クリップ全体に掛ける
        # 影エフェクトとは別物で、こちらは 1 文字ずつの輪郭に付く
        TrackSpec("shadow_x", "影の X", -200, 200, 0, step=1, unit="px"),
        TrackSpec("shadow_y", "影の Y", -200, 200, 0, step=1, unit="px"),
        TrackSpec("shadow_blur", "影のぼかし", 0, 64, 0, step=1, unit="px"),
        ColorSpec("shadow_color", "影の色", (0.0, 0.0, 0.0, 1.0)),
        CheckSpec("vertical", "縦書き", False),
        TrackSpec("reveal", "文字送り", 0, 100, 100, step=1, unit="%"),
        # タイマー 書式が空でなければ、文字の代わりに時間を出す（YMM4 のタイマーの図形）
        # 書式は .NET の時間の書式（h m s f、\\ で文字をそのまま出す）
        TextSpec("timer_format", "タイマーの書式", "", multiline=False),
        TrackSpec("timer_start", "タイマーの初めの値", -360000, 360000, 0, step=0.01, unit="秒"),
        TrackSpec("timer_rate", "タイマーの速さ", -10000, 10000, 100, unit="%"),
        CheckSpec("timer_countdown", "数え下げる", False),
        ValueSpec("timer_length", "数え下げる長さ", 0, minimum=0, maximum=10**9),
        TrackSpec("pos_x", "X", -4000, 4000, 0, step=1, unit="px"),
        TrackSpec("pos_y", "Y", -4000, 4000, 0, step=1, unit="px"),
        # 組み方 AviUtl2 から読んだテキストだけが ``aviutl`` を持つ
        # AviUtl2 はテキストの入れ物を字の形ではなく文字の枠（送り幅 x 行の高さ）にし、
        # 太字の太らせ方も Qt と違う 画像合成・縁取りの模様・万華鏡・オブジェクト分割は
        # この入れ物を基準に動くので、字の形で代わりにすると位置も大きさもずれる（#64）
        # 既定の ``native`` は今までの組み方のまま 既にある作品の見た目を変えない
        SelectSpec(
            "layout",
            "組み方",
            (("native", "標準"), ("aviutl", "AviUtl2 と同じ")),
            "native",
        ),
    ),
)


#: 移動軌跡の先端と星空の粒に使える形 AviUtl の図形（``obj.load("figure")``）に当たる
#: 三角形は円に内接する形 AviUtl2 の先端を測ると、大きさ 48 で高さ 36・底辺 41 だった
_FIGURE_CHOICES = (
    ("ellipse", "円"),
    ("rect", "四角形"),
    ("inscribed_triangle", "三角形"),
    ("pentagon", "五角形"),
    ("hexagon", "六角形"),
    ("star", "星型"),
)


#: 図形の種類ごとの専用の項目 描く所（:mod:`sashimono.engine.sources`）が読む物だけ
_SHAPE_OWN: dict[str, frozenset[str]] = {
    "rounded": frozenset({"corner_radius"}),
    "fan": frozenset({"span"}),
    "arrow": frozenset({"bar_length", "bar_thickness"}),
    "superformula": frozenset({"formula_m", "formula_n"}),
    "polyline": frozenset(
        {"points", "points_from", "line_type", "closed", "fill_color", "dash"}
        | {"trim_start", "trim_end"}
    ),
    "concentration": frozenset(
        {"density", "line_thickness", "line_length", "softness", "center_gap"}
        | {"fill_frame", "flicker"}
    ),
    "motion_trail": frozenset(
        {"trail_interval", "trail_min_step", "trail_core", "trail_band", "trail_speed"}
        | {"trail_head_size", "trail_head_angle", "trail_head_offset", "trail_head_shape"}
    ),
    "starfield": frozenset(
        {"star_count", "star_speed", "star_spread", "star_depth", "star_size", "star_shape"}
        | {"star_fade_in", "star_fade_out"}
    ),
    "waveform": frozenset(
        {"audio_path", "wave_volume", "wave_spectrum", "wave_mirror", "wave_columns"}
        | {"wave_rows", "wave_gap_x", "wave_gap_y", "audio_end_ms"}
    ),
}
#: 輪郭の形（パス）で描く図形だけが使う項目 線・集中線・軌跡・星空・音声波形は別の道で描く
_OUTLINED = frozenset({"outline_only", "line_align", "rotation"})
#: 専用の道で描く図形
_OWN_WAY = frozenset({"polyline", "concentration", "motion_trail", "starfield", "waveform"})


def _shape_unused(params: Mapping[str, ParamValue]) -> frozenset[str]:
    """図形の種類で使わない項目

    ほかの種類の専用の項目を隠す 大きさ・位置・線の太さも、描く所が読まない種類では隠す
    （背景は画面の全体、軌跡と星空は画面の座標、線は点の広がりで大きさが決まる）
    """
    shape = str(params.get("shape", "rect"))
    unused = set().union(*(own for kind, own in _SHAPE_OWN.items() if kind != shape))
    if shape in _OWN_WAY:
        unused |= _OUTLINED
    if shape in ("background", "polyline", "motion_trail", "starfield"):
        unused |= {"width", "height"}
    if shape == "concentration":
        # 集中線の大きさは幅だけで決まる（円の直径）
        unused |= {"height", "line_width"}
    if shape in ("motion_trail", "starfield"):
        unused |= {"pos_x", "pos_y"}
    if shape in ("starfield", "waveform"):
        unused.add("line_width")
    return frozenset(unused)


SHAPE = SourceDefinition(
    kind="shape",
    label="図形",
    unused=_shape_unused,
    parameters=(
        SelectSpec(
            "shape",
            "種類",
            (
                ("rect", "矩形"),
                ("rounded", "角丸矩形"),
                ("ellipse", "楕円"),
                ("triangle", "三角形"),
                ("pentagon", "五角形"),
                ("hexagon", "六角形"),
                ("star", "星"),
                ("background", "背景"),
                ("inscribed_triangle", "三角形（円に内接）"),
                ("fan", "扇"),
                ("arrow", "矢印"),
                ("superformula", "スーパーフォーミュラ"),
                ("polyline", "線"),
                ("concentration", "集中線"),
                ("motion_trail", "移動軌跡"),
                ("starfield", "星空"),
                ("waveform", "音声波形"),
            ),
            "rect",
        ),
        TrackSpec("width", "幅", 1, 8000, 400, step=1, unit="px"),
        TrackSpec("height", "高さ", 1, 8000, 400, step=1, unit="px"),
        ColorSpec("color", "色", (1.0, 1.0, 1.0, 1.0)),
        TrackSpec("corner_radius", "角の丸み", 0, 500, 24, step=1, unit="px"),
        TrackSpec("line_width", "線の太さ", 0, 200, 0, step=1, unit="px"),
        CheckSpec("outline_only", "線のみ", False),
        # 線をどこへ引くか 既定は AviUtl2 と同じ**内側**
        # 内側なら「幅」がそのまま外形になる 中央に引くと線の太さの半分
        # （太さ 20 で約 10 画素）外へはみ出し、幅 400 の図形が 420 に見える（#87）
        # 前の版で作った作品を開くと、輪郭だけの図形が線の太さぶん小さくなる
        # 元の見た目に戻したいときは ``輪郭の中央`` を選ぶ
        # YMM4 も内側に引くが、読み込む側が大きさを太さぶん縮めて写しているので、
        # `compat/ymm4/template.py` は ``center`` を明に渡す（二重に細るのを防ぐ）
        SelectSpec(
            "line_align",
            "線の位置",
            (("inside", "図形の内側"), ("center", "輪郭の中央")),
            "inside",
        ),
        TrackSpec("span", "扇の角度", 0, 360, 360, unit="度"),
        TrackSpec("bar_length", "矢印の軸の長さ", 0, 1000, 50, unit="%"),
        TrackSpec("bar_thickness", "矢印の軸の太さ", 0, 1000, 50, unit="%"),
        TrackSpec("formula_m", "スーパーフォーミュラ M", 0, 100, 4, step=0.1),
        TrackSpec("formula_n", "スーパーフォーミュラ N", 0.05, 100, 1, step=0.05),
        TextSpec("points", "線の点（x,y;x,y 中心から）", "", multiline=False),
        # 点の数え方 YMM4 のペンは画面の左上から数える（下が正） 画面の大きさは描くときに引く
        SelectSpec(
            "points_from",
            "線の点の基準",
            (("center", "中心から（上が正）"), ("corner", "画面の左上から（下が正）")),
            "center",
        ),
        SelectSpec(
            "line_type", "線の種類", (("straight", "直線"), ("quadratic", "2 次ベジェ")), "straight"
        ),
        CheckSpec("closed", "線を閉じる", False),
        ColorSpec("fill_color", "線の中の色", (1.0, 1.0, 1.0, 0.0)),
        TextSpec("dash", "破線（線の太さに対する長さ、カンマ区切り）", "", multiline=False),
        TrackSpec("trim_start", "線を描き始める位置", 0, 100, 0, unit="%"),
        TrackSpec("trim_end", "線を描き終える位置", 0, 100, 100, unit="%"),
        TrackSpec("density", "集中線の本数", 1, 1000, 80, step=1),
        # 100% で線が隙間なく並ぶ それより上は重なる（AviUtl の濃い集中線がここを使う）
        TrackSpec("line_thickness", "集中線の太さ", 0, 400, 50, unit="%"),
        TrackSpec("line_length", "集中線の長さ", 0, 100, 70, unit="%"),
        TrackSpec("softness", "集中線のぼかし", 0, 100, 50, unit="%"),
        TrackSpec("center_gap", "集中線の真ん中の空き", 0, 4000, 0, step=1, unit="px"),
        # AviUtl の集中線は画面いっぱい YMM4 のものは「幅」の円に収まる
        # 空きの有無で分けると、空きを 0 にした AviUtl の集中線が
        # 小さな円に縮んでしまうので、届く先は別の項目で持つ
        CheckSpec("fill_frame", "集中線を画面いっぱいに", False),
        TrackSpec("flicker", "集中線の切り替え", 0, 240, 5, unit="回/秒"),
        # 移動軌跡（AviUtl2 の ``ライン(移動軌跡)``） X と Y の動きをたどって線を引く
        # 線は ``線の太さ`` の円を一定の間隔で押して作る 間隔を広げると点線になる
        TrackSpec("trail_interval", "軌跡の点の間隔", 0, 1000, 10, unit="%"),
        TrackSpec("trail_min_step", "軌跡の点の最小間隔", 0, 1000, 2, step=1, unit="px"),
        TrackSpec("trail_core", "軌跡の点の大きさ", 0, 100, 100, unit="%"),
        TrackSpec("trail_band", "軌跡の点をつなぐ帯の太さ", 0, 100, 0, unit="%"),
        # 0 なら動きの時刻どおり 正の値なら、1 フレームにその画素ずつ道をたどって伸びる
        TrackSpec(
            "trail_speed", "軌跡の伸びる速さ（0 で動きどおり）", 0, 100, 0, unit="px/フレーム"
        ),
        TrackSpec("trail_head_size", "軌跡の先端の大きさ", 0, 500, 48, step=1, unit="px"),
        TrackSpec("trail_head_angle", "軌跡の先端の角度", 0, 360, 0, unit="度"),
        # 50% で図形の中心が今の位置に来る 大きいほど進む向きへ出る
        TrackSpec("trail_head_offset", "軌跡の先端の位置", -500, 500, 70, unit="%"),
        SelectSpec("trail_head_shape", "軌跡の先端の形", _FIGURE_CHOICES, "inscribed_triangle"),
        # 星空（AviUtl2 の ``星``） 粒が奥から手前へ流れてくる
        TrackSpec("star_count", "星の数", 1, 5000, 1500, step=1),
        # 負にすると手前から奥へ流れる
        TrackSpec("star_speed", "星の速さ", -50, 50, 6, step=0.1),
        TrackSpec("star_spread", "星の広がり", 0, 50, 12, step=0.1),
        TrackSpec("star_depth", "星の奥行き", 0, 50, 20, step=0.1),
        TrackSpec("star_size", "星の大きさ", 1, 100, 30, step=1, unit="px"),
        SelectSpec("star_shape", "星の形", _FIGURE_CHOICES, "ellipse"),
        TrackSpec("star_fade_in", "星のフェードイン", 0, 10, 0.15, step=0.01, unit="秒"),
        TrackSpec("star_fade_out", "星のフェードアウト", 0, 10, 0.15, step=0.01, unit="秒"),
        # 音声波形（AviUtl2 の ``音声波形表示``） 素材の音を今の時刻から 1 画素 1 サンプルで描く
        # クリップが素材を持てばそちらを使い、無ければこの道の音を読む
        TextSpec("audio_path", "音声波形の音声ファイル", "", multiline=False),
        TrackSpec("wave_volume", "音声波形の音量", 0, 500, 100, unit="%"),
        # 周波数ごとの大きさを下から塗る（AviUtl2 の スペクトラム表示）
        CheckSpec("wave_spectrum", "音声波形をスペクトラムにする", False),
        # スペクトラムの棒を上下の真ん中に置く（AviUtl2 の ミラー表示） 線には効かない
        CheckSpec("wave_mirror", "音声波形のスペクトラムを上下の真ん中に置く", False),
        # 0 なら 1 画素ずつ 数を決めると、その升目の数の絵に描いてから引き伸ばす
        TrackSpec("wave_columns", "音声波形の横の升目", 0, 4000, 0, step=1),
        TrackSpec("wave_rows", "音声波形の縦の升目", 0, 4000, 0, step=1),
        TrackSpec("wave_gap_x", "音声波形の升目の横のすき間", 0, 100, 0, unit="%"),
        TrackSpec("wave_gap_y", "音声波形の升目の縦のすき間", 0, 100, 0, unit="%"),
        # 素材のこのミリ秒より先は描かない（AviUtl の 再生範囲 の終わり） 負なら素材の終わりまで
        # 整数しか持てない項目なのでミリ秒 秒で持つと 80.448 秒が 80 秒に切れる
        ValueSpec(
            "audio_end_ms",
            "音声波形を読む終わり（ミリ秒 負で最後まで）",
            -1,
            minimum=-1,
            # 上限は設定画面の整数の欄が持てる所まで それでも 24 日を超える 10**10 に
            # していた頃は、設定画面へ出すたびに int のあふれの警告が出ていた（#148）
            maximum=2**31 - 1,
        ),
        TrackSpec("pos_x", "X", -4000, 4000, 0, step=1, unit="px"),
        TrackSpec("pos_y", "Y", -4000, 4000, 0, step=1, unit="px"),
        TrackSpec("rotation", "回転", -3600, 3600, 0, unit="度"),
    ),
)


#: それまでに重ねた画面を、そのまま素材として使う（YMM4 の ``FrameBufferItem``）
#: 下にある絵へぼかしや色調補正を掛けた帯を作るのに使われる 絵は CPU では作らず、
#: レンダラが GPU の中で写し取る（:mod:`sashimono.engine.render.renderer`）
#: ``transparent`` は何も無い所を透明のまま写す（AviUtl2 のフレームバッファ） 既定の偽は
#: YMM4 と同じく不透明な黒として写す（#143 反転で周りが白くなる）
FRAMEBUFFER = SourceDefinition(
    kind="framebuffer",
    label="フレームバッファ",
    parameters=(CheckSpec("transparent", "何も無い所を透明のまま写す", False),),
)


#: 下のトラックを重ね終えた絵へ、クリップのエフェクトを掛ける（AviUtl のフィルタオブジェクト）
#: 掛けた絵で下の絵を**置き換える** 上に重ねるフレームバッファと違い、黒を敷かないので
#: 透明な所は透明のまま残る（入れ子のシーンの中で使っても、外の絵を黒で隠さない）
#: 不透明度は掛ける前と後の混ぜ具合 合成方法は使わない（AviUtl のフィルタオブジェクトにも無い）
#: 絵はレンダラが GPU の中で作る（:mod:`sashimono.engine.render.renderer`）
FILTER = SourceDefinition(kind=FILTER_KIND, label="フィルタ")


#: 手前のレイヤーのオブジェクトを、1 本ずつ自分の配置と不透明度とエフェクトで動かす
#: （AviUtl の拡張編集のグループ制御） 自分では何も描かない 掛け方は
#: :mod:`sashimono.engine.render.groups` 対象レイヤー数の既定は AviUtl と同じ 1
GROUP = SourceDefinition(
    kind=GROUP_KIND,
    label="グループ制御",
    parameters=(
        ValueSpec(GROUP_LAYERS, "対象レイヤー数（0 で手前の全部）", 1, minimum=0, maximum=1000),
        CheckSpec(GROUP_AS_ONE, "1 枚の絵として扱う（重なりが透けない）", False),
    ),
)


#: すぐ下に重ねたクリップの絵を、自分の位置へ写す（AviUtl の ``直前オブジェクト``）
#: 写すのは下のクリップのエフェクトを掛けた絵で、下のクリップの描画の欄（反転・配置）と
#: 不透明度・合成方法は写さない 置く位置・大きさは自分の描画の欄で決める
#: AviUtl2 に描かせて、下の位置は足されず、下に掛けた単色化は写ることを確かめた（#195）
#: 絵はレンダラが GPU の中で作る（:mod:`sashimono.engine.render.renderer`）
PREVIOUS_OBJECT = SourceDefinition(kind="previous_object", label="直前オブジェクト")


def _transition_unused(params: Mapping[str, ParamValue]) -> frozenset[str]:
    """切り替え方で使わない項目

    向きは押し出しとスライドだけが動かす向き 動かす・手前にする場面はスライドと重ねるだけ
    イージングは進み具合を使う物だけ（切り替えは真ん中で入れ替わり、重ねるは進まない）
    """
    style = str(params.get("style", "fade"))
    unused: set[str] = set()
    if style not in ("push", "slide"):
        unused.add("angle")
    if style not in ("slide", "overlay"):
        unused.add("target")
    if style in ("switch", "overlay"):
        unused |= {"easing", "easing_mode"}
    return frozenset(unused)


#: 下のトラックの絵を、前の場面から後の場面へ切り替える（YMM4 の ``TransitionItem``）
#: 前の場面はクリップに掛けたエフェクト、後の場面は ``Clip.after_effects`` を通す
#: 絵はレンダラが GPU の中で作る（:mod:`sashimono.engine.render.renderer`）
TRANSITION = SourceDefinition(
    kind="transition",
    label="場面切り替え",
    unused=_transition_unused,
    parameters=(
        SelectSpec(
            "style",
            "切り替え方",
            (
                ("switch", "切り替え"),
                ("fade", "クロスフェード"),
                ("push", "押し出し"),
                ("slide", "スライド"),
                ("overlay", "重ねる"),
            ),
            "fade",
        ),
        TrackSpec("angle", "向き", -360, 360, 0, unit="度"),
        SelectSpec(
            "target", "動かす・手前にする場面", (("before", "前"), ("after", "後")), "after"
        ),
        SelectSpec("easing", "イージング", EASING_KINDS, "linear"),
        SelectSpec("easing_mode", "イージングの向き", EASING_MODES, "in"),
    ),
)


class SourceRegistry:
    """生成オブジェクトの一覧"""

    def __init__(self, definitions: tuple[SourceDefinition, ...]) -> None:
        self._definitions = {definition.kind: definition for definition in definitions}

    def get(self, kind: str) -> SourceDefinition | None:
        return self._definitions.get(kind)

    def all(self) -> tuple[SourceDefinition, ...]:
        return tuple(self._definitions.values())

    def __contains__(self, kind: object) -> bool:
        return kind in self._definitions


source_registry = SourceRegistry(
    (TEXT, SHAPE, FRAMEBUFFER, FILTER, GROUP, TRANSITION, PREVIOUS_OBJECT)
)
