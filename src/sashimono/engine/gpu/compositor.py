"""GPU での映像合成

素材は sRGB で符号化されているので、テクスチャを ``GL_SRGB8_ALPHA8`` で作って
サンプリング時に自動でリニアへ戻す 合成は ``RGBA16F`` のフレームバッファ上で行い、
最後に sRGB へ符号化して出す

キャンバスに溜める値は、重ね合わせの方法（Issue #65）で 2 通りある

- リニア（``encoded=False``） 光の量で混ぜる 半透明の所が明るく出る
- sRGB（``encoded=True``） 描く直前に符号化し、符号化した値のまま混ぜる AviUtl2 と
  YMM4 がこちらで、黒の上に 50% の白を重ねると 128 になる（リニアでは 188）

どちらでも、描く絵（テクスチャやエフェクトの結果）はリニアで渡す 事前乗算で渡す絵
（``premultiplied=True``）は合成先のキャンバスかその写しで、同じ方法の値が入っている
ものとして扱う エフェクトはどちらでもリニアで動く（:class:`EffectProcessor`）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from sashimono.effects.blending import BLEND_FUNCTIONS, blend_index
from sashimono.effects.sampling import AREA_SAMPLING
from sashimono.engine.gpu.glutil import (
    FULL_RECT,
    IDENTITY,
    MAPPED_VERTEX_SHADER,
    VERTEX_SHADER,
    Framebuffer,
    Program,
    ScreenQuad,
    Texture,
)
from sashimono.engine.gpu.projection import Corners, homography, project, rotate, to_clip

__all__ = [
    "BlendMode",
    "Compositor",
    "Corners",
    "Placement",
    "Texture",
    "Transform",
    "fit_placement",
]


#: リニアと sRGB の行き来 IEC 61966-2-1 の定義そのまま
_SRGB_FUNCTIONS = """
vec3 srgb_encode(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 srgb_decode(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
"""

_FRAGMENT_SHADER = (
    """
#version 430 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
uniform float u_opacity;
// 事前乗算アルファで溜まった絵（入れ子のシーンのキャンバス）を渡すとき
uniform bool u_premultiplied;
// キャンバスが sRGB で符号化した値を溜めているか（重ね合わせを sRGB で行う）
uniform bool u_encoded;
"""
    + _SRGB_FUNCTIONS
    + AREA_SAMPLING
    + """
void main() {
    // sRGB テクスチャなので、この時点で値はリニア 縮めて置くときは事前乗算で平均する
    // GL の補間のままだと、透明な所に残った色が縁へにじむ（#179）
    vec4 color = area_premul(u_texture, v_uv, dFdx(v_uv), dFdy(v_uv), u_premultiplied);
    color = color.a > 0.0001 ? vec4(color.rgb / color.a, color.a) : vec4(0.0);
    // 事前乗算の絵はキャンバスの写しで、もう符号化されている もう 1 度掛けると白っぽく浮く
    if (u_encoded && !u_premultiplied) color.rgb = srgb_encode(color.rgb);
    frag_color = vec4(color.rgb, color.a * u_opacity);
}
"""
)

_RESOLVE_FRAGMENT_SHADER = (
    """
#version 430 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
uniform bool u_encoded;
// 事前乗算を外して読む（ストレートアルファの絵として CPU へ渡すとき）
// 符号化より前に外す リニアのキャンバスで符号化した後に割ると、半透明の所の色がずれる
uniform bool u_straight;
"""
    + _SRGB_FUNCTIONS
    + """
void main() {
    vec4 color = texture(u_texture, v_uv);
    if (u_straight && color.a > 0.0001) color.rgb /= color.a;
    // sRGB で混ぜたキャンバスは符号化済み ここで掛けると 2 回になって明るく飛ぶ
    vec3 rgb = u_encoded ? clamp(color.rgb, 0.0, 1.0) : srgb_encode(color.rgb);
    frag_color = vec4(rgb, color.a);
}
"""
)


#: 1 色で塗る 背景を敷くときに使う（色は事前乗算で渡す）
_FILL_FRAGMENT_SHADER = """
#version 430 core
in vec2 v_uv;
out vec4 frag_color;
uniform vec4 u_color;

void main() {
    frag_color = u_color;
}
"""


#: 絵の中身がどの列・どの行にあるかを 1 画素ずつの帯に書く（``u_axis`` 0 で列、1 で行）
#: 不透明度が 0 より大きい画素が 1 つでもあれば 1 キャンバスは半精度なので、ここで
#: 比べれば 8 ビットへ丸めると 0 になる薄い絵も拾える 帯だけを読み戻すので、
#: 画面全体を CPU へ読むより転送が 1000 分の 1 ほどで済む
_EXTENT_FRAGMENT_SHADER = """
#version 430 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
uniform int u_axis;

void main() {
    ivec2 size = textureSize(u_texture, 0);
    ivec2 here = ivec2(gl_FragCoord.xy);
    float found = 0.0;
    if (u_axis == 0) {
        for (int y = 0; y < size.y; ++y) {
            if (texelFetch(u_texture, ivec2(here.x, y), 0).a > 0.0) { found = 1.0; break; }
        }
    } else {
        for (int x = 0; x < size.x; ++x) {
            if (texelFetch(u_texture, ivec2(x, here.y), 0).a > 0.0) { found = 1.0; break; }
        }
    }
    frag_color = vec4(found);
}
"""


#: 下の絵を読んで混ぜる合成 ``glBlendFunc`` の係数では式が書けないもの
#: 描く直前に下の絵を別のバッファへ写し、シェーダの中で混ぜる
#:
#: 下の絵（キャンバス）は事前乗算アルファで溜まっている（``SRC_ALPHA`` と
#: ``ONE_MINUS_SRC_ALPHA`` で重ねてきた結果） 混ぜる式は W3C の合成の定義どおり
_BLEND_FRAGMENT_SHADER = (
    """
#version 430 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
uniform sampler2D u_backdrop;
uniform vec2 u_canvas;
uniform float u_opacity;
uniform int u_mode;
uniform bool u_premultiplied;
uniform bool u_encoded;
"""
    + _SRGB_FUNCTIONS
    + BLEND_FUNCTIONS
    + AREA_SAMPLING
    + """
// 混ぜる式は符号化した値（sRGB）で計算する AviUtl も YMM4 もそうしている
// sRGB で重ねるキャンバスでは、下の絵も上の絵ももう符号化されている
vec3 blend(vec3 below, vec3 above) {
    if (u_encoded) return blend_colors(u_mode - 100, below, above);
    vec3 mixed = blend_colors(u_mode - 100, srgb_encode(below), srgb_encode(above));
    return srgb_decode(mixed);
}

void main() {
    // 置き換えで使う、事前乗算のままの値 ストレートへ戻してからもう 1 度掛けると、
    // 戻すのを飛ばすごく薄い所（0.0001 以下）で不透明度が 2 回掛かって暗くなる
    // 縮めて置くときは事前乗算で平均する（#179）
    vec4 sampled = area_premul(u_texture, v_uv, dFdx(v_uv), dFdy(v_uv), u_premultiplied);
    vec4 source = sampled.a > 0.0001
        ? vec4(sampled.rgb / sampled.a, sampled.a)
        : vec4(0.0, 0.0, 0.0, sampled.a);
    if (u_encoded && !u_premultiplied) source.rgb = srgb_encode(source.rgb);
    float above_alpha = clamp(source.a * u_opacity, 0.0, 1.0);
    vec4 backdrop = texture(u_backdrop, gl_FragCoord.xy / u_canvas);
    float below_alpha = backdrop.a;
    vec3 below = below_alpha > 0.0001 ? backdrop.rgb / below_alpha : vec3(0.0);

    if (u_mode == 300) {
        // 黒の上に置いた絵どうしを、符号化した値のまま混ぜる（YMM4 の場面切り替えのフェード）
        // 結果は黒を含んだ色なので不透明で書く
        float amount = clamp(u_opacity, 0.0, 1.0);
        float upper_alpha = clamp(source.a, 0.0, 1.0);
        if (u_encoded) {
            frag_color = vec4(mix(below * below_alpha, source.rgb * upper_alpha, amount), 1.0);
            return;
        }
        vec3 lower = srgb_encode(below) * below_alpha;
        vec3 upper = srgb_encode(source.rgb) * upper_alpha;
        frag_color = vec4(srgb_decode(mix(lower, upper, amount)), 1.0);
        return;
    }
    if (u_mode == 301) {
        // 置き換え 下の絵もキャンバスに溜まった値（事前乗算）のまま混ぜる 混ぜる色の空間は
        // キャンバスに従う（プロジェクトの重ね合わせの設定） 事前乗算のまま混ぜないと、
        // 透明な所と混ぜたときに色だけが残って縁が明るく浮く
        float amount = clamp(u_opacity, 0.0, 1.0);
        // 事前乗算で届いた絵はキャンバスの写しで、キャンバスと同じ色の空間にある 素材（リニア）は
        // sRGB のキャンバスなら上で符号化した色を使う sampled のままだとリニアの値が混ざり暗く沈む
        vec4 upper = u_premultiplied ? sampled : vec4(source.rgb * source.a, source.a);
        frag_color = mix(backdrop, upper, amount);
        return;
    }

    vec3 mixed = blend(below, source.rgb);
    vec3 color = above_alpha * (1.0 - below_alpha) * source.rgb
               + above_alpha * below_alpha * mixed
               + (1.0 - above_alpha) * backdrop.rgb;
    frag_color = vec4(color, above_alpha + below_alpha * (1.0 - above_alpha));
}
"""
)


class BlendMode:
    """トラックを重ねるときの合成方法

    値は :class:`~sashimono.core.model.Clip` に文字列で入るので、そのまま
    プロジェクトファイルにも出る 名前を変えると古いファイルが読めなくなる
    """

    NORMAL = "normal"
    ADD = "add"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    #: AviUtl の「比較(明)」「比較(暗)」
    LIGHTEN = "lighten"
    DARKEN = "darken"
    SUBTRACT = "subtract"

    #: YMM4 の合成 式は塗りのエフェクト（:mod:`sashimono.effects.blending`）と同じ
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    LIGHTER_COLOR = "lighter_color"
    DARKER_COLOR = "darker_color"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    LINEAR_BURN = "linear_burn"
    LINEAR_LIGHT = "linear_light"
    VIVID_LIGHT = "vivid_light"
    PIN_LIGHT = "pin_light"
    HARD_MIX = "hard_mix"
    DIVISION = "division"
    HUE = "hue"
    SATURATION = "saturation"
    COLOR = "color"
    LUMINOSITY = "luminosity"
    EXTENDED = (
        SOFT_LIGHT,
        HARD_LIGHT,
        COLOR_DODGE,
        COLOR_BURN,
        LIGHTER_COLOR,
        DARKER_COLOR,
        DIFFERENCE,
        EXCLUSION,
        LINEAR_BURN,
        LINEAR_LIGHT,
        VIVID_LIGHT,
        PIN_LIGHT,
        HARD_MIX,
        DIVISION,
        HUE,
        SATURATION,
        COLOR,
        LUMINOSITY,
    )

    ALL = (NORMAL, ADD, SUBTRACT, MULTIPLY, SCREEN, OVERLAY, LIGHTEN, DARKEN, *EXTENDED)
    #: 描いた絵の不透明度で、下の絵を切り抜く（色は使わない） 選べる合成ではなく、
    #: クリップを下のクリップの形で切り抜くときにレンダラが使う
    MASK = "mask"
    #: 黒の上に置いた 2 枚の絵を sRGB の値で混ぜる 不透明度が混ぜる割合 選べる合成ではなく、
    #: 場面切り替えのフェードでレンダラが使う
    SRGB_MIX = "srgb_mix"
    #: 下の絵を、描く絵へ置き換える 不透明度が置き換える割合（0 で下の絵のまま、1 で描く絵
    #: だけ） 選べる合成ではなく、フィルタのクリップが掛けた後の絵を戻すときにレンダラが使う
    #: 通常の重ね方だと、掛けた後の絵の透明な所から掛ける前の絵が透けて残る（縮めたり
    #: 動かしたりするエフェクトで、元の絵が後ろに居残る）
    REPLACE = "replace"


def _encode(value: float) -> float:
    """リニアの 0..1 を sRGB へ シェーダの ``srgb_encode`` と同じ式"""
    value = min(max(value, 0.0), 1.0)
    if value <= 0.0031308:
        return value * 12.92
    return float(1.055 * value ** (1.0 / 2.4) - 0.055)


#: シェーダで混ぜる合成と、シェーダに渡す番号
_SHADER_BLENDS: dict[str, int] = {
    # 通常以外はシェーダの中で混ぜる 係数（glBlendFunc）ではリニアの値で混ざるが、
    # AviUtl も YMM4（Direct2D）も符号化した値（sRGB）のまま混ぜる
    # 乗算を係数で書くと、下に何も無い所（透明）で絵ごと消えるという違いもある
    **{mode: 100 + blend_index(mode) for mode in BlendMode.ALL if mode != BlendMode.NORMAL},
    BlendMode.SRGB_MIX: 300,
    BlendMode.REPLACE: 301,
}


#: 合成方法ごとの ``glBlendFuncSeparate`` の設定
#: どれもストレートアルファ（非乗算済み）前提
_BLEND_FUNCS: dict[str, tuple[int, int]] = {
    BlendMode.NORMAL: (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
}


@dataclass(frozen=True, slots=True)
class Placement:
    """描画先の矩形 ピクセル単位で、原点は左上"""

    left: float
    top: float
    width: float
    height: float

    def to_clip(self, target_width: int, target_height: int) -> tuple[float, ...]:
        """クリップ空間 (-1..1) の ``(左, 下, 右, 上)`` へ"""
        left = self.left / target_width * 2.0 - 1.0
        right = (self.left + self.width) / target_width * 2.0 - 1.0
        # GL の Y は上が正 画像座標とは向きが逆なので入れ替える
        top = 1.0 - self.top / target_height * 2.0
        bottom = 1.0 - (self.top + self.height) / target_height * 2.0
        return (left, bottom, right, top)


@dataclass(frozen=True, slots=True)
class Transform:
    """回転や拡大を伴う配置

    値の意味は AviUtl の描画パラメータに合わせてある 移植した資産が同じ数値で
    同じ見た目になるようにするためで、こちら独自の単位に直すと、スクリプトの
    値をいちいち換算することになる

    - 位置は画面中央からのずれ（ピクセル、Y は下が正）
    - ``zoom`` は 1.0 が等倍
    - ``rotation`` は度 画面上で時計回りが正
    - ``aspect`` は -1..1 正で横が縮み、負で縦が縮む
    - ``pivot`` は回転と拡大の中心を、オブジェクトの中心からずらす量
    """

    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0
    rotation: float = 0.0
    aspect: float = 0.0
    pivot_x: float = 0.0
    pivot_y: float = 0.0
    #: 軸ごとの追加の倍率 ``zoom`` とは別に掛かる
    scale_x: float = 1.0
    scale_y: float = 1.0
    #: 奥行き（画素、奥が正）と、X 軸・Y 軸の回転（度） どれかが 0 でなければ
    #: :meth:`corners` の四角形で描く 平らなままなら今までの矩形と行列で描く
    z: float = 0.0
    rotation_x: float = 0.0
    rotation_y: float = 0.0

    @property
    def is_flat(self) -> bool:
        """奥行きも傾きも無い 画面に平行な板のまま"""
        return not (self.z or self.rotation_x or self.rotation_y)

    def corners(
        self, source_width: int, source_height: int, target_width: int, target_height: int
    ) -> Corners | None:
        """四隅が画面のどこへ来るか 左上・右上・右下・左下の順（画素）

        どれかの隅がカメラを越えたら ``None``（描かない）

        中心と回転の支点を 3 次元で回してから、カメラから見た位置へ写す
        平らな板でも使えるが、そのときは :meth:`placement` と :meth:`matrix` の
        方が軽い
        """
        scale_x, scale_y = self.scale()
        half_width = source_width * scale_x / 2.0
        half_height = source_height * scale_y / 2.0
        points = []
        for sign_x, sign_y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            local = (
                sign_x * half_width - self.pivot_x,
                sign_y * half_height - self.pivot_y,
                0.0,
            )
            x, y, z = rotate(local, self.rotation_x, self.rotation_y, self.rotation)
            placed = (x + self.pivot_x + self.x, y + self.pivot_y + self.y, z + self.z)
            point = project(placed, target_width, target_height)
            if point is None:
                return None
            points.append(point)
        return (points[0], points[1], points[2], points[3])

    def scale(self) -> tuple[float, float]:
        """縦横それぞれの倍率 ``aspect`` と軸ごとの倍率を反映する"""
        wide = 1.0 - max(0.0, min(self.aspect, 1.0))
        tall = 1.0 - max(0.0, min(-self.aspect, 1.0))
        return self.zoom * wide * self.scale_x, self.zoom * tall * self.scale_y

    def placement(
        self, source_width: int, source_height: int, target_width: int, target_height: int
    ) -> Placement:
        """回転を除いた配置 中心が画面中央 + ``(x, y)`` に来る"""
        scale_x, scale_y = self.scale()
        width = max(1.0, source_width * scale_x)
        height = max(1.0, source_height * scale_y)
        return Placement(
            left=target_width / 2.0 + self.x - width / 2.0,
            top=target_height / 2.0 + self.y - height / 2.0,
            width=width,
            height=height,
        )

    def matrix(self, target_width: int, target_height: int) -> tuple[float, ...]:
        """回転をクリップ空間の 3x3 行列にする

        クリップ空間は縦横で 1 単位あたりのピクセル数が違うので、そのまま回すと
        画面の縦横比の分だけ歪む ピクセルの尺度へ直してから回し、戻す
        """
        if not self.rotation:
            return IDENTITY

        radians = math.radians(self.rotation)
        cosine, sine = math.cos(radians), math.sin(radians)
        half_width = target_width / 2.0
        half_height = target_height / 2.0

        # 回転の中心 クリップ空間で 画面の Y は下が正、クリップは上が正
        pivot_x = (self.x + self.pivot_x) / half_width
        pivot_y = -(self.y + self.pivot_y) / half_height

        # ピクセル尺度へ直して回し、戻す 時計回りを正にするため符号を入れ替える
        a = cosine
        b = sine * half_height / half_width
        c = -sine * half_width / half_height
        d = cosine
        return (
            a,
            b,
            pivot_x - a * pivot_x - b * pivot_y,
            c,
            d,
            pivot_y - c * pivot_x - d * pivot_y,
            0.0,
            0.0,
            1.0,
        )


def _rect_corners(rect: Placement) -> Corners:
    """矩形の四隅 左上・右上・右下・左下"""
    right, bottom = rect.left + rect.width, rect.top + rect.height
    return ((rect.left, rect.top), (right, rect.top), (right, bottom), (rect.left, bottom))


def fit_placement(
    source_width: int, source_height: int, target_width: int, target_height: int
) -> Placement:
    """縦横比を保ったまま画面に収まる配置を返す

    Premiere の「フレームサイズに合わせる」と同じ振る舞い 素材とプロジェクトの
    解像度が違うときに、引き伸ばして歪ませるより収める方が驚きが少ない
    """
    if source_width <= 0 or source_height <= 0:
        return Placement(0.0, 0.0, float(target_width), float(target_height))

    scale = min(target_width / source_width, target_height / source_height)
    width = source_width * scale
    height = source_height * scale
    return Placement(
        left=(target_width - width) / 2.0,
        top=(target_height - height) / 2.0,
        width=width,
        height=height,
    )


class Compositor:
    """トラックを重ねて 1 枚の絵にする

    GL コンテキストが current な状態で使うこと 生成も描画も解放もすべて同じ
    コンテキスト上で行う必要がある
    """

    def __init__(self, width: int, height: int, *, encoded: bool = False) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"解像度が不正: {width}x{height}")
        #: 真なら sRGB で符号化した値のまま重ねる（AviUtl2 と YMM4 の混ぜ方）
        #: 描いている途中で切り替えると、溜まった値の意味が食い違う 切り替えは
        #: :meth:`begin` の前に行う
        self.encoded = encoded

        self._program = Program(VERTEX_SHADER, _FRAGMENT_SHADER)
        self._blend_program = Program(VERTEX_SHADER, _BLEND_FRAGMENT_SHADER)
        # 四角形へ貼る描画 平らな矩形とは頂点の計算だけが違う
        self._mapped_program = Program(MAPPED_VERTEX_SHADER, _FRAGMENT_SHADER)
        self._mapped_blend_program = Program(MAPPED_VERTEX_SHADER, _BLEND_FRAGMENT_SHADER)
        self._resolve_program = Program(VERTEX_SHADER, _RESOLVE_FRAGMENT_SHADER)
        self._fill_program = Program(VERTEX_SHADER, _FILL_FRAGMENT_SHADER)
        self._quad = ScreenQuad()
        self._canvas = Framebuffer(width, height)
        # シェーダで混ぜる合成のとき、下の絵を写しておく先 描いている最中の
        # キャンバスを自分で読むことはできない（読みながら書くと結果が定まらない）
        self._backdrop = Framebuffer(width, height)
        # 読み出し用 リニアの合成結果を sRGB へ符号化して受け取る
        self._resolved = Framebuffer(width, height, internal_format=GL.GL_RGBA8)
        # 中身の範囲を測る帯（列と行） 場面切り替えで要るときに初めて作る
        # 合成先はレイヤーごとに何枚も作るので、使わないものにまで持たせない
        self._extent_program: Program | None = None
        self._columns: Framebuffer | None = None
        self._rows: Framebuffer | None = None

    @property
    def width(self) -> int:
        return self._canvas.width

    @property
    def height(self) -> int:
        return self._canvas.height

    @property
    def quad(self) -> ScreenQuad:
        """全画面四角形 エフェクト処理と共有する"""
        return self._quad

    def resize(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"解像度が不正: {width}x{height}")
        self._canvas.resize(width, height)
        self._backdrop.resize(width, height)
        self._resolved.resize(width, height)
        if self._columns is not None:
            self._columns.resize(width, 1)
        if self._rows is not None:
            self._rows.resize(1, height)

    @property
    def canvas(self) -> Framebuffer:
        """合成途中の絵（事前乗算アルファ :attr:`encoded` なら sRGB、でなければリニア）

        YMM4 の ``FrameBufferItem`` のように、それまでに重ねた絵を素材として
        使うときに読む 読んだものを同じキャンバスへ描くときは、先に別の
        バッファへ写すこと エフェクトへ渡すときは、値の種類を
        :attr:`EffectProcessor.canvas_encoded` で伝える
        """
        return self._canvas

    def underlay(
        self,
        color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        target: Framebuffer | None = None,
    ) -> None:
        """重ね終わった絵の下へ色を敷く 透明な所だけがその色になる（リニア値）

        合成は透明な下地の上で行い、最後に背景を敷く 不透明な黒の上で合成すると、
        乗算などが下に何も無い所でも黒と混ざり、YMM4 と見え方が変わる
        ``target`` を渡すと、キャンバスの代わりにそのバッファ（事前乗算アルファ）へ敷く
        """
        (target or self._canvas).bind()
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(
            GL.GL_ONE_MINUS_DST_ALPHA, GL.GL_ONE, GL.GL_ONE_MINUS_DST_ALPHA, GL.GL_ONE
        )
        self._fill_program.use()
        self._fill_program.set_vec4("u_rect", FULL_RECT)
        self._fill_program.set_bool("u_flip", False)
        red, green, blue, alpha = self._canvas_color(color)
        self._fill_program.set_vec4("u_color", (red * alpha, green * alpha, blue * alpha, alpha))
        self._quad.draw()
        self._set_blend(BlendMode.NORMAL)

    def begin(self, background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """合成を始める 背景色はリニア値で指定する"""
        self._canvas.bind(clear=self._canvas_color(background))
        GL.glEnable(GL.GL_BLEND)
        self._set_blend(BlendMode.NORMAL)

    def draw(
        self,
        texture: Texture,
        *,
        placement: Placement | None = None,
        opacity: float = 1.0,
        flip: bool = True,
        blend: str = BlendMode.NORMAL,
        matrix: tuple[float, ...] | None = None,
    ) -> None:
        """1 枚を重ねる

        ``flip`` の既定が真なのは、デコードした画像が左上原点で、GL のテクスチャ座標が
        左下原点だから 素材由来の画像はほぼ常に反転が要る エフェクトを通した後の
        テクスチャはすでに GL の向きなので、そちらは偽で渡す
        """
        if placement is None:
            placement = fit_placement(texture.width, texture.height, self.width, self.height)
        self.draw_handle(
            texture.handle, placement, opacity=opacity, flip=flip, blend=blend, matrix=matrix
        )

    def draw_handle(
        self,
        handle: int,
        placement: Placement,
        *,
        opacity: float = 1.0,
        flip: bool = True,
        blend: str = BlendMode.NORMAL,
        matrix: tuple[float, ...] | None = None,
        premultiplied: bool = False,
    ) -> None:
        """GL のテクスチャ番号を直接指定して重ねる

        エフェクトを通した結果はフレームバッファの中にあり、:class:`Texture` では
        包まれていない それを合成するための入口

        ``matrix`` を渡すと、矩形に加えてその変換が掛かる（回転など）
        ``premultiplied`` は、渡す絵が事前乗算アルファで溜まっているとき（入れ子の
        シーンのキャンバス） そのまま重ねると、半透明の縁が 2 回薄まって黒ずむ
        """
        program = self._begin_draw(blend, self._program, self._blend_program)
        program.set_bool("u_premultiplied", premultiplied)
        program.set_vec4("u_rect", placement.to_clip(self.width, self.height))
        program.set_bool("u_flip", flip)
        program.set_float("u_opacity", float(np.clip(opacity, 0.0, 1.0)))
        program.set_mat3("u_transform", matrix if matrix is not None else IDENTITY)
        program.bind_texture("u_texture", handle)
        self._quad.draw()
        if matrix is not None:
            # 次の描画へ持ち越さない 持ち越すと、回転を掛けた次のクリップまで
            # 一緒に回る
            program.set_mat3("u_transform", IDENTITY)

    def draw_mapped(
        self,
        handle: int,
        *,
        source: Placement,
        anchor: Placement,
        corners: Corners,
        opacity: float = 1.0,
        flip: bool = True,
        blend: str = BlendMode.NORMAL,
        uv: Corners | None = None,
    ) -> bool:
        """絵を任意の四角形へ貼る 描けなければ偽

        ``source`` はテクスチャが占める矩形、``anchor`` はそのうち ``corners`` へ
        写す矩形（どちらも同じ画素の座標） 素材をそのまま貼るなら 2 つは同じで、
        エフェクトを通した画面いっぱいの結果を貼るなら ``source`` は画面全体、
        ``anchor`` はその中で絵が置かれた矩形になる 射影変換は矩形の外まで
        同じ式で伸びるので、画面全体を送っても絵の部分が正しく四隅に来る

        ``uv`` は絵のどこを貼るか（左上・右上・右下・左下、0..1、画像の上が 0）
        """
        rows = homography(_rect_corners(anchor), corners)
        if rows is None:
            return False
        matrix = to_clip(self.width, self.height) @ rows
        # 四角形の四隅がカメラの後ろへ回ると、w の符号が混ざって面が裏返しに
        # 広がる 板が真横を向いた瞬間の前後にあたり、描かないのが正しい
        for x, y in _rect_corners(source):
            if float(matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]) <= 0.0:
                return False

        program = self._begin_draw(blend, self._mapped_program, self._mapped_blend_program)
        program.set_bool("u_premultiplied", False)
        program.set_vec4(
            "u_rect",
            (source.left, source.top + source.height, source.left + source.width, source.top),
        )
        program.set_mat3("u_homography", tuple(float(v) for v in matrix.reshape(-1)))
        program.set_bool("u_flip", flip)
        program.set_bool("u_use_uv", uv is not None)
        if uv is not None:
            program.set_vec2_array("u_uv", uv)
        program.set_float("u_opacity", float(np.clip(opacity, 0.0, 1.0)))
        program.bind_texture("u_texture", handle)
        self._quad.draw()
        return True

    def _begin_draw(self, blend: str, plain: Program, shaded: Program) -> Program:
        """描く準備をして、使うシェーダを返す

        係数で書ける合成は GL の合成に任せる シェーダで混ぜる合成は、下の絵を
        写してから GL の合成を切って描く（混ぜた結果をそのまま書き込むため）
        """
        mode = _SHADER_BLENDS.get(blend)
        if mode is None:
            self._canvas.bind()
            GL.glEnable(GL.GL_BLEND)
            self._set_blend(blend)
            plain.use()
            plain.set_bool("u_encoded", self.encoded)
            return plain

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._canvas.handle)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._backdrop.handle)
        GL.glBlitFramebuffer(
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
            GL.GL_COLOR_BUFFER_BIT,
            GL.GL_NEAREST,
        )
        self._canvas.bind()
        GL.glDisable(GL.GL_BLEND)
        shaded.use()
        shaded.set_bool("u_encoded", self.encoded)
        shaded.set_int("u_mode", mode)
        shaded.set_vec2("u_canvas", (float(self.width), float(self.height)))
        shaded.bind_texture("u_backdrop", self._backdrop.color, unit=1)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        return shaded

    def read(self, *, straight: bool = False) -> np.ndarray:
        """合成結果を sRGB 符号化した ``(高さ, 幅, 4)`` の uint8 配列で返す

        色は事前乗算のまま返す（黒を敷いた後の画面なら不透明なので同じこと）
        ``straight`` を立てるとストレートアルファで返す 半透明の絵を素材として
        スクリプトへ渡すときに使う 事前乗算のまま渡すと、素材として重ねるときに
        不透明度がもう 1 度掛かって半透明の所が暗くなる
        """
        # 符号化した先のフレームバッファが結ばれたまま戻るので、そのまま読む
        self.resolved_texture(straight=straight)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        # 受け皿を先に作って、そこへ直接書かせる 受け取り先を省くと PyOpenGL が
        # bytes を作り、そこから numpy へもう 1 回写す 4K（1 枚 33MB）では
        # この 2 回の写しと並べ替えで 28.5ms かかっていたものが 10.1ms になる
        image = np.empty((self.height, self.width, 4), dtype=np.uint8)
        GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, image)
        return image

    def resolved_texture(self, *, straight: bool = False) -> int:
        """合成結果を sRGB 符号化した RGBA8 にして、そのテクスチャ番号を返す

        :meth:`read` が CPU へ読み戻すのと同じ値（行 0 が絵の上の端） GPU の上で続けて
        使うとき（書き出しの色変換 #372）にここから読む 別に符号化し直すと、
        CPU の道と 8bit への丸め方が食い違い、同じ絵でも書き出しの色が変わる
        次に合成するまでの間だけ使える（同じテクスチャへ上書きする）
        """
        # リニアの結果をもう 1 パス通して sRGB へ符号化する glReadPixels に
        # RGBA16F から直接 uint8 で読ませると、変換式が実装依存になる
        # 符号化のついでに上下も返しておく GL は左下から行を返すので、返して
        # おけば CPU で並べ替えずに済む（4K で 33MB の写しが 1 回消える）
        self._resolve(
            self._resolved.handle, (0, 0, self.width, self.height), flip=True, straight=straight
        )
        return self._resolved.color

    def content_box(self) -> tuple[int, int, int, int] | None:
        """合成途中の絵で、不透明度が 0 でない範囲（画素、左・上・右・下 左上が原点）

        何も無ければ ``None`` 範囲は GPU の上で列と行の帯にまとめてから読む
        画面全体を毎フレーム読み戻すと、1080p で 8MB の転送と GPU の待ちが入り、
        場面切り替えの間だけ再生が重くなる
        """
        if self._extent_program is None:
            self._extent_program = Program(VERTEX_SHADER, _EXTENT_FRAGMENT_SHADER)
        if self._columns is None:
            self._columns = Framebuffer(self.width, 1, internal_format=GL.GL_RGBA8)
        if self._rows is None:
            self._rows = Framebuffer(1, self.height, internal_format=GL.GL_RGBA8)
        columns = np.flatnonzero(self._extent(self._columns, 0))
        # GL の行は下から数える 画像の向き（上から）へ直す
        rows = np.flatnonzero(self._extent(self._rows, 1)[::-1])
        if columns.size == 0 or rows.size == 0:
            return None
        return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1

    def _extent(self, band: Framebuffer, axis: int) -> np.ndarray:
        """帯へ中身の有無を書き、真偽の 1 次元配列で読む"""
        assert self._extent_program is not None
        band.bind(clear=(0.0, 0.0, 0.0, 0.0))
        GL.glDisable(GL.GL_BLEND)
        program = self._extent_program
        program.use()
        program.set_vec4("u_rect", FULL_RECT)
        program.set_bool("u_flip", False)
        program.set_mat3("u_transform", IDENTITY)
        program.set_int("u_axis", axis)
        program.bind_texture("u_texture", self._canvas.color)
        self._quad.draw()
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        raw = GL.glReadPixels(0, 0, band.width, band.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        values = np.frombuffer(raw, dtype=np.uint8).reshape(band.height * band.width, 4)
        return np.asarray(values[:, 0] > 127)

    def present(
        self,
        framebuffer: int,
        viewport: tuple[int, int, int, int],
        *,
        letterbox: bool = True,
    ) -> None:
        """合成結果を画面（や任意のフレームバッファ）へ直接出す

        プレビュー表示用 :meth:`read` と違って CPU へ戻さないので、GPU から
        CPU へ、また GPU へ、という往復が要らない 1080p なら 1 フレームあたり
        8MB の転送が消える

        ``letterbox`` が真なら、``viewport`` の中で縦横比を保って収める
        プレビュー枠の形が映像と違っても歪まない
        """
        target = self._fit(viewport, letterbox)
        if target is None:
            return
        self._clear_for(framebuffer, viewport)
        self._resolve(framebuffer, target)

    def show(
        self,
        texture: int,
        framebuffer: int,
        viewport: tuple[int, int, int, int],
        *,
        letterbox: bool = True,
    ) -> None:
        """**すでに sRGB へ符号化された絵**を、そのまま出す

        先読みして取っておいた絵を画面へ出すための入口 :meth:`present` は
        リニアの合成結果を符号化しながら出すので、符号化済みの絵に使うと
        2 回掛かって白っぽくなる

        置き方（余白の付け方）は :meth:`present` と同じにする 違うと、
        先読みが当たったコマだけ絵の位置がずれる
        """
        target = self._fit(viewport, letterbox)
        if target is None:
            return
        self._clear_for(framebuffer, viewport)
        GL.glViewport(*target)
        self._program.use()
        self._program.set_bool("u_premultiplied", False)
        # 符号化済みの絵なので、sRGB で重ねるキャンバスのときも符号化しない
        self._program.set_bool("u_encoded", False)
        self._program.set_vec4("u_rect", FULL_RECT)
        self._program.set_bool("u_flip", False)
        self._program.set_float("u_opacity", 1.0)
        self._program.set_mat3("u_transform", IDENTITY)
        self._program.bind_texture("u_texture", texture)
        self._quad.draw()

    def _fit(
        self, viewport: tuple[int, int, int, int], letterbox: bool
    ) -> tuple[int, int, int, int] | None:
        """出す先の矩形 幅か高さが無ければ ``None``（描く場所が無い）"""
        x, y, width, height = viewport
        if width <= 0 or height <= 0:
            return None
        if not letterbox:
            return viewport
        placed = fit_placement(self.width, self.height, width, height)
        return (
            x + int(placed.left),
            y + int(placed.top),
            max(1, int(placed.width)),
            max(1, int(placed.height)),
        )

    def _clear_for(self, framebuffer: int, viewport: tuple[int, int, int, int]) -> None:
        """出す先を黒で塗る 余白に前のコマが残らないようにする"""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        GL.glViewport(*viewport)
        GL.glDisable(GL.GL_BLEND)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

    def release(self) -> None:
        self._canvas.release()
        self._backdrop.release()
        self._resolved.release()
        for band in (self._columns, self._rows):
            if band is not None:
                band.release()
        if self._extent_program is not None:
            self._extent_program.release()
        self._quad.release()
        for program in (
            self._program,
            self._blend_program,
            self._mapped_program,
            self._mapped_blend_program,
            self._resolve_program,
            self._fill_program,
        ):
            program.release()

    def _set_blend(self, mode: str) -> None:
        if mode == BlendMode.MASK:
            # 色も不透明度も、描いた絵の不透明度を掛けるだけ
            GL.glBlendFuncSeparate(GL.GL_ZERO, GL.GL_SRC_ALPHA, GL.GL_ZERO, GL.GL_SRC_ALPHA)
            return
        source, destination = _BLEND_FUNCS.get(mode, _BLEND_FUNCS[BlendMode.NORMAL])
        GL.glBlendFuncSeparate(source, destination, GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)

    def _canvas_color(
        self, color: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """リニアで指定された色を、キャンバスに溜める値へ直す"""
        if not self.encoded:
            return color
        red, green, blue, alpha = color
        return (_encode(red), _encode(green), _encode(blue), alpha)

    def _resolve(
        self,
        framebuffer: int,
        viewport: tuple[int, int, int, int],
        *,
        flip: bool = False,
        straight: bool = False,
    ) -> None:
        """合成結果を sRGB で符号化した値にして ``framebuffer`` へ描く

        ``flip`` を立てると上下を返して描く 読み戻す側のためのもので、GL は
        左下から行を返すので、先に返しておけば CPU で並べ替えずに済む
        ``straight`` を立てると事前乗算を外してから符号化する（:meth:`read`）
        """
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        GL.glViewport(*viewport)
        GL.glDisable(GL.GL_BLEND)
        self._resolve_program.use()
        self._resolve_program.set_bool("u_encoded", self.encoded)
        self._resolve_program.set_bool("u_straight", straight)
        self._resolve_program.set_vec4("u_rect", FULL_RECT)
        self._resolve_program.set_bool("u_flip", flip)
        self._resolve_program.bind_texture("u_texture", self._canvas.color)
        self._quad.draw()
