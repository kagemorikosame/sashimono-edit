"""絵の色を模様で塗るエフェクト 単色・線形と円形のグラデーション・ストライプ・水玉・格子

YMM4 の「前景を塗りつぶし」「グラデーション」エフェクトと、図形のブラシを 1 つで受ける
値の意味は YMM4 本体に描かせた試験テンプレート（``tools/ymm4_probes.py``）の絵から
読み取った

- 座標は絵の中心が原点、**Y は下が正**（YMM4 と同じ） 角度は画面上で時計回り
- 線形は角度 0 で左から右 ``size`` がグラデーションの長さ（画素）で、中心で 0.5
- 模様の色の補間と合成モードの計算は sRGB のまま行う YMM4（Direct2D）がそうしている
  リニアで混ぜると、乗算や中間色が暗く沈んで配布物と違う色になる
"""

from __future__ import annotations

from collections.abc import Mapping

from sashimono.core.model import AnimatedValue, ParamValue
from sashimono.effects.blending import BLEND_FUNCTIONS, BLEND_MODES
from sashimono.effects.builtin import PRELUDE
from sashimono.effects.definition import EffectDefinition, registry
from sashimono.effects.spec import (
    MAX_GRADIENT_STOPS,
    CheckSpec,
    ColorSpec,
    GradientSpec,
    SelectSpec,
    TrackSpec,
    ValueSpec,
)

__all__ = ["BLEND_MODES", "MAX_STOPS", "PATTERNS", "register_paint_effects"]

#: グラデーションの色の数の上限 配布物の多くは 2〜5 色 金属風の文字で 10 を超えるものが
#: あるが、上限を超えた分は間引いて写す（:mod:`sashimono.compat.ymm4.brushes`）
#: テキストの塗りのグラデーション（#312）と同じ数にする 違うと、エフェクトから塗りへ
#: 写した色の並びが片方の上限で切れる
MAX_STOPS = MAX_GRADIENT_STOPS

#: 模様の種類 シェーダの番号と同じ並び
PATTERNS = (
    ("solid", "単色"),
    ("linear", "線形グラデーション"),
    ("radial", "円形グラデーション"),
    ("convex", "山形グラデーション"),
    ("stripe", "ストライプ"),
    ("dot", "水玉"),
    ("grid", "格子"),
    ("noise", "ノイズ"),
)

#: ノイズの種類 シェーダの番号と同じ並び
NOISE_KINDS = (
    ("perlin", "なめらか"),
    ("random", "砂嵐"),
    ("fractal", "粗いなめらか"),
    ("curl", "うねり"),
    ("voronoi", "ボロノイ（面ごとの色）"),
    ("cellular", "セル（点からの距離）"),
    ("block", "ブロック"),
)


#: ノイズで薄める先 塗らずに不透明度か色を 1 - 値 倍にする（YMM4 の NoiseEffect の IsAlpha）
_NOISE_MASKS = (("off", "使わない"), ("alpha", "不透明度"), ("color", "色"))

_EXTEND = (("clamp", "端の色"), ("wrap", "繰り返し"), ("mirror", "折り返し"))

_STOP_UNIFORMS = "\n".join(
    f"uniform vec4 color{index};\nuniform float offset{index};" for index in range(MAX_STOPS)
)

#: 番号から色と位置を引く GLSL の uniform は配列にせず名前で持つ（パラメータの名前と揃える）
_STOP_LOOKUP = (
    "vec4 stop_color(int i) {\n"
    + "".join(f"    if (i == {i}) return color{i};\n" for i in range(MAX_STOPS - 1))
    + f"    return color{MAX_STOPS - 1};\n}}\n"
    + "float stop_offset(int i) {\n"
    + "".join(f"    if (i == {i}) return offset{i};\n" for i in range(MAX_STOPS - 1))
    + f"    return offset{MAX_STOPS - 1};\n}}\n"
)

_PAINT = (
    PRELUDE
    + _STOP_UNIFORMS
    + """
uniform int pattern;
uniform int extend;
uniform int stops;
"""
    + f"const int MAX_STOPS = {MAX_STOPS};\n"
    + """
uniform int blend;
uniform float opacity;
uniform float center_x;
uniform float center_y;
uniform float angle;
uniform float size;
uniform float offset;
uniform float radius_x;
uniform float radius_y;
uniform float width_a;
uniform float width_b;
uniform float dot_radius;
uniform float span;
uniform float aspect;
uniform float thickness;
uniform float cell_width;
uniform float cell_height;
uniform float zoom;
uniform bool inverted;
uniform bool pattern_only;
uniform bool key_only;
uniform vec4 key_color;
uniform vec4 keep_color;
uniform int noise_mask;
uniform bool relative;
uniform int noise_kind;
uniform float noise_strength;
uniform float noise_threshold;
uniform float noise_levels;
uniform int noise_octaves;
uniform bool turbulence;
uniform bool colorful;
uniform float noise_scale_x;
uniform float noise_scale_y;
uniform float noise_x;
uniform float noise_y;
uniform float noise_z;
uniform float speed_x;
uniform float speed_y;
uniform float speed_z;

vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

"""
    + _STOP_LOOKUP
    + """
// 色のバーで並べた止まり（#364） 数が 2 より少なければ前からの「色 1」「位置 1」… で塗る
// 色は sRGB のまま届く（混ぜるのも sRGB なので、リニアへ往復させない）
uniform int gradient_count;
uniform vec4 gradient_colors[MAX_STOPS];
uniform float gradient_offsets[MAX_STOPS];

// 単色の模様はバーを使わず「色 1」で塗る（設定パネルも単色ではバーを出さず色の欄を出す）
bool from_bar() { return gradient_count >= 2 && pattern != 0; }
int stop_count() {
    return from_bar() ? min(gradient_count, MAX_STOPS) : clamp(stops, 1, MAX_STOPS);
}
float stop_at(int i) { return from_bar() ? gradient_offsets[i] : stop_offset(i); }
vec4 stop_srgb(int i) {
    if (from_bar()) return gradient_colors[i];
    vec4 c = stop_color(i);
    return vec4(to_srgb(c.rgb), c.a);
}

// 色は sRGB で補間する（YMM4 と同じ） 前からの項目の色はリニアで届くので先に戻す
vec4 ramp(float t) {
    int count = stop_count();
    if (count == 1 || t <= stop_at(0)) return stop_srgb(0);
    for (int i = 1; i < count; ++i) {
        float right = stop_at(i);
        if (t <= right) {
            float left = stop_at(i - 1);
            float k = right > left ? (t - left) / (right - left) : 1.0;
            return mix(stop_srgb(i - 1), stop_srgb(i), clamp(k, 0.0, 1.0));
        }
    }
    return stop_srgb(count - 1);
}

float extended(float t) {
    if (extend == 1) return fract(t);
    if (extend == 2) return 1.0 - abs(fract(t * 0.5) * 2.0 - 1.0);
    return clamp(t, 0.0, 1.0);
}

vec2 rotated(vec2 p, float degrees) {
    float r = radians(degrees);
    return mat2(cos(r), sin(r), -sin(r), cos(r)) * p;
}


// sin を使う乱数は大きな座標で桁が落ち、格子の値が横に揃って筋が出る 小数部だけで混ぜる
float lattice_value(vec2 cell, float slice) {
    vec3 p = fract(vec3(cell.xyx + vec3(slice * 17.13, slice * 5.71, slice * 11.3)) * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float smooth_value(vec2 q, float slice) {
    vec2 cell = floor(q);
    vec2 f = fract(q);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    if (noise_kind == 2) {
        // 粗いなめらか（Fractal）は格子の値をつなぐ 四角い粒が残る YMM4 の絵もそうだった
        float a = lattice_value(cell, slice);
        float b = lattice_value(cell + vec2(1.0, 0.0), slice);
        float c = lattice_value(cell + vec2(0.0, 1.0), slice);
        float d = lattice_value(cell + vec2(1.0, 1.0), slice);
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }
    // なめらかな種類は勾配ノイズ 格子の値をつなぐと四角い粒が見える
    float corners[4];
    for (int k = 0; k < 4; ++k) {
        vec2 offset = vec2(float(k % 2), float(k / 2));
        float a = lattice_value(cell + offset, slice) * 6.2831853;
        corners[k] = dot(vec2(cos(a), sin(a)), f - offset);
    }
    float n = mix(mix(corners[0], corners[1], u.x), mix(corners[2], corners[3], u.x), u.y);
    return clamp(0.5 + n * 0.5, 0.0, 1.0);
}

// 奥行き（z）は整数の層の間を補間して、時間で滑らかに変わるようにする
float layered(vec2 q, float z, float salt) {
    float lower = floor(z) + salt * 31.0;
    return mix(smooth_value(q, lower), smooth_value(q, lower + 1.0), fract(z));
}

float fractal_sum(vec2 q, float z, float salt) {
    float total = 0.0;
    float weight = 0.5;
    float norm = 0.0;
    for (int i = 0; i < clamp(noise_octaves, 1, 8); ++i) {
        float n = layered(q, z, salt + float(i));
        // 乱流は 0.5 からの隔たりを裏返して足す 谷が白い筋になる（YMM4 の絵は明るい地に筋）
        total += weight * (turbulence ? 1.0 - abs(n * 2.0 - 1.0) : n);
        norm += weight;
        weight *= 0.5;
        q = q * 2.0 + 13.7;
    }
    return total / max(norm, 1e-4);
}

float cell_noise(vec2 q, float z, float salt, bool distance_only) {
    vec2 cell = floor(q);
    float nearest = 10.0;
    float value = 0.0;
    float slice = floor(z) + salt * 31.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 c = cell + vec2(i, j);
            vec2 point = c + vec2(lattice_value(c, slice + 0.3), lattice_value(c, slice + 0.7));
            float d = length(q - point);
            if (d < nearest) { nearest = d; value = lattice_value(c, slice + 0.9); }
        }
    }
    return distance_only ? clamp(nearest, 0.0, 1.0) : value;
}

// YMM4 のノイズのブラシ 大きさ 100% の目の粗さは、YMM4 に描かせた絵の粒の数から決めた
float shaped_noise(float v, float strength);

float noise_raw(vec2 p, float salt) {
    // 粒の大きさ（下の 2〜60）は画面の画素 画質を落とした下書きでも粒が絵に対して同じ大きさに
    // 見えるよう、画面 1 画素あたりの画素数を掛ける 掛けないと下書きだけ粒が 2〜4 倍に粗くなる
    vec2 scale = max(vec2(noise_scale_x, noise_scale_y) * 0.01, vec2(0.0001)) * u_pixel_scale;
    vec2 moved = rotated(p, -angle) + vec2(noise_x, noise_y) + vec2(speed_x, speed_y) * u_time;
    float z = noise_z + speed_z * u_time;
    float v;
    if (noise_kind == 1) {
        // 砂嵐は升ごとにばらばらの値 勾配ノイズを升の角（整数の点）で引くと、どこでも 0.5 の
        // 一様な灰になっていた（#177 大きさ 1000% の不透明度が YMM4 は 0.03〜0.97 に散り、
        // こちらは 0.5 の一枚だった） 升の大きさは大きさ 100% で 1 画素 YMM4 の絵は
        // 大きさ 1000% で隣の画素との相関が 0.98、8 画素離れて 0.22 だった
        vec2 cell = floor(moved / scale);
        float depth = z * 7.0;
        float lower = floor(depth) + salt * 31.0;
        v = mix(lattice_value(cell, lower), lattice_value(cell, lower + 1.0), fract(depth));
    } else if (noise_kind == 4 || noise_kind == 5) {
        float grain = noise_kind == 5 ? 60.0 : 25.0;
        v = cell_noise(moved / (grain * scale), z, salt, noise_kind == 5);
    } else if (noise_kind == 6) {
        v = lattice_value(floor(moved / (30.0 * scale)), floor(z) + salt * 31.0);
    } else if (noise_kind == 2) {
        v = fractal_sum(moved / (35.0 * scale), z, salt);
    } else {
        // なめらかな種類の値は 0.35〜0.65 ほどに収まる（YMM4 の絵と同じ散らばり）
        v = fractal_sum(moved / (48.0 * scale), z, salt);
    }
    return v;
}

float noise_value(vec2 p, float salt) {
    return shaped_noise(noise_raw(p, salt), max(noise_strength, 0.0) * 0.01);
}

// 強さは値に掛ける しきい値を上げると明るい側へ寄る 段階は値を量子化する
float shaped_noise(float v, float strength) {
    v *= strength;
    v /= max(1.0 - clamp(noise_threshold * 0.01, 0.0, 0.99), 0.01);
    float levels = max(noise_levels, 2.0);
    if (levels < 255.5) v = floor(clamp(v, 0.0, 0.9999) * levels) / (levels - 1.0);
    return clamp(v, 0.0, 1.0);
}

// 模様の色（sRGB、ストレートアルファ） p は絵の中心からの画素で Y は下が正
vec4 pattern_color(vec2 p) {
    float z = max(zoom, 0.0001) * 0.01;
    vec2 q = p - vec2(center_x, center_y);
    if (pattern == 0) return ramp(0.0);
    if (pattern == 7) {
        if (!colorful) return ramp(noise_value(q, 0.0));
        // 色つきは色の通り道ごとに別のノイズを引く
        vec3 channels = vec3(noise_value(q, 0.0), noise_value(q, 1.0), noise_value(q, 2.0));
        vec4 low = ramp(0.0);
        vec4 high = ramp(1.0);
        return vec4(mix(low.rgb, high.rgb, channels), mix(low.a, high.a, channels.g));
    }
    if (pattern == 1 || pattern == 3) {
        vec2 direction = vec2(cos(radians(angle)), sin(radians(angle)));
        // 割合で指定したときは絵の幅が 100（YMM4 の CoordinateMode が Relative）
        // 長さとずらしは画素の値として画質の倍率を掛けて届くが、割合のときは倍率が要らない
        // 絵の幅がもう下書きの画素で測られているので、ここで倍率を戻さないと二重に縮む
        float unit = relative ? object_size().x * 0.01 / max(u_pixel_scale, 0.0001) : 1.0;
        float along = dot(q, direction) - offset * unit;
        float length_ = max(size * unit, 1.0);
        // 山形は中心で終わりの色、長さの半分で始まりの色に戻る
        if (pattern == 3) return ramp(extended(1.0 - abs(along) / (length_ * 0.5)));
        return ramp(extended(along / length_ + 0.5));
    }
    if (pattern == 2) {
        vec2 r = rotated(q, -angle);
        vec2 radius = max(vec2(radius_x, radius_y) * z, vec2(1.0));
        radius.x *= 1.0 - max(aspect, 0.0) * 0.01;
        radius.y *= 1.0 + min(aspect, 0.0) * 0.01;
        return ramp(extended(length(r / radius)));
    }
    if (pattern == 4) {
        vec2 direction = vec2(cos(radians(angle)), sin(radians(angle)));
        float a = max(width_a * z, 0.0);
        float b = max(width_b * z, 0.0);
        float period = max(a + b, 1.0);
        // 1 本目の帯は絵の中心にまたがる（YMM4 の絵で中心の画素が帯の真ん中だった）
        // 中心のずらしはほかの模様と同じく q（中心を引いたあと）で測る
        float m = mod(dot(q, direction) - offset * z + a * 0.5, period);
        return m < a ? ramp(0.0) : ramp(1.0);
    }
    vec2 r = rotated(q, -angle);
    if (pattern == 5) {
        // 縦横比は模様ごと縮める 正なら横、負なら縦（間隔も水玉の形も同じ割合）
        vec2 squash = vec2(1.0 - max(aspect, 0.0) * 0.01, 1.0 + min(aspect, 0.0) * 0.01);
        squash = max(squash, vec2(0.01));
        vec2 cell = vec2(max(span * z, 1.0)) * squash;
        // 水玉は中心から半マスずれた所に並ぶ 反転の設定は YMM4 の絵でも変わらなかった
        vec2 local = mod(r, cell) - cell * 0.5;
        bool inside = length(local / squash) <= dot_radius * z;
        return inside ? ramp(0.0) : ramp(1.0);
    }
    vec2 cell = max(vec2(cell_width, cell_height) * z, vec2(1.0));
    // 格子の線は中心を通り、線の太さの真ん中が中心に来る
    vec2 local = mod(r + thickness * z * 0.5, cell);
    bool line = min(local.x, local.y) < thickness * z;
    if (inverted) line = !line;
    return line ? ramp(0.0) : ramp(1.0);
}

// 縁のある模様（ストライプ・水玉・格子）は、画素の中を 4x4 に分けて平均する
// 画素ごとに「乗る・乗らない」で描くと、太さ 1.3 の斜めの線が場所によって 1 画素にも
// 2 画素にもなり、菱形の大きな濃淡（モアレ）が浮く（SFっぽい吹き出し(右) の格子 #179）
// YMM4 は線の縁をなめらかに描く 事前乗算で平均するので、透明な色の RGB は混ざらない
// グラデーションとノイズはもともとなめらかで、ノイズは重いので 1 点のまま
vec4 pattern_smooth(vec2 p) {
    if (pattern < 4 || pattern > 6) return pattern_color(p);
    vec4 sum = vec4(0.0);
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            vec4 c = pattern_color(p + (vec2(float(i), float(j)) + 0.5) * 0.25 - 0.5);
            sum += vec4(c.rgb * c.a, c.a);
        }
    }
    sum *= 1.0 / 16.0;
    return sum.a > 0.0 ? vec4(sum.rgb / sum.a, sum.a) : vec4(0.0);
}

"""
    + BLEND_FUNCTIONS
    + """
vec3 blended(vec3 b, vec3 s) { return blend_colors(blend, b, s); }

void main() {
    vec4 base = texture(u_texture, v_uv);
    vec2 pixel = v_uv * u_size - object_center();
    pixel.y = -pixel.y;
    if (noise_mask != 0) {
        // YMM4 のノイズ（NoiseEffect） 模様の値だけ薄める 1 で消え 0 で元のまま
        // 不透明度なら不透明度に、色なら符号化した色に 1 - 値 を掛ける（#177 の探りで
        // 強さ 100 の乱数が不透明度の平均 0.50、強さ 50 が 0.75、灰 128 の色が 64 になった）
        // 強さ s が 1 まではノイズ n で 1 - s n、1 を越えると (2 - s)(1 - n) 強さ 120・150・200 で
        // 不透明度の平均が 0.41・0.25・0（散らばりは平均に比例して縮む）だった 値を s 倍して
        // 切り詰めると 0.42・0.33・0.25 になり、強い所が消えきらない
        float n = shaped_noise(noise_raw(pixel - vec2(center_x, center_y), 0.0), 1.0);
        float s = max(noise_strength, 0.0) * 0.01;
        float keep = s <= 1.0 ? 1.0 - s * n : max(2.0 - s, 0.0) * (1.0 - n);
        if (noise_mask == 1) { frag_color = vec4(base.rgb, base.a * keep); return; }
        frag_color = vec4(to_linear(to_srgb(base.rgb) * keep), base.a);
        return;
    }
    vec4 paint = pattern_smooth(pixel);
    float amount = clamp(opacity * 0.01, 0.0, 1.0) * paint.a;
    if (pattern_only) {
        // 模様だけで塗る 元の絵の色は使わず、形（不透明度）だけを借りる
        // 透明な所は色も 0 にする 後ろの変形は隣の画素と RGB のまま混ぜるので、
        // 透明な白（YMM4 の格子の背景に多い）を残すと、縮めた格子が白っぽい板になる
        float alpha = base.a * amount;
        frag_color = alpha > 0.0 ? vec4(to_linear(paint.rgb), alpha) : vec4(0.0);
        return;
    }
    vec3 under = to_srgb(base.rgb);
    if (key_only) {
        // 目印の色で塗った所だけを模様に替える share は画素のうち塗りが占める割合
        vec3 key = to_srgb(key_color.rgb);
        if (keep_color.a <= 0.0) {
            // 線の色を持たない前の版の保存（目印はマゼンタ）は、前の版と同じ絵にする
            // 目印からの近さで混ぜ、α は元の絵のまま 下の透かし方を当てると、透明な模様の
            // 所で開き直した作品の塗りが消える
            float near = clamp(1.0 - length(under - key) * 3.0, 0.0, 1.0);
            frag_color = vec4(to_linear(mix(under, paint.rgb, amount * near)), base.a);
            return;
        }
        // 絵は線の色と目印の色の 2 色だけ（縁はその間の色）なので、画素の色が
        // 2 色を結ぶ線のどこにあるかで割合が決まる 目印からの近さで決めると、
        // 境の中間色で割合がずれて目印の色が残り、目印に近い線の色は模様に食われる
        vec3 kept = to_srgb(keep_color.rgb);
        vec3 axis = key - kept;
        float share = clamp(dot(under - kept, axis) / max(dot(axis, axis), 1e-6), 0.0, 1.0);
        // 塗りの分は模様の不透明度で透かす 目印と混ぜると、透明な模様から目印が透ける
        float kept_alpha = base.a * (1.0 - share);
        float fill_alpha = base.a * share * amount;
        float alpha = kept_alpha + fill_alpha;
        vec3 rgb = (kept * kept_alpha + paint.rgb * fill_alpha) / max(alpha, 1e-6);
        frag_color = alpha > 0.0 ? vec4(to_linear(rgb), alpha) : vec4(0.0);
        return;
    }
    vec3 rgb = mix(under, blended(under, paint.rgb), amount);
    frag_color = vec4(to_linear(rgb), base.a);
}
"""
)


def _stop_parameters() -> tuple[ColorSpec | TrackSpec, ...]:
    specs: list[ColorSpec | TrackSpec] = []
    defaults = ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    for index in range(MAX_STOPS):
        colour = defaults[min(index, 1)]
        specs.append(ColorSpec(f"color{index}", f"色 {index + 1}", colour))
        specs.append(
            TrackSpec(f"offset{index}", f"位置 {index + 1}", 0, 1, min(index, 1), step=0.001)
        )
    return tuple(specs)


#: 前からの色の数・色・位置の欄（YMM4 の読み込みが書く形）
_STOP_COUNT = ValueSpec("stops", "色の数", 2, minimum=1, maximum=MAX_STOPS)
_STOP_FIELDS = _stop_parameters()
_STOP_COLOURS = tuple(spec for spec in _STOP_FIELDS if isinstance(spec, ColorSpec))
_STOP_OFFSETS = tuple(spec for spec in _STOP_FIELDS if isinstance(spec, TrackSpec))
_LEGACY_NAMES = frozenset({_STOP_COUNT.name, *(spec.name for spec in _STOP_FIELDS)})


def _legacy_count(params: Mapping[str, ParamValue]) -> int:
    return _STOP_COUNT.coerce(params.get(_STOP_COUNT.name))


def _legacy_stops(params: Mapping[str, ParamValue]) -> tuple[float, ...]:
    """前からの「色 1」「位置 1」… を止まりの並びにする（#364 設定パネルのバーに見せる）

    位置はキーフレームの無い値（動く位置はバーに出さず、前からの欄で見せる :func:`_paint_unused`）
    色が 1 つ（単色の形）なら同じ色を両端に置く バーは 2 つより少ない止まりを持てない
    """
    count = _legacy_count(params)
    stops: list[tuple[float, tuple[float, float, float, float]]] = []
    for index in range(count):
        colour_spec, offset_spec = _STOP_COLOURS[index], _STOP_OFFSETS[index]
        red, green, blue, alpha = colour_spec.coerce(params.get(colour_spec.name))
        offset = offset_spec.coerce(params.get(offset_spec.name)).static
        stops.append((offset, (red, green, blue, alpha)))
    if count == 1:
        stops = [(0.0, stops[0][1]), (1.0, stops[0][1])]
    return _STOPS.coerce(GradientSpec.pack(stops))


def _moving_offsets(params: Mapping[str, ParamValue]) -> bool:
    """前からの位置のどれかがキーフレームで動くか"""
    count = _legacy_count(params)
    for spec in _STOP_OFFSETS[:count]:
        value = params.get(spec.name)
        if isinstance(value, AnimatedValue) and value.is_animated:
            return True
    return False


def _paint_unused(params: Mapping[str, ParamValue]) -> frozenset[str]:
    """「模様で塗る」の欄で出さない項目（#364）

    色の止まりはバー 1 つで見せ、前からの 33 の欄（色の数・色 1・位置 1 …）を隠す 単色の模様は
    バーではなく「色 1」の欄だけを出す（描く所も単色は色 1 で塗る）
    前からの位置がキーフレームで動くときは、バーに出すと動きを見せられず、触ると動きが
    消えるので、前からの欄（使っている色の数のぶん）を出してバーを隠す
    """
    if params.get("pattern", "linear") == "solid":
        return (_LEGACY_NAMES - {"color0"}) | {_STOPS.name}
    if not _STOPS.coerce(params.get(_STOPS.name)) and _moving_offsets(params):
        count = _legacy_count(params)
        used = {_STOP_COUNT.name}
        used |= {spec.name for spec in _STOP_COLOURS[:count]}
        used |= {spec.name for spec in _STOP_OFFSETS[:count]}
        return (_LEGACY_NAMES - used) | {_STOPS.name}
    return _LEGACY_NAMES


#: 色のバーで並べる止まり（#364） 空の間は前からの「色 1」「位置 1」… で塗る（YMM4 の
#: 読み込みと前の版の作品はどれもこの形で、値も絵も前のまま） バーで触った時から並びで塗る
_STOPS = GradientSpec("gradient", "色", default=(), legacy=_legacy_stops)


def register_paint_effects() -> None:
    registry.register(
        EffectDefinition(
            kind="brush_fill",
            label="模様で塗る",
            category="色",
            keeps_content=True,
            parameters=(
                SelectSpec("pattern", "模様", PATTERNS, "linear"),
                SelectSpec("blend", "合成", BLEND_MODES, "normal"),
                TrackSpec("opacity", "濃さ", 0, 100, 100, unit="%"),
                CheckSpec("pattern_only", "模様だけで塗る", False),
                CheckSpec("key_only", "目印の色の所だけ塗る", False),
                ColorSpec("key_color", "目印の色", (1.0, 0.0, 1.0, 1.0)),
                # 不透明度 0 は「渡していない」印 前の版の保存は線の色を持たないので、
                # 目印からの近さで塗りの所を決める
                ColorSpec("keep_color", "目印と一緒に描いた線の色", (0.0, 0.0, 0.0, 0.0)),
                # 模様を塗らずにノイズの値で薄める（YMM4 の NoiseEffect）
                SelectSpec("noise_mask", "ノイズで薄める", _NOISE_MASKS, "off"),
                _STOPS,
                _STOP_COUNT,
                *_STOP_FIELDS,
                SelectSpec("extend", "端の扱い", _EXTEND, "clamp"),
                TrackSpec("center_x", "中心 X", -8000, 8000, 0, step=1, unit="px"),
                TrackSpec("center_y", "中心 Y（下が正）", -8000, 8000, 0, step=1, unit="px"),
                TrackSpec("angle", "角度", -3600, 3600, 0, unit="度"),
                TrackSpec("size", "長さ", 1, 20000, 400, step=1, unit="px"),
                TrackSpec("offset", "ずらし", -20000, 20000, 0, step=1, unit="px"),
                TrackSpec("radius_x", "半径 X", 1, 20000, 300, step=1, unit="px"),
                TrackSpec("radius_y", "半径 Y", 1, 20000, 300, step=1, unit="px"),
                TrackSpec("width_a", "帯の幅 1", 0, 4000, 40, step=1, unit="px"),
                TrackSpec("width_b", "帯の幅 2", 0, 4000, 40, step=1, unit="px"),
                TrackSpec("dot_radius", "水玉の半径", 0, 2000, 10, step=1, unit="px"),
                TrackSpec("span", "水玉の間隔", 1, 4000, 40, step=1, unit="px"),
                TrackSpec("aspect", "縦横比", -100, 100, 0, unit="%"),
                TrackSpec("thickness", "線の太さ", 0, 2000, 2, step=1, unit="px"),
                TrackSpec("cell_width", "格子の幅", 1, 4000, 40, step=1, unit="px"),
                TrackSpec("cell_height", "格子の高さ", 1, 4000, 40, step=1, unit="px"),
                TrackSpec("zoom", "拡大率", 1, 10000, 100, unit="%"),
                CheckSpec("inverted", "反転", False),
                CheckSpec("relative", "長さを絵の幅の割合で決める", False),
                SelectSpec("noise_kind", "ノイズの種類", NOISE_KINDS, "perlin"),
                TrackSpec("noise_strength", "ノイズの強さ", 0, 1000, 100, unit="%"),
                TrackSpec("noise_threshold", "ノイズのしきい値", 0, 100, 0, unit="%"),
                TrackSpec("noise_levels", "ノイズの段階", 2, 256, 256, step=1),
                ValueSpec("noise_octaves", "ノイズの重ね数", 5, minimum=1, maximum=8),
                CheckSpec("turbulence", "乱流", False),
                CheckSpec("colorful", "ノイズを色ごとに変える", False),
                TrackSpec("noise_scale_x", "ノイズの横の大きさ", 1, 100000, 100, unit="%"),
                TrackSpec("noise_scale_y", "ノイズの縦の大きさ", 1, 100000, 100, unit="%"),
                TrackSpec("noise_x", "ノイズの位置 X", -100000, 100000, 0, step=1, unit="px"),
                TrackSpec("noise_y", "ノイズの位置 Y", -100000, 100000, 0, step=1, unit="px"),
                TrackSpec("noise_z", "ノイズの奥行き", -100000, 100000, 0, step=0.01),
                # 速さは 1 秒あたりの画面の画素 単位の表示が px/秒 なので画素の長さだと明に書く
                # 書かないと下書きでノイズの流れだけが 2〜4 倍に速く見える
                TrackSpec(
                    "speed_x", "ノイズの速さ X", -100000, 100000, 0, unit="px/秒", pixels=True
                ),
                TrackSpec(
                    "speed_y", "ノイズの速さ Y", -100000, 100000, 0, unit="px/秒", pixels=True
                ),
                TrackSpec("speed_z", "奥行きの速さ", -1000, 1000, 0, step=0.01, unit="/秒"),
            ),
            fragment_shader=_PAINT,
            unused=_paint_unused,
        )
    )
