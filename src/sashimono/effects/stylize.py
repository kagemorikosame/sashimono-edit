"""見た目を加工するエフェクト 形を削る・色を変える・ぼかす・グリッチ

YMM4 の配布テンプレートに出てくるものを、同じ効き方になるように作ったもの
（:mod:`sashimono.compat.ymm4.effects` が写す） 値の意味は配布物に入っていた値の
並びから読み取っている（:mod:`sashimono.effects.motion` の説明と同じ事情）

シェーダはリニア空間・ストレートアルファで受け取り、同じ形で返す
"""

from __future__ import annotations

from sashimono.effects.blending import BLEND_FUNCTIONS, BLEND_MODES
from sashimono.effects.builtin import PRELUDE
from sashimono.effects.definition import EffectDefinition, registry
from sashimono.effects.spec import CheckSpec, ColorSpec, SelectSpec, TrackSpec

__all__ = ["register_stylize_effects"]


def _shader(body: str) -> str:
    return PRELUDE + body


#: リニアと sRGB の行き来 反転は見た目（sRGB）で行う リニアのまま 1 から引くと、
#: 中間の灰色が反転しても灰色にならず、白っぽく浮く
_SRGB = """
vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
"""


_MORPHOLOGY = _shader("""
uniform int mode;
uniform float width;
uniform float height;

void main() {
    // 横と縦に分けて調べる 膨張は周りで一番濃い画素、収縮は一番薄い画素を採る
    bool horizontal = u_pass == 0;
    float radius = horizontal ? width : height;
    vec2 direction = horizontal ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 chosen = texture(u_texture, v_uv);
    int taps = int(min(radius, 64.0));
    for (int i = -taps; i <= taps; ++i) {
        vec4 c = texture(u_texture, v_uv + direction * float(i) / u_size);
        if (mode == 0 ? c.a > chosen.a : c.a < chosen.a) chosen = c;
    }
    frag_color = chosen;
}
""")

_CROP_ANGLE = _shader("""
uniform float center_x;
uniform float center_y;
uniform float angle;
uniform float width;
uniform float blur;
uniform int pivot_h;
uniform int pivot_v;
uniform float anchor_x;
uniform float anchor_y;

void main() {
    // 中心を通り angle の向きに伸びる帯（幅 width）だけを残す
    // 中心は基準の点から測る 既定は絵の中央 YMM4 で前に中心点があれば、その点が基準に
    // なる（SFっぽい吹き出し(右) の名札は左端から X の所で切れていた） 並びは変形の
    // 中心の選び方と同じ
    vec2 base = object_center();
    if (pivot_h == 0) base.x = u_size.x * 0.5;
    if (pivot_v == 0) base.y = u_size.y * 0.5;
    if (pivot_h == 1) base.x = u_object.x;
    if (pivot_h == 2) base.x = u_object.z;
    if (pivot_v == 1) base.y = u_object.w;
    if (pivot_v == 2) base.y = u_object.y;
    if (pivot_h == 4) base.x = object_origin().x;
    if (pivot_v == 4) base.y = object_origin().y;
    vec2 centre = base + vec2(anchor_x, anchor_y) + vec2(center_x, center_y);
    vec2 pixel = v_uv * u_size;
    vec2 normal = vec2(cos(radians(angle)), sin(radians(angle)));
    float distance = abs(dot(pixel - centre, normal));
    float edge = max(blur, 0.5);
    float keep = 1.0 - smoothstep(width * 0.5 - edge, width * 0.5 + edge, distance);
    vec4 color = texture(u_texture, v_uv);
    frag_color = vec4(color.rgb, color.a * keep);
}
""")

_CROP_SLANT = _shader("""
uniform float center_x;
uniform float center_y;
uniform float angle;
uniform float blur;
uniform float width;

void main() {
    // 中心を通る線の片側を切り落とす（AviUtl の 斜めクリッピング）
    // 角度 0 で線は横向き、中心より下を落として上を残す 角度は画面上で時計回りが正
    // どちらも sigma の 単純図形σ が菱形を 4 回の切り落としで作る並びから読んだ
    // （向きを 1 つでも取り違えると、角の三角形ではなく真ん中を落として何も残らない）
    // 画面の Y 下向きで落とす側は (-sin, cos) ここは Y 上向きなので (-sin, -cos)
    vec2 p = v_uv * u_size - object_center() - vec2(center_x, center_y);
    float r = radians(angle);
    float side = dot(p, vec2(-sin(r), -cos(r)));
    float edge = max(blur * 0.5, 0.5);
    float keep = 1.0 - smoothstep(-edge, edge, side);
    // 幅が正なら線を真ん中にした幅の帯だけを残し、負なら帯を消して両側を残す AviUtl2 に
    // 白い四角 300 の真ん中へ幅 100 と -100 で描かせると、98 の帯だけ残る・帯だけ消える（#188）
    if (abs(width) > 0.0) {
        float half_width = abs(width) * 0.5;
        float inside = 1.0 - smoothstep(half_width - edge, half_width + edge, abs(side));
        keep = width > 0.0 ? inside : 1.0 - inside;
    }
    vec4 color = texture(u_texture, v_uv);
    frag_color = vec4(color.rgb, color.a * keep);
}
""")

_ROUND_CORNER = _shader("""
uniform float radius;
uniform float blur;

void main() {
    // 絵が置かれた範囲を角丸の四角で切り抜く
    vec2 half_size = object_size() * 0.5;
    vec2 point = abs(v_uv * u_size - object_center());
    float r = clamp(radius, 0.0, min(half_size.x, half_size.y));
    vec2 q = point - (half_size - vec2(r));
    float distance = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    float keep = 1.0 - smoothstep(-max(blur, 0.5), max(blur, 0.5), distance);
    vec4 color = texture(u_texture, v_uv);
    frag_color = vec4(color.rgb, color.a * keep);
}
""")

_EDGE_TRIM = _shader("""
uniform float thickness;

void main() {
    // 輪郭を内側へ削る 周りに透明な画素があれば、その分だけ薄くする
    vec4 color = texture(u_texture, v_uv);
    if (thickness < 0.5 || color.a <= 0.0) { frag_color = color; return; }
    float coverage = color.a;
    int steps = int(min(thickness, 32.0));
    for (int y = -steps; y <= steps; ++y) {
        for (int x = -steps; x <= steps; ++x) {
            vec2 offset = vec2(float(x), float(y));
            if (length(offset) > thickness) continue;
            coverage = min(coverage, texture(u_texture, v_uv + offset / u_size).a);
        }
    }
    frag_color = vec4(color.rgb, coverage);
}
""")

_EXPOSURE = _shader("""
uniform float amount;

void main() {
    // 100% で元のまま 0% で真っ暗、200% で光の量が 2 倍
    vec4 color = texture(u_texture, v_uv);
    frag_color = vec4(color.rgb * max(amount, 0.0) / 100.0, color.a);
}
""")

_HALFTONE_BORDER = _shader("""
uniform float blur;
uniform int grid;
uniform float spacing;
uniform float dot_size;
uniform float strength;

void main() {
    // 輪郭のぼけた部分を、網点（ドットの大きさ）で表す
    vec4 color = texture(u_texture, v_uv);
    if (blur < 0.5) { frag_color = color; return; }

    float soft = 0.0;
    float total = 0.0;
    for (int ring = 0; ring <= 4; ++ring) {
        float r = blur * float(ring) / 4.0;
        for (int k = 0; k < 8; ++k) {
            float a = PI * 0.25 * float(k);
            soft += texture(u_texture, v_uv + vec2(cos(a), sin(a)) * r / u_size).a;
            total += 1.0;
        }
    }
    soft /= total;

    vec2 pixel = v_uv * u_size;
    // 下限は画面の 1 画素（縮小表示なら 1 画素より細かい） 1.0 のままだと 1/4 表示で
    // 間隔 1〜4 がみな同じ粗さになり、書き出しと網目の細かさが食い違う
    float pitch = max(spacing, u_pixel_scale) * 4.0;
    if (grid == 0) {
        // 菱形 格子を 45 度回す
        pixel = mat2(0.7071, 0.7071, -0.7071, 0.7071) * pixel;
    }
    vec2 cell = fract(pixel / pitch) - 0.5;
    float reach = soft * 0.7071 * dot_size / 100.0;
    float dotted = 1.0 - smoothstep(reach - 0.05, reach + 0.05, length(cell));
    float alpha = mix(soft, dotted, clamp(strength / 100.0, 0.0, 1.0) * step(soft, 0.999));
    frag_color = vec4(color.a > 0.0 ? color.rgb : texture(u_source, v_uv).rgb, alpha);
}
""")

_STRIPE_GLITCH = _shader("""
uniform float count;
uniform float max_width;
uniform float max_shift;
uniform float color_shift;
uniform float rate;
uniform bool hard;
uniform float width_attenuation;
uniform float shift_attenuation;
uniform float repeat;

void main() {
    // 横の帯をいくつか選び、左右へずらす 帯と量は rate 回/秒で選び直す
    vec2 pixel = v_uv * u_size;
    float tick = floor(u_time * max(rate, 0.0));
    float height = object_size().y;
    float shift = 0.0;
    int stripes = int(clamp(count, 0.0, 64.0));
    // 繰り返しは帯の組を別の乱数で何度か選び直して重ねる
    int rounds = int(clamp(repeat, 1.0, 16.0));
    for (int round_ = 0; round_ < rounds; ++round_)
    for (int i = 0; i < stripes; ++i) {
        float salt = float(i) * 3.7 + tick * 11.3 + float(round_) * 101.9;
        float fade_width = 1.0 / (1.0 + float(i) * width_attenuation / 100.0);
        float fade_shift = 1.0 / (1.0 + float(i) * shift_attenuation / 100.0);
        float centre = u_object.y + hash(vec2(salt, 1.0)) * height;
        float half_height = hash(vec2(salt, 2.0)) * height * max_width / 100.0 * 0.5 * fade_width;
        float inside = hard
            ? step(abs(pixel.y - centre), half_height)
            : 1.0 - smoothstep(half_height * 0.7, half_height, abs(pixel.y - centre));
        shift += inside * (hash(vec2(salt, 3.0)) * 2.0 - 1.0) * max_shift * fade_shift;
    }
    vec2 source = pixel - vec2(shift, 0.0);
    vec4 red = sample_pixel(source + vec2(color_shift, 0.0));
    vec4 middle = sample_pixel(source);
    vec4 blue = sample_pixel(source - vec2(color_shift, 0.0));
    frag_color = vec4(red.r, middle.g, blue.b, max(max(red.a, middle.a), blue.a));
}
""")

_LONG_SHADOW = _shader("""
uniform float angle;
uniform float length_;
uniform float opacity;
uniform float attenuation;
uniform int shadow_type;
uniform vec4 color1;
uniform vec4 color2;

void main() {
    // 絵を angle の向きへ length_ 画素ぶん引き伸ばした影を、絵の後ろに敷く
    vec4 base = texture(u_texture, v_uv);
    vec2 direction = vec2(cos(radians(angle)), sin(radians(angle)));
    int taps = int(clamp(length_, 0.0, 512.0));
    float coverage = 0.0;
    float where = 0.0;
    vec3 image = vec3(0.0);
    for (int i = 1; i <= taps; ++i) {
        float t = float(i) / max(float(taps), 1.0);
        vec4 found = texture(u_texture, v_uv - direction * float(i) / u_size);
        float a = found.a * (1.0 - clamp(attenuation / 100.0, 0.0, 1.0) * t);
        if (a > coverage) { coverage = a; where = t; image = found.rgb; }
    }
    // 画像: 絵そのものの色で伸ばす（YMM4 の ShadowType が Image のとき）
    vec4 tint = shadow_type == 0 ? color1
        : shadow_type == 2 ? vec4(image, 1.0)
        : mix(color1, color2, where);
    vec4 shadow = vec4(tint.rgb, tint.a * coverage * clamp(opacity / 100.0, 0.0, 1.0));
    frag_color = over(base, shadow);
}
""")

_INNER_SHADOW = _shader(
    _SRGB
    + BLEND_FUNCTIONS
    + """
uniform float offset_x;
uniform float offset_y;
uniform float blur;
uniform float opacity;
uniform vec4 color;
uniform int blend;

void main() {
    if (u_pass == 0) {
        // 影は絵の内側で、ずらした絵に隠れない所に落ちる まず横にぼかす
        // ずらす向きは影が落ちる向き（Y は上が正）
        vec2 shift = -vec2(offset_x, offset_y) / u_size;
        float covered = blur1d(u_texture, v_uv + shift, vec2(1.0, 0.0), blur).a;
        frag_color = vec4(1.0, 1.0, 1.0, 1.0 - covered);
        return;
    }
    float shadow = blur1d(u_texture, v_uv, vec2(0.0, 1.0), blur).a;
    vec4 base = texture(u_source, v_uv);
    // 合成は sRGB で行う（YMM4 と同じ）
    vec3 under = to_srgb(base.rgb);
    vec3 mixed = blend_colors(blend, under, to_srgb(color.rgb));
    float amount = clamp(shadow * color.a * opacity * 0.01, 0.0, 1.0);
    frag_color = vec4(to_linear(mix(under, mixed, amount)), base.a);
}
"""
)

_INNER_HALFTONE = _shader(
    _SRGB
    + BLEND_FUNCTIONS
    + """
uniform float offset_x;
uniform float offset_y;
uniform float blur;
uniform float opacity;
uniform vec4 color;
uniform int blend;
uniform int grid;
uniform float spacing;
uniform float dot_size;
uniform float strength;

const mat2 TURN = mat2(0.7071, 0.7071, -0.7071, 0.7071);

void main() {
    if (u_pass == 0) {
        vec2 shift = -vec2(offset_x, offset_y) / u_size;
        float covered = blur1d(u_texture, v_uv + shift, vec2(1.0, 0.0), blur).a;
        frag_color = vec4(1.0, 1.0, 1.0, 1.0 - covered);
        return;
    }
    // 影の濃さを網点の大きさで表す 点の中心で影を読み、濃さの平方根に比例した半径で描く
    // 濃さ 1 で半径が格子の対角の半分になり、隣の点と重なって隙間なく塗る
    // 半径は隣のマスまで届くので、周りの 9 マスの点を調べる
    // 下限は画面の 1 画素 網点の輪郭ぼかしと同じく、縮小表示の 1 画素に合わせると粗くなる
    float pitch = max(spacing, u_pixel_scale);
    vec2 pixel = v_uv * u_size - object_center();
    vec2 lattice = grid == 0 ? TURN * pixel : pixel;
    vec2 cell = floor(lattice / pitch);
    float dotted = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 centre = (cell + vec2(i, j) + 0.5) * pitch;
            vec2 at = grid == 0 ? transpose(TURN) * centre : centre;
            vec2 uv = (at + object_center()) / u_size;
            float amount = blur1d(u_texture, uv, vec2(0.0, 1.0), blur).a;
            float reach = pitch * 0.7072 * sqrt(max(amount, 0.0)) * dot_size / 100.0;
            float edge = clamp(reach - length(lattice - centre) + 0.5, 0.0, 1.0);
            dotted = max(dotted, edge);
        }
    }
    float shadow = blur1d(u_texture, v_uv, vec2(0.0, 1.0), blur).a;
    float halftone = mix(shadow, dotted, clamp(strength / 100.0, 0.0, 1.0));
    vec4 base = texture(u_source, v_uv);
    vec3 under = to_srgb(base.rgb);
    vec3 mixed = blend_colors(blend, under, to_srgb(color.rgb));
    float amount = clamp(halftone * color.a * opacity * 0.01, 0.0, 1.0);
    frag_color = vec4(to_linear(mix(under, mixed, amount)), base.a);
}
"""
)

_INNER_OUTLINE = _shader(
    _SRGB
    + BLEND_FUNCTIONS
    + """
uniform float thickness;
uniform float blur;
uniform float opacity;
uniform vec4 color;
uniform int blend;
uniform bool outline_only;
uniform bool angular;

void main() {
    if (u_pass == 0) {
        // 縁から thickness 以内の内側を帯にする 角ばらせるなら正方形、そうでなければ円で削る
        float t = min(max(thickness, 0.0), 256.0);
        float inside = texture(u_texture, v_uv).a;
        float kept = inside;
        if (t >= 0.5) {
            int rings = int(ceil(min(t, 64.0)));
            for (int ring = 1; ring <= rings; ++ring) {
                float r = t * float(ring) / float(rings);
                for (int k = 0; k < 32; ++k) {
                    float a = PI * 2.0 * float(k) / 32.0;
                    vec2 d = vec2(cos(a), sin(a));
                    if (angular) d /= max(abs(d.x), abs(d.y));
                    kept = min(kept, texture(u_texture, v_uv + d * r / u_size).a);
                }
            }
        }
        frag_color = vec4(1.0, 1.0, 1.0, clamp(inside - kept, 0.0, 1.0));
        return;
    }
    if (u_pass == 1) {
        frag_color = blur1d(u_texture, v_uv, vec2(1.0, 0.0), blur);
        return;
    }
    float band = blur1d(u_texture, v_uv, vec2(0.0, 1.0), blur).a;
    vec4 base = texture(u_source, v_uv);
    // ぼかした帯は絵の外へはみ出さない 内側の縁取りなので
    float amount = clamp(band * base.a * color.a * opacity * 0.01, 0.0, 1.0);
    if (outline_only) {
        frag_color = vec4(color.rgb, amount);
        return;
    }
    vec3 under = to_srgb(base.rgb);
    vec3 mixed = blend_colors(blend, under, to_srgb(color.rgb));
    frag_color = vec4(to_linear(mix(under, mixed, amount / max(base.a, 1e-4))), base.a);
}
"""
)

_SHAPE_MASK = _shader("""
uniform int shape;
uniform float width;
uniform float height;
uniform float corner;
uniform float span;
uniform float center_x;
uniform float center_y;
uniform float rotation;
uniform float blur;
uniform bool invert;

float sd_box(vec2 p, vec2 half_size, float radius) {
    vec2 q = abs(p) - (half_size - radius);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

// 頂点を上に向けた正三角形 外接円の半径 r
float sd_triangle(vec2 p, float r) {
    const float k = sqrt(3.0);
    p.y = -p.y - r * 0.25;
    float half_side = r * k * 0.5;
    p.x = abs(p.x) - half_side;
    p.y = p.y + half_side / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0 * half_side, 0.0);
    return -length(p) * sign(p.y);
}

void main() {
    // 図形の当たり判定は Y 下向きで書いてある（扇や矢印の向きがそのまま読める）
    // 設定の中心はほかのエフェクトと同じ Y 上向きなので、ここで符号を合わせる
    // 図形は絵の原点に置く 範囲の中央に置くと、場面切り替えで場面の図形が
    // 画面の端に寄っているとき、YMM4 は画面の中央に出す円が図形の真ん中へずれる
    vec2 p = v_uv * u_size - object_origin();
    p.y = -p.y;
    p -= vec2(center_x, -center_y);
    float r = radians(-rotation);
    p = mat2(cos(r), sin(r), -sin(r), cos(r)) * p;
    vec2 half_size = max(vec2(width, height) * 0.5, vec2(0.5));

    float d;
    if (shape == 0) {
        d = -1.0e6;
    } else if (shape == 1) {
        d = (length(p / half_size) - 1.0) * min(half_size.x, half_size.y);
    } else if (shape == 2) {
        d = sd_box(p, half_size, clamp(corner, 0.0, min(half_size.x, half_size.y)));
    } else if (shape == 3) {
        // 扇 上を 0 として反時計回りに span 度ぶん（YMM4 の CenterAngle）
        d = (length(p / half_size) - 1.0) * min(half_size.x, half_size.y);
        float theta = degrees(atan(p.x, -p.y));
        if (theta > 0.0) theta -= 360.0;
        if (theta < -clamp(span, 0.0, 360.0)) d = max(d, 1.0e6);
    } else {
        d = sd_triangle(p, min(half_size.x, half_size.y));
    }

    float edge = max(blur, 0.75);
    // ぼかしは縁の前後に広げる 幅は YMM4 の絵に近づけて 2 倍にした
    float inside = 1.0 - smoothstep(-edge, edge, d);
    if (invert) inside = 1.0 - inside;
    vec4 color = texture(u_texture, v_uv);
    frag_color = vec4(color.rgb, color.a * inside);
}
""")

_COPY_REVERSE = _shader("""
uniform int position;
uniform float distance;
uniform bool flip_horizontal;
uniform bool flip_vertical;
uniform bool centering;

// 絵の範囲の中で左右や上下を入れ替えて読む
vec4 mirrored(vec2 pixel, vec2 center) {
    if (flip_horizontal) pixel.x = 2.0 * center.x - pixel.x;
    if (flip_vertical) pixel.y = 2.0 * center.y - pixel.y;
    return sample_pixel(pixel);
}

void main() {
    // 並べる向き 0 右 1 左 2 下 3 上（画面の見た目 GL の Y は上が正）
    vec2 size = object_size();
    vec2 step_ = position == 0 ? vec2(size.x + distance, 0.0)
        : position == 1 ? vec2(-(size.x + distance), 0.0)
        : position == 2 ? vec2(0.0, -(size.y + distance))
        : vec2(0.0, size.y + distance);
    // 中央に寄せると、元と写しの組の真ん中が元の中心に来る
    vec2 base_shift = centering ? -step_ * 0.5 : vec2(0.0);
    vec2 pixel = v_uv * u_size;
    vec4 original = sample_pixel(pixel - base_shift);
    vec4 copy = mirrored(pixel - base_shift - step_, object_center());
    frag_color = over(original, copy);
}
""")

_FILL_BACKGROUND = _shader("""
uniform vec4 color;
uniform float opacity;
uniform float corner;
uniform float margin_top;
uniform float margin_bottom;
uniform float margin_left;
uniform float margin_right;
uniform bool background_only;

void main() {
    // 絵の範囲を上下左右に広げた角丸の四角を、絵の後ろに敷く（負の値なら狭める）
    vec2 pixel = v_uv * u_size;
    vec2 low = u_object.xy - vec2(margin_left, margin_bottom);
    vec2 high = u_object.zw + vec2(margin_right, margin_top);
    vec2 center = (low + high) * 0.5;
    vec2 half_size = max((high - low) * 0.5, vec2(0.0));
    float radius = clamp(corner, 0.0, min(half_size.x, half_size.y));
    vec2 q = abs(pixel - center) - (half_size - radius);
    float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
    float inside = 1.0 - smoothstep(-0.5, 0.5, d);
    vec4 plate = vec4(color.rgb, color.a * inside * clamp(opacity * 0.01, 0.0, 1.0));
    vec4 base = texture(u_texture, v_uv);
    frag_color = background_only ? plate : over(base, plate);
}
""")

_BINARIZE = _shader(
    _SRGB
    + """
uniform float threshold;
uniform bool invert;
uniform bool keep_color;

void main() {
    // 明るさ（sRGB の 3 色の平均）が threshold% 以上の所だけ残す 残した所は白か元の色
    vec4 base = texture(u_texture, v_uv);
    vec3 srgb = to_srgb(base.rgb);
    bool on = (srgb.r + srgb.g + srgb.b) / 3.0 * 100.0 >= threshold;
    if (invert) on = !on;
    vec3 rgb = keep_color ? base.rgb : vec3(1.0);
    frag_color = vec4(rgb, on ? base.a : 0.0);
}
"""
)

_COLOR_KEY = _shader(
    _SRGB
    + """
uniform vec4 key_color;
uniform float tolerance;
uniform bool feather;
uniform bool invert;

void main() {
    // 指定の色に近い所を抜く 近さは sRGB の 3 色の差の大きさ（0〜100）
    vec4 base = texture(u_texture, v_uv);
    float distance = length(to_srgb(base.rgb) - to_srgb(key_color.rgb)) / sqrt(3.0) * 100.0;
    float keep = feather
        ? smoothstep(tolerance * 0.5, max(tolerance, 0.0001), distance)
        : step(tolerance, distance);
    if (invert) keep = 1.0 - keep;
    frag_color = vec4(base.rgb, base.a * keep);
}
"""
)

_LINEAR_TRANSFER = _shader(
    _SRGB
    + """
uniform float red_slope;
uniform float red_intercept;
uniform float green_slope;
uniform float green_intercept;
uniform float blue_slope;
uniform float blue_intercept;
uniform float alpha_slope;
uniform float alpha_intercept;

void main() {
    // 色ごとに 傾き% × 値 + 切片% （sRGB で計算する）
    vec4 base = texture(u_texture, v_uv);
    vec3 c = to_srgb(base.rgb);
    vec3 slope = vec3(red_slope, green_slope, blue_slope) * 0.01;
    vec3 intercept = vec3(red_intercept, green_intercept, blue_intercept) * 0.01;
    vec3 rgb = clamp(c * slope + intercept, 0.0, 1.0);
    float a = clamp(base.a * alpha_slope * 0.01 + alpha_intercept * 0.01, 0.0, 1.0);
    frag_color = vec4(to_linear(rgb), a);
}
"""
)

_BORDER_BLUR = _shader("""
uniform float blur;

void main() {
    // 縁を内側へ向けて透明にする 不透明度をぼかし、半分まで下がった所で消える
    vec2 direction = u_pass == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    if (u_pass == 0) {
        float a = blur1d(u_texture, v_uv, direction, blur).a;
        frag_color = vec4(1.0, 1.0, 1.0, a);
        return;
    }
    float softened = blur1d(u_texture, v_uv, direction, blur).a;
    vec4 base = texture(u_source, v_uv);
    frag_color = vec4(base.rgb, base.a * smoothstep(0.5, 1.0, softened));
}
""")

_HIGHLIGHTS_SHADOWS = _shader("""
uniform float highlights;
uniform float shadows;

void main() {
    // 明るい部分と暗い部分を別々に持ち上げたり沈めたりする
    vec4 color = texture(u_texture, v_uv);
    float luma = dot(max(color.rgb, 0.0), LUMA);
    float bright = smoothstep(0.18, 1.0, luma);
    float dark = 1.0 - smoothstep(0.0, 0.18, luma);
    vec3 rgb = color.rgb * (1.0 + highlights / 100.0 * bright);
    rgb += (shadows / 100.0) * dark * 0.18;
    frag_color = vec4(max(rgb, 0.0), color.a);
}
""")

_COLOR_SHIFT = _shader("""
uniform float shift;
uniform float angle;
uniform float strength;
uniform int order;

void main() {
    // 3 つの色の成分を、angle の向きへ前・そのまま・後ろにずらす
    // order は、どの成分をどの位置に置くか（RGB・RBG・GRB・GBR・BRG・BGR）
    vec2 pixel = v_uv * u_size;
    vec2 offset = vec2(cos(radians(angle)), sin(radians(angle))) * shift * strength / 100.0;
    vec4 ahead = sample_pixel(pixel + offset);
    vec4 still = sample_pixel(pixel);
    vec4 behind = sample_pixel(pixel - offset);
    ivec3 slots[6] = ivec3[6](ivec3(0, 1, 2), ivec3(0, 2, 1), ivec3(1, 0, 2),
                              ivec3(1, 2, 0), ivec3(2, 0, 1), ivec3(2, 1, 0));
    ivec3 slot = slots[clamp(order, 0, 5)];
    vec4 picks[3] = vec4[3](ahead, still, behind);
    vec3 rgb = vec3(0.0);
    float alpha = 0.0;
    for (int channel = 0; channel < 3; ++channel) {
        vec4 chosen = picks[slot[channel]];
        rgb[channel] = chosen.rgb[channel] * chosen.a;
        alpha = max(alpha, chosen.a);
    }
    frag_color = alpha > 0.0001 ? vec4(rgb / alpha, alpha) : vec4(0.0);
}
""")

_RADIAL_BLUR = _shader("""
uniform float amount;
uniform float center_x;
uniform float center_y;
uniform bool hard;

void main() {
    // 中心から外へ向かって伸ばす
    vec2 centre = object_center() + vec2(center_x, center_y);
    vec2 pixel = v_uv * u_size;
    vec4 sum = vec4(0.0);
    float span = clamp(amount / 100.0, 0.0, 1.0) * 0.5;
    for (int i = 0; i < 48; ++i) {
        float scale = 1.0 - span * float(i) / 47.0;
        sum += premul(sample_pixel(centre + (pixel - centre) * scale));
    }
    vec4 result = unpremul(sum / 48.0);
    if (hard) result.a = texture(u_texture, v_uv).a;
    frag_color = result;
}
""")

_CIRCULAR_BLUR = _shader("""
uniform float angle;
uniform float center_x;
uniform float center_y;
uniform bool hard;

void main() {
    // 中心の周りに回す向きへ伸ばす
    vec2 centre = object_center() + vec2(center_x, center_y);
    vec2 point = v_uv * u_size - centre;
    vec4 sum = vec4(0.0);
    for (int i = 0; i < 48; ++i) {
        float a = radians(angle) * (float(i) / 47.0 - 0.5);
        float c = cos(a);
        float s = sin(a);
        sum += premul(sample_pixel(centre + mat2(c, s, -s, c) * point));
    }
    vec4 result = unpremul(sum / 48.0);
    if (hard) result.a = texture(u_texture, v_uv).a;
    frag_color = result;
}
""")

_INVERT = _shader(
    _SRGB
    + """
void main() {
    vec4 color = texture(u_texture, v_uv);
    frag_color = vec4(to_linear(1.0 - to_srgb(color.rgb)), color.a);
}
"""
)

_TINT = _shader("""
uniform vec4 color;

void main() {
    // 明るさを残したまま、色だけを指定の色にする
    vec4 base = texture(u_texture, v_uv);
    float luma = dot(max(base.rgb, 0.0), LUMA);
    float tint_luma = max(dot(color.rgb, LUMA), 0.0001);
    vec3 rgb = color.rgb * (luma / tint_luma);
    frag_color = vec4(mix(base.rgb, rgb, color.a), base.a);
}
""")

_EDGE_DETECT = _shader("""
uniform float strength;
uniform float radius;
uniform int mode;
uniform bool overlay;

float luma_at(vec2 offset) {
    vec4 c = texture(u_texture, v_uv + offset / u_size);
    return dot(max(c.rgb, 0.0), LUMA) * c.a;
}

void main() {
    // 明るさの変わり目を線にする ソーベルとプレウィットは重みだけが違う
    float d = max(radius, 1.0);
    float w = mode == 0 ? 2.0 : 1.0;
    float gx = -luma_at(vec2(-d, d)) - w * luma_at(vec2(-d, 0.0)) - luma_at(vec2(-d, -d))
             + luma_at(vec2(d, d)) + w * luma_at(vec2(d, 0.0)) + luma_at(vec2(d, -d));
    float gy = -luma_at(vec2(-d, -d)) - w * luma_at(vec2(0.0, -d)) - luma_at(vec2(d, -d))
             + luma_at(vec2(-d, d)) + w * luma_at(vec2(0.0, d)) + luma_at(vec2(d, d));
    float edge = clamp(length(vec2(gx, gy)) * strength / 25.0, 0.0, 1.0);
    vec4 base = texture(u_texture, v_uv);
    if (overlay) {
        frag_color = vec4(base.rgb + vec3(edge), base.a);
    } else {
        frag_color = vec4(vec3(edge), base.a);
    }
}
""")


_EMBOSS = _shader(
    _SRGB
    + """
uniform float angle;
uniform float height;
uniform float amount;
uniform int basis;
uniform bool keep_color;
uniform float reach;

// 高さは見た目（sRGB）の値で取る リニアのままだと暗い所の段差がつぶれ、明るい所ばかり
// 浮き上がって、Photoshop や Premiere のエンボスと凹凸の付き方が変わる
// 色に α を掛けて、透明な所を一番低い所と見る 掛けないと、透明な画素に残った色
// （多くは黒）の段差が縁に出たり出なかったりする
vec3 height_at(vec2 pixel) {
    vec4 c = sample_pixel(pixel);
    if (basis == 2) return vec3(c.a);
    vec3 s = to_srgb(c.rgb) * c.a;
    if (basis == 1) return s;
    return vec3(dot(s, LUMA));
}

void main() {
    vec4 base = texture(u_texture, v_uv);
    // 光の方向は右が 0 度で反時計回り（Y は上が正） Photoshop の角度と同じ数え方
    vec2 light = vec2(cos(radians(angle)), sin(radians(angle)));
    vec2 pixel = v_uv * u_size;
    vec2 step_to_light = light * max(reach, 0.0);
    // 光から遠い側が高いほど、光の方を向いた斜面になって明るい 1 画素の隣だけ見ると
    // 取り込み幅を広げても線が太らないので、幅だけ離れた 2 点の差を取る
    vec3 relief = (height_at(pixel - step_to_light) - height_at(pixel + step_to_light))
                * (height / 100.0) * 0.5;
    vec3 embossed;
    if (keep_color) {
        // 元の色へは、凹凸で変わった分だけをリニアの差で足す sRGB へ直した値で置き換えると、
        // 露出などで 1 を超えた明るさが 1 に詰められ、高さ 0 でも元の絵から変わる
        // 1 を超える所に照る側の凹凸を足しても sRGB の 1 で止まるので明るくはならず、
        // 陰の側だけが差の分暗くなる
        vec3 shown = to_srgb(base.rgb);
        embossed = base.rgb + (to_linear(clamp(shown + relief, 0.0, 1.0)) - to_linear(shown));
    } else {
        // 灰色の浮き彫りは元の色を残さない新しい面なので 0..1 に収める 高さも to_srgb で
        // 1 に詰めて読む 画面に出る白より明るい所は、白と同じ高さの平らな所に見える
        embossed = to_linear(clamp(vec3(0.5) + relief, 0.0, 1.0));
    }
    // なじませるのはリニアで行う 量 0 で元の値がそのまま残り、1 を超える明るさも切れない
    float weight = clamp(amount / 100.0, 0.0, 1.0);
    vec3 rgb = mix(base.rgb, embossed, weight);
    frag_color = vec4(rgb, base.a);
}
"""
)


def register_stylize_effects() -> None:
    """加工のエフェクトを一覧へ登録する 何度呼んでも 1 回だけ"""
    if "morphology" in registry:
        return

    definitions = (
        EffectDefinition(
            kind="morphology",
            label="膨張・収縮",
            category="形",
            parameters=(
                SelectSpec("mode", "方法", (("dilate", "膨張"), ("erode", "収縮")), "dilate"),
                TrackSpec("width", "横", 0, 64, 2, step=1, unit="px"),
                TrackSpec("height", "縦", 0, 64, 2, step=1, unit="px"),
            ),
            fragment_shader=_MORPHOLOGY,
            passes=2,
        ),
        EffectDefinition(
            kind="crop_angle",
            label="角度で切り抜き",
            category="形",
            parameters=(
                TrackSpec("center_x", "中心 X", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("center_y", "中心 Y", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("angle", "角度", -360, 360, 0, unit="度"),
                TrackSpec("width", "幅", 0, 8000, 400, step=1, unit="px"),
                TrackSpec("blur", "ぼかし", 0, 400, 0, unit="px"),
                SelectSpec(
                    "pivot_h",
                    "基準の横",
                    (
                        ("screen", "画面の中央"),
                        ("left", "絵の左端"),
                        ("right", "絵の右端"),
                        ("center", "絵の中央"),
                        ("origin", "絵の原点"),
                    ),
                    "center",
                ),
                SelectSpec(
                    "pivot_v",
                    "基準の縦",
                    (
                        ("screen", "画面の中央"),
                        ("top", "絵の上端"),
                        ("bottom", "絵の下端"),
                        ("middle", "絵の中央"),
                        ("origin", "絵の原点"),
                    ),
                    "middle",
                ),
                TrackSpec("anchor_x", "基準のずれ X", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("anchor_y", "基準のずれ Y", -4000, 4000, 0, step=1, unit="px"),
            ),
            fragment_shader=_CROP_ANGLE,
        ),
        EffectDefinition(
            kind="crop_slant",
            label="斜めクリッピング",
            category="形",
            parameters=(
                TrackSpec("center_x", "中心 X", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("center_y", "中心 Y", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("angle", "角度", -360, 360, 0, unit="度"),
                TrackSpec("blur", "ぼかし", 0, 400, 0, unit="px"),
                TrackSpec("width", "幅", -8000, 8000, 0, step=1, unit="px"),
            ),
            fragment_shader=_CROP_SLANT,
        ),
        EffectDefinition(
            kind="round_corner",
            label="角丸",
            category="形",
            parameters=(
                TrackSpec("radius", "半径", 0, 4000, 20, step=1, unit="px"),
                TrackSpec("blur", "ぼかし", 0, 100, 1, unit="px"),
            ),
            fragment_shader=_ROUND_CORNER,
        ),
        EffectDefinition(
            kind="edge_trim",
            label="輪郭を削る",
            category="形",
            parameters=(TrackSpec("thickness", "太さ", 0, 32, 2, step=1, unit="px"),),
            fragment_shader=_EDGE_TRIM,
        ),
        EffectDefinition(
            kind="exposure",
            label="露出",
            category="色",
            keeps_content=True,
            parameters=(TrackSpec("amount", "露出", 0, 1000, 100, unit="%"),),
            fragment_shader=_EXPOSURE,
        ),
        EffectDefinition(
            kind="halftone_border",
            label="網点の輪郭ぼかし",
            category="形",
            parameters=(
                TrackSpec("blur", "ぼかし", 0, 400, 20, unit="px"),
                SelectSpec("grid", "並び", (("rhombus", "菱形"), ("square", "正方形")), "rhombus"),
                TrackSpec("spacing", "間隔", 1, 100, 5, step=1, unit="px"),
                TrackSpec("dot_size", "点の大きさ", 0, 200, 100, unit="%"),
                TrackSpec("strength", "強さ", 0, 100, 100, unit="%"),
            ),
            fragment_shader=_HALFTONE_BORDER,
        ),
        EffectDefinition(
            kind="stripe_glitch",
            label="横ずれグリッチ",
            category="装飾",
            parameters=(
                TrackSpec("count", "本数", 0, 64, 10, step=1),
                TrackSpec("max_width", "帯の最大幅", 0, 100, 10, unit="%"),
                TrackSpec("max_shift", "最大のずれ", 0, 4000, 100, step=1, unit="px"),
                TrackSpec("color_shift", "色ずれ", 0, 400, 5, step=1, unit="px"),
                TrackSpec("rate", "切り替え", 0, 240, 30, step=1, unit="回/秒"),
                CheckSpec("hard", "境目をぼかさない", False),
                TrackSpec("width_attenuation", "幅の減衰", 0, 1000, 10, unit="%"),
                TrackSpec("shift_attenuation", "ずれの減衰", 0, 1000, 50, unit="%"),
                TrackSpec("repeat", "重ねる回数", 1, 16, 1, step=1),
            ),
            fragment_shader=_STRIPE_GLITCH,
        ),
        EffectDefinition(
            kind="long_shadow",
            label="長い影",
            category="装飾",
            parameters=(
                TrackSpec("angle", "角度", -360, 360, -45, unit="度"),
                TrackSpec("length_", "長さ", 0, 512, 50, step=1, unit="px"),
                TrackSpec("opacity", "濃さ", 0, 100, 100, unit="%"),
                TrackSpec("attenuation", "薄れ", 0, 100, 0, unit="%"),
                SelectSpec(
                    "shadow_type",
                    "塗り",
                    (("solid", "単色"), ("gradient", "グラデーション"), ("image", "絵の色")),
                    "solid",
                ),
                ColorSpec("color1", "色 1", (0.0, 0.0, 0.0, 1.0)),
                ColorSpec("color2", "色 2", (0.0, 0.0, 0.0, 0.0)),
            ),
            fragment_shader=_LONG_SHADOW,
        ),
        EffectDefinition(
            kind="highlights_shadows",
            label="ハイライトとシャドウ",
            category="色",
            keeps_content=True,
            parameters=(
                TrackSpec("highlights", "ハイライト", -100, 100, 0, unit="%"),
                TrackSpec("shadows", "シャドウ", -100, 100, 0, unit="%"),
            ),
            fragment_shader=_HIGHLIGHTS_SHADOWS,
        ),
        EffectDefinition(
            kind="color_shift",
            label="色ずれ",
            category="装飾",
            parameters=(
                TrackSpec("shift", "ずれ", 0, 400, 10, step=1, unit="px"),
                TrackSpec("angle", "角度", -360, 360, 0, unit="度"),
                TrackSpec("strength", "強さ", 0, 100, 100, unit="%"),
                SelectSpec(
                    "order",
                    "並び",
                    tuple((name, name) for name in ("RGB", "RBG", "GRB", "GBR", "BRG", "BGR")),
                    "RGB",
                ),
            ),
            fragment_shader=_COLOR_SHIFT,
        ),
        EffectDefinition(
            kind="radial_blur",
            label="放射ぼかし",
            category="ぼかし",
            parameters=(
                TrackSpec("amount", "強さ", 0, 100, 20, unit="%"),
                TrackSpec("center_x", "中心 X", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("center_y", "中心 Y", -4000, 4000, 0, step=1, unit="px"),
                CheckSpec("hard", "輪郭を保つ", False),
            ),
            fragment_shader=_RADIAL_BLUR,
        ),
        EffectDefinition(
            kind="circular_blur",
            label="回転ぼかし",
            category="ぼかし",
            parameters=(
                TrackSpec("angle", "角度", 0, 360, 10, unit="度"),
                TrackSpec("center_x", "中心 X", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("center_y", "中心 Y", -4000, 4000, 0, step=1, unit="px"),
                CheckSpec("hard", "輪郭を保つ", False),
            ),
            fragment_shader=_CIRCULAR_BLUR,
        ),
        EffectDefinition(
            kind="invert",
            label="色の反転",
            category="色",
            keeps_content=True,
            fragment_shader=_INVERT,
        ),
        EffectDefinition(
            kind="tint",
            label="色付け",
            category="色",
            keeps_content=True,
            parameters=(ColorSpec("color", "色", (1.0, 0.9, 0.7, 1.0)),),
            fragment_shader=_TINT,
        ),
        EffectDefinition(
            kind="inner_shadow",
            label="内側の影",
            category="装飾",
            parameters=(
                TrackSpec("offset_x", "X", -2000, 2000, 6, step=1, unit="px"),
                TrackSpec("offset_y", "Y", -2000, 2000, -6, step=1, unit="px"),
                TrackSpec("blur", "ぼかし", 0, 96, 0, unit="px"),
                TrackSpec("opacity", "濃さ", 0, 100, 100, unit="%"),
                ColorSpec("color", "色", (0.0, 0.0, 0.0, 1.0)),
                SelectSpec("blend", "合成", BLEND_MODES, "normal"),
            ),
            fragment_shader=_INNER_SHADOW,
            passes=2,
        ),
        EffectDefinition(
            kind="inner_halftone",
            label="網点の内側の影",
            category="装飾",
            parameters=(
                TrackSpec("offset_x", "X", -2000, 2000, 6, step=1, unit="px"),
                TrackSpec("offset_y", "Y", -2000, 2000, -6, step=1, unit="px"),
                TrackSpec("blur", "ぼかし", 0, 96, 0, unit="px"),
                TrackSpec("opacity", "濃さ", 0, 100, 100, unit="%"),
                ColorSpec("color", "色", (0.0, 0.0, 0.0, 1.0)),
                SelectSpec("blend", "合成", BLEND_MODES, "normal"),
                SelectSpec("grid", "並び", (("rhombus", "菱形"), ("square", "正方形")), "rhombus"),
                TrackSpec("spacing", "間隔", 1, 200, 10, step=1, unit="px"),
                TrackSpec("dot_size", "点の大きさ", 0, 200, 100, unit="%"),
                TrackSpec("strength", "強さ", 0, 100, 100, unit="%"),
            ),
            fragment_shader=_INNER_HALFTONE,
            passes=2,
        ),
        EffectDefinition(
            kind="inner_outline",
            label="内側の縁取り",
            category="装飾",
            parameters=(
                TrackSpec("thickness", "太さ", 0, 256, 4, step=1, unit="px"),
                TrackSpec("blur", "ぼかし", 0, 96, 0, unit="px"),
                TrackSpec("opacity", "濃さ", 0, 100, 100, unit="%"),
                ColorSpec("color", "色", (1.0, 1.0, 1.0, 1.0)),
                SelectSpec("blend", "合成", BLEND_MODES, "normal"),
                CheckSpec("outline_only", "縁だけ残す", False),
                CheckSpec("angular", "角ばらせる", False),
            ),
            fragment_shader=_INNER_OUTLINE,
            passes=3,
        ),
        EffectDefinition(
            kind="shape_mask",
            label="図形で切り抜く",
            category="形",
            parameters=(
                SelectSpec(
                    "shape",
                    "図形",
                    (
                        ("background", "全体"),
                        ("ellipse", "楕円"),
                        ("rect", "四角"),
                        ("fan", "扇"),
                        ("triangle", "三角"),
                    ),
                    "ellipse",
                ),
                TrackSpec("width", "幅", 0, 20000, 400, step=1, unit="px"),
                TrackSpec("height", "高さ", 0, 20000, 400, step=1, unit="px"),
                TrackSpec("corner", "角の丸み", 0, 10000, 0, step=1, unit="px"),
                TrackSpec("span", "扇の角度", 0, 360, 360, unit="度"),
                TrackSpec("center_x", "X", -20000, 20000, 0, step=1, unit="px"),
                # Y は上が正（シェーダが符号を合わせる） YMM4 と AviUtl の下が正の値は
                # 読み込みが裏返して入れる 表示に「下が正」と書くと、入れた値が上下逆に出る
                TrackSpec("center_y", "Y", -20000, 20000, 0, step=1, unit="px"),
                TrackSpec("rotation", "回転", -3600, 3600, 0, unit="度"),
                TrackSpec("blur", "ぼかし", 0, 1000, 0, unit="px"),
                CheckSpec("invert", "反転", False),
            ),
            fragment_shader=_SHAPE_MASK,
        ),
        EffectDefinition(
            kind="copy_reverse",
            label="反転コピー",
            category="形",
            parameters=(
                SelectSpec(
                    "position",
                    "並べる向き",
                    (("right", "右"), ("left", "左"), ("bottom", "下"), ("top", "上")),
                    "right",
                ),
                TrackSpec("distance", "間隔", -4000, 4000, 0, step=1, unit="px"),
                CheckSpec("flip_horizontal", "左右を反転", True),
                CheckSpec("flip_vertical", "上下を反転", False),
                CheckSpec("centering", "中央に寄せる", True),
            ),
            fragment_shader=_COPY_REVERSE,
        ),
        EffectDefinition(
            kind="fill_background",
            label="背景を塗る",
            category="装飾",
            parameters=(
                ColorSpec("color", "色", (1.0, 1.0, 1.0, 1.0)),
                TrackSpec("opacity", "濃さ", 0, 100, 100, unit="%"),
                TrackSpec("corner", "角の丸み", 0, 2000, 0, step=1, unit="px"),
                TrackSpec("margin_top", "上の余白", -4000, 4000, 10, step=1, unit="px"),
                TrackSpec("margin_bottom", "下の余白", -4000, 4000, 10, step=1, unit="px"),
                TrackSpec("margin_left", "左の余白", -4000, 4000, 10, step=1, unit="px"),
                TrackSpec("margin_right", "右の余白", -4000, 4000, 10, step=1, unit="px"),
                CheckSpec("background_only", "背景だけ", False),
            ),
            fragment_shader=_FILL_BACKGROUND,
        ),
        EffectDefinition(
            kind="binarize",
            label="2 値化",
            category="色",
            keeps_content=True,
            parameters=(
                TrackSpec("threshold", "しきい値", 0, 100, 50, unit="%"),
                CheckSpec("invert", "反転", False),
                CheckSpec("keep_color", "元の色を残す", False),
            ),
            fragment_shader=_BINARIZE,
        ),
        EffectDefinition(
            kind="color_key",
            label="色で抜く",
            category="合成",
            parameters=(
                ColorSpec("key_color", "抜く色", (0.0, 0.0, 0.0, 1.0)),
                TrackSpec("tolerance", "許容範囲", 0, 100, 10, unit="%"),
                CheckSpec("feather", "境界をぼかす", True),
                CheckSpec("invert", "反転", False),
            ),
            fragment_shader=_COLOR_KEY,
        ),
        EffectDefinition(
            kind="linear_transfer",
            label="色の直線変換",
            category="色",
            # 中身の範囲を保つ印は付けない 不透明度の切片が正なら透明な所にも α を置くので、
            # 付けると後ろの粒や欠片が、新しく見えた所を探さずに切れる
            parameters=(
                TrackSpec("red_slope", "赤の傾き", -1000, 1000, 100, unit="%"),
                TrackSpec("red_intercept", "赤の切片", -100, 100, 0, unit="%"),
                TrackSpec("green_slope", "緑の傾き", -1000, 1000, 100, unit="%"),
                TrackSpec("green_intercept", "緑の切片", -100, 100, 0, unit="%"),
                TrackSpec("blue_slope", "青の傾き", -1000, 1000, 100, unit="%"),
                TrackSpec("blue_intercept", "青の切片", -100, 100, 0, unit="%"),
                TrackSpec("alpha_slope", "不透明度の傾き", -1000, 1000, 100, unit="%"),
                TrackSpec("alpha_intercept", "不透明度の切片", -100, 100, 0, unit="%"),
            ),
            fragment_shader=_LINEAR_TRANSFER,
        ),
        EffectDefinition(
            kind="border_blur",
            label="縁のぼかし",
            category="ぼかし",
            keeps_content=True,
            parameters=(TrackSpec("blur", "ぼかし", 0, 96, 10, unit="px"),),
            fragment_shader=_BORDER_BLUR,
            passes=2,
        ),
        EffectDefinition(
            kind="edge_detect",
            label="輪郭抽出",
            category="装飾",
            parameters=(
                TrackSpec("strength", "強さ", 0, 400, 50, unit="%"),
                TrackSpec("radius", "太さ", 1, 16, 1, step=1, unit="px"),
                SelectSpec(
                    "mode", "方法", (("sobel", "ソーベル"), ("prewitt", "プレウィット")), "sobel"
                ),
                CheckSpec("overlay", "元の絵に重ねる", False),
            ),
            fragment_shader=_EDGE_DETECT,
        ),
        # 絵の明るさを高さと見て、光を当てた凹凸を灰色の面で出す（Photoshop・Premiere の
        # エンボス） 縁の反射（bevel_light）は不透明度の境目を高さと見るので、写真や模様の
        # 中の凹凸は出ない
        EffectDefinition(
            kind="emboss",
            label="エンボス",
            category="装飾",
            # α は触らず、透明な所に色を置かない 印が無いと後ろの粒を探す範囲が広がる
            keeps_content=True,
            parameters=(
                # 既定の 135 度（左上から光）は Photoshop の既定と同じ 浮き彫りの見本で
                # いちばん見慣れた向き
                TrackSpec("angle", "光の方向", -360, 360, 135, unit="度"),
                # 1 段の白黒の差を 100% で白（黒）まで振り切る Photoshop の「量」と同じ割合で、
                # 500% まであれば淡い写真の凹凸も強められる
                TrackSpec("height", "高さ", 0, 500, 100, unit="%"),
                TrackSpec("amount", "量", 0, 100, 100, unit="%"),
                SelectSpec(
                    "basis",
                    "高さの基準",
                    (("luma", "明るさ"), ("rgb", "各色"), ("alpha", "不透明度")),
                    "luma",
                ),
                CheckSpec("keep_color", "元の色を残す", False),
                # 画面の画素で数えるので、画質を落としたプレビューでは縮めて渡す（単位が px）
                # 既定の 3 は Photoshop の「高さ」の既定 1 だと線画の縁しか拾えない
                TrackSpec("reach", "取り込み幅", 0.5, 64, 3, unit="px"),
            ),
            fragment_shader=_EMBOSS,
        ),
    )
    for definition in definitions:
        registry.register(definition)
