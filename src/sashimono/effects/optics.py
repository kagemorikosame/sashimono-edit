"""光と歪みと繰り返しのエフェクト YMM4 の残りの映像エフェクトを受ける

どれも YMM4 本体に描かせた試験（``tools/ymm4_probes.py`` の 5 回目）の絵から値の意味を
読み取った 読み取れなかった所はコメントに書いてある

座標はほかのエフェクトと同じく画素で Y は上が正 YMM4 の向き（Y 下向き、角度は画面で
時計回り）は写す側（:mod:`sashimono.compat.ymm4.effects`）で直す
"""

from __future__ import annotations

from sashimono.effects.blending import BLEND_FUNCTIONS, BLEND_MODES
from sashimono.effects.builtin import PRELUDE
from sashimono.effects.definition import EffectDefinition, registry
from sashimono.effects.edges import flipped_edges
from sashimono.effects.motion import _IN_OUT, _JUMP_SHAPE, _WAVE, _easing, _in_out_specs
from sashimono.effects.spec import CheckSpec, ColorSpec, SelectSpec, TrackSpec, ValueSpec

__all__ = ["register_optics_effects"]

_SRGB = """
vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
mat2 turn(float degrees) {
    float r = radians(degrees);
    return mat2(cos(r), sin(r), -sin(r), cos(r));
}
"""


def _shader(body: str) -> str:
    return PRELUDE + _SRGB + body


_BEVEL_LIGHT = _shader(
    BLEND_FUNCTIONS
    + """
uniform int lighting;
uniform float azimuth;
uniform float elevation;
uniform float constant;
uniform float exponent;
uniform vec4 color;
uniform int blend;
uniform float surface_scale;
uniform int profile;
uniform float thickness;
uniform float blur;
uniform bool inverted;

float profile_height(float t) {
    t = clamp(t, 0.0, 1.0);
    float h = t;
    if (profile == 1) h = sqrt(max(1.0 - (1.0 - t) * (1.0 - t), 0.0));
    if (profile == 2) h = 1.0 - sqrt(max(1.0 - t * t, 0.0));
    if (profile == 3) h = step(0.5, t);
    return inverted ? -h : h;
}

float height_at(vec2 uv) {
    return profile_height(texture(u_texture, uv).a);
}

float shade(vec3 normal, vec3 light, float k) {
    if (lighting == 0) {
        vec3 halfway = normalize(light + vec3(0.0, 0.0, 1.0));
        return k * pow(max(dot(normal, halfway), 0.0), max(exponent, 0.01));
    }
    return k * max(dot(normal, light), 0.0);
}

// YMM4 は光の色を sRGB の値へ足す（加算） 平らな面にも光が当たって明るくなる
vec3 lit_color(vec3 under, float amount) {
    if (blend == 0) return mix(under, color.rgb, clamp(amount * color.a, 0.0, 1.0));
    return blend_colors(blend, under, clamp(color.rgb * amount * color.a, 0.0, 1.0));
}

// 画質を落とした合成で、書き出しの画素を 1 つずつ思い描いて光を当て、色で平均するか
// 書き出しの帯は、縁から画面の 1 画素ずつ刻んだ段になる 合成の 1 画素には、急な段・
// 緩い段・平らな面が太さの端数や縁の向きに応じて入り交じる 合成の画素で刻み直すと、
// 端数（6 の縁を 1/4 で 1.5 画素など）で段の数と傾きが書き出しと合わず、縮めた書き出しと
// 光り方がずれる（#182） 光は傾きに比例しないので、傾きを平均してから光を当てても
// 合わない 覆う書き出しの画素ごとに段の高さを求め、光を当てて色で平均する
// 書き出しの画素で 1 に満たない太さ（キーフレームの 0 や負の値も）は、書き出しが 1 へ
// 切り上げて別の式で描くので思い描けない
// 画質の分母が 8 を超えると、覆う書き出しの画素を 8 × 8 までしか数えられず平均が偏る
bool fine_rings() {
    return u_pixel_scale < 0.999 && u_pixel_scale >= 0.125 && thickness >= u_pixel_scale;
}

// 1 辺で数える書き出しの画素の数 画質の分母と同じ
int fine_count() {
    return int(clamp(floor(1.0 / u_pixel_scale + 0.5), 1.0, 8.0));
}

// 書き出しの太さ（書き出しの画素） 書き出しの pass 0 と同じ範囲へ収める
float fine_width() {
    return clamp(thickness / u_pixel_scale, 1.0, 128.0);
}

// 合成の画素 1 つが形に掛かる割合 絵の外は端の画素を読む（テクスチャの読み方と同じ）
float coverage_at(int x, int y) {
    ivec2 last = ivec2(u_size) - 1;
    return texelFetch(u_texture, clamp(ivec2(x, y), ivec2(0), last), 0).a;
}

// 縁が 1 行（swap なら 1 列）の真ん中の高さを通る位置（合成の画素）
// 直線の縁なら、高さ 1 の帯に入る形の面積は、帯の真ん中の線の上で形に入る長さに等しい
// 帯の 7 画素の割合を足せば、アンチエイリアスの階調から縁の位置が端数まで分かる
float strip_edge(int line, int start, bool swap, bool inward_positive) {
    float sum = 0.0;
    for (int i = 0; i < 7; ++i) {
        sum += swap ? coverage_at(line, start + i) : coverage_at(start + i, line);
    }
    return inward_positive ? float(start + 7) - sum : float(start) + sum;
}

// crossing（合成の画素の座標）の近くの縁を直線で近づける 近づけられなければ偽
// アルファ 0.5 の境を補って読むと、斜めや曲がった縁では割合が距離に比例しないので、
// 境が本当の縁から 1 合成画素の 1 割ほど（書き出しの画素で 0.2〜0.5）ずれる 書き出しは
// 自分の細かい画素で境を読むので、ずれは書き出しの 1 画素の段 1 つ分に近い
// 縁の 3 行ぶんの面積から直線を求めると、ずれが残らない
bool edge_line(vec2 crossing, out vec2 point, out vec2 inward) {
    vec2 uv = crossing / u_size;
    vec2 step_ = 1.0 / u_size;
    vec2 across = vec2(step_.x, 0.0);
    vec2 down = vec2(0.0, step_.y);
    vec2 slope = vec2(
        texture(u_texture, uv + across).a - texture(u_texture, uv - across).a,
        texture(u_texture, uv + down).a - texture(u_texture, uv - down).a
    );
    point = crossing;
    inward = vec2(0.0);
    if (length(slope) < 1e-3) return false;
    // 縁が横に近ければ列ごとに足す 行ごとに足すと、縁が帯の外へ抜ける
    bool swap = abs(slope.y) > abs(slope.x);
    vec2 p = swap ? crossing.yx : crossing;
    bool inward_positive = (swap ? slope.y : slope.x) > 0.0;
    int start = int(floor(p.x)) - 3;
    int line = int(floor(p.y));
    float before = strip_edge(line - 1, start, swap, inward_positive);
    float middle = strip_edge(line, start, swap, inward_positive);
    float after = strip_edge(line + 1, start, swap, inward_positive);
    // 3 行が 1 本の線に乗らない所（角や細い所）は直線で近づけられない
    if (abs(after - 2.0 * middle + before) > 0.25) return false;
    vec2 tangent = normalize(vec2(after - before, 2.0));
    vec2 normal = vec2(tangent.y, -tangent.x);
    if ((normal.x > 0.0) != inward_positive) normal = -normal;
    vec2 at = vec2(middle, float(line) + 0.5);
    point = swap ? at.yx : at;
    inward = swap ? normal.yx : normal;
    return true;
}

// 縁の内向きが n の直線の縁で、24 方向のうち縁へいちばん真っ直ぐ向く方向の余弦
// 書き出しはその方向で縁を見つけるので、縁までの距離をこれで割った値が測りになる
float facing(vec2 n) {
    float step_ = PI / 12.0;
    return cos(abs(mod(atan(n.y, n.x) + step_ * 0.5, step_) - step_ * 0.5));
}

// 縁の向きを持たない（縁が測る範囲に無かった）印 角度は -π..π なので重ならない
const float NO_EDGE = 10.0;

// 書き出しの pass 0 は、画素から 24 方向へ輪を広げ、形の外（アルファ 0.5 未満）に掛かった
// 最初の輪で縁までを測る 斜めや曲がった縁では、いちばん近い方向が縁へ真っ直ぐ向かない
// ので、縁までの本当の距離より少し遠く出る 合成の画素で輪を刻むと、この測り方の癖と
// 1 合成画素（書き出しの 2〜4 画素）より細かい縁の位置が消え、楕円の段が書き出しとずれる
// （#194） 同じ 24 方向で縁を越える所を刻まずに探し、書き出しの画素で数えて置く
// 返すのは測りと、縁の内向きの角度 間の書き出しの画素は、周りの合成の画素の測りを縁の
// 向きへ伸ばして求める（measure_at） 伸ばせるように、形の外は形までの距離を負で
// 3 合成画素まで置き、太さの先も 2 合成画素ぶんまで測る
vec2 edge_measure() {
    bool inside = texture(u_texture, v_uv).a >= 0.5;
    float reach = inside ? fine_width() * u_pixel_scale + 2.0 : 3.0;
    int rings = int(ceil(reach));
    vec2 centre = v_uv * u_size;
    for (int ring = 1; ring <= rings; ++ring) {
        float best = -1.0;
        vec2 best_toward = vec2(0.0);
        for (int k = 0; k < 24; ++k) {
            float a = PI * 2.0 * float(k) / 24.0;
            vec2 toward = vec2(cos(a), sin(a));
            if ((texture(u_texture, v_uv + toward * float(ring) / u_size).a >= 0.5) == inside) {
                continue;
            }
            // 越えた輪と 1 つ内の輪の間を半分ずつ詰める 10 回で 1/1024 合成画素
            float near_ = float(ring - 1);
            float far_ = float(ring);
            for (int i = 0; i < 10; ++i) {
                float middle = (near_ + far_) * 0.5;
                if ((texture(u_texture, v_uv + toward * middle / u_size).a >= 0.5) == inside) {
                    near_ = middle;
                } else {
                    far_ = middle;
                }
            }
            float crossing = (near_ + far_) * 0.5;
            if (best < 0.0 || crossing < best) {
                best = crossing;
                best_toward = toward;
            }
        }
        // 先に越えた輪の方向より、後の輪で越える方向が近いことは無い
        if (best < 0.0) continue;
        vec2 point;
        vec2 inward;
        vec2 direction = inside ? -best_toward : best_toward;
        bool straight = edge_line(centre + best_toward * best, point, inward);
        // 縁とほぼ平行に見つけた方向は、直線との交わりが遠くへ飛ぶので決め直さない
        if (straight && abs(dot(best_toward, inward)) > 0.5) {
            // 見つけた方向のまま、越える所だけを縁の直線で決め直す 縁までの距離を 24 方向の
            // 癖で割る形にすると、曲がりのきつい縁（楕円の長い軸の端）から離れた所で外れる
            float refined = dot(point - centre, inward) / dot(best_toward, inward);
            // 補った境から大きく離れるなら、直線で近づけたのが外れている
            if (abs(refined - best) < 0.5) {
                best = refined;
                direction = inward;
            }
        }
        return vec2((inside ? best : -best) / u_pixel_scale, atan(direction.y, direction.x));
    }
    return vec2((inside ? reach : -reach) / u_pixel_scale, NO_EDGE);
}

// 書き出しの画素（uv）の縁までの測り
// 周りの合成の画素 4 つの測りを、それぞれの縁の向きへ伸ばして小さい方を取る 直線の縁なら
// どれから伸ばしても同じ値になる 補う（線形に混ぜる）と、上下の縁から等しく離れた
// 真ん中の尾根で測りが低く出る 尾根は太い縁の反射でよく掛かり、段が 1 つずれる
float measure_at(vec2 uv) {
    vec2 p = uv * u_size;
    ivec2 corner = ivec2(floor(p - 0.5));
    ivec2 last = ivec2(u_size) - 1;
    float best = 1e9;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            ivec2 texel = clamp(corner + ivec2(i, j), ivec2(0), last);
            vec4 field = texelFetch(u_texture, texel, 0);
            float extended = field.r;
            if (field.b < NO_EDGE - 1.0) {
                vec2 n = vec2(cos(field.b), sin(field.b));
                extended += dot(p - (vec2(texel) + 0.5), n) / (facing(n) * u_pixel_scale);
            }
            best = min(best, extended);
        }
    }
    return best;
}

// 書き出しの pass 0 が、縁から measure 画素（24 方向で測った値）の所へ置く値
// 形の外は 0、縁の外に掛かった最初の輪の半径を太さで割る 太さの先は 1
// 輪は太さ（上限 128）を 48 までの数で刻む
float export_value(float measure, float width) {
    if (measure <= 0.0) return 0.0;
    float rings = ceil(min(width, 48.0));
    // 輪の半径ちょうどに縁がある所は、書き出しではその輪が形の外に掛からない（アルファが
    // ちょうど 0.5） 太さが 48 を超えて輪の間が 1 画素でなくなると、四角の縁から画素の
    // 中心までの距離が輪の半径と揃う所が出る 測りの丸めで下へ外れないよう少し足す
    return min((floor(measure * rings / width + 0.002) + 1.0) / rings, 1.0);
}

// 書き出しがぼかすか 書き出しのぼかしは 1 画素に満たない半径では何もしない
bool fine_blurred() {
    return blur / u_pixel_scale >= 1.0;
}

float normal_cdf(float z) {
    // erf の近似（Abramowitz と Stegun 7.1.26 誤差 1.5e-7） GLSL に erf が無い
    float x = abs(z) * 0.70710678;
    float t = 1.0 / (1.0 + 0.3275911 * x);
    float poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t;
    float erf_ = 1.0 - poly * exp(-x * x);
    return 0.5 * (1.0 + sign(z) * erf_);
}

// 書き出しのぼかし（1 次元）で、縁の内へ x 画素の所の段 1 つがどれだけ上がって見えるか
// 書き出しのぼかしは半径の半分を標準偏差とし、半径で打ち切った重み
float blurred_step(float x) {
    float taps = floor(min(blur / u_pixel_scale, 96.0));
    float reach = taps + 0.5;
    if (x >= reach) return 1.0;
    if (x <= -reach) return 0.0;
    float sigma = max(blur / u_pixel_scale * 0.5, 0.5);
    float low = normal_cdf(-reach / sigma);
    float high = normal_cdf(reach / sigma);
    return clamp((normal_cdf(x / sigma) - low) / (high - low), 0.0, 1.0);
}

// 縁を直線と見て、書き出しの段を縁に垂直な向きにだけぼかした値
// 段は縁から 1 輪ごとに 1/輪の数ずつ上がる ぼかした段を足し合わせる
// 24 方向の測りと縁までの本当の距離の違い（1 % 未満）は見ない
float blurred_value(float measure, float width) {
    float rings = ceil(min(width, 48.0));
    float spacing = width / rings;
    float sum = 0.0;
    for (int k = 0; k < int(rings); ++k) {
        sum += blurred_step(measure - float(k) * spacing);
    }
    return sum / rings;
}

// ぼかしのある時の縁の坂の広さ（太さとぼかしの広がり 合成の画素）
float blurred_slope() {
    float sigma = max(blur / u_pixel_scale * 0.5, 0.5);
    return (fine_width() + 2.5 * sigma) * u_pixel_scale;
}

// 坂がこれより広いと、書き出しの画素を思い描かずに前の描き方で描く（draw_mode）
const float LEGACY_SLOPE = 9.0;

// ぼかしのある時に、合成の画素でぼかした値を補うだけで足りる割合
// 坂が合成の画素 3 つより狭いと、補うと坂が均されて緩む所が内へずれる 広いほど補うだけの
// 方が合う 縁を直線と見る値は、上下の縁や角をまたぐぼかしで折れ目が出て、補っても
// 消えない 坂の広さでなめらかに混ぜて、ぼかしを動かした時に光り方が跳ねないようにする
float coarse_share() {
    return clamp((blurred_slope() - 3.0) / (LEGACY_SLOPE - 3.0), 0.0, 1.0);
}

// 描き方 書き出しと同じ式・前の思い描き方（#182）・今の思い描き方（#194）
const int DRAW_PLAIN = 0;
const int DRAW_LEGACY = 1;
const int DRAW_FINE = 2;

// 今の思い描き方が前の描き方より縮めた書き出しから離れる所は、前の描き方のまま描く
// 四角と楕円・太さ 1〜128・ぼかし 0〜96 の 592 通りで前と今を比べて、どれも前より
// 離れないように決めた（#194）
// ぼかしの無い 48 を超える太さは、輪の間が 1 画素でなくなり、輪の半径と画素の中心が
// 揃う所で書き出しの丸め方を当てきれない ぼかしの坂が広い所は、前の合成の画素で刻んで
// ぼかす描き方の方が四角の角や尾根で合う 坂の広さの境で描き方が切り替わるが、境の
// 両側はどちらも縮めた書き出しに近いので、跳ねは小さい
int draw_mode() {
    if (!fine_rings()) return DRAW_PLAIN;
    bool legacy = !fine_blurred() ? fine_width() > 48.0 : blurred_slope() >= LEGACY_SLOPE;
    if (!legacy) return DRAW_FINE;
    // 前の思い描き方は、ぼかしの無い 46 合成画素までの太さだけだった
    return blur <= 0.0 && thickness <= 46.0 ? DRAW_LEGACY : DRAW_PLAIN;
}

// 書き出しの画素 1 つの高さ
// ぼかしが無ければ、縁までの測りを求めてから書き出しと同じく段に刻む 刻んだ段を補うと、
// 段の境が合成の画素の升目に揃い、急な段が書き出しの 2〜4 倍の幅へ緩む
// ぼかしがあれば、縁を直線と見てぼかした段に、合成の画素でぼかした値との違いを足す
// 直線と見た値は角で外れるが、外れは合成の画素でぼかした値と比べると分かり、
// なだらかなので補っても崩れない（pass 1 の緑とアルファ）
float fine_height(vec2 uv) {
    vec4 field = texture(u_texture, uv);
    if (fine_blurred()) {
        float share = coarse_share();
        float straight = blurred_value(measure_at(uv), fine_width()) + field.a - field.g;
        return profile_height(mix(straight, field.a, share));
    }
    return profile_height(export_value(measure_at(uv), fine_width()));
}

// 合成の 1 画素が覆う書き出しの画素に光を当てて、色で平均する
// 書き出しでは形の縁の画素も、周りの段の傾きで光ってから覆う割合だけ混ざる 書き出しの
// 画素が形に掛かる割合（縁からの距離 + 0.5）で重みを付ける
// 高さは覆う書き出しの画素と、その上下左右の 1 画素ずつで求める 隣どうしで同じ画素を
// 何度も読まないよう、先に升目へ溜める（1/8 で 10 × 10）
vec3 fine_lit(vec3 under, vec3 light, float k) {
    int count = fine_count();
    float strength = surface_scale / u_pixel_scale;
    float heights[10][10];
    for (int y = 0; y < count + 2; ++y) {
        for (int x = 0; x < count + 2; ++x) {
            // 四隅は傾きに使わない
            if ((x == 0 || x == count + 1) && (y == 0 || y == count + 1)) continue;
            vec2 offset = (vec2(float(x - 1), float(y - 1)) + 0.5) / float(count) - 0.5;
            heights[y][x] = fine_height(v_uv + offset / u_size);
        }
    }
    vec3 sum = vec3(0.0);
    float taken = 0.0;
    for (int y = 1; y <= count; ++y) {
        for (int x = 1; x <= count; ++x) {
            vec2 offset = (vec2(float(x - 1), float(y - 1)) + 0.5) / float(count) - 0.5;
            float weight = clamp(measure_at(v_uv + offset / u_size) + 0.5, 0.0, 1.0);
            if (weight <= 0.0) continue;
            float gx = (heights[y][x + 1] - heights[y][x - 1]) * 0.5;
            float gy = (heights[y + 1][x] - heights[y - 1][x]) * 0.5;
            vec3 normal = normalize(vec3(-gx * strength, -gy * strength, 1.0));
            sum += lit_color(under, shade(normal, light, k)) * weight;
            taken += weight;
        }
    }
    if (taken <= 0.0) return lit_color(under, shade(vec3(0.0, 0.0, 1.0), light, k));
    return sum / taken;
}

// 前の思い描き方（#182）で、縁から distance 画素の所に書き出しが置く高さ
float legacy_height(float distance, float width) {
    if (distance <= 0.0) return profile_height(0.0);
    float rings = ceil(min(width, 48.0));
    float ring = floor(distance * rings / width) + 1.0;
    return profile_height(min(ring / rings, 1.0));
}

// 前の思い描き方（#182） 縁からの距離と向きを合成の画素で刻んだ輪から取り、縁を直線と
// 見て覆う書き出しの画素に光を当てる 四角の縁ならどの画素も書き出しと同じ段に入る
vec3 legacy_lit(vec3 under, vec3 light, float k, float width) {
    vec2 step_ = 1.0 / u_size;
    float here = texture(u_texture, v_uv).a * width - 0.5;
    vec2 slope = vec2(
        texture(u_texture, v_uv + vec2(step_.x, 0.0)).a
            - texture(u_texture, v_uv - vec2(step_.x, 0.0)).a,
        texture(u_texture, v_uv + vec2(0.0, step_.y)).a
            - texture(u_texture, v_uv - vec2(0.0, step_.y)).a
    );
    vec2 inward = length(slope) > 1e-6 ? normalize(slope) : vec2(0.0);
    float scale = 1.0 / u_pixel_scale;
    int count = fine_count();
    float full = min(thickness * scale, 128.0);
    float strength = surface_scale * scale;
    vec3 sum = vec3(0.0);
    float taken = 0.0;
    for (int y = 0; y < count; ++y) {
        for (int x = 0; x < count; ++x) {
            vec2 offset = (vec2(float(x), float(y)) + 0.5) / float(count) - 0.5;
            float distance = (here + dot(offset, inward)) * scale;
            // 縁の外に掛かる書き出しの画素は、書き出しでも光らない（形の外）
            if (distance <= 0.0) continue;
            float g = (legacy_height(distance + 1.0, full) - legacy_height(distance - 1.0, full))
                * 0.5;
            vec3 normal = normalize(vec3(-inward * g * strength, 1.0));
            sum += lit_color(under, shade(normal, light, k));
            taken += 1.0;
        }
    }
    if (taken <= 0.0) return lit_color(under, shade(vec3(0.0, 0.0, 1.0), light, k));
    return sum / taken;
}

void main() {
    int mode = draw_mode();
    bool fine = mode == DRAW_FINE;
    // 前の思い描き方は、縁から帯の外の平らな所まで（太さ + 2 画素）を 1 画素ずつ刻む
    float legacy_width = ceil(thickness) + 2.0;
    if (u_pass == 0) {
        // 書き出しの画素を思い描く時は、縁までの測りを赤へ、縁の内向きの角度を青へ置く
        if (fine) {
            vec2 measured = edge_measure();
            frag_color = vec4(measured.x, 0.0, measured.y, 1.0);
            return;
        }
        // 縁からの距離を太さで割った値（0 が縁、1 が太さ以上の内側）を作る
        float inside = texture(u_texture, v_uv).a;
        if (inside < 0.5) { frag_color = vec4(1.0, 1.0, 1.0, 0.0); return; }
        float width = mode == DRAW_LEGACY ? legacy_width : clamp(thickness, 1.0, 128.0);
        float nearest = width;
        int rings = int(ceil(min(width, 48.0)));
        for (int ring = 1; ring <= rings; ++ring) {
            float r = width * float(ring) / float(rings);
            if (r >= nearest) break;
            for (int k = 0; k < 24; ++k) {
                float a = PI * 2.0 * float(k) / 24.0;
                if (texture(u_texture, v_uv + vec2(cos(a), sin(a)) * r / u_size).a < 0.5) {
                    nearest = min(nearest, r);
                }
            }
        }
        frag_color = vec4(1.0, 1.0, 1.0, nearest / width);
        return;
    }
    vec4 here = texelFetch(u_texture, ivec2(gl_FragCoord.xy), 0);
    bool blurred = fine && fine_blurred();
    if (u_pass == 1) {
        // ぼかす値は、書き出しと同じく覆う書き出しの画素で段に刻んでから平均する（縮める）
        // 縁を直線と見てぼかした値も同じ所で平均して緑へ置く 赤と青（測りと縁の向き）は残す
        if (blurred) {
            int count = fine_count();
            float width = fine_width();
            float steps = 0.0;
            float straight = 0.0;
            for (int y = 0; y < count; ++y) {
                for (int x = 0; x < count; ++x) {
                    vec2 offset = (vec2(float(x), float(y)) + 0.5) / float(count) - 0.5;
                    float measure = measure_at(v_uv + offset / u_size);
                    steps += export_value(measure, width);
                    straight += blurred_value(measure, width);
                }
            }
            float taken = float(count * count);
            frag_color = vec4(here.r, straight / taken, here.b, steps / taken);
            return;
        }
        // 書き出しでは何もしない（画質を落とした時の刻み直しのための段） 画素をそのまま
        // 写す 補って読むと、画素の中心でも値が丸めで揺れ、等倍の絵が前と変わりうる
        frag_color = here;
        return;
    }
    if (u_pass == 2 || u_pass == 3) {
        if (fine && !blurred) { frag_color = here; return; }
        vec2 direction = u_pass == 2 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        // blur は合成の画素へ縮めて渡される（TrackSpec の px） ぼかす絵も合成の大きさなので、
        // そのまま渡すと書き出しの半径を縮めた幅でぼかす ここで u_pixel_scale で割ると
        // 2 回縮める逆になり、2 倍・4 倍の幅でぼかしてしまう
        // blurred_step は書き出しの画素で数えるので、同じ blur を u_pixel_scale で割って
        // 半径を揃える 求め方は別（こちらは画素ごとの重みの畳み込み、あちらは段 1 つを
        // 同じ重みでぼかした形を正規分布の累積で近づけた物）
        vec4 spread = blur1d(u_texture, v_uv, direction, blur);
        // 思い描く時にぼかすのは段の値（アルファ）だけ 赤・緑・青は光を当てる所で読む
        frag_color = blurred ? vec4(here.rgb, spread.a) : spread;
        return;
    }
    vec4 base = texture(u_source, v_uv);
    if (base.a <= 0.0001) { frag_color = base; return; }
    // 方位は画面で右が 0、時計回り（YMM4 に描かせた絵で、-85 は上から当たった）
    float a = radians(azimuth);
    float e = radians(elevation);
    vec3 light = normalize(vec3(cos(e) * cos(a), -cos(e) * sin(a), sin(e)));
    float k = max(constant, 0.0) * 0.01;
    vec3 under = to_srgb(base.rgb);
    if (fine) {
        frag_color = vec4(to_linear(fine_lit(under, light, k)), base.a);
        return;
    }
    if (mode == DRAW_LEGACY) {
        frag_color = vec4(to_linear(legacy_lit(under, light, k, legacy_width)), base.a);
        return;
    }
    vec2 step_ = 1.0 / u_size;
    float gx = (height_at(v_uv + vec2(step_.x, 0.0)) - height_at(v_uv - vec2(step_.x, 0.0))) * 0.5;
    float gy = (height_at(v_uv + vec2(0.0, step_.y)) - height_at(v_uv - vec2(0.0, step_.y))) * 0.5;
    // 1 画素に満たない太さ（画質を落とした合成で、書き出しの画素を思い描けない時）は、
    // 1 画素の帯として描いてから、帯の画素に占める縁の割合だけ平らな面の光と混ぜる
    // 書き出しを縮めると、細い縁の急な面と内側の平らな面が 1 画素の中で平均される 1 画素に
    // 切り上げたままだと、面が緩く帯が太い別の光り方になる 傾きは太さで割って、書き出しの
    // 急な面に合わせる
    float share = clamp(thickness, 0.0001, 1.0);
    vec3 normal = normalize(vec3(-gx * surface_scale, -gy * surface_scale, share));
    vec3 lit = lit_color(under, shade(normal, light, k));
    // 混ぜるのは色にしてから 光の量で混ぜると、強く当てて白く飽和する所が縁の割合より濃く出る
    if (share < 1.0) {
        lit = mix(lit_color(under, shade(vec3(0.0, 0.0, 1.0), light, k)), lit, share);
    }
    frag_color = vec4(to_linear(lit), base.a);
}
"""
)

_LENS_BLUR = _shader(
    """
uniform float radius;
uniform float brightness;
uniform float edge_strength;
uniform bool fixed_size;

// 大きさを変えない（AviUtl の サイズ固定）なら、絵の置かれた範囲の外を読まずに端を読み直す
vec4 lens_sample(vec2 uv) {
    if (!fixed_size) return texture(u_texture, uv);
    vec2 pixel = clamp(uv * u_size, u_object.xy + 0.5, u_object.zw - 0.5);
    return texture(u_texture, pixel / u_size);
}

void main() {
    // 丸い絞りのぼかし 縁ほど重くして、玉ボケの輪が明るく中が抜けて見えるようにする
    vec4 centre = texture(u_texture, v_uv);
    if (radius < 0.5) { frag_color = centre; return; }
    // AviUtl2 に白い四角 300 を範囲 30 でぼかさせると、サイズ固定では端が薄れず 300 のまま
    // だった（#188 #210） 範囲の外はそのまま返す
    vec2 here = v_uv * u_size;
    if (fixed_size && (any(lessThan(here, u_object.xy)) || any(greaterThan(here, u_object.zw)))) {
        frag_color = centre;
        return;
    }
    int rings = int(clamp(ceil(radius / 3.0), 2.0, 10.0));
    vec3 sum = centre.rgb * centre.a;
    float alpha = centre.a;
    float total = 1.0;
    for (int ring = 1; ring <= rings; ++ring) {
        float r = radius * float(ring) / float(rings);
        int count = 6 * ring;
        float weight = 1.0 + max(edge_strength, 0.0) * pow(float(ring) / float(rings), 2.0);
        for (int k = 0; k < count; ++k) {
            float a = PI * 2.0 * (float(k) + 0.5 * float(ring % 2)) / float(count);
            vec4 c = lens_sample(v_uv + vec2(cos(a), sin(a)) * r / u_size);
            sum += c.rgb * c.a * weight;
            alpha += c.a * weight;
            total += weight;
        }
    }
    vec3 rgb = sum / total * max(brightness, 0.0) * 0.01;
    float a = alpha / total;
    frag_color = a > 0.0001 ? vec4(rgb / a, a) : vec4(0.0);
}
"""
)

_FISH_EYE = _shader(
    """
uniform int projection;
uniform float angle;
uniform float zoom;

void main() {
    // 絵の対角の半分を 1 として、魚眼の像の高さから元の（普通のレンズの）高さへ戻す
    // 四隅は動かず、辺の中ほどが外へ膨らむ（YMM4 の絵もそうだった）
    vec2 centre = object_center();
    vec2 p = (v_uv * u_size - centre) / max(zoom * 0.01, 0.01);
    float corner = length(object_size()) * 0.5;
    float rho = length(p) / corner;
    if (rho < 1e-5) { frag_color = sample_pixel(centre); return; }
    if (rho > 1.0) { frag_color = vec4(0.0); return; }
    float half_angle = radians(clamp(angle, 1.0, 179.0)) * 0.5;
    float theta;
    if (projection == 1) theta = rho * half_angle;
    else if (projection == 2) theta = 2.0 * atan(rho * tan(half_angle * 0.5));
    else if (projection == 3) theta = 2.0 * asin(clamp(rho * sin(half_angle * 0.5), -1.0, 1.0));
    else theta = asin(clamp(rho * sin(half_angle), -1.0, 1.0));
    float source = tan(min(theta, radians(89.0))) / tan(half_angle) * corner;
    frag_color = sample_pixel(centre + normalize(p) * source);
}
"""
)

_RIPPLE = _shader(
    """
uniform float center_x;
uniform float center_y;
uniform float amplitude;
uniform float wavelength;
uniform float period;

void main() {
    // 中心から広がる波で、半径の向きにずらす 時間で外へ進む
    vec2 centre = object_center() + vec2(center_x, center_y);
    vec2 p = v_uv * u_size - centre;
    float r = length(p);
    if (r < 1e-4) { frag_color = sample_pixel(centre); return; }
    float phase = r / max(wavelength, 1.0) - u_time / max(period, 0.01);
    float shift = amplitude * sin(2.0 * PI * phase);
    // はみ出した所は絵の端の画素を読む YMM4 の絵は波打っても外形が四角のままだった
    vec2 low = object_center() - object_size() * 0.5 + 0.5;
    vec2 high = object_center() + object_size() * 0.5 - 0.5;
    vec2 source = clamp(centre + p / r * (r - shift), low, high);
    vec2 here = v_uv * u_size;
    if (any(lessThan(here, low - 0.5)) || any(greaterThan(here, high + 0.5))) {
        frag_color = vec4(0.0);
        return;
    }
    frag_color = sample_pixel(source);
}
"""
)

_POLAR = _shader(
    """
uniform float core;
uniform float twist;

void main() {
    // 絵を円に巻く 横は真上から反時計回りの角度、縦は中心の穴の縁から外への距離
    // YMM4 に左が赤・右が青の横のグラデーションを巻かせると、真上の継ぎ目の左が赤、右が青に
    // なった（絵の左端が真上で、右へ進むほど反時計回り #198） 時計回りに巻くと左右が裏返る
    // ねじれも同じ向きに裏返す（配布物のねじれはどれも 0 で、向きは測っていない）
    vec2 centre = object_center();
    vec2 size = object_size();
    vec2 p = v_uv * u_size - centre;
    float r = length(p);
    float inner = max(core, 0.0);
    if (r < inner || r > inner + size.y) { frag_color = vec4(0.0); return; }
    float depth = (r - inner) / size.y;
    float a = -(atan(p.x, p.y) + radians(twist) * depth);
    float along = fract(a / (2.0 * PI));
    vec2 corner = centre - size * 0.5;
    frag_color = sample_pixel(corner + vec2(along * size.x, size.y * (1.0 - depth)));
}
"""
)

_STRETCH = _shader(
    """
uniform float center_x;
uniform float center_y;
uniform float angle;
uniform float stretch;
uniform float range;
uniform bool centering;

void main() {
    // 中心を通る線で絵を切り、両側を離す 間は線の近く（幅 range）の画素を引き伸ばして埋める
    // 角度 0 で上下に離れる（YMM4 の絵）
    vec2 normal = turn(-angle) * vec2(0.0, -1.0);
    vec2 centre = object_center() + vec2(center_x, center_y);
    vec2 p = v_uv * u_size - centre;
    float length_ = max(stretch, 0.0);
    // 中央揃えでなければ、片側だけが離れる（真ん中を半分ずらして同じ式で扱う）
    if (!centering) p += normal * length_ * 0.5;
    float s = dot(p, normal);
    float band = max(range, 0.0) * 0.5;
    float reach = band + length_ * 0.5;
    float source_s;
    if (s > reach) source_s = s - length_ * 0.5;
    else if (s < -reach) source_s = s + length_ * 0.5;
    else source_s = reach > 1e-4 ? s / reach * band : 0.0;
    frag_color = sample_pixel(centre + p + normal * (source_s - s));
}
"""
)

_REEL_SPIN = _shader(
    """
uniform float rotation;
uniform float direction;

void main() {
    // 絵の範囲の中で、向きに沿ってずらして折り返す（リールが回るように） 100% で 1 周
    vec2 size = object_size();
    vec2 corner = object_center() - size * 0.5;
    vec2 d = turn(-direction) * vec2(1.0, 0.0);
    vec2 extent = abs(d) * size;
    vec2 p = v_uv * u_size - corner;
    if (p.x < 0.0 || p.y < 0.0 || p.x > size.x || p.y > size.y) { frag_color = vec4(0.0); return; }
    vec2 source = mod(p - d * length(extent) * rotation * 0.01, size);
    frag_color = sample_pixel(corner + source);
}
"""
)

_TILE = _shader(
    """
uniform float count_x;
uniform float count_y;

void main() {
    // 絵を横に count_x 個、縦に count_y 個、隙間なく並べる 真ん中は元の位置
    vec2 counts = max(floor(vec2(count_x, count_y) + 0.5), vec2(1.0));
    vec2 size = object_size();
    vec2 centre = object_center();
    vec2 p = v_uv * u_size - centre;
    vec2 index = floor(p / size + counts * 0.5);
    if (any(lessThan(index, vec2(0.0))) || any(greaterThanEqual(index, counts))) {
        frag_color = vec4(0.0);
        return;
    }
    vec2 offset = (index - (counts - 1.0) * 0.5) * size;
    frag_color = sample_pixel(centre + p - offset);
}
"""
)

_INOUT_WIPE = _shader(
    """
uniform int pattern;
uniform float tolerance;
uniform float angle;
uniform bool reverse_in;
uniform bool reverse_out;
"""
    + _IN_OUT
    + """
// YMM4 に付いている切り替え画像を式にしたもの 絵を覆う正方形に合わせる（縦横比を保つ）
float wipe_value(vec2 p) {
    float half_side = max(object_size().x, object_size().y) * 0.5;
    vec2 u = turn(angle) * (p / half_side);
    u.y = -u.y;
    if (pattern == 0) return 1.0 - (u.x + 1.0) * 0.5;
    if (pattern == 1) return 1.0 - (u.y + 1.0) * 0.5;
    if (pattern == 2) return min(length(u) / sqrt(2.0), 1.0);
    if (pattern == 3) return min((abs(u.x) + abs(u.y)) * 0.5, 1.0);
    return fract(atan(u.x, -u.y) / (2.0 * PI) + 1.0);
}

float reveal(float value, float amount) {
    float soft = max(tolerance * 0.01, 1.0 / 256.0);
    return clamp((amount * (1.0 + soft) - value) / soft, 0.0, 1.0);
}

void main() {
    vec4 color = texture(u_texture, v_uv);
    vec2 p = v_uv * u_size - object_center();
    float span = max(effect_time, 0.0001);
    float keep = 1.0;
    if (effect_in) {
        float amount = ease(u_time / span, easing, easing_mode);
        if (pattern == 5) keep = min(keep, amount);
        else {
            float value = wipe_value(p);
            keep = min(keep, reveal(reverse_in ? 1.0 - value : value, amount));
        }
    }
    if (effect_out) {
        // 退場は登場を時間で裏返す 終わりから数えた残りで登場と同じ曲線を引く
        // （motion の hidden_amount と同じ決まり YMM4 の木製看板テロップで測った #177）
        float amount = ease((u_duration - u_time) / span, easing, easing_mode);
        if (pattern == 5) keep = min(keep, amount);
        else {
            float value = wipe_value(p);
            keep = min(keep, reveal(reverse_out ? 1.0 - value : value, amount));
        }
    }
    frag_color = vec4(color.rgb, color.a * keep);
}
"""
)

_DIRECTIONAL_KEY = _shader(
    """
uniform vec4 background;
uniform vec4 foreground;
uniform float softness;
uniform float threshold;
uniform bool output_foreground;

void main() {
    // 背景の色から前景の色へ向かう線の上で、どこまで前景に寄っているかを不透明度にする
    vec4 base = texture(u_texture, v_uv);
    vec3 c = to_srgb(base.rgb);
    vec3 bg = to_srgb(background.rgb);
    vec3 fg = to_srgb(foreground.rgb);
    vec3 axis = fg - bg;
    float t = dot(c - bg, axis) / max(dot(axis, axis), 1e-5);
    float soft = max(softness * 0.01, 0.01);
    float alpha = smoothstep(threshold, threshold + soft, t);
    vec3 rgb = c;
    if (output_foreground && alpha > 0.001) {
        // 背景の色が混ざった分を取り除く
        rgb = clamp((c - (1.0 - alpha) * bg) / alpha, 0.0, 1.0);
    }
    frag_color = vec4(to_linear(rgb), base.a * alpha);
}
"""
)

_REPEAT_ZOOM = _shader(
    """
uniform float zoom;
uniform float zoom_x;
uniform float zoom_y;
uniform float interval;
uniform bool centering;
uniform int easing;
uniform int easing_mode;
"""
    + _WAVE
    + """
void main() {
    // 等倍と「拡大率 × 縦横の割合」の間を往復する 中央揃えなら等倍を挟んで振れる
    float k = centering ? repeat_swing() : repeat_wave();
    vec2 peak = vec2(zoom_x, zoom_y) * 0.01 * zoom * 0.01;
    vec2 scale = max(1.0 + (peak - 1.0) * k, vec2(0.0001));
    vec2 centre = object_center();
    frag_color = sample_pixel(centre + (v_uv * u_size - centre) / scale);
}
"""
)

_JUMP = _shader(
    """
uniform float height;
uniform float stretch;
uniform float period;
uniform float distortion;
uniform float interval;
"""
    + _JUMP_SHAPE
    + """
void main() {
    // 1 回の跳びは period 秒の半周期の正弦 着いたら interval 秒潰れる（:data:`_JUMP_SHAPE`）
    vec2 lifted;
    vec2 tall;
    vec2 flat_;
    jump_shape(u_time, height, stretch, period, distortion, interval, lifted, tall, flat_);
    vec2 centre = object_center();
    vec2 foot = vec2(centre.x, u_object.y);
    vec2 p = centre + (v_uv * u_size - lifted - centre) / tall;
    frag_color = sample_pixel(foot + (p - foot) / flat_);
}
"""
)

#: 1 画素で調べる粒の数の上限 上限を超えた古い粒は描かない（読み込む所で記録に残す）
#: 重さは居る粒の数に比例し、上限は GPU が止まるほど重い設定を防ぐ栓 1080p の書き出しで
#: 1500 粒（配布物の雪）が 1 枚 35ms、4096 粒で 95ms ほど（#199 で測った）
#: 前の 512 では配布物の雨（800 粒）と雪（1500 粒）の古い粒が黙って消えていた
MAX_PARTICLES = 4096

_PARTICLES = _shader(
    f"const int MAX_PARTICLES = {MAX_PARTICLES};\n"
    + """
uniform float rate;
uniform float lifetime;
uniform float preroll;
uniform float size;
uniform float end_scale;
uniform float emitter_x;
uniform float emitter_y;
uniform float emit_range;
uniform float emit_angle;
uniform float spread;
uniform float speed;
uniform float gravity;
uniform float wind_angle;
uniform float wind_speed;
uniform float turbulence;
uniform float rotation;
uniform float fade;
uniform float randomness;

float rand(float index, float salt) { return hash(vec2(index * 0.1371 + 3.7, salt)); }

void main() {
    // 絵を粒として放つ 放ち口は絵の中心から (emitter_x, emitter_y)（画面の Y 下向き）で、
    // 横に emit_range の幅でばらつく 重力は下、風は角度の向き 絵は粒ごとに中心へ置き直す
    float now = u_time + max(preroll, 0.0);
    float per_second = max(rate, 0.0);
    float life = max(lifetime, 0.01);
    if (per_second <= 0.0) { frag_color = vec4(0.0); return; }
    float newest = floor(now * per_second);
    float count = min(ceil(life * per_second), float(MAX_PARTICLES));
    vec2 centre = object_center();
    vec2 pixel = v_uv * u_size;
    vec2 wind = vec2(cos(radians(wind_angle)), sin(radians(wind_angle))) * wind_speed;
    // 粒の中心から絵の中身の一番遠い角まで（等倍） どう回しても粒の絵はこの円の中に収まる
    // u_object ではなく u_content で測る 前の変形で広げた絵は u_object の外まである
    vec2 corner = max(abs(u_content.xy - centre), abs(u_content.zw - centre));
    float reach = length(corner) + 1.0;
    vec4 result = vec4(0.0);
    for (int k = 0; k < MAX_PARTICLES; ++k) {
        if (float(k) >= count) break;
        float index = newest - float(k);
        // 番号が負の粒はまだ生まれていない（先に進めておく時間が 0 のとき、頭から密集する）
        if (index < 0.0) continue;
        float age = now - index / per_second;
        if (age < 0.0 || age > life) continue;
        float jitter = randomness * 0.01;
        float a = radians(emit_angle + (rand(index, 1.0) - 0.5) * spread);
        vec2 start = vec2(emitter_x + (rand(index, 2.0) * 2.0 - 1.0) * emit_range, emitter_y);
        float v = speed * (1.0 + (rand(index, 3.0) - 0.5) * jitter);
        vec2 moved = start + vec2(cos(a), sin(a)) * v * age + wind * age
                   + vec2(0.0, 0.5 * gravity * age * age)
                   + vec2(sin(age * 3.0 + index), cos(age * 2.3 + index)) * turbulence;
        // 画面の Y 下向きから、このバッファの Y 上向きへ
        vec2 place = centre + vec2(moved.x, -moved.y);
        float t = age / life;
        float scale = mix(size, size * end_scale * 0.01, t) * 0.01;
        if (scale <= 0.0001) continue;
        // 粒の絵が届かない画素は回す前に外す 絵を読む（4 点を混ぜる）所が一番重く、
        // 粒の数だけ全画素で読むと上限を上げたぶんだけ書き出しが重くなる
        vec2 offset = pixel - place;
        if (dot(offset, offset) > reach * reach * scale * scale) continue;
        float fade_from = 1.0 - clamp(fade * 0.01, 0.0, 1.0);
        float alpha = t > fade_from ? 1.0 - (t - fade_from) / max(1.0 - fade_from, 1e-4) : 1.0;
        vec2 local = offset / scale;
        float spin = radians(rotation * (rand(index, 4.0) * 2.0 - 1.0) * jitter + rotation);
        local = mat2(cos(spin), -sin(spin), sin(spin), cos(spin)) * local;
        vec4 c = sample_pixel(centre + local);
        if (c.a <= 0.0) continue;
        result = over(vec4(c.rgb, c.a * alpha), result);
    }
    frag_color = result;
}
"""
)

_FLIP = _shader(
    """
uniform bool horizontal;
uniform bool vertical;

void main() {
    // 絵の中心を軸に裏返す
    vec2 centre = object_center();
    vec2 p = v_uv * u_size - centre;
    if (horizontal) p.x = -p.x;
    if (vertical) p.y = -p.y;
    frag_color = sample_pixel(centre + p);
}
"""
)

_PASS_THROUGH = (
    PRELUDE
    + """
void main() { frag_color = texture(u_texture, v_uv); }
"""
)


def register_optics_effects() -> None:
    """一覧へ登録する 何度呼んでも 1 回だけ"""
    if "bevel_light" in registry:
        return
    definitions = (
        EffectDefinition(
            kind="bevel_light",
            label="縁の反射",
            category="装飾",
            parameters=(
                SelectSpec(
                    "lighting",
                    "光の当て方",
                    (("specular", "照り返し"), ("diffuse", "拡散")),
                    "specular",
                ),
                TrackSpec("azimuth", "光の方位", -360, 360, -90, unit="度"),
                TrackSpec("elevation", "光の高さ", 0, 90, 0, unit="度"),
                TrackSpec("constant", "強さ", 0, 1000, 50, unit="%"),
                TrackSpec("exponent", "鋭さ", 0.01, 128, 1, step=0.01),
                ColorSpec("color", "光の色", (1.0, 1.0, 1.0, 1.0)),
                SelectSpec("blend", "合成", BLEND_MODES, "add"),
                # 見た目は倍率だが、シェーダは 1 画素あたりの高さの差に掛けて面の傾きを出す
                # 縁の太さを縮めた合成では 1 画素あたりの差が大きくなるので、倍率も同じだけ
                # 縮めないと、1/2 画質で面が 2 倍急になって光り方が変わる
                TrackSpec("surface_scale", "高さの倍率", 0, 100, 10, step=0.1, pixels=True),
                SelectSpec(
                    "profile",
                    "縁の形",
                    (
                        ("straight", "直線"),
                        ("round", "丸"),
                        ("inverted_round", "くぼんだ丸"),
                        ("step", "段"),
                    ),
                    "straight",
                ),
                TrackSpec("thickness", "縁の太さ", 1, 128, 10, step=0.1, unit="px"),
                TrackSpec("blur", "ぼかし", 0, 96, 0, unit="px"),
                CheckSpec("inverted", "へこませる", False),
            ),
            fragment_shader=_BEVEL_LIGHT,
            # 0 が縁までの測り、1 が画質を落とした時の刻み直し（書き出しでは写すだけ）、
            # 2・3 が横と縦のぼかし、4 が光
            passes=5,
        ),
        EffectDefinition(
            kind="lens_blur",
            label="レンズぼかし",
            category="ぼかし",
            parameters=(
                TrackSpec("radius", "範囲", 0, 200, 10, unit="px"),
                TrackSpec("brightness", "明るさ", 0, 1000, 100, unit="%"),
                TrackSpec("edge_strength", "輪の強さ", 0, 20, 2, step=0.1),
                CheckSpec("fixed_size", "大きさを変えない", False),
            ),
            fragment_shader=_LENS_BLUR,
        ),
        EffectDefinition(
            kind="fish_eye",
            label="魚眼",
            category="変形",
            parameters=(
                SelectSpec(
                    "projection",
                    "射影",
                    (
                        ("orthographic", "正射影"),
                        ("equidistant", "等距離射影"),
                        ("stereographic", "立体射影"),
                        ("equisolid", "等立体角射影"),
                    ),
                    "orthographic",
                ),
                TrackSpec("angle", "画角", 1, 179, 90, unit="度"),
                TrackSpec("zoom", "拡大率", 1, 1000, 100, unit="%"),
            ),
            fragment_shader=_FISH_EYE,
        ),
        EffectDefinition(
            kind="ripple",
            label="波紋",
            category="変形",
            parameters=(
                TrackSpec("center_x", "中心 X", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("center_y", "中心 Y", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("amplitude", "振れ幅", -1000, 1000, 20, unit="px"),
                TrackSpec("wavelength", "波長", 1, 4000, 100, unit="px"),
                TrackSpec("period", "周期", 0.01, 60, 2, step=0.01, unit="秒"),
            ),
            fragment_shader=_RIPPLE,
        ),
        EffectDefinition(
            kind="polar",
            label="極座標",
            category="変形",
            parameters=(
                TrackSpec("core", "中心の穴", 0, 4000, 0, step=1, unit="px"),
                TrackSpec("twist", "ねじれ", -3600, 3600, 0, unit="度"),
            ),
            fragment_shader=_POLAR,
        ),
        EffectDefinition(
            kind="stretch",
            label="引き伸ばし",
            category="変形",
            parameters=(
                TrackSpec("center_x", "中心 X", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("center_y", "中心 Y", -4000, 4000, 0, step=1, unit="px"),
                TrackSpec("angle", "角度", -360, 360, 0, unit="度"),
                TrackSpec("stretch", "伸ばす長さ", 0, 10000, 100, step=1, unit="px"),
                TrackSpec("range", "伸ばす幅", 0, 4000, 0, step=1, unit="px"),
                CheckSpec("centering", "両側へ伸ばす", True),
            ),
            fragment_shader=_STRETCH,
        ),
        EffectDefinition(
            kind="reel_spin",
            label="リール回転",
            category="動き",
            parameters=(
                TrackSpec("rotation", "回転", -10000, 10000, 0, unit="%"),
                TrackSpec("direction", "向き", -360, 360, 0, unit="度"),
            ),
            fragment_shader=_REEL_SPIN,
        ),
        EffectDefinition(
            kind="tile",
            label="タイル",
            category="変形",
            parameters=(
                TrackSpec("count_x", "横の数", 1, 100, 2, step=1),
                TrackSpec("count_y", "縦の数", 1, 100, 2, step=1),
            ),
            fragment_shader=_TILE,
        ),
        EffectDefinition(
            kind="inout_wipe",
            label="ワイプで登場",
            category="登場・退場",
            parameters=(
                SelectSpec(
                    "pattern",
                    "形",
                    (
                        ("horizontal", "横"),
                        ("vertical", "縦"),
                        ("circle", "円"),
                        ("square", "四角"),
                        ("clockwise", "時計回り"),
                        ("fade", "フェード"),
                    ),
                    "horizontal",
                ),
                TrackSpec("tolerance", "境目のぼかし", 0, 100, 3, unit="%"),
                TrackSpec("angle", "角度", -360, 360, 0, unit="度"),
                CheckSpec("reverse_in", "登場を逆向きに", False),
                CheckSpec("reverse_out", "退場を逆向きに", False),
                *_in_out_specs(),
            ),
            fragment_shader=_INOUT_WIPE,
        ),
        EffectDefinition(
            kind="directional_key",
            label="2 色の間で抜く",
            category="色",
            keeps_content=True,
            parameters=(
                ColorSpec("background", "抜く色", (0.02, 0.02, 0.02, 1.0)),
                ColorSpec("foreground", "残す色", (0.8, 0.8, 0.8, 1.0)),
                TrackSpec("softness", "境目のぼかし", 0, 100, 3, unit="%"),
                TrackSpec("threshold", "しきい値", 0, 1, 0.02, step=0.001),
                CheckSpec("output_foreground", "混ざった背景の色を取り除く", True),
            ),
            fragment_shader=_DIRECTIONAL_KEY,
        ),
        EffectDefinition(
            kind="repeat_zoom",
            label="反復拡大",
            category="動き",
            parameters=(
                TrackSpec("zoom", "拡大率", 0, 1000, 150, unit="%"),
                TrackSpec("zoom_x", "横の割合", 0, 1000, 100, unit="%"),
                TrackSpec("zoom_y", "縦の割合", 0, 1000, 100, unit="%"),
                TrackSpec("interval", "周期", 0.01, 60, 1, step=0.01, unit="秒"),
                CheckSpec("centering", "等倍を挟んで往復", False),
                *_easing(),
            ),
            fragment_shader=_REPEAT_ZOOM,
        ),
        EffectDefinition(
            kind="jump",
            label="跳ねる",
            category="動き",
            parameters=(
                TrackSpec("height", "高さ", 0, 4000, 50, step=1, unit="px"),
                TrackSpec("stretch", "伸び縮み", 0, 100, 0, step=0.01),
                TrackSpec("period", "1 回の長さ", 0.01, 60, 0.5, step=0.01, unit="秒"),
                TrackSpec("distortion", "着地の潰れ", 0, 100, 0, step=0.01),
                TrackSpec("interval", "休み", 0, 60, 0, step=0.01, unit="秒"),
            ),
            fragment_shader=_JUMP,
        ),
        EffectDefinition(
            kind="particles",
            label="パーティクル",
            category="動き",
            parameters=(
                TrackSpec("rate", "1 秒の数", 0, 2000, 50, step=0.1),
                TrackSpec("lifetime", "寿命", 0.01, 120, 2, step=0.01, unit="秒"),
                TrackSpec("preroll", "先に進めておく時間", 0, 120, 0, step=0.1, unit="秒"),
                TrackSpec("size", "大きさ", 0, 1000, 100, unit="%"),
                TrackSpec("end_scale", "消えるときの大きさ", 0, 1000, 100, unit="%"),
                TrackSpec("emitter_x", "放つ位置 X", -10000, 10000, 0, step=1, unit="px"),
                TrackSpec("emitter_y", "放つ位置 Y（下が正）", -10000, 10000, 0, step=1, unit="px"),
                TrackSpec("emit_range", "放つ幅", 0, 10000, 0, step=1, unit="px"),
                TrackSpec("emit_angle", "放つ角度", -360, 360, 90, unit="度"),
                TrackSpec("spread", "広がり", 0, 360, 0, unit="度"),
                # 速さと重力は秒あたりの画面の画素 単位が PIXEL_UNITS に無いので明に書く
                # 書かないと、1/2 画質で粒が 2 倍遠くまで飛ぶ
                TrackSpec("speed", "速さ", 0, 10000, 100, unit="px/秒", pixels=True),
                TrackSpec("gravity", "重力", -100000, 100000, 0, unit="px/秒²", pixels=True),
                TrackSpec("wind_angle", "風の角度", -360, 360, 0, unit="度"),
                TrackSpec("wind_speed", "風の速さ", 0, 10000, 0, unit="px/秒", pixels=True),
                TrackSpec("turbulence", "揺らぎ", 0, 1000, 0, unit="px"),
                TrackSpec("rotation", "回転", -3600, 3600, 0, unit="度"),
                TrackSpec("fade", "消えていく割合", 0, 100, 0, unit="%"),
                TrackSpec("randomness", "ばらつき", 0, 100, 50, unit="%"),
            ),
            fragment_shader=_PARTICLES,
        ),
        EffectDefinition(
            kind="flip",
            label="反転",
            category="変形",
            parameters=(
                CheckSpec("horizontal", "左右", True),
                CheckSpec("vertical", "上下", False),
            ),
            fragment_shader=_FLIP,
            moves_edges=flipped_edges,
            # どちらも裏返さなければ絵は変わらない クリップが最初から持つ左右反転は
            # この値で付く（:mod:`sashimono.core.commands.fixed`）
            idle_when=(("horizontal", False), ("vertical", False)),
        ),
        EffectDefinition(
            kind="after_image",
            label="残像",
            category="動き",
            parameters=(
                TrackSpec("strength", "残る強さ", 0, 99, 50, unit="%"),
                SelectSpec("mode", "重ね方", (("front", "手前"), ("back", "奥")), "front"),
                ValueSpec("samples", "残す枚数", 12, minimum=1, maximum=60),
            ),
            # 絵は変えない 前のフレームを重ねるのはレンダラ（sashimono.engine.render.renderer）
            fragment_shader=_PASS_THROUGH,
        ),
    )
    for definition in definitions:
        registry.register(definition)
