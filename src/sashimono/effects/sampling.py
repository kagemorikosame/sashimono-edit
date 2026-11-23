"""絵を拡大・縮小して読むときのシェーダの関数 変形のエフェクトと合成で同じ読み方をする

エンジンは絵をストレートのアルファで持つ GL の補間はストレートのまま RGB を混ぜるので、
透明な画素に残った色（YMM4 の格子の透明な白など）が隣へにじみ、縮めた縁が白っぽく
浮くか黒ずむ ここでは事前乗算にしてから補うので、透明な画素の色はどこにも出ない
（#179） 片方だけ直すと、同じ絵を変形で縮めたときと素材を小さく置いたときで縁の色が違う

縮めるときは、出力の 1 画素が入力で覆う範囲を平均する 1 点だけを読むと、細かい格子の
線に乗るか隙間に乗るかで濃さが周期的に揺れ、大きな縞（モアレ）が浮く
"""

from __future__ import annotations

__all__ = ["AREA_SAMPLING"]

#: GLSL の関数 ``premultiplied`` は渡す絵がすでに事前乗算か（合成先のキャンバスの写し）
#: どれも事前乗算の値を返す ストレートへ戻すのは呼ぶ側（戻し方が呼ぶ所ごとに違う）
AREA_SAMPLING = """
// 1 画素を事前乗算で読む 外は端の画素（CLAMP_TO_EDGE と同じ） 外を透明にすると、
// 画面いっぱいの素材を拡大して置いたときに端の 1 列が薄れる
vec4 texel_premul(sampler2D tex, ivec2 p, bool premultiplied) {
    ivec2 size = textureSize(tex, 0);
    vec4 c = texelFetch(tex, clamp(p, ivec2(0), size - 1), 0);
    return premultiplied ? c : vec4(c.rgb * c.a, c.a);
}

// 画素の間を事前乗算のまま直線で補う pixel は入力の画素（左下が原点）
vec4 bilinear_premul(sampler2D tex, vec2 pixel, bool premultiplied) {
    vec2 p = pixel - 0.5;
    ivec2 i = ivec2(floor(p));
    // 端数は GL の補間と同じく 1/256 に丸める 画素の真ん中を読んだつもりが計算の誤差で
    // 1e-7 だけずれると、隣の画素がわずかに混ざって等倍の写しが 1 だけ変わる
    vec2 f = floor((p - floor(p)) * 256.0 + 0.5) / 256.0;
    vec4 low = mix(
        texel_premul(tex, i, premultiplied), texel_premul(tex, i + ivec2(1, 0), premultiplied), f.x
    );
    vec4 high = mix(
        texel_premul(tex, i + ivec2(0, 1), premultiplied),
        texel_premul(tex, i + ivec2(1, 1), premultiplied),
        f.x
    );
    return mix(low, high, f.y);
}

// area_premul が 1 辺に読む点の数 読む位置を寄せる側（変形の画面の端）も同じ数で広がりを測る
int area_steps(float reach) { return int(clamp(ceil(reach - 0.001), 1.0, 8.0)); }

// uv の 1 画素が入力で覆う範囲の平均（事前乗算） du と dv は出力の 1 画素ぶん動いたときの
// uv の動き（dFdx と dFdy） 分かれ道の中では測れないので、呼ぶ側が先に測って渡す
// 読む数は覆う画素の数に合わせ、極端に縮めたときも 8x8 で止めて重くしすぎない
vec4 area_premul(sampler2D tex, vec2 uv, vec2 du, vec2 dv, bool premultiplied) {
    vec2 size = vec2(textureSize(tex, 0));
    vec2 pixel = uv * size;
    vec2 dx = du * size;
    vec2 dy = dv * size;
    float reach = max(length(dx), length(dy));
    // 傾けた板の裏へ回った所の隣は、とても遠い点になる そこまでを範囲にすると、
    // 縁の画素が遠くを平均して薄れる
    if (reach > 256.0) return bilinear_premul(tex, pixel, premultiplied);
    int n = area_steps(reach);
    vec4 sum = vec4(0.0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            vec2 at = (vec2(float(i), float(j)) + 0.5) / float(n) - 0.5;
            sum += bilinear_premul(tex, pixel + dx * at.x + dy * at.y, premultiplied);
        }
    }
    return sum / float(n * n);
}
"""
