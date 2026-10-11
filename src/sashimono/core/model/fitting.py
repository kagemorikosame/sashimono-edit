"""動く値（キーフレーム）を、別の長さのクリップへ伸び縮みさせる

テンプレートの棚の着せ替え（:mod:`sashimono.compat.catalog`）とプリセットの適用
（:mod:`sashimono.core.commands.preset`）が同じ決まりで合わせる 互換層に置いたままだと、
コア層のプリセットが互換層を読むことになるので、コア層のここへ置く
"""

from __future__ import annotations

from dataclasses import replace

from sashimono.core.model.effect import AnimatedValue, Effect, Keyframe, ParamValue

__all__ = ["fitted_effect", "fitted_value"]


def fitted_effect(effect: Effect, span: int, target_last: int) -> Effect:
    """エフェクトの動く値を、着せる先の長さへ合わせた写し"""
    return replace(
        effect,
        params={
            name: fitted_value(value, span, target_last) for name, value in effect.params.items()
        },
    )


def fitted_value(value: ParamValue, span: int, target_last: int) -> ParamValue:
    """動く値の時刻を、テンプレートの長さから着せる先の長さへ伸び縮みさせる

    AviUtl の中間点は**そのエイリアス自身の長さに対する絶対フレーム**で書かれて
    いる（``frame=244,333,423`` のように始まり・中間点・終わりが並ぶ）
    そのまま写すと、180 フレームのテンプレートを 60 フレームの字幕に着せたときに
    動きの 3 分の 1 で止まり、残りは静止する 着せるときは文字と長さを今のまま
    残す決まりなので、動きの側を尺に合わせる

    最初と最後の点がクリップの両端に来るように写す 端どうしを合わせないと、
    テンプレートの終わりの見た目（着地した位置）が出ないまま終わる

    ``target_last`` は**終わりの点を置くフレーム**で、呼ぶ側が決める
    タイムラインのクリップへ着せるなら ``長さ - 1``、YMM4 のアイテムへ揃えるなら
    その ``Length`` そのもの（YMM4 は最後の点を長さの位置に置く）
    ここを 1 つに決め打ちすると、どちらかで 1 フレームずれる
    """
    if not isinstance(value, AnimatedValue) or not value.keyframes:
        return value
    if target_last <= 0:
        # 1 フレームのクリップ 動く余地が無いので終わりの値だけを残す
        # そのまま返すと、クリップの外に出たキーフレームが残ったままになり、
        # 唯一のフレームではテンプレートの**始まり**の値が出る
        return replace(value, keyframes=(replace(value.keyframes[-1], frame=0),))

    # 終わりのフレームの決まりが 2 つある AviUtl は最後の点を span - 1 に置き
    # （``frame=0,89,179`` で長さ 180）、YMM4 は span に置く（``Length`` そのもの）
    # 取り違えると倍率の分母が 1 ずれ、中間点が 1 フレームずれた所へ移る
    # 実際の最後の点が span まで届いていれば、そちらを終わりとして読む
    source_last = max(span - 1, value.keyframes[-1].frame)
    last = target_last
    if source_last <= 0 or source_last == last:
        return value
    scale = last / source_last
    moved: list[Keyframe] = []
    for keyframe in value.keyframes:
        frame = min(round(keyframe.frame * scale), last)
        if moved and frame <= moved[-1].frame:
            # 縮めると同じフレームに重なる 同じ所に 2 つは置けないので 1 つずらす
            frame = moved[-1].frame + 1
        if frame > last:
            # ずらす先が無いほど短い 途中の点を落としてでも**終わりの値**は残す
            # 終わりを落とすと、着地した見た目にならないまま止まる
            moved[-1] = replace(keyframe, frame=last)
            continue
        moved.append(replace(keyframe, frame=frame))
    return replace(value, keyframes=tuple(moved))
