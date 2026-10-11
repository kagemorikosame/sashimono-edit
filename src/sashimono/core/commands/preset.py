"""プリセットをクリップへ当てるコマンドを組む（#275）

プリセットは「いまあるクリップに見た目を当てる」物 クリップを置き直すエイリアスとは違い、
当てる先のクリップの場所の事情（タイムラインの位置・長さ・リンク）には触らない

当てる・当てないの線引き

* **当てる**: 足したエフェクト・テキストや図形の見た目・最初から持つ欄（描画・音声）の値・
  不透明度・合成モード・下で切り抜く これらは見た目そのもの
* **既定では当てない**: 文字そのもの（:data:`CONTENT_PARAMS`）と画面の中の位置
  （:data:`POSITION_PARAMS`） 文字は当てる先のクリップの持ち物で、見た目を変えたいだけなのに
  打った文字が消えたら困る（テンプレートの棚の着せ替えと同じ考え） 位置は置いた場所の
  事情で、字幕の見た目を当てたら全部が保存した所へ寄ってしまう どちらも
  :class:`PresetOptions` で当てる側へ切り替えられる
* **当てない**: クリップの長さとタイムラインの位置 長さを変えると後ろのクリップと重なり、
  場所の事情そのもの 代わりにキーフレームを当てる先の長さへ伸び縮みさせる
  （棚と同じ :func:`~sashimono.core.model.fitting.fitted_value`）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sashimono.core.commands.base import Command
from sashimono.core.commands.effects import (
    AddEffect,
    ParamPath,
    RemoveEffect,
    SetClipProperty,
    SetEffectEnabled,
    SetParam,
    SetSource,
)
from sashimono.core.commands.fixed import (
    FIXED_ORDER,
    PICTURE_FIXED,
    SOUND_FIXED,
    TRANSFORM_EFFECT_KIND,
    fixed_effect,
    takes_picture_items,
)
from sashimono.core.io.presets import Preset
from sashimono.core.model import Clip, Effect, GeneratedSource, ParamValue
from sashimono.core.model.fitting import fitted_value

__all__ = [
    "CONTENT_PARAMS",
    "PICTURE_SIDE",
    "POSITION_PARAMS",
    "SOUND_SIDE",
    "BodyOf",
    "PresetOptions",
    "origin_sides",
    "preset_commands",
]

#: 中身の値 当てる先のクリップの持ち物なので、既定では当てない
#:
#: 文字と文字送りは棚の着せ替え（``compat.catalog._KEPT_ON_RESTYLE``）と同じ
#: タイマーの書式と数え方は、書式が空でなければ文字の代わりに出る物で、文字と同じく
#: 何を見せるかを決める 当てると、ふつうのテキストがタイマーに変わる（逆も）
#: 音声波形の音声ファイルは、その場所で使う音の道で、見た目ではない
CONTENT_PARAMS = frozenset(
    {
        "text",
        "reveal",
        "timer_format",
        "timer_start",
        "timer_rate",
        "timer_countdown",
        "timer_length",
        "audio_path",
    }
)

#: 画面の中の位置 テキスト・図形の中身と、描画の欄（配置）の X と Y
#: 拡大率・回転・中心は見た目なので当てる
POSITION_PARAMS = frozenset({"pos_x", "pos_y"})

#: クリップの中身を作っているエフェクト（本体）を返す手 :func:`preset_commands` を見る
BodyOf = Callable[[Clip], Effect | None]

#: エフェクトが掛かる側 絵に掛かる物と音に掛かる物
PICTURE_SIDE = "picture"
SOUND_SIDE = "sound"

#: 場面切り替えの中身の種類 後の場面の列を持つのはこれだけ
_TRANSITION = "transition"


@dataclass(frozen=True, slots=True)
class PresetOptions:
    """当て方 どれも設定（``表示 → 設定…``）から変える 既定は当てる先を壊さない側"""

    #: 文字そのもの（:data:`CONTENT_PARAMS`）も当てる
    with_text: bool = False
    #: 画面の中の位置（:data:`POSITION_PARAMS`）も当てる
    with_position: bool = False
    #: 当てる先に足してあるエフェクトを残し、プリセットのエフェクトを後ろへ足す
    #: 偽なら入れ替える（保存したクリップと同じ見た目にする） 前の版のプリセット
    #: （エフェクトの列だけ）は、この設定に関係なく足す 前の版ではそれが当て方だった
    keep_effects: bool = False


def preset_commands(
    preset: Preset,
    clip: Clip,
    *,
    picture: bool = True,
    sound: bool = False,
    options: PresetOptions | None = None,
    accepts: Callable[[str], bool] | None = None,
    body_of: BodyOf | None = None,
    side_of: Callable[[str], str] | None = None,
    defaults_of: Callable[[str], dict[str, ParamValue] | None] | None = None,
) -> list[Command]:
    """``preset`` を ``clip`` へ当てるコマンド 変わる所が無ければ空

    ``picture`` と ``sound`` は当てる先が絵を描くか・音を鳴らすか（設定パネルが描画と
    音声の組を出すかと同じ） 出さない組の欄の値は当てない
    ``accepts`` はエフェクトの種類を足してよいか 渡さなければ全部足す（コア層はエフェクトの
    定義を読めないので、音だけのクリップに映像のエフェクトを足さないのは画面の側で決める）
    ``body_of`` はクリップの中身を作っているエフェクト（本体）を返す手 AviUtl の
    カスタムオブジェクトは空のテキストの最初のエフェクトにスクリプトを置いて中身にする
    （``compat.aviutl.custom_object.custom_object_script``） 見分けるのに互換層とエフェクトの
    定義が要るので、画面の側から渡す

    本体は見た目ではなく中身として扱う（文字と同じ）
    * 当てる先の本体は入れ替えでも消さない 消すと何も描かないクリップが残る
    * プリセットの本体は「文字も一緒に」のときだけ当てる そのとき当てる先の本体は入れ替える
      （同じクリップを作り直したいときの当て方 テキストへ当てるとカスタムオブジェクトになる）
    * 本体を持つクリップへ本体の無いプリセットを当てても、文字は当てない 文字を入れると
      空のテキストでなくなり、本体がカスタムオブジェクトの中身として読まれなくなる
    当てる前に断る形は取らない 本体を残せば、足したエフェクト・欄の値・不透明度は
    カスタムオブジェクトにもそのまま意味を持つ

    ``side_of`` はエフェクトの種類が絵（:data:`PICTURE_SIDE`）と音（:data:`SOUND_SIDE`）の
    どちらに掛かるか 渡さなければどれも絵 入れ替えは側ごとに、保存したクリップと当てる先の
    両方が持つ側だけ（:func:`_effect_commands`）
    ``defaults_of`` は中身の種類の既定の値 中身を当てるときの土台にする 渡さなければ
    当てる先の値を土台にする（保存したクリップが持たない項目は当てる先の値のまま残る）
    """
    chosen = options if options is not None else PresetOptions()
    fit = _fitter(preset, clip)
    target_body = body_of(clip) if body_of is not None else None
    saved_body = _saved_body(preset, body_of)
    commands: list[Command] = []
    keep_content = target_body is not None and saved_body is None
    commands.extend(
        _source_commands(
            preset, clip, chosen, fit, keep_content=keep_content, defaults_of=defaults_of
        )
    )
    target_sides = frozenset(
        side for side, on in ((PICTURE_SIDE, picture), (SOUND_SIDE, sound)) if on
    )
    commands.extend(
        _effect_commands(
            preset,
            clip,
            chosen,
            fit,
            accepts,
            target_body=target_body,
            saved_body=saved_body,
            replaced_sides=origin_sides(preset) & target_sides,
            side_of=side_of if side_of is not None else (lambda _kind: PICTURE_SIDE),
        )
    )
    commands.extend(_fixed_commands(preset, clip, chosen, fit, picture=picture, sound=sound))
    if picture and takes_picture_items(clip):
        # 場面切り替えとフィルタは不透明度・合成モード・切り抜きを読まない（設定パネルも
        # 描画の組を出さない） 当てても何も変わらない段が積まれるだけ
        commands.extend(_clip_commands(preset, clip, fit))
    return commands


def _fitter(preset: Preset, clip: Clip) -> Callable[[ParamValue], ParamValue]:
    """キーフレームを当てる先の長さへ合わせる手 前の版のプリセットは長さを持たないのでそのまま

    長さ 60 で作った (0, 50) の動きを長さ 120 のクリップへそのまま当てると、半分で止まる
    """
    span = preset.span
    if span is None:
        return lambda value: value
    last = clip.duration - 1
    return lambda value: fitted_value(value, span, last)


def origin_sides(preset: Preset) -> frozenset[str]:
    """保存したクリップが持っていた側（絵・音） 当てる先のどちらの側を入れ替えてよいか

    保存した物は :attr:`Preset.picture` と :attr:`Preset.sound` に持つ 項目の無いファイルは
    中身から推し量る 不透明度・中身・描画の欄があれば絵、音声の欄があれば音
    どちらも分からない物（前の版のエフェクトだけのプリセット）は空 そういう物はもともと
    足すだけで入れ替えないので、空でも困らない
    """
    picture = preset.picture
    if picture is None:
        picture = (
            preset.opacity is not None
            or preset.source is not None
            or any(effect.kind in PICTURE_FIXED for effect in preset.fixed)
        )
    sound = preset.sound
    if sound is None:
        sound = any(effect.kind in SOUND_FIXED for effect in preset.fixed)
    return frozenset(side for side, on in ((PICTURE_SIDE, picture), (SOUND_SIDE, sound)) if on)


def _saved_body(preset: Preset, body_of: BodyOf | None) -> Effect | None:
    """プリセットを保存したクリップの本体 保存したときのクリップの形へ組み直して見分ける"""
    if body_of is None or preset.source is None:
        return None
    return body_of(
        Clip(
            timeline_start=0,
            duration=preset.span or 1,
            source=preset.source,
            effects=preset.effects,
        )
    )


def _source_commands(
    preset: Preset,
    clip: Clip,
    options: PresetOptions,
    fit: Callable[[ParamValue], ParamValue],
    *,
    keep_content: bool,
    defaults_of: Callable[[str], dict[str, ParamValue] | None] | None,
) -> list[Command]:
    saved = preset.source
    current = clip.source
    if saved is None or current is None or saved.kind != current.kind:
        # 種類の違う中身は当てない テキストのクリップを図形に変えるのは置き直しで、
        # 見た目を当てる操作ではない（置き直すならエイリアス）
        return []
    kept = set(POSITION_PARAMS) if not options.with_position else set()
    if keep_content or not options.with_text:
        kept |= CONTENT_PARAMS
    # 土台は種類の既定の値 AviUtl や YMM4 から読んだテキストは持つ項目が少なく、当てる先の
    # 値を土台にすると、保存したクリップに無い項目（スタイル・折り返しの幅など）が当てる先の
    # まま残り、保存した見た目にならない（棚の着せ替えの #283 と同じ考え） 残す項目だけを
    # 当てる先から戻す
    known = defaults_of(current.kind) if defaults_of is not None else None
    # 定義の分からない種類（手元に無いスクリプトなど）は当てる先の値を土台にする
    # 空を土台にすると、当てる先の項目が保存したクリップに無いだけで消える
    defaults = known if known is not None else dict(current.params)
    params = {
        **defaults,
        **{name: fit(value) for name, value in saved.params.items() if name not in kept},
    }
    params.update({name: current.params[name] for name in kept if name in current.params})
    if params == {**defaults, **current.params}:
        return []
    return [SetSource(clip.id, GeneratedSource(kind=current.kind, params=params))]


def _effect_commands(
    preset: Preset,
    clip: Clip,
    options: PresetOptions,
    fit: Callable[[ParamValue], ParamValue],
    accepts: Callable[[str], bool] | None,
    *,
    target_body: Effect | None,
    saved_body: Effect | None,
    replaced_sides: frozenset[str],
    side_of: Callable[[str], str],
) -> list[Command]:
    transition = clip.source is not None and clip.source.kind == _TRANSITION
    # 前の版のプリセットは長さを持たない それは足すだけの当て方で作られた物なので、
    # 入れ替えると、今まで足して重ねていた人の手元でエフェクトが消える
    replacing = preset.span is not None and not options.keep_effects
    # 中身の種類が違えば本体も当てない（中身と同じ決まり） 図形に本体を足しても、
    # 空のテキストでないのでカスタムオブジェクトにならず、ふつうのエフェクトとして残る
    same_kind = (
        clip.source is not None
        and preset.source is not None
        and clip.source.kind == preset.source.kind
    )
    swap_body = options.with_text and saved_body is not None and same_kind
    commands: list[Command] = []
    if swap_body and saved_body is not None:
        if target_body is not None:
            commands.append(RemoveEffect(clip.id, target_body.id))
        # 本体は列の頭（カスタムオブジェクトは最初のエフェクトを本体として読む）
        # 足してよい種類かは見ない 中身なので、エフェクトの一覧に出るかとは関係ない
        commands.append(AddEffect(clip.id, _fresh(saved_body, fit), index=0))
    for after, raw in ((False, preset.effects), (True, preset.after_effects)):
        if after and not transition:
            # 後の場面の列は場面切り替えだけが読む ほかのクリップへ足しても描かれず、
            # 設定パネルにも出ないので、消すこともできない列が残る
            continue
        # 本体は上で中身として扱ったので、見た目の列からは外す
        stack = [e for e in raw if after or saved_body is None or e.id != saved_body.id]
        added = [effect for effect in stack if accepts is None or accepts(effect.kind)]
        # 入れ替えは側（絵・音）ごとに、列ごとに決める 入れ替えるのは次の両方のとき
        # * 保存したクリップと当てる先の両方がその側を持つ（``replaced_sides``）
        #   映像の見た目を保存した物を音のクリップへ当てて、音の側（残響など）を消さない
        #   保存したクリップが持たない側は、エフェクトが無かったのではなく見ていなかっただけ
        # * プリセットのその側の列から当たる物がある か、その側の列がもともと空
        #   空なら、その側にエフェクトの無い見た目を保存した物なので、消すのが保存した通り
        #   エフェクトがあるのに 1 つも当たらない（足せない種類だけ）とき、消してから何も
        #   足さないと、当てる先のエフェクトだけが消える
        # 種類ごとに入れ替える形は取らない ぼかしを当てたら影も消えた、のように何が残るかが
        # プリセットの中身しだいで読めなくなる
        own = clip.after_effects if after else clip.effects
        for side in sorted(replaced_sides if replacing else ()):
            saved_side = [e for e in stack if side_of(e.kind) == side]
            if saved_side and not any(side_of(e.kind) == side for e in added):
                continue
            # 固定の項目（最初から持つ欄）は外せないので残す 外そうとすると命令が断られ、
            # 当てる操作ごと取り消しになる 本体も残す（入れ替えるときは上で外してある）
            commands.extend(
                RemoveEffect(clip.id, e.id, after=after)
                for e in own
                if not e.fixed
                and side_of(e.kind) == side
                and (after or target_body is None or e.id != target_body.id)
            )
        commands.extend(AddEffect(clip.id, _fresh(e, fit), after=after) for e in added)
    return commands


def _fresh(effect: Effect, fit: Callable[[ParamValue], ParamValue]) -> Effect:
    # ID を振り直す 同じプリセットを 2 回当てたときに ID が重なると、片方を消したつもりで
    # 両方消える 固定の印も付けない（印のまま足すと外せないエフェクトが増える）
    return Effect(
        kind=effect.kind,
        params={name: fit(value) for name, value in effect.params.items()},
        enabled=effect.enabled,
    )


def _fixed_commands(
    preset: Preset,
    clip: Clip,
    options: PresetOptions,
    fit: Callable[[ParamValue], ParamValue],
    *,
    picture: bool,
    sound: bool,
) -> list[Command]:
    commands: list[Command] = []
    for item in preset.fixed:
        if item.kind not in FIXED_ORDER:
            # 知らない欄（新しい版の本体が足した物） どこへ並べるか決められない
            continue
        if item.kind in PICTURE_FIXED and not (picture and takes_picture_items(clip)):
            continue
        if item.kind in SOUND_FIXED and not sound:
            continue
        values = {
            name: fit(value)
            for name, value in item.params.items()
            if options.with_position
            or item.kind != TRANSFORM_EFFECT_KIND
            or name not in POSITION_PARAMS
        }
        own = next((e for e in clip.effects if e.fixed and e.kind == item.kind), None)
        if own is None:
            # 前の版のファイルのクリップは欄をまだ持たない 設定パネルで触ったときと同じく
            # 欄を足してから値を入れる（同じ段に積むので、1 回の取り消しで戻る）
            base = fixed_effect(item.kind)
            commands.append(
                AddEffect(
                    clip.id,
                    Effect(
                        kind=item.kind,
                        params={**base.params, **values},
                        enabled=item.enabled,
                        fixed=True,
                    ),
                )
            )
            continue
        commands.extend(
            SetParam(ParamPath.of_effect(clip.id, own.id, name), value)
            for name, value in values.items()
            if own.params.get(name) != value
        )
        if own.enabled != item.enabled:
            commands.append(SetEffectEnabled(clip.id, own.id, item.enabled))
    return commands


def _clip_commands(
    preset: Preset, clip: Clip, fit: Callable[[ParamValue], ParamValue]
) -> list[Command]:
    commands: list[Command] = []
    if preset.opacity is not None:
        opacity = fit(preset.opacity)
        if opacity != clip.opacity:
            commands.append(SetParam(ParamPath.of_clip(clip.id, "opacity"), opacity))
    if preset.blend_mode is not None and preset.blend_mode != clip.blend_mode:
        commands.append(SetClipProperty(clip.id, "blend_mode", preset.blend_mode))
    if preset.clip_to_below is not None and preset.clip_to_below != clip.clip_to_below:
        commands.append(SetClipProperty(clip.id, "clip_to_below", preset.clip_to_below))
    return commands
