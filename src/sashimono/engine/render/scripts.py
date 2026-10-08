"""AviUtl スクリプトを描画の流れに組み込む

スクリプトは 1 枚の画像と描画パラメータを書き換える その結果は「何回・どこへ・
どう変形して描くか」の一覧（:class:`~sashimono.compat.aviutl.objapi.DrawCall`）に
なるので、レンダラはそれを順に合成すればよい

**Lua は Qt を知らない** テキストや図形を作る ``obj.load`` は
:func:`~sashimono.engine.sources.render_source` に委ねる 互換層をエンジンから
切り離しておくと、互換層だけをテストできる
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from sashimono.compat.aviutl import PREFIX
from sashimono.compat.aviutl.catalog import SCRIPT_SUFFIXES, ScriptCatalog, script_catalog
from sashimono.compat.aviutl.control import lua_value
from sashimono.compat.aviutl.mapping import script_filter_effects
from sashimono.compat.aviutl.objapi import DrawCall, EffectRequest, ObjectState
from sashimono.compat.aviutl.runtime import LuaScriptRuntime, blank_image
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND
from sashimono.core.model import AnimatedValue, Clip, Effect, GeneratedSource, ParamValue
from sashimono.effects.definition import registry
from sashimono.engine.sources import drawn_bold_italic, render_source

__all__ = [
    "ScriptStage",
    "is_scene_change",
    "requested_effects",
    "script_catalog",
    "script_effects",
    "split_effects",
    "text_font",
]


#: 積んだ効果を絵へ掛けて返す関数 ``(絵, 効果, フレーム, fps, クリップの長さ) -> 絵``
#: 掛けられないとき（GPU の作れる大きさを超える）は ``None``
#: GPU を持つレンダラが渡す（:class:`~sashimono.engine.render.script_bake.ScriptEffectBaker`）
ApplyEffects = Callable[[np.ndarray, tuple[Effect, ...], int, float, int], np.ndarray | None]


def split_effects(effects: tuple[Effect, ...]) -> tuple[tuple[Effect, ...], tuple[Effect, ...]]:
    """エフェクトを「GPU で処理するもの」と「スクリプト」に分ける"""
    gpu = tuple(e for e in effects if not e.kind.startswith(PREFIX))
    scripts = tuple(e for e in effects if e.kind.startswith(PREFIX))
    return gpu, scripts


def script_effects(effects: tuple[Effect, ...]) -> tuple[Effect, ...]:
    return tuple(e for e in effects if e.kind.startswith(PREFIX) and e.enabled)


def is_scene_change(kind: str) -> bool:
    """シーンチェンジ（``.scn`` ``.scn2``）のスクリプトを指す種別か

    種別の中のファイル名の拡張子で見る（``aviutl:フォルダ/@ディザσ.scn:ディザワイプ(時計)``）
    一覧に引きに行かないのは、スクリプトが手元に無い機械でも前の場面の効果と取り違えない
    ため 取り違えると、前の場面へ掛けるスクリプトとして数えられ、場面も切り替わらない
    """
    if not kind.startswith(PREFIX):
        return False
    path = kind[len(PREFIX) :].rpartition(":")[0]
    suffix = path[path.rfind(".") :].lower() if "." in path else ""
    return SCRIPT_SUFFIXES.get(suffix) == "scn"


class ScriptStage:
    """クリップに積まれた AviUtl スクリプトを走らせる係

    Lua ランタイムは 1 つを使い回す フレームごとに作り直すと、標準ライブラリの
    用意だけで描画の余裕を食う
    """

    def __init__(
        self,
        catalog: ScriptCatalog,
        *,
        screen: tuple[int, int],
        apply_effects: ApplyEffects | None = None,
    ) -> None:
        self._catalog = catalog
        self._screen = screen
        #: ``obj.effect`` で積んだ効果を、絵を読む・変える呼び出しの前に掛ける（#176）
        self._apply_effects = apply_effects
        #: いま走らせているクリップのフレーム・fps・長さ 効果の動きが時刻を見る
        self._timing: tuple[int, float, int] = (0, 30.0, 1)
        self._runtime = LuaScriptRuntime(
            render_source=self._render_source,
            apply_effects=self._apply_requested if apply_effects is not None else None,
        )
        # 共通処理のファイル（.mod2）はスクリプトと同じ場所に置かれている
        self._runtime.set_roots(catalog.roots)

    def set_screen(self, width: int, height: int) -> None:
        self._screen = (width, height)

    def run(
        self,
        clip: Clip,
        effects: tuple[Effect, ...],
        image: np.ndarray,
        *,
        frame: int,
        fps: float,
        layer: int = 0,
        framebuffer: Callable[[], np.ndarray | None] | None = None,
    ) -> tuple[DrawCall, ...]:
        """スクリプトを順に走らせ、描画の一覧を返す

        1 つの状態を渡していくのは AviUtl と同じ 前のスクリプトが動かした位置に、
        次のスクリプトがさらに手を入れる形になる

        ``framebuffer`` はそれまでに下へ重ねた画面を読む関数（``obj.copybuffer`` の ``frm``）
        画面の画素の大きさで、ストレートアルファの絵を返す
        """
        draws, _ = self._run(
            clip, effects, image, frame=frame, fps=fps, layer=layer, framebuffer=framebuffer
        )
        return draws

    def run_scene_change(
        self,
        clip: Clip,
        effects: tuple[Effect, ...],
        before: np.ndarray,
        after: np.ndarray,
        *,
        frame: int,
        fps: float,
        progress: float,
    ) -> tuple[tuple[DrawCall, ...], tuple[Effect, ...]]:
        """シーンチェンジのスクリプトを積んだ順に走らせ、描画の一覧と走らなかった効果を返す

        何本も積んだときは、アニメーション効果と同じく 1 つの ``obj`` を順に渡していく
        （前のスクリプトが削った絵に、次のスクリプトがさらに手を入れる）

        **前の場面をオブジェクト、後の場面をフレームバッファ**に入れて走らせる 返した描画を
        フレームバッファ（後の場面）の上へ重ねた物がその時刻の絵になる 進み具合（0〜1）は
        ``obj.getvalue("scenechange")`` で読ませる

        向きは配布物 4 本（sigma の ``@ディザσ.scn``）の書かれ方から読んだ どれも進み具合が
        0 のときオブジェクトをそのまま描き、1 に近づくほどオブジェクトを削って（ディザで
        抜く・扇や図形や斜めに切る）フレームバッファを見せる 逆に入れると、4 本とも後の
        場面から始まって前の場面で終わる 実物の AviUtl で並べて測ってはいない

        走らなかった効果を返すのは、失敗したときの描画（前の場面をそのまま 1 回）を重ねると、
        最後まで前の場面のまま切り替わらないため 空でなければ、呼ぶ側は切り替え方どおりに
        描き直し、走らなかった物だけを数える（走った物まで数えると、どれを直せばよいか分からない）
        """
        return self._run(
            clip,
            effects,
            before,
            frame=frame,
            fps=fps,
            framebuffer=lambda: after,
            scenechange=progress,
        )

    def _run(
        self,
        clip: Clip,
        effects: tuple[Effect, ...],
        image: np.ndarray,
        *,
        frame: int,
        fps: float,
        layer: int = 0,
        framebuffer: Callable[[], np.ndarray | None] | None = None,
        scenechange: float | None = None,
    ) -> tuple[tuple[DrawCall, ...], tuple[Effect, ...]]:
        """スクリプトを順に走らせ、描画の一覧と、走らなかった効果を返す"""
        state = ObjectState(
            image=image,
            screen_w=self._screen[0],
            screen_h=self._screen[1],
            frame=frame,
            totalframe=max(1, clip.duration),
            framerate=fps,
            layer=layer,
            framebuffer=framebuffer,
            scenechange=scenechange,
        )
        self._timing = (frame, fps, max(1, clip.duration))
        state.base_x, state.base_y = _placed_at(clip, frame)

        failed: list[Effect] = []
        for effect in effects:
            entry = self._catalog.get(effect.kind)
            if entry is None:
                # 手元に無いスクリプトは走らせていない シーンチェンジでは、失敗と同じく
                # 切り替え方どおりに描き直してもらう
                failed.append(effect)
                continue
            _apply_params(state, effect, frame)
            result = self._runtime.run(
                entry.source,
                state,
                header=entry.header,
                script=entry.label,
                folder=entry.folder,
            )
            if result.failed:
                failed.append(effect)

        return state.result(), tuple(failed)

    def expand_text(
        self,
        text: str,
        *,
        frame: int,
        fps: float,
        duration: int,
        font: dict[str, Any] | None = None,
        layer: int = 0,
    ) -> str:
        """テキスト欄に埋め込んだ Lua を、そのフレームの文字にする

        ``font`` はそのテキストオブジェクトの書体の設定 ``obj.getfont`` が
        これを返す 渡さないと、合成フォントのエイリアスのように
        ``obj.getfont`` で大きさや字間を取って組むスクリプトが、
        設定欄に書いた値ではなく既定値で組む

        ``layer`` は ``obj.layer`` PSDToolKit の字幕はテキストの Lua でレイヤー番号ごとに
        字幕の状態を置き、同じレイヤーの吹き出しがそれを読む
        """
        state = ObjectState(
            image=blank_image(1, 1),
            screen_w=self._screen[0],
            screen_h=self._screen[1],
            frame=frame,
            totalframe=max(1, duration),
            framerate=fps,
            layer=layer,
            font=dict(font) if font else {},
        )
        return self._runtime.expand_text(text, state)

    def _apply_requested(
        self, image: np.ndarray, requests: tuple[EffectRequest, ...]
    ) -> np.ndarray | None:
        """``obj.effect`` で積んだ効果を、いまの絵へ掛ける"""
        if self._apply_effects is None:  # pragma: no cover - 渡されたときだけランタイムへ渡す
            return image
        frame, fps, duration = self._timing
        # 描くときと同じ写し方をする（:func:`requested_effects`） 別に写すと、焼き込んだときだけ
        # 項目名や Y の向きが違って掛かる
        effects = tuple(effect for request in requests for effect in _to_effects(request))
        return self._apply_effects(image, effects, frame, fps, duration)

    def _render_source(
        self, kind: str, params: dict[str, object], width: int, height: int
    ) -> np.ndarray:
        """``obj.load`` から呼ばれる テキストや図形を絵にする"""
        source = GeneratedSource(kind=kind, params=_as_params(kind, params))
        drawn = render_source(source, max(1, width), max(1, height))
        if drawn is None:  # pragma: no cover - 種類は呼び出し側が決めている
            return np.zeros((max(1, height), max(1, width), 4), dtype=np.uint8)
        return drawn


def text_font(params: Mapping[str, ParamValue], frame: int) -> dict[str, Any]:
    """テキストオブジェクトの書体の設定を ``obj`` の書体の形にする

    並びは AviUtl2 の ``obj.setfont`` と同じ（lua.txt）
    ``名前, 大きさ, 装飾, 文字色, 影・縁色, 太字, 斜体, 字間, 行間``
    ``obj.getfont`` はこれをそのまま返す

    名前は ``font`` があればその名前、無ければ空 ここで既定の書体名
    （``Yu Gothic UI``）を作らない AviUtl2 の ``obj.getfont`` も、書体を
    選んでいなければ空を返す（lua.txt「フォント名の初期値は空」）

    ただし ``font`` を持たない読み込み経路（AviUtl1 の ``フォント`` が空の
    エイリアス）と、本当に ``Yu Gothic UI`` を選んだ場合は、ここでは区別できない
    合成フォントの ``decorate`` ``decorate_layout`` は書体名を渡しても渡さなくても
    同じ結果を返すことを実物で確かめてあるので、いまはこの粗さで困らない
    """

    def number(name: str, default: float = 0.0) -> float:
        raw = params.get(name, default)
        value = raw.at(frame) if isinstance(raw, AnimatedValue) else raw
        return float(value) if isinstance(value, int | float) else default

    color = params.get("color")
    name = params.get("font")
    # 太字と斜体は描く字に合わせる スタイルを選んでいると欄の値は灰色で効かず、そのまま
    # 渡すと展開した結果だけが古い値で変わる（:func:`drawn_bold_italic`）
    bold, italic = drawn_bold_italic(params, bool(number("bold")), bool(number("italic")))
    return {
        "name": str(name) if isinstance(name, str) else "",
        "size": number("size", 48.0),
        "bold": bold,
        "italic": italic,
        "color": color,
        "given": (
            str(name) if isinstance(name, str) else "",
            number("size", 48.0),
            # 装飾の番号は縁取りや影の組み合わせから作り直すことになる
            # 取り違えた番号を返すより、装飾なし（0）で通す
            0,
            _number_color(color),
            0x000000,
            bold,
            italic,
            number("letter_spacing"),
            number("line_spacing"),
        ),
    }


def _number_color(value: object) -> int:
    """色を ``obj.setfont`` が使う 0xRRGGBB の数にする 読めなければ白"""
    if isinstance(value, tuple) and len(value) >= 3:
        red, green, blue = (max(0, min(255, round(float(part) * 255))) for part in value[:3])
        return (red << 16) | (green << 8) | blue
    if isinstance(value, int):
        return int(value)
    return 0xFFFFFF


def _placed_at(clip: Clip, frame: int) -> tuple[float, float]:
    """クリップの配置の欄の位置 ``obj.x`` ``obj.y`` の値（AviUtl に合わせて Y は下が正）

    見るのは固定の欄（標準描画に当たる物）だけ 後から足した変形は、AviUtl でも
    ``座標`` などのフィルタで、表示基準座標には入らない
    """
    placing = next(
        (e for e in clip.effects if e.fixed and e.enabled and e.kind == TRANSFORM_EFFECT_KIND),
        None,
    )
    if placing is None:
        return 0.0, 0.0

    def number(name: str) -> float:
        raw = placing.params.get(name, 0.0)
        value = raw.at(frame) if isinstance(raw, AnimatedValue) else raw
        return float(value) if isinstance(value, int | float) else 0.0

    return number("pos_x"), -number("pos_y")


def _apply_params(state: ObjectState, effect: Effect, frame: int) -> None:
    """設定欄の値を ``obj`` から見える形へ移す

    ``track0``…``track3`` と ``check0`` は AviUtl の決まった名前へ、それ以外は
    名前付きの値として置く スクリプトはどちらの書き方もする
    """
    definition = registry.get(effect.kind)
    specs = definition.parameters if definition is not None else ()

    # 設定欄は効果ごとの物 トラックバーと同じくチェックも戻す 戻さないと、チェックを持たない
    # 後のスクリプトが、前のスクリプトで入れたチェックを自分の物として読む
    state.track = [0.0, 0.0, 0.0, 0.0]
    state.check0 = False
    state.values.clear()
    for spec in specs:
        raw = effect.params.get(spec.name, spec.default_value())
        value = raw.at(frame) if isinstance(raw, AnimatedValue) else raw
        state.values[spec.name] = lua_value(spec, value)

        if spec.name.startswith("track") and spec.name[5:].isdigit():
            slot = int(spec.name[5:])
            if slot < len(state.track):
                state.track[slot] = float(value)  # type: ignore[arg-type]
        elif spec.name == "check0":
            state.check0 = bool(value)


def _as_params(kind: str, values: dict[str, object]) -> dict[str, ParamValue]:
    """``obj.load`` の引数を、生成オブジェクトのパラメータへ

    数値は :class:`AnimatedValue` に包む 生成側はキーフレームを想定した形で
    値を読むため
    """
    del kind
    params: dict[str, ParamValue] = {}
    for name, value in values.items():
        if isinstance(value, str):
            params[name] = value
        elif isinstance(value, tuple):
            params[name] = tuple(float(part) for part in value)
        elif isinstance(value, bool):
            params[name] = int(value)
        elif isinstance(value, int | float):
            params[name] = AnimatedValue(float(value))
    return params


def requested_effects(call: DrawCall) -> tuple[Effect, ...]:
    """``obj.effect`` で頼まれたフィルタを、こちらのエフェクトへ"""
    return tuple(effect for request in call.effects for effect in _to_effects(request))


def _to_effects(request: EffectRequest) -> tuple[Effect, ...]:
    """項目名と値はエイリアスの読み込みと同じ対応表で写す

    以前はここに別の小さな表を持っていて、``輝度`` と ``明るさ`` が同じ項目へ入り、
    Y の向きも直していなかった 1 つの頼みが 2 つになることがある（クリッピング の
    中心の位置を変更 は切った後の平行移動が続く）
    """
    effects = script_filter_effects(request.original, request.params)
    if not effects:  # pragma: no cover - 対応表にある名前しか頼まれない
        return (Effect(kind=request.kind, params={}),)
    return effects
