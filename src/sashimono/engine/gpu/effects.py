"""エフェクトチェーンの実行

クリップ 1 本ぶんの絵に、エフェクトを順に掛ける 2 枚のフレームバッファを
交互に使い（ピンポン）、1 つのエフェクトの出力が次の入力になる

シェーダのコンパイルは初回だけ行い、以降は使い回す エフェクトを付け外しする
たびにコンパイルが走ると、パラメータをスライダーで動かしただけで固まる
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection
from dataclasses import dataclass

from OpenGL import GL

from sashimono.core.model import Effect, ParamValue
from sashimono.effects import EffectDefinition, registry
from sashimono.effects.definition import object_pivot, turned_object
from sashimono.effects.sampling import AREA_SAMPLING
from sashimono.effects.spec import (
    CheckSpec,
    ColorSpec,
    FileSpec,
    GradientSpec,
    GridSpec,
    SelectSpec,
    TrackSpec,
    ValueSpec,
)
from sashimono.engine.gpu.glutil import (
    FULL_RECT,
    VERTEX_SHADER,
    Framebuffer,
    Program,
    ScreenQuad,
    ShaderError,
    Texture,
)
from sashimono.engine.gpu.images import EffectImages

__all__ = ["EffectProcessor", "piece_grid", "srgb_to_linear"]


_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Compiled:
    """コンパイル済みのエフェクト"""

    definition: EffectDefinition
    program: Program


def srgb_to_linear(value: float) -> float:
    """sRGB の 0..1 をリニアへ

    色パラメータは sRGB で持っている（ユーザーが指定するのも画面に出るのも
    sRGB だから） シェーダはリニアで動くので、渡す直前に 1 度だけ変換する
    """
    if value <= 0.04045:
        return value / 12.92
    return float(((value + 0.055) / 1.055) ** 2.4)


def _set_gradient(program: Program, spec: GradientSpec, value: tuple[float, ...]) -> None:
    """色の止まりの並び（#364）を ``名前_count`` ``名前_colors`` ``名前_offsets`` へ

    数は**毎回**渡す uniform の値はプログラムに残るので、並びを持たない（前からの色の項目で
    塗る）クリップが、前に描いた別のクリップの止まりで塗られる
    色は sRGB のまま渡す シェーダは sRGB のまま混ぜるので、リニアへ往復させると端数が揺れる
    """
    stops = GradientSpec.stops(value)
    program.set_int(f"{spec.name}_count", len(stops))
    if not stops:
        return
    program.set_vec4_array(f"{spec.name}_colors", [colour for _, colour in stops])
    program.set_float_array(f"{spec.name}_offsets", [position for position, _ in stops])


class EffectProcessor:
    """エフェクトを順に適用する

    GL コンテキストが current な状態で使うこと ``quad`` は
    :class:`~sashimono.engine.gpu.Compositor` と共有する
    """

    def __init__(self, width: int, height: int, quad: ScreenQuad) -> None:
        self._quad = quad
        self._buffers = (Framebuffer(width, height), Framebuffer(width, height))
        #: エフェクトに入ってきた元の絵 ``u_source`` として渡す
        #: グローや影は「ぼかした結果」と「元の絵」の両方を要るので、
        #: ピンポンで上書きされる前に取っておく必要がある
        self._source = Framebuffer(width, height)
        #: 部分フィルタへ来たときの絵 後ろのエフェクトを掛け終えたら、範囲の外をこれへ戻す
        #: 使うまで作らない 部分フィルタを積まないクリップで画面 1 枚ぶんの GPU メモリを取らない
        self._held: Framebuffer | None = None
        self._blit = Program(VERTEX_SHADER, _BLIT_FRAGMENT)
        self._programs: dict[str, _Compiled | None] = {}
        #: エフェクトが読む画像（画像合成の絵、縁取りの模様）
        self._images = EffectImages()
        self._front = 0
        self._object: tuple[float, float, float, float] = (0.0, 0.0, float(width), float(height))
        self._origin: tuple[float, float] = (float(width) * 0.5, float(height) * 0.5)
        #: 絵の中身が載りうる範囲（u_content） 素材を置いた直後は u_object と同じ
        self._content: tuple[float, float, float, float] = self._object
        self._duration = 0
        #: 事前乗算で渡される絵（合成先のキャンバス）が sRGB で符号化した値か
        #: 重ね合わせを sRGB で行うプロジェクトで真にする（:class:`Compositor` の ``encoded``）
        #: 素材のテクスチャやエフェクトの結果はどちらでもリニアなので、事前乗算の絵だけに効く
        self.canvas_encoded = False
        #: 合成の画素 1 つが画面（プロジェクトの解像度）の画素いくつ分かの逆数
        #: 等倍の書き出しは 1、1/2 画質のプレビューは 0.5 画素で決める設定
        #: （:attr:`TrackSpec.in_pixels`）はこれを掛けてから渡す 掛けないと、小さく
        #: 合成したプレビューで位置もぼかしの強さも 2 倍・4 倍に出る（Issue #151）
        self.pixel_scale = 1.0

    @property
    def width(self) -> int:
        return self._buffers[0].width

    @property
    def height(self) -> int:
        return self._buffers[0].height

    def resize(self, width: int, height: int) -> None:
        for buffer in (*self._buffers, self._source):
            buffer.resize(width, height)
        if self._held is not None:
            self._held.resize(width, height)

    def has_work(self, effects: tuple[Effect, ...]) -> bool:
        """描画を伴うエフェクトが 1 つでもあるか

        無ければ中間バッファを経由せず、素材をそのまま合成できる エフェクトの
        無いクリップで全画面のパスが 1 回増えるのは、そのまま再生の余裕を削る
        既定のままの配置と反転（:meth:`EffectDefinition.is_idle`）は数えない 置いた
        クリップすべてに付くので、数えると全クリップが中間バッファを通る
        """
        return any(
            self._compile(effect) is not None and not _idle(effect)
            for effect in effects
            if effect.enabled
        )

    def apply(
        self,
        source: Texture | Framebuffer,
        effects: tuple[Effect, ...],
        *,
        frame: int,
        fps: float,
        source_rect: tuple[float, ...] = FULL_RECT,
        flip_source: bool = True,
        duration: int = 0,
        bounds: tuple[float, float, float, float] | None = None,
        origin: tuple[float, float] | None = None,
        premultiplied: bool = False,
    ) -> Framebuffer:
        """``source`` にエフェクトを掛けた結果のバッファを返す

        ``source_rect`` は、入力をバッファのどこに置くかをクリップ空間で指定する
        素材とプロジェクトの解像度が違うときに、ここで収める

        ``duration`` はクリップの長さ（フレーム） 登場と退場の動き（YMM4 の
        ``InOut*``）は終わりから逆算するので、長さを知らないと退場が始まらない

        ``bounds`` は絵の中身が実際にある範囲（画素、左・上・右・下、左上が原点）
        省くと ``source_rect`` 全体 テキストや図形は画面と同じ大きさの絵で届くので、
        全体を基準にすると角丸が画面の角に付き、中心基準の動きが画面の中央で回る

        ``origin`` は絵の原点（画素、左上が原点） 省くと ``bounds`` の中央
        範囲の中央と原点が離れる絵（場面切り替えの場面）だけが渡す
        """
        self._front = 0
        self._draw_source(source, source_rect, flip_source, premultiplied)
        width, height = float(self.width), float(self.height)
        #: 絵が置かれた範囲（画素、GL の向き） 角丸や中心基準の動きが使う
        if bounds is not None:
            left_px, top_px, right_px, bottom_px = bounds
            self._object = (left_px, height - bottom_px, right_px, height - top_px)
        else:
            left, bottom, right, top = (float(v) for v in source_rect[:4])
            self._object = (
                (left + 1.0) * 0.5 * width,
                (bottom + 1.0) * 0.5 * height,
                (right + 1.0) * 0.5 * width,
                (top + 1.0) * 0.5 * height,
            )
        if origin is not None:
            self._origin = (origin[0], height - origin[1])
        else:
            self._origin = (
                (self._object[0] + self._object[2]) * 0.5,
                (self._object[1] + self._object[3]) * 0.5,
            )
        self._duration = max(duration, 0)
        self._content = self._object

        #: 開いている部分フィルタ 後ろのエフェクトはこの範囲の中だけに効く
        scope: tuple[_Compiled, Effect] | None = None
        for effect in effects:
            # 何もしない値の物はパスを通さない 通すと 1 回ごとに画素を読み直すだけ遅くなる
            if not effect.enabled or _idle(effect):
                continue
            compiled = self._compile(effect)
            if compiled is None:
                continue
            if effect.fixed and scope is not None:
                # クリップが最初から持つ欄（配置・反転）は範囲の外にある 閉じずに掛けると、
                # 部分フィルタを足したクリップの X を動かしたとき、範囲の中の絵だけが動く
                self._close_scope(*scope, frame=frame, fps=fps)
                scope = None
            if compiled.definition.scopes_following:
                # 次の部分フィルタで前の範囲を閉じる 入れ子にしないのは AviUtl の
                # 部分フィルタと同じ 入れ子にすると、並びだけ見てどこまで効くか読めない
                if scope is not None:
                    self._close_scope(*scope, frame=frame, fps=fps)
                self._copy(self._buffers[self._front], self._held_buffer())
                scope = (compiled, effect)
                continue
            self._apply_one(compiled, effect, frame=frame, fps=fps)
        if scope is not None:
            self._close_scope(*scope, frame=frame, fps=fps)

        return self._buffers[self._front]

    def release(self) -> None:
        for buffer in (*self._buffers, self._source):
            buffer.release()
        if self._held is not None:
            self._held.release()
            self._held = None
        for compiled in self._programs.values():
            if compiled is not None:
                compiled.program.release()
        self._programs.clear()
        self._images.release()
        self._blit.release()

    # --- 内部 ---

    def _draw_source(
        self,
        source: Texture | Framebuffer,
        rect: tuple[float, ...],
        flip: bool,
        premultiplied: bool = False,
    ) -> None:
        """素材を先頭のバッファへ置く フレームバッファなら、その色のテクスチャを読む"""
        target = self._buffers[self._front]
        target.bind(clear=(0.0, 0.0, 0.0, 0.0))
        GL.glDisable(GL.GL_BLEND)
        self._blit.use()
        self._blit.set_vec4("u_rect", rect)
        self._blit.set_bool("u_flip", flip)
        # エフェクトはストレートアルファで受け取る 事前乗算で溜まった絵は戻してから置く
        self._blit.set_bool("u_premultiplied", premultiplied)
        # エフェクトはリニアで動く sRGB で重ねたキャンバスを符号化したまま渡すと、
        # ぼかしやグローの広がりが暗く沈み、戻すときにもう 1 度符号化されて白っぽく浮く
        self._blit.set_bool("u_decode", premultiplied and self.canvas_encoded)
        handle = source.color if isinstance(source, Framebuffer) else source.handle
        self._blit.bind_texture("u_texture", handle)
        self._quad.draw()

    def _held_buffer(self) -> Framebuffer:
        if self._held is None:
            self._held = Framebuffer(self.width, self.height)
        return self._held

    def _close_scope(self, compiled: _Compiled, effect: Effect, *, frame: int, fps: float) -> None:
        """部分フィルタを閉じる 範囲の外を、部分フィルタへ来たときの絵へ戻す

        部分フィルタのシェーダは ``u_source`` を「掛ける前」、``u_texture`` を「掛けた後」
        として範囲で混ぜる 取っておいた絵を ``u_source`` に置いてから掛ける
        """
        self._copy(self._held_buffer(), self._source)
        self._apply_one(compiled, effect, frame=frame, fps=fps, keep_source=True)

    def _apply_one(
        self,
        compiled: _Compiled,
        effect: Effect,
        *,
        frame: int,
        fps: float,
        keep_source: bool = False,
    ) -> None:
        """エフェクト 1 つを、必要なパス数だけ掛ける

        ``keep_source`` が真なら ``u_source`` を書き換えない（部分フィルタを閉じるとき）
        """
        definition = compiled.definition
        program = compiled.program

        # 元の絵を控えておく u_source を使うエフェクト（グロー・影）が要る
        if not keep_source:
            self._copy(self._buffers[self._front], self._source)

        for index in range(definition.passes):
            source_buffer = self._buffers[self._front]
            target = self._buffers[1 - self._front]

            target.bind(clear=(0.0, 0.0, 0.0, 0.0))
            GL.glDisable(GL.GL_BLEND)
            program.use()
            program.set_vec4("u_rect", FULL_RECT)
            program.set_bool("u_flip", False)
            program.set_vec2("u_size", (float(target.width), float(target.height)))
            program.set_int("u_pass", index)
            program.set_float("u_frame", float(frame))
            program.set_float("u_time", float(frame) / fps if fps else 0.0)
            program.set_float("u_fps", fps)
            program.set_float("u_duration", float(self._duration) / fps if fps else 0.0)
            program.set_vec4("u_object", self._object)
            program.set_vec2("u_origin", self._origin)
            program.set_vec4("u_content", self._content)
            program.set_float("u_pixel_scale", self.pixel_scale)
            program.bind_texture("u_texture", source_buffer.color, unit=0)
            program.bind_texture("u_source", self._source.color, unit=1)
            self._set_parameters(program, definition, effect, frame)

            if definition.pieces is not None:
                self._draw_pieces(program, definition, effect, frame, target)
                continue
            self._quad.draw()
            self._front = 1 - self._front

        self._grow_object(definition, effect, frame)
        if not definition.keeps_content:
            # 中身をどこへ動かしたかは分からない（変形・揺れ・散らす物）ので、後ろの
            # エフェクトにはバッファ全体を中身の範囲として渡す u_object で決めると、
            # 前の変形で広げた絵の粒や欠片が元の大きさで切れる（#199）
            self._content = (0.0, 0.0, float(self.width), float(self.height))

    def _draw_pieces(
        self,
        program: Program,
        definition: EffectDefinition,
        effect: Effect,
        frame: int,
        target: Framebuffer,
    ) -> None:
        """中身の範囲に掛かる升目を 1 つずつ四角として ``target`` へ描き、先頭のバッファへ戻す

        欠片は重なりうるので、事前乗算で描いた順に重ねる エフェクトの間はストレートアルファで
        受け渡すので、重ね終えてから戻す 描いた後の先頭（``_front``）は変わらない
        """
        cell = _piece_size(definition, effect, frame, self.pixel_scale)
        first, columns, rows = piece_grid(self._content, cell, self.width, self.height)
        program.set_float("u_cell", cell)
        program.set_ivec2("u_first", first)
        program.set_int("u_columns", max(columns, 1))

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        self._quad.draw_instanced(columns * rows)
        GL.glDisable(GL.GL_BLEND)

        # 読み終えた入力のバッファへ、ストレートアルファに戻して書く 入力はもう読まない
        self._buffers[self._front].bind(clear=(0.0, 0.0, 0.0, 0.0))
        self._blit.use()
        self._blit.set_bool("u_premultiplied", True)
        self._blit.set_bool("u_decode", False)
        self._blit.set_vec4("u_rect", FULL_RECT)
        self._blit.set_bool("u_flip", False)
        self._blit.bind_texture("u_texture", target.color)
        self._quad.draw()

    def _grow_object(self, definition: EffectDefinition, effect: Effect, frame: int) -> None:
        """入れ物を広げるエフェクトの後で、絵の置かれた範囲を広げる

        広げないと、後ろに積んだミラーや角丸が**広げる前の範囲**で動く
        AviUtl の 領域拡張 → ミラー は、広げたぶんだけ鏡像が離れる並べ方
        """
        if definition.turns_object:
            pivot = object_pivot(
                _choice(definition, effect, "pivot_h"),
                _choice(definition, effect, "pivot_v"),
                (
                    _number(definition, effect, "anchor_x", frame, self.pixel_scale),
                    _number(definition, effect, "anchor_y", frame, self.pixel_scale),
                ),
                self._object,
                (float(self.width), float(self.height)),
                self._origin,
            )
            self._object = turned_object(self._object, pivot)
        if definition.expands_object is None:
            return
        top, bottom, left, right = (
            _number(definition, effect, name, frame, self.pixel_scale)
            for name in definition.expands_object
        )
        self._object = (
            self._object[0] - left,
            self._object[1] - bottom,
            self._object[2] + right,
            self._object[3] + top,
        )

    def _set_parameters(
        self, program: Program, definition: EffectDefinition, effect: Effect, frame: int
    ) -> None:
        """パラメータを uniform へ 名前はそのまま使う

        値は仕様の型へ寄せてから渡す ファイルから読んだ値は型までしか確かめて
        いないので、数のはずの所に文字が、整数の所に 32 ビットを超える数が入りうる
        そのまま渡すと GL の呼び出しが例外を投げ、プレビューも書き出しも止まる

        画像は 2 番のユニットから順に繋ぐ 0 と 1 は ``u_texture`` と ``u_source``
        """
        unit = 2
        for spec in definition.parameters:
            value: ParamValue | None = effect.params.get(spec.name)
            if value is None:
                value = spec.default_value()

            if isinstance(spec, TrackSpec):
                program.set_float(
                    spec.name, spec.scaled_at(spec.coerce(value), frame, self.pixel_scale)
                )
            elif isinstance(spec, ColorSpec):
                color = spec.coerce(value)
                program.set_vec4(
                    spec.name,
                    (
                        srgb_to_linear(color[0]),
                        srgb_to_linear(color[1]),
                        srgb_to_linear(color[2]),
                        color[3],
                    ),
                )
            elif isinstance(spec, SelectSpec):
                program.set_int(spec.name, spec.index_of(spec.coerce(value)))
            elif isinstance(spec, CheckSpec):
                program.set_bool(spec.name, spec.coerce(value))
            elif isinstance(spec, ValueSpec):
                program.set_int(spec.name, spec.coerce(value))
            elif isinstance(spec, GridSpec):
                self._set_grid(program, spec, spec.coerce(value))
            elif isinstance(spec, GradientSpec):
                _set_gradient(program, spec, spec.coerce(value))
            elif isinstance(spec, FileSpec) and spec.texture:
                unit = self._bind_image(program, spec, spec.coerce(value), unit)

    def _set_grid(self, program: Program, spec: GridSpec, value: tuple[float, ...]) -> None:
        """格子の点を uniform へ 点数は**毎回**渡す

        uniform の値はプログラムに残るので、格子の無いクリップが、前に描いた
        別のクリップの点数を引き継いで、覚えのない歪み方をする

        点をスカラーの uniform にしないのは、9x9 で 162 個になるため
        ``vec2`` の配列 1 本なら、点数が増えても送り方が変わらない
        """
        columns, rows = spec.size(value)
        program.set_int(f"{spec.name}_columns", columns)
        program.set_int(f"{spec.name}_rows", rows)
        if columns < 2 or rows < 2:
            return
        scale = self.pixel_scale if spec.pixels else 1.0
        points = [
            (value[2 + index * 2] * scale, value[3 + index * 2] * scale)
            for index in range(columns * rows)
        ]
        program.set_vec2_array(f"{spec.name}_points", points)

    def _bind_image(self, program: Program, spec: FileSpec, path: str, unit: int) -> int:
        """画像を次の空いたテクスチャユニットへ繋ぐ 次に使うユニットを返す

        大きさは**毎回**渡す uniform の値はプログラムに残るので、画像を外した
        クリップが、前に描いた別のクリップの大きさを引き継いで空の画像を読む
        """
        texture = self._images.get(path)
        if texture is None:
            program.set_vec2(f"{spec.name}_size", (0.0, 0.0))
            return unit
        program.bind_texture(spec.name, texture.handle, unit=unit)
        # 大きさは画像の画素ではなく、画像を等倍で置いたときに占める合成の画素で渡す
        # 画像は画面の画素 1 つに画像の画素 1 つで重なる物なので、画質を落とした
        # プレビューでは画像も同じだけ縮めて置かないと、模様だけ 2 倍に出る
        program.set_vec2(
            f"{spec.name}_size",
            (float(texture.width) * self.pixel_scale, float(texture.height) * self.pixel_scale),
        )
        return unit + 1

    def retain_images(self, keep: Collection[str]) -> None:
        """``keep`` に無い画像を GPU から手放す GL が current な所で呼ぶこと"""
        self._images.retain(keep)

    def stale_images(self) -> frozenset[str]:
        """前に読んだときから書き換わった画像のパス GL は触らない"""
        return self._images.stale()

    def _copy(self, source: Framebuffer, target: Framebuffer) -> None:
        target.bind(clear=(0.0, 0.0, 0.0, 0.0))
        GL.glDisable(GL.GL_BLEND)
        self._blit.use()
        self._blit.set_bool("u_premultiplied", False)
        self._blit.set_bool("u_decode", False)
        self._blit.set_vec4("u_rect", FULL_RECT)
        self._blit.set_bool("u_flip", False)
        self._blit.bind_texture("u_texture", source.color)
        self._quad.draw()

    def _compile(self, effect: Effect) -> _Compiled | None:
        """エフェクトのシェーダを用意する 描画を伴わないものは ``None``

        コンパイルに失敗したエフェクトも ``None`` を覚えて、二度と試さない
        毎フレーム失敗し続けると、ログが埋まるうえに描画が止まる
        """
        if effect.kind in self._programs:
            return self._programs[effect.kind]

        definition = registry.get(effect.kind)
        compiled: _Compiled | None = None
        if definition is not None and definition.fragment_shader is not None:
            try:
                vertex = (
                    VERTEX_SHADER if definition.pieces is None else definition.pieces.vertex_shader
                )
                compiled = _Compiled(
                    definition=definition,
                    program=Program(vertex, definition.fragment_shader),
                )
            except ShaderError as error:
                # 1 度だけ残す 黙って捨てると、エフェクトが何も起きないまま
                # 「値の写し間違い」を探すことになる（実際に閃光で 1 度やった）
                _log.warning("エフェクト %s のシェーダを組めなかった: %s", effect.kind, error)
                compiled = None

        self._programs[effect.kind] = compiled
        return compiled


def _idle(effect: Effect) -> bool:
    """絵を何も変えない値のエフェクトか（既定のままの配置と反転）"""
    definition = registry.get(effect.kind)
    return definition is not None and definition.is_idle(effect)


def _number(
    definition: EffectDefinition, effect: Effect, name: str, frame: int, scale: float
) -> float:
    """エフェクトの数の項目を 1 つ、シェーダへ渡すのと同じ形で読む 読めなければ 0

    値の通し方は :meth:`EffectProcessor._set_parameters` と**同じにする**
    （``spec.coerce`` と :meth:`TrackSpec.scaled_at` を通す） 別の読み方をすると、
    シェーダへ渡る値と入れ物を広げる量が食い違い、後ろのエフェクトだけずれる
    """
    spec = definition.spec(name)
    if not isinstance(spec, TrackSpec):
        return 0.0
    value = effect.params.get(name)
    raw = spec.default_value() if value is None else value
    return float(spec.scaled_at(spec.coerce(raw), frame, scale))


def _choice(definition: EffectDefinition, effect: Effect, name: str) -> str:
    """選択肢の項目を 1 つ読む 項目が無ければ空（支点なら絵の中央の扱い）"""
    spec = definition.spec(name)
    if not isinstance(spec, SelectSpec):
        return ""
    value = spec.coerce(effect.params.get(name))
    return value if isinstance(value, str) else ""


def piece_grid(
    content: tuple[float, float, float, float], cell: float, width: int, height: int
) -> tuple[tuple[int, int], int, int]:
    """中身の範囲（左・下・右・上）に掛かる升目 左下の升目の番号・横の数・縦の数を返す

    欠片は元の絵の中身の範囲にしか無い そこに掛かる升目だけを描けば、散った後でも
    元の絵を割った数より多くは描かない u_object ではなく u_content で切る
    前の変形で広げた絵は u_object の外まである（#199）
    バッファの外でも切る 外の画素は元から無い（読めば透明）ので描いても何も出ず、
    画面より大きく置いた絵で四角の数だけが膨らむ

    右と上の端は、端に少しでも掛かる升目まで数える 画質を落とすと升目の一辺も中身の端も
    半端な数になり、端の画素の中心だけで数えると、中身の残る最後の升目が丸ごと抜ける
    """
    left, bottom, right, top = content
    left, bottom = max(left, 0.0), max(bottom, 0.0)
    right, top = min(right, float(width)), min(top, float(height))
    first = (math.floor(left / cell), math.floor(bottom / cell))
    columns = max(math.ceil(right / cell) - first[0], 0)
    rows = max(math.ceil(top / cell) - first[1], 0)
    return first, columns, rows


def _piece_size(definition: EffectDefinition, effect: Effect, frame: int, scale: float) -> float:
    """升目の一辺（合成の画素） 下限（画面の画素）より小さくしない

    下限も画面の画素なので縮めてから比べる 縮めないと、画質を落としたプレビューでだけ
    細かい欠片が下限で太り、書き出しと割れ方が変わる 数でない値（NaN・無限）は下限に
    寄せる そのまま割ると升目の数が求まらず、描く数が壊れる
    """
    pieces = definition.pieces
    assert pieces is not None
    minimum = pieces.minimum * scale
    size = _number(definition, effect, pieces.size, frame, scale)
    return size if math.isfinite(size) and size > minimum else minimum


_BLIT_FRAGMENT = (
    """
#version 430 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
uniform bool u_premultiplied;
// 渡された絵が sRGB で符号化した値か（sRGB で重ねた合成先のキャンバス）
uniform bool u_decode;
"""
    + AREA_SAMPLING
    + """
void main() {
    // 素材を枠へ収めて縮めるときも、変形のエフェクトと同じく事前乗算で平均する
    // GL の補間のままだと、透明な所に残った色が縁へにじむ（#179）
    vec4 color = area_premul(u_texture, v_uv, dFdx(v_uv), dFdy(v_uv), u_premultiplied);
    color = color.a > 0.0001 ? vec4(color.rgb / color.a, color.a) : vec4(0.0);
    if (u_decode) {
        vec3 c = clamp(color.rgb, 0.0, 1.0);
        color.rgb = mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
    }
    frag_color = color;
}
"""
)
