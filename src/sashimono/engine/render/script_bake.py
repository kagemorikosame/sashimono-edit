"""スクリプトが ``obj.effect`` で積んだ効果を、その場で絵へ掛ける（Issue #176）

AviUtl の ``obj.effect`` は呼んだ所で絵を変える こちらの効果は GPU のシェーダなので、
ふだんは積んでおいて描くときにまとめて掛ける 間に ``obj.effect("リサイズ")`` や
``obj.copybuffer`` のような絵を読む・変える呼び出しが挟まったときだけ、ここで 1 枚の
絵に掛けて読み戻す（:meth:`~sashimono.compat.aviutl.objapi.ObjApi._settle_effects`）

絵は**画面の画素**のまま扱う スクリプトは画質に関わらずプロジェクトの解像度で動くので
（:meth:`FrameRenderer._draw_scripted`）、画素で決める設定も等倍で掛ける
"""

from __future__ import annotations

import math

import numpy as np

from sashimono.compat.aviutl.report import CompatibilityReport, global_report
from sashimono.core.model import Effect
from sashimono.effects.strokes import pixel_reach
from sashimono.engine.gpu import Compositor, EffectProcessor, Placement, Texture

__all__ = ["BAKE_CANVAS_LIMIT", "BAKE_MARGIN", "ScriptEffectBaker", "bake_margin", "fitted_margin"]

#: 絵の周りに空ける余白の下限（画素） ぼかしや影は絵の外へ広がる AviUtl のぼかしも絵を
#: 広げるので、広がった所まで残す 画素で決める項目から読める分は :func:`bake_margin` が
#: 足す ここは項目から読めない広がり（光の尾など）のための幅で、ぼかし・グローの範囲の
#: 上限 96 が収まる
BAKE_MARGIN = 128

#: 絵と余白を合わせた作業場の一辺の上限（画素） 作業場は RGBA16F のバッファを何枚も
#: 同時に持つ 絵の上限 4096 に同じだけの余白を足した 12288 画素四方では、1 回で数 GB になり
#: GPU のメモリが尽きる 6144 なら 1 枚 300MB 足らずで、4096 の絵にも両側 1024 の余白が残る
BAKE_CANVAS_LIMIT = 6144


class ScriptEffectBaker:
    """効果を掛けて読み戻す係 GL のコンテキストが current な所で使う

    合成先と効果の作業場は、描画の物とは別に持つ 大きさが絵ごとに違ううえ、
    描画の作業場を使うと、描いている途中の絵を壊す 使うまで作らない
    """

    def __init__(self) -> None:
        self._compositor: Compositor | None = None
        self._effects: EffectProcessor | None = None
        self._texture: Texture | None = None
        #: GPU が作れるテクスチャとレンダーターゲットの一辺の上限（画素） 初めて使うときに GL から
        #: 読む 試験は小さい値を入れて、上限を超えたときの動きを確かめる
        self.gpu_limit: int | None = None

    def apply(
        self,
        image: np.ndarray,
        effects: tuple[Effect, ...],
        frame: int,
        fps: float,
        duration: int,
    ) -> np.ndarray | None:
        """``image`` へ ``effects`` を掛けた絵 広がった所まで含め、真ん中は動かさない

        作業場が GPU の作れる大きさを超えるときは掛けずに ``None`` を返す

        戻す絵は、余白のうち透明なままの所を左右（上下）同じ幅だけ削った物
        同じ幅で削るのは、絵の真ん中がオブジェクトの位置だから 片側だけ削ると、
        効果を掛けただけで絵が横へずれる 元の絵の内側までは削らない 透明な縁を持つ絵が、
        効果を掛けただけで小さくなる
        """
        height, width = image.shape[:2]
        if max(width, height) > BAKE_CANVAS_LIMIT:
            # 絵そのものが上限を超える（8192 のプロジェクトの背景の図形など）と、余白を 0 にしても
            # 上限より大きいバッファを何枚も作る GPU の上限が大きい機械ではそのまま作り、メモリが
            # 尽きて例外が描画まで伝わる 焼き込まずに返し、効果は積んだまま描くときに掛ける
            global_report.note_missing(
                f"obj.effect の焼き込み（絵 {width}x{height} が作業場の上限"
                f" {BAKE_CANVAS_LIMIT} を超える）"
            )
            return None
        # 余白は効果が絵を運ぶ量から決める 決め打ちにすると、それより遠くへずらす影が
        # 作業場の外へ出て消え、obj.w や写し取った絵からも消える（#186）
        margin = fitted_margin(width, height, bake_margin(effects, frame))
        canvas_w, canvas_h = width + 2 * margin, height + 2 * margin
        if max(canvas_w, canvas_h) > self._gpu_limit():
            # 作れない大きさのフレームバッファを作ろうとすると例外が描画まで伝わり、フレームごと
            # 描けなくなる 焼き込まずに返し、効果は積んだまま描くときに掛ける（順は入れ替わる）
            global_report.note_missing(
                f"obj.effect の焼き込み（作業場 {canvas_w}x{canvas_h} が GPU の上限"
                f" {self._gpu_limit()} を超える）"
            )
            return None
        compositor, processor, texture = self._prepared(canvas_w, canvas_h)
        if not processor.has_work(effects):
            return image

        texture.upload(image)
        placed = Placement(float(margin), float(margin), float(width), float(height))
        result = processor.apply(
            texture,
            effects,
            frame=frame,
            fps=fps,
            source_rect=placed.to_clip(canvas_w, canvas_h),
            duration=duration,
            bounds=(
                float(margin),
                float(margin),
                float(margin + width),
                float(margin + height),
            ),
        )
        compositor.begin((0.0, 0.0, 0.0, 0.0))
        compositor.draw_handle(
            result.color,
            Placement(0.0, 0.0, float(canvas_w), float(canvas_h)),
            flip=False,
        )
        # スクリプトの絵はストレートアルファ 事前乗算のまま返すと、描くときに不透明度が
        # もう 1 度掛かって、ぼけた縁が暗くなる
        baked = compositor.read(straight=True)
        return _trimmed(baked, margin)

    def apply_within(
        self,
        image: np.ndarray,
        effects: tuple[Effect, ...],
        *,
        frame: int,
        fps: float,
        duration: int,
        pixel_scale: float = 1.0,
    ) -> np.ndarray:
        """``image`` と同じ大きさのまま ``effects`` を掛けた絵 縁取りの層の絵に使う（#273）

        層の絵は字の絵と同じ大きさで、広がる分は字の絵の大きさの見積もり（``source_canvas``）が
        もう入れてある 大きさを変えて返すと、重ねる位置が層ごとにずれる
        周りには広がる分だけ余白を足して掛け、真ん中を切り出す 余白無しで掛けると、絵の端に
        届いたぼかしが端の色を引き伸ばして濃くなる

        ``pixel_scale`` は画質を落としたプレビューの縮め方 層の絵は合成の画素で描いてあるので、
        画素で決める設定（ぼかしの範囲など）も同じだけ縮めないとプレビューだけ 2 倍にぼける

        掛けるのは縁のある所の周りだけ（広がる分を足した四角） 字の絵は画面の大きさか、流れる
        テロップでは一辺 8000 近くになり、全体を作業場にすると字幕 1 本ごとに画面何枚ぶんも
        GPU を通す 作業場が GPU の作れる大きさを超えるときは、掛けずにそのまま返す
        （互換性レポートに残す）
        """
        height, width = image.shape[:2]
        alpha = image[..., 3] > 0
        rows = np.flatnonzero(alpha.any(axis=1))
        if rows.size == 0:
            return image
        columns = np.flatnonzero(alpha.any(axis=0))
        reach = math.ceil(pixel_reach(effects, frame) * pixel_scale) + 2
        top, bottom = max(0, int(rows[0]) - reach), min(height, int(rows[-1]) + 1 + reach)
        left, right = max(0, int(columns[0]) - reach), min(width, int(columns[-1]) + 1 + reach)
        piece = np.ascontiguousarray(image[top:bottom, left:right])
        piece_h, piece_w = piece.shape[:2]
        margin = fitted_margin(piece_w, piece_h, reach)
        canvas_w, canvas_h = piece_w + 2 * margin, piece_h + 2 * margin
        if max(piece_w, piece_h) > BAKE_CANVAS_LIMIT or max(canvas_w, canvas_h) > self._gpu_limit():
            global_report.note_missing(
                f"縁取りの層のエフェクト（絵 {piece_w}x{piece_h} が作業場の上限を超える）"
            )
            return image
        compositor, processor, texture = self._prepared(canvas_w, canvas_h)
        if not processor.has_work(effects):
            return image
        texture.upload(piece)
        placed = Placement(float(margin), float(margin), float(piece_w), float(piece_h))
        processor.pixel_scale = pixel_scale
        try:
            result = processor.apply(
                texture,
                effects,
                frame=frame,
                fps=fps,
                source_rect=placed.to_clip(canvas_w, canvas_h),
                duration=duration,
                # 層の絵の中身のある所を入れ物にする グラデーションや角丸が縁の広がりに合う
                bounds=(
                    float(margin - left + int(columns[0])),
                    float(margin - top + int(rows[0])),
                    float(margin - left + int(columns[-1]) + 1),
                    float(margin - top + int(rows[-1]) + 1),
                ),
            )
        finally:
            # スクリプトの焼き込み（:meth:`apply`）は画面の画素で掛ける 戻さないと、画質を
            # 落としたプレビューの後に走ったスクリプトの効果だけ小さく掛かる
            processor.pixel_scale = 1.0
        compositor.begin((0.0, 0.0, 0.0, 0.0))
        compositor.draw_handle(
            result.color, Placement(0.0, 0.0, float(canvas_w), float(canvas_h)), flip=False
        )
        baked = compositor.read(straight=True)
        # 切り出した四角の外は層の絵でも透明 広がる分は四角に入れてあるので、外へは出ない
        whole = np.zeros_like(image)
        whole[top:bottom, left:right] = baked[margin : margin + piece_h, margin : margin + piece_w]
        return whole

    def _gpu_limit(self) -> int:
        """テクスチャとレンダーターゲットの両方で作れる一辺の上限 GL のコンテキストの中で読む"""
        if self.gpu_limit is None:
            from OpenGL.GL import GL_MAX_RENDERBUFFER_SIZE, GL_MAX_TEXTURE_SIZE, glGetIntegerv

            self.gpu_limit = int(
                min(glGetIntegerv(GL_MAX_TEXTURE_SIZE), glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE))
            )
        return self.gpu_limit

    def release(self) -> None:
        if self._compositor is not None:
            self._compositor.release()
        if self._effects is not None:
            self._effects.release()
        if self._texture is not None:
            self._texture.release()
        self._compositor = self._effects = self._texture = None

    def _prepared(self, width: int, height: int) -> tuple[Compositor, EffectProcessor, Texture]:
        if self._compositor is None or self._effects is None or self._texture is None:
            self._compositor = Compositor(width, height)
            self._effects = EffectProcessor(width, height, self._compositor.quad)
            self._texture = Texture(1, 1)
        elif (self._compositor.width, self._compositor.height) != (width, height):
            self._compositor.resize(width, height)
            self._effects.resize(width, height)
        return self._compositor, self._effects, self._texture


def _trimmed(image: np.ndarray, margin: int) -> np.ndarray:
    """余白の透明な所を、左右・上下それぞれ同じ幅だけ削る ``margin`` より多くは削らない"""
    alpha = image[..., 3] > 0
    height, width = alpha.shape
    rows = np.flatnonzero(alpha.any(axis=1))
    columns = np.flatnonzero(alpha.any(axis=0))
    if rows.size == 0:
        # 何も残らない絵 元の大きさの透明な絵にする 大きさまで消すと ``obj.w`` が 0 になる
        return np.ascontiguousarray(image[margin : height - margin, margin : width - margin])
    top = min(int(rows[0]), height - 1 - int(rows[-1]), margin)
    left = min(int(columns[0]), width - 1 - int(columns[-1]), margin)
    return np.ascontiguousarray(image[top : height - top, left : width - left])


def bake_margin(effects: tuple[Effect, ...], frame: int) -> int:
    """``effects`` を掛けるときに絵の周りへ空ける余白（画素）

    効果が絵を外へ動かしうる量を足し合わせる 画素で決める項目（影のずれ・ぼかしの範囲・
    縁取りの太さなど）はどれもその値より遠くへは絵を運ばない 入れ物を広げる効果
    （``expands_object``）は広げる量のうち大きい方 順に掛かるので、効果ごとの量を足す

    どの効果の項目か分からない物（範囲を画素で持たない光など）のために、
    :data:`BAKE_MARGIN` より狭くはしない ここでは上限で丸めない 丸めると、足りない余白で
    掛けたことが分からなくなる 作業場に収まるかは :func:`fitted_margin` が見る
    量の数え方は縁取りの層の絵の広がり（:func:`~sashimono.effects.strokes.pixel_reach`）と同じ
    """
    return max(BAKE_MARGIN, math.ceil(pixel_reach(effects, frame)))


def fitted_margin(
    width: int, height: int, needed: int, report: CompatibilityReport | None = None
) -> int:
    """``width`` x ``height`` の絵に ``needed`` の余白を付けて、作業場が上限に収まる余白

    収まらなければ上限まで縮め、縮めたことを互換性レポートに残す 黙って縮めると、
    遠くへ動かす効果の外側が欠けた理由が分からない 絵そのものが上限を超えていれば余白は 0
    """
    room = max(0, (BAKE_CANVAS_LIMIT - max(width, height)) // 2)
    if needed <= room:
        return needed
    (report if report is not None else global_report).note_missing(
        f"obj.effect の焼き込みの余白 {needed} 画素（作業場の上限 {BAKE_CANVAS_LIMIT} に"
        f"収めるため {room} 画素にした）"
    )
    return room
