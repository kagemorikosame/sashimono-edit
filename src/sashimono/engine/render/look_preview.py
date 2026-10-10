"""プリセットとエイリアスの見本の絵（#277）

保存した時の画面を写すのではなく、中身から**置いて描く** 保存した時の絵を持つと、
前の版で保存した物（絵を持たない）だけ見本が出ず、描き方を直しても古い絵のまま残る
中身から描けば、どの版で保存した物も同じ見本になり、絵の鍵は中身の指紋
（:func:`~sashimono.core.io.library.look_fingerprint`）で済む

描き方は 2 つ

* **GPU**（既定） 書き出しと同じ :class:`~sashimono.engine.render.FrameRenderer` で、
  1920x1080 の空のプロジェクトに置いて 1/2 の画質で描く 縁取り・グロー・グラデーション
  などのエフェクトも、当てたときと同じ絵で出る
* **簡易** テキストと図形の中身だけを CPU で描く（テンプレートの棚の下絵と同じ） エフェクトは
  出ないので、見る側（管理の窓）は「エフェクトは出ていない」と書く GL の使えない機械と、
  設定で軽い方を選んだときに使う

どちらも絵の中の物の周りを切り出して返す 1920x1080 の画面のまま縮めると、字幕の
ような小さな物が見本の中で点になる 切り出す大きさには下限があり（画面の幅の 1/3）、
小さな物を画面いっぱいまで引き伸ばさない（実際の大きさの感じが分からなくなる）

描くのはクリップの真ん中のコマ 登場の動きは頭、退場の動きはお尻にあることが多く、
真ん中なら動きの途中ではない、止まった見た目が出やすい
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QCoreApplication, QThread, Signal

from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.commands.insert import DEFAULT_GENERATED_FRAMES, insert_clip
from sashimono.core.commands.preset import PresetOptions, preset_commands
from sashimono.core.io.aliases import Alias
from sashimono.core.io.presets import Preset
from sashimono.core.model import Clip, GeneratedSource, Project, Track, TrackKind
from sashimono.effects.sources import TEXT

if TYPE_CHECKING:
    from sashimono.engine.gpu import OffscreenGLContext
    from sashimono.engine.gpu.context import GLScope

__all__ = [
    "CANVAS",
    "MEASURED_LOOK_MS",
    "SAMPLE_TEXT",
    "LookPicture",
    "LookRenderer",
    "LookWorker",
    "crop_to_content",
    "sample_project",
]

#: 置いて描く画面の大きさ 文字の大きさや位置は 1080p を前提に決められている
CANVAS = (1920, 1080)
#: GPU で描くときの画質（分母） 見本は 256x144 ほどで見るので等倍は要らない
#: 1/4 まで落とすと、細い縁取りが 1 画素に満たず淡くなる（#173）
_QUALITY_DIVISOR = 2
#: 文字を持たない見た目（エフェクトだけのプリセット）を当てる見本の文字
#: ひらがな・カタカナ・英字を並べ、書体の違いが分かるようにする
SAMPLE_TEXT = "あア Aa"
#: 切り出す大きさの下限（画面の幅に対する割合）
_MIN_CROP = 1 / 3
#: 物の周りに残す余白（切り出す幅に対する割合）
_MARGIN = 0.08
#: 見えていると数える不透明度（0〜255） 影のぼかしの裾まで数えると、切り出しが広がりすぎる
_VISIBLE = 8

Item = Preset | Alias

#: 見本 1 枚を描くのにかかった時間（ミリ秒） GPU（描画係を作った後）と簡易の描き方
#: 設定の画面に出す RTX 5060 Ti で、グローを積んだテキストのプリセット 30 本の中央値
#: GPU の描画係を最初に作る 440ms は別のスレッド（:class:`LookWorker`）で待つので、
#: 一覧を開く速さには響かない 簡易の方が軽いのは、描画係を作る分と GPU のメモリの方
#: 測り直したらここだけ直す
MEASURED_LOOK_MS = (8.3, 8.3)


@dataclass(frozen=True, slots=True)
class LookPicture:
    """描いた見本 ``image`` はストレートアルファの RGBA（高さ, 幅, 4）"""

    image: np.ndarray
    #: エフェクトを描いていない（簡易の描き方） 見る側はその旨を書く
    simple: bool
    #: 見える物が何も無かった（全部透明） 見る側は空の地だけを出さずに一言添える
    empty: bool


def sample_project(item: Item) -> tuple[Project, int]:
    """``item`` を置いたプロジェクトと、描くコマ

    プリセットは文字と位置も当てる（設定の当て方に関係なく） 見本は保存した見た目
    そのものを見せる物で、当てる先の事情は無い 文字を持たないプリセットは見本の文字に当てる
    """
    project = Project.create()
    if isinstance(item, Alias):
        clip = item.instantiate(0)
        for command in insert_clip(project, clip, at_frame=0):
            project = command.apply(project)
        return project, clip.duration // 2

    source = item.source if item.source is not None else TEXT.create(text=SAMPLE_TEXT)
    duration = item.span or DEFAULT_GENERATED_FRAMES
    clip = with_fixed_items(Clip(timeline_start=0, duration=duration, source=source), picture=True)
    track = Track(TrackKind.VIDEO, "V1", (clip,))
    project = project.with_timeline(replace(project.timeline, tracks=(track,)))
    options = PresetOptions(with_text=True, with_position=True)
    for command in preset_commands(item, clip, options=options):
        project = command.apply(project)
    return project, duration // 2


def crop_to_content(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """見える物の周りを 16:9 で切り出す 見える物が無ければ全体と真を返す"""
    height, width = image.shape[:2]
    alpha = image[:, :, 3]
    rows = np.flatnonzero(alpha.max(axis=1) >= _VISIBLE)
    columns = np.flatnonzero(alpha.max(axis=0) >= _VISIBLE)
    if rows.size == 0 or columns.size == 0:
        return image, True
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    aspect = width / height
    crop_width = max(right - left, (bottom - top) * aspect, width * _MIN_CROP)
    crop_width = min(width, crop_width * (1 + 2 * _MARGIN))
    crop_height = min(height, crop_width / aspect)
    crop_width = crop_height * aspect
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    x0 = round(min(max(center_x - crop_width / 2, 0), width - crop_width))
    y0 = round(min(max(center_y - crop_height / 2, 0), height - crop_height))
    x1 = min(width, x0 + round(crop_width))
    y1 = min(height, y0 + round(crop_height))
    return np.ascontiguousarray(image[y0:y1, x0:x1]), False


class LookRenderer:
    """見本を描く係 GPU の描画係を 1 つだけ作って使い回す

    描画係はシェーダや合成先を持ち、作るのに時間がかかる（RTX 5060 Ti で 1 つ 440ms
    1 枚描くのは 8ms） 一覧を開いて何十枚も描くときに毎回作ると、見本が出そろうまでが
    何十倍にも延びる GL のコンテキストを持つので、作ったスレッドだけで使うこと
    ``context`` は描画係に使わせるコンテキスト（:class:`LookWorker` が渡す） 省くと自前で作る
    """

    def __init__(self, *, gpu: bool = True, context: GLScope | None = None) -> None:
        self._gpu = gpu
        self._context = context
        # 型は描画係 読み込みに GL の部品が要るので、使うまで import しない
        self._renderer: object | None = None

    @property
    def gpu(self) -> bool:
        return self._gpu

    def render(self, item: Item) -> LookPicture:
        if self._gpu:
            image = self._render_gpu(item)
            cropped, empty = crop_to_content(image)
            return LookPicture(cropped, simple=False, empty=empty)
        image = _render_simple(item)
        cropped, empty = crop_to_content(image)
        return LookPicture(cropped, simple=True, empty=empty)

    def close(self) -> None:
        renderer = self._renderer
        self._renderer = None
        if renderer is not None:
            from sashimono.engine.render.renderer import FrameRenderer

            assert isinstance(renderer, FrameRenderer)
            renderer.close()

    def _render_gpu(self, item: Item) -> np.ndarray:
        from sashimono.engine.render.renderer import FrameRenderer, RenderQuality

        project, frame = sample_project(item)
        renderer = self._renderer
        if renderer is None:
            renderer = FrameRenderer(
                project, context=self._context, quality=RenderQuality(_QUALITY_DIVISOR)
            )
            self._renderer = renderer
        assert isinstance(renderer, FrameRenderer)
        renderer.set_project(project)
        return renderer.render(frame, transparent=True)


def _render_simple(item: Item) -> np.ndarray:
    """中身だけを CPU で描く エフェクト・配置の欄・不透明度は描かない"""
    from sashimono.engine.sources import render_source

    source: GeneratedSource | None
    duration = DEFAULT_GENERATED_FRAMES
    if isinstance(item, Alias):
        source = item.clip.source
        duration = item.clip.duration
    else:
        source = item.source
        duration = item.span or duration
    if source is None:
        source = TEXT.create(text=SAMPLE_TEXT)
    image = render_source(source, *CANVAS, frame=duration // 2, duration=duration)
    if image is None:
        return np.zeros((CANVAS[1], CANVAS[0], 4), dtype=np.uint8)
    return image


#: 走り係へ渡す頼み（絵の鍵, 描く物） ``None`` は止まれの合図
_Request = tuple[str, Item] | None


class LookWorker(QThread):
    """見本を別のスレッドで描く走り係

    コンテキストを作る 280ms・描画係を作る 440ms・1 枚ずつの描画を、画面のスレッドから
    外す 一覧を開いた瞬間に画面が止まると、件数が多いほど開くのが遅く見える 描いた物は
    :attr:`drawn` で画面のスレッドへ返す（Qt が鍵の付いた順で渡す）

    サーフェスだけは画面のスレッドで作る決まりなので、ここで作ってからコンテキストごと
    このスレッドへ移し、コンテキストはこのスレッドで作る（``deferred``）
    止めた後は :meth:`stop` が画面のスレッドへ戻して捨てる
    """

    #: 描き終えた（鍵, :class:`LookPicture` か描けなかった理由の文）
    drawn = Signal(str, object)
    #: GPU のコンテキストを作れなかった（理由） 以後は簡易の描き方で描く
    gpu_failed = Signal(str)

    def __init__(self, *, gpu: bool) -> None:
        super().__init__()
        self.setObjectName("sashimono-look-preview")
        self._requests: queue.SimpleQueue[_Request] = queue.SimpleQueue()
        self._context: OffscreenGLContext | None = None
        if gpu:
            from sashimono.engine.gpu import OffscreenGLContext

            self._context = OffscreenGLContext(deferred=True)
            self._context.move_to_thread(self)

    def submit(self, key: str, item: Item) -> None:
        self._requests.put((key, item))

    def stop(self) -> None:
        """描きかけの 1 枚を終えて止まる 頼んだまま描いていない物は捨てる"""
        self._requests.put(None)
        self.wait()
        if self._context is not None:
            self._context.release()
            self._context = None

    def run(self) -> None:
        context = self._context
        entered = False
        if context is not None:
            from sashimono.engine.gpu import GLContextError

            try:
                context.complete()
                # 走っている間ずっと current にしておく 1 枚ごとに付け外しすると、
                # そのたびにドライバの切り替えが入る
                context.__enter__()
                entered = True
            except GLContextError as exc:
                # GPU の無い機械 黙って空の見本を並べず、簡易の描き方に替えて知らせる
                self.gpu_failed.emit(str(exc))
        renderer = LookRenderer(gpu=entered, context=context if entered else None)
        try:
            while True:
                request = self._requests.get()
                if request is None:
                    return
                key, item = request
                try:
                    picture: LookPicture | str = renderer.render(item)
                except Exception as exc:
                    # 1 枚が描けなくても走り係は止めない 知らないエフェクトを持つ 1 件の
                    # せいで、ほかの見本まで出なくなる
                    picture = f"見本を描けなかった: {exc}"
                self.drawn.emit(key, picture)
        finally:
            try:
                renderer.close()
            finally:
                if entered and context is not None:
                    context.__exit__(None, None, None)
                application = QCoreApplication.instance()
                if context is not None and application is not None:
                    # 画面のスレッドへ返す 返さないと、捨てるときに別のスレッドに
                    # 付いたままのコンテキストを触る
                    context.move_to_thread(application.thread())
