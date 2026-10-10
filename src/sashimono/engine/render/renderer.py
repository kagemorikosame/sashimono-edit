"""プロジェクトの 1 フレームを合成する

プレビューも書き出しもここを通る 同じ経路を使うことで「プレビューでは出るのに
書き出すと出ない」が構造的に起きない
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Collection, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from OpenGL import GL
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from sashimono.compat.aviutl.embedded import has_embedded
from sashimono.compat.aviutl.objapi import DrawCall
from sashimono.compat.aviutl.report import global_report
from sashimono.core.commands.fixed import PICTURE_FIXED
from sashimono.core.model import (
    AnimatedValue,
    Blending,
    Clip,
    ClipId,
    Effect,
    GeneratedSource,
    MediaId,
    MediaItem,
    ParamValue,
    Project,
    Timeline,
    Track,
    TrackId,
    controlling_groups,
    group_as_one,
    group_reaches,
)
from sashimono.core.timebase import FrameRate, seconds_to_frame
from sashimono.effects.easing import ease
from sashimono.effects.sources import PREVIOUS_OBJECT
from sashimono.engine.audio_shapes import spectrum_window
from sashimono.engine.cache.proxy import ProxyStore
from sashimono.engine.decode import AudioDecoder, ProbeError, VideoDecoder
from sashimono.engine.gpu import (
    BlendMode,
    Compositor,
    Corners,
    EffectProcessor,
    Framebuffer,
    GLScope,
    OffscreenGLContext,
    Placement,
    Texture,
    Transform,
    fit_placement,
)
from sashimono.engine.gpu.projection import project
from sashimono.engine.motion_shapes import TrailPaths
from sashimono.engine.render.groups import grouped
from sashimono.engine.render.invalidate import image_paths
from sashimono.engine.render.outline import canvas_scale, is_generated, media_pixel_size
from sashimono.engine.render.script_bake import ScriptEffectBaker
from sashimono.engine.render.scripts import (
    ScriptStage,
    is_scene_change,
    requested_effects,
    script_catalog,
    script_effects,
    split_effects,
    text_font,
)
from sashimono.engine.sources import (
    MAX_CANVAS,
    Frame,
    corner_points_centred,
    render_source_framed,
    source_canvas,
)

__all__ = ["FrameRenderer", "RenderQuality"]


#: RGBA の 4 バイトをリトルエンディアンの 32bit 1 つとして読んだとき、α が 1 以上なら
#: この値以上になる（α が一番上の桁に来る） 並びの向きを型に書いてあるので、機械の
#: 向きに関係なく同じ値になる
_INKED = 1 << 24

#: 色の付いた範囲（画素、左・上・右・下 どれも整数）
_Box = tuple[int, int, int, int]


def _packed(image: np.ndarray) -> np.ndarray | None:
    """RGBA uint8 の絵を、1 画素 1 つの 32bit の 2 次元配列として見る 写しは作らない

    見られない並び（uint8 でない・4 つ組でない・1 画素の中が飛び飛び）は ``None``
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        return None
    if image.strides[2] != 1 or image.strides[1] != 4:
        return None
    try:
        return image.view(np.dtype("<u4"))[..., 0]
    except ValueError:
        return None


def _content_box(image: np.ndarray) -> _Box | None:
    """絵の中で色が付いている範囲（画素、左・上・右・下） 何も無ければ ``None``

    テキストや図形は画面と同じ大きさの絵で届く 角丸や中心基準の動きは、この範囲を
    絵の大きさとして扱わないと、画面の角や画面の中央を基準にしてしまう

    エフェクトを積んだクリップは描くたびにここを通るので、絵の全体を α だけ飛び飛びに
    読む素直な書き方では 4K で 1 枚 10ms 掛かっていた（Issue #128 3 枚重ねで 30ms）
    答えは素直な書き方と同じまま、読む量を減らす

    - 縁の 1 周を先に見る 上と下の行に色があれば上下は絵の端、左と右の列に色が
      あれば左右は絵の端と決まり、その向きは探さなくてよい 動画のような不透明な絵は
      4 辺とも決まってここで終わる（4K で 0.02ms）
    - 決まらない向きは全体を読むしかない 色の無い画素がどこにあっても答えが変わりうる
      1 画素を 32bit 1 つとして読み、行ごとの最大で色のある行を探してから、その行の間
      だけで列を探す α を 1 バイトずつ飛び飛びに読むより 3〜8 倍速い
    """
    alpha = image[..., 3]
    height, width = int(alpha.shape[0]), int(alpha.shape[1])
    if height == 0 or width == 0:
        return None
    rows_known = bool(alpha[0].any() and alpha[-1].any())
    columns_known = bool(alpha[:, 0].any() and alpha[:, -1].any())
    if rows_known and columns_known:
        return 0, 0, width, height
    pixels = _packed(image)
    if rows_known:
        top, bottom = 0, height
    else:
        if pixels is None:
            rows = np.flatnonzero(alpha.any(axis=1))
        else:
            rows = np.flatnonzero(pixels.max(axis=1) >= _INKED)
        if rows.size == 0:
            return None
        top, bottom = int(rows[0]), int(rows[-1]) + 1
    if columns_known:
        return 0, top, width, bottom
    # 色のある行はどれも top から bottom の間にあるので、列はその間だけ見れば足りる
    if pixels is None:
        columns = np.flatnonzero(alpha[top:bottom].any(axis=0))
    else:
        columns = np.flatnonzero(pixels[top:bottom].max(axis=0) >= _INKED)
    return int(columns[0]), top, int(columns[-1]) + 1, bottom


def _object_box(image: np.ndarray, framed: Frame | None = None) -> Frame | None:
    """オブジェクトの入れ物（画素、左・上・右・下）

    ふつうは色の付いた範囲 文字の枠を持つテキスト（AviUtl2 の組み方）は、その枠と
    色の付いた範囲を合わせた範囲にする AviUtl2 のテキストの入れ物は送り幅 x 行の高さの
    枠で、字の形より上下左右に 20 画素ほど広い 字の形で代わりにすると、画像合成が出る
    範囲・万華鏡の中心・分割したマスの動く量がその分ずれる 字が枠からはみ出すとき
    （斜体の張り出しや縁取り）に切れないよう、色の付いた範囲も含める
    """
    return _merged_box(_content_box(image), framed)


def _merged_box(box: _Box | None, framed: Frame | None) -> Frame | None:
    """色の付いた範囲と文字の枠を合わせた入れ物 決め方は :func:`_object_box`"""
    if framed is None:
        return None if box is None else (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if box is None:
        return framed
    return (
        min(framed[0], box[0]),
        min(framed[1], box[1]),
        max(framed[2], box[2]),
        max(framed[3], box[3]),
    )


def _placed_bounds(
    box: tuple[float, float, float, float] | None,
    image: np.ndarray | None,
    placement: Placement,
    image_width: int = 0,
    image_height: int = 0,
) -> tuple[float, float, float, float] | None:
    """絵の中の範囲を、置いた先（画面の画素）の範囲へ写す"""
    if box is None:
        return None
    if image is not None:
        image_height, image_width = int(image.shape[0]), int(image.shape[1])
    scale_x = placement.width / max(image_width, 1)
    scale_y = placement.height / max(image_height, 1)
    left, top, right, bottom = box
    return (
        placement.left + left * scale_x,
        placement.top + top * scale_y,
        placement.left + right * scale_x,
        placement.top + bottom * scale_y,
    )


def _object_sized(image: np.ndarray, found: Frame | None) -> tuple[np.ndarray, tuple[float, float]]:
    """入れ物 ``found``（:func:`_object_box`）だけを切り出した絵と、画面の中心からのずれ
    （画素、Y は下が正） 入れ物が無ければ元の絵

    AviUtl のスクリプトは ``obj.w`` ``obj.h`` をオブジェクト自身の大きさとして読む
    画面と同じ大きさの絵を渡すと、画面の幅で位置を計算してしまう

    入れ物は呼ぶ側で求めて渡す 作った絵の入れ物はレンダラが覚えているので、
    ここで探し直すと毎フレーム絵の全体を読むことになる（Issue #128）
    """
    if found is None:
        return image, (0.0, 0.0)
    height, width = int(image.shape[0]), int(image.shape[1])
    # 枠は小数で届く 内側へ丸めると枠の端の字が欠けるので外側へ広げ、絵の中に収める
    left = max(0, math.floor(found[0]))
    top = max(0, math.floor(found[1]))
    right = min(width, math.ceil(found[2]))
    bottom = min(height, math.ceil(found[3]))
    if right <= left or bottom <= top:
        return image, (0.0, 0.0)
    if (left, top, right, bottom) == (0, 0, width, height):
        return image, (0.0, 0.0)
    offset = ((left + right) / 2.0 - width / 2.0, (top + bottom) / 2.0 - height / 2.0)
    return image[top:bottom, left:right], offset


def _resized(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """RGBA uint8 の絵を ``size``（幅, 高さ）へ引き伸ばす 画質を落とした合成の絵を、
    スクリプトへ画面の画素の絵として渡すため（:meth:`FrameRenderer._draw_scripted`）

    滑らかには伸ばさず、升目を並べるだけにする スクリプトの後で同じ所へ縮め戻すので、
    分母の升目がそろっていれば元の画素へそのまま戻る 滑らかに伸ばすと、行きと帰りで
    2 回補間され、絵に何もしないスクリプトでも下の絵がぼける
    """
    height, width = int(image.shape[0]), int(image.shape[1])
    data = np.ascontiguousarray(image, dtype=np.uint8)
    source = QImage(data.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
    scaled = source.scaled(
        max(1, size[0]),
        max(1, size[1]),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    ).convertToFormat(QImage.Format.Format_RGBA8888)
    stride = scaled.bytesPerLine()
    raw = np.frombuffer(scaled.constBits(), dtype=np.uint8, count=stride * scaled.height())
    rows = raw.reshape(scaled.height(), stride)[:, : scaled.width() * 4]
    return rows.reshape(scaled.height(), scaled.width(), 4).copy()


def _as_corners(points: tuple[tuple[float, float], ...] | None) -> Corners | None:
    """4 点の組を四隅の型へ 数が合わなければ ``None``"""
    if points is None or len(points) != 4:
        return None
    return (points[0], points[1], points[2], points[3])


#: シーンの入れ子の深さの上限 循環はコマンドとファイルの読み込みで止めてあるが、
#: 深く積んだだけでも描画の手間が掛け算で増える 画面の外から来た壊れた
#: プロジェクトで固まらないための最後の歯止め
MAX_SCENE_DEPTH = 8

#: 同時に開いておくデコーダの上限 素材ごとにコンテナとスレッドを抱えるので、
#: 際限なく開くとファイルハンドルとメモリを食い潰す
MAX_OPEN_DECODERS = 8

#: 作った絵を覚えておくクリップの数 1 枚で画面 1 枚ぶんのメモリを使う
#: 画面より大きく作った絵（はみ出す図形やテキスト）は、画面何枚ぶんかで数える（:func:`_make_room`）
MAX_GENERATED_CACHE = 48


def _make_room(cache: OrderedDict[ClipId, _Made], image: np.ndarray, screen_bytes: int) -> None:
    """``image`` を覚える前に、古い絵から捨てて画面 :data:`MAX_GENERATED_CACHE` 枚ぶんに収める

    数ではなく大きさで数える 画面より大きい絵は 1 枚で画面の何倍にもなり（一辺 8192 まで
    広げると 256MB）、数だけで数えると長いテロップを並べたタイムラインでメモリを食い潰す
    画面と同じ大きさ以下の絵は 1 枚で 1 つ分なので、今までと同じ数だけ覚える
    入れ替える絵は呼ぶ側が先に外しておく（入れ替えでは関係ないクリップの絵を捨てない）
    """

    def weight(picture: np.ndarray) -> int:
        return max(1, -(-int(picture.nbytes) // max(1, screen_bytes)))

    used = sum(weight(made.image) for made in cache.values())
    needed = weight(image)
    while cache and used + needed > MAX_GENERATED_CACHE:
        _, dropped = cache.popitem(last=False)
        used -= weight(dropped.image)


#: レイヤーごとの並列デコードに使うスレッドの数の既定 1 なら並べない
#: PyAV のデコードは GIL を解放するので、別の素材どうしなら本当に重なる
#: （1080p 3 枚重ねで 1.80 倍、4K 2 枚重ねで 1.50 倍 ``tools/bench_export.py``）
DEFAULT_DECODE_THREADS = 4

#: 並列デコードのスレッド数として受け付ける上限 デコーダ 1 つが
#: :data:`MAX_OPEN_DECODERS` 本までしか開かないので、それを超えて並べても相手がいない
MAX_DECODE_THREADS = MAX_OPEN_DECODERS

#: デコーダ 1 つを指す鍵（素材とストリーム番号） 先読みはこの鍵ごとに 1 本だけ走らせる
#: 同じデコーダを 2 スレッドから触ると、コンテナの位置が食い違って絵が壊れる
_DecodeKey = tuple[MediaId, int]


#: 値が止まっていても時計で絵が変わる図形
#: 星空は粒が時間で流れ、移動軌跡は線が時間で伸びる（固定速度） 載せ忘れると、
#: 最初のフレームの絵を覚えたまま使い回して、止まった星空になる
_CLOCK_SHAPES = frozenset({"concentration", "starfield", "motion_trail", "waveform"})


#: 絵をこちらで作るクリップか 外枠の計算（:mod:`sashimono.engine.render.outline`）と
#: 同じ物を使う 別々に持つと、枠の置き方と描く置き方が食い違う
_is_generated = is_generated


def _source_time(clip: Clip, media: MediaItem, frame: int, rate: FrameRate) -> Fraction:
    """``clip`` を ``frame`` に描くとき、素材のどの時刻を読むか

    先読みと本番で同じ式を使う ずれると先読みが当たらず、並べた意味が消える
    絵を止めたクリップ（:attr:`Clip.hold_at`）は止めた時刻より先を読まない 先読みも
    同じ時刻を頼むので、止めた後の時刻をデコードしに行かず、デコーダも同じ位置に
    留まって読み直さない
    """
    if media.is_still:
        # 静止画は時間を持たない 常に先頭を返す
        return Fraction(0)
    return clip.picture_time(frame - clip.timeline_start, rate)


#: 音声波形の音を読むデコーダの鍵 素材の道・音声ストリームの番号・読むレート
WaveformKey = tuple[Path, int, int]

#: 音声波形の音を、時刻のどれだけ前から読み始めるか（サンプル 読んだ後で捨てる）
#: 理由は :meth:`FrameRenderer._waveform_audio` に書いた
_WAVEFORM_PREROLL = 512


def _waveform_key(project: Project, clip: Clip, source: GeneratedSource) -> WaveformKey | None:
    """このクリップの音声波形が読む音の鍵 読む音が無ければ ``None``"""
    path: Path | None = None
    if clip.media_id is not None:
        media = project.find_media(clip.media_id)
        if media is not None:
            path = Path(media.path)
    if path is None:
        written = source.params.get("audio_path")
        if isinstance(written, str) and written:
            path = Path(written)
    if path is None:
        return None
    return path, clip.stream_index, project.settings.sample_rate


#: 覚えておく絵の鍵のうち、音声波形が読む音（``WaveformKey``）が入る位置
#: 末尾（設定の指紋）の 1 つ前に置く 先頭から数えると、途中に項目を足したときに
#: 別の項目を読んで判定が崩れる
_HEARD_AT = -2


def _hears(key: object) -> bool:
    """覚えておいた絵が、音声波形のものか（鍵のその位置に読む音が入っている）"""
    return isinstance(key, tuple) and len(key) >= 2 and key[_HEARD_AT] is not None


def _waveform_keys(project: Project) -> set[WaveformKey]:
    """プロジェクトの音声波形が読む音の鍵をすべて 入れ子のシーンの中も見る"""
    keys: set[WaveformKey] = set()
    for timeline in (project.timeline, *(scene.timeline for scene in project.scenes)):
        for track in timeline.tracks:
            for clip in track.clips:
                source = clip.source
                if source is None or source.params.get("shape") != "waveform":
                    continue
                key = _waveform_key(project, clip, source)
                if key is not None:
                    keys.add(key)
    return keys


def _varies_over_time(source: GeneratedSource) -> bool:
    """フレームごとに絵が変わる生成オブジェクトか

    キーフレームの付いた値、埋め込んだ Lua、時間を数えるタイマー、集中線の
    切り替えのように時計を見るものは、毎フレーム作り直す
    """
    for value in source.params.values():
        if isinstance(value, AnimatedValue) and value.is_animated:
            return True
    if source.params.get("timer_format"):
        return True
    if source.params.get("shape") in _CLOCK_SHAPES:
        return True
    text = source.params.get("text")
    return isinstance(text, str) and has_embedded(text)


def _fingerprint(params: dict[str, ParamValue]) -> tuple[tuple[str, str], ...]:
    """生成オブジェクトの設定の指紋 値が同じなら同じ絵になる

    アニメーションの値も丸ごと文字にする キーフレームの有無で分けると、
    動く値を持つクリップの絵を毎フレーム作り直すことになる（フレームは鍵の別の項目）
    """
    return tuple(sorted((name, repr(value)) for name, value in params.items()))


@dataclass(frozen=True, slots=True)
class RenderQuality:
    """プレビューの解像度を落として再生を軽くするための設定

    ``divisor`` が 2 なら縦横半分 合成そのものが軽くなるので、重いタイムラインでも
    実時間再生を維持できる 書き出しでは常に 1 を使う
    """

    divisor: int = 1

    def __post_init__(self) -> None:
        if self.divisor < 1:
            raise ValueError(f"分母は 1 以上必要: {self.divisor}")

    def apply(self, width: int, height: int) -> tuple[int, int]:
        return max(1, width // self.divisor), max(1, height // self.divisor)


FULL_QUALITY = RenderQuality(1)


@dataclass(slots=True)
class _Made:
    """作って覚えた絵 1 枚ぶん（:meth:`FrameRenderer._generate`）"""

    #: 同じ絵になるかを見分ける鍵（設定・大きさ・フレームなど）
    key: object
    image: np.ndarray
    #: 文字の枠（AviUtl2 の組み方のテキストだけが持つ :func:`_object_box`）
    framed: Frame | None
    #: 色の付いた範囲（:func:`_content_box`） 要るまで探さないので、探したかは
    #: :attr:`scanned` で見る 動かない字幕や図形は同じ絵を毎フレーム描くのに、
    #: エフェクトを積むと範囲を毎回全体から探していた（1080p の図形 20 本で 1 フレーム
    #: 48ms Issue #128） 絵と同じ所に持つので、絵が作り直されれば一緒に消える
    box: _Box | None = None
    scanned: bool = False


class FrameRenderer:
    """タイムラインの指定フレームを 1 枚の画像に合成する

    GL コンテキストを持つので、生成したスレッドの上でだけ使うこと 再生用と
    書き出し用で別インスタンスにする

    ``context`` を省略すると自前でオフスクリーンのコンテキストを作る Qt の
    ウィジェットの中から使うときは、すでに current になっているので
    :class:`~sashimono.engine.gpu.CurrentGLContext` を渡す
    """

    def __init__(
        self,
        project: Project,
        *,
        context: GLScope | None = None,
        quality: RenderQuality = FULL_QUALITY,
        proxies: ProxyStore | None = None,
        decode_threads: int = DEFAULT_DECODE_THREADS,
    ) -> None:
        self._project = project
        self._quality = quality
        #: 使えないので捨てた控えの素材 作り直しを頼む側が拾う
        self._discarded: set[MediaId] = set()
        #: プレビュー用の控えの置き場 既定は使わない
        #: **書き出しでは必ず None** 混ざると、画面では気付かないまま
        #: 低解像度の絵が最終出力に入る
        self._proxies = proxies
        self._owns_context = context is None
        self._context = context if context is not None else OffscreenGLContext()

        width, height = quality.apply(*project.settings.resolution)
        with self._context:
            self._compositor = Compositor(width, height, encoded=self._encoded)
            # エフェクト処理は合成と同じ全画面四角形を使い回す
            self._effects = EffectProcessor(width, height, self._compositor.quad)
            self._effects.canvas_encoded = self._encoded
        # 画素で決める設定を合成の画素へ縮める割合 画質を落としたプレビューで 1 より小さい
        self._effects.pixel_scale = self._scale[0]
        #: 素材ごとのデコーダ 最近使ったものを残す
        self._decoders: OrderedDict[_DecodeKey, VideoDecoder] = OrderedDict()
        #: レイヤーごとの並列デコードに使うスレッドの数 1 なら並べない
        self._decode_threads = max(1, min(int(decode_threads), MAX_DECODE_THREADS))
        #: 先読みの走り係 並べないなら作らない
        self._decode_pool: ThreadPoolExecutor | None = None
        #: いま先読み中のデコーダ 鍵ごとに 1 本だけ 値は（頼んだ時刻, 結果）
        #: 鍵ごとに 1 本に絞ることが、同じデコーダを 2 スレッドから触らない保証になる
        self._decoding: dict[_DecodeKey, tuple[Fraction, Future[np.ndarray | None]]] = {}
        #: トラックごとの転送用テクスチャ 毎フレーム作り直すと確保と解放で時間を食う
        self._textures: dict[str, Texture] = {}
        #: AviUtl スクリプトを走らせる係 使うまで作らない
        self._scripts: ScriptStage | None = None
        #: スクリプトが積んだ効果を、その場で絵へ掛ける係（#176） GPU の物は使うまで作らない
        self._script_baker = ScriptEffectBaker()
        #: フレームバッファのクリップが画面を写し取る先 使うまで作らない
        self._grab: Framebuffer | None = None
        #: クリップごとに作った絵 同じ設定と同じフレームなら作り直さない
        self._generated: OrderedDict[ClipId, _Made] = OrderedDict()
        #: 入れ子のシーンを描く合成先 深さごとに 1 つ 使うまで作らない
        self._nested: dict[int, Compositor] = {}
        #: クリップを下のクリップの形で切り抜くときに使う合成先（役目と深さごと）
        self._layers: dict[tuple[str, int], Compositor] = {}
        #: いま重ねている段で、自分の絵を描いたクリップ（下から順） 直前オブジェクトが写す相手
        #: 段ごと（:meth:`_compose_tracks`）に作り直す 入れ子のシーンや場面切り替えの中の
        #: クリップを、外の直前オブジェクトの相手にしないため
        self._drawn: list[tuple[Track, Clip]] = []
        #: いま重ねているタイムラインの絵を描くトラックの並び（描かない物も含む）
        #: グループ制御が受け持つ本数を数える（:meth:`_compose_timeline`）
        self._picture_order: tuple[Track, ...] = ()
        #: 段ごとに、最後に写し取ったフレームバッファのクリップ 写し取った絵は
        #: ``framebuffer_seen`` の合成先にある（:meth:`_draw_previous` が下の写しに使う）
        self._framebuffer_seen: dict[int, str] = {}
        #: 音声波形が音を読むデコーダ（道ごと） 映像のデコーダとは別に持つ
        self._audio: OrderedDict[WaveformKey, AudioDecoder] = OrderedDict()
        #: 開けなかった音 毎フレーム開き直さないために覚えておく
        self._audio_missing: set[WaveformKey] = set()
        #: 移動軌跡の道の置き場 描くたびに頭から位置を引き直さないために覚えておく
        #: レンダラごとに持つ 共有すると、1 つを閉じたときに他のレンダラの道まで消える
        self._trail_paths = TrailPaths()
        #: クリップごとのレイヤー番号（スクリプトの ``obj.layer``） プロジェクトごとに 1 度数える
        self._layer_numbers: tuple[Project, dict[int, int], dict[ClipId, int]] | None = None
        self._closed = False

    @property
    def project(self) -> Project:
        return self._project

    @property
    def size(self) -> tuple[int, int]:
        return self._compositor.width, self._compositor.height

    def set_project(self, project: Project) -> None:
        """編集後のプロジェクトに差し替える

        解像度が変わればフレームバッファも作り直す 参照されなくなった素材の
        デコーダはここで閉じる 開いたままだとファイルを差し替えられない
        """
        previous = self._project
        self._project = project
        # 先読みが走ったままデコーダを閉じると、走っているスレッドが解放済みの
        # コンテナを触る 閉じる前に必ず受け取り終える
        self._settle_decodes()

        if project.settings.resolution != previous.settings.resolution:
            self._resize(*self._quality.apply(*project.settings.resolution))
        if project.settings.blending != previous.settings.blending:
            # 入れ子や切り抜きの合成先も同じ方法にそろえる 1 つでも残ると、そこで
            # 符号化した値とリニアの値が混ざり、入れ子のシーンだけ明るさが変わる
            for compositor in (self._compositor, *self._nested.values(), *self._layers.values()):
                compositor.encoded = self._encoded
            self._effects.canvas_encoded = self._encoded

        alive = {m.id for m in project.media}
        for key in [k for k in self._decoders if k[0] not in alive]:
            self._decoders.pop(key).close()

        # 音声波形の音も、使われなくなったものは閉じる 開いたままだと、クリップを消しても
        # レンダラを閉じるまで音声ファイルを差し替えられない
        # 開けなかった記録は忘れる 後から置いた素材を、差し替えのたびに読み直せるように
        # 移動軌跡の覚えた道も捨てる 長い道は 1 本で数十 MB あり、使わなくなった
        # プロジェクトのぶんを残さない（次に描くときに今のフレームまで引き直す）
        # 置き場はこのレンダラのもの 書き出しのような別のレンダラの道は捨てない
        self._trail_paths.clear()
        heard = _waveform_keys(project)
        for sound in [k for k in self._audio if k not in heard]:
            self._audio.pop(sound).close()
        if self._audio_missing:
            self._audio_missing.clear()
            # 音が無くて空で描いた波形の絵も忘れる 鍵（設定とフレーム）は同じなので、
            # 残しておくと素材を置いた後も空の絵を使い回す
            for clip_id in [cid for cid, made in self._generated.items() if _hears(made.key)]:
                del self._generated[clip_id]

        # 使われなくなったエフェクトの画像を GPU から手放す 手放さないと、模様を
        # 差し替えるたびに前の画像が閉じるまで残り、長く編集するほどメモリを食う
        with self._context:
            self._effects.retain_images(image_paths(project))

    def set_decode_threads(self, threads: int) -> None:
        """レイヤーごとの並列デコードのスレッド数を変える 1 で並べない

        走り係は畳んでから作り直す 減らしたのに前の本数のまま走り続けると、
        設定で減らした意味が無い
        """
        wanted = max(1, min(int(threads), MAX_DECODE_THREADS))
        if wanted == self._decode_threads:
            return
        self._shutdown_decodes()
        self._decode_threads = wanted

    def set_quality(self, quality: RenderQuality) -> None:
        self._quality = quality
        self._resize(*quality.apply(*self._project.settings.resolution))

    def _resize(self, width: int, height: int) -> None:
        with self._context:
            self._compositor.resize(width, height)
            self._effects.resize(width, height)
        self._effects.pixel_scale = self._scale[0]

    @property
    def _scale(self) -> tuple[float, float]:
        """合成の画素 1 つが画面の画素いくつ分かの逆数（:func:`canvas_scale`） 書き出しは 1"""
        return canvas_scale(
            self._project.settings.resolution,
            (self._compositor.width, self._compositor.height),
        )

    @property
    def _encoded(self) -> bool:
        """sRGB で符号化した値のまま重ねるか（プロジェクトの重ね合わせの設定）"""
        return self._project.settings.blending == Blending.SRGB

    @property
    def compositor(self) -> Compositor:
        """合成結果を直接画面へ出したいときの逃げ道 プレビューが使う"""
        return self._compositor

    def compose(self, frame: int, *, underlay: bool = True) -> None:
        """``frame`` を合成する 結果は CPU へ戻さず GPU 上に残る

        画面に出すだけなら往復が要らない :meth:`render` はこれを呼んでから
        読み出しているだけ ``underlay`` が偽なら最後の黒を敷かず、透けたまま残す
        """
        if self._closed:
            raise RuntimeError("閉じたレンダラは使えない")

        # 透明な下地の上で重ね、最後に黒を敷く 黒の上で重ねると、乗算などの合成が
        # 下に何も無い所でも黒と混ざる（YMM4 は透明な所では上の絵をそのまま出す）
        # 写し取る絵（フレームバッファ）は、写すときに黒を敷く（YMM4 と同じ）
        self._compositor.begin((0.0, 0.0, 0.0, 0.0))
        try:
            self._compose_timeline(self._project.timeline, frame, depth=0)
        except BaseException:
            # 描く側が投げたときは、先読みを受け取るだけにして元の失敗を残す
            # ここで先読みの失敗を投げ直すと、本当の原因がそれに置き換わって消える
            self._settle_decodes()
            raise
        # 使われなかった先読みを必ず回収する 置き去りにすると、次のフレームで
        # 同じデコーダへ頼んだときに前の走りと重なり、例外も誰も受け取らない
        self._drain_decodes()
        if underlay:
            self._compositor.underlay((0.0, 0.0, 0.0, 1.0))

    def _compose_timeline(self, timeline: Timeline, frame: int, *, depth: int) -> None:
        """1 本のタイムラインを、いまの合成先へ重ねる シーンの入れ子でも同じ道を通る

        映像トラックと混合トラックを、並びの順（先頭が一番奥）に重ねる
        （:class:`~sashimono.core.model.Timeline` の重なり順）
        """
        # グループ制御が受け持つ本数は、描かないトラックも含めた並びで数える（AviUtl は
        # 非表示のレイヤーも番号として数える） 入れ子のシーンでは中の並びへ差し替えて戻す
        outer = self._picture_order
        self._picture_order = timeline.picture_tracks()
        try:
            self._compose_tracks(list(timeline.active_picture_tracks()), frame, depth)
        finally:
            self._picture_order = outer

    def _compose_tracks(
        self,
        tracks: list[Track],
        frame: int,
        depth: int,
        *,
        started_before: int | None = None,
    ) -> None:
        """下のトラックから順に重ねる 場面切り替えは、それより下のトラックを見て描き直す

        ``started_before`` を渡すと、その時刻より前に始まったクリップだけを重ねる
        場面切り替えの前の場面を作るため

        絵を描かないクリップ（混合トラックの音だけのクリップ）はここで外す 残すと
        すぐ上のクリップの切り抜き（``clip_to_below``）の相手になり、形の無い物で
        切り抜いて何も映らない
        """
        rate = self._project.rate
        visible = [
            (index, track, clip)
            for index, track in enumerate(tracks)
            if (clip := track.clip_at(frame)) is not None
            and clip.enabled
            and (started_before is None or clip.timeline_start < started_before)
            and self._project.draws_picture(track, clip)
        ]
        # 描き始める前に、この段のクリップぶんのデコードを走らせておく
        # 描きながら 1 本ずつデコードすると、重ねた枚数だけ待ちが直列に並ぶ
        self._prefetch_decodes([clip for _, _, clip in visible], frame, rate)
        outer_drawn = self._drawn
        self._drawn = []
        try:
            self._compose_visible(tracks, visible, frame, rate, depth)
        finally:
            self._drawn = outer_drawn

    def _compose_visible(
        self,
        tracks: list[Track],
        visible: list[tuple[int, Track, Clip]],
        frame: int,
        rate: FrameRate,
        depth: int,
    ) -> None:
        """:meth:`_compose_tracks` が選んだクリップを下から重ねる"""
        below: Compositor | None = None
        drawn_tracks = {track.id for track in tracks}
        #: 1 枚の絵にまとめて描いたトラック 外の重ねでは描かない
        bundled: set[TrackId] = set()
        for position, (index, track, clip) in enumerate(visible):
            if track.id in bundled:
                continue
            if clip.is_group:
                # グループ制御は自分では描かない 1 本ずつに掛ける物は、受け持つクリップを
                # 描くときに当てる 1 枚にまとめる物は、ここで受け持つトラックを 1 枚に描いて
                # から掛ける 下の形も残さない（上のクリップの切り抜きの相手にならない）
                if group_as_one(clip):
                    targets = [
                        other
                        for other in tracks
                        if group_reaches(self._picture_order, track.id, clip, other.id)
                    ]
                    groups = [
                        (group_track, group)
                        for group_track, group in controlling_groups(
                            self._picture_order, track.id, frame
                        )
                        if group_track.id in drawn_tracks and not group_as_one(group)
                    ]
                    bundled.update(other.id for other in targets)
                    self._draw_group(track, clip, targets, groups, frame, rate, depth)
                below = None
                continue
            if clip.source is not None and clip.source.kind == "transition":
                self._draw_transition(tracks[:index], clip, frame, rate, depth)
                below = None
                continue
            if clip.is_filter:
                # 切り抜きや残像の道へは入れない どちらもクリップ 1 本だけを透明な所へ
                # 描き直すので、下の絵を持たないフィルタは何も描かない（形も残さない）
                self._draw_filter(track, clip, frame, rate, depth)
                below = None
                continue
            groups = [
                (group_track, group)
                for group_track, group in controlling_groups(self._picture_order, track.id, frame)
                if group_track.id in drawn_tracks and not group_as_one(group)
            ]
            original = clip
            if groups:
                clip = grouped(original, groups, frame)
            if clip.clip_to_below:
                self._draw_clipped(track, clip, frame, rate, depth, below)
            else:
                self._draw_trail(track, original, frame, rate, depth, groups)
                self._draw_clip(track, clip, frame, rate, depth)
            above = visible[position + 1][2] if position + 1 < len(visible) else None
            # すぐ上のクリップがこのクリップの形で切り抜くなら、形を取っておく
            below = (
                self._capture(track, clip, frame, rate, depth)
                if above is not None and above.clip_to_below
                else None
            )
            # 自分の絵を持つクリップだけを数える フィルタと場面切り替えは写す絵を持たない
            # それらを挟んだときに AviUtl がどれを写すかは測っていない
            self._drawn.append((track, clip))

    def _draw_trail(
        self,
        track: Track,
        clip: Clip,
        frame: int,
        rate: FrameRate,
        depth: int,
        groups: Sequence[tuple[Track, Clip]] = (),
    ) -> None:
        """残像（``after_image``）を積んだクリップの、前のフレームの絵を薄くして先に描く

        1 フレーム前ほど濃く、強さの累乗で薄れる クリップの頭より前は描かない
        エフェクトはフレームごとに独立して描けるので、前のフレームを描き直せば済む
        """
        current = grouped(clip, groups, frame)
        trail = next((e for e in current.effects if e.enabled and e.kind == "after_image"), None)
        if trail is None:
            return
        local_frame = frame - clip.timeline_start
        strength = trail.params.get("strength")
        keep = strength.at(local_frame) if isinstance(strength, AnimatedValue) else 50.0
        fade = min(max(keep / 100.0, 0.0), 0.99)
        samples = trail.params.get("samples")
        count = int(samples) if isinstance(samples, int | float) else 12
        for back in range(min(count, local_frame), 0, -1):
            weight = fade**back
            if weight < 0.02:
                continue
            earlier = frame - back
            sample = grouped(clip, groups, earlier)
            faded = replace(
                sample,
                effects=tuple(e for e in sample.effects if e.kind != "after_image"),
                opacity=AnimatedValue(sample.opacity.at(earlier - sample.timeline_start) * weight),
            )
            self._draw_clip(track, faded, earlier, rate, depth)

    def _draw_transition(
        self, tracks: list[Track], clip: Clip, frame: int, rate: FrameRate, depth: int
    ) -> None:
        """下のトラックの絵を、前の場面から後の場面へ切り替える（YMM4 の ``TransitionItem``）

        決まりは YMM4 に描かせた試験（``tools/ymm4_probes.py`` の 4 回目）と、
        配布されている場面切り替え 11 本を YMM4 で書き出した絵から読んだ

        - 後の場面は、いまの時刻の下の絵そのもの
        - 前の場面は、**切り替えの頭より前に始まったクリップだけ**の絵 頭と同時か後に
          始まったクリップは入れない（前に何も無ければ透明で、書き出すと黒）
          入れると、テンプレートだけを置いたときに前の場面にも後の場面が映り、
          黒からの切り替えが最初から明るく出る（ペイントトランジションで差が 249）
        - 前の場面は、範囲の中で終わる前の場面のクリップの終わり（切れ目）の直前で止まる
          それより前は動き続ける 範囲の中で終わるものが無ければ止めない
        - 進み具合は範囲の頭から終わりまで 押し出しとスライドは画面 1 枚分を動く
        - 切り替え（switch）は範囲の真ん中で入れ替わる 切れ目ではない
          （じわっと抽象化切り替えは長さ 40、切れ目 30 で、YMM4 は 20 から後の場面を出す）
        - 上のトラックには効かない
        - 場面に掛けるエフェクトは、場面の中身が描かれた範囲を絵の大きさとして見る
          原点は画面の中央のまま（:meth:`_transition_scene`）
        """
        if depth >= MAX_SCENE_DEPTH:
            return
        start, end = clip.timeline_start, clip.timeline_end
        # 前の場面に入らないクリップの終わりで止めると、前の場面のクリップが範囲の途中で
        # 動かなくなる 前の場面に入るもの（頭より前に始まったもの）だけを見る
        cut = max(
            (
                other.timeline_end
                for track in tracks
                for other in track.clips
                if other.enabled
                and self._project.draws_picture(track, other)
                and other.timeline_start < start
                and start < other.timeline_end <= end
            ),
            default=end,
        )
        before_frame = min(frame, cut - 1)
        local_frame = frame - start
        params = clip.source.params if clip.source is not None else {}
        style = str(params.get("style", "fade"))
        target = str(params.get("target", "after"))
        angle = params.get("angle")
        degrees = angle.at(local_frame) if isinstance(angle, AnimatedValue) else 0.0
        progress = ease(
            local_frame / clip.duration,
            str(params.get("easing", "linear")),
            str(params.get("easing_mode", "in")),
        )

        width, height = self._compositor.width, self._compositor.height
        full = Placement(0.0, 0.0, float(width), float(height))
        outer = self._compositor
        after = self._layer("transition_after", depth)
        after.begin((0.0, 0.0, 0.0, 0.0))
        after.draw_handle(outer.canvas.color, full, flip=False, premultiplied=True)
        before = self._layer("transition_before", depth)
        self._compositor = before
        try:
            before.begin((0.0, 0.0, 0.0, 0.0))
            # 止める前で、映るクリップがどれも頭より前に始まっているなら、前の場面は
            # 後の場面と同じ絵 描き直さずに写す
            if before_frame == frame and not self._starts_within(tracks, frame, start):
                before.draw_handle(after.canvas.color, full, flip=False, premultiplied=True)
            else:
                self._compose_tracks(tracks, before_frame, depth + 1, started_before=start)
        finally:
            self._compositor = outer

        # AviUtl のシーンチェンジのスクリプトは前の場面に掛ける効果ではなく、切り替えそのもの
        # 前の場面の効果に混ぜると、場面へ掛けられないスクリプトとして数えられて何も起きない
        # 積んだ順にすべて走らせる 先頭の 1 本だけにすると、2 本目以降が描かれず記録にも残らない
        # 外すのは走らせる物（有効な物）だけ 切った物は前の場面の列に残し、ほかの切った効果と
        # 同じく何もしない
        scene_changes = tuple(e for e in clip.effects if e.enabled and is_scene_change(e.kind))
        before_image = self._transition_scene(
            before,
            tuple(e for e in clip.effects if not any(e is s for s in scene_changes)),
            "before",
            clip,
            local_frame,
            rate,
            depth,
        )
        after_image = self._transition_scene(
            after, clip.after_effects, "after", clip, local_frame, rate, depth
        )

        outer.begin((0.0, 0.0, 0.0, 0.0))
        if scene_changes and self._draw_scene_change(
            scene_changes, before_image, after_image, clip, local_frame, rate, progress
        ):
            return
        direction = (math.cos(math.radians(degrees)), math.sin(math.radians(degrees)))
        travel = abs(direction[0]) * width + abs(direction[1]) * height

        def moved(amount: float) -> Placement:
            # 角度 0 で右へ、90 で下へ（画面の Y は下が正）
            return Placement(
                direction[0] * amount, direction[1] * amount, float(width), float(height)
            )

        def put(
            image: Compositor,
            placement: Placement = full,
            opacity: float = 1.0,
            blend: str = BlendMode.NORMAL,
        ) -> None:
            outer.draw_handle(
                image.canvas.color,
                placement,
                opacity=opacity,
                flip=False,
                blend=blend,
                premultiplied=True,
            )

        if style == "switch":
            put(before_image if local_frame * 2 < clip.duration else after_image)
        elif style == "fade":
            # YMM4 は黒の上の絵として sRGB の値のまま混ぜる リニアで混ぜると中間が明るく浮く
            put(before_image)
            outer.underlay((0.0, 0.0, 0.0, 1.0))
            put(after_image, opacity=progress, blend=BlendMode.SRGB_MIX)
        elif style == "push":
            put(after_image, moved(-(1.0 - progress) * travel))
            put(before_image, moved(progress * travel))
        elif style == "slide" and target == "before":
            put(after_image)
            put(before_image, moved(progress * travel))
        elif style == "slide":
            put(before_image)
            put(after_image, moved(-(1.0 - progress) * travel))
        elif target == "before":
            # 重ねるだけ 手前にする場面を後に描く
            put(after_image)
            put(before_image)
        else:
            put(before_image)
            put(after_image)

    def _draw_scene_change(
        self,
        effects: tuple[Effect, ...],
        before_image: Compositor,
        after_image: Compositor,
        clip: Clip,
        local_frame: int,
        rate: FrameRate,
        progress: float,
    ) -> bool:
        """AviUtl のシーンチェンジのスクリプトで、いまの合成先へ切り替えの絵を描く（#196）

        前の場面をオブジェクト、後の場面をフレームバッファとして走らせ、後の場面の上へ
        スクリプトの描画を重ねる（:meth:`ScriptStage.run_scene_change` 向きもそこに書いた）
        走らなければ ``False`` を返し、呼ぶ側が切り替え方どおりに描く（素通し）

        どちらの場面も黒を敷いて渡す AviUtl のフレームバッファは何も無い所も不透明な黒で
        （:meth:`_screen_picture`）、前の場面も同じ画面を描いた物なので同じにした
        前後の場面の取り方・進み具合・黒を敷くことは、実物の AviUtl で測っていない
        """
        stage = self._script_stage()
        if stage is None:  # pragma: no cover - 係は必ず作れる
            return False
        calls, failed = stage.run_scene_change(
            clip,
            effects,
            self._screen_picture(before_image),
            self._screen_picture(after_image),
            frame=local_frame,
            fps=float(rate.fps),
            progress=progress,
        )
        if failed:
            # ffi が要るスクリプト（sigma のディザ 4 本）はここへ来る 利用者の決定で ffi は
            # 許さない（配布スクリプトから任意のメモリや DLL に触れられるため） 何が走らずに
            # 素通しになったのかを、スクリプトの名前で数えて残す 何本も積んだときも数えるのは
            # 走らなかった物だけ 走った物まで数えると、どれを直せばよいか分からない
            for effect in failed:
                entry = script_catalog().get(effect.kind)
                label = entry.label if entry is not None else effect.kind
                global_report.note_missing(
                    f"シーンチェンジのスクリプトが走らない（素通し）: {label}"
                )
            return False
        outer = self._compositor
        full = Placement(0.0, 0.0, float(outer.width), float(outer.height))
        outer.draw_handle(after_image.canvas.color, full, flip=False, premultiplied=True)
        outer.underlay((0.0, 0.0, 0.0, 1.0))
        self._draw_calls(self._texture_for("scene_change"), clip, calls, (), local_frame, rate, 1.0)
        return True

    def _starts_within(self, tracks: list[Track], frame: int, start: int) -> bool:
        """前の場面を描き直さず、いまの合成結果（後の場面）を写して済ませてよいかを決める

        頭以降に始まったクリップが映っていると、前の場面は後の場面と別の絵になる
        それでも写すと、前の場面に後の場面が混ざって黒から出る切り替えが明るくなる
        映っていなければ同じ絵なので、描き直す手間を省ける（``True`` は描き直しが要る）
        絵を描かないクリップは映っていないので数えない
        """
        return any(
            clip.enabled
            and clip.timeline_start >= start
            and self._project.draws_picture(track, clip)
            for track in tracks
            if (clip := track.clip_at(frame)) is not None
        )

    def _transition_scene(
        self,
        image: Compositor,
        effects: tuple[Effect, ...],
        role: str,
        clip: Clip,
        local_frame: int,
        rate: FrameRate,
        depth: int,
    ) -> Compositor:
        """場面にエフェクトを掛けた絵 掛けるものが無ければそのまま返す

        エフェクトの結果は次に掛けるまでしか残らない（中のバッファを使い回す）ので、
        別の合成先へ描き写してから返す
        """
        gpu_effects, scripts = split_effects(effects)
        # 切ったスクリプトは描かないので数えない 数えると、切ったシーンチェンジ（前の場面の
        # 列に残る）まで場面へ掛けられない穴として出る
        if any(script.enabled for script in scripts):
            global_report.note_missing("場面切り替えに積んだ AviUtl スクリプト")
        if not self._effects.has_work(gpu_effects):
            return image
        box = image.content_box()
        result = self._effects.apply(
            image.canvas,
            gpu_effects,
            frame=local_frame,
            fps=float(rate.fps),
            flip_source=False,
            duration=clip.duration,
            # 場面の絵の大きさは、中身が描かれた範囲 画面全体を絵の大きさにすると、
            # 中心点の「下端」が画面の下端になって図形ごと画面の外へ回り（ローテンション
            # トランジション）、タイルが画面の高さごとに並んで隙間が空く（リール回転風）
            # YMM4 の書き出しでは、どちらも図形の範囲で回り、並んでいた
            bounds=None
            if box is None
            else (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            # 原点は画面の中央のまま 図形のマスクは範囲ではなく原点に置かれる
            # （円形端から暗転の円は、図形が右へ寄っていても画面の中央から開いた）
            origin=(image.width / 2.0, image.height / 2.0),
            premultiplied=True,
        )
        done = self._layer(f"transition_{role}_done", depth)
        done.begin((0.0, 0.0, 0.0, 0.0))
        full = Placement(0.0, 0.0, float(done.width), float(done.height))
        done.draw_handle(result.color, full, flip=False)
        return done

    def render(self, frame: int, *, transparent: bool = False) -> np.ndarray:
        """``frame`` の合成結果を sRGB の ``(高さ, 幅, 4)`` uint8 で返す

        映像トラックを下から順に重ねる タイムラインの下のトラックが奥、
        上のトラックが手前という Premiere / AviUtl と同じ並び

        ``transparent`` なら黒を敷かず、ストレートアルファで返す プリセットの見本の絵（#277）は
        地を表示の側で選べる（暗い地・明るい地・市松）ので、黒を焼き込むと白以外の地で見られない
        """
        with self._context:
            self.compose(frame, underlay=not transparent)
            return self._compositor.read(straight=transparent)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # 走っている先読みを先に片付ける デコーダを閉じてからスレッドが動くと、
        # 解放済みのコンテナを触って落ちる 例外は握り潰す ここは後始末で、
        # 読めなかった素材より「必ず手放す」方が大事
        self._shutdown_decodes()
        # 覚えた移動軌跡の道も手放す 閉じたレンダラのぶんを次のプロジェクトまで残さない
        self._trail_paths.clear()
        for decoder in self._decoders.values():
            decoder.close()
        self._decoders.clear()
        for audio in self._audio.values():
            audio.close()
        self._audio.clear()
        with self._context:
            for texture in self._textures.values():
                texture.release()
            self._textures.clear()
            self._effects.release()
            self._script_baker.release()
            if self._grab is not None:
                self._grab.release()
            for nested in (*self._nested.values(), *self._layers.values()):
                nested.release()
            self._nested.clear()
            self._layers.clear()
            self._compositor.release()
        if self._owns_context:
            self._context.release()

    def _draw_clip(
        self,
        track: Track,
        clip: Clip,
        frame: int,
        rate: FrameRate,
        depth: int = 0,
        *,
        below: Compositor | None = None,
    ) -> None:
        """クリップ 1 本を描く

        ``below`` はスクリプトの ``frm`` とフレームバッファが写す合成先 別の合成先へ
        描き分けるとき（:meth:`_draw_into`）に、下の絵を溜めた外側を渡す
        """
        if clip.scene_id is not None:
            self._draw_scene(track, clip, frame, rate, depth)
            return
        if clip.source is not None and clip.source.kind == "framebuffer":
            self._draw_framebuffer(track, clip, frame, rate, depth, below=below)
            return
        if clip.source is not None and clip.source.kind == PREVIOUS_OBJECT.kind:
            self._draw_previous(track, clip, frame, rate, depth)
            return
        local_frame = frame - clip.timeline_start
        opacity = clip.opacity.at(local_frame)
        gpu_effects, scripts = split_effects(clip.effects)
        # スクリプトは画面の画素で数える（:meth:`_draw_scripted`） 画質を落としていても
        # 書き出しと同じ大きさの絵を渡す 縮めた絵を渡すと obj.w から決める位置や大きさがずれる
        image = (
            self._generate(clip, frame, rate, screen=True)
            if scripts and _is_generated(clip)
            else self._image_for(clip, frame, rate)
        )
        if image is None:
            return

        if scripts:
            # スクリプトは渡した絵を書き換えることがある 覚えておいた絵は渡さない
            # 生成オブジェクトは画面と同じ大きさで作るので、AviUtl と同じ「自分の大きさ」
            # （obj.w / obj.h）になるよう、色の付いた所だけを切り出して渡す
            cropped, offset = (
                _object_sized(image, self._object_box_of(clip, image))
                if _is_generated(clip)
                else (image, (0.0, 0.0))
            )
            self._draw_scripted(
                track,
                clip,
                cropped.copy(),
                gpu_effects,
                local_frame,
                rate,
                opacity,
                offset=offset,
                below=below,
            )
            return

        texture = self._texture_for(track.id)
        texture.upload(image)

        screen_width, screen_height = self._compositor.width, self._compositor.height
        if _is_generated(clip) and (texture.width > screen_width or texture.height > screen_height):
            # 画面より大きく作った生成オブジェクト 縮めて収めず、画面の中心に等倍で置く
            self._draw_oversized(texture, image, clip, gpu_effects, local_frame, rate, opacity)
            return

        placement = self._media_placement(clip, texture)
        if not self._effects.has_work(gpu_effects):
            # エフェクトが無ければ中間バッファを通さない 全画面のパスが 1 回
            # 増えるだけで、エフェクト無しのクリップでも再生の余裕が削られる
            self._compositor.draw(
                texture, placement=placement, opacity=opacity, blend=clip.blend_mode
            )
            return

        result = self._effects.apply(
            texture,
            gpu_effects,
            frame=local_frame,
            fps=float(rate.fps),
            source_rect=placement.to_clip(self._compositor.width, self._compositor.height),
            duration=clip.duration,
            bounds=_placed_bounds(self._object_box_of(clip, image), image, placement),
        )
        # エフェクトを通した結果は画面いっぱいで GL の向き 収め直しも反転も要らない
        self._compositor.draw_handle(
            result.color,
            Placement(0.0, 0.0, float(self._compositor.width), float(self._compositor.height)),
            opacity=opacity,
            flip=False,
            blend=clip.blend_mode,
        )

    def _media_placement(self, clip: Clip, texture: Texture) -> Placement:
        """絵を置く矩形（合成先の画素）

        ``native_size`` のクリップは素材の画素の大きさで画面の中央に置く（YMM4・AviUtl の
        拡大率 100%） 大きさは届いた絵ではなく**素材の解像度**から出す 控え（プロキシ）は
        縮めて作るので、届いた絵の大きさで置くと控えの有無で大きさが変わる プレビューの
        解像度を落としているときは、プロジェクトの解像度に対する割合で縮める
        それ以外（前の版のクリップ・生成オブジェクト）は縦横比を保って画面に収める
        """
        width, height = self._compositor.width, self._compositor.height
        size = self._native_size(clip) if clip.native_size else None
        if size is None:
            return fit_placement(texture.width, texture.height, width, height)
        project_width, project_height = self._project.settings.resolution
        placed_width = size[0] * width / max(project_width, 1)
        placed_height = size[1] * height / max(project_height, 1)
        return Placement(
            (width - placed_width) / 2.0,
            (height - placed_height) / 2.0,
            placed_width,
            placed_height,
        )

    def _native_size(self, clip: Clip) -> tuple[int, int] | None:
        """素材の映像の、回転を当てた後の画素の大きさ 分からなければ ``None``

        外枠の計算（:func:`media_pixel_size`）と同じ物 別々に持つと枠と絵の大きさが食い違う
        """
        return media_pixel_size(self._project, clip)

    def object_extent(
        self, clip: Clip, frame: int
    ) -> tuple[tuple[float, float, float, float], tuple[int, int]] | None:
        """生成オブジェクトの入れ物（画素 左・上・右・下）と絵の大きさ 外枠を出すため

        描く道と同じ入れ物（:meth:`_object_box_of` 文字の枠も含む）を返す 画面より大きい絵
        （:meth:`_draw_oversized`）も同じ はみ出す長さのテキストだけ入れ物が変わらないように

        **GL を使わない** 絵は CPU で作り、同じ設定なら描いたときに覚えた物を使う 別のスレッドの
        先読みが出した絵には、画面の側のレンダラはまだ何も作っていない 選んだクリップの
        1 本ぶんだけここで作る（テキスト 1 枚で数ミリ秒 動かない字幕なら次からは覚えた物）
        """
        if not is_generated(clip):
            return None
        if split_effects(clip.effects)[1]:
            # スクリプトを積んだ物は、描く側が画面の画素で作った絵を使う 合成の画素で作り直すと、
            # 選んでいる間は描くたびに 2 つの大きさを作り直すことになる 枠はスクリプト次第で
            # 近い値でしかないので、画面の画素の入れ物を合成の画素へ縮めて返す
            return self._screen_extent(clip, frame)
        image = self._generate(clip, frame, self._project.rate)
        if image is None:
            return None
        height, width = int(image.shape[0]), int(image.shape[1])
        box = self._object_box_of(clip, image)
        if box is None:
            return None
        return box, (width, height)

    def _screen_extent(
        self, clip: Clip, frame: int
    ) -> tuple[tuple[float, float, float, float], tuple[int, int]] | None:
        """:meth:`object_extent` を画面の画素で作った絵から求め、合成の画素へ縮めた物"""
        image = self._generate(clip, frame, self._project.rate, screen=True)
        if image is None:
            return None
        height, width = int(image.shape[0]), int(image.shape[1])
        box = self._object_box_of(clip, image)
        if box is None:
            return None
        scale_x, scale_y = self._scale
        return (
            (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y),
            (max(1, round(width * scale_x)), max(1, round(height * scale_y))),
        )

    def _layer(self, role: str, depth: int) -> Compositor:
        """クリップ 1 本ぶんを描く透明な合成先 役目と入れ子の深さごとに使い回す"""
        width, height = self._compositor.width, self._compositor.height
        key = (role, depth)
        layer = self._layers.get(key)
        if layer is None:
            layer = Compositor(width, height, encoded=self._encoded)
            self._layers[key] = layer
        layer.resize(width, height)
        return layer

    def _draw_into(
        self,
        layer: Compositor,
        track: Track,
        clip: Clip,
        frame: int,
        rate: FrameRate,
        depth: int,
        *,
        trail: bool = False,
    ) -> None:
        """``clip`` だけを空の ``layer`` へ描く ``trail`` なら残像（:meth:`_draw_trail`）も描く"""
        outer = self._compositor
        self._compositor = layer
        try:
            layer.begin((0.0, 0.0, 0.0, 0.0))
            if trail:
                self._draw_trail(track, clip, frame, rate, depth)
            # 描く先は空の layer に切り替えてある スクリプトが画面（frm）として写すのは
            # 下の絵を溜めた outer 空の方を写すと、下の絵ではなく黒を写す
            self._draw_clip(track, clip, frame, rate, depth, below=outer)
        finally:
            self._compositor = outer

    def _capture(
        self, track: Track, clip: Clip, frame: int, rate: FrameRate, depth: int
    ) -> Compositor:
        """上のクリップに切り抜きの形として使わせるため、このクリップだけを別に描く

        **不透明度は当てない** 切り抜く形は、その絵が下へ重なるときの薄さではなく
        形そのもの 当てると、薄い形で切り抜かれた上のクリップまで同じだけ薄くなる
        （木製看板テロップは、板の下に不透明度 56.4%（YMM4 の設定の値）の陰の図形を敷いている
        当てていたころは板が 56.4% しか出ず、下の木枠が透けて見えていた）
        不透明度は、この形で切り抜いたクリップ自身を描くときに当たる
        """
        layer = self._layer("below", depth)
        shape = replace(clip, opacity=AnimatedValue(1.0))
        self._draw_into(layer, track, shape, frame, rate, depth)
        return layer

    def _draw_clipped(
        self,
        track: Track,
        clip: Clip,
        frame: int,
        rate: FrameRate,
        depth: int,
        below: Compositor | None,
    ) -> None:
        """すぐ下のクリップの形で切り抜いて重ねる（YMM4 の「上のオブジェクトでクリッピング」）

        下にクリップが無ければ、切り抜く形が無いので何も見えない 合成モードは、切り抜く前の
        透明な合成先で当てる（下の絵とは通常で重ねる）
        """
        if below is None:
            return
        layer = self._layer("clipped", depth)
        self._draw_into(layer, track, clip, frame, rate, depth)
        full = Placement(0.0, 0.0, float(layer.width), float(layer.height))
        layer.draw_handle(
            below.canvas.color, full, flip=False, blend=BlendMode.MASK, premultiplied=True
        )
        self._compositor.draw_handle(layer.canvas.color, full, flip=False, premultiplied=True)

    def _draw_oversized(
        self,
        texture: Texture,
        image: np.ndarray,
        clip: Clip,
        gpu_effects: tuple[Effect, ...],
        local_frame: int,
        rate: FrameRate,
        opacity: float,
    ) -> None:
        """画面より大きい絵を、画面の中心に等倍で置く

        エフェクトは絵と同じ大きさのバッファで掛ける 画面の大きさのバッファへ先に
        置くと、そこで端が切れ、あとから回したり動かしたりしたときに切れ目が見える
        バッファの中心は画面の中心と同じなので、位置の計算（画素）は変わらない
        """
        screen_width, screen_height = self._compositor.width, self._compositor.height
        placed = Placement(
            (screen_width - texture.width) / 2.0,
            (screen_height - texture.height) / 2.0,
            float(texture.width),
            float(texture.height),
        )
        if not self._effects.has_work(gpu_effects):
            self._compositor.draw(texture, placement=placed, opacity=opacity, blend=clip.blend_mode)
            return
        self._effects.resize(texture.width, texture.height)
        try:
            result = self._effects.apply(
                texture,
                gpu_effects,
                frame=local_frame,
                fps=float(rate.fps),
                duration=clip.duration,
                # 画面に収まる絵と同じ入れ物（AviUtl2 の組み方のテキストなら文字の枠も入れる）
                # 色の付いた範囲だけにすると、はみ出す長さの字だけ効果の基準が変わる
                bounds=self._object_box_of(clip, image),
            )
            self._compositor.draw_handle(
                result.color, placed, opacity=opacity, flip=False, blend=clip.blend_mode
            )
        finally:
            self._effects.resize(screen_width, screen_height)

    def _draw_scene(
        self, track: Track, clip: Clip, frame: int, rate: FrameRate, depth: int
    ) -> None:
        """入れ子のシーンを別の合成先に描き、1 本のクリップとして重ねる

        シーンの中の時刻は、クリップの ``source_in``（秒）と速度で決まる 素材の
        クリップと同じ決まりなので、分割やトリムをしても中身がずれない

        シーンのキャンバスは透明から始める 黒で始めると、シーンを重ねた場所の
        下の絵が隠れる
        """
        scene = self._project.find_scene(clip.scene_id) if clip.scene_id else None
        if scene is None or depth >= MAX_SCENE_DEPTH:
            return
        # 秒のまま足してから 1 度だけフレームへ落とす 別々に切り捨てると端数が 2 回落ち、
        # 1 フレーム前の絵になることがある 絵を止めた時刻も素材のクリップと同じく効かせる
        local_frame = frame - clip.timeline_start
        scene_frame = seconds_to_frame(clip.picture_time(local_frame, rate), rate)

        nested = self._nested_layer(depth + 1)
        outer = self._compositor
        self._compositor = nested
        try:
            nested.begin((0.0, 0.0, 0.0, 0.0))
            self._compose_timeline(scene.timeline, scene_frame, depth=depth + 1)
        finally:
            self._compositor = outer
        self._draw_nested(track, clip, nested, local_frame, rate)

    def _nested_layer(self, depth: int) -> Compositor:
        """入れ子の絵（シーン・1 枚にまとめるグループ制御）を描く合成先 深さごとに使い回す"""
        width, height = self._compositor.width, self._compositor.height
        nested = self._nested.get(depth)
        if nested is None:
            nested = Compositor(width, height, encoded=self._encoded)
            self._nested[depth] = nested
        nested.resize(width, height)
        return nested

    def _draw_group(
        self,
        track: Track,
        group: Clip,
        targets: list[Track],
        groups: list[tuple[Track, Clip]],
        frame: int,
        rate: FrameRate,
        depth: int,
    ) -> None:
        """「1 枚の絵として扱う」グループ制御 受け持つトラックを 1 枚に重ねてから掛ける

        入れ子のシーンと同じ道で描く（透明から始めた合成先へ重ね、その絵へエフェクトと
        不透明度を掛けて外へ重ねる） 1 本ずつに掛けると、重なった半透明どうしが透けて、
        下の物が上の物越しに見える
        """
        if depth >= MAX_SCENE_DEPTH or not targets:
            return
        nested = self._nested_layer(depth + 1)
        outer = self._compositor
        self._compositor = nested
        try:
            nested.begin((0.0, 0.0, 0.0, 0.0))
            self._compose_tracks(targets, frame, depth + 1)
        finally:
            self._compositor = outer
        group = grouped(group, groups, frame)
        self._draw_nested(track, group, nested, frame - group.timeline_start, rate)

    def _draw_nested(
        self, track: Track, clip: Clip, nested: Compositor, local_frame: int, rate: FrameRate
    ) -> None:
        """入れ子に描いた 1 枚の絵へ、`clip` のエフェクトと不透明度を掛けて今の合成先へ重ねる"""
        outer = self._compositor
        width, height = outer.width, outer.height
        gpu_effects, scripts = split_effects(clip.effects)
        full = Placement(0.0, 0.0, float(width), float(height))
        opacity = clip.opacity.at(local_frame)
        if scripts:
            # スクリプトは CPU の画像を書き換える作り シーンの絵を 1 枚読み戻して渡す
            # （毎フレームの往復になるので、スクリプトを積んだシーンだけで行う）
            # ストレートアルファで読む 事前乗算のまま渡すと、シーンの半透明の所が暗くなる
            self._draw_scripted(
                track,
                clip,
                nested.read(straight=True),
                gpu_effects,
                local_frame,
                rate,
                opacity,
                on_canvas=True,
            )
            return
        if not self._effects.has_work(gpu_effects):
            outer.draw_handle(
                nested.canvas.color,
                full,
                opacity=opacity,
                flip=False,
                blend=clip.blend_mode,
                premultiplied=True,
            )
            return
        result = self._effects.apply(
            nested.canvas,
            gpu_effects,
            frame=local_frame,
            fps=float(rate.fps),
            flip_source=False,
            duration=clip.duration,
            premultiplied=True,
        )
        outer.draw_handle(result.color, full, opacity=opacity, flip=False, blend=clip.blend_mode)

    def _draw_framebuffer(
        self,
        track: Track,
        clip: Clip,
        frame: int,
        rate: FrameRate,
        depth: int = 0,
        *,
        below: Compositor | None = None,
    ) -> None:
        """それまでに重ねた画面を写し取り、エフェクトを掛けて重ねる

        キャンバスは描いている最中なので、そのまま読みながら同じキャンバスへ描くことは
        できない いったん別のバッファへ写す キャンバスは事前乗算アルファで溜まって
        いるので、そう伝えて渡す 合成は透明な下地から始まるので、伝えないと半透明の
        縁が暗くなる

        ``below`` は別の空の合成先へ描き分けているとき（:meth:`_draw_into`）の、下の絵を
        溜めた外側 いまの合成先は空なので、そちらを写すと黒い画面で全体を覆う

        AviUtl スクリプトを積んでいれば、写し取った画面を CPU へ読み戻して渡す
        毎フレームの往復になるので、積んでいるクリップだけで行う
        """
        local_frame = frame - clip.timeline_start
        gpu_effects, scripts = split_effects(clip.effects)
        width, height = self._compositor.width, self._compositor.height
        screen = below if below is not None else self._compositor
        with self._context:
            if self._grab is None:
                self._grab = Framebuffer(width, height)
            self._grab.resize(width, height)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, screen.canvas.handle)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._grab.handle)
            GL.glBlitFramebuffer(
                0, 0, width, height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
            )
            # 写し取った画面は黒の上に置いた絵にする YMM4 は何も無い所も不透明な黒として
            # 写すので、反転すると白くなる 透明のまま渡すと反転しても黒のまま残る
            # AviUtl2 は透明のまま写す（``transparent``） 半分に縮めて重ねた写しの黒い所が、
            # 下の四角を隠さなかった（#195）
            origin = clip.source
            if origin is None or origin.params.get("transparent") is not True:
                self._compositor.underlay((0.0, 0.0, 0.0, 1.0), target=self._grab)
            # 直前オブジェクトが写すのは、このとき写し取った絵（:meth:`_draw_previous`）
            # 後から写し直すと、自分が置いた縮めた写しまで入る（#195 の探り po05）
            seen = self._layer("framebuffer_seen", depth)
            seen.begin((0.0, 0.0, 0.0, 0.0))
            seen.draw_handle(
                self._grab.color,
                Placement(0.0, 0.0, float(width), float(height)),
                flip=False,
                premultiplied=True,
            )
            self._framebuffer_seen[depth] = clip.id
            if scripts:
                grabbed = self._layer("framebuffer", depth)
                grabbed.begin((0.0, 0.0, 0.0, 0.0))
                grabbed.draw_handle(
                    self._grab.color,
                    Placement(0.0, 0.0, float(width), float(height)),
                    flip=False,
                    premultiplied=True,
                )
                self._draw_scripted(
                    track,
                    clip,
                    grabbed.read(),
                    gpu_effects,
                    local_frame,
                    rate,
                    clip.opacity.at(local_frame),
                    on_canvas=True,
                )
                return
            source: Framebuffer = self._grab
            if self._effects.has_work(gpu_effects):
                source = self._effects.apply(
                    self._grab,
                    gpu_effects,
                    frame=local_frame,
                    fps=float(rate.fps),
                    flip_source=False,
                    duration=clip.duration,
                    premultiplied=True,
                )
            self._compositor.draw_handle(
                source.color,
                Placement(0.0, 0.0, float(width), float(height)),
                opacity=clip.opacity.at(local_frame),
                flip=False,
                blend=clip.blend_mode,
                # エフェクトを通した結果はストレートアルファ 写しただけなら事前乗算のまま
                premultiplied=source is self._grab,
            )

    def _draw_previous(
        self, track: Track, clip: Clip, frame: int, rate: FrameRate, depth: int
    ) -> None:
        """すぐ下に描いたクリップの絵を、自分の描画の欄で置き直す（AviUtl の 直前オブジェクト）

        AviUtl2 に描かせると、下の四角の位置は足されず自分の位置にだけ写り、下に掛けた
        単色化は写った（#195） 下のクリップの描画の欄（反転・配置）を外して画面の真ん中に
        描き、その絵を入れ子のシーンと同じく 1 枚の絵として自分のエフェクトへ渡す

        下の不透明度と合成方法も写さない AviUtl ではどちらも位置と同じ 標準描画 の項目で、
        写さなかった位置と同じ扱いにした AviUtl2 で測り、下を不透明度 50 にしても写しは
        不透明のまま、灰の下地の上で下を加算にしても写しは通常で重なった 下が上のオブジェクトで
        クリッピングされていても、写しは切る前の絵だった 下に何も無ければ何も描かない
        """
        before = self._drawn
        if not before:
            return
        below_track, below_clip = before[-1]
        picture = replace(
            below_clip,
            effects=tuple(
                effect
                for effect in below_clip.effects
                if not (effect.fixed and effect.kind in PICTURE_FIXED)
            ),
            opacity=AnimatedValue(1.0),
            blend_mode=BlendMode.NORMAL,
        )
        # 直前オブジェクトを重ねたときは下の写しをさらに写す 段ごとに合成先を分けないと、
        # 写している最中の絵へ下の写しを描き込んでしまう
        layer = self._layer(f"previous{len(before)}", depth)
        if (
            below_clip.source is not None
            and below_clip.source.kind == "framebuffer"
            and self._framebuffer_seen.get(depth) == below_clip.id
        ):
            # 下がフレームバッファなら、写し取ったときの絵をそのまま使う AviUtl2 で下の四角を
            # 写したフレームバッファ（拡大率 50・Y -250）を写すと、下の四角が元の大きさで
            # 自分の位置にだけ出た（#195 の探り po05） 描き直すと、フレームバッファが置いた
            # 縮めた写しまで写し取られて入る フレームバッファのエフェクトは掛けていない
            # （測っていない）
            layer.begin((0.0, 0.0, 0.0, 0.0))
            layer.draw_handle(
                self._layer("framebuffer_seen", depth).canvas.color,
                Placement(0.0, 0.0, float(layer.width), float(layer.height)),
                flip=False,
                premultiplied=True,
            )
        else:
            # 下のクリップにとっての「すぐ下」は、自分より 1 つ前まで
            self._drawn = before[:-1]
            try:
                # 残像は本体とは別に前のフレームを描いて作る ここで描かないと、下のクリップには
                # 付いている残像が写しにだけ無い
                self._draw_into(layer, below_track, picture, frame, rate, depth, trail=True)
            finally:
                self._drawn = before

        local_frame = frame - clip.timeline_start
        opacity = clip.opacity.at(local_frame)
        gpu_effects, scripts = split_effects(clip.effects)
        full = Placement(0.0, 0.0, float(layer.width), float(layer.height))
        if scripts:
            # スクリプトは CPU の画像を書き換える作り 写した絵を 1 枚読み戻して渡す
            # ストレートアルファで読む 事前乗算のまま渡すと、半透明の所が暗くなる
            self._draw_scripted(
                track,
                clip,
                layer.read(straight=True),
                gpu_effects,
                local_frame,
                rate,
                opacity,
                on_canvas=True,
            )
            return
        if not self._effects.has_work(gpu_effects):
            self._compositor.draw_handle(
                layer.canvas.color,
                full,
                opacity=opacity,
                flip=False,
                blend=clip.blend_mode,
                premultiplied=True,
            )
            return
        result = self._effects.apply(
            layer.canvas,
            gpu_effects,
            frame=local_frame,
            fps=float(rate.fps),
            flip_source=False,
            duration=clip.duration,
            premultiplied=True,
        )
        self._compositor.draw_handle(
            result.color, full, opacity=opacity, flip=False, blend=clip.blend_mode
        )

    def _draw_filter(
        self, track: Track, clip: Clip, frame: int, rate: FrameRate, depth: int
    ) -> None:
        """それまでに重ねた絵へエフェクトを掛けて、掛けた絵で置き換える（フィルタのクリップ）

        フレームバッファ（:meth:`_draw_framebuffer`）と違い、黒を敷かず上へも重ねない
        黒を敷くと入れ子のシーンの中で外の絵を隠し、上へ重ねると縮めたり動かしたりする
        エフェクトで元の絵が後ろに残る 不透明度は掛ける前と後の混ぜ具合で、
        合成方法は使わない（AviUtl のフィルタオブジェクトにも無い）

        掛けるのはいまの合成先に溜まった絵だけ シーンの中なら、そのシーンの下の段だけに効く
        """
        gpu_effects, scripts = split_effects(clip.effects)
        if not scripts and not self._effects.has_work(gpu_effects):
            return
        local_frame = frame - clip.timeline_start
        opacity = clip.opacity.at(local_frame)
        if opacity <= 0.0:
            return
        outer = self._compositor
        width, height = outer.width, outer.height
        full = Placement(0.0, 0.0, float(width), float(height))
        # 描いている最中のキャンバスは、読みながら同じ所へ描けない 先に別の合成先へ写す
        before = self._layer("filter_before", depth)
        before.begin((0.0, 0.0, 0.0, 0.0))
        before.draw_handle(outer.canvas.color, full, flip=False, premultiplied=True)
        after = self._layer("filter_after", depth)
        after.begin((0.0, 0.0, 0.0, 0.0))
        if scripts:
            # スクリプトは CPU の画像を書き換える作り 写した絵を 1 枚読み戻して渡す
            # 不透明度はここでは当てない 置き換えるときに混ぜ具合として 1 度だけ当てる
            # スクリプトの絵は素材と同じストレートアルファとして重なる 事前乗算のまま
            # 渡すと、半透明の所で不透明度が 2 回掛かって暗くなる
            self._compositor = after
            try:
                self._draw_scripted(
                    track,
                    replace(clip, blend_mode=BlendMode.NORMAL),
                    before.read(straight=True),
                    gpu_effects,
                    local_frame,
                    rate,
                    1.0,
                    on_canvas=True,
                    # 合成先は空の after に切り替えてある 画面として写すのは下の絵を
                    # 溜めた outer 空の方を写すと黒一色の絵で下の絵を置き換える
                    below=outer,
                )
            finally:
                self._compositor = outer
        else:
            result = self._effects.apply(
                before.canvas,
                gpu_effects,
                frame=local_frame,
                fps=float(rate.fps),
                flip_source=False,
                duration=clip.duration,
                premultiplied=True,
            )
            after.draw_handle(result.color, full, flip=False)
        outer.draw_handle(
            after.canvas.color,
            full,
            opacity=opacity,
            flip=False,
            blend=BlendMode.REPLACE,
            premultiplied=True,
        )

    def _draw_scripted(
        self,
        track: Track,
        clip: Clip,
        image: np.ndarray,
        gpu_effects: tuple[Effect, ...],
        local_frame: int,
        rate: FrameRate,
        opacity: float,
        offset: tuple[float, float] = (0.0, 0.0),
        *,
        on_canvas: bool = False,
        below: Compositor | None = None,
    ) -> None:
        """AviUtl スクリプトを積んだクリップを描く

        ``below`` は ``obj.copybuffer`` の ``frm`` で読む合成先（:meth:`_screen_picture`）

        スクリプトは「何回・どこへ・どう変形して描くか」を返す 1 回とは限らない
        （残像や複製を作るスクリプトがある）ので、返ってきた分だけ合成する

        スクリプトは**画面（プロジェクトの解像度）の画素**で動かす 画質を落としたプレビューでも、
        スクリプトが見る画面の大きさ・絵の大きさ（obj.w）・トラックバーの値は書き出しと同じで、
        返ってきた位置と大きさを描く直前に合成の画素へ縮める スクリプトの値はどれが画素かを
        定義から読めない（``--track@`` に単位が無い）ので、値ではなく結果の方を縮める
        ``on_canvas`` は ``image`` が合成の画素の絵（写し取った画面・シーン）のとき 画面の
        画素へ引き伸ばしてから渡す
        """
        stage = self._script_stage()
        if stage is None:
            return

        scale_x, scale_y = self._scale
        if on_canvas and (scale_x, scale_y) != (1.0, 1.0):
            image = _resized(image, self._project.settings.resolution)
        calls = stage.run(
            clip,
            script_effects(clip.effects),
            image,
            frame=local_frame,
            fps=float(rate.fps),
            layer=self._layer_number(clip, track),
            framebuffer=lambda: self._screen_picture(below),
        )
        self._draw_calls(
            self._texture_for(track.id),
            clip,
            calls,
            gpu_effects,
            local_frame,
            rate,
            opacity,
            offset,
        )

    def _draw_calls(
        self,
        texture: Texture,
        clip: Clip,
        calls: tuple[DrawCall, ...],
        gpu_effects: tuple[Effect, ...],
        local_frame: int,
        rate: FrameRate,
        opacity: float,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """スクリプトが返した描画を、いまの合成先へ順に重ねる

        位置と大きさは画面の画素で返ってくるので、描く直前に合成の画素へ縮める
        （:meth:`_draw_scripted`）
        """
        scale_x, scale_y = self._scale
        screen_width, screen_height = self._project.settings.resolution

        def shrunk(point: tuple[float, float]) -> tuple[float, float]:
            return point[0] * scale_x, point[1] * scale_y

        # スクリプトが obj.effect で頼んだ効果は、置き場所を決める固定の欄（配置・反転）より
        # 先に掛ける AviUtl でもスクリプトの中の効果は 標準描画 の前に掛かる 後に掛けると、
        # 動かした後の絵を切ることになり、位置を変えた菱形が斜めの切り落としで消えていた
        leading = tuple(effect for effect in gpu_effects if not effect.fixed)
        placing = tuple(effect for effect in gpu_effects if effect.fixed)
        for call in calls:
            texture.upload(call.image)
            combined = leading + requested_effects(call) + placing
            alpha = opacity * call.alpha

            if call.quad is not None:
                shifted = tuple((x + offset[0], y + offset[1], z) for x, y, z in call.quad)
                # 遠近は画面の画素で写してから縮める カメラの距離は画面の画素で決まっている
                projected = [project(point, screen_width, screen_height) for point in shifted]
                points = [shrunk(point) for point in projected if point is not None]
                if len(points) != 4:
                    # カメラを越えた隅がある 写せる隅だけで描くと形の違う板になる
                    continue
                uv = None
                if call.uv is not None:
                    uv = tuple(
                        (u / max(texture.width, 1), v / max(texture.height, 1)) for u, v in call.uv
                    )
                self._draw_on_quad(
                    texture,
                    (points[0], points[1], points[2], points[3]),
                    _as_corners(uv),
                    combined,
                    local_frame,
                    rate,
                    alpha,
                    clip.blend_mode,
                    clip.duration,
                    _content_box(call.image) if combined else None,
                )
                continue

            transform = Transform(
                x=call.x + offset[0],
                y=call.y + offset[1],
                zoom=call.zoom,
                rotation=call.rz,
                aspect=call.aspect,
                pivot_x=call.cx,
                pivot_y=call.cy,
                scale_x=call.sx,
                scale_y=call.sy,
                z=call.z,
                rotation_x=call.rx,
                rotation_y=call.ry,
            )
            if not transform.is_flat:
                corners = transform.corners(
                    texture.width, texture.height, screen_width, screen_height
                )
                if corners is None:
                    continue
                self._draw_on_quad(
                    texture,
                    (
                        shrunk(corners[0]),
                        shrunk(corners[1]),
                        shrunk(corners[2]),
                        shrunk(corners[3]),
                    ),
                    None,
                    combined,
                    local_frame,
                    rate,
                    alpha,
                    clip.blend_mode,
                    clip.duration,
                    _content_box(call.image) if combined else None,
                )
                continue

            placed = transform.placement(texture.width, texture.height, screen_width, screen_height)
            placement = Placement(
                placed.left * scale_x,
                placed.top * scale_y,
                placed.width * scale_x,
                placed.height * scale_y,
            )
            # 回転の行列はクリップ空間（画面の幅と高さを 1 にした物差し）で決まるので、
            # 画面の画素で求めた物がそのまま小さい合成にも当たる
            matrix = transform.matrix(screen_width, screen_height)

            if not self._effects.has_work(combined):
                self._compositor.draw(
                    texture,
                    placement=placement,
                    opacity=opacity * call.alpha,
                    blend=clip.blend_mode,
                    matrix=matrix,
                )
                continue

            result = self._effects.apply(
                texture,
                combined,
                frame=local_frame,
                fps=float(rate.fps),
                source_rect=placement.to_clip(self._compositor.width, self._compositor.height),
                duration=clip.duration,
                bounds=_placed_bounds(_content_box(call.image), call.image, placement),
            )
            self._compositor.draw_handle(
                result.color,
                Placement(0.0, 0.0, float(self._compositor.width), float(self._compositor.height)),
                opacity=opacity * call.alpha,
                flip=False,
                blend=clip.blend_mode,
                matrix=matrix,
            )

    def _layer_number(self, clip: Clip, track: Track | None = None) -> int:
        """スクリプトへ渡す ``obj.layer`` 絵を描くトラックの奥から何本目か（1 から）

        混合トラックでは AviUtl のレイヤー番号と同じ（:mod:`sashimono.compat.layers`） 0 のまま
        渡すと、PSDToolKit のようにレイヤー番号で字幕や口パクの状態を分けるスクリプトで、
        別のレイヤーのオブジェクトどうしが同じ状態を書き換え合う
        どこにも無いクリップ（切り替えの途中に作る物など）は 1

        ``track`` は置かれたトラック 分かる所では渡す フィルタや切り抜きは描く前にクリップを
        作り直す（``replace``）ので、実体で引けずに識別子へ落ち、別のシーンに同じ識別子の
        クリップがあるとそちらの番号になる トラックはタイムラインの物をそのまま渡している
        """
        cached = self._layer_numbers
        if cached is None or cached[0] is not self._project:
            # クリップの実体で引く 識別子だけで引くと、別のシーンに同じ識別子のクリップがある
            # プロジェクト（識別子はタイムラインをまたいで重ならないと確かめていない）で、後から
            # 数えたシーンの番号が前の番号を上書きする 実体はプロジェクトを持っている間は
            # 生きているので ``id()`` が変わらない 描くときに作り直したクリップは識別子で引く
            # （そのときは先に数えたメインのタイムラインの番号）
            by_object: dict[int, int] = {}
            by_id: dict[ClipId, int] = {}
            timelines = (self._project.timeline, *(s.timeline for s in self._project.scenes))
            for timeline in timelines:
                # 絵を描くトラック（映像と混合）をまとめて奥から数える 種類ごとに数えると、
                # 映像と混合のトラックが並ぶタイムラインで両方に同じ番号が渡り、番号で状態を
                # 分けるスクリプトで別のトラックの状態が混ざる 混合だけなら今までと同じ番号
                for layer, placing in enumerate(timeline.picture_tracks(), 1):
                    by_object[id(placing)] = layer
                    for placed in placing.clips:
                        by_object[id(placed)] = layer
                        by_id.setdefault(placed.id, layer)
            cached = (self._project, by_object, by_id)
            self._layer_numbers = cached
        _, by_object, by_id = cached
        if track is not None and id(track) in by_object:
            return by_object[id(track)]
        return by_object.get(id(clip), by_id.get(clip.id, 1))

    def _screen_picture(self, below: Compositor | None = None) -> np.ndarray:
        """それまでに重ねた画面を、スクリプトへ渡す絵にする（``obj.copybuffer`` の ``frm``）

        ``below`` は読む合成先 省けばいまの合成先 フィルタのクリップは掛けた結果を空の
        合成先へ描くので、下の絵を溜めた外側の合成先を渡す（:meth:`_draw_filter`）

        AviUtl のフレームバッファは何も描いていない所も不透明な黒 :meth:`_draw_framebuffer`
        と同じく黒を敷いて写す 透明のまま渡すと、アクリル矩形のように下の絵をぼかして
        透かす板が、何も無い所で透明になって消える

        描いている最中の合成先は読みながら同じ所へ描けないので、いったん別の所へ写す
        大きさは画面の画素 画質を落としたプレビューでも、スクリプトが切り出す量
        （``obj.w`` から決める）は書き出しと同じにしておく（:meth:`_draw_scripted`）
        """
        source = below if below is not None else self._compositor
        width, height = source.width, source.height
        with self._context:
            if self._grab is None:
                self._grab = Framebuffer(width, height)
            self._grab.resize(width, height)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, source.canvas.handle)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._grab.handle)
            GL.glBlitFramebuffer(
                0, 0, width, height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
            )
            source.underlay((0.0, 0.0, 0.0, 1.0), target=self._grab)
            copied = self._layer("script_framebuffer", 0)
            copied.begin((0.0, 0.0, 0.0, 0.0))
            copied.draw_handle(
                self._grab.color,
                Placement(0.0, 0.0, float(width), float(height)),
                flip=False,
                premultiplied=True,
            )
            picture = copied.read()
        if self._scale != (1.0, 1.0):
            picture = _resized(picture, self._project.settings.resolution)
        return picture

    def _draw_on_quad(
        self,
        texture: Texture,
        corners: Corners,
        uv: Corners | None,
        effects: tuple[Effect, ...],
        local_frame: int,
        rate: FrameRate,
        opacity: float,
        blend: str,
        duration: int = 0,
        box: tuple[int, int, int, int] | None = None,
    ) -> None:
        """絵を四角形へ貼る エフェクトがあれば、先に平らなまま掛けてから貼る

        エフェクトは画面と同じ大きさのバッファで動く 傾けてから掛けると、
        ぼかしや縁取りの幅まで遠近で歪む 平らな板に掛けてから板ごと傾けるのが
        AviUtl の見え方と同じ

        絵は画面の画素の大きさで届く（スクリプトの絵） 画質を落としていれば、バッファへは
        合成の画素へ縮めて置く 縮めずに置くと、エフェクトの掛かり方が絵に対して半分になり、
        小さいバッファから絵がはみ出して切れる
        """
        width, height = self._compositor.width, self._compositor.height
        own = Placement(0.0, 0.0, float(texture.width), float(texture.height))
        if not self._effects.has_work(effects):
            self._compositor.draw_mapped(
                texture.handle,
                source=own,
                anchor=own,
                corners=corners,
                opacity=opacity,
                blend=blend,
                uv=uv,
            )
            return

        scale_x, scale_y = self._scale
        placed_width, placed_height = texture.width * scale_x, texture.height * scale_y
        centred = Placement(
            (width - placed_width) / 2.0,
            (height - placed_height) / 2.0,
            placed_width,
            placed_height,
        )
        result = self._effects.apply(
            texture,
            effects,
            frame=local_frame,
            fps=float(rate.fps),
            source_rect=centred.to_clip(width, height),
            duration=duration,
            bounds=_placed_bounds(box, None, centred, texture.width, texture.height),
        )
        if uv is not None:
            # 絵の一部だけを貼るときは、切り出す四隅を結果のバッファの中の位置へ
            # 読み替えて貼る 外接矩形にまとめると、斜めに切り出した形が崩れる
            # 結果のバッファは GL の向き（下が 0）なので、縦は裏返す
            placed = tuple(
                (
                    (centred.left + u * centred.width) / width,
                    1.0 - (centred.top + v * centred.height) / height,
                )
                for u, v in uv
            )
            self._compositor.draw_mapped(
                result.color,
                source=own,
                anchor=own,
                corners=corners,
                opacity=opacity,
                blend=blend,
                uv=_as_corners(placed),
            )
            return
        self._compositor.draw_mapped(
            result.color,
            source=Placement(0.0, 0.0, float(width), float(height)),
            anchor=centred,
            corners=corners,
            opacity=opacity,
            flip=False,
            blend=blend,
        )

    def _script_stage(self) -> ScriptStage | None:
        """スクリプトを走らせる係 初めて必要になったときに作る

        Lua ランタイムの用意は数ミリ秒かかる スクリプトを使わない
        プロジェクトでその代金を払わせない
        """
        if self._scripts is None:
            self._scripts = ScriptStage(
                script_catalog(),
                screen=self._project.settings.resolution,
                apply_effects=self._script_baker.apply,
            )
        else:
            # スクリプトへ見せる画面は画質に関わらずプロジェクトの解像度（:meth:`_draw_scripted`）
            self._scripts.set_screen(*self._project.settings.resolution)
        return self._scripts

    def _waveform_audio(
        self, clip: Clip, source: GeneratedSource, local_frame: int, rate: FrameRate
    ) -> tuple[np.ndarray, int] | None:
        """音声波形が描く音と、そのレート 1 チャンネルで、今の時刻から

        線は横幅ぶん、スペクトラムは頭の :func:`spectrum_window` ぶんを使う
        サンプルはプロジェクトの音のレート（AviUtl2 は 44.1kHz の書き出しで 1 画素
        1 サンプルだった） クリップの外（頭より前・終わりより先）は 0 実物も最後の
        フレームでは、終わりから先が平らな線になっていた（素材の続きを読むと、そこに音が出る）
        """
        if source.kind != "shape" or source.params.get("shape") != "waveform":
            return None
        decoder = self._audio_decoder_for(clip, source)
        if decoder is None:
            return None
        sample_rate = decoder.sample_rate
        seconds = clip.source_in + local_frame * rate.frame_duration * clip.speed
        end_ms = source.params.get("audio_end_ms")
        remaining: Fraction | None = None
        if isinstance(end_ms, int) and not isinstance(end_ms, bool) and end_ms >= 0:
            remaining = (Fraction(end_ms, 1000) - seconds) * sample_rate / clip.speed
            if remaining <= 0:
                # 読む範囲（AviUtl の 再生範囲）をもう過ぎている 実物は平らな線も出さず、
                # 何も描かなかった（再生位置を 10,10 にした見本が 81 フレームとも真っ黒）
                return None

        width = source.params.get("width")
        # 横幅は素の数でも持てる（設定の決まり） 数を既定の 800 に置き換えると、
        # 描く幅より読む音が短くなり、右側が平らな線になる
        if isinstance(width, AnimatedValue):
            wide = width.at(local_frame)
        elif isinstance(width, int | float) and not isinstance(width, bool):
            wide = float(width)
        else:
            wide = 800.0
        # 描く大きさの上限より多くは読まない 壊れた横幅で何億サンプルも読んで止まらないように
        span = max(1, round(min(wide, float(MAX_CANVAS)))) if math.isfinite(wide) else 800
        count = max(span, spectrum_window(sample_rate))
        speed = float(clip.speed)
        # 読み始めは時刻の少し前 デコーダはシークした所からリサンプラを作り直すので、
        # 読み始めの位置で同じ時刻のサンプルが僅かに変わる（この曲で振幅 0.01〜0.05）
        # 前と同じ位置から読み、実物と突き合わせた線の絵を変えない（時刻ちょうどから
        # 読むと、76 フレーム目の差が 0.50 から 0.78 へ開いた）
        lead = int(_WAVEFORM_PREROLL * speed)
        first = int(seconds * sample_rate) - lead
        raw = decoder.read(first, lead + int(np.ceil(count * speed)) + 1)[lead:]
        # float32 のまま平均する 既定の float64 にすると、毎フレーム型を変えて配列を作り直す
        mono = raw.mean(axis=1, dtype=np.float32) if raw.ndim == 2 else raw
        # 速く回すと 1 画素に何サンプルも入る 間引いて 1 画素 1 サンプルに合わせる
        picked = (np.arange(count) * speed).astype(int).clip(0, max(len(mono) - 1, 0))
        samples = mono[picked].astype(np.float32) if len(mono) else np.zeros(count, np.float32)
        # 今の時刻からの位置（サンプル） クリップの終わりより先を 0 にする
        # 窓は今の時刻から先だけを見るので、クリップの頭より前は読まない
        offset = np.arange(count)
        after = float((clip.duration - local_frame) * rate.frame_duration * sample_rate)
        samples[offset >= round(after)] = 0.0
        if remaining is not None:
            samples[offset >= int(np.ceil(float(remaining)))] = 0.0
        return samples, sample_rate

    def _audio_decoder_for(self, clip: Clip, source: GeneratedSource) -> AudioDecoder | None:
        """音声波形の音を読む係 素材を持つクリップならその素材、無ければ設定の道

        素材とストリームの番号とレートの組で持つ 道だけで持つと、音声を複数持つ素材で
        2 本目を選んだクリップにも 1 本目の波形が出る
        """
        key = _waveform_key(self._project, clip, source)
        if key is None:
            return None
        decoder = self._audio.get(key)
        if decoder is not None:
            self._audio.move_to_end(key)
            return decoder
        if key in self._audio_missing:
            return None
        path, stream_index, rate = key
        stream = stream_index if stream_index else None
        try:
            decoder = AudioDecoder(path, sample_rate=rate, channels=2, stream_index=stream)
            if decoder.info.channels == 1:
                # モノラルの素材を 2 チャンネルへ広げると、変換が振幅を 0.7 倍に落とす
                # 素材のままの 1 チャンネルで読み、波形の振れ幅を変えない
                decoder.close()
                decoder = AudioDecoder(path, sample_rate=rate, channels=1, stream_index=stream)
        except (ProbeError, OSError):
            # 開けない音は何度も開き直さない 毎フレーム探しに行って再生が止まる
            # （プロジェクトを差し替えたときに忘れて、置き直した素材を読み直す）
            self._audio_missing.add(key)
            return None
        self._audio[key] = decoder
        while len(self._audio) > MAX_OPEN_DECODERS:
            _, evicted = self._audio.popitem(last=False)
            evicted.close()
        return decoder

    def _image_for(self, clip: Clip, frame: int, rate: FrameRate) -> np.ndarray | None:
        """クリップの元絵 素材由来と生成オブジェクトの両方をここで扱う"""
        if _is_generated(clip):
            return self._generate(clip, frame, rate)
        return self._decode(clip, frame, rate)

    def _generate(
        self, clip: Clip, frame: int, rate: FrameRate, *, screen: bool = False
    ) -> np.ndarray | None:
        """素材を持たないクリップ（テキスト・図形）の絵を作る

        ふつうは合成の画素で作る（画質を落としたプレビューでは縮めて作る）
        ``screen`` なら画面（プロジェクトの解像度）の画素で作る スクリプトへ渡す絵で、
        スクリプトは書き出しと同じ大きさの絵を相手に位置や大きさを数える
        """
        source = clip.source
        if source is None:
            return None
        local_frame = frame - clip.timeline_start
        text = source.params.get("text") if source.kind == "text" else None
        if isinstance(text, str) and has_embedded(text):
            # 埋め込んだ Lua は描くたびに走らせる 時刻で数え上げる字幕のように、
            # フレームごとに文字が変わるため
            stage = self._script_stage()
            if stage is not None:
                expanded = stage.expand_text(
                    text,
                    frame=local_frame,
                    fps=float(rate.fps),
                    duration=clip.duration,
                    # 設定欄の書体を ``obj.getfont`` から読めるようにする
                    # 渡さないと、合成フォントのエイリアスが大きさも字間も
                    # 既定値で組み、設定欄をいじっても絵が変わらない
                    font=text_font(source.params, local_frame),
                    layer=self._layer_number(clip),
                )
                source = source.with_param("text", expanded)
        # 画面の左上から数えた線の点（YMM4 のペン）は、ここで画面の大きさを引いて中心からの点へ
        # 直す 絵を描く所は広げた絵や縮めたプレビューの大きさしか知らず、画面の左上が分からない
        source = corner_points_centred(source, *self._project.settings.resolution)
        if screen:
            canvas, scale = self._project.settings.resolution, (1.0, 1.0)
        else:
            canvas, scale = (self._compositor.width, self._compositor.height), self._scale
        width, height = source_canvas(
            source, *canvas, frame=local_frame, scale=scale, fps=float(rate.fps)
        )
        # 同じ絵をもう一度作らない テキストは 1 枚で数ミリ秒かかり、動かない字幕を
        # 何本も重ねたタイムラインでは、そこが再生の足を引っ張る
        # 時間で変わらない絵は、フレームを鍵に入れない（毎フレーム作り直さない）
        when = local_frame if _varies_over_time(source) else -1
        # 長さも鍵に入れる 移動軌跡の先端の向きはクリップの終わりまでの動きで決まるので、
        # 伸び縮みさせただけのクリップに前の長さの絵を出さないように
        # 切り出し位置と速度は音声波形が読む音の位置を決める 変えたのに前の波形が出ないように
        key = (
            source.kind,
            width,
            height,
            scale,
            when,
            clip.duration,
            clip.source_in,
            clip.speed,
            # 音声波形は読む音（素材の道・ストリーム・レート）も鍵に入れる 素材を
            # 繋ぎ直したりストリームを選び直したりしても、前の音の波形が出ないように
            _waveform_key(self._project, clip, source)
            if source.params.get("shape") == "waveform"
            else None,
            _fingerprint(source.params),
        )
        cached = self._generated.get(clip.id)
        if cached is not None and cached.key == key:
            return cached.image
        heard = self._waveform_audio(clip, source, local_frame, rate)
        image, framed = render_source_framed(
            source,
            width,
            height,
            frame=local_frame,
            fps=float(rate.fps),
            duration=clip.duration,
            audio=heard[0] if heard is not None else None,
            audio_rate=heard[1] if heard is not None else 44100,
            trail_paths=self._trail_paths,
            scale=scale,
            screen=canvas,
        )
        if image is not None:
            self._generated.pop(clip.id, None)
            screen_width, screen_height = self._project.settings.resolution
            _make_room(self._generated, image, screen_width * screen_height * 4)
            self._generated[clip.id] = _Made(key, image, framed)
        return image

    def _content_box_of(self, clip: Clip, image: np.ndarray) -> _Box | None:
        """``image`` の色の付いた範囲 作って覚えた絵なら、1 度だけ探して覚える

        覚えるのは :attr:`_generated` が今も持っている絵に限る 素材から取り出した絵は
        毎フレーム別の物で、覚えても当たらないうえ、手放すまで画面 1 枚ぶん抱える
        作って覚えた絵はどこからも書き換えない（スクリプトへは写しを渡す）ので、
        同じ物なら範囲も同じ
        """
        made = self._generated.get(clip.id)
        if made is None or made.image is not image:
            return _content_box(image)
        if not made.scanned:
            made.box = _content_box(image)
            made.scanned = True
        return made.box

    def _object_box_of(self, clip: Clip, image: np.ndarray) -> Frame | None:
        """クリップの入れ物（:func:`_object_box`） 範囲は :meth:`_content_box_of` から取る"""
        return _merged_box(self._content_box_of(clip, image), self._frame_of(clip, image))

    def _frame_of(self, clip: Clip, image: np.ndarray) -> Frame | None:
        """いま描いた生成オブジェクトの文字の枠 無ければ ``None``

        覚えた絵と同じ物を描いているときだけ返す 違う絵（入れ替わった直後など）に
        前の枠を当てると、別の大きさの入れ物で効果を掛けてしまう
        """
        cached = self._generated.get(clip.id)
        if cached is None or cached.image is not image:
            return None
        return cached.framed

    def _decode(self, clip: Clip, frame: int, rate: FrameRate) -> np.ndarray | None:
        assert clip.media_id is not None
        media = self._project.find_media(clip.media_id)
        if media is None or not media.has_video:
            return None

        key = (clip.media_id, clip.stream_index)
        source_time = _source_time(clip, media, frame, rate)
        pending = self._decoding.pop(key, None)
        if pending is not None:
            when, future = pending
            # 必ず受け取ってから自分で触る 受け取らずに触ると、同じデコーダを
            # 2 スレッドが同時に進めることになり、絵が飛ぶ
            # 例外もここで投げ直す 握り潰すと、読めない素材が黒いまま書き出される
            image = future.result()
            if when == source_time:
                return image

        decoder = self._decoder_for(*key)
        if decoder is None:
            return None
        return decoder.frame_at(source_time)

    def _pool(self) -> ThreadPoolExecutor:
        """先読みの走り係 使うまで作らない"""
        if self._decode_pool is None:
            self._decode_pool = ThreadPoolExecutor(
                max_workers=self._decode_threads, thread_name_prefix="sashimono-decode"
            )
        return self._decode_pool

    def _decode_request(
        self, clip: Clip, frame: int, rate: FrameRate
    ) -> tuple[_DecodeKey, Fraction] | None:
        """``clip`` を描くときに素材から取り出すフレーム 先読みに向かないものは ``None``

        向かないのは、絵を素材から取らないもの（生成オブジェクト・入れ子のシーン・
        場面切り替え・画面の写し取り）と、**残像を積んだもの** 残像は同じデコーダへ
        前のフレームを先に頼むので、いまのフレームを先読みしても読み直しになり、
        戻る向きのシークまで増える
        """
        if clip.media_id is None or clip.scene_id is not None or _is_generated(clip):
            return None
        source = clip.source
        if source is not None and source.kind in ("transition", "framebuffer"):
            return None
        if any(effect.enabled and effect.kind == "after_image" for effect in clip.effects):
            return None
        media = self._project.find_media(clip.media_id)
        if media is None or not media.has_video:
            return None
        return (clip.media_id, clip.stream_index), _source_time(clip, media, frame, rate)

    def prime(self, frame: int) -> None:
        """``frame`` で映る素材のデコードを、描く前から裏で走らせておく GL は触らない

        先読みを別のスレッドで描くと（:mod:`sashimono.engine.render.background`）、
        画面の側のデコーダは、貯めた所を再生している間は止まったまま残る
        貯めた所の端を越えた 1 コマ目で、頭の鍵フレームから読み直すことになり、
        再生がそこで 1 度引っかかる（GOP 250 の 1080p を 3 枚重ねて 129ms）
        再生を始めたときに端のコマを頼んでおけば、端へ着くまでに読み進めておける

        受け取るのは描く側の :meth:`_decode` で、並列デコードと同じ仕組みを通る
        同じデコーダを 2 スレッドから触らない約束はそのまま守られる
        並列デコードを切っている（1 本）ときは何もしない 1 を選んだ人は、描く所の
        ほかでデコードが走ることを望んでいない
        """
        if self._closed:
            return
        rate = self._project.rate
        placed: list[tuple[Clip, int]] = []
        self._visible_clips(self._project.timeline, frame, rate, 0, placed)
        requests = [
            request
            for clip, at in placed
            if (request := self._decode_request(clip, at, rate)) is not None
        ]
        self._submit_decodes(requests, minimum=1)

    def _visible_clips(
        self,
        timeline: Timeline,
        frame: int,
        rate: FrameRate,
        depth: int,
        placed: list[tuple[Clip, int]],
    ) -> None:
        """``frame`` で映るクリップと、そのクリップの中で使うフレームを集める

        入れ子のシーンの中へも入る シーンの中は外と別の時刻で動くので、描く道
        （:meth:`_draw_scene`）と同じ式で中のフレームを出す 入らないと、シーンの中の
        素材は端を越えたところで鍵フレームから読み直し、頼んだ意味が無くなる
        深さの上限も描く道と同じ 描かない段のデコーダを開いても使われない
        """
        for track in timeline.active_picture_tracks():
            clip = track.clip_at(frame)
            # 描く道（:meth:`_compose_tracks`）と同じ物だけを頼む 描かないクリップの
            # デコーダを開くと、使われない絵のために走り係を 1 本塞ぐ
            if clip is None or not clip.enabled or not self._project.draws_picture(track, clip):
                continue
            if clip.scene_id is None:
                placed.append((clip, frame))
                continue
            scene = self._project.find_scene(clip.scene_id)
            if scene is None or depth >= MAX_SCENE_DEPTH:
                continue
            local_frame = frame - clip.timeline_start
            scene_frame = seconds_to_frame(clip.picture_time(local_frame, rate), rate)
            self._visible_clips(scene.timeline, scene_frame, rate, depth + 1, placed)

    def _prefetch_decodes(
        self, clips: list[Clip], frame: int, rate: FrameRate, *, minimum: int = 2
    ) -> None:
        """この段で描くクリップのデコードを、デコーダごとに分けて先に走らせる

        並べてよいのは**別々のデコーダ**の間だけなので、同じ鍵の 2 本目は出さない
        結果は変わらない :meth:`VideoDecoder.frame_at` は、いまの読み位置に関係なく
        同じ時刻には同じ絵を返す（届かなければシークして読み直す）
        先に走らせた絵が使われずに終わっても、捨てるだけで絵には出ない

        ``minimum`` 本より少なければ走らせない 描く直前なら、1 本だけ渡しても
        受け渡しの分だけ遅い 描くより前から頼む :meth:`prime` は 1 本でも渡す
        """
        if self._decode_threads <= 1:
            return
        requests = [
            request
            for clip in clips
            if (request := self._decode_request(clip, frame, rate)) is not None
        ]
        self._submit_decodes(requests, minimum=minimum)

    def _submit_decodes(
        self, candidates: list[tuple[_DecodeKey, Fraction]], *, minimum: int
    ) -> None:
        """頼みを走り係へ出す 同じ鍵は 1 本だけ、本数は走り係の数まで

        同じ鍵の 2 本目を出さないことが、同じデコーダを 2 スレッドから触らない保証
        入れ子のシーンの中と外で同じ素材を使っていても、先に来た方だけを頼む
        """
        if self._decode_threads <= 1:
            return
        requests: list[tuple[_DecodeKey, Fraction]] = []
        taken = set(self._decoding)
        for request in candidates:
            if request[0] in taken:
                continue
            taken.add(request[0])
            requests.append(request)
        if len(requests) < minimum:
            # 相手がいないなら走り係を起こさない 1 本だけ渡しても、受け渡しの分だけ遅い
            return
        # 走り係の本数より多く頼まない 多く頼んでも順番待ちになるだけで、その間
        # デコーダを開いたまま抱え込み、開けるデコーダの上限（8 本）を押し上げる
        # あふれた分は、描く順に :meth:`_decode` がそのまま読む
        requests = requests[: self._decode_threads]
        pool = self._pool()
        for key, when in requests:
            # デコーダを開くのはこちらの側 開く所（ファイルを掴み、控えを見に行く）まで
            # スレッドへ出すと、同じ鍵を 2 回開いて別のデコーダを 2 つ持つことがある
            decoder = self._decoder_for(*key)
            if decoder is None:
                continue
            self._decoding[key] = (when, pool.submit(decoder.frame_at, when))

    def _drain_decodes(self) -> None:
        """使われずに残った先読みを、全部受け取ってから捨てる

        受け取らずに捨てると、失敗した先読みの例外が誰にも届かず、素材が読めない
        まま黒い絵が書き出される 走ったままのスレッドが次のフレームの先読みと
        同じデコーダで重なることにもなる
        """
        pending = list(self._decoding.values())
        self._decoding.clear()
        error: Exception | None = None
        for _, future in pending:
            try:
                future.result()
            except Exception as exc:
                # 最初の失敗を覚えて、残りも必ず受け取ってから投げ直す
                # ここで抜けると、受け取っていない先読みが残る
                if error is None:
                    error = exc
        # 受け取り終えた所で、先読みを守るために超えていた分を閉じる
        # ここで減らさないと、同じ素材を描き続ける間は誰も減らす者がいない
        # （:meth:`_decoder_for` は既にある鍵ならすぐ返るので通らない）
        self._trim_decoders()
        if error is not None:
            raise error

    def _settle_decodes(self) -> None:
        """走っている先読みを受け取り終える デコーダを閉じ替える前に必ず呼ぶ

        例外は投げ直さない 絵を作る道ではなく後始末なので、ここで投げると
        デコーダを掴んだままファイルを差し替えられなくなる
        """
        try:
            self._drain_decodes()
        except Exception:
            self._decoding.clear()

    def _shutdown_decodes(self) -> None:
        """先読みを止めて走り係を畳む デコーダを閉じる前に必ず通る"""
        self._settle_decodes()
        if self._decode_pool is not None:
            self._decode_pool.shutdown(wait=True)
            self._decode_pool = None

    def _decoder_for(self, media_id: MediaId, stream_index: int) -> VideoDecoder | None:
        key = (media_id, stream_index)
        existing = self._decoders.get(key)
        if existing is not None:
            self._decoders.move_to_end(key)
            return existing

        media = self._project.find_media(media_id)
        if media is None:
            return None
        decoder = self._open(media, stream_index)
        if decoder is None:
            return None

        self._decoders[key] = decoder
        # **いま開いたもの自身**は閉じない 先に開いた分が全部先読み中だと、
        # 残る相手が自分だけになり、開いた直後に閉じたデコーダを呼ぶ側へ返してしまう
        # （9 本以上の別素材が同時に映るフレームでそうなる）
        self._trim_decoders(keep=key)
        return decoder

    def _trim_decoders(self, *, keep: _DecodeKey | None = None) -> None:
        """開いたままのデコーダを上限まで減らす

        **先読みが走っているデコーダは閉じない** 読んでいる最中に閉じると、
        走っているスレッドが解放済みのコンテナを触って落ちる
        閉じる相手がいなければ、上限を一時的に超えたままにする 上限は
        「開きっぱなしを増やさない」ための目安で、正しさの条件ではない
        超えた分は :meth:`_drain_decodes` が受け取り終えた所で閉じる
        """
        spare = [k for k in self._decoders if k not in self._decoding and k != keep]
        for old in spare[: max(0, len(self._decoders) - MAX_OPEN_DECODERS)]:
            self._decoders.pop(old).close()

    def _open(self, media: MediaItem, stream_index: int) -> VideoDecoder | None:
        """素材を開く 控えが使えなければ捨てて、元の素材で開き直す

        壊れた控えには 2 通りある 開けないもの（書きかけのまま落ちた等）と、
        **見出しは読めるのに 1 枚も出せないもの**（途中で切れたファイル）
        後者はそのまま掴むと、そのクリップだけ白いまま何も映らない
        置き場から捨てておけば、次の求めで作り直せる
        """
        path, index = self._source_for(media, stream_index)
        if path != media.path:
            decoder = self._usable(path, index)
            if decoder is not None:
                return decoder
            if self._proxies is not None:
                self._proxies.discard(media)
                # 捨てただけでは作り直されない 控えを作る係へ伝える道が要る
                # 伝えないと、その回だけでなく**そのあとずっと**元の素材を読む
                self._discarded.add(media.id)
        return self._usable(media.path, stream_index)

    def _usable(self, path: Path, stream_index: int | None) -> VideoDecoder | None:
        """開けて、1 枚目を出せるなら返す

        開けるかどうかだけでは足りない 見出しだけ正しいファイルを掴むと、
        映らない理由が分からないまま残る 確かめるのは開いた 1 回だけで、
        描くたびには走らない

        時刻 0 で確かめてよいのは、:meth:`VideoDecoder._decode_at` が
        まだ 1 枚も読んでいないときに**読めた 1 枚目をそのまま採る**ため
        先頭が 0 より後から始まる素材（切り出したもの）でも絵が返る
        ここは ``test_a_source_starting_after_zero_still_plays`` で見張る
        """
        try:
            decoder = VideoDecoder(path, stream_index)
        except ProbeError:
            # オフライン素材や壊れたファイル ここで落とすと、1 本壊れただけで
            # プロジェクト全体が開けなくなる そのクリップだけ映らない扱いにする
            return None
        if decoder.frame_at(Fraction(0)) is None:
            decoder.close()
            return None
        return decoder

    def set_proxies(self, proxies: ProxyStore | None) -> None:
        """控えの置き場を差し替える 開いているデコーダは閉じる

        閉じないと、すでに開いているものは元のファイル（または古い控え）を
        掴んだままになり、設定を変えても見た目が変わらない
        """
        if proxies is self._proxies:
            return
        self._proxies = proxies
        self.reopen_sources()

    def take_discarded(self) -> set[MediaId]:
        """捨てた控えの素材を取り出して忘れる

        呼ぶ側が作り直しを頼む 取り出した時点で忘れるので、同じ素材を
        何度も作り直すことにはならない
        """
        found, self._discarded = self._discarded, set()
        return found

    def stale_images(self) -> frozenset[str]:
        """エフェクトが読む画像のうち、前に読んだときから書き換わったもののパス

        プロジェクトは同じままなので、編集の差分からは分からない 呼ぶ側が
        :func:`~sashimono.engine.render.image_spans` でその画像を使う範囲を出し、
        先読みした絵を捨てる GL は触らない
        """
        return self._effects.stale_images()

    def reopen_sources(self, media_ids: Collection[MediaId] | None = None) -> None:
        """デコーダを閉じる 次に要るときに開き直す

        控えができた直後に呼ぶ 開きっぱなしだと、先にプレビューした素材は
        元のファイルを掴んだままになり、控えができても切り替わらない

        ``media_ids`` を渡すと、その素材のぶんだけ閉じる 全部閉じると、
        別の素材の控えができるたびに再生中のクリップまで開き直しと
        シークが走り、素材の本数だけ再生が途切れる
        """
        # 先読みが走ったまま閉じると、走っているスレッドが解放済みのコンテナを触る
        self._settle_decodes()
        for key in [k for k in self._decoders if media_ids is None or k[0] in media_ids]:
            self._decoders.pop(key).close()

    def _source_for(self, media: MediaItem, stream_index: int) -> tuple[Path, int | None]:
        """実際に読むファイル 控えがあればそちら

        控えは映像 1 本だけを持つので、元のストリーム番号は渡さない
        渡すと、元では 3 本目だった番号を控えの中で探して見つからない
        """
        if self._proxies is None or not media.video_streams:
            return media.path, stream_index
        # 控えに入っているのは**1 本目の映像**だけ 2 本目を指しているクリップに
        # 渡すと、別の絵が映る（素材によっては本編と副音声の絵が入れ替わる）
        #
        # どの映像を指しているかは :class:`VideoDecoder` と同じ読み方で決める
        # 番号で比べるだけだと、音が先に入っている素材（映像が 1 番から始まる）で
        # 既定の 0 が当たらず、控えがあるのに黙って使われない
        first = media.video_streams[0]
        chosen = next((s for s in media.video_streams if s.index == stream_index), first)
        if chosen.index != first.index:
            return media.path, stream_index
        found = self._proxies.find(media)
        return (media.path, stream_index) if found is None else (found, None)

    def _texture_for(self, key: str) -> Texture:
        texture = self._textures.get(key)
        if texture is None:
            texture = Texture(1, 1)
            self._textures[key] = texture
        return texture
