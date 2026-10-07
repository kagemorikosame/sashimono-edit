"""テキストと図形を絵にする

Qt の描画系（``QPainter``）を使う 日本語の禁則処理やフォントの字形選択、
縁取りの輪郭生成を自前で書くのは現実的ではなく、Qt はそれをすべて持っている

戻り値は常に sRGB・ストレートアルファの ``(高さ, 幅, 4)`` uint8 素材から
デコードした絵とまったく同じ形なので、この先の合成は区別せずに済む
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PySide6.QtCore import QLocale, QPointF, QRectF, Qt, QTextBoundaryFinder
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPolygonF,
    QRadialGradient,
    QRawFont,
    QTransform,
)

from sashimono.compat.aviutl.text_tags import (
    DECORATION_NAMES,
    TaggedLine,
    TextRun,
    TextStyle,
    parse_tags,
)
from sashimono.compat.decoration import decoration_params, find_decoration
from sashimono.core.model import AnimatedValue, GeneratedSource, ParamValue
from sashimono.effects.sources import SourceDefinition, source_registry
from sashimono.engine.audio_shapes import (
    WAVEFORM_LINE,
    bar_mask,
    cell_mask,
    spectrum_cells,
    spectrum_levels,
    waveform_cells,
    waveform_points,
)
from sashimono.engine.motion_shapes import (
    TrailPath,
    TrailPaths,
    sample_value,
    star_field,
    trail,
    unit_randoms,
)

__all__ = ["Frame", "render_source", "render_source_framed", "waveform_points"]

#: 縦の基準ごとに、指定した位置より上へ出す割合 ``下`` なら全部が上に出る
_VERTICAL_SHARE = {"top": 0.0, "middle": 0.5, "bottom": 1.0}
#: 横の基準ごとに、指定した位置より左へ出す割合 ``右`` なら全部が左に出る
_HORIZONTAL_SHARE = {"left": 0.0, "center": 0.5, "right": 1.0}


def render_source(
    source: GeneratedSource,
    width: int,
    height: int,
    *,
    frame: int = 0,
    fps: float = 30.0,
    duration: int = 0,
    audio: np.ndarray | None = None,
    audio_rate: int = 44100,
    trail_paths: TrailPaths | None = None,
) -> np.ndarray | None:
    """生成オブジェクトを描いて配列で返す 未知の種類なら ``None``

    ``fps`` は時間で変わる図形（タイマー・集中線）がフレームを秒へ直すのに使う
    ``duration`` はクリップの長さ（フレーム） 移動軌跡が先端の向きを決めるときに、
    クリップの終わりより先の動きを見ないために使う 分からなければ 0
    ``trail_paths`` は移動軌跡の道の置き場 レンダラが自分のものを渡し、使い回す
    ``audio`` は音声波形が描く音（今の時刻からの 1 チャンネルのサンプル）
    ``audio_rate`` はそのレート（スペクトラムの周波数に使う）
    音を読むのはレンダラの仕事 ここは渡された数を線にするだけ
    """
    return render_source_framed(
        source,
        width,
        height,
        frame=frame,
        fps=fps,
        duration=duration,
        audio=audio,
        audio_rate=audio_rate,
        trail_paths=trail_paths,
    )[0]


#: 絵の中のオブジェクトの枠（画素、左・上・右・下 小数のまま）
Frame = tuple[float, float, float, float]


def render_source_framed(
    source: GeneratedSource,
    width: int,
    height: int,
    *,
    frame: int = 0,
    fps: float = 30.0,
    duration: int = 0,
    audio: np.ndarray | None = None,
    audio_rate: int = 44100,
    trail_paths: TrailPaths | None = None,
    scale: tuple[float, float] = (1.0, 1.0),
    screen: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, Frame | None]:
    """:func:`render_source` と同じ絵と、オブジェクトの枠

    枠は、字の形ではなく**文字の枠**を入れ物にする物（AviUtl2 の組み方のテキスト）
    だけが返す それ以外は ``None`` で、入れ物は色の付いた範囲から求める

    ``scale`` は絵の画素 1 つが画面の画素いくつ分かの逆数（横, 縦） 画質を落とした
    プレビューは 1 より小さい 設定の値（文字の大きさ・縁取りの太さ・影の距離・図形の幅と
    位置）はどれも画面の画素なので、画面の大きさのつもりで描いて、絵へは縮めて写す
    値を 1 つずつ縮めるのではなく描く座標ごと縮めるのは、文字の中の制御文字
    （``<s大きさ>``）のように設定の欄に無い画素の値まで漏れなく縮めるため
    枠も絵の画素で返す

    ``screen`` は広げる前の大きさ（合成の画素 :func:`source_canvas` に渡した物）
    絵がそれより大きければ、文字の影は影の周りだけで作る（:func:`_shadow_layer`）
    """
    definition = source_registry.get(source.kind)
    if definition is None:
        return None, None

    values = _resolve(definition, source.params, frame)
    values["_seconds"] = frame / max(fps, 1e-6)
    values["_fps"] = fps
    values["_frame"] = frame
    values["_duration"] = duration
    # 移動軌跡は、今の値ではなく**動きそのもの**（過去の位置）を読む
    values["_motion"] = _motion_of(definition, source.params)
    values["_audio"] = audio
    values["_audio_rate"] = audio_rate
    # 移動軌跡の道の置き場 レンダラが自分のものを渡す（他のレンダラが描く道を
    # 巻き込んで捨てないように） 無ければ移動軌跡を描くときだけその場で作る
    # （文字や普通の図形のたびに作らない）
    values["_trail_paths"] = trail_paths
    values["_crop_shadow"] = screen is not None and (width > screen[0] or height > screen[1])
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    # 画面の画素で数えた大きさ 絵の中心と画面の中心を重ねて縮める（大きく作った絵でも同じ）
    screen_width, screen_height, transform = _screen_space(width, height, scale)
    painter.setTransform(transform)
    framed: Frame | None = None
    try:
        if source.kind == "text":
            framed = _draw_text(painter, values, screen_width, screen_height)
        elif source.kind == "shape":
            _draw_shape(painter, values, screen_width, screen_height)
    finally:
        painter.end()

    if framed is not None and not transform.isIdentity():
        mapped = transform.mapRect(
            QRectF(QPointF(framed[0], framed[1]), QPointF(framed[2], framed[3]))
        )
        framed = (mapped.left(), mapped.top(), mapped.right(), mapped.bottom())
    return _to_array(image), framed


def _screen_space(
    width: int, height: int, scale: tuple[float, float]
) -> tuple[int, int, QTransform]:
    """``width`` × ``height`` の絵に、画面の画素で描くときの大きさと写し方

    縮めないとき（書き出し）は恒等の写し方になり、今までと 1 画素も変わらない
    """
    scale_x, scale_y = scale
    if scale_x == 1.0 and scale_y == 1.0:
        return width, height, QTransform()
    screen_width = max(1, round(width / scale_x))
    screen_height = max(1, round(height / scale_y))
    transform = QTransform()
    transform.translate(width / 2.0, height / 2.0)
    transform.scale(scale_x, scale_y)
    transform.translate(-screen_width / 2.0, -screen_height / 2.0)
    return screen_width, screen_height, transform


#: 画面より大きい絵を作るときの一辺の上限（画素） GPU のテクスチャの上限より十分小さく
MAX_CANVAS = 8192


def source_canvas(
    source: GeneratedSource,
    width: int,
    height: int,
    *,
    frame: int = 0,
    scale: tuple[float, float] = (1.0, 1.0),
    fps: float = 30.0,
) -> tuple[int, int]:
    """生成オブジェクトを描く絵の大きさ 画面からはみ出す図形とテキストは、はみ出す分まで広げる

    中心は画面の中心のまま広げる（描く位置の計算は変えない） 画面の大きさで
    切ってしまうと、画面より大きい図形を回したり動かしたりしたときに、切れた端が
    見えてしまう（YMM4 の斜めの帯のトランジションは高さ 2160 の図形を 45 度回す）
    テキストも同じで、画面の幅を超える 1 行や、位置をずらして画面の外へ寄せた字が
    端で切れていた（#256）

    ``width`` と ``height`` は合成の大きさ ``scale`` は :func:`render_source_framed` と同じ
    ``fps`` はタイマーの文字を数えるため（:func:`render_source_framed` と同じ値を渡す）
    """
    if source.kind == "text":
        return _text_canvas(source, width, height, frame, scale, fps)
    if source.kind != "shape":
        return width, height
    definition = source_registry.get(source.kind)
    if definition is None:
        return width, height
    values = _resolve(definition, source.params, frame)
    if values.get("shape") in ("background", "motion_trail", "starfield"):
        # 移動軌跡と星空は画面の座標で描く 今の位置や大きさの設定から広げると、
        # 軌跡の通った所とは関係の無い大きさの絵を毎フレーム作ることになる
        return width, height
    if values.get("shape") == "polyline":
        # 線の図形は点の広がりで見積もる
        points = centred_points(values, width, height)
        line = float(values.get("line_width", 0.0))  # type: ignore[arg-type]
        reach_x = max((abs(x) for x, _ in points), default=0.0) + line
        reach_y = max((abs(y) for _, y in points), default=0.0) + line
        values = {**values, "width": reach_x * 2.0, "height": reach_y * 2.0, "line_width": 0.0}
    shape_width = max(1.0, float(values.get("width", 400)))  # type: ignore[arg-type]
    shape_height = max(1.0, float(values.get("height", 400)))  # type: ignore[arg-type]
    line = float(values.get("line_width", 0.0))  # type: ignore[arg-type]
    if str(values.get("line_align", "center")) == "inside":
        # 内側に引く線は外形を超えない 太さぶん広げたままだと、線の太い大きな図形で
        # 毎フレーム必要のない大きさの絵を作ることになる
        line = 0.0
    # 回しても収まるよう、対角線の長さで見積もる
    reach = (shape_width**2 + shape_height**2) ** 0.5 / 2.0 + line
    # 設定は画面の画素 画質を落とした合成（``scale`` が 1 より小さい）では、その分だけ小さい
    # 絵で足りる 縮めずに見積もると、画面より大きいと見なして毎フレーム余分に広い絵を作る
    needed_width = 2.0 * (abs(_number(values, "pos_x", 0.0)) + reach) * scale[0]
    needed_height = 2.0 * (abs(_number(values, "pos_y", 0.0)) + reach) * scale[1]
    return _grown(width, height, needed_width, needed_height)


def _grown(width: int, height: int, needed_width: float, needed_height: float) -> tuple[int, int]:
    """画面 ``width`` x ``height`` を、中心を変えずに ``needed`` まで広げた大きさ

    収まるなら画面のまま（今までと同じ絵） 一辺は :data:`MAX_CANVAS` まで それより外は切れる
    """
    grown_width = min(MAX_CANVAS, max(width, int(np.ceil(needed_width))))
    grown_height = min(MAX_CANVAS, max(height, int(np.ceil(needed_height))))
    # 画面と偶奇をそろえる 差が奇数だと、中心が半画素ずれて輪郭がにじむ
    # 上限が画面と偶奇の違う数でも上限を超えないよう、そのときは 1 つ減らす
    grown_width += (grown_width - width) % 2 * (1 if grown_width < MAX_CANVAS else -1)
    grown_height += (grown_height - height) % 2 * (1 if grown_height < MAX_CANVAS else -1)
    return max(width, grown_width), max(height, grown_height)


#: 字の広がりに足す余白（画面の画素） 輪郭の滑らかにした端の 1 画素と、Qt が字の形の
#: 外へ少しだけ塗る分 足りないと、ちょうど画面の端に届く字の端が 1 画素だけ欠ける
_TEXT_MARGIN = 2.0


def _text_canvas(
    source: GeneratedSource,
    width: int,
    height: int,
    frame: int,
    scale: tuple[float, float],
    fps: float,
) -> tuple[int, int]:
    """テキストを描く絵の大きさ 字・縁取り・影が画面からはみ出すなら、その分まで広げる"""
    definition = source_registry.get(source.kind)
    if definition is None:
        return width, height
    values = _resolve(definition, source.params, frame)
    if str(values.get("timer_format", "")):
        # タイマーは描く所（:func:`_text_layers`）と同じ文字を、ここで先に作っておく
        values["_seconds"] = frame / max(fps, 1e-6)
        values["_fps"] = fps
        values["text"] = timer_text(values)[:200]
        values["timer_format"] = ""
    pos_x = _number(values, "pos_x", 0.0)
    pos_y = _number(values, "pos_y", 0.0)
    # 位置と文字送りは鍵から外す 位置は広がりをずらすだけで、文字送りは全部を出した字の
    # 広がりに収まる（出ている字は全体の頭の部分） 鍵に入れると、流れるテロップや
    # 文字送りの字幕で毎フレーム字を組み直し、絵の大きさもフレームごとに揺れる
    values["pos_x"], values["pos_y"], values["reveal"] = 0.0, 0.0, 100.0
    reach = _text_reach(
        tuple(sorted((name, value) for name, value in values.items() if not name.startswith("_")))
    )
    if reach is None:
        return width, height
    left, top, right, bottom = reach
    # 画面の Y は下が正 設定の Y は上が正なので、上へずらすと字の上端は小さくなる
    needed_width = 2.0 * max(abs(left + pos_x), abs(right + pos_x)) * scale[0]
    needed_height = 2.0 * max(abs(top - pos_y), abs(bottom - pos_y)) * scale[1]
    return _grown(width, height, needed_width, needed_height)


@lru_cache(maxsize=256)
def _text_reach(items: tuple[tuple[str, object], ...]) -> Frame | None:
    """位置 0 に置いた字が、置いた所からどこまで届くか（画面の画素 左・上・右・下）

    字の輪郭に縁取りの太さ、影のずれとぼかしを足す AviUtl2 の組み方の文字の枠も入れる
    （効果はこの枠を入れ物にするので、絵の外に出ると入れ物が切れる） 字が無ければ ``None``

    同じ設定なら同じ答え 動かない字幕を描くたびに字を組み直さないよう覚えておく
    """
    values = dict(items)
    laid = _text_layers(values, 0, 0)
    if laid is None:
        return None
    layers, framed = laid
    reach = QRectF()
    for path, look, clip in layers:
        box = path.boundingRect()
        if clip is not None:
            box = box.intersected(clip.boundingRect())
        if box.isEmpty():
            continue
        border = max(0.0, _number(look, "border_width", 0.0))
        inked = box.adjusted(-border, -border, border, border)
        reach = reach.united(inked)
        shift_x = _number(look, "shadow_x", 0.0)
        shift_y = _number(look, "shadow_y", 0.0)
        blur = max(0.0, _number(look, "shadow_blur", 0.0))
        if (shift_x, shift_y, blur) != (0.0, 0.0, 0.0) and _color(look.get("shadow_color")).alpha():
            # 箱ぼかしを 2 回掛けるので、影はぼかしの幅の 2 倍まで広がる（:func:`_blur_alpha`）
            spread = 2.0 * blur + 1.0
            shadow = inked.translated(shift_x, -shift_y)
            reach = reach.united(shadow.adjusted(-spread, -spread, spread, spread))
    if framed is not None:
        reach = reach.united(QRectF(QPointF(framed[0], framed[1]), QPointF(framed[2], framed[3])))
    if reach.isEmpty():
        return None
    margin = _TEXT_MARGIN
    return (
        reach.left() - margin,
        reach.top() - margin,
        reach.right() + margin,
        reach.bottom() + margin,
    )


def _resolve(
    definition: SourceDefinition, params: dict[str, ParamValue], frame: int
) -> dict[str, object]:
    """パラメータを、その時刻での素の値へ

    数値は :class:`~sashimono.core.model.AnimatedValue` なので、フレームを与えて
    評価する ここを飛ばすとキーフレームが効かない
    """
    resolved: dict[str, object] = {}
    for spec in definition.parameters:
        value = spec.coerce(params.get(spec.name))
        resolved[spec.name] = value.at(frame) if isinstance(value, AnimatedValue) else value
    return resolved


def _motion_of(
    definition: SourceDefinition, params: dict[str, ParamValue]
) -> tuple[AnimatedValue, AnimatedValue]:
    """位置の X と Y を、時刻で引ける形のまま返す 位置を持たない種類は 0 に止まった値"""
    pair: list[AnimatedValue] = []
    for name in ("pos_x", "pos_y"):
        spec = definition.spec(name)
        value = spec.coerce(params.get(name)) if spec is not None else None
        pair.append(value if isinstance(value, AnimatedValue) else AnimatedValue(0.0))
    return pair[0], pair[1]


def timer_text(values: dict[str, object]) -> str:
    """タイマーの文字 数え下げは、クリップの終わりで初めの値になるように数える

    YMM4 に 2 秒のクリップ・初めの値 0.99 で描かせると、頭で 2、1 秒で 1 だった
    （終わりから逆算した残り時間に初めの値を足している）
    """
    seconds = float(values.get("_seconds", 0.0))  # type: ignore[arg-type]
    rate = float(values.get("timer_rate", 100.0)) / 100.0  # type: ignore[arg-type]
    start = float(values.get("timer_start", 0.0))  # type: ignore[arg-type]
    if bool(values.get("timer_countdown", False)):
        fps = float(values.get("_fps", 30.0))  # type: ignore[arg-type]
        total = float(values.get("timer_length", 0)) / max(fps, 1e-6)  # type: ignore[arg-type]
        value = start + (total - seconds) * rate
    else:
        value = start + seconds * rate
    pattern = str(values.get("timer_format", ""))
    # 時刻の書式（h・m・s）は 0 で止める 時計が負になることはない
    # 通算の n だけ負の値を出す（AviUtl のカウンターは数え下げで負になる）
    if not _counts_total(pattern):
        value = max(value, 0.0)
    return format_time(value, pattern)


def _counts_total(pattern: str) -> bool:
    """書式に通算の ``n`` が入っているか

    ただの文字列の検索では、``s\n`` のように逃がした（文字としての）``n`` まで
    拾ってしまい、時計の書式が 0 で止まらなくなる 逃がした文字は飛ばして見る
    """
    index = 0
    while index < len(pattern):
        if pattern[index] == chr(92):
            index += 2
            continue
        if pattern[index] == "n":
            return True
        index += 1
    return False


#: 時間の書式で 1 つの文字を並べられる数の上限 壊れたファイルの巨大な書式で固まらないため
MAX_TIME_DIGITS = 9


def format_time(value: float, pattern: str) -> str:
    """.NET の時間の書式（``h`` ``m`` ``s`` ``f`` と ``\\`` の逃がし）で秒を文字にする

    ``n`` だけは .NET に無いこちらの追加で、**60 で折り返さない通算の値**
    AviUtl のカウンターのように、ただ数を数えるものに使う（``s`` は分に繰り上がる）
    """
    # 非有限は 0 として扱う int() が例外になり、描画がフレームごと止まる
    if not math.isfinite(value):
        value = 0.0
    # 負の値も出す AviUtl のカウンターは負の初めの値や数え下げを持てる
    # 0 で止めると、下がっていくはずの数字が途中から動かなくなる
    sign = "-" if value < 0.0 else ""
    value = min(abs(value), 10.0**9)
    whole = int(value)
    parts = {
        "h": whole // 3600,
        "m": (whole // 60) % 60,
        "s": whole % 60,
        "n": whole,
    }
    out: list[str] = []
    index = 0
    while index < len(pattern):
        letter = pattern[index]
        if letter == "\\" and index + 1 < len(pattern):
            out.append(pattern[index + 1])
            index += 2
            continue
        run = 1
        while index + run < len(pattern) and pattern[index + run] == letter:
            run += 1
        digits = min(run, MAX_TIME_DIGITS)
        if letter in parts:
            out.append(str(parts[letter]).zfill(digits))
        elif letter in ("f", "F"):
            fraction = value - whole
            out.append(str(int(fraction * 10**digits)).zfill(digits))
        else:
            out.append(letter * digits)
        index += run
    return sign + "".join(out)


#: AviUtl2 の太字が字を太らせる量（文字サイズに対する割合）
#: MS UI Gothic（太字の書体を持たない）の 180 で、字の外形が右と上へ約 3.7 画素ずつ
#: 広がり、1 文字の送り幅も同じだけ広がっていた（枠は 540 から 551） 1/48 はその量
#: Qt の太字は同じ書体で右へ約 10 画素太らせ送り幅は 1 しか広げないので、字が太く、
#: 中央揃えでも右へ 4〜5 画素寄る
_AVIUTL_BOLD = 1.0 / 48.0


#: 描く字の 1 層 字の輪郭・その字の飾りの値・切る形（変形した字だけ文字の枠 ほかは ``None``）
_Layer = tuple[QPainterPath, dict[str, object], QPainterPath | None]


def _draw_text(
    painter: QPainter, values: dict[str, object], width: int, height: int
) -> Frame | None:
    """横書きと縦書きを描き分ける AviUtl2 の組み方なら文字の枠を返す"""
    laid = _text_layers(values, width, height)
    if laid is None:
        return None
    layers, framed = laid
    _paint_layers(painter, layers)
    return framed


def _text_layers(
    values: dict[str, object], width: int, height: int
) -> tuple[list[_Layer], Frame | None] | None:
    """字を組んで、描く層と文字の枠を返す 字が無ければ ``None``

    組むのと塗るのを分けておくのは、描く前に字の広がりを測るため（:func:`source_canvas`）
    同じ組み方を 2 つ書くと、測った広がりと描いた字が食い違って端が切れる
    """
    raw = str(values.get("text", ""))
    if str(values.get("timer_format", "")):
        raw = timer_text(values)[:200]

    # 縦書きは AviUtl2 で測っていないので、AviUtl2 の組み方でも標準の縦書きで描く
    # （枠は返さず、太字も Qt に任せる） ここで外さないと、Qt の太字を切ったまま
    # 自分で太らせる横書きの処理も通らず、太字が細字で出る
    aviutl = values.get("layout") == "aviutl" and not bool(values.get("vertical", False))
    bold = bool(values.get("bold", False))
    size = max(1, int(float(values.get("size", 64))))  # type: ignore[arg-type]
    if aviutl:
        # 制御文字（``<@書体>`` ``<#色>`` ``<s大きさ>``）を読んでから文字送りを掛ける
        # 先に掛けると、タグの文字まで 1 文字として数えて出す字がずれる
        tagged = _revealed_lines(parse_tags(raw, float(size)), values)
        if not any(line.text for line in tagged):
            return None
        centre_x = width / 2.0 + _number(values, "pos_x", 0.0)
        centre_y = height / 2.0 - _number(values, "pos_y", 0.0)
        family = aviutl_font_family(str(values.get("font", AVIUTL_DEFAULT_FONT)))
        groups, framed = _aviutl_lines(tagged, family, size, bold, values, centre_x, centre_y)
        return _group_layers(groups, values), framed

    text = _revealed(raw, values)
    if not text:
        return None
    font = QFont(str(values.get("font", "Yu Gothic UI")))
    font.setPixelSize(size)
    # AviUtl2 の組み方の太字は、細字の輪郭を自分で太らせる（:func:`_emboldened`）
    # Qt に太字を頼むと太り方も送り幅も AviUtl2 と違う
    font.setBold(bold and not aviutl)
    font.setItalic(bool(values.get("italic", False)))
    letter_spacing = float(values.get("letter_spacing", 0.0))  # type: ignore[arg-type]
    # AviUtl2 の字間は文字と文字の間にだけ入る（自分で足す） Qt に頼むと最後の字の
    # 後ろにも入り、3 文字の行が字間 1 つ分広く、中央揃えで半分だけ左へ寄る
    if letter_spacing and not aviutl:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)

    metrics = QFontMetricsF(font)
    if bool(values.get("vertical", False)):
        vertical = _vertical_text_path(text, font, metrics, values, width, height)
        return [(vertical, values, None)], None

    lines = text.split(chr(10))
    centre_x = width / 2.0 + float(values.get("pos_x", 0.0))  # type: ignore[arg-type]
    centre_y = height / 2.0 - float(values.get("pos_y", 0.0))  # type: ignore[arg-type]

    line_height = metrics.height() + float(values.get("line_spacing", 0.0))  # type: ignore[arg-type]
    block_height = line_height * len(lines)
    align = str(values.get("align", "center"))
    top = centre_y - block_height * _VERTICAL_SHARE.get(str(values.get("valign", "middle")), 0.5)
    widest = max((metrics.horizontalAdvance(line) for line in lines), default=0.0)

    # 文字を輪郭（パス）として組み立てる 縁取りを外側だけに出すには、
    # 塗りとは別に輪郭を太らせる必要があり、それはパスでしかできない
    # 太字は、同じ揃え方で細字を置いたときの字の外形の中心へ戻す（:func:`_bold_drift`）
    plain_font = QFont(font)
    plain_font.setBold(False)
    plain_metrics = QFontMetricsF(plain_font)
    plain_widest = max((plain_metrics.horizontalAdvance(line) for line in lines), default=0.0)

    # 文字の塊の横の基準（``anchor``） 既定は塊の真ん中が位置 行揃えはその塊の中で揃える
    anchor_share = _HORIZONTAL_SHARE.get(str(values.get("anchor", "center")), 0.5)

    def start(line_width: float, block_width: float) -> float:
        left = centre_x - block_width * anchor_share
        if align == "left":
            return left
        if align == "right":
            return left + block_width - line_width
        return left + (block_width - line_width) / 2.0

    path = QPainterPath()
    for index, line in enumerate(lines):
        x = start(metrics.horizontalAdvance(line), widest)
        baseline = top + line_height * index + metrics.ascent()
        placed = QPainterPath()
        placed.addText(QPointF(x, baseline), font, line)
        if bold:
            plain = QPainterPath()
            plain_x = start(plain_metrics.horizontalAdvance(line), plain_widest)
            plain.addText(QPointF(plain_x, baseline), plain_font, line)
            placed.translate(_bold_drift(plain, placed), 0.0)
        path.addPath(placed)

    return [(path, values, None)], None


def _bold_drift(plain: QPainterPath, bold: QPainterPath) -> float:
    """Qt の太字を、細字で置いたときと同じ中心へ戻す横のずれ（画素）

    Qt の合成の太字（太字の書体を持たない MS UI Gothic など）は右へだけ太らせ、
    送り幅は 1 文字 1 画素しか広げないので、中央揃えでも字が右へ寄る（180 の
    「田田田」で 4〜5 画素） 太字の書体を持つフォントでも測って戻すだけなので、
    寄っていなければ 0 に近くなる
    """
    plain_box, bold_box = plain.boundingRect(), bold.boundingRect()
    if plain_box.isEmpty() or bold_box.isEmpty():
        return 0.0
    return plain_box.center().x() - bold_box.center().x()


#: AviUtl2 の既定の書体 書体名が見つからないときもこれで描く
#: 英語の名前で書いた ``<@Meiryo>`` ``<@MS Gothic>`` ``<@Yu Mincho>`` も、設定欄の
#: ``フォント=Meiryo`` も、設定欄の書体（Arial や MS UI Gothic）ではなくこの書体で描かれた
#: （H の送り幅 71・字の高さ 70 が Yu Gothic UI と一致 #108）
AVIUTL_DEFAULT_FONT = "Yu Gothic UI"

#: 書体の ``name`` 表の言語 Windows の言語番号
_JAPANESE, _ENGLISH = 0x0411, 0x0409


def _system_is_japanese() -> bool:
    return QLocale.system().language() == QLocale.Language.Japanese


def aviutl_font_family(name: str, japanese: bool | None = None) -> str:
    """AviUtl2 が ``name`` で見つける書体の名前 見つけられない名前なら既定の書体

    AviUtl2 v2.1.6a（日本語の Windows）は、日本語の名前を持つ書体を日本語の名前でしか
    見つけない ``<@メイリオ>`` はメイリオで描き、``<@Meiryo>`` は既定の Yu Gothic UI で
    描いた 書体の一覧（``aviutl2.ini`` の ``[Font.…]``）も ``メイリオ`` ``ＭＳ ゴシック``
    ``游明朝`` と日本語で並ぶ Qt はどちらの名前でも同じ書体を返すので、Qt に任せると
    AviUtl2 と違う書体で描く

    英語の名前しか持たない書体（Arial・Yu Gothic UI・MS UI Gothic）は英語の名前で見つかる
    ``japanese`` は名前を引く言語 省略すると、この機械の言語が日本語かで決める
    """
    if japanese is None:
        japanese = _system_is_japanese()
    return _aviutl_font_family(name.strip(), japanese)


@lru_cache(maxsize=256)
def _aviutl_font_family(name: str, japanese: bool) -> str:
    if not name:
        return AVIUTL_DEFAULT_FONT
    raw = QRawFont.fromFont(QFont(name))
    names = _family_names(bytes(raw.fontTable("name").data())) if raw.isValid() else {}
    preferred = names.get(_JAPANESE) if japanese else None
    preferred = preferred or names.get(_ENGLISH) or next(iter(names.values()), None)
    if preferred is not None and preferred.casefold() == name.casefold():
        return name
    return AVIUTL_DEFAULT_FONT


def _family_names(table: bytes) -> dict[int, str]:
    """``name`` 表の書体名（番号 1）を言語ごとに Windows の記録（UTF-16）だけを読む

    GDI の書体名と同じ番号 1 を読む 番号 16（組版上の書体名）は「游明朝 Demibold」を
    「游明朝」にまとめてしまい、AviUtl2 の一覧（太さごとに別の名前）と合わない
    """
    names: dict[int, str] = {}
    if len(table) < 6:
        return names
    count, strings = struct.unpack(">HH", table[2:6])
    for index in range(count):
        offset = 6 + index * 12
        if offset + 12 > len(table):
            break
        platform, _encoding, language, number, length, where = struct.unpack(
            ">HHHHHH", table[offset : offset + 12]
        )
        if platform != 3 or number != 1:
            continue
        begin = strings + where
        raw = table[begin : begin + length]
        if len(raw) == length:
            names.setdefault(language, raw.decode("utf-16-be", errors="replace"))
    return names


@dataclass(frozen=True)
class _Look:
    """描き分ける見た目 同じ見た目の字は 1 つの輪郭にまとめて描く

    色は 16 進（``None`` は設定欄の色） ``decoration`` は文字装飾の番号 ``edge_width`` は
    ``<s>`` の縁取りの太さ（AviUtl2 の数のまま） どちらも ``None`` なら設定欄の文字装飾
    """

    color: str | None = None
    edge: str | None = None
    embolden: float = 0.0
    decoration: int | None = None
    edge_width: float | None = None


#: 1 文字ぶんの置き方 字・書体・送り幅・見た目・前に空ける字間・変形（横・縦の倍と時計回りの度）
_Part = tuple[str, QFont, float, _Look, float, tuple[float, float, float]]


#: 見た目 1 つぶんの字 3 つ目は描くときに切る形（変形した字だけ文字の枠 ほかは ``None``）
_Group = tuple[QPainterPath, "_Look", QPainterPath | None]


def _aviutl_lines(
    lines: list[TaggedLine],
    family: str,
    size: int,
    bold: bool,
    values: dict[str, object],
    centre_x: float,
    centre_y: float,
) -> tuple[list[_Group], Frame]:
    """AviUtl2 の組み方で行を並べ、見た目ごとの字の輪郭と文字の枠を返す

    AviUtl2 に描かせて測った決まり（#64 の見本 ``kumiki_p8_t_*``）

    - 行の高さは書体の行送り（上の高さ + 下の深さ + 行間の余白） Arial 120 で 139
      ベースラインは行の上端から上の高さ 余白は下に付く
    - 字間は文字の間にだけ、行間は行の間にだけ入る（字間 20 の 3 文字で枠が 40 広がる）
    - 文字揃えの横は、左寄せなら枠の左端、右寄せなら右端を置いた位置に合わせる
      縦も同じで、上なら上端、下なら下端 行はそれぞれ枠の中で左・中央・右へ揃える
    - 縁取りは枠を広げない（縁は枠の内側に描かれていた）

    書体や大きさが混ざった行（#108 の見本 ``tag02``〜``tag22``）

    - 字はどれも 1 本のベースラインに乗る（書体ごとの字の枠の中心で揃えるのではない）
    - 行の高さとベースラインの位置は、その行で**行送りのいちばん大きい書体 1 つ**で決まる
      上の高さの最大と下の深さの最大を足すのではない Arial・メイリオ（106 + 44）・
      ＭＳ ゴシック・游明朝（99.5 + 余白込みで 160.2）の行は、上 99・下 61 の 160 だった
      上の高さが最大のメイリオでは組んでいない
    - 文字の無い行は、その行の終わりの書体と大きさの行送り
    - 送り幅は字ごとにその字の書体と大きさで測る

    制御文字の残り（#134 の見本 ``tag24``〜``tag52``）

    - ``<gw>`` の字間はその字の前に入り、``<gh>`` の行間はその行の終わりの値で次の行との間に入る
    - ``<tw>`` ``<th>`` ``<tr>`` は字の送り幅を変えず、字の枠（送り幅 x 行の高さ）の真ん中を
      中心に 1 文字ずつ伸び縮み・回る（``<tw0.5>`` の H は枠の真ん中に半分の幅で出た）
    """
    letter_spacing = _number(values, "letter_spacing", 0.0)
    line_spacing = _number(values, "line_spacing", 0.0)
    italic = bool(values.get("italic", False))
    measured: dict[tuple[str, int, bool], tuple[QFont, QFontMetricsF, float]] = {}

    def typeface(style: TextStyle) -> tuple[QFont, QFontMetricsF, float, int]:
        name = family if style.font is None else aviutl_font_family(style.font)
        pixels = size if style.size is None else max(1, round(style.size))
        slanted = italic if style.italic is None else style.italic
        key = (name, pixels, slanted)
        if key not in measured:
            font = QFont(name)
            font.setPixelSize(pixels)
            font.setItalic(slanted)
            metrics = QFontMetricsF(font)
            measured[key] = (font, metrics, metrics.height() + _external_leading(font))
        font, metrics, pitch = measured[key]
        return font, metrics, pitch, pixels

    laid: list[tuple[list[_Part], float, float, float, float]] = []
    for line in lines:
        parts: list[_Part] = []
        # 行の高さを決める書体 行送りが同じなら先に出た方（max は最初の最大を返す）
        faces = [typeface(run.style) for run in line.runs if run.text] or [typeface(line.end)]
        _font, tallest, pitch, _pixels = max(faces, key=lambda face: face[2])
        for run in line.runs:
            style = run.style
            font, metrics, _pitch, pixels = typeface(style)
            thick = bold if style.bold is None else style.bold
            embolden = pixels * _AVIUTL_BOLD if thick else 0.0
            look = _Look(style.color, style.edge, embolden, style.decoration, style.edge_width)
            gap = letter_spacing if style.letter_gap is None else style.letter_gap
            shape = (
                1.0 if style.scale_x is None else style.scale_x,
                1.0 if style.scale_y is None else style.scale_y,
                0.0 if style.turn is None else style.turn,
            )
            for part in _graphemes(run.text):
                step = metrics.horizontalAdvance(part) + embolden
                # 字間は行の最初の字の前には入れない（文字の間にだけ入る）
                parts.append((part, font, step, look, gap if parts else 0.0, shape))
        advance = sum(part[2] + part[4] for part in parts)
        after = line_spacing if line.end.line_gap is None else line.end.line_gap
        laid.append((parts, advance, pitch, tallest.ascent(), after))

    widest = max((row[1] for row in laid), default=0.0)
    # 最後の行の後ろには行間を入れない（行の間にだけ入る）
    block_height = sum(row[2] for row in laid) + sum(row[4] for row in laid[:-1])
    align = str(values.get("align", "center"))
    left = centre_x - widest * {"left": 0.0, "right": 1.0}.get(align, 0.5)
    top = centre_y - block_height * _VERTICAL_SHARE.get(str(values.get("valign", "middle")), 0.5)

    # 見た目ごとに、そのまま置いた字と変形した字を分けて持つ（:data:`_Group`）
    paths: dict[tuple[_Look, bool], QPainterPath] = {}
    # 伸ばしたり回したりした字は文字の枠（テキストの入れ物）で切る AviUtl2 の絵は枠の大きさ
    # なので、はみ出した所は描かれない（``H<th2>H`` の真ん中の字が枠の上下で切れた #184）
    # 切るのは描くとき（:func:`_paint_layers`） 字の形を先に切ると、後から付ける縁と影が
    # 枠の外へ広がり、切った面にも元の字に無い縁が付く
    frame = QPainterPath()
    frame.addRect(QRectF(left, top, widest, block_height))
    line_top = top
    for parts, advance, pitch, ascent, after in laid:
        x = left + (widest - advance) * {"left": 0.0, "right": 1.0}.get(align, 0.5)
        baseline = line_top + ascent
        # 1 文字ずつ置く 太字の分だけ送り幅を広げるのは文字ごとで、行をまとめて
        # 置くと 2 文字目から先が太った分だけ前の字に食い込む
        for part, font, step, look, gap, shape in parts:
            x += gap
            plain = shape == (1.0, 1.0, 0.0)
            target = paths.setdefault((look, not plain), QPainterPath())
            if plain:
                target.addText(QPointF(x, baseline), font, part)
            else:
                glyph = QPainterPath()
                glyph.addText(QPointF(x, baseline), font, part)
                target.addPath(_shaped(glyph, x + step / 2.0, line_top + pitch / 2.0, shape))
            x += step
        line_top += pitch + after

    groups: list[_Group] = []
    for (look, shaped), path in paths.items():
        embolden = look.embolden
        drawn = _emboldened(path, embolden) if embolden > 0.0 else path
        groups.append((drawn, look, frame if shaped else None))
    return groups, (left, top, left + widest, top + block_height)


def _shaped(
    glyph: QPainterPath, centre_x: float, centre_y: float, shape: tuple[float, float, float]
) -> QPainterPath:
    """``<tw>`` ``<th>`` ``<tr>`` 字の枠の真ん中を中心に伸び縮みさせてから時計回りに回す

    見本 tag36 と tag49 で、送り幅と行の高さの枠の真ん中が動かなかった 字の形の真ん中を
    中心にすると、下の深さの無い H では縦に縮めた字が 2 画素ほど上へずれる
    """
    across, tall, turn = shape
    transform = QTransform()
    transform.translate(centre_x, centre_y)
    # 画面の Y は下向きなので、Qt の正の角度がそのまま時計回りになる
    transform.rotate(turn)
    transform.scale(across, tall)
    transform.translate(-centre_x, -centre_y)
    return transform.map(glyph)


def _revealed_lines(lines: list[TaggedLine], values: dict[str, object]) -> list[TaggedLine]:
    """制御文字を読んだ後の文字送り 数え方は :func:`_revealed` と同じ（改行は数えない）

    出た文字だけで行を組む まだ出ていない後ろの行は行の高さにも枠にも数えない
    これは制御文字を読む前の組み方（:func:`_revealed` で切った本文を行に分けて組んでいた）を
    そのまま写した物で、AviUtl2 で測った決まりではない 文字送りは Sashimono だけの設定で
    （AviUtl の表示速度は読み込んでいない）、合わせる相手の振る舞いが無い 根拠無しに変えると、
    前の版で作った文字送りの字幕の位置が動く

    切れ目の扱いも :func:`_revealed` と同じにする 改行はいつでも通し、出す文字が尽きた後に
    次の字に当たった所で止める 行末でちょうど尽きると、次の行が空の行として 1 つ残る
    """
    ratio = _number(values, "reveal", 100.0) / 100.0
    if ratio >= 1.0:
        return lines
    if ratio <= 0.0:
        return []
    total = sum(len(line.text) for line in lines)
    visible = round(total * ratio)
    shown: list[TaggedLine] = []
    for line in lines:
        kept = TaggedLine(end=line.end)
        shown.append(kept)
        for run in line.runs:
            if not run.text:
                continue
            if visible <= 0:
                # 次の字に当たった所で止める 空の行になったときは、止まった所の見た目の高さ
                kept.end = run.style
                return shown
            piece = run.text[:visible]
            visible -= len(piece)
            kept.runs.append(TextRun(piece, run.style))
            if len(piece) < len(run.text):
                return shown
    return shown


def _external_leading(font: QFont) -> float:
    """行の下に足す余白（画素） Windows の GDI が ``tmExternalLeading`` と呼ぶもの

    ``hhea`` の上の高さ + 下の深さ + 行間の余白が、``OS/2`` の win の上下より長い分
    Arial 120 で 3.9 画素 AviUtl2 の枠は 139 で、Qt の行の高さ（134、余白 0）より
    これだけ高かった Qt は余白を 0 として返すので、表から読む
    """
    raw = QRawFont.fromFont(font)
    hhea = bytes(raw.fontTable("hhea").data())
    os2 = bytes(raw.fontTable("OS/2").data())
    if len(hhea) < 10 or len(os2) < 78 or raw.unitsPerEm() <= 0:
        return 0.0
    ascender, descender, gap = struct.unpack(">hhh", hhea[4:10])
    win_ascent, win_descent = struct.unpack(">HH", os2[74:78])
    extra = (ascender - descender + gap) - (win_ascent + win_descent)
    return max(0.0, float(extra) * float(raw.pixelSize()) / float(raw.unitsPerEm()))


def _graphemes(line: str) -> list[str]:
    """見た目の 1 文字ずつに分ける

    コードポイントで分けると、サロゲートペアや結合文字（濁点の付く仮名、絵文字の
    修飾）が 2 つに割れ、別々に置かれて崩れる
    """
    finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Grapheme, line)
    # 境目の位置は UTF-16 の数え方で返る Python の文字列の添字（コードポイント）で
    # 切ると、絵文字より後ろの位置が 1 つずつずれる
    units = line.encode("utf-16-le")
    parts: list[str] = []
    start = 0
    while (end := finder.toNextBoundary()) != -1:
        if end > start:
            parts.append(units[2 * start : 2 * end].decode("utf-16-le"))
        start = end
    return parts


def _emboldened(path: QPainterPath, amount: float) -> QPainterPath:
    """字の輪郭を右と上へ ``amount`` だけ太らせる（AviUtl2 の太字）

    左端と下端（ベースライン）は動かさない 両側へ太らせると、左の余白と下の位置が
    AviUtl2 とずれる 角は丸めない 丸めると「田」の角が AviUtl2 より甘くなる
    """
    stroker = QPainterPathStroker()
    stroker.setWidth(amount)
    stroker.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    grown = path.united(stroker.createStroke(path))
    grown.translate(amount / 2.0, -amount / 2.0)
    return grown


def _revealed(text: str, values: dict[str, object]) -> str:
    """文字送り 先頭から指定の割合だけを出す

    テロップを 1 文字ずつ出す表現は AviUtl でも定番で、こちらでもキーフレームを
    打てば同じことができる 改行は文字数に数えない 数えると、行が変わる瞬間に
    見た目の速度が変わる
    """
    ratio = float(values.get("reveal", 100.0)) / 100.0  # type: ignore[arg-type]
    if ratio >= 1.0:
        return text
    if ratio <= 0.0:
        return ""

    visible = round(len([c for c in text if c != chr(10)]) * ratio)
    shown: list[str] = []
    for character in text:
        if character == chr(10):
            shown.append(character)
            continue
        if visible <= 0:
            break
        shown.append(character)
        visible -= 1
    return "".join(shown)


def _vertical_text_path(
    text: str,
    font: QFont,
    metrics: QFontMetricsF,
    values: dict[str, object],
    width: int,
    height: int,
) -> QPainterPath:
    """縦書きの字の輪郭 行は右から左へ並べる

    Qt に縦書きの組版は無いので、1 文字ずつ縦に置く 日本語のテロップでは
    使う場面がはっきりあるので、簡素でも入れておく
    """
    columns = text.split(chr(10))
    advance = metrics.height() + float(values.get("letter_spacing", 0.0))  # type: ignore[arg-type]
    column_width = metrics.height() + float(values.get("line_spacing", 0.0))  # type: ignore[arg-type]

    centre_x = width / 2.0 + float(values.get("pos_x", 0.0))  # type: ignore[arg-type]
    centre_y = height / 2.0 - float(values.get("pos_y", 0.0))  # type: ignore[arg-type]
    tallest = max((len(column) for column in columns), default=0)
    # 塊の基準は横書きと同じ ``anchor`` と ``valign`` 既定（中・中）は塊の真ん中が位置で、
    # 前の置き方と同じ 見ずにいると、左上を基準にした縦書きも真ん中に置かれる
    block_width = column_width * len(columns)
    block_left = centre_x - block_width * _HORIZONTAL_SHARE.get(
        str(values.get("anchor", "center")), 0.5
    )
    # 右端の列の真ん中 列は右から左へ並ぶ
    left = block_left + block_width - column_width / 2.0
    top = centre_y - advance * tallest * _VERTICAL_SHARE.get(
        str(values.get("valign", "middle")), 0.5
    )

    path = QPainterPath()
    for column_index, column in enumerate(columns):
        x = left - column_width * column_index
        for row_index, character in enumerate(column):
            baseline = top + advance * row_index + metrics.ascent()
            offset = metrics.horizontalAdvance(character) / 2.0
            path.addText(QPointF(x - offset, baseline), font, character)
    return path


def _group_layers(groups: list[_Group], values: dict[str, object]) -> list[_Layer]:
    """制御文字で色や装飾を変えた字を、見た目ごとの層にする 影・縁・塗りの順は全体で守る"""
    return [
        (path, _redecorated(_recoloured(values, look.color, look.edge), look), clip)
        for path, look, clip in groups
    ]


#: 文字装飾の飾りの項目 ``<@書体,番号>`` で装飾を変えるときに一度外してから付け直す
_DECORATION_KEYS = ("border_width", "shadow_x", "shadow_y", "shadow_blur")


def _redecorated(values: dict[str, object], look: _Look) -> dict[str, object]:
    """``<@書体,番号>`` の文字装飾と ``<s,,,太さ>`` の縁取りの太さを当てた値

    飾りの割合は設定欄の文字装飾と同じ表（:data:`sashimono.compat.decoration.DECORATIONS`）
    から、設定欄の大きさで引く 大きさを変えた字の縁の太さは測っていないので、設定欄の字と
    揃えておく（:func:`sashimono.compat.aviutl.mapping._decoration_of` と同じ決まり）
    """
    if look.decoration is None and look.edge_width is None:
        return values
    changed = dict(values)
    if look.decoration is not None:
        decoration = find_decoration(DECORATION_NAMES[look.decoration])
        if decoration is not None:
            for key in _DECORATION_KEYS:
                changed.pop(key, None)
            colour = _color(values.get("border_color"))
            rgba = (colour.redF(), colour.greenF(), colour.blueF(), 1.0)
            params = decoration_params(decoration, _number(values, "size", 64.0), rgba)
            # 描く所の値は解けた数 動く値の形のまま混ぜると数として読めない
            changed.update(
                {
                    key: value.static if isinstance(value, AnimatedValue) else value
                    for key, value in params.items()
                }
            )
    if look.edge_width is not None and _number(changed, "border_width", 0.0) > 0.0:
        # 縁は片側に指定の半分ずつ付いた（見本 tag44 0・4・8・20 で字の幅が 0・4・8・19 広がった）
        changed["border_width"] = look.edge_width / 2.0
    return changed


def _paint_layers(
    painter: QPainter,
    layers: list[tuple[QPainterPath, dict[str, object], QPainterPath | None]],
) -> None:
    """影を全部、縁を全部、塗りを全部の順に描く

    色が 1 つのときと同じ順にする 色ごとに影・縁・塗りを描き切ると、字が近い所で
    後の色の縁が前の色の塗りに被さり、色を変えただけで前の字が欠ける
    3 つ目の形があれば、影・縁・塗りのどれもその内側だけに描く（変形した字を文字の枠で切る）
    """
    for path, look, clip in layers:
        made = _shadow_layer(path, look, painter)
        if made is not None:
            shadow, (shadow_left, shadow_top) = made
            # 影の面は絵の画素で作ってある 描く座標の縮め方を外してから重ねる
            # 外さないと、画質を落としたプレビューで影の面がもう 1 度縮む
            # 切る形は縮め方を外す前に当てる（当てたときの座標で持たれる）
            painter.save()
            if clip is not None:
                painter.setClipPath(clip)
            painter.resetTransform()
            painter.drawImage(shadow_left, shadow_top, shadow)
            painter.restore()
    for path, look, clip in layers:
        border_width = float(look.get("border_width", 0.0))  # type: ignore[arg-type]
        if border_width > 0:
            _fill_within(
                painter, _stroke(path, border_width), _color(look.get("border_color")), clip
            )
    for path, look, clip in layers:
        _fill_within(painter, path, _color(look.get("color")), clip)


def _fill_within(
    painter: QPainter, path: QPainterPath, colour: QColor, clip: QPainterPath | None
) -> None:
    """``path`` を塗る ``clip`` があればその内側だけ"""
    if clip is None:
        painter.fillPath(path, colour)
        return
    painter.save()
    painter.setClipPath(clip)
    painter.fillPath(path, colour)
    painter.restore()


def _recoloured(
    values: dict[str, object], color: str | None, edge: str | None
) -> dict[str, object]:
    """``<#文字色,影・縁色>`` で変えた色を当てた値 不透明度は設定欄のまま

    影・縁色は縁取りと影の両方の色（設定欄の ``影・縁色`` と同じ） 影は装飾ごとの
    薄さ（不透明度）を持つので、色の 3 成分だけを差し替える
    """
    if color is None and edge is None:
        return values
    changed = dict(values)
    for name, hex_value in (("color", color), ("border_color", edge), ("shadow_color", edge)):
        if hex_value is None:
            continue
        old = values.get(name)
        alpha = float(old[3]) if isinstance(old, tuple) and len(old) >= 4 else 1.0
        rgb = tuple(int(hex_value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        changed[name] = (*rgb, alpha)
    return changed


def _stroke(path: QPainterPath, width: float) -> QPainterPath:
    """輪郭を太らせたパス

    太さは輪郭の中心から両側へ広がるので、指定の 2 倍にして外側に指定幅を出す
    """
    stroker = QPainterPathStroker()
    stroker.setWidth(width * 2.0)
    stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return stroker.createStroke(path)


def _shadow_layer(
    path: QPainterPath, values: dict[str, object], painter: QPainter
) -> tuple[QImage, tuple[int, int]] | None:
    """文字の影を別の面に描いて、面と面を置く左上（絵の画素）を返す 影が無ければ ``None``

    ぼかしのために 1 枚離す 影は単色なので、ぼかすのは不透明度だけでよく、
    色の 3 成分はそのままにできる RGB ごとぼかすと、縁で色がにじむ

    面は ``painter`` の描く先と同じ大きさ（絵の画素）で作り、同じ縮め方で描く
    ぼかしの幅も絵の画素へ直す 画面の画素のままぼかすと、画質を落としたプレビューで
    影だけ 2 倍・4 倍にぼける

    画面より大きく広げた絵（``_crop_shadow``）では、面を影の周りだけにする 広げた絵は
    流れるテロップで一辺 8000 近くになり、全体をぼかすと 1 枚で 0.4 秒かかった
    画面に収まる絵は今までどおり絵の全体で作る ぼかしの足し算の順が変わると、
    端の濃さが 1 だけ変わることがあり、収まる字の見た目を 1 画素も変えないため
    """
    offset_x = float(values.get("shadow_x", 0.0))  # type: ignore[arg-type]
    offset_y = float(values.get("shadow_y", 0.0))  # type: ignore[arg-type]
    blur = float(values.get("shadow_blur", 0.0))  # type: ignore[arg-type]
    colour = _color(values.get("shadow_color"))
    if (offset_x, offset_y, blur) == (0.0, 0.0, 0.0) or colour.alpha() == 0:
        return None

    device = painter.device()
    assert device is not None
    transform = painter.transform()
    shifted = QPainterPath(path)
    # 画面の Y は下向き 設定の Y は上向きなので符号を反転する
    shifted.translate(offset_x, -offset_y)
    blur *= math.sqrt(abs(transform.determinant()))
    border_width = float(values.get("border_width", 0.0))  # type: ignore[arg-type]
    outline = _stroke(shifted, border_width) if border_width > 0 else None

    left, top, right, bottom = 0, 0, device.width(), device.height()
    if bool(values.get("_crop_shadow", False)):
        drawn = transform.mapRect((outline if outline is not None else shifted).boundingRect())
        # 箱ぼかしを 2 回掛けるので、ぼかしの幅の 2 倍まで広がる 端の 0 の所も同じ値に
        # なるよう、さらに少し余らせる
        margin = 2.0 * math.ceil(blur) + 4.0
        left = max(left, math.floor(drawn.left() - margin))
        top = max(top, math.floor(drawn.top() - margin))
        right = min(right, math.ceil(drawn.right() + margin))
        bottom = min(bottom, math.ceil(drawn.bottom() + margin))
        if right <= left or bottom <= top:
            return None
    layer = QImage(right - left, bottom - top, QImage.Format.Format_RGBA8888)
    layer.fill(Qt.GlobalColor.transparent)

    shadow_painter = QPainter(layer)
    shadow_painter.setTransform(transform * QTransform.fromTranslate(-left, -top))
    shadow_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    if outline is not None:
        # 縁取りがあるときは、その外形の影が落ちる 塗りだけの影にすると
        # 縁の分だけ影が細く見える
        shadow_painter.fillPath(outline, colour)
    shadow_painter.fillPath(shifted, colour)
    shadow_painter.end()

    if blur <= 0.0:
        return layer, (left, top)
    return _blur_alpha(layer, blur), (left, top)


def _blur_alpha(image: QImage, radius: float) -> QImage:
    """不透明度だけを平均化する

    箱ぼかしを縦横 2 回ずつ掛ける 厳密なガウスではないが、影の輪郭を
    やわらげる用途では見分けが付かず、こちらは掛け算が要らない
    """
    # 窓の幅は絵の大きさまで 壊れたファイルの巨大な値でも、絵より広くぼかす意味は無い
    # （そのまま渡すと、埋めた配列が何百 GB にもなって落ちる）
    span = max(1, min(round(radius), max(image.width(), image.height())))
    # ``_to_array`` は元のバッファをそのまま見ていることがあり、書き込めない
    array = _to_array(image).copy()
    alpha = array[:, :, 3].astype(np.float32)
    for _ in range(2):
        alpha = _box_blur(alpha, span)
    array[:, :, 3] = np.clip(alpha, 0.0, 255.0).astype(np.uint8)
    return QImage(
        array.tobytes(), image.width(), image.height(), QImage.Format.Format_RGBA8888
    ).copy()


def _box_blur(values: np.ndarray, span: int) -> np.ndarray:
    """縦横に窓幅 ``2*span+1`` の移動平均を掛ける

    累積和で求めるので、窓の幅を広げても速さは変わらない
    """
    result = values
    for axis in (0, 1):
        padded = np.pad(result, [(span, span) if a == axis else (0, 0) for a in (0, 1)], "edge")
        cumulative = np.cumsum(padded, axis=axis)
        zero = np.zeros_like(np.take(cumulative, [0], axis=axis))
        cumulative = np.concatenate([zero, cumulative], axis=axis)
        length = result.shape[axis]
        upper = np.take(cumulative, range(2 * span + 1, 2 * span + 1 + length), axis=axis)
        lower = np.take(cumulative, range(0, length), axis=axis)
        result = (upper - lower) / (2 * span + 1)
    return result


def _draw_shape(painter: QPainter, values: dict[str, object], width: int, height: int) -> None:
    shape_width = max(1.0, float(values.get("width", 400)))  # type: ignore[arg-type]
    shape_height = max(1.0, float(values.get("height", 400)))  # type: ignore[arg-type]
    centre_x = width / 2.0 + float(values.get("pos_x", 0.0))  # type: ignore[arg-type]
    centre_y = height / 2.0 - float(values.get("pos_y", 0.0))  # type: ignore[arg-type]

    if str(values.get("shape", "rect")) == "background":
        # 背景は大きさの設定を見ず、絵の全体を塗る 設定の大きさで描くと、画面より
        # 大きく広げた絵や、大きさを持たない読み込み元（YMM4 の背景）で隙間が出る
        shape_width = float(width) + abs(float(values.get("pos_x", 0.0))) * 2.0  # type: ignore[arg-type]
        shape_height = float(height) + abs(float(values.get("pos_y", 0.0))) * 2.0  # type: ignore[arg-type]
    if str(values.get("shape", "rect")) == "polyline":
        _draw_polyline(painter, values, centre_x, centre_y, width, height)
        return
    if str(values.get("shape", "rect")) == "concentration":
        _draw_concentration(painter, values, centre_x, centre_y, width, height)
        return
    if str(values.get("shape", "rect")) == "motion_trail":
        _draw_motion_trail(painter, values, width, height)
        return
    if str(values.get("shape", "rect")) == "starfield":
        _draw_star_field(painter, values, width, height)
        return
    if str(values.get("shape", "rect")) == "waveform":
        _draw_waveform(painter, values, centre_x, centre_y, shape_width, shape_height)
        return
    rect = QRectF(-shape_width / 2.0, -shape_height / 2.0, shape_width, shape_height)
    path = _shape_path(str(values.get("shape", "rect")), rect, values)

    transform = QTransform()
    transform.translate(centre_x, centre_y)
    transform.rotate(float(values.get("rotation", 0.0)))  # type: ignore[arg-type]
    path = transform.map(path)

    color = _color(values.get("color"))
    line_width = float(values.get("line_width", 0.0))  # type: ignore[arg-type]
    inside = str(values.get("line_align", "center")) == "inside"

    if bool(values.get("outline_only", False)):
        edge = max(line_width, 1.0)
        if inside:
            painter.fillPath(_inside_outline(path, edge), color)
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, edge))
        painter.drawPath(path)
        return

    painter.fillPath(path, color)
    if line_width > 0 and not inside:
        # 内側に引くときは塗りつぶしの中に収まるので、引き直す意味が無い
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, line_width))
        painter.drawPath(path)


def _inside_outline(path: QPainterPath, width: float) -> QPainterPath:
    """輪郭を図形の**内側**だけに引いた形

    AviUtl2 の図形のライン幅は内側に引かれる 外形は塗りつぶしたときと同じで、
    太さを増やすと内側の穴が小さくなる（三角形 サイズ 400 ライン幅 20 を
    AviUtl2 に描かせると、外形は塗りつぶしと同じ x±173・y -200..+99 のまま、
    横に切った線の帯が 24 画素 = 20 / sin60 だった）

    輪郭の中央に太さぶんを引いて図形と重ねる 塗りの上に線を引くやり方では
    外へ太さの半分（20 なら約 10 画素）はみ出し、図形が一回り大きく見える
    ``QPainter`` の切り抜き（``setClipPath``）だと外形の縁がぎざぎざになるので、
    形どうしを重ねてから塗る
    """
    return _stroke(path, width).intersected(path)


def polyline_points(text: str) -> list[tuple[float, float]]:
    """``"x,y;x,y"`` を点の並びへ 読めない組は飛ばす 座標は中心からの画素で Y は上が正

    点の数は :data:`MAX_POLYLINE_POINTS` で頭を抑える 壊れたファイルや巨大な線を
    読み込むと、1 フレームごとに点列を解き直して再生が止まる
    """
    points: list[tuple[float, float]] = []
    # 区切りの数を抑えたうえで、余りの 1 つ（上限より後ろの全部）は捨てる
    # 残しておくと、読めない組があったときに巨大な余りをカンマで割ってしまう
    for pair in text.split(";", MAX_POLYLINE_POINTS)[:MAX_POLYLINE_POINTS]:
        parts = pair.split(",", 2)
        if len(parts) != 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(y):
            points.append((x, y))
    return points


def centred_points(
    values: Mapping[str, object], width: float, height: float
) -> list[tuple[float, float]]:
    """線の点を、中心からの画素（Y は上が正）でそろえて返す

    ``points_from`` が ``corner`` なら点は画面の左上からの画素で Y は下が正（YMM4 のペン）
    画面の大きさはここで引く YMM4 に同じ点を 1920x1080 と 1280x720 で描かせると、どちらも
    画面の左上から同じ画素の所に線が出た（#198） 1920x1080 の真ん中を決め打ちで引くと、
    ほかの大きさの画面で線が 320・180 ずれる
    """
    points = polyline_points(str(values.get("points", "")))
    if str(values.get("points_from", "center")) == "corner":
        return [(x - width / 2.0, height / 2.0 - y) for x, y in points]
    return points


def corner_points_centred(source: GeneratedSource, width: int, height: int) -> GeneratedSource:
    """画面の左上から数えた線の点を、画面 ``width`` x ``height`` の中心からの点へ直す

    それ以外の生成オブジェクトはそのまま返す 描く所（:func:`render_source_framed`）が
    受け取るのは広げた絵や縮めたプレビューの大きさで、画面の左上が分からないため、
    画面の大きさを知っているレンダラが描く前に呼ぶ
    """
    if source.kind != "shape" or source.params.get("points_from") != "corner":
        return source
    points = centred_points(dict(source.params), width, height)
    text = ";".join(f"{x:g},{y:g}" for x, y in points)
    return source.with_param("points", text).with_param("points_from", "center")


def _polyline_path(
    values: dict[str, object],
    centre_x: float,
    centre_y: float,
    width: float = 0.0,
    height: float = 0.0,
) -> QPainterPath:
    points = [
        # 点も Y は上が正 ほかの位置の設定と向きがそろう
        (centre_x + x, centre_y - y)
        for x, y in centred_points(values, width, height)
    ]
    path = QPainterPath()
    if len(points) < 2:
        return path
    closed = bool(values.get("closed", False))
    path.moveTo(*points[0])
    if str(values.get("line_type", "straight")) == "quadratic" and len(points) >= 3:
        # 2 次ベジェ 点を 1 つおきに制御点として読む（YMM4 の QuadraticBezier）
        # 閉じるときは始点へ戻る曲線にする
        sequence = [*points, points[0]] if closed else points
        index = 1
        while index + 1 < len(sequence):
            control, end = sequence[index], sequence[index + 1]
            path.quadTo(control[0], control[1], end[0], end[1])
            index += 2
        if index < len(sequence):
            path.lineTo(*sequence[index])
    else:
        for point in points[1:]:
            path.lineTo(*point)
    if closed:
        path.closeSubpath()
    return path


def _draw_concentration(
    painter: QPainter,
    values: dict[str, object],
    centre_x: float,
    centre_y: float,
    width: int,
    height: int,
) -> None:
    """集中線（YMM4 の ConcentrationLine） 中心から放つ細い三角を、半径で濃さを変えて描く

    本数・太さ・長さ・ぼかしの効き方は YMM4 に描かせた絵から近づけた ぼかし 0 は大きさの
    半分の円の中に硬い線、ぼかすと線は画面の外まで伸びて、中心側がぼんやり抜ける
    """
    if bool(values.get("fill_frame", False)):
        _draw_concentration_frame(painter, values, centre_x, centre_y, width, height)
        return
    radius = max(_number(values, "width", 400.0) * 0.5, 1.0)
    count = max(1, min(1000, int(_number(values, "density", 80.0))))
    thickness = _number(values, "line_thickness", 50.0) / 100.0
    length = float(values.get("line_length", 70.0)) / 100.0  # type: ignore[arg-type]
    soft = max(0.0, min(1.0, float(values.get("softness", 50.0)) / 100.0))  # type: ignore[arg-type]
    flicker = float(values.get("flicker", 5.0))  # type: ignore[arg-type]
    seconds = float(values.get("_seconds", 0.0))  # type: ignore[arg-type]
    tick = int(seconds * flicker) if flicker > 0 else 0
    # 線の向き・太さ・根元の位置 切り替えの回ごとに引き直す
    dice = unit_randoms(tick * 7919 + 17, count, 3)

    inner = radius * (1.0 - length)
    far = radius * (1.0 + soft * 2.0) if soft > 0 else radius
    reach = max(far, float(np.hypot(width, height)))
    gradient = QRadialGradient(centre_x, centre_y, far)
    colour = _color(values.get("color"))
    # ぼかすと線は半透明になる（YMM4 の絵は白い線でも 170 ほどで、真っ白にならない）
    colour.setAlphaF(colour.alphaF() * (1.0 - soft * 0.45))
    clear = QColor(colour)
    clear.setAlpha(0)
    if soft <= 0:
        gradient.setColorAt(0.0, colour)
        gradient.setColorAt(1.0, colour)
    else:
        start = max(0.0, (inner - radius * soft) / far)
        full = min(1.0, (inner + radius * soft) / far)
        gradient.setColorAt(0.0, clear)
        gradient.setColorAt(start, clear)
        gradient.setColorAt(max(full, start + 1e-3), colour)
        gradient.setColorAt(min(1.0, radius / far), colour)
        gradient.setColorAt(1.0, clear)
    path = QPainterPath()
    spacing = 2.0 * np.pi / count
    for line in range(count):
        angle = dice[line, 0] * 2.0 * np.pi
        half = spacing * thickness * (0.2 + dice[line, 1] * 0.8) * 0.5
        start_radius = inner * (0.8 + dice[line, 2] * 0.4) if soft <= 0 else 0.0
        end_radius = radius if soft <= 0 else reach
        tip_x = centre_x + np.cos(angle) * start_radius
        tip_y = centre_y + np.sin(angle) * start_radius
        path.moveTo(tip_x, tip_y)
        path.lineTo(
            centre_x + np.cos(angle - half) * end_radius,
            centre_y + np.sin(angle - half) * end_radius,
        )
        path.lineTo(
            centre_x + np.cos(angle + half) * end_radius,
            centre_y + np.sin(angle + half) * end_radius,
        )
        path.closeSubpath()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(gradient))
    painter.drawPath(path)


#: 真ん中の空きを持つ集中線の、1 本あたりの薄さ
#: AviUtl2 の絵で、線が 1 枚だけ載っている所の明るさが 255 中の 67 ほどだった
CONCENTRATION_LINE_ALPHA = 67.0 / 255.0


def _draw_concentration_frame(
    painter: QPainter,
    values: dict[str, object],
    centre_x: float,
    centre_y: float,
    width: int,
    height: int,
) -> None:
    """画面いっぱいの集中線（AviUtl の ``集中線``）

    YMM4 のものとは絵の作りが違うので分けてある AviUtl2 に ``中心幅`` と ``濃さ``
    を変えた 3 本を描かせて測った結果、

    * ``中心幅`` は真ん中の空きの **半径**（px） 300 → 341、600 → 599、100 → 119
    * 線は空きの縁から画面の外まで伸びる 半径を変えても角度の占有率が変わらない
      ので、中心から広がる三角ではなく、中心を頂点とする扇（角度が一定）
    * ``濃さ`` は本数と 1 本の太さの両方に効く 占有率が 40 → 18%、80 → 約 65%、
      160 → 100% と、おおよそ濃さの 2 乗で増える
    * 1 本は真っ白ではない 占有率 18% のときの明るさの平均が 255 中の 67 ほど
      重なった所だけが 200 を超えるので、薄い線を重ねている

    YMM4 の方の描き方（大きさの円に収め、ぼかしで中心を抜く）で代えると、
    小さな円が真ん中に浮くだけの別物になる
    """
    gap = max(0.0, _number(values, "center_gap", 300.0))
    count = max(1, min(1000, int(_number(values, "density", 64.0))))
    thickness = max(0.0, _number(values, "line_thickness", 30.0) / 100.0)
    flicker = _number(values, "flicker", 25.0)
    seconds = _number(values, "_seconds", 0.0)
    tick = int(seconds * flicker) if flicker > 0 else 0
    # 線の向きと太さ 切り替えの回ごとに引き直す
    dice = unit_randoms(tick * 7919 + 17, count, 2)

    colour = _color(values.get("color"))
    # 実測に合わせた 1 本あたりの薄さ 不透明で描くと、同じ占有率でも真っ白な絵になる
    colour.setAlphaF(colour.alphaF() * CONCENTRATION_LINE_ALPHA)
    # 画面の四隅まで届かせる 中心をずらしても端が空かないよう、ずれのぶんを足す
    reach = float(np.hypot(width, height)) + float(
        np.hypot(centre_x - width / 2.0, centre_y - height / 2.0)
    )
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(colour))
    spacing = 2.0 * np.pi / count
    for line in range(count):
        angle = dice[line, 0] * 2.0 * np.pi
        half = spacing * thickness * (0.2 + dice[line, 1] * 0.8) * 0.5
        lo, hi = angle - half, angle + half
        # 内も外も円弧で閉じる 直線（弦）で閉じると中心寄りに食い込み、
        # 太い線では空けたはずの真ん中に線が入る（外側の弦は中心を横切る）
        #
        # Qt の角度は度で、反時計回りが正 こちらの角度は下向きの Y で持っているので
        # 符号を反転してから渡す 走る量は hi − lo（内側は逆向きに戻る）
        start, sweep = float(np.degrees(-lo)), float(np.degrees(lo - hi))
        wedge = QPainterPath()
        wedge.arcMoveTo(_around(centre_x, centre_y, reach), start)
        wedge.arcTo(_around(centre_x, centre_y, reach), start, sweep)
        if gap > 0.0:
            wedge.arcTo(_around(centre_x, centre_y, gap), start + sweep, -sweep)
        else:
            wedge.lineTo(centre_x, centre_y)
        # 1 本ずつ描く 1 つのパスにまとめると、重なった所が塗り分けの規則で
        # 抜けたり、重ねても濃くならなかったりする
        painter.drawPath(wedge)


#: 移動軌跡の円を、設定の太さより半径でこれだけ細く押す
#: 実物の円（図形の画像）は、にじませた縁が画像の内側に収まっている ライン幅 16 の線を
#: 測ると、縁の行は 255 中の 133 と 165 で、塗り切った幅は 15 画素だった
#: 設定の太さどおりに押すと、何百回も重ねるうちに縁が塗り切られて 1 画素ずつ太る
_STAMP_EDGE = 0.5


def _trail_paths_of(
    store: TrailPaths, x_value: AnimatedValue, y_value: AnimatedValue
) -> Callable[[int], TrailPath]:
    """その動きの道を、要るフレームまで伸ばして返す係"""

    def positions(first: int, stop: int) -> np.ndarray:
        # 設定の Y は上が正 形の計算は実物のまま下が正で持つ
        xs = sample_value(x_value, first, stop)
        ys = -sample_value(y_value, first, stop)
        return np.stack([xs, ys], axis=1)

    def paths(frames: int) -> TrailPath:
        return store.get((x_value, y_value), positions, frames)

    return paths


def _draw_motion_trail(
    painter: QPainter, values: dict[str, object], width: int, height: int
) -> None:
    """移動軌跡（AviUtl2 の ``ライン(移動軌跡)``）
    形は :func:`~sashimono.engine.motion_shapes.trail`

    円は 1 つずつ重ねて描く 1 つのパスにまとめて塗ると、縁のにじみが 1 回分しか
    付かない 実物は細かい間隔で円を押し重ねるので、縁がくっきりする
    """
    motion = values.get("_motion")
    if not isinstance(motion, tuple):
        return
    x_value, y_value = motion
    store = values.get("_trail_paths")
    if not isinstance(store, TrailPaths):
        store = TrailPaths()

    def position(at: float) -> tuple[float, float]:
        # 設定の Y は上が正 形の計算は実物のまま下が正で持つ
        return x_value.at(at), -y_value.at(at)

    frame = _number(values, "_frame", 0.0)
    duration = _number(values, "_duration", 0.0)
    line_width = _number(values, "line_width", 16.0)
    if line_width <= 0:
        # 図形の線の太さの既定は 0（塗りつぶし）だが、移動軌跡で 0 は線が無い絵になる
        # 実物の ライン幅 は 2 から始まるので、0 以下は実物の既定の 16 として描く
        # 線を消したいときは 軌跡の点の大きさ を 0 にする
        line_width = 16.0
    head_size = max(0.0, _number(values, "trail_head_size", 48.0))
    shape = trail(
        position,
        frame=frame,
        # 長さが分からないときは今のフレームで止める 先の動きを勝手に読まない
        total=duration if duration > 0 else frame,
        line_width=line_width,
        interval=_number(values, "trail_interval", 10.0),
        min_step=_number(values, "trail_min_step", 2.0),
        fixed_speed=_number(values, "trail_speed", 0.0),
        head_size=head_size,
        head_angle=_number(values, "trail_head_angle", 0.0),
        head_offset=_number(values, "trail_head_offset", 70.0),
        paths=_trail_paths_of(store, x_value, y_value),
    )
    centre_x, centre_y = width / 2.0, height / 2.0
    colour = _color(values.get("color"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(colour))

    core = line_width * _number(values, "trail_core", 100.0) / 200.0 - _STAMP_EDGE
    if core > 0:
        for x, y in shape.stamps:
            painter.drawEllipse(QPointF(centre_x + x, centre_y + y), core, core)
    band = line_width * _number(values, "trail_band", 0.0) / 200.0
    if band > 0:
        for x0, y0, x1, y1 in shape.bands:
            length = math.hypot(x1 - x0, y1 - y0)
            if length <= 0:
                continue
            # 進む向きに直角な向きへ、太さの半分ずつ広げた四角
            normal_x, normal_y = -(y1 - y0) / length * band, (x1 - x0) / length * band
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(centre_x + x0 + normal_x, centre_y + y0 + normal_y),
                        QPointF(centre_x + x1 + normal_x, centre_y + y1 + normal_y),
                        QPointF(centre_x + x1 - normal_x, centre_y + y1 - normal_y),
                        QPointF(centre_x + x0 - normal_x, centre_y + y0 - normal_y),
                    ]
                )
            )
    # 今の位置には、点の大きさの設定に関係なく線の太さの円を置く（実物のまま）
    last = line_width / 2.0 - _STAMP_EDGE
    if last > 0:
        painter.drawEllipse(QPointF(centre_x + shape.last[0], centre_y + shape.last[1]), last, last)

    if shape.head is None:
        return
    rect = QRectF(-head_size / 2.0, -head_size / 2.0, head_size, head_size)
    path = _shape_path(str(values.get("trail_head_shape", "inscribed_triangle")), rect, values)
    transform = QTransform()
    transform.translate(centre_x + shape.head[0], centre_y + shape.head[1])
    transform.rotate(math.degrees(shape.head_turn))
    painter.fillPath(transform.map(path), colour)


def _draw_waveform(
    painter: QPainter,
    values: dict[str, object],
    centre_x: float,
    centre_y: float,
    width: float,
    height: float,
) -> None:
    """音声波形（AviUtl2 の ``音声波形表示``） 音はレンダラが ``_audio`` に入れて渡す

    音はフレームの時刻から届く 線はそこから横幅ぶん、スペクトラムは頭の
    :func:`~sashimono.engine.audio_shapes.spectrum_window` ぶんを使う
    音が無ければ何も描かない 実物も、再生範囲が 0 秒の見本では何も出さなかった
    """
    audio = values.get("_audio")
    if not isinstance(audio, np.ndarray) or audio.size <= 1:
        return
    volume = _number(values, "wave_volume", 100.0)
    # 升目の数は描く大きさより細かくしない 壊れた値で巨大な升目を作ると描画が止まる
    width = _within(width, 1.0, float(MAX_CANVAS), 800.0)
    height = _within(height, 1.0, float(MAX_CANVAS), 400.0)
    columns = round(_within(_number(values, "wave_columns", 0.0), 0.0, width, 0.0))
    rows = round(_within(_number(values, "wave_rows", 0.0), 0.0, height, 0.0))
    spectrum = bool(values.get("wave_spectrum", False))
    if columns <= 0 and rows <= 0 and not spectrum:
        # 升目を持たない線は、にじませて細く引く 升目の絵と同じ道を通すと、
        # 斜めの所がぎざぎざになる（実物の線は縁がにじんでいる）
        _draw_waveform_line(painter, values, audio, centre_x, centre_y, width, height, volume)
        return

    box_w, box_h = max(1, round(width)), max(1, round(height))
    grid_w = columns if columns > 0 else box_w
    grid_h = rows if rows > 0 else box_h
    gap_x = _number(values, "wave_gap_x", 0.0)
    gap_y = _number(values, "wave_gap_y", 0.0)
    if spectrum:
        rate = round(_number(values, "_audio_rate", 44100.0))
        # ミラーは線では絵が変わらなかった（p5 と p6 の見本） 効くのはスペクトラムだけ
        mirror = bool(values.get("wave_mirror", False))
        cells = spectrum_cells(spectrum_levels(audio, grid_w, rate, volume), grid_h, mirror=mirror)
        mask = bar_mask(cells, grid_h, box_w, box_h, gap_x, gap_y, mirror=mirror)
    else:
        lit = waveform_cells(audio, grid_w, grid_h, volume)
        mask = cell_mask(lit, box_w, box_h, gap_x, gap_y)
    if not mask.any():
        return
    colour = _color(values.get("color"))
    pixels = np.zeros((box_h, box_w, 4), dtype=np.uint8)
    pixels[mask] = (colour.red(), colour.green(), colour.blue(), colour.alpha())
    image = QImage(pixels.tobytes(), box_w, box_h, QImage.Format.Format_RGBA8888).copy()
    painter.drawImage(QPointF(centre_x - box_w / 2.0, centre_y - box_h / 2.0), image)


def _draw_waveform_line(
    painter: QPainter,
    values: dict[str, object],
    samples: np.ndarray,
    centre_x: float,
    centre_y: float,
    width: float,
    height: float,
    volume: float,
) -> None:
    points = waveform_points(samples, width, height, volume)
    if len(points) < 2:
        return
    line = QPolygonF([QPointF(centre_x + x, centre_y + y) for x, y in points])
    pen = QPen(_color(values.get("color")), WAVEFORM_LINE)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(pen)
    painter.drawPolyline(line)


#: 星空の粒の絵をこの大きさ（画素）まで半分ずつ縮めて用意しておく
#: 遠くの粒は数画素しかない 大きな絵を一度に縮めると間引きになり、粒が欠けてちらつく
_SMALLEST_SPRITE = 2


def _draw_star_field(painter: QPainter, values: dict[str, object], width: int, height: int) -> None:
    """星空（AviUtl2 の ``星``） 位置は :func:`~sashimono.engine.motion_shapes.star_field`

    粒は ``大きさ`` の図形を ``大きさ / 3`` だけぼかした絵 実物はぼかしで絵が広がり、
    その広がった絵を遠近で縮めて置く ぼかさずに置くと、遠くの粒が硬い点になる
    """
    # 大きさは設定の範囲（1〜100）へ収める キーフレームの値は範囲を守らないことがあり、
    # 無限大は粒の絵の大きさの計算で例外、巨大な値は巨大な絵を作って描画が止まる
    size = _within(_number(values, "star_size", 30.0), 1.0, 100.0, 30.0)
    field = star_field(
        seconds=_number(values, "_seconds", 0.0),
        count=_number(values, "star_count", 1500.0),
        speed=_number(values, "star_speed", 6.0),
        spread=_number(values, "star_spread", 12.0),
        depth=_number(values, "star_depth", 20.0),
        fade_in=_number(values, "star_fade_in", 0.15),
        fade_out=_number(values, "star_fade_out", 0.15),
        screen_width=float(width),
        screen_height=float(height),
    )
    if field.x.size == 0:
        return
    sprites = _star_sprites(values, size)
    base = float(sprites[0].width())
    centre_x, centre_y = width / 2.0, height / 2.0
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    for x, y, scale, alpha in zip(field.x, field.y, field.scale, field.alpha, strict=True):
        side = base * float(scale)
        # 画面に掛からない粒は描かない 手前へ来た粒は画面の何倍も外にある
        if (
            abs(float(x)) - side / 2.0 > centre_x
            or abs(float(y)) - side / 2.0 > centre_y
            or side < 0.05
        ):
            continue
        sprite = sprites[0]
        for smaller in sprites[1:]:
            if smaller.width() < side:
                break
            sprite = smaller
        painter.setOpacity(float(alpha))
        painter.drawImage(
            QRectF(centre_x + float(x) - side / 2.0, centre_y + float(y) - side / 2.0, side, side),
            sprite,
            QRectF(sprite.rect()),
        )
    painter.setOpacity(1.0)


def _star_sprites(values: dict[str, object], size: float) -> list[QImage]:
    """星の粒 1 つの絵 大きい順に、半分ずつ縮めたものを並べる"""
    blur = size / 3.0
    side = max(1, math.ceil(size + blur * 2.0))
    sprite = QImage(side, side, QImage.Format.Format_RGBA8888)
    # 透けた所も色は粒の色にしておく 黒のまま不透明度だけぼかすと、粒の縁が暗くにじむ
    clear = _color(values.get("color"))
    clear.setAlpha(0)
    sprite.fill(clear)
    sprite_painter = QPainter(sprite)
    sprite_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    rect = QRectF((side - size) / 2.0, (side - size) / 2.0, size, size)
    sprite_painter.fillPath(
        _shape_path(str(values.get("star_shape", "ellipse")), rect, values),
        _color(values.get("color")),
    )
    sprite_painter.end()
    # 箱ぼかしを 2 回 窓の半分の幅を ぼかしの半分にすると、広がりがぼかしの幅にそろう
    sprite = _blur_alpha(sprite, max(1.0, blur / 2.0))
    chain = [sprite]
    while chain[-1].width() // 2 >= _SMALLEST_SPRITE:
        last = chain[-1]
        chain.append(
            last.scaled(
                last.width() // 2,
                last.height() // 2,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
    return chain


#: 折れ線で読み取る点の数の上限 これ以上は捨てる（描画は 1 フレームごとに走る）
MAX_POLYLINE_POINTS = 4096

#: 線の一部を描き直すときの刻みの上限 長い線で刻みが増えると、1 フレームに何秒もかかる
MAX_TRIM_STEPS = 2000


def _trimmed(path: QPainterPath, start: float, end: float) -> QPainterPath:
    """線の途中だけを残す ``start`` と ``end`` は全長に対する 0..1"""
    if start <= 0.0 and end >= 1.0:
        return path
    if end <= start:
        return QPainterPath()
    total = path.length()
    trimmed = QPainterPath()
    # 2 画素ごとに点を打つ 長い線でも刻みが増えすぎないように頭を抑える
    steps = max(8, min(int(total / 2.0), MAX_TRIM_STEPS))
    first = True
    for index in range(steps + 1):
        fraction = start + (end - start) * index / steps
        point = path.pointAtPercent(path.percentAtLength(total * fraction))
        if first:
            trimmed.moveTo(point)
            first = False
        else:
            trimmed.lineTo(point)
    return trimmed


def _draw_polyline(
    painter: QPainter,
    values: dict[str, object],
    centre_x: float,
    centre_y: float,
    width: float,
    height: float,
) -> None:
    """線の図形 閉じていれば中を塗ってから線を引く 端と角は丸める（配布物はすべて丸）"""
    path = _polyline_path(values, centre_x, centre_y, width, height)
    if path.isEmpty():
        return
    trim_start = float(values.get("trim_start", 0.0)) / 100.0  # type: ignore[arg-type]
    trim_end = float(values.get("trim_end", 100.0)) / 100.0  # type: ignore[arg-type]
    path = _trimmed(path, max(0.0, trim_start), min(1.0, trim_end))
    if path.isEmpty():
        return
    if bool(values.get("closed", False)):
        fill = _color(values.get("fill_color"))
        if fill.alpha() > 0:
            painter.fillPath(path, fill)
    width = float(values.get("line_width", 0.0))  # type: ignore[arg-type]
    if width <= 0:
        return
    pen = QPen(_color(values.get("color")), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    dashes = [float(v) for v in str(values.get("dash", "")).split(",") if _is_number(v)]
    if len(dashes) >= 2 and all(v >= 0 for v in dashes) and sum(dashes) > 0:
        # 破線の長さは線の太さを 1 とする割合（Qt も YMM4 も同じ決まり）
        pen.setDashPattern(dashes)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(pen)
    painter.drawPath(path)


def _is_number(text: str) -> bool:
    try:
        return bool(np.isfinite(float(text)))
    except ValueError:
        return False


def _shape_path(kind: str, rect: QRectF, values: dict[str, object]) -> QPainterPath:
    path = QPainterPath()
    if kind == "ellipse":
        path.addEllipse(rect)
    elif kind == "rounded":
        radius = float(values.get("corner_radius", 24))  # type: ignore[arg-type]
        path.addRoundedRect(rect, radius, radius)
    elif kind == "triangle":
        path.moveTo(rect.center().x(), rect.top())
        path.lineTo(rect.right(), rect.bottom())
        path.lineTo(rect.left(), rect.bottom())
        path.closeSubpath()
    elif kind == "pentagon":
        path = _polygon_path(rect, 5)
    elif kind == "hexagon":
        path = _polygon_path(rect, 6)
    elif kind == "star":
        path = _star_path(rect)
    elif kind == "inscribed_triangle":
        # 頂点を上にして楕円に内接する三角形（YMM4 の三角形） 四角に合わせた三角形とは
        # 底辺の位置と幅が違う
        path = _polygon_path(rect, 3)
    elif kind == "fan":
        path = _fan_path(rect, float(values.get("span", 360.0)))  # type: ignore[arg-type]
    elif kind == "arrow":
        path = _arrow_path(
            rect,
            float(values.get("bar_length", 50.0)),  # type: ignore[arg-type]
            float(values.get("bar_thickness", 50.0)),  # type: ignore[arg-type]
        )
    elif kind == "superformula":
        path = _superformula_path(
            rect,
            float(values.get("formula_m", 4.0)),  # type: ignore[arg-type]
            float(values.get("formula_n", 1.0)),  # type: ignore[arg-type]
        )
    else:
        # ``background`` もここ 大きさは呼び出し側が画面いっぱいに指定する
        path.addRect(rect)
    return path


def _polygon_path(rect: QRectF, sides: int) -> QPainterPath:
    """正多角形 頂点を上に向けて置く

    AviUtl の五角形・六角形に対応する 頂点の向きを合わせておかないと、
    移植した資産で図形だけ傾いて見える
    """
    path = QPainterPath()
    radius_x, radius_y = rect.width() / 2.0, rect.height() / 2.0
    centre = rect.center()
    for index in range(sides):
        angle = 2.0 * np.pi * index / sides - np.pi / 2.0
        x = centre.x() + np.cos(angle) * radius_x
        y = centre.y() + np.sin(angle) * radius_y
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.closeSubpath()
    return path


def _star_path(rect: QRectF, points: int = 5) -> QPainterPath:
    """5 芒星 外周と内周の頂点を交互に結ぶ"""
    path = QPainterPath()
    outer_x, outer_y = rect.width() / 2.0, rect.height() / 2.0
    inner_ratio = 0.382  # 正五芒星の内接比
    centre = rect.center()

    for index in range(points * 2):
        angle = np.pi * index / points - np.pi / 2.0
        ratio = 1.0 if index % 2 == 0 else inner_ratio
        x = centre.x() + np.cos(angle) * outer_x * ratio
        y = centre.y() + np.sin(angle) * outer_y * ratio
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.closeSubpath()
    return path


def _fan_path(rect: QRectF, span: float) -> QPainterPath:
    """扇 上を 0 として反時計回りに ``span`` 度（YMM4 の CenterAngle）"""
    path = QPainterPath()
    path.moveTo(rect.center())
    # Qt の角度は右が 0 で反時計回り 上（90 度）から反時計回りに広げる
    path.arcTo(rect, 90.0, max(0.0, min(span, 360.0)))
    path.closeSubpath()
    return path


def _arrow_path(rect: QRectF, bar_length: float, bar_thickness: float) -> QPainterPath:
    """上向きの矢印 頭は楕円に内接する三角形、軸は頭の底辺から下へ伸びる

    軸の長さは半径の 3 倍を 100、太さは半径を 100 とする割合（YMM4 の絵に合わせた）
    """
    radius_x, radius_y = rect.width() / 2.0, rect.height() / 2.0
    centre = rect.center()
    base_y = centre.y() + radius_y * 0.5
    half_head = radius_x * np.sqrt(3.0) / 2.0
    half_bar = radius_x * max(bar_thickness, 0.0) / 200.0
    bar_end = base_y + radius_y * 3.0 * max(bar_length, 0.0) / 100.0
    path = QPainterPath()
    path.moveTo(centre.x(), centre.y() - radius_y)
    path.lineTo(centre.x() + half_head, base_y)
    path.lineTo(centre.x() + half_bar, base_y)
    path.lineTo(centre.x() + half_bar, bar_end)
    path.lineTo(centre.x() - half_bar, bar_end)
    path.lineTo(centre.x() - half_bar, base_y)
    path.lineTo(centre.x() - half_head, base_y)
    path.closeSubpath()
    return path


def _superformula_path(rect: QRectF, m: float, n: float) -> QPainterPath:
    """スーパーフォーミュラ（Gielis の式、n1 = n2 = n3 = n） 一番遠い点が楕円に届くよう縮める"""
    angles = np.linspace(0.0, 2.0 * np.pi, 721)
    exponent = max(abs(n), 0.05)
    quarter = m * angles / 4.0
    radius = (np.abs(np.cos(quarter)) ** exponent + np.abs(np.sin(quarter)) ** exponent) ** (
        -1.0 / exponent
    )
    radius = np.nan_to_num(radius, nan=0.0, posinf=0.0)
    peak = float(radius.max()) if radius.size else 1.0
    radius = radius / (peak if peak > 0 else 1.0)
    centre = rect.center()
    path = QPainterPath()
    for index, (angle, r) in enumerate(zip(angles, radius, strict=True)):
        x = centre.x() + np.cos(angle - np.pi / 2.0) * r * rect.width() / 2.0
        y = centre.y() + np.sin(angle - np.pi / 2.0) * r * rect.height() / 2.0
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.closeSubpath()
    return path


def _around(centre_x: float, centre_y: float, radius: float) -> QRectF:
    """中心と半径から、円弧を描くための四角"""
    return QRectF(centre_x - radius, centre_y - radius, radius * 2.0, radius * 2.0)


def _within(value: float, low: float, high: float, default: float) -> float:
    """範囲へ収める 無限大や非数は既定値にする（``round`` が例外で描画ごと止まる）"""
    if not math.isfinite(value):
        return default
    return min(max(value, low), high)


def _number(values: dict[str, object], name: str, default: float) -> float:
    """解けた設定から数を 1 つ読む

    :func:`_resolve` を通った後の値は数（か数の文字）になっているが、型としては
    ``object`` のまま ここで 1 か所にまとめておくと、読む側に
    理由の無い ``type: ignore`` を並べずに済む
    """
    value = values.get(name, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return default


def _color(value: object) -> QColor:
    """``(R, G, B, A)`` の 0..1（sRGB）を :class:`QColor` へ"""
    if not isinstance(value, tuple) or len(value) < 3:
        return QColor(255, 255, 255, 255)
    channels = [round(min(max(float(v), 0.0), 1.0) * 255) for v in value[:4]]
    while len(channels) < 4:
        channels.append(255)
    return QColor(*channels)


def _to_array(image: QImage) -> np.ndarray:
    """``QImage`` を ``(高さ, 幅, 4)`` の配列へ

    ``QImage`` は行ごとに詰め物を入れることがあるので、``bytesPerLine`` を見て
    余りを落とす 幅だけで整形すると絵が斜めにずれる
    """
    width, height = image.width(), image.height()
    buffer = bytes(image.constBits())
    stride = image.bytesPerLine()
    rows = np.frombuffer(buffer, dtype=np.uint8).reshape(height, stride)
    return np.ascontiguousarray(rows[:, : width * 4].reshape(height, width, 4))
