"""タイムラインの描画

``QGraphicsView`` を使わず自前で描く クリップが数千個になっても、見えている範囲
だけを描けば済むからで、シーングラフに全部を載せると生成だけで時間を食う

描画関数はウィジェットの状態を持たない 引数で受け取ったものだけを描くので、
書き出しプレビューや単体テストからも同じ関数を呼べる
"""

from __future__ import annotations

import bisect
import math
import weakref
from collections import OrderedDict
from collections.abc import Callable, Collection, Hashable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen

from sashimono.compat.aviutl.custom_object import (
    CUSTOM_OBJECT_LABEL,
    custom_object_script,
    script_label,
)
from sashimono.core.model import (
    Clip,
    ClipId,
    MediaItem,
    Timeline,
    Track,
    TrackKind,
    draws_picture,
    heard_stream,
    plays_sound,
)
from sashimono.core.timebase import FrameRate, format_timecode
from sashimono.effects.sources import source_registry
from sashimono.engine.audio import Waveform
from sashimono.engine.audio.shape import shape_envelope, shape_history_frames, shape_key
from sashimono.engine.cache import Filmstrip
from sashimono.ui.theme import Colors, Metrics
from sashimono.ui.timeline.layout import TimelineLayout, TrackBand

__all__ = [
    "ADD_TRACK_BUTTON_HEIGHT",
    "ADD_TRACK_BUTTON_SPACE",
    "ADD_TRACK_BUTTON_TEXT",
    "DENSE_MARK_WIDTH",
    "DETAIL_MIN_WIDTH",
    "TRACK_BUTTONS",
    "WAVEFORM_CACHE_BYTES",
    "WAVEFORM_IMAGE_MAX_COLUMNS",
    "ClipGlance",
    "clear_waveform_images",
    "clip_content",
    "clip_summary",
    "clips_in_range",
    "draw_clip",
    "draw_dense_clips",
    "draw_playhead",
    "draw_ruler",
    "draw_track_add_button",
    "draw_track_background",
    "draw_track_header",
    "filmstrip_tint",
    "shown_track_name",
    "to_qimage",
    "track_add_button_rect",
    "track_button_rects",
    "track_name_rect",
    "voice_label",
    "waveform_level",
]

#: 目盛りの間隔として使える値（フレーム数の基準となる秒数）
#: 1 目盛りが最低でもこのピクセル数を超えるものを選ぶ
_MIN_TICK_SPACING = 70
_TICK_SECONDS = (
    Fraction(1, 30),
    Fraction(1, 10),
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(1),
    Fraction(2),
    Fraction(5),
    Fraction(10),
    Fraction(15),
    Fraction(30),
    Fraction(60),
    Fraction(120),
    Fraction(300),
    Fraction(600),
    Fraction(1800),
    Fraction(3600),
)


def to_qimage(array: np.ndarray) -> QImage:
    """``(高さ, 幅, 4)`` の uint8 配列を :class:`QImage` にする

    ``QImage`` は渡したバッファを参照するだけでコピーしない 元の配列が
    先に解放されると描画時に落ちるので、必ずコピーを作って渡す
    """
    data = np.ascontiguousarray(array, dtype=np.uint8)
    height, width = data.shape[:2]
    image = QImage(data.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
    return image.copy()


def draw_ruler(painter: QPainter, layout: TimelineLayout, width: int, rate: FrameRate) -> None:
    """時間目盛りを描く"""
    rect = QRect(0, 0, width, Metrics.RULER_HEIGHT)
    painter.fillRect(rect, Colors.TIMELINE_RULER)
    painter.setPen(QPen(Colors.BORDER, 1))
    painter.drawLine(0, Metrics.RULER_HEIGHT - 1, width, Metrics.RULER_HEIGHT - 1)

    step = _tick_step(layout, rate)
    if step <= 0:
        return

    font = QFont(painter.font())
    font.setPointSizeF(8.5)
    painter.setFont(font)
    metrics = QFontMetrics(font)

    start_frame, end_frame = layout.visible_range(width)
    first_tick = (start_frame // step) * step
    for frame in range(first_tick, end_frame + step, step):
        x = layout.frame_to_x(frame)
        if x < Metrics.TRACK_HEADER_WIDTH - 1:
            continue
        painter.setPen(QPen(Colors.BORDER, 1))
        painter.drawLine(int(x), 4, int(x), Metrics.RULER_HEIGHT - 1)
        painter.setPen(QPen(Colors.TEXT_MUTED, 1))
        label = format_timecode(frame, rate)
        painter.drawText(QPointF(x + 3, Metrics.RULER_HEIGHT - 6 + metrics.descent() - 2), label)


def _tick_step(layout: TimelineLayout, rate: FrameRate) -> int:
    """目盛りの間隔（フレーム数）

    表示倍率に応じて、ラベルが重ならない中で最も細かい間隔を選ぶ
    """
    for seconds in _TICK_SECONDS:
        frames = max(1, int(seconds * rate.fps))
        if frames * layout.pixels_per_frame >= _MIN_TICK_SPACING:
            return frames
    return max(1, int(_TICK_SECONDS[-1] * rate.fps))


def draw_track_background(painter: QPainter, band: TrackBand, width: int) -> None:
    """トラック 1 本分の下地と区切り線"""
    painter.fillRect(
        QRect(Metrics.TRACK_HEADER_WIDTH, band.top, width, band.height),
        Colors.TIMELINE_BACKGROUND,
    )
    painter.setPen(QPen(Colors.TRACK_SEPARATOR, 1))
    painter.drawLine(0, band.bottom - 1, width, band.bottom - 1)


#: ヘッダの切り替えボタン（属性名、表示、説明、押している間の色）
#: 描画と当たり判定の両方がこの並びを使う
TRACK_BUTTONS: tuple[tuple[str, str, str, QColor], ...] = (
    ("muted", "M", "ミュート", Colors.TRACK_MUTE),
    # ソロは絵と音の役割の中で効く（Timeline の決まり） レイヤーのソロは映像トラックの絵と
    # 音声トラックの音の両方を止めるので、「同じ種類」とは書かない
    ("solo", "S", "ソロ（絵・音のそれぞれで、ほかのトラックを止める）", Colors.TRACK_SOLO),
    ("locked", "L", "ロック（クリップを動かせなくする）", Colors.TRACK_LOCK),
)

#: 名前の無いトラックに出す種類の言葉 2 択で書くと、名前の無いレイヤーが「音声」と出る
_KIND_NAMES = {TrackKind.VIDEO: "映像", TrackKind.AUDIO: "音声", TrackKind.MIXED: "レイヤー"}

_BUTTON_WIDTH = 18
_BUTTON_HEIGHT = 16
_BUTTON_GAP = 2


def track_button_rects(band: TrackBand) -> list[tuple[str, str, QRect]]:
    """ヘッダの切り替えボタンの位置 ``(属性名, 説明, 矩形)`` の並び

    名前と同じ行の右端に置く 名前の下の段に置くと、トラックを最小の高さ
    （28 画素）まで縮めたときにボタンがはみ出して押せなくなる
    """
    count = len(TRACK_BUTTONS)
    left = Metrics.TRACK_HEADER_WIDTH - 6 - count * _BUTTON_WIDTH - (count - 1) * _BUTTON_GAP
    return [
        (
            attribute,
            tip,
            QRect(
                left + index * (_BUTTON_WIDTH + _BUTTON_GAP),
                band.top + 5,
                _BUTTON_WIDTH,
                _BUTTON_HEIGHT,
            ),
        )
        for index, (attribute, _, tip, _) in enumerate(TRACK_BUTTONS)
    ]


def track_name_rect(band: TrackBand) -> QRect:
    """ヘッダの名前を書く所 左の余白からボタンの手前まで"""
    left = 8
    right = track_button_rects(band)[0][2].left() - 4
    return QRect(left, band.top + 5, right - left, _BUTTON_HEIGHT)


def shown_track_name(track: Track, metrics: QFontMetrics) -> str:
    """ヘッダに出す名前 描くのと試験が同じ物を見る

    収まらない名前は真ん中を詰める 末尾を詰めると「レイヤー 1」から「レイヤー 4」までが
    どれも「レイヤ…」になり、何番のレイヤーなのかが読めない（番号は名前の末尾にある）
    """
    name = track.name or _KIND_NAMES[track.kind]
    width = track_name_rect(TrackBand(track, 0, 0)).width()
    return metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, width)


def draw_track_header(painter: QPainter, band: TrackBand, *, active: bool = True) -> None:
    """トラック名と、ミュート・ソロ・ロックの切り替えボタン

    ``active`` が偽なら名前を薄くする ミュートだけでなく、ほかのトラックの
    ソロで止まっている場合も同じ見た目にする どちらも「いま出ていない」ことに
    変わりはなく、ボタンの色だけでは後者に気付けない
    """
    rect = QRect(0, band.top, Metrics.TRACK_HEADER_WIDTH, band.height)
    painter.fillRect(rect, Colors.TRACK_HEADER)
    painter.setPen(QPen(Colors.BORDER, 1))
    painter.drawLine(
        Metrics.TRACK_HEADER_WIDTH - 1, band.top, Metrics.TRACK_HEADER_WIDTH - 1, band.bottom
    )

    track = band.track
    buttons = track_button_rects(band)
    painter.setPen(QPen(Colors.TEXT if active else Colors.TEXT_MUTED, 1))
    painter.drawText(
        track_name_rect(band),
        Qt.AlignmentFlag.AlignVCenter,
        shown_track_name(track, QFontMetrics(painter.font())),
    )

    font = QFont(painter.font())
    font.setPointSizeF(7.5)
    font.setBold(True)
    painter.save()
    painter.setFont(font)
    for (attribute, _, button), (_, letter, _, colour) in zip(buttons, TRACK_BUTTONS, strict=True):
        on = bool(getattr(track, attribute))
        if on:
            painter.fillRect(button, colour)
        painter.setPen(QPen(colour if on else Colors.BORDER, 1))
        painter.drawRect(button.adjusted(0, 0, -1, -1))
        painter.setPen(QPen(Colors.TRACK_TOGGLE_TEXT if on else Colors.TEXT_MUTED, 1))
        painter.drawText(button, Qt.AlignmentFlag.AlignCenter, letter)
    painter.restore()


#: 「＋ トラック追加」の高さと、最後のトラックとの間（画素）
ADD_TRACK_BUTTON_HEIGHT = 22
_ADD_TRACK_BUTTON_GAP = 6

#: ボタンがトラックの帯の下に取る高さ 上の間とボタンと、下にも同じ間を空ける
#: 縦スクロールの範囲に足す（:meth:`TimelineView._scrollable_height`） 足さないと、
#: トラックが溢れたときにいちばん下まで送ってもボタンが画面の外に残る
ADD_TRACK_BUTTON_SPACE = _ADD_TRACK_BUTTON_GAP + ADD_TRACK_BUTTON_HEIGHT + _ADD_TRACK_BUTTON_GAP

#: ボタンに出す文字 上に乗せたときの説明にも使う
ADD_TRACK_BUTTON_TEXT = "＋ トラック追加"


def track_add_button_rect(layout: TimelineLayout, timeline: Timeline) -> QRect | None:
    """ヘッダの欄の、最後のトラックの下に置く「＋ トラック追加」の矩形

    トラックの並びの続きに置く 欄の上や別のボタンに置くと、トラックを足す操作が
    トラックを並べた所から離れ、何本あっても同じ所を探すことになる
    縦に送って目盛りの下へ隠れたら ``None``（目盛りの上で押せてしまうと、
    再生ヘッドを動かすつもりでトラックが増える）
    """
    bands = layout.bands(timeline)
    bottom = bands[-1].bottom if bands else Metrics.RULER_HEIGHT - layout.scroll_y
    rect = QRect(
        _ADD_TRACK_BUTTON_GAP,
        bottom + _ADD_TRACK_BUTTON_GAP,
        Metrics.TRACK_HEADER_WIDTH - 2 * _ADD_TRACK_BUTTON_GAP,
        ADD_TRACK_BUTTON_HEIGHT,
    )
    if rect.top() < Metrics.RULER_HEIGHT:
        return None
    return rect


def draw_track_add_button(painter: QPainter, rect: QRect, *, hovered: bool = False) -> None:
    """「＋ トラック追加」 M・S・L のボタンと同じ枠線の描き方にそろえる"""
    painter.save()
    painter.fillRect(rect, Colors.TRACK_HEADER)
    painter.setPen(QPen(Colors.SELECTION if hovered else Colors.BORDER, 1))
    painter.drawRect(rect.adjusted(0, 0, -1, -1))
    painter.setPen(QPen(Colors.TEXT if hovered else Colors.TEXT_MUTED, 1))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, ADD_TRACK_BUTTON_TEXT)
    painter.restore()


def draw_clip(
    painter: QPainter,
    clip: Clip,
    band: TrackBand,
    layout: TimelineLayout,
    rate: FrameRate,
    *,
    media: MediaItem | None,
    filmstrip: Filmstrip | None,
    waveform: Waveform | None,
    selected: bool,
    clip_rect: QRect,
    scene_name: str | None = None,
    editing: bool = False,
) -> None:
    """クリップ 1 個を描く

    ``clip_rect`` は画面に見えている部分に切り詰めた矩形 クリップ全体の矩形を
    渡すと、長いクリップで画面外まで描こうとして無駄が出る

    ``editing`` はオブジェクト設定が今出しているクリップか グループやリンクの仲間は
    一緒に選ばれて同じ白い枠が付くが、設定パネルが直すのはそのうちの 1 本だけ
    太い枠と内側の色の線、名前の帯の色で、その 1 本を仲間と見分けられるようにする
    """
    picture, sound = clip_content(band.track, clip, media)
    # 色は絵を描くかで決める レイヤーの BGM やナレーションを映像の色で塗ると、
    # 音だけの物がどれなのかを名前を読むまで見分けられない
    is_video = picture or not sound
    body = Colors.VIDEO_CLIP if is_video else Colors.AUDIO_CLIP
    border = Colors.VIDEO_CLIP_BORDER if is_video else Colors.AUDIO_CLIP_BORDER
    if clip.is_filter or clip.is_group:
        # グループ制御もフィルタと同じく自分の絵を持たず、ほかのクリップへ掛ける物
        body, border = Colors.FILTER_CLIP, Colors.FILTER_CLIP_BORDER

    painter.save()
    painter.setClipRect(clip_rect)
    painter.fillRect(clip_rect, body if clip.enabled else _dimmed(body))

    content = QRect(
        clip_rect.left(),
        clip_rect.top() + Metrics.CLIP_LABEL_HEIGHT,
        clip_rect.width(),
        max(0, clip_rect.height() - Metrics.CLIP_LABEL_HEIGHT),
    )
    if content.height() > 4:
        picture_rect, sound_rect = _split_content(content, picture, sound)
        if picture_rect is not None and filmstrip is not None:
            _draw_filmstrip(painter, picture_rect, clip, layout, rate, filmstrip)
        if sound_rect is not None and waveform is not None:
            # 音を鳴らすトラックの音量も波形に映す 映像トラックの音量は鳴らす所でも使わない
            heard = band.track.kind is not TrackKind.VIDEO
            gain = 10.0 ** (band.track.volume_db / 20.0) if heard else 1.0
            _draw_waveform(painter, sound_rect, clip, layout, rate, waveform, track_gain=gain)

    _draw_clip_label(
        painter,
        clip_rect,
        clip,
        media,
        scene_name,
        editing=editing,
        voice=voice_label(band.track, clip, media),
    )
    if clip.group_id is not None:
        # 束ねたクリップの下端に、グループごとの色の線を引く 同じ色の線どうしが
        # 同じグループ 選ばなくても、どれとどれが一緒に動くのかが分かる
        hue = int(clip.group_id[:6], 16) % 360 if _is_hex(clip.group_id[:6]) else 200
        painter.fillRect(
            QRect(clip_rect.left(), clip_rect.bottom() - 3, clip_rect.width(), 3),
            QColor.fromHsv(hue, 170, 235),
        )

    if editing:
        # 外に選んだ印（白）を太く、内側に色の線を引く 色だけを変えると、白い枠の
        # 仲間と並んだときに線の太さが同じで見落とす
        painter.setPen(QPen(Colors.SELECTION, EDITING_BORDER))
        painter.drawRect(clip_rect.adjusted(1, 1, -2, -2))
        painter.setPen(QPen(Colors.EDITING, 2))
        painter.drawRect(clip_rect.adjusted(4, 4, -5, -5))
    else:
        painter.setPen(QPen(Colors.SELECTION if selected else border, 2 if selected else 1))
        painter.drawRect(clip_rect.adjusted(0, 0, -1, -1))
    painter.restore()


#: 設定パネルが出しているクリップの外枠の太さ（画素） 選んだだけの枠は 2
EDITING_BORDER = 3


def clip_content(track: Track, clip: Clip, media: MediaItem | None) -> tuple[bool, bool]:
    """クリップの中に描く物（サムネイル, 波形）

    映像・音声のトラックは種類で決まる（今までどおり 映像トラックに置いたシーンは音も
    鳴るが、帯には絵だけを描く） レイヤー（混合）は 1 本のクリップが絵も音も持てるので、
    描く・鳴らすかで決める（:func:`~sashimono.core.model.draws_picture` と
    :func:`~sashimono.core.model.plays_sound`） 種類だけで見ると、音付きの動画の音が
    帯に出ず、音量を下げた所も無音の所も見えない
    """
    if track.kind is TrackKind.MIXED:
        return draws_picture(track, clip, media), plays_sound(track, clip, media)
    video = track.kind is TrackKind.VIDEO
    return video, not video


#: 絵と音の両方を描くとき、波形に回す高さの割合と、これより低ければ波形を諦める高さ（画素）
#: 絵を上、波形を下に置く（YMM4 の音付き動画と同じ並び） 波形が細すぎると線にしか見えない
_SOUND_SHARE = 0.4
_MIN_SPLIT_HEIGHT = 20


def _split_content(content: QRect, picture: bool, sound: bool) -> tuple[QRect | None, QRect | None]:
    """中身の矩形を、サムネイルの所と波形の所に分ける 描かない方は ``None``

    両方あるときは 1 本の中を上下に分ける 重ねて描くと、波形がサムネイルに溶けて読めない
    低いトラックでは絵だけにする（どちらかを諦めるなら、何のクリップかが分かる絵を残す）
    """
    if not sound:
        return (content if picture else None), None
    if not picture:
        return None, content
    if content.height() < _MIN_SPLIT_HEIGHT:
        return content, None
    wave = max(1, round(content.height() * _SOUND_SHARE))
    top = QRect(content.left(), content.top(), content.width(), content.height() - wave)
    bottom = QRect(content.left(), top.bottom() + 1, content.width(), wave)
    return top, bottom


#: これより細いクリップは名前もサムネイルも描かない 字が 1 文字も入らない幅
#: 既定の値 設定（:attr:`Preferences.detail_min_width`）で変えられる
DETAIL_MIN_WIDTH = 24


#: これより細いクリップには境目の線も引かない 線だけが縞模様になって読めない
_EDGE_MIN_WIDTH = 3

#: 細い帯に中身の目安（絵の平均の色・音の大きさ）を描く幅の下限（画素） 1 画素の帯まで
#: 1 本ずつ塗ると、全体表示の 1 万本で矩形を作るだけで描く予算を超える（:func:`draw_dense_clips`）
_GLANCE_MIN_WIDTH = 2

#: 細い帯で、選んだ・設定パネルが出している枠の太さ（画素） 隣の帯の境目の線（1 画素）と
#: 見分けられる太さ 2 では、細い帯が並んだ所で境目と同じ縞に見えた（#247）
DENSE_MARK_WIDTH = 3

#: 細い帯の枠を描くときの最小の幅（画素） 1 画素の帯でも枠が潰れて線に見えないようにする
_DENSE_MARK_MIN_SPAN = 6


@dataclass(frozen=True, slots=True)
class ClipGlance:
    """細い帯に描く、クリップの中身の目安 名前やサムネイルが入らない幅でも中身が分かる

    ``tint`` は絵のクリップのサムネイルの平均の色 ``level`` は音のクリップの範囲の
    いちばん大きい音（0..1） どちらも無ければ描かない（解析がまだ・生成オブジェクト）
    """

    tint: QColor | None = None
    level: float | None = None


def clips_in_range(track: Track, start: int, end: int) -> Sequence[Clip]:
    """``start`` から ``end`` までに掛かるクリップ

    クリップは開始順に並び、重ならない（:class:`Track` の約束） 終わりも同じ順に
    並ぶので、両端を二分探索で探せる 全部を舐めると、拡大して 10 本しか
    見えていないときも 1 万本ぶん回ることになる
    """
    clips = track.clips
    first = bisect.bisect_right(clips, start, key=lambda clip: clip.timeline_end)
    last = bisect.bisect_right(clips, end, lo=first, key=lambda clip: clip.timeline_start)
    return clips[first:last]


def draw_dense_clips(
    painter: QPainter,
    band: TrackBand,
    clips: Sequence[Clip],
    layout: TimelineLayout,
    width: int,
    selected: Collection[ClipId],
    sound_only: Callable[[Clip], bool] | None = None,
    editing: ClipId | None = None,
    glance: Callable[[Clip], ClipGlance | None] | None = None,
) -> None:
    """名前も入らない細いクリップを、色の帯としてまとめて塗る

    ``editing`` はオブジェクト設定が今出しているクリップ（:func:`draw_clip` と同じ）
    細い帯でも、選んだ白い枠の上下に色の帯を重ねて仲間と見分けられるようにする

    ``sound_only`` はレイヤー（混合）で音だけのクリップか レイヤーは 1 本の中に絵と音の
    クリップが混ざるので、音だけの物を音声の色で塗る 渡さなければトラックの種類で決める

    ``glance`` はクリップの中身の目安（:class:`ClipGlance`）を返す 絵はサムネイルの平均の
    色で帯を塗り、音はいちばん大きい音の高さの棒を立てる 無地の帯だけだと、引いた表示で
    数秒のクリップを選んだときに中身が無いように見え、読み込みに失敗したのかと迷った（#247）
    名前とサムネイルそのものは描かない 細い所に詰めても読めない

    全体を表示すると数千本が数画素ずつになる 1 本ずつ名前・枠・切り抜きを描くと
    3000 本で 58ms（60fps の予算の 3 倍半）かかった さらに 1 万本では、描く前の
    矩形作りだけで予算を超えた ここは整数の計算だけで済ませ、隙間なく続く
    クリップを 1 本の帯にまとめてから塗る 中身の目安も 2 画素に満たない帯には描かない
    """
    top, height = band.top + 1, band.height - 3
    if height <= 0 or not clips:
        return
    # 色の番号 0 が映像・1 が音声 帯を分ける目印にも使う
    bodies = (Colors.VIDEO_CLIP, Colors.AUDIO_CLIP)
    dims = (_dimmed(bodies[0]), _dimmed(bodies[1]))
    borders = (Colors.VIDEO_CLIP_BORDER, Colors.AUDIO_CLIP_BORDER)
    fixed = 1 if band.track.kind is TrackKind.AUDIO else 0

    header = Metrics.TRACK_HEADER_WIDTH
    scroll, scale = layout.scroll_frame, layout.pixels_per_frame
    # 帯はあとから右へ伸ばすので、組ではなく書き換えられる list で持つ
    # 組にすると、クリップ 1 本ごとに帯を作り直すことになる
    runs: list[list[int]] = []
    edges: list[tuple[int, int]] = []
    glances: list[tuple[int, int, bool, ClipGlance]] = []
    marked: list[tuple[int, int]] = []
    focused: tuple[int, int] | None = None
    for clip in clips:
        left = max(header, int(header + (clip.timeline_start - scroll) * scale))
        right = max(left + 1, min(width, int(header + (clip.timeline_end - scroll) * scale)))
        enabled = 1 if clip.enabled else 0
        colour = fixed if sound_only is None else (1 if sound_only(clip) else 0)
        last = runs[-1] if runs else None
        if last is not None and last[2] == enabled and last[3] == colour and left <= last[1]:
            last[1] = max(last[1], right)
        else:
            runs.append([left, right, enabled, colour])
        if right - left >= _EDGE_MIN_WIDTH:
            edges.append((left, colour))
        if glance is not None and right - left >= _GLANCE_MIN_WIDTH:
            found = glance(clip)
            if found is not None:
                # 隣と同じ目安（同じ素材の続きの所など）は 1 つの塗りにまとめる 境目の線は
                # あとから引くので見た目はほぼ変わらず、塗る回数が減る
                previous = glances[-1] if glances else None
                if (
                    previous is not None
                    and previous[2] == clip.enabled
                    and previous[3] == found
                    and left <= previous[1]
                ):
                    glances[-1] = (previous[0], max(previous[1], right), clip.enabled, found)
                else:
                    glances.append((left, right, clip.enabled, found))
        if clip.id in selected:
            marked.append((left, right))
        if clip.id == editing:
            focused = (left, right)

    # 塗りを全部済ませてから線を引く 交互にすると、あとの帯が前の線を塗りつぶす
    for left, right, enabled, colour in runs:
        fill = bodies[colour] if enabled else dims[colour]
        painter.fillRect(left, top, right - left, height, fill)
    if glances:
        _draw_glances(painter, glances, top, height)
    for left, colour in edges:
        painter.fillRect(left, top, 1, height, borders[colour])
    for left, right in marked:
        _outline(painter, _mark_span(left, right), top, height, Colors.SELECTION)
    if focused is not None:
        # 選んだ枠の上下に色の帯を渡す 枠の内側に細い色の枠を描いていたときは、
        # 数画素の帯では潰れて仲間の白い枠と見分けられなかった
        left, right = _mark_span(*focused)
        bar = min(_DENSE_EDITING_BAR, max(1, (height - 2 * DENSE_MARK_WIDTH) // 3))
        painter.fillRect(left, top + DENSE_MARK_WIDTH, right - left, bar, Colors.EDITING)
        painter.fillRect(
            left, top + height - DENSE_MARK_WIDTH - bar, right - left, bar, Colors.EDITING
        )


#: 設定パネルが出している細い帯の、枠の上下に渡す色の帯の高さ（画素）
_DENSE_EDITING_BAR = 4

#: 細い帯の中身の目安を、帯の頭から空けて描く高さ（画素） 頭に地の色を残して、絵の
#: 平均の色が地の色と似ていても、ほかの帯と同じ種類の物だと分かるようにする
_GLANCE_TOP = 3


def _mark_span(left: int, right: int) -> tuple[int, int]:
    """枠を描く左右 細すぎる帯は真ん中を軸に :data:`_DENSE_MARK_MIN_SPAN` まで広げる"""
    if right - left >= _DENSE_MARK_MIN_SPAN:
        return left, right
    middle = (left + right) // 2
    start = middle - _DENSE_MARK_MIN_SPAN // 2
    return start, start + _DENSE_MARK_MIN_SPAN


def _outline(
    painter: QPainter, span: tuple[int, int], top: int, height: int, colour: QColor
) -> None:
    """帯の内側に :data:`DENSE_MARK_WIDTH` 画素の枠を塗る

    ペンで矩形を描くと線が帯の外へ半分はみ出し、隣の帯の上に乗って、どちらの帯の枠
    なのかが分かりにくい 内側へ塗れば、枠はその帯の中に収まる（:func:`_mark_span` で
    広げた数画素の帯だけは隣へ掛かるが、そうしないと枠が線 1 本に潰れる）
    """
    left, right = span
    thick = DENSE_MARK_WIDTH
    width = right - left
    painter.fillRect(left, top, width, thick, colour)
    painter.fillRect(left, top + height - thick, width, thick, colour)
    painter.fillRect(left, top, min(thick, width), height, colour)
    painter.fillRect(max(left, right - thick), top, min(thick, width), height, colour)


def _draw_glances(
    painter: QPainter,
    glances: Sequence[tuple[int, int, bool, ClipGlance]],
    top: int,
    height: int,
) -> None:
    """細い帯に中身の目安を描く 絵は平均の色で塗り、音は大きさの棒を立てる

    音の棒は無音でも 1 画素の線を残す（波形と同じ） 何も描かないと、解析がまだの
    音と無音の音を見分けられない 棒の高さが 1 画素なら無音だと分かる
    """
    inner_top = top + _GLANCE_TOP
    inner = height - _GLANCE_TOP
    if inner <= 0:
        return
    wave = Colors.WAVEFORM
    wave_dim = _dimmed(wave)
    for left, right, enabled, found in glances:
        span = right - left
        if found.tint is not None:
            painter.fillRect(
                left, inner_top, span, inner, found.tint if enabled else _dimmed(found.tint)
            )
        if found.level is not None:
            level = min(max(found.level, 0.0), 1.0)
            bar = max(1, round(level * (inner - 2)))
            painter.fillRect(
                left + (1 if span > 2 else 0),
                inner_top + (inner - bar) // 2,
                max(1, span - 2),
                bar,
                wave if enabled else wave_dim,
            )


def filmstrip_tint(filmstrip: Filmstrip, clip: Clip, rate: FrameRate) -> QColor | None:
    """クリップの真ん中の時刻のサムネイルの平均の色 細い帯を塗るのに使う

    サムネイルごとの平均は素材ごとに 1 度だけ求めて貯める 描くたびに画素を数えると、
    細い帯が数百本ある全体表示で予算を超える 真ん中を取るのは、頭の 1 枚だと
    フェードインの黒や場面転換の前の絵になりやすいため
    """
    colours = _FILMSTRIP_TINTS.colours(filmstrip)
    if not colours:
        return None
    seconds = clip.picture_time(clip.duration // 2, rate)
    if filmstrip.interval <= 0:
        return colours[0]
    index = int(max(Fraction(0), seconds) / filmstrip.interval)
    return colours[min(index, len(colours) - 1)]


class _FilmstripTints:
    """サムネイルごとの平均の色を、素材のサムネイルの束ごとに貯める

    サムネイルの束（:class:`Filmstrip`）は弱参照を取れないので、束の画素の配列を弱参照で
    持つ 束を強く持つと、素材を外して解析を捨てても、ここがサムネイルを抱えてメモリが空かない
    """

    #: 貯める束の数の上限 素材の数だけあれば足りる 古く使った物から捨てる
    LIMIT = 512

    def __init__(self) -> None:
        self._entries: OrderedDict[int, tuple[weakref.ref[np.ndarray], tuple[QColor, ...]]] = (
            OrderedDict()
        )

    def colours(self, filmstrip: Filmstrip) -> tuple[QColor, ...]:
        sheet = filmstrip.sheet
        key = id(sheet)
        entry = self._entries.get(key)
        # id は解放された物の番号を使い回す 弱参照が同じ物を指すときだけ使う
        if entry is not None and entry[0]() is sheet:
            self._entries.move_to_end(key)
            return entry[1]
        found = _tile_means(filmstrip)
        self._entries[key] = (weakref.ref(sheet), found)
        while len(self._entries) > self.LIMIT:
            self._entries.popitem(last=False)
        return found

    def clear(self) -> None:
        self._entries.clear()


#: 平均を取るときに飛ばす画素の間隔 細い帯の 1 色に全部の画素は要らない 全部を数えると
#: 600 枚の素材で 1 度に数十 ms 掛かり、その描画 1 回が引っかかって見える
_TINT_STRIDE = 4


def _tile_means(filmstrip: Filmstrip) -> tuple[QColor, ...]:
    count, tile = filmstrip.count, filmstrip.tile_width
    if count == 0 or filmstrip.height == 0:
        return ()
    sheet = filmstrip.sheet[::_TINT_STRIDE, : count * tile, :3]
    tiles = sheet.reshape(sheet.shape[0], count, tile, 3)[:, :, ::_TINT_STRIDE, :]
    means = tiles.mean(axis=(0, 2))
    return tuple(QColor(int(r), int(g), int(b)) for r, g, b in means)


_FILMSTRIP_TINTS = _FilmstripTints()


def waveform_level(
    waveform: Waveform, clip: Clip, rate: FrameRate, *, track_gain: float = 1.0
) -> float:
    """クリップの範囲のいちばん大きい音（0..1） 細い帯に立てる棒の高さに使う

    音量・フェード・トラックの音量を波形と同じく掛ける（:func:`shape_envelope`）
    掛けないと、音量を 0 にしたクリップも鳴っているように見え、無音かどうかが分からない
    求めた値はクリップの範囲と効き方ごとに貯める（:class:`_WaveformLevels`）
    """
    end_seconds = clip.source_in + clip.duration * rate.frame_duration * clip.speed
    return _WAVEFORM_LEVELS.get(
        waveform,
        int(clip.source_in * waveform.sample_rate),
        int(end_seconds * waveform.sample_rate),
        _Shaping(clip, rate, 0.0, float(clip.duration), track_gain),
    )


#: 音の大きさを求めるときに束ねる列の数 1 列にすると、フェードの掛け方を真ん中の
#: 1 点で見ることになり、頭と終わりだけ鳴る音が拾えない
_LEVEL_COLUMNS = 16


class _WaveformLevels:
    """求めた音の大きさを、古く使った物から捨てながら貯める

    :class:`_WaveformImages` と同じ作り 解析の結果は弱参照で持つ
    """

    LIMIT = 8192

    def __init__(self) -> None:
        self._entries: OrderedDict[
            tuple[int, int, int, Hashable], tuple[weakref.ref[Waveform], float]
        ] = OrderedDict()

    def get(self, waveform: Waveform, start: int, end: int, shaping: _Shaping) -> float:
        shaped = shaping.key()
        key = (id(waveform), start, end, shaped)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is waveform:
            self._entries.move_to_end(key)
            return entry[1]
        if end <= start:
            return 0.0
        envelope = waveform.envelope(start, end, _LEVEL_COLUMNS)
        low, high = envelope[:, :, 0].min(axis=1), envelope[:, :, 1].max(axis=1)
        if shaped is not None:
            low, high = shape_envelope(
                low,
                high,
                shaping.clip,
                shaping.rate,
                shaping.first,
                shaping.last,
                track_gain=shaping.track_gain,
            )
        level = float(max(np.abs(low).max(), np.abs(high).max()))
        self._entries[key] = (weakref.ref(waveform), level)
        while len(self._entries) > self.LIMIT:
            self._entries.popitem(last=False)
        return level

    def clear(self) -> None:
        self._entries.clear()


_WAVEFORM_LEVELS = _WaveformLevels()


def clip_summary(
    clip: Clip, media: MediaItem | None, rate: FrameRate, scene_name: str | None = None
) -> str:
    """細い帯に載せたときのツールチップ 名前と長さ

    細い帯には名前を描かないので、何のクリップかを確かめる手段がここしかない
    長さは秒と、タイムラインの目盛りと同じタイムコードの両方で出す（短いクリップは
    秒の方が分かりやすく、目盛りと見比べるならタイムコードの方が早い）
    """
    name = f"シーン: {scene_name}" if clip.scene_id is not None else _clip_name(clip, media)
    seconds = float(clip.duration * rate.frame_duration)
    return f"{name}\n長さ {seconds:.2f} 秒（{format_timecode(clip.duration, rate)}）"


def _is_hex(text: str) -> bool:
    return bool(text) and all(character in "0123456789abcdefABCDEF" for character in text)


def _draw_clip_label(
    painter: QPainter,
    rect: QRect,
    clip: Clip,
    media: MediaItem | None,
    scene_name: str | None = None,
    *,
    editing: bool = False,
    voice: str | None = None,
) -> None:
    label_rect = QRect(rect.left(), rect.top(), rect.width(), Metrics.CLIP_LABEL_HEIGHT)
    # 設定パネルが出しているクリップは名前の帯を色で塗る 枠が画面の外に切れていても
    # 名前の見えている所で見分けられる
    shade = QColor(Colors.EDITING) if editing else QColor(Colors.CLIP_LABEL_SHADE)
    if editing:
        shade.setAlpha(170)
    painter.fillRect(label_rect, shade)

    name = f"シーン: {scene_name}" if clip.scene_id is not None else _clip_name(clip, media)
    if voice is not None:
        name = f"{name}  {voice}"
    if clip.speed != 1:
        name = f"{name}  ×{float(clip.speed):g}"
    painter.setPen(QPen(Colors.CLIP_LABEL, 1))
    font = QFont(painter.font())
    font.setPointSizeF(8.5)
    painter.setFont(font)
    painter.drawText(
        label_rect.adjusted(4, 0, -4, 0),
        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        name,
    )


def voice_label(track: Track, clip: Clip, media: MediaItem | None) -> str | None:
    """音声が何本もある素材の音を鳴らすクリップに添える「音声 N」 ほかは ``None``

    音ごとに分けて置くと、どのレイヤーのクリップも同じ素材の名前になり、どれがゲームの
    音でどれがマイクの声なのかを波形の形で見分けるしかなかった（利用者の画面の 4 本）
    番号は素材の音声ストリームの並びで 1 から数える（ffprobe の番号は映像を含むので使わない）
    """
    if media is None or len(media.audio_streams) < 2 or not clip_content(track, clip, media)[1]:
        return None
    stream = heard_stream(track, clip)
    numbers = [s.index for s in media.audio_streams]
    # 素材に無い番号は、デコーダと同じく 1 本目として数える
    number = numbers.index(stream) + 1 if stream in numbers else 1
    return f"音声 {number}"


def _clip_name(clip: Clip, media: MediaItem | None) -> str:
    """クリップに出す名前

    生成オブジェクトは素材を持たないので、素材名だけを見ると全部「素材なし」に
    なってしまう テキストは中身の先頭を添えると、並んだときに見分けが付く
    """
    if media is not None:
        return media.name
    if clip.source is None:
        return "（空）"
    script = custom_object_script(clip)
    if script is not None:
        # 土台は空のテキスト 種類のまま出すと「テキスト」になり、何を置いたのか分からない
        return f"{CUSTOM_OBJECT_LABEL}: {script_label(script.kind)}"

    definition = source_registry.get(clip.source.kind)
    label = definition.label if definition is not None else clip.source.kind
    text = clip.source.params.get("text")
    if isinstance(text, str) and text.strip():
        return f"{label}: {text.splitlines()[0][:16]}"
    return label


def _draw_filmstrip(
    painter: QPainter,
    rect: QRect,
    clip: Clip,
    layout: TimelineLayout,
    rate: FrameRate,
    filmstrip: Filmstrip,
) -> None:
    """クリップの上にサムネイルを敷き詰める

    サムネイルは元の縦横比のまま並べる 引き伸ばすと、何が映っているのか
    判断できなくなって用を成さない
    """
    if filmstrip.count == 0 or rect.height() <= 0:
        return

    scale = rect.height() / filmstrip.height
    tile_width = max(1, int(filmstrip.tile_width * scale))

    x = rect.left()
    while x < rect.right():
        # 秒の計算に float を混ぜないよう、まずフレーム番号（整数）へ落とす
        frame = layout.frame_at(x)
        # 描画と同じ式で引く 絵を止めたクリップで、止めた後の所に動く絵が並ばないように
        tile = filmstrip.at(clip.picture_time(frame - clip.timeline_start, rate))
        if tile is None:
            break
        painter.drawImage(QRectF(x, rect.top(), tile_width, rect.height()), to_qimage(tile))
        x += tile_width


def _draw_waveform(
    painter: QPainter,
    rect: QRect,
    clip: Clip,
    layout: TimelineLayout,
    rate: FrameRate,
    waveform: Waveform,
    *,
    track_gain: float = 1.0,
) -> None:
    """クリップの上に波形を描く 音量やリバーブなどの効き方を大まかに映す（:mod:`shape`）

    1 ピクセル 1 本の縦線を塗った画像を作り、貯めておいて貼る 同じ倍率なら、
    再生ヘッドが動くたびの描き直しでもスクロールでも、束ねる所から作り直さずに済む
    （実素材を 100 本並べた全体表示で、毎回作ると波形だけで 5ms を超えた #205）

    クリップ全体を 1 枚にするのは幅が :data:`WAVEFORM_IMAGE_MAX_COLUMNS` までのとき
    それより広い（長尺素材を大きく拡大した）ときは見えている範囲だけを作る 素材全体を
    画像にすると、長尺素材でメモリと時間を食う
    """
    height = rect.height()
    if rect.width() <= 0 or height <= 2:
        return

    clip_left = layout.frame_to_x(clip.timeline_start)
    # 列の数はクリップが画面で占める画素の数（clip_rect_for と同じく左右を画素へ切り捨てる）
    # 長さ × 倍率を切り上げると、左端に端数があるとき最後の列が矩形の外へ出て、
    # 音の終わりのピークが描かれない（#217 の指摘） 端数で 1 列増減するだけなので、
    # 貯める画像は 1 本のクリップにつき 2 枚まで
    total_columns = max(
        1,
        math.floor(layout.frame_to_x(clip.timeline_end)) - math.floor(clip_left),
    )
    if total_columns <= WAVEFORM_IMAGE_MAX_COLUMNS:
        # クリップの頭から数えた列で作る 見えている左端から数えると、スクロールで
        # 1 画素動くたびに列の区切りが変わり、画像を使い回せないうえ波形が揺れて見える
        end_seconds = clip.source_in + clip.duration * rate.frame_duration * clip.speed
        image = _WAVEFORM_IMAGES.get(
            waveform,
            int(clip.source_in * waveform.sample_rate),
            int(end_seconds * waveform.sample_rate),
            total_columns,
            height,
            _Shaping(clip, rate, 0.0, float(clip.duration), track_gain),
        )
        if image is None:
            return
        # クリップの矩形（clip_rect_for）と同じく画素へ切り捨てた左端に揃える
        offset = rect.left() - math.floor(clip_left)
        width = min(rect.width(), image.width() - max(0, offset))
        if width <= 0:
            return
        painter.drawImage(
            QPoint(rect.left() + max(0, -offset), rect.top()),
            image,
            QRect(max(0, offset), 0, width, height),
        )
        return

    # 見えている左端・右端が、素材のどのサンプルにあたるかを求める
    start_frame = layout.frame_at(rect.left()) - clip.timeline_start
    end_frame = layout.frame_at(rect.right()) - clip.timeline_start
    # ディレイ・リバーブの形は手前の音から作る 見える所の手前も同じ列の幅で取って形を
    # 掛け、見える列だけを貼る 手前を取らないと、見える範囲の手前で鳴った音のやまびこや
    # 尾が、スクロールで左端を越えると消えた（PR #231 の指摘） 手前の列は上限までに抑える
    # （大きく拡大して長いやまびこを掛けると、手前だけで何十万列にもなる）
    span = max(end_frame - start_frame, 1)
    per_frame = rect.width() / span
    reach = min(float(start_frame), shape_history_frames(clip, rate))
    extra = min(math.ceil(reach * per_frame), WAVEFORM_HISTORY_MAX_COLUMNS) if reach > 0 else 0
    first_frame = start_frame - extra / per_frame
    start_seconds = clip.source_in + Fraction(first_frame) * rate.frame_duration * clip.speed
    end_seconds = clip.source_in + end_frame * rate.frame_duration * clip.speed
    image = _WAVEFORM_IMAGES.get(
        waveform,
        int(start_seconds * waveform.sample_rate),
        int(end_seconds * waveform.sample_rate),
        rect.width() + extra,
        height,
        _Shaping(clip, rate, float(first_frame), float(end_frame), track_gain),
    )
    if image is not None:
        painter.drawImage(rect.topLeft(), image, QRect(extra, 0, rect.width(), height))


#: クリップ全体の波形を 1 枚の画像にする幅の上限（画素） 1920 幅の画面で 4 画面分
#: これより広いときは見えている範囲だけを作る
WAVEFORM_IMAGE_MAX_COLUMNS = 8192

#: 見えている範囲だけを作るとき、ディレイ・リバーブの形のために手前に取る列の上限
#: 越えた分の手前の音の響きは映さない（大きく拡大して 100 秒を超えるやまびこを掛けたときだけ）
WAVEFORM_HISTORY_MAX_COLUMNS = 16384

#: 波形の画像を貯めておく量の上限（バイト） 高さ 40 画素で 1920 幅のクリップが 1 枚 300KB ほど
#: 全体表示で見える数（数十枚）と、少し前の倍率の分が入れば足りる
WAVEFORM_CACHE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _Shaping:
    """波形に映す効き方（:func:`shape_envelope`） 列の両端はクリップの頭から数えたフレーム"""

    clip: Clip
    rate: FrameRate
    first: float
    last: float
    track_gain: float

    def key(self) -> Hashable:
        found = shape_key(self.clip, self.track_gain)
        return None if found is None else (found, self.first, self.last)


class _WaveformImages:
    """作った波形の画像を、古く使った物から捨てながら貯める

    素材の解析結果（:class:`Waveform`）は弱参照で持つ 強く持つと、使わなくなった素材の
    解析をメインウィンドウが捨てても、ここが抱えてメモリが空かない
    """

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._used = 0
        self._entries: OrderedDict[
            tuple[int, int, int, int, int, int, Hashable], tuple[weakref.ref[Waveform], QImage]
        ] = OrderedDict()

    def get(
        self,
        waveform: Waveform,
        start: int,
        end: int,
        columns: int,
        height: int,
        shaping: _Shaping | None = None,
    ) -> QImage | None:
        # 色も鍵に入れる 見た目を切り替えたのに前の色の画像が残らないように
        # 効き方も入れる 音量を変えたのに前の大きさの画像が残らないように
        shaped = shaping.key() if shaping is not None else None
        key = (id(waveform), start, end, columns, height, Colors.WAVEFORM.rgba(), shaped)
        entry = self._entries.get(key)
        # id は解放された物の番号を使い回す 弱参照が同じ物を指すときだけ使う
        if entry is not None and entry[0]() is waveform:
            self._entries.move_to_end(key)
            return entry[1]
        if entry is not None:
            self._drop(key)
        if end <= start:
            return None
        envelope = waveform.envelope(start, end, columns)
        # チャンネルをまとめて 1 本の波形にする ステレオを上下に分けるのは
        # トラックを高くしたときの表示として P2 で入れる
        low, high = envelope[:, :, 0].min(axis=1), envelope[:, :, 1].max(axis=1)
        if shaping is not None and shaped is not None:
            low, high = shape_envelope(
                low,
                high,
                shaping.clip,
                shaping.rate,
                shaping.first,
                shaping.last,
                track_gain=shaping.track_gain,
            )
        image = waveform_image(low, high, height)
        # 1 枚で上限を超える画像（高いトラックの幅の広いクリップ）は貯めずに返す
        # 貯めると、ほかを全部捨てても上限を超えたまま残る
        if image.sizeInBytes() > self._budget:
            return image
        self._entries[key] = (weakref.ref(waveform), image)
        self._used += image.sizeInBytes()
        while self._used > self._budget:
            self._drop(next(iter(self._entries)))
        return image

    def clear(self) -> None:
        self._entries.clear()
        self._used = 0

    def _drop(self, key: tuple[int, int, int, int, int, int, Hashable]) -> None:
        _, image = self._entries.pop(key)
        self._used -= image.sizeInBytes()


_WAVEFORM_IMAGES = _WaveformImages(WAVEFORM_CACHE_BYTES)


def clear_waveform_images() -> None:
    """貯めた波形の画像と、細い帯の目安（音の大きさ・絵の平均の色）を捨てる

    貯めていないときの速さを測る道具と試験が使う 目安も一緒に捨てる 残すと、倍率を
    変えた直後の 1 回の重さを測ったつもりで、細い帯の目安を求める分が抜ける
    """
    _WAVEFORM_IMAGES.clear()
    _WAVEFORM_LEVELS.clear()
    _FILMSTRIP_TINTS.clear()


def waveform_image(minimum: np.ndarray, maximum: np.ndarray, height: int) -> QImage:
    """列ごとの最小・最大から、1 列 1 本の縦線を塗った画像を作る

    幅は列の数、線の無い所は透明 真ん中から振幅の分だけ上下へ伸ばし、1 を超える値は
    切り詰める 無音でも 1 画素は残す（何も描かないと音のクリップなのか見分けられない）

    線を 1 本ずつ ``QLineF`` にして ``drawLines`` へ渡していたときは、実素材を 100 本並べた
    全体表示で線を作るだけで 8ms ほど掛かり、60fps の予算（16.7ms）を超えた（#205）
    numpy で画素をまとめて塗れば、Python で列を回す所が無くなる
    """
    columns = int(minimum.shape[0])
    centre = height / 2.0
    half = height / 2.0 - 1.0
    tops = centre - np.clip(maximum, -1.0, 1.0).astype(np.float64) * half
    bottoms = np.maximum(centre - np.clip(minimum, -1.0, 1.0).astype(np.float64) * half, tops + 1.0)
    # 端の座標を含む行から塗る 太さ 1 のペン（アンチエイリアス無し）で線を引いたときと同じ行になる
    rows = np.arange(height, dtype=np.float64)[:, np.newaxis]
    mask = (rows >= np.floor(tops)[np.newaxis, :]) & (rows <= np.floor(bottoms)[np.newaxis, :])
    colour = Colors.WAVEFORM
    # 乗算済みの ARGB で持つ 色が半透明でも、そのまま重ねれば線で描いたときと同じ色になる
    alpha = colour.alpha()
    premultiplied = (
        (alpha << 24)
        | ((colour.red() * alpha // 255) << 16)
        | ((colour.green() * alpha // 255) << 8)
        | (colour.blue() * alpha // 255)
    )
    pixels = np.where(mask, np.uint32(premultiplied), np.uint32(0)).astype(np.uint32)
    image = QImage(
        pixels.tobytes(), columns, height, columns * 4, QImage.Format.Format_ARGB32_Premultiplied
    )
    # QImage は渡したバイト列を参照するだけ 元が先に消えると描く所で落ちるので写しを返す
    return image.copy()


def draw_playhead(painter: QPainter, layout: TimelineLayout, frame: int, height: int) -> None:
    """再生ヘッド 上の三角と縦線"""
    x = layout.frame_to_x(frame)
    if x < Metrics.TRACK_HEADER_WIDTH:
        return

    painter.setPen(QPen(Colors.PLAYHEAD, 1))
    painter.drawLine(QPointF(x, 0), QPointF(x, height))

    painter.setBrush(Colors.PLAYHEAD)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRect(QRectF(x - 4.5, 0, 9, 9))


def clip_rect_for(clip: Clip, band: TrackBand, layout: TimelineLayout, width: int) -> QRect | None:
    """クリップの矩形を、画面に見えている範囲へ切り詰めて返す

    見えていなければ ``None`` 描画対象を絞るのに使う
    """
    left = layout.frame_to_x(clip.timeline_start)
    right = layout.frame_to_x(clip.timeline_end)
    visible_left = max(left, Metrics.TRACK_HEADER_WIDTH)
    visible_right = min(right, width)
    if visible_right <= visible_left or band.height <= 2:
        return None
    return QRect(
        int(visible_left),
        band.top + 1,
        max(1, int(visible_right) - int(visible_left)),
        band.height - 3,
    )


def _dimmed(color: QColor) -> QColor:
    """無効なクリップ用に彩度と明度を落とす"""
    dimmed = QColor(color)
    dimmed.setAlpha(110)
    return dimmed


def visible_clips(
    timeline: Timeline, layout: TimelineLayout, width: int
) -> list[tuple[TrackBand, Clip, QRect]]:
    """見えているクリップと、その矩形の一覧

    描画と当たり判定の両方がこれを使う 別々に計算すると、見えているのに
    掴めないクリップのようなずれが生まれる
    """
    start_frame, end_frame = layout.visible_range(width)
    found: list[tuple[TrackBand, Clip, QRect]] = []
    for band in layout.bands(timeline):
        if band.bottom <= Metrics.RULER_HEIGHT:
            continue
        for clip in clips_in_range(band.track, start_frame, end_frame):
            rect = clip_rect_for(clip, band, layout, width)
            if rect is not None:
                found.append((band, clip, rect))
    return found
