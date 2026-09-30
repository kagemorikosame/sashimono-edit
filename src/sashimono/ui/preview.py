"""プレビュー表示

``QOpenGLWidget`` の中で合成して、そのまま画面へ出す 合成結果を numpy へ読み出して
``QImage`` にしてから描くこともできるが、それだと毎フレーム GPU → CPU → GPU の
往復が入る 1080p で 8MB の転送が 30 回/秒、再生には致命的
"""

from __future__ import annotations

import time
from collections.abc import Collection
from dataclasses import dataclass, field, replace

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QOpenGLContext, QPainter, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from sashimono.core.commands import Command
from sashimono.core.model import (
    Clip,
    ClipId,
    EffectId,
    MediaId,
    Project,
    Track,
    controlling_groups,
)
from sashimono.engine.cache.proxy import ProxyStore
from sashimono.engine.gpu import CurrentGLContext, fit_placement
from sashimono.engine.render import (
    DEFAULT_DECODE_THREADS,
    FULL_QUALITY,
    FrameRenderer,
    Invalidation,
    PreviewCache,
    RenderQuality,
    changed_spans,
    image_spans,
)
from sashimono.engine.render.background import BackgroundPrefetch
from sashimono.engine.render.groups import grouped
from sashimono.engine.render.outline import (
    Outline,
    Point,
    canvas_scale,
    clip_outline,
    has_outline,
    is_generated,
)
from sashimono.engine.render.region_outline import RegionFrame, region_effects, region_frame
from sashimono.ui.preview_handles import (
    ALIGNMENTS,
    KEYFRAME_DRAG_AT_PLAYHEAD,
    Grip,
    Guide,
    Hit,
    aligned_values,
    bounding_box,
    hit_test,
    moved_values,
    pick_clip,
    rotated_values,
    rotation_knob,
    scaled_values,
    snap_offset,
    snap_targets,
    start_values,
    transform_commands,
    turn_between,
    value_commands,
)
from sashimono.ui.region_handles import (
    edge_points,
    region_changes,
    region_hit_test,
    region_turn,
)

__all__ = ["GRIP_SIZE", "KNOB_LENGTH", "ROTATE_REACH", "SLOW_FRAME_MS", "PreviewWidget"]

#: 角の掴み所の大きさ（画面の画素） 掴める半径もこれ 合成の画素で数えると、
#: プレビューを小さくしたときに掴めなくなる
GRIP_SIZE = 8.0
#: 角の外側で回せる範囲（画面の画素） 広すぎると、隣の絵を掴むつもりで回してしまう
ROTATE_REACH = 24.0
#: 回転の掴み所を上の辺からどれだけ離すか（画面の画素）
KNOB_LENGTH = 22.0

#: 1 回のドラッグを取り消しの一覧に出すときの名前
_LABELS = {
    Grip.MOVE: "プレビューで位置を変更",
    Grip.SCALE: "プレビューで拡大率を変更",
    Grip.ROTATE: "プレビューで回転を変更",
}


@dataclass(slots=True)
class _Drag:
    """掴んでいる途中の状態 命令は掴んだ時点のプロジェクトから毎回作り直す

    途中の絵は窓がプレビューへ渡すだけで、画面のプロジェクトはその途中の物に替わる
    そこから作ると、動かした分をもう 1 度足していく
    """

    hit: Hit
    clip_id: ClipId
    project: Project
    press: Point
    last: Point
    start: dict[str, float]
    corners: tuple[Point, Point, Point, Point]
    pivot: Point
    #: 掴んだ時点のフレーム 途中でコマ送りしても、点を打つ時刻を元の値の時刻とそろえる
    frame: int
    #: 回した角度の合計（度） 1 回ごとの差を足す 押した所との差では半周で跳ぶ
    turned: float = 0.0
    commands: list[Command] = field(default_factory=list)
    #: 部分フィルタの範囲を掴んでいれば、そのエフェクトと掴んだ時点の枠
    region: tuple[EffectId, RegionFrame] | None = None
    #: 動かすときに吸い付く先（横, 縦） 掴んだときに 1 度だけ集める 動かすたびに
    #: ほかの物の枠を作り直すと、テキストの多い画面で指に遅れる
    snap: tuple[list[float], list[float]] | None = None


#: 範囲の掴み所の色 クリップの枠（水色）と見分ける 再生ヘッド（赤）とも離す
_REGION_COLOR = QColor(255, 170, 60)

#: 範囲を掴んだドラッグを取り消しの一覧に出すときの名前
_REGION_LABELS = {
    Grip.MOVE: "プレビューで範囲を移動",
    Grip.SCALE: "プレビューで範囲の大きさを変更",
    Grip.EDGE: "プレビューで範囲の大きさを変更",
    Grip.ROTATE: "プレビューで範囲を回転",
}


#: 先読みの 1 コマにこれ以上掛かるなら、先読みそのものをやめる（ミリ秒）
#: **画面のスレッドで貯めるときだけ** 別のスレッドで貯めるときは画面が止まらないので、
#: 重い所ほど貯める値打ちがある
#:
#: 描いているのは編集画面と同じ GL コンテキストなので、1 コマ描く間は
#: 操作を受け付けられない 重すぎる素材（動画を 20 本重ねて効果を積むと
#: 1 コマ 2 秒 測った値は sashimono.engine.cache.proxy）では、貯まる値打ちより
#: 固まる方が大きい 0.2 秒は、押してから反応するまでに引っかかりを感じ始める辺り
SLOW_FRAME_MS = 200.0

#: エフェクトが読む画像（画像合成の絵、縁取りの模様）が書き換わったかを見る間隔（ミリ秒）
#:
#: 画像はパスで指すだけなので、別のソフトで描き直してもプロジェクトは変わらず、
#: 先読みした絵が古いまま残る 見るのは更新時刻・大きさ・ファイルの番号だけで
#: 中身は読まない（1 枚あたり stat 1 回、数十マイクロ秒）ので、いま使っている
#: 画像を 1 秒おきに見ても手間にならない 描き直してから画面へ
#: 戻ってくるまでの間には気付ける
IMAGE_WATCH_MS = 1000


class PreviewWidget(QOpenGLWidget):
    """タイムラインの指定フレームを映す"""

    #: GL の準備ができた レンダラを使い始めてよい合図
    ready = Signal()
    #: 先読みをやめた 引数は理由 画面へ出して、黙って効かない状態を避ける
    prefetch_stopped = Signal(str)
    #: プレビューを押してクリップを選んだ 引数はクリップの ID タイムラインの選択を合わせる
    clip_picked = Signal(str)
    #: 掴んで動かし終えた 命令の一覧と取り消しの名前 1 段にまとめて流してもらう
    commands_requested = Signal(list, str)
    #: 掴んでいる途中 命令の一覧を履歴に積まずに見せてもらう 空なら元へ戻す
    preview_requested = Signal(list)

    def __init__(
        self,
        project: Project,
        parent: object = None,
        *,
        proxies: ProxyStore | None = None,
        prefetch_bytes: int = 0,
        decode_threads: int = DEFAULT_DECODE_THREADS,
        prefetch_thread: bool = True,
    ) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self._project = project
        self._frame = 0
        self._quality = FULL_QUALITY
        self._renderer: FrameRenderer | None = None
        #: プレビュー用の控えの置き場 **書き出しには渡さない**
        self._proxies = proxies
        #: 先読みに使えるバイト数 0 なら先読みしない
        self._prefetch_bytes = max(0, prefetch_bytes)
        #: レイヤーごとの並列デコードのスレッド数 GL を作る前に決まっていることがある
        self._decode_threads = decode_threads
        self._cache: PreviewCache | None = None
        #: まだ置き場へ渡していないメモリの量 GL を確実に使える所で渡す
        self._pending_budget: int | None = None
        #: 手が空いたら 1 コマずつ描く 間隔 0 は「ほかにすることが無くなったら」
        #: という意味で、入力の処理より後になる 待ち時間を入れると、貯まるまでが
        #: 枚数 × その待ち時間ぶん延びる
        self._idle = QTimer(self)
        self._idle.setInterval(0)
        self._idle.timeout.connect(self._prefetch_step)
        #: 再生中は先読みを止める 出す側と同じ GPU を奪い合って、
        #: いま出すべきコマが遅れる
        self._playing = False
        self._image_watch = QTimer(self)
        self._image_watch.setInterval(IMAGE_WATCH_MS)
        self._image_watch.timeout.connect(self.check_images)
        #: 先読みを別のスレッドで描くか（設定） 切ると画面のスレッドで 1 コマずつ描く
        self._prefetch_thread = prefetch_thread
        #: 別のスレッドの先読み 作れなかった・止まったときは None のまま、
        #: 画面のスレッドの先読み（``_idle`` と ``_cache``）へ戻る
        self._background: BackgroundPrefetch | None = None
        #: 閉じた走り係が捨てていた控えの素材 次の :meth:`take_discarded` で渡す
        self._closed_discarded: set[MediaId] = set()
        #: 走り係を作るときに画面の側の絵を手放せなかった（current にできなかった）
        #: 次にコンテキストが current な所（:meth:`_apply_pending`）で手放す
        self._release_cache_pending = False
        #: 別のスレッドで先読みできなかった 何度も作り直して失敗し続けないよう覚える
        #: 設定を入れ直したときだけ忘れる
        self._background_broken = False
        #: 再生を始めたコマを描き終えたら、貯めた所の端を裏で読ませる（:meth:`_prime_edge`）
        self._prime_after_paint = False
        #: 外枠を出すクリップ（タイムラインで選んだ主の 1 本）
        self._selection: ClipId | None = None
        #: 外枠を出して直接動かすか（設定）
        self._handles_enabled = True
        #: キーフレームのある値を動かしたときの決まり（設定）
        self._keyframe_drag = KEYFRAME_DRAG_AT_PLAYHEAD
        self._drag: _Drag | None = None
        #: 範囲の枠を出す部分フィルタ（設定パネルで最後に触った物） 選んだクリップに無ければ、
        #: そのクリップの最初の範囲を出す
        self._region_effect: EffectId | None = None
        #: 掴んだ途中の絵を頼んでいる最中 その頼みで届く中身は掴むのをやめる理由にならない
        self._showing_drag = False
        #: 動かすときに画面の中央やほかの物の端へ吸い付くか（設定 タイムラインの磁石とは別）
        self._snap = True
        #: 吸い付く距離（画面の画素）
        self._snap_distance = 8
        #: 吸い付いた所に引く線 動かしている間だけ出す
        self._guides: list[Guide] = []
        # 押していない間も矢印の形を変えるため 掴める所が見えるように
        self.setMouseTracking(True)
        # 押してもフォーカスを取らない 取ると、プレビューで選んだ後のコマ送りや削除の
        # キーがタイムラインへ届かなくなる Esc は掴んでいる間だけキーボードを借りて受ける
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumSize(240, 135)

    @property
    def renderer(self) -> FrameRenderer | None:
        return self._renderer

    def set_project(self, project: Project) -> None:
        if self._drag is not None and not self._showing_drag:
            # 掴んでいる途中に別の道（取り消しやほかのパネル）で中身が変わった 掴んだ時点の
            # プロジェクトから作った値を離したときに当てると、その変更を上書きしてしまう
            # 途中の絵はこの新しい中身で描き直されるので、元へ戻す頼みは要らない
            self._end_drag()
        previous = self._project
        self._project = project
        if self._renderer is not None:
            changed = changed_spans(previous, project)
            # レンダラは GL を触るので、コンテキストを current にしてから渡す
            self.makeCurrent()
            self._renderer.set_project(project)
            self._invalidate(changed)
            self.doneCurrent()
            if self._background is not None:
                # 捨てる範囲とプロジェクトを 1 つの頼みで渡す 別々に渡すと、
                # 走り係が新しいプロジェクトで描く前に古い絵を出せる間ができる
                self._background.set_project(project, changed)
        self.update()
        self._restart_prefetch()

    def set_proxies(self, proxies: ProxyStore | None) -> None:
        """控えの置き場を差し替える

        開いているデコーダは元のファイルを掴んだままなので、開き直させる
        （設定で切ったのに控えのままだと、切った意味が無い）
        """
        if proxies is self._proxies:
            return
        self._proxies = proxies
        if self._renderer is not None:
            self.makeCurrent()
            self._renderer.set_proxies(proxies)
            # 読む元が変われば絵も変わる 取ってある絵は全部使えない
            self._invalidate(Invalidation.all())
            self.doneCurrent()
            if self._background is not None:
                self._background.set_proxies(proxies)
        self.update()
        self._restart_prefetch()

    def take_discarded(self) -> set[MediaId]:
        """レンダラが捨てた控えの素材 呼ぶ側が作り直しを頼む

        先読みの走り係のレンダラが捨てた分も合わせる 走り係は画面の側より先の
        コマを読むので、壊れた控えに先に当たるのはたいてい走り係の方
        """
        found = self._renderer.take_discarded() if self._renderer is not None else set()
        found |= self._closed_discarded
        self._closed_discarded = set()
        if self._background is not None:
            found |= self._background.take_discarded()
        return found

    def reload_sources(self, media_ids: Collection[MediaId] | None = None) -> None:
        """素材を開き直させる 控えができた直後に呼ぶ

        描き直すだけでは切り替わらない 先にプレビューした素材は、
        レンダラが元のファイルを掴んだままになっている

        ``media_ids`` を渡すと、その素材のぶんだけ開き直す
        """
        if self._renderer is None:
            return
        self.makeCurrent()
        self._renderer.reopen_sources(media_ids)
        # 控えができた・壊れていて元へ戻した どちらも開き直した素材の絵が変わる
        self._invalidate(Invalidation.all())
        self.doneCurrent()
        if self._background is not None:
            self._background.reopen_sources(media_ids)
        self.update()
        self._restart_prefetch()

    def set_frame(self, frame: int) -> None:
        frame = max(0, frame)
        if frame == self._frame:
            return
        self._frame = frame
        self.update()
        self._restart_prefetch()

    def set_quality(self, quality: RenderQuality) -> None:
        self._quality = quality
        if self._renderer is not None:
            self.makeCurrent()
            self._renderer.set_quality(quality)
            self.doneCurrent()
        if self._background is not None:
            self._background.set_quality(quality)
        self.update()
        self._restart_prefetch()

    def refresh_all(self) -> None:
        """取ってある絵を全部捨てて描き直す 絵の出方を変える設定を切り替えたとき

        中身（プロジェクト）は同じなので、編集の差分からは捨てる範囲が出てこない
        """
        if self._renderer is not None:
            self.makeCurrent()
            self._invalidate(Invalidation.all())
            self.doneCurrent()
        if self._background is not None:
            self._background.invalidate(Invalidation.all())
        self.update()
        self._restart_prefetch()

    def check_images(self) -> None:
        """エフェクトが読む画像が書き換わっていたら、それを使う所を描き直す

        **コンテキストを current にしない** タイマーから呼ばれるので、窓が隠れて
        いると current にならないまま戻り、そこで doneCurrent を呼ぶとほかが使って
        いるコンテキストを外してしまう ここで捨てる絵は描画先を空きへ回すだけで
        （:meth:`FrameCache.invalidate` は GL を呼ばない）、current でなくてよい
        current にできなかったからと捨てずに戻ると、書き換わりは 1 度しか
        伝わらないので、古い絵が残ったままになる
        """
        if self._renderer is None or self._cache is None:
            return
        changed = self._renderer.stale_images()
        if not changed:
            return
        spans = image_spans(self._project, changed)
        self._cache.invalidate(spans)
        if self._background is not None:
            # 走り係への頼みも GL を触らない 一覧から外すのはこちらのスレッドで済む
            self._background.invalidate(spans)
        self.update()
        self._restart_prefetch()

    def set_prefetch_bytes(self, prefetch_bytes: int) -> None:
        """先読みに使えるメモリを変える 0 で止める

        その場では渡さない 減らすと置き場は描画先を手放すので GL を触るが、
        設定の窓から戻ってきた所が GL を使える状態とは限らない
        描く直前（:meth:`paintGL` と先読みの 1 コマ）まで持ち越す
        """
        prefetch_bytes = max(0, prefetch_bytes)
        if prefetch_bytes == self._prefetch_bytes:
            return
        self._prefetch_bytes = prefetch_bytes
        self._pending_budget = prefetch_bytes
        if prefetch_bytes == 0:
            # 切ったら走り係ごと止める 予算 0 を渡すだけでは、スレッド・レンダラ・
            # デコーダ（素材のファイルを掴んだまま）・共有したコンテキスト・効果の
            # GPU の資源が窓を閉じるまで残る 入れ直せば次の空き時間に作り直す
            self._close_background()
        elif self._background is not None:
            # 走り係は自分のスレッドで描画先を手放すので、その場で渡してよい
            self._background.set_budget(prefetch_bytes, self._frame)
        self.update()
        self._restart_prefetch()

    def set_decode_threads(self, threads: int) -> None:
        """レイヤーごとの並列デコードのスレッド数を変える 1 で並べない

        GL を触らないので、その場でレンダラへ渡してよい まだ GL ができていない
        （``initializeGL`` の前）なら、覚えておいて作るときに渡す
        """
        if threads == self._decode_threads:
            return
        self._decode_threads = threads
        if self._renderer is not None:
            self._renderer.set_decode_threads(threads)
        if self._background is not None:
            self._background.set_decode_threads(threads)

    def set_prefetch_thread(self, enabled: bool) -> None:
        """先読みを別のスレッドで描くかを切り替える

        切ったら走り係を止めて、画面のスレッドで貯め直す 入れ直したときは、
        前に作れなかったことも忘れる（ドライバを入れ替えた後などに試せるように）
        """
        if enabled == self._prefetch_thread:
            return
        self._prefetch_thread = enabled
        self._background_broken = False
        if not enabled:
            self._close_background()
        self._restart_prefetch()

    @property
    def prefetch_in_background(self) -> bool:
        """いま別のスレッドで先読みしているか"""
        return self._background is not None

    def set_playing(self, playing: bool) -> None:
        """再生中かどうか 再生中は先読みを止める"""
        if playing:
            # 再生中は枠を出さない 見えない枠を掴んだまま離すと、見ていない値が確定する
            self._cancel_drag()
        self._playing = playing
        self._prime_after_paint = False
        if playing:
            if self._background is not None and self._frame not in self._background.cached:
                # いまのコマは画面の側でこれから描く 先に端を頼むと、描くときに
                # 同じデコーダの頼みを待ったうえで、頭へ戻って読み直す
                # 描き終えてから頼む
                self._prime_after_paint = True
            else:
                self._prime_edge()
        self._restart_prefetch()

    def _prime_edge(self) -> None:
        """再生を始めたら、貯めた所の端のコマのデコードを裏で走らせておく

        別のスレッドで貯めると、画面の側のデコーダは貯めた所を再生する間は動かない
        端を越えた 1 コマ目で鍵フレームから読み直し、再生がそこで 1 度引っかかる
        （GOP 250 の 1080p を 3 枚重ねて 129ms 画面のスレッドで貯めたときは 14ms、
        ここで頼んでおくと 12ms）
        画面のスレッドで貯めるときは、貯めたデコーダがそのまま端に居るので要らない
        """
        if self._background is None or self._renderer is None:
            return
        cached = self._background.cached
        # いまのコマは貯まっていないことがある 画面の側で描いたコマは、走り係に
        # 描かせない（次のコマから貯めさせる）ので、その次から数える
        start = self._frame if self._frame in cached else self._frame + 1
        edge = start
        while edge in cached:
            edge += 1
        if edge == start:
            # 先に貯まった所が無い 再生すれば描く道がそのまま順に読む
            return
        if edge < self._project.duration:
            self._renderer.prime(edge)

    @property
    def cached_frames(self) -> frozenset[int]:
        """取ってある絵のフレーム番号 どこまで貯まったかを画面に出すため"""
        if self._background is not None:
            return self._background.cached
        return self._cache.cache.cached if self._cache is not None else frozenset()

    def shutdown(self) -> None:
        """GL 資源を解放する ウィンドウを閉じる前に呼ぶこと

        ``QOpenGLWidget`` が破棄された後ではコンテキストが無く、テクスチャの
        解放ができない
        """
        self._idle.stop()
        self._image_watch.stop()
        # 走り係を先に止める 走り係のコンテキストは画面のコンテキストと共有しているので、
        # 画面の側を先に捨てると、走り係が使っている共有の資源まで消えることがある
        self._close_background()
        if self._renderer is None:
            return
        self.makeCurrent()
        if self._cache is not None:
            self._cache.release()
            self._cache = None
        self._renderer.close()
        self._renderer = None
        self.doneCurrent()

    def initializeGL(self) -> None:  # noqa: N802 - Qt の命名規約
        # 窓を付け替えるとコンテキストが作り直され、ここへもう 1 度来る 前の走り係は
        # 前のコンテキストと共有していて、新しいコンテキストからは絵が見えない
        # 止めておけば、次の空き時間に新しいコンテキストと共有して作り直す
        self._close_background()
        # ここでは Qt がすでにコンテキストを current にしている 自前の
        # オフスクリーンコンテキストを使うと描画先を見失う
        self._renderer = FrameRenderer(
            self._project,
            context=CurrentGLContext(),
            quality=self._quality,
            proxies=self._proxies,
            decode_threads=self._decode_threads,
        )
        self._cache = PreviewCache(self._renderer, budget_bytes=self._prefetch_bytes)
        self._image_watch.start()
        self.ready.emit()
        self._restart_prefetch()

    def paintGL(self) -> None:  # noqa: N802 - Qt の命名規約
        if self._renderer is None:
            return
        ratio = self.devicePixelRatioF()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))

        # Qt はここでコンテキストを current にしている 持ち越した設定を当てる
        self._apply_pending()
        self._paint_picture(width, height)
        # 3 つの道（走り係の絵・貯めた絵・その場で描いた絵）のどれを通っても枠を重ねる
        # 道ごとに描くと、先読みが当たったコマだけ枠が消える
        self._paint_overlay()
        self._paint_region()

    def _paint_picture(self, width: int, height: int) -> None:
        """絵を出す **コンテキストが current な所（paintGL）で呼ぶこと**"""
        assert self._renderer is not None
        target = self.defaultFramebufferObject()
        if self._background is not None:
            # 走り係の絵を出すだけ 外れたらこちらで描くが、取ってはおかない
            # 取っておくと画面の側にも描画先を持つことになり、同じメモリで
            # 貯まる枚数が減る 再生ヘッドの所は走り係が真っ先に描く
            if self._background.show(
                self._frame, self._renderer.compositor, target, (0, 0, width, height)
            ):
                return
            # このコマはこちらで描く 走り係には次のコマから貯めさせる 同じコマを
            # 走り係も描くと、GIL と GPU を取り合って両方が遅くなり、再生ヘッドを
            # 送り続ける間は走り係がいつまでも追い付けない
            self._background.set_playhead(self._frame + 1)
            self._renderer.compose(self._frame)
            self._renderer.compositor.present(target, (0, 0, width, height))
            if self._prime_after_paint:
                self._prime_after_paint = False
                self._prime_edge()
            return
        if self._cache is not None and self._cache.enabled:
            self._cache.draw(self._frame, target, (0, 0, width, height))
            return
        self._renderer.compose(self._frame)
        self._renderer.compositor.present(target, (0, 0, width, height))

    # --- 外枠と直接の操作 ---

    def set_selection(self, clip_id: ClipId | None) -> None:
        """外枠を出すクリップ タイムラインで選んだ主の 1 本"""
        if clip_id == self._selection:
            return
        self._selection = clip_id
        self._cancel_drag()
        self.update()

    @property
    def selection(self) -> ClipId | None:
        return self._selection

    def set_region_effect(self, effect_id: EffectId | None) -> None:
        """範囲の枠を出すエフェクト（設定パネルで触った物） 範囲を持たない物なら今のまま

        範囲を持たないエフェクトを触っただけで枠を消すと、ぼかしの強さを直しながら範囲を
        見比べる、ができない
        """
        located = (
            self._project.timeline.locate_clip(self._selection)
            if self._selection is not None
            else None
        )
        if located is None or effect_id is None:
            return
        if all(effect.id != effect_id for effect in region_effects(located[1])):
            return
        if effect_id != self._region_effect:
            self._region_effect = effect_id
            self.update()

    @property
    def region_effect(self) -> EffectId | None:
        """いま範囲の枠を出しているエフェクト 出していなければ ``None``"""
        found = self._region()
        return found[2] if found is not None else None

    def set_handles_enabled(self, enabled: bool) -> None:
        """外枠を出して直接動かすか（設定） 切ったら枠も掴む所も出さない"""
        if enabled == self._handles_enabled:
            return
        self._handles_enabled = enabled
        self._cancel_drag()
        self.unsetCursor()
        self.update()

    @property
    def handles_enabled(self) -> bool:
        return self._handles_enabled

    def set_snap(self, enabled: bool, distance: int) -> None:
        """位置を動かすときの磁石（設定） ``distance`` は画面の画素"""
        self._snap = enabled
        self._snap_distance = max(1, distance)

    def align_selected(self, anchor: str) -> bool:
        """選んだクリップを、見えている範囲ごと画面の ``anchor`` の所へ寄せる（配置のテンプレート）

        枠を出せない物（今のコマに無い・絵を描かない）は寄せずに偽を返す 大きさは描く側と
        同じ枠（:meth:`outline_of`）から取る 回した絵は見えている範囲で揃える
        """
        if self._selection is None:
            return False
        located = self._project.timeline.locate_clip(self._selection)
        if located is None or located[0].locked:
            return False
        track, clip = located
        if not self._project.draws_picture(track, clip):
            return False
        outline = self.outline_of(clip)
        if outline is None:
            return False
        changes = aligned_values(
            start_values(clip, self._frame - clip.timeline_start),
            bounding_box(outline.corners),
            self.canvas_size(),
            anchor,
            scale=self.canvas_scale(),
        )
        commands = transform_commands(
            self._project, clip.id, changes, self._frame, keyframes=self._keyframe_drag
        )
        if commands:
            label = next(a[1] for a in ALIGNMENTS if a[0] == anchor)
            self.commands_requested.emit(commands, f"配置を{label}へ")
        return True

    def _snap_targets(self, moving: ClipId) -> tuple[list[float], list[float]]:
        """掴んだ物のほかに今のコマで描いている物の枠と、画面の端と中央"""
        boxes = []
        for track in self._project.timeline.active_picture_tracks():
            clip = track.clip_at(self._frame)
            if clip is None or clip.id == moving or not clip.enabled or not has_outline(clip):
                continue
            if not self._project.draws_picture(track, clip):
                continue
            outline = self.outline_of(clip)
            if outline is not None:
                boxes.append(bounding_box(outline.corners))
        return snap_targets(self.canvas_size(), boxes)

    def set_keyframe_drag(self, mode: str) -> None:
        """キーフレームのある値を動かしたときの決まり（設定）"""
        self._keyframe_drag = mode

    def canvas_size(self) -> tuple[int, int]:
        """合成の大きさ 画質を落としていれば小さい 枠はこの画素で数える"""
        return self._quality.apply(*self._project.settings.resolution)

    def canvas_scale(self) -> float:
        """合成の画素 1 つが画面の画素いくつ分かの逆数 描く側と同じく横の比で数える

        ドラッグで動かした合成の画素を設定の X・Y（画面の画素）へ直すのに使う
        """
        return canvas_scale(self._project.settings.resolution, self.canvas_size())[0]

    def canvas_rect(self) -> tuple[float, float, float, float]:
        """絵が出ている所（ウィジェットの座標 左・上・幅・高さ）

        :meth:`Compositor.present` の置き方（縦横比を保って収め、画素へ切り捨てる）と
        同じに数える GL の原点は左下なので、上からの位置へ直す
        """
        ratio = self.devicePixelRatioF()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))
        canvas_width, canvas_height = self.canvas_size()
        placed = fit_placement(canvas_width, canvas_height, width, height)
        shown_width = max(1, int(placed.width))
        shown_height = max(1, int(placed.height))
        top = height - int(placed.top) - shown_height
        return (
            int(placed.left) / ratio,
            top / ratio,
            shown_width / ratio,
            shown_height / ratio,
        )

    def to_canvas(self, point: QPointF) -> Point:
        """ウィジェットの座標 → 合成の画素（Y は下が正）"""
        left, top, width, height = self.canvas_rect()
        canvas_width, canvas_height = self.canvas_size()
        return (
            (point.x() - left) * canvas_width / width,
            (point.y() - top) * canvas_height / height,
        )

    def to_widget(self, point: Point) -> QPointF:
        """合成の画素 → ウィジェットの座標"""
        left, top, width, height = self.canvas_rect()
        canvas_width, canvas_height = self.canvas_size()
        return QPointF(
            left + point[0] * width / canvas_width, top + point[1] * height / canvas_height
        )

    def outline_of(self, clip: Clip) -> Outline | None:
        """いまのコマの ``clip`` の外枠（合成の画素）

        生成オブジェクトの入れ物は画面の側のレンダラに作らせる（GL は使わない）
        別のスレッドの先読みが出したコマでは、画面の側はまだ何も作っていない
        """
        if not clip.timeline_start <= self._frame < clip.timeline_end or not has_outline(clip):
            # 枠を出さない物（フィルタ・シーン・グループ制御）の絵を作りに行かない
            return None
        extent = self._extent_of(clip)
        if is_generated(clip) and extent is None:
            return None
        shown = self._grouped(clip)
        outline = clip_outline(
            self._project, shown, self._frame, canvas=self.canvas_size(), extent=extent
        )
        if outline is not None and shown is not clip:
            # グループ制御で動かした物の枠は、描いた所に出す ただ掴んで動かすのはクリップ自身の
            # 配置なので、グループで拡大・回転していると指と絵の動きが合わない 点線にして知らせる
            outline = replace(outline, approximate=True)
        return outline

    def _grouped(self, clip: Clip) -> Clip:
        """グループ制御を当てた、描くときと同じクリップ 受け持つグループが無ければそのまま"""
        located = self._project.timeline.locate_clip(clip.id)
        if located is None:
            return clip
        timeline = self._project.timeline
        drawn = {track.id for track in timeline.active_picture_tracks()}
        groups = [
            (track, group)
            for track, group in controlling_groups(
                timeline.picture_tracks(), located[0].id, self._frame
            )
            if track.id in drawn
        ]
        return grouped(clip, groups, self._frame)

    def _extent_of(
        self, clip: Clip
    ) -> tuple[tuple[float, float, float, float], tuple[int, int]] | None:
        """生成オブジェクトの入れ物（外枠と範囲の枠を出すため） フィルタは画面そのもの"""
        if not is_generated(clip) or clip.is_filter or self._renderer is None:
            return None
        return self._renderer.object_extent(clip, self._frame)

    def _region(self) -> tuple[Track, Clip, EffectId, RegionFrame] | None:
        """範囲の枠を出す相手 選んだクリップに範囲のエフェクトが無ければ ``None``

        クリップの外枠（:meth:`_selected`）と違い、フィルタのクリップでも出す 画面全体に
        掛けるフィルタで、隠したい所だけを範囲で選ぶのが一番よく使う形
        """
        if self._selection is None or not self._handles_enabled or self._playing:
            return None
        located = self._project.timeline.locate_clip(self._selection)
        if located is None:
            return None
        track, clip = located
        if not clip.enabled or not clip.timeline_start <= self._frame < clip.timeline_end:
            return None
        if not self._project.draws_picture(track, clip):
            return None
        if all(t.id != track.id for t in self._project.timeline.active_picture_tracks()):
            return None
        effects = region_effects(clip)
        if not effects:
            return None
        chosen = next((e for e in effects if e.id == self._region_effect), effects[0])
        frame = region_frame(
            self._project,
            self._grouped(clip),
            chosen.id,
            self._frame,
            canvas=self.canvas_size(),
            extent=self._extent_of(clip),
        )
        return None if frame is None else (track, clip, chosen.id, frame)

    def _selected(self) -> tuple[Track, Clip, Outline] | None:
        """外枠を出す相手 いまのコマに絵を描いていなければ ``None``"""
        if self._selection is None or not self._handles_enabled or self._playing:
            return None
        located = self._project.timeline.locate_clip(self._selection)
        if located is None:
            return None
        track, clip = located
        if not clip.enabled or not self._project.draws_picture(track, clip):
            return None
        # ミュートやほかのトラックのソロで映っていないクリップは掴ませない 描かれていない
        # 物の枠が残ると、見えない絵を動かすことになる
        if all(t.id != track.id for t in self._project.timeline.active_picture_tracks()):
            return None
        outline = self.outline_of(clip)
        return None if outline is None else (track, clip, outline)

    def _widget_corners(self, outline: Outline) -> list[Point]:
        corners = []
        for corner in outline.corners:
            shown = self.to_widget(corner)
            corners.append((shown.x(), shown.y()))
        return corners

    def _paint_overlay(self) -> None:
        """選んだクリップの外枠と掴む所を重ねる

        **再生中は描かない** テキストの入れ物を画面のスレッドで作ることがあり、再生の
        1 コマごとに文字を組み直すと再生が遅れる 止めたコマで描けば足りる
        """
        found = self._selected()
        if found is None:
            return
        track, _, outline = found
        corners = [QPointF(x, y) for x, y in self._widget_corners(outline)]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color = QColor(160, 160, 160) if track.locked else QColor(80, 200, 255)
            # 下に黒い線を敷く 白っぽい絵の上でも枠が見えるように 近い値の枠は下の線も
            # 点線にする 下を実線のままにすると、点線の隙間が埋まって実線に見える
            under = QPen(QColor(0, 0, 0, 160), 3)
            line = QPen(color, 1.5)
            if outline.approximate:
                for pen in (under, line):
                    pen.setStyle(Qt.PenStyle.CustomDashLine)
                    pen.setDashPattern([4.0 * 1.5 / pen.widthF(), 4.0 * 1.5 / pen.widthF()])
            painter.setPen(under)
            painter.drawPolygon(corners)
            painter.setPen(line)
            painter.drawPolygon(corners)
            if track.locked:
                # ロック中は掴めない 掴む所を出すと動かせるように見える
                return
            knob = rotation_knob(self._widget_corners(outline), KNOB_LENGTH)
            if knob is not None:
                painter.setPen(QPen(color, 1))
                painter.drawLine(QPointF(*knob[0]), QPointF(*knob[1]))
                painter.setBrush(QColor(255, 255, 255))
                painter.drawEllipse(QPointF(*knob[1]), GRIP_SIZE / 2.0, GRIP_SIZE / 2.0)
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            half = GRIP_SIZE / 2.0
            for corner in corners:
                painter.drawRect(QRectF(corner.x() - half, corner.y() - half, GRIP_SIZE, GRIP_SIZE))
            # 拡大と回転の中心 どこを軸に回るかが見えないと、支点をずらした絵で戸惑う
            pivot = self.to_widget(outline.pivot)
            painter.setPen(QPen(color, 1))
            painter.drawLine(pivot + QPointF(-4, 0), pivot + QPointF(4, 0))
            painter.drawLine(pivot + QPointF(0, -4), pivot + QPointF(0, 4))
            self._paint_guides(painter)
        finally:
            painter.end()

    def _paint_guides(self, painter: QPainter) -> None:
        """吸い付いた所に線を引く（タイムラインの磁石と同じ黄色） 何に揃ったのかが見えるように"""
        if not self._guides:
            return
        left, top, width, height = self.canvas_rect()
        painter.setPen(QPen(QColor(255, 220, 60), 1))
        for axis, value in self._guides:
            point = self.to_widget((value, value))
            if axis == "x":
                painter.drawLine(QPointF(point.x(), top), QPointF(point.x(), top + height))
            else:
                painter.drawLine(QPointF(left, point.y()), QPointF(left + width, point.y()))

    def _region_corners(self, frame: RegionFrame) -> list[Point]:
        corners = []
        for corner in frame.corners:
            shown = self.to_widget(corner)
            corners.append((shown.x(), shown.y()))
        return corners

    def _paint_region(self) -> None:
        """部分モザイク・ぼかしと部分フィルタの範囲の枠と掴む所を重ねる（再生中は描かない）

        クリップの枠と色を分ける（橙） 同じ色だと、どちらを掴んでいるのか分からない
        楕円の範囲も外接する四角で出す 掴む所（角と辺の真ん中）は四角の方が分かりやすい
        """
        found = self._region()
        if found is None:
            return
        track, _, _, frame = found
        corners = self._region_corners(frame)
        polygon = [QPointF(x, y) for x, y in corners]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            under = QPen(QColor(0, 0, 0, 160), 3)
            line = QPen(QColor(160, 160, 160) if track.locked else _REGION_COLOR, 1.5)
            for pen in (under, line):
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([6.0 * 1.5 / pen.widthF(), 3.0 * 1.5 / pen.widthF()])
            painter.setPen(under)
            painter.drawPolygon(polygon)
            painter.setPen(line)
            painter.drawPolygon(polygon)
            if track.locked:
                return
            knob = rotation_knob(corners, KNOB_LENGTH)
            if knob is not None:
                painter.setPen(QPen(_REGION_COLOR, 1))
                painter.drawLine(QPointF(*knob[0]), QPointF(*knob[1]))
                painter.setBrush(_REGION_COLOR)
                painter.drawEllipse(QPointF(*knob[1]), GRIP_SIZE / 2.0, GRIP_SIZE / 2.0)
            painter.setBrush(_REGION_COLOR)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            half = GRIP_SIZE / 2.0
            for x, y in (*corners, *edge_points(corners)):
                painter.drawRect(QRectF(x - half, y - half, GRIP_SIZE, GRIP_SIZE))
        finally:
            painter.end()

    def _hit_region(
        self, position: QPointF
    ) -> tuple[Hit, Track, Clip, EffectId, RegionFrame] | None:
        found = self._region()
        if found is None:
            return None
        track, clip, effect_id, frame = found
        hit = region_hit_test(
            self._region_corners(frame),
            (position.x(), position.y()),
            grip=GRIP_SIZE,
            knob=KNOB_LENGTH,
        )
        return None if hit is None else (hit, track, clip, effect_id, frame)

    def _hit(self, position: QPointF) -> tuple[Hit, Track, Clip, Outline] | None:
        found = self._selected()
        if found is None:
            return None
        track, clip, outline = found
        hit = hit_test(
            self._widget_corners(outline),
            (position.x(), position.y()),
            grip=GRIP_SIZE,
            reach=ROTATE_REACH,
            knob=KNOB_LENGTH,
        )
        return None if hit is None else (hit, track, clip, outline)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if (
            not self._handles_enabled
            or self._playing
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mousePressEvent(event)
            return
        position = event.position()
        # 範囲を先に見る 範囲はクリップの枠の中にあることが多く、枠を先に見ると範囲を掴めない
        region = self._hit_region(position)
        if region is not None:
            hit, track, clip, effect_id, frame = region
            if track.locked:
                return
            self._begin_drag(
                _Drag(
                    hit=hit,
                    clip_id=clip.id,
                    project=self._project,
                    press=self.to_canvas(position),
                    last=self.to_canvas(position),
                    start=dict(frame.values),
                    corners=frame.corners,
                    pivot=frame.corners[0],
                    frame=self._frame,
                    region=(effect_id, frame),
                )
            )
            event.accept()
            return
        found = self._hit(position)
        if found is None:
            # 選んだクリップの枠の外 そこに描かれている一番手前のクリップを選び直す
            # 何も無い所では選択を変えない 絵の無い所を押しただけで選択が外れると、
            # 設定パネルが空になって戸惑う
            picked = pick_clip(
                self._project, self._frame, self.to_canvas(position), self.outline_of
            )
            if picked is None or picked == self._selection:
                return
            self._selection = picked
            self.clip_picked.emit(str(picked))
            self.update()
            found = self._hit(position)
            if found is None or found[0].grip is not Grip.MOVE:
                return
        hit, track, clip, outline = found
        if track.locked:
            return
        self._begin_drag(
            _Drag(
                hit=hit,
                clip_id=clip.id,
                project=self._project,
                press=self.to_canvas(position),
                last=self.to_canvas(position),
                start=start_values(clip, self._frame - clip.timeline_start),
                corners=outline.corners,
                pivot=outline.pivot,
                frame=self._frame,
            )
        )
        event.accept()

    def _snapped(self, drag: _Drag, current: Point, *, off: bool) -> Point:
        """動かした先を、画面の中央やほかの物の端へ吸い付けた所

        Shift を押している間は吸い付かない（タイムラインと同じ） 片方の向きだけに動かす
        決まりも Shift なので、押している間は向きをそろえるだけになる
        """
        self._guides = []
        if not self._snap or off or drag.snap is None:
            return current
        dx, dy = current[0] - drag.press[0], current[1] - drag.press[1]
        left, top, right, bottom = bounding_box(drag.corners)
        moving = (left + dx, top + dy, right + dx, bottom + dy)
        _, _, shown_width, _ = self.canvas_rect()
        reach = self._snap_distance * self.canvas_size()[0] / max(shown_width, 1.0)
        shift_x, shift_y, self._guides = snap_offset(moving, drag.snap, reach)
        return current[0] + shift_x, current[1] + shift_y

    def _begin_drag(self, drag: _Drag) -> None:
        """掴み始める 掴んでいる間だけキーボードを借りて Esc を受ける"""
        if drag.region is None and drag.hit.grip is Grip.MOVE and self._snap:
            drag.snap = self._snap_targets(drag.clip_id)
        self._drag = drag
        self.grabKeyboard()

    def _end_drag(self) -> _Drag | None:
        """掴むのをやめて、掴んでいた物を返す 借りたキーボードはタイムラインへ返す"""
        drag, self._drag = self._drag, None
        self._guides = []
        if drag is not None:
            self.releaseKeyboard()
        return drag

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        drag = self._drag
        if drag is None:
            self._hover(event.position())
            super().mouseMoveEvent(event)
            return
        current = self.to_canvas(event.position())
        modifiers = event.modifiers()
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if drag.region is not None:
            self._drag_region(drag, current, modifiers)
            event.accept()
            return
        if drag.hit.grip is Grip.MOVE:
            current = self._snapped(drag, current, off=shift)
            changes = moved_values(
                drag.start, drag.press, current, one_axis=shift, scale=self.canvas_scale()
            )
        elif drag.hit.grip is Grip.SCALE:
            changes = scaled_values(
                drag.start,
                drag.corners,
                drag.pivot,
                drag.press,
                current,
                separate=bool(modifiers & Qt.KeyboardModifier.AltModifier),
            )
        else:
            drag.turned += turn_between(drag.pivot, drag.last, current)
            changes = rotated_values(drag.start, drag.turned, snap=shift)
        drag.last = current
        drag.commands = transform_commands(
            drag.project, drag.clip_id, changes, drag.frame, keyframes=self._keyframe_drag
        )
        # 履歴に積まずに見せる 1 回のドラッグで何十段も積まない 離したときに 1 段
        self._showing_drag = True
        try:
            self.preview_requested.emit(list(drag.commands))
        finally:
            self._showing_drag = False
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        # 左を離したときだけ終える 先に消すと、左で掴んだまま右を離しただけで、確定も
        # 取り消しもされずに途中の絵がプレビューに残る
        if self._drag is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        drag = self._end_drag()
        assert drag is not None
        if drag.commands:
            labels = _REGION_LABELS if drag.region is not None else _LABELS
            self.commands_requested.emit(list(drag.commands), labels[drag.hit.grip])
        else:
            # 動かしてから元の所へ戻した 見せていた途中の絵を元のプロジェクトへ戻す
            self.preview_requested.emit([])
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self._drag is not None and event.key() == Qt.Key.Key_Escape:
            # 掴んでいる途中でやめる 見せていた途中の絵を元へ戻す
            self._cancel_drag()
            event.accept()
            return
        super().keyPressEvent(event)

    def _drag_region(self, drag: _Drag, current: Point, modifiers: Qt.KeyboardModifier) -> None:
        """範囲を掴んで動かした途中 値は掴んだ時点の枠から毎回出し直す"""
        assert drag.region is not None
        effect_id, frame = drag.region
        if drag.hit.grip is Grip.ROTATE:
            drag.turned += region_turn(frame, drag.last, current)
        drag.last = current
        changes = region_changes(
            frame,
            drag.hit,
            drag.press,
            current,
            turned=drag.turned,
            one_axis=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
            symmetric=bool(modifiers & Qt.KeyboardModifier.AltModifier),
            snap=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
        )
        located = drag.project.timeline.locate_clip(drag.clip_id)
        effect = (
            next((e for e in located[1].effects if e.id == effect_id), None)
            if located is not None
            else None
        )
        if located is None or effect is None:
            return
        clip = located[1]
        # キーのある値は再生位置（クリップの中に収めた時刻）のキーを動かす 設定パネルの ◆ と同じ
        local = min(max(drag.frame - clip.timeline_start, 0), max(clip.duration - 1, 0))
        drag.commands = value_commands(
            clip.id, effect, changes, local, keyframes=self._keyframe_drag
        )
        self._showing_drag = True
        try:
            self.preview_requested.emit(list(drag.commands))
        finally:
            self._showing_drag = False

    def _cancel_drag(self) -> None:
        drag = self._end_drag()
        if drag is not None and drag.commands:
            self.preview_requested.emit([])

    def _hover(self, position: QPointF) -> None:
        """掴める所の上で矢印の形を変える 何が起きるかを押す前に分かるように"""
        region = self._hit_region(position) if self._handles_enabled else None
        if region is not None and not region[1].locked:
            grip = region[0].grip
            if grip is Grip.MOVE:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif grip is Grip.ROTATE:
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif grip is Grip.EDGE:
                self.setCursor(
                    Qt.CursorShape.SizeVerCursor
                    if region[0].corner in (0, 2)
                    else Qt.CursorShape.SizeHorCursor
                )
            else:
                self.setCursor(
                    Qt.CursorShape.SizeFDiagCursor
                    if region[0].corner in (0, 2)
                    else Qt.CursorShape.SizeBDiagCursor
                )
            return
        found = self._hit(position) if self._handles_enabled else None
        if found is None or found[1].locked:
            self.unsetCursor()
            return
        grip = found[0].grip
        if grip is Grip.MOVE:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif grip is Grip.SCALE:
            self.setCursor(
                Qt.CursorShape.SizeFDiagCursor
                if found[0].corner in (0, 2)
                else Qt.CursorShape.SizeBDiagCursor
            )
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _apply_pending(self) -> None:
        """持ち越していた設定を置き場へ渡す **コンテキストが current な所で呼ぶこと**"""
        if self._release_cache_pending and self._cache is not None:
            self._release_cache_pending = False
            self._cache.release()
        if self._pending_budget is None or self._cache is None:
            return
        budget, self._pending_budget = self._pending_budget, None
        self._cache.set_budget(budget, self._frame)
        # 0 から増やしたときは、ここで初めて先読みできるようになる
        self._restart_prefetch()

    def _invalidate(self, invalidation: Invalidation) -> None:
        """変わった範囲の先読みを捨てる コンテキストが current な所で呼ぶこと"""
        if self._cache is not None:
            self._cache.invalidate(invalidation)

    def _restart_prefetch(self) -> None:
        """先読みを動かし直す 何か変わるたびに呼ぶ

        止まったままにしない 再生ヘッドが動けば、次に貯める先も変わっている
        """
        if self._background is not None:
            # 走り係は自分で回る 再生ヘッドと、再生中かどうかを渡すだけ
            self._idle.stop()
            self._background.set_paused(self._playing)
            self._background.set_playhead(self._frame)
            return
        if self._playing:
            self._idle.stop()
            return
        if self._wants_background():
            # 走り係はイベントループへ戻ってから作る ここは initializeGL や paintGL の
            # 中からも呼ばれ、そこで別のコンテキストを作ると画面のコンテキストが外れる
            self._idle.start()
            return
        if self._cache is None or not self._cache.enabled:
            self._idle.stop()
            return
        self._idle.start()

    def _wants_background(self) -> bool:
        """別のスレッドの先読みを作りに行くか"""
        if not self._prefetch_thread or self._background_broken or self._prefetch_bytes <= 0:
            return False
        if self._renderer is None or self._cache is None:
            return False
        context = self.context()
        return context is not None and context.isValid()

    def _start_background(self) -> bool:
        """別のスレッドの先読みを作る 作れたら ``True``

        作れなければ、理由を伝えて画面のスレッドの先読みへ戻る 共有した
        コンテキストを作れないドライバや、GL の版が足りない仮想環境がある
        """
        assert self._renderer is not None and self._cache is not None
        try:
            # 画面のスレッドで貯めていた絵を先に手放す 残すと、同じメモリを
            # 2 か所で使うことになる current にできなければ、手放すのは後回し
            self.makeCurrent()
            if QOpenGLContext.currentContext() is self.context():
                try:
                    self._cache.release()
                finally:
                    self.doneCurrent()
            else:
                # 窓が隠れていると current にならないまま戻る 覚えておかないと、
                # 走り係が動く間ずっと画面の側の絵も抱え、先読みの予算の 2 倍の
                # GPU のメモリを使い続ける
                self._release_cache_pending = True
            background = BackgroundPrefetch(
                self._project,
                share=self.context(),
                quality=self._quality,
                proxies=self._proxies,
                decode_threads=self._decode_threads,
                budget_bytes=self._prefetch_bytes,
                playhead=self._frame,
            )
        except Exception as exc:
            # 共有したコンテキストを作れないドライバや、GL の版が足りない環境がある
            # 先読みそのものは画面のスレッドで続けられる
            self._background_broken = True
            self.prefetch_stopped.emit(
                f"別のスレッドで先読みできないので、画面のスレッドで先読みする: {exc}"
            )
            return False
        background.failed.connect(self._background_failed)
        background.ended.connect(self._background_ended)
        self._background = background
        self._restart_prefetch()
        return True

    def _background_failed(self, background: object, reason: str) -> None:
        """走り係が 1 コマ描けなかった 走り係は次に何か変わるまで休む"""
        if background is self._background:
            self.prefetch_stopped.emit(reason)

    def _background_ended(self, background: object, reason: str) -> None:
        """走り係が続けられなくなった 画面のスレッドの先読みへ戻る

        閉じた後の走り係から遅れて届いた合図は捨てる 拾うと、設定で切って
        入れ直しただけの走り係まで「作れない」扱いになる
        """
        if background is not self._background:
            return
        self._background_broken = True
        self._close_background()
        self.prefetch_stopped.emit(reason)
        self._restart_prefetch()

    def _close_background(self) -> None:
        background, self._background = self._background, None
        if background is not None:
            background.close()
            # 閉じる前に走り係が捨てた控えも預かる 捨てたまま閉じると、壊れた控えを
            # 作り直す頼みが出ず、その素材は元の素材から読み続ける
            self._closed_discarded |= background.take_discarded()

    def _prefetch_step(self) -> None:
        """空き時間に 1 コマだけ描く

        描けなくなったら止める 止めないと、貯まりきった後も空き時間の
        たびに「次はどれか」を数え続ける
        """
        if self._playing or self._cache is None:
            # タイマーを止めても、すでに積まれた合図は届く 再生が始まった後に
            # 1 コマ描くと、出す側と GL を奪い合う
            self._idle.stop()
            return
        if self._background is not None:
            self._idle.stop()
            return
        if self._wants_background() and self._start_background():
            return
        if not self._cache.enabled and self._pending_budget is None:
            # 走り係を作れなかった所へ来た 画面のスレッドにも貯める場所が無い
            self._idle.stop()
            return
        started = time.perf_counter()
        try:
            # コンテキストを current にする所も中へ入れる ここで落ちたときに
            # だけ外へ抜けるのでは、守ったことにならない
            self.makeCurrent()
            if QOpenGLContext.currentContext() is not self.context():
                # **戻り値では分からない** makeCurrent は何も返さず、窓が隠れている
                # ときや端末が休んだ後は current にならないまま戻る
                # そのまま GL を触ると、別のコンテキストへ描くことになる
                self._idle.stop()
                self.prefetch_stopped.emit("GL を使えないので先読みを止めた")
                return
            try:
                self._apply_pending()
                filled = self._cache.step(self._frame)
            finally:
                # current にできたときだけ戻す できていないのに戻すと、
                # ほかが使っているコンテキストを外すことになる
                self.doneCurrent()
        except Exception as exc:
            # **先読みの失敗でプレビューを落とさない** ここは Qt のタイマーから
            # 呼ばれるので、投げるとイベントループの外まで抜けてアプリが終わる
            # 先読みは無くても絵は出る 止めて、理由を伝えるに留める
            self._idle.stop()
            self.prefetch_stopped.emit(f"先読みを止めた: {exc}")
            return

        if not filled:
            self._idle.stop()
            return

        elapsed = (time.perf_counter() - started) * 1000
        if elapsed >= SLOW_FRAME_MS:
            # 1 コマにこれだけ掛かるなら、貯まるまでずっと操作を受け付けられない
            # 次に再生ヘッドか中身が変わったら、また試す
            self._idle.stop()
            self.prefetch_stopped.emit(f"1 コマ {elapsed / 1000:.1f} 秒掛かるので先読みを止めた")
