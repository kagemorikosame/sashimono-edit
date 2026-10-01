"""プレビュー用の低解像度の控え（プロキシ）を作る

4K の素材は、重ねるとプレビューが 60fps（1 フレーム 16.7ms）に入らない
読む元のファイルを小さくして、デコードの分を軽くする

**測った値はここに集める** 他の所（設定の画面や :mod:`sashimono.ui.workspace`）は
この表を指す 同じ数を何か所にも書くと、測り直したときに片方だけ古くなる

``.venv\\Scripts\\python.exe tools\\bench_proxy.py``（3840x2160 を 30fps で 3 秒 95 パーセンタイル
プレビューと同じ道＝合成と、1920x1080 の描画先への転送 CPU へ読み戻す分は数えない
重ねる枚数は**別々の素材**として置く 同じ素材を重ねるとデコードが 1 回で済む）

=================== ========= ======== ========== ========== ========== ==========
条件                  元 + 等倍  元 + 1/2  控え + 等倍  控え + 1/2  控え + 1/4  入る組
=================== ========= ======== ========== ========== ========== ==========
1 枚                  12.3ms    11.7ms   6.1ms      5.7ms      5.6ms      元 + 等倍
2 枚                  20.1ms    19.3ms   8.0ms      7.8ms      7.6ms      控え + 等倍
3 枚                  29.0ms    28.5ms   11.0ms     10.1ms     10.1ms     控え + 等倍
3 枚 + blur           62.1ms    61.2ms   16.4ms     13.6ms     13.6ms     控え + 等倍
3 枚 + blur/glow      65.8ms    62.7ms   26.3ms     18.6ms     17.3ms     **無い**
=================== ========= ======== ========== ========== ========== ==========

「入る組」は予算（60fps の 1 コマ）に入る一番きれいな組

読み取れること

* **1 枚だけなら元の素材のままで入る** 重ねた時点で外れる（2 枚で 20.1ms）
* **効くのは控えの側** 3 枚で 29.0 → 11.0ms 画面を落とすだけでは 28.5ms で
  ほとんど効かない（重ねた絵ではデコードが重さの中心）
* **4K を 3 枚重ねて blur を積んでも、控えなら入る**（16.4ms 予算まで 0.3ms しかないので、
  画面も 1/2 に落とすと 13.6ms で余裕が出る）
* **blur/glow まで積むと、どの組でも入らない**（一番速い控え + 1/4 で 17.3ms）
  ここから先は先読み（バックグラウンドレンダリング）の仕事で、控えでは届かない
* 1ms 前後のばらつきがあるので、細かい上下は読まない（控え + 1/4 が
  控え + 1/2 と同じに出ることもある）

先読み（:mod:`sashimono.engine.render.prefetch`）

``--prefetch`` を足して測った 4K 3 枚 + blur（元の素材 + 等倍 大きさは既定のまま）

.. code-block:: text

    .venv\\Scripts\\python.exe tools\\bench_proxy.py --layers 3 --effects blur --prefetch

===================== ========
貯める 1 枚            66.1ms
出す 1 枚（貯まった後） 0.3ms
3 秒ぶん貯めるのに      5.5 秒
===================== ========

* **出すのは 0.3ms** 重さに関係なく転送だけになる どんなに積んでも、
  貯まってさえいれば予算に入る
* **貯めるのは元の重さのまま** 3 秒ぶんに 5.5 秒 掛かるのは手が止まっている間で、
  再生中ではない 先読みは「間に合わせる」仕組みではなく「先に払っておく」仕組み
* 控えと組にすると貯めるのも速くなる（控え + 1/4 なら 1 枚 17.5ms #98 の前に測った値）
* **貯めるのが現実的でない所もある** 1080p の動画を 20 本重ねて blur/glow を
  積むと 1 枚 2178.2ms で、3 秒ぶん（90 枚）貯めるのに 175 秒（#98 の前に測った値）
  出すのは 0.3ms のままだが、先読みは万能ではない
  そこまで重ねる作り方は、素材を先に書き出して差し替える
  （1 枚の値は 95 パーセンタイル、175 秒は 90 枚の実測の合計 割り算で一致はしない
  中央の重さは 95 パーセンタイルより軽い）

控えを使うのは**プレビューだけ** 書き出しは必ず元の素材から読む 混ざると、
画面では気付かないまま低解像度の絵が最終出力に入る

鍵は解析キャッシュと同じ「パス + サイズ + 更新時刻」に、控えの高さを足して作る
素材を差し替えても設定で大きさを変えても別の鍵になるので、古い控えを掴むことはない
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import av.container
import av.error
import av.video.stream
import numpy as np

from sashimono.core.model import MediaId, MediaItem
from sashimono.engine.cache.progress import JobBoard, ProgressSnapshot
from sashimono.engine.cache.store import CacheStore, media_key
from sashimono.engine.colorspace import tag_bt709, to_bt709, to_rgb_array
from sashimono.engine.decode import ProbeError, VideoDecoder, probe_media
from sashimono.engine.decode.rational import as_fraction

__all__ = [
    "MEASURED_PREFETCH_MS",
    "PROXY_HEIGHT",
    "ProxyBuilder",
    "ProxyStore",
    "create_proxy",
    "is_worth_proxying",
    "proxy_codecs",
]

#: 上の表の「3 枚」の行（元 + 等倍 / 控え + 等倍 / 控え + 1/4）と「1 枚」の元 + 等倍
#: 設定の画面がここから文言を組み立てる 数を画面へ直に書くと、測り直したときに
#: 片方だけ古くなる 機械によって変わるので、あくまで目安
MEASURED_THREE_LAYERS_MS = (29.0, 11.0, 10.1)
MEASURED_ONE_LAYER_MS = 12.3

#: 先読みの表（貯める 1 枚 / 出す 1 枚） 4K 3 枚 + blur を元の素材のまま
#: 設定の画面はここを指す
MEASURED_PREFETCH_MS = (66.1, 0.3)

#: 60fps の 1 コマ（ミリ秒） 上の表の予算
BUDGET_MS = 1000 / 60

#: 控えの縦の画素数 1080p の素材なら作らない（下の :func:`is_worth_proxying` 参照）
PROXY_HEIGHT = 540

#: 控えを作る値打ちがある縦の画素数 これ以下なら元のまま読む方が速い
#: 変換にかかる時間の方が、再生で取り返す時間より長くなる
MIN_SOURCE_HEIGHT = 1081

#: 控えの置き場（:class:`~sashimono.engine.cache.store.CacheStore` の名前空間）
NAMESPACE = "proxy"

#: 控えの中身の版 作り方を変えて、前の控えをそのまま読むと絵が違ってしまうときに上げる
#: 2: BT.709 へ変換してタグを付けるようにした（#61）
FORMAT_VERSION = 2

#: 控えに使うコーデックの優先順 NVENC は CPU をほとんど使わないので、
#: 変換しながら編集を続けられる 無い環境では CPU の libx264 へ落ちる
CODEC_PREFERENCE = ("h264_nvenc", "h264_qsv", "libx264")


def proxy_codecs() -> list[str]:
    """この環境で控えを作れるコーデックを、優先順に返す"""
    found: list[str] = []
    for name in CODEC_PREFERENCE:
        try:
            av.codec.Codec(name, "w")
        except Exception:
            # 入っていないコーデックは Codec() が投げる 種類は環境によって違うので
            # 名前で拾わない ここで落とすと、1 つ無いだけで控えが作れなくなる
            continue
        found.append(name)
    return found


def is_worth_proxying(media: MediaItem) -> bool:
    """控えを作る値打ちがあるか

    小さい素材まで変換すると、待たされるだけで速くならない
    """
    if not media.has_video or media.is_still:
        return False
    return any(stream.display_size[1] >= MIN_SOURCE_HEIGHT for stream in media.video_streams)


class ProxyStore:
    """素材ごとの控えを置く場所

    :class:`~sashimono.engine.cache.store.CacheStore` の上に乗る 作るのは
    :func:`create_proxy` で、こちらは「どこに置くか」と「あるか」だけを見る
    """

    def __init__(self, store: CacheStore | None = None, *, height: int = PROXY_HEIGHT) -> None:
        self._store = store if store is not None else CacheStore()
        self._height = height

    @property
    def height(self) -> int:
        return self._height

    def key_for(self, media: MediaItem) -> str:
        # 縦の画素数を鍵に混ぜる 設定を変えたときに、前の大きさの控えを掴まない
        # 書式の版も混ぜる 版 1 の控えはタグが無く元の行列のままなので、今の読み方では
        # BT.601 とみなされて色がずれる 版を上げれば作り直される
        return media_key(media.path, extra=f"proxy{self._height}v{FORMAT_VERSION}")

    def path_for(self, media: MediaItem) -> Path:
        """控えの置き場 まだ無くてもパスは返す"""
        return self._store.path_for(NAMESPACE, self.key_for(media), ".mp4")

    def find(self, media: MediaItem) -> Path | None:
        """使える控え 無ければ ``None``"""
        path = self.path_for(media)
        try:
            return path if path.stat().st_size > 0 else None
        except OSError:
            # 空のファイルは作りかけか失敗の跡 掴むと「映らない素材」になる
            return None

    def discard(self, media: MediaItem) -> None:
        """使えない控えを捨てる

        残すと :meth:`find` が毎回それを返し、作り直す機会も無いまま
        「その素材だけ映らない」が続く 捨てておけば次の求めで作り直せる
        """
        self.path_for(media).unlink(missing_ok=True)

    def prepare(self, media: MediaItem) -> Path:
        """書き込み先を用意して返す 親フォルダも作る"""
        return self._store.prepare(NAMESPACE, self.key_for(media), ".mp4")


def create_proxy(
    source: Path,
    target: Path,
    *,
    height: int = PROXY_HEIGHT,
    stream_index: int | None = None,
    codec: str | None = None,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path | None:
    """``source`` の映像を縮めて ``target`` へ書く 作れなければ ``None``

    重い処理なのでバックグラウンドで呼ぶこと ``should_cancel`` が真を返したら
    書きかけを消して ``None`` を返す

    音は入れない プレビューの音は元の素材から混ぜるので、控えに入れても
    使われないまま場所を取る

    **時刻は元の素材と同じに保つ** 縮めるのは大きさだけで、フレームの並びも
    長さも変えない ここがずれると、控えのときだけ絵が 1 フレーム早い/遅い
    という、原因の分かりにくい不具合になる
    """
    names = [codec] if codec is not None else proxy_codecs()
    if not names:
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    # 書きかけの名前を分けておく 同じ素材を 2 回読み込んでも取り合いにならず、
    # 途中で落ちても壊れた控えが残らない
    working = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.part")

    for name in names:
        try:
            made = _transcode(
                source,
                working,
                height=height,
                stream_index=stream_index,
                codec=name,
                progress=progress,
                should_cancel=should_cancel,
            ) and _has_a_frame(working)
            # 最後の 1 枚を書いたあとにやめると言われることがある 置かずに捨てる
            # 置くと、素材を外したり控えを切ったりしたのに控えが残り、
            # 「止めたはずなのに使われている」ことになる
            if made and (should_cancel is None or not should_cancel()):
                working.replace(target)
                return target
        except (av.error.FFmpegError, OSError, ValueError):
            # このコーデックでは作れなかった 次の候補へ移るので、ここで投げると
            # NVENC が使えない環境で控えが一切作れなくなる
            pass
        working.unlink(missing_ok=True)
        if should_cancel is not None and should_cancel():
            # やめると言われている 次の候補を試さない 試すと、候補の数だけ
            # 変換を始め直すことになり、止めたのに止まらないように見える
            return None
    return None


def _has_a_frame(path: Path) -> bool:
    """作った控えが 1 枚でも出せるか

    映像の見出しはあるのに 1 枚も復号できない素材では、変換そのものは
    成功して**中身の無い控え**ができる 置いてしまうと、描く側が捨てて
    作り直しを頼み、また同じものができる、の繰り返しになる ここで止める
    """
    try:
        with VideoDecoder(path) as decoder:
            return decoder.frame_at(Fraction(0)) is not None
    except ProbeError:
        return False


def _transcode(
    source: Path,
    target: Path,
    *,
    height: int,
    stream_index: int | None,
    codec: str,
    progress: Callable[[float], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> bool:
    """1 つのコーデックで変換を試す 終わりまで書けたら ``True``"""
    with av.open(str(source)) as container:
        streams = container.streams.video
        if not streams:
            return False
        stream = (
            streams[0]
            if stream_index is None
            else next((s for s in streams if s.index == stream_index), streams[0])
        )
        stream.thread_type = "AUTO"

        # 回転の印はコンテナに付いている 控えへは引き継がれないので、
        # **画素の側を回して**焼き込む 引き継がずに黙って写すと、スマホで撮った
        # 縦の映像が控えのときだけ横向きになる（デコーダは開いたファイルの
        # 印だけを見て回すため）
        rotation = _rotation_of(source, stream.index)
        size = _target_size(stream, height, rotation)
        if size is None:
            return False
        width, scaled = size
        # 回す前の大きさ 90 度と 270 度では縦横が入れ替わる
        before = (scaled, width) if rotation in (90, 270) else (width, scaled)
        rate = stream.average_rate or Fraction(30)
        duration = _duration_of(container, stream)

        # 書式を名前から当てさせない 書きかけの名前は ``.part`` で終わるので、
        # 拡張子から当てる作りだと「書式が分からない」で落ちる
        with av.open(str(target), mode="w", format="mp4") as output:
            video = cast(
                "av.video.stream.VideoStream", output.add_stream(codec, rate=as_fraction(rate))
            )
            video.width = width
            video.height = scaled
            video.pix_fmt = "yuv420p"
            tag_bt709(video)
            # 控えは見るためだけのもの 画質より、小ささと変換の速さを取る
            video.bit_rate = width * scaled * 4
            video.time_base = stream.time_base

            for frame in container.decode(stream):
                if should_cancel is not None and should_cancel():
                    return False
                converted = _shrunk(frame, before, rotation)
                # 元の時刻をそのまま持たせる 振り直すと、可変フレームレートの
                # 素材で控えと元の絵がずれる
                converted.pts = frame.pts
                converted.time_base = frame.time_base
                for packet in video.encode(converted):
                    output.mux(packet)
                if progress is not None and duration > 0 and frame.time is not None:
                    progress(min(1.0, float(frame.time) / float(duration)))

            for packet in video.encode():
                output.mux(packet)

    if progress is not None:
        progress(1.0)
    return True


def _shrunk(frame: av.VideoFrame, size: tuple[int, int], rotation: int) -> av.VideoFrame:
    """縮めて、必要なら回した 1 枚

    回すのはここだけ 控えを作るときの 1 回で済み、再生のたびには走らない
    """
    # 元の行列のまま縮めてはいけない 控えは 540p なので、タグが無いと読む側は SD とみなして
    # BT.601 で読む 元が HD の BT.709 なら、控えのときだけ色がずれる BT.709 へ変換して
    # タグも付け（:func:`tag_bt709`）、控えの大きさに関係なく同じ色で読めるようにする
    converted = to_bt709(frame, "yuv420p", width=size[0], height=size[1])
    if rotation == 0:
        return converted
    # 回すのは色の並びが素直な rgb24 で yuv420p のまま回すと、
    # 色差の面が半分の大きさなので縦横がずれる
    image = np.rot90(to_rgb_array(converted, "rgb24"), k=-rotation // 90)
    turned = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
    return to_bt709(turned, "yuv420p")


def _rotation_of(source: Path, stream_index: int) -> int:
    """写すストリームに付いている回転角 読めなければ 0

    :func:`~sashimono.engine.decode.probe_media` と同じ所から読む 別の読み方を
    すると、元の素材と控えで向きが食い違う 写すストリームを指して読むのは、
    1 本目とは別のストリームを控えにする日が来ても食い違わないようにするため
    """
    try:
        info = probe_media(source)
    except ProbeError:
        return 0
    for stream in info.video_streams:
        if stream.index == stream_index:
            return stream.rotation
    return info.video_streams[0].rotation if info.video_streams else 0


def _target_size(
    stream: av.video.stream.VideoStream, height: int, rotation: int = 0
) -> tuple[int, int] | None:
    """縮めたあとの**見た目の**大きさ 縦横とも偶数にする（yuv420p が奇数を受けない）

    90 度と 270 度の素材は、縦横が入れ替わった側が見た目の大きさ
    こちらで縮めないと、縦の映像の控えだけ小さすぎる/大きすぎることになる
    """
    source_width = stream.codec_context.width
    source_height = stream.codec_context.height
    if source_width <= 0 or source_height <= 0:
        return None
    if rotation in (90, 270):
        source_width, source_height = source_height, source_width
    scaled = min(height, source_height)
    width = max(2, round(source_width * scaled / source_height))
    return width - width % 2, max(2, scaled - scaled % 2)


def _duration_of(
    container: av.container.InputContainer, stream: av.video.stream.VideoStream
) -> float:
    """素材の長さ（秒） 進み具合を出すためだけに使う"""
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration) / av.time_base
    return 0.0


class ProxyBuilder:
    """控えをバックグラウンドで作る

    変換は**同時に 1 本だけ**にする 何本も並べると CPU と GPU の符号化器を
    取り合って、編集中のプレビューそのものが重くなる 控えは待てば済むが、
    操作が重いのは待てない

    :class:`~sashimono.engine.cache.analyzer.MediaAnalyzer` と同じ作りで、
    取り消しは「鍵の集合」で持つ Future を辞書に入れ直す形にすると、
    投入前に終わったものを消し損ねる
    """

    def __init__(self, store: ProxyStore | None = None) -> None:
        self._store = store if store is not None else ProxyStore()
        # 本数は外から変えられないようにする 増やせる形にすると、
        # 「同時に 1 本だけ」という約束が呼ぶ側の都合で破れる
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sashimono-proxy")
        self._lock = threading.Lock()
        #: 作っている最中の進み具合 UI に出すために持つ
        self._progress: dict[MediaId, float] = {}
        self._running: set[MediaId] = set()
        self._cancelled: set[MediaId] = set()
        self._closed = False
        #: 画面へ出す進み具合 ``_progress`` は描画の側が 1 本ずつ引くためのもので、
        #: 何本のうち何本目か・失敗したかは持たない
        self._board = JobBoard()

    @property
    def store(self) -> ProxyStore:
        return self._store

    def poll(self) -> ProgressSnapshot:
        """画面へ出す進み具合 画面のスレッドのタイマーから呼ぶ（:meth:`JobBoard.poll`）"""
        return self._board.poll()

    def settle(self, seen: ProgressSnapshot) -> bool:
        """画面が終わりを見届けた ひと続きの数を戻す（:meth:`JobBoard.settle`）"""
        return self._board.settle(seen)

    def progress(self, media_id: MediaId) -> float | None:
        """作っている最中なら 0..1 それ以外は ``None``

        描画のたびに呼ばれるので、ここでは決してブロックしない
        """
        with self._lock:
            return self._progress.get(media_id)

    def request(
        self,
        media: MediaItem,
        *,
        on_ready: Callable[[MediaId], None] | None = None,
        on_progress: Callable[[MediaId], None] | None = None,
    ) -> None:
        """控えを作るよう予約する すでにあるもの・作っている最中のものは何もしない"""
        if not is_worth_proxying(media) or self._store.find(media) is not None:
            return

        with self._lock:
            if self._closed or media.id in self._running:
                return
            self._cancelled.discard(media.id)
            self._running.add(media.id)
            self._progress[media.id] = 0.0
            self._board.start(media.id, media.id)

        def report(value: float) -> None:
            with self._lock:
                self._progress[media.id] = value
            self._board.report(media.id, value)
            if on_progress is not None:
                on_progress(media.id)

        def cancelled() -> bool:
            with self._lock:
                return media.id in self._cancelled

        def run() -> None:
            made: Path | None = None
            # 止められていたかどうかを、消す前に控えておく 消してから見ると
            # 分からなくなり、閉じたあとに「控えができた」と伝えてしまう
            stopped = False
            failure: str | None = None
            try:
                made = create_proxy(
                    media.path,
                    self._store.prepare(media),
                    height=self._store.height,
                    # 1 本目の映像だけを控えにする 読む側（レンダラ）も
                    # 1 本目を指すクリップにしか渡さない
                    stream_index=media.video_streams[0].index if media.video_streams else None,
                    progress=report,
                    should_cancel=cancelled,
                )
            except Exception as exc:  # 裏のスレッドの例外は誰にも見えずに消える
                # 投げ直さずに失敗として数える 前は executor の中で黙って消え、
                # 控えが無いまま「作っている最中」にも見えなかった
                failure = f"控えを作れなかった: {exc}"
            finally:
                with self._lock:
                    stopped = self._closed or media.id in self._cancelled
                    self._running.discard(media.id)
                    self._cancelled.discard(media.id)
                    self._progress.pop(media.id, None)
            if stopped:
                self._board.drop(media.id)
            elif made is None:
                # 開けない・映像の大きさが分からない素材は None で返る 何も出さないと、
                # プレビューが重いまま控えを待ち続けることになる
                self._board.finish(media.id, failure or "控えを作れなかった")
            else:
                self._board.finish(media.id)
            # 止められていたなら伝えない 伝えると、窓を閉じている最中や
            # 控えを切った直後に「控えができた」として描き直しが走る
            if made is not None and not stopped and on_ready is not None:
                on_ready(media.id)

        with self._lock:
            # close と同じロックの中で投げる 外で投げると、止めた直後の
            # executor へ投げて RuntimeError になる
            if self._closed:
                self._running.discard(media.id)
                self._progress.pop(media.id, None)
                self._board.drop(media.id)
                return
            self._executor.submit(run)

    def forget(self, media_id: MediaId) -> None:
        """素材を外したときに、作りかけを止める"""
        with self._lock:
            if media_id in self._running:
                self._cancelled.add(media_id)
        self._board.forget(media_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled.update(self._running)
            self._executor.shutdown(wait=False, cancel_futures=True)
