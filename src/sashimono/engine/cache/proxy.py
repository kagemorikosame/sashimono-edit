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

キーフレームの間隔（#430）

``.venv\\Scripts\\python.exe tools\\bench_seek.py``（1920x1080 30fps 60 秒の素材を
キーフレームの間隔だけ変えて作り、ランダムな 80 か所へ飛んで 1 コマ出すまで
控えは 50 秒間隔の素材から :func:`create_proxy` で作った 540p 版 2 の控えは
変える前の :func:`create_proxy` で同じ素材から作り、40 か所で測った）

======================= ========== ================
読む物                    中央値       95 パーセンタイル
======================= ========== ================
元 1 秒間隔                21.9ms     29.7ms
元 2 秒間隔                23.6ms     36.3ms
元 4 秒間隔                31.1ms     50.1ms
元 8 秒間隔                46.7ms     90.2ms
元 16 秒間隔               79.9ms     170.8ms
元 50 秒間隔               197.3ms    419.4ms
控え 8.3 秒間隔（版 2）     39.4ms     60.2ms
控え 0.5 秒間隔（版 3）     5.8ms      7.6ms
======================= ========== ================

読み取れること

* **飛ぶ待ちを決めるのはキーフレームの間隔** 2 秒までは 1 秒と変わらず、4 秒で少し増え、
  8 秒から先は間隔に比例して増える 素材の長さそのものはほとんど響かない
* **1080p でも間隔が長ければ控えが効く** 50 秒間隔なら 197.3 → 5.8ms
* **控え自身の間隔も短くないと効かない** 版 2 までの控えは符号化器の既定（250 コマ
  30fps で 8.3 秒）のままで、540p に縮めても 39.4ms 1080p で 2 秒間隔の元（23.6ms）より遅かった
  版 3 から 0.5 秒ごとにキーフレームを置く（:data:`PROXY_KEYFRAME_SECONDS`）
* 線は :data:`KEYFRAME_GAP_LIMIT`（4 秒） 4 秒までは元のまま読んでも 2 秒間隔と 8ms ほどしか
  違わず、変換で CPU か GPU を使う方が損 8 秒では倍になる 録画ソフトの既定に多い
  250 コマごと（60fps で 4.2 秒 30fps で 8.3 秒）は線を超えるので控えを作る

控えを使うのは**プレビューだけ** 書き出しは必ず元の素材から読む 混ざると、
画面では気付かないまま低解像度の絵が最終出力に入る

鍵は解析キャッシュと同じ「パス + サイズ + 更新時刻」に、控えの高さを足して作る
素材を差し替えても設定で大きさを変えても別の鍵になるので、古い控えを掴むことはない
"""

from __future__ import annotations

import contextlib
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
from sashimono.engine.decode import ProbeError, VideoDecoder, keyframe_gap, probe_media
from sashimono.engine.decode.rational import as_fraction
from sashimono.engine.encoders import (
    PRIORITY_AUTO,
    MonotonicProgress,
    encoder_options,
    order_by_priority,
    usable,
)

__all__ = [
    "KEYFRAME_GAP_LIMIT",
    "MEASURED_KEYFRAME_SEEK_MS",
    "MEASURED_PREFETCH_MS",
    "PROXY_HEIGHT",
    "PROXY_KEYFRAME_SECONDS",
    "ProxyBuilder",
    "ProxyStore",
    "create_proxy",
    "is_worth_proxying",
    "needs_keyframe_check",
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
#: ただしキーフレームの間隔が長い素材は別（:data:`KEYFRAME_GAP_LIMIT`）
MIN_SOURCE_HEIGHT = 1081

#: これより長い間キーフレームが無い素材は、小さくても控えを作る（秒 上の表の「線」）
#: 飛ぶたびに遠くのキーフレームから読み直すので、縦の画素数が小さくても飛ぶ待ちが長い
KEYFRAME_GAP_LIMIT = Fraction(4)

#: 控えに置くキーフレームの間隔（秒） 飛んだ先の 1 コマを出すのに読み直す量を決める
#: 符号化器の既定（250 コマ）に任せると 30fps で 8.3 秒になり、控えなのに飛ぶ待ちが残る
#: 0 にして全部をキーフレームにしないのは、同じビットレートで絵が粗くなり、
#: NVENC が B フレームの数より短い間隔を断るため
PROXY_KEYFRAME_SECONDS = Fraction(1, 2)

#: 上の表の、元 1080p の 2 秒・8 秒・50 秒間隔と 0.5 秒間隔の控え（中央値 ミリ秒）
#: 設定の画面がここから文言を組み立てる
MEASURED_KEYFRAME_SEEK_MS = (23.6, 46.7, 197.3, 5.8)

#: 控えの置き場（:class:`~sashimono.engine.cache.store.CacheStore` の名前空間）
NAMESPACE = "proxy"

#: 控えの中身の版 作り方を変えて、前の控えをそのまま読むと絵が違ってしまうときに上げる
#: 2: BT.709 へ変換してタグを付けるようにした（#61）
#: 3: キーフレームを 0.5 秒ごとに置くようにした（#430） 絵は同じだが、版 2 の控えは
#: 8 秒ほどの間隔で、飛ぶ待ちが元の 1080p より長いことがある 作り直させる
FORMAT_VERSION = 3

#: 控えに使うコーデックの優先順 GPU のエンコーダは CPU をほとんど使わないので、
#: 変換しながら編集を続けられる 無い環境では CPU の libx264 へ落ちる
#: 並びは書き出し（:data:`~sashimono.engine.encode.exporter.VIDEO_CODEC_PREFERENCE`）と同じ
#: NVIDIA → AMD → Intel → CPU（#365）
CODEC_PREFERENCE = ("h264_nvenc", "h264_amf", "h264_qsv", "libx264")


def proxy_codecs(priority: str = PRIORITY_AUTO) -> list[str]:
    """この環境で控えを作れるコーデックを、優先順に返す

    入っているだけでは数えず、開けて 1 コマ書けた物だけを返す（:func:`usable`）
    前は入っているかだけを見ていた AMF や QSV は GPU の無い機械でも FFmpeg に入っているので、
    素材を読み込むたびに開けない物から変換を始め、失敗してから次へ移っていた
    ``priority`` の作り手を前へ寄せる 知らない値は ``ValueError``
    """
    return order_by_priority([name for name in CODEC_PREFERENCE if usable(name)], priority)


def is_worth_proxying(media: MediaItem, keyframe_gap: Fraction | None = None) -> bool:
    """控えを作る値打ちがあるか

    小さい素材まで変換すると、待たされるだけで速くならない ただしキーフレームの間隔が
    :data:`KEYFRAME_GAP_LIMIT` より長い素材（``keyframe_gap`` 調べていなければ ``None``）は、
    小さくても飛ぶたびに待たされるので作る
    """
    if not media.has_video or media.is_still:
        return False
    if any(stream.display_size[1] >= MIN_SOURCE_HEIGHT for stream in media.video_streams):
        return True
    return keyframe_gap is not None and keyframe_gap > KEYFRAME_GAP_LIMIT


def needs_keyframe_check(media: MediaItem) -> bool:
    """大きさでは決まらず、キーフレームの間隔を調べて決める素材か

    大きい素材は調べなくても作る 調べると、4K の素材のたびに無駄に読むことになる
    """
    return media.has_video and not media.is_still and not is_worth_proxying(media)


class ProxyStore:
    """素材ごとの控えを置く場所

    :class:`~sashimono.engine.cache.store.CacheStore` の上に乗る 作るのは
    :func:`create_proxy` で、こちらは「どこに置くか」と「あるか」だけを見る

    ``long_gop`` は、キーフレームの間隔が長い小さい素材にも控えを使うか（本人の設定）
    切ったときは作らないだけでなく、前に作った控えも :meth:`find` で返さない 返すと、
    切ったのに 1080p の素材が粗い控えのまま映り続ける
    """

    def __init__(
        self,
        store: CacheStore | None = None,
        *,
        height: int = PROXY_HEIGHT,
        long_gop: bool = True,
    ) -> None:
        self._store = store if store is not None else CacheStore()
        self._height = height
        self._long_gop = long_gop

    @property
    def height(self) -> int:
        return self._height

    @property
    def long_gop(self) -> bool:
        """キーフレームの間隔が長い小さい素材にも控えを使うか"""
        return self._long_gop

    def key_for(self, media: MediaItem, version: int = FORMAT_VERSION) -> str:
        # 縦の画素数を鍵に混ぜる 設定を変えたときに、前の大きさの控えを掴まない
        # 書式の版も混ぜる 版 1 の控えはタグが無く元の行列のままなので、今の読み方では
        # BT.601 とみなされて色がずれる 版を上げれば作り直される
        return media_key(media.path, extra=f"proxy{self._height}v{version}")

    def discard_older(self, media: MediaItem) -> None:
        """同じ素材・同じ大きさの、前の版の控えを捨てる 今の版を作り終えたときに呼ぶ

        置き場には古い物を片付ける仕組みが無い 版を上げると前の控えはもう読まれないのに、
        素材 1 本ぶんの控え（1080p の 1 分で十数 MB）がずっと残る
        """
        for version in range(1, FORMAT_VERSION):
            path = self._store.path_for(NAMESPACE, self.key_for(media, version), ".mp4")
            path.unlink(missing_ok=True)

    def path_for(self, media: MediaItem) -> Path:
        """控えの置き場 まだ無くてもパスは返す"""
        return self._store.path_for(NAMESPACE, self.key_for(media), ".mp4")

    def find(self, media: MediaItem) -> Path | None:
        """使える控え 無ければ ``None``"""
        if not self._long_gop and needs_keyframe_check(media):
            return None
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
    encoder_priority: str = PRIORITY_AUTO,
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

    ``codec`` を名指ししなければ、``encoder_priority`` の作り手を先に試す（#365）
    """
    names = [codec] if codec is not None else proxy_codecs(encoder_priority)
    if not names:
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    # 書きかけの名前を分けておく 同じ素材を 2 回読み込んでも取り合いにならず、
    # 途中で落ちても壊れた控えが残らない
    working = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.part")
    # 次の候補で作り直しても進み具合を戻さない（#365） 戻ると素材の一覧の進み具合が下がる
    shown = MonotonicProgress(progress) if progress is not None else None

    for index, name in enumerate(names):
        if index > 0 and shown is not None:
            shown.start_again()
        try:
            made = _transcode(
                source,
                working,
                height=height,
                stream_index=stream_index,
                codec=name,
                progress=shown.report if shown is not None else None,
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
            # AMF は品質と上限のビットレートを書き出しと同じ組み方で渡す（#365） 試し
            # （:func:`usable`）も同じ組み方で開くので、選択肢の名前を駆動が受けないなら試しで外れる
            options = encoder_options(codec, video.bit_rate)
            if options:
                video.options = dict(options)
            video.time_base = stream.time_base
            # 飛んだ先を早く出すための控えなので、キーフレームを細かく置く（上の表）
            # B フレームも切る NVENC は B フレームの数 + 1 より短い間隔を断り、
            # 断られると CPU の libx264 まで落ちて変換が遅くなる
            video.codec_context.gop_size = _keyframe_every(as_fraction(rate))
            video.codec_context.max_b_frames = 0

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


def _keyframe_every(rate: Fraction) -> int:
    """控えのキーフレームを何コマごとに置くか（:data:`PROXY_KEYFRAME_SECONDS` をコマに直す）

    2 コマより細かくしない NVENC は 1 コマごと（全部キーフレーム）を B フレームが
    無くても断る 低いフレームレート（1fps の定点の録画など）で 0 や 1 になるのを防ぐ
    """
    return max(2, round(rate * PROXY_KEYFRAME_SECONDS))


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

    def __init__(
        self, store: ProxyStore | None = None, *, encoder_priority: str = PRIORITY_AUTO
    ) -> None:
        self._store = store if store is not None else ProxyStore()
        # どの作り手のエンコーダを先に試すか（#365） 本人の設定から来る 変えたときは
        # 呼ぶ側が作り直す（大きさを変えたときと同じ） 途中で替えられる形にすると、
        # 走っている変換と次の変換で選び方が混ざる
        self._encoder_priority = encoder_priority
        # 本数は外から変えられないようにする 増やせる形にすると、
        # 「同時に 1 本だけ」という約束が呼ぶ側の都合で破れる
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sashimono-proxy")
        self._lock = threading.Lock()
        #: 作っている最中の進み具合 UI に出すために持つ
        self._progress: dict[MediaId, float] = {}
        self._running: set[MediaId] = set()
        self._cancelled: set[MediaId] = set()
        #: 取り消し印の付いた変換が走っている間に、また頼まれた物（素材と知らせ先）
        #: 走っている変換は控えを残さずに終わるので、終わってから作り直す
        #: （:class:`~sashimono.engine.cache.analyzer.MediaAnalyzer` と同じ） 捨てると、
        #: 素材一覧から消してすぐ取り消したときに、控えがいつまでも作られない
        self._again: dict[
            MediaId,
            tuple[MediaItem, Callable[[MediaId], None] | None, Callable[[MediaId], None] | None],
        ] = {}
        self._closed = False
        #: 画面へ出す進み具合 ``_progress`` は描画の側が 1 本ずつ引くためのもので、
        #: 何本のうち何本目か・失敗したかは持たない
        self._board = JobBoard()
        #: キーフレームの間隔を調べて、控えは要らないと分かった素材（控えの鍵）
        #: 設定を当て直すたびに全部の素材を頼み直すので、覚えないと毎回読み直す
        #: 鍵は素材の大きさと更新時刻を含むので、差し替えた素材は調べ直す
        self._short: set[str] = set()

    @property
    def store(self) -> ProxyStore:
        return self._store

    @property
    def encoder_priority(self) -> str:
        """どの作り手のエンコーダを先に試すか"""
        return self._encoder_priority

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
        """控えを作るよう予約する すでにあるもの・作っている最中のものは何もしない

        小さい素材は、裏のスレッドでキーフレームの間隔を調べてから決める（#430）
        読むのは頭と真ん中の十数秒ずつのパケットだけ（:func:`keyframe_gap`）でも、画面のスレッドで
        読むと素材を読み込むたびに手が止まる
        """
        check = needs_keyframe_check(media)
        if check and not self._store.long_gop:
            return
        if not check and not is_worth_proxying(media):
            return
        if self._store.find(media) is not None:
            return
        # 鍵はファイルの大きさと時刻を読んで作るので、錠の外で求める
        key = self._store.key_for(media) if check else None

        with self._lock:
            if self._closed:
                return
            if key is not None and key in self._short:
                return
            if media.id in self._running:
                if media.id in self._cancelled:
                    self._again[media.id] = (media, on_ready, on_progress)
                return
            self._cancelled.discard(media.id)
            self._running.add(media.id)
            self._begin(media, counted=not check)
        self._launch(media, on_ready, on_progress)

    def _begin(self, media: MediaItem, *, counted: bool) -> None:
        """変換を数え始める ``_lock`` を持って呼ぶ

        調べてから決める素材（``counted`` が偽）は、作ると決まるまで数えない 数えると、
        間隔の短い 1080p を読み込むたびに「控えを作っています」が出て、何もできずに消える
        """
        if counted:
            self._progress[media.id] = 0.0
            self._board.start(media.id, media.id)

    def _long_enough(self, media: MediaItem) -> bool:
        """キーフレームの間隔を調べて、控えを作るか決める 裏のスレッドで呼ぶ

        要らなければ覚えておく 分からなかった（開けない・キーフレームが無い）ときは
        覚えない 一時的に読めなかっただけなら、次に頼まれたときに調べ直せる
        """
        gap = keyframe_gap(
            media.path,
            media.video_streams[0].index if media.video_streams else None,
            enough=KEYFRAME_GAP_LIMIT,
        )
        if is_worth_proxying(media, gap):
            return True
        if gap is not None:
            key = self._store.key_for(media)
            with self._lock:
                self._short.add(key)
        return False

    def _launch(
        self,
        media: MediaItem,
        on_ready: Callable[[MediaId], None] | None,
        on_progress: Callable[[MediaId], None] | None,
    ) -> None:
        """``_running`` に載せた変換を投げる 載せるのは呼ぶ側（:meth:`request` と作り直し）"""

        def report(value: float) -> None:
            with self._lock:
                self._progress[media.id] = value
            self._board.report(media.id, value)
            if on_progress is not None:
                on_progress(media.id)

        def cancelled() -> bool:
            with self._lock:
                return media.id in self._cancelled

        check = needs_keyframe_check(media)

        def run() -> None:
            made: Path | None = None
            # 止められていたかどうかを、消す前に控えておく 消してから見ると
            # 分からなくなり、閉じたあとに「控えができた」と伝えてしまう
            stopped = False
            failure: str | None = None
            # 調べた結果、控えは要らなかった 失敗ではないので数えずに外す
            skipped = False
            try:
                if check:
                    skipped = cancelled() or not self._long_enough(media)
                    if not skipped:
                        with self._lock:
                            # 調べている間に外された素材は数え始めない 数えると、
                            # 一覧に無い素材の分が全体の割合に一瞬混ざる
                            if media.id not in self._cancelled and not self._closed:
                                self._begin(media, counted=True)
                if not skipped:
                    made = create_proxy(
                        media.path,
                        self._store.prepare(media),
                        height=self._store.height,
                        # 1 本目の映像だけを控えにする 読む側（レンダラ）も
                        # 1 本目を指すクリップにしか渡さない
                        stream_index=media.video_streams[0].index if media.video_streams else None,
                        encoder_priority=self._encoder_priority,
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
                    self._cancelled.discard(media.id)
                    self._progress.pop(media.id, None)
                    again = self._again.pop(media.id, None)
                    # 作り直すなら ``_running`` に載せたまま次へ渡す 1 度外してから頼み直すと、
                    # その間に素材を外された（forget）とき、forget は旧い変換も次の変換も
                    # 見つけられず、外した素材の控えを作り始める（解析と同じ）
                    if again is None:
                        self._running.discard(media.id)
            if stopped or skipped:
                self._board.drop(media.id)
            elif made is None:
                # 開けない・映像の大きさが分からない素材は None で返る 何も出さないと、
                # プレビューが重いまま控えを待ち続けることになる
                self._board.finish(media.id, failure or "控えを作れなかった")
            else:
                self._board.finish(media.id)
                # 前の控えを開いたままの読み手がいると Windows では消せない 残っても
                # 読まれないだけなので、できた控えの知らせは止めない
                with contextlib.suppress(OSError):
                    self._store.discard_older(media)
            # 止められていたなら伝えない 伝えると、窓を閉じている最中や
            # 控えを切った直後に「控えができた」として描き直しが走る
            if made is not None and not stopped and on_ready is not None:
                on_ready(media.id)
            if again is not None:
                with self._lock:
                    # 間に外されていれば ``_cancelled`` に入っている 次の変換はすぐ止まる
                    self._begin(again[0], counted=not needs_keyframe_check(again[0]))
                self._launch(*again)

        with self._lock:
            # close と同じロックの中で投げる 外で投げると、止めた直後の
            # executor へ投げて RuntimeError になる
            if self._closed:
                self._running.discard(media.id)
                self._cancelled.discard(media.id)
                self._progress.pop(media.id, None)
                self._board.drop(media.id)
                return
            self._executor.submit(run)

    def forget(self, media_id: MediaId) -> None:
        """素材を外したときに、作りかけを止める"""
        with self._lock:
            if media_id in self._running:
                self._cancelled.add(media_id)
            # 外した素材を、取り消し中の変換が終わった後に作り直さない
            self._again.pop(media_id, None)
        self._board.forget(media_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled.update(self._running)
            self._again.clear()
            self._executor.shutdown(wait=False, cancel_futures=True)
