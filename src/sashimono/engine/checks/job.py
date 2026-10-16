"""完成物の検査を裏のスレッドで走らせ、途中を見せ、止められるようにする（#337）

長い作品は全部のコマを描いて混ぜるので、1080p の 10 分で数分かかる 画面のスレッドで
回すと、その間は画面が固まる 走り係（:class:`_Worker`）が自分のレンダラとミキサを持ち、
拾った物を鍵で守った入れ物（:class:`_Progress`）へ積む 画面と AI はそこから写しを読む

- AI は終わるまで待つか（最長の秒を決めて）、待ちきれなければ途中までを受け取る
  （場面の検出 #358 の ``Pending`` と同じ待ち方 待つのは AI のスレッドで、画面は止めない）
- 人はステータスバーの〔止める〕で止められる 窓を閉じても止まる
- 同じ頼み（同じ作品・範囲・しきい値）なら、走っている物か終わった物をそのまま使う
  作品が変わっていれば別の頼みになり、調べ直す 前の結果を今の作品の結果として返さない

GL のコンテキストはプレビューの先読みと同じ作り（サーフェスは画面のスレッドで作り、
コンテキストは走り係のスレッドで作る :class:`~sashimono.engine.render.look_preview.LookWorker`）
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import av.audio.frame
import av.video.frame
import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal

from sashimono.core.model import Project
from sashimono.engine.audio import AudioMixer
from sashimono.engine.checks.overflow import OverflowScan
from sashimono.engine.checks.report import CheckReport, ClipIndex, assemble
from sashimono.engine.checks.scan import (
    CLIPPING,
    LOUDNESS,
    PICTURE_KINDS,
    SOUND_KINDS,
    TEXT_OVERFLOW,
    AudioScan,
    CheckLimits,
    Run,
    VideoScan,
)
from sashimono.engine.decode.probe import media_origin
from sashimono.engine.decode.rational import Ratio, as_fraction
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import DEFAULT_DECODE_THREADS, FrameRenderer, RenderQuality

__all__ = ["ANALYSIS_WIDTH", "CheckJob", "CheckRequest", "CheckState"]

#: 絵を調べる横幅（画素） 黒と止まりは画面全体の明るさで決まり、細かさは要らない
#: 1080p を 1/4（480x270）で描くと、合成と読み戻しが 16 分の 1 になる 時間のほとんどは
#: 素材のデコードで、これ以上小さくしても速くならない
ANALYSIS_WIDTH = 480
#: 画面へ進み具合を知らせる間（秒） 1 コマごとに知らせると、短いコマの作品で画面の
#: スレッドが知らせを捌くだけで忙しくなる
_NOTIFY_SECONDS = 0.2
#: 符号化しても標本の値が変わらない音のコーデック（名前の頭） これ以外は音割れを取りこぼしうる
_LOSSLESS_AUDIO = ("pcm_", "flac", "alac", "wavpack", "tta")


@dataclass(frozen=True, slots=True)
class CheckRequest:
    """検査の頼み 同じ頼みなら同じ結果を使い回す（比べるのは中身）

    タイムラインなら ``project`` と範囲、書き出した動画なら ``path`` と ``stamp``
    """

    limits: CheckLimits
    project: Project | None = None
    start: int = 0
    end: int = 0
    path: Path | None = None
    #: ファイルの（大きさ, 更新時刻） 書き出し直したファイルを前の結果で答えない
    stamp: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class CheckState:
    """検査の今"""

    request: CheckRequest | None = None
    report: CheckReport | None = None
    running: bool = False
    failure: str | None = None
    #: 人が〔止める〕で止めた（窓を閉じたときも）
    stopped: bool = False
    #: 進み具合（0〜1）
    progress: float = 0.0


class _Progress:
    """走り係と画面のスレッドが分け合う入れ物 中身はすべて ``lock`` の中で触る"""

    def __init__(self, request: CheckRequest, notify: Callable[[], None]) -> None:
        self.request = request
        self.lock = threading.Condition()
        self.cancel = threading.Event()
        self._notify = notify
        self._notified = 0.0
        self.video: VideoScan | None = None
        self.audio: AudioScan | None = None
        self.overflow: OverflowScan | None = None
        self.index: ClipIndex | None = None
        self.start = request.start
        self.end = request.end
        self.fps = 30.0
        self.analyzed = request.start
        self.length_seconds: float | None = None
        self.notes: list[str] = []
        #: 頼まれたのに調べられなかった種類 返事では調べた種類から外し、別に返す
        #: 外さないと、調べていない種類を「問題は無かった」と読ませる（GPU の無い機械の黒など）
        self.skipped: set[str] = set()
        self.running = True
        self.failure: str | None = None
        self.stopped = False

    def skip(self, kinds: frozenset[str], note: str) -> None:
        """頼まれた種類のうち ``kinds`` を調べられなかったと記す 頼まれていなければ何もしない"""
        missed = kinds & self.request.limits.kinds
        if missed:
            self.skipped |= missed
            self.notes.append(note)

    def skip_pictures(self, reason: object) -> None:
        """GPU が使えず、絵を描く種類（黒・止まり・字幕）を調べられない"""
        self.skip(
            PICTURE_KINDS | {TEXT_OVERFLOW},
            f"GPU が使えないので、絵と字幕は調べていません（{reason}）",
        )

    def touch(self, *, force: bool = False) -> None:
        """画面へ知らせる 間引いて知らせる（:data:`_NOTIFY_SECONDS`）"""
        now = time.monotonic()
        if force or now - self._notified >= _NOTIFY_SECONDS:
            self._notified = now
            self._notify()

    def report(self) -> CheckReport | None:
        """拾った所までの結果 鍵を持って呼ぶこと"""
        final = not self.running and self.failure is None and not self.stopped
        runs: list[Run] = []
        if self.video is not None:
            runs += self.video.runs(final=final)
        if self.audio is not None:
            runs += self.audio.runs(final=final)
        if self.overflow is not None:
            runs += self.overflow.runs()
        loudness = self.audio.loudness() if self.audio is not None else None
        skipped = set(self.skipped)
        notes = list(self.notes)
        limits = self.request.limits
        if final and LOUDNESS in limits.kinds and LOUDNESS not in skipped and loudness is None:
            # 400 ms に満たない音・-70 LUFS より静かな音だけの作品は、規格の決まりで値が出ない
            # 調べたことにすると「音量は基準どおり」と読まれる 測れなかったと分けて返す
            skipped.add(LOUDNESS)
            notes.append(
                "音量（ラウドネス）は測れませんでした 音が無いか、静かすぎるか（-70 LUFS 未満）、"
                "短すぎます（400 ms 未満）"
            )
        return assemble(
            runs,
            start=self.start,
            end=max(self.end, self.analyzed),
            analyzed=self.analyzed,
            fps=self.fps,
            complete=final,
            limits=self.request.limits,
            index=self.index,
            loudness=loudness,
            peak=self.audio.peak if self.audio is not None else None,
            length_seconds=self.length_seconds,
            notes=tuple(notes),
            skipped=frozenset(skipped),
        )

    def fraction(self) -> float:
        span = self.end - self.start
        if span <= 0:
            return 1.0 if not self.running else 0.0
        return max(0.0, min(1.0, (self.analyzed - self.start) / span))


class CheckJob(QObject):
    """検査を 1 本ずつ走らせる係 画面のスレッドで作り、画面のスレッドから頼む

    :meth:`wait` だけはどのスレッドから呼んでもよい（AI のスレッドが待つ）
    """

    #: 進み具合か様子が変わった（走り係のスレッドから出る 受け手には Qt が順に渡す）
    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._progress: _Progress | None = None
        self._worker: _Worker | None = None
        self._context: OffscreenGLContext | None = None

    def start(
        self,
        request: CheckRequest,
        *,
        decode_threads: int = DEFAULT_DECODE_THREADS,
        smooth_history: bool = True,
    ) -> None:
        """``request`` を調べ始める 同じ頼みが走っているか、最後まで終わっていれば何もしない

        止めた物・失敗した物は、同じ頼みでも調べ直す（直してから頼み直すことがあるため）
        """
        current = self._progress
        if current is not None and current.request == request:
            with current.lock:
                if current.running or (current.failure is None and not current.stopped):
                    return
        self._halt()
        progress = _Progress(request, self.changed.emit)
        context: OffscreenGLContext | None = None
        limits = request.limits
        if request.project is not None and (limits.needs_picture or TEXT_OVERFLOW in limits.kinds):
            try:
                context = OffscreenGLContext(deferred=True)
            except GLContextError as exc:
                # GPU の無い機械でも音は調べられる 絵の検査だけ外したと結果に書く
                progress.skip_pictures(exc)
        worker = _Worker(
            request,
            progress,
            context,
            decode_threads=decode_threads,
            smooth_history=smooth_history,
        )
        worker.setObjectName("sashimono-check")
        if context is not None:
            context.move_to_thread(worker)
        worker.finished.connect(self._reap)
        self._progress, self._worker, self._context = progress, worker, context
        worker.start()
        self.changed.emit()

    def state(self) -> CheckState:
        progress = self._progress
        if progress is None:
            return CheckState()
        with progress.lock:
            return CheckState(
                request=progress.request,
                report=progress.report(),
                running=progress.running,
                failure=progress.failure,
                stopped=progress.stopped,
                progress=progress.fraction(),
            )

    def waiter(self) -> Callable[[float], bool]:
        """今の検査が終わるまで、最長で引数の秒だけ待つ関数 終わっていれば真

        頼み直しで入れ物が替わっても、受け取った時の検査を待つ（替わった後の検査を待つと、
        AI が頼んでいない検査の終わりまで待たされる）
        """
        progress = self._progress

        def wait(seconds: float) -> bool:
            if progress is None:
                return True
            with progress.lock:
                return progress.lock.wait_for(lambda: not progress.running, max(0.0, seconds))

        return wait

    @property
    def running(self) -> bool:
        progress = self._progress
        if progress is None:
            return False
        with progress.lock:
            return progress.running

    def stop(self) -> None:
        """止めるよう頼む 待たない（走り係は今のコマを終えて止まる）"""
        progress = self._progress
        if progress is not None:
            progress.cancel.set()

    def close(self) -> None:
        """止まるまで待ち、コンテキストを手放す 窓を閉じるときに呼ぶ"""
        self._halt()

    def _halt(self) -> None:
        worker = self._worker
        if worker is not None:
            if self._progress is not None:
                self._progress.cancel.set()
            worker.wait()
            self._worker = None
        self._release()

    def _reap(self) -> None:
        """走り係が終わった 画面のスレッドへ戻ったコンテキストを手放す"""
        worker = self._worker
        if worker is not None and worker.isFinished():
            worker.wait()
            self._worker = None
            self._release()

    def _release(self) -> None:
        if self._context is not None:
            self._context.release()
            self._context = None


class _Worker(QThread):
    """検査の走り係 レンダラとミキサは、作るのも捨てるのもこのスレッド"""

    def __init__(
        self,
        request: CheckRequest,
        progress: _Progress,
        context: OffscreenGLContext | None,
        *,
        decode_threads: int,
        smooth_history: bool,
    ) -> None:
        super().__init__()
        self._request = request
        self._progress = progress
        self._context = context
        self._decode_threads = decode_threads
        self._smooth_history = smooth_history

    def run(self) -> None:
        progress = self._progress
        failure: str | None = None
        context = self._context
        entered = False
        try:
            if context is not None:
                try:
                    context.complete()
                    context.__enter__()
                    entered = True
                except GLContextError as exc:
                    with progress.lock:
                        progress.skip_pictures(exc)
            if self._request.path is not None:
                _check_file(self._request.path, progress)
            else:
                _check_timeline(
                    self._request,
                    progress,
                    context if entered else None,
                    decode_threads=self._decode_threads,
                    smooth_history=self._smooth_history,
                )
        except Exception as exc:
            # 黙って消えない AI にも人にも、なぜ調べられなかったかを返す
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if entered and context is not None:
                context.__exit__(None, None, None)
            application = QCoreApplication.instance()
            if context is not None and application is not None:
                # 画面のスレッドへ返す 返さないと、捨てるときに別のスレッドに付いたままの
                # コンテキストを触る
                context.move_to_thread(application.thread())
            with progress.lock:
                if progress.audio is not None:
                    with contextlib.suppress(Exception):
                        progress.audio.finish()
                progress.running = False
                progress.failure = failure
                progress.stopped = failure is None and progress.cancel.is_set()
                progress.lock.notify_all()
            progress.touch(force=True)


def _luma(image: np.ndarray) -> np.ndarray:
    """合成した sRGB の絵（RGBA uint8）の輝度（0〜1） 重みは BT.709（書き出しと同じ行列）"""
    rgb = image[..., :3].astype(np.float32)
    return np.asarray((rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0)


def _check_timeline(
    request: CheckRequest,
    progress: _Progress,
    context: OffscreenGLContext | None,
    *,
    decode_threads: int,
    smooth_history: bool,
) -> None:
    project = request.project
    if project is None:
        return
    limits = request.limits
    rate = project.rate
    fps = float(rate.fps)
    width, _height = project.settings.resolution
    overflow = (
        OverflowScan(project, request.start, request.end, limits)
        if TEXT_OVERFLOW in limits.kinds and context is not None
        else None
    )
    renderer: FrameRenderer | None = None
    mixer: AudioMixer | None = None
    try:
        if context is not None and (limits.needs_picture or (overflow and overflow.frames)):
            renderer = FrameRenderer(
                project,
                context=context,
                quality=RenderQuality(max(1, width // ANALYSIS_WIDTH)),
                decode_threads=decode_threads,
            )
        if limits.needs_sound:
            mixer = AudioMixer(project, smooth_history=smooth_history)
        with progress.lock:
            progress.fps = fps
            progress.index = ClipIndex(project)
            progress.length_seconds = (request.end - request.start) / fps
            progress.video = (
                VideoScan(limits, fps) if renderer is not None and limits.needs_picture else None
            )
            progress.audio = (
                AudioScan(
                    limits,
                    sample_rate=mixer.sample_rate,
                    channels=mixer.channels,
                    fps=fps,
                    origin=request.start,
                )
                if mixer is not None
                else None
            )
            progress.overflow = overflow
        canvas = renderer.size if renderer is not None else (0, 0)
        frames: range | list[int] = range(request.start, request.end)
        if progress.video is None and progress.audio is None:
            # 字幕だけを調べる 全部のコマを回さず、測るコマだけ見る
            frames = sorted(overflow.frames) if overflow is not None else []
        for frame in frames:
            if progress.cancel.is_set():
                return
            luma = (
                _luma(renderer.render(frame))
                if renderer is not None and progress.video is not None
                else None
            )
            sound = mixer.render_frames(frame, 1) if mixer is not None else None
            with progress.lock:
                if luma is not None and progress.video is not None:
                    progress.video.add(frame, luma)
                if sound is not None and progress.audio is not None:
                    progress.audio.add(sound)
                if overflow is not None and renderer is not None:
                    overflow.measure(renderer, frame, canvas)
                progress.analyzed = frame + 1
            progress.touch()
        with progress.lock:
            progress.analyzed = request.end
    finally:
        if renderer is not None:
            renderer.close()
        if mixer is not None:
            mixer.close()


def _check_file(path: Path, progress: _Progress) -> None:
    """書き出した動画を頭から読んで調べる 字幕は字の形を持たないので調べない"""
    limits = progress.request.limits
    with av.open(str(path)) as container:
        videos = container.streams.video
        audios = container.streams.audio
        rate = videos[0].average_rate if videos else None
        fps = float(rate) if rate else 30.0
        duration = float(container.duration) / 1_000_000 if container.duration else 0.0
        video = videos[0] if videos and limits.needs_picture else None
        audio = audios[0] if audios and limits.needs_sound else None
        with progress.lock:
            progress.fps = fps
            progress.start = 0
            progress.end = max(1, round(duration * fps))
            progress.analyzed = 0
            progress.video = VideoScan(limits, fps) if video is not None else None
            progress.skip(
                frozenset({TEXT_OVERFLOW}),
                "書き出した動画からは字幕のはみ出しを調べられません（字の形が残らない）"
                " タイムラインを調べると分かります",
            )
            if audio is None:
                progress.skip(SOUND_KINDS, "この動画には音がありません")
            if video is None:
                progress.skip(PICTURE_KINDS, "この動画には絵がありません")
            if (
                CLIPPING in limits.kinds
                and audio is not None
                and not audio.codec_context.name.startswith(_LOSSLESS_AUDIO)
            ):
                # AAC などは頭打ちの平らな所を崩す（ふくらんだり 0.96 ほどまで下がったりする）
                # 崩れた所は続けて最大に届かないので、音割れの一部しか拾えない 拾えなかった物を
                # 「無い」と読ませない 書き出しは混ぜた音を ±1 で切ってから符号化するので、
                # 符号化する前の音（タイムライン）を調べれば確かに拾える
                progress.notes.append(
                    f"音が圧縮されている（{audio.codec_context.name}）ので、音割れは一部しか"
                    "拾えないことがあります 確かめるにはタイムラインを調べてください"
                )
        resampler = av.AudioResampler(format="fltp")
        streams = [stream for stream in (video, audio) if stream is not None]
        if not streams:
            # 読む流れが無い（字幕だけ・音の無い動画で音だけを頼んだなど） demux へ空で渡すと
            # 全部の流れを読み、長い動画を頭から終わりまでデコードして、頼んでいない音の
            # 大きさまで返す 長さは入れ物が名乗る長さで足りる
            with progress.lock:
                progress.length_seconds = duration
                progress.analyzed = progress.end
            return
        # 時刻は素材の原点から数える（デコーダと同じ media_origin） 映像は PTS からコマの番号を、
        # 音は PTS からサンプルの位置を出す 連番と隙間の無い並びとして渡すと、音が遅れて始まる
        # ファイルの頭の無音や、VFR で長く出るコマ（止まった絵）を見落とし、時刻もずれる
        origin = media_origin(container)
        # PyAV 19 から average_rate は AVRational（Fraction に直に渡せない） デコーダと同じく直す
        frame_rate = as_fraction(rate) if rate else Fraction(30)
        pictures = 0
        last_luma: np.ndarray | None = None
        size: tuple[int, int] | None = None
        for packet in container.demux(*streams):
            if progress.cancel.is_set():
                return
            for decoded in packet.decode():
                if isinstance(decoded, av.video.frame.VideoFrame):
                    when = _seconds(decoded.pts, decoded.time_base, origin)
                    index = pictures if when is None else math.floor(when * frame_rate + 0.5)
                    if index < pictures:
                        # 同じコマに 2 枚（VFR の詰まった所・並べ替えの残り）は前の 1 枚だけを数える
                        continue
                    if size is None:
                        scale = min(1.0, ANALYSIS_WIDTH / max(decoded.width, 1))
                        size = (
                            max(2, round(decoded.width * scale) // 2 * 2),
                            max(2, round(decoded.height * scale) // 2 * 2),
                        )
                    gray = decoded.reformat(width=size[0], height=size[1], format="gray")
                    luma = gray.to_ndarray().astype(np.float32) / 255.0
                    with progress.lock:
                        scan = progress.video
                        if scan is not None:
                            # 欠けたコマは、再生では前の絵が出続ける 頭より前に絵が無ければ黒
                            held = last_luma if last_luma is not None else np.zeros_like(luma)
                            for missing in range(pictures, index):
                                scan.add(missing, held)
                            scan.add(index, luma)
                    last_luma = luma
                    pictures = index + 1
                elif isinstance(decoded, av.audio.frame.AudioFrame):
                    when = _seconds(decoded.pts, decoded.time_base, origin)
                    _feed_audio(progress, resampler.resample(decoded), fps, when)
            with progress.lock:
                heard = progress.audio.seconds * fps if progress.audio is not None else 0
                progress.analyzed = max(pictures, round(heard))
            progress.touch()
        _feed_audio(progress, resampler.resample(None), fps, None)
        with progress.lock:
            if audio is not None and progress.audio is None:
                # 音の流れはあるのに 1 サンプルも出なかった 測っていない物を「無音は無い」と言わない
                progress.skip(SOUND_KINDS, "音の流れから音を読めませんでした")
            if video is not None and pictures == 0:
                progress.skip(PICTURE_KINDS, "映像の流れから絵を読めませんでした")
            if progress.audio is not None and pictures / fps > progress.audio.seconds:
                # 音が映像より先に終わる所は、再生では無音になる 無音として数える
                _pad_silence(progress.audio, round(pictures / fps * progress.audio.rate))
            heard = progress.audio.seconds if progress.audio is not None else 0.0
            length = max(pictures / fps, heard)
            progress.length_seconds = length
            progress.end = max(1, round(length * fps))
            progress.analyzed = progress.end


def _seconds(pts: int | None, time_base: Ratio | None, origin: Fraction) -> Fraction | None:
    """フレームの PTS を、素材の原点からの秒にする 分からなければ ``None``"""
    if pts is None or time_base is None:
        return None
    return pts * as_fraction(time_base) - origin


#: 音の位置のずれを詰めずに許す長さ（秒） 符号化の丸めで PTS は 1 サンプルほど揺れる
#: 揺れのたびに無音を足したり削ったりすると、音割れの続きがそこで切れる
_AUDIO_SLACK = 0.002


def _pad_silence(scan: AudioScan, until: int) -> None:
    """``until`` サンプル目まで無音を足す 長い隙間は 1 秒ずつ渡す（一度に大きな配列を作らない）"""
    step = scan.rate
    while scan.samples < until:
        count = min(step, until - scan.samples)
        scan.add(np.zeros((count, scan.channels), dtype=np.float32))


def _feed_audio(
    progress: _Progress,
    frames: list[av.audio.frame.AudioFrame],
    fps: float,
    when: Fraction | None,
) -> None:
    """リサンプルした音を渡す ``when`` は元のフレームの頭の時刻（原点から 秒）

    PTS が今までに渡した長さより後ろなら、間を無音で埋める（遅れて始まる音・途中の抜け）
    前なら重なった分を捨てる（原点より前の前置き・重なった PTS）
    """
    for frame in frames:
        planar = frame.to_ndarray()
        with progress.lock:
            if progress.audio is None:
                progress.audio = AudioScan(
                    progress.request.limits,
                    sample_rate=frame.sample_rate,
                    channels=planar.shape[0],
                    fps=fps,
                )
            scan = progress.audio
            block = planar.T
            if when is not None:
                at = math.floor(when * scan.rate)
                slack = round(scan.rate * _AUDIO_SLACK)
                if at > scan.samples + slack:
                    _pad_silence(scan, at)
                elif at < scan.samples - slack:
                    block = block[scan.samples - at :]
                # 1 つの元のフレームから出た塊は続けて並べる 2 つ目からは位置を見ない
                when = None
            scan.add(block)
