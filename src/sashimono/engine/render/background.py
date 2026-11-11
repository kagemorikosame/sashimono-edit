"""先読みを別のスレッドで描く（Issue #56 の 3）

画面と同じスレッドで先読みすると、1 コマ描く間はイベントループが止まる
``tools/bench_prefetch.py`` で測った、先読みで貯めている間の 5ms おきの見張りの遅れ
（95 パーセンタイル RTX 5060 Ti 素材は testsrc2 を libx264 で焼いた物を層ごとに重ねる）

======================================  ==================  ==============
素材                                    画面と同じスレッド  別のスレッド
======================================  ==================  ==============
1920x1080 を 1 枚                       1.9 ms              0.7 ms
1920x1080 を 3 枚重ね                   7.5 ms              0.7 ms
1920x1080 を 3 枚 + ぼかし・発光        17.8 ms             0.7 ms
1920x1080 を 6 枚重ね                   21.8 ms             0.7 ms
3840x2160 を 2 枚重ね（1/2 画質）       16.0 ms             0.9 ms
3840x2160 を 3 枚 + ぼかし・発光        56.1 ms             0.7 ms
======================================  ==================  ==============

1 コマの合成は分けられない（途中で返すと、描きかけの絵が残る）ので、画面の側で
刻んでも 1 コマぶんは止まる 別のスレッドへ出すと、止まるのは Python の切り替え
（既定 5ms）の間だけになる デコードと numpy の計算は GIL を手放すので、その間は
画面の側が走れる

作り

- 走り係は**自分の** :class:`FrameRenderer`（デコーダ・合成先・シェーダ）と、
  画面のコンテキストと共有した自分の GL コンテキストを持つ 画面の側のレンダラは
  一切触らない 同じ（素材, ストリーム）のデコーダを 2 スレッドから触ると壊れる
  （PR #98）が、デコーダはレンダラごとに別に開くので、そもそも同じ物を持たない
- 貯めた絵の描画先（テクスチャとフレームバッファ）は走り係のコンテキストで作り、
  走り係のコンテキストで捨てる テクスチャは共有されるが、フレームバッファは
  コンテキストごとの物なので、ほかのスレッドから捨てられない
- 画面の側へは「出してよい絵」の一覧（:class:`_Shelf`）だけを渡す 一覧は鍵で守り、
  画面の側は鍵を持ったまま出して ``glFinish`` まで待つ 走り係は描画先を使い回す前に
  鍵を取って一覧から外すので、画面の側が出している最中の絵を書き換えることはない
- 走り係は描き終えたら ``glFinish`` してから一覧へ載せる 共有した物の書き換えは、
  書いた側の命令が終わるまでほかのコンテキストから見える保証が無い
- 編集で絵が変わったら、画面の側がその場で一覧から外し、代の番号を上げる
  走り係は描き始めたときの代と一覧の代が違えば、描いた絵を載せずに捨てる
  古いプロジェクトで描いた絵を、編集の後に出さないため
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from OpenGL import GL
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal

from sashimono.core.model import MediaId, Project
from sashimono.engine.gpu import OffscreenGLContext
from sashimono.engine.render.backdrop import BLACK, Backdrop
from sashimono.engine.render.invalidate import Invalidation
from sashimono.engine.render.prefetch import CacheSurface, PreviewCache
from sashimono.engine.render.renderer import FrameRenderer, RenderQuality

if TYPE_CHECKING:
    from PySide6.QtGui import QOpenGLContext

    from sashimono.engine.cache.proxy import ProxyStore
    from sashimono.engine.gpu import Compositor

__all__ = ["MEASURED_PREFETCH_STALL_MS", "WAIT_FOR_WORKER_S", "BackgroundPrefetch"]

#: 先読みで貯めている間に、操作を受け付けられなかった長さ（ミリ秒 95 パーセンタイル）
#: 画面と同じスレッドで描いた場合と、別のスレッドで描いた場合
#: 3840x2160 を 3 枚重ねてぼかしと発光を積んだ所（RTX 5060 Ti）
#: 設定の画面に出す 測り直したらここだけ直す
MEASURED_PREFETCH_STALL_MS = (56.1, 0.7)

#: 画面の側が要るコマを走り係がちょうど描いている最中なら、描き終わるまで待つ上限（秒）
#:
#: 待たずに画面の側でも描くと、同じコマを 2 つのスレッドで描くことになり、GIL と GPU を
#: 取り合って両方が遅くなる 貯めた所の外で再生ヘッドを送り続けると、走り係が追い付けない
#: まま画面の側も遅くなり、先読みを切ったときより重くなった（3840x2160 を 2 枚重ねで
#: 1 コマ 19ms の所が 34ms） 走り係は描き始めているぶん、画面の側で描き直すより早く終わる
#:
#: **待つ間はクリックもキーも受け付けない** なので上限は 1 コマの予算に合わせる
#: 測った中で一番重い 1 コマ（3840x2160 を 3 枚 + ぼかし・発光）が 57ms、画面の
#: スレッドの先読みをやめる目安（``ui/preview.py`` の ``SLOW_FRAME_MS``）が 200ms
#: その 200ms に揃える 60ms の所なら 3 倍の余裕で待ち切れ、それより重いコマや
#: 固まった走り係は待たずに画面の側で描く 2 秒のように長く取ると、走り係が
#: 固まったときに操作が 2 秒止まる
WAIT_FOR_WORKER_S = 0.2


@dataclass(slots=True)
class _Shelf:
    """画面の側へ出してよい絵の一覧 必ず ``lock`` を取ってから触る"""

    lock: threading.Lock = field(default_factory=threading.Lock)
    #: 一覧か「いま描いているコマ」が変わったときに起こす 鍵は ``lock`` と同じ
    changed: threading.Condition = field(init=False)
    #: 走り係がいま描いているコマ 描いていなければ None
    rendering: int | None = None
    #: 編集で絵が変わるたびに上がる番号 走り係が描き始めたときの番号と
    #: 違えば、その絵は古いプロジェクトで描いた物
    generation: int = 0
    frames: dict[int, CacheSurface] = field(default_factory=dict)
    #: 走り係のレンダラが捨てた控えの素材 呼ぶ側が作り直しを頼む
    discarded: set[MediaId] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.changed = threading.Condition(self.lock)


#: 走り係への頼み （代の番号, 名前, 引数）
_Command = tuple[int, str, tuple[object, ...]]


class BackgroundPrefetch(QObject):
    """先読みを別のスレッドで描き、画面の側へ出せるようにする

    作るのも呼ぶのも**画面のスレッドから** 作れなければ
    :class:`~sashimono.engine.gpu.GLContextError` を投げるので、呼ぶ側は
    画面のスレッドで貯める作り（:class:`PreviewCache` を直に回す）へ戻る
    """

    #: 1 コマ描けなかった 次に何か変わったら、また試す 引数は出した走り係と理由
    #: 走り係を引数に入れるのは、止めた後に届いた合図を呼ぶ側が見分けるため
    #: 合図は画面のスレッドのイベントループを通って遅れて届くので、閉じて
    #: 作り直した後に前の走り係の「止まった」が届くことがある
    failed = Signal(object, str)
    #: 走り係がもう続けられない 呼ぶ側は :meth:`close` して、画面のスレッドへ戻す
    ended = Signal(object, str)

    def __init__(
        self,
        project: Project,
        *,
        share: QOpenGLContext,
        quality: RenderQuality,
        proxies: ProxyStore | None,
        decode_threads: int,
        budget_bytes: int,
        playhead: int,
        backdrop: Backdrop = BLACK,
    ) -> None:
        super().__init__()
        self._shelf = _Shelf()
        self._queue: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._closed = False
        # サーフェスは GUI のスレッドで作る決まりなので、コンテキストごとここで作る
        # 作れなければ投げて、呼ぶ側が画面のスレッドの先読みへ戻る
        self._context = OffscreenGLContext(share=share)
        try:
            self._worker = _Worker(
                self._shelf,
                self._queue,
                self._context,
                project=project,
                quality=quality,
                proxies=proxies,
                decode_threads=decode_threads,
                budget_bytes=budget_bytes,
                playhead=playhead,
                backdrop=backdrop,
                failed=lambda reason: self.failed.emit(self, reason),
                ended=lambda reason: self.ended.emit(self, reason),
            )
            self._worker.setObjectName("sashimono-prefetch")
            self._context.move_to_thread(self._worker)
            self._worker.start()
        except BaseException:
            # 走り係を出せなかった 作ったサーフェスを残すと、作り直すたびに増える
            self._context.release()
            raise

    @property
    def running(self) -> bool:
        return self._worker.isRunning()

    @property
    def cached(self) -> frozenset[int]:
        """画面の側へ出せる絵のフレーム番号"""
        with self._shelf.lock:
            return frozenset(self._shelf.frames)

    def worker_thread(self) -> QThread:
        """走り係のスレッド 試験で、誰がどのスレッドで触ったかを見るため"""
        return self._worker

    # --- 画面の側から頼む ---

    def set_project(self, project: Project, invalidation: Invalidation) -> None:
        """プロジェクトを差し替える ``invalidation`` の範囲の絵はその場で出さなくなる"""
        self._send("project", project, invalidation, invalidation=invalidation)

    def invalidate(self, invalidation: Invalidation) -> None:
        """変わった範囲の絵を捨てる GL は触らない（見張りのタイマーから呼ばれる）"""
        self._send("invalidate", invalidation, invalidation=invalidation)

    def set_quality(self, quality: RenderQuality) -> None:
        # 大きさが変わると、取ってある絵は全部出せない
        self._send("quality", quality, invalidation=Invalidation.all())

    def set_proxies(self, proxies: ProxyStore | None) -> None:
        # 読む元が変われば絵も変わる
        self._send("proxies", proxies, invalidation=Invalidation.all())

    def reopen_sources(self, media_ids: Collection[MediaId] | None = None) -> None:
        # 控えができた・壊れていて元へ戻した どちらも開き直した素材の絵が変わる
        self._send("reopen", media_ids, invalidation=Invalidation.all())

    def set_budget(self, budget_bytes: int, playhead: int) -> None:
        self._send("budget", budget_bytes, playhead)

    def set_backdrop(self, backdrop: Backdrop) -> None:
        """何も描いていない所に敷く地を変える（#456）

        貯めた絵には地が焼き込まれているので、全部をその場で出さなくする
        外さないと、走り係が描き直すまでの間、前の地の絵が出る
        """
        self._send("backdrop", backdrop, invalidation=Invalidation.all())

    def set_decode_threads(self, threads: int) -> None:
        self._send("threads", threads)

    def set_playhead(self, frame: int) -> None:
        """再生ヘッドを動かす 貯め終えて休んでいた走り係もここで起きる"""
        self._send("playhead", frame)

    def set_paused(self, paused: bool) -> None:
        """再生中は止める 出す側と同じ GPU を奪い合うと、いま出すコマが遅れる

        描いている最中の 1 コマは描き終えるまで止まらない 合成は途中で抜けられない
        """
        self._send("pause", paused)

    def take_discarded(self) -> set[MediaId]:
        with self._shelf.lock:
            found, self._shelf.discarded = self._shelf.discarded, set()
        return found

    def show(
        self,
        frame: int,
        compositor: Compositor,
        framebuffer: int,
        viewport: tuple[int, int, int, int],
    ) -> bool:
        """貯めてあれば ``frame`` を出して ``True`` **画面のコンテキストが current な所で呼ぶこと**

        鍵を持ったまま出して、GPU が読み終えるまで待つ 待たずに鍵を放すと、
        走り係がその描画先を使い回して書き換え、画面に別のコマが混ざることがある

        走り係がちょうどそのコマを描いているなら、描き終わるまで待ってから出す
        （:data:`WAIT_FOR_WORKER_S`）
        """
        shelf = self._shelf
        with shelf.changed:
            if frame not in shelf.frames and shelf.rendering == frame:
                shelf.changed.wait_for(
                    lambda: frame in shelf.frames or shelf.rendering != frame,
                    timeout=WAIT_FOR_WORKER_S,
                )
            surface = shelf.frames.get(frame)
            if surface is None:
                return False
            if (surface.width, surface.height) != (compositor.width, compositor.height):
                # 画質を変えた直後 大きさの違う絵を出すと、余白の付き方が変わって跳ねる
                return False
            compositor.show(surface.color, framebuffer, viewport)
            GL.glFinish()
            return True

    def close(self) -> None:
        """走り係を止めて、描画先とレンダラを手放させる 何度呼んでもよい

        走り係が描いている最中の 1 コマは描き終えるまで待つ 途中で放り出すと、
        デコーダと GL の資源を掴んだままスレッドが消える
        """
        if self._closed:
            return
        self._closed = True
        self._send("stop")
        self._worker.wait()
        with self._shelf.lock:
            self._shelf.frames.clear()
        # コンテキストは走り係が抜ける前にこちらのスレッドへ戻している
        self._context.release()

    def _send(self, name: str, *args: object, invalidation: Invalidation | None = None) -> None:
        """頼みを積む 絵が変わる頼みなら、代を上げて一覧から外すのと同じ鍵の中で積む

        鍵の外で積むと、走り係が「上がった代」を読んだのに頼みはまだ届いていない
        瞬間ができる そこで描いた古い絵が、新しい代の絵として載ってしまう
        """
        if self._closed and name != "stop":
            return
        shelf = self._shelf
        with shelf.lock:
            if invalidation:
                shelf.generation += 1
                for frame in [f for f in shelf.frames if invalidation.contains(f)]:
                    del shelf.frames[frame]
            self._queue.put((shelf.generation, name, args))


class _Worker(QThread):
    """先読みの走り係 レンダラと描画先は、作るのも捨てるのもこのスレッド"""

    def __init__(
        self,
        shelf: _Shelf,
        commands: queue.SimpleQueue[_Command],
        context: OffscreenGLContext,
        *,
        project: Project,
        quality: RenderQuality,
        proxies: ProxyStore | None,
        decode_threads: int,
        budget_bytes: int,
        playhead: int,
        backdrop: Backdrop,
        failed: Callable[[str], None],
        ended: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._shelf = shelf
        self._commands = commands
        self._context = context
        self._project = project
        self._quality = quality
        self._proxies = proxies
        self._decode_threads = decode_threads
        self._budget = budget_bytes
        self._playhead = playhead
        self._backdrop = backdrop
        self._failed = failed
        self._ended = ended
        #: いま描いている頼みの代 一覧の代と同じときだけ載せる
        self._generation = 0
        self._paused = False
        #: 描けなくなった 次に何か頼まれるまで描かない（毎回同じ所で失敗し続けない）
        self._halted = False
        #: 貯めきった 次に何か頼まれるまで休む
        self._idle = False

    def run(self) -> None:
        renderer: FrameRenderer | None = None
        cache: PreviewCache | None = None
        entered = False
        try:
            # 走り係の間ずっと current にしておく 1 コマごとに付け外しすると、
            # そのたびにドライバの切り替えが入る
            self._context.__enter__()
            entered = True
            renderer = FrameRenderer(
                self._project,
                context=self._context,
                quality=self._quality,
                proxies=self._proxies,
                decode_threads=self._decode_threads,
            )
            cache = PreviewCache(
                renderer,
                budget_bytes=self._budget,
                forget=self._unpublish,
                backdrop=self._backdrop,
            )
            self._loop(renderer, cache)
        except BaseException as exc:
            # **黙って消えない** 走り係が消えたのに気付かないと、先読みが
            # 効かないまま理由も分からない 呼ぶ側は画面のスレッドへ戻す
            self._ended(f"別のスレッドの先読みを止めた: {exc}")
        finally:
            self._tear_down(renderer, cache, entered)

    def _tear_down(
        self, renderer: FrameRenderer | None, cache: PreviewCache | None, entered: bool
    ) -> None:
        """描画先とレンダラを、作ったこのスレッドで手放す

        1 つ失敗しても残りは必ず手放す 途中で抜けると、デコーダがファイルを
        掴んだまま、GPU のメモリも返らない
        """
        for release in (
            cache.release if cache is not None else None,
            renderer.close if renderer is not None else None,
        ):
            if release is None:
                continue
            try:
                release()
            except Exception as exc:
                self._ended(f"先読みの後片付けに失敗した: {exc}")
        with self._shelf.lock:
            self._shelf.frames.clear()
        if entered:
            self._context.__exit__(None, None, None)
        # 画面のスレッドへ返す ここで返さないと、呼ぶ側が捨てるときに
        # 別のスレッドに付いたままのコンテキストを触ることになる
        application = QCoreApplication.instance()
        if application is not None:
            self._context.move_to_thread(application.thread())

    def _loop(self, renderer: FrameRenderer, cache: PreviewCache) -> None:
        while True:
            waiting = self._paused or self._halted or self._idle or not cache.enabled
            for generation, name, args in self._take(block=waiting):
                if name == "stop":
                    return
                self._apply(renderer, cache, name, args)
                self._generation = generation
                if name != "pause":
                    # 何か変わったら、休んでいた所も失敗していた所もやり直す
                    self._idle = False
                    self._halted = False
            if self._paused or self._halted or self._idle or not cache.enabled:
                continue
            self._set_rendering(cache.next_frame(self._playhead))
            try:
                frame = cache.step_frame(self._playhead)
                if frame is not None:
                    # 載せる前に GPU の終わりを待つ 共有したテクスチャの中身は、
                    # 書いた側の命令が終わるまで画面の側から見える保証が無い
                    GL.glFinish()
            except Exception as exc:
                # 先読みは無くても絵は出る 止めて伝えるだけにする
                self._set_rendering(None)
                # 失敗する前に捨てた控えも画面の側へ渡す 渡さないと、壊れた控えを
                # 作り直す頼みが出ず、画面の側は同じ控えを使い続ける
                self._collect_discarded(renderer)
                self._halted = True
                self._failed(f"先読みを止めた: {exc}")
                continue
            self._collect_discarded(renderer)
            if frame is None:
                self._set_rendering(None)
                self._idle = True
                continue
            self._publish(frame, cache)

    def _set_rendering(self, frame: int | None) -> None:
        """いま描いているコマを画面の側へ知らせる 待っている画面の側を起こす"""
        with self._shelf.changed:
            self._shelf.rendering = frame
            self._shelf.changed.notify_all()

    def _take(self, *, block: bool) -> list[_Command]:
        """届いている頼みを全部受け取る 休んでいるなら 1 つ届くまで待つ"""
        taken: list[_Command] = []
        if block:
            taken.append(self._commands.get())
        while True:
            try:
                taken.append(self._commands.get_nowait())
            except queue.Empty:
                return taken

    def _apply(
        self, renderer: FrameRenderer, cache: PreviewCache, name: str, args: tuple[object, ...]
    ) -> None:
        if name == "project":
            project, invalidation = args
            assert isinstance(project, Project) and isinstance(invalidation, Invalidation)
            renderer.set_project(project)
            cache.invalidate(invalidation)
        elif name == "invalidate":
            (invalidation,) = args
            assert isinstance(invalidation, Invalidation)
            cache.invalidate(invalidation)
        elif name == "quality":
            (quality,) = args
            assert isinstance(quality, RenderQuality)
            renderer.set_quality(quality)
            cache.invalidate(Invalidation.all())
        elif name == "proxies":
            (proxies,) = args
            renderer.set_proxies(proxies)  # type: ignore[arg-type]  # ProxyStore か None
            cache.invalidate(Invalidation.all())
        elif name == "reopen":
            (media_ids,) = args
            renderer.reopen_sources(media_ids)  # type: ignore[arg-type]  # 素材の集まりか None
            cache.invalidate(Invalidation.all())
        elif name == "budget":
            budget, playhead = args
            assert isinstance(budget, int) and isinstance(playhead, int)
            cache.set_budget(budget, playhead)
        elif name == "backdrop":
            (backdrop,) = args
            assert isinstance(backdrop, Backdrop)
            cache.set_backdrop(backdrop)
        elif name == "threads":
            (threads,) = args
            assert isinstance(threads, int)
            renderer.set_decode_threads(threads)
        elif name == "playhead":
            (frame,) = args
            assert isinstance(frame, int)
            self._playhead = max(0, frame)
        elif name == "pause":
            (paused,) = args
            self._paused = bool(paused)
        else:
            raise ValueError(f"知らない頼み: {name}")

    def _publish(self, frame: int, cache: PreviewCache) -> None:
        """描いた絵を画面の側の一覧へ載せる 描いている間に編集が入っていたら捨てる"""
        surface = cache.cache.get(frame)
        with self._shelf.changed:
            # 載せても載せなくても「描いている」は終わる 待っている画面の側を起こす
            self._shelf.rendering = None
            self._shelf.changed.notify_all()
            if surface is not None and self._shelf.generation == self._generation:
                self._shelf.frames[frame] = surface
                return
        # 描いている間に絵の変わる頼みが届いた その頼みはまだ受け取っていないので、
        # この絵が変わる範囲に入るかは分からない 捨てて描き直す方へ倒す
        cache.invalidate(Invalidation.over([(frame, frame + 1)]))

    def _unpublish(self, frame: int) -> None:
        """描画先を使い回す・手放す前に、画面の側の一覧から外す

        鍵を取るので、画面の側がその絵を出している最中なら、出し終えるまで待つ
        """
        with self._shelf.lock:
            self._shelf.frames.pop(frame, None)

    def _collect_discarded(self, renderer: FrameRenderer) -> None:
        found = renderer.take_discarded()
        if found:
            with self._shelf.lock:
                self._shelf.discarded |= found
