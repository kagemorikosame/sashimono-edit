"""プレビューの画面と、別のスレッドの先読み（Issue #56 の 3）のつなぎ

走り係そのものは tests/engine/test_background_prefetch.py で見ている ここで見るのは
**何を渡し、いつ止め、作れなかったらどこへ戻るか** つなぎ間違えても絵は出続ける
（画面の側がその場で描く）ので、画面を見ていても気づけない
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Collection, Iterator
from dataclasses import replace
from typing import Any

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Keyframe,
    MediaId,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache.proxy import ProxyStore
from sashimono.engine.gpu import GLContextError
from sashimono.engine.render import Invalidation, PreviewCache, RenderQuality
from sashimono.engine.render.background import WAIT_FOR_WORKER_S, BackgroundPrefetch
from sashimono.engine.render.prefetch import CacheSurface
from sashimono.ui.preview import PreviewWidget
from tests.ui.test_preview_prefetch import StubCache, StubRenderer, _project


class _Signal:
    """つなぐ先を覚えるだけの合図"""

    def __init__(self) -> None:
        self.slots: list[Callable[[str], object]] = []

    def connect(self, slot: Callable[[str], object]) -> None:
        self.slots.append(slot)


class StubBackground:
    """GL もスレッドも持たない走り係 頼まれたことを順に覚える"""

    def __init__(self, log: list[str] | None = None) -> None:
        self.failed = _Signal()
        self.ended = _Signal()
        self.calls: list[tuple[str, object]] = []
        self.log = log if log is not None else []
        self.cached: frozenset[int] = frozenset()
        self.discarded: set[MediaId] = set()
        self.closed = 0

    def set_project(self, project: Project, invalidation: Invalidation) -> None:
        self.calls.append(("project", invalidation))

    def invalidate(self, invalidation: Invalidation) -> None:
        self.calls.append(("invalidate", invalidation))

    def set_quality(self, quality: RenderQuality) -> None:
        self.calls.append(("quality", quality))

    def set_proxies(self, proxies: ProxyStore | None) -> None:
        self.calls.append(("proxies", proxies))

    def reopen_sources(self, media_ids: Collection[MediaId] | None = None) -> None:
        self.calls.append(("reopen", media_ids))

    def set_budget(self, budget_bytes: int, playhead: int) -> None:
        self.calls.append(("budget", (budget_bytes, playhead)))

    def set_decode_threads(self, threads: int) -> None:
        self.calls.append(("threads", threads))

    def set_playhead(self, frame: int) -> None:
        self.calls.append(("playhead", frame))

    def set_paused(self, paused: bool) -> None:
        self.calls.append(("pause", paused))

    def take_discarded(self) -> set[MediaId]:
        found, self.discarded = self.discarded, set()
        return found

    def close(self) -> None:
        self.closed += 1
        self.log.append("走り係を止めた")

    def named(self, name: str) -> list[object]:
        return [value for called, value in self.calls if called == name]


class ClosingRenderer(StubRenderer):
    """閉じた順番を覚えるレンダラ"""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log
        self.discarded: set[MediaId] = set()

    def take_discarded(self) -> set[MediaId]:
        found, self.discarded = self.discarded, set()
        return found

    def set_decode_threads(self, threads: int) -> None:
        pass

    def prime(self, frame: int) -> None:
        self.log.append(f"{frame} を裏でデコード")

    def close(self) -> None:
        self.log.append("レンダラを閉じた")


class _ValidContext:
    """GL を作らずに「共有できるコンテキストがある」ことにする"""

    def isValid(self) -> bool:  # noqa: N802 - Qt の命名に合わせる
        return True


class _CurrentIs:
    """``QOpenGLContext.currentContext`` の代わり いつも画面のコンテキストが current"""

    def __init__(self, context: _ValidContext) -> None:
        self._context = context

    def currentContext(self) -> _ValidContext:  # noqa: N802 - Qt の命名に合わせる
        return self._context


def _patch_preview(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    """``PreviewWidget`` が実際に引く名前を差し替える

    モジュールの名前（``"sashimono.ui.preview.…"``）で差し替えない ほかの試験が
    モジュールを読み直していると、名前で引いた先は別のモジュールになり、
    ここで使う ``PreviewWidget`` には効かないまま通ってしまう
    """
    monkeypatch.setitem(PreviewWidget._start_background.__globals__, name, value)


@pytest.fixture
def parts(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[PreviewWidget, StubCache, list[str]]]:
    """GL を持たない画面 走り係はまだ無い"""
    del qt_application
    project, _, _ = _project()
    widget = PreviewWidget(project, prefetch_bytes=1024 * 1024 * 1024)
    log: list[str] = []
    monkeypatch.setattr(widget, "makeCurrent", lambda: None)
    monkeypatch.setattr(widget, "doneCurrent", lambda: None)
    context = _ValidContext()
    monkeypatch.setattr(widget, "context", lambda: context)
    _patch_preview(monkeypatch, "QOpenGLContext", _CurrentIs(context))
    monkeypatch.setattr(widget, "_renderer", ClosingRenderer(log))
    stub = StubCache()
    monkeypatch.setattr(widget, "_cache", stub)
    yield widget, stub, log
    widget._idle.stop()


@pytest.fixture
def running(
    parts: tuple[PreviewWidget, StubCache, list[str]], monkeypatch: pytest.MonkeyPatch
) -> tuple[PreviewWidget, StubCache, StubBackground]:
    """走り係が動いている画面"""
    widget, stub, log = parts
    background = StubBackground(log)
    monkeypatch.setattr(widget, "_background", background)
    return widget, stub, background


class TestWhatItPassesOn:
    def test_an_edit_goes_over_with_its_range(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """編集はプロジェクトと捨てる範囲を 1 つにして渡す

        範囲を渡さずに全部捨てると、編集のたびに貯めた絵が消えて貯まらない
        範囲を渡し損ねると、走り係の絵が直す前のまま出続ける
        """
        widget, _, background = running
        project, _, far = _project()
        track = project.timeline.tracks[0]
        kept = tuple(clip for clip in track.clips if clip.id != far.id)
        after = replace(
            project,
            timeline=project.timeline.replace_track(track.with_clips((*kept, far.moved_to(200)))),
        )
        widget.set_project(project)
        background.calls.clear()
        widget.set_project(after)
        thrown = background.named("project")
        assert len(thrown) == 1
        assert isinstance(thrown[0], Invalidation)
        assert not thrown[0].contains(10), "触っていない所まで捨てさせている"
        assert thrown[0].contains(200), "動かした先を捨てさせていない"

    def test_a_rewritten_image_goes_over_without_touching_gl(
        self,
        running: tuple[PreviewWidget, StubCache, StubBackground],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """画像の見張りはタイマーから走る 走り係へ渡すときも GL を触らない"""
        widget, _, background = running

        def refuse() -> None:
            raise AssertionError("見張りが GL のコンテキストを触った")

        monkeypatch.setattr(widget, "makeCurrent", refuse)
        renderer = widget.renderer
        assert isinstance(renderer, StubRenderer)
        renderer.stale = frozenset({"模様.png"})
        widget.check_images()
        assert len(background.named("invalidate")) == 1

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            (lambda widget: widget.set_proxies(ProxyStore(height=540)), "proxies"),
            (lambda widget: widget.reload_sources(), "reopen"),
            (lambda widget: widget.set_quality(RenderQuality(2)), "quality"),
            (lambda widget: widget.set_decode_threads(1), "threads"),
            (lambda widget: widget.refresh_all(), "invalidate"),
        ],
    )
    def test_every_change_reaches_the_worker(
        self,
        running: tuple[PreviewWidget, StubCache, StubBackground],
        action: object,
        expected: str,
    ) -> None:
        """走り係は自分のレンダラを持つ 渡し損ねると、走り係だけ古い設定で描き続ける

        控えを切ったのに走り係だけ控えを読む、画質を上げたのに走り係の絵だけ粗い、など
        """
        widget, _, background = running
        action(widget)  # type: ignore[operator]
        assert background.named(expected), f"{expected} を走り係へ渡していない"

    def test_the_budget_goes_over_at_once(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        # 走り係は自分のスレッドで描画先を手放すので、画面の GL を待たずに渡してよい
        widget, _, background = running
        widget.set_frame(30)
        widget.set_prefetch_bytes(256 * 1024 * 1024)
        assert background.named("budget") == [(256 * 1024 * 1024, 30)]

    def test_the_discarded_proxies_of_the_worker_are_collected(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """走り係のレンダラが捨てた控えも作り直しに回す

        走り係は先のコマを読むので、壊れた控えに先に当たるのはたいてい走り係
        拾わないと、その控えは作り直されないまま元の素材を読み続ける
        """
        widget, _, background = running
        worker_found = MediaId("走り係が捨てた素材")
        background.discarded = {worker_found}
        assert worker_found in widget.take_discarded()


class TestWhenItRuns:
    def test_the_playhead_goes_over(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        # 走り係に再生位置が届かないと、先読みが前の位置のまわりを描き続け、動かした先の絵が遅れる
        widget, stub, background = running
        widget.set_frame(42)
        assert background.named("playhead")[-1] == 42
        # 走り係は自分で回る 画面のスレッドで同じ所を描くと、同じ絵を 2 か所で貯める
        assert not widget._idle.isActive()
        widget._prefetch_step()
        assert stub.steps == []

    def test_playing_pauses_it(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """再生中は止める 出す側と同じ GPU を奪い合うと、いま出すコマが遅れる"""
        widget, _, background = running
        widget.set_playing(True)
        assert background.named("pause")[-1] is True
        widget.set_playing(False)
        assert background.named("pause")[-1] is False

    def test_playing_starts_decoding_the_edge_of_the_stored_frames(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """再生を始めたら、貯めた所の端のコマを画面の側のデコーダで先に読ませる

        画面の側のデコーダは、貯めた所を再生する間は動かない 読ませておかないと、
        端を越えた 1 コマ目で鍵フレームから読み直し、再生がそこで引っかかる
        """
        widget, _, background = running
        background.cached = frozenset(range(10))
        widget.set_frame(3)
        widget.set_playing(True)
        assert "10 を裏でデコード" in background.log

    def test_the_edge_is_found_past_a_frame_drawn_on_the_screen(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """いまのコマを画面の側で描いていても、その先に貯まった所があれば端を読ませる

        画面の側で描いたコマは走り係に描かせない（次のコマから貯めさせる）ので、
        いまのコマだけ抜けているのがふつう ここで諦めると、端で引っかかる
        """
        widget, _, background = running
        background.cached = frozenset(range(4, 20))
        widget.set_frame(3)
        widget.set_playing(True)
        # いまのコマを描く前に頼むと、描くときにその頼みを待ったうえで頭へ戻って
        # 読み直し、頼んだ意味が無くなる 描き終えてから頼む
        assert not [entry for entry in background.log if "裏でデコード" in entry]
        assert widget._prime_after_paint, "描き終えた後に頼む印を付けていない"
        # paintGL が描き終えた所で呼ぶのと同じ（ここでは GL が無いので直に呼ぶ）
        widget._prime_edge()
        assert "20 を裏でデコード" in background.log

    def test_nothing_is_primed_without_stored_frames_ahead(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        # 先に貯まった所が無ければ、描く道がそのまま順に読む 先に頼むと同じ所を 2 度頼む
        widget, _, background = running
        background.cached = frozenset(range(10, 20))
        widget.set_frame(3)
        widget.set_playing(True)
        assert not [entry for entry in background.log if "裏でデコード" in entry]

    def test_it_is_stopped_before_the_renderer_is_closed(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """窓を閉じるときは、走り係を先に止める

        走り係のコンテキストは画面のコンテキストと共有している 画面の側を先に
        捨てると、走り係が描いている最中の共有の資源が消える
        """
        widget, _, background = running
        widget.shutdown()
        assert background.log == ["走り係を止めた", "レンダラを閉じた"]
        assert widget._background is None

    def test_turning_the_setting_off_stops_it_and_falls_back(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """設定で切ったら走り係を止めて、画面のスレッドで貯める

        止めないと、切ったのに裏でスレッドが GPU を使い続ける
        """
        widget, stub, background = running
        widget.set_prefetch_thread(False)
        assert background.closed == 1
        assert not widget.prefetch_in_background
        widget.set_frame(9)
        widget._prefetch_step()
        assert stub.steps == [9], "画面のスレッドの先読みへ戻っていない"


class TestWhenItCannotRun:
    def test_it_starts_on_the_first_idle_moment(
        self,
        parts: tuple[PreviewWidget, StubCache, list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """作れたら走り係へ任せ、画面のスレッドでは描かない"""
        widget, stub, _ = parts
        made: list[StubBackground] = []

        def make(*args: object, **kwargs: object) -> StubBackground:
            made.append(StubBackground())
            return made[-1]

        _patch_preview(monkeypatch, "BackgroundPrefetch", make)
        widget.set_frame(12)
        widget._prefetch_step()
        assert len(made) == 1
        assert widget.prefetch_in_background
        assert made[0].named("playhead") == [12]
        assert stub.steps == []

    def test_frames_that_could_not_be_let_go_are_let_go_at_the_next_paint(
        self,
        parts: tuple[PreviewWidget, StubCache, list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """走り係を作るときに current にできなかったら、次に描く所で画面の側の絵を手放す

        手放さないと、走り係が動く間ずっと画面の側の絵も抱え、先読みの予算の 2 倍の
        GPU のメモリを使い続ける 窓が隠れていると current にならないまま戻る
        """
        widget, stub, _ = parts
        _patch_preview(monkeypatch, "QOpenGLContext", _CurrentIs(None))  # type: ignore[arg-type]
        _patch_preview(monkeypatch, "BackgroundPrefetch", lambda *a, **k: StubBackground())
        widget._prefetch_step()
        assert widget.prefetch_in_background
        assert getattr(stub, "released", 0) == 0
        # paintGL の頭で Qt が current にしてから呼ぶ所
        widget._apply_pending()
        assert getattr(stub, "released", 0) == 1
        widget._apply_pending()
        assert getattr(stub, "released", 0) == 1, "二度手放している"

    def test_a_context_that_cannot_be_shared_falls_back(
        self,
        parts: tuple[PreviewWidget, StubCache, list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """共有したコンテキストを作れなければ、画面のスレッドで貯める

        作れないドライバや、GL の版が足りない仮想環境がある そこで先読みごと
        やめると、別のスレッドを入れる前より悪くなる
        """
        widget, stub, _ = parts
        attempts: list[object] = []

        def refuse(*args: object, **kwargs: object) -> None:
            attempts.append(args)
            raise GLContextError("共有できない")

        _patch_preview(monkeypatch, "BackgroundPrefetch", refuse)
        stopped: list[str] = []
        widget.prefetch_stopped.connect(stopped.append)
        widget.set_frame(5)
        widget._prefetch_step()
        assert stub.steps == [5], "作れなかったのに、画面のスレッドでも貯めていない"
        assert stopped and "共有できない" in stopped[0], "作れなかったことを伝えていない"
        # 何度も作り直さない 失敗し続けると、空き時間のたびに GL を作りに行く
        widget.set_frame(6)
        widget._prefetch_step()
        assert len(attempts) == 1

    def test_a_worker_that_ends_hands_back_to_the_screen_thread(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """走り係が続けられなくなったら、止めて画面のスレッドへ戻る"""
        widget, stub, background = running
        stopped: list[str] = []
        widget.prefetch_stopped.connect(stopped.append)
        widget._background_ended(background, "走り係が落ちた")
        assert background.closed == 1
        assert not widget.prefetch_in_background
        assert stopped == ["走り係が落ちた"]
        assert widget._idle.isActive()
        widget._prefetch_step()
        assert stub.steps, "画面のスレッドの先読みへ戻っていない"

    def test_a_late_signal_from_a_closed_worker_is_ignored(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        """閉じた走り係から遅れて届いた「止まった」で、いまの走り係を止めない

        合図はイベントループを通って遅れて届く 拾うと、設定を切って入れ直した
        だけで、動いている走り係まで「作れない」扱いにして画面のスレッドへ戻る
        """
        widget, _, background = running
        stopped: list[str] = []
        widget.prefetch_stopped.connect(stopped.append)
        old = StubBackground()
        widget._background_ended(old, "前の走り係が止まった")
        widget._background_failed(old, "前の走り係が失敗した")
        assert widget.prefetch_in_background
        assert background.closed == 0
        assert stopped == []

    def test_turning_prefetch_off_stops_the_worker_and_on_starts_a_new_one(
        self,
        running: tuple[PreviewWidget, StubCache, StubBackground],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """先読みを切ったら走り係ごと止め、入れ直したら作り直す

        予算 0 を渡すだけだと、スレッド・レンダラ・素材を掴んだデコーダ・共有した
        GL・効果の GPU の資源が窓を閉じるまで残る 切った意味が無い
        """
        widget, _, background = running
        widget.set_prefetch_bytes(0)
        assert background.closed == 1, "切ったのに走り係が動いたまま"
        assert not widget.prefetch_in_background
        made: list[StubBackground] = []

        def make(*args: object, **kwargs: object) -> StubBackground:
            made.append(StubBackground())
            return made[-1]

        _patch_preview(monkeypatch, "BackgroundPrefetch", make)
        widget.set_prefetch_bytes(512 * 1024 * 1024)
        widget._prefetch_step()
        assert len(made) == 1, "入れ直したのに走り係を作り直していない"
        assert widget.prefetch_in_background

    def test_proxies_dropped_by_a_closed_worker_are_still_handed_over(
        self, running: tuple[PreviewWidget, StubCache, StubBackground]
    ) -> None:
        # 走り係が壊れた控えを捨てた後で先読みを切ると、預かっておかない限り作り直しの
        # 頼みが出ず、その素材は元の素材から読み続ける
        widget, _, background = running
        background.discarded = {MediaId("broken")}
        widget.set_prefetch_bytes(0)
        assert background.closed == 1
        assert MediaId("broken") in widget.take_discarded()
        assert widget.take_discarded() == set(), "同じ素材を何度も渡している"

    def test_turning_it_back_on_tries_again(
        self,
        parts: tuple[PreviewWidget, StubCache, list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ドライバを入れ替えた後などに、アプリを起動し直さずに試せる
        widget, _, _ = parts
        widget._background_broken = True
        _patch_preview(monkeypatch, "BackgroundPrefetch", lambda *a, **k: StubBackground())
        widget.set_prefetch_thread(False)
        widget.set_prefetch_thread(True)
        widget._prefetch_step()
        assert widget.prefetch_in_background


def _fading_project(frames: int = 60) -> Project:
    settings = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30))
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create(settings))
    source = GeneratedSource(
        kind="shape", params={"shape": "background", "color": (0.5, 0.5, 0.5, 1.0)}
    )
    fading = AnimatedValue(
        keyframes=(Keyframe(frame=0, value=1.0), Keyframe(frame=frames - 1, value=0.1))
    )
    clip = Clip(timeline_start=0, duration=frames, source=source, opacity=fading)
    return AddClip(track.id, clip).apply(project)


@pytest.mark.usefixtures("gpu")
class TestTheScreenKeepsAnswering:
    def test_a_slow_frame_does_not_hold_up_the_event_loop(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """先読みの 1 コマが遅くても、画面は操作を受け付け続ける

        画面のスレッドで貯めると、1 コマ描く間はイベントループが止まる
        （4K を 3 枚重ねて効果を積むと 1 コマ 60ms） ここでは貯める側の 1 コマを
        わざと遅くして、その間を**どのスレッドが**過ごしたかと、過ごす間に画面の
        スレッドの 10ms おきのタイマーが 1 度でも届いたかを見る 遅くするのは貯める道
        （``_compose_into``）だけで、貯まっていないコマを画面の側でその場で描く道は速いまま

        タイマーの間隔の長さ（壁時計）では判定しない 画面の側は、要るコマを走り係が
        ちょうど描いていると ``WAIT_FOR_WORKER_S``（200ms）まで待つ作りで、機械が混んで
        その順番になると 202ms 止まり、150ms の上限で変更と関係なく落ちた（#214）
        画面のスレッドで描くか、画面のスレッドが走り係の描き終わりを待つ形に壊すと、
        遅いコマの間タイマーは 1 度も届かない 混み具合に関係なく 0 回になる

        その「待つ」所も必ず通す 走り係が描いている最中のコマへ再生ヘッドを動かし、
        そのコマは画面のタイマーが届くまで描き終えさせない 待つ所が上限で諦めず
        描き終わりまで待つ形に壊すと、画面は走り係を、走り係は画面を待って進まなくなる
        待つ長さの上限そのものは tests/engine/test_background_prefetch.py が見ている
        """
        original = PreviewCache._compose_into
        original_show = BackgroundPrefetch.show
        # 画面の側が走り係を待つ上限より十分長くする 同じ長さだと、正しく作っていても
        # 待つ間がコマの間を埋めてタイマーが 1 度も届かないことがある
        slow_s = WAIT_FOR_WORKER_S + 0.3
        # 待たせるコマを描き終えるのに要るタイマーの回数 画面が待つ所から抜けて
        # イベントループへ戻ったことの証しにする
        release_ticks = 3
        screen_thread = threading.get_ident()
        ticks = [0]
        # タイマーを動かしている間に始まって終わったコマだけを見る 動かす前から描いて
        # いたコマは、動かした直後に終わるとタイマーが届く暇が無い
        watching = [False]
        # (描いたスレッド, 描き始めのタイマーの回数, 描き終わりのタイマーの回数)
        slow_frames: list[tuple[int, int, int]] = []
        # 画面に待たせるコマ 用意ができたら次に走り係が描き始めたコマを選ぶ
        armed = [False]
        target: list[int | None] = [None]
        # 画面の側がそのコマを出しに来たときのタイマーの回数
        asked: list[int] = []
        # 待たせたコマを描き終えられたか 画面が戻らず逃げ道で抜けたら False
        released: list[bool] = []
        claim = threading.Lock()
        # 再生ヘッドを最初に動かす先
        start = 30

        def slow(cache: PreviewCache, frame: int, surface: CacheSurface) -> None:
            began, before = watching[0], ticks[0]
            if threading.get_ident() != screen_thread and armed[0]:
                with claim:
                    # 画面がいま出しているコマは選ばない ``set_frame`` は同じ番号だと
                    # 何もせず戻るので、画面が出しに来ないまま逃げ道で抜けて落ちる
                    # 走り係は再生ヘッドを受け取った直後、画面が描き終えて次のコマへ
                    # 送る前に、いまのコマを描き始めることがある
                    mine = target[0] is None and frame != start
                    if mine:
                        target[0] = frame
                if mine:
                    # 画面が出しに来てから、タイマーが届くまで描き終えない 逃げ道の
                    # 長さは判定に使わない 固まったまま試験が終わらないのを防ぐだけ
                    limit = time.perf_counter() + 10
                    while time.perf_counter() < limit and not (
                        asked and ticks[0] >= asked[0] + release_ticks
                    ):
                        time.sleep(0.005)
                    released.append(bool(asked) and ticks[0] >= asked[0] + release_ticks)
                    original(cache, frame, surface)
                    return
            time.sleep(slow_s)
            if began and watching[0]:
                slow_frames.append((threading.get_ident(), before, ticks[0]))
            original(cache, frame, surface)

        def show(background: BackgroundPrefetch, frame: int, *args: Any) -> bool:
            if frame == target[0] and not asked:
                asked.append(ticks[0])
            return original_show(background, frame, *args)

        monkeypatch.setattr(PreviewCache, "_compose_into", slow)
        monkeypatch.setattr(BackgroundPrefetch, "show", show)
        widget = PreviewWidget(_fading_project(), prefetch_bytes=64 * 64 * 4 * 6)
        # 画面へは出さない 本人の画面に窓を出したり、マウスを取り合ったりしない
        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        widget.resize(64, 64)
        widget.show()
        try:
            deadline = time.perf_counter() + 10
            while widget.renderer is None and time.perf_counter() < deadline:
                qt_application.processEvents()
            if widget.renderer is None:
                pytest.skip("プレビュー窓の GL を作れない")
            # 走り係ができるまで待つ 作るのは 1 度きりで、GL のコンテキストを作る
            # 100ms ほどは画面のスレッドで掛かる ここで見たいのは貯める間の方
            # 待つ長さは混んだ機械でも足りるよう長く取る 作れれば即座に抜ける
            ready = time.perf_counter() + 10
            while time.perf_counter() < ready and not getattr(
                widget, "prefetch_in_background", False
            ):
                qt_application.processEvents()

            def tick() -> None:
                ticks[0] += 1

            timer = QTimer()
            timer.setTimerType(Qt.TimerType.PreciseTimer)
            timer.setInterval(10)
            timer.timeout.connect(tick)
            timer.start()
            watching[0] = True
            # 再生ヘッドを動かすと、そこから貯め直す
            widget.set_frame(start)
            # 遅いコマを描き終え、画面に待たせるコマを描き終え、動かした先が貯まるまで
            # 回す 決まった長さで切ると、混んだ機械では貯まる前に切れて落ちる 上限は
            # 固まったときの逃げ道
            ahead = set(range(start, start + 10))
            moved = False
            end = time.perf_counter() + 30
            while time.perf_counter() < end and (
                len(slow_frames) < 2 or not released or not widget.cached_frames & ahead
            ):
                qt_application.processEvents()
                if slow_frames and not armed[0]:
                    armed[0] = True
                if target[0] is not None and not moved:
                    # 走り係が描いている最中のコマを画面が要るようにする
                    moved = True
                    widget.set_frame(target[0])
            watching[0] = False
            timer.stop()
            background = getattr(widget, "prefetch_in_background", False)
            stored = widget.cached_frames & ahead
        finally:
            widget.shutdown()
            widget.deleteLater()
            qt_application.processEvents()
        assert slow_frames, "遅くしたコマをタイマーを動かす間に 1 枚も描き終えていない"
        on_screen = [entry for entry in slow_frames if entry[0] == screen_thread]
        assert not on_screen, f"遅いコマを画面のスレッドで {len(on_screen)} 枚描いた"
        # 別のスレッドで描いていても、画面のスレッドがその描き終わりを待っていれば
        # ループは止まる 1 コマずつ見る 1 枚でも 0 回なら、そのコマの間は止まっていた
        answered = [after - before for _, before, after in slow_frames]
        assert min(answered) > 0, (
            f"描く間にタイマーが届かないコマがある（コマごとの回数 {answered}）"
        )
        assert asked, "走り係が描いている最中のコマを画面の側が出しに来ていない"
        assert released == [True], (
            "走り係が描いている最中のコマを出すときに、画面が上限で諦めず描き終わりを待った"
        )
        assert background, "別のスレッドで先読みしていない"
        assert stored, "動かした先を貯めていない（遅くした道を通っていない）"
