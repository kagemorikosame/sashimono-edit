"""プレビューの画面と先読みのつなぎ

貯める仕組みそのものは tests/engine/test_prefetch.py で見ている ここで見るのは
**いつ動かして、いつ止めて、何を捨てるか** つなぎ間違えても絵は正しく出続ける
（作り直されるだけ）ので、画面を見ていても気づけない
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import replace

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.model import Clip, MediaId, Project, Track, TrackKind
from sashimono.effects import registry
from sashimono.engine.cache.proxy import ProxyStore
from sashimono.engine.render import Invalidation
from sashimono.ui.preview import SLOW_FRAME_MS, PreviewWidget


class StubRenderer:
    """GL を持たないレンダラ 渡されたものを覚えるだけ"""

    def __init__(self) -> None:
        self.reopened: list[Collection[MediaId] | None] = []
        #: 書き換わったことにする画像のパス
        self.stale: frozenset[str] = frozenset()

    def stale_images(self) -> frozenset[str]:
        found, self.stale = self.stale, frozenset()
        return found

    def set_project(self, project: Project) -> None:
        pass

    def set_proxies(self, proxies: ProxyStore | None) -> None:
        pass

    def set_quality(self, quality: object) -> None:
        pass

    def reopen_sources(self, media_ids: Collection[MediaId] | None = None) -> None:
        self.reopened.append(media_ids)


class StubCache:
    """GL を持たない置き場 捨てるよう言われた範囲を覚える"""

    def __init__(self) -> None:
        self.enabled = True
        self.thrown: list[Invalidation] = []
        self.steps: list[int] = []
        self.budget: int | None = None
        self.budget_playhead: int | None = None

    def invalidate(self, invalidation: Invalidation) -> int:
        self.thrown.append(invalidation)
        return 0

    def step(self, playhead: int) -> bool:
        self.steps.append(playhead)
        return False

    def set_budget(self, budget_bytes: int, playhead: int) -> None:
        self.budget = budget_bytes
        self.budget_playhead = playhead

    def release(self) -> None:
        self.released = getattr(self, "released", 0) + 1


def _project() -> tuple[Project, Clip, Clip]:
    """離れた所に 2 本 片方を触っても、もう片方の絵は変わらない"""
    track = Track(TrackKind.VIDEO, "V1")
    project = AddTrack(track).apply(Project.create())
    near = Clip(timeline_start=0, duration=30)
    far = Clip(timeline_start=100, duration=30)
    project = AddClip(track.id, near).apply(project)
    return AddClip(track.id, far).apply(project), near, far


@pytest.fixture
def preview(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[PreviewWidget, StubCache]]:
    del qt_application
    project, _, _ = _project()
    # ここで見るのは画面のスレッドで貯める道 別のスレッドの道は test_preview_background.py
    widget = PreviewWidget(project, prefetch_bytes=1024 * 1024 * 1024, prefetch_thread=False)
    # GL を作らずに中身だけ差し替える makeCurrent は本物の窓が無いと効かない
    monkeypatch.setattr(widget, "makeCurrent", lambda: None)
    monkeypatch.setattr(widget, "doneCurrent", lambda: None)
    monkeypatch.setattr(widget, "_renderer", StubRenderer())
    stub = StubCache()
    monkeypatch.setattr(widget, "_cache", stub)
    yield widget, stub
    widget._idle.stop()


class TestWhenItRuns:
    def test_it_starts_when_the_playhead_moves(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """再生ヘッドが動いたら貯め直す 次に貯める先が変わっている"""
        widget, _ = preview
        widget.set_frame(42)
        assert widget._idle.isActive()

    def test_it_stops_while_playing(self, preview: tuple[PreviewWidget, StubCache]) -> None:
        """再生中は止める 出す側と同じ GPU を奪い合うと、いま出すコマが遅れる"""
        widget, _ = preview
        widget.set_frame(42)
        widget.set_playing(True)
        assert not widget._idle.isActive()

    def test_it_comes_back_when_playing_stops(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        # 止めた後に先読みが戻らないと、止めた所の前後へ動かすたびに絵が出るまで待たされる
        widget, _ = preview
        widget.set_playing(True)
        widget.set_playing(False)
        assert widget._idle.isActive()

    def test_it_does_not_run_without_a_budget(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        # 貯める場所が無いのに先読みが回ると、使われない絵を描き続けて編集の操作が重くなる
        widget, stub = preview
        stub.enabled = False
        widget.set_frame(5)
        assert not widget._idle.isActive()

    def test_it_stops_when_there_is_nothing_left(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """貯まりきったら止める 止めないと、空き時間のたびに数え続ける"""
        widget, stub = preview
        widget.set_frame(7)
        widget._prefetch_step()
        assert stub.steps == [7]
        assert not widget._idle.isActive()


class TestWhatItThrowsAway:
    def test_an_edit_far_away_keeps_the_near_frames(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """離れたクリップを触っても、こちらの絵は残る 先読みの値打ちはここ"""
        widget, stub = preview
        project, _, far = _project()
        widget.set_project(project)
        track = project.timeline.tracks[0]
        kept = tuple(clip for clip in track.clips if clip.id != far.id)
        after = replace(
            project,
            timeline=project.timeline.replace_track(track.with_clips((*kept, far.moved_to(200)))),
        )
        stub.thrown.clear()
        widget.set_project(after)
        assert len(stub.thrown) == 1
        assert not stub.thrown[0].contains(10), "触っていない所まで捨てている"
        assert stub.thrown[0].contains(100), "動かす前の場所を捨て損ねている"
        assert stub.thrown[0].contains(200), "動かした先を捨て損ねている"

    def test_swapping_the_proxies_throws_everything(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """読む元が変われば絵も変わる 残すと、控えに切り替えても画面が変わらない"""
        widget, stub = preview
        widget.set_proxies(ProxyStore(height=540))
        assert [thrown.everything for thrown in stub.thrown] == [True]

    def test_the_same_proxies_throw_nothing(self, preview: tuple[PreviewWidget, StubCache]) -> None:
        # 設定を開いて閉じただけで貯めた絵が消えない
        widget, stub = preview
        widget.set_proxies(None)
        assert stub.thrown == []

    def test_reopening_sources_throws_everything(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """控えができた・壊れていて元へ戻した どちらも開き直した素材の絵が変わる"""
        widget, stub = preview
        widget.reload_sources()
        assert [thrown.everything for thrown in stub.thrown] == [True]

    def test_a_rewritten_image_drops_stale_frames_but_keeps_unrelated_prefetch(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """画像を別のソフトで描き直しても、プロジェクトは変わらない

        編集の差分では気付けないので、見張りが書き換わりを拾って捨てる
        捨てないと、先読みした所だけ古い模様のまま残る 画像を読まないクリップ
        まで捨てると、描き直すたびに先読みが消えて貯まらない
        """
        widget, stub = preview
        project, near, _ = _project()
        track = project.timeline.tracks[0]
        border = registry.require("border").create(pattern="模様.png")
        painted = replace(near, effects=(border,))
        others = tuple(clip for clip in track.clips if clip.id != near.id)
        project = replace(
            project,
            timeline=project.timeline.replace_track(track.with_clips((painted, *others))),
        )
        widget.set_project(project)
        stub.thrown.clear()
        renderer = widget.renderer
        assert isinstance(renderer, StubRenderer)
        renderer.stale = frozenset({"模様.png"})
        widget.check_images()
        assert len(stub.thrown) == 1
        assert stub.thrown[0].contains(10)
        assert not stub.thrown[0].contains(100), "画像を読んでいないクリップまで捨てている"

    def test_the_watch_does_not_touch_the_gl_context(
        self, preview: tuple[PreviewWidget, StubCache], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """タイマーからの見張りは、current にできない（窓が隠れている）ときにも走る

        そこで doneCurrent を呼ぶとほかのコンテキストを外してしまう 捨てるだけなら
        GL は要らないので、current にせずに捨てる
        """
        widget, stub = preview

        def refuse() -> None:
            raise AssertionError("見張りが GL のコンテキストを触った")

        monkeypatch.setattr(widget, "makeCurrent", refuse)
        monkeypatch.setattr(widget, "doneCurrent", refuse)
        renderer = widget.renderer
        assert isinstance(renderer, StubRenderer)
        renderer.stale = frozenset({"模様.png"})
        widget.check_images()
        assert len(stub.thrown) == 1

    def test_nothing_rewritten_throws_nothing(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        # 1 秒おきに見張るので、何も変わっていないときに捨てると先読みが貯まらない
        widget, stub = preview
        widget.check_images()
        assert stub.thrown == []


class TestTheBudget:
    def test_it_reaches_the_cache_when_gl_is_usable(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """選んだメモリは、GL を確実に使える所まで持ち越してから渡す

        減らすと置き場は描画先を手放す（GL を触る）ので、設定の窓から
        戻ってきたその場で渡すと、current でないまま解放しに行くことがある
        """
        widget, stub = preview
        widget.set_prefetch_bytes(512 * 1024 * 1024)
        assert stub.budget is None, "その場で GL を触りに行っている"
        widget._prefetch_step()
        assert stub.budget == 512 * 1024 * 1024, "持ち越したまま渡していない"

    def test_it_is_handed_over_only_once(self, preview: tuple[PreviewWidget, StubCache]) -> None:
        # 毎コマ渡すと、置き場が余分な入れ替えを繰り返す
        widget, stub = preview
        widget.set_prefetch_bytes(512 * 1024 * 1024)
        widget._prefetch_step()
        stub.budget = None
        widget._prefetch_step()
        assert stub.budget is None

    def test_the_same_budget_changes_nothing(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        # 毎回渡される所なので、同じ値なら触らない
        widget, stub = preview
        widget.set_prefetch_bytes(widget._prefetch_bytes)
        assert stub.budget is None


class TestWhenPrefetchingGoesWrong:
    """先読みは無くても絵は出る 先読みの都合でアプリを落とさない"""

    def test_a_failure_does_not_escape_the_timer(
        self, preview: tuple[PreviewWidget, StubCache], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """描けなくても投げ返さない

        ここは Qt のタイマーから呼ばれる 投げるとイベントループの外まで抜け、
        編集中のプロジェクトごとアプリが終わる
        """
        widget, stub = preview

        def explode(playhead: int) -> bool:
            raise RuntimeError("デコーダが開けない")

        monkeypatch.setattr(stub, "step", explode)
        stopped: list[str] = []
        widget.prefetch_stopped.connect(stopped.append)
        widget.set_frame(3)
        widget._prefetch_step()
        assert not widget._idle.isActive()
        assert stopped and "デコーダが開けない" in stopped[0]

    def test_a_slow_frame_stops_it(
        self, preview: tuple[PreviewWidget, StubCache], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """1 コマに掛かりすぎるなら貯めるのをやめる

        描いているのは編集画面と同じ GL コンテキストなので、その間は操作を
        受け付けられない 2 秒掛かる素材で貯め続けると、ずっと固まって見える
        """
        widget, stub = preview
        ticks = iter([0.0, (SLOW_FRAME_MS + 1) / 1000])
        monkeypatch.setattr("sashimono.ui.preview.time.perf_counter", lambda: next(ticks))
        monkeypatch.setattr(stub, "step", lambda playhead: True)
        stopped: list[str] = []
        widget.prefetch_stopped.connect(stopped.append)
        widget.set_frame(3)
        widget._prefetch_step()
        assert not widget._idle.isActive()
        assert stopped, "止めたことを伝えていない"

    def test_a_quick_frame_keeps_going(
        self, preview: tuple[PreviewWidget, StubCache], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 速く描けたコマで先読みが止まると、先の絵が貯まらず再生や移動のたびにまた待たされる
        widget, stub = preview
        ticks = iter([0.0, 0.001])
        monkeypatch.setattr("sashimono.ui.preview.time.perf_counter", lambda: next(ticks))
        monkeypatch.setattr(stub, "step", lambda playhead: True)
        widget.set_frame(3)
        widget._prefetch_step()
        assert widget._idle.isActive()

    def test_a_context_failure_does_not_escape_either(
        self, preview: tuple[PreviewWidget, StubCache], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """コンテキストを current にする所で落ちても同じ

        描く所だけを守っても、その手前で落ちたら結局アプリが終わる
        """
        widget, _ = preview

        def explode() -> None:
            raise RuntimeError("コンテキストを current にできない")

        monkeypatch.setattr(widget, "makeCurrent", explode)
        stopped: list[str] = []
        widget.prefetch_stopped.connect(stopped.append)
        widget.set_frame(3)
        widget._prefetch_step()
        assert not widget._idle.isActive()
        assert stopped

    def test_a_context_that_did_not_become_current_stops_it(
        self, preview: tuple[PreviewWidget, StubCache], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """current にならなくても makeCurrent は黙って戻る

        戻り値では分からないので、実際に current かどうかを見る 見ないと、
        窓が隠れている間に別のコンテキストへ描きに行く
        """
        widget, stub = preview
        monkeypatch.setattr(widget, "context", lambda: object())
        released: list[bool] = []
        monkeypatch.setattr(widget, "doneCurrent", lambda: released.append(True))
        stopped: list[str] = []
        widget.prefetch_stopped.connect(stopped.append)
        widget.set_frame(3)
        widget._prefetch_step()
        assert stub.steps == [], "current でないのに描きに行っている"
        # 取れていないのに戻すと、ほかが使っているコンテキストを外すことになる
        assert released == [], "取れていないコンテキストを戻している"
        assert not widget._idle.isActive()
        assert stopped


class TestTheQueuedTick:
    def test_a_tick_that_arrives_after_playback_starts_does_nothing(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """止めた後に届いた合図で描かない

        タイマーを止めても、すでに積まれた合図は届く そこで 1 コマ描くと、
        いま出すべきコマと GL を奪い合う
        """
        widget, stub = preview
        widget.set_frame(3)
        widget.set_playing(True)
        widget._prefetch_step()
        assert stub.steps == []

    def test_it_hands_over_where_the_playhead_is(
        self, preview: tuple[PreviewWidget, StubCache]
    ) -> None:
        """減らすときは、いま見ている位置を基準に捨てる

        渡さないと最後に描いた位置が基準になり、その間に再生ヘッドが動いていれば
        いま見ている辺りを「遠い絵」として捨てる
        """
        widget, stub = preview
        widget.set_frame(120)
        widget.set_prefetch_bytes(256 * 1024 * 1024)
        widget._prefetch_step()
        assert stub.budget_playhead == 120
