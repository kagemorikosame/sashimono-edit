"""先読み（バックグラウンドレンダリング）

置き場の入れ替えは、外から見ても分からない所で効く 遠い絵を追い出すつもりで
すぐ使う絵を捨てていても、画面は正しく出続ける（作り直されるだけ）ので
気づけない 何を残して何を捨てたかを直に確かめる
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from OpenGL.error import GLError

from sashimono.engine.gpu import ShaderError
from sashimono.engine.render import FrameCache, Invalidation
from sashimono.engine.render.prefetch import CacheSurface

#: 予算を「何枚ぶん」で書くための単位 枚数で書かないと、遠い絵を捨てる所の
#: 試験が「この予算なら何枚入るのか」を数える所から始まってしまう
ONE_FRAME = 1920 * 1080 * 4


@dataclass
class FakeSurface:
    """GL を持たない描画先 何回描かれたか・手放されたかを覚える"""

    width: int
    height: int
    handle: int = 0
    color: int = 0
    released: bool = False
    drawn: list[int] = field(default_factory=list)

    def release(self) -> None:
        self.released = True


class Factory:
    """作った描画先を覚えておく係 使い回されているかを見る"""

    def __init__(self) -> None:
        self.made: list[FakeSurface] = []

    def __call__(self, width: int, height: int) -> CacheSurface:
        surface = FakeSurface(width, height)
        self.made.append(surface)
        return surface


def _cache(frames: int, *, factory: Factory | None = None) -> tuple[FrameCache, Factory]:
    made = factory if factory is not None else Factory()
    cache = FrameCache(1920, 1080, budget_bytes=ONE_FRAME * frames, surface=made)
    return cache, made


def _fill(cache: FrameCache, *frames: int) -> None:
    for frame in frames:
        cache.store(frame, lambda surface: None)


class TestHowManyFit:
    def test_the_budget_decides_the_count(self) -> None:
        # 設定した先読みのメモリと貯める枚数が合わないと、GPU のメモリを食い過ぎるか、余裕があるのに
        # 先読みが短くなる
        cache, _ = _cache(4)
        assert cache.capacity == 4

    def test_a_budget_too_small_holds_nothing(self) -> None:
        """1 枚も入らない予算では先読みしない

        1 枚だけ置くと、再生ヘッドが動くたびにその 1 枚を作り直して捨てる
        """
        cache = FrameCache(1920, 1080, budget_bytes=ONE_FRAME - 1, surface=Factory())
        assert cache.capacity == 0
        assert cache.store(0, lambda surface: None) is None

    def test_a_smaller_picture_fits_more(self) -> None:
        # 画質を下げれば同じメモリで長く貯まる 設定の説明はこの関係に基づく
        cache = FrameCache(960, 540, budget_bytes=ONE_FRAME, surface=Factory())
        assert cache.capacity == 4


class TestKeepingWhatIsNear:
    def test_a_stored_frame_comes_back(self) -> None:
        # 先読みした絵を取り出せないと、貯めたのに毎回描き直し、プレビューの再生が滑らかにならない
        cache, _ = _cache(4)
        surface = cache.store(7, lambda target: None)
        assert surface is not None
        assert cache.get(7) is surface

    def test_the_far_frame_goes_first(self) -> None:
        """いっぱいになったら、再生ヘッドから遠い絵を捨てる

        古い順で捨てると、再生ヘッドのすぐ先に貯めたばかりの絵を、さらに先を
        貯めるために捨てることになる
        """
        cache, _ = _cache(3)
        cache.set_playhead(0)
        _fill(cache, 0, 1, 50)
        cache.set_playhead(0)
        _fill(cache, 2)
        assert cache.cached == {0, 1, 2}

    def test_a_frame_farther_than_everything_is_not_stored(self) -> None:
        """近い絵を追い出してまで遠い絵は置かない

        置いてしまうと、すぐ使う所を捨てて誰も見ない先を貯め続ける
        """
        cache, _ = _cache(2)
        cache.set_playhead(0)
        _fill(cache, 0, 1)
        assert cache.store(99, lambda surface: None) is None
        assert cache.cached == {0, 1}

    def test_behind_goes_before_ahead_at_the_same_distance(self) -> None:
        """同じ距離なら後ろから捨てる 再生は前へ進む"""
        cache, _ = _cache(2)
        cache.set_playhead(10)
        _fill(cache, 5, 15)
        cache.set_playhead(10)
        assert cache.store(16, lambda surface: None) is not None
        assert cache.cached == {15, 16}

    def test_the_freed_surface_is_used_again(self) -> None:
        """捨てた枠は作り直さずに使い回す 毎回作ると確保と解放で時間を食う"""
        cache, factory = _cache(2)
        cache.set_playhead(0)
        _fill(cache, 0, 1)
        cache.invalidate(Invalidation.over([(0, 1)]))
        _fill(cache, 2)
        assert len(factory.made) == 2


class TestWhenDrawingFails:
    def test_nothing_is_kept(self) -> None:
        """途中で落ちた絵は残さない

        残すと、その位置だけ壊れた絵が出たまま直らない（次からは当たってしまう）
        """
        cache, _ = _cache(2)

        def explode(surface: CacheSurface) -> None:
            raise RuntimeError("描けない")

        with pytest.raises(RuntimeError):
            cache.store(3, explode)
        assert cache.cached == frozenset()

    def test_the_surface_comes_back_for_the_next_try(self) -> None:
        # 描けなかった枠が使い回されないと、失敗のたびに新しい枠を確保し、GPU のメモリを余分に取る
        cache, factory = _cache(2)

        def explode(surface: CacheSurface) -> None:
            raise RuntimeError("描けない")

        with pytest.raises(RuntimeError):
            cache.store(3, explode)
        _fill(cache, 3)
        assert len(factory.made) == 1


class TestThrowingAway:
    def test_only_the_changed_range_goes(self) -> None:
        """変わった範囲だけ捨てる 先読みの値打ちはここ"""
        cache, _ = _cache(5)
        _fill(cache, 0, 1, 10, 11)
        assert cache.invalidate(Invalidation.over([(10, 12)])) == 2
        assert cache.cached == {0, 1}

    def test_everything_goes_when_asked(self) -> None:
        # 作品全体に掛かる変更のあとも古い絵が残ると、再生したときに直す前の見た目が混ざる
        cache, _ = _cache(5)
        _fill(cache, 0, 1, 10)
        assert cache.invalidate(Invalidation.all()) == 3
        assert cache.cached == frozenset()

    def test_nothing_to_throw_away_touches_nothing(self) -> None:
        # 絵に関わらない操作のたびに先読みが捨てられると、貯めた絵を一から描き直して再生がもたつく
        cache, _ = _cache(5)
        _fill(cache, 0, 1)
        assert cache.invalidate(Invalidation.nothing()) == 0
        assert cache.cached == {0, 1}

    def test_a_new_size_lets_go_of_the_surfaces(self) -> None:
        """大きさが変われば取ってある絵は使えない 枠ごと手放す

        使い回すと、次に何枚持てるかの勘定が合わなくなる
        """
        cache, factory = _cache(2)
        _fill(cache, 0, 1)
        cache.resize(960, 540)
        assert cache.cached == frozenset()
        assert all(surface.released for surface in factory.made)

    def test_the_same_size_keeps_them(self) -> None:
        # 毎コマ呼ばれる所なので、同じ大きさなら何もしない
        cache, _ = _cache(2)
        _fill(cache, 0, 1)
        cache.resize(1920, 1080)
        assert cache.cached == {0, 1}


class TestChangingTheBudget:
    def test_shrinking_drops_the_far_ones(self) -> None:
        # 先読みのメモリを減らしたときに再生ヘッドのすぐ先の絵が捨てられると、
        # 再生のすぐ先で描き直しになる
        cache, _ = _cache(4)
        cache.set_playhead(0)
        _fill(cache, 0, 1, 2, 3)
        cache.set_budget(ONE_FRAME * 2)
        assert cache.cached == {0, 1}

    def test_turning_it_off_drops_all(self) -> None:
        """切ったら本当に空にする 残すと、切ったのにメモリを持ったままになる"""
        cache, factory = _cache(4)
        _fill(cache, 0, 1)
        cache.set_budget(0)
        assert cache.cached == frozenset()
        assert cache.capacity == 0
        assert all(surface.released for surface in factory.made)

    def test_growing_keeps_what_is_there(self) -> None:
        # 先読みのメモリを増やしただけで貯めた絵が捨てられると、描き直しで再生がもたつく
        cache, _ = _cache(2)
        _fill(cache, 0, 1)
        cache.set_budget(ONE_FRAME * 4)
        assert cache.cached == {0, 1}
        assert cache.capacity == 4


class TestWhenTheGpuRunsOut:
    """予算は「使っていい上限」であって「必ず取れる量」ではない

    載っているメモリの少ない機械では、途中で確保できなくなる そこで投げると、
    先読みの都合でプレビューそのものが落ちる
    """

    class Stingy(Factory):
        """``limit`` 枚までしか作れない描画先の工場"""

        def __init__(self, limit: int) -> None:
            super().__init__()
            self.limit = limit

        def __call__(self, width: int, height: int) -> CacheSurface:
            if len(self.made) >= self.limit:
                raise ShaderError("フレームバッファを構成できない")
            return super().__call__(width, height)

    def test_it_stops_at_what_it_could_take(self) -> None:
        # GPU のメモリが足りない機械で取れた分で止まらないと、毎コマ確保を試して失敗し続ける
        cache, _ = _cache(8, factory=self.Stingy(3))
        _fill(cache, 0, 1, 2, 3)
        assert cache.cached == {0, 1, 2}
        assert cache.capacity == 3

    def test_it_keeps_serving_what_it_has(self) -> None:
        """取れなくなっても、取ってある絵は出せる"""
        cache, _ = _cache(8, factory=self.Stingy(2))
        _fill(cache, 0, 1, 2)
        assert cache.get(0) is not None

    def test_changing_the_budget_tries_again(self) -> None:
        """設定を変えたら上限は忘れる ほかのアプリが掴んでいただけのことがある"""
        stingy = self.Stingy(1)
        cache, _ = _cache(8, factory=stingy)
        _fill(cache, 0, 1)
        assert cache.capacity == 1
        stingy.limit = 4
        cache.set_budget(ONE_FRAME * 8)
        _fill(cache, 1, 2)
        assert cache.cached == {0, 1, 2}

    def test_a_raw_gl_error_is_also_a_shortage(self) -> None:
        """GL の生のエラーでも同じ扱い

        フレームバッファが組めない形で返るとは限らない 確保そのものが
        GLError になることもあり、拾い分けるとそちらだけ画面まで抜ける
        """

        class Failing(Factory):
            def __call__(self, width: int, height: int) -> CacheSurface:
                raise GLError(err=1285)

        cache, _ = _cache(4, factory=Failing())
        _fill(cache, 0)
        assert cache.cached == frozenset()
        assert cache.capacity == 0

    def test_emptying_the_cache_lets_it_try_again(self) -> None:
        """空にしたら上限は忘れる ほかのアプリが掴んでいただけのことがある

        忘れないと、一度足りなかった機械では次の編集からも先読みが効かない
        """
        stingy = self.Stingy(1)
        cache, _ = _cache(4, factory=stingy)
        _fill(cache, 0, 1)
        assert cache.capacity == 1
        stingy.limit = 4
        cache.clear()
        _fill(cache, 0, 1, 2)
        assert cache.cached == {0, 1, 2}


class TestWhichOneGoesFirst:
    def test_the_tie_does_not_depend_on_the_order_they_went_in(self) -> None:
        """同じ遠さなら、入れた順に関係なく後ろを捨てる

        遠さだけで比べると Python の max は先に入れた方を選ぶので、
        入れる順で捨てるものが変わる
        """
        # 再生ヘッド 10 に対し、6 は後ろへ 4（重み 2 倍で 8）、18 は前へ 8 同じ遠さ
        for order in ((6, 18), (18, 6)):
            cache, _ = _cache(2)
            cache.set_playhead(10)
            _fill(cache, *order)
            cache.set_playhead(10)
            assert cache.store(11, lambda surface: None) is not None
            assert cache.cached == {18, 11}, f"入れた順 {order} で結果が変わる"

    def test_a_frame_as_far_ahead_as_one_behind_takes_its_place(self) -> None:
        """同じ遠さなら、後ろの絵を追い出して前の絵を置く

        置くかどうかを遠さだけで決めると、**捨てる 1 枚に選ばれている後ろの絵**を
        追い出せない 捨てる側と置く側で、ものさしが食い違う
        """
        cache, _ = _cache(2)
        cache.set_playhead(10)
        _fill(cache, 6, 12)
        cache.set_playhead(10)
        # 6 は後ろへ 4（重み 2 倍で 8）、18 は前へ 8 遠さは同じ
        assert cache.store(18, lambda surface: None) is not None
        assert cache.cached == {12, 18}
