"""折れ線の点の読み取り

点列は 1 フレームごとに解き直すので、壊れたファイルの巨大な入力でも頭を抑える
"""

from __future__ import annotations

from sashimono.engine.sources import MAX_POLYLINE_POINTS, polyline_points


def test_the_number_of_points_is_capped() -> None:
    # 上限が無いと、巨大な点列を毎フレーム解き直して再生が止まる
    text = ";".join(f"{i},0" for i in range(MAX_POLYLINE_POINTS + 50))
    assert len(polyline_points(text)) == MAX_POLYLINE_POINTS


def test_unreadable_pairs_are_skipped() -> None:
    # 1 組の書き損じで点列の全体を諦めると、点を 1 つ打ち間違えただけで線が丸ごと画面から消える
    assert polyline_points("1,2;だめ;3,4") == [(1.0, 2.0), (3.0, 4.0)]


def test_the_y_axis_points_up() -> None:
    # ほかの位置と同じ向き 下向きのままだと線だけ上下が逆に出る
    assert polyline_points("0,10")[0][1] == 10.0


def test_a_long_tail_is_dropped_even_when_pairs_are_unreadable() -> None:
    # 余りを残すと、読めない組のぶんだけ巨大な文字列をカンマで割ることになる
    text = ";".join(["だめ"] * 10 + [f"{i},0" for i in range(MAX_POLYLINE_POINTS + 50)])
    assert len(polyline_points(text)) == MAX_POLYLINE_POINTS - 10
