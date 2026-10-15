"""コンタクトシートと範囲の連続したコマ（#320 A2）

何コマも 1 枚に並べて AI へ見せる道具 偽のホストで、どのコマを・どの順に・どの大きさで
並べるか、長く描くときに編集画面へ手を返すかを確かめる 本物の窓で絵を描く所は
``tests/ui/test_ai_contact_sheet.py``

各試験の頭の 1 行は、壊れたときに利用者に何が起きるか
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from sashimono.ai import operations
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Continue, ImageResult, Operation, Pending, find_operation
from sashimono.ai.safety import Safety
from sashimono.ai.sheet import (
    GAP,
    LABEL_HEIGHT,
    MAX_SHEET_EDGE,
    MIN_CELL_WIDTH,
    SheetLayout,
    sheet_layout,
)
from sashimono.core.model import Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import SceneState
from sashimono.engine.scenes import SceneSignal
from tests.ai.conftest import FakeHost, FakeSheet
from tests.ai.test_bridge import call_in_thread, pump_until


def run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(host, arguments)


def _sheet(host: FakeHost) -> FakeSheet:
    assert host.sheets, "コマを並べる 1 枚を用意していない"
    return host.sheets[-1]


def _frames(sheet: FakeSheet) -> list[int]:
    return [frame for _, frame, _ in sheet.drawn]


class TestContactSheet:
    def test_the_default_sheet_takes_the_middle_of_twelve_parts(self, host: FakeHost) -> None:
        # 壊れると、AI が全体を見渡したつもりで頭の黒いコマや一部の区間だけを見て判断する
        result = run(host, "contact_sheet")
        assert isinstance(result, ImageResult)
        assert result.png.startswith(b"\x89PNG")
        sheet = _sheet(host)
        # 300 フレームを 12 等分（25 ずつ）した、それぞれの真ん中
        assert _frames(sheet) == [12 + 25 * index for index in range(12)]
        assert [index for index, _, _ in sheet.drawn] == list(range(12))
        assert sheet.finished
        assert host.stopped == 1

    def test_each_frame_is_labelled_with_its_timecode_and_number(self, host: FakeHost) -> None:
        # 壊れると、並んだコマがいつの絵か分からず、AI が見つけた場面へ戻れない
        result = run(host, "contact_sheet", frames=[0, 45, 299])
        sheet = _sheet(host)
        labels = [label for _, _, label in sheet.drawn]
        assert labels == ["00:00:00:00 0", "00:00:01:15 45", "00:00:09:29 299"]
        # 画像の説明にも同じ並びで書く 絵の字を読み違えても、番号で引ける
        assert "1) 00:00:00:00 0 / 2) 00:00:01:15 45 / 3) 00:00:09:29 299" in result.caption

    def test_frames_are_drawn_in_the_order_given(self, host: FakeHost) -> None:
        # 壊れると、場面の切り替わりの一覧を渡しても並びが変わり、何番目がどの場面か食い違う
        run(host, "contact_sheet", frames=[250, 10, 90])
        assert _frames(_sheet(host)) == [250, 10, 90]

    def test_a_range_narrows_the_sheet(self, host: FakeHost) -> None:
        # 壊れると、範囲を区切っても全体から取ってしまい、見たい所が粗くしか見えない
        run(host, "contact_sheet", from_frame=100, to_frame=200, count=4)
        assert _frames(_sheet(host)) == [112, 137, 162, 187]

    def test_a_short_range_does_not_repeat_frames(self, host: FakeHost) -> None:
        # 壊れると、同じコマが何枚も並び、費用だけ掛かって AI が動きがないと読み違える
        run(host, "contact_sheet", from_frame=10, to_frame=14)
        assert _frames(_sheet(host)) == [10, 11, 12, 13]

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"frames": [0, 300]}, "タイムラインの外"),
            ({"frames": []}, "1 つ以上"),
            ({"frames": list(range(37))}, "36 個まで"),
            ({"frames": [1], "count": 3}, "一緒に使えません"),
            ({"count": 0}, "1〜36"),
            ({"count": 37}, "1〜36"),
            ({"count": 2.5}, "整数"),
            ({"count": "12"}, "整数"),
            ({"cell_width": 80}, "120〜640"),
            ({"cell_width": 2000}, "120〜640"),
            ({"from_frame": 300}, "描けるコマがありません"),
            ({"columns": 9}, "1〜8"),
        ],
    )
    def test_bad_requests_are_refused_with_a_way_to_fix_them(
        self, host: FakeHost, arguments: dict[str, Any], message: str
    ) -> None:
        # 壊れると、黙って端へ寄せた 1 枚を返し、AI が頼んでいないコマについて判断する
        with pytest.raises(ToolError, match=message):
            run(host, "contact_sheet", **arguments)
        assert host.sheets == []

    def test_an_empty_timeline_is_refused(self) -> None:
        # 壊れると、空のタイムラインで 0 枚の絵を描こうとして落ちる
        host = FakeHost(Project.create(ProjectSettings(frame_rate=FrameRate(30))))
        with pytest.raises(ToolError, match="空です"):
            run(host, "contact_sheet")
        with pytest.raises(ToolError, match="空です"):
            run(host, "preview_range")


class TestPreviewRange:
    def test_consecutive_frames_from_the_playhead(self, host: FakeHost) -> None:
        # 壊れると、動きを見たい所と違う所が並び、場面切り替えの出来を確かめられない
        host.frame = 120
        result = run(host, "preview_range")
        assert isinstance(result, ImageResult)
        sheet = _sheet(host)
        assert _frames(sheet) == list(range(120, 128))
        # 続いたコマは横へ読む 既定は 4 列
        assert (sheet.layout.columns, sheet.layout.rows) == (4, 2)

    def test_step_spreads_the_frames(self, host: FakeHost) -> None:
        # 壊れると、長い動き（ゆっくりした拡大など）を見ようとしても 1 コマずつしか取れない
        run(host, "preview_range", start_frame=100, count=4, step=10)
        assert _frames(_sheet(host)) == [100, 110, 120, 130]

    def test_frames_past_the_end_are_left_out(self, host: FakeHost) -> None:
        # 壊れると、終わりより後の黒いコマが並び、AI が最後に暗転があると読み違える
        result = run(host, "preview_range", start_frame=295, count=8)
        assert _frames(_sheet(host)) == [295, 296, 297, 298, 299]
        # 減らしたことは黙らず返事に書く 書かないと、AI は 8 枚を見たつもりで続きを判断する
        assert "頼んだ 8 コマのうち 5 コマ" in result.caption

    @pytest.mark.parametrize(
        "arguments",
        [
            {"start_frame": 0, "count": 8, "step": 300},
            {"start_frame": 100, "count": 4, "step": 250},
            {"start_frame": 299},
        ],
    )
    def test_a_range_that_leaves_one_frame_is_refused(
        self, host: FakeHost, arguments: dict[str, Any]
    ) -> None:
        # 壊れると、step が大きすぎるか終わりの間際から頼んだとき 1 コマだけの 1 枚が返り、
        # 動きを確かめたつもりの AI が止まった絵だと読み違える
        with pytest.raises(ToolError, match="2 コマ以上"):
            run(host, "preview_range", **arguments)
        assert host.sheets == []

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"start_frame": 300}, "タイムラインの外"),
            ({"start_frame": -1}, "タイムラインの外"),
            ({"count": 1}, "2〜16"),
            ({"count": 17}, "2〜16"),
            ({"step": 0}, "step"),
            ({"start_frame": True}, "整数"),
        ],
    )
    def test_bad_requests_are_refused(
        self, host: FakeHost, arguments: dict[str, Any], message: str
    ) -> None:
        # 壊れると、範囲の外や 0 おきの頼みで同じコマが並ぶか、落ちる
        with pytest.raises(ToolError, match=message):
            run(host, "preview_range", **arguments)


class TestLayout:
    @pytest.mark.parametrize("count", [1, 2, 5, 12, 16, 36])
    @pytest.mark.parametrize("cell_width", [120, 240, 640])
    @pytest.mark.parametrize("frame_size", [(1920, 1080), (1080, 1920), (3840, 2160), (640, 480)])
    def test_the_sheet_stays_within_the_limit(
        self, count: int, cell_width: int, frame_size: tuple[int, int]
    ) -> None:
        # 壊れると、1 枚が上限を超えて費用が膨らむか、モデルが縮めて読み時刻の字が潰れる
        layout = sheet_layout(count, cell_width=cell_width, frame_size=frame_size)
        assert layout.width <= MAX_SHEET_EDGE
        assert layout.height <= MAX_SHEET_EDGE
        assert MIN_CELL_WIDTH <= layout.cell_width <= cell_width
        width, height = frame_size
        assert abs(layout.cell_height - layout.cell_width * height / width) < 1
        _assert_cells_fit(layout)

    def test_cells_read_left_to_right_then_down(self) -> None:
        # 壊れると、絵の並びと説明の番号が食い違い、AI が別のコマの時刻を信じる
        layout = SheetLayout(count=5, columns=3, cell_width=200, cell_height=100)
        assert [layout.cell(index) for index in range(5)] == [
            (GAP, GAP),
            (GAP + 204, GAP),
            (GAP + 408, GAP),
            (GAP, GAP + 100 + LABEL_HEIGHT + GAP),
            (GAP + 204, GAP + 100 + LABEL_HEIGHT + GAP),
        ]
        with pytest.raises(IndexError):
            layout.cell(5)

    def test_the_default_sheet_is_near_square(self) -> None:
        # 壊れると、12 枚が 1 行に並んで長い辺の上限に当たり、升が小さくなって見分けられない
        layout = sheet_layout(12, cell_width=240, frame_size=(1920, 1080))
        assert (layout.columns, layout.rows, layout.cell_width) == (3, 4, 240)
        assert (layout.width, layout.height) == (736, 624)

    def test_a_sheet_too_small_to_read_is_refused(self, host: FakeHost) -> None:
        # 壊れると、36 枚を 1 列に詰めた豆粒の絵を返し、AI が何も見分けられないまま判断する
        with pytest.raises(ToolError, match="小さすぎます"):
            run(host, "contact_sheet", count=36, columns=1)


def _assert_cells_fit(layout: SheetLayout) -> None:
    """升と帯がどれも 1 枚の中にあり、重ならない"""
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(layout.count):
        x, y = layout.cell(index)
        end_x, end_y = x + layout.cell_width, y + layout.cell_height + LABEL_HEIGHT
        assert x > 0 and end_x < layout.width
        assert y > 0 and end_y < layout.height
        boxes.append((x, y, end_x, end_y))
    for first, one in enumerate(boxes):
        for other in boxes[first + 1 :]:
            assert (
                one[2] <= other[0] or other[2] <= one[0] or one[3] <= other[1] or other[3] <= one[1]
            )


class TestSafety:
    @pytest.mark.parametrize("name", ["contact_sheet", "preview_range"])
    def test_the_tools_only_read(self, name: str) -> None:
        # 壊れると、見るだけの道具に毎回確かめの窓が出るか、作品を変える道具が確かめなしで走る
        operation = find_operation(name)
        assert operation is not None
        assert operation.safety is Safety.READ_ONLY

    @pytest.mark.parametrize("name", ["contact_sheet", "preview_range"])
    def test_drawing_does_not_change_the_project(self, host: FakeHost, name: str) -> None:
        # 壊れると、見ただけで履歴が積まれ、「元に戻す」が AI の見た回数ぶん空振りする
        before = host.document.project
        labels = host.document.history_labels
        run(host, name)
        assert host.document.project == before
        assert host.document.history_labels == labels


class _SlowSheet(FakeSheet):
    """1 コマに時間の掛かる作品（4K の動画のシークなど）のふり"""

    seconds = 0.04

    def draw(self, index: int, frame: int, label: str) -> None:
        time.sleep(self.seconds)
        super().draw(index, frame, label)


class _SlowHost(FakeHost):
    def start_sheet(self, layout: SheetLayout) -> FakeSheet:
        sheet = _SlowSheet(layout)
        self.sheets.append(sheet)
        return sheet


class TestHandingBack:
    def test_the_drawing_is_split_by_time(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、何十コマを 1 回で描き切り、その間は画面が止まって中断も効かない
        monkeypatch.setattr(operations, "SHEET_SLICE_SECONDS", 0.0)
        operation = find_operation("contact_sheet")
        assert operation is not None
        result = operation.start(host, {"count": 5})
        steps = 0
        while isinstance(result, Continue):
            result = result.step()
            steps += 1
        assert isinstance(result, ImageResult)
        # 時間の区切りが 0 でも 1 回に 1 コマは描く（描かずに返すと先へ進まない）
        assert steps == 5
        assert len(_sheet(host).drawn) == 5

    def test_the_editor_keeps_responding_while_a_big_sheet_is_drawn(self, loaded: Project) -> None:
        # 壊れると、大きい作品のコンタクトシートを描く数秒の間、画面が描き直されず固まって見える
        host = _SlowHost(loaded)
        bridge = EditorBridge(host)
        operation = find_operation("contact_sheet")
        assert operation is not None
        done, box = call_in_thread(bridge, operation, {"count": 24})
        longest = 0.0
        turns = 0
        deadline = time.monotonic() + 20.0
        while not done.is_set() and time.monotonic() < deadline:
            started = time.monotonic()
            bridge.pump()
            longest = max(longest, time.monotonic() - started)
            turns += 1
            time.sleep(0.002)
        assert done.is_set(), "描き終わらない"
        assert isinstance(box.get("result"), ImageResult), box.get("error")
        assert len(_sheet(host).drawn) == 24
        # 1 回の pump は区切りの時間（0.1 秒）と 1 コマぶんで終わる 24 コマで約 1 秒を
        # 1 回で描き切ったら 0.9 秒を超える
        assert longest < operations.SHEET_SLICE_SECONDS + 3 * _SlowSheet.seconds
        assert turns > 5

    def test_cancelling_stops_a_sheet_halfway(self, loaded: Project) -> None:
        # 壊れると、中断の釦を押しても、何十コマを描き終えるまで止まらない
        host = _SlowHost(loaded)
        bridge = EditorBridge(host)
        operation = find_operation("contact_sheet")
        assert operation is not None
        done, box = call_in_thread(bridge, operation, {"count": 36})
        deadline = time.monotonic() + 20.0
        while not host.sheets or not host.sheets[-1].drawn:
            bridge.pump()
            time.sleep(0.002)
            assert time.monotonic() < deadline
        bridge.cancel()
        while not done.is_set() and time.monotonic() < deadline:
            bridge.pump()
            time.sleep(0.002)
        assert isinstance(box.get("error"), ToolError)
        assert len(host.sheets[-1].drawn) < 36
        assert not host.sheets[-1].finished

    def test_a_pump_hands_back_after_a_continuation(self, host: FakeHost) -> None:
        # 壊れると、続きの仕事が先に積まれていたとき同じ回で全部を拾い、分けても画面が止まる
        bridge = EditorBridge(host)
        results: list[object] = []

        def ask(work: Any) -> threading.Thread:
            thread = threading.Thread(target=lambda: results.append(bridge.call(work)))
            thread.start()
            return thread

        first = ask(lambda: Continue(lambda: None))
        deadline = time.monotonic() + 5.0
        # 糸の側が積み終えるのを待つ 積み終えたかは外から見えないので、待ち行列の長さを見る
        while bridge._work.qsize() < 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        second = ask(lambda: "後の仕事")
        while bridge._work.qsize() < 2:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        bridge.pump()
        first.join(5.0)
        assert len(results) == 1 and isinstance(results[0], Continue)
        assert bridge._work.qsize() == 1, "続きの出た回で後の仕事まで拾った"
        bridge.pump()
        second.join(5.0)
        assert results[1] == "後の仕事"

    def test_other_work_waits_its_turn_without_being_lost(self, host: FakeHost) -> None:
        # 壊れると、コマを描く間に頼まれた別の道具の仕事が拾われず、AI が待ち続ける
        bridge = EditorBridge(host)
        sheet = find_operation("contact_sheet")
        read = find_operation("get_project")
        assert sheet is not None and read is not None
        first, first_box = call_in_thread(bridge, sheet, {"count": 3})
        second, second_box = call_in_thread(bridge, read, {})
        deadline = time.monotonic() + 10.0
        while not (first.is_set() and second.is_set()) and time.monotonic() < deadline:
            bridge.pump()
            time.sleep(0.002)
        assert isinstance(first_box.get("result"), ImageResult)
        assert isinstance(second_box.get("result"), dict)


class TestWithDetectedScenes:
    def test_the_scene_list_goes_straight_into_a_sheet(self, host: FakeHost) -> None:
        # 壊れると、detect_scenes の timeline を渡しても断られるか違うコマが並び、AI が
        # 場面ごとの様子を 1 枚で見られず、場面の数だけ preview_frame を呼ぶ
        media = host.project.media[0]
        host.scene_states[media.id] = SceneState(_signal([2.0, 6.5]), running=False)
        found = run(host, "detect_scenes", media_id=str(media.id))
        frames = [place["frame"] for row in found["cuts"] for place in row["timeline"]]
        assert frames == [60, 195]

        bridge = EditorBridge(host)
        sheet = find_operation("contact_sheet")
        assert sheet is not None
        done, box = call_in_thread(bridge, sheet, {"frames": frames})
        pump_until(bridge, done)
        assert isinstance(box.get("result"), ImageResult), box.get("error")
        assert _frames(_sheet(host)) == [60, 195]

    def test_a_waited_analysis_can_hand_over_to_drawing(self, host: FakeHost) -> None:
        # 壊れると、用意を待つ道具（Pending）の続きが描き続ける道具（Continue）だったとき、
        # 途中の印が AI へ返り、絵の代わりに読めない答えが届く
        drawn: list[int] = []

        def step() -> object:
            drawn.append(len(drawn))
            return Continue(step) if len(drawn) < 3 else "描き終えた"

        ready = threading.Event()
        ready.set()
        pending = Pending(wait=ready.wait, seconds=1.0, then=lambda: Continue(step))
        both = Operation(
            name="試験の道具",
            description="待ってから描く",
            schema={},
            handler=lambda _host, _arguments: pending,
        )
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, both, {})
        pump_until(bridge, done)
        assert box.get("result") == "描き終えた", box.get("error")
        assert drawn == [0, 1, 2]


def _signal(cuts: list[float], *, until: float = 10.0) -> SceneSignal:
    """30fps で ``until`` 秒まで ``cuts`` の秒で跳ねる差の並び（test_detect_scenes と同じ形）"""
    fps = 30
    times = np.arange(1, round(until * fps) + 1, dtype=np.float64) / fps
    pixel = np.full(times.shape, 0.01, np.float32)
    histogram = np.full(times.shape, 0.01, np.float32)
    for moment in cuts:
        index = round(moment * fps) - 1
        pixel[index], histogram[index] = 0.3, 0.8
    return SceneSignal(times, pixel, histogram, analyzed=until, complete=True)
