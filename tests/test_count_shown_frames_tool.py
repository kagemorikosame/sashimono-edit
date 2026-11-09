"""画面に出たコマを数える道具（tools/count_shown_frames.py #443 #463）

画面を撮る所は試験では動かせない（撮る画面が無い・人の机が写る） 数える所と、撮る側の
引数の組み立てを見る 素材を数える所は ffmpeg で作った素材で通す
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.media_fixtures import ffmpeg_available

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "count_shown_frames", ROOT / "tools" / "count_shown_frames.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SHOWINFO = """\
[Parsed_showinfo_0 @ 000001] n:   0 pts:      0 pts_time:0       duration: 1
[Parsed_showinfo_3 @ 000002] n:   0 pts:      0 pts_time:0       duration: 1
[Parsed_showinfo_0 @ 000001] n:   1 pts:      1 pts_time:0.0166667 duration: 1
[Parsed_showinfo_0 @ 000001] n:   2 pts:      2 pts_time:0.0333333 duration: 1
[Parsed_showinfo_3 @ 000002] n:   1 pts:      2 pts_time:0.0333333 duration: 1
[mpdecimate @ 000003] keep pts:2 pts_time:0.0333333
"""


class TestCounting:
    def test_showinfo_lines_are_split_by_where_they_sit(self, tool: ModuleType) -> None:
        # 間引きの前（0）が撮れたコマ、後ろ（3）が映ったコマ 混ぜると数が合わない
        times = tool.parse_showinfo(SHOWINFO)
        assert times[0] == pytest.approx([0.0, 0.0166667, 0.0333333])
        assert times[3] == pytest.approx([0.0, 0.0333333])

    def test_intervals_and_stalls(self, tool: ModuleType) -> None:
        # 間隔と止まりを取り違えると、規則的な飛び（#463）と長い止まり（#455）を見分けられない
        captured = [k / 60 for k in range(600)]
        # 2 コマに 1 回しか変わらず、途中で 0.2 秒止まる
        shown = [k / 30 for k in range(100)] + [100 / 30 + 0.2 + k / 30 for k in range(100)]
        count = tool.summarize("試し", captured, shown)
        assert count.seconds == pytest.approx(10.0)
        assert count.captured == 600
        assert count.shown == 200
        assert count.shown_fps == pytest.approx(20.0)
        assert count.interval_median_ms == pytest.approx(1000 / 30)
        assert count.stalls == 1
        assert not count.low

    def test_a_capture_that_fell_behind_is_marked_low(self, tool: ModuleType) -> None:
        # 撮る側が落とした間は映ったコマも数えられない 数字だけ並べると、再生の落としと
        # 見分けが付かない
        captured = [k / 60 for k in range(0, 1200, 2)] + [20.0]
        count = tool.summarize("試し", captured, captured)
        assert count.low
        assert "低め" in tool.format_table([count])

    def test_updates_only_capture_is_not_marked_low(self, tool: ModuleType) -> None:
        # ddagrab は画面が書き換わったときだけ届く 少ないのは撮る側の落としとは限らない
        # 絵が止まっていると頭から尻までが縮むので、頼んだ秒数で割る
        captured = [k / 30 for k in range(150)]
        count = tool.summarize("試し", captured, captured, seconds=10.0, fixed_rate=False)
        assert not count.low
        assert count.shown_fps == pytest.approx(15.0)

    def test_the_table_has_the_captured_count_beside_the_shown_count(
        self, tool: ModuleType
    ) -> None:
        # 撮れたコマが並んでいないと、映ったコマが少ないのが撮る側のせいかを読めない
        captured = [k / 60 for k in range(1200)]
        table = tool.format_table([tool.summarize("画面", captured, captured[::2])])
        header, rule, row = table.splitlines()
        assert "撮れたコマ" in header and "映ったコマ" in header and "95%" in header
        assert rule.count("---") == header.count("|") - 1
        assert "| 画面 | 20.0 | 1200 | 600 | 30.0 |" in row


class TestGrabArguments:
    def test_ddagrab_sends_only_updates_from_the_region(self, tool: ModuleType) -> None:
        # 決まった間隔で同じ絵を足すと、画面の切り替わりとの拍のずれで落とす（gdigrab と同じ）
        arguments = tool.grab_arguments("ddagrab", (10, 20, 640, 360), 5.0, output=1)
        source = arguments[arguments.index("-i") + 1]
        assert "dup_frames=0" in source
        assert "output_idx=1" in source
        assert "offset_x=10" in source and "offset_y=20" in source
        assert "video_size=640x360" in source
        assert "hwdownload" in arguments[arguments.index("-vf") + 1]

    def test_gdigrab_reads_the_region_from_the_desktop(self, tool: ModuleType) -> None:
        # 枠の指定が渡らないと画面全体を撮り、プレビューの外の動き（再生ヘッド）まで数える
        arguments = tool.grab_arguments("gdigrab", (10, 20, 640, 360), 5.0)
        assert arguments[arguments.index("-f") + 1] == "gdigrab"
        assert arguments[arguments.index("-offset_x") + 1] == "10"
        assert arguments[arguments.index("-video_size") + 1] == "640x360"
        assert arguments[-1] == "desktop"

    def test_an_unknown_grabber_is_refused(self, tool: ModuleType) -> None:
        # 黙って既定の撮り方へ落とすと、頼んだのと違う物差しの数字が表に並ぶ
        with pytest.raises(ValueError, match="知らない撮り方"):
            tool.grab_arguments("x11grab", (0, 0, 2, 2), 1.0)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg が無いので素材を作れない")
class TestCountFile:
    def _make(self, path: Path, rate: int) -> Path:
        # 速さ ``rate`` で動く絵を 60fps へ並べ直す 30 なら同じ絵が 2 枚ずつ続く
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=320x180:rate={rate}:duration=2",
                "-vf",
                "fps=60",
                "-c:v",
                "ffv1",
                str(path),
            ],
            check=True,
        )
        return path

    def test_every_frame_changing_counts_60(self, tool: ModuleType, tmp_path: Path) -> None:
        # 元の素材の数が 60 に届かないと、画面の数字と比べる基準がずれる
        count = tool.count_file(self._make(tmp_path / "a.mkv", 60), 2.0, 0.0, "元")
        assert count.captured == 120
        assert count.shown == 120
        assert count.interval_median_ms == pytest.approx(1000 / 60, abs=0.5)

    def test_repeated_frames_are_not_counted(self, tool: ModuleType, tmp_path: Path) -> None:
        # 同じ絵が続いた所は「映ったコマ」に入らない 入ると落としが見えない
        count = tool.count_file(self._make(tmp_path / "b.mkv", 30), 2.0, 0.0, "元")
        assert count.captured == 120
        assert count.shown == 60
        assert count.interval_median_ms == pytest.approx(1000 / 30, abs=0.5)
