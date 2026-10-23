"""書き出しの重さを測る道具（tools/bench_export.py）の、CPU と GPU の比べ方（#372）

書き出しは GPU の変換を作れないと黙って CPU で変換する それを知らずに GPU の行を測ると、
CPU で測った時間が「色変換 GPU」の行に並び、GPU へ移しても速くならないように読める
ここでは ffmpeg も GL も使わず、測る所を偽物に替えて、どの行を測るかだけを見る
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    # 道具は読み込んだ時に APPDATA と LOCALAPPDATA を一時フォルダへ向ける 試験の間だけに
    # とどめる（monkeypatch が元へ戻す） 戻さないと後の試験が別の置き場を読む
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    spec = importlib.util.spec_from_file_location(
        "bench_export", ROOT / "tools" / "bench_export.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _record_ways(monkeypatch: pytest.MonkeyPatch, tool: ModuleType) -> list[bool]:
    ways: list[bool] = []

    def fake_compare(*_arguments: object, gpu: bool) -> None:
        ways.append(gpu)

    monkeypatch.setattr(tool, "_compare_depths", fake_compare)
    return ways


class TestCompareWays:
    def test_without_a_gpu_conversion_the_gpu_row_is_skipped(
        self, tool: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # GPU の変換を作れない機械で GPU の行を測ると、CPU の時間が GPU の時間として並ぶ
        ways = _record_ways(monkeypatch, tool)
        tool.compare_pipeline(None, "libx264", 4, ROOT, [0], 1, gpu_available=False)
        assert ways == [False]
        assert "GPU の変換を作れない" in capsys.readouterr().out

    def test_with_a_gpu_conversion_both_rows_are_measured(
        self, tool: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # GPU を作れる機械で GPU の行を飛ばすと、#372 で速くなったかを比べられない
        ways = _record_ways(monkeypatch, tool)
        tool.compare_pipeline(None, "libx264", 4, ROOT, [0], 1, gpu_available=True)
        assert ways == [False, True]

    @pytest.mark.parametrize("made", [False, True], ids=["no-gpu", "gpu"])
    def test_main_passes_what_the_breakdown_found(
        self, tool: ModuleType, monkeypatch: pytest.MonkeyPatch, made: bool
    ) -> None:
        # 内訳で GPU を作れたかを渡し忘れると、作れない機械でも GPU の行が CPU の時間で出る
        class _Context:
            def release(self) -> None:
                """GL を持っていないので何もしない"""

        def fake_breakdown(*_arguments: object, gpu: bool) -> dict[str, list[float]] | None:
            if gpu and not made:
                return None
            return {"合成": [1.0]}

        passed: list[bool] = []

        def fake_compare(*_arguments: object, gpu_available: bool) -> None:
            passed.append(gpu_available)

        monkeypatch.setattr(tool.shutil, "which", lambda _name: "ffmpeg")
        monkeypatch.setattr(tool, "available_video_codecs", lambda: ["libx264"])
        monkeypatch.setattr(tool, "_make_source", lambda *_a, **_k: Path("source.mp4"))
        monkeypatch.setattr(tool, "OffscreenGLContext", _Context)
        monkeypatch.setattr(tool, "_project", lambda *_a: None)
        monkeypatch.setattr(tool, "breakdown", fake_breakdown)
        monkeypatch.setattr(tool, "decode_by_layers", lambda *_a: None)
        monkeypatch.setattr(tool, "gil_release", lambda *_a: None)
        monkeypatch.setattr(tool, "compare_pipeline", fake_compare)
        assert tool.main(["--frames", "1", "--layers", "1", "--compare"]) == 0
        assert passed == [made]
