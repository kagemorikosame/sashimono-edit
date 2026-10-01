"""字幕パネルの速さを測る道具（tools/bench_subtitles.py）が作品を組めること

字幕を音声ごとに持つようにしたとき、道具だけが前の書き方（素材に ``transcript=``）の
ままで、回すと作品を組む所で TypeError になっていた（PR #231 の指摘） 道具は試験から
呼ばれないので、壊れても誰も気づかない
"""

from __future__ import annotations

import importlib.util
import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_the_bench_builds_its_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 道具は読み込んだ時に置き場と画面の出し方を書き換える 試験の間だけにとどめる
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "bench_subtitles", ROOT / "tools" / "bench_subtitles.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = module._project(12, Fraction(10))
    transcript = project.media[0].transcript
    assert transcript is not None and len(transcript) == 12
