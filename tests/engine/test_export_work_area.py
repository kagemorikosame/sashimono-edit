"""タイムラインで決めた範囲だけを書き出す（Issue #27）

書き出しの画面が作る設定をそのまま書き出しへ渡し、出来た動画の長さと、音が範囲に
そろっているかを見る 映像だけ範囲で切って音を頭から流すと、絵と音がずれた動画になる
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from sashimono.core.commands import Document, SetWorkArea, insert_media
from sashimono.core.model import Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import AudioDecoder, probe_media
from sashimono.engine.encode import export_project
from sashimono.ui.export_dialog import ExportDialog
from tests.media_fixtures import make_sample
from tests.ui.export_codecs import settled

pytestmark = pytest.mark.usefixtures("gpu")

#: 素材を置く位置 範囲の途中から音が鳴り出すようにして、音が範囲にそろっているかを見る
_PLACED_AT = 30
#: 書き出す範囲 前半の 15 コマは何も置いていない所、後半の 15 コマは素材の頭
_AREA = (15, 45)


@pytest.fixture
def placed_late(media_dir: Path) -> Project:
    """2 秒の音付きの素材を 1 秒目（30 コマ目）から置き、範囲を 15〜45 コマにした作品"""
    loud = make_sample(media_dir, "loud-range.mp4", duration=2.0, gain_db=20)
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(loud.path), at_frame=_PLACED_AT):
        document.execute(command)
    document.execute(SetWorkArea(_AREA))
    return document.project


def _peak(samples: np.ndarray) -> float:
    return float(np.abs(samples).max()) if samples.size else 0.0


def test_only_the_range_is_exported_with_its_audio(placed_late: Project, tmp_path: Path) -> None:
    """範囲の 30 コマ（1 秒）だけが出て、音も範囲の時刻から始まる

    範囲を渡し忘れると 90 コマの動画になる 音だけ頭から流すと、前半に素材の音が入り、
    後半（素材が映っている所）が無音になる
    """
    dialog = settled(ExportDialog(placed_late))
    try:
        settings = dialog._settings()
    finally:
        dialog.deleteLater()
    if settings is None:
        # 開けるコーデックが 1 つも無い機械では画面が設定を作らない 書き出しそのものができない
        pytest.skip("書き出しに使えるコーデックが無い")
    output = tmp_path / "range.mp4"
    # 出力先とコーデックだけ差し替える 範囲は画面が作ったものをそのまま使う
    export_project(placed_late, replace(settings, path=output, video_codec="libx264"))

    with av.open(str(output)) as container:
        frames = sum(1 for _ in container.decode(video=0))
        duration = Fraction(container.duration or 0, av.time_base)
    assert frames == _AREA[1] - _AREA[0]
    assert abs(duration - 1) < Fraction(1, 10)

    with AudioDecoder(output, sample_rate=48000) as decoder:
        silent = decoder.read_seconds(Fraction(1, 20), Fraction(7, 20))
        loud = decoder.read_seconds(Fraction(3, 5), Fraction(3, 10))
    assert _peak(silent) < 0.01, "何も置いていない所に音が入った（音が範囲にそろっていない）"
    assert _peak(loud) > 0.05, "素材が映っている所が無音（音が範囲にそろっていない）"
