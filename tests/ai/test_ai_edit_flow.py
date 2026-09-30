"""AI の道具だけで 1 本の動画を作って書き出す（AI テスト #3）

ffmpeg でその場で作った素材（音付きの動画・画像・BGM）を読み込み、並べ、テキストと
図形を置き、エフェクトと場面切り替えを掛け、音量を変えてから書き出す 書き出した動画を
読み戻し、長さ・絵・音を数で確かめる 利用者の素材は使わない

道具を 1 つずつ見る試験は別にある ここは道具をつないだときに壊れる所（前の道具が作った
物を次の道具が見つけられない・置いた物が書き出しに出ない）を見る
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from sashimono.ai.operations import find_operation
from sashimono.core.model import LayerMode, MediaItem, Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import probe_media
from tests.ai.conftest import FakeHost
from tests.media_fixtures import make_sample

pytestmark = pytest.mark.usefixtures("gpu")

WIDTH, HEIGHT = 320, 180


class _Host(FakeHost):
    """素材は本当に調べる（ほかは偽物のまま）"""

    def probe(self, path: Path) -> MediaItem:
        self.probed.append(path)
        return probe_media(path)


def _call(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"{tool} という道具が無い"
    return operation(host, arguments)


def _clips(host: FakeHost) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = _call(host, "list_clips")
    return clips


def _materials(directory: Path) -> tuple[Path, Path, Path]:
    movie = make_sample(directory, "flow-movie.mp4", width=WIDTH, height=HEIGHT, duration=2.0)
    picture = directory / "flow-picture.png"
    bgm = directory / "flow-bgm.wav"
    for command in (
        ["-f", "lavfi", "-i", "testsrc2=size=160x90", "-frames:v", "1", str(picture)],
        ["-f", "lavfi", "-i", "sine=frequency=220:duration=4:sample_rate=48000", str(bgm)],
    ):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *command],
            check=True,
            capture_output=True,
        )
    return movie.path, picture, bgm


@pytest.mark.parametrize("mode", [LayerMode.MIXED, LayerMode.SEPARATED])
def test_a_short_video_made_only_with_the_tools(media_dir: Path, tmp_path: Path, mode: str) -> None:
    from sashimono.engine.encode import ExportSettings, available_video_codecs, export_project

    if not available_video_codecs():
        pytest.skip("映像のコーデックが無い")
    movie, picture, bgm = _materials(media_dir)
    settings = ProjectSettings(
        width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30), layer_mode=mode
    )
    host = _Host(Project.create(settings))

    # 動画（2 秒 = 60 コマ）の後ろへ画像（既定の長さ）を並べる
    _call(host, "import_media", paths=[str(movie), str(picture)])
    clips = _clips(host)
    movie_clip = next(c for c in clips if c["media"] == movie.name and c["start"] == 0)
    picture_clip = next(c for c in clips if c["media"] == picture.name)
    assert picture_clip["start"] == 60

    # BGM は頭から 音量を半分に
    _call(host, "import_media", paths=[str(bgm)])
    bgm_clip = next(c for c in _clips(host) if c["media"] == bgm.name)
    # 頭は動画の音と重なるので、BGM 用のトラックを足してそこへ移す
    kind = "mixed" if mode == LayerMode.MIXED else "audio"
    track = _call(host, "add_track", kind=kind, name="BGM")
    _call(
        host,
        "move_clip",
        clip_id=bgm_clip["clip_id"],
        timeline_start=0,
        track_id=track["track_id"],
    )
    bgm_clip = next(c for c in _clips(host) if c["media"] == bgm.name)
    volume = next(e for e in bgm_clip["effects"] if e["kind"] == "audio_volume")
    _call(
        host,
        "set_param",
        clip_id=bgm_clip["clip_id"],
        effect_id=volume["effect_id"],
        name="volume",
        value=50,
    )

    # 見出しの文字と、下に敷く帯（図形） 帯には影、動画にはぼかし
    _call(host, "add_shape", shape="rect", width=300, height=60, color="#203060", at_frame=0,
          duration=90, pos_y=-50)  # fmt: skip
    _call(host, "add_text", text="テスト", at_frame=0, duration=90, size=40, pos_y=-50)
    shape_clip = next(c for c in _clips(host) if c["source"] == "shape")
    text_clip = next(c for c in _clips(host) if c["source"] == "text")
    _call(host, "add_effect", clip_id=shape_clip["clip_id"], kind="shadow")
    _call(host, "add_effect", clip_id=movie_clip["clip_id"], kind="blur")
    # 文字は右から入ってくる
    _call(host, "add_keyframe", clip_id=text_clip["clip_id"], name="pos_x", frame=0, value=200)
    _call(host, "add_keyframe", clip_id=text_clip["clip_id"], name="pos_x", frame=20, value=0)

    # 動画と画像の切れ目（60）に、前後 15 コマのクロスフェード
    _call(host, "add_transition", style="fade", at_frame=45, duration=30)

    project = host.document.project
    output = tmp_path / f"flow-{mode}.mp4"
    export_project(project, ExportSettings(path=output))

    with av.open(str(output)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    with av.open(str(output)) as container:
        heard = np.concatenate([f.to_ndarray()[0] for f in container.decode(audio=0)])
        rate = container.streams.audio[0].rate
    assert len(frames) == project.duration

    band = (slice(HEIGHT // 2 + 30, HEIGHT // 2 + 70), slice(WIDTH // 2 - 100, WIDTH // 2 + 100))
    # 帯（紺）と文字が下の方に出ている 帯の色が混ざった所があれば置けている
    lower = frames[30][band].astype(np.float64)
    assert (lower[..., 2] - lower[..., 0]).max() > 20, "帯が映っていない"
    # 文字は右から入る 頭のコマと 30 コマ目で帯の中の絵が違う
    assert np.abs(frames[0][band].astype(int) - frames[30][band].astype(int)).mean() > 1
    # 切り替えの真ん中（60）は、前（40）とも後（80）とも違う混ざった絵
    top = (slice(0, HEIGHT // 2), slice(None))
    middle, before, after = (frames[i][top].astype(float) for i in (60, 40, 80))
    assert np.abs(middle - before).mean() > 3 and np.abs(middle - after).mean() > 3
    # 音は出ている（動画の音と BGM） 無音なら音量の設定か置き方が壊れている
    assert np.abs(heard[: rate * 2]).max() > 0.05
