"""AviUtl2 本体の書き出しで測った置き方に、読み込んで描いた絵が合うか（Issue #167）

値はどれも 2026-09-25 に AviUtl2 v2.1.6a で ``tools/aviutl_scale_probes.py`` の見本を
連番の PNG で書き出して測った物（1920x1080・60fps） 見本は道具と同じ物をここで作り、
同じ読み込み（``map_object``）と描画を通して白の外形を比べる

* 拡大率は横と縦に 1 回ずつ掛かる 読み込みが ``scale_y`` にも拡大率を入れていたので、
  拡大率 200 の 200x200 が 400x800 に、50 が 100x50 になっていた（#166 の YMM4 と同じ誤り）
* 描画設定の縦横比は正で横を、負で縦を縮める 拡大率 と リサイズ の X Y は軸ごとの拡大率
* クリッピングは絵の置かれた範囲の端から数える（#174） 中心の位置を変更 は残りを真ん中へ戻す
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from sashimono.compat.aviutl.exo import load_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import gather_media, place
from sashimono.core.model import MediaItem, Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import ProbeError, probe_media
from sashimono.engine.render import FrameRenderer

pytestmark = pytest.mark.usefixtures("gpu")

ROOT = Path(__file__).resolve().parents[2]

#: 見本ごとに AviUtl2 の書き出しに出た白の外形 ``(左, 上, 右, 下)`` 右と下は外形の 1 つ外
#: どの見本も 200x200 の四角（cr は 300x300、画像は 160x90）を画面の真ん中に置いた
MEASURED: dict[str, tuple[int, int, int, int]] = {
    "sc01_base": (860, 440, 1060, 640),
    "sc02_draw_200": (760, 340, 1160, 740),
    "sc03_draw_50": (910, 490, 1010, 590),
    "sc04_aspect_50": (910, 440, 1010, 640),
    "sc05_aspect_minus_50": (860, 490, 1060, 590),
    "sc06_draw_200_aspect_50": (860, 340, 1060, 740),
    "sc07_zoom_filter_200": (760, 340, 1160, 740),
    "sc08_zoom_filter_xy": (760, 490, 1160, 590),
    "sc09_zoom_filter_150_xy": (660, 465, 1260, 615),
    "sc10_resize_200": (760, 340, 1160, 740),
    "sc11_resize_xy": (910, 390, 1010, 690),
    "sc12_image_draw_200": (800, 450, 1120, 630),
    "cr01_clip": (830, 400, 1030, 650),
    "cr02_clip_centre": (860, 415, 1060, 665),
    "cr03_clip_draw_200": (700, 260, 1100, 760),
    "cr04_clip_image": (910, 505, 1040, 565),
}
#: 外形の端の許し（画素） 縁のにじみを 128 で切る所の揺れ
TOLERANCE = 1


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aviutl_scale_probes", ROOT / "tools" / "aviutl_scale_probes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _probe_or_none(path: Path) -> MediaItem | None:
    try:
        return probe_media(path)
    except ProbeError:
        return None


def test_every_probe_has_a_measurement(tool: ModuleType) -> None:
    """見本を足したのに測り値を書き忘れると、その見本は何も守らない"""
    assert {probe.name for probe in tool.PROBES} == set(MEASURED)


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_the_picture_lands_where_aviutl2_put_it(
    tool: ModuleType, tmp_path: Path, name: str
) -> None:
    # 拡大率・縦横比・リサイズ・クリッピングの読み方がずれると、AviUtl2 のオブジェクトを
    # 読み込んだ絵が AviUtl2 と違う大きさや位置に描かれる
    (probe,) = [probe for probe in tool.PROBES if probe.name == name]
    image = tmp_path / "white.png"
    image.write_bytes(tool.solid_png(*tool.IMAGE_SIZE, (255, 255, 255)))
    source = tmp_path / f"{name}.object"
    source.write_bytes(tool.probe_text(probe, image).encode("utf-8"))

    rate = FrameRate(60)
    report = CompatibilityReport()
    objects = [
        mapped
        for obj in load_exo(source).objects
        if (mapped := map_object(obj, rate, report=report)) is not None
    ]
    project = Project.create(ProjectSettings(width=1920, height=1080, frame_rate=rate))
    plan = gather_media(objects, project, _probe_or_none)
    assert not plan.missing
    for command in [*plan.commands, *place(objects, project, at_frame=0, media=plan.media)]:
        project = command.apply(project)
    renderer = FrameRenderer(project)
    try:
        picture = np.asarray(renderer.render(2))
    finally:
        renderer.close()

    box = tool.lit_box(picture)
    assert box is not None, f"{name} が何も描かれなかった"
    expected = MEASURED[name]
    assert max(abs(a - b) for a, b in zip(box, expected, strict=True)) <= TOLERANCE, (
        f"{name}: AviUtl2 {expected} こちら {box}"
    )
    # 見本の項目はどれも写せるはず 残ると、写していない項目で絵が合っただけになる
    assert report.is_empty, report.lines()
