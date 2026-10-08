"""AviUtl2 の組み方のテキスト 入れ物は文字の枠、太字は AviUtl2 の太らせ方（#64）

AviUtl2 のテキストの入れ物は、字の形ではなく送り幅 x 行の高さの枠 画像合成・縁取りの
模様・万華鏡・オブジェクト分割・ミラーはこの入れ物を基準に動くので、字の形で代わりに
すると、どれも 20 画素ほど内側を基準にしてしまう
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from PySide6.QtGui import QFont, QFontMetricsF

from sashimono.compat.aviutl.catalog import ScriptCatalog, set_script_catalog
from sashimono.core.commands import AddClip, AddEffect, AddTrack
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    ParamValue,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.definition import registry
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from sashimono.engine.render.renderer import _object_box, _object_sized
from sashimono.engine.sources import render_source_framed
from sashimono.engine.text_wrap import graphemes

SCREEN = (640, 360)


@pytest.fixture
def ms_ui_gothic() -> None:
    """AviUtl2 で測った見本と同じ書体 無い機械では、別の書体で数が合わず落ちるので飛ばす"""
    if not QFont("MS UI Gothic").exactMatch():
        pytest.skip("MS UI Gothic が入っていない")


def text(**params: ParamValue) -> GeneratedSource:
    base: dict[str, ParamValue] = {
        "text": "田田田",
        "font": "MS UI Gothic",
        "size": AnimatedValue(60.0),
        "color": (1.0, 1.0, 1.0, 1.0),
        "layout": "aviutl",
    }
    base.update(params)
    return GeneratedSource(kind="text", params=base)


def ink(image: np.ndarray) -> tuple[float, float, float, float]:
    """字の外形（画素、左・上・右・下）"""
    ys, xs = np.nonzero(image[..., 3] > 8)
    assert len(xs), "何も描かれていない"
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def advance(font_name: str, size: int, word: str) -> float:
    font = QFont(font_name)
    font.setPixelSize(size)
    return QFontMetricsF(font).horizontalAdvance(word)


@pytest.mark.usefixtures("ms_ui_gothic")
class TestFrame:
    def test_the_frame_is_the_advance_times_the_line_height(self) -> None:
        # 字の形を入れ物にすると、左右の余白（MS UI Gothic の 田 で 1 文字あたり約 8%）と
        # ベースラインの下の分だけ入れ物が小さくなる
        image, framed = render_source_framed(text(), *SCREEN)
        assert image is not None and framed is not None
        left, top, right, bottom = framed
        assert right - left == pytest.approx(advance("MS UI Gothic", 60, "田田田"), abs=0.5)
        font = QFont("MS UI Gothic")
        font.setPixelSize(60)
        assert bottom - top == pytest.approx(QFontMetricsF(font).height(), abs=0.5)
        # 中央揃え[中] は枠の中心が置いた位置（画面の中心）に来る
        assert (left + right) / 2 == pytest.approx(SCREEN[0] / 2, abs=0.5)
        assert (top + bottom) / 2 == pytest.approx(SCREEN[1] / 2, abs=0.5)

    def test_the_frame_is_wider_than_the_ink(self) -> None:
        image, framed = render_source_framed(text(), *SCREEN)
        assert image is not None and framed is not None
        ink_left, ink_top, ink_right, ink_bottom = ink(image)
        assert framed[0] < ink_left - 3
        assert framed[2] > ink_right - 1
        assert framed[1] <= ink_top
        assert framed[3] >= ink_bottom

    def test_native_text_has_no_frame(self) -> None:
        # 既定の組み方は今までどおり字の形が入れ物 枠を返すと、既にある作品の
        # 効果の掛かる範囲が変わる
        image, framed = render_source_framed(text(layout="native"), *SCREEN)
        assert image is not None
        assert framed is None

    def test_vertical_aviutl_text_stays_bold(self) -> None:
        # 縦書きは AviUtl2 の太らせ方を通らないので、Qt の太字まで切ると細字で出る
        plain, _ = render_source_framed(text(vertical=True), *SCREEN)
        bold, framed = render_source_framed(text(vertical=True, bold=True), *SCREEN)
        assert plain is not None and bold is not None
        assert int((bold[..., 3] > 128).sum()) > int((plain[..., 3] > 128).sum()) * 1.05
        # 縦書きの枠は測っていない 決まりの分からない枠を返さず、字の形を入れ物にする
        assert framed is None

    def test_shapes_have_no_frame(self) -> None:
        shape = GeneratedSource(kind="shape", params={"shape": "rect"})
        assert render_source_framed(shape, *SCREEN)[1] is None


@pytest.mark.usefixtures("ms_ui_gothic")
class TestBold:
    def test_bold_widens_each_advance_by_a_48th_of_the_size(self) -> None:
        # AviUtl2 の MS UI Gothic 180 太字の枠は 551（細字の送り幅 540 + 1/48 x 3 文字）
        # Qt の太字に任せると 1 文字 1 画素しか広がらず、枠が 543 になる
        _, framed = render_source_framed(text(size=AnimatedValue(180.0), bold=True), 1920, 1080)
        assert framed is not None
        plain = advance("MS UI Gothic", 180, "田田田")
        assert framed[2] - framed[0] == pytest.approx(plain + 3 * 180 / 48, abs=0.5)

    def test_bold_text_stays_centred_in_its_frame(self) -> None:
        # Qt の太字は右へだけ 10 画素ほど太らせるので、中央揃えでも字が 4〜5 画素右へ
        # 寄っていた AviUtl2 の書き出しでは 708..1212 で、画面の中心 960 に対して左右同じ
        image, framed = render_source_framed(text(size=AnimatedValue(180.0), bold=True), 1920, 1080)
        assert image is not None and framed is not None
        left, top, right, bottom = ink(image)
        assert (left + right) / 2 == pytest.approx(960.0, abs=1.0)
        assert right - left == pytest.approx(504.0, abs=2.0)
        # 太らせるのは上へ（ベースラインは動かさない） AviUtl2 は 472..611
        assert top == pytest.approx(472.0, abs=1.0)
        assert bottom == pytest.approx(611.0, abs=1.0)

    def test_bold_grows_right_and_up_only(self) -> None:
        # 両側へ太らせると左の余白が AviUtl2 より 2 画素狭くなり、ベースラインが下がる
        plain, _ = render_source_framed(text(size=AnimatedValue(180.0)), 1920, 1080)
        bold, _ = render_source_framed(text(size=AnimatedValue(180.0), bold=True), 1920, 1080)
        assert plain is not None and bold is not None
        plain_box, bold_box = ink(plain), ink(bold)
        widened = 3 * 180 / 48
        # 枠が太字の分だけ左へ広がるので、字の左端も枠と一緒に半分だけ左へ動く
        assert bold_box[0] == pytest.approx(plain_box[0] - widened / 2, abs=1.0)
        assert bold_box[3] == pytest.approx(plain_box[3], abs=1.0)
        assert bold_box[1] < plain_box[1] - 2


@pytest.mark.usefixtures("ms_ui_gothic")
class TestAviUtlLayout:
    """AviUtl2 に描かせた ``kumiki_p8_t_*`` の見本で測った並べ方"""

    def test_left_top_puts_the_frame_corner_on_the_position(self) -> None:
        # 左寄せ[上] は枠の左上が置いた位置に来る（見本 lt の枠 960..1320 x 540..780）
        # 標準の組み方のように枠を横の中心へ置くと、枠の幅の半分だけ左へずれる
        _, framed = render_source_framed(
            text(text="田田田\n田", size=AnimatedValue(120.0), align="left", valign="top"),
            1920,
            1080,
        )
        assert framed == pytest.approx((960.0, 540.0, 1320.0, 780.0), abs=0.5)

    def test_right_bottom_puts_the_opposite_corner_there(self) -> None:
        # 右寄せ[下] は枠の右下が置いた位置に来る（見本 rb の枠 600..960 x 300..540）
        image, framed = render_source_framed(
            text(text="田田田\n田", size=AnimatedValue(120.0), align="right", valign="bottom"),
            1920,
            1080,
        )
        assert image is not None
        assert framed == pytest.approx((600.0, 300.0, 960.0, 540.0), abs=0.5)
        # 2 行目は枠の右へ揃う（見本では 2 行目の字の右端が 1 行目と同じ 944）
        lower = image[420:540]
        columns = np.flatnonzero((lower[..., 3] > 8).any(axis=0))
        assert columns[-1] + 1 == pytest.approx(944, abs=1)

    def test_spacing_goes_only_between_letters_and_lines(self) -> None:
        # 字間 20・行間 30 で枠は 400 x 270（見本 spacing） 最後の字の後ろや最後の行の
        # 下にも入れると 420 x 300 になり、中央揃えで字が 10 画素ずつ左上へ寄る
        _, framed = render_source_framed(
            text(
                text="田田田\n田",
                size=AnimatedValue(120.0),
                letter_spacing=AnimatedValue(20.0),
                line_spacing=AnimatedValue(30.0),
            ),
            1920,
            1080,
        )
        assert framed == pytest.approx((760.0, 405.0, 1160.0, 675.0), abs=0.5)


class TestLineGap:
    def test_the_line_gap_goes_under_the_line(self) -> None:
        # Arial 120 の枠は AviUtl2 で 139 画素 Qt の行の高さ（134）のままだと枠が低く、
        # 字が 2 画素下がる 余白（hhea の行間 67/2048）は行の下に付く
        if not QFont("Arial").exactMatch():
            pytest.skip("Arial が入っていない")
        image, framed = render_source_framed(
            text(text="AVg", font="Arial", size=AnimatedValue(120.0)), 1920, 1080
        )
        assert image is not None and framed is not None
        assert framed[3] - framed[1] == pytest.approx(138.0, abs=1.0)
        # AviUtl2 の書き出しでは字の上端が 494
        assert ink(image)[1] == pytest.approx(494.0, abs=1.0)


@pytest.mark.usefixtures("ms_ui_gothic")
class TestNativeBold:
    def test_native_bold_text_is_centred(self) -> None:
        # Qt の合成の太字は右へだけ太るので、標準の組み方でも中央揃えの字が
        # 4〜5 画素右へ寄っていた 細字と同じ中心に戻す
        image, framed = render_source_framed(
            text(size=AnimatedValue(180.0), bold=True, layout="native"), 1920, 1080
        )
        assert image is not None and framed is None
        left, _, right, _ = ink(image)
        assert (left + right) / 2 == pytest.approx(960.0, abs=1.0)


class TestGraphemes:
    def test_surrogate_pairs_and_combining_marks_stay_whole(self) -> None:
        # 1 文字ずつ置くので、割れると絵文字や濁点が崩れて別々に並ぶ
        word = "か" + chr(0x3099) + chr(0x1F44D) + "a"
        assert graphemes(word) == ["か" + chr(0x3099), chr(0x1F44D), "a"]


class TestObjectBox:
    def test_the_frame_is_the_object_box(self) -> None:
        image = np.zeros((100, 200, 4), np.uint8)
        image[40:60, 80:120, 3] = 255
        assert _object_box(image, (60.5, 30.0, 140.5, 70.0)) == (60.5, 30.0, 140.5, 70.0)

    def test_ink_outside_the_frame_widens_the_box(self) -> None:
        # 斜体の張り出しや縁取りは枠の外へ出る 枠で切ると、効果の範囲から外れて欠ける
        image = np.zeros((100, 200, 4), np.uint8)
        image[40:60, 50:120, 3] = 255
        assert _object_box(image, (60.0, 30.0, 140.0, 70.0)) == (50.0, 30.0, 140.0, 70.0)

    def test_without_a_frame_the_ink_is_the_box(self) -> None:
        image = np.zeros((100, 200, 4), np.uint8)
        image[40:60, 80:120, 3] = 255
        assert _object_box(image) == (80.0, 40.0, 120.0, 60.0)

    def test_scripts_see_the_frame_as_the_object_size(self) -> None:
        # AviUtl のスクリプトは obj.w を文字の枠の幅として読む 字の形で切ると、
        # 分割したマスや並べる間隔が AviUtl2 より 3〜8% 狭くなる
        image = np.zeros((100, 200, 4), np.uint8)
        image[40:60, 80:120, 3] = 255
        cropped, offset = _object_sized(image, _object_box(image, (60.5, 30.0, 140.5, 70.0)))
        assert cropped.shape[:2] == (40, 81)
        assert offset == pytest.approx((0.5, 0.0))


#: 自分の幅だけ右へ動く AviUtl の obj.w は「オブジェクト自身の幅」
OWN_SIZE = "obj.ox = obj.w\n"


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.mark.usefixtures("ms_ui_gothic")
class TestRenderer:
    def test_a_script_moves_text_by_the_frame_width(self, gl_context: OffscreenGLContext) -> None:
        catalog = ScriptCatalog(roots=())
        catalog.add_text("aviutl:枠の試験.anm:自分の幅", OWN_SIZE)
        set_script_catalog(catalog)
        width, height = SCREEN
        project = Project.create(
            ProjectSettings(width=width, height=height, frame_rate=FrameRate(30))
        )
        track = Track(kind=TrackKind.VIDEO, name="V1")
        project = AddTrack(track).apply(project)
        clip = Clip(timeline_start=0, duration=30, source=text())
        project = AddClip(track.id, clip).apply(project)
        definition = registry.get("aviutl:枠の試験.anm:自分の幅")
        assert definition is not None
        project = AddEffect(clip.id, definition.create()).apply(project)

        renderer = FrameRenderer(project, context=gl_context)
        try:
            moved = renderer.render(0)
        finally:
            renderer.close()
        still, framed = render_source_framed(text(), *SCREEN)
        assert still is not None and framed is not None
        bright = np.nonzero(moved[..., :3].max(axis=2) > 100)[1]
        shift = (bright.min() + bright.max() + 1) / 2 - sum(ink(still)[0::2]) / 2
        assert shift == pytest.approx(framed[2] - framed[0], abs=2.0)
