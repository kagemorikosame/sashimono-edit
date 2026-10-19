"""エフェクトが読む画像（画像合成の絵、縁取りの模様）

絵の置き方は AviUtl2 v2.1.6a に描かせて測った（``kumiki_p5_*`` ``kumiki_p6_*``
``kumiki_p7_blend_*`` 200x200 の 4 色の画像を使った）

- 画像合成は画像の**中心**を絵の中心に合わせ、X と Y でずらす ずらす量は拡大率で縮まない
- ループ画像を切ると画像の外は消える 半透明の所は絵の濃さと掛け算になる
- 縁取りの模様は、縁の分だけ広げた範囲の左上から敷き詰める
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.model import (
    Clip,
    Effect,
    GeneratedSource,
    Project,
    ProjectSettings,
    Scene,
    Timeline,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE
from sashimono.effects.spec import IMAGE_SUFFIXES
from sashimono.engine.decode.image import read_image
from sashimono.engine.decode.probe import STILL_SUFFIXES
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.gpu.images import EffectImages
from sashimono.engine.render import FrameRenderer, image_spans
from sashimono.engine.render.invalidate import image_paths

WIDTH, HEIGHT = 200, 200
RATE = FrameRate(30)

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
WHITE = (255, 255, 255, 255)


def write_png(path: Path, image: np.ndarray) -> Path:
    """RGBA の配列を PNG へ 画像のためだけに Pillow を足さず、入っている Qt で書く"""
    from PySide6.QtGui import QImage

    height, width = image.shape[:2]
    data = np.ascontiguousarray(image, dtype=np.uint8)
    qimage = QImage(data.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
    assert qimage.save(str(path)), f"{path} を書けない"
    return path


def quadrants(path: Path, size: int) -> Path:
    """左上 赤、右上 緑、左下 青、右下 白 の画像 AviUtl2 で測ったのと同じ並び"""
    half = size // 2
    image = np.zeros((size, size, 4), dtype=np.uint8)
    image[:half, :half] = RED
    image[:half, half:] = GREEN
    image[half:, :half] = BLUE
    image[half:, half:] = WHITE
    return write_png(path, image)


def solid(path: Path, colour: tuple[int, int, int, int], size: int = 16) -> Path:
    image = np.zeros((size, size, 4), dtype=np.uint8)
    image[:, :] = colour
    return write_png(path, image)


def white_square(size: int = 100) -> GeneratedSource:
    """中央に置いた白い正方形 画面の 50..150 を占める"""
    return SHAPE.create(shape="rect", width=size, height=size, color=(1.0, 1.0, 1.0, 1.0))


def colour_at(image: np.ndarray, x: int, y: int) -> str:
    """その画素がどの色か 判定を読みやすくするための名前"""
    r, g, b = (int(v) for v in image[y, x, :3])
    if max(r, g, b) < 30:
        return "黒"
    if r > 200 and g < 60 and b < 60:
        return "赤"
    if g > 200 and r < 60 and b < 60:
        return "緑"
    if b > 200 and r < 60 and g < 60:
        return "青"
    if min(r, g, b) > 200:
        return "白"
    return f"その他 {r},{g},{b}"


def blend(path: Path | str, **values: object) -> Effect:
    return registry.require("image_blend").create(image_file=str(path), **values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _project(*clips: Clip) -> Project:
    project = Project.create(ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=RATE))
    track = Track(kind=TrackKind.VIDEO, clips=clips)
    return project.with_timeline(Timeline(rate=project.rate, tracks=(track,)))


@pytest.fixture
def draw(gl: OffscreenGLContext) -> Callable[..., np.ndarray]:
    def render(source: GeneratedSource, effects: tuple[Effect, ...] = ()) -> np.ndarray:
        project = _project(Clip(timeline_start=0, duration=30, source=source, effects=effects))
        renderer = FrameRenderer(project, context=gl)
        try:
            return renderer.render(0)
        finally:
            renderer.close()

    return render


class TestImageBlend:
    def test_the_image_centre_sits_on_the_object_centre(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 左上から敷くと、100 画素の正方形に 100 画素の画像がちょうど重ならない
        # AviUtl2 では 1 枚だけ置いたとき、画像が文字の真ん中に出た
        image = draw(white_square(), (blend(quadrants(tmp_path / "q.png", 100)),))
        assert colour_at(image, 75, 75) == "赤"
        assert colour_at(image, 125, 75) == "緑"
        assert colour_at(image, 75, 125) == "青"
        assert colour_at(image, 125, 125) == "白"

    def test_the_image_only_paints_inside_the_object(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 濃さは絵と画像の掛け算 絵の外まで画像で埋めると、文字が四角い板になる
        image = draw(white_square(), (blend(quadrants(tmp_path / "q.png", 100)),))
        assert colour_at(image, 20, 20) == "黒"

    def test_without_loop_the_outside_of_the_image_disappears(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # AviUtl2 で ループ画像=0 にすると、画像から外れた両端の文字が消えた
        small = quadrants(tmp_path / "small.png", 40)
        tiled = draw(white_square(), (blend(small, loop=True),))
        single = draw(white_square(), (blend(small, loop=False),))
        assert colour_at(tiled, 65, 65) == "白"
        assert colour_at(single, 65, 65) == "黒"
        assert colour_at(single, 90, 90) == "赤"

    def test_positive_y_moves_the_image_up_and_x_to_the_right(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # Y は上が正（AviUtl の 下が正 は読み込みで反転する） 逆にすると、
        # AviUtl2 で下へ 30 ずらした模様が上へずれる
        picture = quadrants(tmp_path / "q.png", 100)
        moved = draw(white_square(), (blend(picture, offset_x=10, offset_y=10),))
        # 赤と青の境目が 100 から 90 へ上がり、赤と緑の境目が 100 から 110 へ動く
        assert colour_at(moved, 75, 95) == "青"
        assert colour_at(moved, 105, 75) == "赤"

    def test_zoom_shrinks_the_image_around_its_centre(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 拡大率 50 で 1 枚が 50 画素になる 中心からの位置で縮めないと、
        # AviUtl2 の拡大率 50 の見本と色の変わり目が合わない
        picture = quadrants(tmp_path / "q.png", 100)
        assert colour_at(draw(white_square(), (blend(picture),)), 60, 60) == "赤"
        assert colour_at(draw(white_square(), (blend(picture, zoom=50),)), 60, 60) == "白"

    def test_a_missing_image_leaves_the_picture_alone(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 消したり動かしたりした画像で文字まで消すと、何が起きたか分からない
        plain = draw(white_square())
        missing = draw(white_square(), (blend(tmp_path / "無い.png"),))
        assert np.array_equal(plain, missing)

    def test_a_clip_without_an_image_does_not_inherit_the_previous_one(
        self, gl: OffscreenGLContext, tmp_path: Path
    ) -> None:
        # uniform の値はプログラムに残る 画像の大きさを毎回渡さないと、
        # 画像を外したクリップが前に描いたクリップの画像で塗られる
        picture = quadrants(tmp_path / "q.png", 100)
        project = _project(
            Clip(timeline_start=0, duration=1, source=white_square(), effects=(blend(picture),)),
            Clip(timeline_start=1, duration=1, source=white_square(), effects=(blend(""),)),
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            assert colour_at(renderer.render(0), 75, 75) == "赤"
            assert colour_at(renderer.render(1), 75, 75) == "白"
        finally:
            renderer.close()

    def test_rewriting_the_file_redraws_with_the_new_image(
        self, gl: OffscreenGLContext, tmp_path: Path
    ) -> None:
        # 1 度読んだ画像を持ち続けると、別のソフトで描き直しても古い絵のまま
        path = solid(tmp_path / "塗り.png", RED)
        project = _project(
            Clip(timeline_start=0, duration=30, source=white_square(), effects=(blend(path),))
        )
        renderer = FrameRenderer(project, context=gl)
        try:
            assert colour_at(renderer.render(0), 100, 100) == "赤"
            solid(path, GREEN)
            _touch_later(path)
            assert renderer.stale_images() == {str(path)}
            assert colour_at(renderer.render(0), 100, 100) == "緑"
        finally:
            renderer.close()


def disc() -> GeneratedSource:
    """中央の白い円 入れ物の四角（50..150）の角は透明"""
    return SHAPE.create(shape="ellipse", width=100, height=100, color=(1.0, 1.0, 1.0, 1.0))


class TestBlendModes:
    """合成モード AviUtl2 v2.1.6a の一覧の 5 つ（``kumiki_p7_blend_*`` で測った）"""

    def test_front_puts_the_image_over_the_whole_box(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 前方から合成 画像は文字の形で切り抜かれず、入れ物の四角いっぱいに文字の上へ出る
        image = draw(disc(), (blend(quadrants(tmp_path / "q.png", 100), blend="front"),))
        assert colour_at(image, 55, 55) == "赤", "円の外の角まで画像が出ていない"
        assert colour_at(image, 75, 125) == "青", "文字の上に画像が来ていない"
        assert colour_at(image, 20, 20) == "黒", "入れ物の外まで画像が広がった"

    def test_back_puts_the_image_behind(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 後方から合成 画像は入れ物の四角を埋め、文字（ここでは円）はその上に残る
        image = draw(disc(), (blend(quadrants(tmp_path / "q.png", 100), blend="back"),))
        assert colour_at(image, 55, 55) == "赤"
        assert colour_at(image, 75, 125) == "白", "絵が画像の下に隠れた"
        assert colour_at(image, 20, 20) == "黒"

    def test_luma_multiplies_the_alpha_by_the_image_brightness(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 輝度をアルファ値として乗算 色は絵のまま、濃さに画像の明るさ（Rec.601）を掛ける
        # AviUtl2 では白い文字が 赤 74・緑 150・青 29 の灰色になった
        def centre_value(colour: tuple[int, int, int, int], name: str) -> int:
            picture = solid(tmp_path / f"{name}.png", colour)
            image = draw(white_square(), (blend(picture, blend="luma_multiply"),))
            r, g, b = (int(v) for v in image[100, 100, :3])
            assert r == g == b, "色が絵の色（白）から変わった"
            return r

        white = centre_value(WHITE, "白")
        green = centre_value(GREEN, "緑")
        red = centre_value(RED, "赤")
        blue = centre_value(BLUE, "青")
        assert white > green > red > blue > 0

    def test_luma_alpha_sets_the_alpha_even_where_the_object_was_see_through(
        self, gl: OffscreenGLContext, tmp_path: Path
    ) -> None:
        # 輝度をアルファ値として上書き 乗算と違い、元の濃さは見ない 円の中は
        # 乗算と同じ絵になり、違いは元が透けていた所（入れ物の角）にだけ出る
        # 乗算と同じ作りに崩れると、角の下の灰色がそのまま透けて見える
        #
        # 透けていた所の色は AviUtl2 で確かめていない（黒の上でしか描かせて
        # いない） いまはこちらの透明な画素の色（黒）が、その濃さで出る
        picture = solid(tmp_path / "白.png", WHITE)
        grey = SHAPE.create(shape="rect", width=200, height=200, color=(0.5, 0.5, 0.5, 1.0))

        def render(mode: str) -> np.ndarray:
            project = Project.create(ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=RATE))
            below = Track(
                kind=TrackKind.VIDEO, clips=(Clip(timeline_start=0, duration=30, source=grey),)
            )
            above = Track(
                kind=TrackKind.VIDEO,
                clips=(
                    Clip(
                        timeline_start=0,
                        duration=30,
                        source=disc(),
                        effects=(blend(picture, blend=mode),),
                    ),
                ),
            )
            project = project.with_timeline(Timeline(rate=RATE, tracks=(below, above)))
            renderer = FrameRenderer(project, context=gl)
            try:
                return renderer.render(0)
            finally:
                renderer.close()

        multiplied = render("luma_multiply")
        overwritten = render("luma_alpha")
        assert np.array_equal(multiplied[100, 100], overwritten[100, 100])
        assert colour_at(multiplied, 55, 55).startswith("その他"), "乗算で角の下が見えない"
        assert colour_at(overwritten, 55, 55) == "黒", "上書きで角が透けたまま"


class TestGradientMapPattern:
    def test_brightness_picks_a_column_of_the_middle_row(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 暗い所は左端、明るい所は右端の色 行は縦の真ん中 AviUtl2 に横で色相、
        # 縦で明るさの変わる画像を渡して、出た色から読んだ
        pattern = np.zeros((3, 4, 4), dtype=np.uint8)
        pattern[:, :] = GREEN
        pattern[1, :2] = RED
        pattern[1, 2:] = BLUE
        path = write_png(tmp_path / "帯.png", pattern)
        mapping = registry.require("gradient_map").create(pattern=str(path))
        black = SHAPE.create(shape="rect", width=100, height=100, color=(0.0, 0.0, 0.0, 1.0))
        assert colour_at(draw(black, (mapping,)), 100, 100) == "赤"
        assert colour_at(draw(white_square(), (mapping,)), 100, 100) == "青"

    def test_without_a_pattern_the_two_colours_are_used(
        self, draw: Callable[..., np.ndarray]
    ) -> None:
        # 模様の欄が空のときに 2 色を使わないと、明部の色を赤にしても白い所が赤く染まらない
        mapping = registry.require("gradient_map").create(light_color=(1.0, 0.0, 0.0, 1.0))
        assert colour_at(draw(white_square(), (mapping,)), 100, 100) == "赤"


class TestBorderPattern:
    def test_the_pattern_replaces_the_border_colour(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # AviUtl2 で 縁色 を赤にして模様を付けても、縁は模様の色のまま（赤に染まらない）
        pattern = solid(tmp_path / "緑.png", GREEN)
        border = registry.require("border").create(
            width=8, color=(1.0, 0.0, 0.0, 1.0), pattern=str(pattern)
        )
        image = draw(white_square(), (border,))
        assert colour_at(image, 100, 154) == "緑"
        assert colour_at(image, 100, 100) == "白", "中身まで模様で塗られている"

    def test_the_pattern_starts_at_the_outer_top_left_of_the_border(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 正方形は 50..150、縁 8 で外形は 42 から 模様の 1 マスは 10 画素
        # 中心から敷く（画像合成と同じ）と、左上の角が赤ではなく緑になる
        # AviUtl2 の図形 200 で縁 10 と 30 を描かせると、起点はそれぞれ 850 と 830 だった
        pattern = quadrants(tmp_path / "q.png", 20)
        border = registry.require("border").create(width=8, pattern=str(pattern))
        image = draw(white_square(), (border,))
        assert colour_at(image, 45, 45) == "赤"
        assert colour_at(image, 55, 45) == "緑"
        assert colour_at(image, 45, 55) == "青"

    def test_without_a_pattern_the_colour_is_used(
        self, draw: Callable[..., np.ndarray], tmp_path: Path
    ) -> None:
        # 模様が無いときに縁色を使わないと、縁取りの色を赤にしても縁が別の色で出る
        border = registry.require("border").create(width=8, color=(1.0, 0.0, 0.0, 1.0))
        assert colour_at(draw(white_square(), (border,)), 100, 154) == "赤"


def _touch_later(path: Path) -> None:
    """更新時刻を確実に進める 書いた直後だと、時刻の刻みによっては前と同じになる"""
    status = path.stat()
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 2_000_000_000))


@dataclass
class _FakeTexture:
    width: int
    height: int
    handle: int = 1
    released: bool = False

    def release(self) -> None:
        self.released = True


@dataclass
class _Reader:
    calls: list[Path] = field(default_factory=list)

    def __call__(self, path: Path) -> np.ndarray | None:
        self.calls.append(path)
        return read_image(path)


def _images(reader: _Reader) -> EffectImages:
    return EffectImages(
        reader=reader, upload=lambda image: _FakeTexture(image.shape[1], image.shape[0])
    )


class TestTheImageStore:
    """GPU に置いた画像の持ち方 GL を使わずに確かめる"""

    def test_an_unchanged_file_is_read_once(self, tmp_path: Path) -> None:
        # 描くたびに読み直すと、1 フレームごとにデコードと転送が走る
        path = str(solid(tmp_path / "a.png", RED))
        reader = _Reader()
        images = _images(reader)
        first = images.get(path)
        assert images.get(path) is first
        assert len(reader.calls) == 1

    def test_a_rewritten_file_is_read_again_and_the_old_one_released(self, tmp_path: Path) -> None:
        # 読み直さないと、模様を描き直して保存しても前の模様のまま出る
        # 古い方を手放さないと、描き直すたびに GPU のメモリが減っていく
        path = solid(tmp_path / "a.png", RED)
        reader = _Reader()
        images = _images(reader)
        first = images.get(str(path))
        solid(path, GREEN, size=32)
        _touch_later(path)
        second = images.get(str(path))
        assert isinstance(first, _FakeTexture) and first.released, "古い画像を掴んだまま"
        assert second is not None and second.width == 32

    def test_a_change_is_reported_once_even_if_drawing_reloaded_it_first(
        self, tmp_path: Path
    ) -> None:
        # 描き直した側が先に読み直しても、先読みした別のフレームは古いまま
        # 変化を 1 度は伝えないと、そこが捨てられずに残る
        # 2 回目も返すと、描き直したばかりの絵まで捨て続ける
        path = solid(tmp_path / "a.png", RED)
        images = _images(_Reader())
        images.get(str(path))
        assert images.stale() == frozenset()
        solid(path, GREEN)
        _touch_later(path)
        images.get(str(path))
        assert images.stale() == {str(path)}
        assert images.stale() == frozenset()

    def test_a_missing_file_is_remembered_and_its_arrival_reported(self, tmp_path: Path) -> None:
        # 無いファイルを毎フレーム探しに行って読もうとしない 置かれたら気付く
        path = tmp_path / "後で置く.png"
        reader = _Reader()
        images = _images(reader)
        assert images.get(str(path)) is None
        assert images.missing == {str(path)}
        assert reader.calls == []
        solid(path, RED)
        assert images.stale() == {str(path)}
        assert images.get(str(path)) is not None
        assert images.missing == frozenset()

    def test_a_swapped_in_file_with_the_same_time_and_size_is_noticed(self, tmp_path: Path) -> None:
        # 別のファイルへ書いてから差し替える保存で、更新時刻と大きさが元と同じ
        # 時刻と大きさだけで見ると気付けず、古い模様が残り続ける
        path = solid(tmp_path / "a.png", RED)
        images = _images(_Reader())
        images.get(str(path))
        before = path.stat()
        replacement = solid(tmp_path / "b.png", GREEN)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert replacement.stat().st_size == before.st_size
        replacement.replace(path)
        if path.stat().st_ino == before.st_ino or path.stat().st_ino == 0:
            pytest.skip("このファイルシステムはファイルの番号を返さない")
        assert images.stale() == {str(path)}

    def test_a_missing_image_is_reported_again_after_release(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 伝えたことを覚えたままだと、次のレンダラで同じ画像が無くても黙る
        path = str(tmp_path / "無い.png")
        images = _images(_Reader())
        with caplog.at_level("WARNING", logger="sashimono.engine.gpu.images"):
            images.get(path)
            images.release()
            images.get(path)
        assert sum("見つからない" in record.getMessage() for record in caplog.records) == 2

    def test_images_no_longer_used_are_let_go(self, tmp_path: Path) -> None:
        # 模様を差し替えるたびに前の画像を GPU に残すと、長く編集するほどメモリを食う
        old = solid(tmp_path / "前.png", RED)
        new = solid(tmp_path / "後.png", GREEN)
        images = _images(_Reader())
        first = images.get(str(old))
        second = images.get(str(new))
        images.retain({str(new)})
        assert isinstance(first, _FakeTexture) and first.released
        assert isinstance(second, _FakeTexture) and not second.released
        # 見張りも外れる 使われていない画像を 1 秒おきに見に行かない
        solid(old, BLUE)
        _touch_later(old)
        assert images.stale() == frozenset()

    def test_an_unreadable_file_is_not_an_error(self, tmp_path: Path) -> None:
        # 壊れた画像 1 枚でプレビューも書き出しも止めない
        path = tmp_path / "壊れた.png"
        path.write_bytes(b"not a png")
        images = _images(_Reader())
        assert images.get(str(path)) is None

    def test_an_empty_path_means_no_image(self) -> None:
        # 空の欄をパスとして探すと、模様を付けていない縁取りのたびに
        # ログへ「画像が見つからない」が書かれる
        reader = _Reader()
        assert _images(reader).get("") is None
        assert reader.calls == []


class TestReadingImages:
    def test_the_picker_offers_what_stills_accept(self) -> None:
        # 選ばせる拡張子だけ広げると、選べるのに読めない画像が出る
        assert set(IMAGE_SUFFIXES) == set(STILL_SUFFIXES)

    def test_alpha_is_kept_straight(self, tmp_path: Path) -> None:
        # 半透明の白を事前乗算で読むと灰色になり、AviUtl2 の 128 の灰色がさらに暗くなる
        image = np.zeros((4, 4, 4), dtype=np.uint8)
        image[:, :] = (255, 255, 255, 128)
        read = read_image(write_png(tmp_path / "half.png", image))
        assert read is not None
        assert tuple(int(v) for v in read[0, 0]) == (255, 255, 255, 128)

    def test_the_first_row_is_the_top(self, tmp_path: Path) -> None:
        # 上下が逆だと、画像合成の赤と青が入れ替わる
        read = read_image(quadrants(tmp_path / "q.png", 10))
        assert read is not None
        assert tuple(int(v) for v in read[0, 0]) == RED
        assert tuple(int(v) for v in read[9, 0]) == BLUE


class TestWhatToRedraw:
    def _clip(self, start: int, pattern: str) -> Clip:
        border = registry.require("border").create(pattern=pattern)
        return Clip(timeline_start=start, duration=10, effects=(border,))

    def test_only_clips_reading_the_image_are_dropped(self) -> None:
        # 画像を書き換えてもプロジェクトは変わらない 編集の差分からは出てこない
        project = _project(self._clip(0, "a.png"), self._clip(20, "b.png"))
        assert image_spans(project, {"a.png"}).spans == ((0, 10),)

    def test_a_scene_that_reads_the_image_drops_where_it_is_placed(self) -> None:
        # シーンの中の位置で捨てると、シーンを置いた 30 フレーム目からの先読みが残り、
        # 模様を差し替えてもその場面だけ古い模様が映る
        inner = Timeline(
            rate=RATE, tracks=(Track(kind=TrackKind.VIDEO, clips=(self._clip(0, "a.png"),)),)
        )
        scene = Scene(name="中身", timeline=inner)
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(Clip(timeline_start=30, duration=15, scene_id=scene.id),),
        )
        project = Project(
            settings=ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=RATE),
            timeline=Timeline(rate=RATE, tracks=(track,)),
            scenes=(scene,),
        )
        assert image_spans(project, {"a.png"}).spans == ((30, 45),)

    def test_the_scene_after_a_transition_is_looked_at_too(self) -> None:
        # 場面切り替えの後の場面だけが画像を読むクリップ 前の場面だけを見ると
        # 捨てる範囲が空になり、先読みした古い模様が残る
        border = registry.require("border").create(pattern="a.png")
        clip = Clip(timeline_start=5, duration=10, after_effects=(border,))
        assert image_spans(_project(clip), {"a.png"}).spans == ((5, 15),)

    def test_every_image_the_project_reads_is_kept(self) -> None:
        # シーンの中と場面切り替えの後の場面も数える 漏らすと、使っている画像を
        # 手放して毎回読み直すことになる
        inner = Timeline(
            rate=RATE, tracks=(Track(kind=TrackKind.VIDEO, clips=(self._clip(0, "中.png"),)),)
        )
        scene = Scene(name="中身", timeline=inner)
        after = Clip(
            timeline_start=40,
            duration=5,
            after_effects=(registry.require("border").create(pattern="後.png"),),
        )
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(
                self._clip(0, "表.png"),
                Clip(timeline_start=20, duration=5, scene_id=scene.id),
                after,
            ),
        )
        project = Project(
            settings=ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=RATE),
            timeline=Timeline(rate=RATE, tracks=(track,)),
            scenes=(scene,),
        )
        assert image_paths(project) == {"表.png", "中.png", "後.png"}

    def test_a_renderer_forgets_images_the_project_dropped(
        self, gl: OffscreenGLContext, tmp_path: Path
    ) -> None:
        # 編集で使われなくなった画像は、レンダラを閉じるまで待たずに手放す
        path = solid(tmp_path / "塗り.png", RED)
        with_image = _project(
            Clip(timeline_start=0, duration=30, source=white_square(), effects=(blend(path),))
        )
        renderer = FrameRenderer(with_image, context=gl)
        try:
            renderer.render(0)
            renderer.set_project(
                _project(Clip(timeline_start=0, duration=30, source=white_square()))
            )
            solid(path, GREEN)
            _touch_later(path)
            assert renderer.stale_images() == frozenset(), "使わない画像を持ち続けている"
        finally:
            renderer.close()

    def test_nothing_changed_drops_nothing(self) -> None:
        # 今は使っていない画像（前に読んだ物）が書き換わっただけで範囲を返すと、関係の無い
        # ファイルを保存するたびに先読みした絵が捨てられ、プレビューが描き直しで引っかかる
        project = _project(self._clip(0, "a.png"))
        assert not image_spans(project, set())
        assert not image_spans(project, {"別の.png"})
