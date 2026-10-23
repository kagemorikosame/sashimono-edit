"""書き出しの色変換を GPU で行う道（#372）

GPU で変換できる機械では GPU で、できない機械（GPU の無い CI・駆動が断った・本人が切った）
では今の CPU の道で書く どちらの道でも、コマの順と色が変わってはいけない
前半は GL を使わない（偽の描画係で、道の選び方と順番と片付けを見る）
後半は実際に書き出して、CPU の道と同じ動画になるかを見る
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import av
import av.video.frame
import av.video.stream
import numpy as np
import pytest
from OpenGL.error import NullFunctionError

from sashimono.core.commands import Document, insert_media
from sashimono.core.model import Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.colorspace import bt709_planes
from sashimono.engine.decode import probe_media
from sashimono.engine.encode import ExportError, ExportSettings, export_project, exporter
from sashimono.engine.encoders import usable
from sashimono.engine.gpu import ShaderError, yuv
from sashimono.engine.gpu.yuv import COLOR_TOLERANCE
from sashimono.engine.render import FrameRenderer
from tests.media_fixtures import SampleMedia

WIDTH, HEIGHT = 32, 16


def _picture(frame_number: int) -> np.ndarray:
    """コマごとに違う RGBA 番号が違えば、明るさも色も違う"""
    image = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    image[..., 0] = (frame_number * 37) % 256
    image[..., 1] = np.arange(WIDTH, dtype=np.uint8)[None, :] * 8
    image[..., 2] = 255 - (frame_number * 37) % 256
    image[..., 3] = 255
    return image


def _black(frame_number: int) -> np.ndarray:
    """暗転の絵 どのコマも同じ黒 色差の間引き方がどう違っても、変換の結果は変わらない"""
    image = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    image[..., 3] = 255
    return image


def _shifted(image: np.ndarray, pixel_format: str) -> np.ndarray:
    """1 行ずれた絵から変換する駆動の真似 縦の端の扱いや面の境目を間違えるとこうなる

    単色の絵では正しい変換と 1 画素も違わないので、単色のコマで比べても見つからない
    """
    return bt709_planes(np.roll(image, 1, axis=0), pixel_format)


class _Scope:
    """GL コンテキストの代わり 入った回数だけ数える"""

    def __init__(self) -> None:
        self.depth = 0

    def __enter__(self) -> _Scope:
        self.depth += 1
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.depth -= 1

    def release(self) -> None:
        """持っていないので何もしない"""


class _Compositor:
    def __init__(self, picture: Callable[[int], np.ndarray]) -> None:
        self.current = -1
        self._picture = picture

    def resolved_texture(self, *, straight: bool = False) -> int:
        # テクスチャの番号の代わりに、合成したコマの番号を渡す
        return self.current

    def read(self, *, straight: bool = False) -> np.ndarray:
        return self._picture(self.current)


class _Renderer:
    """FrameRenderer の代わり CPU の道（render）を通ったコマを覚える"""

    def __init__(self, picture: Callable[[int], np.ndarray] = _picture) -> None:
        self.context = _Scope()
        self.compositor = _Compositor(picture)
        self.rendered: list[int] = []
        self._picture = picture

    def compose(self, frame: int) -> None:
        assert self.context.depth > 0, "GL コンテキストの外で合成した"
        self.compositor.current = frame

    def render(self, frame: int) -> np.ndarray:
        self.rendered.append(frame)
        return self._picture(frame)


class _Texture:
    """Texture の代わり 上げた絵を、負の番号で読み戻しの偽物へ預ける

    合成したコマの番号（0 以上）と重ならないので、偽物はどちらの絵を変換するかを番号で見分ける
    作った番号は返すまで ``readback.live_textures`` に残る 返し忘れはそこで分かる
    """

    def __init__(self, readback: _Readback, width: int, height: int, *, srgb: bool) -> None:
        # 合成の結果（resolved_texture）は符号化済みの RGBA8 sRGB で上げると、読む時に
        # リニアへ戻されて試験の絵が別の色になる
        assert not srgb, "試験の絵を sRGB で上げた"
        self._readback = readback
        readback.textures_made += 1
        self.handle = -readback.textures_made
        readback.live_textures.add(self.handle)

    def upload(self, image: np.ndarray) -> None:
        if self._readback.refuse_upload:
            raise ShaderError("テクスチャへ上げられない")
        self._readback.uploads[self.handle] = image

    def release(self) -> None:
        self._readback.live_textures.discard(self.handle)
        self._readback.uploads.pop(self.handle, None)
        if self._readback.refuse_texture_release:
            raise ShaderError("テクスチャを返せない")


class _TextureMaker:
    """exporter が使う ``Texture`` の代わり 作る口（``Texture(...)``）も ``from_array`` も受ける"""

    def __init__(self, readback: _Readback) -> None:
        self._readback = readback

    def __call__(self, width: int, height: int, *, srgb: bool = True) -> _Texture:
        return _Texture(self._readback, width, height, srgb=srgb)

    def from_array(self, image: np.ndarray, *, srgb: bool = True) -> _Texture:
        # 本物と同じく、作ってから上げる 上げる所で断られると作った 1 枚が手元に残る
        height, width = image.shape[:2]
        texture = self(width, height, srgb=srgb)
        texture.upload(image)
        return texture


class _Readback:
    """YuvReadback の代わり 本物と同じく、PBO の数だけ溜めて先に入れた物から出す"""

    def __init__(
        self,
        pixel_format: str,
        *,
        buffers: int = 2,
        skew: int = 0,
        convert: Callable[[np.ndarray, str], np.ndarray] = bt709_planes,
        picture: Callable[[int], np.ndarray] = _picture,
    ) -> None:
        self.pixel_format = pixel_format
        self.buffers = buffers
        #: 変換の結果に足す値 駆動が食い違ったときの真似
        self.skew = skew
        #: GPU の変換の代わり 既定は CPU の変換そのもの
        self.convert = convert
        self.picture = picture
        #: 上げた試験の絵 :class:`_Texture` が番号を付けて入れる
        self.uploads: dict[int, np.ndarray] = {}
        #: 作ってまだ返していないテクスチャの番号と、作った数
        self.live_textures: set[int] = set()
        self.textures_made = 0
        #: 真ならテクスチャへ上げる所・テクスチャを返す所で駆動が断る
        self.refuse_upload = False
        self.refuse_texture_release = False
        self._pending: deque[tuple[int, tuple[int, int]]] = deque()
        self.released = False
        self.pushed: list[int] = []

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def full(self) -> bool:
        return len(self._pending) >= self.buffers

    def push(self, texture: int, tag: tuple[int, int]) -> None:
        assert not self.released, "返した後に使った"
        assert not self.full, "受け取らずに上書きした"
        self.pushed.append(texture)
        self._pending.append((texture, tag))

    def pop(self) -> tuple[tuple[int, int], np.ndarray]:
        texture, tag = self._pending.popleft()
        image = self.uploads[texture] if texture < 0 else self.picture(texture)
        planes = self.convert(image, self.pixel_format).astype(np.int16) + self.skew
        return tag, np.clip(planes, 0, 255).astype(np.uint8)

    def release(self) -> None:
        self._pending.clear()
        self.released = True


def _source(
    monkeypatch: pytest.MonkeyPatch,
    readback: _Readback | None,
    *,
    pixel_format: str | None = "yuv420p",
    picture: Callable[[int], np.ndarray] = _picture,
) -> tuple[exporter._FrameSource, _Renderer, list[str]]:
    opened: list[str] = []

    def fake_open(width: int, height: int, fmt: str) -> _Readback | None:
        opened.append(fmt)
        return readback

    monkeypatch.setattr(exporter, "open_readback", fake_open)
    if readback is not None:
        monkeypatch.setattr(exporter, "Texture", _TextureMaker(readback))
    renderer = _Renderer(picture)
    source = exporter._FrameSource(cast("FrameRenderer", renderer), pixel_format, WIDTH, HEIGHT)
    return source, renderer, opened


def _run(source: exporter._FrameSource, frames: int) -> list[tuple[int, int, np.ndarray]]:
    seats = [0]

    def reserve() -> None:
        seats[0] += 1

    items: list[tuple[int, int, np.ndarray]] = []
    for index in range(frames):
        items.extend(source.frame(index, 100 + index, reserve))
    items.extend(source.drain(reserve))
    # 1 コマにつき席を 1 つ取る 取らずに渡すと、選んだ深さより多くの絵を抱える
    assert seats[0] == len(items)
    return items


class TestChoosingTheWay:
    """GPU を使えないときは今の CPU の道で書く（GPU の無い CI もここを通る）"""

    def test_turned_off_never_asks_the_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 切った人に GPU を使うと、駆動の不具合を避けたくて切った人の書き出しが壊れたまま
        source, renderer, opened = _source(monkeypatch, _Readback("yuv420p"), pixel_format=None)
        items = _run(source, 3)
        assert opened == []
        assert not source.on_gpu
        assert renderer.rendered == [100, 101, 102]
        assert all(image.ndim == 3 for _i, _n, image in items)

    def test_a_refused_gpu_falls_back_to_the_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # GPU の無い機械や駆動が断ったときに書き出しごと失敗すると、今まで書き出せた人が書けない
        source, renderer, opened = _source(monkeypatch, None)
        items = _run(source, 4)
        assert opened == ["yuv420p"]
        assert not source.on_gpu
        assert [(index, number) for index, number, _image in items] == [
            (0, 100),
            (1, 101),
            (2, 102),
            (3, 103),
        ]
        assert renderer.rendered == [100, 101, 102, 103]

    def test_a_driver_without_gl_functions_is_a_fallback_not_a_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # GL の関数が引けない機械（GPU の無い仮想機械）で投げたままにすると、書き出しが始まらない
        def refuse(*_arguments: object, **_options: object) -> None:
            raise NullFunctionError("glGenBuffers")

        monkeypatch.setattr(yuv, "YuvReadback", refuse)
        assert yuv.open_readback(64, 64, "yuv420p") is None


class TestOrder:
    """GPU の道でも、コマは合成した順に 1 枚ずつ出る"""

    @pytest.mark.parametrize("buffers", [1, 2, 3])
    def test_frames_keep_their_order(self, monkeypatch: pytest.MonkeyPatch, buffers: int) -> None:
        # 読み戻しの遅れの間に順が入れ替わると、動画のコマの並びが崩れる
        readback = _Readback("yuv420p", buffers=buffers)
        source, renderer, _opened = _source(monkeypatch, readback)
        items = _run(source, 7)
        assert [(index, number) for index, number, _image in items] == [
            (index, 100 + index) for index in range(7)
        ]
        # 中身もそのコマの物 番号だけ合っていて絵がずれると、動きが 1 コマ遅れる
        for _index, number, image in items:
            assert image.ndim == 2
            assert np.array_equal(image, bt709_planes(_picture(number), "yuv420p"))
        assert renderer.rendered == []
        assert source.on_gpu

    def test_a_disagreeing_driver_writes_every_frame_on_the_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 手元で確かめていない駆動で色が食い違ったら、色のずれた動画を黙って書かない
        readback = _Readback("yuv420p", skew=COLOR_TOLERANCE + 1)
        source, renderer, _opened = _source(monkeypatch, readback)
        items = _run(source, 4)
        assert readback.released
        assert not source.on_gpu
        assert all(image.ndim == 3 for _i, _n, image in items)
        assert [number for _i, number, _image in items] == [100, 101, 102, 103]
        # 合成する前に決めるので、どのコマも 1 回だけ CPU の道で合成する
        # 2 回合成すると動きの記憶がずれる
        assert renderer.rendered == [100, 101, 102, 103]
        assert readback.pushed == [-1], "合成したコマを食い違った GPU へ渡した"

    def test_a_black_opening_still_catches_a_misplaced_conversion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 暗転で始まる書き出しを最初のコマで比べると差が 0 で通り、面の境目や縦の端を
        # 間違える駆動でも、残りのコマ全部を色のずれたまま書いてしまう
        readback = _Readback("yuv420p", convert=_shifted, picture=_black)
        source, renderer, _opened = _source(monkeypatch, readback, picture=_black)
        items = _run(source, 3)
        assert readback.released
        assert not source.on_gpu
        assert all(image.ndim == 3 for _i, _n, image in items)
        assert renderer.rendered == [100, 101, 102]

    def test_the_check_picture_is_the_size_of_the_export(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # yuv420p の U と V の面の境目は大きさで動く 小さな絵で確かめると、書き出しの
        # 大きさでだけ境目を間違える駆動を見逃す
        sizes: list[tuple[int, ...]] = []
        original = exporter._check_picture

        def spy(width: int, height: int) -> np.ndarray:
            image = original(width, height)
            sizes.append(image.shape)
            return image

        monkeypatch.setattr(exporter, "_check_picture", spy)
        _source(monkeypatch, _Readback("yuv420p"))
        assert sizes == [(HEIGHT, WIDTH, 4)]

    def test_the_check_picture_shows_chroma_and_row_mistakes(self) -> None:
        # 試験の絵が平らだと、色差の間引きや 1 行のずれがあっても差が出ず、比べても素通りする
        image = exporter._check_picture(64, 48)
        right = bt709_planes(image, "yuv420p").astype(np.int16)
        assert int(np.abs(_shifted(image, "yuv420p") - right).max()) > COLOR_TOLERANCE
        assert np.array_equal(image, exporter._check_picture(64, 48)), "毎回違う絵になった"

    def test_a_driver_that_fails_the_check_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 確かめる途中で駆動が断ったときに投げたままにすると、CPU なら書けた書き出しが始まらない
        readback = _Readback("yuv420p")

        def refuse() -> tuple[tuple[int, int], np.ndarray]:
            raise ShaderError("読み戻しのバッファを開けない")

        monkeypatch.setattr(readback, "pop", refuse)
        source, renderer, _opened = _source(monkeypatch, readback)
        assert not source.on_gpu
        assert readback.released
        assert readback.uploads == {}, "試験の絵のテクスチャを返していない"
        _run(source, 2)
        assert renderer.rendered == [100, 101]

    def test_a_disagreement_within_rounding_stays_on_the_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 丸めの 1 で GPU をやめると、どの機械でも速くならない
        readback = _Readback("yuv420p", skew=COLOR_TOLERANCE)
        source, _renderer, _opened = _source(monkeypatch, readback)
        _run(source, 3)
        assert source.on_gpu


class TestCleanup:
    """中止や失敗で抜けても、読み戻しのバッファを返す"""

    def test_release_with_frames_in_flight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 受け取らないまま抜けたときに返さないと、中止のたびに GPU のメモリが減る
        readback = _Readback("yuv420p")
        source, renderer, _opened = _source(monkeypatch, readback)
        for index in range(3):
            source.frame(index, index, lambda: None)
        assert readback.pending > 0
        source.release()
        source.release()
        assert readback.released
        assert renderer.context.depth == 0

    @pytest.mark.parametrize("where", ["frame", "drain"])
    def test_a_readback_failure_mid_way_is_a_signal_to_start_again(
        self, monkeypatch: pytest.MonkeyPatch, where: str
    ) -> None:
        # 素の GL の例外のまま出ると、CPU でなら書ける書き出しが丸ごと失敗する
        # 書き出しの側が見分けられる合図にして、色変換を CPU にして頭から書き直させる
        readback = _Readback("yuv420p")
        popped = [0]
        real_pop = readback.pop

        def failing() -> tuple[tuple[int, int], np.ndarray]:
            popped[0] += 1
            # 1 回目は始める前の試験の絵 2 回目は 2 コマ目を入れて溜まりきったときの実のコマ
            # 3 回目は drain で受け取る最後のコマ
            if popped[0] >= (2 if where == "frame" else 3):
                raise ShaderError("読み戻しのバッファを開けない")
            return real_pop()

        monkeypatch.setattr(readback, "pop", failing)
        source, _renderer, _opened = _source(monkeypatch, readback)
        assert source.on_gpu
        with pytest.raises(exporter._GpuColorError, match="途中で使えなくなった"):
            for index in range(2):
                source.frame(index, index, lambda: None)
            list(source.drain(lambda: None))
        source.release()
        assert readback.released

    def test_a_release_refused_while_failing_is_only_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 失敗で抜ける途中の片付けで投げると、元の失敗（書き直しの合図・中止など）が
        # 片付けの GL のエラーに差し替わり、CPU でなら書ける書き出しまで丸ごと失敗する
        readback = _Readback("yuv420p")
        real_release = readback.release

        def refusing() -> None:
            real_release()
            raise ShaderError("バッファを返せない")

        monkeypatch.setattr(readback, "release", refusing)
        source, _renderer, _opened = _source(monkeypatch, readback)
        with caplog.at_level("WARNING", logger=exporter.__name__):
            exporter._release_quietly(source)
        assert readback.released
        assert not source.on_gpu
        assert "返せなかった" in caplog.text, "返せなかったことが記録に残らない"

    def test_the_check_texture_is_returned_when_uploading_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 作った後に上げる所で断られたテクスチャを返さないと、書き出すたびに GPU のメモリが
        # 1 枚ぶん（4K で 33MB）ずつ残る GL のコンテキストを返すときに消えるかには頼らない
        readback = _Readback("yuv420p")
        readback.refuse_upload = True
        source, renderer, _opened = _source(monkeypatch, readback)
        assert readback.textures_made == 1, "試験の絵のテクスチャを作る所を通っていない"
        assert readback.live_textures == set(), "上げられなかったテクスチャを返していない"
        assert not source.on_gpu
        assert readback.released
        _run(source, 2)
        assert renderer.rendered == [100, 101]

    def test_a_refused_texture_release_keeps_the_check_result(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 比べ終えた後にテクスチャを返すのを断られただけで GPU をやめると、確かめに通った
        # 機械でも遅い CPU の道になる 黙って捨てると、返せなかったことも追えない
        readback = _Readback("yuv420p")
        readback.refuse_texture_release = True
        with caplog.at_level("WARNING", logger=exporter.__name__):
            source, _renderer, _opened = _source(monkeypatch, readback)
        assert source.on_gpu
        assert "返せなかった" in caplog.text, "返せなかったことが記録に残らない"

    @pytest.mark.parametrize("why", ["disagrees", "refused"])
    def test_a_release_refused_after_the_check_falls_back_to_the_cpu(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, why: str
    ) -> None:
        # 確かめで GPU をやめた後に返すのを駆動が断ると、素の GL のエラーが書き出しを始める前に
        # 外へ出る 書き直しの合図にもならず、CPU でなら書けた書き出しが丸ごと失敗する
        if why == "disagrees":
            readback = _Readback("yuv420p", skew=COLOR_TOLERANCE + 1)
        else:
            readback = _Readback("yuv420p")

            def refuse_pop() -> tuple[tuple[int, int], np.ndarray]:
                raise ShaderError("読み戻しのバッファを開けない")

            monkeypatch.setattr(readback, "pop", refuse_pop)
        real_release = readback.release

        def refusing() -> None:
            real_release()
            raise ShaderError("バッファを返せない")

        monkeypatch.setattr(readback, "release", refusing)
        with caplog.at_level("WARNING", logger=exporter.__name__):
            source, renderer, _opened = _source(monkeypatch, readback)
        assert readback.released
        assert not source.on_gpu
        assert "返せなかった" in caplog.text, "返せなかったことが記録に残らない"
        items = _run(source, 3)
        assert all(image.ndim == 3 for _i, _n, image in items)
        assert renderer.rendered == [100, 101, 102]


# ここから下は実際に書き出す GPU の無い機械では飛ばす


def _planes(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray() for frame in container.decode(video=0)]


@pytest.fixture
def moving_project(sample_av: SampleMedia) -> Project:
    """動く素材（testsrc） コマが入れ替わると絵が変わる"""
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(sample_av.path)):
        document.execute(command)
    return document.project


def _lossless(path: Path, *, gpu_color: bool, depth: int = 2) -> ExportSettings:
    # 可逆で書けば、デコードした面がエンコーダへ渡した面そのものになる
    return ExportSettings(
        path=path,
        video_codec="libx264",
        frame_range=(0, 12),
        options={"qp": "0"},
        pipeline_depth=depth,
        gpu_color=gpu_color,
    )


@pytest.mark.usefixtures("gpu")
class TestExport:
    @pytest.mark.parametrize("depth", [0, 2])
    def test_the_gpu_writes_the_same_video_as_the_cpu(
        self,
        moving_project: Project,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        depth: int,
    ) -> None:
        # 速くするために絵が変わったら本末転倒 コマの数・順・色が CPU の道と丸めの 1 まで同じ
        pushed: list[tuple[int, int]] = []
        original = yuv.YuvReadback.push

        def spy(self: yuv.YuvReadback[tuple[int, int]], texture: int, tag: tuple[int, int]) -> None:
            pushed.append(tag)
            original(self, texture, tag)

        monkeypatch.setattr(yuv.YuvReadback, "push", spy)
        gpu = tmp_path / "gpu.mp4"
        cpu = tmp_path / "cpu.mp4"
        export_project(moving_project, _lossless(gpu, gpu_color=True, depth=depth))
        # 頭の 1 枚は始める前の試験の絵 それが通ってから 12 コマを GPU で変換する
        assert pushed[0] == exporter._CHECK_TAG, "始める前に GPU の変換を確かめていない"
        assert len(pushed) == 13, "GPU の道を通っていない"
        export_project(moving_project, _lossless(cpu, gpu_color=False, depth=depth))
        assert len(pushed) == 13, "切ったのに GPU の道を通った"

        left, right = _planes(gpu), _planes(cpu)
        assert len(left) == len(right) == 12
        for number, (a, b) in enumerate(zip(left, right, strict=True)):
            difference = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
            assert difference <= COLOR_TOLERANCE, f"{number} 枚目が {difference} ずれた"

    def test_the_tags_stay_bt709(self, moving_project: Project, tmp_path: Path) -> None:
        # タグが変わると、再生側が別の行列で読み、GPU で書いた動画だけ色が変わって見える
        from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

        output = tmp_path / "tags.mp4"
        export_project(moving_project, _lossless(output, gpu_color=True))
        with av.open(str(output)) as container:
            context = container.streams.video[0].codec_context
            assert context.color_primaries == ColorPrimaries.BT709
            assert context.color_trc == ColorTrc.BT709
            assert context.color_range == ColorRange.MPEG

    def test_cancelling_returns_the_buffers(
        self, moving_project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 中止のたびに読み戻しのバッファが残ると、何度か中止した後で GPU のメモリが尽きる
        released: list[bool] = []
        original = yuv.YuvReadback.release

        def spy(self: yuv.YuvReadback[tuple[int, int]]) -> None:
            released.append(self.pending > 0)
            original(self)

        monkeypatch.setattr(yuv.YuvReadback, "release", spy)
        calls = [0]

        def should_cancel() -> bool:
            calls[0] += 1
            return calls[0] > 5

        output = tmp_path / "cancelled.mp4"
        with pytest.raises(ExportError, match="中止"):
            export_project(
                moving_project,
                _lossless(output, gpu_color=True),
                should_cancel=should_cancel,
            )
        assert released, "中止したのに読み戻しのバッファを返していない"
        assert not output.exists()

    def test_a_writer_failure_returns_the_buffers(
        self, moving_project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 書き込みの失敗で抜けたときも返す 返さないと、失敗を繰り返すたびに GPU のメモリが減る
        released = threading.Event()
        original = yuv.YuvReadback.release

        def spy(self: yuv.YuvReadback[tuple[int, int]]) -> None:
            released.set()
            original(self)

        def failing(
            self: exporter._FrameWriter, index: int, frame_number: int, image: np.ndarray
        ) -> None:
            raise RuntimeError("書き込みで失敗した")

        monkeypatch.setattr(yuv.YuvReadback, "release", spy)
        monkeypatch.setattr(exporter._FrameWriter, "write", failing)
        with pytest.raises(RuntimeError, match="書き込みで失敗した"):
            export_project(moving_project, _lossless(tmp_path / "f.mp4", gpu_color=True))
        assert released.is_set()

    def test_starting_again_with_the_next_encoder_reopens_the_readback(
        self,
        moving_project: Project,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """候補のエンコーダが途中で書けなくなり、次の候補で書き直しても GPU の道で書き上げる（#365）

        1 本目の読み戻しを返さないと、書き直すたびに GPU のメモリが残る 2 本目で開き直して
        確かめ直さないと、返した PBO を使って落ちるか、黙って CPU の道へ落ちて遅くなる
        進み具合が戻ると、書き出しの窓のバーが下がって壊れたように見える
        """
        for name in ("mpeg4", "libx264"):
            if not usable(name):
                pytest.skip(f"{name} で書けない")
        # 開けるかの答えはプロセスの中で覚えている 偽の答えを後の試験へ漏らさない
        exporter._opens.cache_clear()
        monkeypatch.setattr(exporter, "_open_encoder", lambda name: None)
        monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("mpeg4", "libx264"))
        real = exporter._encode_frame
        written = [0]

        def breaking(
            video: av.video.stream.VideoStream, frame: av.video.frame.VideoFrame | None
        ) -> list[av.Packet[Any]]:
            if video.codec_context.name == "mpeg4" and frame is not None:
                written[0] += 1
                if written[0] >= 4:
                    raise exporter._EncoderWriteError("偽の駆動の失敗")
            return real(video, frame)

        opened: list[yuv.YuvReadback[tuple[int, int]]] = []
        real_open = yuv.open_readback

        def spy_open(width: int, height: int, fmt: str) -> yuv.YuvReadback[tuple[int, int]] | None:
            readback: yuv.YuvReadback[tuple[int, int]] | None = real_open(width, height, fmt)
            if readback is not None:
                opened.append(readback)
            return readback

        pushed: list[tuple[int, int]] = []
        real_push = yuv.YuvReadback.push

        def spy_push(
            self: yuv.YuvReadback[tuple[int, int]], texture: int, tag: tuple[int, int]
        ) -> None:
            pushed.append(tag)
            real_push(self, texture, tag)

        monkeypatch.setattr(exporter, "_encode_frame", breaking)
        monkeypatch.setattr(exporter, "open_readback", spy_open)
        monkeypatch.setattr(yuv.YuvReadback, "push", spy_push)
        seen: list[float] = []
        notices: list[str] = []
        output = tmp_path / "again.mp4"
        try:
            export_project(
                moving_project,
                ExportSettings(path=output, frame_range=(0, 12), pipeline_depth=2),
                progress=seen.append,
                notice=notices.append,
            )
        finally:
            exporter._opens.cache_clear()

        assert len(opened) == 2, "書き直しで GPU の読み戻しを開き直していない"
        assert all(readback.released for readback in opened), "読み戻しを返していない"
        checks = [at for at, tag in enumerate(pushed) if tag == exporter._CHECK_TAG]
        assert len(checks) == 2, "書き直しで GPU の変換を確かめ直していない"
        # 2 本目は試験の絵の後に、12 コマを頭から順に GPU で変換する
        assert pushed[checks[1] + 1 :] == [(index, index) for index in range(12)]
        assert len(notices) == 1 and "libx264" in notices[0]
        assert seen == sorted(seen), f"進み具合が戻った: {seen}"
        assert seen[-1] == pytest.approx(1.0)
        assert len(_planes(output)) == 12

    @pytest.mark.parametrize("depth", [0, 2])
    def test_a_readback_failure_mid_way_writes_again_on_the_cpu(
        self,
        moving_project: Project,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        depth: int,
    ) -> None:
        """GPU の読み戻しが書き出しの途中で失敗しても、同じコーデックで CPU の色変換で書き上げる

        素の GL の例外のまま出すと、CPU でなら書ける書き出しが丸ごと失敗して書きかけも消える
        その場から CPU に切り替えると、読み戻している途中のコマが欠けるか順が乱れる
        頭から書き直せば、CPU の道だけで書いた動画と 1 画素も違わず、音の長さもそろう
        """
        pops = [0]
        real_pop = yuv.YuvReadback.pop

        def failing(
            self: yuv.YuvReadback[tuple[int, int]],
        ) -> tuple[tuple[int, int], np.ndarray]:
            tag, image = real_pop(self)
            if tag != exporter._CHECK_TAG:
                pops[0] += 1
                # 実のコマを 4 枚受け取った所で駆動が断ったことにする 3 枚はもう書いてある
                if pops[0] == 4:
                    raise ShaderError("読み戻しのバッファを開けない")
            return tag, image

        opened: list[yuv.YuvReadback[tuple[int, int]]] = []
        real_open = yuv.open_readback

        def spy_open(width: int, height: int, fmt: str) -> yuv.YuvReadback[tuple[int, int]] | None:
            readback: yuv.YuvReadback[tuple[int, int]] | None = real_open(width, height, fmt)
            if readback is not None:
                opened.append(readback)
            return readback

        # 書き直す前に書きかけを消したかを、書き始めるときにファイルがあるかで見る
        left_over: list[bool] = []
        real_encode = exporter._encode

        def spy_encode(
            project: Project, settings: ExportSettings, *arguments: Any, **options: Any
        ) -> None:
            left_over.append(settings.path.exists())
            real_encode(project, settings, *arguments, **options)

        monkeypatch.setattr(yuv.YuvReadback, "pop", failing)
        monkeypatch.setattr(exporter, "open_readback", spy_open)
        monkeypatch.setattr(exporter, "_encode", spy_encode)
        seen: list[float] = []
        notices: list[str] = []
        gpu = tmp_path / "gpu.mp4"
        export_project(
            moving_project,
            _lossless(gpu, gpu_color=True, depth=depth),
            progress=seen.append,
            notice=notices.append,
        )
        monkeypatch.undo()
        cpu = tmp_path / "cpu.mp4"
        export_project(moving_project, _lossless(cpu, gpu_color=False, depth=depth))

        assert pops[0] == 4, "GPU の道で途中まで書いていない"
        assert len(opened) == 1, "書き直しで、断った GPU の読み戻しをもう 1 度開いた"
        assert opened[0].released, "読み戻しのバッファを返していない"
        assert left_over == [False, False], "書きかけのファイルを残したまま書き直した"
        left, right = _planes(gpu), _planes(cpu)
        assert len(left) == len(right) == 12
        for number, (a, b) in enumerate(zip(left, right, strict=True)):
            assert np.array_equal(a, b), f"{number} 枚目が CPU の道と違う（欠けたか順が乱れた）"
        assert _samples(gpu) == _samples(cpu) > 0, "書き直しで音の長さが変わった"
        assert len(notices) == 1 and "CPU" in notices[0]
        assert seen == sorted(seen), f"進み具合が戻った: {seen}"
        assert seen[-1] == pytest.approx(1.0)

    @pytest.mark.parametrize("pop_fails", [True, False], ids=["mid-way", "at-the-end"])
    def test_a_refused_release_still_ends_in_a_cpu_video(
        self,
        moving_project: Project,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        pop_fails: bool,
    ) -> None:
        """読み戻しを返すのを駆動が断っても、CPU の色変換で書き直して書き上げる

        途中で読み戻しが失敗した後の片付けで投げると、書き直しの合図がその GL のエラーに
        差し替わり、CPU でなら書ける書き出しが丸ごと失敗する 最後まで書いた後の片付けで
        断ったときも、失敗として上げると書き出しが失敗するので、CPU で頭から書き直す
        """
        pops = [0]
        real_pop = yuv.YuvReadback.pop
        real_release = yuv.YuvReadback.release
        refused: list[bool] = []

        def failing(
            self: yuv.YuvReadback[tuple[int, int]],
        ) -> tuple[tuple[int, int], np.ndarray]:
            tag, image = real_pop(self)
            if tag != exporter._CHECK_TAG:
                pops[0] += 1
                if pop_fails and pops[0] == 4:
                    raise ShaderError("読み戻しのバッファを開けない")
            return tag, image

        def refusing(self: yuv.YuvReadback[tuple[int, int]]) -> None:
            real_release(self)
            # 試験の絵の確かめの後ではなく、書き出しの片付けで断ったことにする
            if pops[0]:
                refused.append(True)
                raise ShaderError("バッファを返せない")

        monkeypatch.setattr(yuv.YuvReadback, "pop", failing)
        monkeypatch.setattr(yuv.YuvReadback, "release", refusing)
        notices: list[str] = []
        gpu = tmp_path / "gpu.mp4"
        export_project(moving_project, _lossless(gpu, gpu_color=True), notice=notices.append)
        monkeypatch.undo()
        cpu = tmp_path / "cpu.mp4"
        export_project(moving_project, _lossless(cpu, gpu_color=False))

        assert refused == [True], "片付けで断る所を通っていない"
        left, right = _planes(gpu), _planes(cpu)
        assert len(left) == len(right) == 12
        for number, (a, b) in enumerate(zip(left, right, strict=True)):
            assert np.array_equal(a, b), f"{number} 枚目が CPU の道と違う"
        assert _samples(gpu) == _samples(cpu) > 0
        assert len(notices) == 1 and "CPU" in notices[0]


def _samples(path: Path) -> int:
    """音声の標本の数 書き直しで頭の音が重なったり欠けたりすると変わる"""
    with av.open(str(path)) as container:
        return sum(frame.samples for frame in container.decode(audio=0))
