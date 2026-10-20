"""書き出しの色変換を GPU で行う道（#372）

GPU で変換できる機械では GPU で、できない機械（GPU の無い CI・駆動が断った・本人が切った）
では今の CPU の道で書く どちらの道でも、コマの順と色が変わってはいけない
前半は GL を使わない（偽の描画係で、道の選び方と順番と片付けを見る）
後半は実際に書き出して、CPU の道と同じ動画になるかを見る
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from types import TracebackType
from typing import cast

import av
import numpy as np
import pytest
from OpenGL.error import NullFunctionError

from sashimono.core.commands import Document, insert_media
from sashimono.core.model import Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.colorspace import bt709_planes
from sashimono.engine.decode import probe_media
from sashimono.engine.encode import ExportError, ExportSettings, export_project, exporter
from sashimono.engine.gpu import yuv
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
    def __init__(self) -> None:
        self.current = -1

    def resolved_texture(self, *, straight: bool = False) -> int:
        # テクスチャの番号の代わりに、合成したコマの番号を渡す
        return self.current

    def read(self, *, straight: bool = False) -> np.ndarray:
        return _picture(self.current)


class _Renderer:
    """FrameRenderer の代わり CPU の道（render）を通ったコマを覚える"""

    def __init__(self) -> None:
        self.context = _Scope()
        self.compositor = _Compositor()
        self.rendered: list[int] = []

    def compose(self, frame: int) -> None:
        assert self.context.depth > 0, "GL コンテキストの外で合成した"
        self.compositor.current = frame

    def render(self, frame: int) -> np.ndarray:
        self.rendered.append(frame)
        return _picture(frame)


class _Readback:
    """YuvReadback の代わり 本物と同じく、PBO の数だけ溜めて先に入れた物から出す"""

    def __init__(self, pixel_format: str, *, buffers: int = 2, skew: int = 0) -> None:
        self.pixel_format = pixel_format
        self.buffers = buffers
        #: 変換の結果に足す値 駆動が食い違ったときの真似
        self.skew = skew
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
        planes = bt709_planes(_picture(texture), self.pixel_format).astype(np.int16) + self.skew
        return tag, np.clip(planes, 0, 255).astype(np.uint8)

    def release(self) -> None:
        self._pending.clear()
        self.released = True


def _source(
    monkeypatch: pytest.MonkeyPatch,
    readback: _Readback | None,
    *,
    pixel_format: str | None = "yuv420p",
) -> tuple[exporter._FrameSource, _Renderer, list[str]]:
    opened: list[str] = []

    def fake_open(width: int, height: int, fmt: str) -> _Readback | None:
        opened.append(fmt)
        return readback

    monkeypatch.setattr(exporter, "open_readback", fake_open)
    renderer = _Renderer()
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

    def test_the_first_frame_that_disagrees_switches_to_the_cpu(
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
        # 最初のコマは比べるために合成した絵をそのまま使う 2 回合成すると動きの記憶がずれる
        assert renderer.rendered == [101, 102, 103]

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
        pushed: list[int] = []
        original = yuv.YuvReadback.push

        def spy(self: yuv.YuvReadback[tuple[int, int]], texture: int, tag: tuple[int, int]) -> None:
            pushed.append(texture)
            original(self, texture, tag)

        monkeypatch.setattr(yuv.YuvReadback, "push", spy)
        gpu = tmp_path / "gpu.mp4"
        cpu = tmp_path / "cpu.mp4"
        export_project(moving_project, _lossless(gpu, gpu_color=True, depth=depth))
        assert len(pushed) == 12, "GPU の道を通っていない"
        export_project(moving_project, _lossless(cpu, gpu_color=False, depth=depth))
        assert len(pushed) == 12, "切ったのに GPU の道を通った"

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
