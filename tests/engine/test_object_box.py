"""入れ物（絵の中で色の付いた範囲）を速く求めても、答えが前と同じであること（Issue #128）

前は絵の全体の α を 1 バイトずつ飛び飛びに読んで範囲を探しており、エフェクトを積んだ
4K の動画 3 枚で 1 フレーム 30ms 掛かっていた 速くした書き方は、縁の 1 周で決まる
向きを探さず、決まらない向きは 1 画素を 32bit 1 つとして読む

範囲が 1 画素でも狭くなると、角丸・回転の中心・影やぼかしの広がりの基準がずれ、
端が切れる なので「だいたい同じ」ではなく、前の書き方（:func:`reference_box`）と
完全に同じ答えになることを押さえる
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Timeline,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE, TEXT
from sashimono.effects.spec import ParamInput
from sashimono.engine.decode import probe_media
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from sashimono.engine.render import renderer as renderer_module
from sashimono.engine.render.renderer import _content_box

Box = tuple[int, int, int, int]


def reference_box(image: np.ndarray) -> Box | None:
    """Issue #128 より前の書き方そのまま 答えの見本として残す"""
    alpha = image[..., 3]
    columns = np.flatnonzero(alpha.any(axis=0))
    rows = np.flatnonzero(alpha.any(axis=1))
    if columns.size == 0 or rows.size == 0:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def blank(height: int = 60, width: int = 90) -> np.ndarray:
    return np.zeros((height, width, 4), np.uint8)


def edge_touching() -> Iterator[tuple[str, np.ndarray]]:
    """上下左右の 4 辺のうち、色が触れる辺の組み合わせ 16 通り

    縁の 1 周で決まる向きと、探す向きが混ざる所 どれか 1 辺でも取り違えると、
    範囲が絵の端まで広がるか、端の 1 列が切れる
    """
    for mask in range(16):
        image = blank()
        top = 0 if mask & 1 else 12
        bottom = 60 if mask & 2 else 47
        left = 0 if mask & 4 else 20
        right = 90 if mask & 8 else 71
        # 端に触れる所は 1 画素だけにする 太いと縁の判定が甘くても通ってしまう
        image[top, (left + right) // 2, 3] = 1
        image[bottom - 1, (left + right) // 3, 3] = 200
        image[(top + bottom) // 2, left, 3] = 7
        image[(top + bottom) // 3, right - 1, 3] = 255
        yield f"辺{mask:04b}", image


def rotated_rectangle() -> np.ndarray:
    """30 度回した長方形 回転の効果を先に焼いた絵（斜めの縁は半透明）"""
    height, width = 120, 160
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    angle = np.radians(30.0)
    u = (xs - 80) * np.cos(angle) + (ys - 60) * np.sin(angle)
    v = -(xs - 80) * np.sin(angle) + (ys - 60) * np.cos(angle)
    inside = np.clip(1.0 - np.maximum(np.abs(u) - 40, np.abs(v) - 20), 0.0, 1.0)
    image = np.zeros((height, width, 4), np.uint8)
    image[..., :3] = 255
    image[..., 3] = (inside * 255).astype(np.uint8)
    return image


def soft_shadow() -> np.ndarray:
    """影やぼかしのように、外へ行くほど薄くなる絵 一番外は α が 1 だけ残る"""
    height, width = 100, 140
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    distance = np.hypot(xs - 70, ys - 50)
    alpha = np.clip(255 * np.exp(-distance / 9.0), 0, 255).astype(np.uint8)
    image = np.zeros((height, width, 4), np.uint8)
    image[..., 3] = alpha
    return image


def random_images() -> Iterator[tuple[str, np.ndarray]]:
    """大きさも色の散らばり方も違う絵 種を固定して毎回同じ物を作る"""
    generator = np.random.default_rng(128)
    for number in range(60):
        height = int(generator.integers(1, 70))
        width = int(generator.integers(1, 70))
        image = generator.integers(0, 256, (height, width, 4), dtype=np.uint8)
        # 色の付く画素を 0〜数個まで減らす 多いと 4 辺すべてに色が付いて縁だけで終わる
        density = float(generator.choice([0.0, 0.0005, 0.005, 0.05, 0.5]))
        keep = generator.random((height, width)) < density
        image[..., 3] = np.where(keep, image[..., 3], 0)
        yield f"乱数{number}", image


def cases() -> Iterator[tuple[str, np.ndarray]]:
    yield "不透明（動画）", np.full((54, 96, 4), 255, np.uint8)
    yield "空", blank()
    yield "1x1 の色", np.full((1, 1, 4), 255, np.uint8)
    yield "1x1 の透明", np.zeros((1, 1, 4), np.uint8)
    yield "0 行", np.zeros((0, 5, 4), np.uint8)
    yield "0 列", np.zeros((5, 0, 4), np.uint8)
    for name, (y, x) in {
        "左上": (0, 0),
        "右上": (0, 89),
        "左下": (59, 0),
        "右下": (59, 89),
        "中央": (30, 45),
    }.items():
        image = blank()
        image[y, x, 3] = 1
        yield f"{name}の 1 画素（α 1）", image
    yield from edge_touching()

    colour_only = blank()
    colour_only[..., :3] = 255
    # 色の値はあっても α が 0 なら何も無い 32bit で読むときに色の桁まで見てしまうと、
    # 画面いっぱいが範囲になり、角丸や回転の中心が画面の角と中央へずれる
    yield "色だけあって α は 0", colour_only
    colour_around = colour_only.copy()
    colour_around[20:30, 40:50, 3] = 255
    yield "色は全体 α は真ん中だけ", colour_around

    yield "回した長方形", rotated_rectangle()
    yield "外へ薄くなる影", soft_shadow()

    half = blank()
    half[:, :45, 3] = 128
    yield "左半分だけ半透明（上下左が端）", half

    base = soft_shadow()
    yield "切り出した一部（行が飛ぶ）", base[10:80, 5:120]
    yield "間引いた絵（画素が飛ぶ）", base[::2, ::3]
    yield "回した見え方（並びが入れ替わる）", np.rot90(base)
    yield "列が先の並び", np.asfortranarray(base)
    yield "上下を返した見え方", base[::-1]

    as_float = soft_shadow().astype(np.float32) / 255.0
    yield "小数の絵", as_float
    yield "16bit の絵", soft_shadow().astype(np.uint16) * 257
    yield from random_images()


CASES = list(cases())


@pytest.mark.parametrize(("name", "image"), CASES, ids=[name for name, _ in CASES])
def test_the_box_is_the_same_as_before(name: str, image: np.ndarray) -> None:
    # 範囲が 1 画素でも狭いと、角丸や回転の中心や影の広がりの基準がずれ、絵の端が 1 列切れる
    assert _content_box(image) == reference_box(image), name


def test_the_image_is_not_changed() -> None:
    # 範囲を探すだけで絵を書き換えてはいけない 覚えておいた作った絵を次のフレームでも使う
    image = soft_shadow()
    before = image.copy()
    _content_box(image)
    assert np.array_equal(image, before)


# --- 描画の道全体 -------------------------------------------------------------

WIDTH, HEIGHT = 160, 120
RATE = FrameRate(30)


@pytest.fixture(scope="module")
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def write_png(path: Path, image: np.ndarray) -> Path:
    """RGBA の配列を PNG へ 画像のためだけに Pillow を足さず、入っている Qt で書く"""
    from PySide6.QtGui import QImage

    height, width = image.shape[:2]
    data = np.ascontiguousarray(image, dtype=np.uint8)
    qimage = QImage(data.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
    assert qimage.save(str(path)), f"{path} を書けない"
    return path


def effect(kind: str, **values: ParamInput) -> Effect:
    return registry.require(kind).create(**values)


#: 入れ物を使うエフェクト 回転・拡大は入れ物の中央を中心に回し、角丸は入れ物の角に付き、
#: 影・ぼかし・発光は入れ物の外へ広がる
EFFECTS: dict[str, tuple[Effect, ...]] = {
    "回転と拡大": (effect("transform", rotation=30, scale=80),),
    "影": (effect("shadow", offset_x=10, offset_y=-8, blur=6),),
    "ぼかし": (effect("blur", radius=6),),
    "発光": (effect("glow", threshold=0.2, radius=12),),
    "角丸": (effect("round_corner"),),
    "全部": (
        effect("round_corner"),
        effect("shadow", offset_x=10, offset_y=-8, blur=6),
        effect("glow", threshold=0.2, radius=12),
        effect("transform", rotation=30, scale=80),
    ),
}


def still(tmp_path: Path, name: str, image: np.ndarray) -> MediaItem:
    return probe_media(write_png(tmp_path / f"{name}.png", image))


def transparent_edge_image() -> np.ndarray:
    """左の辺にだけ触れる半透明の絵 上下右は透明の余白（配布の立ち絵のような PNG）"""
    image = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    image[..., 0] = 200
    image[..., 1] = 120
    image[30:90, 0:100, 3] = 180
    image[40:80, 0:60, 3] = 255
    return image


def sources(tmp_path: Path) -> dict[str, tuple[Clip, tuple[MediaItem, ...]]]:
    """絵の出どころ 素材の静止画（半透明の余白あり・不透明）と、作る絵（図形・文字）"""
    edge = still(tmp_path, "edge", transparent_edge_image())
    opaque_image = np.full((HEIGHT, WIDTH, 4), 255, np.uint8)
    opaque_image[..., 2] = 40
    opaque = still(tmp_path, "opaque", opaque_image)
    shape = SHAPE.create(shape="rect", width=50, height=30, color=(1.0, 0.8, 0.2, 1.0))
    text = TEXT.create(text="入れ物", size=28)
    return {
        "余白のある静止画": (Clip(timeline_start=0, duration=10, media_id=edge.id), (edge,)),
        "不透明な静止画": (Clip(timeline_start=0, duration=10, media_id=opaque.id), (opaque,)),
        "図形": (Clip(timeline_start=0, duration=10, source=shape), ()),
        "文字": (Clip(timeline_start=0, duration=10, source=text), ()),
    }


def project_of(clip: Clip, media: tuple[MediaItem, ...]) -> Project:
    project = Project.create(
        ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=RATE), media=media
    )
    track = Track(kind=TrackKind.VIDEO, clips=(clip,))
    return project.with_timeline(Timeline(rate=project.rate, tracks=(track,)))


def frames_of(project: Project, gl: OffscreenGLContext, count: int = 3) -> list[np.ndarray]:
    """同じレンダラで頭から ``count`` 枚 覚えた範囲を使い回す道も通す"""
    renderer = FrameRenderer(project, context=gl)
    try:
        return [renderer.render(frame) for frame in range(count)]
    finally:
        renderer.close()


def fresh_frames(project: Project, gl: OffscreenGLContext, count: int = 3) -> list[np.ndarray]:
    """1 枚ごとに新しいレンダラで描く 覚えた物が何も無いので、見本の側に使う

    同じレンダラで見本を描くと、覚えた範囲を取り違える作りでも両方が同じだけ
    取り違え、比べても差が出ない
    """
    frames = []
    for frame in range(count):
        renderer = FrameRenderer(project, context=gl)
        try:
            frames.append(renderer.render(frame))
        finally:
            renderer.close()
    return frames


def swapped(monkeypatch: pytest.MonkeyPatch, finder: Callable[[np.ndarray], Box | None]) -> None:
    monkeypatch.setattr(renderer_module, "_content_box", finder)


@pytest.mark.parametrize("effects", list(EFFECTS), ids=list(EFFECTS))
def test_rendering_matches_the_old_box(
    effects: str, gl: OffscreenGLContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 入れ物は素材の絵だけで決まり、回転や影はそれを基準に後から掛かる 前の書き方で
    # 範囲を探したときと、1 画素も違わない絵になることを描いて確かめる
    for name, (clip, media) in sources(tmp_path).items():
        project = project_of(replace(clip, effects=EFFECTS[effects]), media)
        now = frames_of(project, gl)
        with monkeypatch.context() as patched:
            swapped(patched, reference_box)
            before = fresh_frames(project, gl)
        for frame, (ours, theirs) in enumerate(zip(now, before, strict=True)):
            assert np.array_equal(ours, theirs), f"{name} {effects} の {frame} フレーム目"


def test_a_smaller_box_would_show(
    gl: OffscreenGLContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 上の比べ方が本当に範囲の違いを拾えるかの確かめ 範囲を 2 画素狭めると絵が変わる
    # 変わらなければ、上の試験は範囲を取り違えても通ってしまう
    def shrunk(image: np.ndarray) -> Box | None:
        found = reference_box(image)
        if found is None:
            return None
        return found[0] + 2, found[1] + 2, found[2] - 2, found[3] - 2

    clip, media = sources(tmp_path)["余白のある静止画"]
    project = project_of(replace(clip, effects=EFFECTS["全部"]), media)
    correct = frames_of(project, gl, count=1)[0]
    with monkeypatch.context() as patched:
        swapped(patched, shrunk)
        narrow = frames_of(project, gl, count=1)[0]
    assert not np.array_equal(correct, narrow)


def counting(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """範囲を探した回数を数える"""
    calls: list[int] = []
    original = renderer_module._content_box

    def counted(image: np.ndarray) -> Box | None:
        calls.append(1)
        return original(image)

    swapped(monkeypatch, counted)
    return calls


def test_a_still_generated_picture_is_scanned_once(
    gl: OffscreenGLContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 動かない図形は同じ絵を毎フレーム描く 範囲を毎回探すと、1080p の図形 20 本で
    # 1 フレーム 48ms 掛かっていた
    shape = SHAPE.create(shape="rect", width=50, height=30, color=(1.0, 1.0, 1.0, 1.0))
    clip = Clip(timeline_start=0, duration=10, source=shape, effects=EFFECTS["発光"])
    calls = counting(monkeypatch)
    frames_of(project_of(clip, ()), gl, count=5)
    assert len(calls) == 1


def growing_shape() -> GeneratedSource:
    """幅がフレームごとに伸びる図形 毎フレーム別の絵になる"""
    shape = SHAPE.create(shape="rect", width=20, height=30, color=(1.0, 1.0, 1.0, 1.0))
    width = AnimatedValue(20.0, (Keyframe(0, 20.0), Keyframe(4, 140.0)))
    return shape.with_param("width", width)


def test_a_changing_generated_picture_is_scanned_every_frame(
    gl: OffscreenGLContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 絵が作り直されたら前の範囲を使ってはいけない 伸びる図形に前の狭い範囲を当てると、
    # 角丸や回転の中心が前の大きさのままになる
    clip = Clip(timeline_start=0, duration=10, source=growing_shape(), effects=EFFECTS["全部"])
    calls = counting(monkeypatch)
    frames_of(project_of(clip, ()), gl, count=5)
    assert len(calls) == 5


def test_a_changing_generated_picture_matches_the_old_box(
    gl: OffscreenGLContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 覚えた範囲を伸びる図形に使い回すと、再生していくうちに角丸や影が前のコマの大きさのまま描かれる
    clip = Clip(timeline_start=0, duration=10, source=growing_shape(), effects=EFFECTS["全部"])
    project = project_of(clip, ())
    now = frames_of(project, gl, count=5)
    with monkeypatch.context() as patched:
        swapped(patched, reference_box)
        before = fresh_frames(project, gl, count=5)
    for frame, (ours, theirs) in enumerate(zip(now, before, strict=True)):
        assert np.array_equal(ours, theirs), f"{frame} フレーム目"
