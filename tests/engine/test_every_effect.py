"""今あるオブジェクトに、エフェクトを全部 1 つずつ掛ける（AI テスト #4）

テキスト・図形・画像の 3 つへ、映像のエフェクトを既定の値で 1 つずつ掛けて描く
落ちない・値が壊れない・絵が変わること、変わらない物は訳（既定の値では何もしない・
中身の色では効かない・ほかの入力が要る）が決まっていることを確かめる 既定で何もしない
物は、値を動かすと変わることを別に見る 最後に全部を 1 本ずつ並べて書き出し、読み戻す

背景に灰色を敷く 黒い影や黒い縁は、何も無い所（黒）の上では見えず変わらないように見える
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from sashimono.core.commands import AddMedia
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.effects.sources import SHAPE, TEXT
from sashimono.engine.decode import probe_media
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

SETTINGS = ProjectSettings(width=320, height=180, frame_rate=FrameRate(30))
OBJECTS = ("text", "shape", "image")
#: 見るコマ 登場・退場の途中（5）と真ん中と終わり 頭と真ん中だけだと、1 回転の登場は
#: 0 度と 360 度で同じ絵になり、変わっていないように見える
FRAMES = (0, 5, 15, 29)
#: 変わったとみなす画素の数（どれかの色が 8 より大きく違う画素）
CHANGED_PIXELS = 20
ORANGE = (0.9, 0.4, 0.2, 1.0)

#: 既定の値のままでは何もしない物と、値を動かしたときの値 動かしたら変わることを見る
IDLE_AT_DEFAULT: dict[str, dict[str, Any]] = {
    "color": {"brightness": 50.0},
    "color_correct": {"brightness": 150.0},
    "color_grade": {"luma_gain": 50.0},
    "crop": {"top": 20.0},
    "crop_angle": {"angle": 30.0, "width": 40.0},
    "expand_area": {"top": 30.0, "fill": True},
    "exposure": {"amount": 200.0},
    "highlights_shadows": {"highlights": -80.0, "shadows": 80.0},
    "linear_transfer": {"red_slope": 30.0},
    "mesh_deform": {"point0_x": 40.0},
    "opacity": {"amount": 40.0},
    "reel_spin": {"rotation": 90.0},
    "skew": {"angle_x": 30.0},
    "split_pieces": {"columns": 3.0, "rows": 2.0, "scale": 80.0},
    "transform": {"rotation": 30.0},
    "directional_key": {"background": ORANGE, "foreground": (0.1, 0.1, 0.9, 1.0)},
    "color_key": {"key_color": ORANGE, "tolerance": 30.0},
    "chroma_key": {"key_color": ORANGE},
    "mask": {"mask_width": 60.0, "mask_height": 40.0},
    "shape_mask": {"width": 60.0, "height": 60.0},
}

#: 中身の色や形によっては既定の値でも変わらない組と、その訳
CONTENT_BOUND: dict[tuple[str, str], str] = {
    ("binarize", "text"): "白い文字は 2 値にしても白のまま",
    ("bevel_light", "text"): "細い字画には反射の幅が乗らない",
    ("fill", "text"): "既定の色は白 白い文字を白で塗る",
    ("gradient_map", "text"): "白はグラデーションの白い端に写る",
    ("color_range_shift", "text"): "白には色相が無く、ずらす色域に入らない",
    ("inner_outline", "text"): "既定の縁の色は白 白い文字の内側を白で縁取る",
    ("luminance_key", "text"): "白は輝度の閾値より明るく抜けない",
    ("noise", "text"): "白の上の明るさの揺れは白で頭打ちになる",
    ("round_corner", "text"): "文字の入れ物の角は透明",
    ("sharpen", "text"): "一色の文字は縁の段差が残るだけ",
    ("sharpen", "shape"): "一色の四角には強める模様が無い",
    ("flip", "shape"): "左右対称の四角は反転しても同じ",
    ("glow", "shape"): "橙の明るさは光らせる閾値（0.6）より暗い",
    ("ripple", "shape"): "外形は四角のまま（YMM4 と同じ） 一色の中身は揺らしても同じ色",
}

#: ほかの入力が無いと何もしない物
NEEDS_INPUT: dict[str, str] = {
    "displacement_map": "ずらしに使う絵を選ぶまで何もしない",
    "image_blend": "重ねる絵を選ぶまで何もしない",
    "partial_filter": "後ろに積んだエフェクトを範囲だけに掛ける物 後ろに何も無ければ何もしない",
    "after_image": "動いた跡を残す物 止まった物には跡が無い",
}


def _kinds() -> list[str]:
    """映像のエフェクト（音と、AviUtl のスクリプトは除く スクリプトは別の試験が見る）"""
    return sorted(
        d.kind
        for d in registry.all()
        if d.audio_process is None and not d.kind.startswith("aviutl:")
    )


@pytest.fixture(scope="module")
def picture(tmp_path_factory: pytest.TempPathFactory) -> MediaItem:
    """画像の素材 模様と色のある 160 × 90 の PNG を numpy から書く（ffmpeg が無くても作れる）"""
    from PySide6.QtGui import QImage

    height, width = 90, 160
    y, x = np.mgrid[0:height, 0:width]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = (x * 255 // width).astype(np.uint8)
    rgba[..., 1] = (y * 255 // height).astype(np.uint8)
    rgba[..., 2] = (((x // 10 + y // 10) % 2) * 200).astype(np.uint8)
    rgba[..., 3] = 255
    path = tmp_path_factory.mktemp("every_effect") / "pattern.png"
    image = QImage(rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
    assert image.save(str(path))
    return probe_media(path)


def _project(obj: str, effects: tuple[Effect, ...], media: MediaItem) -> Project:
    base = Project.create(SETTINGS)
    if obj == "text":
        clip = Clip(
            timeline_start=0,
            duration=30,
            source=TEXT.create(text="字あA", size=60.0),
            effects=effects,
        )
    elif obj == "shape":
        clip = Clip(
            timeline_start=0,
            duration=30,
            source=SHAPE.create(shape="rect", width=120, height=80, color=ORANGE),
            effects=effects,
        )
    else:
        base = AddMedia(media).apply(base)
        placed = with_fixed_items(
            Clip(timeline_start=0, duration=30, media_id=media.id, native_size=True), picture=True
        )
        # 固定の欄（配置）は列の末尾 掛けるエフェクトはその前
        clip = replace(placed, effects=(*effects, *placed.effects))
    back = Clip(
        timeline_start=0,
        duration=30,
        source=SHAPE.create(shape="background", color=(0.5, 0.55, 0.6, 1.0)),
    )
    tracks = (Track(TrackKind.VIDEO, "V0", (back,)), Track(TrackKind.VIDEO, "V1", (clip,)))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


@pytest.fixture(scope="module")
def renderer(picture: MediaItem) -> Iterator[FrameRenderer]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    made = FrameRenderer(_project("text", (), picture), context=context)
    yield made
    made.close()
    context.release()


def _frames(renderer: FrameRenderer, project: Project) -> list[np.ndarray]:
    renderer.set_project(project)
    images = [renderer.render(frame) for frame in FRAMES]
    for image in images:
        assert image.shape[:2] == (SETTINGS.height, SETTINGS.width)
        assert np.isfinite(image.astype(np.float64)).all()
    return images


def _changed(before: list[np.ndarray], after: list[np.ndarray]) -> int:
    """コマごとに変わった画素の数の、一番多いコマの数

    はっきり変わった画素（どれかの色が 8 より大きく違う）を数える 画面いっぱいに薄く
    掛かる物（閃光の靄）は 1 画素ずつは 8 に届かないので、平均の差が 0.5 を超えたら
    変わった画素が十分あったものとして数える
    """
    most = 0
    for a, b in zip(before, after, strict=True):
        difference = np.abs(a.astype(np.int32) - b.astype(np.int32))
        count = int((difference.max(axis=2) > 8).sum())
        if float(difference.mean()) > 0.5:
            count = max(count, CHANGED_PIXELS)
        most = max(most, count)
    return most


@pytest.fixture(scope="module")
def plain(renderer: FrameRenderer, picture: MediaItem) -> dict[str, list[np.ndarray]]:
    return {obj: _frames(renderer, _project(obj, (), picture)) for obj in OBJECTS}


@pytest.mark.parametrize("kind", _kinds())
def test_each_effect_draws_and_changes_the_picture(
    kind: str,
    renderer: FrameRenderer,
    picture: MediaItem,
    plain: dict[str, list[np.ndarray]],
) -> None:
    """既定の値で掛けて、落ちずに描け、絵が変わる 変わらないなら訳が決まっている

    訳の無いまま変わらなくなったら、掛けても効かない不具合（値の読み違い・入れ物の取り違え）
    訳の決まった物は変わっても断らない（止まった物の残像は縁の半透明が重なって濃くなる など）
    """
    effect = registry.require(kind).create()
    for obj in OBJECTS:
        changed = _changed(plain[obj], _frames(renderer, _project(obj, (effect,), picture)))
        idle = kind in IDLE_AT_DEFAULT or kind in NEEDS_INPUT or (kind, obj) in CONTENT_BOUND
        if not idle:
            assert changed >= CHANGED_PIXELS, f"{kind} を {obj} へ掛けても絵が変わらない"


@pytest.mark.parametrize("kind", sorted(IDLE_AT_DEFAULT))
def test_an_idle_effect_changes_once_its_value_moves(
    kind: str,
    renderer: FrameRenderer,
    picture: MediaItem,
    plain: dict[str, list[np.ndarray]],
) -> None:
    """既定で何もしない物も、値を動かせば絵が変わる（効かないまま何もしないのではない）

    色で抜く物は図形の橙を抜かせる 画像の模様には同じ色が無い
    """
    effect = registry.require(kind).create(**IDLE_AT_DEFAULT[kind])
    target = "shape" if "key" in kind else "image"
    changed = _changed(plain[target], _frames(renderer, _project(target, (effect,), picture)))
    assert changed >= CHANGED_PIXELS, f"{kind} を {IDLE_AT_DEFAULT[kind]} にしても変わらない"


def test_the_lists_name_real_effects() -> None:
    # 名前を変えた・消したエフェクトが一覧に残ると、その分だけ何も確かめていない
    kinds = set(_kinds())
    assert set(IDLE_AT_DEFAULT) <= kinds
    assert set(NEEDS_INPUT) <= kinds
    assert {kind for kind, _ in CONTENT_BOUND} <= kinds


def test_a_partial_filter_limits_the_next_effect(
    renderer: FrameRenderer, picture: MediaItem, plain: dict[str, list[np.ndarray]]
) -> None:
    # 部分フィルタは後ろのエフェクトを範囲だけに掛ける 後ろにぼかしがあれば絵が変わる
    effects = (registry.require("partial_filter").create(), registry.require("blur").create())
    changed = _changed(plain["image"], _frames(renderer, _project("image", effects, picture)))
    assert changed >= CHANGED_PIXELS


def test_an_after_image_trails_a_moving_object(renderer: FrameRenderer, picture: MediaItem) -> None:
    # 動く物には跡が残る 残像を掛けない同じ動きの絵と比べる
    moving = registry.require("transform").create(
        pos_x=AnimatedValue(
            keyframes=(Keyframe(frame=0, value=-80.0), Keyframe(frame=29, value=80.0))
        )
    )
    trail = registry.require("after_image").create()
    without = _frames(renderer, _project("shape", (moving,), picture))
    with_trail = _frames(renderer, _project("shape", (moving, trail), picture))
    assert _changed(without, with_trail) >= CHANGED_PIXELS


@pytest.mark.usefixtures("gpu")
def test_every_effect_survives_an_export(
    tmp_path: Path, picture: MediaItem, renderer: FrameRenderer
) -> None:
    """全部のエフェクトを 1 本ずつ 2 コマ並べて書き出し、読み戻す

    プレビューで描けても、書き出しは別の合成器と先読みの道を通る 途中で落ちると、
    それまでの書き出しが無駄になる
    """
    del renderer
    from sashimono.engine.encode import ExportSettings, available_video_codecs, export_project

    if not available_video_codecs():
        pytest.skip("映像のコーデックが無い")
    base = AddMedia(picture).apply(Project.create(SETTINGS))
    clips = []
    for index, kind in enumerate(_kinds()):
        placed = with_fixed_items(
            Clip(timeline_start=index * 2, duration=2, media_id=picture.id, native_size=True),
            picture=True,
        )
        clips.append(replace(placed, effects=(registry.require(kind).create(), *placed.effects)))
    project = base.with_timeline(
        replace(base.timeline, tracks=(Track(TrackKind.VIDEO, "V1", tuple(clips)),))
    )
    output = tmp_path / "every.mp4"
    export_project(project, ExportSettings(path=output))
    with av.open(str(output)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(frames) == project.duration
    # 真っ黒のコマばかりなら、合成が絵を落としている（抜く物・切る物のコマは暗くてよい）
    lit = sum(1 for frame in frames if frame.mean() > 10)
    assert lit > len(frames) * 0.8
