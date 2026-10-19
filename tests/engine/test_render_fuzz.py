"""読み込めてしまうおかしな値で、描画が例外を投げないこと

ファイルの読み込みは型までしか確かめない エフェクトの値が型違い（数のはずが文字）、
極端な数、負の数でも読み込めてしまう 描画の途中で例外が出ると、プレビューが止まり、
書き出しが途中で落ちる すべてのエフェクトに、決まった種で壊した値を入れて描く
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    Keyframe,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.definition import registry
from sashimono.effects.sources import source_registry
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

SETTINGS = ProjectSettings(width=64, height=36, frame_rate=FrameRate(30))

#: 入れてみる値 読み込みを通る型（数・文字・真偽・数の並び・キーフレーム）の中で壊れたもの
_ODD_VALUES: tuple[Any, ...] = (
    AnimatedValue(-1e9),
    AnimatedValue(1e9),
    AnimatedValue(0.0),
    AnimatedValue(-1.0),
    AnimatedValue(
        keyframes=(Keyframe(frame=0, value=-1e6), Keyframe(frame=10, value=1e6)),
    ),
    "abc",
    "",
    True,
    -5,
    10**12,
    (),
    (1e9, -1e9),
    (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
)


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _text() -> GeneratedSource:
    return GeneratedSource(kind="text", params={"text": "字", "size": AnimatedValue(20.0)})


def _project(effects: tuple[Effect, ...], source: GeneratedSource) -> Project:
    base = Project.create(SETTINGS)
    clip = Clip(timeline_start=0, duration=30, source=source, effects=effects)
    track = Track(TrackKind.VIDEO, "V1", (clip,))
    return base.with_timeline(replace(base.timeline, tracks=(track,)))


@pytest.mark.parametrize("kind", sorted(d.kind for d in registry.all()))
def test_odd_values_do_not_break_rendering(kind: str, gl_context: OffscreenGLContext) -> None:
    # 配布された作品の 1 つのエフェクトにおかしな値があるだけで、プレビューが止まり
    # 書き出しが途中で落ちる
    definition = registry.get(kind)
    assert definition is not None
    rng = random.Random(f"{kind}-20260917")
    renderer = FrameRenderer(_project((), _text()), context=gl_context)
    try:
        for _ in range(6):
            params = dict(definition.default_params())
            for spec in definition.parameters:
                if rng.random() < 0.6:
                    params[spec.name] = rng.choice(_ODD_VALUES)
            effect = Effect(kind=kind, params=params)
            renderer.set_project(_project((effect,), _text()))
            for frame in (0, 29):
                image = renderer.render(frame)
                assert image.shape[:2] == (SETTINGS.height, SETTINGS.width)
                assert np.isfinite(image.astype(np.float64)).all()
    finally:
        renderer.close()


@pytest.mark.parametrize("kind", sorted(d.kind for d in source_registry.all()))
def test_odd_source_values_do_not_break_rendering(
    kind: str, gl_context: OffscreenGLContext
) -> None:
    # テキストの大きさや図形の色が壊れていても、そのクリップだけが崩れて止まらないこと
    definition = source_registry.get(kind)
    assert definition is not None
    rng = random.Random(f"source-{kind}-20260917")
    renderer = FrameRenderer(_project((), _text()), context=gl_context)
    try:
        for _ in range(12):
            params = dict(definition.default_params())
            for spec in definition.parameters:
                if rng.random() < 0.6:
                    params[spec.name] = rng.choice(_ODD_VALUES)
            renderer.set_project(_project((), GeneratedSource(kind=kind, params=params)))
            image = renderer.render(0)
            assert image.shape[:2] == (SETTINGS.height, SETTINGS.width)
    finally:
        renderer.close()
