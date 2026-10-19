"""スクリプトの残り アンカー・32 ビットの DLL のモジュール・``obj.layer``（Issue #212）

- sigma の 単純変形σ は 5 本が ``obj.setanchor`` を呼ぶ 編集画面にアンカーを出すだけの物で、
  描く絵は変わらない 未対応に数えると、描けているのに互換性レポートに並ぶ
- PSDToolKit は ``require("PSDToolKitBridge")`` で 32 ビットの DLL を読む 置いてあるのに
  「見つかりません」と残すと、置き場を探し直すことになる
- PSDToolKit の字幕と吹き出しは ``obj.layer`` でレイヤーごとに状態を分ける 0 のまま渡すと、
  別のレイヤーのオブジェクトどうしが同じ状態を書き換え合う
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.compat.aviutl.catalog import ScriptCatalog
from sashimono.compat.aviutl.objapi import ObjectState
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime
from sashimono.engine.render.scripts import ScriptStage


def _run(source: str, *, folder: Path | None = None) -> tuple[ObjectState, CompatibilityReport]:
    state = ObjectState(image=np.full((4, 4, 4), 255, np.uint8), screen_w=32, screen_h=18)
    report = CompatibilityReport()
    runtime = LuaScriptRuntime(report=report)
    if folder is not None:
        runtime.set_roots((folder,))
    result = runtime.run(source, state, folder=folder)
    assert not result.failed, result.message
    return state, report


class TestAnchor:
    def test_showing_anchors_is_not_counted_as_missing(self) -> None:
        """``obj.setanchor`` は編集画面にアンカーを出すだけで、絵は変わらない（lua.txt）

        前は未対応として数え、sigma の 領域サイズ指定・回転中心 などが描けているのに
        互換性レポートに並んだ
        """
        state, report = _run(
            'local o = obj; o.setanchor("track",0,"line")'
            ' obj.setanchor("track",0,"star","xyz") obj.ox = 3'
        )
        assert not report.missing
        assert state.ox == 3.0

    def test_the_number_of_anchors_is_returned(self) -> None:
        # 戻り値はアンカーの数 nil を返すと ``for i=0,num-1`` がいきなり落ちる
        # トラックバーはいまの時刻の値 1 つだけがスクリプトから見える
        state, _ = _run(
            'obj.ox = obj.setanchor("pos", 3, "loop") obj.oy = obj.setanchor("track", 0, "line")'
            ' obj.oz = obj.setanchor("x,y,z", 0, "xyz")'
        )
        assert (state.ox, state.oy, state.oz) == (3.0, 1.0, 1.0)


def _pe(path: Path, machine: int) -> None:
    """PE の見出しだけを持つ小さなファイル 中身は読ませないので、見出しの形だけあればよい"""
    head = bytearray(128)
    head[:2] = b"MZ"
    head[60:64] = (64).to_bytes(4, "little")
    head[64:68] = b"PE\0\0"
    head[68:70] = machine.to_bytes(2, "little")
    path.write_bytes(bytes(head))


class TestCModule:
    def test_a_32_bit_bridge_is_named_as_such(self, tmp_path: Path) -> None:
        """32 ビットの C の DLL を ``require`` したら、読めない理由をそのまま残す

        前は「見つかりません」と残り、置いてあるのに読まれない理由に辿り着けなかった
        PSDToolKit は ``PSDToolKit`` のフォルダに ``PSDToolKitBridge.dll`` を置く
        """
        (tmp_path / "PSDToolKit").mkdir()
        _pe(tmp_path / "PSDToolKit" / "PSDToolKitBridge.dll", 0x014C)
        _, report = _run('local b = require("PSDToolKitBridge")', folder=tmp_path)
        (line,) = report.missing
        assert "32 ビット" in line and "PSDToolKitBridge.dll" in line

    def test_a_32_bit_bridge_deep_in_the_folder_is_named_as_such(self, tmp_path: Path) -> None:
        """置き場の 2 段より下に置いた DLL も、読めない理由で記録する（PR #218 の指摘）

        深い所の索引が Lua のモジュールだけを拾っていた頃は「見つかりません」と残った
        配布のまま置いた PSDToolKit は ``src/lua/PSDToolKit`` のように深い
        """
        deep = tmp_path / "aviutl_psdtoolkit" / "src" / "lua"
        deep.mkdir(parents=True)
        _pe(deep / "PSDToolKitBridge.dll", 0x014C)
        state = ObjectState(image=np.full((4, 4, 4), 255, np.uint8), screen_w=32, screen_h=18)
        report = CompatibilityReport()
        runtime = LuaScriptRuntime(report=report)
        runtime.set_roots((tmp_path,))
        result = runtime.run('local b = require("PSDToolKitBridge")', state)
        assert not result.failed, result.message
        (line,) = report.missing
        assert "32 ビット" in line

    def test_a_64_bit_c_module_is_not_loaded_either(self, tmp_path: Path) -> None:
        # Lua の C の窓口を呼ぶ DLL は 64 ビットでも読まない 読めない理由は DLL だから
        _pe(tmp_path / "bridge.dll", 0x8664)
        _, report = _run('local b = require("bridge")', folder=tmp_path)
        (line,) = report.missing
        assert "C モジュールの DLL" in line and "32 ビット" not in line

    def test_a_broken_dll_is_not_called_a_c_module(self, tmp_path: Path) -> None:
        # PE の見出しを読めない DLL を C のモジュールと決めつけると、壊れたファイルを置き直せば
        # 済む所で読めない理由を取り違える（PR #218 の指摘）
        (tmp_path / "bridge.dll").write_bytes(b"not a dll")
        _, report = _run('local b = require("bridge")', folder=tmp_path)
        (line,) = report.missing
        assert "読み取れなかった" in line and "C モジュール" not in line

    def test_a_module_that_is_not_there_is_still_not_found(self, tmp_path: Path) -> None:
        # 無いモジュールを「見つかりません」と名前付きで残せないと、
        # 互換性レポートからスクリプトの置き忘れたモジュールに気付けない
        _, report = _run('local b = require("nothing")', folder=tmp_path)
        assert list(report.missing) == ['モジュール "nothing" が見つかりません']


class TestLayer:
    def test_the_text_lua_sees_its_layer(self) -> None:
        """テキスト欄の Lua にも ``obj.layer`` を渡す

        PSDToolKit の字幕はテキストの Lua でレイヤー番号ごとに状態を置く 渡さないと 0 になり、
        別のレイヤーの字幕が同じ所へ書く
        """
        stage = ScriptStage(ScriptCatalog(roots=()), screen=(320, 180))
        text = stage.expand_text("<?mes(obj.layer)?>", frame=0, fps=30.0, duration=30, layer=3)
        assert text == "3"


@pytest.fixture(scope="module")
def gl_context() -> Iterator[Any]:
    from sashimono.engine.gpu import GLContextError, OffscreenGLContext

    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _layered_project(script: str) -> Any:
    """1 本目の映像トラックは空、2 本目にスクリプトのクリップを置いた 64x64 のプロジェクト"""
    from sashimono.compat.aviutl import catalog as catalog_module
    from sashimono.compat.aviutl.catalog import set_script_catalog
    from sashimono.compat.aviutl.custom_object import custom_object_clip
    from sashimono.core.commands import AddClip, AddTrack
    from sashimono.core.model import Project, ProjectSettings, Track, TrackKind
    from sashimono.core.timebase import FrameRate
    from sashimono.effects import registry

    saved = catalog_module._catalog
    catalog = ScriptCatalog(roots=())
    catalog.add_text("aviutl:試験/@層.obj:層", script, kind="obj")
    set_script_catalog(catalog)
    try:
        definition = registry.get("aviutl:試験/@層.obj:層")
        assert definition is not None
        project = Project.create(ProjectSettings(width=64, height=64, frame_rate=FrameRate(30)))
        first = Track(kind=TrackKind.VIDEO, name="V1")
        second = Track(kind=TrackKind.VIDEO, name="V2")
        project = AddTrack(second).apply(AddTrack(first).apply(project))
        clip = custom_object_clip(definition.create(), duration=10)
        return AddClip(second.id, clip).apply(project), saved
    except BaseException:
        catalog_module._catalog = saved
        raise


def test_a_script_sees_the_layer_it_is_placed_on(gl_context: Any) -> None:
    """スクリプトの ``obj.layer`` は置かれたトラックの番号（奥から 1）

    前はいつも 0 で、レイヤーごとに状態を分けるスクリプトが別のレイヤーと混ざった
    番号の大きさの四角を描かせて、2 本目のトラックでは 20 画素四方になることを見る
    """
    from sashimono.compat.aviutl import catalog as catalog_module
    from sashimono.engine.render import FrameRenderer

    project, saved = _layered_project('obj.load("figure", "四角形", 0xffffff, 10 * obj.layer)')
    try:
        renderer = FrameRenderer(project, context=gl_context)
        try:
            image = renderer.render(0)
        finally:
            renderer.close()
    finally:
        catalog_module._catalog = saved
        if saved is not None:
            saved.register_all()
    lit = image[..., 0] > 128
    rows, columns = np.nonzero(lit)
    assert rows.max() - rows.min() + 1 == 20
    assert columns.max() - columns.min() + 1 == 20


def test_video_and_mixed_tracks_are_counted_together(gl_context: Any) -> None:
    """映像と混合のトラックが並ぶタイムラインでは、絵を描くトラックをまとめて数える

    種類ごとに数えていた頃は、1 本目の映像と 1 本目の混合の両方に 1 を渡し、番号で状態を
    分けるスクリプトで別のトラックの状態が混ざった（PR #218 の指摘） 音声トラックは数えない
    """
    from dataclasses import replace

    from sashimono.core.model import (
        Clip,
        GeneratedSource,
        Project,
        ProjectSettings,
        Timeline,
        Track,
        TrackKind,
    )
    from sashimono.core.timebase import FrameRate
    from sashimono.engine.render import FrameRenderer

    rate = FrameRate(30)
    below = Clip(timeline_start=0, duration=10, source=GeneratedSource(kind="text", params={}))
    above = Clip(timeline_start=0, duration=10, source=GeneratedSource(kind="text", params={}))
    main = Timeline(
        rate=rate,
        tracks=(
            Track(TrackKind.VIDEO, "V1", (below,)),
            Track(TrackKind.AUDIO, "A1"),
            Track(TrackKind.MIXED, "L1", (above,)),
        ),
    )
    project = replace(
        Project.create(ProjectSettings(width=64, height=64, frame_rate=rate)), timeline=main
    )
    renderer = FrameRenderer(project, context=gl_context)
    try:
        assert renderer._layer_number(below) == 1
        assert renderer._layer_number(above) == 2
    finally:
        renderer.close()


def test_a_clip_id_shared_with_a_scene_keeps_its_own_layer(gl_context: Any) -> None:
    """別のシーンに同じ識別子のクリップがあっても、クリップごとに自分のトラックの番号を渡す

    識別子だけで引いていた頃は、後から数えたシーンの番号でメインの番号が上書きされた
    （PR #218 の指摘） メインでは 2 本目、シーンでは 1 本目に同じ識別子のクリップを置く
    """
    from dataclasses import replace

    from sashimono.core.model import (
        Clip,
        GeneratedSource,
        Project,
        ProjectSettings,
        Scene,
        Timeline,
        Track,
        TrackKind,
    )
    from sashimono.core.timebase import FrameRate
    from sashimono.engine.render import FrameRenderer

    rate = FrameRate(30)
    clip = Clip(timeline_start=0, duration=10, source=GeneratedSource(kind="text", params={}))
    twin = replace(clip, duration=20)
    main = Timeline(
        rate=rate,
        tracks=(Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2", (clip,))),
    )
    scene = Scene(
        name="場面", timeline=Timeline(rate=rate, tracks=(Track(TrackKind.VIDEO, "V1", (twin,)),))
    )
    project = replace(
        Project.create(ProjectSettings(width=64, height=64, frame_rate=rate)),
        timeline=main,
        scenes=(scene,),
    )
    assert clip.id == twin.id
    renderer = FrameRenderer(project, context=gl_context)
    try:
        assert renderer._layer_number(clip) == 2
        assert renderer._layer_number(twin) == 1
        # フィルタや切り抜きは描く前にクリップを作り直す 実体で引けなくても、置かれた
        # トラックを渡せばシーンの番号になる 識別子へ落ちるとメインの 2 になる（PR #218 の指摘）
        remade = replace(twin, opacity=twin.opacity)
        assert renderer._layer_number(remade, scene.timeline.tracks[0]) == 1
    finally:
        renderer.close()
