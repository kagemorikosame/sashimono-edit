"""読めないバイトを持つ値を、保存して読み直しても同じに保つ

AviUtl1 のダイアログの ``"\\255"`` は、こちらでは U+DCFF の 1 文字（``surrogateescape``）
として値に入る JSON を ``ensure_ascii=False`` の UTF-8 で書くとそこで例外になり、
その値を持つプロジェクトは保存できなかった
"""

from __future__ import annotations

import json
from pathlib import Path

from sashimono.compat.aviutl.control import lua_string
from sashimono.compat.aviutl.objapi import LUA_ENCODING, ObjectState, lua_text
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.aviutl.runtime import LuaScriptRuntime, blank_image
from sashimono.core.commands import AddClip, AddTrack
from sashimono.core.io.presets import Preset, PresetStore
from sashimono.core.io.serialize import json_text, load_project, save_project
from sashimono.core.model import Clip, Effect, GeneratedSource, Project, Track, TrackKind

#: ``"a\255b"`` を戻した値 真ん中は 0xFF の 1 バイト
HIGH = lua_string('"a\\255b"')


def test_the_value_holds_the_raw_byte() -> None:
    # \255 を 1 バイトの文字として持てないと、ダイアログのその値が AviUtl1 と違う
    # バイト列でスクリプトに届く
    assert HIGH == "a\udcffb"
    # 読めないバイトは Lua との間の符号化（LUA_ENCODING）がそのまま渡すので、ここでは触らない
    assert lua_text(HIGH) == HIGH
    assert HIGH.encode(LUA_ENCODING) == b"a\xffb"


def test_a_high_byte_goes_into_lua_and_comes_back_unchanged() -> None:
    """Lua へ渡した値を、スクリプトが Python の窓口へ返しても同じバイトのまま

    lupa の既定の UTF-8 は Lua から戻る文字の 0xFF を読めずに例外を出し、
    ``obj.名前 = 値`` や ``debug_print`` へ渡した所でスクリプトが落ちていた
    """
    report = CompatibilityReport()
    runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
    state = ObjectState(image=blank_image(8, 8))
    state.values["_2"] = HIGH
    state.values["jp"] = "字幕の本文"
    result = runtime.run(
        "obj.copy = _2 obj.field = obj._2 obj.made = string.char(0x61, 0xff, 0x62) "
        "obj.jp_copy = jp obj.literal = '日本語' debug_print(_2)",
        state,
    )
    assert not result.failed, result.message
    assert state.values["copy"] == HIGH
    assert state.values["field"] == HIGH
    assert state.values["made"] == HIGH
    assert state.values["copy"].encode(LUA_ENCODING) == b"a\xffb"
    # ふつうの UTF-8 の文字はどちらの向きも変わらない
    assert state.values["jp_copy"] == "字幕の本文"
    assert state.values["literal"] == "日本語"
    assert "debug_print: " + HIGH in report.missing


def test_a_surrogate_that_is_not_a_byte_is_replaced_and_recorded() -> None:
    # U+D800 はどのバイトにも戻せない そのまま渡すと準備の段で例外になり、描画ごと落ちる
    report = CompatibilityReport()
    runtime = LuaScriptRuntime(report=report, instruction_limit=100_000)
    state = ObjectState(image=blank_image(8, 8))
    state.values["bad"] = "x\ud800y"
    result = runtime.run("obj.back = bad obj.field = obj.bad", state)
    assert not result.failed, result.message
    assert state.values["back"] == "x\ufffdy"
    assert state.values["field"] == "x\ufffdy"
    assert report.missing["Lua へ渡せない文字（対になっていない代用符号）"] >= 1


def test_a_project_with_a_high_byte_saves_and_reloads(tmp_path: Path) -> None:
    # 読めないバイトを持つ値で保存が例外になると、AviUtl1 のダイアログの値を持つ
    # プロジェクトが保存できない
    track = Track(kind=TrackKind.VIDEO, name="V1")
    project = AddTrack(track).apply(Project.create())
    clip = Clip(
        timeline_start=0,
        duration=30,
        source=GeneratedSource(kind="text", params={"text": HIGH}),
        effects=(Effect(kind="aviutl:見本.anm:0", params={"_2": HIGH}),),
    )
    project = AddClip(track.id, clip).apply(project)

    path = tmp_path / "高いバイト.sme"
    save_project(project, path)
    loaded = load_project(path)
    # 名前は読むときにファイル名から付くので、中身だけを比べる
    assert loaded.timeline == project.timeline

    (placed,) = loaded.timeline.tracks[0].clips
    assert placed.source is not None
    assert placed.source.params["text"] == HIGH
    value = placed.effects[0].params["_2"]
    assert value == HIGH

    # 読み直した値が、スクリプトに同じバイトで届く
    runtime = LuaScriptRuntime(instruction_limit=100_000)
    state = ObjectState(image=blank_image(8, 8))
    state.values["_2"] = value
    result = runtime.run("obj.second = string.byte(_2, 2) obj.size = #_2", state)
    assert not result.failed, result.message
    assert (state.values["second"], state.values["size"]) == (255, 3)


def test_the_japanese_stays_readable(tmp_path: Path) -> None:
    # ファイル全体を ensure_ascii にすると日本語が \\uXXXX になって読めなくなる
    text = json_text({"text": "字幕" + HIGH})
    assert "字幕" in text
    assert "\\udcff" in text
    assert json.loads(text) == {"text": "字幕" + HIGH}
    text.encode("utf-8")


def test_a_preset_with_a_high_byte_saves_and_reloads(tmp_path: Path) -> None:
    # 読めないバイトを持つ値のプリセットが保存で例外になるか、読み直して別の文字に
    # 変わると、当てたスクリプトに違う値が渡る
    store = PresetStore(root=tmp_path)
    preset = Preset(name="高いバイト", effects=(Effect(kind="blur", params={"memo": HIGH}),))
    path = store.save(preset)
    assert store.load(path).effects[0].params["memo"] == HIGH
