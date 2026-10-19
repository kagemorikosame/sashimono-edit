"""スクリプトの走査と、エフェクトとしての登録

登録できていれば、設定 UI もプリセットもキーフレームも自前のエフェクトと
同じ経路で動く ここはその継ぎ目を見る
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sashimono.compat.aviutl.catalog import ScriptCatalog, set_script_catalog
from sashimono.effects.definition import registry
from sashimono.effects.spec import TrackSpec

ANM = """--track0:振れ幅,0,200,20,1
--track1:速さ,0,10,2,0.1
--check0:横に振る,0
local amount = obj.track0 * math.sin(obj.time * obj.track1)
if obj.check0 == 1 then obj.ox = amount else obj.oy = amount end
"""

MULTI = """@ゆらゆら
--track0:幅,0,100,10
obj.ox = obj.track0
@ぐるぐる
--track0:速さ,0,10,1
obj.rz = obj.track0 * obj.time
"""


@pytest.fixture
def script_dir(tmp_path: Path) -> Path:
    root = tmp_path / "scripts"
    (root / "動き").mkdir(parents=True)
    (root / "動き" / "ゆれ.anm").write_text(ANM, encoding="utf-8")
    (root / "まとめ.anm").write_text(MULTI, encoding="utf-8")
    # AviUtl1 世代は Shift_JIS
    (root / "旧世代.obj").write_text("--track0:大きさ,1,100,50\n", encoding="cp932")
    return root


class TestScanning:
    def test_scripts_are_found_recursively(self, script_dir: Path) -> None:
        # サブフォルダを見ないと、スクリプトフォルダの中で分けて置いた .anm が効果の一覧に出ない
        catalog = ScriptCatalog(roots=(script_dir,))
        entries = catalog.scan()
        assert len(entries) == 4  # ゆれ + まとめ 2 本 + 旧世代

    def test_the_kind_comes_from_the_suffix(self, script_dir: Path) -> None:
        # 拡張子から種類を取り違えると、タイムラインの〔追加〕のカスタムオブジェクトに
        # .obj が出ないか、.anm が混ざる
        catalog = ScriptCatalog(roots=(script_dir,))
        catalog.scan()
        assert [e.label for e in catalog.of_kind("obj")] == ["旧世代"]
        assert len(catalog.of_kind("anm")) == 3

    def test_a_file_with_markers_yields_several_scripts(self, script_dir: Path) -> None:
        # @ で区切った 1 ファイルを 1 本として読むと、まとめて配られた物の 2 本目から使えない
        catalog = ScriptCatalog(roots=(script_dir,))
        catalog.scan()
        labels = {entry.label for entry in catalog.all()}
        assert {"ゆらゆら", "ぐるぐる"} <= labels

    def test_shift_jis_files_are_read(self, script_dir: Path) -> None:
        # AviUtl1 世代の Shift_JIS のスクリプトの設定欄の名前が文字化けするか、一覧から消える
        catalog = ScriptCatalog(roots=(script_dir,))
        catalog.scan()
        entry = catalog.of_kind("obj")[0]
        assert entry.header.parameters[0].label == "大きさ"

    def test_a_missing_folder_is_not_an_error(self, tmp_path: Path) -> None:
        # 無い探索先で例外を出すと、AviUtl2 を入れていない人の所では走査が止まり、
        # 置いたスクリプトが 1 本も読めない
        assert ScriptCatalog(roots=(tmp_path / "無い",)).scan() == ()

    def test_identifiers_are_stable(self, script_dir: Path) -> None:
        # プロジェクトファイルに出るので、走査のたびに変わってはいけない
        first = {entry.identifier for entry in ScriptCatalog(roots=(script_dir,)).scan()}
        second = {entry.identifier for entry in ScriptCatalog(roots=(script_dir,)).scan()}
        assert first == second

    def test_identifiers_carry_the_relative_path(self, script_dir: Path) -> None:
        # 識別子にフォルダの中の位置が入らないと、別のフォルダの同じ名前のスクリプトが
        # 同じ識別子になり、後に読んだ方だけが残る
        catalog = ScriptCatalog(roots=(script_dir,))
        catalog.scan()
        entry = next(e for e in catalog.all() if e.label == "ゆれ")
        assert entry.identifier.startswith("aviutl:動き/ゆれ.anm")


class TestRegistration:
    def test_a_script_becomes_an_effect_definition(self, script_dir: Path) -> None:
        # 登録簿へ入らないと、スクリプトが効果の一覧に出ず、置いたクリップでも
        # --track0 などの設定欄が出ない
        catalog = ScriptCatalog(roots=(script_dir,))
        catalog.scan()
        assert catalog.register_all() == 4

        entry = next(e for e in catalog.all() if e.label == "ゆれ")
        definition = registry.get(entry.identifier)
        assert definition is not None
        assert definition.category == "アニメーション効果"
        assert [spec.name for spec in definition.parameters] == ["track0", "track1", "check0"]

    def test_the_parameters_keep_their_ranges(self, script_dir: Path) -> None:
        # --track0 の最小・最大・既定が設定欄に届かず、0 から 200 の振れ幅が別の範囲で動く
        catalog = ScriptCatalog(roots=(script_dir,))
        catalog.scan()
        catalog.register_all()
        entry = next(e for e in catalog.all() if e.label == "ゆれ")
        definition = registry.get(entry.identifier)
        assert definition is not None
        spec = definition.parameters[0]
        assert isinstance(spec, TrackSpec)
        assert (spec.minimum, spec.maximum, spec.default) == (0.0, 200.0, 20.0)

    def test_a_script_has_no_shader(self, script_dir: Path) -> None:
        # GPU のエフェクト処理はシェーダの無い定義を飛ばす Lua の分岐は
        # 描画側が種別で行う
        catalog = ScriptCatalog(roots=(script_dir,))
        catalog.scan()
        catalog.register_all()
        entry = catalog.all()[0]
        definition = registry.get(entry.identifier)
        assert definition is not None
        assert definition.fragment_shader is None

    def test_effects_can_be_created_from_the_definition(self, script_dir: Path) -> None:
        # 定義から効果を作れないと、スクリプトの効果を足しても渡した track0 の値が入らない
        catalog = ScriptCatalog(roots=(script_dir,))
        catalog.scan()
        catalog.register_all()
        entry = next(e for e in catalog.all() if e.label == "ゆれ")
        definition = registry.get(entry.identifier)
        assert definition is not None
        effect = definition.create(track0=100)
        assert effect.kind == entry.identifier
        assert float(effect.params["track0"].at(0)) == 100.0  # type: ignore[union-attr]


class TestAddText:
    def test_a_script_can_be_added_without_a_file(self) -> None:
        # 足した 1 本を get で引けないと、呼び手はファイル無しで足した直後のスクリプトを使えない
        catalog = ScriptCatalog(roots=())
        entry = catalog.add_text("aviutl:ためし.anm:試", "--track0:量,0,10,1\nobj.ox = obj.track0")
        assert catalog.get(entry.identifier) is entry
        assert entry.header.parameters[0].label == "量"

    def test_setting_the_shared_catalogue_registers_it(self) -> None:
        # 差し替えた一覧が登録簿へ入らないと、渡したスクリプトの効果を
        # 描画も設定欄も見つけられない
        catalog = ScriptCatalog(roots=())
        entry = catalog.add_text("aviutl:共有.anm:共", "--track0:A,0,1,0")
        set_script_catalog(catalog)
        assert registry.get(entry.identifier) is not None
