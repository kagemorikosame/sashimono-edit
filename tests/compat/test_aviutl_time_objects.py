"""時間で絵が決まる AviUtl2 のカスタムオブジェクト（ライン(移動軌跡) と 星）の読み込み

項目名は AviUtl2 に置かせたエイリアス（``kumiki_p5_line_x`` ``kumiki_p5_star_n30``）から
読み取った 名前に反して、ライン は折れ線ではなく通った跡、星 は星形ではなく星空
"""

from __future__ import annotations

from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import map_object, media_paths
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.mapped import MappedObject
from sashimono.core.model import AnimatedValue
from sashimono.core.timebase import FrameRate

LINE = """[Object]
frame=0,80
[Object.0]
effect.name=ライン(移動軌跡)
ライン幅=16.0
先端=48.0
先端角度=0.00
先端位置補正=70.0
先端図形=三角形
固定速度=0.00
描画間隔=10.0
最小間隔=2
主線描画(%)=100.0
補助描画(%)=0.0
色=ffffff
[Object.1]
effect.name=標準描画
X=-600.00,600.00,直線移動,0
Y=-300.00,300.00,直線移動,0
Z=0.00
拡大率=100.000
透明度=0.00
合成モード=通常
"""

STAR = """[Object]
frame=0,80
[Object.0]
effect.name=星
個数=30.0
速度=12.0
広がり=12.0
奥行き=20.0
サイズ=30.0
色=ddddff
形状=円
フェードイン時間=0.15
フェードアウト時間=0.15
[Object.1]
effect.name=標準描画
X=0.00
Y=0.00
合成モード=通常
"""


def _mapped(text: str) -> tuple[MappedObject, CompatibilityReport]:
    report = CompatibilityReport()
    item = map_object(parse_exo(text).objects[0], FrameRate(60), report=report)
    assert item is not None
    return item, report


def _static(value: object) -> float:
    assert isinstance(value, AnimatedValue)
    return value.static


class TestMotionTrail:
    def test_it_becomes_a_trail_that_follows_the_position(self) -> None:
        # 読めないと 中身の無いクリップになり、何も描かれない
        item, report = _mapped(LINE)
        source = item.clip.source
        assert source is not None
        assert source.params["shape"] == "motion_trail"
        x = source.params["pos_x"]
        assert isinstance(x, AnimatedValue)
        assert x.at(0) == -600.0
        assert x.at(80) == 600.0
        y = source.params["pos_y"]
        # Y は上が正へ直す AviUtl の 300（下）はこちらの -300
        assert isinstance(y, AnimatedValue)
        assert y.at(80) == -300.0
        assert not report.missing

    def test_the_position_is_not_applied_twice(self) -> None:
        # 変形にも位置を残すと、軌跡を描いた絵がもう一度ずれて 2 倍の所に出る
        item, _ = _mapped(LINE)
        for effect in item.clip.effects:
            if effect.kind == "transform":
                assert _static(effect.params.get("pos_x", AnimatedValue(0.0))) == 0.0
                assert not effect.params["pos_x"].is_animated  # type: ignore[union-attr]

    def test_the_items_are_read(self) -> None:
        # 項目を読み落として既定へ落ちると、ライン幅や先端を変えた AviUtl2 の
        # ライン(移動軌跡) が既定の太さや先端で描かれる
        item, _ = _mapped(LINE)
        source = item.clip.source
        assert source is not None
        assert _static(source.params["line_width"]) == 16.0
        assert _static(source.params["trail_head_size"]) == 48.0
        assert _static(source.params["trail_head_offset"]) == 70.0
        assert _static(source.params["trail_interval"]) == 10.0
        # 実物の三角形は円に内接する形（大きさ 48 で高さ 36）
        assert source.params["trail_head_shape"] == "inscribed_triangle"
        # 見本の値は読み込みの既定と同じで、読み落としても上は通る
        # 全部の項目を既定と違う値にして、読んだ値が届くことを見る
        changed = (
            LINE.replace("ライン幅=16.0", "ライン幅=7.0")
            .replace("先端=48.0", "先端=21.0")
            .replace("先端角度=0.00", "先端角度=30.00")
            .replace("先端位置補正=70.0", "先端位置補正=55.0")
            .replace("先端図形=三角形", "先端図形=星型")
            .replace("固定速度=0.00", "固定速度=4.00")
            .replace("描画間隔=10.0", "描画間隔=3.0")
            .replace("最小間隔=2", "最小間隔=5")
            .replace("主線描画(%)=100.0", "主線描画(%)=60.0")
            .replace("補助描画(%)=0.0", "補助描画(%)=25.0")
        )
        item, _ = _mapped(changed)
        source = item.clip.source
        assert source is not None
        read = {
            name: _static(source.params[name])
            for name in (
                "line_width",
                "trail_head_size",
                "trail_head_angle",
                "trail_head_offset",
                "trail_speed",
                "trail_interval",
                "trail_min_step",
                "trail_core",
                "trail_band",
            )
        }
        assert read == {
            "line_width": 7.0,
            "trail_head_size": 21.0,
            "trail_head_angle": 30.0,
            "trail_head_offset": 55.0,
            "trail_speed": 4.0,
            "trail_interval": 3.0,
            "trail_min_step": 5.0,
            "trail_core": 60.0,
            "trail_band": 25.0,
        }
        assert source.params["trail_head_shape"] == "star"

    def test_an_unknown_figure_is_recorded(self) -> None:
        # 自分で足した図形を黙って三角形にすると、違う先端に気付けない
        _, report = _mapped(LINE.replace("先端図形=三角形", "先端図形=ハート"))
        assert any("ハート" in entry for entry in report.missing)


class TestStarField:
    def test_it_becomes_a_star_field(self) -> None:
        # 星形の図形として読むと、流れない星が 1 つ出るだけになる
        item, report = _mapped(STAR)
        source = item.clip.source
        assert source is not None
        assert source.params["shape"] == "starfield"
        assert _static(source.params["star_count"]) == 30.0
        assert _static(source.params["star_speed"]) == 12.0
        assert _static(source.params["star_spread"]) == 12.0
        assert _static(source.params["star_depth"]) == 20.0
        assert _static(source.params["star_size"]) == 30.0
        assert _static(source.params["star_fade_in"]) == 0.15
        assert source.params["star_shape"] == "ellipse"
        assert source.params["color"][2] == 1.0  # type: ignore[index]
        assert not report.missing


#: 音声波形表示の見本が読む音（道だけを使う 開かない）
BGM = r"D:\V用のBGM\138_BPM150.mp3"

WAVE = """[Object]
frame=0,80
[Object.0]
effect.name=音声波形表示
横幅=800
高さ=400
再生位置=0.000,80.448,再生範囲,0
再生速度=100.00
音量=100.00
ファイル={bgm}
波形の色=ffffff
波形のプリセット=
スペクトラム表示=0
ミラー表示=0
横解像度=0
縦解像度=0
横スペース=0
縦スペース=0
[Object.1]
effect.name=標準描画
X=0.00
Y=0.00
合成モード=通常
""".replace("{bgm}", BGM)


class TestWaveform:
    def test_it_becomes_a_waveform_of_its_own_file(self) -> None:
        # フィルタではなくメディアオブジェクト 自分の ファイル の音を描く
        # 読めないと中身の無いクリップになり、何も描かれない
        item, report = _mapped(WAVE)
        source = item.clip.source
        assert source is not None
        assert source.params["shape"] == "waveform"
        assert source.params["audio_path"] == BGM
        assert item.media_path == BGM
        assert _static(source.params["width"]) == 800.0
        assert _static(source.params["height"]) == 400.0
        assert source.params["audio_end_ms"] == 80448
        assert not report.missing

    def test_the_media_path_is_listed(self) -> None:
        # 素材として読み込ませないと、別の機械へ持っていったときに探し直せない
        assert media_paths(parse_exo(WAVE)) == (BGM,)

    def test_an_empty_range_is_kept(self) -> None:
        # 10,10 のように始めと終わりが同じなら、読む範囲は 0 秒（実物は何も描かない）
        item, _ = _mapped(WAVE.replace("0.000,80.448,再生範囲", "10.000,10.000,再生範囲"))
        assert item.clip.source is not None
        assert item.clip.source.params["audio_end_ms"] == 10000
        assert item.clip.source_in == 10

    def test_the_grid_and_the_spectrum_are_read(self) -> None:
        # 読まないと、升目やスペクトラムの見本が細い線で描かれる
        item, report = _mapped(
            WAVE.replace("スペクトラム表示=0", "スペクトラム表示=1")
            .replace("横解像度=0", "横解像度=16")
            .replace("縦解像度=0", "縦解像度=32")
            .replace("横スペース=0", "横スペース=4")
            .replace("縦スペース=0", "縦スペース=2")
        )
        source = item.clip.source
        assert source is not None
        assert source.params["wave_spectrum"] is True
        assert _static(source.params["wave_columns"]) == 16.0
        assert _static(source.params["wave_rows"]) == 32.0
        assert _static(source.params["wave_gap_x"]) == 4.0
        assert _static(source.params["wave_gap_y"]) == 2.0
        assert not report.missing

    def test_the_preset_name_changes_nothing(self) -> None:
        # ファイルに Type1〜5 と書いた 5 本は、既定と同じ絵だった 名前から値を引き直すと、
        # 名前だけを書いたファイルが AviUtl2 と違う絵（Type3 ならスペクトラム）になる
        # 記録に並べると、本当に写せていない項目が埋もれる
        plain, _ = _mapped(WAVE)
        named, report = _mapped(WAVE.replace("波形のプリセット=", "波形のプリセット=Type3"))
        assert not report.missing
        assert named.clip.source is not None and plain.clip.source is not None
        assert named.clip.source.params == plain.clip.source.params

    def test_a_chosen_preset_is_read_from_its_values(self) -> None:
        # UI で Type5 を選ぶと、AviUtl2 は名前を空に戻して値を書き込んだ（v2.1.6a）
        # スペクトラム・ミラー・40x40・スペース 80/0 その値から同じ絵を作る
        item, report = _mapped(
            WAVE.replace("スペクトラム表示=0", "スペクトラム表示=1")
            .replace("ミラー表示=0", "ミラー表示=1")
            .replace("横解像度=0", "横解像度=40")
            .replace("縦解像度=0", "縦解像度=40")
            .replace("横スペース=0", "横スペース=80")
        )
        source = item.clip.source
        assert source is not None
        assert source.params["wave_spectrum"] is True
        assert source.params["wave_mirror"] is True
        assert _static(source.params["wave_columns"]) == 40.0
        assert _static(source.params["wave_gap_x"]) == 80.0
        assert not report.missing

    def test_the_mirror_is_read(self) -> None:
        # 前はスペクトラムと組んだミラーを写せない記録に残し、棒を下から積んでいた
        # 実物（Type5）は棒を上下の真ん中に置く 線のミラーは絵が変わらないので同じ旗のまま渡す
        item, report = _mapped(
            WAVE.replace("ミラー表示=0", "ミラー表示=1").replace(
                "スペクトラム表示=0", "スペクトラム表示=1"
            )
        )
        assert item.clip.source is not None
        assert item.clip.source.params["wave_mirror"] is True
        assert not report.missing
        plain, _ = _mapped(WAVE)
        assert plain.clip.source is not None
        assert plain.clip.source.params["wave_mirror"] is False
