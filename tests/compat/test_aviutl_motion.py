"""AviUtl のトラックバーに付く「移動方法」

書き方は AviUtl2（v2.1.6a）に実際に作らせたエイリアスから読み取った
ここに並ぶ文字列は、その実物から取った行そのもの
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import _spec_value, map_object
from sashimono.compat.aviutl.motion import Motion, animated_value, parse_motion
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.mapped import MappedObject
from sashimono.core.model import AnimatedValue, Interpolation
from sashimono.core.timebase import FrameRate
from sashimono.effects.spec import CheckSpec, SelectSpec, TrackSpec

RATE = FrameRate(60)


def _object(draw: str, *, frame: str = "0,59", content: str = "テキスト=あ") -> str:
    """1 オブジェクトぶんの本文 ``draw`` は標準描画に入れる行"""
    return f"""[Object]
frame={frame}
[Object.0]
effect.name=テキスト
サイズ=64.00
{content}
[Object.1]
effect.name=標準描画
{draw}
合成モード=通常
"""


def _mapped(draw: str, *, frame: str = "0,59") -> tuple[MappedObject, CompatibilityReport]:
    report = CompatibilityReport()
    document = parse_exo(_object(draw, frame=frame))
    item = map_object(document.objects[0], RATE, report=report)
    assert item is not None
    return item, report


def _transform(draw: str, *, frame: str = "0,59") -> tuple[dict[str, object], CompatibilityReport]:
    item, report = _mapped(draw, frame=frame)
    effects = item.clip.effects
    assert effects, "変形エフェクトが作られていない"
    return dict(effects[0].params), report


def _animated(value: object) -> AnimatedValue:
    assert isinstance(value, AnimatedValue)
    return value


class TestParsing:
    def test_a_plain_number_has_no_method(self) -> None:
        # 動きの無いただの数を読めないと、縦横比=0.000 のような行まで既定値へ置き換わり、
        # 互換性レポートが読めない値の行で埋まる
        assert parse_motion("0.000") == Motion(values=(0.0,))

    def test_values_method_and_flags(self) -> None:
        # 値と移動方法と旗の切れ目を取り違えると、AviUtl で動いていた項目が読み込み後に
        # 止まるか、別の移動方法として記録される
        motion = parse_motion("-0.01,149.16,300.00,直線移動,0")
        assert motion == Motion(values=(-0.01, 149.16, 300.0), method="直線移動", flags=0)

    def test_the_script_setting_comes_after_the_bar(self) -> None:
        # 反復移動 と 回転 は .tra2 のスクリプト 設定値が旗の後ろに付く
        assert parse_motion("0,0,回転,4|360") == Motion(
            values=(0.0, 0.0), method="回転", flags=4, extra="360"
        )

    def test_an_expression_keeps_its_commas(self) -> None:
        # 参照式にはカンマが入りうる 先に割ると式が途中で切れる
        motion = parse_motion("100.000,200.000,瞬間移動,8|max(0,time)")
        assert motion is not None
        assert motion.extra == "max(0,time)"
        assert motion.flags == 8

    def test_text_is_not_a_number(self) -> None:
        # None で例外を出すと項目の無いオブジェクトで .exo の読み込みが止まり、
        # 文字を数として返すと既定値へ戻したことが互換性レポートに残らない
        assert parse_motion("Rounded Mplus 1c Black") is None
        assert parse_motion(None) is None


class TestKeyframes:
    def test_a_middle_point_becomes_a_keyframe(self) -> None:
        # 中間点を置けないと、途中で 150 を通る AviUtl の動きが 0 から 300 への直線になる
        params, _ = _transform("X=0,150,300,直線移動,0", frame="244,333,423")
        pos_x = _animated(params["pos_x"])
        placed = [(k.frame, k.value) for k in pos_x.keyframes]
        assert placed == [(0, 0.0), (89, 150.0), (179, 300.0)]
        assert pos_x.at(89) == 150.0

    def test_two_values_span_the_whole_clip(self) -> None:
        # 中間点より値が少ないファイル 動きは両端へ寄せて残し、落ちた分は記録する
        # ここが壊れると、普通の時間制御まで「値数不足」として記録され、
        # 多い順の並びが実際の穴と食い違う
        params, report = _transform("X=0,300,直線移動(時間制御),0", frame="244,333,423")
        pos_x = _animated(params["pos_x"])
        assert [k.frame for k in pos_x.keyframes] == [0, 179]
        # 時間制御は中間点を見ない決まりなので、値 2 つは壊れた形ではない
        assert not any("中間点より値が少ない" in line for line in report.lines())

    def test_an_instant_move_holds_its_value(self) -> None:
        # 瞬間移動で切り替わる拡大率が、読み込み後は区間をかけてじわじわ大きくなる
        params, _ = _transform("拡大率=100,200,瞬間移動,0")
        scale = _animated(params["scale"])
        assert scale.keyframes[0].interpolation is Interpolation.HOLD
        assert scale.at(30) == 100.0
        assert scale.at(59) == 200.0

    def test_the_y_axis_is_flipped(self) -> None:
        # AviUtl の Y は下が正 こちらは上が正
        params, _ = _transform("Y=0,100,直線移動,0")
        assert _animated(params["pos_y"]).at(59) == -100.0

    def test_a_still_value_stays_still(self) -> None:
        # 同じ値が並ぶだけならキーフレームは要らない
        params, _ = _transform("X=50,50,50,直線移動,0", frame="0,30,59")
        assert _animated(params["pos_x"]) == AnimatedValue(50.0)


class TestAcceleration:
    """加速と減速は旗で付く 補間移動でだけ選べる"""

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            (0, Interpolation.LINEAR),
            (1, Interpolation.EASE_IN),
            (2, Interpolation.EASE_OUT),
            (3, Interpolation.EASE_IN_OUT),
        ],
    )
    def test_the_flags_choose_the_curve(self, flags: int, expected: Interpolation) -> None:
        # 旗を取り違えると、AviUtl で減速して止まる動きが加速して止まるなど、緩急が逆になる
        params, _ = _transform(f"X=0,300,補間移動,{flags}")
        assert _animated(params["pos_x"]).keyframes[0].interpolation is expected


class TestWhatCannotBeCopied:
    def test_a_script_move_is_recorded_and_frozen(self) -> None:
        # ランダム移動は乱数で動く 直線で動かすと実物と違う動きが出たまま気付けない
        params, report = _transform("X=10,300,ランダム移動,4|15")
        assert _animated(params["pos_x"]) == AnimatedValue(10.0)
        assert any("ランダム移動" in line for line in report.lines())

    def test_an_expression_is_recorded_and_frozen(self) -> None:
        # 参照式を値の並びのまま動かすと、AviUtl と違う動きが出たまま
        # 互換性レポートにも残らない
        params, report = _transform("拡大率=150,200,瞬間移動,8|100+time*10")
        assert _animated(params["scale"]) == AnimatedValue(150.0)
        assert any("参照式" in line for line in report.lines())

    def test_time_control_keeps_the_values_and_is_recorded(self) -> None:
        # 時間制御付きの動きが止まるか、時間の伸縮を写せていないことが
        # 互換性レポートに出ず、速さの違いに気付けない
        params, report = _transform("X=0,300,直線移動(時間制御),0")
        assert _animated(params["pos_x"]).is_animated
        assert any("時間制御" in line for line in report.lines())

    def test_an_unknown_method_is_recorded(self) -> None:
        # 知らない移動方法を直線で動かすと AviUtl と違う動きが出て、
        # 互換性レポートにも名前が残らず、次に埋める穴が分からない
        params, report = _transform("X=10,300,未知の移動,0")
        assert _animated(params["pos_x"]) == AnimatedValue(10.0)
        assert any("未知の移動" in line for line in report.lines())

    def test_a_spline_says_it_is_an_approximation(self) -> None:
        # 補間移動をイージングで近似していることが互換性レポートに出ず、曲線の違いに気付けない
        _, report = _transform("X=0,300,補間移動,3")
        assert any("補間移動" in line for line in report.lines())


class TestTheObjectSpan:
    def test_a_middle_point_does_not_shorten_the_object(self) -> None:
        # ``frame=244,333,423`` の 2 つ目を終了として読んでいた
        document = parse_exo(_object("X=0", frame="244,333,423"))
        obj = document.objects[0]
        assert (obj.start, obj.end) == (244, 423)
        assert obj.duration == 180
        assert obj.relative_points() == (0, 89, 179)

    def test_without_a_middle_point_the_points_are_the_two_ends(self) -> None:
        # 中間点の無いオブジェクトで両端が取れないと、X=0,300 のような 2 つの値の動きが
        # 中間点と数が合わないとして止まる
        document = parse_exo(_object("X=0", frame="0,59"))
        assert document.objects[0].relative_points() == (0, 59)


class TestPlayback:
    def test_the_playback_range_becomes_the_source_position(self) -> None:
        # 実物の書き方 ``再生位置=0.967,6.151,再生範囲,0`` 値は秒
        body = """[Object]
frame=0,310
[Object.0]
effect.name=動画ファイル
再生位置=0.967,6.151,再生範囲,0
再生速度=200.00
ファイル=D:/録画先/a.mp4
[Object.1]
effect.name=映像再生
X=0.00
"""
        report = CompatibilityReport()
        item = map_object(parse_exo(body).objects[0], RATE, report=report)
        assert item is not None
        assert item.clip.source_in == Fraction(967, 1000)
        assert item.clip.speed == Fraction(2)
        # 再生範囲の 2 つの値は切り出しの始めと終わりで、動きではない
        # 動きとして数えると、普通のクリップまで未対応の記録で埋まる
        assert not any("再生位置" in line for line in report.lines())

    def test_a_missing_speed_means_normal(self) -> None:
        # 再生速度の書かれていない動画が 1 倍以外で再生されるか、素材の頭から始まらない
        item, _ = _mapped("X=0")
        assert item.clip.speed == Fraction(1)
        assert item.clip.source_in == Fraction(0)

    def test_a_changing_speed_is_recorded(self) -> None:
        # クリップの速度は 1 つしか持てない 黙って先頭へ固定すると気付けない
        body = """[Object]
frame=0,310
[Object.0]
effect.name=動画ファイル
再生位置=1.0,2.0,直線移動,0
再生速度=100.00,200.00,直線移動,0
ファイル=D:/a.mp4
[Object.1]
effect.name=映像再生
X=0.00
"""
        report = CompatibilityReport()
        item = map_object(parse_exo(body).objects[0], RATE, report=report)
        assert item is not None
        assert item.clip.speed == Fraction(1)
        assert any("変速" in line for line in report.lines())
        assert any("動く再生位置" in line for line in report.lines())

    def test_an_unknown_playback_method_is_recorded(self) -> None:
        # 知らない移動方法は中身が分からない 止めたことを残さないと気付けない
        body = """[Object]
frame=0,310
[Object.0]
effect.name=動画ファイル
再生位置=1.0,1.0,未知の再生,0
ファイル=D:/a.mp4
[Object.1]
effect.name=映像再生
X=0.00
"""
        report = CompatibilityReport()
        item = map_object(parse_exo(body).objects[0], RATE, report=report)
        assert item is not None
        assert any("動く再生位置" in line for line in report.lines())

    def test_a_playback_value_that_is_not_a_number_is_recorded(self) -> None:
        # 記録しないと、指定した速度が黙って 1 倍へ置き換わる
        body = """[Object]
frame=0,310
[Object.0]
effect.name=動画ファイル
再生速度=はやい
ファイル=D:/a.mp4
[Object.1]
effect.name=映像再生
X=0.00
"""
        report = CompatibilityReport()
        item = map_object(parse_exo(body).objects[0], RATE, report=report)
        assert item is not None
        assert item.clip.speed == Fraction(1)
        assert any("数として読めない" in line for line in report.lines())

    def test_a_still_script_playback_is_recorded(self) -> None:
        # 値が同じでもスクリプトの移動方法なら時間で変わりうる
        body = """[Object]
frame=0,310
[Object.0]
effect.name=動画ファイル
再生位置=100,100,回転,4|360
ファイル=D:/a.mp4
[Object.1]
effect.name=映像再生
X=0.00
"""
        report = CompatibilityReport()
        item = map_object(parse_exo(body).objects[0], RATE, report=report)
        assert item is not None
        assert any("動く再生位置" in line for line in report.lines())


class TestTheOldReaders:
    def test_number_falls_back_to_the_first_value(self) -> None:
        # 以前はここで既定値へ落ちて、動きの付いた項目が黙って消えていた
        document = parse_exo(_object("X=0,300,直線移動,0"))
        draw = document.objects[0].find("標準描画")
        assert draw is not None
        assert draw.number("X") == 0.0
        assert draw.motion("X") is not None


def test_a_value_without_points_stays_still() -> None:
    # 区間が 1 フレームだと置き場が無い 先頭の値で止める
    report = CompatibilityReport()
    value = animated_value("0,300,直線移動,0", points=(0,), log=report, label="試し")
    assert value == AnimatedValue(0.0)


class TestTheThingsThatUsedToBreak:
    def test_a_draw_without_opacity_stays_opaque(self) -> None:
        # 透明度が書かれていない描画設定で、既定値 0 をそのまま不透明度にしていた
        item, _ = _mapped("X=100")
        assert item.clip.opacity == AnimatedValue(1.0)

    def test_a_one_frame_object_does_not_crash(self) -> None:
        # 同じフレームに 2 つキーフレームを置くと AnimatedValue が例外を出す
        params, report = _transform("X=10,300,直線移動,0", frame="5,5")
        assert _animated(params["pos_x"]) == AnimatedValue(10.0)
        assert any("中間点と値の数が合わない" in line for line in report.lines())

    def test_points_that_go_backwards_are_refused(self) -> None:
        # 断らないと同じフレームへ 2 つキーフレームが並び、AnimatedValue が例外を出す
        params, _ = _transform("X=10,20,30,直線移動,0", frame="10,20,15")
        assert _animated(params["pos_x"]) == AnimatedValue(10.0)

    def test_a_value_that_is_not_a_number_is_recorded(self) -> None:
        # 既定値へ置き換えたことを残さないと、設定が消えたことに気付けない
        params, report = _transform("X=10\n拡大率=おかしな値")
        assert _animated(params["scale"]) == AnimatedValue(100.0)
        assert any("数として読めない値" in line for line in report.lines())

    def test_a_still_expression_is_still_recorded(self) -> None:
        # 値が同じでも、式なら時間で変わりうる
        _, report = _transform("X=100,100,瞬間移動,8|100+time*10")
        assert any("参照式" in line for line in report.lines())

    def test_a_track_value_is_clamped_to_its_range(self) -> None:
        # 範囲の外の値をそのまま持つと、エフェクトの仕様と食い違う
        report = CompatibilityReport()
        document = parse_exo(
            """[Object]
frame=0,59
[Object.0]
effect.name=テキスト
テキスト=あ
[Object.1]
effect.name=ぼかし
範囲=0,100000,直線移動,0
"""
        )
        item = map_object(document.objects[0], RATE, report=report)
        assert item is not None
        blur = next(effect for effect in item.clip.effects if effect.kind == "blur")
        radius = blur.params["radius"]
        assert isinstance(radius, AnimatedValue)
        assert radius.at(59) < 100000


class TestHowValuesReachTheSpec:
    """動きを読めるのはトラックバーだけ ほかは元の文字列のまま渡す"""

    def test_a_track_becomes_an_animated_value(self) -> None:
        # ここが壊れると、動きを読めずに数値 1 つ（か既定値）へ落ちる
        spec = TrackSpec("radius", "範囲", 0, 100, 10)
        value = _spec_value(spec, "0,50,直線移動,0", (0, 59), CompatibilityReport(), "試し")
        assert isinstance(value, AnimatedValue)
        assert value.at(59) == 50.0

    def test_a_check_keeps_its_text(self) -> None:
        # AnimatedValue に包むと coerce が型違いとして既定値（False）へ落とし、
        # スクリプトのチェック項目が指定と逆になる
        spec = CheckSpec("bold", "太字", default=False)
        value = _spec_value(spec, "true", (0, 59), CompatibilityReport(), "試し")
        assert value == "true"
        assert spec.coerce("true") is True

    def test_a_select_keeps_its_text(self) -> None:
        # 包むと選択肢が既定値（linear）に戻り、形が別物になる
        spec = SelectSpec("shape", "形状", (("linear", "線形"), ("radial", "円形")), "linear")
        value = _spec_value(spec, "radial", (0, 59), CompatibilityReport(), "試し")
        assert value == "radial"
        assert spec.coerce("radial") == "radial"
