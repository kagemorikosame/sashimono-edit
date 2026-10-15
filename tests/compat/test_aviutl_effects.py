"""AviUtl の効果を、こちらのエフェクトへ写す

項目の名前と既定値は AviUtl2（v2.1.6a）に効果を積んだエイリアスを作らせて
読み取った ここに並ぶ名前はその実物から取ったもので、推測していない
"""

from __future__ import annotations

import pytest

from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.core.model import AnimatedValue, Effect
from sashimono.core.timebase import FrameRate

RATE = FrameRate(60)


def _object(*filters: str) -> str:
    """図形 1 つに ``filters`` を積んだ本文 各要素は ``名前\n項目=値`` の形"""
    blocks = [
        "[Object]",
        "frame=0,59",
        "[Object.0]",
        "effect.name=テキスト",
        "テキスト=あ",
        "[Object.1]",
        "effect.name=標準描画",
        "X=0.00",
        "合成モード=通常",
    ]
    for index, block in enumerate(filters, start=2):
        name, _, body = block.partition("\n")
        blocks.append(f"[Object.{index}]")
        blocks.append(f"effect.name={name}")
        blocks.extend(line for line in body.splitlines() if line)
    return "\n".join(blocks) + "\n"


def _effects(*filters: str) -> tuple[list[Effect], CompatibilityReport]:
    report = CompatibilityReport()
    item = map_object(parse_exo(_object(*filters)).objects[0], RATE, report=report)
    assert item is not None
    # 標準描画は既定のままなので変形は作られない 積んだフィルタだけが並ぶ
    return list(item.clip.effects), report


def _one(*filters: str) -> Effect:
    effects, _ = _effects(*filters)
    assert len(effects) == 1, [effect.kind for effect in effects]
    return effects[0]


def _value(effect: Effect, name: str) -> float:
    value = effect.params[name]
    assert isinstance(value, AnimatedValue)
    return value.static


class TestTheTableIsKeyedByTheAviutlName:
    """``座標`` ``拡大率`` ``回転`` はどれも変形へ写すが、``X`` の意味が違う"""

    def test_the_coordinate_filter_moves(self) -> None:
        effect = _one("座標\nX=100\nY=50\nZ=0")
        assert effect.kind == "transform"
        assert _value(effect, "pos_x") == 100.0
        # AviUtl の Y は下が正 こちらは上が正
        assert _value(effect, "pos_y") == -50.0

    def test_the_zoom_filter_scales_both_axes_once(self) -> None:
        # 種別で引いていたころは、ここで X が位置として読まれていた
        # 実物の配布物の 拡大率 は X Y Z がどれも 100 で書かれていた
        # 変形の scale_y は scale に重ねて掛かる縦の比 200 を入れると縦だけ 400% になる
        # （AviUtl2 v2.1.6a では 200x200 の四角が 400x400 #167）
        effect = _one("拡大率\n拡大率=200.000\nX=100.000\nY=100.000\nZ=100.000")
        assert _value(effect, "scale") == 200.0
        assert _value(effect, "scale_y") == 100.0
        assert _value(effect, "pos_x") == 0.0

    def test_a_flip_reached_only_by_a_later_keyframe_is_recorded(self) -> None:
        # Y が 100 から -100 へ動くと AviUtl2 では途中で裏返る 変形は裏返せないので
        # 大きさだけを写す 初めの値だけを見ると記録が空のまま裏返しが消える（#183 のレビュー）
        effects, report = _effects("拡大率\n拡大率=100\nX=100\nY=100,-100,直線移動,0\nZ=100")
        assert len(effects) == 1
        assert any("負の拡大率" in line for line in report.lines())

    def test_the_zoom_filter_x_and_y_scale_each_axis(self) -> None:
        # X と Y は拡大率に重ねて掛かる軸ごとの拡大率 AviUtl2 で拡大率 150・X 200・Y 50 の
        # 200x200 の四角は 600x150 だった 横 = 150 x 2 縦 = 横 x (50 / 200)
        effect = _one("拡大率\n拡大率=150\nX=200\nY=50\nZ=100")
        assert _value(effect, "scale") == 300.0
        assert _value(effect, "scale_y") == 25.0

    def test_the_rotate_filter_turns(self) -> None:
        effect = _one("回転\nX=0\nY=0\nZ=45")
        assert _value(effect, "rotation") == 45.0


class TestValuesThatNeedFixing:
    def test_transparency_becomes_opacity(self) -> None:
        # AviUtl の透明度は 0 で不透明 そのまま入れると真っ透明になる
        effect = _one("透明度\n透明度=30")
        assert effect.kind == "opacity"
        assert _value(effect, "amount") == 70.0

    def test_the_shadow_offset_flips(self) -> None:
        effect = _one("ドロップシャドウ\nX=10\nY=20\n濃さ=80\n拡散=5\n影色=000000")
        assert effect.kind == "shadow"
        assert _value(effect, "offset_y") == -20.0
        assert _value(effect, "blur") == 5.0

    def test_a_shake_fills_both_axes(self) -> None:
        # 震えるは振幅を 1 つしか持たない 片方だけ入れると横にしか揺れない
        effect = _one("震える\n振幅=8\n角度=0\n間隔=2")
        assert effect.kind == "random_move"
        assert _value(effect, "range_x") == 8.0
        assert _value(effect, "range_y") == 8.0

    def test_a_quarter_turn_becomes_degrees(self) -> None:
        effect = _one("ローテーション\n90度回転=2")
        assert _value(effect, "rotation") == 180.0


class TestAppearance:
    """登場と退場 AviUtl はフェードとワイプだけ「イン」「アウト」を秒で持つ"""

    def test_fade_reads_both_ends(self) -> None:
        effect = _one("フェード\nイン=0.50\nアウト=0.50")
        assert effect.kind == "inout_fade"
        assert effect.params["effect_in"] is True
        assert effect.params["effect_out"] is True
        assert _value(effect, "effect_time") == 0.5

    def test_only_an_out_time_means_no_entrance(self) -> None:
        effect = _one("フェード\nイン=0\nアウト=1.0")
        assert effect.params["effect_in"] is False
        assert effect.params["effect_out"] is True
        assert _value(effect, "effect_time") == 1.0

    def test_two_different_times_are_recorded(self) -> None:
        # こちらは時間を 1 つしか持てない 黙って片方を捨てると気付けない
        effects, report = _effects("フェード\nイン=0.20\nアウト=1.00")
        assert _value(effects[0], "effect_time") == 1.0
        assert any("違う時間" in line for line in report.lines())

    def test_a_wipe_keeps_its_shape(self) -> None:
        effect = _one(
            "ワイプ\nイン=0.5\nアウト=0\nぼかし=2\nワイプの種類=ワイプ(円)\n反転(イン)=1\n反転(アウト)=0"
        )
        assert effect.kind == "inout_wipe"
        assert effect.params["pattern"] == "circle"
        assert effect.params["reverse_in"] is True
        assert _value(effect, "tolerance") == 2.0

    def test_an_unknown_wipe_shape_falls_back_to_fade(self) -> None:
        # 付属の 5 枚以外の絵は式で作れない 黙って別の形にすると見た目が変わる
        effects, report = _effects("ワイプ\nイン=0.5\nアウト=0\nワイプの種類=ワイプ(渦巻)")
        assert effects[0].params["pattern"] == "fade"
        assert any("ワイプの種類" in line for line in report.lines())

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [(0.0, "right"), (90.0, "bottom"), (180.0, "left"), (270.0, "top")],
    )
    def test_the_angle_chooses_the_direction(self, angle: float, expected: str) -> None:
        effect = _one(f"画面外から登場\n時間=0.5\n角度={angle}\n数=1\nランダム方向=0")
        assert effect.kind == "inout_move"
        assert effect.params["direction"] == expected

    def test_an_angle_between_directions_is_recorded(self) -> None:
        effects, report = _effects("画面外から登場\n時間=0.5\n角度=45\n数=1\nランダム方向=0")
        assert effects[0].params["direction"] in ("right", "bottom")
        assert any("角度" in line for line in report.lines())

    def test_spreading_flattens_one_axis(self) -> None:
        effect = _one("広がって登場\n時間=0.3\n縦方向=0")
        assert effect.kind == "inout_zoom"
        # 隠れたときは横が 0（縦方向の旗が立つと軸が入れ替わる）
        assert _value(effect, "zoom_x") == 0.0
        assert _value(effect, "zoom_y") == 100.0

    def test_spreading_vertically_swaps_the_axes(self) -> None:
        effect = _one("広がって登場\n時間=0.3\n縦方向=1")
        assert _value(effect, "zoom_x") == 100.0
        assert _value(effect, "zoom_y") == 0.0

    def test_the_bounce_count_becomes_one_period(self) -> None:
        effect = _one("弾んで登場\n時間=2.0\n高さ=200\n回数=4")
        assert effect.kind == "inout_jump"
        assert _value(effect, "height") == 200.0
        assert _value(effect, "period") == 0.5

    def test_the_easing_flag_smooths_both_ends(self) -> None:
        effect = _one("拡大縮小して登場\n時間=0.3\n拡大率=300\n加減速=1")
        assert effect.kind == "inout_zoom"
        assert _value(effect, "zoom") == 300.0
        assert effect.params["easing"] == "sine"
        assert effect.params["easing_mode"] == "inout"

    def test_settings_that_cannot_be_copied_are_recorded(self) -> None:
        _, report = _effects("起き上がって登場\n時間=0.6\n勢い=2.0")
        assert any("勢い" in line for line in report.lines())


class TestShapesAndSwings:
    def test_the_fan_clip_uses_the_fan_shape(self) -> None:
        effect = _one("扇クリッピング\n中心X=0\n中心Y=0\n基準角=0\n範囲角=90\nぼかし=0")
        assert effect.kind == "shape_mask"
        assert effect.params["shape"] == "fan"
        assert _value(effect, "span") == 90.0

    def test_a_pendulum_swings_around_its_start(self) -> None:
        # 片側にだけ振れると、振り子ではなく回転になる
        effect = _one("振り子\n速さ=2.0\n角度=30\nずらし=0")
        assert effect.kind == "repeat_rotate"
        assert effect.params["centering"] is True
        assert _value(effect, "angle_z") == 30.0


class TestTheThingsReviewFound:
    def test_an_angle_near_zero_goes_right(self) -> None:
        # 角度は 1 周でつながっている まっすぐ引き算すると 359 度が上になる
        effect = _one("画面外から登場\n時間=0.5\n角度=359\n数=1\nランダム方向=0")
        assert effect.params["direction"] == "right"

    def test_a_negative_angle_also_goes_right(self) -> None:
        # 折り返しを見落とすと、-2 度が上（270 度に近い）として扱われる
        effect = _one("画面外から登場\n時間=0.5\n角度=-2\n数=1\nランダム方向=0")
        assert effect.params["direction"] == "right"

    def test_a_plain_number_reaches_a_value_spec(self) -> None:
        # スライダーを持たない数値を文字列のまま渡すと、既定値（8 個）へ落ちる
        effect = _one("円形配置\n円周=100\n半径=200\n数=3")
        assert effect.kind == "circular_duplicate"
        assert effect.params["count"] == 3

    def test_an_unreadable_value_keeps_the_target_default(self) -> None:
        # 写し先の既定値に変換を掛けると、不透明度 100 が 0 になって全透明になる
        effects, report = _effects("透明度\n透明度=おかしな値")
        assert _value(effects[0], "amount") == 100.0
        assert any("数として読めない値" in line for line in report.lines())

    def test_an_appearance_value_can_move(self) -> None:
        # 登場の値に動きが付いていたら、キーフレームとして残す
        effect = _one("拡大縮小して登場\n時間=0.3\n拡大率=100,300,直線移動,0\n加減速=0")
        value = effect.params["zoom"]
        assert isinstance(value, AnimatedValue)
        assert value.is_animated

    def test_a_dropped_setting_is_recorded(self) -> None:
        # 効果を写せても、項目を落とせば見た目は変わる（震えるの角度）
        _, report = _effects("震える\n振幅=8\n角度=45\n間隔=2")
        assert any("震えるの項目: 角度" in line for line in report.lines())

    def test_structural_keys_are_not_recorded(self) -> None:
        # Group は並びの区切りで、値ではない 記録に出すと本当の穴が埋もれる
        _, report = _effects("座標\nX=10\nY=0\nZ=0\nGroup=1")
        assert not any("Group" in line for line in report.lines())

    def test_an_empty_value_is_recorded(self) -> None:
        # 記録しないと、指定したつもりの値が既定値へ置き換わったことに気付けない
        _, report = _effects("透明度\n透明度=")
        assert any("数として読めない値" in line for line in report.lines())

    def test_an_unknown_choice_is_recorded(self) -> None:
        # 表に無い選択肢は既定値のままになる 写せたことにすると形が変わったまま残る
        _, report = _effects("マスク\n種類=三角\n中心X=0\n中心Y=0")
        assert any("マスクの項目: 種類" in line for line in report.lines())

    def test_a_still_expression_on_a_plain_number_is_recorded(self) -> None:
        # 値が同じでも、式なら時間で変わりうる
        _, report = _effects("円形配置\n円周=100\n半径=200\n数=3,3,瞬間移動,8|3+time")
        assert any("動く値を写せない項目" in line for line in report.lines())


class TestTheLeftoverSettings:
    """取りこぼしていた項目 記録の数え方を直したら、本当の穴だけが残った"""

    def test_the_gradient_keeps_its_blend(self) -> None:
        # 配布物 36 本で 8 回使われていた 通常のままだと見た目が別物になる
        effect = _one("グラデーション\n強さ=100\n合成モード=加算\n形状=線形\n開始色=ffffff")
        assert effect.kind == "gradient"
        assert effect.params["blend"] == "add"

    def test_a_single_colour_fill_keeps_its_strength(self) -> None:
        # 単色化は色と強さを持つ 強さを落とすと必ず真っ白（指定色）になる
        effect = _one("単色化\n強さ=40\n色=ff0000\n輝度を保持する=0")
        assert effect.kind == "fill"
        assert _value(effect, "amount") == 40.0
        assert effect.params["color"] == (1.0, 0.0, 0.0, 1.0)

    def test_a_single_colour_fill_can_keep_the_brightness(self) -> None:
        """単色化 の 輝度を保持する を写す（#170）

        落とすと、明るさを残して色だけを付けるつもりの絵が一色の板になる
        """
        effect = _one("単色化\n強さ=40\n色=ff0000\n輝度を保持する=1")
        assert effect.params["keep_luma"] is True

    def test_the_slant_clip_cuts_off_one_side(self) -> None:
        """斜めクリッピング は線の片側を落とす crop_slant へ写す（#170）

        以前は帯だけを残す crop_angle へ写していて、幅 0 では線 1 本しか残らず絵が消えた
        中心Y は AviUtl の下が正からこちらの上が正へ直す
        """
        effect = _one("斜めクリッピング\n中心X=5\n中心Y=10\n角度=30\nぼかし=2\n幅=0")
        assert effect.kind == "crop_slant"
        assert _value(effect, "center_x") == 5.0
        assert _value(effect, "center_y") == -10.0
        assert _value(effect, "angle") == 30.0
        assert _value(effect, "blur") == 2.0

    def test_the_lens_blur_light_is_not_a_brightness(self) -> None:
        """レンズブラー の 光の強さ を明るさの倍率へ入れない（#170）

        写し先は 100% が元のままの倍率で、光の強さ 0（既定）を入れると真っ黒になる
        写せない値は記録に残す
        """
        effect, report = _effects("レンズブラー\n範囲=16\n光の強さ=32\nサイズ固定=0")
        assert effect[0].kind == "lens_blur"
        assert _value(effect[0], "brightness") == 100.0
        assert any("レンズブラーの項目: 光の強さ" in line for line in report.lines())

    def test_the_flip_filter_reads_both_axes(self) -> None:
        # 旗を取り違えると、上下だけ反転させたつもりが左右にひっくり返る
        effect = _one("反転\n上下反転=1\n左右反転=0\n輝度反転=0\n色相反転=0\n透明度反転=0")
        assert effect.kind == "flip"
        assert effect.params["vertical"] is True
        assert effect.params["horizontal"] is False

    def test_a_colour_inversion_is_recorded(self) -> None:
        # 上下左右の反転しか写せない 輝度や色相の反転は別の効果
        _, report = _effects("反転\n上下反転=0\n左右反転=0\n輝度反転=1\n色相反転=0\n透明度反転=0")
        assert any("反転の項目: 輝度反転" in line for line in report.lines())

    def test_the_mirror_filter_is_not_a_flip(self) -> None:
        # AviUtl の ミラー は鏡像を映す効果（透明度・減衰・境目調整・向きを持つ）
        # 反転として写すと、上下がひっくり返った別の絵になる
        effect = _one("ミラー\n透明度=0\n減衰=50\n境目調整=0\nミラーの方向=下側")
        assert effect.kind == "mirror"
        assert effect.params["side"] == "bottom"
        assert _value(effect, "falloff") == 50.0

    def test_the_mirror_side_comes_from_the_name(self) -> None:
        # AviUtl2 は向きを名前で書く（``ミラーの方向=右側``）
        # 読み落とすと、右へ映すはずの鏡像が下に出る
        effect = _one("ミラー\n透明度=0\n減衰=0\n境目調整=0\nミラーの方向=右側")
        assert effect.params["side"] == "right"

    def test_the_luminance_key_mode(self) -> None:
        # 逆に読むと、抜ける所と残る所が入れ替わって絵が反転して見える
        effect = _one("ルミナンスキー\n基準輝度=2048\n輝度範囲=512\nモード=明るい部分を透過")
        assert effect.kind == "luminance_key"
        assert effect.params["invert"] is True

    def test_settings_left_at_zero_are_not_recorded(self) -> None:
        # 0 や空は「使っていない」 記録に出すと、本当に埋めるべき穴が埋もれる
        # （縁取りのぼかしが 26 本とも 0 なのに、一番多い穴として並んでいた）
        _, report = _effects("縁取り\nサイズ=6\nぼかし=0\n縁色=ffffff\nパターン画像=")
        assert not any("縁取りの項目" in line for line in report.lines())

    def test_a_setting_in_use_is_still_recorded(self) -> None:
        # 使っている設定まで数えるのをやめると、落とした所が記録から消えて
        # 「写せたつもりで違う絵」に気付けなくなる
        # 縁取りのぼかしは #192 で写すようになったので、まだ写さないレンズブラーの光の強さで見る
        _, report = _effects("レンズブラー\n範囲=16\n光の強さ=32")
        assert any("レンズブラーの項目: 光の強さ" in line for line in report.lines())

    def test_a_zero_that_moves_is_still_in_use(self) -> None:
        # 0 から動く値を「使っていない」と数えると、動きを落としたことが記録から消える
        # 逆に ``0,0,直線移動,0`` のような動かない値まで数えると、記録が埋まって
        # 本当に埋めるべき穴が見えなくなる
        _, report = _effects("縁取り\nサイズ=6\nぼかし=0,10,直線移動,0\n縁色=ffffff")
        assert any("縁取りの項目: ぼかし" in line for line in report.lines())

    def test_a_zero_with_an_expression_is_still_in_use(self) -> None:
        # 参照式は 0 から始まっても時間で変わる ここで未使用と見ると、
        # 写せていない ぼかし が記録から消えて、見た目の違いに気付けない
        _, report = _effects("縁取り\nサイズ=6\nぼかし=0,0,瞬間移動,8|time\n縁色=ffffff")
        assert any("縁取りの項目: ぼかし" in line for line in report.lines())

    def test_a_named_move_that_does_not_move_is_off(self) -> None:
        # 移動方法の名前が付いていても、値が動かなければ見た目は変わらない
        # ここを使用中と数えると、記録が埋まって多い順の並びが役に立たなくなる
        _, report = _effects("縁取り\nサイズ=6\nぼかし=0,0,直線移動,0\n縁色=ffffff")
        assert not any("縁取りの項目: ぼかし" in line for line in report.lines())

    def test_an_easing_flag_alone_is_off(self) -> None:
        # 加速や減速の旗が付いていても、値が動かなければ見た目は変わらない
        # ここを使用中と数えると、直したばかりの多い順の並びがまた埋まる
        _, report = _effects("縁取り\nサイズ=6\nぼかし=0,0,補間移動,3\n縁色=ffffff")
        assert not any("縁取りの項目: ぼかし" in line for line in report.lines())


class TestWhatTheRealOutputFound:
    """AviUtl2 に書き出させた動画と 1 枚ずつ比べて見つかったもの"""

    def test_black_stays_black(self) -> None:
        """``000000`` が白へ化けない

        頭の飾りを ``lstrip`` で落としていたので、``000000`` が空文字になり
        「読めない色」＝白へ落ちていた 黒は縁取りと影の既定色なので、
        配布物のほとんどが白い縁に覆われて別物になる
        """
        effect = _one("縁取り\nサイズ=6\n縁色=000000")
        assert effect.params["color"] == (0.0, 0.0, 0.0, 1.0)

    def test_a_black_gradient_end_stays_black(self) -> None:
        # 終了色が白へ化けると、暗く落ちていくはずのグラデーションが
        # 明るい方へ伸びる 文字の下半分が白飛びして読めなくなる
        # テキストのすぐ後ろのグラデーションは塗りへ写る（#312） 塗りの最後の止まりで見る
        report = CompatibilityReport()
        body = _object(
            "グラデーション\n強さ=100.0\n角度=90.00\n幅=100\n形状=線形\n"
            "開始色=6c6c6c\n終了色=000000"
        )
        item = map_object(parse_exo(body).objects[0], RATE, report=report)
        assert item is not None and item.clip.source is not None
        stops = item.clip.source.params["fill_gradient"]
        assert isinstance(stops, tuple)
        assert stops[-5:] == (1.0, 0.0, 0.0, 0.0, 1.0)

    def test_a_hash_or_an_0x_prefix_is_still_dropped(self) -> None:
        # 飾りを 1 つずつ落とす作りにしたので、付いていても読めること
        # 落とし損ねると色として読めず、白（読めない色の逃げ先）になる
        # 緑の縁取りが白い縁になって、文字の周りだけ配色が変わる
        assert _one("縁取り\nサイズ=6\n縁色=#00ff00").params["color"] == (0.0, 1.0, 0.0, 1.0)
        assert _one("縁取り\nサイズ=6\n縁色=0x00ff00").params["color"] == (0.0, 1.0, 0.0, 1.0)


class TestSelectsWrittenAsNumbers:
    """AviUtl1 世代は選択肢を名前ではなく番号で書く

    番号がどの選択肢かは実物で確かめていない 既定値のまま黙って進むと、
    向きや形の違う絵が出たことに気付けないので、必ず記録へ残す
    """

    def test_a_numeric_choice_is_recorded(self) -> None:
        _, report = _effects("ミラー\n透明度=0\n減衰=0\n境目調整=0\nミラーの方向=0")
        assert any("ミラーの項目: ミラーの方向" in line for line in report.lines())

    def test_a_named_choice_is_not_recorded(self) -> None:
        # 名前で書いてあれば読めているので、記録へ出してはいけない
        # 出すと、本当に埋めるべき穴の並びがこれで埋まる
        _, report = _effects("ミラー\n透明度=0\n減衰=0\n境目調整=0\nミラーの方向=右側")
        assert not any("ミラーの方向" in line for line in report.lines())
