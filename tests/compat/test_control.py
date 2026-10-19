"""制御文字からパラメータ定義へ

ここが通れば、AviUtl のスクリプトも自前のエフェクトとまったく同じ設定 UI に
載る P2 でパラメータ定義を AviUtl 互換の 1 形式に統一した狙いがこれ
"""

from __future__ import annotations

from sashimono.compat.aviutl.control import parse_control, split_scripts
from sashimono.effects.spec import (
    CheckSpec,
    ColorSpec,
    FileSpec,
    FontSpec,
    SelectSpec,
    TextSpec,
    TrackSpec,
    ValueSpec,
)


class TestTrack:
    def test_a_slider_becomes_a_track_spec(self) -> None:
        # --track0 の範囲と既定値を読み違えると、置いたばかりのスクリプトの欄が
        # 作者の決めた -100〜100 と 20 から外れて始まる
        header = parse_control("--track0:移動量,-100,100,20,0.1")
        spec = header.parameters[0]
        assert isinstance(spec, TrackSpec)
        assert (spec.name, spec.label) == ("track0", "移動量")
        assert (spec.minimum, spec.maximum, spec.default) == (-100.0, 100.0, 20.0)

    def test_all_four_slots(self) -> None:
        # 4 本の番号を取りこぼすと、track1〜3 の欄が track0 に重なり、
        # obj.track1〜3 が 0 のまま動かない
        header = parse_control(
            "--track0:A,0,1,0\n--track1:B,0,1,0\n--track2:C,0,1,0\n--track3:D,0,1,0"
        )
        assert [spec.name for spec in header.parameters] == [
            "track0",
            "track1",
            "track2",
            "track3",
        ]

    def test_a_reversed_range_is_straightened(self) -> None:
        # 最小と最大を入れ替えずに渡すと TrackSpec が範囲の誤りで断り、その節ごと
        # 一覧から消えて互換性レポートに読めない記録が残る
        spec = parse_control("--track0:逆,100,-100,0").parameters[0]
        assert isinstance(spec, TrackSpec)
        assert (spec.minimum, spec.maximum) == (-100.0, 100.0)

    def test_a_default_outside_the_range_is_clamped(self) -> None:
        # 範囲の外の既定値をそのまま渡すと TrackSpec が断り、その節ごと一覧から消える
        spec = parse_control("--track0:はみ出し,0,10,999").parameters[0]
        assert isinstance(spec, TrackSpec)
        assert spec.default == 10.0

    def test_the_named_form_from_the_newer_generation(self) -> None:
        # --track@speed の名前を読み落とすと、本文が素の speed を nil で読んで計算で落ちる
        spec = parse_control("--track@speed:速度,0,10,1").parameters[0]
        assert spec.name == "speed"
        assert spec.label == "速度"


class TestCheckAndColor:
    def test_a_checkbox(self) -> None:
        # --check0 の既定の 1 を落とすと、obj.check0 で切り替えるスクリプトがオフの側で始まる
        spec = parse_control("--check0:反転,1").parameters[0]
        assert isinstance(spec, CheckSpec)
        assert (spec.name, spec.label, spec.default) == ("check0", "反転", True)

    def test_a_colour(self) -> None:
        # 0xff8000 を読み違えると、置いたばかりのスクリプトが作者の決めた橙ではない色で描く
        spec = parse_control("--color:線の色,0xff8000").parameters[0]
        assert isinstance(spec, ColorSpec)
        assert spec.default[0] == 1.0
        assert abs(spec.default[1] - 128 / 255) < 0.01
        assert spec.default[2] == 0.0

    def test_a_file_and_a_font(self) -> None:
        # --file と --font が専用の欄にならないと、画像のパスや書体を選ぶ窓が出ず、
        # スクリプトへ名前を渡せない
        header = parse_control("--file:読み込むもの\n--font:書体")
        assert isinstance(header.parameters[0], FileSpec)
        assert isinstance(header.parameters[1], FontSpec)


class TestDialog:
    def test_plain_numbers_become_sliders(self) -> None:
        # AviUtl のダイアログは入力欄だが、こちらではキーフレームを打てる方が
        # 使い出がある
        spec = parse_control("--dialog:回数,count=3;").parameters[0]
        assert isinstance(spec, TrackSpec)
        assert (spec.name, spec.default) == ("count", 3.0)

    def test_a_quoted_default_becomes_text(self) -> None:
        # 引用符の初期値を数として読むと、本文の欄がスライダーになり、文字ではなく 0 が渡る
        spec = parse_control('--dialog:本文,body="こんにちは";').parameters[0]
        assert isinstance(spec, TextSpec)
        assert spec.default == "こんにちは"

    def test_the_chk_suffix_makes_a_checkbox(self) -> None:
        # /chk を読み落とすと、オンとオフだけの欄が -100〜100 のスライダーとして設定に出る
        spec = parse_control("--dialog:影を付ける/chk,shadow=1;").parameters[0]
        assert isinstance(spec, CheckSpec)
        assert (spec.label, spec.default) == ("影を付ける", True)

    def test_the_col_suffix_makes_a_colour(self) -> None:
        # /col を読み落とすと、色の欄が数のスライダーになり、0x00ff00 を数と読めずに
        # 0（黒）がスクリプトへ渡る
        spec = parse_control("--dialog:色/col,col=0x00ff00;").parameters[0]
        assert isinstance(spec, ColorSpec)
        assert spec.default[:3] == (0.0, 1.0, 0.0)

    def test_the_fig_suffix_lists_the_aviutl_figures(self) -> None:
        # /fig を読み落とすと、図形を一覧から選べず、文字の欄に図形の名前を手で打つことになる
        spec = parse_control('--dialog:図形/fig,fig="星形";').parameters[0]
        assert isinstance(spec, SelectSpec)
        assert spec.default == "星形"
        assert "六角形" in dict(spec.choices)

    def test_several_items_in_one_line(self) -> None:
        # ; で区切った 2 つ目から先を取りこぼすと、y と c の欄が出ず、本文がその変数を
        # nil で読む
        header = parse_control("--dialog:X,x=10;Y,y=20;色/col,c=0xffffff;")
        assert [spec.name for spec in header.parameters] == ["x", "y", "c"]

    def test_items_without_an_assignment_are_skipped(self) -> None:
        # 変数の無い項目まで欄にすると、スクリプトが読まない欄が設定に並ぶ
        assert parse_control("--dialog:ラベルだけ;").parameters == ()


class TestOtherControls:
    def test_param_is_kept_as_setup_code(self) -> None:
        # --param を拾えないと、本文より先に走るはずの a=1;b=2 が走らず、本文がその変数を
        # nil で読む
        header = parse_control("--param:a=1;b=2;")
        assert header.setup == "a=1;b=2;"

    def test_labels_are_collected(self) -> None:
        # 見出しを集めないと ScriptHeader.labels を読む側に --label の文字が渡らない
        # （いまは読む所が無い）
        assert parse_control("--label:見出し").labels == ("見出し",)

    def test_the_newer_select_control(self) -> None:
        # --select@ を読み落とすと、直線と曲線を選ぶ欄が出ず、本文の mode に既定の curve が
        # 渡らない
        spec = parse_control("--select@mode:種類,直線=line/曲線=curve,curve").parameters[0]
        assert isinstance(spec, SelectSpec)
        assert dict(spec.choices) == {"line": "直線", "curve": "曲線"}
        assert spec.default == "curve"

    def test_value_stays_a_plain_number(self) -> None:
        # --value@ がスライダーになると、シードのように途中の値に意味の無い欄に
        # キーフレームが打て、時間で値が流れる
        spec = parse_control("--value@seed:シード,42").parameters[0]
        assert isinstance(spec, ValueSpec)
        assert spec.default == 42

    def test_unsupported_controls_are_recorded(self) -> None:
        # 記録が残っていれば、次に何を実装すべきかをデータで決められる
        header = parse_control("--twopoint\n--speed:10,20")
        assert len(header.unknown) == 2

    def test_ordinary_comments_are_not_recorded(self) -> None:
        # ふつうのコメントを未対応に数えると、互換性レポートの未対応の制御文字が注釈の数だけ
        # 増え、埋める順を数で決められない
        header = parse_control("-- ただのコメント\n--[[ ブロック ]]")
        assert header.unknown == ()

    def test_lua_code_is_ignored(self) -> None:
        # 本文の Lua の行から欄を作ると、スクリプトが読まない欄が設定に並ぶ
        # 制御行に似た綴りを、行末のコメントと文字列の中に置く -- で始まらない行だけでは、
        # 行の頭を見ずに探す壊れ方でも何も見つからず通ってしまう 本物の制御行を 1 本混ぜ、
        # それだけが欄になることを見る
        header = parse_control(
            "local x = 1\n"
            "obj.ox = x --track0:速さ,0,100,10\n"
            "local s = '--check0:反転,0'\n"
            "--track1:大きさ,0,100,50"
        )
        assert [spec.name for spec in header.parameters] == ["track1"]


class TestSplitScripts:
    def test_a_file_without_markers_is_one_script(self) -> None:
        # @ の無いファイルを 1 本と数えないと、そのスクリプトが一覧に出ず置けない
        sections = split_scripts("--track0:A,0,1,0\nobj.ox = obj.track0")
        assert len(sections) == 1
        assert sections[0].header.parameters[0].name == "track0"

    def test_markers_split_the_file(self) -> None:
        # @名前 で分けないと、1 ファイルに入った ふわふわ と ぐるぐる が別々の効果として
        # 一覧に並ばない
        text = (
            "@ふわふわ\n--track0:幅,0,100,10\nobj.oy = 1\n"
            "@ぐるぐる\n--track0:速さ,0,10,1\nobj.rz = 2"
        )
        sections = split_scripts(text)
        assert [section.header.name for section in sections] == ["ふわふわ", "ぐるぐる"]
        assert sections[0].header.parameters[0].label == "幅"
        assert sections[1].header.parameters[0].label == "速さ"

    def test_each_section_keeps_only_its_own_code(self) -> None:
        # 節の本文が隣の節に混ざると、1 つの効果を置いただけで別の効果の動きまで走る
        sections = split_scripts("@あ\nobj.ox = 1\n@い\nobj.oy = 2")
        assert "obj.oy" not in sections[0].source
        assert "obj.ox" not in sections[1].source

    def test_the_text_before_the_first_marker_is_not_a_script(self) -> None:
        """最初の ``@名前`` より前（使用許諾の注釈）はどのスクリプトにも属さない（#170）

        sigma の配布物は全部、頭に MIT の注釈を置いてから ``@矩形`` を始める
        ここを 1 本として数えると、右クリックの〔追加〕に ``@単純図形σ`` という
        名前の、置いても何も描かない項目が並ぶ
        """
        text = "--[[\nThe MIT License (MIT)\n]]\n\n@矩形\nobj.ox = 1\n@楕円\nobj.oy = 2"
        sections = split_scripts(text)
        assert [section.header.name for section in sections] == ["矩形", "楕円"]
        assert all("MIT" not in section.source for section in sections)
