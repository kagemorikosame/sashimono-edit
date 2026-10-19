"""``.exo`` / ``.exa`` の読み込み

文字コードの判別を厚めに見る 間違えると読めないのではなく、**文字化けしたまま
読めてしまう**ので、あとから原因を追いにくい
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sashimono.compat.aviutl.encoding import (
    decode_bytes,
    decode_utf16_hex,
    encode_utf16_hex,
)
from sashimono.compat.aviutl.exo import ExoParseError, load_exo, parse_exo

SAMPLE = """[exedit]
width=1920
height=1080
rate=30
scale=1
length=300
audio_rate=44100
audio_ch=2
[0]
start=1
end=60
layer=1
group=1
overlay=1
camera=0
[0.0]
_name=テキスト
サイズ=48
表示速度=0.0
color=ffffff
color2=000000
font=Yu Gothic UI
text=53004b0055005400
[0.1]
_name=標準描画
X=100.0
Y=-50.0
Z=0.0
拡大率=200.00
透明度=25.0
回転=45.00
blend=0
"""


class TestParsing:
    def test_settings_are_read(self) -> None:
        # [exedit] を読み落とすと、呼び手はどの解像度・フレームレートのファイルでも
        # 既定の 1920x1080・30fps を受け取る
        # 見本は既定と同じ値なので、読み落としても通ってしまう 既定と違う値で読ませる
        exo = parse_exo("[exedit]\nwidth=1280\nheight=720\nrate=30000\nscale=1001\n")
        assert (exo.width, exo.height) == (1280, 720)
        assert exo.frame_rate == (30000, 1001)

    def test_object_bounds(self) -> None:
        # 1 始まりのまま返すと、〔オブジェクトを読み込む…〕で置いたクリップが元より 1 フレーム遅れる
        obj = parse_exo(SAMPLE).objects[0]
        # ファイルには 1 始まりで書かれている（start=1 end=60） 読んだ側では
        # 0 始まりに揃えるので 0..59 終端を含むので長さは 60
        assert (obj.start, obj.end, obj.duration) == (0, 59, 60)
        assert obj.layer == 1

    def test_entries_keep_their_order(self) -> None:
        # 先頭を中身として写すので、並びが崩れると 標準描画 が中身に回り、
        # 読み込んだテキストが未知の中身として捨てられる
        obj = parse_exo(SAMPLE).objects[0]
        assert [entry.name for entry in obj.entries] == ["テキスト", "標準描画"]
        assert obj.content is not None
        assert obj.content.name == "テキスト"
        assert [entry.name for entry in obj.filters()] == ["標準描画"]

    def test_numbers_are_read_as_numbers(self) -> None:
        # 数として読めないと呼び手の既定値へ落ち、読み込んだ図形のサイズなどが黙って既定に戻る
        draw = parse_exo(SAMPLE).objects[0].find("標準描画")
        assert draw is not None
        assert draw.number("X") == 100.0
        assert draw.number("拡大率") == 200.0
        assert draw.integer("blend") == 0

    def test_a_missing_value_falls_back(self) -> None:
        # 書かれていない項目で呼び手の既定値を返さないと、サイズを省いた図形が
        # 既定の 100 ではない大きさで置かれる
        draw = parse_exo(SAMPLE).objects[0].find("標準描画")
        assert draw is not None
        assert draw.number("そんな項目は無い", 7.5) == 7.5

    def test_the_text_field_is_utf16_hex(self) -> None:
        # AviUtl1 のテキスト欄を復号できないと、読み込んだテキストのクリップに元の文字が入らない
        content = parse_exo(SAMPLE).objects[0].content
        assert content is not None
        assert content.text() == "SKUT"

    def test_comments_and_blank_lines_are_skipped(self) -> None:
        # 注釈や空行でつまずくと、注釈を書き足した .exo の全体設定が読めなくなる
        # 注釈に = を含めておく 含めないと、読み飛ばしを外しても値として入らず見分けられない
        exo = parse_exo(";コメント=1\n\n[exedit]\nwidth=640\n;height=1\n")
        assert exo.width == 640
        assert exo.settings == {"width": "640"}

    def test_an_empty_file_is_refused(self) -> None:
        # 空のファイルを断らないと、〔オブジェクトを読み込む…〕が読めない理由を出さず
        # 「読み込めるオブジェクトがありませんでした」とだけ出す
        with pytest.raises(ExoParseError):
            parse_exo("")


class TestAlias:
    def test_an_alias_has_one_object_and_no_screen_size(self) -> None:
        # 全体設定の無いエイリアスを見分けられないと、呼び手が部品 1 個分を作品と取り違える
        # 区間を読み違えると、棚から置いたクリップの長さが 30 フレームからずれる
        alias = parse_exo("[0]\nstart=1\nend=30\nlayer=1\n[0.0]\n_name=図形\nサイズ=100\n")
        assert alias.is_alias is True
        assert len(alias.objects) == 1
        assert alias.objects[0].duration == 30

    def test_a_full_file_is_not_an_alias(self) -> None:
        # 全体設定のある .exo をエイリアスと見なすと、呼び手が作品全体を部品 1 個分として扱う
        assert parse_exo(SAMPLE).is_alias is False


class TestEncoding:
    def test_utf8_is_preferred(self) -> None:
        # UTF-8 を先に試さないと、AviUtl2 の .exo やスクリプトを cp932 として読み、
        # 名前が化けたまま通る
        text, encoding = decode_bytes("テキスト".encode())
        assert (text, encoding) == ("テキスト", "utf-8")

    def test_a_bom_is_stripped(self) -> None:
        # BOM を本文に残すと、BOM 付きの .exo で 1 行目の節の見出しを読み落とし、
        # その節の値が全体設定へ紛れる
        text, encoding = decode_bytes(b"\xef\xbb\xbf" + "あ".encode())
        assert text == "あ"
        assert encoding == "utf-8-sig"

    def test_shift_jis_falls_back(self) -> None:
        # AviUtl1 世代のファイル UTF-8 としては読めないバイト列
        text, encoding = decode_bytes("テキスト".encode("cp932"))
        assert (text, encoding) == ("テキスト", "cp932")

    def test_broken_bytes_still_return_something(self) -> None:
        # 読めないバイトで例外を出すと、load_exo は ExoParseError に包まないので、
        # 〔オブジェクトを読み込む…〕が警告の窓を出せない
        text, _ = decode_bytes(b"\x81\xff\xfe")
        assert isinstance(text, str)

    def test_the_text_field_round_trips(self) -> None:
        # 書く形と読む形が食い違うと、呼び手が書いたテキスト欄を読み戻したときに文字が変わる
        encoded = encode_utf16_hex("こんにちは", length=16)
        assert decode_utf16_hex(encoded) == "こんにちは"

    def test_padding_after_the_terminator_is_dropped(self) -> None:
        # 終端の後ろの 0 埋めを本文に含めると、読み込んだテキストの末尾に見えない NUL 文字が並ぶ
        assert decode_utf16_hex("42003100" + "0000" * 10) == "B1"

    def test_a_broken_hex_string_gives_an_empty_text(self) -> None:
        # 崩れた 16 進で例外を出すと、テキスト欄が 1 つ壊れただけで .exo の読み込み全体が止まる
        # 16 進以外を除く処理と 4 桁へ切り詰める処理のどちらかが欠けると、fromhex の
        # 例外が漏れる 混じり物と半端な桁の後ろに読める文字を残し、本文が残ることを見る
        assert decode_utf16_hex("zzz") == ""
        assert decode_utf16_hex("4200 zz 3100") == "B1"
        assert decode_utf16_hex("420031003") == "B1"

    @settings(max_examples=300, deadline=None)
    @given(st.text())
    def test_any_text_field_decodes_without_an_error(self, value: str) -> None:
        # 復号は念のための except を持たない（#384） 手前の 2 つの処理が fromhex へ
        # 渡せる形に必ず整える約束で、破れると壊れた欄 1 つで .exo の読み込み全体が止まる
        # 全角の数字や 16 進に似た字を含む任意の文字列で、例外にならず文字を返すことを見る
        assert isinstance(decode_utf16_hex(value), str)


class TestLoading:
    def test_a_shift_jis_file_is_read(self, tmp_path: Path) -> None:
        # cp932 の .exo を読めないと、AviUtl1 で作ったファイルの中身の名前が化け、
        # テキストも図形も未知の中身として捨てられる
        target = tmp_path / "旧世代.exo"
        target.write_bytes(SAMPLE.encode("cp932"))
        exo = load_exo(target)
        assert exo.encoding == "cp932"
        content = exo.objects[0].content
        assert content is not None
        assert content.name == "テキスト"

    def test_a_utf8_file_is_read(self, tmp_path: Path) -> None:
        # UTF-8 の .exo2 を cp932 と取り違えると、AviUtl2 のファイルの日本語が化けたまま読まれる
        target = tmp_path / "新世代.exo2"
        target.write_text(SAMPLE, encoding="utf-8")
        assert load_exo(target).encoding == "utf-8"

    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        # 開けないファイルを ExoParseError に包まないと、〔オブジェクトを読み込む…〕が
        # 「読み込めない」の窓で理由を見せられない
        with pytest.raises(ExoParseError, match="開けない"):
            load_exo(tmp_path / "無い.exo")
