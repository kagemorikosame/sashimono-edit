"""AviUtl のテキストで読んでいない 5 つの項目を、未対応として数える（#284）

表示速度・文字毎に個別オブジェクト・自動スクロール・移動座標上に表示・
オブジェクトの長さを自動調節は、写す先がまだ無い 0 以外を黙って落とすと、実物を
数えて多い順に埋めるときに数に上がらない

見本の書き方は手元の実物（``%PROGRAMDATA%\\aviutl2\\Alias`` の ``.object`` と、
PSDToolKit などの AviUtl1 の ``.exa``）から写した 実物そのものを通す試験は下の
``TestRealFiles`` で、置き場が無い機械では飛ぶ
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sashimono.compat.aviutl.exo import load_exo, parse_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.core.timebase import FrameRate

RATE = FrameRate(30)

#: 報告に出る名前 5 つ
UNREAD = (
    "テキストの表示速度",
    "テキストの文字毎に個別オブジェクト",
    "テキストの自動スクロール",
    "テキストの移動座標上に表示",
    "テキストのオブジェクトの長さを自動調節",
)


def _aviutl2(**values: str) -> str:
    """手元の字幕テンプレート（01_Premiere風_標準字幕.object）の骨組み"""
    plain = {
        "表示速度": "0.00",
        "文字毎に個別オブジェクト": "0",
        "自動スクロール": "0",
        "移動座標上に表示": "0",
        "オブジェクトの長さを自動調節": "0",
    }
    plain.update(values)
    lines = [
        "[Object]",
        "frame=0,179",
        "[Object.0]",
        "effect.name=テキスト",
        "サイズ=60.00",
        "フォント=Noto Sans JP",
        "文字装飾=標準文字",
        "テキスト=字幕テキスト",
        *(f"{key}={value}" for key, value in plain.items()),
        "[Object.1]",
        "effect.name=標準描画",
        "X=0.00",
        "Y=400.00",
    ]
    return "\n".join(lines) + "\n"


def _aviutl1(**values: str) -> str:
    """AviUtl1 の ``.exa`` の骨組み 項目名が AviUtl2 と少し違う"""
    plain = {
        "表示速度": "0.0",
        "文字毎に個別オブジェクト": "0",
        "移動座標上に表示する": "0",
        "自動スクロール": "0",
        "autoadjust": "0",
    }
    plain.update(values)
    lines = [
        "[vo]",
        "length=64",
        "[vo.0]",
        "_name=テキスト",
        "サイズ=34",
        "font=MS UI Gothic",
        "type=0",
        "text=53004b00",
        *(f"{key}={value}" for key, value in plain.items()),
        "[vo.1]",
        "_name=標準描画",
        "X=0.0",
    ]
    return "\n".join(lines) + "\n"


def _missing(text: str) -> CompatibilityReport:
    report = CompatibilityReport()
    for obj in parse_exo(text).objects:
        map_object(obj, RATE, report=report)
    return report


def _unread(report: CompatibilityReport) -> dict[str, int]:
    return {name: count for name, count in report.missing.items() if name in UNREAD}


class TestAviUtl2:
    def test_all_zero_records_nothing(self) -> None:
        # 手元の実物 36 本はどれも全部 0 0 で記録が出ると、報告が意味の無い行で埋まる
        assert _unread(_missing(_aviutl2())) == {}

    @pytest.mark.parametrize(
        ("key", "value", "name"),
        [
            ("表示速度", "10.00", "テキストの表示速度"),
            ("文字毎に個別オブジェクト", "1", "テキストの文字毎に個別オブジェクト"),
            ("自動スクロール", "1", "テキストの自動スクロール"),
            ("移動座標上に表示", "1", "テキストの移動座標上に表示"),
            ("オブジェクトの長さを自動調節", "1", "テキストのオブジェクトの長さを自動調節"),
        ],
    )
    def test_a_used_item_is_recorded_by_name(self, key: str, value: str, name: str) -> None:
        # 黙って落ちると、実物を数えても穴として上がらず、埋める順を決められない
        assert _unread(_missing(_aviutl2(**{key: value}))) == {name: 1}

    def test_a_speed_moving_from_zero_is_recorded(self) -> None:
        # 始めの値だけ見ると、0 から動かした表示速度を「使っていない」と取り違える
        report = _missing(_aviutl2(表示速度="0.00,20.00,直線移動,0"))
        assert _unread(report) == {"テキストの表示速度": 1}

    def test_a_value_that_is_not_a_number_is_recorded(self) -> None:
        # 数として読めない値を「使っていない」と捨てると、知らない書き方の実物が数に上がらない
        assert _unread(_missing(_aviutl2(表示速度="速い"))) == {"テキストの表示速度": 1}

    def test_a_still_zero_with_a_method_records_nothing(self) -> None:
        # 移動方法の名前が付いていても、値が動かなければ見た目は変わらない
        assert _unread(_missing(_aviutl2(表示速度="0.00,0.00,直線移動,0"))) == {}


class TestAviUtl1:
    def test_all_zero_records_nothing(self) -> None:
        assert _unread(_missing(_aviutl1())) == {}

    def test_the_first_generation_names_are_read(self) -> None:
        # AviUtl1 は「移動座標上に表示する」と ``autoadjust`` と書く AviUtl2 の名前だけを
        # 引くと、AviUtl1 の 2 つは 0 以外でも記録に出ない
        report = _missing(_aviutl1(移動座標上に表示する="1", autoadjust="1"))
        assert _unread(report) == {
            "テキストの移動座標上に表示": 1,
            "テキストのオブジェクトの長さを自動調節": 1,
        }

    def test_the_english_names_are_read(self) -> None:
        # 英語版の AviUtl1 は ``vDisplay`` ``1char1obj`` と書く 日本語へ寄せてから見る
        text = (
            "[vo.0]\n_name=Text\nSize=34\nvDisplay=5.0\n1char1obj=1\n"
            "Automatic scrolling=1\nShow on motion coordinate=1\nautoadjust=0\n"
            "font=Segoe UI\ntext=53004b00\n[vo.1]\n_name=Standard drawing\nX=0.0\n"
        )
        assert _unread(_missing(text)) == {
            "テキストの表示速度": 1,
            "テキストの文字毎に個別オブジェクト": 1,
            "テキストの自動スクロール": 1,
            "テキストの移動座標上に表示": 1,
        }


def _real_files() -> list[Path]:
    roots: list[Path] = []
    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        roots.append(Path(program_data) / "aviutl2" / "Alias")
    roots.append(Path(__file__).resolve().parent.parent / "fixtures" / "aviutl")
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            for suffix in ("*.object", "*.exa"):
                found.extend(root.rglob(suffix))
    return sorted(set(found))


REAL = _real_files()


@pytest.mark.skipif(not REAL, reason="このマシンに AviUtl の配布エイリアスが無い")
class TestRealFiles:
    def test_the_real_files_are_read_and_counted_per_text(self) -> None:
        # 実物の中身が 0 だとは決めつけない 0 以外の実物が来たら、記録に出るのが正しい
        # 見るのは、落ちずに読めることと、1 つのテキストで 1 項目が 1 回より多く数えられないこと
        # （2026-10-10 に数えた手元の 132 個は 5 つとも全部 0 だった 0 と 0 以外は上の見本で見る）
        report = CompatibilityReport()
        texts = 0
        for path in REAL:
            for obj in load_exo(path).objects:
                if obj.entries and obj.entries[0].name == "テキスト":
                    texts += 1
                map_object(obj, RATE, report=report)
        if texts == 0:
            pytest.skip("手元の実物にテキストが無い")
        assert all(count <= texts for count in _unread(report).values())
