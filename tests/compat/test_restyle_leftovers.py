"""着せ替えで、テンプレートが持たない項目が前の値のまま残らないこと（#283）

AviUtl と YMM4 のテンプレートには、フォントのスタイルや折り返しの幅という項目が無い
今の値の上にテンプレートを重ねるだけだと、それらが前のまま残り、棚の見本と違う
見た目になる（Segoe UI の Semibold に Noto Sans JP を着せると、Noto Sans JP の
Semibold を探して描いた）
"""

from __future__ import annotations

from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import restyle
from sashimono.compat.mapped import MappedObject
from sashimono.core.commands import SetSource
from sashimono.core.model import AnimatedValue, Clip, GeneratedSource
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT

#: 手元の字幕テンプレート（01_Premiere風_標準字幕.object）の骨組み 縁取りも影も
#: スタイルも折り返しの幅も持たない
TEMPLATE = """[Object]
frame=0,179
[Object.0]
effect.name=テキスト
サイズ=60.00
字間=0.00
行間=8.00
表示速度=0.00
フォント=Noto Sans JP
文字色=ffffff
影・縁色=000000
文字装飾=標準文字
文字揃え=中央揃え[下]
B=1
I=0
テキスト=字幕テキスト
[Object.1]
effect.name=標準描画
X=0.00
Y=400.00
"""


def _template() -> list[MappedObject]:
    report = CompatibilityReport()
    mapped = [map_object(obj, FrameRate(30), report=report) for obj in parse_exo(TEMPLATE).objects]
    return [item for item in mapped if item is not None]


def _styled() -> Clip:
    """Segoe UI の Semibold に、折り返しの幅・縁取り・影・文字送りを付けたテキスト"""
    return Clip(
        timeline_start=0,
        duration=90,
        source=TEXT.create(
            text="自分で打った字幕",
            font="Segoe UI",
            font_style="Semibold",
            wrap_width=640,
            border_width=6,
            shadow_x=4,
            shadow_y=-4,
            reveal=40,
        ),
    )


def _params(clip: Clip, *, keep_wrap: bool = False) -> dict[str, object]:
    commands = restyle(_template(), clip, keep_wrap=keep_wrap)
    source = next(c for c in commands if isinstance(c, SetSource)).source
    assert source is not None
    return dict(source.params)


def _number(value: object) -> float:
    assert isinstance(value, AnimatedValue)
    return value.at(0)


class TestLeftovers:
    def test_the_old_family_style_is_dropped(self) -> None:
        # 別のファミリのスタイル名は当たらない 残すと標準の組み方では、新しいファミリの
        # 中で同じ名前を探し、見つかれば見本と違う太さで描く
        params = _params(_styled())
        assert params["font"] == "Noto Sans JP"
        assert params["font_style"] == ""

    def test_the_wrap_width_goes_back_to_the_default(self) -> None:
        # 既定は棚の見本と同じ見た目 見本は折り返さない
        assert _number(_params(_styled())["wrap_width"]) == 0.0

    def test_the_border_and_shadow_the_template_lacks_are_cleared(self) -> None:
        # 標準文字のテンプレートを着せたのに前の縁と影が残ると、見本と違う字になる
        params = _params(_styled())
        assert _number(params["border_width"]) == 0.0
        assert (_number(params["shadow_x"]), _number(params["shadow_y"])) == (0.0, 0.0)

    def test_the_text_and_the_reveal_are_kept(self) -> None:
        # 文字と文字送りは中身 見た目を変えたいだけで消えたら着せ替えではない
        params = _params(_styled())
        assert params["text"] == "自分で打った字幕"
        assert _number(params["reveal"]) == 40.0

    def test_the_template_values_are_taken(self) -> None:
        params = _params(_styled())
        assert _number(params["size"]) == 60.0
        assert params["layout"] == "aviutl"

    def test_the_same_template_gives_the_same_look_from_any_clip(self) -> None:
        # 着せ替えた結果が、前に何を着ていたかで変わると、棚の見本を信じられない
        plain = Clip(
            timeline_start=0,
            duration=90,
            source=GeneratedSource(kind="text", params={"text": "自分で打った字幕"}),
        )
        from_plain = _params(plain)
        from_styled = _params(_styled())
        from_styled["reveal"] = from_plain["reveal"]
        assert from_styled == from_plain


class TestKeepWrap:
    def test_the_width_is_kept_when_asked(self) -> None:
        # 字幕の枠の幅を先に決めてから着せ替える人向け（Preferences.restyle_keep_wrap）
        params = _params(_styled(), keep_wrap=True)
        assert _number(params["wrap_width"]) == 640.0

    def test_keeping_the_width_does_not_keep_the_style(self) -> None:
        # 残すのは幅だけ スタイルは別のファミリでは当たらないので、設定に関係なく空に戻す
        assert _params(_styled(), keep_wrap=True)["font_style"] == ""


class TestTimer:
    def test_a_timer_stays_a_timer(self) -> None:
        # タイマーの書式は何を出すかを決める中身 既定へ戻すと、字幕の見た目を着せただけで
        # 時間が消えて文字が出る
        timer = Clip(
            timeline_start=0,
            duration=90,
            source=TEXT.create(text="", timer_format="mm\\:ss", timer_countdown=True),
        )
        params = _params(timer)
        assert params["timer_format"] == "mm\\:ss"
        assert params["timer_countdown"] is True

    def test_an_empty_format_in_the_template_keeps_the_timer(self) -> None:
        # 画面で作ったテキストは空の書式を持つ 項目の有無で決めると、そういうテンプレートを
        # 着せただけでタイマーの時間が消える
        timer = Clip(
            timeline_start=0,
            duration=90,
            source=TEXT.create(text="", timer_format="mm\\:ss"),
        )
        plain = TEXT.create(text="字幕", font="Noto Sans JP")
        assert plain.params["timer_format"] == ""
        template = [MappedObject(clip=Clip(timeline_start=0, duration=30, source=plain), layer=1)]
        commands = restyle(template, timer)
        source = next(c for c in commands if isinstance(c, SetSource)).source
        assert source is not None
        assert source.params["timer_format"] == "mm\\:ss"
        assert source.params["font"] == "Noto Sans JP"
