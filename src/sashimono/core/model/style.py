"""編集の方針（スタイルガイド #343）

作品ごとに「この作品はこの書体・この色・この効果だけ」「テンポはこう」と本人が決めた物
AI アシスタントは読んで従う（``get_style_guide``） 方針に照らした食い違いは
``check_style_guide`` が調べる

書体・色・効果は形のある値で持つ 照らして調べられるのはこの 3 つだけで、文で持つと
「どの書体が方針の外か」を機械で決められない 字幕の見た目・テンポ・そのほかは人によって
書き方が違い、項目を決めると書けない事が出るので文のまま持つ（AI が読んで汲む）

本人が AI に渡す決まりなので、AI の道具からは変えない（変えられると AI が自分で緩められる）
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

__all__ = [
    "STYLE_EFFECTS_LIMIT",
    "STYLE_ITEMS_LIMIT",
    "STYLE_TEXT_LIMIT",
    "StyleGuide",
    "valid_style_color",
]

#: 文の欄 1 つの長さの上限（文字） 方針は AI へ毎回送る 長すぎると読むたびに費用が膨らみ、
#: 要点が埋もれる 説明書を丸ごと貼るような使い方は断る
STYLE_TEXT_LIMIT = 2000

#: 書体・色の数の上限 方針は絞るための物 これより多いのは一覧を丸ごと入れた取り違えで、
#: 照らしても何も外れない
STYLE_ITEMS_LIMIT = 100

#: 効果の数の上限 書体と色より大きくする 効果は窓の一覧（今の環境のエフェクト 自前だけで
#: 105 種 読み込んだ AviUtl のスクリプトも並ぶ）から印を付けて選ぶ 書体と同じ 100 にすると、
#: 「1 つだけ使わない」と決めた人が 104 個に印を付けられず、窓の OK が押せない
#: 上限は AI へ送る方針の長さを抑えるために残す 一覧の全部より多くは選べないので、ふつうは
#: 当たらない（AI の道具は効果の一覧を切って返す）
STYLE_EFFECTS_LIMIT = 300

_COLOR = re.compile(r"#[0-9a-f]{6}")


def valid_style_color(color: str) -> bool:
    """色が ``#rrggbb``（小文字）か 書き方を 1 つにしておくと、照らすときに書き方の違いで
    同じ色を別の色と数えない（大文字は :meth:`StyleGuide.cleaned` が小文字へ直す）"""
    return _COLOR.fullmatch(color) is not None


@dataclass(frozen=True, slots=True)
class StyleGuide:
    """作品の編集の方針 どの欄も空なら「決めていない」（AI は好きに選んでよい）"""

    #: 使う書体の名前 空なら決めない テキストの書体がこの外なら照らしで返す
    fonts: tuple[str, ...] = ()
    #: 使う色 ``#rrggbb`` 空なら決めない 字の色・図形の色・縁取りの色がこの外なら照らしで返す
    colors: tuple[str, ...] = ()
    #: 字幕の見た目（文） 例: 下に白の字・黒の縁 保存したプリセット「〜」を使う
    subtitles: str = ""
    #: テンポ（文） 例: 間は 0.3 秒まで詰める 1 カットは 5 秒まで
    tempo: str = ""
    #: 使ってよい効果の種類（エフェクトの ``kind``） 空なら決めない
    effects: tuple[str, ...] = ()
    #: そのほか（文）
    notes: str = ""

    @property
    def empty(self) -> bool:
        """何も決めていないか"""
        return self == StyleGuide()

    def cleaned(self) -> StyleGuide:
        """前後の空白を落とし、空の項目と重なりを除き、色を小文字にした方針

        画面の欄は 1 行に 1 つ書く 空の行や同じ書体を 2 度書いた行を残すと、照らしの結果に
        同じ物が 2 度並び、空の書体名が「決めた書体」に数えられる
        """
        return replace(
            self,
            fonts=_unique(item.strip() for item in self.fonts),
            colors=_unique(item.strip().lower() for item in self.colors),
            subtitles=self.subtitles.strip(),
            tempo=self.tempo.strip(),
            effects=_unique(item.strip() for item in self.effects),
            notes=self.notes.strip(),
        )

    def problems(self) -> list[str]:
        """方針として持てない所 無ければ空 画面とコマンドが同じ理由で断る"""
        found: list[str] = []
        bad = [color for color in self.colors if not valid_style_color(color)]
        if bad:
            found.append(f"色は #RRGGBB で書いてください: {', '.join(bad)}")
        for label, items, limit in _lists(self):
            if len(items) > limit:
                found.append(f"{label}は {limit} 個までです（{len(items)} 個）")
            if any(not item or _has_line_break(item) for item in items):
                # 1 行に 1 つの欄から来る 改行の入った名前は欄の区切りを取り違えた物
                found.append(f"{label}に空の名前か、改行の入った名前があります")
        for label, text in (
            ("字幕の見た目", self.subtitles),
            ("テンポ", self.tempo),
            ("そのほか", self.notes),
        ):
            if len(text) > STYLE_TEXT_LIMIT:
                found.append(f"{label}は {STYLE_TEXT_LIMIT} 文字までです（{len(text)} 文字）")
        return found

    def fitted(self) -> StyleGuide:
        """持てない所だけを落とした方針（:meth:`problems` が空になる）

        読めない色・改行の入った名前を除き、上限を越えた分の項目と文の後ろを切る 本人の設定を
        読むときに使う 丸ごと捨てると、手で足しすぎた 1 つのために書いた方針が全部消える
        """
        guide = self.cleaned()
        return replace(
            guide,
            fonts=_kept(guide.fonts, STYLE_ITEMS_LIMIT),
            colors=_kept(tuple(c for c in guide.colors if valid_style_color(c)), STYLE_ITEMS_LIMIT),
            subtitles=guide.subtitles[:STYLE_TEXT_LIMIT],
            tempo=guide.tempo[:STYLE_TEXT_LIMIT],
            effects=_kept(guide.effects, STYLE_EFFECTS_LIMIT),
            notes=guide.notes[:STYLE_TEXT_LIMIT],
        )


def _lists(guide: StyleGuide) -> tuple[tuple[str, tuple[str, ...], int], ...]:
    """名前の並びの欄（表示名, 中身, 上限）"""
    return (
        ("書体", guide.fonts, STYLE_ITEMS_LIMIT),
        ("色", guide.colors, STYLE_ITEMS_LIMIT),
        ("効果", guide.effects, STYLE_EFFECTS_LIMIT),
    )


def _kept(items: tuple[str, ...], limit: int) -> tuple[str, ...]:
    """改行の入った名前を除き、先頭から ``limit`` 個 本人が先に書いた物を残す"""
    return tuple(item for item in items if not _has_line_break(item))[:limit]


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    """空を除いて、初めに出た順で重なりを除く 並びは本人が書いた順を保つ"""
    seen: dict[str, None] = {}
    for item in items:
        if item:
            seen.setdefault(item, None)
    return tuple(seen)


def _has_line_break(text: str) -> bool:
    return len(text.splitlines()) > 1
