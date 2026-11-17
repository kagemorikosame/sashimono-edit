"""演出のレシピとジャンルごとのひとそろい、強さの加減（#470 段 3 #478）

「笑い声の直後に寄ってから戻す」のような演出の決まりを、指示の文でなく宣言として持つ
指示の文に書くと、同じ素材・同じ頼みでも AI の気分で付ける数が変わり、数えて試せない
宣言にしておけば、強さごとに何回まで・何秒あけるかを段 1（:mod:`sashimono.core.direction`）の
測り方で確かめられる

レシピは「何を手がかりに（``cue``）」「何を付けるか（``action``）」「強さごとに 1 分に何回まで・
どれだけ強く」を持つ 手がかりの名前は盛り上がりの道具（``engine/highlights.py`` の
``SIGNALS``）と同じ文字列 コア層は engine を読まないので、名前がそろうことは試験で見る

効果音は本人の素材だけを使う（同梱しない 2026-11-16 に本人が決めた） レシピは「効果音を
置く」とだけ言い、どの音を使うかは本人の素材から AI が選ぶ
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from sashimono.core.direction import KINDS_BY_KEY, SHAKE, SOUND, ZOOM

__all__ = [
    "ACTIONS",
    "ACTION_SHAKE",
    "ACTION_SOUND",
    "ACTION_ZOOM",
    "BOLD",
    "CUES",
    "CUE_LAUGHTER",
    "CUE_LOUD",
    "CUE_MOTION",
    "DEFAULT_GENRE",
    "DEFAULT_STRENGTH",
    "GENRES",
    "LETS_PLAY",
    "NORMAL",
    "RECIPE_SETS",
    "STRENGTHS",
    "SUBTLE",
    "Cue",
    "DirectionRecipe",
    "DirectionTaste",
    "PlannedDirection",
    "RecipeBook",
    "RecipeRule",
    "RecipeSet",
    "plan_direction",
    "recipe_book",
]

#: 手がかりの名前 盛り上がりの道具（find_highlights）の signals と同じ
CUE_LAUGHTER = "laughter"
CUE_LOUD = "loud"
CUE_MOTION = "motion"
CUES: tuple[str, ...] = (CUE_LAUGHTER, CUE_LOUD, CUE_MOTION)

#: 強さの加減 並びは弱い順
SUBTLE = "subtle"
NORMAL = "normal"
BOLD = "bold"
STRENGTHS: Mapping[str, str] = {SUBTLE: "控えめ", NORMAL: "ふつう", BOLD: "派手"}
#: 知らない人が困らない側 付けすぎた演出は 1 つずつ外す手間が要るが、足りない分は頼み直せば
#: 足せる（2026-11-16 に本人が決めた）
DEFAULT_STRENGTH = SUBTLE

#: ジャンル 最初は実況だけ（2026-11-16 に本人が決めた） 足すときは :data:`RECIPE_SETS` へ
LETS_PLAY = "lets_play"
GENRES: Mapping[str, str] = {LETS_PLAY: "実況"}
DEFAULT_GENRE = LETS_PLAY

#: 付ける物 寄る・揺らすは強調の動き（core/emphasis.py の種類）、効果音は本人の素材を置く
ACTION_ZOOM = "zoom"
ACTION_SHAKE = "shake"
ACTION_SOUND = "sound"
ACTIONS: Mapping[str, str] = {
    ACTION_ZOOM: "寄って戻す",
    ACTION_SHAKE: "揺らす",
    ACTION_SOUND: "効果音を置く",
}
#: 付ける物が段 1 のどの型に数えられるか 測り方と同じ型で間隔と回数を見る
_KIND_OF_ACTION: Mapping[str, str] = {
    ACTION_ZOOM: ZOOM,
    ACTION_SHAKE: SHAKE,
    ACTION_SOUND: SOUND,
}
#: 付ける物が使うエフェクトの種類 方針（使ってよい効果）の外ならレシピから外す
#: 寄るは配置の欄（全部のクリップが持つ）で付けるので、方針の効果の欄とは関わらない
_EFFECT_OF_ACTION: Mapping[str, str] = {ACTION_SHAKE: "random_move"}


@dataclass(frozen=True, slots=True)
class DirectionTaste:
    """本人の好み（ジャンルと強さ） 設定画面で選び、頼みの文で上書きできる"""

    genre: str = DEFAULT_GENRE
    strength: str = DEFAULT_STRENGTH

    def __post_init__(self) -> None:
        # 知らない値のまま進むと、レシピを引く所で落ちる 作る所で断る
        if self.genre not in GENRES:
            raise ValueError(f"知らないジャンル: {self.genre}")
        if self.strength not in STRENGTHS:
            raise ValueError(f"知らない強さ: {self.strength}")


@dataclass(frozen=True, slots=True)
class DirectionRecipe:
    """演出の決まり 1 つ

    ``per_minute`` は強さごとの 1 分あたりの上限（0 はその強さでは使わない）
    ``amount`` は強さごとの強さの値（寄るは拡大率の %、揺らすは幅の画素 効果音は持たない）
    同じ型の演出どうしの間隔は ``60 / per_minute`` 秒より詰めない 上限の回数を
    1 か所へ固めて置くと、1 分の中では回数どおりでも見ている人には付けすぎに見える
    """

    key: str
    name: str
    cue: str
    action: str
    #: AI と人へそのまま見せる、なぜこの手がかりにこれを付けるか
    reason: str
    per_minute: Mapping[str, float]
    amount: Mapping[str, int] = field(default_factory=dict)
    #: 付ける長さ（秒） 寄って戻す・揺らす長さ 効果音は素材の長さで決まるので見ない
    seconds: float = 1.0

    @property
    def kind(self) -> str:
        """段 1 の演出の型"""
        return _KIND_OF_ACTION[self.action]


@dataclass(frozen=True, slots=True)
class RecipeSet:
    """ジャンルごとのひとそろい"""

    genre: str
    recipes: tuple[DirectionRecipe, ...]


#: ジャンルごとのひとそろい
#: 実況の数は、作品 A〜C（#477 で測った本人の実況）の寄る・揺らす・場面切り替えが 0 回、
#: 効果音が 1 分に 0〜1.9 回だったことから、控えめを「ほぼ今の作品のまま少し足す」に置いた
#: 派手でも段 1 のやりすぎの線（寄る 8 秒・揺らす 10 秒・効果音 2 秒）より必ず間をあける
RECIPE_SETS: Mapping[str, RecipeSet] = {
    LETS_PLAY: RecipeSet(
        LETS_PLAY,
        (
            DirectionRecipe(
                key="laugh_zoom",
                name="笑いに寄る",
                cue=CUE_LAUGHTER,
                action=ACTION_ZOOM,
                reason="笑い声の所で話者へ寄ってから戻すと、どこが面白かったかが伝わる",
                per_minute={SUBTLE: 1.0, NORMAL: 2.0, BOLD: 3.0},
                # 110% は寄ったと分かる最小の量 130% を超えると横並びの 2 人の片方が外れる
                amount={SUBTLE: 110, NORMAL: 120, BOLD: 130},
                seconds=1.5,
            ),
            DirectionRecipe(
                key="loud_shake",
                name="叫びに揺らす",
                cue=CUE_LOUD,
                action=ACTION_SHAKE,
                reason="急に大きな声が出た所を揺らすと、驚きやツッコミが伝わる",
                per_minute={SUBTLE: 0.5, NORMAL: 1.0, BOLD: 2.0},
                # 6px は揺れたと分かる最小の量 20px を超えると字幕が読めない
                amount={SUBTLE: 6, NORMAL: 12, BOLD: 20},
                seconds=0.5,
            ),
            DirectionRecipe(
                key="laugh_sound",
                name="笑いに効果音",
                cue=CUE_LAUGHTER,
                action=ACTION_SOUND,
                reason="笑いの所に短い効果音を重ねると、間が締まる 音は本人の素材から選ぶ",
                per_minute={SUBTLE: 0.5, NORMAL: 1.0, BOLD: 2.0},
            ),
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class RecipeRule:
    """強さを決めたレシピ 1 つ（AI へそのまま見せる形）"""

    recipe: DirectionRecipe
    per_minute: float
    min_gap_seconds: float
    #: 強さの値 持たない物は ``None``
    amount: int | None


@dataclass(frozen=True, slots=True)
class RecipeBook:
    """好みに合わせて引いたレシピの一覧"""

    taste: DirectionTaste
    rules: tuple[RecipeRule, ...]
    #: 方針の外なので外したレシピと、その理由
    left_out: tuple[tuple[DirectionRecipe, str], ...] = ()


def recipe_book(taste: DirectionTaste, allowed_effects: Iterable[str] = ()) -> RecipeBook:
    """``taste`` のジャンルと強さでレシピを引く

    ``allowed_effects`` は方針（#343）の使ってよい効果 空なら決めていないので外さない
    方針の外の効果を使うレシピは外す 付けてから照らすと、方針の外として本人に直させる
    """
    allowed = frozenset(allowed_effects)
    rules: list[RecipeRule] = []
    left_out: list[tuple[DirectionRecipe, str]] = []
    for recipe in RECIPE_SETS[taste.genre].recipes:
        effect = _EFFECT_OF_ACTION.get(recipe.action)
        if allowed and effect is not None and effect not in allowed:
            left_out.append((recipe, f"方針の使ってよい効果に {effect} がありません"))
            continue
        per_minute = recipe.per_minute.get(taste.strength, 0.0)
        if per_minute <= 0:
            continue
        rules.append(
            RecipeRule(
                recipe=recipe,
                per_minute=per_minute,
                min_gap_seconds=_min_gap(recipe, per_minute),
                amount=recipe.amount.get(taste.strength),
            )
        )
    return RecipeBook(taste, tuple(rules), tuple(left_out))


def _min_gap(recipe: DirectionRecipe, per_minute: float) -> float:
    """同じ型どうしの間隔の下限 回数の上限から決め、段 1 のやりすぎの線より詰めない"""
    limit = KINDS_BY_KEY[recipe.kind].min_gap_seconds or 0.0
    return max(60.0 / per_minute, limit)


@dataclass(frozen=True, slots=True)
class Cue:
    """手がかり 1 つ（盛り上がりの道具が返した区間） 秒はタイムラインの位置"""

    signal: str
    start: float
    end: float
    #: 盛り上がりの強さ（0〜1） 強い所から選ぶ
    score: float


@dataclass(frozen=True, slots=True)
class PlannedDirection:
    """付ける演出 1 つ（計画の項目） 秒はタイムラインの位置"""

    recipe: str
    action: str
    kind: str
    start: float
    end: float
    amount: int | None
    reason: str


def plan_direction(
    cues: Sequence[Cue], book: RecipeBook, start: float, end: float
) -> list[PlannedDirection]:
    """手がかりから付ける演出を選ぶ 同じ手がかり・同じ好みなら毎回同じ物を返す

    レシピごとに、強い手がかりから順に、範囲の長さに見合う回数まで選ぶ 同じ型の演出は
    レシピをまたいで間隔の下限より詰めない（寄るのレシピが 2 つあっても、合わせて詰まらない）
    並びは頭の位置の順
    """
    if end <= start:
        raise ValueError("範囲が空です")
    minutes = (end - start) / 60
    chosen: list[PlannedDirection] = []
    taken: dict[str, list[float]] = {}
    for rule in book.rules:
        recipe = rule.recipe
        # 範囲の長さに見合う回数 1 分に満たない範囲でも、半分を越えていれば 1 回は置ける
        budget = math.floor(rule.per_minute * minutes + 0.5)
        if budget <= 0:
            continue
        # 強い順 同じ強さは早い順にして、何度呼んでも同じ物を選ぶ
        candidates = sorted(
            (c for c in cues if c.signal == recipe.cue and start <= c.start < end),
            key=lambda c: (-c.score, c.start, c.end),
        )
        placed = taken.setdefault(recipe.kind, [])
        count = 0
        for cue in candidates:
            if count >= budget:
                break
            if any(abs(cue.start - other) < rule.min_gap_seconds for other in placed):
                continue
            placed.append(cue.start)
            count += 1
            chosen.append(
                PlannedDirection(
                    recipe=recipe.key,
                    action=recipe.action,
                    kind=recipe.kind,
                    start=cue.start,
                    end=min(cue.start + recipe.seconds, end),
                    amount=rule.amount,
                    reason=recipe.reason,
                )
            )
    chosen.sort(key=lambda item: (item.start, item.kind, item.recipe))
    return chosen
