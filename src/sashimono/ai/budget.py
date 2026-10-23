"""AI が使った量（トークン・送った絵・時間）を数え、上限に当たったかを見る（#344）

数えるのは 3 つ

- **トークン** API を呼んだ 1 回ごとの入力 + キャッシュの書き込み + 出力 キャッシュの読み込みは
  数えない 値段が入力の 1/10 で、API を呼ぶたびに会話の全部を読み直すので、数えると 20 回の
  道具で数十万になり、上限が「長い会話を続けたか」の目安になってしまう
- **送った絵** 道具が返した絵（:class:`~sashimono.ai.operations.ImageResult`）の枚数 1 枚の
  費用は大きさでほぼ決まる（長い辺の上限は設定 :attr:`Limits.image_edge`）
- **時間** 指示に取りかかってからの秒

上限は 1 回の指示と 1 日の 2 つの単位で持つ 0 は上限なし 確かめるのはブリッジ
（:meth:`~sashimono.ai.bridge.EditorBridge.invoke`）が道具を動かす前 道具を動かさずに止めれば、
絵を描く前に止まる 文を書いているだけの間は止めない（1 回の応答の長さは API が区切る）

1 日の量はアプリを閉じても残す（:class:`DailyUsageStore` 本人の設定の置き場の ``ai_usage.json``）
外へは送らない 編集画面の数え方を覚えておくだけの物

このモジュールは Qt を知らない 試験と CLI からも同じ物を使う
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from sashimono.core.io.atomic import held_for_update, write_atomic

__all__ = [
    "DEFAULT_DAY_IMAGES",
    "DEFAULT_DAY_TOKENS",
    "DEFAULT_IMAGE_EDGE",
    "DEFAULT_TURN_IMAGES",
    "DEFAULT_TURN_MINUTES",
    "DEFAULT_TURN_TOKENS",
    "IMAGE_EDGES",
    "KEEP_DAYS",
    "DailyUsageStore",
    "LimitAction",
    "LimitKind",
    "Limits",
    "Overrun",
    "Usage",
    "UsageMeter",
    "format_minutes",
    "format_tokens",
    "message_tokens",
]

#: 1 回の指示で使ってよいトークンの既定 道具を 20 回ほど呼ぶ指示でおよそ 10 万〜20 万
#: （道具の説明を初めに書き込む 4〜6 万を含む） 100 回呼ぶ大きな指示でも収まり、同じ道具を
#: 繰り返して止まらなくなった指示はここで止まる
DEFAULT_TURN_TOKENS = 1_000_000
#: 1 回の指示で送ってよい絵の枚数の既定 場面ごとに見て確かめる指示（見渡しの 1 枚と、
#: 場面ごとに 1 枚 20 場面ほど）が収まる数
DEFAULT_TURN_IMAGES = 30
#: 1 回の指示の時間の上限の既定（分） 書き出しを待つ指示でも、長い作品でなければ収まる
DEFAULT_TURN_MINUTES = 30
#: 1 日に使ってよいトークンの既定 1 回の指示の上限の 10 回分
DEFAULT_DAY_TOKENS = 10_000_000
#: 1 日に送ってよい絵の枚数の既定 1 回の指示の上限の 10 回分
DEFAULT_DAY_IMAGES = 300
#: 絵 1 枚の長い辺の上限として選べる画素 1568 は古い型のモデルが縮めずに読める大きさ（#361）で、
#: これより大きくしても AI が細かく見られるわけではない 小さくすると費用が減る代わりに、
#: コマを並べた 1 枚の升が小さくなり、時刻の文字が読みにくくなる
IMAGE_EDGES = (768, 1024, 1280, 1568)
#: 絵 1 枚の長い辺の上限の既定 #361 で決めた上限のまま（設定を足す前と同じ絵を送る）
DEFAULT_IMAGE_EDGE = 1568
#: 1 日の量を何日分残すか 日をまたいだ確かめには今日の分しか使わないが、残しておけば
#: 本人がファイルを開いて最近の使い方を見られる 古い物を消さないとファイルが育ち続ける
KEEP_DAYS = 31
#: 時間だけが進んだとき、1 日の置き場へ書くまで溜める秒（:meth:`UsageMeter._save`）
SAVE_SECONDS = 5.0


class LimitKind(Enum):
    """どの上限か"""

    TURN_TOKENS = "turn_tokens"
    TURN_IMAGES = "turn_images"
    TURN_SECONDS = "turn_seconds"
    DAY_TOKENS = "day_tokens"
    DAY_IMAGES = "day_images"

    @property
    def label(self) -> str:
        """人と AI に見せる名前"""
        return _KIND_LABELS[self]

    @property
    def daily(self) -> bool:
        """1 日の単位か 広げた分を指示が終わっても残すかの見分けに使う"""
        return self in (LimitKind.DAY_TOKENS, LimitKind.DAY_IMAGES)


_KIND_LABELS = {
    LimitKind.TURN_TOKENS: "1 回の指示のトークン",
    LimitKind.TURN_IMAGES: "1 回の指示で送る絵",
    LimitKind.TURN_SECONDS: "1 回の指示の時間",
    LimitKind.DAY_TOKENS: "1 日のトークン",
    LimitKind.DAY_IMAGES: "1 日に送る絵",
}


class LimitAction(Enum):
    """上限に当たったときにどうするか（本人の設定 :attr:`Limits.action`）"""

    #: 止めて、続けるかを人に尋ねる 尋ねている間は道具を動かさない
    ASK = "ask"
    #: 尋ねずに止めて、止めたことを知らせる
    STOP = "stop"

    @property
    def label(self) -> str:
        return _ACTION_LABELS[self]


_ACTION_LABELS = {
    LimitAction.ASK: "止めて、続けるかを尋ねる",
    LimitAction.STOP: "尋ねずに止めて知らせる",
}


@dataclass(frozen=True, slots=True)
class Limits:
    """上限 0 は上限なし 本人の設定（``Preferences``）から作る"""

    turn_tokens: int = DEFAULT_TURN_TOKENS
    turn_images: int = DEFAULT_TURN_IMAGES
    #: 秒で持つ 設定画面は分で出す（秒の単位で決めたい人はいない）
    turn_seconds: int = DEFAULT_TURN_MINUTES * 60
    day_tokens: int = DEFAULT_DAY_TOKENS
    day_images: int = DEFAULT_DAY_IMAGES
    action: LimitAction = LimitAction.ASK
    #: 絵 1 枚の長い辺の上限（画素） 数える上限ではなく、道具が描く大きさを決める
    image_edge: int = DEFAULT_IMAGE_EDGE

    def value(self, kind: LimitKind) -> int:
        return {
            LimitKind.TURN_TOKENS: self.turn_tokens,
            LimitKind.TURN_IMAGES: self.turn_images,
            LimitKind.TURN_SECONDS: self.turn_seconds,
            LimitKind.DAY_TOKENS: self.day_tokens,
            LimitKind.DAY_IMAGES: self.day_images,
        }[kind]


@dataclass(frozen=True, slots=True)
class Usage:
    """使った量"""

    tokens: int = 0
    images: int = 0
    seconds: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            tokens=self.tokens + other.tokens,
            images=self.images + other.images,
            seconds=self.seconds + other.seconds,
        )

    def __sub__(self, other: Usage) -> Usage:
        return Usage(
            tokens=self.tokens - other.tokens,
            images=self.images - other.images,
            seconds=self.seconds - other.seconds,
        )


@dataclass(frozen=True, slots=True)
class Overrun:
    """当たった上限 1 つ"""

    kind: LimitKind
    #: 今の量（絵は、これから送る 1 枚を足した数）
    used: float
    #: 当たった上限（広げた分を含む）
    limit: int
    #: 1 回で広げる量（設定の上限そのもの）
    step: int

    def text(self) -> str:
        """人と AI に見せる 1 文"""
        return (
            f"{self.kind.label}が上限（{_amount(self.kind, self.limit)}）に達しました"
            f"（今 {_amount(self.kind, self.used)}）"
        )

    def extended(self) -> str:
        """〔続ける〕で広げた後の上限"""
        return _amount(self.kind, self.limit + self.step)


def _amount(kind: LimitKind, value: float) -> str:
    if kind in (LimitKind.TURN_TOKENS, LimitKind.DAY_TOKENS):
        return f"{format_tokens(int(value))} トークン"
    if kind is LimitKind.TURN_SECONDS:
        return format_minutes(value)
    return f"{int(value)} 枚"


def format_tokens(tokens: int) -> str:
    """トークンの数を読みやすく 1 万を超えたら「万」で出す（1234567 より 123.5 万の方が掴める）"""
    if tokens < 10_000:
        return f"{tokens:,}"
    man = tokens / 10_000
    return f"{man:.0f} 万" if man >= 100 else f"{man:.1f} 万"


def format_minutes(seconds: float) -> str:
    """時間を読みやすく 1 分を超えたら分も出す"""
    whole = max(0, int(seconds))
    minutes, rest = divmod(whole, 60)
    if not minutes:
        return f"{rest} 秒"
    return f"{minutes} 分" if not rest else f"{minutes} 分 {rest} 秒"


#: 数えるトークンの欄 キャッシュの読み込み（``cache_read_input_tokens``）は数えない
#: （訳はモジュールの説明）
_COUNTED = ("input_tokens", "cache_creation_input_tokens", "output_tokens")


def message_tokens(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """SDK の 1 通の ``usage`` から、数える欄だけを整数で取り出す 壊れた値は 0 にする"""
    if not usage:
        return {}
    counted: dict[str, int] = {}
    for key in _COUNTED:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            counted[key] = value
    return counted


class DailyUsageStore:
    """1 日ごとの使った量を持つファイル

    形は ``{"2026-10-19": {"tokens": 0, "images": 0, "seconds": 0}}``

    書くのは足す分だけ（:meth:`add`） 読んで足して書き戻すまでを錠で守る 2 つの窓が
    それぞれ自分の合計を書くと、後から書いた方が先の窓の分を消す
    読めない・書けないときは数えないだけで、AI の作業は止めない（設定の置き場が書けない機械で
    AI が使えなくなる方が困る）
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, day: date) -> Usage:
        return _usage_of(self._read().get(day.isoformat()))

    def add(self, day: date, usage: Usage) -> bool:
        """足せたか 足せなかった差は呼んだ側が持ち続け、次の機会に足し直す

        書けないときに AI を止めないのは同じ 偽を返すだけにする 書けなかった差を足し終えた
        扱いにすると、その分は二度と足されず、1 日の量を少なく数えて上限を越えて使える
        """
        if usage == Usage():
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with held_for_update(self.path.with_name(self.path.name + ".lock")):
                data = self._read()
                key = day.isoformat()
                data[key] = _usage_json(_usage_of(data.get(key)) + usage)
                oldest = (day - timedelta(days=KEEP_DAYS)).isoformat()
                kept = {name: value for name, value in data.items() if name > oldest}
                write_atomic(self.path, json.dumps(kept, ensure_ascii=False, indent=2))
        except OSError:
            return False
        return True

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


def _usage_of(value: object) -> Usage:
    """ファイルの 1 日分 壊れた欄は 0 にする（手で書き換えられても起動と AI を止めない）"""
    if not isinstance(value, dict):
        return Usage()

    def number(key: str) -> float:
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int | float) or raw < 0:
            return 0
        return raw

    return Usage(
        tokens=int(number("tokens")), images=int(number("images")), seconds=number("seconds")
    )


def _usage_json(usage: Usage) -> dict[str, float]:
    return {"tokens": usage.tokens, "images": usage.images, "seconds": round(usage.seconds, 1)}


class _MemoryStore:
    """覚えない置き場（試験・CLI） 1 日の量はこの数えが生きている間だけ持つ

    置き場が無いときも同じ道（足してから読み直す）で数える 道を分けると、片方だけで
    1 日の上限の数え方が食い違う
    """

    def __init__(self) -> None:
        self._days: dict[date, Usage] = {}

    def load(self, day: date) -> Usage:
        return self._days.get(day, Usage())

    def add(self, day: date, usage: Usage) -> bool:
        self._days[day] = self._days.get(day, Usage()) + usage
        return True


@dataclass(slots=True)
class _Turn:
    """いま応えている指示の数え"""

    started: float
    day: date
    #: 1 日の置き場の今日の量（最後に足して読み直した時点） ほかの窓の分と、この指示の
    #: 足し終えた分（:attr:`saved`）を含む
    day_seen: Usage
    tokens: int = 0
    images: int = 0
    #: この指示の量のうち、1 日の置き場へ足し終えた分 足すのは差だけにする
    saved: Usage = field(default_factory=Usage)
    #: API の 1 回（``message_id``）ごとに、欄ごとの一番大きい値 同じ応答が中身の塊ごとに
    #: 何通かに分かれて届くので、足すと 2 重 3 重に数える
    messages: dict[str, dict[str, int]] = field(default_factory=dict)
    #: この指示で数えた出力のトークン（途中の知らせの仮の値と、結果で確かめた差の合計）
    #: 結果（:meth:`UsageMeter.settle_output`）はこれとの差だけを足す
    output: int = 0
    #: 〔続ける〕で広げた分（この指示の間だけ）
    extra: dict[LimitKind, int] = field(default_factory=dict)
    #: 止めると決まった上限 決まった後の道具は尋ね直さずに断る
    stopped: Overrun | None = None


class UsageMeter:
    """使った量を数える AI のスレッド（トークン・絵）と画面のスレッド（指示の始まりと終わり）
    の両方から呼ばれるので、中は錠で守る

    量の流れ

    1. **数える** 指示の中で届いた量は指示（``_Turn``）に、指示の外で届いた量は日ごとの
       「まだ書いていない分」（``_unsaved``）に入れる 指示の外の量を次の指示へ入れると、
       前の指示の量で次の指示の 1 回の上限に当たる
    2. **足す** 指示の差は、上限を確かめるたび（道具の前）・:meth:`flush`・指示の終わりに
       1 日の置き場へ足す 終わりにだけ足すと、窓を閉じた・落ちた指示の量が残らない
       書けたときだけ「足し終えた」所を進める 書けなかった差は ``_unsaved`` へ移し、
       次の機会に足し直す（1 回の指示の量には入れない）
    3. **読み直す** 足した後に置き場を読み直す（ほかの窓の分を含む） 始めた時の量のままだと、
       同時に指示した 2 つの窓が上限の 2 倍まで使える
    4. **確かめる** 1 日の量は 置き場の量 + まだ書けていない分（``_unsaved`` と指示の差）
       書けない間も、1 日の上限から量が抜けない
    """

    def __init__(
        self,
        store: DailyUsageStore | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._store: DailyUsageStore | _MemoryStore = store if store is not None else _MemoryStore()
        self._clock = clock
        self._today = today
        self._lock = threading.Lock()
        self.limits = Limits()
        self._turn: _Turn | None = None
        #: 1 日の置き場へまだ書いていない分（日ごと） 指示の外で届いた量と、書けなかった差
        #: 1 日の量には入れ、1 回の指示の量には入れない
        self._unsaved: dict[date, Usage] = {}
        #: 1 日の上限を〔続ける〕で広げた分 その日のうちは残す（指示ごとに尋ね直さない）
        self._day_extra: tuple[date, dict[LimitKind, int]] = (today(), {})
        #: 最後に終えた指示の量 区切りの行に出す
        self.last = Usage()
        #: 最後に終えた指示を上限で止めたなら、その上限 区切りの行を「上限で停止」にする
        self.last_stop: Overrun | None = None
        #: 指示の外で見せる今日の置き場の量（日, 量） 日が違えば読み直す
        #: 初めは必ず読むよう、ありえない日にする
        self._day_base: tuple[date, Usage] = (date.min, Usage())

    # --- 画面のスレッドから ---

    def begin_turn(self) -> None:
        """指示に取りかかった 前の指示が終わっていなければ、先に締める"""
        if self._turn is not None:
            self.end_turn()
        day = self._today()
        with self._lock:
            self._retry_unsaved()
            self._turn = _Turn(started=self._clock(), day=day, day_seen=self._store.load(day))

    def end_turn(self) -> Usage:
        """指示が終わった 残りの差を 1 日の置き場へ足し、この指示の量を返す

        窓を閉じるとき（応えている途中でも）も呼ぶ まだ書いていない分もここで足し直す
        """
        with self._lock:
            turn = self._turn
            self._turn = None
            used = Usage()
            # 数えていなかった指示（部品の入れ替え中に送り、渡す前に終わった指示など）も、
            # 「最後に終えた指示」は今の指示 前の指示の量と上限を残すと、関係の無い指示の
            # 区切りの行と引き継ぎ（#346）に「上限で停止」と前の量が書かれる
            self.last = used
            self.last_stop = None
            if turn is not None:
                self._save(turn, force=True)
                # 書けなかったときも、区切りの行には使った量を出す
                used = self._current(turn)
                self.last = used
                self.last_stop = turn.stopped
                pending = used - turn.saved
                if pending.tokens or pending.images:
                    # 書けなかった差は、その指示の日の分として持ち続ける 次の指示の量には入れない
                    self._keep_unsaved(turn.day, pending)
            self._retry_unsaved()
            today = self._today()
            self._day_base = (today, self._store.load(today))
            return used

    def flush(self) -> None:
        """まだ書いていない分を 1 日の置き場へ足す 画面が時々呼ぶ（落ちたときに失う量を減らす）"""
        with self._lock:
            self._retry_unsaved()
            if self._turn is not None:
                self._save(self._turn)
            else:
                # 指示の外で見せる量は 置き場の量（_day_base）+ まだ書いていない分 書いた分は
                # まだ書いていない分から消えるので、置き場の量も読み直す 読み直さないと、
                # 書けた分だけ今日の量が少なく見える
                today = self._today()
                self._day_base = (today, self._store.load(today))

    # --- AI のスレッドから ---

    def record_message(self, message_id: str | None, usage: Mapping[str, Any] | None) -> None:
        """API の 1 回の ``usage`` を足す 同じ ``message_id`` は欄ごとに大きい方だけを残す"""
        counted = message_tokens(usage)
        if not counted:
            return
        with self._lock:
            turn = self._turn
            if turn is None or message_id is None:
                # 印の無い知らせ（古い SDK）はそのまま足す 2 重に数えうるが、少なく数えて
                # 上限が効かないより安全な側
                fresh = dict(counted)
            else:
                seen = turn.messages.setdefault(message_id, {})
                fresh = {}
                for key, value in counted.items():
                    if value > seen.get(key, 0):
                        fresh[key] = value - seen.get(key, 0)
                        seen[key] = value
            added = sum(fresh.values())
            if turn is None:
                # 指示の外で届いた量は 1 日の分にだけ入れる（次の指示の上限に混ぜない）
                self._keep_unsaved(self._today(), Usage(tokens=added))
            else:
                turn.tokens += added
                turn.output += fresh.get("output_tokens", 0)

    def settle_output(self, usage: Mapping[str, Any] | None) -> None:
        """1 回の指示の結果（SDK の ``ResultMessage``）の出力のトークンで、数えた出力を確かめる

        途中の知らせ（``AssistantMessage``）の ``output_tokens`` は応答の始めの仮の値で、
        同じ応答の知らせはどれも同じ値を繰り返す（claude-agent-sdk 0.2.164 で 332 の応答が
        6 と届いた） 本当の出力は結果に、その指示（query 1 回）の合計として届く 足すのは
        それまでに数えた出力との差だけ 同じ結果を 2 度受けても、2 度目の差は 0 になる
        指示の外で届いた結果は、どれだけ数えたかを持っていないので足さない
        """
        total = message_tokens(usage).get("output_tokens", 0)
        with self._lock:
            turn = self._turn
            if turn is None or total <= turn.output:
                return
            turn.tokens += total - turn.output
            turn.output = total

    def record_image(self) -> None:
        """絵を 1 枚送った"""
        with self._lock:
            if self._turn is None:
                self._keep_unsaved(self._today(), Usage(images=1))
            else:
                self._turn.images += 1

    def overrun(self, *, adding_images: int = 0) -> Overrun | None:
        """当たった上限 無ければ ``None`` ``adding_images`` はこれから送る絵の枚数

        止めると決まった指示では、決まった上限を返し続ける 1 日の量は、まだ書いていない分を
        足し直して置き場を読み直した量 + それでも書けなかった分（:meth:`_day_used`）で見る
        """
        with self._lock:
            turn = self._turn
            if turn is None:
                return None
            if turn.stopped is not None:
                return turn.stopped
            self._retry_unsaved()
            self._save(turn)
            limits = self.limits
            day_extra = self._day_extra_now()
            day_used = self._day_used(turn)
            checks: list[tuple[LimitKind, float, int]] = [
                (LimitKind.TURN_SECONDS, self._clock() - turn.started, 0),
                (LimitKind.TURN_TOKENS, turn.tokens, 0),
                (LimitKind.DAY_TOKENS, day_used.tokens, 0),
            ]
            if adding_images > 0:
                # 絵の上限は絵を送る道具だけを止める 送らない道具まで止めると、上限まで
                # 見た後の編集や後片付けもできなくなる
                checks += [
                    (LimitKind.TURN_IMAGES, turn.images, adding_images),
                    (LimitKind.DAY_IMAGES, day_used.images, adding_images),
                ]
            for kind, used, adding in checks:
                step = limits.value(kind)
                if step <= 0:
                    continue
                extra = day_extra.get(kind, 0) if kind.daily else turn.extra.get(kind, 0)
                limit = step + extra
                # 絵は送った後に越える 1 枚を先に止める トークンと時間は使い切った所で止める
                over = used + adding > limit if adding else used >= limit
                if over:
                    return Overrun(kind=kind, used=used + adding, limit=limit, step=step)
            return None

    def extend(self, kind: LimitKind) -> None:
        """〔続ける〕 当たった上限を設定の量だけ広げる 1 日の分はその日のうち残す"""
        with self._lock:
            step = self.limits.value(kind)
            if kind.daily:
                extra = self._day_extra_now()
                extra[kind] = extra.get(kind, 0) + step
            elif self._turn is not None:
                self._turn.extra[kind] = self._turn.extra.get(kind, 0) + step

    def stop(self, overrun: Overrun) -> None:
        """この指示は止めると決まった 残りの道具は尋ね直さずに断る"""
        with self._lock:
            if self._turn is not None:
                self._turn.stopped = overrun

    @property
    def stopped(self) -> Overrun | None:
        with self._lock:
            return self._turn.stopped if self._turn is not None else None

    # --- 見せる ---

    def turn_usage(self) -> Usage | None:
        """いまの指示の量 応えている指示が無ければ ``None``"""
        with self._lock:
            turn = self._turn
            return None if turn is None else self._current(turn)

    def day_usage(self) -> Usage:
        """今日の量（ほかの窓の分を含む）

        置き場を読むのは差を足した時と、日が変わった後に初めて見た時だけ 画面は 80 ミリ秒
        ごとに量を出し直すので、毎回読むとディスクを叩き続ける 見せる量は少し古くてよいが、
        上限の確かめ（:meth:`overrun`）は必ず読み直す
        """
        with self._lock:
            turn = self._turn
            if turn is not None:
                return self._day_used(turn)
            day, base = self._day_base
            today = self._today()
            if day != today:
                base = self._store.load(today)
                self._day_base = (today, base)
            return base + self._unsaved.get(today, Usage())

    # --- 錠の中で呼ぶ ---

    def _current(self, turn: _Turn) -> Usage:
        return Usage(tokens=turn.tokens, images=turn.images, seconds=self._clock() - turn.started)

    def _day_used(self, turn: _Turn) -> Usage:
        """指示の日の量 置き場の量（最後に読んだ時点）+ まだ書けていない分"""
        return (
            turn.day_seen
            + self._unsaved.get(turn.day, Usage())
            + (self._current(turn) - turn.saved)
        )

    def _save(self, turn: _Turn, *, force: bool = False) -> None:
        """この指示のまだ足していない差を置き場へ足し、今日の量を読み直す

        時間だけの差は :data:`SAVE_SECONDS` に満ちるまで書かない 道具のたびに数ミリ秒の
        差を書くと、ファイルを書き換えるだけの手間が増える（時間は 1 日の上限に使わない）
        """
        current = self._current(turn)
        delta = current - turn.saved
        today = self._today()
        if today != turn.day:
            # 指示の途中で日が変わった 足し終えた分は前の日に残し、ここからは今日へ足して
            # 今日の量で確かめる 前の日のまま見ると、0 時を過ぎた指示が昨日の量で止まり、
            # 今日の分まで昨日へ書かれる 前に足してから日をまたぐまでの差は、いつ使ったかを
            # 分けられないので今日の分にする
            turn.day = today
        due = force or delta.tokens or delta.images or delta.seconds >= SAVE_SECONDS
        # 書けたときだけ足し終えた所を進める 書けなかった差は次の機会に足し直す
        if due and self._store.add(turn.day, delta):
            turn.saved = current
        turn.day_seen = self._store.load(turn.day)

    def _keep_unsaved(self, day: date, usage: Usage) -> None:
        self._unsaved[day] = self._unsaved.get(day, Usage()) + usage

    def _retry_unsaved(self) -> None:
        """まだ書いていない分を足し直す 書けた日だけ消す"""
        for day, pending in list(self._unsaved.items()):
            if self._store.add(day, pending):
                del self._unsaved[day]

    def _day_extra_now(self) -> dict[LimitKind, int]:
        """今日の広げた分 日が変わっていたら捨てる"""
        day, extra = self._day_extra
        today = self._today()
        if day != today:
            extra = {}
            self._day_extra = (today, extra)
        return extra
