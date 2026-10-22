"""互換性の穴を記録する

AviUtl の ``obj`` API は広く、全部を一度に実装することはできない 大事なのは
「動かない」ことではなく、**何が足りなくて動かないのかが分かること**

スクリプトが未対応の関数を呼んだら、落とさずに記録して先へ進む あとで一覧を
見れば、実際に使われている API が使用回数つきで並ぶので、次に何を実装すべきかを
勘ではなくデータで決められる
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field

__all__ = [
    "AVIUTL_OBJ_NAMES",
    "KNOWN_RECORDS",
    "CompatibilityReport",
    "ScriptFailure",
    "global_report",
    "is_known_record",
]


@dataclass(frozen=True, slots=True)
class ScriptFailure:
    """スクリプトの実行が失敗した記録"""

    script: str
    message: str


@dataclass
class CompatibilityReport:
    """未対応 API と失敗の記録

    複数のスクリプトが描画スレッドから同時に触るので、数える処理だけ錠を掛ける
    """

    #: 呼ばれた未対応 API の名前と回数
    missing: Counter[str] = field(default_factory=Counter)
    #: 解釈できなかった制御文字
    controls: Counter[str] = field(default_factory=Counter)
    #: 実行時の失敗 同じものが何度も出るので、最初の 1 件だけ残す
    failures: dict[str, ScriptFailure] = field(default_factory=dict)
    #: 消した回数 覚えた記録を器へ 1 度だけ伝える所（汎用プラグインの読めなかった
    #: 理由など）が、消されたことに気付いて伝え直すのに使う
    cleared: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def note_missing(self, name: str) -> None:
        with self._lock:
            self.missing[_bounded(self.missing, name)] += 1

    def note_control(self, line: str) -> None:
        with self._lock:
            self.controls[_bounded(self.controls, line)] += 1

    def note_failure(self, script: str, message: str) -> None:
        with self._lock:
            self.failures.setdefault(script, ScriptFailure(script, message))

    def clear(self) -> None:
        with self._lock:
            self.missing.clear()
            self.controls.clear()
            self.failures.clear()
            self.cleared += 1

    @property
    def is_empty(self) -> bool:
        return not (self.missing or self.controls or self.failures)

    def summary(self) -> str:
        """1 行の要約 ステータスバーに出す"""
        if self.is_empty:
            return "未対応の呼び出しはありません"
        parts = []
        if self.missing:
            parts.append(f"未実装 API {len(self.missing)} 種")
        if self.controls:
            parts.append(f"未対応の制御文字 {len(self.controls)} 種")
        if self.failures:
            parts.append(f"失敗 {len(self.failures)} 件")
        return "、".join(parts)

    def records(
        self,
    ) -> tuple[
        tuple[tuple[str, int], ...], tuple[tuple[str, int], ...], tuple[tuple[str, str], ...]
    ]:
        """記録の写し（未実装の API・制御文字は多い順、失敗はスクリプトと文）

        AI が読む口（#352）が使う 3 つを同じ錠の中で写す 錠の外で辞書を辿ると、描く裏の
        スレッドが同じときに数え足して「辿る間に大きさが変わった」で落ちる
        """
        with self._lock:
            return (
                tuple(self.missing.most_common()),
                tuple(self.controls.most_common()),
                tuple((f.script, f.message) for f in self.failures.values()),
            )

    def lines(self) -> tuple[str, ...]:
        """一覧に出す行 多い順に並べる"""
        with self._lock:
            rows = [f"{name} — {count} 回" for name, count in self.missing.most_common()]
            rows.extend(f"{line} — {count} 回" for line, count in self.controls.most_common())
            rows.extend(f"{f.script}: {f.message}" for f in self.failures.values())
        return tuple(rows)


#: AviUtl のスクリプトから見える ``obj`` の名前（関数と値） AviUtl と AviUtl2 の決まった名前で、
#: アプリ側が持つ一覧（:mod:`~sashimono.compat.aviutl.objapi` が受ける名前もすべて入る 試験が見る）
#:
#: 互換性レポートを AI へ送るとき、記録の文がこれと決まった形（:data:`KNOWN_RECORDS`）に
#: ちょうど一致する物だけは、利用者の文字が入る余地が無いのでそのまま送る（#391） 記録の文は
#: 作品の中の名前や debug_print の文や場所を持ちうるので、ほかは送らない設定なら伏せる
#: 一致を見るので、ここへ足す名前は固定の文字だけにする（作品やスクリプトから組み立てない）
AVIUTL_OBJ_NAMES: frozenset[str] = frozenset(
    {
        # 関数
        "mes",
        "effect",
        "draw",
        "drawpoly",
        "load",
        "setfont",
        "getfont",
        "rand",
        "setoption",
        "getoption",
        "getvalue",
        "setanchor",
        "getaudio",
        "filter",
        "copybuffer",
        "getpixel",
        "putpixel",
        "copypixel",
        "pixeloption",
        "getpixeldata",
        "putpixeldata",
        "getinfo",
        "interpolation",
        "getpoint",
        "module",
        "pixelshader",
        "computeshader",
        "clearbuffer",
        "multiobject",
        "text",
        # 変えられる値
        "ox",
        "oy",
        "oz",
        "rx",
        "ry",
        "rz",
        "cx",
        "cy",
        "cz",
        "zoom",
        "alpha",
        "aspect",
        "sx",
        "sy",
        "sz",
        # 読むだけの値
        "x",
        "y",
        "z",
        "w",
        "h",
        "screen_w",
        "screen_h",
        "framerate",
        "frame",
        "time",
        "totalframe",
        "totaltime",
        "layer",
        "index",
        "num",
        "track0",
        "track1",
        "track2",
        "track3",
        "check0",
        "id",
        "effect_id",
        "objectindex",
    }
)
#: 記録の文のうち、決まった名前だけで組み立てた物（``obj.名前`` と ``obj.名前 への代入``
#: :mod:`~sashimono.compat.aviutl.objapi` が書く形）
KNOWN_RECORDS: frozenset[str] = frozenset(
    {f"obj.{name}" for name in AVIUTL_OBJ_NAMES}
    | {f"obj.{name} への代入" for name in AVIUTL_OBJ_NAMES}
)


def is_known_record(text: str) -> bool:
    """記録の文が、アプリ側の決まった名前とちょうど一致するか（利用者の文字が入らない）"""
    return text in KNOWN_RECORDS


#: 記録する種類の上限と、1 件の長さの上限 名前はスクリプトが決められる
#: （``debug_print`` の中身、``obj.effect`` の名前など） ループで毎回違う名前を
#: 出されると、記録だけでメモリを使い切る
MAX_KINDS = 500
MAX_NAME = 200
#: 上限を超えた分をまとめて数える行
OVERFLOW = "（記録の種類が多すぎるため、ほかは省略）"


def _bounded(counter: Counter[str], name: str) -> str:
    name = name[:MAX_NAME]
    if name in counter or len(counter) < MAX_KINDS:
        return name
    return OVERFLOW


#: アプリ全体で 1 つ スクリプトはどこから走っても同じ場所へ記録する
global_report = CompatibilityReport()
