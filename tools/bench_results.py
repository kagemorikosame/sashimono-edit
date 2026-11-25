"""測る道具（``tools/bench_*.py``）が ``--json`` で書く結果の形（#513）

代表の測り（``bench_seek.py`` ``bench_render.py`` ``bench_timeline.py``）は、画面へ出す表とは別に
この形の JSON を書く 版ごとに残して比べるのは ``tools/bench_history.py``

1 回の走りが 1 つの記録になる 中身は次の通り

- ``schema`` 形の名前と版 読む側は知らない版を読まない（中身の意味が変わっても気付けない）
- ``bench`` 測りの名前（``seek`` など）と ``conditions`` 測った条件（引数そのまま）
- ``machine`` 測った機械 CPU・コアの数・OS・GPU（GL を使う測りだけ）・Python
  機械の名前や利用者の名前は入れない（記録を人へ渡しても困らないように）
- ``status`` ``ok``（測れた）・``skipped``（ffmpeg や GPU が無く測らなかった）・``failed``
  （測ろうとして落ちた） 飛ばしたのと落ちたのを分けないと、GPU の無い機械で回した記録が
  「全部の項目が消えた」と読めてしまう
- ``items`` 項目ごとの標本（ミリ秒）と中央値・95 パーセンタイル 標本をそのまま残すのは、
  後でばらつきの見方を変えても読み直せるようにするため

**標準ライブラリだけで書く** 測る道具は PySide6 や PyAV を読むが、比べる道具はそれらの無い
所（CI の要約など）でも記録を読めるようにしておく
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 1 回の走りの記録の形 中身の意味を変えたら数を上げる（読む側が古い形と混ぜない）
SCHEMA = "sashimono-bench/1"

OK = "ok"
SKIPPED = "skipped"
FAILED = "failed"


def percentile95(samples: list[float]) -> float:
    """95 パーセンタイル 標本の外側へ外挿しない（inclusive）

    既定の exclusive は、標本が少ないと一番大きい値より外へ出た数を返す 測りは 20 回前後なので、
    実際には出ていない値を比べることになる（bench_render.py と同じ取り方）
    """
    if len(samples) < 2:
        return max(samples)
    return statistics.quantiles(samples, n=20, method="inclusive")[18]


def cpu_name() -> str:
    """CPU の名前 Windows はレジストリの名前（``platform.processor`` は型番の数字しか返さない）

    読めなければ ``platform.processor`` の値 機械を見分ける鍵に使うので、空にはしない
    """
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        if isinstance(value, str) and value.strip():
            return value.strip()
    except (ImportError, OSError):
        # Windows 以外か、レジストリが読めない 名前が粗くなるだけで測りは続ける
        pass
    return platform.processor() or platform.machine() or "不明"


def machine(gpu: str | None = None) -> dict[str, Any]:
    """測った機械 ``gpu`` は GL を使う測りだけが渡す（使わない測りの速さは GPU で変わらない）

    OS は名前と大きな版まで（``Windows 11``） 細かい版（ビルドの番号）まで鍵にすると、
    Windows の更新のたびに前の版と比べられなくなる 細かい版は ``os_build`` に分けて残す
    """
    return {
        "cpu": cpu_name(),
        "cores": os.cpu_count() or 0,
        "os": f"{platform.system()} {platform.release()}",
        "os_build": platform.version(),
        "gpu": gpu,
        "python": platform.python_version(),
    }


#: 機械を見分ける鍵 これが違う記録どうしは比べない（GPU や CPU が違えば速さが違って当然）
#: Python の版は入れない Python が上がって遅くなったのは、見つけたい退行の 1 つ
MACHINE_KEYS = ("cpu", "cores", "os", "gpu")


@dataclass
class BenchRecord:
    """1 回の走りの記録を組む 測りの道具は ``add`` で項目を足し、最後に ``write`` する"""

    bench: str
    conditions: dict[str, Any]
    gpu: str | None = None
    status: str = OK
    reason: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, samples: list[float], *, label: str = "") -> None:
        """項目を足す ``name`` は版をまたいで同じにする鍵（ASCII） ``label`` は人が読む名前

        標本が空なら落ちた扱いにする 空のまま中央値を出そうとすると、測れなかった項目が
        例外で記録ごと消える
        """
        if not samples:
            self.items.append({"name": name, "label": label or name, "status": FAILED})
            return
        self.items.append(
            {
                "name": name,
                "label": label or name,
                "status": OK,
                "unit": "ms",
                # どの項目も短いほど良い 比べる側が向きを取り違えないよう書いておく
                "better": "lower",
                "samples": [round(value, 4) for value in samples],
                "median": round(statistics.median(samples), 4),
                "p95": round(percentile95(samples), 4),
            }
        )

    def skip(self, reason: str) -> None:
        """測らなかった（ffmpeg や GPU が無い） 落ちたのとは分ける"""
        self.status = SKIPPED
        self.reason = reason

    def fail(self, reason: str) -> None:
        """測ろうとして落ちた 測れた所までの項目は残す"""
        self.status = FAILED
        self.reason = reason

    def data(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "bench": self.bench,
            "conditions": self.conditions,
            "machine": machine(self.gpu),
            "status": self.status,
            "reason": self.reason,
            "items": self.items,
        }

    def write(self, path: Path | None) -> None:
        """``path`` が無ければ何もしない（``--json`` を付けずに走らせたとき）

        書いてから名前を付ける 測りの途中で止めると、書きかけの JSON が前の記録の名前で残り、
        比べる道具が読めずに「落ちた」と出す
        """
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".writing")
        temporary.write_text(
            json.dumps(self.data(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


def read_record(path: Path) -> dict[str, Any]:
    """記録を読む 形の違う物・壊れた物は ``ValueError``

    読めない記録を空の記録として扱うと、落ちた測りが「項目が無い」と読めて、欠けと区別できない
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} を読めない: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{path.name} は {SCHEMA} の記録ではない")
    check_run(data, path.name)
    return data


def finite(value: Any) -> bool:
    """有限の数か（真偽値は数に数えない JSON の true が 1 として通るのを防ぐ）"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def check_run(data: Any, where: str) -> None:
    """1 回の走りの記録の形を見る 崩れていれば ``ValueError``

    壊れた項目を通すと、比べる道具が数える所で落ち、残した記録ごと読めなくなる
    版ごとの記録（bench_history）も、回ごとにこれで見る
    """
    if not isinstance(data, dict) or data.get("status") not in (OK, SKIPPED, FAILED):
        raise ValueError(f"{where} の中身が欠けている")
    if "items" not in data:
        # 記録を書けずに落ちた回（bench_history が理由だけを残す）
        if data.get("status") != FAILED:
            raise ValueError(f"{where} の中身が欠けている")
        return
    if not isinstance(data["items"], list) or not isinstance(data.get("machine"), dict):
        raise ValueError(f"{where} の中身が欠けている")
    for item in data["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"{where} に読めない項目がある")
        if item.get("status") != OK:
            continue
        samples = item.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{where} の {item['name']} に値が無い")
        # 数として使える有限の値だけを受け取る 桁あふれ（10**400 など）や NaN を通すと、数える所で
        # 落ちるか、中央や幅が NaN になり判定が黙って崩れる
        for value in (item.get("median"), item.get("p95", item.get("median")), *samples):
            if not finite(value):
                raise ValueError(f"{where} の {item['name']} に数として読めない値がある")
