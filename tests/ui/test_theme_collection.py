"""テーマを切り替えて全部の部品へ配っている最中に、ごみ集めで部品が壊れても落ちないか

Qt はスタイルシートを当て直すと全部の部品へ知らせを配り、その途中で Python の受け手
（設定パネルの数値欄の eventFilter など）が動く そこで閾値を越えてごみ集めが走ると、
輪になって捨てられた Python 持ちの部品がその場で壊れ、Qt は壊れた部品へ配り続けて
access violation で落ちる（PR #236 の CI の 3.12） 配っている間はごみ集めを止める

別のプロセスで走らせる 落ちるときは例外ではなくプロセスが消えるので、同じプロセスで
走らせると残りの試験の結果まで失われる
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: 捨てた部品（自分を指す輪があるのでごみ集めまで残る）に、Python の eventFilter を
#: 持つ子を入れる 配っている最中の 200 回目の受け取りで閾値を 1 にして、次に物を作った
#: 所でごみ集めが走るようにする（本物では閾値を越えた所で走る 走る所を決めるため）
_SCRIPT = """
import gc
import sys

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QSlider, QWidget

from sashimono.ui.theme import THEME_DARK, THEME_LIGHT, apply_theme

application = QApplication(sys.argv[:1])


class Row(QWidget):
    calls = 0

    def __init__(self, parent):
        super().__init__(parent)
        self._slider = QSlider(self)
        self._slider.installEventFilter(self)
        self.seen = []

    def eventFilter(self, watched, event):
        Row.calls += 1
        if Row.calls == 200:
            gc.set_threshold(1, 1, 1)
        self.seen = [event.type(), [0] * 8]
        return False


class Dropped(QWidget):
    def __init__(self):
        super().__init__()
        self.me = self
        self.rows = [Row(self) for _ in range(10)]


gc.disable()
for _ in range(300):
    Dropped()
gc.set_threshold(1_000_000, 100, 100)
gc.enable()
for mode in (THEME_LIGHT, THEME_DARK, THEME_LIGHT):
    apply_theme(application, mode)
gc.disable()
print("survived", flush=True)
"""

ROOT = Path(__file__).resolve().parents[2]


def test_switching_survives_a_collection_while_it_hands_out_the_change() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [environment.get("PYTHONPATH")])]
    )
    done = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "survived" in done.stdout
