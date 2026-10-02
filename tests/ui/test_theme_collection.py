"""テーマを切り替えて全部の部品へ配っている最中に、ごみ集めで部品が壊れても落ちないか

Qt はスタイルシートを当て直すと全部の部品へ知らせを配り、その途中で Python の受け手
（設定パネルの数値欄の eventFilter など）が動く そこで閾値を越えてごみ集めが走ると、
輪になって捨てられた Python 持ちの部品がその場で壊れ、Qt は壊れた部品へ配り続けて
access violation で落ちる（PR #236 の CI の 3.12） 配っている間はごみ集めを止める

別のプロセスで走らせる 落ちるときは例外ではなくプロセスが消えるので、同じプロセスで
走らせると残りの試験の結果まで失われる
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

#: 捨てた部品（自分を指す輪があるのでごみ集めまで残る）に、Python の eventFilter を
#: 持つ子を入れ、テーマを 3 回当てる 当てる回ごとに部品を捨て直し、閾値を戻してから当てる
#: 配っている最中に届いた 200 回目の知らせで閾値を 1 にして、次に物を作った所でごみ集めが
#: 走るようにする（本物では閾値を越えた所で走る 走る所を決めるため）
#:
#: 「配っている間」の始まりは ``_activate`` に入った所、終わりはごみ集めを動ける状態へ
#: 戻した所（``gc.enable``） 戻す前に走ったごみ集めの始まりを ``gc.callbacks`` で直に数え、
#: 当てた回ごとに、終わった時点の値を書き出す 知らせの受け取りの途中で数えると、最後の
#: 知らせの後に始まったごみ集めを見逃す ごみ集めを止める作りが無ければ、閾値を下げた
#: 直後に配っている最中のごみ集めが始まって数に出る（落ちなければ）
_SCRIPT = """
import gc
import json
import sys

from PySide6.QtWidgets import QApplication, QSlider, QWidget

from sashimono.ui import theme
from sashimono.ui.theme import THEME_DARK, THEME_LIGHT, apply_theme

application = QApplication(sys.argv[:1])
handing_out = False
calls = 0
enabled = set()
starts = 0


def count(phase, info):
    global starts
    if handing_out and phase == "start":
        starts += 1


gc.callbacks.append(count)
real_enable = gc.enable


def enable():
    # ごみ集めを動ける状態へ戻した所が、配り終えた所 戻してから走った分は数えない
    global handing_out
    handing_out = False
    real_enable()


gc.enable = enable
real_activate = theme._activate


def activate(app, name):
    global handing_out
    handing_out = True
    try:
        real_activate(app, name)
    finally:
        handing_out = False


theme._activate = activate


class Row(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._slider = QSlider(self)
        self._slider.installEventFilter(self)
        self.seen = []

    def eventFilter(self, watched, event):
        global calls
        if handing_out:
            calls += 1
            enabled.add(gc.isenabled())
            if calls == 200:
                gc.set_threshold(1, 1, 1)
        self.seen = [event.type(), [0] * 8]
        return False


class Dropped(QWidget):
    def __init__(self):
        super().__init__()
        self.me = self
        self.rows = [Row(self) for _ in range(10)]


rounds = []
for mode in (THEME_LIGHT, THEME_DARK, THEME_LIGHT):
    gc.disable()
    for _ in range(300):
        Dropped()
    gc.set_threshold(1_000_000, 100, 100)
    calls, starts = 0, 0
    real_enable()
    apply_theme(application, mode)
    rounds.append({"calls": calls, "starts": starts})
gc.disable()
print(json.dumps({"rounds": rounds, "enabled": sorted(enabled)}))
"""

ROOT = Path(__file__).resolve().parents[2]


def _child_environment(tmp_path: Path) -> dict[str, str]:
    """子のプロセスへ渡す環境 test_window_teardown と同じ作りにする

    Qt の描き先は、この試験を走らせている Qt と同じ物を渡す 画面の無い所で走らせて
    いても（offscreen）、子だけ画面を探して QApplication を作れずに落ちることがない
    ``PROGRAMDATA`` は子で差し替える 子には conftest の見張りが無く、既定の置き場を
    読みに行くと本人の置き場に触れうる（#135）
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [environment.get("PYTHONPATH")])]
    )
    environment["QT_QPA_PLATFORM"] = QGuiApplication.platformName()
    program_data = tmp_path / "programdata"
    program_data.mkdir()
    environment["PROGRAMDATA"] = str(program_data)
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def test_switching_survives_a_collection_while_it_hands_out_the_change(
    qt_application: QApplication, tmp_path: Path
) -> None:
    del qt_application
    done = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_environment(tmp_path),
        timeout=300,
        check=False,
    )
    detail = f"終了コード {done.returncode:#x}\n{done.stdout}\n{done.stderr[-2000:]}"
    assert done.returncode == 0, detail
    result = json.loads(done.stdout.strip().splitlines()[-1])
    # 狙った道を通ったこと どの回も、配っている間に閾値を下げる 200 回目まで知らせが届いた
    # 届かなければ、ごみ集めを走らせる所まで行っておらず、落ちないのは当たり前で何も言えない
    assert len(result["rounds"]) == 3, detail
    assert all(round_["calls"] >= 200 for round_ in result["rounds"]), detail
    # 配っている間はごみ集めが止まっていて、閾値を下げても配り終えるまで 1 度も始まらなかった
    # （配り終えて動ける状態へ戻した後に、溜まった分が片付くのはよい）
    assert result["enabled"] == [False], detail
    assert [round_["starts"] for round_ in result["rounds"]] == [0, 0, 0], detail
