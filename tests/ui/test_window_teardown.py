"""アプリと同じ手順で窓を作って閉じ、ごみ集めで壊しても落ちないか（#149）

#145 の CI（GPU の無い Windows）で、``MainWindow`` を閉じて参照を捨てた後にごみ集めが
走ると、プロセスごと落ちた（access violation） 一時の枝で絞ると、スタイルシートを窓と
アプリのどちらへ当てても、当てなくても落ちた 落ちるのは、GL のプレビューを出した窓を
閉じた後のごみ集めか、Python の終わりの片付けで、プレビューやドックを先に壊す・窓を
その場で壊すなど片付けの順を変えても直らなかった 落ちなかったのは、GL のプレビューを
窓に入れなかった窓だけ そこで OpenGL 4.3 を使えない機械では、プレビューを窓に入れず
案内の文を置く（:func:`~sashimono.engine.gpu.opengl_usable`）

別のプロセスで走らせる 落ちるときは例外ではなくプロセスが消えるので、同じ
プロセスで走らせると残りの試験の結果まで失われ、どこで落ちたのかも分からない
GPU のある機械では、この試験はプレビューを出す側を通る
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtGui import QOpenGLContext
from PySide6.QtWidgets import QApplication, QLabel

from sashimono.core.model import Project
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow

#: アプリの起動（``sashimono.app.main``）と同じ順で組み立てる 窓の作り直しを何度か
#: 繰り返し、閉じた窓をごみ集めで壊す 最後はイベントループから戻った後に壊し、
#: そのまま Python を終わらせる（アプリを閉じたときと同じ）
_SCRIPT = """
import gc
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Project
from sashimono.engine.gpu import preferred_surface_format
from sashimono.ui.main_window import MainWindow
from sashimono.ui.theme import style_sheet
from sashimono.ui.translation import install_qt_translation

QSurfaceFormat.setDefaultFormat(preferred_surface_format())
application = QApplication(sys.argv[:1])
application.setStyleSheet(style_sheet())
install_qt_translation(application)

for _ in range(3):
    window = MainWindow(Project.create(), confirm_unsaved=False)
    window.show()
    application.processEvents()
    window.close()
    application.processEvents()
    del window
    gc.collect()
    application.processEvents()
print("closed and collected", flush=True)

window = MainWindow(Project.create(), confirm_unsaved=False)
window.show()
QTimer.singleShot(200, window.close)
application.exec()
del window
gc.collect()
print("after the event loop", flush=True)
"""


def _isolated_environment(tmp_path: Path) -> dict[str, str]:
    """子のプロセスへ渡す環境 本人の置き場へは触らせない

    設定と退避の置き場（``APPDATA`` ``LOCALAPPDATA``）は conftest が一時フォルダへ
    向けてあり、そのまま引き継がれる ``PROGRAMDATA`` は子で差し替える 子には
    conftest の見張りが無く、AviUtl2 の汎用プラグインやスクリプトの既定の置き場を
    読みに行くと、本人の置き場が書き換わりうる（#135）
    落ちたときに Python の呼び出しの跡が読めるよう、文字の符号も決めておく
    """
    environment = dict(os.environ)
    program_data = tmp_path / "programdata"
    program_data.mkdir()
    environment["PROGRAMDATA"] = str(program_data)
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def test_the_app_window_survives_close_and_garbage_collection(tmp_path: Path) -> None:
    # 落ちると、GPU の無い機械でアプリを閉じるたびにプロセスが異常終了で消える
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=_isolated_environment(tmp_path),
        check=False,
    )
    detail = f"終了コード {completed.returncode:#x}\n{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, detail
    assert "closed and collected" in completed.stdout, detail
    assert "after the event loop" in completed.stdout, detail


class TestWithoutOpenGL:
    """OpenGL 4.3 を使えない機械の窓 GPU のある機械でも、使えないことにして確かめる"""

    @pytest.fixture
    def window(self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> MainWindow:
        del qt_application
        monkeypatch.setattr(main_window_module, "opengl_usable", lambda: False)
        return MainWindow(Project.create(), confirm_unsaved=False)

    def test_the_preview_is_never_shown(self, window: MainWindow) -> None:
        # 出すと窓ごと GL で描くようになり、閉じた後の片付けでプロセスごと落ちる
        # 隠すだけでは足りない Qt は GL の部品が子にいるだけで窓を GL で描く
        try:
            window.show()
            QApplication.processEvents()
            assert window._preview.parent() is None
            assert not window._preview.isVisible()
            assert QOpenGLContext.currentContext() is None
        finally:
            window.close()

    def test_the_place_of_the_preview_says_why(self, window: MainWindow) -> None:
        # 黙って黒いままにすると、壊れたのか何も置いていないのか見分けが付かない
        try:
            notices = [label for label in window.findChildren(QLabel) if "OpenGL" in label.text()]
            assert notices
            assert "GPU" in notices[0].text()
        finally:
            window.close()
