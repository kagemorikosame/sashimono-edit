"""試験の窓を本人の画面へ出さない仕組み

conftest と、試験が子として起こす Python（自分で ``QApplication`` を作る台本）の両方から使う
子には conftest の見張りが届かないので、子の台本はここを読み込んで自分のアプリに掛ける
"""

from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QWidget

#: 試験の窓を本人の画面へ出すときに立てる環境変数（値は ``1``） 子のプロセスへもそのまま渡る
SHOW_WINDOWS_VARIABLE = "SASHIMONO_TEST_SHOW_WINDOWS"

#: 窓が表示中になる前に必ず届く知らせ 窓を作った（``WinIdChange``）と、見た目を整える
#: （``Polish``）の 2 つ どちらも ``isVisible`` が立つ前に配られる
#: 表示の知らせ（``Show``）では付けない その時点の窓は表示中で、Qt は属性が変わると
#: 窓を隠して出し直す その出し直しが、アプリ全体へスタイルシートを当てた直後の
#: ``QPushButton.show()`` で access violation になった（test_issue27_polish の補足の試験）
_BEFORE_SHOWING = frozenset({QEvent.Type.WinIdChange, QEvent.Type.Polish})


def windows_on_screen() -> bool:
    """試験の窓を本人の画面へ出すか

    既定は出さない 手元で走らせるたびに編集画面が数百回出ては消え、本人の作業の前へ
    割り込んでいた CI も既定のまま走らせる 手元と CI で形が違うと、手元で通った試験が
    CI でだけ落ちる 出すのは、Windows が窓を実際に映したときだけの振る舞いを、人が
    目で確かめたいときだけ
    """
    return os.environ.get(SHOW_WINDOWS_VARIABLE, "").strip() == "1"


class OffScreenWindows(QObject):
    """表示される前の窓に ``WA_DontShowOnScreen`` を付けて、画面へ出さない

    窓の作り方は試験ごとにばらばらで（編集画面・ダイアログ・メニュー・試験の中の
    ``QWidget``）、1 つずつ付けて回ると付け忘れた所だけがちらつく アプリ全体の知らせを
    見張り、どの窓にも表示中になる前に届く知らせ（:data:`_BEFORE_SHOWING`）で付ける
    Qt の上では表示中のまま（``isVisible``・大きさ・前面化・GL の描画）なので、試験が
    見る形は変わらない ただし OS は窓を映さないので、``QTest.qWaitForWindowExposed`` は
    来ない知らせを待って毎回 5 秒で切れる 描かせたいときは ``grab()`` を使う

    ``QT_QPA_PLATFORM=offscreen`` を使わないのは、Windows のオフスクリーンでは
    OpenGL を作れず、GPU の試験がまとめて飛ばされるため 書体の幅も変わる
    """

    def __init__(self, application: QApplication) -> None:
        super().__init__(application)
        application.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt の命名規約
        if (
            event.type() in _BEFORE_SHOWING
            and isinstance(watched, QWidget)
            and watched.isWindow()
            and not watched.isVisible()
            and not watched.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        ):
            watched.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        return False


def hide_windows(application: QApplication) -> OffScreenWindows | None:
    """``application`` の窓を画面へ出さないようにする 出す指定なら何もしない

    返した物は、見張りを外すとき（``removeEventFilter``）に使う 持たなくても、親の
    ``application`` が掴んでいるので消えない
    """
    return None if windows_on_screen() else OffScreenWindows(application)
