r"""アプリケーションの入口

.venv\Scripts\python.exe -m sashimono

**一番上では Qt を読まない** 読むのは、いつもの起動（:func:`_start_editor`）の中
配布版は同じ exe が 3 つの役をする（編集画面・自己診断・導入ボタンの pip）
一番上で Qt と編集画面を読むと、

- Qt の部品が欠けた配布版では、自己診断にたどり着く前に落ちる
  （結果を見せる窓も出ない 欠けたことを知りたいまさにそのときに何も出ない）
- 導入ボタンの pip も、Qt 一式を読み込んでから走ることになる
"""

from __future__ import annotations

import sys
from pathlib import Path

# この 2 つは Qt を読まない（読まないことを試験で押さえている）
from sashimono.asr import activate_runtime
from sashimono.core.userdirs import migrate_legacy_folders
from sashimono.runtime import pip_arguments, run_pip

__all__ = ["IMPORT_CHECK_FLAG", "SELF_CHECK_FLAG", "main"]

#: 画面を出さずに、同梱した部品が動くかだけを確かめる
SELF_CHECK_FLAG = "--self-check"
#: 画面を出さずに、渡した置き場（``;`` 区切り）を足して部品を import してみる
#: 配布版を組み立てる道具が、後から入れる部品に要る標準ライブラリが揃っているかを見る
IMPORT_CHECK_FLAG = "--import-check"


def import_check(places: str, modules: list[str]) -> int:
    """``places`` を探す道の前へ足して ``modules`` を import する 読めない物があれば 1

    前へ足す 配布版は、画面のボタンで入れた部品の置き場を前へ足して読む
    （:func:`~sashimono.runtime.activate_runtime`） 同じ順で読まないと、使う人の手元とは
    違う物を読んで確かめることになる 標準ライブラリは置き場に無いので、配布版の持ち物しか
    見えない
    """
    import importlib
    import os

    from sashimono.runtime import read_path_files

    for place in reversed(places.split(os.pathsep)):
        if place and place not in sys.path:
            sys.path.insert(0, place)
        if place:
            # 配布版が入れた置き場を読むのと同じく .pth も読む（pywin32 が要る）
            read_path_files(place)
    failed = 0
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed += 1
            print(f"[NG] {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"[ok] {name}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv if argv is None else argv

    # 配布版は自分自身が pip の代わりになる 導入ボタンは ``sys.executable -m pip`` を
    # 呼ぶが、配布版の sys.executable はこの exe なので、ここで受けないと
    # 導入するつもりで Sashimono がもう 1 つ起動する
    pip_args = pip_arguments(arguments)
    if pip_args is not None:
        return run_pip(pip_args)

    # 自己診断だけを頼まれたときに限る ほかの引数（プロジェクトの場所）と一緒に
    # 渡されたら、開くつもりの起動として扱う 自己診断を優先すると、頼んだ
    # プロジェクトが開かずに黙って終わる
    if len(arguments) >= 3 and arguments[1] == IMPORT_CHECK_FLAG:
        return import_check(arguments[2], arguments[3:])

    if list(arguments[1:]) == [SELF_CHECK_FLAG]:
        from sashimono.selfcheck import main as self_check

        return self_check()

    return _start_editor(arguments)


def _start_editor(arguments: list[str]) -> int:
    """いつもの起動 Qt と編集画面はここで初めて読む"""
    # 改名前の置き場（設定・退避・導入した実行環境）を引き継ぐ 何より先に行う
    # 設定を読んだあとでは、既定の設定で新しい置き場ができてしまい、引き継ぎが
    # 「もう在る」と見て何もしなくなる 実行環境の置き場も、次の行で探す前に移しておく
    # 自己診断と pip の役では行わない どちらも本人の置き場を使わないので、そこで
    # 移すと、画面を 1 度も出さないうちに旧版の置き場が消えることになる
    for note in migrate_legacy_folders():
        # 窓の無い配布版では誰も読まないが、コマンドから起動した人と開発者には手掛かりになる
        print(f"置き場の引き継ぎ: {note.action} {note.source} → {note.target} {note.detail}")

    # 次の起動で入れると決めた新しい版があれば、編集画面を出す前に入れ替え係へ任せて終わる
    # 入れ替え係が新しい版を起こし直す 画面を出してからだと、開いた作品を閉じさせることになる
    # 配布版だけ（開発の環境では置き場が無いので何もしない）
    from sashimono.update.flow import apply_on_start

    if apply_on_start(arguments):
        return 0

    # ソフト内から導入した字幕起こしの実行環境を import できるようにする
    # 通常の実行では何もしない（パッケージ版のためだけの手当て）
    activate_runtime()

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon, QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    from sashimono.core.io import ProjectFileError, load_project
    from sashimono.engine.gpu import preferred_surface_format
    from sashimono.resources import ICON_FILE, path_to
    from sashimono.ui.main_window import MainWindow
    from sashimono.ui.theme import STYLE_SHEET
    from sashimono.ui.translation import install_qt_translation
    from sashimono.ui.updates import STARTUP_DELAY_MS
    from sashimono.update.swap import mark_started

    # サーフェス形式は QApplication を作る前に決めておく必要がある
    # 後から設定しても、ウィジェットのコンテキストには反映されない
    QSurfaceFormat.setDefaultFormat(preferred_surface_format())

    application = QApplication(arguments)
    application.setApplicationName("Sashimono")
    application.setWindowIcon(QIcon(str(path_to(ICON_FILE))))
    application.setStyleSheet(STYLE_SHEET)
    # 窓を作る前に読む 後から読むと、先に作った部品の文言は英語のまま残る
    install_qt_translation(application)

    project = None
    path = None
    if len(arguments) > 1:
        try:
            project = load_project(Path(arguments[1]))
            path = Path(arguments[1])
        except ProjectFileError as exc:
            print(f"プロジェクトを開けない: {exc}", file=sys.stderr)

    window = MainWindow(project, path=path)
    window.show()
    # 窓を出せたことを入れ替え係へ知らせる 知らせないと、入れたばかりの版が起動できなかった
    # ものとして前の版へ戻される（入れ替え係が起こしたときだけ書く）
    QTimer.singleShot(0, mark_started)
    # 窓が描かれてから尋ねる 先に尋ねると、何のソフトの話かが分からない
    QTimer.singleShot(0, window.offer_recovery)
    # 新しい版を確かめるのは、起動が落ち着いてから 起動を待たせない
    QTimer.singleShot(STARTUP_DELAY_MS, window.start_updates)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
