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

__all__ = ["ADD_ON_CHECK_FLAG", "IMPORT_CHECK_FLAG", "SELF_CHECK_FLAG", "main"]

#: 画面を出さずに、同梱した部品が動くかだけを確かめる
SELF_CHECK_FLAG = "--self-check"
#: 画面を出さずに、渡した置き場（``;`` 区切り）を足して部品を import してみる
#: 配布版を組み立てる道具が、後から入れる部品に要る標準ライブラリが揃っているかを見る
IMPORT_CHECK_FLAG = "--import-check"
#: 画面を出さずに、後から入れる部品を導入ボタンと同じ入れ方で渡した置き場へ入れ、読んでみる
#: 配る zip の確かめが使う（:mod:`sashimono.addon_check`） ネットにつなぐ
ADD_ON_CHECK_FLAG = "--add-on-check"


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


def _add_on_check(place: str, rest: list[str]) -> int:
    """``--add-on-check <置き場> [<子を 1 つ待つ秒数>]`` 秒数が読めなければ 2 で終わる

    読めない秒数を既定に置き換えて走らせると、CI の待ちに収まらない長さで待ち、要約を
    書く前にジョブが取り消される
    """
    from sashimono.addon_check import main as add_on_check

    if not rest:
        return add_on_check(place)
    try:
        seconds = float(rest[0])
    except ValueError:
        seconds = 0.0
    # nan と inf も断る どちらも比べると外れる
    if not 0 < seconds < float("inf"):
        if sys.stderr is not None:
            print(f"{ADD_ON_CHECK_FLAG} の秒数が読めない: {rest[0]}", file=sys.stderr)
        return 2
    return add_on_check(place, seconds)


def is_python_script_start(arguments: list[str]) -> bool:
    """配布版の exe が ``Sashimono.exe 何か.py ...`` と、Python の台本を渡されて起こされたか

    pip はソースの形（sdist）から組むとき、``sys.executable`` に自分の起動部
    （``__pip-runner__.py``）や組み立ての係（``_in_process.py``）を渡して子を立てる
    配布版の ``sys.executable`` はこの exe なので、受けないと台本の場所をプロジェクトとして
    編集画面が裏で立ち、導入が終わらなくなる（0.1.0 の zip で起きた）
    開発の環境では ``sys.executable`` が本物の Python なので、ここへは来ない
    """
    from sashimono.runtime import is_frozen

    return is_frozen() and len(arguments) >= 2 and arguments[1].lower().endswith(".py")


def refuse_script(script: str) -> int:
    """台本は走らせずに、理由を出して 1 で終わる

    ``python 台本.py`` と同じに走らせることはしない 走らせても、配布版は環境変数の
    ``PYTHONPATH`` を読まない（PyInstaller が切り離している 0.1.0 の exe で確かめた）ので、
    pip が組み立て用に入れた部品を係が見つけられず、組み立ては結局通らない そのうえ
    exe へ落とした .py がそのまま動くことになる 失敗で終われば、pip は組めなかったと言って止まる
    導入ボタンは wheel だけを入れる（``--only-binary :all:``）ので、ふだんはここへ来ない
    """
    message = (
        f"Sashimono.exe は Python の台本を走らせない: {script}\n"
        "配布版はソースの形の部品を組めないので、wheel のある版を入れる（--only-binary :all:）"
    )
    # 窓の無い配布版をコマンドから出力を受けずに起こすと sys.stderr が無い
    if sys.stderr is not None:
        print(message, file=sys.stderr)
    return 1


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

    if list(arguments[1:2]) == [ADD_ON_CHECK_FLAG] and len(arguments) in (3, 4):
        return _add_on_check(arguments[2], arguments[3:])

    if is_python_script_start(arguments):
        return refuse_script(arguments[1])

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
    from sashimono.ui.theme import apply_theme
    from sashimono.ui.translation import install_qt_translation
    from sashimono.ui.updates import STARTUP_DELAY_MS
    from sashimono.ui.workspace import PreferenceStore
    from sashimono.update.swap import mark_started

    # サーフェス形式は QApplication を作る前に決めておく必要がある
    # 後から設定しても、ウィジェットのコンテキストには反映されない
    QSurfaceFormat.setDefaultFormat(preferred_surface_format())

    application = QApplication(arguments)
    application.setApplicationName("Sashimono")
    application.setWindowIcon(QIcon(str(path_to(ICON_FILE))))
    # 窓を作る前に当てる 部品は作った時点のテーマの色で組み立てられるので、後から
    # 当てると暗いテーマの色で一瞬描かれてから切り替わる
    apply_theme(application, PreferenceStore().load().theme)
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
