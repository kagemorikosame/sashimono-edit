r"""同梱した部品が、この機械で本当に動くかを確かめる

    Sashimono.exe --self-check
    .venv\Scripts\python.exe -m sashimono --self-check

編集画面は出さない 1 項目 1 行で結果を出し、どれか 1 つでも動かなければ終了コード 1
結果は標準出力へ書く 標準出力が無いとき（窓だけの exe をパイプを付けずに起動した）は、
Windows のメッセージ窓で見せる

配った zip で一番起きやすい失敗は「組み立てたときに部品を積み忘れた」
（動的に読み込む Lua の実体、FFmpeg の DLL、Qt の GL プラグイン）
起動しただけでは分からない 描いて・書き出して・走らせて、初めて分かる
使う人の機械で確かめてもらうときも、この 1 行を打ってもらえば足りる

どの項目も**実際の道を通す** 読み込めるかだけを見ると、DLL が足りないのに
「読み込めた」と通ってしまう（読み込みは遅延で、呼んだ時点で初めて探しに行く）
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

__all__ = ["CheckResult", "format_results", "run_self_check"]

#: 書き出しを確かめる符号化器 どの機械にもある CPU のもの
CPU_CODEC = "libx264"

#: 見本を描く大きさ 小さくてよい 動くかどうかだけを見る
#: 偶数にする h.264 は幅も高さも偶数でないと符号化できない
SAMPLE_SIZE = 64

#: 書き出す先のフォルダの名前 日本語と空白を含める
#: 使う人の置き場（ユーザー名が日本語の人の一時フォルダ・書き出し先）はこういう名前になる
#: ASCII の名前でだけ確かめると、パスを狭い文字コードで渡す部品の積み違えを見逃す
JAPANESE_FOLDER = "確かめる 日本語"

#: GL を使わずに符号化する項目の名前 配る zip をまっさらな Windows で確かめる道具
#: （tools/check_clean_machine.ps1）が、GPU の無い機械でもこの項目は通ることを見る
ENCODE_CHECK_NAME = "FFmpeg で符号化（日本語のパス）"

#: 符号化して読み戻すコマ数
ENCODE_FRAMES = 3

#: Qt の日本語訳を読めたときの、取り消しのボタンの文言
JAPANESE_CANCEL = "キャンセル"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """1 項目の結果"""

    name: str
    ok: bool
    detail: str = ""
    #: 動かなくても困らない項目（音の出口が無い機械など） 合否に混ぜない
    optional: bool = False


def run_self_check() -> list[CheckResult]:
    """全項目を順に確かめる 途中で落ちても、残りは続ける

    1 つの失敗で止めると、ほかに何が足りないのかが分からないまま
    組み立て直しを繰り返すことになる
    """
    checks: list[tuple[str, Callable[[], str], bool]] = [
        ("版", _version, False),
        ("Qt", _qt, False),
        ("Qt の日本語訳", _qt_translation, False),
        ("編集画面を組み立てる", _editor, False),
        ("GL で描く", _render, False),
        (ENCODE_CHECK_NAME, _encode, False),
        ("書き出す（FFmpeg）", _export, False),
        ("AviUtl スクリプト（Lua）", _lua, False),
        ("音の出口", _sound, True),
        ("スクリプト置き場", _script_roots, False),
        ("同梱の絵（アイコン・ボタンの印）", _bundled_files, False),
        ("追加機能の導入（pip）", _pip, False),
        (UPDATE_CHECK_NAME, _update, False),
        # 最後に置く ほかの項目が Qt・FFmpeg・Lua の DLL を読み終えてから数えないと、
        # まだ読んでいない部品の分を見逃す
        (VC_RUNTIME_CHECK_NAME, _vc_runtime, False),
    ]
    results = []
    for name, check, optional in checks:
        try:
            detail = check()
        except Exception as exc:
            last = traceback.extract_tb(exc.__traceback__)[-1:]
            where = f"（{Path(last[0].filename).name}:{last[0].lineno}）" if last else ""
            results.append(
                CheckResult(name, False, f"{type(exc).__name__}: {exc}{where}", optional)
            )
            continue
        results.append(CheckResult(name, True, detail, optional))
    return results


def _bundled_files() -> str:
    """同梱の絵が揃っているか 積み忘れた物があれば名前を挙げて落とす"""
    from sashimono.resources import BUNDLED_FILES, path_to

    missing = [name for name in BUNDLED_FILES if not path_to(name).is_file()]
    if missing:
        raise FileNotFoundError("見つからない: " + "、".join(missing))
    return f"{len(BUNDLED_FILES)} 個"


def format_results(results: list[CheckResult]) -> str:
    """結果を人が読める形に"""
    lines = []
    for result in results:
        mark = "ok" if result.ok else ("--" if result.optional else "NG")
        lines.append(f"[{mark}] {result.name}: {result.detail}")
    failed = [result for result in results if not result.ok and not result.optional]
    lines.append("すべて動いた" if not failed else f"動かない項目が {len(failed)} 個")
    return "\n".join(lines)


def main() -> int:
    results = run_self_check()
    text = format_results(results)
    passed = all(result.ok or result.optional for result in results)
    if sys.stdout is None:
        # 窓だけの exe を、パイプを付けずに起動した（ダブルクリックや、コマンドで
        # そのまま打った） 標準出力が無いので、書いても誰にも届かない
        # 結果を窓で見せる パイプを付けたとき（``| more`` や組み立ての確認）は
        # 標準出力がある
        _show(text, passed)
    else:
        _make_writable(sys.stdout, text)
        print(text, flush=True)
    return 0 if passed else 1


def _make_writable(stream: TextIO, text: str) -> None:
    """``text`` を書けない文字コードの出口なら UTF-8 へ切り替える

    パイプへ書くときの文字コードは Windows の既定（日本語の Windows なら cp932）
    英語の Windows（cp1252 CI の Windows も同じ）では日本語を 1 文字も書けず、結果を
    出す前に ``UnicodeEncodeError`` で落ちる 結果は何も出ないまま終了コードだけが 1 になり、
    どの部品が動かないのかを知る手段が無くなる UTF-8 ならファイルへ受けて読める
    書ける出口（日本語の Windows）は変えない ``| more`` で読む人の画面が化ける
    """
    encoding = stream.encoding or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


#: メッセージ窓の題
MESSAGE_TITLE = "Sashimono Edit の自己診断"


def _show(text: str, passed: bool) -> None:
    """結果を窓で見せる

    **Qt には頼らない** Qt の項目が落ちた（Qt の DLL や窓の部品を積み忘れた）ときこそ
    結果を見せたいのに、Qt の窓で出そうとすると何も出ない Windows 自身の
    メッセージ窓（user32 の MessageBoxW）を使う
    """
    _native_message(text, MESSAGE_TITLE, warning=not passed)


def _native_message(text: str, title: str, *, warning: bool) -> None:
    """Windows のメッセージ窓 Windows 以外では何もしない（配るのは Windows 版だけ）"""
    if sys.platform != "win32":
        return
    import ctypes

    # 0x40 は情報、0x30 は警告の印 落ちた項目があるときは目立つ方で出す
    icon = 0x30 if warning else 0x40
    ctypes.windll.user32.MessageBoxW(None, text, title, icon)


def _version() -> str:
    from sashimono import __version__

    frozen = "配布版" if getattr(sys, "frozen", False) else "開発環境"
    return f"{__version__}（{frozen} Python {sys.version.split()[0]}）"


#: 自己診断のあいだ生かしておく Qt のアプリケーション
#: 作っただけで持っておかないと、Python の側の参照が切れた時点で壊され、
#: 後の項目（編集画面・GL）がアプリケーション無しで動くことになる
_application: object = None


def _qt() -> str:
    from PySide6.QtCore import qVersion
    from PySide6.QtWidgets import QApplication

    global _application
    # 画面は出さないが、GL のコンテキストは QGuiApplication が無いと作れない
    if QApplication.instance() is None:
        _application = QApplication([sys.argv[0] if sys.argv else "sashimono"])
    return f"Qt {qVersion()}"


def _qt_translation() -> str:
    """Qt 標準の文言の日本語訳を積んだか

    積み忘れても起動はするが、確認の窓のボタンが Save / Discard / Cancel と英語で出る
    起動しただけでは気付きにくいので、ここで落とす
    """
    from PySide6.QtWidgets import QApplication, QMessageBox

    from sashimono.ui.translation import QT_TRANSLATION, install_qt_translation

    application = QApplication.instance()
    if application is None:
        raise RuntimeError("Qt のアプリケーションが無い（Qt の項目が落ちている）")
    folder = install_qt_translation(application)
    if folder is None:
        raise RuntimeError(f"{QT_TRANSLATION}.qm が見つからない")
    box = QMessageBox()
    box.setStandardButtons(QMessageBox.StandardButton.Cancel)
    label = box.button(QMessageBox.StandardButton.Cancel).text()
    # 英語でないことではなく日本語であることを見る 別の言語の翻訳を積み違えても
    # 「英語ではない」で通ってしまう
    if label != JAPANESE_CANCEL:
        raise RuntimeError(f"{folder} の翻訳を読んだが、ボタンが日本語にならない（{label}）")
    return label


#: 編集画面が読み書きする、本人の置き場 確かめる間だけ一時フォルダへ向ける
USER_FOLDER_VARIABLES = ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_STATE_HOME")


@contextmanager
def _isolated_user_folders() -> Iterator[Path]:
    """本人の設定・退避・キャッシュの置き場を、確かめる間だけ一時フォルダへ向ける

    編集画面は閉じるときに**画面の並びを保存する** 見せていない窓を閉じると、
    本人が整えた並びを既定の並びで上書きしてしまう 退避と二重起動の印も同じ
    """
    saved = {name: os.environ.get(name) for name in USER_FOLDER_VARIABLES}
    with tempfile.TemporaryDirectory(prefix="sashimono-check-home-") as folder:
        for name in USER_FOLDER_VARIABLES:
            os.environ[name] = str(Path(folder) / name.lower())
        try:
            yield Path(folder)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _editor() -> str:
    """編集画面を、見せずに組み立てて閉じる

    部品を 1 つずつ動かす項目だけでは、**画面の側だけが読み込む部品**の
    積み忘れを見逃す（自己診断は通るのに、起動すると落ちる）
    """
    from PySide6.QtWidgets import QWidget

    from sashimono.core.model import Project
    from sashimono.ui.main_window import MainWindow

    with _isolated_user_folders():
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            parts = len(window.findChildren(QWidget))
        finally:
            window.close()
    return f"部品 {parts} 個"


def _sample_project() -> object:
    """文字と図形を 1 つずつ置いた見本 2 コマ

    文字を入れるのは、書体の読み込み（Qt のフォントの仕組み）まで通すため
    図形だけだとシェーダしか通らない
    """
    from sashimono.core.commands import AddClip, AddTrack
    from sashimono.core.model import (
        Clip,
        GeneratedSource,
        Project,
        ProjectSettings,
        Track,
        TrackKind,
    )
    from sashimono.core.timebase import FrameRate
    from sashimono.effects.sources import TEXT

    settings = ProjectSettings(width=SAMPLE_SIZE, height=SAMPLE_SIZE, frame_rate=FrameRate(30))
    project = Project.create(settings)
    below, above = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2")
    project = AddTrack(below).apply(project)
    project = AddTrack(above).apply(project)
    shape = GeneratedSource(
        kind="shape", params={"shape": "background", "color": (0.2, 0.4, 0.8, 1.0)}
    )
    project = AddClip(below.id, Clip(timeline_start=0, duration=2, source=shape)).apply(project)
    text = TEXT.create(text="木", size=32)
    return AddClip(above.id, Clip(timeline_start=0, duration=2, source=text)).apply(project)


def _render() -> str:
    import numpy as np

    from sashimono.core.model import Project
    from sashimono.engine.render import FrameRenderer

    project = _sample_project()
    assert isinstance(project, Project)
    renderer = FrameRenderer(project)
    try:
        image = renderer.render(0)
    finally:
        renderer.close()
    # 背景の青が出ていること 真っ黒なら、GL は動いたふりをして何も描いていない
    blue = int(np.median(image[..., 2]))
    if blue < 100:
        raise RuntimeError(f"描いた絵が暗すぎる（青 {blue}） シェーダが動いていない")
    return f"{image.shape[1]}x{image.shape[0]} を描けた"


def _encode() -> str:
    """GL を通さずに FFmpeg で符号化し、読み戻す 置き場は日本語と空白を含むフォルダ

    書き出す項目は合成に GL を使うので、GPU の無い機械では FFmpeg まで届かずに落ちる
    それでは FFmpeg の DLL を積み忘れたのか GL が無いだけなのかを分けられない
    FFmpeg だけをここで通しておけば、GL の無い機械でも符号化の部品は確かめられる
    """
    import av
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="sashimono-check-") as folder:
        target = Path(folder) / JAPANESE_FOLDER / "符号化.mp4"
        target.parent.mkdir()
        with av.open(str(target), mode="w") as container:
            # add_stream は種類の union を返す 映像として扱うので型を確定させる
            stream = cast("av.video.stream.VideoStream", container.add_stream(CPU_CODEC, rate=30))
            stream.width = SAMPLE_SIZE
            stream.height = SAMPLE_SIZE
            stream.pix_fmt = "yuv420p"
            for index in range(ENCODE_FRAMES):
                # コマごとに明るさを変える 同じ絵が続くと符号化器が中身を省き、
                # 読み戻す側の確かめが緩くなる
                image = np.full((SAMPLE_SIZE, SAMPLE_SIZE, 3), 60 * index, dtype=np.uint8)
                frame = av.video.frame.VideoFrame.from_ndarray(image, format="rgb24")
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        with av.open(str(target)) as container:
            frames = sum(1 for _ in container.decode(video=0))
    if frames != ENCODE_FRAMES:
        raise RuntimeError(f"符号化した動画が {frames} コマ（{ENCODE_FRAMES} コマのはず）")
    return f"{CPU_CODEC} で {frames} コマ書いて読み戻せた（{JAPANESE_FOLDER}）"


def _export() -> str:
    import av

    from sashimono.core.model import Project
    from sashimono.engine.encode import ExportSettings, available_video_codecs, export_project

    codecs = available_video_codecs()
    if not codecs:
        raise RuntimeError("使える映像コーデックが 1 つも無い（FFmpeg を積み忘れている）")
    project = _sample_project()
    assert isinstance(project, Project)
    with tempfile.TemporaryDirectory(prefix="sashimono-check-") as folder:
        target = Path(folder) / JAPANESE_FOLDER / "書き出し.mp4"
        target.parent.mkdir()
        # 実際の書き出しと同じ道を通す 符号化だけを試すと、書き出しが使う
        # 合成・音の混ぜ・mux のどこかで足りないものを見逃す
        # 符号化器は CPU のものを使う 機械ごとに有無が違う GPU の符号化器で
        # 確かめると、同じ zip が機械によって通ったり落ちたりする
        # CPU の符号化器は libx264 しかない（VIDEO_CODEC_PREFERENCE） 無ければ、
        # GPU の符号化器が無い機械で書き出せないということなので、落とす
        if CPU_CODEC not in codecs:
            raise RuntimeError(
                f"CPU の符号化器 {CPU_CODEC} が無い（GPU の符号化器が無い機械で書き出せない）"
            )
        export_project(project, ExportSettings(path=target, video_codec=CPU_CODEC))
        with av.open(str(target)) as container:
            frames = sum(1 for _ in container.decode(video=0))
    # 見本の長さと同じだけ読めること 1 コマでも読めれば通す形だと、
    # 書き出しが黙って途中で切れても気づけない
    if frames != project.duration:
        raise RuntimeError(f"書き出した動画が {frames} コマ（{project.duration} コマのはず）")
    return (
        f"{CPU_CODEC} で書き出して {frames} コマ読み戻せた（使える符号化器: {', '.join(codecs)}）"
    )


def _lua() -> str:
    from sashimono.compat.aviutl.objapi import ObjectState
    from sashimono.compat.aviutl.runtime import LuaScriptRuntime, blank_image

    runtime = LuaScriptRuntime()
    state = ObjectState(image=blank_image(8, 8))
    # 値を書き換えるだけの 1 行 Lua の実体（lupa の中の DLL）が無いと、
    # ランタイムを作る所か、ここで落ちる
    result = runtime.run("obj.ox = 1 + 2", state, script="動作確認")
    if result.failed:
        raise RuntimeError(result.message)
    if state.ox != 3:
        raise RuntimeError(f"計算が合わない（{state.ox}）")
    return "スクリプトを走らせられた"


def _sound() -> str:
    import sounddevice

    # 出口が無い機械（音の装置を外した PC、リモート接続）でも編集はできる
    # 合否には混ぜないが、PortAudio の DLL が無いのは import の時点で分かる
    device = sounddevice.query_devices(kind="output")
    return f"{device['name']}"


def _script_roots() -> str:
    """見に行く場所と、**実際に読めた本数**

    場所を並べるだけでは、置いたのに読まれない（拡張子の見落とし・文字コード）を
    見逃す 置き場ごとに走査して数える
    """
    from sashimono.compat.aviutl.catalog import ScriptCatalog, default_script_roots
    from sashimono.compat.aviutl.report import CompatibilityReport

    roots = default_script_roots()
    if not roots:
        return "（見に行く場所が無い）"
    parts = []
    for root in roots:
        if not root.exists():
            parts.append(f"{root}（無い）")
            continue
        # 読めなかった記録を本体の記録へ混ぜない 自己診断で動いた跡が残ると、
        # 互換の穴の数え方が狂う
        found = ScriptCatalog((root,), report=CompatibilityReport()).scan()
        parts.append(f"{root}（{len(found)} 本）")
    return " / ".join(parts)


def _pip() -> str:
    """導入ボタンが使う pip が、この実行環境の中にあるか

    配布版には Python の本体が無いので、pip を積み忘れると導入ボタンが黙って動かない
    """
    from pip import __version__ as pip_version

    return f"pip {pip_version}"


#: 自動更新の項目の名前 配る zip を組み立てる道具が、この項目が通ったかを見る
UPDATE_CHECK_NAME = "自動更新"


def _update() -> str:
    """更新の部品（署名・照合・展開・入れ替え係）が、この実行環境で動くか

    ネットワークには出ない その場で作った使い捨ての鍵と見本のリリースで通す
    埋め込んだ公開鍵の数も出す 0 本の版は更新を確かめない（配る前に気付けるように）
    """
    from sashimono.update.rehearsal import rehearse
    from sashimono.update.signing import TRUSTED_PUBLIC_KEYS, trusted_keys

    with tempfile.TemporaryDirectory(prefix="sashimono-check-update-") as folder:
        detail = rehearse(Path(folder))
    keys = len(trusted_keys())
    if keys != len(TRUSTED_PUBLIC_KEYS):
        raise RuntimeError("埋め込んだ公開鍵に読めない物がある（貼り間違い）")
    suffix = f"公開鍵 {keys} 本" if keys else "公開鍵が入っていない（この版は更新を確かめない）"
    return f"{detail} {suffix}"


#: Visual C++ の実行時の部品の項目の名前
VC_RUNTIME_CHECK_NAME = "Visual C++ の実行時の部品"

#: Visual C++ の実行時の部品の名前の頭（小文字） 再頒布可能パッケージが System32 へ入れる物
#: ucrtbase と api-ms-win-crt-* は Windows 10 から Windows の一部なので数えない
VC_RUNTIME_PREFIXES = ("vcruntime140", "msvcp140", "concrt140", "vcomp140")


def _vc_runtime() -> str:
    """読み込んだ Visual C++ の実行時の部品が、配布版の中から来ているか

    開発機にも CI の Windows にも再頒布可能パッケージが入っていて、System32 に同じ名前の
    DLL がある zip に積み忘れていても、そちらを読んで動いてしまう 入っていない機械では
    起動の時点で「VCRUNTIME140.dll が見つからない」と出て、自己診断にすらたどり着かない
    読み込んだ部品の場所を 1 つずつ見て、配布版に無い物を Windows 側から借りていれば落とす
    """
    if not getattr(sys, "frozen", False):
        return "開発環境では見ない（配布版で確かめる）"
    return vc_runtime_report(loaded_modules(), Path(sys.executable).parent)


def vc_runtime_report(loaded: Iterable[Path], bundle: Path) -> str:
    """``loaded``（読み込んだ DLL）のうち Visual C++ の実行時の部品を、``bundle`` と照らす

    配布版の中から読んだ物は良し Windows 側から読んでいても、同じ名前の物を配布版が
    持っていれば、入っていない機械ではそちらが読まれるので良し どちらでもない物があれば落とす
    """
    runtime = [path for path in loaded if path.name.lower().startswith(VC_RUNTIME_PREFIXES)]
    if not runtime:
        # 配布版の Python 自身が vcruntime140.dll を読む 1 つも無いのは数え方が壊れている
        raise RuntimeError("読み込んだ部品を数えられない")
    root = _normalized(bundle)
    own = {path.name.lower() for path in bundle.rglob("*.dll")}
    inside = [path for path in runtime if _normalized(path).startswith(root + os.sep)]
    borrowed = [path for path in runtime if path not in inside]
    missing = [path for path in borrowed if path.name.lower() not in own]
    if missing:
        raise RuntimeError(
            "配布版に無い部品を Windows 側から読んでいる（再頒布可能パッケージの無い機械では"
            "起動しない）: " + "、".join(str(path) for path in missing)
        )
    detail = f"配布版の中の {len(inside)} 個を読んだ"
    if borrowed:
        names = "、".join(sorted({path.name for path in borrowed}))
        detail += f" Windows 側の物を先に読んだ物（配布版にもある）: {names}"
    return detail


def _normalized(path: Path) -> str:
    # 大文字小文字と区切りを揃える Windows が返す綴りと sys.executable の綴りは揺れる
    return os.path.normcase(os.path.normpath(str(path)))


def loaded_modules() -> list[Path]:
    """このプロセスが読み込んでいる DLL の場所 Windows 以外では空"""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.K32EnumProcessModules.restype = wintypes.BOOL
    kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetModuleFileNameW.restype = wintypes.DWORD

    process = kernel.GetCurrentProcess()
    count = 1024
    while True:
        handles = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not kernel.K32EnumProcessModules(
            process, handles, ctypes.sizeof(handles), ctypes.byref(needed)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        # 読み込んだ数が枠を超えたら、枠を広げて数え直す 切れた所までで返すと、
        # 後ろに並んだ部品を見ないまま「良し」と言う
        if needed.value <= ctypes.sizeof(handles):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 64
    paths = []
    buffer = ctypes.create_unicode_buffer(32768)
    for handle in handles[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        if kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
            paths.append(Path(buffer.value))
    return paths


if __name__ == "__main__":  # pragma: no cover - 入口は sashimono.app
    raise SystemExit(main())
