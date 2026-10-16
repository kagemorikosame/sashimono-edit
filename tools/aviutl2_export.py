r"""AviUtl2 に .aup2 を連番の PNG で書き出させる（本体の画面を操作する）

    .venv\Scripts\python.exe tools\aviutl2_export.py --project X.aup2 --output X_frames

AviUtl2 との突き合わせ（``tools/aviutl_compare.py``）は、AviUtl2 で書き出す所だけが
手作業だった これを本人の手を借りずに回すための道具（Issue #108） YMM4 の
``tools/ymm4_export.py`` と同じ形で、画面の操作は同梱の ``aviutl2_export.ps1`` が受け持つ

**動かしている間は AviUtl2 の窓が前に出る** 本人に断ってから走らせる AviUtl2 か YMM4 が
既に開いていれば、作業中の物を壊さないように何もせず止まる

書き出すのは連番の PNG（AviUtl2 に元から入っている〔連番ファイル出力〕） 動画の書き出しと
違って圧縮で色が痩せないので、文字の基準線のような 1 画素の違いまで測れる
書き出しの設定（PNG の透明色・JPEG の品質）は変えない 窓に出ていた設定を出力へ中継する

AviUtl2 は閉じるときに設定（``aviutl2.ini`` の保存の窓の置き場など）と開いた履歴
（``history.ini``）を書き直す 起動する前にこれらの控えを取り、閉じたあとに元の中身へ戻して
読み返して確かめる 戻せなかったときは大きく知らせ、終了コード 2 で終わる
自動控え（``Backup`` の ``AutoBackup_*.aup2``）は AviUtl2 自身の物なので触らず、増えた物を出す

書き出しは出力の隣の一時のフォルダ（``<出力>.sashimono-<16 進 8 桁>.part``）へ行い、枚数が
揃ったと確かめてから出力の名前へ変える 出力が既にあって空でなければ、書き出す前に断る
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ymm4_export import run_script, ymm4_running

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT = Path(__file__).with_name("aviutl2_export.ps1")
EXE_NAME = "aviutl2.exe"
#: 開いていたら止める相手 AviUtl2 のほか、同じ機械で比べる YMM4 も画面を取り合う
BLOCKING_IMAGES = (EXE_NAME, "YukkuriMovieMaker.exe")
#: AviUtl2 の場所を教える環境変数 引数より弱く、関連付けより強い
ENVIRONMENT_KEY = "AVIUTL2_EXE"
DEFAULT_TIMEOUT = 1800
#: PowerShell が固まったときに見切るまでの上乗せ（秒） 起動と閉じる待ちの分
WATCHDOG_MARGIN = 600
#: 設定を戻せなかったときの終了コード
RESTORE_FAILED = 2
#: AviUtl2 が閉じるときに書き直す、置き場の直下の設定
#: 2026-09-24 に v2.1.6a を開いて書き出して閉じると、aviutl2.ini（保存の窓の置き場が
#: 1 行増え、節の並びが入れ替わる）と history.ini（開いた履歴）と explorer.ini が書き直された
SETTING_FILES = ("aviutl2.ini", "history.ini", "explorer.ini")


def positive_seconds(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("1 秒以上を指定してください")
    return value


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, required=True, help="書き出す .aup2")
    parser.add_argument("--output", type=Path, required=True, help="PNG を並べるフォルダ")
    parser.add_argument(
        "--aviutl2",
        type=Path,
        default=None,
        help=f"aviutl2.exe の場所（既定は {ENVIRONMENT_KEY}・.aup2 の関連付けの順に探す）",
    )
    parser.add_argument(
        "--timeout",
        type=positive_seconds,
        default=DEFAULT_TIMEOUT,
        help=f"書き出しを待つ秒数（既定 {DEFAULT_TIMEOUT}）",
    )
    return parser.parse_args(argv)


def association_command() -> str | None:
    """.aup2 を開く命令 AviUtl2 の導入で関連付けたときだけある"""
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, ".aup2") as key:
            kind = str(winreg.QueryValue(key, None))
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, rf"{kind}\shell\open\command") as key:
            return str(winreg.QueryValue(key, None))
    except OSError:
        return None


def exe_from_command(command: str) -> Path | None:
    """関連付けの開く命令（``"J:\\…\\aviutl2.exe" "%1"``）から exe を取り出す"""
    command = command.strip()
    if command.startswith('"'):
        end = command.find('"', 1)
        head = command[1:end] if end > 0 else ""
    else:
        head = command.split(" ", 1)[0]
    return Path(head) if head.lower().endswith(".exe") else None


def find_aviutl2(
    explicit: Path | None,
    environ: Mapping[str, str],
    association: Callable[[], str | None] | None = None,
) -> tuple[Path | None, list[str]]:
    """AviUtl2 の exe と、見つからなかったときに案内する探した先

    名指しされた場所に無ければほかを探さない 黙って別の AviUtl2 を使うと、名指しした版と
    違う版で書き出した絵を突き合わせてしまう AviUtl2 は zip を好きな所へ展開して使うので、
    決まった置き場を当て推量で探すことはしない
    """
    if explicit is not None:
        return (explicit, []) if explicit.is_file() else (None, [f"--aviutl2 {explicit}"])
    named = environ.get(ENVIRONMENT_KEY)
    if named:
        path = Path(named)
        return (path, []) if path.is_file() else (None, [f"{ENVIRONMENT_KEY}={named}"])
    command = (association or association_command)()
    if command:
        found = exe_from_command(command)
        if found is not None and found.is_file():
            return found, []
        return None, [f".aup2 の関連付け {command}"]
    return None, [".aup2 の関連付け（登録なし）"]


def data_folder(exe: Path, environ: Mapping[str, str]) -> Path | None:
    """AviUtl2 が設定を書く置き場

    ``aviutl2.txt`` の決まり 本体の隣に ``data`` があればそこ、無ければ
    ``ProgramData\\aviutl2`` 違う方の控えを取ると、書き直された設定が戻らない
    """
    beside = exe.parent / "data"
    if beside.is_dir():
        return beside
    program_data = environ.get("PROGRAMDATA")
    return Path(program_data) / "aviutl2" if program_data else None


_SECTION = re.compile(r"^\[(\d+)\]$")


def project_frames(text: str) -> int:
    """書き出される枚数 オブジェクトの区間（``frame=始まり,終わり``）の終わりの最大 + 1

    AviUtl2 は最後のオブジェクトの終わりまでを書き出す（``tools/aviutl_compare.py`` が並べた
    354 フレームのプロジェクトで frame000〜frame353 が出た） 数え違うと、枚数が揃うのを
    待ち続けるか、書き終える前に止めてしまう オブジェクトの節（``[0]`` ``[1]`` …）の
    ``frame=`` だけを数え、中のフィルタの節（``[0.1]``）やシーンの節は見ない

    中間点のあるオブジェクトは ``frame=開始,中間点…,終了`` と同じ行に並ぶ（AviUtl2 に
    作らせた ``kumiki_motion_probe.object`` の ``frame=244,333,423``、読み手は
    ``compat/aviutl/exo.py``） 2 つ目だけを終わりとして読むと、中間点のある
    オブジェクトの終わりを数え落とし、書き出された PNG が見込みより多いとして失敗する
    """
    last = -1
    in_object = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_object = bool(_SECTION.match(line))
            continue
        if not in_object or not line.startswith("frame="):
            continue
        for part in line.removeprefix("frame=").split(","):
            try:
                last = max(last, int(part))
            except ValueError:
                continue
    return last + 1


@dataclass(frozen=True)
class SavedSetting:
    """控えを取った設定 1 つ ``content`` が ``None`` なら、控えを取ったときに無かった"""

    path: Path
    content: bytes | None


def save_settings(folder: Path) -> list[SavedSetting]:
    """AviUtl2 が閉じるときに書き直す設定の控え 中身を丸ごと覚える"""
    saved: list[SavedSetting] = []
    for name in SETTING_FILES:
        path = folder / name
        saved.append(SavedSetting(path, path.read_bytes() if path.is_file() else None))
    return saved


def restore_settings(saved: list[SavedSetting]) -> list[str]:
    """控えの中身へ戻し、読み返して確かめる 戻らなかった物の説明を返す（空なら全部戻った）

    戻したつもりで確かめずに終えると、本人の保存の窓の置き場や履歴が変わったまま残る
    """
    problems: list[str] = []
    for item in saved:
        try:
            if item.content is None:
                # 控えを取ったときに無かった物は、AviUtl2 が作った物なので取り除く
                if item.path.exists():
                    item.path.unlink()
            else:
                if not item.path.is_file() or item.path.read_bytes() != item.content:
                    item.path.write_bytes(item.content)
        except OSError as exc:
            problems.append(f"{item.path} を戻せない（{exc}）")
            continue
        now = item.path.read_bytes() if item.path.is_file() else None
        if now != item.content:
            problems.append(f"{item.path} を戻したが中身が違う")
    return problems


def digest(content: bytes | None) -> str:
    """戻したことを示す印 中身の SHA-256 の頭 無いファイルは ``無し``"""
    return "無し" if content is None else hashlib.sha256(content).hexdigest()[:16]


def partial_folder(output: Path, mark: Callable[[], str] | None = None) -> Path:
    """書き出している間のフォルダ 毎回違う印を入れ、既にある物と重ならない名前を選ぶ

    決まった名前だと、前の回に失敗して残った絵と今回の絵が混ざり、枚数が揃ったように見える
    """
    pick = mark or (lambda: uuid.uuid4().hex[:8])
    for _ in range(20):
        candidate = output.with_name(f"{output.name}.sashimono-{pick()}.part")
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"一時のフォルダの名前が既にある物と重ならずに選べない（{output.parent}）"
    )


def output_ready(output: Path) -> str | None:
    """出力へ書き出してよいか 断る理由を返す（``None`` ならよい）

    空でないフォルダへ入れ替えると前の書き出しを消すことになる 消さずに断る
    """
    if output.is_file():
        return f"{output} はファイルです 書き出し先はフォルダの名前にしてください"
    if output.is_dir() and any(output.iterdir()):
        return f"{output} は空ではありません 前の書き出しを退けてから走らせてください"
    return None


def publish(partial: Path, output: Path) -> None:
    """書き終えた一時のフォルダを出力の名前にする 出力は無いか空（:func:`output_ready`）"""
    if output.is_dir():
        output.rmdir()
    partial.rename(output)


def powershell_command(
    aviutl2: Path, project: Path, folder: Path, *, frames: int, timeout: int
) -> list[str]:
    """aviutl2_export.ps1 を呼ぶ命令

    ``-File`` で渡すと引数が 1 つずつ届く ``-ExecutionPolicy Bypass`` はこの 1 回だけに効き、
    本人の実行ポリシーの設定は変えない
    """
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-Aviutl2",
        str(aviutl2),
        "-Project",
        str(project),
        "-Folder",
        str(folder),
        "-Frames",
        str(frames),
        "-TimeoutSeconds",
        str(timeout),
    ]


def _alarm(lines: list[str]) -> None:
    bar = "!" * 60
    print(bar)
    for line in lines:
        print(f"!!! {line}")
    print(bar)


def _backups(folder: Path) -> set[str]:
    backup = folder / "Backup"
    return {path.name for path in backup.iterdir()} if backup.is_dir() else set()


def _refuse(arguments: argparse.Namespace) -> tuple[Path, Path, Path, Path, int] | None:
    """走らせる前の確かめ 走らせてよければ exe・プロジェクト・出力・置き場・枚数を返す"""
    if sys.platform != "win32":
        print("AviUtl2 は Windows でしか動かないので、この道具も Windows 専用です")
        return None
    project: Path = arguments.project.resolve()
    if not project.is_file() or project.suffix.lower() != ".aup2":
        print(f"{project} は .aup2 のファイルではありません")
        return None
    output: Path = arguments.output.resolve()
    reason = output_ready(output)
    if reason:
        print(reason)
        return None
    frames = project_frames(project.read_text(encoding="utf-8-sig", errors="replace"))
    if frames <= 0:
        print(f"{project} にオブジェクトが無いので、何も書き出されません")
        return None
    aviutl2, looked = find_aviutl2(arguments.aviutl2, os.environ)
    if aviutl2 is None:
        print("AviUtl2（aviutl2.exe）が見つかりません 探した所")
        for place in looked:
            print(f"  {place}")
        print(f"--aviutl2 か環境変数 {ENVIRONMENT_KEY} で場所を教えてください")
        return None
    for image in BLOCKING_IMAGES:
        if ymm4_running(image):
            # 開いているプロジェクトの上から別のプロジェクトを開くと、作業中の物を壊しかねない
            # 閉じる前の設定の書き直しも、道具の控えと食い違う
            print(f"{image} が開いています 作業中の物を壊さないように止めました")
            return None
    folder = data_folder(aviutl2, os.environ)
    if folder is None or not folder.is_dir():
        print("AviUtl2 の設定の置き場が分からないので、閉じたあとに設定を戻せません 止めました")
        return None
    return aviutl2, project, output, folder, frames


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    checked = _refuse(arguments)
    if checked is None:
        return 1
    aviutl2, project, output, folder, frames = checked
    saved = save_settings(folder)
    for item in saved:
        print(f"設定の控え {item.path.name} {digest(item.content)}")
    backups_before = _backups(folder)
    partial = partial_folder(output)
    partial.mkdir(parents=True)
    print(f"AviUtl2 を動かします（{frames} 枚） 終わるまでマウスとキーボードに触らないでください")
    command = powershell_command(
        aviutl2, project, partial, frames=frames, timeout=arguments.timeout
    )
    limit = arguments.timeout + WATCHDOG_MARGIN
    code, expired = run_script(command, limit)
    if expired or ymm4_running(EXE_NAME):
        # AviUtl2 がまだ開いているうちに戻すと、閉じるときにまた書き直される 戻さずに知らせる
        _alarm(
            [
                "AviUtl2 が閉じていないので、設定を戻していません",
                "AviUtl2 を閉じてから、次の控えの中身へ戻してください",
                *(f"{item.path} {digest(item.content)}" for item in saved),
            ]
        )
        return RESTORE_FAILED
    problems = restore_settings(saved)
    if problems:
        _alarm(["AviUtl2 の設定を元へ戻せませんでした", *problems])
        return RESTORE_FAILED
    for item in saved:
        print(f"設定を戻しました {item.path.name} {digest(item.content)}")
    for name in sorted(_backups(folder) - backups_before):
        # AviUtl2 自身の自動控え 本人の控えと同じ置き場なので消さずに知らせる
        print(f"AviUtl2 が自動控えを足しました {folder / 'Backup' / name}")
    written = len(list(partial.glob("*.png"))) if code == 0 else 0
    if code == 0 and written != frames:
        # PowerShell の終了コードだけを信じない YMM4 の道具では、書き出しが途中で止まっても
        # PowerShell が書き終えたと言った（Issue #210） 足りない連番を出すと、compare が
        # 欠けた枠を「AviUtl2 が何も描かない」と読む
        print(f"AviUtl2 の書き出しが途中で止まりました {written} / {frames} 枚")
        code = 1
    if code != 0:
        # 書き終えなかった絵は捨てる この名前は始めに無いことを確かめて作ったので、中身は今回の物
        shutil.rmtree(partial, ignore_errors=True)
        return code
    publish(partial, output)
    print(f"書き出しました {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
