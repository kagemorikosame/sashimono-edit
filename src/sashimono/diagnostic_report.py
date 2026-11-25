r"""利用者が自分で確かめて送る診断情報（#516）

利用者の機械でだけ起きる不具合を、個人の情報を守りながら調べるための報告を組む
**自動では送らない** ヘルプの〔診断情報を作る…〕が窓に全文を出し、本人が読んで（消したい所を
消して）ファイルに保存し、報告の窓口（Discussions）へ自分で添える 送る口とサーバーは持たない
（送るかどうかを本人より先に決めないため 要るかは別に判断する #502）

載せるのは調べるのに要る最小限

- 版と環境（Python・Qt・OS・配布版か・画面）
- 設定（好みの値） 置き場の欄は「既定か・選んであるか」だけ 新しい作品の編集の方針は有る無しだけ
- 画面の止まりの記録を数えた表（:mod:`sashimono.stall_summary` 記録を入れた人だけ）
- 拾われなかったエラーと、正しく終わらなかった起動（:mod:`sashimono.fault_log`）

載せた後で伏せる（:func:`mask`） 伏せ方は書き足さず、AI へ送る物の伏せ方
（:mod:`sashimono.ai.outbound` 素材の名前と場所・字幕と文字）と、報告に貼る文の伏せ方
（:mod:`sashimono.ui.report_masking` 本人の置き場・秘密）を使う 別々に書くと、片方だけ
直したときにもう片方から漏れる

Qt を使わない 窓の側（:mod:`sashimono.ui.diagnostic_dialog`）が Qt の版と画面を渡す
"""

from __future__ import annotations

import json
import platform
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from sashimono import __version__
from sashimono.ai.outbound import PATH_MARK, TEXT_MARK, hide_text, redact
from sashimono.fault_log import FaultLog, LostSession
from sashimono.runtime import is_frozen
from sashimono.stall_summary import default_paths, read, report, summarize
from sashimono.stall_watch import program_place
from sashimono.ui.report_masking import (
    mask_places,
    mask_record_text,
    mask_report,
    mask_secrets,
)

__all__ = [
    "Environment",
    "Gathered",
    "build_report",
    "environment",
    "gather",
    "mask",
    "preference_lines",
    "program_paths",
    "suggested_name",
]

#: 載せるエラーの数 新しい方から 古いエラーは今の不具合と関係が薄く、長いと読んで確かめる気が失せる
ERROR_COUNT = 5
#: 載せる落ちた起動の数
LOST_COUNT = 3
#: 1 件のエラーや落ちた起動の記録から載せる行の数（後ろから） 落ちた所に近い行ほど要る
TAIL_LINES = 60
#: 止まりの表に出す場所の数
STALL_TOP = 8

#: 並びの中のファイルの場所 ``File "C:\…\x.py", line 3``（Python のエラーと faulthandler の両方）
_FILE_IN_TRACE = re.compile(r'File "([^"]+)"')


@dataclass(frozen=True, slots=True)
class Environment:
    """版と環境"""

    version: str
    python: str
    qt: str
    system: str
    frozen: bool
    #: 画面ごとの ``1920x1080 150%`` のような行
    screens: tuple[str, ...] = ()


def environment(qt: str = "", screens: Sequence[str] = ()) -> Environment:
    """この機械の版と環境 Qt の版と画面は窓の側から渡す（ここは Qt を読まない）"""
    return Environment(
        version=__version__,
        python=platform.python_version(),
        qt=qt,
        system=platform.platform(),
        frozen=is_frozen(),
        screens=tuple(screens),
    )


@dataclass(frozen=True, slots=True)
class Gathered:
    """記録から集めた物 集めるのは窓を開いたとき 失敗した物は ``problems`` に理由を残す"""

    stalls: str = ""
    errors: tuple[str, ...] = ()
    lost: tuple[LostSession, ...] = ()
    problems: tuple[str, ...] = field(default_factory=tuple)


def gather(log: FaultLog) -> Gathered:
    """止まりの記録・エラー・落ちた起動を集める どれかが読めなくても残りは載せる

    1 つ読めないだけで報告そのものを作れないと、調べる手掛かりが何も残らない 読めなかった
    ことは報告に書く（黙って抜くと、記録が無かったのか読めなかったのか分からない）
    """
    problems: list[str] = []
    stalls = ""
    try:
        unread: list[Path] = []
        records = read(default_paths(), unread=unread)
        if records:
            stalls = report(summarize(records), top=STALL_TOP)
        # 読む側は無いファイルも読めなかった物に数える 1 世代前は無いのが普通なので、
        # 在るのに読めなかった物だけを書く 名前だけを書く（場所は本人の置き場）
        problems.extend(
            f"画面の止まりの記録を読めなかった: {path.name}" for path in unread if path.exists()
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"画面の止まりの記録を読めなかった: {type(exc).__name__}")
    errors: list[str] = []
    unread_errors: list[Path] = []
    try:
        errors = log.recent_errors(ERROR_COUNT, unread=unread_errors)
    except OSError as exc:
        problems.append(f"エラーの記録を読めなかった: {type(exc).__name__}")
    problems.extend(f"エラーの記録を読めなかった: {path.name}" for path in unread_errors)
    lost: list[LostSession] = []
    unread_sessions: list[Path] = []
    try:
        lost = log.lost_sessions(unread=unread_sessions)[-LOST_COUNT:]
    except OSError as exc:
        problems.append(f"落ちた起動の記録を読めなかった: {type(exc).__name__}")
    problems.extend(f"落ちた起動の記録を読めなかった: {path.name}" for path in unread_sessions)
    # 前の起動の記録は、今の作品から作る伏せる表に無い名前と本文を含みうる 例外の文が名前と
    # 本文を入れる形（囲んだ文・名前と拡張子の語）を、集めた所で伏せる（PR #550 の指摘）
    errors = [mask_record_text(entry, TEXT_MARK) for entry in errors]
    lost = [replace(session, text=mask_record_text(session.text, TEXT_MARK)) for session in lost]
    return Gathered(stalls, tuple(errors), tuple(lost), tuple(problems))


def preference_lines(preferences: Mapping[str, Any]) -> list[str]:
    """設定を 1 行 1 項目で 置き場の欄は値を伏せる（AI へ送る設定と同じ伏せ方）

    文字・数・真偽だけを載せる 並びや辞書（新しい作品の編集の方針など）は本人の文を含みうるので
    有る無しだけにする
    """
    hidden = redact(dict(preferences), ())
    lines: list[str] = []
    for key in sorted(hidden):
        value = hidden[key]
        if isinstance(value, bool | int | float) or value is None:
            lines.append(f"  {key}: {json.dumps(value)}")
        elif isinstance(value, str):
            # 空の置き場は「既定」 値を伏せた欄は印のまま（選んであることだけ分かる）
            lines.append(f"  {key}: {value if value else '（空 既定）'}")
        else:
            lines.append(f"  {key}: {'（有り）' if value else '（無し）'}")
    return lines


def build_report(
    env: Environment,
    preferences: Mapping[str, Any] | None,
    gathered: Gathered,
    *,
    now: datetime | None = None,
) -> str:
    """伏せる前の報告の全文"""
    made = (now or datetime.now()).isoformat(timespec="seconds")
    out = [
        "Sashimono Edit の診断情報",
        f"作った時刻: {made}",
        "この報告は自動では送られません 中身を確かめてから報告に添えてください",
        "",
        "## 版と環境",
        f"版: {env.version}（{'配布版' if env.frozen else '開発の環境'}）",
        f"Python {env.python} / Qt {env.qt or '不明'}",
        f"OS: {env.system}",
    ]
    out.extend(f"画面: {screen}" for screen in env.screens)
    out += ["", "## 設定"]
    out.extend(preference_lines(preferences) if preferences is not None else ["  （読めなかった）"])
    out += ["", "## 画面の止まりの記録"]
    out.append(
        gathered.stalls
        if gathered.stalls
        else "記録なし（表示 → 設定… の「画面の止まりを記録する」が切か、まだ止まっていない）"
    )
    out += ["", f"## 拾われなかったエラー（新しい {ERROR_COUNT} 件まで）"]
    if gathered.errors:
        for entry in gathered.errors:
            out.append(_tail(entry))
            out.append("")
    else:
        out.append("なし")
    out += ["", f"## 正しく終わらなかった起動（新しい {LOST_COUNT} 件まで）"]
    if gathered.lost:
        for session in gathered.lost:
            out.append(f"起動 {session.started} 版 {session.version}")
            out.append(
                _tail(session.text)
                if session.text
                else "  落ちた時の記録なし（強制終了・電源断か、Python の外で止まった）"
            )
            out.append("")
    else:
        out.append("なし")
    if gathered.problems:
        out += ["", "## 読めなかった記録"]
        out.extend(gathered.problems)
    return "\n".join(out).rstrip() + "\n"


def _tail(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) > TAIL_LINES:
        lines = [f"（前の {len(lines) - TAIL_LINES} 行を省いた）", *lines[-TAIL_LINES:]]
    return "\n".join(lines)


def program_paths(text: str) -> str:
    r"""並びの中のファイルの場所を、本体の中の場所（``sashimono/ui/x.py``）か名前だけにする

    伏せる前に行う 置き場ごと伏せると、どのファイルの何行目で落ちたかが読めなくなる
    削り方は画面の止まりの記録と同じ（:func:`sashimono.stall_watch.program_place`） 外の部品は
    名前だけになるので、インストール先（利用者名を含みうる）は残らない
    """
    return _FILE_IN_TRACE.sub(lambda found: f'File "{program_place(found.group(1))}"', text)


def mask(
    text: str,
    *,
    names: Iterable[tuple[str, str]] = (),
    texts: Iterable[str] = (),
    folders: Sequence[tuple[str, str]] | None = None,
) -> str:
    r"""報告を伏せる

    ``names`` は素材の名前と場所の表（:func:`sashimono.ai.outbound.file_secrets` を
    :func:`~sashimono.ai.outbound.secret_pairs` で並べた物） ``texts`` は字幕と文字
    （:func:`~sashimono.ai.outbound.text_secrets`） ``folders`` は試験で差し替える本人の置き場

    順に意味がある

    1. 並びの中の場所を本体の中の場所にする（伏せると何行目で落ちたかが消える）
    2. 字幕と文字（長い順） 場所を伏せた後だと、文の中の場所の印と混ざる
    3. 本人の置き場を印にする 空白を含む利用者名（``C:\Users\Kage Mori``）は、形で場所を
       拾う伏せ方では空白の所で切れて名前の後ろが残るので、置き場の名前で先に伏せる
    4. 場所（置き場の印の下 ``%USERPROFILE%\Videos\旅行 1 日目.mp4`` とドライブ名・UNC の場所）を
       空白で止めずに伏せる 形で拾う伏せ方は空白で止まり、名前の後ろが残る
    5. 素材の名前の表と、表に無い残りの絶対の場所
    6. 秘密（API の鍵・トークン） 名前の表より前にも 1 度伏せる（表が鍵の名前を消すため）
    """
    text = program_paths(text)
    text = str(hide_text(text, sorted(set(texts), key=len, reverse=True), labels=False))
    text = mask_report(text, folders=folders)
    text = mask_places(text, PATH_MARK)
    # 秘密は名前の表より先に伏せる 素材や作品の名前が鍵の名前（password など）と同じだと、
    # 表が鍵の名前を消し、後の秘密の伏せ方が値を見つけられない（PR #550 の指摘）
    text = mask_secrets(text)
    text = str(redact(text, names))
    return mask_secrets(text)


def suggested_name(now: datetime | None = None) -> str:
    """保存するファイルの名前 時刻を入れる 何度か作ったときに上書きせず並ぶ"""
    return f"sashimono-診断情報-{(now or datetime.now()):%Y%m%d-%H%M%S}.txt"
