"""書いてから置き換える書き込みと、読んでから書き戻す所の錠

設定・プリセット・プロジェクトなど、途中で落ちても元のファイルを壊したくない物は、
隣の仮のファイルへ書き切ってから名前を付け替える その仮のファイルの名前を 1 か所で決める

Qt を使わない コア層の保存（プロジェクト・プリセット・エイリアス・退避）からも呼ぶため
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from sashimono.core.io.locks import try_hold

__all__ = ["STALE_WRITING_SECONDS", "WRITING_SUFFIX", "held_for_update", "write_atomic"]

#: 書きかけの仮のファイルの名前の後ろ 置き場の引き継ぎ（``core/userdirs.py``）は、
#: これで終わる物を書きかけとして移さない 名前の形を変えるときは、そちらも合わせる
WRITING_SUFFIX = ".writing"

#: 書き手ごとの印の形（``uuid4`` の頭 8 桁）
_MARK = re.compile(r"[0-9a-f]{8}")

#: ファイルの名前の長さの上限（Windows は UTF-16 の 255 字 Linux などは 255 バイト）
_NAME_MAX = 255

#: 仮のファイルの名前で、元の名前の頭の後ろに足す長さ（``.<印 8 桁>.writing``）
_SCRATCH_TAIL = len(".") + 8 + len(WRITING_SUFFIX)

#: 落ちた書き手が残した仮のファイルを片付けるまでの古さ（秒）
#: 書いている最中の物は書くたびに更新時刻が進むので、これより古いのは持ち主がもういない
#: 短くすると、遅いドライブで大きなプロジェクトを書いている別の窓の仮のファイルを消してしまう
STALE_WRITING_SECONDS = 60 * 60

#: 置き換えが「使用中」で断られたときに試し直す長さ（秒） 相手の置き換えは一瞬で終わる
#: 長くすると、本当に開いたままのファイル（エディタで掴んでいる など）で保存が長く止まる
_REPLACE_WAIT = 1.0

#: 錠を待つ長さ（秒） 読んで書き戻すのは一瞬で終わるので、これだけ待っても取れないのは
#: 持ち主が止まっているとき 保存を長く止めない
_LOCK_WAIT = 2.0


def write_atomic(path: Path, text: str, *, durable: bool = False) -> None:
    """``path`` へ ``text`` を UTF-8 で書く 書き終えるまで元のファイルは無傷で残る

    仮のファイルの名前に書き手ごとの印を混ぜる（``<名前>.<印>.writing``） 同じ名前だと、
    2 つの窓が同時に同じファイルを書いたときに、片方が相手の書きかけを上書きしたり、
    相手が先に置き換えて仮のファイルが消え、もう片方の置き換えが落ちたりする
    名前の上限に近い長い名前は、頭を切って収める（:func:`_scratch_head`）

    書くか置き換えるのに失敗したら、仮のファイルを片付けてから上げ直す 片付けないと、
    印が書き手ごとに違うので、落ちるたびに書きかけが積もる

    ``durable`` を真にすると、名前を付ける前にディスクへ書き切る（fsync） 電源が切れても
    中身の無いファイルに置き換わらない 作り直せない物（プロジェクト）だけに使う
    数十秒おきの退避や設定まで待たせると、遅いドライブで編集が引っかかる

    改行は ``\\n`` のまま書く OS ごとに書く物が変わらないように
    """
    path = Path(path)
    temporary = path.with_name(f"{_scratch_head(path.name)}.{uuid.uuid4().hex[:8]}{WRITING_SUFFIX}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
        _replace(temporary, path)
    except BaseException:
        # 後片付けの失敗で元の理由を隠さない
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise
    _sweep_stale(path)


def _replace(temporary: Path, path: Path) -> None:
    """仮のファイルに ``path`` の名前を付ける 相手が同じ名前を付け替えている間は少し待って試し直す

    Windows は、ほかの窓が同じファイルへ名前を付けている最中や読んでいる最中の置き換えを
    「使用中」（``PermissionError``）で断る 一瞬で空くので、すぐに上げるとその保存だけが落ちる
    待っても空かなければ上げる（呼ぶ側が保存できなかったと伝える）
    """
    deadline = time.monotonic() + _REPLACE_WAIT
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _sweep_stale(path: Path) -> None:
    """前に落ちた書き手が ``path`` の隣に残した仮のファイルを片付ける 片付けられなくても止めない

    名前に印を混ぜたので、落ちて残った物は次の書き込みでも上書きされない ここで消さないと、
    同じファイルを保存するたびに落ちた分だけ積もる 前の版の決まった名前（``<名前>.writing``）
    も拾う 古さで分けるのは、別の窓が今まさに書いている物を消さないため
    """
    limit = time.time() - STALE_WRITING_SECONDS
    try:
        with os.scandir(path.parent) as entries:
            stale = [
                Path(entry.path)
                for entry in entries
                if _is_leftover(entry.name, path.name)
                and entry.is_file(follow_symlinks=False)
                and entry.stat(follow_symlinks=False).st_mtime < limit
            ]
    except OSError:
        return
    for leftover in stale:
        with contextlib.suppress(OSError):
            leftover.unlink(missing_ok=True)


def _scratch_head(name: str) -> str:
    """仮のファイルの名前の頭 元の名前が長ければ、印と後ろを足しても収まるところで切る

    ファイルの名前は Windows で 255 文字（UTF-16）、Linux などで 255 バイト（UTF-8 で日本語は
    1 字 3 バイト）まで 元の名前を丸ごと頭にすると、上限に近い名前（長い日本語の題の
    プロジェクトなど）では仮のファイルを作れず、保存そのものができない

    切ったときは、切る前の名前全体から作る短い印（``~<sha256 の頭 8 桁>``）を足す 足さないと、
    頭の同じ別の長い名前と頭が重なり、片方の片付け（:func:`_sweep_stale`）がもう片方の
    書きかけを自分の物と見なして消す
    """
    if len(name.encode("utf-8")) <= _NAME_MAX - _SCRATCH_TAIL:
        return name
    digest = "~" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    room = _NAME_MAX - _SCRATCH_TAIL - len(digest)
    head = name
    # バイトで収めれば字数も収まる UTF-8 のバイト数は、どの文字でも UTF-16 の字数以上
    while len(head.encode("utf-8")) > room:
        head = head[:-1]
    return head + digest


def _is_leftover(name: str, target: str) -> bool:
    """``target`` のファイルの仮のファイルの名前か 本人の物を取り違えて消さないよう形を厳しく見る"""
    if name == f"{target}{WRITING_SUFFIX}":
        # 前の版の決まった名前（``<名前>.writing``）
        return True
    head = f"{_scratch_head(target)}."
    if not name.startswith(head) or not name.endswith(WRITING_SUFFIX):
        return False
    return _MARK.fullmatch(name[len(head) : -len(WRITING_SUFFIX)]) is not None


@contextlib.contextmanager
def held_for_update(lock: Path, wait: float | None = None) -> Iterator[None]:
    """読んでから書き戻すまでを、プロセスの間で 1 人ずつにする

    守らないと、2 つの窓が同時に読んで書き戻したとき、後から書いた方が先の変更を消す
    錠は窓の錠と同じ仕組み（:func:`~sashimono.core.io.locks.try_hold`） 持ち主が落ちれば
    次に取りに来た方が片付けて取り直す 待っても取れなければ ``TimeoutError``（``OSError``）
    ``wait`` を省くと :data:`_LOCK_WAIT` 秒待つ
    """
    deadline = time.monotonic() + (_LOCK_WAIT if wait is None else wait)
    while (held := try_hold(lock)) is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"錠を取れない: {lock}")
        time.sleep(0.01)
    try:
        yield
    finally:
        held.release()
