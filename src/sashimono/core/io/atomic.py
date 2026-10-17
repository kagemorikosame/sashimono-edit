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
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from sashimono.core.io.locks import try_hold

__all__ = [
    "STALE_WRITING_SECONDS",
    "WRITING_SUFFIX",
    "Expected",
    "TargetChangedError",
    "copy_atomic",
    "expect_now",
    "held_for_update",
    "write_atomic",
    "write_atomic_bytes",
]

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

#: 大きな物を写すときに 1 度に読む長さ 手元に動画を丸ごと抱えないように
_COPY_CHUNK = 1 << 20

#: 錠を待つ長さ（秒） 読んで書き戻すのは一瞬で終わるので、これだけ待っても取れないのは
#: 持ち主が止まっているとき 保存を長く止めない
_LOCK_WAIT = 2.0


def write_atomic(
    path: Path, text: str, *, durable: bool = False, expect: Expected | None = None
) -> None:
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
    _write(path, text.encode("utf-8"), durable=durable, expect=expect)


def write_atomic_bytes(
    path: Path, data: bytes, *, durable: bool = False, expect: Expected | None = None
) -> None:
    """:func:`write_atomic` のバイト列の版 絵（PNG）のように文字ではない物を書く

    仮のファイルの名前・片付け・置き換えの決まりは :func:`write_atomic` と同じ 見本の絵
    （``ui/library_thumbnails.py``）が使う
    """
    _write(path, data, durable=durable, expect=expect)


def copy_atomic(
    source: Path, path: Path, *, durable: bool = False, expect: Expected | None = None
) -> os.stat_result:
    """``source`` の中身を ``path`` へ写す 書き終えるまで元のファイルは無傷で残る

    仮のファイルの名前・片付け・置き換えの決まりは :func:`write_atomic` と同じ 大きな物
    （書き出した動画）を手元へ読み込まずに少しずつ写す 書いた仮のファイルの身元
    （:func:`os.fstat`）を返す 置き換えた後に、名前の先が自分の書いた物かを照らすため
    """
    with Path(source).open("rb") as reader:
        return _fill(
            Path(path),
            lambda handle: shutil.copyfileobj(reader, handle, _COPY_CHUNK),
            durable,
            expect,
        )


#: 置き換えた前のファイルを残す世代の既定 画面の設定のバックアップの世代数の既定と同じ
KEPT_GENERATIONS = 20


@dataclass(frozen=True, slots=True)
class Expected:
    """書く先が、確かめたときのままかを照らすための覚え（#327）

    ``identity`` は確かめたときの身元（装置・番号・大きさ・更新時刻） 無かったなら ``None``
    人が許した後、書き終えて名前を付けるまでに、同じ名前が現れたり中身が変わったりすると、
    許していない物を置き換える :func:`expect_now` で作り、書く関数の ``expect`` に渡す
    """

    identity: tuple[int, int, int, int] | None
    #: 置き換えた前のファイルを残す世代の数（同じ書く先ごと） 照らす身元ではないので比べない
    #: 書く側（画面の設定を持つホスト）が、バックアップの世代数
    #: （``Preferences.backup_generations``）を入れる
    keep: int = field(default=KEPT_GENERATIONS, compare=False)
    #: 置き換えた前のファイルを残した場所（書いた後に足す） 呼んだ側が AI と人に知らせる
    kept: list[Path] = field(default_factory=list, compare=False)


class TargetChangedError(FileExistsError):
    """書く先が、確かめたときから変わった（名前が現れた・中身が変わった） 何も置いていない"""


class PlacingError(OSError):
    """許した前のファイルを退けた後、新しいファイルを付けられなかった

    文には、前のファイルを元の名前へ戻せたか、戻せずに残した名前を入れる
    """


def expect_now(path: Path) -> Expected:
    """``path`` の今の身元を覚える リンクは辿らない（リンクそのものを書く先と見る）"""
    try:
        status = Path(path).stat(follow_symlinks=False)
    except FileNotFoundError:
        return Expected(None)
    return Expected((status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns))


def _write(path: Path, data: bytes, *, durable: bool, expect: Expected | None = None) -> None:
    _fill(Path(path), lambda handle: handle.write(data), durable, expect)


def _fill(
    path: Path,
    fill: Callable[[BinaryIO], object],
    durable: bool,
    expect: Expected | None = None,
) -> os.stat_result:
    """仮のファイルを新しく作って ``fill`` で中身を入れ、``path`` の名前を付ける

    ``expect`` を渡すと、名前を付ける所で書く先が覚えたときのままかを照らす
    無かった名前は置き換えない付け方で付け（後から現れていれば付けずに断る）、あった名前は
    覚えた身元のままかを付ける直前に照らす どちらも違えば :class:`TargetChangedError`
    """
    temporary = path.with_name(f"{_scratch_head(path.name)}.{uuid.uuid4().hex[:8]}{WRITING_SUFFIX}")
    # 仮のファイルは必ず新しく作る（"x"） 同じ名前の物が先に置かれていたら、その中身へは
    # 書かずに断る 開いて上書きすると、置かれていたのが元の素材へのハードリンクだったとき
    # 元の素材の中身を書き換える 断った名前は自分の物ではないので片付けない
    handle = temporary.open("xb")
    try:
        with handle:
            fill(handle)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
            written = os.fstat(handle.fileno())
        if expect is None:
            _replace(temporary, path)
        elif expect.identity is None:
            _place_new(temporary, path)
        else:
            _replace_expected(temporary, path, expect)
    except BaseException:
        # 後片付けの失敗で元の理由を隠さない
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise
    _sweep_stale(path)
    return written


#: 置き換える前のファイルを退ける名前の後ろ 書きかけ（``.writing``）と分けるのは、古い
#: 書きかけの片付け（:func:`_sweep_stale`）で消させないため 戻せずに残した物は人が見て決める
ASIDE_SUFFIX = ".replaced"


def _replace_expected(temporary: Path, path: Path, expect: Expected) -> None:
    """許した前のファイル（``expect``）だけを、新しいファイルで置き換える

    照らしてから置き換えるまでを別の操作にすると、その間に別のアプリが置いたファイルを
    上書きする（照らした物と置き換えた物が違う） そこで次の順に行い、どの間に割り込まれても
    相手のファイルを潰さない

    1. 書く先を、置き換えない改名で自分だけが知る名前へ退ける 退けた物は自分の物になる
    2. 退けた物の身元を照らす 違えば、退けた物を置き換えない改名で元の名前へ戻して断る
    3. 新しいファイルを、置き換えない付け方（:func:`_place_new`）で元の名前に付ける
       その間に名前が現れていれば付けずに、退けた物を戻そうとして断る
    4. 退けた物は消さずに、同じフォルダの決まった名前で残す（:func:`_keep_beside`）
       消すと、別のアプリが消す共有を許して開いたまま書いていた更新ごと失う

    戻せない（元の名前がもう別の物で埋まっている）ときは、退けた物を消さずに残し、その名前を
    断りの文に入れる 黙って消すと、人が許していない形で前のファイルが失われる
    ``ReplaceFileW`` は置き換える物の身元を照らせず、照らしと置き換えの間の割り込みは
    同じく防げないので使わない
    """
    aside = path.with_name(f"{_scratch_head(path.name)}.{uuid.uuid4().hex[:8]}{ASIDE_SUFFIX}")
    try:
        _rename_new(path, aside)
    except FileNotFoundError as exc:
        raise TargetChangedError(
            f"書く先が許した後に消えたので、何も置きませんでした: {path}"
        ) from exc
    # 退けてから付け終えるまでのどこで失敗しても（照らせない・ウイルス対策や権限で付けられない
    # など）、退けた物を元の名前へ戻そうとする 戻せなければ残した名前を断りの文に入れる
    # 捕まえ損ねると、元の名前が消えたまま退けた名前だけが残り、人はどこにあるか分からない
    try:
        if expect_now(aside) != expect:
            raise TargetChangedError(
                f"書く先が許した後に変わったので、前のファイルを残しました: {path}"
            )
        _place_new(temporary, path)
    except TargetChangedError as exc:
        raise TargetChangedError(f"{exc}{_put_back(aside, path)}") from exc
    except OSError as exc:
        raise PlacingError(f"{exc}{_put_back(aside, path)}") from exc
    except BaseException:
        # 中断など 戻すだけ試して、元の理由のまま上げる
        _put_back(aside, path)
        raise
    try:
        kept = _keep_beside(aside, path, expect.keep)
    except Exception:
        # 新しいファイルは付け終えている 退けた物は退けた名前のまま残っているので、それを伝える
        kept = aside
    expect.kept.append(kept)


#: 置き換えた前のファイルを残す名前の形 ``<元の名前の頭>.before-<時刻><元の後ろ>``
#: 元と同じ後ろ（.mp4 .sme など）にして、人が見つけてそのまま開けるようにする
_BEFORE = "{stem}.before-{stamp}{suffix}"
_BEFORE_STAMP = re.compile(r"\d{8}-\d{6}-\d{6}-\d{3}")


def _keep_beside(aside: Path, path: Path, keep: int) -> Path:
    """退けた前のファイルを、消さずに同じフォルダの決まった名前で残す 残した場所を返す

    消さないのは、別のアプリが消す共有（FILE_SHARE_DELETE）を許して書く先を開いたまま
    書いていると、改名した後も同じ実体へ書き続けられ、照らした後の更新ごと消してしまうため
    改名は同じ実体を動かすだけなので、その更新もここに残る 同じフォルダにするのは、別の
    ドライブへ動かすと写して消すことになり、同じく消してしまうから

    溜まり続けないよう、同じ書く先の物を ``keep`` 本に保つ 1 回で消すのは 1 本まで
    （バックアップの世代の入れ替えと同じ決まり 設定を減らしても、保存のたびにまとめて消さない）
    残す名前へ付けられなければ、退けた名前のまま残して返す
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stem, suffix = path.stem, path.suffix
    # 連番は桁をそろえて必ず付ける（recovery の控えと同じ） 付けない物と付けた物、1 桁と 2 桁が
    # 混ざると、名前の順（片付けが古い順と見なす順）と作った順が食い違い、新しい方を消す
    # 時計の刻み（Windows で約 15ms）で同じ時刻が続くのはふつうに起こる
    # 空いた番号を使い直さない 片付けで古い 000 が消えた後に新しい物へ 000 を付けると、
    # 新しい物が一番古く見えて次に消される 同じ時刻の物の一番大きい番号の次から付ける
    for number in range(_next_number(path, stamp), 1000):
        name = _BEFORE.format(stem=stem, stamp=f"{stamp}-{number:03d}", suffix=suffix)
        try:
            _rename_new(aside, path.with_name(name))
        except FileExistsError:
            continue
        except OSError:
            return aside
        kept = path.with_name(name)
        _tidy_kept(path, max(1, keep))
        return kept
    return aside


def _next_number(path: Path, stamp: str) -> int:
    """同じ時刻で残した物の、一番大きい連番の次 無ければ 0"""
    head = f"{path.stem}.before-{stamp}-"
    numbers = [-1]
    try:
        for entry in os.scandir(path.parent):
            name = entry.name
            if name.startswith(head) and name.endswith(path.suffix):
                digits = name[len(head) : len(name) - len(path.suffix)]
                if len(digits) == 3 and digits.isdigit():
                    numbers.append(int(digits))
    except OSError:
        pass
    return max(numbers) + 1


def _tidy_kept(path: Path, keep: int) -> None:
    """同じ書く先の、残した前のファイルを古い物から 1 本だけ消して ``keep`` 本に近づける"""
    head = f"{path.stem}.before-"
    try:
        kept = sorted(
            entry.name
            for entry in os.scandir(path.parent)
            if entry.name.startswith(head)
            and entry.name.endswith(path.suffix)
            and _BEFORE_STAMP.fullmatch(entry.name[len(head) : len(entry.name) - len(path.suffix)])
            and entry.is_file(follow_symlinks=False)
        )
    except OSError:
        return
    for name in kept[: max(0, len(kept) - keep)][:1]:
        _delete_if_unheld(path.parent / name)


def _delete_if_unheld(path: Path) -> bool:
    """誰も開いていないときだけ消す 消せたら真

    残した前のファイルは、別のアプリが消す共有を許して開いたまま書いている物かもしれない
    （そのために消さずに残した） 普通に消すと、その更新ごと失う Windows では、共有を一切
    許さない開き方で、閉じたら消す印を付けて開く 誰かが開いていれば開けないので消さない
    （次の片付けで改めて見る） 開けたら、閉じた時に消える 開いている人を確かめられない
    OS では消さない
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.restype = wintypes.HANDLE
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    close = kernel32.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    delete_access = 0x00010000
    no_sharing = 0
    open_existing = 3
    # 閉じたら消す・リンクなら辿らずにリンクそのものを相手にする
    flags = 0x04000000 | 0x00200000
    handle = create(str(path), delete_access, no_sharing, None, open_existing, flags, None)
    if handle is None or handle == wintypes.HANDLE(-1).value:
        return False
    close(handle)
    return not path.exists()


def _put_back(aside: Path, path: Path) -> str:
    """退けた物を元の名前へ置き換えない改名で戻す 戻せなければ、残した名前を知らせる言葉を返す"""
    try:
        _rename_new(aside, path)
    except OSError as exc:
        # 場所はフォルダまで書く 名前だけだと、どのフォルダかを人が探すことになる
        return (
            f" 許した前のファイルは元の名前へ戻せなかったので、{aside} として残しました"
            f"（{exc}） 元の名前へ戻すかは人が決めてください"
        )
    return ""


def _rename_new(source: Path, destination: Path) -> None:
    """``source`` を ``destination`` へ改名する ``destination`` があれば置き換えずに断る

    Windows の改名は置き換えずに断る ほかの OS は黙って置き換えるので、リンクを張ってから
    元の名前を外す 使用中で断られたら、:func:`_replace` と同じく少し待って試し直す
    """
    deadline = time.monotonic() + _REPLACE_WAIT
    while True:
        try:
            if sys.platform == "win32":
                source.rename(destination)
            else:
                os.link(source, destination)
                source.unlink()
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _place_new(temporary: Path, path: Path) -> None:
    """仮のファイルに ``path`` の名前を付ける ただし同じ名前がもうあれば置き換えずに断る

    確かめてから付けるまでの間に名前が現れても置き換えないよう、1 度の呼び出しで行う
    Windows の改名（``MoveFileEx`` に置き換えの印を付けない）はあれば断る ほかの OS の改名は
    黙って置き換えるので、リンクを張って（あれば断る）から仮の名前を外す
    """
    try:
        # Path.rename は Windows で置き換えずに断る（MoveFileEx に置き換えの印を付けない）
        _rename_new(temporary, path)
    except FileExistsError as exc:
        raise TargetChangedError(
            f"書く先に、許した後で同じ名前のファイルができたので、前のファイルを残しました: {path}"
        ) from exc


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
