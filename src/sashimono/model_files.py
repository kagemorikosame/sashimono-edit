"""後から入れる部品が使う、pip では入らないファイル（学習済みのモデルなど）を落とす

画面の解析（#324）の顔の検出は、YuNet のモデル（ONNX 約 230 KB）を読む PyPI に包みが無いので、
導入のボタンで pip の後に取りに行く（:attr:`~sashimono.runtime.FeaturePack.files`）
zip に積まないのは、使わない人にまで他人の作った物を配らずに済み、開発の環境と配布版で
置き場と入れ方が同じになるため リポジトリにも入れない（配布物はリポジトリに入れない決まり）

取得元はコミットで固定した URL で、落とした物は sha256 で照らす 照らさないと、取得元が
差し替えられたり途中で壊れたりした物を、そのまま読み込んで動かす 合わなければ置かない
置くのは書き終えてから名前を付け替える（:func:`~sashimono.core.io.atomic.write_atomic_bytes`）
途中で止まっても、半端なファイルを「入っている」と見ない
"""

from __future__ import annotations

import hashlib
import http.client
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from urllib.parse import urlsplit

from sashimono.core import userdirs

__all__ = [
    "MODELS_DIR",
    "HttpsOnlyRedirect",
    "Opener",
    "RemoteFile",
    "fetch_files",
    "missing_files",
    "models_root",
]

#: 落とした物を置くフォルダの名前（本人の置き場 ``%LOCALAPPDATA%\\Sashimono`` の下）
#: 配布版の導入先（``runtime``）とは分ける 導入先は Python の版が上がると入れ直すが、
#: モデルは Python の版に関係なく使えるので、入れ直しのたびに落とし直さずに済む
MODELS_DIR = "models"

#: 待つ長さ（秒） 繋がらないときは早めに諦める 導入の欄は中断のボタンを持つが、
#: 1 つの読み込みの中では中断が届かない
TIMEOUT_SECONDS = 30.0

#: 1 回に読む大きさ 中断を見る間隔にもなる
_CHUNK = 64 * 1024

#: URL の頼みを開く物 試験で差し替える（本物の取得元へ出ないように）
Opener = Callable[[urllib.request.Request], IO[bytes]]


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """pip では入らない 1 つのファイル"""

    #: 置き場（:func:`models_root`）の中の名前
    name: str
    #: 取得元 中身が変わらないよう、コミットや版で固定した HTTPS の URL にする
    url: str
    #: 中身の sha256（16 進の小文字） 合わなければ置かない
    sha256: str
    #: 大きさ（バイト） これを超えて送られてきたら読むのをやめる（取得元の取り違えで
    #: 大きな物を延々と落とさないため）
    size: int
    #: 画面と使用許諾の一覧に出す、何の物か
    label: str = ""

    def path(self, root: Path | None = None) -> Path:
        """置き場の中の場所"""
        return (root if root is not None else models_root()) / self.name

    def present(self, root: Path | None = None) -> bool:
        """置いてあるか 大きさまで見る（中身の照らしは置くときに済ませてある）

        名前だけを見ると、手で置いた別の物や 0 バイトのファイルを「入っている」と見て、
        読み込む所で初めて落ちる
        """
        try:
            return self.path(root).stat().st_size == self.size
        except OSError:
            return False


def models_root() -> Path:
    """落とした物の置き場"""
    return userdirs.data_root() / MODELS_DIR


def missing_files(files: Sequence[RemoteFile], root: Path | None = None) -> tuple[RemoteFile, ...]:
    """まだ置いていない物"""
    return tuple(file for file in files if not file.present(root))


class HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """転送先も HTTPS に限る

    既定の転送は https から http へも従うので、最初の URL だけを確かめても、転送の後は平文で
    つなぎ、名乗り（版）を載せた頼みが外から読める 中身は sha256 で照らすので差し替えた物は
    置かないが、つなぐ先そのものを HTTPS に限る約束（:func:`_fetch`）を転送の後も守る
    断りは ``URLError``（``OSError`` の仲間）なので、:func:`_fetch` が理由として返す
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if urlsplit(newurl).scheme != "https":
            raise urllib.error.URLError(f"HTTPS でない転送先は使いません（{newurl}）")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


#: 転送先を確かめる開き方 :func:`_open` が使う
_OPENER = urllib.request.build_opener(HttpsOnlyRedirect)


def _open(request: urllib.request.Request) -> IO[bytes]:
    response: IO[bytes] = _OPENER.open(request, timeout=TIMEOUT_SECONDS)
    return response


def fetch_files(
    files: Sequence[RemoteFile],
    *,
    root: Path | None = None,
    on_output: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    opener: Opener | None = None,
) -> int:
    """置いていない物を落として置く 戻り値は導入と同じ終了コード（0 が成功）

    置いてある物は落とし直さない 1 つでも落とせなければ 1 を返し、理由を ``on_output`` へ書く
    """
    write = on_output if on_output is not None else _ignore
    place = root if root is not None else models_root()
    for file in missing_files(files, place):
        if should_cancel is not None and should_cancel():
            write("中断した")
            return 1
        reason = _fetch(file, place, write, should_cancel, opener or _open)
        if reason is not None:
            write(f"{file.label or file.name} を落とせませんでした: {reason}")
            return 1
    return 0


def _fetch(
    file: RemoteFile,
    place: Path,
    write: Callable[[str], None],
    should_cancel: Callable[[], bool] | None,
    opener: Opener,
) -> str | None:
    """1 つを落として照らして置く 置けなければ理由を返す"""
    if urlsplit(file.url).scheme != "https":
        # 平文で取ると、途中で差し替えられても気付くのは照らしの所だけになる
        # 照らしで止まるとはいえ、つなぐ先そのものを HTTPS に限っておく
        return f"HTTPS でない取得元は使いません（{file.url}）"
    from sashimono import __version__

    write(f"> {file.url}")
    request = urllib.request.Request(
        file.url, headers={"User-Agent": f"SashimonoEdit/{__version__} (model-download)"}
    )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    received = 0
    try:
        with opener(request) as response:
            while True:
                if should_cancel is not None and should_cancel():
                    return "中断した"
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                received += len(chunk)
                if received > file.size:
                    return f"大きさが {file.size} バイトを超えました 取得元が変わった恐れがあります"
                digest.update(chunk)
                chunks.append(chunk)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        # 繋がらない・途中で切れた（IncompleteRead は OSError ではない）・URL が壊れている
        return str(exc) or type(exc).__name__
    if digest.hexdigest() != file.sha256:
        return (
            "中身が決めた物と違います（sha256 が合わない） 取得元が差し替えられたか、"
            "途中で壊れました 置かずに止めます"
        )
    # 書く所でだけ読む 導入の仕組み（runtime）は起動の頭で読まれ、ここで作品の入出力の
    # 部品（core.io）まで一緒に読むと、編集画面が出るまでが延びる
    from sashimono.core.io.atomic import write_atomic_bytes

    target = file.path(place)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_bytes(target, b"".join(chunks))
    except OSError as exc:
        return f"{target} へ書けません: {exc}"
    write(f"{file.label or file.name} を置きました（{received} バイト）")
    return None


def _ignore(_: str) -> None:
    pass
