"""目録と zip を落とす 転送は自分で追い、行き先のホストを限る

urllib に転送を任せると、どこへ飛ばされても付いていく 自動更新の取り口がそのまま
攻撃の入口になるので、転送は 1 回ずつ受け取り、HTTPS で決めたホストの所だけ追う

ネットワークに出る所は :class:`Transport` の 1 か所 試験は偽物を渡し、本物の GitHub へは出ない
"""

from __future__ import annotations

import hashlib
import http.client
import io
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import urljoin, urlsplit

from sashimono import __version__

__all__ = [
    "ALLOWED_HOSTS",
    "MAX_REDIRECTS",
    "FetchError",
    "MemoryTransport",
    "RawResponse",
    "Transport",
    "UrllibTransport",
    "check_url",
    "download",
    "fetch_bytes",
]

#: 追ってよいホスト
#: ``github.com`` が目録の固定の URL の持ち主で、リリースの資産はそこから CDN へ転送される
#: 転送先は 2025 年に ``objects.githubusercontent.com`` から
#: ``release-assets.githubusercontent.com`` へ移った（2026-10-02 に cli/cli のリリース資産の
#: 転送先を見て確かめた） 古い方も残すのは、GitHub が戻したり振り分けたりしても、
#: 配った版が更新を見失わないため
#: ここを広げると、署名が守る範囲は変わらないが、どこから取ったかを言えなくなる
ALLOWED_HOSTS = frozenset(
    {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
)

#: 転送を追う回数の上限 ``latest`` → タグ → CDN の 2 回で足りる 輪になった転送で回り続けない
MAX_REDIRECTS = 5

#: 1 回の読み書きを待つ秒数 起動時の確認は裏で走るので長くてもよいが、止まったまま
#: 残らないように区切る
TIMEOUT_SECONDS = 30.0

#: 読む塊の大きさ
_CHUNK = 1024 * 1024

#: 本文を読む途中で起きうる失敗 途中で切れた返事（IncompleteRead）は http.client の例外で、
#: OSError ではない 拾わないと FetchError にならず、起動時の確認が黙らずに落ちる
_READ_ERRORS = (OSError, http.client.HTTPException)


class FetchError(Exception):
    """取れなかった（つながらない・転送先が違う・大きさや中身が合わない）"""


@dataclass
class RawResponse:
    """転送を追う前の 1 回ぶんの返事"""

    status: int
    #: 転送先（``Location``） 転送でなければ ``None``
    location: str | None
    body: BinaryIO | None = None

    def close(self) -> None:
        if self.body is not None:
            self.body.close()


class Transport(Protocol):
    """1 回だけ GET する 転送は追わない（追うのは :func:`_follow`）"""

    def get(self, url: str) -> RawResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib に転送を追わせない 追うかどうかはこちらで決める"""

    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


class UrllibTransport:
    """本物の取り口 標準ライブラリだけで HTTPS を話す

    証明書は Windows の証明書の置き場で確かめる（``ssl.create_default_context`` が読む）
    配布版に certifi を積まずに済み、会社の機械の独自の証明書も通る
    """

    def __init__(self, timeout: float = TIMEOUT_SECONDS) -> None:
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)

    def get(self, url: str) -> RawResponse:
        request = urllib.request.Request(
            url, headers={"User-Agent": f"SashimonoEdit/{__version__} (auto-update)"}
        )
        try:
            # 監査済み URL は check_url で HTTPS と決めたホストに限ってから渡す
            response = self._opener.open(request, timeout=self._timeout)
        except urllib.error.HTTPError as error:
            # 転送も 4xx も HTTPError で来る（転送を追わせていないため）
            location = error.headers.get("Location") if error.headers is not None else None
            error.close()
            return RawResponse(error.code, location)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            # 返事の頭が壊れている（BadStatusLine など）のは OSError ではない
            raise FetchError(f"つながらない: {exc}") from exc
        return RawResponse(response.status, None, response)


class MemoryTransport:
    """手元の表から返す取り口 ネットワークへ出ずに、取る道（転送・照合）を通すため

    自己診断（配布版の中で更新の部品が動くか）と試験が使う 表に無い URL は 404
    """

    def __init__(self) -> None:
        self.pages: dict[str, bytes] = {}
        self.redirects: dict[str, str] = {}
        #: 頼まれた URL の順 転送をどう追ったかを試験で見る
        self.requested: list[str] = []

    def get(self, url: str) -> RawResponse:
        self.requested.append(url)
        if url in self.redirects:
            return RawResponse(302, self.redirects[url])
        if url in self.pages:
            return RawResponse(200, None, io.BytesIO(self.pages[url]))
        return RawResponse(404, None)


def check_url(url: str) -> None:
    """追ってよい URL か HTTPS で、決めたホストで、利用者名や別の番号の口が無い"""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise FetchError(f"HTTPS でない: {url}")
    host = (parts.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise FetchError(f"決めたホストでない: {host or url}")
    if parts.username is not None or parts.password is not None:
        raise FetchError(f"利用者名の付いた URL: {url}")
    try:
        port = parts.port
    except ValueError as exc:
        raise FetchError(f"番号の口が読めない: {url}") from exc
    if port not in (None, 443):
        raise FetchError(f"443 でない口: {url}")


def _follow(url: str, transport: Transport) -> RawResponse:
    """転送を追って、中身のある返事を返す 追うたびに行き先を確かめる"""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_url(current)
        response = transport.get(current)
        if response.status in (301, 302, 303, 307, 308):
            response.close()
            if not response.location:
                raise FetchError(f"転送先が書かれていない: {current}")
            # 相対の転送先もある（GitHub は使わないが、決まりでは許される）
            current = urljoin(current, response.location)
            continue
        if response.status != 200 or response.body is None:
            response.close()
            raise FetchError(f"取れない（{response.status}）: {current}")
        return response
    raise FetchError(f"転送が {MAX_REDIRECTS} 回を超えた: {url}")


def fetch_bytes(url: str, transport: Transport, *, limit: int) -> bytes:
    """小さな物（目録・署名）を丸ごと読む ``limit`` バイトを超えたら読むのをやめる

    上限を置かないと、壊れた・すり替わった返事で何 GB でも抱え込む
    """
    response = _follow(url, transport)
    try:
        assert response.body is not None
        data = response.body.read(limit + 1)
    except _READ_ERRORS as exc:
        raise FetchError(f"読めない: {exc}") from exc
    finally:
        response.close()
    if len(data) > limit:
        raise FetchError(f"{limit} バイトより大きい: {url}")
    return data


def download(
    url: str,
    transport: Transport,
    target: Path,
    *,
    size: int,
    sha256: str,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """``target`` へ落とし、大きさと SHA-256 を目録と照らす 合わなければ消して止まる

    書いている途中は別の名前にする 途中で落ちた物を、次の起動が落とし終えた物と
    取り違えないため 大きさは読みながら見る 目録より大きい返事は、最後まで読まずに止める
    """
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    total = 0
    response = _follow(url, transport)
    try:
        assert response.body is not None
        with partial.open("wb") as out:
            while True:
                if should_cancel is not None and should_cancel():
                    raise FetchError("止めた")
                chunk = response.body.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > size:
                    raise FetchError(f"目録の大きさ（{size} バイト）より大きい")
                digest.update(chunk)
                out.write(chunk)
        if total != size:
            raise FetchError(f"大きさが目録と違う（{total} / {size} バイト）")
        if digest.hexdigest() != sha256.lower():
            raise FetchError("SHA-256 が目録と違う")
        partial.replace(target)
    except BaseException as exc:
        partial.unlink(missing_ok=True)
        if isinstance(exc, _READ_ERRORS):
            raise FetchError(f"書けない・読めない: {exc}") from exc
        raise
    finally:
        response.close()
