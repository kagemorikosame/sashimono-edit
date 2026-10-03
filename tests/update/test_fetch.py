"""転送の追い方と、落とした物の照合 ネットワークは偽物（本物の GitHub へは出ない）"""

from __future__ import annotations

import hashlib
import http.client
import http.server
import io
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from sashimono.update.fetch import (
    ALLOWED_HOSTS,
    MAX_REDIRECTS,
    FetchError,
    MemoryTransport,
    RawResponse,
    UrllibTransport,
    check_url,
    download,
    fetch_bytes,
)

LATEST = "https://github.com/owner/repo/releases/latest/download/update.json"
TAGGED = "https://github.com/owner/repo/releases/download/v1.2.3/update.json"
CDN = "https://release-assets.githubusercontent.com/github-production-release-asset/1/abc"


@pytest.fixture
def transport() -> MemoryTransport:
    return MemoryTransport()


class TestTheHosts:
    def test_github_and_its_download_hosts(self) -> None:
        """固定の URL の持ち主と、リリースの資産の転送先（今と前の CDN）"""
        assert {
            "github.com",
            "objects.githubusercontent.com",
            "release-assets.githubusercontent.com",
        } == ALLOWED_HOSTS

    @pytest.mark.parametrize(
        "url",
        [
            "http://github.com/owner/repo/releases/latest/download/update.json",
            "https://evil.example.com/update.json",
            "https://github.com.evil.example.com/update.json",
            "https://user:pass@github.com/update.json",
            "https://github.com:8443/update.json",
            "ftp://github.com/update.json",
            "file:///C:/update.json",
        ],
    )
    def test_anything_else_is_refused(self, url: str) -> None:
        """平文・別のホスト・似た名前・利用者名つき・別の口は追わない"""
        with pytest.raises(FetchError):
            check_url(url)


class TestFollowingRedirects:
    def test_it_follows_to_the_cdn(self, transport: MemoryTransport) -> None:
        """``latest`` はタグへ、タグは CDN へ転送される 追わないと空を掴む"""
        transport.redirects[LATEST] = TAGGED
        transport.redirects[TAGGED] = CDN
        transport.pages[CDN] = b"{}"
        assert fetch_bytes(LATEST, transport, limit=100) == b"{}"
        assert transport.requested == [LATEST, TAGGED, CDN]

    def test_a_relative_redirect_is_resolved(self, transport: MemoryTransport) -> None:
        transport.redirects[LATEST] = "/owner/repo/releases/download/v1.2.3/update.json"
        transport.pages[TAGGED] = b"{}"
        assert fetch_bytes(LATEST, transport, limit=100) == b"{}"

    @pytest.mark.parametrize(
        "target",
        [
            "https://evil.example.com/update.json",
            "http://objects.githubusercontent.com/update.json",
        ],
    )
    def test_it_stops_at_a_foreign_host(self, transport: MemoryTransport, target: str) -> None:
        """転送先が決めたホストでない・平文に落ちたら、そこへは行かない"""
        transport.redirects[LATEST] = target
        transport.pages[target] = b"{}"
        with pytest.raises(FetchError):
            fetch_bytes(LATEST, transport, limit=100)
        assert target not in transport.requested

    def test_a_loop_ends(self, transport: MemoryTransport) -> None:
        """輪になった転送で回り続けない"""
        transport.redirects[LATEST] = TAGGED
        transport.redirects[TAGGED] = LATEST
        with pytest.raises(FetchError):
            fetch_bytes(LATEST, transport, limit=100)
        assert len(transport.requested) == MAX_REDIRECTS + 1

    def test_a_redirect_without_a_target_fails(self, transport: MemoryTransport) -> None:
        transport.redirects[LATEST] = ""
        with pytest.raises(FetchError):
            fetch_bytes(LATEST, transport, limit=100)

    def test_a_missing_file_fails(self, transport: MemoryTransport) -> None:
        with pytest.raises(FetchError):
            fetch_bytes(LATEST, transport, limit=100)

    def test_a_big_answer_is_cut(self, transport: MemoryTransport) -> None:
        """目録の上限を超える返事は抱え込まない"""
        transport.pages[LATEST] = b"x" * 101
        with pytest.raises(FetchError):
            fetch_bytes(LATEST, transport, limit=100)


class TestDownloading:
    BODY = b"zip body" * 1000

    @pytest.fixture
    def served(self, transport: MemoryTransport) -> MemoryTransport:
        transport.redirects[TAGGED] = CDN
        transport.pages[CDN] = self.BODY
        return transport

    def _download(self, transport: MemoryTransport, target: Path, **changes: object) -> None:
        options: dict[str, object] = {
            "size": len(self.BODY),
            "sha256": hashlib.sha256(self.BODY).hexdigest(),
        }
        options.update(changes)
        download(TAGGED, transport, target, **options)  # type: ignore[arg-type]

    def test_a_matching_file_is_kept(self, served: MemoryTransport, tmp_path: Path) -> None:
        target = tmp_path / "package.zip"
        self._download(served, target)
        assert target.read_bytes() == self.BODY
        assert list(tmp_path.iterdir()) == [target]

    def test_a_wrong_hash_leaves_nothing(self, served: MemoryTransport, tmp_path: Path) -> None:
        """目録と SHA-256 が違う zip は、置き場に 1 バイトも残さない（中を開かせない）"""
        target = tmp_path / "package.zip"
        with pytest.raises(FetchError):
            self._download(served, target, sha256="0" * 64)
        assert list(tmp_path.iterdir()) == []

    def test_a_longer_file_stops_early(self, served: MemoryTransport, tmp_path: Path) -> None:
        """目録より大きい返事は、最後まで読まずに止める"""
        target = tmp_path / "package.zip"
        with pytest.raises(FetchError):
            self._download(served, target, size=10)
        assert list(tmp_path.iterdir()) == []

    def test_a_shorter_file_fails(self, served: MemoryTransport, tmp_path: Path) -> None:
        target = tmp_path / "package.zip"
        with pytest.raises(FetchError):
            self._download(served, target, size=len(self.BODY) + 1)
        assert list(tmp_path.iterdir()) == []

    def test_it_can_be_stopped(self, served: MemoryTransport, tmp_path: Path) -> None:
        target = tmp_path / "package.zip"
        with pytest.raises(FetchError):
            self._download(served, target, should_cancel=lambda: True)
        assert list(tmp_path.iterdir()) == []

    def test_a_foreign_cdn_is_not_used(self, transport: MemoryTransport, tmp_path: Path) -> None:
        transport.redirects[TAGGED] = "https://cdn.example.com/package.zip"
        transport.pages["https://cdn.example.com/package.zip"] = self.BODY
        with pytest.raises(FetchError):
            self._download(transport, tmp_path / "package.zip")


class _CutOff(io.RawIOBase):
    """途中で切れる本文 http.client は OSError ではなく IncompleteRead を投げる"""

    def __init__(self, head: bytes) -> None:
        self._head = head

    def read(self, size: int = -1) -> bytes:
        if self._head:
            head, self._head = self._head, b""
            return head
        raise http.client.IncompleteRead(b"", 100)


class _CuttingTransport(MemoryTransport):
    def get(self, url: str) -> RawResponse:
        response = super().get(url)
        if response.status == 200:
            return RawResponse(200, None, _CutOff(b""))  # type: ignore[arg-type]
        return response


class TestACutOffAnswer:
    """本文の途中で繋がりが切れても FetchError にする 起動時の確認は FetchError を黙って流す"""

    def test_a_small_file(self) -> None:
        transport = _CuttingTransport()
        transport.pages[LATEST] = b"{}"
        with pytest.raises(FetchError):
            fetch_bytes(LATEST, transport, limit=100)

    def test_a_download_leaves_nothing(self, tmp_path: Path) -> None:
        transport = _CuttingTransport()
        transport.pages[TAGGED] = b"body"
        target = tmp_path / "package.zip"
        with pytest.raises(FetchError):
            download(TAGGED, transport, target, size=1000, sha256="0" * 64)
        assert list(tmp_path.iterdir()) == []

    def test_the_check_stays_quiet(self) -> None:
        """確かめる所まで例外が漏れると、起動時の裏の確認が黙らずに落ちる"""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        from sashimono.links import STABLE_MANIFEST_URL
        from sashimono.update.check import Outcome, check_for_update

        key = Ed25519PrivateKey.generate()
        transport = _CuttingTransport()
        transport.pages[LATEST] = b"{}"
        transport.redirects[STABLE_MANIFEST_URL] = LATEST
        result = check_for_update("0.0.1", transport=transport, keys=[key.public_key()])
        assert result.outcome is Outcome.FAILED


class _Redirecting(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(302)
        self.send_header("Location", "https://evil.example.com/update.json")
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture
def local_server() -> Iterator[str]:
    """この機械の中だけで返事をする HTTP の相手 外へは出ない"""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Redirecting)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/update.json"
    finally:
        server.shutdown()
        server.server_close()


class TestTheRealTransport:
    def test_it_does_not_follow_by_itself(self, local_server: str) -> None:
        """urllib に転送を追わせない 追わせると、決めたホストかを確かめる前に行ってしまう"""
        response = UrllibTransport(timeout=5).get(local_server)
        assert response.status == 302
        assert response.location == "https://evil.example.com/update.json"

    def test_a_refused_connection_is_a_fetch_error(self) -> None:
        """繋がらないときは FetchError（起動時の確認はこれを黙って受け流す）"""
        with pytest.raises(FetchError):
            UrllibTransport(timeout=2).get("http://127.0.0.1:9/update.json")
