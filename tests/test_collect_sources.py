"""リリースのときに GPL / LGPL の部品のソースを集める道具（tools/collect_sources.py）

本物の置き場からは落とさない（qtwebengine だけで 500 MB を超える） 手元で HTTP の
置き場を立てて、落とし方・照合・後始末を確かめる
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parent.parent

ARCHIVE = b"source archive body " * 100
PAGE = b"<!doctype html><title>Making sure you're not a bot!</title>"


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "collect_sources", ROOT / "tools" / "collect_sources.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Handler(BaseHTTPRequestHandler):
    """``/archive`` はアーカイブ、``/page`` はボット避けの HTML、``/nohead`` は HEAD を断る"""

    #: 届いた頼み（方法、道） 置き場はスレッドをまたぐので、インスタンスではなく型に持つ
    requests: ClassVar[list[tuple[str, str]]] = []

    def _answer(self, body: bool) -> None:
        type(self).requests.append((self.command, self.path))
        if self.path == "/nohead" and self.command == "HEAD":
            self.send_response(403)
            self.end_headers()
            return
        if self.path in ("/archive", "/nohead"):
            payload, kind = ARCHIVE, "application/x-gzip"
        elif self.path == "/page":
            payload, kind = PAGE, "text/html; charset=utf-8"
        elif self.path == "/xhtml":
            payload, kind = PAGE, "Application/XHTML+XML; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if body:
            self.wfile.write(payload)

    def do_GET(self) -> None:
        self._answer(body=True)

    def do_HEAD(self) -> None:
        self._answer(body=False)

    def log_message(self, format: str, *args: object) -> None:
        # 試験の出力を置き場の記録で埋めない
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _source(tool: ModuleType, url: str, sha256: str | None = None) -> object:
    return tool.Source("sample", "1.0", url, sha256, "GPL-2.0-or-later", "sample-1.0.tar.gz")


class TestCollect:
    def test_it_writes_the_archive_and_the_manifest(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        """落とした物と一緒に、取得元と sha256 を残す

        残さないと、添付したソースが積んだ版の物かを後から確かめられない
        """
        expected = hashlib.sha256(ARCHIVE).hexdigest()
        manifest = tool.collect([_source(tool, f"{server}/archive", expected)], tmp_path)
        assert (tmp_path / "sample-1.0.tar.gz").read_bytes() == ARCHIVE
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        assert entries[0]["sha256"] == expected
        assert entries[0]["url"] == f"{server}/archive"
        assert entries[0]["verified_against_upstream"] is True
        sums = (tmp_path / tool.SUMS_NAME).read_text(encoding="utf-8")
        assert sums == f"{expected}  sample-1.0.tar.gz\n"

    def test_an_unpinned_archive_records_what_came(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        # 上流が sha256 を出していない物も、落とした時点の値を残す
        manifest = tool.collect([_source(tool, f"{server}/archive")], tmp_path)
        entry = json.loads(manifest.read_text(encoding="utf-8"))[0]
        assert entry["sha256"] == hashlib.sha256(ARCHIVE).hexdigest()
        assert entry["verified_against_upstream"] is False

    def test_a_wrong_checksum_leaves_nothing(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        """上流の公開値と合わなければ止まり、何も残さない

        残すと、中身の違うソースを「積んだ版の物」として添付する
        """
        with pytest.raises(RuntimeError, match="sha256"):
            tool.download(_source(tool, f"{server}/archive", "0" * 64), tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_a_bot_page_is_not_taken_for_the_source(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        # ボット避けの画面は 200 で返る 数えると HTML をソースとして添付する
        with pytest.raises(RuntimeError, match="HTML"):
            tool.download(_source(tool, f"{server}/page"), tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_an_xhtml_page_in_any_case_is_refused(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        # 型の名前は大文字小文字を区別しない 小文字の text/html だけを見ると素通りする
        with pytest.raises(RuntimeError, match="HTML"):
            tool.download(_source(tool, f"{server}/xhtml"), tmp_path)

    def test_a_rejected_archive_does_not_survive_a_failed_refetch(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        """公開値と合わない物は、落とし直しに失敗しても残さない

        残すと完成品の名前のまま、次にフォルダを丸ごと添付したときに一緒に配る
        """
        (tmp_path / "sample-1.0.tar.gz").write_bytes(b"old")
        with pytest.raises(RuntimeError, match="sha256"):
            tool.download(_source(tool, f"{server}/archive", "0" * 64), tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_a_finished_archive_is_not_fetched_again(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        """落とし終えた物は落とし直さない qtwebengine は 500 MB を超える"""
        (tmp_path / "sample-1.0.tar.gz").write_bytes(ARCHIVE)
        source = _source(tool, f"{server}/archive", hashlib.sha256(ARCHIVE).hexdigest())
        tool.download(source, tmp_path)
        assert _Handler.requests == []

    def test_an_unpinned_archive_is_reused_only_as_recorded(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        """公開値の無い物は、前の manifest に同じ取得元と sha256 で残っているときだけ使い回す

        名前だけで使い回すと、取得元を変えたときに前の中身を新しい取得元の物として記録する
        """
        (tmp_path / "sample-1.0.tar.gz").write_bytes(ARCHIVE)
        tool.collect([_source(tool, f"{server}/archive")], tmp_path)
        assert [path for _, path in _Handler.requests] == ["/archive"], (
            "manifest が無いのに使い回した"
        )

        _Handler.requests.clear()
        tool.collect([_source(tool, f"{server}/archive")], tmp_path)
        assert _Handler.requests == [], "記録どおりの物を落とし直した"

        # 取得元が変わったら落とし直す
        tool.collect([_source(tool, f"{server}/nohead")], tmp_path)
        assert [path for _, path in _Handler.requests] == ["/nohead"]

    def test_leftovers_are_removed(self, tool: ModuleType, server: str, tmp_path: Path) -> None:
        """一覧に無い物（前の版のソース）はフォルダから消す

        残すと、フォルダを丸ごと Release へ添付したときに古いソースまで配る
        """
        (tmp_path / "old-0.9.tar.gz").write_bytes(b"old")
        tool.collect([_source(tool, f"{server}/archive")], tmp_path)
        assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
            ["sample-1.0.tar.gz", tool.SUMS_NAME, tool.MANIFEST_NAME]
        )

    def test_a_stale_archive_is_fetched_again(
        self, tool: ModuleType, server: str, tmp_path: Path
    ) -> None:
        # 公開値と合わない物（前の版の残りなど）は落とし直す
        (tmp_path / "sample-1.0.tar.gz").write_bytes(b"old")
        source = _source(tool, f"{server}/archive", hashlib.sha256(ARCHIVE).hexdigest())
        tool.download(source, tmp_path)
        assert (tmp_path / "sample-1.0.tar.gz").read_bytes() == ARCHIVE


class TestCheck:
    def test_a_reachable_archive_passes(self, tool: ModuleType, server: str) -> None:
        # 届く置き場で止まるか中身まで落とすと、
        # リリースする人の --check が届いているのに NG を出すか、確かめるだけで数百 MB を落とす
        assert tool.check_urls([_source(tool, f"{server}/archive")]) == []
        # 確かめるだけで中身は落とさない
        assert [method for method, _ in _Handler.requests] == ["HEAD"]

    def test_a_server_refusing_head_is_asked_for_one_byte(
        self, tool: ModuleType, server: str
    ) -> None:
        """Bitbucket の置き場は HEAD に 403 を返す それだけで届かないと数えない"""
        assert tool.check_urls([_source(tool, f"{server}/nohead")]) == []

    def test_a_missing_archive_is_reported(self, tool: ModuleType, server: str) -> None:
        # 消えた置き場を見落とすと、--check が [ok] を出し、
        # リリースする人が入手先の無くなった部品に気付かないまま進む
        problems = tool.check_urls([_source(tool, f"{server}/gone")])
        assert len(problems) == 1 and "/gone" in problems[0]

    def test_a_bot_page_is_reported(self, tool: ModuleType, server: str) -> None:
        # ボット避けの画面を届いたと数えると、--check が通り、
        # 入手先からソースを落とせないことにリリースする人が気付けない
        assert tool.check_urls([_source(tool, f"{server}/page")]) != []


def _pyside(root: Path, *names: str) -> Path:
    pyside = root / "Sashimono" / "_internal" / "PySide6"
    for name in names:
        (pyside / name).parent.mkdir(parents=True, exist_ok=True)
        (pyside / name).write_bytes(b"")
    return pyside


class TestOnlyWhatIsBundled:
    """添付する Qt のソースは、zip に積んだ Qt のファイルから決める

    決め打ちで全部添付すると、積んでいない qtwebengine（580 MB）まで毎回落とす
    """

    def test_the_modules_come_from_the_files(self, tool: ModuleType, tmp_path: Path) -> None:
        # 積んだファイルから Qt のモジュールを数え損ねると、
        # 積んだ qtsvg などのソースを Release へ添付し損ねるか、Qt でない DLL で道具が止まる
        pyside = _pyside(
            tmp_path,
            "Qt6Core.dll",
            "Qt6Svg.dll",
            "plugins/imageformats/qwebp.dll",
            "translations/qt_ja.qm",
            "QtCore.pyd",
            "pyside6.abi3.dll",
            "opengl32sw.dll",
            "MSVCP140.dll",
        )
        modules, unknown = tool.bundled_qt_modules(pyside)
        assert modules == {"qtbase", "qtsvg", "qtimageformats", "qttranslations"}
        assert unknown == []

    def test_an_unshipped_module_is_not_attached(self, tool: ModuleType) -> None:
        # 積んだ Qt で絞れないと、積んでいない qtwebengine（580 MB）までリリースのたびに落とすか、
        # Qt に依らない x264 などのソースが添付から抜ける
        attached = {source.qt_module for source in tool.sources_for({"qtbase"})}
        assert "qtwebengine" not in attached
        assert "qtbase" in attached
        # Qt ではない部品（FFmpeg や x264）は積んだ Qt に関係なく添付する
        assert any(source.name == "x264" for source in tool.sources_for(set()))

    def test_a_pdf_plugin_brings_webengine(self, tool: ModuleType, tmp_path: Path) -> None:
        # Qt6Pdf を積んだら、そのソースの入っている qtwebengine も添付しなければならない
        pyside = _pyside(tmp_path, "Qt6Pdf.dll", "plugins/imageformats/qpdf.dll")
        modules, _ = tool.bundled_qt_modules(pyside)
        assert modules == {"qtwebengine"}

    def test_an_unknown_qt_file_stops_it(self, tool: ModuleType, tmp_path: Path) -> None:
        """どのモジュールの物か分からない Qt の DLL があれば止める

        黙って通すと、そのモジュールのソースを添付し損ねたまま配る
        """
        pyside = _pyside(tmp_path, "Qt6Multimedia.dll")
        _, unknown = tool.bundled_qt_modules(pyside)
        assert unknown == ["Qt6Multimedia.dll"]

    def test_it_needs_a_built_bundle(
        self, tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 組み立てる前に走らせると、何を添付すればよいか決められない
        monkeypatch.setattr(tool, "installed_mismatches", list)
        assert tool.main(["--check"], dist=tmp_path) == 1

    def test_every_module_has_a_source(self, tool: ModuleType) -> None:
        # 表に書いたモジュールのどれにも、落とす先がある
        assert set(tool.QT_FILE_MODULES.values()) | {"qttranslations"} <= set(tool.QT_MODULES)


class TestTheList:
    def test_the_gpl_parts_are_all_there(self, tool: ModuleType) -> None:
        """GPL と LGPL の部品のうち、ソースを渡さなければならない物が全部ある"""
        names = {source.name.split()[0] for source in tool.SOURCES}
        assert {"FFmpeg", "pyav-ffmpeg", "x264", "x265", "LAME", "libiconv", "Qt"} <= names
        assert any(source.name.startswith("PySide6") for source in tool.SOURCES)

    def test_every_source_is_fetched_over_https(self, tool: ModuleType) -> None:
        # http だと転送の途中で差し替えられても気付けない（公開値で照合できない物もある）
        assert all(source.url.startswith("https://") for source in tool.SOURCES)

    def test_the_qt_folder_follows_the_version(self, tool: ModuleType) -> None:
        # Qt の置き場は major.minor のフォルダの下 版を上げてここだけ古いと取得先が壊れる
        series = ".".join(tool.QT_VERSION.split(".")[:2])
        for source in tool.SOURCES:
            if source.qt_module is not None:
                assert f"/qt/{series}/{tool.QT_VERSION}/" in source.url

    def test_file_names_do_not_collide(self, tool: ModuleType) -> None:
        # 名前がぶつかると、後から落とした物が前の物を黙って上書きする
        names = [source.filename for source in tool.SOURCES]
        assert len(names) == len(set(names))
        assert all("/" not in name and "\\" not in name for name in names)

    def test_the_notices_say_where_each_one_comes_from(self, tool: ModuleType) -> None:
        """一覧（THIRD_PARTY_NOTICES.md）にも同じ部品と版が載っている

        道具だけ直して一覧を直し忘れると、使う人に見せる入手先と添付する物が食い違う
        """
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for source in tool.SOURCES:
            assert source.version in notices, source.name
            assert source.name.split()[0] in notices, source.name

    def test_a_version_mismatch_stops_the_tool(
        self, tool: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 版が食い違ったまま先へ進むと、
        # --check が zip に積んだのと別の版の入手先を確かめて [ok] を出す
        monkeypatch.setattr(tool, "installed_mismatches", lambda: ["積んでいる FFmpeg は 9.0"])
        called: list[object] = []

        def check(sources: object) -> list[str]:
            called.append(sources)
            return []

        monkeypatch.setattr(tool, "check_urls", check)
        assert tool.main(["--check"]) == 1
        assert called == [], "版が食い違っているのに先へ進んだ"
