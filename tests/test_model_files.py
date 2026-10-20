"""pip では入らないファイル（画面の解析の顔の検出のモデル #324）を導入のボタンで落とす

本物の取得元へは出ない 頼みを開く所（opener）を偽物に替え、返す中身を決める
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
from pathlib import Path
from typing import IO

import pytest

from sashimono import model_files, runtime
from sashimono.model_files import RemoteFile, fetch_files, missing_files
from sashimono.runtime import FeaturePack, install_runtime

BODY = b"model-bytes" * 100


def _file(body: bytes = BODY, *, url: str = "https://example.invalid/m.onnx") -> RemoteFile:
    return RemoteFile(
        name="m.onnx",
        url=url,
        sha256=hashlib.sha256(body).hexdigest(),
        size=len(body),
        label="試験のモデル",
    )


class _Opener:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> IO[bytes]:
        self.requests.append(request)
        return io.BytesIO(self.body)


class TestFetch:
    def test_the_file_is_placed_after_its_hash_matches(self, tmp_path: Path) -> None:
        # 壊れると、導入の後も顔の検出のモデルが無く、AI が顔を調べようとすると毎回断られる
        file = _file()
        opener = _Opener(BODY)
        lines: list[str] = []
        assert fetch_files((file,), root=tmp_path, opener=opener, on_output=lines.append) == 0
        assert (tmp_path / "m.onnx").read_bytes() == BODY
        assert file.present(tmp_path)
        assert missing_files((file,), tmp_path) == ()
        # 名乗りを付ける（通信する所の一覧に書いた物と同じ）
        assert "model-download" in str(opener.requests[0].get_header("User-agent"))
        assert any("置きました" in line for line in lines)

    def test_a_changed_file_is_not_placed(self, tmp_path: Path) -> None:
        # 壊れると、取得元が差し替えられたり途中で壊れたりした物を、そのまま読み込んで動かす
        file = _file()
        lines: list[str] = []
        code = fetch_files(
            (file,), root=tmp_path, opener=_Opener(b"X" + BODY[1:]), on_output=lines.append
        )
        assert code == 1
        assert not (tmp_path / "m.onnx").exists()
        assert any("sha256" in line for line in lines)

    def test_a_larger_file_is_cut_off(self, tmp_path: Path) -> None:
        # 壊れると、取得元を取り違えたときに大きな物を延々と落とし続ける
        code = fetch_files((_file(),), root=tmp_path, opener=_Opener(BODY * 3))
        assert code == 1
        assert not (tmp_path / "m.onnx").exists()

    def test_plain_http_is_refused(self, tmp_path: Path) -> None:
        # 壊れると、途中で差し替えられうる平文の取得元へつなぐ
        opener = _Opener(BODY)
        code = fetch_files(
            (_file(url="http://example.invalid/m.onnx"),), root=tmp_path, opener=opener
        )
        assert code == 1
        assert opener.requests == []

    def test_a_placed_file_is_not_fetched_again(self, tmp_path: Path) -> None:
        # 壊れると、導入のボタンを押すたびに同じモデルを落とし直す
        file = _file()
        (tmp_path / "m.onnx").write_bytes(BODY)
        opener = _Opener(BODY)
        assert fetch_files((file,), root=tmp_path, opener=opener) == 0
        assert opener.requests == []

    def test_a_wrong_sized_file_counts_as_missing(self, tmp_path: Path) -> None:
        # 壊れると、途中で切れた（0 バイトの）ファイルを「入っている」と見て、読む所で落ちる
        file = _file()
        (tmp_path / "m.onnx").write_bytes(b"")
        assert not file.present(tmp_path)
        assert missing_files((file,), tmp_path) == (file,)

    def test_a_network_failure_is_reported(self, tmp_path: Path) -> None:
        # 壊れると、繋がらない機械で導入が例外で落ち、ログに何が起きたかが残らない
        def broken(_request: urllib.request.Request) -> IO[bytes]:
            raise OSError("繋がらない")

        lines: list[str] = []
        assert fetch_files((_file(),), root=tmp_path, opener=broken, on_output=lines.append) == 1
        assert any("繋がらない" in line for line in lines)

    def test_cancelling_stops_before_placing(self, tmp_path: Path) -> None:
        # 壊れると、導入の欄で中断を押しても落とし続け、半端な物を置く
        code = fetch_files(
            (_file(),), root=tmp_path, opener=_Opener(BODY), should_cancel=lambda: True
        )
        assert code == 1
        assert not (tmp_path / "m.onnx").exists()


class TestPackStatus:
    def test_a_missing_file_keeps_the_pack_from_being_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、包みだけ入ってモデルの無い機械で「導入済み」と出て、使う所で初めて落ちる
        monkeypatch.setattr(model_files, "models_root", lambda: tmp_path)
        pack = FeaturePack(key="x", label="x", required=("pip",), files=(_file(),))
        status = pack.status()
        assert status.installed
        assert not status.ready
        assert "試験のモデル" in status.summary()
        (tmp_path / "m.onnx").write_bytes(BODY)
        assert pack.status().ready


class TestInstall:
    def test_the_file_is_fetched_after_pip_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、導入のボタンで包みは入るのにモデルが落ちず、画面の解析が使えないまま残る
        monkeypatch.setattr(model_files, "models_root", lambda: tmp_path)
        asked: list[tuple[RemoteFile, ...]] = []

        def fetch(files: tuple[RemoteFile, ...], **_options: object) -> int:
            asked.append(files)
            return 0

        monkeypatch.setattr(runtime, "fetch_files", fetch)
        pack = FeaturePack(key="x", label="x", required=("pip",), files=(_file(),))
        code = install_runtime(pack, command=[sys.executable, "-c", "pass"])
        assert code == 0
        assert asked == [pack.files]

    def test_nothing_is_fetched_when_pip_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 壊れると、包みが入らなかったのに使えないモデルだけを落として残す
        asked: list[object] = []

        def fetch(*arguments: object, **_options: object) -> int:
            asked.append(arguments)
            return 0

        monkeypatch.setattr(runtime, "fetch_files", fetch)
        pack = FeaturePack(key="x", label="x", required=("pip",), files=(_file(),))
        code = install_runtime(pack, command=[sys.executable, "-c", "raise SystemExit(3)"])
        assert code == 3
        assert asked == []

    def test_a_failed_fetch_fails_the_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 壊れると、モデルを落とせなかったのに「導入が完了しました」と出る
        monkeypatch.setattr(runtime, "fetch_files", lambda *_a, **_k: 1)
        pack = FeaturePack(key="x", label="x", required=("pip",), files=(_file(),))
        assert install_runtime(pack, command=[sys.executable, "-c", "pass"]) == 1
