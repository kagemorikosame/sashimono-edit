"""素材を読み込む操作では、覚えた調べた結果を使わずに開き直すこと（#227 のレビュー）

調べた結果を覚えるのは、再生やシークのたびにデコーダが素材を調べ直すのを軽くするため
中身の印は真ん中だけの書き換え（同じ大きさ・時刻を戻す）を見分けられないので、読み込み
直しでまで覚えた結果を返すと、差し替えた素材を前の中身で置く 窓は表示しない
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.model import Project
from sashimono.engine.decode.probe import clear_probe_cache
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from tests.media_fixtures import SampleMedia


@pytest.fixture(autouse=True)
def fresh_cache() -> None:
    clear_probe_cache()


def _count_opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []
    real = av.open

    def counting(target: str, *args: object, **kwargs: object) -> object:
        opened.append(target)
        return real(target, *args, **kwargs)  # type: ignore[call-overload]  # 数えるだけの素通し

    monkeypatch.setattr(av, "open", counting)
    return opened


@pytest.fixture
def window(qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    # 控えと解析は頼まない どちらも素材を開くので、数える回数が読み込みの分だけにならない
    monkeypatch.setattr(created, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(created._analyzer, "request", lambda _media, **_kwargs: None)
    monkeypatch.setattr(created, "_match_project_to", lambda _media: None)
    yield created
    created.close()


def test_importing_again_opens_the_file_again(
    window: MainWindow, sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2 度目の読み込みで開き直さないと、差し替えたファイルの長さや音声の本数が古いまま使われる
    opened = _count_opens(monkeypatch)
    for _ in range(2):
        window.import_media([sample_av.path])
        assert window.wait_for_imports()
    assert len(opened) == 2


def test_template_media_is_opened_again(
    sample_av: SampleMedia, monkeypatch: pytest.MonkeyPatch
) -> None:
    # テンプレートの素材も読み込む操作 前に置いた同じ場所の素材の結果を使わない
    opened = _count_opens(monkeypatch)
    main_window_module._probe_or_none(sample_av.path)
    main_window_module._probe_or_none(sample_av.path)
    assert len(opened) == 2


def test_a_missing_template_file_is_none(tmp_path: Path) -> None:
    # 覚えた結果を捨てる所で無いファイルに当たっても、開けない素材として数えるだけ
    assert main_window_module._probe_or_none(tmp_path / "無い.mp4") is None
