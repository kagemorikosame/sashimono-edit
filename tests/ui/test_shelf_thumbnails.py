"""テンプレートの棚の一覧の見本（#277 利用者の決定で棚にも出す）

プリセット・エイリアスと同じ見本の係（作り置き・見えている物から・GPU の無い機械の扱い）を
使い回す 鍵は配布物の中身の指紋 読めない配布物は失敗の印を出し、読めなかった事実を
覚えて開くたびに読み直さない
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from sashimono.compat.catalog import TemplateCatalog, TemplateEntry
from sashimono.ui.library_thumbnails import THUMBNAILS_SIMPLE, LookThumbnails
from sashimono.ui.library_view import LibraryOptions
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.shelf_looks import shelf_look
from sashimono.ui.template_dialog import TemplateDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.compat.test_ymm4 import template as ymm4_template
from tests.compat.test_ymm4 import text_item, write_ymmt


def _alias(text: str) -> str:
    return "\n".join(
        [
            "[Object]",
            "frame=0,89",
            "[Object.0]",
            "effect.name=テキスト",
            "サイズ=80",
            f"テキスト={text}",
            "",
        ]
    )


def _shelf(root: Path, count: int = 1) -> Path:
    """AviUtl のエイリアス ``count`` 本と、YMM4 の 1 本入りと、壊れた YMM4 を置いた棚"""
    (root / "字幕").mkdir(parents=True)
    for number in range(count):
        (root / "字幕" / f"見出し{number:03}.object").write_text(_alias(f"見出し{number}"), "utf-8")
    write_ymmt(root / "束.ymmt", ymm4_template("テロップ", text_item()))
    (root / "壊れ.ymmt").write_bytes(b"not a zip")
    return root


def _wait(condition: Callable[[], bool], seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def _entry(root: Path, name: str) -> TemplateEntry:
    catalog = TemplateCatalog()
    catalog.scan((root,))
    found = catalog.find(name)
    assert found is not None
    return found


def _nodes(dialog: TemplateDialog) -> dict[str, QTreeWidgetItem]:
    return {node.text(0): node for node in dialog._leaves()}


@pytest.fixture
def thumbnails(tmp_path: Path) -> Iterator[LookThumbnails]:
    created = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_SIMPLE)
    yield created
    created.release()


def _dialog(root: Path, thumbnails: LookThumbnails, *, shelf: bool = True) -> TemplateDialog:
    options = LibraryOptions(thumbnails=thumbnails.mode, shelf=shelf)
    dialog = TemplateDialog(
        TemplateCatalog(), roots=(root,), thumbnails=thumbnails, options=options
    )
    dialog.show()
    QApplication.processEvents()
    return dialog


class TestFingerprint:
    def test_the_key_follows_the_content_not_the_place_or_time(self, tmp_path: Path) -> None:
        root = _shelf(tmp_path / "棚")
        first = shelf_look(_entry(root, "見出し000"))
        # 別の置き場へ写しただけ・時刻だけ変わっただけでは描き直さない
        copied = tmp_path / "写し"
        shutil.copytree(root, copied)
        path = copied / "字幕" / "見出し000.object"
        os.utime(path, (1_000_000_000, 1_000_000_000))
        assert shelf_look(_entry(copied, "見出し000")).fingerprint == first.fingerprint
        # 中身が変われば描き直す
        path.write_text(_alias("別の文字"), "utf-8")
        assert shelf_look(_entry(copied, "見出し000")).fingerprint != first.fingerprint

    def test_templates_in_one_ymm4_file_are_told_apart(self, tmp_path: Path) -> None:
        write_ymmt(
            tmp_path / "束.ymmt",
            ymm4_template("一", text_item()),
            ymm4_template("二", text_item()),
        )
        assert shelf_look(_entry(tmp_path, "一")).fingerprint != (
            shelf_look(_entry(tmp_path, "二")).fingerprint
        )


class TestShelf:
    def test_visible_templates_get_a_picture(
        self, tmp_path: Path, thumbnails: LookThumbnails
    ) -> None:
        root = _shelf(tmp_path / "棚")
        dialog = _dialog(root, thumbnails)
        try:
            look = shelf_look(_entry(root, "見出し000"))
            _wait(lambda: thumbnails.cached(look) is not None)
            found = thumbnails.cached(look)
            assert found is not None and found.image is not None
            ymm4 = shelf_look(_entry(root, "テロップ"))
            _wait(lambda: thumbnails.cached(ymm4) is not None)
            assert thumbnails.cached(ymm4) is not None
            # 今の大きな下絵も残る
            assert dialog._preview is not None
        finally:
            dialog.reject()
            shiboken6.delete(dialog)

    def test_an_unreadable_one_is_marked_and_not_read_again(
        self, tmp_path: Path, thumbnails: LookThumbnails
    ) -> None:
        root = _shelf(tmp_path / "棚")
        broken = shelf_look(_entry(root, "壊れ"))
        key = thumbnails.request(broken)
        assert key is not None
        _wait(lambda: thumbnails.thumbnail(key) is not None)
        found = thumbnails.thumbnail(key)
        assert found is not None and found.failed and found.image is None
        assert found.error

        # 次に開いたとき（新しい係）は、読み直さずに失敗の印を出す
        again = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_SIMPLE)
        try:
            remembered = again.cached(broken)
            assert remembered is not None and remembered.failed
            assert again._worker is None
        finally:
            again.release()

    def test_the_failure_mark_is_shown_in_the_list(
        self, tmp_path: Path, thumbnails: LookThumbnails
    ) -> None:
        root = _shelf(tmp_path / "棚")
        dialog = _dialog(root, thumbnails)
        try:
            node = next(n for name, n in _nodes(dialog).items() if name.startswith("壊れ"))
            look = shelf_look(_entry(root, "壊れ"))
            _wait(lambda: thumbnails.cached(look) is not None and bool(node.toolTip(0)))
            QApplication.processEvents()
            assert node.toolTip(0)
        finally:
            dialog.reject()
            shiboken6.delete(dialog)

    def test_only_visible_ones_are_drawn(self, tmp_path: Path, thumbnails: LookThumbnails) -> None:
        # 棚は数百本ある 全部を頼むと、開いてからしばらく描き続けて重くなる
        root = _shelf(tmp_path / "棚", count=200)
        started = time.perf_counter()
        dialog = _dialog(root, thumbnails)
        opened = time.perf_counter() - started
        try:
            # 頼むのは窓が出てから少し後（VISIBLE_DELAY_MS） その間を待つ
            def asked_nodes() -> list[QTreeWidgetItem]:
                role = Qt.ItemDataRole.UserRole + 1
                return [n for n in dialog._leaves() if n.data(0, role) is not None]

            _wait(lambda: bool(asked_nodes()), seconds=10)
            QApplication.processEvents()
            asked = asked_nodes()
            assert 0 < len(asked) < 100
            # 開く速さに見本が響かない（手元で 0.3 秒ほど 遅い CI でも窓が出る目安）
            assert opened < 10
        finally:
            dialog.reject()
            shiboken6.delete(dialog)

    def test_turning_it_off_draws_nothing(self, tmp_path: Path, thumbnails: LookThumbnails) -> None:
        root = _shelf(tmp_path / "棚")
        dialog = _dialog(root, thumbnails, shelf=False)
        try:
            QApplication.processEvents()
            assert thumbnails._worker is None
            assert all(node.icon(0).isNull() for node in dialog._leaves())
            assert not (tmp_path / "cache").exists()
        finally:
            dialog.reject()
            shiboken6.delete(dialog)


class TestPreference:
    def test_it_is_on_by_default_and_comes_back(self, tmp_path: Path) -> None:
        assert Preferences().library_options.shelf
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(shelf_thumbnails=False))
        assert store.load().library_options.shelf is False

    def test_the_dialog_offers_it(self) -> None:
        dialog = PreferencesDialog(Preferences())
        try:
            assert dialog._shelf_thumbnails.isChecked()
            dialog._shelf_thumbnails.setChecked(False)
            assert dialog.preferences().shelf_thumbnails is False
        finally:
            shiboken6.delete(dialog)


#: 見本を描いて係を手放し、ごみ集めを回して終わる台本 引数は置き場と、棚の置き場（省略可）
_EXIT_SCRIPT = """
import gc
import sys
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
from sashimono.compat.catalog import TemplateCatalog
from sashimono.core.io import Preset
from sashimono.core.model import Effect
from sashimono.effects.sources import TEXT
from sashimono.ui.library_thumbnails import LookThumbnails
from sashimono.ui.shelf_looks import shelf_look
thumbs = LookThumbnails(Path(sys.argv[1]), mode="full")
items = [
    Preset(name=str(n), source=TEXT.create(text=str(n)), effects=(Effect(kind="glow"),))
    for n in range(5)
]
if len(sys.argv) > 2:
    items += [shelf_look(e) for e in TemplateCatalog().scan((Path(sys.argv[2]),))]
keys = [thumbs.request(item) for item in items]
deadline = time.monotonic() + 300
while any(thumbs.thumbnail(k) is None for k in keys) and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.01)
thumbs.release()
gc.collect()
print("ok", all(thumbs.thumbnail(k) is not None for k in keys), flush=True)
"""

#: 手元の AviUtl2 のエイリアス（配布物 リポジトリには入れない）
_REAL_ALIASES = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")) / "aviutl2" / "Alias"


@pytest.mark.usefixtures("gpu")
@pytest.mark.parametrize(
    "shelf",
    [
        pytest.param(None, id="プリセットだけ"),
        pytest.param(
            _REAL_ALIASES,
            id="手元の AviUtl2 のエイリアス",
            marks=pytest.mark.skipif(
                not _REAL_ALIASES.is_dir(), reason="AviUtl2 のエイリアスが置かれていない"
            ),
        ),
    ],
)
def test_the_process_ends_cleanly_after_drawing(tmp_path: Path, shelf: Path | None) -> None:
    # 配布物の DLL のモジュールを読んだ Lua を、描画係と一緒に手放すと、ごみ集めの中で
    # アクセス違反で落ちた（AviUtl2 のエイリアスを続けて描いた後）
    script = tmp_path / "exit.py"
    script.write_text(_EXIT_SCRIPT, encoding="utf-8")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    arguments = [sys.executable, str(script), str(tmp_path / "cache")]
    if shelf is not None:
        arguments.append(str(shelf))
    result = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=600,
        check=False,
    )
    assert result.stdout.strip() == "ok True", result.stderr[-2000:]
    assert result.returncode == 0
