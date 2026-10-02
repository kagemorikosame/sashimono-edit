"""README の写真を撮る道具（tools/shots.py）

写真は手で撮ると、撮った人の画面配置や本人のファイル名が写り、画面が変わった
ときに同じ絵を作り直せない この道具はそこを引き受けているので、壊れると
README の写真が古いまま残る

GPU が要る検査は ``gpu`` フィクスチャを付けてある GPU の無い所（CI）では飛ぶ
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest
from PySide6.QtCore import QRect
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QLabel, QTreeWidget, QWidget

from sashimono.compat.aviutl import catalog as catalog_module
from sashimono.compat.aviutl.catalog import ScriptCatalog, script_catalog, set_script_catalog
from sashimono.compat.catalog import TemplateCatalog
from sashimono.core import userdirs
from sashimono.effects.definition import registry
from tests.media_fixtures import libx264_available

ROOT = Path(__file__).resolve().parents[2]

#: 見本のエイリアス 実配布物はリポジトリに入れないので、同じ書き方の最小の物を作る
SAMPLE_ALIAS = "\n".join(
    (
        "[Object]",
        "frame=0,89",
        "[Object.0]",
        "effect.name=テキスト",
        "サイズ=64.00",
        "文字色=ffee00",
        "テキスト=見本の字幕",
        "",
    )
)


@pytest.fixture(scope="module")
def shots() -> ModuleType:
    """道具を 1 つのモジュールとして読み込む インストールされた包みではない"""
    spec = importlib.util.spec_from_file_location("shots", ROOT / "tools" / "shots.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestShelfRoots:
    """棚の置き場を差し替えられること

    差し替えられないと、撮った機械に入っている配布物が丸ごと写真に写る
    """

    def test_the_dialog_lists_only_the_given_roots(
        self, shots: ModuleType, qt_application: QApplication, tmp_path: Path
    ) -> None:
        del shots, qt_application
        from sashimono.ui.template_dialog import TemplateDialog

        (tmp_path / "見本のエイリアス.object").write_text(SAMPLE_ALIAS, encoding="utf-8")
        dialog = TemplateDialog(TemplateCatalog(), roots=(tmp_path,))
        try:
            tree = dialog.findChild(QTreeWidget)
            assert tree is not None
            names: list[str] = []
            for index in range(tree.topLevelItemCount()):
                group = tree.topLevelItem(index)
                assert group is not None
                for position in range(group.childCount()):
                    child = group.child(position)
                    assert child is not None
                    names.append(child.text(0))
            assert names == ["見本のエイリアス"]
        finally:
            dialog.close()


class TestSampleTemplates:
    """写真に写すテンプレートは、道具が書いた見本だけ

    配布物は再配布の条件が作者ごとに違うので写さない 見本が読めなくなると、
    棚とテンプレートの写真が撮れなくなる
    """

    def test_every_sample_is_on_the_shelf(self, shots: ModuleType, tmp_path: Path) -> None:
        root = shots.write_sample_templates(tmp_path / "見本")
        names = {entry.name.rsplit("/", 1)[-1] for entry in TemplateCatalog().scan((root,))}
        expected = set(shots.SAMPLE_ALIASES) | {name for name, _ in shots.SAMPLE_YMM4}
        assert names == expected

    def test_every_sample_carries_text(self, shots: ModuleType, tmp_path: Path) -> None:
        # 文字を持たない見本は、着せる写真（自分で打った字幕に見た目を写す）に使えない
        root = shots.write_sample_templates(tmp_path / "見本")
        for entry in TemplateCatalog().scan((root,)):
            kinds = {
                inner.clip.source.kind
                for item in entry.load()
                for inner in item.walk()
                if inner.clip.source is not None
            }
            assert "text" in kinds, entry.name

    def test_the_named_samples_are_found(self, shots: ModuleType, tmp_path: Path) -> None:
        # README の本文が名前を引き合いに出している 見つからないと写真が撮れない
        root = shots.write_sample_templates(tmp_path / "見本")
        context = shots.Context(media=None, script_root=tmp_path / "s", template_root=root)
        for name in (shots.SAMPLE_ALIAS_NAME, shots.SAMPLE_YMM4_NAME):
            assert shots._named(shots.find_template(context, name), name)


class TestWorkFolder:
    def test_the_home_is_refused(self, shots: ModuleType, tmp_path: Path) -> None:
        """作業用のフォルダをホームの下に作らない

        作ると、ユーザー名を含む場所が素材や設定の置き場として画面に出たとき、
        そのまま写真に写る
        """
        home = tmp_path / "ホーム"
        (home / "下").mkdir(parents=True)
        with pytest.raises(shots.ShotError):
            shots.work_folder_base(home / "下", home)
        with pytest.raises(shots.ShotError):
            shots.work_folder_base(home, home)

    def test_a_folder_outside_the_home_is_used(self, shots: ModuleType, tmp_path: Path) -> None:
        outside = tmp_path / "外"
        outside.mkdir()
        assert shots.work_folder_base(outside, tmp_path / "ホーム") == outside.resolve()


class TestIsolation:
    def test_every_folder_the_app_reads_is_redirected(
        self, shots: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """設定・ProgramData・ホーム・一時フォルダのどれも作業用のフォルダへ向く

        ProgramData が漏れると、AviUtl2 の配布エイリアスとスクリプトが棚と一覧に
        並んで写真に写る ホームが漏れると、書き出し先の欄にユーザー名が出る
        """
        names = ("APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "USERPROFILE", "HOME", "TEMP", "TMP")
        for name in names:
            # 試験の後に元へ戻す印として、今の値を monkeypatch に覚えさせる
            monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
        base = tmp_path / "作業"
        shots.isolate_user_folders(base)
        for name in names:
            assert Path(os.environ[name]).is_relative_to(base), name
        assert Path(tempfile.gettempdir()).is_relative_to(base)


class TestMarking:
    def test_a_mark_at_the_edge_stays_visible(
        self, shots: ModuleType, qt_application: QApplication
    ) -> None:
        """窓の端の部品を囲んでも、枠の辺が画像の中に残ること

        広げたまま描くと、左の辺が画像の外（x が負）に出て、囲みが欠けて見える
        """
        del qt_application
        image = QImage(100, 100, QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        shots.mark(image, [QRect(0, 0, 50, 50)])
        column = [image.pixelColor(x, 25) for x in range(0, 5)]
        assert any(color.red() > 200 and color.blue() < 80 for color in column)


class TestSettling:
    def test_it_really_waits(
        self, shots: ModuleType, qt_application: QApplication, tmp_path: Path
    ) -> None:
        """撮る前の待ちが、本当に時間を使っていること

        ``processEvents`` に時間を渡しても、処理するイベントが尽きれば
        すぐ戻る 待っているつもりで待っていないと、解析の反映（250ms ごと）を
        1 度も通さないまま撮り、波形とサムネイルの無いタイムラインが写る
        """
        del tmp_path, qt_application
        widget = QWidget()
        rounds = 6
        started = time.monotonic()
        shots.settle(widget, rounds=rounds)
        elapsed = (time.monotonic() - started) * 1000

        # 端数で落ちないよう 8 割で見る 待っていなければ 1 ミリ秒も経たない
        assert elapsed >= rounds * shots.SETTLE_MS * 0.8


class TestPasting:
    def test_it_ignores_the_scaling_of_the_target(
        self, shots: ModuleType, qt_application: QApplication
    ) -> None:
        """拡大率の付いた画像へも、渡した画素の位置に貼れること

        ``QPainter`` は相手の QImage が持つ devicePixelRatio で座標を変換する
        1.0 に戻さずに貼ると、拡大率 125% の機械では GL の面が二重に拡大されて
        画面の外へ出て、プレビューの場所には何も写らない写真ができる
        """
        del qt_application
        image = QImage(200, 200, QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        image.setDevicePixelRatio(2.0)
        source = QImage(10, 10, QImage.Format.Format_RGB32)
        source.fill(QColor("red"))

        shots.paste(image, QRect(100, 100, 40, 40), source)

        # 渡したのは画素そのものの座標 2 倍に変換されると (200, 200) は画像の外で、
        # ここは黒いままになる
        assert image.pixelColor(120, 120) == QColor("red")


class TestSampleScript:
    def test_the_sample_script_becomes_a_settings_panel(
        self, shots: ModuleType, tmp_path: Path
    ) -> None:
        """制御行が設定欄になる これが AviUtl の写真で見せている所そのもの

        壊れると、README の AviUtl の写真に実際とは違う設定欄が写る
        """
        from sashimono.effects.definition import registry

        kind = shots.install_sample_script(tmp_path / "scripts")
        definition = registry.require(kind)
        assert [spec.label for spec in definition.parameters] == ["振れ幅", "速さ", "横に揺れる"]


class TestCompatibilityShot:
    def test_the_report_shot_hides_where_the_sample_lives(
        self,
        shots: ModuleType,
        qt_application: QApplication,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """見本のスクリプトは一時フォルダに置く その場所を出したまま撮ると、
        撮った人のユーザー名を含むフォルダが Wiki の写真に写る
        """
        del qt_application
        # ホームを試験の中で決める 見本はその下に置くので、撮った人の名前の代わりに
        # この名前が写っていないかで見られる
        home = tmp_path / "ホーム名は写らない"
        home.mkdir()
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.setenv("HOME", str(home))
        context = shots.Context(
            media=None, script_root=home / "scripts", template_root=home / "templates"
        )
        with _restored_scripts():
            dialog = shots.compatibility_dialog(context)
            try:
                shown = "\n".join(label.text() for label in dialog.findChildren(QLabel))
            finally:
                dialog.close()
        assert str(tmp_path) not in shown
        assert str(Path.home()) not in shown
        # 場所の文字列が無いだけでは、一部だけが写って名前が残っても通る 名前そのものも見る
        # ホームの名前は試験の中で決めた物（一時フォルダの名前）を使う 走らせる機械の
        # ホームの名前は、空（ホームが `/`）や `1` のように短いことがあり、写っていなくても
        # 画面の決まった文（「スクリプト 1 本」など）と一致してしまう
        assert home.name.casefold() not in shown.casefold()
        assert "スクリプト 1 本" in shown

    def test_scripts_in_the_default_folder_are_not_kept(
        self, qt_application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """後始末の基準に、既定の探索先のスクリプトを混ぜない

        一覧がまだ無い所で基準を取ると、既定の探索先を走査して登録した定義が基準に
        入り、試験のあとも登録簿に残る
        """
        del qt_application, tmp_path
        # 探索先は設定の置き場（conftest が一時フォルダへ向けている）だけにする
        monkeypatch.delenv("PROGRAMDATA", raising=False)
        monkeypatch.setattr(catalog_module, "_catalog", None)
        scripts = userdirs.config_root() / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "既定の置き場の見本.anm2").write_text("--track0:量,0,100,50\n", "utf-8")
        # 手助けに入るだけで何もしない 見るのは、手助け自身が基準を取るときに既定の
        # 探索先を走査しないか（前の手助けは走査して、見本を基準に入れて残していた）
        with _restored_scripts():
            pass
        assert not [d.kind for d in registry.all() if "既定の置き場の見本" in d.kind]
        # 置いた見本が本当に走査される物か 走査されない物なら、上の確かめは何も言っていない
        try:
            script_catalog()
            assert [d.kind for d in registry.all() if "既定の置き場の見本" in d.kind]
        finally:
            for definition in registry.all():
                if "既定の置き場の見本" in definition.kind:
                    registry.unregister(definition.kind)
            monkeypatch.setattr(catalog_module, "_catalog", None)

    def test_the_report_shot_leaves_no_sample_behind(
        self, shots: ModuleType, qt_application: QApplication, tmp_path: Path
    ) -> None:
        """見本のスクリプトの定義を登録簿に残さない

        残すと、同じモジュールで後に走る試験が走る順番しだいで見本を見てしまう
        （``forget_scripts`` が片付けるのはモジュールの終わり）
        """
        del qt_application
        context = shots.Context(
            media=None, script_root=tmp_path / "scripts", template_root=tmp_path / "t"
        )
        kinds = {definition.kind for definition in registry.all()}
        with _restored_scripts():
            shots.compatibility_dialog(context).close()
        assert {definition.kind for definition in registry.all()} == kinds


@contextmanager
def _restored_scripts() -> Iterator[None]:
    """写真の道具が差し替える物を、試験の前の形へ戻す

    道具はアプリ全体の一覧を見本の物と差し替え、見本のスクリプトを登録簿に足す
    一覧を戻すだけでは足した定義が残るので、増えた種類も外す
    モジュールの終わりにまとめて外す ``forget_scripts`` を待つと、同じモジュールの
    後の試験が見本を見る
    """
    # 基準を取る前に一覧を空の物にしておく 一覧がまだ無いまま ``script_catalog()`` を
    # 呼ぶと既定の探索先を走査して定義を登録し、それが基準に入って片付けから漏れる
    # 前の一覧は ``_catalog`` から直に取る 関数で取ると、同じ走査が起きる
    previous = catalog_module._catalog
    set_script_catalog(ScriptCatalog(roots=()))
    kinds = {definition.kind for definition in registry.all()}
    try:
        yield
    finally:
        for definition in registry.all():
            if definition.kind not in kinds:
                registry.unregister(definition.kind)
        # 前の一覧の定義は基準を取る前から登録されているので、差し戻すだけでよい
        catalog_module._catalog = previous


@pytest.mark.usefixtures("gpu")
class TestTakingTheEditorShot:
    def test_the_editor_shot_shows_the_preview(
        self, shots: ModuleType, qt_application: QApplication, tmp_path: Path
    ) -> None:
        """撮れた絵の**プレビューの場所**が真っ黒でないこと

        ``QWidget.grab()`` は環境によって GL の中身を拾わない 拾えていないと、
        気付かないまま真っ黒なプレビューの写真が README に載る
        窓全体の明るさで見ると、メニューやタイムラインが明るいだけで通ってしまう
        ので、プレビューの占める範囲だけを切って見る
        """
        del qt_application
        if not libx264_available():
            pytest.skip("libx264 の入った ffmpeg が無いので見本の素材を作れない")

        context = shots.Context(
            media=shots.make_sample_media(tmp_path / "media"),
            script_root=tmp_path / "scripts",
            template_root=tmp_path / "templates",
        )
        with shots.editor(shots.sample_project()) as window:
            shots.build_sample_timeline(window, context)
            image = shots.take_editor_shot(window)
            rect = shots.preview_rect(window)
            # 窓は部品の最小の幅より狭くできないので、頼んだ幅より広がることがある
            # 撮った絵は窓の論理的な大きさそのもの（画面の拡大率に左右されない）
            size = (window.width(), window.height())

        assert (image.width(), image.height()) == size
        assert size[1] == shots.WINDOW_SIZE[1]
        assert rect is not None
        assert shots.brightest(image, rect) > shots.BLACK_LEVEL


class TestAssistantShot:
    def test_the_sample_conversation_is_never_sent(
        self, shots: ModuleType, qt_application: QApplication
    ) -> None:
        """AI の写真の会話は見せるだけで、Claude には送らない

        送ると、撮るたびに文面が変わり、課金の要る呼び出しになる 会話は本物と同じ
        描き方で並び、会話の相手（セッション）は作られないままであること
        """
        del qt_application
        from sashimono.ui.chat import ChatPanel
        from sashimono.ui.main_window import MainWindow

        window = MainWindow(shots.sample_project(), confirm_unsaved=False)
        try:
            panel = window.findChild(ChatPanel)
            assert panel is not None
            shots._fake_conversation(panel)
            shown = panel._view.toPlainText()
            assert panel._session is None
            for _, text in shots.SAMPLE_CHAT:
                assert text.replace("**", "") in shown
        finally:
            window.close()
