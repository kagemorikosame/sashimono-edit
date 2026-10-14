"""README の写真を撮る道具（tools/shots.py）

写真は手で撮ると、撮った人の画面配置や本人のファイル名が写り、画面が変わった
ときに同じ絵を作り直せない この道具はそこを引き受けているので、壊れると
README の写真が古いまま残る

GPU が要る検査は ``gpu`` フィクスチャを付けてある GPU の無い所（CI）では飛ぶ
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PySide6.QtCore import QEvent, QRect
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QWidget,
)

from sashimono.asr import environment
from sashimono.asr.service import TranscriptionService
from sashimono.asr.whisper import FasterWhisperBackend
from sashimono.compat.aviutl import catalog as catalog_module
from sashimono.compat.aviutl.catalog import ScriptCatalog, script_catalog, set_script_catalog
from sashimono.compat.catalog import TemplateCatalog
from sashimono.core import userdirs
from sashimono.core.io import Preset
from sashimono.core.io.library import PRESET
from sashimono.core.model import Clip, MediaItem
from sashimono.effects.definition import registry
from sashimono.ui.library_thumbnails import THUMBNAILS_SIMPLE, LookThumbnails
from sashimono.ui.library_view import LibraryOptions
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


@pytest.fixture(autouse=True)
def no_modal_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """撮る道具が窓（確かめの小窓・ファイルの窓・exec で開く窓）を開いたら、その場で落とす

    撮影は人の居ない所で走る 窓を開くと誰も閉じずに止まり、CI は時間切れまで待つ
    """

    def refuse(*args: object, **_kwargs: object) -> object:
        pytest.fail(f"撮る道具が窓を開いた: {args}")

    for name in ("information", "warning", "question", "critical"):
        monkeypatch.setattr(QMessageBox, name, refuse)
    monkeypatch.setattr(QMessageBox, "exec", refuse)
    monkeypatch.setattr(QDialog, "exec", refuse)
    monkeypatch.setattr(QMenu, "exec", refuse)
    for name in ("getText", "getItem", "getInt", "getDouble"):
        monkeypatch.setattr(QInputDialog, name, refuse)
    for name in ("getExistingDirectory", "getOpenFileName", "getOpenFileNames", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, name, refuse)


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


class TestShelfThumbnailsInShots:
    """棚の写真は、一覧の見本が描き終わってから撮る

    見本は別のスレッドで後から埋まる 待たずに撮ると、地だけの小さな見本が写る
    """

    def test_it_waits_for_the_pictures_not_for_time(
        self, shots: ModuleType, qt_application: QApplication, tmp_path: Path
    ) -> None:
        del qt_application
        from sashimono.ui.library_thumbnails import THUMBNAILS_SIMPLE, LookThumbnails
        from sashimono.ui.library_view import LibraryOptions
        from sashimono.ui.shelf_looks import shelf_look
        from sashimono.ui.template_dialog import TemplateDialog

        for number in range(3):
            (tmp_path / f"見本{number}.object").write_text(SAMPLE_ALIAS, encoding="utf-8")
        thumbnails = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_SIMPLE)
        dialog = TemplateDialog(
            TemplateCatalog(),
            roots=(tmp_path,),
            thumbnails=thumbnails,
            options=LibraryOptions(thumbnails=THUMBNAILS_SIMPLE),
        )
        try:
            dialog.show()
            shots.wait_for_thumbnails(dialog)
            assert dialog.thumbnails_settled()
            for node in dialog._leaves():
                found = thumbnails.cached(shelf_look(node.data(0, 256)))
                assert found is not None and found.image is not None
        finally:
            dialog.close()
            thumbnails.release()


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
        """ホームの外の場所は、そのまま作業用のフォルダの置き場として使う

        壊れてホームの外まで断るようになると、``--work`` でホームの外の場所を渡しても
        撮影が始まらず、写真を 1 枚も撮り直せなくなる
        """
        outside = tmp_path / "外"
        outside.mkdir()
        assert shots.work_folder_base(outside, tmp_path / "ホーム") == outside.resolve()


class TestTranscribeShot:
    """字幕起こしの環境が入っている機械でも、「環境を導入」の写真が撮れること

    窓は導入済みならボタンを「環境を更新」にする 機械の状態のまま開くと、囲む
    「環境を導入」が見つからずに撮影が失敗する 起こしの機能を使っている開発者の
    機械ほど撮れなくなる
    """

    @pytest.fixture
    def installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """この機械には字幕起こしの環境が入っている、ということにする"""
        from sashimono.runtime import PackageStatus, PackStatus
        from sashimono.ui.subtitle import transcribe_dialog

        pack = environment.ASR_PACK
        status = PackStatus(
            pack=pack,
            packages=tuple(PackageStatus(name, "99.0") for name in pack.required),
            extras=tuple(PackageStatus(name, "99.0") for name in pack.extra),
        )
        monkeypatch.setattr(transcribe_dialog, "runtime_status", lambda: status)

    @pytest.mark.usefixtures("installed")
    def test_the_machine_state_says_update(
        self, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        # 前提の確かめ 状態を渡さなければ機械の状態（導入済み）で開く
        del qt_application
        from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog

        dialog = TranscribeDialog(video_media, TranscriptionService(FasterWhisperBackend()))
        try:
            assert dialog._install_button.text() == "環境を更新"
        finally:
            dialog.deleteLater()

    @pytest.mark.usefixtures("installed")
    def test_the_shot_opens_the_first_time_state(
        self, shots: ModuleType, qt_application: QApplication, video_media: MediaItem
    ) -> None:
        del qt_application
        from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog

        dialog = TranscribeDialog(
            video_media,
            TranscriptionService(FasterWhisperBackend()),
            status=shots.asr_not_installed,
        )
        shots.hide_from_screen(dialog)
        dialog.show()
        QApplication.processEvents()
        try:
            assert dialog._install_button.text() == "環境を導入"
            # 撮影が囲むボタンを探す所を、そのまま通る
            assert shots._buttons(dialog, ("環境を導入",))
            assert not dialog._run_button.isEnabled()
        finally:
            dialog.close()
            dialog.deleteLater()


class TestDefaultWorkFolder:
    """``--work`` を渡さないときの作業用のフォルダの置き場

    候補はドライブの根、次にリポジトリの中の ``.work/shots`` どちらもホームの下なら
    使わず、書けるかは作って消してみて決める
    """

    def test_the_drive_root_comes_first(
        self, shots: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 根はどのフォルダの名前も含まないので、どちらにも書けるなら根を使う
        # 壊れて順番が入れ替わると、書ける根があってもリポジトリの中を使い、作業用の
        # フォルダの場所にリポジトリのフォルダの名前が入る
        asked: list[Path] = []

        def writable(folder: Path) -> bool:
            asked.append(folder)
            return True

        monkeypatch.setattr(shots, "_writable", writable)
        chosen = shots.work_folder_base(None, tmp_path / "ホーム")
        assert chosen == Path(shots.ROOT.anchor).resolve()
        # 根で決まったら、リポジトリの中に .work/shots を作りに行かない
        assert asked == [chosen]

    def test_an_unwritable_root_falls_back_into_the_repository(
        self, shots: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、システムのドライブの根に書けない標準の利用者は、リポジトリの中へ
        # 落ちられず、引数なしの撮影が権限の誤りで止まる
        root = Path(shots.ROOT.anchor).resolve()
        monkeypatch.setattr(shots, "_writable", lambda folder: folder != root)
        chosen = shots.work_folder_base(None, tmp_path / "ホーム")
        assert chosen == (shots.ROOT / ".work" / "shots").resolve()

    def test_a_repository_in_the_home_is_not_used(
        self, shots: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、リポジトリがホームの下にあるとき、書けるからと .work/shots を使い、
        # ユーザー名を含む場所に作業用のフォルダを作る 使える所が無ければ --work を求めて止める
        root = Path(shots.ROOT.anchor).resolve()
        monkeypatch.setattr(shots, "_writable", lambda folder: folder != root)
        with pytest.raises(shots.ShotError, match="--work"):
            shots.work_folder_base(None, shots.ROOT.resolve())

    def test_nothing_writable_stops_with_the_way_out(
        self, shots: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、どちらにも書けないときに書けない場所を返し、撮り始めてから権限の
        # 誤りで落ちる 止めて、--work で渡せばよいと告げる
        monkeypatch.setattr(shots, "_writable", lambda folder: False)
        with pytest.raises(shots.ShotError, match="--work"):
            shots.work_folder_base(None, tmp_path / "ホーム")

    def test_writable_is_judged_by_making_a_folder(self, shots: ModuleType, tmp_path: Path) -> None:
        """書けるかの確かめが、実際に作れるかで決まり、確かめた跡を残さないこと

        ほかの試験は確かめを差し替えるので、本物の確かめはここで見る 壊れて常に
        書けると答えると、根に書けない利用者でも根が選ばれて撮影が止まる 跡を残すと、
        撮るたびにドライブの根に空のフォルダが溜まる
        """
        folder = tmp_path / "書ける"
        assert shots._writable(folder)
        assert list(folder.iterdir()) == []
        # フォルダの代わりにファイルがある所には作れない
        blocked = tmp_path / "ファイル"
        blocked.write_text("", encoding="utf-8")
        assert not shots._writable(blocked)


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

    def test_the_preferences_shot_hides_the_work_folder(
        self,
        shots: ModuleType,
        qt_application: QApplication,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """設定の窓の置き場の欄に、作業用のフォルダではなく ``%LOCALAPPDATA%`` の形を出す

        出したままだと、撮るたびに名前の変わる ``J:\\sashimono-shots-…`` が README の
        写真に写り、読む人の機械では出ない場所に見える（#313 の PR で写った）
        """
        del qt_application
        work = tmp_path / "sashimono-shots-撮るたびに変わる"
        monkeypatch.setenv("LOCALAPPDATA", str(work / "local"))
        monkeypatch.setenv("APPDATA", str(work / "roaming"))
        dialog = shots._full_preferences()
        try:
            shown = [
                text
                for box in dialog.findChildren(QLineEdit)
                for text in (box.text(), box.placeholderText())
            ]
        finally:
            dialog.close()
        assert not [text for text in shown if work.name in text]
        # 置き換えただけで既定の表示が消えると、置き場の欄が空に見えて何が既定か分からない
        assert any(text.startswith("既定（%LOCALAPPDATA%") for text in shown)


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


class TestWikiVersion:
    """手順書の頭に書いた「この版で撮りました」を、撮り直したときに今の版へ書き換える"""

    def test_the_version_line_follows_the_current_version(
        self, shots: ModuleType, tmp_path: Path
    ) -> None:
        # 手で書き換えると、版を上げて撮り直したのに古い版のまま残る
        page = tmp_path / "手順書-最初の1本.md"
        page.write_bytes(
            "# 最初の 1 本\r\n\r\n> この手順書の画面と文言は **Sashimono Edit 0.0.1** で撮りました"
            " 版が上がって画面が違うときは、その版で変わった所です\r\n本文\r\n".encode()
        )
        other = tmp_path / "Home.md"
        other.write_bytes(b"# Home\r\n")

        changed = shots.stamp_wiki_version(tmp_path, "9.8.7")

        assert changed == [page]
        text = page.read_bytes().decode("utf-8")
        assert "**Sashimono Edit 9.8.7** で撮りました" in text
        assert "0.0.1" not in text
        # 改行の形は変えない 変えると頁全体が差分になる
        assert text.count("\r\n") == 4
        assert other.read_bytes() == b"# Home\r\n"

    def test_the_default_is_the_running_version(self, shots: ModuleType) -> None:
        from sashimono import __version__

        assert shots.stamp_wiki_version.__defaults__ == (__version__,)


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
            # 撮った絵は窓の論理的な大きさそのもの（画面の拡大率に左右されない）
            # 前は部品の最小の幅の和（1367）が頼んだ幅を超え、窓が勝手に広がっていた
            # README の写真の幅が画面の作りで変わらないよう、頼んだ大きさのままを確かめる
            size = (window.width(), window.height())

        assert (image.width(), image.height()) == size
        assert size == shots.WINDOW_SIZE
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

    def test_the_sample_question_shows_every_choice_without_a_session(
        self, shots: ModuleType, qt_application: QApplication
    ) -> None:
        """AI の質問の写真（#317）は、見本の質問を本物と同じ箱に出し、Claude には送らない

        壊れると、Wiki の質問の写真に選択肢の欠けた箱が写るか、撮るたびに課金の要る
        呼び出しが走る 見本の質問が道具の決まり（選択肢の数や名前の長さ）から外れると、
        撮影がこの写真だけ失敗する
        """
        del qt_application
        from sashimono.ui.chat import ChatPanel
        from sashimono.ui.main_window import MainWindow

        window = MainWindow(shots.sample_project(), confirm_unsaved=False)
        try:
            panel = window.findChild(ChatPanel)
            assert panel is not None
            shots.show_sample_question(panel)
            assert panel._session is None
            assert not panel._question_box.isHidden()
            buttons = panel._question_choices.findChildren(QPushButton)
            choices = shots.SAMPLE_QUESTION["choices"]
            assert isinstance(choices, list)
            assert [button.text() for button in buttons] == [c["label"] for c in choices]
            # 何を頼んで聞き返されたのかが、会話の欄に残っている
            shown = panel._view.toPlainText()
            for _, text in shots.SAMPLE_QUESTION_CHAT:
                assert text in shown
        finally:
            window.close()


class TestLibraryShots:
    """プリセットとエイリアスの一覧の窓と保存の窓の写真（#317）

    見本の絵は別のスレッドで後から埋まる ここでは GPU の要らない描き方（文字と図形だけ）の
    係を渡して、撮る道具の待ち方と組み立てを確かめる
    """

    @pytest.fixture
    def thumbnails(self, tmp_path: Path) -> Iterator[LookThumbnails]:
        created = LookThumbnails(tmp_path / "cache", mode=THUMBNAILS_SIMPLE)
        yield created
        created.release()

    def test_the_sample_library_holds_only_the_samples(
        self, shots: ModuleType, tmp_path: Path
    ) -> None:
        # 壊れると、前に撮った写真で書いた物が次の一覧に混ざり、撮る順番で一覧の絵が変わる
        root = tmp_path / "置き場"
        shots.sample_library(root)
        stale = shots.sample_library(root)
        stale.presets.save(Preset.capture("前の撮影の残り", _text_clip()))
        library = shots.sample_library(root)
        found = {(entry.name, entry.category) for entry in library.entries(PRESET)}
        assert found == {(name, category) for name, category, *_ in shots.SAMPLE_PRESETS}

    def test_the_manage_window_waits_for_every_picture(
        self,
        shots: ModuleType,
        qt_application: QApplication,
        tmp_path: Path,
        thumbnails: LookThumbnails,
    ) -> None:
        # 壊れて待たずに撮ると、遅い機械では地だけの見本が並ぶ写真になる
        del qt_application
        library = shots.sample_library(tmp_path / "置き場")
        dialog = shots.library_dialog(library, thumbnails, _simple_options())
        try:
            shots.hide_from_screen(dialog)
            dialog.show()
            shots.wait_for_library_pictures(dialog)
            for entry in library.entries(PRESET):
                assert thumbnails.cached(entry.item) is not None, entry.name
            # 1 つ選んだ所を見せる 選んでいないと、整理のボタンが全部灰色で写る
            assert [item.text() for item in dialog.findChild(QListWidget).selectedItems()]
        finally:
            dialog.reject()

    def test_the_save_window_shows_the_overwrite(
        self,
        shots: ModuleType,
        qt_application: QApplication,
        tmp_path: Path,
        thumbnails: LookThumbnails,
    ) -> None:
        # 壊れると、Wiki の保存の窓の写真に上書きの印も、前といまの見本も写らない
        del qt_application
        library = shots.sample_library(tmp_path / "置き場")
        dialog = shots.save_dialog(library, thumbnails, _simple_options())
        try:
            shots.hide_from_screen(dialog)
            dialog.show()
            shots.wait_for_library_pictures(dialog)
            overwrite = next(
                box for box in dialog.findChildren(QCheckBox) if box.text().startswith("同じ名前")
            )
            assert overwrite.isChecked()
            assert dialog.can_save()
            assert "もうあります" in dialog.warning()
            assert dialog.choice().name == shots.SAMPLE_OVERWRITE
            assert not dialog._before.pixmap().isNull()
            assert not dialog._after.pixmap().isNull()
            # 前といまが同じ絵だと、上書きで何が変わるのかが写らない
            assert dialog._before.pixmap().toImage() != dialog._after.pixmap().toImage()
        finally:
            dialog.reject()

    @pytest.mark.parametrize("which", ["library_dialog", "save_dialog"])
    def test_the_window_is_the_same_every_time(
        self,
        shots: ModuleType,
        qt_application: QApplication,
        tmp_path: Path,
        which: str,
    ) -> None:
        # 壊れると、撮り直すたびに写真が変わり、画面が変わっていないのに Wiki の差分が出る
        # 覚えた絵を使い回して同じになるのを避け、2 回とも別の置き場と別の係で描く
        del qt_application
        pictures = []
        for attempt in ("1", "2"):
            thumbnails = LookThumbnails(tmp_path / f"cache{attempt}", mode=THUMBNAILS_SIMPLE)
            try:
                library = shots.sample_library(tmp_path / f"置き場{attempt}")
                dialog = getattr(shots, which)(library, thumbnails, _simple_options())
                pictures.append(shots._grab_library(dialog))
            finally:
                thumbnails.release()
        assert pictures[0] == pictures[1]

    @pytest.mark.parametrize("which", ["library_dialog", "save_dialog"])
    def test_no_folder_is_shown(
        self,
        shots: ModuleType,
        qt_application: QApplication,
        tmp_path: Path,
        thumbnails: LookThumbnails,
        which: str,
    ) -> None:
        # 壊れると、撮るたびに名前の変わる作業用のフォルダ（J:\sashimono-shots-…）が Wiki の
        # 写真に写る 一覧の補足（ツールチップ）は写らないので見ない
        del qt_application
        root = tmp_path / "sashimono-shots-撮るたびに変わる"
        library = shots.sample_library(root)
        dialog = getattr(shots, which)(library, thumbnails, _simple_options())
        try:
            shown = [w.text() for w in dialog.findChildren(QLabel)]
            shown += [w.text() for w in dialog.findChildren(QAbstractButton)]
            shown += [w.text() for w in dialog.findChildren(QLineEdit)]
            shown += [w.placeholderText() for w in dialog.findChildren(QLineEdit)]
            for listing in dialog.findChildren(QListWidget):
                shown += [listing.item(row).text() for row in range(listing.count())]
        finally:
            dialog.reject()
        assert not [text for text in shown if root.name in text or str(tmp_path) in text]


@pytest.mark.usefixtures("gpu")
class TestNewEditorShots:
    """設定パネルの縁取りの層と組み方のボタンの写真（#317）"""

    def test_the_shots_are_the_same_every_time(
        self, shots: ModuleType, qt_application: QApplication, tmp_path: Path
    ) -> None:
        # 壊れると、撮り直すたびに写真が変わり、画面が変わっていないのに Wiki の差分が出る
        del qt_application
        context = _template_context(shots, tmp_path)
        for take in (shots.wiki_edit_strokes, shots.wiki_edit_layout):
            assert take(context) == take(context), take.__name__

    def test_a_button_out_of_view_is_not_marked(
        self, shots: ModuleType, qt_application: QApplication
    ) -> None:
        # 壊れると、設定パネルの巻物の外にあるボタンを囲む枠が、タイムラインの上など何も無い所に
        # 描かれ、手順書が何も無い所を指す
        del qt_application
        with shots.editor(shots.sample_project()) as window:
            text = shots.TEXT.create(text="縁", size=96).with_strokes(
                shots.sample_strokes((6.0, (0.0, 0.0, 0.0, 1.0)), (12.0, (1.0, 1.0, 1.0, 1.0)))
            )
            window.apply_commands(shots._place_text(window, text, at_frame=0, duration=30), "置く")
            window.select_clip(shots.top_text_clip(window))
            shots.settle(window, rounds=4)
            button = shots._named_button(window, "add_stroke")
            with pytest.raises(shots.ShotError, match="見えている所"):
                shots.inspector_rect(window, button)
            shots._scroll_inspector_to(window, button)
            assert shots.inspector_rect(window, button).height() == button.height()

    def test_a_stretched_window_gets_the_default_layout_back(
        self, shots: ModuleType, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """出したときに縦へ伸びた窓（#316）を、頼んだ大きさと既定の画面配置へ戻す

        大きさだけを戻すと、伸びた分がタイムラインから削られ、タイムラインが 1 段しか写らない
        （0.2.2 の Wiki の写真で、レイヤー 2 の音声が写らなかった）
        """
        from sashimono.ai import AI_PACK
        from sashimono.runtime import PackageStatus, PackStatus
        from sashimono.ui.chat import panel as panel_module
        from sashimono.ui.setup import SetupSection
        from sashimono.ui.theme import style_sheet

        # 窓が伸びる機械の形にする AI の部品は入っていてログインがまだ（#316 の PR の試験と同じ）
        # 本人の機械の状態に任せると、手元では伸びずに通ってしまう 本番のスタイルシートも当てる
        # （ボタンの余白で高さが変わる）
        status = PackStatus(pack=AI_PACK, packages=(PackageStatus("claude-agent-sdk", "0.2.158"),))
        monkeypatch.setattr(SetupSection, "status", property(lambda _self: status))
        monkeypatch.setattr(panel_module, "credentials_found", lambda: False)
        sheet = qt_application.styleSheet()
        qt_application.setStyleSheet(style_sheet())
        try:
            with shots.editor(shots.sample_project()) as window:
                timeline = window.findChild(QDockWidget, "timeline")
                assert timeline is not None
                shown = timeline.height()
                size = (window.width(), window.height())
                # 頼んだ大きさの窓に既定の画面配置を当て直した高さと比べる
                window.reset_layout()
                shots.settle(window, rounds=4)
                expected = timeline.height()
            QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
            gc.collect()
        finally:
            qt_application.setStyleSheet(sheet)
        assert size == shots.WINDOW_SIZE
        assert shown == expected


def _text_clip() -> Clip:
    from sashimono.core.commands.fixed import with_fixed_items
    from sashimono.effects.sources import TEXT

    clip = Clip(timeline_start=0, duration=30, source=TEXT.create(text="残り"))
    return with_fixed_items(clip, picture=True)


def _simple_options() -> LibraryOptions:
    """GPU の要らない見本の描き方 試験の機械（CI）で同じ絵になる"""
    return LibraryOptions(thumbnails=THUMBNAILS_SIMPLE)


def _template_context(shots: ModuleType, tmp_path: Path) -> Any:
    root = shots.write_sample_templates(tmp_path / "見本のテンプレート")
    return shots.Context(media=None, script_root=tmp_path / "scripts", template_root=root)
