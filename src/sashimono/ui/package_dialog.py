"""持ち出し用パッケージ（#423）の窓と知らせ

書き出しの窓は、何を入れるか（使う範囲だけ・素材の全体）と、付け足しの情報（撮った場所
など）を残すかを選ぶ 既定は知らない人が困らない側（使う範囲だけ・取り除く）
選んだ物は覚えて、次に開いた窓の初めの値にする

編集用の大きな形式を符号にし直す選択（#446）だけは覚えない 画質を元のまま持ち出す約束
（#423）から外れる選択なので、前に 1 度選んだだけで次からも黙って画質が変わると困る
作品にその形式の素材があるときだけ押せるようにし、何が変わるかを窓に書く
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sashimono.compat.aviutl.catalog import script_catalog
from sashimono.core.userdirs import state_root
from sashimono.engine.package.archive import (
    PACKAGE_SUFFIX,
    RIGHTS_NOTE,
    OpenedPackage,
    PackOptions,
    PackResult,
)
from sashimono.engine.package.plan import REENCODE_NOTE
from sashimono.engine.package.remux import Reencoded
from sashimono.ui.workspace import PACKAGE_MARGINS, Preferences

__all__ = [
    "PackageExportDialog",
    "PackageFolderField",
    "missing_fonts",
    "missing_scripts",
    "opened_summary",
    "package_root",
    "packed_summary",
]


def package_root(preferences: Preferences) -> Path:
    """パッケージを展開するフォルダ 設定が空なら既定（本人の置き場の下）"""
    if preferences.package_folder:
        return Path(preferences.package_folder)
    return state_root() / "packages"


class PackageFolderField(QWidget):
    """展開するフォルダの欄 空は既定 選ぶボタンと既定へ戻すボタンを添える"""

    def __init__(self, folder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.line = QLineEdit(folder, self)
        # 空を「既定」と見せる 既定の場所を文字で入れると、ユーザー名の違う機械へ設定を
        # 写したときに、その場所を指したまま書けなくなる
        self.line.setPlaceholderText(f"既定（{state_root() / 'packages'}）")
        self.line.setReadOnly(True)
        choose = QPushButton("選ぶ…", self)
        choose.clicked.connect(self._choose)
        reset = QPushButton("既定に戻す", self)
        reset.clicked.connect(self.line.clear)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line, 1)
        layout.addWidget(choose)
        layout.addWidget(reset)

    def folder(self) -> str:
        """選んだ場所 空は既定の場所"""
        return self.line.text().strip()

    def _choose(self) -> None:
        """場所を選ぶ窓を出す"""
        start = self.line.text() or str(state_root())
        chosen = QFileDialog.getExistingDirectory(self, "持ち出し用パッケージを開く場所", start)
        if chosen:
            self.line.setText(str(Path(chosen)))


class PackageExportDialog(QDialog):
    """持ち出し用に書き出す窓 OK で書き出し先と選び方が決まる（書くのは呼ぶ側）"""

    def __init__(
        self,
        preferences: Preferences,
        suggested: Path,
        parent: QWidget | None = None,
        *,
        editing: tuple[str, ...] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("持ち出し用に書き出す")
        self._path = QLineEdit(str(suggested), self)
        choose = QPushButton("選ぶ…", self)
        choose.clicked.connect(self._choose)
        place = QHBoxLayout()
        place.addWidget(self._path, 1)
        place.addWidget(choose)

        self._used = QRadioButton(
            "使っている範囲だけ（小さい 範囲の外へはトリムを伸ばせない）", self
        )
        self._whole = QRadioButton("素材の全体（大きい 範囲の外へもトリムを伸ばせる）", self)
        (self._whole if preferences.package_whole else self._used).setChecked(True)
        self._margin = QSpinBox(self)
        self._margin.setRange(*PACKAGE_MARGINS)
        self._margin.setSuffix(" 秒")
        self._margin.setValue(preferences.package_margin_seconds)
        self._margin.setToolTip(
            "使っている範囲の前後に足す長さ 動画はキーフレームでしか切れないので、"
            "実際はその手前のキーフレームから入る"
        )
        self._used.toggled.connect(self._margin.setEnabled)
        self._margin.setEnabled(self._used.isChecked())
        self._keep = QCheckBox("撮った場所・機器の名前などの付け足しの情報を残す", self)
        self._keep.setChecked(preferences.package_keep_metadata)
        self._keep.setToolTip(
            "切っていると取り除く（人へ渡すときに場所が分からないように） "
            "回転と色の情報は切っていても残る 画質と音質はどちらでも元のまま"
        )

        self._reencode = QCheckBox(
            "編集用の大きな形式（ProRes・DNxHR・無圧縮など）を H.264 にし直して小さくする"
            "（画質がわずかに変わる）",
            self,
        )
        # 覚えない（モジュールの説明） いつも切った状態から始める
        self._reencode.setChecked(False)
        self._reencode.setEnabled(bool(editing))
        self._reencode_note = QLabel(
            (
                "対象: "
                + "、".join(editing[:10])
                + (f" ほか {len(editing) - 10}" if len(editing) > 10 else "")
                if editing
                else "この作品には編集用の大きな形式の素材がありません"
            )
            + "\n"
            + REENCODE_NOTE,
            self,
        )
        self._reencode_note.setWordWrap(True)
        self._reencode_note.setEnabled(bool(editing))

        form = QFormLayout()
        form.addRow("書き出し先", place)
        form.addRow("入れる素材", self._used)
        form.addRow("", self._whole)
        form.addRow("前後の余裕", self._margin)
        form.addRow(self._keep)
        form.addRow(self._reencode)
        form.addRow(self._reencode_note)
        note = QLabel(
            "上の選択を切っていれば、画質と音質は元のまま（再エンコードしない） "
            "書体と AviUtl のスクリプトは入れない"
            "（配る許しが物ごとに違うため） 使っている物の一覧を書き出した後に知らせる\n"
            + RIGHTS_NOTE,
            self,
        )
        note.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def target(self) -> Path:
        """書き出し先 拡張子が .smex でなければ足す"""
        path = Path(self._path.text().strip())
        return (
            path
            if path.suffix.lower() == PACKAGE_SUFFIX
            else path.with_name(path.name + PACKAGE_SUFFIX)
        )

    def options(self) -> PackOptions:
        """窓で選んだ書き出し方"""
        return PackOptions(
            whole=self._whole.isChecked(),
            margin=Fraction(self._margin.value()),
            keep_metadata=self._keep.isChecked(),
            reencode=self._reencode.isChecked(),
        )

    def remembered(self, preferences: Preferences) -> Preferences:
        """選んだ物を覚えた設定"""
        from dataclasses import replace

        return replace(
            preferences,
            package_whole=self._whole.isChecked(),
            package_margin_seconds=self._margin.value(),
            package_keep_metadata=self._keep.isChecked(),
        )

    def _choose(self) -> None:
        """場所を選ぶ窓を出す"""
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "持ち出し用パッケージの書き出し先",
            self._path.text(),
            f"Sashimono Edit 持ち出し用パッケージ (*{PACKAGE_SUFFIX})",
        )
        if chosen:
            self._path.setText(str(Path(chosen)))

    def _accept(self) -> None:
        """書き出し先を確かめてから閉じる"""
        text = self._path.text().strip()
        if not text or not Path(text).is_absolute():
            QMessageBox.warning(self, "書き出せない", "書き出し先を選んでください")
            return
        if not self.target().parent.is_dir():
            QMessageBox.warning(
                self, "書き出せない", f"フォルダがありません: {self.target().parent}"
            )
            return
        self.accept()


def missing_fonts(fonts: tuple[str, ...]) -> list[str]:
    """この PC に入っていない書体 入っていなければ別の書体で描かれる"""
    installed = {family.casefold() for family in QFontDatabase.families()}
    return [font for font in fonts if font.casefold() not in installed]


def missing_scripts(scripts: tuple[str, ...]) -> list[str]:
    """この PC のスクリプトの置き場に無い AviUtl のスクリプト"""
    roots = script_catalog().roots
    return [place for place in scripts if not any((root / place).is_file() for root in roots)]


def _listed(title: str, names: list[str] | tuple[str, ...]) -> str:
    """名前の一覧を知らせの文の 1 段にする 空なら空"""
    if not names:
        return ""
    shown = "、".join(names[:20])
    more = f" ほか {len(names) - 20}" if len(names) > 20 else ""
    return f"\n{title}: {shown}{more}"


def packed_summary(result: PackResult) -> str:
    """書き出した後の知らせ 入れなかった物を並べる"""
    size = result.size / (1 << 20)
    text = f"書き出しました: {result.path}（{size:.1f} MB 素材 {result.media_count} 本）"
    text += _listed("入れていない書体（相手の PC に無ければ別の書体になる）", result.fonts)
    text += _listed("入れていない AviUtl のスクリプト", result.scripts)
    text += _listed("範囲だけを切り出せず、全体を入れた素材", result.whole_instead)
    text += _listed("付け足しの情報を取り除けずに入れた物", result.metadata_kept)
    text += _listed("見つからなかったファイル（設定が指していた物）", result.missing_files)
    text += _listed("使っていないので入れなかった素材（素材一覧には残る）", result.unused_media)
    text += _listed(
        "H.264 にし直して入れた素材（元との差 PSNR・大きさ）",
        [_reencoded_text(name, report) for name, report in result.reencoded],
    )
    text += _listed(
        "し直せず（差が大きい・小さくならない・形が合わない）元のまま入れた素材",
        result.not_reencoded,
    )
    return text


def _reencoded_text(name: str, report: Reencoded) -> str:
    """符号にし直した素材 1 つの知らせ"""
    shrink = f" {report.after / report.before:.0%} の大きさ" if report.before else ""
    difference = f"{report.psnr:.1f} dB" if math.isfinite(report.psnr) else "元と同じ絵"
    return f"{name}（{difference}{shrink}）"


def opened_summary(opened: OpenedPackage) -> str:
    """開いた後の知らせ この PC に足りない書体とスクリプトを並べる 足りなければ空"""
    text = _listed("この PC に無い書体（別の書体で描かれる）", missing_fonts(opened.fonts))
    text += _listed(
        "この PC に無い AviUtl のスクリプト（そのエフェクトは描かれない）",
        missing_scripts(opened.scripts),
    )
    return text.strip()
