"""自動の退避と世代バックアップの設定（#271）

設定画面の欄と、選んだ値を確かめる決まりをここにまとめる 設定画面（``preferences_dialog``）
には多くの作業が項目を足すので、この塊を 1 か所に置いて、画面の側の差分を小さく保つ

決めたこと

- 置き場は退避（``recovery``）とバックアップ（``backups``）だけを動かす 窓ごとの錠（``open``）は
  既定の置き場のまま 錠は同じ機械のすべての窓が同じ所を見ないと、2 つの窓で同じプロジェクトを
  開いたことに気付けない
- 置き場を変えても、前の置き場の中身は移さない（古い物は前の置き場に残る） 移す途中で
  落ちたり書けなかったりすると、守るはずの物を失う
- 書けない所（読み取り専用・exe の隣・消えたドライブ）は選ばせない 起動したときや退避の途中で
  書けなくなったら、その起動は既定の置き場へ戻して知らせる（黙って退避を止めない）
- 同期フォルダ（OneDrive など）とネットワークの置き場は、選べるが確かめる 数十秒おきの退避が
  そのまま同期され、回線が切れると書けなくなる
- 世代数を減らすときは、次の上書き保存で消える数を見せて確かめる その場では消さない
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from sashimono.core.io import backups_over, default_state_root, folder_problem
from sashimono.runtime import app_dir
from sashimono.ui.workspace import (
    AUTOSAVE_SECONDS_RANGE,
    BACKUP_GENERATIONS_RANGE,
    Preferences,
)
from sashimono.update.portable import in_synced_folder

__all__ = [
    "BackupSection",
    "confirm_backup_changes",
    "folder_caution",
    "folder_refusal",
    "state_root_for",
]


def state_root_for(preferences: Preferences) -> Path:
    """設定で選んだ退避とバックアップの置き場 空なら既定"""
    if preferences.state_folder:
        return Path(preferences.state_folder)
    return default_state_root()


def folder_refusal(folder: Path, install: Path | None = None) -> str | None:
    """置き場に選ばせない理由 選べるなら ``None``

    ``install`` は配布版の exe の置き場（試験では差し替える） 省けば今の起動の置き場を見る
    """
    if not folder.is_absolute():
        return "フォルダは C:\\ のようにドライブから書いた場所で選んでください"
    install = install if install is not None else app_dir()
    if install is not None and _inside(folder, install):
        # exe の隣は zip を展開し直すと丸ごと入れ替わる 更新で消えない置き場の決まり
        # （docs/development.md「更新で消えない置き場」）で、本人の物は置かない
        return "Sashimono を置いたフォルダの中は、展開し直すと消えるので選べません"
    problem = folder_problem(folder)
    if problem is not None:
        return f"このフォルダには書き込めません\n{problem}"
    return None


def folder_caution(folder: Path) -> str | None:
    """選べるが確かめたい理由 気になる所が無ければ ``None``"""
    if str(folder).startswith(("\\\\", "//")):
        return (
            "ネットワークの場所です 繋がっていないときは退避とバックアップを既定の置き場へ"
            "書くので、置き場が 2 か所に分かれます"
        )
    if in_synced_folder(folder):
        return (
            "OneDrive などの同期フォルダの中です 保存していない変更を数十秒おきに書くので、"
            "そのたびに同期されて回線と同期先の容量を使います"
        )
    return None


def _inside(folder: Path, parent: Path) -> bool:
    try:
        return folder.resolve().is_relative_to(parent.resolve())
    except (OSError, ValueError):
        return False


class StateFolderField(QWidget):
    """置き場の欄 空は既定 選ぶボタンと既定へ戻すボタンを添える"""

    def __init__(self, folder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.line = QLineEdit(folder, self)
        # 空を「既定」と見せる 既定の場所を文字で入れると、ユーザー名の違う機械へ
        # 設定を写したときに、その場所を指したまま書けなくなる
        self.line.setPlaceholderText(f"既定（{default_state_root()}）")
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
        """選んだ置き場 既定と同じ場所を選んだときも空にする（既定に付いていくため）"""
        text = self.line.text().strip()
        if text and _same(Path(text), default_state_root()):
            return ""
        return text

    def _choose(self) -> None:
        start = self.line.text() or str(default_state_root())
        chosen = QFileDialog.getExistingDirectory(self, "退避とバックアップの置き場", start)
        if chosen:
            self.line.setText(str(Path(chosen)))


def _same(first: Path, second: Path) -> bool:
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return first == second


class BackupSection:
    """設定画面の退避とバックアップの欄"""

    def __init__(self, preferences: Preferences, parent: QWidget) -> None:
        self.before = preferences
        self.autosave = QCheckBox("保存していない変更を自動で退避する", parent)
        self.autosave.setChecked(preferences.autosave)
        self.autosave.setToolTip(
            "一定の間隔で、保存していない変更を退避の置き場へ書く 落ちたあとの次の起動で、"
            "復元するかを尋ねる 切ると、落ちたときに最後に保存した所までしか戻らない"
        )
        self.autosave_seconds = QSpinBox(parent)
        self.autosave_seconds.setRange(*AUTOSAVE_SECONDS_RANGE)
        self.autosave_seconds.setSuffix(" 秒")
        self.autosave_seconds.setValue(preferences.autosave_seconds)
        self.autosave_seconds.setToolTip(
            "落ちたときに失うのは最大でこの長さの作業 短くするほど書き込みが増える"
            " 変わっていなければ書かない OK を押すとその場で効く"
        )
        self.autosave.toggled.connect(self.autosave_seconds.setEnabled)
        self.autosave_seconds.setEnabled(preferences.autosave)

        self.backup = QCheckBox("上書き保存の前の中身をバックアップに残す", parent)
        self.backup.setChecked(preferences.backup)
        self.backup.setToolTip(
            "上書き保存で消える前の中身を、プロジェクトごとに残す 〔ファイル〕→"
            "〔バックアップのフォルダを開く〕から開ける 切っても今あるバックアップは消さない"
        )
        self.backup_generations = QSpinBox(parent)
        self.backup_generations.setRange(*BACKUP_GENERATIONS_RANGE)
        self.backup_generations.setSuffix(" 世代")
        self.backup_generations.setValue(preferences.backup_generations)
        self.backup_generations.setToolTip(
            "1 つのプロジェクトについて残す数 超えたら古い物から消す 減らしたときは、"
            "次にそのプロジェクトを上書き保存したときに消える（OK を押す前に数を見せて確かめる）"
        )
        self.backup.toggled.connect(self.backup_generations.setEnabled)
        self.backup_generations.setEnabled(preferences.backup)

        self.state_folder = StateFolderField(preferences.state_folder, parent)
        self.state_folder.setToolTip(
            "退避とバックアップを書くフォルダ 空は既定の場所 変えても前の置き場の中身は移さない"
            " 書けないときは、その起動の間は既定の場所へ書いて知らせる"
            " 落ちた作業は、選んだ場所と既定の場所の両方から探す"
        )

    def add_rows(self, form: QFormLayout) -> None:
        form.addRow(self.autosave)
        form.addRow("退避の間隔", self.autosave_seconds)
        form.addRow(self.backup)
        form.addRow("残すバックアップの数", self.backup_generations)
        form.addRow("退避とバックアップの置き場", self.state_folder)


def confirm_backup_changes(parent: QWidget | None, before: Preferences, after: Preferences) -> bool:
    """設定画面の OK で、退避とバックアップの変更を確かめる 進めてよければ真

    書けない置き場はここで止める OK の後で書けないと分かると、本人は設定が効いたと思ったまま
    退避が既定の側へ戻り続ける
    """
    if after.state_folder and after.state_folder != before.state_folder:
        folder = Path(after.state_folder)
        refusal = folder_refusal(folder)
        if refusal is not None:
            QMessageBox.warning(parent, "この置き場は使えません", f"{folder}\n\n{refusal}")
            return False
        caution = folder_caution(folder)
        if caution is not None:
            answer = QMessageBox.question(
                parent,
                "退避とバックアップの置き場",
                f"{folder}\n\n{caution}\n\nこの置き場を使いますか",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
    if after.backup and after.backup_generations < before.backup_generations:
        doomed = backups_over(after.backup_generations, state_root_for(after))
        if doomed > 0:
            answer = QMessageBox.question(
                parent,
                "バックアップを減らす",
                f"残す数を {after.backup_generations} 世代にすると、今あるバックアップのうち "
                f"{doomed} 本が、それぞれのプロジェクトを次に上書き保存したときに消えます"
                "（古い物から 消したバックアップは戻せません）\n\n減らしますか",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
    return True
