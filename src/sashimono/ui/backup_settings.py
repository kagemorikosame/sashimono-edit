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
- 世代数を減らす・バックアップを入れ直す・置き場を変えるときは、当てる先で世代数を超えた
  控えを見せて確かめ、OK でその場で見せた物だけを消す 保存のときは 1 本作って 1 本消す
  入れ替えだけにして、それより多くは消さない（見せた数と消える数を一致させる）
- 残った退避を日数で片付けるのは、起動して復元を尋ねた後だけ 一度も勧めていない退避は消さない
- 容量の上限は、選んだ置き場と既定の置き場のそれぞれに掛ける 世代数の上限とは別に効き、
  先に当たった方で消える 開いている作業の今の退避・まだ勧めていない落ちた作業・
  各プロジェクトのいちばん新しいバックアップは、超えていても消さない 上限を入れる・下げるときは
  消える物を見せて確かめ、超えて消したときはステータスバーで何を消したかを知らせる
"""

from __future__ import annotations

from collections.abc import Iterable
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

from sashimono.core.io import (
    default_state_root,
    folder_problem,
    plan_prune,
    plan_trim,
    remembered_roots,
)
from sashimono.core.io.recovery import TrimItem
from sashimono.runtime import app_dir
from sashimono.ui.workspace import (
    AUTOSAVE_SECONDS_RANGE,
    BACKUP_GENERATIONS_RANGE,
    RECOVERY_KEEP_DAYS_RANGE,
    STATE_LIMIT_MB_RANGE,
    Preferences,
)
from sashimono.update.portable import in_synced_folder

__all__ = [
    "BackupSection",
    "confirm_backup_changes",
    "describe_trim",
    "folder_caution",
    "folder_refusal",
    "plan_prune_all",
    "plan_trim_all",
    "recovery_roots",
    "state_root_for",
    "state_roots",
]

#: 消える物の名前を並べる数 多すぎると確かめの窓が画面からはみ出す
_LISTED = 8


def state_root_for(preferences: Preferences) -> Path:
    """設定で選んだ退避とバックアップの置き場 空なら既定"""
    if preferences.state_folder:
        return Path(preferences.state_folder)
    return default_state_root()


def state_roots(preferences: Preferences) -> list[Path]:
    """片付けと復元で見る置き場 選んだ置き場と既定の置き場（同じなら 1 つ）

    書けずに既定へ戻した起動の退避と控えは既定の側にある 選んだ側だけ見ると、
    そちらは上限も日数も効かないまま溜まり続ける
    """
    chosen, default = state_root_for(preferences), default_state_root()
    return [chosen] if chosen == default else [chosen, default]


def recovery_roots(preferences: Preferences) -> list[Path]:
    """起動したときに落ちた作業を探す置き場 設定の置き場・既定の置き場・前に退避を書いた置き場

    前に書いた置き場も見るのは、置き場を変えた後に古い置き場へ書いた退避（新しい置き場へ
    書けずに戻した、別の窓が古い設定のまま動いていた など）を見失わないため
    """
    roots = state_roots(preferences)
    try:
        remembered = remembered_roots()
    except OSError:
        remembered = []
    keys = {str(root) for root in roots}
    return [*roots, *(root for root in remembered if str(root) not in keys)]


def describe_trim(items: list[TrimItem]) -> str:
    """片付ける物の一覧 名前ごとに数をまとめ、多ければ残りの数だけ書く"""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.label] = counts.get(item.label, 0) + 1
    lines = [f"{label}（{count} 件）" for label, count in list(counts.items())[:_LISTED]]
    rest = len(counts) - _LISTED
    if rest > 0:
        lines.append(f"ほか {rest} 種類")
    size = sum(item.size for item in items) / (1024 * 1024)
    return f"{len(items)} 件 {size:.1f}MB\n" + "\n".join(lines)


def plan_trim_all(preferences: Preferences, already: Iterable[Path] = ()) -> list[TrimItem]:
    """容量の上限で片付ける物（全部の置き場） 上限が無ければ空

    ``already`` は同じ確かめで先に消すと決めた物（:func:`plan_prune_all`）
    """
    if preferences.state_limit_mb <= 0:
        return []
    limit = preferences.state_limit_mb * 1024 * 1024
    gone = list(already)
    return [
        item for root in state_roots(preferences) for item in plan_trim(limit, root, already=gone)
    ]


def plan_prune_all(preferences: Preferences) -> list[TrimItem]:
    """世代数を超えた控え（全部の置き場） バックアップを切っていれば空"""
    if not preferences.backup:
        return []
    return [
        item
        for root in state_roots(preferences)
        for item in plan_prune(preferences.backup_generations, root)
    ]


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
            "1 つのプロジェクトについて残す数 上書き保存のたびに 1 本作って、超えた古い 1 本を消す"
            " 減らしたとき・置き場を変えたときに超えている分は、OK を押す前に消える物を見せて"
            "確かめ、その場で消す"
        )
        self.backup.toggled.connect(self.backup_generations.setEnabled)
        self.backup_generations.setEnabled(preferences.backup)

        self.state_folder = StateFolderField(preferences.state_folder, parent)
        self.state_folder.setToolTip(
            "退避とバックアップを書くフォルダ 空は既定の場所 変えても前の置き場の中身は移さない"
            " 書けないときは、その起動の間は既定の場所へ書いて知らせる"
            " 落ちた作業は、選んだ場所と既定の場所の両方から探す"
        )

        self.recovery_keep_days = QSpinBox(parent)
        self.recovery_keep_days.setRange(*RECOVERY_KEEP_DAYS_RANGE)
        self.recovery_keep_days.setSuffix(" 日")
        self.recovery_keep_days.setSpecialValueText("片付けない（既定）")
        self.recovery_keep_days.setValue(preferences.recovery_keep_days)
        self.recovery_keep_days.setToolTip(
            "起動したときに復元を尋ね、復元も破棄もしないまま残った落ちた作業を、退避してから"
            "この日数がたったら片付ける 片付けるのは起動して復元を尋ねた後だけで、まだ一度も"
            "尋ねていない物は古くても消さない 片付けた数はステータスバーで知らせる"
        )

        self.state_limit_mb = QSpinBox(parent)
        self.state_limit_mb.setRange(*STATE_LIMIT_MB_RANGE)
        self.state_limit_mb.setSingleStep(100)
        self.state_limit_mb.setSuffix(" MB")
        self.state_limit_mb.setSpecialValueText("上限なし（既定）")
        self.state_limit_mb.setValue(preferences.state_limit_mb)
        self.state_limit_mb.setToolTip(
            "退避とバックアップが置き場で使う大きさの上限 選んだ置き場と既定の置き場の"
            "それぞれに掛ける 超えたら古い物から片付け、何を消したかをステータスバーで知らせる"
            " 残すバックアップの数とは別に効き、先に当たった方で消える"
            " 開いている作業の退避・まだ復元を尋ねていない落ちた作業・各プロジェクトの"
            "いちばん新しいバックアップは、超えていても消さない"
            " 上限を入れる・下げるときは、消える物を見せて確かめる"
        )

    def add_rows(self, form: QFormLayout) -> None:
        form.addRow(self.autosave)
        form.addRow("退避の間隔", self.autosave_seconds)
        form.addRow(self.backup)
        form.addRow("残すバックアップの数", self.backup_generations)
        form.addRow("退避とバックアップの置き場", self.state_folder)
        form.addRow("残った退避を片付ける", self.recovery_keep_days)
        form.addRow("置き場の容量の上限", self.state_limit_mb)


def confirm_backup_changes(
    parent: QWidget | None,
    before: Preferences,
    after: Preferences,
    approved: list[TrimItem] | None = None,
) -> bool:
    """設定画面の OK で、退避とバックアップの変更を確かめる 進めてよければ真

    書けない置き場はここで止める OK の後で書けないと分かると、本人は設定が効いたと思ったまま
    退避が既定の側へ戻り続ける

    ``approved`` を渡すと、容量の上限で消してよいと確かめた物をそこへ足す 設定を当てるときは
    この一覧に入っている物だけを消す（確かめに出していない物は消さない）
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
    pruned: list[TrimItem] = []
    if after.backup and (
        after.backup_generations < before.backup_generations
        or not before.backup
        or state_roots(after) != state_roots(before)
    ):
        # 世代数を減らした・バックアップを入れ直した・置き場を変えたときは、当てる先の置き場に
        # 世代数を超えた控えがあれば、OK でその場で詰める（見せた物だけを消す）
        # 置き場を変えた先に控えが多くても世代数は同じなので、減らしたときだけ見ると
        # 次の保存で黙って詰めてしまう（保存のときは 1 本ずつの入れ替えだけにしてある）
        # 既定の置き場も数える 選んだ置き場へ書けなかった保存の控えは既定の側にある
        pruned = plan_prune_all(after)
        if pruned:
            answer = QMessageBox.question(
                parent,
                "バックアップを減らす",
                f"残す数を {after.backup_generations} 世代にすると、OK を押したときに"
                f"今あるバックアップのうち {len(pruned)} 本を消します"
                "（古い物から 消したバックアップは戻せません）\n\n"
                f"{describe_trim(pruned)}\n\n減らしますか",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
            if approved is not None:
                approved.extend(pruned)
    tightened = after.state_limit_mb > 0 and (
        before.state_limit_mb == 0
        or after.state_limit_mb < before.state_limit_mb
        or state_roots(after) != state_roots(before)
    )
    if tightened:
        doomed_items = plan_trim_all(after, already=[p for item in pruned for p in item.paths])
        if doomed_items:
            # OK を押すとその場で片付ける 何が消えるかを先に見せないと、本人は上限を
            # 入れただけのつもりで控えを失う
            answer = QMessageBox.question(
                parent,
                "容量の上限",
                f"上限を {after.state_limit_mb}MB にすると、古い物から次を片付けます"
                "（消した物は戻せません）\n\n"
                f"{describe_trim(doomed_items)}\n\n"
                "開いている作業の退避・まだ復元を尋ねていない落ちた作業・"
                "各プロジェクトのいちばん新しいバックアップは消しません\n\n上限を入れますか",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
            if approved is not None:
                approved.extend(doomed_items)
    return True
