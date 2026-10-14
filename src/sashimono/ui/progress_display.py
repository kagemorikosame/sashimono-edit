"""時間の掛かる処理の進み具合を、ステータスバーと素材一覧へ出す

見た目は書き出し・字幕起こし・追加機能の導入の進み具合と同じ
（``QProgressBar`` を 0..1000 で使う 色は :mod:`sashimono.ui.theme` がまとめて決める）

文言を組み立てる所は部品から分けてある 部品を作らずに、何が出るかを試験で確かめるため
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

from sashimono.core.model import MediaId
from sashimono.engine.cache.progress import ProgressSnapshot

__all__ = [
    "ProgressIndicator",
    "describe_background",
    "describe_import",
    "finished_message",
    "overall_fraction",
    "row_notes",
]

#: 進み具合の棒の目盛り 書き出しの画面と同じにそろえる
STEPS = 1000


class ProgressIndicator(QWidget):
    """文言と進み具合の棒、あれば止めるボタン ステータスバーの右端に置く

    何も走っていない間は隠す 出したままにすると、終わったのか止まったのか分からない
    """

    def __init__(self, parent: QWidget | None = None, *, cancel_text: str | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel(self)
        self._bar = QProgressBar(self)
        self._bar.setRange(0, STEPS)
        # 数字は文言の側に出す 棒の中にも出すと同じ数が 2 か所に並ぶ
        self._bar.setTextVisible(False)
        self._bar.setFixedWidth(140)
        self._cancel: QPushButton | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._label)
        layout.addWidget(self._bar)
        if cancel_text is not None:
            self._cancel = QPushButton(cancel_text, self)
            layout.addWidget(self._cancel)
        self.hide()

    @property
    def cancel_button(self) -> QPushButton | None:
        return self._cancel

    @property
    def text(self) -> str:
        return self._label.text()

    @property
    def value(self) -> int:
        return self._bar.value()

    def show_progress(self, text: str, fraction: float, *, tooltip: str = "") -> None:
        self._label.setText(text)
        self._bar.setValue(round(max(0.0, min(1.0, fraction)) * STEPS))
        self.setToolTip(tooltip)
        self.show()


def describe_import(done: int, total: int, waiting: int) -> str:
    """読み込みの進み具合 ``waiting`` はこの後に控えている読み込みの回数"""
    text = f"素材を調べている {done}/{total} 本"
    if waiting:
        text += f"（ほかに {waiting} 回分が待っている）"
    return text


def overall_fraction(proxy: ProgressSnapshot, analysis: ProgressSnapshot) -> float:
    """控えと解析を合わせた進み具合 0..1

    片方ずつ割ってから平均しない 控え 1 本と解析 10 件なら、控えの 1 本が
    全体の半分に見えてしまう
    """
    total = proxy.total + analysis.total
    if total <= 0:
        return 1.0
    done = proxy.finished + proxy.partial + analysis.finished + analysis.partial
    return min(1.0, done / total)


def describe_background(proxy: ProgressSnapshot, analysis: ProgressSnapshot) -> str:
    """ステータスバーに出す 1 行 走っている方だけを並べる

    控えと解析を分けて出すのは、4K を読み込んだ直後にプレビューが重い理由が
    控えを作っている最中だからなのかを、見て分かるようにするため
    """
    parts: list[str] = []
    if proxy.busy:
        parts.append(f"控え {proxy.finished}/{proxy.total} 本 {round(proxy.fraction * 100)}%")
    if analysis.busy:
        # 解析は波形・サムネイル・場面の一覧（#321）を合わせて数える 前は「波形とサムネイル」と
        # 書いていたが、場面を調べている間もそう出ると、何に CPU を使っているのかを取り違える
        parts.append(
            f"解析 {analysis.finished}/{analysis.total} 件 {round(analysis.fraction * 100)}%"
        )
    failed = proxy.failed + analysis.failed
    text = "・".join(parts)
    if failed:
        text += f" 失敗 {failed} 件"
    return text


def finished_message(proxy: ProgressSnapshot, analysis: ProgressSnapshot) -> str:
    """ひと続きが終わったときの文言 失敗があれば 1 つ目の理由も出す

    理由を行の表示にだけ出すと、行の表示を切った人には理由が届かない
    理由はこのひと続きの物から取る 行に残っている失敗から取ると、前に失敗した
    別の素材の理由が、今の失敗の数と並んで出る
    """
    failed = proxy.failed + analysis.failed
    if not failed:
        return "控えと解析が終わった"
    reasons = [*proxy.recent_failures, *analysis.recent_failures]
    head = f"控えと解析が終わった 失敗 {failed} 件"
    return f"{head}: {reasons[0]}" if reasons else head


def row_notes(
    proxy: ProgressSnapshot, analysis: ProgressSnapshot
) -> dict[MediaId, tuple[str, str]]:
    """素材一覧の行に添える文言と、そのツールチップ（失敗の理由）

    失敗は作り直しを頼むまで残り、頼めば :class:`~sashimono.engine.cache.progress.JobBoard`
    の側で消える 同じ素材の同じ仕事で、走っている物と失敗が並ぶことは無い
    """
    notes: dict[MediaId, tuple[str, str]] = {}
    for media_id in {*proxy.running, *analysis.running, *proxy.failures, *analysis.failures}:
        words: list[str] = []
        reasons: list[str] = []
        _add(words, reasons, "控え", proxy, media_id)
        _add(words, reasons, "解析", analysis, media_id)
        if words:
            notes[media_id] = ("・".join(words), "\n".join(reasons))
    return notes


def _add(
    words: list[str],
    reasons: list[str],
    name: str,
    snapshot: ProgressSnapshot,
    media_id: MediaId,
) -> None:
    failure = snapshot.failures.get(media_id)
    running = snapshot.running.get(media_id)
    if running is not None:
        words.append(f"{name} {round(running * 100)}%")
    elif failure is not None:
        words.append(f"{name}できなかった" if name == "解析" else f"{name}を作れなかった")
        reasons.append(failure)
