"""素材の字幕と場面の説明から探す窓（#326 素材の意味検索の人の入口）

素材一覧の〔探す…〕で開く AI の道具 ``search_media`` と同じ引き方
（:func:`~sashimono.core.search.search_media`）で、タイムラインに置いていない素材の字幕と、
AI が絵を見て覚えた場面の説明まで引く 見つけた行を選ぶと、編集画面がその所へ行く

引き方を AI と分けない 分けると、AI に頼めば見つかる場面が人の欄では見つからない
（またはその逆）になり、同じ「探す」の答えが入口で変わる
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.model import MediaItem, Project
from sashimono.core.search import SCENE, SceneNote, search_media
from sashimono.core.timebase import Rounding, format_timecode, seconds_to_frame

__all__ = ["MAX_SHOWN", "MediaSearchDialog"]

#: 一覧に並べる数 何万行の字幕で 1 文字だけ打っても一覧が固まらないように切る
#: 当たった語の多い順に並ぶので、上から見れば足りる 足りなければ語を足して絞る
MAX_SHOWN = 500

#: 行に持たせる 素材の ID と素材の秒
_MEDIA = Qt.ItemDataRole.UserRole
_SECONDS = Qt.ItemDataRole.UserRole + 1
_STREAM = Qt.ItemDataRole.UserRole + 2


class MediaSearchDialog(QDialog):
    """言葉を打って Enter で探す 行を選ぶ（ダブルクリック・Enter）とその所へ行く

    作品と説明は開くたび・探すたびに読み直す 窓を開いたまま編集しても、前の作品の
    字幕で探さない
    """

    #: 見つけた所へ行くよう要求された 引数は素材 ID と素材の秒と、字幕の当たりの音声ストリームの
    #: 番号（場面の説明は None） 番号を渡さないと、同じ素材の別の言語の音のクリップへ行く
    found_activated = Signal(str, float, object)

    def __init__(
        self,
        project: Callable[[], Project],
        notes_of: Callable[[MediaItem], Sequence[SceneNote]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("素材を探す")
        self._project = project
        self._notes_of = notes_of
        self._field = QLineEdit(self)
        self._field.setPlaceholderText("探す言葉（空白で区切ると、どれかを含む所）")
        self._field.setClearButtonEnabled(True)
        self._field.returnPressed.connect(self.search)
        self._summary = QLabel(
            "ひらがなとカタカナ・全角と半角は同じに見ます 置いていない素材の字幕と、"
            "AI が覚えた場面の説明も探します",
            self,
        )
        self._summary.setWordWrap(True)
        self._results = QListWidget(self)
        self._results.itemActivated.connect(self._activate)
        layout = QVBoxLayout(self)
        layout.addWidget(self._field)
        layout.addWidget(self._summary)
        layout.addWidget(self._results)
        self.resize(560, 420)

    @property
    def field(self) -> QLineEdit:
        return self._field

    @property
    def results(self) -> QListWidget:
        return self._results

    @property
    def summary(self) -> str:
        return self._summary.text()

    def search(self) -> None:
        """欄の言葉で探し直す 空なら一覧を空にする"""
        words = self._field.text().split()
        self._results.clear()
        if not words:
            self._summary.setText("探す言葉を打って Enter")
            return
        project = self._project()
        try:
            hits = search_media(project, words, notes_of=self._notes_of)
        except ValueError:
            # 揃えると空になる語（記号だけなど）は、どの行にも当たって絞れないので探さない
            self._summary.setText("記号だけの言葉では探せません")
            return
        names = {media.id: media.name for media in project.media}
        rate = project.rate
        for hit in hits[:MAX_SHOWN]:
            # 時刻は素材の頭から数える（タイムラインの位置ではない 置いていない素材もある）
            moment = Fraction(str(hit.start))
            when = format_timecode(seconds_to_frame(moment, rate, Rounding.FLOOR), rate)
            kind = "場面" if hit.kind == SCENE else "字幕"
            item = QListWidgetItem(f"{names.get(hit.media_id, '')}  {when}  [{kind}]  {hit.text}")
            item.setData(_MEDIA, str(hit.media_id))
            item.setData(_SECONDS, hit.start)
            item.setData(_STREAM, hit.stream)
            item.setToolTip(hit.text)
            self._results.addItem(item)
        if not hits:
            self._summary.setText(
                "見つかりません 別の言い方も空白で区切って並べると、どれかを含む所を探します"
            )
        elif len(hits) > MAX_SHOWN:
            self._summary.setText(
                f"{len(hits)} 件（上の {MAX_SHOWN} 件を表示 言葉を変えて絞ってください）"
            )
        else:
            self._summary.setText(f"{len(hits)} 件 選ぶとその所へ行きます")
        if hits:
            self._results.setCurrentRow(0)

    def _activate(self, item: QListWidgetItem) -> None:
        stream = item.data(_STREAM)
        self.found_activated.emit(
            str(item.data(_MEDIA)),
            float(item.data(_SECONDS)),
            int(stream) if stream is not None else None,
        )
