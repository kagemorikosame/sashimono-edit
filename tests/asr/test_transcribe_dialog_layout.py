"""字幕起こしの窓の並び 「起こす音声」の選びが「モデル」の行に重ならない

利用者の画面では、窓の左上の「モデル」の行に「(2ch 4…」と見える選びが重なり、名前の列が
潰れていた 音声が 1 本の素材では「起こす音声」の行を置かないのに、選びの部品は窓の子として
作ったままだったので、窓の左上（0, 0）に浮いて出ていた 窓を開いて部品の位置で確かめる
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QWidget

from sashimono.asr.service import TranscriptionService
from sashimono.asr.whisper import FasterWhisperBackend
from sashimono.engine.decode import probe_media
from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog
from tests.asr.test_gpu_fallback import _two_voices
from tests.media_fixtures import SampleMedia


def _opened(path: Path) -> TranscribeDialog:
    dialog = TranscribeDialog(probe_media(path), TranscriptionService(FasterWhisperBackend()))
    dialog.show()
    QApplication.processEvents()
    return dialog


def _shown(dialog: TranscribeDialog) -> list[tuple[QWidget, QRect]]:
    """見えている選びと名前の、窓の中の位置"""
    found: list[tuple[QWidget, QRect]] = []
    for widget in dialog.findChildren(QWidget):
        if isinstance(widget, QComboBox | QLabel) and widget.isVisible():
            top_left = widget.mapTo(dialog, widget.rect().topLeft())
            found.append((widget, QRect(top_left, widget.size())))
    return found


def _no_overlap(dialog: TranscribeDialog) -> None:
    placed = _shown(dialog)
    for index, (first, a) in enumerate(placed):
        for second, b in placed[index + 1 :]:
            if first.isAncestorOf(second) or second.isAncestorOf(first):
                continue
            assert not a.intersects(b), f"{first.objectName() or first} と {second} が重なる"


def test_a_single_voice_leaves_no_floating_choice(
    sample_av: SampleMedia, qt_application: object
) -> None:
    # 壊れると、行に置かない選びが窓の左上に浮き、「モデル」の行を潰す
    del qt_application
    dialog = _opened(sample_av.path)
    try:
        assert not dialog._stream.isVisible()
        _no_overlap(dialog)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_two_voices_get_their_own_row(media_dir: Path, qt_application: object) -> None:
    del qt_application
    dialog = _opened(_two_voices(media_dir))
    try:
        assert dialog._stream.isVisible()
        _no_overlap(dialog)
        # 名前の列は切れない（名前の文字が収まる幅がある）
        for widget, _ in _shown(dialog):
            if isinstance(widget, QLabel) and widget.text() in ("モデル", "起こす音声"):
                assert widget.width() >= widget.sizeHint().width()
        # 窓を狭めても重ならない
        dialog.resize(360, dialog.height())
        QApplication.processEvents()
        _no_overlap(dialog)
    finally:
        dialog.close()
        dialog.deleteLater()
