"""起動画面（#526）

アイコンを押してから編集画面が出るまで、何も映らないと起動できているのかが分からない
``QApplication`` を作った直後、編集画面の部品（numpy・PyAV・OpenGL）を読む前に、ロゴと版と
「起動しています」を出す 2026-11-24 に開発の環境で測った値は、``QApplication`` まで 0.29 秒・
編集画面の部品の import が温まって 1.66 秒・起動して最初の 1 回は 6.69 秒・``MainWindow`` を
作るのが 0.09 秒 待たせているのはほぼ部品の import

**このモジュールは軽く保つ** 読むのは Qt とテーマと同梱の絵だけ 編集画面の部品を読むと、
読み終えるまで起動画面を出せなくなる（``tests/ui/test_splash.py`` が別のプロセスで見張る）

ロゴは動かさない 重い import の間は Python が画面を描けない（import が終わるまでイベントを
回せない） 動かすには描く係を別のスレッドに置くか import を細かく分けて間に描くことになり、
どちらも起動の作りを複雑にする割に、止まる所では結局止まる 代わりに段の間で進み具合の
文字を切り替える（設定の「起動画面の動き」 切ると「起動しています」のまま動かない）
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImageReader, QMouseEvent, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from sashimono import __version__
from sashimono.core import userdirs
from sashimono.resources import LOGO_FILE, path_to
from sashimono.ui.theme import THEME_DARK, THEME_MODES, Colors

__all__ = [
    "SPLASH_MOTION_DEFAULT",
    "STEP_PARTS",
    "STEP_PROJECT",
    "STEP_WINDOW",
    "STILL_MESSAGE",
    "SplashChoices",
    "StartupSplash",
    "showing_splash",
    "splash_choices",
    "splash_picture",
]

#: 設定の「起動画面の動き」の既定（``Preferences.splash_motion`` もここから取る）
#: 起動画面は設定を読む前に出るので、既定を 2 か所に書くと食い違ったときに気付けない
SPLASH_MOTION_DEFAULT = True

#: 動きを切ったときと、最初の段に入る前に出す文字
STILL_MESSAGE = "起動しています"
#: 段ごとの文字 動きが入のときだけ切り替える
STEP_PARTS = "編集画面の部品を読んでいます"
STEP_PROJECT = "作品を開いています"
STEP_WINDOW = "編集画面を組み立てています"

#: 起動画面の大きさ（論理の画素） 1280x720 の画面でも邪魔にならず、版の文字が読める大きさ
_SIZE = QSize(440, 260)
#: ロゴの一辺
_LOGO = 112


@dataclass(frozen=True, slots=True)
class SplashChoices:
    """起動画面が使う本人の好み（``Preferences`` と同じ既定）"""

    theme: str = THEME_DARK
    motion: bool = SPLASH_MOTION_DEFAULT


def splash_choices(path: Path | None = None) -> SplashChoices:
    """設定のファイルからテーマと「起動画面の動き」だけを直に読む

    設定の読み手（``ui.workspace.PreferenceStore``）は編集画面の部品を読むので、起動画面より
    前には使えない 先にテーマだけ読んで明暗を合わせる 暗い地の起動画面から明るい編集画面へ
    切り替わると、明るいテーマを選んだ人の目に光る 読めない・無い・壊れた値は既定
    （読み手と同じ決まり ``tests/ui/test_splash.py`` が読み手と照らす）
    """
    # 名前は更新の係が起動の頭で読むのと同じ物を使う（ファイルの名前を 2 か所に書かない）
    from sashimono.update.flow import PREFERENCES_FILE

    place = path if path is not None else userdirs.config_root() / PREFERENCES_FILE
    try:
        data = json.loads(place.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return SplashChoices()
    if not isinstance(data, dict):
        return SplashChoices()
    plain = SplashChoices()
    theme = data.get("theme")
    motion = data.get("splash_motion")
    return SplashChoices(
        theme=theme if isinstance(theme, str) and theme in THEME_MODES else plain.theme,
        motion=motion if isinstance(motion, bool) else plain.motion,
    )


def _logo(side: int) -> QPixmap | None:
    """ロゴを ``side`` 画素の正方形で描く 描けなければ ``None``（文字だけの起動画面にする）

    SVG を読む部品（Qt の画像の差し込み）が欠けた配布版でも、起動そのものは止めない
    """
    reader = QImageReader(str(path_to(LOGO_FILE)))
    reader.setScaledSize(QSize(side, side))
    image = reader.read()
    if image.isNull():
        return None
    return QPixmap.fromImage(image)


def splash_picture(ratio: float = 1.0) -> QPixmap:
    """起動画面の地 ロゴ・名前・版 進み具合の文字は :class:`StartupSplash` が下に重ねる

    色は今のテーマから取る（:func:`~sashimono.ui.theme.apply_theme` を当てた後に呼ぶ）
    ``ratio`` は画面の拡大率 1 倍で描いて引き伸ばすと、150% の画面で文字がにじむ
    """
    pixmap = QPixmap(round(_SIZE.width() * ratio), round(_SIZE.height() * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Colors.WINDOW)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Colors.BORDER)
        painter.drawRect(QRectF(0.5, 0.5, _SIZE.width() - 1, _SIZE.height() - 1))
        logo = _logo(round(_LOGO * ratio))
        if logo is not None:
            logo.setDevicePixelRatio(ratio)
            painter.drawPixmap(int((_SIZE.width() - _LOGO) / 2), 28, logo)
        title = QFont(painter.font())
        title.setPixelSize(22)
        title.setBold(True)
        painter.setFont(title)
        painter.setPen(Colors.TEXT)
        painter.drawText(
            QRectF(0, 148, _SIZE.width(), 32), Qt.AlignmentFlag.AlignCenter, "Sashimono Edit"
        )
        small = QFont(painter.font())
        small.setPixelSize(13)
        small.setBold(False)
        painter.setFont(small)
        painter.setPen(Colors.TEXT_MUTED)
        painter.drawText(
            QRectF(0, 180, _SIZE.width(), 20), Qt.AlignmentFlag.AlignCenter, f"版 {__version__}"
        )
    finally:
        painter.end()
    return pixmap


class StartupSplash(QSplashScreen):
    """起動画面 進み具合の文字は :meth:`step` で切り替える

    前に出し続ける印（``WindowStaysOnTopHint``）は付けない 編集画面を組み立てる途中で出る
    確かめの窓や、ほかのソフトの窓を隠さないため
    """

    def __init__(self, *, motion: bool = SPLASH_MOTION_DEFAULT, steps: int = 3) -> None:
        screen = QApplication.primaryScreen()
        ratio = screen.devicePixelRatio() if screen is not None else 1.0
        super().__init__(splash_picture(ratio))
        self.setObjectName("startupSplash")
        self.motion = motion
        self._steps = max(1, steps)
        self._done = 0
        self._say(STILL_MESSAGE)

    def step(self, message: str) -> None:
        """次の段へ進んだことを出す 動きが切なら何も変えない

        出した後にイベントを 1 度回して描かせる 回さないと、次の重い import が終わるまで
        文字が変わらない
        """
        if not self.motion:
            return
        self._done = min(self._done + 1, self._steps)
        self._say(f"{message}（{self._done}/{self._steps}）")
        QApplication.processEvents()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        """押されても隠れない

        元の ``QSplashScreen`` は押されると隠れ、後の :meth:`step` では出し直さない 待ちかねて
        押した人ほど、残りの読み込みの間に起動しているかが分からなくなる（PR #550 の指摘）
        閉じるのは編集画面を出したとき（``finish``）と失敗したとき（``close``）だけ
        """
        event.accept()

    def _say(self, text: str) -> None:
        self.showMessage(
            text,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            QColor(Colors.TEXT_MUTED),
        )


@contextmanager
def showing_splash(
    application: QApplication, choices: SplashChoices, *, steps: int = 3
) -> Iterator[StartupSplash]:
    """起動画面を出して渡す 抜けるときは必ず閉じる

    途中で投げられても（部品が読めない・編集画面を作れない）閉じてから投げ直す 閉じないと、
    起動に失敗した後も起動画面だけが残り、固まったように見える
    編集画面を出したら :meth:`QSplashScreen.finish` で閉じる（窓が映るまで待ってから閉じる）
    """
    splash = StartupSplash(motion=choices.motion, steps=steps)
    splash.show()
    # 出した直後に描かせる 回さないと、最初の重い import が終わるまで何も映らない
    application.processEvents()
    try:
        yield splash
    finally:
        splash.close()
        splash.deleteLater()
