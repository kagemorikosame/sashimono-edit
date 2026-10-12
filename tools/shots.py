r"""README と Wiki に載せる画面写真を、アプリ自身に撮らせる

    .venv\Scripts\python.exe tools\shots.py
    .venv\Scripts\python.exe tools\shots.py --list
    .venv\Scripts\python.exe tools\shots.py --only screenshot subtitle
    .venv\Scripts\python.exe tools\shots.py --wiki ..\sashimono-edit.wiki

**手で撮らない** 手で撮ると、撮った人の画面配置・重なった別の窓・本人の
ファイル名が写り込み、画面が変わるたびに同じ絵を作り直せない ここでは
見本のプロジェクトをその場で組み立て、``QWidget.grab()`` で窓そのものを
写すので、机の上に何が出ていても結果は変わらない 窓は画面に出さずに描かせる
（``WA_DontShowOnScreen``） 撮っている間に本人の机の上へ窓が出たり、別の窓が
重なって写ったりしない

見本の素材は ffmpeg でその場で作る（色の帯と正弦波） 本人の動画や音声は
絶対に使わない 設定・退避・キャッシュ・ホーム・一時フォルダ・ProgramData の
置き場も作業用のフォルダへ向けるので、撮る人の並びや前回の作業が写ることも、
撮り終わったあとに本人の設定が見本の状態で上書きされることもない
作業用のフォルダは ``%USERPROFILE%`` の外に作る（ユーザー名を含む場所が、
読み込んだ素材の場所として画面のどこかに出ても、写真に名前が残らないように）
既定はドライブの根、書けなければリポジトリの中の ``.work/shots`` リポジトリが
ホームの下にあってドライブの根にも書けないときは ``--work`` が要る

他人が作った配布物（AviUtl2 のエイリアス・YMM4 のアイテムテンプレート・
スクリプト）は写さない 再配布の条件が作者ごとに違うので、棚やテンプレートの
写真に使う物も、ここで書いた見本だけにする

Wiki の手順書の写真は ``--wiki`` に Wiki の clone を渡したときだけ撮り、
その ``images`` へ書き出す
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtCore import QEvent, QEventLoop, QPoint, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDockWidget,
    QLineEdit,
    QMainWindow,
    QMenu,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from sashimono import __version__
from sashimono.ai.host import ToolError
from sashimono.compat.aviutl.catalog import (
    PORTABLE_SCRIPTS_DIR,
    ScriptCatalog,
    script_catalog,
    set_script_catalog,
)
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import (
    TemplateCatalog,
    TemplateEntry,
    place,
    restyle,
)
from sashimono.core.commands import (
    AddEffect,
    Command,
    ParamPath,
    SetKeyframe,
    SetTranscript,
    insert_generated,
)
from sashimono.core.model import (
    ClipId,
    GeneratedSource,
    Project,
    ProjectSettings,
    Transcript,
    TranscriptSegment,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.definition import registry
from sashimono.effects.sources import TEXT
from sashimono.runtime import PackageStatus, PackStatus
from sashimono.ui.compat_dialog import CompatibilityDialog
from sashimono.ui.inspector import InspectorPanel
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import AUTO_QUALITY_TEXT, PreferencesDialog
from sashimono.ui.preview import PreviewWidget
from sashimono.ui.subtitle import SubtitlePanel
from sashimono.ui.template_dialog import TemplateDialog
from sashimono.ui.workspace import Preferences, PreferenceStore

ROOT = Path(__file__).resolve().parent.parent

#: 窓の大きさ README と Wiki に並べる幅（1280 前後）に合わせる 変えると README の
#: 見た目の縦横比まで変わるので、揃える意味でここ 1 か所に持つ
WINDOW_SIZE = (1280, 800)

#: 素材一覧と設定パネルの幅 窓が狭いぶん、プレビューを詰めて設定パネルへ回す
#: プレビューの列には下の帯が全部（全体の長さと「再生品質」の名前まで）入る幅を残す
#: 残さないと帯がそれを隠し、手順書の〔再生品質〕が写真のどこにも書かれていない
DOCK_WIDTHS = (250, 450)

#: 見本の素材 小さすぎるとサムネイルが潰れ、長すぎるとタイムラインが余る
SAMPLE_WIDTH, SAMPLE_HEIGHT, SAMPLE_SECONDS = 1280, 720, 8.0
SAMPLE_RATE = FrameRate(30)

#: テンプレートを置く写真の画面 見本のテンプレートは配布物と同じく 1080p 前提で
#: 座標と文字の大きさを書いている
FULL_HD = (1920, 1080)

#: 撮る前に画面を落ち着かせる回数 GL の用意・素材の解析・サムネイルの生成は
#: どれも後から届くので、1 回描いただけだと波形やサムネイルが入らない絵になる
SETTLE_ROUNDS = 40

#: 1 回あたりに待つ長さ（ミリ秒） 合計で 1 秒ほど回す
SETTLE_MS = 25

#: 真っ黒と見なす明るさ GL の中身が写らなかったときは 0 が並ぶ
BLACK_LEVEL = 8

#: 見本の素材を作る ffmpeg を待つ上限（秒）
FFMPEG_TIMEOUT = 300

#: 手順書で「ここを押す」を囲む枠の色 テーマの色と違う色にして、画面の部品と
#: 見分けが付くようにする（暗いテーマの上でも明るいテーマの上でも目立つ橙）
MARK_COLOR = QColor(255, 138, 0)
MARK_WIDTH = 3

#: 重い素材の手順書で、設定の窓を先読みの項目の下どこまで見せるか（論理画素）
#: 先読みのメモリ・別のスレッドの 2 行が入る高さ
PREFERENCES_TAIL = 76

#: 写真に写す見本のテンプレート README の本文がこの名前を引き合いに出す
SAMPLE_ALIAS_NAME = "見本の金文字"
SAMPLE_YMM4_NAME = "回転しながら登場"

#: 棚に並べる置き場の名前 **相対のまま渡す** 棚は選んだテンプレートの場所を
#: 画面に出すので、絶対パスだと撮った機械のフォルダ構成が写真に写る
TEMPLATE_DIR = "見本のテンプレート"

#: 見本のテンプレートを止めるフレーム 登場は 30 フレームかけて動くが、
#: Expo のイージングは前半でほとんど動き切る 頭で止めると大きさ 0 で何も
#: 映らず、30 フレーム目では動き終わって見えるので、途中の分かる所で止める
YMM4_TEMPLATE_FRAME = 6


class ShotError(RuntimeError):
    """写真が作れなかった 中身が黒いなど、出してはいけない絵のとき"""


class ShotSkippedError(RuntimeError):
    """この写真は撮らない 今ある写真をそのまま残す"""


@dataclass(frozen=True, slots=True)
class Context:
    """1 回の撮影で共有するもの"""

    #: 見本の素材（ffmpeg で作ったもの） 作れなかったときは ``None``
    media: Path | None
    #: 見本のスクリプトを置いた場所
    script_root: Path
    #: 見本のテンプレートを置いた場所（作業用のフォルダからの相対）
    template_root: Path


@dataclass(frozen=True, slots=True)
class Shot:
    """撮る写真 1 枚"""

    name: str
    #: README か Wiki のどこに出るか 一覧に出して、撮り直す前に見当が付くようにする
    caption: str
    take: Callable[[Context], QImage]
    #: 見本の素材（ffmpeg で作る映像）が要るか
    #: 要らない写真まで ffmpeg に付き合わせない ffmpeg の無い機械で棚や
    #: スクリプトの写真だけを撮りたいことがある
    needs_media: bool = True
    #: Wiki の手順書の写真か 真なら ``--wiki`` の ``images`` へ書き出す
    wiki: bool = False


# --- 見本の素材・スクリプト・テンプレート ---


def make_sample_media(directory: Path) -> Path:
    """見本の映像と音声を作る

    本人の素材は使わない 色の帯は絵が毎フレーム変わるのでサムネイルが揃わず、
    正弦波に強弱を付けてあるので波形も平らにならない
    """
    path = directory / "見本.mp4"
    if path.exists():
        return path
    directory.mkdir(parents=True, exist_ok=True)
    video = f"testsrc2=size={SAMPLE_WIDTH}x{SAMPLE_HEIGHT}:rate=30:duration={SAMPLE_SECONDS}"
    # 素の正弦波は振幅が一定で、波形が 1 本の帯になる 無音カットの説明に使う絵
    # なので、強弱と切れ目が見えるように揺らしてから持ち上げる
    audio = (
        f"sine=frequency=440:duration={SAMPLE_SECONDS}:sample_rate=48000"
        ",tremolo=f=1.5:d=0.9,volume=14dB"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        video,
        "-f",
        "lavfi",
        "-i",
        audio,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ac",
        "2",
        str(path),
    ]
    # 出力を捨てない 捨てると、素材が作れなかったときに終了コードしか残らず、
    # 何が悪いのか（PATH に無い・codec が無い）が分からない
    # 待ち時間にも上限を置く ffmpeg が固まると、撮影がそこで止まったままになる
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    except FileNotFoundError as exc:
        raise ShotError("ffmpeg が PATH に無いので見本の素材を作れない") from exc
    except subprocess.CalledProcessError as exc:
        raise ShotError(f"見本の素材を作れない: {(exc.stderr or '').strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ShotError("見本の素材を作る ffmpeg が終わらない") from exc
    return path


#: 見本の AviUtl スクリプト 配布物は再配布の条件が作者ごとに違うのでリポジトリに
#: 入れられない 制御行から設定欄が組み上がることを見せるのが目的なので、
#: その 3 行を持つ最小のスクリプトをここで作る（README に載っている 3 行と同じ）
SAMPLE_SCRIPT = """@ゆらゆら
--track0:振れ幅,0,500,40,1
--track1:速さ,0.1,10,2,0.1
--check0:横に揺れる,0
local swing = math.sin(obj.time * obj.track1 * 2 * math.pi) * obj.track0
if obj.check0 == 1 then
    obj.ox = obj.ox + swing
else
    obj.oy = obj.oy + swing
end
"""


def install_sample_script(directory: Path) -> str:
    """見本のスクリプトを置いて登録し、エフェクト種別を返す"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "見本のゆらゆら.anm2"
    path.write_text(SAMPLE_SCRIPT, encoding="utf-8")

    catalog = ScriptCatalog(roots=(directory,))
    set_script_catalog(catalog)
    catalog.scan()
    catalog.register_all()
    entries = catalog.all()
    if not entries:
        raise ShotError("見本のスクリプトを読み込めなかった")
    return entries[0].identifier


def _alias(*sections: tuple[str, Sequence[str]]) -> str:
    """AviUtl2 のエイリアス（``.object``）の本文 節の名前と ``キー=値`` の行の組から作る"""
    lines = ["[Object]", "frame=0,149"]
    for index, (name, rows) in enumerate(sections):
        lines += [f"[Object.{index}]", f"effect.name={name}", *rows]
    return "\n".join(lines) + "\n"


#: 見本のエイリアス 書き方は AviUtl2 が書き出す ``.object`` に合わせてある
#: 名前と見た目はここで決めた物で、配布物の写しではない
SAMPLE_ALIASES: dict[str, str] = {
    SAMPLE_ALIAS_NAME: _alias(
        (
            "テキスト",
            (
                "サイズ=96.00",
                "文字色=ffd94a",
                "文字装飾=標準文字",
                "文字揃え=中央揃え[中]",
                "テキスト=見本の金文字",
            ),
        ),
        # グラデーションは縁取りより先に掛ける 後に掛けると縁まで金色に塗られ、
        # 文字の形が縁と混ざって読めなくなる
        (
            "グラデーション",
            ("強さ=100.00", "角度=0.00", "幅=100.00", "開始色=fff6c0", "終了色=c98a00"),
        ),
        ("縁取り", ("サイズ=5", "縁色=3a2000")),
        ("標準描画", ("X=0.00", "Y=380.00")),
    ),
    "見本の白縁": _alias(
        (
            "テキスト",
            (
                "サイズ=72.00",
                "文字色=ffffff",
                "影・縁色=000000",
                "文字装飾=縁取り文字（太）",
                "文字揃え=中央揃え[中]",
                "テキスト=見本の字幕",
            ),
        ),
        ("標準描画", ("X=0.00", "Y=400.00")),
    ),
    "見本の影付き": _alias(
        (
            "テキスト",
            (
                "サイズ=80.00",
                "文字色=7fd8ff",
                "影・縁色=002040",
                "文字装飾=影付き文字",
                "文字揃え=中央揃え[中]",
                "テキスト=見本の見出し",
            ),
        ),
        ("標準描画", ("X=0.00", "Y=-380.00")),
    ),
}

_TEXT_ITEM = "YukkuriMovieMaker.Project.Items.TextItem, YukkuriMovieMaker"


def _ymm4_text(text: str, *, length: int, **values: object) -> dict[str, object]:
    """YMM4 のテキストアイテム 1 つ 項目の名前は YMM4 が書き出す形に合わせる"""
    return {"$type": _TEXT_ITEM, "Text": text, "Length": length, "Layer": 0, "Frame": 0, **values}


def _animation(*values: float, style: str = "Expo_Out") -> dict[str, object]:
    """YMM4 の値のアニメーション 並びは中間点（``KeyFrames.Frames``）の区切りに対応する"""
    return {"Values": [{"Value": value} for value in values], "Span": 0.0, "AnimationType": style}


#: 見本の YMM4 のアイテムテンプレート 1 ファイルに何本も入る形（ZIP の中の
#: ``catalog.json``）は実物どおり 中身はここで決めた物
SAMPLE_YMM4: tuple[tuple[str, dict[str, object]], ...] = (
    (
        SAMPLE_YMM4_NAME,
        _ymm4_text(
            "見本のタイトル",
            length=90,
            FontSize=110.0,
            FontColor="#FFFFE680",
            KeyFrames={"Frames": [30]},
            Zoom=_animation(0.0, 100.0, 100.0),
            Rotation=_animation(-90.0, 0.0, 0.0),
        ),
    ),
    (
        "下から出る字幕",
        _ymm4_text(
            "見本の字幕",
            length=90,
            FontSize=72.0,
            FontColor="#FFFFFFFF",
            KeyFrames={"Frames": [20]},
            Y=_animation(540.0, 400.0, 400.0),
        ),
    ),
    (
        "ふわっと出る見出し",
        _ymm4_text(
            "見本の見出し",
            length=90,
            FontSize=88.0,
            FontColor="#FF9FE0FF",
            KeyFrames={"Frames": [20]},
            Opacity=_animation(0.0, 100.0, 100.0, style="Sine_Out"),
            Y=_animation(-300.0, -360.0, -360.0, style="Sine_Out"),
        ),
    ),
)


def write_sample_templates(directory: Path) -> Path:
    """見本のテンプレートを書き出し、その置き場を返す

    配布物は写さない 再配布の条件が作者ごとに違い、条件が分からない物を写真に
    載せると、その作者の物を勝手に配っているのと同じになる
    """
    aliases = directory / "見本の字幕"
    aliases.mkdir(parents=True, exist_ok=True)
    for name, body in SAMPLE_ALIASES.items():
        (aliases / f"{name}.object").write_text(body, encoding="utf-8")

    document = {
        "ItemTemplates": [
            {"Name": f"見本/{name}", "Path": ["見本", name], "Items": [item]}
            for name, item in SAMPLE_YMM4
        ]
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("catalog.json", json.dumps(document, ensure_ascii=False))
    (directory / "見本のテンプレート.ymmt").write_bytes(buffer.getvalue())
    return directory


# --- 撮る ---


def settle(widget: QWidget, rounds: int = SETTLE_ROUNDS) -> None:
    """届いていない仕事を片付けてから撮る

    解析（波形・サムネイル）はワーカースレッドで終わり、タイマーで画面へ
    入る 回さずに撮ると、波形もサムネイルも無いタイムラインが写る
    時間を渡して回すのは、間隔の空いたタイマー（解析の反映は 250ms ごと）を
    確実に 1 度は通すため 回数だけだと一瞬で回り切って何も届かない
    """
    for _ in range(rounds):
        # **本当に時間を進める** ``processEvents`` に時間を渡しても、処理する
        # イベントが尽きた時点で戻ってくる それだと 40 回が一瞬で回り切り、
        # 解析の反映（250ms ごと）を 1 度も通さないまま撮ることになる
        # （波形とサムネイルの無いタイムラインが写る）
        loop = QEventLoop()
        QTimer.singleShot(SETTLE_MS, loop.quit)
        loop.exec()
    # 捨てる約束になったウィジェットを本当に捨てる 画面配置を戻すと Qt は古い
    # タブの帯を deleteLater で捨てるが、その始末は「そのイベントループを抜けた
    # とき」に行われる ここは自前で回しているだけで抜けないので、捨てられていない
    # 帯が窓の左上（位置 0,0）のまま残り、メニューに重なって写る
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    widget.repaint()
    QApplication.processEvents()


def hide_from_screen(widget: QWidget) -> None:
    """窓を机の上に出さずに描かせる

    出すと、撮っている間ずっと本人の机の上に見本の窓が出て、触ると撮る中身が
    変わる 画面に出さなくても Qt は窓として描くので、``grab()`` で写せる
    """
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)


def logical(image: QImage, width: int, height: int) -> QImage:
    """画面の拡大率（125% など）で撮った画像を、論理的な大きさへ戻す

    戻さないと撮った機械によって大きさが変わり、README に並べる写真が揃わない
    """
    if (image.width(), image.height()) == (width, height):
        return image
    scaled = image.scaled(
        width,
        height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    scaled.setDevicePixelRatio(1.0)
    return scaled


def grab(widget: QWidget) -> QImage:
    """ウィジェットを画像にする GL の中身は自分で貼り直す

    ``QWidget.grab()`` は裏の描画面（バッキングストア）を写す QOpenGLWidget の
    中身がそこへ合成されるかは環境で変わり、外れるとプレビューだけ真っ黒になる
    ので、GL の面は ``grabFramebuffer()`` で取り直して同じ場所へ重ねる
    """
    settle(widget)
    # 1 枚目は捨てる パネルを重ねたときのタブは、窓の大きさを変えた直後だと
    # 前の位置のまま描かれることがある（メニューの上に重なった絵になる）
    # 1 度描かせると並べ直しが終わるので、2 枚目を使う
    widget.grab()
    settle(widget, rounds=4)

    image = widget.grab().toImage()
    ratio = image.width() / max(1, widget.width())

    for surface in widget.findChildren(QOpenGLWidget):
        if not surface.isVisible():
            continue
        top_left = surface.mapTo(widget, QPoint(0, 0))
        target = QRect(
            round(top_left.x() * ratio),
            round(top_left.y() * ratio),
            round(surface.width() * ratio),
            round(surface.height() * ratio),
        )
        paste(image, target, surface.grabFramebuffer())

    return logical(image, widget.width(), widget.height())


def paste(image: QImage, target: QRect, source: QImage) -> None:
    """``target``（画素そのものの座標）へ ``source`` を貼る

    **貼る前に画像の拡大率を 1.0 にする** ``QPainter`` は相手の QImage が持つ
    devicePixelRatio で座標を変換するので、そのままだと拡大率 125% の機械では
    位置も大きさも二重に拡大され、GL の面が画面の外まではみ出して貼られる
    （プレビューの場所には何も写らないまま、明るさの見張りだけが通る）
    """
    image.setDevicePixelRatio(1.0)
    painter = QPainter(image)
    painter.drawImage(target, source)
    painter.end()


def mark(image: QImage, rects: Sequence[QRect], *, grow: tuple[int, int] = (4, 3)) -> None:
    """手順書で「ここを押す」所を枠で囲む ``rects`` は窓の論理座標

    枠は部品より少し外へ広げる（``grow`` は横と縦） 部品の縁にぴったり重ねると、
    部品の枠線と見分けが付かない 広げた枠は画像の内側へ収める 窓の端の部品を
    囲むと、枠の外側の辺が画像の外へ出て、囲みが欠けて見える
    """
    if not rects:
        return
    image.setDevicePixelRatio(1.0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(MARK_COLOR)
    pen.setWidth(MARK_WIDTH)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    half = MARK_WIDTH // 2 + 1
    inside = QRect(0, 0, image.width(), image.height()).adjusted(half, half, -half, -half)
    across, down = grow
    for rect in rects:
        grown = rect.adjusted(-across, -down, across, down).intersected(inside)
        painter.drawRoundedRect(QRectF(grown), 5, 5)
    painter.end()


def rect_in(window: QWidget, widget: QWidget) -> QRect:
    """``widget`` が ``window`` の中で占める範囲"""
    return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())


def brightest(image: QImage, rect: QRect) -> int:
    """その範囲でいちばん明るい画素 全部を見ずに格子状に拾う"""
    peak = 0
    steps = 48
    for row in range(steps):
        for column in range(steps):
            x = rect.x() + rect.width() * column // steps
            y = rect.y() + rect.height() * row // steps
            color = image.pixelColor(x, y)
            peak = max(peak, color.red(), color.green(), color.blue())
    return peak


def preview_rect(window: QWidget) -> QRect | None:
    """窓の中でプレビューが占める範囲 **窓を閉じる前に呼ぶ**"""
    preview = window.findChild(PreviewWidget)
    if preview is None:
        return None
    return rect_in(window, preview)


def take_editor_shot(window: MainWindow) -> QImage:
    """編集画面を写して、プレビューの中身まで写っているか確かめる

    GL の面が撮れないと、気付かないまま真っ黒なプレビューの写真が README に
    載る 黙って出さず、ここで止める

    **窓を閉じる前に確かめる** 閉じたあとでは位置も大きさも当てにならず、
    真っ黒を見つけられないまま通ってしまう
    """
    image = grab(window)
    rect = preview_rect(window)
    # 何も置いていない画面（手順書の最初の 1 枚）はプレビューが黒くて正しいので見ない
    showing = window.project.timeline.duration > 0
    if rect is not None and showing and brightest(image, rect) < BLACK_LEVEL:
        raise ShotError("プレビューが真っ黒 GL の中身が撮れていない")
    return image


def menu_of(window: QMainWindow, title: str) -> QMenu:
    """メニューバーの ``title`` のメニュー"""
    for action in window.menuBar().actions():
        menu = action.menu()
        if isinstance(menu, QMenu) and menu.title() == title:
            return menu
    raise ShotError(f"メニューが無い: {title}")


def action_in(menu: QMenu, text: str) -> QRect:
    """メニューの中の項目 ``text`` の範囲（メニューの中の座標）"""
    for action in menu.actions():
        if action.text() == text:
            return menu.actionGeometry(action)
    raise ShotError(f"{menu.title()} に {text} が無い")


def with_menu(
    window: MainWindow, title: str, items: Sequence[str], *, extra: Sequence[QRect] = ()
) -> QImage:
    """メニュー ``title`` を開いた絵 ``items`` の項目を枠で囲む

    本物のメニューは開くと別の窓（ポップアップ）になり、``grab()`` では親の窓に
    写らない メニューだけを描かせて、メニューバーの下へ重ねる
    """
    image = take_editor_shot(window)
    bar = window.menuBar()
    menu = menu_of(window, title)
    picture = menu_picture(menu)
    title_rect = bar.actionGeometry(menu.menuAction())
    origin = bar.mapTo(window, title_rect.bottomLeft()) + QPoint(0, 1)
    paste(image, QRect(origin, picture.size()), picture)
    mark(image, [QRect(bar.mapTo(window, title_rect.topLeft()), title_rect.size()), *extra])
    # 項目は縦へ広げない 隣り合う 2 つを囲むと、広げた分だけ枠が重なって 1 つに見える
    mark(image, [action_in(menu, item).translated(origin) for item in items], grow=(2, -1))
    return image


def menu_picture(menu: QMenu) -> QImage:
    """メニューを机に出さずに開き、その絵を返す

    開かずに ``grab()`` すると、項目の幅が文言の変わる前（「元に戻す: …」の後ろの
    部分を足す前）のまま測られ、文言とショートカットが重なって描かれる 開けば
    Qt が項目を並べ直す 机に出さないのは、マウスの位置で別の項目が光ると、
    撮るたびに絵が変わるため
    """
    hide_from_screen(menu)
    menu.popup(QPoint(0, 0))
    settle(menu, rounds=4)
    picture = logical(menu.grab().toImage(), menu.width(), menu.height())
    menu.hide()
    QApplication.processEvents()
    return picture


@contextmanager
def editor(project: Project | None = None) -> Iterator[MainWindow]:
    """編集画面を出す 撮り終わったら必ず閉じる"""
    window = MainWindow(project, confirm_unsaved=False)
    hide_from_screen(window)
    # 窓を閉じるたびに画面配置が保存され、次の 1 枚がそれを引き継ぐ 直前に撮った
    # 写真でどのパネルが前に出ていたかによって絵が変わるので、毎回既定へ戻す
    window.reset_layout()
    window.resize(*WINDOW_SIZE)
    window.show()
    settle(window)
    # 設定パネルを広げる 既定の幅だとエフェクトの見出しの ✕ や値の単位が切れ、
    # 写真の中で積んだエフェクトの名前と操作が読めない
    docks = [window.findChild(QDockWidget, name) for name in ("media", "inspector")]
    if all(dock is not None for dock in docks):
        window.resizeDocks(
            [dock for dock in docks if dock is not None],
            list(DOCK_WIDTHS),
            Qt.Orientation.Horizontal,
        )
        settle(window, rounds=4)
    try:
        yield window
    finally:
        window.close()
        QApplication.processEvents()


def sample_project(width: int = SAMPLE_WIDTH, height: int = SAMPLE_HEIGHT) -> Project:
    """見本のプロジェクト 素材はまだ持たない

    置き方の方式は、画面から新しく作ったときの既定（設定の既定）に合わせる
    写真だけ別の方式だと、手順書のとおりに作った人の画面とトラックの名前が食い違う
    テンプレートを置く写真だけ 1080p にする 見本のテンプレートは 1920x1080 を
    前提に座標と文字の大きさを書いているので、小さい画面で開くと指定どおりの
    位置に出ない
    """
    layer_mode = Preferences().new_project_layers
    return Project.create(
        ProjectSettings(width=width, height=height, frame_rate=SAMPLE_RATE, layer_mode=layer_mode)
    )


def first_video_clip(window: MainWindow) -> ClipId:
    """素材の絵を出すクリップのうち、いちばん頭の物

    映像トラックだけを見ない 混合の方式（新しく作ったときの既定）では、絵も音も
    同じ種類のレイヤーに並ぶ
    """
    clips = [
        clip
        for track in window.project.timeline.tracks
        for clip in track.clips
        if clip.media_id is not None and clip.show_picture and clip.audio_stream is None
    ]
    if not clips:
        raise ShotError("映像クリップが無い")
    return min(clips, key=lambda clip: clip.timeline_start).id


def top_text_clip(window: MainWindow) -> ClipId:
    """いちばん上に置いたテキストのクリップ"""
    for track in reversed(list(window.project.timeline.tracks)):
        for clip in track.clips:
            if clip.source is not None and clip.source.kind == "text":
                return clip.id
    raise ShotError("テキストのクリップが無い")


def import_sample(window: MainWindow, context: Context) -> None:
    """見本の素材を読み込み、置き終わるまで待つ

    読み込みは裏で素材を調べてから置く 待たずに撮ると、素材の無い空の
    タイムラインが写る
    """
    window.import_media([sample_media(context)])
    if not window.wait_for_imports():
        raise RuntimeError("見本の素材の読み込みが終わらない")
    settle(window)


def build_sample_timeline(window: MainWindow, context: Context) -> None:
    """見本の素材を読み込み、テロップとエフェクトを載せる

    README の先頭に出る絵なので、この 1 枚で「素材・波形・テロップ・
    エフェクトの設定」が一度に見えるようにしてある
    """
    import_sample(window, context)

    text = TEXT.create(
        text="見本のテロップ",
        size=72,
        border_width=6,
        pos_y=-260,
        color=(1.0, 0.93, 0.2, 1.0),
    )
    window.apply_commands(_place_text(window, text, at_frame=30, duration=150), "テキストを追加")

    clip = first_video_clip(window)
    glow = registry.require("glow").create(threshold=0.35, strength=1.8, radius=32.0)
    commands: list[Command] = [AddEffect(clip, glow)]
    path = ParamPath.of_effect(clip, glow.id, "radius")
    # 曲線を出すために点を 3 つ置く 1 つだけだとグラフが平らな線になり、
    # 「キーフレームが打てる」ことが絵から伝わらない
    commands += [
        SetKeyframe(path, 0, 4.0),
        SetKeyframe(path, 60, 64.0),
        SetKeyframe(path, 150, 8.0),
    ]
    window.apply_commands(commands, "エフェクトを追加")
    window.select_clip(clip)
    window.seek(45)
    settle(window)
    # 足したエフェクトの欄まで送る 上のままだとクリップが最初から持つ描画の欄だけが
    # 写り、「エフェクトを積んだ」ことが絵から分からない
    _scroll_to_end(window.findChild(InspectorPanel))


def _place_text(
    window: MainWindow, source: GeneratedSource, *, at_frame: int, duration: int
) -> list[Command]:
    return insert_generated(window.project, source, at_frame=at_frame, duration=duration)


SAMPLE_SEGMENTS = (
    ("えーと 今日は見本のプロジェクトを開いています", 0.4, 3.0),
    ("字幕は素材に紐付くので カットしてもずれません", 3.4, 5.8),
    ("あのー 無音カットも同じ仕組みの上で動きます", 6.2, 7.9),
)


def sample_transcript() -> Transcript:
    """見本の起こし結果 実際に音声認識を走らせない

    走らせると、撮るたびに文言が変わり、2 GB の実行環境も要る 字幕パネルの
    見た目を見せるのが目的なので、同じ形の結果をここで組む
    """
    return Transcript(
        segments=tuple(
            TranscriptSegment(
                start=Fraction(start).limit_denominator(1000),
                end=Fraction(end).limit_denominator(1000),
                text=text,
            )
            for text, start, end in SAMPLE_SEGMENTS
        ),
        language="ja",
        model="見本",
    )


# --- 写真ごとの組み立て ---


def sample_media(context: Context) -> Path:
    """見本の素材 作れていなければ、この写真は失敗として数える

    飛ばす（``ShotSkippedError``）にしないのは、ffmpeg は開発環境の前提だから
    素材の要らない写真（棚・スクリプト）はこの失敗に巻き込まれず撮れる
    """
    if context.media is None:
        raise ShotError("見本の素材が無い（ffmpeg が PATH に要る）")
    return context.media


def shot_editor(context: Context) -> QImage:
    with editor(sample_project()) as window:
        build_sample_timeline(window, context)
        return take_editor_shot(window)


def _with_transcript(window: MainWindow, context: Context) -> SubtitlePanel:
    """見本の素材を読み込み、見本の起こし結果を入れて字幕パネルを前に出す"""
    import_sample(window, context)
    media = window.project.media[0]
    window.apply_commands([SetTranscript(media.id, sample_transcript())], "字幕を更新")
    window.show_subtitles()
    panel = window.findChild(SubtitlePanel)
    if panel is None:
        raise ShotError("字幕パネルが無い")
    panel.select_media(media.id)
    return panel


def shot_subtitle(context: Context) -> QImage:
    with editor(sample_project()) as window:
        _with_transcript(window, context)
        window.seek(100)
        settle(window)
        return take_editor_shot(window)


#: AI の写真に見せる会話 **本物の Claude には送らない** 送ると、撮るたびに文面が
#: 変わり、課金の要る呼び出しになる 道具の名前と引数の書き方は、パネルが本物の
#: 会話で出す形（``▸ 道具 引数``）に合わせる 絵の中のタイムラインも、この会話の
#: とおりに道具が実際に組み立てる
SAMPLE_CHAT: tuple[tuple[str | None, str], ...] = (
    ("あなた", "冒頭 1 秒を切って 画面下に黄色いテロップを 3 秒入れて"),
    (None, "▸ list_clips"),
    (None, "▸ split_clip frame=30"),
    (None, "▸ delete_clip ripple=True"),
    (None, "▸ add_text text=見本のテロップ, at_frame=0, duration=90, pos_y=-280"),
    (
        "Claude",
        "冒頭の 1 秒（30 フレーム）を切って詰め、**見本のテロップ** を頭から 3 秒、"
        "画面下に黄色で置きました 色や位置を変えたいときは続けて書いてください",
    ),
)


def shot_ai(context: Context) -> QImage:
    """AI パネル 見本の会話を見せる **本物のセッションは走らせない**"""
    from sashimono.ui.chat import ChatPanel

    with editor(sample_project()) as window:
        import_sample(window, context)
        # 会話のとおりに組み立てる 絵のタイムラインと会話が食い違うと、AI が
        # 言ったことと違う編集をしたように見える
        clip = first_video_clip(window)
        window.select_clip(clip)
        window.seek(30)
        # 画面の〔再生ヘッドで分割〕〔削除して詰める〕と同じ入口を通す
        window._timeline.split_at_playhead()
        window.select_clip(first_video_clip(window))
        window._timeline.delete_selected(ripple=True)
        text = TEXT.create(
            text="見本のテロップ", size=72, border_width=6, pos_y=-280, color=(1.0, 0.9, 0.1, 1.0)
        )
        window.apply_commands(_place_text(window, text, at_frame=0, duration=90), "AI: テロップ")
        window.seek(40)

        window.show_chat()
        panel = window.findChild(ChatPanel)
        if panel is None:
            raise ShotError("AI パネルが無い")
        _fake_conversation(panel)
        settle(window)
        return take_editor_shot(window)


def _fake_conversation(panel: QWidget) -> None:
    """AI パネルに見本の会話を並べ、使える状態の見た目にする

    実行環境（Claude Agent SDK と Claude Code 本体）が入っていない機械では、パネルは
    入力欄を閉じて「環境を導入」の案内を出す 撮る機械によって別の絵にならないよう、
    導入とログインの案内は下げる 会話は本物の会話と同じ描き方（``_say`` と ``_note``）で
    並べるので、色も字の組み方も本物と変わらない
    """
    from sashimono.ui.chat import ChatPanel

    if not isinstance(panel, ChatPanel):
        raise ShotError("AI パネルではない")
    panel._show_parts(False)
    panel._choices.setVisible(True)
    panel._login_box.setVisible(False)
    panel._input.setEnabled(True)
    panel._send_button.setEnabled(True)
    panel._view.clear()
    panel._log.clear()
    for who, text in SAMPLE_CHAT:
        if who is None:
            panel._note(text)
        else:
            panel._say(who, text)


def _with_script(window: MainWindow, context: Context) -> ClipId:
    """見本のスクリプトを積んだテキストを置き、選んだ状態にする"""
    kind = install_sample_script(context.script_root)
    text = TEXT.create(text="AviUtl の\nスクリプト", size=96, border_width=4)
    window.apply_commands(_place_text(window, text, at_frame=0, duration=120), "テキストを追加")
    clip = top_text_clip(window)
    window.apply_commands([AddEffect(clip, registry.require(kind).create())], "エフェクト")
    window.select_clip(clip)
    window.seek(20)
    settle(window)
    return clip


def shot_aviutl(context: Context) -> QImage:
    """AviUtl のスクリプト 見本のスクリプトを 1 本だけ積む"""
    with editor(sample_project()) as window:
        _with_script(window, context)
        # 設定パネルを下まで送る 見せたいのは、制御行から組み上がったスクリプトの
        # 設定欄 上のままだとテキストの項目だけが写り、肝心の所が切れる
        _scroll_to_end(window.findChild(InspectorPanel))
        return take_editor_shot(window)


def _scroll_to_end(panel: QWidget | None) -> None:
    """パネルの縦送りを終わりまで送る 送れないものは何もしない"""
    if panel is None:
        return
    area = panel.findChild(QScrollArea)
    if area is None:
        return
    settle(panel, rounds=4)
    bar = area.verticalScrollBar()
    bar.setValue(bar.maximum())


def template_entries(context: Context) -> list[TemplateEntry]:
    return TemplateCatalog().scan((context.template_root,))


def _named(entry: TemplateEntry, name: str) -> bool:
    """テンプレートの名前が ``name`` か YMM4 の物は名前にフォルダ（``見本/``）が付く"""
    return entry.name.rsplit("/", 1)[-1] == name


def find_template(context: Context, name: str) -> TemplateEntry:
    for entry in template_entries(context):
        if _named(entry, name):
            return entry
    raise ShotError(f"見本のテンプレートが無い: {name}")


def _shelf(context: Context, name: str) -> TemplateDialog:
    """棚を開いて、``name`` のテンプレートを選んだ状態にする"""
    # 走査は棚（ダイアログ）に任せて 1 度だけにする 自分でも数えると、同じ
    # `rglob` が 2 回走るうえ、選ぶ相手と画面に並んでいる物がずれる余地が残る
    catalog = TemplateCatalog()
    dialog = TemplateDialog(catalog, roots=(context.template_root,))
    tree = dialog.findChild(QTreeWidget)
    entry = next((entry for entry in catalog.all() if _named(entry, name)), None)
    if tree is None or entry is None:
        dialog.close()
        raise ShotError(f"棚に見本のテンプレートが並ばない: {name}")
    chosen = _item_for(tree, entry)
    if chosen is not None:
        tree.setCurrentItem(chosen)
        tree.scrollToItem(chosen)
    return dialog


def shot_templates(context: Context) -> QImage:
    """テンプレートの棚（AviUtl2 のエイリアス）"""
    return _grab_shelf(_shelf(context, SAMPLE_ALIAS_NAME))


def shot_ymm4_shelf(context: Context) -> QImage:
    """テンプレートの棚（YMM4 のアイテムテンプレート）"""
    return _grab_shelf(_shelf(context, SAMPLE_YMM4_NAME))


#: 棚の見本がそろうのを待つ上限（ミリ秒） 待つ長さではなく、知らせが来ないまま止まり
#: 続けないための見張り そろった知らせ（``thumbnails_done``）が来ればすぐに撮る
SHELF_GUARD_MS = 120_000


def _grab_shelf(dialog: TemplateDialog) -> QImage:
    """棚を撮る 一覧の見本が描き終わってから撮る

    見本は別のスレッドで描いて後から埋まる 描き終わる前に撮ると、地だけの小さな見本が写る
    """
    hide_from_screen(dialog)
    dialog.show()
    wait_for_thumbnails(dialog)
    return _grab_dialog(dialog)


def wait_for_thumbnails(dialog: TemplateDialog) -> None:
    """見えている見本がそろうまで待つ 時間ではなく、棚のそろった知らせで抜ける"""
    if dialog.thumbnails_settled():
        return
    loop = QEventLoop()
    dialog.thumbnails_done.connect(loop.quit)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(SHELF_GUARD_MS)
    try:
        loop.exec()
    finally:
        guard.stop()
        dialog.thumbnails_done.disconnect(loop.quit)
    if not dialog.thumbnails_settled():
        raise ShotError("棚の見本が描き終わらない（そろった知らせが来なかった）")


def _item_for(tree: QTreeWidget, entry: TemplateEntry | None) -> QTreeWidgetItem | None:
    """棚の一覧から、そのテンプレートの行を探す"""
    if entry is None:
        return None
    for index in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(index)
        if group is None:
            continue
        for child_index in range(group.childCount()):
            item = group.child(child_index)
            if item.data(0, Qt.ItemDataRole.UserRole) == entry:
                return item
    return None


def shot_ymm4(context: Context) -> QImage:
    """見本のエイリアスを、自分で打った字幕に**着せた**ところ"""
    entry = find_template(context, SAMPLE_ALIAS_NAME)
    with editor(sample_project(*FULL_HD)) as window:
        text = TEXT.create(text="自分で打った字幕です", size=72)
        window.apply_commands(_place_text(window, text, at_frame=0, duration=150), "テキストを追加")
        clip_id = top_text_clip(window)
        located = window.project.timeline.locate_clip(clip_id)
        if located is None:
            raise ShotError("置いたテキストが見つからない")
        # 着せ替えは文字と長さを残す 打った文字がそのまま残っているのが要点なので、
        # あとから文字を入れ直さない
        window.apply_commands(restyle(entry.load(), located[1]), "テンプレートを適用")
        window.select_clip(clip_id)
        window.seek(10)
        settle(window)
        return take_editor_shot(window)


def shot_ymm4_template(context: Context) -> QImage:
    """見本の YMM4 のアイテムテンプレートを置いて、登場の途中で止めたところ"""
    entry = find_template(context, SAMPLE_YMM4_NAME)
    with editor(sample_project(*FULL_HD)) as window:
        commands = place(entry.load(), window.project, at_frame=0, media={})
        if not commands:
            raise ShotError(f"置けるテンプレートではない: {entry.name}")
        window.apply_commands(list(commands), "テンプレートを配置")
        window.select_clip(top_text_clip(window))
        window.seek(YMM4_TEMPLATE_FRAME)
        settle(window)
        return take_editor_shot(window)


#: 互換性レポートの写真に並べる記録 （名前, 回数）
#: 本物のスクリプトを走らせて集めない 配布スクリプトはリポジトリに入れられず、
#: 撮る機械に何が入っているかで写る物が変わる 名前は README が「大きな穴」として
#: 挙げている、実際にまだ無い関数にする（在る関数を並べると写真が嘘になる）
SAMPLE_MISSING = (("obj.getpixeldata", 24), ("obj.putpixeldata", 24))


def shot_preferences(context: Context) -> QImage:
    """〔表示〕→〔設定…〕 既定の値のまま開いたところ"""
    del context
    return _grab_dialog(_full_preferences())


def _full_preferences() -> PreferencesDialog:
    """設定の窓を、巻物にせず全部が見える高さで開く

    窓は画面の 9 割で止めて残りを巻物にする 撮る機械の画面の大きさで写る項目が
    変わらないよう、中身の高さまで伸ばしてから撮る
    """
    dialog = PreferencesDialog(Preferences())
    area = dialog.findChild(QScrollArea)
    if area is not None and area.widget() is not None:
        body = area.widget()
        dialog.resize(dialog.width(), body.sizeHint().height() + 64)
    return dialog


def compatibility_dialog(context: Context) -> CompatibilityDialog:
    """写真に出す互換性レポート 見本のスクリプト 1 本と、見本の記録を持つ

    探索先は相対の名前に差し替える 見本のスクリプトは作業用のフォルダに置くので、
    そのまま出すとその場所が写真に写る
    """
    install_sample_script(context.script_root)
    script_catalog().roots = (Path(PORTABLE_SCRIPTS_DIR),)
    report = CompatibilityReport()
    for name, count in SAMPLE_MISSING:
        for _ in range(count):
            report.note_missing(name)
    return CompatibilityDialog(report)


def shot_compat_report(context: Context) -> QImage:
    """〔互換〕→〔互換性レポート…〕"""
    return _grab_dialog(compatibility_dialog(context))


def _grab_dialog(dialog: QWidget, marks: Sequence[str] = ()) -> QImage:
    """ダイアログを撮る ``marks`` の文言のボタンを枠で囲む"""
    hide_from_screen(dialog)
    dialog.show()
    settle(dialog)
    image = grab(dialog)
    mark(image, [rect_in(dialog, button) for button in _buttons(dialog, marks)])
    dialog.close()
    QApplication.processEvents()
    return image


def _buttons(widget: QWidget, texts: Sequence[str]) -> list[QAbstractButton]:
    found = [b for b in widget.findChildren(QAbstractButton) if b.text() in texts and b.isVisible()]
    if len(found) < len(texts):
        raise ShotError(f"ボタンが見つからない: {', '.join(texts)}")
    return found


# --- Wiki の手順書 ---


def wiki_first_new(context: Context) -> QImage:
    """〔ファイル〕→〔新規〕で出る、解像度とフレームレートを選ぶ窓"""
    del context
    from sashimono.ui.project_settings_dialog import ProjectSettingsDialog

    settings = ProjectSettings(layer_mode=Preferences().new_project_layers)
    return _grab_dialog(ProjectSettingsDialog(settings, new=True), marks=("OK",))


def wiki_first_import(context: Context) -> QImage:
    """何も無い編集画面で〔ファイル〕→〔素材を読み込む…〕"""
    del context
    with editor(sample_project()) as window:
        return with_menu(window, "ファイル", ["素材を読み込む…"])


def wiki_first_placed(context: Context) -> QImage:
    """読み込んだ素材が素材一覧とタイムラインに並んだところ"""
    from sashimono.ui.media_pool import MediaPoolWidget
    from sashimono.ui.timeline import TimelineView

    with editor(sample_project()) as window:
        import_sample(window, context)
        window.seek(0)
        image = take_editor_shot(window)
        pool = window.findChild(MediaPoolWidget)
        timeline = window.findChild(TimelineView)
        mark(image, [rect_in(window, w) for w in (pool, timeline) if w is not None][:1])
        return image


def wiki_first_split(context: Context) -> QImage:
    """再生ヘッドを動かして〔編集〕→〔再生ヘッドで分割〕"""
    with editor(sample_project()) as window:
        import_sample(window, context)
        window.select_clip(first_video_clip(window))
        window.seek(60)
        settle(window)
        return with_menu(window, "編集", ["再生ヘッドで分割"])


def wiki_first_cut(context: Context) -> QImage:
    """分けた頭の側を選んで〔削除して詰める〕したあと"""
    with editor(sample_project()) as window:
        import_sample(window, context)
        window.select_clip(first_video_clip(window))
        window.seek(60)
        window._timeline.split_at_playhead()
        window.select_clip(first_video_clip(window))
        window._timeline.delete_selected(ripple=True)
        window.seek(0)
        settle(window)
        return take_editor_shot(window)


def wiki_first_export(context: Context) -> QImage:
    """〔ファイル〕→〔書き出し…〕の窓"""
    from sashimono.ui.export_dialog import ExportDialog

    with editor(sample_project()) as window:
        import_sample(window, context)
        dialog = ExportDialog(window.project, window)
        # 出力先の既定はホームの下 ホームは作業用のフォルダへ向けてあり、その場所が
        # そのまま写ると読む人の手元とかけ離れた絵になる 本人の機械で出る形
        # （ホームの下の Videos）を、名前を伏せた書き方で見せる
        for box in dialog.findChildren(QLineEdit):
            text = box.text()
            if text.startswith(str(Path.home())):
                box.setText("%USERPROFILE%" + text[len(str(Path.home())) :])
        return _grab_dialog(dialog, marks=("書き出し",))


def wiki_subtitle_menu(context: Context) -> QImage:
    """素材を置いた所で〔字幕〕→〔起こす…〕"""
    with editor(sample_project()) as window:
        import_sample(window, context)
        window.show_subtitles()
        return with_menu(window, "字幕", ["起こす…"])


def wiki_subtitle_install(context: Context) -> QImage:
    """起こしの環境が入っていないときの〔起こす…〕の窓

    手順書が見せるのは初めて使う人の画面 撮る機械に環境が入っていても未導入の状態を
    渡す 機械の状態のまま開くと、導入済みの機械ではボタンが「環境を更新」になり、
    囲む「環境を導入」が見つからずに撮れない
    """
    from sashimono.asr import TranscriptionService, default_backend
    from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog

    with editor(sample_project()) as window:
        import_sample(window, context)
        media = window.project.media[0]
        dialog = TranscribeDialog(
            media,
            TranscriptionService(default_backend()),
            window,
            status=asr_not_installed,
        )
        return _grab_dialog(dialog, marks=("環境を導入",))


def asr_not_installed() -> PackStatus:
    """字幕起こしの環境が 1 つも入っていない状態 初めて使う人の機械と同じ"""
    from sashimono.asr import ASR_PACK

    return PackStatus(
        pack=ASR_PACK,
        packages=tuple(_not_installed(name) for name in ASR_PACK.required),
        extras=tuple(_not_installed(name) for name in ASR_PACK.extra),
    )


def _not_installed(requirement: str) -> PackageStatus:
    """入っていないパッケージ 最低の版は条件の書き方（``>=``）から読む"""
    minimum = (
        requirement.split(">=", 1)[1].split(",", 1)[0].strip() if ">=" in requirement else None
    )
    return PackageStatus(requirement, None, minimum or None)


def wiki_subtitle_clean(context: Context) -> QImage:
    """〔字幕〕→〔整形…〕の窓"""
    from sashimono.ui.subtitle.dialogs import CleanupDialog

    del context
    return _grab_dialog(CleanupDialog(sample_transcript()), marks=("整形する",))


def wiki_subtitle_output(context: Context) -> QImage:
    """〔字幕〕→〔焼き込み〕と〔書き出し…〕"""
    with editor(sample_project()) as window:
        _with_transcript(window, context)
        window.seek(100)
        settle(window)
        return with_menu(window, "字幕", ["焼き込み", "書き出し…"])


def wiki_subtitle_burned(context: Context) -> QImage:
    """焼き込んだあと 字幕がテキストのクリップとして並ぶ"""
    with editor(sample_project()) as window:
        panel = _with_transcript(window, context)
        # 窓で選ぶ所は全部の話し手のまま進める 窓を開くと撮影がそこで止まる
        panel.ask_burn = lambda voices, _note: [voice for voice, _ in voices]
        panel.burn()
        # 2 枚目の字幕の途中で止める 字幕と字幕の間で止めると、焼き込んだ字幕が画面に出ない
        window.seek(130)
        settle(window)
        return take_editor_shot(window)


def wiki_template_menu(context: Context) -> QImage:
    """〔互換〕→〔テンプレート…〕"""
    del context
    with editor(sample_project(*FULL_HD)) as window:
        return with_menu(window, "互換", ["テンプレート…"])


def wiki_template_shelf(context: Context) -> QImage:
    """棚で見本を選び、〔タイムラインへ置く〕を囲んだところ"""
    return _grab_dialog(_shelf(context, SAMPLE_ALIAS_NAME), marks=("タイムラインへ置く",))


def wiki_template_placed(context: Context) -> QImage:
    """〔タイムラインへ置く〕で見本を置いたところ"""
    entry = find_template(context, SAMPLE_ALIAS_NAME)
    with editor(sample_project(*FULL_HD)) as window:
        if not window.place_template_entry(entry, 0):
            raise ShotError("見本のテンプレートを置けない")
        window.select_clip(top_text_clip(window))
        window.seek(10)
        settle(window)
        return take_editor_shot(window)


def wiki_template_restyle(context: Context) -> QImage:
    """自分で打った字幕を選んだ編集画面から棚を開き、〔選択中のクリップに適用〕

    手順の前提（着せる字幕を先に選ぶ）が写るよう、字幕を選んだ編集画面の上に棚を
    重ねる 棚だけを撮ると、何に着せるのかが絵から分からない
    """
    with editor(sample_project(*FULL_HD)) as window:
        text = TEXT.create(text="自分で打った字幕です", size=72, pos_y=-380)
        window.apply_commands(_place_text(window, text, at_frame=0, duration=150), "テキストを追加")
        clip = top_text_clip(window)
        window.select_clip(clip)
        window.seek(10)
        settle(window)
        image = take_editor_shot(window)
        shelf = _grab_dialog(_shelf(context, SAMPLE_ALIAS_NAME), marks=("選択中のクリップに適用",))
        # 棚は右上に重ねる 真ん中に置くと、選んだ字幕のクリップ（タイムラインの左）が
        # 隠れて、何を選んでから開いたのかが写らない
        origin = QPoint(image.width() - shelf.width() - 12, 40)
        _shade(image)
        place = QRect(origin, shelf.size())
        paste(image, place, shelf)
        # 窓の縁を引く 地の色が編集画面と同じなので、引かないと境目が分からない
        painter = QPainter(image)
        painter.setPen(QPen(QColor(150, 150, 160), 1))
        painter.drawRect(place.adjusted(-1, -1, 0, 0))
        painter.end()
        return image


def _shade(image: QImage) -> None:
    """下の編集画面を少し暗くして、上に開いた棚と見分けが付くようにする"""
    image.setDevicePixelRatio(1.0)
    painter = QPainter(image)
    painter.fillRect(QRect(0, 0, image.width(), image.height()), QColor(0, 0, 0, 90))
    painter.end()


def wiki_script_menu(context: Context) -> QImage:
    """〔互換〕→〔スクリプトフォルダを開く〕と〔スクリプトを読み直す〕"""
    del context
    with editor(sample_project()) as window:
        return with_menu(window, "互換", ["スクリプトを読み直す", "スクリプトフォルダを開く"])


def wiki_script_add(context: Context) -> QImage:
    """設定パネルの〔エフェクトを追加…〕から見本のスクリプトを選ぶところ"""
    kind = install_sample_script(context.script_root)
    definition = registry.require(kind)
    with editor(sample_project()) as window:
        text = TEXT.create(text="AviUtl の\nスクリプト", size=96, border_width=4)
        window.apply_commands(_place_text(window, text, at_frame=0, duration=120), "テキストを追加")
        window.select_clip(top_text_clip(window))
        window.seek(20)
        settle(window)
        inspector = window.findChild(InspectorPanel)
        if inspector is None:
            raise ShotError("設定パネルが無い")
        menu = inspector.effect_menu()
        if menu is None:
            raise ShotError("エフェクトのメニューが出ない")
        image = take_editor_shot(window)
        button = next(
            (b for b in inspector.findChildren(QAbstractButton) if b.text() == "エフェクトを追加…"),
            None,
        )
        if button is None:
            raise ShotError("〔エフェクトを追加…〕が無い")
        return _overlay_submenu(window, image, menu, button, definition.category, definition.label)


def _overlay_submenu(
    window: MainWindow,
    image: QImage,
    menu: QMenu,
    button: QWidget,
    category: str,
    label: str,
) -> QImage:
    """ボタンから開くメニューと、その中の ``category`` の下のメニューを重ねる

    メニューはボタンの上へ開く形で置く 下は窓の外で、本物も画面の端で上へ折り返す
    """
    submenu = next(
        (
            action.menu()
            for action in menu.actions()
            if isinstance(action.menu(), QMenu) and action.text() == category
        ),
        None,
    )
    if not isinstance(submenu, QMenu):
        raise ShotError(f"エフェクトのメニューに {category} が無い")
    pictures = []
    for each in (menu, submenu):
        pictures.append(menu_picture(each))
    main, sub = pictures
    anchor = button.mapTo(window, QPoint(0, 0))
    top = max(0, anchor.y() - main.height())
    left = min(anchor.x(), window.width() - main.width() - sub.width())
    paste(image, QRect(QPoint(left, top), main.size()), main)
    category_rect = action_in(menu, category).translated(left, top)
    sub_top = max(0, min(category_rect.top(), window.height() - sub.height()))
    sub_left = left + main.width() - 2
    paste(image, QRect(QPoint(sub_left, sub_top), sub.size()), sub)
    mark(image, [rect_in(window, button)])
    mark(
        image,
        [category_rect, action_in(submenu, label).translated(sub_left, sub_top)],
        grow=(2, -1),
    )
    return image


def wiki_heavy_quality(context: Context) -> QImage:
    """プレビューの下の〔再生品質〕"""
    from sashimono.ui.transport import TransportBar

    with editor(sample_project()) as window:
        import_sample(window, context)
        window.seek(45)
        image = take_editor_shot(window)
        bar = window.findChild(TransportBar)
        boxes = bar.findChildren(QComboBox) if bar is not None else []
        mark(image, [rect_in(window, box) for box in boxes[:1]])
        return image


def wiki_heavy_preferences(context: Context) -> QImage:
    """〔表示〕→〔設定…〕の、重い素材に効く項目（控え・画質・先読み）を囲む"""
    del context
    dialog = _full_preferences()
    hide_from_screen(dialog)
    dialog.show()
    settle(dialog)
    image = grab(dialog)
    wanted = (
        "プレビューに低解像度の控えを使う",
        AUTO_QUALITY_TEXT,
        "手が止まっている間に、先のコマを描いておく",
    )
    buttons = [b for b in dialog.findChildren(QAbstractButton) if b.text() in wanted]
    if len(buttons) < len(wanted):
        dialog.close()
        raise ShotError("設定の窓に重い素材の項目が見つからない")
    mark(image, [rect_in(dialog, button) for button in buttons])
    # 囲んだ項目の組（先読みのメモリの行）までで切る 窓全体は README の設定の写真と
    # 同じ絵になるので、手順書では関わる所だけを見せる
    prefetch = next(b for b in buttons if b.text() == wanted[-1])
    bottom = rect_in(dialog, prefetch).bottom() + PREFERENCES_TAIL
    dialog.close()
    QApplication.processEvents()
    return image.copy(0, 0, image.width(), min(image.height(), bottom))


def wiki_trouble_help(context: Context) -> QImage:
    """〔ヘルプ〕の中身（使い方・不具合・要望を送る・バージョン情報）"""
    del context
    with editor(sample_project()) as window:
        return with_menu(window, "ヘルプ", ["使い方", "不具合・要望を送る"])


def wiki_trouble_report(context: Context) -> QImage:
    """互換性レポートの〔内容をコピー〕を囲んだところ"""
    return _grab_dialog(compatibility_dialog(context), marks=("内容をコピー",))


SHOTS: tuple[Shot, ...] = (
    Shot("screenshot", "README の先頭（編集画面）", shot_editor),
    Shot("subtitle", "字幕パネル", shot_subtitle),
    Shot("ai", "AI アシスタント（見本の会話）", shot_ai),
    Shot("aviutl", "AviUtl スクリプト", shot_aviutl, needs_media=False),
    Shot("templates", "テンプレートの棚（見本のエイリアス）", shot_templates, needs_media=False),
    Shot("ymm4", "テンプレートを着せたところ", shot_ymm4, needs_media=False),
    Shot("ymm4-shelf", "テンプレートの棚（見本の YMM4）", shot_ymm4_shelf, needs_media=False),
    Shot("ymm4-template", "YMM4 のテンプレートの再現", shot_ymm4_template, needs_media=False),
    Shot("preferences", "Wiki の設定のページ", shot_preferences, needs_media=False),
    Shot("compat-report", "Wiki の困ったときのページ", shot_compat_report, needs_media=False),
    # ここから Wiki の手順書 名前の頭は手順書のページ
    Shot("first-new", "手順書 最初の 1 本: 新規", wiki_first_new, needs_media=False, wiki=True),
    Shot(
        "first-import",
        "手順書 最初の 1 本: 素材を読み込む",
        wiki_first_import,
        needs_media=False,
        wiki=True,
    ),
    Shot("first-placed", "手順書 最初の 1 本: 並んだところ", wiki_first_placed, wiki=True),
    Shot("first-split", "手順書 最初の 1 本: 分割", wiki_first_split, wiki=True),
    Shot("first-cut", "手順書 最初の 1 本: 削除して詰める", wiki_first_cut, wiki=True),
    Shot("first-export", "手順書 最初の 1 本: 書き出し", wiki_first_export, wiki=True),
    Shot("subtitle-menu", "手順書 字幕: 起こす", wiki_subtitle_menu, wiki=True),
    Shot("subtitle-install", "手順書 字幕: 環境を導入", wiki_subtitle_install, wiki=True),
    Shot("subtitle-clean", "手順書 字幕: 整形", wiki_subtitle_clean, needs_media=False, wiki=True),
    Shot("subtitle-output", "手順書 字幕: 焼き込みと書き出し", wiki_subtitle_output, wiki=True),
    Shot("subtitle-burned", "手順書 字幕: 焼き込んだあと", wiki_subtitle_burned, wiki=True),
    Shot(
        "template-menu",
        "手順書 テンプレート: 棚を開く",
        wiki_template_menu,
        needs_media=False,
        wiki=True,
    ),
    Shot(
        "template-shelf",
        "手順書 テンプレート: 置く",
        wiki_template_shelf,
        needs_media=False,
        wiki=True,
    ),
    Shot(
        "template-placed",
        "手順書 テンプレート: 置いたところ",
        wiki_template_placed,
        needs_media=False,
        wiki=True,
    ),
    Shot(
        "template-restyle",
        "手順書 テンプレート: 着せる",
        wiki_template_restyle,
        needs_media=False,
        wiki=True,
    ),
    Shot(
        "script-menu",
        "手順書 AviUtl のスクリプト: 置き場と読み直し",
        wiki_script_menu,
        needs_media=False,
        wiki=True,
    ),
    Shot(
        "script-add",
        "手順書 AviUtl のスクリプト: エフェクトとして積む",
        wiki_script_add,
        needs_media=False,
        wiki=True,
    ),
    Shot("heavy-quality", "手順書 重い素材: 再生品質", wiki_heavy_quality, wiki=True),
    Shot(
        "heavy-preferences",
        "手順書 重い素材: 設定",
        wiki_heavy_preferences,
        needs_media=False,
        wiki=True,
    ),
    Shot(
        "trouble-help", "手順書 困ったとき: ヘルプ", wiki_trouble_help, needs_media=False, wiki=True
    ),
    Shot(
        "trouble-report",
        "手順書 困ったとき: 内容をコピー",
        wiki_trouble_report,
        needs_media=False,
        wiki=True,
    ),
)


# --- 入口 ---


def work_folder_base(requested: Path | None, home: Path) -> Path:
    """作業用のフォルダを作る場所 ``%USERPROFILE%`` の下は断る

    ホームの下（既定の一時フォルダもそこ）に作ると、ユーザー名を含む場所が
    素材や設定の置き場として画面のどこかに出たとき、そのまま写真に写る

    渡されなければ、書ける場所を順に探す 先はリポジトリのあるドライブの根（どの
    フォルダの名前も含まない） 標準の利用者ではシステムのドライブの根に書けないので、
    次はリポジトリの中の ``.work/shots``（リポジトリは本人が書ける 中身は git に入らない）
    どちらもホームの下は使わない 使える所が無ければ、``--work`` を求めて止める
    """
    if requested is not None:
        base = requested.resolve()
        if _under(base, home):
            raise ShotError(
                f"作業用のフォルダを %USERPROFILE% の下には作らない: {base}"
                "（--work で別の場所を渡す）"
            )
        return base
    for candidate in (Path(ROOT.anchor), ROOT / ".work" / "shots"):
        base = candidate.resolve()
        if not _under(base, home) and _writable(base):
            return base
    raise ShotError(
        "作業用のフォルダを作れる場所が無い（ドライブの根に書けず、リポジトリの中の "
        ".work/shots は %USERPROFILE% の下にあるか書けない） --work で "
        "%USERPROFILE% の外の書ける場所を渡す"
    )


def _under(path: Path, home: Path) -> bool:
    return path == home or home in path.parents


def _writable(folder: Path) -> bool:
    """そこに作業用のフォルダを作れるか 作って消してみる"""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sashimono-shots-", dir=folder):
            pass
    except OSError:
        return False
    return True


def isolate_user_folders(base: Path) -> None:
    """設定・退避・キャッシュ・ホーム・一時フォルダ・ProgramData の置き場を ``base`` へ向ける

    向けないと、撮る人の画面配置とショートカットが写り、撮り終わったあとに
    その設定が見本の状態で上書きされる ProgramData は AviUtl2 の置き場（エイリアスと
    スクリプト）を棚とスクリプトの一覧が自動で見に行くので、向けないと撮る機械に
    入っている配布物が写真に写る ホームは書き出しの出力先などの既定に出る
    """
    for name, folder in (
        ("APPDATA", "roaming"),
        ("LOCALAPPDATA", "local"),
        ("PROGRAMDATA", "programdata"),
        ("USERPROFILE", "home"),
        ("HOME", "home"),
        ("TEMP", "temp"),
        ("TMP", "temp"),
    ):
        target = base / folder
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    # 既に読んだ一時フォルダの場所を捨てる 捨てないと、作業用のフォルダへ向ける前に
    # 決まった本人の一時フォルダを使い続ける
    tempfile.tempdir = None

    # 先読みは切る 重い絵を出す写真では「1 コマ 0.4 秒掛かるので先読みを止めた」と
    # いう知らせがステータスバーに出て、説明の写真に不具合のように写る
    PreferenceStore().save(Preferences(prefetch=False))


def build_application() -> QApplication:
    """本番と同じ見た目で出す 配色が違うと写真だけ別のソフトに見える

    テーマは既定（暗い）に揃える 撮る人が明るいテーマを選んでいても、設定の置き場は
    作業用のフォルダへ向けてあるので既定になる ボタンの文言（OK・キャンセル）も
    本番と同じく日本語の訳を当てる 当てないと写真だけ英語のボタンになる
    """
    from sashimono.engine.gpu import preferred_surface_format
    from sashimono.resources import ICON_FILE, path_to
    from sashimono.ui.theme import apply_theme
    from sashimono.ui.translation import install_qt_translation

    QSurfaceFormat.setDefaultFormat(preferred_surface_format())
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    application.setApplicationName("Sashimono")
    application.setWindowIcon(QIcon(str(path_to(ICON_FILE))))
    apply_theme(application, Preferences().theme)
    install_qt_translation(application)
    return application


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="README と Wiki の画面写真を撮り直す")
    parser.add_argument("--output", type=Path, default=ROOT / "docs", help="README の写真の行き先")
    parser.add_argument(
        "--wiki",
        type=Path,
        default=None,
        help="Wiki の clone 渡したときだけ手順書の写真を撮り、その images へ書き出す",
    )
    # nargs は "+" 名前を 1 つも書かない ``--only`` は使い方の誤りとして断る
    # "*" だと空の指定が「全部」に化け、絞ったつもりで全部を撮り直すことになる
    parser.add_argument("--only", nargs="+", default=None, help="撮る写真の名前")
    parser.add_argument("--list", action="store_true", help="撮れる写真を並べる")
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="作業用のフォルダを作る場所 既定はリポジトリのあるドライブの根、書けなければ"
        "リポジトリの中の .work/shots %%USERPROFILE%% の下は使わないので、リポジトリが"
        "その下にあってドライブの根にも書けないときは、ここでホームの外を渡す",
    )
    arguments = parser.parse_args(argv)

    if arguments.list:
        for shot in SHOTS:
            where = "wiki" if shot.wiki else "docs"
            print(f"{shot.name:<18} {where:<5} {shot.caption}")
        return 0

    names = set(arguments.only) if arguments.only is not None else None
    if names is not None:
        unknown = names - {shot.name for shot in SHOTS}
        if unknown:
            print(f"知らない写真の名前: {'、'.join(sorted(unknown))}", file=sys.stderr)
            return 2
        if arguments.wiki is None and any(s.wiki for s in SHOTS if s.name in names):
            print("手順書の写真は --wiki に Wiki の clone を渡して撮る", file=sys.stderr)
            return 2
    targets = [
        shot
        for shot in SHOTS
        if (names is None or shot.name in names) and (arguments.wiki is not None or not shot.wiki)
    ]

    # 行き先は作業用のフォルダへ移る前に決める 相対で渡された場所が、移った先からの
    # 相対に化ける
    output: Path = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wiki_images = arguments.wiki.resolve() / "images" if arguments.wiki is not None else None
    if wiki_images is not None:
        wiki_images.mkdir(parents=True, exist_ok=True)

    try:
        base = work_folder_base(arguments.work, Path.home().resolve())
    except ShotError as exc:
        print(exc, file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(
        prefix="sashimono-shots-", dir=base, ignore_cleanup_errors=True
    ) as temporary:
        work = Path(temporary)
        isolate_user_folders(work / "user")
        # 作業用のフォルダから走らせる 棚は相対の置き場をそのまま画面に出すので、
        # 見本の置き場の名前だけが写り、作業用のフォルダの場所は写らない
        previous = Path.cwd()
        os.chdir(work)
        try:
            build_application()
            context = Context(
                # 素材を作るのは、要る写真が選ばれているときだけ 要らない写真まで
                # ffmpeg に付き合わせると、ffmpeg の無い機械では棚の写真も撮れない
                media=_media_or_none(
                    work / "media", needed=any(shot.needs_media for shot in targets)
                ),
                script_root=work / "scripts",
                template_root=write_sample_templates(Path(TEMPLATE_DIR)),
            )
            result = run(targets, context, output, wiki_images)
        finally:
            os.chdir(previous)
    # 手順書の写真を全部撮り直せたときだけ、頭に書いた版を今の版にする 一部だけ撮った
    # （--only・失敗した）のに書き換えると、古い版の写真が新しい版の物として残る
    if result == 0 and names is None and wiki_images is not None:
        for page in stamp_wiki_version(wiki_images.parent):
            print(f"版を書き換えた {page.name}")
    return result


#: 手順書の頭の「この手順書の画面と文言は **Sashimono Edit 0.1.0** で撮りました」の版
WIKI_VERSION = re.compile(r"(\*\*Sashimono Edit )([^*\s]+)(\*\* で撮りました)")


def stamp_wiki_version(wiki: Path, version: str = __version__) -> list[Path]:
    """Wiki の頁の頭に書いた、写真を撮った版を ``version`` に書き換える 書き換えた頁を返す

    手で書き換えると、版を上げて撮り直したのに「0.0.1 で撮りました」のまま残った
    改行は元のまま残す（文字として読んで書くと、Windows では改行の形が変わって頁全体が
    差分になる）
    """
    changed: list[Path] = []
    for page in sorted(wiki.glob("*.md")):
        text = page.read_bytes().decode("utf-8")
        stamped = WIKI_VERSION.sub(lambda found: f"{found[1]}{version}{found[3]}", text)
        if stamped != text:
            page.write_bytes(stamped.encode("utf-8"))
            changed.append(page)
    return changed


def _media_or_none(directory: Path, *, needed: bool) -> Path | None:
    """見本の素材 作れなければ理由を出して ``None``

    ここで止めない 素材の要らない写真は撮れるので、素材が作れないことは
    その写真だけの失敗として :func:`run` が数える
    """
    if not needed:
        return None
    try:
        return make_sample_media(directory)
    except ShotError as exc:
        print(f"見本の素材を作れない: {exc}", file=sys.stderr)
        return None


def run(shots: Sequence[Shot], context: Context, output: Path, wiki: Path | None = None) -> int:
    """並んだ写真を順に撮る 飛ばしたものは今ある写真を残す"""
    failures = 0
    for shot in shots:
        folder = wiki if shot.wiki else output
        if folder is None:
            print(f"飛ばす {shot.name}: 手順書の写真は --wiki を渡したときだけ撮る")
            continue
        try:
            image = shot.take(context)
        except ShotSkippedError as exc:
            print(f"飛ばす {shot.name}: {exc} 今ある写真をそのまま残す")
            continue
        except (ShotError, ToolError, OSError, ValueError, KeyError) as exc:
            print(f"失敗 {shot.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        path = folder / f"{shot.name}.png"
        if not image.save(str(path)):
            print(f"失敗 {shot.name}: 書き出せない {path}", file=sys.stderr)
            failures += 1
            continue
        print(f"撮った {path.name} {image.width()}x{image.height()} {shot.caption}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
