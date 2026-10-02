r"""README と Wiki に載せる画面写真を、アプリ自身に撮らせる

    .venv\Scripts\python.exe tools\shots.py
    .venv\Scripts\python.exe tools\shots.py --list
    .venv\Scripts\python.exe tools\shots.py --only screenshot subtitle

**手で撮らない** 手で撮ると、撮った人の画面配置・重なった別の窓・本人の
ファイル名が写り込み、画面が変わるたびに同じ絵を作り直せない ここでは
見本のプロジェクトをその場で組み立て、``QWidget.grab()`` で窓そのものを
写すので、机の上に何が出ていても結果は変わらない

見本の素材は ffmpeg でその場で作る（色の帯と正弦波） 本人の動画や音声は
絶対に使わない 設定・画面配置の置き場も一時フォルダへ向けるので、撮る人の
並びや前回の作業が写ることもない

配布物（AviUtl2 のエイリアス・YMM4 のアイテムテンプレート）が要る写真は、
その置き場が無い機械では**飛ばす** 無いのに撮ると、空の棚の絵ができて
今ある写真より悪くなる
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtCore import QEvent, QEventLoop, QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QIcon, QImage, QPainter, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QPlainTextEdit,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

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
    TemplateError,
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
from sashimono.ui.compat_dialog import CompatibilityDialog
from sashimono.ui.inspector import InspectorPanel
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.preview import PreviewWidget
from sashimono.ui.subtitle import SubtitlePanel
from sashimono.ui.template_dialog import TemplateDialog
from sashimono.ui.theme import style_sheet
from sashimono.ui.workspace import Preferences, PreferenceStore

ROOT = Path(__file__).resolve().parent.parent

#: 窓の大きさ 今ある写真（約 1500x950）に合わせてある 変えると README の
#: 見た目の縦横比まで変わるので、揃える意味でここ 1 か所に持つ
WINDOW_SIZE = (1500, 950)

#: 見本の素材 小さすぎるとサムネイルが潰れ、長すぎるとタイムラインが余る
SAMPLE_WIDTH, SAMPLE_HEIGHT, SAMPLE_SECONDS = 1280, 720, 8.0
SAMPLE_RATE = FrameRate(30)

#: 配布物を置く写真の画面 配布エイリアスとテンプレートは 1080p 前提で書かれている
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

#: 写したい配布物の名前（一部でよい） README の本文がこの名前を引き合いに
#: 出しているので、同じ物が写るようにしておく 無い機械では使える物の先頭に落ちる
PREFERRED_ALIAS = ("13_金ピカテキスト",)
PREFERRED_YMM4 = ("回転・拡大縮小しながら登場退場/c 登場退場", "回転・拡大縮小しながら登場退場")

#: 配布物が無くて飛ばすときの文言 どこへ置けば撮れるのかまで書く
NO_YMM4 = "YMM4 の .ymmt が置き場に無い（--ymm4-root で指定する 既定は tests/fixtures/ymm4）"
NO_ALIAS = "AviUtl2 のエイリアスの置き場が無い（%PROGRAMDATA%\\aviutl2\\Alias）"

#: YMM4 のテンプレートを止めるフレーム 登場は 60 フレームかけて動くが、
#: Expo のイージングは前半でほとんど動き切る 頭で止めると大きさ 0 で何も
#: 映らず、30 フレーム目では動き終わって見えるので、途中の分かる所で止める
YMM4_TEMPLATE_FRAME = 8


class ShotError(RuntimeError):
    """写真が作れなかった 中身が黒いなど、出してはいけない絵のとき"""


class ShotSkippedError(RuntimeError):
    """この機械では撮れない 今ある写真をそのまま残す"""


@dataclass(frozen=True, slots=True)
class Context:
    """1 回の撮影で共有するもの"""

    #: 見本の素材（ffmpeg で作ったもの） 作れなかったときは ``None``
    media: Path | None
    #: AviUtl2 のエイリアスの置き場 無ければ ``None``
    alias_root: Path | None
    #: YMM4 のアイテムテンプレートの置き場 無ければ ``None``
    ymm4_root: Path | None
    #: 見本の Lua スクリプトを置いた場所
    script_root: Path


@dataclass(frozen=True, slots=True)
class Shot:
    """撮る写真 1 枚"""

    name: str
    #: README のどこに出るか 一覧に出して、撮り直す前に見当が付くようにする
    caption: str
    take: Callable[[Context], QImage]
    #: 見本の素材（ffmpeg で作る映像）が要るか
    #: 要らない写真まで ffmpeg に付き合わせない ffmpeg の無い機械で棚や
    #: スクリプトの写真だけを撮りたいことがある
    needs_media: bool = True


# --- 見本の素材とスクリプト ---


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

    if ratio != 1.0:
        # 画面の拡大率（125% など）で撮ると、撮った機械によって大きさが変わる
        # README に並べる写真の大きさを揃えるため、論理的な大きさへ戻す
        image = image.scaled(
            widget.width(),
            widget.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return image


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
    return QRect(preview.mapTo(window, QPoint(0, 0)), preview.size())


def take_editor_shot(window: MainWindow) -> QImage:
    """編集画面を写して、プレビューの中身まで写っているか確かめる

    GL の面が撮れないと、気付かないまま真っ黒なプレビューの写真が README に
    載る 黙って出さず、ここで止める

    **窓を閉じる前に確かめる** 閉じたあとでは位置も大きさも当てにならず、
    真っ黒を見つけられないまま通ってしまう
    """
    image = grab(window)
    rect = preview_rect(window)
    if rect is not None and brightest(image, rect) < BLACK_LEVEL:
        raise ShotError("プレビューが真っ黒 GL の中身が撮れていない")
    return image


@contextmanager
def editor(project: Project | None = None) -> Iterator[MainWindow]:
    """編集画面を出す 撮り終わったら必ず閉じる"""
    window = MainWindow(project, confirm_unsaved=False)
    # 窓を閉じるたびに画面配置が保存され、次の 1 枚がそれを引き継ぐ 直前に撮った
    # 写真でどのパネルが前に出ていたかによって絵が変わるので、毎回既定へ戻す
    window.reset_layout()
    window.resize(*WINDOW_SIZE)
    window.show()
    settle(window)
    try:
        yield window
    finally:
        window.close()
        QApplication.processEvents()


def sample_project(width: int = SAMPLE_WIDTH, height: int = SAMPLE_HEIGHT) -> Project:
    """見本のプロジェクト 素材はまだ持たない

    配布物を置く写真だけ 1080p にする 配布エイリアスとテンプレートは
    1920x1080 を前提に座標と文字の大きさを書いているので、小さい画面で開くと
    指定どおりの位置に出ない（README の「画面下から 380px」が合わなくなる）
    """
    return Project.create(ProjectSettings(width=width, height=height, frame_rate=SAMPLE_RATE))


def first_video_clip(window: MainWindow) -> ClipId:
    for track in window.project.timeline.video_tracks():
        if track.clips:
            return track.clips[0].id
    raise ShotError("映像クリップが無い")


def top_text_clip(window: MainWindow) -> ClipId:
    """いちばん上に置いたテキストのクリップ"""
    for track in reversed(list(window.project.timeline.video_tracks())):
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


def build_sample_timeline(window: MainWindow, context: Context) -> None:
    """見本の素材を読み込み、テロップとエフェクトを載せる

    README の先頭に出る絵なので、この 1 枚で「素材・波形・テロップ・
    エフェクトの設定」が一度に見えるようにしてある
    """
    import_sample(window, context)
    settle(window)

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
    配布物と違って「無くて当たり前」ではなく、直すべき手落ちとして出す
    素材の要らない写真（棚・スクリプト）はこの失敗に巻き込まれず撮れる
    """
    if context.media is None:
        raise ShotError("見本の素材が無い（ffmpeg が PATH に要る）")
    return context.media


def shot_editor(context: Context) -> QImage:
    with editor(sample_project()) as window:
        build_sample_timeline(window, context)
        return take_editor_shot(window)


def shot_subtitle(context: Context) -> QImage:
    with editor(sample_project()) as window:
        import_sample(window, context)
        settle(window)
        media = window.project.media[0]
        window.apply_commands([SetTranscript(media.id, sample_transcript())], "字幕を更新")
        window.show_subtitles()
        panel = window.findChild(SubtitlePanel)
        if panel is not None:
            panel.select_media(media.id)
        window.seek(100)
        settle(window)
        return take_editor_shot(window)


def shot_ai(context: Context) -> QImage:
    """AI パネル **本物のセッションは走らせない**

    走らせると、返ってくる文面は撮るたびに変わり、課金の要る呼び出しになる
    載せるのは「話しかける前のパネル」で、指示だけを入力欄に書いておく
    （書いただけで送っていない状態は、実際に画面で作れる状態そのもの）
    """
    with editor(sample_project()) as window:
        import_sample(window, context)
        settle(window)
        window.show_chat()
        settle(window, rounds=4)
        box = _chat_input(window)
        if box is None:
            # 実行環境（Claude Agent SDK と Claude Code 本体）が無いと、パネルは
            # 入力欄を閉じて「環境を導入」の案内を出す 撮る機械によって別の絵に
            # なるので、揃っていない機械では今ある写真を残す
            raise ShotSkippedError("AI の実行環境が入っていない（入力欄が使えない）")
        box.setPlainText("冒頭 1 秒を切って 画面下に黄色いテロップを 3 秒入れて")
        settle(window)
        return take_editor_shot(window)


def _chat_input(window: QWidget) -> QPlainTextEdit | None:
    """AI パネルの入力欄 使える状態のものだけを返す"""
    from sashimono.ui.chat import ChatPanel

    panel = window.findChild(ChatPanel)
    if panel is None:
        return None
    return next(
        (
            box
            for box in panel.findChildren(QPlainTextEdit)
            if not box.isReadOnly() and box.isVisible() and box.isEnabled()
        ),
        None,
    )


def shot_aviutl(context: Context) -> QImage:
    """AviUtl のスクリプト 見本のスクリプトを 1 本だけ積む"""
    kind = install_sample_script(context.script_root)
    with editor(sample_project()) as window:
        text = TEXT.create(text="AviUtl の\nスクリプト", size=96, border_width=4)
        window.apply_commands(_place_text(window, text, at_frame=0, duration=120), "テキストを追加")
        clip = top_text_clip(window)
        window.apply_commands([AddEffect(clip, registry.require(kind).create())], "エフェクト")
        window.select_clip(clip)
        window.seek(20)
        settle(window)
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
    bar = area.verticalScrollBar()
    bar.setValue(bar.maximum())


def shot_templates(context: Context) -> QImage:
    """テンプレートの棚（AviUtl2 のエイリアス）"""
    root = context.alias_root
    if root is None:
        raise ShotSkippedError(NO_ALIAS)
    return _shelf((root,), PREFERRED_ALIAS)


def shot_ymm4_shelf(context: Context) -> QImage:
    """テンプレートの棚（YMM4 のアイテムテンプレート）"""
    root = context.ymm4_root
    if root is None:
        raise ShotSkippedError(NO_YMM4)
    return _shelf((root,), PREFERRED_YMM4)


def _shelf(roots: tuple[Path, ...], preferred: Sequence[str]) -> QImage:
    """棚を開いて、中身のあるテンプレートを 1 つ選んだ状態で撮る"""
    # 走査は棚（ダイアログ）に任せて 1 度だけにする 自分でも数えると、同じ
    # `rglob` が 2 回走るうえ、選ぶ相手と画面に並んでいる物がずれる余地が残る
    catalog = TemplateCatalog()
    dialog = TemplateDialog(catalog, roots=roots)
    entries = catalog.all()
    tree = dialog.findChild(QTreeWidget)
    if not entries or tree is None:
        dialog.close()
        raise ShotSkippedError(f"テンプレートが 1 つも見つからない: {', '.join(map(str, roots))}")

    chosen = _item_for(tree, _choose(entries, preferred, _has_text))
    if chosen is not None:
        tree.setCurrentItem(chosen)
        tree.scrollToItem(chosen)
    dialog.show()
    settle(dialog)
    image = grab(dialog)
    dialog.close()
    QApplication.processEvents()
    return image


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


def _choose(
    entries: Sequence[TemplateEntry],
    preferred: Sequence[str],
    usable: Callable[[TemplateEntry], bool],
) -> TemplateEntry | None:
    """写真に写すテンプレートを 1 つ決める

    名前で選べるようにしてあるのは、README の本文がその名前を引き合いに
    出しているため（どれが写るか分からないと、本文と絵が食い違う） その
    配布物が無い機械では、使える物の先頭に落ちる
    """
    for want in preferred:
        found = next((entry for entry in entries if want in entry.name and usable(entry)), None)
        if found is not None:
            return found
    return next((entry for entry in entries if usable(entry)), None)


def _has_text(entry: TemplateEntry) -> bool:
    try:
        loaded = entry.load()
    except (TemplateError, OSError):
        # 読めない物は候補から外すだけ 棚は読めない物も並べる（開くまで分からない）
        return False
    return any(
        inner.clip.source is not None and inner.clip.source.kind == "text"
        for item in loaded
        for inner in item.walk()
    )


def shot_ymm4(context: Context) -> QImage:
    """配布エイリアスを、自分で打った字幕に**着せた**ところ"""
    root = context.alias_root
    if root is None:
        raise ShotSkippedError(NO_ALIAS)
    entry = _find_entry((root,), PREFERRED_ALIAS)
    if entry is None:
        raise ShotSkippedError("文字を持つエイリアスが見つからない")

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
        window.seek(10)
        settle(window)
        return take_editor_shot(window)


def shot_ymm4_template(context: Context) -> QImage:
    """YMM4 のアイテムテンプレートを置いて、登場の途中で止めたところ"""
    root = context.ymm4_root
    if root is None:
        raise ShotSkippedError(NO_YMM4)
    entry = _find_entry((root,), PREFERRED_YMM4)
    if entry is None:
        raise ShotSkippedError("下絵を持つ .ymmt が見つからない")

    with editor(sample_project(*FULL_HD)) as window:
        commands = place(entry.load(), window.project, at_frame=0, media={})
        if not commands:
            raise ShotSkippedError(f"置けるテンプレートではない: {entry.name}")
        window.apply_commands(list(commands), "テンプレートを配置")
        window.seek(YMM4_TEMPLATE_FRAME)
        settle(window)
        return take_editor_shot(window)


def _find_entry(roots: tuple[Path, ...], preferred: Sequence[str]) -> TemplateEntry | None:
    catalog = TemplateCatalog()
    return _choose(catalog.scan(roots), preferred, _has_text)


#: 互換性レポートの写真に並べる記録 （名前, 回数）
#: 本物のスクリプトを走らせて集めない 配布スクリプトはリポジトリに入れられず、
#: 撮る機械に何が入っているかで写る物が変わる 名前は README が「大きな穴」として
#: 挙げている、実際にまだ無い関数にする（在る関数を並べると写真が嘘になる）
SAMPLE_MISSING = (("obj.getpixeldata", 24), ("obj.putpixeldata", 24))


def shot_preferences(context: Context) -> QImage:
    """〔表示〕→〔設定…〕 既定の値のまま開いたところ"""
    del context
    dialog = PreferencesDialog(Preferences())
    return _grab_dialog(dialog)


def compatibility_dialog(context: Context) -> CompatibilityDialog:
    """写真に出す互換性レポート 見本のスクリプト 1 本と、見本の記録を持つ

    探索先は相対の名前に差し替える 見本のスクリプトは一時フォルダに置くので、
    そのまま出すと撮った人のユーザー名を含む場所が写真に写る
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


def _grab_dialog(dialog: QWidget) -> QImage:
    dialog.show()
    settle(dialog)
    image = grab(dialog)
    dialog.close()
    QApplication.processEvents()
    return image


SHOTS: tuple[Shot, ...] = (
    Shot("screenshot", "README の先頭（編集画面）", shot_editor),
    Shot("subtitle", "字幕パネル", shot_subtitle),
    Shot("ai", "AI アシスタント", shot_ai),
    Shot("aviutl", "AviUtl スクリプト", shot_aviutl, needs_media=False),
    Shot(
        "templates", "テンプレートの棚（AviUtl2 のエイリアス）", shot_templates, needs_media=False
    ),
    Shot("ymm4", "テンプレートを着せたところ", shot_ymm4, needs_media=False),
    Shot("ymm4-shelf", "テンプレートの棚（YMM4）", shot_ymm4_shelf, needs_media=False),
    Shot("ymm4-template", "YMM4 のテンプレートの再現", shot_ymm4_template, needs_media=False),
    Shot("preferences", "Wiki の設定のページ", shot_preferences, needs_media=False),
    Shot("compat-report", "Wiki の困ったときのページ", shot_compat_report, needs_media=False),
)


# --- 入口 ---


def isolate_user_folders(base: Path) -> None:
    """設定・退避・キャッシュの置き場を一時フォルダへ向ける

    向けないと、撮る人の画面配置とショートカットが写り、撮り終わったあとに
    その設定が見本の状態で上書きされる 本人のフォルダ名が写真に出るのも避ける
    """
    for name, folder in (("APPDATA", "roaming"), ("LOCALAPPDATA", "local")):
        target = base / folder
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)

    # 先読みは切る 重い絵を出す写真では「1 コマ 0.4 秒掛かるので先読みを止めた」と
    # いう知らせがステータスバーに出て、説明の写真に不具合のように写る
    PreferenceStore().save(Preferences(prefetch=False))


def build_application() -> QApplication:
    """本番と同じ見た目で出す 配色が違うと写真だけ別のソフトに見える

    名前と絵も本番（``app.py``）と同じにする 写真そのものには窓の飾りが
    入らないが、撮っている間に出る窓が別のソフトのように見えるのを避ける
    """
    from sashimono.engine.gpu import preferred_surface_format
    from sashimono.resources import ICON_FILE, path_to

    QSurfaceFormat.setDefaultFormat(preferred_surface_format())
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    application.setApplicationName("Sashimono")
    application.setWindowIcon(QIcon(str(path_to(ICON_FILE))))
    application.setStyleSheet(style_sheet())
    return application


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="README の画面写真を撮り直す")
    parser.add_argument("--output", type=Path, default=ROOT / "docs", help="書き出し先")
    # nargs は "+" 名前を 1 つも書かない ``--only`` は使い方の誤りとして断る
    # "*" だと空の指定が「全部」に化け、絞ったつもりで全部を撮り直すことになる
    parser.add_argument("--only", nargs="+", default=None, help="撮る写真の名前")
    parser.add_argument("--list", action="store_true", help="撮れる写真を並べる")
    parser.add_argument(
        "--ymm4-root",
        type=Path,
        # 相対のまま持つ 棚は選んだテンプレートの場所を画面に出すので、絶対パスだと
        # 撮った機械のフォルダ構成が写真に写る リポジトリの根から走らせる前提
        default=Path("tests") / "fixtures" / "ymm4",
        help="YMM4 のアイテムテンプレート（.ymmt）の置き場 既定は tests/fixtures/ymm4",
    )
    parser.add_argument(
        "--alias-root",
        type=Path,
        default=None,
        help="AviUtl2 のエイリアスの置き場 既定は %%PROGRAMDATA%%\\aviutl2\\Alias",
    )
    arguments = parser.parse_args(argv)

    if arguments.list:
        for shot in SHOTS:
            print(f"{shot.name:<14} {shot.caption}")
        return 0

    names = set(arguments.only) if arguments.only is not None else None
    if names is not None:
        unknown = names - {shot.name for shot in SHOTS}
        if unknown:
            print(f"知らない写真の名前: {'、'.join(sorted(unknown))}", file=sys.stderr)
            return 2
    targets = [shot for shot in SHOTS if names is None or shot.name in names]

    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sashimono-shots-") as temporary:
        work = Path(temporary)
        isolate_user_folders(work / "user")
        build_application()
        context = Context(
            # 素材を作るのは、要る写真が選ばれているときだけ 要らない写真まで
            # ffmpeg に付き合わせると、ffmpeg の無い機械では棚の写真も撮れない
            media=_media_or_none(work / "media", needed=any(shot.needs_media for shot in targets)),
            alias_root=_existing(arguments.alias_root or _program_data_aliases()),
            ymm4_root=_with_templates(arguments.ymm4_root),
            script_root=work / "scripts",
        )
        return run(targets, context, output)


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


def run(shots: Sequence[Shot], context: Context, output: Path) -> int:
    """並んだ写真を順に撮る 飛ばしたものは今ある写真を残す"""
    failures = 0
    for shot in shots:
        try:
            image = shot.take(context)
        except ShotSkippedError as exc:
            print(f"飛ばす {shot.name}: {exc} 今ある写真をそのまま残す")
            continue
        except (ShotError, ToolError, OSError, ValueError, KeyError) as exc:
            print(f"失敗 {shot.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        path = output / f"{shot.name}.png"
        if not image.save(str(path)):
            print(f"失敗 {shot.name}: 書き出せない {path}", file=sys.stderr)
            failures += 1
            continue
        print(f"撮った {path.name} {image.width()}x{image.height()} {shot.caption}")
    return 1 if failures else 0


def _program_data_aliases() -> Path | None:
    program_data = os.environ.get("PROGRAMDATA")
    return Path(program_data) / "aviutl2" / "Alias" if program_data else None


def _existing(path: Path | None) -> Path | None:
    return path if path is not None and path.is_dir() else None


def _with_templates(path: Path | None) -> Path | None:
    """``.ymmt`` が 1 つでも入っているときだけ置き場として認める"""
    root = _existing(path)
    return root if root is not None and any(root.rglob("*.ymmt")) else None


if __name__ == "__main__":
    raise SystemExit(main())
