"""テスト全体で使う素材とプロジェクトの組み立て"""

from __future__ import annotations

import functools
import gc
import shutil
import sys
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QClipboard, QSurfaceFormat
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from sashimono.compat.aviutl import plugin
from sashimono.compat.aviutl.native import NativeModule
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects import registry
from sashimono.engine.gpu import GLContextError, OffscreenGLContext, preferred_surface_format
from sashimono.ui import system_clipboard
from tests.fake_clipboard import FakeClipboard
from tests.media_fixtures import SampleMedia, ffmpeg_available, make_sample

RATE_30 = FrameRate(30)

#: 試験から読ませない置き場 アプリが既定で探す所（本人の AviUtl2 の ``Plugin``）を、
#: アプリと同じ求め方で、差し替える前に求めておく 差し替えた後に求めると
#: 一時フォルダを指してしまい、守るはずの本物の置き場を見失う
PROTECTED_PLUGIN_ROOTS: tuple[Path, ...] = plugin.default_plugin_roots()
#: 本物の合成フォント 実物を使う試験はこれを一時フォルダへ写して読む
REAL_COMFONT = PROTECTED_PLUGIN_ROOTS[0] / "comfont.aux2" if PROTECTED_PLUGIN_ROOTS else None


def real_comfont_missing() -> str | None:
    """実物の合成フォントを使う試験を飛ばす理由 走らせられるなら ``None``

    Windows を先に見る ほかの OS では DLL を読めず、ファイルがあっても
    （共有のフォルダから見えている、など）試験は読み込みで落ちる
    """
    if sys.platform != "win32":
        return "comfont.aux2 は Windows の DLL で、この OS では読めない"
    if REAL_COMFONT is None or not REAL_COMFONT.is_file():
        return "comfont.aux2 が入っていない"
    return None


def touches_real_aviutl2(path: Path) -> bool:
    """守る置き場（本人の AviUtl2 の ``Plugin``）の中を指しているか"""
    target = path.resolve()
    return any(target.is_relative_to(root.resolve()) for root in PROTECTED_PLUGIN_ROOTS)


@pytest.fixture(scope="session", autouse=True)
def qt_application() -> Iterator[QApplication]:
    """テスト全体で 1 つだけ QApplication を用意する

    ウィジェットには QApplication が要るが、GL のテストが先に走ると
    QGuiApplication だけが作られ、あとから QApplication を作れなくなる
    （Qt の制約） ここで最初に上位の QApplication を作っておけば、
    どちらのテストも同じインスタンスを使える
    """
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    QSurfaceFormat.setDefaultFormat(preferred_surface_format())
    yield application


@pytest.fixture(autouse=True)
def widgets_left_behind(qt_application: QApplication) -> Iterator[None]:
    """試験が捨てた部品を、試験の切れ目（Qt が何も配っていない所）で壊す

    閉じただけの窓や ``deleteLater`` した部品は、試験が終わっても残る
    ``processEvents`` は ``deleteLater`` を片付けず、編集画面は参照を捨てても消えない
    （下の後片付けの理由） 全体を通すと、test_theme に来た時点で 4 万を超える部品・
    160 の編集画面・1900 のメニューが残り、Qt が溜めた知らせ（部品を整える頼みなど
    2000 件近く）がその試験の最初の ``processEvents`` で一度に配られていた

    配る途中で Python の受け手（設定パネルの数値欄の eventFilter・時計につないだ関数）が
    動き、そこで閾値を越えてごみ集めが走ると、輪になって捨てられた Python 持ちの部品が
    その場で壊れ、Qt は壊れた部品へ配り続けて access violation で落ちる（PR #236 の CI の
    3.12 で test_theme の窓を出した所で落ちた ごみ集めの間合いで、落ちるかどうかが
    版と走る速さで変わる） 試験の切れ目で片付けて溜めなければ、配る最中に壊れる物が無い
    """
    count = len(QApplication.allWidgets())
    before = set(QApplication.topLevelWidgets())
    yield
    if len(QApplication.allWidgets()) <= count:
        return
    # 試験の中で作った窓を閉じて壊す 編集画面は自分の部品のシグナルに self を掴んだ
    # lambda をつなぐので、参照を捨てても Python のごみ集めでは消えない（つないだ先は
    # C++ の側にあって輪が見えない） 閉じるのは、走っている時計やスレッドを窓の作法で
    # 畳ませてから壊すため
    for widget in QApplication.topLevelWidgets():
        if widget in before or not shiboken6.isValid(widget):
            continue
        widget.close()
        widget.deleteLater()
    # どちらも Qt が何も配っていないここで行う 部品を作らない試験（大半）では走らせない
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    gc.collect()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


@pytest.fixture(autouse=True)
def released_modifier_keys(qt_application: QApplication) -> Iterator[None]:
    """試験が押したまま残した修飾キー（Shift など）を、次の試験へ持ち越さない

    ``QApplication.keyboardModifiers()`` はアプリ全体で 1 つで、最後に配った入力の
    キーを覚えている ``QTest.mouseClick(..., ShiftModifier, ...)`` で終わった試験の後は、
    次の入力が来るまで Shift が押されたままに見える タイムラインの磁石はこれを見て
    吸着を切るので、直後に走った磁石の試験が吸い付かずに落ちた 直列では間に挟まる試験が
    たまたま戻していたが、並列（#239）で並びが変わると表に出た

    Shift に限らず Control・Alt・Meta も同じ所に残るので、どれも戻す
    押していない試験（大半）では何もしない
    """
    yield
    release_modifier_keys()


def release_modifier_keys() -> None:
    """アプリが覚えている修飾キーを、どれも押していない状態へ戻す 戻らなければ落とす

    修飾キーを付けない Shift の離しを 1 つ配る Qt は配った入力の修飾キーをそのまま
    覚え直すので、Shift だけでなく Control・Alt・Meta とその組み合わせも消える
    （PySide6 6.11 のオフスクリーンと Windows の画面ありの両方で確かめた） 離しに
    ``ShiftModifier`` を付けると、それを押したままと覚え直して Shift が残る

    配る相手は、ここで作って表示もフォーカスもしない部品 アプリの部品へ配ると、
    その部品のキーの処理が試験の外で動く 配るのは試験が終わった後の片付けの中
    カーソルを動かす ``mouseMove`` は使わない 画面ありで走らせると本人のカーソルが飛ぶ

    戻らなかったら黙らずに落とす Qt の版が変わって戻し方が効かなくなると、持ち越した
    修飾キーで別の試験がたまにだけ落ち、原因がここだと分からなくなる
    """
    if QApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier:
        return
    widget = QWidget()
    try:
        QTest.keyRelease(widget, Qt.Key.Key_Shift, Qt.KeyboardModifier.NoModifier)
    finally:
        shiboken6.delete(widget)
    left = QApplication.keyboardModifiers()
    if left != Qt.KeyboardModifier.NoModifier:
        pytest.fail(f"試験が残した修飾キーを戻せなかった: {left!r}")


@pytest.fixture(autouse=True)
def isolated_user_folders(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """設定・退避・キャッシュの置き場を、テストごとの一時フォルダへ向ける

    向けないと、テストで窓を閉じるたびに開発者本人の画面配置とショートカットが
    上書きされる 途中で落ちたテストの退避は、次にアプリを起動したときに
    「前回の作業を復元しますか」と出てくる
    """
    base = tmp_path_factory.mktemp("user")
    monkeypatch.setenv("APPDATA", str(base / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(base / "local"))


@pytest.fixture(autouse=True)
def decline_matching_video(monkeypatch: pytest.MonkeyPatch) -> None:
    """「プロジェクトを動画に合わせますか」には合わせないと答える

    尋ねる窓を出すと、答える人がいないまま試験が止まる 合わせないと答えるのは、
    尋ねるようになる前と同じ結果にして、ほかの試験が置いた素材の長さを変えないため
    尋ね方そのものを試すときは、その試験の中で差し替え直す
    """
    from sashimono.ui import media_match

    monkeypatch.setattr(media_match, "ask_to_match", lambda *_args: False)


@pytest.fixture(autouse=True)
def silent_hdr_notice(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """HDR の素材の知らせの窓を出さず、知らせた素材の名前を記録する

    窓を出すと、閉じる人がいないまま試験が止まる 知らせ方そのものを試す試験は、
    返す記録を見るか、その試験の中で差し替え直す
    """
    from sashimono.ui import hdr_notice

    shown: list[list[str]] = []

    def record(_parent: object, media: list[MediaItem]) -> bool:
        shown.append([item.name for item in media])
        return True

    monkeypatch.setattr(hdr_notice, "ask_hdr_notice", record)
    return shown


#: 本物のクリップボードへ書く口 どれも見張りで塞ぐ
_CLIPBOARD_WRITERS = ("setText", "setImage", "setPixmap", "setMimeData", "clear")


@pytest.fixture(autouse=True)
def fake_clipboard(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeClipboard]:
    """アプリのクリップボードを偽物へ差し替え、本物へ書いたら試験を落とす（#154）

    本物へ書くと、試験を走らせるたびに本人がコピーしていた物が消え、同時に走る
    ほかの作業と取り合うと読み戻しが空になって落ちる 試験の中身は、返す偽物から読む

    見張りは 2 重にする ``QClipboard`` の書く口を塞ぐだけでは、ボタンの信号から
    呼ばれた所で投げた例外を Qt が握り、試験が通ってしまう 書こうとした記録を残し、
    後片付けでも落とす Qt の中（C++）から書かれた分は Python の口を通らないので、
    試験の前後でこのプロセスがクリップボードの持ち主になったかも見る
    """
    fake = FakeClipboard()
    monkeypatch.setattr(system_clipboard, "_replacement", fake)

    def forbid(name: str) -> Callable[..., None]:
        def refuse(*_arguments: object, **_options: object) -> None:
            fake.refused.append(name)
            pytest.fail(f"試験が本物のクリップボードへ書こうとした: QClipboard.{name}")

        return refuse

    for name in _CLIPBOARD_WRITERS:
        monkeypatch.setattr(QClipboard, name, forbid(name))
    real = qt_application.clipboard()
    owned_before = real.ownsClipboard()
    yield fake
    if fake.refused:
        pytest.fail(f"試験が本物のクリップボードへ書こうとした: {', '.join(fake.refused)}")
    if not owned_before and real.ownsClipboard():
        pytest.fail("試験の間に本物のクリップボードが書き換わった（Qt の中から書かれた）")


@pytest.fixture(scope="session")
def empty_plugin_folders(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """試験の間だけ使う汎用プラグインの置き場と、プラグインへ渡す設定の置き場"""
    base = tmp_path_factory.mktemp("aviutl2")
    (base / "Plugin").mkdir()
    return base / "Plugin", base


@pytest.fixture(autouse=True)
def isolated_aviutl_plugins(
    empty_plugin_folders: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """本人の AviUtl2 の汎用プラグインを、試験から読ませない（Issue #135）

    汎用プラグインは初期化で自分の処理を走らせる WhisperAutoSub は Python を起動して
    環境を調べ、``%PROGRAMDATA%\\aviutl2\\Plugin`` の下へ設定と一時ファイルを書く
    テレビ字幕の試験を走らせただけで本人の置き場が書き換わり、試験の結果もその機械の
    Python に左右されていた

    既定の置き場は空の一時フォルダへ向ける 実物を使う試験は :func:`real_comfont` で
    使う物だけを写して読む 本物の置き場を直に読もうとしたら、その場で試験を落とす
    （描く側は読み込みの失敗を握って先へ進むので、例外ではなく ``pytest.fail`` にする
    ``Exception`` ではないので握られない）
    """
    plugins, app_data = empty_plugin_folders
    monkeypatch.setattr(plugin, "default_plugin_roots", lambda: (plugins,))
    # プラグインへ渡す設定の置き場も一時フォルダへ 合成フォントは ``profiles.json`` を
    # ここから読む 本人の置き場を渡すと、本人の設定で試験の結果が変わる
    monkeypatch.setattr(plugin, "_app_data", app_data)
    original = plugin._load

    def guarded(path: Path) -> dict[str, NativeModule]:
        if touches_real_aviutl2(path):
            pytest.fail(f"試験が本人の AviUtl2 の汎用プラグインを読もうとした: {path}")
        return original(path)

    monkeypatch.setattr(plugin, "_load", guarded)
    plugin.forget()
    yield
    plugin.forget()


@pytest.fixture(scope="session")
def real_comfont(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """本物の ``comfont.aux2`` だけを写した置き場 無ければ飛ばす

    ``Plugin`` フォルダを丸ごと渡すと、合成フォントを確かめるだけのはずが
    ほかの汎用プラグインまで全部初期化する 写すのは 1 度だけ 同じ実体を
    何度も読み込むと、試験のたびに DLL が 1 つずつ増える
    """
    reason = real_comfont_missing()
    if reason is not None or REAL_COMFONT is None:
        pytest.skip(reason or "comfont.aux2 が入っていない")
    folder = tmp_path_factory.mktemp("comfont")
    shutil.copy2(REAL_COMFONT, folder / REAL_COMFONT.name)
    return folder


@pytest.fixture(autouse=True, scope="module")
def forget_scripts() -> Iterator[None]:
    """テストが登録した AviUtl スクリプトを、モジュールごとに片付ける

    スクリプトの定義はエフェクトの登録簿というアプリ全体の状態に入る
    残したままにすると、別のテストが「シェーダの無いエフェクト」を見つけて
    落ちる
    """
    yield
    for definition in registry.all():
        if definition.kind.startswith("aviutl:"):
            registry.unregister(definition.kind)


@pytest.fixture
def video_media() -> MediaItem:
    """10 秒の 1080p30 素材 映像 1 本と音声 1 本を持つ"""
    return MediaItem(
        path=Path("C:/素材/本編.mp4"),
        duration=Fraction(10),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1920,
                height=1080,
                frame_rate=RATE_30,
                time_base=Fraction(1, 15360),
                codec="h264",
                pixel_format="yuv420p",
            ),
        ),
        audio_streams=(
            AudioStreamInfo(
                index=1,
                sample_rate=48000,
                channels=2,
                time_base=Fraction(1, 48000),
                codec="aac",
            ),
        ),
    )


@pytest.fixture
def audio_media() -> MediaItem:
    """30 秒の BGM"""
    return MediaItem(
        path=Path("C:/素材/bgm.wav"),
        duration=Fraction(30),
        audio_streams=(
            AudioStreamInfo(
                index=0,
                sample_rate=48000,
                channels=2,
                time_base=Fraction(1, 48000),
                codec="pcm_s16le",
            ),
        ),
    )


@pytest.fixture
def transcript() -> Transcript:
    """3 つの発話区間を持つ起こし結果 時刻はすべてソース秒"""
    return Transcript(
        segments=(
            TranscriptSegment(start=Fraction(1), end=Fraction(3), text="今日は"),
            TranscriptSegment(start=Fraction(4), end=Fraction(6), text="編集ソフトを"),
            TranscriptSegment(start=Fraction(7), end=Fraction(9), text="作ります"),
        ),
        language="ja",
        model="large-v3",
    )


@pytest.fixture
def video_track() -> Track:
    return Track(kind=TrackKind.VIDEO, name="V1")


@pytest.fixture
def audio_track() -> Track:
    return Track(kind=TrackKind.AUDIO, name="A1")


@pytest.fixture
def project(video_media: MediaItem, video_track: Track) -> Project:
    """素材 1 つと空の映像トラック 1 本を持つプロジェクト"""
    base = Project.create(ProjectSettings(frame_rate=RATE_30), media=(video_media,))
    from dataclasses import replace

    return base.with_timeline(replace(base.timeline, tracks=(video_track,)))


def make_clip(start: int, duration: int, media: MediaItem, source_in: int = 0) -> Clip:
    """テスト用のクリップを手短に作る"""
    return Clip(
        timeline_start=start,
        duration=duration,
        media_id=media.id,
        source_in=Fraction(source_in),
    )


@functools.cache
def gpu_available() -> bool:
    """OpenGL 4.3 が本当に使えるか 1 セッションで 1 度だけ確かめる

    GPU の無い環境（CI など）でも Qt はコンテキストを「作れて」しまう
    :class:`OffscreenGLContext` は作った直後に関数が呼べるかまで確かめて
    :class:`GLContextError` を出すので、それを見て判断する

    書き出しのように**内部で**コンテキストを作るテストは、自前の ``gl``
    フィクスチャを持たない そういうテストはこれで飛ばす
    """
    try:
        context = OffscreenGLContext()
    except GLContextError:
        return False
    context.release()
    return True


@pytest.fixture
def gpu() -> None:
    """GPU が要るテストに付ける 無い環境では失敗ではなく飛ばす

    ``pytestmark = pytest.mark.usefixtures("gpu")`` でモジュールごと付けられる
    """
    if not gpu_available():
        pytest.skip("OpenGL 4.3 が使えない（GPU ドライバが無い環境）")


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """生成した素材を置く場所 セッション内で使い回す"""
    if not ffmpeg_available():
        pytest.skip("ffmpeg が PATH に無いので実素材のテストを飛ばす")
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def sample_av(media_dir: Path) -> SampleMedia:
    """映像 + 音声、320x240 / 30fps / 2 秒"""
    return make_sample(media_dir, "av.mp4")


@pytest.fixture(scope="session")
def sample_long(media_dir: Path) -> SampleMedia:
    """4 秒・GOP 12 の映像のみ素材 シークが GOP をまたぐ様子を見るため"""
    return make_sample(
        media_dir,
        "long.mp4",
        duration=4.0,
        audio=False,
        keyframe_interval=12,
    )


@pytest.fixture(scope="session")
def sample_ntsc(media_dir: Path) -> SampleMedia:
    """29.97fps の素材 分数フレームレートの扱いを確かめるため"""
    return make_sample(media_dir, "ntsc.mp4", fps="30000/1001", duration=2.0)
