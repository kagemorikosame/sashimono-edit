"""画面の操作が、操作の台帳（:mod:`sashimono.ai.ledger`）に 1 つ残らず載っているかを見張る（#348）

画面にある操作を推測で並べると、足したばかりの項目や、場面でしか出ない項目を必ず落とす
ここでは本物の :class:`MainWindow` を組み、メニューバー・右クリック・パネルのボタンを
実際に取り出して、台帳の行と突き合わせる

- 画面にあって台帳に無い操作があれば落ちる 新しいメニューや右クリックの項目を足した
  PR は、台帳に行を足さない限り通らない（画面と AI の道具が黙ってずれるのを止める）
- 台帳にあって画面に無い行があっても落ちる 消した・名前を変えた項目の行が残ると、
  AI にできることの数え方が実物とずれる

右クリックとボタンは、場面ごとに出る物が違う（クリップの上・空いた所・目盛り・混合の方式）
出る場面を下の :func:`scan_screen` で 1 つずつ作って取り出す 新しい場面でしか出ない項目を
足したときは、ここにその場面を足す 場面を作り忘れたボタンは、ボタンを作る所をソースから
全部拾って比べる見張り（:func:`unreached_sites`）が見つける
"""

from __future__ import annotations

import ast
import contextlib
import fnmatch
import importlib
import importlib.util
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QToolButton,
    QWidget,
)

from sashimono.ai import ledger
from sashimono.core.io.aliases import AliasStore
from sashimono.core.io.presets import PresetStore
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Keyframe,
    LayerMode,
    Marker,
    MediaItem,
    Project,
    ProjectSettings,
    Scene,
    Stroke,
    Timeline,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.effects import registry
from sashimono.effects.sources import GROUP, TEXT, TRANSITION, source_registry
from sashimono.ui.inspector.panel import InspectorPanel, KeyframeControls
from sashimono.ui.inspector.widgets import ParameterEditor
from sashimono.ui.library_save import SaveChoice
from sashimono.ui.main_window import MainWindow
from sashimono.ui.theme import Metrics
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.add_menu import AddSources

#: 名前を付けて置くトラック 項目の名前に入る（「甲 をミュート」）ので、台帳では
#: :data:`TRACK` に置き換えて数える ほかの項目の名前に紛れない字を選んである
#: ``FX1`` はフィルタを置くトラックの見分け方（名前が FX と番号）のための名前
_TRACK_NAMES = ("甲", "乙", "丙", "丁", "戊", "己", "FX1")
#: 何本か選んでいるときに項目の名前の後ろへ付く本数（「削除（3 本）」）
_COUNT = re.compile(r"（\d+ 本）$")
#: 台帳の書き方で、トラックの名前を指す所
TRACK = "〔トラック〕"


@dataclass
class Scan:
    """画面から取り出した操作"""

    #: 操作の在りか（台帳の ``key`` と同じ書き方）
    keys: set[str] = field(default_factory=set)
    #: サブメニューの在りか 中身がデータの一覧（エフェクト・エイリアスなど）の行が、
    #: 一覧が空の場面でも在ると分かるようにする
    submenus: set[str] = field(default_factory=set)
    #: 数えたボタンを作った所（モジュールと行） :func:`button_sites` と比べ、どの場面でも
    #: 出なかったボタンを見つける
    sites: set[tuple[str, int]] = field(default_factory=set)


# --- ボタンを作る所 ---

#: ボタンを数えるパネルのモジュール ここのボタンは、作る所がどれか 1 つの場面で出ないと落ちる
PANEL_MODULES = (
    "sashimono.ui.inspector.panel",
    "sashimono.ui.media_pool",
    "sashimono.ui.subtitle.panel",
    "sashimono.ui.scene_bar",
    "sashimono.ui.transport",
)
_BUTTON_TYPES: tuple[type[QAbstractButton], ...] = (
    QPushButton,
    QToolButton,
    QCheckBox,
    QRadioButton,
)
#: ボタンに作った所を書いておく印の名前
_SITE = "ledger_site"


def _calls(module: str) -> list[ast.Call]:
    """``module`` の中で、ボタンの型を呼んで作る所"""
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None, module
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    names = {kind.__name__ for kind in _BUTTON_TYPES}
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
    ]


def button_sites(module: str) -> dict[int, int]:
    """``module`` の中でボタンを作る所（呼び出しの頭の行 → 終わりの行）

    場面で出る出ないが変わるボタン（AviUtl2 の組み方のときだけ出る〔組み方を「標準」にする〕
    など）は、場面を作り忘れると数えられず、台帳から黙って漏れる 作る所をソースから全部
    拾い、どれか 1 つの場面で出たかを見れば、作り忘れた場面が分かる
    """
    return {node.lineno: node.end_lineno or node.lineno for node in _calls(module)}


def _tracked(module: str, base: type[QAbstractButton]) -> Callable[..., QAbstractButton]:
    """作った所の行を印に書いてからボタンを返す ``module`` の中の型の名前をこれへ差し替える"""

    def make(*arguments: Any, **options: Any) -> QAbstractButton:
        button = base(*arguments, **options)
        button.setProperty(_SITE, f"{module}:{sys._getframe(1).f_lineno}")
        return button

    return make


@contextlib.contextmanager
def tracked_buttons() -> Iterator[None]:
    """パネルのモジュールが作るボタンに、作った所の印を付ける

    差し替えるのは、そのモジュールで実際に呼んで作っている型の名前だけ 設定パネルは
    ``findChild(QCheckBox)`` のように型を探す側でも名前を使うので、作っていない型まで
    差し替えると探し方が変わる
    """
    with contextlib.ExitStack() as stack:
        for module in PANEL_MODULES:
            loaded = importlib.import_module(module)
            used = {node.func.id for node in _calls(module) if isinstance(node.func, ast.Name)}
            for kind in _BUTTON_TYPES:
                if kind.__name__ in used:
                    stack.enter_context(
                        mock.patch.object(loaded, kind.__name__, _tracked(module, kind))
                    )
        yield


def unreached_sites(scan: Scan) -> list[str]:
    """ボタンを作る所のうち、どの場面でも出なかった所"""
    left: list[str] = []
    for module in PANEL_MODULES:
        for start, end in sorted(button_sites(module).items()):
            if not any(m == module and start <= line <= end for m, line in scan.sites):
                left.append(f"{module}:{start}")
    return left


# --- 窓を開かせない ---


@contextlib.contextmanager
def refuse_windows() -> Iterator[list[str]]:
    """取り出しの途中で確認・警告・ファイルの窓が開いたら、その場で止めて落とす

    開いた窓は誰も閉じないので、試験が止まったまま返ってこない Qt の信号の先で
    投げた例外は Qt が飲み込むので、開いた窓を覚えておき、抜けるときにも落とす
    """
    opened: list[str] = []

    def refuse(name: str) -> Callable[..., object]:
        def record(*_arguments: object, **_options: object) -> object:
            opened.append(name)
            pytest.fail(f"操作を取り出す途中で窓が開いた: {name}")

        return record

    targets: list[tuple[type, str]] = [(QDialog, "exec"), (QMessageBox, "exec")]
    targets += [(QMessageBox, n) for n in ("question", "warning", "information", "critical")]
    targets += [(QInputDialog, n) for n in ("getText", "getItem", "getInt", "getDouble")]
    targets += [
        (QFileDialog, n)
        for n in ("getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory")
    ]
    with contextlib.ExitStack() as stack:
        for owner, name in targets:
            stack.enter_context(mock.patch.object(owner, name, refuse(f"{owner.__name__}.{name}")))
        yield opened
    if opened:
        pytest.fail(f"操作を取り出す途中で窓が開いた: {opened}")


@contextlib.contextmanager
def caught_menus(module: str) -> Iterator[list[QMenu]]:
    """``module`` が作るメニューを、開かずに覚えるだけの物へ差し替える

    字幕パネルや設定パネルの右クリックは、作る所と開く所が 1 つの関数にある
    開かせると押す人を待って止まるので、何も選ばなかったことにして中身だけ取る
    ``QMenu.exec`` を型ごと差し替えても、Qt の型の関数は Python の差し替えを見ずに
    本物を呼ぶ（試すと止まった） 作る所の名前を、``exec`` を書き換えた子の型へ向ける
    """
    menus: list[QMenu] = []

    def caught(parent: QWidget | None = None) -> QMenu:
        # 作る所はどれも ``QMenu(self)`` 見出しを渡す作り方を足したら、ここにも足す
        menu = QMenu(parent)
        menus.append(menu)
        # 何も選ばなかったことにする 選んだことにすると、その操作が動いて中身が変わる
        # 子の型で書き直すと、型の情報の ``exec`` が「必ず項目を返す」になっていて合わない
        setattr(menu, "exec", lambda *_arguments, **_options: None)  # noqa: B010
        return menu

    with mock.patch(f"{module}.QMenu", caught):
        yield menus


# --- 取り出し方 ---


#: 押せない理由を名前の後ろに足す項目（「この見た目を保存…（保存できる設定がありません）」）
_REASON = re.compile(r"（[^（）]*）$")


def _label(action: QAction) -> str:
    """項目の名前から、場面で変わる所（トラックの名前・本数・押せない理由）を外す

    押せない理由を外すのは、押した先を項目のデータで持つ物（設定パネルのメニュー）だけ
    名前そのものに括弧のある項目（「貼り付け（再生ヘッドの位置）」）は、押せないときも
    名前を変えないので、外すと別の項目になる
    """
    text = _COUNT.sub("", action.text())
    if not action.isEnabled() and action.data() is not None:
        text = _REASON.sub("", text)
    for name in _TRACK_NAMES:
        text = text.replace(name, TRACK)
    return text


def _receivers(action: QAction) -> int:
    return int(action.receivers("2triggered(bool)")) + int(action.receivers("2triggered()"))


def _is_note(action: QAction, baseline: int) -> bool:
    """押せない説明の行か（「（まだありません）」「時間で動く物は掛けられません」など）

    押せず、押したときの行き先（つないだ受け手・項目のデータ）も無い物だけを外す
    今押せないだけの操作（貼る物が無いときの貼り付け）は受け手があるので残る
    ``QMenu`` は足した項目に自分でも受け手をつなぐので、何もつながない項目の数
    （``baseline``）と比べる 0 と比べると、説明の行まで操作に数える
    """
    if action.isEnabled():
        return False
    return _receivers(action) <= baseline and action.data() is None


def _bare_receivers() -> int:
    """メニューに足しただけで何もつないでいない項目の受け手の数"""
    menu = QMenu()
    try:
        return _receivers(menu.addAction("試し"))
    finally:
        shiboken6.delete(menu)


def walk_menu(menu: QMenu, place: str, scan: Scan) -> None:
    """メニューの項目を ``place/名前/…`` の形で集める 開いたときに中身を作る物も作らせる"""
    # 一番上のメニューも在りかとして覚える 中身が一覧だけのメニュー（保存したプリセット）は、
    # 一覧が空の場面では項目が 1 つも無く、一覧の行が画面から消えたように見える
    scan.submenus.add(place)
    menu.aboutToShow.emit()
    baseline = _bare_receivers()
    for action in menu.actions():
        if action.isSeparator() or _is_note(action, baseline):
            continue
        path = f"{place}/{_label(action)}"
        submenu = action.menu()
        if isinstance(submenu, QMenu):
            scan.submenus.add(path)
            walk_menu(submenu, path, scan)
        else:
            scan.keys.add(path)


def _inside(widget: QWidget, top: QWidget, kind: type[QWidget]) -> bool:
    current: QWidget | None = widget
    while current is not None and current is not top:
        if isinstance(current, kind):
            return True
        current = current.parentWidget()
    return False


def button_key(button: QAbstractButton) -> str:
    """ボタンの台帳の名前 字と説明の頭をつなぐ

    字だけでは同じ「有効」「▲」が別の操作に付いていて見分けられず、説明だけでは
    説明の無いボタン（〔読み込み…〕）が拾えない 説明の「: 入」より後ろは今の状態なので外す
    ◆ は再生位置にキーがあるかで字と説明が変わる 同じボタンなので 1 つにまとめる
    """
    text = button.text().strip()
    if text in ("◇", "◆"):
        return "◆｜再生位置のキーを打つ・消す"
    tip = button.toolTip().split("\n")[0].split(": ")[0].strip()
    if text and tip:
        return f"{text}｜{tip}"
    return text or tip


def walk_buttons(top: QWidget, place: str, scan: Scan) -> None:
    """``top`` の中の、見えているボタンを集める 値の欄の中のボタンは「欄の値」の 1 行にする"""
    # 作り直して捨てた部品は ``deleteLater`` の後も子に残る 片付けてから数える
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    for button in top.findChildren(QAbstractButton):
        if not shiboken6.isValid(button) or not button.isVisibleTo(top):
            continue
        # 値の欄（数・色・チェックなど）の中のボタンは 1 つずつ数えず「欄の値」の 1 行にする
        # 表の左上の〔すべて選ぶ〕のような Qt の部品の一部は、アプリの操作ではないので外す
        if _inside(button, top, ParameterEditor) or _inside(button, top, QAbstractItemView):
            continue
        key = button_key(button)
        assert key, f"{place} に名前も説明も無いボタンがある 台帳に書けないので名前か説明を付ける"
        scan.keys.add(f"{place}/{key}")
        site = button.property(_SITE)
        if isinstance(site, str):
            module, line = site.rsplit(":", 1)
            scan.sites.add((module, int(line)))
    owned = _clip_fields(top)
    for name in owned:
        scan.keys.add(f"{place}/クリップ自身の欄/{name}")
    others = [
        e
        for e in top.findChildren(ParameterEditor)
        if e.isVisibleTo(top) and not any(_within(e, w) for w in owned.values())
    ]
    if others:
        scan.keys.add(f"{place}/欄の値")


def _within(widget: QWidget, owner: QWidget) -> bool:
    """``widget`` が ``owner`` そのものか、その中の部品か"""
    current: QWidget | None = widget
    while current is not None:
        if current is owner:
            return True
        current = current.parentWidget()
    return False


def _clip_fields(top: QWidget) -> dict[str, QWidget]:
    """設定パネルの、クリップ自身を変える欄（合成モード・再生速度・クリッピングなど）

    これらは ``SetClipProperty`` で変わり、エフェクトや中身の値（``SetParam``）とは別の道具に
    当たる 1 行の「欄の値」にまとめると、どの道具で変えるのかを台帳から取り違える
    見分けは設定パネルの持ち方による 「クリップ自身」（``("clip", 名前)``）の鍵を持ち、
    パラメータの欄の一覧（``_editors``）に無い物 パラメータの欄もクリップに付く値は同じ
    形の鍵を持つので、鍵だけでは分けられない
    """
    if not isinstance(top, InspectorPanel):
        return {}
    params = {id(widget) for widget in top._editors.values()}
    return {
        name: widget
        for (owner, name), widget in top._focus_owners.items()
        if owner == "clip"
        and id(widget) not in params
        and shiboken6.isValid(widget)
        and widget.isVisibleTo(top)
    }


# --- 場面を作る ---


def _sources(folder: Path) -> AddSources:
    """〔追加〕の出どころを、置き場を読まない物にする 本人の AviUtl やテンプレートを読ませない"""

    def take(*_args: object, suggestion: str, category: str, **_kwargs: object) -> SaveChoice:
        return SaveChoice(name=suggestion, category=category, replacing=None)

    return AddSources(
        custom_objects=lambda: (),
        templates=lambda: (),
        aliases=AliasStore(folder / "aliases"),
        ask_save=take,
        presets=PresetStore(folder / "presets"),
    )


def _point(view: TimelineView, track: str, frame: int) -> QPoint:
    band = next(b for b in view._layout.bands(view.project.timeline) if b.track.name == track)
    return QPoint(int(view._layout.frame_to_x(frame)) + 1, band.top + band.height // 2)


@dataclass(frozen=True)
class _Stage:
    """1 つの窓で取り出す場面"""

    project: Project
    #: 右クリックする所（トラックの名前とフレーム） 目盛りは ``None``
    clicks: tuple[tuple[str | None, int], ...]
    #: 設定パネルに出して数えるクリップ
    inspected: tuple[Clip, ...] = ()
    #: 素材一覧と字幕パネルに出す素材
    media: MediaItem | None = None


def _separated(media: MediaItem) -> _Stage:
    """映像と音声を分ける方式 クリップ・空いた所・フィルタのトラック・シーン・範囲を作る"""
    size = AnimatedValue(64, keyframes=(Keyframe(frame=0, value=64), Keyframe(frame=30, value=96)))
    text_source = TEXT.create(text="見出し", size=size).with_strokes([Stroke()])
    blur = registry.require("blur")
    text = Clip(timeline_start=0, duration=60, source=text_source, effects=(blur.create(),))
    transition = Clip(timeline_start=200, duration=30, source=TRANSITION.create())
    group = Clip(timeline_start=400, duration=30, source=GROUP.create())
    # AviUtl2 の組み方のテキスト 灰色の欄に〔組み方を「標準」にする〕が出る（#301）
    aviutl = Clip(timeline_start=500, duration=30, source=TEXT.create(text="字", layout="aviutl"))
    scene = Scene("場面", Timeline(media.video_streams[0].frame_rate))
    nested = Clip(timeline_start=100, duration=60, scene_id=scene.id)
    held = Clip(timeline_start=0, duration=90, media_id=media.id, hold_at=Fraction(1))
    sound = Clip(timeline_start=0, duration=90, media_id=media.id)
    base = Project.create(media=(media,))
    tracks = (
        Track(TrackKind.VIDEO, "甲", (text, nested, transition, group, aviutl)),
        Track(TrackKind.VIDEO, "乙", (held,)),
        Track(TrackKind.VIDEO, "FX1"),
        Track(TrackKind.AUDIO, "丙", (sound,)),
        Track(TrackKind.AUDIO, "丁"),
    )
    # 目印の上の右クリックでだけ、名前の変更と削除が出る（#330）
    timeline = replace(
        base.timeline, tracks=tracks, work_area=(600, 700), markers=(Marker(300, "印"),)
    )
    project = replace(base, scenes=(scene,), timeline=timeline)
    return _Stage(
        project,
        clicks=(
            ("甲", 10),  # テキスト
            ("甲", 120),  # シーンを置いたクリップ
            ("甲", 210),  # 場面切り替え
            ("乙", 10),  # 動画
            ("丙", 10),  # 音声
            ("丁", 10),  # 空いた音声トラック
            ("FX1", 10),  # フィルタを置くトラック
            ("丁", 650),  # 書き出し範囲の上
            (None, 650),  # 目盛り
            (None, 300),  # 目盛りの目印
        ),
        inspected=(text, transition, group, aviutl, held, sound),
        media=media,
    )


def _mixed(media: MediaItem) -> _Stage:
    """混合の方式 レイヤーを足す項目と、不透明度・音量の線の切り替えが出る"""
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED), media=(media,))
    # 音の番号を持たせる 持たないと絵だけのクリップになり、音量の線の切り替えが出ない
    movie = Clip(
        timeline_start=0,
        duration=90,
        media_id=media.id,
        audio_stream=media.audio_streams[0].index,
    )
    tracks = (Track(TrackKind.MIXED, "戊", (movie,)), Track(TrackKind.MIXED, "己"))
    project = base.with_timeline(replace(base.timeline, tracks=tracks))
    return _Stage(project, clicks=(("戊", 10), ("己", 10)), inspected=(movie,))


def _scan_stage(window: MainWindow, stage: _Stage, folder: Path, scan: Scan) -> None:
    view = window._timeline
    view.add_sources = _sources(folder)
    window._inspector.set_preset_store(PresetStore(folder / "presets"))
    window._inspector.set_alias_store(AliasStore(folder / "aliases"))
    # 値の線（不透明度・音量）の切り替えは線を出しているときだけ右クリックに並ぶ
    view.set_value_lines(True)
    view.resize(1600, 600)

    for name in window._actions:
        scan.keys.add(f"メニュー/{name}")

    for track, frame in stage.clicks:
        if track is None:
            position = QPoint(int(view._layout.frame_to_x(frame)) + 1, Metrics.RULER_HEIGHT // 2)
        else:
            position = _point(view, track, frame)
        walk_menu(view.build_context_menu(position), "タイムラインの右クリック", scan)
    walk_menu(view.build_track_add_menu(), "タイムラインの〔＋ トラック追加〕", scan)
    walk_menu(window._preview_menu, "プレビューの右クリック", scan)

    walk_buttons(window._scene_bar, "シーンの帯", scan)
    walk_buttons(window._transport, "再生の操作", scan)
    walk_buttons(window._media_pool, "素材一覧", scan)
    walk_buttons(window._subtitles, "字幕パネル", scan)

    if stage.media is not None:
        walk_menu(window._media_pool.build_menu(stage.media.id), "素材一覧の右クリック", scan)
        subtitles = window._subtitles
        subtitles.select_media(stage.media.id)
        subtitles._table.selectRow(0)
        with caught_menus("sashimono.ui.subtitle.panel") as menus:
            subtitles._show_menu(QPoint(1, 1))
        assert menus, "字幕パネルの右クリックが出なかった 行を選べていない"
        for menu in menus:
            walk_menu(menu, "字幕パネルの右クリック", scan)

    inspector = window._inspector
    for clip in stage.inspected:
        view.select(clip.id)
        walk_buttons(inspector, "設定パネル", scan)
        effects = inspector.effect_menu()
        if effects is not None:
            walk_menu(effects, "設定パネルの〔エフェクトを追加…〕", scan)
        presets = inspector.preset_menu()
        if presets is not None:
            walk_menu(presets, "設定パネルの〔プリセット…〕", scan)
        source = clip.source
        for stroke in source.strokes if source is not None else ():
            walk_menu(
                inspector.stroke_effect_menu(clip.id, stroke.id),
                "設定パネルの縁取りの〔効果〕",
                scan,
            )
        for controls in inspector.findChildren(KeyframeControls):
            if not shiboken6.isValid(controls) or not controls.toggle.isEnabled():
                continue
            if controls.path.name != "size":
                continue
            with caught_menus("sashimono.ui.inspector.panel") as menus:
                controls.menu_requested.emit()
            for menu in menus:
                walk_menu(menu, "設定パネルの ◆ の右クリック", scan)
            break


def scan_screen(folder: Path, media: MediaItem) -> Scan:
    """場面を 1 つずつ作り、画面の操作を全部取り出す"""
    scan = Scan()
    with tracked_buttons():
        for stage in (_separated(media), _mixed(media)):
            window = MainWindow(stage.project, confirm_unsaved=False)
            try:
                _scan_stage(window, stage, folder, scan)
            finally:
                window.close()
    return scan


# --- 突き合わせ ---


def _matches(pattern: str, key: str) -> bool:
    """台帳の行 ``pattern`` が操作 ``key`` を受けるか ``*`` は ``/`` で区切った 1 段だけに当たる

    ``fnmatch`` の ``*`` は ``/`` もまたぐので、「〔エフェクトを追加…〕/*」が「前の場面へ/ぼかし/
    ぼかし」まで覆い、段の違う操作（前と後の場面へ積む）を足して行を書き忘れても通った
    段の数をそろえて 1 段ずつ照らす ``*`` の無い行は名前そのものと比べる（「再生 / 停止」の
    ように名前の中に ``/`` がある項目を、段に割らない）
    """
    if "*" not in pattern:
        return key == pattern
    parts, segments = pattern.split("/"), key.split("/")
    return len(parts) == len(segments) and all(
        fnmatch.fnmatchcase(segment, part) for part, segment in zip(parts, segments, strict=True)
    )


def unlisted(scan: Scan) -> list[str]:
    """画面にあって台帳に無い操作"""
    patterns = [entry.key for entry in ledger.LEDGER]
    return sorted(k for k in scan.keys if not any(_matches(p, k) for p in patterns))


def stale(scan: Scan) -> list[str]:
    """台帳にあって画面に無い行 一覧の行（``/*``）は、一覧のサブメニューが在れば在るとみなす"""
    missing: list[str] = []
    for entry in ledger.LEDGER:
        pattern = entry.key
        if "*" in pattern:
            head = pattern.split("/*", 1)[0]
            if head in scan.submenus or any(_matches(pattern, k) for k in scan.keys):
                continue
        elif pattern in scan.keys:
            continue
        missing.append(pattern)
    return missing


def _howto(keys: list[str]) -> str:
    rows = "\n".join(f'    missing("{k}", <Issue の番号>),' for k in keys)
    return (
        "src/sashimono/ai/ledger.py の LEDGER に行を足す（AI の道具があれば same / partial、"
        "無ければ missing、出さないなら hidden）\n" + rows
    )


@pytest.fixture
def media(video_media: MediaItem, transcript: Transcript) -> MediaItem:
    return video_media.with_transcript(transcript)


@pytest.fixture
def screen(tmp_path: Path, media: MediaItem, qt_application: QApplication) -> Scan:
    del qt_application
    with refuse_windows():
        return scan_screen(tmp_path, media)


class TestEveryScreenOperationIsListed:
    def test_nothing_on_screen_is_missing_from_the_ledger(self, screen: Scan) -> None:
        # 落ちると: メニューや右クリックに足した操作を AI ができるのか誰も確かめないまま出て、
        # 「人ができて AI にできない操作」が黙って増える
        missing = unlisted(screen)
        assert not missing, "画面の操作が台帳に無い\n" + _howto(missing)

    def test_every_ledger_row_is_still_on_screen(self, screen: Scan) -> None:
        # 落ちると: 消した・名前を変えた項目の行が残り、AI にできる操作の数え方が実物とずれる
        left = stale(screen)
        assert not left, f"台帳にあって画面に無い行（消すか、場面を scan_screen に足す）: {left}"

    def test_every_button_is_shown_in_some_scene(self, screen: Scan) -> None:
        # 落ちると: ある場面でしか出ないボタン（AviUtl2 の組み方の〔組み方を「標準」にする〕
        # など）を取り出す場面が無く、台帳に行が無くても上の見張りが通ってしまう
        left = unreached_sites(screen)
        assert not left, (
            "どの場面でも出なかったボタンを作る所がある その所のボタンが出る場面を "
            f"scan_screen に足す: {left}"
        )

    def test_every_unlock_button_has_a_row(self) -> None:
        # 落ちると: 灰色の欄を使えるようにする切り替え（#301）を足したのに、出る場面が
        # 試験に無いまま台帳から漏れる 切り替えは定義から全部引いて確かめる
        patterns = [entry.key for entry in ledger.LEDGER]
        keys = [
            f"設定パネル/{unlock.button}｜{unlock.detail.split(chr(10))[0].split(': ')[0]}"
            for definition in source_registry.all()
            for unlock in definition.unlocks.values()
        ]
        assert keys
        missing = sorted({k for k in keys if not any(_matches(p, k) for p in patterns)})
        assert not missing, "灰色の欄の切り替えが台帳に無い\n" + _howto(missing)

    def test_the_scan_reaches_every_place(self, screen: Scan) -> None:
        # 落ちると: どこかのパネルを組めずに 0 件で通り、そのパネルの足し忘れを見逃す
        places = {key.split("/", 1)[0] for key in screen.keys}
        assert places >= {
            "メニュー",
            "タイムラインの右クリック",
            "タイムラインの〔＋ トラック追加〕",
            "プレビューの右クリック",
            "素材一覧",
            "素材一覧の右クリック",
            "字幕パネル",
            "字幕パネルの右クリック",
            "設定パネル",
            "設定パネルの〔エフェクトを追加…〕",
            "設定パネルの〔プリセット…〕",
            "設定パネルの縁取りの〔効果〕",
            "設定パネルの ◆ の右クリック",
            "シーンの帯",
            "再生の操作",
        }
        assert len([k for k in screen.keys if k.startswith("メニュー/")]) >= 60


class TestTheGuardCatchesAdditions:
    """足したのに台帳に無い操作を、本当に見つけるか（見張りそのものの試験）"""

    def test_a_new_menu_bar_item_is_caught(self, qt_application: QApplication) -> None:
        # 落ちると: メニューバーに足した項目を見張りが見ず、台帳の足し忘れが通る
        del qt_application
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            with refuse_windows():
                window._add(window._menu("ファイル"), "試しの項目", QKeySequence(), lambda: None)
                scan = Scan()
                for name in window._actions:
                    scan.keys.add(f"メニュー/{name}")
        finally:
            window.close()
        assert "メニュー/ファイル/試しの項目" in unlisted(scan)

    def test_a_new_right_click_item_is_caught(
        self, tmp_path: Path, media: MediaItem, qt_application: QApplication
    ) -> None:
        # 落ちると: タイムラインの右クリックに足した項目を見張りが見ず、台帳の足し忘れが通る
        del qt_application
        original = TimelineView.build_context_menu

        def with_extra(view: TimelineView, position: QPoint) -> QMenu:
            menu = original(view, position)
            menu.addAction("試しの右クリック").triggered.connect(lambda: None)
            return menu

        with refuse_windows(), mock.patch.object(TimelineView, "build_context_menu", with_extra):
            scan = scan_screen(tmp_path, media)
        assert "タイムラインの右クリック/試しの右クリック" in unlisted(scan)

    def test_a_list_row_covers_only_its_own_depth(self) -> None:
        # 落ちると: 一覧の行の * が段をまたいで下のサブメニューまで覆い、段の違う操作
        # （場面切り替えの前と後の場面へ積む など）を足して行を書き忘れても通る
        assert _matches("甲/*", "甲/乙")
        assert not _matches("甲/*", "甲/乙/丙")
        assert _matches("甲/*/*", "甲/乙/丙")
        assert not _matches("甲/*/*", "甲/乙")
        assert _matches("メニュー/再生/再生 / 停止", "メニュー/再生/再生 / 停止")
        deeper = "設定パネルの〔エフェクトを追加…〕/新しい段/ぼかし/ぼかし"
        assert unlisted(Scan(keys={deeper})) == [deeper]

    def test_a_button_never_shown_is_caught(self) -> None:
        # 落ちると: 作る所を数え損ね、どの場面でも出ないボタンを見張りが見逃す
        everything = Scan(sites={(m, line) for m in PANEL_MODULES for line in button_sites(m)})
        assert not unreached_sites(everything)
        inspector = "sashimono.ui.inspector.panel"
        hidden_one = min(button_sites(inspector))
        everything.sites.discard((inspector, hidden_one))
        assert unreached_sites(everything) == [f"{inspector}:{hidden_one}"]

    def test_an_opened_window_stops_the_scan(self, qt_application: QApplication) -> None:
        # 落ちると: 取り出しの途中で開いた窓を誰も閉じず、CI が時間切れまで止まる
        del qt_application
        for opening in (
            lambda: QDialog().exec(),
            lambda: QMessageBox.question(None, "試し", "試し"),
            lambda: QFileDialog.getSaveFileName(None),
        ):
            with pytest.raises(pytest.fail.Exception), refuse_windows():
                opening()

    def test_a_removed_item_leaves_a_stale_row(self) -> None:
        # 落ちると: 画面から消えた項目の行が台帳に残っても見張りが気付かない
        scan = Scan(keys={entry.key for entry in ledger.LEDGER if "*" not in entry.key})
        scan.keys.discard("メニュー/編集/元に戻す")
        assert "メニュー/編集/元に戻す" in stale(scan)
