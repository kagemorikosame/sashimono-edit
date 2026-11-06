r"""選択が変わったときと編集のたびの重さを、実際の窓（MainWindow）で測る

    .venv\Scripts\python.exe tools\bench_inspector.py
    .venv\Scripts\python.exe tools\bench_inspector.py --repeat 5

Issue #429・#431・#432 と同じ測り方 テキストクリップを並べた窓を画面に出さずに作り、
操作のあとに溜まったイベント（消す部品の後片付けと描き直し）を流し終えるまでを 1 回とする
流し終えるまでを入れないと、部品を消す重さ（アプリ全体のイベントフィルタを通る分）が測れない

測る物

- 範囲選択 空いた所から囲み、40 回に分けて動かす 1 回の動きの中央値と最大（#429）
  グループなし 200 本と、2000 本を 2 本ずつ束ねた物
- 選び直し 選んだクリップのエフェクトの数ごと（#431）
- 選んでいるクリップにエフェクトが 15 個あるときの、別のクリップを動かす編集と値の変更（#431）
- 素材の多い窓での、クリップを動かす編集 1 回（#432）

同じ機械でも数 ms は揺れる 前後を比べるときは同じ日に交互に測る
"""

from __future__ import annotations

import argparse
import atexit
import io
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 開発者本人の設定に触らない
_base = Path(tempfile.mkdtemp(prefix="sashimono-inspector-bench-"))
atexit.register(shutil.rmtree, _base, True)
os.environ["APPDATA"] = str(_base / "roaming")
os.environ["LOCALAPPDATA"] = str(_base / "local")
# 画面を開かない 測るのは部品を作る・消す重さで、画面への転送ではない
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sashimono.compat.looks import effects_for_clip  # noqa: E402
from sashimono.core.commands import (  # noqa: E402
    AddClip,
    AddEffect,
    AddTrack,
    MoveClip,
    ParamPath,
    SetParam,
)
from sashimono.core.commands.group import GroupClips  # noqa: E402
from sashimono.core.model import (  # noqa: E402
    AnimatedValue,
    Clip,
    ClipId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate  # noqa: E402
from sashimono.effects.sources import TEXT  # noqa: E402
from sashimono.ui.main_window import MainWindow  # noqa: E402

#: 囲みを何回の動きに分けるか（Issue #429 と同じ）
MARQUEE_STEPS = 40
#: クリップ 1 本の長さと間（フレーム）
CLIP_FRAMES = 30
GAP_FRAMES = 6


def _project(clips: int, tracks: int, *, media: int = 0) -> Project:
    """テキストクリップを ``tracks`` 本のトラックへ順に並べたプロジェクト"""
    project = Project.create(ProjectSettings(width=640, height=360, frame_rate=FrameRate(30)))
    made = [Track(TrackKind.VIDEO, f"V{i + 1}") for i in range(tracks)]
    for track in made:
        project = AddTrack(track).apply(project)
    for index in range(clips):
        track = made[index % tracks]
        start = (index // tracks) * (CLIP_FRAMES + GAP_FRAMES)
        clip = Clip(timeline_start=start, duration=CLIP_FRAMES, source=TEXT.create(text=f"{index}"))
        project = AddClip(track.id, clip).apply(project)
    if media:
        items = tuple(
            MediaItem(path=_base / f"media-{index}.mp4", duration=Fraction(10))
            for index in range(media)
        )
        project = project.with_media((*project.media, *items))
    return project


def _with_effects(project: Project, clip_id: ClipId, count: int) -> Project:
    """設定パネルの〔エフェクトを追加〕に並ぶ物を頭から順に積む（足りなければ繰り返す）"""
    located = project.timeline.locate_clip(clip_id)
    assert located is not None
    usable = effects_for_clip(project, *located)
    for index in range(count):
        effect = usable[index % len(usable)].create()
        project = AddEffect(clip_id, effect).apply(project)
    return project


def _settle(app: QApplication) -> None:
    """溜まったイベントを流し切る 後で消す部品（deleteLater）も消し終える"""
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    app.processEvents()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _timed[A](app: QApplication, action: Callable[[A], object], argument: A) -> float:
    """``action(argument)`` から溜まったイベントを流し終えるまでの時間（ms）"""
    began = time.perf_counter()
    action(argument)
    _settle(app)
    return (time.perf_counter() - began) * 1000


def _open(app: QApplication, project: Project) -> MainWindow:
    window = MainWindow(project, confirm_unsaved=False)
    window.resize(1600, 900)
    window.show()
    view = window._timeline
    view.resize(1586, 262)
    view.zoom_to_fit()
    _settle(app)
    return window


def _close(app: QApplication, window: MainWindow) -> None:
    window.close()
    window.deleteLater()
    _settle(app)


def _mouse(kind: QEvent.Type, point: QPoint, *, held: bool = True) -> QMouseEvent:
    buttons = Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton
    button = (
        Qt.MouseButton.LeftButton if kind is not QEvent.Type.MouseMove else Qt.MouseButton.NoButton
    )
    local = QPointF(point)
    return QMouseEvent(kind, local, local, button, buttons, Qt.KeyboardModifier.NoModifier)


def bench_marquee(app: QApplication, clips: int, tracks: int, *, grouped: bool) -> list[float]:
    """空いた所から右上へ囲む 1 回の動きごとの時間（ms）"""
    project = _project(clips, tracks)
    if grouped:
        ids = [c.id for t in project.timeline.tracks for c in t.clips]
        for index in range(0, len(ids) - 1, 2):
            project = GroupClips((ids[index], ids[index + 1])).apply(project)
    window = _open(app, project)
    view = window._timeline
    bands = view._layout.bands(project.timeline)
    origin = QPoint(view.width() - 4, min(bands[-1].bottom + 4, view.height() - 2))
    target = QPoint(100, bands[0].top + 2)
    view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, origin))
    _settle(app)
    steps: list[float] = []
    for step in range(1, MARQUEE_STEPS + 1):
        ratio = step / MARQUEE_STEPS
        point = QPoint(
            round(origin.x() + (target.x() - origin.x()) * ratio),
            round(origin.y() + (target.y() - origin.y()) * ratio),
        )
        steps.append(_timed(app, view.mouseMoveEvent, _mouse(QEvent.Type.MouseMove, point)))
    release = _timed(
        app,
        view.mouseReleaseEvent,
        _mouse(QEvent.Type.MouseButtonRelease, target, held=False),
    )
    selected = len(view._selection)
    _close(app, window)
    print(
        f"  囲み {clips} 本{'（2 本ずつグループ）' if grouped else ''}: "
        f"合計 {sum(steps) + release:.0f} ms 1 回の動き 中央値 {statistics.median(steps):.1f} "
        f"最大 {max(steps):.1f} 離す {release:.1f}（{selected} 本を選択）"
    )
    return steps


def bench_reselect(app: QApplication, effects: int, repeat: int) -> float:
    """エフェクトを ``effects`` 個持つ同じ形のクリップ 2 本を交互に選ぶ 1 回の時間（ms）"""
    project = _project(300, 6)
    first, second = (c.id for c in project.timeline.tracks[0].clips[:2])
    project = _with_effects(project, first, effects)
    project = _with_effects(project, second, effects)
    window = _open(app, project)
    view = window._timeline
    times = [
        _timed(app, view.set_selection, (clip,)) for _ in range(repeat) for clip in (first, second)
    ]
    _close(app, window)
    return statistics.median(times)


def bench_edits_with_effects(app: QApplication, effects: int, repeat: int) -> tuple[float, float]:
    """``effects`` 個のエフェクトを持つクリップを選んだまま、別のクリップを動かす・値を変える"""
    project = _project(300, 6)
    chosen = project.timeline.tracks[0].clips[0].id
    other_track = project.timeline.tracks[1]
    other = other_track.clips[-1]
    project = _with_effects(project, chosen, effects)
    window = _open(app, project)
    window._timeline.set_selection((chosen,))
    _settle(app)
    moves = [
        _timed(
            app,
            window.execute,
            MoveClip(other.id, other.timeline_start + 1 + n % 2, other_track.id),
        )
        for n in range(repeat * 2)
    ]
    path = ParamPath.of_clip(chosen, "opacity")
    values = [
        _timed(app, window.execute, SetParam(path, AnimatedValue(0.5 + 0.01 * (n % 10))))
        for n in range(repeat * 2)
    ]
    _close(app, window)
    return statistics.median(moves), statistics.median(values)


def bench_media_edit(app: QApplication, media: int, repeat: int) -> float:
    """素材を ``media`` 本持つ窓で、クリップを 1 本動かす編集 1 回（ms）"""
    project = _project(100, 4, media=media)
    track = project.timeline.tracks[0]
    clip = track.clips[-1]
    window = _open(app, project)
    times = [
        _timed(app, window.execute, MoveClip(clip.id, clip.timeline_start + 1 + n % 2, track.id))
        for n in range(repeat * 2)
    ]
    _close(app, window)
    return statistics.median(times)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--repeat", type=int, default=5, help="選び直しと編集を何往復するか")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat は 1 以上にする")
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)

    print("範囲選択（#429）")
    bench_marquee(app, 200, 4, grouped=False)
    bench_marquee(app, 2000, 10, grouped=True)

    print("選び直し 1 回の中央値（#431）")
    for effects in (0, 3, 8, 15):
        print(f"  エフェクト {effects:>2} 個: {bench_reselect(app, effects, args.repeat):.1f} ms")

    print("エフェクト 15 個のクリップを選んだままの編集（#431）")
    move, value = bench_edits_with_effects(app, 15, args.repeat)
    print(
        f"  別のクリップを 1 コマ動かす: {move:.1f} ms 選んだクリップの値を変える: {value:.1f} ms"
    )
    move, value = bench_edits_with_effects(app, 0, args.repeat)
    print(f"  （エフェクト 0 個のとき {move:.1f} ms / {value:.1f} ms）")

    print("素材の多い窓でクリップを動かす編集 1 回（#432）")
    for media in (20, 200, 1000):
        print(f"  素材 {media:>4} 本: {bench_media_edit(app, media, args.repeat):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
