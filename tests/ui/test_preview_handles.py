"""プレビューで絵を直接動かす（#27 P7）

枠の中をドラッグで位置、角で拡大率、角の外で回転 離したら 1 段 キーフレームのある値は
再生ヘッドの所へ点を打つ（利用者の決定） どれも壊れても絵は出続けるので、画面を
見ていても「動かしたのに値が違う」「取り消しが何段も要る」に気付きにくい

プレビューは GL を作らずに使う（窓に出さない） 素材のクリップの枠は GL 無しで求まる
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddMedia,
    AddTrack,
    Command,
    Document,
    SetKeyframe,
    SetParam,
)
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, with_fixed_items
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Keyframe,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.render import RenderQuality
from sashimono.ui.main_window import MainWindow
from sashimono.ui.preview import PreviewWidget
from sashimono.ui.preview_handles import (
    KEYFRAME_DRAG_SHIFT_ALL,
    Grip,
    fixed_transform,
    hit_test,
    transform_commands,
)
from sashimono.ui.workspace import Preferences, PreferenceStore

SETTINGS = ProjectSettings(width=320, height=180, frame_rate=FrameRate(30))

#: 試験用のプレビューを作る 窓には出さない
MakeWidget = Callable[..., PreviewWidget]


#: 作った素材 置くときに素材の一覧へ入れる（素材の無いクリップは置けない）
_MADE: dict[MediaId, MediaItem] = {}


def _media(name: str = "絵.png") -> MediaItem:
    """160 × 90 の素材 画素で置くと画面（320 × 180）の真ん中の (80, 45)-(240, 135)"""
    item = MediaItem(
        path=Path(f"C:/素材/{name}"),
        duration=Fraction(10),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=160,
                height=90,
                frame_rate=FrameRate(30),
                time_base=Fraction(1, 30),
                codec="png",
            ),
        ),
    )
    _MADE[item.id] = item
    return item


def _clip(media: MediaItem, *, fixed: bool = True, start: int = 0) -> Clip:
    clip = Clip(timeline_start=start, duration=60, media_id=media.id, native_size=True)
    return with_fixed_items(clip, picture=True) if fixed else clip


def _project(*clips: Clip, locked: bool = False) -> tuple[Project, list[Track]]:
    """1 本ずつ別のトラックへ 後のトラックほど手前 素材は :func:`_media` で作った物を先に入れる"""
    project = Project.create(SETTINGS)
    for clip in clips:
        if clip.media_id is not None and project.find_media(clip.media_id) is None:
            project = AddMedia(_MADE[clip.media_id]).apply(project)
    tracks = []
    for index, clip in enumerate(clips):
        track = Track(kind=TrackKind.VIDEO, name=f"V{index + 1}")
        project = AddTrack(track).apply(project)
        project = AddClip(track.id, clip).apply(project)
        tracks.append(track)
    if locked:
        # ロック中のトラックへは置けないので、置いてから鍵を掛ける
        timeline = project.timeline
        locked_tracks = tuple(replace(track, locked=True) for track in timeline.tracks)
        project = project.with_timeline(replace(timeline, tracks=locked_tracks))
    return project, tracks


def _current(project: Project, clip_id: ClipId) -> Clip:
    located = project.timeline.locate_clip(clip_id)
    assert located is not None
    return located[1]


def _value(project: Project, clip_id: ClipId, name: str, frame: int = 0) -> float:
    effect = fixed_transform(_current(project, clip_id))
    assert effect is not None
    value = effect.params[name]
    assert isinstance(value, AnimatedValue)
    return value.at(frame)


class _Recorder:
    """プレビューが出した合図を覚える"""

    def __init__(self, widget: PreviewWidget) -> None:
        self.committed: list[tuple[list[Command], str]] = []
        self.previewed: list[list[Command]] = []
        self.picked: list[str] = []
        widget.commands_requested.connect(
            lambda commands, label: self.committed.append((list(commands), label))
        )
        widget.preview_requested.connect(lambda commands: self.previewed.append(list(commands)))
        widget.clip_picked.connect(self.picked.append)


def _send(
    widget: PreviewWidget,
    kind: QEvent.Type,
    point: tuple[float, float],
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    pressed = kind is not QEvent.Type.MouseButtonRelease
    event = QMouseEvent(
        kind,
        QPointF(*point),
        QPointF(*point),
        Qt.MouseButton.LeftButton if kind is not QEvent.Type.MouseMove else Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton if pressed else Qt.MouseButton.NoButton,
        modifiers,
    )
    QApplication.sendEvent(widget, event)


def _drag(
    widget: PreviewWidget,
    start: tuple[float, float],
    *path: tuple[float, float],
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    _send(widget, QEvent.Type.MouseButtonPress, start, modifiers)
    for point in path:
        _send(widget, QEvent.Type.MouseMove, point, modifiers)
    _send(widget, QEvent.Type.MouseButtonRelease, path[-1] if path else start, modifiers)


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


@pytest.fixture
def make_widget(qt_application: QApplication) -> Iterator[MakeWidget]:
    del qt_application
    made: list[PreviewWidget] = []

    def make(project: Project, selection: ClipId | None = None) -> PreviewWidget:
        widget = PreviewWidget(project, prefetch_bytes=0, prefetch_thread=False)
        # 絵の出る所と合成を 1 対 1 にする 画面の座標がそのまま合成の画素
        widget.resize(320, 180)
        widget.set_selection(selection)
        made.append(widget)
        return widget

    yield make
    for widget in made:
        widget.shutdown()
        widget.deleteLater()
    # その場で壊す イベントループを回さない試験では、消す頼みが最後まで残る GL を作った
    # ウィジェットがアプリより後まで残ると、pytest が終わるときに落ちる（0xC0000409）
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


class TestMovingInside:
    def test_dragging_inside_moves_x_and_y(self, make_widget: MakeWidget) -> None:
        # Y は上が正 画面で上へ引いたら Y が増える 逆だと枠と絵が反対へ動く
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (160, 90), (175, 85), (190, 70))
        assert len(seen.committed) == 1
        commands, label = seen.committed[0]
        assert label == "プレビューで位置を変更"
        moved = _apply(project, commands)
        assert _value(moved, clip.id, "pos_x") == pytest.approx(30.0)
        assert _value(moved, clip.id, "pos_y") == pytest.approx(20.0)

    def test_it_is_one_undo_step(self, make_widget: MakeWidget) -> None:
        # 動かしている途中は履歴に積まない 離したときに 1 段 何十段も積むと戻すのに困る
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (160, 90), (170, 90), (180, 90), (190, 90))
        # 途中の 3 回は見せるだけ
        assert len(seen.previewed) == 3
        document = Document(project)
        commands, label = seen.committed[0]
        with document.checkpoint(label):
            for command in commands:
                document.execute(command)
        assert _value(document.project, clip.id, "pos_x") == pytest.approx(30.0)
        document.undo()
        assert document.project == project
        assert not document.can_undo

    def test_shift_keeps_one_axis(self, make_widget: MakeWidget) -> None:
        # Shift を押しても両方が動くと、横だけ揃えたいのに手ぶれで縦もずれる
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (160, 90), (190, 84), modifiers=Qt.KeyboardModifier.ShiftModifier)
        moved = _apply(project, seen.committed[0][0])
        assert _value(moved, clip.id, "pos_x") == pytest.approx(30.0)
        assert _value(moved, clip.id, "pos_y") == pytest.approx(0.0)

    def test_the_drag_starts_from_the_project_at_the_press(self, make_widget: MakeWidget) -> None:
        # 途中の絵は窓がプレビューへ渡す それを元に数えると、動かした分をもう 1 度足していく
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        widget.preview_requested.connect(
            lambda commands: widget.set_project(_apply(project, list(commands)))
        )
        _drag(widget, (160, 90), (170, 90), (180, 90), (190, 90))
        moved = _apply(project, seen.committed[0][0])
        assert _value(moved, clip.id, "pos_x") == pytest.approx(30.0)


class TestCornersAndTurning:
    def test_a_corner_scales(self, make_widget: MakeWidget) -> None:
        # 中心から角までの距離の比で拡げる 縦の拡大率は横に掛ける比なので触らない
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (240, 135), (280, 157.5))
        commands, label = seen.committed[0]
        assert label == "プレビューで拡大率を変更"
        scaled = _apply(project, commands)
        assert _value(scaled, clip.id, "scale") == pytest.approx(150.0)
        assert _value(scaled, clip.id, "scale_y") == pytest.approx(100.0)

    def test_alt_scales_across_and_down_apart(self, make_widget: MakeWidget) -> None:
        # Alt で横と縦を別々に変えられないと、絵を横長や縦長に伸ばすのに数を打つことになる
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        # 横は 2 倍、縦はそのまま
        _drag(widget, (240, 135), (320, 135), modifiers=Qt.KeyboardModifier.AltModifier)
        scaled = _apply(project, seen.committed[0][0])
        assert _value(scaled, clip.id, "scale") == pytest.approx(200.0)
        # 縦は横に掛ける比 横を 2 倍にしたぶん半分にしておかないと、縦まで伸びる
        assert _value(scaled, clip.id, "scale_y") == pytest.approx(50.0)

    def test_outside_a_corner_turns_clockwise(self, make_widget: MakeWidget) -> None:
        # 回転は時計回りが正（YMM4 と同じ） 画面の下向きの Y で数え間違えると逆に回る
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        pivot = (160.0, 90.0)
        start = (252.0, 142.0)
        radius = math.hypot(start[0] - pivot[0], start[1] - pivot[1])
        begin = math.atan2(start[1] - pivot[1], start[0] - pivot[0])
        # 画面で時計回り（下向きの Y で角度が増える向き）に 40 度
        path = [
            (
                pivot[0] + radius * math.cos(begin + math.radians(step)),
                pivot[1] + radius * math.sin(begin + math.radians(step)),
            )
            for step in (10, 20, 30, 40)
        ]
        _drag(widget, start, *path)
        commands, label = seen.committed[0]
        assert label == "プレビューで回転を変更"
        assert _value(_apply(project, commands), clip.id, "rotation") == pytest.approx(40.0)

    def test_shift_turns_in_fifteen_degree_steps(self, make_widget: MakeWidget) -> None:
        # 刻みが効かないと、ちょうど 30 度や 45 度に合わせられず数を打ち直すことになる
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        pivot = (160.0, 90.0)
        start = (252.0, 142.0)
        radius = math.hypot(start[0] - pivot[0], start[1] - pivot[1])
        begin = math.atan2(start[1] - pivot[1], start[0] - pivot[0])
        path = [
            (
                pivot[0] + radius * math.cos(begin + math.radians(step)),
                pivot[1] + radius * math.sin(begin + math.radians(step)),
            )
            for step in (10, 20, 26)
        ]
        _drag(widget, start, *path, modifiers=Qt.KeyboardModifier.ShiftModifier)
        rotation = _value(_apply(project, seen.committed[0][0]), clip.id, "rotation")
        assert rotation == pytest.approx(30.0)

    def test_turning_past_half_a_turn_keeps_going(self, make_widget: MakeWidget) -> None:
        # 押した所との差で数えると、半周を越えた所で -180 へ跳んで逆へ回る
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        pivot = (160.0, 90.0)
        start = (252.0, 142.0)
        radius = math.hypot(start[0] - pivot[0], start[1] - pivot[1])
        begin = math.atan2(start[1] - pivot[1], start[0] - pivot[0])
        path = [
            (
                pivot[0] + radius * math.cos(begin + math.radians(step)),
                pivot[1] + radius * math.sin(begin + math.radians(step)),
            )
            for step in range(30, 271, 30)
        ]
        _drag(widget, start, *path)
        rotation = _value(_apply(project, seen.committed[0][0]), clip.id, "rotation")
        assert rotation == pytest.approx(270.0)

    def test_the_grips_are_found_in_widget_pixels(self) -> None:
        # 角は枠の中と重なる 先に中を見ると、角をつまんでも動かすだけになる
        corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
        found = hit_test(corners, (2.0, 2.0), grip=8, reach=24, knob=22)
        assert found is not None and found.grip is Grip.SCALE and found.corner == 0
        found = hit_test(corners, (50.0, 25.0), grip=8, reach=24, knob=22)
        assert found is not None and found.grip is Grip.MOVE
        found = hit_test(corners, (110.0, 60.0), grip=8, reach=24, knob=22)
        assert found is not None and found.grip is Grip.ROTATE
        assert hit_test(corners, (200.0, 200.0), grip=8, reach=24, knob=22) is None


class TestKeyframes:
    def _animated(self, media: MediaItem) -> Clip:
        clip = _clip(media)
        effects = tuple(
            effect.with_param(
                "pos_x",
                AnimatedValue(0.0, keyframes=(Keyframe(0, 0.0), Keyframe(20, 100.0))),
            )
            if effect.fixed and effect.kind == TRANSFORM_EFFECT_KIND
            else effect
            for effect in clip.effects
        )
        return replace(clip, effects=effects)

    def test_a_point_goes_at_the_playhead(self, make_widget: MakeWidget) -> None:
        # 利用者の決定 SetParam で差し替えると点が全部消えて止まった値になる
        media = _media()
        clip = self._animated(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        widget.set_frame(10)
        seen = _Recorder(widget)
        # 10 フレーム目の X は 50 枠は右へ 50 ずれた所にある
        _drag(widget, (210, 90), (240, 90))
        commands = seen.committed[0][0]
        assert not any(isinstance(c, SetParam) and c.path.name == "pos_x" for c in commands)
        moved = _apply(project, commands)
        effect = fixed_transform(_current(moved, clip.id))
        assert effect is not None
        value = effect.params["pos_x"]
        assert isinstance(value, AnimatedValue)
        assert [k.frame for k in value.keyframes] == [0, 10, 20]
        assert [k.value for k in value.keyframes] == pytest.approx([0.0, 80.0, 100.0])
        # 動かしていない Y は点を持たないので、そのまま差し替える
        assert _value(moved, clip.id, "pos_y") == pytest.approx(0.0)

    def test_the_playhead_is_kept_inside_the_clip(self) -> None:
        # クリップの外の時刻に点を打つと、クリップの中では何も変わらない
        media = _media()
        clip = self._animated(media)
        project, _ = _project(clip)
        commands = transform_commands(project, clip.id, {"pos_x": 7.0}, 500)
        assert [c.frame for c in commands if isinstance(c, SetKeyframe)] == [59]

    def test_the_setting_can_shift_every_point(self, make_widget: MakeWidget) -> None:
        # 設定が効かずに再生ヘッドへ点を打つと、動き全体をずらしたい人の曲線に余計な点が増える
        media = _media()
        clip = self._animated(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        widget.set_frame(10)
        widget.set_keyframe_drag(KEYFRAME_DRAG_SHIFT_ALL)
        seen = _Recorder(widget)
        _drag(widget, (210, 90), (240, 90))
        effect = fixed_transform(_current(_apply(project, seen.committed[0][0]), clip.id))
        assert effect is not None
        value = effect.params["pos_x"]
        assert isinstance(value, AnimatedValue)
        assert [k.frame for k in value.keyframes] == [0, 20]
        assert [k.value for k in value.keyframes] == pytest.approx([30.0, 130.0])

    def test_shifting_every_point_stays_in_range(self) -> None:
        # 再生ヘッドの値が範囲の中でも、別の時刻の点は上限の近くにあることがある
        # 範囲の外の拡大率は、設定パネルで開いたときに黙って端へ丸められる
        media = _media()
        clip = _clip(media)
        effects = tuple(
            e.with_param(
                "scale", AnimatedValue(100.0, keyframes=(Keyframe(0, 100.0), Keyframe(20, 700.0)))
            )
            if e.kind == TRANSFORM_EFFECT_KIND
            else e
            for e in clip.effects
        )
        clip = replace(clip, effects=effects)
        project, _ = _project(clip)
        commands = transform_commands(
            project, clip.id, {"scale": 300.0}, 0, keyframes=KEYFRAME_DRAG_SHIFT_ALL
        )
        assert [c.value for c in commands if isinstance(c, SetKeyframe)] == [300.0, 800.0]

    def test_returning_to_the_start_adds_no_point(self) -> None:
        # 元の値へ戻して離しただけで、その時刻に直線の点が入ると曲線の出方が変わる
        media = _media()
        clip = self._animated(media)
        project, _ = _project(clip)
        assert transform_commands(project, clip.id, {"pos_x": 50.0}, 10) == []
        shifted = transform_commands(
            project, clip.id, {"pos_x": 50.0}, 10, keyframes=KEYFRAME_DRAG_SHIFT_ALL
        )
        assert shifted == []


class TestDragEndings:
    def test_releasing_another_button_keeps_the_drag(self, make_widget: MakeWidget) -> None:
        # 左で掴んだまま右を離しただけで掴むのが消えると、途中の絵が残ったままになる
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _send(widget, QEvent.Type.MouseButtonPress, (160, 90))
        _send(widget, QEvent.Type.MouseMove, (180, 90))
        right = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(180, 90),
            QPointF(180, 90),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(widget, right)
        assert seen.committed == []
        _send(widget, QEvent.Type.MouseButtonRelease, (180, 90))
        assert len(seen.committed) == 1

    def test_a_change_from_elsewhere_drops_the_drag(self, make_widget: MakeWidget) -> None:
        # 掴んでいる途中に取り消しなどで中身が変わった 掴んだ時点から作った値を離したときに
        # 当てると、その変更を上書きする
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        widget.preview_requested.connect(
            lambda commands: widget.set_project(_apply(project, list(commands)))
        )
        _send(widget, QEvent.Type.MouseButtonPress, (160, 90))
        _send(widget, QEvent.Type.MouseMove, (180, 90))
        changed = _apply(project, transform_commands(project, clip.id, {"scale": 200.0}, 0))
        widget.set_project(changed)
        _send(widget, QEvent.Type.MouseMove, (190, 90))
        _send(widget, QEvent.Type.MouseButtonRelease, (190, 90))
        assert seen.committed == []

    def test_stepping_frames_keeps_the_pressed_frame(self, make_widget: MakeWidget) -> None:
        # 掴んだまま矢印でコマを送ると、送った先の時刻に、掴んだ時刻の値から数えた点が入る
        media = _media()
        clip = TestKeyframes()._animated(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        widget.set_frame(10)
        seen = _Recorder(widget)
        _send(widget, QEvent.Type.MouseButtonPress, (210, 90))
        widget.set_frame(15)
        _send(widget, QEvent.Type.MouseMove, (240, 90))
        _send(widget, QEvent.Type.MouseButtonRelease, (240, 90))
        commands = seen.committed[0][0]
        keys = [c for c in commands if isinstance(c, SetKeyframe)]
        assert [c.frame for c in keys] == [10]
        assert keys[0].value == pytest.approx(80.0)

    def test_playing_drops_the_drag(self, make_widget: MakeWidget) -> None:
        # 再生中は枠を出さない 見えない枠を掴んだまま離すと、見ていない値が確定する
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _send(widget, QEvent.Type.MouseButtonPress, (160, 90))
        _send(widget, QEvent.Type.MouseMove, (180, 90))
        widget.set_playing(True)
        _send(widget, QEvent.Type.MouseMove, (190, 90))
        _send(widget, QEvent.Type.MouseButtonRelease, (190, 90))
        assert seen.committed == []
        # 見せていた途中の絵を元へ戻す頼みが出ている
        assert seen.previewed[-1] == []

    def test_a_click_leaves_the_keys_with_the_timeline(self, make_widget: MakeWidget) -> None:
        # プレビューがフォーカスを取ると、選んだ後の矢印（コマ送り）や Delete が
        # タイムラインへ届かなくなる 掴み終えたら借りたキーボードも返す
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        assert widget.focusPolicy() == Qt.FocusPolicy.NoFocus
        _drag(widget, (160, 90), (180, 90))
        assert QWidget.keyboardGrabber() is not widget

    def test_escape_still_cancels_while_dragging(self, make_widget: MakeWidget) -> None:
        # フォーカスを取らなくても、掴んでいる間の Esc は受けて途中の絵を元へ戻す
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _send(widget, QEvent.Type.MouseButtonPress, (160, 90))
        _send(widget, QEvent.Type.MouseMove, (180, 90))
        escape = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(widget, escape)
        _send(widget, QEvent.Type.MouseButtonRelease, (180, 90))
        assert seen.committed == []
        assert seen.previewed[-1] == []
        assert QWidget.keyboardGrabber() is not widget

    def test_a_muted_track_has_no_outline(self, make_widget: MakeWidget) -> None:
        # ミュートしたトラックの絵は描かれない 枠が残ると見えない絵を動かすことになる
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        timeline = project.timeline
        muted = tuple(replace(track, muted=True) for track in timeline.tracks)
        project = project.with_timeline(replace(timeline, tracks=muted))
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (160, 90), (200, 90))
        assert seen.committed == [] and seen.previewed == []


class TestOldClips:
    def test_a_clip_without_the_placement_gets_it_in_the_same_step(
        self, make_widget: MakeWidget
    ) -> None:
        # 前の版のファイルのクリップは配置を持たない 足してから値を入れ、1 回の取り消しで戻す
        media = _media()
        clip = _clip(media, fixed=False)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (160, 90), (170, 100))
        commands = seen.committed[0][0]
        assert isinstance(commands[0], AddEffect) and commands[0].effect.fixed
        moved = _apply(project, commands)
        assert _value(moved, clip.id, "pos_x") == pytest.approx(10.0)
        assert _value(moved, clip.id, "pos_y") == pytest.approx(-10.0)

    def test_no_change_adds_nothing(self) -> None:
        # 動かしていないのに配置を足すと、前の版のクリップを選んだだけで取り消しの段が増える
        media = _media()
        clip = _clip(media, fixed=False)
        project, _ = _project(clip)
        assert transform_commands(project, clip.id, {"pos_x": 0.0}, 0) == []


class TestPicking:
    def test_a_click_picks_the_front_clip(self, make_widget: MakeWidget) -> None:
        # 後のトラックほど手前に描く 奥のクリップを選ぶと、見えている物と違う物が動く
        back_media, front_media = _media("奥.png"), _media("手前.png")
        back, front = _clip(back_media), _clip(front_media)
        project, _ = _project(back, front)
        widget = make_widget(project)
        seen = _Recorder(widget)
        _send(widget, QEvent.Type.MouseButtonPress, (160, 90))
        _send(widget, QEvent.Type.MouseButtonRelease, (160, 90))
        assert seen.picked == [str(front.id)]
        assert widget.selection == front.id
        # 押しただけなら何も変えない
        assert seen.committed == []

    def test_a_click_where_only_the_back_one_is(self, make_widget: MakeWidget) -> None:
        # 手前の絵が無い所で奥を選べないと、重なった奥のクリップをプレビューから動かせない
        back_media, front_media = _media("奥.png"), _media("手前.png")
        back = _clip(back_media)
        front = _clip(front_media)
        # 手前の絵を右へ寄せる 左の端には奥の絵だけ
        front = replace(
            front,
            effects=tuple(
                e.with_param("pos_x", AnimatedValue(70.0)) if e.kind == TRANSFORM_EFFECT_KIND else e
                for e in front.effects
            ),
        )
        project, _ = _project(back, front)
        widget = make_widget(project)
        seen = _Recorder(widget)
        _send(widget, QEvent.Type.MouseButtonPress, (90, 90))
        _send(widget, QEvent.Type.MouseButtonRelease, (90, 90))
        assert seen.picked == [str(back.id)]

    def test_empty_space_keeps_the_selection(self, make_widget: MakeWidget) -> None:
        # 何も無い所を押しただけで選択が外れると、設定パネルが空になって戸惑う
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project)
        seen = _Recorder(widget)
        _send(widget, QEvent.Type.MouseButtonPress, (10, 10))
        assert seen.picked == []
        assert widget.selection is None


class TestLockAndSetting:
    def test_a_locked_track_does_not_move(self, make_widget: MakeWidget) -> None:
        # ロックは誤って動かさないための物 プレビューから動かせると、ロックが効かない所ができる
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip, locked=True)
        widget = make_widget(project, clip.id)
        seen = _Recorder(widget)
        _drag(widget, (160, 90), (200, 90))
        _drag(widget, (240, 135), (280, 160))
        assert seen.committed == []
        assert seen.previewed == []

    def test_turning_the_setting_off_stops_everything(self, make_widget: MakeWidget) -> None:
        # 切っても掴めるなら、設定がある方が質が悪い
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project)
        widget.set_handles_enabled(False)
        seen = _Recorder(widget)
        _drag(widget, (160, 90), (200, 90))
        assert seen.picked == []
        widget.set_selection(clip.id)
        _drag(widget, (160, 90), (200, 90))
        assert seen.committed == [] and seen.previewed == []

    @pytest.mark.parametrize("divisor", [2, 4])
    def test_a_lighter_preview_moves_with_the_mouse(
        self, make_widget: MakeWidget, divisor: int
    ) -> None:
        # 画質を落とすと合成は小さいが、X・Y は画面（プロジェクトの解像度）の画素で持ち、
        # 描く側も合成の画素へ縮めて当てる（Issue #151） 窓はプロジェクトと同じ大きさなので、
        # 窓の 20 は X の 20 合成の画素のまま足すと 10（1/4 なら 5）になり、絵がマウスに付いて来ない
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        widget._quality = RenderQuality(divisor)
        seen = _Recorder(widget)
        _drag(widget, (160, 90), (180, 90))
        assert _value(_apply(project, seen.committed[0][0]), clip.id, "pos_x") == pytest.approx(
            20.0
        )


class TestTheOverlayIsDrawnOnEveryPath:
    """paintGL の 3 つの道のどれを通っても枠を重ねる 道ごとに描くと先読みが当たったコマで消える"""

    class _Compositor:
        def present(self, target: int, viewport: tuple[int, int, int, int]) -> None:
            del target, viewport

    class _Renderer:
        def __init__(self) -> None:
            self.compositor = TestTheOverlayIsDrawnOnEveryPath._Compositor()

        def compose(self, frame: int) -> None:
            del frame

        def close(self) -> None:
            pass

    class _Cache:
        enabled = True

        def release(self) -> None:
            pass

        def draw(self, frame: int, target: int, viewport: tuple[int, int, int, int]) -> bool:
            del frame, target, viewport
            return True

    class _Background:
        def __init__(self, hit: bool) -> None:
            self.hit = hit

        def show(self, *args: object) -> bool:
            del args
            return self.hit

        def set_playhead(self, frame: int) -> None:
            del frame

        def close(self) -> None:
            pass

        def take_discarded(self) -> set[MediaId]:
            return set()

    @pytest.mark.parametrize("path", ["background", "background_miss", "cache", "compose"])
    def test_each_path_paints_the_overlay(
        self, make_widget: MakeWidget, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        # どれかの道で枠を描き忘れると、先読みが当たったコマでだけ選んだ絵の枠と掴み所が消える
        project, _ = _project()
        widget = make_widget(project)
        painted: list[int] = []
        monkeypatch.setattr(widget, "_paint_overlay", lambda: painted.append(1))
        monkeypatch.setattr(widget, "_renderer", self._Renderer())
        if path.startswith("background"):
            monkeypatch.setattr(widget, "_background", self._Background(path == "background"))
        elif path == "cache":
            monkeypatch.setattr(widget, "_cache", self._Cache())
        widget.paintGL()
        assert painted == [1]


@pytest.mark.usefixtures("gpu")
class TestTheOverlayOnScreen:
    """本当に GL で描いて、枠の線が絵の縁に出ること（窓には出さずに取り込む）"""

    def _edge_pixel(self, widget: PreviewWidget) -> tuple[int, int, int]:
        shot = widget.grabFramebuffer()
        ratio = shot.width() / widget.width()
        # 160 × 90 の絵は (80, 45)-(240, 135) 上の辺の真ん中から少し左（回転の掴み所を避ける）
        color = shot.pixelColor(int(130 * ratio), int(45 * ratio))
        return color.red(), color.green(), color.blue()

    def test_the_selected_clip_has_a_line(self, make_widget: MakeWidget) -> None:
        # 枠が画面に出ないと、どのクリップを選んでいてどこを掴めば動くのか分からない
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        red, _, blue = self._edge_pixel(widget)
        # 素材のファイルは無いので絵は黒 枠の水色だけが出る
        assert blue > 150 and blue > red

    def test_the_setting_off_draws_no_line(self, make_widget: MakeWidget) -> None:
        # 切っても枠が描かれると、設定を切った人のプレビューに要らない線が残る
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        widget = make_widget(project, clip.id)
        widget.set_handles_enabled(False)
        assert max(self._edge_pixel(widget)) < 40


class TestPreferences:
    def test_the_defaults(self) -> None:
        # 既定は入 再生ヘッドの所へ点を打つ（利用者の決定）
        plain = Preferences()
        assert plain.preview_handles is True
        assert plain.keyframe_drag == "playhead"

    def test_they_survive_a_round_trip(self, tmp_path: Path) -> None:
        # 保存した設定が戻らないと、切ったはずの枠が起動のたびにまた出る
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(preview_handles=False, keyframe_drag=KEYFRAME_DRAG_SHIFT_ALL))
        loaded = store.load()
        assert loaded.preview_handles is False
        assert loaded.keyframe_drag == KEYFRAME_DRAG_SHIFT_ALL

    def test_an_unknown_mode_falls_back(self, tmp_path: Path) -> None:
        # 知らない値のまま使うと、どちらの決まりでも無い動きになり、画面の選択肢も空になる
        path = tmp_path / "preferences.json"
        path.write_text('{"keyframe_drag": "どこか", "preview_handles": 1}', encoding="utf-8")
        loaded = PreferenceStore(path).load()
        assert loaded.keyframe_drag == "playhead"
        assert loaded.preview_handles is True

    def test_the_dialog_returns_them(self, qt_application: QApplication) -> None:
        # 画面の値を返し忘れると、OK を押しただけで設定が既定へ戻る
        del qt_application
        from sashimono.ui.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(
            Preferences(preview_handles=False, keyframe_drag=KEYFRAME_DRAG_SHIFT_ALL)
        )
        chosen = dialog.preferences()
        assert chosen.preview_handles is False
        assert chosen.keyframe_drag == KEYFRAME_DRAG_SHIFT_ALL
        # 直接動かさないなら、キーフレームの決まりは効かない
        assert not dialog._keyframe_drag.isEnabled()


class TestTheWindow:
    """窓のつなぎ タイムラインと選択を 1 つにそろえ、離したら 1 段で積む"""

    @pytest.fixture
    def placed(self, qt_application: QApplication) -> Iterator[tuple[MainWindow, Clip]]:
        del qt_application
        media = _media()
        clip = _clip(media)
        project, _ = _project(clip)
        created = MainWindow(project, confirm_unsaved=False)
        yield created, clip
        created.close()

    def test_the_timeline_selection_reaches_the_preview(
        self, placed: tuple[MainWindow, Clip]
    ) -> None:
        # 届かないと、タイムラインで選んだ物と別のクリップ（か何も無い所）に枠が出る
        window, clip = placed
        window._timeline.select(clip.id)
        assert window._preview.selection == clip.id

    def test_a_pick_selects_on_the_timeline(self, placed: tuple[MainWindow, Clip]) -> None:
        # 届かないと、プレビューで選んだクリップと設定パネルが別の物を指す
        window, clip = placed
        window._preview.clip_picked.emit(str(clip.id))
        assert window._timeline.selected_clip == clip.id

    def test_a_finished_drag_is_one_undo_step(self, placed: tuple[MainWindow, Clip]) -> None:
        # 途中の絵まで履歴に積むと、1 回動かしただけで取り消しが何十回も要る
        window, clip = placed
        before = window.document.project
        commands = transform_commands(before, clip.id, {"pos_x": 12.0, "pos_y": 3.0}, 0)
        window._preview.preview_requested.emit(commands)
        # 途中は履歴に積まない
        assert window.document.project == before
        window._preview.commands_requested.emit(commands, "プレビューで位置を変更")
        assert _value(window.document.project, clip.id, "pos_x") == pytest.approx(12.0)
        window.undo()
        assert window.document.project == before

    def test_the_setting_reaches_the_preview(self, placed: tuple[MainWindow, Clip]) -> None:
        # 窓が設定を渡さないと、切っても枠が出続ける
        window, _ = placed
        window._apply_preferences(
            Preferences(preview_handles=False, keyframe_drag=KEYFRAME_DRAG_SHIFT_ALL)
        )
        assert not window._preview.handles_enabled
        assert window._preview._keyframe_drag == KEYFRAME_DRAG_SHIFT_ALL
