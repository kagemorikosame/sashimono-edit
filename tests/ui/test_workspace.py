"""編集画面まわりの足回り

トラックヘッダのボタン、保存していない変更の扱い、退避からの復元、
画面配置とショートカットの保存 どれも「無くても編集はできる」が、
無いと作業を失ったり毎回やり直したりするもの
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from sashimono.core.commands import AddClip, Command, RenameProject, SetTrackState
from sashimono.core.io import RecoverySession, backup_folder, find_orphans
from sashimono.core.model import Blending, Clip, Project, ProjectSettings, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.main_window import MainWindow
from sashimono.ui.project_settings_dialog import FRAME_RATE_PRESETS, ProjectSettingsDialog
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.painter import track_button_rects
from sashimono.ui.workspace import ShortcutStore, Workspace, find_conflicts

_PORTABLE = QKeySequence.SequenceFormat.PortableText


def _with_tracks() -> Project:
    base = Project.create(ProjectSettings(width=320, height=240))
    tracks = (Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1"))
    return base.with_timeline(replace(base.timeline, tracks=tracks))


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(_with_tracks(), confirm_unsaved=False)
    yield created
    created.close()


class TestTrackButtons:
    @pytest.fixture
    def view(self, qt_application: QApplication) -> Iterator[TimelineView]:
        del qt_application
        analyzer = MediaAnalyzer(sample_rate=48000, channels=2)
        created = TimelineView(_with_tracks(), analyzer)
        created.resize(800, 300)
        yield created
        analyzer.close()

    def click_button(self, view: TimelineView, track_name: str, attribute: str) -> None:
        band = next(
            b for b in view._layout.bands(view.project.timeline) if b.track.name == track_name
        )
        rect = next(r for name, _, r in track_button_rects(band) if name == attribute)
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=rect.center())

    def test_a_click_asks_for_the_toggle(self, view: TimelineView) -> None:
        # 押してもソロの切り替えが頼まれないと、ボタンを押しても何も起きず、聞き比べができない
        received: list[list[Command]] = []
        view.commands_requested.connect(lambda commands, _label: received.append(commands))
        self.click_button(view, "A1", "solo")

        track = view.project.timeline.tracks[1]
        assert received == [[SetTrackState(track.id, solo=True)]]

    def test_the_playhead_stays_put(self, view: TimelineView) -> None:
        # ミュートを押すたびに見ていた場所が飛ぶと、聞き比べるたびに戻すことになる
        view.set_playhead(40)
        self.click_button(view, "V1", "muted")
        assert view.playhead == 40

    def test_the_rest_of_the_header_leaves_the_playhead(self, view: TimelineView) -> None:
        # ヘッダは時間の軸の外 ここを押して再生ヘッドを動かすと、負のフレームが 0 に
        # 丸められ、名前やボタンの脇を押しただけで先頭へ飛んで見えなくなる（Issue #27）
        view.set_playhead(40)
        band = view._layout.bands(view.project.timeline)[0]
        for point in (QPoint(10, band.top + 10), QPoint(10, 280)):
            QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=point)
            assert view.playhead == 40

    def test_a_near_miss_on_a_button_leaves_the_playhead(self, view: TimelineView) -> None:
        # ボタンの押し損ね（ボタンのすぐ下）でも動かない 押し損ねるたびに見ていた所を
        # 失うと、ボタンを押すのが怖くなる
        view.set_playhead(40)
        band = view._layout.bands(view.project.timeline)[0]
        rect = next(r for name, _, r in track_button_rects(band) if name == "muted")
        QTest.mouseClick(
            view, Qt.MouseButton.LeftButton, pos=QPoint(rect.center().x(), rect.bottom() + 4)
        )
        assert view.playhead == 40

    def test_the_ruler_still_scrubs(self, view: TimelineView) -> None:
        # ヘッダを止めたせいで目盛りまで効かなくなると、再生ヘッドをマウスで動かせない
        view.set_playhead(40)
        x = int(view._layout.frame_to_x(100))
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(x, 10))
        assert view.playhead == 100

    def test_the_keyboard_can_toggle_the_selected_track(self, view: TimelineView) -> None:
        # ボタンは描いた矩形なのでフォーカスが来ない マウスを使えない人が
        # ミュートできなくならないよう、選んだクリップのトラックをキーで切り替える
        track = view.project.timeline.tracks[0]
        clip = Clip(timeline_start=0, duration=30, source=TEXT.create())
        view.set_project(AddClip(track.id, clip).apply(view.project))
        view.select(clip.id)
        received: list[list[Command]] = []
        view.commands_requested.connect(lambda commands, _label: received.append(commands))

        assert view.toggle_selected_track("muted")
        assert received == [[SetTrackState(track.id, muted=True)]]

    def test_the_keyboard_toggle_needs_a_selection(self, view: TimelineView) -> None:
        # どのトラックか分からないまま切り替えると、見えていないトラックが消音される
        assert not view.toggle_selected_track("solo")

    def test_buttons_fit_in_the_smallest_track(self, view: TimelineView) -> None:
        # 名前の下の段に置くと、最小の高さでボタンがはみ出して押せない
        band = view._layout.bands(view.project.timeline)[0]
        small = replace(band, height=28)
        assert all(rect.bottom() < small.bottom for _, _, rect in track_button_rects(small))


class TestUnsavedChanges:
    def test_a_fresh_window_is_clean(self, window: MainWindow) -> None:
        # 開いたばかりで変更ありになると、何も触らずに閉じても保存を尋ねられ、題に * が付く
        assert not window.is_modified
        assert "*" not in window.windowTitle()

    def test_an_edit_marks_it_and_undo_clears_it(self, window: MainWindow) -> None:
        # 印が付かないと、保存を尋ねずに閉じて編集を失う 取り消しで外れないと、
        # 元どおりなのに保存を尋ねられる
        window.execute(RenameProject("別名"))
        assert window.is_modified
        assert "*" in window.windowTitle()
        window.undo()
        assert not window.is_modified

    def test_autosave_writes_only_when_changed(self, window: MainWindow) -> None:
        # 変えたのに書かないと、落ちたときに作業を失う 変えていないのに書くと、
        # 何も失っていないのに次の起動で復元を勧められる
        window.autosave()
        assert not window._recovery.path.exists()
        window.execute(RenameProject("別名"))
        window.autosave()
        assert window._recovery.path.exists()

    def test_saving_clears_the_autosave_and_backs_up(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        # 保存しても退避が残ると、次の起動で古い作業の復元を勧められる 前の版を控えないと、
        # 上書きで壊したときに戻れない
        window._path = tmp_path / "本編.sme"
        window.execute(RenameProject("一回目"))
        window.autosave()
        assert window.save_project()
        assert not window._recovery.path.exists()
        assert not window.is_modified

        window.execute(RenameProject("二回目"))
        assert window.save_project()
        assert len(list(backup_folder(window._path).iterdir())) == 1

    def test_a_live_window_is_never_offered_for_recovery(self, window: MainWindow) -> None:
        # 動いている窓の退避を拾うと、もう 1 つ窓を開いたときに、今編集している作業を
        # 復元するよう勧めて同じ作品が 2 つに分かれる
        window.execute(RenameProject("作業中"))
        window.autosave()
        assert find_orphans() == []


class TestRestoring:
    def test_a_crashed_session_comes_back(self, window: MainWindow, tmp_path: Path) -> None:
        # 落ちたあとの退避から戻せないと、最後に保存してからの作業がすべて失われる
        source = tmp_path / "本編.sme"
        crashed = RecoverySession()
        crashed.save(_with_tracks().renamed("落ちた作業"), source)
        # 落ちたことにする 錠は片付けずに手放す（プロセスが消えたときと同じ）
        lock = crashed._lock
        assert lock is not None
        lock.abandon()
        crashed._lock = None

        (entry,) = find_orphans()
        assert window.restore_recovery(entry)

        assert window.document.project.name == "落ちた作業"
        assert window._path == source
        # 開いただけで保存はしていない 閉じるときに尋ねなければ、復元した意味が無い
        assert window.is_modified
        # 自分の退避へ書き写したので、元の退避は消えている
        assert find_orphans() == []
        assert window._recovery.path.exists()


class TestLayout:
    def test_the_layout_is_saved_on_close(self, qt_application: QApplication) -> None:
        # 閉じるときに残さないと、パネルを並べ直しても次の起動で毎回元の配置へ戻る
        del qt_application
        first = MainWindow(confirm_unsaved=False)
        first.close()
        assert Workspace().path.is_file()

        second = MainWindow(confirm_unsaved=False)
        try:
            assert Workspace().restore(second)
        finally:
            second.close()

    def test_every_panel_has_a_name(self, window: MainWindow) -> None:
        # 名前の無いパネルは、Qt が黙って保存から外す
        from PySide6.QtWidgets import QDockWidget

        assert all(dock.objectName() for dock in window.findChildren(QDockWidget))

    def test_reset_hides_the_graph_editor_again(self, window: MainWindow) -> None:
        # 初期配置に戻してもグラフエディタが出たままだと、戻したのにタイムラインが狭いまま
        window._graph_dock.setVisible(True)
        window.reset_layout()
        assert window._graph_dock.isHidden()


class TestShortcuts:
    def test_an_override_is_applied_at_start(self, qt_application: QApplication) -> None:
        # 変えたキーが起動のときに効かないと、毎回割り当て直すことになる 版が変わって
        # 消えた項目の名前が残っていても、起動は止めない
        del qt_application
        ShortcutStore().save({"ファイル/保存": "Ctrl+Alt+S", "消えた/項目": "F9"})
        created = MainWindow(confirm_unsaved=False)
        try:
            action, default = created._actions["ファイル/保存"]
            assert action.shortcut().toString(_PORTABLE) == "Ctrl+Alt+S"
            assert default == "Ctrl+S"
        finally:
            created.close()

    def test_a_broken_file_means_defaults(self, tmp_path: Path) -> None:
        # ショートカットのファイルが壊れただけで起動が止まると、既定のキーで使うこともできない
        path = tmp_path / "shortcuts.json"
        path.write_text("{壊れている", "utf-8")
        assert ShortcutStore(path).load() == {}

    def test_conflicts_are_found(self) -> None:
        # 重なりを見つけないと、同じキーを 2 つに割り当てても気づかず、押してもどちらも動かない
        bindings = {"a": "Ctrl+S", "b": "Ctrl+S", "c": "", "d": ""}
        # 空は「割り当てなし」 いくつあっても重なりではない
        assert find_conflicts(bindings) == {"Ctrl+S": ["a", "b"]}

    def test_names_do_not_follow_the_changing_text(self, window: MainWindow) -> None:
        # 「元に戻す: 分割」のように表示が変わっても、名前は作ったときのまま
        window.execute(RenameProject("別名"))
        assert "編集/元に戻す" in window._actions


class TestSettingsDialog:
    def test_the_preset_follows_the_numbers(self, qt_application: QApplication) -> None:
        # 縦横を入れ替えても一覧が前の名前のままだと、縦動画のつもりが表示と数が食い違って迷う
        del qt_application
        dialog = ProjectSettingsDialog(ProjectSettings())
        dialog._swap()
        assert dialog.resolution() == (1080, 1920)
        assert "縦" in dialog._preset.currentText()

    def test_a_new_project_can_choose_its_frame_rate(self, qt_application: QApplication) -> None:
        # 選べないと、60fps のゲーム実況も 30fps で作るしかなく、動きがカクつく
        del qt_application
        dialog = ProjectSettingsDialog(ProjectSettings(), new=True)
        assert dialog._rate is not None
        dialog._rate.setCurrentIndex(
            next(i for i, (_, r) in enumerate(FRAME_RATE_PRESETS) if r == FrameRate(60))
        )
        assert dialog.settings().frame_rate == FrameRate(60)

    def test_ntsc_rates_stay_exact(self, qt_application: QApplication) -> None:
        # 29.97 を小数で持つと、1 時間で 3 フレーム以上ずれる
        del qt_application
        rates = [rate for _, rate in FRAME_RATE_PRESETS]
        assert FrameRate(30000, 1001) in rates

    def test_an_existing_project_cannot_change_its_frame_rate(
        self, qt_application: QApplication
    ) -> None:
        # 変えられると、フレーム番号で持っている全クリップの時刻がずれる
        del qt_application
        dialog = ProjectSettingsDialog(ProjectSettings(frame_rate=FrameRate(24)))
        assert dialog._rate is None
        assert dialog.settings().frame_rate == FrameRate(24)

    def test_a_new_project_blends_like_aviutl(self, qt_application: QApplication) -> None:
        # 新規作成の既定が AviUtl・YMM4 と違うと、読み込んだ素材の半透明が明るく浮く
        del qt_application
        dialog = ProjectSettingsDialog(ProjectSettings(), new=True)
        assert dialog.settings().blending == Blending.SRGB

    def test_the_dialog_keeps_an_old_projects_blending(self, qt_application: QApplication) -> None:
        # 開いただけで sRGB に戻ると、解像度だけ変えたつもりの OK で見た目まで変わる
        del qt_application
        dialog = ProjectSettingsDialog(ProjectSettings(blending=Blending.LINEAR))
        assert dialog.settings().blending == Blending.LINEAR

    def test_the_chosen_blending_is_applied_and_undone_at_once(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """設定画面で解像度と重ね合わせを変え、1 回の取り消しで両方戻る

        別々に積むと、取り消しの途中で見たことのない組み合わせを通る 選んだ値が
        コマンドへ届かないと、画面で選んでも絵が変わらない
        """

        def choose(dialog: ProjectSettingsDialog) -> int:
            dialog._width.setValue(640)
            dialog._blending.setCurrentIndex(dialog._blending.findData(Blending.LINEAR))
            return int(QDialog.DialogCode.Accepted)

        monkeypatch.setattr(ProjectSettingsDialog, "exec", choose)
        before = window.document.project.settings
        window.edit_settings()
        after = window.document.project.settings
        assert (after.width, after.blending) == (640, Blending.LINEAR)
        window.undo()
        assert window.document.project.settings == before

    def test_odd_numbers_cannot_be_confirmed(self, qt_application: QApplication) -> None:
        # 奇数の幅を通すと、h.264 は幅も高さも偶数でないと符号化できないので、
        # 編集を終えてから書き出しで失敗する
        del qt_application
        dialog = ProjectSettingsDialog(ProjectSettings())
        dialog._width.setValue(1919)
        ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert ok is not None
        assert not ok.isEnabled()
