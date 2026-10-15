"""素材の読み込みで画面を固めないことと、裏の処理の進み具合を画面へ出すこと

窓は表示しない（表示すると GL の初期化が走り、画面が点滅する） 素材を調べる所は
差し替えて、調べている最中の状態を狙って作る 本物の素材では、調べ終わる前に
確かめられるかどうかが機械の速さ次第になる
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog

from sashimono.core.commands import AddMedia, insert_media
from sashimono.core.model import MediaId, MediaItem, Project, ProjectSettings, new_media_id
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache.progress import JobBoard, ProgressSnapshot
from sashimono.engine.decode import ProbeError
from sashimono.ui import main_window as main_window_module
from sashimono.ui.main_window import MainWindow
from sashimono.ui.progress_display import describe_background, row_notes
from sashimono.ui.workspace import Preferences, PreferenceStore


@pytest.fixture
def window(qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30))),
        confirm_unsaved=False,
    )
    # 置いた素材は本物のファイルではない 解析と控えを頼むと、裏で「開けない」が数えられて
    # この試験が見たい読み込みの表示に混ざる 進み具合の表示は別の試験で数を直に入れて見る
    monkeypatch.setattr(created, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(created._analyzer, "request", lambda _media, **_kwargs: None)
    yield created
    created.close()


class _Gate:
    """素材を調べる所の差し替え 開けるまで調べ終わらない

    調べた所のスレッドを覚える 画面のスレッドで調べていたら、それが固まる原因
    """

    def __init__(self, template: MediaItem, broken: tuple[str, ...] = ()) -> None:
        self.template = template
        self.broken = broken
        self.release = threading.Event()
        self.threads: list[threading.Thread] = []
        self.started = threading.Event()

    def __call__(self, path: Path) -> MediaItem:
        self.threads.append(threading.current_thread())
        self.started.set()
        assert self.release.wait(20.0), "試験が開け忘れた"
        if path.name in self.broken:
            raise ProbeError(f"素材を開けない: {path}")
        # 素材ごとに別の id にする 同じ id だと 2 本目が 1 本目と同じ素材として扱われる
        return replace(self.template, path=path, id=new_media_id())


def _names(window: MainWindow) -> list[str]:
    """一覧に入った素材のファイル名 置いた順"""
    return [media.path.name for media in window.document.project.media]


def _new_project(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    """〔新規〕を押して、尋ねる画面で OK を押した所まで 画面は出さない"""
    from sashimono.ui.project_settings_dialog import ProjectSettingsDialog

    monkeypatch.setattr(
        ProjectSettingsDialog, "exec", lambda _self: QDialog.DialogCode.Accepted.value
    )
    window.new_project()


def _spin(application: QApplication, until: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if until():
            return True
        time.sleep(0.005)
    return False


class TestImportDoesNotFreeze:
    def test_it_returns_before_the_files_are_read(
        self,
        window: MainWindow,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        qt_application: QApplication,
    ) -> None:
        # 前は素材を 1 本ずつ画面のスレッドで調べ、調べ終わるまで返らなかった
        # （1080p の mp4 で 1 本 55ms 24 本で 1.4 秒固まる） ここで待たされる形に
        # 戻ると、門が開かないまま 20 秒待って試験が落ちる
        gate = _Gate(video_media)
        monkeypatch.setattr(main_window_module, "probe_media", gate)

        started = time.monotonic()
        window.import_media([tmp_path / "a.mp4", tmp_path / "b.mp4"])
        assert time.monotonic() - started < 1.0
        assert window.importing
        assert gate.started.wait(5.0)
        assert all(thread is not threading.main_thread() for thread in gate.threads)

        # 調べている間も画面のイベントが回る（タイマーが鳴る）
        ticked: list[bool] = []
        QTimer.singleShot(0, lambda: ticked.append(True))
        assert _spin(qt_application, lambda: bool(ticked))
        assert _names(window) == []

        gate.release.set()
        assert window.wait_for_imports()
        assert _names(window) == ["a.mp4", "b.mp4"]

    def test_the_progress_is_shown_while_reading(
        self,
        window: MainWindow,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        qt_application: QApplication,
    ) -> None:
        # 読み込み中の表示が出ないと、調べている間に画面は動くのに何が起きているか分からない
        gate = _Gate(video_media)
        monkeypatch.setattr(main_window_module, "probe_media", gate)
        window.import_media([tmp_path / "a.mp4", tmp_path / "b.mp4"])
        try:
            indicator = window._import_indicator
            assert _spin(qt_application, lambda: not indicator.isHidden())
            assert indicator.text == "素材を調べている 0/2 本"
            assert indicator.cancel_button is not None
        finally:
            gate.release.set()
        assert window.wait_for_imports()
        # 終わったら消す 出したままだと、終わったのか止まったのか分からない
        assert window._import_indicator.isHidden()
        assert window.statusBar().currentMessage() == "2 件を読み込んだ"

    def test_cancel_places_nothing(
        self,
        window: MainWindow,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 取り消した後に調べ終わった素材を置くと、取り消したはずの物が
        # タイムラインに入り、履歴にも残る
        gate = _Gate(video_media)
        monkeypatch.setattr(main_window_module, "probe_media", gate)
        window.import_media([tmp_path / "a.mp4", tmp_path / "b.mp4"])
        window.import_media([tmp_path / "c.mp4"])
        assert gate.started.wait(5.0)

        cancel = window._import_indicator.cancel_button
        assert cancel is not None
        cancel.click()
        gate.release.set()
        assert not window.importing
        assert window.wait_for_imports()
        # 取り消した後に調べ終わった分を置きに来ると、取り消しが効いていない
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
        assert _names(window) == []
        assert window.document.history_labels == ()
        assert window._import_indicator.isHidden()
        assert window.statusBar().currentMessage() == "素材の読み込みを取り消した"

    def test_one_undo_takes_the_whole_import_back(
        self,
        window: MainWindow,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 裏へ移しても 1 回の操作にまとめる 1 本ずつ置くと、10 本読み込んで 10 回取り消すことになる
        gate = _Gate(video_media)
        gate.release.set()
        monkeypatch.setattr(main_window_module, "probe_media", gate)
        window.import_media([tmp_path / f"{index}.mp4" for index in range(5)])
        assert window.wait_for_imports()
        assert len(window.document.project.media) == 5
        assert len(window.document.history_labels) == 1

        window.undo()
        assert _names(window) == []
        assert window.document.project.timeline.tracks == () or all(
            not track.clips for track in window.document.project.timeline.tracks
        )

    def test_a_second_import_waits_for_the_first(
        self,
        window: MainWindow,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        qt_application: QApplication,
    ) -> None:
        # 並べて走らせると、後から頼んだ方が先に置かれ、置く順が頼んだ順と食い違う
        gate = _Gate(video_media)
        monkeypatch.setattr(main_window_module, "probe_media", gate)
        window.import_media([tmp_path / "先.mp4"])
        window.import_media([tmp_path / "後.mp4"])
        assert _spin(qt_application, lambda: "ほかに 1 回分" in window._import_indicator.text)
        gate.release.set()
        assert window.wait_for_imports()
        assert _names(window) == ["先.mp4", "後.mp4"]
        # 頼んだ回ごとに 1 手 まとめて 1 手にすると、後の分だけ戻せない
        assert len(window.document.history_labels) == 2

    def test_a_broken_file_is_reported_and_the_rest_placed(
        self,
        window: MainWindow,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 1 本開けないだけで全部を止めると、良い素材まで置かれない
        # 開けない素材を黙って落とすと理由が分からない
        gate = _Gate(video_media, broken=("壊れた.mp4",))
        gate.release.set()
        monkeypatch.setattr(main_window_module, "probe_media", gate)
        window.import_media([tmp_path / "壊れた.mp4", tmp_path / "良い.mp4"])
        assert window.wait_for_imports()
        assert _names(window) == ["良い.mp4"]
        assert "壊れた.mp4" in window.statusBar().currentMessage()

    def test_a_drop_on_the_pool_goes_the_same_way(
        self,
        window: MainWindow,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 一覧への落とし込みとボタンは import_requested を通る 別の道で読むと、
        # そちらだけ画面が固まったまま残る
        gate = _Gate(video_media)
        monkeypatch.setattr(main_window_module, "probe_media", gate)
        window._media_pool.import_requested.emit([tmp_path / "落とした.mp4"])
        assert window.importing
        gate.release.set()
        assert window.wait_for_imports()
        assert _names(window) == ["落とした.mp4"]

    def test_a_new_project_drops_the_import(
        self,
        window: MainWindow,
        video_media: MediaItem,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 読み込みは始めたときのプロジェクトへ置く約束 捨てずに置くと、調べ終わった
        # 素材が新しいプロジェクトに入り、その取り消しの履歴にまで載る
        gate = _Gate(video_media)
        monkeypatch.setattr(main_window_module, "probe_media", gate)
        window.import_media([tmp_path / "前の.mp4"])
        window.import_media([tmp_path / "待ち.mp4"])
        assert gate.started.wait(5.0)

        _new_project(window, monkeypatch)
        gate.release.set()
        assert window.wait_for_imports()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            QApplication.processEvents()
            window._poll_import()
            time.sleep(0.01)
        assert _names(window) == []
        assert window.document.history_labels == ()
        assert window._import_indicator.isHidden()
        assert "読み込みを取り消した" in window.statusBar().currentMessage()


class TestBackgroundProgress:
    """控えと解析の進み具合 数は裏の係の数え板へ直に入れる

    本物の変換は機械の速さで終わる時刻が変わり、途中の割合を狙って見られない
    """

    def _place(self, window: MainWindow, media: MediaItem) -> None:
        window.execute_all(insert_media(window.view_project, media), "置く")

    def test_the_status_bar_and_the_row_show_it(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # ステータスバーと行に割合が出ないと、どの素材の解析を待っているのか分からない
        self._place(window, video_media)
        board = window._analyzer._board
        board.start(("waveform", video_media.id), video_media.id)
        board.start(("filmstrip", video_media.id), video_media.id)
        board.report(("waveform", video_media.id), 1.0)

        window._flush_analysis()
        indicator = window._background_indicator
        assert not indicator.isHidden()
        assert indicator.text == "解析 0/2 件 50%"
        assert indicator.value == 500
        row = window._media_pool.row_text(video_media.id)
        assert row is not None and row.endswith("[解析 50%]")

    def test_the_end_and_the_failure_are_told(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # 終わりと失敗を知らせないと、波形が出ない理由が分からないまま待ち続ける
        self._place(window, video_media)
        board = window._analyzer._board
        key = ("waveform", video_media.id)
        board.start(key, video_media.id)
        window._flush_analysis()
        board.finish(key, "波形を作れなかった: 素材を開けない")
        window._flush_analysis()

        assert window._background_indicator.isHidden()
        message = window.statusBar().currentMessage()
        assert "失敗 1 件" in message
        assert "素材を開けない" in message
        row = window._media_pool.row_text(video_media.id)
        assert row is not None and row.endswith("[解析できなかった]")

    def test_a_quiet_finish_says_nothing(self, window: MainWindow, video_media: MediaItem) -> None:
        # キャッシュから一瞬で済んだ分まで「終わった」と出すと、開くたびに文言が出る
        board = window._analyzer._board
        key = ("waveform", video_media.id)
        board.start(key, video_media.id)
        board.finish(key)
        window.statusBar().clearMessage()
        window._flush_analysis()
        assert window.statusBar().currentMessage() == ""

    def test_the_proxy_is_counted_apart(self, window: MainWindow, video_media: MediaItem) -> None:
        # 読み込んだ直後にプレビューが重い理由が、控えを作っている最中だからなのかを見て分かるように
        board = window._proxies._board
        board.start(video_media.id, video_media.id)
        board.report(video_media.id, 0.4)
        window._flush_analysis()
        assert window._background_indicator.text == "控え 0/1 本 40%"

    def test_turning_the_rows_off_leaves_the_status_bar(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # 切っても行に割合が残るなら、設定がある方が質が悪い
        self._place(window, video_media)
        window._apply_preferences(Preferences(pool_progress=False))
        board = window._analyzer._board
        board.start(("waveform", video_media.id), video_media.id)
        board.report(("waveform", video_media.id), 0.5)
        window._flush_analysis()
        row = window._media_pool.row_text(video_media.id)
        assert row is not None and "[" not in row
        assert not window._background_indicator.isHidden()

    def test_turning_the_rows_off_clears_them_at_once(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # 切っても次の更新まで行の注記が残ると、切ったのに効いていないように見える
        self._place(window, video_media)
        board = window._analyzer._board
        board.start(("waveform", video_media.id), video_media.id)
        window._flush_analysis()
        window._apply_preferences(Preferences(pool_progress=False))
        row = window._media_pool.row_text(video_media.id)
        assert row is not None and "[" not in row

    def test_a_proxy_failure_survives_while_the_analysis_runs(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # 控えが止まった時点で控えの数だけ数え直すと、解析が終わったときの知らせに
        # 控えの失敗が出ず、全体の割合も 50% から 0% へ巻き戻る
        self._place(window, video_media)
        proxy = window._proxies._board
        analysis = window._analyzer._board
        proxy.start(video_media.id, video_media.id)
        analysis.start(("waveform", video_media.id), video_media.id)
        window._flush_analysis()
        proxy.finish(video_media.id, "控えを作れなかった: 符号化器が無い")
        window._flush_analysis()
        window._flush_analysis()
        indicator = window._background_indicator
        assert indicator.text == "解析 0/1 件 0% 失敗 1 件"
        assert indicator.value == 500

        analysis.finish(("waveform", video_media.id))
        window._flush_analysis()
        message = window.statusBar().currentMessage()
        assert "失敗 1 件" in message
        assert "符号化器が無い" in message
        # 知らせた後は数え直す 次のひと続きが前の失敗を抱えたまま始まらない
        assert window._proxies.poll().total == 0
        assert window._analyzer.poll().total == 0

    def test_an_analysis_failure_survives_while_the_proxy_runs(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # 4K の控えは数分続く その間に波形が失敗して解析の数だけ数え直すと、失敗は
        # 250ms しか出ず、控えが終わったときの知らせは「終わった」だけになる
        self._place(window, video_media)
        proxy = window._proxies._board
        analysis = window._analyzer._board
        proxy.start(video_media.id, video_media.id)
        analysis.start(("waveform", video_media.id), video_media.id)
        window._flush_analysis()
        analysis.finish(("waveform", video_media.id), "波形を作れなかった: 音が読めない")
        for _ in range(3):
            window._flush_analysis()
        assert window._background_indicator.text == "控え 0/1 本 0% 失敗 1 件"

        proxy.finish(video_media.id)
        window._flush_analysis()
        message = window.statusBar().currentMessage()
        assert "失敗 1 件" in message
        assert "音が読めない" in message

    def test_the_notice_names_this_run_s_failure(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # 行に残っている前の失敗（素材 A）から理由を取ると、今失敗した素材 B の
        # 理由の代わりに A の理由が出る
        board = window._analyzer._board
        first, second = MediaId("A"), MediaId("B")
        board.start(("waveform", first), first)
        window._flush_analysis()
        board.finish(("waveform", first), "A の波形を作れなかった")
        window._flush_analysis()
        board.start(("filmstrip", second), second)
        window._flush_analysis()
        board.finish(("filmstrip", second), "B のサムネイルを作れなかった")
        window._flush_analysis()
        message = window.statusBar().currentMessage()
        assert "失敗 1 件" in message
        assert "B のサムネイルを作れなかった" in message
        assert "A の" not in message

    def test_removing_a_failed_media_takes_its_failure_back(
        self, window: MainWindow, video_media: MediaItem
    ) -> None:
        # 理由だけ消えて数が残ると、「失敗 1 件」と出るのに何が失敗したのか分からない
        # 一覧に入れるだけにする タイムラインで使っている素材は外せない
        assert window.execute(AddMedia(video_media))
        board = window._analyzer._board
        board.start(("waveform", video_media.id), video_media.id)
        board.start(("filmstrip", video_media.id), video_media.id)
        window._flush_analysis()
        board.finish(("waveform", video_media.id), "波形を作れなかった")
        window._remove_media(str(video_media.id))
        assert window.document.project.find_media(video_media.id) is None
        snapshot = window._analyzer.poll()
        assert (snapshot.total, snapshot.failed) == (0, 0)

    def test_a_new_project_does_not_show_the_old_progress(
        self, window: MainWindow, video_media: MediaItem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 前のプロジェクトの素材の本数と失敗が、新しいプロジェクトのステータスバーに出続けない
        self._place(window, video_media)
        board = window._analyzer._board
        board.start(("waveform", video_media.id), video_media.id)
        board.start(("filmstrip", video_media.id), video_media.id)
        board.finish(("filmstrip", video_media.id), "サムネイルを作れなかった")
        window._flush_analysis()
        assert not window._background_indicator.isHidden()

        _new_project(window, monkeypatch)
        window.statusBar().clearMessage()
        window._flush_analysis()
        snapshot = window._analyzer.poll()
        assert (snapshot.total, snapshot.failed) == (0, 0)
        assert snapshot.failures == {}
        assert window._background_indicator.isHidden()
        assert window.statusBar().currentMessage() == ""


class TestWording:
    def test_both_kinds_are_listed(self) -> None:
        # 控えと解析の片方しか出さないと、もう片方を待っている間の重さの理由が分からない
        media = MediaId("m")
        proxy, analysis = JobBoard(), JobBoard()
        proxy.start("p", media)
        analysis.start("a", media)
        analysis.finish("a", "開けない")
        analysis.start("b", media)
        text = describe_background(proxy.poll(), analysis.poll())
        assert text == "控え 0/1 本 0%・解析 1/2 件 50% 失敗 1 件"

    def test_a_row_without_work_has_no_note(self) -> None:
        # 仕事の無い行に注記を付けると、終わった素材まで何かを作っているように見える
        assert row_notes(ProgressSnapshot(), ProgressSnapshot()) == {}


class TestPreference:
    def test_it_is_on_by_default(self) -> None:
        # 既定は知らない人が困らない側 どの素材の控えを作っているのかが見える側
        assert Preferences().pool_progress is True

    def test_it_comes_back(self, tmp_path: Path) -> None:
        # 保存して読み直せないと、行の注記を切った人が起動のたびに切り直すことになる
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(pool_progress=False))
        assert store.load().pool_progress is False

    def test_a_broken_value_falls_back(self, tmp_path: Path) -> None:
        # 壊れた値で既定へ戻らないと、設定ファイルが壊れただけで起動に失敗する
        path = tmp_path / "preferences.json"
        path.write_text('{"pool_progress": "off"}', encoding="utf-8")
        assert PreferenceStore(path).load().pool_progress is True

    def test_the_dialog_shows_and_returns_it(self, qt_application: QApplication) -> None:
        # 設定画面が値を出し入れしないと、画面で切り替えても効かない
        del qt_application
        from sashimono.ui.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(Preferences(pool_progress=False))
        assert dialog.preferences().pool_progress is False
        dialog._pool_progress.setChecked(True)
        assert dialog.preferences().pool_progress is True
