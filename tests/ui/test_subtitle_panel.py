"""字幕パネル

パネルは自分ではプロジェクトを書き換えない 出てくるのはコマンドだけなので、
ここでは「どの操作でどのコマンドが出るか」を見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.core.commands import (
    AddClip,
    Command,
    RenameProject,
    RippleCut,
    SetSegmentText,
    SetTranscript,
    SetWorkArea,
    TrimClip,
)
from sashimono.core.model import AnimatedValue, Clip, MediaItem, Project, Transcript
from sashimono.effects.sources import TEXT
from sashimono.engine.audio.waveform import BASE_SAMPLES_PER_PEAK, PeakLevel, Waveform
from sashimono.engine.cache import MediaAnalyzer
from sashimono.ui.export_dialog import RANGE_ALL, RANGE_WORK_AREA
from sashimono.ui.subtitle import SubtitlePanel
from sashimono.ui.subtitle import panel as panel_module
from sashimono.ui.subtitle.dialogs import JetCutDialog
from tests.conftest import make_clip


class StubAnalyzer(MediaAnalyzer):
    """波形を持っているふりをする解析器"""

    def __init__(self, waveform: Waveform | None = None) -> None:
        self._stub = waveform

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None:
        del media, stream
        return self._stub

    def close(self) -> None:
        return None


def make_waveform(pattern: list[tuple[float, int]]) -> Waveform:
    blocks = []
    for amplitude, count in pattern:
        block = np.zeros((count, 1, 2), dtype=np.float32)
        block[:, :, 0] = -amplitude
        block[:, :, 1] = amplitude
        blocks.append(block)
    peaks = np.concatenate(blocks, axis=0)
    return Waveform(
        sample_rate=48000,
        channels=1,
        total_samples=peaks.shape[0] * BASE_SAMPLES_PER_PEAK,
        levels=(PeakLevel(BASE_SAMPLES_PER_PEAK, peaks),),
    )


@pytest.fixture
def placed(project: Project, video_media: MediaItem, transcript: Transcript) -> Project:
    with_transcript = SetTranscript(video_media.id, transcript).apply(project)
    track = with_transcript.timeline.tracks[0]
    return AddClip(track.id, make_clip(0, 300, video_media)).apply(with_transcript)


@pytest.fixture
def panel(
    qt_application: QApplication, placed: Project
) -> Iterator[tuple[SubtitlePanel, list[tuple[list[Command], str]]]]:
    del qt_application
    created = SubtitlePanel(placed, StubAnalyzer())
    issued: list[tuple[list[Command], str]] = []
    created.commands_requested.connect(lambda commands, label: issued.append((commands, label)))
    yield created, issued
    created.deleteLater()


class TestListing:
    def test_only_media_with_audio_is_offered(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], video_media: MediaItem
    ) -> None:
        widget, _ = panel
        assert widget.media_id == video_media.id

    def test_rows_show_timeline_positions_not_source_times(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        widget, _ = panel
        # 素材の 1 秒は 30 フレーム目 30fps なので 00:00:01:00
        assert widget._table.rowCount() == 3
        item = widget._table.item(0, 0)
        assert item is not None
        assert item.text() == "00:00:01:00"

    def test_the_cut_moves_the_shown_times(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], placed: Project
    ) -> None:
        widget, _ = panel
        widget.set_project(RippleCut(((0, 30),)).apply(placed))
        item = widget._table.item(0, 0)
        assert item is not None
        assert item.text() == "00:00:00:00"

    def test_subtitles_outside_the_timeline_are_marked(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], placed: Project
    ) -> None:
        widget, _ = panel
        # クリップを冒頭 2 秒だけにすると、後ろ 2 枚はどこにも出なくなる
        trimmed = placed.timeline.tracks[0].clips[0]
        from dataclasses import replace

        shortened = placed.with_timeline(
            placed.timeline.replace_track(
                placed.timeline.tracks[0].with_clips((replace(trimmed, duration=60),))
            )
        )
        widget.set_project(shortened)
        item = widget._table.item(2, 0)
        assert item is not None
        assert item.text() == "—"


class TestRebuilding:
    """一覧の作り直しは**編集のたび**に通る 字幕が多いと、そこが重さになる"""

    def test_it_skips_when_nothing_changed(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], placed: Project
    ) -> None:
        """中身が前と同じなら作り直さない

        字幕 2000 本で 70ms 掛かる所 字幕に関係のない編集（プロジェクト名を
        変えるなど）でそれを払うのは無駄 作り直していないことは、表の中身が
        同じ物のままかどうかで見る
        """
        widget, _ = panel
        before = widget._table.item(0, 1)
        widget.set_project(RenameProject("別の名前").apply(placed))
        assert widget._table.item(0, 1) is before, "作り直している"

    def test_it_rebuilds_when_the_text_changes(
        self,
        panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]],
        placed: Project,
        video_media: MediaItem,
        transcript: Transcript,
    ) -> None:
        """字幕を直したら作り直す 飛ばすと、直した文字が画面に出ない"""
        widget, _ = panel
        from dataclasses import replace

        first = transcript.segments[0]
        edited = Transcript(
            segments=(replace(first, text="書き直した字幕"), *transcript.segments[1:])
        )
        widget.set_project(SetTranscript(video_media.id, edited).apply(placed))
        item = widget._table.item(0, 1)
        assert item is not None
        assert item.text() == "書き直した字幕"

    def test_the_time_column_fits_a_long_timeline(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], placed: Project
    ) -> None:
        """長いタイムラインでも時刻が切れない

        見本を決め打ちにすると、100 時間を超えた所で時が 3 桁になり、
        2 桁ぶんの幅で切れる
        """
        widget, _ = panel
        narrow = widget._table.horizontalHeader().sectionSize(0)
        clip = placed.timeline.tracks[0].clips[0]
        # 命令で伸ばす モデルを直に差し替えると、命令の側の決まりが変わっても
        # この試験は気付かない
        longer = TrimClip(clip.id, tail_delta=100 * 60 * 60 * 30).apply(placed)
        widget.set_project(longer)
        assert widget._table.horizontalHeader().sectionSize(0) > narrow, "幅が足りない"

    def test_widening_the_time_column_reflows_the_rows(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        """時刻の列が広がったら、行の高さを取り直す

        本文の列は残りを埋める作りなので、時刻が広がるとそのぶん狭くなり、
        折り返しの行数が変わる 取り直さないと 2 行目が隠れて末尾が読めない

        本文の列の合図は、画面に出ていないと飛ばないことがある
        時刻の列（0 番）の合図でも取り直す ここを本文の列だけに絞ると、
        長さが変わったときに折り返しが古いままになる
        """
        widget, _ = panel
        asked: list[bool] = []
        # 本物は画面の大きさを測るので、画面の無い試験では意味のある値にならない
        # 呼ばれたかどうかだけが見たい
        widget._table.resizeRowsToContents = lambda: asked.append(True)  # type: ignore[method-assign]
        try:
            widget._on_section_resized(0, 80, 200)
            assert asked, "時刻の列の合図で取り直していない"

            # 作り直しの最中は走らせない 1 行入れるたびに全部の行を測り直すと、
            # 本数の 2 乗で遅くなる
            asked.clear()
            widget._updating = True
            widget._on_section_resized(0, 200, 240)
            widget._updating = False
            assert not asked, "作り直しの最中に取り直している"
        finally:
            del widget._table.resizeRowsToContents

    def test_an_unrelated_edit_measures_nothing(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], placed: Project
    ) -> None:
        """字幕に関係のない編集では、行の高さを測り直さない

        時刻の幅は毎回入れ直すが、値が同じなら Qt は合図を出さない
        ここが変わると、編集のたびに全部の行を測り直すことになる
        """
        widget, _ = panel
        asked: list[bool] = []
        # 本物は画面の大きさを測るので、画面の無い試験では意味のある値にならない
        widget._table.resizeRowsToContents = lambda: asked.append(True)  # type: ignore[method-assign]
        try:
            widget.set_project(RenameProject("別の名前").apply(placed))
        finally:
            del widget._table.resizeRowsToContents
        assert not asked, "関係のない編集で行を測り直している"

    def test_the_rows_are_not_measured_before_they_are_replaced(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], placed: Project
    ) -> None:
        """これから捨てる行の高さを測り直さない

        時刻の幅を先に変えると、古い行を測り直してから作り直すことになる
        幅が変われば折り返しも変わるので、その測り直しは丸ごと無駄になる
        """
        widget, _ = panel
        order: list[str] = []
        # 呼ばれた順が見たいだけなので、本物は呼ばない（画面の無い試験では
        # 測った高さに意味が無い）
        widget._table.resizeRowsToContents = lambda: order.append("測り直し")  # type: ignore[method-assign]
        original_set = widget._table.setItem

        def spy(row: int, column: int, item: object) -> None:
            order.append("入れ替え")
            original_set(row, column, item)  # type: ignore[arg-type]

        # 入れ替えの起きた時点を知りたい 本物も呼ぶので中身は普通に入る
        widget._table.setItem = spy  # type: ignore[method-assign]
        try:
            clip = placed.timeline.tracks[0].clips[0]
            widget.set_project(TrimClip(clip.id, tail_delta=100 * 60 * 60 * 30).apply(placed))
        finally:
            del widget._table.resizeRowsToContents
            del widget._table.setItem
        assert order, "何も起きていない"
        assert order[0] == "入れ替え", f"入れ替えの前に測り直している: {order[:3]}"
        assert order.count("測り直し") == 1, f"2 度測っている: {order.count('測り直し')} 回"

    def test_the_table_recovers_if_filling_fails(
        self,
        panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]],
        placed: Project,
        video_media: MediaItem,
        transcript: Transcript,
    ) -> None:
        """中身を入れ替える途中で落ちても、表が固まったままにならない

        描き直しを止めたまま戻さないと、以降なにも映らない
        """
        from dataclasses import replace

        widget, _ = panel

        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("わざと落とす")

        # 作り直しが本当に走る変更にする（字幕を 1 本書き直す）
        edited = Transcript(
            segments=(replace(transcript.segments[0], text="別の字幕"), *transcript.segments[1:])
        )
        changed = SetTranscript(video_media.id, edited).apply(placed)
        # 入れ替えの途中でわざと落とす 本物を落とす手立てが他に無い
        widget._table.setItem = boom  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError):
                widget.set_project(changed)
        finally:
            del widget._table.setItem
        assert widget._table.updatesEnabled(), "描き直しが止まったまま"
        assert not widget._updating, "作り直し中の印が残ったまま"

        # 落ちたあとに同じものを渡したら、作り直す
        # 印を先に立てていると素通りして、半端な表が残ったままになる
        widget.set_project(changed)
        item = widget._table.item(0, 1)
        assert item is not None
        assert item.text() == "別の字幕", "半端な表のまま作り直していない"

    def test_the_time_column_does_not_measure_every_row(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        """時刻の列は**固定幅** 中身に合わせると本数の 2 乗で遅くなる

        Qt は幅を中身に合わせるとき、1 行足すたびに全部の行を測り直す
        字幕 2000 本では作り直しに 6 秒掛かっていた
        """
        from PySide6.QtWidgets import QHeaderView

        widget, _ = panel
        header = widget._table.horizontalHeader()
        assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Fixed
        assert header.sectionSize(0) > 0, "幅が 0 だと時刻が読めない"


class TestEditing:
    def test_changing_a_cell_issues_a_command(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], video_media: MediaItem
    ) -> None:
        widget, issued = panel
        item = widget._table.item(1, 1)
        assert item is not None
        item.setText("直した")

        commands, label = issued[-1]
        assert label == "字幕を編集"
        assert isinstance(commands[0], SetSegmentText)
        assert commands[0].text == "直した"
        assert commands[0].media_id == video_media.id

    def test_rebuilding_the_list_does_not_issue_commands(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], placed: Project
    ) -> None:
        # 作り直しのたびに編集コマンドが飛ぶと、履歴が埋まる
        widget, issued = panel
        widget.set_project(placed)
        assert issued == []


class TestPlayheadFollowing:
    def test_the_row_under_the_playhead_is_selected(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        widget, _ = panel
        widget.set_frame(150)  # 5 秒目 2 枚目（4..6 秒）の範囲
        assert widget._table.currentRow() == 1

    def test_clicking_a_row_asks_to_seek(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        widget, _ = panel
        seeks: list[int] = []
        widget.seek_requested.connect(seeks.append)
        widget._table.selectRow(2)
        assert seeks == [210]

    def test_following_the_playhead_does_not_seek_back(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        # 再生ヘッドに追従して行を選んだだけで移動を要求すると、再生が引っ掛かる
        widget, _ = panel
        seeks: list[int] = []
        widget.seek_requested.connect(seeks.append)
        widget.set_frame(150)
        assert seeks == []


class TestJetCut:
    def test_without_a_waveform_it_says_so(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        widget, issued = panel
        messages: list[str] = []
        widget.status_message.connect(messages.append)
        widget.jet_cut()
        assert issued == []
        assert "波形" in messages[0]

    def test_silence_becomes_a_ripple_cut(
        self,
        qt_application: QApplication,
        placed: Project,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del qt_application
        # 先頭 200 ピーク（約 1.07 秒）だけ無音の素材
        analyzer = StubAnalyzer(make_waveform([(0.0, 400), (0.5, 1500)]))
        widget = SubtitlePanel(placed, analyzer)
        issued: list[tuple[list[Command], str]] = []
        widget.commands_requested.connect(lambda commands, label: issued.append((commands, label)))

        monkeypatch.setattr(JetCutDialog, "exec", lambda self: 1)
        widget.jet_cut()

        commands, label = issued[-1]
        assert isinstance(commands[0], RippleCut)
        # 無音は 0..2.13 秒 余白 0.1 秒で内側へ寄り、さらに 1 秒から始まる字幕を
        # 守るので 0.1..0.9 秒 フレームへ落として 3..27 波形が静かでも、
        # 起こせている区間は切らない
        assert commands[0].ranges == ((3, 27),)
        assert "無音カット" in label
        widget.deleteLater()


class TestBurnAndExport:
    def test_burning_places_text_clips(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        widget, issued = panel
        # 焼き込む話し手を尋ねる窓は、全部を選んだものとして返す（窓を開くと試験が止まる）
        widget.ask_burn = lambda voices, note: [voice for voice, _ in voices]
        widget.burn()
        commands, label = issued[-1]
        assert label == "字幕を焼き込み"
        # トラックを 1 本足して、字幕 3 枚をテキストとして置く
        assert len(commands) == 4

    def test_burned_text_is_inside_a_720p_frame(
        self, qt_application: QApplication, placed: Project
    ) -> None:
        """既定の見た目で焼いた字幕が、720p の作品でも画面の中に置かれること

        前は 1080p 用の縦位置（-380）を画素のまま使い、720p の作品では画面の下端
        （-360）より下に置かれて、焼き込んだ字幕がどこにも映らなかった
        """
        del qt_application
        small = replace(placed, settings=replace(placed.settings, width=1280, height=720))
        widget = SubtitlePanel(small, StubAnalyzer())
        issued: list[tuple[list[Command], str]] = []
        widget.commands_requested.connect(lambda commands, label: issued.append((commands, label)))
        widget.ask_burn = lambda voices, note: [voice for voice, _ in voices]
        widget.burn()
        widget.deleteLater()
        commands, _ = issued[-1]
        half = 720 / 2
        clips = [c.clip for c in commands if isinstance(c, AddClip)]
        assert clips
        for clip in clips:
            assert clip.source is not None
            position = clip.source.params["pos_y"]
            assert isinstance(position, AnimatedValue)
            assert -half < position.at(0) < 0

    @pytest.mark.parametrize(("share", "expected"), [(90, 1920 * 0.9), (100, 1920.0), (0, 0.0)])
    def test_burned_text_wraps_at_the_chosen_share(
        self,
        panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]],
        share: int,
        expected: float,
    ) -> None:
        """既定の見た目で焼いた字幕に、設定の割合の折り返しの幅が入る（#249）

        入らないと、長い字幕が画面の端で切れる 設定で切ったら（0）入れない
        """
        widget, issued = panel
        widget.wrap_share = share
        widget.ask_burn = lambda voices, note: [voice for voice, _ in voices]
        widget.burn()
        clips = [c.clip for c in issued[-1][0] if isinstance(c, AddClip)]
        assert clips
        for clip in clips:
            assert clip.source is not None
            width = clip.source.params["wrap_width"]
            assert isinstance(width, AnimatedValue)
            assert width.static == pytest.approx(expected)

    def test_a_template_keeps_its_own_wrap(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]]
    ) -> None:
        # ひな形に選んだテキストの見た目は、折り返しの幅も含めてそのまま写す 設定で上書きすると、
        # 手で決めた幅が焼き込むたびに戻る
        widget, issued = panel
        template = Clip(
            timeline_start=0,
            duration=30,
            source=TEXT.create(text="ひな形", wrap_width=500),
        )
        widget.template_provider = lambda: template
        widget.ask_burn = lambda voices, note: [voice for voice, _ in voices]
        widget.burn()
        clips = [c.clip for c in issued[-1][0] if isinstance(c, AddClip)]
        assert clips
        for clip in clips:
            assert clip.source is not None
            width = clip.source.params["wrap_width"]
            assert isinstance(width, AnimatedValue) and width.static == 500

    def test_export_writes_the_file(
        self,
        panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        widget, _ = panel
        target = tmp_path / "出力.srt"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "")
        )
        messages: list[str] = []
        widget.status_message.connect(messages.append)

        widget.export_file()
        assert target.read_text(encoding="utf-8").startswith("1\n00:00:01,000")
        assert "書き出した" in messages[0]

    def test_cancelling_the_dialog_writes_nothing(
        self,
        panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        widget, _ = panel
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))
        messages: list[str] = []
        widget.status_message.connect(messages.append)
        widget.export_file()
        assert messages == []


class TestExportWithWorkArea:
    """書き出し範囲（#140）を決めたプロジェクトの字幕の書き出し（#141）

    範囲で出した動画は範囲の頭が 0 秒になる 字幕の書き出しが範囲を見ないと、
    動画と字幕が範囲の頭の分だけずれる
    """

    @pytest.fixture
    def ranged(
        self,
        panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]],
        placed: Project,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[SubtitlePanel, Path, list[bool]]:
        widget, _ = panel
        widget.set_project(SetWorkArea((90, 300)).apply(placed))
        target = tmp_path / "出力.srt"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "")
        )
        asked: list[bool] = []
        return widget, target, asked

    def test_choosing_the_range_shifts_the_head_to_zero(
        self, ranged: tuple[SubtitlePanel, Path, list[bool]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget, target, asked = ranged

        def answer(*args: object) -> str:
            asked.append(True)
            return RANGE_WORK_AREA

        monkeypatch.setattr(panel_module, "ask_subtitle_range", answer)
        widget.export_file()
        assert asked == [True]
        # 範囲 [90, 300) の頭の 3 秒を引いて、4 秒の字幕が 1 秒に来る
        assert target.read_text(encoding="utf-8").startswith(
            "1\n00:00:01,000 --> 00:00:03,000\n編集ソフトを\n"
        )

    def test_choosing_the_whole_keeps_timeline_times(
        self, ranged: tuple[SubtitlePanel, Path, list[bool]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget, target, _ = ranged
        monkeypatch.setattr(panel_module, "ask_subtitle_range", lambda *args: RANGE_ALL)
        widget.export_file()
        assert target.read_text(encoding="utf-8").startswith("1\n00:00:01,000 --> 00:00:03,000\n")

    def test_cancelling_the_choice_writes_nothing(
        self, ranged: tuple[SubtitlePanel, Path, list[bool]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget, target, _ = ranged
        monkeypatch.setattr(panel_module, "ask_subtitle_range", lambda *args: None)
        widget.export_file()
        assert not target.exists()

    def test_no_work_area_does_not_ask(
        self,
        panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 範囲が無いのに毎回尋ねると、範囲を使わない人の手間が 1 つ増える
        widget, _ = panel
        target = tmp_path / "出力.srt"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "")
        )

        def refuse(*args: object) -> str:
            raise AssertionError("範囲が無いのに尋ねた")

        monkeypatch.setattr(panel_module, "ask_subtitle_range", refuse)
        widget.export_file()
        assert target.exists()

    def test_the_default_answer_is_the_range(
        self, qt_application: QApplication, placed: Project, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 書き出しダイアログと同じく、範囲を決めた人は範囲を出したくて決めている
        # 既定が全体だと、動画は範囲・字幕は全体で出してずれに気付かない
        del qt_application
        seen: list[str] = []

        def pick_default(box: QMessageBox) -> int:
            default = box.defaultButton()
            seen.append(default.text())
            default.click()
            return 0

        monkeypatch.setattr(QMessageBox, "exec", pick_default)
        ranged = SetWorkArea((90, 300)).apply(placed)
        assert panel_module.ask_subtitle_range(None, ranged) == RANGE_WORK_AREA
        assert "範囲" in seen[0]


class TestSourceTime:
    def test_the_playhead_maps_back_into_the_media(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], video_media: MediaItem
    ) -> None:
        widget, _ = panel
        assert widget._source_time(video_media, 90) == Fraction(3)

    def test_a_frame_outside_every_clip_has_no_source_time(
        self, panel: tuple[SubtitlePanel, list[tuple[list[Command], str]]], video_media: MediaItem
    ) -> None:
        widget, _ = panel
        assert widget._source_time(video_media, 9999) is None


def test_the_panels_jet_cut_of_the_first_voice_leaves_voice_two_alone(
    qt_application: QApplication,
) -> None:
    # AI の jet_cut と同じ穴が画面に無いことを押さえる パネルの見ている音はいつも番号に
    # そろっているが、None のまま渡ると音で絞らず、別の所へ置いた音声 2 の位置でも音声 1 の
    # 無音を切る 最初に開いたときと音声 1 を選んだときで、切る所が同じになる（PR #231）
    from sashimono.engine.audio.silence import SilenceOptions
    from tests.ai.test_operations import _moved_voice_two

    del qt_application
    project = _moved_voice_two()
    media = project.media[0]
    widget = SubtitlePanel(project, StubAnalyzer(make_waveform([(0.0, 400), (0.5, 1500)])))
    try:
        widget.select_media(media.id)
        options = SilenceOptions()
        unchosen = widget._plan(media, options, False)
        widget.select_stream(1)
        chosen = widget._plan(media, options, False)
        assert unchosen == chosen
        assert unchosen and all(end <= 600 for _, end in unchosen)
    finally:
        widget.deleteLater()


class TestDialogsAreFreed:
    """字幕の窓（起こす・整形・無音カット・焼き込み）を、閉じたあとに捨てること

    捨てないと、開くたびに窓が字幕パネルの子として残り続ける 起こすの窓は時計と
    音声の選びを持ち、焼き込みの窓は話し手ごとの印を持つので、開くほど溜まる
    """

    def test_opening_and_closing_does_not_pile_up(
        self, qt_application: QApplication, placed: Project, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del qt_application
        from sashimono.ui.subtitle.dialogs import BurnDialog, CleanupDialog
        from sashimono.ui.subtitle.transcribe_dialog import TranscribeDialog

        # 開く所だけを差し替える 開くと押す人を待って止まる 取り消したことにする
        for dialog in (TranscribeDialog, CleanupDialog, JetCutDialog, BurnDialog):
            monkeypatch.setattr(dialog, "exec", lambda self: 0)
        widget = SubtitlePanel(placed, StubAnalyzer(make_waveform([(0.5, 400)])))
        try:
            for _ in range(3):
                widget.transcribe()
                widget.clean()
                widget.jet_cut()
                widget.burn()
            opened = {type(child).__name__ for child in widget.findChildren(QDialog)}
            QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
            # 窓が本当に開いたこと 開いていなければ、残らないのは当たり前で何も言えない
            assert opened == {"TranscribeDialog", "CleanupDialog", "JetCutDialog", "BurnDialog"}
            assert widget.findChildren(QDialog) == []
        finally:
            widget.deleteLater()
