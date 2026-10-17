"""P1 の完了条件

実素材で「読込 → カット → 並べ替え → 保存・読み直し → 書き出し」が一本通ること
UI を組み立てて、実際に使う経路をそのまま辿る

ウィンドウは表示しない 表示すると GL の初期化が走ってプレビューまで動くが、
テストのたびに画面が点滅するのは邪魔なので、合成そのものは
``tests/engine/test_render.py`` に任せる
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.io import load_project
from sashimono.core.model import Project, ProjectSettings, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.engine.encode import ExportSettings, available_video_codecs, export_project
from sashimono.ui.main_window import MainWindow
from tests.media_fixtures import SampleMedia, make_sample


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30))),
        confirm_unsaved=False,
    )
    yield created
    created.close()


@pytest.fixture(scope="session")
def two_clips(media_dir: Path) -> tuple[SampleMedia, SampleMedia]:
    first = make_sample(media_dir, "flow_a.mp4", width=320, height=240, duration=3.0)
    second = make_sample(media_dir, "flow_b.mp4", width=320, height=240, duration=2.0, tone_hz=660)
    return first, second


def _clip_counts(project: Project) -> list[int]:
    return [len(track.clips) for track in project.timeline.tracks]


class TestEditingFlow:
    def test_import_creates_linked_video_and_audio(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia]
    ) -> None:
        # 映像と音声がつながっていないと、片方だけ動かしたり消したりして口と声がずれる
        window.import_media([two_clips[0].path])
        assert window.wait_for_imports()
        project = window._document.project

        assert len(project.media) == 1
        kinds = [track.kind for track in project.timeline.tracks]
        assert TrackKind.VIDEO in kinds
        assert TrackKind.AUDIO in kinds

        video_clip = next(iter(project.timeline.video_tracks())).clips[0]
        audio_clip = next(iter(project.timeline.audio_tracks())).clips[0]
        assert video_clip.link_group is not None
        assert video_clip.link_group == audio_clip.link_group

    def test_import_is_a_single_undo_step(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia]
    ) -> None:
        # 2 本読み込んで 2 回取り消す、という操作は誰も望まない
        window.import_media([two_clips[0].path, two_clips[1].path])
        assert window.wait_for_imports()
        assert len(window._document.history_labels) == 1

        window.undo()
        assert window._document.project.media == ()

    def test_clips_are_placed_end_to_end(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia]
    ) -> None:
        # 隙間や重なりができると、読み込んだ直後から黒い間が入るか、後の素材が前を隠す
        window.import_media([two_clips[0].path, two_clips[1].path])
        assert window.wait_for_imports()
        project = window._document.project
        video_track = next(iter(project.timeline.video_tracks()))
        starts = [clip.timeline_start for clip in video_track.clips]
        assert starts == sorted(starts)
        assert starts[1] == video_track.clips[0].timeline_end

    def test_split_then_ripple_delete_shortens_the_timeline(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia]
    ) -> None:
        # 詰める削除が効かないと、切った頭を消しても空白が残り手で詰め直すことになる
        window.import_media([two_clips[0].path, two_clips[1].path])
        assert window.wait_for_imports()
        before = window._document.project.duration

        window._seek(45)
        window._timeline.split_at_playhead()
        assert _clip_counts(window._document.project) == [3, 3]

        head = next(iter(window._document.project.timeline.video_tracks())).clips[0]
        window._timeline.select(head.id)
        window._timeline.delete_selected(ripple=True)

        project = window._document.project
        # 映像と音声がリンクしているので、両方が同時に消えて詰まる
        assert _clip_counts(project) == [2, 2]
        assert project.duration == before - 45

    def test_undo_restores_everything(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia]
    ) -> None:
        # 取り消しで元に戻り切らないと、試しに切ったつもりの編集が戻せず作品が壊れたままになる
        window.import_media([two_clips[0].path])
        assert window.wait_for_imports()
        original = window._document.project

        window._seek(30)
        window._timeline.split_at_playhead()
        assert len(window._document.history_labels) == 2, "分割が 1 手になっていない"

        head = next(iter(window._document.project.timeline.video_tracks())).clips[0]
        window._timeline.select(head.id)
        window._timeline.delete_selected(ripple=True)

        # 分割は映像・音声をまとめて 1 手 取り消しも 1 回で済む
        window.undo()  # 削除
        window.undo()  # 分割
        assert window._document.project == original

    def test_playhead_is_clamped_to_the_timeline(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia]
    ) -> None:
        # 再生位置が終わりを越えると、何も無い所を指したまま切る・置くの操作が空振りする
        window.import_media([two_clips[0].path])
        assert window.wait_for_imports()
        window._seek(10**6)
        assert window._timeline.playhead == window._document.project.duration

    def test_selection_is_cleared_when_the_clip_disappears(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia]
    ) -> None:
        # 存在しない ID を持ち続けると、次の操作で「見つからない」例外になる
        window.import_media([two_clips[0].path])
        assert window.wait_for_imports()
        clip = next(iter(window._document.project.timeline.video_tracks())).clips[0]
        window._timeline.select(clip.id)
        window._timeline.delete_selected()
        assert window._timeline.selected_clip is None

    def test_broken_media_does_not_stop_the_others(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia], tmp_path: Path
    ) -> None:
        # 1 本の壊れたファイルで止まると、一緒に選んだ残りの素材も読み込まれない
        missing = tmp_path / "行方不明.mp4"
        window.import_media([missing, two_clips[0].path])
        assert window.wait_for_imports()
        assert len(window._document.project.media) == 1


class TestSaveAndExport:
    def test_save_and_reload(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia], tmp_path: Path
    ) -> None:
        # 保存して開き直すと切った所が消えていたら、作業した分がまるごと失われる
        window.import_media([two_clips[0].path, two_clips[1].path])
        assert window.wait_for_imports()
        window._seek(45)
        window._timeline.split_at_playhead()

        path = tmp_path / "flow.sme"
        window._path = path
        window.save_project()

        reloaded = load_project(path)
        assert _clip_counts(reloaded) == _clip_counts(window._document.project)
        assert reloaded.duration == window._document.project.duration

    @pytest.mark.usefixtures("gpu")
    def test_export_produces_a_file(
        self, window: MainWindow, two_clips: tuple[SampleMedia, SampleMedia], tmp_path: Path
    ) -> None:
        # ここが通らないと、編集はできても動画として外へ出せず作品が仕上がらない
        codecs = available_video_codecs()
        if not codecs:
            pytest.skip("使えるコーデックが無い")

        window.import_media([two_clips[0].path])
        assert window.wait_for_imports()
        window._seek(30)
        window._timeline.split_at_playhead()

        output = tmp_path / "flow.mp4"
        export_project(window._document.project, ExportSettings(path=output, video_codec=codecs[0]))
        assert output.exists()
        assert output.stat().st_size > 0
