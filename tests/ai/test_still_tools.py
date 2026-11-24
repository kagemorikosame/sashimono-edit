"""静止画を素材にする・このコマで止める AI の道具（#525）

書く先の守り（場所の決まり・身元の照らし）は保存と書き出しの試験（test_save_and_export.py）と
同じ関数を通るので、ここでは書く場所・名前・読み込みと置き方・断り方を見る
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import (
    still_consent,
    still_plan,
    write_and_import_still,
)
from sashimono.core.commands import STILLS_FOLDER
from sashimono.core.io.atomic import expect_now
from sashimono.core.model import MediaItem, Transcript
from tests.ai.conftest import FakeHost
from tests.ai.test_outbound import _with_text_clip


@pytest.fixture
def host(video_media: MediaItem, transcript: Transcript, tmp_path: Path) -> FakeHost:
    created = _with_text_clip(video_media, transcript)
    folder = tmp_path / "作品"
    folder.mkdir()
    created.saved_at = folder / "作品.sme"
    created.write_project(created.saved_at, adopt=False, expect=expect_now(created.saved_at))
    # 本物の読み込みと同じく、読むたびに別の素材にする（偽の素材は ID を使い回す）
    setattr(created, "probe", lambda path: MediaItem(path=path))  # noqa: B010  mypy がメソッドへの代入を断る
    return created


def _make(host: FakeHost, **arguments: object) -> dict[str, object]:
    plan = still_plan(host, dict(arguments))
    consent = still_consent(host, dict(arguments))
    return write_and_import_still(host, plan, consent, host.apply_commands)


class TestSnapshotToMedia:
    def test_the_png_goes_to_the_stills_folder_and_into_the_pool(self, host: FakeHost) -> None:
        # 壊れると、AI が素材にした静止画が作品のフォルダの外へ書かれるか、素材一覧に出ず使えない
        before = len(host.document.project.media)
        result = _make(host, frame=0)
        written = Path(str(result["file"]))
        assert written.parent.name == STILLS_FOLDER
        assert written.is_file()
        assert len(host.document.project.media) == before + 1
        # 置くと頼まなければタイムラインは変えない
        assert result["clip_ids"] == []

    def test_placing_puts_the_still_for_the_set_length(self, host: FakeHost) -> None:
        # 壊れると、置いた静止画が再生ヘッドからずれるか、設定した長さにならない
        host.seek(30)
        result = _make(host, place=True)
        clip_ids = result["clip_ids"]
        assert isinstance(clip_ids, list)
        (clip_id,) = clip_ids
        clip = next(
            c
            for t in host.document.project.timeline.tracks
            for c in t.clips
            if str(c.id) == clip_id
        )
        assert (clip.timeline_start, clip.duration) == (30, host.still_frames)

    def test_a_second_still_of_the_same_frame_gets_a_new_name(self, host: FakeHost) -> None:
        # 上書きすると、先に置いた静止画の絵が後から黙って変わる
        first = Path(str(_make(host, frame=0)["file"]))
        second = Path(str(_make(host, frame=0)["file"]))
        assert first != second
        assert first.is_file() and second.is_file()

    def test_one_undo_takes_back_the_import_and_the_clip(self, host: FakeHost) -> None:
        # 壊れると、取り消しても読み込んだ素材か置いた静止画が残り、何回も戻す羽目になる
        before = host.document.project
        result = _make(host, place=True)
        host.document.undo()
        assert host.document.project == before
        # 書いた PNG は残す 元に戻しても人が素材として読み込み直せる
        assert Path(str(result["file"])).is_file()

    def test_an_unsaved_project_is_refused(self, host: FakeHost) -> None:
        # 壊れると、保存していない作品でどこへ書いたか分からない静止画ができ、作品を移すと見失う
        host.saved_at = None
        with pytest.raises(ToolError, match="保存していない"):
            still_plan(host, {})

    def test_a_file_named_stills_is_refused(self, host: FakeHost) -> None:
        # 壊れると、同じ名前のファイルを潰すか、書けずに途中で落ちる
        assert host.saved_at is not None
        (host.saved_at.parent / STILLS_FOLDER).write_text("", encoding="utf-8")
        with pytest.raises(ToolError, match="ファイルがある"):
            still_plan(host, {})

    def test_a_frame_that_is_not_a_number_is_refused(self, host: FakeHost) -> None:
        # 壊れると、AI が渡した文字のフレームで落ちるか、意図しないコマを描く
        with pytest.raises(ToolError, match="フレーム番号"):
            still_plan(host, {"frame": "最初"})


class TestFreezeFrameTool:
    def test_the_video_clip_is_held_from_the_frame(self, host: FakeHost) -> None:
        # 壊れると、AI の〔このコマで止める〕が別の時刻の絵で止まる
        from sashimono.ai.operations import find_operation

        video = next(
            c
            for t in host.document.project.timeline.tracks
            for c in t.clips
            if c.media_id is not None
        )
        operation = find_operation("freeze_frame")
        assert operation is not None
        result = operation(host, {"clip_id": str(video.id), "frame": 15})
        assert isinstance(result, dict)
        assert result["hold_seconds"] == 0.5

    def test_a_text_clip_is_refused(self, host: FakeHost) -> None:
        # 壊れると、止める絵の無いテキストに止めた印だけ付き、何も変わらず理由も分からない
        from sashimono.ai.operations import find_operation

        text = next(
            c for t in host.document.project.timeline.tracks for c in t.clips if c.media_id is None
        )
        operation = find_operation("freeze_frame")
        assert operation is not None
        with pytest.raises(ToolError, match="動画の絵"):
            operation(host, {"clip_id": str(text.id), "frame": 0})
        assert replace(host.document.project) == host.document.project
