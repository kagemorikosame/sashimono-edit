"""プレビューの 1 コマを編集の中で使う（#525）の core の分

〔このコマで止める〕（FreezeFrame）と、静止画を書く名前の付け方と、静止画を置く長さ
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands import (
    MAX_STILL_FILES,
    Command,
    Document,
    FreezeFrame,
    insert_media,
    place_media,
    still_file_name,
)
from sashimono.core.model import (
    Clip,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from tests.conftest import RATE_30


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _placed(media: MediaItem, mode: str = LayerMode.SEPARATED) -> Project:
    base = Project.create(ProjectSettings(frame_rate=RATE_30, layer_mode=mode))
    return _apply(base, insert_media(base, media))


def _clips(project: Project) -> list[tuple[Track, Clip]]:
    return [(t, c) for t in project.timeline.tracks for c in t.clips]


def _picture(project: Project) -> Clip:
    return next(c for t, c in _clips(project) if project.draws_picture(t, c))


class TestFreezeFrame:
    def test_the_back_half_holds_the_frame_at_the_playhead(self, video_media: MediaItem) -> None:
        # 壊れると、〔このコマで止める〕で前側まで止まるか、再生ヘッドと違うコマで止まる
        project = _placed(video_media)
        picture = _picture(project)
        frozen = FreezeFrame(picture.id, 60).apply(project)
        pictures = sorted(
            (c for t, c in _clips(frozen) if frozen.draws_picture(t, c)),
            key=lambda c: c.timeline_start,
        )
        front, back = pictures
        # 前は動いたまま 後ろは 60 コマ目（2 秒）の絵で止まる
        assert front.hold_at is None
        assert (front.timeline_start, front.timeline_end) == (0, 60)
        assert back.timeline_start == 60
        assert back.hold_at == Fraction(2)
        assert back.picture_time(100, RATE_30) == Fraction(2)

    def test_the_linked_sound_is_split_but_keeps_playing(self, video_media: MediaItem) -> None:
        # 音まで止めると、止め絵の間も流れてほしい声と BGM が切れる
        project = _placed(video_media)
        frozen = FreezeFrame(_picture(project).id, 60).apply(project)
        sounds = [c for t, c in _clips(frozen) if t.kind is TrackKind.AUDIO]
        assert sorted(c.timeline_start for c in sounds) == [0, 60]
        assert all(c.hold_at is None for c in sounds)

    def test_at_the_head_the_whole_clip_holds(self, video_media: MediaItem) -> None:
        # 壊れると、クリップの頭で止めたときに長さ 0 の切れ端ができる
        project = _placed(video_media)
        picture = _picture(project)
        frozen = FreezeFrame(picture.id, 0).apply(project)
        held = _picture(frozen)
        assert held.id == picture.id
        assert held.hold_at == Fraction(0)
        assert len(_clips(frozen)) == len(_clips(project))

    def test_a_merged_mixed_clip_holds_only_the_picture(self, video_media: MediaItem) -> None:
        # 混合の方式で 1 本にまとめたクリップも、音は止めない（絵の時刻だけが止まる）
        base = Project.create(ProjectSettings(frame_rate=RATE_30, layer_mode=LayerMode.MIXED))
        project = _apply(base, insert_media(base, video_media, split_audio=False))
        merged = _picture(project)
        frozen = FreezeFrame(merged.id, 30).apply(project)
        back = next(c for _, c in _clips(frozen) if c.timeline_start == 30)
        assert back.hold_at == Fraction(1)
        assert back.audio_stream is not None

    def test_outside_the_clip_is_refused(self, video_media: MediaItem) -> None:
        # 壊れると、クリップの外で止めて何も起きないか、割れない所で落ちる
        project = _placed(video_media)
        with pytest.raises(ValueError, match="内側にない"):
            FreezeFrame(_picture(project).id, 9999).apply(project)

    def test_a_sound_or_text_clip_is_refused(
        self, video_media: MediaItem, audio_media: MediaItem
    ) -> None:
        # 止める絵の動きを持たない物を止めた印だけ付けると、何も変わらず理由も分からない
        project = _placed(audio_media)
        (_, sound), *_ = _clips(project)
        with pytest.raises(ValueError, match="動画の絵"):
            FreezeFrame(sound.id, 10).apply(project)

    def test_one_undo_brings_back_the_moving_clip(self, video_media: MediaItem) -> None:
        # 壊れると、止めた絵を取り消しで元の動く映像へ戻せない
        project = _placed(video_media)
        document = Document(project)
        document.execute(FreezeFrame(_picture(project).id, 60))
        document.undo()
        assert document.project == project


class TestFileName:
    def test_the_name_has_the_project_and_the_timecode(self) -> None:
        # 壊れると、静止画の名前からどの作品のどのコマか分からなくなる
        name = still_file_name("旅行", 45, RATE_30, lambda _name: False)
        assert name == "旅行_00-00-01-15.png"

    def test_characters_windows_refuses_are_replaced(self) -> None:
        # 壊れると、作品名に記号があるときに PNG を書けず止まる
        name = still_file_name('a:b*c?"', 0, RATE_30, lambda _name: False)
        assert not set(name) & set(':*?"<>|')

    def test_an_existing_name_gets_a_number(self) -> None:
        # 上書きすると、先に置いた静止画の絵が後から黙って変わる
        taken = {"旅行_00-00-00-00.png", "旅行_00-00-00-00_2.png"}
        name = still_file_name("旅行", 0, RATE_30, taken.__contains__)
        assert name == "旅行_00-00-00-00_3.png"

    def test_running_out_of_numbers_is_refused(self) -> None:
        # 壊れると、番号を使い切ったときに前の静止画を上書きする
        with pytest.raises(ValueError, match="多すぎる"):
            still_file_name("旅行", 0, RATE_30, lambda _name: True)
        assert MAX_STILL_FILES >= 99


class TestStillLength:
    def test_a_still_is_placed_for_the_asked_frames(self) -> None:
        # 壊れると、設定した静止画の長さが効かない
        still = MediaItem(path=Path("C:/素材/表紙.png"))
        project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(still,))
        placed = _apply(project, place_media(project, [still], at_frame=0, still_frames=45))
        ((_, clip),) = _clips(placed)
        assert clip.duration == 45

    def test_without_a_length_the_default_is_kept(self) -> None:
        # 壊れると、ふつうに置いた静止画の長さまで変わる
        still = MediaItem(path=Path("C:/素材/表紙.png"))
        project = Project.create(ProjectSettings(frame_rate=RATE_30), media=(still,))
        placed = _apply(project, place_media(project, [still], at_frame=0))
        ((_, clip),) = _clips(placed)
        assert clip.duration == 150

    def test_moving_media_ignores_the_still_length(self, video_media: MediaItem) -> None:
        # 壊れると、静止画の長さの設定で動画まで短く切られて置かれる
        project = replace(Project.create(ProjectSettings(frame_rate=RATE_30)))
        placed = _apply(project, place_media(project, [video_media], at_frame=0, still_frames=10))
        assert {c.duration for _, c in _clips(placed)} == {300}
