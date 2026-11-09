"""持ち出し用パッケージに入っている範囲の外を読む編集を断る（#445）"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands import AddClip, Command, Document, TrimClip
from sashimono.core.commands.insert import place_media
from sashimono.core.commands.speed import ChangeSpeed
from sashimono.core.io.serialize import load_project, project_from_dict, project_to_dict
from sashimono.core.io.serialize import save_project as save
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    ClipId,
    MediaItem,
    PackedSpan,
    Project,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.packed import packed_refusal
from sashimono.core.timebase import FrameRate

RATE = FrameRate(30, 1)


def _media(packed: tuple[PackedSpan, ...]) -> MediaItem:
    return MediaItem(
        path=Path("C:/展開/media/0001.mp4"),
        duration=Fraction(60),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1920,
                height=1080,
                frame_rate=RATE,
                time_base=Fraction(1, 30),
                codec="h264",
                pixel_format="yuv420p",
            ),
        ),
        audio_streams=(
            AudioStreamInfo(
                index=1, sample_rate=48000, channels=2, time_base=Fraction(1, 48000), codec="aac"
            ),
        ),
        display_name="本編.mp4",
        packed=packed,
    )


def _document(
    packed: tuple[PackedSpan, ...] = (PackedSpan(Fraction(10), Fraction(20)),),
) -> tuple[Document, Clip]:
    item = _media(packed)
    # 素材の 12 秒から 3 秒（90 コマ）を読むクリップ 入っている範囲は 10〜20 秒
    clip = Clip(timeline_start=600, duration=90, media_id=item.id, source_in=Fraction(12))
    project = Project.create(media=(item,))
    track = Track(kind=TrackKind.VIDEO, name="映像 1", clips=(clip,))
    project = replace(project, timeline=replace(project.timeline, tracks=(track,)))
    return Document(project), clip


@dataclass(frozen=True)
class _Slip(Command):
    """素材の中の位置をずらす（スリップ） 画面と AI の位置の欄が当てる変更と同じ形"""

    clip_id: ClipId
    source_in: Fraction

    @property
    def label(self) -> str:
        return "位置をずらす"

    def apply(self, project: Project) -> Project:
        track = project.timeline.tracks[0]
        clips = tuple(
            replace(c, source_in=self.source_in) if c.id == self.clip_id else c for c in track.clips
        )
        return replace(
            project, timeline=replace(project.timeline, tracks=(track.with_clips(clips),))
        )


class TestRefusal:
    def test_trimming_out_of_the_range_is_refused_with_the_reason(self) -> None:
        """範囲の外へ頭を伸ばせると、開いた先で黙って錨の絵か何も無い所が出る"""
        document, clip = _document()
        before = document.project
        with pytest.raises(ValueError, match="持ち出し用パッケージに入っている範囲") as caught:
            document.execute(TrimClip(clip.id, head_delta=-90))
        # 何を確かめれば直せるかが分かる文 入っている範囲と、読もうとした所を言う
        assert "10.0〜20.0 秒" in str(caught.value)
        assert "本編.mp4" in str(caught.value)
        assert document.project is before

    def test_trimming_inside_the_range_still_works(self) -> None:
        """範囲の中の編集まで断ると、持ち出した作品を少しも直せない"""
        document, clip = _document()
        document.execute(TrimClip(clip.id, head_delta=-30, tail_delta=60))
        moved = document.project.timeline.tracks[0].clips[0]
        assert moved.source_in == Fraction(11)

    def test_speeding_up_past_the_end_is_refused(self) -> None:
        """長さを保って速くすると、範囲の終わりより後ろを読む"""
        document, clip = _document()
        with pytest.raises(ValueError, match="範囲"):
            document.execute(ChangeSpeed((clip.id,), Fraction(4), keep_length=True))

    def test_slipping_out_of_the_range_is_refused(self) -> None:
        """素材の中の位置をずらすコマンドも、コマンドの中で確かめなくても断られる"""
        document, clip = _document()
        with pytest.raises(ValueError, match="範囲"):
            document.execute(_Slip(clip.id, Fraction(30)))
        document.execute(_Slip(clip.id, Fraction(15)))

    def test_a_clip_already_outside_does_not_block_other_edits(self) -> None:
        """前の本体で外へ伸ばしたクリップがあるだけで、ほかの編集が全部断られると困る"""
        document, clip = _document()
        outside = replace(clip, id=ClipId("c-outside"), timeline_start=900, source_in=Fraction(40))
        project = document.project
        track = project.timeline.tracks[0].with_clips((clip, outside))
        document = Document(replace(project, timeline=replace(project.timeline, tracks=(track,))))
        document.execute(TrimClip(clip.id, tail_delta=-10))
        # 外のクリップを縮めるのは通し、さらに外へ伸ばすのは断る
        document.execute(TrimClip(outside.id, tail_delta=-10))
        with pytest.raises(ValueError, match="範囲"):
            document.execute(TrimClip(outside.id, tail_delta=30))

    def test_a_clip_across_two_ranges_is_refused(self) -> None:
        """2 か所の範囲の間は素材が無いので、またぐと途中が抜ける"""
        packed = (PackedSpan(Fraction(10), Fraction(14)), PackedSpan(Fraction(16), Fraction(20)))
        document, clip = _document(packed)
        with pytest.raises(ValueError, match=r"10\.0〜14\.0 秒、16\.0〜20\.0 秒"):
            document.execute(TrimClip(clip.id, tail_delta=30))

    def test_media_without_ranges_is_never_checked(self) -> None:
        """普段の作品の素材は全体があるので、どこを読んでも断らない"""
        document, clip = _document(())
        document.execute(TrimClip(clip.id, head_delta=-300))
        assert packed_refusal(document.project, document.project) is None


class TestPlacing:
    def test_placing_again_fits_inside_the_longest_range(self) -> None:
        """素材一覧から置き直すと全体を読もうとして断られ、置き直せなくなる"""
        packed = (PackedSpan(Fraction(10), Fraction(14)), PackedSpan(Fraction(30), Fraction(81, 2)))
        document, _clip = _document(packed)
        item = document.project.media[0]
        commands = place_media(document.project, [item], at_frame=300)
        with document.checkpoint("置く"):
            for command in commands:
                document.execute(command)
        placed = [c.clip for c in commands if isinstance(c, AddClip) and c.clip.media_id == item.id]
        assert placed
        for clip in placed:
            assert clip.source_in == Fraction(30)
            # 切り捨てるので、丸めて区間の外へ 1 コマ出ない
            assert clip.source_out(RATE) <= Fraction(81, 2)


class TestSaving:
    def test_ranges_survive_saving_and_opening(self, tmp_path: Path) -> None:
        """保存し直すと範囲を忘れると、次に開いたときから断らなくなる"""
        document, _clip = _document(
            (PackedSpan(Fraction(10), Fraction(20)), PackedSpan(Fraction(1, 3), Fraction(2)))
        )
        path = tmp_path / "作品.sme"
        save(document.project, path)
        assert load_project(path).media[0].packed == document.project.media[0].packed

    def test_ordinary_media_writes_nothing_new(self) -> None:
        """範囲の無い素材まで項目を書くと、普段の作品のファイルの中身が変わる"""
        document, _clip = _document(())
        assert "packed" not in project_to_dict(document.project)["media"][0]

    def test_a_file_without_ranges_opens_as_whole_media(self) -> None:
        """前の本体が書いたファイル（項目が無い）は、素材の全体がある物として開く"""
        document, _clip = _document()
        data = project_to_dict(document.project)
        del data["media"][0]["packed"]
        assert project_from_dict(data).media[0].packed == ()
