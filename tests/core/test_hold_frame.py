"""絵を止めたクリップ（``Clip.hold_at`` Issue #115）

YMM4 は素材より長い動画アイテムで最後の絵を、再生速度 0 で素材の頭の絵を出し続ける
それを写すための持ち方で、壊れ方は 3 通りある

- 編集の命令で止める位置がずれる 分割した後半が動き出したり、止める絵が変わったりする
- 保存し直すと消える 開き直しただけで止めた絵が動き出す
- 古いファイルが開けなくなる
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.commands import AddClip, RippleCut, SetClipProperty, SplitClip, TrimClip
from sashimono.core.io import (
    FORMAT_VERSION,
    ProjectFileError,
    project_from_dict,
    project_to_dict,
)
from sashimono.core.model import Clip, MediaItem, Project
from sashimono.core.timebase import FrameRate

RATE = FrameRate(30)


def _placed(project: Project, clip: Clip) -> Project:
    return AddClip(project.timeline.tracks[0].id, clip).apply(project)


def _clips(project: Project) -> tuple[Clip, ...]:
    return project.timeline.tracks[0].clips


def _times(clip: Clip) -> list[Fraction]:
    """クリップの頭から終わりまで、各フレームで絵を取る素材の時刻"""
    return [clip.picture_time(frame, RATE) for frame in range(clip.duration)]


@pytest.fixture
def held(project: Project, video_media: MediaItem) -> Project:
    """10 秒の素材の 1 秒目から読み、素材の 5 秒の所で止めるクリップ（300 フレーム）"""
    clip = Clip(
        timeline_start=0,
        duration=300,
        media_id=video_media.id,
        source_in=Fraction(1),
        hold_at=Fraction(5),
    )
    return _placed(project, clip)


class TestPictureTime:
    def test_the_picture_moves_until_the_hold(self) -> None:
        # YMM4 から読んだ素材より長い動画アイテムが、素材の終わりより前から静止画になったり
        # 頭が飛んだりする
        clip = Clip(timeline_start=0, duration=300, source_in=Fraction(1), hold_at=Fraction(5))
        # 止める前は素のクリップと同じく進む 止める所を早めに効かせると、動くはずの所が止まる
        assert clip.picture_time(30, RATE) == Fraction(2)
        assert clip.picture_time(120, RATE) == Fraction(5)

    def test_the_picture_never_reads_past_the_hold(self) -> None:
        # 越えて読むと、止まるはずの絵が動き出す（素材の終わりの後なら何も映らない）
        clip = Clip(timeline_start=0, duration=300, source_in=Fraction(1), hold_at=Fraction(5))
        assert clip.picture_time(121, RATE) == Fraction(5)
        assert clip.picture_time(299, RATE) == Fraction(5)

    def test_a_hold_at_the_start_freezes_every_frame(self) -> None:
        # YMM4 の再生速度 0 は素材の頭の絵のまま 1 フレームでも進むと別の絵になる
        clip = Clip(timeline_start=0, duration=180, source_in=Fraction(2), hold_at=Fraction(2))
        assert set(_times(clip)) == {Fraction(2)}

    def test_a_clip_without_a_hold_keeps_reading_the_source(self) -> None:
        # 既定で止めると、素材の終わりまで使うだけのクリップが動かなくなる
        clip = Clip(timeline_start=0, duration=300, speed=Fraction(2))
        assert clip.picture_time(299, RATE) == Fraction(299, 15)

    def test_a_negative_hold_is_refused(self) -> None:
        # 素材の頭より前は無い 通すと、どの絵を出すのかが決まらない
        with pytest.raises(ValueError, match="止める時刻"):
            Clip(timeline_start=0, duration=10, hold_at=Fraction(-1))


class TestEditing:
    def test_the_later_half_of_a_split_stays_frozen(self, held: Project) -> None:
        """止める位置より後で割った後半は、ずっと止まった絵

        後半の ``source_in`` は止める位置より後になる 止める位置をクリップの中の経過で
        持っていると、後半は 0 から数え直して動き出す
        """
        (clip,) = _clips(held)
        _, right = _clips(SplitClip(clip.id, 200).apply(held))
        assert right.source_in > Fraction(5)
        assert set(_times(right)) == {Fraction(5)}

    def test_a_split_before_the_hold_keeps_the_same_pictures(self, held: Project) -> None:
        # 割っても、各フレームに出る絵は割る前と同じ 止める所が前半の長さの分ずれると、
        # 割った所から後の絵が早く止まる
        (clip,) = _clips(held)
        left, right = _clips(SplitClip(clip.id, 60).apply(held))
        assert _times(left) + _times(right) == _times(clip)

    def test_trimming_the_head_past_the_hold_freezes_the_rest(self, held: Project) -> None:
        # 止める位置が付いていかないと、頭を削っただけで止まっていた絵が動き出す
        (clip,) = _clips(held)
        (trimmed,) = _clips(TrimClip(clip.id, head_delta=150).apply(held))
        assert set(_times(trimmed)) == {Fraction(5)}

    def test_trimming_the_head_keeps_the_pictures_in_place(self, held: Project) -> None:
        # 頭を削ったぶん、残った所は同じ絵のまま 止める所がずれると、削っただけで止まる所が動く
        (clip,) = _clips(held)
        (trimmed,) = _clips(TrimClip(clip.id, head_delta=30).apply(held))
        assert _times(trimmed) == _times(clip)[30:]

    def test_a_faster_clip_still_stops_on_the_same_picture(self, held: Project) -> None:
        # 速さを変えても止める絵は同じ（素材の中の時刻で持つので） 止める所へ早く着くだけ
        (clip,) = _clips(held)
        (fast,) = _clips(SetClipProperty(clip.id, "speed", Fraction(2)).apply(held))
        assert fast.picture_time(60, RATE) == Fraction(5)
        assert fast.picture_time(299, RATE) == Fraction(5)

    def test_a_ripple_cut_keeps_the_rest_frozen(self, held: Project) -> None:
        # 無音のカットで真ん中を抜いても、残った後ろは止まったまま
        _, right = _clips(RippleCut(((100, 160),)).apply(held))
        assert set(_times(right)) == {Fraction(5)}

    def test_the_hold_can_be_released(self, held: Project) -> None:
        # インスペクタの「解除」 外せないと、読み込みで付いた止め方を直す道が無い
        (clip,) = _clips(held)
        (released,) = _clips(SetClipProperty(clip.id, "hold_at", None).apply(held))
        assert released.hold_at is None
        assert released.picture_time(299, RATE) == Fraction(1) + Fraction(299, 30)


class TestFile:
    def test_the_hold_survives_saving(self, held: Project) -> None:
        # 保存し直すと消えるなら、開き直しただけで止めた絵が動き出す
        (clip,) = _clips(project_from_dict(project_to_dict(held)))
        assert clip.hold_at == Fraction(5)

    def test_a_clip_without_a_hold_reopens_without_one(self, held: Project) -> None:
        # 解除した止め方が開き直すと戻ると、動かしたはずの絵が止まって見える
        (clip,) = _clips(held)
        loose = SetClipProperty(clip.id, "hold_at", None).apply(held)
        (reopened,) = _clips(project_from_dict(project_to_dict(loose)))
        assert reopened.hold_at is None

    def test_an_old_file_opens_without_holding(self, held: Project) -> None:
        """版 3 までのファイル（項目が無い）は止めないクリップとして開く

        項目が無いのを壊れたファイルとして断ると、今までのプロジェクトが全部開けなくなる
        """
        data = project_to_dict(held)
        data["version"] = 3
        timeline = data["timeline"]
        assert isinstance(timeline, dict)
        for track in timeline["tracks"]:
            for clip in track["clips"]:
                del clip["hold_at"]
        (clip,) = _clips(project_from_dict(data))
        assert clip.hold_at is None

    def test_the_format_version_says_the_hold_is_there(self, held: Project) -> None:
        # 版を上げないと、3 までの本体が項目を黙って捨て、保存し直すと止め方ごと消える
        assert FORMAT_VERSION >= 4
        assert project_to_dict(held)["version"] == FORMAT_VERSION

    def test_the_later_half_of_a_split_keeps_its_hold_after_saving(self, held: Project) -> None:
        # 割った後半も保存して開き直せる（別の ID でも止める位置を持つ）
        (clip,) = _clips(held)
        split = SplitClip(clip.id, 200).apply(held)
        reopened = project_from_dict(project_to_dict(split))
        assert [c.hold_at for c in _clips(reopened)] == [Fraction(5), Fraction(5)]

    def test_a_true_in_a_broken_file_is_not_read_as_one_second(self, held: Project) -> None:
        """壊れたファイルの ``true`` は断る 1 秒として読むと、どこで止めたのか分からない絵になる

        真偽値は ``int`` の仲間なので、分数の項目を読む所で断らないと 1 として通る
        こちらは分数の項目へ真偽値を書かないので、どの項目でも壊れたファイル
        """
        data = project_to_dict(held)
        timeline = data["timeline"]
        assert isinstance(timeline, dict)
        timeline["tracks"][0]["clips"][0]["hold_at"] = True
        with pytest.raises(ProjectFileError, match="hold_at"):
            project_from_dict(data)

    def test_a_true_speed_is_refused_too(self, held: Project) -> None:
        # 速さの true を 1 倍と読むと、壊れた所に気付かないまま保存し直して直ってしまう
        data = project_to_dict(held)
        timeline = data["timeline"]
        assert isinstance(timeline, dict)
        timeline["tracks"][0]["clips"][0]["speed"] = True
        with pytest.raises(ProjectFileError, match="speed"):
            project_from_dict(data)

    def test_the_end_of_the_picture_survives_saving(self, held: Project) -> None:
        # 映像の終わりが消えると、開き直した後に置いた動画はコンテナの長さで止まる
        media = held.media[0]
        (stream,) = media.video_streams
        marked = held.replace_media(
            replace(media, video_streams=(replace(stream, end_time=Fraction(9)),))
        )
        reopened = project_from_dict(project_to_dict(marked))
        assert reopened.media[0].video_streams[0].end_time == Fraction(9)

    def test_media_saved_without_the_end_of_the_picture_still_opens(self, held: Project) -> None:
        # 映像の終わりを覚える前に取り込んだ素材が開けないと、今までのプロジェクトが開けない
        data = project_to_dict(held)
        media_list = data["media"]
        assert isinstance(media_list, list)
        for media in media_list:
            for stream in media["video_streams"]:
                del stream["end_time"]
        reopened = project_from_dict(data)
        assert reopened.media[0].video_streams[0].end_time is None

    def test_a_version_4_end_of_the_picture_is_not_trusted(self, held: Project) -> None:
        """版 4 の映像の終わりは PTS そのままの数え方なので捨てる（Issue #123）

        頭 5 秒・長さ 2 秒の素材なら版 4 は 7 秒と書いている 今の数え方（頭から 2 秒）として
        読むと、テンプレートを置いたときに映像の終わりの後で絵を止め、止めた所から何も映らない
        捨てておけば、絵を止める所で素材を開き直して取る（版 3 までの素材と同じ道）
        """
        media = held.media[0]
        (stream,) = media.video_streams
        marked = held.replace_media(
            replace(media, video_streams=(replace(stream, end_time=Fraction(7)),))
        )
        data = project_to_dict(marked)
        data["version"] = 4
        reopened = project_from_dict(data)
        assert reopened.media[0].video_streams[0].end_time is None

    def test_the_format_version_says_media_are_timed_from_their_head(self, held: Project) -> None:
        # 版を上げないと、版 4 の値（PTS そのまま）と今の値（頭から）を見分けられない
        assert FORMAT_VERSION >= 5
        assert project_to_dict(held)["version"] == FORMAT_VERSION


class TestCommand:
    def test_a_number_that_is_not_a_fraction_is_refused(self, held: Project) -> None:
        # 整数や小数を通すと、モデルには入るが保存の所で分数として書けずに落ちる
        (clip,) = _clips(held)
        with pytest.raises(ValueError, match="止める時刻"):
            SetClipProperty(clip.id, "hold_at", 2).apply(held)
