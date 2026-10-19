"""モデルの不変条件と補間の振る舞い"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

import pytest

from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    Interpolation,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Timeline,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
)
from sashimono.core.timebase import FrameRate
from tests.conftest import RATE_30, make_clip


class TestClip:
    def test_rejects_zero_duration(self, video_media: MediaItem) -> None:
        # 長さ 0 のクリップを持てると、どのフレームにも当たらず絵も音も出ないまま作品と保存に残る
        with pytest.raises(ValueError, match="1 フレーム以上"):
            Clip(timeline_start=0, duration=0, media_id=video_media.id)

    def test_rejects_non_positive_speed(self, video_media: MediaItem) -> None:
        # 速さが 0 や負だと素材から読む長さが 0 や負になり、絵が止まったまま進まないか
        # 素材の頭より前を読みに行く
        with pytest.raises(ValueError, match="正でなければ"):
            Clip(timeline_start=0, duration=10, media_id=video_media.id, speed=Fraction(0))

    def test_source_out_accounts_for_speed(self, video_media: MediaItem) -> None:
        # 30 フレーム = 1 秒 2 倍速なら 2 秒分のソースを消費する
        clip = Clip(timeline_start=0, duration=30, media_id=video_media.id, speed=Fraction(2))
        assert clip.source_out(RATE_30) == Fraction(2)

    def test_contains_excludes_end(self, video_media: MediaItem) -> None:
        # 終わりのフレームを含めると、カットで接した 2 つのクリップが継ぎ目で両方当たり、
        # プレビューと書き出しで前のクリップが 1 フレーム長く映る
        clip = make_clip(10, 20, video_media)
        assert clip.contains(10)
        assert clip.contains(29)
        assert not clip.contains(30)


class TestTrack:
    def test_rejects_overlapping_clips(self, video_media: MediaItem) -> None:
        # 同じトラックの重なりを通すと、重なった所は先に並んだクリップだけが描かれ、
        # 後ろのクリップの頭が黙って欠ける
        with pytest.raises(ValueError, match="重なっている"):
            Track(
                kind=TrackKind.VIDEO,
                clips=(make_clip(0, 30, video_media), make_clip(29, 30, video_media)),
            )

    def test_allows_touching_clips(self, video_media: MediaItem) -> None:
        # 端が接するのは重なりではない カット直後の状態がこれ
        track = Track(
            kind=TrackKind.VIDEO,
            clips=(make_clip(0, 30, video_media), make_clip(30, 30, video_media)),
        )
        assert track.end_frame == 60

    def test_rejects_unsorted_clips(self, video_media: MediaItem) -> None:
        # 並びが崩れると、最後のクリップの終わりを作品の長さとする所が途中のクリップを掴み、
        # 書き出しの「全体」が手前で切れる
        with pytest.raises(ValueError, match="開始位置順"):
            Track(
                kind=TrackKind.VIDEO,
                clips=(make_clip(60, 30, video_media), make_clip(0, 30, video_media)),
            )

    def test_with_clips_sorts(self, video_media: MediaItem) -> None:
        # 移動や貼り付けで順の変わったクリップを並べ直さずに渡すと、並びの検査で落ちて
        # 編集の操作がエラーで止まる
        track = Track(kind=TrackKind.VIDEO).with_clips(
            (make_clip(60, 30, video_media), make_clip(0, 30, video_media))
        )
        assert [c.timeline_start for c in track.clips] == [0, 60]

    def test_clip_at(self, video_media: MediaItem) -> None:
        # プレビューと書き出しはこれでフレームごとに描くクリップを引く 外れると、何も無い所に
        # 絵が出たり、クリップの最中に絵が消えたりする
        track = Track(kind=TrackKind.VIDEO, clips=(make_clip(10, 20, video_media),))
        assert track.clip_at(15) is not None
        assert track.clip_at(5) is None
        assert track.clip_at(30) is None


class TestTimeline:
    def test_duration_is_longest_track(self, video_media: MediaItem) -> None:
        # 一番長いトラックで数えないと、書き出しの「全体」と素材を末尾へ足す位置が
        # 短いトラックの終わりになり、長い方の後半が書き出されない
        timeline = Timeline(
            rate=RATE_30,
            tracks=(
                Track(kind=TrackKind.VIDEO, clips=(make_clip(0, 30, video_media),)),
                Track(kind=TrackKind.VIDEO, clips=(make_clip(0, 90, video_media),)),
            ),
        )
        assert timeline.duration == 90

    def test_empty_timeline_has_zero_duration(self) -> None:
        # 空の作品の長さが 0 でないと、書き出しの窓が空の作品を書き出せる物として扱い、
        # 何も映らない動画を作らせる
        assert Timeline(rate=RATE_30).duration == 0

    def test_rejects_duplicate_track_ids(self) -> None:
        # 同じ ID のトラックが 2 本あると、ID で差し替える所が先に見つけた方だけを書き換え、
        # 操作したトラックが変わらない
        track = Track(kind=TrackKind.VIDEO)
        with pytest.raises(ValueError, match="重複"):
            Timeline(rate=RATE_30, tracks=(track, track))

    def test_rejects_inverted_work_area(self) -> None:
        # 終わりが頭以前の書き出し範囲を持てると、範囲を指定して書き出したときに中身の
        # 無い動画になる
        with pytest.raises(ValueError, match="書き出し範囲"):
            Timeline(rate=RATE_30, work_area=(100, 100))


class TestProject:
    def test_rate_must_match_timeline(self) -> None:
        # 設定とタイムラインでフレームレートが違うと、フレームで数えた位置を秒へ直す割合が
        # 所ごとに違い、絵と音や字幕の時刻がずれる
        with pytest.raises(ValueError, match="一致しない"):
            Project(
                settings=ProjectSettings(frame_rate=FrameRate(30)),
                timeline=Timeline(rate=FrameRate(25)),
            )

    def test_create_uses_settings_rate(self) -> None:
        # 25 fps で作ってもタイムラインが 30 fps のままだと、作る所で設定との食い違いの
        # 検査に落ち、新規作成ができない
        project = Project.create(ProjectSettings(frame_rate=FrameRate(25)))
        assert project.timeline.rate == FrameRate(25)

    def test_rejects_duplicate_media(self, video_media: MediaItem) -> None:
        # 素材を ID で引く所は先に見つけた方だけを返す 同じ ID が 2 つあると、後ろの方へ
        # 付けた字幕や差し替えが作品に効かない
        with pytest.raises(ValueError, match="重複"):
            Project.create(media=(video_media, video_media))

    def test_require_media_raises_for_unknown(self, project: Project) -> None:
        # 無い素材で止めないと、素材を消した後の字幕の編集が行き先の無いまま進み、
        # 直した中身がどこにも残らない
        from sashimono.core.model import MediaId

        with pytest.raises(KeyError):
            project.require_media(MediaId("存在しない"))


class TestAnimatedValue:
    def test_static_when_no_keyframes(self) -> None:
        # キーフレームの無い値が時刻で変わると、動かしていないはずの不透明度やぼかしが
        # クリップの途中で変わる
        value = AnimatedValue(static=42.0)
        assert not value.is_animated
        assert value.at(0) == 42.0
        assert value.at(10_000) == 42.0

    def test_linear_interpolation(self) -> None:
        # 直線の点の間が比で埋まらないと、2 つの点で付けたフェードや移動が途中で跳ぶ
        value = AnimatedValue(
            keyframes=(Keyframe(frame=0, value=0.0), Keyframe(frame=10, value=100.0))
        )
        assert value.at(0) == 0.0
        assert value.at(5) == pytest.approx(50.0)
        assert value.at(10) == 100.0

    def test_clamps_outside_range(self) -> None:
        # 外挿するとトリムしただけで画面外へ飛んでいく 端で頭打ちにする
        value = AnimatedValue(
            keyframes=(Keyframe(frame=10, value=5.0), Keyframe(frame=20, value=15.0))
        )
        assert value.at(0) == 5.0
        assert value.at(1000) == 15.0

    def test_hold_keeps_left_value(self) -> None:
        # 瞬間移動の点が補間されると、次の点まで止まっているはずの位置や色がじわじわ動き、
        # パッと切り替える演出にならない
        value = AnimatedValue(
            keyframes=(
                Keyframe(frame=0, value=1.0, interpolation=Interpolation.HOLD),
                Keyframe(frame=10, value=9.0),
            )
        )
        assert value.at(5) == 1.0
        assert value.at(9) == 1.0
        assert value.at(10) == 9.0

    @pytest.mark.parametrize(
        "interpolation",
        [Interpolation.EASE_IN, Interpolation.EASE_OUT, Interpolation.EASE_IN_OUT],
    )
    def test_easing_stays_monotonic_between_endpoints(self, interpolation: Interpolation) -> None:
        # 加減速の曲線が途中で戻ると、ゆっくり止まるはずのフェードや移動が一度行き過ぎて
        # 戻り、ちらついて見える
        value = AnimatedValue(
            keyframes=(
                Keyframe(frame=0, value=0.0, interpolation=interpolation),
                Keyframe(frame=100, value=1.0),
            )
        )
        samples = [value.at(f) for f in range(0, 101)]
        assert samples[0] == pytest.approx(0.0, abs=1e-6)
        assert samples[-1] == pytest.approx(1.0, abs=1e-6)
        assert all(b >= a - 1e-9 for a, b in pairwise(samples))

    def test_ease_in_starts_slower_than_linear(self) -> None:
        # EASE_IN の向きが逆になると、選んだのと反対の加減速で動き、頭でいきなり速く動き出す
        keyframes = (Keyframe(frame=0, value=0.0), Keyframe(frame=100, value=1.0))
        linear = AnimatedValue(keyframes=keyframes)
        eased = AnimatedValue(
            keyframes=(
                Keyframe(frame=0, value=0.0, interpolation=Interpolation.EASE_IN),
                Keyframe(frame=100, value=1.0),
            )
        )
        assert eased.at(20) < linear.at(20)

    def test_custom_bezier(self) -> None:
        # 制御点の読み方がずれると、ベジェで作った動きが再生では別の形の曲線で動く
        value = AnimatedValue(
            keyframes=(
                Keyframe(
                    frame=0,
                    value=0.0,
                    interpolation=Interpolation.BEZIER,
                    control_points=(0.0, 0.0, 1.0, 1.0),
                ),
                Keyframe(frame=10, value=10.0),
            )
        )
        # (0,0,1,1) は直線と同じ曲線
        assert value.at(5) == pytest.approx(5.0, abs=1e-4)

    def test_a_named_curve_overshoots_like_back(self) -> None:
        """曲線の名前（``back``）を持つキーフレームは、その形で動く

        名前を落として 3 種の加減速へ丸めると、行き過ぎて戻る動きが消え、
        YMM4 の Back_InOut で回るローテンショントランジションが 30 度ずれた
        """
        value = AnimatedValue(
            keyframes=(
                Keyframe(frame=0, value=0.0, interpolation=Interpolation.EASE_IN_OUT, curve="back"),
                Keyframe(frame=100, value=180.0),
            )
        )
        # Back_InOut は前半で一度逆へ振れ、後半で行き過ぎる
        assert min(value.at(f) for f in range(0, 101)) < -5.0
        assert max(value.at(f) for f in range(0, 101)) > 185.0
        assert value.at(100) == 180.0

    def test_the_curve_takes_its_direction_from_the_interpolation(self) -> None:
        # Quart_In は 4 乗 向き（In）は補間方法が持つ 25% の所で 0.25^4
        value = AnimatedValue(
            keyframes=(
                Keyframe(frame=0, value=0.0, interpolation=Interpolation.EASE_IN, curve="quart"),
                Keyframe(frame=100, value=1.0),
            )
        )
        assert value.at(25) == pytest.approx(0.25**4)

    def test_a_curve_on_a_straight_point_is_rejected(self) -> None:
        # 直線や瞬間移動の点は曲線を読まない 持たせると効かない名前がファイルに残り、
        # 読んだ人が「Back で動くはず」と取り違える
        with pytest.raises(ValueError, match="イージング"):
            Keyframe(frame=0, value=0.0, interpolation=Interpolation.LINEAR, curve="back")

    def test_an_unknown_curve_is_rejected(self) -> None:
        # 知らない名前を通すと、描くときに直線へ落ちて誰も気付かない
        with pytest.raises(ValueError, match="曲線"):
            Keyframe(frame=0, value=0.0, interpolation=Interpolation.EASE_IN, curve="魔法")

    def test_bezier_requires_control_points(self) -> None:
        # 制御点の無いベジェの点を作れると、描く所で例外になり、そのクリップのある
        # プレビューと書き出しが止まる
        with pytest.raises(ValueError, match="control_points"):
            Keyframe(frame=0, value=0.0, interpolation=Interpolation.BEZIER)

    def test_rejects_unsorted_keyframes(self) -> None:
        # 点の並びが崩れると、端の値を先頭と末尾で決める所が取り違え、なめらかな動きの
        # 代わりに値が 1 フレームで飛ぶ
        with pytest.raises(ValueError, match="昇順"):
            AnimatedValue(keyframes=(Keyframe(frame=10, value=0.0), Keyframe(frame=0, value=1.0)))

    def test_rejects_duplicate_frames(self) -> None:
        # 同じフレームに点が 2 つあると、そこで値が段差で飛び、どちらの点の値が映るかが
        # 並び次第になる
        with pytest.raises(ValueError, match="複数のキーフレーム"):
            AnimatedValue(keyframes=(Keyframe(frame=5, value=0.0), Keyframe(frame=5, value=1.0)))


class TestEffect:
    def test_with_param_does_not_mutate_original(self) -> None:
        # 元のエフェクトを書き換えると、Ctrl+Z で戻した先の作品も新しい値を持ち、
        # ぼかしの強さを前へ戻せない
        original = Effect(kind="blur", params={"radius": AnimatedValue(4.0)})
        updated = original.with_param("radius", AnimatedValue(8.0))
        assert original.params["radius"] == AnimatedValue(4.0)
        assert updated.params["radius"] == AnimatedValue(8.0)
        assert updated.id == original.id


class TestTranscript:
    def test_rejects_inverted_segment(self) -> None:
        # 終わりが頭より前の字幕を持てると長さが負になり、タイムラインへ投影するときに
        # どこへ出すかが決まらない
        with pytest.raises(ValueError, match="終了が開始より前"):
            TranscriptSegment(start=Fraction(5), end=Fraction(1), text="x")

    def test_rejects_unsorted_segments(self) -> None:
        # 字幕が時刻順に並ばないと、「次の字幕とつなぐ」が並びの隣を相手にし、時刻の
        # 離れた字幕を 1 枚にまとめる
        with pytest.raises(ValueError, match="昇順"):
            Transcript(
                segments=(
                    TranscriptSegment(start=Fraction(5), end=Fraction(6), text="b"),
                    TranscriptSegment(start=Fraction(1), end=Fraction(2), text="a"),
                )
            )

    def test_overlapping_excludes_touching(self, transcript: Transcript) -> None:
        # 1..3 のセグメントに対して 3..4 は接するだけ 重なり扱いしない
        assert list(transcript.overlapping(Fraction(3), Fraction(4))) == []
        assert len(list(transcript.overlapping(Fraction(2), Fraction(5)))) == 2

    def test_with_text_marks_edited(self, transcript: Transcript) -> None:
        # 手で直した印が付かないと、整理の「手で直した字幕には触らない」が効かず直した文を
        # 書き換える ID が変わると差し替え先が見つからず編集が落ちる
        segment = transcript.segments[0]
        updated = segment.with_text("こんにちは")
        assert updated.text == "こんにちは"
        assert updated.edited
        assert updated.id == segment.id
        assert not segment.edited

    def test_replace_segment(self, transcript: Transcript) -> None:
        # 差し替えで隣の字幕まで変わると、1 枚の字幕を直しただけでほかの字幕の本文が消える
        updated = transcript.replace_segment(transcript.segments[1].with_text("差し替え"))
        assert updated.segments[1].text == "差し替え"
        assert updated.segments[0].text == transcript.segments[0].text

    def test_replace_unknown_segment_raises(self, transcript: Transcript) -> None:
        # 知らない字幕を黙って飛ばすと、「字幕を編集」が履歴に残るのに本文は変わらない
        stranger = TranscriptSegment(start=Fraction(0), end=Fraction(1), text="x")
        with pytest.raises(KeyError):
            transcript.replace_segment(stranger)
