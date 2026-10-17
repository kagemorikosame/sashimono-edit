"""タイムラインの座標変換

描画・当たり判定・ドラッグがすべてこの変換を通る ここがずれると「見えているのに
掴めない」といった、原因の分かりにくい不具合になる GUI に依存しない計算なので
単体で固める
"""

from __future__ import annotations

import pytest

from sashimono.core.model import MediaItem, Timeline, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.ui.theme import Metrics
from sashimono.ui.timeline.layout import (
    MAX_PIXELS_PER_FRAME,
    MIN_PIXELS_PER_FRAME,
    TimelineLayout,
)
from tests.conftest import make_clip

HEADER = Metrics.TRACK_HEADER_WIDTH


class TestHorizontal:
    def test_frame_zero_sits_at_the_header_edge(self) -> None:
        # ずれると、先頭のクリップがトラックヘッダの下に潜るか、手前に隙間が空いて見える
        layout = TimelineLayout(pixels_per_frame=2.0)
        assert layout.frame_to_x(0) == HEADER

    def test_round_trip(self) -> None:
        # 行きと帰りが合わないと、クリップを掴んで離しただけで数フレームずつ動いてしまう
        layout = TimelineLayout(pixels_per_frame=3.0, scroll_frame=120.0)
        for frame in (0, 1, 137, 10_000):
            assert layout.x_to_frame(layout.frame_to_x(frame)) == pytest.approx(frame)

    def test_scrolling_shifts_the_view(self) -> None:
        # 送った分だけ絵がずれないと、目盛りとクリップが食い違い、狙いと違う所で切ってしまう
        layout = TimelineLayout(pixels_per_frame=2.0)
        scrolled = layout.scrolled_to(50)
        assert scrolled.frame_to_x(50) == HEADER

    def test_frame_at_never_goes_negative(self) -> None:
        # ヘッダの上をクリックしても負のフレームにはしない
        layout = TimelineLayout(pixels_per_frame=2.0)
        assert layout.frame_at(0) == 0
        assert layout.frame_at(HEADER - 40) == 0

    def test_scroll_cannot_go_negative(self) -> None:
        # 負まで送れると、0 フレームより手前の空白が出て、先頭のクリップが右へ逃げる
        assert TimelineLayout(scroll_frame=-100).scroll_frame == 0.0

    def test_frames_in_width(self) -> None:
        # 幅をフレーム数に直し損ねると、画面の右端までクリップが描かれず途中で切れて見える
        layout = TimelineLayout(pixels_per_frame=2.0)
        assert layout.frames_in(HEADER + 200) == 100.0
        # ヘッダより狭いウィンドウでも負にならない
        assert layout.frames_in(10) == 0.0

    def test_visible_range_covers_the_screen(self) -> None:
        # 見えている範囲を狭く見積もると、画面の右の方のクリップや目盛りが描かれずに抜ける
        layout = TimelineLayout(pixels_per_frame=2.0, scroll_frame=30.0)
        start, end = layout.visible_range(HEADER + 200)
        assert start == 30
        assert end >= 130


class TestZoom:
    def test_anchor_frame_stays_put(self) -> None:
        # マウス位置を基準にしないと、拡大するたびに見ていた場所が画面外へ逃げる
        layout = TimelineLayout(pixels_per_frame=2.0, scroll_frame=100.0)
        anchor_x = HEADER + 300.0
        before = layout.x_to_frame(anchor_x)

        zoomed = layout.zoomed(2.5, anchor_x=anchor_x)
        assert zoomed.x_to_frame(anchor_x) == pytest.approx(before, abs=0.01)

    def test_zoom_out_keeps_the_anchor(self) -> None:
        # 縮めたときに基準がずれると、マウスの下で見ていた場所が画面の外へ飛んで探し直す
        layout = TimelineLayout(pixels_per_frame=8.0, scroll_frame=500.0)
        anchor_x = HEADER + 120.0
        before = layout.x_to_frame(anchor_x)
        zoomed = layout.zoomed(0.25, anchor_x=anchor_x)
        assert zoomed.x_to_frame(anchor_x) == pytest.approx(before, abs=0.01)

    def test_clamped_at_both_ends(self) -> None:
        # 倍率に限りが無いと、ホイールを回し続けたときにクリップが線 1 本になるか、
        # 1 コマが画面より広がって戻せなくなる
        assert TimelineLayout(pixels_per_frame=10**9).pixels_per_frame == MAX_PIXELS_PER_FRAME
        assert TimelineLayout(pixels_per_frame=10**-9).pixels_per_frame == MIN_PIXELS_PER_FRAME

    def test_zoom_does_not_scroll_before_the_start(self) -> None:
        # 先頭の近くで縮めたときに 0 より手前へ送られると、先頭のクリップが右へずれて空白が出る
        layout = TimelineLayout(pixels_per_frame=2.0, scroll_frame=0.0)
        zoomed = layout.zoomed(0.5, anchor_x=HEADER + 10.0)
        assert zoomed.scroll_frame >= 0.0


class TestFollowPlayhead:
    def test_no_scroll_while_visible(self) -> None:
        # 見えているのに送ると、再生やクリックのたびにタイムラインが勝手に動いて目で追えない
        layout = TimelineLayout(pixels_per_frame=2.0)
        assert layout.ensure_visible(50, HEADER + 400) is layout

    def test_scrolls_when_the_playhead_runs_off(self) -> None:
        # 追わないと、再生ヘッドが画面の右へ抜けて、今どこを再生しているか見えなくなる
        layout = TimelineLayout(pixels_per_frame=2.0)
        followed = layout.ensure_visible(500, HEADER + 400)
        assert followed.scroll_frame > 0
        # 端ぴったりではなく余裕を持たせる 再生中に細かく折り返すと見づらい
        assert followed.scroll_frame < 500

    def test_scrolls_back_when_the_playhead_moves_left(self) -> None:
        # 左へ戻ったときに追わないと、先頭へ戻しても画面は後ろのままで再生ヘッドが見えない
        layout = TimelineLayout(pixels_per_frame=2.0, scroll_frame=1000.0)
        followed = layout.ensure_visible(10, HEADER + 400)
        assert followed.scroll_frame < 1000


class TestVertical:
    def _timeline(self, video_media: MediaItem) -> Timeline:
        return Timeline(
            rate=FrameRate(30),
            tracks=(
                Track(kind=TrackKind.AUDIO, name="A1", height=50),
                Track(kind=TrackKind.VIDEO, name="V1", height=70),
            ),
        )

    def test_video_sits_above_audio(self, video_media: MediaItem) -> None:
        # Premiere / AviUtl と同じ並び 全トラックを一律に逆順にすると
        # 音声が映像より上へ来てしまう
        timeline = self._timeline(video_media)
        bands = TimelineLayout().bands(timeline)
        assert [band.track.name for band in bands] == ["V1", "A1"]

    def test_track_order_within_each_kind(self) -> None:
        # 映像は番号が大きいほど上、音声は番号が小さいほど上
        timeline = Timeline(
            rate=FrameRate(30),
            tracks=(
                Track(kind=TrackKind.VIDEO, name="V1"),
                Track(kind=TrackKind.VIDEO, name="V2"),
                Track(kind=TrackKind.AUDIO, name="A1"),
                Track(kind=TrackKind.AUDIO, name="A2"),
            ),
        )
        bands = TimelineLayout().bands(timeline)
        assert [band.track.name for band in bands] == ["V2", "V1", "A1", "A2"]

    def test_bands_start_below_the_ruler(self, video_media: MediaItem) -> None:
        # 目盛りの裏にトラックが潜ると、1 本目のトラックの上の方が隠れて掴めない
        bands = TimelineLayout().bands(self._timeline(video_media))
        assert bands[0].top == Metrics.RULER_HEIGHT

    def test_bands_do_not_overlap(self, video_media: MediaItem) -> None:
        # 重なったり隙間が空いたりすると、描いた所と押して当たる所がずれ、
        # 隣のトラックのクリップを掴んでしまう
        bands = TimelineLayout().bands(self._timeline(video_media))
        assert bands[0].bottom == bands[1].top

    def test_band_at(self, video_media: MediaItem) -> None:
        # 高さからトラックを引き違えると、V1 へ落とした素材が A1 へ入る 目盛りや下の空きを
        # 押しただけでも、どれかのトラックを掴んでしまう
        timeline = self._timeline(video_media)
        layout = TimelineLayout()
        top = Metrics.RULER_HEIGHT

        assert layout.band_at(timeline, top + 5) is not None
        assert layout.band_at(timeline, top + 5).track.name == "V1"  # type: ignore[union-attr]
        assert layout.band_at(timeline, top + 75).track.name == "A1"  # type: ignore[union-attr]
        assert layout.band_at(timeline, top + 500) is None
        assert layout.band_at(timeline, 0) is None

    def test_vertical_scroll_moves_the_bands(self, video_media: MediaItem) -> None:
        # 縦に送っても帯が動かないと、トラックが多いときに下の方のトラックへたどり着けない
        timeline = self._timeline(video_media)
        bands = TimelineLayout(scroll_y=30).bands(timeline)
        assert bands[0].top == Metrics.RULER_HEIGHT - 30

    def test_scroll_y_cannot_go_negative(self) -> None:
        # 負まで送れると、目盛りの下に何も無い隙間が出て、1 本目のトラックが下へずれる
        assert TimelineLayout(scroll_y=-50).scroll_y == 0

    def test_content_height(self, video_media: MediaItem) -> None:
        # 全体の高さを小さく見積もると、縦のスクロールが足りず一番下のトラックまで送れない
        timeline = self._timeline(video_media)
        assert TimelineLayout().content_height(timeline) == 50 + 70 + Metrics.RULER_HEIGHT

    def test_track_heights_are_clamped(self) -> None:
        # ファイルに細すぎる高さが入っていてもそのまま使うと、名前とミュートのボタンが
        # 収まらず、クリップも細すぎて掴めない
        timeline = Timeline(
            rate=FrameRate(30),
            tracks=(Track(kind=TrackKind.VIDEO, height=5),),
        )
        assert TimelineLayout().bands(timeline)[0].height == Metrics.MIN_TRACK_HEIGHT

    def test_empty_timeline(self) -> None:
        # トラックが 1 本も無いときに落ちると、新しく作った空のプロジェクトでタイムラインが開けない
        timeline = Timeline(rate=FrameRate(30))
        assert TimelineLayout().bands(timeline) == ()
        assert TimelineLayout().content_height(timeline) == Metrics.RULER_HEIGHT


class TestVisibleClips:
    def test_only_visible_clips_are_returned(self, video_media: MediaItem) -> None:
        # 画面の外のクリップまで描くと、長い作品ほどスクロールやズームが重くなる
        from sashimono.ui.timeline.painter import visible_clips

        timeline = Timeline(
            rate=FrameRate(30),
            tracks=(
                Track(
                    kind=TrackKind.VIDEO,
                    clips=(
                        make_clip(0, 30, video_media),
                        make_clip(10_000, 30, video_media),
                    ),
                ),
            ),
        )
        layout = TimelineLayout(pixels_per_frame=2.0)
        found = visible_clips(timeline, layout, HEADER + 400)
        assert [clip.timeline_start for _, clip, _ in found] == [0]

    def test_clip_rect_is_clipped_to_the_screen(self, video_media: MediaItem) -> None:
        # 長いクリップの矩形を画面の幅で切らないと、左のヘッダの上へはみ出して
        # トラックの名前やボタンを塗り潰す
        from sashimono.ui.timeline.painter import visible_clips

        timeline = Timeline(
            rate=FrameRate(30),
            tracks=(Track(kind=TrackKind.VIDEO, clips=(make_clip(0, 100_000, video_media),)),),
        )
        width = HEADER + 400
        found = visible_clips(timeline, TimelineLayout(pixels_per_frame=2.0), width)
        assert len(found) == 1
        _, _, rect = found[0]
        assert rect.left() >= HEADER
        assert rect.right() <= width
