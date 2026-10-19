"""目印（マーカー）のコマンドと、目印から作るチャプター（#330）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.chapters import chapters_of, clock, youtube_text
from sashimono.core.commands import AddMarkers, Document, RemoveMarkers, UpdateMarker
from sashimono.core.model import MediaItem, Project, Timeline, Track, TrackKind
from sashimono.core.model.timeline import Marker
from sashimono.core.timebase import FrameRate
from tests.conftest import make_clip

RATE = FrameRate(30)


def _project(markers: tuple[Marker, ...] = ()) -> Project:
    base = Project.create()
    return base.with_timeline(replace(base.timeline, markers=markers))


def _frames(project: Project) -> list[int]:
    return [marker.frame for marker in project.timeline.markers]


class TestAddMarkers:
    def test_markers_are_kept_in_frame_order(self) -> None:
        # 並びが乱れると、一覧とチャプターの順が置いた順になり、目次の時刻が前後する
        project = AddMarkers((Marker(300, "後"), Marker(30, "前"))).apply(_project((Marker(90),)))
        assert _frames(project) == [30, 90, 300]

    def test_one_undo_removes_all_of_them(self) -> None:
        # 壊れると、AI が場面ごとに 20 個置いた目印を、人が 20 回取り消すことになる
        document = Document(_project())
        document.execute(AddMarkers(tuple(Marker(frame * 30, f"{frame}") for frame in range(20))))
        document.undo()
        assert document.project.timeline.markers == ()

    def test_the_same_frame_twice_is_refused(self) -> None:
        # 通すと同じ位置に 2 つ並び、位置で指して直す・消すときにどちらか分からない
        with pytest.raises(ValueError, match="もう目印"):
            AddMarkers((Marker(30, "新"),)).apply(_project((Marker(30, "元"),)))

    def test_nothing_is_added_when_one_is_bad(self) -> None:
        # 途中まで置くと、AI は失敗と受け取ってもう一度置き、前半が 2 つずつになる
        project = _project()
        with pytest.raises(ValueError, match="色"):
            AddMarkers((Marker(0, "良"), Marker(30, "悪", "赤"))).apply(project)
        assert project.timeline.markers == ()

    def test_a_negative_frame_is_refused(self) -> None:
        # 通すとタイムラインの頭より前に置かれ、画面にもチャプターにも出ない
        with pytest.raises(ValueError, match="負"):
            AddMarkers((Marker(-1),)).apply(_project())

    @pytest.mark.parametrize("label", ["前半\n後半", "前半\r後半", "前半 後半"])
    def test_a_line_break_inside_the_label_is_refused(self, label: str) -> None:
        # 通すと YouTube の目次の 1 行が 2 行に割れ、後ろの行が時刻の無いただの文になる
        with pytest.raises(ValueError, match="改行"):
            AddMarkers((Marker(0, label),)).apply(_project())
        with pytest.raises(ValueError, match="改行"):
            UpdateMarker(0, Marker(0, label)).apply(_project((Marker(0, "仮"),)))

    def test_the_label_is_trimmed(self) -> None:
        # 空白だけの名前を残すと、名前のある目印として目次に空の行が入る
        project = AddMarkers((Marker(0, "  はじめに \n"),)).apply(_project())
        assert project.timeline.markers[0].label == "はじめに"


class TestUpdateMarker:
    def test_it_moves_and_renames(self) -> None:
        # 壊れると、動かしたつもりの目印が元の位置に残る
        project = UpdateMarker(30, Marker(60, "本編", "#00ff00")).apply(
            _project((Marker(30, "仮"), Marker(90)))
        )
        assert project.timeline.markers[0] == Marker(60, "本編", "#00ff00")

    def test_moving_past_another_keeps_the_order(self) -> None:
        # 並べ直さないと、後ろへ動かした目印が一覧の先頭に残り、目次の時刻が戻る
        project = UpdateMarker(30, Marker(120)).apply(_project((Marker(30), Marker(90))))
        assert _frames(project) == [90, 120]

    def test_moving_onto_another_is_refused(self) -> None:
        # 通すと 2 つが重なり、片方を位置で指せなくなる
        with pytest.raises(ValueError, match="もう目印"):
            UpdateMarker(30, Marker(90)).apply(_project((Marker(30), Marker(90))))

    def test_a_bad_colour_from_a_file_does_not_block_a_rename(self) -> None:
        # 保存形式は色を確かめずに読む 変えていない色まで確かめると、読み込んだ目印の名前を
        # 変えるたびに必ず断られ、画面からも AI からも直せない
        project = _project((Marker(30, "仮", "赤"),))
        renamed = UpdateMarker(30, Marker(30, "本編", "赤")).apply(project)
        assert renamed.timeline.markers == (Marker(30, "本編", "赤"),)

    def test_a_changed_colour_is_still_checked(self) -> None:
        # 変える欄まで確かめを外すと、読めない色を新しく書き込める
        with pytest.raises(ValueError, match="色"):
            UpdateMarker(30, Marker(30, "仮", "青")).apply(_project((Marker(30, "仮", "赤"),)))

    def test_a_missing_marker_is_an_error(self) -> None:
        # 黙って何もしないと、AI は直したつもりで次へ進む
        with pytest.raises(ValueError, match="目印がありません"):
            UpdateMarker(45, Marker(45, "名")).apply(_project((Marker(30),)))


class TestRemoveMarkers:
    def test_only_the_named_frames_go(self) -> None:
        # 2 つ選んで消したのに間の目印まで消えると、その場面のチャプターが目次から黙って抜ける
        project = RemoveMarkers((30, 120)).apply(_project((Marker(30), Marker(90), Marker(120))))
        assert _frames(project) == [90]

    def test_only_the_first_of_a_shared_frame_goes(self) -> None:
        # 手で書いた作品などで同じ位置に 2 つあるとき、位置の全部を消すと、右クリックで指した
        # 1 つ（先頭 marker_at と同じ）のほかに、別の名前の目印まで黙って消える
        project = _project((Marker(30, "甲"), Marker(30, "乙"), Marker(90, "丙")))
        removed = RemoveMarkers((30,)).apply(project)
        assert removed.timeline.markers == (Marker(30, "乙"), Marker(90, "丙"))

    def test_a_frame_named_twice_removes_two(self) -> None:
        # すべて消すときは位置を目印の数だけ渡す 1 つずつ消さないと同じ位置の残りが消えない
        project = _project((Marker(30, "甲"), Marker(30, "乙")))
        assert RemoveMarkers((30, 30)).apply(project).timeline.markers == ()
        with pytest.raises(ValueError, match="30"):
            RemoveMarkers((30, 30, 30)).apply(project)

    def test_a_missing_frame_removes_nothing(self) -> None:
        # 飛ばして残りだけ消すと、古い一覧の位置で頼んだ AI は消えたかを取り違える
        project = _project((Marker(30), Marker(90)))
        with pytest.raises(ValueError, match="45"):
            RemoveMarkers((30, 45)).apply(project)


def _timeline(video_media: MediaItem, frames: int, markers: tuple[Marker, ...]) -> Timeline:
    track = Track(TrackKind.VIDEO, "V1", (make_clip(0, frames, video_media),))
    return Timeline(RATE, (track,), markers)


class TestChapters:
    def test_named_markers_become_chapters_up_to_the_next(self, video_media: MediaItem) -> None:
        # 壊れると、目次の終わりが次の目次の頭とずれ、長さの検査が外れる
        markers = (Marker(0, "はじめに"), Marker(600, "本編"), Marker(1500, "まとめ"))
        result = chapters_of(_timeline(video_media, 1800, markers))
        assert [(c.start, c.end, c.title) for c in result.chapters] == [
            (Fraction(0), Fraction(20), "はじめに"),
            (Fraction(20), Fraction(50), "本編"),
            (Fraction(50), Fraction(60), "まとめ"),
        ]
        assert result.warnings == ()

    def test_unnamed_markers_are_left_out(self, video_media: MediaItem) -> None:
        # 入れると、編集の覚え書きに置いた目印が名前の無い目次の行として出る
        markers = (Marker(0, "はじめに"), Marker(300), Marker(600, "本編"))
        result = chapters_of(_timeline(video_media, 900, markers))
        assert [c.title for c in result.chapters] == ["はじめに", "本編"]
        assert result.chapters[0].end == 20
        assert result.unnamed == 1

    def test_times_count_from_the_export_range(self, video_media: MediaItem) -> None:
        # タイムラインの時刻のまま書くと、範囲を決めて書き出した動画で目次が範囲の頭の分だけ遅れる
        markers = (Marker(0, "外"), Marker(300, "頭"), Marker(900, "中"), Marker(1500, "後ろの外"))
        timeline = replace(_timeline(video_media, 1800, markers), work_area=(300, 1200))
        result = chapters_of(timeline)
        assert [(c.start, c.end, c.title) for c in result.chapters] == [
            (Fraction(0), Fraction(20), "頭"),
            (Fraction(20), Fraction(30), "中"),
        ]
        assert result.outside == 2

    def test_youtube_rules_are_reported_not_fixed(self, video_media: MediaItem) -> None:
        # 黙って 0:00 の行を足したり短い目次をまとめたりすると、本人の付けた名前と位置が食い違う
        markers = (Marker(150, "本編"), Marker(300, "おまけ"))
        result = chapters_of(_timeline(video_media, 900, markers))
        warnings = " ".join(result.warnings)
        assert "0:00" in warnings
        assert "3 行以上" in warnings
        assert "10 秒より短い" in warnings
        assert youtube_text(result) == "0:05 本編\n0:10 おまけ"

    def test_a_first_chapter_shown_as_zero_is_not_warned(self, video_media: MediaItem) -> None:
        # 1 秒未満の所の先頭は概要欄で 0:00 と出て YouTube も目次にする 警告すると、AI が
        # 直さなくてよい目次を直しに行く
        markers = (Marker(15, "はじめに"), Marker(600, "本編"), Marker(1500, "まとめ"))
        result = chapters_of(_timeline(video_media, 1800, markers))
        assert youtube_text(result).startswith("0:00 はじめに")
        assert result.warnings == ()

    def test_unnamed_markers_outside_the_range_count_as_outside(
        self, video_media: MediaItem
    ) -> None:
        # 名前の有無で先に分けると、範囲の外の名前の無い目印が「範囲の外」に数えられず、
        # 書き出した動画に無い目印の数を少なく伝える
        markers = (Marker(0), Marker(300, "頭"), Marker(450), Marker(1500))
        timeline = replace(_timeline(video_media, 1800, markers), work_area=(300, 1200))
        result = chapters_of(timeline)
        assert result.outside == 2
        assert result.unnamed == 1

    def test_a_line_break_read_from_a_file_is_warned(self, video_media: MediaItem) -> None:
        # 保存形式は名前を確かめずに読む 黙って目次に入れると、概要欄の 1 行が割れて目次にならない
        markers = (Marker(0, "前半\n後半"), Marker(600, "本編"), Marker(1500, "まとめ"))
        result = chapters_of(_timeline(video_media, 1800, markers))
        assert any("改行" in warning for warning in result.warnings)

    def test_the_clock_rounds_down_and_shows_hours(self) -> None:
        # 繰り上げると、目次を押したとき場面の頭より後ろから始まり、頭の言葉が欠ける
        assert clock(Fraction(119, 2)) == "0:59"
        assert clock(Fraction(3723)) == "1:02:03"
