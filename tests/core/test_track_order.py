"""トラックの並べ替え（:class:`MoveTrack`）

映像トラックの並びは重ね順そのもの 並びを間違えると、上にあるはずの字幕が絵の下に隠れる
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.core.commands import Document, MoveTrack, reorder_group
from sashimono.core.model import Project, Track, TrackKind


def _project(*tracks: Track) -> Project:
    base = Project.create()
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _names(project: Project) -> list[str]:
    return [track.name for track in project.timeline.tracks]


class TestMoveTrack:
    def test_moves_within_the_same_kind(self) -> None:
        # 行き先がずれると、手前へ動かしたつもりの映像トラックが別の重ね順に入り、字幕が絵に隠れる
        v1, v2, v3 = (Track(TrackKind.VIDEO, name) for name in ("V1", "V2", "V3"))
        project = _project(v1, v2, v3)
        moved = MoveTrack(v1.id, 2).apply(project)
        # 後ろほど手前 V1 を 2 番へ動かすと、いちばん手前（画面のいちばん上）になる
        assert _names(moved) == ["V2", "V3", "V1"]

    def test_other_kinds_stay_where_they_are(self) -> None:
        # 映像と音声が交互に並んだファイルでも、音声の並びは動かない 動くと、
        # 映像を並べ替えただけで音声トラックの番号がずれる
        v1, a1, v2, a2 = (
            Track(TrackKind.VIDEO, "V1"),
            Track(TrackKind.AUDIO, "A1"),
            Track(TrackKind.VIDEO, "V2"),
            Track(TrackKind.AUDIO, "A2"),
        )
        # 番号（ファイルに保存する並び）もそのまま 映像が占めていた席だけが入れ替わる
        # （PR #155 の指摘 抜いて差し込むと A1 が 1 つ後ろへずれていた）
        moved = MoveTrack(v2.id, 0).apply(_project(v1, a1, v2, a2))
        assert _names(moved) == ["V2", "A1", "V1", "A2"]
        # 後ろへ動かしても同じ
        assert _names(MoveTrack(v1.id, 1).apply(_project(v1, a1, v2, a2))) == [
            "V2",
            "A1",
            "V1",
            "A2",
        ]

    def test_a_locked_track_is_refused(self) -> None:
        # ロックしたトラックを動かすと重ね順が変わり、守っているはずの絵が変わる
        v1 = Track(TrackKind.VIDEO, "V1", locked=True)
        v2 = Track(TrackKind.VIDEO, "V2")
        with pytest.raises(ValueError, match="ロック"):
            MoveTrack(v1.id, 1).apply(_project(v1, v2))

    def test_passing_a_locked_track_is_allowed(self) -> None:
        # ロックしたトラックを越えて動かすのまで断ると、ロックが 1 本あるだけで重ね順を変えられない
        v1 = Track(TrackKind.VIDEO, "V1")
        v2 = Track(TrackKind.VIDEO, "V2", locked=True)
        moved = MoveTrack(v1.id, 1).apply(_project(v1, v2))
        assert _names(moved) == ["V2", "V1"]

    def test_an_index_outside_the_group_is_refused(self) -> None:
        # 音声の番号で映像を動かすと、音声の欄のどこかへ映像が紛れ込む
        v1, v2 = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2")
        a1 = Track(TrackKind.AUDIO, "A1")
        with pytest.raises(ValueError, match="範囲"):
            MoveTrack(v1.id, 3).apply(_project(v1, v2, a1))

    def test_staying_put_changes_nothing(self) -> None:
        # 動かない並べ替えで新しい作品を返すと、取り消しても何も起きない段が履歴に積まれる
        v1, v2 = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2")
        project = _project(v1, v2)
        assert MoveTrack(v1.id, 0).apply(project) is project

    def test_it_can_be_undone(self) -> None:
        # 並べ替えを取り消しても戻らないと、重ね順を間違えたときに手で並べ直すことになる
        v1, v2 = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2")
        document = Document(_project(v1, v2))
        document.execute(MoveTrack(v1.id, 1))
        assert _names(document.project) == ["V2", "V1"]
        document.undo()
        assert _names(document.project) == ["V1", "V2"]

    def test_the_group_is_the_same_kind(self) -> None:
        # 種類の規則はここ 1 か所 混ぜたレイヤーへ広げるときはここを差し替える
        v1, a1 = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1")
        project = _project(v1, a1)
        assert reorder_group(project.timeline, v1) == (v1,)
