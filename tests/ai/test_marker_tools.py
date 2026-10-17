"""AI からマーカーを置く・読む・動かす・消す道具と、チャプターを作る道具（#330）

人の操作（目盛りの右クリック）と同じコマンドを通り、取り消し 1 回で戻ることを見る
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import MAX_RESULT_CHARS, find_operation
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddScene, AddTrack, SetWorkArea, new_scene
from sashimono.core.io.serialize import json_text
from sashimono.core.model import Marker, Project, Track, TrackKind
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run
from tests.conftest import make_clip


def _frames(host: FakeHost) -> list[int]:
    return [marker.frame for marker in host.document.project.timeline.markers]


class TestPlace:
    def test_several_go_in_with_one_undo(self, host: FakeHost) -> None:
        # 壊れると、AI が置いた目印を人が 1 つずつ取り消すことになる
        run(
            host,
            "add_markers",
            markers=[
                {"frame": 90, "label": "本編"},
                {"frame": 0, "label": "はじめに", "color": "#00ff00"},
            ],
        )
        assert _frames(host) == [0, 90]
        assert host.document.project.timeline.markers[0].color == "#00ff00"
        run(host, "undo")
        assert _frames(host) == []

    def test_a_taken_frame_says_how_to_change_it(self, host: FakeHost) -> None:
        # 理由を返さないと、AI は同じ位置へ置き直し続ける
        run(host, "add_markers", markers=[{"frame": 30, "label": "仮"}])
        with pytest.raises(ToolError, match="update_marker"):
            run(host, "add_markers", markers=[{"frame": 30, "label": "本"}])
        assert len(host.document.project.timeline.markers) == 1

    def test_a_bad_colour_is_refused(self, host: FakeHost) -> None:
        # 通すと、保存した作品に読めない色が残り、目盛りに何色で出るかが決まらない
        with pytest.raises(ToolError, match="#RRGGBB"):
            run(host, "add_markers", markers=[{"frame": 0, "color": "red"}])

    @pytest.mark.parametrize("color", [0, False, "", None])
    def test_a_given_colour_is_always_checked(self, host: FakeHost, color: object) -> None:
        # 値の真偽で既定の色へ回すと、渡し間違えた 0 や空の色が黙って黄色の目印になる
        with pytest.raises(ToolError, match="#RRGGBB"):
            run(host, "add_markers", markers=[{"frame": 0, "color": color}])
        assert _frames(host) == []

    @pytest.mark.parametrize("label", [123, None, ["本編"]])
    def test_a_label_must_be_text(self, host: FakeHost, label: object) -> None:
        # 文字にして受けると、数や None が "123" "None" という名前で目次に入る
        with pytest.raises(ToolError, match="文字列"):
            run(host, "add_markers", markers=[{"frame": 0, "label": label}])
        run(host, "add_markers", markers=[{"frame": 0, "label": "仮"}])
        with pytest.raises(ToolError, match="文字列"):
            run(host, "update_marker", frame=0, label=label)
        assert host.document.project.timeline.markers[0].label == "仮"

    def test_it_goes_into_the_open_scene(self, host: FakeHost) -> None:
        # メインへ置くと、シーンを開いて頼んだ目印がシーンに出ない
        scene = new_scene(host.document.project, "場面")
        host.apply_commands([AddScene(scene)], "シーン")
        host.set_active_scene(scene.id)
        run(host, "add_markers", markers=[{"frame": 15, "label": "中"}])
        assert host.document.project.timeline.markers == ()
        assert [m.frame for m in host.project.timeline.markers] == [15]


class TestRead:
    def test_the_list_has_time_and_name(self, host: FakeHost) -> None:
        # 位置しか返さないと、AI は「本編の目印」を名前で探せない
        run(host, "add_markers", markers=[{"frame": 45, "label": "本編"}])
        (row,) = run(host, "list_markers")["markers"]
        assert row["frame"] == 45
        assert row["label"] == "本編"
        assert row["seconds"] == 1.5
        assert row["timecode"] == "00:00:01:15"


class TestChange:
    def test_it_moves_and_renames_with_one_undo(self, host: FakeHost) -> None:
        # 壊れると、動かしたつもりの目印が元の位置に残る
        run(host, "add_markers", markers=[{"frame": 30, "label": "仮"}])
        run(host, "update_marker", frame=30, to_frame=60, label="本編")
        assert host.document.project.timeline.markers == (Marker(60, "本編"),)
        run(host, "undo")
        assert host.document.project.timeline.markers == (Marker(30, "仮"),)

    def test_an_unknown_frame_says_where_to_look(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="list_markers"):
            run(host, "update_marker", frame=30, label="名")

    def test_nothing_to_change_is_an_error(self, host: FakeHost) -> None:
        # 黙って成功を返すと、AI は直したつもりで次へ進む
        run(host, "add_markers", markers=[{"frame": 30}])
        with pytest.raises(ToolError):
            run(host, "update_marker", frame=30)

    def test_delete_takes_several_at_once(self, host: FakeHost) -> None:
        # 番号で消すと 1 つ目を消した時点で後ろがずれる 位置で指せば並びに左右されない
        run(host, "add_markers", markers=[{"frame": f} for f in (0, 30, 60)])
        result = run(host, "delete_markers", frames=[0, 60])
        assert _frames(host) == [30]
        assert result["deleted"] == 2
        run(host, "undo")
        assert _frames(host) == [0, 30, 60]

    def test_delete_all(self, host: FakeHost) -> None:
        run(host, "add_markers", markers=[{"frame": f} for f in (0, 30)])
        run(host, "delete_markers", all=True)
        assert _frames(host) == []

    def test_delete_all_takes_markers_sharing_a_frame(self, loaded: Project) -> None:
        # 手で書いた作品で同じ位置に 2 つあると、位置ごとに 1 つずつ消すだけでは片方が残る
        markers = (Marker(30, "甲"), Marker(30, "乙"))
        host = FakeHost(loaded.with_timeline(replace(loaded.timeline, markers=markers)))
        result = run(host, "delete_markers", all=True)
        assert host.document.project.timeline.markers == ()
        assert result["deleted"] == 2

    def test_all_and_frames_together_is_refused(self, host: FakeHost) -> None:
        # all を先に見ると、位置を指したつもりの AI の頼みで本人の目印が全部消える
        run(host, "add_markers", markers=[{"frame": f} for f in (0, 30)])
        with pytest.raises(ToolError, match="一緒に"):
            run(host, "delete_markers", frames=[0], all=True)
        assert _frames(host) == [0, 30]

    def test_delete_without_a_target_is_an_error(self, host: FakeHost) -> None:
        # 何も渡さずに全部消えると、指定し忘れた AI が本人の目印を丸ごと消す
        run(host, "add_markers", markers=[{"frame": 0}])
        with pytest.raises(ToolError):
            run(host, "delete_markers")
        assert _frames(host) == [0]


class TestChapters:
    def test_the_youtube_text_and_its_warnings(self, host: FakeHost) -> None:
        # 壊れると、AI が渡す目次が YouTube で目次にならないのに、本人は気付かず貼る
        project = host.document.project
        long_clip = make_clip(0, 1800, project.media[0])
        host.apply_commands([AddTrack(Track(TrackKind.VIDEO, "V2", (long_clip,)))], "長い")
        run(
            host,
            "add_markers",
            markers=[
                {"frame": 0, "label": "はじめに"},
                {"frame": 600, "label": "本編"},
                {"frame": 900},
                {"frame": 1500, "label": "まとめ"},
            ],
        )
        result = run(host, "get_chapters")
        assert result["youtube"] == "0:00 はじめに\n0:20 本編\n0:50 まとめ"
        assert result["warnings"] == []
        assert result["unnamed_markers"] == 1
        assert result["chapters"][1] == {
            "frame": 600,
            "start_seconds": 20.0,
            "end_seconds": 50.0,
            "title": "本編",
        }

    def test_the_rules_come_back_as_warnings(self, host: FakeHost) -> None:
        # 足りない所を言わないと、AI は貼れる目次ができたと伝えてしまう
        run(host, "add_markers", markers=[{"frame": 60, "label": "本編"}])
        result = run(host, "get_chapters")
        assert any("0:00" in warning for warning in result["warnings"])

    def test_the_export_range_is_followed(self, host: FakeHost) -> None:
        # 範囲を無視すると、範囲を決めて書き出した動画で目次の時刻がずれる
        host.apply_commands([SetWorkArea((60, 300))], "範囲")
        run(
            host, "add_markers", markers=[{"frame": 0, "label": "外"}, {"frame": 90, "label": "中"}]
        )
        result = run(host, "get_chapters")
        assert result["youtube"] == "0:01 中"
        assert result["outside_markers"] == 1


class TestChapterSize:
    def test_a_long_list_fits_the_result_budget_and_pages(self, loaded: Project) -> None:
        # 長い目次をそのまま返すと、道具の結果の上限を越えて途中で切れ、AI は目次の後ろを読めない
        title = "長い見出し" * 18
        markers = tuple(
            Marker(frame, f"{index} {title}") for index, frame in enumerate(range(0, 300, 1))
        )
        host = FakeHost(loaded.with_timeline(replace(loaded.timeline, markers=markers)))
        first = run(host, "get_chapters")
        assert len(json_text(first)) <= MAX_RESULT_CHARS
        assert first["total"] == 300
        assert first["count"] < 300
        following = run(host, "get_chapters", offset=first["next_offset"])
        assert following["chapters"][0]["frame"] == first["next_offset"]
        # 概要欄に入らない長さの目次は文を返さず、わけを書く
        assert first["youtube"] is None
        assert any("5000" in warning for warning in first["warnings"])

    def test_a_label_with_a_line_break_is_refused(self, host: FakeHost) -> None:
        # 通すと目次の 1 行が割れる 人の右クリックと同じく断る
        with pytest.raises(ToolError, match="改行"):
            run(host, "add_markers", markers=[{"frame": 0, "label": "前半\n後半"}])


class TestSafety:
    @pytest.mark.parametrize(
        ("name", "safety"),
        [
            ("list_markers", Safety.READ_ONLY),
            ("get_chapters", Safety.READ_ONLY),
            ("add_markers", Safety.UNDOABLE),
            ("update_marker", Safety.UNDOABLE),
            ("delete_markers", Safety.UNDOABLE),
        ],
    )
    def test_the_tiers(self, name: str, safety: Safety) -> None:
        # 読むだけの道具が変える側に入ると、目次を読むたびに確かめの窓が出る
        # 変える道具が読む側に入ると、確かめずに本人の目印が消える
        operation = find_operation(name)
        assert operation is not None
        assert operation.safety is safety
