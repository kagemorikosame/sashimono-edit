"""配列の引数の中身を黙って寄せない（#320 のレビュー）

配列の中に null や違う型が混ざったとき、既定の値や文字の "None" に読み替えて進めると、
AI は頼んでいない物について判断する 中身を 1 つずつ確かめ、何番目が違うかを返して断る

各試験の頭の 1 行は、壊れたときに利用者に何が起きるか
"""

from __future__ import annotations

from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from tests.ai.conftest import FakeHost


def run(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(host, arguments)


class TestFrames:
    @pytest.mark.parametrize("bad", [None, "120", 1.5, True, [120]])
    def test_a_frame_that_is_not_a_number_is_refused(self, host: FakeHost, bad: object) -> None:
        # 壊れると、[null, 120] の null を 0 と読み、頼んでいない頭のコマを 1 枚目に描く
        with pytest.raises(ToolError, match="frames の 1 個目"):
            run(host, "contact_sheet", frames=[bad, 120])
        assert host.sheets == []

    def test_the_position_of_the_bad_entry_is_told(self, host: FakeHost) -> None:
        # 壊れると、長い一覧のどこが違うのか分からず、AI が全部を作り直す
        with pytest.raises(ToolError, match="3 個目"):
            run(host, "contact_sheet", frames=[10, 20, None])


class TestOtherArrays:
    def test_a_null_clip_id_is_refused(self, host: FakeHost) -> None:
        # 壊れると、null を "None" という ID と読み、見当違いの「見つかりません」を返す
        clip = host.document.project.timeline.tracks[0].clips[0]
        with pytest.raises(ToolError, match="clip_ids の 1 個目"):
            run(host, "select_clips", clip_ids=[None, str(clip.id)])

    def test_a_null_path_is_refused(self, host: FakeHost) -> None:
        # 壊れると、null を "None" という名前のファイルと読み、見当違いの「ありません」を返す
        with pytest.raises(ToolError, match="paths の 1 個目"):
            run(host, "import_media", paths=[None])
        assert host.probed == []

    def test_a_null_segment_id_is_refused(self, host: FakeHost) -> None:
        # 壊れると、null の行が何にも当たらず、頼んだ字幕の一部が黙って置かれない
        with pytest.raises(ToolError, match="segment_ids の 1 個目"):
            run(host, "place_subtitles", segment_ids=[None])

    @pytest.mark.parametrize("bad", [None, 5])
    def test_a_replacement_that_is_not_text_is_refused(self, host: FakeHost, bad: object) -> None:
        # 壊れると、"to": null を空の文字と読み、誤植の言葉を字幕から黙って消す
        before = host.document.project
        with pytest.raises(ToolError, match="to"):
            run(host, "replace_subtitle_text", pairs=[{"from": "テスト", "to": bad}])
        assert host.document.project == before

    def test_a_replacement_left_out_still_means_empty(self, host: FakeHost) -> None:
        # 壊れると（省いた to まで断ると）、言葉を消すだけの直しが頼めなくなる
        result = run(host, "replace_subtitle_text", pairs=[{"from": "存在しない言葉"}])
        assert result["unmatched"] == ["存在しない言葉"]


class TestExplicitNull:
    """配列の引数に null を明示したときは、省いたときと同じに扱わない"""

    def test_null_frames_do_not_draw_the_default_sheet(self, host: FakeHost) -> None:
        # 壊れると、frames に null を渡しただけで全体の 12 コマを描き、AI は渡したつもりの
        # 場面の一覧を見たと思い込む
        with pytest.raises(ToolError, match="frames"):
            run(host, "contact_sheet", frames=None)
        assert host.sheets == []

    @pytest.mark.parametrize("tool", ["delete_clips", "move_clips"])
    def test_null_clip_ids_do_not_fall_back_to_the_selection(
        self, host: FakeHost, tool: str
    ) -> None:
        # 壊れると、clip_ids に null が来ただけで、人が選んでいたクリップを消したり動かしたりする
        clip = host.document.project.timeline.tracks[0].clips[0]
        host.select_clips([clip.id])
        before = host.document.project
        with pytest.raises(ToolError, match="clip_ids"):
            run(host, tool, clip_ids=None, delta=30)
        assert host.document.project == before

    def test_null_segment_ids_do_not_place_every_subtitle(self, host: FakeHost) -> None:
        # 壊れると、segment_ids に null が来ただけで全部の字幕を置き、選んだ行だけのつもりが
        # タイムラインが字幕で埋まる
        before = host.document.project
        with pytest.raises(ToolError, match="segment_ids"):
            run(host, "place_subtitles", segment_ids=None)
        assert host.document.project == before
