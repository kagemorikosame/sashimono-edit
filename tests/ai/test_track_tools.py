"""AI からのミュート・ソロと解像度

UI にだけ入口を付けると「画面ではできるのに AI に頼むとできない」になる
"""

from __future__ import annotations

import pytest

from sashimono.ai.host import ToolError
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run


class TestTrackState:
    def test_solo_is_set(self, host: FakeHost) -> None:
        # 壊れると「このトラックだけ聞かせて」と頼んでも何も変わらない
        track = host.document.project.timeline.tracks[0]
        run(host, "set_track_state", track_id=str(track.id), solo=True)
        assert host.document.project.timeline.tracks[0].solo

    def test_it_shows_up_in_the_list(self, host: FakeHost) -> None:
        # 一覧に出ないと、AI は切り替えた結果を確かめられず、同じ操作を繰り返す
        track = host.document.project.timeline.tracks[0]
        run(host, "set_track_state", track_id=str(track.id), muted=True, solo=True)
        (listed,) = run(host, "list_tracks")["tracks"]
        assert listed["muted"] is True
        assert listed["solo"] is True

    def test_an_unknown_track_says_where_to_look(self, host: FakeHost) -> None:
        with pytest.raises(ToolError, match="list_tracks"):
            run(host, "set_track_state", track_id="無い", muted=True)

    def test_nothing_to_change_is_an_error(self, host: FakeHost) -> None:
        # 黙って成功を返すと、AI は切り替えたつもりで次へ進む
        track = host.document.project.timeline.tracks[0]
        with pytest.raises(ToolError):
            run(host, "set_track_state", track_id=str(track.id))


class TestResolution:
    def test_it_changes(self, host: FakeHost) -> None:
        # 壊れると「縦動画にして」と頼んでも横のまま書き出される
        run(host, "set_resolution", width=1080, height=1920)
        assert run(host, "get_project")["resolution"] == "1080x1920"

    def test_odd_sizes_come_back_as_a_reason(self, host: FakeHost) -> None:
        # 受け付けてしまうと、書き出しの最後でエンコーダに断られるまで誰も気付かない
        with pytest.raises(ToolError, match="偶数"):
            run(host, "set_resolution", width=1081, height=1920)
