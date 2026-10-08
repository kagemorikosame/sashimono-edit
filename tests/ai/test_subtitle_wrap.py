"""AI が焼き込む字幕も、画面の〔焼き込み〕と同じ設定の幅で折り返す（#249）"""

from __future__ import annotations

import pytest

from sashimono.core.model import AnimatedValue, MediaItem
from tests.ai.test_long_results import _call, _host


@pytest.mark.parametrize(("share", "expected"), [(90, 1920 * 0.9), (0, 0.0)])
def test_ai_burned_subtitles_follow_the_setting(
    video_media: MediaItem, share: int, expected: float
) -> None:
    # 頼み方（画面か AI か）で字幕が折り返したり画面の端で切れたりしないように
    host = _host(video_media)
    host.wrap_share = share
    _call(host, "place_subtitles")
    burned = [
        c.source
        for t in host.document.project.timeline.tracks
        for c in t.clips
        if c.source is not None and c.source.kind == "text"
    ]
    assert burned
    for source in burned:
        width = source.params["wrap_width"]
        assert isinstance(width, AnimatedValue)
        assert width.static == pytest.approx(expected)
