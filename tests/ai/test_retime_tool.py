"""AI の道具 ``retime_subtitles``（字幕パネルの〔時刻を合わせる…〕と同じ #426）"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sashimono.ai.host import ToolError
from sashimono.core.model import Transcript
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run
from tests.ui.test_subtitle_panel import make_waveform

#: 1.2〜2.8 秒だけ声がある波形 1 つ目の字幕（1〜3 秒）は声より 0.2 秒早く出て遅く消える
VOICE = [(0.0, 225), (0.5, 300), (0.0, 1500)]


def _transcript(host: FakeHost) -> Transcript:
    transcript = host.document.project.media[0].transcript
    assert transcript is not None
    return transcript


def _media(host: FakeHost) -> str:
    return str(host.document.project.media[0].id)


class TestRetimeSubtitles:
    def test_the_line_moves_to_the_voice_and_undoes_in_one_step(self, host: FakeHost) -> None:
        # 壊れると、字幕が声より早く出ると言われても AI は起こし直すしかなく、人の直しが消える
        host.stub_waveform = make_waveform(VOICE)
        before = host.document.project
        result = run(host, "retime_subtitles", media_id=_media(host))
        assert result["moved"] == 1
        first = _transcript(host).segments[0]
        assert float(first.start) == pytest.approx(1.2, abs=0.02)
        assert float(first.end) == pytest.approx(2.8, abs=0.02)
        assert first.text == "今日は"
        host.document.undo()
        assert host.document.project == before

    def test_offsets_without_fitting_need_no_waveform(self, host: FakeHost) -> None:
        # 波形が無くてもずらすだけはできる 断ると、字幕を早めに出したい頼みに応えられない
        result = run(
            host,
            "retime_subtitles",
            media_id=_media(host),
            fit_voice=False,
            start_offset_ms=-200,
            segment_ids=[str(_transcript(host).segments[1].id)],
        )
        assert result["moved"] == 1
        assert result["segments"][0]["start_delta_seconds"] == -0.2
        assert _transcript(host).segments[1].start == Fraction(19, 5)

    def test_fitting_without_a_waveform_is_refused(self, host: FakeHost) -> None:
        # ずらすだけで返すと、声に合わせたつもりの字幕がずれたまま残る
        with pytest.raises(ToolError, match="波形"):
            run(host, "retime_subtitles", media_id=_media(host))

    def test_an_offset_out_of_range_is_refused(self, host: FakeHost) -> None:
        # 黙って丸めると、頼んだのと違う量で字幕が動く
        with pytest.raises(ToolError, match="-1000〜2000"):
            run(
                host, "retime_subtitles", media_id=_media(host), fit_voice=False, end_offset_ms=5000
            )

    def test_an_unknown_line_is_refused(self, host: FakeHost) -> None:
        # 黙って飛ばすと、合わせたつもりの行がずれたまま残る
        with pytest.raises(ToolError, match="見つかりません"):
            run(
                host,
                "retime_subtitles",
                media_id=_media(host),
                fit_voice=False,
                start_offset_ms=-100,
                segment_ids=["無い"],
            )
