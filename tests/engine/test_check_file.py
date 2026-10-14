"""書き出した動画のファイルを調べる（#337）

GPU を使わない道なので、GPU の無い CI でも走る 問題の位置が分かっている動画を PyAV で焼き、
それぞれを拾うこと・問題の無い動画で何も拾わないことを見る
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.engine.checks import CheckJob, CheckLimits, CheckReport, CheckRequest
from tests.check_media import write_video


@pytest.fixture
def job(qt_application: QApplication) -> Iterator[CheckJob]:
    del qt_application
    created = CheckJob()
    yield created
    created.close()


def _check(job: CheckJob, path: Path, limits: CheckLimits | None = None) -> CheckReport:
    stat = path.stat()
    job.start(
        CheckRequest(limits or CheckLimits(), path=path, stamp=(stat.st_size, stat.st_mtime_ns))
    )
    assert job.waiter()(60.0), "60 秒で調べ終わらなかった"
    state = job.state()
    assert state.failure is None, state.failure
    assert state.report is not None
    return state.report


def _kinds(report: CheckReport) -> dict[str, list[tuple[float, float]]]:
    found: dict[str, list[tuple[float, float]]] = {}
    for problem in report.problems:
        found.setdefault(problem.kind, []).append(
            (problem.start / report.fps, problem.end / report.fps)
        )
    return found


class TestWrittenFiles:
    def test_a_clean_video_has_no_problems(self, job: CheckJob, tmp_path: Path) -> None:
        # 壊れると、問題の無い書き出しにまで問題を並べ、AI が直さなくてよい所を直しに行く
        path = write_video(tmp_path / "clean.mp4", 6.0, tone_db=-14.0)
        report = _check(job, path)
        assert report.complete
        assert report.problems == (), report.problems
        assert report.loudness is not None
        assert report.loudness == pytest.approx(-14.0, abs=0.5)
        assert report.seconds == pytest.approx(6.0, abs=0.1)

    def test_each_planted_problem_is_found_where_it_is(self, job: CheckJob, tmp_path: Path) -> None:
        # 壊れると、書き出した動画の黒・止まり・無音・音割れを見落とし、そのまま人へ渡る
        path = write_video(
            tmp_path / "flawed.mp4",
            12.0,
            black=(2.0, 2.5),
            frozen=(4.0, 7.0),
            tone_db=-14.0,
            silent=(8.0, 10.5),
            clipped=(1.0, 1.2),
        )
        report = _check(job, path)
        found = _kinds(report)
        (black,) = found["black"]
        assert black == pytest.approx((2.0, 2.5), abs=0.05)
        (freeze,) = found["freeze"]
        assert freeze == pytest.approx((4.0, 7.0), abs=0.1)
        (silence,) = found["silence"]
        assert silence[0] == pytest.approx(8.0, abs=0.1)
        assert silence[1] == pytest.approx(10.5, abs=0.1)
        (clipping,) = found["clipping"]
        assert clipping[0] == pytest.approx(1.0, abs=0.05)
        assert clipping[1] == pytest.approx(1.2, abs=0.05)

    def test_text_overflow_is_not_claimed_for_a_file(self, job: CheckJob, tmp_path: Path) -> None:
        # 壊れると、字の形の残らない動画を「字幕のはみ出し無し」と言い切り、AI が安心して渡す
        path = write_video(tmp_path / "clean.mp4", 2.0, tone_db=-14.0)
        report = _check(job, path)
        assert any("字幕のはみ出し" in note for note in report.notes)

    def test_only_the_asked_kinds_are_checked(self, job: CheckJob, tmp_path: Path) -> None:
        # 壊れると、音だけを頼んでも絵まで読み、長い動画で待たされる（逆も同じ）
        path = write_video(tmp_path / "flawed.mp4", 6.0, black=(2.0, 3.0), tone_db=-14.0)
        report = _check(job, path, CheckLimits(kinds=frozenset({"silence"})))
        assert report.problems == ()
        report = _check(job, path, CheckLimits(kinds=frozenset({"black"})))
        assert [problem.kind for problem in report.problems] == ["black"]

    def test_a_stopped_check_says_so_and_restarts_when_asked_again(
        self, job: CheckJob, tmp_path: Path
    ) -> None:
        # 壊れると、〔止める〕を押しても走り続けるか、止めた途中の結果を全部調べた物として返す
        path = write_video(tmp_path / "long.mp4", 20.0, tone_db=-14.0)
        stat = path.stat()
        request = CheckRequest(CheckLimits(), path=path, stamp=(stat.st_size, stat.st_mtime_ns))
        job.start(request)
        job.stop()
        assert job.waiter()(30.0)
        state = job.state()
        assert state.stopped
        assert state.report is not None
        assert not state.report.complete
        job.start(request)
        assert job.waiter()(60.0)
        state = job.state()
        assert not state.stopped
        assert state.report is not None
        assert state.report.complete

    def test_an_unreadable_file_fails_with_the_reason(self, job: CheckJob, tmp_path: Path) -> None:
        # 壊れると、読めないファイルで走り係が黙って消え、AI はいつまでも途中を受け取る
        path = tmp_path / "broken.mp4"
        path.write_bytes(b"not a video")
        job.start(CheckRequest(CheckLimits(), path=path, stamp=(1, 1)))
        assert job.waiter()(30.0)
        state = job.state()
        assert not state.running
        assert state.failure is not None
