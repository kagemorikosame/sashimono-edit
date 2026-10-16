"""書き出した動画のファイルを調べる（#337）

GPU を使わない道なので、GPU の無い CI でも走る 問題の位置が分かっている動画を PyAV で焼き、
それぞれを拾うこと・問題の無い動画で何も拾わないことを見る
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
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
        # 音は圧縮しない PCM で焼く AAC は頭打ちの平らな所を崩し、崩れ方がエンコーダの版で
        # 変わる（CI では音割れの終わりが 0.17 秒手前に来た） ここで見たいのは検査の位置の正しさ
        path = write_video(
            tmp_path / "flawed.mkv",
            12.0,
            black=(2.0, 2.5),
            frozen=(4.0, 7.0),
            tone_db=-14.0,
            silent=(8.0, 10.5),
            clipped=(1.0, 1.2),
            audio_codec="pcm_s16le",
        )
        report = _check(job, path)
        assert not any("圧縮" in note for note in report.notes)
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

    def test_compressed_sound_says_clipping_may_be_missed(
        self, job: CheckJob, tmp_path: Path
    ) -> None:
        # 壊れると、AAC で崩れて拾えなかった音割れを、AI が「音割れ無し」と言い切る
        path = write_video(tmp_path / "aac.mp4", 2.0, tone_db=-14.0)
        report = _check(job, path)
        assert any("圧縮" in note and "タイムライン" in note for note in report.notes)
        only_loudness = _check(job, path, CheckLimits(kinds=frozenset({"loudness"})))
        assert not any("圧縮" in note for note in only_loudness.notes)

    def test_kinds_a_file_cannot_have_are_marked_skipped(
        self, job: CheckJob, tmp_path: Path
    ) -> None:
        # 壊れると、音の無い動画で無音を頼んだ AI が「無音は無い」と伝え、字幕も調べたことになる
        path = write_video(tmp_path / "絵だけ.mp4", 2.0)
        report = _check(
            job, path, CheckLimits(kinds=frozenset({"silence", "black", "text_overflow"}))
        )
        assert report.skipped == frozenset({"silence", "text_overflow"})
        assert report.checked == frozenset({"black"})

    @pytest.mark.parametrize(
        ("audio", "kinds"),
        [(True, {"text_overflow"}), (False, {"silence"}), (False, {"silence", "text_overflow"})],
    )
    def test_nothing_is_read_when_no_asked_stream_exists(
        self,
        job: CheckJob,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio: bool,
        kinds: set[str],
    ) -> None:
        # 壊れると、調べる流れが 1 つも無いのに長い動画を頭から終わりまで読み、
        # 頼んでいない音の大きさまで返す（字幕だけ・音の無い動画で音だけを頼んだとき）
        path = write_video(tmp_path / "読まない.mp4", 2.0, tone_db=-14.0 if audio else None)
        packets: list[object] = []
        real_open = av.open

        class _Counted:
            def __init__(self, container: Any) -> None:
                self._container = container

            def __enter__(self) -> _Counted:
                self._container.__enter__()
                return self

            def __exit__(self, *details: object) -> None:
                self._container.__exit__(*details)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._container, name)

            def demux(self, *streams: Any) -> Iterator[Any]:
                for packet in self._container.demux(*streams):
                    packets.append(packet)
                    yield packet

        monkeypatch.setattr(av, "open", lambda *a, **k: _Counted(real_open(*a, **k)))
        report = _check(job, path, CheckLimits(kinds=frozenset(kinds)))
        assert packets == []
        assert report.peak_db is None
        assert report.checked == frozenset()
        assert report.skipped == frozenset(kinds)

    def test_late_and_missing_sound_follows_the_timestamps(
        self, job: CheckJob, tmp_path: Path
    ) -> None:
        # 壊れると、音が 1 秒遅れて始まる動画の頭の無音と、途中で音のフレームが抜けた所を見落とし、
        # 後ろの問題の時刻も抜けた分だけ手前にずれる
        path = write_video(
            tmp_path / "遅れる音.mkv",
            10.0,
            tone_db=-14.0,
            audio_codec="pcm_s16le",
            audio_delay=1.0,
            audio_gap=(3.0, 5.5),
            clipped=(7.0, 7.2),
        )
        report = _check(
            job,
            path,
            CheckLimits(kinds=frozenset({"silence", "clipping"}), min_silence_seconds=0.5),
        )
        found = _kinds(report)
        # 無音は 50 ms の区切りで測るので、区切りとコマの丸めの分（0.1 秒）まで許す
        assert found["silence"] == [
            pytest.approx((0.0, 1.0), abs=0.1),
            pytest.approx((4.0, 6.5), abs=0.1),
        ]
        (clipping,) = found["clipping"]
        assert clipping == pytest.approx((8.0, 8.2), abs=0.05)

    def test_a_long_held_frame_in_a_vfr_video_is_a_freeze(
        self, job: CheckJob, tmp_path: Path
    ) -> None:
        # 壊れると、VFR の動画で 3 秒出続けるコマを 1 コマと数えて止まりを見落とし、
        # その後の黒の時刻も 3 秒手前にずれる
        path = write_video(tmp_path / "VFR.mkv", 10.0, held=(2.0, 5.0), black=(7.0, 7.5))
        report = _check(job, path, CheckLimits(kinds=frozenset({"black", "freeze"})))
        found = _kinds(report)
        (freeze,) = found["freeze"]
        assert freeze == pytest.approx((2.0, 5.0), abs=0.05)
        (black,) = found["black"]
        assert black == pytest.approx((7.0, 7.5), abs=0.05)
        assert report.seconds == pytest.approx(10.0, abs=0.05)

    def test_sound_longer_than_the_picture_is_black_after_the_last_frame(
        self, job: CheckJob, tmp_path: Path
    ) -> None:
        # 壊れると、映像が 4 秒で終わり音が 6 秒まで続く動画で、後ろの 2 秒（再生では絵が出ない）を
        # 調べないまま、黒いコマは無いと返す
        path = write_video(
            tmp_path / "音が長い.mkv",
            4.0,
            tone_db=-14.0,
            audio_codec="pcm_s16le",
            audio_seconds=6.0,
        )
        report = _check(job, path, CheckLimits(kinds=frozenset({"black", "freeze"})))
        assert report.seconds == pytest.approx(6.0, abs=0.05)
        (black,) = _kinds(report)["black"]
        assert black == pytest.approx((4.0, 6.0), abs=0.05)
        assert report.problems[0].severity == "low"

    @pytest.mark.parametrize(("seconds", "db"), [(3.0, -200.0), (3.0, -80.0), (0.3, -14.0)])
    def test_unmeasurable_loudness_is_not_reported_as_fine(
        self, job: CheckJob, tmp_path: Path, seconds: float, db: float
    ) -> None:
        # 壊れると、無音・静かすぎる・短すぎる動画で音量を測れないのに「問題は無い」と返す
        path = write_video(tmp_path / "測れない.mkv", seconds, tone_db=db, audio_codec="pcm_s16le")
        report = _check(job, path, CheckLimits(kinds=frozenset({"loudness"})))
        assert report.loudness is None
        assert report.skipped == frozenset({"loudness"})
        assert report.checked == frozenset()
        assert any("測れませんでした" in note for note in report.notes)

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
