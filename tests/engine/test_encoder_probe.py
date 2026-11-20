"""エンコーダの確かめを別のプロセスで行う（#464）

GPU の駆動が壊れた機械では、NVENC を開くだけで access violation でプロセスごと落ちた
確かめを子のプロセスへ移し、子が落ちても固まってもアプリは生き残って、そのエンコーダを
使えない物として次の候補へ進むことをここで押さえる 壊れた駆動は作れないので、
落ちる・固まる子は :mod:`tests.encoder_probe_jobs` の作り物で代える
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from sashimono.engine import encoders, worker
from sashimono.engine.encode import exporter
from sashimono.engine.worker import WorkerUnavailableError
from tests.encoder_probe_jobs import broken_driver_job


@pytest.fixture(autouse=True)
def forget() -> Iterator[None]:
    """覚えた答えを試験の前後で忘れる 作り物の答えが後の試験へ漏れると本物の NVENC が外れる"""
    encoders.forget_probes()
    yield
    encoders.forget_probes()


@pytest.fixture
def broken_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """子で走る確かめを、名前の頭で落ちる・固まる作り物に替える"""
    monkeypatch.setattr(encoders, "_probe_job", broken_driver_job)


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """起こした子ごとに、渡した名前を記録する 子は本物を起こす"""
    calls: list[list[str]] = []
    real = worker.run_in_process

    def counting(job: Callable[..., Any], kwargs: dict[str, Any], **options: Any) -> Any:
        calls.append(list(kwargs["names"]))
        return real(job, kwargs, **options)

    monkeypatch.setattr(encoders, "run_in_process", counting)
    return calls


class TestHealthyChild:
    def test_the_child_answers_for_each_encoder(self, spawned: list[list[str]]) -> None:
        """普通の機械では 1 つの子で順に確かめ、書ける物だけを返す

        子を候補ごとに起こすと、4 つで 1 秒近く書き出しの窓を待たせる
        """
        names = ["libx264", "no_such_encoder"]
        assert encoders.usable_encoders(names) == ["libx264"]
        assert spawned == [names]
        assert encoders.probe_troubles() == {}

    def test_the_answers_are_kept_until_forgotten(self, spawned: list[list[str]]) -> None:
        """答えは覚えて、書き出しのたびに子を起こさない 忘れたら確かめ直す

        覚えないと、書き出しの窓を開くたびに子を起こす分（0.2 秒ほど）待たされる
        忘れられないと、駆動が直っても閉じるまで GPU へ戻れない
        """
        assert encoders.usable("libx264")
        assert encoders.usable("libx264")
        assert len(spawned) == 1
        encoders.forget_probes()
        assert encoders.usable("libx264")
        assert len(spawned) == 2

    def test_only_unknown_names_are_sent_to_the_child(self, spawned: list[list[str]]) -> None:
        # 覚えた物まで送ると、書き出しと控えで候補が違うたびに NVENC を開き直す
        encoders.usable_encoders(["libx264"])
        encoders.usable_encoders(["libx264", "no_such_encoder"])
        assert spawned == [["libx264"], ["no_such_encoder"]]


class TestBrokenDriver:
    def test_a_crashing_encoder_is_dropped_and_the_rest_are_tried(
        self, broken_driver: None, spawned: list[list[str]]
    ) -> None:
        """子が落ちても親は生き残り、落とした物を外して残りを次の子で確かめる

        親で確かめていた頃は、ここでアプリごと落ちて保存していない編集が消えた
        """
        names = ["crash_nvenc", "libx264"]
        assert encoders.usable_encoders(names) == ["libx264"]
        assert spawned == [names, ["libx264"]]
        troubles = encoders.probe_troubles()
        assert list(troubles) == ["crash_nvenc"]
        assert "落ちた" in troubles["crash_nvenc"]

    def test_answers_before_the_crash_are_kept(self, broken_driver: None) -> None:
        # 落ちる前に返した答えまで捨てると、書ける libx264 を使えない物にする
        assert encoders.usable_encoders(["libx264", "crash_nvenc", "no_such_encoder"]) == [
            "libx264"
        ]
        assert list(encoders.probe_troubles()) == ["crash_nvenc"]

    def test_a_crash_before_naming_blames_the_first_left(self, broken_driver: None) -> None:
        """どれを試していたか知らせる前に落ちたら、残りの先頭を外す

        外さないと、同じ所で落ちる子を起こし続けて確かめが終わらない
        """
        assert encoders.usable_encoders(["vanish_nvenc", "libx264"]) == ["libx264"]
        assert list(encoders.probe_troubles()) == ["vanish_nvenc"]

    def test_a_hanging_encoder_is_given_up(
        self, broken_driver: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """固まった子は時間切れで止め、そのエンコーダを使えない物にする

        待ち続けると、書き出しの窓が開かないまま止まる
        """
        monkeypatch.setattr(encoders, "PROBE_TIMEOUT_S", 0.5)
        monkeypatch.setattr(worker, "CANCEL_GRACE_S", 0.5)
        assert encoders.usable_encoders(["hang_nvenc", "libx264"]) == ["libx264"]
        troubles = encoders.probe_troubles()
        assert list(troubles) == ["hang_nvenc"]
        assert "終わらなかった" in troubles["hang_nvenc"]

    def test_forgetting_clears_the_troubles(self, broken_driver: None) -> None:
        # 残ると、駆動が直って確かめ直した後も「再起動すると直る」の知らせが出続ける
        encoders.usable_encoders(["crash_nvenc"])
        encoders.forget_probes()
        assert encoders.probe_troubles() == {}

    def test_the_notice_says_why_and_how_to_recover(self, broken_driver: None) -> None:
        """知らせには、戻したことと、再起動で直りうることと、確かめ直す場所を書く

        理由が無いと、GPU で書いたつもりの人が遅さの原因を探せない
        """
        encoders.usable_encoders(["crash_nvenc", "libx264"])
        told = encoders.trouble_notice(["crash_nvenc", "libx264"])
        assert told is not None
        assert "crash_nvenc" in told
        assert "再起動" in told
        assert "確かめ直す" in told
        assert encoders.trouble_notice(["libx264"]) is None


class TestNoChild:
    def test_without_a_child_the_probe_runs_here(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """子を起こせない機械では、この中で確かめる

        すべて使えない物にすると、見張りのソフトが子を止める普通の機械で書き出せなくなる
        """

        def refuse(*_args: Any, **_options: Any) -> None:
            raise WorkerUnavailableError("起こせない")

        monkeypatch.setattr(encoders, "run_in_process", refuse)
        assert encoders.usable_encoders(["libx264", "no_such_encoder"]) == ["libx264"]

    def test_after_a_crash_an_unstartable_child_drops_the_rest(
        self, broken_driver: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """落ちた後に子を起こせなくなったら、残りの GPU の物はこの中で試さず使えない物にする

        この中で試すと、子を落とした駆動を読んで今度はアプリごと落ちうる
        CPU の物は駆動を読まないのでこの中で試す 外すと書き出せる物が 1 つも残らない
        """
        real = worker.run_in_process
        calls = [0]

        def once(*args: Any, **options: Any) -> Any:
            calls[0] += 1
            if calls[0] > 1:
                raise WorkerUnavailableError("起こせない")
            return real(*args, **options)

        monkeypatch.setattr(encoders, "run_in_process", once)
        assert encoders.usable_encoders(["crash_nvenc", "hevc_nvenc", "libx264"]) == ["libx264"]
        assert set(encoders.probe_troubles()) == {"crash_nvenc", "hevc_nvenc"}


class TestExport:
    def test_a_named_encoder_that_crashed_falls_back_to_the_candidates(
        self, broken_driver: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """名指しした GPU のエンコーダが確かめで落ちたら、自動の候補で書き出す

        名指しのままこのプロセスで開くと、確かめで子を落とした駆動でアプリごと落ちる
        """
        monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("crash_nvenc", "libx264"))
        named = exporter.ExportSettings(path=Path("x.mp4"), video_codec="crash_nvenc")
        assert exporter.export_candidates(named) == ["libx264"]
        told = exporter._probe_notice(named, "libx264")
        assert told is not None and "crash_nvenc" in told

    def test_a_named_cpu_encoder_is_not_probed(self, spawned: list[list[str]]) -> None:
        # CPU の物は駆動に関わらない 確かめると、名指しの書き出しのたびに子を起こす分待たされる
        settings = exporter.ExportSettings(path=Path("x.mp4"), video_codec="libx264")
        assert exporter.export_candidates(settings) == ["libx264"]
        assert spawned == []

    def test_no_notice_when_the_trouble_was_behind_the_used_one(
        self, broken_driver: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CPU を先にと選んだ人には、後ろで落ちた NVENC は書き出しに関わらない 知らせると要らない注意
        monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("crash_nvenc", "libx264"))
        encoders.usable_encoders(["crash_nvenc", "libx264"])
        settings = exporter.ExportSettings(
            path=Path("x.mp4"),
            encoder_priority=encoders.PRIORITY_CPU,
        )
        assert exporter._probe_notice(settings, "libx264") is None


class TestExportChild:
    def test_the_answers_reach_the_export_child(self, broken_driver: None) -> None:
        """親が覚えた答えと落ちた理由を、書き出しの子へ写せる

        写さないと、子が書き出しのたびに孫を起こして確かめ直し、知らせから
        「再起動で直る」が消える
        """
        encoders.usable_encoders(["crash_nvenc", "libx264"])
        memory = encoders.remembered()
        encoders.forget_probes()
        encoders.remember(memory)
        assert encoders.usable_encoders(["crash_nvenc", "libx264"]) == ["libx264"]
        assert list(encoders.probe_troubles()) == ["crash_nvenc"]

    def test_the_export_passes_the_answers_to_the_child(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 渡し忘れると、上の写しの口があっても子は何も覚えていない
        from sashimono.core.model import Project
        from sashimono.engine.encode import export_job

        encoders.remember(encoders.ProbeMemory(verdicts=(("libx264", True),)))
        seen: dict[str, Any] = {}

        def fake(_job: Callable[..., Any], kwargs: dict[str, Any], **_options: Any) -> None:
            seen.update(kwargs)

        monkeypatch.setattr(export_job, "run_in_process", fake)
        with pytest.raises(exporter.ExportError):
            export_job.export_anywhere(
                # 子は起こさないので中身は要らない
                cast("Project", object()),
                exporter.ExportSettings(path=tmp_path / "x.mp4"),
                separate_process=True,
            )
        # 親が書き出しの前に確かめた分も足されるので、渡した答えが含まれているかを見る
        assert ("libx264", True) in seen["probes"].verdicts

    def test_the_parent_probes_before_starting_the_export_child(
        self, broken_driver: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """まだ確かめていない候補は、書き出しの子を起こす前に親で確かめる

        書き出しの子は daemon で孫を起こせず、知らない物を子の中で試す 壊れた駆動なら
        書き出しの子ごと落ち、CPU へ戻れずに書き出しが失敗する（PR #492 の CodeRabbit の指摘）
        """
        from sashimono.core.model import Project
        from sashimono.engine.encode import export_job

        monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("crash_nvenc", "libx264"))
        seen: dict[str, Any] = {}

        def fake(_job: Callable[..., Any], kwargs: dict[str, Any], **_options: Any) -> None:
            seen.update(kwargs)

        monkeypatch.setattr(export_job, "run_in_process", fake)
        with pytest.raises(exporter.ExportError):
            export_job.export_anywhere(
                cast("Project", object()),
                exporter.ExportSettings(path=tmp_path / "x.mp4"),
                separate_process=True,
            )
        memory: encoders.ProbeMemory = seen["probes"]
        assert dict(memory.verdicts) == {"crash_nvenc": False, "libx264": True}
        assert "crash_nvenc" in dict(memory.troubles)


class TestNamedCpu:
    def test_a_cpu_export_does_not_probe_the_gpus(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """CPU を名指しした書き出しは、子を起こせない機械でも GPU をこの中で試さない

        試すと、CPU を選んだのに壊れた駆動でアプリごと落ちうる
        （PR #492 の CodeRabbit の 3 回目の指摘）
        """
        from sashimono.core.model import Project
        from sashimono.engine.encode import export_job

        def refuse(*_args: Any, **_options: Any) -> None:
            raise WorkerUnavailableError("起こせない")

        def here(name: str) -> bool:
            pytest.fail(f"CPU を名指しした書き出しが {name} をこの中で試した")

        seen: dict[str, Any] = {}

        def fake(_job: Callable[..., Any], kwargs: dict[str, Any], **_options: Any) -> None:
            seen.update(kwargs)

        monkeypatch.setattr(encoders, "run_in_process", refuse)
        monkeypatch.setattr(encoders, "_probe_here", here)
        monkeypatch.setattr(export_job, "run_in_process", fake)
        with pytest.raises(exporter.ExportError):
            export_job.export_anywhere(
                cast("Project", object()),
                exporter.ExportSettings(path=tmp_path / "x.mp4", video_codec="libx264"),
                separate_process=True,
            )
        assert seen["probes"] == encoders.ProbeMemory()


class TestRaces:
    def test_the_verdict_and_its_reason_come_together(self, broken_driver: None) -> None:
        # 別々に問うと、間に確かめ直しが割り込んで理由だけ消え、落ちた物をこの中で開きにいく
        ok, trouble = encoders.verdict("crash_nvenc")
        assert not ok
        assert trouble is not None and "落ちた" in trouble
        assert encoders.verdict("no_such_encoder") == (False, None)

    def test_a_running_probe_is_not_waited_for_by_the_window(self) -> None:
        """ほかの確かめが走っている間は、書き出しの窓の問いは待たずに ``None`` を返す

        待つと、固まった駆動を待ち切るまで書き出しの窓が開かない
        """
        encoders.remember(encoders.ProbeMemory(verdicts=(("libx264", True),)))
        with encoders._probe_lock:
            assert encoders.usable_encoders_now(["libx264"]) is None
        assert encoders.usable_encoders_now(["libx264"]) == ["libx264"]


class TestStartup:
    def test_without_a_child_the_startup_probe_does_nothing_here(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """起動の後の確かめは、子を起こせなければこの中で試さずに止める

        この中で試すと、書き出しを押してもいないのに壊れた駆動で編集画面ごと落ちる
        """

        def refuse(*_args: Any, **_options: Any) -> None:
            raise WorkerUnavailableError("起こせない")

        def here(name: str) -> bool:
            pytest.fail(f"起動の後の確かめが {name} をこの中で試した")

        monkeypatch.setattr(encoders, "run_in_process", refuse)
        monkeypatch.setattr(encoders, "_probe_here", here)
        encoders.probe_in_children(["h264_nvenc", "libx264"])
        assert encoders.remembered() == encoders.ProbeMemory()

    def test_the_startup_probe_remembers_the_answers(self, spawned: list[list[str]]) -> None:
        # 覚えないと、書き出しの窓を初めて開いたときに改めて子を起こして待たせる
        encoders.probe_in_children(["libx264"])
        encoders.probe_in_children(["libx264"])
        assert spawned == [["libx264"]]
        assert encoders.usable_encoders_now(["libx264"]) == ["libx264"]


def test_the_self_check_starts_the_probe_child() -> None:
    """自己診断が、エンコーダの確かめの子を起こして答えを受けられる

    子を起こせない配布版は確かめをアプリの中へ戻し、駆動の壊れた機械でアプリごと落ちる
    書き出しの項目はそれでも通るので、子の道をこの項目で別に見る
    """
    from sashimono import selfcheck

    assert "libx264" in selfcheck._encoder_probe()


def test_no_notice_when_the_named_encoder_was_used(
    broken_driver: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """名指しした物で書けたら、ほかの GPU の不調は知らせない

    名指しで libx264 を選んだ人に NVENC の不調を知らせても、書き出しは何も変わっていない
    """
    monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("crash_nvenc", "libx264"))
    encoders.usable_encoders(["crash_nvenc", "libx264"])
    settings = exporter.ExportSettings(path=Path("x.mp4"), video_codec="libx264")
    assert exporter._probe_notice(settings, "libx264") is None


def test_a_recheck_cannot_slip_between_probing_and_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """書き出しの子へ渡す写しは、確かめと同じ錠の中で取る

    間に確かめ直しが割り込むと空の写しが子に渡り、子が自分の中で GPU を試して落ちる
    （PR #492 の CodeRabbit の 2 回目の指摘） 確かめの最中に確かめ直しを始めさせ、
    写しを取り終えるまで待たされることを見る
    """
    import threading

    real = encoders._usable_locked
    blocked: list[bool] = []
    # 確かめ直しのスレッドは錠を放した後で待ち切る 錠の中で待ち切れないまま試験を終えると、
    # 遅れて走った確かめ直しが次の試験の覚えた答えを消す（CodeRabbit の 3 回目の指摘）
    rechecks: list[threading.Thread] = []

    def racing(wanted: list[str]) -> list[str]:
        found = real(wanted)
        recheck = threading.Thread(target=encoders.forget_probes)
        rechecks.append(recheck)
        recheck.start()
        recheck.join(0.2)
        blocked.append(recheck.is_alive())
        return found

    monkeypatch.setattr(encoders, "_usable_locked", racing)
    try:
        memory = encoders.probe_and_remember(["libx264"])
    finally:
        for recheck in rechecks:
            recheck.join()
    assert blocked == [True], "確かめ直しが写しより先に答えを消せた"
    assert ("libx264", True) in memory.verdicts


def test_the_startup_probe_tries_nothing_here_after_a_crash(
    broken_driver: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """起動の後の確かめは、子が落ちた後に起こし直せなくても、CPU の物までこの中で試さない

    起動の後の確かめは子の中だけに限る（PR #492 の CodeRabbit の 4 回目の指摘）
    残った CPU の物は答えの無いまま残し、書き出しや控えで問われたときに確かめる
    """
    real = worker.run_in_process
    calls = [0]

    def once(*args: Any, **options: Any) -> Any:
        calls[0] += 1
        if calls[0] > 1:
            raise WorkerUnavailableError("起こせない")
        return real(*args, **options)

    def here(name: str) -> bool:
        pytest.fail(f"起動の後の確かめが {name} をこの中で試した")

    monkeypatch.setattr(encoders, "run_in_process", once)
    monkeypatch.setattr(encoders, "_probe_here", here)
    encoders.probe_in_children(["crash_nvenc", "hevc_nvenc", "libx264"])
    memory = encoders.remembered()
    assert dict(memory.verdicts) == {"crash_nvenc": False, "hevc_nvenc": False}
    assert set(dict(memory.troubles)) == {"crash_nvenc", "hevc_nvenc"}
