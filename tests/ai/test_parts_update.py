"""AI の部品が古いときの扱い（#250 の確認で見つかった）

Claude Opus 5.5 を選ぶと、0.2.152 に同梱の Claude Code 2.1.259 では「does not support this
model」で断られた 部品は入れた時点の版のまま上がらないので、新しいモデルが出るたびに
同じことが起きる ここでは次を見る

- 断られた文面を見分け、日本語の案内と更新のボタンを出す
- 入れる部品の下限が、一覧のモデルに足りる版になっている
- PyPI に新しい版があるかを尋ね、範囲の中の版だけを入れる（偽の PyPI）
- 新しい版を別の置き場へ入れてから入れ替え、途中で落ちても今の版を壊さない（偽の pip）
- 自動の更新はセッションの間は待ち、断られたら更新して 1 度だけ送り直す

本物の PyPI・pip・Claude Code は使わない
"""

from __future__ import annotations

import io
import json
import time
import tomllib
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any, ClassVar, cast

import pytest
from packaging.requirements import Requirement
from packaging.version import Version
from PySide6.QtWidgets import QApplication

from sashimono.ai import AI_PACK
from sashimono.ai.environment import MINIMUM_CLAUDE_CODE, REQUIRED_PACKAGES
from sashimono.ai.parts_update import (
    FAILURE_NOTICE_COUNT,
    PartsState,
    PartsStateStore,
    find_update,
    install_staged,
    is_due,
    swap_in,
)
from sashimono.ai.session import (
    SYSTEM_PROMPT,
    UPDATE_PARTS_LABEL,
    AgentEvent,
    EventKind,
    outdated_claude_code,
)
from sashimono.core.model import MediaItem, Project, Transcript
from sashimono.package_index import Latest, latest_release, newest_installable
from sashimono.runtime import ABI_MARKER, FeaturePack, PackageStatus, PackStatus
from sashimono.ui.chat import ChatPanel
from sashimono.ui.chat.parts_updater import PartsUpdater, Phase
from sashimono.ui.setup import SetupSection, describe_updates
from sashimono.ui.workspace import Preferences
from tests.ai.conftest import FakeHost, make_loaded

ROOT = Path(__file__).resolve().parents[2]

#: 利用者の画面に出た文面そのまま
REAL_ERROR = (
    "API Error: 400 Claude Code 2.1.259 does not support this model; version 2.1.280 or newer"
    " is required. Run 'claude update', or update the Claude desktop app, then try again."
)

SDK = "claude-agent-sdk"


# --- 断られた文面 ---


class TestOutdatedMessage:
    def test_the_real_message_is_recognised_with_its_versions(self) -> None:
        # 見分けられないと、英語の「claude update を打て」だけが出て、ソフトの中で
        # 何をすればよいか分からない（同梱の物は claude update では上がらない）
        found = outdated_claude_code(REAL_ERROR)
        assert found is not None
        assert (found.current, found.required) == ("2.1.259", "2.1.280")
        message = found.message()
        assert "AI の部品（Claude Agent SDK に同梱の Claude Code）が古く" in message
        assert f"〔{UPDATE_PARTS_LABEL}〕" in message
        assert "2.1.259" in message and "2.1.280" in message

    def test_without_versions_it_is_still_recognised(self) -> None:
        found = outdated_claude_code("Claude Code does not support this model")
        assert found is not None
        assert "入っている版" not in found.message()

    def test_other_errors_are_left_alone(self) -> None:
        # 取り違えると、ログインの失敗などで部品の更新を勧めてしまう
        assert outdated_claude_code("Invalid API key · Please run /login") is None


# --- 入れる部品の下限 ---


class TestMinimumVersion:
    def test_the_sdk_bundles_a_claude_code_that_knows_every_listed_model(self) -> None:
        """0.2.152 は Claude Code 2.1.259 を同梱し、Claude Opus 5.5 で断られた

        0.2.158 が 2.1.280 を同梱する最初の版（PyPI の Windows の wheel の _cli_version.py で
        確かめた） 下限が下がると、入れたばかりの人が Opus 5.5 を選んだ所で断られる
        """
        (requirement,) = REQUIRED_PACKAGES
        parsed = Requirement(requirement)
        assert parsed.name == SDK
        assert Version("0.2.157") not in parsed.specifier
        assert Version("0.2.158") in parsed.specifier
        assert Version(MINIMUM_CLAUDE_CODE) >= Version("2.1.280")

    def test_untested_major_versions_are_not_taken(self) -> None:
        # 0.3 で使い方が変わっても、自動の更新が入れて会話が始まらなくなることが無いように
        parsed = Requirement(REQUIRED_PACKAGES[0])
        assert Version("0.2.999") in parsed.specifier
        assert Version("0.3.0") not in parsed.specifier

    def test_pyproject_says_the_same(self) -> None:
        # 片方だけ上げると、開発の環境と配布版で入る版が食い違う
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert REQUIRED_PACKAGES[0] in data["project"]["optional-dependencies"]["ai"]


# --- PyPI に尋ねる ---


def _wheel(version: str, tag: str = "win_amd64", *, yanked: bool = False) -> dict[str, object]:
    return {
        "packagetype": "bdist_wheel",
        "filename": f"claude_agent_sdk-{version}-py3-none-{tag}.whl",
        "yanked": yanked,
    }


def _index(*releases: tuple[str, list[dict[str, object]]]) -> dict[str, object]:
    return {"releases": dict(releases)}


#: 実際の PyPI の並びを縮めた物 0.2.160 は Windows の wheel が無かった
INDEX = _index(
    ("0.2.158", [_wheel("0.2.158")]),
    ("0.2.159", [_wheel("0.2.159")]),
    ("0.2.160", [_wheel("0.2.160", "manylinux_2_17_x86_64")]),
    ("0.2.164", [_wheel("0.2.164")]),
    ("0.2.165", [_wheel("0.2.165", yanked=True)]),
    ("0.3.0rc1", [_wheel("0.3.0rc1")]),
    ("0.3.1", [_wheel("0.3.1")]),
)


class TestPackageIndex:
    def test_only_versions_with_a_wheel_for_this_machine_are_chosen(self) -> None:
        # wheel の無い版を勧めると、〔環境を更新〕を押しても入らない
        data = _index(("0.2.159", [_wheel("0.2.159")]), ("0.2.160", [_wheel("0.2.160", "x")]))
        assert newest_installable(data, "win_amd64") == "0.2.159"

    def test_the_range_is_respected(self) -> None:
        from packaging.specifiers import SpecifierSet

        within = newest_installable(INDEX, "win_amd64", SpecifierSet(">=0.2.158,<0.3"))
        assert within == "0.2.164"  # 取り下げた 0.2.165 と前触れの 0.3.0rc1 は選ばない
        assert newest_installable(INDEX, "win_amd64") == "0.3.1"

    def test_asking_reads_the_json_and_names_the_app(self) -> None:
        asked: list[urllib.request.Request] = []

        def opener(request: urllib.request.Request) -> IO[bytes]:
            asked.append(request)
            return io.BytesIO(json.dumps(INDEX).encode())

        latest = latest_release(REQUIRED_PACKAGES[0], opener=opener)
        assert latest is not None
        assert latest.allowed == "0.2.164"
        assert asked[0].full_url == f"https://pypi.org/pypi/{SDK}/json"
        assert "SashimonoEdit/" in str(asked[0].get_header("User-agent"))

    def test_no_network_is_not_an_error(self) -> None:
        def opener(request: urllib.request.Request) -> IO[bytes]:
            raise OSError("繋がらない")

        assert latest_release(REQUIRED_PACKAGES[0], opener=opener) is None


def _status(version: str | None) -> PackStatus:
    return PackStatus(
        pack=AI_PACK, packages=(PackageStatus(REQUIRED_PACKAGES[0], version, "0.2.158"),)
    )


class _Pack:
    """状態だけを返す部品の組 入っている版を思いどおりにする"""

    key = "ai"

    def __init__(self, version: str | None) -> None:
        self.version = version

    def status(self) -> PackStatus:
        return _status(self.version)


def _lookup(allowed: str | None, newest: str | None = None) -> Callable[[str], Latest | None]:
    return lambda _requirement: Latest(allowed, newest or allowed)


class TestFindUpdate:
    def test_a_newer_version_in_range_is_pinned(self) -> None:
        pins = find_update(cast(FeaturePack, _Pack("0.2.158")), _lookup("0.2.164", "0.3.1"))
        assert pins == (f"{SDK}==0.2.164",)

    def test_versions_outside_the_range_are_not_installed(self) -> None:
        # 試していない大きな版上げを自動で入れると、使い方の変わった SDK で会話が始まらない
        pins = find_update(cast(FeaturePack, _Pack("0.2.164")), _lookup("0.2.164", "0.3.1"))
        assert pins == ()

    def test_nothing_is_installed_when_the_pack_is_missing(self) -> None:
        # 入れるかどうかは本人が決める 黙って 250 MB を落とさない
        assert find_update(cast(FeaturePack, _Pack(None)), _lookup("0.2.164")) == ()

    def test_the_settings_text_mentions_versions_out_of_range(self) -> None:
        note, upgradable = describe_updates(
            _status("0.2.164").packages, {REQUIRED_PACKAGES[0]: Latest("0.2.164", "0.3.1")}
        )
        assert upgradable is False
        assert "いちばん新しい版" in note
        assert "0.3.1" in note and "入れません" in note

        note, upgradable = describe_updates(
            _status("0.2.152").packages, {REQUIRED_PACKAGES[0]: Latest("0.2.164", "0.2.164")}
        )
        assert upgradable is True
        assert "更新があります" in note and "0.2.152 → 0.2.164" in note


# --- 別の置き場へ入れてから入れ替える ---


def _write_dist(place: Path, dist: str, module: str, version: str, marker: str) -> None:
    package = place / module
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(f"# {marker}\n", encoding="utf-8")
    info = place / f"{dist}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(f"Name: {dist}\nVersion: {version}\n", encoding="utf-8")
    (info / "RECORD").write_text(
        f"{module}/__init__.py,,\n{info.name}/METADATA,,\n", encoding="utf-8"
    )


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    """今の導入先 古い SDK と、版の変わらない pydantic が入っている"""
    target = tmp_path / "runtime"
    _write_dist(target, "claude_agent_sdk", "claude_agent_sdk", "0.2.152", "old sdk")
    _write_dist(target, "pydantic", "pydantic", "2.0", "loaded pydantic")
    return target


def _fake_pip(code: int = 0) -> Callable[..., int]:
    """pip の代わり ``--target`` の置き場へ、新しい SDK と同じ版の pydantic を置く"""

    def run(arguments: list[str], **_kwargs: Any) -> int:
        staging = Path(arguments[arguments.index("--target") + 1])
        _write_dist(staging, "claude_agent_sdk", "claude_agent_sdk", "0.2.164", "new sdk")
        _write_dist(staging, "pydantic", "pydantic", "2.0", "staged pydantic")
        return code

    return run


def _sdk_marker(target: Path) -> str:
    return (target / "claude_agent_sdk" / "__init__.py").read_text(encoding="utf-8")


class TestStagedInstall:
    def test_the_new_version_replaces_the_old_one(self, runtime: Path) -> None:
        swapped: list[bool] = []
        ok = install_staged(
            (f"{SDK}==0.2.164",),
            target=runtime,
            key="ai",
            run_pip=_fake_pip(),
            before_swap=lambda: swapped.append(True),
        )
        assert ok is True
        assert "new sdk" in _sdk_marker(runtime)
        assert (runtime / "claude_agent_sdk-0.2.164.dist-info").is_dir()
        # 入れ替える直前に会話が畳み終わるのを待つ 待たないと動いている claude.exe を動かせない
        assert swapped == [True]
        # 版の変わらない部品は動かさない 読み込み済みの拡張モジュールは Windows では動かせない
        pydantic = (runtime / "pydantic" / "__init__.py").read_text(encoding="utf-8")
        assert "loaded pydantic" in pydantic
        assert (runtime / ABI_MARKER).is_file()
        assert not (runtime.parent / "runtime-staging").exists()

    def test_a_failed_pip_leaves_the_current_version(self, runtime: Path) -> None:
        # ネットが切れた・Defender に止められた 今の版で使い続けられる
        ok = install_staged(
            (f"{SDK}==0.2.164",), target=runtime, key="ai", run_pip=_fake_pip(code=1)
        )
        assert ok is False
        assert "old sdk" in _sdk_marker(runtime)
        assert not (runtime.parent / "runtime-staging").exists()

    def test_a_failure_while_swapping_puts_everything_back(
        self, runtime: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """入れ替えの途中で落ちても、新旧が混ざった導入先を残さない

        混ざると、今まで動いていた版まで読めなくなる
        """
        staging = runtime.parent / "staging"
        _fake_pip()(["install", "--target", str(staging)])
        original = Path.replace
        calls: list[Path] = []

        def flaky(self: Path, target: Path) -> Path:
            calls.append(self)
            if self.name.endswith(".dist-info") and self.parent == staging:
                raise PermissionError("使用中")
            return original(self, target)

        monkeypatch.setattr(Path, "replace", flaky)
        assert swap_in(staging, runtime) is False
        monkeypatch.setattr(Path, "replace", original)
        assert "old sdk" in _sdk_marker(runtime)
        assert (runtime / "claude_agent_sdk-0.2.152.dist-info").is_dir()
        assert not (runtime / "claude_agent_sdk-0.2.164.dist-info").exists()


# --- いつ確かめるか ---


class TestSchedule:
    def test_once_a_day(self) -> None:
        day = 24 * 60 * 60
        assert is_due(0.0, float(day)) is True  # 確かめたことが無い
        assert is_due(1000.0, 1000.0 + day - 1) is False
        assert is_due(1000.0, 1000.0 + day) is True
        # 時計が戻ったら確かめる 待つと、時計を直すまで何日も確かめなくなる
        assert is_due(5000.0, 1000.0) is True

    def test_a_broken_state_file_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "ai-parts.json"
        path.write_text("{壊れている", encoding="utf-8")
        assert PartsStateStore(path).load() == PartsState()
        PartsStateStore(path).save(PartsState(last_checked=5.0, failures=2))
        assert PartsStateStore(path).load() == PartsState(last_checked=5.0, failures=2)


# --- 画面を固めずに回す ---


class _Installer:
    """入れる物の代わり 呼ばれた引数を覚え、決めた結果を返す"""

    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def __call__(
        self,
        pins: tuple[str, ...],
        *,
        target: Path,
        key: str,
        before_swap: Callable[[], None] | None = None,
    ) -> bool:
        del key
        if before_swap is not None:
            before_swap()
        self.calls.append((tuple(pins), target))
        return self.result


class _Lookup:
    def __init__(self, allowed: str | None) -> None:
        self.allowed = allowed
        self.asked = 0

    def __call__(self, requirement: str) -> Latest | None:
        del requirement
        self.asked += 1
        return Latest(self.allowed, self.allowed)


def _updater(
    tmp_path: Path,
    *,
    installed: str = "0.2.158",
    allowed: str | None = "0.2.164",
    result: bool = True,
    now: float = 10 * 24 * 60 * 60.0,
) -> tuple[PartsUpdater, _Lookup, _Installer]:
    lookup = _Lookup(allowed)
    installer = _Installer(result)
    updater = PartsUpdater(
        cast(FeaturePack, _Pack(installed)),
        store=PartsStateStore(tmp_path / "ai-parts.json"),
        clock=lambda: now,
        lookup=lookup,
        installer=installer,
        target=lambda: tmp_path / "runtime",
    )
    return updater, lookup, installer


def _settle(updater: PartsUpdater, timeout: float = 5.0) -> None:
    """裏の作業が終わるまで見張りの時計の代わりに回す"""
    deadline = time.monotonic() + timeout
    while updater.phase in (Phase.CHECKING, Phase.INSTALLING) and time.monotonic() < deadline:
        work = updater._work
        if work is not None:
            work.wait(0.05)
        updater._poll()


@pytest.mark.usefixtures("qt_application")
class TestPartsUpdater:
    def test_a_new_version_waits_for_the_session_then_installs(self, tmp_path: Path) -> None:
        # 応えている途中で Claude Code を入れ替えると、その応答が途中で切れる
        updater, _, installer = _updater(tmp_path)
        released: list[bool] = []
        updater.prepare = lambda: lambda: released.append(True)
        finished: list[tuple[bool, str]] = []
        updater.finished.connect(lambda ok, pins: finished.append((ok, pins)))

        updater._check(asked=False)
        _settle(updater)
        assert updater.phase is Phase.PENDING
        updater.tick(idle=False)
        assert installer.calls == []

        updater.tick(idle=True)
        assert updater.installing is True
        _settle(updater)
        assert installer.calls == [((f"{SDK}==0.2.164",), tmp_path / "runtime")]
        # 今の会話を畳んでから入れ替える 次の指示から新しい版を使う
        assert released == [True]
        assert finished == [(True, f"{SDK}==0.2.164")]
        updater.stop()

    def test_it_asks_at_most_once_a_day(self, tmp_path: Path) -> None:
        updater, lookup, _ = _updater(tmp_path)
        updater._check(asked=False)
        _settle(updater)
        updater.phase = Phase.IDLE
        updater._check(asked=False)
        assert lookup.asked == 1
        updater.stop()

    def test_turning_it_off_stops_checking(self, tmp_path: Path) -> None:
        # 切ったら確かめも入れ替えもしない（通信を増やしたくない人のため）
        updater, lookup, _ = _updater(tmp_path)
        updater.set_enabled(False)
        updater.schedule()
        updater._check(asked=False)
        assert lookup.asked == 0
        assert updater.request_now() is False
        updater.stop()

    def test_the_development_environment_is_never_touched(self, tmp_path: Path) -> None:
        # 開発の環境（導入先が無い）で走ると、開発者の .venv を黙って書き換える
        updater, lookup, _ = _updater(tmp_path)
        updater._target = lambda: None
        assert updater.available is False
        updater._check(asked=True)
        assert lookup.asked == 0
        updater.stop()

    def test_failures_are_quiet_until_they_repeat(self, tmp_path: Path) -> None:
        # 1 回の失敗（たまたま繋がらない）で見せると、気にしなくてよい人を驚かせる
        updater, _, _ = _updater(tmp_path, result=False)
        updater.prepare = lambda: lambda: None
        for attempt in range(FAILURE_NOTICE_COUNT):
            assert updater.struggling is False, attempt
            updater._check(asked=True)
            _settle(updater)
            updater.tick(idle=True)
            _settle(updater)
        assert updater.struggling is True
        updater.stop()


# --- パネルで ---


class _Session:
    """会話の差し替え Claude へは繋がない"""

    made: ClassVar[list[_Session]] = []

    def __init__(
        self,
        bridge: object,
        *,
        model: str | None,
        effort: str | None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        del bridge, model, effort, system_prompt
        self.prompts: list[str] = []
        self.closed = False
        self.busy = False
        self.model: str | None = None
        self.effort: str | None = None
        _Session.made.append(self)

    def acknowledge_turn(self) -> None:
        return

    def send(self, prompt: str) -> None:
        self.prompts.append(prompt)

    def poll(self) -> list[AgentEvent]:
        return []

    def close(self, *, wait: bool = True) -> None:
        del wait
        self.closed = True

    def wait_closed(self, timeout: float = 30.0) -> None:
        del timeout

    def interrupt(self) -> None:
        return


@pytest.fixture
def loaded(video_media: MediaItem, transcript: Transcript) -> Project:
    return make_loaded(video_media, transcript)


def _make_panel(
    loaded: Project, monkeypatch: pytest.MonkeyPatch, updater: PartsUpdater | None
) -> ChatPanel:
    from sashimono.ui.chat import panel as panel_module

    status = PackStatus(pack=AI_PACK, packages=(PackageStatus(SDK, "0.2.158"),))
    monkeypatch.setattr(SetupSection, "status", property(lambda _self: status))
    monkeypatch.setattr(panel_module, "AgentSession", _Session)
    monkeypatch.setattr(panel_module, "credentials_found", lambda: True)
    _Session.made = []
    return ChatPanel(FakeHost(loaded), updater=updater)


def _text(widget: ChatPanel) -> str:
    return widget._view.toPlainText()


def _send(widget: ChatPanel, prompt: str) -> None:
    widget._input.setPlainText(prompt)
    widget.send()


def _outdated_turn(widget: ChatPanel) -> None:
    widget._handle(AgentEvent(EventKind.READY))
    widget._handle(AgentEvent(EventKind.TEXT, text=REAL_ERROR))
    widget._handle(AgentEvent(EventKind.ERROR, text=REAL_ERROR))
    widget._handle(AgentEvent(EventKind.TURN_DONE))


@pytest.fixture
def unavailable(
    qt_application: QApplication, loaded: Project, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[ChatPanel]:
    """自動で更新できない（開発の環境か、設定で切った）パネル"""
    del qt_application
    updater, _, _ = _updater(tmp_path)
    updater.set_enabled(False)
    widget = _make_panel(loaded, monkeypatch, updater)
    yield widget
    widget.close_session()
    widget.deleteLater()


@pytest.fixture
def automatic(
    qt_application: QApplication, loaded: Project, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[ChatPanel, PartsUpdater, _Installer]]:
    del qt_application
    updater, _, installer = _updater(tmp_path)
    widget = _make_panel(loaded, monkeypatch, updater)
    yield widget, updater, installer
    widget.close_session()
    widget.deleteLater()


def _run_update(widget: ChatPanel, updater: PartsUpdater) -> None:
    """見張りの時計の代わりに回して、確かめ・入れ替えを終わらせる"""
    _settle(updater)
    widget._poll()
    _settle(updater)
    widget._poll()


class TestPanelGuidance:
    def test_the_error_becomes_a_japanese_guide_with_a_button(self, unavailable: ChatPanel) -> None:
        widget = unavailable
        _send(widget, "切って")
        _outdated_turn(widget)
        shown = _text(widget)
        assert "AI の部品（Claude Agent SDK に同梱の Claude Code）が古く" in shown
        # 英語の文面のまま出すと「claude update」を探しに行ってしまう
        assert "claude update" not in shown
        assert shown.count("AI の部品（") == 1  # 文面と失敗の 2 通で 2 度出さない
        assert widget._update_box.isHidden() is False
        assert widget._update_button.text() == UPDATE_PARTS_LABEL
        # 区切りの行は「失敗」のまま
        assert "失敗（" in shown

    def test_the_button_updates_the_parts(
        self, unavailable: ChatPanel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widget = unavailable
        started: list[bool] = []
        monkeypatch.setattr(
            SetupSection, "start", lambda _self, *, upgrade=False: started.append(upgrade)
        )
        _send(widget, "切って")
        _outdated_turn(widget)
        widget._update_button.click()
        # 入っている版のままでも入れ替える（--upgrade） 付けないと名前が在るだけで飛ばされる
        assert started == [True]
        assert _Session.made[0].closed is True
        assert widget._update_box.isHidden() is True
        assert widget._setup.isHidden() is False

    def test_the_setup_section_forces_an_upgrade_when_asked(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del qt_application
        status = PackStatus(pack=AI_PACK, packages=(PackageStatus(SDK, "0.2.158"),))
        monkeypatch.setattr(SetupSection, "status", property(lambda _self: status))
        section = SetupSection(AI_PACK)
        assert "--upgrade" not in section.command_text()
        section._force_upgrade = True
        assert "--upgrade" in section.command_text()
        section.deleteLater()


class TestPanelAutomatic:
    def test_a_refused_prompt_is_sent_again_after_updating(
        self, automatic: tuple[ChatPanel, PartsUpdater, _Installer]
    ) -> None:
        """断られたら部品を新しくして、同じ指示を 1 度だけ送り直す 利用者にはエラーを見せない"""
        widget, updater, installer = automatic
        _send(widget, "切って")
        _outdated_turn(widget)
        assert "AI の部品（" not in _text(widget)
        assert widget._status_text.text().startswith("AI の部品を更新しています…")
        _run_update(widget, updater)

        assert installer.calls, "更新されていない"
        assert "AI の部品を更新して送り直しました" in _text(widget)
        assert _Session.made[0].closed is True
        assert _Session.made[-1].prompts == ["切って"]
        # 送り直した応答が終わって初めて「完了」 区切りは 1 本だけ
        widget._handle(AgentEvent(EventKind.TURN_DONE))
        assert _text(widget).count("──── ") == 1
        assert "完了（" in _text(widget)

    def test_a_failed_update_falls_back_to_the_guide(
        self, automatic: tuple[ChatPanel, PartsUpdater, _Installer]
    ) -> None:
        widget, updater, installer = automatic
        installer.result = False
        _send(widget, "切って")
        _outdated_turn(widget)
        _run_update(widget, updater)
        assert "AI の部品（Claude Agent SDK に同梱の Claude Code）が古く" in _text(widget)
        assert widget._update_box.isHidden() is False
        assert "失敗（" in _text(widget)
        assert widget._status_text.text() == "待機中"

    def test_it_is_sent_again_only_once(
        self, automatic: tuple[ChatPanel, PartsUpdater, _Installer]
    ) -> None:
        # 新しくしても断られるとき（もっと新しい系列が要る）に、更新と送り直しを繰り返さない
        widget, updater, _ = automatic
        _send(widget, "切って")
        _outdated_turn(widget)
        _run_update(widget, updater)
        _outdated_turn(widget)
        assert "AI の部品（Claude Agent SDK に同梱の Claude Code）が古く" in _text(widget)
        assert "失敗（" in _text(widget)

    def test_prompts_sent_while_updating_wait(
        self, automatic: tuple[ChatPanel, PartsUpdater, _Installer]
    ) -> None:
        # 入れ替えの途中で会話を始めると、入れ替える途中の Claude Code を起動してしまう
        widget, updater, _ = automatic
        updater._check(asked=False)
        _settle(updater)
        assert updater.phase is Phase.PENDING
        widget._poll()  # 暇なので入れ始める
        assert updater.installing is True
        _send(widget, "切って")
        assert widget._status_text.text() == "AI の部品を更新しています…・待ち 1 件"
        assert all(not session.prompts for session in _Session.made)
        _settle(updater)
        assert _Session.made[-1].prompts == ["切って"]

    def test_the_setting_turns_off_the_retry(
        self, automatic: tuple[ChatPanel, PartsUpdater, _Installer]
    ) -> None:
        widget, _, installer = automatic
        widget.apply_preferences(Preferences(ai_auto_update=False))
        _send(widget, "切って")
        _outdated_turn(widget)
        assert installer.calls == []
        assert "AI の部品（" in _text(widget)


class TestPreference:
    def test_it_is_on_by_default_and_kept(
        self, tmp_path: Path, qt_application: QApplication
    ) -> None:
        # 既定は入 知らない人が、新しいモデルを選んだ所でエラーに困らない側
        from sashimono.ui.preferences_dialog import PreferencesDialog
        from sashimono.ui.workspace import PreferenceStore

        del qt_application
        assert Preferences().ai_auto_update is True
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(ai_auto_update=False))
        assert store.load().ai_auto_update is False
        dialog = PreferencesDialog(Preferences(ai_auto_update=False))
        assert dialog.preferences().ai_auto_update is False
        dialog.deleteLater()
