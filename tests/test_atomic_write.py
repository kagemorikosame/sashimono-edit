"""書いてから置き換える書き込み（``core.io.atomic``）と、それを使う保存の場所

ここが落ちると、2 つの窓が同じファイルを同時に書いたときに、片方の保存が黙って落ちるか、
書きかけ同士が混ざった中身で設定やプロジェクトが置き換わる
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from sashimono.ai.parts_update import PartsState, PartsStateStore
from sashimono.core.io import RecoverySession, atomic, discard, find_orphans, save_project
from sashimono.core.io.aliases import Alias, AliasStore
from sashimono.core.io.atomic import STALE_WRITING_SECONDS, held_for_update, write_atomic
from sashimono.core.io.locks import try_hold
from sashimono.core.io.presets import Preset, PresetStore
from sashimono.core.io.recovery import remember_root
from sashimono.core.model import Blending, Clip, Project
from sashimono.core.timebase import FrameRate
from sashimono.core.userdirs import _can_take
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.ui.project_presets import ProjectPreset, ProjectPresetStore
from sashimono.ui.workspace import Preferences, PreferenceStore, ShortcutStore
from sashimono.update.state import UpdateState, UpdateStateStore

#: 書き手ごとの仮のファイルの名前 ``<名前>.<印 8 桁>.writing``
_SCRATCH = re.compile(r"(?P<name>.+)\.[0-9a-f]{8}\.writing")


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """試験の途中で窓（尋ねる・知らせる・ファイルを選ぶ）が開いたら、その場で落とす

    開くと、閉じる人がいないまま試験が止まる 保存の失敗を窓で知らせる道に入ったことも分かる
    """

    def opened(*_args: object, **_kwargs: object) -> None:
        pytest.fail("試験の途中で窓が開いた")

    monkeypatch.setattr(QDialog, "exec", opened)
    monkeypatch.setattr(QMessageBox, "exec", opened)
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, opened)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, opened)


@pytest.fixture
def scratch_names(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """名前を付け替えた（仮のファイル, 行き先）の組を記録する"""
    seen: list[tuple[str, str]] = []
    plain_replace = Path.replace

    def replace(self: Path, target: str | os.PathLike[str]) -> Path:
        seen.append((self.name, Path(target).name))
        return plain_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    return seen


def _aged(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


class TestEachWriterHasItsOwnScratchFile:
    def test_two_writes_of_one_file_use_different_scratch_names(
        self, tmp_path: Path, scratch_names: list[tuple[str, str]]
    ) -> None:
        # 壊れると、2 つの窓が同じ仮のファイルを取り合い、片方の書きかけをもう片方が置き換えに使う
        target = tmp_path / "preferences.json"
        write_atomic(target, "1")
        write_atomic(target, "2")
        (first, _), (second, _) = scratch_names
        assert first != second
        for name in (first, second):
            match = _SCRATCH.fullmatch(name)
            assert match is not None and match["name"] == target.name

    def test_writers_at_the_same_time_leave_one_whole_content(self, tmp_path: Path) -> None:
        # 壊れると、片方の保存が「見つからない・使用中」で落ちるか、2 人の書きかけが混ざって残る
        target = tmp_path / "本編.sme"
        payloads = {who: who * 200_000 for who in "AB"}
        errors: list[BaseException] = []
        start = threading.Barrier(2)

        def run(who: str) -> None:
            start.wait()
            try:
                for _ in range(30):
                    write_atomic(target, payloads[who])
            except BaseException as exc:  # 何で落ちたかを試験の結果に出す
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(who,)) for who in payloads]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert errors == []
        assert target.read_text(encoding="utf-8") in payloads.values()
        assert [path.name for path in tmp_path.iterdir()] == [target.name]


#: 名前の上限に近い名前 ASCII（Windows の字数）・日本語（Linux などのバイト数）・字の外の文字
_LONG_NAMES = ["a" * 250 + ".sme", "長い題の動画" * 13 + ".sme", "🎬" * 120 + ".sme"]


class TestLongNames:
    @pytest.mark.parametrize("name", _LONG_NAMES)
    def test_the_scratch_name_fits_the_limit_of_a_file_name(
        self, tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、題の長いプロジェクトや設定は仮のファイルを作れず、保存そのものができない
        opened: list[str] = []

        def record(self: Path, *_args: object, **_kwargs: object) -> None:
            opened.append(self.name)
            raise OSError("名前だけを見る")

        monkeypatch.setattr(Path, "open", record)
        with pytest.raises(OSError, match="名前だけを見る"):
            write_atomic(tmp_path / name, "{}")
        (scratch,) = opened
        assert len(scratch.encode("utf-8")) <= 255
        assert len(scratch.encode("utf-16-le")) // 2 <= 255
        assert scratch.endswith(".writing")

    @pytest.mark.parametrize("name", _LONG_NAMES)
    def test_a_shortened_scratch_is_still_known_as_a_scratch(
        self, tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、切った名前の書きかけが片付けから漏れて積もるか、引き継ぎで完成品として移る
        opened: list[str] = []

        def record(self: Path, *_args: object, **_kwargs: object) -> None:
            opened.append(self.name)
            raise OSError("名前だけを見る")

        monkeypatch.setattr(Path, "open", record)
        with pytest.raises(OSError):
            write_atomic(tmp_path / name, "{}")
        (scratch,) = opened
        assert atomic._is_leftover(scratch, name)
        assert not atomic._is_leftover(scratch, "別の" + name)
        assert not _can_take(tmp_path / scratch, "recovery")

    @pytest.mark.parametrize("name", _LONG_NAMES)
    def test_a_scratch_of_another_long_name_with_the_same_head_is_not_taken(
        self, tmp_path: Path, name: str
    ) -> None:
        # 壊れると、頭の同じ別の長い名前のファイルを保存したときに、こちらの書きかけを消す
        other = name[:-4] + "-別.sme"
        mine = atomic._scratch_head(name)
        assert mine != atomic._scratch_head(other)
        scratch = f"{mine}.0a1b2c3d.writing"
        assert atomic._is_leftover(scratch, name)
        assert not atomic._is_leftover(scratch, other)
        assert len(scratch.encode("utf-8")) <= 255

    def test_a_short_name_keeps_the_whole_name(self) -> None:
        # 壊れると、書きかけの名前から元のファイルが分からず、手で片付けるときに迷う
        assert atomic._scratch_head("preferences.json") == "preferences.json"


class TestAFailedWriteLeavesTheOldFile:
    def test_a_failure_while_writing_keeps_the_old_file_and_no_scratch(
        self, tmp_path: Path
    ) -> None:
        # 壊れると、書けなかった保存のたびに書きかけが積もるか、前の中身が半分の物に置き換わる
        target = tmp_path / "preferences.json"
        target.write_text("前の中身", encoding="utf-8")
        with pytest.raises(UnicodeEncodeError):
            write_atomic(target, "途中まで" * 1000 + "\ud800")
        assert target.read_text(encoding="utf-8") == "前の中身"
        assert [path.name for path in tmp_path.iterdir()] == [target.name]

    def test_a_refused_replace_keeps_the_old_file_and_no_scratch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、置き換えを断られた（ドライブを抜いた など）保存の書きかけが残り続ける
        target = tmp_path / "preferences.json"
        target.write_text("前の中身", encoding="utf-8")

        def refuse(self: Path, target: Path) -> Path:
            raise OSError("置き換えられない")

        monkeypatch.setattr(Path, "replace", refuse)
        with pytest.raises(OSError, match="置き換えられない"):
            write_atomic(target, "新しい中身")
        assert target.read_text(encoding="utf-8") == "前の中身"
        assert [path.name for path in tmp_path.iterdir()] == [target.name]

    def test_a_moment_of_in_use_is_waited_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、ほかの窓が同じファイルへ名前を付けている一瞬に当たった保存が黙って落ちる
        target = tmp_path / "preferences.json"
        plain_replace = Path.replace
        refused: list[str] = []

        def busy_once(self: Path, destination: Path) -> Path:
            if not refused:
                refused.append(self.name)
                raise PermissionError("使用中")
            return plain_replace(self, destination)

        monkeypatch.setattr(Path, "replace", busy_once)
        write_atomic(target, "新しい中身")
        assert refused
        assert target.read_text(encoding="utf-8") == "新しい中身"

    def test_a_file_that_stays_in_use_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、ほかの道具が掴んだままのファイルへの保存で、いつまでも待って画面が止まる
        target = tmp_path / "preferences.json"
        tries: list[str] = []

        def busy(self: Path, destination: Path) -> Path:
            tries.append(self.name)
            if len(tries) > 1000:
                # 待つのに限りが無いと、ここで試験ごと止まる 止まる代わりに落とす
                raise RuntimeError("いつまでも試し直している")
            raise PermissionError("使用中")

        monkeypatch.setattr(Path, "replace", busy)
        monkeypatch.setattr(atomic, "_REPLACE_WAIT", 0.05)
        with pytest.raises(PermissionError):
            write_atomic(target, "新しい中身")
        assert list(tmp_path.iterdir()) == []

    def test_a_durable_write_reaches_the_disk_before_the_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、保存の直後に電源が切れたとき、プロジェクトが中身の無いファイルに置き換わる
        order: list[str] = []
        plain_fsync = os.fsync
        plain_replace = Path.replace

        def fsync(descriptor: int) -> None:
            order.append("fsync")
            plain_fsync(descriptor)

        def replace(self: Path, destination: Path) -> Path:
            order.append("replace")
            return plain_replace(self, destination)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(Path, "replace", replace)
        save_project(Project.create(), tmp_path / "本編.sme")
        assert order == ["fsync", "replace"]


class TestLeftoversAreTidied:
    def test_an_old_scratch_of_a_crashed_writer_goes_on_the_next_write(
        self, tmp_path: Path
    ) -> None:
        # 壊れると、名前が書き手ごとに違うので、落ちた分の書きかけが保存のたびに積もる
        target = tmp_path / "preferences.json"
        crashed = tmp_path / "preferences.json.0a1b2c3d.writing"
        legacy = tmp_path / "preferences.json.writing"
        for leftover in (crashed, legacy):
            leftover.write_text("{", encoding="utf-8")
            _aged(leftover, atomic.STALE_WRITING_SECONDS + 60)
        write_atomic(target, "{}")
        assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]

    def test_a_fresh_scratch_of_another_writer_is_left_alone(self, tmp_path: Path) -> None:
        # 壊れると、別の窓が今まさに書いている仮のファイルを消し、その窓の保存が落ちる
        target = tmp_path / "本編.sme"
        writing = tmp_path / "本編.sme.0a1b2c3d.writing"
        writing.write_text("{", encoding="utf-8")
        write_atomic(target, "{}")
        assert writing.is_file()

    def test_files_of_other_shapes_are_left_alone(self, tmp_path: Path) -> None:
        # 壊れると、本人が同じフォルダに置いた似た名前のファイルや、ほかのファイルの書きかけを消す
        target = tmp_path / "本編.sme"
        others = [
            tmp_path / "本編.sme.控え.writing",
            tmp_path / "本編.sme.0a1b2c3d.writing.txt",
            tmp_path / "別.sme.0a1b2c3d.writing",
            tmp_path / "本編.sme.0A1B2C3D.writing",
        ]
        for other in others:
            other.write_text("x", encoding="utf-8")
            _aged(other, atomic.STALE_WRITING_SECONDS + 60)
        write_atomic(target, "{}")
        assert all(other.is_file() for other in others)

    def test_a_crashed_recovery_scratch_goes_with_its_session(self, tmp_path: Path) -> None:
        # 壊れると、退避を書く途中で落ちた書きかけが、容量には数えられるのにどの片付けでも消えない
        session = RecoverySession(tmp_path)
        session.save(Project.create(), None)
        leftover = session.path.with_name(f"{session.path.name}.0a1b2c3d.writing")
        leftover.write_text("{", encoding="utf-8")
        lock = session._lock
        assert lock is not None
        lock.abandon()
        session._lock = None
        discard(find_orphans(tmp_path)[0])
        assert list((tmp_path / "recovery").iterdir()) == []


class TestScratchOnlySessions:
    """最初の退避を書く途中で落ちて、書きかけしか残らなかった起動"""

    def _scratch_of(self, folder: Path, session: str, age: float) -> Path:
        path = folder / f"{session}.sme.0a1b2c3d.writing"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{", encoding="utf-8")
        _aged(path, age)
        return path

    def test_an_old_scratch_of_an_ended_session_goes_on_the_next_start(
        self, tmp_path: Path
    ) -> None:
        # 壊れると、復元の候補に出ない書きかけが、容量に数えられたままどの片付けでも消えない
        folder = tmp_path / "recovery"
        crashed = self._scratch_of(folder, "0" * 32, STALE_WRITING_SECONDS + 60)
        assert find_orphans(tmp_path) == []
        assert not crashed.exists()

    def test_a_scratch_of_a_live_window_is_left_alone(self, tmp_path: Path) -> None:
        # 壊れると、別の窓が今まさに書いている退避の書きかけを消し、その窓の退避が落ちる
        session = RecoverySession(tmp_path)
        try:
            writing = self._scratch_of(
                tmp_path / "recovery", session.session, STALE_WRITING_SECONDS + 60
            )
            find_orphans(tmp_path)
            assert writing.is_file()
        finally:
            session.close()

    def test_a_fresh_scratch_of_an_ended_session_is_left_alone(self, tmp_path: Path) -> None:
        # 壊れると、錠が効かない置き場で生死を取り違えたときに、書いている最中の物を消す
        writing = self._scratch_of(tmp_path / "recovery", "0" * 32, 0)
        find_orphans(tmp_path)
        assert writing.is_file()

    def test_scratch_files_of_other_shapes_are_left_alone(self, tmp_path: Path) -> None:
        # 壊れると、本人が退避の置き場に置いた似た名前のファイルを消す
        mine = self._scratch_of(tmp_path / "recovery", "メモ", STALE_WRITING_SECONDS + 60)
        find_orphans(tmp_path)
        assert mine.is_file()


class TestMovingTheFoldersSkipsScratchFiles:
    @pytest.mark.parametrize(
        "name", ["abc.sme.0a1b2c3d.writing", "abc.sme.writing", "places.json.0a1b2c3d.writing"]
    )
    def test_a_scratch_file_is_not_moved(self, tmp_path: Path, name: str) -> None:
        # 壊れると、置き場を引き継ぐときに書きかけの中身を完成品として新しい置き場へ移す
        path = tmp_path / name
        path.write_text("{", encoding="utf-8")
        assert not _can_take(path, "backups")
        assert not _can_take(path, "recovery")


def _alias() -> Alias:
    clip = Clip(timeline_start=0, duration=30, source=TEXT.create(text="見出し"))
    return Alias.of("見出し", clip)


def _preset() -> Preset:
    blur = registry.get("blur")
    assert blur is not None
    return Preset(name="ぼかし", effects=(blur.create(),))


#: 保存の場所ごとの書き方 引数は書く先のフォルダ
_SAVES: dict[str, Callable[[Path], object]] = {
    "プロジェクト": lambda root: save_project(Project.create(), root / "本編.sme"),
    "プリセット": lambda root: PresetStore(root).save(_preset()),
    "エイリアス": lambda root: AliasStore(root).save(_alias()),
    "設定": lambda root: PreferenceStore(root / "preferences.json").save(Preferences()),
    "ショートカット": lambda root: ShortcutStore(root / "shortcuts.json").save({}),
    "自動更新の覚え書き": lambda root: UpdateStateStore(root / "state.json").save(UpdateState()),
    "AI の部品の覚え書き": lambda root: PartsStateStore(root / "ai-parts.json").save(PartsState()),
    "テンプレート": lambda root: ProjectPresetStore(root / "project_presets.json").put(
        ProjectPreset("放送", 1920, 1080, FrameRate(30), Blending.LINEAR)
    ),
    "退避の置き場": lambda root: remember_root(root / "A", root),
}


@pytest.mark.parametrize("place", list(_SAVES))
def test_every_save_uses_its_own_scratch_file(
    place: str, tmp_path: Path, scratch_names: list[tuple[str, str]]
) -> None:
    # 壊れると、その場所だけ 2 つの窓が同じ仮のファイルを取り合い、保存が落ちるか混ざる
    _SAVES[place](tmp_path)
    assert scratch_names
    for scratch, target in scratch_names:
        match = _SCRATCH.fullmatch(scratch)
        assert match is not None and match["name"] == target


class TestPreferencesFromTwoWindows:
    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        return tmp_path / "preferences.json"

    def _other_window_sets(self, path: Path, **changes: object) -> None:
        """ほかの窓（別のプロセス）が書いたことにする この窓の覚えには触らない"""
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(changes)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_a_change_in_another_window_survives_a_save_here(self, path: Path) -> None:
        # 壊れると、ほかの窓で変えたテーマが、この窓で磁石を切り替えただけで元へ戻る
        store = PreferenceStore(path)
        store.save(Preferences(theme="dark", timeline_snap=True))
        mine = store.load()
        self._other_window_sets(path, theme="light")
        store.save(Preferences(theme=mine.theme, timeline_snap=False))
        loaded = PreferenceStore(path).load()
        assert (loaded.theme, loaded.timeline_snap) == ("light", False)

    def test_the_same_item_changed_here_last_wins(self, path: Path) -> None:
        # 壊れると、この窓で選び直したテーマが、先にほかの窓で選んだテーマに負けて保存されない
        store = PreferenceStore(path)
        store.save(Preferences(theme="dark"))
        mine = store.load()
        self._other_window_sets(path, theme="light")
        store.save(Preferences(theme="system", timeline_snap=mine.timeline_snap))
        assert PreferenceStore(path).load().theme == "system"

    def test_a_save_after_another_window_does_not_undo_it_later(self, path: Path) -> None:
        # 壊れると、1 度目の保存では残ったほかの窓の変更が、この窓の 2 度目の保存で消える
        store = PreferenceStore(path)
        store.save(Preferences(theme="dark", timeline_snap=True))
        mine = store.load()
        self._other_window_sets(path, theme="light")
        mine = Preferences(theme=mine.theme, timeline_snap=False)
        store.save(mine)
        store.save(Preferences(theme=mine.theme, timeline_snap=True))
        assert PreferenceStore(path).load().theme == "light"

    def test_a_broken_file_is_written_whole(self, path: Path) -> None:
        # 壊れると、壊れたファイルに重ねたせいで、この窓の設定のうち変えなかった項目が既定へ戻る
        store = PreferenceStore(path)
        store.save(Preferences(theme="light"))
        store.load()
        path.write_text("{", encoding="utf-8")
        store.save(Preferences(theme="light", timeline_snap=False))
        loaded = PreferenceStore(path).load()
        assert (loaded.theme, loaded.timeline_snap) == ("light", False)

    def test_items_of_a_newer_version_are_kept(self, path: Path) -> None:
        # 壊れると、新しい版で選んだ項目が、前の版へ戻して設定を 1 つ変えただけで消える
        store = PreferenceStore(path)
        store.save(Preferences())
        self._other_window_sets(path, future_item=3)
        store.save(Preferences(timeline_snap=False))
        assert json.loads(path.read_text(encoding="utf-8"))["future_item"] == 3

    def test_a_save_waits_for_another_window_writing(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、2 つの窓が同時に読んで書き戻し、先に書いた方の変更を後の方が読まずに消す
        PreferenceStore(path).save(Preferences())
        held = try_hold(path.with_name(path.name + ".lock"))
        assert held is not None
        monkeypatch.setattr(atomic, "_LOCK_WAIT", 0.05)
        try:
            with pytest.raises(OSError):
                PreferenceStore(path).save(Preferences(timeline_snap=False))
        finally:
            held.release()


class TestHeldForUpdate:
    def test_a_second_holder_waits_for_the_first(self, tmp_path: Path) -> None:
        # 壊れると、読んでから書き戻す間に別の窓が割り込み、どちらかの変更が消える
        lock = tmp_path / "places.lock"
        order: list[str] = []
        inside = threading.Event()
        go_on = threading.Event()

        def first() -> None:
            with held_for_update(lock):
                order.append("1 人目が取った")
                inside.set()
                go_on.wait(5)
                order.append("1 人目が放した")

        thread = threading.Thread(target=first)
        thread.start()
        assert inside.wait(5)
        timer = threading.Timer(0.1, go_on.set)
        timer.start()
        with held_for_update(lock, wait=5):
            order.append("2 人目が取った")
        thread.join(5)
        timer.join(5)
        assert order == ["1 人目が取った", "1 人目が放した", "2 人目が取った"]

    def test_a_lock_that_never_frees_is_reported(self, tmp_path: Path) -> None:
        # 壊れると、止まった窓が錠を持ったままのときに、保存がいつまでも返らない
        lock = tmp_path / "places.lock"
        held = try_hold(lock)
        assert held is not None
        try:
            with pytest.raises(OSError), held_for_update(lock, wait=0.05):
                pass
        finally:
            held.release()
