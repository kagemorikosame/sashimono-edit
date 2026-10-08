"""AI の部品（claude-agent-sdk と、それに同梱の Claude Code）を裏で新しくする

新しいモデルが出るたびに、入れた時点の部品のままでは「このモデルには対応していません」と
断られる（Claude Opus 5.5 と、0.2.152 に同梱の Claude Code 2.1.259 で起きた） 利用者に
エラーを見せずに済むよう、配布版では 1 日に 1 回まで PyPI を確かめ、指定の範囲
（:data:`sashimono.ai.environment.REQUIRED_PACKAGES`）の中の新しい版を裏で入れる

入れ方は 2 段 新しい版をまず別の置き場（``runtime-staging``）へ入れ、入れ終えてから導入先の
中身と入れ替える 導入先へ直に入れると、途中で落ちたとき（ネットが切れた・Defender に
止められた・pip が落ちた）に新旧が混ざり、今まで動いていた版まで壊れる 入れ替えの途中で
失敗したら、動かした物を元へ戻す 入れ替えの途中でプロセスごと終わったときは、次の起動が
記録を見て戻す（:func:`sashimono.runtime.recover_runtime_swap`）

ここは Qt を使わない 時計とスレッドと画面への知らせは :mod:`sashimono.ui.chat.parts_updater`
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from sashimono.core import userdirs
from sashimono.package_index import Latest, latest_release
from sashimono.runtime import (
    FeaturePack,
    finish_swap,
    hold_swap_lock,
    is_newer_version,
    mark_installed,
    recover_swap_locked,
    roll_back_swap,
    run_pip_in_worker,
    swap_backup,
    swap_journal,
    write_swap_journal,
    writing_runtime,
)

__all__ = [
    "CHECK_INTERVAL_SECONDS",
    "FAILURE_NOTICE_COUNT",
    "PartsState",
    "PartsStateStore",
    "find_update",
    "install_staged",
    "is_due",
    "swap_in",
]

#: 確かめる間隔（秒） 1 日に 1 回まで 起動のたびに尋ねると、繋がっていない機械で毎回待つ
CHECK_INTERVAL_SECONDS = 24 * 60 * 60

#: 続けてこの回数だけ入れ替えに失敗したら、状態の行に小さく出す 1 回ごとに出すと、
#: たまたまネットが切れただけの人にもエラーを見せることになる
FAILURE_NOTICE_COUNT = 3

#: pip を走らせる物 試験で偽の pip に差し替える 引数は pip への引数・出力の受け口・中断の問い
RunPip = Callable[..., int]

#: 部品の新しい版を尋ねる物 試験で偽の PyPI に差し替える
Lookup = Callable[[str], Latest | None]


@dataclass(frozen=True, slots=True)
class PartsState:
    """次の起動へ渡すこと"""

    #: 最後に確かめた時刻（UNIX 秒） 繋がらなかった回も数える（繋がらない間に何度も尋ねない）
    last_checked: float = 0.0
    #: 続けて入れ替えに失敗した回数 入れ替えられたら 0 へ戻す
    failures: int = 0


class PartsStateStore:
    """:class:`PartsState` の読み書き 壊れていても既定のまま使う（起動は止めない）"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else userdirs.config_root() / "ai-parts.json"

    def load(self) -> PartsState:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return PartsState()
        if not isinstance(data, dict):
            return PartsState()
        checked = data.get("last_checked")
        failures = data.get("failures")
        return PartsState(
            last_checked=(
                float(checked)
                if isinstance(checked, int | float) and not isinstance(checked, bool)
                else 0.0
            ),
            failures=failures
            if isinstance(failures, int) and not isinstance(failures, bool)
            else 0,
        )

    def save(self, state: PartsState) -> None:
        # 書けなくても次の起動でもう一度確かめるだけ 止めるほどのことではない
        with contextlib.suppress(OSError):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".writing")
            temporary.write_text(json.dumps(asdict(state)), encoding="utf-8")
            temporary.replace(self.path)


def is_due(last_checked: float, now: float, interval: float = CHECK_INTERVAL_SECONDS) -> bool:
    """確かめる頃か 時計が戻った（最後に確かめた時刻が未来）ときも確かめる

    戻ったときに待つと、時計を直すまで何日も確かめなくなる
    """
    return last_checked > now or now - last_checked >= interval


def find_update(pack: FeaturePack, lookup: Lookup = latest_release) -> tuple[str, ...]:
    """範囲の中に、入れてある版より新しい版がある部品を ``名前==版`` で 無ければ空

    入っていない部品があるときは何もしない 入れるかどうかは本人が導入の欄で決める
    範囲の外（試していない大きな版上げ）は選ばない :func:`latest_release` が範囲の中の
    版（``allowed``）を分けて返す
    """
    pins: list[str] = []
    for package in pack.status().packages:
        if package.version is None:
            return ()
        try:
            name = Requirement(package.name).name
        except InvalidRequirement:
            return ()
        latest = lookup(package.name)
        if latest is None or latest.allowed is None:
            continue
        if is_newer_version(latest.allowed, package.version):
            pins.append(f"{name}=={latest.allowed}")
    return tuple(pins)


def install_staged(
    pins: Sequence[str],
    *,
    target: Path,
    key: str,
    run_pip: RunPip = run_pip_in_worker,
    on_output: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    before_swap: Callable[[], None] | None = None,
) -> bool:
    """``pins`` を別の置き場へ入れてから、導入先（``target``）と入れ替える 成功したか

    ``before_swap`` は入れ替える直前に呼ぶ 走っている Claude Code が終わるのを待つのに使う
    Windows では動いている exe を動かせないので、待たないと入れ替えが失敗する

    ``should_cancel`` が真を返したら（窓を閉じる）、入れ替えを始めずに偽で返す 始めた
    入れ替えは途中で止めない（止めると新旧が混ざる 入れ替えそのものは数秒で終わる）
    """
    staging = target.parent / f"{target.name}-staging"
    shutil.rmtree(staging, ignore_errors=True)
    arguments = ["install", "--only-binary", ":all:", "--target", str(staging), *pins]
    try:
        code = run_pip(arguments, on_output=on_output, should_cancel=should_cancel)
        if code != 0:
            return False
        if before_swap is not None:
            before_swap()
        if should_cancel is not None and should_cancel():
            # 会話が畳み終わるのを待つ間に閉じられた 今から入れ替えると、終了の待ちが切れた
            # 所で入れ替えの途中のままプロセスが終わることがある
            return False
        # 導入先へ書く所は導入のボタンと同じ錠で並べる 同時に書くと新旧が混ざる
        # 会話が畳み終わるのを待つ（before_swap）のは錠の外 持ったまま待つと、その間ずっと
        # 導入のボタンを待たせる
        with writing_runtime(on_output, should_cancel) as granted:
            if not granted or not swap_in(staging, target):
                return False
            mark_installed(target, key)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return True


def swap_in(staging: Path, target: Path) -> bool:
    """別の置き場に入れた物で導入先の中身を入れ替える 途中で失敗したら元へ戻して偽

    版の変わらなかった部品（``*.dist-info`` の名前が同じ物）は動かさない 読み込み済みの
    拡張モジュール（pydantic-core など）は Windows では動かせず、動かそうとすると毎回
    入れ替えが失敗する 変わった物だけを動かせば、たいていは SDK とその同梱の Claude Code だけで済む

    動かす前に、何を入れ替えて何を足すかの記録を導入先の隣へ書く（:func:`write_swap_journal`）
    どの瞬間に落ちても、次の起動が記録のとおりに元へ戻す（:func:`recover_runtime_swap`）
    """
    try:
        target.mkdir(parents=True, exist_ok=True)
        entries = sorted(staging.iterdir())
    except OSError:
        return False
    release = hold_swap_lock(target)
    if release is None:
        # ほかの窓が入れ替えている 次の機会に試す
        return False
    try:
        return _swap_locked(staging, target, entries)
    finally:
        release()


def _swap_locked(staging: Path, target: Path, entries: list[Path]) -> bool:
    # 前に落ちた入れ替えが残っていれば先に戻す 戻せないまま重ねると、記録も退けた物も
    # 今回の物で上書きされ、前の古い物が戻らなくなる
    recover_swap_locked(target)
    if swap_journal(target).exists():
        return False
    keep = _unchanged(staging, target)
    names = [e.name for e in entries if e.name not in keep and e.name != "__pycache__"]
    replaced = [name for name in names if os.path.lexists(target / name)]
    added = [name for name in names if name not in replaced]
    backup = swap_backup(target)
    shutil.rmtree(backup, ignore_errors=True)
    try:
        write_swap_journal(target, replaced, added)
        backup.mkdir(parents=True, exist_ok=True)
        for name in names:
            destination = target / name
            if name in replaced:
                destination.replace(backup / name)
            (staging / name).replace(destination)
    except OSError:
        # 置いた新しい物を下げ、退けた古い物を戻す 戻せない物が残ったら記録と退けた物を
        # 残し、次の起動でもう一度戻す（捨てると、退けた古い物ごと失う）
        if roll_back_swap(target, replaced, added):
            finish_swap(target)
        return False
    finish_swap(target)
    return True


def _unchanged(staging: Path, target: Path) -> set[str]:
    """別の置き場の項目のうち、入れ替えなくてよい物（版の変わらなかった部品の物）の名前

    名前空間のように複数の部品が同じ項目を分け合うときは、全部が変わらないときだけ残す
    1 つでも変わった部品があれば入れ替える（残すとその部品だけ古いまま残る）
    """
    owners: dict[str, set[bool]] = {}
    for info in staging.glob("*.dist-info"):
        same = (target / info.name).is_dir()
        owners.setdefault(info.name, set()).add(same)
        for top in _top_levels(info):
            owners.setdefault(top, set()).add(same)
    return {name for name, flags in owners.items() if flags == {True}}


def _top_levels(info: Path) -> set[str]:
    """``RECORD`` に書かれた、その部品が置いた一番上の項目の名前"""
    try:
        lines = (info / "RECORD").read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    tops: set[str] = set()
    for line in lines:
        path = line.split(",", 1)[0].strip()
        if not path or path.startswith(".."):
            continue
        tops.add(path.replace("\\", "/").split("/", 1)[0])
    return tops
