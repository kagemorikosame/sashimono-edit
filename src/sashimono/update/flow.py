"""起動の頭と画面の側から呼ぶ、更新の段取り

- :func:`apply_on_start` 次の起動で入れると決めた版を、編集画面を出す前に入れる
- :func:`settle` 前の入れ替えの結果を読み、本人へ知らせる 1 文を返す
- :func:`prepare` 新しい版を落として確かめ、入れ替えを待つ所へ置く（裏のスレッドで呼ぶ）
- :func:`start_swap` 入れ替え係を起こし、走り始めたのを確かめる

ここも Qt を読まない 起動の頭（:mod:`sashimono.app`）は編集画面より前に呼ぶ
"""

from __future__ import annotations

import contextlib
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import replace

from sashimono import __version__
from sashimono.update.fetch import Transport
from sashimono.update.manifest import Manifest, is_newer
from sashimono.update.package import Layout, carry_user_files, current_layout, stage
from sashimono.update.state import UpdateState, UpdateStateStore, update_dir
from sashimono.update.swap import SwapPlan, launch, take_result, wait_started

__all__ = ["apply_on_start", "prepare", "settle", "start_swap"]

#: 入れ替え係が書く結果のうち、入れられなかったことを表すもの → 本人に言う理由
_FAILURES = {
    "busy": "ほかの Sashimono の窓が開いていた",
    "previous-locked": "前の版のフォルダを片付けられなかった",
    "install-locked": "今の版のフォルダを動かせなかった（ほかのプログラムが開いている）",
    "staged-locked": "新しい版のフォルダを動かせなかった",
    "rollback-failed": "新しい版が起動できず、前の版へも戻し切れなかった",
}


def start_swap(plan: SwapPlan) -> bool:
    """入れ替え係を起こす 走り始めたら真（呼んだ側はこの後に終わる） 起こせなければ偽"""
    try:
        launched = launch(plan)
    except OSError:
        return False
    return wait_started(launched)


def apply_on_start(
    arguments: Sequence[str],
    *,
    layout: Layout | None = None,
    store: UpdateStateStore | None = None,
    swap: Callable[[SwapPlan], bool] = start_swap,
    current: str = __version__,
) -> bool:
    """次の起動で入れると決めた版があれば、入れ替え係を起こす 起こしたら真（すぐ終わる）

    編集画面を出す前に呼ぶ 出してからだと、開いたプロジェクトを閉じさせることになる
    印は入れ替え係を起こす前に下ろす 入れ替えが毎回失敗する機械で、起動のたびに
    入れ替えを試して起動できなくなるのを防ぐ（失敗は結果のファイルで次の起動が知らせる）
    """
    layout = layout if layout is not None else current_layout()
    store = store if store is not None else UpdateStateStore()
    if layout is None:
        return False
    state = store.load()
    if not state.apply_on_start or not state.ready_version:
        return False
    try:
        store.save(replace(state, apply_on_start=False))
    except OSError:
        return False  # 印を下ろせない所で入れ替えると、失敗したときに毎回繰り返す
    if layout.staged_version() != state.ready_version or not is_newer(state.ready_version, current):
        return False
    carry_user_files(layout.install, layout.staged)
    plan = SwapPlan("apply", layout, pid=os.getpid(), arguments=tuple(arguments[1:]))
    return swap(plan)


def settle(
    *,
    layout: Layout | None = None,
    store: UpdateStateStore | None = None,
    current: str = __version__,
    results: Sequence[str] | None = None,
) -> str | None:
    """前の入れ替えの結果を片付け、知らせる 1 文を返す 知らせることが無ければ ``None``"""
    layout = layout if layout is not None else current_layout()
    store = store if store is not None else UpdateStateStore()
    lines = list(results) if results is not None else take_result(update_dir())
    state = store.load()
    notice: str | None = None
    ready = state.ready_version

    if "rolled-back" in lines and ready:
        # 入れた版が 2 回続けて起動できなかった 同じ版を次の確認でまた落とさない
        state = replace(state, skipped=(*state.skipped, ready))
        notice = (
            f"新しい版 {ready} は起動できなかったので、前の版 {current} に戻しました"
            " この版は自動では入れません"
        )
        state = _clear_ready(state)
    elif ready and not is_newer(ready, current):
        if ready == current:
            notice = f"Sashimono Edit {current} に更新しました"
        state = _clear_ready(state)
    else:
        failure = next((_FAILURES[line] for line in lines if line in _FAILURES), None)
        if failure is None:
            failure = next(
                (line[len("error ") :] for line in lines if line.startswith("error ")), None
            )
        if failure is not None:
            notice = (
                f"更新を入れられませんでした（{failure}）"
                " ヘルプの〔更新を確かめる…〕から入れ直せます"
            )
        if ready and (layout is None or layout.staged_version() != ready):
            # 入れ替えを待っていた版が無くなった（本人が消した・別の起動が入れた）
            state = _clear_ready(state)
    if layout is not None and not state.ready_version and layout.staged.exists():
        # 待つ物の無い置き場を残さない 100 MB を超える
        shutil.rmtree(layout.staged, ignore_errors=True)
    # 書けなくても次の起動で同じ片付けをやり直すだけ
    with contextlib.suppress(OSError):
        store.save(state)
    return notice


def prepare(
    manifest: Manifest,
    *,
    transport: Transport,
    layout: Layout,
    store: UpdateStateStore | None = None,
    apply_on_start: bool = False,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """新しい版を落として確かめ、入れ替えを待つ所へ置き、覚え書きに残す 失敗は例外のまま返す"""
    store = store if store is not None else UpdateStateStore()
    stage(manifest, transport, layout, should_cancel=should_cancel)
    state = store.load()
    store.save(
        replace(
            state,
            ready_version=manifest.version,
            ready_notes_url=manifest.notes_url,
            ready_python_abi=manifest.python_abi,
            apply_on_start=apply_on_start,
        )
    )


def _clear_ready(state: UpdateState) -> UpdateState:
    return replace(
        state, ready_version="", ready_notes_url="", ready_python_abi="", apply_on_start=False
    )
