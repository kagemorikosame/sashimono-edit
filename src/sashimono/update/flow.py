"""起動の頭と画面の側から呼ぶ、更新の段取り

- :func:`apply_on_start` 次の起動で入れると決めた版を、編集画面を出す前に入れる
- :func:`settle` 前の入れ替えの結果を読み、本人へ知らせる 1 文を返す
- :func:`prepare` 新しい版を落として確かめ、入れ替えを待つ所へ置く（裏のスレッドで呼ぶ）
- :func:`start_swap` 入れ替え係を起こし、走り始めたのを確かめる

ここも Qt を読まない 起動の頭（:mod:`sashimono.app`）は編集画面より前に呼ぶ
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from sashimono import __version__
from sashimono.core import userdirs
from sashimono.core.io.locks import try_hold
from sashimono.links import RELEASES_URL
from sashimono.update.fetch import Transport
from sashimono.update.manifest import Manifest, is_newer, is_prerelease
from sashimono.update.package import CarryError, Layout, carry_user_files, current_layout, stage
from sashimono.update.portable import clear_swap_pending, new_swap_token
from sashimono.update.state import (
    STAGE_LOCK,
    UpdateState,
    UpdateStateStore,
    busy_with,
    lock_path,
)
from sashimono.update.swap import (
    SwapPlan,
    launch,
    started_by_swapper,
    take_result,
    wait_started,
)

__all__ = [
    "PREFERENCES_FILE",
    "UpdateBusyError",
    "UpdateChoices",
    "apply_on_start",
    "prepare",
    "reconcile",
    "settle",
    "start_swap",
    "update_choices",
    "updates_allowed",
]

#: 入れ替え係が書く結果のうち、入れられなかったことを表すもの → 本人に言う理由
#: 重い物から並べる 1 回の入れ替えで幾つも書かれたら、先に見つかった物を言う
#: （戻しまで失敗した回は「新しい版を動かせなかった」より、今どこから動いているかが要る）
_FAILURES = {
    "restore-failed": (
        "新しい版へ入れ替えられず、元の名前へも戻せなかったので、前の版を隣のフォルダ"
        "（.previous か .rolling）から起こした 閉じてからフォルダの名前を元に戻してください"
    ),
    "rollback-failed": (
        "新しい版が起動できず、元の名前へ戻し切れなかったので、前の版を隣のフォルダ"
        "（.previous）から起こした 閉じてからフォルダの名前を元に戻してください"
    ),
    "busy": "ほかの Sashimono の窓が開いていた",
    "previous-locked": "前の版のフォルダを片付けられなかった",
    # 断ったプログラムは名前を出せない 作業場所として持っているだけのプロセスは、開いた
    # ファイルを調べる Windows の仕組み（Restart Manager）にも出てこない よくある例を挙げる
    "install-locked": (
        "今の版のフォルダの名前を変えるのを断られた Explorer やターミナルでこのフォルダを"
        "開いていないか、ウイルス対策が検査していないかを確かめてください"
    ),
    "staged-locked": "新しい版のフォルダを動かせなかった",
}

#: 同じわけで続けて失敗したら、手で入れ替える手順を案内する回数
#: 1 回目は一時的な錠（ウイルス対策の検査など）かもしれないので、入れ直しを勧める
MANUAL_AFTER = 2


def start_swap(plan: SwapPlan) -> bool:
    """入れ替え係を起こす 走り始めたら真（呼んだ側はこの後に終わる） 起こせなければ偽"""
    try:
        launched = launch(plan)
    except OSError:
        return False
    return wait_started(launched)


#: 本人の好みの設定のファイルの名前（``ui.workspace.PreferenceStore`` と同じ 試験が照らす）
PREFERENCES_FILE = "preferences.json"


@dataclass(frozen=True, slots=True)
class UpdateChoices:
    """更新に関わる本人の好みの 3 項目（``Preferences`` と同じ既定）"""

    #: 〔起動したときに新しい版を確かめる〕
    check: bool = True
    #: 〔新しい版を入れる前に尋ねる〕
    confirm: bool = True
    #: 〔ベータ版も受け取る〕
    beta: bool = False


def update_choices() -> UpdateChoices:
    """起動の頭は Qt を読む前なので、``ui.workspace`` の読み手（Qt を読む）は使えない
    設定のファイルのこの 3 項目だけを直に読む 読めない・無い・壊れた値は既定
    """
    try:
        data = json.loads((userdirs.config_root() / PREFERENCES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return UpdateChoices()
    if not isinstance(data, dict):
        return UpdateChoices()
    plain = UpdateChoices()

    def flag(key: str, default: bool) -> bool:
        value = data.get(key)
        return value if isinstance(value, bool) else default

    return UpdateChoices(
        check=flag("update_check", plain.check),
        confirm=flag("update_confirm", plain.confirm),
        beta=flag("update_beta", plain.beta),
    )


def reconcile(state: UpdateState, choices: UpdateChoices) -> UpdateState:
    """入れ替えを待つ版と次の起動の予約を、今の好みにそろえる

    設定を変えたとき（画面の側）と起動の頭の両方がこれを通す 片方だけで決めると、
    画面で切り替えた後に起動の頭が古い予約を当てる（またはその逆）ことになる
    どの向きに切り替えても、結果は「今の好みで初めから決めた形」と同じになる

    1. 〔ベータ版も受け取る〕が切で、待っている版が先行版 待っている版ごと外す
       （予約も、入れるかを尋ねるボタンも出さない 正式版は次の確認で落とし直す）
    2. 〔起動したときに新しい版を確かめる〕が切 今の版に留まりたい合図なので予約を全部外す
       待っている版は残す（〔更新を確かめる…〕から手で入れられる）
    3. 本人が〔次の起動で入れる〕を選んだ予約は残す（尋ねて答えをもらってある）
    4. それ以外は〔入れる前に尋ねる〕で決める 切なら自動で予約し、入なら自動の予約を外す
       ただし入れ替えに失敗した版は自動では予約しない（起動のたびに失敗し続けないため）
    """
    ready = state.ready_version
    if not ready:
        return replace(state, apply_on_start=False, apply_chosen=False)
    if is_prerelease(ready) and not choices.beta:
        return _clear_ready(state)
    if not choices.check:
        return replace(state, apply_on_start=False, apply_chosen=False)
    if state.apply_on_start and state.apply_chosen:
        return state
    automatic = not choices.confirm and state.auto_blocked != ready
    return replace(state, apply_on_start=automatic, apply_chosen=False)


def updates_allowed() -> bool:
    """本人が〔起動したときに新しい版を確かめる〕を切っていないか"""
    return update_choices().check


def apply_on_start(
    arguments: Sequence[str],
    *,
    layout: Layout | None = None,
    store: UpdateStateStore | None = None,
    swap: Callable[[SwapPlan], bool] = start_swap,
    current: str = __version__,
    choices: Callable[[], UpdateChoices] | None = None,
) -> bool:
    """次の起動で入れると決めた版があれば、入れ替え係を起こす 真なら呼んだ側はすぐ終わる

    編集画面を出す前に呼ぶ 出してからだと、開いたプロジェクトを閉じさせることになる
    印は入れ替え係を起こす前に下ろす 失敗は結果のファイルで次の起動が知らせ、その版は
    自動では予約し直さない（``auto_blocked``） 入れ替えが毎回失敗する機械で、起動のたびに
    入れ替えを試して待たされるのを防ぐ

    予約は今の好みにそろえてから見る（:func:`reconcile` ``choices`` 既定は設定のファイルを読む）
    画面で設定を変えたときと同じ決まりで、起動の頭でも決める

    ほかの入れ替え係が走っていれば（2 つ目の起動）何も起こさずに真を返す 入れ替え係は
    窓が全部閉じるのを待って入れ替え、新しい版を起こし直す ここで画面を出すと、
    入れ替え係はこの窓が閉じるまで待たされる
    """
    layout = layout if layout is not None else current_layout()
    store = store if store is not None else UpdateStateStore()
    choices = choices if choices is not None else update_choices
    if layout is None:
        return False
    busy = busy_with(store.path.parent)
    if busy == "swap":
        # 入れ替え係に起こされた版は、その入れ替え係が錠を持ったまま窓を待っている相手
        # ここで終わると、入れ替えたばかりの版が起動できなかったと数えられて前の版へ戻される
        # 0.2.0 までの入れ替え係が起こした版でも、起動できた印の場所が渡るので見分けられる
        return not started_by_swapper()
    if busy is not None:
        return False  # ほかの窓が新しい版を落としている 置き場が替わる途中なので触らない
    loaded = store.load()
    state = reconcile(loaded, choices())
    if not state.apply_on_start:
        if state != loaded:
            with contextlib.suppress(OSError):
                store.save(state)
        return False
    try:
        store.save(replace(state, apply_on_start=False, apply_chosen=False))
    except OSError:
        return False  # 印を下ろせない所で入れ替えると、失敗したときに毎回繰り返す
    if layout.staged_version() != state.ready_version or not is_newer(state.ready_version, current):
        return False
    token = new_swap_token()
    try:
        carry_user_files(
            layout.install,
            layout.staged,
            aside=userdirs.config_root() / "scripts",
            swap_mark=(current, token),
        )
    except CarryError as exc:
        # 写せないまま入れ替えると、本人の物は次の更新で消える版にだけ残る 入れ替えずに今の版で
        # 起動し、画面を出した後に知らせる（UpdateController.start） 落として確かめた .new は
        # 完全な新しい版なので残す（直してから選べば落とし直さずに入れられる） 起動のたびに
        # 試して待たされないよう、この版は本人が選ぶまで自動では入れない
        with contextlib.suppress(OSError):
            store.save(
                replace(
                    store.load(), auto_blocked=state.ready_version, pending_notice=exc.explain()
                )
            )
        return False
    plan = SwapPlan("apply", layout, pid=os.getpid(), arguments=tuple(arguments[1:]))
    if swap(plan):
        return True
    # 入れ替え係を起こせなかった 自分の待っている印を外す（残すと、移しが印の切れるまで止まる
    # ほかの窓の印は残す）
    clear_swap_pending(userdirs.config_root(), token)
    # 入れ替え係を起こせなかった（台本の実行が止められている など） 次の起動でまた
    # 自動で予約して試すと、起動のたびに待たされる この版は本人が選ぶまで自動では入れない
    with contextlib.suppress(OSError):
        store.save(replace(store.load(), auto_blocked=state.ready_version))
    return False


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
    lines = list(results) if results is not None else take_result(store.path.parent)
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
        found = next(((key, text) for key, text in _FAILURES.items() if key in lines), None)
        if found is None:
            found = next(
                (("error", line[len("error ") :]) for line in lines if line.startswith("error ")),
                None,
            )
        if found is not None and ready:
            # 入れ替えに失敗した版は、尋ねない設定でも自動では予約し直さない（reconcile）
            state = replace(state, auto_blocked=ready)
        if found is not None:
            reason, failure = found
            key = _failure_key(ready, reason, failure)
            count = state.failure_count + 1 if state.last_failure == key else 1
            state = replace(state, last_failure=key, failure_count=count)
            if count >= MANUAL_AFTER:
                notice = (
                    f"更新を入れられませんでした（{failure}）"
                    f" 同じ理由で {count} 回続けて入れられませんでした {_manual_steps(layout)}"
                )
            else:
                notice = (
                    f"更新を入れられませんでした（{failure}）"
                    " ヘルプの〔更新を確かめる…〕から入れ直せます"
                )
        if "rollback-failed" in lines and ready:
            # 起動できなかった版は、戻し切れなかったときも次から飛ばす
            state = _clear_ready(replace(state, skipped=(*state.skipped, ready)))
        if ready and (layout is None or layout.staged_version() != ready):
            # 入れ替えを待っていた版が無くなった（本人が消した・別の起動が入れた）
            state = _clear_ready(state)
    if (
        layout is not None
        and not state.ready_version
        and layout.staged.exists()
        and busy_with(store.path.parent) is None
    ):
        # 待つ物の無い置き場を残さない 100 MB を超える ほかの窓が落としている・入れ替え係が
        # 走っている間は触らない（展開し終えて覚え書きを書く前の物を消してしまう）
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
    """新しい版を落として確かめ、入れ替えを待つ所へ置き、覚え書きに残す 失敗は例外のまま返す

    窓を 2 つ開いていると、どちらも起動の後に確かめに来る 錠を取れた方だけが落とす
    （取れなければ :class:`UpdateBusyError`） 同時に落とすと、片方が展開している ``.new`` を
    もう片方が消す 入れ替え係が走っている間も落とさない（入れ替えている最中の ``.new`` を消す）
    ``apply_on_start`` は尋ねない設定が自動で付ける予約 前の版に本人が選んだ予約は持ち越さない
    """
    store = store if store is not None else UpdateStateStore()
    folder = store.path.parent
    if busy_with(folder) == "swap":
        raise UpdateBusyError("入れ替えの最中")
    lock = try_hold(lock_path(STAGE_LOCK, folder))
    if lock is None:
        raise UpdateBusyError("ほかの窓が新しい版を落としている")
    try:
        stage(manifest, transport, layout, should_cancel=should_cancel)
        state = store.load()
        if state.ready_version != manifest.version:
            # 待たせる版が替わったら、前の版の失敗の続きを忘れる 残すと、版 A で 1 回失敗した
            # 後、版 B の最初の失敗で手で入れ替える案内を出す（印にも版を含めて比べる）
            state = replace(state, last_failure="", failure_count=0)
        store.save(
            replace(
                state,
                ready_version=manifest.version,
                ready_notes_url=manifest.notes_url,
                ready_python_abi=manifest.python_abi,
                apply_on_start=apply_on_start,
                apply_chosen=False,
            )
        )
    finally:
        lock.release()


class UpdateBusyError(Exception):
    """ほかの窓か入れ替え係が更新を進めている 待てば済むので、起動時の確認では黙る"""


#: 一般のエラー（``error ...``）の文面を比べるときに見る長さ 例外の文面は長いことがあり、
#: 覚え書きを大きくしない 頭が同じなら同じ原因と見てよい
_ERROR_TEXT_CHARS = 120


def _failure_key(ready: str, reason: str, text: str) -> str:
    """続けて失敗したかを比べる印 版と、失敗のわけ

    版を含める 版 A で 1 回失敗した後、版 B の最初の失敗を「2 回続けて」と数えない
    一般のエラー（``error``）は言葉が 1 つなので、文面も含める 別の原因の失敗 2 回を
    同じ理由と数えない 数字（時刻・PID・行の番号など）は毎回変わりうるので落とし、
    空白をそろえ、頭の :data:`_ERROR_TEXT_CHARS` 文字だけを見る
    """
    if reason == "error":
        text = re.sub(r"\s+", " ", re.sub(r"\d+", "#", text)).strip()[:_ERROR_TEXT_CHARS]
        reason = f"error {text}"
    return f"{ready} {reason}"


def _manual_steps(layout: Layout | None) -> str:
    """入れ直しても同じ失敗を繰り返すときに、手で入れ替える手順

    展開し終えた新しい版（``.new``）が残っていれば、名前を 2 回変えるだけで入れ替えられる
    無ければ配布のページから入れ直してもらう
    """
    page = f"配布のページ（{RELEASES_URL}）から zip を落として入れ直すこともできます"
    if layout is None or layout.staged_version() is None:
        return f"入れ直しても同じ所で止まります {page}"
    install, previous, staged = layout.install.name, layout.previous.name, layout.staged.name
    return (
        f"手で入れ替えるには、Sashimono を閉じてから、{layout.install.parent} を Explorer で開き、"
        f"「{install}」を「{previous}」へ、「{staged}」を「{install}」へ名前を変えてください"
        f"（「{previous}」が前からあれば、先に消すか別の名前にする） {page}"
    )


def _clear_ready(state: UpdateState) -> UpdateState:
    # 失敗の続いた回数も消す 次に待つ版は別の版で、前の版の失敗を数えると、1 回目から
    # 手で入れ替える案内を出すことになる
    return replace(
        state,
        ready_version="",
        ready_notes_url="",
        ready_python_abi="",
        apply_on_start=False,
        apply_chosen=False,
        last_failure="",
        failure_count=0,
    )
