"""AI 連携の実行環境

字幕起こしと同じく、既定では入っていない（:mod:`sashimono.runtime` を参照）
実行には **Claude Code 本体**（``claude`` コマンド）が要る 今の SDK は本体を
同梱しているので pip だけで揃う 同梱の無い古い SDK のときだけ、別に入れた
``claude`` を探し、見つからなければ案内を出す

動かすには Claude へのログイン（または API キー）も要る これはこのソフトが
預かるものではないので、Claude Code 自身のログイン画面を開くところまでを受け持つ
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

from sashimono.core.userdirs import state_root
from sashimono.runtime import FeaturePack, PackStatus
from sashimono.runtime import install_command as _install_command

__all__ = [
    "AI_PACK",
    "REQUIRED_PACKAGES",
    "bundled_claude_cli",
    "claude_cli",
    "claude_environment",
    "conversation_folder",
    "credentials_found",
    "find_claude_cli",
    "install_command",
    "open_login_window",
    "runtime_status",
    "uses_saved_login",
]

#: 一覧（:data:`sashimono.ai.models.MODELS`）のモデルがすべて使える、同梱の Claude Code の版
#: Claude Opus 5.5 は 2.1.280 より前の Claude Code では「does not support this model」で
#: 断られる（0.2.152 に同梱の 2.1.259 で、利用者の画面で起きた）
#: 新しいモデルを一覧へ足すときは、ここと下の SDK の下限を一緒に上げる
#: 2.1.310 は、自動メモを止める変数と記録を残さない引数を実物の会話で確かめた版（#412）
#: 本人のメモが同意の外で渡らないことをこの版で約束するので、古い部品のままにしない
MINIMUM_CLAUDE_CODE = "2.1.310"

#: SDK の下限 0.2.174 は Claude Code 2.1.310 を同梱する（入れた SDK の ``claude.exe --version``
#: で確かめた） Opus 5.5 に要る 2.1.280 は 0.2.158 から入るが、それより古い版で #412 の
#: 設定が効くかは確かめていない モデルとエフォートの指定は 0.2.152 から使える
#: 上限は試した系列（0.2）の外へ自動で上がらないため 0.3 で使い方が変わっても、
#: 自動の更新が入れて会話が始まらなくなる、ということが起きない 新しい系列は試してから上げる
REQUIRED_PACKAGES: tuple[str, ...] = ("claude-agent-sdk>=0.2.174,<0.3",)


def bundled_claude_cli() -> Path | None:
    """SDK に同梱された Claude Code を探す 無ければ ``None``

    claude-agent-sdk は Claude Code 本体を ``_bundled`` に入れて配っている
    （0.2.152 で確かめた） SDK を導入すればそれで動くのに、PATH の ``claude``
    だけを見ていたため、Node で別に入れていない人は導入のあとも「claude が
    見つかりません」と出て入力欄が開かなかった（Issue #27）

    SDK 本体は import しない 読み込みが重く、状態を見るたびに払う重さではない
    """
    try:
        spec = importlib.util.find_spec("claude_agent_sdk")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        for name in ("claude.exe", "claude"):
            candidate = Path(location) / "_bundled" / name
            if candidate.is_file():
                return candidate
    return None


def find_claude_cli() -> Path | None:
    """別に入れた ``claude`` の実行ファイルを探す

    PATH だけを見ると取りこぼす Claude Code の公式インストーラは
    ``~/.local/bin`` へ置くが、そこを PATH に足すのは shell の設定であって、
    エクスプローラから起動した GUI アプリには引き継がれないことがある
    「入れてあるのに見つからない」が一番分かりにくい失敗なので、既知の場所も見る
    """
    found = shutil.which("claude")
    if found:
        return Path(found)

    candidates = [Path.home() / ".local" / "bin" / name for name in ("claude.exe", "claude")]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.extend(Path(appdata) / "npm" / name for name in ("claude.cmd", "claude.exe"))
    return next((path for path in candidates if path.exists()), None)


def claude_cli() -> Path | None:
    """会話に使う Claude Code SDK に同梱の物を先に使う

    SDK 自身も同梱の物を先に探す ここで順番を変えると、画面が「使える」と
    言った物と実際に起動する物が食い違う
    """
    return bundled_claude_cli() or find_claude_cli()


def credentials_found() -> bool:
    """Claude Code がログイン済みか、API キーが渡されているように見えるか

    確かめるのは置き場があるかだけで、中身は読まない 秘密の値をこのソフトが
    触る理由が無い 外れても案内を出すかどうかが変わるだけなので、使うのは
    止めない
    """
    return _key_in_environment() or _saved_login_found()


def uses_saved_login() -> bool:
    """鍵を環境変数で渡さず、Claude Code に保存したログインで使っているか

    ログインが切れるのはこの人だけ 環境変数の鍵が無効な人に「切れた」と言うと、
    直す所を取り違える 中身は読まない
    """
    return _saved_login_found() and not _key_in_environment()


def _key_in_environment() -> bool:
    return any(os.environ.get(name) for name in _CREDENTIAL_VARIABLES)


def _saved_login_found() -> bool:
    config = os.environ.get(_CONFIG_VARIABLE)
    root = Path(config).absolute() if config else Path.home() / ".claude"
    return (root / ".credentials.json").is_file()


def claude_environment() -> dict[str, str]:
    """起動する Claude Code へ渡す環境変数のうち、このソフトが直す物

    Claude Code の置き場の指定が相対だと、起動する場所で指す先が変わる 会話は専用のフォルダ、
    ログインの窓はホームで開くので、そのままではログインの確かめ・会話・窓が別の場所を見る
    このソフトの起動の場所を基準に絶対の場所へ直して、3 つを揃える
    """
    config = os.environ.get(_CONFIG_VARIABLE)
    return {_CONFIG_VARIABLE: str(Path(config).absolute())} if config else {}


def conversation_folder() -> Path:
    """会話の間 Claude Code を置いておく、空のフォルダ

    渡さないと起動した場所を受け継ぎ、その場所に紐付くメモや設定の持ち主になる（#412）
    配布版でも起動の場所は開き方で変わるので、決めておく 道具は渡していないので、
    Claude Code がこの中のファイルを読み書きすることは無い
    """
    return state_root() / "ai-conversation"


def open_login_window(cli: Path) -> None:
    """Claude Code を別の窓で起動する ログインがまだなら、そこで案内が出る

    ログインはブラウザと Claude Code の間で済ませてもらう パスワードや鍵を
    このソフトの入力欄で受け取ると、預かる責任まで背負うことになる
    窓を分けるのは、配布版には黒い画面（コンソール）が無く、同じ窓では
    何も見えないため
    """
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    # 引数はここで決めた実行ファイルだけで、shell も通さない
    environment = {**os.environ, **claude_environment()}
    subprocess.Popen([str(cli)], cwd=str(Path.home()), env=environment, creationflags=flags)


#: Claude Code の置き場を変える環境変数
_CONFIG_VARIABLE = "CLAUDE_CONFIG_DIR"
#: Claude Code が読む、鍵の入った環境変数 どれかがあればログインは要らない
_CREDENTIAL_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")


AI_PACK = FeaturePack(
    key="ai",
    label="AI 連携",
    required=REQUIRED_PACKAGES,
    # SDK が Claude Code 本体（0.2.152 で約 220 MB）を同梱しているぶんを含める
    size_mb=250,
    commands=("claude",),
    command_hint="npm install -g @anthropic-ai/claude-code で入れられます",
    locate=lambda name: claude_cli() if name == "claude" else shutil.which(name),
)


def runtime_status() -> PackStatus:
    return AI_PACK.status()


def install_command(*, upgrade: bool = False) -> list[str]:
    return _install_command(AI_PACK, extra=False, upgrade=upgrade)
