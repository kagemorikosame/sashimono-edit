"""AI の道具の区分と、取り消せない操作の守り（#350 #341）

道具は次の 3 つのどれかに入る（:attr:`~sashimono.ai.operations.Operation.safety`）
2026-10-14 に利用者が決めたこと（#298 #350 のコメント）をそのまま形にしてある

- 読むだけ 何も変えない 確かめずに通す アプリ自身の設定・ショートカット・画面配置・
  更新・ヘルプは、AI にはこれだけを出す（変えるのは人だけ）
- 取り消せる編集 プロジェクトを ``Command`` で変える 「元に戻す」1 回で戻るので、
  確かめは今までどおり「変更を自動で承認」で省ける
- 取り消せない操作 上書き保存・書き出し・ファイルを開く・素材の削除・外へ送る操作など
  「元に戻す」では戻らないので、「変更を自動で承認」を入れていても**毎回**確かめる
  確かめでは、何をどこへするかを出す（:class:`Consent`）

区分の値（``read_only`` など）は、操作の台帳（#347）にもそのまま載せられる文字にしてある

書ける場所も決まっている（:func:`writable_path`） 今の作品（``.sme``）のフォルダと、
書き出しの窓で前に選んだフォルダだけ それ以外は人がファイルの窓で選ぶ 元の素材は
その中にあっても書かない（#341 元の素材を壊さない）
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sashimono.ai.host import EditorHost, ToolError

__all__ = [
    "Consent",
    "Safety",
    "check_readable",
    "check_writable",
    "readable_output",
    "writable_path",
    "write_roots",
]


class Safety(StrEnum):
    """道具の区分 確かめの要否がこれで決まる"""

    READ_ONLY = "read_only"
    UNDOABLE = "undoable"
    IRREVERSIBLE = "irreversible"

    @property
    def label(self) -> str:
        """画面と台帳に出す名前"""
        if self is Safety.READ_ONLY:
            return "読むだけ"
        if self is Safety.UNDOABLE:
            return "取り消せる編集"
        return "取り消せない操作"


@dataclass(frozen=True, slots=True)
class Consent:
    """取り消せない操作の前に人へ見せる中身

    道具の名前と引数だけでは、どのファイルが消えるのか・どこへ書くのかが読めない
    押す前に、何を（``action``）どこへ（``target``）するかを言葉で出す
    """

    #: 何をするか 例「プロジェクトを上書き保存します」
    action: str
    #: どこへ・何に 例 書き込むファイルの場所
    target: str
    #: 添える注意 例「同じ名前のファイルがあり、中身を置き換えます」
    detail: str = ""

    def text(self) -> str:
        """確かめの窓に出す文"""
        lines = [self.action, f"対象: {self.target}"]
        if self.detail:
            lines.append(self.detail)
        return "\n".join(lines)


def write_roots(project_path: Path | None, export_folder: Path | None) -> tuple[Path, ...]:
    """AI が書いてよいフォルダ 作品のフォルダと、書き出しの窓で前に選んだフォルダ

    保存していない作品にはフォルダが無い そのときは書き出しのフォルダだけになる
    """
    roots: list[Path] = []
    if project_path is not None:
        roots.append(project_path.parent)
    if export_folder is not None:
        roots.append(export_folder)
    return tuple(roots)


def writable_path(host: EditorHost, raw: object) -> Path:
    """AI が書こうとしている場所を確かめ、解決した場所を返す 書けなければ断る

    取り消せない道具は、確かめの窓を出す前（:class:`Consent` を作るとき）と、実際に
    書く直前の 2 度ここを通す 窓が出ている間に、フォルダがリンクへ差し替えられても
    書く直前に気付けるように
    """
    project = host.document.project
    return check_writable(
        raw,
        project_path=host.project_path,
        export_folder=host.export_folder,
        protected=(media.path for media in project.media),
    )


def check_writable(
    raw: object,
    *,
    project_path: Path | None,
    export_folder: Path | None,
    protected: Iterable[Path] = (),
) -> Path:
    """:func:`writable_path` の中身 ホストを持たない試験からも呼べるように分けてある

    断る理由の文面はそのまま AI へ返る 書ける場所を添えて、人に選んでもらう道を示す
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError("path（書く場所）を文字で渡してください")
    roots = write_roots(project_path, export_folder)
    if not roots:
        raise ToolError(
            "AI が書ける場所がありません 作品を一度保存するか、書き出しの窓で書き出し先を"
            "選んでもらうと、そのフォルダへ書けます それ以外の場所は人がファイルの窓で選びます"
        )
    candidate = Path(raw.strip())
    if any(":" in part for part in candidate.parts[1 if candidate.anchor else 0 :]):
        # NTFS の代替データストリーム（名前:流れ） 見た目のファイルは変わらないまま、
        # 書ける場所の中のフォルダやファイルへ別の中身を足せてしまう
        raise ToolError(f"名前に「:」を含む場所には書けません: {raw}")
    if ".." in candidate.parts:
        # 解決すれば中へ戻る道でも断る 意図の読めない道を通すと、確かめの窓に出した
        # 場所と、AI が思っている場所が食い違う
        raise ToolError(f"「..」を含む場所には書けません 書く場所をそのまま渡してください: {raw}")
    if not candidate.is_absolute():
        if project_path is None:
            raise ToolError(
                "作品をまだ保存していないので、相対の場所は使えません 絶対の場所を渡してください"
            )
        candidate = project_path.parent / candidate
    # リンク（シンボリックリンク・ジャンクション）を辿った先で判定する 辿らずに見ると、
    # 作品のフォルダの中に置いたリンクを通って、外のファイルを上書きできてしまう
    resolved = candidate.resolve()
    allowed = [root.resolve() for root in roots]
    if not any(_inside(resolved, root) for root in allowed):
        places = " / ".join(str(root) for root in allowed)
        raise ToolError(
            f"{resolved} は AI が書ける場所の外です 書けるのは {places} の中だけです"
            " ほかの場所へは、人がファイルの窓で選んで保存してもらってください"
        )
    for original in protected:
        if _same(resolved, original.resolve()) or _same_file(resolved, original):
            # 元の素材は作品の外の物 上書きすると、取り消しても戻らない（#341）
            raise ToolError(f"元の素材のファイルには書けません: {resolved}")
    if resolved.is_dir():
        raise ToolError(f"フォルダではなくファイルの場所を渡してください: {resolved}")
    if not resolved.parent.is_dir():
        # フォルダは作らない 作れるようにすると、書ける場所の中へ AI が好きな形の
        # フォルダを増やせる 人が知らない置き場が残る
        raise ToolError(f"フォルダがありません: {resolved.parent}")
    return resolved


def readable_output(host: EditorHost, raw: object) -> Path:
    """AI が調べたい書き出した動画の場所を確かめ、解決した場所を返す 読めなければ断る（#337）

    読むだけでも、場所は書ける所（作品のフォルダと、書き出しの窓で前に選んだフォルダ）に
    限る 書き出した物はそこにある どこでも読めるようにすると、作品と関係の無い動画の
    中身（長さや音の大きさ）まで AI へ渡ることになる
    """
    return check_readable(raw, project_path=host.project_path, export_folder=host.export_folder)


def check_readable(raw: object, *, project_path: Path | None, export_folder: Path | None) -> Path:
    """:func:`readable_output` の中身 ホストを持たない試験からも呼べるように分けてある"""
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError("path（調べる動画の場所）を文字で渡してください")
    roots = write_roots(project_path, export_folder)
    if not roots:
        raise ToolError(
            "調べられる場所がありません 書き出した動画を調べられるのは、作品のフォルダと"
            "書き出しの窓で選んだフォルダの中だけです path を省くと今のタイムラインを調べます"
        )
    candidate = Path(raw.strip())
    if any(":" in part for part in candidate.parts[1 if candidate.anchor else 0 :]):
        raise ToolError(f"名前に「:」を含む場所は読めません: {raw}")
    if ".." in candidate.parts:
        raise ToolError(f"「..」を含む場所は読めません 場所をそのまま渡してください: {raw}")
    if not candidate.is_absolute():
        if project_path is None:
            raise ToolError(
                "作品をまだ保存していないので、相対の場所は使えません 絶対の場所を渡してください"
            )
        candidate = project_path.parent / candidate
    resolved = candidate.resolve()
    allowed = [root.resolve() for root in roots]
    if not any(_inside(resolved, root) for root in allowed):
        places = " / ".join(str(root) for root in allowed)
        raise ToolError(f"{resolved} は調べられる場所の外です 調べられるのは {places} の中だけです")
    if not resolved.is_file():
        raise ToolError(f"ファイルがありません: {resolved}")
    return resolved


def _key(path: Path) -> str:
    """比べ方 Windows では大文字小文字と区切りの違いで別の場所と見なさない"""
    return os.path.normcase(str(path))


def _inside(path: Path, root: Path) -> bool:
    """``root`` の中か 文字の頭だけで比べない（「作品」と「作品2」を同じと見なさない）"""
    path_key, root_key = _key(path), _key(root)
    try:
        common = os.path.commonpath([path_key, root_key])
    except ValueError:
        # ドライブが違う
        return False
    return common == root_key and path_key != root_key


def _same(left: Path, right: Path) -> bool:
    return _key(left) == _key(right)


def _same_file(path: Path, original: Path) -> bool:
    """中身が同じファイルか（ファイルの ID で比べる）

    場所の文字だけで比べると、元の素材へのハードリンクを別のファイルと見なし、
    書くと元の素材の中身を上書きする リンクは :meth:`Path.resolve` でも辿れない
    どちらかがまだ無い（新しく作るファイル・オフラインの素材）なら、同じではない
    """
    try:
        return path.samefile(original)
    except OSError:
        return False
