"""現在のプロジェクトと Undo 履歴を保持する

モデルが frozen なので、履歴は「変更前のプロジェクト」をそのまま持つだけで足りる
構造共有により、1 クリップの変更で複製されるのは根までの経路上のオブジェクトだけなので、
数百段の履歴でもメモリは問題にならない
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sashimono.core.commands.base import Command
from sashimono.core.model import MediaId, MediaItem, Project

__all__ = ["Document", "HistoryEntry"]

#: 保持する履歴の上限 これを超えた分は古い方から捨てる
DEFAULT_HISTORY_LIMIT = 200


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """履歴 1 段 ``before`` はこの操作を行う前のプロジェクト"""

    label: str
    before: Project


class Document:
    """編集対象のプロジェクトと、その変更履歴

    UI もエンジンも AI もこのオブジェクト越しにプロジェクトを触る
    :meth:`execute` 以外の経路でプロジェクトが変わることはない
    """

    def __init__(self, project: Project, *, history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        if history_limit < 1:
            raise ValueError(f"履歴上限は 1 以上必要: {history_limit}")
        self._project = project
        self._history_limit = history_limit
        self._undo: list[HistoryEntry] = []
        self._redo: list[HistoryEntry] = []
        self._listeners: list[Callable[[Project], None]] = []
        self._checkpoint_depth = 0
        self._checkpoint_label = ""
        self._checkpoint_before: Project | None = None
        self._checkpoint_merge = False
        #: 最後に積んだ段 続けて行う操作（ホイールで高さを変えるなど）は、この段が
        #: まだ一番上に残っているときだけ同じ段へまとめる 取り消したあとの段へ
        #: まとめると、関係の無い古い操作と一緒に戻ってしまう
        self._merge_anchor: HistoryEntry | None = None
        #: この文書が一度でも持った素材と作品の名前 :attr:`seen_media` を見る
        self._seen_media: dict[tuple[MediaId, str, str], MediaItem] = {}
        self._seen_names: set[str] = set()
        self._remember(project)

    @property
    def seen_media(self) -> tuple[MediaItem, ...]:
        """この文書が一度でも持った素材（消した物・開き直す前の作品の物も）

        AI へ送る結果の伏せ字（#342）が、作品の外の記録（字幕起こしの知らせ・履歴の名前）に
        残った名前を拾うのに使う 履歴は上限で古い段が落ち、別の作品を開くと消えるので、
        履歴からは拾い切れない 中身の大きい物（作品）は持たず、素材の情報だけを覚える
        """
        return tuple(self._seen_media.values())

    @property
    def seen_project_names(self) -> tuple[str, ...]:
        """この文書が一度でも持った作品の名前（名前を変える前・開き直す前の物も）"""
        return tuple(sorted(self._seen_names))

    def _remember(self, project: Project) -> None:
        for media in project.media:
            # 名前を変えた素材は変える前の名前も残す（ID だけで見ると前の名前を忘れる）
            self._seen_media.setdefault((media.id, media.name, str(media.path)), media)
        self._seen_names.add(project.name)

    @property
    def project(self) -> Project:
        return self._project

    def subscribe(self, listener: Callable[[Project], None]) -> Callable[[], None]:
        """プロジェクトが変わったときに呼ばれる関数を登録する

        戻り値を呼ぶと解除される
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def execute(self, command: Command) -> Project:
        """コマンドを適用し、新しいプロジェクトを返す

        コマンドが例外を投げた場合、プロジェクトも履歴も一切変わらない
        """
        updated = command.apply(self._project)

        if self._checkpoint_depth == 0:
            self._push_undo(HistoryEntry(command.label, self._project))
            self._redo.clear()

        self._set_project(updated)
        return updated

    @contextmanager
    def checkpoint(self, label: str, *, merge: bool = False) -> Iterator[None]:
        """複数のコマンドを 1 回の Undo でまとめて戻せるようにする

        AI エージェントが 1 つの指示で何十回も編集を行うため、これがないと
        取り消しに同じ回数の操作が必要になる 入れ子にした場合は一番外側だけが
        履歴に載る

        途中のコマンドが失敗したら、この ``with`` に入る前の状態へ戻す 複数の
        クリップをまとめて動かして 3 本目で失敗したとき、2 本だけ動いた中途半端な
        状態が 1 段として残ると、何が起きたのか分からなくなる

        ``merge`` については :meth:`begin_checkpoint` を参照
        """
        start = self._project
        self.begin_checkpoint(label, merge=merge)
        try:
            yield
        except BaseException:
            if self._project is not start:
                self._set_project(start)
            raise
        finally:
            self.end_checkpoint()

    def begin_checkpoint(self, label: str, *, merge: bool = False) -> None:
        """チェックポイントを開く :meth:`end_checkpoint` と必ず対にする

        ``with`` で囲めない場合のための入口 AI エージェントの 1 往復は、開始と
        終了が別のスレッドから・別の時点で来るので、文の構造には収まらない

        ``merge`` が真で、直前に積んだ同じ名前の段がまだ一番上にあれば、新しい段を
        積まずにそこへまとめる ホイールを回し続けたぶんが 1 段ずつ積まれると、
        戻すのに同じ回数だけ取り消すことになる
        """
        if self._checkpoint_depth == 0:
            self._checkpoint_label = label
            self._checkpoint_before = self._project
            self._checkpoint_merge = merge
        self._checkpoint_depth += 1

    def end_checkpoint(self) -> None:
        """チェックポイントを閉じる 開いていなければ何もしない"""
        if self._checkpoint_depth == 0:
            return
        self._checkpoint_depth -= 1
        if self._checkpoint_depth > 0:
            return

        before = self._checkpoint_before
        self._checkpoint_before = None
        # 何も変わらなかったチェックポイントは履歴に残さない
        # 「取り消しても何も起きない」段が挟まると操作感が悪い
        if before is None or before is self._project:
            # 何も変わらなかった操作でも、続けて行う操作の区切りにはなる（続きとして
            # 頼まれたものでも） ここで区切らないと、上限に張り付いて変わらなかった
            # ホイールのあとの操作が、前の続けた操作の段へまとまってしまう
            self._merge_anchor = None
            return
        top = self._undo[-1] if self._undo else None
        continued = (
            self._checkpoint_merge
            and top is not None
            and top is self._merge_anchor
            and top.label == self._checkpoint_label
        )
        if not continued:
            self._push_undo(HistoryEntry(self._checkpoint_label, before))
        self._redo.clear()

    @property
    def in_checkpoint(self) -> bool:
        return self._checkpoint_depth > 0

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0

    @property
    def undo_label(self) -> str | None:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> str | None:
        return self._redo[-1].label if self._redo else None

    def undo(self) -> Project:
        """直前の操作を取り消す 履歴が空なら何もしない"""
        if not self._undo:
            return self._project
        entry = self._undo.pop()
        self._redo.append(HistoryEntry(entry.label, self._project))
        self._merge_anchor = None
        self._set_project(entry.before)
        return self._project

    def redo(self) -> Project:
        """取り消した操作をやり直す"""
        if not self._redo:
            return self._project
        entry = self._redo.pop()
        self._undo.append(HistoryEntry(entry.label, self._project))
        self._merge_anchor = None
        self._set_project(entry.before)
        return self._project

    @property
    def history_labels(self) -> tuple[str, ...]:
        """古い順に並べた Undo 可能な操作名 履歴パネル用"""
        return tuple(entry.label for entry in self._undo)

    @property
    def past_projects(self) -> tuple[Project, ...]:
        """履歴（取り消しとやり直し）に残ったプロジェクト 古い順

        AI へ送る結果の伏せ字（#342）が、今の作品から消えた素材の名前も拾うのに使う
        履歴の名前（「配置: 〜」）には、消した素材の名前が残るため
        """
        return tuple(entry.before for entry in (*self._undo, *reversed(self._redo)))

    def reset(self, project: Project) -> None:
        """別のプロジェクトを読み込み、履歴を捨てる ファイルを開いたときに使う"""
        if self._checkpoint_depth:
            raise RuntimeError("チェックポイントの途中でプロジェクトを差し替えられない")
        self._undo.clear()
        self._redo.clear()
        self._merge_anchor = None
        self._set_project(project)

    def _push_undo(self, entry: HistoryEntry) -> None:
        self._undo.append(entry)
        self._merge_anchor = entry
        if len(self._undo) > self._history_limit:
            del self._undo[: len(self._undo) - self._history_limit]

    def _set_project(self, project: Project) -> None:
        self._project = project
        self._remember(project)
        for listener in list(self._listeners):
            listener(project)
