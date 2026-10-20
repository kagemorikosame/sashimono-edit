"""現在のプロジェクトと Undo 履歴を保持する

モデルが frozen なので、履歴は「変更前のプロジェクト」をそのまま持つだけで足りる
構造共有により、1 クリップの変更で複製されるのは根までの経路上のオブジェクトだけなので、
数百段の履歴でもメモリは問題にならない
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sashimono.core.commands.base import Command
from sashimono.core.model import MediaId, Project

__all__ = ["Author", "Document", "HistoryEntry", "SeenMedia"]

#: 保持する履歴の上限 これを超えた分は古い方から捨てる
DEFAULT_HISTORY_LIMIT = 200


class Author(Enum):
    """履歴の段を誰が変えたか（#333）

    AI の取り消しとやり直しは、人が手で変えた段を戻さない（2026-10-18 利用者の決定）
    印が無いと、AI が「さっきの指示を取り消して」で人の後からの変更を巻き戻す
    アプリが自分で当てる変更（字幕起こしの結果など）も人の側に数える AI が頼んだ物でも、
    AI の段に混ぜると AI の指示を戻したときに一緒に消える
    """

    PERSON = "person"
    AI = "ai"


@dataclass(frozen=True, slots=True)
class SeenMedia:
    """文書が一度でも持った素材の、名前と場所だけの軽い記録（:attr:`Document.seen_media`）"""

    id: MediaId
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """履歴 1 段 ``before`` はこの操作を行う前のプロジェクト ``by`` は誰が変えたか"""

    label: str
    before: Project
    by: Author = Author.PERSON


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
        self._checkpoint_by = Author.PERSON
        #: 最後に積んだ段 続けて行う操作（ホイールで高さを変えるなど）は、この段が
        #: まだ一番上に残っているときだけ同じ段へまとめる 取り消したあとの段へ
        #: まとめると、関係の無い古い操作と一緒に戻ってしまう
        self._merge_anchor: HistoryEntry | None = None
        #: 開いた段の中の小さな段（道具 1 回分・コマンド 1 つ分） 閉じると 1 段にまとまる
        #: AI の 1 回の指示は段を開いたまま進むので、その間の取り消しはここから戻す
        #: ここを持たずに段の外の履歴を引き抜くと、閉じた後の履歴から 1 段が消える（#333）
        self._inner_undo: list[HistoryEntry] = []
        self._inner_redo: list[HistoryEntry] = []
        self._inner_anchor: HistoryEntry | None = None
        #: 開いた段の中で何か変えたか 変えた後の段の外のやり直しは、もう当てられない
        #: （当てると、段の中の変更を飛ばして古い形へ跳ぶ） ただし段が断られて元へ戻ったら
        #: 変えなかったことにする 断られただけで、前に取り消した物をやり直せなくなる
        self._inner_touched = False
        #: 開いた段の中で、もう 1 つ内側に開いた段（道具 1 回分のまとめ）の始まり
        self._nested_before: Project | None = None
        self._nested_label = ""
        self._nested_merge = False
        self._nested_by = Author.PERSON
        #: 開いた段が別の人の変更で割れたとき、先に積んだ前半の段の ``before`` 段の外へ
        #: 積んでも、この段（指示）の物として数える（:attr:`undo_in_checkpoint`）
        #: 段を覚えず ``before`` で見分けるのは、取り消して・やり直すと段が作り直されるため
        self._earlier_parts: list[Project] = []
        #: この文書が一度でも持った素材と作品の名前 :attr:`seen_media` を見る
        self._seen_media: set[SeenMedia] = set()
        self._seen_names: set[str] = set()
        #: 別の作品へ差し替えた（:meth:`reset`）回数 :attr:`generation` を見る
        self._generation = 0
        self._remember(project)

    @property
    def generation(self) -> int:
        """別の作品へ差し替えるたびに 1 つ増える数

        作品の外で、開いている作品に付けて持つ物（AI の作業の計画 #340）が、今の作品の物かを
        見分けるのに使う 文書は作品を開き直しても同じ物を使い回すので、文書だけで見ると
        前の作品の計画を次の作品で続けてしまう
        """
        return self._generation

    @property
    def seen_media(self) -> tuple[SeenMedia, ...]:
        """この文書が一度でも持った素材の ID・名前・場所（消した物・開き直す前の作品の物も）

        AI へ送る結果の伏せ字（#342）が、作品の外の記録（字幕起こしの知らせ・履歴の名前）に
        残った名前を拾うのに使う 履歴は上限で古い段が落ち、別の作品を開くと消えるので、
        履歴からは拾い切れない 素材そのもの（:class:`MediaItem`）は持たない 字幕と単語の
        時刻まで抱え、起こし直すたびに古い字幕が溜まっていく
        """
        return tuple(sorted(self._seen_media, key=lambda seen: (str(seen.id), seen.name)))

    @property
    def seen_project_names(self) -> tuple[str, ...]:
        """この文書が一度でも持った作品の名前（名前を変える前・開き直す前の物も）"""
        return tuple(sorted(self._seen_names))

    def _remember(self, project: Project) -> None:
        for media in project.media:
            # 名前を変えた素材は変える前の名前も残す（ID だけで見ると前の名前を忘れる）
            self._seen_media.add(SeenMedia(media.id, media.name, media.path))
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

    def execute(self, command: Command, *, by: Author | None = None) -> Project:
        """コマンドを適用し、新しいプロジェクトを返す ``by`` は誰が変えたか

        ``by`` を省くと、開いている段の印を継ぐ（段の外なら人） 編集画面の人の操作は
        ``Author.PERSON`` を書く 書かないと、AI の指示の途中の人の変更が AI の段に混ざる
        まとめ（段の中の段）の中では、まとめの印が勝つ（1 つのまとめは 1 人の変更）
        コマンドが例外を投げた場合、プロジェクトも履歴も一切変わらない
        何も変えなかったコマンド（同じプロジェクトを返した）は履歴に積まない 積むと、
        取り消しても何も起きない段が挟まり、人の段に数えられて AI の取り消しを止める
        """
        updated = command.apply(self._project)
        if updated is self._project:
            return updated

        author = by or (self._checkpoint_by if self._checkpoint_depth else Author.PERSON)
        if self._checkpoint_depth == 0:
            self._push_undo(HistoryEntry(command.label, self._project, author))
            self._redo.clear()
        elif self._checkpoint_depth == 1:
            if author is not self._checkpoint_by:
                self._land_other(command.label, self._project, updated, author, merge=False)
            else:
                self._push_inner(HistoryEntry(command.label, self._project, author), merge=False)

        self._set_project(updated)
        return updated

    @contextmanager
    def checkpoint(
        self, label: str, *, merge: bool = False, by: Author | None = None
    ) -> Iterator[None]:
        """複数のコマンドを 1 回の Undo でまとめて戻せるようにする

        AI エージェントが 1 つの指示で何十回も編集を行うため、これがないと
        取り消しに同じ回数の操作が必要になる 入れ子にした場合は一番外側だけが
        履歴に載る

        途中のコマンドが失敗したら、この ``with`` に入る前の状態へ戻す 複数の
        クリップをまとめて動かして 3 本目で失敗したとき、2 本だけ動いた中途半端な
        状態が 1 段として残ると、何が起きたのか分からなくなる

        ``merge`` と ``by`` については :meth:`begin_checkpoint` を参照
        """
        start = self._project
        # 一番外の段なら、始めた時の履歴を覚える 段の途中で別の人の変更を分けると、段を
        # その場で履歴へ積む（_land_other） 断られて形だけ戻すと、戻した後の形に合わない段が残る
        outermost = self._checkpoint_depth == 0
        history = (list(self._undo), list(self._redo), self._merge_anchor)
        self.begin_checkpoint(label, merge=merge, by=by)
        try:
            yield
        except BaseException:
            if outermost and self._checkpoint_depth == 1:
                # 一番外の段が断られた 段の中の小さな段は戻した先に無いので捨て、
                # 履歴も始めた時へ戻す 残すと、閉じる所で前に取り消した物のやり直しまで消し、
                # 途中で積んだ段が戻した後の形に合わないまま残る
                self._reset_inner()
                self._undo, self._redo, self._merge_anchor = history
                self._checkpoint_before = start
            if self._project is not start:
                self._set_project(start)
            raise
        finally:
            self.end_checkpoint()

    def begin_checkpoint(
        self, label: str, *, merge: bool = False, by: Author | None = None
    ) -> None:
        """チェックポイントを開く :meth:`end_checkpoint` と必ず対にする

        ``with`` で囲めない場合のための入口 AI エージェントの 1 往復は、開始と
        終了が別のスレッドから・別の時点で来るので、文の構造には収まらない

        ``merge`` が真で、直前に積んだ同じ名前の段がまだ一番上にあれば、新しい段を
        積まずにそこへまとめる ホイールを回し続けたぶんが 1 段ずつ積まれると、
        戻すのに同じ回数だけ取り消すことになる

        ``by`` は誰の変更か 段は 1 人の変更だけを持つ AI の段（AI の指示）を開いている間に
        人が変えた物は、その段に混ぜずに人の段として分けて積む（2026-10-18 利用者の決定）
        混ぜると、AI の指示を取り消したときに人の変更まで消える 逆に人の段の中の AI の変更も
        分ける 省くと開いている段の印を継ぐ（段の外なら人）
        """
        if self._checkpoint_depth == 0:
            self._checkpoint_label = label
            self._checkpoint_before = self._project
            self._checkpoint_merge = merge
            self._checkpoint_by = by or Author.PERSON
            self._reset_inner()
            self._earlier_parts = []
        elif self._checkpoint_depth == 1:
            self._nested_before = self._project
            self._nested_label = label
            self._nested_merge = merge
            self._nested_by = by or self._checkpoint_by
        self._checkpoint_depth += 1

    def end_checkpoint(self) -> None:
        """チェックポイントを閉じる 開いていなければ何もしない"""
        if self._checkpoint_depth == 0:
            return
        self._checkpoint_depth -= 1
        if self._checkpoint_depth == 1:
            self._close_nested()
            return
        if self._checkpoint_depth > 0:
            return

        before = self._checkpoint_before
        self._checkpoint_before = None
        inner_redo = list(self._inner_redo)
        touched = self._inner_touched
        self._reset_inner()
        self._earlier_parts = []
        changed = before is not None and before is not self._project
        if touched:
            # 段の中で変えた後は、段の外の古いやり直しは当てられない 段の中で最後に
            # 取り消した物だけを、閉じた後も人が Ctrl+Y でやり直せるよう残す
            # 段の中で変えていなければ（段の外を取り消し・やり直しで辿っただけなら）、
            # 段の外のやり直しの残りはそのまま残す 新しい変更が無いのに捨てる訳が無い
            self._redo = inner_redo
        # 何も変わらなかったチェックポイントは履歴に残さない
        # 「取り消しても何も起きない」段が挟まると操作感が悪い
        if not changed or before is None:
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
            and top.by is self._checkpoint_by
        )
        if not continued:
            self._push_undo(HistoryEntry(self._checkpoint_label, before, self._checkpoint_by))

    def _reset_inner(self) -> None:
        self._inner_undo.clear()
        self._inner_redo.clear()
        self._inner_anchor = None
        self._inner_touched = False

    def _close_nested(self) -> None:
        """開いた段の中の段（道具 1 回分）を閉じ、変わっていれば小さな段として積む"""
        before = self._nested_before
        self._nested_before = None
        if before is None or before is self._project:
            self._inner_anchor = None
            return
        if self._nested_by is not self._checkpoint_by:
            self._land_other(
                self._nested_label, before, self._project, self._nested_by, merge=self._nested_merge
            )
            return
        entry = HistoryEntry(self._nested_label, before, self._nested_by)
        self._push_inner(entry, merge=self._nested_merge)

    def _land_other(
        self, label: str, before: Project, after: Project, by: Author, *, merge: bool
    ) -> None:
        """開いた段の途中に別の人（``by``）が変えた ``before`` → ``after`` を、別の段として積む

        それまでの段の変更は、開いた段と同じ名前の段として先に積み、別の人の段の後の
        変更はまた同じ名前で始め直す 1 つの指示が 2 段に割れるが、別の人の変更を挟んだ
        前後の形しか持たないので、その人の変更だけを残して指示を戻すにはこう分けるしかない
        先に積んだ段は「この段の前半」として覚える（:attr:`undo_in_checkpoint`）
        呼ぶ側は変わった時だけ呼ぶ（断られた・何も変えなかった操作では割らない）
        """
        own_before = self._checkpoint_before
        if own_before is not None and own_before is not before:
            self._push_undo(HistoryEntry(self._checkpoint_label, own_before, self._checkpoint_by))
            self._earlier_parts.append(own_before)
        self._reset_inner()
        top = self._undo[-1] if self._undo else None
        continued = (
            merge
            and top is not None
            and top is self._merge_anchor
            and top.label == label
            and top.by is by
        )
        if not continued:
            self._push_undo(HistoryEntry(label, before, by))
        # 新しい変更の後は、前に取り消した物はやり直せない（人の Ctrl+Y と同じ 1 本の履歴）
        self._redo.clear()
        self._checkpoint_before = after

    def _push_inner(self, entry: HistoryEntry, *, merge: bool) -> None:
        top = self._inner_undo[-1] if self._inner_undo else None
        continued = (
            merge and top is not None and top is self._inner_anchor and top.label == entry.label
        )
        if not continued:
            self._inner_undo.append(entry)
            self._inner_anchor = entry
            if len(self._inner_undo) > self._history_limit:
                # 段の中の小さな段も上限まで 1 つの指示で何百手も積むと、手ごとの形を
                # 閉じるまで抱えて重くなる 一番古い 2 手を 1 手にまとめて数を保つ
                # 捨てると、戻し切った時に段の始まりへ届かず、段の外を引き抜いてしまう
                first, second = self._inner_undo[0], self._inner_undo[1]
                self._inner_undo[:2] = [HistoryEntry(second.label, first.before, first.by)]
        self._inner_redo.clear()
        self._inner_touched = True

    @property
    def in_checkpoint(self) -> bool:
        return self._checkpoint_depth > 0

    @property
    def steps_in_checkpoint(self) -> tuple[str, ...]:
        """開いた段の中で積んだ小さな段の名前 古い順 段を開いていなければ空

        AI の道具が「この指示で変えた物」を数えるのに使う（#333）
        """
        return tuple(entry.label for entry in self._inner_undo)

    @property
    def undo_in_checkpoint(self) -> bool:
        """次に戻す段が、開いている段（指示）の物か

        段の中の小さな段と、別の人の変更で割れて段の外へ先に積んだ前半の段が当たる
        AI の「この指示を全部戻す」は、これが真の間だけ戻す（#333）
        """
        entry = self._next_undo()
        if entry is None or not self._checkpoint_depth:
            return False
        if self._inner_undo:
            return True
        return self._is_earlier_part(entry)

    @property
    def parts_left_in_checkpoint(self) -> bool:
        """開いている段（指示）の前半の段が、まだ段の外の履歴に残っているか

        別の人の段の下にあると戻せない AI に「前半は戻していない」と伝えるのに使う
        """
        return any(self._is_earlier_part(entry) for entry in self._undo)

    def _is_earlier_part(self, entry: HistoryEntry) -> bool:
        return entry.by is self._checkpoint_by and any(
            entry.before is part for part in self._earlier_parts
        )

    @property
    def can_undo(self) -> bool:
        return self._next_undo() is not None

    @property
    def can_redo(self) -> bool:
        return self._next_redo() is not None

    @property
    def undo_label(self) -> str | None:
        entry = self._next_undo()
        return entry.label if entry is not None else None

    @property
    def redo_label(self) -> str | None:
        entry = self._next_redo()
        return entry.label if entry is not None else None

    @property
    def undo_by(self) -> Author | None:
        """次に戻す段を誰が変えたか AI の取り消しは人の段なら断る（#333）"""
        entry = self._next_undo()
        return entry.by if entry is not None else None

    @property
    def redo_by(self) -> Author | None:
        """次にやり直す段を誰が変えたか"""
        entry = self._next_redo()
        return entry.by if entry is not None else None

    @property
    def _applying(self) -> bool:
        """まとめて当てている途中（段の中の段が開いている）か

        この間に取り消すと、まとめの始まりが行き先と食い違い、閉じる所で戻した物を
        「変えた」と数える 窓の中で待つ間に Ctrl+Z が届いても、何もしないで済ませる
        """
        return self._checkpoint_depth > 1

    def _next_undo(self) -> HistoryEntry | None:
        """次に戻す段 段を開いている間は段の中の小さな段から"""
        if self._applying:
            return None
        if self._inner_undo:
            return self._inner_undo[-1]
        return self._undo[-1] if self._undo else None

    def _next_redo(self) -> HistoryEntry | None:
        """次にやり直す段 段の中で変えた後は、段の外のやり直しは当てない"""
        if self._applying:
            return None
        if self._inner_redo:
            return self._inner_redo[-1]
        if self._checkpoint_depth and self._inner_touched:
            return None
        return self._redo[-1] if self._redo else None

    def undo(self) -> Project:
        """直前の操作を取り消す 履歴が空なら何もしない

        段を開いている間は、段の中の小さな段から戻す 段の中に戻す物が無ければ段の外の
        1 段を戻し、段の始まりをそこへ動かす 動かさないと、閉じる所でその段の取り消しが
        「変えた」と数えられ、戻した段の代わりに今の段が積まれて履歴から 1 段が消える
        """
        if self._applying:
            return self._project
        if self._checkpoint_depth == 1:
            if self._inner_undo:
                entry = self._inner_undo.pop()
                self._inner_redo.append(HistoryEntry(entry.label, self._project, entry.by))
                self._inner_anchor = None
                self._set_project(entry.before)
                return self._project
            if not self._undo:
                return self._project
            # 段の中のやり直しは、段の外を戻すと順が崩れるので段の外のやり直しへ移す
            # 段の中で変えた後なら、段の外の古いやり直しは当てられないので捨てる
            if self._inner_touched:
                self._redo = list(self._inner_redo)
            else:
                self._redo.extend(self._inner_redo)
            self._inner_redo.clear()
            self._inner_touched = False
            entry = self._undo.pop()
            self._redo.append(HistoryEntry(entry.label, self._project, entry.by))
            self._merge_anchor = None
            self._checkpoint_before = entry.before
            self._set_project(entry.before)
            return self._project
        if not self._undo:
            return self._project
        entry = self._undo.pop()
        self._redo.append(HistoryEntry(entry.label, self._project, entry.by))
        self._merge_anchor = None
        self._set_project(entry.before)
        return self._project

    def redo(self) -> Project:
        """取り消した操作をやり直す 段を開いている間は :meth:`undo` と同じ順で辿る"""
        if self._applying:
            return self._project
        if self._checkpoint_depth == 1 and self._inner_redo:
            entry = self._inner_redo.pop()
            self._inner_undo.append(HistoryEntry(entry.label, self._project, entry.by))
            self._inner_anchor = None
            self._set_project(entry.before)
            return self._project
        if self._next_redo() is None:
            return self._project
        entry = self._redo.pop()
        self._undo.append(HistoryEntry(entry.label, self._project, entry.by))
        self._merge_anchor = None
        if self._checkpoint_depth == 1:
            self._checkpoint_before = entry.before
        self._set_project(entry.before)
        return self._project

    @property
    def history_labels(self) -> tuple[str, ...]:
        """古い順に並べた Undo 可能な操作名 履歴パネル用

        段を開いている間は、段の中の小さな段も後ろに並ぶ（取り消すとそこから戻る）
        """
        return tuple(entry.label for entry in (*self._undo, *self._inner_undo))

    @property
    def past_projects(self) -> tuple[Project, ...]:
        """履歴（取り消しとやり直し）に残ったプロジェクト 古い順

        AI へ送る結果の伏せ字（#342）が、今の作品から消えた素材の名前も拾うのに使う
        履歴の名前（「配置: 〜」）には、消した素材の名前が残るため 開いた段の中の
        取り消しとやり直しの分も入れる 入れないと、指示の中で取り消した物の名前が漏れる
        """
        entries = (
            *self._undo,
            *self._inner_undo,
            *reversed(self._inner_redo),
            *reversed(self._redo),
        )
        return tuple(entry.before for entry in entries)

    def reset(self, project: Project) -> None:
        """別のプロジェクトを読み込み、履歴を捨てる ファイルを開いたときに使う"""
        if self._checkpoint_depth:
            raise RuntimeError("チェックポイントの途中でプロジェクトを差し替えられない")
        self._undo.clear()
        self._redo.clear()
        self._merge_anchor = None
        self._generation += 1
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
