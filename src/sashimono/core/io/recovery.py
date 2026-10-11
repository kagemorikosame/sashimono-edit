"""保存していない作業の退避と、保存したファイルの世代バックアップ

2 つは守るものが違う

- **退避** 保存していない変更 落ちたときに、次の起動で拾い直す
- **バックアップ** 上書き保存で消える前の中身 「さっきの保存で壊した」を戻す

どちらもプロジェクトの隣ではなく ``%LOCALAPPDATA%\\Sashimono`` に置く 隣に置くと、
プロジェクトを同期フォルダ（OneDrive など）に置いている人のところで、数十秒おきの
退避がそのまま同期されて回線と相手のフォルダを埋める 本人が設定で別の置き場を選べるが
（``ui/backup_settings.py``）、ここは置き場を引数で受け取るだけで、選び方は知らない

Qt を使わない 退避の判断はテストで直接確かめたいので、ここは素の Python で書く
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sashimono.core import userdirs
from sashimono.core.io.locks import HeldLock, is_held, try_hold
from sashimono.core.io.serialize import LEGACY_SUFFIXES, SUFFIX, save_project
from sashimono.core.model import Project

__all__ = [
    "BACKUP_GENERATIONS",
    "RecoveryEntry",
    "RecoverySession",
    "TrimItem",
    "backup_before_save",
    "backup_folder",
    "backups_over",
    "default_state_root",
    "discard",
    "discard_items",
    "find_orphans",
    "find_orphans_in",
    "folder_problem",
    "forget_empty_roots",
    "latest_backup_folder",
    "mark_offered",
    "plan_prune",
    "plan_trim",
    "project_presence_dir",
    "remember_root",
    "remembered_roots",
    "state_usage",
    "tidy_orphans",
    "trim_state",
]

#: 1 つのプロジェクトについて残すバックアップの数
#: 保存は頻繁に押すものなので、少ないとすぐ押し流される 1 本は数百 KB 程度
BACKUP_GENERATIONS = 20


def default_state_root() -> Path:
    """退避とバックアップを置く既定の場所

    キャッシュ（``cache``）と同じ ``%LOCALAPPDATA%\\Sashimono`` の下だが、別のフォルダに
    分ける キャッシュは消してよいものとして案内するので、同じ所にあると一緒に消される
    """
    return userdirs.state_root()


@dataclass(frozen=True, slots=True)
class RecoveryEntry:
    """前回のどこかの起動が残していった退避 1 件

    メモ（``.json``）が無いときは ``name`` が「無題」、``saved_at`` がファイルの
    更新時刻になる 最初の退避の途中で落ちると、中身だけ書けてメモが無い
    """

    session: str
    path: Path
    #: 退避する前に開いていたファイル 1 度も保存していなければ ``None``
    source: Path | None
    name: str
    saved_at: datetime
    #: 起動したときに復元を 1 度でも勧めたか（:func:`mark_offered`） 勧めていない退避は、
    #: 日数や容量で片付けない 本人が一度も見ていない作業を黙って消すことになる
    offered: bool = False
    #: 初めて勧めた時刻 勧めていなければ ``None`` 日数の片付けはここからも数える
    offered_at: datetime | None = None


class RecoverySession:
    """この起動の退避先

    起動ごとに別の名前を使う 同時に 2 つ開いたときに、互いの退避を上書きしない

    生きているかどうかは、開いたままにしている錠のファイルで見分ける Windows では
    開いているファイルを消せないので、「消せたら持ち主はもういない」と判断できる
    プロセス番号で見る方法は使わない 番号は使い回されるうえ、Windows の
    ``os.kill`` は存在の確認ではなく強制終了になる
    """

    def __init__(self, root: Path | None = None) -> None:
        self._folder = (root if root is not None else default_state_root()) / "recovery"
        self._folder.mkdir(parents=True, exist_ok=True)
        self.session = uuid.uuid4().hex
        # 名前は起動ごとに違うので、取れないのは何かが壊れているときだけ
        lock = try_hold(self._lock_path(self._folder, self.session))
        if lock is None:
            raise RuntimeError(f"退避の錠を作れない: {self._folder}")
        self._lock: HeldLock | None = lock

    @property
    def path(self) -> Path:
        return self._folder / f"{self.session}{SUFFIX}"

    def save(self, project: Project, source: Path | None) -> None:
        """いまの状態を退避する 書き込み中に落ちても前回の退避は残る"""
        save_project(project, self.path)
        meta = {
            "source": str(source) if source is not None else None,
            "name": project.name,
            # 秒で丸めると、同じ秒に落ちた 2 つの窓の退避で新旧の順が決まらない
            # 前の版が書いた秒単位のメモも fromisoformat でそのまま読める
            "saved_at": datetime.now().isoformat(timespec="microseconds"),
        }
        _write_atomic(self._meta_path(self._folder, self.session), json.dumps(meta))

    def clear(self) -> None:
        """退避を消す 保存した直後など、守るものが無くなったときに呼ぶ"""
        for path in (self.path, self._meta_path(self._folder, self.session)):
            path.unlink(missing_ok=True)

    def close(self) -> None:
        """正常に終わる 退避も錠も残さない

        退避を消せなくても（置き場のドライブを抜いた など）錠は手放す 手放さないと、
        この起動が終わるまで錠のファイルを開いたままになる
        """
        try:
            self.clear()
        finally:
            if self._lock is not None:
                self._lock.release()
                self._lock = None

    @staticmethod
    def _lock_path(folder: Path, session: str) -> Path:
        return folder / f"{session}.lock"

    @staticmethod
    def _meta_path(folder: Path, session: str) -> Path:
        return folder / f"{session}.json"


def find_orphans(root: Path | None = None) -> list[RecoveryEntry]:
    """持ち主が終わっているのに残っている退避 新しい順

    正常に終わった起動は退避を消していくので、ここに出てくるのは落ちたか
    強制終了されたものだけ 読めない退避は黙って飛ばす（消しはしない）
    """
    folder = (root if root is not None else default_state_root()) / "recovery"
    if not folder.is_dir():
        return []

    # メモと中身のどちらか一方しか無いものも拾う 中身から先に書くので、最初の
    # 退避の途中で落ちると中身だけが残る メモだけを数えるとそれを見落とす
    # 旧い拡張子の退避も拾う 改名前の版で落ちた退避は、置き場を引き継いだあと
    # 旧い拡張子のまま残っている 拾わないと、その作業は復元を勧められずに埋もれる
    sessions = {path.stem for path in folder.glob("*.json")} | {
        path.name.removesuffix(suffix)
        for suffix in (SUFFIX, *LEGACY_SUFFIXES)
        for path in folder.glob(f"*{suffix}")
    }
    found: list[RecoveryEntry] = []
    for session in sessions:
        if _is_alive(folder, session):
            continue
        meta_path = folder / f"{session}.json"
        project_path = _saved_project(folder, session)
        if project_path is None:
            # 中身の無いメモと印 消せなくても検索は止めない（次の検索でまた消しに行く）
            # 1 つずつ包む まとめて包むと、メモを消せないときに印を消しに進まず、印だけが残る
            with contextlib.suppress(OSError):
                meta_path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                _offered_path(folder, session).unlink(missing_ok=True)
            continue
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                saved_at = datetime.fromisoformat(str(meta["saved_at"]))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        else:
            meta = {}
            saved_at = datetime.fromtimestamp(project_path.stat().st_mtime)
        source = meta.get("source")
        found.append(
            RecoveryEntry(
                session=session,
                path=project_path,
                source=Path(source) if isinstance(source, str) else None,
                name=str(meta.get("name") or "無題"),
                saved_at=saved_at,
                offered=(offered_at := _offered_at(folder, session)) is not None,
                offered_at=offered_at,
            )
        )
    return sorted(found, key=lambda entry: entry.saved_at, reverse=True)


def find_orphans_in(roots: list[Path]) -> list[RecoveryEntry]:
    """いくつかの置き場の退避をまとめて 新しい順 同じ置き場を 2 度数えない

    置き場を設定で変えた人のところでは、退避は選んだ置き場と既定の置き場の両方にありうる
    選んだ置き場へ書けずに既定へ戻した起動が落ちると、退避は既定の側に残る
    片方しか見ないと、その作業は復元を勧められずに埋もれる
    """
    seen: set[str] = set()
    found: list[RecoveryEntry] = []
    for root in roots:
        key = os.path.normcase(str(Path(root).resolve()))
        if key in seen:
            continue
        seen.add(key)
        found.extend(find_orphans(root))
    return sorted(found, key=lambda entry: entry.saved_at, reverse=True)


#: 退避を書いたことのある置き場を覚えるファイル（既定の置き場の中） 名前は 1 か所で持つ
_PLACES = "places.json"

#: 覚えておく置き場の数 置き場を何度も変える人でも、ファイルが際限なく伸びない
_PLACES_KEPT = 20


def remember_root(root: Path, registry: Path | None = None) -> None:
    """退避を書く置き場を覚える 次の起動の復元はここに挙げた置き場もすべて探す

    設定の置き場だけを探すと、置き場を A から B へ変えた後に A へ書いた退避（B へ書けずに
    A へ戻した、別の窓が A のまま動いていた など）が、落ちた後に見つからない
    覚えるのは既定の置き場の中 既定の置き場は設定で動かないので、どの設定でも同じ所を読める
    """
    places = remembered_roots(registry)
    key = _root_key(root)
    places = [Path(root), *(place for place in places if _root_key(place) != key)]
    _write_places(places[:_PLACES_KEPT], registry)


def remembered_roots(registry: Path | None = None) -> list[Path]:
    """覚えている置き場 新しく使った順 読めなければ空（復元の検索を止めない）"""
    path = (registry if registry is not None else default_state_root()) / _PLACES
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [Path(item) for item in data if isinstance(item, str) and Path(item).is_absolute()]


def forget_empty_roots(keep: list[Path], registry: Path | None = None) -> None:
    """覚えている置き場のうち、退避が 1 つも無い所を忘れる ``keep`` は残す

    動いている窓の錠も退避の置き場にあるので、錠が残っている置き場は忘れない
    （その窓が後で落ちたときに、探す所から外れないように）
    """
    kept = {_root_key(root) for root in keep}
    places = remembered_roots(registry)
    remaining = [
        place for place in places if _root_key(place) in kept or _has_files(place / "recovery")
    ]
    if len(remaining) != len(places):
        _write_places(remaining, registry)


def _root_key(root: Path) -> str:
    return os.path.normcase(str(Path(root).resolve()))


def _has_files(folder: Path) -> bool:
    try:
        return any(path.is_file() for path in folder.iterdir())
    except FileNotFoundError:
        # フォルダが無い ドライブごと見えない（抜いた・回線が切れた）ときは忘れない
        # 挿し直せば退避が戻ってくる ドライブはあるのにフォルダが無いなら、何も残っていない
        return not Path(folder.anchor).exists()
    except OSError:
        # 読めないだけでは忘れない
        return True


def _write_places(places: list[Path], registry: Path | None) -> None:
    base = registry if registry is not None else default_state_root()
    base.mkdir(parents=True, exist_ok=True)
    _write_atomic(base / _PLACES, json.dumps([str(place) for place in places], ensure_ascii=False))


def folder_problem(root: Path) -> str | None:
    """退避とバックアップの置き場として書けるか 書けなければその理由 書ければ ``None``

    フォルダがあるかどうかではなく、実際に 1 つ書いて消して確かめる 読み取り専用の
    フォルダ・抜いたドライブ・切れたネットワークの置き場は、あるように見えても書けない
    書けない所を選んだまま気付かないと、退避が黙って止まる
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".sashimono-write-test-{uuid.uuid4().hex}"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        return str(exc) or type(exc).__name__
    return None


def discard(entry: RecoveryEntry) -> None:
    """退避を捨てる 復元し終えたときと、要らないと言われたときに呼ぶ

    中身を先に消す 中身を消せなければ ``OSError`` を上げ、何も消していない
    中身を消せた後のメモ・錠・印は消せなくても上げない（中身の無いメモは次の検索で片付く）
    途中で上げると、中身は消えたのに「消せなかった」と伝わる
    """
    _remove_files(_session_files(entry))


def _remove_files(paths: tuple[Path, ...]) -> None:
    """先頭（中身）を消し、残りは消せる分だけ消す 中身を消せなければ ``OSError``"""
    head, *rest = paths
    head.unlink(missing_ok=True)
    for path in rest:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def _session_files(entry: RecoveryEntry) -> tuple[Path, ...]:
    """その退避に属するファイル 中身・メモ・錠・勧めた印"""
    folder = entry.path.parent
    return (
        entry.path,
        folder / f"{entry.session}.json",
        folder / f"{entry.session}.lock",
        _offered_path(folder, entry.session),
    )


def _offered_path(folder: Path, session: str) -> Path:
    return folder / f"{session}.offered"


def _offered_at(folder: Path, session: str) -> datetime | None:
    """初めて勧めた時刻（印を作った時刻） 印が無ければ ``None``"""
    try:
        return datetime.fromtimestamp(_offered_path(folder, session).stat().st_mtime)
    except OSError:
        return None


def mark_offered(entry: RecoveryEntry) -> None:
    """起動したときに復元を勧めた印を付ける 日数と容量の片付けは、印の付いた退避だけを見る

    メモ（``.json``）に書き足さず別のファイルにするのは、メモが無い退避（最初の退避の
    途中で落ちた物）にも印を付けるため

    印が既にあれば触らない 印の更新時刻が「初めて勧めた時刻」で、日数の片付けはそこから
    数える 勧めるたびに付け直すと、いつまでも片付かない（逆に古い退避を初めて見せた直後に
    片付けないための時刻でもある）
    """
    with (
        contextlib.suppress(FileExistsError),
        _offered_path(entry.path.parent, entry.session).open("xb"),
    ):
        pass


def tidy_orphans(
    roots: list[Path],
    days: int,
    now: datetime | None = None,
    *,
    failed: list[RecoveryEntry] | None = None,
) -> list[RecoveryEntry]:
    """復元を勧めたのに残っている退避のうち、``days`` 日より前の物を片付ける 片付けた物を返す

    ``days`` が 0 なら何もしない（既定 前からの動き） 勧めていない退避は古くても残す
    長く起動しなかった人のところで、一度も見ていない落ちた作業が消えないように

    1 件消せなくても止めずに続ける 消せなかった物は ``failed`` へ足す 途中で上げると、
    それまでに消した物が知らされない
    """
    if days <= 0:
        return []
    limit = (now if now is not None else datetime.now()) - timedelta(days=days)
    tidied: list[RecoveryEntry] = []
    for entry in find_orphans_in(roots):
        # 退避した時刻と初めて勧めた時刻の遅い方から数える 退避した時刻だけで数えると、
        # 長く起動しなかった人が初めて見た古い退避に〔あとで決める〕を選んでも、
        # その直後の片付けで消える（「次の起動でまた出る」が守られない）
        if entry.offered_at is not None and max(entry.saved_at, entry.offered_at) <= limit:
            try:
                discard(entry)
            except OSError:
                if failed is not None:
                    failed.append(entry)
                continue
            tidied.append(entry)
    return tidied


def _is_alive(folder: Path, session: str) -> bool:
    return is_held(folder / f"{session}.lock")


def _saved_project(folder: Path, session: str) -> Path | None:
    """その起動の退避の中身 新しい拡張子を先に見る 無ければ ``None``"""
    for suffix in (SUFFIX, *LEGACY_SUFFIXES):
        path = folder / f"{session}{suffix}"
        if path.is_file():
            return path
    return None


def project_presence_dir(target: Path, root: Path | None = None) -> Path:
    """そのプロジェクトを開いている窓が、1 枚ずつ錠を置く場所

    錠を 1 つだけ取り合う形にすると、「それでも開く」を選んだ窓が錠を持てず、
    先の窓が閉じたあとに 3 つ目の窓が警告なしで開けてしまう（PR #13） 窓ごとに
    置けば、まだ開いている窓は必ず数に入る

    プロジェクトの隣には置かない 同期フォルダに置いている人のところで、錠まで
    同期されて別の機械の窓と取り合いになる
    """
    base = (root if root is not None else default_state_root()) / "open"
    return base / _path_digest(target)


def _path_digest(target: Path) -> str:
    """場所の要約 大文字小文字や ``..`` の違いで別物にならないよう正規化してから取る"""
    resolved = os.path.normcase(str(Path(target).resolve()))
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:10]


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


# --- 世代バックアップ ------------------------------------------------------


def backup_folder(target: Path, root: Path | None = None) -> Path:
    """そのプロジェクトのバックアップを置く場所

    ファイル名だけで分けると、別のフォルダにある同じ名前のプロジェクト
    （「本編.sme」はどこにでもある）が 1 つの棚に混ざる 場所の要約を添える
    """
    base = (root if root is not None else default_state_root()) / "backups"
    digest = _path_digest(target)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(target).stem)[:40] or "project"
    return base / f"{stem}-{digest}"


def backup_before_save(
    target: Path,
    root: Path | None = None,
    *,
    keep: int = BACKUP_GENERATIONS,
    most: int | None = None,
    failed: list[Path] | None = None,
) -> Path | None:
    """上書きする前の中身を控える ``target`` がまだ無ければ何もしない

    古いものから消して ``keep`` 本に保つ 名前に時刻を入れてあるので、並べれば
    そのまま古い順になる 消すのは新しい控えを書き終えてから

    ``most`` を渡すと、1 回で消すのはその本数まで 画面からの保存は 1 を渡す
    世代の入れ替え（1 本作って 1 本消す）は決まりとして消してよいが、それより多く
    消すのは、本人に見せて確かめた時（:func:`plan_prune`）だけにするため 置き場を
    変えた先や別の窓の設定のせいで ``keep`` を超えていても、保存のたびに黙ってまとめて消さない
    """
    target = Path(target)
    if not target.is_file():
        return None
    folder = backup_folder(target, root)
    existed = folder.is_dir()
    folder.mkdir(parents=True, exist_ok=True)
    # 時刻が同じなら連番を足す Windows の Python 3.12 は時刻の刻みが約 15ms と粗く、
    # 続けて保存すると同じ名前になって前の控えを上書きしていた（CI で 3.12 だけ落ちた）
    # 連番は時刻の後ろに付けるので、名前で並べれば古い順のまま
    # 名前は排他作成（"x"）で先に押さえる 「空いているか見てから書く」の 2 段だと、
    # 同じプロジェクトを 2 つの窓で開いて同時に保存したとき、同じ名前を選んで上書きしうる
    stamp = f"{_next_stamp():%Y%m%d-%H%M%S-%f}"
    number = 0
    copied: Path | None = None
    try:
        while True:
            candidate = folder / f"{stamp}-{number:03d}{SUFFIX}"
            try:
                with candidate.open("xb"):
                    copied = candidate
                    break
            except FileExistsError:
                number += 1
        shutil.copyfile(target, candidate)
        shutil.copystat(target, candidate)
    except OSError:
        # 途中まで写した控えと、今作った空のフォルダを片付けてから上げる 残すと、
        # 〔バックアップのフォルダを開く〕が不完全な控えのある方を開き、呼ぶ側が既定の
        # 置き場へ控え直した正しい物へたどり着けない 消すのは今この場で作った物だけ
        # （前からある控えとフォルダには触らない）
        if copied is not None:
            with contextlib.suppress(OSError):
                copied.unlink(missing_ok=True)
        if not existed:
            with contextlib.suppress(OSError):
                folder.rmdir()
        raise

    # 旧い拡張子の控えも世代に数える 数えないと、改名前の控えはいつまでも消えずに残り、
    # 20 本に保つ約束が崩れる 名前は時刻から始まるので、拡張子が混ざっても古い順に並ぶ
    generations = _generations(folder)
    doomed = generations[: max(0, len(generations) - keep)]
    if most is not None:
        doomed = doomed[:most]
    for old in doomed:
        # 入れ替えで古い 1 本を消せなくても、新しい控えは書けている 上げると保存の知らせが
        # 「バックアップは作れなかった」になり、作れた控えが無いことにされる
        # 消せなかった物は ``failed`` へ足し、呼ぶ側が保存の知らせに添える（黙らない）
        try:
            old.unlink(missing_ok=True)
        except OSError:
            if failed is not None:
                failed.append(old)
    return candidate


#: この起動で最後に控えへ付けた時刻 :func:`_next_stamp` が次の時刻をこれより後にする
_last_stamp: datetime | None = None
_stamp_lock = threading.Lock()


def _next_stamp() -> datetime:
    """控えの名前に付ける時刻 この起動の中では必ず前の控えより後になる

    Windows の時計の刻みは約 15ms と粗く、選んだ置き場へ書けずに既定の置き場へ控え直すと、
    2 つの置き場の控えが同じ時刻の名前になる 同じ時刻だと、どちらが後で作った（正しい）
    控えか名前から分からない 前と同じか前より前なら 1 マイクロ秒だけ進める

    作った順を名前で持つのは、ほかの手掛かりが当てにならないため 更新時刻は元のファイルから
    写す（``copystat``）ので作った時刻ではなく、作成時刻（ctime）も同じ粗い時計で付く
    フォルダの中の連番は置き場ごとに数えるので、置き場をまたいでは比べられない
    """
    global _last_stamp
    with _stamp_lock:
        now = datetime.now()
        if _last_stamp is not None and now <= _last_stamp:
            now = _last_stamp + timedelta(microseconds=1)
        _last_stamp = now
        return now


def latest_backup_folder(target: Path, roots: list[Path]) -> Path | None:
    """そのプロジェクトの控えがある置き場のうち、いちばん新しい控えのある方のフォルダ

    フォルダがあるかでは決めない 中身の無い控え（途中で書けなかった物）しか無いフォルダを
    開くと、別の置き場にある正しい控えへたどり着けない 使える控えが無ければ ``None``

    新しさは控えの名前（時刻と連番）で比べる 名前の時刻は :func:`_next_stamp` で、同じ起動の
    中なら後で作った方が必ず後になる 別々の起動が同じ時刻に控えた（ほとんど起きない）ときは
    ``roots`` の先にある置き場（設定の置き場）を開く
    """
    best: tuple[str, Path] | None = None
    for root in roots:
        folder = backup_folder(target, root)
        usable = [path for path in _generations(folder) if _size(path) > 0]
        if not usable:
            continue
        name = usable[-1].name
        if best is None or name > best[0]:
            best = (name, folder)
    return best[1] if best is not None else None


def _generations(folder: Path) -> list[Path]:
    """そのプロジェクトの控え 古い順"""
    return sorted(
        path for suffix in (SUFFIX, *LEGACY_SUFFIXES) for path in folder.glob(f"*{suffix}")
    )


def _backup_folders(root: Path | None) -> list[Path]:
    base = (root if root is not None else default_state_root()) / "backups"
    try:
        return [folder for folder in base.iterdir() if folder.is_dir()]
    except OSError:
        return []


def _backup_label(folder: Path) -> str:
    return f"バックアップ {folder.name.rsplit('-', 1)[0]}"


def plan_prune(keep: int, root: Path | None = None) -> list[TrimItem]:
    """世代数を ``keep`` に詰めるために消す控え（全部のプロジェクト） 古い順 消しはしない

    確かめで見せた物をそのまま消す（:func:`discard_items`） 見せた数と消える数を一致させる
    詰めた後の保存は、1 本作って 1 本消す入れ替えだけになる
    """
    items: list[TrimItem] = []
    for folder in _backup_folders(root):
        generations = _generations(folder)
        label = _backup_label(folder)
        items.extend(
            TrimItem(label, (path,), _size(path), _backup_stamp(path))
            for path in generations[: max(0, len(generations) - keep)]
        )
    return sorted(items, key=lambda item: item.stamp)


def backups_over(keep: int, root: Path | None = None) -> int:
    """世代数を ``keep`` に詰めると消える控えの数（全部のプロジェクトの合計 :func:`plan_prune`）"""
    return len(plan_prune(keep, root))


def discard_items(
    items: Iterable[TrimItem], *, failed: list[TrimItem] | None = None
) -> list[TrimItem]:
    """確かめた物を消す 消せた物を返す 消せなかった物（置き場が書けない）は飛ばして続け、
    ``failed`` へ足す 途中で上げると、それまでに消した物が知らされない

    同じファイルが 2 つの一覧（世代数と容量）に入っていても 1 度だけ数える
    """
    done: list[TrimItem] = []
    seen: set[str] = set()
    for item in items:
        keys = [_root_key(path) for path in item.paths]
        if all(key in seen for key in keys):
            continue
        seen.update(keys)
        try:
            _remove_files(item.paths)
        except OSError:
            if failed is not None:
                failed.append(item)
            continue
        done.append(item)
    return done


# --- 容量の上限 -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrimItem:
    """容量の上限で片付ける物 1 つ（バックアップ 1 本か、落ちた作業の退避 1 件）"""

    #: 何が消えるのかを本人へ見せる名前
    label: str
    paths: tuple[Path, ...]
    size: int
    #: 古い順に並べる時刻
    stamp: float


def state_usage(root: Path | None = None) -> int:
    """置き場の退避とバックアップが使っている大きさ（バイト）"""
    base = root if root is not None else default_state_root()
    return sum(_size(path) for name in ("recovery", "backups") for path in _files(base / name))


def plan_trim(
    limit: int,
    root: Path | None = None,
    *,
    already: Iterable[Path] = (),
    spare: Iterable[Path] = (),
) -> list[TrimItem]:
    """容量を ``limit`` バイトに収めるために片付ける物 古い順 収まっていれば空 消しはしない

    ``already`` は同じ確かめで先に消すと決めた物（世代数で詰める控え） その分は使っている
    大きさから引き、候補からも外す 引かないと、世代数で空く分を知らずに余計に挙げる

    消さない物
    - 動いている起動の退避（開いている作業の今の退避）
    - 復元をまだ勧めていない落ちた作業の退避（本人が一度も見ていない）
    - 各プロジェクトのいちばん新しいバックアップ（最後の保存の前の中身 これが無いと
      「さっきの保存で壊した」を戻せない）

    消さない物だけで上限を超えるときは、消せる物を全部挙げて止める 上限を守るために
    大事な物まで消すと、上限を入れた意味が逆になる
    """
    base = root if root is not None else default_state_root()
    gone = {_root_key(path) for path in already}
    freed = sum(
        _size(path)
        for name in ("recovery", "backups")
        for path in _files(base / name)
        if _root_key(path) in gone
    )
    excess = state_usage(base) - freed - limit
    if excess <= 0:
        return []
    candidates: list[TrimItem] = []
    for folder in _backup_folders(base):
        # このプロジェクトのいちばん新しい控えは数から外す 世代数で消える物を除いた後の物
        remaining = [path for path in _generations(folder) if _root_key(path) not in gone]
        label = _backup_label(folder)
        candidates.extend(
            TrimItem(label, (path,), _size(path), _backup_stamp(path)) for path in remaining[:-1]
        )
    # この起動で初めて勧めた退避（``spare`` 中身の場所）は外す 〔あとで決める〕を選んだ
    # 直後に容量で消すと、「次の起動でまた出る」が守られない
    spared = {_root_key(path) for path in spare}
    for entry in find_orphans(base):
        if not entry.offered or _root_key(entry.path) in spared:
            continue
        files = tuple(path for path in _session_files(entry) if path.is_file())
        candidates.append(
            TrimItem(
                f"落ちた作業 {entry.name}",
                files,
                sum(_size(path) for path in files),
                entry.saved_at.timestamp(),
            )
        )
    chosen: list[TrimItem] = []
    for item in sorted(candidates, key=lambda item: item.stamp):
        if excess <= 0:
            break
        chosen.append(item)
        excess -= item.size
    return chosen


def trim_state(
    limit: int,
    root: Path | None = None,
    *,
    allowed: Iterable[Path] | None = None,
    failed: list[TrimItem] | None = None,
    spare: Iterable[Path] = (),
) -> list[TrimItem]:
    """容量を ``limit`` バイトに収めるよう古い物から片付ける 片付けた物を返す

    ``spare`` はこの起動で初めて勧めた退避の場所（:func:`plan_trim`）

    何を消すかの決まりは :func:`plan_trim` ``allowed`` を渡すと、その中にある物だけを消す
    本人に見せて確かめた一覧を渡す 確かめた後に置き場の中身が変わっても（退避を移した、
    など）、確かめに出していない物は消さない 消せなかった物は飛ばして ``failed`` へ足す
    """
    items = plan_trim(limit, root, spare=spare)
    if allowed is not None:
        approved = {_root_key(path) for path in allowed}
        items = [item for item in items if all(_root_key(path) in approved for path in item.paths)]
    return discard_items(items, failed=failed)


def _files(folder: Path) -> list[Path]:
    try:
        return [path for path in folder.rglob("*") if path.is_file()]
    except OSError:
        return []


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _backup_stamp(path: Path) -> float:
    """控えを作った時刻 名前の頭から読む

    ファイルの更新時刻は使わない 控えは元の更新時刻を写す（``copystat``）ので、
    「前に保存した時刻」になり、控えを作った順と食い違う
    """
    try:
        return datetime.strptime(path.name[:22], "%Y%m%d-%H%M%S-%f").timestamp()
    except ValueError:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0
