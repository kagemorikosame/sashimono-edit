"""外部のテンプレートを 1 つの棚に並べる

集めるのは 2 種類

* AviUtl のエイリアス — ``.exa`` ``.exa2`` ``.object``（AviUtl2 世代）
* YMM4 のアイテムテンプレート — ``.ymmt``

読み方は違うが、出てくるものは同じ :class:`~sashimono.compat.mapped.MappedObject`
なので、タイムラインへ置く処理は 1 つで済む

**字幕テンプレートは「置く」だけでなく「今のクリップに着せる」ことができる**
配布されている字幕エイリアスは、見本の文字（``字幕テキスト`` など）が入った
テキストオブジェクトとして配られている そのまま置くと、字幕を打ち直すことに
なる :func:`restyle` は文字と時間を今のクリップのまま残し、見た目だけを
入れ替える
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path, PureWindowsPath

from sashimono.compat.aviutl.exo import ExoParseError, load_exo
from sashimono.compat.aviutl.mapping import SILENT_SOUND, map_object
from sashimono.compat.aviutl.report import CompatibilityReport, global_report
from sashimono.compat.layers import heard_stream, layer_tracks
from sashimono.compat.mapped import MappedObject, fitted_effect, fitted_value
from sashimono.compat.ymm4.template import Ymm4ParseError, load_template, map_template
from sashimono.core import userdirs
from sashimono.core.commands import (
    AddClip,
    AddEffect,
    AddMedia,
    AddScene,
    AddTrack,
    Command,
    InScene,
    ParamPath,
    RemoveEffect,
    SetParam,
    SetSource,
    new_scene,
)
from sashimono.core.commands.fixed import (
    PICTURE_FIXED,
    fixed_effect,
    takes_picture_items,
    with_fixed_items,
)
from sashimono.core.commands.insert import DEFAULT_GENERATED_FRAMES
from sashimono.core.commands.layers import active_layers, media_placements, places_mixed
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    MediaItem,
    Project,
    SceneId,
    Track,
    TrackId,
    TrackKind,
    new_clip_id,
    new_group_id,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT

__all__ = [
    "MediaPlan",
    "Probe",
    "TemplateCatalog",
    "TemplateEntry",
    "TemplateError",
    "default_template_roots",
    "gather_media",
    "place",
    "restyle",
    "set_template_catalog",
    "template_catalog",
]

#: AviUtl 側で読む拡張子
_AVIUTL_SUFFIXES = (".exa", ".exa2", ".object", ".exo", ".exo2")

#: YMM4 側で読む拡張子
_YMM4_SUFFIXES = (".ymmt",)

#: 文字だけを差し替えて着せ替えるときに、テンプレート側から**取らない**設定
#:
#: 文字そのものと文字送りは、今のクリップの持ち物 見た目を変えたいだけなのに
#: 中身まで置き換わったら、それは着せ替えではない
_KEPT_ON_RESTYLE = frozenset({"text", "reveal"})

#: 時間を出すテキスト（YMM4 のタイマー）の項目 テンプレートが時間の書式を持たなければ
#: 今のクリップのまま残す 何が出るかを決める中身で、見た目ではない 既定へ戻すと、
#: タイマーに字幕の見た目を着せただけで時間が消えて文字が出る
_TIMER_ON_RESTYLE = frozenset(
    {"timer_format", "timer_start", "timer_rate", "timer_countdown", "timer_length"}
)


@dataclass(frozen=True, slots=True)
class TemplateEntry:
    """棚に並ぶテンプレート 1 つ"""

    name: str
    path: Path
    #: 置かれていたフォルダ名 配布物はフォルダで分かれているので、そのまま出す
    folder: str = ""
    #: ``"aviutl"`` か ``"ymm4"``
    source: str = "aviutl"
    #: ``.ymmt`` の中の何本目か
    #:
    #: AviUtl のエイリアスは 1 ファイル 1 本だが、YMM4 のアイテムテンプレートは
    #: **1 ファイルに何本も入っている**（手元の配布物は 17 本と 106 本だった）
    index: int = 0
    #: 棚を作るときに読めなかった理由 読めた物は空
    #:
    #: YMM4 のファイルは中を開かないと何本入っているか分からないので、読めない物は
    #: 本数を出せない それでも棚から消すと、壊れた・形の違うファイルを本人が選べず、
    #: 何が起きたかを報告することもできない ファイル 1 つを 1 項目として並べ、
    #: 選んだときに AviUtl のエイリアスと同じく「読み込めません」と理由を出す
    error: str = ""
    #: ``.ymmt`` の原本の中の位置（:attr:`ItemTemplate.origin`） 報告に書く
    #: ``index`` は読める物だけを数えた番号で、原本と照らし合わせるのには使えない
    origin: str = ""

    @property
    def label(self) -> str:
        return self.name

    def load(self, *, report: CompatibilityReport | None = None) -> list[MappedObject]:
        """中身を読んで、写した結果を返す"""
        log = report if report is not None else global_report
        if self.source == "ymm4":
            if self.error:
                raise Ymm4ParseError(self.error)
            templates = load_template(self.path)
            if not 0 <= self.index < len(templates):
                return []
            return map_template(list(templates[self.index].items), report=log)

        exo = load_exo(self.path)
        mapped = [map_object(obj, FrameRate(30, 1), report=log) for obj in exo.objects]
        return [item for item in mapped if item is not None]


class TemplateCatalog:
    """フォルダを走査して並べる"""

    def __init__(self) -> None:
        self._entries: list[TemplateEntry] = []

    def scan(self, roots: tuple[Path, ...]) -> list[TemplateEntry]:
        """走査してこの棚を入れ替える 読めないファイルは黙って飛ばす"""
        found: list[TemplateEntry] = []
        seen: set[Path] = set()
        for root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                entries = _entries_for(path, root)
                if entries:
                    seen.add(resolved)
                    found.extend(entries)
        self._entries = found
        return found

    def all(self) -> tuple[TemplateEntry, ...]:
        return tuple(self._entries)

    def folders(self) -> tuple[str, ...]:
        """出てきたフォルダ名を、並んだ順のまま重複なく"""
        names: list[str] = []
        for entry in self._entries:
            if entry.folder not in names:
                names.append(entry.folder)
        return tuple(names)

    def find(self, name: str) -> TemplateEntry | None:
        return next((entry for entry in self._entries if entry.name == name), None)


def _entries_for(path: Path, root: Path) -> list[TemplateEntry]:
    """1 ファイルから並ぶテンプレート

    AviUtl のエイリアスは 1 本 YMM4 のアイテムテンプレートは中を開いて数える
    """
    suffix = path.suffix.lower()
    relative = path.parent.relative_to(root)
    folder = str(relative) if str(relative) != "." else root.name

    if suffix in _AVIUTL_SUFFIXES:
        return [TemplateEntry(name=path.stem, path=path, folder=folder, source="aviutl")]
    if suffix not in _YMM4_SUFFIXES:
        return []

    try:
        templates = load_template(path)
    except (Ymm4ParseError, OSError) as exc:
        # 黙って捨てない 捨てると、壊れた・形の違うファイルは棚に出ず、本人は
        # 置いたはずの物が無い理由も分からず、互換の報告にも写せない
        # AviUtl のエイリアスと同じく並べておき、選んだときに理由を出す
        # 受けるのはファイルの側の事情だけ（ZIP や文字の失敗は load_template が
        # Ymm4ParseError に変えて渡す） ここで何でも受けると、読み方の誤り（型の
        # 取り違えなど）まで「ファイルが壊れている」と出て、直すきっかけを失う
        return [
            TemplateEntry(
                name=path.stem, path=path, folder=path.stem, source="ymm4", error=str(exc)
            )
        ]

    return [
        TemplateEntry(
            name=template.name or f"{path.stem} {index + 1}",
            path=path,
            # 配布物は ``アニメーション効果/振り子`` のように分類を持っている
            # ファイル名だけで並べると 100 本超が 1 つの見出しに潰れる
            folder=f"{path.stem} / {template.folder}" if template.folder else path.stem,
            source="ymm4",
            index=index,
            origin=template.origin,
        )
        for index, template in enumerate(templates)
    ]


def default_template_roots() -> tuple[Path, ...]:
    """既定で見に行くフォルダ

    スクリプトと同じ考え方で、**すでに持っている資産をコピーせずに使える**ことを
    優先する AviUtl2 や YMM4 が入っていれば、そのフォルダをそのまま見る
    """
    roots: list[Path] = []
    # APPDATA の有無で分けない（スクリプトの置き場と同じ理由 引き継ぎで写した先と揃える）
    roots.append(userdirs.config_root() / "templates")

    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        roots.append(Path(program_data) / "aviutl2" / "Alias")

    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "YukkuriMovieMaker" / "ItemTemplate")
    return tuple(roots)


_catalog = TemplateCatalog()


def template_catalog() -> TemplateCatalog:
    return _catalog


def set_template_catalog(catalog: TemplateCatalog) -> None:
    """棚を差し替える テストと、フォルダ設定を変えたときに使う"""
    global _catalog
    _catalog = catalog


#: 素材ファイルを開いて :class:`MediaItem` にする関数 開けなければ ``None``
#:
#: 互換層はファイルを開かない（:func:`~sashimono.compat.aviutl.mapping.media_paths` と
#: 同じ決まり） 開く道具は呼び出し側が渡す
type Probe = Callable[[Path], MediaItem | None]


@dataclass(frozen=True, slots=True)
class MediaPlan:
    """テンプレートが参照している素材を、プロジェクトへ登録する段取り"""

    #: 新しく登録する素材の :class:`AddMedia` 置くコマンドより先に実行する
    commands: tuple[Command, ...] = ()
    #: テンプレートに書かれたパス → 使う素材 すでに登録済みの素材も入る
    media: Mapping[str, MediaItem] = field(default_factory=dict)
    #: 見つからなかった・開けなかったパス
    missing: tuple[str, ...] = ()

    @property
    def added(self) -> tuple[MediaItem, ...]:
        """新しく登録する素材 解析や控えの作成を頼む相手"""
        return tuple(c.item for c in self.commands if isinstance(c, AddMedia))


def _media_paths(objects: list[MappedObject]) -> list[str]:
    """素材として登録するパス まとめた中身（シーン）の中も見る

    中身を自分で描くもの（音声波形など）は除く 絵は描いて作るので素材を
    クリップに結ばない 結ぶと、音声しか無い素材を映像トラックへ置くことになり
    置く時点で断られる
    """
    # 辞書で順序を保ったまま重複を落とす 一覧で ``in`` を引くと数が増えるほど遅くなる
    return list(
        dict.fromkeys(
            inner.media_path
            for item in objects
            for inner in item.walk()
            if inner.media_path and inner.clip.source is None
        )
    )


def _same_file(path: Path) -> str:
    """同じファイルかどうかを見分ける鍵

    Windows ではパスの大文字小文字を区別しない 素の文字列で比べると、
    同じ画像を書き方違いで 2 つの素材として登録してしまう
    """
    return os.path.normcase(str(path.resolve()))


def gather_media(
    objects: list[MappedObject],
    project: Project,
    probe: Probe,
    *,
    near: Path | None = None,
) -> MediaPlan:
    """テンプレートの画像・音声・動画を素材として登録する段取りを作る

    これを通さずに :func:`place` すると、素材を参照するクリップは ``media_id`` を
    持たず、置いても描かれず鳴らない

    * **同じファイルは 1 つの素材にする** プロジェクトにすでにあれば、それを使う
      テンプレートを 2 度置くたびに素材が増えると、素材一覧が同じ名前で埋まる
    * 書かれたパスに無ければ ``near``（テンプレートの置き場）で同じ名前を探す
      配布物のパスは作者の機械のもの（``C:\\Users\\作者\\…``）で、受け取った側の
      機械にはまず無い 素材を同じフォルダに添えて配る作者はいる
    """
    known = {_same_file(item.path): item for item in project.media}
    held = {
        inner.media_path
        for item in objects
        for inner in item.walk()
        if inner.media_path and inner.hold_last_frame
    }
    chosen: dict[str, MediaItem] = {}
    commands: list[Command] = []
    missing: list[str] = []
    for raw in _media_paths(objects):
        # 書かれたパスは Windows の形 ``Path`` で名前を取ると、Windows 以外では
        # ``\\`` を区切りと見ずにパス全体を名前として探しに行く
        candidates = [Path(raw)]
        if near is not None:
            candidates.append(near / PureWindowsPath(raw).name)
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            missing.append(raw)
            continue
        key = _same_file(path)
        media = known.get(key)
        if media is None:
            media = probe(path)
            if media is None:
                missing.append(raw)
                continue
            commands.append(AddMedia(media))
            known[key] = media
        elif raw in held and _lacks_video_end(media):
            media = _with_video_end(media, probe(path))
            known[key] = media
        chosen[raw] = media
    return MediaPlan(commands=tuple(commands), media=chosen, missing=tuple(missing))


def _lacks_video_end(media: MediaItem) -> bool:
    """映像の道の終わりを持たない登録済みの素材か（道の終わりを記録する前の版で登録した物）"""
    return (
        bool(media.video_streams) and not media.is_still and media.video_streams[0].end_time is None
    )


def _with_video_end(media: MediaItem, fresh: MediaItem | None) -> MediaItem:
    """登録済みの素材へ、開き直して取った映像の道の終わりを添える

    絵を止める時刻（:func:`_held_at_end`）を決めるためだけに使う 無いままだと
    コンテナの長さで見るしかなく、音の方が長い素材（映像 2 秒・音 3 秒）では止める時刻が
    映像の最後のフレームより後ろになり、止めた後もデコーダが毎フレーム終わり付近を読み直す
    素材の ``id`` はそのまま残すので、クリップは登録済みの素材に結ばれる
    プロジェクトの素材は書き換えない 書き換えるなら元に戻せるコマンドを通す必要があり、
    テンプレートを置くだけで素材一覧が変わるのは本人の予想を外れる
    長さも開き直した物に替える 版 4 までに覚えた長さはコンテナの頭から数えてあり、
    映像より早く始まる音の前置きを含む 道の終わりが分からない素材は止める時刻を長さで
    決めるので、古い長さのままだと映像の終わりより後ろを基準にして、止めるべき
    クリップを止めない
    開けないときは元のまま使う
    """
    if fresh is None:
        return media
    end = fresh.video_streams[0].end_time if fresh.video_streams else None
    first, *rest = media.video_streams
    duration = fresh.duration if fresh.duration > 0 else media.duration
    return replace(media, duration=duration, video_streams=(replace(first, end_time=end), *rest))


def place(
    objects: list[MappedObject],
    project: Project,
    *,
    at_frame: int = 0,
    track_id: TrackId | None = None,
    default_duration: int = DEFAULT_GENERATED_FRAMES,
    media: Mapping[str, MediaItem] | None = None,
    report: CompatibilityReport | None = None,
) -> list[Command]:
    """写した結果をタイムラインへ置くコマンドの列

    ``track_id`` を渡せばそのトラックへまとめて置く 渡さなければ、元の
    レイヤー番号に対応する映像トラックへ置く（無ければ作る）

    ``media`` は :func:`gather_media` で登録する素材 素材を参照するクリップへ
    ``media_id`` を結ぶ 音声しか無い素材は映像トラックでは鳴らないので、
    ``track_id`` を渡していても音声トラックへ置く

    映像と音の両方を持つ素材（YMM4 の動画アイテム :attr:`MappedObject.with_sound`）は、
    素材の読み込み（:func:`~sashimono.core.commands.insert_media`）と同じく映像の
    クリップと音のクリップへ分け、リンクで結ぶ 映像トラックへ 1 本置くだけでは
    音が鳴らない（Issue #89）

    **混合の方式のプロジェクト**（:func:`~sashimono.core.commands.layers.places_mixed`）では、
    どれも元のレイヤー番号どおりの混合トラック（レイヤー）へ 1 本ずつ置く
    （:func:`_put_on_layers`） 動画アイテムは分けずに、1 本のクリップで絵と音
    （:attr:`~sashimono.core.model.Clip.audio_stream`）を持つ 音だけの物も同じレイヤーへ、
    絵を隠して（``show_picture`` を偽）置く YMM4・AviUtl の並びをそのまま写すため
    ``track_id`` を渡したときは、そこへ置ける物をまとめて置く（レイヤーなら全部）
    """
    # 中身を持たないもの（エフェクトだけのテンプレート）は置けない
    # 空のクリップを置いても何も映らないので、:func:`restyle` で着せて使う
    objects = [item for item in objects if item.has_picture]
    if not objects:
        return []

    known = media or {}
    log = report if report is not None else global_report
    mixed = places_mixed(project)
    if not mixed:
        # 分ける方式では、音の無い素材を指す音声ファイルを置かない 元のソフトでも何も
        # 鳴らず何も描かず、置くと ``AddClip`` が断って 1 回の Undo にまとめた配置が全部
        # 取り消される 下で頭を揃える前に除く 除かずに一番早い物がこれだと、残りが
        # ``at_frame`` ではなくその分だけ後ろへ置かれる
        # 黙って落とすと、読み込んだ数が合わない理由を追えないので数えて残す
        # （混合のレイヤーは絵を隠して鳴らさずに置けるので、そのまま置く）
        kept = [item for item in objects if not _silent_sound(item, known)]
        for _ in range(len(objects) - len(kept)):
            log.note_missing(SILENT_SOUND)
        objects = kept
        if not objects:
            return []
    seen = [item for item in objects if not _is_sound(item, known)]

    # 一番早いオブジェクトが ``at_frame`` に来るように、まとめてずらす
    # エイリアスは元のタイムライン上の位置を持ったままなので、そのまま置くと
    # 指定した場所ではなく元あった場所へ行く
    origin = min(item.clip.timeline_start for item in objects)

    def timed(item: MappedObject) -> Clip:
        duration = item.clip.duration if item.has_span else default_duration
        return replace(
            item.clip,
            timeline_start=item.clip.timeline_start - origin + max(0, at_frame),
            duration=max(1, duration),
        )

    commands: list[Command] = []
    # 置くクリップを先に全部作る 音声トラックの割り当ては、分けて作った音も
    # 含めて重なりを見ないと、同じトラックへ重ねて置いて ``AddClip`` に断られる
    prepared: list[tuple[MappedObject, Clip | None, Clip | None]] = []
    # 混合の方式で置く物（レイヤーの番号とクリップ） 分けないので 1 つに 1 本
    layered: list[tuple[MappedObject, Clip]] = []
    for item in objects:
        placed = timed(item)
        linked = _media_of(item, known)
        if linked is not None:
            placed = replace(placed, media_id=linked.id)
            if item.hold_last_frame and placed.hold_at is None:
                placed = replace(placed, hold_at=_held_at_end(placed, linked, project.rate))
        if item.children:
            placed = replace(
                placed,
                scene_id=_scene_for(item, project, commands, known, log),
                # シーンの中の時刻は秒で持つ（素材のクリップと同じ決まり）
                source_in=item.scene_offset * project.rate.frame_duration,
            )
        if mixed:
            layered.append((item, _layer_clip(project, item, placed, linked, known, log)))
            continue
        if item.audio_track:
            # 分ける方式の音のクリップは、素材の 1 本目の音を読む（:func:`_split_sound`）
            # 選び直すと、今までと違う音が鳴り出す 混合の方式でだけ選んだ音を鳴らす
            log.note_missing("YMM4 の音声トラックの選択（AudioTrackIndex）")
        if _is_sound(item, known):
            # 音だけの素材を読む動画アイテムでも、止めるのは絵だけ 音のクリップには持たせない
            heard = replace(placed, hold_at=None, native_size=False)
            if linked is not None:
                # 動画を指すときに 0 番のまま鳴らすと、映像のストリームを音として読みに行く
                heard = replace(heard, stream_index=linked.audio_streams[0].index)
            prepared.append((item, None, _with_audio_effects(heard, item)))
            continue
        prepared.append((item, *_split_sound(placed, item, linked)))

    if mixed:
        _put_on_layers(project, layered, commands, track_id)
        return commands

    tracks = (
        {}
        if track_id is not None or not seen
        else _tracks_for(project, {item.layer for item in seen}, commands)
    )
    sound_tracks = _sound_tracks_for(
        project, [(item, sound) for item, _, sound in prepared if sound is not None], commands
    )

    for item, picture, sound in prepared:
        if picture is not None:
            target = track_id if track_id is not None else tracks[item.layer].id
            # 素材を置いたときと同じく、描画の欄を持たせる 読み込みが写した配置と反転は
            # 印が付いているので、足りない物（既定のままで写さなかった欄）だけが足される
            if takes_picture_items(picture):
                picture = with_fixed_items(picture, picture=True)
            commands.append(AddClip(target, picture))
        if sound is not None:
            commands.append(
                AddClip(sound_tracks[id(sound)].id, with_fixed_items(sound, sound=True))
            )
    return commands


def _held_at_end(clip: Clip, media: MediaItem, rate: FrameRate) -> Fraction | None:
    """素材の終わりを越えて読むクリップの、最後の絵の時刻 越えなければ ``None``

    越えないクリップに持たせないのは、止まらないのに設定画面へ「絵を止める」が出て、
    何を止めているのか分からなくなるため

    最後の絵の時刻は、映像の終わりの **time_base の 1 刻み手前** デコーダは「その時刻を
    越えない最後のフレーム」を返し（:meth:`~sashimono.engine.decode.VideoDecoder.frame_at`）、
    終わりちょうどでは何も返さない 最後のフレームの PTS は必ず終わりより 1 刻み以上前に
    あるので、フレームの間隔によらず最後のフレームが出る 平均のフレーム 1 つ分を引くと、
    可変フレームレートの素材で最後の間隔が平均より短いとき、1 つ前の絵で止まる

    映像の終わりは映像の道の長さ
    （:attr:`~sashimono.core.model.VideoStreamInfo.end_time`）で見る コンテナの
    長さ（:attr:`MediaItem.duration`）で見ると、音の方が長い素材で最後の映像フレームより
    後ろを指し、止めた後もデコーダが毎フレーム終わり付近へシークし直してデコードする
    道の終わりもコンテナの長さも、素材の頭（:func:`~sashimono.engine.decode.probe.media_origin`）
    から数えてあり、クリップの ``source_in``・YMM4 の ``ContentOffset``・デコーダが読む時刻と
    同じ数え方（Issue #123） 頭が 0 より後ろの素材（分割して書き出した物）でも、頭 5 秒・
    長さ 2 秒なら映像の終わりは 2 秒 PTS そのまま（7 秒）で数えると、デコーダの映像の
    終わりを越えた所で止めることになり、止めた所から何も映らない
    道の長さが分からない素材（古いプロジェクトに入っていた素材など）はコンテナの長さで見る

    長さの分からない素材（0 と読めた物）と静止画は止めない 静止画はもともと
    いつでも同じ絵で、長さの分からない素材はどこが最後か決められない
    """
    if media.is_still or media.duration <= 0 or not media.video_streams:
        return None
    # 映像のクリップが読むのは最初の映像ストリーム（:func:`_split_sound` と同じ）
    stream = media.video_streams[0]
    end = media.duration if stream.end_time is None else stream.end_time
    if clip.source_out(rate) <= end:
        return None
    # 1 刻みがフレームより長い（壊れた time_base）ときはフレーム 1 つ分に抑える 大きく引くと
    # 最後より前の絵で止まる 0 以下の刻みは引いても終わりちょうどになり、何も映らない
    frame = stream.frame_rate.frame_duration
    tick = stream.time_base if 0 < stream.time_base < frame else frame
    return max(Fraction(0), end - tick)


def _with_audio_effects(clip: Clip, item: MappedObject) -> Clip:
    """音のクリップへ、音量などのエフェクトを足す"""
    if not item.audio_effects:
        return clip
    return replace(clip, effects=clip.effects + item.audio_effects)


def _split_sound(
    clip: Clip, item: MappedObject, linked: MediaItem | None, *, heard: int | None = None
) -> tuple[Clip, Clip | None]:
    """映像と音を両方持つ素材を、映像のクリップと音のクリップへ分ける

    分けないと音声トラックにクリップが無いまま置かれ、音が鳴らない
    リンクで結ぶのは素材の読み込みと同じ 片方だけ動かすと絵と音がずれる

    音のクリップは映像のエフェクト（変形や色）を持たない 音に効かないものを
    持ち回ると、クリップの設定画面に効かないエフェクトが並ぶ

    ``heard`` は鳴らす音の素材の中の番号（:func:`~sashimono.compat.layers.heard_stream`）
    省くと 1 本目の音 分ける方式は今までどおり 1 本目を読む
    """
    if not item.with_sound or linked is None or not (linked.has_video and linked.has_audio):
        return clip, None
    group = new_group_id()
    picture = replace(clip, stream_index=linked.video_streams[0].index, link_group=group)
    sound = replace(
        clip,
        stream_index=linked.audio_streams[0].index if heard is None else heard,
        link_group=group,
        effects=item.audio_effects,
        after_effects=(),
        opacity=AnimatedValue(1.0),
        blend_mode="normal",
        clip_to_below=False,
        # 音は止めない（ミキサーは読まない） 持たせたままだと、音のクリップの設定画面に
        # 効かない「絵を止める」が出る
        hold_at=None,
        native_size=False,
        id=new_clip_id(),
    )
    return picture, sound


def _layer_clip(
    project: Project,
    item: MappedObject,
    placed: Clip,
    linked: MediaItem | None,
    known: Mapping[str, MediaItem],
    log: CompatibilityReport,
) -> Clip:
    """混合の方式でレイヤーへ置く 1 本のクリップ

    分ける方式と同じ絵のクリップと音のクリップをいったん作り、素材の読み込みと同じ
    まとめ方（:func:`~sashimono.core.commands.layers.media_placements`）で 1 本にする
    別に組み立てると、固定の項目の並びや大きさの決め方が、素材を置いたときのクリップと
    食い違う 同じ物から作るので、分ける方式と同じ絵・同じ音になる

    鳴らす音は :attr:`MappedObject.audio_track` 本目（YMM4 の ``AudioTrackIndex``）
    AviUtl の動画ファイルは音を持たせない（:attr:`MappedObject.with_sound` が偽）
    AviUtl は同じ動画の音を別の 音声ファイル として書くので、持たせると二重に鳴る

    音声ファイル（AviUtl の 音声ファイル・YMM4 の音声アイテム）は、素材が映像も持つ
    動画でも音だけにする AviUtl が動画の音を書くのはこの形で、絵を描かせると同じ動画が
    2 枚重なり、上の 動画ファイル の切り抜きの相手まで変わる
    """
    if _is_sound(item, known) or (item.kind == "音声ファイル" and item.clip.source is None):
        # 分ける方式の音声トラックのクリップと同じ物 止めるのは絵だけなので hold_at は持たない
        heard = replace(placed, hold_at=None, native_size=False)
        stream = heard_stream(linked, item.audio_track, log)
        sound = with_fixed_items(_with_audio_effects(heard, item), sound=True)
        if stream is not None:
            sound = replace(sound, stream_index=stream)
        ((_, merged),) = media_placements(project, None, sound)
        # 素材が見つからない・音を持たない物は鳴らす番号を持たせない 番号だけ残すと、
        # あとで別の素材へ差し替えたときに、選んでいない番号の音を探しに行く
        return replace(merged, audio_stream=stream)
    stream = heard_stream(linked, item.audio_track, log) if item.with_sound else None
    picture, sound_part = _split_sound(placed, item, linked, heard=stream)
    if takes_picture_items(picture):
        picture = with_fixed_items(picture, picture=True)
    if sound_part is not None:
        sound_part = with_fixed_items(sound_part, sound=True)
    ((_, merged),) = media_placements(project, picture, sound_part)
    return merged


def _put_on_layers(
    project: Project,
    layered: list[tuple[MappedObject, Clip]],
    commands: list[Command],
    track_id: TrackId | None,
) -> None:
    """混合の方式のクリップを、元のレイヤー番号どおりのレイヤーへ置くコマンドを積む

    ``track_id`` のトラックへは、そこで同じ絵と音になる物だけを置く レイヤーなら映る
    （音だけの物は鳴る）間は全部、映像トラック（方式を切り替えた作品に残る物）なら
    音を鳴らさず絵を描く物だけ ほかは元のレイヤーへ回す 映像トラックへ音のある物を
    置くと、置く時点で断られる
    """
    target = project.timeline.find_track(track_id) if track_id is not None else None
    # 選ばれたレイヤーでも、ミュートやソロの外で映らない・鳴らない所へは置かない 置くと、
    # 置いた直後からプレビューにも書き出しにも出ない（素材を置くときの free_layer と同じ決まり）
    shown = {t.id for t in active_layers(project, picture=True)}
    heard = {t.id for t in active_layers(project, picture=False)}

    def fits(clip: Clip) -> bool:
        if target is None:
            return False
        if target.kind is TrackKind.MIXED:
            # 絵と音を両方持つ物は両方を見る 絵だけ見ると、音声トラックのソロで
            # 鳴らなくなったレイヤーへ置き、置いた動画やシーンの音が聞こえない
            # 鳴るかは audio_stream から推さずに plays_sound で決める シーンは番号を
            # 持たずに鳴る 音だけの物は素材が見つからなくても鳴らす側に数える
            sounds = project.plays_sound(target, clip) or not clip.show_picture
            return (not clip.show_picture or target.id in shown) and (
                not sounds or target.id in heard
            )
        return target.kind is TrackKind.VIDEO and clip.show_picture and clip.audio_stream is None

    rest = [(item, clip) for item, clip in layered if not fits(clip)]
    layers = {item.layer for item, _ in rest}
    drawn = {item.layer for item, clip in rest if clip.show_picture}
    tracks = layer_tracks(project, layers, commands, heard_only=layers - drawn)
    for item, clip in layered:
        where = target.id if target is not None and fits(clip) else tracks[item.layer].id
        commands.append(AddClip(where, clip))


def _media_of(item: MappedObject, known: Mapping[str, MediaItem]) -> MediaItem | None:
    """このクリップに結ぶ素材 中身を描くもの（音声波形など）には結ばない"""
    if not item.media_path or item.clip.source is not None:
        return None
    return known.get(item.media_path)


def _is_sound(item: MappedObject, known: Mapping[str, MediaItem]) -> bool:
    """音声トラックへ置くものか

    音声ファイル（YMM4 の音声アイテムも同じ種類）は、指す素材に映像があっても音だけ
    AviUtl は動画の音の半分を、同じ .mp4 を指す音声ファイルとして書く 素材の中身で
    決めると映像トラックへ置かれ、動画がもう 1 枚描かれて音は鳴らない
    素材の無い音声を映像トラックへ置くと、あとで素材を足しても映像トラックでは鳴らない

    ほかの種類は、素材が見つかっていれば映像を持つかどうかで決める
    """
    if item.kind == "音声ファイル" and item.clip.source is None:
        return True
    linked = _media_of(item, known)
    return linked is not None and not (linked.has_video or linked.is_still)


def _silent_sound(item: MappedObject, known: Mapping[str, MediaItem]) -> bool:
    """音の無い素材を指す音声ファイルか 素材が見つからない物は分からないので偽"""
    linked = _media_of(item, known)
    return _is_sound(item, known) and linked is not None and not linked.audio_streams


def _sound_tracks_for(
    project: Project, sounds: list[tuple[MappedObject, Clip]], commands: list[Command]
) -> dict[int, Track]:
    """音声を置く音声トラック（``id(音のクリップ)`` → トラック）

    鍵をクリップにするのは、動画アイテムが映像と音の 2 本に分かれるため
    元のオブジェクトを鍵にすると、分けて作った音のクリップを引けない

    元のレイヤーの低い順に、1 つずつ空いている音声トラックを上から探す
    映像と違い、レイヤー番号をそのままトラックの番号にしない YMM4 は映像と音声を
    同じレイヤーの並びに置くので、10 段目の効果音のために音声トラックを 10 本作ることになる

    使うのは、ロックもミュートもされておらず、置く範囲がほかの音（元からある音と、
    今回先に割り当てた音の両方）と重ならないトラックだけ 重なる所やロックされた
    トラックへ置くと ``AddClip`` が断り、1 回の Undo にまとめた配置が画像も素材の登録も
    含めて全部取り消される ミュートされたトラックでは置けても鳴らない
    空きが無ければ新しく作る
    """
    pool: list[tuple[Track, list[Clip]]] = [
        (track, list(track.clips))
        for track in project.timeline.audio_tracks()
        if not track.locked and not track.muted
    ]
    count = len(list(project.timeline.audio_tracks()))
    chosen: dict[int, Track] = {}
    for _item, clip in sorted(sounds, key=lambda pair: (pair[0].layer, pair[1].timeline_start)):
        start, end = clip.timeline_start, clip.timeline_end
        slot = next(
            (
                (track, used)
                for track, used in pool
                if not any(other.overlaps(start, end) for other in used)
            ),
            None,
        )
        if slot is None:
            count += 1
            slot = (Track(kind=TrackKind.AUDIO, name=f"A{count}"), [])
            commands.append(AddTrack(slot[0]))
            pool.append(slot)
        slot[1].append(clip)
        chosen[id(clip)] = slot[0]
    return chosen


def _scene_for(
    item: MappedObject,
    project: Project,
    commands: list[Command],
    media: Mapping[str, MediaItem],
    log: CompatibilityReport,
) -> SceneId:
    """まとめて 1 枚にする中身をシーンへ置き、そのシーンを返す

    中身の位置はまとめた入れ物の頭からの時刻で持っているので、そのまま置く
    （``at_frame`` を中身の一番早い位置にして、ずらさない） 頭へ詰めると、
    遅れて出てくる中身が入れ物の頭から出てしまう

    ``log`` は呼んだ側のレポートをそのまま渡す 渡さないと、まとめた中身の
    未対応だけが共通のレポートへ紛れ、呼んだ側の数に出ない
    """
    scene = new_scene(project, item.label or "まとめた絵")
    commands.append(AddScene(scene))
    # 中身を置くコマンドはシーンのタイムラインを相手に作る 置き先のトラックを
    # 探すのにメインのトラックを見ると、シーンに無いトラックへ置こうとして落ちる
    inside = replace(project, timeline=scene.timeline)
    earliest = min((child.clip.timeline_start for child in item.children), default=0)
    commands.extend(
        InScene(scene.id, command)
        for command in place(
            list(item.children), inside, at_frame=earliest, media=media, report=log
        )
    )
    return scene.id


def restyle(objects: list[MappedObject], clip: Clip, *, keep_wrap: bool = False) -> list[Command]:
    """テンプレートの見た目を、今あるクリップへ着せる

    2 通りある

    * **中身のあるテンプレート** — テキストオブジェクトを持つ最初の 1 つを使い、
      文字と時間は今のまま、見た目だけを入れ替える 字幕テンプレートはこれ
      テンプレートが持たない項目はテキストの既定値で埋める（#283） 今の値の上に
      重ねるだけだと、テンプレートの形式に無い項目（フォントのスタイル・折り返しの幅
      など）が前の値のまま残り、テンプレートと違う見た目になる ファミリを替えても
      前のファミリのスタイル名が残り、標準の組み方ではそのスタイルで描かれた
      ``keep_wrap`` を立てると折り返しの幅だけは今の値を残す（本人の設定
      ``Preferences.restyle_keep_wrap`` 字幕の枠の幅を決めてから着せ替える人向け）
    * **エフェクトだけのテンプレート** — YMM4 の「アニメーション効果」のように
      中身を持たないもの 今のクリップに**エフェクトを足す**だけで、
      中身には触らない だからテキスト以外のクリップにも着せられる
    """
    if not objects:
        return []

    effects_only = not any(item.has_picture for item in objects)
    if effects_only:
        # 着せる先のクリップは自分の描画の欄を持っている テンプレートの欄（配置・反転）は
        # ふつうのエフェクトとして足す 印のまま足すと、外せない配置が着せるたびに増える
        added = [
            replace(fitted_effect(effect, item.clip.duration, clip.duration - 1), fixed=False)
            for item in objects
            for effect in item.clip.effects
        ]
        if not added:
            return []
        return [AddEffect(clip.id, effect) for effect in added]

    # 文字はまとめた中身（合成するグループ）の中にあることがある 上だけを見ると、
    # 吹き出しの字幕テンプレートが「文字の無いテンプレート」として断られる
    template = next(
        (
            inner
            for item in objects
            for inner in item.walk()
            if inner.clip.source and inner.clip.source.kind == "text"
        ),
        None,
    )
    if template is None or template.clip.source is None:
        return []
    if clip.source is None or clip.source.kind != "text":
        return []

    span = template.clip.duration
    given = template.clip.source.params
    params = {
        **TEXT.default_params(),
        **{
            name: fitted_value(value, span, clip.duration - 1)
            for name, value in given.items()
            if name not in _KEPT_ON_RESTYLE
        },
    }
    staying = set(_KEPT_ON_RESTYLE)
    if keep_wrap:
        staying.add("wrap_width")
    # 項目があっても空なら時間を出さないテンプレート（TEXT.create で作る物は空の書式を持つ）
    if not str(given.get("timer_format", "") or ""):
        staying |= _TIMER_ON_RESTYLE
    current = clip.source.params
    params.update({name: current[name] for name in staying if name in current})

    commands: list[Command] = [SetSource(clip.id, GeneratedSource(kind="text", params=params))]
    # 固定の項目（クリップが最初から持つ欄）は外せないので残す 外そうとすると
    # 命令が断られ、着せる操作ごと取り消しになる
    commands.extend(RemoveEffect(clip.id, effect.id) for effect in clip.effects if not effect.fixed)
    carried = {effect.kind for effect in template.clip.effects if effect.fixed}
    for kept in clip.effects:
        if kept.fixed and kept.kind in PICTURE_FIXED and kept.kind not in carried:
            # テンプレートが欄を持たないのは既定のまま（読み込みは既定の配置や反転を写さない）
            # 戻さないと、前に着せたテンプレートの位置や反転が残り、着せ直しても見た目が揃わない
            commands.extend(
                SetParam(ParamPath.of_effect(clip.id, kept.id, name), value)
                for name, value in fixed_effect(kept.kind).params.items()
            )
    for effect in template.clip.effects:
        fitted = fitted_effect(effect, span, clip.duration - 1)
        own = next((e for e in clip.effects if e.fixed and e.kind == effect.kind), None)
        if effect.fixed and own is not None:
            # テンプレートの描画の欄（字幕の位置など）は、着せる先の同じ欄へ値を写す
            # 足すと、クリップの欄とテンプレートの欄の 2 つの配置が重なって掛かる
            commands.extend(
                SetParam(ParamPath.of_effect(clip.id, own.id, name), value)
                for name, value in fitted.params.items()
            )
            continue
        commands.append(AddEffect(clip.id, replace(fitted, fixed=False)))
    return commands


def _tracks_for(project: Project, layers: set[int], commands: list[Command]) -> dict[int, Track]:
    existing = list(project.timeline.video_tracks())
    tracks: dict[int, Track] = {}
    for layer in range(1, max(layers, default=0) + 1):
        if layer - 1 < len(existing):
            tracks[layer] = existing[layer - 1]
            continue
        track = Track(kind=TrackKind.VIDEO, name=f"V{layer}")
        commands.append(AddTrack(track))
        tracks[layer] = track
    return tracks


#: 読み込みに失敗したときに投げられる例外 呼び出し側はこれだけ捕まえればよい
TemplateError = (ExoParseError, Ymm4ParseError)
