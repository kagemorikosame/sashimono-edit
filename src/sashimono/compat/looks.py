"""保存した見た目と棚の配布物を、クリップへ当てる・タイムラインへ置く手順（#328）

当てる・置く物は 4 つ

* プリセット（設定パネルの〔プリセット…〕） いまあるクリップに見た目を当てる
* 自作のエイリアス（タイムラインの〔追加〕→〔エイリアス〕） クリップを置き直す
* テンプレートの棚（AviUtl のエイリアス・YMM4 のアイテムテンプレート） 置くか、テキストに着せる
* AviUtl のオブジェクトのファイル（``.exo`` ``.exa`` 互換の〔オブジェクトを読み込む…〕）

画面の操作と AI の道具の両方がここを通る 画面の側（Qt）に書くと、Qt を知らない作りの
AI の道具からは使えず、同じ決まりを 2 か所に書くと、片方だけ直したときに頼み方で当たり方や
置き方が変わる 決まりそのもの（どの値を当てるか・どこへ置くか）は今までどおり
:mod:`sashimono.core.commands.preset` と :mod:`sashimono.compat.catalog` にあり、ここは
エフェクトの定義や素材の開き方のような、コア層が自分では読めない知識を渡す所
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sashimono.compat.aviutl.catalog import KIND_LABELS
from sashimono.compat.aviutl.custom_object import custom_object_script
from sashimono.compat.aviutl.exo import ExoFile
from sashimono.compat.aviutl.mapping import map_exo, media_paths
from sashimono.compat.catalog import MediaPlan, Probe, gather_media, place
from sashimono.compat.mapped import MappedObject
from sashimono.core.commands import AddClip, AddMedia, Command
from sashimono.core.commands.insert import DEFAULT_GENERATED_FRAMES
from sashimono.core.commands.preset import (
    PICTURE_SIDE,
    SOUND_SIDE,
    PresetOptions,
    preset_commands,
)
from sashimono.core.io.presets import Preset
from sashimono.core.model import (
    Clip,
    ClipId,
    MediaId,
    MediaItem,
    ParamValue,
    Project,
    Track,
    TrackId,
    TrackKind,
    draws_picture,
    plays_sound,
)
from sashimono.effects.definition import EffectDefinition, registry
from sashimono.effects.sources import source_registry

__all__ = [
    "ExoMedia",
    "PresetMaker",
    "TemplatePlacement",
    "apply_preset_commands",
    "capture_preset",
    "clip_roles",
    "effect_side",
    "effects_for",
    "effects_for_clip",
    "exo_commands",
    "exo_media",
    "picture_and_sound",
    "preset_commands_for",
    "restyle_refusal",
    "template_placement",
]

#: カスタムオブジェクト（中身を作るスクリプト）の分類 エフェクトの一覧には出さない
#: クリップに掛けると、今の絵を捨てて別の物を描くので、掛けたつもりの絵が消える
_CUSTOM_OBJECT = KIND_LABELS["obj"]


# --- エフェクトが効く先 ---


def effects_for(kind: TrackKind) -> tuple[EffectDefinition, ...]:
    """そのトラックの種類のクリップへ掛けられるエフェクト

    音のエフェクトは音声トラックのクリップだけ、映像のエフェクトは映像トラックのクリップだけ
    逆に掛けると、積めても何も起きない
    """
    audio = kind is TrackKind.AUDIO
    return _effects(picture=not audio, sound=audio)


def _effects(*, picture: bool, sound: bool) -> tuple[EffectDefinition, ...]:
    return tuple(
        definition
        for definition in registry.all()
        if (sound if definition.audio_process is not None else picture)
        and definition.category != _CUSTOM_OBJECT
    )


def clip_roles(project: Project, track: Track, clip: Clip) -> tuple[bool, bool]:
    """クリップへ掛けて効くのは（絵のエフェクト, 音のエフェクト）か

    映像・音声のトラックは種類で決まる（今までどおり） レイヤー（混合）は 1 本のクリップが
    絵も音も持てるので、描く・鳴らすかで決める 種類だけで見ると、レイヤーの音付き動画や
    BGM に音のエフェクトを掛けられず、BGM には効かない絵のエフェクトが並ぶ
    """
    if track.kind is TrackKind.MIXED:
        return project.draws_picture(track, clip), project.plays_sound(track, clip)
    audio = track.kind is TrackKind.AUDIO
    return not audio, audio


def effects_for_clip(project: Project, track: Track, clip: Clip) -> tuple[EffectDefinition, ...]:
    """``clip`` へ掛けられるエフェクト（:func:`clip_roles`）"""
    picture, sound = clip_roles(project, track, clip)
    return _effects(picture=picture, sound=sound)


# --- プリセット ---


def picture_and_sound(project: Project | None, track: Track, clip: Clip) -> tuple[bool, bool]:
    """絵を描くクリップか・音を鳴らすクリップか（設定パネルが描画と音声の組を出すか）

    混合トラックはクリップが絵と音の両方を持てるので、トラックの種類ではなく
    draws_picture と plays_sound で決める
    """
    media = (
        project.find_media(clip.media_id)
        if project is not None and clip.media_id is not None
        else None
    )
    picture = draws_picture(track, clip, media)
    sound = clip.media_id is not None and clip.source is None and plays_sound(track, clip, media)
    return picture, sound


def capture_preset(name: str, project: Project | None, track: Track, clip: Clip) -> Preset:
    """画面から保存するときのプリセット

    不透明度などを持つかは描くかで決め、どちらの側のエフェクトを持つかは〔＋ エフェクト〕と
    同じ :func:`clip_roles` で決める 設定パネルが音声の欄を出すか（素材の音だけ）で決めると、
    混合のトラックに置いたシーンのように欄は無いのに音のエフェクトを持つクリップの音の側が
    「持っていない」扱いになり、当てた先の音のエフェクトと重なる
    """
    picture, _ = picture_and_sound(project, track, clip)
    _, sound_effects = _effect_roles(project, track, clip)
    return Preset.capture(name, clip, picture=picture, sound=sound_effects)


def apply_preset_commands(
    project: Project, track: Track, clip: Clip, preset: Preset, *, options: PresetOptions
) -> list[Command]:
    """``preset`` を ``clip`` へ当てるコマンド（設定パネルの〔プリセット…〕と AI の道具）

    コア層が自分では読めない知識（エフェクトの定義・中身の既定の値・カスタムオブジェクトの
    見分け）をここで渡す 試験も同じ入口を通し、画面と同じ当て方を確かめる
    """
    picture, sound = picture_and_sound(project, track, clip)
    effect_picture, effect_sound = _effect_roles(project, track, clip)
    allowed = frozenset(d.kind for d in effects_for_clip(project, track, clip))

    def accepts(kind: str) -> bool:
        # 定義の無い種類（別の機械で作った・スクリプトを外した AviUtl のエフェクト）は、
        # 絵の側が効く先なら足す 設定パネルは未知のエフェクトを残して見せ、側も絵として
        # 扱う（:func:`effect_side`） 一覧に無いからと捨てると、プリセットの見た目が黙って欠ける
        # 音だけの先には今までどおり足さない
        if kind in allowed:
            return True
        return effect_picture and registry.get(kind) is None

    return preset_commands(
        preset,
        clip,
        picture=picture,
        sound=sound,
        # どちらの側のエフェクトを入れ替えるかは、欄を出すかではなく掛けられるかで決める
        # （:func:`capture_preset` と同じ理由）
        effect_sides=(effect_picture, effect_sound),
        options=options,
        # 音だけのクリップに映像のエフェクトを足さない（〔＋ エフェクト〕と同じ決まり）
        accepts=accepts,
        # カスタムオブジェクトの本体（最初のエフェクトのスクリプト）は中身として扱う
        # 見た目の入れ替えで消すと、何も描かないクリップが残る
        body_of=custom_object_script,
        side_of=effect_side,
        defaults_of=_source_defaults,
    )


#: 1 本のクリップへ当てるコマンドを作る手 :func:`apply_preset_commands` と同じ形
PresetMaker = Callable[..., list[Command]]


def preset_commands_for(
    project: Project,
    clip_ids: Sequence[ClipId],
    preset: Preset,
    *,
    options: PresetOptions,
    make: PresetMaker = apply_preset_commands,
) -> tuple[list[Command], tuple[ClipId, ...]]:
    """何本かのクリップへ当てるコマンドと、変わる所のあったクリップ 1 回の取り消しで全部戻す

    主の 1 本だけに当てると、何本も選んで当てたつもりが残りは前の見た目のまま残る
    見つからない ID と変わる所の無いクリップは飛ばす ``make`` は画面の試験が当てる入口を
    差し替えて、一覧から当てても同じ入口を通るかを見るのに使う
    """
    commands: list[Command] = []
    touched: list[ClipId] = []
    for clip_id in dict.fromkeys(clip_ids):
        located = project.timeline.locate_clip(clip_id)
        if located is None:
            continue
        made = make(project, *located, preset, options=options)
        if made:
            commands.extend(made)
            touched.append(clip_id)
    return commands, tuple(touched)


def _effect_roles(project: Project | None, track: Track, clip: Clip) -> tuple[bool, bool]:
    """絵のエフェクトと音のエフェクトが効くか（〔＋ エフェクト〕と同じ :func:`clip_roles`）"""
    if project is None:
        return picture_and_sound(project, track, clip)
    return clip_roles(project, track, clip)


def effect_side(kind: str) -> str:
    """エフェクトが絵と音のどちらに掛かるか 〔＋ エフェクト〕の分け方（音を処理するか）と同じ

    手元に無い種類（読めないスクリプトなど）は絵とする 音のエフェクトはどれも本体の物で、
    手元に無いことが無い
    """
    definition = registry.get(kind)
    if definition is not None and definition.audio_process is not None:
        return SOUND_SIDE
    return PICTURE_SIDE


def _source_defaults(kind: str) -> dict[str, ParamValue] | None:
    """中身の種類の既定の値 プリセットの中身を当てるときの土台 知らない種類は ``None``"""
    definition = source_registry.get(kind)
    return definition.default_params() if definition is not None else None


# --- テンプレートの棚 ---


@dataclass(frozen=True, slots=True)
class TemplatePlacement:
    """棚の配布物を置く段取り 素材の登録と配置を 1 回の取り消しにまとめて出す"""

    #: 素材の登録（先）と配置 分けて出すと、戻したときに使われていない素材だけが一覧に残る
    commands: tuple[Command, ...]
    plan: MediaPlan
    #: 一番上に置いたクリップ トラックやシーンを足すコマンドは数えない（画像 1 つでも
    #: 「2 個を置いた」と出さないため）
    placed: tuple[ClipId, ...]


def template_placement(
    objects: list[MappedObject],
    project: Project,
    frame: int,
    probe: Probe,
    *,
    near: Path | None = None,
    track_id: TrackId | None = None,
) -> TemplatePlacement:
    """棚の配布物を ``frame`` へ置く段取り 置ける物が無ければ ``commands`` は空

    画像・音声のアイテムは素材として登録してからクリップに結ぶ 結ばないと、置いたクリップは
    描かれず鳴らない（素材の無いクリップになる） ``near`` は配布物の置き場 書かれた素材の
    場所に無ければ、同じ名前をそこで探す

    ``track_id`` は右クリックしたトラック 置く物が 1 つでそこが空いていればそこへ、そうで
    なければ棚の窓から置くときと同じく元のレイヤーの並びで置く 何個もの物を 1 本のトラックへ
    置くと、重なった所で断られる
    """
    plan = gather_media(objects, project, probe, near=near)
    pictures = [item for item in objects if item.has_picture]
    target = None
    if track_id is not None and len(pictures) == 1 and not pictures[0].children:
        track = project.timeline.find_track(track_id)
        clip = pictures[0].clip
        end = frame + (clip.duration if pictures[0].has_span else DEFAULT_GENERATED_FRAMES)
        free = track is not None and not any(c.overlaps(frame, end) for c in track.clips)
        # レイヤー（混合トラック）も受ける 映像トラックに限ると、混合の方式では
        # 右クリックしたレイヤーを無視して元のレイヤー番号の所へ入る
        placeable = (TrackKind.VIDEO, TrackKind.MIXED)
        if free and track is not None and track.kind in placeable and not track.locked:
            target = track_id
    commands = place(objects, project, at_frame=frame, track_id=target, media=plan.media)
    if not commands:
        return TemplatePlacement((), plan, ())
    placed = tuple(c.clip.id for c in commands if isinstance(c, AddClip))
    return TemplatePlacement((*plan.commands, *commands), plan, placed)


def restyle_refusal(objects: list[MappedObject], clip: Clip) -> str:
    """棚の 1 本を ``clip`` に着せても何も変わらない（着せ替えが空を返した）理由

    理由を 1 つの文で済ませると、効果だけのテンプレートで効果を 1 つも読めなかった物
    （AviUtl1 の ``.anm`` のエイリアスでスクリプトが置き場に無い など）まで「テキストの
    クリップにしか適用できません」と出て、テキストを選び直しても直らない
    """
    if not any(item.has_picture for item in objects):
        return (
            "効果だけのテンプレートで、読めた効果がありません"
            "（AviUtl のスクリプトが置き場に無いなど 互換性レポートに出ます）"
        )
    if clip.source is None or clip.source.kind != "text":
        return "テキストのクリップにしか適用できません"
    return "文字を持たないテンプレートなので着せられません（置くことはできます）"


# --- AviUtl のオブジェクトのファイル ---


@dataclass(frozen=True, slots=True)
class ExoMedia:
    """``.exo`` が参照している素材を読んだ結果 登録はまだしていない

    登録をクリップの配置と同じ 1 回の取り消しで行うため、ここではコマンドを作るだけにする
    先に登録すると、配置を断られたときに使われない素材と、その解析・控えの重い処理だけが残る
    （テンプレートの棚の :func:`~sashimono.compat.catalog.gather_media` と同じ作り）
    """

    #: 書かれていたパス → 結ぶ素材の id（:func:`map_exo` がそのまま受ける形）
    ids: dict[str, MediaId]
    #: 見つからないか開けなかったパス
    missing: tuple[str, ...]
    #: 素材一覧へ入れるコマンド
    commands: tuple[Command, ...]
    #: 入ったあとに解析と控えを頼む素材
    items: tuple[MediaItem, ...]


def exo_media(exo: ExoFile, source: Path, probe: Probe) -> ExoMedia:
    """``.exo`` が参照している素材を読む 一覧へ入れるのは呼んだ側

    相対パスは ``.exo`` のある場所からも探す AviUtl のファイルは素材と一緒に配られることがある
    """
    ids: dict[str, MediaId] = {}
    missing: list[str] = []
    items: list[MediaItem] = []
    for raw in media_paths(exo):
        candidates = [Path(raw), source.parent / Path(raw).name]
        path = next((c for c in candidates if c.exists()), None)
        media = probe(path) if path is not None else None
        if media is None:
            missing.append(raw)
            continue
        items.append(media)
        ids[raw] = media.id
    return ExoMedia(
        ids=ids,
        missing=tuple(missing),
        commands=tuple(AddMedia(media) for media in items),
        items=tuple(items),
    )


def exo_commands(
    exo: ExoFile, source: Path, project: Project, probe: Probe
) -> tuple[ExoMedia, list[Command]]:
    """``.exo`` を置くコマンド（素材の登録を頭に足した物）と、読んだ素材 置ける物が無ければ空"""
    found = exo_media(exo, source, probe)
    commands = map_exo(exo, project, media=found.ids, items=found.items)
    if not commands:
        return found, []
    return found, [*found.commands, *commands]
