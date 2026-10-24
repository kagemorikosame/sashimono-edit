"""プロジェクト全体 モデルツリーの根"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction

from sashimono.core.model.ids import MediaId, SceneId, new_scene_id
from sashimono.core.model.media import MediaItem
from sashimono.core.model.style import StyleGuide
from sashimono.core.model.timeline import (
    Clip,
    Timeline,
    Track,
    TrackKind,
    draws_picture,
    plays_sound,
)
from sashimono.core.timebase import FrameRate

__all__ = ["Blending", "LayerMode", "Project", "ProjectSettings", "Scene"]


@dataclass(frozen=True, slots=True)
class Scene:
    """メインとは別のタイムライン（AviUtl のシーン）

    ほかのタイムラインへ 1 本のクリップとして置ける（:attr:`Clip.scene_id`）
    オープニングのように何度も使う部分を 1 か所で直せる
    """

    name: str
    timeline: Timeline
    id: SceneId = field(default_factory=new_scene_id)


class Blending:
    """半透明の絵を重ねるときに、どの値で混ぜるか（Issue #65）

    値はプロジェクトファイルにそのまま出る 名前を変えると、保存した作品の見た目が変わる
    """

    #: sRGB で符号化した値のまま混ぜる AviUtl2 と YMM4 はこちら（黒の上に 50% の白で 128）
    #: 配布されている素材は、この混ざり方を前提に色と不透明度を決めてある
    SRGB = "srgb"
    #: 光の量（リニア）に直して混ぜる 半透明の所が明るく出る（黒の上に 50% の白で 188）
    #: この項目ができる前に保存したプロジェクトは、すべてこちらで描いていた
    LINEAR = "linear"
    ALL = (SRGB, LINEAR)


class LayerMode:
    """素材を置くトラックの方式（Issue #27）

    値はプロジェクトファイルにそのまま出る 名前を変えると、保存した作品の方式が変わる
    ここで決まるのは**これから置く所**だけ 置いてあるトラックはどちらの方式でも
    そのまま描き・鳴らす（変換は別の命令）
    """

    #: 1 本のレイヤー（:attr:`~sashimono.core.model.TrackKind.MIXED`）に何でも置く
    #: YMM4・AviUtl と同じ 音付きの動画は絵と音を 1 本のクリップで持つ
    MIXED = "mixed"
    #: 映像トラックと音声トラックに分けて置く 絵と音を 2 本のクリップにしてリンクで結ぶ
    #: この項目ができる前のプロジェクトは、すべてこちら
    SEPARATED = "separated"
    ALL = (MIXED, SEPARATED)


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """出力の基準となる設定

    タイムライン上のすべての時刻はここの :attr:`frame_rate` を基準に解釈される
    素材のフレームレートがこれと違っても構わない（読み込み時に秒へ正規化される）
    """

    width: int = 1920
    height: int = 1080
    frame_rate: FrameRate = field(default_factory=lambda: FrameRate(30))
    sample_rate: int = 48000
    channels: int = 2
    #: 当面は sRGB / Rec.709 のみ HDR は将来対応
    color_space: str = "rec709"
    #: 半透明の重ね合わせ（:class:`Blending`） 新しく作るプロジェクトは AviUtl と YMM4 に
    #: 合わせて sRGB で混ぜる この項目の無い古いファイルは、読むときにリニアとして開く
    #: （:mod:`sashimono.core.io.serialize`） 既定で開くと、保存したときと見た目が変わる
    blending: str = Blending.SRGB
    #: 素材を置くトラックの方式（:class:`LayerMode`） モデルの既定は分ける方式
    #: 古いファイルと、既定のプロジェクトで組み立てる今の試験の動きを変えないため
    #: 新規作成で混合にするかは本人の好みで決める（画面の側）
    layer_mode: str = LayerMode.SEPARATED

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"解像度が不正: {self.width}x{self.height}")
        if self.sample_rate <= 0:
            raise ValueError(f"サンプリングレートが不正: {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"チャンネル数が不正: {self.channels}")
        if self.blending not in Blending.ALL:
            # 知らない値のまま描くと、どちらの混ぜ方になるかがレンダラの作り次第になる
            raise ValueError(f"重ね合わせの方法が不正: {self.blending!r}")
        if self.layer_mode not in LayerMode.ALL:
            # 知らない値のまま持つと、置き方がどちらになるかが置く側の作り次第になる
            raise ValueError(f"トラックの方式が不正: {self.layer_mode!r}")

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class Project:
    """編集中のプロジェクト

    このオブジェクトを差し替えることが「編集した」という意味になる UI も AI も
    直接書き換えず、:mod:`sashimono.core.commands` を通して新しい :class:`Project` を
    受け取る

    既定値付きで作りたい場合は :meth:`create` を使う タイムラインのフレームレートは
    常に :attr:`settings` と一致していなければならないので、素の構築では両方を渡す
    """

    settings: ProjectSettings
    timeline: Timeline
    #: メディアプール 表示順を保つため辞書ではなくタプル
    media: tuple[MediaItem, ...] = ()
    name: str = "無題"
    #: メイン（:attr:`timeline`）とは別のシーン 表示順を保つためタプル
    scenes: tuple[Scene, ...] = ()
    #: 編集の方針（#343） 作品の見た目そのものの決まりなので、本人の好み（``Preferences``）
    #: ではなく作品に付ける 同じ作品を別の機械で開いても、AI が同じ方針で編集する
    style_guide: StyleGuide = field(default_factory=StyleGuide)
    #: 書き出した動画に添える表記（「VOICEVOX:ずんだもん」など #335） 作った声の規約が
    #: 求める物で、書き出しのときに知らせる 作品に付けるのは、別の機械で開いて書き出しても
    #: 知らせ漏らさないため 足すのはナレーションを作る道具だけで、取り消し 1 回で戻る
    credits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.timeline.rate != self.settings.frame_rate:
            raise ValueError(
                "タイムラインのフレームレートがプロジェクト設定と一致しない: "
                f"{self.timeline.rate} と {self.settings.frame_rate}"
            )
        ids = [m.id for m in self.media]
        if len(set(ids)) != len(ids):
            raise ValueError("素材 ID が重複している")
        scene_ids = [s.id for s in self.scenes]
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("シーン ID が重複している")
        for scene in self.scenes:
            if scene.timeline.rate != self.settings.frame_rate:
                # シーンだけ別のフレームレートにすると、置いたときの時刻の換算が要る
                raise ValueError(f"シーン {scene.name!r} のフレームレートがプロジェクトと違う")
        known = set(scene_ids)
        for timeline in (self.timeline, *(scene.timeline for scene in self.scenes)):
            unknown = timeline.scene_references() - known
            if unknown:
                # 開けてしまうと、そのクリップは絵も音も出さずに黙って残る
                raise ValueError(f"無いシーンを指すクリップがある: {sorted(unknown)}")
        cycle = self.scene_cycle()
        if cycle is not None:
            raise ValueError(f"シーンが自分自身を入れ子にしている: {cycle}")

    @classmethod
    def create(
        cls,
        settings: ProjectSettings | None = None,
        *,
        media: tuple[MediaItem, ...] = (),
        name: str = "無題",
    ) -> Project:
        """設定に合ったフレームレートの空タイムラインを用意して生成する"""
        resolved = settings if settings is not None else ProjectSettings()
        return cls(
            settings=resolved,
            timeline=Timeline(rate=resolved.frame_rate),
            media=media,
            name=name,
        )

    @property
    def rate(self) -> FrameRate:
        return self.settings.frame_rate

    @property
    def duration(self) -> int:
        """タイムラインの長さ（フレーム）"""
        return self.timeline.duration

    @property
    def duration_seconds(self) -> Fraction:
        return self.duration * self.rate.frame_duration

    def find_media(self, media_id: MediaId) -> MediaItem | None:
        for item in self.media:
            if item.id == media_id:
                return item
        return None

    def draws_picture(self, track: Track, clip: Clip) -> bool:
        """``track`` の ``clip`` が絵を描くか（:func:`~sashimono.core.model.draws_picture`）

        素材を引くのは混合トラックだけ ほかの種類は素材を見ずに決まるので、
        毎フレーム通る描画の道で素材の一覧をなめない
        """
        if track.kind is not TrackKind.MIXED or clip.media_id is None:
            return draws_picture(track, clip, None)
        return draws_picture(track, clip, self.find_media(clip.media_id))

    def plays_sound(self, track: Track, clip: Clip) -> bool:
        """``track`` の ``clip`` の音を鳴らすか（:func:`~sashimono.core.model.plays_sound`）"""
        if track.kind is not TrackKind.MIXED or clip.media_id is None:
            return plays_sound(track, clip, None)
        return plays_sound(track, clip, self.find_media(clip.media_id))

    def require_media(self, media_id: MediaId) -> MediaItem:
        """素材を引く 無ければ :class:`KeyError`"""
        item = self.find_media(media_id)
        if item is None:
            raise KeyError(f"素材が見つからない: {media_id}")
        return item

    def with_timeline(self, timeline: Timeline) -> Project:
        return replace(self, timeline=timeline)

    def with_media(self, media: tuple[MediaItem, ...]) -> Project:
        return replace(self, media=media)

    def replace_media(self, item: MediaItem) -> Project:
        """同じ ID の素材を差し替えた新しい :class:`Project` を返す"""
        for index, existing in enumerate(self.media):
            if existing.id == item.id:
                return self.with_media((*self.media[:index], item, *self.media[index + 1 :]))
        raise KeyError(f"素材が見つからない: {item.id}")

    def renamed(self, name: str) -> Project:
        return replace(self, name=name)

    def find_scene(self, scene_id: SceneId) -> Scene | None:
        for scene in self.scenes:
            if scene.id == scene_id:
                return scene
        return None

    def require_scene(self, scene_id: SceneId) -> Scene:
        scene = self.find_scene(scene_id)
        if scene is None:
            raise KeyError(f"シーンが見つからない: {scene_id}")
        return scene

    def replace_scene(self, scene: Scene) -> Project:
        """同じ ID のシーンを差し替えた新しい :class:`Project` を返す"""
        for index, existing in enumerate(self.scenes):
            if existing.id == scene.id:
                scenes = (*self.scenes[:index], scene, *self.scenes[index + 1 :])
                return replace(self, scenes=scenes)
        raise KeyError(f"シーンが見つからない: {scene.id}")

    def with_scenes(self, scenes: tuple[Scene, ...]) -> Project:
        return replace(self, scenes=scenes)

    def scene_cycle(self) -> str | None:
        """入れ子が自分へ戻るシーンの名前 無ければ ``None``

        描くときに無限に潜り続けるので、作らせない（コマンドはこれで止まる）
        """
        edges = {scene.id: scene.timeline.scene_references() for scene in self.scenes}
        names = {scene.id: scene.name for scene in self.scenes}
        visiting: set[SceneId] = set()
        done: set[SceneId] = set()

        def visit(scene_id: SceneId) -> SceneId | None:
            if scene_id in done:
                return None
            if scene_id in visiting:
                return scene_id
            visiting.add(scene_id)
            for target in edges.get(scene_id, set()):
                found = visit(target)
                if found is not None:
                    return found
            visiting.discard(scene_id)
            done.add(scene_id)
            return None

        for scene_id in edges:
            found = visit(scene_id)
            if found is not None:
                return names.get(found, str(found))
        return None
