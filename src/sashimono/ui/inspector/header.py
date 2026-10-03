"""オブジェクト設定の上に出す「いま何の設定を見ているか」

名前だけでは、映像と音声がリンクした素材でどちらの設定を見ているのかが分からず、
音量を変えたつもりが映像の側を開いていた、が起きた（Issue #27）
種類（映像・音声・画像・テキスト・フィルタ・シーンなど）を色の帯と言葉で出し、
クリップの名前と置いてあるトラックを添える 帯の色はタイムラインのクリップの色と
同じにして、タイムラインのどのクリップを開いているのかを色でも結び付ける
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from sashimono.compat.aviutl.custom_object import (
    CUSTOM_OBJECT_LABEL,
    custom_object_script,
    script_label,
)
from sashimono.core.model import (
    Clip,
    ClipId,
    GeneratedSource,
    MediaItem,
    Project,
    Track,
    TrackKind,
)
from sashimono.effects import registry
from sashimono.effects.sources import source_registry
from sashimono.ui.theme import Colors, themed_style

__all__ = ["ClipHeader", "ClipIdentity", "identify_clip"]

#: 生成オブジェクトの中身を名前として添えるときの長さ 長い文を全部出すと 1 行に収まらない
_TEXT_PREVIEW = 16


@dataclass(frozen=True, slots=True)
class ClipIdentity:
    """設定パネルが見ているクリップが何か"""

    #: 種類の言葉（映像・音声・画像・テキスト・フィルタ・シーンなど）
    kind: str
    #: クリップの名前 リンクした素材では「の絵」「の音」まで付ける
    name: str
    #: 置いてあるトラックの名前
    track: str
    #: 帯の色 タイムラインのクリップの枠と同じ
    color: QColor

    @property
    def title(self) -> str:
        return f"{self.kind}（{self.name}）" if self.name else self.kind

    def summary(self) -> str:
        """1 行にまとめた形 読み上げと試験が使う"""
        return f"{self.title} / {self.track}"


def identify_clip(project: Project, clip_id: ClipId) -> ClipIdentity | None:
    """クリップの種類・名前・トラックを調べる 見つからなければ ``None``"""
    located = project.timeline.locate_clip(clip_id)
    if located is None:
        return None
    track, clip = located
    kind, name = _kind_and_name(project, track, clip)
    return ClipIdentity(kind, name, _track_name(project, track), _band_color(project, track, clip))


def _kind_and_name(project: Project, track: Track, clip: Clip) -> tuple[str, str]:
    if clip.scene_id is not None:
        scene = project.find_scene(clip.scene_id)
        return "シーン", scene.name if scene is not None else ""
    if clip.source is not None:
        return _generated(clip, clip.source)
    media = project.find_media(clip.media_id) if clip.media_id is not None else None
    if track.kind is TrackKind.MIXED:
        return _on_layer(project, track, clip, media)
    if media is None:
        return ("映像" if track.kind is TrackKind.VIDEO else "音声"), ""
    if track.kind is TrackKind.AUDIO:
        # 映像の方と結ばれていれば、同じ素材の音だと言う 素材の名前だけでは、
        # 映像のクリップの設定と見分けが付かない
        linked = _linked_to(project, clip, TrackKind.VIDEO)
        return "音声", f"{media.name} の音" if linked else media.name
    if media.is_still:
        return "画像", media.name
    linked = _linked_to(project, clip, TrackKind.AUDIO)
    return "映像", f"{media.name} の絵" if linked else media.name


def _on_layer(
    project: Project, track: Track, clip: Clip, media: MediaItem | None
) -> tuple[str, str]:
    """レイヤー（混合）に置いた素材のクリップ 描く・鳴らすかで言葉を選ぶ

    トラックの種類では決まらない 音だけの素材を「映像」と出すと、音量を探して描画の欄を
    開くことになる 音付きの動画は 1 本で絵も音も持つので「音付き」と添える
    """
    name = media.name if media is not None else ""
    picture = project.draws_picture(track, clip)
    sound = project.plays_sound(track, clip)
    if picture and media is not None and media.is_still:
        return "画像", name
    if picture:
        return ("映像（音付き）" if sound else "映像"), name
    return "音声", name


def _generated(clip: Clip, source: GeneratedSource) -> tuple[str, str]:
    script = custom_object_script(clip)
    if script is not None:
        # 土台は空のテキスト 「テキスト」と出すと、下に並ぶ書体の欄を触ればよいと読める
        return CUSTOM_OBJECT_LABEL, script_label(script.kind)
    definition = source_registry.get(source.kind)
    kind = definition.label if definition is not None else source.kind
    if clip.is_filter:
        # フィルタは中身がエフェクトだけ 何を掛けているかが名前の代わりになる
        labels = [_effect_label(effect.kind) for effect in clip.effects]
        return kind, "、".join(labels) if labels else "エフェクトなし"
    text = source.params.get("text")
    if isinstance(text, str) and text.strip():
        return kind, text.strip().splitlines()[0][:_TEXT_PREVIEW]
    return kind, ""


def _effect_label(kind: str) -> str:
    definition = registry.get(kind)
    return definition.label if definition is not None else kind


def _linked_to(project: Project, clip: Clip, kind: TrackKind) -> bool:
    """同じリンクで結ばれたクリップが、``kind`` のトラックにあるか"""
    if clip.link_group is None:
        return False
    return any(
        other.link_group == clip.link_group and other.id != clip.id
        for track in project.timeline.tracks
        if track.kind is kind
        for other in track.clips
    )


def _track_name(project: Project, track: Track) -> str:
    """トラックの名前 名前の無いトラックは種類と上からの番号で呼ぶ"""
    if track.name:
        return track.name
    same = [t for t in project.timeline.tracks if t.kind is track.kind]
    number = next((i for i, t in enumerate(same, start=1) if t is track), 0)
    if track.kind is TrackKind.MIXED:
        # タイムラインの並び（レイヤー 1 が上）と同じ番号で呼ぶ
        return f"レイヤー {number}"
    return f"{'映像' if track.kind is TrackKind.VIDEO else '音声'}トラック {number}"


def _band_color(project: Project, track: Track, clip: Clip) -> QColor:
    """タイムラインのクリップの枠と同じ色 レイヤーは描く・鳴らすかで選ぶ（painter と同じ）"""
    if clip.is_filter or clip.is_group:
        return Colors.FILTER_CLIP_BORDER
    if track.kind is TrackKind.MIXED:
        sound_only = not project.draws_picture(track, clip) and project.plays_sound(track, clip)
        return Colors.AUDIO_CLIP_BORDER if sound_only else Colors.VIDEO_CLIP_BORDER
    if track.kind is TrackKind.AUDIO:
        return Colors.AUDIO_CLIP_BORDER
    return Colors.VIDEO_CLIP_BORDER


class ClipHeader(QWidget):
    """色の帯と、種類・名前・トラック"""

    #: 帯の太さ（ピクセル） 細すぎると色が見分けられない
    BAND_WIDTH = 5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._identity: ClipIdentity | None = None
        self._band = QFrame(self)
        self._band.setFixedWidth(self.BAND_WIDTH)
        self._title = QLabel(self)
        themed_style(self._title, lambda: f"color: {Colors.CLIP_LABEL.name()}; font-weight: bold;")
        self._detail = QLabel(self)
        themed_style(self._detail, lambda: f"color: {Colors.TEXT_MUTED.name()};")
        for label in (self._title, self._detail):
            # 長い名前で横に伸びてパネルの幅を押し広げない 収まらない分は切れて見える
            label.setMinimumWidth(1)
            # 名前は本人が付けた文字 ``<b>`` などを含むと、既定の AutoText では装飾として
            # 読まれ、名前がそのまま出ない
            label.setTextFormat(Qt.TextFormat.PlainText)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        text.addWidget(self._title)
        text.addWidget(self._detail)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        layout.addWidget(self._band)
        layout.addLayout(text, 1)
        self.show_identity(None)

    @property
    def identity(self) -> ClipIdentity | None:
        return self._identity

    def title_text(self) -> str:
        return self._title.text()

    def detail_text(self) -> str:
        return self._detail.text()

    def show_identity(self, identity: ClipIdentity | None, *, others: int = 0) -> None:
        """見ているクリップを出す ``None`` なら選んでいないことを出す

        ``others`` は一緒に選んでいるほかのクリップの本数 同じ設定の値はそちらにも当たるので、
        知らずに何本もまとめて変えないよう添える
        """
        self._identity = identity
        if identity is None:
            themed_style(self._band, lambda: "background: transparent;")
            self._title.setText("クリップを選んでください")
            themed_style(self._title, lambda: f"color: {Colors.TEXT_MUTED.name()};")
            self._detail.hide()
            self.setAccessibleName("クリップを選んでいない")
            # 前のクリップの名前が補足に残ると、選んでいないのに何かを開いているように見える
            self.setToolTip("")
            return
        # 帯の色は Colors の QColor そのもの（写しではない） テーマを切り替えると中身が
        # 書き換わるので、作り直すたびに読めば新しいテーマの色になる
        band = identity.color
        themed_style(self._band, lambda: f"background-color: {band.name()};")
        themed_style(self._title, lambda: f"color: {Colors.CLIP_LABEL.name()}; font-weight: bold;")
        self._title.setText(identity.title)
        detail = f"トラック {identity.track}"
        if others:
            # 「当てる」と言い切らない 当たるのは同じ設定を持つクリップの値だけで、
            # エフェクトの追加や並べ替えは主のクリップにしか入らない（``_for_clip``）
            detail += f"  ほか {others} 本も選択中（同じ設定の値だけ一緒に変わる）"
        self._detail.setText(detail)
        self._detail.show()
        self.setAccessibleName(identity.summary())
        self.setToolTip(identity.summary())
