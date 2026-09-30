"""オブジェクト設定パネル

選んだクリップの中身とエフェクトを、パラメータ定義から自動で組み立てて見せる
エフェクトを増やしてもここに手を入れる必要は無い

自分ではプロジェクトを書き換えない 操作はすべてコマンドとして外へ出す
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.commands import (
    AddEffect,
    ClearKeyframes,
    Command,
    MoveEffect,
    ParamPath,
    ParamTarget,
    RemoveEffect,
    RemoveKeyframe,
    SetClipProperty,
    SetEffectEnabled,
    SetKeyframe,
    SetParam,
)
from sashimono.core.commands.fixed import (
    FADE_EFFECT_KIND,
    FLIP_EFFECT_KIND,
    TRANSFORM_EFFECT_KIND,
    VOLUME_EFFECT_KIND,
    fixed_effect,
    takes_picture_items,
)
from sashimono.core.io import Preset, PresetStore
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    ClipId,
    Effect,
    EffectId,
    ParamValue,
    Project,
    Track,
    draws_picture,
    plays_sound,
)
from sashimono.effects import (
    CheckSpec,
    FileSpec,
    GridSpec,
    ParameterSpec,
    TextSpec,
    TrackSpec,
    registry,
)
from sashimono.effects.blending import BLEND_MODES
from sashimono.effects.sources import source_registry
from sashimono.engine.gpu import BlendMode
from sashimono.ui.inspector.header import ClipHeader, identify_clip
from sashimono.ui.inspector.widgets import ParameterEditor, TrackEditor, create_editor
from sashimono.ui.theme import Colors
from sashimono.ui.timeline.add_menu import effects_for_clip

__all__ = ["InspectorPanel", "KeyframeControls"]

#: 合成方法の表示名
BLEND_LABELS = {
    BlendMode.NORMAL: "通常",
    BlendMode.ADD: "加算",
    BlendMode.MULTIPLY: "乗算",
    BlendMode.SCREEN: "スクリーン",
    BlendMode.SUBTRACT: "減算",
    BlendMode.OVERLAY: "オーバーレイ",
    BlendMode.LIGHTEN: "比較(明)",
    BlendMode.DARKEN: "比較(暗)",
    **{mode: label for mode, label in BLEND_MODES if mode in BlendMode.EXTENDED},
}


def _same_effect(
    primary: Clip, other: Clip, effect_id: EffectId, *, after: bool = False
) -> Effect | None:
    """主のクリップのエフェクトに当たる、相手側のエフェクト 同じ種類の同じ順番で探す

    場面切り替えは前の場面と後の場面で別の列を持つので、同じ列の中で探す
    """
    mine = primary.after_effects if after else primary.effects
    theirs = other.after_effects if after else other.effects
    found = next((e for e in mine if e.id == effect_id), None)
    if found is None:
        return None
    if found.fixed:
        # 描画・音声の欄どうしで当てる 何個目かで数えると、相手が同じ種類のふつうの
        # エフェクトを欄より前に持つとき、欄ではなくそちらの値が変わる
        return next((e for e in theirs if e.fixed and e.kind == found.kind), None)
    # 同じ種類が何個目かを数える 種類の一覧から探すと、2 個目以降でも 0 番目が出る
    index = sum(1 for e in mine[: mine.index(found)] if e.kind == found.kind)
    same = [e for e in theirs if e.kind == found.kind]
    return same[index] if index < len(same) else None


def _moved_path(path: ParamPath, primary: Clip, other: Clip) -> ParamPath | None:
    if path.target is ParamTarget.CLIP:
        return replace(path, clip_id=other.id)
    if path.target is ParamTarget.SOURCE:
        if primary.source is None or other.source is None:
            return None
        if primary.source.kind != other.source.kind:
            return None
        if path.name not in other.source.params:
            return None
        return replace(path, clip_id=other.id)
    if path.effect_id is None:
        return None
    twin = _same_effect(primary, other, path.effect_id, after=path.after)
    if twin is None or path.name not in twin.params:
        return None
    return replace(path, clip_id=other.id, effect_id=twin.id)


def _for_clip(command: Command, primary: Clip, other: Clip) -> Command | None:
    """主のクリップ向けのコマンドを、ほかのクリップ向けに作り直す 当てられなければ ``None``"""
    if isinstance(command, SetClipProperty):
        return replace(command, clip_id=other.id)
    if isinstance(command, SetParam | SetKeyframe | RemoveKeyframe | ClearKeyframes):
        path = _moved_path(command.path, primary, other)
        return None if path is None else replace(command, path=path)
    if isinstance(command, SetEffectEnabled):
        # 描画・音声の組の切り替えは選んだ全部へ 主だけ切り替わると、一緒に値を変えた
        # ほかのクリップと欄の効き方が食い違う
        twin = _same_effect(primary, other, command.effect_id, after=command.after)
        return None if twin is None else replace(command, clip_id=other.id, effect_id=twin.id)
    # エフェクトの追加や並べ替えは、主のクリップだけに当てる（増やすと元へ戻しにくい）
    return None


class InspectorPanel(QWidget):
    """選択中のクリップの設定"""

    #: 編集操作 引数はコマンドの一覧と、履歴に出す操作名
    commands_requested = Signal(list, str)
    #: ドラッグ中の途中経過 履歴に残さずプレビューだけ更新する
    preview_requested = Signal(object)
    #: グラフエディタで開くパラメータが選ばれた
    curve_selected = Signal(object)
    #: キーフレームを打った・消した・前後へ飛んだ値 グラフエディタに同じ値を出す
    #: （開いていなければ開かない 押すたびに窓が開くと、打つだけの人の邪魔になる）
    param_focused = Signal(object)
    #: 再生位置を動かしたい（◀ ▶ で前後のキーへ） 引数はタイムラインのフレーム
    seek_requested = Signal(int)
    #: 触ったエフェクト（値を変えた・組を押した） プレビューが部分フィルタの範囲の枠を
    #: どのエフェクトについて出すかを決めるのに使う
    effect_focused = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._clip_id: ClipId | None = None
        self._selection: tuple[ClipId, ...] = ()
        self._presets = PresetStore()
        self._frame = 0
        #: パラメータごとの入力欄 プロジェクトが変わったときに値を入れ直す
        self._editors: dict[tuple[str, str], ParameterEditor] = {}
        #: パラメータごとのキーフレームの ◀ ◆ ▶ 再生位置が動くたびに見た目を直す
        self._key_controls: dict[tuple[str, str], KeyframeControls] = {}
        #: 前の版のファイルで、クリップがまだ持っていない描画・音声の欄 既定の値で見せ、
        #: 触ったときに :class:`AddEffect` で足してから値を入れる（1 回の取り消しで戻る）
        #: 開いただけで足すと、見ただけのクリップまで変更が入り、保存を促される
        self._virtual: dict[EffectId, tuple[ClipId, Effect]] = {}
        #: ダブルクリックで初期値へ戻すか（設定 :attr:`Preferences.double_click_reset`）
        self._double_click_reset = True

        #: 何のクリップの設定を見ているか（種類・名前・トラック）
        self._title = ClipHeader(self)

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 8, 8)
        self._body_layout.setSpacing(10)
        self._body_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidget(self._body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._add_button = QPushButton("エフェクトを追加…", self)
        self._add_button.clicked.connect(self._show_effect_menu)
        self._add_button.setEnabled(False)

        self._preset_button = QPushButton("プリセット…", self)
        self._preset_button.clicked.connect(self._show_preset_menu)
        self._preset_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(0)
        buttons.addWidget(self._add_button, 1)
        buttons.addWidget(self._preset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._title)
        layout.addWidget(scroll, 1)
        layout.addLayout(buttons)

    # --- 外から差し替えるもの ---

    def set_project(self, project: Project) -> None:
        self._project = project
        self._rebuild()

    def set_clip(self, clip_id: ClipId | None) -> None:
        self.set_selection((clip_id,) if clip_id is not None else ())

    def set_selection(self, clip_ids: tuple[ClipId, ...]) -> None:
        """選んでいるクリップ 先頭が主のクリップで、設定パネルはそれを出す

        何本も選んでいれば、触った設定を選んだ全部へ当てる（同じ設定を持つものだけ）
        """
        primary = clip_ids[0] if clip_ids else None
        if primary == self._clip_id and tuple(clip_ids) == self._selection:
            return
        self._selection = tuple(clip_ids)
        self._clip_id = primary
        self._rebuild()

    def set_double_click_reset(self, enabled: bool) -> None:
        """名前（数はスライダーも）のダブルクリックで初期値へ戻すか 設定から

        切ったら本当に戻さない 行の説明（補足）からも消すので、作り直す
        """
        if enabled == self._double_click_reset:
            return
        self._double_click_reset = enabled
        self._rebuild()

    def _if_resettable(self, reset: Callable[[], None]) -> Callable[[], None] | None:
        """ダブルクリックで戻す手 設定で切ってあれば ``None``（行に付けない）"""
        return reset if self._double_click_reset else None

    def set_frame(self, frame: int) -> None:
        """再生位置 キーフレームの打点とアニメーション中の表示値に使う"""
        if frame == self._frame:
            return
        self._frame = frame
        self._refresh_animated()

    # --- 組み立て ---

    def _clip(self) -> Clip | None:
        located = self._located()
        return located[1] if located is not None else None

    def _located(self) -> tuple[Track, Clip] | None:
        if self._project is None or self._clip_id is None:
            return None
        return self._project.timeline.locate_clip(self._clip_id)

    def _rebuild(self) -> None:
        """中身を作り直す

        値だけ入れ直せば済む場合も多いが、エフェクトの増減や並べ替えを
        差分で追うと取りこぼしが出る 組み直す方が確実で、選択中の 1 クリップ
        ぶんなら十分に速い
        """
        self._editors.clear()
        self._key_controls.clear()
        self._virtual.clear()
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

        located = self._located()
        if located is None:
            self._title.show_identity(None)
            self._add_button.setEnabled(False)
            self._preset_button.setEnabled(False)
            self._body_layout.addStretch(1)
            return
        track, clip = located

        self._add_button.setEnabled(True)
        self._preset_button.setEnabled(True)
        self._show_identity(clip)

        # YMM4 のアイテムの並び 描画 → 中身 → 動画・音声 → 足したエフェクト
        # 絵を描かないクリップ（音声トラック）には描画の組を出さない（出すと、動かしても
        # 何も変わらない合成方法や不透明度が並ぶ） 音を鳴らさないクリップ（映像トラック）には
        # 音量を出さない（リンクした音は音声トラックのクリップの側にある）
        # 混合トラックはクリップが絵と音の両方を持てるので、トラックの種類ではなく
        # draws_picture と plays_sound で決める
        media = (
            self._project.find_media(clip.media_id)
            if self._project is not None and clip.media_id is not None
            else None
        )
        picture = draws_picture(track, clip, media)
        sound = (
            clip.media_id is not None and clip.source is None and plays_sound(track, clip, media)
        )
        shown: set[EffectId] = set()
        # 場面切り替えは下の絵をそのまま入れ替えて描き、不透明度・合成モード・クリッピングを
        # 読まない（描画の欄も持たない） 出すと、動かしても何も変わらない欄が並ぶ
        transition = clip.source is not None and clip.source.kind == "transition"
        if picture and not transition:
            self._body_layout.addWidget(self._build_picture_group(clip, shown))
        if clip.source is not None:
            section = self._build_source_section(clip)
            if section is not None:
                self._body_layout.addWidget(section)
        if picture and self._is_movie(clip):
            self._body_layout.addWidget(self._build_movie_group(clip, shown, sound=sound))
        elif sound:
            self._body_layout.addWidget(self._build_sound_group(clip, shown))

        heading = "映像エフェクト" if picture else "音声エフェクト"
        if picture and sound:
            heading = "映像・音声エフェクト"
        self._body_layout.addWidget(_heading(heading))
        for index, effect in enumerate(clip.effects):
            if effect.id in shown:
                continue
            self._body_layout.addWidget(self._build_effect_section(clip, effect, index))
        # 場面切り替えは、前の場面（上のエフェクト）と後の場面で別に積む
        if clip.source is not None and clip.source.kind == "transition":
            for index, effect in enumerate(clip.after_effects):
                self._body_layout.addWidget(
                    self._build_effect_section(clip, effect, index, after=True)
                )

        self._body_layout.addStretch(1)
        self._refresh_animated()

    def _show_identity(self, clip: Clip) -> None:
        identity = identify_clip(self._project, clip.id) if self._project is not None else None
        others = sum(1 for clip_id in self._selection if clip_id != clip.id)
        self._title.show_identity(identity, others=others)

    @property
    def header(self) -> ClipHeader:
        """上の見出し（何のクリップの設定か）"""
        return self._title

    def _describe(self, clip: Clip) -> str:
        if clip.source is not None:
            definition = source_registry.get(clip.source.kind)
            return definition.label if definition is not None else clip.source.kind
        if self._project is not None and clip.media_id is not None:
            media = self._project.find_media(clip.media_id)
            if media is not None:
                return media.name
        return "クリップ"

    def _blend_editor(self, parent: QWidget, clip: Clip) -> QComboBox:
        blend = QComboBox(parent)
        for mode in BlendMode.ALL:
            blend.addItem(BLEND_LABELS.get(mode, mode), mode)
        blend.setCurrentIndex(max(0, blend.findData(clip.blend_mode)))
        blend.currentIndexChanged.connect(
            lambda index: self._emit(
                SetClipProperty(clip.id, "blend_mode", str(blend.itemData(index)))
            )
        )
        return blend

    # --- 最初から持つ欄（YMM4 の描画・動画・音声の組） ---

    def _fixed_of(self, clip: Clip, kind: str) -> Effect:
        """クリップが持つ ``kind`` の欄 前の版のファイルで持っていなければ、既定の値の仮の物

        仮の物は触ったときに初めてクリップへ足す（:meth:`_send`）
        """
        found = next((e for e in clip.effects if e.fixed and e.kind == kind), None)
        if found is not None:
            return found
        virtual = fixed_effect(kind)
        self._virtual[virtual.id] = (clip.id, virtual)
        return virtual

    def _effect_row(
        self, section: _Section, clip: Clip, effect: Effect, name: str, label: str
    ) -> None:
        """欄のエフェクトの項目を 1 行 表示名は YMM4 の欄の名前にする"""
        definition = registry.get(effect.kind)
        spec = definition.spec(name) if definition is not None else None
        if spec is None:  # pragma: no cover - 固定の項目の定義は必ずある
            return
        path = ParamPath.of_effect(clip.id, effect.id, name)
        self._param_row(section, label, spec, path, effect.params.get(name))

    def _param_row(
        self,
        section: _Section,
        label: str,
        spec: ParameterSpec,
        path: ParamPath,
        value: ParamValue | None,
    ) -> None:
        """パラメータ 1 つの行 名前（ダブルクリックで初期値）・入力欄・キーフレームの ◀ ◆ ▶"""
        section.add_row(
            label,
            self._make_editor(spec, path, value),
            self._keyframe_controls(spec, path, value),
            reset=self._resetter(spec, path),
        )

    def _fixed_header(self, section: _Section, clip: Clip, effects: Sequence[Effect]) -> None:
        """組の見出しに、欄をまとめて切る切り替えと鍵の印を出す

        欄は外せないが無効にはできる（P1 の決まり） 1 つずつの切り替えを並べると、
        YMM4 の組の中に無い項目が増えて並びが崩れる
        """
        enabled = all(effect.enabled for effect in effects)
        toggle = QToolButton()
        toggle.setObjectName("fixed_toggle")
        toggle.setCheckable(True)
        toggle.setChecked(enabled)
        toggle.setText("有効" if enabled else "無効")
        toggle.setToolTip("この組の欄を掛けるかどうか 無効にすると既定の置き方・鳴り方に戻る")
        toggle.setAutoRaise(True)
        toggle.toggled.connect(
            lambda state: self._send(
                [
                    command
                    for effect in effects
                    for base in (SetEffectEnabled(clip.id, effect.id, bool(state)),)
                    for command in (base, *self._also_for_others(base))
                ],
                "欄を有効化" if state else "欄を無効化",
            )
        )
        section.add_header_widget(toggle)
        section.add_header_widget(_lock_label())

    def _build_picture_group(self, clip: Clip, shown: set[EffectId]) -> QWidget:
        """描画の組 YMM4 の並び（X・Y・不透明度・拡大率・回転角・合成モード・左右反転・
        クリッピング）のうち、今あるものだけを出す"""
        section = _Section("描画")
        placed = takes_picture_items(clip)
        transform = flip = None
        if placed:
            flip = self._fixed_of(clip, FLIP_EFFECT_KIND)
            transform = self._fixed_of(clip, TRANSFORM_EFFECT_KIND)
            shown.update((flip.id, transform.id))
            self._fixed_header(section, clip, (flip, transform))
            self._effect_row(section, clip, transform, "pos_x", "X")
            self._effect_row(section, clip, transform, "pos_y", "Y")

        opacity_spec = TrackSpec("opacity", "不透明度", 0, 1, 1, step=0.01)
        opacity_path = ParamPath.of_clip(clip.id, "opacity")
        self._param_row(section, "不透明度", opacity_spec, opacity_path, clip.opacity)
        if transform is not None:
            self._effect_row(section, clip, transform, "scale", "拡大率")
            self._effect_row(section, clip, transform, "rotation", "回転角")
        # フィルタは下の絵を置き換えるだけで、合成方法も切り抜きも使わない 出しておくと、
        # 選んでも何も変わらない欄を触らせることになる グループ制御も自分の絵を持たない
        if not (clip.is_filter or clip.is_group):
            section.add_row(
                "合成モード",
                self._blend_editor(section, clip),
                reset=self._if_resettable(
                    lambda: self._reset_clip(clip, "blend_mode", BlendMode.NORMAL, "合成モード")
                ),
            )
        if flip is not None:
            self._effect_row(section, clip, flip, "horizontal", "左右反転")
        if not (clip.is_filter or clip.is_group):
            self._clip_check(
                section, clip, "clip_to_below", "クリッピング", clip.clip_to_below, resettable=True
            )
        if self._native_capable(clip):
            # 前の版で置いた物は画面に収めて描いている 見た目を変えずに開くため、勝手には
            # 切り替えない 本人が素材の画素の大きさ（YMM4 の拡大率 100%）へ揃えたいときの道
            # 表示名は短くする 見出しの列は 96 画素で、長いと頭が切れる
            self._clip_check(
                section,
                clip,
                "native_size",
                "画素で置く",
                clip.native_size,
                tooltip="拡大率 100% を素材の画素の大きさにします 外すと画面に収めます",
            )
        return section

    def _clip_check(
        self,
        section: _Section,
        clip: Clip,
        name: str,
        label: str,
        value: bool,
        *,
        tooltip: str = "",
        resettable: bool = False,
    ) -> None:
        """クリップ自身の入り切り ``resettable`` なら名前のダブルクリックで切る（初期値）

        「画素で置く」は戻せる項目にしない 置いたときの値（入）と前の版の値（切）が
        違い、どちらが初期値かが決まらない
        """
        editor = create_editor(CheckSpec(name, label, False))
        editor.setObjectName(f"clip_{name}")
        editor.setToolTip(tooltip)
        editor.set_value(value)
        editor.value_changed.connect(
            lambda state: self._emit(SetClipProperty(clip.id, name, bool(state)))
        )
        reset = (lambda: self._reset_clip(clip, name, False, label)) if resettable else None
        section.add_row(label, editor, reset=self._if_resettable(reset) if reset else None)

    def _native_capable(self, clip: Clip) -> bool:
        """素材の画素の大きさで置けるクリップか（素材の絵を描くもの）"""
        if clip.media_id is None or clip.source is not None or self._project is None:
            return False
        media = self._project.find_media(clip.media_id)
        return media is not None and media.has_video

    def _is_movie(self, clip: Clip) -> bool:
        """動画の組を出すクリップか 静止画には再生の速さも位置も無い"""
        if clip.media_id is None or clip.source is not None or self._project is None:
            return False
        media = self._project.find_media(clip.media_id)
        return media is not None and media.has_video and not media.is_still

    def _build_movie_group(self, clip: Clip, shown: set[EffectId], *, sound: bool) -> QWidget:
        """動画の組 YMM4 の並び（音量・パン・再生速度・再生開始位置）

        音量とパンは ``sound``（混合トラックで音も鳴らすクリップ）のときだけ出す 分ける方式の
        映像のクリップは鳴らないので、出すと動かしても音が変わらない
        """
        section = _Section("動画")
        fade: Effect | None = None
        if sound:
            volume = self._fixed_of(clip, VOLUME_EFFECT_KIND)
            fade = self._fixed_of(clip, FADE_EFFECT_KIND)
            shown.update((volume.id, fade.id))
            self._fixed_header(section, clip, (volume, fade))
            self._effect_row(section, clip, volume, "volume", "音量")
            self._effect_row(section, clip, volume, "pan", "パン")
        self._playback_rows(section, clip)
        if fade is not None:
            self._effect_row(section, clip, fade, "fade_in", "フェードイン")
            self._effect_row(section, clip, fade, "fade_out", "フェードアウト")
        if clip.hold_at is not None:
            # 止めた絵は読み込み（YMM4 の素材より長い動画・再生速度 0）で付く 見えないままだと、
            # 絵が動かない理由がどこにも出ず、素材の不具合と取り違える 外す道も置く
            held = QLabel(f"素材の {float(clip.hold_at):.3f} 秒の絵で止める", section)
            held.setStyleSheet("border: none;")
            release = QPushButton("解除", section)
            release.clicked.connect(lambda: self._emit(SetClipProperty(clip.id, "hold_at", None)))
            section.add_row("絵を止める", held, release)
        return section

    def _build_sound_group(self, clip: Clip, shown: set[EffectId]) -> QWidget:
        """音声の組 YMM4 の並び（音量・パン・再生速度・再生開始位置・フェードイン・
        フェードアウト）"""
        section = _Section("音声")
        volume = self._fixed_of(clip, VOLUME_EFFECT_KIND)
        fade = self._fixed_of(clip, FADE_EFFECT_KIND)
        shown.update((volume.id, fade.id))
        self._fixed_header(section, clip, (volume, fade))
        self._effect_row(section, clip, volume, "volume", "音量")
        self._effect_row(section, clip, volume, "pan", "パン")
        if clip.media_id is not None and clip.source is None:
            self._playback_rows(section, clip)
        self._effect_row(section, clip, fade, "fade_in", "フェードイン")
        self._effect_row(section, clip, fade, "fade_out", "フェードアウト")
        return section

    def _playback_rows(self, section: _Section, clip: Clip) -> None:
        """再生速度（%）と再生開始位置（秒） どちらもクリップ自身の値

        リンクした相手（同じ素材の絵と音）にも同じ値を入れる 片方だけ変えると、絵と音が
        ずれていく
        """
        speed_spec = TrackSpec("speed", "再生速度", 1, 1000, 100, step=1, unit="%")
        speed = create_editor(speed_spec)
        speed.setObjectName("clip_speed")
        speed.set_value(AnimatedValue(float(clip.speed * 100)))
        speed.value_changed.connect(
            lambda value: self._set_linked(clip, "speed", _fraction(value, 100), "再生速度を変更")
        )
        speed.reset_requested.connect(lambda: self._reset_linked(clip, "speed", Fraction(1)))
        section.add_row(
            "再生速度",
            speed,
            reset=self._if_resettable(lambda: self._reset_linked(clip, "speed", Fraction(1))),
        )

        media = self._project.find_media(clip.media_id) if self._project and clip.media_id else None
        # 上限は素材の長さ 分からない素材は 10 時間まで（スライダーが整数で持てる範囲）
        length = float(media.duration) if media is not None and media.duration > 0 else 36000.0
        start_spec = TrackSpec(
            "source_in",
            "再生開始位置",
            0,
            max(length, float(clip.source_in)),
            0,
            step=0.01,
            unit="秒",
        )
        start = create_editor(start_spec)
        start.setObjectName("clip_source_in")
        start.set_value(AnimatedValue(float(clip.source_in)))
        start.value_changed.connect(
            lambda value: self._set_linked(
                clip, "source_in", _fraction(value, 1), "再生開始位置を変更"
            )
        )
        start.reset_requested.connect(lambda: self._reset_linked(clip, "source_in", Fraction(0)))
        section.add_row(
            "再生開始位置",
            start,
            reset=self._if_resettable(lambda: self._reset_linked(clip, "source_in", Fraction(0))),
        )

    def _reset_linked(self, clip: Clip, name: str, value: Fraction) -> None:
        """再生速度・再生開始位置を初期値へ戻す 相手にも入れる（:meth:`_set_linked`）"""
        current = self._clip() if self._clip_id == clip.id else None
        if not self._double_click_reset or getattr(current or clip, name) == value:
            return
        label = "再生速度" if name == "speed" else "再生開始位置"
        self._set_linked(clip, name, value, f"{label}を初期値に戻す")

    def _set_linked(self, clip: Clip, name: str, value: Fraction, label: str) -> None:
        """クリップ自身の値を、選んだほかのクリップとリンクした相手にも入れる"""
        if name == "speed" and value <= 0:
            return
        base = SetClipProperty(clip.id, name, value)
        commands: list[Command] = [base, *self._also_for_others(base)]
        touched = {c.clip_id for c in commands if isinstance(c, SetClipProperty)}
        for partner in self._link_partners(touched):
            commands.append(SetClipProperty(partner, name, value))
        self._send(commands, label)

    def _link_partners(self, clip_ids: set[ClipId]) -> list[ClipId]:
        if self._project is None:
            return []
        groups = {
            clip.link_group
            for track in self._project.timeline.tracks
            for clip in track.clips
            if clip.id in clip_ids and clip.link_group is not None
        }
        return [
            clip.id
            for track in self._project.timeline.tracks
            for clip in track.clips
            if clip.link_group in groups and clip.id not in clip_ids
        ]

    def _build_source_section(self, clip: Clip) -> QWidget | None:
        assert clip.source is not None
        definition = source_registry.get(clip.source.kind)
        if definition is None:
            return None

        section = _Section(definition.label)
        if clip.is_group:
            # 自分では描かないので、何を動かすのかをここで言う
            section.add_note(
                "手前に描くレイヤー（混合の方式では番号の大きい側）の対象レイヤー数ぶんの、"
                "同じ時間にある物を"
                " 1 つずつ、上の描画の X・Y・拡大率・回転角・不透明度で動かし、"
                "下に積んだエフェクトを掛けます 位置と拡大と回転は画面の中央（X・Y の所）を"
                "中心に掛かります 「1 枚の絵として扱う」を入れると、重ねて 1 枚にしてから"
                "掛けます（重なった半透明の物どうしが透けません）"
            )
        if clip.is_filter:
            # 設定の項目を持たないので、何もしない箱に見える 何に効くのかをここで言う
            section.add_note(
                "このトラックより下を重ねた絵に、下に積んだエフェクトを掛けます"
                " 不透明度は掛ける前と後の混ぜ具合です"
                " 部分モザイク・ぼかしと部分フィルタの範囲は、画面の中央から数えます"
                "（右と上が正）"
            )
        unused = definition.unused_names(clip.source.params)
        for spec in definition.parameters:
            if spec.name in unused:
                continue
            path = ParamPath.of_source(clip.id, spec.name)
            value = clip.source.params.get(spec.name)
            if clip.is_group:
                self._group_row(section, spec, path, value)
                continue
            self._param_row(section, spec.label, spec, path, value)
        return section

    def _group_row(
        self, section: _Section, spec: ParameterSpec, path: ParamPath, value: ParamValue | None
    ) -> None:
        """グループ制御の行 名前の列（96 画素）に長い名前を置くと頭しか見えないので短くし、
        入り切りは欄の中に言葉を添える 前は「1 枚の絵として扱う」の名前が切れて四角だけが並び、
        設定パネルにあるのに見つけられなかった（利用者の報告）"""
        editor = self._make_editor(spec, path, value)
        editor.setToolTip(spec.label)
        if isinstance(spec, CheckSpec):
            box = editor.findChild(QCheckBox)
            if box is not None:
                box.setText(spec.label)
            section.add_row("重ね方", editor, reset=self._resetter(spec, path))
            return
        section.add_row("対象レイヤー数", editor, reset=self._resetter(spec, path))

    def _build_effect_section(
        self, clip: Clip, effect: Effect, index: int, *, after: bool = False
    ) -> QWidget:
        definition = registry.get(effect.kind)
        label = definition.label if definition is not None else f"{effect.kind}（未知）"
        if after:
            label = f"{label}（後の場面）"
        stack = clip.after_effects if after else clip.effects
        # 隣が固定の項目なら、その向きへは動かせない（命令がまたぐ動きを断る）
        section = _Section(
            label,
            effect=effect,
            clip_id=clip.id,
            index=index,
            up_movable=index > 0 and not stack[index - 1].fixed,
            down_movable=index < len(stack) - 1 and not stack[index + 1].fixed,
            after=after,
        )
        section.action_requested.connect(self._emit)
        section.pressed.connect(lambda: self.effect_focused.emit(str(effect.id)))

        if definition is None:
            # 定義の無いエフェクトは触らせない 値の意味が分からないまま
            # 書き換えると、対応する版で開いたときに壊れて見える
            section.add_note("このエフェクトの定義が見つかりません 設定は保持されます")
            return section

        for spec in definition.parameters:
            path = ParamPath.of_effect(clip.id, effect.id, spec.name, after=after)
            self._param_row(section, spec.label, spec, path, effect.params.get(spec.name))
        return section

    def _make_editor(
        self, spec: ParameterSpec, path: ParamPath, value: ParamValue | None
    ) -> ParameterEditor:
        editor = create_editor(spec)
        editor.set_value(value)
        editor.value_changed.connect(lambda new: self._on_value_changed(path, new))
        editor.value_previewed.connect(lambda new: self._on_value_previewed(path, new))
        editor.reset_requested.connect(lambda: self._reset(spec, path))
        self._editors[_editor_key(path)] = editor
        return editor

    def _keyframe_controls(
        self, spec: ParameterSpec, path: ParamPath, value: ParamValue | None
    ) -> KeyframeControls | None:
        """キーフレームの ◀ ◆ ▶ 時間で動かせる値（数のスライダー）にだけ付く

        ◆ は再生位置にキーを打つ・そこにあるキーを消す ◀ ▶ は前後のキーへ再生位置を
        動かす（利用者の決定） キーの値は隣の入力欄で直す（再生位置のキーの値が変わる）
        """
        if not isinstance(spec, TrackSpec):
            return None
        base = spec.coerce(value)
        controls = KeyframeControls(path)
        controls.toggled.connect(lambda: self._toggle_keyframe(path, base))
        controls.stepped.connect(lambda direction: self._step_to_key(path, direction))
        controls.menu_requested.connect(lambda: self._keyframe_menu(path, base))
        self._key_controls[_editor_key(path)] = controls
        return controls

    # --- 操作 ---

    def _local_frame(self, clip: Clip) -> int | None:
        """再生位置をクリップの頭から数えたフレーム（キーフレームの持ち方） 外なら ``None``

        キーフレームはクリップの頭から数える 再生位置（タイムラインのフレーム）のまま
        打つと、頭が 0 より後ろのクリップでは、打った所と違う（多くはクリップの外の）
        時刻に点が入り、曲線にもタイムラインにも出ないまま値だけが変わる
        """
        local = self._frame - clip.timeline_start
        return local if 0 <= local < clip.duration else None

    def _key_frame(self, clip: Clip) -> int:
        """値を直したときにキーを入れるフレーム 再生位置がクリップの外なら近い端"""
        return min(max(self._frame - clip.timeline_start, 0), max(clip.duration - 1, 0))

    def _animated_of(self, path: ParamPath, base: AnimatedValue) -> AnimatedValue:
        """今の値 まだクリップに無い欄（:attr:`_virtual`）は作ったときの値 ``base`` を使う"""
        current = self._current_value(path)
        return current if isinstance(current, AnimatedValue) else base

    def _on_value_changed(self, path: ParamPath, value: ParamValue) -> None:
        if path.effect_id is not None:
            self.effect_focused.emit(str(path.effect_id))
        current = self._current_value(path)
        animating = isinstance(current, AnimatedValue) and current.is_animated
        clip = self._clip()
        if animating and isinstance(value, AnimatedValue) and clip is not None:
            # アニメーション中の値を触ったら、再生位置のキーフレームの値を変える（無ければ打つ）
            # 静的値で上書きすると、打ったキーフレームが黙って消える
            self._emit(SetKeyframe(path, self._key_frame(clip), value.static))
            return
        self._emit(SetParam(path, value))

    def _on_value_previewed(self, path: ParamPath, value: ParamValue) -> None:
        if path.effect_id is not None:
            self.effect_focused.emit(str(path.effect_id))
        pending = self._virtual.get(path.effect_id) if path.effect_id is not None else None
        if pending is not None:
            # まだ無い欄は、値を入れた欄を足した絵で見せる 値だけ変えようとすると、
            # 欄が見つからずにドラッグ中の絵が動かない
            clip_id, effect = pending
            self.preview_requested.emit(AddEffect(clip_id, effect.with_param(path.name, value)))
            return
        self.preview_requested.emit(SetParam(path, value))

    def _current_value(self, path: ParamPath) -> ParamValue | None:
        from sashimono.core.commands import resolve_param

        if self._project is None:
            return None
        return resolve_param(self._project, path)

    def _toggle_keyframe(self, path: ParamPath, base: AnimatedValue) -> None:
        """再生位置にキーがあれば消し、無ければ今の値で打つ

        選んだほかのクリップには当てない（主の 1 本だけ） まとめて当てると、ほかの
        クリップのキーの値が主のクリップの値に書き換わる
        """
        clip = self._clip()
        local = self._local_frame(clip) if clip is not None else None
        if local is None:
            return
        animated = self._animated_of(path, base)
        if any(k.frame == local for k in animated.keyframes):
            command: Command = RemoveKeyframe(path, local)
            label = "キーフレームを削除"
        else:
            command = SetKeyframe(path, local, animated.at(local))
            label = "キーフレームを打つ"
        self._send([command], label)
        self.param_focused.emit(path)

    def _step_to_key(self, path: ParamPath, direction: int) -> None:
        """前（``-1``）か次（``1``）のキーへ再生位置を動かす クリップの外のキーは見ない"""
        clip = self._clip()
        current = self._current_value(path)
        if clip is None or not isinstance(current, AnimatedValue):
            return
        local = self._frame - clip.timeline_start
        frames = [k.frame for k in current.keyframes if 0 <= k.frame < clip.duration]
        if direction < 0:
            target = max((f for f in frames if f < local), default=None)
        else:
            target = min((f for f in frames if f > local), default=None)
        if target is None:
            return
        self.seek_requested.emit(clip.timeline_start + target)
        self.param_focused.emit(path)

    def _keyframe_menu(self, path: ParamPath, base: AnimatedValue) -> None:
        animated = self._animated_of(path, base)
        clip = self._clip()
        menu = QMenu(self)
        curve = menu.addAction("グラフエディタで開く")
        clear = menu.addAction("アニメーションを解除")
        clear.setEnabled(animated.is_animated)

        chosen = menu.exec(self.cursor().pos())
        if chosen is curve:
            self.curve_selected.emit(path)
        elif chosen is clear and clip is not None:
            # 解除した後に残す値は再生位置の値 キーはクリップの頭から数えるので、
            # 再生位置もクリップの頭から数えて渡す
            self._emit(ClearKeyframes(path, self._key_frame(clip)))

    # --- 初期値へ戻す ---

    def _resetter(self, spec: ParameterSpec, path: ParamPath) -> Callable[[], None] | None:
        """名前のダブルクリックで初期値へ戻す手 戻せない項目（文字・ファイル・格子）は ``None``

        文字やファイルの場所は、戻すと打った中身や選んだ素材が消える ダブルクリックは
        うっかり起きやすいので、打ち直しの利かない物には付けない
        """
        if isinstance(spec, TextSpec | FileSpec | GridSpec):
            return None
        return self._if_resettable(lambda: self._reset(spec, path))

    def _reset(self, spec: ParameterSpec, path: ParamPath) -> None:
        """パラメータを初期値へ戻す（取り消せる）

        キーフレームのある数の値は、再生位置のキーの値だけを初期値にする（そこにキーが
        無ければ初期値のキーを打つ） ほかのキーは残す（利用者の決定） アニメーションごと
        消すと、1 か所を戻したいだけでも打ったキーが全部消える
        """
        if not self._double_click_reset:
            return
        current = self._current_value(path)
        clip = self._clip()
        label = f"{spec.label}を初期値に戻す"
        if (
            isinstance(spec, TrackSpec)
            and isinstance(current, AnimatedValue)
            and current.is_animated
            and clip is not None
        ):
            frame = self._key_frame(clip)
            here = next((k for k in current.keyframes if k.frame == frame), None)
            if here is not None and here.value == spec.default:
                return
            self._emit(SetKeyframe(path, frame, spec.default), label)
            return
        default = spec.default_value()
        pending = path.effect_id is not None and path.effect_id in self._virtual
        if (current is None and pending) or spec.coerce(current) == default:
            # もう初期値（まだクリップに無い欄は初期値のまま） 同じ値を入れると、戻しても
            # 何も変わらない取り消しの段が積まれる
            return
        self._emit(SetParam(path, default), label)

    def _reset_clip(self, clip: Clip, name: str, value: object, label: str) -> None:
        """クリップ自身の値（合成モード・クリッピング）を初期値へ戻す

        今の値はプロジェクトから引き直す 行を作ったときのクリップで比べると、作った後に
        変えた値を「もう初期値」と見誤ることがある
        """
        current = self._clip() if self._clip_id == clip.id else None
        if not self._double_click_reset or getattr(current or clip, name) == value:
            return
        self._emit(SetClipProperty(clip.id, name, value), f"{label}を初期値に戻す")

    def _show_effect_menu(self) -> None:
        menu = self.effect_menu()
        if menu is None:
            return
        chosen = menu.exec(self._add_button.mapToGlobal(self._add_button.rect().bottomLeft()))
        if chosen is None:
            return
        self._add_chosen_effect(chosen)

    def effect_menu(self) -> QMenu | None:
        """〔＋ エフェクト〕のメニュー 選んでいるクリップに効く物だけを並べる

        音だけのクリップに映像のエフェクト、絵だけのクリップに音のエフェクトを並べると、
        積めても何も起きない（タイムラインの右クリックと同じ決まり :func:`effects_for_clip`）
        """
        located = self._located()
        if located is None or self._project is None:
            return None
        track, clip = located
        definitions = effects_for_clip(self._project, track, clip)

        menu = QMenu(self)
        # 場面切り替えは、前の場面と後の場面で積む先が違う
        transition = clip.source is not None and clip.source.kind == "transition"
        roots: dict[bool, QMenu] = {False: menu}
        if transition:
            roots = {False: menu.addMenu("前の場面へ"), True: menu.addMenu("後の場面へ")}
        submenus: dict[tuple[bool, str], QMenu] = {}
        for after, root in roots.items():
            for definition in definitions:
                submenu = submenus.get((after, definition.category))
                if submenu is None:
                    submenu = root.addMenu(definition.category)
                    submenus[(after, definition.category)] = submenu
                action = submenu.addAction(definition.label)
                action.setData((definition.kind, after))
        return menu

    def _add_chosen_effect(self, chosen: QAction) -> None:
        clip = self._clip()
        if clip is None:
            return
        kind, after = chosen.data()
        definition = registry.require(str(kind))
        where = "（後の場面）" if after else ""
        self._emit(
            AddEffect(clip.id, definition.create(), after=bool(after)),
            f"{definition.label}を追加{where}",
        )

    def _show_preset_menu(self) -> None:
        """プリセットの保存と適用

        保存するのはエフェクトの列ごと 見た目のほとんどは複数のエフェクトの
        組み合わせでできているので、1 つずつ保存しても使い物にならない
        """
        clip = self._clip()
        if clip is None:
            return

        menu = QMenu(self)
        save = menu.addAction("この構成を保存…")
        save.setEnabled(bool(_loose(clip)))
        menu.addSeparator()

        presets = self._presets.all()
        if not presets:
            placeholder = menu.addAction("（保存されたプリセットはありません）")
            placeholder.setEnabled(False)
        else:
            submenus: dict[str, QMenu] = {}
            for preset in presets:
                submenu = submenus.get(preset.category)
                if submenu is None:
                    submenu = menu.addMenu(preset.category)
                    submenus[preset.category] = submenu
                action = submenu.addAction(preset.name)
                action.setData(preset.name)

        chosen = menu.exec(self._preset_button.mapToGlobal(self._preset_button.rect().bottomLeft()))
        if chosen is None:
            return
        if chosen is save:
            self._save_preset(clip)
            return

        selected = next((p for p in presets if p.name == chosen.data()), None)
        if selected is not None:
            self.commands_requested.emit(
                [AddEffect(clip.id, effect) for effect in selected.instantiate()],
                f"プリセット: {selected.name}",
            )

    def _save_preset(self, clip: Clip) -> None:
        name, accepted = QInputDialog.getText(
            self, "プリセットを保存", "名前", text=self._describe(clip)
        )
        if not accepted or not name.strip():
            return
        self._presets.save(Preset(name=name.strip(), effects=_loose(clip)))

    def _emit(self, command: Command, label: str | None = None) -> None:
        commands = [command, *self._also_for_others(command)]
        text = label or command.label
        if len(commands) > 1:
            text = f"{text}（{len(commands)} 本）"
        self._send(commands, text)

    def _send(self, commands: list[Command], label: str) -> None:
        """コマンドをまとめて出す（1 回の取り消しで戻る）

        まだクリップに無い欄（:attr:`_virtual`）を指すものがあれば、その前に欄を足す
        足すのと値を入れるのを別々に出すと、取り消しが 2 段になり、1 回戻しただけでは
        既定の値の欄が残る
        """
        materialized: list[Command] = []
        added: set[EffectId] = set()
        for command in commands:
            target = _effect_of(command)
            pending = self._virtual.get(target) if target is not None else None
            if pending is not None and target not in added:
                clip_id, effect = pending
                materialized.append(AddEffect(clip_id, effect))
                added.add(effect.id)
            materialized.append(command)
        self.commands_requested.emit(materialized, label)

    def _also_for_others(self, command: Command) -> list[Command]:
        """同じ設定を、選んでいるほかのクリップにも当てるコマンド

        エフェクトのパラメータは「同じ種類の何番目か」で相手を探す 相手が持って
        いなければ飛ばす（無いものを作ると、選んだだけで中身が増える）
        """
        others = [clip_id for clip_id in self._selection if clip_id != self._clip_id]
        if not others or self._project is None:
            return []
        primary = self._clip()
        if primary is None:
            return []
        target = _effect_of(command)
        pending = self._virtual.get(target) if target is not None else None
        if pending is not None:
            # 主のクリップの欄がまだ無い（前の版のファイル）ときは、仮の欄を持たせた形で
            # 相手を探す 探さないと、相手が実在の欄を持っていても値が当たらない
            # 相手に欄が無ければ作らない（ほかのエフェクトと同じく飛ばす）
            primary = replace(primary, effects=(*primary.effects, pending[1]))
        extra: list[Command] = []
        for clip_id in others:
            located = self._project.timeline.locate_clip(clip_id)
            if located is None:
                continue
            copied = _for_clip(command, primary, located[1])
            if copied is not None:
                extra.append(copied)
        return extra

    def _refresh_animated(self) -> None:
        """キーフレームで決まる値を、今のフレームの値に更新する"""
        clip = self._clip()
        if clip is None:
            return

        local = self._frame - clip.timeline_start
        for (owner, name), editor in self._editors.items():
            if not isinstance(editor, TrackEditor):
                continue
            value = self._lookup(clip, owner, name)
            if isinstance(value, AnimatedValue) and value.is_animated:
                editor.set_animated_value(value.at(local))
        inside = 0 <= local < clip.duration
        for (owner, name), controls in self._key_controls.items():
            value = self._lookup(clip, owner, name)
            frames = (
                [k.frame for k in value.keyframes if 0 <= k.frame < clip.duration]
                if isinstance(value, AnimatedValue)
                else []
            )
            controls.show_state(frames, local, inside=inside)

    def _lookup(self, clip: Clip, owner: str, name: str) -> ParamValue | None:
        if owner == "clip":
            return getattr(clip, name, None)
        if owner == "source":
            return clip.source.params.get(name) if clip.source is not None else None
        # 場面切り替えは前の場面と後の場面の 2 列を持つ どちらに積んだものも拾う
        both = (*clip.effects, *clip.after_effects)
        effect = next((e for e in both if e.id == owner), None)
        return effect.params.get(name) if effect is not None else None


def _effect_of(command: Command) -> EffectId | None:
    """コマンドが指すエフェクト（値を変える・点を打つ・切り替える物）"""
    if isinstance(command, SetParam | SetKeyframe | RemoveKeyframe | ClearKeyframes):
        return command.path.effect_id
    if isinstance(command, SetEffectEnabled):
        return command.effect_id
    return None


def _loose(clip: Clip) -> tuple[Effect, ...]:
    """足したエフェクト（最初から持つ欄を除く）

    プリセットには欄を入れない 入れると当てるたびに既定のままの配置や反転が
    ふつうのエフェクトとして増え、足した物の一覧が読めなくなる
    """
    return tuple(effect for effect in clip.effects if not effect.fixed)


def _fraction(value: ParamValue, scale: int) -> Fraction:
    """数の入力欄の値を、クリップが持つ分数へ ``scale`` で割る（% を倍率へ）

    小数のまま渡すと保存の所で分数に直せない（SetClipProperty が断る） 入力欄の
    刻みより細かい桁は意味が無いので丸める
    """
    number = value.static if isinstance(value, AnimatedValue) else 0.0
    return Fraction(number).limit_denominator(1_000_000) / scale


def _heading(text: str) -> QLabel:
    """足したエフェクトの一覧の見出し（YMM4 の「映像エフェクト」「音声エフェクト」）"""
    label = QLabel(text)
    label.setObjectName("effects_heading")
    label.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()}; font-weight: bold;")
    return label


def _lock_label() -> QLabel:
    lock = QLabel()
    lock.setObjectName("fixed_lock")
    lock.setPixmap(lock_pixmap(lock.devicePixelRatioF()))
    lock.setAccessibleName("固定の項目")
    lock.setToolTip(
        "クリップが最初から持つ項目です 外すことと並べ替えはできません"
        " 無効にはできます 重ねて掛けたいときは同じエフェクトを追加してください"
    )
    lock.setStyleSheet("border: none;")
    return lock


#: 鍵の印を見せる大きさ（論理画素） 見出しの ▲ ▼ ✕ の文字と同じくらい
LOCK_SIZE = 14

#: 描く細かさ :func:`sashimono.ui.transport.transport_icon` と同じく、大きめに描いて
#: 縮めて見せる 画面の拡大率で引き伸ばしても角がぼけない
_LOCK_SCALE = 4


def lock_pixmap(device_pixel_ratio: float = 1.0) -> QPixmap:
    """固定の項目の鍵の印

    文字（絵文字の鍵）で出すと、Windows ではカラーの絵文字の書体で描かれ、ほかの
    ボタンと揃わない（再生ボタンの ``⏸`` が青い四角になったのと同じ Issue #27）
    書体に頼らず、見出しのボタンの文字と同じ色で自前で描く 16 × 16 の枠で描く
    """
    pixmap = QPixmap(16 * _LOCK_SCALE, 16 * _LOCK_SCALE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(_LOCK_SCALE, _LOCK_SCALE)

    # つる 本体に隠れる所まで伸ばし、付け根に隙間が見えないようにする
    shackle = QPainterPath()
    shackle.moveTo(5, 9)
    shackle.lineTo(5, 6)
    shackle.arcTo(QRectF(5, 2.5, 6, 7), 180, -180)
    shackle.lineTo(11, 9)
    pen = QPen(Colors.TEXT, 1.8)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(shackle)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(Colors.TEXT)
    painter.drawRoundedRect(QRectF(3, 7.5, 10, 7), 1.2, 1.2)
    # 鍵穴は抜いて見せる 塗りつぶしの四角だけだと、鍵ではなく箱に見える
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    painter.drawEllipse(QPointF(8, 10.5), 1.2, 1.2)
    painter.drawRect(QRectF(7.5, 10.5, 1, 2.2))
    painter.end()

    icon = QIcon()
    icon.addPixmap(pixmap)
    return icon.pixmap(QSize(LOCK_SIZE, LOCK_SIZE), device_pixel_ratio)


class _Section(QFrame):
    """1 つの見出しと、その下のパラメータ行"""

    action_requested = Signal(object)
    #: 組の地（見出しや行の間）が押された どのエフェクトを見ているかを知らせるため
    pressed = Signal()

    def __init__(
        self,
        title: str,
        *,
        effect: Effect | None = None,
        clip_id: ClipId | None = None,
        index: int = 0,
        up_movable: bool = False,
        down_movable: bool = False,
        after: bool = False,
    ) -> None:
        super().__init__()
        #: 見出しの言葉 組の並び（描画 → 中身 → 動画・音声 → エフェクト）を試験で見る
        self.heading = title
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Colors.PANEL.name()};"
            f" border: 1px solid {Colors.BORDER.name()}; border-radius: 4px; }}"
        )

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(8, 6, 8, 8)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(6)
        self._grid.setColumnStretch(1, 1)
        self._row = 0

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setStyleSheet(f"color: {Colors.TEXT.name()}; font-weight: bold; border: none;")
        header.addWidget(label)
        header.addStretch(1)
        self._header = header

        self._after = after
        if effect is not None and clip_id is not None:
            header.addWidget(self._toggle(effect, clip_id))
            if effect.fixed:
                # 外すことも並べ替えることもできない 押せないボタンを並べるより、
                # 鍵の印で「最初からある欄」だと示す方が、押せない理由まで伝わる
                header.addWidget(_lock_label())
            else:
                header.addWidget(self._move(effect, clip_id, index - 1, "▲", up_movable))
                header.addWidget(self._move(effect, clip_id, index + 1, "▼", down_movable))
                header.addWidget(self._remove(effect, clip_id))

        container = QWidget(self)
        container.setStyleSheet("border: none;")
        container.setLayout(header)
        self._grid.addWidget(container, 0, 0, 1, 3)
        self._row = 1

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        self.pressed.emit()
        super().mousePressEvent(event)

    def add_row(
        self,
        label: str,
        editor: QWidget,
        extra: QWidget | None = None,
        *,
        reset: Callable[[], None] | None = None,
    ) -> None:
        """1 行足す ``reset`` を渡すと、名前のダブルクリックで初期値へ戻す"""
        text = _RowLabel(label, reset)
        text.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()}; border: none;")
        text.setFixedWidth(96)
        text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._grid.addWidget(text, self._row, 0)
        self._grid.addWidget(editor, self._row, 1)
        if extra is not None:
            self._grid.addWidget(extra, self._row, 2)
        self._row += 1

    def add_note(self, message: str) -> None:
        note = QLabel(message)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {Colors.TEXT_MUTED.name()}; border: none;")
        self._grid.addWidget(note, self._row, 0, 1, 3)
        self._row += 1

    def _toggle(self, effect: Effect, clip_id: ClipId) -> QToolButton:
        button = QToolButton()
        button.setCheckable(True)
        button.setChecked(effect.enabled)
        button.setText("有効" if effect.enabled else "無効")
        button.setToolTip("掛ける前と後を見比べる")
        button.setAutoRaise(True)
        button.toggled.connect(
            lambda state: self.action_requested.emit(
                SetEffectEnabled(clip_id, effect.id, bool(state), after=self._after)
            )
        )
        return button

    def _move(
        self, effect: Effect, clip_id: ClipId, index: int, text: str, enabled: bool
    ) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip("順番を変える 掛ける順で結果が変わる")
        button.setAutoRaise(True)
        button.setEnabled(enabled)
        button.clicked.connect(
            lambda: self.action_requested.emit(
                MoveEffect(clip_id, effect.id, index, after=self._after)
            )
        )
        return button

    def add_header_widget(self, widget: QWidget) -> None:
        """見出しの右端へ部品を足す（描画・音声の組の切り替えと鍵の印）"""
        self._header.addWidget(widget)

    def _remove(self, effect: Effect, clip_id: ClipId) -> QToolButton:
        button = QToolButton()
        button.setText("✕")
        button.setToolTip("このエフェクトを外す")
        button.setAutoRaise(True)
        button.clicked.connect(
            lambda: self.action_requested.emit(RemoveEffect(clip_id, effect.id, after=self._after))
        )
        return button


def _editor_key(path: ParamPath) -> tuple[str, str]:
    """入力欄と ◀ ◆ ▶ を引く鍵（エフェクトの ID か持ち主の種類, 名前）"""
    return str(path.effect_id or path.target.value), path.name


class _RowLabel(QLabel):
    """行の名前 ダブルクリックで初期値へ戻す（戻せる項目だけ）

    戻すのを名前のダブルクリックにしたのは、入力欄のダブルクリックがすでに別の意味を
    持つから（数値欄は数字を選んで打ち直す・色は 1 回目の押下で色の窓が開く）
    数のスライダーは入力欄の側でもダブルクリックで戻す（:class:`TrackEditor`）
    """

    def __init__(self, text: str, reset: Callable[[], None] | None) -> None:
        super().__init__(text)
        self._reset = reset
        # 名前の列は狭く、長い名前は頭しか見えない 補足で全部を読めるようにする
        self.setToolTip(f"{text}（ダブルクリックで初期値に戻す）" if reset is not None else text)

    @property
    def resettable(self) -> bool:
        return self._reset is not None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt の命名規約
        if self._reset is not None and event.button() == Qt.MouseButton.LeftButton:
            self._reset()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


#: キーフレームの ◀ ◆ ▶ のボタン 1 つの幅（画素） 3 つ並べて設定パネルの幅
#: （320 画素）に数値欄とスライダーが収まる幅
_KEY_BUTTON_WIDTH = 20


class KeyframeControls(QWidget):
    """1 つの値のキーフレームの操作 ◀（前のキーへ）◆（打つ・消す）▶（次のキーへ）

    ◆ は再生位置にキーがあれば塗りつぶし（押すと消す）、無ければ白抜き（押すと打つ）
    その値にキーが 1 つでもあればアクセント色、無ければ薄い色 右クリックで
    グラフエディタで開く・アニメーションを解除
    自分ではプロジェクトを変えない 押されたことを知らせるだけ
    """

    #: ◆ が押された
    toggled = Signal()
    #: ◀（``-1``）か ▶（``1``）が押された
    stepped = Signal(int)
    #: ◆ の右クリック
    menu_requested = Signal()

    def __init__(self, path: ParamPath) -> None:
        super().__init__()
        self.path = path
        self.setObjectName("keyframe_controls")
        self.setStyleSheet("border: none;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.previous = self._button("◀", "前のキーへ再生位置を動かす")
        self.toggle = self._button("◇", "")
        self.next = self._button("▶", "次のキーへ再生位置を動かす")
        self.previous.clicked.connect(lambda: self.stepped.emit(-1))
        self.next.clicked.connect(lambda: self.stepped.emit(1))
        self.toggle.clicked.connect(self.toggled.emit)
        self.toggle.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.toggle.customContextMenuRequested.connect(lambda _: self.menu_requested.emit())
        for button in (self.previous, self.toggle, self.next):
            layout.addWidget(button)
        self.show_state((), 0, inside=True)

    def _button(self, text: str, tip: str) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setToolTip(tip)
        button.setAutoRaise(True)
        button.setFixedWidth(_KEY_BUTTON_WIDTH)
        return button

    def show_state(self, frames: Sequence[int], local: int, *, inside: bool) -> None:
        """キーの位置（クリップの頭から数えたフレーム）と再生位置 ``local`` で見た目を決める

        ``inside`` が偽（再生位置がクリップの外）なら ◆ を押せなくする 外に打った
        キーは描く所が無く、曲線にもタイムラインにも出ない
        """
        here = local in frames
        self.toggle.setText("◆" if here else "◇")
        colour = Colors.ACCENT if frames else Colors.TEXT_MUTED
        self.toggle.setStyleSheet(_key_style(colour))
        self.toggle.setEnabled(inside)
        if not inside:
            tip = "再生位置がクリップの外なので打てません"
        elif here:
            tip = "再生位置のキーを消す（右クリックでグラフエディタ・解除）"
        elif frames:
            tip = "再生位置にキーを打つ（右クリックでグラフエディタ・解除）"
        else:
            tip = "再生位置にキーを打つ 打つとこの値が時間で動くようになる"
        self.toggle.setToolTip(tip)
        for button in (self.previous, self.next):
            button.setStyleSheet(_key_style(Colors.TEXT))
        self.previous.setEnabled(any(f < local for f in frames))
        self.next.setEnabled(any(f > local for f in frames))


def _key_style(colour: QColor) -> str:
    """◀ ◆ ▶ の見た目 押せないときは枠の色まで薄くする

    色を決め打ちにすると、押せないボタンも押せるボタンと同じ色で描かれ、飛べるキーが
    無いことが見えない 全体のスタイルの左右 10 画素の余白も外す（20 画素の幅に字が入らない）
    """
    return (
        f"QToolButton {{ color: {colour.name()}; padding: 0px; border: none; }}"
        f" QToolButton:disabled {{ color: {Colors.BORDER.name()}; }}"
    )
