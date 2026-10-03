"""プロジェクト設定（解像度と重ね合わせの方法） 新規作成のときはフレームレートも

解像度の選びには、決まった組み合わせ（:data:`RESOLUTION_PRESETS`）に続けて、本人が名前を
付けて保存した組み合わせ（:mod:`sashimono.ui.project_presets`）も並べる
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sashimono.core.commands.edit import MAX_RESOLUTION, MIN_RESOLUTION
from sashimono.core.model import Blending, LayerMode, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.ui.project_presets import PresetStoreError, ProjectPreset, ProjectPresetStore
from sashimono.ui.theme import Colors, themed_style

__all__ = [
    "BLENDING_CHOICES",
    "FRAME_RATE_PRESETS",
    "LAYER_MODE_CHOICES",
    "RESOLUTION_PRESETS",
    "SAVED_PREFIX",
    "ProjectSettingsDialog",
]

#: 重ね合わせの方法 表示名には、どちらを選ぶと何が起きるかを測った値で書く
#: （黒の上に不透明度 50% の白を重ねたときの明るさ 0〜255）
BLENDING_CHOICES: tuple[tuple[str, str], ...] = (
    (Blending.SRGB, "sRGB（AviUtl・YMM4 と同じ 黒に 50% の白で 128）"),
    (Blending.LINEAR, "リニア（光の量で混ぜる 黒に 50% の白で 188）"),
)

#: レイヤーの方式 新規作成の窓と、設定（新しいプロジェクトの初期値）の両方に並べる
LAYER_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    (LayerMode.MIXED, "混合（YMM4・AviUtl と同じ 1 本のレイヤーに何でも置く）"),
    (LayerMode.SEPARATED, "分ける（映像トラックと音声トラック）"),
)

#: 選べるフレームレート 分数のものは分数のまま持つ 29.97 を小数で持つと、
#: 1 時間で 3 フレーム以上ずれる（:meth:`FrameRate.from_decimal` を参照）
FRAME_RATE_PRESETS: tuple[tuple[str, FrameRate], ...] = (
    ("23.976 fps（映画の NTSC 版）", FrameRate(24000, 1001)),
    ("24 fps（映画）", FrameRate(24)),
    ("25 fps（PAL）", FrameRate(25)),
    ("29.97 fps（テレビ・NTSC）", FrameRate(30000, 1001)),
    ("30 fps（配信・ゆっくり実況）", FrameRate(30)),
    ("50 fps", FrameRate(50)),
    ("59.94 fps", FrameRate(60000, 1001)),
    ("60 fps（ゲーム実況）", FrameRate(60)),
)

#: よく使う解像度 表示名は用途で書く 数字だけだと縦か横かを取り違える
RESOLUTION_PRESETS: tuple[tuple[str, int, int], ...] = (
    ("フル HD 横（1920×1080）", 1920, 1080),
    ("HD 横（1280×720）", 1280, 720),
    ("4K 横（3840×2160）", 3840, 2160),
    ("縦動画・ショート（1080×1920）", 1080, 1920),
    ("正方形（1080×1080）", 1080, 1080),
)

#: 保存した組み合わせの表示名の頭 決まった組み合わせと同じ一覧に並ぶので、
#: 見ただけで本人が保存したものだと分かるようにする（削除できるのもこちらだけ）
SAVED_PREFIX = "保存: "

_CUSTOM = "指定する"

#: 選びの項目に持たせる値の形 ``builtin:<番号>`` / ``saved:<名前>`` / ``custom``
_BUILTIN_KEY = "builtin:"
_SAVED_KEY = "saved:"
_CUSTOM_KEY = "custom"

_BLENDING_SHORT = {Blending.SRGB: "sRGB", Blending.LINEAR: "リニア"}


class ProjectSettingsDialog(QDialog):
    """解像度を選ぶ フレームレートは ``new`` のとき（新規作成）だけ選ばせる

    タイムラインの位置はフレーム番号で持っている あとからフレームレートを変えると、
    すべてのクリップとキーフレームを換算し直すことになり、端数の丸めで 1 フレームの
    隙間や重なりが出る いまは作るときにだけ決める

    ``presets`` は保存した組み合わせの置き場 渡さなければ本人の設定の置き場を使う
    """

    def __init__(
        self,
        settings: ProjectSettings,
        parent: QWidget | None = None,
        *,
        new: bool = False,
        presets: ProjectPresetStore | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("新規プロジェクト" if new else "プロジェクト設定")
        self._base = settings
        self._rate: QComboBox | None = None
        self._store = presets if presets is not None else ProjectPresetStore()
        self._saved: list[ProjectPreset] = self._store.load()
        #: 組み合わせを当てている途中 1 つ目の欄を変えた所で選びを合わせ直すと、
        #: 残りの欄を変える前の中途半端な数字で別の項目が選ばれる
        self._applying = False

        self._preset = QComboBox(self)
        self._fill_presets()

        self._width = self._spin(settings.width)
        self._height = self._spin(settings.height)
        swap = QPushButton("縦横を入れ替える", self)
        swap.clicked.connect(self._swap)

        size_row = QHBoxLayout()
        size_row.addWidget(self._width)
        size_row.addWidget(QLabel("×", self))
        size_row.addWidget(self._height)
        size_row.addWidget(swap)

        self._save_button = QPushButton("名前を付けて保存…", self)
        self._save_button.setToolTip(
            "今の解像度・フレームレート・重ね合わせを、名前を付けて一覧に足す\n"
            "新しいプロジェクトを作るときにも選べる"
        )
        self._save_button.clicked.connect(self._ask_save)
        self._delete_button = QPushButton("削除", self)
        self._delete_button.setToolTip("選んでいる保存した組み合わせを一覧から消す")
        self._delete_button.clicked.connect(self._ask_delete)
        preset_row = QHBoxLayout()
        preset_row.addWidget(self._preset, 1)
        preset_row.addWidget(self._save_button)
        preset_row.addWidget(self._delete_button)

        rate: QWidget
        if new:
            self._rate = QComboBox(self)
            for label, preset_rate in FRAME_RATE_PRESETS:
                self._rate.addItem(label, preset_rate)
            # 保存したテンプレートの中の、表に無いレート（読み込んだ作品の 15fps で保存した物・
            # 手で書いた 48fps など）も選べるようにする 選べないと、そのテンプレートを選んでも
            # フレームレートだけ前のまま残り、黙って別のレートで作ることになる
            for extra in (settings.frame_rate, *(p.frame_rate for p in self._saved)):
                if self._rate_index(extra) < 0:
                    self._rate.addItem(f"{extra} fps（保存したテンプレート）", extra)
            self._rate.setCurrentIndex(max(0, self._rate_index(settings.frame_rate)))
            rate = self._rate
        else:
            rate = QLabel(f"{settings.frame_rate} fps（作成後は変えられません）", self)
            rate.setEnabled(False)

        self._blending = QComboBox(self)
        for value, label in BLENDING_CHOICES:
            self._blending.addItem(label, value)
        values = [value for value, _ in BLENDING_CHOICES]
        if settings.blending in values:
            self._blending.setCurrentIndex(values.index(settings.blending))
        # 半透明の文字・影・フェードの明るさがすべて変わる 何が変わるのかを選ぶ所で言う
        self._blending.setToolTip(
            "半透明の絵を重ねるときに、どの値で混ぜるか\n"
            "AviUtl や YMM4 の素材を使うなら sRGB（同じ明るさになる）\n"
            "リニアは物理的に正しい混ぜ方で、半透明の所やフェードの途中が明るく出る\n"
            "この設定ができる前に保存したプロジェクトはリニアで開く"
        )

        #: レイヤーの方式 新規作成のときだけ出す 作ったあとに変えるのは、置いてあるトラックを
        #: 変換するかを尋ねる別の入口（窓の :meth:`MainWindow.switch_layer_mode` P5 #163）
        #: ここで方式だけを黙って変えると、置いてあるトラックはそのままで置き方だけが変わる
        self._layers: QComboBox | None = None
        if new:
            self._layers = QComboBox(self)
            for value, label in LAYER_MODE_CHOICES:
                self._layers.addItem(label, value)
            self._layers.setCurrentIndex(max(0, self._layers.findData(settings.layer_mode)))
            self._layers.setToolTip(
                "混合は YMM4・AviUtl と同じく、1 本のレイヤーに動画・音声・テキストを何でも置く\n"
                "音付きの動画は 1 本のクリップになり、番号が大きい（下の）レイヤーほど手前に描く\n"
                "分けるは映像トラックと音声トラックを別に並べ、動画は絵と音の 2 本を結んで置く\n"
                "初めの値は 表示 → 設定… の「新しいプロジェクトの置き方」で変えられる\n"
                "作ったあとは ファイル → 置き方の方式を切り替える… で変える"
                "（置いてあるトラックも変換できる）"
            )

        self._warning = QLabel(self)
        themed_style(self._warning, lambda: f"color: {Colors.WARNING.name()};")
        self._warning.setWordWrap(True)

        form = QFormLayout()
        form.addRow("解像度", preset_row)
        form.addRow("", size_row)
        form.addRow("フレームレート", rate)
        form.addRow("重ね合わせ", self._blending)
        if self._layers is not None:
            form.addRow("置き方の方式", self._layers)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._warning)
        layout.addWidget(self._buttons)

        self._preset.currentIndexChanged.connect(self._on_preset)
        self._width.valueChanged.connect(self._sync)
        self._height.valueChanged.connect(self._sync)
        self._blending.currentIndexChanged.connect(self._sync)
        if self._rate is not None:
            self._rate.currentIndexChanged.connect(self._sync)
        self._sync()

    def resolution(self) -> tuple[int, int]:
        return self._width.value(), self._height.value()

    def settings(self) -> ProjectSettings:
        """選んだ内容を反映した設定 音声の設定などは渡されたものを引き継ぐ"""
        width, height = self.resolution()
        blending = str(self._blending.currentData())
        layers = str(self._layers.currentData()) if self._layers is not None else None
        return replace(
            self._base,
            width=width,
            height=height,
            frame_rate=self._frame_rate(),
            blending=blending,
            layer_mode=layers if layers is not None else self._base.layer_mode,
        )

    @property
    def saved_presets(self) -> tuple[ProjectPreset, ...]:
        return tuple(self._saved)

    def save_preset(self, name: str) -> ProjectPreset | None:
        """今の組み合わせを ``name`` で保存して、それを選んだ状態にする

        同じ名前があれば置き換える 名前が空、または縦横が奇数（書き出せない）なら断る
        """
        name = name.strip()
        width, height = self.resolution()
        if not name or width % 2 or height % 2:
            return None
        preset = ProjectPreset(
            name=name,
            width=width,
            height=height,
            frame_rate=self._frame_rate(),
            blending=str(self._blending.currentData()),
        )
        try:
            self._saved = self._store.put(preset)
        except PresetStoreError as exc:
            # 書き込めない置き場・容量不足など 例外のまま上げると、押したボタンが
            # 何も言わずに効かなかったように見える
            QMessageBox.warning(self, "組み合わせを保存", str(exc))
            return None
        self._tell_backup()
        self._fill_presets(select=_SAVED_KEY + name)
        self._sync()
        return preset

    def delete_preset(self, name: str) -> bool:
        """保存した組み合わせを消す 決まった組み合わせは消せない"""
        if all(preset.name != name for preset in self._saved):
            return False
        try:
            self._saved = self._store.remove(name)
        except PresetStoreError as exc:
            QMessageBox.warning(self, "組み合わせを削除", str(exc))
            return False
        self._tell_backup()
        # 消した物を選んでいた欄は、数字はそのままで、合う物か「指定する」へ移る
        self._fill_presets()
        self._sync()
        return True

    # --- 画面の部品 ---

    def _tell_backup(self) -> None:
        """壊れていた一覧を写してから作り直したなら、その写しの場所を知らせる

        黙って作り直すと、前に保存した物が一覧から消えた理由が分からない
        """
        backup = self._store.last_backup
        if backup is not None:
            QMessageBox.information(
                self,
                "テンプレートの一覧",
                f"一覧のファイルが壊れていたので作り直した 前の中身はここに残してある\n{backup}",
            )

    def _fill_presets(self, *, select: str | None = None) -> None:
        """選びの項目を並べ直す 決まった物、区切り、保存した物、「指定する」の順"""
        self._preset.blockSignals(True)
        self._preset.clear()
        for index, (label, _, _) in enumerate(RESOLUTION_PRESETS):
            self._preset.addItem(label, _BUILTIN_KEY + str(index))
        if self._saved:
            self._preset.insertSeparator(self._preset.count())
            for preset in self._saved:
                self._preset.addItem(_saved_label(preset), _SAVED_KEY + preset.name)
        self._preset.addItem(_CUSTOM, _CUSTOM_KEY)
        if select is not None:
            found = self._preset.findData(select)
            if found >= 0:
                self._preset.setCurrentIndex(found)
        self._preset.blockSignals(False)

    def _frame_rate(self) -> FrameRate:
        if self._rate is None:
            return self._base.frame_rate
        chosen = self._rate.currentData()
        return chosen if isinstance(chosen, FrameRate) else self._base.frame_rate

    def _rate_index(self, rate: FrameRate) -> int:
        """フレームレートの選びの中で ``rate`` の項目 無ければ -1"""
        if self._rate is None:
            return -1
        return next(
            (i for i in range(self._rate.count()) if self._rate.itemData(i) == rate),
            -1,
        )

    def _spin(self, value: int) -> QSpinBox:
        spin = QSpinBox(self)
        spin.setRange(MIN_RESOLUTION, MAX_RESOLUTION)
        spin.setSingleStep(2)
        spin.setValue(value)
        return spin

    def _saved_preset(self, key: object) -> ProjectPreset | None:
        if not isinstance(key, str) or not key.startswith(_SAVED_KEY):
            return None
        name = key.removeprefix(_SAVED_KEY)
        return next((preset for preset in self._saved if preset.name == name), None)

    def _on_preset(self, index: int) -> None:
        key = self._preset.itemData(index)
        saved = self._saved_preset(key)
        self._applying = True
        try:
            if saved is not None:
                self._width.setValue(saved.width)
                self._height.setValue(saved.height)
                self._blending.setCurrentIndex(self._blending.findData(saved.blending))
                # 保存した物のレートは組み立てのときに選びへ足してあるので、必ず見つかる
                if self._rate is not None and self._rate_index(saved.frame_rate) >= 0:
                    self._rate.setCurrentIndex(self._rate_index(saved.frame_rate))
            elif isinstance(key, str) and key.startswith(_BUILTIN_KEY):
                _, width, height = RESOLUTION_PRESETS[int(key.removeprefix(_BUILTIN_KEY))]
                self._width.setValue(width)
                self._height.setValue(height)
        finally:
            self._applying = False
        self._sync()

    def _ask_save(self) -> None:
        width, height = self.resolution()
        suggestion = f"{width}×{height} {self._frame_rate()}fps"
        name, accepted = QInputDialog.getText(self, "組み合わせを保存", "名前", text=suggestion)
        if not accepted or not name.strip():
            return
        if any(preset.name == name.strip() for preset in self._saved):
            answer = QMessageBox.question(
                self, "組み合わせを保存", f"「{name.strip()}」はもうあります 置き換えますか"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.save_preset(name)

    def _ask_delete(self) -> None:
        saved = self._saved_preset(self._preset.currentData())
        if saved is None:
            return
        answer = QMessageBox.question(
            self, "組み合わせを削除", f"保存した「{saved.name}」を一覧から消しますか"
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.delete_preset(saved.name)

    def _swap(self) -> None:
        width, height = self.resolution()
        self._width.setValue(height)
        self._height.setValue(width)

    def _matches(self, key: object) -> bool:
        """``key`` の項目が今の数字と同じ組み合わせか

        決まった物は縦横だけで見る（フレームレートと重ね合わせを持たない）
        保存した物は重ね合わせまで見る フレームレートは新規作成のときだけ見る
        作ったあとは変えられないので、見ると 60fps の物を選んだ途端に外れる
        """
        size = self.resolution()
        saved = self._saved_preset(key)
        if saved is not None:
            if (saved.width, saved.height) != size:
                return False
            if saved.blending != self._blending.currentData():
                return False
            return self._rate is None or saved.frame_rate == self._frame_rate()
        if isinstance(key, str) and key.startswith(_BUILTIN_KEY):
            _, width, height = RESOLUTION_PRESETS[int(key.removeprefix(_BUILTIN_KEY))]
            return (width, height) == size
        return False

    def _matching_index(self) -> int:
        """今の数字に合う項目 今選んでいる物が合っていればそのまま

        保存した物を決まった物より先に探す 同じ縦横の決まった物へ移ると、
        保存した物を選んだはずが別の名前に変わって見える
        """
        if self._matches(self._preset.currentData()):
            return self._preset.currentIndex()
        keys = [self._preset.itemData(i) for i in range(self._preset.count())]
        saved = [i for i, key in enumerate(keys) if self._saved_preset(key) is not None]
        builtin = [
            i for i, key in enumerate(keys) if isinstance(key, str) and key.startswith(_BUILTIN_KEY)
        ]
        for index in (*saved, *builtin):
            if self._matches(keys[index]):
                return index
        return self._preset.findData(_CUSTOM_KEY)

    def _sync(self) -> None:
        """選択肢の表示と、注意の文言と、押せるボタンを今の数字に合わせる"""
        if self._applying:
            return
        self._preset.blockSignals(True)
        self._preset.setCurrentIndex(self._matching_index())
        self._preset.blockSignals(False)

        notes: list[str] = []
        odd = any(value % 2 for value in self.resolution())
        if odd:
            notes.append("縦横とも偶数にしてください（奇数だと書き出せません）")
        saved = self._saved_preset(self._preset.currentData())
        if saved is not None and self._frame_rate() != saved.frame_rate:
            # 作ったあとのプロジェクトではフレームレートを変えない 黙って飛ばすと、
            # 60fps の組み合わせを選んだのに 30fps のままの理由が分からない
            notes.append(
                f"フレームレートは作成後は変えられないので、{saved.frame_rate} fps は使いません"
            )
        self._warning.setText("\n".join(notes))
        # 奇数は書き出しで断られる 閉じてからでは気付けないので、ここで押せなくする
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(not odd)
        self._save_button.setEnabled(not odd)
        self._delete_button.setEnabled(saved is not None)


def _saved_label(preset: ProjectPreset) -> str:
    """保存した物の表示名 名前だけだと、何が入っているのかを選ぶまで分からない"""
    blending = _BLENDING_SHORT.get(preset.blending, preset.blending)
    return (
        f"{SAVED_PREFIX}{preset.name}"
        f"（{preset.width}×{preset.height}・{preset.frame_rate} fps・{blending}）"
    )
