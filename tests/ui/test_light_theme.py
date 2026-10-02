"""明るいテーマ（表示 → 設定… の「画面の色」）

テーマを足すと、色を 1 か所で決めていない所だけが前のテーマのまま残る 残っても
例外にはならず、暗い地の白い字が明るい地の上に来て初めて「読めない」と気付く
色の数（文字と地の比）、描いた絵の画素、色を書いた所の数で確かめる

アプリ全体のスタイルシートを当てた試験は、終わったら暗いテーマへ戻して外す
外さないと、ほかの試験の窓まで見た目が変わる
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QLabel

from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.core.timebase import FrameRate
from sashimono.effects.sources import TEXT
from sashimono.engine.audio import PeakLevel, Waveform
from sashimono.engine.cache import MediaAnalyzer
from sashimono.resources import MAGNET_ICONS, MAGNET_ICONS_LIGHT, SPIN_ARROWS_LIGHT, path_to
from sashimono.ui.inspector.panel import _lock_label
from sashimono.ui.media_pool import MediaPoolWidget
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.scene_bar import SceneBar
from sashimono.ui.theme import (
    PALETTES,
    THEME_CHOICES,
    THEME_DARK,
    THEME_LIGHT,
    THEME_MODES,
    THEME_SYSTEM,
    Colors,
    apply_theme,
    current_theme,
    follow_system,
    magnet_icons,
    resolve_theme,
    theme_signals,
    themed_style,
    use_palette,
)
from sashimono.ui.timeline import TimelineView
from sashimono.ui.timeline.layout import TimelineLayout
from sashimono.ui.timeline.painter import TRACK_BUTTONS, _draw_waveform, clear_waveform_images
from sashimono.ui.transport import TransportBar
from sashimono.ui.workspace import Preferences, PreferenceStore

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "src" / "sashimono" / "ui"


@pytest.fixture(autouse=True)
def _back_to_dark(qt_application: QApplication) -> Iterator[None]:
    yield
    apply_theme(qt_application, THEME_DARK)
    qt_application.setStyleSheet("")


def _luminance(color: QColor) -> float:
    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(color.redF())
        + 0.7152 * channel(color.greenF())
        + 0.0722 * channel(color.blueF())
    )


def _over(top: QColor, base: QColor) -> QColor:
    """半透明の ``top`` を ``base`` の上に重ねた色"""
    alpha = top.alphaF()
    return QColor.fromRgbF(
        top.redF() * alpha + base.redF() * (1 - alpha),
        top.greenF() * alpha + base.greenF() * (1 - alpha),
        top.blueF() * alpha + base.blueF() * (1 - alpha),
    )


def _contrast(first: QColor, second: QColor) -> float:
    """2 色の明るさの比（WCAG の式） 文字は 4.5、線や印は 3 あれば見分けられる"""
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _colors(image: QImage, step: int = 2) -> set[str]:
    return {
        image.pixelColor(x, y).name()
        for x in range(0, image.width(), step)
        for y in range(0, image.height(), step)
    }


def _opaque(image: QImage) -> set[str]:
    return {
        image.pixelColor(x, y).name()
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 250
    }


class TestPalettes:
    def test_the_light_theme_names_every_color(self) -> None:
        # 足し忘れた色だけ、明るいテーマへ切り替えても暗いテーマの値のまま残る
        names = {name for name, value in vars(Colors).items() if isinstance(value, QColor)}
        names.discard("EDITING")
        assert set(PALETTES[THEME_LIGHT]) == names
        assert set(PALETTES[THEME_DARK]) == names

    def test_the_default_is_the_dark_theme(self) -> None:
        # 既定を変えると、版を上げただけで画面の色が黙って変わる
        assert Preferences().theme == THEME_DARK
        assert current_theme() == THEME_DARK
        assert Colors.WINDOW.name() == "#1b1b1e"
        assert THEME_CHOICES[0][0] == THEME_DARK

    @pytest.mark.parametrize("name", [THEME_DARK, THEME_LIGHT])
    def test_text_is_readable(self, name: str) -> None:
        # 4.5 を切ると、小さい字が地に溶けて読めない
        colors = PALETTES[name]
        pairs = [
            ("TEXT", ("WINDOW", "PANEL", "PANEL_ALT", "TRACK_HEADER", "VIEWER_BACKGROUND")),
            (
                "TEXT_MUTED",
                ("WINDOW", "PANEL", "TIMELINE_RULER", "TRACK_HEADER", "VIEWER_BACKGROUND"),
            ),
            ("ACCENT_TEXT", ("ACCENT",)),
            (
                "CLIP_LABEL",
                ("TAB_SELECTED", "TOOL_TIP", "VIDEO_CLIP", "AUDIO_CLIP", "FILTER_CLIP"),
            ),
            ("WARNING", ("WINDOW",)),
        ]
        for text, grounds in pairs:
            for ground in grounds:
                ratio = _contrast(colors[text], colors[ground])
                assert ratio >= 4.5, (name, text, ground, round(ratio, 2))
        for body in ("VIDEO_CLIP", "AUDIO_CLIP", "FILTER_CLIP"):
            band = _over(colors["CLIP_LABEL_SHADE"], colors[body])
            assert _contrast(colors["CLIP_LABEL"], band) >= 4.5, (name, body)

    @pytest.mark.parametrize("name", [THEME_DARK, THEME_LIGHT])
    def test_marks_stand_out(self, name: str) -> None:
        # 3 を切ると、再生ヘッドや磁石の線、波形が地に紛れて見落とす
        colors = PALETTES[name]
        ground = colors["TIMELINE_BACKGROUND"]
        for mark in ("PLAYHEAD", "SNAP_LINE", "SELECTION", "WORK_AREA_EDGE", "ACCENT"):
            assert _contrast(colors[mark], ground) >= 3, (name, mark)
        assert _contrast(colors["WAVEFORM"], colors["AUDIO_CLIP"]) >= 3
        assert _contrast(colors["VOLUME_LINE"], colors["AUDIO_CLIP"]) >= 3
        assert _contrast(colors["OPACITY_LINE"], colors["VIDEO_CLIP"]) >= 3
        for button in ("TRACK_MUTE", "TRACK_SOLO", "TRACK_LOCK"):
            # 1 文字の太字なので 3 で足りる
            assert _contrast(colors["TRACK_TOGGLE_TEXT"], colors[button]) >= 3, (name, button)
        for body in ("VIDEO_CLIP", "AUDIO_CLIP", "FILTER_CLIP"):
            for fill in ("KEYFRAME", "KEYFRAME_SELECTED"):
                # 塗りか縁のどちらかが地から浮いていれば、ひし形の形が読める
                shape = max(
                    _contrast(colors[fill], colors[body]),
                    _contrast(colors["KEYFRAME_OUTLINE"], colors[body]),
                )
                assert shape >= 3, (name, body, fill)


class TestSwitching:
    def test_colors_held_elsewhere_follow(self) -> None:
        # QColor を差し替えると、先に受け取って持っている所（トラックのボタンの表）だけが
        # 前のテーマの色で描き続ける
        held = TRACK_BUTTONS[0][3]
        use_palette(THEME_LIGHT)
        assert held.name() == PALETTES[THEME_LIGHT]["TRACK_MUTE"].name()
        assert Colors.EDITING is Colors.ACCENT
        assert Colors.EDITING.name() == PALETTES[THEME_LIGHT]["ACCENT"].name()
        use_palette(THEME_DARK)
        assert held.name() == PALETTES[THEME_DARK]["TRACK_MUTE"].name()
        assert Colors.WINDOW.name() == "#1b1b1e"

    def test_an_unknown_theme_is_dark(self) -> None:
        use_palette("purple")
        assert current_theme() == THEME_DARK
        assert resolve_theme("purple") == THEME_DARK

    def test_the_whole_app_switches_without_a_restart(self, qt_application: QApplication) -> None:
        # 再起動を求めると、試しに切り替えて見比べることができない
        label = QLabel("見本")
        themed_style(label, lambda: f"color: {Colors.TEXT_MUTED.name()};")
        try:
            assert apply_theme(qt_application, THEME_LIGHT) == THEME_LIGHT
            light = PALETTES[THEME_LIGHT]
            assert light["TEXT_MUTED"].name() in label.styleSheet()
            sheet = qt_application.styleSheet()
            assert light["WINDOW"].name() in sheet
            # 暗いテーマの白に近い矢印が残ると、数値欄の増減ボタンが見えない
            assert path_to(SPIN_ARROWS_LIGHT[0]).as_posix() in sheet
            assert magnet_icons() == MAGNET_ICONS_LIGHT
            apply_theme(qt_application, THEME_DARK)
            assert PALETTES[THEME_DARK]["TEXT_MUTED"].name() in label.styleSheet()
            assert magnet_icons() == MAGNET_ICONS
        finally:
            shiboken6.delete(label)

    def test_a_deleted_part_does_not_stop_the_switch(self, qt_application: QApplication) -> None:
        # 消えた部品へ当て直そうとして例外になると、そこで切り替えが止まって半分だけ変わる
        label = QLabel()
        themed_style(label, lambda: f"color: {Colors.TEXT.name()};")
        bar = TransportBar(FrameRate(30))
        shiboken6.delete(label)
        shiboken6.delete(bar)
        assert apply_theme(qt_application, THEME_LIGHT) == THEME_LIGHT

    def test_the_system_choice_follows_windows(self, qt_application: QApplication) -> None:
        # Windows を明るくしたのに付いてこないと、「合わせる」を選んだ意味が無い
        assert resolve_theme(THEME_SYSTEM, Qt.ColorScheme.Light) == THEME_LIGHT
        assert resolve_theme(THEME_SYSTEM, Qt.ColorScheme.Dark) == THEME_DARK
        # 読めない機械（試験の画面の無い Qt も）では、既定の暗い側
        assert resolve_theme(THEME_SYSTEM, Qt.ColorScheme.Unknown) == THEME_DARK
        apply_theme(qt_application, THEME_SYSTEM)
        follow_system(qt_application, Qt.ColorScheme.Light)
        assert current_theme() == THEME_LIGHT
        follow_system(qt_application, Qt.ColorScheme.Dark)
        assert current_theme() == THEME_DARK

    def test_a_fixed_choice_ignores_windows(self, qt_application: QApplication) -> None:
        # 暗いと決めた人の画面が、Windows の設定を変えただけで明るくなっては困る
        apply_theme(qt_application, THEME_DARK)
        follow_system(qt_application, Qt.ColorScheme.Light)
        assert current_theme() == THEME_DARK


@pytest.fixture
def analyzer(qt_application: QApplication) -> Iterator[MediaAnalyzer]:
    del qt_application
    created = MediaAnalyzer(sample_rate=48000, channels=2)
    yield created
    created.close()


def _timeline_image(view: TimelineView) -> QImage:
    image = QImage(view.size(), QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    view.render(painter, QPoint())
    painter.end()
    return image


class TestPaintedParts:
    """自前で描く所は描くたびに Colors を読む 切り替えて描き直せば新しい色になる"""

    def test_the_timeline_repaints_in_the_new_colors(self, analyzer: MediaAnalyzer) -> None:
        clip = Clip(timeline_start=0, duration=60, source=TEXT.create())
        base = Project.create()
        track = Track(TrackKind.VIDEO, "V1", (clip,))
        project = base.with_timeline(replace(base.timeline, tracks=(track,)))
        view = TimelineView(project, analyzer)
        try:
            view.resize(600, 160)
            view.zoom_to_fit()
            dark = _colors(_timeline_image(view))
            use_palette(THEME_LIGHT)
            light = _colors(_timeline_image(view))
        finally:
            shiboken6.delete(view)
        for token in ("TIMELINE_BACKGROUND", "TRACK_HEADER", "VIDEO_CLIP", "TIMELINE_RULER"):
            assert PALETTES[THEME_DARK][token].name() in dark, token
            assert PALETTES[THEME_LIGHT][token].name() in light, token
            assert PALETTES[THEME_DARK][token].name() not in light, token

    def test_the_waveform_is_not_served_from_the_old_theme(self) -> None:
        # 波形は画像に塗って貯めている 色を鍵に入れないと、切り替えても暗いテーマの
        # 明るい緑が明るい地の上に残る
        count = 48000 * 4 // 256
        peaks = np.empty((count, 2, 2), dtype=np.float32)
        peaks[:, :, 0] = -0.8
        peaks[:, :, 1] = 0.8
        waveform = Waveform(
            sample_rate=48000,
            channels=2,
            total_samples=48000 * 4,
            levels=(PeakLevel(256, peaks),),
        )
        clip = Clip(timeline_start=0, duration=90, source=TEXT.create())
        rect = QRect(0, 0, 300, 40)
        layout = TimelineLayout()
        clear_waveform_images()

        def paint() -> set[str]:
            canvas = QImage(320, 60, QImage.Format.Format_ARGB32)
            canvas.fill(0)
            painter = QPainter(canvas)
            _draw_waveform(painter, rect, clip, layout, FrameRate(30), waveform)
            painter.end()
            return _opaque(canvas)

        try:
            assert PALETTES[THEME_DARK]["WAVEFORM"].name() in paint()
            use_palette(THEME_LIGHT)
            light = paint()
            assert PALETTES[THEME_LIGHT]["WAVEFORM"].name() in light
            assert PALETTES[THEME_DARK]["WAVEFORM"].name() not in light
        finally:
            clear_waveform_images()

    def test_drawn_marks_are_redrawn(self, qt_application: QApplication) -> None:
        # 再生ボタンと鍵の印は文字の色で描いた絵 描き直さないと、暗いテーマの白に近い
        # 印が明るい地の上で見えなくなる
        del qt_application
        bar = TransportBar(FrameRate(30))
        lock = _lock_label()
        try:
            use_palette(THEME_LIGHT)
            theme_signals().changed.emit()
            text = PALETTES[THEME_LIGHT]["TEXT"].name()
            assert _opaque(bar._play.icon().pixmap(64, 64).toImage()) == {text}
            assert _opaque(bar._to_end.icon().pixmap(64, 64).toImage()) == {text}
            pixmap = lock.pixmap()
            assert _opaque(pixmap.toImage()) == {text}
        finally:
            shiboken6.delete(bar)
            shiboken6.delete(lock)

    def test_the_magnet_and_the_media_marks_follow(self, qt_application: QApplication) -> None:
        del qt_application
        scenes = SceneBar()
        pool = MediaPoolWidget(Project.create())
        try:
            before = pool._audio_icon
            use_palette(THEME_LIGHT)
            theme_signals().changed.emit()
            assert pool._audio_icon is not before
            accent = PALETTES[THEME_LIGHT]["ACCENT"].name()
            icon = scenes._snap_button.icon().pixmap(48, 48).toImage()
            assert accent in _opaque(icon)
            body = PALETTES[THEME_LIGHT]["AUDIO_CLIP"].name()
            assert body in _colors(pool._audio_icon.pixmap(128, 72).toImage(), 1)
        finally:
            shiboken6.delete(scenes)
            shiboken6.delete(pool)


#: 色をそのまま書いてよい所 地がテーマではなく別の物で決まる
_FIXED_COLORS = {
    # プレビューの絵の上に重ねる枠・吸着の線・範囲の印 地は映像なのでテーマで変えない
    # 暗い縁を敷いて、どんな映像の上でも見えるようにしてある
    "preview.py",
    # 色の見本のボタンの文字 地は選んだ色で、明るさで黒か白を選ぶ
    "inspector/widgets.py",
    # 値を動かしている間の吹き出し 自分で黒い地を敷いてから白い字を載せる
    "timeline/value_line.py",
}

_LITERAL = re.compile(r"""QColor\(\s*(["']#|\d)|["']#[0-9a-fA-F]{3,8}["']|color:\s*#""")


class TestNoColorsOutsideTheTheme:
    def test_the_ui_takes_its_colors_from_the_theme(self) -> None:
        # 色をその場で書くと、テーマを切り替えてもそこだけ前の色のまま残る
        # （キーフレームのひし形・仮の行・素材一覧の地・プロジェクト設定の注意が残っていた）
        found = []
        for path in sorted(UI.rglob("*.py")):
            name = path.relative_to(UI).as_posix()
            if name == "theme.py" or name in _FIXED_COLORS:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.lstrip().startswith("#") and _LITERAL.search(line):
                    found.append(f"{name}:{number}: {line.strip()}")
        assert found == []


class TestTheSetting:
    def test_it_comes_back(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(theme=THEME_LIGHT))
        assert store.load().theme == THEME_LIGHT

    def test_an_unknown_value_is_the_default(self, tmp_path: Path) -> None:
        # 新しい版で足した選び方を古い版で開いたとき、何も当たらない画面にしない
        path = tmp_path / "preferences.json"
        path.write_text(json.dumps({"theme": "purple", "use_proxy": False}), encoding="utf-8")
        loaded = PreferenceStore(path).load()
        assert loaded.theme == THEME_DARK
        assert loaded.use_proxy is False

    def test_the_dialog_offers_every_choice(self, qt_application: QApplication) -> None:
        del qt_application
        dialog = PreferencesDialog(Preferences(theme=THEME_LIGHT))
        try:
            box = dialog._theme
            assert [box.itemData(i) for i in range(box.count())] == list(THEME_MODES)
            assert box.currentData() == THEME_LIGHT
            assert dialog.preferences().theme == THEME_LIGHT
            box.setCurrentIndex(box.findData(THEME_SYSTEM))
            assert dialog.preferences().theme == THEME_SYSTEM
        finally:
            shiboken6.delete(dialog)

    def test_the_window_switches_only_when_it_changes(
        self, qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 当てるたびに全部の部品を描き直すので、ほかの設定を触っただけで当て直すと
        # 画面がちらつく 変えたのに当てないと、OK を押しても色が変わらない
        del qt_application
        from sashimono.ui import main_window
        from sashimono.ui.main_window import MainWindow

        applied: list[str] = []

        def record(_application: QApplication, mode: str) -> str:
            applied.append(mode)
            return mode

        monkeypatch.setattr(main_window, "apply_theme", record)
        window = MainWindow(Project.create(), confirm_unsaved=False)
        try:
            window._apply_preferences(Preferences(use_proxy=False))
            assert applied == []
            window._apply_preferences(Preferences(use_proxy=False, theme=THEME_LIGHT))
            assert applied == [THEME_LIGHT]
            window._apply_preferences(Preferences(use_proxy=True, theme=THEME_LIGHT))
            assert applied == [THEME_LIGHT]
        finally:
            window.close()


def test_the_light_arrows_and_magnets_are_shipped() -> None:
    # 配る版に積み忘れると、明るいテーマで数値欄の矢印と磁石の印が黙って消える
    from sashimono.resources import BUNDLED_FILES

    for name in (*SPIN_ARROWS_LIGHT, *MAGNET_ICONS_LIGHT):
        assert name in BUNDLED_FILES
        assert path_to(name).is_file()
