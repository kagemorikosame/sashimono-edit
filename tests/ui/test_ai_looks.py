"""AI のプリセット・エイリアス・棚の道具（#328）を本物の窓で動かす

窓が AI に渡す物（保存した見た目の写し・本人の当て方・棚）が、人の操作と同じ置き場と設定を
見ているかを確かめる 道具の決まりそのものは ``tests/ai/test_look_tools.py`` が偽のホストで見る

各試験の頭の 1 行は、壊れたときに利用者に何が起きるか
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from sashimono.ai.operations import find_operation
from sashimono.compat.catalog import TemplateCatalog, TemplateEntry
from sashimono.compat.looks import capture_preset
from sashimono.compat.subtitle_looks import template_choice
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import AnimatedValue, Clip, Project, Track, TrackKind
from sashimono.effects import registry
from sashimono.effects.sources import TEXT
from sashimono.ui.main_window import MainWindow
from tests.ai.test_look_tools import _EXO_TWO_LAYERS, _SUBTITLE_OBJECT
from tests.compat.test_subtitle_looks import _styled_alias


@pytest.fixture(autouse=True)
def no_popups(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 道具は人に何も尋ねないはず（取り消せる編集だけ）"""

    def popped(*_args: object, **_kwargs: object) -> object:
        pytest.fail("試験の途中で窓が開いた")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, popped)
    monkeypatch.setattr(QDialog, "exec", popped)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", popped)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", popped)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", popped)


def _text(text: str, start: int = 0) -> Clip:
    return with_fixed_items(
        Clip(timeline_start=start, duration=90, source=TEXT.create(text=text)), picture=True
    )


@pytest.fixture
def shelf_entry(tmp_path: Path) -> TemplateEntry:
    (tmp_path / "棚" / "字幕").mkdir(parents=True)
    (tmp_path / "棚" / "字幕" / "大きい.object").write_text(_SUBTITLE_OBJECT, encoding="utf-8")
    (entry,) = TemplateCatalog().scan((tmp_path / "棚",))
    return entry


@pytest.fixture
def window(qt_application: QApplication, shelf_entry: TemplateEntry) -> Iterator[MainWindow]:
    del qt_application
    base = Project.create()
    track = Track(TrackKind.VIDEO, "V1", (_text("一つ目"), _text("二つ目", start=100)))
    created = MainWindow(
        base.with_timeline(replace(base.timeline, tracks=(track,))), confirm_unsaved=False
    )
    # 棚は試験の一時フォルダの物だけ 既定のままだと本人の AviUtl2 の置き場を読む
    created._timeline.add_sources.templates = lambda: (shelf_entry,)
    yield created
    created.close()
    created.deleteLater()


def _run(window: MainWindow, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"そんなツールは無い: {tool}"
    return operation(window, arguments)


def _clips(window: MainWindow) -> tuple[Clip, ...]:
    return window.document.project.timeline.tracks[0].clips


def _save_yellow(window: MainWindow) -> None:
    look = with_fixed_items(
        Clip(
            timeline_start=0,
            duration=90,
            source=TEXT.create(text="保存した文字", color=(1.0, 1.0, 0.0, 1.0)),
            effects=(registry.require("blur").create(radius=4),),
        ),
        picture=True,
    )
    preset = capture_preset("黄色", None, Track(TrackKind.VIDEO, "V", (look,)), look)
    window.look_library().presets.save(preset)


class TestTheWindowHandsTheSameThings:
    def test_the_shelf_is_the_one_the_timeline_offers(
        self, window: MainWindow, shelf_entry: TemplateEntry
    ) -> None:
        # 壊れると、右クリックの〔追加〕に並ぶ棚と、AI が読む棚が食い違う
        assert window.template_shelf == (shelf_entry,)
        looks = _run(window, "list_looks", kind="template")["looks"]
        assert [row["template_id"] for row in looks] == [template_choice(shelf_entry).key]

    def test_saved_looks_are_read_from_the_same_library(self, window: MainWindow) -> None:
        # 壊れると、人が保存したプリセットとエイリアスが AI の一覧に出ない
        _save_yellow(window)
        window.look_library().aliases.save(_styled_alias("テロップ"))
        names = {(row["kind"], row["name"]) for row in _run(window, "list_looks")["looks"]}
        assert {("preset", "黄色"), ("alias", "テロップ")} <= names

    def test_a_preset_follows_the_person_s_setting(self, window: MainWindow) -> None:
        # 壊れると、設定で「文字も当てる」にしている人の所で、AI だけ文字を当てない
        _save_yellow(window)
        window._preferences = replace(window._preferences, preset_with_text=True)
        first = _clips(window)[0]
        _run(window, "apply_preset", name="黄色", clip_ids=[str(first.id)])
        source = _clips(window)[0].source
        assert source is not None and source.params["text"] == "保存した文字"
        window.document.undo()
        assert _clips(window)[0] == first


class TestPlacingThroughTheWindow:
    def test_a_template_and_an_alias_each_undo_in_one(
        self, window: MainWindow, shelf_entry: TemplateEntry
    ) -> None:
        # 壊れると、AI が窓へ置いた物を戻すのに何回も押すか、窓が確かめの箱を出して止まる
        before = window.document.project
        window.look_library().aliases.save(_styled_alias("テロップ"))
        _run(window, "place_alias", name="テロップ", at_frame=300)
        window.document.undo()
        assert window.document.project == before
        key = template_choice(shelf_entry).key
        result = _run(window, "place_template", template_id=key, at_frame=300)
        assert result["placed"] == 1
        window.document.undo()
        assert window.document.project == before

    def test_a_template_is_worn_like_the_shelf_dialog(
        self, window: MainWindow, shelf_entry: TemplateEntry
    ) -> None:
        # 壊れると、棚の〔選択中のクリップに適用〕と AI の着せ方が食い違う
        first = _clips(window)[0]
        key = template_choice(shelf_entry).key
        _run(window, "apply_template", template_id=key, clip_id=str(first.id))
        worn = _clips(window)[0]
        assert worn.source is not None
        assert worn.source.params["text"] == "一つ目"
        assert worn.source.params["size"] == AnimatedValue(88.0)


def _two_layers(tmp_path: Path) -> TemplateEntry:
    """レイヤー 1 に小さい字、レイヤー 2 に大きい字の ``.exo`` を、棚の 1 本として読む"""
    folder = tmp_path / "二段"
    folder.mkdir()
    (folder / "二段.exo").write_text(_EXO_TWO_LAYERS, encoding="utf-8")
    (entry,) = TemplateCatalog().scan((folder,))
    return entry


def _new_tracks(window: MainWindow, before: Project) -> list[list[float]]:
    """足したトラックごとの、置いたテキストの大きさ 並び（奥から手前）の順"""
    old = {track.id for track in before.timeline.tracks}
    found: list[list[float]] = []
    for track in window.document.project.timeline.tracks:
        if track.id in old:
            assert track == before.timeline.find_track(track.id), "前からあるトラックが変わった"
            continue
        sizes: list[float] = []
        for clip in track.clips:
            assert clip.source is not None
            size = clip.source.params["size"]
            assert isinstance(size, AnimatedValue)
            sizes.append(size.static)
        found.append(sizes)
    return found


class TestCrowdedLayers:
    """元のレイヤーの所が埋まっているとき（2026-10-17 本人の決定 一番手前の新しいトラックへ）"""

    def test_the_shelf_puts_everything_in_front(self, window: MainWindow, tmp_path: Path) -> None:
        # 壊れると、字幕のある所へ棚のテンプレートを置くだけで、重なって断られる
        entry = _two_layers(tmp_path)
        before = window.document.project
        assert window.place_template_entry(entry, 30)
        assert _new_tracks(window, before) == [[40.0], [80.0]]
        window.document.undo()
        assert window.document.project == before

    def test_reading_objects_puts_everything_in_front(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、クリップのある作品へ〔オブジェクトを読み込む…〕で読むと、重なって断られる
        source = _two_layers(tmp_path).path
        # ファイルを選ぶ窓だけは選んだことにする（ほかの窓が開けば落ちる）
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName", staticmethod(lambda *_a, **_k: (str(source), ""))
        )
        before = window.document.project
        window.import_exo()
        assert _new_tracks(window, before) == [[40.0], [80.0]]
        window.document.undo()
        assert window.document.project == before

    def test_the_ai_places_the_same_way(self, window: MainWindow, tmp_path: Path) -> None:
        # 壊れると、人が棚から置いた並びと、AI が置いた並びが食い違う
        entry = _two_layers(tmp_path)
        window._timeline.add_sources.templates = lambda: (entry,)
        before = window.document.project
        _run(window, "place_template", template_id=template_choice(entry).key, at_frame=30)
        assert _new_tracks(window, before) == [[40.0], [80.0]]
        window.document.undo()
        _run(window, "import_objects", path=str(entry.path))
        assert _new_tracks(window, before) == [[40.0], [80.0]]
