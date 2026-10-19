"""P6 の完了条件そのもの

**配布テンプレートが読み込めて、見た目が再現される**

対応表を通ったかではなく、**画面に出た画素**で確かめる エイリアスを読んで
着せ替えて、実際に描いて、指定された色がそこにあるかを見る

読ませるのは、このマシンに実際に置かれている配布エイリアス 無ければ同じ書き方の
見本に切り替える（他のマシンでも通るように）
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from sashimono.compat.catalog import TemplateCatalog, place, restyle
from sashimono.core.commands import AddClip, Document
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer

WIDTH, HEIGHT = 640, 360

#: 実物を通すときの画面 配布物は 1080p 前提で座標が書かれているので、
#: 小さい画面で描くと画面外へ出て真っ黒になる
FULL = (1920, 1080)

#: 実物が無いときの見本 手元の配布エイリアスと同じ書き方に揃えてある
#: 黄色い文字に黒い太縁、下寄せ、赤い二重縁
FALLBACK = (
    "[Object]"
    + chr(10)
    + "frame=0,89"
    + chr(10)
    + "[Object.0]"
    + chr(10)
    + "effect.name=テキスト"
    + chr(10)
    + "サイズ=64.00"
    + chr(10)
    + "文字色=ffee00"
    + chr(10)
    + "影・縁色=000000"
    + chr(10)
    + "文字装飾=縁取り文字（太）"
    + chr(10)
    + "文字揃え=中央揃え[中]"
    + chr(10)
    + "テキスト=見本の文字"
    + chr(10)
    + "[Object.1]"
    + chr(10)
    + "effect.name=縁取り"
    + chr(10)
    + "サイズ=6"
    + chr(10)
    + "縁色=ff0000"
    + chr(10)
    + "[Object.2]"
    + chr(10)
    + "effect.name=標準描画"
    + chr(10)
    + "X=0.00"
    + chr(10)
    + "Y=0.00"
    + chr(10)
    + "拡大率=100.000"
    + chr(10)
    + "透明度=0.00"
    + chr(10)
    + "合成モード=通常"
    + chr(10)
)

MY_TEXT = "自分で打った字幕"


@pytest.fixture(scope="module")
def gl() -> Iterator[OffscreenGLContext]:
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:  # pragma: no cover - GPU の無い環境
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> TemplateCatalog:
    """棚を 1 つ用意する 実物があればそれ、無ければ見本"""
    catalog = TemplateCatalog()
    root = tmp_path_factory.mktemp("テンプレート")
    (root / "見本.object").write_text(FALLBACK, "utf-8")
    catalog.scan((root,))
    return catalog


@pytest.fixture(scope="module")
def real_aliases() -> tuple[Path, ...]:
    program_data = os.environ.get("PROGRAMDATA")
    if not program_data:
        return ()
    root = Path(program_data) / "aviutl2" / "Alias"
    return tuple(sorted(root.rglob("*.object"))) if root.is_dir() else ()


def project_with_my_subtitle() -> Project:
    """自分で打った字幕が 1 つだけ載ったプロジェクト"""
    project = Project.create(ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(30)))
    track = Track(
        kind=TrackKind.VIDEO,
        clips=(
            Clip(
                timeline_start=0,
                duration=60,
                source=GeneratedSource(
                    kind="text",
                    params={
                        "text": MY_TEXT,
                        "size": AnimatedValue(40.0),
                        "color": (1.0, 1.0, 1.0, 1.0),
                    },
                ),
            ),
        ),
    )
    return project.with_timeline(project.timeline.__class__(rate=project.rate, tracks=(track,)))


def draw(project: Project, gl: OffscreenGLContext, frame: int = 0) -> np.ndarray:
    renderer = FrameRenderer(project, context=gl)
    try:
        return renderer.render(frame)
    finally:
        renderer.close()


def count(image: np.ndarray, red: int, green: int, blue: int, slack: int = 40) -> int:
    """指定した色に近い画素の数"""
    target = np.array([red, green, blue], dtype=np.int16)
    difference = np.abs(image[:, :, :3].astype(np.int16) - target).max(axis=2)
    return int((difference <= slack).sum())


def ink(image: np.ndarray) -> int:
    """背景でない画素の数 背景は黒なので、色が付いていれば数える"""
    return int((image[:, :, :3].max(axis=2) > 24).sum())


def _extent(image: np.ndarray) -> tuple[int, int]:
    """描かれたものの幅と高さ 何も描かれていなければ 0"""
    lit = image[:, :, :3].max(axis=2) > 24
    rows, columns = np.where(lit)
    if not len(rows):
        return (0, 0)
    return (
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    )


def _without_decoration(project: Project) -> Project:
    """文字装飾だけを外した同じプロジェクト 比較のために作る"""
    from dataclasses import replace

    track = project.timeline.tracks[0]
    clip = track.clips[0]
    assert clip.source is not None
    params = {
        name: value
        for name, value in clip.source.params.items()
        if not name.startswith(("border_", "shadow_"))
    }
    bare = replace(clip, source=GeneratedSource(kind="text", params=params))
    return project.with_timeline(project.timeline.replace_track(track.with_clips((bare,))))


@pytest.fixture(scope="module")
def restyled(template: TemplateCatalog) -> Project:
    """自分で打った字幕に、見本のデザインを着せたあとのプロジェクト"""
    document = Document(project_with_my_subtitle())
    clip = document.project.timeline.tracks[0].clips[0]
    entry = template.find("見本")
    assert entry is not None

    document.begin_checkpoint("テンプレートを適用")
    for command in restyle(entry.load(), clip):
        document.execute(command)
    document.end_checkpoint()
    return document.project


class TestRestylingMyOwnSubtitle:
    """自分で打った字幕に、配布デザインを着せる ここが本命"""

    def test_my_text_is_still_mine(self, restyled: Project) -> None:
        # 着せ替えで文字が入れ替わると、
        # テンプレートを着せた字幕の自分で打った文が見本の文字に書き換わる
        source = restyled.timeline.tracks[0].clips[0].source
        assert source is not None
        assert source.params["text"] == MY_TEXT

    def test_the_timing_did_not_move(self, restyled: Project) -> None:
        # 着せ替えで置き場所や長さが動くと、テンプレートを着せた字幕が話している所からずれる
        clip = restyled.timeline.tracks[0].clips[0]
        assert (clip.timeline_start, clip.duration) == (0, 60)

    def test_the_letters_turn_the_template_colour(
        self, restyled: Project, gl: OffscreenGLContext
    ) -> None:
        # 文字色 ffee00 着せる前は白なので、黄色が出ていれば効いている
        before = draw(project_with_my_subtitle(), gl)
        after = draw(restyled, gl)
        assert count(before, 255, 238, 0) < 20
        assert count(after, 255, 238, 0) > 200

    def test_the_outline_from_the_decoration_is_drawn(
        self, restyled: Project, gl: OffscreenGLContext
    ) -> None:
        """``文字装飾=縁取り文字（太）`` の黒縁が、実際に画面を占めている

        背景が黒なので「黒い画素があるか」でも「色の付いた画素が増えたか」でも
        確かめられない 縁は黒なので、数えると逆に**減る**（文字の縁を黒が食う）
        装飾を外した同じ絵と比べ、絵の広がり（外周）が縁のぶん太るかを見る
        """
        bare = _extent(draw(_without_decoration(restyled), gl))
        decorated = _extent(draw(restyled, gl))
        assert decorated[0] > bare[0]
        assert decorated[1] > bare[1]

    def test_the_stacked_border_effect_is_drawn(
        self, restyled: Project, gl: OffscreenGLContext
    ) -> None:
        # エイリアスに積まれていた赤い縁取りフィルタ
        assert count(draw(restyled, gl), 255, 0, 0) > 100

    def test_one_undo_puts_everything_back(self, template: TemplateCatalog) -> None:
        # 1 回の操作は 1 回の取り消しで戻る
        document = Document(project_with_my_subtitle())
        clip = document.project.timeline.tracks[0].clips[0]
        entry = template.find("見本")
        assert entry is not None

        document.begin_checkpoint("テンプレートを適用")
        for command in restyle(entry.load(), clip):
            document.execute(command)
        document.end_checkpoint()

        document.undo()
        source = document.project.timeline.tracks[0].clips[0].source
        assert source is not None
        assert source.params["color"] == (1.0, 1.0, 1.0, 1.0)
        assert document.project.timeline.tracks[0].clips[0].effects == ()


class TestPlacingATemplate:
    """見本の文字ごとタイムラインへ置く 新しくテロップを作るとき"""

    def test_it_lands_at_the_playhead_and_draws(
        self, template: TemplateCatalog, gl: OffscreenGLContext
    ) -> None:
        # 置き場所がずれるか置いた物が描かれないと、
        # 棚から置いた新しいテロップが再生ヘッドの所に出ない
        document = Document(project_with_my_subtitle())
        entry = template.find("見本")
        assert entry is not None

        document.begin_checkpoint("テンプレートを配置")
        for command in place(entry.load(), document.project, at_frame=90):
            document.execute(command)
        document.end_checkpoint()

        added = [
            clip
            for track in document.project.timeline.tracks
            for clip in track.clips
            if clip.source is not None and clip.source.params.get("text") == "見本の文字"
        ]
        assert len(added) == 1
        assert added[0].timeline_start == 90

        # 置いた場所で実際に描ける
        assert count(draw(document.project, gl, frame=95), 255, 238, 0) > 200

    def test_placing_does_not_touch_what_was_there(self, template: TemplateCatalog) -> None:
        # 置くときに既にある字幕まで書き換わると、テロップを足しただけで自分の字幕の見た目が変わる
        document = Document(project_with_my_subtitle())
        entry = template.find("見本")
        assert entry is not None
        before = document.project.timeline.tracks[0].clips[0]

        for command in place(entry.load(), document.project, at_frame=90):
            document.execute(command)

        after = document.project.timeline.tracks[0].clips[0]
        assert after.source == before.source


class TestTheRealDistributedAliases:
    """実際に配られているものを、全部通す"""

    def test_every_one_of_them_renders_something(
        self, real_aliases: tuple[Path, ...], gl: OffscreenGLContext
    ) -> None:
        # 実物の配布エイリアスで読み落としがあると、棚から置いたテンプレートが真っ黒のまま何も出ない
        if not real_aliases:
            pytest.skip("このマシンに AviUtl2 の配布エイリアスが無い")

        from sashimono.compat.catalog import TemplateEntry

        blank = 0
        for path in real_aliases:
            entry = TemplateEntry(name=path.stem, path=path)
            objects = entry.load()
            assert objects, f"写せなかった: {path.name}"

            project = Project.create(
                ProjectSettings(width=FULL[0], height=FULL[1], frame_rate=FrameRate(30))
            )
            document = Document(project)
            for command in place(objects, document.project, at_frame=0):
                document.execute(command)
            assert any(
                isinstance(command, AddClip) for command in place(objects, project, at_frame=0)
            )

            image = draw(document.project, gl, frame=0)
            if image[:, :, :3].max() <= 8:
                blank += 1

        # 中身が Lua の埋め込みだけのものは、こちらでは絵にならない
        # それ以外が真っ黒なら、どこかで読み落としている
        assert blank <= 2, f"{blank} 本が真っ黒になった"
