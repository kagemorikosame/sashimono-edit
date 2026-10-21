"""素材を置いて書き出したとき、動画の終わりに黒いコマが出ないこと（#402）

音の方が長い素材は、コンテナの長さ（音の終わり）までクリップが伸びる 絵を最後の
コマで止めない（:attr:`~sashimono.core.model.Clip.hold_at` が無い）と、映像が終わった
後ろが真っ黒で書き出される 音が少しだけ長い素材では、切り上げた最後の 1 コマだけが黒い

置く入口は画面の読み込み（:meth:`MainWindow.import_media`）と AI の道具（``import_media``・
``place_media``）の 2 つで、どちらも :func:`~sashimono.core.commands.insert_media` を通る
入口ごとに確かめるのは、片方だけ別の置き方へ変わったときに気付くため

書き出しは GPU が要るので CI では飛ぶ GPU の無い所でも、置いたクリップの最後のコマを
デコーダが返すか（黒の元になる「何も返さない」にならないか）で同じ壊れ方を拾う
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import pytest
from PySide6.QtWidgets import QApplication

from sashimono.ai.operations import find_operation
from sashimono.core.io.serialize import _media_from_json, _media_to_json
from sashimono.core.model import Clip, LayerMode, MediaItem, Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import VideoDecoder, probe_media
from sashimono.engine.encode import ExportSettings, export_project
from sashimono.ui.main_window import MainWindow
from tests.ai.conftest import FakeHost
from tests.check_media import FPS, HEIGHT, WIDTH, write_video

#: 素材の作り方 名前 → （映像の秒, 音の秒）
#: ``longer`` は #402 で見つかった 0.4 秒ほど音が長い素材 ``barely`` は音がコマ 1 つ分に
#: 満たないだけ長い素材（切り上げで 1 コマ増え、そこだけ黒くなる） ``even`` は長さが
#: そろった素材で、止めなくてよいのに止めた印を付けていないかを見る
_SHAPES = {"longer": (2.0, 2.4), "barely": (2.0, 2.02), "even": (2.0, 2.0)}

#: 黒と見なす明るさの平均 模様の明るさは 60〜230 なので、平均が 5 を割るのは黒だけ
_BLACK = 5.0


@pytest.fixture(scope="module")
def materials(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    directory = tmp_path_factory.mktemp("tail")
    return {
        name: write_video(directory / f"{name}.mp4", video, tone_db=-20, audio_seconds=audio)
        for name, (video, audio) in _SHAPES.items()
    }


def _settings(mode: str) -> ProjectSettings:
    return ProjectSettings(width=WIDTH, height=HEIGHT, frame_rate=FrameRate(FPS), layer_mode=mode)


class _Host(FakeHost):
    """素材は本当に調べる（ほかは偽物のまま）"""

    def probe(self, path: Path) -> MediaItem:
        self.probed.append(path)
        return probe_media(path)


def _call(host: FakeHost, tool: str, /, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None, f"{tool} という道具が無い"
    return operation(host, arguments)


def _by_ai(path: Path, mode: str) -> Project:
    """AI の道具で読み込んで置いた作品"""
    host = _Host(Project.create(_settings(mode)))
    _call(host, "import_media", paths=[str(path)])
    return host.document.project


def _by_ai_placing(path: Path, mode: str) -> Project:
    """AI の道具で、素材一覧にある素材をもう 1 本後ろへ置いた作品"""
    host = _Host(Project.create(_settings(mode)))
    _call(host, "import_media", paths=[str(path)])
    media = host.document.project.media[0]
    _call(host, "place_media", media_id=str(media.id))
    return host.document.project


@pytest.fixture
def window(qt_application: QApplication, request: pytest.FixtureRequest) -> Iterator[MainWindow]:
    # 方式は作品を作るときに決まるので、試験の mode ごとに窓を作る
    del qt_application
    mode = request.node.callspec.params["mode"]
    created = MainWindow(Project.create(_settings(mode)), confirm_unsaved=False)
    yield created
    created.close()


def _by_window(window: MainWindow, path: Path) -> Project:
    """画面の読み込みで置いた作品"""
    window.import_media([path])
    assert window.wait_for_imports()
    return window._document.project


def _pictures(project: Project) -> list[tuple[Clip, MediaItem]]:
    found = []
    for track in project.timeline.tracks:
        for clip in track.clips:
            if clip.media_id is None or not project.draws_picture(track, clip):
                continue
            media = project.find_media(clip.media_id)
            assert media is not None
            found.append((clip, media))
    assert found, "絵を出すクリップが置かれていない"
    return found


def _assert_tail_decodes(project: Project) -> None:
    """絵を出すクリップの最後のコマを、デコーダが返すこと"""
    rate = project.rate
    for clip, media in _pictures(project):
        last = clip.picture_time(clip.duration - 1, rate)
        decoder = VideoDecoder(media.path, clip.stream_index)
        try:
            frame = decoder.frame_at(last)
        finally:
            decoder.close()
        assert frame is not None, (
            f"{media.path.name} を置いたクリップの最後のコマ（素材の {float(last):.3f} 秒）に"
            "絵が無く、書き出すと黒くなる"
        )


def _assert_export_has_no_black(project: Project, path: Path) -> None:
    # エンコーダは libx264 に決める 選ばせると機械ごとに NVENC などへ変わり、試験の結果が
    # GPU のエンコーダの具合に左右される 入っていない機械では書き出せないので、落とさず飛ばす
    if "libx264" not in av.codecs_available:
        pytest.skip("libx264 が入っていない")
    export_project(project, ExportSettings(path=path, video_codec="libx264"))
    with av.open(str(path)) as container:
        means = [float(f.to_ndarray(format="rgb24").mean()) for f in container.decode(video=0)]
    assert len(means) == project.duration
    black = [index for index, mean in enumerate(means) if mean < _BLACK]
    assert not black, f"書き出した {len(means)} コマのうち {black} が黒い"


MODES = [LayerMode.SEPARATED, LayerMode.MIXED]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_placed_by_the_ai_keeps_its_last_picture(
    materials: dict[str, Path], shape: str, mode: str
) -> None:
    """AI が読み込んで置いた素材の最後のコマに絵があること

    止めないと、音の方が長い素材の終わりがデコーダから何も返らず、書き出すと黒くなる
    （#402 で 0.44 秒と最後の 1 コマが黒く書き出された）
    """
    _assert_tail_decodes(_by_ai(materials[shape], mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_placed_again_by_the_ai_keeps_its_last_picture(
    materials: dict[str, Path], shape: str, mode: str
) -> None:
    """AI が素材一覧から置き直した素材も、最後のコマに絵があること

    ``place_media`` は読み込みと別の道具 片方の道具だけ止める印を付け忘れると、置き直した
    方の終わりが黒く書き出される
    """
    _assert_tail_decodes(_by_ai_placing(materials[shape], mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_imported_on_screen_keeps_its_last_picture(
    window: MainWindow, materials: dict[str, Path], shape: str, mode: str
) -> None:
    """画面で読み込んだ素材の最後のコマに絵があること

    利用者が読み込みのボタンやドロップで置いた素材の終わりが、書き出すと黒くなる
    """
    _assert_tail_decodes(_by_window(window, materials[shape]))


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_matching_lengths_are_not_held(materials: dict[str, Path], shape: str) -> None:
    """映像の終わりを越えないクリップには、絵を止める印を付けないこと

    越えないのに付けると、設定パネルに「絵を止める」と解除のボタンが出て、何を止めて
    いるのか分からなくなる 越えるのに付けないと、終わりが黒く書き出される
    """
    project = _by_ai(materials[shape], LayerMode.SEPARATED)
    for clip, media in _pictures(project):
        end = media.video_streams[0].end_time
        assert end is not None
        runs_over = clip.source_out(project.rate) > end
        assert (clip.hold_at is not None) == runs_over, (
            f"{shape}: 映像の終わり {float(end):.3f} 秒 クリップの終わり"
            f" {float(clip.source_out(project.rate)):.3f} 秒 止める時刻 {clip.hold_at}"
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["longer", "barely"])
@pytest.mark.usefixtures("gpu")
def test_export_after_the_ai_places_has_no_black_frame(
    materials: dict[str, Path], tmp_path: Path, shape: str, mode: str
) -> None:
    """AI が置いた素材を書き出した動画に、黒いコマが無いこと

    #402 の見え方そのもの 書き出しが範囲の終わりを 1 コマ多く数えても、止めた絵を
    読まなくても、動画の終わりが一瞬黒くなる
    """
    _assert_export_has_no_black(_by_ai(materials[shape], mode), tmp_path / "out.mp4")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["longer", "barely"])
@pytest.mark.usefixtures("gpu")
def test_export_after_the_window_imports_has_no_black_frame(
    window: MainWindow, materials: dict[str, Path], tmp_path: Path, shape: str, mode: str
) -> None:
    """画面で読み込んだ素材を書き出した動画に、黒いコマが無いこと

    利用者が置いて書き出しただけの動画の終わりが、一瞬黒くなる
    """
    project = _by_window(window, materials[shape])
    _assert_export_has_no_black(project, tmp_path / "out.mp4")


def test_the_sound_is_not_cut(materials: dict[str, Path]) -> None:
    """音の方が長い素材を置いたとき、音のクリップは音の終わりまで残ること

    絵に合わせて切ると、話し声やBGMの終わりが黙って欠ける 止めるのは絵だけ
    """
    project = _by_ai(materials["longer"], LayerMode.SEPARATED)
    sounds = [c for t in project.timeline.audio_tracks() for c in t.clips]
    assert sounds
    assert all(clip.source_out(project.rate) >= Fraction(12, 5) for clip in sounds)


def _old_media(path: Path) -> MediaItem:
    """形式 4 の作品から読んだ素材 映像の終わり（``end_time``）を持たない

    4 までの値は数え方が違うので読み捨てる（``_media_from_json``） 長さはコンテナの長さ
    （音の終わり）なので、これだけ見て止めるかを決めると、音の方が長い素材を止めない
    """
    media = _media_from_json(_media_to_json(probe_media(path)), 4)
    assert media.video_streams[0].end_time is None, "古い作品の素材に映像の終わりが残っている"
    return media


def _old_project(path: Path, mode: str) -> tuple[Project, MediaItem]:
    """古い作品を開いた所 素材一覧にだけ素材があり、タイムラインは空"""
    media = _old_media(path)
    return Project.create(_settings(mode)).with_media((media,)), media


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["longer", "barely"])
def test_the_ai_placing_from_an_old_project_keeps_its_last_picture(
    materials: dict[str, Path], shape: str, mode: str
) -> None:
    """古い作品の素材一覧から AI が置いた素材も、最後のコマに絵があること

    古い作品の素材は映像の終わりを持たない 長さ（音の終わり）だけで止めるかを決めると
    止めず、置き直した素材の終わりが黒く書き出される（#404 の Codex の指摘）
    """
    project, media = _old_project(materials[shape], mode)
    host = _Host(project)
    _call(host, "place_media", media_id=str(media.id))
    _assert_tail_decodes(host.document.project)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["longer", "barely"])
@pytest.mark.parametrize("place", ["button", "drop"])
def test_placing_from_an_old_project_on_screen_keeps_its_last_picture(
    qt_application: QApplication, materials: dict[str, Path], shape: str, mode: str, place: str
) -> None:
    """古い作品の素材一覧から画面で置いた素材も、最後のコマに絵があること

    一覧の「配置」とタイムラインへのドラッグは別の入口 片方だけ映像の終わりを取り直し
    忘れると、そちらで置いた素材の終わりが黒く書き出される
    """
    del qt_application
    project, media = _old_project(materials[shape], mode)
    window = MainWindow(project, confirm_unsaved=False)
    try:
        if place == "button":
            window._insert_media_by_id(str(media.id))
        else:
            window._on_media_dropped([str(media.id)], 0, "")
        placed = window._document.project
        _assert_tail_decodes(placed)
        # 素材一覧の素材は書き換えない 書き換えるなら取り消せる形にする必要があり、
        # 置いただけで一覧の中身が変わるのは本人の予想を外れる
        assert placed.media == (media,), "置いたら素材一覧が書き換わった"
    finally:
        window.close()
