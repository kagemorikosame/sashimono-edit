"""コマンドが断られたことを、呼び出し側が知れるか

テンプレートの配置は素材の登録と配置を 1 回の Undo にまとめる 途中で断られると
全部戻るので、戻った後に「置いた」と出したり、戻した素材の解析を頼んだりしてはいけない
``.exo`` の読み込みと素材の読み込み、生成オブジェクトの追加も同じ道を通る
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

from sashimono.compat.mapped import MappedObject
from sashimono.core.commands import AddClip, AddMedia, insert_generated
from sashimono.core.model import Clip, GeneratedSource, MediaItem, Project, TrackId
from sashimono.ui import main_window as main_window_module
from sashimono.ui import template_dialog
from sashimono.ui.main_window import MainWindow


@pytest.fixture
def window(qt_application: QApplication) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(Project.create(), confirm_unsaved=False)
    yield created
    created.close()


def test_a_refused_batch_reports_failure_and_rolls_back(window: MainWindow) -> None:
    # 無いトラックへ置くコマンドで断らせる 先に足した素材も一緒に戻る
    media = MediaItem(path=Path("C:/素材/効果音.mp3"))
    refused = window.execute_all(
        [AddMedia(media), AddClip(TrackId("無いトラック"), Clip(timeline_start=0, duration=10))],
        "テンプレートを配置",
    )
    assert refused is False
    assert window.document.project.media == ()


def test_a_successful_batch_reports_success(window: MainWindow) -> None:
    # 通ったのに偽を返すと、テンプレートを置いても素材の解析と控えの作成に進まず、
    # 「置いた」とも出ない
    media = MediaItem(path=Path("C:/素材/効果音.mp3"))
    assert window.execute_all([AddMedia(media)], "素材を追加") is True
    assert window.document.project.media == (media,)


class _Chosen:
    """テンプレートの棚の代わり 開くとすぐ、決めておいた選択で閉じる"""

    choice: tuple[str, list[MappedObject]] = ("place", [])

    def __init__(self, parent: object = None) -> None:
        del parent
        self.origin: Path | None = None

    def exec(self) -> QDialog.DialogCode:
        return QDialog.DialogCode.Accepted

    def deleteLater(self) -> None:  # noqa: N802 - 本物の棚（Qt の窓）と同じ名前
        """開いた側は閉じたあとに捨てる 代わりの物には捨てる中身が無い"""


def _refuse_everything(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, choice: tuple[str, list[MappedObject]]
) -> None:
    monkeypatch.setattr(_Chosen, "choice", choice)
    monkeypatch.setattr(template_dialog, "TemplateDialog", _Chosen)
    monkeypatch.setattr(window, "execute_all", lambda *_args, **_kw: False)


def test_a_refused_restyle_does_not_claim_success(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 断られて戻ったのに「適用した」と出すと、何も変わっていないことに気付けない
    text = GeneratedSource(kind="text", params={"text": "字幕"})
    for command in insert_generated(window.document.project, text, at_frame=0):
        window.execute(command)
    clip = window.document.project.timeline.tracks[0].clips[0]
    window.select_clip(clip.id)
    template = MappedObject(clip=Clip(timeline_start=0, duration=30, source=text), layer=1)
    _refuse_everything(window, monkeypatch, ("restyle", [template]))
    window.statusBar().clearMessage()

    window.show_templates()
    assert "適用した" not in window.statusBar().currentMessage()


def test_a_refused_place_does_not_analyze_the_rolled_back_media(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 戻した素材の解析や控えを頼むと、一覧に無い素材のために裏で重い処理が走る
    picture = tmp_path / "絵.png"
    picture.write_bytes(b"")
    item = MappedObject(
        clip=Clip(timeline_start=0, duration=30), layer=1, media_path=str(picture), kind="画像"
    )
    monkeypatch.setattr(
        "sashimono.ui.main_window._probe_or_none", lambda path: MediaItem(path=path)
    )
    requested: list[MediaItem] = []
    analyzed: list[MediaItem] = []
    monkeypatch.setattr(window, "_request_proxy", requested.append)
    monkeypatch.setattr(
        window._analyzer, "request", lambda media, **_kwargs: analyzed.append(media)
    )
    _refuse_everything(window, monkeypatch, ("place", [item]))
    window.statusBar().clearMessage()

    window.show_templates()
    assert analyzed == []
    assert requested == []
    assert "置いた" not in window.statusBar().currentMessage()


#: テキスト 1 つだけの ``.exo`` 素材を参照しないので、読み込みは配置だけで終わる
_EXO_TEXT = """[exedit]
width=1920
height=1080
rate=30
scale=1
length=300
audio_rate=44100
audio_ch=2
[0]
start=1
end=60
layer=1
group=1
overlay=1
camera=0
[0.0]
_name=テキスト
サイズ=48
表示速度=0.0
color=ffffff
color2=000000
font=Yu Gothic UI
text=53004b0055005400
[0.1]
_name=標準描画
X=0.0
Y=0.0
Z=0.0
拡大率=100.00
透明度=0.0
回転=0.00
blend=0
"""

#: 画像を 1 つ参照する ``.exo`` 素材の登録が先に走る
_EXO_IMAGE = """[exedit]
width=1920
height=1080
rate=30
scale=1
length=300
audio_rate=44100
audio_ch=2
[0]
start=1
end=60
layer=1
group=1
overlay=1
camera=0
[0.0]
_name=画像ファイル
ファイル={path}
[0.1]
_name=標準描画
X=0.0
Y=0.0
Z=0.0
拡大率=100.00
透明度=0.0
回転=0.00
blend=0
"""


def _write_exo(path: Path, text: str) -> Path:
    # AviUtl2 世代は UTF-8 判別は中身を見て決まるので、そのまま書いてよい
    path.write_text(text, encoding="utf-8")
    return path


def _chosen_file(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """ファイルを選ぶダイアログの代わり 開くとすぐ ``path`` を選んで閉じる"""
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *_args, **_kwargs: (str(path), ""))
    )


def test_a_refused_exo_import_does_not_claim_it_read_anything(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 断られて全部戻ったのに「N 個を読み込んだ」と出すと、タイムラインに
    # 何も増えていないことに気付けないまま作業を続けることになる
    _chosen_file(monkeypatch, _write_exo(tmp_path / "見出し.exo", _EXO_TEXT))
    monkeypatch.setattr(window, "execute_all", lambda *_args, **_kwargs: False)
    window.statusBar().clearMessage()

    window.import_exo()
    assert "読み込んだ" not in window.statusBar().currentMessage()


def test_a_refused_exo_placement_leaves_no_media_behind(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 素材の登録を配置と別の段で先に済ませると、配置を断られたときに素材だけが
    # 一覧に残り、その解析と控えの重い処理も裏で走り続ける
    picture = tmp_path / "絵.png"
    picture.write_bytes(b"")
    source = _write_exo(tmp_path / "素材つき.exo", _EXO_IMAGE.format(path=picture))
    _chosen_file(monkeypatch, source)
    monkeypatch.setattr(main_window_module, "probe_media", lambda path: MediaItem(path=path))
    monkeypatch.setattr(window, "execute_all", lambda *_args, **_kwargs: False)
    asked: list[MediaItem] = []
    monkeypatch.setattr(window, "_request_proxy", asked.append)
    monkeypatch.setattr(window._analyzer, "request", lambda media, **_kwargs: asked.append(media))
    window.statusBar().clearMessage()

    window.import_exo()
    assert window.document.project.media == ()
    assert asked == []
    assert "読み込んだ" not in window.statusBar().currentMessage()


def test_an_exo_registers_its_media_together_with_the_clips(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1 回の取り消しで素材もクリップも戻る 分けて実行すると、戻したときに
    # 使われていない素材だけが一覧に残る
    picture = tmp_path / "絵.png"
    picture.write_bytes(b"")
    source = _write_exo(tmp_path / "素材つき.exo", _EXO_IMAGE.format(path=picture))
    _chosen_file(monkeypatch, source)
    monkeypatch.setattr(main_window_module, "probe_media", lambda path: MediaItem(path=path))
    monkeypatch.setattr(window, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(window._analyzer, "request", lambda _media, **_kwargs: None)

    window.import_exo()
    assert len(window.document.project.media) == 1
    window.document.undo()
    assert len(window.document.project.media) == 0


def test_a_refused_media_import_does_not_claim_it_loaded_anything(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 断られて戻ったのに「1 件を読み込んだ」と出し、さらに一覧に無い素材の解析と
    # 控えを頼むと、取り消しても止まらない重い処理が裏で走り続ける
    movie = tmp_path / "映像.mp4"
    movie.write_bytes(b"")
    monkeypatch.setattr(main_window_module, "probe_media", lambda path: MediaItem(path=path))
    monkeypatch.setattr(window, "execute_all", lambda *_args, **_kwargs: False)
    asked: list[MediaItem] = []
    monkeypatch.setattr(window, "_request_proxy", asked.append)
    monkeypatch.setattr(window._analyzer, "request", lambda media, **_kwargs: asked.append(media))
    window.statusBar().clearMessage()

    window.import_media([movie])
    assert window.wait_for_imports()
    assert asked == []
    assert "読み込んだ" not in window.statusBar().currentMessage()


def test_a_refused_insert_does_not_select_something_else(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 置けていないのに再生ヘッドの位置にあった別のクリップを選ぶと、そのクリップの
    # 設定パネルが開き、追加できたように見える
    text = GeneratedSource(kind="text", params={"text": "先にあった字幕"})
    for command in insert_generated(window.document.project, text, at_frame=0):
        window.execute(command)
    window.select_clip(None)
    monkeypatch.setattr(window, "execute_all", lambda *_args, **_kwargs: False)

    window.add_text()
    assert window.selected_clip is None
