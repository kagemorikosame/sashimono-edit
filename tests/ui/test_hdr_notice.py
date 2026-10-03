"""HDR と広い色域の素材を読み込んだときの知らせ

Sashimono は HDR（PQ・HLG）と BT.2020 の素材を変換せずに SDR として描く 何も言わずに
出すと、白っぽく褪せた絵をソフトの不具合だと受け取られる 素材の印（伝達特性と原色）で
見分け、読み込んだときに 1 度だけ知らせ、素材一覧の行にも印を付ける

素材は ffmpeg で印を付けた短い動画を作る（:func:`tests.media_fixtures.make_color_tagged`）
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.core.io import project_from_dict, project_to_dict
from sashimono.core.model import Project, ProjectSettings
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import probe_media
from sashimono.engine.decode.probe import clear_probe_cache
from sashimono.ui import hdr_notice
from sashimono.ui.main_window import MainWindow
from sashimono.ui.media_pool import MediaPoolWidget
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.media_fixtures import make_color_tagged, make_sample


@pytest.fixture(scope="module")
def tagged(media_dir: Path) -> dict[str, Path]:
    """印だけを変えた 4 本 PQ・HLG・原色だけ BT.2020・BT.709（SDR）"""
    return {
        "pq": make_color_tagged(media_dir, "hdr-pq.mp4", transfer="smpte2084", primaries="bt2020"),
        "hlg": make_color_tagged(
            media_dir, "hdr-hlg.mp4", transfer="arib-std-b67", primaries="bt2020"
        ),
        "wide": make_color_tagged(
            media_dir, "wide-bt2020.mp4", transfer="bt709", primaries="bt2020"
        ),
        "sdr": make_color_tagged(
            media_dir, "sdr-bt709.mp4", transfer="bt709", primaries="bt709", matrix="bt709"
        ),
    }


class TestTheProbeReadsTheTags:
    """読み込みの調べで、伝達特性と原色の印を読む"""

    @pytest.mark.parametrize(
        ("name", "transfer", "primaries", "outside"),
        [
            ("pq", "smpte2084", "bt2020", "HDR（PQ）"),
            ("hlg", "arib-std-b67", "bt2020", "HDR（HLG）"),
            ("wide", "bt709", "bt2020", "広色域（BT.2020）"),
            ("sdr", "bt709", "bt709", ""),
        ],
    )
    def test_the_tags_name_the_colors(
        self, tagged: dict[str, Path], name: str, transfer: str, primaries: str, outside: str
    ) -> None:
        # 印を読まないと、HDR の素材を SDR の素材と見分けられず、褪せた理由を知らせられない
        clear_probe_cache()
        media = probe_media(tagged[name])
        (stream,) = media.video_streams
        assert stream.color_transfer == transfer
        assert stream.color_primaries == primaries
        assert media.color_outside_sdr == outside

    def test_an_untagged_video_is_not_called_hdr(self, media_dir: Path) -> None:
        # 印の無い素材の中身を推し量ると、SDR の素材まで HDR と言い当ててしまう
        plain = make_sample(media_dir, "untagged.mp4", audio=False, duration=0.5)
        clear_probe_cache()
        media = probe_media(plain.path)
        assert media.video_streams[0].color_transfer == ""
        assert media.color_outside_sdr == ""

    def test_the_tags_survive_saving(self, tagged: dict[str, Path]) -> None:
        # 保存して開き直したら印が消える、では、開き直した作品の一覧から HDR の印が消える
        media = probe_media(tagged["pq"])
        project = Project.create()
        project = replace(project, media=(media,))
        again = project_from_dict(project_to_dict(project))
        assert again.media[0].color_outside_sdr == "HDR（PQ）"


@pytest.fixture
def window(qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    del qt_application
    created = MainWindow(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30))),
        confirm_unsaved=False,
    )
    # 控えと解析は裏のスレッドで素材を開く この試験が見たいのは知らせだけ
    monkeypatch.setattr(created, "_request_proxy", lambda _media: None)
    monkeypatch.setattr(created._analyzer, "request", lambda _media, **_kwargs: None)
    yield created
    created.close()


def _import(window: MainWindow, *paths: Path) -> None:
    window.import_media(list(paths))
    assert window.wait_for_imports()


class TestTheNoticeOnImport:
    """読み込んだときに 1 度だけ知らせる 何度も出して邪魔にしない"""

    def test_reading_hdr_material_tells_once(
        self,
        window: MainWindow,
        tagged: dict[str, Path],
        silent_hdr_notice: list[list[str]],
    ) -> None:
        # まとめて読み込んだ分は 1 つの窓にまとめ、SDR の素材は並べない
        _import(window, tagged["pq"], tagged["sdr"], tagged["hlg"])
        assert silent_hdr_notice == [["hdr-pq.mp4", "hdr-hlg.mp4"]]

        # 同じ素材を読み込み直しても、もう出さない 出すと、読み込むたびに閉じることになる
        _import(window, tagged["pq"])
        assert len(silent_hdr_notice) == 1

        # 知らせていない素材なら出す
        _import(window, tagged["wide"])
        assert silent_hdr_notice[-1] == ["wide-bt2020.mp4"]

    def test_sdr_material_is_not_told(
        self, window: MainWindow, tagged: dict[str, Path], silent_hdr_notice: list[list[str]]
    ) -> None:
        _import(window, tagged["sdr"])
        assert silent_hdr_notice == []

    def test_turning_it_off_in_the_preferences_stops_the_window(
        self, window: MainWindow, tagged: dict[str, Path], silent_hdr_notice: list[list[str]]
    ) -> None:
        # 切ったのに出るなら、設定がある方が質が悪い
        window._apply_preferences(Preferences(hdr_notice=False))
        _import(window, tagged["pq"])
        assert silent_hdr_notice == []

    def test_turning_it_back_on_tells_material_read_while_it_was_off(
        self, window: MainWindow, tagged: dict[str, Path], silent_hdr_notice: list[list[str]]
    ) -> None:
        # 切っている間に読み込んだ素材を「知らせた」と覚えると、同じ窓のまま設定を入れ直して
        # 読み込み直しても、1 度も知らせていないのに知らせが出ない
        window._apply_preferences(Preferences(hdr_notice=False))
        _import(window, tagged["pq"])
        window._apply_preferences(Preferences(hdr_notice=True))
        _import(window, tagged["pq"])
        assert silent_hdr_notice == [["hdr-pq.mp4"]]

    def test_dont_tell_again_turns_the_preference_off(
        self, window: MainWindow, tagged: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 窓の〔次から知らせない〕は設定を切って保存する 次に起動したときも出さない
        shown: list[int] = []

        def decline(_parent: object, media: list[object]) -> bool:
            shown.append(len(media))
            return False

        monkeypatch.setattr(hdr_notice, "ask_hdr_notice", decline)
        _import(window, tagged["pq"])
        _import(window, tagged["hlg"])
        assert shown == [1]
        assert window._preferences.hdr_notice is False
        assert PreferenceStore().load().hdr_notice is False

    def test_the_media_list_marks_the_material(
        self, window: MainWindow, tagged: dict[str, Path]
    ) -> None:
        # 窓を閉じた後も、どの素材が褪せて出るのかを一覧で見分けられる
        _import(window, tagged["pq"], tagged["sdr"])
        pool = window.findChild(MediaPoolWidget)
        assert pool is not None
        by_name = {media.name: media for media in window.project.media}
        hdr_row = pool.row_text(by_name["hdr-pq.mp4"].id)
        sdr_row = pool.row_text(by_name["sdr-bt709.mp4"].id)
        assert hdr_row is not None and "HDR（PQ）" in hdr_row
        assert sdr_row is not None and "HDR" not in sdr_row


class TestThePreference:
    def test_the_default_tells(self) -> None:
        # 既定は知らせる HDR を知らない人ほど、褪せた絵を不具合だと受け取る
        assert Preferences().hdr_notice is True

    def test_it_is_saved_and_a_broken_value_falls_back(self, tmp_path: Path) -> None:
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(Preferences(hdr_notice=False))
        assert store.load().hdr_notice is False
        broken = tmp_path / "broken.json"
        broken.write_text('{"hdr_notice": "off"}', encoding="utf-8")
        assert PreferenceStore(broken).load().hdr_notice is True

    def test_the_dialog_shows_and_returns_it(self, qt_application: QApplication) -> None:
        """設定画面の印が今の値を映し、OK で返す値に入る

        映さないと、切ってあるのに入に見える 返さないと、印を付け外しして OK を押しても
        設定が変わらず、知らせを切りたい人が切れない（入れ直したい人が戻せない）
        """
        del qt_application
        dialog = PreferencesDialog(Preferences(hdr_notice=False))
        try:
            assert dialog.preferences().hdr_notice is False
            dialog._hdr_notice.setChecked(True)
            assert dialog.preferences().hdr_notice is True
        finally:
            dialog.close()

    def test_the_notice_says_it_will_look_washed_out(self, tagged: dict[str, Path]) -> None:
        # 何が起きるか（白っぽく出る）と、どうすればよいか（SDR へ変換してから読み込む）を言う
        media = probe_media(tagged["pq"])
        text = hdr_notice.describe_outside_sdr([media])
        assert "hdr-pq.mp4（HDR（PQ））" in text
        assert "SDR（Rec.709）" in text
        assert "白っぽく" in text
