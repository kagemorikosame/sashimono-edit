"""映像のエンコーダの選び方（#365 AMD の GPU の AMF）

AMD の GPU の無い機械（この開発機と CI）では、AMF は FFmpeg に入っていても開けない
候補に足しても、必ず次の候補へ落ちることをここで押さえる AMD の実機での確かめは Issue に残す
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.model import MediaItem, VideoStreamInfo
from sashimono.core.timebase import FrameRate
from sashimono.engine import encoders
from sashimono.engine.cache import proxy
from sashimono.engine.cache.store import CacheStore
from sashimono.engine.encode import exporter
from sashimono.engine.encoders import (
    ENCODER_PRIORITIES,
    PRIORITY_AMD,
    PRIORITY_AUTO,
    PRIORITY_CPU,
    PRIORITY_INTEL,
    encoder_vendor,
    order_by_priority,
)
from tests.media_fixtures import SampleMedia

#: 試験で使う並び 書き出しと控えの既定と同じ形
ALL = ["h264_nvenc", "h264_amf", "h264_qsv", "libx264"]


@pytest.fixture
def forget_usable() -> Iterator[None]:
    """書けるかの答えを、試験の前後で忘れる 偽の答えが後の試験へ漏れると本物の NVENC が外れる"""
    encoders.usable.cache_clear()
    exporter._opens.cache_clear()
    yield
    encoders.usable.cache_clear()
    exporter._opens.cache_clear()


def _no_amd(name: str) -> None:
    """AMD の GPU の無い機械を真似る AMF だけ、この開発機と同じ失敗で断る"""
    if name.endswith("_amf"):
        raise RuntimeError("AMF を開けない（AMD の GPU が無い）")


class TestOrder:
    def test_amf_sits_between_nvenc_and_qsv(self) -> None:
        # 崩れると、NVENC の無い AMD の機械で CPU に落ちて書き出しが遅いまま、
        # または NVIDIA と AMD の両方がある機械で NVENC より AMF が先に選ばれる
        for preference in (exporter.VIDEO_CODEC_PREFERENCE, proxy.CODEC_PREFERENCE):
            assert list(preference) == ALL

    def test_auto_keeps_the_order(self) -> None:
        # 崩れると、設定を触っていない人の既定のエンコーダが変わる
        assert order_by_priority(ALL, PRIORITY_AUTO) == ALL

    @pytest.mark.parametrize(
        ("priority", "first"),
        [(PRIORITY_AMD, "h264_amf"), (PRIORITY_INTEL, "h264_qsv"), (PRIORITY_CPU, "libx264")],
    )
    def test_the_chosen_side_comes_first_and_the_rest_keep_their_order(
        self, priority: str, first: str
    ) -> None:
        # 崩れると、設定で選んだ側が使われないか、選んだ側が使えないときの落ち先が変わる
        ordered = order_by_priority(ALL, priority)
        assert ordered[0] == first
        assert [name for name in ordered if name != first] == [n for n in ALL if n != first]

    def test_an_unknown_priority_is_refused(self) -> None:
        # 黙って自動に戻すと、選んだ物が効かない理由が分からない
        with pytest.raises(ValueError, match="優先"):
            order_by_priority(ALL, "matrox")

    @pytest.mark.parametrize(
        ("name", "vendor"),
        [
            ("h264_nvenc", "nvidia"),
            ("hevc_amf", PRIORITY_AMD),
            ("h264_amf", PRIORITY_AMD),
            ("h264_qsv", PRIORITY_INTEL),
            ("libx264", PRIORITY_CPU),
        ],
    )
    def test_the_vendor_is_read_from_the_name(self, name: str, vendor: str) -> None:
        # 外れると、AMD を先にと選んでも AMF が前に来ない
        assert encoder_vendor(name) == vendor

    def test_every_priority_is_known(self) -> None:
        # 設定画面に出す項目と、ここで受け付ける項目が食い違うと、選んだだけで書き出しが断られる
        assert set(ENCODER_PRIORITIES) == {
            PRIORITY_AUTO,
            PRIORITY_AMD,
            PRIORITY_INTEL,
            PRIORITY_CPU,
        }


class TestProbe:
    def test_amf_without_an_amd_gpu_is_not_usable(self, forget_usable: None) -> None:
        """この開発機と CI には AMD の GPU が無い 入っているだけの AMF を使えると数えない

        数えると、書き出しの既定や控えの作成が AMF を先に試し、失敗してから次へ移る
        """
        try:
            encoders.probe_encoder("h264_amf")
        except Exception:
            assert not encoders.usable("h264_amf")
        else:
            pytest.skip("この機械では AMF で書ける（AMD の GPU がある）")

    def test_an_encoder_that_opens_but_writes_nothing_is_not_usable(
        self, monkeypatch: pytest.MonkeyPatch, forget_usable: None
    ) -> None:
        """開けても 1 コマも書けないエンコーダは使えないと数える

        開けるかだけで選ぶと、駆動が最初のコマで断る機械では書き出しの途中で失敗する
        """

        class Silent:
            """開けるが何も出さない文脈"""

            def encode(self, _frame: object) -> list[object]:
                return []

        monkeypatch.setattr(encoders, "_open_context", lambda _name: Silent())
        assert not encoders.usable("h264_amf")

    def test_libx264_is_usable(self, forget_usable: None) -> None:
        # 書けないと判定されると、GPU の無い機械で書き出しも控えも作れなくなる
        assert encoders.usable("libx264")


class TestExportCandidates:
    def test_amf_is_dropped_on_a_machine_without_amd(
        self, monkeypatch: pytest.MonkeyPatch, forget_usable: None
    ) -> None:
        # 落ちないと、AMD の無い機械で書き出しの窓の既定が開けない AMF になる
        monkeypatch.setattr(exporter, "_open_encoder", _no_amd)
        monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("h264_amf", "libx264"))
        assert exporter.available_video_codecs() == ["libx264"]

    def test_the_priority_reorders_the_candidates(
        self, monkeypatch: pytest.MonkeyPatch, forget_usable: None
    ) -> None:
        # 効かないと、設定で CPU を先にと選んでも GPU で書き出される
        monkeypatch.setattr(exporter, "_open_encoder", lambda _name: None)
        monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", tuple(ALL))
        assert exporter.available_video_codecs() == ALL
        assert exporter.available_video_codecs(PRIORITY_CPU)[0] == "libx264"
        assert exporter.available_video_codecs(PRIORITY_AMD)[0] == "h264_amf"


class TestProxyCandidates:
    def test_amf_is_dropped_on_a_machine_without_amd(
        self, monkeypatch: pytest.MonkeyPatch, forget_usable: None
    ) -> None:
        # 落ちないと、素材を読み込むたびに控えの作成が開けない AMF から試してやり直す
        monkeypatch.setattr(encoders, "probe_encoder", _no_amd)
        monkeypatch.setattr(proxy, "CODEC_PREFERENCE", ("h264_amf", "libx264"))
        assert proxy.proxy_codecs() == ["libx264"]

    def test_the_priority_reorders_the_candidates(
        self, monkeypatch: pytest.MonkeyPatch, forget_usable: None
    ) -> None:
        # 効かないと、設定で選んだ側が控えの作成にだけ効かない
        monkeypatch.setattr(encoders, "probe_encoder", lambda _name: None)
        monkeypatch.setattr(proxy, "CODEC_PREFERENCE", tuple(ALL))
        assert proxy.proxy_codecs(PRIORITY_INTEL)[0] == "h264_qsv"

    def test_a_proxy_is_made_with_the_next_candidate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        forget_usable: None,
        sample_av: SampleMedia,
        tmp_path: Path,
    ) -> None:
        """AMF を先に置いても、AMD の無い機械では次の候補で控えができる

        できないと、AMD の GPU の無い機械で 4K の素材を読み込んでも、プレビューが軽くならない
        """
        monkeypatch.setattr(proxy, "CODEC_PREFERENCE", ("h264_amf", "libx264"))
        target = tmp_path / "proxy.mp4"
        assert proxy.create_proxy(sample_av.path, target, height=120) == target

    def test_the_builder_passes_the_priority(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 渡さないと、設定で選んだ優先が控えの作成にだけ効かない
        seen: list[str] = []
        done = threading.Event()

        def fake(*_args: object, encoder_priority: str, **_kwargs: object) -> None:
            seen.append(encoder_priority)
            done.set()

        monkeypatch.setattr(proxy, "create_proxy", fake)
        builder = proxy.ProxyBuilder(
            proxy.ProxyStore(CacheStore(tmp_path), height=120), encoder_priority=PRIORITY_CPU
        )
        media = MediaItem(
            path=tmp_path / "素材.mp4",
            duration=Fraction(2),
            video_streams=(
                VideoStreamInfo(
                    index=0,
                    width=3840,
                    height=2160,
                    frame_rate=FrameRate(30),
                    time_base=Fraction(1, 30),
                    codec="h264",
                    pixel_format="yuv420p",
                ),
            ),
        )
        try:
            assert builder.encoder_priority == PRIORITY_CPU
            builder.request(media)
            assert done.wait(30)
        finally:
            builder.close()
        assert seen == [PRIORITY_CPU]
