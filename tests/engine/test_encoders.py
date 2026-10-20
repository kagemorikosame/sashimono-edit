"""映像のエンコーダの選び方（#365 AMD の GPU の AMF）

AMD の GPU の無い機械（この開発機と CI）では、AMF は FFmpeg に入っていても開けない
候補に足しても、必ず次の候補へ落ちることをここで押さえる AMD の実機での確かめは Issue に残す
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import av.error
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


class _Recording:
    """開けたときの選択肢とビットレートを覚える偽のエンコーダ 1 コマ渡せば 1 つ出す"""

    def __init__(self) -> None:
        self.width = self.height = 0
        self.pix_fmt = ""
        self.time_base: object = None
        self.bit_rate = 0
        self.options: dict[str, str] = {}
        self.opened_with: dict[str, str] | None = None

    def open(self) -> None:
        self.opened_with = dict(self.options)

    def encode(self, frame: object) -> list[object]:
        return [] if frame is None else [object()]


class TestAmfOptions:
    """AMF に渡す品質と上限のビットレート（2026-10-20 利用者の決定 AMD の実機で確かめていない）"""

    def test_amf_gets_balanced_quality_and_a_peak_of_one_and_a_half(self) -> None:
        # 崩れると、AMF の駆動の既定任せになり、機械ごとに画とファイルの大きさが変わる
        assert encoders.encoder_options("h264_amf", 12_000_000) == {
            "quality": "balanced",
            "maxrate": "18000000",
        }
        assert encoders.encoder_options("hevc_amf", 8_000_000)["maxrate"] == "12000000"

    def test_without_a_bit_rate_only_the_quality_is_given(self) -> None:
        # 上限だけ 0 で渡すと、amfenc は上限なしの扱いと食い違い、駆動に断られうる
        assert encoders.encoder_options("h264_amf", None) == {"quality": "balanced"}

    @pytest.mark.parametrize("name", ["h264_nvenc", "h264_qsv", "libx264"])
    def test_other_encoders_get_nothing(self, name: str) -> None:
        # AMF の選択肢を NVENC や libx264 へ渡すと、知らない名前で開けずに書き出せなくなる
        assert encoders.encoder_options(name, 12_000_000) == {}

    def test_the_probe_opens_amf_with_the_same_options_as_the_export(
        self, monkeypatch: pytest.MonkeyPatch, forget_usable: None
    ) -> None:
        """試しに開くときも、書き出しと同じ選択肢を渡す

        渡さずに試すと、駆動が選択肢を断る機械で、試しには通って書き出しで落ちる
        """
        made: list[_Recording] = []

        def fake(_name: str) -> _Recording:
            made.append(_Recording())
            return made[-1]

        monkeypatch.setattr(encoders, "_new_context", fake)
        assert encoders.usable("h264_amf")
        assert made[0].bit_rate == encoders.PROBE_BIT_RATE
        assert made[0].opened_with == encoders.encoder_options("h264_amf", encoders.PROBE_BIT_RATE)

    def test_the_export_lets_the_users_own_options_win(self, tmp_path: Path) -> None:
        # 既定が後に勝つと、品質を名指しした人の指定が黙って消える
        settings = exporter.ExportSettings(
            path=tmp_path / "a.mp4", video_bitrate=10_000_000, options={"quality": "speed"}
        )
        assert exporter._video_options("h264_amf", settings) == {
            "quality": "speed",
            "maxrate": "15000000",
        }
        assert exporter._video_options("libx264", settings) == {"quality": "speed"}


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
        # 落ちる先の libx264 で書けない機械では、確かめる相手がいない 落とすと理由が AMF に見える
        if not encoders.usable("libx264"):
            pytest.skip("libx264 で書けない")
        monkeypatch.setattr(proxy, "CODEC_PREFERENCE", ("h264_amf", "libx264"))
        target = tmp_path / "proxy.mp4"
        assert proxy.create_proxy(sample_av.path, target, height=120) == target

    def test_the_progress_does_not_go_back_when_the_next_candidate_starts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_av: SampleMedia,
        tmp_path: Path,
    ) -> None:
        """先頭の候補が途中で失敗し次の候補で作り直しても、進み具合は戻らない

        戻ると、素材の一覧のプロキシの進み具合が 60% から 1% へ下がり、壊れたように見える
        """
        if not encoders.usable("libx264"):
            pytest.skip("libx264 で書けない")
        real = proxy._transcode

        def fails_mid_way(*args: Any, **options: Any) -> bool:
            if options["codec"] == "h264_amf":
                # 途中まで進めてから、駆動に断られた形で落ちる
                options["progress"](0.6)
                raise av.error.ExternalError(-542398533, "偽の駆動の失敗")
            made: bool = real(*args, **options)
            return made

        monkeypatch.setattr(proxy, "CODEC_PREFERENCE", ("h264_amf", "libx264"))
        monkeypatch.setattr(proxy, "proxy_codecs", lambda _priority: ["h264_amf", "libx264"])
        monkeypatch.setattr(proxy, "_transcode", fails_mid_way)
        seen: list[float] = []
        target = tmp_path / "proxy.mp4"
        assert proxy.create_proxy(sample_av.path, target, height=120, progress=seen.append)
        assert seen == sorted(seen), f"進み具合が戻った: {seen}"
        assert seen[-1] == pytest.approx(1.0)

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
