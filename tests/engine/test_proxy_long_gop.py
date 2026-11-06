"""キーフレームの間隔が長い素材の控え（#430）

録画ソフトの既定で撮った 1080p は、キーフレームが何秒も空く 飛ぶたびに遠くの
キーフレームから読み直すので、縦の画素数が小さくても控えが効く
（測った値は :mod:`sashimono.engine.cache.proxy` の「キーフレームの間隔」の表）
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import sashimono.engine.cache.proxy as proxy_module
import sashimono.engine.decode.probe as probe_module
from sashimono.core.model import MediaId, MediaItem, VideoStreamInfo
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache.proxy import (
    KEYFRAME_GAP_LIMIT,
    PROXY_KEYFRAME_SECONDS,
    ProxyBuilder,
    ProxyStore,
    create_proxy,
    is_worth_proxying,
    needs_keyframe_check,
    proxy_codecs,
)
from sashimono.engine.cache.store import CacheStore
from sashimono.engine.decode import keyframe_gap, probe_media
from tests.media_fixtures import SampleMedia, libx264_available


def _hd(path: Path, *, height: int = 1080) -> MediaItem:
    return MediaItem(
        path=path,
        duration=Fraction(60),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=1920,
                height=height,
                frame_rate=FrameRate(30),
                time_base=Fraction(1, 30),
                codec="h264",
                pixel_format="yuv420p",
            ),
        ),
    )


#: 止める所の幅 B フレームの並べ替えで、表示の時刻が数コマ先のパケットが先に来る
#: 10fps の素材で 3 コマ
_REORDER = Fraction(3, 10)


def _ffmpeg(path: Path, arguments: str) -> Path:
    """ffmpeg で試験の素材を作る すでにあればそれを使う"""
    if not path.exists():
        if not libx264_available():
            pytest.skip("ffmpeg に libx264 が無い")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments.split()]
        subprocess.run([*command, str(path)], check=True, capture_output=True)
    return path


def _settle(builder: ProxyBuilder) -> None:
    """裏の仕事が全部終わるまで待つ 係は 1 本ずつ順に回すので、後ろへ足した空の仕事を待てばよい"""
    builder._executor.submit(lambda: None).result(timeout=60.0)


class TestReadingTheGap:
    """取り出すのはパケットだけ 復号せずに、キーフレームの並びから間隔を出す"""

    def test_a_single_keyframe_counts_to_the_end(self, sample_long_gop: SampleMedia) -> None:
        # 頭に 1 つしか無ければ、どこへ飛んでも頭から読み直す 間は終わりまで
        gap = keyframe_gap(sample_long_gop.path)
        assert gap is not None
        assert gap > KEYFRAME_GAP_LIMIT
        assert gap <= Fraction(6)

    def test_a_regular_gop_is_read_exactly(self, sample_long: SampleMedia) -> None:
        # 12 コマごと（30fps で 0.4 秒）に置いた素材 間を長く見誤ると、要らない控えを作る
        assert keyframe_gap(sample_long.path) == Fraction(12, 30)

    def test_it_stops_once_it_is_long_enough(self, sample_long_gop: SampleMedia) -> None:
        # 長いと分かった所でやめる 50 秒間隔の素材を 50 秒ぶん読むと、裏の係が詰まる
        gap = keyframe_gap(sample_long_gop.path, enough=Fraction(1))
        assert gap is not None
        assert Fraction(1) < gap <= Fraction(1) + _REORDER

    def test_it_reads_only_the_head(
        self, sample_long_gop: SampleMedia, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 読む長さの上限を外すと、3 時間の録画を最後まで読むことになる
        monkeypatch.setattr(probe_module, "KEYFRAME_SCAN_SECONDS", Fraction(2))
        gap = keyframe_gap(sample_long_gop.path)
        assert gap is not None
        assert gap <= Fraction(2) + _REORDER

    def test_it_stops_after_enough_gaps(
        self, media_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 全部がキーフレームの素材は 1 つずつが重い 間隔を数えて早く切り上げる
        # 頭だけ 0.4 秒ごとで、その後は終わりまで空く素材を作る 数えて止まれば、
        # 後ろの長い間を読む前に 0.4 秒と答える
        path = _ffmpeg(
            media_dir / "short_then_long.mp4",
            "-f lavfi -i testsrc2=size=160x120:rate=10:duration=6 -c:v libx264 "
            "-g 1000 -keyint_min 1000 -sc_threshold 0 -force_key_frames 0,0.4,0.8",
        )
        whole = keyframe_gap(path)
        assert whole is not None
        assert whole > Fraction(4)
        monkeypatch.setattr(probe_module, "KEYFRAME_SCAN_GAPS", 2)
        assert keyframe_gap(path) == Fraction(2, 5)

    def test_audio_only_is_unknown(self, media_dir: Path) -> None:
        # 音だけの素材で値を返すと、映像の無い素材を控えの変換へ回して失敗を数える
        path = _ffmpeg(media_dir / "tone.m4a", "-f lavfi -i sine=duration=1 -c:a aac")
        assert keyframe_gap(path) is None

    def test_a_missing_file_is_unknown(self, tmp_path: Path) -> None:
        # 外付けの円盤を抜いた素材で投げると、裏の係が失敗として数える
        assert keyframe_gap(tmp_path / "missing.mp4") is None


class TestWhenItIsWorthIt:
    def test_a_long_gap_is_worth_it_at_1080p(self, tmp_path: Path) -> None:
        # 作らないと、録画ソフトで撮った 1080p は飛ぶたびに 200ms 近く待たされる
        assert is_worth_proxying(_hd(tmp_path / "a.mp4"), Fraction(8))

    def test_the_limit_itself_is_not(self, tmp_path: Path) -> None:
        # 境目を動かしたら、この 2 つのどちらかが落ちる
        media = _hd(tmp_path / "a.mp4")
        assert not is_worth_proxying(media, KEYFRAME_GAP_LIMIT)
        assert is_worth_proxying(media, KEYFRAME_GAP_LIMIT + Fraction(1, 30))

    def test_an_unknown_gap_is_not(self, tmp_path: Path) -> None:
        # 分からない素材に作ると、開けない素材のたびに変換を試して失敗を数える
        assert not is_worth_proxying(_hd(tmp_path / "a.mp4"), None)

    def test_a_still_is_never_checked(self, tmp_path: Path) -> None:
        # 静止画は 1 枚読むだけ 調べると読み込むたびに無駄に開き、作ると変換に失敗する
        still = replace(_hd(tmp_path / "a.png"), duration=Fraction(0))
        assert not needs_keyframe_check(still)
        assert not is_worth_proxying(still, Fraction(50))

    def test_a_tall_source_is_not_checked(self, tmp_path: Path) -> None:
        # 4K は大きさだけで作る 調べると読み込むたびに無駄に読む
        assert not needs_keyframe_check(_hd(tmp_path / "a.mp4", height=2160))
        assert needs_keyframe_check(_hd(tmp_path / "a.mp4"))


@pytest.mark.skipif(not proxy_codecs(), reason="控えを作れるコーデックが無い")
class TestTheProxyItself:
    def test_it_has_short_gops(self, sample_long_gop: SampleMedia, tmp_path: Path) -> None:
        # 符号化器の既定（250 コマ）のままだと、控えなのに飛ぶ待ちが元より長いことがある
        made = create_proxy(sample_long_gop.path, tmp_path / "p.mp4", height=120)
        assert made is not None
        gap = keyframe_gap(made)
        assert gap is not None
        assert gap <= PROXY_KEYFRAME_SECONDS

    def test_a_slow_rate_still_has_two_frames_per_key(self) -> None:
        # NVENC は 1 コマごとを断る 断られると CPU まで落ちて変換が遅くなる
        assert proxy_module._keyframe_every(Fraction(1)) == 2
        assert proxy_module._keyframe_every(Fraction(30)) == 15
        assert proxy_module._keyframe_every(Fraction(60000, 1001)) == 30


class TestTheStore:
    def test_switched_off_it_hides_a_made_proxy(self, tmp_path: Path) -> None:
        # 切ったのに前に作った控えを返すと、1080p の素材が粗いまま映り続ける
        media = _hd(tmp_path / "a.mp4")
        cache = CacheStore(tmp_path / "cache")
        on = ProxyStore(cache, height=120)
        on.prepare(media).write_bytes(b"made")
        assert on.find(media) is not None
        assert ProxyStore(cache, height=120, long_gop=False).find(media) is None

    def test_switched_off_it_still_serves_a_tall_source(self, tmp_path: Path) -> None:
        # 切るのは間隔で作る分だけ 4K の控えまで止めると、4K が重くなる
        media = _hd(tmp_path / "a.mp4", height=2160)
        cache = CacheStore(tmp_path / "cache")
        ProxyStore(cache, height=120).prepare(media).write_bytes(b"made")
        assert ProxyStore(cache, height=120, long_gop=False).find(media) is not None


@pytest.mark.skipif(not proxy_codecs(), reason="控えを作れるコーデックが無い")
class TestBuildingInTheBackground:
    def test_a_long_gap_gets_a_proxy(self, sample_long_gop: SampleMedia, tmp_path: Path) -> None:
        # 小さくても間隔が長ければ作る ここが通らないと、録画の素材は飛ぶたびに待たされる
        store = ProxyStore(CacheStore(tmp_path), height=120)
        builder = ProxyBuilder(store)
        media = probe_media(sample_long_gop.path)
        ready: list[MediaId] = []
        try:
            builder.request(media, on_ready=ready.append)
            _settle(builder)
        finally:
            builder.close()
        assert ready == [media.id]
        assert store.find(media) is not None

    def test_the_older_version_is_thrown_away(
        self, sample_long_gop: SampleMedia, tmp_path: Path
    ) -> None:
        # 版を上げると前の控えはもう読まれない 置き場は片付けないので、残すと溜まり続ける
        cache = CacheStore(tmp_path)
        store = ProxyStore(cache, height=120)
        media = probe_media(sample_long_gop.path)
        older = cache.prepare(
            proxy_module.NAMESPACE, store.key_for(media, proxy_module.FORMAT_VERSION - 1), ".mp4"
        )
        older.write_bytes(b"old")
        other = ProxyStore(cache, height=240).prepare(media)
        other.write_bytes(b"another size")
        builder = ProxyBuilder(store)
        try:
            builder.request(media)
            _settle(builder)
        finally:
            builder.close()
        assert store.find(media) is not None
        assert not older.exists()
        # 別の大きさは今も設定を戻せば使う 消すと、大きさを戻したときに作り直しになる
        assert other.exists()

    def test_a_short_gap_is_left_alone_quietly(
        self, sample_long: SampleMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 間隔の短い素材は作らない 調べている間も「控えを作っています」を出さない
        # 出すと、1080p を読み込むたびに進み具合が出て、何もせずに消える
        started = threading.Event()
        release = threading.Event()
        original = keyframe_gap

        def blocking(*args: object, **kwargs: object) -> Fraction | None:
            started.set()
            release.wait(30.0)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(proxy_module, "keyframe_gap", blocking)
        store = ProxyStore(CacheStore(tmp_path), height=120)
        builder = ProxyBuilder(store)
        media = probe_media(sample_long.path)
        try:
            builder.request(media)
            assert started.wait(30.0), "間隔を調べ始めない"
            assert builder.poll().total == 0
            assert builder.progress(media.id) is None
            release.set()
            _settle(builder)
            snapshot = builder.poll()
        finally:
            release.set()
            builder.close()
        assert snapshot.total == 0
        assert not snapshot.failures
        assert store.find(media) is None

    def test_a_short_gap_is_read_once(
        self, sample_long: SampleMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 設定を当て直すたびに全部の素材を頼み直す 覚えないと毎回読み直す
        calls: list[Path] = []
        original = keyframe_gap

        def counting(path: Path, *args: object, **kwargs: object) -> Fraction | None:
            calls.append(path)
            return original(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(proxy_module, "keyframe_gap", counting)
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path), height=120))
        media = probe_media(sample_long.path)
        try:
            builder.request(media)
            _settle(builder)
            builder.request(media)
            _settle(builder)
        finally:
            builder.close()
        assert len(calls) == 1

    def test_switched_off_it_does_not_even_look(
        self, sample_long_gop: SampleMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 切ったら読みもしない 切っても裏で素材を読み続けるなら、切った意味が薄い
        calls: list[object] = []
        monkeypatch.setattr(proxy_module, "keyframe_gap", lambda *a, **k: calls.append(a))
        store = ProxyStore(CacheStore(tmp_path), height=120, long_gop=False)
        builder = ProxyBuilder(store)
        media = probe_media(sample_long_gop.path)
        try:
            builder.request(media)
            _settle(builder)
        finally:
            builder.close()
        assert calls == []
        assert store.find(media) is None

    def test_forgetting_while_reading_makes_nothing(
        self, sample_long_gop: SampleMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 調べている間に素材を外したら作らない 作ると、一覧に無い素材の変換が走る
        started = threading.Event()
        release = threading.Event()
        original = keyframe_gap

        def blocking(*args: object, **kwargs: object) -> Fraction | None:
            started.set()
            release.wait(30.0)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(proxy_module, "keyframe_gap", blocking)
        store = ProxyStore(CacheStore(tmp_path), height=120)
        builder = ProxyBuilder(store)
        media = probe_media(sample_long_gop.path)
        ready: list[MediaId] = []
        try:
            builder.request(media, on_ready=ready.append)
            assert started.wait(30.0), "間隔を調べ始めない"
            builder.forget(media.id)
            release.set()
            _settle(builder)
            time.sleep(0.05)
            snapshot = builder.poll()
        finally:
            release.set()
            builder.close()
        assert ready == []
        assert snapshot.total == 0
        assert store.find(media) is None
