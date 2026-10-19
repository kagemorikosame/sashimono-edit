"""プレビュー用の控え（プロキシ）

控えは**見るためだけ**のもの 大きさだけを縮め、フレームの時刻と本数は
元の素材と同じに保つ ここがずれると、プレビューのときだけ絵が前後にずれる
という、原因の分かりにくい不具合になるので、時刻の一致を厳密に見る
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import sashimono.engine.cache.proxy as proxy_module
from sashimono.core.commands import AddClip, AddMedia, AddTrack
from sashimono.core.model import (
    Clip,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache.proxy import (
    MIN_SOURCE_HEIGHT,
    ProxyBuilder,
    ProxyStore,
    create_proxy,
    is_worth_proxying,
    proxy_codecs,
)
from sashimono.engine.cache.store import CacheStore
from sashimono.engine.decode import VideoDecoder, probe_media
from sashimono.engine.gpu import GLContextError, OffscreenGLContext
from sashimono.engine.render import FrameRenderer
from tests.media_fixtures import (
    SampleMedia,
    decode_all_frames,
    make_delayed,
    make_rotated,
)

pytestmark = pytest.mark.skipif(not proxy_codecs(), reason="控えを作れるコーデックが無い")


@pytest.fixture(scope="module")
def gl_context() -> Iterator[OffscreenGLContext]:
    """オフスクリーンの GL コンテキスト 作れない環境ではテストを飛ばす"""
    try:
        context = OffscreenGLContext()
    except GLContextError as exc:
        pytest.skip(f"OpenGL コンテキストを作れない: {exc}")
    yield context
    context.release()


def _media(path: Path, width: int, height: int, *, still: bool = False) -> MediaItem:
    return MediaItem(
        path=path,
        duration=Fraction(0) if still else Fraction(2),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=width,
                height=height,
                frame_rate=FrameRate(30),
                time_base=Fraction(1, 30),
                codec="h264",
                pixel_format="yuv420p",
            ),
        ),
    )


def _tall(path: Path) -> tuple[VideoStreamInfo, ...]:
    """控えを作る値打ちがある大きさに見せかけたストリーム

    4K の素材を試験のたびに作ると時間がかかりすぎる 判定に使うのは
    高さだけなので、そこだけ差し替えて小さい素材で確かめる
    """
    return (
        VideoStreamInfo(
            index=0,
            width=320,
            height=MIN_SOURCE_HEIGHT,
            frame_rate=FrameRate(30),
            time_base=Fraction(1, 30),
            codec="h264",
            pixel_format="yuv420p",
        ),
    )


class TestWhenItIsWorthIt:
    """小さい素材まで変換すると、待たされるだけで速くならない"""

    def test_a_4k_source_is_worth_it(self, tmp_path: Path) -> None:
        # 4K の素材に控えが作られないと、プレビューが重いままコマ落ちする
        assert is_worth_proxying(_media(tmp_path / "a.mp4", 3840, 2160))

    def test_a_1080p_source_is_not(self, tmp_path: Path) -> None:
        # 1080p の素材まで控えを作ると、置くたびに変換を待たされるのに再生は速くならない
        assert not is_worth_proxying(_media(tmp_path / "a.mp4", 1920, 1080))

    def test_the_boundary_is_the_stated_height(self, tmp_path: Path) -> None:
        # 境目を動かしたら、この 2 つのどちらかが落ちる
        assert is_worth_proxying(_media(tmp_path / "a.mp4", 1920, MIN_SOURCE_HEIGHT))
        assert not is_worth_proxying(_media(tmp_path / "a.mp4", 1920, MIN_SOURCE_HEIGHT - 1))

    def test_a_still_is_not(self, tmp_path: Path) -> None:
        # 静止画は 1 枚読むだけ 控えを作っても速くならない
        assert not is_worth_proxying(_media(tmp_path / "a.png", 4000, 3000, still=True))

    def test_audio_only_is_not(self, tmp_path: Path) -> None:
        # 映像の無い素材を控えの変換に回すと、作れない控えのために無駄な処理が走る
        assert not is_worth_proxying(MediaItem(path=tmp_path / "a.wav", duration=Fraction(2)))


class TestMakingOne:
    def test_it_shrinks_the_picture(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        # 控えが元の大きさのまま作られると、変換を待ったのにプレビューが軽くならない
        target = tmp_path / "proxy.mp4"
        assert create_proxy(sample_av.path, target, height=120) == target
        made = probe_media(target)
        assert made.video_streams[0].height == 120
        # 縦横比を保つ 崩すと、プレビューだけ伸びた絵になる
        assert made.video_streams[0].width == 160

    def test_the_frame_times_match_the_source(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        """フレームの時刻が元の素材と**1 つ残らず**同じ

        時刻を振り直すと、可変フレームレートの素材で控えと元の絵がずれる
        """
        target = tmp_path / "proxy.mp4"
        assert create_proxy(sample_av.path, target, height=120) is not None
        source_times = [time for time, _ in decode_all_frames(sample_av.path)]
        proxy_times = [time for time, _ in decode_all_frames(target)]
        assert proxy_times == pytest.approx(source_times, abs=1e-6)

    def test_a_fractional_rate_survives(self, sample_ntsc: SampleMedia, tmp_path: Path) -> None:
        # 29.97 が 30 に丸まると、1 時間で 3 フレーム以上ずれる
        target = tmp_path / "proxy.mp4"
        assert create_proxy(sample_ntsc.path, target, height=120) is not None
        assert probe_media(target).video_streams[0].frame_rate == FrameRate(30000, 1001)

    def test_it_is_smaller_than_the_source(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        # 小さくならないなら、読む速さも変わらない
        target = tmp_path / "proxy.mp4"
        assert create_proxy(sample_av.path, target, height=120) is not None
        assert target.stat().st_size < sample_av.path.stat().st_size

    def test_it_has_no_sound(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        # 音は元の素材から混ぜる 控えに入れても使われないまま場所を取る
        target = tmp_path / "proxy.mp4"
        assert create_proxy(sample_av.path, target, height=120) is not None
        assert not probe_media(target).has_audio

    def test_a_taller_request_does_not_blow_it_up(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        # 元より大きくしても重くなるだけ 元の大きさで止める
        target = tmp_path / "proxy.mp4"
        assert create_proxy(sample_av.path, target, height=2160) is not None
        assert probe_media(target).video_streams[0].height == sample_av.height

    def test_a_rotated_source_keeps_its_direction(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """回転の印が付いた素材でも、向きと縦横が変わらない

        印はコンテナに付いていて控えへは引き継がれない 画素を回さずに写すと、
        スマホで撮った縦の映像が控えのときだけ横向きになる
        （デコーダは開いたファイルの印だけを見て回すため）
        """
        turned = make_rotated(tmp_path, "turned.mp4", sample_av.path, 90)
        target = tmp_path / "proxy.mp4"
        assert create_proxy(turned, target, height=120) is not None

        # 元の素材をデコーダに通したときの見た目（印のぶん回った後）
        source_frame = VideoDecoder(turned).frame_at(Fraction(0))
        proxy_frame = VideoDecoder(target).frame_at(Fraction(0))
        assert source_frame is not None and proxy_frame is not None
        expected = source_frame.shape[:2]
        actual = proxy_frame.shape[:2]
        assert actual[0] / actual[1] == pytest.approx(expected[0] / expected[1], abs=0.02), (
            f"縦横比が変わっている 元 {expected} 控え {actual}"
        )
        assert actual[0] == 120, f"見た目の高さが指定と違う {actual}"

    def test_cancelling_leaves_nothing_behind(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        """途中でやめたら**ファイルを残さない**

        書きかけを残すと、次に開いたときに「途中までしか映らない素材」を掴む
        """
        target = tmp_path / "proxy.mp4"
        assert create_proxy(sample_av.path, target, height=120, should_cancel=lambda: True) is None
        assert list(tmp_path.iterdir()) == []

    def test_cancelling_after_the_last_frame_still_discards(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """最後の 1 枚を書いたあとにやめても、控えを置かない

        置くと、素材を外したり控えを切ったりしたのに控えが残り、
        「止めたはずなのに使われている」ことになる
        """
        stopped = False
        original = proxy_module._transcode

        def then_stop(
            source: Path,
            target: Path,
            *,
            height: int,
            stream_index: int | None,
            codec: str,
            progress: Callable[[float], None] | None,
            should_cancel: Callable[[], bool] | None,
        ) -> bool:
            """変換を終わらせてから、やめると言う 狙った隙間を必ず通る"""
            nonlocal stopped
            result = original(
                source,
                target,
                height=height,
                stream_index=stream_index,
                codec=codec,
                progress=progress,
                should_cancel=should_cancel,
            )
            stopped = True
            return result

        proxy_module._transcode = then_stop
        target = tmp_path / "proxy.mp4"
        try:
            assert (
                create_proxy(sample_av.path, target, height=120, should_cancel=lambda: stopped)
                is None
            )
        finally:
            proxy_module._transcode = original
        assert not target.exists(), "やめたのに控えを置いている"
        assert list(tmp_path.iterdir()) == [], "書きかけが残っている"

    def test_cancelling_stops_before_the_next_codec(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """やめると言われたら、次の候補を試さない

        試すと候補の数だけ変換を始め直すので、止めたのに止まらないように見える
        （素材を外したときや窓を閉じるときの効きが悪くなる）
        """
        tried: list[str] = []
        original = proxy_module._transcode

        # 本物と同じ形で受ける 省略して型の検査を外すと、本物の引数が
        # 変わったときにここが気付かず、素通りする試験になる
        def counting(
            source: Path,
            target: Path,
            *,
            height: int,
            stream_index: int | None,
            codec: str,
            progress: Callable[[float], None] | None,
            should_cancel: Callable[[], bool] | None,
        ) -> bool:
            tried.append(codec)
            return original(
                source,
                target,
                height=height,
                stream_index=stream_index,
                codec=codec,
                progress=progress,
                should_cancel=should_cancel,
            )

        proxy_module._transcode = counting
        try:
            assert (
                create_proxy(
                    sample_av.path, tmp_path / "proxy.mp4", height=120, should_cancel=lambda: True
                )
                is None
            )
        finally:
            proxy_module._transcode = original
        assert len(tried) == 1, f"やめたのに {len(tried)} 個の候補を試した: {tried}"

    def test_it_reports_progress(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        # 控えを作る進み具合が 100% まで行かないか途中で戻ると、
        # 変換が終わったのか止まったのか見分けられない
        seen: list[float] = []
        create_proxy(sample_av.path, tmp_path / "proxy.mp4", height=120, progress=seen.append)
        assert seen and seen[-1] == 1.0
        assert seen == sorted(seen), "進み具合が戻っている"

    def test_an_empty_result_is_not_kept(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        """1 枚も出せない控えは置かない

        映像の見出しはあるのに復号できない素材では、変換そのものは成功して
        中身の無い控えができる 置くと、描く側が捨てて作り直しを頼み、
        また同じものができる、の繰り返しになる
        """
        import sashimono.engine.cache.proxy as proxy_module

        target = tmp_path / "proxy.mp4"
        original = proxy_module._has_a_frame
        proxy_module._has_a_frame = lambda path: False
        try:
            assert create_proxy(sample_av.path, target, height=120) is None
        finally:
            proxy_module._has_a_frame = original
        assert not target.exists(), "使えない控えを置いている"
        assert list(tmp_path.iterdir()) == [], "書きかけが残っている"

    def test_a_broken_source_is_not_an_error(self, tmp_path: Path) -> None:
        """壊れた素材で例外を投げない

        投げると、1 本壊れただけで素材の読み込み全体が止まる
        """
        broken = tmp_path / "broken.mp4"
        broken.write_text("これは動画ではない", encoding="utf-8")
        assert create_proxy(broken, tmp_path / "proxy.mp4", height=120) is None


class TestTheStore:
    def test_it_is_absent_until_made(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        # 作る前から控えがあると答えると無いファイルを開き、
        # 作ったあとも見つけられないと重い元の素材で再生し続ける
        store = ProxyStore(CacheStore(tmp_path), height=120)
        media = _media(sample_av.path, sample_av.width, sample_av.height)
        assert store.find(media) is None
        assert create_proxy(sample_av.path, store.prepare(media), height=120) is not None
        assert store.find(media) == store.path_for(media)

    def test_an_empty_file_is_not_used(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        """空のファイルは作りかけか失敗の跡 掴むと映らない素材になる"""
        store = ProxyStore(CacheStore(tmp_path), height=120)
        media = _media(sample_av.path, sample_av.width, sample_av.height)
        store.prepare(media).touch()
        assert store.find(media) is None

    def test_a_different_height_is_a_different_key(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        # 設定を変えたときに、前の大きさの控えを掴まない
        media = _media(sample_av.path, sample_av.width, sample_av.height)
        small = ProxyStore(CacheStore(tmp_path), height=120)
        large = ProxyStore(CacheStore(tmp_path), height=360)
        assert small.key_for(media) != large.key_for(media)

    def test_a_changed_source_is_a_different_key(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """素材を差し替えたら別の鍵 古い控えを掴むと、差し替えが反映されない"""
        copy = tmp_path / "copy.mp4"
        copy.write_bytes(sample_av.path.read_bytes())
        store = ProxyStore(CacheStore(tmp_path), height=120)
        before = store.key_for(_media(copy, sample_av.width, sample_av.height))
        copy.write_bytes(sample_av.path.read_bytes() + b"\0")
        assert store.key_for(_media(copy, sample_av.width, sample_av.height)) != before


class TestTheRendererUsesIt:
    """控えはプレビューだけ 書き出しは必ず元の素材から読む"""

    def _project(self, media_path: Path) -> Project:
        media = probe_media(media_path)
        project = Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
        project = AddMedia(media).apply(project)
        track = Track(kind=TrackKind.VIDEO, name="V1")
        project = AddTrack(track).apply(project)
        return AddClip(track.id, Clip(timeline_start=0, duration=60, media_id=media.id)).apply(
            project
        )

    @pytest.fixture
    def shelf(self, sample_av: SampleMedia, tmp_path: Path) -> ProxyStore:
        """控えを 1 本作った置き場"""
        store = ProxyStore(CacheStore(tmp_path), height=120)
        media = probe_media(sample_av.path)
        assert create_proxy(sample_av.path, store.prepare(media), height=120) is not None
        return store

    def test_it_reads_the_proxy(
        self, sample_av: SampleMedia, shelf: ProxyStore, gl_context: OffscreenGLContext
    ) -> None:
        """控えがあれば、そちらを開く

        ここが元の素材のままなら、控えを作った意味が無い
        """
        renderer = FrameRenderer(self._project(sample_av.path), context=gl_context, proxies=shelf)
        try:
            renderer.render(0)
            opened = [decoder.info for decoder in renderer._decoders.values()]
        finally:
            renderer.close()
        assert opened and opened[0].height == 120

    def test_without_a_store_it_reads_the_source(
        self, sample_av: SampleMedia, gl_context: OffscreenGLContext
    ) -> None:
        """控えを渡さないときは元の素材 書き出しがこの道を通る

        既定で控えを使うと、低解像度の絵が黙って最終出力に入る
        """
        renderer = FrameRenderer(self._project(sample_av.path), context=gl_context)
        try:
            renderer.render(0)
            opened = [decoder.info for decoder in renderer._decoders.values()]
        finally:
            renderer.close()
        assert opened and opened[0].height == sample_av.height

    def test_the_picture_is_in_the_same_place(
        self, sample_av: SampleMedia, shelf: ProxyStore, gl_context: OffscreenGLContext
    ) -> None:
        """控えでも絵の位置と大きさは変わらない

        縮めた絵をそのまま貼ると、画面の隅に小さく映る
        """
        project = self._project(sample_av.path)
        plain = FrameRenderer(project, context=gl_context)
        try:
            expected = plain.render(0)
        finally:
            plain.close()
        cheap = FrameRenderer(project, context=gl_context, proxies=shelf)
        try:
            actual = cheap.render(0)
        finally:
            cheap.close()

        assert actual.shape == expected.shape
        # 細部は粗くなるが、絵としては同じもの 位置がずれれば平均の差が跳ね上がる
        difference = float(
            np.abs(actual[:, :, :3].astype(np.int16) - expected[:, :, :3].astype(np.int16)).mean()
        )
        assert difference < 24, f"絵が変わっている 平均の差 {difference:.1f}"

    def test_a_second_video_stream_reads_the_source(
        self, sample_av: SampleMedia, shelf: ProxyStore, gl_context: OffscreenGLContext
    ) -> None:
        """控えに入っているのは**1 本目の映像**だけ

        2 本目を指しているクリップに渡すと、別の絵が映る
        """
        probed = probe_media(sample_av.path)
        first = probed.video_streams[0]
        # 映像が 2 本ある素材 実際に 2 本入ったファイルを作らなくても、
        # 「1 本目でなければ渡さない」判断はこの形で確かめられる
        second = replace(first, index=first.index + 1)
        media = replace(probed, video_streams=(first, second))

        project = Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
        project = AddMedia(media).apply(project)
        track = Track(kind=TrackKind.VIDEO, name="V1")
        project = AddTrack(track).apply(project)
        clip = Clip(timeline_start=0, duration=60, media_id=media.id, stream_index=second.index)
        project = AddClip(track.id, clip).apply(project)

        renderer = FrameRenderer(project, context=gl_context, proxies=shelf)
        try:
            renderer.render(0)
            opened = [decoder.info for decoder in renderer._decoders.values()]
        finally:
            renderer.close()
        assert opened and opened[0].height == sample_av.height

    def test_reopening_picks_up_a_new_proxy(
        self, sample_av: SampleMedia, tmp_path: Path, gl_context: OffscreenGLContext
    ) -> None:
        """あとからできた控えに切り替わる

        先にプレビューした素材はデコーダを掴んだままなので、開き直させないと
        控えができても元の素材を読み続ける
        """
        store = ProxyStore(CacheStore(tmp_path), height=120)
        project = self._project(sample_av.path)
        renderer = FrameRenderer(project, context=gl_context, proxies=store)
        try:
            renderer.render(0)
            assert [d.info.height for d in renderer._decoders.values()] == [sample_av.height]
            assert create_proxy(sample_av.path, store.prepare(project.media[0]), height=120)
            renderer.reopen_sources()
            renderer.render(1)
            assert [d.info.height for d in renderer._decoders.values()] == [120]
        finally:
            renderer.close()

    def test_audio_first_media_still_uses_the_proxy(
        self, sample_av: SampleMedia, shelf: ProxyStore, gl_context: OffscreenGLContext
    ) -> None:
        """音が先に入っている素材でも控えを使う

        映像が 1 番から始まる素材では、クリップの既定の 0 が映像の番号と
        一致しない 番号で比べるだけだと、控えがあるのに黙って使われない
        （デコーダは番号が当たらなければ 1 本目の映像へ落ちるので、
        絵は映るが遅いまま 気付きにくい）
        """
        probed = probe_media(sample_av.path)
        # 映像が 1 番にある素材に見せかける（音が 0 番）
        media = replace(probed, video_streams=(replace(probed.video_streams[0], index=1),))
        project = Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
        project = AddMedia(media).apply(project)
        track = Track(kind=TrackKind.VIDEO, name="V1")
        project = AddTrack(track).apply(project)
        # stream_index は既定のまま（0）
        project = AddClip(track.id, Clip(timeline_start=0, duration=60, media_id=media.id)).apply(
            project
        )

        renderer = FrameRenderer(project, context=gl_context, proxies=shelf)
        try:
            renderer.render(0)
            opened = [decoder.info for decoder in renderer._decoders.values()]
        finally:
            renderer.close()
        assert opened and opened[0].height == 120, "控えがあるのに使っていない"

    def test_reopening_only_touches_the_named_media(
        self, sample_av: SampleMedia, shelf: ProxyStore, gl_context: OffscreenGLContext
    ) -> None:
        """名指しした素材のデコーダだけ閉じる

        全部閉じると、別の素材の控えができるたびに再生中のクリップまで
        開き直しとシークが走り、素材の本数だけ再生が途切れる
        """
        project = self._project(sample_av.path)
        renderer = FrameRenderer(project, context=gl_context, proxies=shelf)
        try:
            renderer.render(0)
            assert renderer._decoders, "前提が崩れている デコーダが開いていない"
            renderer.reopen_sources([MediaId("ほかの素材")])
            assert renderer._decoders, "関係の無い素材まで閉じている"
            renderer.reopen_sources([project.media[0].id])
            assert not renderer._decoders, "名指しした素材が閉じていない"
        finally:
            renderer.close()

    def test_a_truncated_proxy_is_thrown_away(
        self, sample_av: SampleMedia, shelf: ProxyStore, gl_context: OffscreenGLContext
    ) -> None:
        """**見出しは読めるのに 1 枚も出せない**控えは捨てる

        開けるかどうかだけを見ていると、途中で切れたファイルを掴んだまま
        そのクリップだけ白く残る 捨てておけば次の求めで作り直せる
        """
        project = self._project(sample_av.path)
        media = project.media[0]
        # 見出しだけ残して切り落とす（書いている途中で落ちたときの形）
        whole = shelf.path_for(media).read_bytes()
        shelf.path_for(media).write_bytes(whole[: len(whole) // 4])

        renderer = FrameRenderer(project, context=gl_context, proxies=shelf)
        try:
            image = renderer.render(0)
            opened = [decoder.info for decoder in renderer._decoders.values()]
            discarded = renderer.take_discarded()
            assert renderer.take_discarded() == set(), "取り出したのに覚えたまま"
        finally:
            renderer.close()
        assert opened and opened[0].height == sample_av.height, "壊れた控えを掴んだまま"
        assert image[:, :, :3].max() > 0, "何も映っていない"
        assert shelf.find(media) is None, "使えない控えが残っている 作り直せない"
        assert discarded == {media.id}, "捨てたことを伝えていない 作り直しが頼まれない"

    def test_a_source_starting_after_zero_still_plays(
        self, sample_av: SampleMedia, tmp_path: Path, gl_context: OffscreenGLContext
    ) -> None:
        """先頭フレームの時刻が 0 より後の素材でも映る

        使える控えかどうかを時刻 0 の 1 枚で確かめているので、
        そこが None になる素材があると**無事な素材まで捨てる**ことになる
        （分割して書き出した素材は先頭が 0 より後ろにある）
        """
        late = make_delayed(tmp_path, "late.mp4", sample_av.path, 5.0)
        store = ProxyStore(CacheStore(tmp_path), height=120)
        project = self._project(late)
        assert create_proxy(late, store.prepare(project.media[0]), height=120) is not None

        renderer = FrameRenderer(project, context=gl_context, proxies=store)
        try:
            image = renderer.render(0)
            opened = [decoder.info for decoder in renderer._decoders.values()]
        finally:
            renderer.close()
        assert opened and opened[0].height == 120, "使える控えを捨てている"
        assert image[:, :, :3].max() > 0, "何も映っていない"
        assert store.find(project.media[0]) is not None, "使える控えを消している"

    def test_a_late_starting_proxy_keeps_the_same_clock(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """頭が 0 より後ろの素材の控えは、元と同じ原点から数えて同じ時刻に終わる（Issue #123）

        控えは映像だけを写す 元の素材の音は映像より 24ms 早く始まるので、原点を
        コンテナの頭（全ストリームの最小）に取ると、元と控えで原点が 24ms 食い違い、
        控えのときだけ絵が 1 つ前のフレームになる 原点を映像の頭に取るのはこのため
        """
        late = make_delayed(tmp_path, "late.mp4", sample_av.path, 5.0)
        made = create_proxy(late, tmp_path / "late-proxy.mp4", height=120)
        assert made is not None
        source_end = probe_media(late).video_streams[0].end_time
        proxy_end = probe_media(made).video_streams[0].end_time
        assert source_end == proxy_end == Fraction(2)

    def test_a_missing_proxy_falls_back_to_the_source(
        self, sample_av: SampleMedia, tmp_path: Path, gl_context: OffscreenGLContext
    ) -> None:
        # まだ作れていない素材は元のまま映る 映らなくなるのが一番まずい
        empty = ProxyStore(CacheStore(tmp_path / "空"), height=120)
        renderer = FrameRenderer(self._project(sample_av.path), context=gl_context, proxies=empty)
        try:
            renderer.render(0)
            opened = [decoder.info for decoder in renderer._decoders.values()]
        finally:
            renderer.close()
        assert opened and opened[0].height == sample_av.height


class TestBuildingInTheBackground:
    """控えは裏で作る UI を止めると、素材を置いた瞬間に固まる"""

    def _wait(self, builder: ProxyBuilder, ready: list[MediaId], seconds: float = 60.0) -> None:
        limit = time.monotonic() + seconds
        while not ready and time.monotonic() < limit:
            time.sleep(0.02)

    def test_it_makes_one(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        # 裏で頼んだ控えができないか、できたと知らされないと、
        # プレビューが重い元の素材のまま切り替わらない
        store = ProxyStore(CacheStore(tmp_path), height=120)
        builder = ProxyBuilder(store)
        media = replace(probe_media(sample_av.path), video_streams=_tall(sample_av.path))
        ready: list[MediaId] = []
        try:
            builder.request(media, on_ready=ready.append)
            self._wait(builder, ready)
        finally:
            builder.close()
        assert ready == [media.id]
        assert store.find(media) is not None

    def test_a_small_source_is_left_alone(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        """1080p 以下は作らない 待たされるだけで速くならない"""
        store = ProxyStore(CacheStore(tmp_path), height=120)
        builder = ProxyBuilder(store)
        media = probe_media(sample_av.path)
        try:
            builder.request(media)
            assert builder.progress(media.id) is None
        finally:
            builder.close()
        assert store.find(media) is None

    def test_asking_twice_makes_one(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        # 2 本走ると、同じ場所を 2 つの変換が取り合う
        store = ProxyStore(CacheStore(tmp_path), height=120)
        builder = ProxyBuilder(store)
        media = replace(probe_media(sample_av.path), video_streams=_tall(sample_av.path))
        ready: list[MediaId] = []
        try:
            builder.request(media, on_ready=ready.append)
            builder.request(media, on_ready=ready.append)
            self._wait(builder, ready)
            time.sleep(0.2)
        finally:
            builder.close()
        assert ready == [media.id], "同じ素材の変換が 2 本走った"

    def test_it_reports_progress_while_running(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        # 裏で作っている控えの進み具合が出ないか、作り終えても残ると、作り続けているように見える
        store = ProxyStore(CacheStore(tmp_path), height=120)
        builder = ProxyBuilder(store)
        media = replace(probe_media(sample_av.path), video_streams=_tall(sample_av.path))
        seen: list[float] = []
        ready: list[MediaId] = []
        try:
            builder.request(
                media,
                on_ready=ready.append,
                on_progress=lambda media_id: seen.append(builder.progress(media_id) or -1.0),
            )
            self._wait(builder, ready)
        finally:
            builder.close()
        assert seen and max(seen) == 1.0
        # 終わったら進み具合は消える 残すと、UI が作り続けているように見える
        assert builder.progress(media.id) is None

    def test_closing_while_running_does_not_report_ready(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """止めたあとに変換が終わっても、できたとは伝えない

        伝えると、窓を閉じている最中や控えを切った直後に「控えができた」と
        して描き直しが走る

        変換の中身は差し替える 本物を使うと、走り始める前に止まるか
        止める前に終わるかが機械の速さ次第になり、狙った隙間を通らない
        """
        started = threading.Event()
        release = threading.Event()
        original = proxy_module.create_proxy

        def blocking(source: Path, target: Path, **kwargs: object) -> Path:
            started.set()
            release.wait(30.0)
            return target

        proxy_module.create_proxy = blocking
        store = ProxyStore(CacheStore(tmp_path), height=120)
        builder = ProxyBuilder(store)
        media = replace(probe_media(sample_av.path), video_streams=_tall(sample_av.path))
        ready: list[MediaId] = []
        try:
            builder.request(media, on_ready=ready.append)
            assert started.wait(30.0), "変換が始まらない"
            builder.close()
            release.set()
            limit = time.monotonic() + 30.0
            while builder.progress(media.id) is not None and time.monotonic() < limit:
                time.sleep(0.02)
            time.sleep(0.3)
        finally:
            release.set()
            proxy_module.create_proxy = original
        assert ready == [], "止めたのに、できたと伝えている"

    def test_closing_does_not_raise(self, sample_av: SampleMedia, tmp_path: Path) -> None:
        """止めたあとに頼んでも落ちない

        窓を閉じる途中に素材を外すと、この順で呼ばれる
        """
        builder = ProxyBuilder(ProxyStore(CacheStore(tmp_path), height=120))
        media = replace(probe_media(sample_av.path), video_streams=_tall(sample_av.path))
        builder.close()
        builder.close()
        builder.request(media)
        builder.forget(media.id)
        assert builder.progress(media.id) is None
