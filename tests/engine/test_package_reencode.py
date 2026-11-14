"""持ち出し用パッケージで、編集用の大きな形式を符号にし直して入れる（#446）

素材は ffmpeg で作る（ProRes・DNxHR・無圧縮・透明つきの ProRes 4444） どれも PCM の音を持たせる
（ProRes の .mov はたいてい PCM の音を持つ）
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import pytest

from sashimono.core.io.serialize import load_project
from sashimono.core.model import Clip, MediaItem, Project, Track, TrackKind, VideoStreamInfo
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import AudioDecoder, probe_media
from sashimono.engine.decode.rational import as_fraction
from sashimono.engine.package.archive import (
    PACKAGE_SUFFIX,
    PackOptions,
    PackResult,
    open_package,
    write_package,
)
from sashimono.engine.package.plan import editing_media
from sashimono.engine.package.remux import (
    REENCODE_LEAST_PSNR,
    Reencoded,
    Span,
    cut_media,
    frame_psnr,
    reencodable,
    reencode_media,
)
from tests.media_fixtures import encoder_available, libx264_available

_KINDS = {
    "prores.mov": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
    "dnxhr.mov": ["-c:v", "dnxhd", "-profile:v", "dnxhr_hq", "-pix_fmt", "yuv422p"],
    "raw.mov": ["-c:v", "rawvideo", "-pix_fmt", "uyvy422"],
    "prores4444.mov": ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"],
}


def _editing(directory: Path, name: str) -> Path:
    """編集用の形式の 4 秒の素材（PCM の音つき）"""
    if not libx264_available():
        pytest.skip("libx264 が無いので符号にし直す試験を飛ばす")
    # 素材を作る符号化器も見る 無い組み立ての ffmpeg では、作る所で落ちて試験の失敗に見える
    maker = _KINDS[name][1]
    if not encoder_available(maker):
        pytest.skip(f"{maker} が無いので {name} を作れない")
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=256x144:rate=30000/1001:duration=4",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
             *_KINDS[name], "-c:a", "pcm_s16le", str(path)],
            check=True,
            capture_output=True,
        )  # fmt: skip
    return path


def _project(item: MediaItem, clip: Clip) -> Project:
    project = Project.create(media=(item,), name="符号の試験")
    track = Track(kind=TrackKind.VIDEO, name="映像 1", clips=(clip,))
    return replace(project, timeline=replace(project.timeline, tracks=(track,)))


def _frames(path: Path, moments: list[Fraction]) -> list[av.VideoFrame]:
    """その時刻に映る絵（YUV のまま）"""
    found = []
    with av.open(str(path)) as reader:
        stream = reader.streams.video[0]
        base = as_fraction(stream.time_base) if stream.time_base else Fraction(1, 1000)
        for moment in moments:
            reader.seek(int(moment / base), stream=stream, backward=True)
            shown = None
            for frame in reader.decode(stream):
                if frame.pts is not None and frame.pts * base > moment:
                    break
                shown = frame
            assert shown is not None
            found.append(shown)
    return found


def test_an_identical_picture_is_told_as_the_same() -> None:
    """元と同じ絵の差は無限大 「inf dB」と出すと何のことか分からない（目録は null にする）"""
    from sashimono.ui.package_dialog import _reencoded_text

    report = Reencoded("prores", "h264", "yuv422p10le", float("inf"), 100, 40)
    assert "元と同じ絵" in _reencoded_text("本編.mov", report)


@pytest.mark.parametrize(
    ("pixel_format", "expected"),
    [
        ("yuv422p10le", True),
        ("uyvy422", True),
        # 透明が消える・RGB の変換で色が変わる・x264 は 10bit までで深さが落ちる
        ("yuva444p10le", False),
        ("gbrp10le", False),
        ("yuv422p12le", False),
        ("gray10le", False),
        # 色差の間引きが x264 に無い形 寄せると色差を捨てるのに確かめの差に出ない
        ("yuv440p", False),
        ("yuv411p", False),
    ],
)
def test_only_formats_kept_as_they_are_are_offered(pixel_format: str, expected: bool) -> None:
    """し直せない形を窓に並べると、選んでも元のまま入り、小さくなる見込みが外れる"""
    assert reencodable(pixel_format) is expected
    stream = VideoStreamInfo(
        index=0,
        width=64,
        height=36,
        frame_rate=FrameRate(25, 1),
        time_base=Fraction(1, 25),
        codec="prores",
        pixel_format=pixel_format,
    )
    item = MediaItem(path=Path("D:/a.mov"), duration=Fraction(4), video_streams=(stream,))
    clip = Clip(timeline_start=0, duration=25, media_id=item.id, source_in=Fraction(0))
    assert bool(editing_media(_project(item, clip))) is expected


def test_two_editing_streams_are_not_offered() -> None:
    """片方がし直せない形でも、し直す側は 2 本と数えて断る 並べると選んでも元のまま入る"""

    def stream(index: int, pixel_format: str) -> VideoStreamInfo:
        return VideoStreamInfo(
            index=index,
            width=64,
            height=36,
            frame_rate=FrameRate(25, 1),
            time_base=Fraction(1, 25),
            codec="prores",
            pixel_format=pixel_format,
        )

    item = MediaItem(
        path=Path("D:/a.mov"),
        duration=Fraction(4),
        video_streams=(stream(0, "yuv422p10le"), stream(1, "yuva444p10le")),
    )
    clip = Clip(timeline_start=0, duration=25, media_id=item.id, source_in=Fraction(0))
    assert editing_media(_project(item, clip)) == ()


class TestReencode:
    @pytest.mark.parametrize("name", ["prores.mov", "dnxhr.mov", "raw.mov"])
    def test_the_same_time_shows_nearly_the_same_picture(
        self, name: str, media_dir: Path, tmp_path: Path
    ) -> None:
        """時刻がずれると、作品のクリップが別の絵を指す 差が大きいと持ち出した先で崩れが見える"""
        source = _editing(media_dir, name)
        target = tmp_path / "out.mov"
        span = Span(Fraction(1), Fraction(3))
        report = reencode_media(source, target, [span], muxer="mov")
        assert report is not None
        assert report.to == "h264" and report.psnr >= REENCODE_LEAST_PSNR
        # 映像は元より小さい（大きくなるなら入れる意味が無い）
        assert report.after < report.before
        moments = [Fraction(1), Fraction(2), Fraction(29, 10)]
        for old, new in zip(_frames(source, moments), _frames(target, moments), strict=True):
            assert old.pts == new.pts
            assert frame_psnr(old, new) >= REENCODE_LEAST_PSNR
            # 範囲と色の行列の印も元のまま（範囲の「決まっていない」は YUV ではテレビの範囲）
            assert (new.color_range or 1, new.colorspace) == (old.color_range or 1, old.colorspace)
        # 前後の絵を使って縮めている 解いた絵の I の型を残すと全部がキーフレームになる
        with av.open(str(target)) as reader:
            keys = [packet.is_keyframe for packet in reader.demux(reader.streams.video[0])]
        assert keys.count(False) > keys.count(True)
        # 色の深さと色差の間引きは元のまま（10bit 4:2:2 を 8bit 4:2:0 に落とさない）
        depth = {"prores.mov": "yuv422p10le", "dnxhr.mov": "yuv422p", "raw.mov": "yuv422p"}
        assert probe_media(target).video_streams[0].pixel_format == depth[name]

    def test_pcm_sound_stays_on_time(self, media_dir: Path, tmp_path: Path) -> None:
        """mov の書き手は PCM の空きを詰めるので、区間の音が前へずれて別の所の音が鳴る"""
        source = _editing(media_dir, "prores.mov")
        target = tmp_path / "cut.mov"
        span = Span(Fraction(2), Fraction(3))
        # 写す側（#449）も同じ所で落ちて、全体を入れていた
        assert cut_media(source, target, [span], muxer="mov")
        with (
            AudioDecoder(source, sample_rate=44100, channels=1, stream_index=1) as old,
            AudioDecoder(target, sample_rate=44100, channels=1, stream_index=1) as new,
        ):
            a = old.read_seconds(span.start, Fraction(1, 2))
            b = new.read_seconds(span.start, Fraction(1, 2))
        assert a.shape == b.shape and float(abs(a - b).max()) < 1e-4

    def test_pcm_between_two_ranges_stays_on_time_and_silent(self, tmp_path: Path) -> None:
        """離れた 2 つの区間の間を詰めると、後ろの区間の音が前へずれる 間の音を入れると、
        使っていない所の声まで持ち出す（区間の外は無音にして時刻だけ保つ）"""
        if not encoder_available("prores_ks"):
            pytest.skip("prores_ks が無いので素材を作れない")
        source = tmp_path / "chirp.mov"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=25:duration=6",
             "-f", "lavfi", "-i", "aevalsrc=0.5*sin(2*PI*(200+150*t)*t):s=44100:d=6",
             "-c:v", "prores_ks", "-c:a", "pcm_s16le", str(source)],
            check=True,
            capture_output=True,
        )  # fmt: skip
        target = tmp_path / "cut.mov"
        first, second = Span(Fraction(1, 2), Fraction(1)), Span(Fraction(4), Fraction(9, 2))
        assert cut_media(source, target, [first, second], muxer="mov")
        with (
            AudioDecoder(source, sample_rate=44100, channels=1, stream_index=1) as old,
            AudioDecoder(target, sample_rate=44100, channels=1, stream_index=1) as new,
        ):
            # 時とともに高さの変わる音なので、ずれれば後ろの区間の形が合わない
            a = old.read_seconds(second.start, Fraction(1, 2))
            b = new.read_seconds(second.start, Fraction(1, 2))
            assert a.shape == b.shape and float(abs(a - b).max()) < 1e-4
            # 区間の間は無音
            gap = new.read_seconds(Fraction(2), Fraction(1, 2))
            assert float(abs(gap).max()) < 1e-6
            # 区間の終わりの直後も無音 区間に掛かるパケットを丸ごと残すと、続きの言葉が入る
            after = new.read_seconds(first.end + Fraction(1, 20), Fraction(2, 5))
            assert float(abs(after).max()) < 1e-6
            assert (
                float(abs(old.read_seconds(first.end + Fraction(1, 20), Fraction(2, 5))).max())
                > 0.1
            )
            assert float(abs(old.read_seconds(Fraction(2), Fraction(1, 2))).max()) > 0.1

    def test_the_whole_media_is_checked_in_the_middle_and_near_the_end(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """全体を入れる区間（10 億秒）のまま測ると、真ん中も終わりも最後の絵どうしを比べる"""
        from sashimono.engine.package import remux

        source = _editing(media_dir, "prores.mov")
        moments: list[Fraction] = []
        original = remux._frame_at

        def spy(path: Path, index: int, moment: Fraction, cancelled: object) -> object:
            moments.append(moment)
            return original(path, index, moment, cancelled)  # type: ignore[arg-type]

        monkeypatch.setattr(remux, "_frame_at", spy)
        report = reencode_media(
            source, tmp_path / "x.mov", [Span(Fraction(0), Fraction(10**9))], muxer="mov"
        )
        assert report is not None
        length = probe_media(source).duration
        # 頭・真ん中・終わり近くの 3 か所を、素材の中で比べる
        assert sorted(set(moments)) == [Fraction(0), length / 2, length * 9 / 10]

    def test_transparent_pictures_are_left_as_they_are(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        """H.264 は透明を持てない し直すと、重ねたときに後ろが見えていた所が黒く塗られる"""
        source = _editing(media_dir, "prores4444.mov")
        assert (
            reencode_media(
                source, tmp_path / "x.mov", [Span(Fraction(1), Fraction(2))], muxer="mov"
            )
            is None
        )
        assert not (tmp_path / "x.mov").exists()


class TestPackage:
    def _pack(
        self, media_dir: Path, tmp_path: Path, options: PackOptions
    ) -> tuple[Path, PackResult]:
        item = probe_media(_editing(media_dir, "prores.mov"))
        clip = Clip(timeline_start=0, duration=45, media_id=item.id, source_in=Fraction(1))
        tmp_path.mkdir(parents=True, exist_ok=True)
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        result = write_package(_project(item, clip), package, options, scratch=tmp_path / "s")
        return package, result

    def test_reencoding_is_not_the_default(self, media_dir: Path, tmp_path: Path) -> None:
        """既定でし直すと、品質を保つ約束（#423）のまま書き出したつもりの人の絵が変わる"""
        package, result = self._pack(media_dir, tmp_path, PackOptions())
        assert result.reencoded == ()
        opened = open_package(package, tmp_path / "opened")
        assert load_project(opened.project_path).media[0].video_streams[0].codec == "prores"

    def test_chosen_reencoding_is_smaller_and_reported(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        """選んだのに知らせが無いと、画質が変わったことを開いた人も渡した人も知らない"""
        plain, _ = self._pack(media_dir, tmp_path / "a", PackOptions())
        package, result = self._pack(media_dir, tmp_path / "b", PackOptions(reencode=True))
        reencoded = result.reencoded
        assert [name for name, _report in reencoded] == ["prores.mov"]
        assert package.stat().st_size < plain.stat().st_size
        with zipfile.ZipFile(package) as zipped:
            manifest = json.loads(zipped.read("manifest.json"))
        media = next(entry for entry in manifest["entries"] if entry["role"] == "media")
        assert media["reencoded"]["from"] == "prores" and media["reencoded"]["to"] == "h264"
        assert media["reencoded"]["psnr_db"] >= REENCODE_LEAST_PSNR
        # 開いた先の素材の情報も中身に合わせる（ProRes のままだと控えの判断や表示が食い違う）
        opened = open_package(package, tmp_path / "opened")
        stream = load_project(opened.project_path).media[0].video_streams[0]
        assert (stream.codec, stream.pixel_format) == ("h264", "yuv422p10le")

    def test_the_dialog_lists_only_editing_formats(self, media_dir: Path) -> None:
        """普通の動画まで並ぶと、何が変わるのかが窓から分からない"""
        item = probe_media(_editing(media_dir, "prores.mov"))
        clip = Clip(timeline_start=0, duration=30, media_id=item.id)
        assert editing_media(_project(item, clip)) == ("prores.mov（ProRes）",)
        h264 = replace(
            item,
            video_streams=tuple(replace(s, codec="h264") for s in item.video_streams),
        )
        assert editing_media(_project(h264, clip)) == ()
