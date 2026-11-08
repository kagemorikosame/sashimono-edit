"""持ち出し用のパッケージ（#423）

書き出して開き直す往復で、作品の形と、素材の使う範囲の絵が元と同じかを見る
素材はその場で ffmpeg で作る
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import zipfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from sashimono.core.io.serialize import load_project
from sashimono.core.model import AnimatedValue, Clip, Effect, MediaItem, Project, Track, TrackKind
from sashimono.engine.decode import VideoDecoder, probe_media
from sashimono.engine.package import archive
from sashimono.engine.package.archive import (
    PACKAGE_SUFFIX,
    PackageError,
    PackOptions,
    open_package,
    write_package,
)
from sashimono.engine.package.plan import fonts_used, media_spans, scripts_used
from sashimono.engine.package.remux import Cancelled, Span, cut_media, output_format
from sashimono.engine.package.scrub import scrub_image
from tests.media_fixtures import libx264_available, make_delayed, make_rotated


def _ffmpeg(*arguments: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        capture_output=True,
    )


def _video(directory: Path, name: str = "package-long.mp4", seconds: int = 8) -> Path:
    """GOP 1 秒・B フレームありの 8 秒の動画 撮った場所と機器の名前を持たせる"""
    if not libx264_available():
        pytest.skip("ffmpeg に libx264 が無いので実素材のテストを飛ばす")
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        _ffmpeg(
            "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate=30000/1001:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-g", "30", "-bf", "2", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-metadata", "location=+35.6586+139.7454/", "-metadata", "make=TestPhone",
            str(path),
        )  # fmt: skip
    return path


def _frames_equal(a: Path, b: Path, spans: list[Span]) -> bool:
    with VideoDecoder(a) as left, VideoDecoder(b) as right:
        for span in spans:
            moment = span.start
            while moment <= span.end:
                one, two = left.frame_at(moment), right.frame_at(moment)
                if one is None or two is None or not np.array_equal(one, two):
                    return False
                moment += Fraction(1001, 30000)
    return True


def _project(media: list[MediaItem], clips: list[Clip]) -> Project:
    project = Project.create(media=tuple(media), name="持ち出しの試験")
    track = Track(kind=TrackKind.VIDEO, name="映像 1", clips=tuple(clips))
    return replace(project, timeline=replace(project.timeline, tracks=(track,)))


def _clip(item: MediaItem, start: int, frames: int, source_in: Fraction) -> Clip:
    return Clip(timeline_start=start, duration=frames, media_id=item.id, source_in=source_in)


class TestCut:
    def test_the_used_range_keeps_every_pixel_and_shrinks_the_file(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        """使う範囲の絵が 1 画素でも違えば品質の約束が崩れる 小さくならなければ切る意味が無い"""
        source = _video(media_dir)
        spans = [Span(Fraction(3), Fraction(4))]
        muxer, extension = output_format(source) or ("", "")
        target = tmp_path / f"cut{extension}"
        assert cut_media(source, target, spans, muxer=muxer)
        assert target.stat().st_size < source.stat().st_size * 0.6
        assert _frames_equal(source, target, spans)

    def test_two_far_ranges_share_one_file(self, media_dir: Path, tmp_path: Path) -> None:
        """離れた 2 つの範囲を 1 つに入れたとき後ろの範囲の絵が崩れると後ろのクリップが壊れる"""
        source = _video(media_dir)
        spans = [Span(Fraction(1), Fraction(3, 2)), Span(Fraction(6), Fraction(13, 2))]
        muxer, extension = output_format(source) or ("", "")
        target = tmp_path / f"two{extension}"
        assert cut_media(source, target, spans, muxer=muxer)
        assert _frames_equal(source, target, spans)

    def test_matroska_keeps_the_frames_too(self, media_dir: Path, tmp_path: Path) -> None:
        """matroska は時刻を 1ms に丸める 丸めで隣の絵を出すなら写した後の確かめで見つける"""
        source = media_dir / "package-long.mkv"
        if not source.exists():
            _ffmpeg("-i", str(_video(media_dir)), "-c", "copy", str(source))
        spans = [Span(Fraction(2), Fraction(5, 2)), Span(Fraction(5), Fraction(6))]
        assert output_format(source) == ("matroska", ".mkv")
        target = tmp_path / "cut.mkv"
        assert cut_media(source, target, spans, muxer="matroska")
        assert _frames_equal(source, target, spans)

    def test_a_late_starting_media_keeps_its_clock(self, media_dir: Path, tmp_path: Path) -> None:
        """頭が 0 より後ろの素材で原点がずれると、作品の中の時刻がすべて別の絵を指す"""
        source = make_delayed(media_dir, "package-late.mp4", _video(media_dir), 2.5)
        spans = [Span(Fraction(3), Fraction(4))]
        target = tmp_path / "late.mp4"
        assert cut_media(source, target, spans, muxer="mp4")
        assert _frames_equal(source, target, spans)

    def test_rotation_stays_and_the_location_goes(self, media_dir: Path, tmp_path: Path) -> None:
        """回転を落とすと縦の動画が寝る 撮った場所を残すと、渡した相手に住まいが分かる"""
        source = make_rotated(media_dir, "package-turned.mp4", _video(media_dir), 90)
        target = tmp_path / "turned.mp4"
        assert cut_media(source, target, [Span(Fraction(2), Fraction(3))], muxer="mp4")
        assert (
            probe_media(target).video_streams[0].rotation
            == probe_media(source).video_streams[0].rotation
            != 0
        )
        with av.open(str(source)) as before, av.open(str(target)) as after:
            assert "location" in before.metadata
            assert "location" not in after.metadata
            assert "make" not in after.metadata

    def test_keeping_metadata_keeps_the_location(self, media_dir: Path, tmp_path: Path) -> None:
        """残すと選んだのに消すと、選んだ意味が無い"""
        source = _video(media_dir)
        target = tmp_path / "kept.mp4"
        assert cut_media(
            source, target, [Span(Fraction(2), Fraction(3))], muxer="mp4", keep_metadata=True
        )
        with av.open(str(target)) as after:
            assert "location" in after.metadata

    def test_stopping_leaves_nothing(self, media_dir: Path, tmp_path: Path) -> None:
        """止めた後に書きかけが残ると、ディスクに使えないファイルが溜まる"""
        target = tmp_path / "stopped.mp4"
        with pytest.raises(Cancelled):
            cut_media(
                _video(media_dir),
                target,
                [Span(Fraction(1), Fraction(7))],
                muxer="mp4",
                cancelled=lambda: True,
            )
        assert not target.exists()


class TestPlan:
    def test_one_media_used_twice_is_one_range_list(self, media_dir: Path) -> None:
        """同じ素材を 2 回入れると大きさが倍になる 1 本にまとめ、範囲は両方を含める"""
        item = probe_media(_video(media_dir))
        project = _project(
            [item], [_clip(item, 0, 30, Fraction(1)), _clip(item, 30, 30, Fraction(6))]
        )
        spans = media_spans(project, Fraction(0))
        assert list(spans) == [item.id]
        found = spans[item.id]
        assert found is not None and len(found) == 2
        assert found[0].start == 1 and found[1].start == 6

    def test_a_reverb_reaches_back_before_the_clip(self, media_dir: Path) -> None:
        """リバーブは頭より前の音を読む 入れないと別の PC で響きの頭が欠ける"""
        item = probe_media(_video(media_dir))
        clip = _clip(item, 0, 30, Fraction(5))
        reverb = Effect(kind="audio_reverb", params={"decay": AnimatedValue(static=2.0)})
        project = _project([item], [replace(clip, effects=(reverb,))])
        found = media_spans(project, Fraction(0))[item.id]
        assert found is not None and found[0].start <= 3

    def test_fonts_and_scripts_are_listed(self, media_dir: Path) -> None:
        """書体と配布物のスクリプトは入れない 一覧に出ないと、開いた人が何が足りないかを知れない"""
        from sashimono.core.model import GeneratedSource

        text = Clip(
            timeline_start=0,
            duration=30,
            source=GeneratedSource(kind="text", params={"text": "題", "font": "游明朝"}),
            effects=(Effect(kind="aviutl:アニメーション効果/震える.anm:震える", params={}),),
        )
        project = _project([], [text])
        assert fonts_used(project) == ("游明朝",)
        assert scripts_used(project) == ("アニメーション効果/震える.anm",)


class TestRoundTrip:
    def test_the_package_opens_as_the_same_work(self, media_dir: Path, tmp_path: Path) -> None:
        """往復でクリップの位置・長さ・エフェクトが変わると、持ち出した先で別の作品になる"""
        item = probe_media(_video(media_dir))
        blend = Effect(kind="image_blend", params={"image_file": str(_picture(tmp_path))})
        clips = [
            _clip(item, 0, 30, Fraction(2)),
            replace(_clip(item, 40, 15, Fraction(6)), effects=(blend,)),
        ]
        project = _project([item], clips)
        package = tmp_path / f"持ち出し{PACKAGE_SUFFIX}"
        result = write_package(
            project, package, PackOptions(margin=Fraction(0)), scratch=tmp_path / "scratch"
        )
        assert result.media_count == 1 and result.file_count == 1
        assert package.stat().st_size < item.path.stat().st_size

        opened = open_package(package, tmp_path / "opened")
        again = load_project(opened.project_path)
        assert [
            (c.timeline_start, c.duration, c.source_in) for c in again.timeline.tracks[0].clips
        ] == [(c.timeline_start, c.duration, c.source_in) for c in clips]
        moved = again.media[0]
        assert moved.id == item.id and moved.path.is_file()
        assert moved.path.is_relative_to(opened.folder)
        image = again.timeline.tracks[0].clips[1].effects[0].params["image_file"]
        assert isinstance(image, str) and Path(image).is_relative_to(opened.folder)
        assert Path(image).read_bytes() == _picture(tmp_path).read_bytes()
        spans = [Span(Fraction(2), Fraction(3)), Span(Fraction(6), Fraction(13, 2))]
        assert _frames_equal(item.path, moved.path, spans)

    def test_the_package_hides_the_original_folders(self, media_dir: Path, tmp_path: Path) -> None:
        """元のフォルダにはユーザー名が入る 渡した相手に見せない"""
        item = probe_media(_video(media_dir))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(
            _project([item], [_clip(item, 0, 30, Fraction(1))]),
            package,
            PackOptions(),
            scratch=tmp_path / "s",
        )
        with zipfile.ZipFile(package) as zipped:
            for name in zipped.namelist():
                if name.endswith((".sme", ".json")):
                    assert str(item.path.parent) not in zipped.read(name).decode("utf-8")
            assert all(info.compress_type == zipfile.ZIP_STORED for info in zipped.infolist())

    def test_a_broken_package_is_not_opened(self, media_dir: Path, tmp_path: Path) -> None:
        """壊れた素材のまま開くと、編集の途中で絵が崩れて原因が分からない 展開した物も残さない"""
        item = probe_media(_video(media_dir))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(
            _project([item], [_clip(item, 0, 30, Fraction(1))]),
            package,
            PackOptions(),
            scratch=tmp_path / "s",
        )
        data = bytearray(package.read_bytes())
        with zipfile.ZipFile(package) as zipped:
            media = next(info for info in zipped.infolist() if info.filename.startswith("media/"))
            offset = (
                media.header_offset + 30 + len(media.filename.encode()) + len(media.extra) + 100
            )
        data[offset] ^= 0xFF
        package.write_bytes(bytes(data))
        with pytest.raises(PackageError):
            open_package(package, tmp_path / "opened")
        assert not any((tmp_path / "opened").iterdir())

    def test_a_cut_short_package_is_not_opened(self, media_dir: Path, tmp_path: Path) -> None:
        """途中で切れたパッケージ（写しかけ）を開くと、素材の足りない作品になる"""
        item = probe_media(_video(media_dir))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(
            _project([item], [_clip(item, 0, 30, Fraction(1))]),
            package,
            PackOptions(),
            scratch=tmp_path / "s",
        )
        package.write_bytes(package.read_bytes()[: package.stat().st_size // 2])
        with pytest.raises(PackageError):
            open_package(package, tmp_path / "opened")

    def test_a_newer_package_asks_for_an_update(self, tmp_path: Path) -> None:
        """読めない形を推して開くと黙って何かを落とす 更新を頼む"""
        package = tmp_path / f"new{PACKAGE_SUFFIX}"
        with zipfile.ZipFile(package, "w") as zipped:
            zipped.writestr(
                "manifest.json",
                json.dumps({"format": "sashimono-package", "version": 9, "minimum": 9}),
            )
        with pytest.raises(PackageError, match="更新"):
            open_package(package, tmp_path / "opened")

    def test_a_name_that_leaves_the_folder_is_refused(self, tmp_path: Path) -> None:
        """``..`` を含む名前を展開すると、展開先の外のファイルを書き換えられる"""
        package = tmp_path / f"evil{PACKAGE_SUFFIX}"
        manifest = {
            "format": "sashimono-package",
            "version": 1,
            "minimum": 1,
            "project": {"path": "project.sme", "size": 0, "sha256": "0" * 64},
            "entries": [{"path": "../../evil.txt", "size": 1, "sha256": "0" * 64}],
        }
        with zipfile.ZipFile(package, "w") as zipped:
            zipped.writestr("manifest.json", json.dumps(manifest))
            zipped.writestr("../../evil.txt", "x")
        with pytest.raises(PackageError, match="扱えない名前"):
            open_package(package, tmp_path / "opened")
        assert not (tmp_path.parent / "evil.txt").exists()

    def test_too_little_space_is_refused_before_writing(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """書いている途中で空きが尽きると、時間を掛けた後で失敗する 先に断る"""
        item = probe_media(_video(media_dir))
        monkeypatch.setattr(archive, "_free_space", lambda _folder: 1024)
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        with pytest.raises(PackageError, match="空きが足りません"):
            write_package(
                _project([item], [_clip(item, 0, 30, Fraction(1))]),
                package,
                PackOptions(),
                scratch=tmp_path / "s",
            )
        assert not package.exists()

    def test_too_little_space_is_refused_before_extracting(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """展開の途中で空きが尽きると、半分だけ取り出した物が残る 先に断る"""
        item = probe_media(_video(media_dir))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(
            _project([item], [_clip(item, 0, 30, Fraction(1))]),
            package,
            PackOptions(),
            scratch=tmp_path / "s",
        )
        monkeypatch.setattr(archive, "_free_space", lambda _folder: 1024)
        with pytest.raises(PackageError, match="空きが足りません"):
            open_package(package, tmp_path / "opened")

    def test_a_missing_media_file_is_refused(self, media_dir: Path, tmp_path: Path) -> None:
        """素材の無い作品をまとめると、開いた先でも絵が出ない 何が無いかを言って断る"""
        source = tmp_path / "消える.mp4"
        shutil.copyfile(_video(media_dir), source)
        item = probe_media(source)
        source.unlink()
        with pytest.raises(PackageError, match="消える"):
            write_package(
                _project([item], [_clip(item, 0, 30, Fraction(1))]),
                tmp_path / f"p{PACKAGE_SUFFIX}",
                PackOptions(),
                scratch=tmp_path / "s",
            )

    def test_whole_media_lets_trims_grow(self, media_dir: Path, tmp_path: Path) -> None:
        """まるごとを選んだのに範囲だけ入ると、範囲の外へトリムを伸ばせない"""
        item = probe_media(_video(media_dir))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(
            _project([item], [_clip(item, 0, 30, Fraction(1))]),
            package,
            PackOptions(whole=True),
            scratch=tmp_path / "s",
        )
        opened = open_package(package, tmp_path / "opened")
        moved = load_project(opened.project_path).media[0]
        assert _frames_equal(item.path, moved.path, [Span(Fraction(0), Fraction(7))])


def _picture(directory: Path) -> Path:
    path = directory / "模様.png"
    if not path.exists():
        _ffmpeg("-f", "lavfi", "-i", "color=c=red:size=8x8", "-frames:v", "1", str(path))
    return path


def _jpeg_with_exif(orientation: int) -> bytes:
    """向きと撮った場所（GPS の札）を持つ EXIF の付いた小さな JPEG"""
    entries = [
        struct.pack(">HHI", 0x0112, 3, 1) + struct.pack(">HH", orientation, 0),
        struct.pack(">HHII", 0x8825, 4, 1, 0),
    ]
    tiff = (
        b"MM\x00\x2a" + struct.pack(">I", 8) + struct.pack(">H", len(entries)) + b"".join(entries)
    )
    tiff += struct.pack(">I", 0) + b"GPS-SECRET-PLACE"
    body = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body
    jfif = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    dqt = b"\xff\xdb" + struct.pack(">H", 67) + b"\x00" + bytes(64)
    return b"\xff\xd8" + jfif + app1 + dqt + b"\xff\xda\x00\x02PIXELS\xff\xd9"


class TestScrub:
    def test_a_jpeg_loses_the_place_but_keeps_the_turn(self) -> None:
        """場所が残ると渡した相手に住まいが分かる 向きが消えると縦の写真が寝る"""
        cleaned = scrub_image(_jpeg_with_exif(6), ".jpg")
        assert cleaned is not None
        assert b"GPS-SECRET-PLACE" not in cleaned
        assert b"Exif\x00\x00" in cleaned
        position = cleaned.index(b"Exif\x00\x00") + 6
        tiff = cleaned[position:]
        assert struct.unpack(">HHIHH", tiff[10:22])[0] == 0x0112
        assert struct.unpack(">H", tiff[18:20])[0] == 6
        assert cleaned.endswith(b"PIXELS\xff\xd9")

    def test_an_upright_jpeg_gets_no_exif(self) -> None:
        """向きが既定なら EXIF ごと要らない 空の EXIF を足すと読めない読み手がある"""
        cleaned = scrub_image(_jpeg_with_exif(1), ".jpg")
        assert cleaned is not None and b"Exif" not in cleaned

    def test_a_png_loses_its_text(self) -> None:
        """PNG の文字の塊にも撮った場所や作った人の名前が入りうる"""
        import zlib

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data))
            )

        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", bytes(13)) + chunk(b"tEXt", b"Author\x00secret")
        png += chunk(b"IDAT", b"pixels") + chunk(b"IEND", b"")
        cleaned = scrub_image(png, ".png")
        assert cleaned is not None and b"secret" not in cleaned and b"pixels" in cleaned
