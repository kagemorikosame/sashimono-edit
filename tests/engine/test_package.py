"""持ち出し用のパッケージ（#423）

書き出して開き直す往復で、作品の形と、素材の使う範囲の絵が元と同じかを見る
素材はその場で ffmpeg で作る
"""

from __future__ import annotations

import hashlib
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
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    GeneratedSource,
    MediaId,
    MediaItem,
    Project,
    Track,
    TrackKind,
)
from sashimono.effects.spec import FileSpec
from sashimono.engine.decode import AudioDecoder, VideoDecoder, probe_media
from sashimono.engine.package import archive, plan, remux
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


def _frames_equal(a: Path, b: Path, spans: list[Span], *, index: int | None = None) -> bool:
    """使う範囲の絵がコマごとに同じで、音も標本ごとに同じか（音の道があれば）

    絵だけを比べると、音のパケットを写し漏らしても通ってしまう
    """
    with VideoDecoder(a, index) as left, VideoDecoder(b, index) as right:
        for span in spans:
            moment = span.start
            while moment <= span.end:
                one, two = left.frame_at(moment), right.frame_at(moment)
                if one is None or two is None or not np.array_equal(one, two):
                    return False
                moment += Fraction(1001, 30000)
    return _sound_equal(a, b, spans)


def _sound_equal(a: Path, b: Path, spans: list[Span]) -> bool:
    for stream in probe_media(a).audio_streams:
        with (
            AudioDecoder(a, sample_rate=stream.sample_rate, stream_index=stream.index) as left,
            AudioDecoder(b, sample_rate=stream.sample_rate, stream_index=stream.index) as right,
        ):
            for span in spans:
                one = left.read_seconds(span.start, span.end - span.start)
                two = right.read_seconds(span.start, span.end - span.start)
                # 解けなかった所は無音で埋まり、両方とも無音なら同じに見える
                if left.decode_error is not None or right.decode_error is not None:
                    return False
                if one.shape != two.shape or not np.allclose(one, two, atol=1e-4):
                    return False
    return True


def _rewrite(package: Path, change: object) -> None:
    """パッケージの作品と目録を書き換える（細工したパッケージを作る） SHA-256 は合わせ直す"""
    assert callable(change)
    with zipfile.ZipFile(package) as zipped:
        files = {name: zipped.read(name) for name in zipped.namelist()}
    manifest = json.loads(files["manifest.json"])
    project = json.loads(files["project.sme"])
    change(manifest, project)
    data = json.dumps(project, ensure_ascii=False).encode("utf-8")
    files["project.sme"] = data
    manifest["project"]["size"] = len(data)
    manifest["project"]["sha256"] = hashlib.sha256(data).hexdigest()
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(package, "w") as zipped:
        for name, content in files.items():
            zipped.writestr(name, content)


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

    def test_keeping_metadata_does_not_drop_trailing_streams(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        """残すと選んだのに後ろの道（字幕・位置の記録）を落とすと選んだ意味が無い まるごと入れる"""
        source = media_dir / "package-trailing.mkv"
        if not source.exists():
            subtitles = media_dir / "package-trailing.srt"
            subtitles.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
            _ffmpeg("-i", str(_video(media_dir)), "-i", str(subtitles), "-map", "0", "-map", "1",
                    "-c", "copy", str(source))  # fmt: skip
        target = tmp_path / "kept.mkv"
        span = [Span(Fraction(2), Fraction(3))]
        assert not cut_media(source, target, span, muxer="matroska", keep_metadata=True)
        assert cut_media(source, target, span, muxer="matroska")

    def test_every_picture_stream_keeps_its_frames(self, media_dir: Path, tmp_path: Path) -> None:
        """2 本目の映像の道はキーフレームの位置が違う 最初の道で切ると 2 本目の頭の絵が欠ける"""
        # 先に元の動画を作る libx264 の無い ffmpeg では、ここで飛ぶ（下の libx264 で落とさない）
        base = _video(media_dir)
        source = media_dir / "package-two-pictures.mkv"
        if not source.exists():
            other = media_dir / "package-other-gop.mp4"
            _ffmpeg("-f", "lavfi", "-i", "testsrc=size=96x64:rate=30000/1001:duration=8",
                    "-c:v", "libx264", "-g", "45", "-pix_fmt", "yuv420p", str(other))  # fmt: skip
            _ffmpeg("-i", str(base), "-i", str(other), "-map", "0:v", "-map", "1:v",
                    "-map", "0:a", "-c", "copy", str(source))  # fmt: skip
        item = probe_media(source)
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(
            _project([item], [_clip(item, 0, 30, Fraction(33, 10))]),
            package,
            PackOptions(margin=Fraction(0)),
            scratch=tmp_path / "s",
        )
        moved = load_project(open_package(package, tmp_path / "opened").project_path).media[0]
        span = [Span(Fraction(33, 10), Fraction(43, 10))]
        assert _frames_equal(source, moved.path, span, index=1)

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

    def test_stopping_while_checking_leaves_nothing(self, media_dir: Path, tmp_path: Path) -> None:
        """写し終えた後の確かめも止められる 止められないと、窓を閉じても比べ終わるまで待たされる"""
        source = _video(media_dir)
        span = [Span(Fraction(1), Fraction(2))]
        checking = False
        real = remux._same_as_source

        def spy(*arguments: object) -> bool:
            # 確かめに入ったら止めるよう頼む（写す間は止めない）
            nonlocal checking
            checking = True
            return real(*arguments)  # type: ignore[arg-type]

        target = tmp_path / "checking.mp4"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(remux, "_same_as_source", spy)
            with pytest.raises(Cancelled):
                cut_media(source, target, span, muxer="mp4", cancelled=lambda: checking)
        assert checking and not target.exists()

    def test_surround_sound_is_checked_in_every_channel(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        """2 つへ混ぜて比べると、混ぜると消える LFE の欠けを見落とす 道のチャンネルのまま比べる"""
        source = media_dir / "package-surround.mp4"
        if not source.exists():
            _ffmpeg("-i", str(_video(media_dir)), "-f", "lavfi", "-i",
                    "sine=frequency=220:duration=8", "-filter_complex",
                    "[1:a]pan=5.1|FL=c0|FR=c0|FC=c0|LFE=c0|BL=c0|BR=c0[a]", "-map", "0:v",
                    "-map", "[a]", "-c:v", "copy", "-c:a", "aac", str(source))  # fmt: skip
        target = tmp_path / "surround.mp4"
        assert cut_media(source, target, [Span(Fraction(2), Fraction(3))], muxer="mp4")
        assert probe_media(target).audio_streams[0].channels == 6

    def test_a_missing_lfe_is_noticed(self, tmp_path: Path) -> None:
        """LFE だけが違う 2 本は、2 つへ混ぜると同じに見える 道のチャンネルのまま比べれば違う"""
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg が無いので素材を作れない")
        paths = []
        for name, lfe in (("with.wav", "c0"), ("without.wav", "0*c0")):
            path = tmp_path / name
            _ffmpeg("-f", "lavfi", "-i", "sine=frequency=60:duration=2", "-filter_complex",
                    f"pan=5.1|FL=c0|FR=c0|FC=c0|LFE={lfe}|BL=c0|BR=c0", str(path))  # fmt: skip
            paths.append(path)
        stream = probe_media(paths[0]).audio_streams[0]
        span = [Span(Fraction(0), Fraction(1))]
        assert remux._same_sound(paths[0], paths[0], stream.index, stream.sample_rate, 6, span,
                                 lambda: False)  # fmt: skip
        assert not remux._same_sound(paths[0], paths[1], stream.index, stream.sample_rate, 6,
                                     span, lambda: False)  # fmt: skip

    def test_unreadable_pictures_are_not_trusted(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """両方とも絵が読めないと、どちらも無いので同じに見える 確かめられないとする"""
        monkeypatch.setattr(VideoDecoder, "decode_error", property(lambda _self: "壊れた"))
        target = tmp_path / "broken-picture.mp4"
        assert not cut_media(
            _video(media_dir), target, [Span(Fraction(2), Fraction(3))], muxer="mp4"
        )
        assert not target.exists()

    def test_unreadable_sound_is_not_trusted(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """解けなかった所は無音で埋まる 両方とも無音なら同じに見えるので、確かめられないとする"""
        monkeypatch.setattr(AudioDecoder, "decode_error", property(lambda _self: "壊れた"))
        target = tmp_path / "broken.mp4"
        assert not cut_media(
            _video(media_dir), target, [Span(Fraction(2), Fraction(3))], muxer="mp4"
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

    def test_a_held_picture_still_keeps_its_sound(self, media_dir: Path) -> None:
        """絵を止めても音は長さの分だけ読み続ける 止めた時刻で切ると持ち出した先で音が欠ける"""
        item = probe_media(_video(media_dir))
        clip = replace(_clip(item, 0, 90, Fraction(1)), hold_at=Fraction(3, 2))
        found = media_spans(_project([item], [clip]), Fraction(0))[item.id]
        assert found is not None and found[0].start == 1 and found[-1].end >= 3

    def test_fonts_and_scripts_are_listed(self, media_dir: Path) -> None:
        """書体と配布物のスクリプトは入れない 一覧に出ないと、開いた人が何が足りないかを知れない"""

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

    def test_an_unused_media_does_not_block_opening(self, media_dir: Path, tmp_path: Path) -> None:
        """素材一覧に使っていない素材があるだけで開けないと、書き出せたのに渡した先で開けない"""
        item = probe_media(_video(media_dir))
        spare = replace(probe_media(_picture(tmp_path)), id=MediaId("spare"))
        project = _project([item, spare], [_clip(item, 0, 30, Fraction(1))])
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        result = write_package(project, package, PackOptions(), scratch=tmp_path / "s")
        assert result.media_count == 1 and result.unused_media == ("模様.png",)
        opened = open_package(package, tmp_path / "opened")
        again = load_project(opened.project_path)
        assert [m.id for m in again.media] == [item.id, spare.id]
        assert str(tmp_path) not in str(again.media[1].path.relative_to(opened.folder))

    def test_an_unloaded_script_stops_the_export(self, media_dir: Path, tmp_path: Path) -> None:
        """読み込んでいないスクリプトはどの欄がファイルか分からず、元の PC の場所が残ったまま渡る"""
        item = probe_media(_video(media_dir))
        unknown = Effect(
            kind="aviutl:無い/無い.anm:無い", params={"file": "C:/Users/秘密/画像.png"}
        )
        clip = replace(_clip(item, 0, 30, Fraction(1)), effects=(unknown,))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        with pytest.raises(PackageError, match="読み込んでいないスクリプト"):
            write_package(_project([item], [clip]), package, PackOptions(), scratch=tmp_path / "s")
        assert not package.exists()

    def test_a_file_outside_the_package_is_not_followed(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        """細工した作品が共有のフォルダを指すと、描くたびにその機械へつないで認証を試みる"""
        item = probe_media(_video(media_dir))
        blend = Effect(kind="image_blend", params={"image_file": str(_picture(tmp_path))})
        clip = replace(_clip(item, 0, 30, Fraction(1)), effects=(blend,))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(_project([item], [clip]), package, PackOptions(), scratch=tmp_path / "s")

        def point_outside(manifest: dict[str, object], project: dict[str, object]) -> None:
            clips = project["timeline"]["tracks"][0]["clips"]  # type: ignore[index]
            clips[0]["effects"][0]["params"]["image_file"] = r"\\attacker\share\x.png"
            clips[0]["source"] = {
                "kind": "aviutl:無い/無い.anm:無い",
                "params": {
                    "f": "files/0002.png",
                    "g": "C:/Users/秘密/y.png",
                    "u": r"\\attacker\z",
                    "n": "files/0001.png",
                    "t": "/字幕",
                },
            }
            # 書き出した側が g をファイルの欄と記した（スクリプトの無い PC でも付け替える）
            # u は記さずに共有のフォルダを指させる（細工したパッケージ）
            fields = manifest["file_fields"]
            assert isinstance(fields, list)
            fields.append(f"source:aviutl:無い/無い.anm:無い:{clips[0]['id']}/g")

        _rewrite(package, point_outside)
        opened = open_package(package, tmp_path / "opened")
        clip = load_project(opened.project_path).timeline.tracks[0].clips[0]
        image = clip.effects[0].params["image_file"]
        assert isinstance(image, str) and Path(image).is_relative_to(opened.folder)
        assert clip.source is not None
        assert "attacker" not in str(clip.source)
        assert Path(str(clip.source.params["g"])) == opened.folder / "missing" / "y.png"
        # ファイルの欄と記していない文字は、場所に見えても、中の名前と同じでも文のまま残す
        assert clip.source.params["t"] == "/字幕"
        assert clip.source.params["n"] == "files/0001.png"

    def test_a_folder_field_stops_the_export(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """フォルダはまだ入れられない 書き出すと欄に元の PC の場所（ユーザー名）が残って渡る"""
        item = probe_media(_video(media_dir))
        effect = Effect(kind="image_blend", params={"image_file": "C:/Users/秘密/素材"})
        clip = replace(_clip(item, 0, 30, Fraction(1)), effects=(effect,))
        folder = (FileSpec("image_file", "フォルダ", directory=True),)
        real = plan._specs_of
        monkeypatch.setattr(
            plan,
            "_specs_of",
            lambda kind, *, source: folder if kind == "image_blend" else real(kind, source=source),
        )
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        with pytest.raises(PackageError, match="フォルダを指すエフェクト"):
            write_package(_project([item], [clip]), package, PackOptions(), scratch=tmp_path / "s")
        assert not package.exists()

    def test_a_file_for_a_missing_script_is_moved_too(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """開く PC にスクリプトが無くても、入れたファイルを指す欄は展開した先へ付け替える
        付け替えないと、後でスクリプトを入れても画像が見つからない"""
        item = probe_media(_video(media_dir))
        blend = Effect(kind="image_blend", params={"image_file": str(_picture(tmp_path))})
        clip = replace(_clip(item, 0, 30, Fraction(1)), effects=(blend,))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(_project([item], [clip]), package, PackOptions(), scratch=tmp_path / "s")
        # 開く PC には定義が無い（スクリプトを入れていない）
        real = plan._specs_of
        monkeypatch.setattr(
            plan,
            "_specs_of",
            lambda kind, *, source: None if kind == "image_blend" else real(kind, source=source),
        )
        opened = open_package(package, tmp_path / "opened")
        monkeypatch.undo()
        params = load_project(opened.project_path).timeline.tracks[0].clips[0].effects[0].params
        assert Path(str(params["image_file"])) == opened.folder / "files" / "0001.png"

    def test_the_media_keeps_its_name(self, media_dir: Path, tmp_path: Path) -> None:
        """中の名前は番号だけなので、名前を残さないと開いた先で 0001.mp4 としか出ない"""
        item = replace(probe_media(_video(media_dir)), display_name="")
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(
            _project([item], [_clip(item, 0, 30, Fraction(1))]),
            package,
            PackOptions(),
            scratch=tmp_path / "s",
        )
        moved = load_project(open_package(package, tmp_path / "opened").project_path).media[0]
        assert moved.display_name == item.path.name

    def test_sound_files_kept_as_they_are_are_reported(self, tmp_path: Path) -> None:
        """WAV も作った人の名前などを持ちうる 取り除けずに入れたのに黙ると、気付かずに渡す"""
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg が無いので素材を作れない")
        sound = tmp_path / "声.wav"
        _ffmpeg("-f", "lavfi", "-i", "sine=duration=2", "-metadata", "artist=秘密の人", str(sound))
        item = probe_media(sound)
        clip = _clip(item, 0, 30, Fraction(0))
        result = write_package(
            _project([item], [clip]),
            tmp_path / f"p{PACKAGE_SUFFIX}",
            PackOptions(),
            scratch=tmp_path / "s",
        )
        assert result.metadata_kept == ("声.wav",)

    def test_a_broken_font_list_leaves_nothing_behind(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        """展開の後に目録の形で落ちると、取り出したフォルダが残り、開き直すたびに増える"""
        item = probe_media(_video(media_dir))
        package = tmp_path / f"p{PACKAGE_SUFFIX}"
        write_package(
            _project([item], [_clip(item, 0, 30, Fraction(1))]),
            package,
            PackOptions(),
            scratch=tmp_path / "s",
        )

        def break_fonts(manifest: dict[str, object], _project: dict[str, object]) -> None:
            manifest["fonts"] = None

        _rewrite(package, break_fonts)
        with pytest.raises(PackageError, match="fonts"):
            open_package(package, tmp_path / "opened")
        opened = tmp_path / "opened"
        assert not opened.exists() or not list(opened.iterdir())

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

    def test_an_exif_after_the_tables_keeps_the_turn(self) -> None:
        """画素の表の後ろに EXIF がある写真で向きを落とすと、寝た絵になる"""
        original = _jpeg_with_exif(8)
        start = original.index(b"\xff\xe1")
        end = original.index(b"\xff\xdb")
        moved = original[:start] + original[end:-12] + original[start:end] + original[-12:]
        cleaned = scrub_image(moved, ".jpg")
        assert cleaned is not None and b"GPS-SECRET-PLACE" not in cleaned
        tiff = cleaned[cleaned.index(b"Exif\x00\x00") + 6 :]
        assert struct.unpack(">H", tiff[18:20])[0] == 8

    def test_metadata_between_scans_and_after_the_end_goes(self) -> None:
        """段階的な JPEG は走査の間にも注記を置ける 終わりの後ろの別の絵も EXIF を持つ"""
        original = _jpeg_with_exif(6)
        head = original[: original.index(b"\xff\xda")]
        comment = b"NAME-SECRET"
        xmp = b"http://ns.adobe.com/xap/1.0/\x00PLACE-SECRET"
        dht = b"\xff\xc4" + struct.pack(">H", 3) + b"\x00"
        # Adobe の APP14 は色の並びの印 走査の間にあっても、落とすと色が変わる
        adobe = b"\xff\xee" + struct.pack(">H", 14) + b"Adobe\x00\x64\x00\x00\x00\x00\x01"
        scans = (
            b"\xff\xda\x00\x02SCAN1\xff\x00a"
            + b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment
            + b"\xff\xe1" + struct.pack(">H", len(xmp) + 2) + xmp
            + dht
            + adobe
            + b"\xff\xda\x00\x02SCAN2\xff\xd0b"
            + b"\xff\xd9"
        )  # fmt: skip
        cleaned = scrub_image(head + scans + b"TRAILER-SECRET", ".jpg")
        assert cleaned is not None
        for secret in (b"NAME-SECRET", b"PLACE-SECRET", b"TRAILER-SECRET", b"GPS-SECRET-PLACE"):
            assert secret not in cleaned
        assert b"SCAN1\xff\x00a" in cleaned and dht in cleaned and adobe in cleaned
        assert cleaned.endswith(b"SCAN2\xff\xd0b\xff\xd9")

    def test_a_jpeg_without_its_end_is_left_alone(self) -> None:
        """終わりが無い（途中で切れた）JPEG は読み違えうる 取り除いたと言わず、そのまま入れる"""
        assert scrub_image(_jpeg_with_exif(6)[:-2], ".jpg") is None

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
