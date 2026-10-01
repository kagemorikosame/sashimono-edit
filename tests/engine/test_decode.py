"""実素材に対する解析とデコード

シークの正しさは「飛んだ結果が、先頭から順に読んだ結果と一致するか」でしか
確かめられない 参照列との厳密比較で押さえる
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import av.error
import numpy as np
import pytest

from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import AudioDecoder, ProbeError, VideoDecoder, probe_media
from sashimono.engine.decode.rational import as_fraction
from tests.media_fixtures import (
    SampleMedia,
    decode_all_frames,
    encoder_available,
    libx264_available,
    make_delayed,
    make_rotated,
    make_sample,
)


def _short_picture(directory: Path) -> Path:
    """映像 1 秒・音 2 秒の素材 音の方が長い素材の、映像の終わりの後を見るため"""
    if not libx264_available():
        pytest.skip("ffmpeg に libx264 が無いので実素材のテストを飛ばす")
    path = directory / "short-picture.mp4"
    if not path.exists():
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=44100",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(path),
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip
    return path


def _covered_mp3(directory: Path) -> Path:
    """カバー画像（attached_pic）の付いた 2 秒の mp3 音楽の配布物によくある形"""
    path = directory / "covered.mp3"
    if path.exists():
        return path
    # 飛ばすのは使えない環境だけ（ffmpeg が無い・使う符号化器を入れずに組み立てた）
    # 作る途中の失敗まで飛ばすと、引数の誤りでもカバー画像の試験が黙って走らなくなる
    # 符号化器は下の 2 つの命令で指定する物すべて（音の libmp3lame と絵の png）
    lacking = [name for name in ("libmp3lame", "png") if not encoder_available(name)]
    if lacking:
        pytest.skip(
            f"ffmpeg か符号化器（{', '.join(lacking)}）が無いのでカバー画像付きの mp3 を作れない"
        )
    # 画像を先に 1 枚作ってから重ねる 1 回で作ろうと -frames:v 1 を付けると、
    # 出力全体がその 1 枚の長さで切れて音が 26ms しか残らない
    cover = directory / "cover.png"
    steps = [
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:size=32x32", "-frames:v", "1", str(cover),
        ],
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=44100",
            "-i", str(cover), "-map", "0:a", "-map", "1:v",
            "-c:a", "libmp3lame", "-c:v", "png", "-disposition:v:0", "attached_pic",
            "-id3v2_version", "3", str(path),
        ],
    ]  # fmt: skip
    for step in steps:
        made = subprocess.run(step, capture_output=True, check=False)
        if made.returncode != 0:
            pytest.fail(f"カバー画像付きの mp3 を作れない: {made.stderr.decode(errors='replace')}")
    return path


class TestProbe:
    def test_reads_video_and_audio(self, sample_av: SampleMedia) -> None:
        item = probe_media(sample_av.path)
        assert item.has_video
        assert item.has_audio
        assert item.name == "av.mp4"

        video = item.video_streams[0]
        assert (video.width, video.height) == (320, 240)
        assert video.frame_rate == FrameRate(30)
        assert video.codec == "h264"
        assert video.pixel_format == "yuv420p"

        audio = item.audio_streams[0]
        assert audio.sample_rate == 44100
        assert audio.channels == 2
        assert audio.codec == "aac"

    def test_duration_is_exact_rational(self, sample_av: SampleMedia) -> None:
        item = probe_media(sample_av.path)
        assert isinstance(item.duration, Fraction)
        assert item.duration == pytest.approx(2.0, abs=0.1)

    def test_fractional_frame_rate_is_preserved(self, sample_ntsc: SampleMedia) -> None:
        # 29.97 が 2997/100 に化けると 1 時間で 3 フレーム以上ずれる
        item = probe_media(sample_ntsc.path)
        assert item.video_streams[0].frame_rate == FrameRate(30000, 1001)

    def test_the_end_of_the_picture_is_read_apart_from_the_sound(self, media_dir: Path) -> None:
        """音の方が長い素材で、映像の道の終わりをコンテナの長さと分けて取る（#115）

        コンテナの長さだけだと、最後の絵で止める時刻が映像の最後のフレームより後ろになり、
        止めた後もデコーダが毎フレーム終わり付近を読み直す
        """
        item = probe_media(_short_picture(media_dir))
        end = item.video_streams[0].end_time
        assert end is not None
        assert float(end) == pytest.approx(1.0, abs=0.05)
        assert item.duration > end + Fraction(1, 2)

    def test_a_late_starting_file_is_measured_from_its_head(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """頭が 5 秒の素材の長さと映像の終わりは、元の素材と同じ 2 秒（Issue #123）

        終わりを PTS そのまま（7 秒）で持つと、最後の絵で止める時刻（``Clip.hold_at``）が
        頭から数えるデコーダの映像の終わりを越え、止めたはずの所で何も映らない
        長さに音の前置き（映像より 24ms 早く始まる）を含めると、置いたクリップが
        1 フレーム長くなり、最後の 1 フレームが何も映らない
        """
        late = probe_media(make_delayed(tmp_path, "late.mp4", sample_av.path, 5.0))
        assert late.video_streams[0].end_time == Fraction(2)
        assert late.duration == Fraction(2)

    def test_the_cover_art_of_an_mp3_is_not_a_video(self, media_dir: Path) -> None:
        """mp3 に付いたカバー画像（attached_pic）は映像ストリームに数えない

        数えると mp3 が動画として扱われ、置くと映像トラックへ絵として置かれて描かれる
        絵が 1 枚しか無いのにデコーダが時刻でシークし、途中の時刻で落ちる
        """
        item = probe_media(_covered_mp3(media_dir))
        assert not item.has_video
        assert item.has_audio
        # 原点と長さもカバー画像の時刻で決めない 音の長さ（2 秒）のまま
        assert float(item.duration) == pytest.approx(2.0, abs=0.1)

    def test_video_only_media(self, sample_long: SampleMedia) -> None:
        item = probe_media(sample_long.path)
        assert item.has_video
        assert not item.has_audio

    def test_rotation_is_detected(self, media_dir: Path, sample_av: SampleMedia) -> None:
        # スマホの縦撮り素材を想定 無視すると横倒しで表示される
        rotated = make_rotated(media_dir, "rot90.mp4", sample_av.path, 90)
        item = probe_media(rotated)
        stream = item.video_streams[0]
        assert stream.rotation == 270
        # 回転を適用すると表示サイズは縦横が入れ替わる
        assert stream.display_size == (240, 320)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ProbeError, match="見つからない"):
            probe_media(tmp_path / "無い.mp4")

    def test_not_media(self, tmp_path: Path) -> None:
        path = tmp_path / "notmedia.mp4"
        path.write_bytes(b"not a media file at all")
        with pytest.raises(ProbeError, match="開けない"):
            probe_media(path)


class TestVideoDecoder:
    def test_an_mp3_with_cover_art_has_no_picture_to_decode(self, media_dir: Path) -> None:
        # カバー画像を映像として開くと、2 枚目の無い絵の中をシークして PermissionError で落ちる
        # 映像の無い素材として断れば、呼ぶ側はほかの音だけの素材と同じに扱える
        with pytest.raises(ProbeError, match="映像ストリームが無い"):
            VideoDecoder(_covered_mp3(media_dir))

    def test_reads_the_first_frame(self, sample_av: SampleMedia) -> None:
        with VideoDecoder(sample_av.path) as decoder:
            frame = decoder.frame_at(Fraction(0))
        assert frame is not None
        assert frame.shape == (240, 320, 4)
        assert frame.dtype == np.uint8

    def test_sequential_reads_match_the_reference(self, sample_long: SampleMedia) -> None:
        reference = decode_all_frames(sample_long.path)
        with VideoDecoder(sample_long.path) as decoder:
            for index in range(0, 40):
                seconds = Fraction(index, 30)
                frame = decoder.frame_at(seconds)
                assert frame is not None, f"{index} フレーム目が読めない"
                assert np.array_equal(frame, reference[index][1]), f"{index} フレーム目が不一致"

    def test_seek_backwards_lands_exactly(self, sample_long: SampleMedia) -> None:
        # 前方へ飛んでから戻る シークがキーフレームまで戻ってから前進デコードする経路
        reference = decode_all_frames(sample_long.path)
        with VideoDecoder(sample_long.path) as decoder:
            for index in (100, 5, 77, 0, 43):
                frame = decoder.frame_at(Fraction(index, 30))
                assert frame is not None, f"{index} フレーム目が読めない"
                assert np.array_equal(frame, reference[index][1]), f"{index} フレーム目が不一致"

    def test_frame_is_held_until_the_next_one(self, sample_long: SampleMedia) -> None:
        # フレームの表示は次のフレームが来るまで続く その間はどの時刻でも同じ絵
        with VideoDecoder(sample_long.path) as decoder:
            at_start = decoder.frame_at(Fraction(10, 30))
            midway = decoder.frame_at(Fraction(10, 30) + Fraction(1, 90))
        assert at_start is not None
        assert midway is not None
        assert np.array_equal(at_start, midway)

    def test_a_late_starting_file_is_timed_from_its_head(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """頭が 5 秒の素材の時刻は頭から数える 元の素材と同じ時刻に同じフレーム（Issue #123）

        PTS そのままで数えると 0〜2 秒はすべて最初のフレームより前で、最初の絵が止まったまま
        飛ぶ順を前後させて、シークも原点を足した PTS へ着地することを見る
        """
        reference = decode_all_frames(sample_av.path)
        late = make_delayed(tmp_path, "late.mp4", sample_av.path, 5.0)
        with VideoDecoder(late) as decoder:
            for index in (0, 1, 45, 30, 59, 2):
                frame = decoder.frame_at(Fraction(index, 30))
                assert frame is not None, f"{index} フレーム目が読めない"
                assert np.array_equal(frame, reference[index][1]), f"{index} フレーム目が不一致"
            assert decoder.frame_at(Fraction(2)) is None

    def test_a_late_file_shows_no_picture_where_only_its_sound_goes_on(
        self, media_dir: Path, tmp_path: Path
    ) -> None:
        """頭 5 秒・映像 1 秒・音 2 秒の素材で、映像の終わり（頭から 1 秒）の後は絵を出さない

        コンテナの終わり（頭から 2 秒）まで出すと、音だけの区間に直前の絵が静止画で残る
        最後の絵を出し続けたいクリップは ``hold_at`` で止める（Issue #115）
        """
        late = make_delayed(tmp_path, "late-short.mp4", _short_picture(media_dir), 5.0)
        with VideoDecoder(late) as decoder:
            assert decoder.frame_at(Fraction(1, 2)) is not None
            assert decoder.frame_at(Fraction(1) - Fraction(1, 1000)) is not None
            assert decoder.frame_at(Fraction(3, 2)) is None

    def test_a_file_whose_sound_runs_longer_shows_no_picture_after_its_picture_ends(
        self, media_dir: Path
    ) -> None:
        # 頭が 0 の素材も同じ 音だけの区間に最後の絵を出すと、映像トラックに置いた動画が
        # 映像の終わりの後も止まった絵のまま、下の層を隠し続ける
        with VideoDecoder(_short_picture(media_dir)) as decoder:
            assert decoder.frame_at(Fraction(1) - Fraction(1, 1000)) is not None
            assert decoder.frame_at(Fraction(3, 2)) is None

    def test_past_the_end_returns_none(self, sample_av: SampleMedia) -> None:
        with VideoDecoder(sample_av.path) as decoder:
            assert decoder.frame_at(Fraction(10)) is None

    def test_negative_time_clamps_to_the_start(self, sample_av: SampleMedia) -> None:
        with VideoDecoder(sample_av.path) as decoder:
            first = decoder.frame_at(Fraction(0))
            before = decoder.frame_at(Fraction(-5))
        assert first is not None
        assert before is not None
        assert np.array_equal(first, before)

    def test_rotation_is_applied(self, media_dir: Path, sample_av: SampleMedia) -> None:
        rotated = make_rotated(media_dir, "rot90b.mp4", sample_av.path, 90)
        with VideoDecoder(rotated) as decoder:
            frame = decoder.frame_at(Fraction(0))
        assert frame is not None
        # 320x240 が縦向きになる 回転を無視していれば (240, 320, 4) のまま
        assert frame.shape == (320, 240, 4)

    def test_media_without_video(self, media_dir: Path) -> None:
        audio_only = make_sample(
            media_dir, "audio_only.m4a", duration=1.0, audio=True, pattern="testsrc2"
        )
        # 映像を含まないファイルを作るため、音声だけ抜き出したものを使う
        import subprocess

        stripped = media_dir / "stripped.m4a"
        if not stripped.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(audio_only.path),
                    "-vn",
                    "-c:a",
                    "copy",
                    str(stripped),
                ],
                check=True,
                capture_output=True,
            )
        with pytest.raises(ProbeError, match="映像ストリームが無い"):
            VideoDecoder(stripped)


class TestAudioDecoder:
    def test_reads_requested_length(self, sample_av: SampleMedia) -> None:
        with AudioDecoder(sample_av.path, sample_rate=48000, channels=2) as decoder:
            samples = decoder.read(0, 4800)
        assert samples.shape == (4800, 2)
        assert samples.dtype == np.float32

    def test_resamples_to_the_requested_rate(self, sample_av: SampleMedia) -> None:
        # 素材は 44100Hz プロジェクトが 48000Hz ならここで揃える
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            assert decoder.info.sample_rate == 44100
            assert decoder.sample_rate == 48000
            one_second = decoder.read(0, 48000)
        assert one_second.shape == (48000, 2)
        # lavfi の sine は振幅が小さいので、無音でないことだけを確かめる
        assert np.abs(one_second).max() > 0.01

    def test_reads_are_contiguous(self, sample_av: SampleMedia) -> None:
        # 分けて読んでも、続けて読んだのと同じ波形になること
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            whole = decoder.read(0, 24000)
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            first = decoder.read(0, 12000)
            second = decoder.read(12000, 12000)
        assert np.allclose(whole, np.concatenate([first, second]), atol=1e-6)

    def test_seek_returns_the_same_audio(self, sample_av: SampleMedia) -> None:
        # 頭から読んだ 1 秒地点と、飛んで読んだ 1 秒地点が同じ音であること
        #
        # 完全一致はしない シーク時にリサンプラを作り直すため、リサンプルの位相が
        # 1 サンプル未満ずれる 44100Hz から 48000Hz への変換ではサンプル境界が
        # 一致しないので、これは避けられない 可聴域の話ではないので、
        # 波形として同じかを相対 RMS 誤差で見る
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            sequential = decoder.read(0, 72000)[48000:]
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            decoder.read(0, 480)
            seeked = decoder.read(48000, 24000)

        signal = float(np.sqrt(np.mean(sequential**2)))
        error = float(np.sqrt(np.mean((sequential - seeked) ** 2)))
        assert error / signal < 0.05, f"相対 RMS 誤差 {error / signal:.3%}"

    def test_a_late_starting_file_sounds_from_its_head(
        self, sample_av: SampleMedia, tmp_path: Path
    ) -> None:
        """頭が 5 秒の素材の音は頭から数える 飛んで読んでも同じ所（Issue #123）

        PTS そのままで数えると、素材の長さ（2 秒）ぶんの範囲はすべて頭より前で無音になる
        シークだけ原点を足し忘れると、飛んだ先で 5 秒前の位置（無音）を読む
        """
        late = make_delayed(tmp_path, "late.mp4", sample_av.path, 5.0)
        with AudioDecoder(late, sample_rate=48000) as decoder:
            sequential = decoder.read(0, 72000)[48000:]
        with AudioDecoder(late, sample_rate=48000) as decoder:
            decoder.read(0, 480)
            seeked = decoder.read(48000, 24000)

        signal = float(np.sqrt(np.mean(sequential**2)))
        assert signal > 0.01, "頭から 1 秒の所が無音"
        error = float(np.sqrt(np.mean((sequential - seeked) ** 2)))
        assert error / signal < 0.05, f"相対 RMS 誤差 {error / signal:.3%}"

    def test_a_seek_near_the_head_reads_from_before_the_origin(
        self, sample_av: SampleMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """頭の近くへ飛ぶときも、原点より手前から余らせて読み始める（#124 のレビュー）

        素材の中の時刻で 0 に丸めてから原点を足すと、原点ちょうどへ飛ぶ 原点をまたぐ
        復号の単位から読めないコンテナでは、頭の音が欠ける
        """
        late = make_delayed(tmp_path, "late.mp4", sample_av.path, 5.0)
        seeks: list[int] = []
        with AudioDecoder(late, sample_rate=48000) as decoder:
            real = decoder._container

            class _Recording:
                def seek(self, offset: int, **kwargs: Any) -> None:
                    seeks.append(offset)
                    real.seek(offset, **kwargs)

                def __getattr__(self, name: str) -> Any:
                    return getattr(real, name)

            # PyAV のコンテナへ委ねるだけの代わり 型は違うので monkeypatch で差し替える
            monkeypatch.setattr(decoder, "_container", _Recording())
            decoder._seek(4800)
            monkeypatch.setattr(decoder, "_container", real)
            assert decoder._stream.time_base is not None
            time_base = as_fraction(decoder._stream.time_base)
            origin = decoder._origin
        # 0.1 秒の所へ飛ぶなら、余らせる 0.25 秒を引いた原点の 0.15 秒手前から
        assert seeks == [int((origin + Fraction(1, 10) - Fraction(1, 4)) / time_base)]

    def test_a_decode_failure_midway_is_remembered(self, sample_av: SampleMedia) -> None:
        """途中で復号に失敗したら、無音で返しつつ理由を覚える

        覚えないと終わりと見分けが付かず、読んだ音を丸ごと使う字幕起こしが、欠けた音を
        そのまま使って成功したように見える
        """
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            real = decoder._frames

            def breaking() -> Any:
                yield next(real)
                raise av.error.InvalidDataError(1094995529, "Invalid data")

            decoder._frames = breaking()
            assert decoder.decode_error is None
            sound = decoder.read(0, 48000)
            assert decoder.decode_error is not None
            assert "Invalid data" in decoder.decode_error
            # 失敗した位置は読めた所の終わり（最初の 1 フレーム分）
            assert decoder.decode_error_at is not None
            assert 0 < decoder.decode_error_at < 24000
            assert float(np.abs(sound[: decoder.decode_error_at]).max()) > 0.0
        # 失敗した所から先は無音
        assert float(np.abs(sound[24000:]).max()) == 0.0

    def test_before_the_start_is_silent(self, sample_av: SampleMedia) -> None:
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            samples = decoder.read(-1000, 2000)
        assert np.all(samples[:1000] == 0.0)
        assert np.abs(samples[1000:]).max() > 0.0

    def test_past_the_end_is_silent(self, sample_av: SampleMedia) -> None:
        # 短い配列を返すと呼び出し側が毎回長さを揃える羽目になる 必ず要求長で返す
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            samples = decoder.read(48000 * 10, 4800)
        assert samples.shape == (4800, 2)
        assert np.all(samples == 0.0)

    def test_mono_downmix(self, sample_av: SampleMedia) -> None:
        with AudioDecoder(sample_av.path, sample_rate=48000, channels=1) as decoder:
            samples = decoder.read(0, 4800)
        assert samples.shape == (4800, 1)

    def test_read_seconds(self, sample_av: SampleMedia) -> None:
        with AudioDecoder(sample_av.path, sample_rate=48000) as decoder:
            samples = decoder.read_seconds(Fraction(1, 2), Fraction(1, 4))
        assert samples.shape == (12000, 2)

    def test_rejects_bad_arguments(self, sample_av: SampleMedia) -> None:
        with pytest.raises(ValueError, match="サンプリングレート"):
            AudioDecoder(sample_av.path, sample_rate=0)
        with pytest.raises(ValueError, match="チャンネル数"):
            AudioDecoder(sample_av.path, sample_rate=48000, channels=3)

    def test_media_without_audio(self, sample_long: SampleMedia) -> None:
        with pytest.raises(ProbeError, match="音声ストリームが無い"):
            AudioDecoder(sample_long.path, sample_rate=48000)
