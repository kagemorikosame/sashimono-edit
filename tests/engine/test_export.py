"""書き出し

実際にファイルを作って、読み直せることまで確かめる 書き出しは最後の工程なので、
ここが壊れていると編集作業がまるごと無駄になる
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import av.codec.codec
import av.error
import av.video.codeccontext
import numpy as np
import pytest
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

from sashimono.core.commands import AddClip, AddTrack, Document, insert_media
from sashimono.core.model import (
    AnimatedValue,
    Blending,
    Clip,
    GeneratedSource,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import VideoDecoder, probe_media
from sashimono.engine.encode import (
    COLOR_OPTIONS,
    MAX_PIPELINE_DEPTH,
    ExportError,
    ExportSettings,
    available_video_codecs,
    export_project,
    exporter,
)
from tests.color_bars import (
    AVCOL_SPC_BT709,
    BT709_YUV,
    COLORS,
    assert_close,
    bars,
    rgb_at_bars,
    yuv_at_bars,
)
from tests.media_fixtures import SampleMedia, make_sample

# 書き出しは内部で GL コンテキストを作って合成する GPU の無い環境では
# 作れても使えないので、失敗ではなく飛ばす
pytestmark = pytest.mark.usefixtures("gpu")


@pytest.fixture
def ready_project(sample_av: SampleMedia) -> Project:
    """映像 + 音声を 1 本置いた 320x240 / 30fps のプロジェクト"""
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(sample_av.path)):
        document.execute(command)
    return document.project


@pytest.fixture
def silent_project(sample_long: SampleMedia) -> Project:
    """映像のみのプロジェクト 音声ストリームを作らない経路の確認用"""
    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(sample_long.path)):
        document.execute(command)
    return document.project


@pytest.fixture
def forget_probes() -> Iterator[None]:
    """開けるかの答えを、試験の前後で忘れる

    答えはプロセスの中で覚えている 残すと、偽の答えが後の試験へ漏れて、
    本物の NVENC が「開けない」扱いになる
    """
    exporter._opens.cache_clear()
    yield
    exporter._opens.cache_clear()


@pytest.fixture
def probed(monkeypatch: pytest.MonkeyPatch, forget_probes: None) -> list[str]:
    """開けない QSV を真似る 試しに開いたコーデックの名前を順に記録する

    この開発機と同じく、QSV は FFmpeg に入っているが ``avcodec_open2`` が 22 を返す
    ほかのコーデックは本物を開く
    """
    calls: list[str] = []
    real = exporter._open_encoder

    def fake(name: str) -> None:
        calls.append(name)
        if name == "h264_qsv":
            raise av.error.ArgumentError(22, "Invalid argument")
        real(name)

    monkeypatch.setattr(exporter, "_open_encoder", fake)
    monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("h264_qsv", "libx264"))
    return calls


def _encoder_of(path: Path) -> str:
    """書き出したファイルを作ったエンコーダ x264 はビットストリームに名前を残す"""
    return "libx264" if b"x264 - core" in path.read_bytes() else "libx264 以外"


class TestCodecs:
    def test_a_codec_that_does_not_open_is_not_offered(self, probed: list[str]) -> None:
        """入っているだけの QSV を候補に出さない（#67）

        出すと先頭に来て既定のコーデックになり、Intel の GPU が無い機械では
        書き出しを押した途端に失敗する
        """
        assert available_video_codecs() == ["libx264"]

    def test_each_codec_is_tried_only_once(self, probed: list[str]) -> None:
        """開けるかは 1 度だけ試して覚える

        毎回試すと、書き出しの画面を開くたびに NVENC を開く 100 ms 待たされる
        """
        available_video_codecs()
        available_video_codecs()
        assert probed == ["h264_qsv", "libx264"]

    def test_a_small_probe_does_not_drop_a_working_nvenc(self, forget_probes: None) -> None:
        """試しに開く大きさが小さすぎて、使える NVENC まで外すことがない

        NVENC は 64x64 を断る 試す大きさをそこまで縮めると、GPU で書き出せる機械でも
        CPU の libx264 しか選べなくなる
        """
        # 入っていても GPU が無ければ開けない それは環境の都合なので、
        # 書き出しでよく使う 1080p で開けると確かめてから比べる
        try:
            context = av.CodecContext.create("h264_nvenc", "w")
            assert isinstance(context, av.video.codeccontext.VideoCodecContext)
            context.width, context.height = 1920, 1080
            context.pix_fmt = "yuv420p"
            context.time_base = Fraction(1, 30)
            context.open()
        except (av.codec.codec.UnknownCodecError, av.error.FFmpegError) as exc:
            pytest.skip(f"h264_nvenc を 1080p で開けない: {exc}")
        assert "h264_nvenc" in available_video_codecs()

    def test_at_least_one_codec_is_available(self) -> None:
        # 書き出しの画面でコーデックの選択肢が空になり、どの作品も動画にできない
        assert available_video_codecs(), "書き出せるコーデックが 1 つも無い"

    def test_preference_order_is_respected(self) -> None:
        # GPU コーデックが使えるなら先に来る CPU より圧倒的に速い
        codecs = available_video_codecs()
        if "h264_nvenc" in codecs and "libx264" in codecs:
            assert codecs.index("h264_nvenc") < codecs.index("libx264")


class TestExport:
    @pytest.mark.parametrize("codec", ["libx264", "h264_nvenc"])
    def test_produces_a_playable_file(
        self, codec: str, ready_project: Project, tmp_path: Path
    ) -> None:
        # 書き出した mp4 が開けない・音が入らない・大きさやコマの速さが設定と違う、のどれかになる
        if codec not in available_video_codecs():
            pytest.skip(f"{codec} が使えない")

        output = tmp_path / "out.mp4"
        export_project(ready_project, ExportSettings(path=output, video_codec=codec))

        assert output.exists()
        assert output.stat().st_size > 0

        item = probe_media(output)
        assert item.has_video
        assert item.has_audio, "音声トラックがあるのに無音のファイルになっている"
        stream = item.video_streams[0]
        assert (stream.width, stream.height) == (320, 240)
        assert stream.frame_rate == FrameRate(30)

    def test_the_default_codec_skips_a_qsv_that_does_not_open(
        self, probed: list[str], ready_project: Project, tmp_path: Path
    ) -> None:
        """コーデックを指定しない書き出しは、開けない QSV を飛ばして libx264 で通る（#67）"""
        output = tmp_path / "out.mp4"
        export_project(ready_project, ExportSettings(path=output, frame_range=(0, 3)))
        assert _encoder_of(output) == "libx264"

    def test_falls_back_when_the_real_size_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        forget_probes: None,
        ready_project: Project,
        tmp_path: Path,
    ) -> None:
        """試しには開けても実際の設定で断られたら、次の候補で書き出す

        NVENC の H.264 は幅 4096 を超える画を開けない 試しに開けたからと 1 つに決めると、
        8K の作品は libx264 なら書けるのに失敗する ここでは yuv420p を受け付けない png を
        「試しには開けた」ことにして同じ状況を作る
        """
        monkeypatch.setattr(exporter, "_open_encoder", lambda name: None)
        monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("png", "libx264"))
        output = tmp_path / "out.mp4"
        export_project(ready_project, ExportSettings(path=output, frame_range=(0, 3)))
        assert _encoder_of(output) == "libx264"
        with av.open(str(output)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        # やり直しの前に音やコマを進めていたら、頭が欠けたり音がずれたりする
        assert frames == 3

    def test_falls_back_when_the_stream_cannot_be_added(
        self,
        monkeypatch: pytest.MonkeyPatch,
        forget_probes: None,
        ready_project: Project,
        tmp_path: Path,
    ) -> None:
        """開く前の add_stream で断られても、次の候補で書き出す

        入っていない名前やコンテナが受けないコーデックは、開くより前にここで断られる
        拾わないと、指定なしの書き出しでも libx264 まで落ちずに失敗する
        """
        monkeypatch.setattr(exporter, "_open_encoder", lambda name: None)
        monkeypatch.setattr(exporter, "VIDEO_CODEC_PREFERENCE", ("no_such_encoder", "libx264"))
        output = tmp_path / "out.mp4"
        export_project(ready_project, ExportSettings(path=output, frame_range=(0, 3)))
        assert _encoder_of(output) == "libx264"

    def test_a_chosen_codec_that_does_not_open_is_an_export_error(
        self, ready_project: Project, tmp_path: Path
    ) -> None:
        """名指ししたコーデックが開けなければ、勝手に変えずに ExportError で断る

        FFmpeg の例外のまま投げると、書き出しの画面が理由を出せず「予期しない失敗」になる
        黙って別のコーデックに変えると、選んだ物で書けたと思い込ませる
        """
        output = tmp_path / "out.mp4"
        settings = ExportSettings(path=output, video_codec="png", frame_range=(0, 3))
        with pytest.raises(ExportError, match="png"):
            export_project(ready_project, settings)
        assert not output.exists()

    def test_frame_count_matches_the_timeline(self, ready_project: Project, tmp_path: Path) -> None:
        # 書き出した動画のコマ数がタイムラインと合わないと、終わりが欠けるか余計なコマが付く
        output = tmp_path / "out.mp4"
        export_project(ready_project, ExportSettings(path=output, video_codec="libx264"))

        with av.open(str(output)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        assert frames == ready_project.duration

    def test_video_only_project(self, silent_project: Project, tmp_path: Path) -> None:
        # 音の無い作品に中身の無い音声トラックが付くと、
        # 再生ソフトや動画サイトが音のある動画として扱う
        output = tmp_path / "silent.mp4"
        export_project(silent_project, ExportSettings(path=output, video_codec="libx264"))

        item = probe_media(output)
        assert item.has_video
        assert not item.has_audio

    def test_frame_range(self, ready_project: Project, tmp_path: Path) -> None:
        # 範囲を決めて書き出しても作品全体や違う長さが出ると、切り出したい場面だけを動画にできない
        output = tmp_path / "range.mp4"
        export_project(
            ready_project,
            ExportSettings(path=output, video_codec="libx264", frame_range=(10, 25)),
        )
        with av.open(str(output)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        assert frames == 15

    def test_reports_progress(self, ready_project: Project, tmp_path: Path) -> None:
        # 進み具合が動かない・戻る・100% に届かないまま終わると、
        # 書き出しが止まったのか終わったのか分からない
        seen: list[float] = []
        export_project(
            ready_project,
            ExportSettings(path=tmp_path / "p.mp4", video_codec="libx264"),
            progress=seen.append,
        )
        assert seen
        assert seen[-1] == pytest.approx(1.0)
        assert seen == sorted(seen)

    def test_cancel_leaves_no_file(self, ready_project: Project, tmp_path: Path) -> None:
        # 中途半端なファイルが残ると、書き出せたのかどうか分からなくなる
        output = tmp_path / "cancelled.mp4"
        stop = threading.Event()

        def should_cancel() -> bool:
            stop.set()
            return True

        with pytest.raises(ExportError, match="中止"):
            export_project(
                ready_project,
                ExportSettings(path=output, video_codec="libx264"),
                should_cancel=should_cancel,
            )
        assert not output.exists()

    def test_failure_leaves_no_file(self, ready_project: Project, tmp_path: Path) -> None:
        # 失敗したのに壊れた mp4 が残ると、出来上がったと思って開いても再生できない
        output = tmp_path / "broken.mp4"
        with pytest.raises(Exception):  # noqa: B017 - コーデック側の例外種別は問わない
            export_project(
                ready_project,
                ExportSettings(path=output, video_codec="存在しないコーデック"),
            )
        assert not output.exists()

    def test_empty_timeline_is_refused(self, tmp_path: Path) -> None:
        # 何も置いていない作品で書き出したとき、理由を示さずに失敗するか中身の無い動画ができる
        empty = Project.create(ProjectSettings(width=320, height=240))
        with pytest.raises(ExportError, match="範囲が空"):
            export_project(empty, ExportSettings(path=tmp_path / "empty.mp4"))

    def test_creates_missing_directories(self, ready_project: Project, tmp_path: Path) -> None:
        # 保存先にまだ無いフォルダを指定すると、フォルダを手で作らない限り書き出せない
        output = tmp_path / "深い" / "階層" / "out.mp4"
        export_project(ready_project, ExportSettings(path=output, video_codec="libx264"))
        assert output.exists()


def _decoded(path: Path) -> list[np.ndarray]:
    """書き出したファイルのフレームを全部読む"""
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def _export_both(project: Project, serial: Path, pipelined: Path) -> None:
    """同じ範囲を、1 枚ずつと重ねた場合の 2 通りで書き出す"""
    for path, depth in ((serial, 0), (pipelined, 3)):
        export_project(
            project,
            ExportSettings(
                path=path, video_codec="libx264", frame_range=(0, 8), pipeline_depth=depth
            ),
        )


class TestPipeline:
    """合成と書き込みを重ねても、出来上がる物が変わらないこと（#56）"""

    def test_the_pipeline_makes_the_same_frames(
        self, ready_project: Project, tmp_path: Path
    ) -> None:
        # 速くするために絵が変わったら本末転倒 順番も内容も 1 枚ずつのときと同じでなければならない
        serial = tmp_path / "serial.mp4"
        pipelined = tmp_path / "pipelined.mp4"
        _export_both(ready_project, serial, pipelined)

        left, right = _decoded(serial), _decoded(pipelined)
        assert len(left) == len(right) == 8
        for number, (a, b) in enumerate(zip(left, right, strict=True)):
            assert np.array_equal(a, b), f"{number} 枚目の絵が違う"

    def test_the_pipeline_keeps_the_audio(self, ready_project: Project, tmp_path: Path) -> None:
        # 音声も同じスレッドで流す 片側だけ別スレッドにすると多重化の順序が崩れ、
        # 音の無いファイルや再生できないファイルができる
        serial = tmp_path / "serial.mp4"
        pipelined = tmp_path / "pipelined.mp4"
        _export_both(ready_project, serial, pipelined)

        def samples(path: Path) -> int:
            with av.open(str(path)) as container:
                return sum(frame.samples for frame in container.decode(audio=0))

        assert samples(pipelined) == samples(serial) > 0

    def test_a_failure_in_the_writer_surfaces(
        self, ready_project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 別スレッドの失敗を握り潰すと、途中までしか書けていないファイルが
        # 「書き出せた」として残る
        output = tmp_path / "broken.mp4"
        original = exporter._FrameWriter.write
        calls = [0]

        def failing(
            self: exporter._FrameWriter, index: int, frame_number: int, image: np.ndarray
        ) -> None:
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("書き込みで失敗した")
            original(self, index, frame_number, image)

        monkeypatch.setattr(exporter._FrameWriter, "write", failing)
        with pytest.raises(RuntimeError, match="書き込みで失敗した"):
            export_project(
                ready_project,
                ExportSettings(path=output, video_codec="libx264", pipeline_depth=2),
            )
        assert not output.exists()

    def test_a_failure_in_the_writer_does_not_hang(
        self, ready_project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 失敗した後に受け取りをやめると、合成の側がキューで止まって書き出しが固まる
        def failing(
            self: exporter._FrameWriter, index: int, frame_number: int, image: np.ndarray
        ) -> None:
            raise RuntimeError("書き込みで失敗した")

        monkeypatch.setattr(exporter._FrameWriter, "write", failing)
        finished = threading.Event()

        def run() -> None:
            with pytest.raises(RuntimeError):
                export_project(
                    ready_project,
                    ExportSettings(
                        path=tmp_path / "hang.mp4", video_codec="libx264", pipeline_depth=1
                    ),
                )
            finished.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert finished.wait(120), "書き込みが失敗した後、書き出しが戻ってこない"

    @pytest.mark.parametrize("depth", [-1, MAX_PIPELINE_DEPTH + 1, 2.5])
    def test_an_out_of_range_depth_is_refused(
        self, ready_project: Project, tmp_path: Path, depth: float
    ) -> None:
        # 黙って通すと、大きい値で合成済みの絵を抱えすぎてメモリを使い切る
        # 負の値は「重ねない」と同じに丸められ、頼んだ設定と実際が食い違う
        # 2.5 のような値は範囲だけ見ると通り、席やキューの数として奥まで流れる
        output = tmp_path / "depth.mp4"
        with pytest.raises(ExportError, match="先読みの深さ"):
            export_project(
                ready_project,
                ExportSettings(
                    path=output,
                    video_codec="libx264",
                    # 型の宣言は int なので mypy は通さない ここで試すのは、
                    # 型検査を通さずに呼ぶ側（設定ファイルや別の言語からの呼び出し）
                    pipeline_depth=depth,  # type: ignore[arg-type]
                ),
            )
        assert not output.exists()

    def test_cancelling_while_the_last_frames_drain_is_a_failure(
        self, ready_project: Project, tmp_path: Path
    ) -> None:
        # 最後の 1 枚を渡した後、書き込みが終わるまでの間にも中止は押せる
        # そこを見ないと「書き出しました」と出てファイルも残る
        output = tmp_path / "late.mp4"
        stop = threading.Event()
        frames = 6

        def should_cancel() -> bool:
            # 全部渡し終えた後にだけ真を返す 合成の途中では止めない
            return stop.is_set()

        original = exporter._WritePipeline.close

        def closing(self: exporter._WritePipeline, *, reraise: bool = True) -> None:
            stop.set()
            original(self, reraise=reraise)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(exporter._WritePipeline, "close", closing)
            with pytest.raises(ExportError, match="中止"):
                export_project(
                    ready_project,
                    ExportSettings(
                        path=output,
                        video_codec="libx264",
                        frame_range=(0, frames),
                        pipeline_depth=2,
                    ),
                    should_cancel=should_cancel,
                )
        assert not output.exists()

    def test_progress_counts_written_frames(self, ready_project: Project, tmp_path: Path) -> None:
        # 合成した枚数で数えると、深さのぶんだけ先に 100% になり、
        # そこから実際の書き込みを待つ
        seen: list[float] = []
        export_project(
            ready_project,
            ExportSettings(
                path=tmp_path / "p.mp4",
                video_codec="libx264",
                frame_range=(0, 8),
                pipeline_depth=3,
            ),
            progress=seen.append,
        )
        assert len(seen) == 8
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)


class TestRoundTrip:
    def test_exported_file_can_be_edited_again(
        self, ready_project: Project, tmp_path: Path
    ) -> None:
        # 書き出したものを読み込み直せること 中間ファイルとして使う経路
        output = tmp_path / "again.mp4"
        export_project(ready_project, ExportSettings(path=output, video_codec="libx264"))

        reimported = probe_media(output)
        document = Document(Project.create(ProjectSettings(width=320, height=240)))
        for command in insert_media(document.project, reimported):
            document.execute(command)

        assert document.project.duration > 0
        assert len(list(document.project.timeline.video_tracks())) == 1
        assert len(list(document.project.timeline.audio_tracks())) == 1

    def test_audio_survives(self, media_dir: Path, tmp_path: Path) -> None:
        # 音が無音になっていないこと 書き出し経路で最も見落としやすい
        loud = make_sample(media_dir, "loud.mp4", duration=1.5, tone_hz=440)
        document = Document(
            Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
        )
        for command in insert_media(document.project, probe_media(loud.path)):
            document.execute(command)

        output = tmp_path / "audio.mp4"
        export_project(document.project, ExportSettings(path=output, video_codec="libx264"))

        from sashimono.engine.decode import AudioDecoder

        with AudioDecoder(output, sample_rate=48000) as decoder:
            samples = decoder.read_seconds(Fraction(1, 4), Fraction(1, 2))
        assert float(abs(samples).max()) > 0.005, "書き出したファイルが無音"


@pytest.fixture
def bars_project(tmp_path: Path) -> Project:
    """原色と灰色の帯の静止画を 1 枚置いたプロジェクト 行列の違いがいちばん大きく出る"""
    still = tmp_path / "bars.png"
    image = av.VideoFrame.from_ndarray(bars(320, 240), format="rgb24")
    with av.open(str(still), mode="w", format="image2") as container:
        stream = container.add_stream("png")
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "rgb24"
        container.mux(stream.encode(image))
        container.mux(stream.encode(None))

    document = Document(
        Project.create(ProjectSettings(width=320, height=240, frame_rate=FrameRate(30)))
    )
    for command in insert_media(document.project, probe_media(still)):
        document.execute(command)
    return document.project


class TestColor:
    """書き出しの色は BT.709 / limited で、同じ値のタグが付く（#61）

    直す前は swscale の既定の BT.601 で変換し、タグも無かった 赤の Y が 63 ではなく 81 に
    なり、再生側は行列を推測で読むしかなかった
    """

    @pytest.mark.parametrize("codec", ["libx264", "h264_nvenc", "h264_qsv"])
    def test_values_and_tags_are_bt709(
        self, codec: str, bars_project: Project, tmp_path: Path
    ) -> None:
        # 候補は開けた物だけ（#67） QSV が入っていても Intel の GPU が無い機械ではここで飛ぶ
        # 飛ばすのは開けないときだけ 書き出し全体の失敗まで飛ばすと、開けたあとの
        # 回帰（変換・エンコード・多重化）が「飛ばした」に紛れる
        if codec not in available_video_codecs():
            pytest.skip(f"{codec} が使えない")
        # 書き出しそのものは囲まない 開けると分かったあとの失敗（変換・エンコード・
        # 多重化）は、どのコーデックでも本物の回帰
        output = tmp_path / f"{codec}.mp4"
        export_project(
            bars_project, ExportSettings(path=output, video_codec=codec, frame_range=(0, 3))
        )

        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            context = stream.codec_context
            # 開き直して読むのは、コンテナとビットストリームに残ったタグ
            assert context.color_primaries == ColorPrimaries.BT709
            assert context.color_trc == ColorTrc.BT709
            assert context.colorspace == AVCOL_SPC_BT709
            assert context.color_range == ColorRange.MPEG
            frame = next(container.decode(stream))
        # ハードウェアのエンコーダは非可逆なので少し幅を持たせる BT.601 との差は 10 以上ある
        assert_close(yuv_at_bars(frame), BT709_YUV, tolerance=3)

    @pytest.mark.parametrize("name", sorted(COLOR_OPTIONS))
    def test_color_options_are_refused(
        self, name: str, bars_project: Project, tmp_path: Path
    ) -> None:
        # options はコーデックを開くときに効くので、付けたタグを後から書き換えてしまう
        # 画素は BT.709 のまま、タグだけ bt470bg などになったファイルができる
        output = tmp_path / "out.mp4"
        settings = ExportSettings(
            path=output, video_codec="libx264", frame_range=(0, 3), options={name: "bt470bg"}
        )
        with pytest.raises(ExportError, match=name):
            export_project(bars_project, settings)
        assert not output.exists()

    @pytest.mark.parametrize(
        ("name", "value", "culprit"),
        [
            ("x264-params", "colorprim=bt470bg:colormatrix=bt470bg", "colorprim"),
            ("x264-params", "keyint=60:Transfer=smpte170m", "transfer"),
            ("x264opts", "keyint=60,fullrange=on", "fullrange"),
            ("x265-params", "range=full", "range"),
        ],
    )
    def test_color_params_inside_encoder_options_are_refused(
        self, name: str, value: str, culprit: str, bars_project: Project, tmp_path: Path
    ) -> None:
        # x264-params の中の色はビットストリームの VUI だけを書き換える MP4 の colr は
        # bt709 のまま残り、生の H.264 を読むと bt470bg、というファイルになっていた
        output = tmp_path / "out.mp4"
        settings = ExportSettings(
            path=output, video_codec="libx264", frame_range=(0, 3), options={name: value}
        )
        with pytest.raises(ExportError, match=culprit):
            export_project(bars_project, settings)
        assert not output.exists()

    def test_other_encoder_params_pass(self, bars_project: Project, tmp_path: Path) -> None:
        # 色と関係の無いまとめ書きまで断ると、プリセットや品質の指定が通せなくなる
        output = tmp_path / "out.mp4"
        settings = ExportSettings(
            path=output,
            video_codec="libx264",
            frame_range=(0, 3),
            options={"x264-params": "keyint=30:bframes=0"},
        )
        export_project(bars_project, settings)
        assert output.exists()

    def test_colors_survive_a_round_trip(self, bars_project: Project, tmp_path: Path) -> None:
        # 書いたタグどおりに読み直せば、元の色へ戻る 変換とタグのどちらかが壊れると、
        # 書き出したものを素材として読み込み直したときに原色がずれる（中間ファイルの色が変わる）
        output = tmp_path / "again.mp4"
        export_project(
            bars_project,
            ExportSettings(path=output, video_codec="libx264", frame_range=(0, 3)),
        )
        with VideoDecoder(output) as decoder:
            image = decoder.frame_at(Fraction(0))
        assert image is not None
        assert_close(rgb_at_bars(image), COLORS, tolerance=4)


class TestBlending:
    """書き出しもプロジェクトの重ね合わせの方法（Issue #65）に従う

    プレビューだけ切り替わって書き出しが既定のままだと、画面で合わせた半透明の明るさが
    書き出した動画で変わる
    """

    @pytest.mark.parametrize(
        ("blending", "expected"), [(Blending.SRGB, 128), (Blending.LINEAR, 188)]
    )
    def test_half_white_on_black(self, blending: str, expected: int, tmp_path: Path) -> None:
        # 見たいのは合成の明るさ エンコーダが入っていない機械の失敗と取り違えないよう飛ばす
        if "libx264" not in available_video_codecs():
            pytest.skip("libx264 が使えない")
        settings = ProjectSettings(width=64, height=64, frame_rate=FrameRate(30), blending=blending)
        project = Project.create(settings)
        track = Track(TrackKind.VIDEO, "V1")
        white = GeneratedSource(
            kind="shape", params={"shape": "background", "color": (1.0, 1.0, 1.0, 1.0)}
        )
        clip = Clip(timeline_start=0, duration=3, source=white, opacity=AnimatedValue(0.5))
        project = AddClip(track.id, clip).apply(AddTrack(track).apply(project))
        output = tmp_path / f"{blending}.mp4"
        export_project(project, ExportSettings(path=output, video_codec="libx264"))
        with VideoDecoder(output) as decoder:
            image = decoder.frame_at(Fraction(0))
        assert image is not None
        # 灰色は行列に左右されない 残るのは limited への変換と圧縮の丸めだけ
        assert abs(int(image[32, 32, 0]) - expected) <= 3
