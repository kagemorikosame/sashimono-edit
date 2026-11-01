"""AI の語の時刻の道具（#420）"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import OPERATIONS
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.model import (
    Clip,
    MediaItem,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
    new_media_id,
)
from sashimono.engine.checks import CheckState
from sashimono.engine.checks.report import assemble
from sashimono.engine.checks.scan import BLACK, SUBTITLE_WORDS, WORD_CUT, WORD_KINDS, Run
from tests.ai.conftest import FakeHost
from tests.words_helpers import in_scene, placed, sample_transcript, wave


def _tool(name: str) -> Any:
    return next(operation for operation in OPERATIONS if operation.name == name)


def _call(host: FakeHost, name: str, **arguments: object) -> dict[str, Any]:
    result = _tool(name)(host, arguments)
    assert isinstance(result, dict)
    return result


def _clip(media: MediaItem, start: int, duration: int, source_in: Fraction) -> Clip:
    return Clip(start, duration, media_id=media.id, source_in=source_in)


@pytest.fixture
def media(video_media: MediaItem) -> MediaItem:
    return video_media.with_transcript(sample_transcript())


def _host(media: MediaItem, *clips: Clip) -> FakeHost:
    host = FakeHost(placed(media, *clips))
    host.stub_waveform = wave()
    return host


class TestOverlappingSounds:
    """シーンを置いたクリップの ID は、中の音すべてに当たる"""

    def _scene(self, *clips: tuple[MediaItem, Clip]) -> tuple[FakeHost, Clip]:
        tracks = tuple(
            Track(TrackKind.AUDIO, name=f"A{n}", clips=(clip,))
            for n, (_, clip) in enumerate(clips, 1)
        )
        media = tuple(dict.fromkeys(item for item, _ in clips))
        inner = replace(placed(clips[0][0]), media=media)
        inner = inner.with_timeline(replace(inner.timeline, tracks=tracks))
        window = Clip(0, 60, source_in=Fraction(0))
        host = FakeHost(in_scene(inner, window))
        host.stub_waveform = wave()
        return host, window

    def test_the_voice_is_chosen_over_the_music(
        self, video_media: MediaItem, media: MediaItem
    ) -> None:
        # 壊れると、シーンの ID を渡しても BGM が選ばれ、字幕が無いと断られる
        music = replace(video_media, id=new_media_id())
        host, window = self._scene(
            (music, _clip(music, 0, 60, Fraction(0))),
            (media, _clip(media, 0, 60, Fraction(1))),
        )
        result = _call(host, "word_boundaries", frame=12, clip_id=str(window.id))
        assert result["word"]["text"] == "カメラを"

    def test_two_voices_ask_which(self, video_media: MediaItem, media: MediaItem) -> None:
        # 壊れると、並びの最初の話し手を黙って選び、別の人の語の境目で切る
        other = replace(media, id=new_media_id())
        second = _clip(other, 0, 60, Fraction(1))
        host, window = self._scene((media, _clip(media, 0, 60, Fraction(1))), (other, second))
        with pytest.raises(ToolError, match="scene_clip_id"):
            _call(host, "word_boundaries", frame=12, clip_id=str(window.id))
        result = _call(host, "word_boundaries", frame=12, clip_id=str(second.id))
        assert result["media_id"] == str(other.id)

    def test_heard_words_names_the_inner_clip(self, media: MediaItem) -> None:
        # 壊れると、語を読んだ音を word_boundaries で指す ID が分からない
        inner = _clip(media, 0, 60, Fraction(1))
        host, _ = self._scene((media, inner))
        (row, *_) = _call(host, "heard_words")["heard"]
        assert row["scene_clip_id"] == str(inner.id)


class TestWordBoundaries:
    def test_a_line_without_words_is_not_called_clean(self, video_media: MediaItem) -> None:
        # 壊れると、語の時刻の無い行の中を「語の途中ではない」と答え、声の途中で切る
        lines = (
            *sample_transcript().segments,
            TranscriptSegment(Fraction(6), Fraction(7), "手で足した行"),
        )
        media = video_media.with_transcript(Transcript(lines, language="ja"))
        host = _host(media, _clip(media, 0, 60, Fraction(6)))
        result = _call(host, "word_boundaries", frame=15)
        assert result["has_words"] is False
        assert result["inside_word"] is None
        assert "words_only" in result["note"]
        # 語の時刻のある所は今までどおり読める
        assert _call(host, "word_boundaries", media_id=str(media.id), seconds=1.4)["has_words"]

    def test_it_only_reads(self) -> None:
        # 壊れると、語の境目を読むたびに確かめの箱が出る
        assert _tool("word_boundaries").safety is Safety.READ_ONLY
        assert _tool("heard_words").safety is Safety.READ_ONLY

    def test_by_the_timeline_frame(self, media: MediaItem) -> None:
        # 壊れると、切ってよい位置のフレームがクリップの置き場所とずれる
        clip = _clip(media, 30, 120, Fraction(1))
        host = _host(media, clip)
        result = _call(host, "word_boundaries", frame=42)
        assert result["source_seconds"] == 1.4
        assert result["inside_word"] is True
        assert result["word"]["text"] == "カメラを"
        assert result["clip_id"] == str(clip.id)
        (first, *_) = result["cut_points"]
        assert first["before"] == "カメラを"
        assert first["after"] == "作ります"
        # 素材の 1.8 秒はこのクリップで 54 コマ目
        assert first["in_frame"] == 54
        assert first["out_frame"] == 54

    def test_by_the_material_time(self, media: MediaItem) -> None:
        # 置く前の切り抜きを選ぶときはタイムラインに音が無い
        host = _host(media)
        result = _call(host, "word_boundaries", media_id=str(media.id), seconds=3.0)
        assert result["inside_word"] is False
        assert result["word"] is None
        assert result["before"][-1]["text"] == "作ります"
        assert result["after"][0]["text"] == "はい"
        (point,) = [p for p in result["cut_points"] if p["after"] == "はい"]
        assert point["kind"] == "pause"
        assert "in_frame" not in point

    def test_it_says_what_it_needs(self, media: MediaItem) -> None:
        # 壊れると、何を渡せばよいか分からず AI が同じ呼び方を繰り返す
        host = _host(media)
        with pytest.raises(ToolError, match="frame か"):
            _call(host, "word_boundaries")
        with pytest.raises(ToolError, match="鳴っている素材の音がありません"):
            _call(host, "word_boundaries", frame=5)

    def test_a_material_without_subtitles(self, video_media: MediaItem) -> None:
        # 壊れると、語を読めない訳が分からず transcribe へ進めない
        host = _host(video_media, _clip(video_media, 0, 30, Fraction(0)))
        with pytest.raises(ToolError, match="transcribe"):
            _call(host, "word_boundaries", frame=0)

    def test_without_word_times_it_uses_the_lines(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """語の時刻の無い作品でも切り所を返す 断ると起こし直すまで手探りで切る"""
        media = video_media.with_transcript(transcript)
        host = _host(media, _clip(media, 0, 150, Fraction(0)))
        result = _call(host, "word_boundaries", frame=105, window_seconds=3)
        assert result["has_words"] is False
        assert result["inside_word"] is None
        assert result["cut_points"]
        assert "words_only" in result["note"]

    def test_without_a_waveform(self, media: MediaItem) -> None:
        # 壊れると、音を確かめていない位置を確かめた物として読む
        host = _host(media, _clip(media, 0, 60, Fraction(1)))
        host.stub_waveform = None
        result = _call(host, "word_boundaries", frame=12)
        assert {point["kind"] for point in result["cut_points"]} == {"unchecked"}
        assert "解析がまだ" in result["note"]

    def test_the_window_is_limited(self, media: MediaItem) -> None:
        # 壊れると、黙って端へ寄せた幅を頼んだ幅と思い込む
        host = _host(media, _clip(media, 0, 60, Fraction(1)))
        with pytest.raises(ToolError, match="window_seconds"):
            _call(host, "word_boundaries", frame=0, window_seconds=99)

    def test_inside_a_scene(self, media: MediaItem) -> None:
        # 壊れると、シーンの中の語を別の時刻で読む
        inner = placed(media, _clip(media, 0, 300, Fraction(0)))
        host = FakeHost(in_scene(inner, Clip(0, 30, source_in=Fraction(1))))
        host.stub_waveform = wave()
        result = _call(host, "word_boundaries", frame=12)
        assert result["source_seconds"] == 1.4
        assert "scene_clip_id" in result

    def test_the_outbound_declaration(self) -> None:
        """語は本人の文 宣言から漏れると、字幕を送らない人の語が送られる"""
        assert Outbound.TEXT in _tool("word_boundaries").sends
        assert Outbound.TEXT in _tool("word_boundaries").hidden
        assert Outbound.TEXT in _tool("heard_words").sends


class TestHeardWords:
    def test_a_scene_keeps_each_material_apart(
        self, video_media: MediaItem, media: MediaItem
    ) -> None:
        # 壊れると、シーンの中で素材が替わっても 1 行に混ざり、別の素材の語が最初の素材の語になる
        other = replace(video_media, id=new_media_id()).with_transcript(sample_transcript())
        inner = replace(placed(media), media=(media, other))
        track = Track(
            TrackKind.AUDIO,
            name="A1",
            clips=(_clip(media, 0, 24, Fraction(1)), _clip(other, 24, 12, Fraction(4))),
        )
        inner = inner.with_timeline(replace(inner.timeline, tracks=(track,)))
        host = FakeHost(in_scene(inner, Clip(0, 36, source_in=Fraction(0))))
        result = _call(host, "heard_words")
        assert [row["media_id"] for row in result["heard"]] == [str(media.id), str(other.id)]

    def test_each_clip_reads_its_own_words(self, media: MediaItem) -> None:
        # 壊れると、どのクリップの語か分からず、直す相手を選べない
        first = _clip(media, 0, 24, Fraction(1))
        second = _clip(media, 24, 12, Fraction(4))
        host = _host(media, first, second)
        result = _call(host, "heard_words")
        assert [row["text"] for row in result["heard"]] == ["カメラを", "はい"]
        assert result["heard"][0]["clip_id"] == str(first.id)

    def test_the_range(self, media: MediaItem) -> None:
        # 壊れると、範囲の外の語まで字幕と比べる
        host = _host(media, _clip(media, 0, 60, Fraction(1)))
        result = _call(host, "heard_words", from_frame=0, to_frame=12)
        assert [row["text"] for row in result["heard"]] == ["カメ"]

    def test_no_word_times_are_named(self, video_media: MediaItem, transcript: Transcript) -> None:
        # 壊れると、語の時刻が無いだけなのに音に語が無いと読む
        media = video_media.with_transcript(transcript)
        host = _host(media, _clip(media, 0, 60, Fraction(0)))
        result = _call(host, "heard_words")
        assert result["heard"] == []
        assert result["no_words"] == [{"media_id": str(media.id), "audio": 1}]
        assert "words_only" in result["note"]

    def test_a_bad_range(self, media: MediaItem) -> None:
        # 壊れると、空の答えを語が無いと読む
        host = _host(media, _clip(media, 0, 60, Fraction(1)))
        with pytest.raises(ToolError, match="from_frame"):
            _call(host, "heard_words", from_frame=10, to_frame=5)


class TestSubtitles:
    def test_a_line_that_crosses_a_cut_has_heard(self, media: MediaItem) -> None:
        # 壊れると、字幕の文の後半が音に無いことに AI が気付けない
        host = _host(media, _clip(media, 0, 24, Fraction(1)))
        (row,) = _call(host, "get_subtitles")["subtitles"]
        assert row["text"] == "カメラを作ります"
        assert row["heard"] == "カメラを"

    def test_a_whole_line_stays_short(self, media: MediaItem) -> None:
        # 全部の行へ付けると字幕の一覧の長さが倍になり、切って返す所が増える
        host = _host(media, _clip(media, 0, 60, Fraction(1)))
        (row,) = _call(host, "get_subtitles")["subtitles"]
        assert "heard" not in row

    def test_asked_for_every_line(self, media: MediaItem) -> None:
        # 壊れると、手で直した行の文言と音を比べる手段が無い
        host = _host(media, _clip(media, 0, 60, Fraction(1)))
        (row,) = _call(host, "get_subtitles", heard=True)["subtitles"]
        assert row["heard"] == "カメラを作ります"

    def test_compact_keeps_it(self, media: MediaItem) -> None:
        # 壊れると、誤植を探す軽い一覧から食い違いが読めない
        host = _host(media, _clip(media, 0, 24, Fraction(1)))
        (row,) = _call(host, "get_subtitles", compact=True)["subtitles"]
        assert row["heard"] == "カメラを"


class TestTranscribe:
    def test_words_only_keeps_the_subtitles(self, media: MediaItem) -> None:
        """人が直した字幕を消さずに語の時刻だけを足す 断ると起こし直すしかなくなる"""
        host = _host(media, _clip(media, 0, 60, Fraction(1)))
        _call(host, "transcribe", media_id=str(media.id), words_only=True)
        assert host.transcribed_words_only

    def test_the_refusal_tells_the_way(self, media: MediaItem) -> None:
        # 壊れると、語の時刻のために replace を選び、直した字幕を消す
        host = _host(media)
        with pytest.raises(ToolError, match="words_only"):
            _call(host, "transcribe", media_id=str(media.id))

    def test_words_only_and_replace_conflict(self, media: MediaItem) -> None:
        # 壊れると、残すはずの字幕を置き換える
        host = _host(media)
        with pytest.raises(ToolError, match="一緒に使えません"):
            _call(host, "transcribe", media_id=str(media.id), words_only=True, replace=True)


class TestCheckOutput:
    def _done(self, host: FakeHost, runs: tuple[Run, ...] = ()) -> None:
        """最後の頼みを調べ終えた様子にする"""
        request = host.check_requests[-1]
        report = assemble(
            list(runs),
            start=request.start,
            end=request.end,
            analyzed=request.end,
            fps=30.0,
            complete=True,
            limits=request.limits,
            index=None,
            loudness=None,
            peak=None,
            length_seconds=None,
        )
        host.check = CheckState(request=request, report=report)

    def _finished(self, host: FakeHost, **arguments: object) -> dict[str, Any]:
        arguments.setdefault("wait_seconds", 0)
        _tool("check_output")(host, arguments)
        self._done(host)
        return _call(host, "check_output", **arguments)

    def test_word_problems_sort_with_the_rest(self, media: MediaItem) -> None:
        # 壊れると、語の問題だけ一覧の後ろに固まり、時刻の順に読めない
        host = _host(media, _clip(media, 0, 12, Fraction(1)), _clip(media, 40, 20, Fraction(8)))
        arguments = {"checks": ["black", WORD_CUT], "wait_seconds": 0}
        _tool("check_output")(host, arguments)
        self._done(host, (Run(BLACK, 3, 5, {}), Run(BLACK, 20, 30, {})))
        result = _call(host, "check_output", **arguments)
        assert [row["kind"] for row in result["problems"]] == ["black", WORD_CUT, "black"]

    def test_the_clip_picks_the_voice(self, video_media: MediaItem, media: MediaItem) -> None:
        # 壊れると、BGM が先に並ぶ所で声の語の境目を読めず、字幕が無いと断られる
        music = replace(video_media, id=new_media_id())
        voice = _clip(media, 0, 60, Fraction(1))
        project = placed(music, _clip(music, 0, 60, Fraction(0)))
        track = Track(kind=TrackKind.AUDIO, name="A2", clips=(voice,))
        project = replace(project, media=(music, media))
        project = project.with_timeline(
            replace(project.timeline, tracks=(*project.timeline.tracks, track))
        )
        host = FakeHost(project)
        host.stub_waveform = wave()
        assert _call(host, "word_boundaries", frame=12)["word"]["text"] == "カメラを"
        result = _call(host, "word_boundaries", frame=12, clip_id=str(voice.id))
        assert result["word"]["text"] == "カメラを"
        with pytest.raises(ToolError, match="クリップ"):
            _call(host, "word_boundaries", frame=12, clip_id="無いクリップ")

    def test_the_suggested_move_leaves_the_word(self, media: MediaItem) -> None:
        """通しの確かめの形 語の終わりの近くの入りで、動かす先が今の切れ目と同じ値で返った

        壊れると、手掛かりにならない量を返し、AI が自分で量を決めることになる
        """
        clip = _clip(media, 60, 60, Fraction(17, 10))
        host = _host(media, clip)
        host.stub_waveform = wave(((0.0, 1.0), (1.72, 1.76), (2.6, 12.0)))
        (cut,) = self._finished(host, checks=[WORD_CUT])["problems"]
        # 語を残す向きが届くので、語の頭まで戻す
        assert cut["suggested_source_seconds"] <= 1.0
        assert cut["suggested_head_delta"] < 0
        _call(host, "trim_clip", clip_id=str(clip.id), head_delta=cut["suggested_head_delta"])
        assert self._finished(host, checks=[WORD_CUT])["problems"] == []

    def test_a_cut_inside_a_word_is_returned(self, media: MediaItem) -> None:
        # 壊れると、言葉の途中の切れ目を AI が直せない
        clip = _clip(media, 0, 12, Fraction(1))
        host = _host(media, clip)
        result = self._finished(host, checks=sorted(WORD_KINDS))
        (cut,) = [row for row in result["problems"] if row["kind"] == WORD_CUT]
        assert cut["severity"] == "high"
        assert cut["clip_ids"] == [str(clip.id)]
        assert cut["fix"]["action"] == "auto"
        assert "trim_clip" in cut["fix"]["tools"]
        assert set(result["checks"]) == WORD_KINDS

    def test_the_tail_delta_goes_straight_to_trim_clip(self, media: MediaItem) -> None:
        """返した量を trim_clip へ渡すと、調べ直して切れ目が消える"""
        clip = _clip(media, 0, 12, Fraction(1))
        host = _host(media, clip)
        host.stub_waveform = wave(((0.0, 1.0), (1.8, 1.86), (2.6, 12.0)))
        result = self._finished(host, checks=[WORD_CUT])
        (cut,) = result["problems"]
        _call(host, "trim_clip", clip_id=str(clip.id), tail_delta=cut["suggested_tail_delta"])
        assert self._finished(host, checks=[WORD_CUT])["problems"] == []

    def test_the_word_is_hidden_from_who_does_not_send_text(self, media: MediaItem) -> None:
        # 欄は _CHECK_TEXT_KEYS で伏せる 道具の宣言に字幕と文字が入っていること
        assert Outbound.TEXT in _tool("check_output").hidden
        host = _host(media, _clip(media, 0, 24, Fraction(1)))
        result = self._finished(host, checks=[SUBTITLE_WORDS])
        (row,) = result["problems"]
        assert {"text", "heard", "missing"} <= set(row)

    def test_no_word_times_are_skipped(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、調べていないのに切れ目は無事と AI が伝える
        media = video_media.with_transcript(transcript)
        host = _host(media, _clip(media, 0, 60, Fraction(0)))
        result = self._finished(host, checks=[WORD_CUT])
        assert result["skipped_checks"] == [WORD_CUT]
        assert "words_only" in result["note"]

    def test_the_project_of_the_request_is_read(self, media: MediaItem) -> None:
        """調べている間に作品が変わっても、調べ始めたときの作品の切れ目を返す

        今の作品で読むと、直した後の問題が古い結果に混ざり、直す回の数えが合わなくなる
        """
        clip = _clip(media, 0, 12, Fraction(1))
        host = _host(media, clip)
        pending = _tool("check_output")(host, {"checks": [WORD_CUT], "wait_seconds": 5})
        self._done(host)
        _call(host, "trim_clip", clip_id=str(clip.id), tail_delta=12)
        result = pending.then()
        assert result["problems"]
        assert "作品が変わりました" in result["note"]
