"""AI の道具 find_highlights（#325）

盛り上がりの区間を、素材を読み直さずに覚えてある波形と場面の解析から返すこと・
タイムラインの範囲を添えること・映像の解析を待つこと・返事が上限に収まることを見る
"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.ledger import AI_ONLY
from sashimono.ai.operations import MAX_RESULT_CHARS, Operation, Pending, find_operation
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddClip, MoveClip, SetClipProperty, SetTranscript
from sashimono.core.io.serialize import json_text
from sashimono.core.model import MediaItem, Track, TrackKind, Transcript
from sashimono.engine.cache import SceneState
from sashimono.engine.scenes import SceneSignal
from tests.ai.conftest import FakeHost, make_loaded
from tests.conftest import make_clip
from tests.highlight_sound import laugh, speech, waveform_of


def _operation() -> Operation:
    found = find_operation("find_highlights")
    assert found is not None
    return found


FIND = _operation()


def _media(host: FakeHost) -> MediaItem:
    return host.project.media[0]


def _call(host: FakeHost, **arguments: Any) -> dict[str, Any]:
    result = FIND(host, {"media_id": str(_media(host).id), **arguments})
    assert isinstance(result, dict), result
    return result


def _laughing(host: FakeHost, at: float = 4.0) -> None:
    """10 秒の素材の ``at`` 秒から 2.5 秒笑う波形を置く"""
    rng = np.random.default_rng(1)
    samples = np.concatenate([speech(at, rng), laugh(2.5, rng), speech(10.0 - at - 2.5, rng)])
    host.stub_waveform = waveform_of(samples)


def _still(
    until: float = 10.0, busy: tuple[float, float] | None = None, *, complete: bool = True
) -> SceneSignal:
    """30fps の据え置きの画面 ``busy`` の秒だけ大きく動く

    ``complete`` が偽なら ``until`` まで読んだ途中
    """
    times = np.arange(1, round(until * 30) + 1, dtype=np.float64) / 30
    pixel = np.full(times.shape, 0.005, np.float32)
    if busy is not None:
        pixel[(times >= busy[0]) & (times < busy[1])] = 0.08
    return SceneSignal(times, pixel, np.full(times.shape, 0.01, np.float32), until, complete)


class TestTheTool:
    def test_it_only_reads(self) -> None:
        """読むだけの道具

        壊れると、見どころを探すたびに変更の許可を求められ、取り消しの履歴も汚れる
        """
        assert not FIND.writes and not FIND.asks
        assert FIND.safety is Safety.READ_ONLY

    def test_it_is_in_the_ledger(self) -> None:
        """操作の台帳に載る

        壊れると、画面と AI の操作の突き合わせ（#347）で抜けとして数えられない
        """
        assert "find_highlights" in AI_ONLY

    def test_the_limits_are_told(self) -> None:
        """笑い声の見分けの限りを説明に書く

        壊れると、AI が点を確かな笑いと思い込んで切り抜く
        """
        assert "話し声" in FIND.description and "音楽" in FIND.description


class TestTheList:
    def test_a_laugh_comes_with_timeline_frames(self, host: FakeHost) -> None:
        """笑いの区間をタイムラインの範囲と一緒に返す

        壊れると、AI が素材の秒からフレームを数え直して外す
        """
        _laughing(host)
        host.scene_states[_media(host).id] = SceneState(_still(), running=False)
        result = _call(host)
        assert result["complete"] is True
        row = result["highlights"][0]
        assert row["reasons"][0] == "laughter"
        assert 3.5 <= row["start_seconds"] <= 4.5
        place = row["timeline"][0]
        assert place["start_frame"] == round(row["start_seconds"] * 30)
        assert place["start_frame"] <= place["peak_frame"] < place["end_frame"]
        assert "text" not in json.dumps(row)

    def test_motion_is_found_without_sound(self, host: FakeHost) -> None:
        """音が無くても動きで拾う 壊れると、波形がまだの素材で動きの区間も返さない"""
        host.scene_states[_media(host).id] = SceneState(_still(busy=(6.0, 8.0)), running=False)
        result = _call(host)
        assert [row["reasons"] for row in result["highlights"]] == [["motion"]]
        assert result["complete"] is False
        # 字幕があるので笑いは字幕の印で調べ、音量だけが調べられない
        assert result["unavailable"] == ["loud"]
        assert result["signals"] == ["laughter", "motion"]
        assert host.analyzed == [_media(host).id]

    def test_a_moved_and_faster_clip_is_followed(self, host: FakeHost) -> None:
        """動かした・速さを変えたクリップに合わせる

        壊れると、違うフレームを preview_range で見せる
        """
        clip = host.project.timeline.tracks[0].clips[0]
        host.apply_commands(
            [MoveClip(clip.id, 100), SetClipProperty(clip.id, "speed", Fraction(2))], "動かす"
        )
        host.scene_states[_media(host).id] = SceneState(_still(busy=(6.0, 8.0)), running=False)
        row = _call(host, signals=["motion"])["highlights"][0]
        place = row["timeline"][0]
        expected = 100 + round((row["start_seconds"]) * 30 / 2)
        assert place["start_frame"] == pytest.approx(expected, abs=1)

    def test_a_linked_sound_clip_is_not_listed_twice(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """絵と音の 2 本に分けて置いた所を 1 つにまとめる

        壊れると、同じ所が 2 か所あると読ませる
        """
        project = make_loaded(video_media, transcript)
        media = project.media[0]
        audio = Track(kind=TrackKind.AUDIO, name="A1")
        project = project.with_timeline(
            replace(project.timeline, tracks=(*project.timeline.tracks, audio))
        )
        host = FakeHost(AddClip(audio.id, make_clip(0, 300, media)).apply(project))
        host.scene_states[media.id] = SceneState(_still(busy=(6.0, 8.0)), running=False)
        row = _call(host)["highlights"][0]
        assert len(row["timeline"]) == 1

    def test_order_by_time(self, host: FakeHost) -> None:
        """強い順と時刻の順を選べる 壊れると、頭から順に見たい頼みで並びがばらばらになる"""
        _laughing(host, at=6.0)
        host.scene_states[_media(host).id] = SceneState(_still(busy=(1.0, 3.0)), running=False)
        by_score = [row["start_seconds"] for row in _call(host)["highlights"]]
        by_time = [row["start_seconds"] for row in _call(host, order="time")["highlights"]]
        assert by_time == sorted(by_time)
        assert len(by_time) == 2
        assert sorted(by_score) == by_time

    def test_only_sound_does_not_ask_for_pictures(self, host: FakeHost) -> None:
        """音だけを頼んだら映像の解析を頼まない

        壊れると、音を見たいだけで長い映像の読み込みが始まる
        """
        _laughing(host)
        _call(host, signals=["laughter", "loud"])
        assert host.scene_requests == []

    def test_subtitles_alone_find_a_laugh(self, host: FakeHost) -> None:
        """波形がまだでも、字幕の笑いの印から笑いを返す 壊れると、待つ間は何も返せない"""
        media = _media(host)
        transcript = media.transcript
        assert transcript is not None
        marked = replace(transcript.segments[1], text="（笑）")
        changed = replace(
            transcript, segments=(transcript.segments[0], marked, transcript.segments[2])
        )
        host.apply_commands([SetTranscript(media.id, changed)], "字幕")
        result = _call(host, signals=["laughter"])
        row = result["highlights"][0]
        assert row["laughter_from_subtitles"] is True
        assert row["start_seconds"] == pytest.approx(4.0)
        assert "（笑）" not in json.dumps(result, ensure_ascii=False)

    def test_subtitles_are_kept_beside_motion(self, host: FakeHost) -> None:
        """波形がまだで動きを見られるときも、字幕の笑いの印を使う

        壊れると、映像の素材を取り込んだ直後に頼むと笑いの主な手がかりを黙って落とし、
        字幕に「（笑）」がある所を盛り上がりとして返さない
        """
        media = _media(host)
        transcript = media.transcript
        assert transcript is not None
        marked = replace(transcript.segments[1], text="（笑）")
        changed = replace(
            transcript, segments=(transcript.segments[0], marked, transcript.segments[2])
        )
        host.apply_commands([SetTranscript(media.id, changed)], "字幕")
        host.scene_states[media.id] = SceneState(_still(busy=(7.5, 9.0)), running=False)
        result = _call(host, order="time")
        assert "laughter" in result["signals"]
        assert "laughter" not in result["unavailable"]
        rows = result["highlights"]
        assert rows[0]["laughter_from_subtitles"] is True
        assert rows[0]["start_seconds"] == pytest.approx(4.0)
        assert any("motion" in row["reasons"] for row in rows)


class TestWaiting:
    def test_it_waits_for_the_pictures(self, host: FakeHost) -> None:
        """映像の解析の途中なら待ってから返す

        壊れると、読み込んだ直後に頼むと動きが空のまま返る
        """
        media = _media(host)
        host.scene_states[media.id] = SceneState(_still(until=3.0, complete=False), running=True)
        pending = FIND(host, {"media_id": str(media.id)})
        assert isinstance(pending, Pending)
        host.scene_states[media.id] = SceneState(_still(busy=(6.0, 8.0)), running=False)
        assert pending.wait(1.0)
        result = pending.then()
        assert isinstance(result, dict)
        assert result["highlights"][0]["reasons"] == ["motion"]

    def test_a_partial_scan_is_told(self, host: FakeHost) -> None:
        """待ちきれなければ読んだ所までで返し、途中だと書く

        壊れると、後ろ半分に盛り上がりは無いと読ませる
        """
        _laughing(host)
        media = _media(host)
        host.scene_states[media.id] = SceneState(_still(until=3.0, complete=False), running=True)
        result = _call(host, wait_seconds=0)
        assert result["complete"] is False
        assert "3.0 秒" in result["note"]


class TestRefusals:
    def test_nothing_to_look_at_is_refused(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """調べられる手がかりが無ければ断る

        壊れると、空の一覧を「盛り上がりは無い」と読ませる
        """
        host = FakeHost(make_loaded(replace(video_media, audio_streams=()), transcript))
        with pytest.raises(ToolError, match="調べられません"):
            _call(host, signals=["laughter", "loud"])

    @pytest.mark.parametrize(
        "arguments",
        [
            {"signals": ["cheer"]},
            {"signals": []},
            {"threshold": 0.01},
            {"order": "loudness"},
            {"from_seconds": 5, "to_seconds": 2},
            {"audio": 3},
        ],
    )
    def test_bad_arguments_are_refused(self, host: FakeHost, arguments: dict[str, Any]) -> None:
        """外れた引数は寄せずに断る 壊れると、AI が頼んだ条件で調べたと思い込む"""
        host.scene_states[_media(host).id] = SceneState(_still(), running=False)
        with pytest.raises(ToolError):
            _call(host, **arguments)


class TestSize:
    def test_a_long_list_fits_the_reply(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        """区間が何百あっても返事を上限に収め、続きの取り方を書く

        壊れると、返事が途中で切れて読めない
        """
        host = FakeHost(make_loaded(replace(video_media, duration=Fraction(3600)), transcript))
        media = _media(host)
        times = np.arange(1, 3600 * 30 + 1, dtype=np.float64) / 30
        pixel = np.full(times.shape, 0.005, np.float32)
        # 6 秒ごとに 2 秒動く 600 の区間
        pixel[(times % 6.0) < 2.0] = 0.08
        signal = SceneSignal(times, pixel, np.full(times.shape, 0.01, np.float32), 3600.0, True)
        host.scene_states[media.id] = SceneState(signal, running=False)
        result = _call(host, limit=300)
        assert len(json_text(result)) <= MAX_RESULT_CHARS
        assert result["total"] >= 500
        assert "next_offset" in result
