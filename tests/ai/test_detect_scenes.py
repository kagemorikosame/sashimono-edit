"""AI の道具 detect_scenes（#319 #321）

場面の切り替わりの一覧を、タイムラインのフレームと一緒に返すこと・解析が終わって
いなければ AI のスレッドで待ってから返すこと・待ちきれなければ読んだ所までを返す
ことを見る 最後の組は、その場で焼いた動画を本物の解析係に読ませて道具まで通す
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Operation, Pending, find_operation
from sashimono.core.commands import AddClip, MoveClip, RemoveClip, RemoveMedia, SetClipProperty
from sashimono.core.model import (
    ClipId,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from sashimono.engine.cache import CacheStore, MediaAnalyzer, SceneState
from sashimono.engine.decode import forget_probe, probe_media
from sashimono.engine.scenes import SceneSignal
from tests.ai.conftest import FakeHost, allow_sending
from tests.ai.test_bridge import call_in_thread, pump_until
from tests.conftest import make_clip
from tests.scene_video import CUTS, FPS, write_scene_video


def _operation() -> Operation:
    found = find_operation("detect_scenes")
    assert found is not None
    return found


DETECT = _operation()


def _signal(cuts: list[float], *, until: float = 10.0, complete: bool = True) -> SceneSignal:
    """30fps で ``until`` 秒まで ``cuts`` の秒で跳ねる差の並び"""
    times = np.arange(1, round(until * FPS) + 1, dtype=np.float64) / FPS
    pixel = np.full(times.shape, 0.01, np.float32)
    histogram = np.full(times.shape, 0.01, np.float32)
    for moment in cuts:
        index = round(moment * FPS) - 1
        pixel[index], histogram[index] = 0.3, 0.8
    return SceneSignal(times, pixel, histogram, analyzed=until, complete=complete)


def _media(host: FakeHost) -> MediaItem:
    return host.project.media[0]


def _call(host: FakeHost, **arguments: Any) -> dict[str, Any]:
    result = DETECT(host, {"media_id": str(_media(host).id), **arguments})
    assert isinstance(result, dict), result
    return result


class TestTheList:
    def test_it_only_reads(self) -> None:
        # 変更の道具にすると、場面を見るたびに許可を求められる
        assert not DETECT.writes and not DETECT.asks

    def test_it_asks_for_the_analysis(self, host: FakeHost) -> None:
        # 頼まないと、取り込み時の解析を切った人のところでいつまでも一覧が出ない
        _call(host, wait_seconds=0)
        assert host.scene_requests == [_media(host).id]

    def test_the_cuts_come_with_timeline_frames(self, host: FakeHost) -> None:
        # タイムラインの位置が無いと、AI が素材の秒からフレームを数え直して外す
        host.scene_states[_media(host).id] = SceneState(_signal([2.0, 6.5]), running=False)
        result = _call(host)
        assert result["complete"] is True
        assert [row["source_seconds"] for row in result["cuts"]] == [2.0, 6.5]
        frames = [[place["frame"] for place in row["timeline"]] for row in result["cuts"]]
        assert frames == [[60], [195]]

    def test_the_cut_lies_after_the_previous_frame(self, host: FakeHost) -> None:
        # 前のフレームの時刻が無いと、AI は切り替わりの前の絵をどこで見ればよいか分からない
        host.scene_states[_media(host).id] = SceneState(_signal([2.0]), running=False)
        row = _call(host)["cuts"][0]
        assert row["after_seconds"] == pytest.approx(2.0 - 1 / FPS, abs=0.001)

    def test_a_moved_and_trimmed_clip_is_followed(self, host: FakeHost) -> None:
        # クリップの位置と素材の頭を見ないと、動かしたクリップで違うフレームを見せる
        media = _media(host)
        clip = host.project.timeline.tracks[0].clips[0]
        host.apply_commands(
            [
                MoveClip(clip.id, 100),
                SetClipProperty(clip.id, "source_in", Fraction(1)),
            ],
            "動かす",
        )
        host.scene_states[media.id] = SceneState(_signal([0.5, 2.0]), running=False)
        rows = _call(host)["cuts"]
        assert rows[0]["timeline"] == []  # 素材の頭を切ったので見えない
        assert [place["frame"] for place in rows[1]["timeline"]] == [130]

    def test_a_faster_clip_is_followed(self, host: FakeHost) -> None:
        # 速さを見ないと、2 倍速のクリップで 2 倍後ろのフレームを見せる
        media = _media(host)
        clip = host.project.timeline.tracks[0].clips[0]
        host.apply_commands([SetClipProperty(clip.id, "speed", Fraction(2))], "速く")
        host.scene_states[media.id] = SceneState(_signal([4.0]), running=False)
        assert [p["frame"] for p in _call(host)["cuts"][0]["timeline"]] == [60]

    def test_sound_only_clips_are_not_places(self, video_media: MediaItem) -> None:
        # 音だけのクリップの位置を返すと、AI はその場面の絵が無いフレームを見る
        project = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(video_media,))
        audio = Track(kind=TrackKind.AUDIO, name="A1")
        project = project.with_timeline(replace(project.timeline, tracks=(audio,)))
        project = AddClip(audio.id, make_clip(0, 300, video_media)).apply(project)
        host = FakeHost(project)
        host.scene_states[video_media.id] = SceneState(_signal([2.0]), running=False)
        assert _call(host)["cuts"][0]["timeline"] == []

    def test_only_clips_of_the_read_stream_are_places(self, video_media: MediaItem) -> None:
        # 映像が 2 本ある素材で、調べていない 2 本目を出すクリップにも位置を付けると、
        # AI は別の映像の絵で切り替わりを確かめ、そこに無いカットで切る（PR #358 の Codex の指摘）
        first = video_media.video_streams[0]
        second = replace(first, index=2)
        media = replace(video_media, video_streams=(first, second))
        project = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(media,))
        video = Track(kind=TrackKind.VIDEO, name="V1")
        project = project.with_timeline(replace(project.timeline, tracks=(video,)))
        shown_first = replace(make_clip(0, 300, media), id=ClipId("first"))
        shown_second = replace(make_clip(300, 300, media), id=ClipId("second"), stream_index=2)
        project = AddClip(video.id, shown_first).apply(project)
        project = AddClip(video.id, shown_second).apply(project)
        host = FakeHost(project)
        host.scene_states[media.id] = SceneState(_signal([2.0]), running=False)
        result = _call(host)
        assert [place["clip_id"] for place in result["cuts"][0]["timeline"]] == ["first"]
        assert result["video_stream"] == 1

    def test_a_broken_video_says_no_more_will_come(self, host: FakeHost) -> None:
        # 「待てば続きが出る」と読ませると、壊れた素材で AI が同じ道具を呼び直し続ける
        signal = replace(_signal([2.0], until=4.0, complete=False), broken=True)
        host.scene_states[_media(host).id] = SceneState(signal, running=False)
        result = _call(host)
        assert result["complete"] is False
        assert [row["source_seconds"] for row in result["cuts"]] == [2.0]
        assert "壊れて" in result["note"] and "続きは出ません" in result["note"]

    def test_a_higher_threshold_gives_fewer(self, host: FakeHost) -> None:
        # 感度が効かないと、AI は細かすぎる一覧を絞れない
        signal = _signal([2.0, 6.0])
        signal.pixel[round(6.0 * FPS) - 1] = 0.08
        signal.histogram[round(6.0 * FPS) - 1] = 0.2
        host.scene_states[_media(host).id] = SceneState(signal, running=False)
        assert len(_call(host)["cuts"]) == 2
        assert len(_call(host, threshold=0.3)["cuts"]) == 1

    def test_the_range_is_in_source_seconds(self, host: FakeHost) -> None:
        # 範囲が効かないと、長い素材で一覧が返事の長さの上限に当たって切れる
        host.scene_states[_media(host).id] = SceneState(_signal([1.0, 5.0, 9.0]), running=False)
        rows = _call(host, from_seconds=2.0, to_seconds=9.0)["cuts"]
        assert [row["source_seconds"] for row in rows] == [5.0]

    def test_none_found_says_how_to_find_more(self, host: FakeHost) -> None:
        # 空の一覧だけを返すと、AI は場面が 1 つしかない素材だと決めてしまう
        host.scene_states[_media(host).id] = SceneState(_signal([]), running=False)
        result = _call(host)
        assert result["cuts"] == [] and "threshold" in result["note"]


class TestArguments:
    @pytest.mark.parametrize(
        ("arguments", "word"),
        [
            ({"threshold": "0.3"}, "数"),
            ({"threshold": True}, "数"),
            ({"threshold": 0.001}, "0.02"),
            ({"min_scene_seconds": -1}, "min_scene_seconds"),
            ({"wait_seconds": 500}, "120"),
            ({"from_seconds": 5, "to_seconds": 5}, "to_seconds"),
            ({"to_seconds": float("nan")}, "to_seconds"),
        ],
    )
    def test_a_wrong_value_is_refused_with_the_range(
        self, host: FakeHost, arguments: dict[str, Any], word: str
    ) -> None:
        # 黙って丸めると、AI は頼んだ条件で探した一覧だと思って読む
        with pytest.raises(ToolError, match=word):
            _call(host, **arguments)

    def test_a_sound_only_media_is_refused(self, host: FakeHost, audio_media: MediaItem) -> None:
        # 音だけの素材で空の一覧を返すと、AI は場面の切り替わりの無い動画だと受け取る
        project = replace(host.project, media=(*host.project.media, audio_media))
        other = FakeHost(project)
        with pytest.raises(ToolError, match="映像がありません"):
            DETECT(other, {"media_id": str(audio_media.id)})

    def test_an_unknown_media_is_refused(self, host: FakeHost) -> None:
        # 直し方を書かずに断ると、AI は当てずっぽうの ID で呼び直し続ける
        with pytest.raises(ToolError, match="list_media"):
            DETECT(host, {"media_id": "nothing"})


class TestWaiting:
    def test_a_running_analysis_is_waited_for(self, host: FakeHost) -> None:
        # 画面のスレッドで待つと、待つ間は画面が固まる 待つのはブリッジ（AI のスレッド）
        host.scene_states[_media(host).id] = SceneState(None, running=True)
        pending = DETECT(host, {"media_id": str(_media(host).id)})
        assert isinstance(pending, Pending)
        assert pending.seconds == 30

    def test_not_waiting_returns_at_once(self, host: FakeHost) -> None:
        # wait_seconds 0 で待つと、急ぎの AI が止まる
        host.scene_states[_media(host).id] = SceneState(None, running=True)
        assert isinstance(_call(host, wait_seconds=0), dict)

    def test_the_bridge_returns_what_was_read_after_the_wait(self, host: FakeHost) -> None:
        # 待ちきれないときに何も返さないと、長い素材で AI が一覧を 1 つも受け取れない
        media = _media(host)
        host.scene_states[media.id] = SceneState(
            _signal([2.0], until=4.0, complete=False), running=True
        )
        host.scene_ready = lambda _media: False
        bridge = allow_sending(EditorBridge(host))
        began = time.monotonic()
        done, box = call_in_thread(bridge, DETECT, {"media_id": str(media.id), "wait_seconds": 0.3})
        pump_until(bridge, done)
        assert time.monotonic() - began < 3.0
        result = box["result"]
        assert isinstance(result, dict)
        assert result["complete"] is False
        assert [row["source_seconds"] for row in result["cuts"]] == [2.0]
        assert "途中" in result["note"]
        assert host.scene_waits and all(seconds <= 0.3 for _media, seconds in host.scene_waits)

    def test_the_bridge_returns_the_finished_list(self, host: FakeHost) -> None:
        # 終わったのに途中の一覧を返すと、AI は続きを取りに同じ道具を呼び直す
        media = _media(host)
        host.scene_states[media.id] = SceneState(None, running=True)

        def finish(_media: MediaId) -> bool:
            host.scene_states[media.id] = SceneState(_signal([2.0]), running=False)
            return True

        host.scene_ready = finish
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, DETECT, {"media_id": str(media.id)})
        pump_until(bridge, done)
        result = box["result"]
        assert isinstance(result, dict) and result["complete"] is True

    def test_cancelling_ends_the_wait(self, host: FakeHost) -> None:
        # 中断のボタンが待つ間に効かないと、最長 2 分止まったままになる
        media = _media(host)
        host.scene_states[media.id] = SceneState(None, running=True)

        def never(_media: MediaId) -> bool:
            # 本物の待ち方と同じく、少し待ってから「まだ」と返す
            time.sleep(0.05)
            return False

        host.scene_ready = never
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, DETECT, {"media_id": str(media.id), "wait_seconds": 60})
        # 待ちに入ったことを確かめてから止める 時間で区切って先へ進むと、遅い機械では待ちに
        # 入る前に止めてしまい、待つ間の中断を確かめないまま通る（#382 で同じ形の試験が揺れた）
        deadline = time.monotonic() + 30
        while not host.scene_waits and not done.is_set() and time.monotonic() < deadline:
            bridge.pump()
            time.sleep(0.01)
        assert host.scene_waits, f"待ちに入らなかった: {box}"
        assert not done.is_set(), f"止める前に抜けた: {box}"
        bridge.cancel()
        # 待つ長さ（60 秒）まで待たずに抜けること 上限は遅い機械でも揺れない広さにする
        pump_until(bridge, done, timeout=30.0)
        assert done.is_set(), "中断を押しても待ち続けた"
        assert isinstance(box.get("error"), ToolError)

    def test_a_failure_is_told(self, host: FakeHost) -> None:
        # 理由を伝えないと、AI は出ない一覧を待ち直し続ける
        host.scene_states[_media(host).id] = SceneState(None, running=False, failure="開けない")
        with pytest.raises(ToolError, match="開けない"):
            _call(host)

    def test_media_removed_while_waiting_is_told(self, host: FakeHost) -> None:
        # 外した素材の一覧を返すと、AI は一覧に無い素材の場面を切りに行く
        media = _media(host)
        host.scene_states[media.id] = SceneState(None, running=True)
        pending = DETECT(host, {"media_id": str(media.id)})
        assert isinstance(pending, Pending)
        clips = [clip.id for track in host.project.timeline.tracks for clip in track.clips]

        host.apply_commands([RemoveClip(clip) for clip in clips] + [RemoveMedia(media.id)], "外す")
        with pytest.raises(ToolError, match="見つかりません"):
            pending.then()


class _AnalyzingHost(FakeHost):
    """場面の解析だけ本物の係に任せる偽のホスト"""

    def __init__(self, project: Project, analyzer: MediaAnalyzer) -> None:
        super().__init__(project)
        self.analyzer = analyzer

    def request_scenes(self, media: MediaItem) -> None:
        self.analyzer.request_scenes(media)

    def scene_state(self, media: MediaItem) -> SceneState:
        return self.analyzer.scene_state(media)

    def scene_waiter(self, media_id: MediaId) -> Callable[[float], bool]:
        analyzer = self.analyzer
        return lambda seconds: analyzer.wait_scenes(media_id, seconds)


class TestWithARealVideo:
    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> Iterator[MediaAnalyzer]:
        created = MediaAnalyzer(CacheStore(tmp_path / "cache"))
        yield created
        created.close()

    def test_the_made_video_gives_every_cut(self, tmp_path: Path, analyzer: MediaAnalyzer) -> None:
        # 本物の素材で道具まで通らないと、AI が実際の動画で場面ごとに切れない
        path = write_scene_video(tmp_path / "scenes.mp4")
        forget_probe(path)
        media = probe_media(path)
        project = Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(media,))
        track = Track(kind=TrackKind.VIDEO, name="V1")
        project = project.with_timeline(replace(project.timeline, tracks=(track,)))
        clip = replace(make_clip(30, 360, media), id=ClipId("clip"))
        project = AddClip(track.id, clip).apply(project)
        host = _AnalyzingHost(project, analyzer)
        bridge = allow_sending(EditorBridge(host))
        done, box = call_in_thread(bridge, DETECT, {"media_id": str(media.id)})
        pump_until(bridge, done, timeout=60)
        result = box.get("result")
        assert isinstance(result, dict), box
        assert result["complete"] is True
        seconds = [row["source_seconds"] for row in result["cuts"]]
        assert seconds == pytest.approx(list(CUTS), abs=0.5 / FPS)
        frames = [row["timeline"][0]["frame"] for row in result["cuts"]]
        assert frames == [30 + round(cut * FPS) for cut in CUTS]
