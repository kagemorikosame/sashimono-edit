"""AI の道具 analyze_audio と audio_graph（音の把握 #322 と、動きと音の確かめ #338）

推移の測り方は tests/engine/test_sound_profile.py、本物の窓と本物の検査は
tests/ui/test_audio_tools.py ここでは偽のホストに推移を入れて、道具が頼む物・使い回す物・
返す形（返事の上限に収まるか）を見る
"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import (
    MAX_RESULT_CHARS,
    ImageResult,
    Operation,
    Pending,
    find_operation,
)
from sashimono.ai.outbound import ASKED, Outbound, SendChoice
from sashimono.ai.safety import Safety
from sashimono.ai.sound_graph import MARGIN_LEFT, plot_width
from sashimono.core.commands import AddClip, AddScene, InScene, insert_scene, new_scene
from sashimono.core.model import MediaItem, Project, ProjectSettings, Track, TrackKind, Transcript
from sashimono.engine.checks import CheckLimits, CheckRequest, CheckState
from sashimono.engine.checks.profile import SoundProfile, SoundProfiler
from sashimono.engine.checks.report import assemble
from sashimono.engine.checks.scan import SOUND_KINDS
from tests.ai.conftest import RATE_30, FakeHost
from tests.ai.test_bridge import call_in_thread, pump_until
from tests.check_media import RATE, chord, talk
from tests.conftest import make_clip

SILENT = np.zeros((RATE, 2), dtype=np.float32)


def _operation(name: str) -> Operation:
    operation = find_operation(name)
    assert operation is not None
    return operation


def _heard(samples: np.ndarray) -> SoundProfile:
    profiler = SoundProfiler(RATE, 2)
    profiler.add(samples)
    return profiler.snapshot()


def _spoken() -> SoundProfile:
    """字幕（1〜3・4〜6・7〜9 秒）のある所で話し、間は無音 最後の 1 秒だけ和音"""
    return _heard(
        np.concatenate(
            [
                SILENT,
                talk(2.0),
                SILENT,
                talk(2.0, seed=1),
                SILENT,
                talk(2.0, seed=2),
                chord(1.0),
            ]
        )
    )


def _voiced(media: MediaItem, transcript: Transcript, seconds: int = 10) -> FakeHost:
    """字幕を付けた素材を音声トラックに 1 本置いた作品の偽のホスト"""
    voiced = replace(media, duration=Fraction(seconds)).with_transcript(transcript)
    base = Project.create(ProjectSettings(frame_rate=RATE_30), media=(voiced,))
    track = Track(kind=TrackKind.AUDIO, name="A1")
    base = base.with_timeline(replace(base.timeline, tracks=(track,)))
    return FakeHost(AddClip(track.id, make_clip(0, seconds * 30, voiced)).apply(base))


def _in_scenes(
    media: MediaItem, transcript: Transcript, *, muted: bool, enabled: bool, nested: bool
) -> FakeHost:
    """字幕を付けた声をシーンの中の音声トラックに置き、そのシーンを外に置いた作品

    ``nested`` ならシーンをもう 1 段のシーンに入れてから置く 声のトラックを消音（``muted``）
    したり、声のクリップを無効（``enabled`` が偽）にしたりできる
    """
    voiced = replace(media, duration=Fraction(10)).with_transcript(transcript)
    base = Project.create(ProjectSettings(frame_rate=RATE_30), media=(voiced,))
    clip = replace(make_clip(0, 300, voiced), enabled=enabled)
    voice = Track(kind=TrackKind.AUDIO, name="声", clips=(clip,), muted=muted)
    inner = replace(new_scene(base, "中"), timeline=replace(base.timeline, tracks=(voice,)))
    project = AddScene(inner).apply(base)
    placed_id = inner.id
    if nested:
        outer = new_scene(project, "外")
        project = AddScene(outer).apply(project)
        inside = replace(project, timeline=project.require_scene(outer.id).timeline)
        for command in insert_scene(inside, inner.id, at_frame=0, duration=300):
            project = InScene(outer.id, command).apply(project)
            inside = replace(project, timeline=project.require_scene(outer.id).timeline)
        placed_id = outer.id
    for command in insert_scene(project, placed_id, at_frame=0, duration=300):
        project = command.apply(project)
    return FakeHost(project)


@pytest.fixture
def host(video_media: MediaItem, transcript: Transcript) -> FakeHost:
    """声のトラックに 10 秒の素材を置いた作品 字幕は 1〜3・4〜6・7〜9 秒"""
    return _voiced(video_media, transcript)


def _done(
    request: CheckRequest, profile: SoundProfile | None, *, running: bool = False
) -> CheckState:
    end = request.end or 300
    report = assemble(
        [],
        start=request.start,
        end=end,
        analyzed=end,
        fps=30.0,
        complete=not running,
        limits=request.limits,
        index=None,
        loudness=None,
        peak=None,
        length_seconds=None,
        sound=profile,
    )
    return CheckState(request=request, report=report, running=running)


class TestAnalyzeAudio:
    def test_it_only_reads(self) -> None:
        # 壊れると、音を聞くだけの道具が「変更」として確かめの窓を出し、自動で承認でも止まる
        for name in ("analyze_audio", "audio_graph"):
            operation = _operation(name)
            assert not operation.writes
            assert operation.safety is Safety.READ_ONLY

    def test_the_kinds_line_up_with_the_timeline(self, host: FakeHost) -> None:
        # 壊れると、話している所・無音の所のフレームがずれ、AI が言葉の途中で切ったり
        # 字幕のある所を音楽と読んで BGM を上げたりする
        host.heard = _spoken()
        result = _operation("analyze_audio")(host, {})
        assert isinstance(result, dict)
        rows = result["segments"]
        assert rows[0]["kind"] == "silence"
        assert (rows[0]["start_frame"], rows[0]["end_frame"]) == (0, 30)
        speech = rows[1]
        assert (speech["kind"], speech["confidence"]) == ("speech", "high")
        assert (speech["start_frame"], speech["end_frame"]) == (30, 90)
        assert speech["subtitles"] == 1.0
        assert speech["timecode"] == "00:00:01:00"
        assert speech["clip_ids"]
        # 字幕のある 6 秒は必ず話し声 終わりの 1 秒の和音は前の話し声に引きずられうる（1 秒より
        # 短い切り替わりは拾わない限り）ので、数に入っても入らなくてもよい
        assert 6.0 <= result["summary"]["seconds_by_kind"]["speech"] <= 7.0
        assert result["complete"] is True

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize(("muted", "enabled"), [(False, True), (True, True), (False, False)])
    def test_only_heard_subtitles_inside_scenes_count(
        self,
        video_media: MediaItem,
        transcript: Transcript,
        muted: bool,
        enabled: bool,
        nested: bool,
    ) -> None:
        # 壊れると、シーンの中で消音した声のトラックや無効にした声のクリップの字幕を根拠に、
        # BGM だけが鳴る所を話し声と返す（字幕を外へ写すとクリップ ID が外側のシーンの物に
        # 書き換わるので、外で鳴るかを照らしても見分けられない）
        host = _in_scenes(video_media, transcript, muted=muted, enabled=enabled, nested=nested)
        host.heard = _heard(chord(10.0))
        result = _operation("analyze_audio")(host, {"points": 0})
        assert isinstance(result, dict)
        heard = not muted and enabled
        covered = [row for row in result["segments"] if row.get("subtitles")]
        assert bool(covered) is heard, result["segments"]

    def test_the_curve_is_bundled_into_points(self, host: FakeHost) -> None:
        # 壊れると、推移を 100 ms ごとに全部返し、長い作品で返事が上限を越えて打ち切られる
        host.heard = _spoken()
        result = _operation("analyze_audio")(host, {"points": 10})
        assert isinstance(result, dict)
        levels = result["levels"]
        assert levels["columns"][0] == "frame"
        assert [row[0] for row in levels["rows"]] == list(range(0, 300, 30))
        # 頭の 1 秒は無音 LUFS は測れない（null）と返す 0 や -inf を返すと AI が数として比べる
        assert levels["rows"][0][1] is None
        assert levels["rows"][2][2] is not None

    def test_silences_are_listed_at_a_tenth_of_a_second(self, host: FakeHost) -> None:
        # 壊れると、話の間を 1 秒単位でしか返さず、切る位置が言葉の頭や終わりにかかる
        host.heard = _heard(np.concatenate([talk(2.0), SILENT[: RATE * 7 // 10], talk(2.0)]))
        result = _operation("analyze_audio")(host, {"list": "silences", "points": 0})
        assert isinstance(result, dict)
        assert "levels" not in result
        # 置いた 0.7 秒の間（2.0〜2.7 秒 = 60〜81 フレーム）を含む無音が 100 ms 刻みで出る
        assert any(
            row["start_frame"] <= 60 and row["end_frame"] >= 81 for row in result["silences"]
        ), result["silences"]
        assert all(row["seconds"] >= 0.5 for row in result["silences"])

    def test_a_finished_check_of_the_same_range_is_reused(self, host: FakeHost) -> None:
        # 壊れると、check_output の直後に音を聞くたびに検査を頼み直し、同じ音を混ぜ直す
        # （長い作品で数分待たせ、AI が頼んだ検査の結果も消える）
        project = host.project
        earlier = CheckRequest(CheckLimits(), project=project, start=0, end=project.duration)
        host.check = _done(earlier, _spoken())
        result = _operation("analyze_audio")(host, {})
        assert isinstance(result, dict)
        assert host.check_requests == []

    def test_a_new_request_mixes_sound_only(self, host: FakeHost) -> None:
        # 壊れると、音を聞くだけで全部のコマを描き、GPU の無い機械では断られ、ある機械でも遅い
        with pytest.raises(ToolError, match="読み始めた"):
            _operation("analyze_audio")(host, {"wait_seconds": 0, "from_frame": 30})
        (request,) = host.check_requests
        assert request.limits.kinds == SOUND_KINDS
        assert not request.limits.needs_picture
        assert request.start == 30

    def test_a_different_range_is_not_reused(self, host: FakeHost) -> None:
        # 壊れると、前に調べた別の範囲の推移を、頼まれた範囲の物として返す
        project = host.project
        earlier = CheckRequest(CheckLimits(), project=project, start=0, end=150)
        host.check = _done(earlier, _spoken())
        host.heard = _spoken()
        _operation("analyze_audio")(host, {})
        (request,) = host.check_requests
        assert (request.start, request.end) == (0, project.duration)

    def test_a_stopped_check_is_not_reused(self, host: FakeHost) -> None:
        # 壊れると、人が〔止める〕で止めた途中までの推移を、ずっと返し続ける
        project = host.project
        earlier = CheckRequest(CheckLimits(), project=project, start=0, end=project.duration)
        host.check = replace(_done(earlier, _spoken()), stopped=True)
        host.heard = _spoken()
        _operation("analyze_audio")(host, {})
        assert len(host.check_requests) == 1

    def test_it_waits_while_the_sound_is_mixed(self, host: FakeHost) -> None:
        # 壊れると、画面のスレッドで解析の終わりを待って画面が固まるか、待たずに空を返す
        result = _operation("analyze_audio")(host, {})
        assert isinstance(result, Pending)
        with pytest.raises(ToolError, match="読み始めた"):
            _operation("analyze_audio")(host, {"wait_seconds": 0})

    def test_a_partial_curve_says_so(self, host: FakeHost) -> None:
        # 壊れると、途中までの推移を全体の物として返し、後半の BGM を AI が見ないまま判断する
        project = host.project
        fresh = CheckRequest(
            CheckLimits(kinds=SOUND_KINDS), project=project, start=0, end=project.duration
        )
        host.check = _done(fresh, _heard(talk(3.0)), running=True)
        result = _operation("analyze_audio")(host, {"wait_seconds": 0})
        assert isinstance(result, dict)
        assert result["complete"] is False
        assert "途中" in result["note"]

    def test_bad_arguments_start_nothing(self, host: FakeHost) -> None:
        # 壊れると、外れた引数でも長い解析を走らせ、AI が頼み直す間も音を混ぜ続ける
        for arguments in (
            {"list": "music"},
            {"points": 999},
            {"silence_db": 0},
            {"wait_seconds": 999},
            {"from_frame": 500},
        ):
            with pytest.raises(ToolError):
                _operation("analyze_audio")(host, arguments)
        assert host.check_requests == []

    def test_no_sound_is_refused_with_the_reason(self, host: FakeHost) -> None:
        # 壊れると、音の無い作品で空の推移を返し、AI が「静かな作品」と読み違える
        project = host.project
        fresh = CheckRequest(
            CheckLimits(kinds=SOUND_KINDS), project=project, start=0, end=project.duration
        )
        state = _done(fresh, None)
        assert state.report is not None
        host.check = replace(
            state, report=replace(state.report, notes=("この動画には音がありません",))
        )
        with pytest.raises(ToolError, match="音がありません"):
            _operation("analyze_audio")(host, {})

    def test_a_long_work_fits_in_one_reply(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、2 時間の作品で推移と区間が返事の上限を越えて打ち切られ、続きも読めない
        hours = 2 * 3600
        host = _voiced(video_media, transcript, hours)
        rng = np.random.default_rng(0)
        # 1 秒ごとに鳴る・鳴らないを切り替えた 2 時間（区間も無音も 3600 ほど）
        ticks = hours * 10
        loud = np.repeat(rng.integers(0, 2, hours), 10).astype(float) * 0.1
        host.heard = SoundProfile(energy=loud**2, rms=loud, peak=loud, flatness=np.full(ticks, 0.1))
        for listing in ("segments", "silences"):
            result = _operation("analyze_audio")(
                host, {"list": listing, "points": 200, "limit": 300}
            )
            assert isinstance(result, dict)
            assert len(json.dumps(result, ensure_ascii=False)) <= MAX_RESULT_CHARS
            assert result["next_offset"] > 0
            assert len(result["levels"]["rows"]) >= 10


class TestAudioGraph:
    def test_the_graph_is_a_picture_with_marks(self, host: FakeHost) -> None:
        # 壊れると、動き（preview_range のコマ）と音を同じ時刻で並べて見られず、場面の
        # 切り替わりと音のずれを AI が確かめられない
        host.heard = _spoken()
        result = _operation("audio_graph")(host, {"marks": [30, 150], "width": 800})
        assert isinstance(result, ImageResult)
        (graph,) = host.graphs
        columns = plot_width(800)
        assert graph.width == 800
        assert len(graph.momentary) == columns
        assert [x for x, _label in graph.marks] == [
            MARGIN_LEFT + round(30 * columns / 300),
            MARGIN_LEFT + round(150 * columns / 300),
        ]
        assert graph.bands[0].label == "silence"
        assert graph.bands[0].left == MARGIN_LEFT
        assert graph.bands[-1].right == MARGIN_LEFT + columns
        assert graph.subtitles
        assert graph.target == pytest.approx(-14.0)
        assert graph.time_ticks[0][1] == "00:00:00:00"
        assert graph.analyzed_until is None
        assert "1) 00:00:01:00（30）" in result.caption
        assert "2) 00:00:05:00（150）" in result.caption

    def test_the_graph_fits_the_set_image_edge(self, host: FakeHost) -> None:
        # 壊れると、絵の長い辺を 768 に下げた人にも既定で 1200 画素のグラフを送り、
        # 選んだ費用の上限が音量のグラフだけ効かない（#344）
        host.heard = _spoken()
        host.image_edge = 768
        result = _operation("audio_graph")(host, {"marks": [30]})
        assert isinstance(result, ImageResult)
        (graph,) = host.graphs
        assert max(graph.width, graph.height) <= 768
        assert graph.width == 768
        assert "768" in result.caption
        # 縦線は縮めた幅の上で同じ時刻に引く 元の幅のまま置くと、縦線が音とずれる
        assert graph.marks[0][0] == MARGIN_LEFT + round(30 * plot_width(768) / 300)

    def test_a_graph_within_the_edge_is_untouched(self, host: FakeHost) -> None:
        # 壊れると、上限の内のグラフまで縮めて送り、短い山や谷が潰れて見えなくなる
        host.heard = _spoken()
        result = _operation("audio_graph")(host, {"width": 800})
        assert isinstance(result, ImageResult)
        (graph,) = host.graphs
        assert graph.width == 800
        assert "縮め" not in result.caption

    def test_marks_outside_the_range_are_refused_before_mixing(self, host: FakeHost) -> None:
        # 壊れると、範囲の外の縦線を端へ寄せて描き、AI が違う時刻の音を見て判断する
        with pytest.raises(ToolError, match="範囲"):
            _operation("audio_graph")(host, {"marks": [400]})
        assert host.check_requests == []
        assert host.graphs == []

    def test_a_partial_graph_greys_out_the_rest(self, host: FakeHost) -> None:
        # 壊れると、まだ測っていない後半を無音のように描き、AI が「後半は音が無い」と読む
        project = host.project
        fresh = CheckRequest(
            CheckLimits(kinds=SOUND_KINDS), project=project, start=0, end=project.duration
        )
        host.check = _done(fresh, _heard(talk(3.0)), running=True)
        result = _operation("audio_graph")(host, {"wait_seconds": 0, "width": 800})
        assert isinstance(result, ImageResult)
        (graph,) = host.graphs
        assert graph.analyzed_until == MARGIN_LEFT + round(30 * plot_width(800) / 100)
        assert all(value is None for value in graph.momentary[-10:])
        assert "まだ解析していない" in result.caption

    def test_a_clipped_peak_is_marked(self, host: FakeHost) -> None:
        # 壊れると、音割れの所がグラフに出ず、絵で確かめた AI が割れを見逃す
        samples = np.concatenate([talk(5.0), np.ones((RATE, 2), np.float32), talk(4.0)])
        host.heard = _heard(samples)
        _operation("audio_graph")(host, {"width": 800})
        (graph,) = host.graphs
        assert graph.clipped
        middle = MARGIN_LEFT + round(55 * plot_width(800) / 100)
        assert any(abs(x - middle) < plot_width(800) // 10 for x in graph.clipped)


class TestSending:
    def test_the_graph_is_sent_as_a_picture(self) -> None:
        # 壊れると、音のグラフ（画像）を同意なしで送る 絵は伏せられないので、宣言で尋ねる
        assert Outbound.FRAMES in _operation("audio_graph").sends

    def test_the_numbers_are_sent_as_structure(self) -> None:
        # 壊れると、絵を送らないと決めた人が、音の推移を数でも読めなくなる
        assert Outbound.FRAMES not in _operation("analyze_audio").sends
        assert Outbound.PROJECT in _operation("analyze_audio").sends

    def test_denying_pictures_still_lets_the_numbers_through(self, host: FakeHost) -> None:
        # 壊れると、絵を断った人の作品で音の推移を読めず、音量をそろえる判断ができない
        host.heard = _spoken()
        bridge = EditorBridge(host)
        bridge.set_send_choices(
            {**dict.fromkeys(ASKED, SendChoice.ALLOW), Outbound.FRAMES: SendChoice.DENY}
        )
        done, box = call_in_thread(bridge, _operation("audio_graph"), {})
        pump_until(bridge, done)
        assert isinstance(box.get("error"), ToolError)
        assert host.graphs == []
        done, box = call_in_thread(bridge, _operation("analyze_audio"), {})
        pump_until(bridge, done)
        result: Any = box.get("result")
        assert isinstance(result, dict), box.get("error")
        assert result["segments"]
