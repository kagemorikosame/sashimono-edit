"""AI の道具 reframe（#334 横の作品を縦の作品に作り替える）

道具の受け口（引数・本人の設定・断る所）・解析を待つ形・取り消し 1 回で戻ることを、偽の
ホストで見る 枠の位置の計算そのものは tests/engine/test_reframe.py と tests/core/test_reframe.py
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from sashimono.ai.operations import Operation, Pending, find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddScene, RemoveClip
from sashimono.core.model import MediaItem, Scene, Timeline
from sashimono.core.reframe import ReframeStyle
from sashimono.engine.cache import SceneState, ScreenState
from sashimono.engine.scenes import SceneSignal
from sashimono.engine.subject import SUBJECT_BINS, SUBJECT_SECONDS, SubjectTrack
from sashimono.engine.vision import Box, Face, ScreenKind, ScreenReport, ScreenSample
from tests.ai.conftest import RATE_30, FakeHost, allow_sending
from tests.ai.test_bridge import call_in_thread, pump_until


def _operation() -> Operation:
    found = find_operation("reframe")
    assert found is not None
    return found


REFRAME = _operation()


def _media(host: FakeHost) -> MediaItem:
    return host.project.media[0]


def _signal(across: float) -> SceneSignal:
    times = np.arange(SUBJECT_SECONDS / 2, 10.0, SUBJECT_SECONDS)
    columns = np.zeros((times.shape[0], SUBJECT_BINS), np.float32)
    columns[:, int(across * SUBJECT_BINS)] = 1.0
    rows = np.zeros((times.shape[0], SUBJECT_BINS), np.float32)
    rows[:, SUBJECT_BINS // 2] = 1.0
    subject = SubjectTrack(times, columns, rows, np.full(times.shape, 0.02, np.float32))
    frames = np.arange(1, 301, dtype=np.float64) / 30
    flat = np.full(frames.shape, 0.01, np.float32)
    return SceneSignal(frames, flat, flat, analyzed=10.0, complete=True, stream=0, subject=subject)


def _call(host: FakeHost, **arguments: Any) -> dict[str, Any]:
    result = REFRAME(host, arguments)
    assert isinstance(result, dict), result
    return result


class TestTheDeclaration:
    def test_it_is_an_undoable_edit(self) -> None:
        # 取り消せない操作にすると、自動で承認の人にも毎回確かめが出る
        # 読むだけにすると確かめずに作品を変える
        assert REFRAME.writes and not REFRAME.asks
        assert REFRAME.safety is Safety.UNDOABLE

    def test_it_sends_the_structure_and_the_settings(self) -> None:
        # 本人の設定の既定を返事に載せるのに宣言しないと、設定画面の一覧に無い物が外へ出る
        assert REFRAME.sends == frozenset({Outbound.PROJECT, Outbound.APP_SETTINGS})


class TestTheAnalysis:
    def test_a_running_analysis_is_waited_for(self, host: FakeHost) -> None:
        # 画面のスレッドで待つと、待つ間は画面が固まる 待つのはブリッジ（AI のスレッド）
        host.scene_states[_media(host).id] = SceneState(None, running=True)
        pending = REFRAME(host, {})
        assert isinstance(pending, Pending)
        assert host.scene_requests == [_media(host).id]
        assert host.project.settings.resolution == (1920, 1080)

    def test_nothing_changes_while_the_analysis_runs(self, host: FakeHost) -> None:
        # 途中までの並びで作り替えると、読んでいない所の枠が真ん中に止まり、呼び直しても直らない
        host.scene_states[_media(host).id] = SceneState(None, running=True)
        result = _call(host, wait_seconds=0)
        assert result["changed"] is False
        assert result["waiting_media"] == [str(_media(host).id)]
        assert host.project.settings.resolution == (1920, 1080)

    def test_center_does_not_wait(self, host: FakeHost) -> None:
        # 真ん中を頼まれても解析を待つと、急ぎの人が何十秒も待たされる
        host.scene_states[_media(host).id] = SceneState(None, running=True)
        result = _call(host, follow="center")
        assert result["changed"] is True
        assert host.scene_requests == []

    def test_a_failed_analysis_cuts_the_middle(self, host: FakeHost) -> None:
        # 解析に失敗した素材で待ち続けると、壊れた素材のある作品をいつまでも作り替えられない
        host.scene_states[_media(host).id] = SceneState(None, running=False, failure="読めない")
        result = _call(host)
        assert result["changed"] is True
        assert result["untracked_media"] == [str(_media(host).id)]

    def test_the_bridge_reframes_after_the_wait(self, host: FakeHost) -> None:
        # 待った後に作り替えないと、AI は同じ道具を呼び直し続ける
        media = _media(host)
        host.scene_states[media.id] = SceneState(None, running=True)

        def finished(_media: object) -> bool:
            host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
            return True

        host.scene_ready = finished
        # 確かめの窓は別の試験（test_bridge.py）が見る ここは待った後の続きだけを見る
        bridge = allow_sending(EditorBridge(host, auto_approve=True))
        began = time.monotonic()
        done, box = call_in_thread(bridge, REFRAME, {"wait_seconds": 5})
        pump_until(bridge, done)
        assert time.monotonic() - began < 5.0
        result = box["result"]
        assert isinstance(result, dict), result
        assert result["changed"] is True
        assert host.project.settings.resolution == (1080, 1920)


class TestTheResult:
    def test_one_undo_brings_the_wide_work_back(self, host: FakeHost) -> None:
        # 何段にも分かれると、人が Ctrl+Z を何度も押さないと横の作品へ戻れない
        before = host.project
        host.scene_states[_media(host).id] = SceneState(_signal(0.8), running=False)
        _call(host)
        label = host.document.undo_label
        assert label is not None and label.startswith("縦の作品に作り替え")
        host.undo()
        assert host.project == before

    def test_the_settings_give_the_defaults(self, host: FakeHost) -> None:
        # 本人の設定を読まないと、4:5 を選んだ人の作品が毎回 9:16 になる
        host.reframing = ReframeStyle(size="1080x1350", follow="steady", smooth_ms=2000)
        result = _call(host, follow="center")
        assert result["resolution"] == "1080x1350"
        assert result["smooth_ms"] == 2000
        assert host.project.settings.resolution == (1080, 1350)

    def test_the_arguments_win_over_the_settings(self, host: FakeHost) -> None:
        # 引数を無視すると、AI が人の頼んだ大きさで作れない
        result = _call(host, width=720, height=1280, follow="center")
        assert result["resolution"] == "720x1280"

    def test_the_shots_are_reported(self, host: FakeHost) -> None:
        # 場面と枠の位置が返らないと、AI がどこを preview_range で確かめればよいか分からない
        host.scene_states[_media(host).id] = SceneState(_signal(0.8), running=False)
        result = _call(host)
        assert result["shot_count"] == 1
        shot = result["shots"][0]
        assert shot["mode"] == "steady"
        assert shot["focus_x"] > 300
        assert "save_project_as" in result["note"]


def _faces(across: float) -> ScreenReport:
    """横の ``across``（左から 0〜1）に顔のある、調べ終えた顔の位置（1920x1080 の素材）"""
    face = Face(Box((across - 0.5) * 1920, 0.0, 120.0, 120.0), 0.9)
    samples = tuple(ScreenSample(moment, faces=(face,)) for moment in (2.0, 5.0, 8.0))
    return ScreenReport(ScreenKind.FACES, "偽の顔", 1920, 1080, samples, 3, True, stream=0)


class TestTheFaces:
    """顔の位置（#407）を被写体の手がかりに足す 部品は偽のホストの ``vision_ready`` で決める"""

    def test_without_the_parts_it_says_so_and_does_not_ask(self, host: FakeHost) -> None:
        # 部品の無い機械で顔を頼むと、素材ごとに「部品が無い」失敗を積む 黙ると、人は顔を
        # 見ていると思い込む
        host.scene_states[_media(host).id] = SceneState(_signal(0.8), running=False)
        result = _call(host)
        assert result["changed"] is True
        assert result["faces"] == "no_parts"
        assert host.screen_requests == []
        assert "部品" in result["note"]

    def test_the_faces_move_the_frame(self, host: FakeHost) -> None:
        # 壊れると、顔の位置を調べ終えていても動きの側（右）へ枠が決まり、左の話し手が映らない
        media = _media(host)
        host.vision_ready = True
        host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
        host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(_faces(0.2), running=False)
        result = _call(host)
        assert host.screen_requests == [(media.id, ScreenKind.FACES)]
        assert result["faces"] == "used"
        assert result["face_shots"] == 1
        shot = result["shots"][0]
        assert shot["faces"] is True
        assert shot["focus_x"] < -300

    def test_the_argument_turns_them_off(self, host: FakeHost) -> None:
        # 切れないと、手元や品物を映したい場面で枠が顔へ取られたままになる
        media = _media(host)
        host.vision_ready = True
        host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
        host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(_faces(0.2), running=False)
        result = _call(host, faces=False)
        assert result["faces"] == "off"
        assert result["shots"][0]["focus_x"] > 300
        assert host.screen_requests == []

    def test_the_setting_turns_them_off(self, host: FakeHost) -> None:
        # 設定で切っても効いたままなら、設定がある方が質が悪い
        media = _media(host)
        host.vision_ready = True
        host.reframing = ReframeStyle(faces=False)
        host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
        host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(_faces(0.2), running=False)
        assert _call(host)["faces"] == "off"

    def test_faces_still_running_do_not_stop_the_work(self, host: FakeHost) -> None:
        # 顔を待って断ると、顔を調べ終えるまで作り替えられない（動きと肌色で作れるのに）
        media = _media(host)
        host.vision_ready = True
        host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
        host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(None, running=True)
        result = _call(host, wait_seconds=0)
        assert result["changed"] is True
        assert result["faces"] == "waiting"
        assert result["shots"][0]["faces"] is False

    def test_running_faces_are_waited_for(self, host: FakeHost) -> None:
        # 待たずに作ると、場面の解析が終わってすぐの呼び出しで、いつも顔を使わずに作る
        media = _media(host)
        host.vision_ready = True
        host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
        host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(None, running=True)
        pending = REFRAME(host, {})
        assert isinstance(pending, Pending)
        host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(_faces(0.2), running=False)
        assert pending.wait(0.1)
        result = pending.then()
        assert isinstance(result, dict)
        assert result["faces"] == "used"

    def test_a_failure_is_told(self, host: FakeHost) -> None:
        # 調べられなかったことを伏せると、人は顔を使ったと思い込む 訳は detect_faces で読む
        media = _media(host)
        host.vision_ready = True
        host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
        host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(
            None, running=False, failure="素材が開けない"
        )
        result = _call(host)
        assert result["faces"] == "none"
        assert "調べられなかった素材 1 本" in result["note"]
        assert "detect_faces" in result["note"]

    def test_center_says_nothing_was_estimated(self, host: FakeHost) -> None:
        # 真ん中の切り出しを「動きと肌色だけで見積もった」と返すと、AI が人へ被写体を追ったと
        # 誤って伝える（PR #408 の CodeRabbit の指摘）
        host.vision_ready = True
        result = _call(host, follow="center")
        assert result["faces"] == "center"
        assert "見積もっていません" in result["note"]
        assert "動きと肌色だけで見積もりました" not in result["note"]

    def test_no_video_to_follow_is_told_apart(self, host: FakeHost) -> None:
        # 追う背景の動画が無い作品で「動きと肌色だけで見積もった」と返すと、何も追っていないのに
        # 追ったと AI が人へ伝える
        host.vision_ready = True
        clips = [clip for track in host.project.timeline.tracks for clip in track.clips]
        host.apply_commands([RemoveClip(clip.id) for clip in clips], "消す")
        result = _call(host)
        assert result["faces"] == "no_video"
        assert "動きと肌色だけで見積もりました" not in result["note"]

    def test_turning_them_off_is_told_as_a_choice(self, host: FakeHost) -> None:
        # 切ったのに「部品が無い」や「顔が無い」と返すと、人は入れ直したり素材を疑ったりする
        media = _media(host)
        host.vision_ready = True
        host.reframing = ReframeStyle(faces=False)
        host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
        result = _call(host)
        assert result["faces"] == "off"
        assert "使わない選び" in result["note"]
        assert "動きと肌色だけで見積もりました" in result["note"]

    def test_a_failure_reason_is_not_sent(self, host: FakeHost) -> None:
        # 失敗の文には素材の場所が入ることがある reframe は作品と設定だけを送ると宣言しているので、
        # 名前と場所を送らない人の所でも場所が外へ出る
        media = _media(host)
        host.vision_ready = True
        host.scene_states[media.id] = SceneState(_signal(0.8), running=False)
        host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(
            None, running=False, failure="素材を開けない: C:/素材/秘密.mp4"
        )
        result = _call(host)
        assert "秘密" not in result["note"]
        assert "調べられなかった素材 1 本" in result["note"]

    def test_a_bad_argument_is_refused(self, host: FakeHost) -> None:
        # 文字列の "false" を真と読むと、切ったつもりの顔で枠が決まる
        with pytest.raises(ToolError, match="faces"):
            REFRAME(host, {"follow": "center", "faces": "false"})


class TestTheRefusals:
    def test_a_scene_opened_while_waiting_is_refused(self, host: FakeHost) -> None:
        # 待つ間に人がシーンを開いたまま作り替えると、シーンのタイムラインをメインと取り違え、
        # 命令も開いたシーンへ包まれて、メインが横のまま残るか命令ごと断られる
        host.scene_states[_media(host).id] = SceneState(None, running=True)
        pending = REFRAME(host, {})
        assert isinstance(pending, Pending)
        before = host.document.project
        scene = Scene("中", Timeline(rate=RATE_30))
        host.apply_commands([AddScene(scene)], "シーン")
        host.set_active_scene(scene.id)
        host.scene_states[_media(host).id] = SceneState(_signal(0.8), running=False)
        with pytest.raises(ToolError, match="シーンが開かれた"):
            pending.then()
        assert host.document.project.settings.resolution == before.settings.resolution

    def test_an_open_scene_is_refused(self, host: FakeHost) -> None:
        # 開いたシーンだけを作り替えると、メインが縦の画面に横の配置のまま残る
        scene = Scene("中", Timeline(rate=RATE_30))
        host.apply_commands([AddScene(scene)], "シーン")
        host.set_active_scene(scene.id)
        with pytest.raises(ToolError, match="メインに"):
            REFRAME(host, {"follow": "center"})

    def test_the_same_size_is_refused(self, host: FakeHost) -> None:
        # 何も変わらない段を積むと、作り替えたつもりで横のまま書き出す
        with pytest.raises(ToolError, match="もう"):
            REFRAME(host, {"width": 1920, "height": 1080, "follow": "center"})

    @pytest.mark.parametrize(
        "arguments",
        [
            {"width": 1081, "height": 1920},
            {"width": 1080},
            {"follow": "zoom"},
            {"smooth_ms": 50},
            {"smooth_ms": "1000"},
        ],
    )
    def test_bad_arguments_are_refused(self, host: FakeHost, arguments: dict[str, Any]) -> None:
        # 黙って端へ寄せたり既定へ倒すと、AI は頼んだ形で作ったと思い込む
        with pytest.raises(ToolError):
            REFRAME(host, {"follow": "center", **arguments})
        assert host.project.settings.resolution == (1920, 1080)
