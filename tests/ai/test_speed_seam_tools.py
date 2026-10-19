"""AI の再生速度とつなぎ目の道具（#332 B6） set_speed・smooth_cuts・jet_cut の smooth

偽のホストで、道具が人の操作と同じ命令を通すこと、取り消し 1 回で戻ること、本人の設定を
既定に使うこと、断る理由を AI に返すことを見る 命令そのものの決まりは tests/core/test_seams.py と
tests/core/test_speed.py、設定画面と本物の窓は tests/ui/test_seam_preferences.py
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import (
    AddClip,
    AddScene,
    InScene,
    SetClipProperty,
    SetTrackState,
    new_scene,
)
from sashimono.core.model import (
    TRANSITION_KIND,
    Clip,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.core.seams import SEAM_ZOOM, UNFIT_SEAM, SeamStyle
from tests.ai.conftest import FakeHost
from tests.core.test_seams import RATE, JumpCut, talking_media, with_bystander


def _run(host: FakeHost, name: str, **arguments: Any) -> Any:
    operation = find_operation(name)
    assert operation is not None
    return operation(host, arguments)


def _transitions(project: Project) -> list[tuple[int, int]]:
    return [
        (clip.timeline_start, clip.duration)
        for track in project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == TRANSITION_KIND
    ]


@pytest.mark.parametrize(
    ("name", "sends"),
    [
        ("set_speed", {Outbound.PROJECT}),
        ("smooth_cuts", {Outbound.PROJECT, Outbound.APP_SETTINGS}),
        ("jet_cut", {Outbound.PROJECT, Outbound.FILES, Outbound.APP_SETTINGS}),
    ],
)
def test_the_tools_are_undoable_edits(name: str, sends: set[Outbound]) -> None:
    # 壊れると、取り消せる編集なのに毎回確かめが出るか、本人の設定の既定を結果に載せるのに
    # 送る物に書かず、設定画面と個人情報の扱いの一覧に無い物が外へ出る
    operation = find_operation(name)
    assert operation is not None
    assert operation.safety is Safety.UNDOABLE
    assert operation.sends == sends


class TestSetSpeed:
    def test_double_speed_shortens_both_and_undoes_once(self) -> None:
        # 壊れると、AI に「2 倍速」と頼んでも長さが変わらないか、音だけ元の速さで残るか、
        # 戻すのに取り消しが何回も要る
        work = JumpCut()
        host = FakeHost(work.project)
        result = _run(host, "set_speed", clip_ids=[str(work.pictures[2].id)], speed_percent=200)
        assert result["speed_percent"] == 200.0
        rows = {row["clip_id"]: row for row in result["clips"]}
        for clip in (work.pictures[2], work.sounds[2]):
            assert rows[str(clip.id)]["duration_before"] == 60
            assert rows[str(clip.id)]["duration"] == 30
        assert "warnings" not in result
        host.document.undo()
        assert host.document.project == work.project

    def test_the_selection_is_used_when_no_clip_is_named(self) -> None:
        # 壊れると、「選んでいるクリップをゆっくりに」と頼んでも ID を聞き返される
        work = JumpCut()
        host = FakeHost(work.project)
        host.selection = (work.pictures[2].id,)
        result = _run(host, "set_speed", speed_percent=50, keep="length")
        assert {row["duration"] for row in result["clips"]} == {60}

    def test_running_past_the_end_of_the_media_is_warned(self) -> None:
        # 壊れると、長さを保ったまま速くして素材の終わりを越え、終わりが黒いまま書き出される
        work = JumpCut(durations=(60, 60, 90))
        host = FakeHost(work.project)
        # 3 本目は素材の 10 秒から 3 秒 1000% で 30 秒流すと 30 秒の素材を越える
        result = _run(
            host,
            "set_speed",
            clip_ids=[str(work.pictures[2].id)],
            speed_percent=1000,
            keep="length",
        )
        assert any("素材の終わり" in warning for warning in result["warnings"])

    def test_a_held_picture_is_not_called_black(self) -> None:
        # 壊れると、絵を止めた（hold_at）クリップでも「何も映らない」と伝わり、AI が直す必要の
        # 無い trim_clip を勧める 音は止まらないので、音のクリップには無音になると伝える
        # （CodeRabbit の指摘）
        work = JumpCut(durations=(60, 60, 90))
        project = SetClipProperty(work.pictures[2].id, "hold_at", Fraction(29)).apply(work.project)
        host = FakeHost(project)
        result = _run(
            host,
            "set_speed",
            clip_ids=[str(work.pictures[2].id)],
            speed_percent=1000,
            keep="length",
        )
        warnings = result["warnings"]
        assert not any(str(work.pictures[2].id) in warning for warning in warnings)
        sound = [w for w in warnings if str(work.sounds[2].id) in w]
        assert len(sound) == 1
        assert "無音" in sound[0]
        assert "何も映らない" not in sound[0]

    @pytest.mark.parametrize(
        ("hold_at", "lost"),
        [(None, "越えた所は映らず、音は無音になる"), (Fraction(29), "越えた所は無音になる")],
    )
    def test_a_layer_clip_warns_for_what_it_loses(
        self, hold_at: Fraction | None, lost: str
    ) -> None:
        # 壊れると、絵と音を 1 本で出すレイヤーのクリップで、音のことしか伝わらず、終わりが黒く
        # なるのに AI が気付かない 絵を止めていれば音のことだけ伝える（CodeRabbit の指摘）
        media = talking_media()
        base = Project.create(ProjectSettings(frame_rate=RATE), media=(media,))
        layer = Track(TrackKind.MIXED, "L1")
        clip = Clip(
            timeline_start=0,
            duration=90,
            media_id=media.id,
            source_in=Fraction(10),
            audio_stream=1,
            hold_at=hold_at,
        )
        project = AddClip(layer.id, clip).apply(
            base.with_timeline(replace(base.timeline, tracks=(layer,)))
        )
        assert project.plays_sound(layer, clip) and project.draws_picture(layer, clip)
        host = FakeHost(project)
        result = _run(host, "set_speed", clip_ids=[str(clip.id)], speed_percent=1000, keep="length")
        (warning,) = result["warnings"]
        assert lost in warning

    def test_running_past_the_end_of_a_scene_is_warned(self) -> None:
        # 壊れると、入れ子のシーンを長さを保ったまま速くしてシーンの終わりを越えても警告が出ず、
        # 終わりが黒く無音のまま書き出される（描画と音のミキサは越えた時間を空として扱う）
        # （Codex の指摘）
        base = Project.create(ProjectSettings(frame_rate=RATE), media=(talking_media(),))
        inner = new_scene(base, "場面")
        media = base.media[0]
        inner_track = Track(TrackKind.VIDEO, "V1")
        inner = replace(inner, timeline=replace(inner.timeline, tracks=(inner_track,)))
        project = AddScene(inner).apply(base)
        project = InScene(
            inner.id,
            AddClip(inner_track.id, Clip(timeline_start=0, duration=60, media_id=media.id)),
        ).apply(project)
        outer = Track(TrackKind.VIDEO, "V1")
        project = project.with_timeline(replace(project.timeline, tracks=(outer,)))
        placed = Clip(timeline_start=0, duration=60, scene_id=inner.id)
        project = AddClip(outer.id, placed).apply(project)
        host = FakeHost(project)
        result = _run(
            host, "set_speed", clip_ids=[str(placed.id)], speed_percent=200, keep="length"
        )
        (warning,) = result["warnings"]
        assert "シーンの終わりを 2.00 秒越えます" in warning

    def test_a_collision_is_explained(self) -> None:
        # 壊れると、ぶつかって断ったことと伸ばせる量が AI に届かず、何度も同じ頼みを繰り返す
        work = JumpCut()
        host = FakeHost(work.project)
        with pytest.raises(ToolError, match="あと 0 フレーム"):
            _run(host, "set_speed", clip_ids=[str(work.pictures[0].id)], speed_percent=50)
        assert host.document.project == work.project

    @pytest.mark.parametrize(
        "arguments",
        [{}, {"speed_percent": 0}, {"speed_percent": 2000}, {"speed_percent": 50, "keep": "x"}],
    )
    def test_bad_arguments_are_refused(self, arguments: dict[str, Any]) -> None:
        # 壊れると、外れた速度を黙って端へ寄せ、頼んだのと違う速さになる
        work = JumpCut()
        host = FakeHost(work.project)
        with pytest.raises(ToolError):
            _run(host, "set_speed", clip_ids=[str(work.pictures[2].id)], **arguments)


class TestSmoothCuts:
    def test_found_seams_get_the_preferred_treatment_in_one_undo(self) -> None:
        # 壊れると、AI がつなぎ目を探せないか、本人が設定で選んだ処理が効かないか、
        # 戻すのに取り消しが何回も要る
        work = JumpCut()
        host = FakeHost(work.project)
        host.seams = SeamStyle(video=SEAM_ZOOM, zoom_percent=120, audio_fade_ms=30)
        result = _run(host, "smooth_cuts")
        assert result["seams"] == 2
        assert result["video"] == SEAM_ZOOM
        assert result["zoom_percent"] == 120
        assert result["zoomed_clip_ids"] == [str(work.pictures[1].id)]
        assert result["audio_faded"] == 2
        assert result["audio_fade_ms"] == 30
        assert len(host.document.history_labels) == 1
        host.document.undo()
        assert host.document.project == work.project

    def test_arguments_override_the_preference(self) -> None:
        # 壊れると、AI が頼みに合わせて長さを変えても、本人の設定の値のまま付く
        work = JumpCut()
        host = FakeHost(work.project)
        result = _run(host, "smooth_cuts", frames=[60], crossfade_ms=200, audio_fade_ms=0)
        assert result["crossfades"] == [{"frame": 60, "frames": 6}]
        assert result["audio_faded"] == 0
        assert _transitions(host.document.project) == [(59, 7)]

    def test_found_seams_leave_a_plain_split_on_another_track_alone(self) -> None:
        # 壊れると、探したつなぎ目と同じフレームで区切れている別のトラックの関係の無いクリップ
        # まで寄せ、そこの 1 フレームのクリップのせいでクロスフェードごと断られる（Codex の指摘）
        work = JumpCut()
        project, bystander, _before, after = with_bystander(work)
        host = FakeHost(project)
        result = _run(host, "smooth_cuts", video="zoom", audio_fade_ms=0)
        assert result["zoomed_clip_ids"] == [str(work.pictures[1].id)]
        located = host.document.project.timeline.locate_clip(after.id)
        assert located is not None and located[1] == after
        host.document.undo()
        # トラックを名指ししたら、そのトラックの所だけ（名指しの位置でも別のトラックは触らない）
        result = _run(
            host,
            "smooth_cuts",
            frames=[60],
            track_id=str(bystander.id),
            video="zoom",
            audio_fade_ms=0,
        )
        assert result["zoomed_clip_ids"] == [str(after.id)]
        located = host.document.project.timeline.locate_clip(work.pictures[1].id)
        assert located is not None and located[1] == work.pictures[1]

    def test_a_refusal_names_the_reason(self) -> None:
        # 壊れると、寄りを頼んだのに音だけのつなぎ目しか無いとき、「前後のクリップが接している所が
        # ありません」と事実と違う理由で断り、AI が見当違いの位置を探し直す（CodeRabbit の指摘から）
        work = JumpCut()
        project = SetTrackState(work.video.id, locked=True).apply(work.project)
        host = FakeHost(project)
        with pytest.raises(ToolError, match=UNFIT_SEAM):
            _run(host, "smooth_cuts", video="zoom", audio_fade_ms=0, frames=[60])

    def test_a_place_that_is_not_a_seam_is_reported(self) -> None:
        # 壊れると、つなぎ目でない所を渡したことに AI が気付かず、付けたと思い込む
        work = JumpCut()
        host = FakeHost(work.project)
        result = _run(host, "smooth_cuts", frames=[30, 60])
        assert result["not_seams"] == [30]
        assert result["handled"] == 1

    def test_nothing_to_smooth_is_refused(self) -> None:
        # 壊れると、つなぎ目の無い作品で「付けた」と答え、空の 1 段が履歴に積まれる
        work = JumpCut()
        host = FakeHost(work.project)
        with pytest.raises(ToolError, match="見つかりません"):
            _run(host, "smooth_cuts", start_frame=130)
        with pytest.raises(ToolError, match="付けられませんでした"):
            _run(host, "smooth_cuts", frames=[30])
        assert host.document.history_labels == ()

    @pytest.mark.parametrize(
        "arguments",
        [
            {"video": "wipe"},
            {"crossfade_ms": 5},
            {"zoom_percent": 300},
            {"audio_fade_ms": -1},
            {"video": "none", "audio_fade_ms": 0},
            {"frames": []},
            {"frames": [0]},
            {"track_id": "無い"},
            # 範囲は整数のフレームだけ 小数を黙って切り捨てない（CodeRabbit の指摘）
            {"start_frame": 59.9},
            {"end_frame": 120.5},
            {"start_frame": -1},
            {"start_frame": 120, "end_frame": 60},
            # 長さと拡大率も整数だけ 丸めて受けると頼んだのと違う長さで付く
            {"crossfade_ms": 100.5},
            {"zoom_percent": 110.4},
            {"audio_fade_ms": 20.7},
        ],
    )
    def test_bad_arguments_are_refused(self, arguments: dict[str, Any]) -> None:
        # 壊れると、外れた値を黙って端へ寄せるか、何もしない頼みで何かを変える
        work = JumpCut()
        host = FakeHost(work.project)
        with pytest.raises(ToolError):
            _run(host, "smooth_cuts", **arguments)
        assert host.document.project == work.project


class TestJetCutSmooth:
    def _host(self) -> FakeHost:
        from tests.ui.test_subtitle_panel import make_waveform

        work = JumpCut(durations=(900, 1, 1))
        host = FakeHost(work.project)
        # 1.6 秒鳴って 1.1 秒黙り、また鳴る 真ん中の無音だけが切れる
        host.stub_waveform = make_waveform([(0.5, 300), (0.0, 200), (0.5, 5500)])
        return host

    def test_cut_and_smoothing_undo_together(self) -> None:
        # 壊れると、詰めたつなぎ目の処理を戻すのに取り消しが 2 回要り、1 回戻すと処理だけ残る
        host = self._host()
        before = host.document.project
        media = before.media[0]
        result = _run(host, "jet_cut", media_id=str(media.id), keep_speech=False, smooth=True)
        assert result["cuts"] == 1
        smoothing = result["smoothing"]
        assert smoothing["video"] == "crossfade"
        assert len(smoothing["crossfades"]) == 1
        # 詰めて割ったクリップの ID は当てるたびに変わるので、寄せた ID は返さない
        assert "zoomed_clip_ids" not in smoothing
        assert len(_transitions(host.document.project)) == 1
        assert len(host.document.history_labels) == 1
        host.document.undo()
        assert host.document.project == before

    def test_without_smooth_nothing_is_added(self) -> None:
        # 壊れると、今までどおり詰めるだけのつもりの頼みで、つなぎ目に切り替えが勝手に付く
        host = self._host()
        media = host.document.project.media[0]
        result = _run(host, "jet_cut", media_id=str(media.id), keep_speech=False)
        assert "smoothing" not in result
        assert _transitions(host.document.project) == []
