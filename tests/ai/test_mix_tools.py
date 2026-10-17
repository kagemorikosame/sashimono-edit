"""AI の音量とミックスの道具（#329 B3） set_volume・duck_audio・normalize_loudness・place_sound

偽のホストで、道具が人の操作と同じ命令（音量の欄・音量の線・置く命令）を通すこと、
取り消し 1 回で戻ること、本人の設定を既定に使うことを見る 本物の窓と本物の検査で
測る試験は tests/ui/test_mix_tools.py
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import Pending, find_operation
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddClip, SetClipProperty, SetTrackState
from sashimono.core.commands.fixed import FADE_EFFECT_KIND, VOLUME_EFFECT_KIND, fixed_effect
from sashimono.core.commands.volume import clip_volume, fixed_volume
from sashimono.core.mixing import DuckShape, db_to_ratio
from sashimono.core.model import (
    AnimatedValue,
    AudioStreamInfo,
    Clip,
    ClipId,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.engine.checks import CheckState
from sashimono.engine.checks.report import assemble
from sashimono.engine.checks.scan import LOUDNESS
from tests.ai.conftest import RATE_30, FakeHost


def _run(host: FakeHost, name: str, **arguments: Any) -> Any:
    operation = find_operation(name)
    assert operation is not None
    return operation(host, arguments)


def _sound(name: str, seconds: int) -> MediaItem:
    return MediaItem(
        path=Path(f"C:/素材/{name}.wav"),
        duration=Fraction(seconds),
        audio_streams=(
            AudioStreamInfo(
                index=0, sample_rate=48000, channels=2, time_base=Fraction(1, 48000), codec="pcm"
            ),
        ),
    )


def _sound_clip(media: MediaItem, start: int, duration: int) -> Clip:
    # 置いたときと同じく音量とフェードの欄を持たせる
    return Clip(
        timeline_start=start,
        duration=duration,
        media_id=media.id,
        effects=(fixed_effect(VOLUME_EFFECT_KIND), fixed_effect(FADE_EFFECT_KIND)),
    )


class _Work:
    """声（字幕付き 10 秒）と BGM（30 秒）を別の音声トラックに置いた作品

    映像トラックにも絵を 1 本置く 音を鳴らさないクリップを断るかを見るため
    """

    def __init__(self, transcript: Transcript | None, video_media: MediaItem) -> None:
        voice_media = _sound("声", 10)
        if transcript is not None:
            voice_media = voice_media.with_transcript(transcript)
        self.bgm_media = _sound("曲", 30)
        self.video_media = video_media
        base = Project.create(
            ProjectSettings(frame_rate=RATE_30),
            media=(voice_media, self.bgm_media, video_media),
        )
        self.picture_track = Track(kind=TrackKind.VIDEO, name="V1")
        self.voice_track = Track(kind=TrackKind.AUDIO, name="声")
        self.bgm_track = Track(kind=TrackKind.AUDIO, name="BGM")
        project = base.with_timeline(
            replace(base.timeline, tracks=(self.picture_track, self.voice_track, self.bgm_track))
        )
        self.picture = Clip(timeline_start=0, duration=300, media_id=video_media.id)
        self.voice = _sound_clip(voice_media, 0, 300)
        self.bgm = _sound_clip(self.bgm_media, 0, 600)
        project = AddClip(self.picture_track.id, self.picture).apply(project)
        project = AddClip(self.voice_track.id, self.voice).apply(project)
        self.project = AddClip(self.bgm_track.id, self.bgm).apply(project)
        self.host = FakeHost(self.project)

    def clip(self, clip_id: ClipId) -> Clip:
        located = self.host.document.project.timeline.locate_clip(clip_id)
        assert located is not None
        return located[1]


@pytest.fixture
def work(transcript: Transcript, video_media: MediaItem) -> _Work:
    return _Work(transcript, video_media)


@pytest.fixture
def silent_work(video_media: MediaItem) -> _Work:
    """声に字幕が無い作品 話している所は波形から拾う"""
    return _Work(None, video_media)


class TestKinds:
    @pytest.mark.parametrize(
        "name", ["set_volume", "duck_audio", "normalize_loudness", "place_sound"]
    )
    def test_they_are_undoable_edits(self, name: str) -> None:
        # 壊れると、取り消せる編集に毎回確かめの窓が出るか、変えたのに確かめが挟まらない
        operation = find_operation(name)
        assert operation is not None
        assert operation.safety is Safety.UNDOABLE


class TestSetVolume:
    def test_a_percent_sets_every_clip_and_one_undo_restores(self, work: _Work) -> None:
        # 壊れると、何本か選んで頼んだ音量が一部にしか入らないか、戻すのに何回も取り消す
        result = _run(
            work.host,
            "set_volume",
            clip_ids=[str(work.voice.id), str(work.bgm.id)],
            volume_percent=50,
        )
        assert [row["volume_percent"] for row in result["clips"]] == [50.0, 50.0]
        assert clip_volume(work.clip(work.voice.id)).static == 50.0
        assert clip_volume(work.clip(work.bgm.id)).static == 50.0
        work.host.document.undo()
        assert work.host.document.project == work.project

    def test_a_percent_uses_the_same_field_as_the_panel(self, work: _Work) -> None:
        # 壊れると、設定パネルの「音量」とは別の音量調整が積まれ、欄を見ても変わっていない
        _run(work.host, "set_volume", clip_ids=[str(work.bgm.id)], volume_percent=70)
        clip = work.clip(work.bgm.id)
        assert [e.kind for e in clip.effects] == [VOLUME_EFFECT_KIND, FADE_EFFECT_KIND]
        volume = fixed_volume(clip)
        assert volume is not None and volume.id == work.bgm.effects[0].id

    def test_gain_keeps_the_shape_of_the_line(self, work: _Work) -> None:
        # 壊れると、ダッキングで下げた BGM を全体で上げたときに下げた所が消える
        line = AnimatedValue(
            static=100.0, keyframes=(Keyframe(frame=0, value=100.0), Keyframe(frame=30, value=25.0))
        )
        effect = work.bgm.effects[0].with_param("volume", line)
        project = work.project
        from sashimono.core.commands import RemoveClip

        project = RemoveClip(work.bgm.id).apply(project)
        bgm = replace(work.bgm, effects=(effect, work.bgm.effects[1]))
        project = AddClip(work.bgm_track.id, bgm).apply(project)
        host = FakeHost(project)
        _run(host, "set_volume", clip_ids=[str(bgm.id)], gain_db=6)
        located = host.document.project.timeline.locate_clip(bgm.id)
        assert located is not None
        values = [k.value for k in clip_volume(located[1]).keyframes]
        assert values == pytest.approx([100.0 * db_to_ratio(6), 25.0 * db_to_ratio(6)])

    def test_gain_says_when_it_stops_at_the_limit(self, work: _Work) -> None:
        # 壊れると、上限で止まったのに AI は頼んだだけ上がったと思い込む
        result = _run(work.host, "set_volume", clip_ids=[str(work.bgm.id)], gain_db=20)
        assert clip_volume(work.clip(work.bgm.id)).static == 400.0
        assert "上限" in result["note"]

    def test_the_selection_is_used_when_no_clips_are_named(self, work: _Work) -> None:
        # 壊れると、「選んでいる BGM を小さく」が通らず ID を聞き返す
        work.host.select_clips([work.bgm.id])
        _run(work.host, "set_volume", volume_percent=30)
        assert clip_volume(work.clip(work.bgm.id)).static == 30.0
        assert clip_volume(work.clip(work.voice.id)).static == 100.0

    def test_pan_alone_keeps_the_volume(self, work: _Work) -> None:
        # 壊れると、左右だけを変えたつもりで音量まで 100% に戻る
        _run(work.host, "set_volume", clip_ids=[str(work.bgm.id)], volume_percent=40)
        _run(work.host, "set_volume", clip_ids=[str(work.bgm.id)], pan_percent=-50)
        volume = fixed_volume(work.clip(work.bgm.id))
        assert volume is not None
        assert volume.params["volume"] == AnimatedValue(static=40.0)
        assert volume.params["pan"] == AnimatedValue(static=-50.0)

    def test_an_old_clip_without_the_field_gets_one(self, work: _Work) -> None:
        # 壊れると、前の版で作った（音量の欄の無い）クリップの音量を AI が変えられない
        from sashimono.core.commands import RemoveClip

        bare = replace(work.bgm, effects=())
        project = AddClip(work.bgm_track.id, bare).apply(
            RemoveClip(work.bgm.id).apply(work.project)
        )
        host = FakeHost(project)
        _run(host, "set_volume", clip_ids=[str(bare.id)], volume_percent=20)
        located = host.document.project.timeline.locate_clip(bare.id)
        assert located is not None
        assert clip_volume(located[1]).static == 20.0
        host.document.undo()
        assert host.document.project == project

    def test_a_picture_clip_is_refused(self, work: _Work) -> None:
        # 壊れると、絵のクリップに音量を入れて何も鳴らないのに、AI は下げたと伝える
        with pytest.raises(ToolError, match="音を鳴らさない"):
            _run(work.host, "set_volume", clip_ids=[str(work.picture.id)], volume_percent=10)
        assert work.host.document.project == work.project

    def test_a_locked_track_is_refused(self, work: _Work) -> None:
        # 壊れると、人が触らせないためにロックしたトラックの音量を AI が変える
        work.host.apply_commands([SetTrackState(work.bgm_track.id, locked=True)], "ロック")
        before = work.host.document.project
        with pytest.raises(ToolError, match="ロック"):
            _run(work.host, "set_volume", clip_ids=[str(work.bgm.id)], volume_percent=10)
        assert work.host.document.project == before

    def test_percent_and_gain_together_are_refused(self, work: _Work) -> None:
        # 壊れると、どちらが効いたのか AI にも人にも分からない
        with pytest.raises(ToolError, match="どちらか"):
            _run(work.host, "set_volume", clip_ids=[str(work.bgm.id)], volume_percent=10, gain_db=3)


def _keys(work: _Work) -> list[tuple[int, float]]:
    return [(k.frame, round(k.value, 2)) for k in clip_volume(work.clip(work.bgm.id)).keyframes]


class TestDuck:
    def test_the_bgm_dips_while_the_subtitles_speak(self, work: _Work) -> None:
        # 壊れると、話している間も BGM が同じ大きさで鳴り、声が聞き取りにくい
        # 字幕は 1〜3・4〜6・7〜9 秒 間の 1 秒（30 フレーム）は既定の下げる速さと戻る速さの和
        # （9 + 24 フレーム）より短いので、1 つの谷になる
        result = _run(
            work.host,
            "duck_audio",
            clip_ids=[str(work.bgm.id)],
            voice_clip_ids=[str(work.voice.id)],
        )
        ducked = round(100.0 * db_to_ratio(-12), 2)
        assert _keys(work) == [(21, 100.0), (30, ducked), (270, ducked), (294, 100.0)]
        assert result["depth_db"] == 12
        assert result["speech_spans"] == 3

    def test_the_preference_is_the_default(self, work: _Work) -> None:
        # 壊れると、設定画面で選んだ下げ幅と速さが効かず、いつも同じ下げ方になる
        work.host.duck = DuckShape(depth_db=6, attack_ms=100, release_ms=200)
        _run(
            work.host,
            "duck_audio",
            track_id=str(work.bgm_track.id),
            voice_track_id=str(work.voice_track.id),
        )
        ducked = round(100.0 * db_to_ratio(-6), 2)
        # 3 フレームで下げ 6 フレームで戻る 間の 30 フレームでは戻るので谷が 3 つ
        assert _keys(work) == [
            (27, 100.0),
            (30, ducked),
            (90, ducked),
            (96, 100.0),
            (117, 100.0),
            (120, ducked),
            (180, ducked),
            (186, 100.0),
            (207, 100.0),
            (210, ducked),
            (270, ducked),
            (276, 100.0),
        ]

    def test_arguments_override_the_preference(self, work: _Work) -> None:
        # 壊れると、AI に「もっと下げて」と頼んでも設定の値のまま
        _run(
            work.host,
            "duck_audio",
            clip_ids=[str(work.bgm.id)],
            voice_clip_ids=[str(work.voice.id)],
            depth_db=20,
        )
        assert min(value for _, value in _keys(work)) == pytest.approx(10.0)

    def test_ducking_again_keeps_the_level(self, work: _Work) -> None:
        # 壊れると、掛け直すたびに下げた所を基準にして、BGM がどんどん小さくなる
        arguments = {"clip_ids": [str(work.bgm.id)], "voice_clip_ids": [str(work.voice.id)]}
        _run(work.host, "duck_audio", **arguments)
        first = _keys(work)
        _run(work.host, "duck_audio", **arguments)
        assert _keys(work) == first

    def test_one_undo_restores(self, work: _Work) -> None:
        # 壊れると、ダッキングを戻すのに点の数だけ取り消す
        _run(
            work.host,
            "duck_audio",
            clip_ids=[str(work.bgm.id)],
            voice_clip_ids=[str(work.voice.id)],
        )
        work.host.document.undo()
        assert work.host.document.project == work.project

    def test_the_waveform_is_used_without_subtitles(self, silent_work: _Work) -> None:
        # 壊れると、字幕を起こしていない作品では BGM を下げられない
        from tests.ui.test_subtitle_panel import make_waveform

        # 1 ピークは 256 / 48000 秒 375 ピークで 2 秒 頭 2 秒は無音、続く 2 秒が声、残りは無音
        silent_work.host.stub_waveform = make_waveform([(0.0, 375), (0.5, 375), (0.0, 1125)])
        _run(
            silent_work.host,
            "duck_audio",
            clip_ids=[str(silent_work.bgm.id)],
            voice_clip_ids=[str(silent_work.voice.id)],
        )
        frames = [frame for frame, _ in _keys(silent_work)]
        assert frames == [51, 60, 120, 144]

    def test_a_missing_waveform_says_to_wait(self, silent_work: _Work) -> None:
        # 壊れると、波形がまだ無いときに何も下げずに成功と返す
        with pytest.raises(ToolError, match="波形"):
            _run(
                silent_work.host,
                "duck_audio",
                clip_ids=[str(silent_work.bgm.id)],
                voice_clip_ids=[str(silent_work.voice.id)],
            )
        assert silent_work.host.analyzed

    def test_subtitles_only_needs_subtitles(self, silent_work: _Work) -> None:
        # 壊れると、字幕で拾うよう頼んだのに黙って波形で拾い、AI の思った所と違う所を下げる
        with pytest.raises(ToolError, match="字幕がありません"):
            _run(
                silent_work.host,
                "duck_audio",
                clip_ids=[str(silent_work.bgm.id)],
                voice_clip_ids=[str(silent_work.voice.id)],
                speech_from="subtitles",
            )

    def test_a_locked_voice_is_still_heard(self, work: _Work) -> None:
        # 壊れると、声のトラックを守るためにロックした人の BGM を下げられない（声は聞くだけ）
        work.host.apply_commands([SetTrackState(work.voice_track.id, locked=True)], "ロック")
        _run(
            work.host,
            "duck_audio",
            clip_ids=[str(work.bgm.id)],
            voice_track_id=str(work.voice_track.id),
        )
        assert _keys(work)
        # 下げる側がロックしていれば断る
        work.host.apply_commands([SetTrackState(work.bgm_track.id, locked=True)], "ロック")
        with pytest.raises(ToolError, match="ロック"):
            _run(
                work.host,
                "duck_audio",
                clip_ids=[str(work.bgm.id)],
                voice_track_id=str(work.voice_track.id),
            )

    @pytest.mark.parametrize("silenced", ["muted", "solo_elsewhere", "disabled"])
    def test_a_silent_voice_does_not_duck(self, work: _Work, silenced: str) -> None:
        # 壊れると、書き出しで声が鳴らない（ミュート・ソロで外れた・無効）のに BGM だけが下がる
        if silenced == "muted":
            command: Any = SetTrackState(work.voice_track.id, muted=True)
        elif silenced == "solo_elsewhere":
            command = SetTrackState(work.bgm_track.id, solo=True)
        else:
            command = SetClipProperty(work.voice.id, "enabled", False)
        work.host.apply_commands([command], "黙らせる")
        before = work.host.document.project
        with pytest.raises(ToolError, match="鳴りません"):
            _run(
                work.host,
                "duck_audio",
                clip_ids=[str(work.bgm.id)],
                voice_track_id=str(work.voice_track.id),
            )
        assert work.host.document.project == before

    def test_the_voice_cannot_duck_itself(self, work: _Work) -> None:
        # 壊れると、話している所だけ声が小さくなる
        with pytest.raises(ToolError, match="重なって"):
            _run(
                work.host,
                "duck_audio",
                clip_ids=[str(work.voice.id)],
                voice_clip_ids=[str(work.voice.id)],
            )

    def test_bgm_outside_the_speech_is_refused(self, work: _Work) -> None:
        # 壊れると、何も変えていないのに AI は下げたと伝える
        from sashimono.core.commands import MoveClip

        work.host.apply_commands([MoveClip(work.bgm.id, 2000)], "動かす")
        before = work.host.document.project
        with pytest.raises(ToolError, match="話している所がありません"):
            _run(
                work.host,
                "duck_audio",
                clip_ids=[str(work.bgm.id)],
                voice_clip_ids=[str(work.voice.id)],
            )
        assert work.host.document.project == before

    def test_an_explicit_null_is_refused(self, work: _Work) -> None:
        # 壊れると、clip_ids に null を渡した頼みが、トラックごと下げる頼みに化ける
        with pytest.raises(ToolError, match="null"):
            _run(
                work.host,
                "duck_audio",
                clip_ids=None,
                track_id=str(work.bgm_track.id),
                voice_clip_ids=[str(work.voice.id)],
            )
        assert work.host.document.project == work.project

    def test_both_ways_of_naming_are_refused(self, work: _Work) -> None:
        # 壊れると、clip_ids と track_id のどちらで下げたのか分からない
        with pytest.raises(ToolError, match="どちらか 1 つ"):
            _run(
                work.host,
                "duck_audio",
                clip_ids=[str(work.bgm.id)],
                track_id=str(work.bgm_track.id),
                voice_clip_ids=[str(work.voice.id)],
            )


def _measured(
    host: FakeHost, loudness: float | None, peak_db: float, *, running: bool = False
) -> None:
    """最後の頼みを、``loudness`` LUFS と山 ``peak_db`` dBFS で測り終えた（か途中の）様子にする"""
    request = host.check_requests[-1]
    report = assemble(
        [],
        start=request.start,
        end=request.end,
        analyzed=request.end,
        fps=30.0,
        complete=not running,
        limits=request.limits,
        index=None,
        loudness=loudness,
        peak=db_to_ratio(peak_db),
        length_seconds=None,
    )
    host.check = CheckState(request=request, report=report, running=running)


class TestNormalize:
    def test_the_request_measures_only_the_chosen_clips(self, work: _Work) -> None:
        # 壊れると、BGM だけをそろえたいのに声まで混ぜて測り、BGM が小さすぎるか大きすぎる
        work.host.apply_commands([SetTrackState(work.voice_track.id, solo=True)], "ソロ")
        result = _run(work.host, "normalize_loudness", clip_ids=[str(work.bgm.id)], wait_seconds=0)
        assert result["measuring"] is True
        (request,) = work.host.check_requests
        assert request.limits.kinds == frozenset({LOUDNESS})
        assert (request.start, request.end) == (0, 600)
        assert request.project is not None
        clips = [c.id for t in request.project.timeline.tracks for c in t.clips]
        assert clips == [work.bgm.id]
        # ソロは聞き比べの物 残すと測りたい BGM が鳴らない
        assert not any(t.solo or t.muted for t in request.project.timeline.tracks)
        # 測り終えていないので何も変えていない
        assert clip_volume(work.clip(work.bgm.id)).static == 100.0

    def test_the_measured_gap_is_applied_to_every_clip(self, work: _Work) -> None:
        # 壊れると、基準より 6 LU 小さい作品をそのまま渡し、配信でほかの動画より小さく聞こえる
        _run(work.host, "normalize_loudness", wait_seconds=0)
        _measured(work.host, -20.0, -10.0)
        result = _run(work.host, "normalize_loudness", wait_seconds=0)
        assert result["gain_db"] == pytest.approx(6.0)
        assert result["expected_lufs"] == pytest.approx(-14.0)
        louder = 100.0 * db_to_ratio(6)
        assert clip_volume(work.clip(work.voice.id)).static == pytest.approx(louder)
        assert clip_volume(work.clip(work.bgm.id)).static == pytest.approx(louder)
        work.host.document.undo()
        assert work.host.document.project == work.project

    def test_the_whole_work_skips_muted_tracks(self, work: _Work) -> None:
        # 壊れると、ミュートした音まで混ぜて測り、書き出しで鳴る音と違う音量にそろえる
        work.host.apply_commands([SetTrackState(work.bgm_track.id, muted=True)], "ミュート")
        _run(work.host, "normalize_loudness", wait_seconds=0)
        request = work.host.check_requests[-1]
        assert request.project is not None
        clips = [c.id for t in request.project.timeline.tracks for c in t.clips]
        assert clips == [work.voice.id]

    def test_an_explicit_null_is_not_the_whole_work(self, work: _Work) -> None:
        # 壊れると、clip_ids に null を渡した AI の頼みで、作品全体の音量が変わる
        with pytest.raises(ToolError, match="null"):
            _run(work.host, "normalize_loudness", clip_ids=None, wait_seconds=0)
        assert work.host.check_requests == []

    def test_the_whole_work_follows_solo(self, work: _Work) -> None:
        # 壊れると、ソロで外れて鳴らない BGM まで測りに混ぜ（測るときはソロを外すので）、
        # 書き出す音が基準からずれる（#375 の Codex の指摘）
        work.host.apply_commands([SetTrackState(work.voice_track.id, solo=True)], "ソロ")
        _run(work.host, "normalize_loudness", wait_seconds=0)
        request = work.host.check_requests[-1]
        assert request.project is not None
        clips = [c.id for t in request.project.timeline.tracks for c in t.clips]
        assert clips == [work.voice.id]

    def test_disabled_clips_are_left_out(self, work: _Work) -> None:
        # 壊れると、鳴らない（無効の）クリップにも同じ dB を掛け、上限 400% の計算にも入る
        work.host.apply_commands([SetClipProperty(work.bgm.id, "enabled", False)], "無効")
        _run(work.host, "normalize_loudness", wait_seconds=0)
        request = work.host.check_requests[-1]
        assert request.project is not None
        clips = [c.id for t in request.project.timeline.tracks for c in t.clips]
        assert clips == [work.voice.id]
        _measured(work.host, -20.0, -10.0)
        _run(work.host, "normalize_loudness", wait_seconds=0)
        assert clip_volume(work.clip(work.bgm.id)).static == 100.0
        with pytest.raises(ToolError, match="無効"):
            _run(
                work.host,
                "normalize_loudness",
                clip_ids=[str(work.bgm.id)],
                wait_seconds=0,
            )

    def test_the_target_follows_the_preference(self, work: _Work) -> None:
        # 壊れると、放送向けに -23 を選んだ人の作品を配信の -14 へそろえる
        work.host.loudness = -23.0
        _run(work.host, "normalize_loudness", wait_seconds=0)
        assert work.host.check_requests[-1].limits.loudness_target == -23.0

    def test_peaks_stop_the_gain(self, work: _Work) -> None:
        # 壊れると、基準まで上げて音の山が 0 dBFS を超え、書き出しで音が割れる
        _run(work.host, "normalize_loudness", wait_seconds=0)
        _measured(work.host, -20.0, -3.0)
        result = _run(work.host, "normalize_loudness", wait_seconds=0)
        assert result["gain_db"] == pytest.approx(2.0)
        assert result["limited_by"] == ["peak"]
        assert result["expected_peak_dbfs"] == pytest.approx(-1.0)

    def test_a_loud_work_is_turned_down(self, work: _Work) -> None:
        # 壊れると、大きすぎる作品が配信で下げて再生され、ほかの音とそろわない
        _run(work.host, "normalize_loudness", wait_seconds=0)
        _measured(work.host, -8.0, -0.5)
        result = _run(work.host, "normalize_loudness", wait_seconds=0)
        assert result["gain_db"] == pytest.approx(-6.0)
        assert "limited_by" not in result

    def test_a_change_while_measuring_is_refused(self, work: _Work) -> None:
        # 壊れると、測った後に足した音まで同じだけ上げ、測った音量と合わなくなる
        pending = _run(work.host, "normalize_loudness", wait_seconds=10)
        assert isinstance(pending, Pending)
        work.host.apply_commands([SetTrackState(work.voice_track.id, solo=True)], "ソロ")
        changed = work.host.document.project
        _measured(work.host, -20.0, -10.0)
        with pytest.raises(ToolError, match="作品が変わりました"):
            pending.then()
        assert work.host.document.project == changed

    def test_a_silent_work_is_refused(self, work: _Work) -> None:
        # 壊れると、測れなかった音へ何 dB 掛けたのか分からないまま変える
        _run(work.host, "normalize_loudness", wait_seconds=0)
        _measured(work.host, None, -120.0)
        with pytest.raises(ToolError, match="測れませんでした"):
            _run(work.host, "normalize_loudness", wait_seconds=0)

    def test_waiting_returns_after_the_measure(self, work: _Work) -> None:
        # 壊れると、測るのを待たずに何も変えずに返り、AI が何度も呼び直す
        result = _run(work.host, "normalize_loudness", wait_seconds=10)
        assert isinstance(result, Pending)
        _measured(work.host, -17.0, -10.0)
        done = result.then()
        assert isinstance(done, dict)
        assert done["gain_db"] == pytest.approx(3.0)


class TestPlaceSound:
    def test_a_bgm_is_looped_and_cut_at_the_end(self, work: _Work) -> None:
        # 壊れると、30 秒の BGM を 70 秒の動画に敷けず、途中で音楽が途切れる
        result = _run(
            work.host,
            "place_sound",
            media_id=str(work.bgm_media.id),
            at_frame=600,
            until_frame=2700,
            track_id=str(work.bgm_track.id),
            volume_percent=40,
            fade_in_seconds=1,
            fade_out_seconds=2,
        )
        assert result["placed"] == 3
        assert result["track_ids"] == [str(work.bgm_track.id)]
        assert (result["from_frame"], result["to_frame"]) == (600, 2700)
        clips = [work.clip(ClipId(clip_id)) for clip_id in result["clip_ids"]]
        assert [(c.timeline_start, c.duration) for c in clips] == [
            (600, 900),
            (1500, 900),
            (2400, 300),
        ]
        assert all(clip_volume(c).static == 40.0 for c in clips)

        def fades(clip: Clip) -> tuple[object, object]:
            fade = next(e for e in clip.effects if e.fixed and e.kind == FADE_EFFECT_KIND)
            return fade.params["fade_in"], fade.params["fade_out"]

        assert fades(clips[0]) == (AnimatedValue(static=1.0), AnimatedValue(static=0.0))
        assert fades(clips[1]) == (AnimatedValue(static=0.0), AnimatedValue(static=0.0))
        assert fades(clips[2]) == (AnimatedValue(static=0.0), AnimatedValue(static=2.0))
        work.host.document.undo()
        assert work.host.document.project == work.project

    def test_without_an_end_it_places_once_at_the_playhead(self, work: _Work) -> None:
        # 壊れると、「ここに効果音」で再生位置ではない所へ置く
        work.host.seek(1000)
        result = _run(work.host, "place_sound", media_id=str(work.bgm_media.id))
        (clip_id,) = result["clip_ids"]
        clip = work.clip(ClipId(clip_id))
        assert (clip.timeline_start, clip.duration) == (1000, 900)

    def test_a_busy_track_moves_to_a_free_one(self, work: _Work) -> None:
        # 壊れると、声の上に効果音を重ねて置けず、何も置かれない
        result = _run(
            work.host,
            "place_sound",
            media_id=str(work.bgm_media.id),
            at_frame=100,
            track_id=str(work.voice_track.id),
        )
        assert result["track_ids"] != [str(work.voice_track.id)]

    def test_no_loop_only_cuts(self, work: _Work) -> None:
        # 壊れると、繰り返さないよう頼んだ効果音が並んで鳴る
        result = _run(
            work.host,
            "place_sound",
            media_id=str(work.bgm_media.id),
            at_frame=600,
            until_frame=2700,
            loop=False,
        )
        assert result["placed"] == 1

    def test_a_video_is_refused(self, work: _Work) -> None:
        # 壊れると、BGM のつもりで動画の絵まで置く
        with pytest.raises(ToolError, match="place_media"):
            _run(work.host, "place_sound", media_id=str(work.video_media.id))
        assert work.host.document.project == work.project
