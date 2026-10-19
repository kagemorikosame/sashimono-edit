"""色と音の雑音を自動で直す道具（#336 auto_color・reduce_noise）

どちらも人が設定パネルで触るのと同じエフェクト（拡張色調補正・ノイズ除去）を、人の操作と
同じ命令で積むだけ 取り消し 1 回で戻り、元の素材は触らない 測るのはサムネイルと波形
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound
from sashimono.core.commands import AddClip, AddEffect, SetTrackState
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Keyframe,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from sashimono.effects import registry
from sashimono.effects.audio import DENOISE_KIND
from sashimono.effects.sources import TEXT
from sashimono.engine.audio.waveform import BASE_SAMPLES_PER_PEAK, PeakLevel, Waveform
from sashimono.engine.auto_color import GRADE_KIND
from sashimono.engine.cache.thumbnails import Filmstrip
from tests.ai.conftest import RATE_30, FakeHost
from tests.conftest import make_clip
from tests.engine.test_auto_color import _scene


def _call(host: FakeHost, name: str, arguments: dict[str, object]) -> dict[str, object]:
    operation = find_operation(name)
    assert operation is not None
    result = operation(host, arguments)
    assert isinstance(result, dict)
    return result


def _strip(image: np.ndarray, count: int = 20) -> Filmstrip:
    """同じ絵を ``count`` 枚並べた 0.5 秒ごとのサムネイルのシート"""
    return Filmstrip(
        sheet=np.concatenate([image] * count, axis=1),
        tile_width=image.shape[1],
        interval=Fraction(1, 2),
    )


def _waveform(noise: float, speech: float = 0.5, seconds: int = 10) -> Waveform:
    """0.5 秒ごとに話しては黙る波形（山の大きさ） 黙った所に ``noise`` の雑音が残る"""
    steps = seconds * 48000 // BASE_SAMPLES_PER_PEAK
    amplitudes = np.full(steps, noise, dtype=np.float32)
    talking = (np.arange(steps) * BASE_SAMPLES_PER_PEAK // 24000) % 2 == 0
    amplitudes[talking] = speech
    peaks = np.zeros((steps, 2, 2), dtype=np.float32)
    peaks[:, :, 0] = -amplitudes[:, None]
    peaks[:, :, 1] = amplitudes[:, None]
    return Waveform(
        sample_rate=48000,
        channels=2,
        total_samples=steps * BASE_SAMPLES_PER_PEAK,
        levels=(PeakLevel(BASE_SAMPLES_PER_PEAK, peaks),),
    )


class Work:
    """動画のクリップ 2 本（映像トラック）と、その音のクリップ（音声トラック）とテキスト"""

    def __init__(self, video_media: MediaItem) -> None:
        other = replace(
            video_media, path=video_media.path.with_name("別のカメラ.mp4"), id=MediaId("camera-b")
        )
        self.media = video_media
        self.other = other
        base = Project.create(ProjectSettings(frame_rate=RATE_30), media=(video_media, other))
        self.video = Track(kind=TrackKind.VIDEO, name="V1")
        self.audio = Track(kind=TrackKind.AUDIO, name="A1")
        base = base.with_timeline(replace(base.timeline, tracks=(self.video, self.audio)))
        self.picture = make_clip(0, 150, video_media)
        self.second = make_clip(150, 150, other)
        self.sound = replace(make_clip(0, 150, video_media), stream_index=1)
        self.text = Clip(timeline_start=300, duration=30, source=TEXT.create(text="題"))
        for track, clip in (
            (self.video, self.picture),
            (self.video, self.second),
            (self.audio, self.sound),
            (self.video, self.text),
        ):
            base = AddClip(track.id, clip).apply(base)
        self.project = base
        self.host = FakeHost(base)

    def clip(self, clip: Clip) -> Clip:
        located = self.host.document.project.timeline.locate_clip(clip.id)
        assert located is not None
        return located[1]


@pytest.fixture
def work(video_media: MediaItem) -> Work:
    return Work(video_media)


class TestAutoColor:
    def test_a_grade_goes_to_the_head_and_one_undo_takes_it_back(self, work: Work) -> None:
        # 壊れると、色の補正を後ろに積んで、前のエフェクトを通した絵に素材の色で決めた値を
        # 掛けるか、取り消しが何回も要る
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.25, light=0.7))
        result = _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})
        head = work.clip(work.picture).effects[0]
        assert head.kind == GRADE_KIND
        rows = result["clips"]
        assert isinstance(rows, list) and rows[0]["replaced"] is False
        assert work.host.document.undo_label is not None
        work.host.document.undo()
        assert work.host.document.project == work.project

    def test_calling_again_replaces_instead_of_stacking(self, work: Work) -> None:
        # 壊れると、掛け直すたびに補正が重なり、2 回目で倍に直って色が飛ぶ
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.25, light=0.7))
        _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})
        result = _call(
            work.host,
            "auto_color",
            {"clip_ids": [str(work.picture.id)], "strength_percent": 50},
        )
        grades = [e for e in work.clip(work.picture).effects if e.kind == GRADE_KIND]
        assert len(grades) == 1
        rows = result["clips"]
        assert isinstance(rows, list) and rows[0]["replaced"] is True
        value = grades[0].params["luma_gain"]
        assert isinstance(value, AnimatedValue)
        assert value.static == rows[0]["params"]["luma_gain"]

    def test_values_a_person_added_to_the_same_grade_are_kept(self, work: Work) -> None:
        # 壊れると、人が同じ補正に足した色相やリフトまで、掛け直しで 0 に戻る
        grade = registry.require(GRADE_KIND).create(hue_offset=30.0, luma_gain=5.0)
        work.host.apply_commands([AddEffect(work.picture.id, grade, index=0)], "人")
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.25, light=0.7))
        _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})
        head = work.clip(work.picture).effects[0]
        hue = head.params["hue_offset"]
        assert isinstance(hue, AnimatedValue) and hue.static == 30.0

    def test_a_turned_off_grade_is_turned_back_on(self, work: Work) -> None:
        # 壊れると、人が切った補正の値だけを直し、効いていないのに整えたと返す
        grade = replace(registry.require(GRADE_KIND).create(), enabled=False)
        work.host.apply_commands([AddEffect(work.picture.id, grade, index=0)], "人")
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.25, light=0.7))
        _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})
        head = work.clip(work.picture).effects[0]
        assert head.id == grade.id and head.enabled

    def test_a_moving_grade_is_not_overwritten(self, work: Work) -> None:
        # 壊れると、キーフレームで動かした補正を 1 つの値で上書きし、動きが消える
        moving = AnimatedValue(5.0, keyframes=(Keyframe(0, 0.0), Keyframe(30, 20.0)))
        grade = registry.require(GRADE_KIND).create()
        grade = grade.with_param("luma_gain", moving)
        work.host.apply_commands([AddEffect(work.picture.id, grade, index=0)], "人")
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.25, light=0.7))
        before = work.host.document.project
        with pytest.raises(ToolError, match="キーフレーム"):
            _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})
        assert work.host.document.project == before

    def test_the_preference_is_the_default_strength(self, work: Work) -> None:
        # 壊れると、設定画面で選んだ強さが AI に効かない
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.25, light=0.7))
        work.host.color_strength = 40
        result = _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})
        assert result["strength_percent"] == 40.0

    def test_a_good_picture_is_left_unchanged(self, work: Work) -> None:
        # 壊れると、整っている絵まで補正を積み、取り消しの履歴に何もしない段が増える
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.0, light=1.0))
        result = _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})
        assert result["unchanged_clip_ids"] == [str(work.picture.id)]
        assert work.host.document.project == work.project

    def test_the_other_camera_is_matched(self, work: Work) -> None:
        # 壊れると、別のカメラのクリップを合わせる相手に寄せられない
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.05, light=0.95))
        work.host.filmstrips[work.other.id] = _strip(
            _scene(dark=0.2, light=0.6, tint=(0.9, 1.0, 1.1))
        )
        result = _call(
            work.host,
            "auto_color",
            {"clip_ids": [str(work.second.id)], "match_clip_id": str(work.picture.id)},
        )
        rows = result["clips"]
        assert isinstance(rows, list)
        after = rows[0]["after"]
        before = rows[0]["before"]
        assert after["white_percent"] > before["white_percent"] + 15

    def test_the_reference_cannot_be_one_of_the_targets(self, work: Work) -> None:
        # 壊れると、自分に合わせて何も変わらないのに、変えたと返す
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.2, light=0.6))
        with pytest.raises(ToolError, match="match_clip_id"):
            _call(
                work.host,
                "auto_color",
                {"clip_ids": [str(work.picture.id)], "match_clip_id": str(work.picture.id)},
            )

    def test_missing_thumbnails_are_asked_for_and_nothing_changes(self, work: Work) -> None:
        # 壊れると、サムネイルがまだ無い素材で黙って何もしないか、作る頼みを出さずに待たせ続ける
        with pytest.raises(ToolError, match="サムネイルがまだ"):
            _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})
        assert work.host.analyzed == [work.media.id]
        assert work.host.document.project == work.project

    def test_a_text_clip_is_refused(self, work: Work) -> None:
        # 壊れると、測る絵の無いテキストに補正を積み、色を整えたと返す
        with pytest.raises(ToolError, match="素材の絵を描くクリップではありません"):
            _call(work.host, "auto_color", {"clip_ids": [str(work.text.id)]})

    def test_a_locked_track_is_refused(self, work: Work) -> None:
        # 壊れると、人が触らせないためにロックしたトラックの色を AI が変える
        work.host.apply_commands([SetTrackState(work.video.id, locked=True)], "ロック")
        work.host.filmstrips[work.media.id] = _strip(_scene(dark=0.2, light=0.6))
        with pytest.raises(ToolError, match="ロック"):
            _call(work.host, "auto_color", {"clip_ids": [str(work.picture.id)]})

    def test_turning_both_off_is_refused(self, work: Work) -> None:
        # 壊れると、何も直さない呼び方で「整えた」と返す
        with pytest.raises(ToolError, match="何も直しません"):
            _call(
                work.host,
                "auto_color",
                {
                    "clip_ids": [str(work.picture.id)],
                    "exposure": False,
                    "white_balance": False,
                },
            )


class TestReduceNoise:
    def test_a_noisy_voice_gets_the_effect_and_one_undo_takes_it_back(self, work: Work) -> None:
        # 壊れると、雑音の聞こえる声に何も掛けないか、取り消しが何回も要る
        work.host.stub_waveform = _waveform(noise=0.01)
        result = _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)]})
        effects = [e for e in work.clip(work.sound).effects if e.kind == DENOISE_KIND]
        assert len(effects) == 1
        reduction = effects[0].params["reduction"]
        assert isinstance(reduction, AnimatedValue) and reduction.static == 12.0
        rows = result["clips"]
        assert isinstance(rows, list) and rows[0]["noise_dbfs"] == pytest.approx(-40.0, abs=0.5)
        work.host.document.undo()
        assert work.host.document.project == work.project

    def test_it_goes_before_the_volume(self, work: Work) -> None:
        # 壊れると、音量を上げた後に雑音を測って下げるので、音量の欄を触るたびに効きが変わる
        work.host.stub_waveform = _waveform(noise=0.01)
        _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)]})
        kinds = [e.kind for e in work.clip(work.sound).effects]
        fixed = [i for i, e in enumerate(work.clip(work.sound).effects) if e.fixed]
        assert not fixed or kinds.index(DENOISE_KIND) < fixed[0]

    def test_calling_again_replaces_the_values(self, work: Work) -> None:
        # 壊れると、掛け直すたびにノイズ除去が重なって声まで削れる
        work.host.stub_waveform = _waveform(noise=0.01)
        _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)]})
        result = _call(
            work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)], "reduction_db": 20}
        )
        effects = [e for e in work.clip(work.sound).effects if e.kind == DENOISE_KIND]
        assert len(effects) == 1
        reduction = effects[0].params["reduction"]
        assert isinstance(reduction, AnimatedValue) and reduction.static == 20.0
        rows = result["clips"]
        assert isinstance(rows, list) and rows[0]["replaced"] is True

    def test_a_quiet_recording_is_skipped_with_a_reason(self, work: Work) -> None:
        # 壊れると、雑音の聞こえない録音にも掛けて、声の小さい所を削る
        work.host.stub_waveform = _waveform(noise=10 ** (-75 / 20))
        result = _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)]})
        skipped = result["skipped"]
        assert isinstance(skipped, list) and "聞こえない" in skipped[0]["reason"]
        assert work.host.document.project == work.project

    def test_music_is_skipped_unless_forced(self, work: Work) -> None:
        # 壊れると、黙る所の無い BGM にも掛けて音楽を削るか、force でも掛けられない
        work.host.stub_waveform = _waveform(noise=0.4, speech=0.5)
        result = _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)]})
        assert "skipped" in result
        assert work.host.document.project == work.project
        forced = _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)], "force": True})
        rows = forced["clips"]
        assert isinstance(rows, list) and "forced" in rows[0]
        assert any(e.kind == DENOISE_KIND for e in work.clip(work.sound).effects)

    def test_the_preference_is_the_default_amount(self, work: Work) -> None:
        # 壊れると、設定画面で選んだ下げる量が AI に効かない
        work.host.stub_waveform = _waveform(noise=0.01)
        work.host.noise_reduction = 6
        result = _call(work.host, "reduce_noise", {"track_id": str(work.audio.id)})
        assert result["reduction_db"] == 6.0

    def test_a_missing_waveform_is_asked_for_and_nothing_changes(self, work: Work) -> None:
        # 壊れると、波形の解析を頼まずに待たせ続けるか、測らずに掛ける
        with pytest.raises(ToolError, match="波形解析がまだ") as caught:
            _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)]})
        assert work.media.name in str(caught.value)
        operation = find_operation("reduce_noise")
        assert operation is not None and Outbound.FILES in operation.sends
        assert work.host.analyzed == [work.media.id]
        assert work.host.document.project == work.project

    def test_a_picture_clip_is_refused(self, work: Work) -> None:
        # 壊れると、音の鳴らない絵のクリップにノイズ除去を積み、減らしたと返す
        with pytest.raises(ToolError, match="音を鳴らさない"):
            _call(work.host, "reduce_noise", {"clip_ids": [str(work.picture.id)]})

    def test_out_of_range_amounts_are_refused(self, work: Work) -> None:
        # 壊れると、範囲の外の量を黙って端へ寄せ、頼んだ量で下げたと思い込ませる
        work.host.stub_waveform = _waveform(noise=0.01)
        with pytest.raises(ToolError, match="reduction_db"):
            _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)], "reduction_db": 80})

    def test_a_turned_off_effect_is_turned_back_on(self, work: Work) -> None:
        # 壊れると、人が切ったノイズ除去の値だけを直し、効いていないのに減らしたと返す
        effect = replace(registry.require(DENOISE_KIND).create(), enabled=False)
        work.host.apply_commands([AddEffect(work.sound.id, effect)], "人")
        work.host.stub_waveform = _waveform(noise=0.01)
        _call(work.host, "reduce_noise", {"clip_ids": [str(work.sound.id)]})
        found = [e for e in work.clip(work.sound).effects if e.kind == DENOISE_KIND]
        assert len(found) == 1 and found[0].enabled
