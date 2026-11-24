"""素材の置き方の既定（#524）

同じ立ち絵や効果音を何度も置く作り方では、置くたびに同じ位置・拡大率・音量へ直していた
クリップで決めた置き方を素材に覚え、次に置くときの初めの値にする
"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sashimono.core.commands import (
    AddClip,
    Command,
    Document,
    SetMediaDefaults,
    defaults_from_clip,
    insert_media,
    placed_offset,
)
from sashimono.core.commands.fixed import (
    FADE_EFFECT_KIND,
    TRANSFORM_EFFECT_KIND,
    VOLUME_EFFECT_KIND,
)
from sashimono.core.io import media_memory
from sashimono.core.io.media_memory import MediaMemory, fingerprint, with_remembered
from sashimono.core.io.serialize import project_from_dict, project_to_dict
from sashimono.core.model import (
    AnimatedValue,
    AudioStreamInfo,
    Clip,
    Effect,
    Interpolation,
    Keyframe,
    LayerMode,
    MediaDefaults,
    MediaItem,
    PackedSpan,
    ParamValue,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    VideoStreamInfo,
)
from tests.conftest import RATE_30


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


def _empty(mode: str = LayerMode.SEPARATED, *media: MediaItem) -> Project:
    return Project.create(ProjectSettings(frame_rate=RATE_30, layer_mode=mode), media=media)


def _placed(project: Project, media: MediaItem, *, split: bool = False) -> Project:
    return _apply(project, insert_media(project, media, split_audio=split))


def _clips(project: Project) -> list[tuple[Track, Clip]]:
    return [(track, clip) for track in project.timeline.tracks for clip in track.clips]


def _fixed(clip: Clip, kind: str) -> Effect:
    return next(e for e in clip.effects if e.fixed and e.kind == kind)


def _value(clip: Clip, kind: str, name: str) -> object:
    value = _fixed(clip, kind).params[name]
    return value.static if isinstance(value, AnimatedValue) else value


def _edited(project: Project, clip_id: object, change: dict[str, dict[str, ParamValue]]) -> Project:
    """クリップの固定の項目の値を書き換える（設定パネルで値を動かしたのと同じ）"""
    for track, clip in _clips(project):
        if clip.id != clip_id:
            continue
        effects = tuple(
            replace(e, params={**e.params, **change[e.kind]}) if e.fixed and e.kind in change else e
            for e in clip.effects
        )
        clips = tuple(replace(c, effects=effects) if c.id == clip_id else c for c in track.clips)
        return project.with_timeline(project.timeline.replace_track(replace(track, clips=clips)))
    raise AssertionError(f"クリップが無い: {clip_id}")


def _with_clip(project: Project, clip_id: object, **fields: Any) -> Project:
    for track, clip in _clips(project):
        if clip.id == clip_id:
            clips = tuple(replace(c, **fields) if c.id == clip_id else c for c in track.clips)
            return project.with_timeline(
                project.timeline.replace_track(replace(track, clips=clips))
            )
    raise AssertionError(f"クリップが無い: {clip_id}")


def _picture(project: Project) -> Clip:
    return next(c for t, c in _clips(project) if project.draws_picture(t, c))


def _sound(project: Project) -> Clip:
    return next(c for t, c in _clips(project) if project.plays_sound(t, c))


def _remembered(project: Project, clip_id: object) -> Project:
    media_id, defaults = defaults_from_clip(project, clip_id)  # type: ignore[arg-type]
    return SetMediaDefaults(media_id, defaults).apply(project)


def _place_again(project: Project, media_id: object) -> list[Clip]:
    """覚えた素材を、空のタイムラインへもう 1 度置く"""
    media = project.find_media(media_id)  # type: ignore[arg-type]
    assert media is not None
    fresh = replace(project, timeline=replace(project.timeline, tracks=()))
    return [c for _, c in _clips(_placed(fresh, media))]


@pytest.fixture
def tuned(video_media: MediaItem) -> Project:
    """動画を置き、位置・拡大率・音量・フェード・不透明度を動かした作品（分ける方式）"""
    placed = _placed(_empty(), video_media)
    picture, sound = _picture(placed), _sound(placed)
    placed = _edited(
        placed,
        picture.id,
        {TRANSFORM_EFFECT_KIND: {"pos_x": AnimatedValue(120.0), "scale": AnimatedValue(55.0)}},
    )
    placed = _with_clip(placed, picture.id, opacity=AnimatedValue(0.5), native_size=False)
    return _edited(
        placed,
        sound.id,
        {
            VOLUME_EFFECT_KIND: {"volume": AnimatedValue(40.0)},
            FADE_EFFECT_KIND: {"fade_in": AnimatedValue(0.5)},
        },
    )


class TestRemember:
    def test_the_picture_and_the_linked_sound_are_both_remembered(self, tuned: Project) -> None:
        # 絵のクリップで選んでも、リンクした音の音量まで覚える 取りこぼすと
        # 「覚えたはずの音量が効かない」になる
        remembered = _remembered(tuned, _picture(tuned).id)
        defaults = remembered.media[0].defaults
        assert defaults is not None
        assert [e.kind for e in defaults.picture] == ["flip", TRANSFORM_EFFECT_KIND]
        assert defaults.opacity == 0.5
        assert defaults.native_size is False
        (stream, effects), *rest = defaults.sounds
        assert not rest
        assert stream == 1
        assert [e.kind for e in effects] == [VOLUME_EFFECT_KIND, FADE_EFFECT_KIND]

    def test_choosing_the_sound_clip_remembers_the_same(self, tuned: Project) -> None:
        # 壊れると、音のクリップで〔既定にする〕を選んだときだけ絵の位置や大きさを覚え損ねる
        by_picture = _remembered(tuned, _picture(tuned).id).media[0].defaults
        by_sound = _remembered(tuned, _sound(tuned).id).media[0].defaults
        assert by_picture is not None and by_sound is not None
        assert by_picture.opacity == by_sound.opacity
        assert [e.params for e in by_picture.picture] == [e.params for e in by_sound.picture]

    def test_clips_already_placed_are_not_changed(self, tuned: Project) -> None:
        # 置いた物まで書き換えると、場面ごとに位置を変えた立ち絵が全部同じ位置へ動く
        remembered = _remembered(tuned, _picture(tuned).id)
        assert remembered.timeline == tuned.timeline

    def test_keyframes_are_remembered_as_the_value_at_the_head(self, tuned: Project) -> None:
        # キーフレームのまま写すと、長さの違うクリップで動きが途中で切れる
        moving = AnimatedValue(
            0.0,
            (
                Keyframe(frame=0, value=30.0, interpolation=Interpolation.LINEAR),
                Keyframe(frame=60, value=90.0, interpolation=Interpolation.LINEAR),
            ),
        )
        picture = _picture(tuned)
        project = _edited(tuned, picture.id, {TRANSFORM_EFFECT_KIND: {"rotation": moving}})
        defaults = _remembered(project, picture.id).media[0].defaults
        assert defaults is not None
        rotation = next(e for e in defaults.picture if e.kind == TRANSFORM_EFFECT_KIND).params[
            "rotation"
        ]
        assert rotation == AnimatedValue(30.0)

    def test_a_text_clip_has_no_media_to_remember_into(self, tuned: Project) -> None:
        # 黙って何もしないと、本人は覚えたつもりで何も効かない
        text = Clip(timeline_start=900, duration=30)
        track = tuned.timeline.tracks[0]
        project = AddClip(track.id, text).apply(tuned)
        with pytest.raises(ValueError, match="素材を置いたクリップではない"):
            defaults_from_clip(project, text.id)

    def test_clearing_forgets_the_defaults(self, tuned: Project) -> None:
        # 壊れると、〔素材の既定を消す〕を押しても次に置くときに前の置き方が残る
        remembered = _remembered(tuned, _picture(tuned).id)
        cleared = SetMediaDefaults(remembered.media[0].id, None).apply(remembered)
        assert cleared.media[0].defaults is None

    def test_one_undo_brings_back_the_media_without_defaults(self, tuned: Project) -> None:
        # 壊れると、間違えて覚えた既定を取り消しで戻せない
        document = Document(tuned)
        media_id, defaults = defaults_from_clip(tuned, _picture(tuned).id)
        document.execute(SetMediaDefaults(media_id, defaults))
        assert document.project.media[0].defaults is not None
        document.undo()
        assert document.project.media[0].defaults is None


class TestPlace:
    def test_placing_again_starts_from_the_remembered_values(self, tuned: Project) -> None:
        # 壊れると、覚えたはずの位置・大きさ・音量が次に置いたときに効かない
        remembered = _remembered(tuned, _picture(tuned).id)
        picture, sound = _place_again(remembered, remembered.media[0].id)
        assert _value(picture, TRANSFORM_EFFECT_KIND, "pos_x") == 120.0
        assert _value(picture, TRANSFORM_EFFECT_KIND, "scale") == 55.0
        assert picture.opacity == AnimatedValue(0.5)
        assert picture.native_size is False
        assert _value(sound, VOLUME_EFFECT_KIND, "volume") == 40.0
        assert _value(sound, FADE_EFFECT_KIND, "fade_in") == 0.5

    def test_every_placed_effect_gets_its_own_id(self, tuned: Project) -> None:
        # 同じ ID が 2 本のクリップに並ぶと、値を動かす命令が別のクリップの物を掴む
        remembered = _remembered(tuned, _picture(tuned).id)
        defaults = remembered.media[0].defaults
        assert defaults is not None
        first = _place_again(remembered, remembered.media[0].id)
        second = _place_again(remembered, remembered.media[0].id)
        ids = [e.id for clip in (*first, *second) for e in clip.effects]
        assert len(ids) == len(set(ids))
        assert not {e.id for e in defaults.picture} & set(ids)

    def test_the_fixed_items_keep_their_order(self, tuned: Project) -> None:
        # 並びが崩れると、反転と配置の掛かる順が変わり絵が変わる
        remembered = _remembered(tuned, _picture(tuned).id)
        picture, sound = _place_again(remembered, remembered.media[0].id)
        assert [e.kind for e in picture.effects if e.fixed] == ["flip", TRANSFORM_EFFECT_KIND]
        assert [e.kind for e in sound.effects if e.fixed] == [VOLUME_EFFECT_KIND, FADE_EFFECT_KIND]

    def test_media_without_defaults_places_as_before(self, video_media: MediaItem) -> None:
        # 壊れると、既定を覚えていない素材まで位置や音量が変わって置かれる
        picture, sound = (c for _, c in _clips(_placed(_empty(), video_media)))
        assert _value(picture, TRANSFORM_EFFECT_KIND, "pos_x") == 0.0
        assert picture.opacity == AnimatedValue(1.0)
        assert picture.native_size
        assert _value(sound, VOLUME_EFFECT_KIND, "volume") == 100.0

    def test_the_mixed_layout_remembers_and_places_one_clip(self, video_media: MediaItem) -> None:
        # 混合の方式で 1 本にまとめたクリップは、絵と音の欄を両方持つ
        project = _placed(_empty(LayerMode.MIXED), video_media, split=False)
        ((_, merged),) = _clips(project)
        project = _edited(
            project,
            merged.id,
            {
                TRANSFORM_EFFECT_KIND: {"pos_y": AnimatedValue(-80.0)},
                VOLUME_EFFECT_KIND: {"volume": AnimatedValue(70.0)},
            },
        )
        remembered = _remembered(project, merged.id)
        fresh = replace(remembered, timeline=replace(remembered.timeline, tracks=()))
        ((_, again),) = _clips(_placed(fresh, remembered.media[0], split=False))
        assert _value(again, TRANSFORM_EFFECT_KIND, "pos_y") == -80.0
        assert _value(again, VOLUME_EFFECT_KIND, "volume") == 70.0

    def test_each_audio_stream_keeps_its_own_volume(self) -> None:
        # ゲームの録画は、声とゲーム音で音量が違う 1 本の値でそろえると声が埋もれる
        def stream(index: int) -> AudioStreamInfo:
            return AudioStreamInfo(index, 48000, 2, Fraction(1, 48000), "aac")

        media = MediaItem(
            path=Path("C:/素材/録画.mkv"),
            duration=Fraction(10),
            video_streams=(VideoStreamInfo(0, 1920, 1080, RATE_30, Fraction(1, 15360), "h264"),),
            audio_streams=(stream(1), stream(2)),
        )
        project = _placed(_empty(LayerMode.SEPARATED, media), media, split=True)
        for clip in [c for t, c in _clips(project) if t.kind is TrackKind.AUDIO]:
            volume = 30.0 if clip.stream_index == 1 else 90.0
            project = _edited(
                project, clip.id, {VOLUME_EFFECT_KIND: {"volume": AnimatedValue(volume)}}
            )
        remembered = _remembered(project, _picture(project).id)
        fresh = replace(remembered, timeline=replace(remembered.timeline, tracks=()))
        placed = _placed(fresh, remembered.media[0], split=True)
        volumes = {
            c.stream_index: _value(c, VOLUME_EFFECT_KIND, "volume")
            for t, c in _clips(placed)
            if t.kind is TrackKind.AUDIO
        }
        assert volumes == {1: 30.0, 2: 90.0}


class TestOffset:
    def test_the_head_offset_is_used_and_the_clip_is_shortened(self, tuned: Project) -> None:
        # ずれのぶん縮めないと、素材の終わりの後ろまでクリップが伸びて黒が映る
        picture = _picture(tuned)
        project = tuned
        for _, clip in _clips(tuned):
            project = _with_clip(project, clip.id, source_in=Fraction(2))
        remembered = _remembered(project, picture.id)
        again = _place_again(remembered, remembered.media[0].id)
        assert {c.source_in for c in again} == {Fraction(2)}
        assert {c.duration for c in again} == {240}

    def test_no_offset_is_remembered_from_the_head(self, tuned: Project) -> None:
        # 壊れると、頭から置いたクリップでも頭のずれ 0 を覚え、
        # 後から素材を差し替えたときに頭が欠ける
        defaults = _remembered(tuned, _picture(tuned).id).media[0].defaults
        assert defaults is not None
        assert defaults.offset is None

    def test_an_offset_past_the_end_is_not_used(self, video_media: MediaItem) -> None:
        # 短い素材へ差し替えたのに既定が残ると、長さ 0 のクリップになる
        media = replace(video_media, defaults=MediaDefaults(offset=Fraction(10)))
        assert placed_offset(media) == 0
        media = replace(video_media, defaults=MediaDefaults(offset=Fraction(25)))
        assert placed_offset(media) == 0

    def test_packed_media_ignores_the_offset(self, video_media: MediaItem) -> None:
        # 範囲だけを入れた素材は入っている区間に合わせて置く ずれで外を読むと断られる
        media = replace(
            video_media,
            packed=(PackedSpan(Fraction(4), Fraction(6)),),
            defaults=MediaDefaults(offset=Fraction(1)),
        )
        assert placed_offset(media) == 0
        ((_, picture), *_) = _clips(_placed(_empty(LayerMode.SEPARATED, media), media))
        assert picture.source_in == Fraction(4)


class TestSaved:
    def test_the_defaults_survive_saving_and_opening(self, tuned: Project) -> None:
        # 壊れると、作品を保存して開き直すたびに覚えた既定が消える
        remembered = _remembered(tuned, _picture(tuned).id)
        reopened = project_from_dict(json.loads(json.dumps(project_to_dict(remembered))))
        assert reopened.media[0].defaults == remembered.media[0].defaults

    def test_media_without_defaults_writes_nothing_new(self, tuned: Project) -> None:
        # 書かなければ、前の版の本体で開いても前と同じ中身に見える
        (written,) = project_to_dict(tuned)["media"]
        assert "defaults" not in written

    def test_broken_defaults_do_not_stop_the_project_from_opening(self, tuned: Project) -> None:
        # 既定のために作品ごと開けなくすると、編集したクリップまで触れなくなる
        data = project_to_dict(_remembered(tuned, _picture(tuned).id))
        data["media"][0]["defaults"]["opacity"] = "濃い"
        reopened = project_from_dict(data)
        assert reopened.media[0].defaults is None


class TestMemory:
    """作品をまたいで本人の置き場に覚える分"""

    @pytest.fixture
    def defaults(self) -> MediaDefaults:
        return MediaDefaults(opacity=0.25, offset=Fraction(3, 2))

    @pytest.fixture
    def sound_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "効果音.wav"
        path.write_bytes(b"RIFF" + bytes(range(256)) * 9000)
        return path

    def test_a_remembered_file_is_recalled(
        self, tmp_path: Path, sound_file: Path, defaults: MediaDefaults
    ) -> None:
        # 壊れると、ほかの作品で同じ素材を読み込んでも覚えた既定が効かない
        memory = MediaMemory(tmp_path / "memory.json")
        key = fingerprint(sound_file)
        assert key is not None
        memory.remember(key, sound_file.name, defaults)
        assert MediaMemory(memory.path).recall(key) == defaults

    def test_a_moved_and_renamed_file_is_still_recognized(
        self, tmp_path: Path, sound_file: Path
    ) -> None:
        # 場所や名前で引くと、素材のフォルダを整理しただけで覚えた既定を見失う
        moved = tmp_path / "整理した" / "ちがう名前.wav"
        moved.parent.mkdir()
        moved.write_bytes(sound_file.read_bytes())
        assert fingerprint(moved) == fingerprint(sound_file)

    def test_a_different_file_is_not_mistaken(self, tmp_path: Path, sound_file: Path) -> None:
        # 壊れると、別の素材に覚えた置き方が当たり、思わぬ位置や音量で置かれる
        other = tmp_path / "別.wav"
        content = bytearray(sound_file.read_bytes())
        content[-1] ^= 0xFF
        other.write_bytes(bytes(content))
        assert fingerprint(other) != fingerprint(sound_file)

    def test_an_unreadable_file_has_no_fingerprint(self, tmp_path: Path) -> None:
        # 壊れると、読めないファイルを読み込んだときに落ちる
        assert fingerprint(tmp_path / "無い.wav") is None

    def test_forgetting_removes_the_entry(
        self, tmp_path: Path, sound_file: Path, defaults: MediaDefaults
    ) -> None:
        # 壊れると、消したはずの既定がほかの作品でまだ効く
        memory = MediaMemory(tmp_path / "memory.json")
        key = fingerprint(sound_file)
        assert key is not None
        memory.remember(key, "効果音", defaults)
        memory.remember(key, "効果音", None)
        assert memory.recall(key) is None

    def test_a_broken_memory_file_recalls_nothing(self, tmp_path: Path) -> None:
        # 壊れていても読み込みは止めない 覚えた既定が効かないだけ
        path = tmp_path / "memory.json"
        path.write_text("{壊れた", encoding="utf-8")
        assert MediaMemory(path).recall("a" * 64) is None
        path.write_text(
            json.dumps({"version": 1, "entries": {"k": {"defaults": {"opacity": 3}}}}),
            encoding="utf-8",
        )
        assert MediaMemory(path).recall("k") is None

    def test_a_newer_format_is_not_read(self, tmp_path: Path, defaults: MediaDefaults) -> None:
        # 壊れると、新しい版が書いた置き場を前の版が読み違えて、壊れた既定を当てる
        path = tmp_path / "memory.json"
        MediaMemory(path).remember("k", "効果音", defaults)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
        assert MediaMemory(path).recall("k") is None

    def test_the_oldest_are_forgotten_past_the_limit(
        self, tmp_path: Path, defaults: MediaDefaults, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 際限なく育つと、読み込みのたびに読む時間が延びる
        monkeypatch.setattr(media_memory, "MEMORY_LIMIT", 2)
        memory = MediaMemory(tmp_path / "memory.json")
        # 覚えた時刻だけを決める 錠と書き込みも時計を読むので、時計そのものは差し替えない
        for when, key in enumerate(("古い", "中", "新しい")):
            monkeypatch.setattr(media_memory, "_now", lambda when=when: float(when))
            memory.remember(key, key, defaults)
        assert memory.recall("古い") is None
        assert memory.recall("中") == defaults
        assert memory.recall("新しい") == defaults

    def test_imported_media_receives_the_remembered_defaults(
        self, tmp_path: Path, sound_file: Path, defaults: MediaDefaults, audio_media: MediaItem
    ) -> None:
        # 壊れると、読み込んだ素材に覚えた既定が写らず、ほかの作品では毎回直すことになる
        memory = MediaMemory(tmp_path / "memory.json")
        key = fingerprint(sound_file)
        assert key is not None
        memory.remember(key, "効果音", defaults)
        imported = replace(audio_media, path=sound_file)
        assert with_remembered(imported, memory).defaults == defaults

    def test_defaults_of_the_project_win_over_the_memory(
        self, tmp_path: Path, sound_file: Path, defaults: MediaDefaults, audio_media: MediaItem
    ) -> None:
        # 作品ごとに決めた置き方の方が、その作品では正しい
        memory = MediaMemory(tmp_path / "memory.json")
        key = fingerprint(sound_file)
        assert key is not None
        memory.remember(key, "効果音", defaults)
        own = MediaDefaults(opacity=0.75)
        imported = replace(audio_media, path=sound_file, defaults=own)
        assert with_remembered(imported, memory).defaults == own
