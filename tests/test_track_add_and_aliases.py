"""トラックを足す決まり・右クリックした所へ置く決まり・自分で保存するエイリアス（Issue #27）"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sashimono.core.commands import AddClip, Command, insert_filter, insert_generated
from sashimono.core.commands.insert import is_effect_track, new_track
from sashimono.core.io.aliases import Alias, AliasStore, alias_refusal
from sashimono.core.io.serialize import FORMAT_VERSION, ProjectFileError
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GroupId,
    MediaId,
    MediaItem,
    Project,
    SceneId,
    SegmentId,
    SubtitleOrigin,
    Track,
    TrackKind,
    new_clip_id,
)
from sashimono.effects import registry
from sashimono.effects.sources import TEXT


def _project(*tracks: Track) -> Project:
    base = Project.create()
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _apply(project: Project, commands: list[Command]) -> Project:
    for command in commands:
        project = command.apply(project)
    return project


class TestNewTrack:
    def test_names_count_up_per_kind(self) -> None:
        # 種類ごとに数えないと、音声を足しただけで V2 が抜けたり A3 から始まったりして、
        # どのトラックが何本目か名前から読めない
        project = _project(Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1"))
        assert new_track(project, TrackKind.VIDEO).track.name == "V2"
        assert new_track(project, TrackKind.AUDIO).track.name == "A2"
        assert new_track(project, TrackKind.VIDEO, effect=True).track.name == "FX1"

    def test_a_name_left_by_a_removed_track_is_skipped(self) -> None:
        # V2 を消した後に V1 と V3 が残ると、数えただけでは V3 がもう 1 本できる
        project = _project(Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V3"))
        assert new_track(project, TrackKind.VIDEO).track.name == "V4"

    def test_an_effect_track_is_known_by_its_name(self) -> None:
        # 形式を変えずに見分ける 種類を増やすと古い版で開けなくなる
        project = _project(Track(TrackKind.VIDEO, "V1"))
        track = new_track(project, TrackKind.VIDEO, effect=True).track
        assert track.kind is TrackKind.VIDEO
        assert is_effect_track(track)
        assert not is_effect_track(Track(TrackKind.VIDEO, "FX 素材"))
        assert not is_effect_track(Track(TrackKind.AUDIO, "FX1"))

    def test_an_audio_effect_track_is_refused(self) -> None:
        # 音声のエフェクトトラックを作れると、そこへ置いたフィルタは映像に何も掛からない
        with pytest.raises(ValueError):
            new_track(_project(), TrackKind.AUDIO, effect=True)

    def test_a_new_track_joins_the_solo(self) -> None:
        # ソロで絞っている間に足したトラックは、ソロが無いと足した所で出なくなる
        project = _project(Track(TrackKind.VIDEO, "V1", solo=True))
        assert new_track(project, TrackKind.VIDEO).track.solo
        assert not new_track(project, TrackKind.AUDIO).track.solo

    def test_video_goes_to_the_end_of_the_order(self) -> None:
        # 並びの末尾が一番上 間に挟まると、足しただけで重なり順が変わる
        project = _project(Track(TrackKind.VIDEO, "V1"), Track(TrackKind.AUDIO, "A1"))
        added = _apply(project, [new_track(project, TrackKind.VIDEO)])
        assert [t.name for t in added.timeline.tracks] == ["V1", "A1", "V2"]
        assert [t.name for t in added.timeline.video_tracks()] == ["V1", "V2"]


class TestPlacingOnAChosenTrack:
    def test_text_goes_to_the_asked_track(self) -> None:
        # いつもの決まり（下から空きを探す）だと V1 へ入り、右クリックした V2 に来ない
        v1, v2 = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2")
        project = _project(v1, v2)
        commands = insert_generated(project, TEXT.create(), at_frame=30, track_id=v2.id)
        (add,) = commands
        assert isinstance(add, AddClip) and add.track_id == v2.id

    def test_a_busy_or_audio_track_falls_back(self) -> None:
        # 断ると、右クリックした所が埋まっていただけで何も置けない
        busy = Clip(timeline_start=0, duration=100, source=TEXT.create())
        v1 = Track(TrackKind.VIDEO, "V1", (busy,))
        a1 = Track(TrackKind.AUDIO, "A1")
        project = _project(v1, a1)
        for asked in (v1.id, a1.id):
            commands = insert_generated(project, TEXT.create(), at_frame=10, track_id=asked)
            placed = _apply(project, commands)
            assert len(list(placed.timeline.video_tracks())) == 2

    def test_a_filter_goes_to_the_asked_track(self) -> None:
        # フィルタの決まり（絵の上）より、右クリックしたトラックを先にする
        v1, v2 = Track(TrackKind.VIDEO, "V1"), Track(TrackKind.VIDEO, "V2")
        project = _project(v1, v2)
        (add,) = insert_filter(project, at_frame=0, track_id=v1.id)
        assert isinstance(add, AddClip) and add.track_id == v1.id
        assert add.clip.is_filter

    def test_a_new_track_for_a_filter_skips_a_name_in_use(self) -> None:
        # V2 を消して V1 と V3 が残ると、本数を数えただけではもう 1 本の V3 ができる
        busy = Clip(timeline_start=0, duration=100, source=TEXT.create())
        v1 = Track(TrackKind.VIDEO, "V1", (busy,))
        v3 = Track(TrackKind.VIDEO, "V3", (replace(busy, id=new_clip_id()),))
        project = _project(v1, v3)
        placed = _apply(project, insert_filter(project, at_frame=10, track_id=v1.id))
        assert [t.name for t in placed.timeline.tracks] == ["V1", "V3", "V4"]
        placed = _apply(project, insert_generated(project, TEXT.create(), at_frame=10))
        assert [t.name for t in placed.timeline.tracks] == ["V1", "V3", "V4"]

    def test_the_old_callers_are_unchanged(self) -> None:
        # 位置を渡せるように広げても、渡さなければ前と同じ所へ置く
        busy = Clip(timeline_start=0, duration=100, source=TEXT.create())
        project = _project(Track(TrackKind.VIDEO, "V1", (busy,)), Track(TrackKind.VIDEO, "V2"))
        (add,) = insert_generated(project, TEXT.create(), at_frame=10)
        assert isinstance(add, AddClip)
        assert add.track_id == project.timeline.tracks[1].id


#: 字幕から焼き込んだテキストが持つ印 素材 1 本の音声 1 の 1 行を指す
_ORIGIN = SubtitleOrigin(media_id=MediaId("本編"), stream=1, segment_id=SegmentId("行"))


def _styled_text() -> Clip:
    blur = registry.get("blur")
    assert blur is not None
    return Clip(
        timeline_start=120,
        duration=45,
        source=TEXT.create(text="決め台詞"),
        effects=(blur.create(),),
        opacity=AnimatedValue(0.5),
        blend_mode="add",
    )


class TestAliases:
    def test_a_round_trip_keeps_the_contents_but_not_the_place(self, tmp_path: Path) -> None:
        # 中身が落ちると置き直したテロップの見た目が変わり、位置を持つと右クリックした所へ来ない
        store = AliasStore(tmp_path)
        clip = _styled_text()
        store.save(Alias.of("見出し", clip))
        (loaded,) = store.all()
        assert loaded.name == "見出し"
        placed = loaded.instantiate(at_frame=10)
        assert placed.timeline_start == 10
        assert placed.duration == 45
        assert placed.source == clip.source
        assert placed.opacity == clip.opacity and placed.blend_mode == "add"
        assert [e.kind for e in placed.effects] == ["blur"]
        # ID が重なると、同じエイリアスを 2 回置いたときに片方を消すと両方を探し当てる
        assert placed.id != clip.id
        assert placed.effects[0].id != clip.effects[0].id

    def test_a_group_or_link_is_not_carried(self, tmp_path: Path) -> None:
        # 保存した時のグループを持ち込むと、別のプロジェクトの見知らぬクリップと束になる
        clip = replace(_styled_text(), group_id=GroupId("g1"), link_group=GroupId("l1"))
        alias = Alias.of("束", clip)
        assert alias.clip.group_id is None and alias.clip.link_group is None

    def test_the_subtitle_mark_is_not_carried(self, tmp_path: Path) -> None:
        # 字幕から焼き込んだテキストの印を持ち込むと、置いた写しが素材と一緒にずれる仲間に
        # 数えられ、字幕の誤植を直すと写しの文字まで書き換わる（#282）
        clip = replace(_styled_text(), subtitle_origin=_ORIGIN)
        alias = Alias.of("焼き込み", clip)
        assert alias.clip.subtitle_origin is None
        store = AliasStore(tmp_path)
        store.save(alias)
        (loaded,) = store.all()
        assert loaded.instantiate(at_frame=10).subtitle_origin is None

    def test_an_old_file_with_the_mark_places_without_it(self, tmp_path: Path) -> None:
        # 印を外す前の版で保存したエイリアスには印が書いてある 置くときに外さないと、
        # 古いファイルから置いた写しだけ字幕の仲間のまま残る
        old = Alias("焼き込み", replace(_styled_text(), timeline_start=0, subtitle_origin=_ORIGIN))
        path = tmp_path / "焼き込み.smea"
        path.write_text(json.dumps(old.to_dict(), ensure_ascii=False), "utf-8")
        assert "subtitle_origin" in json.loads(path.read_text("utf-8"))["clip"]
        loaded = AliasStore(tmp_path).load(path)
        placed = loaded.instantiate(at_frame=10)
        assert placed.subtitle_origin is None
        # 印のほかは前のまま置く
        assert placed.source == old.clip.source

    def test_clips_with_media_or_scenes_are_refused(self, video_media: MediaItem) -> None:
        # 素材の道は本人の機械にしか無く、シーンの中身は元のプロジェクトにしか無い
        movie = Clip(timeline_start=0, duration=30, media_id=video_media.id)
        assert alias_refusal(movie) is not None
        with pytest.raises(ValueError):
            Alias.of("動画", movie)
        scene_clip = Clip(timeline_start=0, duration=30, scene_id=SceneId("s1"))
        assert alias_refusal(scene_clip) is not None
        with pytest.raises(ValueError):
            Alias.of("シーン", scene_clip)
        assert alias_refusal(_styled_text()) is None

    def test_a_newer_alias_is_not_read(self, tmp_path: Path) -> None:
        # 新しい版の項目を読み飛ばして置き直すと、作った物が黙って変わる
        data = Alias.of("新", _styled_text()).to_dict()
        for key, value in (("version", 99), ("clip_version", FORMAT_VERSION + 1)):
            with pytest.raises(ProjectFileError):
                Alias.from_dict({**data, key: value})

    def test_a_broken_file_does_not_hide_the_others(self, tmp_path: Path) -> None:
        # 壊れた 1 つで例外になると、〔エイリアス〕の一覧全体が出ず、正しい物も置けない
        store = AliasStore(tmp_path)
        store.save(Alias.of("残る", _styled_text()))
        (tmp_path / "壊れ.smea").write_text("{", "utf-8")
        (tmp_path / "別物.smea").write_text(json.dumps({"format": "other"}), "utf-8")
        assert [a.name for a in store.all()] == ["残る"]

    def test_a_file_that_is_not_utf8_does_not_hide_the_others(self, tmp_path: Path) -> None:
        # UnicodeDecodeError は JSON の失敗とは別 変えずに通すと一覧を作る所で落ちる
        store = AliasStore(tmp_path)
        store.save(Alias.of("残る", _styled_text()))
        (tmp_path / "シフトJIS.smea").write_bytes('{"name": "字"}'.encode("cp932"))
        assert [a.name for a in store.all()] == ["残る"]

    def test_names_that_fold_to_one_file_are_kept_apart(self, tmp_path: Path) -> None:
        # 使えない文字を置き換えると同じ名前になり、Windows は大文字と小文字も区別しない
        # 同じファイルへ書くと、後から保存した方が前の物を黙って消す
        store = AliasStore(tmp_path)
        names = ["赤:文字", "赤?文字", "Abc", "abc", "ABC"]
        paths = {store.save(Alias.of(name, _styled_text())).name.casefold() for name in names}
        assert len(paths) == len(names)
        assert sorted(a.name for a in store.all()) == sorted(names)

    @pytest.mark.parametrize("name", ["CON", "nul", "Com1", "LPT1", "aux.txt"])
    def test_windows_device_names_still_save(self, tmp_path: Path, name: str) -> None:
        # CON.smea は Windows が機器として扱い、書き込みが失敗する
        store = AliasStore(tmp_path)
        path = store.save(Alias.of(name, _styled_text()))
        assert path.stem.split(".")[0].upper() not in {"CON", "NUL", "COM1", "LPT1", "AUX"}
        assert [a.name for a in store.all()] == [name]
        assert store.exists(name)

    def test_names_that_windows_rejects_still_save(self, tmp_path: Path) -> None:
        # ファイル名に使えない文字で書き込みが失敗すると、その名前ではいつまでも保存できない
        store = AliasStore(tmp_path)
        path = store.save(Alias.of('見出し: "赤"?', _styled_text()))
        assert path.parent == tmp_path
        assert [a.name for a in store.all()] == ['見出し: "赤"?']
