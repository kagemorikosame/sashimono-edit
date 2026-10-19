"""トラック（レイヤー）の名前を変える命令（利用者の要望）

名前が付けられないと、レイヤーが 10 本を超えた作品で「どれがナレーションか」を
番号で覚えることになる 名前は保存に残り、取り消せ、空にすると既定の名前へ戻る
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sashimono.core.commands import Document, RenameTrack
from sashimono.core.io.serialize import load_project, save_project
from sashimono.core.model import LayerMode, Project, ProjectSettings, Track, TrackKind


def _layers(*names: str) -> Project:
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    tracks = tuple(Track(TrackKind.MIXED, name) for name in names)
    return base.with_timeline(replace(base.timeline, tracks=tracks))


def _names(project: Project) -> list[str]:
    return [t.name for t in project.timeline.tracks]


def test_a_name_is_set_and_trimmed() -> None:
    # 前後の空白を残すと、見出しで名前が右へずれ、同じ名前を探しても当たらない
    project = _layers("レイヤー 1", "レイヤー 2")
    second = project.timeline.tracks[1]
    renamed = RenameTrack(second.id, "  ナレーション ").apply(project)
    assert _names(renamed) == ["レイヤー 1", "ナレーション"]


def test_an_empty_name_goes_back_to_the_default() -> None:
    # 空の名前を残すと見出しに種類の名前しか出ず、どのレイヤーかを並びで数えることになる
    project = _layers("レイヤー 1", "BGM", "声")
    third = project.timeline.tracks[2]
    assert _names(RenameTrack(third.id, "   ").apply(project))[2] == "レイヤー 3"


def test_the_default_skips_a_name_taken_by_another_track() -> None:
    # 既定の名前がほかのトラックの名前と重なると、同じ名前が 2 本並ぶ
    project = _layers("レイヤー 2", "BGM")
    first = project.timeline.tracks[0]
    renamed = RenameTrack(project.timeline.tracks[1].id, "").apply(project)
    assert _names(renamed) == ["レイヤー 2", "レイヤー 3"]
    assert _names(RenameTrack(first.id, "").apply(project))[0] == "レイヤー 1"


def test_separated_tracks_get_their_own_default() -> None:
    # 分ける方式の映像トラックに「レイヤー」の名前が付くと、種類を取り違える
    base = Project.create()
    tracks = (Track(TrackKind.VIDEO, "背景"), Track(TrackKind.AUDIO, "BGM"))
    project = base.with_timeline(replace(base.timeline, tracks=tracks))
    renamed = RenameTrack(tracks[1].id, "").apply(RenameTrack(tracks[0].id, "").apply(project))
    assert _names(renamed) == ["V1", "A1"]


def test_the_same_name_changes_nothing() -> None:
    # 何も変えていないのに取り消しの段が積まれると、戻しても何も起きない段が増える
    project = _layers("レイヤー 1")
    track = project.timeline.tracks[0]
    assert RenameTrack(track.id, "レイヤー 1").apply(project) is project


def test_an_unknown_track_is_refused() -> None:
    # 消えたトラックを指したとき KeyError 以外で失敗すると、断りの知らせにならず編集の途中で落ちる
    from sashimono.core.model import new_track_id

    with pytest.raises(KeyError):
        RenameTrack(new_track_id(), "名前").apply(_layers("レイヤー 1"))


def test_it_can_be_undone() -> None:
    # 取り消せないと、打ち間違えた名前を元の名前を思い出して打ち直すことになる
    document = Document(_layers("レイヤー 1"))
    track = document.project.timeline.tracks[0]
    document.execute(RenameTrack(track.id, "テロップ"))
    assert _names(document.project) == ["テロップ"]
    document.undo()
    assert _names(document.project) == ["レイヤー 1"]


def test_the_name_survives_saving(tmp_path: Path) -> None:
    # 保存に残らないと、開き直すたびに名前を付け直すことになる
    project = _layers("レイヤー 1", "レイヤー 2")
    track = project.timeline.tracks[1]
    renamed = RenameTrack(track.id, "ゲームの音").apply(project)
    path = tmp_path / "名前.sme"
    save_project(renamed, path)
    assert _names(load_project(path)) == ["レイヤー 1", "ゲームの音"]
