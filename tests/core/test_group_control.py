"""グループ制御がどのクリップを受け持つか（利用者の要望 AviUtl の拡張編集のグループ制御）

受け持つのは、置いたトラックより手前（並びの後ろ 混合の方式ではレイヤーの番号が大きい側）の
対象レイヤー数ぶんのトラックの、同じ時刻にあるクリップ 0 なら手前のすべて
数えるのは描かないトラックも含めた並び（AviUtl は非表示のレイヤーも番号として数える）
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sashimono.core.io.serialize import load_project, save_project
from sashimono.core.model import (
    GROUP_KIND,
    GROUP_LAYERS,
    Clip,
    GeneratedSource,
    LayerMode,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    controlling_groups,
    group_layers,
)
from sashimono.effects.sources import GROUP, TEXT


def _group(start: int = 0, duration: int = 60, layers: int = 1) -> Clip:
    return Clip(
        timeline_start=start,
        duration=duration,
        source=GeneratedSource(kind=GROUP_KIND, params={GROUP_LAYERS: layers}),
    )


def _text(start: int = 0, duration: int = 60) -> Clip:
    return Clip(timeline_start=start, duration=duration, source=TEXT.create())


def _layers(*clips: Clip | None) -> tuple[Track, ...]:
    return tuple(
        Track(TrackKind.MIXED, f"レイヤー {n + 1}", (clip,) if clip is not None else ())
        for n, clip in enumerate(clips)
    )


def _ids(found: list[tuple[Track, Clip]]) -> list[str]:
    return [clip.id for _, clip in found]


def test_one_layer_in_front_by_default() -> None:
    # AviUtl の既定の対象レイヤー数は 1 すぐ手前の 1 本だけを動かす
    group = GROUP.create()
    assert group_layers(Clip(timeline_start=0, duration=1, source=group)) == 1
    tracks = _layers(g := _group(), _text(), _text())
    assert _ids(controlling_groups(tracks, tracks[1].id, 10)) == [g.id]
    assert controlling_groups(tracks, tracks[2].id, 10) == []
    # 奥（並びの前）は受け持たない 逆にすると、背景までグループと一緒に動く
    assert controlling_groups(tracks, tracks[0].id, 10) == []


def test_zero_means_every_layer_in_front() -> None:
    # AviUtl で対象を 0（手前の全部）にしたグループが、読み込むと何も動かさないか一部しか動かさない
    tracks = _layers(g := _group(layers=0), _text(), _text(), _text())
    assert all(_ids(controlling_groups(tracks, t.id, 5)) == [g.id] for t in tracks[1:])


def test_only_while_the_group_is_there() -> None:
    # 時間が重ならない所まで動かすと、グループの後ろに置いた物まで動く
    tracks = _layers(_group(start=0, duration=30), _text(start=0, duration=90))
    assert controlling_groups(tracks, tracks[1].id, 29) != []
    assert controlling_groups(tracks, tracks[1].id, 30) == []


def test_a_disabled_group_does_nothing() -> None:
    # 見比べようとグループを無効にしても、手前の物が動いたままで外したときの絵が見られない
    tracks = _layers(replace(_group(), enabled=False), _text())
    assert controlling_groups(tracks, tracks[1].id, 0) == []


def test_empty_layers_count_too() -> None:
    # 間に空いたレイヤーがあっても本数に数える（AviUtl の番号の数え方）
    tracks = _layers(_group(layers=2), None, _text(), _text())
    assert controlling_groups(tracks, tracks[2].id, 0) != []
    assert controlling_groups(tracks, tracks[3].id, 0) == []


def test_nested_groups_come_nearest_first() -> None:
    # 内側（近い）グループで動かした物を外側がさらに動かす 並びが逆だと入れ子の動きが入れ替わる
    tracks = _layers(outer := _group(layers=0), inner := _group(layers=1), _text())
    assert _ids(controlling_groups(tracks, tracks[2].id, 0)) == [inner.id, outer.id]


def test_separated_video_tracks_work_the_same_way() -> None:
    # 分ける方式の映像トラックでも使える（並びの後ろ＝画面では上の V が手前）
    tracks = (
        Track(TrackKind.VIDEO, "V1", (g := _group(),)),
        Track(TrackKind.VIDEO, "V2", (_text(),)),
    )
    assert _ids(controlling_groups(tracks, tracks[1].id, 0)) == [g.id]


def test_it_survives_saving(tmp_path: Path) -> None:
    # 保存に残らないと、開き直すとグループ制御がただの空の物になる
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    project = base.with_timeline(replace(base.timeline, tracks=_layers(_group(layers=3), _text())))
    path = tmp_path / "グループ.sme"
    save_project(project, path)
    loaded = load_project(path).timeline.tracks[0].clips[0]
    assert loaded.is_group
    assert group_layers(loaded) == 3


def test_as_one_is_off_by_default_and_survives_saving(tmp_path: Path) -> None:
    # 既定は 1 本ずつに掛ける（今までの動き） 入れた状態は保存に残る
    from sashimono.core.model import GROUP_AS_ONE, group_as_one, group_reaches

    assert not group_as_one(Clip(timeline_start=0, duration=1, source=GROUP.create()))
    group = _group(layers=2)
    assert group.source is not None
    group = replace(group, source=group.source.with_param(GROUP_AS_ONE, True))
    base = Project.create(ProjectSettings(layer_mode=LayerMode.MIXED))
    tracks = _layers(group, _text(), _text(), _text())
    project = base.with_timeline(replace(base.timeline, tracks=tracks))
    assert [group_reaches(tracks, tracks[0].id, group, t.id) for t in tracks] == [
        False,
        True,
        True,
        False,
    ]
    path = tmp_path / "まとめる.sme"
    save_project(project, path)
    assert group_as_one(load_project(path).timeline.tracks[0].clips[0])
