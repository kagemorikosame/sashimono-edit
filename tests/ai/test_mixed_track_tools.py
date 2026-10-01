"""AI から混合トラック（YMM4 型のレイヤー）を足して見る

UI にだけ入口を付けると「画面ではできるのに AI に頼むとできない」になる
"""

from __future__ import annotations

import pytest

from sashimono.ai.host import ToolError
from sashimono.ai.session import SYSTEM_PROMPT, system_prompt
from sashimono.core.commands import AddClip, RemoveTrack, SetTrackState
from sashimono.core.model import (
    Clip,
    LayerMode,
    MediaItem,
    Project,
    ProjectSettings,
    TrackKind,
)
from sashimono.core.timebase import FrameRate
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run


def test_a_layer_is_added_with_a_ymm4_name(host: FakeHost) -> None:
    # video か audio しか受けないと、AI は混合の作品にレイヤーを足せない
    result = run(host, "add_track", kind="mixed")
    track = host.document.project.timeline.tracks[-1]
    assert track.kind is TrackKind.MIXED
    assert result["name"] == track.name == "レイヤー 1"


def test_a_removed_layer_does_not_cause_a_twin(host: FakeHost) -> None:
    # 本数だけで数えると、消した後にもう 1 本の「レイヤー 2」ができる
    run(host, "add_track", kind="mixed")
    run(host, "add_track", kind="mixed")
    first = next(t for t in host.document.project.timeline.tracks if t.name == "レイヤー 1")
    host.apply_commands([RemoveTrack(first.id)], "消す")
    run(host, "add_track", kind="mixed")
    names = [t.name for t in host.document.project.timeline.tracks if t.kind is TrackKind.MIXED]
    assert sorted(names) == ["レイヤー 2", "レイヤー 3"]


def test_the_list_says_mixed(host: FakeHost) -> None:
    run(host, "add_track", kind="mixed")
    kinds = [entry["kind"] for entry in run(host, "list_tracks")["tracks"]]
    assert kinds[-1] == "mixed"


def test_an_unknown_kind_names_the_choices(host: FakeHost) -> None:
    with pytest.raises(ToolError, match="mixed"):
        run(host, "add_track", kind="effect")


def test_layer_clips_show_what_they_play(host: FakeHost) -> None:
    # 絵と音のどちらを出すかが見えないと、AI は音の消えた動画を直せない
    run(host, "add_track", kind="mixed")
    project = host.document.project
    layer = project.timeline.tracks[-1]
    media = project.media[0]
    clip = Clip(0, 30, media_id=media.id, audio_stream=1, show_picture=False)
    host.apply_commands([AddClip(layer.id, clip)], "置く")
    listed = next(c for c in run(host, "list_clips")["clips"] if c["track_kind"] == "mixed")
    assert listed["audio_stream"] == 1
    assert listed["show_picture"] is False
    others = [c for c in run(host, "list_clips")["clips"] if c["track_kind"] != "mixed"]
    assert all("audio_stream" not in c for c in others)


def _mixed_host(video_media: MediaItem) -> FakeHost:
    """混合の方式で、素材を 1 本読み込んだ（まだ置いていない）プロジェクト"""
    settings = ProjectSettings(frame_rate=FrameRate(30), layer_mode=LayerMode.MIXED)
    return FakeHost(Project.create(settings, media=(video_media,)))


class TestMixedMode:
    """混合の方式のプロジェクトでは、AI の操作も方式に従う"""

    def test_add_track_defaults_to_a_layer(self, video_media: MediaItem) -> None:
        # 省いたときに映像トラックを足すと、混合の作品に AI の置いた物だけ別の種類で並ぶ
        host = _mixed_host(video_media)
        result = run(host, "add_track")
        track = host.document.project.timeline.tracks[-1]
        assert track.kind is TrackKind.MIXED
        assert result["name"] == "レイヤー 1"

    def test_add_track_still_defaults_to_video_when_separated(self, host: FakeHost) -> None:
        # 分ける方式（既定）の既定まで変わると、今までの頼み方で別の種類のトラックが増える
        run(host, "add_track")
        assert host.document.project.timeline.tracks[-1].kind is TrackKind.VIDEO

    def test_add_track_joins_the_solo(self, video_media: MediaItem) -> None:
        # ソロの間に足したレイヤーにソロが無いと、AI がそこへ置いた物が出ない
        host = _mixed_host(video_media)
        run(host, "add_track")
        layer = host.document.project.timeline.tracks[0]
        host.apply_commands([SetTrackState(layer.id, solo=True)], "ソロ")
        run(host, "add_track", name="テロップ")
        added = host.document.project.timeline.tracks[-1]
        assert added.name == "テロップ"
        assert added.solo

    def test_place_media_splits_onto_layers(self, video_media: MediaItem) -> None:
        # 本人の設定（既定は分ける）を見ずに 1 本で置くと、画面から置いた物と形が変わる
        # 分けても音声トラックは作らない（混合の作品に別の種類のトラックが増える）
        host = _mixed_host(video_media)
        run(host, "place_media", media_id=str(video_media.id))
        picture_layer, sound_layer = host.document.project.timeline.tracks
        assert picture_layer.kind is sound_layer.kind is TrackKind.MIXED
        (picture,), (sound,) = picture_layer.clips, sound_layer.clips
        assert picture.audio_stream is None
        assert sound.audio_stream == video_media.audio_streams[0].index
        assert picture.link_group is not None
        assert picture.link_group == sound.link_group

    def test_place_media_makes_one_clip_when_told(self, video_media: MediaItem) -> None:
        # 分けない設定を選んだ人の作品で AI だけ分けると、頼み方でレイヤーの数が変わる
        host = _mixed_host(video_media)
        host.split_audio = False
        run(host, "place_media", media_id=str(video_media.id))
        (layer,) = host.document.project.timeline.tracks
        assert layer.kind is TrackKind.MIXED
        (clip,) = layer.clips
        assert clip.audio_stream == video_media.audio_streams[0].index
        assert clip.link_group is None

    def test_get_project_tells_the_mode(self, video_media: MediaItem) -> None:
        # 方式が分からないと、AI は混合の作品でもリンクした音声クリップを探しに行く
        assert run(_mixed_host(video_media), "get_project")["layer_mode"] == "mixed"


class TestPrompt:
    def test_the_prompt_explains_the_mode(self) -> None:
        # 分ける方式の説明のまま混合の作品を触らせると、無い組の片方を探し回る
        separated = system_prompt(LayerMode.SEPARATED)
        mixed = system_prompt(LayerMode.MIXED)
        assert "分ける方式です" in separated
        assert "混合の方式です" in mixed
        assert "{linked_clips}" not in mixed
        assert system_prompt() == SYSTEM_PROMPT == separated

    def test_placed_clips_are_checked_one_by_one(self) -> None:
        # 方式が決めるのはこれから置く物だけ 方式で置いてある物の形まで言い切ると、
        # 方式を途中で変えた作品で、AI が組の片方やレイヤーの 1 本を見落とす
        for mode in LayerMode.ALL:
            prompt = system_prompt(mode)
            assert "link_group" in prompt
            assert "track_kind が mixed" in prompt

    def test_list_clips_shows_the_link(self, video_media: MediaItem) -> None:
        # 組が見えないと、プロンプトの言う確かめ方ができない
        host = FakeHost(
            Project.create(ProjectSettings(frame_rate=FrameRate(30)), media=(video_media,))
        )
        run(host, "place_media", media_id=str(video_media.id))
        video, audio = run(host, "list_clips")["clips"]
        assert video["link_group"] is not None
        assert video["link_group"] == audio["link_group"]
