"""書き出しのビットレートの既定を素材に合わせる（#414 ``encode/choices.py`` の ``BitrateChoice``）

通しの確かめで、12 Mbps 決め打ちの既定は 1〜2 Mbps の素材を 7 倍の大きさにした
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Scene,
    SceneId,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.engine.encode import (
    DEFAULT_BITRATE_MBPS,
    MATCH_FLOOR_MBPS,
    BitrateChoice,
    choices,
    export_target,
    matched_bitrate_mbps,
)
from tests.ai.conftest import make_loaded

#: 10 秒の素材が 2 Mbps になる大きさ（バイト）
TWO_MBPS = 2_000_000 * 10 // 8


def _sizes(**sizes: int) -> Callable[[Path], int]:
    def size_of(path: Path) -> int:
        found = sizes.get(path.stem)
        if found is None:
            raise FileNotFoundError(path)
        return found

    return size_of


def test_a_low_bitrate_source_is_matched(video_media: MediaItem) -> None:
    """壊れると、2 Mbps の素材を 12 Mbps で書き出し、見た目の変わらないファイルが 6 倍になる"""
    project = make_loaded(replace(video_media, path=Path("C:/素材/low.mp4")), _no_words())
    size_of = _sizes(low=TWO_MBPS)
    # 2 Mbps の 1.25 倍を切り上げて 3 Mbps
    assert matched_bitrate_mbps(project, size_of=size_of) == 3
    assert BitrateChoice().for_project(project, size_of=size_of) == 3


def test_the_upper_value_stays_a_cap(video_media: MediaItem) -> None:
    """壊れると、カメラの 50 Mbps の素材に合わせて、上限を決めた人の書き出しが巨大になる"""
    project = make_loaded(replace(video_media, path=Path("C:/素材/high.mp4")), _no_words())
    size_of = _sizes(high=TWO_MBPS * 25)
    assert BitrateChoice(mbps=12).for_project(project, size_of=size_of) == 12


def test_a_bigger_output_scales_up(video_media: MediaItem) -> None:
    """作品が素材より大きければ画素の数の比で上げる 上げないと拡大した絵が崩れる"""
    loaded = make_loaded(replace(video_media, path=Path("C:/素材/low.mp4")), _no_words())
    project = replace(loaded, settings=replace(loaded.settings, width=3840, height=2160))
    assert matched_bitrate_mbps(project, size_of=_sizes(low=TWO_MBPS)) == 10


def test_a_tiny_source_keeps_the_floor(video_media: MediaItem) -> None:
    """画面の録画の 0.3 Mbps に合わせると、重ねた文字の縁が崩れる"""
    project = make_loaded(replace(video_media, path=Path("C:/素材/tiny.mp4")), _no_words())
    assert matched_bitrate_mbps(project, size_of=_sizes(tiny=TWO_MBPS // 7)) == MATCH_FLOOR_MBPS


def test_a_missing_file_falls_back(video_media: MediaItem) -> None:
    """外したドライブの素材で止まらず、決めた値で書き出せる"""
    project = make_loaded(video_media, _no_words())
    assert matched_bitrate_mbps(project, size_of=_sizes()) is None
    assert BitrateChoice().for_project(project, size_of=_sizes()) == DEFAULT_BITRATE_MBPS


def test_switched_off_uses_the_fixed_value(video_media: MediaItem) -> None:
    """切った人は、素材に関係なく決めた値で書き出す"""
    project = make_loaded(replace(video_media, path=Path("C:/素材/low.mp4")), _no_words())
    size_of = _sizes(low=TWO_MBPS)
    assert BitrateChoice(auto=False, mbps=20).for_project(project, size_of=size_of) == 20


def test_a_project_without_video_uses_the_fixed_value() -> None:
    """文字だけの作品は合わせる素材が無い 0 や下限にすると図形の縁が崩れる"""
    project = Project.create(ProjectSettings())
    assert BitrateChoice().for_project(project) == DEFAULT_BITRATE_MBPS


def _no_words() -> Transcript:
    return Transcript(segments=(), language="ja")


def test_the_duration_is_used(video_media: MediaItem) -> None:
    """長さで割らないと、長い素材ほど高いビットレートと読む"""
    longer = replace(video_media, path=Path("C:/素材/low.mp4"), duration=Fraction(20))
    project = make_loaded(longer, _no_words())
    assert matched_bitrate_mbps(project, size_of=_sizes(low=TWO_MBPS)) == 2


def test_uncompressed_audio_is_not_read_as_picture(video_media: MediaItem) -> None:
    """録画機の 8ch PCM は音だけで 9 Mbps 映像の量と読むと、4 Mbps の素材まで上限で書き出す"""
    pcm = AudioStreamInfo(
        index=1, sample_rate=48000, channels=8, time_base=Fraction(1, 48000), codec="pcm_s24le"
    )
    media = replace(video_media, path=Path("C:/素材/rec.mov"), audio_streams=(pcm,))
    project = make_loaded(media, _no_words())
    size = (4_000_000 + 48000 * 8 * 24) * 10 // 8
    # 映像の 4 Mbps の 1.25 倍で 5 Mbps
    assert matched_bitrate_mbps(project, size_of=_sizes(rec=size)) == 5


def _with_scene(video_media: MediaItem, *, placed: bool) -> tuple[Project, SceneId]:
    """低い素材をメインに置き、高い素材だけのシーンを作った作品

    ``placed`` ならシーンもメインに置く
    """
    project = make_loaded(replace(video_media, path=Path("C:/素材/low.mp4")), _no_words())
    high = MediaItem(
        path=Path("C:/素材/high.mp4"),
        duration=video_media.duration,
        video_streams=video_media.video_streams,
    )
    inner = Track(kind=TrackKind.VIDEO, name="V1", clips=(Clip(0, 300, media_id=high.id),))
    scene = Scene(name="山場", timeline=replace(project.timeline, tracks=(inner,)))
    project = replace(project, media=(*project.media, high), scenes=(scene,))
    if placed:
        outer = Track(kind=TrackKind.VIDEO, name="V2", clips=(Clip(0, 300, scene_id=scene.id),))
        project = project.with_timeline(
            replace(project.timeline, tracks=(*project.timeline.tracks, outer))
        )
    return project, scene.id


def test_only_the_exported_scene_counts(video_media: MediaItem) -> None:
    """選んだ物の外の高い素材に引かれると、低い素材の場面だけでも大きなファイルになる"""
    size_of = _sizes(low=TWO_MBPS, high=TWO_MBPS * 4)
    project, scene_id = _with_scene(video_media, placed=False)
    # メインには低い素材だけ 置いていないシーンの素材は書き出さない
    assert matched_bitrate_mbps(project, size_of=size_of) == 3
    # シーンを書き出すなら、その中の高い素材に合わせる
    assert matched_bitrate_mbps(export_target(project, scene_id), size_of=size_of) == 10


def test_a_nested_scene_counts(video_media: MediaItem) -> None:
    """メインに置いたシーンの中の素材も書き出す 数えないと、その場面が崩れる"""
    project, _ = _with_scene(video_media, placed=True)
    size_of = _sizes(low=TWO_MBPS, high=TWO_MBPS * 4)
    assert matched_bitrate_mbps(project, size_of=size_of) == 10


def test_mu_law_audio_is_eight_bits(video_media: MediaItem) -> None:
    """μ-law は 8 ビット 16 ビットとして引くと、4 Mbps の素材を 3 Mbps で書き出す（CodeRabbit）"""
    pcm = AudioStreamInfo(
        index=1, sample_rate=48000, channels=8, time_base=Fraction(1, 48000), codec="pcm_mulaw"
    )
    media = replace(video_media, path=Path("C:/素材/rec.mov"), audio_streams=(pcm,))
    project = make_loaded(media, _no_words())
    size = (4_000_000 + 48000 * 8 * 8) * 10 // 8
    assert matched_bitrate_mbps(project, size_of=_sizes(rec=size)) == 5


def test_too_much_audio_to_subtract_keeps_the_whole(video_media: MediaItem) -> None:
    """音が素材より短い PCM を全長で引くと映像の分が残らない そこで素材を外すと低い値になる"""
    pcm = AudioStreamInfo(
        index=1, sample_rate=48000, channels=8, time_base=Fraction(1, 48000), codec="pcm_s24le"
    )
    media = replace(video_media, path=Path("C:/素材/rec.mov"), audio_streams=(pcm,))
    project = make_loaded(media, _no_words())
    # 10 秒の 4 Mbps の映像と、1 秒だけの 8ch PCM
    size = (4_000_000 * 10 + 48000 * 8 * 24) // 8
    # 引かずに 4.9 Mbps と見て、1.25 倍を切り上げて 7 Mbps
    assert matched_bitrate_mbps(project, size_of=_sizes(rec=size)) == 7


def test_a_hidden_scene_does_not_count(video_media: MediaItem) -> None:
    """消音・非表示のトラックのシーンは映らない 数えると要らない大きさで書き出す（CodeRabbit）"""
    project, _ = _with_scene(video_media, placed=True)
    tracks = tuple(
        replace(track, muted=True) if track.name == "V2" else track
        for track in project.timeline.tracks
    )
    project = project.with_timeline(replace(project.timeline, tracks=tracks))
    size_of = _sizes(low=TWO_MBPS, high=TWO_MBPS * 4)
    assert matched_bitrate_mbps(project, size_of=size_of) == 3


def test_a_disabled_clip_does_not_count(video_media: MediaItem) -> None:
    # 壊れると、無効にして映らないシーンの高い素材に引かれて、大きなファイルで書き出す
    project, _ = _with_scene(video_media, placed=True)
    outer = project.timeline.tracks[-1]
    off = replace(outer, clips=tuple(replace(clip, enabled=False) for clip in outer.clips))
    project = project.with_timeline(
        replace(project.timeline, tracks=(*project.timeline.tracks[:-1], off))
    )
    size_of = _sizes(low=TWO_MBPS, high=TWO_MBPS * 4)
    assert matched_bitrate_mbps(project, size_of=size_of) == 3


def test_the_chosen_video_stream_is_used(video_media: MediaItem) -> None:
    """クリップが描く道で見る 1 本目で見ると、別の大きさの道を描くクリップの値がずれる"""
    (first,) = video_media.video_streams
    big = replace(first, index=5, width=first.width * 2, height=first.height * 2)
    media = replace(video_media, path=Path("C:/素材/two.mp4"), video_streams=(first, big))
    project = make_loaded(media, _no_words())
    size_of = _sizes(two=TWO_MBPS * 8)
    # 1 本目なら 16 Mbps の 1.25 倍で 20 Mbps
    assert matched_bitrate_mbps(project, size_of=size_of) == 20
    track = next(t for t in project.timeline.tracks if t.kind is TrackKind.VIDEO)
    chosen = replace(track, clips=tuple(replace(c, stream_index=5) for c in track.clips))
    project = project.with_timeline(
        replace(
            project.timeline,
            tracks=tuple(chosen if t.id == track.id else t for t in project.timeline.tracks),
        )
    )
    # 4 倍の画素の道を描くので 4 Mbps 相当 1.25 倍で 5 Mbps
    assert matched_bitrate_mbps(project, size_of=size_of) == 5


def test_a_shared_scene_reached_deep_first_is_walked_again(video_media: MediaItem) -> None:
    """深い道で先に着いたシーンを、浅い道からもたどる 止めると、その中の高い素材を数え落とす"""
    project, inner_id = _with_scene(video_media, placed=False)
    inner = project.find_scene(inner_id)
    assert inner is not None
    # 高い素材を入れた「山場」を置いただけのシーン これを深い道と浅い道の両方から置く
    shared = Scene(
        name="共有",
        timeline=replace(
            project.timeline,
            tracks=(
                Track(kind=TrackKind.VIDEO, name="V", clips=(Clip(0, 300, scene_id=inner_id),)),
            ),
        ),
    )
    chain: list[Scene] = []
    below = shared.id
    # 15 段のシーンの底に「共有」を置く 「共有」は深さ 16 で、中の「山場」へは進まない
    for depth in range(15):
        step = Scene(
            name=f"段 {depth}",
            timeline=replace(
                project.timeline,
                tracks=(
                    Track(kind=TrackKind.VIDEO, name="V", clips=(Clip(0, 300, scene_id=below),)),
                ),
            ),
        )
        chain.append(step)
        below = step.id
    # 浅い道は 1 段を挟んで「共有」へ（深さ 2） 鎖を後の段に置くと鎖から先にたどり、
    # 浅い道が「共有」に着くのは鎖の底で先に着いた後になる
    middle = Scene(
        name="中継ぎ",
        timeline=replace(
            project.timeline,
            tracks=(
                Track(kind=TrackKind.VIDEO, name="V", clips=(Clip(0, 300, scene_id=shared.id),)),
            ),
        ),
    )
    near = Track(kind=TrackKind.VIDEO, name="近", clips=(Clip(0, 300, scene_id=middle.id),))
    far = Track(kind=TrackKind.VIDEO, name="遠", clips=(Clip(0, 300, scene_id=below),))
    project = replace(project, scenes=(*project.scenes, shared, middle, *chain))
    project = project.with_timeline(
        replace(project.timeline, tracks=(*project.timeline.tracks, near, far))
    )
    size_of = _sizes(low=TWO_MBPS, high=TWO_MBPS * 4)
    assert matched_bitrate_mbps(project, size_of=size_of) == 10


def test_the_video_length_is_used(video_media: MediaItem) -> None:
    """音の方が長い素材は映像の道の長さで割る

    入れ物の長さで割ると 2 Mbps の映像を 1 Mbps と読んで、低く書き出す（CodeRabbit）
    """
    (stream,) = video_media.video_streams
    media = replace(
        video_media,
        path=Path("C:/素材/long_audio.mp4"),
        duration=Fraction(20),
        video_streams=(replace(stream, end_time=Fraction(10)),),
    )
    project = make_loaded(media, _no_words())
    # 10 秒の映像で 2 Mbps 入れ物の 20 秒で割ると 1 Mbps と読んで下限の 2 Mbps になる
    assert matched_bitrate_mbps(project, size_of=_sizes(long_audio=TWO_MBPS)) == 3


def test_a_scene_reached_twice_at_the_same_depth_is_walked_once(video_media: MediaItem) -> None:
    """同じシーンを各段に 2 つ置いても、段ごとに 1 度だけたどる

    壊れると、16 段で 13 万回たどり、書き出しの窓のビットレートがいつまでも決まらない（CodeRabbit）
    """
    project, inner_id = _with_scene(video_media, placed=False)
    below = inner_id
    scenes: list[Scene] = []
    for depth in range(6):
        clips = (Clip(0, 300, scene_id=below), Clip(300, 300, scene_id=below))
        step = Scene(
            name=f"段 {depth}",
            timeline=replace(
                project.timeline, tracks=(Track(kind=TrackKind.VIDEO, name="V", clips=clips),)
            ),
        )
        scenes.append(step)
        below = step.id
    twice = (Clip(0, 300, scene_id=below), Clip(300, 300, scene_id=below))
    outer = Track(kind=TrackKind.VIDEO, name="外", clips=twice)
    project = replace(project, scenes=(*project.scenes, *scenes))
    project = project.with_timeline(
        replace(project.timeline, tracks=(*project.timeline.tracks, outer))
    )
    # メインと 6 段と「山場」で 8 枚 2 つずつたどると 2 の 7 乗を越える
    assert len(list(choices._timelines(project))) == 8
