"""互換の読み込みを、混合の方式（YMM4・AviUtl と同じレイヤー）で置くときの決まり

テンプレートの棚（:func:`~sashimono.compat.catalog.place`）と ``.exo`` の読み込み
（:func:`~sashimono.compat.aviutl.mapping.map_exo`）が同じ答えを出すよう、ここに置く
片方にだけ書くと、同じ AviUtl のエイリアスが読み込み方によって別のレイヤーへ入る

**レイヤー番号はそのまま混合トラックの並びの番号にする** YMM4 も AviUtl も、番号の
小さいレイヤーが奥で、番号が大きいレイヤーほど手前に描く（YMM4 は 0 始まりなので
写す側で 1 を足してある） こちらの混合トラックもレイヤー 1 が一番奥
（:class:`~sashimono.core.model.Timeline`）なので、向きを裏返さずに並べられる

**元のレイヤーの所が埋まっていれば、全部を一番手前の新しいトラックへ並べる**（#328
2026-10-17 本人の決定） 置く物のどれか 1 つでも、番号の当たるトラックの同じ時刻に
クリップがあるか、そのトラックがロックされていれば、番号を並びに当てるのをやめ、使う
番号の分だけ新しいトラックを末尾（一番手前）へ足して、番号の小さい順に置く 上下の並びは
保つ 1 つだけ外へ逃がすと、テンプレートの中の上下が入れ替わる 断ると、動画の上に字幕の
テンプレートを置くだけで重なって読み込みが止まっていた
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.core.commands import Command
from sashimono.core.commands.layers import new_layer
from sashimono.core.model import MediaItem, Project, Track, TrackKind

__all__ = ["Spans", "crowded", "heard_stream", "in_front", "layer_tracks"]

#: 元のレイヤー番号 → そこへ置く物の時刻の範囲（始まり, 終わり） 埋まっているかを見るのに使う
Spans = Mapping[int, Sequence[tuple[int, int]]]


def crowded(existing: Sequence[Track], spans: Spans | None) -> bool:
    """番号の当たるトラックのどれかが、置く物と重なるかロックされているか

    番号がトラックの数を越える所は、新しく足すので埋まっていない
    """
    if not spans:
        return False
    for layer, ranges in spans.items():
        if not 1 <= layer <= len(existing):
            continue
        track = existing[layer - 1]
        if track.locked and ranges:
            return True
        if any(clip.overlaps(start, end) for clip in track.clips for start, end in ranges):
            return True
    return False


def in_front(layers: Iterable[int], make: Callable[[int], Track]) -> dict[int, Track]:
    """使う番号の分だけ新しいトラックを作り、番号の小さい順に末尾（手前）へ並べる

    間の空いた番号は詰める 一番手前へ足すので、詰めても上下の並びは変わらない
    """
    return {layer: make(layer) for layer in sorted(set(layers))}


def layer_tracks(
    project: Project,
    layers: Iterable[int],
    commands: list[Command],
    *,
    heard_only: Collection[int] = (),
    spans: Spans | None = None,
) -> dict[int, Track]:
    """元のレイヤー番号 → 置く混合トラック 足りない分は足すコマンドを ``commands`` へ積む

    レイヤー n は、今ある混合トラックの n 本目 間が空いていても埋める 詰めると、
    間を空けて重ねた物の前後が入れ替わる（分ける方式の映像トラックと同じ決まり）

    ``heard_only`` は音だけを置くレイヤー 新しく足すときのソロを音の側で決める
    （:func:`~sashimono.core.commands.layers.solo_for_new_track`） 絵の側で決めると、
    絵のトラックをソロにしているだけで、足した効果音のレイヤーが鳴らなくなる

    ``spans`` は置く物の時刻 番号の当たる所が埋まっていれば、全部を一番手前の新しい
    レイヤーへ並べる（:func:`crowded` モジュールの説明）
    """
    existing = [track for track in project.timeline.tracks if track.kind is TrackKind.MIXED]
    if crowded(existing, spans):
        return in_front(
            layers,
            lambda layer: new_layer(project, commands, picture=layer not in heard_only),
        )
    tracks: dict[int, Track] = {}
    for layer in range(1, max(layers, default=0) + 1):
        if layer - 1 < len(existing):
            tracks[layer] = existing[layer - 1]
            continue
        tracks[layer] = new_layer(project, commands, picture=layer not in heard_only)
    return tracks


def heard_stream(media: MediaItem | None, track: int, log: CompatibilityReport) -> int | None:
    """素材の ``track`` 本目の音の、素材の中の番号

    置くクリップの :attr:`~sashimono.core.model.Clip.audio_stream` にする
    音を持たない素材と、素材が見つからないときは ``None``（鳴らさない）

    素材に無い本数を選んでいたら 1 本目を鳴らし、数えて残す 無い番号のまま置くと
    混合トラックは置く時点で断り（:class:`~sashimono.core.commands.AddClip`）、
    同じテンプレートのほかの物まで置けなくなる 本数を選べるのは YMM4 だけ
    """
    if media is None or not media.audio_streams:
        return None
    if 0 <= track < len(media.audio_streams):
        return media.audio_streams[track].index
    log.note_missing("YMM4 の音声トラックの選択（AudioTrackIndex）が素材に無い番号")
    return media.audio_streams[0].index
