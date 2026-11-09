"""持ち出し用パッケージに入っている範囲の外を読む編集を断る（#445）

パッケージは素材の使っている範囲だけを入れる（:attr:`MediaItem.packed`） 開いた先で範囲の
外へトリムを伸ばす・速度を上げる・素材の中の位置をずらすと、そこには絵も音も無い
断る所を :meth:`Document.execute` の 1 か所にまとめ、画面の操作も AI の道具も同じ理由で断る
コマンドごとに確かめると、新しい編集のコマンドを足すたびに確かめ忘れる
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

from sashimono.core.model import Clip, ClipId, MediaItem, PackedSpan, Project, Timeline

__all__ = ["outside_packed", "packed_refusal", "packed_text", "read_window"]


def _timelines(project: Project) -> Iterator[Timeline]:
    yield project.timeline
    for scene in project.scenes:
        yield scene.timeline


def read_window(clip: Clip, timeline: Timeline) -> tuple[Fraction, Fraction]:
    """クリップが素材のどこからどこまでを読むか（秒）

    絵を止める時刻（``hold_at``）が頭より前なら、そこも読む 音は頭から長さの分を読み続ける
    （パッケージの書き出しが入れる範囲 :func:`~sashimono.engine.package.plan.media_spans` と同じ）
    """
    start = clip.source_in
    if clip.hold_at is not None:
        start = min(start, clip.hold_at)
    return start, clip.source_out(timeline.rate)


def outside_packed(project: Project) -> dict[ClipId, tuple[MediaItem, Fraction, Fraction]]:
    """入っている範囲の外を読むクリップ（素材と読む区間）"""
    packed = {item.id: item for item in project.media if item.packed}
    if not packed:
        return {}
    found: dict[ClipId, tuple[MediaItem, Fraction, Fraction]] = {}
    for timeline in _timelines(project):
        for track in timeline.tracks:
            for clip in track.clips:
                item = packed.get(clip.media_id) if clip.media_id is not None else None
                if item is None:
                    continue
                start, end = read_window(clip, timeline)
                if not item.covers(start, end):
                    found[clip.id] = (item, start, end)
    return found


def packed_text(spans: tuple[PackedSpan, ...]) -> str:
    """入っている範囲を人が読む形にする（``12.0〜34.5 秒``）"""
    return "、".join(f"{float(span.start):.1f}〜{float(span.end):.1f} 秒" for span in spans)


def packed_refusal(before: Project, after: Project) -> str | None:
    """``before`` から ``after`` への変更が、新しく範囲の外を読ませるなら断りの文

    前から外を読んでいたクリップ（前の本体で編集した作品など）は、読む区間を外へ広げない限り
    通す 外にあるだけで全部の編集を断ると、関係の無いクリップも動かせなくなる
    """
    now = outside_packed(after)
    if not now:
        return None
    earlier = outside_packed(before)
    for clip_id, (item, start, end) in now.items():
        old = earlier.get(clip_id)
        if old is not None and old[1] <= start and end <= old[2]:
            continue
        return (
            f"「{item.name}」は持ち出し用パッケージに入っている範囲（{packed_text(item.packed)}）"
            f"の外を読むことになるので、できません（読もうとした所 {float(start):.1f}〜"
            f"{float(end):.1f} 秒） 範囲の外も使うには、元の素材を探し直して付け替えてください"
        )
    return None
