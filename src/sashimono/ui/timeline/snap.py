"""タイムラインの磁石（吸着） クリップを動かす・端を伸び縮みさせる・置くときに近くへ吸い付く

吸い付く先は利用者の決定で 3 種類
- ほかのクリップの頭と終わり（どのトラックの物も 動かしている物は除く）
- 再生位置
- キーフレームのコマと、書き出し範囲の端

目盛りで再生ヘッドを動かすときも同じ先へ吸い付く（Issue #278） ただし再生位置そのものは
先に入れない 自分の位置へ吸い付くと、動かし始めた所から離れられない どの先へ吸い付くかと、
Shift で入れるか外すかは好みが分かれるので設定に持つ（:class:`PlayheadSnap`）

画面（ウィジェット）から切り離してある 窓を作らずに試験で押さえるため 吸い付く距離は
画面の画素で決め、フレームへ直すのは呼ぶ側の倍率で行う（拡大しても縮小しても、指で
感じる吸い付きの強さを変えない）
"""

from __future__ import annotations

import bisect
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from sashimono.core.model import ClipId, Project
from sashimono.ui.timeline.keyframes import keyframe_frames

# 吸い付き方の値は設定（workspace）が持つ 設定の側からこのパッケージを読むと、パッケージの
# 頭がタイムラインの画面を読み、画面がまた設定を読んで、読み終わる前の設定を引いて落ちる
from sashimono.ui.workspace import PLAYHEAD_SNAP_SHIFT, Preferences

__all__ = [
    "DEFAULT_SNAP_DISTANCE",
    "PlayheadSnap",
    "nearest_snap",
    "snap_targets",
]

#: 吸い付く距離の既定（画面の画素） 小さいと吸い付いたことに気付けず、大きいと
#: 1 コマずらしたいときに隣のクリップの端へ引き戻される
DEFAULT_SNAP_DISTANCE = 8


@dataclass(frozen=True, slots=True)
class PlayheadSnap:
    """目盛りで再生ヘッドを動かすときの吸い付き方と、吸い付く先

    既定はクリップの磁石と同じ先から再生位置を除いた物 目印（マーカー）は既定で入れない
    いまのタイムラインは目印を描かないので、見えない所へ吸い付くと、知らない人には
    何も無い所で引っ掛かったように見える
    """

    mode: str = PLAYHEAD_SNAP_SHIFT
    clip_edges: bool = True
    keyframes: bool = True
    work_area: bool = True
    markers: bool = False

    @classmethod
    def from_preferences(cls, preferences: Preferences) -> PlayheadSnap:
        """設定から作る 起動のときと設定を変えたときの 2 か所で同じ物を作るため"""
        return cls(
            mode=preferences.playhead_snap,
            clip_edges=preferences.playhead_snap_clips,
            keyframes=preferences.playhead_snap_keyframes,
            work_area=preferences.playhead_snap_work_area,
            markers=preferences.playhead_snap_markers,
        )


def snap_targets(
    project: Project,
    playhead: int | None,
    *,
    exclude: Collection[ClipId] = (),
    clip_edges: bool = True,
    keyframes: bool = True,
    work_area: bool = True,
    markers: bool = False,
) -> list[int]:
    """吸い付く先のフレーム（小さい順 重なりは 1 つ）

    ``exclude`` は動かしているクリップ 自分の端やキーフレームへ吸い付くと、動かした量が
    0 に引き戻されて動かせない ``playhead`` が ``None`` なら再生位置を入れない（再生ヘッド
    そのものを動かすとき） 残りの旗は吸い付く先の種類ごとの入り切り
    """
    frames = set() if playhead is None else {playhead}
    timeline = project.timeline
    for track in timeline.tracks:
        for clip in track.clips:
            if clip.id in exclude:
                continue
            if clip_edges:
                frames.add(clip.timeline_start)
                frames.add(clip.timeline_end)
            if keyframes:
                frames.update(clip.timeline_start + local for local in keyframe_frames(clip))
    if work_area and timeline.work_area is not None:
        frames.update(timeline.work_area)
    if markers:
        frames.update(marker.frame for marker in timeline.markers)
    return sorted(frames)


def nearest_snap(
    edges: Sequence[int], targets: Sequence[int], reach: float
) -> tuple[int, int] | None:
    """``edges``（動かしている物の端）のどれかが ``reach`` フレーム以内に近づいた吸い付く先

    返すのは ``(ずらす量, 吸い付いた先)`` 一番近い物 無ければ ``None``
    ``targets`` は小さい順（:func:`snap_targets`）
    """
    best: tuple[int, int] | None = None
    for edge in edges:
        at = bisect.bisect_left(targets, edge)
        for index in (at - 1, at):
            if not 0 <= index < len(targets):
                continue
            shift = targets[index] - edge
            if abs(shift) <= reach and (best is None or abs(shift) < abs(best[0])):
                best = (shift, targets[index])
    return best
