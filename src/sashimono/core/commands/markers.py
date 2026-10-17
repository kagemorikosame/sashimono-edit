"""目印（マーカー :class:`~sashimono.core.model.Marker`）を置く・直す・消すコマンド（#330）

目印はタイムラインごとに持つ シーンを開いているときは :class:`InScene` で包まれて、
そのシーンの目印が変わる

目印は位置（フレーム）で見分ける モデルの目印には ID が無く、並びの番号で指すと、
1 つ消しただけで後ろの番号が全部ずれる AI が一覧を読んでから続けて消す・直すとき、
ずれた番号で別の目印を消してしまう 同じフレームに 2 つ置くことは断る
（チャプターにしても同じ時刻の目次が 2 行並ぶだけで、見分ける手段も無くなる）

コマンドが返す目印の並びはいつもフレームの順 ファイルから読んだ並びが乱れていても、
一度でも直せば揃う 順に並んでいれば、一覧もチャプターも並べ直さずに済む
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from sashimono.core.commands.base import Command
from sashimono.core.model import Marker, Project, Timeline

__all__ = [
    "MARKER_LABEL_LIMIT",
    "AddMarkers",
    "RemoveMarkers",
    "UpdateMarker",
    "marker_at",
    "valid_marker_color",
]

#: 名前の長さの上限（文字） YouTube の目次 1 行に収まり、目盛りの上でも読める長さ
#: 長い文を貼ると、目盛りの上で隣の目印の名前に重なり、どちらも読めなくなる
MARKER_LABEL_LIMIT = 100

_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


def valid_marker_color(color: str) -> bool:
    """色が ``#RRGGBB`` か 保存する形を 1 つにしておく ほかの書き方を許すと、
    読み込んだ側（画面や互換の書き出し）が書き方ごとに読み分けることになる"""
    return _COLOR.fullmatch(color) is not None


def marker_at(timeline: Timeline, frame: int) -> Marker | None:
    """``frame`` にある目印 無ければ ``None``"""
    return next((marker for marker in timeline.markers if marker.frame == frame), None)


def _check(marker: Marker, current: Marker | None = None) -> Marker:
    """置く前に確かめる 名前の前後の空白は落とす

    ``current`` は直す前の目印 渡したときは変えた欄だけを確かめる 保存形式は色や位置を
    確かめずに読むので（手で書いた作品・ほかの道具が書いた作品）、変えていない欄まで
    確かめると、読み込んだ目印の名前を変えるたびに必ず断られ、画面からも AI からも直せない
    """
    label = marker.label.strip()
    if (current is None or marker.frame != current.frame) and marker.frame < 0:
        raise ValueError(f"目印の位置が負: {marker.frame}")
    if (current is None or label != current.label) and len(label) > MARKER_LABEL_LIMIT:
        raise ValueError(f"目印の名前が長すぎる（{len(label)} 文字 上限 {MARKER_LABEL_LIMIT}）")
    changed_color = current is None or marker.color != current.color
    if changed_color and not valid_marker_color(marker.color):
        raise ValueError(f"目印の色は #RRGGBB で渡す: {marker.color!r}")
    return replace(marker, label=label)


def _with_markers(project: Project, markers: list[Marker]) -> Project:
    ordered = tuple(sorted(markers, key=lambda marker: marker.frame))
    return project.with_timeline(replace(project.timeline, markers=ordered))


@dataclass(frozen=True, slots=True)
class AddMarkers(Command):
    """目印をまとめて置く まとめるのは、AI が場面の一覧から何十個も置いたとき、
    取り消し 1 回で全部戻せるようにするため 1 つでも置けなければ何も置かない"""

    markers: tuple[Marker, ...]

    @property
    def label(self) -> str:
        return "マーカーを置く"

    def apply(self, project: Project) -> Project:
        if not self.markers:
            raise ValueError("置く目印がありません")
        taken = {marker.frame for marker in project.timeline.markers}
        added: list[Marker] = []
        for marker in self.markers:
            checked = _check(marker)
            if checked.frame in taken:
                raise ValueError(f"{checked.frame} フレームにはもう目印があります")
            taken.add(checked.frame)
            added.append(checked)
        return _with_markers(project, [*project.timeline.markers, *added])


@dataclass(frozen=True, slots=True)
class UpdateMarker(Command):
    """``frame`` にある目印を ``marker`` に置き換える（動かす・名前と色を変える）"""

    frame: int
    marker: Marker

    @property
    def label(self) -> str:
        return "マーカーを変更"

    def apply(self, project: Project) -> Project:
        timeline = project.timeline
        current = marker_at(timeline, self.frame)
        if current is None:
            raise ValueError(f"{self.frame} フレームに目印がありません")
        checked = _check(self.marker, current)
        if checked.frame != self.frame and marker_at(timeline, checked.frame) is not None:
            raise ValueError(f"{checked.frame} フレームにはもう目印があります")
        if checked == current:
            return project
        # 同じフレームの目印が（手で書いたファイルなどで）2 つあっても、置き換えるのは
        # 先頭の 1 つだけ 両方を書き換えると、片方の名前が黙って消える
        markers = list(timeline.markers)
        markers[markers.index(current)] = checked
        return _with_markers(project, markers)


@dataclass(frozen=True, slots=True)
class RemoveMarkers(Command):
    """``frames`` にある目印を消す 1 つでも無ければ何も消さない

    無い位置を黙って飛ばすと、AI は古い一覧の位置で消したつもりのまま次へ進む

    位置 1 つで消すのは、その位置の先頭の 1 つだけ（:func:`marker_at` と右クリックが指す物）
    手で書いた作品などで同じ位置に 2 つあるとき、位置の全部を消すと、指していない別の名前の
    目印まで黙って消える 同じ位置の目印を全部消すときは、位置を目印の数だけ並べて渡す
    """

    frames: tuple[int, ...]

    @property
    def label(self) -> str:
        return "マーカーを削除"

    def apply(self, project: Project) -> Project:
        if not self.frames:
            raise ValueError("消す目印がありません")
        remaining = list(project.timeline.markers)
        missing: set[int] = set()
        for frame in self.frames:
            index = next((i for i, m in enumerate(remaining) if m.frame == frame), None)
            if index is None:
                missing.add(frame)
            else:
                del remaining[index]
        if missing:
            raise ValueError(f"目印がありません: {', '.join(map(str, sorted(missing)))} フレーム")
        return _with_markers(project, remaining)
