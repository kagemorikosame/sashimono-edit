"""探し直して選んだファイルが、作品の覚えている素材と同じ物かを確かめる（#447）

付け替えは素材の情報（長さ・道・字幕の時刻）をそのまま使う 別の動画を選ぶと、クリップの
時刻が別の絵を指し、字幕が声とずれる 選び間違いに気付けるよう、ファイルを開いて
長さと道の数を比べる（中身の画素までは比べない 開くだけで済む物にして、数十本でも待たせない）
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from sashimono.core.model import MediaId, MediaItem, Project
from sashimono.core.relink import siblings
from sashimono.engine.decode.probe import ProbeError, probe_media

__all__ = ["LENGTH_TOLERANCE", "RelinkPlan", "differs", "plan_relink"]

#: 長さの違いを許す幅（秒） 書き出し直した素材や入れ物を替えた素材は、頭と終わりの端数で
#: 数フレーム違うことがある それより大きく違えば別の素材と見る
LENGTH_TOLERANCE = Fraction(1, 2)


def differs(item: MediaItem, path: Path) -> str | None:
    """``path`` が ``item`` と違う素材に見えるなら、その訳 同じに見えれば ``None``"""
    try:
        found = probe_media(Path(path))
    except (ProbeError, OSError) as exc:
        return f"素材として読めません（{exc}）"
    reasons = []
    if abs(found.duration - item.duration) > LENGTH_TOLERANCE:
        reasons.append(
            f"長さが違います（作品では {float(item.duration):.1f} 秒、"
            f"選んだ物は {float(found.duration):.1f} 秒）"
        )
    if len(found.video_streams) != len(item.video_streams):
        reasons.append(
            f"映像の数が違います（作品では {len(item.video_streams)}、"
            f"選んだ物は {len(found.video_streams)}）"
        )
    if len(found.audio_streams) != len(item.audio_streams):
        reasons.append(
            f"音声の数が違います（作品では {len(item.audio_streams)}、"
            f"選んだ物は {len(found.audio_streams)}）"
        )
    return " ".join(reasons) or None


@dataclass(frozen=True, slots=True)
class RelinkPlan:
    """付け替えの段取り 画面の窓と AI の道具が同じ物を作る"""

    #: 付け替える素材と場所（選んだ 1 本が先頭）
    changes: tuple[tuple[MediaId, Path], ...]
    #: 選んだファイルが違う素材に見える訳 同じに見えれば ``None``
    concern: str | None
    #: 名前で見つかったが違う素材に見えたので付け替えない物（素材の名前と訳）
    skipped: tuple[tuple[str, str], ...]


def plan_relink(
    project: Project, media_id: MediaId, chosen: Path, *, search: bool = True
) -> RelinkPlan:
    """``media_id`` を ``chosen`` へ付け替える段取り ``search`` なら残りの見つからない素材も探す

    名前で見つけた残りは、違う素材に見えた物を外す（同じ名前の別の撮影を黙って付け替えない）
    """
    item = project.require_media(media_id)
    changes: list[tuple[MediaId, Path]] = [(media_id, Path(chosen))]
    skipped: list[tuple[str, str]] = []
    if search:
        by_id = {media.id: media for media in project.media}
        for other, place in siblings(project, media_id, Path(chosen)).items():
            reason = differs(by_id[other], place)
            if reason is None:
                changes.append((other, place))
            else:
                skipped.append((by_id[other].name, reason))
    return RelinkPlan(
        changes=tuple(changes), concern=differs(item, Path(chosen)), skipped=tuple(skipped)
    )
