"""拾った区間に、程度（high / medium / low）と人に読める説明を付けて、問題の一覧にする（#337）

程度の決め方

| 種類 | high | medium | low |
|---|---|---|---|
| 黒いコマ | 何も置いていない隙間 | 置いた絵が黒い | 頭か終わり（フェードの演出が多い） |
| 止まった絵 | - | 動画の絵が止まっている | 静止画・文字だけの区間 |
| 無音 | 全体が無音 | 途中の無音 | 頭か終わり |
| 音割れ | いつも | - | - |
| 音量の基準 | 6 LU を超えて外れる | 幅を超えて外れる | - |
| 字幕のはみ出し | 画面の外へ出て切れる | 余白（安全域）にだけ入る | 動く字幕（テロップなど） |
| 長さ | - | 目安から外れる | - |

頭と終わりの黒と無音を low にするのは、フェードの入りと抜けで普通に作る形だから
直すかどうかは AI と人が絵を見て決める（:func:`preview_frame` で確かめる）
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from sashimono.core.model import MediaItem, Project
from sashimono.engine.checks.scan import (
    BLACK,
    CLIPPING,
    FREEZE,
    LENGTH,
    LOUDNESS,
    SILENCE,
    TEXT_OVERFLOW,
    CheckLimits,
    Run,
)
from sashimono.engine.render.outline import is_generated

__all__ = [
    "HIGH",
    "KIND_NAMES",
    "LOW",
    "MEDIUM",
    "CheckReport",
    "ClipIndex",
    "Problem",
    "assemble",
]

HIGH = "high"
MEDIUM = "medium"
LOW = "low"
_ORDER = {HIGH: 0, MEDIUM: 1, LOW: 2}
#: 音量の基準から大きく外れたとみなす差（LU） 配信の -14 に対して -20 を下回る作品は、
#: 再生側がそろえても（持ち上げない配信が多い）ほかの動画の後で明らかに小さく聞こえる
_FAR_LOUDNESS = 6.0
#: 1 つの問題に添えるクリップの数 長い隙間の前後で何十本も並べない
_CLIPS_PER_PROBLEM = 3


@dataclass(frozen=True, slots=True)
class Problem:
    """問題 1 つ フレームは ``[start, end)``（書き出した動画なら、その動画のコマの番号）"""

    kind: str
    severity: str
    start: int
    end: int
    detail: str
    measure: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    clip_ids: tuple[str, ...] = ()
    #: 解析の途中で、まだ続いているかもしれない
    ongoing: bool = False


@dataclass(frozen=True, slots=True)
class _Placed:
    start: int
    end: int
    clip_id: str
    #: 動く絵（動画の素材）を描くか 静止画・文字・図形は偽
    moving: bool
    #: 文字（テキスト）のクリップか
    text: bool = False


class ClipIndex:
    """タイムラインのどこに何を置いたか 隙間と、止まった絵が静止画かを見分けるため"""

    def __init__(self, project: Project) -> None:
        timeline = project.timeline
        self._pictures: list[_Placed] = []
        self._sounds: list[_Placed] = []
        for track in timeline.active_picture_tracks():
            for clip in track.clips:
                if not clip.enabled or clip.is_group or not project.draws_picture(track, clip):
                    continue
                media = project.find_media(clip.media_id) if clip.media_id is not None else None
                self._pictures.append(
                    _Placed(
                        clip.timeline_start,
                        clip.timeline_end,
                        str(clip.id),
                        _moving(media) and not is_generated(clip) and clip.hold_at is None,
                        clip.source is not None and clip.source.kind == "text",
                    )
                )
        for track in timeline.active_sound_tracks():
            for clip in track.clips:
                if clip.enabled and project.plays_sound(track, clip):
                    self._sounds.append(
                        _Placed(clip.timeline_start, clip.timeline_end, str(clip.id), False)
                    )

    def pictures(self, start: int, end: int) -> list[_Placed]:
        return [item for item in self._pictures if item.start < end and start < item.end]

    def sounds(self, start: int, end: int) -> list[_Placed]:
        return [item for item in self._sounds if item.start < end and start < item.end]


def _moving(media: MediaItem | None) -> bool:
    return media is not None and media.has_video and not media.is_still


@dataclass(frozen=True, slots=True)
class CheckReport:
    """検査の結果（途中なら途中まで）"""

    problems: tuple[Problem, ...]
    #: 調べる範囲（フレーム） ``[start, end)``
    start: int
    end: int
    #: どのフレームの手前まで調べたか
    analyzed: int
    fps: float
    complete: bool
    limits: CheckLimits
    #: 積分ラウドネス（LUFS） 測れなければ ``None``
    loudness: float | None = None
    #: 最大のサンプル（dBFS） 音を調べなければ ``None``
    peak_db: float | None = None
    notes: tuple[str, ...] = ()
    #: 頼まれたのに調べられなかった種類（GPU が無い・ファイルに音が無いなど）
    #: 調べた種類と混ぜると、問題が 0 件のときに「その種類の問題は無い」と読まれる
    skipped: frozenset[str] = frozenset()

    @property
    def checked(self) -> frozenset[str]:
        """実際に調べた種類

        長さは目安（``expected_seconds``）が無ければ比べる相手が無いので数えない 数えると、
        目安を渡さない呼び方で「長さに問題は無い」と読まれる（長さそのものは要約に出す）
        """
        unchecked = self.skipped
        if self.limits.expected_seconds is None:
            unchecked = unchecked | {LENGTH}
        return self.limits.kinds - unchecked

    @property
    def seconds(self) -> float:
        return (self.end - self.start) / self.fps

    def counts(self) -> dict[str, int]:
        found: dict[str, int] = {}
        for problem in self.problems:
            found[problem.kind] = found.get(problem.kind, 0) + 1
        return found


def assemble(
    runs: list[Run],
    *,
    start: int,
    end: int,
    analyzed: int,
    fps: float,
    complete: bool,
    limits: CheckLimits,
    index: ClipIndex | None,
    loudness: float | None,
    peak: float | None,
    length_seconds: float | None,
    notes: tuple[str, ...] = (),
    skipped: frozenset[str] = frozenset(),
) -> CheckReport:
    """区間に程度と説明を付けて並べる ``length_seconds`` は長さを比べられるときだけ渡す"""
    problems = [_problem(run, start, end, index) for run in runs if run.kind in limits.kinds]
    whole = (start, max(start + 1, end))
    if LOUDNESS in limits.kinds and loudness is not None:
        difference = loudness - limits.loudness_target
        if abs(difference) > limits.loudness_tolerance:
            louder = difference > 0
            problems.append(
                Problem(
                    LOUDNESS,
                    HIGH if abs(difference) > _FAR_LOUDNESS else MEDIUM,
                    *whole,
                    (
                        f"全体の音量（積分ラウドネス）が基準 {limits.loudness_target:g} LUFS より "
                        f"{abs(difference):.1f} LU {'大きい' if louder else '小さい'}"
                        + (
                            " 配信では下げて再生され、放送では基準を超える"
                            if louder
                            else " 配信では持ち上げられないことが多く、ほかの動画より小さく聞こえる"
                        )
                    ),
                    {
                        "lufs": round(loudness, 1),
                        "target_lufs": limits.loudness_target,
                        "difference_lu": round(difference, 1),
                    },
                    ongoing=not complete,
                )
            )
    if LENGTH in limits.kinds and length_seconds is not None and limits.expected_seconds:
        difference = length_seconds - limits.expected_seconds
        if abs(difference) > limits.length_tolerance:
            problems.append(
                Problem(
                    LENGTH,
                    MEDIUM,
                    *whole,
                    f"長さ {length_seconds:.2f} 秒が目安 {limits.expected_seconds:g} 秒より "
                    f"{abs(difference):.2f} 秒{'長い' if difference > 0 else '短い'}",
                    {
                        "seconds": round(length_seconds, 3),
                        "expected_seconds": limits.expected_seconds,
                        "difference_seconds": round(difference, 3),
                    },
                )
            )
    problems.sort(key=lambda problem: (problem.start, _ORDER[problem.severity], problem.kind))
    peak_db = None if peak is None else (-120.0 if peak <= 0 else 20.0 * math.log10(peak))
    return CheckReport(
        problems=tuple(problems),
        start=start,
        end=end,
        analyzed=analyzed,
        fps=fps,
        complete=complete,
        limits=limits,
        loudness=loudness,
        peak_db=peak_db,
        notes=notes,
        skipped=skipped,
    )


def _problem(run: Run, start: int, end: int, index: ClipIndex | None) -> Problem:
    edge = run.start <= start or run.end >= end
    values = dict(run.values)
    if run.kind == BLACK:
        placed = index.pictures(run.start, run.end) if index is not None else None
        if edge:
            severity, detail = LOW, "頭か終わりの黒いコマ フェードの演出ならそのままでよい"
        elif placed is not None and all(item.text for item in placed):
            # 字幕だけが載った黒も隙間 字幕は下の絵を隠さないので、黒い地に字だけが出る
            severity, detail = HIGH, "絵を何も置いていない隙間 黒いコマが出る"
        else:
            severity, detail = MEDIUM, "画面がほぼ真っ黒 置いた絵が暗いか、透明になっている"
        return _made(run, severity, detail, values, placed)
    if run.kind == FREEZE:
        placed = index.pictures(run.start, run.end) if index is not None else None
        if placed is not None and not any(item.moving for item in placed):
            severity, detail = LOW, "静止画・文字・図形だけの区間 意図した止め絵ならそのままでよい"
        else:
            severity, detail = (
                MEDIUM,
                "動画の絵が止まっている（止め絵・素材の終わり・読めないコマ）",
            )
        return _made(run, severity, detail, values, placed)
    if run.kind == SILENCE:
        placed = index.sounds(run.start, run.end) if index is not None else None
        if run.start <= start and run.end >= end:
            severity, detail = HIGH, "全体が無音 音を置いていないか、すべて消音になっている"
        elif edge:
            severity, detail = LOW, "頭か終わりの無音"
        elif placed is not None and not placed:
            severity, detail = MEDIUM, "音を何も置いていない区間"
        else:
            severity, detail = MEDIUM, "置いた音が無音（録り漏れ・音量 0・消音）"
        return _made(run, severity, detail, values, placed)
    if run.kind == CLIPPING:
        detail = (
            "音割れ 混ぜた音が最大を超えていて、書き出すと頭打ちになる 音量を下げる"
            if values.get("over_full_scale")
            else "音割れ 最大の大きさで続けて頭打ちになっている（素材の音が割れているか、上げすぎ）"
        )
        return _made(run, HIGH, detail, values, None)
    if run.kind == TEXT_OVERFLOW:
        if values.get("moving"):
            severity = LOW
            detail = "動く文字が画面の外へ出る 流れるテロップなど、わざとならそのままでよい"
        elif values.get("beyond_screen"):
            severity = HIGH
            detail = f"字幕が画面の{values.get('sides', '')}からはみ出して切れる"
        else:
            severity = MEDIUM
            detail = f"字幕が画面の{values.get('sides', '')}の余白（安全域）に入っている"
        return Problem(
            run.kind, severity, run.start, run.end, detail, values, run.clip_ids, run.ongoing
        )
    return Problem(
        run.kind, MEDIUM, run.start, run.end, run.kind, values, run.clip_ids, run.ongoing
    )


def _made(
    run: Run,
    severity: str,
    detail: str,
    values: dict[str, float | int | str | bool],
    placed: list[_Placed] | None,
) -> Problem:
    clip_ids = tuple(item.clip_id for item in placed[:_CLIPS_PER_PROBLEM]) if placed else ()
    return Problem(run.kind, severity, run.start, run.end, detail, values, clip_ids, run.ongoing)


#: 返事の説明に使う、調べる種類の名前
KIND_NAMES: Mapping[str, str] = {
    BLACK: "黒いコマ",
    FREEZE: "止まった絵",
    SILENCE: "無音",
    CLIPPING: "音割れ",
    LOUDNESS: "音量の基準",
    TEXT_OVERFLOW: "字幕のはみ出し",
    LENGTH: "長さ",
}
