"""書き出しの窓で選べる物と、選んだ物から書き出しの設定を作る所

画面の書き出しの窓と AI の書き出し（#327）は、ここを通って同じ設定を作る 別々に組むと、
同じ範囲・同じ画質を頼んだのに、AI が書き出した物だけ範囲や先読みの深さが違う、が起きる
Qt を使わない AI の道具の確かめ（編集画面のスレッドの外でも呼ぶ）からも使うため
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from sashimono.core.commands import export_range
from sashimono.core.model import MediaItem, Project, SceneId, Timeline
from sashimono.engine.encode.exporter import ExportSettings

__all__ = [
    "BITRATE_RANGE_MBPS",
    "CONTAINER_SUFFIXES",
    "DEFAULT_BITRATE_MBPS",
    "MATCH_FLOOR_MBPS",
    "MATCH_HEADROOM",
    "RANGE_ALL",
    "RANGE_WORK_AREA",
    "BitrateChoice",
    "build_settings",
    "chosen_range",
    "export_target",
    "matched_bitrate_mbps",
]

#: 「書き出す範囲」の選びに持たせる値
RANGE_ALL = "all"
RANGE_WORK_AREA = "work_area"

#: ビットレートの欄の範囲と初めの値（Mbps）
BITRATE_RANGE_MBPS = (1, 200)
DEFAULT_BITRATE_MBPS = 12

#: 素材に合わせるとき、素材のビットレートに掛ける余裕 書き出し直すと同じビットレートでも絵が少し
#: 崩れる（世代の劣化）うえ、字幕や図形を重ねた絵は素材より細かい所が増える
MATCH_HEADROOM = 1.25
#: 素材に合わせても下回らない値（Mbps） 画面の録画のように 1 Mbps を切る素材でも、重ねた文字の縁が
#: 崩れない所に置く
MATCH_FLOOR_MBPS = 2


@dataclass(frozen=True, slots=True)
class BitrateChoice:
    """書き出しのビットレートの既定の決め方（本人の設定 #414）

    ``auto`` なら素材のビットレートに合わせ、``mbps`` を上限にする 切れば ``mbps`` のまま
    通しの確かめで、12 Mbps 決め打ちの既定は低いビットレートの素材を 7 倍の大きさにした
    """

    auto: bool = True
    mbps: int = DEFAULT_BITRATE_MBPS

    def for_project(self, project: Project, *, size_of: Callable[[Path], int] | None = None) -> int:
        """この作品を書き出すときの既定（Mbps）"""
        if not self.auto:
            return self.mbps
        matched = matched_bitrate_mbps(project, size_of=size_of)
        return self.mbps if matched is None else min(self.mbps, matched)


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _timelines(project: Project) -> Iterator[Timeline]:
    yield project.timeline
    for scene in project.scenes:
        yield scene.timeline


def _placed_videos(project: Project) -> list[MediaItem]:
    """タイムライン（とシーンの中）に置いた、動く絵を持つ素材 静止画は数えない"""
    wanted = {
        clip.media_id
        for timeline in _timelines(project)
        for track in timeline.tracks
        for clip in track.clips
        if clip.media_id is not None
    }
    return [
        media
        for media in project.media
        if media.id in wanted and media.has_video and not media.is_still and media.duration > 0
    ]


def matched_bitrate_mbps(
    project: Project, *, size_of: Callable[[Path], int] | None = None
) -> int | None:
    """置いた素材に合わせたビットレート（Mbps） 合わせる素材が無ければ ``None``

    素材のビットレートはファイルの大きさを長さで割って求める（音の分も入るが、映像に比べて
    小さく、多めに見積もる向きなので崩れない） 作品の大きさと fps が素材と違えば、1 秒の画素の
    数の比で直す 素材が何本もあれば一番高い物に合わせる（低い方に合わせると、その素材の場面が
    崩れる） 読めない素材（外したドライブ）は数えない
    """
    measure = size_of or _file_size
    settings = project.settings
    out_rate = settings.width * settings.height * float(settings.frame_rate.fps)
    best: float | None = None
    for media in _placed_videos(project):
        try:
            size = measure(media.path)
        except OSError:
            continue
        stream = media.video_streams[0]
        pixels = stream.width * stream.height * float(stream.frame_rate.fps)
        if size <= 0 or pixels <= 0:
            continue
        bits = size * 8 / float(media.duration)
        scaled = bits * out_rate / pixels
        best = scaled if best is None else max(best, scaled)
    if best is None:
        return None
    low, high = BITRATE_RANGE_MBPS
    wanted = math.ceil(best * MATCH_HEADROOM / 1_000_000)
    return min(high, max(low, MATCH_FLOOR_MBPS, wanted))


#: 書き出せる入れ物（拡張子） 映像 H.264/H.265 と音声 AAC をそのまま入れられる物だけ
#: 拡張子で入れ物が決まる（FFmpeg が名前から選ぶ） 知らない拡張子を通すと、
#: 作品のファイル（.sme）や実行できるファイルの名前で動画を書けてしまう
CONTAINER_SUFFIXES = (".mp4", ".mov", ".mkv")


def export_target(whole: Project, scene_id: SceneId | None) -> Project:
    """書き出すプロジェクト 選んだシーンのタイムラインをメインの所へ差し込んだ物

    ``scene_id`` が ``None`` ならメインのまま 無いシーンなら ``KeyError``
    """
    if scene_id is None:
        return whole
    scene = whole.find_scene(scene_id)
    if scene is None:
        raise KeyError(f"シーンがありません: {scene_id}")
    return replace(whole, timeline=scene.timeline)


def chosen_range(project: Project, choice: str) -> tuple[int, int] | None:
    """選んだ範囲 全体なら ``None``（書き出し側がタイムラインの長さを使う）

    範囲を選んだのに使える範囲が無ければ ``ValueError`` 黙って全体にすると、範囲だけを
    出すつもりの人に長い全体を書き出して待たせる
    """
    if choice == RANGE_ALL:
        return None
    if choice != RANGE_WORK_AREA:
        raise ValueError(f"書き出す範囲は {RANGE_ALL} か {RANGE_WORK_AREA} です: {choice}")
    area = export_range(project.timeline)
    if area is None:
        raise ValueError(
            "書き出す範囲が決まっていないか、タイムラインの終わりより後ろにあります"
            " 全体を書き出すか、タイムラインの目盛りで範囲を決めてください"
        )
    return area


def build_settings(
    path: Path,
    *,
    codec: str | None,
    bitrate_mbps: int,
    frame_range: tuple[int, int] | None,
    pipeline_depth: int,
    decode_threads: int,
    smooth_history: bool,
    gpu_color: bool,
    chapters: bool,
    encoder_priority: str,
) -> ExportSettings:
    """書き出しの設定を作る

    ``codec`` が ``None`` なら書き出し側に選ばせる（使えなければ次の候補へ落ちる）
    先読みの深さ・デコードの並列数・音の動きのつなぎは、書き出しごとに変える物ではなく
    本人の設定（表示 → 設定…）から渡す ``chapters``（目印をチャプターとして埋める #330）も
    本人の設定から渡す 既定を置かないのは、AI の書き出しが設定を渡し忘れて、切った人の
    動画に章が入るのを型で止めるため ``gpu_color``（色変換を GPU で行う #372）も同じ理由で
    既定を置かない 駆動との相性で切った人の書き出しが、AI からだけ GPU で変換されないように
    ``encoder_priority``（どの作り手のエンコーダを先に試すか #365）も同じ理由で既定を置かない
    """
    low, high = BITRATE_RANGE_MBPS
    if not low <= bitrate_mbps <= high:
        raise ValueError(f"ビットレートは {low}〜{high} Mbps です: {bitrate_mbps}")
    return ExportSettings(
        path=path,
        video_codec=codec or None,
        video_bitrate=bitrate_mbps * 1_000_000,
        frame_range=frame_range,
        pipeline_depth=pipeline_depth,
        decode_threads=decode_threads,
        smooth_history=smooth_history,
        gpu_color=gpu_color,
        chapters=chapters,
        encoder_priority=encoder_priority,
    )
