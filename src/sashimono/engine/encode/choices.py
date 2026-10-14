"""書き出しの窓で選べる物と、選んだ物から書き出しの設定を作る所

画面の書き出しの窓と AI の書き出し（#327）は、ここを通って同じ設定を作る 別々に組むと、
同じ範囲・同じ画質を頼んだのに、AI が書き出した物だけ範囲や先読みの深さが違う、が起きる
Qt を使わない AI の道具の確かめ（編集画面のスレッドの外でも呼ぶ）からも使うため
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sashimono.core.commands import export_range
from sashimono.core.model import Project, SceneId
from sashimono.engine.encode.exporter import ExportSettings

__all__ = [
    "BITRATE_RANGE_MBPS",
    "CONTAINER_SUFFIXES",
    "DEFAULT_BITRATE_MBPS",
    "RANGE_ALL",
    "RANGE_WORK_AREA",
    "build_settings",
    "chosen_range",
    "export_target",
]

#: 「書き出す範囲」の選びに持たせる値
RANGE_ALL = "all"
RANGE_WORK_AREA = "work_area"

#: ビットレートの欄の範囲と初めの値（Mbps）
BITRATE_RANGE_MBPS = (1, 200)
DEFAULT_BITRATE_MBPS = 12

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
) -> ExportSettings:
    """書き出しの設定を作る

    ``codec`` が ``None`` なら書き出し側に選ばせる（使えなければ次の候補へ落ちる）
    先読みの深さ・デコードの並列数・音の動きのつなぎは、書き出しごとに変える物ではなく
    本人の設定（表示 → 設定…）から渡す
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
    )
