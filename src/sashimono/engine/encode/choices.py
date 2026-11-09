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
from sashimono.core.model import (
    Clip,
    MediaId,
    MediaItem,
    Project,
    SceneId,
    Timeline,
    VideoStreamInfo,
)
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


#: 中のシーンをたどる深さの上限 シーンを互いに置き合うと終わらない
_MAX_SCENE_DEPTH = 16


def _drawn(project: Project, timeline: Timeline) -> Iterator[Clip]:
    """``timeline`` で書き出しに映るクリップ 描画と同じ決まり（映るトラック・有効・絵を描く）

    消音・非表示のトラックや無効のクリップを数えると、映らない高い素材に引かれて自動の値が
    上限まで上がり、要らない大きさで書き出す（PR #451 の CodeRabbit の指摘）
    """
    for track in timeline.active_picture_tracks():
        for clip in track.clips:
            if clip.enabled and project.draws_picture(track, clip):
                yield clip


def _timelines(project: Project) -> Iterator[Timeline]:
    """書き出すタイムラインと、そこに映るシーン（入れ子も）のタイムライン

    置いていないシーンは書き出さないので数えない 数えると、選んだシーンの外の高い素材に
    引かれて、低い素材の場面だけを書き出すときも大きなファイルになる（PR #451 の CodeRabbit）
    同じシーンへ浅い道からも届くなら浅い方でたどり直す 先に深い道で着いて止めると、浅い道
    なら上限の内側で描く子のシーンを数え落とす
    """
    reached: dict[SceneId, int] = {}
    stack: list[tuple[Timeline, int]] = [(project.timeline, 0)]
    while stack:
        timeline, depth = stack.pop()
        yield timeline
        if depth >= _MAX_SCENE_DEPTH:
            continue
        for clip in _drawn(project, timeline):
            # 同じ深さで着いたのもたどり直さない 同じシーンを各段に 2 つ置くと、16 段で 13 万回
            # たどって、書き出しの窓の枠を埋める（PR #451 の CodeRabbit の指摘）
            if (
                clip.scene_id is None
                or reached.get(clip.scene_id, _MAX_SCENE_DEPTH + 1) <= depth + 1
            ):
                continue
            reached[clip.scene_id] = depth + 1
            scene = project.find_scene(clip.scene_id)
            if scene is not None:
                stack.append((scene.timeline, depth + 1))


#: 数字で幅を書かない PCM の 1 標本のビット数 μ-law と A-law は 8 ビット
_PCM_DEPTHS = {"pcm_mulaw": 8, "pcm_alaw": 8}


def _audio_bits(media: MediaItem) -> float:
    """非圧縮の音（PCM）の全体の量（ビット） 圧縮した音は映像に比べて小さいので数えない

    多チャンネルの PCM を入れた素材（録画機の mov）は音だけで数 Mbps あり、ファイルの大きさを
    そのまま映像の量と読むと、映像の低い素材まで上限で書き出す 音の長さは素材の記録に無いので
    素材の長さで見る 音が短い素材で引きすぎた分は :func:`matched_bitrate_mbps` が受ける
    """
    total = 0.0
    for stream in media.audio_streams:
        codec = stream.codec.lower()
        if not codec.startswith("pcm_"):
            continue
        digits = "".join(letter for letter in codec if letter.isdigit())
        depth = int(digits) if digits else _PCM_DEPTHS.get(codec, 16)
        total += stream.sample_rate * stream.channels * depth
    return total * float(media.duration)


def _placed_videos(project: Project) -> list[tuple[MediaItem, VideoStreamInfo]]:
    """書き出しに映る、動く絵を持つ素材と、クリップが描く映像の道 静止画は数えない

    道は描画と同じく、クリップの ``stream_index`` の道（無ければ 1 本目 ``engine/decode/video.py``）
    1 本目で見ると、2 本目の大きい道を描くクリップの自動の値が低すぎる
    """
    wanted = {
        (clip.media_id, clip.stream_index)
        for timeline in _timelines(project)
        for clip in _drawn(project, timeline)
        if clip.media_id is not None
    }
    found: dict[tuple[MediaId, int], tuple[MediaItem, VideoStreamInfo]] = {}
    for media_id, index in wanted:
        media = project.find_media(media_id)
        if media is None or not media.has_video or media.is_still or media.duration <= 0:
            continue
        streams = media.video_streams
        stream = next((s for s in streams if s.index == index), streams[0])
        found[(media_id, stream.index)] = (media, stream)
    return list(found.values())


def matched_bitrate_mbps(
    project: Project, *, size_of: Callable[[Path], int] | None = None
) -> int | None:
    """置いた素材に合わせたビットレート（Mbps） 合わせる素材が無ければ ``None``

    ``project`` は書き出す形（:func:`export_target` で選んだシーンを差し込んだ物）
    素材のビットレートはファイルの大きさを長さで割って求める 非圧縮の音の分は引く（圧縮した音は
    映像に比べて小さく、多めに見積もる向きなので崩れない） 引いて映像の分が残らなければ（音が
    素材より短い PCM で引きすぎた）引かずに見る 低く見積もって画質を落とすより、大きめに出す方へ
    倒す 作品の大きさと fps が素材と違えば、1 秒の画素の数の比で直す 比べる映像はクリップが
    描く道 素材が何本もあれば一番高い物に合わせる（低い方に合わせると、その素材の場面が崩れる）
    読めない素材（外したドライブ）は数えない ファイルの大きさを読むので、画面の窓は
    スレッドの外から呼ぶ（ネットワークの共有が応えないと待つ）
    """
    measure = size_of or _file_size
    settings = project.settings
    out_rate = settings.width * settings.height * float(settings.frame_rate.fps)
    best: float | None = None
    sizes: dict[Path, int | None] = {}
    for media, stream in _placed_videos(project):
        if media.path not in sizes:
            try:
                sizes[media.path] = measure(media.path)
            except OSError:
                sizes[media.path] = None
        size = sizes[media.path]
        pixels = stream.width * stream.height * float(stream.frame_rate.fps)
        if size is None or size <= 0 or pixels <= 0:
            continue
        whole = size * 8
        video = whole - _audio_bits(media)
        # 映像の道の長さで割る 音の方が長い素材を入れ物の長さで割ると、映像の量を低く見積もる
        end = stream.end_time
        seconds = float(end) if end is not None and end > 0 else float(media.duration)
        bits = (video if video > 0 else whole) / seconds
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
