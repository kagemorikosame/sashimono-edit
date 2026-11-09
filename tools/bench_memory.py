r"""アプリのメモリの内訳を測る（#454 #443 の段階 4）

    .venv\Scripts\python.exe tools\bench_memory.py
    .venv\Scripts\python.exe tools\bench_memory.py --clips 1322 --effects 4018 --edits 200

実行中のアプリが 1.6 GB 前後あった（#454） 何がどれだけ持っているかを、段を 1 つずつ足して
コミット（Windows の Private Bytes）の増え方で測る

1. 部品を読んだだけ（Python・Qt・PyAV・numpy・OpenGL）
2. 空の編集画面を出した
3. 実プロジェクトと同じ数のクリップとエフェクトを開いた（文字のクリップに、設定パネルの
   〔エフェクトを追加〕に並ぶ物を順に積む）
4. 編集を ``--edits`` 回積んだ（履歴は既定で 200 段 1 段ごとにプロジェクトの写しを持つ）
5. ``--media`` を付けると、ffmpeg で作った素材を置いて先読みを貯め切った（``--onscreen`` が要る）

画面は出さない（offscreen） 先読みは GL を共有するので、5 を測るときだけ ``--onscreen``
素材は作った物だけ 本人の作品は使わない（並びの数だけ同じにする）
"""

from __future__ import annotations

import argparse
import atexit
import gc
import io
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 開発者本人の設定に触らない
_base = Path(tempfile.mkdtemp(prefix="sashimono-memory-bench-"))
atexit.register(shutil.rmtree, _base, True)
os.environ["APPDATA"] = str(_base / "roaming")
os.environ["LOCALAPPDATA"] = str(_base / "local")

from sashimono.memory_usage import process_memory  # noqa: E402


def _private_mb() -> float:
    usage = process_memory()
    if usage is None:
        return 0.0
    return usage.private_mb if usage.private_mb is not None else usage.resident_mb


class Steps:
    """段ごとのメモリを積み、前の段との差を出す"""

    def __init__(self) -> None:
        self._rows: list[tuple[str, float]] = []

    def mark(self, name: str) -> float:
        gc.collect()
        value = _private_mb()
        previous = self._rows[-1][1] if self._rows else 0.0
        self._rows.append((name, value))
        print(f"{name}: {value:.0f} MB（+{value - previous:.0f}）")
        return value

    def table(self) -> str:
        lines = ["| 段 | コミット MB | 前の段から |", "|---|---|---|"]
        previous = 0.0
        for name, value in self._rows:
            lines.append(f"| {name} | {value:.0f} | +{value - previous:.0f} |")
            previous = value
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--clips", type=int, default=1322, help="文字のクリップの数")
    parser.add_argument("--effects", type=int, default=4018, help="クリップに積むエフェクトの合計")
    parser.add_argument("--tracks", type=int, default=12, help="トラックの数")
    parser.add_argument("--edits", type=int, default=200, help="積む編集の数（履歴の段）")
    parser.add_argument(
        "--media", type=int, default=0, help="置く素材（1080p60 10 秒）の数 先読みも貯める"
    )
    parser.add_argument("--onscreen", action="store_true", help="窓を本物の画面に出す")
    args = parser.parse_args(argv)
    if args.media and not args.onscreen:
        print("--media は先読みの GL を作るので --onscreen と一緒に使う")
        return 2
    if not args.onscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if process_memory() is None:
        print("この機械ではメモリを測れない")
        return 0

    steps = Steps()
    steps.mark("起動した Python")

    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    from sashimono.compat.looks import effects_for_clip
    from sashimono.core.commands import AddClip, AddEffect, AddMedia, AddTrack, MoveClip
    from sashimono.core.model import Clip, Project, ProjectSettings, Track, TrackKind
    from sashimono.core.timebase import FrameRate
    from sashimono.effects.sources import TEXT
    from sashimono.engine.decode import probe_media
    from sashimono.ui.main_window import MainWindow
    from sashimono.ui.workspace import SCENES_ON_DEMAND, Preferences, PreferenceStore

    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)

    def settle() -> None:
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)

    steps.mark("部品を読んだ")

    # 窓が自分で控えと解析を走らせないようにする 測りたいのは持っている物で、作っている物ではない
    PreferenceStore().save(replace(Preferences(), use_proxy=False, scene_analysis=SCENES_ON_DEMAND))
    empty = MainWindow(Project.create(ProjectSettings()), confirm_unsaved=False)
    empty.resize(1600, 900)
    empty.show()
    settle()
    steps.mark("空の編集画面")

    project = Project.create(ProjectSettings(width=1920, height=1080, frame_rate=FrameRate(60)))
    tracks = [Track(TrackKind.VIDEO, f"V{index + 1}") for index in range(args.tracks)]
    for track in tracks:
        project = AddTrack(track).apply(project)
    clips = []
    for index in range(args.clips):
        track = tracks[index % args.tracks]
        clip = Clip(
            timeline_start=(index // args.tracks) * 72,
            duration=60,
            source=TEXT.create(text=f"{index}"),
        )
        project = AddClip(track.id, clip).apply(project)
        clips.append((clip.id, track.id))
    if args.clips:
        located = project.timeline.locate_clip(clips[0][0])
        assert located is not None
        usable = effects_for_clip(project, *located)
        for index in range(args.effects):
            clip_id = clips[index % len(clips)][0]
            project = AddEffect(clip_id, usable[index % len(usable)].create()).apply(project)
    began = time.perf_counter()
    empty.close()
    empty.deleteLater()
    settle()
    window = MainWindow(project, confirm_unsaved=False)
    window.resize(1600, 900)
    window.show()
    settle()
    print(f"  開くのに {time.perf_counter() - began:.1f} 秒")
    steps.mark(f"クリップ {args.clips} 本・エフェクト {args.effects} 個")

    if args.edits and clips:
        for n in range(args.edits):
            clip_id, track_id = clips[(n * 37) % len(clips)]
            located = window.view_project.timeline.locate_clip(clip_id)
            assert located is not None
            start = located[1].timeline_start
            window.execute(MoveClip(clip_id, start + (1 if n % 2 == 0 else -1), track_id))
        settle()
        steps.mark(f"編集 {args.edits} 回（履歴）")

    if args.media:
        import subprocess

        made = []
        for index in range(args.media):
            path = _base / f"media-{index}.mp4"
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc2=size=1920x1080:rate=60:duration=10,hue=h={index * 40}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
            made.append(probe_media(path))
        video = window.view_project
        for index, media in enumerate(made):
            video = AddMedia(media).apply(video)
            track = Track(TrackKind.VIDEO, f"M{index + 1}")
            video = AddTrack(track).apply(video)
            video = AddClip(
                track.id, Clip(timeline_start=0, duration=600, media_id=media.id)
            ).apply(video)
        window.close()
        window.deleteLater()
        settle()
        window = MainWindow(video, confirm_unsaved=False)
        window.resize(1600, 900)
        window.show()
        settle()
        steps.mark(f"素材 {args.media} 本を置いた")
        # 先読みが貯め終えるのを待つ（貯めた絵は GPU のメモリだが、駆動が控えを持つことがある）
        until = time.perf_counter() + 20
        while time.perf_counter() < until:
            settle()
            time.sleep(0.05)
        steps.mark("先読みを貯めた（20 秒）")

    print()
    print(steps.table())
    window.close()
    window.deleteLater()
    settle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
