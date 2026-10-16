"""環境の疎通確認 実装に入る前に、前提が揃っているかをここで潰す

.venv/Scripts/python.exe tools/check_env.py
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

# cmd.exe の既定コードページ (cp932) だと日本語が化けるので UTF-8 に固定する
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

results: list[tuple[str, bool, str]] = []


def check(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    def deco(fn: Callable[[], str]) -> Callable[[], str]:
        try:
            detail = fn()
            results.append((name, True, detail))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
        return fn

    return deco


@check("Python")
def _python() -> str:
    v = sys.version_info
    if v < (3, 12):
        raise RuntimeError(f"3.12 以上が必要 (現在 {v.major}.{v.minor})")
    return f"{v.major}.{v.minor}.{v.micro}"


@check("ffmpeg CLI")
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("PATH に見つからない")
    out = subprocess.run([exe, "-version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0]


@check("PySide6")
def _pyside() -> str:
    import PySide6
    from PySide6.QtCore import qVersion

    return f"PySide6 {PySide6.__version__} / Qt {qVersion()}"


@check("OpenGL コンテキスト (Qt + PyOpenGL)")
def _gl() -> str:
    from OpenGL import GL
    from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext, QSurfaceFormat

    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)

    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()

    ctx = QOpenGLContext()
    ctx.setFormat(fmt)
    if not ctx.create():
        raise RuntimeError("QOpenGLContext.create() が失敗")
    if not ctx.makeCurrent(surface):
        raise RuntimeError("makeCurrent() が失敗")

    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    version = GL.glGetString(GL.GL_VERSION).decode()
    glsl = GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode()
    ctx.doneCurrent()
    del app
    return f"{renderer} / GL {version} / GLSL {glsl}"


@check("PyAV デコード")
def _decode() -> str:
    import av

    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample.mp4"
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
                "testsrc2=size=1280x720:rate=30:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(sample),
            ],
            check=True,
            capture_output=True,
        )
        with av.open(str(sample)) as container:
            vstream = container.streams.video[0]
            astream = container.streams.audio[0]
            frames = sum(1 for _ in container.decode(video=0))
        return (
            f"av {av.__version__} / {frames} フレーム復号 / "
            f"{vstream.codec_context.name} {vstream.width}x{vstream.height} "
            f"+ {astream.codec_context.name} {astream.sample_rate}Hz"
        )


@check("NVENC / NVDEC")
def _nvenc() -> str:
    import av.codec

    wanted = ["h264_nvenc", "hevc_nvenc", "av1_nvenc", "h264_cuvid"]
    found = []
    for name in wanted:
        try:
            av.codec.Codec(name, "w" if name.endswith("nvenc") else "r")
            found.append(name)
        except Exception:
            pass
    if not found:
        raise RuntimeError("ハードウェアコーデックが 1 つも見つからない")
    return ", ".join(found)


@check("オーディオ出力")
def _audio() -> str:
    import sounddevice as sd

    default_out = sd.query_devices(kind="output")
    return f"{default_out['name']} / {int(default_out['default_samplerate'])}Hz"


@check("numpy")
def _numpy() -> str:
    import numpy as np

    return np.__version__


width = max(len(n) for n, _, _ in results)
failed = 0
for name, ok, detail in results:
    mark = "OK  " if ok else "NG  "
    if not ok:
        failed += 1
    print(f"{mark}{name.ljust(width)}  {detail}")

print()
if failed:
    print(f"{failed} 件が未達です")
    sys.exit(1)
print("すべて通過 実装に進めます")
