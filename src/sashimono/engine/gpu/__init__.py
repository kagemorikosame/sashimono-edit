"""OpenGL による合成とエフェクト処理"""

from sashimono.engine.gpu.compositor import (
    BlendMode,
    Compositor,
    Corners,
    Placement,
    Transform,
    fit_placement,
)
from sashimono.engine.gpu.context import (
    CurrentGLContext,
    GLContextError,
    GLScope,
    OffscreenGLContext,
    ensure_qt_application,
    opengl_usable,
    preferred_surface_format,
)
from sashimono.engine.gpu.effects import OFF_SCREEN, EffectProcessor, srgb_to_linear
from sashimono.engine.gpu.glutil import (
    FULL_RECT,
    IDENTITY,
    VERTEX_SHADER,
    Framebuffer,
    Program,
    ScreenQuad,
    ShaderError,
    Texture,
)

__all__ = [
    "FULL_RECT",
    "IDENTITY",
    "OFF_SCREEN",
    "VERTEX_SHADER",
    "BlendMode",
    "Compositor",
    "Corners",
    "CurrentGLContext",
    "EffectProcessor",
    "Framebuffer",
    "GLContextError",
    "GLScope",
    "OffscreenGLContext",
    "Placement",
    "Program",
    "ScreenQuad",
    "ShaderError",
    "Texture",
    "Transform",
    "ensure_qt_application",
    "fit_placement",
    "opengl_usable",
    "preferred_surface_format",
    "srgb_to_linear",
]
