"""エフェクトの定義

パラメータの仕様は AviUtl のスクリプト制御文字と 1 対 1 に対応させてある
自前のエフェクトも配布スクリプトも同じ形に載るので、設定 UI の自動生成と
プリセットの保存が 1 つの実装で済む
"""

# 読み込んだ時点で標準エフェクトを一覧へ入れる 使う側が登録を忘れると、
# プロジェクトを開いたときに全部「未知のエフェクト」になる
from sashimono.effects import audio as _audio
from sashimono.effects import builtin as _builtin
from sashimono.effects import grading as _grading
from sashimono.effects import motion as _motion
from sashimono.effects import optics as _optics
from sashimono.effects import paint as _paint
from sashimono.effects import region as _region
from sashimono.effects import spawn as _spawn
from sashimono.effects import stylize as _stylize
from sashimono.effects import warp as _warp
from sashimono.effects.definition import EffectDefinition, EffectRegistry, Pieces, registry
from sashimono.effects.spec import (
    CheckSpec,
    ColorSpec,
    FileSpec,
    FontSpec,
    FontStyleSpec,
    GridSpec,
    ParameterGroup,
    ParameterKind,
    ParameterSpec,
    ParamInput,
    SelectSpec,
    TextSpec,
    TrackSpec,
    ValueSpec,
)

_builtin.register_builtin_effects()
_motion.register_motion_effects()
_stylize.register_stylize_effects()
_paint.register_paint_effects()
_optics.register_optics_effects()
_grading.register_grading_effects()
_warp.register_warp_effects()
_spawn.register_spawn_effects()
_region.register_region_effects()
_audio.register_audio_effects()

__all__ = [
    "CheckSpec",
    "ColorSpec",
    "EffectDefinition",
    "EffectRegistry",
    "FileSpec",
    "FontSpec",
    "FontStyleSpec",
    "GridSpec",
    "ParamInput",
    "ParameterGroup",
    "ParameterKind",
    "ParameterSpec",
    "Pieces",
    "SelectSpec",
    "TextSpec",
    "TrackSpec",
    "ValueSpec",
    "registry",
]
