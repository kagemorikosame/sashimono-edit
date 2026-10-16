"""完成物の検査（#337） 黒いコマ・止まった絵・無音・音割れ・音量の基準・字幕のはみ出し・長さ

外のモデルは使わない（#298 の方針） 絵は描いた輝度、音は混ぜた音を numpy で調べる
"""

from sashimono.engine.checks.job import ANALYSIS_WIDTH, CheckJob, CheckRequest, CheckState
from sashimono.engine.checks.loudness import LoudnessMeter
from sashimono.engine.checks.report import (
    HIGH,
    KIND_NAMES,
    LOW,
    MEDIUM,
    CheckReport,
    Problem,
)
from sashimono.engine.checks.scan import KINDS, LOUDNESS_TARGETS, CheckLimits

__all__ = [
    "ANALYSIS_WIDTH",
    "HIGH",
    "KINDS",
    "KIND_NAMES",
    "LOUDNESS_TARGETS",
    "LOW",
    "MEDIUM",
    "CheckJob",
    "CheckLimits",
    "CheckReport",
    "CheckRequest",
    "CheckState",
    "LoudnessMeter",
    "Problem",
]
