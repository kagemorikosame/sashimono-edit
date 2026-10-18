"""編集の方針（#343）に照らして、方針の外の書体・色・効果を使うクリップを探す

AI の道具 ``check_style_guide`` の中身 画面の側に置くと Qt を知らない試験と AI から使えず、
コア層に置くとエフェクトの定義（項目の既定の色）を読めないので、AI の層に置く

照らすのは形のある 3 つ（書体・色・効果）だけ 字幕の見た目・テンポ・そのほかは文なので、
AI が ``get_style_guide`` で読んで自分で照らす

返すのは書体の名前・色・効果の種類とクリップの在りかだけで、テキストの本文も方針の文も
返さない（字幕と文字を送らないと決めた人にも使える #342）
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

from sashimono.core.model import (
    Clip,
    Effect,
    ParamValue,
    Project,
    StyleGuide,
    Timeline,
    Track,
    valid_style_color,
)
from sashimono.effects.fills import is_gradient
from sashimono.effects.sources import SHAPE, TEXT, SourceDefinition
from sashimono.effects.spec import ColorSpec, GradientSpec
from sashimono.effects.strokes import STROKE

__all__ = [
    "COLOR",
    "COLOR_TOLERANCE",
    "DEFAULT_FONT",
    "EFFECT",
    "FONT",
    "StyleProblem",
    "color_hex",
    "style_problems",
]

#: 食い違いの種類 値は AI へ返す文字そのまま
FONT = "font"
COLOR = "color"
EFFECT = "effect"

#: 書体を選んでいないテキスト（アプリの既定の書体で描く）を返すときの名前
DEFAULT_FONT = "（既定の書体）"

#: 同じ色と見なす差（0〜255 の各成分） 画面の色の窓と 0..1 の値の行き来で 1 ずれることがある
#: ちょうど一致だけを見ると、方針の色で塗った字まで「方針の外」と返す
COLOR_TOLERANCE = 2

#: 塗りの項目の頭 字と図形と縁取りの層はどれも同じ名前（``effects/sources.py`` ``strokes.py``）
_FILL = "fill"
#: グラデーションの止まり 1 つの長さ（位置・R・G・B・A）
_STOP = 5


@dataclass(frozen=True, slots=True)
class StyleProblem:
    """方針の外の物 1 つ"""

    kind: str
    clip_id: str | None
    track_id: str
    #: シーンの中なら、そのシーンの ID メインなら ``None``
    scene_id: str | None
    start_frame: int
    #: 方針の外だった値（書体の名前・``#rrggbb``・エフェクトの種類）
    value: str
    #: クリップの中のどこか（字の塗り・縁取りの層 2 など）
    where: str

    def row(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "clip_id": self.clip_id,
            "track_id": self.track_id,
            "scene_id": self.scene_id,
            "start_frame": self.start_frame,
            "value": self.value,
            "where": self.where,
        }


def color_hex(value: object) -> str | None:
    """sRGB の 0..1 の色を ``#rrggbb`` へ 色でなければ ``None``（透明度は見ない）"""
    if not isinstance(value, tuple) or len(value) < 3:
        return None
    parts = [round(min(max(float(part), 0.0), 1.0) * 255) for part in value[:3]]
    return "#" + "".join(f"{part:02x}" for part in parts)


def _close(color: str, allowed: tuple[str, ...]) -> bool:
    def rgb(text: str) -> tuple[int, int, int]:
        return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)

    mine = rgb(color)
    return any(
        all(abs(a - b) <= COLOR_TOLERANCE for a, b in zip(mine, rgb(other), strict=True))
        for other in allowed
    )


def _default(definition: SourceDefinition, name: str) -> ParamValue | None:
    """項目の既定 保存の形は既定のままの項目を書かないことがあり、無ければ既定で描かれる"""
    for spec in definition.parameters:
        if spec.name == name and isinstance(spec, ColorSpec | GradientSpec):
            return spec.default_value()
    return None


def _fill_colors(params: Mapping[str, ParamValue], definition: SourceDefinition) -> list[str]:
    """塗りの色 単色なら ``color``、グラデーションなら止まりの色を全部"""
    if is_gradient(params, _FILL):
        stops = params.get(f"{_FILL}_gradient", _default(definition, f"{_FILL}_gradient"))
        if not isinstance(stops, tuple):
            return []
        found = [
            color_hex(tuple(stops[index + 1 : index + 4]))
            for index in range(0, len(stops) - _STOP + 1, _STOP)
        ]
        return [color for color in found if color is not None]
    color = color_hex(params.get("color", _default(definition, "color")))
    return [color] if color is not None else []


def _timelines(project: Project) -> Iterator[tuple[str | None, Timeline]]:
    yield None, project.timeline
    for scene in project.scenes:
        yield str(scene.id), scene.timeline


def _effects_of(clip: Clip) -> Iterator[tuple[Effect, str]]:
    """クリップに掛かるエフェクト 描画と音声の欄（``fixed``）は全部のクリップが持つので数えない
    切ってあるエフェクトは絵にも音にも出ないので数えない"""
    for effect in (*clip.effects, *clip.after_effects):
        if not effect.fixed and effect.enabled:
            yield effect, "エフェクト"
    if clip.source is not None:
        for number, stroke in enumerate(clip.source.strokes, start=1):
            for effect in stroke.effects:
                if not effect.fixed and effect.enabled:
                    yield effect, f"縁取りの層 {number} の効果"


def _clip_problems(guide: StyleGuide, clip: Clip) -> Iterator[tuple[str, str, str]]:
    """（種類, 値, どこ）"""
    source = clip.source
    if source is not None and source.kind == TEXT.kind and guide.fonts:
        font = source.params.get("font", "")
        name = font if isinstance(font, str) and font else DEFAULT_FONT
        if name not in guide.fonts:
            yield FONT, name, "書体"
    if source is not None and source.kind in (TEXT.kind, SHAPE.kind) and guide.colors:
        definition = TEXT if source.kind == TEXT.kind else SHAPE
        for color in _fill_colors(source.params, definition):
            if not _close(color, guide.colors):
                yield COLOR, color, "字の塗り" if definition is TEXT else "図形の塗り"
        for number, stroke in enumerate(source.strokes, start=1):
            if not stroke.enabled:
                continue
            for color in _fill_colors(stroke.params, STROKE):
                if not _close(color, guide.colors):
                    yield COLOR, color, f"縁取りの層 {number}"
    if guide.effects:
        for effect, where in _effects_of(clip):
            if effect.kind not in guide.effects:
                yield EFFECT, effect.kind, where


def _track_problems(guide: StyleGuide, track: Track) -> Iterator[tuple[str, str, str]]:
    if not guide.effects:
        return
    for effect in track.effects:
        if not effect.fixed and effect.enabled and effect.kind not in guide.effects:
            yield EFFECT, effect.kind, "トラックのエフェクト"


def style_problems(project: Project) -> list[StyleProblem]:
    """方針の外の書体・色・効果 メインからシーンの順、トラックとクリップは並びの順

    方針の空の欄（決めていない物）は照らさない 全部空なら何も返さない
    同じクリップで同じ値が何か所かに出たら（縁取りの層が 2 つとも同じ色など）、場所ごとに返す
    """
    guide = project.style_guide.cleaned()
    # 保存の形は色を確かめずに読む（手で書き換えたファイル） 読めない色で照らすと数に直せずに落ちる
    guide = replace(guide, colors=tuple(c for c in guide.colors if valid_style_color(c)))
    found: list[StyleProblem] = []
    for scene_id, timeline in _timelines(project):
        for track in timeline.tracks:
            for kind, value, where in _track_problems(guide, track):
                found.append(StyleProblem(kind, None, str(track.id), scene_id, 0, value, where))
            for clip in track.clips:
                for kind, value, where in _clip_problems(guide, clip):
                    found.append(
                        StyleProblem(
                            kind,
                            str(clip.id),
                            str(track.id),
                            scene_id,
                            clip.timeline_start,
                            value,
                            where,
                        )
                    )
    return found
