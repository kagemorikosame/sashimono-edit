"""画素で置いて小さく入った動画を見つけ、画面に合わせる（#494）

置き方の既定（拡大率 100% が素材の画素）は変えない YMM4・AviUtl と値が合わなくなる
1080p の作品に 360p の動画を置くと中央に 1/3 で入るので、知らせと、合わせる道を出す
画面と AI の道具が同じ所を通る
"""

from __future__ import annotations

from sashimono.core.commands.base import Command
from sashimono.core.commands.effects import ParamPath, SetClipProperty, SetParam
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND
from sashimono.core.model import AnimatedValue, Clip, Effect, MediaItem, Project, ProjectSettings

__all__ = ["fit_commands", "placed_small", "small_note", "small_share"]

#: 縦も横も画面のこの割合に届かなければ小さく入ったと読む 720p と 1080p（67%）は知らせ、
#: 上下だけ少し足りない素材（1920x800 など）は横が届くので知らせない
_SMALL_SHARE = 0.9


def small_share(settings: ProjectSettings, media: MediaItem) -> float | None:
    """画素で置いたときに画面の何割になるか 小さくなければ ``None``

    静止画は数えない ロゴや立ち絵は小さく置くのがふつうで、知らせると邪魔になる
    """
    if not media.has_video or media.is_still or not media.video_streams:
        return None
    width, height = media.video_streams[0].display_size
    share = max(width / max(settings.width, 1), height / max(settings.height, 1))
    return share if 0 < share < _SMALL_SHARE else None


def _transform(clip: Clip) -> Effect | None:
    return next((e for e in clip.effects if e.kind == TRANSFORM_EFFECT_KIND and e.fixed), None)


def placed_small(project: Project, clip: Clip) -> float | None:
    """置いたクリップが小さく映っていれば、画面に対する割合

    拡大率を手で上げた物と、拡大率を動かしている物（ズームの演出）は数えない
    """
    if not clip.native_size or clip.media_id is None or not clip.show_picture:
        return None
    media = project.find_media(clip.media_id)
    share = None if media is None else small_share(project.settings, media)
    if share is None:
        return None
    transform = _transform(clip)
    scale = None if transform is None else transform.params.get("scale")
    if isinstance(scale, AnimatedValue):
        if scale.is_animated or scale.static > 100.0:
            return None
        share *= scale.static / 100.0
    return share if share < _SMALL_SHARE else None


def fit_commands(clip: Clip) -> list[Command]:
    """画面に収める命令 画素で置くのをやめて、拡大率を 100% へ戻す

    収めて描く置き方にするので、縁が透明と混ざらず、作品の解像度を変えても画面いっぱいのまま
    """
    commands: list[Command] = [SetClipProperty(clip.id, "native_size", False)]
    transform = _transform(clip)
    scale = None if transform is None else transform.params.get("scale")
    if transform is None or not isinstance(scale, AnimatedValue) or scale.is_animated:
        return commands
    if scale.static != 100.0:
        path = ParamPath.of_effect(clip.id, transform.id, "scale")
        commands.append(SetParam(path, AnimatedValue(static=100.0)))
    return commands


def small_note(share: float) -> str:
    """小さく入った知らせの文の頭"""
    return f"動画が画面の {share * 100:.0f}% の大きさで入っています（画素で置く）"
