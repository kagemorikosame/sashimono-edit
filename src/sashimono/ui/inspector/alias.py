"""設定パネルの欄を、作り直さずに同じ構成の別のクリップへ向け直すための ID の入れ替え（#431）

設定パネルの欄は、作ったときのクリップ・エフェクト・縁取りの層の ID を覚えている
（値を変える命令の行き先 :class:`~sashimono.core.commands.ParamPath` など） 選び替えの
たびに作り直すと、テキストだけのクリップでも 1 回 50 ms、エフェクトが 15 個で 200 ms
ほど掛かり、クリックで選ぶだけで引っ掛かった

欄の構成（エフェクトの種類と並び）が同じなら、欄はそのままで中身だけ入れ替えられる
ここでは「欄を作ったクリップ（元）」と「今選んでいるクリップ（今）」の ID を組にして
入れ替える 設定パネルは入れ替えたプロジェクト（今のクリップが元の ID を名乗る）を見て、
欄から出ていく命令や在りかは同じ入れ替えで今の ID へ戻す 入れ替えは自分自身の逆
（2 度当てると元に戻る）なので、入るときも出るときも同じ :meth:`IdSwap.apply` を使う

組にするのはクリップの ID・エフェクト（前の場面と後の場面）の ID・縁取りの層と層の
エフェクトの ID 並びの何番目どうしで組む 数が違えば組まない（構成が違うので作り直す）
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar, cast

from sashimono.core.commands.base import Command
from sashimono.core.model import Clip, Effect, Project, Stroke

__all__ = ["IdSwap"]

_T = TypeVar("_T")


def _stack_pairs(
    built: tuple[Effect, ...], current: tuple[Effect, ...]
) -> Iterator[tuple[str, str]]:
    for mine, theirs in zip(built, current, strict=True):
        yield mine.id, theirs.id


def _strokes(clip: Clip) -> tuple[Stroke, ...]:
    return clip.source.strokes if clip.source is not None else ()


@dataclass(frozen=True, slots=True)
class IdSwap:
    """元の ID と今の ID を入れ替える組 両向きに持つ 空なら何もしない"""

    pairs: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def between(cls, built: Clip, current: Clip) -> IdSwap | None:
        """``built`` の欄を ``current`` へ向け直す組 並びの数が違えば ``None``

        種類まで同じかはここでは見ない 種類の違いは欄の構成の指紋
        （:meth:`InspectorPanel._layout_key`）が見分けて作り直す
        """
        if built.id == current.id:
            return cls()
        if (
            len(built.effects) != len(current.effects)
            or len(built.after_effects) != len(current.after_effects)
            or len(_strokes(built)) != len(_strokes(current))
        ):
            return None
        pairs: list[tuple[str, str]] = [(built.id, current.id)]
        pairs.extend(_stack_pairs(built.effects, current.effects))
        pairs.extend(_stack_pairs(built.after_effects, current.after_effects))
        for held, other in zip(_strokes(built), _strokes(current), strict=True):
            if len(held.effects) != len(other.effects):
                return None
            pairs.append((held.id, other.id))
            pairs.extend(_stack_pairs(held.effects, other.effects))
        both: dict[str, str] = {}
        for mine, theirs in pairs:
            if mine == theirs:
                continue
            both[mine] = theirs
            both[theirs] = mine
        return cls(both)

    @property
    def active(self) -> bool:
        return bool(self.pairs)

    def project(self, project: Project) -> Project:
        """組の ID を持つクリップの ID を入れ替えたプロジェクト ほかのクリップはそのまま

        クリップごと入れ替えるので、元のクリップが残っていれば今の ID を名乗る（どちらの
        ID も 1 本ずつのまま） 元のクリップが消えていれば、今のクリップだけが元の ID になる
        """
        if not self.active:
            return project
        timeline = project.timeline
        for track in timeline.tracks:
            if not any(clip.id in self.pairs for clip in track.clips):
                continue
            clips = tuple(
                self.apply(clip) if clip.id in self.pairs else clip for clip in track.clips
            )
            timeline = timeline.replace_track(track.with_clips(clips))
        return project.with_timeline(timeline)

    def apply(self, value: _T) -> _T:
        """``value`` の中の組の ID を入れ替える 命令・在りか・クリップ・ID の文字のどれでも

        命令はデータクラスの欄を辿って作り直す データクラスでない命令は中が読めず、
        元の ID のまま外へ出ると、選んでいないクリップへ値が入る 見落とさないよう止める
        """
        if not self.active:
            return value
        # 作り直した物は入れた物と同じ型（同じデータクラスか、同じ入れ物）
        return cast("_T", self._swap(value))

    def _swap(self, value: object) -> object:
        if isinstance(value, str):
            # ID は str の NewType 列挙の値（str を継いだ物）は組に入らないのでそのまま
            return self.pairs.get(value, value)
        if isinstance(value, tuple):
            return tuple(self._swap(item) for item in value)
        if isinstance(value, list):
            return [self._swap(item) for item in value]
        if isinstance(value, frozenset):
            return frozenset(self._swap(item) for item in value)
        if isinstance(value, MappingProxyType):
            return MappingProxyType({key: self._swap(item) for key, item in value.items()})
        if isinstance(value, dict):
            return {key: self._swap(item) for key, item in value.items()}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            changes: dict[str, object] = {}
            for each in dataclasses.fields(value):
                if not each.init:
                    continue
                before = getattr(value, each.name)
                after = self._swap(before)
                if after is not before and after != before:
                    changes[each.name] = after
            return dataclasses.replace(value, **changes) if changes else value
        if isinstance(value, Command):
            raise TypeError(f"中を辿れない命令は向け直せない: {type(value).__name__}")
        return value
