"""自分で保存したエイリアス クリップの中身を名前を付けて取っておき、あとで置き直す

AviUtl のエイリアスと同じ使い方をする 作り込んだテロップ（書体・色・縁・動き）を
保存しておけば、右クリックの〔追加〕→〔エイリアス〕から同じ物を置ける

保存するのはクリップの中身だけ 置いた位置・リンク・グループ・字幕の出どころはその場所の
事情なので持たない
長さは持つ（置き直したときに同じ長さで出る方が、作った時の動きと合う）

クリップの書き方はプロジェクトファイルと同じ（:func:`~sashimono.core.io.serialize.clip_to_json`）
別の書き方にすると、クリップに項目が増えるたびに 2 か所を直すことになり、片方だけ
直したときに保存したエイリアスから項目が黙って落ちる クリップの書き方の版
（プロジェクトファイルの版）も一緒に書き、この本体より新しい物は読まない

保存できるのは素材を持たないクリップだけ（テキスト・図形・フィルタ・場面切り替えなど）
素材を持つクリップはその素材がプロジェクトに入っていないと置けず、素材の道は本人の
機械にしか無い 道だけを持ち回ると、別のプロジェクトで置いたときに素材の読み直しと
登録が要り、見つからなければ何も映らないクリップができる シーンも同じで、
中身は保存したプロジェクトにしか無い
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sashimono.core import userdirs
from sashimono.core.io.serialize import (
    FORMAT_VERSION as CLIP_FORMAT_VERSION,
)
from sashimono.core.io.serialize import (
    ProjectFileError,
    clip_from_json,
    clip_to_json,
    json_text,
)
from sashimono.core.model import Clip, Effect, new_clip_id

__all__ = [
    "DEFAULT_CATEGORY",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "SUFFIX",
    "Alias",
    "AliasStore",
    "alias_refusal",
    "default_alias_root",
]

FORMAT_NAME = "sashimono-alias"
FORMAT_VERSION = 1
#: 分類の既定 プリセットの既定（``ユーザー``）とそろえ、管理の画面で 2 つを同じ並びで見せる
DEFAULT_CATEGORY = "ユーザー"
SUFFIX = ".smea"

#: ファイル名に使えない文字 Windows の制限に合わせる（プリセットと同じ）
_UNSAFE_CHARACTERS = '<>:"/\\|?*'

#: Windows が機器の名前として取っておく名前 大文字と小文字は区別しない
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(10)}
    | {f"LPT{number}" for number in range(10)}
)


def alias_refusal(clip: Clip) -> str | None:
    """このクリップをエイリアスにできない理由 できるなら ``None``"""
    if clip.media_id is not None:
        return "素材を使うクリップはエイリアスにできません（テキストや図形などだけ）"
    if clip.scene_id is not None:
        return "シーンのクリップはエイリアスにできません"
    if clip.source is None:
        return "中身の無いクリップはエイリアスにできません"
    return None


@dataclass(frozen=True, slots=True)
class Alias:
    """名前付きのクリップの中身 ``clip`` は位置 0 の見本"""

    name: str
    clip: Clip
    #: 分類 管理の画面（#276）のフォルダ分けと、〔追加〕→〔エイリアス〕の並べ方に使う
    #: ファイルの置き場は名前だけで決める（分類で分けない） 名前がエイリアスを指す 1 つの
    #: 鍵で、分類ごとに同じ名前を許すと、置き場と上書きの確かめが食い違う
    category: str = DEFAULT_CATEGORY

    @classmethod
    def of(cls, name: str, clip: Clip, *, category: str = DEFAULT_CATEGORY) -> Alias:
        """クリップから作る 置いた場所の事情（位置・リンク・グループ・字幕の出どころ）は外す

        字幕の出どころ（:attr:`Clip.subtitle_origin`）は、字幕から焼き込んだテキストが
        どの字幕の行から来たかの印 エイリアスは見た目を持ち運ぶ物で、置いた写しは字幕の
        行とつながらない 残すと、写しが素材と一緒にずれる仲間に数えられ、字幕の誤植を
        直すと写しの文字まで書き換わる（#282）
        """
        reason = alias_refusal(clip)
        if reason is not None:
            raise ValueError(reason)
        return cls(
            name=name,
            clip=replace(
                clip, timeline_start=0, link_group=None, group_id=None, subtitle_origin=None
            ),
            category=category,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "clip_version": CLIP_FORMAT_VERSION,
            "name": self.name,
            # 項目を足しただけなので版は上げない 前の版の本体は読み飛ばして置ける
            "category": self.category,
            "clip": clip_to_json(self.clip),
        }

    @classmethod
    def from_dict(cls, data: object) -> Alias:
        if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
            raise ProjectFileError("Sashimono のエイリアスではない")
        version = data.get("version", 0)
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise ProjectFileError(f"新しい形式のエイリアス (version {version})")
        clip_version = data.get("clip_version", 0)
        # 新しい本体が足した項目は、この本体では読み飛ばされる 置き直して上書きすると
        # 消えるので、新しい物は読まずに断る
        if not isinstance(clip_version, int) or clip_version > CLIP_FORMAT_VERSION:
            raise ProjectFileError(f"新しい版で保存したエイリアス (clip_version {clip_version})")
        try:
            clip = clip_from_json(data.get("clip"))
        except (ValueError, TypeError) as exc:
            raise ProjectFileError(f"エイリアスのクリップが読めない: {exc}") from exc
        if alias_refusal(clip) is not None:
            raise ProjectFileError("素材やシーンを使うクリップはエイリアスとして読まない")
        category = data.get("category")
        if not isinstance(category, str) or not category.strip():
            # 前の版が保存した物は分類を持たない 既定の分類にまとめる
            category = DEFAULT_CATEGORY
        return cls(name=str(data.get("name", "無題")), clip=clip, category=category)

    def instantiate(self, at_frame: int = 0) -> Clip:
        """置くためのクリップ クリップとエフェクトの ID を振り直す

        同じエイリアスを 2 回置いたときに ID が重なると、片方を消したつもりで両方を
        探し当てたり、プロジェクトの検査に断られたりする

        字幕の出どころもここで外す :meth:`of` で外すより前の版が保存したエイリアスには
        印が残っていて、読み込んだだけでは消えない（#282）
        """
        clip = self.clip
        return replace(
            clip,
            timeline_start=max(0, at_frame),
            effects=_fresh(clip.effects),
            after_effects=_fresh(clip.after_effects),
            id=new_clip_id(),
            subtitle_origin=None,
        )


def _fresh(effects: tuple[Effect, ...]) -> tuple[Effect, ...]:
    # 固定の印は残す エイリアスはクリップを丸ごと写す物で、置いたクリップは元の
    # クリップと同じ欄を持つ（貼り付けと同じ） プリセットはよそのクリップへ足す物
    # なので外すが、こちらは足し先が無い
    return tuple(
        Effect(
            kind=effect.kind, params=dict(effect.params), enabled=effect.enabled, fixed=effect.fixed
        )
        for effect in effects
    )


def default_alias_root() -> Path:
    """エイリアスを置く既定の場所 作り直せないので ``%APPDATA%`` の側（プリセットと同じ）"""
    return userdirs.config_root() / "aliases"


@dataclass(slots=True)
class AliasStore:
    """エイリアスの読み書き"""

    root: Path = field(default_factory=default_alias_root)

    def path_for(self, name: str) -> Path:
        """名前ごとのファイル 違う名前が同じファイルに落ちないようにする

        ファイル名に使えない文字を置き換えると ``赤:文字`` と ``赤?文字`` が同じ名前になり、
        Windows は大文字と小文字も区別しない 置き換えた・大文字を含む・予約された名前には
        元の名前から作る短い印を添える 添えないと、後から保存した方が前の物を黙って消す
        """
        return self.root / f"{_file_stem(name)}{SUFFIX}"

    def exists(self, name: str) -> bool:
        """同じ名前のエイリアスが保存されているか 上書きの確かめに使う"""
        return self.path_for(name).is_file()

    def save(self, alias: Alias) -> Path:
        target = self.path_for(alias.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        # 一時ファイルへ書いてから差し替える 書いている途中で落ちると、前に保存した
        # 同じ名前のエイリアスまで壊れる
        temporary = target.with_name(target.name + ".writing")
        temporary.write_text(json_text(alias.to_dict()), encoding="utf-8")
        temporary.replace(target)
        return target

    def load(self, path: Path) -> Alias:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ProjectFileError(f"エイリアスを開けない: {path}") from exc
        except UnicodeDecodeError as exc:
            # ValueError の子で JSONDecodeError とは別 ここで変えないと :meth:`all` の
            # 読み飛ばしを抜け、壊れた 1 つで一覧全体が出なくなる
            raise ProjectFileError(f"エイリアスが UTF-8 として読めない: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ProjectFileError(f"エイリアスが JSON として読めない: {path} ({exc})") from exc
        return Alias.from_dict(data)

    def all(self) -> tuple[Alias, ...]:
        """読める物だけを名前の順に 壊れた 1 つで一覧全体が出なくなると困る"""
        if not self.root.is_dir():
            return ()
        found: list[Alias] = []
        for path in sorted(self.root.glob(f"*{SUFFIX}")):
            try:
                found.append(self.load(path))
            except ProjectFileError:
                continue
        return tuple(sorted(found, key=lambda alias: alias.name))

    def delete(self, name: str) -> None:
        self.path_for(name).unlink(missing_ok=True)


def _file_stem(name: str) -> str:
    """ファイル名に使える形へ 元の名前と違う形になったら、元の名前から作る印を添える"""
    cleaned = "".join(
        "_" if character in _UNSAFE_CHARACTERS or ord(character) < 0x20 else character
        for character in name
    )
    cleaned = cleaned.strip().strip(".") or "無題"
    # CON や NUL は拡張子を付けても Windows が機器として扱い、書き込みが失敗する
    # 見るのは最初の点より前（aux.txt も機器の名前） 後ろに印を足しても逃げられないので頭に付ける
    reserved = cleaned.split(".")[0].strip().upper() in _RESERVED_NAMES
    if cleaned == name and name == name.lower() and not reserved:
        return cleaned
    mark = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{'_' if reserved else ''}{cleaned}-{mark}"
