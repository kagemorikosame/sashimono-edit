"""プリセット クリップの見た目を名前付きで保存し、ほかのクリップへ当てる

保存するのはクリップの見た目ひとそろい（#275）

* 足したエフェクトの列（場面切り替えの後の場面の列も） 1 つのエフェクトだけでなく
  列ごと持つのは、見た目のほとんどが複数のエフェクトの組み合わせでできているため
  （縁取り + 影 + グロー、など）
* テキストや図形の中身（文字・色・大きさ・縁取り・影など）
* 最初から持つ欄（描画・音声）の値と、不透明度・合成モード・下で切り抜く
* 保存したクリップの長さ 当てる先の長さへキーフレームを伸び縮みさせるのに使う

エイリアス（:mod:`sashimono.core.io.aliases`）は「クリップを置き直す」物、プリセットは
「いまあるクリップに当てる」物 当て方（文字や位置を当てるか）は
:mod:`sashimono.core.commands.preset` が決める

プロジェクトファイルと同じ JSON の形を使う 前の版のプリセット（エフェクトの列だけ）は
項目を足しただけなので、そのまま読めて当てられる
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from sashimono.core import userdirs
from sashimono.core.io.serialize import (
    ProjectFileError,
    effect_from_json,
    effect_to_json,
    json_text,
    source_from_json,
    source_to_json,
)
from sashimono.core.model import AnimatedValue, Clip, Effect, GeneratedSource, ParamValue

__all__ = [
    "FORMAT_NAME",
    "LEGACY_FORMAT_NAMES",
    "LEGACY_SUFFIXES",
    "SUFFIX",
    "TRASH_FOLDER",
    "Preset",
    "PresetStore",
    "default_preset_root",
]

FORMAT_NAME = "sashimono-preset"
FORMAT_VERSION = 1
SUFFIX = ".smep"

#: 読むときだけ受け付ける、昔の名前と拡張子 書くときは常に新しい名前で書く
#: プリセットは作り直せない物なので、改名前に保存したものが一覧から消えると、
#: 本人には「プリセットが全部消えた」に見える
# 旧名を残す: ここから（名前の一括置換でも書き換えない 古い版のファイルを読むのに要る）
LEGACY_FORMAT_NAMES = ("kumiki-preset", "novaedit-preset")
LEGACY_SUFFIXES = (".kmkp", ".nvpreset")
# 旧名を残す: ここまで

#: 置き場の中のごみ箱 管理の画面（#276）で消した物をここへ移し、戻せるようにする
#: 置き場の中に置くのは、置き場を別の所へ移したり試験で差し替えたりしても、ごみ箱が
#: 一緒に付いて行くため 点で始まる名前は分類の名前に使えない（:func:`_safe_name` が外す）
TRASH_FOLDER = ".trash"

#: ファイル名に使えない文字 Windows の制限に合わせる
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


#: クリップ自身の値を書くときの入れ物の種類 書き方はテキストの中身と同じ ``params`` の辞書
#: 不透明度はキーフレームを持つ動く値なので、エフェクトの値と同じ書き方にそろえる
_CLIP_VALUES = "clip"


@dataclass(frozen=True, slots=True)
class Preset:
    """名前付きの見た目 :meth:`capture` でクリップから作る"""

    name: str
    #: 足したエフェクトの列（最初から持つ欄は :attr:`fixed` に分けて持つ）
    effects: tuple[Effect, ...] = ()
    #: テキストや図形のプリセットではここに中身が入る
    source: GeneratedSource | None = None
    #: 分類 UI のフォルダ分けに使う
    category: str = "ユーザー"
    #: 場面切り替えの後の場面に足したエフェクト
    after_effects: tuple[Effect, ...] = ()
    #: 最初から持つ欄（描画・音声）の値 当てるときは当てる先の同じ欄へ値を写す
    fixed: tuple[Effect, ...] = ()
    #: 不透明度 絵を描かないクリップから保存したときは ``None``（当てても触らない）
    opacity: AnimatedValue | None = None
    #: 合成モード ``None`` は上と同じく触らない
    blend_mode: str | None = None
    #: 下のクリップで切り抜くか ``None`` は上と同じく触らない
    clip_to_below: bool | None = None
    #: 保存したクリップの長さ（フレーム） キーフレームを当てる先の長さへ合わせるのに使う
    #: 前の版のプリセットは持たない（``None``） そのときは合わせずにそのまま当てる
    span: int | None = None

    def __post_init__(self) -> None:
        # 足したエフェクトの列には固定の印（クリップが最初から持つ項目）を持たせない
        # 印のまま足すと、当てるたびに外せないエフェクトが増えていく 保存する側で外し
        # 忘れても、ここを通れば外れる 欄の値は :attr:`fixed` に分けて持つ
        for name in ("effects", "after_effects"):
            effects: tuple[Effect, ...] = getattr(self, name)
            if any(effect.fixed for effect in effects):
                object.__setattr__(
                    self, name, tuple(replace(effect, fixed=False) for effect in effects)
                )
        if not all(effect.fixed for effect in self.fixed):
            # 逆に欄の値には印を付けておく 読み直したときに足したエフェクトと見分けるため
            object.__setattr__(
                self, "fixed", tuple(replace(effect, fixed=True) for effect in self.fixed)
            )

    @classmethod
    def capture(
        cls, name: str, clip: Clip, *, picture: bool = True, category: str = "ユーザー"
    ) -> Preset:
        """``clip`` の見た目を写したプリセット

        ``picture`` は絵を描くクリップか 偽なら不透明度・合成モード・切り抜きを持たない
        音だけのクリップの不透明度は既定のまま意味が無く、持つと絵のクリップへ当てたときに
        向こうの不透明度を 100% へ戻してしまう
        """
        return cls(
            name=name,
            effects=tuple(effect for effect in clip.effects if not effect.fixed),
            source=clip.source,
            category=category,
            after_effects=tuple(effect for effect in clip.after_effects if not effect.fixed),
            fixed=tuple(effect for effect in clip.effects if effect.fixed),
            opacity=clip.opacity if picture else None,
            blend_mode=clip.blend_mode if picture else None,
            clip_to_below=clip.clip_to_below if picture else None,
            span=clip.duration,
        )

    @property
    def has_look(self) -> bool:
        """当てられる中身を 1 つでも持つか 何も持たない物は保存しても当てて何も変わらない"""
        return bool(
            self.effects
            or self.after_effects
            or self.fixed
            or self.source is not None
            or self.opacity is not None
            or self.blend_mode is not None
            or self.clip_to_below is not None
        )

    def to_dict(self) -> dict[str, object]:
        values: dict[str, ParamValue] = {}
        if self.opacity is not None:
            values["opacity"] = self.opacity
        if self.blend_mode is not None:
            values["blend_mode"] = self.blend_mode
        if self.clip_to_below is not None:
            values["clip_to_below"] = self.clip_to_below
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "name": self.name,
            "category": self.category,
            # 前の版の本体はこの 2 つだけを読む 版の数字を上げずに項目を足したので、
            # 前の版でも足したエフェクトだけは当てられる（上げると一覧から消える）
            "effects": [effect_to_json(effect) for effect in self.effects],
            "source": source_to_json(self.source) if self.source is not None else None,
            "after_effects": [effect_to_json(effect) for effect in self.after_effects],
            "fixed": [effect_to_json(effect) for effect in self.fixed],
            "clip": source_to_json(GeneratedSource(kind=_CLIP_VALUES, params=values))
            if values
            else None,
            "span": self.span,
        }

    @classmethod
    def from_dict(cls, data: object) -> Preset:
        if not isinstance(data, dict):
            raise ProjectFileError("プリセットがオブジェクトではない")
        if data.get("format") not in (FORMAT_NAME, *LEGACY_FORMAT_NAMES):
            raise ProjectFileError("Sashimono のプリセットではない")
        version = data.get("version", 0)
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise ProjectFileError(f"新しい形式のプリセット (version {version})")

        source_raw = data.get("source")
        clip_raw = data.get("clip")
        values = source_from_json(clip_raw).params if clip_raw is not None else {}
        opacity = values.get("opacity")
        blend_mode = values.get("blend_mode")
        clip_to_below = values.get("clip_to_below")
        span = data.get("span")
        if span is not None and (not isinstance(span, int) or isinstance(span, bool) or span < 1):
            raise ProjectFileError(f"span が 1 以上の整数ではない: {span!r}")

        return cls(
            name=str(data.get("name", "無題")),
            effects=_effects(data, "effects"),
            source=source_from_json(source_raw) if source_raw is not None else None,
            category=str(data.get("category", "ユーザー")),
            after_effects=_effects(data, "after_effects"),
            fixed=_effects(data, "fixed"),
            # 型の違う値は持たない（触らない）扱いにする 手で書き換えたファイルの 1 項目の
            # 誤りで、プリセットごと一覧から消すより、ほかの中身を当てられる方がよい
            opacity=opacity if isinstance(opacity, AnimatedValue) else None,
            blend_mode=blend_mode if isinstance(blend_mode, str) else None,
            clip_to_below=clip_to_below if isinstance(clip_to_below, bool) else None,
            span=span,
        )

    def instantiate(self) -> tuple[Effect, ...]:
        """このプリセットを適用するためのエフェクト列を返す

        ID を振り直す 同じプリセットを 2 回適用したときに ID が衝突すると、
        片方を消したつもりで両方消える 固定の印も付けない（:meth:`__post_init__` と同じ理由）
        """
        return tuple(
            Effect(kind=effect.kind, params=dict(effect.params), enabled=effect.enabled)
            for effect in self.effects
        )


def _effects(data: dict[str, object], key: str) -> tuple[Effect, ...]:
    """エフェクトの列を読む 項目が無い（前の版のプリセット）ときは空"""
    raw = data.get(key, [])
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ProjectFileError(f"{key} が配列ではない")
    return tuple(effect_from_json(item) for item in raw)


def default_preset_root() -> Path:
    """プリセットを置く既定の場所

    キャッシュと違い、消えると作り直せない ``%APPDATA%`` に置く
    """
    return userdirs.config_root() / "presets"


@dataclass(slots=True)
class PresetStore:
    """プリセットの読み書き"""

    root: Path = field(default_factory=default_preset_root)

    def path_for(self, preset: Preset) -> Path:
        # 分類もファイル名と同じ決まりで整える 管理の画面（#276）から打った分類に ``..`` や
        # ``/`` が入ると、置き場の外やごみ箱（:data:`TRASH_FOLDER`）へ書いてしまう
        # 前からある ``ユーザー`` は整えても変わらない
        folder = _safe_name(preset.category)
        return self.root / folder / f"{_safe_name(preset.name)}{SUFFIX}"

    def exists(self, preset: Preset) -> bool:
        """同じ分類に同じ名前（同じファイル）のプリセットがもうあるか 旧い拡張子の物も見る

        保存は確かめずに書き換えるので、作り直せないプリセットを黙って消さないよう、
        画面の側が先にこれで上書きしてよいか尋ねる 旧い拡張子の物は書き換えないが、
        新しい方が一覧で勝つので、本人には同じ名前の物が入れ替わったように見える
        """
        current = self.path_for(preset)
        return any(
            path.exists()
            for path in (current, *(current.with_suffix(suffix) for suffix in LEGACY_SUFFIXES))
        )

    def save(self, preset: Preset) -> Path:
        target = self.path_for(preset)
        target.parent.mkdir(parents=True, exist_ok=True)
        # 一時ファイルへ書いてから差し替える プリセットは作り直せないので、
        # 書き込み中に落ちて壊れると手作業で復元することになる
        temporary = target.with_name(target.name + ".writing")
        # 値には読めないバイトを持つ文字が入りうる（json_text を参照）
        temporary.write_text(json_text(preset.to_dict()), encoding="utf-8")
        temporary.replace(target)
        return target

    def load(self, path: Path) -> Preset:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ProjectFileError(f"プリセットを開けない: {path}") from exc
        except UnicodeDecodeError as exc:
            # JSONDecodeError とは別の ValueError 変えないと、管理の画面の読み込み（#276）で
            # 違うファイルを選んだときに、知らせではなく落ちる
            raise ProjectFileError(f"プリセットが UTF-8 として読めない: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ProjectFileError(f"プリセットが JSON として読めない: {path} ({exc})") from exc
        return Preset.from_dict(data)

    def all(self) -> tuple[Preset, ...]:
        """読めるものだけを返す

        壊れた 1 つで一覧全体が出なくなると、他のプリセットまで使えなくなる
        """
        found: list[Preset] = []
        for path in self.files():
            try:
                found.append(self.load(path))
            except ProjectFileError:
                continue
        return tuple(found)

    def delete(self, preset: Preset) -> None:
        # 旧い拡張子の同じ名前も消す 残すと、消したはずのプリセットが一覧に戻ってくる
        current = self.path_for(preset)
        for path in (current, *(current.with_suffix(suffix) for suffix in LEGACY_SUFFIXES)):
            path.unlink(missing_ok=True)

    def files(self) -> list[Path]:
        """一覧に出すファイル 旧い拡張子のものも拾う 管理の画面（#276）も同じ物を並べる

        同じ名前が新旧両方にあるときは新しい方だけを出す 改名前のプリセットを
        上書き保存すると新しい拡張子で書かれ、旧いファイルは残る 両方出すと、
        同じ名前が 2 つ並び、選んだ方によって中身が違う
        """
        if not self.root.is_dir():
            return []
        current = {path for path in self.root.rglob(f"*{SUFFIX}") if not self._trashed(path)}
        legacy = {
            path
            for suffix in LEGACY_SUFFIXES
            for path in self.root.rglob(f"*{suffix}")
            if path.with_suffix(SUFFIX) not in current and not self._trashed(path)
        }
        return sorted(current | legacy)

    def _trashed(self, path: Path) -> bool:
        """ごみ箱（:data:`TRASH_FOLDER`）の中の物か 一覧に出すと、消したはずの物が戻って見える"""
        return TRASH_FOLDER in path.relative_to(self.root).parts


def _safe_name(name: str) -> str:
    """ファイル名に使える形へ 空になったら既定の名前を返す"""
    cleaned = _UNSAFE.sub("_", name).strip().strip(".")
    return cleaned or "無題"
