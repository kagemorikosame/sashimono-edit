"""字幕を焼き込むときに選ぶ見た目（#308）

字幕をテキストとしてタイムラインへ置くとき、見た目の出どころを 5 つから選べる

* 既定の見た目（大きさ・下寄せ・縁取り・折り返し 今まで通り）
* タイムラインで選んでいるテキスト（今まで通り クリップを丸ごと写して本文だけ差し替える）
* 自作のエイリアス（文字を持つ物だけ）
* プリセット（テキストの中身を持つ物と、エフェクトだけの物）
* テンプレートの棚（AviUtl のエイリアス・YMM4 のテンプレート）

エイリアスと棚は、棚の〔選択中のクリップに適用〕と同じ決まり
（:func:`~sashimono.compat.catalog.restyle`）で、既定の見た目で置いた字幕に着せる 今の文字と
長さを残し、動きと縁取りの層を字幕の尺に合わせる プリセットは設定パネルから当てるのと同じ決まり
（:func:`~sashimono.core.commands.preset.preset_commands`） 決まりを 2 か所に書くと、
片方だけ直したときに、置いてから着せ替えた字幕と、選んで置いた字幕が食い違う

ここに置くのは、棚の着せ替えが互換層にあり、コア層からは読めないため 画面の窓と AI の道具の
両方から使う（画面だけに書くと、AI が同じ見た目で置けない）
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from sashimono.compat.aviutl.custom_object import custom_object_script
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import TemplateEntry, TemplateError, restyle, wraps
from sashimono.compat.mapped import MappedObject
from sashimono.core.commands import Command, SubtitleLook, SubtitleTemplate
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.commands.preset import PresetOptions, preset_commands
from sashimono.core.io.aliases import Alias
from sashimono.core.io.library import ALIAS as LIBRARY_ALIAS
from sashimono.core.io.library import PRESET as LIBRARY_PRESET
from sashimono.core.io.library import Library
from sashimono.core.io.presets import Preset
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    GeneratedSource,
    ParamValue,
    Project,
    Track,
    TrackKind,
)
from sashimono.effects.sources import TO_NATIVE_LAYOUT, source_registry

__all__ = [
    "ALIAS",
    "DEFAULT",
    "NO_TEXT",
    "PRESET",
    "SELECTED",
    "TEMPLATE",
    "LookChoice",
    "LookError",
    "LookRules",
    "alias_choice",
    "default_choice",
    "find_choice",
    "objects_look",
    "preset_choice",
    "selected_choice",
    "subtitle_look",
    "template_choice",
    "template_refusal",
]

#: 見た目の出どころ
DEFAULT = "default"
SELECTED = "selected"
PRESET = "preset"
ALIAS = "alias"
TEMPLATE = "template"

#: 字幕の本文を入れるテキストの項目
_TEXT = "text"

#: 文字を持たない見た目を断る理由 一覧で灰色にした項目の説明に出す
NO_TEXT = "文字を持たないので字幕の見た目にできない（図形・スクリプトのオブジェクトなど）"


class LookError(Exception):
    """選んだ見た目で字幕を作れない（棚の配布物が読めない・文字を持たない など）"""


@dataclass(frozen=True, slots=True, eq=False)
class LookChoice:
    """選べる見た目 1 つ

    ``key`` は次に開いたときに同じ物を選び直すための鍵（設定に覚える） ``refusal`` が
    空でなければ選べない（一覧では灰色にして理由を添える）
    """

    key: str
    label: str
    kind: str
    item: Preset | Alias | TemplateEntry | Clip | None = None
    refusal: str = ""


def default_choice() -> LookChoice:
    return LookChoice(DEFAULT, "既定の見た目", DEFAULT)


def selected_choice(clip: Clip) -> LookChoice:
    """タイムラインで選んでいるテキスト 鍵は 1 つだけ（選ぶテキストは開くたびに違う）"""
    head = ""
    if clip.source is not None:
        lines = str(clip.source.params.get(_TEXT, "")).splitlines()
        head = lines[0][:12] if lines else ""
    return LookChoice(SELECTED, f"選んでいるテキスト「{head}」", SELECTED, clip)


def preset_choice(preset: Preset) -> LookChoice:
    refusal = ""
    if preset.source is not None and preset.source.kind != "text":
        refusal = NO_TEXT
    elif not preset.has_look:
        refusal = "当てる見た目を持たない"
    return LookChoice(
        f"{PRESET}:{preset.category}/{preset.name}", preset.name, PRESET, preset, refusal
    )


def alias_choice(alias: Alias) -> LookChoice:
    source = alias.clip.source
    refusal = "" if source is not None and source.kind == "text" else NO_TEXT
    return LookChoice(f"{ALIAS}:{alias.name}", alias.name, ALIAS, alias, refusal)


def template_choice(entry: TemplateEntry) -> LookChoice:
    """棚の 1 本 中身を読まないと文字を持つか分からないので、断るのは読めなかった物だけ"""
    # 鍵は置き場と何本目か 中身の指紋にすると、配布物を直すたびに覚えた選択が外れる
    where = hashlib.sha256(f"{entry.path}#{entry.index}".encode()).hexdigest()[:16]
    return LookChoice(f"{TEMPLATE}:{where}", entry.name, TEMPLATE, entry, entry.error)


def find_choice(library: Library, kind: str, name: str, category: str = "") -> LookChoice:
    """保存したプリセットかエイリアスを名前で探す（AI の道具が使う）

    見つからない・プリセットの名前が分類違いで重なるときは :class:`LookError` 重なった物の
    どちらかを黙って選ぶと、頼んだのと違う見た目で何十枚も置く
    """
    if kind == ALIAS:
        aliases = [e.item for e in library.entries(LIBRARY_ALIAS) if isinstance(e.item, Alias)]
        found_alias = next((alias for alias in aliases if alias.name == name), None)
        if found_alias is None:
            names = "、".join(alias.name for alias in aliases) or "（無い）"
            raise LookError(f"エイリアス「{name}」が無い 保存してある物: {names}")
        return alias_choice(found_alias)
    if kind != PRESET:
        raise LookError(f"見た目の種類は preset か alias: {kind}")
    presets = [e.item for e in library.entries(LIBRARY_PRESET) if isinstance(e.item, Preset)]
    matches = [p for p in presets if p.name == name and (not category or p.category == category)]
    if not matches:
        names = "、".join(f"{p.category}/{p.name}" for p in presets) or "（無い）"
        raise LookError(f"プリセット「{name}」が無い 保存してある物（分類/名前）: {names}")
    if len(matches) > 1:
        kinds = "、".join(p.category for p in matches)
        raise LookError(
            f"プリセット「{name}」は分類違いで {len(matches)} つある 分類を指定する: {kinds}"
        )
    return preset_choice(matches[0])


def template_refusal(objects: list[MappedObject]) -> str:
    """棚の配布物を字幕に着せられない理由 着せられるなら空"""
    if not objects:
        return "置ける物が無い"
    if not any(item.has_picture for item in objects):
        # エフェクトだけの物（YMM4 のアニメーション効果） 既定の見た目の字幕に足す
        return ""
    if any(
        inner.clip.source is not None and inner.clip.source.kind == "text"
        for item in objects
        for inner in item.walk()
    ):
        return ""
    return NO_TEXT


@dataclass(frozen=True, slots=True)
class LookRules:
    """着せ方の本人の設定 画面の窓と AI の道具で同じ物を渡す（頼み方で見た目を変えない）"""

    #: プリセットの当て方（``Preferences.preset_options``） 文字は当てる側にしても当てない
    #: 当てると、字幕がどれもプリセットの文字になる
    preset_options: PresetOptions = field(default_factory=PresetOptions)
    #: 棚の着せ替えで今の折り返しの幅を残す（``Preferences.restyle_keep_wrap``）
    keep_wrap: bool = False


def subtitle_look(
    choice: LookChoice, base: GeneratedSource, rules: LookRules | None = None
) -> SubtitleTemplate:
    """``choice`` の見た目で字幕を作るひな形 ``base`` は既定の見た目（折り返しの幅も入れた物）

    使えない物は :class:`LookError` 棚の配布物はここで読む（選ぶ前に全部を読むと、
    数百本の棚で窓が開くのが遅れる）
    """
    chosen = rules if rules is not None else LookRules()
    keep_wrap = chosen.keep_wrap
    if choice.refusal:
        raise LookError(f"「{choice.label}」は使えない: {choice.refusal}")
    item = choice.item
    if choice.kind == DEFAULT or item is None:
        return base
    if isinstance(item, Clip):
        return item
    if isinstance(item, Preset):
        options = replace(chosen.preset_options, with_text=False)
        preset = item

        def by_preset(clip: Clip) -> list[Command]:
            return preset_commands(
                preset,
                clip,
                options=options,
                # 設定パネルから当てるのと同じく、中身の土台は種類の既定の値にする
                # 当てる先（既定の見た目）を土台にすると、プリセットが持たない項目に
                # 字幕の既定の大きさや縁が残り、保存した見た目と違って出る
                defaults_of=_source_defaults,
                body_of=custom_object_script,
            )

        return _dresser(base, by_preset)
    if isinstance(item, Alias):
        objects = [MappedObject(clip=item.instantiate(0), layer=1, kind="text")]
    else:
        try:
            # 棚の注意書きは字幕の窓では見せない 棚の窓の報告に紛れないよう自前の入れ物へ
            objects = item.load(report=CompatibilityReport())
        except (*TemplateError, OSError, ValueError) as exc:
            raise LookError(f"「{choice.label}」を読み込めない: {exc}") from exc
        refusal = template_refusal(objects)
        if refusal:
            raise LookError(f"「{choice.label}」は使えない: {refusal}")

    return objects_look(objects, base, keep_wrap=keep_wrap)


def objects_look(
    objects: list[MappedObject], base: GeneratedSource, *, keep_wrap: bool = False
) -> SubtitleLook:
    """エイリアスや棚の配布物を写した物を、棚の着せ替え（``restyle``）で字幕に着せる手"""

    def by_restyle(clip: Clip) -> list[Command]:
        return restyle(objects, clip, keep_wrap=keep_wrap)

    return _dresser(base, by_restyle)


def _source_defaults(kind: str) -> dict[str, ParamValue] | None:
    definition = source_registry.get(kind)
    return definition.default_params() if definition is not None else None


def _dresser(base: GeneratedSource, commands_for: Callable[[Clip], list[Command]]) -> SubtitleLook:
    """既定の見た目で置いた字幕 1 枚に、見た目を着せる手

    着せる操作は今あるクリップへのコマンドなので、字幕 1 枚だけの仮のプロジェクトに置いて
    当て、できたクリップを取り出す 字幕ごとに長さが違い、動きの長さ合わせは長さごとに要る
    """

    def make(text: str, duration: int) -> Clip:
        clip = with_fixed_items(
            Clip(
                timeline_start=0,
                duration=max(1, duration),
                source=base.with_param(_TEXT, text),
            ),
            picture=True,
        )
        project = Project.create()
        track = Track(TrackKind.VIDEO, "字幕", (clip,))
        project = project.with_timeline(replace(project.timeline, tracks=(track,)))
        for command in commands_for(clip):
            project = command.apply(project)
        located = project.timeline.locate_clip(clip.id)
        assert located is not None
        return _wrapped(located[1], base)

    return make


def _wrapped(clip: Clip, base: GeneratedSource) -> Clip:
    """選んだ見た目が折り返さないとき、字幕の折り返し（本人の設定 #249）を足す

    配布物の多くは折り返しの幅を持たない そのまま置くと、長い字幕が画面の端で切れる（#256）
    字幕の折り返しを切っている人（幅 0）には足さない AviUtl2 の組み方は折り返さない（#301）
    ので、設定パネルの〔組み方を「標準」にする〕と同じ値で標準の組み方へ替える
    """
    width = base.params.get("wrap_width")
    source = clip.source
    if not isinstance(width, AnimatedValue) or not wraps(width):
        return clip
    if source is None or source.kind != "text":
        return clip
    own = source.params.get("wrap_width")
    # 見た目が自分の幅を持てばそのまま使う エフェクトだけの見た目は中身に触らないので、
    # 既定の見た目の幅（字幕の折り返し）のまま来る これは足した物として拡大に合わせ直す
    if wraps(own) and own != width:
        return clip
    params = dict(source.params)
    # 折り返しの幅は字を組むときの画素で、配置の拡大は組んだ後に掛かる 拡大する見た目
    # （実物ではドット絵風加工の 120%）に画面の幅の 90% をそのまま入れると、拡大した字幕が
    # 画面の幅を超えて端で切れる 拡大のいちばん大きい所で画面に収まる幅にする
    widening = _widening(clip)
    params["wrap_width"] = replace(width, static=width.static / widening) if widening > 1 else width
    vertical = params.get("vertical")
    upright = vertical is True or (isinstance(vertical, int) and bool(vertical))
    if params.get("layout") == "aviutl" and not upright:
        params.update(TO_NATIVE_LAYOUT.switch(params))
    return replace(clip, source=replace(source, params=params))


#: 配置のエフェクト（クリップの欄と、見た目が足した物の両方）
_TRANSFORM = "transform"


def _widening(clip: Clip) -> float:
    """配置が字幕を横へ広げる倍率 欄と足した配置の拡大率と横の拡大率を掛け合わせる

    キーフレームで動く物はいちばん大きい値を取る（動きの途中で画面からはみ出さないように）
    """
    total = 1.0
    for effect in clip.effects:
        if effect.kind != _TRANSFORM or not effect.enabled:
            continue
        for name in ("scale", "scale_x"):
            total *= _largest(effect.params.get(name)) / 100.0
    return total


def _largest(value: ParamValue | None) -> float:
    # キーフレームがあれば :attr:`AnimatedValue.static` は使われない 裏返しの負の値は大きさで見る
    if isinstance(value, AnimatedValue):
        if value.keyframes:
            return max(abs(k.value) for k in value.keyframes)
        return abs(value.static)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return abs(float(value))
    return 100.0
