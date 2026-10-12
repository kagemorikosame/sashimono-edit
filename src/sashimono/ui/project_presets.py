"""本人が保存したプロジェクト設定のテンプレート（解像度・フレームレート・重ね合わせ）

プロジェクトではなく本人に付く いつも同じ組み合わせで作る人が、毎回数字を打たずに済むため
置き場は画面配置やショートカットと同じ ``%APPDATA%\\Sashimono``（:func:`config_root`）
:class:`~sashimono.ui.workspace.Preferences` とは別のファイルにする 名前付きの一覧で、
項目が 1 つずつ決まっている好みの設定とは形が違う 混ぜると、一覧が壊れたときに
好みの設定まで既定へ戻る
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sashimono.core.commands.edit import MAX_RESOLUTION, MIN_RESOLUTION
from sashimono.core.io.atomic import write_atomic
from sashimono.core.model import Blending
from sashimono.core.timebase import FrameRate
from sashimono.ui.workspace import config_root

__all__ = ["PresetStoreError", "ProjectPreset", "ProjectPresetStore"]

#: ファイルの形の版 項目を足したり意味を変えたりしたら上げる
_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProjectPreset:
    """名前を付けて保存した設定の組み合わせ"""

    name: str
    width: int
    height: int
    frame_rate: FrameRate
    blending: str

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            # 分数のまま持つ 29.97 を小数で書くと、読み戻したときに 30000/1001 へ戻らない
            "frame_rate": [self.frame_rate.num, self.frame_rate.den],
            "blending": self.blending,
        }


class PresetStoreError(Exception):
    """一覧を書き換えられなかった 文言はそのまま画面に出す"""


class _BrokenError(Exception):
    """ファイルはあるが、一覧として読めない（JSON が壊れている・形が違う）"""


class _NewerError(Exception):
    """この版より新しい形で書かれている"""


class ProjectPresetStore:
    """:class:`ProjectPreset` の一覧の読み書き

    壊れた項目は飛ばして残りを読む 1 つ壊れただけで全部が消えると、作り直す手間が大きい
    ファイルごと読めなくても起動は止めない（ショートカットの保存と同じ考え方）

    書き換える（:meth:`put` と :meth:`remove`）ときは、読めなかった理由で扱いを分ける
    無いだけなら新しく作る 壊れていれば写しを残してから作り直す 一時的に読めない
    （ほかの道具が開いている）か、新しい版の形なら書き換えずに断る どれも同じ
    「空の一覧」として扱って上書きすると、手で直しかけた一覧や新しい版の一覧が
    新しい 1 件だけになって消える

    別々の窓で同時に保存したときは、後から書いた方が残る（錠は持たない） 同じ瞬間に
    2 つの窓でテンプレートを保存することはまず無く、錠を足す手間と釣り合わない
    書きかけのファイルは窓ごとに別の名前にしてあるので、壊れたファイルは残らない
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else config_root() / "project_presets.json"
        #: 直前の書き換えで、壊れていた一覧を写した先 画面が知らせるために持つ
        self.last_backup: Path | None = None

    def load(self) -> list[ProjectPreset]:
        """一覧 読めなければ空（画面を開けなくしない）"""
        try:
            return self._read()
        except (OSError, _BrokenError, _NewerError):
            return []

    def save(self, presets: list[ProjectPreset]) -> None:
        """一覧を丸ごと書く 書けなければ :class:`PresetStoreError`"""
        body = {"version": _VERSION, "presets": [preset.to_json() for preset in presets]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 書き終えてから入れ替える 書いている途中で落ちると、一覧が半分だけ残る
            # 書きかけの名前を窓ごとに変えるのと、失敗したときの片付けは write_atomic が持つ
            write_atomic(self.path, json.dumps(body, ensure_ascii=False, indent=2))
        except OSError as exc:
            raise PresetStoreError(f"テンプレートを保存できなかった: {self.path}（{exc}）") from exc

    def put(self, preset: ProjectPreset) -> list[ProjectPreset]:
        """同じ名前があれば置き換え、無ければ後ろへ足して保存する 並びは保存した順"""
        presets = self._for_update()
        names = [p.name for p in presets]
        if preset.name in names:
            presets[names.index(preset.name)] = preset
        else:
            presets.append(preset)
        self.save(presets)
        return presets

    def remove(self, name: str) -> list[ProjectPreset]:
        presets = [p for p in self._for_update() if p.name != name]
        self.save(presets)
        return presets

    def _read(self) -> list[ProjectPreset]:
        """一覧を読む 無ければ空 読めない理由は例外で分ける"""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except UnicodeDecodeError as exc:
            # メモ帳などで Shift_JIS に保存し直した一覧 ValueError の仲間で OSError ではないので、
            # 分けないと画面を開くだけで落ちる 中身はあるので、壊れた一覧として写してから作り直す
            raise _BrokenError(str(exc)) from exc
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise _BrokenError(str(exc)) from exc
        if not isinstance(data, dict):
            raise _BrokenError("一覧の形ではない")
        version = data.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise _BrokenError(f"版が読めない: {version!r}")
        if version > _VERSION:
            # 新しい版は項目の意味を変えているかもしれない 読めた所だけ並べると、
            # 違う意味の値で新しいプロジェクトを作ってしまう
            raise _NewerError(str(version))
        items = data.get("presets")
        if not isinstance(items, list):
            raise _BrokenError("presets が一覧ではない")
        presets: list[ProjectPreset] = []
        for item in items:
            preset = _preset_from(item)
            # 同じ名前が 2 つあると、選んだ名前からどちらを消すのかが決まらない 先の方を残す
            if preset is not None and all(p.name != preset.name for p in presets):
                presets.append(preset)
        return presets

    def _for_update(self) -> list[ProjectPreset]:
        """書き換える前に今の一覧を読む 失う物があれば、守ってから空で始めるか断る"""
        self.last_backup = None
        try:
            return self._read()
        except _NewerError as exc:
            raise PresetStoreError(
                f"テンプレートの一覧は新しい版の Sashimono Edit で保存されている（形の版 {exc}）"
                " 消してしまわないよう、この版では書き換えない"
            ) from exc
        except OSError as exc:
            raise PresetStoreError(
                f"テンプレートの一覧を読めなかったので、上書きせずにやめた: {self.path}（{exc}）"
            ) from exc
        except _BrokenError:
            self.last_backup = self._back_up()
            return []

    def _back_up(self) -> Path:
        """壊れた一覧を、上書きする前に隣へ写す 手で直しかけた中身を取り戻せるように"""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.path.with_name(f"{self.path.stem}.broken-{stamp}-{uuid.uuid4().hex[:4]}.json")
        try:
            shutil.copy2(self.path, target)
        except OSError as exc:
            raise PresetStoreError(
                f"壊れたテンプレートの一覧を控えに写せなかったので、上書きせずにやめた: {exc}"
            ) from exc
        return target


def _preset_from(item: object) -> ProjectPreset | None:
    """1 項目を読む 書き出せない値（奇数・範囲外）や知らない重ね合わせは捨てる

    手で書き換えたファイルの値をそのまま選べるようにすると、選んだ後に OK を押せない
    （奇数）か、プロジェクトの検査で断られる
    """
    if not isinstance(item, dict):
        return None
    name, rate, blending = item.get("name"), item.get("frame_rate"), item.get("blending")
    width, height = _size(item.get("width")), _size(item.get("height"))
    if not isinstance(name, str) or not name.strip() or width is None or height is None:
        return None
    if not isinstance(blending, str) or blending not in Blending.ALL:
        return None
    if not isinstance(rate, list) or len(rate) != 2:
        return None
    num, den = _integer(rate[0]), _integer(rate[1])
    if num is None or den is None or num <= 0 or den <= 0:
        return None
    return ProjectPreset(
        name=name.strip(),
        width=width,
        height=height,
        frame_rate=FrameRate(num, den),
        blending=blending,
    )


def _integer(value: object) -> int | None:
    # JSON の true は int として読めてしまう 1 として通すと、手で壊した値が黙って通る
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _size(value: object) -> int | None:
    """書き出せる縦横の画素数 偶数で、設定画面が受け付ける範囲の中"""
    number = _integer(value)
    if number is None or not MIN_RESOLUTION <= number <= MAX_RESOLUTION or number % 2:
        return None
    return number
