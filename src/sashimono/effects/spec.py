"""エフェクトのパラメータ定義

AviUtl のスクリプト制御文字（``--track@`` ``--check@`` ``--color@`` など）と
1 対 1 に対応させてある 自前のエフェクトも配布スクリプトも同じ定義形式に載るので、
設定 UI の自動生成もプリセットの保存も 1 つの実装で済む

別々の形式にすると、AviUtl 互換（P5）で UI 生成をもう一度書くことになる

GUI にも OpenGL にも依存しない 定義はただのデータで、それをどう表示するか・
どう描画するかは別の層が決める
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from sashimono.core.model import AnimatedValue, ParamValue

#: まだ整えていない入力 モデルに入る前の値はここまで緩い
#:
#: :data:`~sashimono.core.model.ParamValue` は「モデルに入った後」の型で、数値は
#: :class:`~sashimono.core.model.AnimatedValue` になっている 呼び出し側に毎回
#: それを組み立てさせるのは煩雑なので、素の数値と ``None`` も受ける
type ParamInput = ParamValue | float | None

__all__ = [
    "IMAGE_FILTER",
    "IMAGE_SUFFIXES",
    "PIXEL_UNITS",
    "CheckSpec",
    "ColorSpec",
    "FileSpec",
    "FontSpec",
    "FontStyleSpec",
    "GridSpec",
    "ParamInput",
    "ParameterKind",
    "ParameterSpec",
    "SelectSpec",
    "TextSpec",
    "TrackSpec",
    "ValueSpec",
]


#: 画面の画素で数える長さを表す単位 :attr:`TrackSpec.pixels` を省いたとき、これで決める
#: 画素の値に単位を添えておけば、設定の画面にも「px」が出て、画質の分だけ縮める印にもなる
PIXEL_UNITS = frozenset({"px", "px/フレーム"})


class ParameterKind(Enum):
    """パラメータの種類 AviUtl の制御文字に対応する"""

    #: ``--track@`` 数値スライダー 時間で変化させられる唯一の種類
    TRACK = "track"
    #: ``--check@`` チェックボックス
    CHECK = "check"
    #: ``--color@`` 色
    COLOR = "color"
    #: ``--select@`` 選択肢
    SELECT = "select"
    #: ``--file@`` / ``--folder@`` パス
    FILE = "file"
    FOLDER = "folder"
    #: ``--font@`` フォント名
    FONT = "font"
    #: ``--text@`` 複数行テキスト、``--string@`` 1 行テキスト
    TEXT = "text"
    STRING = "string"
    #: ``--value@`` スライダーを持たない数値 時間で変化させられない
    VALUE = "value"
    #: AviUtl に対応する制御文字は無い 格子の点のずれをまとめて持つ
    GRID = "grid"


@dataclass(frozen=True, slots=True)
class TrackSpec:
    """数値スライダー キーフレームを打てる

    エフェクトの数値パラメータは原則これにする あとから「ここを動かしたい」と
    思ったときに、種類を変えずに済む
    """

    name: str
    label: str
    minimum: float
    maximum: float
    default: float
    #: スライダーの刻み 0.1 なら小数第 1 位まで
    step: float = 0.1
    #: 画面に添える単位（``"px"`` ``"%"`` ``"度"`` など）
    unit: str = ""
    #: 値が画面（プロジェクトの解像度）の画素で数える長さか 省くと単位で決める
    #: （:data:`PIXEL_UNITS` なら真） 単位と食い違う物だけ明に書く
    #:
    #: 真の値は、プレビューの画質を落として小さく合成するとき、合成の画素へ縮めて渡す
    #: （:meth:`scaled_at`） 縮めないと、1/2 画質では位置も大きさも 2 倍に出る
    #: 素材の画像の画素で数える物（読み込んだ画像の中の位置など）は画面の画素ではないので偽
    pixels: bool | None = None

    kind = ParameterKind.TRACK

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError(f"{self.name}: 最小値が最大値より大きい")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(f"{self.name}: 既定値が範囲外")

    @property
    def in_pixels(self) -> bool:
        """画面の画素で数える長さか（:attr:`pixels`）"""
        if self.pixels is not None:
            return self.pixels
        return self.unit in PIXEL_UNITS

    def scaled_at(self, value: AnimatedValue, frame: int, scale: float) -> float:
        """``frame`` の値を、合成の画素 1 つが画面の画素 ``1 / scale`` つに当たる所で使う形で返す

        壊れた数（NaN や無限大）は既定へ戻す 範囲では切らない 読み込んだテンプレートは
        表示の範囲を超える値を正しく使っていることがある
        """
        number = value.at(frame)
        if not math.isfinite(number):
            number = self.default
        return number * scale if self.in_pixels else number

    def default_value(self) -> AnimatedValue:
        return AnimatedValue(static=self.default)

    def clamp(self, value: float) -> float:
        return min(max(value, self.minimum), self.maximum)

    def coerce(self, value: ParamInput) -> AnimatedValue:
        """外から来た値を、この仕様に合う形へ寄せる

        プロジェクトファイルや AviUtl のエイリアスから読んだ値は、型も範囲も
        信用できない ここで 1 度だけ整える
        """
        if value is None:
            return self.default_value()
        if isinstance(value, AnimatedValue):
            return value
        if isinstance(value, bool | int | float):
            return AnimatedValue(static=self.clamp(float(value)))
        return self.default_value()


@dataclass(frozen=True, slots=True)
class CheckSpec:
    """チェックボックス"""

    name: str
    label: str
    default: bool = False
    #: スクリプトへ真偽ではなく 0 と 1 の数で渡すか
    #: AviUtl1 の ``--dialog`` の ``/chk`` は、ダイアログの中身を Lua のコード
    #: （``_3=1;``）として流すので数になる 配布スクリプトは ``_3==1`` と比べており、
    #: 真偽で渡すと Lua では ``true==1`` が偽なので、入れたチェックが効かない
    as_number: bool = False

    kind = ParameterKind.CHECK

    def default_value(self) -> bool:
        return self.default

    def coerce(self, value: ParamInput) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return self.default


@dataclass(frozen=True, slots=True)
class ColorSpec:
    """色 値は sRGB の 0..1 で ``(R, G, B, A)``

    リニアではなく sRGB で持つのは、ユーザーが指定するのも画面に出すのも
    sRGB だから リニアへの変換は描画の直前に 1 度だけ行う
    """

    name: str
    label: str
    default: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    #: アルファを編集させるか 縁取りの色など、不透明が前提のものは偽
    with_alpha: bool = True

    kind = ParameterKind.COLOR

    def default_value(self) -> tuple[float, ...]:
        return self.default

    def coerce(self, value: ParamInput) -> tuple[float, ...]:
        if isinstance(value, tuple) and len(value) >= 3:
            values = [min(max(float(v), 0.0), 1.0) for v in value[:4]]
            while len(values) < 4:
                values.append(1.0)
            return tuple(values)
        return self.default


@dataclass(frozen=True, slots=True)
class SelectSpec:
    """選択肢 値は選ばれた項目の識別子"""

    name: str
    label: str
    #: ``(識別子, 表示名)`` の並び
    choices: tuple[tuple[str, str], ...]
    default: str = ""

    kind = ParameterKind.SELECT

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"{self.name}: 選択肢が空")
        if self.default and self.default not in dict(self.choices):
            raise ValueError(f"{self.name}: 既定値 {self.default!r} が選択肢に無い")
        if not self.default:
            object.__setattr__(self, "default", self.choices[0][0])

    def default_value(self) -> str:
        return self.default

    def coerce(self, value: ParamInput) -> str:
        if isinstance(value, str) and value in dict(self.choices):
            return value
        return self.default

    def index_of(self, value: str) -> int:
        """シェーダへ渡すための番号 GLSL に文字列は無い"""
        for index, (identifier, _) in enumerate(self.choices):
            if identifier == value:
                return index
        return 0


@dataclass(frozen=True, slots=True)
class TextSpec:
    """テキスト ``multiline`` が偽なら 1 行"""

    name: str
    label: str
    default: str = ""
    multiline: bool = True

    kind = ParameterKind.TEXT

    def default_value(self) -> str:
        return self.default

    def coerce(self, value: ParamInput) -> str:
        return value if isinstance(value, str) else self.default


#: :class:`ValueSpec` の範囲が持てる端 設定画面の整数の欄（C++ の 4 バイトの int）に合わせる
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

#: エフェクトが画像として読む拡張子 素材の静止画
#: （``sashimono.engine.decode.probe.STILL_SUFFIXES``）と同じにしてある（試験で見ている）
#: ここだけ広げると、選べるのに読めない画像が出る
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

#: 画像を選ばせるときのファイル選択ダイアログのフィルタ
IMAGE_FILTER = f"画像 ({' '.join('*' + suffix for suffix in IMAGE_SUFFIXES)});;すべてのファイル (*)"


@dataclass(frozen=True, slots=True)
class FileSpec:
    """ファイルまたはフォルダのパス

    ``texture`` が真なら、そのパスの画像をシェーダへ 2 枚目の絵として渡す
    （画像合成の絵、縁取りの模様） uniform は項目名の ``sampler2D`` と、
    画像の大きさ（画素）の ``<項目名>_size`` 読めなかったときは大きさが 0 になる
    ので、シェーダはそれを見て画像なしの描き方へ戻る

    種類を分けずに旗にしたのは、AviUtl の ``--file@`` と同じく「パスを 1 つ持つ」
    点は変わらないため 別の種類にすると、設定 UI・プリセット・読み込みの
    どれもが同じ入力欄を 2 回書くことになる
    """

    name: str
    label: str
    default: str = ""
    #: 真ならフォルダを選ばせる
    directory: bool = False
    #: ファイル選択ダイアログのフィルタ
    filter: str = ""
    #: 真ならシェーダへ画像として渡す
    texture: bool = False

    kind = ParameterKind.FILE

    def default_value(self) -> str:
        return self.default

    def coerce(self, value: ParamInput) -> str:
        return value if isinstance(value, str) else self.default


@dataclass(frozen=True, slots=True)
class FontSpec:
    """フォント名"""

    name: str
    label: str
    default: str = "Yu Gothic UI"

    kind = ParameterKind.FONT

    def default_value(self) -> str:
        return self.default

    def coerce(self, value: ParamInput) -> str:
        return value if isinstance(value, str) and value else self.default


@dataclass(frozen=True, slots=True)
class FontStyleSpec:
    """フォントのファミリの中のスタイルの名前（``Light`` ``Bold`` ルイカの ``０９`` など）

    空文字は「既定」で、ファミリ名だけで選んだ今までの描き方になる 既定を空にしたのは、
    この項目を持たない前の版のプロジェクトを開いても見た目を変えないため
    選べるスタイルはファミリで決まるので、どの :class:`FontSpec` の項目から選ぶかを
    ``font`` に持つ 一覧を作るのは画面の側（ここは Qt を持ち込まない層）
    """

    name: str
    label: str
    #: スタイルを選ぶ元のファミリを持つ項目の名前
    font: str = "font"
    default: str = ""

    kind = ParameterKind.TEXT

    def default_value(self) -> str:
        return self.default

    def coerce(self, value: ParamInput) -> str:
        return value if isinstance(value, str) else self.default


@dataclass(frozen=True, slots=True)
class ValueSpec:
    """スライダーを持たない数値 時間で変化させられない

    シード値やループ回数のように、途中の値に意味が無いものに使う
    """

    name: str
    label: str
    default: int = 0
    minimum: int = -(2**31)
    maximum: int = 2**31 - 1

    kind = ParameterKind.VALUE

    def __post_init__(self) -> None:
        # 範囲は 4 バイトの int に収める 設定画面の整数の欄はそれより広い値を持てず、
        # 広いまま残すと、仕様では入れられる値が画面では入れられず、画面に出る値も
        # 仕様の値と食い違う 拒まずに縮めるのは、AviUtl のスクリプトから作る仕様で
        # 読み込みそのものを止めないため
        low = min(max(self.minimum, _INT32_MIN), _INT32_MAX)
        high = min(max(self.maximum, low), _INT32_MAX)
        object.__setattr__(self, "minimum", low)
        object.__setattr__(self, "maximum", high)
        object.__setattr__(self, "default", min(max(self.default, low), high))

    def default_value(self) -> int:
        return self.default

    def coerce(self, value: ParamInput) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return min(max(int(value), self.minimum), self.maximum)
        return self.default


def _finite_floats(value: tuple[object, ...]) -> list[float] | None:
    """並びを全部 ``float`` にする 1 つでも数でなければ ``None``

    ``float("x")`` は例外を投げる 整える役目の関数から例外が出ると、
    「辻褄が合わなければ格子なしへ戻す」はずが、プロジェクトを開く所で落ちる
    NaN と無限大も弾く シェーダへ渡すと、その画素の行き先が決まらない
    """
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return None
        try:
            number = float(item)
        except OverflowError:
            # Python の整数には桁の上限が無い 10**1000 のような値が保存ファイルに
            # 入っていると、float へ直す所で落ちて設定画面も描画も開けなくなる
            return None
        if not math.isfinite(number):
            return None
        numbers.append(number)
    return numbers


@dataclass(frozen=True, slots=True)
class GridSpec:
    """格子の点のずれ 値は ``(横の点数, 縦の点数, x0, y0, x1, y1, ...)`` の平らな並び

    点は**左上から行ごと**に並べる（YMM4 の ``MeshDeformationEffect`` の
    ``Points`` と同じ順） 並べ替えを挟むと、読むときと描くときで順が食い違って
    絵が対角に折れる

    値を 1 本の ``tuple[float, ...]`` にしてあるのは、これが
    :data:`~sashimono.core.model.ParamValue` に既にある型だから 保存形式も
    設定 UI も新しい入れ物を覚えずに済み、点数と点が必ず一緒に読み書きされる
    （別々の項目にすると、点数だけ書き換わった半端な状態が作れてしまう）

    空の並びは「格子を使わない」で、``mesh_deform`` はそのとき四隅のスライダで
    動く 既存のプロジェクトファイルには格子が入っていないので、既定はこちら

    シェーダへは 3 つの uniform で渡す（:mod:`sashimono.engine.gpu.effects`）
    ``<項目名>_columns`` ``<項目名>_rows``（``int``、格子が無ければ 0）と、
    ``<項目名>_points``（``vec2`` の配列）

    点はスライダーで触るものではない（5x5 で 50 個になる） 互換層が作った値を
    そのまま持ち運ぶための入れ物で、設定 UI には大きさだけを出す
    """

    name: str
    label: str
    #: 1 辺の点の数の下限と上限 上限はシェーダの配列の大きさと同じにすること
    minimum: int = 2
    maximum: int = 9
    #: 点のずれが画面の画素か（:attr:`TrackSpec.pixels` と同じ意味） 真なら、画質を落とした
    #: プレビューでは合成の画素へ縮めて渡す
    pixels: bool = False

    kind = ParameterKind.GRID

    def default_value(self) -> tuple[float, ...]:
        return ()

    def size(self, value: tuple[float, ...]) -> tuple[int, int]:
        """整えた値から ``(横の点数, 縦の点数)`` を読む 格子が無ければ ``(0, 0)``"""
        if len(value) < 2:
            return (0, 0)
        return (int(value[0]), int(value[1]))

    def coerce(self, value: ParamInput) -> tuple[float, ...]:
        """外から来た値を整える 少しでも辻褄が合わなければ「格子なし」へ戻す

        点数と点の数が食い違う値をそのまま渡すと、シェーダが配列の外を読む
        黙って欠けた点を 0 で埋めると、絵が畳まれて出るので、丸ごと捨てて
        四隅のスライダへ戻す方が直しやすい
        """
        if not isinstance(value, tuple) or len(value) < 2:
            return ()
        numbers = _finite_floats(value)
        if numbers is None:
            return ()
        # 点数は先に丸めない 2.9 を int() で 2 にすると、点の数の検査を
        # すり抜けて別の格子として通り、絵が畳まれて出る
        if numbers[0] != int(numbers[0]) or numbers[1] != int(numbers[1]):
            return ()
        columns, rows = int(numbers[0]), int(numbers[1])
        if not self.minimum <= columns <= self.maximum:
            return ()
        if not self.minimum <= rows <= self.maximum:
            return ()
        if len(numbers) != 2 + columns * rows * 2:
            return ()
        return (float(columns), float(rows), *numbers[2:])


#: パラメータ定義の総称
type ParameterSpec = (
    TrackSpec
    | CheckSpec
    | ColorSpec
    | SelectSpec
    | TextSpec
    | FileSpec
    | FontSpec
    | FontStyleSpec
    | ValueSpec
    | GridSpec
)


@dataclass(frozen=True, slots=True)
class ParameterGroup:
    """設定 UI で 1 つの見出しにまとめるパラメータ

    エフェクトのパラメータが 10 個を超えると、並べただけでは何がどこにあるか
    分からなくなる
    """

    label: str
    names: tuple[str, ...] = field(default_factory=tuple)
