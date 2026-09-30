"""エフェクトの定義と、その一覧

定義はデータでしかない GLSL のソースは文字列として持つだけで、コンパイルは
エンジン層（:mod:`sashimono.engine.gpu.effects`）が行う この分離のおかげで、
設定 UI もプリセットも GPU を持たずに扱える
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sashimono.core.model import AnimatedValue, Effect, ParamValue
from sashimono.effects.spec import ParameterGroup, ParameterSpec, ParamInput, TrackSpec

__all__ = [
    "EffectDefinition",
    "EffectRegistry",
    "Pieces",
    "object_pivot",
    "registry",
    "turned_object",
]


#: 音を加工する関数の形 引数は サンプル・解いた値・時間まわりの手がかり
#:
#: サンプルと手がかりを :any:`Any` にしてあるのは、音の仕組み
#: （:mod:`sashimono.effects.audio`）がこの定義を取り込むため
#: 実体の型で書くと取り込みが輪になる
AudioProcess = Callable[[Any, dict[str, float], Any], Any]

#: 音のエフェクトが前の音をどれだけ（秒）読むか 引数は解いた値
AudioHistory = Callable[[dict[str, float]], float]


@dataclass(frozen=True, slots=True)
class Pieces:
    """絵を升目に割り、升目 1 つずつを四角として描くエフェクトの描き方（破片）

    出力の画素ごとに「ここへ来る欠片」を探すと、欠片を遠くまで散らすほど 1 画素で
    調べる欠片が増える 1080p を 4 画素の欠片に割ると 1 コマ 90ms を超えた（#207）
    四角として描けば、欠片 1 つの手間は欠片の面積だけで済み、どこへ散っても変わらない

    エンジンは中身の範囲（``u_content``）に掛かる升目の数だけ四角を描く（インスタンス描画）
    升目は ``(0, 0)`` から ``u_cell`` 刻み ``vertex_shader`` は
    :data:`sashimono.effects.builtin.PIECE_PRELUDE` から始め、升目ごとの動きを決めて
    ``place_piece`` を呼ぶ ``fragment_shader`` は ``v_source``（元の絵のどこを読むか）を
    受け取り、**事前乗算**の色を返す 重なった欠片は描いた順に手前へ重なる
    """

    #: 升目の一辺を持つ項目名 画素で決める項目（``px``）なので、画質を落とせば縮む
    size: str
    #: 一辺の下限（画面の画素） 細かすぎる升目で四角の数が膨らまないように止める
    minimum: float
    vertex_shader: str


@dataclass(frozen=True, slots=True)
class EffectDefinition:
    """1 種類のエフェクト

    ``fragment_shader`` は :data:`sashimono.engine.gpu.VERTEX_SHADER` と組み合わせる
    フラグメントシェーダ 次の uniform が自動で渡る

    ``u_texture``   入力（リニア、ストレートアルファ）
    ``u_size``      入力の大きさ（ピクセル）
    ``u_pass``      複数パスのときの通し番号（0 から）
    ``u_time``      クリップ先頭からの経過秒
    ``u_frame``     クリップ先頭からの経過フレーム

    パラメータは名前をそのまま uniform 名にする 数値は ``float``、チェックは
    ``bool``、色は ``vec4``（リニアへ変換済み）、選択肢は番号の ``int``
    """

    kind: str
    label: str
    #: 分類 設定 UI のツリーで使う
    category: str
    parameters: tuple[ParameterSpec, ...] = ()
    fragment_shader: str | None = None
    #: 何回シェーダを通すか ぼかしは横方向と縦方向で 2 回に分ける
    #: 1 回で 2 次元のカーネルを畳むと計算量が半径の 2 乗になる
    passes: int = 1
    #: 設定 UI での見出し分け 空なら並べるだけ
    groups: tuple[ParameterGroup, ...] = field(default_factory=tuple)
    #: 音を加工する関数 映像のエフェクトはシェーダだが、音は GPU を通さない
    #:
    #: ``(サンプル, 解いた値, 時間まわりの手がかり) -> サンプル`` の形
    #: これが入っていれば音のエフェクト、入っていなければ映像のエフェクト
    audio_process: AudioProcess | None = None
    #: 前の音を読むエフェクト（残響・ディレイ・音程）が、塊の頭より前をどれだけ（秒）要るか
    #:
    #: ミキサは塊を細かく切って頼む 前の塊の音を覚えずに済むよう、要るだけ前から読み直して
    #: 掛け、頭を捨てる 入っているエフェクトは塊を切らずに 1 度で掛ける（値は塊の頭の値）
    #: 前の音を覚えて持ち越す作りにすると、シークや書き出しの並列で覚えた音が食い違う
    audio_history: AudioHistory | None = None
    #: 絵の置かれた範囲（``u_object``）を広げるエフェクトの、上・下・左・右の項目名
    #:
    #: 領域拡張のように入れ物そのものを広げるものは、後ろに積んだエフェクト
    #: （ミラーの折り返す線・角丸・中心基準の動き）も広げた後の範囲で動くべき
    #: 印を付けないと、広げる前の範囲のまま後ろが動いて位置がずれる
    expands_object: tuple[str, str, str, str] | None = None
    #: 絵を中心の周りに回すので、後ろのエフェクトには回しても収まる範囲（範囲の対角線を
    #: 直径とする円を囲む正方形）を絵の範囲として渡すか
    #:
    #: YMM4 は渦巻き（SpiralTransform）の後ろの跳ねて登場の潰れを、その正方形の下端を
    #: 支点にして潰した（お辞儀(120F) の 640x360 の四角で、下端が四角の下端より
    #: 187 画素下の支点から潰れた分だけ下がった 対角線の半分 367 と合う #205）
    #: 広げる前の範囲のままだと、潰れた四角が 3 画素上に浮く
    turns_object: bool = False
    #: 後ろに積んだエフェクトを、このエフェクトの決めた範囲の中だけに効かせるか（部分フィルタ）
    #:
    #: 真のとき、エンジンはここへ来た時点の絵を取っておき、次の同じ印のエフェクトか
    #: 並びの終わりで、このシェーダに ``u_source``（取っておいた絵）と ``u_texture``
    #: （後ろを掛け終えた絵）を渡して混ぜさせる その場では何も描かない
    scopes_following: bool = False
    #: 絵の中身を動かさず、透明な所に色を置かないか（色だけを変える・縁を削る）
    #:
    #: エンジンはこれの立たないエフェクトを掛けた後、中身の範囲（``u_content``）を
    #: バッファ全体に広げる 後ろの粒や欠片はその範囲で探す所を狭めるので、立てれば
    #: 速いまま描ける 透明な所に色を置く物に立てると、はみ出した分が粒や欠片から切れる
    keeps_content: bool = False
    #: 画面いっぱいの四角 1 枚ではなく、升目ごとの四角で描くか（:class:`Pieces`）
    pieces: Pieces | None = None
    #: この値の組なら絵を変えない、という項目と値（動かない値だけが当たる）
    #:
    #: クリップが最初から持つ配置と反転（固定の項目）は、置いたクリップすべてに付く
    #: 既定のままでもシェーダを通すと、全クリップで画面 1 枚ぶんのパスが増え、中間の
    #: バッファを通る分だけ絵も前と変わる ここに当たる物はエンジンが掛けずに飛ばす
    #: （:meth:`is_idle`） 挙げていない項目は結果に効かない物に限る
    idle_when: tuple[tuple[str, float | bool], ...] = ()

    def __post_init__(self) -> None:
        if self.passes < 1:
            raise ValueError(f"{self.kind}: パス数は 1 以上必要")
        names = [spec.name for spec in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.kind}: パラメータ名が重複している")
        unknown = [name for name, _ in self.idle_when if name not in names]
        if unknown:
            raise ValueError(f"{self.kind}: 何もしない値の項目が定義に無い: {unknown}")
        if self.pieces is not None:
            # 下限は升目の一辺として割る数になる 0 や NaN を通すと、大きさの項目が 0 のときに
            # 升目の数が求まらず、プレビューも書き出しも例外で止まる
            minimum = self.pieces.minimum
            if not (math.isfinite(minimum) and minimum > 0.0):
                raise ValueError(f"{self.kind}: 升目の一辺の下限は正の数にする: {minimum}")
            # 升目の大きさが読めないと、エンジンは下限の大きさで割って四角の数が膨らむ
            if not isinstance(self.spec(self.pieces.size), TrackSpec):
                raise ValueError(f"{self.kind}: 升目の大きさの項目が数の項目に無い")
            # 四角で描く道は 1 回しか描かない 2 回目以降のパスは黙って捨てることになる
            if self.passes != 1 or self.fragment_shader is None:
                raise ValueError(f"{self.kind}: 升目で描くのは 1 パスのシェーダだけ")

    def is_idle(self, effect: Effect) -> bool:
        """``effect`` が絵を何も変えない値か :attr:`idle_when` が空なら常に偽

        動く値（キーフレームを持つ値）は、途中の点がすべて同じでも偽にする 描くたびに
        すべての点を見るのは、飛ばして得る分より高く付く
        """
        if not self.idle_when:
            return False
        for name, idle in self.idle_when:
            spec = self.spec(name)
            if spec is None:  # pragma: no cover - __post_init__ で断っている
                return False
            raw = effect.params.get(name)
            value = spec.coerce(spec.default_value() if raw is None else raw)
            if isinstance(value, AnimatedValue):
                if value.is_animated or value.static != idle:
                    return False
            elif value != idle:
                return False
        return True

    def spec(self, name: str) -> ParameterSpec | None:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        return None

    def default_params(self) -> dict[str, ParamValue]:
        return {spec.name: spec.default_value() for spec in self.parameters}

    def create(self, **overrides: ParamInput) -> Effect:
        """既定値で埋めたエフェクトを作る"""
        params = self.default_params()
        for name, value in overrides.items():
            spec = self.spec(name)
            if spec is not None:
                params[name] = spec.coerce(value)
        return Effect(kind=self.kind, params=params)

    def normalize(self, params: dict[str, ParamValue]) -> dict[str, ParamValue]:
        """外から来たパラメータを、定義に合う形へ整える

        プロジェクトファイルや配布エイリアスから読んだ値は、型も範囲も、
        そもそも存在するかも信用できない 欠けているものは既定値で埋め、
        定義に無いものは落とす
        """
        return {spec.name: spec.coerce(params.get(spec.name)) for spec in self.parameters}


def turned_object(
    box: tuple[float, float, float, float], pivot: tuple[float, float] | None = None
) -> tuple[float, float, float, float]:
    """回しても収まる範囲（:attr:`EffectDefinition.turns_object`） 左・下・右・上の並び

    ``pivot`` の周りに回したときに四隅が通る円を囲む正方形 省くと範囲の中心で、そのときは
    対角線を直径とする円になる 支点を端や画面の中央へ動かした渦巻きで中心の正方形を渡すと、
    回った絵がはみ出し、後ろの跳ねや拡大が別の中心と下端を使う（#217 の指摘 YMM4 で測ったのは
    中心を支点にした渦巻きだけ）
    """
    left, bottom, right, top = box
    if pivot is None:
        pivot = ((left + right) * 0.5, (bottom + top) * 0.5)
    x, y = pivot
    half = max(math.hypot(cx - x, cy - y) for cx in (left, right) for cy in (bottom, top))
    return (x - half, y - half, x + half, y + half)


def object_pivot(
    horizontal: str,
    vertical: str,
    anchor: tuple[float, float],
    box: tuple[float, float, float, float],
    size: tuple[float, float],
    origin: tuple[float, float],
) -> tuple[float, float]:
    """支点の選び方（``pivot_h`` ``pivot_v`` と中心 X Y）から支点を求める

    シェーダの ``pivot_point``（:mod:`sashimono.effects.motion` の ``_PIVOT``）と同じ決まり
    向きは GL と同じ（Y は上が正 ``box`` は左・下・右・上）
    """
    left, bottom, right, top = box
    x = {"screen": size[0] * 0.5, "left": left, "right": right, "origin": origin[0]}.get(
        horizontal, (left + right) * 0.5
    )
    y = {"screen": size[1] * 0.5, "top": top, "bottom": bottom, "origin": origin[1]}.get(
        vertical, (bottom + top) * 0.5
    )
    return (x + anchor[0], y + anchor[1])


class EffectRegistry:
    """エフェクト定義の一覧

    ``kind`` を鍵に引く プロジェクトファイルには ``kind`` しか入らないので、
    定義が見つからないエフェクトは「未知」として素通しにする（後述）
    """

    def __init__(self) -> None:
        self._definitions: dict[str, EffectDefinition] = {}

    def register(self, definition: EffectDefinition, *, replace: bool = False) -> EffectDefinition:
        """エフェクトを登録する

        既定では重複を拒む 自前のエフェクトで名前がぶつかるのは書き間違いで、
        黙って上書きすると、どちらが効いているのか分からなくなる

        ``replace`` は外から読み込む定義（AviUtl スクリプト）のためにある
        こちらは走査のたびに読み直すのが正しく、内容が変わっていれば新しい方を
        使ってほしい
        """
        if not replace and definition.kind in self._definitions:
            raise ValueError(f"すでに登録されているエフェクト: {definition.kind}")
        self._definitions[definition.kind] = definition
        return definition

    def unregister(self, kind: str) -> None:
        """登録を外す スクリプトのフォルダを変えたときに使う"""
        self._definitions.pop(kind, None)

    def get(self, kind: str) -> EffectDefinition | None:
        return self._definitions.get(kind)

    def require(self, kind: str) -> EffectDefinition:
        definition = self._definitions.get(kind)
        if definition is None:
            raise KeyError(f"未知のエフェクト: {kind}")
        return definition

    def all(self) -> tuple[EffectDefinition, ...]:
        """登録順ではなく、分類 → 表示名の順で返す UI の一覧用"""
        return tuple(sorted(self._definitions.values(), key=lambda d: (d.category, d.label)))

    def sound_kinds(self) -> frozenset[str]:
        """音を加工するエフェクト（:attr:`EffectDefinition.audio_process` を持つ物）の種類

        コア層は定義を読めないので、絵と音のエフェクトを振り分ける命令
        （:class:`~sashimono.core.commands.ConvertLayers`）へはこれを渡す
        """
        return frozenset(k for k, d in self._definitions.items() if d.audio_process is not None)

    def categories(self) -> tuple[str, ...]:
        seen: list[str] = []
        for definition in self.all():
            if definition.category not in seen:
                seen.append(definition.category)
        return tuple(seen)

    def __contains__(self, kind: object) -> bool:
        return kind in self._definitions

    def __len__(self) -> int:
        return len(self._definitions)


#: アプリ全体で使う一覧 :mod:`sashimono.effects.builtin` が読み込み時に登録する
registry = EffectRegistry()
