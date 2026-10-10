"""外部形式を写した結果 AviUtl 側と YMM4 側で共通に使う

どちらのソフトから来ても「1 オブジェクト = 1 クリップ + 置くレイヤー + 参照して
いる素材」に落ちる 同じ形にしておくと、タイムラインへ置く処理
（:mod:`sashimono.compat.catalog`）を 1 つで済ませられる
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from sashimono.core.model import Clip, Effect

# 着せ替えの長さ合わせは、プリセットの適用（コア層）と同じ物を使う 2 か所に書くと、
# 片方だけ直したときに棚とプリセットで同じ動きの伸び方が食い違う
from sashimono.core.model.fitting import fitted_effect, fitted_value

__all__ = ["MappedObject", "fitted_effect", "fitted_value"]


@dataclass(frozen=True, slots=True)
class MappedObject:
    """1 オブジェクトを写した結果"""

    clip: Clip
    layer: int
    #: 素材ファイルを参照している場合のパス 読み込みは呼び出し側が行う
    media_path: str = ""
    kind: str = ""
    #: 元のファイルが長さを指定していたか
    #:
    #: エイリアスは長さを持たないことがある その場合、1 フレームのクリップを
    #: 置くのではなく、置く側が既定の長さを決める
    has_span: bool = True
    #: 1 枚の絵にまとめてから重ねる中身 空でなければ、置く側はこれをシーンにして
    #: :attr:`clip` をそのシーンのクリップとして置く
    #:
    #: YMM4 の「合成する」グループがこれ 中身を 1 つずつ置いてグループのエフェクトを
    #: 配ると、反転や拡大が 1 つずつの中心で掛かり、まとめた絵とは別の形になる
    children: tuple[MappedObject, ...] = ()
    #: シーンにするときの名前
    label: str = ""
    #: シーンのどのフレームから映し始めるか 中身がまとめた入れ物より先に始まるとき、
    #: シーンの頭は一番早い中身に合わせ、入れ物のクリップはその分だけ進めた所から映す
    scene_offset: int = 0
    #: 素材の音も一緒に鳴らすか（YMM4 の ``VideoItem``）
    #:
    #: YMM4 の動画アイテムは 1 つで映像と音の両方を持つので、置く側が音声トラックへも
    #: 展開しないと音が鳴らない AviUtl は同じ動画を「動画ファイル」と「音声ファイル」の
    #: 2 つのオブジェクトに分けて書き出すので、そちらでこれを立てると音が二重になる
    with_sound: bool = False
    #: 音として鳴らすときに掛けるエフェクト（音量など） 音声トラックへ置くクリップに付く
    #: 映像のエフェクトは音には関係ないので、:attr:`clip` の側とは分けて持つ
    audio_effects: tuple[Effect, ...] = ()
    #: 素材の終わりを越えた所で、最後の絵を出し続けるか（YMM4 の動画アイテム）
    #:
    #: 素材の長さは写す段では分からない（素材を読むのは置く側） 置く側が素材と結んだ
    #: ときに、クリップが素材の終わりを越えて読むなら :attr:`Clip.hold_at` を最後の絵の
    #: 時刻にする 立てないと、越えた所は何も映らない（こちらの素のクリップの決まり）
    hold_last_frame: bool = False
    #: 素材の何本目の音を鳴らすか（YMM4 の ``AudioTrackIndex`` 0 始まり）
    #:
    #: 音の道だけを数えた順番で持つ 素材の中の番号（映像も含めて数える）は素材を読む
    #: まで分からないので、置く側が素材と結んだときに直す
    audio_track: int = 0

    @property
    def has_picture(self) -> bool:
        """置けば何かが映るか エフェクトだけのものは置いても映らない"""
        return self.clip.source is not None or bool(self.media_path) or bool(self.children)

    def walk(self) -> Iterator[MappedObject]:
        """自分と、まとめた中身をすべて 中の文字を探すときに使う"""
        yield self
        for child in self.children:
            yield from child.walk()
