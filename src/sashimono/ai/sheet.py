"""コンタクトシートと範囲の連続したコマの並べ方（#320 A2）

何枚ものコマを 1 枚の画像に並べて AI へ見せる 並べ方（升の大きさ・列の数・全体の大きさ）
はここで決め、描くのは編集画面（:meth:`~sashimono.ai.host.EditorHost.start_sheet`）
ここは Qt を知らない 偽のホストの試験でも同じ並べ方を確かめられるように

大きさの上限の決め方（#298 の「映像を AI に見せる量と費用の釣り合い」）

- 画像の費用は画素の数でほぼ決まる（横 x 縦 / 750 トークンほど） 枚数を増やすより、
  1 枚に並べた方が同じ費用で多くのコマを見せられる
- 1 枚の長い辺は :data:`MAX_SHEET_EDGE`（1568 画素）まで 古い型のモデルはこれより大きい絵を
  縮めて読むので、超えると升の下に書いた時刻が潰れて読めなくなる 費用も 1 枚 3300 トークン
  ほどで止まる（preview_frame の最大 1280x720 の 3 枚ぶん弱）
- 升の幅は :data:`MIN_CELL_WIDTH` より狭くしない 狭いと絵が何かを見分けられず、時刻の字も
  升に収まらない 収まらない頼みは断り、枚数を減らすか列を変えるよう返す
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GAP",
    "LABEL_HEIGHT",
    "MAX_COLUMNS",
    "MAX_SHEET_EDGE",
    "MIN_CELL_WIDTH",
    "SheetError",
    "SheetLayout",
    "sheet_layout",
]


class SheetError(ValueError):
    """並べられない頼み 文面は直し方まで書く（道具が :class:`ToolError` にして AI へ返す）

    ``ToolError`` をここで使わないのは、ホストの窓口（``ai/host.py``）がこの並べ方を
    読むため 互いを読み合うと、どちらかが読み終わる前に名前を探して失敗する
    """


#: 1 枚の長い辺の上限（画素） 理由はモジュールの説明
MAX_SHEET_EDGE = 1568
#: 升の幅の下限（画素） タイムコード（00:00:00:00）と番号が 1 行に収まる幅
MIN_CELL_WIDTH = 120
#: 列の数の上限 これより多く並べると、どの升が何番目かを AI が数え違える
MAX_COLUMNS = 8
#: 升の下の、時刻を書く帯の高さ（画素） 絵の上に重ねないのは、画面の下に置いた字幕が
#: 時刻の字で隠れて、字幕の出来を確かめられなくなるため
LABEL_HEIGHT = 16
#: 升の間の隙間（画素） 隙間が無いと、隣のコマと続いた 1 枚の絵に見える
GAP = 4


@dataclass(frozen=True, slots=True)
class SheetLayout:
    """1 枚の並べ方 左上から右へ、行が尽きたら次の行へ並べる"""

    count: int
    columns: int
    cell_width: int
    cell_height: int

    @property
    def rows(self) -> int:
        return -(-self.count // self.columns)

    @property
    def width(self) -> int:
        return self.columns * self.cell_width + (self.columns + 1) * GAP

    @property
    def height(self) -> int:
        return self.rows * (self.cell_height + LABEL_HEIGHT) + (self.rows + 1) * GAP

    def cell(self, index: int) -> tuple[int, int]:
        """``index`` 番目（0 から）の絵の左上 時刻の帯はその真下"""
        if not 0 <= index < self.count:
            raise IndexError(f"升の番号が範囲の外です: {index}")
        row, column = divmod(index, self.columns)
        x = GAP + column * (self.cell_width + GAP)
        y = GAP + row * (self.cell_height + LABEL_HEIGHT + GAP)
        return x, y


def sheet_layout(
    count: int,
    *,
    cell_width: int,
    frame_size: tuple[int, int],
    columns: int | None = None,
    max_edge: int = MAX_SHEET_EDGE,
) -> SheetLayout:
    """``count`` 枚を並べる形を決める

    ``columns`` を省くと、升を一番大きく取れる列の数を選ぶ 同じ大きさなら空いた升が
    少なく、縦横の差が小さい方 横に長い 1 行や縦に長い 1 列にすると、長い辺の上限に
    先に当たって升が小さくなる

    升の幅は頼まれた幅から、全体が ``max_edge``（本人の設定 #344 既定は :data:`MAX_SHEET_EDGE`）
    に収まるまで縮める
    縮めても :data:`MIN_CELL_WIDTH` に届かなければ断る
    """
    if count < 1:
        raise SheetError("並べるコマがありません")
    frame_width, frame_height = frame_size
    aspect = frame_height / max(frame_width, 1)
    if columns is not None:
        if not 1 <= columns <= MAX_COLUMNS:
            raise SheetError(f"columns は 1〜{MAX_COLUMNS} にしてください")
        candidates = [min(columns, count)]
    else:
        candidates = list(range(1, min(count, MAX_COLUMNS) + 1))

    best: tuple[tuple[int, int, int], SheetLayout] | None = None
    for candidate in candidates:
        layout = _fit(count, candidate, cell_width, aspect, max_edge)
        empty = layout.rows * layout.columns - count
        # 升の幅が大きいほど、空きが少ないほど、縦横が近いほどよい
        score = (-layout.cell_width, empty, abs(layout.width - layout.height))
        if best is None or score < best[0]:
            best = (score, layout)
    assert best is not None
    layout = best[1]
    if layout.cell_width < MIN_CELL_WIDTH:
        raise SheetError(
            f"{count} 枚を 1 枚（長い辺 {max_edge} 画素まで）に並べると、1 コマの幅が"
            f" {layout.cell_width} 画素になり小さすぎます 枚数を減らすか、columns を変えてください"
        )
    return layout


def _fit(count: int, columns: int, cell_width: int, aspect: float, max_edge: int) -> SheetLayout:
    """頼まれた幅から、全体が上限に収まる所まで升を縮める"""
    rows = -(-count // columns)
    # 横の上限から決まる幅
    by_width = (max_edge - (columns + 1) * GAP) // columns
    # 縦の上限から決まる幅 1 行の高さは 絵の高さ（幅 x 縦横比）+ 帯
    room = max_edge - (rows + 1) * GAP - rows * LABEL_HEIGHT
    by_height = int(room / rows / aspect) if aspect > 0 else by_width
    width = max(1, min(cell_width, by_width, by_height))
    # 切り捨てる 丸めて上げると、行の数だけ積もって長い辺の上限を数画素超える
    height = max(1, int(width * aspect))
    return SheetLayout(count=count, columns=columns, cell_width=width, cell_height=height)
