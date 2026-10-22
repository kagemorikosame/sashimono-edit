"""画面の中身を調べた結果（#324） 文字（OCR）と顔の位置

位置は作品の座標と同じ決まりで持つ 絵の中心が原点で、右と上が正（Y 軸は上が正）
調べる部品（Windows の OCR・YuNet）は左上が原点で下が正の画素を返すので、受け取った所で
:meth:`Box.from_corner` で直す 直さずに持つと、AI が字幕やテキストを避けて置くときに上下が
逆になり、画面の文字の上へ重ねる
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "Box",
    "Face",
    "ScreenKind",
    "ScreenReport",
    "ScreenSample",
    "TextLine",
]


class ScreenKind(StrEnum):
    """何を調べるか 値は控えの鍵と AI の道具の引数にそのまま使う"""

    #: 画面の中の文字（OCR）
    TEXT = "text"
    #: 顔の位置
    FACES = "faces"

    @property
    def label(self) -> str:
        """画面と断りの文に出す名前"""
        return "画面の文字" if self is ScreenKind.TEXT else "顔の位置"


@dataclass(frozen=True, slots=True)
class Box:
    """絵の中の枠 ``x`` と ``y`` は枠の中心で、絵の中心からの画素 右と上が正"""

    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_corner(
        cls,
        left: float,
        top: float,
        width: float,
        height: float,
        frame: tuple[int, int],
    ) -> Box:
        """左上が原点で下が正の枠（調べる部品の返す形）から作る ``frame`` は絵の（幅, 高さ）"""
        frame_width, frame_height = frame
        return cls(
            x=left + width / 2 - frame_width / 2,
            y=frame_height / 2 - (top + height / 2),
            width=width,
            height=height,
        )

    def to_json(self) -> dict[str, float]:
        # 画素の 1/10 より細かい値は意味が無い 控えと AI へ返す文字を短くする
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
        }

    @classmethod
    def from_json(cls, data: Any) -> Box:
        values = [float(data[key]) for key in ("x", "y", "width", "height")]
        if not all(math.isfinite(value) for value in values) or min(values[2:]) < 0:
            raise ValueError("枠の値が壊れている")
        return cls(*values)


@dataclass(frozen=True, slots=True)
class TextLine:
    """読めた 1 行"""

    text: str
    box: Box


@dataclass(frozen=True, slots=True)
class Face:
    """見つけた顔 1 つ ``score`` は 0〜1 の確からしさ"""

    box: Box
    score: float


@dataclass(frozen=True, slots=True)
class ScreenSample:
    """調べた 1 枚 ``seconds`` は素材の原点から数えた秒（場面の切り替わりと同じ数え方）"""

    seconds: float
    lines: tuple[TextLine, ...] = ()
    faces: tuple[Face, ...] = ()

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"seconds": round(self.seconds, 6)}
        if self.lines:
            data["lines"] = [{"text": line.text, **line.box.to_json()} for line in self.lines]
        if self.faces:
            data["faces"] = [
                {"score": round(face.score, 4), **face.box.to_json()} for face in self.faces
            ]
        return data

    @classmethod
    def from_json(cls, data: Any) -> ScreenSample:
        return cls(
            seconds=float(data["seconds"]),
            lines=tuple(
                TextLine(str(item["text"]), Box.from_json(item)) for item in data.get("lines", ())
            ),
            faces=tuple(
                Face(Box.from_json(item), float(item["score"])) for item in data.get("faces", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ScreenReport:
    """1 つの素材を 1 つの種類で調べた結果"""

    kind: ScreenKind
    #: 調べた部品の名前と版（``windows-ocr:ja`` ``yunet-2026may``） 控えの鍵にも入る
    engine: str
    #: 調べた絵の大きさ（素材の表示の大きさ 回転を直した後）
    width: int
    height: int
    #: 調べた順（時刻の順）の結果
    samples: tuple[ScreenSample, ...]
    #: 調べる予定だった枚数 途中で止まった物は :attr:`samples` の方が少ない
    planned: int
    #: 終わりまで調べたか
    complete: bool
    #: 調べた映像ストリームの番号（素材の中の ``index``） ``None`` は先頭の映像
    stream: int | None = None
    #: 場面の切り替わり（#319）で調べる所を選んだか 偽なら場面が分からず等間隔で選んだ
    by_scene: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "engine": self.engine,
            "width": self.width,
            "height": self.height,
            "planned": self.planned,
            "complete": self.complete,
            "stream": self.stream,
            "by_scene": self.by_scene,
            "samples": [sample.to_json() for sample in self.samples],
        }

    @classmethod
    def from_json(cls, data: Any) -> ScreenReport:
        """控えから読む 形が違えば ``ValueError``（読み直す）"""
        try:
            stream = data.get("stream")
            return cls(
                kind=ScreenKind(data["kind"]),
                engine=str(data["engine"]),
                width=int(data["width"]),
                height=int(data["height"]),
                samples=tuple(ScreenSample.from_json(item) for item in data["samples"]),
                planned=int(data["planned"]),
                complete=bool(data["complete"]),
                stream=None if stream is None else int(stream),
                by_scene=bool(data.get("by_scene", True)),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"控えの形が違う: {exc}") from exc
