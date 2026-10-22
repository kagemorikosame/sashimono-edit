"""顔の位置を見つける（#324） YuNet（OpenCV Zoo MIT）を onnxruntime で動かす

OpenCV の ``FaceDetectorYN`` を使えば前後の処理も揃っているが、opencv-python（数十 MB）を
引くので、前後の処理だけを numpy で書く 中身は OpenCV の ``face_detect.cpp`` と同じ
（縮めた絵を 32 の倍数まで右と下に 0 で埋め、BGR の 0〜255 のまま渡し、3 つの細かさの
出力から枠を作って重なりを除く）

モデルは入力の大きさが決まっていない版（``face_detection_yunet_2026may.onnx``）を使う
決まった版（2023mar）は 640x640 などへ伸ばして渡す必要があり、縦横の比が崩れる
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sashimono.model_files import RemoteFile

from .report import Box, Face

__all__ = [
    "DETECT_EDGE",
    "ENGINE",
    "NMS_OVERLAP",
    "SCORE_FLOOR",
    "YUNET",
    "FaceDetector",
    "decode_faces",
    "prepare",
    "shrink",
    "suppress",
]

#: 顔の検出のモデル 取得元はコミットで固定し、中身を sha256 で照らす
#: 大きさと sha256 は Git LFS の記録（取得元のコミットの ``.onnx`` の pointer）と同じ
YUNET = RemoteFile(
    name="face_detection_yunet_2026may.onnx",
    url=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/face_detection_yunet/"
        "face_detection_yunet_2026may.onnx"
    ),
    sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    size=229738,
    label="顔の検出のモデル（YuNet MIT）",
)

#: 結果と控えに書く部品の名前 モデルを替えたら変える（控えの鍵に入るので読み直す）
ENGINE = "yunet-2026may"

#: 長い辺をこの画素まで縮めて調べる YuNet は 10〜300 画素ほどの顔を見つけるように学習して
#: あり、1920x1080 の話し手の顔（400 画素を超える）をそのまま渡すと見落とす 640 なら顔の
#: 大きさが 1/3 になり、画面の 1/60 ほどの小さな顔まで拾える 時間も 1 枚 10 ms ほどで済む
DETECT_EDGE = 640

#: 覚える確からしさの下限 AI へ返すときは道具の引数（既定 0.9 OpenCV Zoo の見本と同じ）で
#: 絞る 覚える所で絞ると、しきい値を下げて聞き直したときに調べ直しになる
SCORE_FLOOR = 0.5

#: 重なりを除くときの重なりの割合（IoU） OpenCV Zoo の見本と同じ
NMS_OVERLAP = 0.3

#: 出力の細かさ（何画素ごとに 1 つの候補を出すか） モデルの出力の名前の数字と同じ
STRIDES = (8, 16, 32)

#: 縮めた絵を埋める大きさの単位 一番粗い細かさに割り切れないと、候補の並びがずれる
_PAD = 32


def shrink(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """絵を ``width`` x ``height`` へ縮める 整数倍の分は平均し、残りは最も近い画素を取る

    最も近い画素だけで 1/6 まで縮めると、細い線や小さな顔がまだらに欠ける 平均だけでは
    割り切れない大きさに合わせられないので、2 段にする
    """
    source_height, source_width = image.shape[:2]
    factor = min(source_height // max(1, height), source_width // max(1, width))
    picture = image.astype(np.float32)
    if factor >= 2:
        rows, columns = source_height // factor * factor, source_width // factor * factor
        channels = picture.shape[2] if picture.ndim == 3 else 1
        picture = (
            picture[:rows, :columns]
            .reshape(rows // factor, factor, columns // factor, factor, channels)
            .mean(axis=(1, 3))
        )
        source_height, source_width = picture.shape[:2]
    pick_rows = (np.arange(height) * source_height // max(1, height)).clip(0, source_height - 1)
    pick_columns = (np.arange(width) * source_width // max(1, width)).clip(0, source_width - 1)
    return np.ascontiguousarray(picture[pick_rows][:, pick_columns])


def prepare(rgba: np.ndarray, edge: int = DETECT_EDGE) -> tuple[np.ndarray, float]:
    """モデルへ渡す形（1x3xHxW BGR float32 0〜255）と、縮めた倍率を返す"""
    height, width = rgba.shape[:2]
    scale = min(1.0, edge / max(height, width, 1))
    small_width, small_height = max(1, round(width * scale)), max(1, round(height * scale))
    small = shrink(rgba[:, :, :3], small_width, small_height)
    padded_width = -(-small_width // _PAD) * _PAD
    padded_height = -(-small_height // _PAD) * _PAD
    blob = np.zeros((1, 3, padded_height, padded_width), np.float32)
    # RGB を BGR へ OpenCV で学習したモデルは BGR の並びで読む
    blob[0, :, :small_height, :small_width] = small[:, :, ::-1].transpose(2, 0, 1)
    return blob, small_width / width if width else 1.0


def suppress(corners: np.ndarray, scores: np.ndarray, overlap: float = NMS_OVERLAP) -> list[int]:
    """重なった枠のうち確からしい方だけを残す（NMS） ``corners`` は（左, 上, 右, 下）の並び"""
    order = list(np.argsort(-scores, kind="stable"))
    kept: list[int] = []
    areas = (corners[:, 2] - corners[:, 0]) * (corners[:, 3] - corners[:, 1])
    while order:
        best = int(order.pop(0))
        kept.append(best)
        if not order:
            break
        rest = np.array(order)
        left = np.maximum(corners[best, 0], corners[rest, 0])
        top = np.maximum(corners[best, 1], corners[rest, 1])
        right = np.minimum(corners[best, 2], corners[rest, 2])
        bottom = np.minimum(corners[best, 3], corners[rest, 3])
        inter = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
        union = areas[best] + areas[rest] - inter
        ratio = np.where(union > 0, inter / np.where(union > 0, union, 1), 0)
        order = [int(index) for index, value in zip(rest, ratio, strict=True) if value <= overlap]
    return kept


def decode_faces(
    outputs: Mapping[str, np.ndarray],
    padded: tuple[int, int],
    scale: float,
    frame: tuple[int, int],
    *,
    floor: float = SCORE_FLOOR,
) -> tuple[Face, ...]:
    """モデルの出力から顔の枠を作る ``padded`` は埋めた後の（幅, 高さ）、``frame`` は元の絵の
    （幅, 高さ） 枠は元の絵の画素で、絵の中心から右と上が正（:class:`~.report.Box`）
    """
    padded_width, padded_height = padded
    corners: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for stride in STRIDES:
        columns, rows = padded_width // stride, padded_height // stride
        cls = np.clip(outputs[f"cls_{stride}"].reshape(-1), 0.0, 1.0)
        obj = np.clip(outputs[f"obj_{stride}"].reshape(-1), 0.0, 1.0)
        bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)
        count = rows * columns
        if cls.shape[0] != count or bbox.shape[0] != count:
            raise ValueError(f"顔の検出の出力の数が合わない（{stride}: {cls.shape[0]} と {count}）")
        scores = np.sqrt(cls * obj)
        chosen = np.flatnonzero(scores >= floor)
        if chosen.size == 0:
            continue
        cell_rows, cell_columns = np.divmod(chosen, columns)
        center_x = (cell_columns + bbox[chosen, 0]) * stride
        center_y = (cell_rows + bbox[chosen, 1]) * stride
        width = np.exp(bbox[chosen, 2]) * stride
        height = np.exp(bbox[chosen, 3]) * stride
        corners.append(
            np.stack(
                [
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ],
                axis=1,
            )
        )
        all_scores.append(scores[chosen])
    if not corners:
        return ()
    boxes = np.concatenate(corners).astype(np.float64)
    scores = np.concatenate(all_scores).astype(np.float64)
    frame_width, frame_height = frame
    faces: list[Face] = []
    for index in suppress(boxes, scores):
        left, top, right, bottom = (boxes[index] / scale).tolist()
        # 絵の外へはみ出た分は切る 端で半分切れた顔の中心が絵の外になると、縦への作り替えが
        # 画面の外へ枠を寄せる
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(float(frame_width), right), min(float(frame_height), bottom)
        if right <= left or bottom <= top:
            continue
        box = Box.from_corner(left, top, right - left, bottom - top, frame)
        faces.append(Face(box=box, score=float(scores[index])))
    return tuple(sorted(faces, key=lambda face: -face.score))


class FaceDetector:
    """YuNet で顔を見つける 1 本のスレッドから使う（onnxruntime の 1 つの読み込みを使い回す）"""

    def __init__(self, model: Path, *, threads: int = 2, edge: int = DETECT_EDGE) -> None:
        # 呼ばれたときに読む 部品を入れていない人の起動で onnxruntime を探さない
        import onnxruntime

        options = onnxruntime.SessionOptions()
        # 裏で調べる間も編集と再生に CPU を残す 場面の解析の既定（2 本）と同じ
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        self._session: Any = onnxruntime.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = str(self._session.get_inputs()[0].name)
        self._names: Sequence[str] = [str(output.name) for output in self._session.get_outputs()]
        self._edge = edge

    @property
    def engine(self) -> str:
        return ENGINE

    def detect(self, rgba: np.ndarray) -> tuple[Face, ...]:
        """絵（高さ x 幅 x 4 の RGBA）の中の顔"""
        blob, scale = prepare(rgba, self._edge)
        values = self._session.run(None, {self._input: blob})
        outputs = dict(zip(self._names, values, strict=True))
        height, width = rgba.shape[:2]
        return decode_faces(outputs, (blob.shape[3], blob.shape[2]), scale, (width, height))
