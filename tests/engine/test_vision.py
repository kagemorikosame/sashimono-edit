"""画面の中身を調べる部品（#324 A6）の前後の処理

調べる部品そのもの（Windows の OCR・onnxruntime と YuNet）は後から入れる物で、CI には無い
ここでは部品の前後（調べる時刻の選び方・位置の向き・モデルの出力から枠を作る所・文字のつなぎ方・
結果の形）を偽の出力で押さえる 部品が入っている機械では、本物の部品でも確かめる（無ければ飛ぶ）
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from sashimono.engine.vision import (
    VISION_PACK,
    YUNET,
    Box,
    PartsMissingError,
    ScreenKind,
    ScreenReport,
    ScreenSample,
    TextLine,
    fallback_cuts,
    make_detector,
    sample_times,
)
from sashimono.engine.vision import faces as faces_module
from sashimono.engine.vision.faces import (
    STRIDES,
    decode_faces,
    prepare,
    shrink,
    suppress,
)
from sashimono.engine.vision.ocr import OcrUnavailableError, WindowsOcr, join_words
from sashimono.engine.vision.report import Face
from sashimono.engine.vision.sampling import EDGE_SECONDS, MAX_SAMPLES, MIN_STEP, scene_bounds
from sashimono.model_files import models_root

# --- 調べる時刻 ---


class TestSampleTimes:
    def test_each_scene_gets_its_share_away_from_the_cuts(self) -> None:
        # 壊れると、切り替わりの前後のぶれた絵や溶ける途中の絵を調べ、文字も顔も読めない
        cuts = (2.0, 4.0)
        times = sample_times(cuts, 6.0, 2)
        assert len(times) == 6
        for start, end in scene_bounds(cuts, 6.0):
            inside = [t for t in times if start <= t < end]
            assert len(inside) == 2
            assert all(start + EDGE_SECONDS <= t <= end - EDGE_SECONDS for t in inside)

    def test_a_short_scene_is_not_looked_at_many_times(self) -> None:
        # 壊れると、0.6 秒の場面でほぼ同じ絵を 8 回調べ、待つだけ長くなる
        times = sample_times((1.0,), 1.6, 8)
        short = [t for t in times if t >= 1.0]
        assert len(short) == math.ceil((0.6 - 2 * min(EDGE_SECONDS, 0.6 / 4)) / MIN_STEP)

    def test_a_long_material_with_many_cuts_stays_under_the_limit(self) -> None:
        # 壊れると、切り替わりが数千ある素材で何千枚も取り出し、AI が何十分も待つ
        cuts = tuple(float(index) for index in range(1, 3000))
        times = sample_times(cuts, 3000.0, 4)
        assert len(times) <= MAX_SAMPLES
        # 間引いても素材の後半まで調べる 頭だけを残すと、後半の文字や顔が無いことにされる
        assert times[-1] > 2900

    def test_the_times_are_in_order_and_inside_the_material(self) -> None:
        # 壊れると、終わりの外を取り出して何も調べられないか、時刻の順が崩れて一覧が読みにくい
        times = sample_times((5.0, 2.0, -1.0, 99.0), 10.0, 3)
        assert list(times) == sorted(times)
        assert all(0.0 <= t < 10.0 for t in times)

    def test_a_still_picture_is_looked_at_once(self) -> None:
        # 壊れると、長さの無い素材（静止画）で 1 枚も調べない
        assert sample_times((), 0.0, 4) == (0.0,)

    def test_without_scenes_the_material_is_cut_evenly(self) -> None:
        # 壊れると、場面の解析に失敗した長い素材で、頭の数枚しか調べない
        cuts = fallback_cuts(10.0)
        assert cuts == (2.0, 4.0, 6.0, 8.0)
        assert len(sample_times(cuts, 10.0, 1)) == 5


# --- 位置の向き ---


class TestCoordinates:
    def test_the_box_is_from_the_center_with_up_positive(self) -> None:
        # 壊れると、上にあるテロップを下にあると読み、AI が字幕を文字の上へ重ねて置く
        box = Box.from_corner(0, 0, 100, 50, (1920, 1080))
        assert (box.x, box.y) == (-910.0, 515.0)
        assert (box.width, box.height) == (100, 50)
        bottom = Box.from_corner(1820, 1030, 100, 50, (1920, 1080))
        assert (bottom.x, bottom.y) == (910.0, -515.0)

    def test_a_report_survives_the_cache(self) -> None:
        # 壊れると、2 度目の起動で控えから読んだ結果の位置や文字が崩れ、別の所を指す
        sample = ScreenSample(
            1.5,
            lines=(TextLine("本日のおすすめ", Box(-10.0, 20.5, 300.0, 40.0)),),
            faces=(Face(Box(1.0, -2.0, 30.0, 40.0), 0.93),),
        )
        report = ScreenReport(
            ScreenKind.TEXT, "試験", 640, 360, (sample,), planned=3, complete=True, stream=1
        )
        again = ScreenReport.from_json(report.to_json())
        assert again == report

    def test_a_broken_cache_is_refused(self) -> None:
        # 壊れると、壊れた控え（枠の大きさが負など）をそのまま AI へ返す
        data = ScreenReport(
            ScreenKind.FACES, "試験", 10, 10, (), planned=0, complete=True
        ).to_json()
        data["samples"] = [{"seconds": 0, "faces": [{"x": 0, "y": 0, "width": -1, "height": 1}]}]
        with pytest.raises(ValueError):
            ScreenReport.from_json(data)
        with pytest.raises(ValueError):
            ScreenReport.from_json({"kind": "text"})


# --- 顔の検出の前後 ---


def _outputs(width: int, height: int) -> dict[str, np.ndarray]:
    """何も見つからない出力（どの候補も確からしさ 0）"""
    outputs: dict[str, np.ndarray] = {}
    for stride in STRIDES:
        count = (width // stride) * (height // stride)
        outputs[f"cls_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"obj_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"bbox_{stride}"] = np.zeros((1, count, 4), np.float32)
        outputs[f"kps_{stride}"] = np.zeros((1, count, 10), np.float32)
    return outputs


def _plant(
    outputs: dict[str, np.ndarray],
    stride: int,
    width: int,
    row: int,
    column: int,
    score: float,
    size: float,
) -> None:
    index = row * (width // stride) + column
    outputs[f"cls_{stride}"][0, index, 0] = score
    outputs[f"obj_{stride}"][0, index, 0] = score
    # 中心をこまの真ん中へ 大きさは exp(値) * stride
    outputs[f"bbox_{stride}"][0, index] = (
        0.5,
        0.5,
        math.log(size / stride),
        math.log(size / stride),
    )


class TestFaceDecoding:
    def test_a_face_comes_back_where_it_is_in_the_picture(self) -> None:
        # 壊れると、顔の位置を縮めた絵の画素のまま返し、縦への作り替えが画面の左上へ寄る
        outputs = _outputs(640, 384)
        # 縮めた絵（倍率 0.5）の左上寄り こま (2, 3) の中心は (3.5*16, 2.5*16) = (56, 40)
        _plant(outputs, 16, 640, 2, 3, 0.95, 32)
        faces = decode_faces(outputs, (640, 384), 0.5, (1280, 720))
        assert len(faces) == 1
        face = faces[0]
        # 元の絵では中心 (112, 80)・大きさ 64 中心が原点で上が正の向きに直す
        assert face.box.x == pytest.approx(112 - 640)
        assert face.box.y == pytest.approx(360 - 80)
        assert face.box.width == pytest.approx(64)
        assert face.score == pytest.approx(0.95)

    def test_overlapping_boxes_keep_only_the_surest(self) -> None:
        # 壊れると、1 つの顔が 2 つに数えられ、AI が 2 人で話していると取り違える
        outputs = _outputs(640, 384)
        _plant(outputs, 8, 640, 10, 10, 0.9, 40)
        _plant(outputs, 8, 640, 10, 11, 0.8, 40)
        faces = decode_faces(outputs, (640, 384), 1.0, (640, 384))
        assert [round(face.score, 2) for face in faces] == [0.9]

    def test_unsure_candidates_are_not_kept(self) -> None:
        # 壊れると、確からしさの低い候補（壁の模様など）を顔として覚える
        outputs = _outputs(640, 384)
        _plant(outputs, 32, 640, 1, 1, 0.3, 64)
        assert decode_faces(outputs, (640, 384), 1.0, (640, 384)) == ()

    def test_a_face_at_the_edge_is_clipped_into_the_picture(self) -> None:
        # 壊れると、端で半分切れた顔の中心が絵の外になり、縦の枠を画面の外へ寄せる
        outputs = _outputs(640, 384)
        _plant(outputs, 32, 640, 0, 0, 0.95, 120)
        faces = decode_faces(outputs, (640, 384), 1.0, (640, 384))
        assert len(faces) == 1
        box = faces[0].box
        assert box.x - box.width / 2 >= -320
        assert box.y + box.height / 2 <= 192

    def test_the_wrong_number_of_outputs_is_refused(self) -> None:
        # 壊れると、別の大きさの絵の出力を読み違え、関係の無い所に顔を出す
        outputs = _outputs(640, 384)
        with pytest.raises(ValueError, match="数が合わない"):
            decode_faces(outputs, (320, 384), 1.0, (640, 384))

    def test_suppress_keeps_separate_faces(self) -> None:
        # 壊れると、離れた 2 人の片方を重なりとして消す
        corners = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [1, 1, 11, 11]], float)
        kept = suppress(corners, np.array([0.9, 0.8, 0.7]))
        assert kept == [0, 1]

    def test_the_picture_is_shrunk_and_padded_for_the_model(self) -> None:
        # 壊れると、モデルが読めない大きさ（32 の倍数でない）を渡して落ちるか、大きな顔を見落とす
        rgba = np.zeros((1080, 1920, 4), np.uint8)
        rgba[:, :, 0] = 200  # 赤
        blob, scale = prepare(rgba, 640)
        assert blob.shape == (1, 3, 384, 640)
        assert scale == pytest.approx(1 / 3)
        # BGR の並びで渡す（青・緑・赤） 赤は 3 番目
        assert blob[0, 2, 0, 0] == pytest.approx(200)
        assert blob[0, 0, 0, 0] == 0
        # 埋めた所は 0
        assert blob[0, 2, 383, 0] == 0

    def test_shrinking_averages_instead_of_skipping(self) -> None:
        # 壊れると、細い線や小さな顔が縮めたときにまだらに消え、見落とす
        image = np.zeros((4, 4, 1), np.uint8)
        image[0, 0] = 255
        small = shrink(image, 2, 2)
        assert small[0, 0, 0] == pytest.approx(255 / 4)


# --- 文字のつなぎ方 ---


class TestWords:
    def test_japanese_words_are_joined_without_spaces(self) -> None:
        # 壊れると、「本 日 の」のまま AI へ渡り、字幕や検索の言葉と突き合わない
        assert join_words(["本", "日", "の", "お", "す", "す", "め", "120", "円"]) == (
            "本日のおすすめ120円"
        )

    def test_english_words_keep_their_spaces(self) -> None:
        # 壊れると、英語のテロップが「HelloWorld」とつながって読めなくなる
        assert join_words(["Hello", "World", "2026"]) == "Hello World 2026"


# --- 部品が無いとき ---


class TestLoadingOrder:
    def test_onnxruntime_is_loaded_before_the_ocr_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、文字を先に頼まれた後で顔を頼まれたとき、winrt の古い msvcp140.dll の上で
        # onnxruntime を読み、アプリごと落ちる（手元の Python 3.14 で access violation になった）
        import builtins

        asked: list[str] = []
        real_import = builtins.__import__

        def recording(name: str, *arguments: object, **options: object) -> object:
            if name == "onnxruntime" or name.startswith("winrt"):
                asked.append(name)
                raise ImportError(name=name)
            return real_import(name, *arguments, **options)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", recording)
        with pytest.raises(OcrUnavailableError):
            WindowsOcr()
        assert asked[0] == "onnxruntime"
        assert any(name.startswith("winrt") for name in asked[1:])


class TestMissingParts:
    def test_without_the_model_faces_are_refused_with_the_way_to_install(self) -> None:
        # 壊れると、部品の無い機械で AI が理由の分からない失敗を受け取り、何を入れればよいか
        # 分からない
        assert not YUNET.present(models_root())
        with pytest.raises(PartsMissingError, match="画面の解析の部品"):
            make_detector(ScreenKind.FACES)

    def test_the_pack_installs_onnxruntime_and_the_ocr_wrapper_and_fetches_the_model(
        self,
    ) -> None:
        # 壊れると、導入のボタンで入れた後も顔の検出のモデルが無く、使う所で初めて落ちる
        names = " ".join(VISION_PACK.required)
        assert "onnxruntime" in names
        assert "winrt-Windows.Media.Ocr" in names
        assert VISION_PACK.files == (YUNET,)
        assert YUNET.url.startswith("https://")
        # 取得元はコミットで固定する 枝の名前（main）を指すと、中身が変わって照らしで止まる
        assert "/main/" not in YUNET.url
        assert len(YUNET.sha256) == 64

    def test_the_model_is_listed_with_its_license(self) -> None:
        # 壊れると（取得元か中身を替えて一覧を直し忘れると）、利用者の機械へ落とす他人の作った物の
        # 使用許諾と出どころが、どこにも書かれないまま配られる
        root = Path(__file__).resolve().parents[2]
        notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        assert YUNET.sha256 in notices
        assert str(YUNET.size) in notices
        commit = YUNET.url.split("/opencv_zoo/")[1].split("/")[0]
        assert commit in notices
        policy = (root / "docs" / "code-signing-policy.md").read_text(encoding="utf-8")
        assert YUNET.name in policy

    def test_the_model_is_not_kept_in_the_repository(self) -> None:
        # 壊れると、他人の作った物（モデル）をリポジトリに入れて配ることになる（配布物の決まり）
        root = Path(__file__).resolve().parents[2]
        for folder in ("src", "tests", "tools", "licenses"):
            assert not list((root / folder).glob("**/*.onnx")), folder


# --- 本物の部品（入っている機械だけ） ---


def _real_model() -> Path | None:
    if importlib.util.find_spec("onnxruntime") is None:
        return None
    path = YUNET.path(models_root())
    return path if YUNET.present(models_root()) else None


class TestRealParts:
    def test_the_real_model_reads_its_outputs(self) -> None:
        # 壊れると（モデルの出力の名前や並びが前後の処理と食い違うと）、本物の素材で顔が
        # 1 つも出ない
        model = _real_model()
        if model is None:
            pytest.skip(
                "onnxruntime か顔の検出のモデルが無い（画面の解析の部品を入れた機械で走る）"
            )
        detector = faces_module.FaceDetector(model)
        # 何も無い絵では顔を出さない（偽の顔を出すと、縦の枠が誤った所へ寄る）
        assert detector.detect(np.zeros((360, 640, 4), np.uint8)) == ()

    def test_the_real_ocr_reads_japanese(self) -> None:
        # 壊れると、Windows の OCR を呼ぶ所の形が合わず、日本語のテロップを 1 つも読めない
        if importlib.util.find_spec("winrt") is None:
            pytest.skip("Windows の OCR を呼ぶ部品が無い（画面の解析の部品を入れた機械で走る）")
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QFont, QImage, QPainter

        image = QImage(640, 360, QImage.Format.Format_RGBA8888)
        image.fill(QColor("#1e3c5a"))
        painter = QPainter(image)
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Meiryo", 40))
        # 上の方に書く 位置の向き（上が正）も一緒に見る
        painter.drawText(40, 20, 560, 80, Qt.AlignmentFlag.AlignLeft, "本日のおすすめ")
        painter.end()
        rows = np.frombuffer(image.constBits(), np.uint8).reshape(360, image.bytesPerLine())
        rgba = rows[:, : 640 * 4].reshape(360, 640, 4).copy()
        detector = make_detector(ScreenKind.TEXT)
        sample = detector.look(rgba, 0.0)
        assert [line.text for line in sample.lines] == ["本日のおすすめ"]
        assert sample.lines[0].box.y > 0
        # 何も無い絵では何も読まない（模様を文字と取り違えると、AI が無い字幕を避けに行く）
        assert detector.look(np.zeros((120, 320, 4), np.uint8), 0.0).lines == ()
