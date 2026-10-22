"""被写体の見積もりに顔の位置を足す所（#407）

顔の位置（#324 の画面の解析の結果）は手で組むか、偽の顔の部品（目の印の色を探す）で作る
本物の YuNet と GPU は使わないので、部品の無い機械（CI）でも走る 肌色の大きな絵と小さな顔の
動画はその場で焼く（tests/subject_video.py）
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.model import (
    Clip,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    Timeline,
    Track,
    TrackKind,
)
from sashimono.core.reframe import AUTO
from sashimono.core.timebase import FrameRate
from sashimono.engine.decode import probe_media
from sashimono.engine.decode.probe import forget_probe
from sashimono.engine.reframe import ReframePlan, ReframeRequest, plan_reframe
from sashimono.engine.scenes import SceneSignal, analyze_scenes
from sashimono.engine.subject import (
    FACE_SHARE,
    FACE_STRENGTH,
    MIN_FACE_SCORE,
    SUBJECT_BINS,
    SUBJECT_SECONDS,
    SubjectTrack,
    best_center,
    face_profiles,
    with_faces,
)
from sashimono.engine.vision import Box, Face, ScreenKind, ScreenReport, ScreenSample, scan_screen
from sashimono.engine.vision.sampling import sample_times
from tests.subject_video import (
    DRAWING,
    EYE,
    FACE_CENTER,
    FACE_RADIUS,
    HEIGHT,
    WIDTH,
    write_drawing_video,
)

#: 縦の枠の幅（1920x1080 から 1080x1920 を切るときの割合）
WINDOW = (607.5 / 1920, 1.0)
#: 手で組む絵の大きさ
FRAME = (1920, 1080)


def _track(columns: np.ndarray, *, until: float = 10.0, strength: float = 0.02) -> SubjectTrack:
    """どの行も同じ横の分布 ``columns`` の並び 縦は真ん中"""
    times = np.arange(SUBJECT_SECONDS / 2, until, SUBJECT_SECONDS)
    rows = np.zeros((times.shape[0], SUBJECT_BINS), np.float32)
    rows[:, SUBJECT_BINS // 2] = 1.0
    spread = np.tile(columns.astype(np.float32) / max(float(columns.sum()), 1e-9), (len(times), 1))
    return SubjectTrack(
        times=times,
        columns=spread,
        rows=rows,
        strength=np.full(times.shape, strength, np.float32),
    )


def _drawing_profile() -> np.ndarray:
    """枠の幅いっぱいの肌色の絵（左の 0.05〜0.36）に、動きと肌色の全部がある分布"""
    columns = np.zeros(SUBJECT_BINS, np.float32)
    columns[2:11] = 1.0
    return columns


def _face(across: float, *, size: float = 80.0, score: float = 0.9) -> Face:
    """横の ``across``（左から 0〜1）・縦の真ん中の顔"""
    return Face(Box((across - 0.5) * FRAME[0], 0.0, size, size), score)


def _samples(*faces: Face, at: tuple[float, ...] = (2.0, 5.0, 8.0)) -> tuple[ScreenSample, ...]:
    return tuple(ScreenSample(moment, faces=faces) for moment in at)


def _across(track: SubjectTrack, row: int = 0) -> float:
    return float(track.centers(WINDOW)[0][row])


class TestTheBlend:
    def test_a_small_face_beats_a_big_skin_drawing(self) -> None:
        # 壊れると、肌色に近い絵を大きく映した配信で、枠が右の小さな話し手ではなく絵に決まる
        # （#334 の実素材で 12 コマのどれにも話し手が入らなかった） 絵は枠の幅いっぱいなので、
        # 顔の重みが半分以下だと絵の側と同じか多くなる
        track = _track(_drawing_profile())
        assert _across(track) < 0.4
        blended, faced = with_faces(track, _samples(_face(0.85)), FRAME, ())
        assert _across(blended) == pytest.approx(0.85, abs=0.05)
        assert faced.all()

    def test_the_weight_must_beat_a_drawing_that_fills_the_window(self) -> None:
        # 顔の重みを 1/2 以下に下げると、上の試験の形で絵と顔が同じだけ枠に入り、枠が絵へ戻る
        assert 0.5 < FACE_SHARE < 1.0

    def test_motion_still_moves_the_frame_inside(self) -> None:
        # 顔だけで決めると、顔を調べた絵の間に人が動いた分を追えない（顔の絵は場面ごとに数枚）
        columns = np.zeros(SUBJECT_BINS, np.float32)
        columns[28] = 1.0
        blended, _faced = with_faces(_track(columns), _samples(_face(0.8)), FRAME, ())
        alone, _ = with_faces(_track(np.zeros(SUBJECT_BINS)), _samples(_face(0.8)), FRAME, ())
        assert _across(blended) > _across(alone)

    def test_an_unsure_face_is_not_used(self) -> None:
        # 誤りの顔（#324 のゲームの画面で 0.5〜0.64）を使うと、人のいない模様へ枠が寄る
        track = _track(_drawing_profile())
        unsure = _face(0.85, score=MIN_FACE_SCORE - 0.1)
        blended, faced = with_faces(track, _samples(unsure), FRAME, ())
        assert not faced.any()
        assert np.array_equal(blended.columns, track.columns)

    def test_no_face_keeps_the_motion_and_skin(self) -> None:
        # 顔の映らない素材（イラストのアバター）で何かを変えると、部品を入れただけで前より外れる
        track = _track(_drawing_profile())
        blended, faced = with_faces(track, _samples(), FRAME, ())
        assert blended is track
        assert not faced.any()

    def test_a_face_from_another_scene_is_not_used(self) -> None:
        # 場面をまたいで使うと、切り替わりの前の人の位置へ次の場面の枠が引かれる
        track = _track(_drawing_profile())
        blended, faced = with_faces(track, _samples(_face(0.85), at=(2.0,)), FRAME, (5.0,))
        times = track.times
        assert faced[times < 5.0].all()
        assert not faced[times > 5.0].any()
        assert _across(blended, int(np.flatnonzero(times > 5.0)[0])) < 0.4

    def test_one_missed_picture_is_bridged(self) -> None:
        # 1 枚だけ顔を見落とした絵（確からしさが下限を下回った）で枠が肌色の絵へ戻ると、実素材の
        # 配信で 16 コマのうち 5 コマがアバターから外れた
        track = _track(_drawing_profile(), until=30.0)
        samples = (
            ScreenSample(5.0, faces=(_face(0.85),)),
            ScreenSample(15.0),
            ScreenSample(25.0, faces=(_face(0.85),)),
        )
        _blended, faced = with_faces(track, samples, FRAME, ())
        assert faced.all()

    def test_a_long_run_without_faces_falls_back(self) -> None:
        # 顔の無い絵が続く所（人が去った）まで前の顔へ寄せると、誰もいない所を切り出し続ける
        track = _track(_drawing_profile(), until=40.0)
        samples = (
            ScreenSample(5.0, faces=(_face(0.85),)),
            ScreenSample(15.0),
            ScreenSample(25.0),
            ScreenSample(35.0),
        )
        _blended, faced = with_faces(track, samples, FRAME, ())
        times = track.times
        assert faced[times < 15.0].all()
        assert not faced[times > 16.0].any()

    def test_a_still_speaker_counts_as_a_clue(self) -> None:
        # 止まって話す人は動きが小さく、濃さが手がかりの下限を下回ると顔があっても真ん中を切る
        track = _track(np.zeros(SUBJECT_BINS), strength=0.0)
        blended, faced = with_faces(track, _samples(_face(0.2)), FRAME, ())
        assert faced.all()
        assert float(blended.strength.min()) == pytest.approx(FACE_STRENGTH)
        assert _across(blended) == pytest.approx(0.2, abs=0.05)

    def test_the_upside_is_up(self) -> None:
        # 上下を取り違えると、上の方に顔がある絵（正方形に切る作品）で枠が下の胸元へずれる
        face = Face(Box(0.0, 0.3 * FRAME[1], 80.0, 80.0), 0.9)
        profile = face_profiles([face], FRAME)
        assert profile is not None
        assert best_center(profile[1], 0.3) == pytest.approx(0.2, abs=0.05)


class TestManyFaces:
    def test_the_bigger_face_wins(self) -> None:
        # 小さな顔（後ろの客席・貼り紙）に枠を取られると、手前で話す人が切れる
        profile = face_profiles([_face(0.15, size=40.0), _face(0.8, size=160.0)], FRAME)
        assert profile is not None
        assert best_center(profile[0], WINDOW[0]) == pytest.approx(0.8, abs=0.05)

    def test_two_people_apart_frame_one_of_them(self) -> None:
        # 端から端の真ん中を取ると、離れて座った同じ大きさの 2 人の間の、誰もいない所を切り出す
        profile = face_profiles([_face(0.15), _face(0.85)], FRAME)
        assert profile is not None
        center = best_center(profile[0], WINDOW[0])
        assert min(abs(center - 0.15), abs(center - 0.85)) < 0.05

    def test_two_people_close_together_are_both_framed(self) -> None:
        # 片方へ寄せると、並んで話す 2 人の 1 人が枠の端で切れる
        profile = face_profiles([_face(0.45), _face(0.55)], FRAME)
        assert profile is not None
        assert best_center(profile[0], WINDOW[0]) == pytest.approx(0.5, abs=0.03)


class FakeFaces:
    """目の印の色（tests/subject_video.py の ``EYE``）を探して顔として返す偽の部品

    本物の YuNet と同じく、絵の左上が原点の枠を作品の座標（中心が原点 上が正）に直して返す
    """

    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.seen = 0

    @property
    def kind(self) -> ScreenKind:
        return ScreenKind.FACES

    @property
    def engine(self) -> str:
        return "偽の顔"

    def look(self, rgba: np.ndarray, seconds: float) -> ScreenSample:
        self.seen += 1
        red, green, blue = (rgba[..., index].astype(np.int16) for index in range(3))
        marks = (blue > 160) & (red < 90) & (green < 110) & (abs(blue - EYE[2]) < 60)
        ys, xs = np.nonzero(marks)
        if xs.size == 0:
            return ScreenSample(seconds)
        height, width = rgba.shape[:2]
        size = FACE_RADIUS * 2.0
        left = float(xs.mean()) - size / 2
        top = float(ys.mean()) + 4 - size / 2
        box = Box.from_corner(left, top, size, size, (width, height))
        return ScreenSample(seconds, faces=(Face(box, self.score),))


@pytest.fixture(scope="module")
def drawing(tmp_path_factory: pytest.TempPathFactory) -> tuple[MediaItem, SceneSignal]:
    path = write_drawing_video(tmp_path_factory.mktemp("drawing") / "drawing.mp4")
    forget_probe(path)
    signal = analyze_scenes(path)
    assert signal is not None
    return probe_media(path), signal


def _faces_of(media: MediaItem, signal: SceneSignal, detector: FakeFaces) -> ScreenReport:
    times = sample_times([cut.seconds for cut in signal.cuts()], float(media.duration), 4)
    report = scan_screen(media.path, times, detector, stream_index=signal.stream)
    assert report is not None
    return report


def _plan(
    media: MediaItem, signal: SceneSignal, faces: dict[MediaId, ScreenReport] | None
) -> ReframePlan:
    rate = FrameRate(30)
    project = Project(
        settings=ProjectSettings(width=WIDTH * 6, height=HEIGHT * 6, frame_rate=rate),
        timeline=Timeline(rate=rate),
        media=(media,),
    )
    clip = with_fixed_items(
        Clip(timeline_start=0, duration=110, media_id=media.id, native_size=False), picture=True
    )
    track = Track(TrackKind.VIDEO, "V1", (clip,))
    project = project.with_timeline(replace(project.timeline, tracks=(track,)))
    request = ReframeRequest(1080, 1920, AUTO, 1000, 90)
    return plan_reframe(project, request, {media.id: signal}, faces)


def _focus_share(plan: ReframePlan) -> float:
    """枠の中心（作品の左から 0〜1）"""
    return plan.shots[0].focus[0] / (WIDTH * 6) + 0.5


class TestTheMadeVideo:
    def test_without_the_parts_the_frame_stays_on_the_drawing(
        self, drawing: tuple[MediaItem, SceneSignal]
    ) -> None:
        # 部品の無い人の作り替えが変わると、#334 で測った今までの枠の決まり方が崩れる
        media, signal = drawing
        plan = _plan(media, signal, None)
        assert _focus_share(plan) < DRAWING[1] / WIDTH
        assert not plan.shots[0].faces

    def test_the_face_pulls_the_frame_off_the_drawing(
        self, drawing: tuple[MediaItem, SceneSignal]
    ) -> None:
        # 壊れると、顔の位置を調べ終えていても、枠が肌色の絵に決まって話し手が映らない
        media, signal = drawing
        report = _faces_of(media, signal, FakeFaces())
        assert any(sample.faces for sample in report.samples)
        plan = _plan(media, signal, {media.id: report})
        assert _focus_share(plan) == pytest.approx(FACE_CENTER[0] / WIDTH, abs=0.06)
        assert plan.shots[0].faces

    def test_an_unsure_face_leaves_the_frame_alone(
        self, drawing: tuple[MediaItem, SceneSignal]
    ) -> None:
        # 確からしさの低い顔で枠を動かすと、誤りの顔（模様）へ寄る
        media, signal = drawing
        report = _faces_of(media, signal, FakeFaces(score=0.6))
        plan = _plan(media, signal, {media.id: report})
        assert _focus_share(plan) == pytest.approx(_focus_share(_plan(media, signal, None)))
        assert not plan.shots[0].faces

    def test_faces_of_another_stream_are_not_used(
        self, drawing: tuple[MediaItem, SceneSignal]
    ) -> None:
        # 別の映像ストリームの顔を使うと、同じ素材の別の絵の人の位置へ枠を寄せる
        media, signal = drawing
        report = replace(_faces_of(media, signal, FakeFaces()), stream=7)
        plan = _plan(media, replace(signal, stream=0), {media.id: report})
        assert not plan.shots[0].faces
