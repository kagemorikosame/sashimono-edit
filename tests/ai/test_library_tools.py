"""登録したフォルダの効果音と BGM を AI が使う（#522）

search_library で AI が使ってよい分類のフォルダだけを探せること、place_sound と
propose_direction が登録したフォルダの音を取り込んで使い、取り消し 1 回で素材ごと戻ること、
使ってよい分類の外と登録の無いときは付けずに理由を返すことを見る
登録と見て回る所そのものは tests/core/test_material_library.py
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sashimono.ai.direction_plan import SOUND_TRACK_NAME, proposal_for
from sashimono.ai.host import ToolError
from sashimono.ai.operations import LIBRARY_MEASURE_PER_CALL, find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.core.commands import AddMedia
from sashimono.core.direction import SOUND
from sashimono.core.direction_recipes import BOLD, DirectionTaste
from sashimono.core.material_library import (
    CATEGORY_MUSIC,
    CATEGORY_SOUND_EFFECT,
    CATEGORY_VIDEO,
    LibraryFolder,
    MaterialLibrary,
    Measure,
    MeasureBook,
    file_id,
)
from sashimono.core.model import (
    AudioStreamInfo,
    Clip,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
)
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run
from tests.conftest import RATE_30

_SECONDS = 120

_AUDIO = (
    AudioStreamInfo(
        index=0, sample_rate=48000, channels=2, time_base=Fraction(1, 48000), codec="pcm_s16le"
    ),
)


class _LibraryHost(FakeHost):
    """開くたびに別の素材（別の ID）を返す 本物の probe と同じく、ファイルごとに素材が違う"""

    def __init__(self, project: Project, lengths: dict[str, float]) -> None:
        super().__init__(project)
        self.lengths = lengths
        #: 測った素材 測り直さないことを見る
        self.measured_calls: list[Path] = []

    def probe(self, path: Path) -> MediaItem:
        self.probed.append(path)
        return MediaItem(
            path=path,
            duration=Fraction(self.lengths[path.name]).limit_denominator(1000),
            audio_streams=_AUDIO,
        )


@pytest.fixture
def lengths() -> dict[str, float]:
    return {}


@pytest.fixture
def folders(tmp_path: Path, lengths: dict[str, float]) -> dict[str, Path]:
    """効果音・BGM・動画のフォルダ 中に音と、分類に合わない物を置く"""
    made: dict[str, Path] = {}
    for category, names in (
        (CATEGORY_SOUND_EFFECT, {"ドン.wav": 0.8, "拍手/拍手大.wav": 2.5, "説明.png": 0}),
        (CATEGORY_MUSIC, {"のんびり.mp3": 95.0, "短いジングル.mp3": 6.0}),
        (CATEGORY_VIDEO, {"空.mp4": 10.0, "隠し.wav": 1.0}),
    ):
        root = tmp_path / category
        for name, seconds in names.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
            lengths[target.name] = seconds
        made[category] = root
    return made


def _measurer(lengths: dict[str, float], calls: list[Path]) -> Any:
    def measure(path: Path) -> Measure | None:
        calls.append(path)
        return Measure(
            seconds=lengths[path.name], has_audio=True, has_video=False, loudness=-20.0, peak=-3.0
        )

    return measure


@pytest.fixture
def work(
    video_media: MediaItem, folders: dict[str, Path], lengths: dict[str, float]
) -> _LibraryHost:
    media = replace(video_media, duration=Fraction(_SECONDS))
    project = Project.create(ProjectSettings(frame_rate=RATE_30))
    project = AddMedia(media).apply(project)
    clip = Clip(timeline_start=0, duration=_SECONDS * 30, media_id=media.id)
    track = Track(kind=TrackKind.VIDEO, name="V1", clips=(clip,))
    host = _LibraryHost(project.with_timeline(replace(project.timeline, tracks=(track,))), lengths)
    host.taste = DirectionTaste(strength=BOLD)
    host.material = MaterialLibrary(
        folders=tuple(LibraryFolder(str(path), key) for key, path in folders.items()),
        book=MeasureBook(),
        measure=_measurer(lengths, host.measured_calls),
    )
    return host


def _file(work: _LibraryHost, name: str) -> dict[str, Any]:
    rows = run(work, "search_library", query=name)["files"]
    return next(row for row in rows if row["name"] == name)


def _cues() -> list[dict[str, Any]]:
    return [
        {"signal": "laughter", "start_frame": t * 30, "end_frame": t * 30 + 30, "score": 0.9}
        for t in range(5, _SECONDS, 10)
    ]


class TestSearch:
    def test_is_read_only_and_declares_names_and_settings(self) -> None:
        # 読むだけでないと毎回確かめが出る 名前を返すのに宣言しないと、送らないと決めた人の所でも
        # 素材の名前が外へ出る
        operation = find_operation("search_library")
        assert operation is not None
        assert operation.safety is Safety.READ_ONLY
        assert {Outbound.FILES, Outbound.APP_SETTINGS} <= operation.sends
        assert {"name", "folder"} <= operation.name_keys

    def test_without_folders_it_says_so_and_finds_nothing(self, video_media: MediaItem) -> None:
        # 壊れると、登録が無いのに AI が手持ちの音を探し回るか、黙って効果音を付けずに終わる
        host = FakeHost(Project.create(ProjectSettings(frame_rate=RATE_30)))
        result = run(host, "search_library")
        assert result["files"] == []
        assert "登録されていません" in result["note"]

    def test_only_the_allowed_categories_are_seen(self, work: _LibraryHost) -> None:
        # 壊れると、本人が AI に使わせないと決めた動画のフォルダの物を AI が選んで置く
        result = run(work, "search_library")
        names = {row["name"] for row in result["files"]}
        assert names == {"ドン.wav", "拍手大.wav", "のんびり.mp3", "短いジングル.mp3"}
        assert {row["category"] for row in result["files"]} == {
            CATEGORY_SOUND_EFFECT,
            CATEGORY_MUSIC,
        }
        with pytest.raises(ToolError, match="使わせない"):
            run(work, "search_library", category=CATEGORY_VIDEO)

    def test_the_folder_name_is_a_clue(self, work: _LibraryHost) -> None:
        # 下のフォルダの名前（拍手）で引けないと、名前の付け方がまちまちな効果音集で選べない
        rows = run(work, "search_library", query="拍手")["files"]
        assert [row["name"] for row in rows] == ["拍手大.wav"]
        assert rows[0]["folder"] == "拍手"

    def test_length_filters_pick_short_effects_and_long_music(self, work: _LibraryHost) -> None:
        # 長さの手がかりが効かないと、効果音の所に 95 秒の曲を置く
        short = run(work, "search_library", max_seconds=10)["files"]
        assert {row["name"] for row in short} == {"ドン.wav", "拍手大.wav", "短いジングル.mp3"}
        long = run(work, "search_library", category=CATEGORY_MUSIC, min_seconds=30)["files"]
        assert [row["name"] for row in long] == ["のんびり.mp3"]
        assert long[0]["seconds"] == 95.0
        assert long[0]["loudness_lufs"] == -20.0

    def test_measures_a_few_at_a_time_and_remembers(
        self, work: _LibraryHost, folders: dict[str, Path], lengths: dict[str, float]
    ) -> None:
        # 壊れると、数百の効果音を一度に測って画面が止まるか、毎回測り直して待たせる
        many = folders[CATEGORY_SOUND_EFFECT] / "たくさん"
        many.mkdir()
        for number in range(LIBRARY_MEASURE_PER_CALL + 5):
            (many / f"音{number:02}.wav").write_bytes(b"x")
            lengths[f"音{number:02}.wav"] = 1.0
        calls = work.measured_calls
        first = run(work, "search_library", category=CATEGORY_SOUND_EFFECT)
        assert len(calls) == LIBRARY_MEASURE_PER_CALL
        assert first["not_measured"] == 7
        assert "測っていません" in first["note"]
        second = run(work, "search_library", category=CATEGORY_SOUND_EFFECT)
        assert second["not_measured"] == 0
        assert len(calls) == LIBRARY_MEASURE_PER_CALL + 7
        assert len(set(calls)) == len(calls)

    def test_a_vanished_folder_is_reported(
        self, work: _LibraryHost, folders: dict[str, Path]
    ) -> None:
        # 壊れると、外付けのドライブを外しただけで道具が落ちる
        gone = LibraryFolder(str(folders[CATEGORY_MUSIC] / "無い"), CATEGORY_MUSIC)
        work.material = replace(work.material, folders=(*work.material.folders, gone))
        result = run(work, "search_library")
        assert result["folders_missing"] == 1
        assert "見つからない" in result["note"]


class TestPlaceSound:
    def test_imports_and_places_as_one_undo(self, work: _LibraryHost) -> None:
        # 取り込みと置くのが別の段だと、取り消した後に素材一覧に音だけが残る
        before = work.document.project
        row = _file(work, "のんびり.mp3")
        result = run(
            work, "place_sound", library_file_id=row["file_id"], at_frame=0, until_frame=600
        )
        project = work.document.project
        added = [m for m in project.media if m.path.name == "のんびり.mp3"]
        assert len(added) == 1 and result["imported"] is True
        assert result["name"] == "のんびり.mp3" and result["folder"] == CATEGORY_MUSIC
        assert "duck_audio" in result["library_note"]
        assert added[0].id in work.analyzed
        work.document.undo()
        assert work.document.project == before

    def test_a_file_imported_before_is_reused(self, work: _LibraryHost) -> None:
        # 壊れると、同じ音を使うたびに素材一覧に同じ名前が並ぶ
        row = _file(work, "ドン.wav")
        run(work, "place_sound", library_file_id=row["file_id"], at_frame=0)
        again = run(work, "place_sound", library_file_id=row["file_id"], at_frame=300)
        assert again["imported"] is False
        assert sum(m.path.name == "ドン.wav" for m in work.document.project.media) == 1

    def test_files_outside_the_allowed_categories_cannot_be_named(
        self, work: _LibraryHost, folders: dict[str, Path]
    ) -> None:
        # 壊れると、一覧に出ない物でも印を作れば置けてしまい、本人の選んだ分類の線が意味を失う
        hidden = file_id(folders[CATEGORY_VIDEO] / "隠し.wav")
        before = work.document.project
        with pytest.raises(ToolError, match="見つかりません"):
            run(work, "place_sound", library_file_id=hidden)
        assert work.document.project == before

    def test_needs_exactly_one_source(self, work: _LibraryHost) -> None:
        # 壊れると、両方を渡されたときに片方を黙って捨て、頼んでいない音を置く
        row = _file(work, "ドン.wav")
        media = work.document.project.media[0]
        with pytest.raises(ToolError, match="どちらか 1 つ"):
            run(work, "place_sound")
        with pytest.raises(ToolError, match="どちらか 1 つ"):
            run(work, "place_sound", media_id=str(media.id), library_file_id=row["file_id"])

    def test_place_media_takes_a_library_file_too(self, work: _LibraryHost) -> None:
        # 壊れると、登録したフォルダの動画や BGM を place_media で置けず、取り消しで素材が残る
        before = work.document.project
        row = _file(work, "短いジングル.mp3")
        result = run(work, "place_media", library_file_id=row["file_id"], at_frame=0)
        assert result["placed"] == "短いジングル.mp3"
        work.document.undo()
        assert work.document.project == before


class TestDirectionPlan:
    def test_plans_without_importing_and_applies_as_one_undo(self, work: _LibraryHost) -> None:
        # 計画を出しただけで素材が増えると、〔やめる〕を押しても素材一覧に音が残る
        # 付けた後は取り込みと置いた音が 1 回の取り消しで戻る
        before = work.document.project
        row = _file(work, "ドン.wav")
        items = run(work, "propose_direction", cues=_cues(), sound_file_id=row["file_id"])["items"]
        sounds = [item for item in items if item["kind"] == SOUND]
        assert sounds
        assert work.document.project == before
        view = proposal_for(work.document).view()
        assert view is not None
        assert all(item.sound_name == "ドン.wav" for item in view.items if item.kind == SOUND)
        result = proposal_for(work.document).apply(work.document)
        assert result is not None and result.skipped == ()
        project = work.document.project
        assert sum(m.path.name == "ドン.wav" for m in project.media) == 1
        track = next(t for t in project.timeline.tracks if t.name == SOUND_TRACK_NAME)
        assert len(track.clips) == len(sounds)
        work.document.undo()
        assert work.document.project == before

    def test_long_music_is_not_an_effect(self, work: _LibraryHost) -> None:
        # 壊れると、効果音の所に 95 秒の曲が何本も重なる
        row = _file(work, "のんびり.mp3")
        with pytest.raises(ToolError, match="秒以下"):
            run(work, "propose_direction", cues=_cues(), sound_file_id=row["file_id"])

    def test_without_a_sound_it_says_how_to_add_one(self, work: _LibraryHost) -> None:
        # 壊れると、効果音の出どころを AI が知らず、登録した音があっても付けない
        result = run(work, "propose_direction", cues=_cues())
        assert "sound_file_id" in result["note"]
        assert "search_library" in result["note"]


class TestImport:
    def test_reads_into_the_pool_only_as_one_undo(self, work: _LibraryHost) -> None:
        # 壊れると、エクスプローラーの〔素材一覧へ読み込む〕と同じ頼みで AI がタイムラインへ置き、
        # 取り消した後も素材一覧に音が残る
        before = work.document.project
        row = _file(work, "ドン.wav")
        result = run(work, "import_media", library_file_ids=[row["file_id"]], pool_only=True)
        project = work.document.project
        assert [m.path.name for m in project.media if m.path.name == "ドン.wav"] == ["ドン.wav"]
        assert project.timeline == before.timeline
        assert len(result["imported"]) == 1
        work.document.undo()
        assert work.document.project == before

    def test_needs_something_to_read(self, work: _LibraryHost) -> None:
        # 壊れると、空の頼みで空の取り消しの段が積まれる
        with pytest.raises(ToolError, match="どちらか"):
            run(work, "import_media")


class TestFavorites:
    def test_favorites_only_stays_inside_the_allowed_categories(
        self, work: _LibraryHost, folders: dict[str, Path]
    ) -> None:
        # 壊れると、「いつもの効果音」の頼みで本人が AI に使わせないと決めた分類の物まで選ぶ
        effects = folders[CATEGORY_SOUND_EFFECT]
        work.material = replace(
            work.material,
            favorites=(str(effects / "拍手"), str(folders[CATEGORY_VIDEO] / "隠し.wav")),
        )
        rows = run(work, "search_library", favorites_only=True)["files"]
        assert [row["name"] for row in rows] == ["拍手大.wav"]
        assert rows[0]["favorite"] is True
        every = run(work, "search_library")["files"]
        assert {row["name"] for row in every if not row["favorite"]} >= {"ドン.wav"}
