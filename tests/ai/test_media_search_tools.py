"""AI の道具 search_media・media_scene_sheet・remember_scene_descriptions（#326 A8）

「あの場面」を探す 作品の中の素材すべての字幕を、表記のゆれを揃えた部分一致で引き、
頼まれたときだけ AI が場面ごとの絵を見て説明文を作って覚え、同じ道具で引く
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import EditorHost, ToolError
from sashimono.ai.operations import (
    MAX_RESULT_CHARS,
    ImageResult,
    Operation,
    Pending,
    find_operation,
    finish,
)
from sashimono.ai.outbound import Outbound, SendChoice
from sashimono.ai.safety import Safety
from sashimono.ai.sheet import MAX_SHEET_EDGE
from sashimono.core.commands import AddClip, SetTranscript
from sashimono.core.io.serialize import json_text
from sashimono.core.model import MediaId, MediaItem, Track, TrackKind, Transcript
from sashimono.core.model.transcript import TranscriptSegment
from sashimono.core.search import SceneNote, hit_places
from sashimono.engine.cache import SceneState
from sashimono.engine.scenes import SceneSignal
from tests.ai.conftest import FakeHost
from tests.ai.test_bridge import call_in_thread, pump_until
from tests.conftest import make_clip


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None, name
    return found


SEARCH = _operation("search_media")
SHEET = _operation("media_scene_sheet")
REMEMBER = _operation("remember_scene_descriptions")

FPS = 30


def _signal(cuts: list[float], *, until: float = 10.0, complete: bool = True) -> SceneSignal:
    """30fps で ``until`` 秒まで ``cuts`` の秒で跳ねる差の並び"""
    times = np.arange(1, round(until * FPS) + 1, dtype=np.float64) / FPS
    pixel = np.full(times.shape, 0.01, np.float32)
    histogram = np.full(times.shape, 0.01, np.float32)
    for moment in cuts:
        index = round(moment * FPS) - 1
        pixel[index], histogram[index] = 0.3, 0.8
    return SceneSignal(times, pixel, histogram, analyzed=until, complete=complete)


def _media(host: FakeHost) -> MediaItem:
    return host.document.project.media[0]


def _result(operation: Operation, host: FakeHost, **arguments: Any) -> dict[str, Any]:
    result = operation(host, arguments)
    assert isinstance(result, dict), result
    return result


def _off_timeline(host: FakeHost, *lines: str) -> MediaItem:
    """タイムラインに置いていない素材を足し、字幕を付ける"""
    media = MediaItem(
        path=Path("C:/素材/置いていない.mp4"),
        duration=Fraction(60),
        video_streams=_media(host).video_streams,
        audio_streams=_media(host).audio_streams,
    )
    transcript = Transcript(
        segments=tuple(
            TranscriptSegment(start=Fraction(10 * n), end=Fraction(10 * n + 2), text=text)
            for n, text in enumerate(lines)
        )
    )
    project = host.document.project
    host.document.reset(replace(project, media=(*project.media, media)))
    host.apply_commands([SetTranscript(media.id, transcript)], "字幕")
    found = host.document.project.find_media(media.id)
    assert found is not None
    return found


class TestDeclarations:
    def test_searching_only_reads(self) -> None:
        # 壊れると、探すたびに「変更を承認しますか」が出て、言い換えで何度も引けない
        assert SEARCH.safety is Safety.READ_ONLY
        assert SHEET.safety is Safety.READ_ONLY

    def test_remembering_does_not_change_the_project(self) -> None:
        # 説明は作品の外（engine/cache）に覚えるので作品を変えない 壊れて変える道具にすると、
        # 場面の数だけ確かめの窓が出るのに、取り消し（Ctrl+Z）では説明が消えない
        assert REMEMBER.safety is Safety.READ_ONLY

    def test_search_declares_subtitle_text(self) -> None:
        # 壊れると、字幕と文字を送らないと決めた人の字幕の文が、尋ねずに外へ出る
        assert SEARCH.sends == {Outbound.PROJECT, Outbound.TEXT}

    def test_the_sheet_declares_frames(self) -> None:
        # 壊れると、素材の絵を送らないと決めた人の場面の絵が、尋ねずに外へ出る
        # 並べられないときの断りの文には本人の設定の長い辺の上限（#344）が載るので、
        # アプリの設定も送る物に数える
        assert SHEET.sends == {Outbound.PROJECT, Outbound.FRAMES, Outbound.APP_SETTINGS}

    def test_remembering_returns_only_counts(self) -> None:
        # 壊れて文を返すようにすると、字幕と文字を送らないと決めた人の説明が尋ねずに外へ出る
        # 文を返すなら TEXT を書く（宣言は結果の中身の見張りも見る）
        assert REMEMBER.sends == {Outbound.PROJECT}


class TestSearch:
    def test_media_off_the_timeline_are_found(self, host: FakeHost) -> None:
        # 壊れると、置いていない素材の「あの場面」は get_subtitles に出ず、AI が見つけられない
        media = _off_timeline(host, "海に行きました", "夜はラーメンを食べました")
        result = _result(SEARCH, host, queries=["らーめん"])
        assert [(hit["media_id"], hit["text"]) for hit in result["hits"]] == [
            (str(media.id), "夜はラーメンを食べました")
        ]
        hit = result["hits"][0]
        assert hit["kind"] == "subtitle"
        assert (hit["source_seconds"], hit["source_end_seconds"]) == (10.0, 12.0)
        assert hit["audio"] == 1
        # 置いていないので、タイムラインの位置は空 AI は place_media で置いてから見る
        assert hit["timeline"] == []

    def test_a_placed_media_comes_with_its_timeline_frame(self, host: FakeHost) -> None:
        # 壊れると、見つけた行のタイムラインの位置を AI が素材の秒から数え直して外す
        result = _result(SEARCH, host, queries=["編集"])
        hit = result["hits"][0]
        assert hit["text"] == "編集ソフトを"
        clip = host.document.project.timeline.tracks[0].clips[0]
        assert hit["timeline"] == [{"clip_id": str(clip.id), "frame": 120}]

    def test_a_subtitle_points_to_the_clip_that_plays_its_language(self, host: FakeHost) -> None:
        # 壊れると、2 か国語の素材で別の言語の音のクリップの位置を返し、AI が探した言葉の
        # 聞こえない所を preview で確かめる
        first = _media(host).audio_streams[0]
        second = replace(first, index=2)
        media = MediaItem(
            path=Path("C:/素材/二か国語.mp4"),
            duration=Fraction(60),
            video_streams=_media(host).video_streams,
            audio_streams=(first, second),
        ).with_transcript(
            Transcript(
                segments=(TranscriptSegment(start=Fraction(4), end=Fraction(6), text="英語で話す"),)
            ),
            2,
        )
        project = host.document.project
        tracks = tuple(
            Track(
                kind=TrackKind.AUDIO,
                name=f"A{stream}",
                clips=(replace(make_clip(start, 300, media), stream_index=stream),),
            )
            for start, stream in ((0, 1), (600, 2))
        )
        host.document.reset(
            replace(
                project,
                media=(*project.media, media),
                timeline=replace(project.timeline, tracks=(*project.timeline.tracks, *tracks)),
            )
        )
        result = _result(SEARCH, host, queries=["英語"])
        (hit,) = result["hits"]
        assert [place["frame"] for place in hit["timeline"]] == [600 + 120]

    def test_the_hint_for_nothing_found_tells_how_to_widen(self, host: FakeHost) -> None:
        # 壊れると、見つからなかったときに AI が言い換えも説明作りもせずに諦める
        result = _result(SEARCH, host, queries=["カレー"])
        assert result["hits"] == []
        assert "言い換え" in result["note"]
        assert "media_scene_sheet" in result["note"]

    def test_media_without_subtitles_are_listed(self, host: FakeHost) -> None:
        # 壊れると、起こしていない素材があるのに「無い」と答え、transcribe を勧めない
        bare = MediaItem(
            path=Path("C:/素材/起こしていない.mp4"),
            duration=Fraction(5),
            audio_streams=_media(host).audio_streams,
        )
        # 音の無い素材は起こせないので並べない 並べると AI が起こせない素材に transcribe を頼む
        still = MediaItem(path=Path("C:/素材/写真.png"))
        project = host.document.project
        host.document.reset(replace(project, media=(*project.media, bare, still)))
        result = _result(SEARCH, host, queries=["カレー"])
        assert result["no_subtitles"] == [str(bare.id)]

    def test_remembered_scene_notes_are_found(self, host: FakeHost) -> None:
        # 壊れると、話していない場面（絵だけの場面）を説明で引けない
        media = _media(host)
        _result(
            REMEMBER,
            host,
            media_id=str(media.id),
            scenes=[{"from_seconds": 2.0, "to_seconds": 6.5, "text": "夜の屋台で湯気が上がる"}],
        )
        result = _result(SEARCH, host, queries=["屋台"])
        hit = result["hits"][0]
        assert (hit["kind"], hit["source_seconds"], hit["source_end_seconds"]) == (
            "scene",
            2.0,
            6.5,
        )
        assert hit["timeline"][0]["frame"] == 60

    def test_a_broken_word_list_is_refused(self, host: FakeHost) -> None:
        # 壊れると、null や数の語を "None" などの文字として引き、見当違いの行を返す
        for words in ([], [None], ["  "], "ラーメン"):
            with pytest.raises(ToolError):
                SEARCH(host, {"queries": words})

    def test_a_wrong_kind_is_refused(self, host: FakeHost) -> None:
        # 壊れると、綴りを間違えた種類を黙って捨て、何も引かずに「無い」と答える
        with pytest.raises(ToolError, match="kinds"):
            SEARCH(host, {"queries": ["a"], "kinds": ["subtitles"]})

    def test_many_hits_fit_in_one_answer(self, host: FakeHost) -> None:
        # 壊れると、数時間分の素材で返事が上限を越え、打ち切られた残りを AI が読めない
        _off_timeline(host, *(f"{n} 行目の長い字幕 " + "あ" * 300 for n in range(600)))
        result = _result(SEARCH, host, queries=["行目"])
        assert len(json_text(result)) <= MAX_RESULT_CHARS
        assert result["total"] == 600
        assert result["next_offset"] == result["count"]

    def test_only_the_page_is_placed_on_the_timeline(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、短い語で何万行も当たったときに返事に入らない行までタイムラインの位置を
        # 求め、詰めて何千本のクリップがある作品では編集画面が長く止まる（PR #399 の指摘）
        from sashimono.ai import operations

        _off_timeline(host, *(f"{n} 行目" for n in range(600)))
        looked: list[float] = []
        original = hit_places

        def counted(*args: Any, **kwargs: Any) -> Any:
            looked.append(args[2])
            return original(*args, **kwargs)

        monkeypatch.setattr(operations, "hit_places", counted)
        result = _result(SEARCH, host, queries=["行目"], limit=10)
        assert result["count"] == 10
        assert len(looked) == 10
        assert (result["total"], result["next_offset"]) == (600, 10)
        looked.clear()
        last = _result(SEARCH, host, queries=["行目"], offset=595, limit=10)
        # 続きを読むときも、そのページの分だけ求め、件数と続きの有無は全部の当たりで数える
        assert len(looked) == 5
        assert (last["total"], last["offset"], last["count"]) == (600, 595, 5)
        assert "next_offset" not in last
        assert last["hits"][0]["text"] == "595 行目"


class TestTheSheet:
    def _host(self, host: FakeHost, cuts: list[float]) -> MediaItem:
        media = _media(host)
        host.scene_states[media.id] = SceneState(_signal(cuts), running=False)
        return media

    def test_one_frame_per_scene_is_drawn_from_the_media(self, host: FakeHost) -> None:
        # 壊れると、AI が見る絵と説明を付ける場面（素材の秒）が食い違い、別の場面に説明が付く
        media = self._host(host, [4.0, 6.0])
        result = SHEET(host, {"media_id": str(media.id)})
        assert isinstance(result, ImageResult)
        sheet = host.media_sheets[-1]
        assert sheet[0] == media.id
        # 場面 0〜4・4〜6・6〜10 の真ん中（2 秒・5 秒・8 秒）を素材の頭から数えたフレームで描く
        assert [frame for _index, frame, _label in sheet[1].drawn] == [60, 150, 240]
        assert "0〜4" in result.caption and "6〜10" in result.caption
        assert "remember_scene_descriptions" in result.caption

    def test_the_sheet_fits_the_set_image_edge(self, host: FakeHost) -> None:
        # 壊れると、絵の長い辺を小さくした人（#344 の設定）にも場面の 1 枚だけ 1568 画素で送り、
        # 選んだ費用と画質の釣り合いが場面の説明作りだけ効かない
        media = self._host(host, [0.5 * n for n in range(1, 20)])
        host.image_edge = 1024
        SHEET(host, {"media_id": str(media.id), "count": 12, "cell_width": 640})
        layout = host.media_sheets[-1][1].layout
        assert max(layout.width, layout.height) <= 1024
        host.image_edge = MAX_SHEET_EDGE
        SHEET(host, {"media_id": str(media.id), "count": 12, "cell_width": 640})
        wide = host.media_sheets[-1][1].layout
        assert max(wide.width, wide.height) > 1024

    def test_a_long_scene_is_split(self, host: FakeHost) -> None:
        # 壊れると、切り替わりの無い長い 1 カット（話しっぱなしの動画）が 1 枚の絵の説明になり、
        # 途中の出来事を引けない
        media = self._host(host, [])
        SHEET(host, {"media_id": str(media.id), "max_scene_seconds": 5})
        assert len(host.media_sheets[-1][1].drawn) == 2

    def test_the_rest_is_read_with_offset(self, host: FakeHost) -> None:
        # 壊れると、場面の多い素材で 2 枚目以降を見られず、後半の場面に説明が付かない
        media = self._host(host, [1.0, 2.0, 3.0, 4.0, 5.0])
        first = SHEET(host, {"media_id": str(media.id), "count": 4})
        assert isinstance(first, ImageResult)
        assert "offset=4" in first.caption
        second = SHEET(host, {"media_id": str(media.id), "count": 4, "offset": 4})
        assert isinstance(second, ImageResult)
        assert len(host.media_sheets[-1][1].drawn) == 2

    def test_described_scenes_are_marked(self, host: FakeHost) -> None:
        # 壊れると、説明済みの場面がどれか分からず、同じ場面の絵を何度も送って費用がかさむ
        media = self._host(host, [4.0])
        host.notes[media.path] = (SceneNote(start=0.0, end=4.0, text="海"),)
        result = SHEET(host, {"media_id": str(media.id)})
        assert isinstance(result, ImageResult)
        assert "1) 0〜4 秒（説明あり）" in result.caption

    def test_it_waits_for_the_analysis_on_the_ai_thread(self, host: FakeHost) -> None:
        # 壊れると、解析が終わる前の頼みで、画面のスレッドが止まるか、場面の無い 1 枚が来る
        media = _media(host)
        host.scene_states[media.id] = SceneState(None, running=True)
        result = SHEET(host, {"media_id": str(media.id)})
        assert isinstance(result, Pending)
        host.scene_states[media.id] = SceneState(_signal([5.0]), running=False)
        assert isinstance(finish(result.then()), ImageResult)
        assert host.scene_requests == [media.id]

    def test_a_media_without_pictures_is_refused(self, host: FakeHost) -> None:
        # 壊れると、音だけの素材で黒い升を並べた 1 枚を送り、費用だけかかる
        bare = MediaItem(path=Path("C:/素材/音.wav"), duration=Fraction(5))
        project = host.document.project
        host.document.reset(replace(project, media=(*project.media, bare)))
        with pytest.raises(ToolError, match="映像"):
            SHEET(host, {"media_id": str(bare.id)})

    def test_the_sheet_is_refused_without_consent_to_send_frames(self, host: FakeHost) -> None:
        # 壊れると、絵を送らないと決めた人の素材の絵を、説明を作るために送ってしまう
        media = self._host(host, [4.0])
        bridge = EditorBridge(host)
        bridge.set_send_choices({Outbound.FRAMES: SendChoice.DENY})
        done, box = call_in_thread(bridge, SHEET, {"media_id": str(media.id)})
        pump_until(bridge, done)
        assert isinstance(box.get("error"), ToolError)
        assert host.media_sheets == []


class TestRemember:
    def test_a_failed_write_is_told_without_the_cache_place(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、控えの置き場に書けない（空きが無い・権限が無い）ときに素の OSError が
        # 伏せ字を通らずに AI へ渡り、ユーザー名入りの置き場の場所が外へ出る
        place = r"C:\Users\本人の名前\AppData\Local\Sashimono\cache\scene_notes\ab\x.notes.json"

        def full(_media: MediaItem, _notes: tuple[SceneNote, ...]) -> None:
            raise OSError(28, "No space left on device", place)

        monkeypatch.setattr(host, "remember_scene_notes", full)
        media = _media(host)
        with pytest.raises(ToolError) as caught:
            REMEMBER(
                host,
                {
                    "media_id": str(media.id),
                    "scenes": [{"from_seconds": 0, "to_seconds": 4, "text": "海"}],
                },
            )
        assert "本人の名前" not in str(caught.value)
        assert "空き" in str(caught.value)

    def test_notes_are_kept_outside_the_project(self, host: FakeHost) -> None:
        # 壊れると、説明を覚えるたびに作品が変わり、取り消しの履歴が説明で埋まる
        before = host.document.project
        media = _media(host)
        result = _result(
            REMEMBER,
            host,
            media_id=str(media.id),
            scenes=[{"from_seconds": 0, "to_seconds": 4, "text": "海辺を歩く"}],
        )
        assert host.document.project is before
        assert result == {
            "media_id": str(media.id),
            "remembered": 1,
            "total": 1,
            "note": result["note"],
        }
        assert host.notes[media.path] == (SceneNote(start=0.0, end=4.0, text="海辺を歩く"),)

    def test_the_same_span_is_replaced(self, host: FakeHost) -> None:
        # 壊れると、説明を直すたびに古い説明が残り、引いたときに同じ場面が食い違う 2 行で出る
        media = _media(host)
        for text in ("海", "夕方の海辺"):
            _result(
                REMEMBER,
                host,
                media_id=str(media.id),
                scenes=[{"from_seconds": 0, "to_seconds": 4, "text": text}],
            )
        assert [note.text for note in host.notes[media.path]] == ["夕方の海辺"]

    def test_clear_forgets_the_media(self, host: FakeHost) -> None:
        # 壊れると、間違った説明を AI が消せず、いつまでも間違った場面へ案内する
        media = _media(host)
        host.notes[media.path] = (SceneNote(start=0.0, end=4.0, text="海"),)
        result = _result(REMEMBER, host, media_id=str(media.id), clear=True)
        assert result["total"] == 0
        assert host.notes[media.path] == ()

    @pytest.mark.parametrize(
        "scene",
        [
            {"from_seconds": 4, "to_seconds": 2, "text": "逆"},
            {"from_seconds": 0, "to_seconds": 99, "text": "素材より長い"},
            {"from_seconds": 0, "to_seconds": 2, "text": " "},
            {"from_seconds": 0, "to_seconds": 2, "text": "長" * 301},
            {"from_seconds": "0", "to_seconds": 2, "text": "文字の秒"},
        ],
    )
    def test_a_broken_scene_is_refused(self, host: FakeHost, scene: dict[str, object]) -> None:
        # 壊れると、行けない時刻や空の説明を覚え、引いたときに無い場面へ案内する
        media = _media(host)
        with pytest.raises(ToolError):
            REMEMBER(host, {"media_id": str(media.id), "scenes": [scene]})
        assert host.notes.get(media.path, ()) == ()


class TestWithTheBridge:
    def test_search_is_refused_when_text_is_not_sent(self, host: FakeHost) -> None:
        # 壊れると、字幕と文字を送らないと決めた人の字幕を、検索の結果として外へ送る
        bridge = EditorBridge(host)
        bridge.set_send_choices({Outbound.TEXT: SendChoice.DENY})
        done, box = call_in_thread(bridge, SEARCH, {"queries": ["今日"]})
        pump_until(bridge, done)
        assert isinstance(box.get("error"), ToolError)


def test_the_host_methods_are_on_the_protocol() -> None:
    """壊れると、本物の窓（MainWindow）が新しい口を持たないまま道具が呼んで落ちる"""
    for name in ("start_media_sheet", "scene_notes", "remember_scene_notes"):
        assert hasattr(EditorHost, name), name


def test_a_clip_of_another_media_is_not_a_place(host: FakeHost) -> None:
    """壊れると、別の素材のクリップの位置を、見つけた場面のタイムラインの位置として返す"""
    project = host.document.project
    other = replace(_media(host), path=Path("C:/素材/別.mp4"), transcripts=(), id=MediaId("別"))
    host.document.reset(replace(project, media=(*project.media, other)))
    track = host.document.project.timeline.tracks[0]
    host.apply_commands([AddClip(track.id, make_clip(400, 100, other))], "置く")
    result = _result(SEARCH, host, queries=["作ります"])
    assert [place["frame"] for place in result["hits"][0]["timeline"]] == [210]
