"""AI の道具が外（Anthropic）へ送る物の宣言と、送る前の同意（#342 D4）

道具の結果はそのまま Anthropic へ渡る 絵・字幕の文・素材の名前と場所は、本人が
許すまで送らない ブリッジ（AI のスレッドと画面のスレッドの間）で確かめるので、
実際にスレッドを 2 本使って見る
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimono.ai import bridge as bridge_module
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import CompatReport, EditorHost, ToolError
from sashimono.ai.operations import OPERATIONS, ImageResult, Operation, Pending, find_operation
from sashimono.ai.outbound import (
    ASKED,
    DESTINATION,
    TEXT_MARK,
    Outbound,
    SendAnswer,
    SendChoice,
    SendRequest,
    file_secrets,
    hide_text,
    redact,
    secret_pairs,
    text_secrets,
)
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import TemplateEntry
from sashimono.compat.subtitle_looks import template_choice
from sashimono.core.commands import (
    AddEffect,
    AddMarkers,
    ParamPath,
    RemoveMedia,
    SetParam,
    SetStyleGuide,
    SetTranscript,
)
from sashimono.core.model import ClipId, Effect, Marker, MediaItem, StyleGuide, Transcript
from sashimono.core.timebase import format_timecode
from sashimono.effects import registry
from sashimono.effects.definition import EffectDefinition
from sashimono.effects.spec import TextSpec
from sashimono.engine.cache import SceneState, ScreenState
from sashimono.engine.checks import CheckState
from sashimono.engine.checks.profile import SoundProfiler
from sashimono.engine.checks.report import Problem, assemble
from sashimono.engine.checks.scan import TEXT_OVERFLOW
from sashimono.engine.vision import Box, ScreenKind, ScreenReport, ScreenSample, TextLine
from tests.ai.conftest import FakeHost, make_loaded
from tests.ai.test_bridge import call_in_thread
from tests.compat.test_subtitle_looks import _styled_alias

SECRET_PATH = Path("C:/Users/秘密の人/動画/結婚式の撮影.mp4")


def _operation(name: str) -> Operation:
    found = find_operation(name)
    assert found is not None
    return found


@pytest.fixture
def secret_host(video_media: MediaItem, transcript: Transcript) -> FakeHost:
    """素材の場所に本人の名前が入った作品"""
    return FakeHost(make_loaded(replace(video_media, path=SECRET_PATH), transcript))


def answer_sends(
    bridge: EditorBridge,
    done: threading.Event,
    reply: Callable[[SendRequest], SendAnswer | None],
    timeout: float = 5.0,
) -> list[SendRequest]:
    """画面のスレッドの代わりに、仕事を拾い、出てきた送る前の確かめに ``reply`` で答える

    ``None`` を返すと答えずに窓を閉じたことにする
    """
    seen: list[SendRequest] = []
    deadline = time.monotonic() + timeout
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        request = bridge.take_send_request()
        if request is not None:
            seen.append(request)
            answer = reply(request)
            if answer is None:
                request.give_up("閉じました")
            else:
                request.reply(answer)
        time.sleep(0.005)
    bridge.pump()
    assert done.is_set(), "道具が終わらなかった（送る前の確かめを待ったまま止まっている）"
    return seen


def _never(request: SendRequest) -> SendAnswer:
    pytest.fail(f"尋ねないはずの所で尋ねた: {request.kinds}")


def _run(
    bridge: EditorBridge,
    name: str,
    reply: Callable[[SendRequest], SendAnswer | None] = _never,
    arguments: dict[str, Any] | None = None,
) -> tuple[list[SendRequest], dict[str, object]]:
    done, box = call_in_thread(bridge, _operation(name), arguments or {})
    return answer_sends(bridge, done, reply), box


# --- 宣言 ---


class TestDeclarations:
    def test_a_tool_without_a_declaration_cannot_be_made(self) -> None:
        # 壊れると、絵や字幕を返す道具を足した人が書き忘れても黙って通り、同意を取らずに送る
        with pytest.raises(TypeError, match="sends"):
            Operation(  # type: ignore[call-arg]  # 書き忘れを再現するため、わざと渡さない
                name="forgot",
                description="",
                schema={},
                handler=lambda _host, _arguments: None,
            )

    def test_an_empty_declaration_is_refused(self) -> None:
        # 壊れると、空の宣言で「何も送らない」ふりをして、結果をそのまま送る道具を作れる
        with pytest.raises(ValueError, match="sends"):
            Operation(
                name="empty",
                description="",
                schema={},
                handler=lambda _host, _arguments: None,
                sends=frozenset(),
            )

    def test_every_tool_declares_what_it_sends(self) -> None:
        # 壊れると、宣言の無い道具が同意の外を通る（作る所で落ちるので、ここは念のため）
        for operation in OPERATIONS:
            assert operation.sends, operation.name
            assert operation.sends <= set(Outbound), operation.name

    def test_the_preview_is_declared_as_sending_frames(self) -> None:
        # 壊れると、プレビューの絵を同意なしで送る（この Issue の中心）
        assert Outbound.FRAMES in _operation("preview_frame").sends
        # 何コマも並べた 1 枚も同じ絵（#320）
        assert Outbound.FRAMES in _operation("contact_sheet").sends
        assert Outbound.FRAMES in _operation("preview_range").sends
        # 音量の推移のグラフも絵（#338） 絵は伏せられないので、同意を取ってから送る
        assert Outbound.FRAMES in _operation("audio_graph").sends
        assert Outbound.TEXT in _operation("get_subtitles").sends
        assert Outbound.FILES in _operation("list_media").sends

    def test_what_each_reading_tool_returns_matches_its_declaration(
        self, secret_host: FakeHost, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、宣言に無い物（素材の場所・字幕の文・絵）を返す道具が、尋ねずに送る
        # 読むだけの道具を全部、引数なしで動かして、結果の中身から宣言を確かめる
        texts = _reading_host(secret_host, transcript, tmp_path)
        media = secret_host.document.project.media[0]
        checked = 0
        for operation in OPERATIONS:
            if operation.writes or operation.asks:
                continue
            checked += 1
            result = _read(operation, secret_host, str(media.id))
            assert not undeclared(operation, result, sentences=texts, places=READ_PLACES)
        assert checked >= 10

    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            # 人が画面で見ている物を読む道具（#352）
            ("get_clip", Outbound.TEXT),
            ("get_clip", Outbound.FILES),
            ("get_compat_report", Outbound.APP_SETTINGS),
            ("get_compat_report", Outbound.FILES),
            ("get_compat_report", Outbound.TEXT),
            ("get_app_settings", Outbound.APP_SETTINGS),
            ("get_app_settings", Outbound.FILES),
        ],
    )
    def test_weakening_a_reading_declaration_is_caught(
        self,
        name: str,
        kind: Outbound,
        secret_host: FakeHost,
        transcript: Transcript,
        tmp_path: Path,
    ) -> None:
        # 壊れると（上の見張りが道具の結果に秘密を載せないまま動いていると）、読む道具の宣言から
        # 本文や場所や設定を外しても通り、同意の窓に出ない物が外へ出る（#352 で足した道具も見る）
        texts = _reading_host(secret_host, transcript, tmp_path)
        operation = _operation(name)
        weakened = replace(
            operation, sends=operation.sends - {kind}, hidden=operation.hidden - {kind}
        )
        media = secret_host.document.project.media[0]
        result = _read(weakened, secret_host, str(media.id))
        assert not undeclared(operation, result, sentences=texts, places=READ_PLACES)
        assert undeclared(weakened, result, sentences=texts, places=READ_PLACES)

    @pytest.mark.parametrize("name", ["analyze_audio", "audio_graph"])
    def test_the_sound_tools_declare_the_file_they_were_given(
        self, name: str, secret_host: FakeHost, tmp_path: Path
    ) -> None:
        # 壊れると、書き出した動画の場所を断りの文に入れて返す音の道具（#322 #338）が、
        # 素材の名前と場所を尋ねずに送る 上の見張りは引数なし（タイムライン）でしか呼ばない
        secret_host.saved_at = tmp_path / "作品.sme"
        operation = _operation(name)
        with pytest.raises(ToolError) as refused:
            operation(secret_host, {"path": str(SECRET_PATH), "wait_seconds": 0})
        assert SECRET_PATH.name in str(refused.value)
        secrets = {str(SECRET_PATH), SECRET_PATH.as_posix(), SECRET_PATH.name}
        assert not undeclared(operation, str(refused.value), places=secrets)

    def test_every_reading_tool_that_needs_arguments_has_an_example(self) -> None:
        # 壊れると、引数の要る読む道具（素材を指す物）が、引数なしで断られるだけで済み、
        # 上の見張りが結果の中身を見ないまま通る（detect_scenes の宣言を外しても落ちなかった）
        needing = {
            operation.name
            for operation in OPERATIONS
            if not (operation.writes or operation.asks) and operation.schema.get("required")
        }
        assert needing <= set(READ_EXAMPLES)

    def test_only_frames_text_and_files_are_asked(self) -> None:
        # 壊れると、指示の文や編集の構成まで尋ねて、AI に何も頼めなくなる（または逆に絵を尋ねない）
        assert ASKED == (Outbound.FRAMES, Outbound.TEXT, Outbound.FILES)
        assert {kind.value for kind in Outbound} == {
            "conversation",
            "project",
            "app_settings",
            "files",
            "text",
            "frames",
        }


#: 変える道具に渡す本文 結果やエラーの文にこれが出たら、字幕と文字を送ると書くべき道具
ARG_TEXT = "引数で渡した秘密の本文"
#: 前からあるテキストのクリップの本文
BODY_TEXT = "前からある秘密の本文"
#: 保存したエイリアスの名前 結果やエラーの文に出たら、アプリの設定を送ると書くべき道具
LOOK_NAME = "秘密の見た目"
#: テンプレートの棚の 1 本（#328） 本人の名前の入ったフォルダに置いてある（ファイルは無い）
SHELF_ENTRY = TemplateEntry(
    name=LOOK_NAME,
    path=SECRET_PATH.parent / "テンプレート" / f"{LOOK_NAME}.object",
    folder=f"{SECRET_PATH.parent.name}のテンプレート",
)
SHELF_ID = template_choice(SHELF_ENTRY).key

#: 引数の要る読む道具の引数（素材の ID を受ける） 足したらここにも足す（足さないと落ちる）
READ_EXAMPLES: dict[str, Callable[[str], dict[str, Any]]] = {
    "detect_scenes": lambda media_id: {"media_id": media_id, "wait_seconds": 0},
    # 素材の意味検索（#326） 字幕の文を返す・素材の絵を描く・説明の文を受けて数だけ返す
    "search_media": lambda _media_id: {"queries": ["今日", "作ります"]},
    "media_scene_sheet": lambda media_id: {"media_id": media_id, "wait_seconds": 0},
    "remember_scene_descriptions": lambda media_id: {
        "media_id": media_id,
        "scenes": [{"from_seconds": 0, "to_seconds": 2, "text": "今日は"}],
    },
    "find_highlights": lambda media_id: {"media_id": media_id, "wait_seconds": 0},
    # 画面の中身（#324） 文字は読めた文を返し、顔は断りの文に素材の名前を書く
    "detect_text": lambda media_id: {"media_id": media_id, "wait_seconds": 0},
    "detect_faces": lambda media_id: {"media_id": media_id, "wait_seconds": 0},
    "seek": lambda _media_id: {"frame": 0},
    # 書き出した動画を指す 調べられる場所の外なので、場所を書いて断る（#337）
    "check_output": lambda _media_id: {"path": str(SECRET_PATH), "wait_seconds": 0},
    "select_clips": lambda _media_id: {"clip_ids": []},
    # 前後のキーへ（#349） キーの無い欄は、名前を書いて断る
    "seek_keyframe": lambda _media_id: {"name": "size"},
    # 字幕の文字を OS のクリップボードへ（#349） 無い字幕は ID を書いて断る
    "copy_subtitle_text": lambda media_id: {"media_id": media_id, "segment_id": "無い"},
}


#: 読む道具の結果に出たら、名前と場所を送ると書くべき文字
READ_PLACES = frozenset({str(SECRET_PATH), SECRET_PATH.as_posix(), SECRET_PATH.name})
#: アプリの設定と互換性レポートに入れる目印 結果に出たら、アプリの設定を送ると書くべき道具
APP_MARK = "秘密の設定の目印"


def _reading_host(host: FakeHost, transcript: Transcript, tmp_path: Path) -> set[str]:
    """読むだけの道具が、宣言した物をどれも結果に載せうる作品にする 字幕の文の組を返す"""
    texts = {segment.text for segment in transcript.segments}
    media = host.document.project.media[0]
    # 保存した作品 書き出した動画を調べる道具（check_output）が、外の場所を書いて断る道を通る
    host.saved_at = tmp_path / "作品.sme"
    # 場面の解析が失敗した素材 断りの文に素材の名前が入る
    host.scene_states[media.id] = SceneState(None, running=False, failure="開けない")
    # 画面の文字（#324） 画面に映った字幕の文を読んだ結果 読む道具が本人の文を返す道を通る
    # 顔は調べられなかった素材 断りの文に素材の名前が入る
    read = ScreenSample(1.0, lines=(TextLine(sorted(texts)[0], Box(0.0, 0.0, 80.0, 20.0)),))
    host.screen_states[(media.id, ScreenKind.TEXT)] = ScreenState(
        ScreenReport(ScreenKind.TEXT, "試験", 640, 360, (read,), planned=1, complete=True),
        running=False,
    )
    host.screen_states[(media.id, ScreenKind.FACES)] = ScreenState(
        None, running=False, failure="部品が無い"
    )
    # 保存した見た目と棚（#328） 一覧の道具が本人の持ち物の名前と、棚のフォルダの名前を返す
    host.library.aliases.save(_styled_alias(LOOK_NAME))
    host.shelf = [SHELF_ENTRY]
    # 字幕の文を名前にした目印 目印の一覧と目次（#330）が本人の文を返す道を通る
    named = [Marker(frame, text) for frame, text in zip((0, 150), sorted(texts), strict=False)]
    host.apply_commands([AddMarkers(tuple(named))], "目印")
    # 字幕の文を書いた編集の方針（#343） 方針を読む道具が本人の文を返し、照らす道具は返さない
    sentence = sorted(texts)[0]
    guide = StyleGuide(fonts=("源ノ角ゴシック",), subtitles=sentence, notes=sentence)
    host.apply_commands([SetStyleGuide(guide)], "方針")
    # 書き出し終えた様子（#327） 書き出しの様子を読む道具が、書き出し先の場所を返す道を通る
    host.export_state = {
        "state": "done",
        "progress": 1.0,
        "path": str(SECRET_PATH),
        "message": f"書き出しました: {SECRET_PATH}",
    }
    # 人が選んで設定パネルに出しているテキスト（#352） 本文は字幕の文、画像を読むエフェクトは
    # 本人の名前の入った場所を読む クリップの値を読む道具（get_clip）がどちらも返す道を通る
    _operation("add_text")(host, {"text": sentence})
    clip = next(
        c.id
        for track in host.document.project.timeline.tracks
        for c in track.clips
        if c.source is not None and c.source.kind == "text"
    )
    image = Effect(kind="image_blend", params={"image_file": str(SECRET_PATH)})
    host.apply_commands([AddEffect(clip, image)], "画像")
    host.select_clip(clip)
    # 互換性レポート（#352） 本人のアプリの記録 置き場と失敗の文に本人の名前の入った場所が入る
    recorded = CompatibilityReport()
    recorded.note_missing(APP_MARK)
    recorded.note_failure(SECRET_PATH.name, f"開けない: {SECRET_PATH}")
    # 読めなかった制御文字の行には、テキストの文やスクリプトの見出しの文が入りうる（#391）
    recorded.note_control(f"<#ff0000>{sentence}")
    host.compat = CompatReport.of(recorded, scripts=1, folders=(str(SECRET_PATH.parent),))
    # アプリの設定 置き場のフォルダ（本人の名前の入った場所）と目印
    host.settings = {
        "preferences": {"theme": APP_MARK, "state_folder": str(SECRET_PATH.parent)},
        "shortcuts": {"編集/分割": "S"},
    }
    # 混ぜた音の推移（#322 #338） 音の道具が待たずに結果（数の並びとグラフの絵）を返す道を通る
    # 入れないと、解析を頼んだだけの断りの文しか見ず、グラフの絵の宣言を外しても落ちない
    profiler = SoundProfiler(48000, 2)
    profiler.add(np.full((48000 * 2, 2), 0.1))
    host.heard = profiler.snapshot()
    return texts


def _read(operation: Operation, host: FakeHost, media_id: str) -> object:
    """読む道具を例の引数で動かした結果 断られたら断りの文（どちらも外へ送る）"""
    example = READ_EXAMPLES.get(operation.name)
    arguments = example(media_id) if example is not None else {}
    try:
        return operation(host, arguments)
    except ToolError as exc:
        # エラーの文も外へ送る 引数が要る道具は、断った文の中身を見る
        return str(exc)


def _ids(host: FakeHost) -> dict[str, str]:
    project = host.document.project
    media = project.media[0]
    transcript = media.transcript
    assert transcript is not None
    clips = [clip for track in project.timeline.tracks for clip in track.clips]
    text = next(c for c in clips if c.source is not None and c.source.kind == "text")
    video = next(c for c in clips if c.media_id is not None)
    return {
        "media": str(media.id),
        "segment": str(transcript.segments[0].id),
        "track": str(project.timeline.tracks[0].id),
        "video": str(video.id),
        "text": str(text.id),
    }


#: 変える道具ごとの、ありそうな引数 道具を足したらここにも足す（足さないと落ちる）
#: 引数が合わずに断られる物もあるが、それでよい 断った文も外へ送るので中身を見る
WRITE_EXAMPLES: dict[str, Callable[[dict[str, str]], dict[str, Any]]] = {
    "import_media": lambda _ids: {"paths": [str(SECRET_PATH)]},
    "place_media": lambda ids: {"media_id": ids["media"]},
    "add_track": lambda _ids: {"kind": "video", "name": "V9"},
    "set_track_state": lambda ids: {"track_id": ids["track"], "muted": True},
    "set_resolution": lambda _ids: {"width": 1280, "height": 720},
    "add_text": lambda _ids: {"text": ARG_TEXT},
    "add_shape": lambda _ids: {},
    "add_transition": lambda _ids: {},
    "split_clip": lambda ids: {"clip_id": ids["video"], "frame": 30},
    "trim_clip": lambda ids: {"clip_id": ids["video"], "tail_delta": -10},
    "move_clip": lambda ids: {"clip_id": ids["video"], "timeline_start": 10},
    "delete_clip": lambda ids: {"clip_id": ids["video"]},
    "add_scene": lambda _ids: {"name": "場面"},
    "place_scene": lambda _ids: {"scene_id": "無い"},
    "group_clips": lambda ids: {"clip_ids": [ids["video"], ids["text"]]},
    "ungroup_clips": lambda ids: {"clip_ids": [ids["video"]]},
    "move_clips": lambda ids: {"clip_ids": [ids["video"]], "delta": 5},
    "delete_clips": lambda ids: {"clip_ids": [ids["text"]]},
    "duplicate_clips": lambda ids: {"clip_ids": [ids["text"]]},
    "set_track_height": lambda _ids: {"height": 80},
    "set_clip_property": lambda ids: {"clip_id": ids["text"], "name": "enabled", "value": True},
    "add_effect": lambda ids: {"clip_id": ids["text"], "kind": "blur"},
    "add_stroke": lambda ids: {"clip_id": ids["text"]},
    "remove_stroke": lambda ids: {"clip_id": ids["text"], "stroke_id": "無い"},
    "move_stroke": lambda ids: {"clip_id": ids["text"], "stroke_id": "無い", "index": 0},
    "set_stroke_enabled": lambda ids: {"clip_id": ids["text"], "stroke_id": "無い", "enabled": 1},
    "set_param": lambda ids: {"clip_id": ids["text"], "name": "text", "value": ARG_TEXT},
    "add_keyframe": lambda ids: {"clip_id": ids["text"], "name": "text", "value": ARG_TEXT},
    # 強調の動き（#331） テキストのクリップを 0〜20 フレームだけ寄せる
    "emphasize": lambda ids: {
        "clip_id": ids["text"],
        "kind": "zoom",
        "start_frame": 0,
        "end_frame": 20,
    },
    # 縦動画への作り替え（#334） 追わずに真ん中を切り出す（解析を待たない）
    "reframe": lambda _ids: {"width": 1080, "height": 1920, "follow": "center"},
    "set_subtitle_text": lambda ids: {
        "media_id": ids["media"],
        "segment_id": ids["segment"],
        "text": ARG_TEXT,
    },
    "replace_subtitle_text": lambda _ids: {"pairs": [{"from": "今日は", "to": ARG_TEXT}]},
    "clean_subtitles": lambda ids: {"media_id": ids["media"]},
    # 無い見た目を頼むと、保存してある見た目の名前を並べて断る（#308）
    "place_subtitles": lambda ids: {"media_id": ids["media"], "look_name": "無い見た目"},
    "jet_cut": lambda ids: {"media_id": ids["media"], "smooth": True},
    # 再生速度とつなぎ目（#332） 動画のクリップを半分の速さにし、10 フレーム目をつなぎ目として扱う
    "set_speed": lambda ids: {"clip_ids": [ids["video"]], "speed_percent": 50},
    "smooth_cuts": lambda _ids: {"frames": [10]},
    # 音量とミックス（#329） 絵のクリップと動画の素材を渡すと断る 断る文に素材の名前が入りうる
    "set_volume": lambda ids: {"clip_ids": [ids["video"]], "volume_percent": 50},
    "duck_audio": lambda ids: {"clip_ids": [ids["video"]], "voice_clip_ids": [ids["video"]]},
    "normalize_loudness": lambda _ids: {"wait_seconds": 0},
    "place_sound": lambda ids: {"media_id": ids["media"]},
    "transcribe": lambda ids: {"media_id": ids["media"], "replace": True},
    # 話者の分離（#323） 始めた知らせに素材の名前が入る 付け替える名前は本文と同じ扱い
    "separate_speakers": lambda ids: {"media_id": ids["media"], "speakers": 2},
    "rename_speaker": lambda ids: {"media_id": ids["media"], "from": "話者 1", "to": ARG_TEXT},
    "set_subtitle_speaker": lambda ids: {
        "media_id": ids["media"],
        "segment_ids": [ids["segment"]],
        "speaker": ARG_TEXT,
    },
    "undo": lambda _ids: {},
    # 取り消した後でなければ、やり直す物が無いと断る 断る文にも名前は入らない（#333）
    "redo": lambda _ids: {},
    # 目印（#330） 前からある目印（0 フレーム 名前は本文）を動かす・消す
    "add_markers": lambda _ids: {"markers": [{"frame": 45, "label": ARG_TEXT}]},
    "update_marker": lambda _ids: {"frame": 0, "label": ARG_TEXT},
    "delete_markers": lambda _ids: {"frames": [0]},
    # 保存と書き出し（#327） 試験の作品は保存してあり（作品のフォルダへ書ける）、結果に
    # 書いた場所が載る 書き出しの止め方は、走っている書き出しを止める試験で見る
    "save_project": lambda _ids: {},
    "save_project_as": lambda _ids: {"path": "第二稿.sme"},
    "export_video": lambda _ids: {"path": "完成.mp4"},
    "cancel_export": lambda _ids: {},
    "export_subtitles": lambda _ids: {"path": "字幕.srt"},
    "save_snapshot": lambda _ids: {"path": "表紙.png"},
    # 保存した見た目と棚（#328） 無い名前は、保存してある見た目の名前を並べて断る
    # 棚の 1 本は読めない場所（本人の名前の入ったフォルダ）を指し、場所を書いて断る道を通る
    "apply_preset": lambda ids: {"name": "無い見た目", "clip_ids": [ids["text"]]},
    "place_alias": lambda _ids: {"name": "無い見た目"},
    "place_template": lambda _ids: {"template_id": SHELF_ID},
    "apply_template": lambda ids: {"template_id": SHELF_ID, "clip_id": ids["text"]},
    "import_objects": lambda _ids: {"path": str(SECRET_PATH.with_suffix(".exo"))},
    # 画面の操作と同じ道具（#349） 入れ物（トラック・素材）は中身があるので断る 断る文に
    # 素材の名前を書かないかを見る 貼り付けはコピーした物が無いので断る
    "cut_clips": lambda ids: {"clip_ids": [ids["text"]]},
    "paste_clips": lambda _ids: {"insert": True},
    "remove_track": lambda ids: {"track_id": ids["track"]},
    "rename_track": lambda ids: {"track_id": ids["track"], "name": "V9"},
    "move_track": lambda ids: {"track_id": ids["track"], "index": 0},
    "rename_scene": lambda _ids: {"scene_id": "無い", "name": "場面"},
    "remove_scene": lambda _ids: {"scene_id": "無い"},
    "remove_media": lambda ids: {"media_id": ids["media"]},
    "set_work_area": lambda _ids: {"start_frame": 0, "end_frame": 30},
    "add_filter": lambda _ids: {},
    "add_group_control": lambda _ids: {},
    "remove_effect": lambda ids: {"clip_id": ids["text"], "effect_id": "無い"},
    "move_effect": lambda ids: {"clip_id": ids["text"], "effect_id": "無い", "index": 0},
    "set_effect_enabled": lambda ids: {
        "clip_id": ids["text"],
        "effect_ids": ["無い"],
        "enabled": False,
    },
    "remove_keyframe": lambda ids: {"clip_id": ids["text"], "name": "size", "frame": 0},
    "clear_keyframes": lambda ids: {"clip_id": ids["text"], "name": "size"},
    "split_subtitle": lambda ids: {
        "media_id": ids["media"],
        "segment_id": ids["segment"],
        "at_seconds": 0.5,
    },
    "merge_subtitles": lambda ids: {"media_id": ids["media"], "segment_id": ids["segment"]},
    "delete_subtitle": lambda ids: {"media_id": ids["media"], "segment_id": ids["segment"]},
    # 開く・新規（#349） 試験の作品は変えてあるので、どちらも保存していない変更で断る
    "open_project": lambda _ids: {"path": "第二稿.sme"},
    "new_project": lambda _ids: {"width": 1280, "height": 720},
    # 作品の設定・置き方の方式・置く・揃える（#349） 無いカスタムオブジェクトは置き場の名前を
    # 並べて断る 揃える枠は偽のホストに無いので断る
    "switch_layer_mode": lambda _ids: {"convert": True},
    "set_project_settings": lambda _ids: {"blending": "linear"},
    "add_custom_object": lambda _ids: {"name": "無い"},
    "align_clip": lambda ids: {"clip_id": ids["text"], "anchor": "center"},
    "use_standard_layout": lambda ids: {"clip_ids": [ids["text"]]},
}


def _with_text_clip(video_media: MediaItem, transcript: Transcript) -> FakeHost:
    """素材の場所に本人の名前が入り、本文のテキストを 1 本と本文を名前にした目印を 1 つ置き、
    見た目を 1 つ保存した作品"""
    created = FakeHost(make_loaded(replace(video_media, path=SECRET_PATH), transcript))
    _operation("add_text")(created, {"text": BODY_TEXT})
    created.apply_commands([AddMarkers((Marker(0, BODY_TEXT),))], "目印")
    created.library.aliases.save(_styled_alias(LOOK_NAME))
    created.shelf = [SHELF_ENTRY]
    return created


class TestWritingToolsDeclareWhatTheyReturn:
    def test_every_writing_tool_has_an_example(self) -> None:
        # 壊れると、変える道具を足した人が例を書かずに済み、下の見張りの外を通る
        writing = {operation.name for operation in OPERATIONS if operation.writes}
        assert set(WRITE_EXAMPLES) == writing

    @pytest.mark.parametrize("name", sorted(WRITE_EXAMPLES))
    def test_what_each_writing_tool_returns_matches_its_declaration(
        self, name: str, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、変える道具が本文や素材の場所を結果に載せるのに宣言せず、字幕と文字や
        # 名前と場所を断った人にも、尋ねず伏せずに送る（add_text が本文を返していた）
        host = _with_text_clip(video_media, transcript)
        # 保存した作品 保存と書き出しの道具（#327）が作品のフォルダへ書き、書いた場所を返す道を通る
        host.saved_at = tmp_path / "作品.sme"
        operation = _operation(name)
        arguments = WRITE_EXAMPLES[name](_ids(host))
        try:
            result: object = operation(host, arguments)
        except (ToolError, ValueError, KeyError) as exc:
            result = str(exc)
        sentences = {ARG_TEXT, BODY_TEXT[:12]} | {s.text for s in transcript.segments}
        places = {str(SECRET_PATH), SECRET_PATH.as_posix(), SECRET_PATH.name, SECRET_PATH.stem}
        assert not undeclared(operation, result, sentences=sentences, places=places)

    def test_stopping_an_export_returns_what_it_declares(self, secret_host: FakeHost) -> None:
        # 壊れると、書き出しを止めた答え（止めた後の様子）に書き出し先の場所が載るのに宣言せず、
        # 名前と場所を断った人にも伏せずに送る 上の例では走っている書き出しが無く断るだけなので、
        # ここで走っている書き出しを止める
        secret_host.export_state = {"state": "rendering", "progress": 0.5, "path": str(SECRET_PATH)}
        operation = _operation("cancel_export")
        result = operation(secret_host, {})
        assert not undeclared(operation, result, places={str(SECRET_PATH)})
        assert isinstance(result, dict)
        status = result["status"]
        assert isinstance(status, dict) and status["path"] == str(SECRET_PATH)


def undeclared(
    operation: Operation,
    result: object,
    *,
    sentences: Iterable[str] = (),
    places: Iterable[str] = (),
) -> list[str]:
    """道具の結果（か断りの文）に載っているのに、送る物に書いていない物

    場所は、渡した名前のほかに、ファイルの場所の形（ドライブ名・POSIX など 伏せ字の処理が
    場所と見なす物）でも見る 渡した名前だけで見ると、試験の作品の外へ書いた場所（一時
    フォルダなど）を返す道具を見逃す（試験用の write_note がそうだった CodeRabbit の指摘 #366）
    試験の中だけで作る道具も、この関数で宣言を確かめる
    """
    if isinstance(result, ImageResult):
        return [] if Outbound.FRAMES in operation.sends else [f"{operation.name}: 絵"]
    text = json.dumps(result, ensure_ascii=False, default=str)
    missing: list[str] = []
    if any(sentence in text for sentence in sentences) and Outbound.TEXT not in operation.sends:
        missing.append(f"{operation.name}: 字幕と文字 {text[:200]}")
    has_place = any(place in text for place in places) or redact(result, []) != result
    if has_place and Outbound.FILES not in operation.sends:
        missing.append(f"{operation.name}: 名前と場所 {text[:200]}")
    if (LOOK_NAME in text or APP_MARK in text) and Outbound.APP_SETTINGS not in operation.sends:
        # 保存したプリセットとエイリアスの名前は本人の持ち物 アプリの設定として送ると書く
        # 設定の値と互換性レポート（#352）も本人のアプリの物
        missing.append(f"{operation.name}: アプリの設定 {text[:200]}")
    return missing


# --- 送る前に尋ねる ---


class TestAskingBeforeSending:
    def test_the_first_preview_asks_before_drawing(self, host: FakeHost) -> None:
        # 壊れると、本人が知らないうちに映像の絵が外へ出る
        bridge = EditorBridge(host)
        drawn_when_asked: list[int] = []

        def deny(request: SendRequest) -> SendAnswer:
            drawn_when_asked.append(len(host.rendered))
            return SendAnswer.DENY

        seen, _box = _run(bridge, "preview_frame", deny)
        assert [request.kinds for request in seen] == [(Outbound.FRAMES,)]
        assert drawn_when_asked == [0]
        assert DESTINATION in seen[0].text()
        assert "映像の絵" in seen[0].text()

    def test_denying_sends_nothing(self, host: FakeHost) -> None:
        # 壊れると、「送らない」を押しても絵が描かれて外へ出る
        bridge = EditorBridge(host)
        _seen, box = _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        assert isinstance(box.get("error"), ToolError)
        assert "映像の絵" in str(box["error"])
        assert host.rendered == []

    def test_a_denial_is_kept_until_the_app_closes(self, host: FakeHost) -> None:
        # 壊れると、断ったのに AI が呼び直すたびに窓が出て、作業が止まり続ける
        bridge = EditorBridge(host)
        _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        _seen, box = _run(bridge, "preview_frame")
        assert isinstance(box.get("error"), ToolError)
        assert host.rendered == []

    def test_allowing_until_close_does_not_ask_again(self, host: FakeHost) -> None:
        # 壊れると、AI が絵を見るたびに窓が出て、1 本の動画を見るのに何十回も押す
        bridge = EditorBridge(host)
        seen, box = _run(bridge, "preview_frame", lambda _request: SendAnswer.UNTIL_CLOSE)
        assert len(seen) == 1
        assert isinstance(box.get("result"), ImageResult)
        _seen, again = _run(bridge, "preview_frame")
        assert isinstance(again.get("result"), ImageResult)
        assert len(host.rendered) == 2

    def test_closing_the_window_is_not_remembered(self, host: FakeHost) -> None:
        # 壊れると、答えずに閉じただけで「送らない」が残り、次に尋ねてもらえない
        bridge = EditorBridge(host)
        _seen, box = _run(bridge, "preview_frame", lambda _request: None)
        assert isinstance(box.get("error"), ToolError)
        seen, _box = _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        assert len(seen) == 1

    def test_auto_approve_does_not_skip_it(self, host: FakeHost) -> None:
        # 壊れると、「変更を自動で承認」を入れた人の絵が、尋ねずに外へ出る（外へ送るのは
        # 取り消せない 2026-10-14 利用者の決定）
        bridge = EditorBridge(host, auto_approve=True)
        seen, _box = _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        assert len(seen) == 1

    def test_a_setting_of_allow_sends_without_asking(self, host: FakeHost) -> None:
        # 壊れると、設定で「尋ねずに送る」にした人にも毎回窓が出る
        bridge = EditorBridge(host)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
        _seen, box = _run(bridge, "preview_frame")
        assert isinstance(box.get("result"), ImageResult)

    def test_a_setting_of_deny_refuses_without_asking(self, host: FakeHost) -> None:
        # 壊れると、設定で「送らない」にしたのに窓が出る（押し間違えると送る）
        bridge = EditorBridge(host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.TEXT: SendChoice.DENY})
        _seen, box = _run(bridge, "get_subtitles")
        assert isinstance(box.get("error"), ToolError)
        assert "字幕と文字" in str(box["error"])

    def test_changing_the_setting_forgets_the_answer_in_the_window(self, host: FakeHost) -> None:
        # 壊れると、設定で「送る前に尋ねる」に戻しても、前に窓で答えたままになり尋ねない
        bridge = EditorBridge(host)
        _run(bridge, "preview_frame", lambda _request: SendAnswer.UNTIL_CLOSE)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FRAMES: SendChoice.ALLOW})
        bridge.set_send_choices({**bridge.send_choices, Outbound.FRAMES: SendChoice.ASK})
        seen, _box = _run(bridge, "preview_frame", lambda _request: SendAnswer.DENY)
        assert len(seen) == 1

    def test_tools_that_send_only_the_structure_never_ask(self, host: FakeHost) -> None:
        # 壊れると、分割や選択のたびに窓が出て、AI に編集を頼めなくなる
        bridge = EditorBridge(host, auto_approve=True)
        clip = host.project.timeline.tracks[0].clips[0]
        _seen, box = _run(bridge, "split_clip", arguments={"clip_id": str(clip.id), "frame": 30})
        assert box.get("error") is None
        _seen, box = _run(bridge, "get_selection")
        assert box.get("error") is None

    def test_an_interrupt_while_asking_folds_it(self, host: FakeHost) -> None:
        # 壊れると、中断を押しても AI のスレッドが送る前の確かめの返事を 10 分待ち続ける
        bridge = EditorBridge(host)

        def interrupt(_request: SendRequest) -> SendAnswer:
            bridge.cancel()
            return SendAnswer.ALWAYS

        _seen, box = _run(bridge, "preview_frame", interrupt)
        assert isinstance(box.get("error"), ToolError)
        assert host.rendered == []

    def test_no_answer_in_time_sends_nothing(
        self, host: FakeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 壊れると、席を外している間に時間が切れ、黙って送る
        monkeypatch.setattr(bridge_module, "SEND_TIMEOUT", 0.05)
        bridge = EditorBridge(host)
        done, box = call_in_thread(bridge, _operation("preview_frame"), {})
        deadline = time.monotonic() + 5
        while not done.is_set() and time.monotonic() < deadline:
            bridge.pump()
            time.sleep(0.005)
        assert isinstance(box.get("error"), ToolError)
        assert host.rendered == []


# --- 素材の名前と場所 ---


class TestFileNames:
    def test_not_allowed_names_are_hidden_but_the_tool_still_works(
        self, secret_host: FakeHost
    ) -> None:
        # 壊れると、名前を送らないと決めた人の素材の場所（ユーザー名入り）が外へ出るか、
        # 一覧の道具ごと断られて AI が素材を指せなくなる
        bridge = EditorBridge(secret_host)
        seen, box = _run(bridge, "list_media", lambda _request: SendAnswer.DENY)
        assert [request.kinds for request in seen] == [(Outbound.FILES,)]
        text = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密の人" not in text
        assert "結婚式" not in text
        media = secret_host.project.media[0]
        assert str(media.id) in text

    def test_errors_are_hidden_too(self, secret_host: FakeHost) -> None:
        # 壊れると、断った人の素材の名前が、エラーの文に混ざって外へ出る
        bridge = EditorBridge(secret_host, auto_approve=True)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.DENY})
        media = secret_host.project.media[0]
        _seen, box = _run(bridge, "jet_cut", arguments={"media_id": str(media.id)})
        error = box.get("error")
        assert isinstance(error, ToolError)
        assert "波形" in str(error)
        assert "結婚式" not in str(error)

    def test_a_tool_that_forgot_to_declare_names_still_hides_them(
        self, secret_host: FakeHost
    ) -> None:
        # 壊れると、宣言を書き漏らした道具が、まだ許していない素材の場所を送る
        leaky = Operation(
            name="leaky",
            description="",
            schema={},
            handler=lambda host, _arguments: {"path": str(host.document.project.media[0].path)},
            sends=frozenset({Outbound.PROJECT}),
        )
        bridge = EditorBridge(secret_host)
        done, box = call_in_thread(bridge, leaky, {})
        answer_sends(bridge, done, _never)
        assert "秘密の人" not in json.dumps(box["result"], ensure_ascii=False)

    def test_allowed_names_pass_through(self, secret_host: FakeHost) -> None:
        # 壊れると、名前を送ってよいと答えた人にも伏せ字が返り、AI が素材を名前で探せない
        bridge = EditorBridge(secret_host)
        _seen, box = _run(bridge, "list_media", lambda _request: SendAnswer.UNTIL_CLOSE)
        assert "結婚式の撮影.mp4" in json.dumps(box["result"], ensure_ascii=False)

    def test_an_image_from_an_undeclared_tool_is_refused(self, host: FakeHost) -> None:
        # 壊れると、絵を返すと宣言していない道具（#320 の見渡しの絵など）が、尋ねずに絵を送る
        def draw(target: EditorHost, _arguments: dict[str, Any]) -> object:
            return ImageResult(png=target.render_png(0, width=160))

        sneaky = Operation(
            name="sneaky",
            description="",
            schema={},
            handler=draw,
            sends=frozenset({Outbound.PROJECT}),
        )
        bridge = EditorBridge(host)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
        done, box = call_in_thread(bridge, sneaky, {})
        answer_sends(bridge, done, _never)
        assert isinstance(box.get("error"), ToolError)


# --- 取り消しと履歴（2026-10-15 利用者の決定 字幕と文字を送らなくても取り消せる） ---

SECRET_TEXT = "秘密の挨拶をここに書きます"


def _add_secret_text(bridge: EditorBridge) -> None:
    _seen, box = _run(bridge, "add_text", arguments={"text": SECRET_TEXT})
    assert box.get("error") is None, box.get("error")


def _type_secret_text(host: FakeHost) -> None:
    """人が画面でテキストを足した、の代わり（ブリッジを通さない 外へは何も送らない）"""
    _operation("add_text")(host, {"text": SECRET_TEXT})


@pytest.fixture
def text_refused(host: FakeHost) -> EditorBridge:
    """字幕と文字を「送らない」にした人 名前と場所は許してある 変更は自動で承認

    名前と場所を許しておくのは、伏せ字が字幕と文字の扱いから来たことを見分けるため
    """
    bridge = EditorBridge(host, auto_approve=True)
    bridge.set_send_choices(
        {Outbound.FRAMES: SendChoice.ASK, Outbound.TEXT: SendChoice.DENY}
        | {Outbound.FILES: SendChoice.ALLOW}
    )
    return bridge


def _text_clips(host: FakeHost) -> int:
    return sum(
        1
        for track in host.document.project.timeline.tracks
        for clip in track.clips
        if clip.source is not None and clip.source.kind == "text"
    )


class TestWritingTextWithoutSendingText:
    """本文を引数で渡して結果にも返す道具（CodeRabbit の指摘 #366）"""

    def test_adding_text_is_refused_and_nothing_is_added(
        self, host: FakeHost, text_refused: EditorBridge
    ) -> None:
        # 壊れると、字幕と文字を断った人の所で、本文が要る道具が尋ねずに動き、本文が結果で返る
        _seen, box = _run(text_refused, "add_text", arguments={"text": SECRET_TEXT})
        assert isinstance(box.get("error"), ToolError)
        assert _text_clips(host) == 0

    @pytest.mark.parametrize("name", ["set_subtitle_text", "replace_subtitle_text"])
    def test_subtitle_edits_are_refused(self, name: str, text_refused: EditorBridge) -> None:
        # 壊れると、字幕を断った人の字幕を、AI が尋ねずに書き換え、直した本文が外へ出る
        _seen, box = _run(text_refused, name, arguments={"pairs": [], "text": SECRET_TEXT})
        assert isinstance(box.get("error"), ToolError)
        assert "字幕と文字" in str(box["error"])

    def test_a_text_value_is_set_but_hidden(self, host: FakeHost) -> None:
        # 壊れると、値を変える道具ごと断られて色や位置も変えられないか、本文が外へ出る
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
        _add_secret_text(bridge)
        bridge.set_send_choices(
            {Outbound.FRAMES: SendChoice.ASK, Outbound.TEXT: SendChoice.DENY}
            | {Outbound.FILES: SendChoice.ALLOW}
        )
        clip = next(
            c
            for track in host.document.project.timeline.tracks
            for c in track.clips
            if c.source is not None and c.source.kind == "text"
        )
        changed = "書き換えた秘密の本文"
        arguments = {"clip_id": str(clip.id), "name": "text", "value": changed}
        _seen, box = _run(bridge, "set_param", arguments=arguments)
        assert box.get("error") is None, box.get("error")
        assert "秘密" not in json.dumps(box["result"], ensure_ascii=False)
        bodies = [
            c.source.params["text"]
            for track in host.document.project.timeline.tracks
            for c in track.clips
            if c.id == clip.id and c.source is not None
        ]
        assert bodies == [changed]


class TestUndoWithoutSendingText:
    def test_undo_still_works_and_the_text_is_hidden(
        self, host: FakeHost, text_refused: EditorBridge
    ) -> None:
        # 壊れると、字幕と文字を送らない人は AI に取り消しを頼めないか、取り消した操作の名前に
        # 混ざったテキストの文が外へ出る
        _type_secret_text(host)
        assert _text_clips(host) == 1
        _seen, box = _run(text_refused, "undo")
        assert box.get("error") is None, box.get("error")
        assert _text_clips(host) == 0
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密" not in sent
        assert SECRET_TEXT[:12] not in sent
        # どの操作を取り消したかは読める
        assert "テキストを追加" in sent

    def test_the_history_is_read_with_the_text_hidden(
        self, host: FakeHost, text_refused: EditorBridge
    ) -> None:
        # 壊れると、履歴を読むだけで、送らないと決めた文の頭が外へ出る（または履歴を読めない）
        _type_secret_text(host)
        _seen, box = _run(text_refused, "get_history")
        assert box.get("error") is None, box.get("error")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密" not in sent
        assert "テキストを追加" in sent
        assert box["result"]["can_undo"] is True  # type: ignore[index]  # 結果は辞書で返る

    def test_undo_does_not_ask_about_text(self, host: FakeHost) -> None:
        # 壊れると、まだ答えていない人が取り消しを頼むたびに、字幕と文字の窓が出る
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.ALLOW})
        _type_secret_text(host)
        _seen, box = _run(bridge, "undo")
        assert box.get("error") is None, box.get("error")
        assert "秘密" not in json.dumps(box["result"], ensure_ascii=False)

    def test_allowed_text_passes_through(self, host: FakeHost) -> None:
        # 壊れると、文字を送ってよいと答えた人にも伏せ字が返り、AI がどの文を消したか読めない
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices(dict.fromkeys(ASKED, SendChoice.ALLOW))
        _add_secret_text(bridge)
        _seen, box = _run(bridge, "undo")
        assert SECRET_TEXT[:12] in json.dumps(box["result"], ensure_ascii=False)

    def test_subtitles_are_still_refused(self, text_refused: EditorBridge) -> None:
        # 壊れると、伏せて送る扱いが字幕の一覧にまで広がり、送らないと決めた字幕の道具が動く
        _seen, box = _run(text_refused, "get_subtitles")
        assert isinstance(box.get("error"), ToolError)

    def test_only_text_can_be_hidden(self) -> None:
        # 壊れると、絵を「伏せて送る」と書いた道具が、尋ねずに絵を送る
        with pytest.raises(ValueError, match="伏せて"):
            Operation(
                name="hidden_frames",
                description="",
                schema={},
                handler=lambda _host, _arguments: None,
                sends=frozenset({Outbound.FRAMES}),
                hidden=frozenset({Outbound.FRAMES}),
            )

    def test_counts_stay_and_quotes_go(self) -> None:
        # 壊れると、数だけの名前（無音カットの数）まで伏せて読めなくなるか、引用した文が外へ出る
        hidden = hide_text(
            ["無音カット: 3 か所", "見た目: 選んでいるテキスト「秘密」を写します", "配置: 本編"],
            [],
        )
        assert hidden[0] == "無音カット: 3 か所"
        assert "秘密" not in hidden[1]
        assert hidden[2] == f"配置: {TEXT_MARK}"


class TestWaitingTools:
    """待ってから返す道具（:class:`Pending` #321 の detect_scenes など）"""

    @staticmethod
    def _waiting(steps: list[str], answer: Callable[[EditorHost], object]) -> Operation:
        def wait(seconds: float) -> bool:
            del seconds
            steps.append("待った")
            return True

        def start(host: EditorHost, _arguments: dict[str, Any]) -> object:
            steps.append("動いた")
            return Pending(wait=wait, seconds=1.0, then=lambda: answer(host))

        return Operation(
            name="waiting",
            description="",
            schema={},
            handler=start,
            sends=frozenset({Outbound.PROJECT, Outbound.FILES}),
        )

    def test_consent_is_asked_before_running_and_waiting(self, secret_host: FakeHost) -> None:
        # 壊れると、待つ道具が同意の前に動き出して待ち、断っても解析や読み込みが走る
        steps: list[str] = []
        operation = self._waiting(
            steps, lambda host: {"media": host.document.project.media[0].name}
        )
        bridge = EditorBridge(secret_host)

        def deny(_request: SendRequest) -> SendAnswer:
            steps.append("尋ねた")
            return SendAnswer.DENY

        done, box = call_in_thread(bridge, operation, {})
        answer_sends(bridge, done, deny)
        assert steps[0] == "尋ねた"
        assert steps == ["尋ねた", "動いた", "待った"]
        # 名前と場所は断っても動かし、待った後に返る名前も伏せる
        assert "結婚式" not in json.dumps(box["result"], ensure_ascii=False)

    def test_what_comes_back_after_waiting_is_hidden_too(self, secret_host: FakeHost) -> None:
        # 壊れると、待つ前の戻り値だけを伏せ、待った後に返る素材の場所がそのまま外へ出る
        steps: list[str] = []
        operation = self._waiting(steps, lambda host: str(host.document.project.media[0].path))
        bridge = EditorBridge(secret_host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.DENY})
        done, box = call_in_thread(bridge, operation, {})
        answer_sends(bridge, done, _never)
        assert "秘密の人" not in str(box["result"])
        assert steps == ["動いた", "待った"]

    def test_detect_scenes_declares_names_and_places(self) -> None:
        # 壊れると、素材の名前の入った断りの文（映像がありません・調べられませんでした）を、
        # 名前と場所を断った人の所でも尋ねずに送る
        assert _operation("detect_scenes").sends == {Outbound.PROJECT, Outbound.FILES}

    def test_find_highlights_sends_names_but_not_subtitles(
        self, secret_host: FakeHost, transcript: Transcript
    ) -> None:
        """盛り上がりの道具（#325）は字幕を読んで笑いの印を探すが、文は返さない

        壊れると、名前と場所を断った人の所で素材の名前入りの断りの文を尋ねずに送るか、
        字幕を断った人の所で字幕の文を結果に載せて送る 宣言を弱めると見張りが落ちることも見る
        """
        media = secret_host.document.project.media[0]
        secret_host.scene_states[media.id] = SceneState(None, running=False, failure="開けない")
        operation = _operation("find_highlights")
        result = operation(secret_host, {"media_id": str(media.id), "wait_seconds": 0})
        secrets = {str(SECRET_PATH), SECRET_PATH.as_posix(), SECRET_PATH.name}
        texts = {segment.text for segment in transcript.segments}
        assert operation.sends == {Outbound.PROJECT, Outbound.FILES}
        assert not undeclared(operation, result, sentences=texts, places=secrets)
        weaker = replace(operation, sends=frozenset({Outbound.PROJECT}))
        assert undeclared(weaker, result, sentences=texts, places=secrets)


class TestPlacesOutsideTheTable:
    """表（今の作品の素材と場所）に無い名前と場所（Codex の指摘 #366）"""

    def test_a_removed_material_name_in_the_history_is_hidden(self, secret_host: FakeHost) -> None:
        # 壊れると、置いてから消した素材の名前（「配置: 結婚式の撮影.mp4」）が履歴に残り、
        # 名前と場所を断った人の所でも外へ出る 今の作品に無い名前は表から伏せられない
        media = secret_host.document.project.media[0]
        _operation("place_media")(secret_host, {"media_id": str(media.id)})
        clips = [
            str(clip.id)
            for track in secret_host.document.project.timeline.tracks
            for clip in track.clips
        ]
        _operation("delete_clips")(secret_host, {"clip_ids": clips})
        secret_host.apply_commands([RemoveMedia(media.id)], "素材を削除")
        assert not secret_host.document.project.media
        bridge = EditorBridge(secret_host)
        bridge.set_send_choices(
            {Outbound.FRAMES: SendChoice.ASK, Outbound.TEXT: SendChoice.ALLOW}
            | {Outbound.FILES: SendChoice.DENY}
        )
        _seen, box = _run(bridge, "get_history")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "結婚式" not in sent
        assert "配置" in sent

    def test_a_folder_at_the_root_is_hidden(self) -> None:
        # 壊れると、根の直下の 1 段だけの場所（/backup）が、区切りが 1 つなので伏せずに送られる
        assert "backup" not in redact("/backup", [])
        assert "backup" not in redact("置き場 \\backup", [])
        assert redact("1/2 画質", []) == "1/2 画質"

    def test_folders_in_the_settings_are_hidden_whole(self, host: FakeHost) -> None:
        # 壊れると、設定の中のフォルダの値が、形（区切りの数など）しだいで伏せずに送られる
        host.settings = {
            "preferences": {
                "state_folder": "backup",
                "export_folder": "/書き出し",
                "theme": "dark",
            },
        }
        bridge = EditorBridge(host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.DENY})
        _seen, box = _run(bridge, "get_app_settings")
        preferences = box["result"]["preferences"]  # type: ignore[index]  # 結果は辞書で返る
        assert preferences["state_folder"] != "backup"
        assert "書き出し" not in preferences["export_folder"]
        assert preferences["theme"] == "dark"


# --- 名前と場所を断った人の所で、目印の名前がどこにも出ないこと（Codex の指摘 #366） ---

#: 目印 素材・作品・置き場の名前に入れ、結果のどこかに出たら漏れ
MARK = "ZZSECRET"


def _answer_sends_and_allow(bridge: EditorBridge, done: threading.Event) -> None:
    """送る前の確かめは出ないはず（出たら落とす） 取り消せない操作（保存と書き出し #327）の
    確かめには許すで答える 「変更を自動で承認」でも出るので、答えないと道具が終わらない
    """
    deadline = time.monotonic() + 5.0
    while not done.is_set() and time.monotonic() < deadline:
        bridge.pump()
        request = bridge.take_send_request()
        if request is not None:
            _never(request)
        approval = bridge.take_approval()
        if approval is not None:
            approval.allow()
        time.sleep(0.005)
    bridge.pump()
    assert done.is_set(), "道具が終わらなかった"


def _marked_host(
    video_media: MediaItem, transcript: Transcript, scene: str, tmp_path: Path
) -> FakeHost:
    """目印の名前を持つ作品を、場面ごとの形にする

    - そのまま: 素材も作品の名前も今の作品にある
    - 消した後: 置いた素材をクリップごと消し、素材も外した 名前は履歴と字幕起こしの記録にだけ残る
    - 名前を変えた後: 作品は別の名前で保存してあり、作品の名前だけが目印を持つ
    - 開き直した後: AI に道具を使わせないまま別の作品を開いた 名前は作品の外にだけ残る
    """
    media = replace(video_media, path=Path(f"C:/Users/{MARK}人/動画/{MARK}素材.mp4"))
    project = replace(make_loaded(media, transcript), name=f"{MARK}作品")
    host = FakeHost(project)
    host.settings = {"preferences": {"state_folder": f"/{MARK}退避", "theme": "dark"}}
    host.transcription = f"{MARK}素材.mp4: 終了した"
    # 棚の 1 本を本人の名前の入ったフォルダに置く（#328） 一覧はフォルダの名前を返し、置く道具は
    # 読めない理由に場所を書く どちらも名前と場所を断った人には伏せる
    host.shelf = [
        TemplateEntry(
            name="見本",
            path=Path(f"C:/Users/{MARK}人/テンプレート/見本.object"),
            folder=f"{MARK}のテンプレート",
        )
    ]
    # 変える道具の例（WRITE_EXAMPLES）はテキストのクリップを指す
    _operation("add_text")(host, {"text": BODY_TEXT})
    if scene == "消した後":
        placed = host.document.project.media[0]
        _operation("place_media")(host, {"media_id": str(placed.id)})
        clips = [
            str(clip.id) for track in host.document.project.timeline.tracks for clip in track.clips
        ]
        _operation("delete_clips")(host, {"clip_ids": clips})
        host.apply_commands([RemoveMedia(placed.id)], "素材を削除")
        assert not host.document.project.media
    elif scene == "名前を変えた後":
        host.saved_at = tmp_path / "ふつうの名前.sme"
    elif scene == "開き直した後":
        # AI に一度も道具を使わせないまま、別の作品を開いた 履歴は消え、前の作品の名前は
        # 字幕起こしの記録と、アプリが作品の外で覚えている所（前の作品の場所）にだけ残る
        host.document.reset(replace(make_loaded(video_media, transcript), name="ふつう"))
        _operation("add_text")(host, {"text": BODY_TEXT})
        host.remembered = [f"{MARK}前の作品"]
        host.transcription = f"{MARK}素材.mp4: 終了した（{MARK}前の作品 で始めた起こし）"
    return host


def _every_example(host: FakeHost) -> list[tuple[str, dict[str, Any]]]:
    """読む道具は引数なしで、変える道具は例の引数で、質問の道具を除いて全部"""
    project = host.document.project
    ids = (
        _ids(host)
        if project.media
        else dict.fromkeys(("media", "segment", "track", "video", "text"), "無い")
    )
    calls: list[tuple[str, dict[str, Any]]] = []
    for operation in OPERATIONS:
        if operation.asks:
            continue
        example = WRITE_EXAMPLES.get(operation.name)
        calls.append((operation.name, example(ids) if example is not None else {}))
    return calls


class TestNoNameLeaksWhenNamesAreRefused:
    @pytest.mark.parametrize("scene", ["そのまま", "消した後", "名前を変えた後", "開き直した後"])
    def test_the_mark_never_reaches_the_results(
        self, scene: str, video_media: MediaItem, transcript: Transcript, tmp_path: Path
    ) -> None:
        # 壊れると、名前と場所を断った人の素材や作品の名前（ユーザー名入りの場所も）が、どれかの
        # 道具の結果か断りの文に混ざって外へ出る 今の作品の素材だけを伏せる作りだと、消した
        # 素材の名前（字幕起こしの記録・履歴）や、保存先と違う作品の名前が漏れる
        leaks: list[str] = []
        for name, arguments in _every_example(
            _marked_host(video_media, transcript, scene, tmp_path)
        ):
            # 道具ごとに作り直す 前の道具が素材を消すと、後の道具が名前を出さずに通ってしまう
            host = _marked_host(video_media, transcript, scene, tmp_path)
            bridge = EditorBridge(host, auto_approve=True)
            bridge.set_send_choices(
                {Outbound.FRAMES: SendChoice.ALLOW, Outbound.TEXT: SendChoice.ALLOW}
                | {Outbound.FILES: SendChoice.DENY}
            )
            done, box = call_in_thread(bridge, _operation(name), arguments)
            _answer_sends_and_allow(bridge, done)
            sent = json.dumps(
                {"result": box.get("result"), "error": str(box.get("error") or "")},
                ensure_ascii=False,
                default=str,
            )
            if MARK in sent:
                leaks.append(f"{name}: {sent[:200]}")
        assert not leaks


class TestShortNames:
    """1 文字と 2 文字の作品名・素材名（Codex の指摘 #366）"""

    @staticmethod
    def _host(video_media: MediaItem, transcript: Transcript, name: str) -> FakeHost:
        media = replace(video_media, path=Path(f"C:/素材/{name}.mp4"), display_name=name)
        host = FakeHost(replace(make_loaded(media, transcript), name=name))
        host.transcription = f"{name}: 終了した"
        return host

    @pytest.mark.parametrize("name", ["Q", "QZ"])
    def test_names_returned_as_fields_are_hidden_whatever_their_length(
        self, name: str, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、「A」のような短い作品名や素材名が、長さの決まりで表から外れ、
        # 名前と場所を断った人の get_project や一覧の欄からそのまま送られる
        host = self._host(video_media, transcript, name)
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices(
            dict.fromkeys(ASKED, SendChoice.ALLOW) | {Outbound.FILES: SendChoice.DENY}
        )
        media_id = str(host.document.project.media[0].id)
        fields: dict[str, Callable[[Any], list[str]]] = {
            "get_project": lambda r: [r["name"]],
            "list_media": lambda r: [row["name"] for row in r["media"]],
            "list_clips": lambda r: [row["media"] for row in r["clips"]],
            "place_media": lambda r: [r["placed"]],
        }
        for tool, pick in fields.items():
            arguments = {"media_id": media_id} if tool == "place_media" else {}
            _seen, box = _run(bridge, tool, arguments=arguments)
            values = pick(box["result"])
            assert values, tool
            assert name not in values, f"{tool}: {values}"
        _seen, box = _run(bridge, "transcription_status")
        status = box["result"]["status"]  # type: ignore[index]  # 結果は辞書で返る
        assert not status.startswith(f"{name}:"), status
        assert "終了した" in status

    @pytest.mark.parametrize("name", ["Q", "QZ"])
    def test_no_tool_sends_a_short_name(
        self, name: str, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、短い作品名・素材名が、見張りの目印（長い名前）では見えない道具から漏れる
        # 大文字の Q と Z は ID（小文字の 16 進）にも道具の返す決まった文にも出ないので、
        # 前後が英数字でない Q（QZ）が結果にあれば漏れと見なせる
        found = re.compile(rf"(?<![A-Za-z0-9]){name}(?![A-Za-z0-9])")
        leaks: list[str] = []
        sample = self._host(video_media, transcript, name)
        _operation("add_text")(sample, {"text": BODY_TEXT})
        for tool, arguments in _every_example(sample):
            host = self._host(video_media, transcript, name)
            _operation("add_text")(host, {"text": BODY_TEXT})
            bridge = EditorBridge(host, auto_approve=True)
            bridge.set_send_choices(
                dict.fromkeys(ASKED, SendChoice.ALLOW) | {Outbound.FILES: SendChoice.DENY}
            )
            done, box = call_in_thread(bridge, _operation(tool), arguments)
            answer_sends(bridge, done, _never)
            sent = json.dumps(
                {"result": box.get("result"), "error": str(box.get("error") or "")},
                ensure_ascii=False,
                default=str,
            )
            if found.search(sent):
                leaks.append(f"{tool}: {sent[:200]}")
        assert not leaks

    def test_a_short_name_is_not_hidden_inside_other_words(self) -> None:
        # 壊れると、1 文字の名前（Q）を伏せるために、関係の無い語（QUIT や 30fps）の中まで消える
        table = [("Q", "〔素材 m〕")]
        assert redact("QUIT Q 30fps", table) == "QUIT 〔素材 m〕 30fps"


class TestTheDocumentKeepsOnlyNames:
    def test_seen_media_does_not_hold_subtitles(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、文書が覚えた素材（消した物も）が字幕と単語の時刻を抱えたまま残り、
        # 起こし直すたびにメモリが増え続ける（Codex の指摘 #366）
        host = FakeHost(make_loaded(video_media, transcript))
        document = host.document
        media = document.project.media[0]
        for _ in range(3):
            host.apply_commands([SetTranscript(media.id, transcript)], "字幕")
        seen = document.seen_media
        assert seen
        assert all(not isinstance(entry, MediaItem) for entry in seen)
        assert all(not hasattr(entry, "transcripts") for entry in seen)
        assert {entry.name for entry in seen} == {media.name}


class TestRedact:
    @pytest.mark.parametrize(
        "place",
        [
            "/home/秘密の人/動画/無い.mp4",
            "/Users/秘密の人/Movies/x.mov",
            "file:///C:/Users/秘密の人/x.mp4",
            "file:///home/秘密の人/x.mp4",
            "~/秘密の人の動画/x.mp4",
            "~秘密の人/動画/x.mp4",
        ],
    )
    def test_other_ways_of_writing_a_place_are_hidden(self, place: str) -> None:
        # 壊れると、AI が Windows 以外の書き方で渡した場所が、断った文に混ざってそのまま外へ出る
        assert "秘密の人" not in redact(f"ファイルがありません: {place}", [])

    @pytest.mark.parametrize(
        "kept",
        [
            "30/1",
            "色/明るさ",
            "https://github.com/kagemorikosame/sashimono-edit/issues",
            "1/2 画質",
        ],
    )
    def test_things_that_only_look_like_places_stay(self, kept: str) -> None:
        # 壊れると、フレームレートや分類の名前、ヘルプの URL まで伏せられ、AI が読めなくなる
        assert redact(kept, []) == kept

    def test_a_missing_posix_place_in_an_import_error_is_hidden(self, host: FakeHost) -> None:
        # 壊れると、名前と場所を断った人の所でも、読み込めなかった場所（ユーザー名入り）が
        # import_media の断りの文に混ざって返る（CodeRabbit の指摘 #366）
        bridge = EditorBridge(host, auto_approve=True)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FILES: SendChoice.DENY})
        place = "/home/秘密の人/動画/無い.mp4"
        _seen, box = _run(bridge, "import_media", arguments={"paths": [place]})
        error = box.get("error")
        assert isinstance(error, ToolError)
        assert "ファイルがありません" in str(error)
        assert "秘密の人" not in str(error)

    def test_a_path_inside_a_sentence_is_hidden(self) -> None:
        # 壊れると、表に無い場所（退避のフォルダなど）が文に続けて書かれたとき、伏せずに送る
        text = redact("置き場はC:\\Users\\秘密の人\\退避 です", [])
        assert "秘密の人" not in text
        assert "置き場は" in text

    def test_longer_secrets_are_replaced_first(
        self, video_media: MediaItem, transcript: Transcript
    ) -> None:
        # 壊れると、場所の中のファイル名だけが先に伏せられ、フォルダの名前が残る
        project = make_loaded(replace(video_media, path=SECRET_PATH), transcript)
        table = secret_pairs(file_secrets(project, project_path=None, export_folder=None))
        hidden = redact({"path": SECRET_PATH.as_posix(), "list": [SECRET_PATH.name]}, table)
        assert "秘密の人" not in json.dumps(hidden, ensure_ascii=False)
        assert "〔素材" in hidden["path"]

    def test_the_caption_of_a_frame_keeps_the_timecode(self, host: FakeHost) -> None:
        # 壊れると、伏せ字の処理で絵の説明（何フレーム目か）まで消え、AI がどこの絵か分からない
        bridge = EditorBridge(host)
        bridge.set_send_choices({**bridge.send_choices, Outbound.FRAMES: SendChoice.ALLOW})
        _seen, box = _run(bridge, "preview_frame", arguments={"frame": 30})
        result = box.get("result")
        assert isinstance(result, ImageResult)
        assert format_timecode(30, host.project.rate) in result.caption


# --- 型の無い文字を返す読む道具（#391 の CodeRabbit の指摘）---

#: はみ出した字幕の文 検査の行の text に載る
OVERFLOW_TEXT = "はみ出した秘密の字幕です"


def _overflowing(host: FakeHost) -> None:
    """字幕のはみ出しを 1 つ見つけて終わった検査の様子を入れる（検査そのものは走らせない）"""
    operation = _operation("check_output")
    operation(host, {"checks": ["text_overflow"], "wait_seconds": 0})
    request = host.check_requests[-1]
    base = assemble(
        [],
        start=request.start,
        end=request.end or 300,
        analyzed=request.end or 300,
        fps=30.0,
        complete=True,
        limits=request.limits,
        index=None,
        loudness=None,
        peak=None,
        length_seconds=None,
    )
    problem = Problem(
        kind=TEXT_OVERFLOW,
        severity="medium",
        start=0,
        end=30,
        detail="字が画面の外へ出て切れる",
        measure={"text": OVERFLOW_TEXT, "overflow_px": 40},
    )
    host.check = CheckState(request=request, report=replace(base, problems=(problem,)))


def _compat_with_text(host: FakeHost) -> None:
    recorded = CompatibilityReport()
    recorded.note_control(f"<#ff0000>{OVERFLOW_TEXT}")
    recorded.note_missing(f"debug_print: {OVERFLOW_TEXT}")
    recorded.note_missing("obj.getpixeldata")
    recorded.note_failure("秘密.anm", "開けない: 秘密.anm")
    host.compat = CompatReport.of(recorded, scripts=1, folders=("秘密のフォルダ",))


class TestUntypedTextIsHidden:
    @pytest.mark.parametrize(
        ("name", "arguments", "prepare"),
        [
            ("check_output", {"checks": ["text_overflow"], "wait_seconds": 0}, _overflowing),
            ("get_compat_report", {}, _compat_with_text),
        ],
    )
    def test_refused_text_is_hidden_and_the_rest_is_read(
        self,
        host: FakeHost,
        text_refused: EditorBridge,
        name: str,
        arguments: dict[str, Any],
        prepare: Callable[[FakeHost], None],
    ) -> None:
        """壊れると、字幕のはみ出しの行の文や、互換性レポートの記録に混ざった本人の文
        （制御文字の行・debug_print）を、字幕と文字を送らない人の所でもそのまま送る
        道具そのものは断らず、文以外（種類・回数・時刻）は読める"""
        prepare(host)
        seen, box = _run(text_refused, name, arguments=arguments)
        assert not seen
        assert box.get("error") is None, box.get("error")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert OVERFLOW_TEXT not in sent
        assert TEXT_MARK in sent

    def test_places_in_the_report_are_hidden_when_names_are_refused(self, host: FakeHost) -> None:
        """壊れると、互換性レポートの置き場や失敗の文の相対の名前（秘密.anm）が、名前と場所を
        送らない人の所で場所の形に当たらずに外へ出る"""
        _compat_with_text(host)
        bridge = EditorBridge(host)
        bridge.set_send_choices(
            dict.fromkeys(ASKED, SendChoice.ALLOW) | {Outbound.FILES: SendChoice.DENY}
        )
        _seen, box = _run(bridge, "get_compat_report")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert "秘密" not in sent
        assert '"count"' in sent

    @pytest.mark.parametrize(
        ("name", "arguments", "prepare"),
        [
            ("check_output", {"checks": ["text_overflow"], "wait_seconds": 0}, _overflowing),
            ("get_compat_report", {}, _compat_with_text),
        ],
    )
    def test_weakening_the_declaration_is_caught(
        self,
        host: FakeHost,
        name: str,
        arguments: dict[str, Any],
        prepare: Callable[[FakeHost], None],
    ) -> None:
        """壊れると、本人の文を返す道具から字幕と文字の宣言を外しても見張りが気付かず、
        設定画面と同意の窓に出ないまま文が送られる"""
        prepare(host)
        operation = _operation(name)
        weakened = replace(
            operation,
            sends=operation.sends - {Outbound.TEXT},
            hidden=operation.hidden - {Outbound.TEXT},
        )
        result = weakened(host, arguments)
        assert not undeclared(operation, result, sentences={OVERFLOW_TEXT})
        assert undeclared(weakened, result, sentences={OVERFLOW_TEXT})


# --- 文字の欄（TextSpec）は名前に関係なく字幕と文字として伏せる（#391 の Codex の指摘）---

#: テキストの「タイマーの書式」（text の名前ではない文字の欄）に書いた本人の文
TYPED_TEXT = "書式に書いた秘密の文です"
#: AviUtl のスクリプトの文字の欄（--text の名前を変えた物）に書いた本人の文
SCRIPT_TEXT = "スクリプトの欄に書いた秘密の文"
#: 文字の欄を持つスクリプトの種類 試験の間だけ登録する
SCRIPT_KIND = "aviutl.script:試しの文字@試験"


@pytest.fixture
def scripted() -> Iterator[str]:
    registry.register(
        EffectDefinition(
            kind=SCRIPT_KIND,
            label="試しの文字",
            category="試し",
            parameters=(TextSpec("message", "文字", ""),),
        )
    )
    yield SCRIPT_KIND
    registry.unregister(SCRIPT_KIND)


def _typed_texts(host: FakeHost, kind: str) -> ClipId:
    """text 以外の名前の文字の欄に本人の文を書いたテキストを 1 本置いて選ぶ"""
    _operation("add_text")(host, {"text": "見出し"})
    clip = next(
        c.id
        for track in host.document.project.timeline.tracks
        for c in track.clips
        if c.source is not None and c.source.kind == "text"
    )
    host.apply_commands(
        [
            SetParam(ParamPath.of_source(clip, "timer_format"), TYPED_TEXT),
            AddEffect(clip, Effect(kind=kind, params={"message": SCRIPT_TEXT})),
        ],
        "文字の欄",
    )
    host.select_clip(clip)
    return clip


class TestTypedTextIsHidden:
    def test_every_text_field_is_in_the_table(self, host: FakeHost, scripted: str) -> None:
        """壊れると、伏せ字の表が text の名前の欄しか集めず、スクリプトの文字の欄やタイマーの
        書式に本人が書いた文を、字幕と文字を送らないと決めた人の所でもそのまま送る"""
        _typed_texts(host, scripted)
        secrets = text_secrets(host.document.project)
        assert TYPED_TEXT in secrets
        assert SCRIPT_TEXT in secrets

    def test_reading_a_clip_hides_every_text_field(
        self, host: FakeHost, scripted: str, text_refused: EditorBridge
    ) -> None:
        """壊れると、get_clip がスクリプトの文字の欄の文を、字幕と文字を送らない人の所でも
        外へ送る"""
        _typed_texts(host, scripted)
        _seen, box = _run(text_refused, "get_clip")
        assert box.get("error") is None, box.get("error")
        sent = json.dumps(box["result"], ensure_ascii=False)
        assert TYPED_TEXT not in sent
        assert SCRIPT_TEXT not in sent
        # 文字でない値（大きさ）は読める
        assert '"size"' in sent

    @pytest.mark.parametrize("kind", [Outbound.TEXT, Outbound.FILES])
    def test_weakening_the_declaration_is_caught(
        self, host: FakeHost, scripted: str, kind: Outbound
    ) -> None:
        """壊れると（get_clip が字幕と文字や名前と場所を送ると書かなくなると）、設定画面と
        同意の窓に出ないまま、文字の欄と場所の欄の値が送られるのに、送る物の見張りが気付かない"""
        clip = _typed_texts(host, scripted)
        image = Effect(kind="image_blend", params={"image_file": str(SECRET_PATH)})
        host.apply_commands([AddEffect(clip, image)], "画像")
        operation = _operation("get_clip")
        weakened = replace(
            operation, sends=operation.sends - {kind}, hidden=operation.hidden - {kind}
        )
        # 編集画面の中から直に呼んだ形（許しを立てない）で、道具が返しうる物を全部見る
        result = weakened(host, {})
        sentences = {SCRIPT_TEXT, TYPED_TEXT}
        assert not undeclared(operation, result, sentences=sentences, places=READ_PLACES)
        assert undeclared(weakened, result, sentences=sentences, places=READ_PLACES)
