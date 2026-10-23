"""コマンドの一覧と AI の道具の対応を見張る（#351 E5）

画面の操作はどれも ``core`` の :class:`~sashimono.core.commands.Command` を通る（変更は必ず
コマンド経由） コマンドを足したのに AI の道具が 1 つも使わず、使わない理由も書いていなければ、
人にできて AI にできない操作が黙って増える ここでそれを止める

AI が使うかどうかは、道具を実際に動かして :meth:`Document.execute` に届いたコマンドで数える
名前を探すだけでは、``place_media`` や ``apply_preset`` のように手順の関数の中で作るコマンドを
取りこぼす 道具を動かす例は ``test_outbound.py`` の ``WRITE_EXAMPLES``（変える道具を全部
持つ）と、ここの ``EXAMPLES``（その例では断られる道具や、別の道を通る引数）

使わないコマンドは ``sashimono.ai.command_ledger.COMMANDS_WITHOUT_TOOLS`` に理由を書く
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import sashimono.ai
import sashimono.core
import sashimono.ui
from sashimono.ai.command_ledger import COMMANDS_WITHOUT_TOOLS, Why
from sashimono.ai.host import ToolError
from sashimono.ai.ledger import CHILD_ISSUES
from sashimono.ai.operations import OPERATIONS, find_operation
from sashimono.compat.aviutl.catalog import ScriptCatalog
from sashimono.compat.looks import capture_preset
from sashimono.core.commands import (
    AddClip,
    Command,
    Document,
    InScene,
    ParamPath,
    RemoveClip,
    RenameTrack,
    SetParam,
)
from sashimono.core.commands.fixed import TRANSFORM_EFFECT_KIND, with_fixed_items
from sashimono.core.io.atomic import expect_now
from sashimono.core.io.serialize import save_project
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    MediaItem,
    Project,
    Track,
    TrackKind,
    Transcript,
)
from sashimono.core.model.ids import ClipId, TrackId
from sashimono.effects.sources import TEXT
from tests.ai.conftest import FakeHost
from tests.ai.test_look_tools import (
    _EXO_IMAGE,
    _empty_host,
    _shelf_entry,
    _template_id,
    _yellow_preset,
)
from tests.ai.test_mix_tools import _measured, _Work
from tests.ai.test_outbound import WRITE_EXAMPLES, _ids, _with_text_clip
from tests.compat.test_subtitle_looks import _styled_alias
from tests.core.test_seams import JumpCut


def _subclasses(base: type) -> Iterator[type]:
    # 要素の型を書いておく 書かないと、キャッシュの無い mypy（CI）が型を決められずに落ちる
    # 基底は抽象クラスの Command なので type[Command] では受けられない
    found: list[type] = base.__subclasses__()
    for sub in found:
        yield sub
        yield from _subclasses(sub)


def core_commands() -> dict[str, type[Command]]:
    """``sashimono.core`` の具体的なコマンドすべて 名前から引く

    ``core/commands/`` の外（``core/seams.py`` の ``SmoothCuts`` など）にも置いてあるので、
    ``core`` の下を全部読み込んでから受け継いだクラスを辿る ``__init__`` の並びで数えると、
    並びに書き忘れたコマンドを見落とす
    """
    for info in pkgutil.walk_packages(sashimono.core.__path__, "sashimono.core."):
        importlib.import_module(info.name)
    return {
        cls.__name__: cls
        for cls in _subclasses(Command)
        if cls.__module__.startswith("sashimono.core.") and not inspect.isabstract(cls)
    }


@contextmanager
def recording() -> Iterator[set[str]]:
    """文書に届いたコマンドの名前を集める シーンの中の実行は中のコマンドも数える

    道具が手順を組むときに ``apply`` で試すだけのコマンドは数えない（作品は変わらない）
    断られたコマンドも数えない 作品を変えていないのに数えると、いつも断られる道のコマンドを
    「AI が使う」と数え、理由を求めずに通してしまう
    """
    seen: set[str] = set()
    original = Document.execute

    def execute(self: Document, command: Command, **options: Any) -> Any:
        # 通った後で数える 先に数えると、例外で断られて作品を変えなかったコマンドまで残る
        result = original(self, command, **options)
        seen.add(type(command).__name__)
        if isinstance(command, InScene):
            seen.add(type(command.command).__name__)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Document, "execute", execute)
        yield seen


@dataclass(frozen=True, slots=True)
class Material:
    """例が作品を組むための材料（試験の fixture）"""

    video_media: MediaItem
    transcript: Transcript
    tmp_path: Path


#: 例 1 つ 作品を組んで、動かす道具・引数・ホストを返す 組む間のコマンドは数えない
Example = Callable[[Material], tuple[FakeHost, str, dict[str, Any]]]


def _run(host: FakeHost, tool: str, **arguments: Any) -> Any:
    operation = find_operation(tool)
    assert operation is not None
    return operation(host, arguments)


def _text_host(material: Material) -> FakeHost:
    return _with_text_clip(material.video_media, material.transcript)


def _keyframe(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    return host, "add_keyframe", {"clip_id": _ids(host)["text"], "name": "size", "value": 60}


def _ripple_delete(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    return host, "delete_clip", {"clip_id": _ids(host)["video"], "ripple": True}


def _stroke(tool: str, **extra: Any) -> Example:
    def example(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
        host = _text_host(material)
        text = _ids(host)["text"]
        stroke = _run(host, "add_stroke", clip_id=text)["stroke_id"]
        _run(host, "add_stroke", clip_id=text)
        return host, tool, {"clip_id": text, "stroke_id": stroke, **extra}

    return example


def _ungroup(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    ids = _ids(host)
    _run(host, "group_clips", clip_ids=[ids["video"], ids["text"]])
    return host, "ungroup_clips", {"clip_ids": [ids["video"]]}


def _place_scene(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    scene = _run(host, "add_scene", name="場面", open=False)["scene_id"]
    return host, "place_scene", {"scene_id": scene, "at_frame": 400}


def _in_scene(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    # 開いたシーンの中の編集は、ホストがシーンの中で実行する命令に包む（画面も同じ）
    host = _text_host(material)
    _run(host, "add_scene", name="場面", open=True)
    return host, "add_text", {"text": "シーンの中"}


def _import_media(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    path = material.tmp_path / "追加.mp4"
    path.write_bytes(b"")
    host.probe_result = material.video_media
    return host, "import_media", {"paths": [str(path)]}


def _place_subtitles(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    return host, "place_subtitles", {"media_id": _ids(host)["media"]}


def _jet_cut(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    from tests.ui.test_subtitle_panel import make_waveform

    work = JumpCut(durations=(900, 1, 1))
    host = FakeHost(work.project)
    host.stub_waveform = make_waveform([(0.5, 300), (0.0, 200), (0.5, 5500)])
    media = str(work.project.media[0].id)
    return host, "jet_cut", {"media_id": media, "keep_speech": False, "smooth": True}


def _rename_speaker(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    # WRITE_EXAMPLES の作品の字幕には話者が無く「話者 1」が見つからずに断られる
    # 分け終えた字幕（話者の付いた形）を置いて、名前を変える道を通す
    named = replace(
        material.transcript,
        segments=tuple(replace(s, speaker="話者 1") for s in material.transcript.segments),
    )
    host = _with_text_clip(material.video_media, named)
    arguments = {"media_id": _ids(host)["media"], "from": "話者 1", "to": "山田"}
    return host, "rename_speaker", arguments


def _smooth_cuts(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    return FakeHost(JumpCut().project), "smooth_cuts", {}


def _mix(tool: str, arguments: Callable[[_Work], dict[str, Any]]) -> Example:
    def example(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
        work = _Work(material.transcript, material.video_media)
        return work.host, tool, arguments(work)

    return example


def _normalize(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    work = _Work(material.transcript, material.video_media)
    _run(work.host, "normalize_loudness", wait_seconds=0)
    _measured(work.host, -20.0, -10.0)
    return work.host, "normalize_loudness", {"wait_seconds": 0}


def _apply_preset(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    _yellow_preset(host)
    return host, "apply_preset", {"name": "黄色", "clip_ids": [_ids(host)["text"]]}


def _apply_switched_off_preset(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    # 最初から持つ欄（配置）を切った見た目を当てると、欄を消さずに入り切りだけを写す
    host = _text_host(material)
    look = with_fixed_items(
        Clip(timeline_start=0, duration=90, source=TEXT.create(text="保存した文字")),
        picture=True,
    )
    look = replace(
        look,
        effects=tuple(
            replace(effect, enabled=False) if effect.kind == TRANSFORM_EFFECT_KIND else effect
            for effect in look.effects
        ),
    )
    preset = capture_preset("配置なし", None, Track(TrackKind.VIDEO, "V", (look,)), look)
    host.library.presets.save(preset)
    return host, "apply_preset", {"name": "配置なし", "clip_ids": [_ids(host)["text"]]}


def _place_alias(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    host.library.aliases.save(_styled_alias("テロップ"))
    return host, "place_alias", {"name": "テロップ", "at_frame": 400}


def _template(tool: str) -> Example:
    def example(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
        host = _text_host(material)
        entry = _shelf_entry(material.tmp_path)
        host.shelf = [entry]
        if tool == "apply_template":
            # 着せる先のエフェクトは外してから着せる（前に着せた物を残さない）
            text = _ids(host)["text"]
            _run(host, "add_effect", clip_id=text, kind="blur")
            return host, tool, {"template_id": _template_id(entry), "clip_id": text}
        return host, tool, {"template_id": _template_id(entry), "at_frame": 400}

    return example


def _import_objects(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _empty_host()
    host.probe_result = MediaItem(path=material.tmp_path / "仮.png")
    picture = material.tmp_path / "絵.png"
    picture.write_bytes(b"")
    source = material.tmp_path / "素材つき.exo"
    source.write_text(_EXO_IMAGE.format(path=picture), encoding="utf-8")
    return host, "import_objects", {"path": str(source)}


def _with_effects(tool: str, **extra: Any) -> Example:
    # エフェクトを 2 つ付けてから 1 つ目を指す 並べ替えは 2 つ無いと動かす先が無い
    def example(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
        host = _text_host(material)
        text = _ids(host)["text"]
        effect = _run(host, "add_effect", clip_id=text, kind="blur")["effect_id"]
        _run(host, "add_effect", clip_id=text, kind="blur")
        if tool == "set_effect_enabled":
            return host, tool, {"clip_id": text, "effect_ids": [effect], **extra}
        return host, tool, {"clip_id": text, "effect_id": effect, **extra}

    return example


def _with_keyframes(tool: str) -> Example:
    def example(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
        host = _text_host(material)
        text = _ids(host)["text"]
        frame = _run(host, "add_keyframe", clip_id=text, name="size", value=60)["frame"]
        _run(host, "add_keyframe", clip_id=text, name="size", value=80, frame=frame + 10)
        return host, tool, {"clip_id": text, "name": "size", "frame": frame}

    return example


def _paste(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    _run(host, "copy_clips", clip_ids=[_ids(host)["text"]])
    return host, "paste_clips", {"insert": True}


def _remove_track(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    # 中身のあるトラックは断るので、空のトラックを足してから消す
    host = _text_host(material)
    track = _run(host, "add_track", kind="video", name="空")["track_id"]
    return host, "remove_track", {"track_id": track}


def _scene(tool: str, **extra: Any) -> Example:
    def example(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
        host = _text_host(material)
        scene = _run(host, "add_scene", name="場面", open=False)["scene_id"]
        return host, tool, {"scene_id": scene, **extra}

    return example


def _remove_media(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    # 使っている素材は断るので、置いたクリップを先に消す
    host = _text_host(material)
    ids = _ids(host)
    _run(host, "delete_clip", clip_id=ids["video"])
    return host, "remove_media", {"media_id": ids["media"]}


def _split_subtitle(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    # 1 枚目の字幕は 1〜3 秒 その内側で割る
    host = _text_host(material)
    ids = _ids(host)
    arguments = {"media_id": ids["media"], "segment_id": ids["segment"], "at_seconds": 2}
    return host, "split_subtitle", arguments


def _saved_host(material: Material) -> FakeHost:
    """保存したての作品と、同じフォルダの別の作品 第二稿.sme 開く・新規は保存していないと断る"""
    host = _text_host(material)
    host.saved_at = material.tmp_path / "作品.sme"
    host.write_project(host.saved_at, adopt=False, expect=expect_now(host.saved_at))
    save_project(replace(Project.create(), name="第二稿"), material.tmp_path / "第二稿.sme")
    return host


def _open_project(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    return _saved_host(material), "open_project", {"path": "第二稿.sme"}


def _new_project(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    return _saved_host(material), "new_project", {"width": 1280, "height": 720}


def _custom_object(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    host = _text_host(material)
    catalog = ScriptCatalog(roots=())
    host.scripts = [catalog.add_text("aviutl:試験.obj:円", "@円\n", kind="obj")]
    return host, "add_custom_object", {"name": "円", "at_frame": 400}


def _align(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    # 偽のホストは描かないので枠を持たない プレビューが作る命令を入れておく
    host = _text_host(material)
    text = ClipId(_ids(host)["text"])
    host.aligned = [SetParam(ParamPath.of_source(text, "size"), AnimatedValue(80.0))]
    return host, "align_clip", {"clip_id": str(text), "anchor": "top_left"}


def _standard_layout(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
    # AviUtl2 の組み方で灰色になっているテキストだけが切り替わる
    host = _text_host(material)
    greyed = with_fixed_items(
        Clip(
            timeline_start=400,
            duration=90,
            source=TEXT.create(text="字", layout="aviutl", align="right"),
        ),
        picture=True,
    )
    track = host.document.project.timeline.tracks[0]
    host.apply_commands([AddClip(track.id, greyed)], "置く")
    return host, "use_standard_layout", {"clip_ids": [str(greyed.id)]}


#: ``WRITE_EXAMPLES`` の例では断られる道具と、別のコマンドを通る引数 足すときは、その例で
#: 道具が断られずに通ることを ``test_every_example_goes_through`` が確かめる
EXAMPLES: dict[str, Example] = {
    "add_keyframe": _keyframe,
    "delete_clip ripple": _ripple_delete,
    "remove_stroke": _stroke("remove_stroke"),
    "move_stroke": _stroke("move_stroke", index=1),
    "set_stroke_enabled": _stroke("set_stroke_enabled", enabled=False),
    "ungroup_clips": _ungroup,
    "place_scene": _place_scene,
    "add_text in a scene": _in_scene,
    "import_media": _import_media,
    "place_subtitles": _place_subtitles,
    "jet_cut": _jet_cut,
    "smooth_cuts": _smooth_cuts,
    "rename_speaker": _rename_speaker,
    "set_volume": _mix("set_volume", lambda w: {"clip_ids": [str(w.bgm.id)], "volume_percent": 50}),
    "duck_audio": _mix(
        "duck_audio",
        lambda w: {"clip_ids": [str(w.bgm.id)], "voice_clip_ids": [str(w.voice.id)]},
    ),
    "normalize_loudness": _normalize,
    "place_sound": _mix("place_sound", lambda w: {"media_id": str(w.bgm_media.id)}),
    "apply_preset": _apply_preset,
    "apply_preset switched off": _apply_switched_off_preset,
    "place_alias": _place_alias,
    "place_template": _template("place_template"),
    "apply_template": _template("apply_template"),
    "import_objects": _import_objects,
    # 画面と同じ道具（#349） WRITE_EXAMPLES は無い物や中身のある入れ物を指して断られる道を見る
    "paste_clips": _paste,
    "remove_track": _remove_track,
    "rename_scene": _scene("rename_scene", name="場面 2"),
    "remove_scene": _scene("remove_scene"),
    "remove_media": _remove_media,
    "remove_effect": _with_effects("remove_effect"),
    "move_effect": _with_effects("move_effect", index=1),
    "set_effect_enabled": _with_effects("set_effect_enabled", enabled=False),
    "remove_keyframe": _with_keyframes("remove_keyframe"),
    "clear_keyframes": _with_keyframes("clear_keyframes"),
    "split_subtitle": _split_subtitle,
    "open_project": _open_project,
    "new_project": _new_project,
    "add_custom_object": _custom_object,
    "align_clip": _align,
    "use_standard_layout": _standard_layout,
}


def _write_example(name: str) -> Example:
    def example(material: Material) -> tuple[FakeHost, str, dict[str, Any]]:
        host = _text_host(material)
        host.saved_at = material.tmp_path / "作品.sme"
        return host, name, WRITE_EXAMPLES[name](_ids(host))

    return example


#: 道具を動かす関数 :data:`OPERATIONS` の 1 つか、試験が作る偽物
Run = Callable[[Any, dict[str, Any]], Any]


def tally(operation: Run, host: FakeHost, arguments: dict[str, Any]) -> tuple[set[str], str | None]:
    """道具を 1 回動かし、作品に残ったコマンドと断りの文（通ったら None）を返す

    断られたら何も数えない 途中まで通ったコマンドも ``checkpoint`` が元へ戻すので作品に
    残らない 数えると、断られる道しか通らないコマンドが見張りの数えから外れる
    """
    with recording() as seen:
        try:
            operation(host, arguments)
        except (ToolError, ValueError, KeyError) as exc:
            return set(), str(exc)
    return seen, None


def _used(material: Material) -> tuple[dict[str, set[str]], dict[str, str]]:
    """道具ごとに通ったコマンドと、断られた例（例の名前 → 断りの文）"""
    used: dict[str, set[str]] = {}
    refused: dict[str, str] = {}
    examples = {f"WRITE_EXAMPLES {name}": _write_example(name) for name in WRITE_EXAMPLES}
    examples |= {f"EXAMPLES {name}": example for name, example in EXAMPLES.items()}
    for label, example in examples.items():
        host, tool, arguments = example(material)
        operation = find_operation(tool)
        assert operation is not None, label
        seen, refusal = tally(operation, host, arguments)
        if refusal is not None:
            refused[label] = refusal
        used.setdefault(tool, set()).update(seen)
    return used, refused


def uncovered(
    commands: set[str], used: set[str], reasons: set[str]
) -> tuple[set[str], set[str], set[str]]:
    """見張りの判定 （道具も理由も無い・道具があるのに理由も書いた・無いコマンドの理由）"""
    return commands - used - reasons, used & reasons, reasons - commands


@pytest.fixture
def material(video_media: MediaItem, transcript: Transcript, tmp_path: Path) -> Material:
    return Material(video_media, transcript, tmp_path)


class TestEveryCommandHasAToolOrAReason:
    def test_a_new_command_without_a_tool_or_a_reason_fails(self, material: Material) -> None:
        # 壊れると、画面の操作のためにコマンドを足しても AI の道具が無いまま黙って通り、
        # 人にできて AI にできない操作が数えられずに増える（#298 の操作の同等性が崩れる）
        used, _ = _used(material)
        commands = set(core_commands())
        lacking, both, unknown = uncovered(
            commands, set().union(*used.values()), set(COMMANDS_WITHOUT_TOOLS)
        )
        assert not lacking, (
            "AI の道具が使わず、理由も無いコマンド 道具を足すか、"
            f"sashimono/ai/command_ledger.py に理由を書いてください: {sorted(lacking)}"
        )
        assert not both, (
            "AI の道具が使っているのに、使わない理由が残っているコマンド "
            f"command_ledger.py から消してください: {sorted(both)}"
        )
        assert not unknown, f"もう無いコマンドの理由: {sorted(unknown)}"

    def test_the_check_names_what_is_missing(self) -> None:
        # 壊れると、見張りの判定が足したコマンドを見落とし、上の試験が何を足しても通る
        lacking, both, unknown = uncovered({"A", "B", "C"}, {"A"}, {"B", "D"})
        assert lacking == {"C"}
        assert both == set()
        assert unknown == {"D"}
        assert uncovered({"A"}, {"A"}, {"A"})[1] == {"A"}

    def test_every_example_goes_through(self, material: Material) -> None:
        # 壊れると、例が断られて道具がコマンドを通さないまま、コマンドを「AI が使わない」と
        # 数えて理由を求めるか、理由を消した後に道を外れても気付かない
        _, refused = _used(material)
        # WRITE_EXAMPLES は外へ送る物を見る例で、無い物を指して断られる道を通す物がある
        # その道具には、通る例を EXAMPLES に置く（例の名前の頭は道具の名前）
        covered = {label.split()[0] for label in EXAMPLES} | _REFUSED_ON_PURPOSE
        expected = {f"WRITE_EXAMPLES {name}" for name in WRITE_EXAMPLES if name in covered}
        assert set(refused) <= expected, {k: v for k, v in refused.items() if k not in expected}
        assert not {k for k in refused if k.startswith("EXAMPLES ")}

    def test_a_refused_run_counts_nothing(self, material: Material) -> None:
        # 壊れると、1 本目を通して 2 本目で断られた道具（作品は checkpoint で元へ戻る）の
        # 1 本目を「AI が使う」と数え、いつも断られる道のコマンドを理由なしに見逃す
        host = _text_host(material)
        track = TrackId(_ids(host)["track"])
        before = host.document.project

        def refused_halfway(host: FakeHost, arguments: dict[str, Any]) -> None:
            host.apply_commands(
                [RenameTrack(track, "通る"), RemoveClip(ClipId("無いクリップ"))], "試し"
            )

        seen, refusal = tally(refused_halfway, host, {})
        assert refusal is not None
        assert host.document.project == before
        assert seen == set()

    def test_a_command_refused_inside_a_tool_is_not_counted(self, material: Material) -> None:
        # 壊れると、道具が試して断られ、別の道へ回ったコマンドを「AI が使う」と数え、
        # 作品へ一度も届かないコマンドを見張りの数えから外す
        host = _text_host(material)
        track = TrackId(_ids(host)["track"])

        def falls_back(host: FakeHost, arguments: dict[str, Any]) -> None:
            with pytest.raises(KeyError):
                host.document.execute(RemoveClip(ClipId("無いクリップ")))
            host.apply_commands([RenameTrack(track, "通る")], "試し")

        seen, refusal = tally(falls_back, host, {})
        assert refusal is None
        assert seen == {"RenameTrack"}

    def test_every_writing_tool_is_run(self) -> None:
        # 壊れると、例の無い変える道具が通すコマンドを数えず、そのコマンドに要らない理由を求める
        writing = {operation.name for operation in OPERATIONS if operation.writes}
        assert writing <= set(WRITE_EXAMPLES)


class TestReasons:
    def test_every_reason_says_why(self) -> None:
        # 壊れると、理由の欄が空のまま通り、後から読む人が道具を足すべきか決められない
        for name, note in COMMANDS_WITHOUT_TOOLS.items():
            assert note.reason.strip(), name
            if note.why is Why.PERSON_ONLY:
                # 人の操作だけの物は、道具を足す先の Issue か、足さない訳を書く
                assert note.issue is not None or "足さない" in note.reason, name
            if note.issue is not None:
                # 足す先は #298 の子（A1〜E6） 範囲の外だと、どの段で足すかが親から辿れない
                assert note.issue in CHILD_ISSUES, name

    def test_commands_said_to_be_unused_by_the_screen_are_not_named_there(self) -> None:
        # 壊れると、画面が使い始めたコマンドを「どちらも使わない」と書いたまま残し、
        # 人にできて AI にできない操作を数えから外す
        named = _names_in(sashimono.ui) | _names_in(sashimono.ai)
        for name, note in COMMANDS_WITHOUT_TOOLS.items():
            if note.why is Why.NEITHER:
                assert name not in named, name


#: WRITE_EXAMPLES のうち、断られてよい例 どちらもコマンドを通さない道具で、作ったばかりの
#: 作品ではやり直す物も止める書き出しも無い
_REFUSED_ON_PURPOSE = frozenset({"redo", "cancel_export"})


def _names_in(package: Any) -> set[str]:
    """パッケージのソースに出てくる名前（読むだけ 読み込まない）"""
    names: set[str] = set()
    for path in Path(package.__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name.rsplit(".", 1)[-1])
    return names
