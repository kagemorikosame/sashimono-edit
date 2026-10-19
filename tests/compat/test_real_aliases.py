"""このマシンに実際に置かれている配布エイリアスを、全部読ませてみる

``%PROGRAMDATA%\\aviutl2\\Alias`` にある実物が対象 合成した見本ではなく、
人が配ったものをそのまま通すのが目的なので、無ければ黙って飛ばす

ここが落ちたときに疑うのは対応表であって、エイリアスではない
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sashimono.compat.aviutl.exo import load_exo
from sashimono.compat.aviutl.mapping import map_object
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.mapped import MappedObject
from sashimono.core.timebase import FrameRate


def alias_root() -> Path | None:
    program_data = os.environ.get("PROGRAMDATA")
    if not program_data:
        return None
    root = Path(program_data) / "aviutl2" / "Alias"
    return root if root.is_dir() else None


def fixture_root() -> Path | None:
    """落としてきた配布物の置き場 リポジトリには入れていない"""
    root = Path(__file__).resolve().parent.parent / "fixtures" / "aviutl"
    return root if root.is_dir() else None


def _files() -> list[Path]:
    """AviUtl2 世代（``.object``）だけ

    AviUtl1 の ``.exa`` は ``test_real_aviutl1_exa.py`` が見る ここへ混ぜると、字幕の
    ひな形（本文が空で、差し込む道具が埋める）やフォント欄を持たない書き方まで
    AviUtl2 の決まりで裁くことになり、世代の違いを読み手の不具合と取り違える
    """
    found: list[Path] = []
    for root in (alias_root(), fixture_root()):
        if root is not None:
            found.extend(root.rglob("*.object"))
    return sorted(set(found))


ROOT = alias_root()
FILES = _files()

pytestmark = pytest.mark.skipif(not FILES, reason="このマシンに AviUtl の配布エイリアスが無い")


@pytest.fixture(scope="module")
def mapped() -> list[tuple[Path, MappedObject]]:
    rate = FrameRate(30)
    report = CompatibilityReport()
    results: list[tuple[Path, MappedObject]] = []
    for path in FILES:
        for obj in load_exo(path).objects:
            item = map_object(obj, rate, report=report)
            if item is not None:
                results.append((path, item))
    return results


def test_every_file_maps_to_at_least_one_clip(mapped: list[tuple[Path, MappedObject]]) -> None:
    # 1 つも写せない配布エイリアスは、
    # AviUtl のオブジェクトの読み込みやテンプレートの棚から置いても何も入らない
    written = {path for path, _ in mapped}
    missing = [path.name for path in FILES if path not in written]
    assert not missing, f"写せなかった: {missing}"


def test_they_are_all_read_as_the_second_generation() -> None:
    # ``.object`` は AviUtl2 世代 1 本でも 1 世代として読まれていたら、
    # テキスト欄を 16 進として復号しようとして中身が消える
    assert all(load_exo(path).generation == 2 for path in FILES)


def test_the_text_survives(mapped: list[tuple[Path, MappedObject]]) -> None:
    # 空文字になっていたら、テキスト欄の読み方が世代と合っていない
    #
    # カウンター（数を数えるカスタムオブジェクト）は除く 文字を持たず、
    # 数字をタイマーが出すので、空なのが正しい姿
    texts = [
        source.params.get("text")
        for _, item in mapped
        if (source := item.clip.source) is not None
        and source.kind == "text"
        and "timer_format" not in source.params
    ]
    assert texts
    assert all(isinstance(text, str) and text for text in texts)


def test_the_fonts_survive(mapped: list[tuple[Path, MappedObject]]) -> None:
    # 配布物はフォント指定が肝 既定フォントに落ちていたら見た目が別物になる
    #
    # 数えるのはテキストのものだけ ファイルの数と比べると、図形や試験用の
    # エイリアスを置いた時点で落ちる（フォントを持たないのが正しい姿なので）
    texts = [
        item.clip.source
        for _, item in mapped
        if item.clip.source is not None and item.clip.source.kind == "text"
    ]
    named = [source for source in texts if "font" in source.params]
    assert texts
    assert len(named) >= len(texts) - 3  # 数本はフォント欄が空のものがある


def test_no_alias_collapses_to_a_single_frame(mapped: list[tuple[Path, MappedObject]]) -> None:
    # 長さを持つと書いてあるのに 1 フレームなら、``frame=`` の読み方が違う
    with_span = [item for _, item in mapped if item.has_span]
    assert with_span
    assert all(item.clip.duration > 1 for item in with_span)


def test_the_decorated_ones_actually_get_a_decoration(
    mapped: list[tuple[Path, MappedObject]],
) -> None:
    """``文字装飾`` が付いているものは、縁か影のどちらかを持つ"""
    decorated = 0
    for path, item in mapped:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if "文字装飾=標準文字" in raw or "文字装飾=" not in raw:
            continue
        source = item.clip.source
        assert source is not None
        assert "border_width" in source.params or "shadow_x" in source.params, path.name
        decorated += 1
    assert decorated > 0


def test_every_middle_point_becomes_a_keyframe(
    mapped: list[tuple[Path, MappedObject]],
) -> None:
    """中間点を持つエイリアスは、動く値を持って写っていること

    中間点があるのにどの値も動いていなければ、移動方法の行を読み落としている
    """
    from sashimono.compat.aviutl.exo import load_exo as _load
    from sashimono.core.model import AnimatedValue

    with_points = [path for path in FILES if len(_load(path).objects[0].points) > 2]
    if not with_points:
        pytest.skip("中間点を持つ配布物が手元に無い")
    moving = [
        item
        for path, item in mapped
        if path in with_points
        for effect in item.clip.effects
        for value in effect.params.values()
        if isinstance(value, AnimatedValue) and value.is_animated
    ]
    assert moving, "中間点のあるエイリアスに動く値が 1 つも無い"


def test_a_real_moving_alias_survives_being_restyled(
    mapped: list[tuple[Path, MappedObject]],
) -> None:
    """中間点を持つ**実物**を着せても、動きが残って尺に合う

    合成した見本だけで確かめると、実物の書き方（中間点の数や移動方法の
    並び）から外れていても気付けない
    """
    from sashimono.compat.aviutl.exo import load_exo as _load
    from sashimono.compat.catalog import restyle
    from sashimono.core.commands import AddEffect
    from sashimono.core.model import AnimatedValue, Clip, GeneratedSource

    with_points = [path for path in FILES if len(_load(path).objects[0].points) > 2]
    moving = [
        (path, item)
        for path, item in mapped
        if path in with_points and item.clip.source is not None and item.clip.source.kind == "text"
    ]
    if not moving:
        pytest.skip("中間点を持つ文字の配布物が手元に無い")

    checked = 0
    for path, item in moving:
        # 元より短いクリップへ着せる ここで動きが切れると、着地した見た目が出ない
        target = Clip(
            timeline_start=0,
            duration=max(2, item.clip.duration // 3),
            source=GeneratedSource(kind="text", params={"text": "自分で打った字幕"}),
        )
        added = [c for c in restyle([item], target) if isinstance(c, AddEffect)]
        last = target.duration - 1
        for command in added:
            for name, value in command.effect.params.items():
                if not isinstance(value, AnimatedValue) or not value.keyframes:
                    continue
                frames = [keyframe.frame for keyframe in value.keyframes]
                assert frames == sorted(set(frames)), f"{path.name}: {name} の順が崩れた"
                assert max(frames) <= last, f"{path.name}: {name} がクリップの外に出た"
                assert max(frames) == last, f"{path.name}: {name} が終わりまで届かない"
                checked += 1
    assert checked, "動く値を持つ実物が 1 つも無い"


def test_the_keyframes_start_at_the_clip_head(
    mapped: list[tuple[Path, MappedObject]],
) -> None:
    """キーフレームは**クリップ先頭から**数える

    エイリアスの ``frame=244,333,423`` はタイムライン上の位置 そのまま
    キーフレームにすると、クリップの先頭では動かず 244 フレーム待ってから
    動き出す 着せるときの尺合わせも 0 から数える前提で組んである
    """
    from sashimono.core.model import AnimatedValue

    checked = 0
    for path, item in mapped:
        for effect in item.clip.effects:
            for name, value in effect.params.items():
                if not isinstance(value, AnimatedValue) or not value.keyframes:
                    continue
                assert value.keyframes[0].frame == 0, f"{path.name}: {name} が 0 から始まらない"
                checked += 1
    if not checked:
        pytest.skip("動く値を持つ配布物が手元に無い")


#: 素材ファイルを読み込む中身 音声波形表示も自分のファイルの音を描く
MEDIA_CONTENTS = frozenset({"動画ファイル", "画像ファイル", "音声ファイル", "音声波形表示"})


def test_every_media_file_lands_in_the_media_list() -> None:
    """素材を読み込むオブジェクトは、どれもパスが素材一覧に載り、クリップと結び付く

    AviUtl2 は ``ファイル=`` と書くのに ``file=`` だけを見ていたので、AviUtl2 に
    作らせた音声ファイルのエイリアスは素材一覧に載らなかった 選ぶのは中身の名前だけで、
    項目名は決め打ちしない（書き方がまた変わっても、この試験は空振りせずに落ちる）
    """
    from sashimono.compat.aviutl.mapping import media_paths

    rate = FrameRate(30)
    checked = 0
    for path in FILES:
        exo = load_exo(path)
        wanted = [obj for obj in exo.objects if obj.content and obj.content.name in MEDIA_CONTENTS]
        if not wanted:
            continue
        listed = media_paths(exo)
        for obj in wanted:
            item = map_object(obj, rate, report=CompatibilityReport())
            assert item is not None, path.name
            assert item.media_path, f"{path.name}: 素材のパスを読めない"
            assert item.media_path in listed, f"{path.name}: {item.media_path!r} が一覧に無い"
            checked += 1
    if not checked:
        pytest.skip("素材を読み込む実物が手元に無い")
