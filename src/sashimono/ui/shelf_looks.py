"""テンプレートの棚の見本の絵（#277）

棚の配布物も、プリセット・エイリアスと同じ見本の絵の係（:mod:`sashimono.ui.library_thumbnails`）
で描く 作り置き・見えている物から描く・GPU の無い機械の扱いはどれも同じ物を使う

* **鍵は中身の指紋** ファイルの中身（YMM4 は何本目かも）から作る 場所と更新時刻だけに
  すると、同じ配布物を別の置き場へ写しただけ・時刻だけ変わっただけで描き直す 中身を読む
  重さは、同じファイル（大きさと更新時刻が同じ）なら 1 度だけにする（YMM4 は 1 つのファイルに
  100 本超が入っている）
* **中身は走り係で読む** 配布物を読んで写すのは重い（YMM4 の大きな物で数十 ms）ので、
  画面のスレッドでは読まない 読めない物は :class:`SampleError` にして、続く失敗として覚える
* **素材は結ばない** 画像や音を使う配布物は、見本では素材を読み込まない（読み込むと
  素材の一覧へ登録する操作になる） 文字と図形とエフェクトの見た目を出す
* **エフェクトだけの配布物**（YMM4 のアニメーション効果） 見本の文字に着せて描く
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.compat.catalog import TemplateEntry, TemplateError, place, restyle
from sashimono.core.commands.fixed import with_fixed_items
from sashimono.core.commands.insert import DEFAULT_GENERATED_FRAMES
from sashimono.core.model import Clip, Project, Track, TrackKind
from sashimono.effects.sources import TEXT
from sashimono.engine.render.look_preview import SAMPLE_TEXT, SampleError, SampleSource

__all__ = ["file_fingerprint", "shelf_look"]

#: ファイルの中身の指紋 （道, 大きさ, 更新時刻）ごとに 1 度だけ読む
_DIGESTS: dict[tuple[str, int, int], str] = {}


def file_fingerprint(path: Path) -> str:
    """ファイルの中身の指紋 読めなければ場所から作る（読めない事実も見本に出すため）"""
    try:
        stat = path.stat()
    except OSError:
        return hashlib.sha256(f"missing|{path}".encode()).hexdigest()
    memo = (str(path), stat.st_size, stat.st_mtime_ns)
    found = _DIGESTS.get(memo)
    if found is not None:
        return found
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = hashlib.sha256(f"unreadable|{path}".encode()).hexdigest()
    _DIGESTS[memo] = digest
    return digest


def shelf_look(entry: TemplateEntry) -> SampleSource:
    """棚の 1 本を見本の絵の係へ渡す形にする"""
    text = f"shelf|{entry.source}|{entry.index}|{file_fingerprint(entry.path)}"
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def build() -> tuple[Project, int]:
        return _placed(entry)

    return SampleSource(fingerprint=fingerprint, build=build)


def _placed(entry: TemplateEntry) -> tuple[Project, int]:
    if entry.error:
        raise SampleError(entry.error)
    try:
        # 注意書きは見本では使わない 棚の窓の報告と混ざらないよう、自前の入れ物へ
        objects = entry.load(report=CompatibilityReport())
    except (*TemplateError, OSError, ValueError) as exc:
        raise SampleError(str(exc)) from exc
    project = Project.create()
    if objects and not any(item.has_picture for item in objects):
        # エフェクトだけのテンプレート 見本の文字に着せる（着せる操作と同じ道）
        clip = with_fixed_items(
            Clip(
                timeline_start=0,
                duration=DEFAULT_GENERATED_FRAMES,
                source=TEXT.create(text=SAMPLE_TEXT),
            ),
            picture=True,
        )
        track = Track(TrackKind.VIDEO, "V1", (clip,))
        project = project.with_timeline(replace(project.timeline, tracks=(track,)))
        for command in restyle(objects, clip):
            project = command.apply(project)
        return project, clip.duration // 2
    for command in place(objects, project, at_frame=0):
        project = command.apply(project)
    if project.duration <= 0:
        raise SampleError("置けるオブジェクトがありません")
    return project, project.duration // 2
