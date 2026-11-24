"""プレビューの 1 コマを編集の中で使う（#525）

止め絵には 2 通りある

- 〔このコマで止める〕 映像のクリップを再生ヘッドで割り、後ろ側をそのコマの絵で止める
  ファイルを作らない（:attr:`~sashimono.core.model.Clip.hold_at` を使う） 止まるのはそのクリップの
  絵だけで、上に重ねたテキストや下の絵は動き続ける
- 〔静止画を素材にする〕 重ねた絵ごと 1 枚の PNG にして、作品のフォルダの下の
  :data:`STILLS_FOLDER` へ書き、素材として読み込む 絵を描くのは画面の側（Qt が要る）

前は〔静止画を保存…〕で書き出してから読み込み直すしかなく、止め絵 1 枚に窓を 2 回くぐっていた
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from sashimono.core.commands.base import Command
from sashimono.core.commands.edit import SplitClips
from sashimono.core.model import ClipId, Project
from sashimono.core.timebase import FrameRate, format_timecode

__all__ = [
    "MAX_STILL_FILES",
    "STILLS_FOLDER",
    "FreezeFrame",
    "still_file_name",
]

#: 素材にした静止画を書く、作品のフォルダの下のフォルダ ナレーション（``narration``）と
#: 同じ考え方で、作品と一緒に持ち運べ、持ち出し用パッケージにも入る アプリのキャッシュへ
#: 書くと、作品を別の PC へ移したときに素材が見つからなくなる
STILLS_FOLDER = "stills"

#: 同じ名前のときに足す番号の上限 1 つのコマからこれを超えて作ることは無い
MAX_STILL_FILES = 999

#: Windows のファイル名に使えない文字 タイムコードの ``:`` と ``;`` もここに入る
_UNSAFE = re.compile(r'[\\/:*?"<>|;\x00-\x1f]')


def still_file_name(
    project_name: str, frame: int, rate: FrameRate, taken: Callable[[str], bool]
) -> str:
    """書く PNG の名前 ``<作品名>_<タイムコード>.png`` 使われていれば ``_2`` から番号を足す

    〔静止画を保存…〕の既定の名前と同じ形にする 前の物を上書きしないのは、同じコマを
    2 回素材にしたとき、先に置いた静止画の絵が後から黙って変わらないようにするため
    番号を使い切ったら :class:`ValueError`
    """
    stem = _UNSAFE.sub("-", f"{project_name}_{format_timecode(frame, rate)}").strip(" .")
    stem = stem or "still"
    for number in range(1, MAX_STILL_FILES + 1):
        name = f"{stem}.png" if number == 1 else f"{stem}_{number}.png"
        if not taken(name):
            return name
    raise ValueError(f"{stem} の静止画が多すぎる（{STILLS_FOLDER} を片付けてください）")


@dataclass(frozen=True, slots=True)
class FreezeFrame(Command):
    """映像のクリップを ``frame`` で割り、後ろ側をそのコマの絵で止める（〔このコマで止める〕）

    クリップの頭なら割らずにクリップ全体を止める リンクした音は一緒に割るが止めない
    （止めた絵の間も声や BGM は流れてほしい 音を止めると同じ音が鳴り続ける）
    """

    clip_id: ClipId
    frame: int

    @property
    def label(self) -> str:
        return "このコマで止める"

    def apply(self, project: Project) -> Project:
        located = project.timeline.locate_clip(self.clip_id)
        if located is None:
            raise KeyError(f"クリップが見つからない: {self.clip_id}")
        track, clip = located
        media = None if clip.media_id is None else project.find_media(clip.media_id)
        if (
            media is None
            or not media.has_video
            or media.is_still
            or not project.draws_picture(track, clip)
        ):
            raise ValueError("止められるのは動画の絵を描くクリップだけ")
        # 頭で止めるときは割らないので、割る命令のロックの確かめを通らない 先に確かめないと、
        # 同じ操作がロックしたトラックで再生ヘッドの位置によって通ったり断られたりする
        if track.locked:
            raise ValueError(f"トラック {track.name or track.id} はロックしているので止められない")
        if not clip.timeline_start <= self.frame < clip.timeline_end:
            raise ValueError(
                f"止める位置がクリップの内側にない: {self.frame} は "
                f"[{clip.timeline_start}, {clip.timeline_end}) の外"
            )
        # 割る前のクリップで時刻を決める 前から止めていれば、その時刻を越えない
        held = clip.picture_time(self.frame - clip.timeline_start, project.rate)
        if self.frame == clip.timeline_start:
            result, target = project, clip.id
        else:
            result = SplitClips((clip.id,), self.frame).apply(project)
            after = result.timeline.find_track(track.id)
            pieces = after.clips if after is not None else ()
            target = next(c.id for c in pieces if c.timeline_start == self.frame)
        found = result.timeline.locate_clip(target)
        if found is None:
            raise KeyError(f"割った後ろのクリップが見つからない: {target}")
        piece_track, piece = found
        clips = tuple(
            replace(c, hold_at=held) if c.id == piece.id else c for c in piece_track.clips
        )
        return result.with_timeline(result.timeline.replace_track(piece_track.with_clips(clips)))
