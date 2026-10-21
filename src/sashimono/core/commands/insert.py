"""素材をタイムラインへ入れる、という一連の操作

「読み込んで置く」は 1 つのコマンドではなく、素材の登録・トラックの用意・
映像と音声のクリップ配置・両者のリンク、の組み合わせになる UI からも AI からも
同じ手順を踏みたいので、ここに置いてコマンドの列として返す

コマンドを返すだけで実行はしない 呼び出し側がチェックポイントで括れば、
まとめて 1 回の Undo で戻せる
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from sashimono.core.commands.base import Command
from sashimono.core.commands.edit import AddClip, AddMedia, AddTrack
from sashimono.core.commands.fixed import (
    FADE_EFFECT_KIND,
    VOLUME_EFFECT_KIND,
    fixed_effect,
    takes_picture_items,
    with_fixed_items,
)
from sashimono.core.commands.layers import (
    active_layers,
    free_layer,
    layer_after,
    media_placements,
    places_mixed,
    solo_for_new_track,
)
from sashimono.core.model import (
    FILTER_KIND,
    TRANSITION_KIND,
    Clip,
    Effect,
    GeneratedSource,
    MediaItem,
    Project,
    Track,
    TrackId,
    TrackKind,
    default_track_name,
    new_group_id,
)
from sashimono.core.timebase import FrameRate, Rounding, seconds_to_frame

__all__ = [
    "DEFAULT_GENERATED_FRAMES",
    "DEFAULT_STILL_FRAMES",
    "EFFECT_TRACK_PREFIX",
    "VOLUME_EFFECT_KIND",
    "default_volume_effect",
    "held_at_end",
    "insert_clip",
    "insert_filter",
    "insert_generated",
    "insert_media",
    "is_effect_track",
    "lacks_video_end",
    "new_track",
    "place_media",
    "with_known_video_end",
    "with_video_end",
]

#: 置く先のトラックを選ぶ関数 （種類, 置くクリップ, 積んでいるコマンド）→ トラック
_TrackPicker = Callable[[TrackKind, Clip, list[Command]], Track]

#: 静止画をタイムラインへ置くときの既定の長さ（フレーム）
#: 30fps で 5 秒 Premiere の既定と同じくらい
DEFAULT_STILL_FRAMES = 150

#: テキストや図形を置くときの既定の長さ（フレーム） 30fps で 5 秒
DEFAULT_GENERATED_FRAMES = 150

#: エフェクトトラック（:func:`new_track`）の名前の頭
EFFECT_TRACK_PREFIX = "FX"

_EFFECT_TRACK_NAME = re.compile(rf"{EFFECT_TRACK_PREFIX}\d+")


def insert_media(
    project: Project,
    media: MediaItem,
    *,
    at_frame: int | None = None,
    split_audio: bool = False,
) -> list[Command]:
    """素材をメディアプールへ入れ、タイムラインの末尾（または指定位置）へ置く

    映像と音声を持つ素材は、別々のトラックへ展開して同じリンクグループに入れる
    片方を動かせばもう片方も追従し、分割も同時に行われる
    混合の方式（:func:`~sashimono.core.commands.layers.places_mixed`）では 1 本の
    クリップにして、範囲の空いたレイヤーへ置く

    ``split_audio`` は映像と音声を分けて置くか（本人の設定 ``Preferences.media_split`` から
    渡す） 真なら音ごとに別のトラックへ置き、混合の方式でも音声が 1 本の動画を含めて、
    絵のレイヤーの次のレイヤーから音を 1 本ずつ並べる（Issue #27 の要望）
    偽なら 1 本目の音だけを置き、混合の方式では絵と音を 1 本のクリップにまとめる
    コアの既定は偽（前の版の置き方） 本人の設定の既定（分ける）は、置く入口（画面・AI）が
    設定を読んで必ず渡す 既定を真にすると、渡し忘れた入口と互換の読み込みや道具で、
    同じ動画が置き方によってレイヤーの数を変える
    """
    start = project.duration if at_frame is None else max(0, at_frame)
    return _place(
        project,
        media,
        start,
        lambda kind, clip, commands: _find_or_create(project, kind, clip, commands),
        split_audio=split_audio,
    )


def place_media(
    project: Project,
    media: Sequence[MediaItem],
    *,
    at_frame: int,
    track_id: TrackId | None = None,
    split_audio: bool = False,
) -> list[Command]:
    """素材を、落とした位置（フレームとトラック）から順に並べて置く

    タイムラインへのドラッグ＆ドロップのためのもの 何本かを落としたら、
    落とした位置から隙間なく後ろへ並べる（読み込みの並び順と同じ）

    トラックは、落としたトラックが種類に合って空いていればそこへ置く 合わない・
    ロックしている・その範囲に別のクリップがいるときは、同じ種類で空いている
    トラックへ回し、それも無ければ新しく作る 落とした所へ無理に置くと、
    重なりで断られて何も置かれない
    映像と音声を持つ素材は、落とした側の種類だけがそのトラックへ入り、
    もう片方は合う種類の空いたトラックへ入る
    混合の方式では落としたレイヤーが空いていれば絵がそこへ入る（まとめる設定なら 1 本のクリップ）
    映像と音声の置き方（``split_audio``）は :func:`insert_media` と同じ
    """
    commands: list[Command] = []
    cursor = max(0, at_frame)
    for item in media:
        placed = _place(
            project,
            item,
            cursor,
            _free_picker(project, cursor, track_id),
            split_audio=split_audio,
        )
        for command in placed:
            project = command.apply(project)
        commands.extend(placed)
        ends = [c.clip.timeline_end for c in placed if isinstance(c, AddClip)]
        cursor = max([cursor, *ends])
    return commands


def _place(
    project: Project,
    media: MediaItem,
    start: int,
    pick: _TrackPicker,
    *,
    split_audio: bool,
) -> list[Command]:
    """素材を 1 本、``start`` から置くコマンド トラックは ``pick`` が決める

    混合の方式で音を分けて置くときは、2 本目からの置き先を ``pick`` に尋ねず、
    前に置いたレイヤーの次から探す（:func:`~sashimono.core.commands.layers.layer_after`）
    ``pick`` は奥から空きを探すので、絵より奥の空いたレイヤーへ音が戻る
    """
    commands: list[Command] = []
    if project.find_media(media.id) is None:
        commands.append(AddMedia(media))

    duration = _timeline_duration(project, media)
    if duration <= 0:
        return commands

    streams = media.audio_streams if split_audio else media.audio_streams[:1]
    parts = (1 if media.has_video or media.is_still else 0) + len(streams)
    # 置くクリップが 2 本以上あるときだけリンクする 1 本しかないのにグループを
    # 付けると、あとで別の素材と誤って連動する余地を残すことになる
    group = new_group_id() if parts > 1 else None

    picture: Clip | None = None
    if media.has_video or media.is_still:
        picture = with_fixed_items(
            Clip(
                timeline_start=start,
                duration=duration,
                media_id=media.id,
                stream_index=media.video_streams[0].index if media.has_video else 0,
                link_group=group,
                # 拡大率 100% を素材の画素にする（YMM4・AviUtl と同じ） 画面に収めると、
                # 同じ拡大率でも素材の解像度ごとに大きさが変わり、YMM4 の値を写しても合わない
                native_size=True,
            ),
            picture=True,
        )
        # 長さは音と映像の長い方（コンテナの長さ）を切り上げてあるので、音の方が長い素材や
        # 端数で 1 コマ増えた素材は、絵のクリップが映像の終わりを越える 越えた所を最後の
        # 絵で止める 止めないと書き出した動画の終わりが黒くなる（#402） 映像の長さで
        # 切らないのは、音の終わり（話し声やBGMの尻）を黙って欠かさないため
        picture = replace(picture, hold_at=held_at_end(picture, media, project.rate))

    # 音ごとに作る 音量とフェードの欄は 2 本目以降の音にも付ける 付けないと、
    # 置いた直後に 2 本目の音だけ音量をいじれない
    sounds = [
        Clip(
            timeline_start=start,
            duration=duration,
            media_id=media.id,
            stream_index=stream.index,
            link_group=group,
            effects=(default_volume_effect(), fixed_effect(FADE_EFFECT_KIND)),
        )
        for stream in streams
    ]

    # 絵と音を 2 本に分けるか 1 本にまとめるかは方式で決まる 置く物（エフェクトなど）は
    # 上で方式を問わずに作り、分かれ道は置く所だけにする 方式ごとに作り分けると、
    # 置いたときに付ける物を片方の方式にだけ足し忘れる
    first_sound = sounds[0] if sounds else None
    previous: Track | None = None
    placements = media_placements(project, picture, first_sound, sounds[1:], split=split_audio)
    for kind, clip in placements:
        if kind is TrackKind.MIXED and previous is not None:
            track = layer_after(project, previous, clip.timeline_start, clip.timeline_end, commands)
        else:
            track = pick(kind, clip, commands)
        commands.append(AddClip(track.id, clip))
        previous = track

    return commands


def default_volume_effect() -> Effect:
    """素材を置いた音声のクリップに付ける、音を変えない音量調整

    置いた直後から設定パネルで音量を動かせるようにする 付いていないと、
    音量を下げたいだけでもエフェクトの一覧から探して足す手間が要った（Issue #27）

    固定の項目として付ける YMM4 の音声アイテムの音量と同じく、外せず、
    重ねて掛けたいときはふつうの音量調整を別に足す
    """
    return fixed_effect(VOLUME_EFFECT_KIND)


def lacks_video_end(media: MediaItem) -> bool:
    """映像の道の終わりを持たない登録済みの素材か（道の終わりを記録する前の版で登録した物）

    形式 4 までの作品は映像の終わりを読み捨てる（:mod:`sashimono.core.io.serialize`）
    """
    return (
        bool(media.video_streams) and not media.is_still and media.video_streams[0].end_time is None
    )


def with_video_end(media: MediaItem, fresh: MediaItem | None) -> MediaItem:
    """登録済みの素材へ、開き直して取った映像の道の終わりを添える

    絵を止める時刻（:func:`held_at_end`）を決めるためだけに使う 無いままだとコンテナの
    長さで見るしかなく、音の方が長い素材（映像 2 秒・音 3 秒）では止めるべきクリップを止めず、
    書き出した動画の終わりが黒くなる（#402）
    素材の ``id`` はそのまま残すので、クリップは登録済みの素材に結ばれる
    プロジェクトの素材は書き換えない 書き換えるなら元に戻せるコマンドを通す必要があり、
    置くだけで素材一覧が変わるのは本人の予想を外れる
    長さも開き直した物に替える 版 4 までに覚えた長さはコンテナの頭から数えてあり、
    映像より早く始まる音の前置きを含む 道の終わりが分からない素材は止める時刻を長さで
    決めるので、古い長さのままだと映像の終わりより後ろを基準にして、止めるべき
    クリップを止めない
    開けないとき（``fresh`` が ``None``）は元のまま使う
    """
    if fresh is None:
        return media
    end = fresh.video_streams[0].end_time if fresh.video_streams else None
    first, *rest = media.video_streams
    duration = fresh.duration if fresh.duration > 0 else media.duration
    return replace(media, duration=duration, video_streams=(replace(first, end_time=end), *rest))


def with_known_video_end(
    media: Sequence[MediaItem], probe: Callable[[Path], MediaItem | None]
) -> list[MediaItem]:
    """素材一覧の素材を置く前に、映像の終わりを持たない物だけ開き直して添える

    素材一覧から置き直す入口（一覧の「配置」・タイムラインへのドラッグ・AI の道具）が
    :func:`insert_media` :func:`place_media` の前に呼ぶ 古い作品の素材は映像の終わりを
    持たず、そのまま置くと音の方が長い素材を止めない（#404 の Codex の指摘）
    開き直すのはコマンドを作る前のここだけにする コマンドの中で開くと、取り消しや
    やり直しのたびに素材を読み直し、ファイルが消えていればやり直しが落ちる
    ``probe`` は呼び出し側が渡す（コア層はファイルを開かない） 開けなければ ``None`` を返す
    物にする 開けない素材も置けるよう、止めずに今までどおり置く
    """
    return [
        with_video_end(item, probe(item.path)) if lacks_video_end(item) else item for item in media
    ]


def held_at_end(clip: Clip, media: MediaItem, rate: FrameRate) -> Fraction | None:
    """素材の映像の終わりを越えて読むクリップの、最後の絵の時刻 越えなければ ``None``

    素材を置くとき（:func:`insert_media` :func:`place_media`）と互換の読み込みで、YMM4 の
    「最後の絵で止める」を写すときに使う 置くときのクリップの長さはコンテナの長さ（音と
    映像の長い方）なので、音の方が長い素材は映像の終わりを越える 止めないと、越えた所は
    デコーダが何も返さず、書き出すと動画の終わりが黒くなる（#402）

    越えないクリップに持たせないのは、止まらないのに設定画面へ「絵を止める」が出て、
    何を止めているのか分からなくなるため

    最後の絵の時刻は、映像の終わりの **time_base の 1 刻み手前** デコーダは「その時刻を
    越えない最後のフレーム」を返し（:meth:`~sashimono.engine.decode.VideoDecoder.frame_at`）、
    終わりちょうどでは何も返さない 最後のフレームの PTS は必ず終わりより 1 刻み以上前に
    あるので、フレームの間隔によらず最後のフレームが出る 平均のフレーム 1 つ分を引くと、
    可変フレームレートの素材で最後の間隔が平均より短いとき、1 つ前の絵で止まる

    映像の終わりは映像の道の長さ
    （:attr:`~sashimono.core.model.VideoStreamInfo.end_time`）で見る コンテナの
    長さ（:attr:`MediaItem.duration`）で見ると、音の方が長い素材で最後の映像フレームより
    後ろを指し、止めた後もデコーダが毎フレーム終わり付近へシークし直してデコードする
    道の終わりもコンテナの長さも、素材の頭（:func:`~sashimono.engine.decode.probe.media_origin`）
    から数えてあり、クリップの ``source_in``・YMM4 の ``ContentOffset``・デコーダが読む時刻と
    同じ数え方（Issue #123） 頭が 0 より後ろの素材（分割して書き出した物）でも、頭 5 秒・
    長さ 2 秒なら映像の終わりは 2 秒 PTS そのまま（7 秒）で数えると、デコーダの映像の
    終わりを越えた所で止めることになり、止めた所から何も映らない
    道の長さが分からない素材（古いプロジェクトに入っていた素材など）はコンテナの長さで見る

    長さの分からない素材（0 と読めた物）と静止画は止めない 静止画はもともと
    いつでも同じ絵で、長さの分からない素材はどこが最後か決められない
    """
    if media.is_still or media.duration <= 0 or not media.video_streams:
        return None
    # 映像のクリップが読むのは最初の映像ストリーム（素材を置くときの絵のクリップと同じ）
    stream = media.video_streams[0]
    end = media.duration if stream.end_time is None else stream.end_time
    if clip.source_out(rate) <= end:
        return None
    # 1 刻みがフレームより長い（壊れた time_base）ときはフレーム 1 つ分に抑える 大きく引くと
    # 最後より前の絵で止まる 0 以下の刻みは引いても終わりちょうどになり、何も映らない
    frame = stream.frame_rate.frame_duration
    tick = stream.time_base if 0 < stream.time_base < frame else frame
    return max(Fraction(0), end - tick)


def _timeline_duration(project: Project, media: MediaItem) -> int:
    """素材をタイムラインへ置いたときの長さ（フレーム）"""
    if media.is_still:
        return DEFAULT_STILL_FRAMES
    if media.duration <= 0:
        return 0
    # 切り上げる 切り捨てると素材の末尾が 1 フレーム欠ける
    return max(1, seconds_to_frame(Fraction(media.duration), project.rate, Rounding.CEIL))


def _find_or_create(
    project: Project, kind: TrackKind, clip: Clip, commands: list[Command]
) -> Track:
    """その種類のトラックを探し、無ければ作るコマンドを積んで返す

    すでに ``commands`` の中で作ったトラックも対象にする 映像と音声を続けて
    置くときに、同じ種類のトラックを 2 本作ってしまわないように

    レイヤー（混合）は範囲の空きを見て選ぶ（:func:`~sashimono.core.commands.layers.free_layer`）
    絵も音も 1 本に置くので、空きを見ずに 1 本目へ入れると、途中の位置へ置いたときに
    重なりで断られる
    """
    if kind is TrackKind.MIXED:
        return free_layer(
            project, clip.timeline_start, clip.timeline_end, commands, picture=clip.show_picture
        )
    taken = _pending_tracks(commands, clip.timeline_start, clip.timeline_end)
    for command in commands:
        if (
            isinstance(command, AddTrack)
            and command.track.kind is kind
            and command.track.id not in taken
        ):
            return command.track

    existing = [
        t for t in project.timeline.tracks if t.kind is kind and not t.locked and t.id not in taken
    ]
    if existing:
        return existing[0]

    return _new_track(project, kind, commands)


def _pending_tracks(commands: list[Command], start: int, end: int) -> set[TrackId]:
    """``commands`` の中で ``[start, end)`` にクリップを置くトラック

    音声が何本もある素材の音を 1 本ずつ置くとき、まだ当てていない 1 本目の音と同じ
    トラックを 2 本目にも選ぶと、重なりで断られて何も置かれない
    """
    return {c.track_id for c in commands if isinstance(c, AddClip) and c.clip.overlaps(start, end)}


def _free_picker(project: Project, start: int, preferred: TrackId | None) -> _TrackPicker:
    def pick(kind: TrackKind, clip: Clip, commands: list[Command]) -> Track:
        if kind is TrackKind.MIXED:
            return free_layer(
                project,
                start,
                start + clip.duration,
                commands,
                picture=clip.show_picture,
                preferred=preferred,
            )
        return _free_or_create(project, kind, start, clip.duration, commands, preferred)

    return pick


def _free_or_create(
    project: Project,
    kind: TrackKind,
    start: int,
    duration: int,
    commands: list[Command],
    preferred: TrackId | None,
) -> Track:
    """``[start, start + duration)`` が空いている ``kind`` のトラック 無ければ作る

    落としたトラック（``preferred``）を先に見る 並びの順だけで探すと、
    V3 へ落としたのに空いている V1 へ入り、落とした所と違う所に出る
    """
    taken = _pending_tracks(commands, start, start + duration)
    for command in commands:
        if (
            isinstance(command, AddTrack)
            and command.track.kind is kind
            and command.track.id not in taken
        ):
            return command.track

    candidates = [
        t for t in project.timeline.tracks if t.kind is kind and not t.locked and t.id not in taken
    ]
    candidates.sort(key=lambda track: track.id != preferred)
    for track in candidates:
        if not any(clip.overlaps(start, start + duration) for clip in track.clips):
            return track
    return _new_track(project, kind, commands)


def _new_track(project: Project, kind: TrackKind, commands: list[Command]) -> Track:
    prefix = "V" if kind is TrackKind.VIDEO else "A"
    # 同じ置き方の中で先に作ったトラックも数える 音声が何本もある素材で音声トラックを
    # 続けて作ると、数えないと A1 が 2 本並ぶ
    created = [c.track for c in commands if isinstance(c, AddTrack)]
    index = sum(1 for t in (*project.timeline.tracks, *created) if t.kind is kind) + 1
    taken = {t.name for t in created}
    name = _unused_name(project, prefix, index)
    while name in taken:
        index += 1
        name = _unused_name(project, prefix, index)
    track = Track(kind=kind, name=name)
    commands.append(AddTrack(track))
    return track


def new_track(project: Project, kind: TrackKind, *, effect: bool = False) -> AddTrack:
    """空のトラックを 1 本足すコマンド 映像は一番上、音声は一番下へ入る

    トラックの並びの末尾へ足す 映像は並びの後ろほど手前（上）に重なり、音声は後ろほど
    下に並ぶ（:meth:`~sashimono.ui.timeline.layout.TimelineLayout.bands`） 読み込みや
    フィルタが新しく作るトラックと同じ所なので、足したトラックの位置が操作ごとに変わらない

    ``effect`` はフィルタを置くための映像トラック（エフェクトトラック） 形式は変えず、
    名前（``FX1`` など）で見分ける 映像トラックと別の種類にすると、読み込み・書き出し・
    互換の読み込みまですべてが新しい種類を知る必要があり、古い版で開けなくなる
    一番上に入るので、置いたフィルタがほかの映像トラックすべてに掛かる

    ソロで絞っている種類なら、足すトラックにもソロを付ける 付けないと足した所で外れ、
    置いたクリップが出ない（:func:`_filter_track` と同じ決まり） レイヤー（混合）は絵の側の
    ソロを見る（:func:`~sashimono.core.commands.layers.solo_for_new_track`）

    レイヤーも末尾（一番手前 番号の一番大きいレイヤー）へ足す 混合の方式にはエフェクトの
    レイヤーを作らない（利用者の要望 分けない方式ではフィルタも普通のレイヤーへ置く）
    フィルタの置き先は :func:`~sashimono.core.commands.layers.free_layer` が普通のレイヤーから
    選ぶので、下の絵すべてに掛かる所へ入る
    """
    if effect and kind is not TrackKind.VIDEO:
        raise ValueError(
            "エフェクトトラックは映像トラックとして作る（レイヤーではフィルタも普通のレイヤーに置く）"
        )
    same = [t for t in project.timeline.tracks if t.kind is kind]
    if effect:
        name = _unused_name(
            project, EFFECT_TRACK_PREFIX, sum(1 for t in same if is_effect_track(t)) + 1
        )
    else:
        name = default_track_name(kind, len(same) + 1, {t.name for t in project.timeline.tracks})
    soloed = solo_for_new_track(project, kind)
    return AddTrack(Track(kind=kind, name=name, solo=soloed))


def _unused_name(project: Project, prefix: str, number: int) -> str:
    """``prefix`` と番号の、まだ使われていないトラック名 ``number`` から数える

    消したトラックの名前が残っていると、本数を数えただけでは同じ名前が 2 本並ぶ
    （V2 を消して V1 と V3 が残ると、もう 1 本の V3 ができる） 空いている番号まで進める
    """
    names = {t.name for t in project.timeline.tracks}
    while f"{prefix}{number}" in names:
        number += 1
    return f"{prefix}{number}"


def is_effect_track(track: Track) -> bool:
    """フィルタを置くために足した映像トラックか（名前が ``FX`` と番号）

    レイヤーは名前が ``FX`` と番号でも普通のレイヤーとして扱う 前の版で作ったエフェクトの
    レイヤーが残った作品や、分ける方式から変換した作品でも、混合の方式では名前で振る舞いを
    変えない（利用者の要望） 名前も中身も変えないので、分ける方式へ戻せばまた
    エフェクトトラックになる
    """
    return track.kind is TrackKind.VIDEO and _EFFECT_TRACK_NAME.fullmatch(track.name) is not None


def insert_generated(
    project: Project,
    source: GeneratedSource,
    *,
    at_frame: int | None = None,
    duration: int = DEFAULT_GENERATED_FRAMES,
    track_id: TrackId | None = None,
) -> list[Command]:
    """テキストや図形をタイムラインへ置く

    素材を持たないので、置く先は必ず映像トラック（混合の方式ではレイヤー） 既存の
    クリップと重ならないよう、指定位置に空きが無ければ新しいトラックを作る テロップは元の映像に
    重ねたいのが普通で、既存クリップを避けて後ろへ並べるのは意図と違う

    ``track_id`` は置きたいトラック（右クリックした所） 置けなければいつもの決まりで選ぶ
    """
    return insert_clip(
        project,
        Clip(timeline_start=0, duration=duration, source=source),
        at_frame=at_frame,
        track_id=track_id,
    )


def insert_filter(
    project: Project,
    *,
    at_frame: int | None = None,
    duration: int = DEFAULT_GENERATED_FRAMES,
    effects: tuple[Effect, ...] = (),
    track_id: TrackId | None = None,
) -> list[Command]:
    """フィルタのクリップ（:data:`~sashimono.core.model.FILTER_KIND`）を置く

    置き先の決め方は :func:`_filter_track` ``track_id`` は :func:`insert_generated` と同じ
    """
    return insert_clip(
        project,
        Clip(
            timeline_start=0,
            duration=duration,
            source=GeneratedSource(kind=FILTER_KIND),
            effects=effects,
        ),
        at_frame=at_frame,
        track_id=track_id,
    )


def insert_clip(
    project: Project,
    clip: Clip,
    *,
    at_frame: int | None = None,
    track_id: TrackId | None = None,
) -> list[Command]:
    """素材を持たないクリップ（生成オブジェクト・フィルタ・シーン）を、中身のまま置く

    位置だけを ``at_frame``（省けば末尾）へ移す 長さ・エフェクト・不透明度はそのまま
    保存したエイリアスも、テキストやフィルタと同じ決まりで置き先が決まる

    ``track_id`` を渡すと、そのトラックが映像かレイヤーでロックされておらず、範囲が空いていれば
    そこへ置く 右クリックした所へ置かないと、どのトラックに入ったのかを探すことになる
    置けないとき（音声のトラック・埋まっている所）は断らず、いつもの決まりで選ぶ
    """
    commands: list[Command] = []
    start = project.duration if at_frame is None else max(0, at_frame)
    placed = replace(clip, timeline_start=start)
    if takes_picture_items(placed):
        # 描画の欄（反転・配置）を持たせる 前の版で保存したエイリアスは欄を持たないので、
        # ここで足さないと、置いたクリップによってパネルの描画の組の中身が食い違う
        placed = with_fixed_items(placed, picture=True)
    track = _wanted_track(project, track_id, start, placed.timeline_end)
    if track is None and places_mixed(project):
        # テキスト・フィルタ・シーンはどれも絵を描く 範囲の絵より手前のレイヤーへ置く
        # フィルタもこれで下の絵すべてに掛かる（:func:`_filter_track` と同じ考え）
        track = free_layer(project, start, placed.timeline_end, commands, picture=True)
    if track is None:
        # 場面切り替えもフィルタと同じく下のトラックの絵にしか効かない（#332）
        # 下から空きを探すと、切れ目の絵より下の空いたトラックへ入り、置いたのに何も変わらない
        track = (
            _filter_track(project, start, placed.timeline_end, commands)
            if placed.is_filter or _covers_below(placed)
            else _free_video_track(project, start, placed.duration, commands)
        )
    commands.append(AddClip(track.id, placed))
    return commands


def _covers_below(clip: Clip) -> bool:
    """下のトラックの絵を切り替えるクリップ（場面切り替え）か"""
    return clip.source is not None and clip.source.kind == TRANSITION_KIND


def _wanted_track(project: Project, track_id: TrackId | None, start: int, end: int) -> Track | None:
    """頼まれたトラックへ ``[start, end)`` を置けるならそのトラック 置けなければ ``None``"""
    if track_id is None:
        return None
    track = project.timeline.find_track(track_id)
    # 絵を描くトラック（映像とレイヤー）なら方式を問わず受ける 右クリックしたのは本人で、
    # 方式を切り替えた途中のプロジェクトでも、選んだ所から外すと探すことになる
    if track is None or track.kind is TrackKind.AUDIO or track.locked:
        return None
    if track.kind is TrackKind.MIXED and track not in active_layers(project, picture=True):
        # ミュートやソロの外のレイヤーへ置くと、置いた直後から映らない 空いたレイヤーを
        # 探す側（:func:`~sashimono.core.commands.layers.free_layer`）と同じ決まりにする
        # 映像トラックは今までどおり受ける 分ける方式の置き方を変えないため
        return None
    if any(clip.overlaps(start, end) for clip in track.clips):
        return None
    return track


def _filter_track(project: Project, start: int, end: int, commands: list[Command]) -> Track:
    """フィルタを置くトラック 無ければ作るコマンドを ``commands`` へ積んで返す

    フィルタはそれより**下**のトラックにしか効かない テキストや図形と同じく下から
    空きを探すと、範囲にある絵より下へ入り、置いたのに何も変わらないことがある
    範囲に絵のある一番上のトラックより上で空いているトラックを使い、無ければ一番上に作る

    見るのは描かれるトラック（:meth:`~sashimono.core.model.Timeline.active_tracks`）だけ
    ミュートしたトラックやソロの外のトラックへ置くと、置いたのにプレビューにも書き出しにも
    効かない 絵の有無も、描かれないトラックの物は数えない（見えない絵より上に置く理由が無い）
    ソロで絞っている間に新しく作るトラックは、ソロを付けて作る 付けないと作った所で外れる
    """
    timeline = project.timeline
    video = list(timeline.video_tracks())
    drawn = {track.id for track in timeline.active_tracks(TrackKind.VIDEO)}
    top = max(
        (
            index
            for index, track in enumerate(video)
            if track.id in drawn and any(clip.overlaps(start, end) for clip in track.clips)
        ),
        default=-1,
    )
    track = next(
        (
            candidate
            for candidate in video[top + 1 :]
            if candidate.id in drawn
            and not candidate.locked
            and not any(clip.overlaps(start, end) for clip in candidate.clips)
        ),
        None,
    )
    if track is None:
        soloed = any(t.solo and not t.muted for t in video)
        name = _unused_name(project, "V", len(video) + 1)
        track = Track(kind=TrackKind.VIDEO, name=name, solo=soloed)
        # 末尾へ足す 映像トラックの重ね順は並びの順なので、末尾が一番上になる
        commands.append(AddTrack(track))
    return track


def _free_video_track(
    project: Project, start: int, duration: int, commands: list[Command]
) -> Track:
    """``[start, start + duration)`` が空いている映像トラックを探す 無ければ作る"""
    for track in project.timeline.video_tracks():
        if track.locked:
            continue
        if not any(clip.overlaps(start, start + duration) for clip in track.clips):
            return track

    index = sum(1 for t in project.timeline.tracks if t.kind is TrackKind.VIDEO) + 1
    track = Track(kind=TrackKind.VIDEO, name=_unused_name(project, "V", index))
    commands.append(AddTrack(track))
    return track
