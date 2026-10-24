"""編集コマンドと Undo 履歴"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from sashimono.core.commands import (
    AddClip,
    AddMedia,
    AddTrack,
    Document,
    MoveClip,
    RemoveClip,
    RemoveMedia,
    RemoveTrack,
    RenameProject,
    SetTranscript,
    SplitClip,
    TrimClip,
)
from sashimono.core.model import (
    Clip,
    GroupId,
    MediaItem,
    Project,
    Track,
    TrackKind,
    Transcript,
    new_group_id,
)
from tests.conftest import RATE_30, make_clip


def _only_track(project: Project) -> Track:
    return project.timeline.tracks[0]


class TestMediaCommands:
    def test_add_media(self, audio_media: MediaItem, project: Project) -> None:
        # 足すたびに元のプロジェクトまで書き換わると、履歴が持つ変更前の形も変わり、
        # Ctrl+Z を押しても足した素材が消えない
        updated = AddMedia(audio_media).apply(project)
        assert len(updated.media) == 2
        assert updated.find_media(audio_media.id) is not None
        # 元のプロジェクトは変わらない
        assert len(project.media) == 1

    def test_add_duplicate_media_fails(self, video_media: MediaItem, project: Project) -> None:
        # 同じ素材を 2 度登録できると、素材の一覧に同じ物が並び、クリップがどちらを指すか決まらない
        with pytest.raises(ValueError, match="すでに登録"):
            AddMedia(video_media).apply(project)

    def test_remove_media(self, audio_media: MediaItem, project: Project) -> None:
        # 外した素材が一覧に残ると、使わなくなった素材が保存したファイルにも残り続ける
        added = AddMedia(audio_media).apply(project)
        removed = RemoveMedia(audio_media.id).apply(added)
        assert removed.find_media(audio_media.id) is None

    def test_remove_media_in_use_fails(self, video_media: MediaItem, project: Project) -> None:
        # 置いてあるクリップの素材を外せると、そのクリップは無い素材を指し、
        # プレビューにも書き出しにも絵が出なくなる
        track = _only_track(project)
        placed = AddClip(track.id, make_clip(0, 30, video_media)).apply(project)
        with pytest.raises(ValueError, match="使われている"):
            RemoveMedia(video_media.id).apply(placed)

    def test_set_transcript(
        self, video_media: MediaItem, transcript: Transcript, project: Project
    ) -> None:
        # 起こした字幕が素材に付かないと、字幕起こしを待った結果が消え、字幕を使う編集が何もできない
        updated = SetTranscript(video_media.id, transcript).apply(project)
        stored = updated.require_media(video_media.id).transcript
        assert stored is not None
        assert len(stored) == 3


class TestTrackCommands:
    def test_add_track_at_index(self, project: Project) -> None:
        # 指した位置に入らないと、新しいトラックが別の重ね順に入り、上に足したつもりの絵が下に隠れる
        new_track = Track(kind=TrackKind.AUDIO, name="A1")
        updated = AddTrack(new_track, index=0).apply(project)
        assert updated.timeline.tracks[0].id == new_track.id
        assert len(updated.timeline.tracks) == 2

    def test_add_track_appends_by_default(self, project: Project) -> None:
        # 位置を指さずに足したトラックが末尾に来ないと、既にあるトラックの番号と重ね順がずれる
        new_track = Track(kind=TrackKind.AUDIO, name="A1")
        updated = AddTrack(new_track).apply(project)
        assert updated.timeline.tracks[-1].id == new_track.id

    def test_remove_track(self, project: Project) -> None:
        # 消したトラックが残ると、そのトラックのクリップがプレビューと書き出しに出続ける
        track = _only_track(project)
        updated = RemoveTrack(track.id).apply(project)
        assert updated.timeline.tracks == ()

    def test_remove_unknown_track_fails(self, project: Project) -> None:
        # 無いトラックを消す指示が黙って通ると、execute_all が成功を返し、
        # 消せなかった知らせがステータスバーに出ない
        from sashimono.core.model import TrackId

        with pytest.raises(KeyError):
            RemoveTrack(TrackId("存在しない")).apply(project)


class TestClipCommands:
    def test_add_clip(self, video_media: MediaItem, project: Project) -> None:
        # 置いたクリップがトラックに入らないと、素材をタイムラインへ落としても何も現れない
        track = _only_track(project)
        updated = AddClip(track.id, make_clip(0, 30, video_media)).apply(project)
        assert len(_only_track(updated).clips) == 1

    def test_add_overlapping_clip_fails(self, video_media: MediaItem, project: Project) -> None:
        # 同じトラックに重ねて置けると、同じ時刻にどちらの絵を出すかが決まらず、片方が隠れる
        track = _only_track(project)
        placed = AddClip(track.id, make_clip(0, 30, video_media)).apply(project)
        with pytest.raises(ValueError, match="重なっている"):
            AddClip(track.id, make_clip(15, 30, video_media)).apply(placed)

    def test_add_clip_to_locked_track_fails(self, video_media: MediaItem, project: Project) -> None:
        # ロックしたトラックに置けると、守っているはずのトラックの絵や音が知らないうちに変わる
        track = replace(_only_track(project), locked=True)
        locked = project.with_timeline(project.timeline.replace_track(track))
        with pytest.raises(ValueError, match="ロック"):
            AddClip(track.id, make_clip(0, 30, video_media)).apply(locked)

    def test_audio_only_media_rejected_on_video_track(
        self, audio_media: MediaItem, project: Project
    ) -> None:
        # 映像トラックに音声素材を置くと、再生時に無音の穴になる 置いた時点で弾く
        with_audio = AddMedia(audio_media).apply(project)
        track = _only_track(with_audio)
        with pytest.raises(ValueError, match="映像が無い"):
            AddClip(track.id, make_clip(0, 30, audio_media)).apply(with_audio)

    def test_remove_clip(self, video_media: MediaItem, project: Project) -> None:
        # 消したクリップがトラックに残ると、削除したはずの絵が書き出しに出続ける
        track = _only_track(project)
        clip = make_clip(0, 30, video_media)
        placed = AddClip(track.id, clip).apply(project)
        removed = RemoveClip(clip.id).apply(placed)
        assert _only_track(removed).clips == ()

    def test_ripple_delete_closes_the_gap(self, video_media: MediaItem, project: Project) -> None:
        # 詰める削除で後ろが前へ寄らないと、消した所が黒い間のまま書き出しに残る
        track = _only_track(project)
        first = make_clip(0, 30, video_media)
        second = make_clip(30, 30, video_media)
        third = make_clip(60, 30, video_media)
        placed = project
        for clip in (first, second, third):
            placed = AddClip(track.id, clip).apply(placed)

        rippled = RemoveClip(second.id, ripple=True).apply(placed)
        starts = [c.timeline_start for c in _only_track(rippled).clips]
        assert starts == [0, 30]

    def test_plain_delete_leaves_the_gap(self, video_media: MediaItem, project: Project) -> None:
        # ふつうの削除で後ろまで寄ると、間を残したいだけなのに後ろのクリップがほかのトラックとずれる
        track = _only_track(project)
        first = make_clip(0, 30, video_media)
        second = make_clip(30, 30, video_media)
        third = make_clip(60, 30, video_media)
        placed = project
        for clip in (first, second, third):
            placed = AddClip(track.id, clip).apply(placed)

        removed = RemoveClip(second.id).apply(placed)
        starts = [c.timeline_start for c in _only_track(removed).clips]
        assert starts == [0, 60]

    def test_move_clip_within_track(self, video_media: MediaItem, project: Project) -> None:
        # 同じトラックの中で動かしたときに元の位置のクリップが残ると、同じ場面が 2 度出る
        track = _only_track(project)
        clip = make_clip(0, 30, video_media)
        placed = AddClip(track.id, clip).apply(project)
        moved = MoveClip(clip.id, 90).apply(placed)
        clips = _only_track(moved).clips
        assert len(clips) == 1
        assert clips[0].timeline_start == 90

    def test_move_clip_to_another_track(self, video_media: MediaItem, project: Project) -> None:
        # 別のトラックへ運んだクリップが元のトラックにも残ると、同じ絵が 2 枚重なって出る
        second_track = Track(kind=TrackKind.VIDEO, name="V2")
        with_track = AddTrack(second_track).apply(project)
        clip = make_clip(0, 30, video_media)
        placed = AddClip(_only_track(with_track).id, clip).apply(with_track)

        moved = MoveClip(clip.id, 60, second_track.id).apply(placed)
        assert moved.timeline.tracks[0].clips == ()
        assert moved.timeline.tracks[1].clips[0].timeline_start == 60

    def test_move_to_negative_position_fails(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # タイムラインの頭より前へ置けると、0 フレームより前に出た部分が再生でも書き出しでも見えない
        track = _only_track(project)
        clip = make_clip(30, 30, video_media)
        placed = AddClip(track.id, clip).apply(project)
        with pytest.raises(ValueError, match="負"):
            MoveClip(clip.id, -1).apply(placed)


class TestSplit:
    def test_split_produces_two_clips(self, video_media: MediaItem, project: Project) -> None:
        # 割った長さがずれると、割った所の前後でコマが欠けるか同じコマが 2 度出る
        track = _only_track(project)
        clip = make_clip(0, 60, video_media)
        placed = AddClip(track.id, clip).apply(project)

        split = SplitClip(clip.id, 20).apply(placed)
        clips = _only_track(split).clips
        assert [(c.timeline_start, c.duration) for c in clips] == [(0, 20), (20, 40)]

    def test_split_advances_source_in(self, video_media: MediaItem, project: Project) -> None:
        # 後ろの片割れが素材の頭から再生し直すと、割っただけで同じ場面がもう 1 度流れる
        track = _only_track(project)
        clip = make_clip(0, 60, video_media, source_in=1)
        placed = AddClip(track.id, clip).apply(project)

        split = SplitClip(clip.id, 30).apply(placed)
        left, right = _only_track(split).clips
        # 左が 30 フレーム = 1 秒を消費したので、右は 1 + 1 = 2 秒から始まる
        assert left.source_in == Fraction(1)
        assert right.source_in == Fraction(2)

    def test_split_respects_speed(self, video_media: MediaItem, project: Project) -> None:
        # 速さを無視して素材の位置を数えると、倍速のクリップを割った所で場面が巻き戻る
        track = _only_track(project)
        clip = Clip(timeline_start=0, duration=60, media_id=video_media.id, speed=Fraction(2))
        placed = AddClip(track.id, clip).apply(project)

        split = SplitClip(clip.id, 30).apply(placed)
        _, right = _only_track(split).clips
        # 2 倍速なので 30 フレーム（1 秒）の再生で 2 秒分のソースを消費する
        assert right.source_in == Fraction(2)

    def test_left_keeps_id(self, video_media: MediaItem, project: Project) -> None:
        # タイムラインの選択はクリップの ID で覚えている 前の片割れまで ID が変わると、
        # 割っただけで選んでいたクリップが選択から外れる
        track = _only_track(project)
        clip = make_clip(0, 60, video_media)
        placed = AddClip(track.id, clip).apply(project)
        left, right = _only_track(SplitClip(clip.id, 30).apply(placed)).clips
        assert left.id == clip.id
        assert right.id != clip.id

    @pytest.mark.parametrize("frame", [0, 60, -5, 100])
    def test_split_outside_clip_fails(
        self, frame: int, video_media: MediaItem, project: Project
    ) -> None:
        # 端や外で割る指示をここで断らないと、長さ 0 以下の片割れを作る所で落ち、
        # 割る位置が悪いと分からない知らせが出る
        track = _only_track(project)
        clip = make_clip(0, 60, video_media)
        placed = AddClip(track.id, clip).apply(project)
        with pytest.raises(ValueError, match="分割位置"):
            SplitClip(clip.id, frame).apply(placed)

    def test_split_follows_linked_audio(self, video_media: MediaItem, project: Project) -> None:
        # 映像と音声がリンクしていれば、片方を割るともう片方も同じ位置で割れる
        group: GroupId = new_group_id()
        audio_track = Track(kind=TrackKind.AUDIO, name="A1")
        with_track = AddTrack(audio_track).apply(project)

        video_clip = replace(make_clip(0, 60, video_media), link_group=group)
        audio_clip = replace(make_clip(0, 60, video_media), link_group=group, stream_index=1)
        placed = AddClip(_only_track(with_track).id, video_clip).apply(with_track)
        placed = AddClip(audio_track.id, audio_clip).apply(placed)

        split = SplitClip(video_clip.id, 25).apply(placed)
        video_bounds = [(c.timeline_start, c.duration) for c in split.timeline.tracks[0].clips]
        audio_bounds = [(c.timeline_start, c.duration) for c in split.timeline.tracks[1].clips]
        assert video_bounds == audio_bounds == [(0, 25), (25, 35)]

    def test_split_halves_get_separate_link_groups(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 分割してできた左右が同じリンクグループに残ると、片方を削除したときに
        # もう片方まで消える 左右は別のグループにし、映像と音声の対応だけ保つ
        group: GroupId = new_group_id()
        audio_track = Track(kind=TrackKind.AUDIO, name="A1")
        with_track = AddTrack(audio_track).apply(project)

        video_clip = replace(make_clip(0, 60, video_media), link_group=group)
        audio_clip = replace(make_clip(0, 60, video_media), link_group=group, stream_index=1)
        placed = AddClip(_only_track(with_track).id, video_clip).apply(with_track)
        placed = AddClip(audio_track.id, audio_clip).apply(placed)

        split = SplitClip(video_clip.id, 25).apply(placed)
        video_left, video_right = split.timeline.tracks[0].clips
        audio_left, audio_right = split.timeline.tracks[1].clips

        assert video_left.link_group == audio_left.link_group == group
        assert video_right.link_group == audio_right.link_group
        assert video_right.link_group != group

    def test_deleting_one_half_keeps_the_other(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 割った前半を消したときに後半まで一緒に消えると、残したかった映像と音声が無くなる
        group: GroupId = new_group_id()
        audio_track = Track(kind=TrackKind.AUDIO, name="A1")
        with_track = AddTrack(audio_track).apply(project)

        video_clip = replace(make_clip(0, 60, video_media), link_group=group)
        audio_clip = replace(make_clip(0, 60, video_media), link_group=group, stream_index=1)
        placed = AddClip(_only_track(with_track).id, video_clip).apply(with_track)
        placed = AddClip(audio_track.id, audio_clip).apply(placed)

        split = SplitClip(video_clip.id, 25).apply(placed)
        head = split.timeline.tracks[0].clips[0]
        removed = RemoveClip(head.id, ripple=True).apply(split)

        # 映像・音声とも後半だけが残り、詰められて先頭に来る
        assert [(c.timeline_start, c.duration) for c in removed.timeline.tracks[0].clips] == [
            (0, 35)
        ]
        assert [(c.timeline_start, c.duration) for c in removed.timeline.tracks[1].clips] == [
            (0, 35)
        ]


class TestTrim:
    def test_trim_head_advances_source(self, video_media: MediaItem, project: Project) -> None:
        # 頭を削っても素材の頭から再生されると、削ったはずの場面が流れ、残したい後ろの場面が切れる
        track = _only_track(project)
        clip = make_clip(0, 60, video_media)
        placed = AddClip(track.id, clip).apply(project)

        trimmed = TrimClip(clip.id, head_delta=30).apply(placed)
        result = _only_track(trimmed).clips[0]
        assert result.timeline_start == 30
        assert result.duration == 30
        # 素材の中身は動かない 先頭を 1 秒分削っただけ
        assert result.source_in == Fraction(1)

    def test_trim_tail_only_changes_duration(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 終わりを削ったときに頭まで動くと、残したい場面の始まりの位置や中身がずれる
        track = _only_track(project)
        clip = make_clip(0, 60, video_media)
        placed = AddClip(track.id, clip).apply(project)

        trimmed = TrimClip(clip.id, tail_delta=-20).apply(placed)
        result = _only_track(trimmed).clips[0]
        assert (result.timeline_start, result.duration) == (0, 40)
        assert result.source_in == Fraction(0)

    def test_trim_to_nothing_fails(self, video_media: MediaItem, project: Project) -> None:
        # 削りすぎを長さが 0 以下になると分かる知らせで断らないと、なぜ端を動かせないのか伝わらない
        track = _only_track(project)
        clip = make_clip(0, 60, video_media)
        placed = AddClip(track.id, clip).apply(project)
        with pytest.raises(ValueError, match="0 以下"):
            TrimClip(clip.id, head_delta=60).apply(placed)

    def test_trim_before_media_start_fails(self, video_media: MediaItem, project: Project) -> None:
        # 素材の頭より前へ伸ばす指示を止まる理由を添えて断らないと、
        # 人も AI も伸ばせない訳が分からず 1 フレームずつ試し直す
        track = _only_track(project)
        clip = make_clip(30, 60, video_media)
        placed = AddClip(track.id, clip).apply(project)
        with pytest.raises(ValueError, match="素材の先頭より前"):
            TrimClip(clip.id, head_delta=-30).apply(placed)


class TestDocument:
    def test_execute_and_undo(self, audio_media: MediaItem, project: Project) -> None:
        # Ctrl+Z で戻らない・Ctrl+Y でやり直せないと、間違えた操作を手で直すしかなくなる
        document = Document(project)
        document.execute(AddMedia(audio_media))
        assert len(document.project.media) == 2
        assert document.can_undo

        document.undo()
        assert len(document.project.media) == 1
        assert not document.can_undo
        assert document.can_redo

        document.redo()
        assert len(document.project.media) == 2

    def test_failed_command_leaves_everything_untouched(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 断られた操作が作品や履歴を変えると、Ctrl+Z を押しても何も戻らない段が挟まる
        document = Document(project)
        before = document.project
        with pytest.raises(ValueError):
            document.execute(AddMedia(video_media))
        assert document.project is before
        assert not document.can_undo

    def test_new_command_clears_redo(self, audio_media: MediaItem, project: Project) -> None:
        # 取り消した後に別の編集をしてもやり直しが残ると、Ctrl+Y で古い形が被さり新しい編集が消える
        document = Document(project)
        document.execute(RenameProject("一回目"))
        document.undo()
        assert document.can_redo
        document.execute(AddMedia(audio_media))
        assert not document.can_redo

    def test_checkpoint_collapses_into_one_step(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # AI は 1 つの指示で何十回も編集する それが 1 回の取り消しで戻ることが要件
        document = Document(project)
        track = _only_track(project)
        with document.checkpoint("AI: 無音をカット"):
            for index in range(5):
                document.execute(AddClip(track.id, make_clip(index * 40, 30, video_media)))
        assert len(_only_track(document.project).clips) == 5
        assert document.history_labels == ("AI: 無音をカット",)

        document.undo()
        assert _only_track(document.project).clips == ()

    def test_nested_checkpoints_record_only_the_outer(
        self, audio_media: MediaItem, project: Project
    ) -> None:
        # 内側のまとめまで履歴に載ると、AI の 1 回の指示を戻すのに何度も取り消すことになる
        document = Document(project)
        with document.checkpoint("外側"):
            document.execute(RenameProject("A"))
            with document.checkpoint("内側"):
                document.execute(AddMedia(audio_media))
        assert document.history_labels == ("外側",)

    def test_empty_checkpoint_is_not_recorded(self, project: Project) -> None:
        # 「取り消しても何も起きない」段が履歴に挟まると操作感が悪い
        document = Document(project)
        with document.checkpoint("何もしない"):
            pass
        assert not document.can_undo

    def test_checkpoint_survives_an_exception(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 失敗した指示の前半（名前の変更）だけが作品に残ると、どの操作の結果か分からない形になる
        document = Document(project)
        with pytest.raises(ValueError), document.checkpoint("途中で失敗"):
            document.execute(RenameProject("途中"))
            document.execute(AddMedia(video_media))

        # 途中まで成功した分も戻る 何本かをまとめて動かして 1 本だけ失敗したとき、
        # 残りだけ動いた状態が 1 段として残ると、何が起きたのか分からない
        assert document.project is project
        assert not document.can_undo

    def test_a_failing_inner_checkpoint_keeps_the_outer_work(self, project: Project) -> None:
        # AI の 1 往復は外側のチェックポイント その中の 1 つのツールが失敗しても、
        # それまでのツールの結果まで消すと、AI が報告した内容と画面が食い違う
        document = Document(project)
        with document.checkpoint("AI"):
            document.execute(RenameProject("一つ目"))
            with pytest.raises(KeyError), document.checkpoint("二つ目"):
                document.execute(RenameProject("二つ目"))
                raise KeyError("失敗")
        assert document.project.name == "一つ目"
        assert document.history_labels == ("AI",)

    def test_subscribe_and_unsubscribe(self, project: Project) -> None:
        # 知らせが届かないと、取り消してもタイムラインやプレビューが古い絵のまま
        # 解除が効かないと、閉じた画面にも知らせが届き続ける
        document = Document(project)
        seen: list[str] = []
        unsubscribe = document.subscribe(lambda p: seen.append(p.name))

        document.execute(RenameProject("一回目"))
        document.undo()
        assert seen == ["一回目", project.name]

        unsubscribe()
        document.execute(RenameProject("二回目"))
        assert len(seen) == 2

    def test_history_limit_drops_oldest(self, project: Project) -> None:
        # 古い段を捨てないと、長く編集するほど作品の形を何百も抱えてメモリが膨らみ続ける
        document = Document(project, history_limit=3)
        for index in range(5):
            document.execute(RenameProject(f"名前{index}"))
        assert len(document.history_labels) == 3

    def test_reset_clears_history(self, audio_media: MediaItem, project: Project) -> None:
        # 別の作品を開いた後も前の履歴が残ると、Ctrl+Z で開く前の作品の形へ戻ってしまう
        document = Document(project)
        document.execute(AddMedia(audio_media))
        document.reset(project)
        assert not document.can_undo
        assert not document.can_redo

    def test_reset_counts_a_new_generation(self, audio_media: MediaItem, project: Project) -> None:
        # 数えないと、作品の外で作品に付けて持つ物（AI の作業の計画 #340）が、別の作品を開いた
        # 後も前の作品の物のまま残り、〔続ける〕で別の作品を手順どおりに変える
        # 編集では増えない（増えると、編集のたびに計画が閉じる）
        document = Document(project)
        start = document.generation
        document.execute(AddMedia(audio_media))
        assert document.generation == start
        document.reset(project)
        assert document.generation == start + 1

    def test_undo_on_empty_history_is_a_noop(self, project: Project) -> None:
        # 履歴が空のときの Ctrl+Z で落ちると、作品を開いた直後に押しただけでアプリが止まる
        document = Document(project)
        assert document.undo() is project
        assert document.redo() is project

    def test_undo_label(self, audio_media: MediaItem, project: Project) -> None:
        # 戻す操作の名前が違うと、編集メニューの「元に戻す: 〜」に何が戻るかが正しく出ない
        document = Document(project)
        document.execute(AddMedia(audio_media))
        assert document.undo_label == f"素材を追加: {audio_media.name}"


class TestFrameRateIsUnusedButExplicit:
    """conftest の RATE_30 がプロジェクト設定と一致していることの確認"""

    def test_project_rate(self, project: Project) -> None:
        # 試験の 30 fps が作品の設定とずれると、「30 フレーム = 1 秒」で書いた分割やトリムの試験が
        # 別の速さで数え、素材の位置のずれを見逃す
        assert project.rate == RATE_30


class TestLinkedMoveAndTrim:
    """リンクされた映像・音声は、移動もトリムも一緒に動くこと

    分割と削除だけが連動して移動とトリムが連動しないと、ドラッグした瞬間に
    音がずれる 連動の約束は全部の操作で同じでなければ意味がない
    """

    def _linked(self, video_media: MediaItem, project: Project) -> tuple[Project, Clip, Clip]:
        group: GroupId = new_group_id()
        audio_track = Track(kind=TrackKind.AUDIO, name="A1")
        with_track = AddTrack(audio_track).apply(project)

        video_clip = replace(make_clip(30, 60, video_media), link_group=group)
        audio_clip = replace(make_clip(30, 60, video_media), link_group=group, stream_index=1)
        placed = AddClip(_only_track(with_track).id, video_clip).apply(with_track)
        placed = AddClip(audio_track.id, audio_clip).apply(placed)
        return placed, video_clip, audio_clip

    def test_moving_the_video_moves_the_audio(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 映像だけ動いて音声が残ると、ドラッグした途端に口の動きと声がずれる
        placed, video_clip, _ = self._linked(video_media, project)
        moved = MoveClip(video_clip.id, 100).apply(placed)
        assert moved.timeline.tracks[0].clips[0].timeline_start == 100
        assert moved.timeline.tracks[1].clips[0].timeline_start == 100

    def test_the_partner_keeps_its_own_track(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 映像を別の映像トラックへ移しても、音声は音声トラックに残る
        placed, video_clip, _ = self._linked(video_media, project)
        second = Track(kind=TrackKind.VIDEO, name="V2")
        placed = AddTrack(second).apply(placed)

        moved = MoveClip(video_clip.id, 100, second.id).apply(placed)
        assert moved.timeline.find_track(second.id).clips[0].id == video_clip.id  # type: ignore[union-attr]
        assert moved.timeline.tracks[1].clips[0].timeline_start == 100
        assert moved.timeline.tracks[1].kind is TrackKind.AUDIO

    def test_a_move_that_pushes_the_partner_negative_fails(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 片方だけ動いて残りがずれる、という中途半端な結果を作らない
        placed, video_clip, _ = self._linked(video_media, project)
        second = Track(kind=TrackKind.VIDEO, name="V2")
        placed = AddTrack(second).apply(placed)
        shifted = MoveClip(video_clip.id, 0, second.id).apply(placed)
        assert shifted.timeline.tracks[1].clips[0].timeline_start == 0

    def test_trimming_the_head_trims_both(self, video_media: MediaItem, project: Project) -> None:
        # 頭を削ったときに音声が削れないと、映像より先に声が鳴り始めて口とずれる
        placed, video_clip, _ = self._linked(video_media, project)
        trimmed = TrimClip(video_clip.id, head_delta=10).apply(placed)
        video = trimmed.timeline.tracks[0].clips[0]
        audio = trimmed.timeline.tracks[1].clips[0]
        assert (video.timeline_start, video.duration) == (40, 50)
        assert (audio.timeline_start, audio.duration) == (40, 50)
        assert video.source_in == audio.source_in

    def test_trimming_the_tail_trims_both(self, video_media: MediaItem, project: Project) -> None:
        # 終わりを削っても音声が縮まないと、映像が終わった後も声だけが鳴り続ける
        placed, video_clip, _ = self._linked(video_media, project)
        trimmed = TrimClip(video_clip.id, tail_delta=-20).apply(placed)
        assert trimmed.timeline.tracks[0].clips[0].duration == 40
        assert trimmed.timeline.tracks[1].clips[0].duration == 40

    def test_an_impossible_trim_changes_nothing(
        self, video_media: MediaItem, project: Project
    ) -> None:
        # 断られたトリムが元の作品まで書き換えると、映像と音声の片方だけ縮んだ形が残る
        placed, video_clip, _ = self._linked(video_media, project)
        with pytest.raises(ValueError):
            TrimClip(video_clip.id, head_delta=999).apply(placed)
        # 例外を投げたので、元のプロジェクトはそのまま
        assert placed.timeline.tracks[0].clips[0].duration == 60

    def test_unlinked_clips_are_untouched(self, video_media: MediaItem, project: Project) -> None:
        # リンクしていないクリップまで一緒に動くと、1 本を動かしただけでほかのクリップの位置が変わる
        track = _only_track(project)
        first = make_clip(0, 30, video_media)
        second = make_clip(60, 30, video_media)
        placed = AddClip(track.id, first).apply(project)
        placed = AddClip(track.id, second).apply(placed)

        moved = MoveClip(first.id, 120).apply(placed)
        assert [c.timeline_start for c in moved.timeline.tracks[0].clips] == [60, 120]
