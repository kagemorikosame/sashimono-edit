"""``.exo`` の中身をこちらのモデルへ写す

値が写っているかだけでなく、**写せなかったものが記録に残るか**も見る
黙って捨てると「なんとなく違う絵」が出て、原因を追えない
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.compat.aviutl.encoding import encode_utf16_hex
from sashimono.compat.aviutl.exo import parse_exo
from sashimono.compat.aviutl.mapping import (
    SILENT_SOUND,
    MappedObject,
    map_exo,
    map_object,
    media_paths,
)
from sashimono.compat.aviutl.report import CompatibilityReport
from sashimono.core.commands import AddClip, AddMedia
from sashimono.core.model import (
    AudioStreamInfo,
    MediaId,
    MediaItem,
    Project,
    ProjectSettings,
    TrackKind,
    VideoStreamInfo,
)
from sashimono.core.timebase import FrameRate

RATE = FrameRate(30)


def build(*entries: str, start: int = 1, end: int = 60, layer: int = 1) -> str:
    body = [f"[0]\nstart={start}\nend={end}\nlayer={layer}\n"]
    body.extend(f"[0.{index}]\n{entry}\n" for index, entry in enumerate(entries))
    return "".join(body)


def one(text: str) -> MappedObject:
    mapped = map_object(parse_exo(text).objects[0], RATE, report=CompatibilityReport())
    assert mapped is not None
    return mapped


class TestText:
    def test_a_text_object_becomes_a_text_source(self) -> None:
        # 写し損ねると、AviUtl1 の .exo から読み込んだテキストの本文・大きさ・色が元と違って出る
        hexed = encode_utf16_hex("こんにちは", length=8)
        mapped = one(build(f"_name=テキスト\nサイズ=64\ncolor=ff0000\ntext={hexed}"))
        source = mapped.clip.source
        assert source is not None
        assert source.kind == "text"
        assert source.params["text"] == "こんにちは"
        assert float(source.params["size"].at(0)) == 64.0  # type: ignore[union-attr]
        assert source.params["color"] == (1.0, 0.0, 0.0, 1.0)

    def test_the_font_is_carried_over(self) -> None:
        # font を落とすと、読み込んだ字幕が元の書体ではなく既定の書体で描かれる
        mapped = one(build("_name=テキスト\nfont=Meiryo\nサイズ=32"))
        source = mapped.clip.source
        assert source is not None
        assert source.params["font"] == "Meiryo"

    def test_bold_and_alignment(self) -> None:
        # B と align を落とすと、読み込んだ太字が細字になり、左揃えの文字が中央揃えでずれて出る
        mapped = one(build("_name=テキスト\nB=1\nalign=1"))
        source = mapped.clip.source
        assert source is not None
        assert source.params["bold"] == 1
        assert source.params["align"] == "left"

    def test_aviutl1_text_keeps_the_native_layout(self) -> None:
        # AviUtl2 の組み方は AviUtl2 の書き出しで測った決まり AviUtl1 に当てると、
        # 測っていない決まりで今まで読めていた字幕の位置と効果の基準が動く
        mapped = one(build("_name=テキスト\nB=1\nalign=1"))
        source = mapped.clip.source
        assert source is not None
        assert source.params["layout"] == "native"


class TestFigure:
    def test_the_figure_number_picks_the_shape(self) -> None:
        # 0=円 1=四角形 2=三角形 3=五角形 4=六角形 5=星形
        for index, expected in enumerate(
            ["ellipse", "rect", "triangle", "pentagon", "hexagon", "star"]
        ):
            mapped = one(build(f"_name=図形\ntype={index}\nサイズ=100"))
            source = mapped.clip.source
            assert source is not None
            assert source.params["shape"] == expected

    def test_the_aspect_ratio_changes_the_size(self) -> None:
        # 縦横比 を読まないと、横を半分に縮めた四角が読み込み後に正方形で出る
        mapped = one(build("_name=図形\ntype=1\nサイズ=100\n縦横比=50"))
        source = mapped.clip.source
        assert source is not None
        assert float(source.params["width"].at(0)) == 50.0  # type: ignore[union-attr]
        assert float(source.params["height"].at(0)) == 100.0  # type: ignore[union-attr]


class TestDrawSettings:
    def test_position_and_scale_become_a_transform(self) -> None:
        # 標準描画 の位置と拡大を変形へ写せないと、読み込んだ物が画面の真ん中に元の大きさで出る
        mapped = one(
            build(
                "_name=図形\ntype=1\nサイズ=100",
                "_name=標準描画\nX=120.0\nY=-40.0\n拡大率=150.0\n回転=30.0\n透明度=0.0\nblend=0",
            )
        )
        effects = mapped.clip.effects
        assert effects[0].kind == "transform"
        assert float(effects[0].params["pos_x"].at(0)) == 120.0  # type: ignore[union-attr]
        # AviUtl の Y は下が正、こちらは上が正 符号が入れ替わる
        assert float(effects[0].params["pos_y"].at(0)) == 40.0  # type: ignore[union-attr]
        assert float(effects[0].params["scale"].at(0)) == 150.0  # type: ignore[union-attr]

    def test_transparency_becomes_opacity(self) -> None:
        # 透明度をそのまま不透明度に入れると、25% 透かしただけの図形が 4 分の 1 の濃さで出る
        mapped = one(build("_name=図形", "_name=標準描画\n透明度=25.0"))
        assert float(mapped.clip.opacity.at(0)) == 0.75

    def test_the_blend_number_is_translated(self) -> None:
        # blend の番号を読み違えると、加算で光らせた物が通常の重ね方になり明るさが乗らない
        mapped = one(build("_name=図形", "_name=標準描画\nblend=1"))
        assert mapped.clip.blend_mode == "add"

    @pytest.mark.parametrize(("number", "mode"), [(5, "overlay"), (6, "lighten"), (7, "darken")])
    def test_the_newer_blend_numbers_are_translated(self, number: int, mode: str) -> None:
        # 通常へ落とすと、配布物の光や影の重ね方が消える
        mapped = one(build("_name=図形", f"_name=標準描画\nblend={number}"))
        assert mapped.clip.blend_mode == mode

    def test_a_blend_that_is_not_a_number_is_recorded(self) -> None:
        # 0 と読むと「通常」に見えて、読めなかったことが記録から漏れる
        report = CompatibilityReport()
        text = build("_name=図形", "_name=標準描画\nblend=foo")
        mapped = map_object(parse_exo(text).objects[0], RATE, report=report)
        assert mapped is not None
        assert mapped.clip.blend_mode == "normal"
        assert any("foo" in line for line in report.lines())

    @pytest.mark.parametrize(("raw", "mode"), [("1.0", "add"), ("1e0", "add"), ("1.5", "normal")])
    def test_a_whole_number_may_be_written_as_a_decimal(self, raw: str, mode: str) -> None:
        # 1.0 を読めないと加算が通常になる 1.5 を 1 と読むと、読めなかったことが隠れる
        mapped = one(build("_name=図形", f"_name=標準描画\nblend={raw}"))
        assert mapped.clip.blend_mode == mode

    def test_a_number_outside_the_table_falls_back_to_normal(self) -> None:
        # 似た別のもので代用すると、直したつもりの無い違いが出る
        mapped = one(build("_name=図形", "_name=標準描画\nblend=9"))
        assert mapped.clip.blend_mode == "normal"


class TestFilters:
    def test_known_filters_become_effects(self) -> None:
        # 知っているフィルタを写せないと読み込んだクリップから ぼかし が消え、
        # 範囲を落とすとぼけ方の強さが変わる
        mapped = one(build("_name=図形", "_name=ぼかし\n範囲=20", "_name=標準描画"))
        effects = mapped.clip.effects
        assert [effect.kind for effect in effects] == ["blur"]
        assert float(effects[0].params["radius"].at(0)) == 20.0  # type: ignore[union-attr]

    def test_unknown_filters_are_recorded(self) -> None:
        # 知らないフィルタを記録しないと、効果が欠けたことが互換性レポートに出ず、
        # 埋める順を数えられない
        report = CompatibilityReport()
        mapped = map_object(
            parse_exo(build("_name=図形", "_name=まだ無いフィルタ\n値=1")).objects[0],
            RATE,
            report=report,
        )
        assert mapped is not None
        assert mapped.clip.effects == ()
        assert any("まだ無いフィルタ" in line for line in report.lines())

    def test_unknown_content_is_recorded(self) -> None:
        # 知らない中身を黙って捨てると、読み込んだ数が減った理由が互換性レポートに出ない
        report = CompatibilityReport()
        map_object(parse_exo(build("_name=謎のオブジェクト")).objects[0], RATE, report=report)
        assert any("謎のオブジェクト" in line for line in report.lines())


class TestTiming:
    def test_frames_are_converted_to_zero_based(self) -> None:
        # AviUtl は 1 始まりで終端を含む 1..60 は 0 から 60 フレーム
        mapped = one(build("_name=図形", start=1, end=60))
        assert (mapped.clip.timeline_start, mapped.clip.duration) == (0, 60)

    def test_a_later_object(self) -> None:
        # 開始を 1 始まりのまま置くと、31 フレーム目から始まる物が読み込み後に 1 フレーム遅れて出る
        mapped = one(build("_name=図形", start=31, end=90))
        assert (mapped.clip.timeline_start, mapped.clip.duration) == (30, 60)


class TestWholeFile:
    def test_layers_become_tracks(self) -> None:
        # レイヤーの番号をトラックへ写せないと、レイヤー 1 と 3 の重なり順が AviUtl と変わる
        text = (
            "[exedit]\nwidth=1920\nheight=1080\nrate=30\nscale=1\n"
            + build("_name=図形\ntype=1", layer=1)
            + "[1]\nstart=1\nend=30\nlayer=3\n[1.0]\n_name=テキスト\nサイズ=40\n"
        )
        project = Project.create(ProjectSettings(frame_rate=RATE))
        commands = map_exo(parse_exo(text), project, report=CompatibilityReport())

        applied = project
        for command in commands:
            applied = command.apply(applied)
        assert len(applied.timeline.tracks) == 3
        assert len(applied.timeline.tracks[0].clips) == 1
        assert len(applied.timeline.tracks[2].clips) == 1

    def test_objects_can_be_dropped_at_a_position(self) -> None:
        # at_frame を足し忘れると、呼び手が置く位置を指定しても頭（0 フレーム）に置かれる
        project = Project.create(ProjectSettings(frame_rate=RATE))
        commands = map_exo(
            parse_exo(build("_name=図形\ntype=1")),
            project,
            at_frame=90,
            report=CompatibilityReport(),
        )
        applied = project
        for command in commands:
            applied = command.apply(applied)
        assert applied.timeline.tracks[0].clips[0].timeline_start == 90

    def test_a_file_with_nothing_mappable_gives_no_commands(self) -> None:
        # 置ける物が無いのにコマンドを返すと、〔オブジェクトを読み込む…〕が
        # 「読み込めるオブジェクトがありませんでした」を出せない
        project = Project.create(ProjectSettings(frame_rate=RATE))
        empty = "[exedit]\nwidth=1920\nheight=1080\n"
        assert map_exo(parse_exo(empty), project, report=CompatibilityReport()) == []


#: 素材のパス 区切りは逆斜線（AviUtl は Windows のパスをそのまま書く）
MEDIA_PATH = "C:\\素材\\素材.mp4"

#: AviUtl2 v2.1.6a が書く素材オブジェクトの項目 ``{path}`` 以外は本体が既定値で埋めたまま
#: 3 つとも道は ``ファイル=`` で、項目の並びと他の項目は種類ごとに違う
SECOND_GENERATION_MEDIA = {
    "動画ファイル": (
        "再生位置=0.000,0.000,再生範囲,0\n再生速度=100.00\nファイル={path}\n"
        "トラック=0\nループ再生=0\n音声付き=0\nYUV=\nfps調整=0\n"
    ),
    "画像ファイル": (
        "ファイル={path}\n表示番号=0,0,再生範囲,0\n再生速度=100.00\nループ再生=0\n連番ファイル=0\n"
    ),
    "音声ファイル": (
        "再生位置=0.000,10.000,再生範囲,0\n再生速度=100.00\nファイル={path}\n"
        "トラック=0\nループ再生=0\n"
    ),
}

MEDIA_NAMES = tuple(SECOND_GENERATION_MEDIA)


def second_generation(name: str, path: str) -> str:
    """AviUtl2 が書く形の素材オブジェクト"""
    body = SECOND_GENERATION_MEDIA[name].replace("{path}", path)
    return f"[Object]\nframe=0,59\n[Object.0]\neffect.name={name}\n{body}"


class TestMediaFiles:
    """素材のパスは AviUtl2 では ``ファイル=``、AviUtl1 では ``file=``"""

    @pytest.mark.parametrize("name", MEDIA_NAMES)
    def test_the_second_generation_path_is_listed(self, name: str) -> None:
        # ``file`` だけを見ていたので、AviUtl2 の素材は素材一覧に 1 本も載らなかった
        exo = parse_exo(second_generation(name, MEDIA_PATH))
        assert media_paths(exo) == (MEDIA_PATH,)

    @pytest.mark.parametrize("name", MEDIA_NAMES)
    def test_the_second_generation_clip_knows_its_file(self, name: str) -> None:
        # 読めないとクリップが素材と結び付かず、中身の無いクリップになる
        exo = parse_exo(second_generation(name, MEDIA_PATH))
        mapped = map_object(exo.objects[0], RATE, report=CompatibilityReport())
        assert mapped is not None
        assert mapped.media_path == MEDIA_PATH

    @pytest.mark.parametrize("name", MEDIA_NAMES)
    def test_the_first_generation_path_is_still_listed(self, name: str) -> None:
        # AviUtl1 の ``.exo`` は ``file=`` と書く こちらを落とすと古い作品の素材が消える
        exo = parse_exo(build(f"_name={name}\nfile={MEDIA_PATH}"))
        assert media_paths(exo) == (MEDIA_PATH,)

    def test_the_placed_clip_points_at_the_loaded_media(self) -> None:
        # 素材一覧に載っても、置いたクリップが同じパスで引けなければ絵が出ない
        exo = parse_exo(second_generation("動画ファイル", MEDIA_PATH))
        project = Project.create(ProjectSettings(frame_rate=RATE))
        media = MediaId("m1")
        commands = map_exo(exo, project, media={MEDIA_PATH: media}, report=CompatibilityReport())
        placed = [command.clip for command in commands if isinstance(command, AddClip)]
        assert [clip.media_id for clip in placed] == [media]


def _media_item(*, video: bool, audio_index: int | None) -> MediaItem:
    """``.exo`` から指す素材 映像は 0 番、音は ``audio_index`` 番"""
    return MediaItem(
        path=Path(MEDIA_PATH),
        duration=Fraction(2),
        video_streams=(
            VideoStreamInfo(
                index=0,
                width=64,
                height=64,
                frame_rate=RATE,
                time_base=Fraction(1, 30),
                codec="h264",
                pixel_format="yuv420p",
            ),
        )
        if video
        else (),
        audio_streams=()
        if audio_index is None
        else (
            AudioStreamInfo(
                index=audio_index,
                sample_rate=48000,
                channels=2,
                time_base=Fraction(1, 48000),
                codec="aac",
            ),
        ),
    )


def _import_exo(name: str, item: MediaItem, report: CompatibilityReport | None = None) -> Project:
    """UI の読み込みと同じ順で、素材の登録と配置を 1 つずつ当てる"""
    exo = parse_exo(second_generation(name, MEDIA_PATH))
    project = AddMedia(item).apply(Project.create(ProjectSettings(frame_rate=RATE)))
    commands = map_exo(
        exo,
        project,
        media={MEDIA_PATH: item.id},
        items=(item,),
        report=report if report is not None else CompatibilityReport(),
    )
    for command in commands:
        project = command.apply(project)
    return project


class TestSoundFiles:
    """分ける方式で ``.exo`` の音声ファイルを置く先"""

    def test_a_sound_only_file_goes_on_an_audio_track(self) -> None:
        """音だけの素材を指す音声ファイルは音声トラックへ置く

        映像トラックへ置くと ``AddClip`` が断り、``.exo`` の読み込み全体が失敗する
        """
        project = _import_exo("音声ファイル", _media_item(video=False, audio_index=0))

        (track,) = project.timeline.tracks
        assert track.kind is TrackKind.AUDIO
        assert len(track.clips) == 1

    def test_a_sound_file_pointing_at_a_video_plays_its_sound(self) -> None:
        """動画を指す音声ファイル（AviUtl が動画の音を書く形）は音声トラックで鳴らす

        映像トラックへ置くと動画がもう 1 枚描かれ、音は鳴らない 音のストリームの番号を
        持たせないと 0 番（映像）を音として読みに行く
        """
        project = _import_exo("音声ファイル", _media_item(video=True, audio_index=1))

        (track,) = project.timeline.tracks
        assert track.kind is TrackKind.AUDIO
        (clip,) = track.clips
        assert clip.stream_index == 1
        # 音の欄だけを持つ 描画の欄があると、設定画面に効かない位置や反転が並ぶ
        assert [effect.kind for effect in clip.effects] == ["audio_volume", "audio_fade"]

    def test_a_sound_file_pointing_at_a_silent_video_is_left_out_and_counted(self) -> None:
        """音の無い素材を指す音声ファイルは置かず、互換性の記録に数える

        音声トラックへ置くと ``AddClip`` が断って読み込み全体が失敗し、映像トラックへ
        置くと動画が描かれる 黙って落とすと、読み込んだ数が合わない理由を追えない
        """
        report = CompatibilityReport()
        project = _import_exo("音声ファイル", _media_item(video=True, audio_index=None), report)

        assert all(not track.clips for track in project.timeline.tracks)
        assert report.missing[SILENT_SOUND] == 1

    def test_a_video_file_stays_on_a_video_track(self) -> None:
        # 音声ファイルを分けた道で、動画ファイルまで音声トラックへ持っていかない
        project = _import_exo("動画ファイル", _media_item(video=True, audio_index=1))

        (track,) = project.timeline.tracks
        assert track.kind is TrackKind.VIDEO
        assert len(track.clips) == 1
