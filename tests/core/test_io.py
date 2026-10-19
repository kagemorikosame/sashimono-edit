"""プロジェクトファイルの読み書き"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands import SetTranscript
from sashimono.core.io import (
    FORMAT_NAME,
    FORMAT_VERSION,
    ProjectFileError,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from sashimono.core.model import (
    AnimatedValue,
    Clip,
    Effect,
    Interpolation,
    Keyframe,
    MediaItem,
    Project,
    ProjectSettings,
    Track,
    TrackKind,
    Transcript,
    TranscriptSegment,
)
from sashimono.core.timebase import FrameRate


class TestRoundTrip:
    def test_empty_project(self) -> None:
        # 空の作品が往復で変わると、新しく作って保存しただけの作品が開き直すと別の設定になる
        original = Project.create()
        assert project_from_dict(project_to_dict(original)) == original

    def test_rich_project(self, rich_project: Project) -> None:
        # 保存で落ちる項目があると、作り込んだエフェクトやキーフレームや字幕が、閉じて
        # 開き直しただけで消える
        assert project_from_dict(project_to_dict(rich_project)) == rich_project

    def test_survives_a_file(self, rich_project: Project, tmp_path: Path) -> None:
        # ファイルへ書いて読む所で崩れると、辞書の往復は通っても、実際に保存して開き直した
        # 作品が編集していた物と違う
        path = tmp_path / "配信回_07.sme"
        save_project(rich_project, path)
        assert load_project(path) == rich_project

    def test_repeated_save_load_is_stable(self, rich_project: Project, tmp_path: Path) -> None:
        # 分数を浮動小数で書き出していると、往復のたびに値が動く
        path = tmp_path / "p.sme"
        current = rich_project
        for _ in range(5):
            save_project(current, path)
            current = load_project(path)
        assert current == rich_project

    def test_a_keyframe_keeps_its_curve(self) -> None:
        # 曲線の名前を保存で落とすと、YMM4 から読んだ Back や Expo の動きが
        # 開き直しただけで 3 種の加減速に変わる
        value = AnimatedValue(
            keyframes=(
                Keyframe(frame=0, value=0.0, interpolation=Interpolation.EASE_OUT, curve="expo"),
                Keyframe(frame=30, value=1.0),
            )
        )
        effect = Effect(kind="blur", params={"radius": value})
        clip = Clip(timeline_start=0, duration=30, effects=(effect,))
        base = Project.create()
        track = Track(TrackKind.VIDEO, "V1", (clip,))
        original = base.with_timeline(replace(base.timeline, tracks=(track,)))
        data = project_to_dict(original)
        keyframes = data["timeline"]["tracks"][0]["clips"][0]["effects"][0]["params"]["radius"]
        assert keyframes["keyframes"][0]["curve"] == "expo"
        # 曲線の無いキーフレームには書かない 前の版で保存したものと同じ形のまま
        assert "curve" not in keyframes["keyframes"][1]
        assert project_from_dict(data) == original

    def test_a_mesh_grid_survives_a_round_trip(self) -> None:
        # 格子は点数と点を 1 本の並びで持つ 落とすと、YMM4 から読んだ 3x3 の歪みが
        # 開き直しただけで四隅の変形に戻る
        grid = (3.0, 3.0, *([0.0] * 16), 12.0, -8.0)
        effect = Effect(kind="mesh_deform", params={"grid": grid})
        clip = Clip(timeline_start=0, duration=30, effects=(effect,))
        base = Project.create()
        track = Track(TrackKind.VIDEO, "V1", (clip,))
        original = base.with_timeline(replace(base.timeline, tracks=(track,)))
        data = project_to_dict(original)
        assert data["timeline"]["tracks"][0]["clips"][0]["effects"][0]["params"]["grid"][:2] == [
            3.0,
            3.0,
        ]
        assert project_from_dict(data) == original

    def test_fractions_are_written_as_strings(self, tmp_path: Path) -> None:
        # 29.97 fps（30000/1001）を小数で書くと、読む側（_fraction_from_json）は小数を
        # 分数ではないと断るので、保存した作品が開けなくなる
        settings = ProjectSettings(frame_rate=FrameRate(30000, 1001))
        project = Project.create(settings)
        path = tmp_path / "ntsc.sme"
        save_project(project, path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["settings"]["frame_rate"] == "30000/1001"
        assert data["timeline"]["rate"] == "30000/1001"

    def test_japanese_is_not_escaped(self, rich_project: Project, tmp_path: Path) -> None:
        # 手で読める・手で直せることが JSON にした理由なので、ここは崩さない
        path = tmp_path / "p.sme"
        save_project(rich_project, path)
        assert "配信回_07" in path.read_text(encoding="utf-8")


class TestFileHandling:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        # 保存先のフォルダがまだ無いときに保存が失敗し、閉じる前の編集を残せない
        path = tmp_path / "深い" / "階層" / "p.sme"
        save_project(Project.create(), path)
        assert path.exists()

    def test_no_temporary_file_is_left_behind(self, tmp_path: Path) -> None:
        # 書きかけの仮のファイルが残ると、保存するたびに作品の横へ .writing のファイルが積もる
        path = tmp_path / "p.sme"
        save_project(Project.create(), path)
        assert [p.name for p in tmp_path.iterdir()] == ["p.sme"]

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        # 上書きの保存が前の中身を残すと、保存し直した編集が開き直したときに消えている
        path = tmp_path / "p.sme"
        save_project(Project.create(name="一回目"), path)
        save_project(Project.create(name="二回目"), path)
        assert load_project(path).name == "二回目"

    def test_untitled_project_takes_its_filename(self, tmp_path: Path) -> None:
        # 名前の無い作品にファイル名を使わないと、書き出しの窓が勧めるファイル名が
        # 保存したファイル名でなく「無題」になる
        path = tmp_path / "夏の思い出.sme"
        save_project(Project.create(), path)
        assert load_project(path).name == "夏の思い出"

    def test_missing_file(self, tmp_path: Path) -> None:
        # 消した・動かした作品を開こうとしたときに、何が悪いか分かる言葉の代わりに
        # ファイルの例外がそのまま出る
        with pytest.raises(ProjectFileError, match="開けない"):
            load_project(tmp_path / "無い.sme")

    def test_not_json(self, tmp_path: Path) -> None:
        # 壊れた保存や別のソフトのファイルを開いたときに、JSON の読み取りの例外がそのまま出て
        # 作品のファイルが壊れていると分からない
        path = tmp_path / "p.sme"
        path.write_text("これは JSON ではない", encoding="utf-8")
        with pytest.raises(ProjectFileError, match="JSON として読めない"):
            load_project(path)


class TestValidation:
    def test_rejects_foreign_format(self) -> None:
        # ほかのソフトの JSON を作品として読み進めると、項目が合わずに崩れた作品が開くか、
        # 読み取りの途中で分かりにくい例外になる
        with pytest.raises(ProjectFileError, match="プロジェクトファイルではない"):
            project_from_dict({"format": "aviutl", "version": 1})

    def test_rejects_newer_version(self) -> None:
        # 新しい版の作品を古い本体で開けると、知らない項目を捨てたまま開き、上書き保存で
        # 新しい版で付けた編集が消える
        with pytest.raises(ProjectFileError, match="新しい形式"):
            project_from_dict({"format": FORMAT_NAME, "version": FORMAT_VERSION + 1})

    def test_rejects_rate_mismatch(self, rich_project: Project) -> None:
        # 手で直して設定とタイムラインのフレームレートが食い違ったファイルを、どこが悪いか
        # 分かる言葉で断れず、開けない理由が分からない
        data = project_to_dict(rich_project)
        data["timeline"]["rate"] = "25/1"
        with pytest.raises(ProjectFileError, match="食い違っている"):
            project_from_dict(data)

    def test_rejects_unknown_track_kind(self, rich_project: Project) -> None:
        # 知らない種類のトラックを読み飛ばすと、新しい版で作った混合トラックが前の版で
        # 黙って消え、上書き保存で失われる
        data = project_to_dict(rich_project)
        data["timeline"]["tracks"][0]["kind"] = "subtitle"
        with pytest.raises(ProjectFileError, match="未知のトラック種別"):
            project_from_dict(data)

    def test_rejects_unknown_interpolation(self, rich_project: Project) -> None:
        # 名前を添えて断らないと、Enum の英語の文が「壊れた値がある」に包まれて出るだけで、
        # 新しい版で付けた補間を前の版で開いたとき、何が読めないのか分からない
        data = project_to_dict(rich_project)
        clip = data["timeline"]["tracks"][0]["clips"][0]
        clip["effects"][0]["params"]["radius"]["keyframes"][0]["interpolation"] = "魔法"
        with pytest.raises(ProjectFileError, match="未知の補間方法"):
            project_from_dict(data)

    def test_rejects_an_unknown_curve(self, rich_project: Project) -> None:
        # 通すと、描くときに Keyframe の検査で落ちる 読み込みの時点で弾く
        data = project_to_dict(rich_project)
        clip = data["timeline"]["tracks"][0]["clips"][0]
        clip["effects"][0]["params"]["radius"]["keyframes"][0]["curve"] = "魔法"
        with pytest.raises(ProjectFileError, match="未知の曲線"):
            project_from_dict(data)

    def test_rejects_a_curve_on_a_straight_point(self, rich_project: Project) -> None:
        # 直線の点に曲線を書いたファイルは、描くと曲線が効かない 壊れたことに気付けるよう弾く
        data = project_to_dict(rich_project)
        clip = data["timeline"]["tracks"][0]["clips"][0]
        keyframe = clip["effects"][0]["params"]["radius"]["keyframes"][0]
        keyframe["interpolation"] = "linear"
        keyframe["curve"] = "back"
        with pytest.raises(ProjectFileError, match="イージングの点にだけ"):
            project_from_dict(data)

    def test_rejects_a_non_numeric_keyframe_that_would_crash_the_audio_mix(
        self, rich_project: Project
    ) -> None:
        """キーフレームの値は**読み込みの時点で**数に限る

        ここを通すと、数でない値が AnimatedValue に入ったまま描画や音の計算へ
        流れ、math.isfinite の所で落ちる 水際で弾く
        """
        data = project_to_dict(rich_project)
        clip = data["timeline"]["tracks"][0]["clips"][0]
        clip["effects"][0]["params"]["radius"]["keyframes"][0]["value"] = "おおきめ"
        with pytest.raises(ProjectFileError, match="数値ではない"):
            project_from_dict(data)

    def test_rejects_a_non_finite_keyframe_that_would_break_the_gl_values(
        self, rich_project: Project
    ) -> None:
        # JSON の読み込みは NaN と Infinity を受けてしまう 通すと GL の値が壊れる
        data = project_to_dict(rich_project)
        clip = data["timeline"]["tracks"][0]["clips"][0]
        clip["effects"][0]["params"]["radius"]["keyframes"][0]["value"] = float("nan")
        with pytest.raises(ProjectFileError, match="有限の数ではない"):
            project_from_dict(data)

    def test_rejects_malformed_fraction(self, rich_project: Project) -> None:
        # 手で直してフレームレートを壊したファイルで、どの項目が悪いかを示せず、
        # 分数の読み取りの例外がそのまま出る
        data = project_to_dict(rich_project)
        data["settings"]["frame_rate"] = "さんじゅう"
        with pytest.raises(ProjectFileError, match="分数として読めない"):
            project_from_dict(data)

    def test_rejects_wrong_type(self, rich_project: Project) -> None:
        # 幅が文字のまま通ると、設定の検査で文字と数を比べる型の例外になり、開けない理由が
        # 分からない
        data = project_to_dict(rich_project)
        data["settings"]["width"] = "1920"
        with pytest.raises(ProjectFileError, match="整数ではない"):
            project_from_dict(data)


class TestCompactness:
    def test_static_values_do_not_write_keyframes(self, tmp_path: Path) -> None:
        # プロジェクトファイルの大半は静的な値 ここが冗長だとファイルが膨れる
        project = Project.create()
        clip = Clip(timeline_start=0, duration=10, opacity=AnimatedValue(0.5))
        track = Track(kind=TrackKind.VIDEO, clips=(clip,))
        project = project.with_timeline(replace(project.timeline, tracks=(track,)))

        data = project_to_dict(project)
        assert data["timeline"]["tracks"][0]["clips"][0]["opacity"] == {"static": 0.5}


class TestTranscriptRoundTrip:
    def test_words_and_speaker_survive(self, rich_project: Project) -> None:
        # 単語ごとの時刻を落とすと、開き直した後に字幕を割るとき単語の切れ目で分けられず、
        # 文字数の比で語の途中から割れる 話者も消える
        restored = project_from_dict(project_to_dict(rich_project))
        media = next(m for m in restored.media if m.transcript is not None)
        assert media.transcript is not None
        first = media.transcript.segments[0]
        assert first.speaker == "話者A"
        assert [w.text for w in first.words] == ["今日", "は"]

    def test_edited_flag_survives(self, project: Project, video_media: MediaItem) -> None:
        # 手で直した印を保存で落とすと、開き直した後の整理が「手で直した字幕には触らない」
        # でも直した文を書き換える
        transcript = Transcript(
            segments=(
                TranscriptSegment(start=Fraction(0), end=Fraction(1), text="編集済み", edited=True),
            )
        )
        source = SetTranscript(video_media.id, transcript).apply(project)
        restored = project_from_dict(project_to_dict(source))
        stored = restored.require_media(video_media.id).transcript
        assert stored is not None
        assert stored.segments[0].edited
