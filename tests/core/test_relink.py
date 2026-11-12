"""見つからない素材の探し直しと、相対の場所の保存（#447）"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sashimono.core.commands import Document
from sashimono.core.io.recovery import RecoverySession
from sashimono.core.io.serialize import load_project, project_to_dict, save_project
from sashimono.core.model import MediaItem, PackedSpan, Project
from sashimono.core.relink import RelinkMedia, missing_media, moved_media, siblings


def _project(*paths: Path) -> Project:
    return Project.create(
        media=tuple(MediaItem(path=path, duration=Fraction(5)) for path in paths), name="移す作品"
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


class TestSiblings:
    def test_the_rest_in_the_same_folder_are_found_by_name(self, tmp_path: Path) -> None:
        """数十本を 1 本ずつ選ばせると、移した作品を直すのをあきらめる"""
        old = Path("D:/元の PC/素材")
        project = _project(old / "a.mp4", old / "b.wav", old / "c.png")
        new = tmp_path / "移した素材"
        _touch(new / "a.mp4")
        _touch(new / "b.wav")
        found = siblings(project, project.media[0].id, new / "a.mp4")
        assert found == {project.media[1].id: new / "b.wav"}

    def test_the_rest_in_other_folders_keep_their_layout(self, tmp_path: Path) -> None:
        """素材を種類ごとのフォルダに分けていると、同じフォルダだけ探しても見つからない"""
        old = Path("D:/作品/素材")
        project = _project(old / "映像" / "a.mp4", old / "音" / "b.wav")
        new = tmp_path / "移した" / "素材"
        _touch(new / "映像" / "a.mp4")
        _touch(new / "音" / "b.wav")
        found = siblings(project, project.media[0].id, new / "映像" / "a.mp4")
        assert found == {project.media[1].id: new / "音" / "b.wav"}

    def test_same_names_from_other_folders_are_not_merged(self, tmp_path: Path) -> None:
        """カメラごとのフォルダに同じ名前の撮影があると、名前だけで拾えば 2 本が同じ絵になる"""
        project = _project(Path("D:/撮影/カメラA/take.mp4"), Path("D:/撮影/カメラB/take.mp4"))
        _touch(tmp_path / "移した" / "take.mp4")
        assert siblings(project, project.media[0].id, tmp_path / "移した" / "take.mp4") == {}

    def test_the_layout_still_finds_the_other_camera(self, tmp_path: Path) -> None:
        """並びを保って移したなら、別のフォルダの同じ名前も正しい方を見つける"""
        project = _project(Path("D:/撮影/カメラA/take.mp4"), Path("D:/撮影/カメラB/take.mp4"))
        new = tmp_path / "撮影"
        _touch(new / "カメラA" / "take.mp4")
        _touch(new / "カメラB" / "take.mp4")
        found = siblings(project, project.media[0].id, new / "カメラA" / "take.mp4")
        assert found == {project.media[1].id: new / "カメラB" / "take.mp4"}

    def test_media_that_is_still_there_is_left_alone(self, tmp_path: Path) -> None:
        """見つかっている素材まで同じ名前の別の物へ付け替えると、正しい素材を失う"""
        here = _touch(tmp_path / "ここ" / "b.wav")
        project = _project(Path("D:/無い/a.mp4"), here)
        _touch(tmp_path / "移した" / "a.mp4")
        _touch(tmp_path / "移した" / "b.wav")
        assert siblings(project, project.media[0].id, tmp_path / "移した" / "a.mp4") == {}
        assert [item.id for item in missing_media(project)] == [project.media[0].id]


class TestRelinkCommand:
    def test_relinking_is_one_undo(self, tmp_path: Path) -> None:
        """付け替えが何段にも分かれると、間違えたときに 1 回で戻せない"""
        project = _project(Path("D:/無い/a.mp4"), Path("D:/無い/b.wav"))
        document = Document(project)
        a, b = (item.id for item in project.media)
        document.execute(RelinkMedia(((a, tmp_path / "a.mp4"), (b, tmp_path / "b.wav"))))
        assert [item.path for item in document.project.media] == [
            tmp_path / "a.mp4",
            tmp_path / "b.wav",
        ]
        document.undo()
        assert document.project is project

    def test_relinking_to_the_original_drops_the_packed_range(self, tmp_path: Path) -> None:
        """元の全体へ付け替えたのに範囲を残すと、範囲の外を断り続けて探し直した意味が無い"""
        item = MediaItem(
            path=Path("D:/展開/media/0001.mp4"),
            duration=Fraction(5),
            packed=(PackedSpan(Fraction(1), Fraction(2)),),
        )
        project = Project.create(media=(item,))
        moved = RelinkMedia(((item.id, tmp_path / "本編.mp4"),)).apply(project)
        assert moved.media[0].packed == ()
        # 字幕や長さなどの情報はそのまま（素材は同じ物）
        assert moved.media[0].duration == item.duration and moved.media[0].id == item.id

    def test_choosing_the_same_packed_file_keeps_its_range(self) -> None:
        """範囲だけのファイルを選び直しただけで範囲を外すと、無い所を読む編集が通る"""
        item = MediaItem(
            path=Path("D:/展開/media/0001.mp4"),
            duration=Fraction(5),
            packed=(PackedSpan(Fraction(1), Fraction(2)),),
        )
        project = Project.create(media=(item,))
        again = RelinkMedia(((item.id, item.path),)).apply(project)
        assert again is project

    @pytest.mark.skipif(sys.platform != "win32", reason="大小文字を同じと見るのは Windows だけ")
    def test_the_same_file_written_in_other_case_keeps_its_range(self) -> None:
        """Windows では大小文字だけ違う書き方も同じファイル 別と見ると範囲を外してしまう"""
        item = MediaItem(
            path=Path("D:/展開/media/0001.mp4"),
            duration=Fraction(5),
            packed=(PackedSpan(Fraction(1), Fraction(2)),),
        )
        project = Project.create(media=(item,))
        again = RelinkMedia(((item.id, Path("d:/展開/media/0001.mp4")),)).apply(project)
        assert again is project

    def test_moved_media_sees_relinking_and_its_undo(self, tmp_path: Path) -> None:
        """場所が変わったのを見落とすと、前のファイルのデコーダと波形を使い続ける"""
        project = _project(Path("D:/無い/a.mp4"), Path("D:/無い/b.wav"))
        a = project.media[0].id
        moved = RelinkMedia(((a, tmp_path / "a.mp4"),)).apply(project)
        assert moved_media(project, moved) == {a}
        assert moved_media(moved, project) == {a}
        assert moved_media(project, project) == set()


class TestRelativePlace:
    def test_a_folder_moved_as_a_whole_still_finds_its_media(self, tmp_path: Path) -> None:
        """フォルダごと別の場所や別の PC へ移すと、絶対の場所だけでは素材が見つからない"""
        before = tmp_path / "前"
        media = _touch(before / "素材" / "本編.mp4")
        save_project(_project(media), before / "作品.sme")
        after = tmp_path / "後"
        shutil.move(before, after)
        opened = load_project(after / "作品.sme")
        assert opened.media[0].path == (after / "素材" / "本編.mp4").resolve()
        assert missing_media(opened) == ()

    def test_the_saved_place_wins_unless_asked(self, tmp_path: Path) -> None:
        """作品のフォルダを写して別の作品にしたとき、どちらを開くかは人によって違う"""
        original = tmp_path / "元"
        media = _touch(original / "素材" / "本編.mp4")
        save_project(_project(media), original / "作品.sme")
        copy = tmp_path / "写し"
        shutil.copytree(original, copy)
        # 既定は前の版と同じ 保存したときの場所（元の素材）
        assert load_project(copy / "作品.sme").media[0].path == media
        relative = load_project(copy / "作品.sme", relative_first=True).media[0].path
        assert relative == (copy / "素材" / "本編.mp4").resolve()

    def test_a_missing_media_keeps_its_saved_place(self, tmp_path: Path) -> None:
        """どちらにも無いとき相対の場所を出すと、知らせの「元の場所」が元の PC の場所でなくなる"""
        save_project(_project(Path("D:/元の PC/本編.mp4")), tmp_path / "作品.sme")
        assert load_project(tmp_path / "作品.sme").media[0].path == Path("D:/元の PC/本編.mp4")

    def test_a_file_from_an_older_version_opens_as_before(self, tmp_path: Path) -> None:
        """前の版のファイル（相対の場所が無い）は保存した場所のまま開く 版は上げていない"""
        media = _touch(tmp_path / "素材" / "本編.mp4")
        data = project_to_dict(_project(media))
        assert "relative" not in data["media"][0]
        (tmp_path / "前の版.sme").write_text(json.dumps(data), encoding="utf-8")
        assert load_project(tmp_path / "前の版.sme").media[0].path == media

    def test_the_relative_place_uses_forward_slashes(self, tmp_path: Path) -> None:
        """区切りを Windows の形で書くと、ほかの OS で開いたときに 1 つの名前に読まれる"""
        media = _touch(tmp_path / "素材" / "下" / "本編.mp4")
        save_project(_project(media), tmp_path / "作品.sme")
        data = json.loads((tmp_path / "作品.sme").read_text(encoding="utf-8"))
        assert data["media"][0]["relative"] == "素材/下/本編.mp4"

    def test_the_recovery_copy_does_not_point_near_itself(self, tmp_path: Path) -> None:
        """退避のフォルダから数えた場所は作品と関係が無く、別の物を指しうる"""
        media = _touch(tmp_path / "素材" / "本編.mp4")
        store = RecoverySession(tmp_path / "退避")
        try:
            store.save(replace(_project(media), name="退避の試験"), None)
            data = json.loads(store.path.read_text(encoding="utf-8"))
        finally:
            store.close()
        assert "relative" not in data["media"][0]
