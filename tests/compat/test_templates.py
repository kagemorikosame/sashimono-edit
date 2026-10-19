"""テンプレートの棚と、置く／着せるの 2 通り

配布されている字幕エイリアスは見本の文字入りで配られる そのまま置くだけだと
毎回打ち直すことになるので、**今の文字を残して見た目だけ入れ替える**道を
用意してある ここではその境目を検査する
"""

from __future__ import annotations

import io
import json
import re
import zipfile
import zlib
from pathlib import Path

import pytest

from sashimono.compat.catalog import TemplateCatalog, TemplateError, place, restyle
from sashimono.compat.ymm4.template import Ymm4ParseError, load_template
from sashimono.core.commands import AddClip, AddEffect, AddTrack, RemoveEffect, SetSource
from sashimono.core.model import AnimatedValue, Clip, GeneratedSource, Project
from sashimono.effects import registry


def value_at(value: object, frame: int = 0) -> float:
    """数値パラメータの、その時刻での値

    :data:`~sashimono.core.model.ParamValue` は数値とは限らないので、
    数値であることをここで 1 度だけ確かめる
    """
    assert isinstance(value, AnimatedValue)
    return value.at(frame)


ALIAS = (
    "[Object]"
    + chr(10)
    + "frame=0,89"
    + chr(10)
    + "[Object.0]"
    + chr(10)
    + "effect.name=テキスト"
    + chr(10)
    + "サイズ=72.00"
    + chr(10)
    + "フォント=Dela Gothic One"
    + chr(10)
    + "文字色=ffee00"
    + chr(10)
    + "影・縁色=000000"
    + chr(10)
    + "文字装飾=縁取り文字（太）"
    + chr(10)
    + "文字揃え=中央揃え[下]"
    + chr(10)
    + "テキスト=見本の文字"
    + chr(10)
    + "[Object.1]"
    + chr(10)
    + "effect.name=縁取り"
    + chr(10)
    + "サイズ=4"
    + chr(10)
    + "縁色=ff0000"
    + chr(10)
)

#: 中間点を持つエイリアス 書き方は実物（AviUtl2 が書き出した ``.object``）から取った
#: ``frame=`` に始まり・中間点・終わりが並び、値もその数だけ並ぶ
MOVING = (
    "[Object]"
    + chr(10)
    + "frame=0,89,179"
    + chr(10)
    + "[Object.0]"
    + chr(10)
    + "effect.name=テキスト"
    + chr(10)
    + "サイズ=72.00"
    + chr(10)
    + "テキスト=見本の文字"
    + chr(10)
    + "[Object.1]"
    + chr(10)
    + "effect.name=標準描画"
    + chr(10)
    + "X=0.00,150.00,300.00,直線移動,0"
    + chr(10)
    + "Y=0.00,0.00,0.00,直線移動,0"
    + chr(10)
)

NO_SPAN = (
    "[Object]" + chr(10) + "[Object.0]" + chr(10) + "effect.name=テキスト" + chr(10) + "テキスト=あ"
)


def _corrupt_zip() -> bytes:
    """終わりの目録は正しく、中の圧縮された所だけが壊れた ZIP

    ZIP とは見分けられるので中を開きに行き、取り出すところで初めて失敗する
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("catalog.json", json.dumps({"ItemTemplates": []}) * 50)
    data = bytearray(buffer.getvalue())
    # ローカルの見出しは固定の 30 バイトの後ろに名前と拡張の欄が続き、その後ろが中身
    # 決め打ちの位置で壊すと名前まで壊れ、見出しの食い違い（BadZipFile）で先に落ちる
    # 0 で埋めると展開は通って CRC の食い違いになるので、展開できない値で埋める
    name_length = int.from_bytes(data[26:28], "little")
    extra_length = int.from_bytes(data[28:30], "little")
    start = 30 + name_length + extra_length
    data[start + 2 : start + 12] = b"\xff" * 10
    return bytes(data)


def _rewritten_zip(*, flags: int = 0, method: int | None = None) -> bytes:
    """見出しの汎用フラグと圧縮方式を書き換えた ZIP

    標準の ``zipfile`` は暗号化した物や知らない圧縮方式の物を書けないので、普通に
    書いてから、ローカルの見出しと中央の目録の両方を書き換える ``zipfile`` は
    目録の値で展開のしかたを決め、ローカルの見出しとの食い違いも見るので、
    片方だけでは別の理由（見出しの食い違い）で読めなくなり、狙った道を通らない
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("catalog.json", json.dumps({"ItemTemplates": []}))
    data = bytearray(buffer.getvalue())
    # ローカルの見出しは頭から 6 バイト目に汎用フラグ、8 バイト目に圧縮方式
    # 中央の目録はそれぞれ 8 バイト目と 10 バイト目
    for signature, flag_at in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = data.index(signature)
        flag = int.from_bytes(data[start + flag_at : start + flag_at + 2], "little") | flags
        data[start + flag_at : start + flag_at + 2] = flag.to_bytes(2, "little")
        if method is not None:
            data[start + flag_at + 2 : start + flag_at + 4] = method.to_bytes(2, "little")
    return bytes(data)


@pytest.fixture
def shelf(tmp_path: Path) -> tuple[TemplateCatalog, Path]:
    root = tmp_path / "テンプレート"
    (root / "字幕").mkdir(parents=True)
    (root / "字幕" / "強調.object").write_text(ALIAS, "utf-8")
    (root / "字幕" / "長さなし.object").write_text(NO_SPAN, "utf-8")
    (root / "字幕" / "動き.object").write_text(MOVING, "utf-8")
    (root / "読まない.txt").write_text("素通り", "utf-8")
    (root / "YMM4").mkdir()
    (root / "YMM4" / "見出し.ymmt").write_text(
        json.dumps(
            {
                "$type": "YukkuriMovieMaker.Project.Items.TextItem, YukkuriMovieMaker",
                "Text": "見出し",
                "FontSize": 80,
                "Length": 60,
            }
        ),
        "utf-8",
    )
    catalog = TemplateCatalog()
    catalog.scan((root,))
    return catalog, root


class TestScanning:
    def test_both_kinds_are_found(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # どちらかの拡張子を拾い損ねると、
        # テンプレートの窓に AviUtl のエイリアスか YMM4 の .ymmt の片方が並ばない
        catalog, _ = shelf
        sources = sorted({entry.source for entry in catalog.all()})
        assert sources == ["aviutl", "ymm4"]

    def test_other_files_are_skipped(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 関係の無いファイルまで拾うと、置き場の .txt などがテンプレートの窓に並ぶ
        catalog, _ = shelf
        assert all(entry.name != "読まない" for entry in catalog.all())

    def test_the_folder_name_is_kept(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 配布物はフォルダで分かれている そのまま見出しに使う
        catalog, _ = shelf
        assert "字幕" in catalog.folders()

    def test_the_place_in_the_original_file_is_kept(self, tmp_path: Path) -> None:
        """報告に書く位置は、原本の一覧の何本目か

        読める物だけの通し番号では、空の物を飛ばした後ろやエフェクトの一覧の物が
        原本の別の物を指し、受けた側が違うテンプレートを調べることになる
        """
        text = {"$type": "YukkuriMovieMaker.Project.Items.TextItem, YukkuriMovieMaker"}
        document = {
            "ItemTemplates": [
                {"Name": "空", "Path": ["空"], "Items": []},
                {"Name": "見出し", "Path": ["見出し"], "Items": [{**text, "Length": 60}]},
            ],
            "VideoEffectTemplates": [{"Name": "揺れ", "Effects": [{"$type": "X"}]}],
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("catalog.json", json.dumps(document, ensure_ascii=False))
        (tmp_path / "束.ymmt").write_bytes(buffer.getvalue())
        catalog = TemplateCatalog()
        catalog.scan((tmp_path,))

        found = {entry.name: entry for entry in catalog.all()}
        assert found["見出し"].origin == "ItemTemplates の 2 本目"
        assert found["揺れ"].origin == "VideoEffectTemplates の 1 本目"
        # 読む側の番号は読める物だけの並びのまま 変えると別のテンプレートを置く
        assert found["見出し"].index == 0
        assert found["見出し"].load()

    def test_an_absent_folder_is_not_an_error(self, tmp_path: Path) -> None:
        # 既定の置き場には AviUtl2 や YMM4 を入れていないと無いフォルダが入る 無いだけで失敗すると、
        # テンプレートの窓が開けない
        # 無いフォルダを先に並べ、後ろの置き場が読まれることも見る 無いフォルダで走査を
        # 打ち切ると、窓は開いても後ろの置き場の物が棚に 1 本も出ない
        (tmp_path / "ある").mkdir()
        (tmp_path / "ある" / "強調.object").write_text(ALIAS, "utf-8")
        catalog = TemplateCatalog()
        assert catalog.scan((tmp_path / "無い",)) == []
        found = catalog.scan((tmp_path / "無い", tmp_path / "ある"))
        assert [entry.name for entry in found] == ["強調"]

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("{これは JSON ではない".encode(), id="壊れた JSON"),
            pytest.param(b"\xff\xfe\x00\x81", id="UTF-8 でない"),
            pytest.param(_corrupt_zip(), id="中の壊れた ZIP"),
            pytest.param(_rewritten_zip(flags=0x1), id="暗号化された ZIP"),
            pytest.param(_rewritten_zip(method=99), id="知らない圧縮方式の ZIP"),
            pytest.param(json.dumps({"ItemTemplates": []}).encode(), id="中身が空"),
        ],
    )
    def test_an_unreadable_ymm4_file_stays_on_the_shelf(
        self, tmp_path: Path, content: bytes
    ) -> None:
        # 壊れると、読めない .ymmt が棚から黙って消える（UTF-8 でない物は走査ごと落ちる）
        # 本人は置いた物が無い理由が分からず、互換の報告にも写せない
        (tmp_path / "壊れ.ymmt").write_bytes(content)
        (tmp_path / "強調.object").write_text(ALIAS, "utf-8")
        catalog = TemplateCatalog()
        catalog.scan((tmp_path,))
        broken = catalog.find("壊れ")
        assert broken is not None
        assert broken.source == "ymm4"
        assert broken.error
        with pytest.raises(TemplateError, match=re.escape(broken.error)):
            broken.load()
        # 読めない物が 1 つあっても、ほかのテンプレートは並ぶ
        assert catalog.find("強調") is not None

    @pytest.mark.parametrize(
        ("content", "cause"),
        [
            pytest.param(_rewritten_zip(flags=0x1), RuntimeError, id="暗号化"),
            pytest.param(_rewritten_zip(method=99), NotImplementedError, id="知らない圧縮方式"),
            pytest.param(_corrupt_zip(), zlib.error, id="中の壊れた ZIP"),
        ],
    )
    def test_the_rewritten_zips_fail_where_intended(
        self, tmp_path: Path, content: bytes, cause: type[Exception]
    ) -> None:
        # 書き換えを誤ると、見出しの食い違い（BadZipFile）のような別の理由で読めなく
        # なり、上の試験は暗号化や圧縮方式を受ける所を通らないまま通ってしまう
        path = tmp_path / "書き換え.ymmt"
        path.write_bytes(content)
        with pytest.raises(Ymm4ParseError) as raised:
            load_template(path)
        assert type(raised.value.__cause__) is cause


class TestPlacing:
    def test_a_clip_is_added(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # AviUtl のエイリアスからクリップを作れないと、
        # テンプレートの窓の〔タイムラインへ置く〕を押しても何も置かれない
        catalog, _ = shelf
        entry = catalog.find("強調")
        assert entry is not None
        commands = place(entry.load(), Project.create(), at_frame=0)
        assert any(isinstance(command, AddClip) for command in commands)

    def test_a_track_is_created_when_there_is_none(
        self, shelf: tuple[TemplateCatalog, Path]
    ) -> None:
        # トラックを作らずに置くと、トラックの無いプロジェクトではクリップの置き先が無く、
        # テンプレートを置けない
        catalog, _ = shelf
        entry = catalog.find("強調")
        assert entry is not None
        commands = place(entry.load(), Project.create())
        assert isinstance(commands[0], AddTrack)
        # 作ったトラックへ実際に置けるかまで当てて見る 置き先が作った物と違うと、
        # 当てたときに無いトラックへ置こうとして配置ごと断られる
        project = Project.create()
        for command in commands:
            project = command.apply(project)
        (track,) = project.timeline.tracks
        assert track.id == commands[0].track.id
        assert [clip.timeline_start for clip in track.clips] == [0]

    def test_it_lands_where_asked(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # エイリアスは元の位置を持ったまま そのまま置くと指定した場所へ来ない
        catalog, _ = shelf
        entry = catalog.find("強調")
        assert entry is not None
        commands = place(entry.load(), Project.create(), at_frame=120)
        added = [c for c in commands if isinstance(c, AddClip)]
        assert added[0].clip.timeline_start == 120

    def test_an_alias_without_a_span_gets_the_default_length(
        self, shelf: tuple[TemplateCatalog, Path]
    ) -> None:
        # 1 フレームのクリップを置かれても使えない
        catalog, _ = shelf
        entry = catalog.find("長さなし")
        assert entry is not None
        commands = place(entry.load(), Project.create(), default_duration=150)
        added = [c for c in commands if isinstance(c, AddClip)]
        assert added[0].clip.duration == 150

    def test_an_alias_with_a_span_keeps_it(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # frame= の長さを捨てて既定の長さで置くと、
        # 90 フレームの字幕エイリアスが配布物と違う長さで置かれる
        catalog, _ = shelf
        entry = catalog.find("強調")
        assert entry is not None
        added = [c for c in place(entry.load(), Project.create()) if isinstance(c, AddClip)]
        assert added[0].clip.duration == 90


class TestRestyling:
    def existing(self) -> Clip:
        blur = registry.get("blur")
        assert blur is not None
        return Clip(
            timeline_start=300,
            duration=45,
            source=GeneratedSource(
                kind="text",
                params={"text": "自分で打った字幕", "size": AnimatedValue(30.0)},
            ),
            effects=(blur.create(radius=8),),
        )

    def commands(self, shelf: tuple[TemplateCatalog, Path]) -> list[object]:
        catalog, _ = shelf
        entry = catalog.find("強調")
        assert entry is not None
        return list(restyle(entry.load(), self.existing()))

    def test_the_text_is_kept(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 見本の文字で上書きしたら、それは着せ替えではない
        source = next(c for c in self.commands(shelf) if isinstance(c, SetSource)).source
        assert source is not None
        assert source.params["text"] == "自分で打った字幕"

    def test_the_look_is_taken_from_the_template(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 大きさ・フォント・縁を写さないと、
        # 〔選択中のクリップに適用〕を押しても字幕の見た目がテンプレートに変わらない
        source = next(c for c in self.commands(shelf) if isinstance(c, SetSource)).source
        assert source is not None
        assert value_at(source.params["size"], 0) == 72.0
        assert source.params["font"] == "Dela Gothic One"
        assert value_at(source.params["border_width"], 0) > 0

    def test_the_old_effects_are_cleared_first(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 残すと、前のデザインの縁とテンプレートの縁が二重に付く
        commands = self.commands(shelf)
        assert any(isinstance(c, RemoveEffect) for c in commands)
        added = [c for c in commands if isinstance(c, AddEffect)]
        assert [c.effect.kind for c in added] == ["border"]

    def test_the_timing_is_untouched(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 位置と長さを変えるコマンドは 1 つも出ない
        assert all(not isinstance(c, AddClip) for c in self.commands(shelf))

    def test_a_non_text_clip_is_refused(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 断らないと、図形のクリップの中身がテンプレートのテキストで置き換わり、
        # 図形が消えて文字になる
        catalog, _ = shelf
        entry = catalog.find("強調")
        assert entry is not None
        shape = Clip(
            timeline_start=0,
            duration=30,
            source=GeneratedSource(kind="shape", params={"shape": "rect"}),
        )
        assert restyle(entry.load(), shape) == []


class TestRestylingKeepsTheMotion:
    """動きの付いたテンプレートを、長さの違うクリップへ着せる

    中間点は**そのエイリアス自身の長さに対する絶対フレーム**で書かれている
    そのまま写すと、着せる先が短ければ動きが途中で切れ、長ければ途中で止まる
    着せるときは文字と長さを今のまま残す決まりなので、動きの側を尺に合わせる
    """

    def moving(self, shelf: tuple[TemplateCatalog, Path], duration: int) -> list[tuple[int, float]]:
        """``duration`` フレームのクリップへ着せたあとの、横位置のキーフレーム"""
        catalog, _ = shelf
        entry = catalog.find("動き")
        assert entry is not None
        clip = Clip(
            timeline_start=300,
            duration=duration,
            source=GeneratedSource(kind="text", params={"text": "自分で打った字幕"}),
        )
        added = [c for c in restyle(entry.load(), clip) if isinstance(c, AddEffect)]
        transform = next(c.effect for c in added if c.effect.kind == "transform")
        value = transform.params["pos_x"]
        assert isinstance(value, AnimatedValue)
        return [(k.frame, k.value) for k in value.keyframes]

    def test_the_template_itself_has_three_points(
        self, shelf: tuple[TemplateCatalog, Path]
    ) -> None:
        # 前提の確認 ここが崩れると、以下の数の意味が変わる
        assert self.moving(shelf, 180) == [(0, 0.0), (89, 150.0), (179, 300.0)]

    def test_a_shorter_clip_still_gets_the_whole_motion(
        self, shelf: tuple[TemplateCatalog, Path]
    ) -> None:
        """短いクリップでも**終わりまで動く**

        そのまま写すと 60 フレームのクリップでは 3 分の 1 で切れ、
        着地した見た目（横位置 300）が一度も出ない
        """
        moved = self.moving(shelf, 60)
        assert moved[0] == (0, 0.0)
        assert moved[-1] == (59, 300.0)
        assert moved[1][0] == 29, "中間点が端に寄っている"

    def test_a_longer_clip_keeps_moving_to_the_end(
        self, shelf: tuple[TemplateCatalog, Path]
    ) -> None:
        # 伸ばさないと、180 フレーム目で止まって残りの 180 フレームは静止する
        moved = self.moving(shelf, 360)
        assert moved[-1] == (359, 300.0)
        assert 170 < moved[1][0] < 190, "中間点が真ん中に来ていない"

    def test_the_same_length_is_left_alone(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 同じ長さで数がずれるなら、丸めの誤差を持ち込んでいる
        assert self.moving(shelf, 180) == [(0, 0.0), (89, 150.0), (179, 300.0)]

    def test_only_the_times_move(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 値まで伸び縮みさせると、動く幅（0 から 300）が変わってしまう
        assert [value for _, value in self.moving(shelf, 45)] == [0.0, 150.0, 300.0]

    def test_a_clip_too_short_for_every_point_keeps_the_end(
        self, shelf: tuple[TemplateCatalog, Path]
    ) -> None:
        """点を全部置けないほど短くても、**終わりの値**は残る

        終わりを落とすと、着地した見た目にならないまま止まる
        """
        moved = self.moving(shelf, 2)
        assert moved[0] == (0, 0.0)
        assert moved[-1] == (1, 300.0)

    def test_an_effects_only_template_is_fitted_too(self) -> None:
        """エフェクトだけのテンプレート（YMM4 のアニメーション効果）も尺に合わせる

        入れ物の長さを捨てていると尺合わせができず、300 フレーム向けの動きが
        60 フレームのクリップの 5 分の 1 で止まる
        """
        from sashimono.compat.mapped import MappedObject
        from sashimono.core.model import Keyframe

        transform = registry.require("transform").create(
            rotation=AnimatedValue(
                keyframes=(Keyframe(frame=0, value=0.0), Keyframe(frame=299, value=30.0))
            )
        )
        template = MappedObject(
            clip=Clip(timeline_start=0, duration=300, effects=(transform,)),
            layer=1,
            kind="effects",
            has_span=False,
        )
        clip = Clip(
            timeline_start=0,
            duration=60,
            source=GeneratedSource(kind="text", params={"text": "字幕"}),
        )
        added = [c for c in restyle([template], clip) if isinstance(c, AddEffect)]
        value = added[0].effect.params["rotation"]
        assert isinstance(value, AnimatedValue)
        assert [k.frame for k in value.keyframes] == [0, 59]

    def test_the_ymm4_end_convention_does_not_shift_the_middle(self) -> None:
        """YMM4 の終わりの決まりでも中間点が**1 フレームもずれない**

        AviUtl は最後の点を長さ - 1 に置き（frame=0,89,179 で長さ 180）、
        YMM4 は長さそのものに置く 分母を取り違えると、長さ 300 の
        フレーム 240 が長さ 600 で 479 ではなく 481 へ移る
        """
        from sashimono.compat.mapped import MappedObject
        from sashimono.core.model import Keyframe

        transform = registry.require("transform").create(
            rotation=AnimatedValue(
                keyframes=(
                    Keyframe(frame=0, value=0.0),
                    Keyframe(frame=240, value=10.0),
                    Keyframe(frame=300, value=30.0),
                )
            )
        )
        template = MappedObject(
            clip=Clip(timeline_start=0, duration=300, effects=(transform,)),
            layer=1,
            kind="effects",
            has_span=False,
        )
        clip = Clip(
            timeline_start=0,
            duration=600,
            source=GeneratedSource(kind="text", params={"text": "字幕"}),
        )
        added = [c for c in restyle([template], clip) if isinstance(c, AddEffect)]
        value = added[0].effect.params["rotation"]
        assert isinstance(value, AnimatedValue)
        assert [k.frame for k in value.keyframes] == [0, 479, 599]

    def test_a_one_frame_clip_shows_the_end_of_the_motion(
        self, shelf: tuple[TemplateCatalog, Path]
    ) -> None:
        """1 フレームのクリップには**終わりの値**が出る

        そのまま写すと、クリップの外に出たキーフレームが残ったままになり、
        唯一のフレームではテンプレートの始まり（横位置 0）が出る
        """
        moved = self.moving(shelf, 1)
        assert moved == [(0, 300.0)]

    def test_the_clip_itself_is_not_resized(self, shelf: tuple[TemplateCatalog, Path]) -> None:
        # 動きを合わせるのであって、クリップの長さは今のまま
        catalog, _ = shelf
        entry = catalog.find("動き")
        assert entry is not None
        clip = Clip(
            timeline_start=300,
            duration=60,
            source=GeneratedSource(kind="text", params={"text": "自分で打った字幕"}),
        )
        commands = restyle(entry.load(), clip)
        assert all(not isinstance(c, AddClip) for c in commands)
        source = next(c for c in commands if isinstance(c, SetSource)).source
        assert source is not None
        assert source.params["text"] == "自分で打った字幕"
