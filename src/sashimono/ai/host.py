"""AI 層が編集ソフトに求めること

AI の操作は最終的に UI と同じ :class:`~sashimono.core.commands.Document` を通る
ただし AI 層はウィジェットを知らないでいたいので、必要な操作だけをここに列挙して
おき、:class:`~sashimono.ui.main_window.MainWindow` がそれを満たす

この分離のおかげで、テストではウィジェットを一切作らずに、偽のホストへ差し替えて
ツールの挙動を確かめられる
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sashimono.ai.sheet import SheetLayout
from sashimono.ai.sound_graph import SoundGraph
from sashimono.compat.catalog import TemplateEntry
from sashimono.compat.subtitle_looks import LookRules, SavedLooks
from sashimono.core.commands import Command, Document
from sashimono.core.emphasis import EmphasisStyle
from sashimono.core.io.atomic import Expected
from sashimono.core.mixing import DuckShape
from sashimono.core.model import ClipId, MediaId, MediaItem, Project, SceneId
from sashimono.core.seams import SeamStyle
from sashimono.core.search import SceneNote
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.cache import SceneState, ScreenState
from sashimono.engine.checks import CheckRequest, CheckState
from sashimono.engine.vision import ScreenKind

__all__ = [
    "EXPORT_RUNNING",
    "EXPORT_STATES",
    "EditorHost",
    "ExportRequest",
    "SheetCanvas",
    "ToolError",
]

#: 書き出しの様子（:meth:`EditorHost.export_status` の ``state``）
#: 走っている間は「書いている」「置いている」 終わった後は残りのどれか
EXPORT_RUNNING = ("rendering", "placing")
EXPORT_STATES = ("idle", *EXPORT_RUNNING, "done", "failed", "cancelled")


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """AI が頼んだ書き出し 選べる物は画面の書き出しの窓と同じ"""

    #: 確かめ済みの書き出し先（リンクを辿った後の場所）
    path: Path
    #: 書き出すシーン ``None`` ならメイン
    scene_id: SceneId | None
    #: ``all`` か ``work_area``（:data:`~sashimono.engine.encode.RANGE_ALL` など）
    range_choice: str
    #: 映像のコーデック ``None`` なら自動（使えなければ次の候補へ落ちる）
    codec: str | None
    #: 映像のビットレート（Mbps）
    bitrate_mbps: int
    #: 許したときの書き出し先の身元（無ければ無かったこと） 置く直前に照らし、名前が現れたり
    #: 中身が変わったりしていれば置かずに断る（#327）
    expect: Expected


class ToolError(RuntimeError):
    """ツールの実行に失敗した

    この例外の文面はそのまま Claude へ返る 「どうすれば直るか」が分かる書き方を
    すること Claude は次の手をこの文面だけで決める
    """


class SheetCanvas(Protocol):
    """コマを並べて描く 1 枚（:meth:`EditorHost.start_sheet`）

    1 コマずつ描かせるのは、間で編集画面へ手を返すため（:class:`~sashimono.ai.operations.Continue`）
    まとめて描くと、大きい作品で何十コマも描く間、画面が止まる
    """

    def draw(self, index: int, frame: int, label: str) -> None:
        """``index`` 番目の升へ ``frame`` を描き、真下の帯へ ``label`` を書く"""
        ...

    def png(self) -> bytes:
        """描き終えた 1 枚を PNG にする"""
        ...


class EditorHost(Protocol):
    """AI からの操作を受け付ける編集ソフト側の窓口

    ここのメソッドは**すべて UI スレッドで呼ばれる**（:mod:`sashimono.ai.bridge` が
    そう仕向ける） 実装側でスレッドを気にする必要はない
    """

    @property
    def document(self) -> Document:
        """編集中のプロジェクトと履歴"""
        ...

    @property
    def project(self) -> Project:
        """いま編集しているタイムラインを ``timeline`` に差し込んだプロジェクト

        シーンを開いていれば、そのシーンのタイムラインが入っている 読み取りは
        すべてこれを見る（画面と同じものを AI も見るため）
        """
        ...

    @property
    def active_scene(self) -> SceneId | None:
        """開いているシーン メインなら ``None``"""
        ...

    def set_active_scene(self, scene_id: SceneId | None) -> None:
        """編集するシーンを切り替える 以後のコマンドはそのシーンの中で実行される"""
        ...

    @property
    def playhead(self) -> int: ...

    def seek(self, frame: int) -> None: ...

    @property
    def selected_clip(self) -> ClipId | None: ...

    def select_clip(self, clip_id: ClipId | None) -> None: ...

    @property
    def selected_clips(self) -> tuple[ClipId, ...]:
        """選んでいるクリップすべて 最後が :attr:`selected_clip`"""
        ...

    def select_clips(self, clip_ids: list[ClipId]) -> None:
        """選択を丸ごと入れ替える 空なら選択を解く"""
        ...

    def apply_commands(self, commands: list[Command], label: str) -> None:
        """コマンドをまとめて実行する 失敗したら :class:`ToolError` を投げること

        シーンを開いていれば、そのシーンの中で実行すること（``InScene`` で包む）

        まとめるのは、1 つの指示による編集を 1 回の Undo で戻せるようにするため
        """
        ...

    def undo(self) -> None:
        """人の〔元に戻す〕と同じく 1 段戻し、画面を今の形に合わせる（#333）

        文書（:attr:`document`）を直に戻すと、タイムラインとメニューの名前が前の形のまま残る
        """
        ...

    def redo(self) -> None:
        """人の〔やり直す〕と同じく 1 段やり直し、画面を今の形に合わせる（#333）"""
        ...

    def stop_playback(self) -> None:
        """再生を止める フレームを描く前に呼ぶ"""
        ...

    def render_png(self, frame: int, *, width: int) -> bytes:
        """そのフレームを合成して PNG で返す"""
        ...

    def start_sheet(self, layout: SheetLayout) -> SheetCanvas:
        """コマを並べる 1 枚を用意する（コンタクトシートと範囲のコマ #320）

        描くのは用意した時点のプロジェクト（開いているシーン） 描く間に編集が入っても、
        1 枚の中で前後の升が別の版の絵にならないように
        """
        ...

    def start_media_sheet(self, media: MediaItem, layout: SheetLayout) -> SheetCanvas:
        """素材 1 本の中の絵を並べる 1 枚を用意する（場面の説明 #326）

        タイムラインに置いていない素材の場面も見られるように、素材だけを頭から置いた作品
        （:func:`~sashimono.core.commands.insert.media_alone`）を描く 升の ``frame`` は
        素材の頭から数えた、作品の fps のフレーム 作品にもアプリの設定にも触らない
        """
        ...

    def scene_notes(self, media: MediaItem) -> tuple[SceneNote, ...]:
        """覚えた場面の説明（#326） 素材の解析と同じ置き場（``engine/cache``）から読む"""
        ...

    def remember_scene_notes(self, media: MediaItem, notes: tuple[SceneNote, ...]) -> None:
        """素材の場面の説明を丸ごと置き換えて覚える（#326） 空なら忘れる

        作品の外の控え（``engine/cache``）に書く 作品は変えないので取り消しの段も積まない
        控えは消しても絵を見せ直せば作れる物で、アプリ自身の設定でもない
        """
        ...

    def draw_sound_graph(self, graph: SoundGraph) -> bytes:
        """音量の推移のグラフを PNG にする（#338） 描く物はすべて ``graph`` が持つ

        作品にもアプリの設定にも触らない 描くのが編集画面なのは、字を描くのに Qt が要るため
        """
        ...

    def probe(self, path: Path) -> MediaItem:
        """素材を読み、メディアプールに入れられる形にする"""
        ...

    @property
    def splits_media(self) -> bool:
        """動画の映像と音声を別のトラックへ分けて置くか 音声が何本もあれば音ごとに分ける
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.media_split`）

        AI が置いた素材も画面から置いた素材と同じ置き方にする 食い違うと、同じ動画でも
        頼み方によってレイヤーの数が変わる
        """
        ...

    @property
    def subtitle_wrap_share(self) -> int:
        """既定の見た目で焼き込む字幕を折り返す幅（画面の幅の % 0 は折り返さない #249）
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.subtitle_wrap_share`）

        画面の〔焼き込み〕と同じにする 頼み方で字幕が折り返したり切れたりしないように
        """
        ...

    @property
    def saved_looks(self) -> SavedLooks:
        """保存したプリセットと自作のエイリアスの写し（タイムラインの〔追加〕と同じ置き場）

        字幕を置く道具が、見た目を名前で選ぶのに使う（#308） 置き場そのものは渡さない
        本人の持ち物の名前の変更や削除まで、AI の手の届く所に出さないため（読むだけ #354）
        """
        ...

    @property
    def template_shelf(self) -> tuple[TemplateEntry, ...]:
        """テンプレートの棚（AviUtl のエイリアス・YMM4 のアイテムテンプレート #328） 読むだけ

        タイムラインの〔追加〕→〔エイリアス〕と互換の〔テンプレート…〕と同じ一覧 AI はここから
        選んで置く・着せる 中身はまだ読まない（選ばれた 1 本だけを読む 全部を読むと、数百本の
        棚で道具が返るまで待たされる）
        """
        ...

    @property
    def look_rules(self) -> LookRules:
        """見た目を着せる決まり（本人のプリセットの当て方と、棚の着せ替えの設定）

        画面の〔焼き込み〕・設定パネルの〔プリセット…〕・棚の〔選択中のクリップに適用〕と同じに
        する 頼み方で同じプリセットや棚の当たり方が変わらないように
        """
        ...

    def analyze(self, media: MediaItem) -> None:
        """波形とサムネイルの用意を予約する"""
        ...

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None:
        """用意できていれば波形を返す 無ければ ``None`` ``stream`` は音声（省けば 1 本目）"""
        ...

    @property
    def loudness_target(self) -> float:
        """完成物の検査で比べる音量の基準（LUFS）の既定（#337）
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.check_loudness_target`）

        配信か放送かで答えが違う（-14 と -23 では 9 LU 違う） AI が引数で渡さなければこれを使う
        """
        ...

    @property
    def duck_shape(self) -> DuckShape:
        """BGM を話している間だけ下げる（ダッキング #329）ときの既定の下げ幅と速さ
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.duck_shape`）

        どれだけ下げたいか・どれだけ速く戻したいかは人と作品で違う AI が引数で渡さなければこれを使う
        """
        ...

    @property
    def emphasis_style(self) -> EmphasisStyle:
        """強調の動き（#331）の既定の強さ・動き切る長さ・動き方
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.emphasis_style`）

        どれだけ寄るか・どれだけ速く動くかは人と作品で違う AI が引数で渡さなければこれを使う
        """
        ...

    @property
    def seam_style(self) -> SeamStyle:
        """ジャンプカットのつなぎ目（#332）の既定の処理（映像の処理・長さ・音のフェード）
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.seam_style`）

        溶かすか寄るか・どれだけの長さかは人と作品で違う AI が引数で渡さなければこれを使う
        """
        ...

    @property
    def image_edge(self) -> int:
        """AI に見せる絵 1 枚の長い辺の上限（画素 #344）
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.ai_image_edge`）

        絵の費用は画素の数でほぼ決まる どこまで払ってよいかは人で違う 絵を返す道具は、
        AI が大きい絵を頼んでもこれに収まるよう縮める
        """
        ...

    def request_check(self, request: CheckRequest) -> None:
        """完成物の検査を裏で始める（#337） 同じ頼みが走っているか終わっていれば何もしない

        止めた物・失敗した物は、同じ頼みでも調べ直す
        """
        ...

    def check_state(self) -> CheckState:
        """完成物の検査の今 拾った所までの結果と、走っているか・失敗の理由"""
        ...

    def check_waiter(self) -> Callable[[float], bool]:
        """今の検査が終わるまで、最長で引数の秒だけ待つ関数 終わっていれば真を返す

        返す関数は**AI のスレッドから呼ばれる**（:class:`~sashimono.ai.operations.Pending`）
        画面のスレッドで待つと、待つ間は画面が固まる
        """
        ...

    def request_scenes(self, media: MediaItem) -> None:
        """場面の解析を頼む（#321） 読み終えた物・走っている物があれば何もしない

        本人の設定で取り込み時の解析を切っていても、AI が頼んだ素材は読む
        """
        ...

    def scene_state(self, media: MediaItem) -> SceneState:
        """場面の解析の今 読み終えた並びか、読んだ所までの写しと、走っているか・失敗の理由

        頼みはしない 失敗の理由を読む前に頼み直すと、理由が消えて「途中」と答えてしまう
        """
        ...

    def scene_waiter(self, media_id: MediaId) -> Callable[[float], bool]:
        """場面の解析が終わるまで、最長で引数の秒だけ待つ関数 終わっていれば真を返す

        返す関数は**AI のスレッドから呼ばれる**（:class:`~sashimono.ai.operations.Pending`）
        画面のスレッドで待つと、待つ間は画面が固まる
        """
        ...

    def request_screen(self, media: MediaItem, kind: ScreenKind) -> None:
        """画面の中身（文字か顔の位置 #324）を調べるよう頼む 調べ終えた物・調べている物が
        あれば何もしない 失敗した物（部品が無かった）は頼み直す

        頼むのは本人のキャッシュへの解析だけで、読み込みのときに調べるかの設定
        （``Preferences.screen_analysis``）は変えない
        """
        ...

    def screen_state(self, media: MediaItem, kind: ScreenKind) -> ScreenState:
        """画面の中身を調べた様子 調べ終えた結果か、調べた所までの写しと、失敗の理由

        頼みはしない（:meth:`scene_state` と同じ 失敗の理由を読む前に頼み直すと理由が消える）
        """
        ...

    def screen_waiter(self, media_id: MediaId, kind: ScreenKind) -> Callable[[float], bool]:
        """画面の中身を調べ終えるまで、最長で引数の秒だけ待つ関数 終わっていれば真を返す

        返す関数は**AI のスレッドから呼ばれる**（:meth:`scene_waiter` と同じ）
        """
        ...

    def start_transcription(
        self, media_id: MediaId, model: str, *, audio_stream: int | None = None
    ) -> str:
        """起こしを始める 戻り値は画面に出す短い文言

        ``audio_stream`` は起こす音声ストリームの番号 ``None`` なら 1 本目
        """
        ...

    def transcription_status(self) -> str:
        """走っている起こし（と話者の分離）の様子"""
        ...

    def start_speaker_separation(
        self, media_id: MediaId, *, audio_stream: int | None = None, speakers: int | None = None
    ) -> str:
        """話者の分離を始める（#323） 戻り値は画面に出す短い文言 起こしと同じ列で順に走る

        終わった物はその素材と音声の今の字幕へ話者を付ける 様子は :meth:`transcription_status`
        ``speakers`` は話す人の数 ``None`` なら自動 部品が無い・字幕が無いと ``RuntimeError``
        """
        ...

    @property
    def project_path(self) -> Path | None:
        """開いている作品（``.sme``）の場所 保存していなければ ``None``

        AI が書ける場所（:func:`~sashimono.ai.safety.writable_path`）はこのフォルダが基準
        """
        ...

    @property
    def export_folder(self) -> Path | None:
        """書き出しの窓で前に選んだフォルダ 一度も書き出していなければ ``None``
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.export_folder`）

        人が窓で選んで書き出した所なので、AI もそこへは書ける（2026-10-14 利用者の決定）
        """
        ...

    def write_project(self, path: Path, *, adopt: bool, expect: Expected) -> str:
        """作品を ``path`` へ保存する（B1 #327） 画面の〔保存〕と同じ控え・書き方を通す

        ``path`` は確かめ済みの場所（:func:`~sashimono.ai.safety.pinned_place` の中で呼ぶ）
        ``adopt`` が真なら、以後の作品の場所を ``path`` にする（名前を付けて保存）
        ``expect`` は許したときの ``path`` の身元 書いて名前を付ける所で照らし、その間に名前が
        現れたり中身が変わったりしていれば、置かずに断る（字幕と静止画の書く口も同じ）
        窓は出さない 断るときは :class:`ToolError` 戻り値は知らせに添える言葉（無ければ空）
        """
        ...

    def write_subtitles(
        self, path: Path, *, frame_range: tuple[int, int] | None, expect: Expected
    ) -> Path:
        """タイムラインの字幕を ``path`` へ書く（#327） 画面の字幕の書き出しと同じ形式と処理

        相手は開いているタイムライン（字幕パネルが見ている物と同じ） ``frame_range`` を
        渡すと、その範囲だけを頭を 0 にずらして書く ``path`` は確かめ済みの場所
        （:func:`~sashimono.ai.safety.pinned_place` の中で呼ぶ）
        """
        ...

    def write_snapshot(self, path: Path, frame: int, *, expect: Expected) -> int:
        """``frame`` を画面の〔静止画を保存〕と同じ描き方で PNG にして書く（#327）

        描いたフレームを返す（終わりを指していれば最後のフレーム） ``path`` は確かめ済みの場所
        書けなければ :class:`ToolError`
        """
        ...

    def start_export(self, request: ExportRequest) -> None:
        """書き出しを裏で始める（B1 #327） 画面の書き出しの窓と同じ設定の作り方と処理を通す

        すぐ戻る 進み具合は :meth:`export_status`、止めるのは :meth:`cancel_export`
        すでに走っていれば :class:`ToolError`
        """
        ...

    def export_status(self) -> dict[str, object]:
        """AI が始めた書き出しの様子 ``state`` は :data:`EXPORT_STATES` のどれか"""
        ...

    def export_waiter(self) -> Callable[[float], bool]:
        """書き出しが終わるまで、最長で引数の秒だけ待つ関数 走っていなければ真を返す

        :meth:`scene_waiter` と同じく、返す関数は**AI のスレッドから呼ばれる**
        （:class:`~sashimono.ai.operations.Pending`） 画面のスレッドで待つと画面が固まる
        """
        ...

    def cancel_export(self) -> bool:
        """走っている書き出しを止める 止めるよう頼めたら真"""
        ...

    def remembered_names(self) -> tuple[str, ...]:
        """アプリが作品の外で覚えている名前と場所（#342） 読むだけ

        AI の道具の結果に混ざりうる物を全部返すこと 字幕起こしの記録に残った素材の名前、
        前に開いた作品の場所、設定に入った置き場など 名前と場所を送らないと決めた人の所で、
        ブリッジがこれを伏せ字の表（:func:`~sashimono.ai.outbound.file_secrets`）へ入れる
        作品の外に名前を覚える所を足したら、ここからも返す 返さないと、別の作品を開いた後に
        その名前が伏せずに送られる（``tests/ui/test_remembered_names.py`` が見張る）
        """
        ...

    def app_settings(self) -> dict[str, object]:
        """アプリ自身の設定（好み・ショートカット・画面配置・版・ヘルプ）の写し

        AI は読むだけ（2026-10-14 利用者の決定） 変える口はこの窓口に置かない 置くと、
        ショートカットや更新の設定を AI が黙って変えられ、人が気付かないまま次の起動から
        動きが変わる 返すのは写しで、受け取った側が書き換えても元は変わらないこと
        """
        ...
