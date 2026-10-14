"""AI 層が編集ソフトに求めること

AI の操作は最終的に UI と同じ :class:`~sashimono.core.commands.Document` を通る
ただし AI 層はウィジェットを知らないでいたいので、必要な操作だけをここに列挙して
おき、:class:`~sashimono.ui.main_window.MainWindow` がそれを満たす

この分離のおかげで、テストではウィジェットを一切作らずに、偽のホストへ差し替えて
ツールの挙動を確かめられる
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sashimono.core.commands import Command, Document
from sashimono.core.model import ClipId, MediaId, MediaItem, Project, SceneId
from sashimono.engine.audio.waveform import Waveform

__all__ = ["EXPORT_RUNNING", "EXPORT_STATES", "EditorHost", "ExportRequest", "ToolError"]

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


class ToolError(RuntimeError):
    """ツールの実行に失敗した

    この例外の文面はそのまま Claude へ返る 「どうすれば直るか」が分かる書き方を
    すること Claude は次の手をこの文面だけで決める
    """


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

    def stop_playback(self) -> None:
        """再生を止める フレームを描く前に呼ぶ"""
        ...

    def render_png(self, frame: int, *, width: int) -> bytes:
        """そのフレームを合成して PNG で返す"""
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

    def analyze(self, media: MediaItem) -> None:
        """波形とサムネイルの用意を予約する"""
        ...

    def waveform(self, media: MediaItem, stream: int | None = None) -> Waveform | None:
        """用意できていれば波形を返す 無ければ ``None`` ``stream`` は音声（省けば 1 本目）"""
        ...

    def start_transcription(
        self, media_id: MediaId, model: str, *, audio_stream: int | None = None
    ) -> str:
        """起こしを始める 戻り値は画面に出す短い文言

        ``audio_stream`` は起こす音声ストリームの番号 ``None`` なら 1 本目
        """
        ...

    def transcription_status(self) -> str:
        """走っている起こしの様子"""
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

    def write_project(self, path: Path, *, adopt: bool) -> str:
        """作品を ``path`` へ保存する（B1 #327） 画面の〔保存〕と同じ控え・書き方を通す

        ``path`` は確かめ済みの場所（:func:`~sashimono.ai.safety.pinned_place` の中で呼ぶ）
        ``adopt`` が真なら、以後の作品の場所を ``path`` にする（名前を付けて保存）
        窓は出さない 断るときは :class:`ToolError` 戻り値は知らせに添える言葉（無ければ空）
        """
        ...

    def write_subtitles(self, path: Path, *, frame_range: tuple[int, int] | None) -> Path:
        """タイムラインの字幕を ``path`` へ書く（#327） 画面の字幕の書き出しと同じ形式と処理

        相手は開いているタイムライン（字幕パネルが見ている物と同じ） ``frame_range`` を
        渡すと、その範囲だけを頭を 0 にずらして書く ``path`` は確かめ済みの場所
        （:func:`~sashimono.ai.safety.pinned_place` の中で呼ぶ）
        """
        ...

    def write_snapshot(self, path: Path, frame: int) -> int:
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

    def cancel_export(self) -> bool:
        """走っている書き出しを止める 止めるよう頼めたら真"""
        ...

    def app_settings(self) -> dict[str, object]:
        """アプリ自身の設定（好み・ショートカット・画面配置・版・ヘルプ）の写し

        AI は読むだけ（2026-10-14 利用者の決定） 変える口はこの窓口に置かない 置くと、
        ショートカットや更新の設定を AI が黙って変えられ、人が気付かないまま次の起動から
        動きが変わる 返すのは写しで、受け取った側が書き換えても元は変わらないこと
        """
        ...
