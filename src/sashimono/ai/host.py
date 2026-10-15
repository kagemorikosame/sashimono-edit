"""AI 層が編集ソフトに求めること

AI の操作は最終的に UI と同じ :class:`~sashimono.core.commands.Document` を通る
ただし AI 層はウィジェットを知らないでいたいので、必要な操作だけをここに列挙して
おき、:class:`~sashimono.ui.main_window.MainWindow` がそれを満たす

この分離のおかげで、テストではウィジェットを一切作らずに、偽のホストへ差し替えて
ツールの挙動を確かめられる
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from sashimono.core.commands import Command, Document
from sashimono.core.model import ClipId, MediaId, MediaItem, Project, SceneId
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.cache import SceneState
from sashimono.engine.checks import CheckRequest, CheckState

__all__ = ["EditorHost", "ToolError"]


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

    @property
    def loudness_target(self) -> float:
        """完成物の検査で比べる音量の基準（LUFS）の既定（#337）
        （本人の設定 :attr:`~sashimono.ui.workspace.Preferences.check_loudness_target`）

        配信か放送かで答えが違う（-14 と -23 では 9 LU 違う） AI が引数で渡さなければこれを使う
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

    def app_settings(self) -> dict[str, object]:
        """アプリ自身の設定（好み・ショートカット・画面配置・版・ヘルプ）の写し

        AI は読むだけ（2026-10-14 利用者の決定） 変える口はこの窓口に置かない 置くと、
        ショートカットや更新の設定を AI が黙って変えられ、人が気付かないまま次の起動から
        動きが変わる 返すのは写しで、受け取った側が書き換えても元は変わらないこと
        """
        ...
