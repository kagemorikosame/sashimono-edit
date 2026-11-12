"""別のプロセスの書き出しへ、親で効いている値をすべて届ける（#453 #466）

子は spawn で起こすので、親のプロセスの中だけで持つ値（モジュールの大域の値・設定画面から
当てた値）は子で既定へ戻る 環境変数は子が起きたときに写るので要らない
設定の項目とモジュールの大域の値を 1 つずつ「子へ渡す」「子は要らない」に分けておき、
どちらにも入らない物が増えたら落とす 足した人が子のことを決めずに済ませると、本人が切った
DLL を子だけが読む（Codex の 2 回目の P1）ような食い違いがまた起きる
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from sashimono.engine.encode.export_job import AviUtlRuntime
from sashimono.engine.encode.exporter import ExportSettings
from sashimono.ui.workspace import Preferences

SRC = Path(__file__).resolve().parents[2] / "src" / "sashimono"

#: プロセスごとの値として当てる設定 → それを運ぶ :class:`AviUtlRuntime` の欄
#: 安全に関わる物（DLL を読むか）もここ 子で既定に戻ると、本人が禁じた物を子が走らせる
BY_RUNTIME = {
    "native_modules": "native_modules",
    "all_aviutl_plugins": "all_plugins",
}

#: 書き出しの窓が :class:`ExportSettings` に入れて渡す設定（子へは書き出しの設定ごと届く）
#: ``export_process`` は子を使うかそのもの ``export_folder`` は書き出し先の既定の置き場
BY_EXPORT_SETTINGS = frozenset(
    {
        "export_pipeline_depth",
        "decode_threads",
        "smooth_audio_motion",
        "export_gpu_color",
        "export_chapters",
        "encoder_priority",
        "export_folder",
        "export_process",
    }
)

#: 子の書き出しが読まない設定 書き出しの絵を変えない物だけをここへ入れる
#: 編集の時に作品へ書き込む物（字幕・強調・つなぎ目など）は作品ごと子へ渡る
NOT_FOR_THE_CHILD = frozenset(
    {
        # 控え・先読み・解析 プレビューの速さ用で、書き出しは控えを使わない
        "use_proxy",
        "proxy_height",
        "proxy_long_gop",
        "proxy_process",
        "analysis_process",
        "proxy_unused",
        "auto_quality",
        "auto_quality_divisor",
        "prefetch",
        "prefetch_budget_mb",
        "prefetch_thread",
        "prefetch_yield",
        "scene_analysis",
        "scene_threads",
        "screen_analysis",
        "screen_samples",
        "ocr_engine",
        # 画面を持つ親のプロセスの速さと見張り 子は画面を持たない
        "responsive_switch",
        "gc_manual",
        "stall_watch",
        # 画面の見た目と操作の癖
        "pool_progress",
        "hdr_notice",
        "media_view",
        "dock_tabs",
        "preview_handles",
        "preview_fill",
        "preview_margin",
        "keyframe_drag",
        "value_lines",
        "detail_min_width",
        "stretch_waves",
        "double_click_reset",
        "library_thumbnails",
        "library_backdrop",
        "library_confirm_delete",
        "shelf_thumbnails",
        "save_overwrite_without_asking",
        "wheel_unfocused",
        "timeline_snap",
        "snap_distance",
        "preview_snap",
        "playhead_snap",
        "playhead_snap_clips",
        "playhead_snap_keyframes",
        "playhead_snap_work_area",
        "playhead_snap_markers",
        "theme",
        # 編集の時に作品へ書き込む物 書き込んだ結果は作品ごと子へ渡る
        "match_video",
        "preset_with_text",
        "preset_with_position",
        "preset_keep_effects",
        "insert_paste",
        "new_project_layers",
        "new_project_style_guide",
        "media_split",
        "subtitle_wrap",
        "subtitle_wrap_percent",
        "subtitle_look_remember",
        "subtitle_look_last",
        "speakers_after_transcribe",
        "restyle_keep_wrap",
        "ripple_playhead",
        "ripple_layers",
        "check_loudness_target",
        "word_timestamps",
        "subtitle_fit_voice",
        "subtitle_start_offset_ms",
        "subtitle_end_offset_ms",
        "duck_depth_db",
        "duck_attack_ms",
        "duck_release_ms",
        "emphasis_zoom_percent",
        "emphasis_pop_percent",
        "emphasis_shake_px",
        "emphasis_pan_px",
        "emphasis_ramp_ms",
        "emphasis_easing",
        "seam_video",
        "seam_crossfade_ms",
        "seam_zoom_percent",
        "seam_audio_fade_ms",
        "narration_engine_url",
        "narration_voice",
        "narration_voice_name",
        "narration_speed_percent",
        "narration_volume_percent",
        "narration_ai_picks_voice",
        "narration_subtitles",
        "auto_color_strength",
        "denoise_reduction_db",
        "reframe_size",
        "reframe_follow",
        "reframe_smooth_ms",
        "reframe_faces",
        # 保存・退避・更新・素材をまとめる
        "update_check",
        "update_beta",
        "update_confirm",
        "scripts_move",
        "autosave",
        "autosave_seconds",
        "backup",
        "backup_generations",
        "state_folder",
        "recovery_keep_days",
        "state_limit_mb",
        "package_folder",
        "package_whole",
        "package_margin_seconds",
        "package_keep_metadata",
        # AI の相手 書き出しの子は AI を呼ばない
        "ai_model",
        "ai_effort",
        "chat_enter_sends",
        "ai_busy_animation",
        "ai_done_alert",
        "ai_handoff",
        "ai_auto_update",
        "ai_ask_questions",
        "ai_plan_first",
        "ai_ask_before_ungroup",
        "ai_fix_rounds",
        "ai_send_frames",
        "ai_send_text",
        "ai_send_files",
        "ai_limit_turn_tokens",
        "ai_limit_turn_images",
        "ai_limit_turn_minutes",
        "ai_limit_day_tokens",
        "ai_limit_day_images",
        "ai_limit_action",
        "ai_image_edge",
    }
)

#: 画面の外（``ui`` の外）でモジュールの大域の値を書き換える所 → 子でどうするか
#: ``global`` で書く所が増えたら、ここで決めるまで落ちる
MODULE_GLOBALS = {
    # DLL を読むか・汎用プラグインを全部読むか :class:`AviUtlRuntime` で渡す
    ("compat/aviutl/native.py", "_enabled"): "渡す",
    ("compat/aviutl/plugin.py", "_scan_all"): "渡す",
    # スクリプトの一覧 置き場は設定に依らないので、子は :meth:`AviUtlRuntime.apply` で読み直す
    ("compat/aviutl/catalog.py", "_catalog"): "子で読む",
    # 見本の設定を試すときだけ指す置き場 アプリは書き換えない
    ("compat/aviutl/plugin.py", "_app_data"): "要らない",
    # 字幕のテンプレートの一覧 編集の時に使い、描くときは引かない
    ("compat/catalog.py", "_catalog"): "要らない",
    # 読み込みの間だけ記録の器を指す 書き出しでは使わない
    ("compat/ymm4/effects.py", "_active_report"): "要らない",
    # 退避のファイルの名前に付ける時刻 保存の側だけ
    ("core/io/recovery.py", "_last_stamp"): "要らない",
    # 自己診断が作る Qt のアプリ 子は自分で作る
    ("selfcheck.py", "_application"): "要らない",
}


def test_every_preference_is_decided_for_the_export_child() -> None:
    """壊れると、設定の項目を足した人が子のことを決めずに済ませ、親と子で絵が食い違う"""
    names = {field.name for field in dataclasses.fields(Preferences)}
    groups = (set(BY_RUNTIME), set(BY_EXPORT_SETTINGS), set(NOT_FOR_THE_CHILD))
    undecided = names - set().union(*groups)
    assert not undecided, f"子の書き出しへ渡すかを決めていない設定: {sorted(undecided)}"
    gone = set().union(*groups) - names
    assert not gone, f"もう無い設定が一覧に残っている: {sorted(gone)}"
    for first, second in ((0, 1), (0, 2), (1, 2)):
        assert not groups[first] & groups[second]


def test_the_runtime_carries_exactly_the_process_wide_preferences() -> None:
    """壊れると、プロセスごとに当てる設定を足しても子へ運ぶ欄が無いか、運ぶ欄だけ残る"""
    assert set(BY_RUNTIME.values()) == {f.name for f in dataclasses.fields(AviUtlRuntime)}


def test_export_settings_keep_their_fields() -> None:
    """壊れると、書き出しの設定から欄が消えたのに、渡しているつもりの一覧が残る"""
    fields = {field.name for field in dataclasses.fields(ExportSettings)}
    assert {"pipeline_depth", "decode_threads", "smooth_history", "gpu_color"} <= fields
    assert {"chapters", "encoder_priority", "path"} <= fields


def _module_globals() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith("ui/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                found.update((relative, name) for name in node.names)
    return found


def test_every_module_global_is_decided_for_the_export_child() -> None:
    """壊れると、描く側に大域の値を足した人が子のことを決めずに済ませ、子だけ既定で描く"""
    found = _module_globals()
    undecided = found - set(MODULE_GLOBALS)
    assert not undecided, f"子の書き出しでどうするかを決めていない大域の値: {sorted(undecided)}"
    assert not set(MODULE_GLOBALS) - found, "もう無い大域の値が一覧に残っている"
