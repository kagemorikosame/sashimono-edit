"""操作の台帳 画面で人ができる操作と、AI の道具の対応（#347）

方針は「画面で人ができる操作は、AI アシスタントも全部できる」（#298 の追記） 足りない道具を
場当たりではなく、この一覧から抜けを数えて埋める（互換と同じ「数えて、多い順に埋める」）

行は画面の操作 1 つ ``key`` はその在りかで、試験（``tests/ui/test_operation_ledger.py``）が
本物の窓から取り出す名前と同じ書き方にする

- ``メニュー/<メニュー>/<項目>`` メニューバー（``MainWindow._actions`` の名前そのまま）
- ``タイムラインの右クリック/<項目>`` サブメニューは ``/`` でつなぐ トラックの名前は
  ``〔トラック〕`` に置き換え、何本か選んだときの「（3 本）」は外す
- ``設定パネル/<字>｜<説明の頭>`` パネルのボタンは字と説明の頭をつなぐ（同じ「有効」が
  別の操作に付いているため） 値の欄は 1 つずつ数えず ``設定パネル/欄の値`` の 1 行
- ``*`` を含む行は一覧（エフェクト・エイリアス・シーンなど データで中身が変わる物）を
  まとめて受ける

道具を足す PR は、ここにも行を足す（``same`` か ``partial`` へ書き換える） 足さないと
``tests/ai/test_operation_ledger.py`` が落ちる 画面に項目を足す PR も同じで、行が無いと
``tests/ui/test_operation_ledger.py`` が落ちる 1 行 1 操作なので、並行して進む PR どうしで
同じ行を書き換えない限り、取り込むときにぶつからない
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from sashimono.ai.safety import Safety

__all__ = [
    "AI_ONLY",
    "CHILD_ISSUES",
    "DECIDED_READ_ONLY",
    "LEDGER",
    "READING",
    "Entry",
    "Safety",
    "Status",
    "counts",
    "hidden",
    "missing",
    "partial",
    "read_only",
    "same",
]

#: 足りない道具を足す先の Issue #298 の子（A1〜E6） 台帳の ``issue`` はこの範囲から選ぶ
#: 範囲の外の番号を書くと、どの段で足すのかが親の Issue から辿れなくなる
CHILD_ISSUES = range(319, 353)

#: AI に出さない線引きを決める Issue（E4）
_LINE_DRAWING = 350

#: 利用者が決めた線引き（2026-10-14 #298 #350 のコメント）
#: アプリ自身の操作（設定・更新・画面配置・ショートカット・ヘルプ）は、AI は読むだけ
#: 変えるのは人だけ
DECIDED_READ_ONLY = "2026-10-14 に決めた: アプリ自身の操作は AI は読むだけ（#350）"


class Status(Enum):
    """AI からどこまでできるか"""

    #: 同じ結果になる道具がある
    SAME = "AI でできる"
    #: 道具はあるが、画面でできることの一部だけ（足りない所は ``note`` に書く）
    PARTIAL = "一部だけできる"
    #: 道具が無い 足す先の Issue を ``issue`` に書く
    MISSING = "まだできない"
    #: アプリ自身の操作 AI は値を読めるが、変えるのは人だけ **利用者が決めた**
    #: （:data:`DECIDED_READ_ONLY`） 読む道具ができたら ``tools`` に書く
    READ_ONLY = "AI は読むだけ（決定）"
    #: AI に出さない **候補** 上の決定に当たらず、利用者がまだ決めていない
    #: 決まるまでは「出さない」と言い切らず、理由の案だけを ``note`` に持つ
    HIDDEN = "出さない候補"


@dataclass(frozen=True, slots=True)
class Entry:
    """画面の操作 1 つ"""

    #: 画面の在りか（上の書き方） ``*`` は一覧をまとめて受ける
    key: str
    status: Status
    #: 対応する AI の道具の名前（``ai/operations.py`` の ``Operation.name``）
    tools: tuple[str, ...] = ()
    #: 足す・決める先の Issue（#298 の子）
    issue: int | None = None
    #: 足りない所、出さない理由の案、など
    note: str = ""
    #: この画面の操作の区分（#350 の 読むだけ・取り消せる・取り消せない） 省くと
    #: :attr:`expected_safety` の決まりで決める 対応する道具はこの区分と同じでなければならない
    #: 取り消せない操作（保存・書き出し）を取り消せる道具で受けると、確かめが省かれる
    safety: Safety | None = None

    @property
    def expected_safety(self) -> Safety | None:
        """対応する道具が持つべき区分 出さない候補は線引きが未決なので ``None``

        アプリ自身の操作は決定どおり読むだけ 書いていない編集の操作は取り消せる編集
        表示や再生位置だけを変える操作と、取り消せない操作は、行に区分を書く
        """
        if self.safety is not None:
            return self.safety
        if self.status is Status.READ_ONLY:
            return Safety.READ_ONLY
        if self.status is Status.HIDDEN:
            return None
        return Safety.UNDOABLE


def same(key: str, *tools: str, note: str = "", safety: Safety | None = None) -> Entry:
    return Entry(key, Status.SAME, tools, note=note, safety=safety)


def partial(key: str, *tools: str, issue: int, note: str, safety: Safety | None = None) -> Entry:
    return Entry(key, Status.PARTIAL, tools, issue, note, safety)


def missing(key: str, issue: int, note: str = "", safety: Safety | None = None) -> Entry:
    return Entry(key, Status.MISSING, (), issue, note, safety)


#: アプリ自身の設定を読む道具（#354） 設定・ショートカット・画面配置・版・ヘルプを返す
_APP_SETTINGS = "get_app_settings"


def read_only(key: str, what: str) -> Entry:
    """アプリ自身の操作 AI は読むだけ（決定） 値は :data:`_APP_SETTINGS` で読む"""
    return Entry(
        key, Status.READ_ONLY, (_APP_SETTINGS,), _LINE_DRAWING, f"{what} {DECIDED_READ_ONLY}"
    )


def hidden(key: str, note: str) -> Entry:
    """出さない候補 線引きは #350（E4）で決める"""
    return Entry(key, Status.HIDDEN, (), _LINE_DRAWING, note)


# 決定に当たる操作の種類（:func:`read_only` の ``what``）
_SETTINGS = "設定"
_LAYOUT = "画面配置（パネルの出し入れ・並べ方）"
_SHORTCUTS = "ショートカット"
_UPDATE = "更新"
_HELP = "ヘルプ"
# まだ決めていない操作の理由の案（#298 の表の「AI には出さない候補」）
_OUTSIDE = "エクスプローラーを開くだけで、プロジェクトを変えない 決定の 5 つに当たらず未決"
#: 表示だけの操作を AI に出すかは #298 の「決めること」の 3 つ目 決まるまでは足す側に数える
_VIEW = "表示だけの操作 出す意味があるかは未決（#298 の決めること）"

LEDGER: tuple[Entry, ...] = (
    # --- メニュー ファイル ---
    missing("メニュー/ファイル/新規", 349, safety=Safety.IRREVERSIBLE),
    missing("メニュー/ファイル/開く…", 349, safety=Safety.IRREVERSIBLE),
    missing("メニュー/ファイル/保存", 327, safety=Safety.IRREVERSIBLE),
    missing("メニュー/ファイル/名前を付けて保存…", 327, safety=Safety.IRREVERSIBLE),
    hidden("メニュー/ファイル/バックアップのフォルダを開く", _OUTSIDE),
    partial(
        "メニュー/ファイル/プロジェクト設定…",
        "set_resolution",
        issue=349,
        note="解像度だけ fps・音の設定・背景などは無い",
    ),
    missing("メニュー/ファイル/置き方の方式を切り替える…", 349),
    same("メニュー/ファイル/素材を読み込む…", "import_media"),
    missing("メニュー/ファイル/書き出し…", 327, safety=Safety.IRREVERSIBLE),
    missing(
        "メニュー/ファイル/静止画を保存…",
        327,
        "絵を見るだけなら preview_frame で足りる",
        Safety.IRREVERSIBLE,
    ),
    hidden("メニュー/ファイル/終了", "アプリを閉じる 編集ではない 決定の 5 つに当たらず未決"),
    # --- メニュー 編集 ---
    same("メニュー/編集/元に戻す", "undo"),
    missing("メニュー/編集/やり直す", 333),
    same("メニュー/編集/再生ヘッドで分割", "split_clip"),
    same("メニュー/編集/削除", "delete_clips", "delete_clip"),
    same("メニュー/編集/削除して詰める", "delete_clips", "delete_clip"),
    partial(
        "メニュー/編集/コピー",
        "duplicate_clips",
        issue=349,
        note="写しと貼り付けが 1 つにまとまっている クリップボードは使わない",
    ),
    missing("メニュー/編集/切り取り", 349),
    partial(
        "メニュー/編集/貼り付け（再生ヘッドの位置）",
        "duplicate_clips",
        issue=349,
        note="コピーと同じ 1 つの道具 アプリのクリップボードからは貼れない",
    ),
    missing("メニュー/編集/貼り付け（挿入）", 349),
    missing(
        "メニュー/編集/すべて選択",
        349,
        "select_clips に全部の ID を渡せば同じになる",
        Safety.READ_ONLY,
    ),
    missing(
        "メニュー/編集/静止画をコピー",
        349,
        "プロジェクトを変えない（絵をクリップボードへ写す）",
        Safety.READ_ONLY,
    ),
    missing("メニュー/編集/書き出し範囲を解除", 349),
    same("メニュー/編集/グループ化", "group_clips"),
    same("メニュー/編集/グループ解除", "ungroup_clips"),
    same("メニュー/編集/トラックをミュート", "set_track_state"),
    same("メニュー/編集/トラックをソロ", "set_track_state"),
    same("メニュー/編集/トラックをロック", "set_track_state"),
    # --- メニュー オブジェクト ---
    same("メニュー/オブジェクト/テキストを追加", "add_text"),
    same("メニュー/オブジェクト/図形を追加", "add_shape"),
    same("メニュー/オブジェクト/場面切り替えを追加", "add_transition"),
    missing("メニュー/オブジェクト/フィルタを追加", 349),
    missing("メニュー/オブジェクト/プリセットとエイリアス…", 328),
    # --- メニュー シーン ---
    same("メニュー/シーン/新しいシーン…", "add_scene"),
    same("メニュー/シーン/シーンを置く…", "place_scene"),
    missing("メニュー/シーン/シーンの名前を変更…", 349),
    missing("メニュー/シーン/シーンを削除", 349),
    same("メニュー/シーン/メインに戻る", "set_active_scene", safety=Safety.READ_ONLY),
    # --- メニュー 字幕 ---
    read_only("メニュー/字幕/字幕パネル", _LAYOUT),
    same("メニュー/字幕/起こす…", "transcribe"),
    same("メニュー/字幕/整形…", "clean_subtitles"),
    same("メニュー/字幕/無音カット…", "jet_cut"),
    same("メニュー/字幕/焼き込み", "place_subtitles"),
    missing(
        "メニュー/字幕/書き出し…", 327, "字幕のファイル（SRT など）へ書き出す", Safety.IRREVERSIBLE
    ),
    # --- メニュー 互換 ---
    missing("メニュー/互換/オブジェクトを読み込む…", 328, ".exo .exa .ymmp の部品"),
    missing("メニュー/互換/テンプレート…", 328),
    missing(
        "メニュー/互換/スクリプトを読み直す",
        349,
        "置き場の AviUtl スクリプトを読み直すだけで、プロジェクトを変えない",
        Safety.READ_ONLY,
    ),
    hidden("メニュー/互換/スクリプトフォルダを開く", _OUTSIDE),
    hidden(
        "メニュー/互換/exe の隣のスクリプトを移す…",
        "アプリの置き場の整理で、編集ではない 更新のときに勧める物だが、更新に数えるかは未決",
    ),
    missing(
        "メニュー/互換/互換性レポート…",
        352,
        "読めなかった物を AI も読めるようにする",
        Safety.READ_ONLY,
    ),
    # --- メニュー AI・表示・再生・ヘルプ ---
    read_only("メニュー/AI/アシスタント", _LAYOUT),
    missing("メニュー/表示/拡大", 350, _VIEW, Safety.READ_ONLY),
    missing("メニュー/表示/縮小", 350, _VIEW, Safety.READ_ONLY),
    missing("メニュー/表示/全体を表示", 350, _VIEW, Safety.READ_ONLY),
    missing("メニュー/表示/グラフエディタ", 350, _VIEW, Safety.READ_ONLY),
    same("メニュー/表示/トラックを高く", "set_track_height"),
    same("メニュー/表示/トラックを低く", "set_track_height"),
    same("メニュー/表示/トラックの高さを戻す", "set_track_height"),
    read_only("メニュー/表示/画面配置を初期に戻す", _LAYOUT),
    read_only("メニュー/表示/ショートカットの設定…", _SHORTCUTS),
    read_only("メニュー/表示/設定…", _SETTINGS),
    missing(
        "メニュー/再生/再生 / 停止",
        338,
        "AI は動きを見られない 範囲の連続フレームで確かめる",
        Safety.READ_ONLY,
    ),
    read_only("メニュー/ヘルプ/使い方", _HELP),
    read_only("メニュー/ヘルプ/不具合・要望を送る", _HELP),
    read_only("メニュー/ヘルプ/更新を確かめる…", _UPDATE),
    read_only("メニュー/ヘルプ/前の版に戻す…", _UPDATE),
    read_only("メニュー/ヘルプ/バージョン情報…", _HELP),
    # --- タイムラインの右クリック クリップの上 ---
    same("タイムラインの右クリック/再生ヘッドで分割", "split_clip"),
    partial(
        "タイムラインの右クリック/コピー",
        "duplicate_clips",
        issue=349,
        note="写しと貼り付けが 1 つにまとまっている",
    ),
    missing("タイムラインの右クリック/切り取り", 349),
    partial(
        "タイムラインの右クリック/貼り付け（再生ヘッドの位置）",
        "duplicate_clips",
        issue=349,
        note="アプリのクリップボードからは貼れない",
    ),
    missing("タイムラインの右クリック/挿入して貼り付け", 349),
    same("タイムラインの右クリック/削除", "delete_clips", "delete_clip"),
    same("タイムラインの右クリック/削除して詰める", "delete_clips", "delete_clip"),
    same("タイムラインの右クリック/グループ化", "group_clips"),
    same("タイムラインの右クリック/グループ解除", "ungroup_clips"),
    # 音量とミックス（#329） 2026-10-17 に利用者が右クリックに足すと決めた 人も AI も
    # core/mixing.py の同じ計算を通る
    same("タイムラインの右クリック/BGM を声に合わせて下げる…", "duck_audio"),
    same("タイムラインの右クリック/音量を基準にそろえる…", "normalize_loudness"),
    same("タイムラインの右クリック/シーンを開く", "set_active_scene", safety=Safety.READ_ONLY),
    same("タイムラインの右クリック/エフェクトを追加/*/*", "add_effect"),
    missing("タイムラインの右クリック/エイリアスとして保存…", 328),
    missing("タイムラインの右クリック/線に不透明度を出す", 350, _VIEW, Safety.READ_ONLY),
    missing("タイムラインの右クリック/線に音量を出す", 350, _VIEW, Safety.READ_ONLY),
    # --- タイムラインの右クリック 空いた所 ---
    missing("タイムラインの右クリック/フィルタを置く", 349),
    same("タイムラインの右クリック/追加/テキスト", "add_text"),
    same("タイムラインの右クリック/追加/図形", "add_shape"),
    same("タイムラインの右クリック/追加/場面切り替え", "add_transition"),
    missing("タイムラインの右クリック/追加/フィルタ", 349),
    missing("タイムラインの右クリック/追加/グループ制御", 349),
    missing("タイムラインの右クリック/追加/カスタムオブジェクト/*", 349, "AviUtl の .obj を置く"),
    missing(
        "タイムラインの右クリック/追加/エイリアス/管理…（見本で選ぶ・名前や分類を変える・消す）",
        328,
    ),
    missing(
        "タイムラインの右クリック/追加/エイリアス/*/*",
        328,
        "保存したエイリアスと、AviUtl・YMM4 のテンプレートの棚",
    ),
    missing(
        "タイムラインの右クリック/追加/エイリアス/保存したもの/*/*",
        328,
        "保存したエイリアスの分類が 2 つ以上あると、分類ごとの段に並ぶ",
    ),
    same("タイムラインの右クリック/追加/シーン/*", "place_scene"),
    # --- タイムラインの右クリック トラック・範囲 ---
    same("タイムラインの右クリック/〔トラック〕 をミュート", "set_track_state"),
    same("タイムラインの右クリック/〔トラック〕 をソロ", "set_track_state"),
    same("タイムラインの右クリック/〔トラック〕 をロック", "set_track_state"),
    same("タイムラインの右クリック/〔トラック〕 の高さを戻す", "set_track_height"),
    missing("タイムラインの右クリック/〔トラック〕 の名前を変更…", 349),
    missing("タイムラインの右クリック/範囲を解除", 349, "書き出し範囲（ワークエリア）"),
    same("タイムラインの右クリック/トラックを追加/映像トラック", "add_track"),
    same("タイムラインの右クリック/トラックを追加/音声トラック", "add_track"),
    partial(
        "タイムラインの右クリック/トラックを追加/エフェクトトラック（フィルタ用）",
        "add_track",
        issue=349,
        note="映像トラックは足せるが、フィルタ用（FX と番号の名前）を選ぶ引数が無い",
    ),
    same("タイムラインの右クリック/レイヤーを追加", "add_track"),
    missing("タイムラインの右クリック/トラックを削除（〔トラック〕）", 349),
    # --- タイムラインの〔＋ トラック追加〕 ---
    same("タイムラインの〔＋ トラック追加〕/映像トラック", "add_track"),
    same("タイムラインの〔＋ トラック追加〕/音声トラック", "add_track"),
    partial(
        "タイムラインの〔＋ トラック追加〕/エフェクトトラック（フィルタ用）",
        "add_track",
        issue=349,
        note="フィルタ用を選ぶ引数が無い",
    ),
    same("タイムラインの〔＋ トラック追加〕/レイヤー", "add_track"),
    # --- プレビューの右クリック ---
    missing(
        "プレビューの右クリック/静止画を保存…",
        327,
        "絵を見るだけなら preview_frame で足りる",
        Safety.IRREVERSIBLE,
    ),
    missing(
        "プレビューの右クリック/静止画をコピー",
        349,
        "プロジェクトを変えない（絵をクリップボードへ写す）",
        Safety.READ_ONLY,
    ),
    # --- シーンの帯 ---
    same("シーンの帯/新規｜空のシーンを作って開く", "add_scene"),
    missing("シーンの帯/名前｜開いているシーンの名前を変える", 349),
    missing("シーンの帯/削除｜開いているシーンを消す（どこにも置かれていないときだけ）", 349),
    same(
        "シーンの帯/置く｜ほかのシーンを、再生ヘッドの位置へ 1 本のクリップとして置く",
        "place_scene",
    ),
    read_only("シーンの帯/磁石（吸着）", _SETTINGS),
    read_only(
        "シーンの帯/再生ヘッドの吸い付き（目盛りで動かすとき 掴んだ後に Shift で一時的に外す）",
        _SETTINGS,
    ),
    # --- 再生の操作 ---
    same("再生の操作/先頭へ (Home)", "seek", safety=Safety.READ_ONLY),
    same("再生の操作/1 フレーム戻る (←)", "seek", safety=Safety.READ_ONLY),
    missing(
        "再生の操作/再生 / 停止 (Space)",
        338,
        "AI は動きを見られない 範囲の連続フレームで確かめる",
        Safety.READ_ONLY,
    ),
    same("再生の操作/1 フレーム進む (→)", "seek", safety=Safety.READ_ONLY),
    same("再生の操作/末尾へ (End)", "seek", safety=Safety.READ_ONLY),
    # --- 素材一覧 ---
    same("素材一覧/読み込み…", "import_media"),
    same("素材一覧/タイムラインへ", "place_media"),
    read_only("素材一覧/一覧｜名前・長さ・大きさを 1 行ずつ並べる", _SETTINGS),
    read_only("素材一覧/アイコン｜サムネイルを大きく並べる", _SETTINGS),
    same("素材一覧の右クリック/タイムラインへ置く", "place_media"),
    same("素材一覧の右クリック/字幕を起こす…", "transcribe"),
    hidden("素材一覧の右クリック/ファイルの場所を開く", _OUTSIDE),
    missing("素材一覧の右クリック/プールから外す", 349),
    # --- 字幕パネル ---
    same("字幕パネル/起こす…", "transcribe"),
    same("字幕パネル/整形…", "clean_subtitles"),
    same("字幕パネル/無音カット…", "jet_cut"),
    same("字幕パネル/焼き込み", "place_subtitles"),
    same(
        "字幕パネル/選んだ行を置く｜選んだ字幕の行をテキストとしてタイムラインへ置く"
        " 選んでいなければ再生位置の 1 行",
        "place_subtitles",
    ),
    missing(
        "字幕パネル/書き出し…", 327, "字幕のファイル（SRT など）へ書き出す", Safety.IRREVERSIBLE
    ),
    same("字幕パネルの右クリック/ここへ移動", "seek", safety=Safety.READ_ONLY),
    partial(
        "字幕パネルの右クリック/文字をコピー",
        "get_subtitles",
        issue=349,
        note="本文は読めるが、クリップボードへは写さない",
        safety=Safety.READ_ONLY,
    ),
    same("字幕パネルの右クリック/テキストとして置く", "place_subtitles"),
    missing("字幕パネルの右クリック/再生ヘッドで分割", 349),
    missing("字幕パネルの右クリック/次と結合", 349),
    missing("字幕パネルの右クリック/削除", 349),
    # --- 設定パネル ---
    same(
        "設定パネル/欄の値",
        "set_param",
        "add_keyframe",
        "set_volume",
        note=(
            "中身・エフェクト・縁取りの層の値 クリップ自身の欄は下の行（set_clip_property）"
            " 音量とパンの欄は set_volume でも変えられる（何本かまとめて dB で動かす #329）"
        ),
    ),
    # クリップ自身の欄 SetClipProperty で変わる set_param では型が合わない（再生速度は分数）
    same("設定パネル/クリップ自身の欄/blend_mode", "set_clip_property", note="合成モード"),
    same("設定パネル/クリップ自身の欄/speed", "set_clip_property", note="再生速度"),
    same("設定パネル/クリップ自身の欄/source_in", "set_clip_property", note="再生開始位置"),
    same("設定パネル/クリップ自身の欄/clip_to_below", "set_clip_property", note="クリッピング"),
    same("設定パネル/クリップ自身の欄/native_size", "set_clip_property", note="画素で置く"),
    same("設定パネル/エフェクトを追加…", "add_effect"),
    missing(
        "設定パネル/プリセット…｜選んでいるクリップの見た目を保存し、ほかのクリップへ当てる", 328
    ),
    missing(
        "設定パネル/有効｜この組の欄を掛けるかどうか 無効にすると既定の置き方・鳴り方に戻る", 349
    ),
    missing("設定パネル/有効｜掛ける前と後を見比べる", 349, "エフェクトの有効の切り替え"),
    missing("設定パネル/▲｜順番を変える 掛ける順で結果が変わる", 349, "エフェクトの並べ替え"),
    missing("設定パネル/▼｜順番を変える 掛ける順で結果が変わる", 349, "エフェクトの並べ替え"),
    missing("設定パネル/✕｜このエフェクトを外す", 349),
    partial(
        "設定パネル/◆｜再生位置のキーを打つ・消す",
        "add_keyframe",
        issue=349,
        note="打てるが消せない",
    ),
    partial(
        "設定パネル/◀｜前のキーへ再生位置を動かす",
        "seek",
        issue=349,
        note="キーの位置が分かれば seek で動ける キーへ飛ぶ道具は無い",
        safety=Safety.READ_ONLY,
    ),
    partial(
        "設定パネル/▶｜次のキーへ再生位置を動かす",
        "seek",
        issue=349,
        note="キーの位置が分かれば seek で動ける キーへ飛ぶ道具は無い",
        safety=Safety.READ_ONLY,
    ),
    same("設定パネル/解除", "set_clip_property", note="止めた絵を解く（hold_at を null）"),
    partial(
        "設定パネル/組み方を「標準」にする｜組み方を「標準」に変えて、この欄を使えるようにします"
        "（取り消しで戻せます）",
        "set_param",
        issue=349,
        note=(
            "AviUtl2 の組み方のテキストの灰色の欄に出る（#301） set_param で layout と anchor を"
            "変えれば同じ値になるが、2 回の取り消しになり、選んだ何本かをまとめて切り替えられない"
        ),
    ),
    same(
        "設定パネル/＋ 縁取りを追加｜今の縁の外側に縁取りを足します"
        " 上の層ほど手前に描き、太さは字の輪郭から数えます",
        "add_stroke",
    ),
    same("設定パネル/有効｜この縁取りを描くかどうか 消さずに隠して見比べる", "set_stroke_enabled"),
    same("設定パネル/▲｜上へ（手前に描く）", "move_stroke"),
    same("設定パネル/▼｜下へ（奥に描く）", "move_stroke"),
    same(
        "設定パネル/効果｜この縁取りだけにエフェクトを掛ける（ぼかし・グロー・色など）",
        "add_effect",
    ),
    same("設定パネル/✕｜この縁取りを外す 掛けたエフェクトも一緒に外れる", "remove_stroke"),
    *(
        partial(
            f"設定パネル/{arrow}｜{place}へ揃える（見えている範囲の端を画面の端へ付ける）",
            "set_param",
            issue=349,
            note="位置は set_param で変えられるが、見えている範囲の端へ揃える計算は無い",
        )
        for arrow, place in (
            ("↖", "左上"),
            ("↑", "上の中央"),
            ("↗", "右上"),
            ("←", "左の中央"),
            ("●", "中央"),
            ("→", "右の中央"),
            ("↙", "左下"),
            ("↓", "下の中央"),
            ("↘", "右下"),
        )
    ),
    same("設定パネルの〔エフェクトを追加…〕/*/*", "add_effect"),
    partial(
        "設定パネルの〔エフェクトを追加…〕/前の場面へ/*/*",
        "add_effect",
        issue=349,
        note="場面切り替えの前と後のどちらの場面へ積むかを選べない",
    ),
    partial(
        "設定パネルの〔エフェクトを追加…〕/後の場面へ/*/*",
        "add_effect",
        issue=349,
        note="場面切り替えの前と後のどちらの場面へ積むかを選べない",
    ),
    missing("設定パネルの〔プリセット…〕/この見た目を保存…", 328),
    missing("設定パネルの〔プリセット…〕/管理…（見本で選ぶ・名前や分類を変える・消す）", 328),
    missing("設定パネルの〔プリセット…〕/*/*", 328, "保存したプリセットを当てる（分類ごとの一覧）"),
    same("設定パネルの縁取りの〔効果〕/*/*", "add_effect", note="stroke_id を渡す"),
    missing("設定パネルの ◆ の右クリック/アニメーションを解除", 349, "キーフレームを消す"),
    missing("設定パネルの ◆ の右クリック/グラフエディタで開く", 350, _VIEW, Safety.READ_ONLY),
)

#: 画面のメニュー・右クリック・ボタンに当たらない道具と、その訳 道具を足して、台帳の
#: どの行にも当たらないときはここへ書く 書かないと ``tests/ai/test_operation_ledger.py`` が落ちる
READING = "人が画面を見て分かることを読む道具"
_GESTURE = "画面ではドラッグや表の書き込みで行う 台帳はメニュー・右クリック・ボタンだけを数える"
AI_ONLY: dict[str, str] = {
    "get_project": READING,
    "list_media": READING,
    "list_tracks": READING,
    "list_clips": READING,
    "get_selection": READING,
    "list_effects": READING,
    "get_history": READING,
    "preview_frame": READING,
    # 完成物の検査（#337） 人は再生して目と耳で確かめる 画面に同じ入口は無い
    # ステータスバーの〔止める〕は走っている検査を止めるだけで、検査を始める操作ではない
    "check_output": READING,
    # 場面の切り替わり（#319） 人は再生して目で見つける物を、フレーム間の差の一覧で読む
    # 読むだけ（Safety.READ_ONLY）で、画面のメニュー・右クリック・ボタンに同じ入口は無い
    "detect_scenes": READING,
    "list_scenes": READING,
    "transcription_status": READING,
    "select_clip": _GESTURE,
    "select_clips": _GESTURE,
    "trim_clip": _GESTURE,
    "move_clip": _GESTURE,
    "move_clips": _GESTURE,
    "set_subtitle_text": _GESTURE,
    "replace_subtitle_text": "字幕と、字幕から置いたテキストをまとめて直す 画面に同じ入口は無い",
    "ask_user": "AI から利用者へ尋ねる",
    # BGM と効果音を置く（#329） 画面では素材をドラッグして置き、右端を詰める
    "place_sound": _GESTURE,
}


def counts() -> dict[Status, int]:
    """状態ごとの行の数 PR の本文と #298 の棚卸しに書く数"""
    tally = Counter(entry.status for entry in LEDGER)
    return {status: tally[status] for status in Status}
