"""ソフトの外へ案内する先（使い方・不具合の受け口）

案内する URL はここ 1 か所に集める メニュー・文言・配る zip の説明書きが
それぞれ URL を書くと、置き場を移したときに 1 つだけ古い先へ飛ばし続ける
（置き場を移したら、ここを差し替えるだけで済む 差し替える所が 1 つなら漏れない）

Qt を読まない 自己診断や配る zip を組み立てる道具からも使えるようにする
"""

from __future__ import annotations

__all__ = [
    "BETA_MANIFEST_URL",
    "DISCUSSION_CATEGORIES",
    "MANUAL_URL",
    "RELEASES_URL",
    "REPORT_URL",
    "REPOSITORY_URL",
    "STABLE_MANIFEST_URL",
]

#: 公開リポジトリ
REPOSITORY_URL = "https://github.com/kagemorikosame/sashimono-edit"

#: 使い方 Wiki のホーム ホームの目次と横の目次から、機能の説明と画像付きの手順書へ
#: たどれる README の操作の節へ飛ばしていたのは Wiki の公開前だけの手当て（公開前の
#: Wiki へ飛ばすと、GitHub は「Wiki を作る」画面を出し、壊れたリンクにしか見えない）
MANUAL_URL = f"{REPOSITORY_URL}/wiki"

#: 不具合・要望・質問の受け口 Discussions のカテゴリを選ぶ画面へ直に飛ばす
#: Issue は開発者が直す作業を管理する置き場にして、使う人の報告は Discussions で受ける
#: （報告は確かめてから Issue に起こす 再現しない物・質問・同じ報告が Issue の一覧に混ざらない）
#: カテゴリを選ぶ画面にするのは、選んだカテゴリの書き込み欄（版や再現手順）が出るため
#: カテゴリを決め打ちにしないので、カテゴリの名前を変えても行き先は壊れない
REPORT_URL = f"{REPOSITORY_URL}/discussions/new/choose"

#: 使う人が書くカテゴリのスラッグ（名前） スラッグ → 画面に出す名前
#: ``.github/DISCUSSION_TEMPLATE/<スラッグ>.yml`` が、そのカテゴリの書き込み欄になる
#: カテゴリは GitHub の画面で作る（``docs/development.md`` の「報告の受け口」）
DISCUSSION_CATEGORIES: dict[str, str] = {
    "bug-reports": "不具合の報告",
    "compatibility": "互換（AviUtl／YMM4）の報告",
    "ideas": "要望",
    "q-a": "質問",
}

#: 配布物の一覧 自動で入れ替えられないとき（置き場に書けない・目録が新しすぎる）に案内する
RELEASES_URL = f"{REPOSITORY_URL}/releases"

#: 自動更新の目録（正式版） **配った版が読み続ける契約なので変えない**
#: ``releases/latest/download/…`` は、プレリリースでない最新のリリースの同じ名前の資産へ
#: 転送される REST API（未認証で 1 時間 60 回）を使わないので、共有回線でも数え切られない
STABLE_MANIFEST_URL = f"{RELEASES_URL}/latest/download/update.json"

#: 自動更新の目録（ベータ） ``latest`` はプレリリースを飛ばすので、動かすタグ ``beta`` に充てる
BETA_MANIFEST_URL = f"{RELEASES_URL}/download/beta/update.json"
