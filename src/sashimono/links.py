"""ソフトの外へ案内する先（使い方・不具合の受け口）

案内する URL はここ 1 か所に集める メニュー・文言・配る zip の説明書きが
それぞれ URL を書くと、置き場を移したときに 1 つだけ古い先へ飛ばし続ける
（Wiki を公開したら「使い方」の先を差し替える 差し替える所が 1 つなら漏れない）

Qt を読まない 自己診断や配る zip を組み立てる道具からも使えるようにする
"""

from __future__ import annotations

__all__ = [
    "BETA_MANIFEST_URL",
    "MANUAL_URL",
    "RELEASES_URL",
    "REPORT_URL",
    "REPOSITORY_URL",
    "STABLE_MANIFEST_URL",
]

#: 公開リポジトリ
REPOSITORY_URL = "https://github.com/kagemorikosame/sashimono-edit"

#: 使い方 Wiki が公開されるまでは README の操作の節へ飛ばす
#: 公開前の Wiki へ飛ばすと、GitHub は「Wiki を作る」画面か空のページを出し、
#: 使う人には壊れたリンクにしか見えない 公開したら ``f"{REPOSITORY_URL}/wiki"`` へ替える
MANUAL_URL = f"{REPOSITORY_URL}#主な操作"

#: 不具合・要望の受け口 雛形を選ぶ画面へ直に飛ばす
#: 白紙の Issue へ飛ばすと、版や再現手順の欄の無い報告になり、聞き返しから始まる
REPORT_URL = f"{REPOSITORY_URL}/issues/new/choose"

#: 配布物の一覧 自動で入れ替えられないとき（置き場に書けない・目録が新しすぎる）に案内する
RELEASES_URL = f"{REPOSITORY_URL}/releases"

#: 自動更新の目録（正式版） **配った版が読み続ける契約なので変えない**
#: ``releases/latest/download/…`` は、プレリリースでない最新のリリースの同じ名前の資産へ
#: 転送される REST API（未認証で 1 時間 60 回）を使わないので、共有回線でも数え切られない
STABLE_MANIFEST_URL = f"{RELEASES_URL}/latest/download/update.json"

#: 自動更新の目録（ベータ） ``latest`` はプレリリースを飛ばすので、動かすタグ ``beta`` に充てる
BETA_MANIFEST_URL = f"{RELEASES_URL}/download/beta/update.json"
