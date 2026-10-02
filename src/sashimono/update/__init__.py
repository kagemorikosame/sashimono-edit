r"""自動更新（Issue #36）

配った zip（``Sashimono.exe``）が、自分で新しい版へ入れ替わるための部品
Qt を読まない 起動の頭（:mod:`sashimono.app`）・自己診断・道具（``tools/update_*.py``）からも使う

流れ

1. 目録（``update.json``）と署名（``update.json.sig``）を GitHub Releases の固定の URL から読む
   （:mod:`.check`） REST API は叩かない
2. 署名を埋め込んだ公開鍵で確かめる（:mod:`.signing`） 確かめられなければ何もしない
3. 新しい版の zip を落とし、目録の大きさと SHA-256 で確かめ、隣のフォルダへ展開する
   （:mod:`.package`）
4. 本体を終えてから、別のプログラム（PowerShell）が改名でフォルダを入れ替え、新しい版を
   起こす 起動できなければ前の版へ戻す（:mod:`.swap`）

設計は計画書の F-12 決めごとと手順は ``docs/development.md`` の「自動更新」
"""

from __future__ import annotations
