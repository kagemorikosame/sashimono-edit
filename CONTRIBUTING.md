# 開発に参加する

Sashimono はまだ **β 版**です 作りが大きく変わることがあります

- **不具合の報告・要望・質問** → [Discussions](../../discussions/new/choose)
  - 動かない・落ちる → 「不具合の報告」
  - AviUtl / YMM4 の配布物が読めない・違って見える → 「互換（AviUtl／YMM4）の報告」
  - こういうことができるようにしてほしい → 「要望」
  - 使い方・どちらとも言えないこと → 「質問」
- **Issue** は開発者が直す作業を管理する置き場です Discussions で確かめた報告を、開発者が Issue に起こします
  （Issue を作る画面を開くと、Discussions のカテゴリへ案内されます）
- **コードを書く** → 以下

**開発ルールの大本は [docs/development.md](docs/development.md)** ここは手順だけ

---

## 1. 環境を作る

Windows 専用です Python 3.12 以上（開発は 3.14）と
[ffmpeg](https://www.gyan.dev/ffmpeg/builds/) が要ります

```
git clone https://github.com/kagemorikosame/sashimono-edit.git
cd sashimono-edit
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

動くか確かめる:

```
.venv\Scripts\python.exe tools\verify.py
.venv\Scripts\python.exe -m sashimono
```

> **`python` ではなく `.venv\Scripts\python.exe` を使ってください**
> 環境によっては素の `python` がランチャースタブで、標準入力から読ませると
> 応答が返らなくなります

字幕起こしと AI 連携は既定では入りません（合計 2 GB を超えるため）
ソフト内の導入ボタンから、必要になった時点で入れてください

---

## 2. 変更を書く

```
git switch -c phase/やりたいこと
```

ブランチの名前:

| | |
|---|---|
| `phase/*` | フェーズ単位のまとまった作業 |
| `fix/*` | 不具合の修正 |

`main` へ直接 push しないでください

書いている間に押さえること（詳しくは [docs/development.md](docs/development.md)）:

- **コメントは日本語で「なぜ」を書く** 何をしているかはコードを読めば分かります
- **文章に句点（まる）を使わない** 文の途中は半角空白で区切ります
  うっかり書いても `tools\punctuation.py --fix` で直せます
- **テストは失敗の仕方を書く** assert が通ることだけを書いたテストは価値が薄い
- **`TODO` を残さない** やり残しや気付いた懸念はできるだけ同じ PR の中で直し、
  直せない物だけ Issue へ
- **互換層は実物で確かめる** 形式の推測だけで書くと必ず外れます

---

## 3. 検証する

出す前に必ず:

```
.venv\Scripts\python.exe tools\verify.py
```

ruff（書式・規約）→ mypy（strict）→ pytest **CI もこれと同じものを走らせます**
ので、手元で通れば CI でも通ります

GPU が無い環境では OpenGL のテストが自動で飛びます ffmpeg が無いときは
素材を使うテストが飛びます どちらも失敗ではありません

---

## 4. PR を出す

**PR はフェーズ単位**です 細かく切りすぎると全体像が見えず、大きすぎると
レビューが成立しません

テンプレートに沿って書いてください とくに:

- **なぜその作りにしたか**（差分を見れば「何をしたか」は分かります）
- `tools/verify.py` の結果（テスト件数）
- 途中で見つけた別の不具合

### AI のレビュー

PR には 4 つの AI（CodeRabbit・Copilot・Sourcery・Codex）がレビューを付けます
自動で走るかどうかは役によって違い、無料枠の回数でも止まるので、PR を出したら Codex 以外の全員に頼んでください
Codex は使う量を抑えるため、CodeRabbit の指摘を直して承認を取ったあと、最後に `@codex review` を 1 回だけ頼みます
頼み直すのは Codex の P0・P1 を直したときだけで、P2 は直して返信すれば締めて構いません
たとえば CodeRabbit は、この公開リポジトリでは自動で走らず、PR に `@coderabbitai review` と
書くと見てくれます ほかの頼み方は
[docs/development.md](docs/development.md) の「AI のレビューを受ける」にあります
どの役にも、このプロジェクトの約束（コメントの書き方、コア層の依存、未対応の記録の仕方など）を
見るように設定してあります

- 指摘は**読んで判断**してください 全部直すのでも、全部無視するのでもありません
- 直さないときは、その理由をコメントに残してください
- 同じ指摘が何役からも来たら、直すのは 1 回で構いません
- 指摘に納得できないときは、そのコメントに返信すると会話できます

### Dependabot の PR

依存の更新は Dependabot が種類（GitHub Actions・pip）ごとにまとめて出します
大きな版上げ（major）と小さい物（minor・patch）は別の PR です
CodeRabbit は契約の席を人にしか割り当てないので、ボットの PR は審査しません
（`.coderabbit.yaml` で黙って飛ばしています） 代わりに次を確かめてマージしてください

- CI がすべて通る
- 変わったのが版の数字（とロックファイル）だけ
- 大きな版上げなら、部品の変更点（リリースノート）に、こちらの使い方が変わる所が無いか読む

大きな版上げのときだけ `@codex review` を 1 回頼みます 詳しくは
[docs/development.md](docs/development.md) の「Dependabot の PR」にあります

---

## ライセンス

MIT です PR を送った時点で、その内容が MIT で配布されることに同意したものと
します

配る zip は、同梱の部品（libx264・libx265）が GPL なので全体として GPL の条件で配ります
新しい依存を足すと zip に積む部品が変わるので、`THIRD_PARTY_NOTICES.md` にも足してください
（足していないと `tools/build_package.py` が zip を作る前に止まります）
