# Code signing policy（コード署名の方針）

[English](#english) | [日本語](#日本語)

---

## English

Free code signing provided by [SignPath.io](https://about.signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).

> Status: the project is preparing its application to the SignPath Foundation.
> Releases up to and including 0.1.4 are **not** signed.
> This page will be updated when signed releases begin.

### What is signed

- Only `Sashimono.exe`, the program of Sashimono Edit, inside the release zip
  (`SashimonoEdit-<version>-windows-x64.zip` on [GitHub Releases](https://github.com/kagemorikosame/sashimono-edit/releases)).
- `Sashimono.exe` is built by GitHub Actions on `windows-latest` from the source code of this
  repository (`.github/workflows/release.yml` calls `.github/workflows/package.yml` when a `v*` tag
  is pushed). The PyInstaller bootloader is compiled from source in the same job; it is not taken
  from the prebuilt PyPI wheel.
- Third-party files in the zip (Python, Qt / PySide6, FFmpeg, and other DLLs and libraries listed in
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)) are **not** signed by this project.
- Nothing built on a developer machine is signed.

### Team roles

Sashimono Edit is developed by a single maintainer. All roles are held by the repository owner.

| Role | Member |
|---|---|
| Committers and reviewers | [kagemorikosame](https://github.com/kagemorikosame) (repository owner) |
| Approvers | [kagemorikosame](https://github.com/kagemorikosame) (repository owner) |

- By project rule, changes reach `main` through pull requests (branch protection requires the CI
  checks to pass). Pull requests from anyone else (including Dependabot) are reviewed and merged by
  the owner.
- Every signing request is approved manually by the approver, once per release.
- Everyone with one of the roles uses multi-factor authentication on GitHub and on SignPath.

### Privacy policy

Sashimono Edit does not send any information to other networked systems unless the user asks for
it or turns on a feature that does so. The program does not collect usage statistics, crash
reports, or identifiers. The only network connections it makes are listed below.

| Feature | When it connects | Where | What is sent | How to turn it off |
|---|---|---|---|---|
| Update check (release zip only) | At startup, at most once every 6 hours, and when the user chooses Help → Check for updates (〔ヘルプ〕→〔更新を確かめる…〕) | `https://github.com/kagemorikosame/sashimono-edit/releases/latest/download/update.json` and `update.json.sig` (and `.../releases/download/beta/update.json` when "receive beta versions" is on). Redirects are followed only to `github.com`, `objects.githubusercontent.com` and `release-assets.githubusercontent.com` | Only plain HTTPS GET requests. The request carries the User-Agent `SashimonoEdit/<version> (auto-update)`. No project data, file names, settings or identifiers are sent. As with any HTTPS connection, GitHub can see the IP address | View → Settings… (〔表示〕→〔設定…〕), uncheck "Check for new versions at startup" (起動したときに新しい版を確かめる). Help → Check for updates still works on request |
| Update download | When the check finds a newer version | The release zip on GitHub Releases (same hosts as above) | Nothing besides the download request. The zip is verified (Ed25519 signature of `update.json` and SHA-256 of the zip) before use. By default the user is asked before it is installed | Same setting as the update check |
| Installing optional components (speech-to-text, AI assistant) | Only when the user presses the install button in the app. The command is shown before it runs | The Python Package Index (`pypi.org`, `files.pythonhosted.org`) through pip | The package download requests. pip adds its standard User-Agent (pip, Python and operating system versions) | Do not press the install button. Nothing is installed by default |
| Speech-to-text models | The first time the user starts a transcription with a model that is not yet downloaded | Hugging Face Hub (`huggingface.co`), through faster-whisper and huggingface_hub; stored in `%USERPROFILE%\.cache\huggingface\hub` (or `HF_HOME`) | Only the model download requests with the standard headers of the download library. **Audio is transcribed locally and never uploaded** | Do not use speech-to-text |
| AI assistant | Only after the user installs it and sends a message in the AI panel | Anthropic, through Claude Code (bundled with the Claude Agent SDK), using the user's own Claude sign-in or API key | The user's message, the assistant's instructions, and the results of the editing tools the model calls: project information (settings, tracks, clips, effects, subtitle text, names and full paths of media files), preview frames rendered as images, and transcription results. Processing by Anthropic is covered by Anthropic's terms and privacy policy. Claude Code's built-in tools (file access, shell, web) are disabled. Sashimono Edit does not read or store the credentials | Do not install the AI assistant, or do not send messages |

Links in the Help menu (manual, reports) and in update messages (release notes, releases page) open
in the user's web browser; the program itself does not connect when they are opened.
Help → Roll back to the previous version (〔ヘルプ〕→〔前の版に戻す…〕) uses the copy kept on the
computer and does not connect.

---

## 日本語

Free code signing provided by [SignPath.io](https://about.signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).

（SignPath.io の無料のコード署名を使い、証明書は SignPath Foundation が持つ）

> いまは SignPath Foundation への申し込みを準備している 0.1.4 までの版は署名していない
> 署名した版を出し始めたら、このページを直す

### 署名する物

- リリースの zip（[GitHub Releases](https://github.com/kagemorikosame/sashimono-edit/releases) の
  `SashimonoEdit-<版>-windows-x64.zip`）の中の、Sashimono Edit 本体の `Sashimono.exe` だけ
- `Sashimono.exe` は、このリポジトリのソースから GitHub Actions（`windows-latest`）で組む
  `v*` のタグを push すると `.github/workflows/release.yml` が `.github/workflows/package.yml` を呼ぶ
  PyInstaller の起動部は同じジョブでソースから組み直し、PyPI の既成の物は使わない
- zip の中の第三者の物（Python・Qt / PySide6・FFmpeg ほか、[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
  にある DLL や部品）は、このプロジェクトでは署名しない
- 開発者の機械で組んだ物には署名しない

### 役割

一人で開発しているので、役割はすべてリポジトリの持ち主が担う

| 役割 | 担う人 |
|---|---|
| 作る人と見る人（Committers and reviewers） | [kagemorikosame](https://github.com/kagemorikosame)（リポジトリの持ち主） |
| 承認する人（Approvers） | [kagemorikosame](https://github.com/kagemorikosame)（リポジトリの持ち主） |

- `main` へは PR で入れる決まり（ブランチの保護で CI が通ることを求める） 他の人（Dependabot を含む）の PR は、
  持ち主が見てからマージする
- 署名の頼みは、版ごとに承認する人が手で承認する
- 役割を持つ人は、GitHub と SignPath の両方で二段階認証を使う

### 個人情報の扱い

Sashimono Edit は、利用者が求めるか、通信する機能を利用者が入れない限り、ほかのネットワークの
システムへ情報を送らない 使い方の統計・落ちたときの報告・利用者を見分ける値は集めない
通信するのは次の所だけ

| 機能 | いつ通信するか | どこへ | 何を送るか | 切り方 |
|---|---|---|---|---|
| 更新の確かめ（配布の zip の版だけ） | 起動したとき（6 時間に 1 回まで）と、〔ヘルプ〕→〔更新を確かめる…〕を選んだとき | `https://github.com/kagemorikosame/sashimono-edit/releases/latest/download/update.json` と `update.json.sig`（「ベータ版も受け取る」を入れていれば `.../releases/download/beta/update.json` も） 転送は `github.com`・`objects.githubusercontent.com`・`release-assets.githubusercontent.com` へだけ追う | HTTPS の GET の頼みだけ 名乗り（User-Agent）は `SashimonoEdit/<版> (auto-update)` プロジェクトの中身・ファイル名・設定・利用者を見分ける値は送らない HTTPS で通信する以上、IP アドレスは GitHub から見える | 〔表示〕→〔設定…〕の「起動したときに新しい版を確かめる」を切る 切っても〔ヘルプ〕→〔更新を確かめる…〕で手で確かめられる |
| 更新を落とす | 確かめて新しい版が見つかったとき | GitHub Releases の zip（ホストは上と同じ） | 落とす頼みだけ 使う前に目録の署名（Ed25519）と zip の SHA-256 を確かめる 既定では入れる前に尋ねる | 更新の確かめと同じ設定 |
| 後から入れる部品（字幕起こし・AI アシスタント）の導入 | 利用者がアプリの中の導入ボタンを押したときだけ 走らせるコマンドは走らせる前に画面に出す | pip で Python Package Index（`pypi.org`・`files.pythonhosted.org`） | 包みを落とす頼み pip は決まった名乗り（pip・Python・OS の版）を付ける | 導入ボタンを押さない 既定では何も入れない |
| 字幕起こしのモデル | まだ落としていないモデルで、初めて字幕起こしを始めたとき | faster-whisper と huggingface_hub を通して Hugging Face Hub（`huggingface.co`） 置き場は `%USERPROFILE%\.cache\huggingface\hub`（`HF_HOME` があればその下） | モデルを落とす頼みだけ（落とす部品の決まった頭書きが付く） **音声は手元で起こし、外へ送らない** | 字幕起こしを使わない |
| AI アシスタント | 利用者が入れて、AI パネルで指示を送ったときだけ | Claude Code（Claude Agent SDK に同梱）を通して Anthropic へ 利用者自身の Claude のログインか API キーを使う | 利用者の指示、アシスタントへの決まった説明、モデルが呼んだ編集の道具の結果（プロジェクトの設定・トラック・クリップ・エフェクト・字幕の文・素材のファイル名と置き場のパス、プレビューを描いた絵、字幕起こしの結果） Anthropic での扱いは Anthropic の規約と個人情報の方針による Claude Code の組み込みの道具（ファイルの読み書き・シェル・Web）は止めてある ログインの情報を Sashimono Edit は読まず、持たない | AI アシスタントを入れない、または指示を送らない |

〔ヘルプ〕の中のリンク（使い方・報告の受け口）と、更新の知らせの中のリンク（変わった所・配布のページ）は、
利用者のブラウザで開く 開くときにアプリ自身は通信しない
〔ヘルプ〕→〔前の版に戻す…〕は手元に残した前の版を使い、通信しない
