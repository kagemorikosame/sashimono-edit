"""報告に貼る文面から、本人の置き場を伏せる

互換性レポートとテンプレートの棚の〔内容をコピー〕は、どちらも公開の Issue に
貼られる文を作る 伏せ方を別々に書くと、片方だけ直したときにもう片方から
利用者名が漏れるので、ここ 1 か所に置いて両方から使う

Qt を使わない 文面を組む所（試験を含む）が窓を作らずに呼べるように
"""

from __future__ import annotations

import ctypes
import ntpath
import os
import re
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "HOME_PLACEHOLDER",
    "SECRET_MARK",
    "marked_root_labels",
    "mask_places",
    "mask_record_text",
    "mask_report",
    "mask_secrets",
    "mask_user_folders",
    "root_key",
    "root_lines",
    "root_markers",
    "short_path",
    "user_folders",
]

#: 写した文面で、本人のホームフォルダの代わりに置く文字
#: 失敗の記録には OS の文言がそのまま入り、ファイルの場所（ユーザー名を含む）が混じる
#: 不具合の報告は公開の Issue に貼られるので、名前が出ないように伏せる
HOME_PLACEHOLDER = "%USERPROFILE%"

#: ホームのほかに伏せる置き場 （環境変数, 置き換える文字）
#: 普段はどちらもホームの下にあるが、移動プロファイルや組織の設定でホームの外
#: （ネットワークの置き場など）へ向いていることがあり、その場所にも利用者名が入る
#: ホームだけを伏せると、そうした機械では名前がそのまま残る
#: 環境変数の名前で置くのは、伏せたあとも「設定の置き場の中」だと読めるようにするため
#: Windows 以外の置き場（`core/userdirs.py` が使う XDG の 4 つ）も同じ理由で伏せる
#: ホームの外へ向けた機械では、ホームを伏せても名前が残る
_FOLDER_VARIABLES = (
    ("APPDATA", "%APPDATA%"),
    ("LOCALAPPDATA", "%LOCALAPPDATA%"),
    ("XDG_CONFIG_HOME", "$XDG_CONFIG_HOME"),
    ("XDG_STATE_HOME", "$XDG_STATE_HOME"),
    ("XDG_CACHE_HOME", "$XDG_CACHE_HOME"),
    ("XDG_DATA_HOME", "$XDG_DATA_HOME"),
    # 利用者名は入らないが名前で置く AviUtl2 の Script（``%PROGRAMDATA%\aviutl2\Script``）
    # は既定の探索先で、伏せないと探索先の印（``<探索先1>``）になり、AviUtl2 の
    # スクリプトを見に行っているかが報告から読めなくなる
    ("PROGRAMDATA", "%PROGRAMDATA%"),
)

#: 区切りとみなす文字 Windows は ``/`` も ``\`` も受け付け、OS の文言や
#: スクリプトの書いた場所ではどちらも混ざる
_SEPARATORS = r"[\\/]+"

#: 伏せた場所のすぐ後に来てよい文字（区切り・空白・引用符・括弧など）
#: これ以外が続くときは名前の途中なので伏せない ``C:\Users\kage`` を伏せるときに
#: ``C:\Users\kagemori`` の頭だけを伏せると、残りから名前が読める
#: 全角の閉じ括弧（``）`` ``」`` ``』`` ``】`` ``〕``）も終わりとみなす 文の中で場所を括弧で
#: 囲む書き方はありうるが、フォルダの名前がそれで続くことはまず無い
#: 読点・句点・全角のカンマとピリオドは認めない 名前の途中に来うる（``山田、太郎``）ので、
#: 認めると別の人の場所の頭だけを伏せ、残りから名前が読める こちらが組む文では、
#: 場所の直後にそれらを置かない（探索先は 1 行に 1 つ）ことで釣り合いを取る
#: ``・`` も同じ理由で足さない カタカナの名前（``ジョン・スミス``）では名前の続きに来る
_JAPANESE_ENDS = "）」』】〕"
_FOLDER_END = r"(?![^\\/\s\"'<>|:;,)\]}" + _JAPANESE_ENDS + r"])"


def user_folders(home: Path | None = None) -> list[tuple[str, str]]:
    r"""伏せる置き場と、その代わりに置く文字

    Windows では 8.3 形式の短い名前（``C:\Users\KAGEMO~1``）も足す 古い API や
    一部の DLL は場所を短い形で返し、それが失敗の文言にそのまま入る 短い形にも
    利用者名の頭が残るので、長い形だけを伏せると公開の Issue に名前が出る
    """
    longs = [(str(home if home is not None else Path.home()), HOME_PLACEHOLDER)]
    for variable, placeholder in _FOLDER_VARIABLES:
        value = os.environ.get(variable)
        if value:
            longs.append((value, placeholder))
    folders = list(longs)
    for folder, placeholder in longs:
        short = short_path(folder)
        # 同じ形を重ねても害は無いが、伏せる回数が増えるだけなので足さない
        if short and short.casefold() != folder.casefold():
            folders.append((short, placeholder))
    return folders


def short_path(folder: str) -> str | None:
    """8.3 形式の短い名前 求められなければ ``None``

    Windows 以外・8.3 を切ってある置き場・無い場所・呼び出しの失敗は、どれも
    ``None`` にして長い形だけで続ける 伏せる対象を足すための手当てで、取れない
    からといってコピーそのものを止めるほどのことではない
    """
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    try:
        get_short = windll.kernel32.GetShortPathNameW
        # 1 回目で要る長さを聞き、2 回目で受け取る 長さを決め打ちすると、深い置き場で切れる
        size = int(get_short(folder, None, 0))
        if size <= 0:
            return None
        buffer = ctypes.create_unicode_buffer(size)
        written = int(get_short(folder, buffer, size))
    except (AttributeError, OSError, ValueError):
        return None
    if written <= 0 or written >= size:
        return None
    return buffer.value or None


def mask_user_folders(text: str, folders: Sequence[tuple[str, str]]) -> str:
    r"""文面の中の置き場を、書き方の揺れごと伏せる

    完全一致で置き換えると、``c:\users\…`` と ``C:\Users\…``、``/`` と ``\``
    の違う書き方が素通りする Windows の場所は大文字と小文字を区別しないので、
    どれも同じ場所で、同じように利用者名を含む

    長い場所から伏せる ``%APPDATA%`` はホームの下にあるので、先にホームを伏せると
    ``%USERPROFILE%\AppData\Roaming`` になり、設定の置き場だと読みにくくなる
    """
    for folder, placeholder in sorted(folders, key=lambda pair: len(pair[0]), reverse=True):
        parts = [part for part in re.split(_SEPARATORS, folder) if part]
        # 短すぎる場所（根だけなど）で置き換えると、関係ない文字まで伏せてしまう
        if len(parts) < 2:
            continue
        pattern = _SEPARATORS.join(re.escape(part) for part in parts) + _FOLDER_END
        # 頭の区切り（ネットワークの置き場の ``\\server`` や、Windows 以外の ``/home``）も
        # 伏せる側に含める 残すと ``\\%APPDATA%`` のような読めない形になる
        if re.match(_SEPARATORS, folder):
            pattern = _SEPARATORS + pattern
        # 置き換える文字は escape して渡す 素のままだと ``\`` を置き換えの書式として読まれる
        replacement = placeholder.replace("\\", "\\\\")
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


#: 秘密を伏せた印
SECRET_MARK = "〔秘密〕"

#: 名前の分かる秘密の形 （正規表現, 残す頭の組の番号） 頭の組は伏せずに残し、何の秘密だったか
#: （API の鍵か、トークンか）が報告から読めるようにする
#: 診断情報（#516）に混ざりうるのは、AI の部品や更新の係が上げたエラーの文に入った鍵と、
#: 設定の窓で打ち込んだ値を含む例外の文 形の分からない秘密は、本人が窓で見て消す
#: どの形も 1 番目の組が残す頭
_SECRET_NAME = r"(?i:api[_\-]?key|access[_\-]?key|secret|token|password|passwd|pwd)"
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Anthropic・OpenAI の API の鍵 ``sk-ant-api03-…`` ``sk-proj-…``
    re.compile(r"(sk-)(?:ant-|proj-)?[A-Za-z0-9_\-]{16,}"),
    # GitHub のトークン ``ghp_…`` ``github_pat_…``
    re.compile(r"((?:gh[pousr]|github_pat)_)[A-Za-z0-9_]{16,}"),
    # AWS の鍵の番号
    re.compile(r"(AKIA)[0-9A-Z]{16}"),
    # ``Authorization: Bearer …`` ``Bearer …``
    re.compile(r"((?i:bearer)\s+)[A-Za-z0-9._~+/\-]+=*"),
    # 引用符で囲んだ値 ``"password": "two words"`` 閉じる引用符まで（中の ``\"`` は飛ばす）
    # 空白で止めると、空白を含むパスワードの後ろが残る（PR #550 の CodeRabbit の指摘）
    re.compile("(" + _SECRET_NAME + r"""["']?\s*[:=]\s*")(?:[^"\\\n]|\\.)*"""),
    re.compile("(" + _SECRET_NAME + r"""["']?\s*[:=]\s*')(?:[^'\\\n]|\\.)*"""),
    # 囲んでいない値 ``api_key=…`` ``password: …`` 空白・区切りの前まで
    # 囲んだ値は上で印になっているので、ここでは引用符で始まる値を飛ばす
    re.compile("(" + _SECRET_NAME + r"""["']?\s*[:=]\s*)(?!["'])[^\s"',;}]+"""),
)


def mask_secrets(text: str) -> str:
    """API の鍵・トークン・パスワードを :data:`SECRET_MARK` に置き換える

    鍵は 1 つ漏れるだけで他人に使われ、伏せすぎても困るのは値が読めないことだけなので、
    名前の分かる形は値を丸ごと伏せる 名前と区切り（``"password": "``）は残す
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(rf"\g<1>{SECRET_MARK}", text)
    return text


#: 場所の頭 置き場の印（:data:`HOME_PLACEHOLDER` など 印は残す）・ドライブ名・UNC
_PLACE_HEAD = r"(?:((?:%[A-Z_]+%|\$XDG_[A-Z_]+))[\\/]|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\)"
#: 場所の続き 空白は含める（空白を含む素材の名前の後ろを残さないため PR #550 の指摘）
#: 改行・引用符・括弧・文の区切りの記号で終わる 記号は ``ai/outbound.py`` と同じく番号で書く
_PLACE_TAIL = r"[^\r\n\"'<>|*?\u3001\u3002\uff0c\uff08\uff09()\u300c\u300d]*"
_PLACE = re.compile(_PLACE_HEAD + _PLACE_TAIL)


def mask_places(text: str, mark: str) -> str:
    r"""ファイルの場所を ``mark`` に置き換える 置き場の印（``%USERPROFILE%``）は残す

    :func:`mask_user_folders` は利用者名の入った置き場だけを伏せ、その下のフォルダと
    ファイルの名前（``%USERPROFILE%\Videos\旅行 1 日目.mp4``）は残す 互換性レポートでは探索先の
    中を読めるよう残すが、診断情報（#516）では素材の名前になるので伏せる
    空白で止めない 形で場所を拾う伏せ方（``ai/outbound.py``）は空白で止まり、空白を含む名前の
    後ろ（`` 1 日目.mp4``）が残る 場所の後ろに続く文まで伏せることがあるが、伏せすぎて困るのは
    読めないことだけで、出た名前は取り戻せない 引用符で囲んだ場所は閉じる引用符で終わる
    印は残す どの置き場（設定・退避・ホーム）の話かは報告から読める
    """

    def one(found: re.Match[str]) -> str:
        folder = found.group(1)
        return f"{folder}\\{mark}" if folder else mark

    return _PLACE.sub(one, text)


#: 引用符と括弧で囲んだ文 例外の文は名前や本文をここに入れる
#: （``'旅行.mp4'`` ``「社長のあいさつ」``） 中の ``\'`` は囲みの終わりと見ない 終わりと見ると、
#: 囲みの後ろ半分が伏せずに残る（PR #550 の CodeRabbit の指摘）
_QUOTED = re.compile(r"""'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|\u300c[^\u300d\n]*\u300d""")
#: 名前と拡張子の形の語 囲まずに書かれた素材の名前（``旅行.mp4 を読めない``）
_FILE_NAME = re.compile(
    r"(?<![\w.\\/])[^\s\"'<>|*?\\/:\u3001\u3002()\uff08\uff09]+\.[A-Za-z][A-Za-z0-9]{1,4}\b"
)
#: 名前の形でも伏せない拡張子 本体と部品のファイル（並びの中の場所や部品の名前）
_PROGRAM_SUFFIXES = frozenset({"py", "pyd", "pyc", "dll", "exe", "so", "dylib", "qml"})


def mask_record_text(text: str, mark: str) -> str:
    """前の起動の記録（エラーや落ちた時の並び）の中の、名前と本文になりうる所を伏せる

    今の作品から作る伏せる表には、前の起動で開いた素材の名前や字幕は入っていない 表を
    起動をまたいで持つと、伏せたい物そのものを本人の置き場へ書き残すことになるので持たない
    代わりに、例外の文が名前と本文を入れる形（引用符と括弧で囲んだ所・名前と拡張子の語）を
    伏せる 並びの行（``File "…", line 3, in 名前``）と本体の部品の名前は残す
    （PR #550 の CodeRabbit の指摘）
    """
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith('File "'):
            lines.append(line)
            continue
        line = _QUOTED.sub(lambda found: found.group(0)[0] + mark + found.group(0)[-1], line)
        lines.append(_hide_file_names(line, mark))
    return "\n".join(lines)


def _hide_file_names(line: str, mark: str) -> str:
    """名前と拡張子の語を、その前の文（行の頭か ``: `` の後ろから）ごと伏せる

    空白を含む名前（``社外秘 顧客名簿.csv``）は、どこから名前かを文から決められない 語だけを
    伏せると頭（``社外秘``）が残るので、例外の種類（``OSError: ``）の後ろから語の終わりまでを
    まとめて伏せる 伏せすぎて困るのは読めないことだけ（PR #550 の CodeRabbit の指摘）
    本体と部品のファイル（:data:`_PROGRAM_SUFFIXES`）は伏せない
    """
    parts: list[str] = []
    position = 0
    for found in _FILE_NAME.finditer(line):
        start, end = found.span()
        if found.group(0).rsplit(".", 1)[1].lower() in _PROGRAM_SUFFIXES or start < position:
            continue
        head = line[position:start]
        colon = max(head.rfind(": "), head.rfind("\uff1a"))
        if colon >= 0:
            begin = position + colon + (2 if head.startswith(": ", colon) else 1)
        else:
            begin = position + len(head) - len(head.lstrip())
        parts += [line[position:begin], mark]
        position = end
    parts.append(line[position:])
    return "".join(parts)


def root_key(text: str) -> str:
    """探索先が同じ場所かを見分ける鍵 大文字小文字と区切りの違いは同じ場所

    頭の区切り（`\a` と `a` の違い）とドライブは残す 捨てると、絶対の場所と
    同じ名前の相対の場所を同じ物とみなし、片方に印が付かない
    まとめる側（`root_markers`）と画面で引く側で同じ物を使う 別々に書くと食い違う
    """
    return ntpath.normcase(ntpath.normpath(text))


def root_markers(
    roots: Sequence[Path | str], folders: Sequence[tuple[str, str]]
) -> list[tuple[str, str]]:
    r"""探索先のうち、伏せる置き場の外にある物と、その代わりに置く印（``<探索先1>`` など）

    探索先は本人が決めた場所で、``D:\<名前>\aviutl\Script`` のように利用者名を
    含みうる 読めなかったスクリプトの記録には OS の文言ごと絶対パスが入るので、
    ホームや設定の置き場の外にある探索先は、それだけで名前が漏れる道になる

    伏せる置き場の下にある探索先（``%APPDATA%\Sashimono\scripts`` など）は印にしない
    そちらの名前で伏せたほうが、どの置き場の話かが報告から読める
    番号は画面の「探索先」の並びの順 どの印がどの場所かは画面にだけ出し、貼る文には出さない
    """
    markers: list[tuple[str, str]] = []
    numbers: dict[str, str] = {}
    written: set[str] = set()
    for root in roots:
        text = str(root)
        # 1 段だけの相対の名前（写真の道具が出す ``scripts`` など）は場所を明かさない
        # 印にすると、文面の中のただの単語まで置き換わる（伏せる側も 1 段は飛ばす）
        if len([part for part in re.split(_SEPARATORS, text) if part]) < 2:
            continue
        if mask_user_folders(text, folders) != text:
            continue
        # 同じ場所には同じ印 印が 2 つ付くと、画面と貼る文で違う印が出て、報告者が
        # 印の指す場所を確かめられない 同じ場所かは `root_key` で見る
        key = root_key(text)
        if key not in numbers:
            numbers[key] = f"<探索先{len(numbers) + 1}>"
        # ただし書き方ごとに全部並べる 同じ場所の別の書き方（`..` を含む・大文字小文字
        # が違う など）を落とすと、そちらは伏せる相手から外れ、貼る文に名前が残る
        if text in written:
            continue
        written.add(text)
        markers.append((text, numbers[key]))
    return markers


def root_lines(labels: Sequence[str]) -> list[str]:
    """探索先の並びの行 1 行に 1 つ

    読点でつなぐと、場所の直後に読点が来る 伏せる側は読点を場所の終わりと認めない
    （名前の途中に来うる）ので、つないだ探索先は最後の 1 つしか伏せられない
    """
    if not labels:
        return ["探索先: （設定なし）"]
    return ["探索先:", *(f"  {label}" for label in labels)]


def mask_report(
    text: str,
    roots: Sequence[Path | str] = (),
    folders: Sequence[tuple[str, str]] | None = None,
) -> str:
    """貼る文を伏せる 置き場（ホーム・設定の置き場など）と、その外の探索先の両方

    探索先の印は置き場を伏せる仕組みに載せる 別々に置き換えると、長い方から
    伏せる順（``mask_user_folders`` が守る）が崩れ、設定の置き場の下の探索先が
    印になって、どの置き場の話かが読めなくなる
    """
    hidden = list(folders if folders is not None else user_folders())
    hidden.extend(root_markers(roots, hidden))
    return mask_user_folders(text, hidden)


def marked_root_labels(
    roots: Sequence[Path | str], folders: Sequence[tuple[str, str]] | None = None
) -> list[str]:
    r"""画面に出す探索先 貼る文で印に置き換える物には、印を添える

    報告を受けた側が「<探索先1> が何か」を聞いたときに、本人が画面で答えられるように
    引くときも、まとめたときと同じ見分け方（``root_key``）で引く 生の文字で引くと、
    ``D:\山田\Script`` と ``d:/山田/script`` が並んだとき後の方に印が付かない
    """
    hidden = folders if folders is not None else user_folders()
    markers = {root_key(root): marker for root, marker in root_markers(roots, hidden)}
    labels: list[str] = []
    for root in roots:
        marker = markers.get(root_key(str(root)))
        labels.append(f"{marker} {root}" if marker is not None else str(root))
    return labels
