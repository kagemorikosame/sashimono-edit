"""VOICEVOX 互換の音声合成のエンジンを呼ぶ口（ナレーションの生成 #335）

2026-10-20 に利用者が決めたこと（#335 の「決めたこと」）

- 声を作る部品は同梱しない 本人が入れて起動した VOICEVOX か AivisSpeech のエンジンを、
  同じ HTTP の口（``/speakers`` → ``/audio_query`` → ``/synthesis``）で呼ぶだけ
  声のモデルの規約への同意は、本人がアプリを入れたときに済んでいる
- 外へ送らない 呼ぶのは自分の機械の中（ループバック）の HTTP だけ 外の有料のサービスは入れない
  台本の文は字幕と文字と同じく本人の文なので、機械の外へ出る道を作らない
  （:func:`check_engine_url` が場所を限り、プロキシと転送を通さない）

ここは Qt を読まない AI の道具（``generate_narration``）と設定の窓の〔声の一覧を読む〕が
同じ所を通る 入口ごとに書くと、同じ声の名前やクレジットの書き方が入口で食い違う
"""

from __future__ import annotations

import http.client
import io
import ipaddress
import json
import re
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

from sashimono import __version__

__all__ = [
    "AIVIS_ENGINE_URL",
    "AUTO_VOICE",
    "CATALOG_TIMEOUT",
    "DEFAULT_ENGINE_URL",
    "INTONATION_PERCENT_RANGE",
    "NARRATION_FOLDER",
    "PIECE_LIMIT",
    "PITCH_RANGE",
    "POLICY_LIMIT",
    "SPEED_PERCENT_RANGE",
    "SYNTHESIS_TIMEOUT",
    "TEXT_LIMIT",
    "VOLUME_PERCENT_RANGE",
    "Catalog",
    "EngineError",
    "Line",
    "NarrationSettings",
    "Speaker",
    "Spoken",
    "Style",
    "Tuning",
    "VoiceEngine",
    "check_engine_url",
    "join_wavs",
    "split_script",
    "wav_seconds",
]

#: VOICEVOX のエンジンの既定の場所 アプリを入れて起動すれば、何も設定せずにつながる
DEFAULT_ENGINE_URL = "http://127.0.0.1:50021"
#: 作品のフォルダの下の、ナレーションの WAV を書くフォルダの名前（2026-10-20 利用者の決定）
NARRATION_FOLDER = "narration"
#: AivisSpeech のエンジンの既定の場所 設定の窓の説明に出す（同じ口で呼べる）
AIVIS_ENGINE_URL = "http://127.0.0.1:10101"
#: 既定の声を決めていない印 エンジンの一覧の最初の声を使う 声の ID はエンジンごとに
#: 違う（VOICEVOX は 0 から、AivisSpeech は 32 ビットの大きな数）ので、決め打ちの ID を
#: 既定にすると、片方のエンジンでは無い声を指して断られる
AUTO_VOICE = -1
#: 1 回に読ませる文の上限（文字） 長い台本は場面ごとに分けて頼んでもらう
#: CPU の AivisSpeech で 500 文字（およそ 1 分半の声）を作るのに数十秒かかる これより長いと
#: 待つ間 AI が人の新しい指示に応えられず、途中で止めても作った所までを残せない
TEXT_LIMIT = 500
#: エンジンへ 1 回に渡す文の長さの上限（文字） Style-Bert-VITS2 系（AivisSpeech）は
#: 長い文で抑揚が崩れ、既定で 100 文字までしか受けない 文の切れ目で分けて作り、つなぐ
PIECE_LIMIT = 100
#: 話す速さ（%）と音量（%）の範囲 VOICEVOX の画面で選べる範囲（速さ 0.5〜2.0 倍）に合わせる
SPEED_PERCENT_RANGE = (50, 200)
VOLUME_PERCENT_RANGE = (0, 200)
#: 声の高さ（VOICEVOX の pitchScale を 100 倍した値）と抑揚（%）の範囲
PITCH_RANGE = (-15, 15)
INTONATION_PERCENT_RANGE = (0, 200)
#: 声の一覧を読むのを待つ秒 自分の機械の中なので短くてよい 動いていなければすぐ断る
CATALOG_TIMEOUT = 5.0
#: 1 つの文の塊を作るのを待つ秒 遅い機械の CPU で 100 文字を作る長さより十分長く取る
SYNTHESIS_TIMEOUT = 120.0
#: 声の規約の文を返す長さの上限（文字） エンジンが返す規約は Markdown で、長い物もある
#: AI へ返す結果と設定の窓に収まる長さで切り、全文はエンジンのアプリで読んでもらう
POLICY_LIMIT = 2000
#: 返事の大きさの上限（バイト） 自分の機械の中の相手でも、壊れた返事で手元の記憶を
#: 使い切らないように区切る 100 文字の声の WAV は 1 MB ほど
_MAX_BODY = 64 * 1024 * 1024
#: 名乗り 本人の機械の中のエンジンにだけ送る
_USER_AGENT = f"SashimonoEdit/{__version__} (narration)"
#: 文の切れ目 ここで分けると、塊の境で読みが変わらない
#: 句点は直に書くと文章の検査（tools/punctuation.py）に掛かるので、番号で持つ
#: 続く区切り記号（句点 2 つ・「！？」「...」）の間では切らない 切ると記号だけの塊をエンジンへ
#: 渡し、読む物が無いと断られてナレーション全体が失敗する（記号は前の文の終わりに付けたままにする）
#: 記号は決まった文字だけで組む（台本の文は式に入らない）
_END_MARKS = chr(0x3002) + r"．.！!？?\n"
_SENTENCE_END = re.compile(f"(?<=[{_END_MARKS}])(?![{_END_MARKS}])")
#: 文の中の区切り 1 つの文が長すぎるときだけここで分ける
_CLAUSE_END = re.compile(r"(?<=[、，,；;：:）)」』])")


class EngineError(RuntimeError):
    """エンジンを呼べなかった 文面はそのまま人と AI へ出す

    台本の文はここに入れない 断りの文は AI へ（外へ）返るので、本人の文が混ざると、
    字幕と文字を送らないと決めた人の文まで送ることになる
    """


@dataclass(frozen=True, slots=True)
class Style:
    """声の 1 つの調子（ノーマル・あまあま など） ``id`` を合成に渡す"""

    id: int
    name: str


@dataclass(frozen=True, slots=True)
class Speaker:
    """声の持ち主（キャラクター・声のモデル）"""

    name: str
    uuid: str
    styles: tuple[Style, ...]


@dataclass(frozen=True, slots=True)
class Catalog:
    """エンジンが返した声の一覧と、エンジンの名乗り"""

    #: 呼んだ場所（:func:`check_engine_url` で整えた物）
    url: str
    #: エンジンの名前（``VOICEVOX Engine`` など） 無ければ空
    engine_name: str
    #: クレジットの頭に付ける名前（``VOICEVOX`` ``AivisSpeech``） エンジンの目録の
    #: ``brand_name`` が無ければエンジンの名前、それも無ければ「音声合成」
    brand: str
    speakers: tuple[Speaker, ...]
    #: エンジンの利用規約の文（目録の ``terms_of_service``） 無ければ空
    terms: str = ""

    def find(self, style_id: int) -> tuple[Speaker, Style] | None:
        """声の ID の持ち主と調子 ``AUTO_VOICE`` なら一覧の最初の声"""
        for speaker in self.speakers:
            for style in speaker.styles:
                if style_id in (style.id, AUTO_VOICE):
                    return speaker, style
        return None

    def credit(self, speaker: Speaker) -> str:
        """書き出した動画に添える表記（「VOICEVOX:ずんだもん」の形）

        VOICEVOX の規約は「VOICEVOX:キャラ名」の表記を求める AivisSpeech はエンジンの
        表記は要らないが、声のモデルごとの規約で求める物があるので同じ形で知らせる
        知らせ過ぎて困ることは無く、知らせ忘れて規約に反するのは本人になる
        """
        return f"{self.brand}:{speaker.name}"


@dataclass(frozen=True, slots=True)
class Tuning:
    """読み方の調整 値はエンジンの ``audio_query`` の欄へそのまま入る"""

    speed_percent: int = 100
    volume_percent: int = 100
    pitch: int = 0
    intonation_percent: int = 100


@dataclass(frozen=True, slots=True)
class Line:
    """作った声の 1 文 字幕の 1 枚になる"""

    text: str
    #: その文の声の長さ（秒 前後の間を含む）
    seconds: float


@dataclass(frozen=True, slots=True)
class Spoken:
    """作った声 つないだ WAV と、文ごとの長さ"""

    wav: bytes
    lines: tuple[Line, ...]


@dataclass(frozen=True, slots=True)
class NarrationSettings:
    """ナレーションの本人の設定（:class:`~sashimono.ui.workspace.Preferences` から作る）

    どのエンジンをどこで動かすか・どの声が好きかは人と機械で違う AI が引数で
    言わなければこれを使う
    """

    #: エンジンの場所（:func:`check_engine_url` を通る物）
    url: str = DEFAULT_ENGINE_URL
    #: 既定の声の ID ``AUTO_VOICE`` ならエンジンの一覧の最初の声
    voice: int = AUTO_VOICE
    #: 既定の声の名前 設定の窓でエンジンが動いていないときにも名前で見せるため覚える
    voice_name: str = ""
    speed_percent: int = 100
    volume_percent: int = 100
    #: AI が既定と違う声を選んでよいか 切ると既定の声だけを使う
    #: 既定は選んでよい 声は毎回の確かめの窓に出るので、人が知らない声で作られることは無い
    ai_picks_voice: bool = True
    #: 作った声に台本から字幕を付けるか（文ごとに区切り、声の長さに合わせる） 既定は付ける
    #: （2026-10-20 利用者の決定） 字幕を別に起こす人や、字幕の要らない人は切れる
    subtitles: bool = True


def check_engine_url(raw: object) -> str:
    """エンジンの場所を確かめ、整えた形（``http://127.0.0.1:50021``）で返す

    自分の機械の中（ループバックの番地か ``localhost``）の ``http`` だけを通す 外の番地を
    通すと、台本の文（本人の文）を知らない機械へ送れてしまう（2026-10-20 利用者の決定
    外へ送らない） 道や問い合わせ・利用者名の付いた場所も断る 付いていると、どこへ何を
    送るのかが場所の文字から読めなくなる 断るときは :class:`ValueError`
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("エンジンの場所が空です")
    text = raw.strip()
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"エンジンの場所を読めません: {text}") from exc
    if parts.scheme != "http":
        raise ValueError(f"エンジンの場所は http:// で始めてください: {text}")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"エンジンの場所に利用者名を入れないでください: {text}")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError(f"エンジンの場所は番地と番号だけにしてください: {text}")
    host = parts.hostname or ""
    if host.lower() != "localhost":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError(
                f"エンジンの場所は自分の機械の中（127.0.0.1 か localhost）だけです: {text}"
                " 台本の文を機械の外へ送らないため"
            )
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}" + (f":{port}" if port is not None else "")


#: 1 回ぶんの HTTP の頼み（方法・道・本文） 試験が偽の相手へ差し替えられるよう関数にしてある
type Requester = Callable[[str, str, bytes | None, float], tuple[int, bytes]]


@dataclass(slots=True)
class VoiceEngine:
    """エンジン 1 つ 頼むたびにつなぐ（つなぎっぱなしにしない エンジンを閉じた後も残らない）"""

    url: str
    #: 試験のための差し替え口 渡さなければ :mod:`http.client` で本当につなぐ
    requester: Requester | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            self.url = check_engine_url(self.url)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc

    def catalog(self) -> Catalog:
        """声の一覧とエンジンの名乗りを読む"""
        manifest: dict[str, Any] = {}
        try:
            raw_manifest = self._json(
                "GET", "/engine_manifest", timeout=CATALOG_TIMEOUT, optional=True
            )
        except _MissingError:
            # 古いエンジンは目録を持たない 名前が分からないだけで声は作れる
            raw_manifest = {}
        if isinstance(raw_manifest, dict):
            manifest = raw_manifest
        raw = self._json("GET", "/speakers", timeout=CATALOG_TIMEOUT)
        if not isinstance(raw, list):
            raise EngineError(f"{self.url} の声の一覧が読めない形でした")
        speakers: list[Speaker] = []
        for item in raw:
            speaker = _speaker(item)
            if speaker is not None:
                speakers.append(speaker)
        if not speakers:
            raise EngineError(
                f"{self.url} のエンジンに使える声がありません エンジンのアプリで声を入れてください"
            )
        engine_name = _text(manifest.get("name"))
        brand = _text(manifest.get("brand_name")) or engine_name or "音声合成"
        return Catalog(
            url=self.url,
            engine_name=engine_name,
            brand=brand,
            speakers=tuple(speakers),
            terms=_text(manifest.get("terms_of_service"))[:POLICY_LIMIT],
        )

    def policy(self, speaker: Speaker) -> str:
        """声の規約の文（エンジンの ``/speaker_info`` の ``policy``） 無ければ空

        アプリは中身を判断しない（企業での利用や非商用だけの声があるが、読むのは本人）
        エンジンが返した文をそのまま、:data:`POLICY_LIMIT` 文字まで見せる
        """
        path = f"/speaker_info?speaker_uuid={quote(speaker.uuid)}&resource_format=url"
        try:
            raw = self._json("GET", path, timeout=CATALOG_TIMEOUT, optional=True)
        except _MissingError:
            return ""
        if not isinstance(raw, dict):
            return ""
        text = _text(raw.get("policy"))
        if len(text) > POLICY_LIMIT:
            return text[:POLICY_LIMIT] + "…（続きはエンジンのアプリで読めます）"
        return text

    def synthesize(
        self,
        text: str,
        style: int,
        tuning: Tuning,
        *,
        stopped: Callable[[], bool] = lambda: False,
    ) -> Spoken:
        """台本を WAV にする 文ごとに作り、つないで 1 本にする

        文ごとに作るのは、字幕を文ごとに区切り、作った声の長さに合わせて置くため
        （2026-10-20 利用者の決定） まとめて作ると、どこで次の文になるかが声から分からない
        ``stopped`` が真を返したら、次の文を作らずに断る（待つのをやめた道具の後で、
        エンジンを使い続けない）
        """
        pieces = split_script(text, PIECE_LIMIT)
        if not pieces:
            raise EngineError("読ませる文が空です")
        waves: list[bytes] = []
        for piece in pieces:
            if stopped():
                raise EngineError("声を作るのを止めました")
            query_path = f"/audio_query?text={quote(piece)}&speaker={style}"
            query = self._json("POST", query_path, timeout=SYNTHESIS_TIMEOUT)
            if not isinstance(query, dict):
                raise EngineError("エンジンが読み方を返しませんでした")
            query.update(
                {
                    "speedScale": tuning.speed_percent / 100,
                    "volumeScale": tuning.volume_percent / 100,
                    "pitchScale": tuning.pitch / 100,
                    "intonationScale": tuning.intonation_percent / 100,
                }
            )
            body = json.dumps(query, ensure_ascii=False).encode("utf-8")
            status, data = self._request(
                "POST", f"/synthesis?speaker={style}", body, SYNTHESIS_TIMEOUT
            )
            self._check(status, "/synthesis")
            waves.append(data)
        joined = join_wavs(waves)
        lines = tuple(
            Line(piece, wav_seconds(data)) for piece, data in zip(pieces, waves, strict=True)
        )
        return Spoken(joined, lines)

    # --- 中身 ---

    def _json(self, method: str, path: str, *, timeout: float, optional: bool = False) -> object:
        """``optional`` の口は、エンジンに無ければ :class:`_MissingError` で知らせる"""
        status, data = self._request(method, path, b"" if method == "POST" else None, timeout)
        if optional and status == http.client.NOT_FOUND:
            raise _MissingError(path)
        self._check(status, path.split("?", 1)[0])
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise EngineError(
                f"{self.url} の返事が読めない形でした（{path.split('?', 1)[0]}）"
            ) from exc

    def _check(self, status: int, where: str) -> None:
        if status == http.client.NOT_FOUND:
            raise EngineError(
                f"{self.url} は VOICEVOX 互換のエンジンではないようです（{where} がありません）"
            )
        if status == http.client.UNPROCESSABLE_ENTITY:
            # 声の ID が無い・文が読めない など 返事の本文には台本の文が入りうるので載せない
            raise EngineError(
                f"エンジンが頼みを受け付けませんでした（{where} HTTP {status}） 声の ID が"
                "エンジンの一覧にあるか、文が長すぎないかを確かめてください"
            )
        if not 200 <= status < 300:
            raise EngineError(f"エンジンが断りました（{where} HTTP {status}）")

    def _request(
        self, method: str, path: str, body: bytes | None, timeout: float
    ) -> tuple[int, bytes]:
        if self.requester is not None:
            return self.requester(method, path, body, timeout)
        parts = urlsplit(self.url)
        connection = http.client.HTTPConnection(
            parts.hostname or "", parts.port or 80, timeout=timeout
        )
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json, audio/wav"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            # 転送（3xx）は追わない http.client は自分では追わないので、返った番号のまま
            # 断る（_check） 追うと、自分の機械の中の相手が外の場所へ飛ばせてしまう
            # プロキシも通さない（http.client は環境変数のプロキシを読まない）
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read(_MAX_BODY + 1)
            status = response.status
        except (OSError, http.client.HTTPException) as exc:
            raise EngineError(
                f"{self.url} の音声合成のエンジンにつながりません VOICEVOX か AivisSpeech を"
                "起動してから、もう 1 度頼んでください（エンジンの場所は 表示 → 設定… の"
                f"「ナレーション: エンジンの場所」で変えられます 理由: {type(exc).__name__}）"
            ) from exc
        finally:
            connection.close()
        if len(data) > _MAX_BODY:
            raise EngineError(f"エンジンの返事が大きすぎます（{path.split('?', 1)[0]}）")
        return status, data


class _MissingError(EngineError):
    """その口がエンジンに無い（404） 無くても進められる口（目録・規約）で使う"""


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _speaker(item: object) -> Speaker | None:
    """一覧の 1 行 読めない行は飛ばす（1 つの壊れた声で一覧ごと断らない）"""
    if not isinstance(item, dict):
        return None
    name = _text(item.get("name"))
    uuid = _text(item.get("speaker_uuid"))
    styles: list[Style] = []
    raw_styles = item.get("styles")
    for raw in raw_styles if isinstance(raw_styles, list) else []:
        if not isinstance(raw, dict):
            continue
        style_id = raw.get("id")
        kind = raw.get("type", "talk")
        # 歌の声（sing・singing_teacher）は文を読めない 選べると合成で断られる
        if isinstance(style_id, bool) or not isinstance(style_id, int) or kind != "talk":
            continue
        styles.append(Style(style_id, _text(raw.get("name")) or str(style_id)))
    if not name or not styles:
        return None
    return Speaker(name, uuid, tuple(styles))


def split_script(text: str, limit: int = PIECE_LIMIT) -> list[str]:
    """台本を文ごとの ``limit`` 文字以下の塊に分ける 空の塊は捨てる

    文の切れ目（句点・！？と改行）で分ける 1 つの塊が字幕の 1 枚になるので、文をまたいで
    詰めない 1 つの文が長すぎれば読点などで分けて詰め、それでも長ければ ``limit`` 文字ずつに
    分ける 切れ目の無い所で切ると読みと抑揚が変わるので、切るのは最後の手
    """
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        current = ""
        for part in _fit(sentence, limit):
            if len(current) + len(part) <= limit:
                current += part
                continue
            if current.strip():
                pieces.append(current.strip())
            current = part
        if current.strip():
            pieces.append(current.strip())
    # 読む文字の無い塊（記号と空白だけ）は送らない エンジンは読む物が無いと断り、
    # 1 つの塊の失敗でナレーション全体が作れなくなる 字幕もこの塊から作るので、声の無い
    # 字幕も出ない
    return [piece for piece in pieces if any(char.isalnum() for char in piece)]


def _fit(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    parts: list[str] = []
    for clause in _CLAUSE_END.split(sentence):
        while len(clause) > limit:
            parts.append(clause[:limit])
            clause = clause[limit:]
        parts.append(clause)
    return parts


def join_wavs(waves: list[bytes]) -> bytes:
    """WAV を順につないで 1 本にする 1 本ならそのまま返す

    形（チャンネル・サンプル幅・サンプリングレート）が違えばつなげないので断る
    エンジンは同じ声なら同じ形で返す
    """
    if not waves:
        raise EngineError("エンジンが声を返しませんでした")
    if len(waves) == 1:
        _params(waves[0])
        return waves[0]
    frames: list[bytes] = []
    shape: tuple[int, int, int] | None = None
    for data in waves:
        params, body = _params(data)
        if shape is None:
            shape = params
        elif params != shape:
            raise EngineError("エンジンが塊ごとに違う形の WAV を返したので、つなげません")
        frames.append(body)
    assert shape is not None
    channels, width, rate = shape
    out = io.BytesIO()
    with wave.open(out, "wb") as joined:
        joined.setnchannels(channels)
        joined.setsampwidth(width)
        joined.setframerate(rate)
        joined.writeframes(b"".join(frames))
    return out.getvalue()


def _params(data: bytes) -> tuple[tuple[int, int, int], bytes]:
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            shape = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            body = source.readframes(source.getnframes())
    except (wave.Error, EOFError) as exc:
        raise EngineError(f"エンジンの返した声が WAV として読めません（{exc}）") from exc
    return shape, body


def wav_seconds(data: bytes) -> float:
    """WAV の長さ（秒） 結果と確かめに出す"""
    with wave.open(io.BytesIO(data), "rb") as source:
        rate = source.getframerate()
        return source.getnframes() / rate if rate > 0 else 0.0
