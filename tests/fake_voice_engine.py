"""VOICEVOX 互換のエンジンのふり（ナレーション #335 の試験用）

本物の VOICEVOX・AivisSpeech は CI の機械に無い 自分の機械の中（127.0.0.1）に HTTP の相手を
立て、同じ口（``/engine_manifest`` ``/speakers`` ``/speaker_info`` ``/audio_query``
``/synthesis``）を返す 本物の HTTP を通すので、つなぎ方（プロキシや転送を通さない・
番号を確かめる）まで試せる

返事の形は VOICEVOX ENGINE の API の説明書きの形に合わせてある 実物の形と違うと、
ここでは通っても本人の機械で読めない 実物で確かめるのは #335 の「確かめること」
"""

from __future__ import annotations

import io
import json
import threading
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tests.waiting import GUARD_SECONDS

__all__ = ["DEFAULT_SPEAKERS", "FakeVoiceEngine", "closed_engine_url", "fake_voice_engine"]

#: 既定の声の一覧 VOICEVOX の /speakers と同じ形 歌の声（sing）は文を読めないので選べない
DEFAULT_SPEAKERS: list[dict[str, Any]] = [
    {
        "name": "四国めたん",
        "speaker_uuid": "7ffcb7ce-00ec-4bdc-82cd-45a8889e43ff",
        "styles": [
            {"name": "ノーマル", "id": 2, "type": "talk"},
            {"name": "あまあま", "id": 0, "type": "talk"},
        ],
        "version": "0.15.0",
    },
    {
        "name": "ずんだもん",
        "speaker_uuid": "388f246b-8c41-4ac1-8e2d-5d79f3ff56d9",
        "styles": [
            {"name": "ノーマル", "id": 3, "type": "talk"},
            {"name": "ハミング", "id": 3001, "type": "sing"},
        ],
        "version": "0.15.0",
    },
]

#: 1 文字あたりの声の長さ（秒） 返す WAV の長さを文の長さから決める
SECONDS_PER_CHAR = 0.02
#: 返す WAV のサンプリングレート VOICEVOX の既定と同じ
RATE = 24000


@dataclass
class FakeVoiceEngine:
    """待ち受けている偽のエンジン 頼まれた物を覚える"""

    url: str = ""
    speakers: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_SPEAKERS))
    #: 目録 ``None`` なら /engine_manifest が無い（古いエンジン）
    manifest: dict[str, Any] | None = field(
        default_factory=lambda: {
            "name": "VOICEVOX Engine",
            "brand_name": "VOICEVOX",
            "terms_of_service": "エンジンの利用規約",
        }
    )
    #: 声の持ち主ごとの規約の文
    policies: dict[str, str] = field(
        default_factory=lambda: {
            "388f246b-8c41-4ac1-8e2d-5d79f3ff56d9": "ずんだもんの規約 クレジットを書くこと",
            "7ffcb7ce-00ec-4bdc-82cd-45a8889e43ff": "四国めたんの規約",
        }
    )
    #: 頼まれた順の（方法, 道, 本文）
    requests: list[tuple[str, str, bytes]] = field(default_factory=list)
    #: /audio_query で受けた文
    texts: list[str] = field(default_factory=list)
    #: /synthesis で受けた読み方の欄
    queries: list[dict[str, Any]] = field(default_factory=list)
    #: 合成の前に待つ秒 待ちの試験で使う
    synthesis_delay: float = 0.0
    #: 声の一覧を返す前に待つ秒 返事の遅いエンジンで窓が固まらないかを見る試験で使う
    speakers_delay: float = 0.0
    #: 立つまで声の一覧を返さない 秒で待つ代わりに、試験が返す時を決める（秒の待ちは、遅い
    #: 機械で試験の側が止まると待ちの長さと順番が入れ替わる）
    speakers_gate: threading.Event | None = None
    #: 声の規約を返す前に待つ秒 待つ間に声の欄を変える試験で使う
    policy_delay: float = 0.0
    #: 合成で返す番号 200 以外にすると断る
    synthesis_status: int = 200
    #: 全部の道に返す転送先 空でなければ 302 で飛ばす
    redirect_to: str = ""
    #: 合成ごとのサンプリングレート（塊ごとに変えて、つなげない形を作る試験用）
    rates: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def styles(self) -> set[int]:
        return {
            style["id"]
            for speaker in self.speakers
            for style in speaker["styles"]
            if style.get("type", "talk") == "talk"
        }


def _wav(seconds: float, rate: int = RATE) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as made:
        made.setnchannels(1)
        made.setsampwidth(2)
        made.setframerate(rate)
        made.writeframes(b"\x00\x01" * max(1, round(seconds * rate)))
    return out.getvalue()


def _handler(engine: FakeVoiceEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: object, status: int = 200) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"))

        def _record(self) -> tuple[str, dict[str, list[str]], bytes]:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            parts = urlsplit(self.path)
            with engine.lock:
                engine.requests.append((self.command, self.path, body))
            return parts.path, parse_qs(parts.query), body

        def _redirected(self) -> bool:
            if not engine.redirect_to:
                return False
            self.send_response(302)
            self.send_header("Location", engine.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True

        def do_GET(self) -> None:
            path, query, _body = self._record()
            if self._redirected():
                return
            if path == "/engine_manifest":
                if engine.manifest is None:
                    self._json({"detail": "Not Found"}, 404)
                else:
                    self._json(engine.manifest)
            elif path == "/speakers":
                if engine.speakers_delay:
                    time.sleep(engine.speakers_delay)
                if engine.speakers_gate is not None:
                    engine.speakers_gate.wait(GUARD_SECONDS)
                self._json(engine.speakers)
            elif path == "/speaker_info":
                if engine.policy_delay:
                    time.sleep(engine.policy_delay)
                uuid = (query.get("speaker_uuid") or [""])[0]
                if uuid not in engine.policies:
                    self._json({"detail": "該当する話者が見つかりません"}, 422)
                else:
                    self._json({"policy": engine.policies[uuid], "portrait": "", "style_infos": []})
            else:
                self._json({"detail": "Not Found"}, 404)

        def do_POST(self) -> None:
            path, query, body = self._record()
            if self._redirected():
                return
            speaker = int((query.get("speaker") or ["-1"])[0])
            if path == "/audio_query":
                text = (query.get("text") or [""])[0]
                if not any(char.isalnum() for char in text):
                    # 読む物の無い文（記号だけ）は断る 読みの無い文を断るエンジンの形
                    # （本物がどう返すかは版で違う 断る側を試せるようにする）
                    self._json({"detail": "読める文字がありません"}, 422)
                    return
                if speaker not in engine.styles():
                    # 本物も声の ID が無ければ 422 で、本文に頼まれた物を書き返す
                    self._json({"detail": f"speaker {speaker} が無い 文: {text}"}, 422)
                    return
                with engine.lock:
                    engine.texts.append(text)
                self._json(
                    {
                        "accent_phrases": [],
                        "speedScale": 1.0,
                        "pitchScale": 0.0,
                        "intonationScale": 1.0,
                        "volumeScale": 1.0,
                        "prePhonemeLength": 0.1,
                        "postPhonemeLength": 0.1,
                        "outputSamplingRate": RATE,
                        "outputStereo": False,
                        "kana": text,
                    }
                )
            elif path == "/synthesis":
                if engine.synthesis_delay:
                    time.sleep(engine.synthesis_delay)
                if engine.synthesis_status != 200:
                    self._json({"detail": "合成できない"}, engine.synthesis_status)
                    return
                query_body = json.loads(body.decode("utf-8"))
                with engine.lock:
                    engine.queries.append(query_body)
                    rate = engine.rates.pop(0) if engine.rates else RATE
                text = str(query_body.get("kana", ""))
                self._send(200, _wav(len(text) * SECONDS_PER_CHAR, rate), "audio/wav")
            else:
                self._json({"detail": "Not Found"}, 404)

    return Handler


@contextmanager
def fake_voice_engine(engine: FakeVoiceEngine | None = None) -> Iterator[FakeVoiceEngine]:
    """127.0.0.1 の空いた番号で偽のエンジンを立てる 抜けたら止める"""
    made = engine if engine is not None else FakeVoiceEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(made))
    made.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield made
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def closed_engine_url() -> str:
    """誰も待ち受けていない 127.0.0.1 の場所 エンジンが動いていない機械と同じ"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"
