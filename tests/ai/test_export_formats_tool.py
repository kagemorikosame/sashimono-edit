"""AI の書き出し（export_video）で形式を選ぶ（#527）

2026-11-24 に利用者が決めた 音声だけの書き出しは別の道具にせず、export_video の形式の 1 つ
（確かめ＝同意の作りを 1 つにする 画面の書き出しの窓と同じ形） 形式は名前の拡張子で決まる
"""

from __future__ import annotations

from typing import Any

import pytest

from sashimono.ai import operations
from sashimono.ai.bridge import EditorBridge
from sashimono.ai.host import ToolError
from tests.ai.conftest import FakeHost, allow_sending
from tests.ai.test_save_and_export import (
    EXPORT,
    WINDOWS_WRITES,
    _run,
    codecs,
    saved,
)

__all__ = ["codecs", "saved"]


@pytest.fixture
def web_codecs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """WebM で使える物を決めておく 機械で確かめの文面が変わらないように"""
    names = ["libvpx-vp9", "libsvtav1"]
    monkeypatch.setattr(operations, "available_web_codecs", lambda: list(names))
    return names


@pytest.fixture
def sounding(monkeypatch: pytest.MonkeyPatch) -> None:
    """偽のホストの作品に鳴る物があることにする（音声だけの書き出しの断りに当てない）"""
    monkeypatch.setattr(operations, "has_sound", lambda _project: True)


@WINDOWS_WRITES
@pytest.mark.usefixtures("sounding")
def test_an_audio_only_export_says_so_and_keeps_the_quality(saved: FakeHost) -> None:
    """壊れると、音だけを頼んだのに確かめの箱が「動画」を言うか、選んだ音質が届かない"""
    assert saved.saved_at is not None
    target = saved.saved_at.parent / "BGM.mp3"
    bridge = allow_sending(EditorBridge(saved, auto_approve=True))
    seen, box = _run(bridge, EXPORT, {"path": str(target), "audio_kbps": 320})
    assert box.get("error") is None
    summary = seen[0].summary
    assert "音声を書き出します" in summary and "MP3" in summary and "320 kbps" in summary
    assert "Mbps" not in summary
    request = saved.exports[-1]
    assert request.path == target.resolve()
    assert request.audio_kbps == 320 and request.codec is None
    assert request.match_bitrate is False


@WINDOWS_WRITES
@pytest.mark.usefixtures("sounding")
def test_the_audio_quality_defaults_like_the_window(saved: FakeHost) -> None:
    """壊れると、AI の音声だけの書き出しが窓の初めの値（192 kbps）と違う音質になる"""
    assert saved.saved_at is not None
    bridge = allow_sending(EditorBridge(saved, auto_approve=True))
    _run(bridge, EXPORT, {"path": str(saved.saved_at.parent / "声.m4a")})
    assert saved.exports[-1].audio_kbps == 192


@WINDOWS_WRITES
def test_webm_offers_vp9_and_av1(saved: FakeHost, web_codecs: list[str]) -> None:
    """壊れると、WebM に H.264 の候補を並べ、許した後に入れ物に断られる"""
    del web_codecs
    assert saved.saved_at is not None
    bridge = allow_sending(EditorBridge(saved, auto_approve=True))
    _seen, box = _run(
        bridge, EXPORT, {"path": str(saved.saved_at.parent / "web.webm"), "codec": "libsvtav1"}
    )
    assert box.get("error") is None
    assert saved.exports[-1].codec == "libsvtav1"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"path": "a.wav", "codec": "libx264"}, "音声だけ"),
        ({"path": "a.mp3", "codec": "auto"}, "音声だけ"),
        ({"path": "a.flac", "bitrate_mbps": 8}, "音声だけ"),
        ({"path": "a.mp4", "audio_kbps": 320}, "audio_kbps"),
        ({"path": "a.wav", "audio_kbps": 320}, "audio_kbps"),
        ({"path": "a.mp3", "audio_kbps": 100}, "audio_kbps"),
        ({"path": "a.mp3", "audio_kbps": 192.0}, "audio_kbps"),
        ({"path": "a.webm", "codec": "libx264"}, "codec"),
        ({"path": "a.ogg"}, ".webm"),
    ],
)
def test_wrong_choices_are_refused_before_asking(
    saved: FakeHost,
    codecs: list[str],
    web_codecs: list[str],
    arguments: dict[str, Any],
    message: str,
) -> None:
    """壊れると、効かない引数を黙って捨てたまま確かめが出るか、書けない形式で許してから失敗する"""
    del codecs, web_codecs
    bridge = allow_sending(EditorBridge(saved, auto_approve=True))
    seen, box = _run(bridge, EXPORT, arguments)
    assert seen == []
    assert isinstance(box.get("error"), ToolError)
    assert message in str(box["error"])
    assert saved.exports == []


def test_webm_auto_without_vp9_is_refused(saved: FakeHost, monkeypatch: pytest.MonkeyPatch) -> None:
    """壊れると、VP9 の無い機械で自動の WebM を許し、書き出しで失敗する"""
    monkeypatch.setattr(operations, "available_web_codecs", lambda: ["libsvtav1"])
    bridge = allow_sending(EditorBridge(saved, auto_approve=True))
    seen, box = _run(bridge, EXPORT, {"path": "a.webm"})
    assert seen == []
    assert "VP9" in str(box.get("error"))
    assert saved.exports == []


@WINDOWS_WRITES
@pytest.mark.usefixtures("sounding")
def test_a_link_to_another_format_is_refused(
    saved: FakeHost, codecs: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、MP3 の名前で確かめを出し、リンク先の MP4 へ書く（許した物と書いた物が違う）

    ファイルへのリンクは Windows では権限が要るので、リンクを辿った後の場所を返す所を差し替える
    """
    del codecs
    assert saved.saved_at is not None
    video = saved.saved_at.parent / "本編.mp4"
    link = saved.saved_at.parent / "音.mp3"
    monkeypatch.setattr(operations, "_write_target", lambda _host, _raw: video)
    bridge = allow_sending(EditorBridge(saved, auto_approve=True))
    seen, box = _run(bridge, EXPORT, {"path": str(link)})
    assert seen == []
    assert "リンク先" in str(box.get("error"))
    assert saved.exports == []


def test_a_project_without_sound_is_refused_for_audio_only(
    saved: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """壊れると、鳴る物の無い作品で確かめが出て、許すと無音のファイルを書くか失敗する"""
    monkeypatch.setattr(operations, "has_sound", lambda _project: False)
    bridge = allow_sending(EditorBridge(saved, auto_approve=True))
    seen, box = _run(bridge, EXPORT, {"path": "a.wav"})
    assert seen == []
    assert "鳴る物が無い" in str(box.get("error"))
    assert saved.exports == []
