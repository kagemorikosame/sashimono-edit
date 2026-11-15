"""AI が字幕を起こすときの既定の方式と、頼まれたときだけの変え方（#469）

AI が速さを気にして自分の判断で小さいモデルを選ぶと、同じ素材でも呼ぶたびに字幕の質が
変わる 省いたら本人の設定で起こし、渡した値はその呼び出しだけに効くことを押さえる
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.session import system_prompt
from sashimono.asr.backend import MODELS
from sashimono.core.subtitle_timing import SubtitleTiming
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import FakeHost


def _transcribe(host: FakeHost, **arguments: Any) -> Any:
    operation = find_operation("transcribe")
    assert operation is not None
    media = host.document.project.media[0]
    # 試験の素材には字幕があるので、起こし直しとして頼む 語の時刻だけを足す頼みは今の字幕を
    # 残す頼みなので replace と一緒には渡せない
    asked = {"media_id": str(media.id), **arguments}
    if not arguments.get("words_only"):
        asked["replace"] = True
    return operation(host, asked)


class TestTheDefault:
    def test_leaving_the_model_out_uses_large_v3(self, host: FakeHost) -> None:
        # 壊れると、設定を触っていない人の起こしが AI の決めた別のモデルで走る
        result = _transcribe(host)
        assert host.transcribed_model == "large-v3"
        assert host.transcribed_word_timestamps is True
        assert host.transcribed_timing == SubtitleTiming()
        assert result["model"] == "large-v3"
        assert result["word_timestamps"] is True
        assert result["fit_to_voice"] is True

    def test_the_setting_decides_the_model(self, host: FakeHost) -> None:
        # 壊れると、GPU の無い PC で下書き用へ下げた人の起こしが large-v3 のまま何十分も走る
        host.transcribe_model = "large-v3-turbo"
        result = _transcribe(host)
        assert host.transcribed_model == "large-v3-turbo"
        assert "large-v3-turbo（設定の既定）" in result["method"]

    def test_the_settings_for_words_and_voice_are_followed(self, host: FakeHost) -> None:
        # 本人が設定で切っていれば、AI が省いた起こしもそれに従う
        host.words = False
        host.timing = SubtitleTiming(fit_to_voice=False, start_offset_ms=-200)
        _transcribe(host)
        assert host.transcribed_word_timestamps is False
        assert host.transcribed_timing == SubtitleTiming(fit_to_voice=False, start_offset_ms=-200)

    def test_null_and_empty_are_left_out(self, host: FakeHost) -> None:
        # 省いた引数を null や空で送ってくる呼び方もある 断ると AI が適当なモデル名で呼び直す
        _transcribe(host, model="", word_timestamps=None, fit_to_voice=None)
        assert host.transcribed_model == "large-v3"
        assert host.transcribed_word_timestamps is True


class TestWhenAsked:
    def test_a_named_model_is_used(self, host: FakeHost) -> None:
        # 本人が「small で」と頼んだら従う 頼みを無視して large-v3 で起こすと待たせる
        result = _transcribe(host, model="small")
        assert host.transcribed_model == "small"
        assert "small（指示）" in result["method"]

    def test_an_unknown_model_is_refused(self, host: FakeHost) -> None:
        # 知らない名前を通すと、裏で読み込みに失敗するまで AI には「始めました」しか返らない
        with pytest.raises(ToolError, match="large-v3"):
            _transcribe(host, model="huge")
        assert host.transcribed_model is None

    def test_word_times_and_fitting_follow_only_that_call(self, host: FakeHost) -> None:
        # 「語の時刻は要らない」「声へ寄せないで」はその起こしだけ 次は設定へ戻る
        host.timing = SubtitleTiming(start_offset_ms=-100, end_offset_ms=300)
        result = _transcribe(host, word_timestamps=False, fit_to_voice=False)
        assert host.transcribed_word_timestamps is False
        # ずらす量は本人の好みなので残す
        assert host.transcribed_timing == SubtitleTiming(False, -100, 300)
        assert result["method"] == (
            "large-v3（設定の既定）・語の時刻なし（指示）・声へ寄せない（指示）"
        )
        # 設定の値は書き換えない
        assert host.words is True
        assert host.timing == SubtitleTiming(start_offset_ms=-100, end_offset_ms=300)
        _transcribe(host)
        assert host.transcribed_word_timestamps is True
        assert host.transcribed_timing == host.timing

    def test_a_call_can_turn_them_on_against_the_setting(self, host: FakeHost) -> None:
        # 設定で切っていても、本人が頼めばその起こしだけ取る
        host.words = False
        host.timing = SubtitleTiming(fit_to_voice=False)
        _transcribe(host, word_timestamps=True, fit_to_voice=True)
        assert host.transcribed_word_timestamps is True
        assert host.transcribed_timing == SubtitleTiming()

    def test_a_string_flag_is_refused(self, host: FakeHost) -> None:
        # "false" を真と読むと、頼みと逆の方式で数分起こす
        with pytest.raises(ToolError, match="fit_to_voice"):
            _transcribe(host, fit_to_voice="false")
        assert host.transcribed_model is None


class TestWordsOnly:
    def test_words_only_without_word_times_is_refused(self, host: FakeHost) -> None:
        # 語の時刻を足す頼みで語の時刻を取らないと、何も足さない起こしが数分走る
        with pytest.raises(ToolError, match="word_timestamps"):
            _transcribe(host, words_only=True, word_timestamps=False)
        assert host.transcribed_model is None

    def test_words_only_with_fitting_is_refused(self, host: FakeHost) -> None:
        # 語の時刻だけを足す頼みは行の時刻を変えない 受けると寄せたと思わせる
        with pytest.raises(ToolError, match="retime_subtitles"):
            _transcribe(host, words_only=True, fit_to_voice=True)
        assert host.transcribed_model is None

    def test_words_only_takes_word_times_even_when_the_setting_is_off(self, host: FakeHost) -> None:
        # 設定で語の時刻を切っていても、足す頼みそのものが語の時刻を取る頼み
        host.words = False
        result = _transcribe(host, words_only=True)
        assert host.transcribed_words_only is True
        assert host.transcribed_timing is not None
        assert host.transcribed_timing.fit_to_voice is False
        assert "語の時刻だけ" in result["method"]


class TestTheWords:
    def test_the_tool_says_to_change_only_when_asked(self) -> None:
        # 書いていないと、AI は速さを気にして small を選ぶ（#469 の起こり）
        operation = find_operation("transcribe")
        assert operation is not None
        assert "自分の判断で小さいモデル" in operation.description
        properties = operation.schema["properties"]
        assert properties["model"]["enum"] == [info.name for info in MODELS]
        for name in ("model", "word_timestamps", "fit_to_voice"):
            assert "頼んだときだけ" in properties[name]["description"]

    def test_the_instructions_say_so_too(self) -> None:
        # 壊れると、頼まれていないのに小さいモデルを選ばない決まりが指示の文から消える
        prompt = system_prompt()
        assert "自分の" in prompt and "小さいモデル" in prompt
        assert "method" in prompt


class TestTheSetting:
    def test_it_is_saved_and_old_files_get_large_v3(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        """壊れると、設定画面で選んだモデルが次の起動で消えるか、古い設定で起動が止まる"""
        del qt_application
        assert Preferences().ai_transcribe_model == "large-v3"
        chosen = Preferences(ai_transcribe_model="small")
        dialog = PreferencesDialog(chosen)
        try:
            assert dialog.preferences().ai_transcribe_model == "small"
        finally:
            dialog.deleteLater()
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(chosen)
        assert store.load().ai_transcribe_model == "small"
        # 項目の無い前の版の設定と、壊れた値は既定へ戻す
        store.path.write_text(json.dumps({"word_timestamps": False}), "utf-8")
        assert store.load().ai_transcribe_model == "large-v3"
        store.path.write_text(json.dumps({"ai_transcribe_model": "huge"}), "utf-8")
        assert store.load().ai_transcribe_model == "large-v3"
