"""演出のレシピを引く道具と、強さとジャンルの設定（#470 段 3 #478） get_direction_recipes

レシピそのものと選び方は tests/core/test_direction_recipes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from sashimono.ai.host import ToolError
from sashimono.ai.operations import find_operation
from sashimono.ai.outbound import Outbound
from sashimono.ai.safety import Safety
from sashimono.ai.session import system_prompt
from sashimono.core.commands import SetStyleGuide
from sashimono.core.direction_recipes import BOLD, NORMAL, SUBTLE, DirectionTaste
from sashimono.core.model import StyleGuide
from sashimono.ui.preferences_dialog import PreferencesDialog
from sashimono.ui.workspace import Preferences, PreferenceStore
from tests.ai.conftest import FakeHost
from tests.ai.test_operations import run


class TestTool:
    def test_is_read_only_and_says_what_it_sends(self) -> None:
        # 読むだけでないと、変更の確かめを切った人に毎回確かめが出る 本人の設定を返すので
        # アプリの設定を送ると書かないと、送らないと決めた人に黙って送る
        operation = find_operation("get_direction_recipes")
        assert operation is not None
        assert operation.safety is Safety.READ_ONLY
        assert operation.sends == {Outbound.PROJECT, Outbound.APP_SETTINGS}

    def test_uses_the_setting_unless_asked(self, host: FakeHost) -> None:
        # 設定を読まないと、設定画面で控えめを選んでも AI が自分の加減で付ける
        host.taste = DirectionTaste(strength=NORMAL)
        result = run(host, "get_direction_recipes")
        assert (result["genre_name"], result["strength_name"]) == ("実況", "ふつう")
        assert result["from_settings"] is True
        zoom = next(row for row in result["recipes"] if row["key"] == "laugh_zoom")
        assert (zoom["per_minute"], zoom["amount"], zoom["cue"]) == (2.0, 120, "laughter")
        # 頼みの文で「派手に」と言われたときは上書きでき、上書きしたと分かる
        loud = run(host, "get_direction_recipes", strength=BOLD)
        assert (loud["strength"], loud["from_settings"]) == (BOLD, False)
        loud_zoom = next(row for row in loud["recipes"] if row["key"] == "laugh_zoom")
        assert loud_zoom["per_minute"] > zoom["per_minute"]

    def test_leaves_out_recipes_outside_the_guide(self, host: FakeHost) -> None:
        # 方針の外の効果を使う演出を付けると、照らしで本人に直させる
        guide = StyleGuide(effects=("glow",), notes="本人の秘密の文")
        host.document.execute(SetStyleGuide(guide))
        result = run(host, "get_direction_recipes")
        assert [row["key"] for row in result["left_out"]] == ["loud_shake"]
        assert "loud_shake" not in [row["key"] for row in result["recipes"]]
        # 方針の文は返さない（レシピの道具は本文を送ると書いていない）
        assert "本人の秘密の文" not in json.dumps(result, ensure_ascii=False)

    def test_refuses_unknown_values(self, host: FakeHost) -> None:
        # 黙って既定へ戻すと、頼んだ強さと違う数を正しいと思って使う
        with pytest.raises(ToolError, match="strength"):
            run(host, "get_direction_recipes", strength="最強")
        with pytest.raises(ToolError, match="genre"):
            run(host, "get_direction_recipes", genre="drama")

    def test_the_instructions_point_to_the_recipes(self) -> None:
        # 指示に無いと、AI はレシピを読まずに自分の加減で演出を付ける
        prompt = system_prompt()
        assert "get_direction_recipes" in prompt
        # 効果音の出どころは本人が登録したフォルダか取り込んだ音だけ（#522） 無ければ付けない
        assert "search_library" in prompt
        assert "sound_file_id" in prompt
        assert "どちらも無ければ付けずに" in prompt


class TestTheSetting:
    def test_it_is_saved_and_unknown_values_fall_back(
        self, qt_application: QApplication, tmp_path: Path
    ) -> None:
        """壊れると、設定画面で選んだ強さが次の起動で消えるか、古い設定で起動が止まる"""
        del qt_application
        plain = Preferences()
        assert plain.direction_taste == DirectionTaste(strength=SUBTLE)
        chosen = Preferences(direction_strength=BOLD)
        dialog = PreferencesDialog(chosen)
        try:
            assert dialog.preferences().direction_strength == BOLD
            assert dialog.preferences().direction_genre == plain.direction_genre
        finally:
            dialog.deleteLater()
        store = PreferenceStore(tmp_path / "preferences.json")
        store.save(chosen)
        assert store.load().direction_taste == DirectionTaste(strength=BOLD)
        # 項目の無い前の版の設定と、壊れた値は既定へ戻す
        store.path.write_text(json.dumps({"word_timestamps": False}), "utf-8")
        assert store.load().direction_strength == SUBTLE
        store.path.write_text(
            json.dumps({"direction_strength": "最強", "direction_genre": "drama"}), "utf-8"
        )
        assert store.load().direction_taste == DirectionTaste()
