"""AI のパネルの演出の計画の欄（#470 段 4 #479）

AI の道具が出した計画が欄に項目ごとの印で出ること、印を外した物を除いて〔付ける〕で
1 回に付くこと、AI が応えている間は付けないこと、〔やめる〕〔×〕を見る
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from sashimono.ai.direction_plan import APPLY_LABEL, ProposalState, proposal_for
from sashimono.ai.operations import find_operation
from sashimono.core.commands import AddMedia
from sashimono.core.direction_recipes import BOLD, DirectionTaste
from sashimono.core.model import Clip, MediaItem, Project, ProjectSettings, Track, TrackKind
from sashimono.ui.chat import ChatPanel
from sashimono.ui.chat.direction_box import item_text
from tests.ai.conftest import FakeHost
from tests.conftest import RATE_30
from tests.ui.test_chat_panel import _ready, _RecordingSession

_SECONDS = 120


@pytest.fixture(autouse=True)
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """窓が開いたらその場で落とす 欄はパネルの中に出す物で、どの操作も窓を出さない"""

    def refuse(*_arguments: object, **_options: object) -> int:
        raise AssertionError("試験の間に窓が開いた")

    monkeypatch.setattr(QDialog, "exec", refuse)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, refuse)


@pytest.fixture
def made(
    qt_application: QApplication, video_media: MediaItem, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ChatPanel, FakeHost]]:
    from sashimono.ui.chat import panel as panel_module

    del qt_application
    _ready(monkeypatch)
    _RecordingSession.made = []
    monkeypatch.setattr(panel_module, "AgentSession", _RecordingSession)
    media = replace(video_media, duration=Fraction(_SECONDS))
    project = AddMedia(media).apply(Project.create(ProjectSettings(frame_rate=RATE_30)))
    clip = Clip(timeline_start=0, duration=_SECONDS * 30, media_id=media.id)
    track = Track(kind=TrackKind.VIDEO, name="V1", clips=(clip,))
    host = FakeHost(project.with_timeline(replace(project.timeline, tracks=(track,))))
    host.taste = DirectionTaste(strength=BOLD)
    widget = ChatPanel(host)
    widget._refresh_availability()
    yield widget, host
    widget.close_session()
    widget.deleteLater()


def _propose(host: FakeHost) -> list[int]:
    operation = find_operation("propose_direction")
    assert operation is not None
    cues = [
        {"signal": "laughter", "start_frame": t * 30, "end_frame": t * 30 + 30, "score": 0.5}
        for t in range(5, _SECONDS, 20)
    ]
    result = operation(host, {"cues": cues})
    assert isinstance(result, dict)
    return [row["number"] for row in result["items"]]


class TestDirectionBox:
    def test_no_plan_no_box(self, made: tuple[ChatPanel, FakeHost]) -> None:
        # 計画の無い間に空の欄が出ると、会話の欄の高さを食う
        widget, _ = made
        widget._poll()
        assert widget.direction_box.isHidden() is True

    def test_unchecked_items_are_left_out_and_the_rest_is_one_undo(
        self, made: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 外した物が付くと一覧で外した意味が無い 段が分かれると戻すのに何回も押す
        widget, host = made
        before = host.document.project
        numbers = _propose(host)
        assert len(numbers) >= 2
        widget._poll()
        box = widget.direction_box
        assert box.isHidden() is False
        view = proposal_for(host.document).view()
        assert view is not None
        assert box.check_box(numbers[0]).text() == item_text(view.items[0])
        box.check_box(numbers[0]).setChecked(False)
        view = proposal_for(host.document).view()
        assert view is not None and view.excluded == {numbers[0]}
        assert f"{len(numbers) - 1}/{len(numbers)}" in box.title()
        assert box.apply_button.isEnabled() is True
        box.apply_button.click()
        view = proposal_for(host.document).view()
        assert view is not None and view.state is ProposalState.APPLIED
        assert view.result is not None and numbers[0] not in view.result.applied
        assert host.document.undo_label == APPLY_LABEL
        assert "取り消し 1 回で全部戻ります" in widget._view.toPlainText()
        assert box.apply_button.isHidden() is True
        assert box.close_button.isEnabled() is True
        host.document.undo()
        assert host.document.project == before
        box.close_button.click()
        assert box.isHidden() is True

    def test_does_not_apply_while_the_ai_is_answering(
        self, made: tuple[ChatPanel, FakeHost]
    ) -> None:
        # 応えている最中に付けると、演出が AI の段に混ざり、計画だけを取り消せない
        widget, host = made
        before = host.document.project
        _propose(host)
        widget._input.setPlainText("ほかも直して")
        widget.send()
        widget._poll()
        assert widget.direction_box.apply_button.isEnabled() is False
        widget.apply_direction()
        assert host.document.project is before

    def test_cancel_adds_nothing(self, made: tuple[ChatPanel, FakeHost]) -> None:
        # やめた後に〔付ける〕が効くと、やめたはずの演出が入る
        widget, host = made
        before = host.document.project
        _propose(host)
        widget._poll()
        box = widget.direction_box
        # 出している間は閉じない 閉じると外した印も〔付ける〕も消える
        assert box.close_button.isEnabled() is False
        box.cancel_button.click()
        assert host.document.project is before
        assert "やめました" in box.status_text()
        assert box.apply_button.isHidden() is True
