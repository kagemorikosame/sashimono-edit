"""HDR や広い色域の素材を読み込んだときの知らせ

Sashimono が扱う色は SDR（Rec.709）だけで、HDR（PQ・HLG）と BT.2020 の素材は変換せずに
SDR として描く（docs/development.md「色の範囲」） 何も言わずに出すと、白っぽく褪せた絵を
ソフトの不具合だと受け取られる 読み込んだときに 1 度だけ知らせ、素材一覧の行にも印を付ける

知らせる窓は設定（:attr:`Preferences.hdr_notice`）で切れる 窓の〔次から知らせない〕も同じ設定を切る
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtWidgets import QCheckBox, QMessageBox, QWidget

from sashimono.core.model import MediaItem

__all__ = ["HDR_NOTE", "ask_hdr_notice", "describe_outside_sdr", "outside_sdr"]

#: 素材一覧の行に添える説明 窓の文と同じことを短く言う
HDR_NOTE = "SDR（Rec.709）として扱うので、白っぽく表示・書き出しされます"

#: 窓に名前を並べる上限 何十本もまとめて読み込んだときに、窓が画面より縦に長くならないように
LISTED = 8


def outside_sdr(media: Sequence[MediaItem]) -> list[MediaItem]:
    """SDR の範囲の外の色の印が付いた素材"""
    return [item for item in media if item.color_outside_sdr]


def describe_outside_sdr(media: Sequence[MediaItem]) -> str:
    """窓に出す文 素材の名前と、HDR か広色域か"""
    lines = [f"・{item.name}（{item.color_outside_sdr}）" for item in media[:LISTED]]
    if len(media) > LISTED:
        lines.append(f"ほか {len(media) - LISTED} 本")
    return (
        "次の素材は HDR または広い色域（BT.2020）です\n\n"
        + "\n".join(lines)
        + "\n\nSashimono は SDR（Rec.709）として扱うので、白っぽく褪せた色で"
        "表示・書き出しされます 元の色のまま使いたいときは、ほかのソフトで SDR に"
        "変換（トーンマッピング）してから読み込んでください"
    )


def ask_hdr_notice(parent: QWidget | None, media: Sequence[MediaItem]) -> bool:
    """知らせる窓を出す 次からも知らせるなら真（〔次から知らせない〕に印を付けたら偽）

    試験では差し替える（``tests/conftest.py``） 窓を出すと、閉じる人がいないまま止まる
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle("HDR の素材")
    box.setText(describe_outside_sdr(media))
    stop = QCheckBox("次から知らせない（表示 → 設定… で戻せます）", box)
    box.setCheckBox(stop)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()
    keep = not stop.isChecked()
    box.deleteLater()
    return keep
