"""互換性の記録の写しを足し込む（#453 別のプロセスの書き出しの記録を親へ戻す）"""

from __future__ import annotations

from sashimono.compat.aviutl.report import MAX_KINDS, OVERFLOW, CompatibilityReport


def test_merging_adds_counts_and_keeps_the_first_failure() -> None:
    """壊れると、子と親で同じ API を呼んだ回数が合わないか、親が先に見た失敗の文が子の文で
    上書きされる"""
    parent = CompatibilityReport()
    parent.note_missing("obj.getpixeldata")
    parent.note_failure("ぼかし.anm", "親で見た")
    child = CompatibilityReport()
    child.note_missing("obj.getpixeldata")
    child.note_missing("obj.getpixeldata")
    child.note_control("<#zz>")
    child.note_failure("ぼかし.anm", "子で見た")
    child.note_failure("揺れ.anm", "壊れた")
    parent.merge(child.records())
    missing, controls, failures = parent.records()
    assert missing == (("obj.getpixeldata", 3),)
    assert controls == (("<#zz>", 1),)
    assert failures == (("ぼかし.anm", "親で見た"), ("揺れ.anm", "壊れた"))


def test_merging_keeps_the_kind_limit() -> None:
    """壊れると、子と親の記録を合わせたときに種類の上限を越えて、記録だけでメモリを使う"""
    parent = CompatibilityReport()
    for index in range(MAX_KINDS):
        parent.note_missing(f"親 {index}")
    child = CompatibilityReport()
    child.note_missing("子だけ")
    parent.merge(child.records())
    assert len(parent.missing) == MAX_KINDS + 1
    assert parent.missing[OVERFLOW] == 1
