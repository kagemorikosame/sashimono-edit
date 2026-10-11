"""README に書いたエフェクトの数が、登録の数とそろっていること

数え方の決まり
- 数えるのは :data:`sashimono.effects.registry` に入っている自前のエフェクト
- AviUtl の配布スクリプト（``aviutl:`` で始まる種類）は入れない 読み込んだスクリプトの分だけ
  増え、README に書く数が手元の環境で変わる
- 音は :attr:`EffectDefinition.audio_process` を持つ物、映像はそれ以外 合計と内訳の両方を書く

エフェクトを足して README の数を直し忘れると、この試験が落ちる（#274 で 80 種のまま
105 種まで増えていたのに気付いた）
"""

from __future__ import annotations

import re
from pathlib import Path

from sashimono.effects import registry

README = Path(__file__).resolve().parents[1] / "README.md"
#: README の 1 行 「エフェクト 105 種（映像 99 種・音 6 種）」の形で書く
PATTERN = re.compile(r"エフェクト (\d+) 種（映像 (\d+) 種・音 (\d+) 種）")


def _counted() -> tuple[int, int, int]:
    own = [d for d in registry.all() if not d.kind.startswith("aviutl:")]
    audio = sum(1 for d in own if d.audio_process is not None)
    return len(own), len(own) - audio, audio


def test_the_readme_counts_every_effect() -> None:
    found = PATTERN.findall(README.read_text(encoding="utf-8"))
    assert len(found) == 1, "README にエフェクトの数の行が無いか、2 か所ある"
    written = tuple(int(n) for n in found[0])
    total, picture, sound = _counted()
    assert written == (total, picture, sound), (
        f"README は {written}、登録は（合計 {total}・映像 {picture}・音 {sound}）"
        " エフェクトを足したら README の数も直す"
    )
