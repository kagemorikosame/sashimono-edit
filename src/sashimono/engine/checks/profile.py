"""音の推移を 100 ms ごとに覚え、ラウドネスの推移・無音・話し声と音楽の見分けを出す（#322 #338）

完成物の検査（:mod:`.scan`）が混ぜた音を 1 塊ずつ受け取るついでに、ここへも渡す 音を
混ぜ直さずに済み、検査と同じ音（書き出す音）を見る 外のモデルは使わない（#298 の方針
配る zip の大きさと使用許諾に響く） 使うのは numpy の FFT だけ

100 ms ごとに覚える物

- K 特性を掛けた平均の二乗（:class:`~.loudness.LoudnessMeter` の区切りをそのまま使う）
  ここから瞬時（400 ms）・短時間（3 秒）のラウドネスと、積分ラウドネス・ラウドネスの幅
  （EBU Tech 3342 の LRA）を出す
- 一番大きいチャンネルの RMS と最大のサンプル（無音と音割れの目安）
- スペクトルの平たさ（左右を混ぜた音の 100〜8000 Hz） 雑音は平たく、楽器と声は山がある

話し声と音楽の見分け（:func:`classify`）の手がかりと限り

- 字幕（文字起こしの区間）が重なる所は話し声 認識器が文字を起こせた所は確かに話している
- 字幕の無い所は音の形で見る 話し声は音節ごとに大きさが上下し、息継ぎで小さくなる区切りが
  多い（前後 1 秒を含む 3 秒の中で、平均の半分を下回る 100 ms の割合 Scheirer と Slaney の
  low-energy ratio と、大きさの揺れと、大きな所と谷の差） 音楽は音が途切れず、谷も鳴っている
- 話し声の谷（静かな 4 分の 1 の区切り）に、無音より大きく山のある音（平たくない音）が
  残っていれば、下に BGM が敷かれているとみなす 谷に残るのが平たい音なら部屋の雑音
- しきい値は合成した音と、本人の許した実際の素材（話し声だけ・音楽だけ・歌と伴奏を分けた物・
  ゲームの録画）で決めた 話し声だけの素材の 94%、音楽だけの素材の 90〜94% を当てた
- 限り: 歌は音楽として返す（歌だけの素材も音楽と読む） ラップ・笑い声・拍手は外れやすい
  ゲームの効果音や中の声は話し声に寄り、録画の話していない所の半分ほどを話し声と読む
  打楽器だけの音楽と、間の多い弾き語りは話し声に似る 谷の音が無音のしきい値より小さい
  BGM は見分けられない
  1 秒ごとに見るので、1 秒より短い切り替わりは拾わない 確からしさ（confidence）は
  これを踏まえて付ける（字幕と合えば high、音の形だけなら medium まで）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sashimono.engine.checks.loudness import (
    ABSOLUTE_GATE,
    LoudnessMeter,
    block_loudness,
    gated_loudness,
)

__all__ = [
    "LABELS",
    "MUSIC",
    "NOISE",
    "SILENCE",
    "SPEECH",
    "SPEECH_AND_MUSIC",
    "TICK_SECONDS",
    "UNCLEAR",
    "WINDOW_TICKS",
    "SoundProfile",
    "SoundProfiler",
    "SoundSegment",
    "classify",
]

#: 覚える区切りの長さ（秒） ラウドネスの区切り（BS.1770 の 100 ms のずらし）に合わせる
TICK_SECONDS = 0.1
#: 見分けの 1 区切り（区切りの数 = 1 秒） 話し声の音節（0.1〜0.3 秒）を何個か含む長さ
WINDOW_TICKS = 10
#: 見分けで前後に見る広さ（区切りの数 前後 1 秒ずつ） 息継ぎの間（0.3〜1 秒）が入る長さ
#: 1 秒だけだと、長い 1 音節の途中で「大きさが上下しない」と読み、音楽に倒れる
CONTEXT_TICKS = 10

SILENCE = "silence"
SPEECH = "speech"
MUSIC = "music"
SPEECH_AND_MUSIC = "speech_and_music"
NOISE = "noise"
UNCLEAR = "unclear"
#: 見分けの名前（並びは説明の順）
LABELS = (SPEECH, MUSIC, SPEECH_AND_MUSIC, SILENCE, NOISE, UNCLEAR)

HIGH = "high"
MEDIUM = "medium"
LOW = "low"
_RANK = {LOW: 0, MEDIUM: 1, HIGH: 2}

#: 平たさを測る帯（Hz） 低い唸り（電源の 50/60 Hz）と、圧縮で削られた高い所を外す
_FLAT_BAND = (100.0, 8000.0)
#: これより平たい音は雑音とみなす（白色雑音で 0.56 ほど、声と楽器は 0.2 未満が多い
#: 合成した音声・和音・雑音で測った）
_NOISY = 0.4
#: 谷の音がこれより平たくなければ、山のある音（BGM）が敷かれているとみなす
_TONAL_BED = 0.3
#: 谷の音と大きな所の差がこれより小さければ、谷にも音が残っている（dB） 話し声だけなら
#: 谷は無音に近く 60 dB を超えて開く BGM を -10 dB で敷くと 20 dB ほど、-20 dB で 30 dB ほど
_BED_GAP = 40.0
#: 大きな所と谷の差（dB）の、音楽らしい端と話し声らしい端 実際の素材で、音楽は 10 dB を
#: 超えることが少なく、話し声は 25 dB を下回ることが少なかった
_GAP_LOW = 10.0
_GAP_HIGH = 25.0
#: 下に音が敷かれているとき、これだけ話し声らしければ敷いた音の上の話し声とみなす
_BED_SPEECH = 0.35
#: 小さすぎて比べられない値の下限 log と割り算を守る
_TINY = 1e-12


class SoundProfiler:
    """混ぜた音を少しずつ受け取り、100 ms ごとの手がかりを覚える

    渡す音は ``(サンプル数, チャンネル数)`` の float 端数は次の塊へ持ち越すので、
    どんな長さで渡しても同じ値になる
    """

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.meter = LoudnessMeter(sample_rate, channels)
        self._rate = int(sample_rate)
        self._channels = int(channels)
        self._step = self.meter.block_samples
        self._partial = np.zeros((0, self._channels))
        self._rms: list[float] = []
        self._peak: list[float] = []
        self._flat: list[float] = []
        self._hann = np.hanning(self._step)
        freqs = np.fft.rfftfreq(self._step, 1.0 / self._rate)
        self._flat_band = (freqs >= _FLAT_BAND[0]) & (freqs < _FLAT_BAND[1])

    def add(self, block: np.ndarray) -> None:
        """音を渡す ラウドネスの区切りも同じ所から測る"""
        samples = np.asarray(block, dtype=np.float64).reshape(-1, self._channels)
        if len(samples) == 0:
            return
        self.meter.add(samples)
        joined = np.concatenate([self._partial, samples])
        whole = len(joined) // self._step * self._step
        self._partial = joined[whole:]
        if not whole:
            return
        ticks = joined[:whole].reshape(-1, self._step, self._channels)
        # 一番大きいチャンネルで見る 片方だけ鳴っている音（左右に振った声）を無音と読まない
        rms = np.sqrt(np.mean(ticks * ticks, axis=1)).max(axis=1)
        peak = np.abs(ticks).max(axis=(1, 2))
        mono = ticks.mean(axis=2) * self._hann
        power = np.abs(np.fft.rfft(mono, axis=1)[:, self._flat_band]) ** 2 + _TINY
        flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
        self._rms.extend(rms.tolist())
        self._peak.extend(peak.tolist())
        self._flat.extend(flatness.tolist())

    def snapshot(self) -> SoundProfile:
        """ここまでの写し 走り係が足し続けても写しは変わらない"""
        energies = self.meter.block_energies()
        count = min(len(energies), len(self._rms))
        return SoundProfile(
            energy=energies[:count] / self._step,
            rms=np.asarray(self._rms[:count]),
            peak=np.asarray(self._peak[:count]),
            flatness=np.asarray(self._flat[:count]),
        )


@dataclass(frozen=True, slots=True, eq=False)
class SoundProfile:
    """100 ms ごとの手がかりの並び 並びの長さはどれも同じ（区切りの数）

    ``eq=False`` なのは numpy の配列を持つため（``==`` が配列を返し、比べると落ちる）
    """

    #: K 特性とチャンネルの重みを掛けた平均の二乗
    energy: np.ndarray
    #: 一番大きいチャンネルの RMS（1.0 がフルスケール）
    rms: np.ndarray
    peak: np.ndarray
    #: スペクトルの平たさ 0（澄んだ音）〜1（雑音）
    flatness: np.ndarray

    @property
    def ticks(self) -> int:
        return len(self.energy)

    def _sliding(self, width: int) -> np.ndarray:
        """``width`` 区切りの平均の二乗 区切りの終わりで揃える 頭の足りない所は有る分で割る"""
        if self.ticks == 0:
            return np.zeros(0)
        sums = np.cumsum(np.concatenate([[0.0], self.energy]))
        ends = np.arange(1, self.ticks + 1)
        starts = np.maximum(0, ends - width)
        return np.asarray((sums[ends] - sums[starts]) / (ends - starts))

    def momentary(self) -> np.ndarray:
        """瞬時のラウドネス（400 ms LUFS）を区切りごとに 無音は ``-inf``"""
        return block_loudness(self._sliding(4))

    def short_term(self) -> np.ndarray:
        """短時間のラウドネス（3 秒 LUFS）を区切りごとに 無音は ``-inf``"""
        return block_loudness(self._sliding(30))

    def integrated(self, start: int = 0, end: int | None = None) -> float | None:
        """``[start, end)`` の区切りの積分ラウドネス（LUFS） 測れなければ ``None``

        400 ms の区切りが 1 つも作れない短い所（4 区切り未満）は測らない（規格どおり）
        """
        energy = self.energy[start:end]
        if len(energy) < 4:
            return None
        blocks = (energy[:-3] + energy[1:-2] + energy[2:-1] + energy[3:]) / 4.0
        return gated_loudness(blocks)

    def loudness_range(self) -> float | None:
        """ラウドネスの幅（LU EBU Tech 3342 の LRA） 大きいほど静かな所と大きな所の差が大きい

        3 秒の区切りを -70 LUFS で切り、残りの平均より 20 LU 静かな所も捨て、10% と 95% の
        間の幅を取る 3 秒に満たない作品では測らない
        """
        if self.ticks < 30:
            return None
        blocks = self._sliding(30)[29:]
        levels = block_loudness(blocks)
        kept = blocks[levels > ABSOLUTE_GATE]
        if len(kept) == 0:
            return None
        relative = float(block_loudness(np.array([np.mean(kept)]))[0]) - 20.0
        gated = levels[(levels > ABSOLUTE_GATE) & (levels > relative)]
        if len(gated) == 0:
            return None
        return float(np.percentile(gated, 95) - np.percentile(gated, 10))

    def rms_db(self) -> np.ndarray:
        """区切りごとの RMS（dBFS） 無音は -120"""
        return _decibels(self.rms)

    def peak_db(self) -> np.ndarray:
        """区切りごとの最大のサンプル（dBFS） 無音は -120"""
        return _decibels(self.peak)

    def silences(self, threshold_db: float, min_ticks: int) -> list[tuple[int, int, float]]:
        """``threshold_db`` より小さい区切りが ``min_ticks`` 以上続く所

        ``(頭, 終わり, その中の最大 dBFS)`` の並び
        """
        levels = self.rms_db()
        quiet = levels < threshold_db
        found: list[tuple[int, int, float]] = []
        edges = np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]]))
        starts = np.flatnonzero(edges == 1).tolist()
        ends = np.flatnonzero(edges == -1).tolist()
        for start, end in zip(starts, ends, strict=True):
            if end - start >= min_ticks:
                found.append((start, end, float(levels[start:end].max())))
        return found


def _decibels(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.asarray(np.maximum(-120.0, 20.0 * np.log10(np.maximum(values, 0.0))))


@dataclass(frozen=True, slots=True)
class SoundSegment:
    """見分けた区間 区切りの番号で ``[start, end)``"""

    start: int
    end: int
    label: str
    #: high / medium / low
    confidence: str
    #: 話し声らしさ・音楽らしさ（0〜1） 区間の 1 秒ごとの値の平均
    speech_score: float
    music_score: float
    #: 字幕が重なっている割合（0〜1） 字幕を渡さなければ ``None``
    subtitles: float | None


@dataclass(frozen=True, slots=True)
class _Window:
    label: str
    confidence: str
    speech: float
    music: float
    subtitles: float | None


@dataclass(frozen=True, slots=True)
class _Cues:
    speech: float
    music: float
    #: 谷に山のある音（BGM）が残っている
    bed: bool
    #: 鳴っている所の平たさ（中央値）
    flatness: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _cues(profile: SoundProfile, start: int, end: int, silence_db: float) -> _Cues:
    """``[start, end)`` の 1 秒の手がかり 前後 1 秒も見る"""
    low = max(0, start - CONTEXT_TICKS)
    high = min(profile.ticks, end + CONTEXT_TICKS)
    energy = profile.rms[low:high] ** 2
    mean = float(np.mean(energy))
    # 平均の半分を下回る区切りの割合 話し声は息継ぎと音節の谷で 0.4 を超え、音楽は 0.2 未満が多い
    dips = float(np.mean(energy < 0.5 * mean)) if mean > 0 else 0.0
    loud = profile.rms_db()[low:high]
    voiced = loud > silence_db
    # 大きさの揺れ（鳴っている区切りの dB の標準偏差） 実際の話し声は 4〜7 dB、音楽は 1〜3 dB
    spread = float(np.std(loud[voiced])) if voiced.sum() > 1 else 0.0
    # 静かな 4 分の 1 の区切り（話し声なら息継ぎと音節の谷）に何が残っているか
    quiet = np.argsort(loud)[: max(1, len(loud) // 4)]
    floor = float(np.median(loud[quiet]))
    # 大きな所と谷の差（dB） 実際の話し声は 25〜50 dB、音楽は 2〜10 dB で、手がかりの中で
    # 一番よく分かれた（谷の割合は、間の多い曲で 0.3 を超え、話し声に寄る）
    gap = float(np.percentile(loud, 90)) - floor
    speech = (
        0.35 * _clamp((dips - 0.15) / 0.3)
        + 0.25 * _clamp((spread - 2.5) / 4.0)
        + 0.4 * _clamp((gap - _GAP_LOW) / (_GAP_HIGH - _GAP_LOW))
    )
    music = (
        0.35 * _clamp((0.3 - dips) / 0.25)
        + 0.25 * _clamp((6.0 - spread) / 4.0)
        + 0.4 * _clamp((_GAP_HIGH - gap) / (_GAP_HIGH - _GAP_LOW))
    )
    bed = (
        floor > silence_db
        and gap < _BED_GAP
        and float(np.median(profile.flatness[low:high][quiet])) < _TONAL_BED
    )
    heard = profile.rms_db()[start:end] > silence_db
    flat = profile.flatness[start:end][heard]
    return _Cues(speech, music, bed, float(np.median(flat)) if len(flat) else 1.0)


def _window(
    profile: SoundProfile,
    start: int,
    end: int,
    covered: float | None,
    silence_db: float,
) -> _Window:
    if float(profile.rms_db()[start:end].max()) < silence_db:
        return _Window(SILENCE, HIGH, 0.0, 0.0, covered)
    cues = _cues(profile, start, end, silence_db)
    speech, music = round(cues.speech, 2), round(cues.music, 2)
    if covered is not None and covered >= 0.5:
        # 字幕のある所は話している 下に山のある音が残るか音楽らしさも強ければ、BGM の上で話している
        if cues.bed or music >= 0.6:
            return _Window(SPEECH_AND_MUSIC, MEDIUM, speech, music, covered)
        return _Window(SPEECH, HIGH if speech >= 0.4 else MEDIUM, speech, music, covered)
    margin = speech - music
    if cues.flatness >= _NOISY and speech < 0.5:
        return _Window(NOISE, LOW, speech, music, covered)
    if speech >= 0.5 and margin >= 0.1:
        if cues.bed:
            return _Window(SPEECH_AND_MUSIC, LOW, speech, music, covered)
        return _Window(SPEECH, MEDIUM if margin >= 0.3 else LOW, speech, music, covered)
    if music >= 0.5 and margin <= -0.1:
        return _Window(MUSIC, MEDIUM if margin <= -0.3 else LOW, speech, music, covered)
    if cues.bed and speech >= _BED_SPEECH:
        # BGM やゲームの音の上で話すと、谷が音で埋まって話し声らしさが半ばで止まる
        # 見分けられないと返すより、敷いた音の上の話し声と返す方が実際の録画で合った
        return _Window(SPEECH_AND_MUSIC, LOW, speech, music, covered)
    return _Window(UNCLEAR, LOW, speech, music, covered)


def _segment(run: list[tuple[int, int, _Window]], label: str) -> SoundSegment:
    covered = [window.subtitles for _start, _end, window in run if window.subtitles is not None]
    rank = round(sum(_RANK[window.confidence] for _start, _end, window in run) / len(run))
    return SoundSegment(
        start=run[0][0],
        end=run[-1][1],
        label=label,
        confidence=next(name for name, value in _RANK.items() if value == rank),
        speech_score=round(sum(window.speech for _s, _e, window in run) / len(run), 2),
        music_score=round(sum(window.music for _s, _e, window in run) / len(run), 2),
        subtitles=round(sum(covered) / len(covered), 2) if covered else None,
    )


def classify(
    profile: SoundProfile,
    *,
    subtitles: np.ndarray | None = None,
    silence_db: float = -50.0,
) -> list[SoundSegment]:
    """1 秒ごとに見分けて、同じ見分けの続きを 1 区間にまとめる

    ``subtitles`` は区切りごとに字幕が重なるか（真偽の並び 長さは区切りの数） 渡さなければ
    音の形だけで見る 1 秒だけ違う見分けが同じ見分けに挟まれたら、前後に合わせる
    （話の途中の 1 秒の盛り上がりで区間を割らない 無音は挟まれても残す 短い間も編集の手がかり）
    """
    windows: list[tuple[int, int, _Window]] = []
    for start in range(0, profile.ticks, WINDOW_TICKS):
        end = min(profile.ticks, start + WINDOW_TICKS)
        covered = None if subtitles is None else float(np.mean(subtitles[start:end]))
        windows.append((start, end, _window(profile, start, end, covered, silence_db)))
    labels = [window.label for _start, _end, window in windows]
    for index in range(1, len(labels) - 1):
        before, now, after = labels[index - 1], labels[index], labels[index + 1]
        if before == after != now and SILENCE not in (now, before):
            labels[index] = before
    segments: list[SoundSegment] = []
    run: list[tuple[int, int, _Window]] = []
    current = ""
    for item, label in zip(windows, labels, strict=True):
        if run and label != current:
            segments.append(_segment(run, current))
            run = []
        run.append(item)
        current = label
    if run:
        segments.append(_segment(run, current))
    return segments
