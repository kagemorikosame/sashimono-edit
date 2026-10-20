"""盛り上がりの区間を、笑い声・音量の山・動きの大きい所から見つける（#325）

素材を読み直さない 使うのは、取り込んだときに作って覚えてある物だけ

- 波形のピーク（:mod:`sashimono.engine.audio.waveform` 約 5 ms ごとの最小と最大）
  音量の山と笑い声はここから見る 無音カット（:mod:`.audio.silence`）と同じ物
- 場面の解析で並べたフレーム間の差（:class:`~sashimono.engine.scenes.SceneSignal`）
  動きの大きさは、切り替わりのフレームを除いた画素の差の平均で見る
- 字幕（文字起こし） 笑い声の印（「（笑）」「ははは」「www」など）と、ふつうの話し声の所

外の機械学習のモデルは使わない（#298 の「決めること」 配る zip の大きさと使用許諾に響く）
笑い声を確かに聞き分けるには音の形を学んだモデルが要るが、ここでは「短い間に、同じ間隔で、
深く上下する大きな音」を笑い声の候補と見る 限りは :data:`LAUGHTER_LIMITS` に書き、道具の
説明にも出す 候補は AI が ``preview_range`` や字幕で確かめる前提で、見落としより拾いすぎに倒す

区間の点数は 3 つの手がかり（どれも 0〜1）を「どれか 1 つでも強ければ高い」形で合わせる
（``1 - (1 - a)(1 - b)(1 - c)``） 足して割る形にすると、動かない画面の大笑いのように
1 つだけ強い所が半分以下の点になり、拾えない
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from sashimono.core.model import Transcript
from sashimono.engine.audio.waveform import Waveform
from sashimono.engine.scenes import DEFAULT_THRESHOLD as SCENE_THRESHOLD
from sashimono.engine.scenes import SceneSignal

__all__ = [
    "DEFAULT_MERGE_GAP",
    "DEFAULT_MIN_SECONDS",
    "DEFAULT_THRESHOLD",
    "HOP_SECONDS",
    "LAUGHTER",
    "LAUGHTER_LIMITS",
    "LOUD",
    "MOTION",
    "SIGNALS",
    "Highlight",
    "HighlightCurves",
    "highlight_curves",
    "is_laughter_text",
]

LAUGHTER = "laughter"
LOUD = "loud"
MOTION = "motion"
#: 手がかりの名前（並びは説明の順）
SIGNALS = (LAUGHTER, LOUD, MOTION)

#: 点数を並べる刻み（秒） 笑い声の 1 息（0.2 秒ほど）より長く、短い盛り上がり（1 秒ほど）を
#: 4 つに分けられる長さ 細かくしても区間の端が細かくなるだけで、AI が見る位置は変わらない
HOP_SECONDS = 0.25
#: 区間とみなす点数の既定 1 つの手がかりが半分以上か、2 つが 3 割ずつ
DEFAULT_THRESHOLD = 0.5
#: これより短い区間は返さない（秒） 一瞬の物音や 1 コマのフラッシュを盛り上がりと見ない
DEFAULT_MIN_SECONDS = 0.5
#: これより短い切れ目でつながる区間は 1 つにまとめる（秒） 笑いの息継ぎで区間が割れないため
DEFAULT_MERGE_GAP = 1.0

#: 笑い声の見分けの限り 道具の説明と開発の文書に同じ物を出す
LAUGHTER_LIMITS = (
    "笑い声は音の大きさの上下のリズム（1 秒に 3〜8 回、深く、同じ間隔）から見る候補で、"
    "声の質までは聞き分けない 話し声（特に 1 音ずつ区切る話し方）・拍手・"
    "一定のリズムの音楽や効果音も拾うことがあり、BGM に埋もれた小さな笑いや"
    "息だけの笑いは拾わない 話し声や BGM と重なった実際の笑いは音だけからはほとんど拾えず"
    "（実際の素材 5 本で 0 件）、字幕の笑いの印（（笑）・ははは・www など）が主な手がかりになる"
    " 笑いを探すときは字幕を起こしてから呼ぶ"
)

# --- 音 ---

#: 波形のピークを束ねる長さ（秒） 笑いの 1 息の山と谷を分けられる細かさ 細かいままだと
#: 声の高さ（1 周期 5 ms 前後）の揺れまで上下に数える
ENVELOPE_SECONDS = 0.01
#: これより小さい音は無いものとして扱う（dBFS） 部屋の雑音で「普段の大きさ」を下げない
SILENT_DB = -50.0
#: 音量を均す長さ（秒） 一瞬の打音（手を叩く・物を置く）を山と見ない
LOUD_SMOOTH_SECONDS = 1.0
#: 普段（均した大きさの中央値）より何 dB 大きければ山の始まりか・山の頂か（点 0.5 は +8 dB）
#: 実際の素材 5 本（雑談・茶番・実況・大人数 各 15 分まで）で、K 特性の大きさが普段より 9 dB
#: 大きい所を正解として振った 始まりを 3 にすると当たり 89%・外れ 52%、4 で 74%・42%、
#: 5 で 61%・39% 当たりを大きく落とさずに外れを減らせる 4 にした
LOUD_RISE_START_DB = 4.0
LOUD_RISE_FULL_DB = 12.0
#: 笑い声のリズムを見る窓（秒） 笑いの 1 息（0.15〜0.3 秒）が 5 つほど入る長さ
LAUGH_WINDOW_SECONDS = 1.5
#: 笑いの 1 息の間隔として見る範囲（秒） 1 秒に 3.3〜8 回
LAUGH_PERIOD_SECONDS = (0.125, 0.3)
#: 同じ間隔で上下する度合い（自己相関 0〜1）の、笑いらしさの始まりと頂
#: 話し声は音節の長さがばらつき、0.3 を越えることが少ない 作った笑い（5 回 / 秒）は 0.7 を越える
LAUGH_PERIODIC = (0.3, 0.65)
#: 上下の深さ（窓の中の大きい 1 割と小さい 1 割の差 dB）の、笑いらしさの始まりと頂
#: 息を吐く「は」の合間は声が切れて 15 dB 以上下がる 音楽や持続音は 6 dB ほどに収まる
LAUGH_DEPTH_DB = (8.0, 16.0)
#: 普段の大きさより何 dB 小さい所まで笑いと見るか 笑いは話し声より大きいことが多いが、
#: 遠くの笑い（聞き手の笑い）も拾うため、少し小さい所から数える
LAUGH_QUIET_DB = 6.0
#: これより長く続くリズムは笑いと見ない（秒） 笑いの 1 回はたいてい数秒で、一定の拍の音楽や
#: 効果音は何十秒も続く
MAX_LAUGH_SECONDS = 8.0
#: これより短く途切れる笑いの点は半分に下げる（秒） 3 息に満たない並びは、話し声の音節が
#: たまたま同じ長さで続いた所と見分けにくい
MIN_LAUGH_SECONDS = 1.0
#: 字幕に笑いの印がある所の、笑い声の点数 印は確かだが、字幕 1 枚のどこで笑ったかは分からない
SUBTITLE_LAUGH = 0.9
#: ふつうの話し声の字幕が重なる所で、笑い声の点数に掛ける値 話し声の音節のリズムを笑いと
#: 取り違えやすいので下げる 0 にしないのは、話しながら笑う所もあるため
SPEECH_DAMPING = 0.5

# --- 動き ---

#: 動きの大きさをそろえるフレームの速さ（毎秒） フレーム間の差は間隔が短いほど小さく出るので、
#: 60 fps の素材の差を 2 倍して 30 fps の素材とそろえる
MOTION_REFERENCE_FPS = 30.0
#: 動きの大きいとみなし始める差（輝度の差の平均 0〜1） 話している人を据え置きで撮った所は
#: 0.01 を下回り（実際の雑談の素材で中央値 0.001）、速いパン・走る人は 0.03 を越える
#: 切り替わり（0.1〜0.3）は数えない
MOTION_FLOOR = 0.01
#: 素材の普段（中央値）の何倍から動きが大きいとみなすか ずっと動いている素材（スポーツ・
#: ゲームの録画）で全部を盛り上がりと見ないため
MOTION_RELATIVE = 1.5
#: 始まりから頂（点数 1）までの差の幅 実際の素材 5 本で、コーデックの動きベクトルが普段の
#: 3 倍を越える所を正解として振った 0.04 では当たりが 18% しかなく（実況の動く所の差は
#: 0.03 前後）、0.02 で当たり 71%・外れ 66%、0.01 では外れがほぼ同じまま見つける数が増えた
MOTION_SPAN = 0.02
#: 動きを均す長さ（秒） 1 コマの揺れを動きと見ない
MOTION_SMOOTH_SECONDS = 1.0

#: 小さすぎて対数を取れない値の下限
_TINY = 1e-10


@dataclass(frozen=True, slots=True)
class Highlight:
    """盛り上がりの区間 1 つ 時刻は素材の原点から数えた秒

    手がかりの点数はどれも区間の中の最大（0〜1） 調べなかった手がかりは ``None``
    """

    start: float
    end: float
    #: 合わせた点数がいちばん高い所（刻みの真ん中）
    peak: float
    score: float
    laughter: float | None
    loud: float | None
    motion: float | None
    #: 字幕の笑いの印から笑い声と見た所を含むか
    laughter_from_subtitles: bool = False

    @property
    def reasons(self) -> tuple[str, ...]:
        """半分以上の点が付いた手がかり 強い順"""
        found = [
            (value, name)
            for name, value in ((LAUGHTER, self.laughter), (LOUD, self.loud), (MOTION, self.motion))
            if value is not None and value >= 0.5
        ]
        return tuple(name for _, name in sorted(found, key=lambda item: -item[0]))


@dataclass(frozen=True, slots=True)
class HighlightCurves:
    """:data:`HOP_SECONDS` ごとの手がかりの点数 調べなかった手がかりは ``None``

    点数の並びは覚えておける大きさ（1 時間で 1 万 4 千 4 百個）なので、しきい値を
    変えて何度聞いても作り直すのは区間の所だけで済む
    """

    duration: float
    laughter: np.ndarray | None
    loud: np.ndarray | None
    motion: np.ndarray | None
    #: 字幕の笑いの印が重なる刻み ``laughter`` を調べたときだけ
    subtitle_laugh: np.ndarray | None = None

    @property
    def count(self) -> int:
        return _hops(self.duration)

    def combined(self) -> np.ndarray:
        """合わせた点数 ``1 - Π(1 - 手がかり)``"""
        rest = np.ones(self.count, np.float64)
        for curve in (self.laughter, self.loud, self.motion):
            if curve is not None:
                rest *= 1.0 - np.clip(curve, 0.0, 1.0)
        result: np.ndarray = (1.0 - rest).astype(np.float32)
        return result

    def highlights(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_seconds: float = DEFAULT_MIN_SECONDS,
        merge_gap: float = DEFAULT_MERGE_GAP,
    ) -> tuple[Highlight, ...]:
        """しきい値を越えた区間 時刻の順"""
        score = self.combined()
        found = []
        for first, last in _joined_runs(score >= threshold, round(merge_gap / HOP_SECONDS)):
            start = first * HOP_SECONDS
            end = min(self.duration, (last + 1) * HOP_SECONDS)
            if end - start < min_seconds:
                continue
            span = slice(first, last + 1)
            best = first + int(np.argmax(score[span]))
            found.append(
                Highlight(
                    start=round(start, 3),
                    end=round(end, 3),
                    peak=round(min(self.duration, (best + 0.5) * HOP_SECONDS), 3),
                    score=float(score[best]),
                    laughter=_peak(self.laughter, span),
                    loud=_peak(self.loud, span),
                    motion=_peak(self.motion, span),
                    laughter_from_subtitles=bool(
                        self.subtitle_laugh is not None and self.subtitle_laugh[span].any()
                    ),
                )
            )
        return tuple(found)


def highlight_curves(
    duration: float,
    *,
    waveform: Waveform | None = None,
    scenes: SceneSignal | None = None,
    transcript: Transcript | None = None,
    signals: Iterable[str] = SIGNALS,
) -> HighlightCurves:
    """素材の手がかりの点数を並べる

    ``waveform`` が無ければ音の手がかり（笑い声・音量）を、``scenes`` が無ければ動きを
    調べない（``None`` で返す） 頼まれていない手がかり（``signals`` に無い物）も調べない
    ``scenes`` が途中までの写しなら、読んだ所より後ろの動きは 0 のまま
    """
    wanted = set(signals)
    count = _hops(duration)
    laughter = loud = motion = subtitle_laugh = None
    if waveform is not None and wanted & {LAUGHTER, LOUD}:
        envelope, step = _envelope(waveform)
        power = _hop_power(envelope, step, count)
        typical = _typical_db(power)
        if LOUD in wanted:
            loud = _loud(power, typical)
        if LAUGHTER in wanted:
            laughter = _laughter(envelope, step, count, typical)
    if LAUGHTER in wanted and transcript is not None and transcript.segments:
        if laughter is None:
            laughter = np.zeros(count, np.float32)
        laughter, subtitle_laugh = _with_subtitles(laughter, transcript, count)
    if scenes is not None and MOTION in wanted:
        motion = _motion(scenes, count)
    return HighlightCurves(
        duration=duration,
        laughter=laughter,
        loud=loud,
        motion=motion,
        subtitle_laugh=subtitle_laugh,
    )


# --- 字幕の笑いの印 ---

#: 笑いの書き添え 「（笑）」「(笑い)」「[laughter]」など 字幕起こしは笑い声をこう書くことがある
_LAUGH_NOTE = re.compile(
    r"[（(\[［【〔]\s*(?:笑|笑い|爆笑|苦笑|laugh(?:s|ing|ter)?)\s*[）)\]］】〕]", re.IGNORECASE
)
#: 同じ笑いの音が 3 つ以上続く所（「ははは」「ふふふ」「www」「hahaha」） 2 つ（「はは」）は
#: 「母」のようなふつうの言葉と見分けが付かない 英字の hi は挨拶（「Hi, hi!」）と
#: 見分けが付かないので数えない
_LAUGH_RUN = re.compile(
    r"(?:[はハ][っッ]?){3,}|(?:[ひヒ][っッ]?){3,}|(?:[ふフ][っッ]?){3,}|(?:[へヘ][っッ]?){3,}"
    r"|[wｗWＷ]{3,}|草{2,}|(?:ha){3,}|(?:he){3,}|\blol\b|\blmao\b",
    re.IGNORECASE,
)
#: 字幕の文から外す飾り 「あははは！」の「！」で笑いの割合を下げない 句点は文の句点の
#: 見張り（tools/punctuation.py）が文章と取り違えないよう、番号で足す
_PUNCTUATION = re.compile("[\\s、" + chr(0x3002) + "，．,.!?！？…・ー〜~「」『』]+")
#: 笑いの音 文がこれだけでできているかを見る 「あ」「っ」は「あはは」「ふふっ」の飾り
_LAUGH_SOUND = re.compile(r"[はハひヒふフへヘあアっッ]|[ｗＷ]|草|笑")
#: 笑いの核の音 「あ」「っ」だけの文（「あああ」「あっ」）は驚きや嘆きで、笑いではない
#: 「草」はそれ 1 字で笑いを表すので 2 つ分に数える
_LAUGH_CORE = re.compile(r"[はハひヒふフへヘ]|[ｗＷ]|草")
#: かなの笑いの音だけの文 2 字（「はは」）は「母」のような言葉と見分けが付かない
_KANA_LAUGH = re.compile(r"[はハひヒふフへヘあアっッ]+")
#: 英字の文で笑いと読める物 ha・he の繰り返しと w の並びだけに絞る hi や英単語の
#: 綴りを笑いの音として数えると「Hi, hi!」の挨拶や「hew」のような言葉を笑いと見る
#: lol・lmao は _LAUGH_RUN が拾う
_LATIN_LAUGH = re.compile(r"(?:ha|he){2,}|[wW]{2,}", re.IGNORECASE)
_LATIN = re.compile(r"[A-Za-z]")


def is_laughter_text(text: str) -> bool:
    """字幕の文が笑いを書いた物か

    書き添え（「（笑）」）・同じ笑いの音が 3 つ以上続く所・笑いの音だけでできていて核の音
    （は・ひ・ふ・へ・w・草）が 2 つ以上ある文（「あはは」「ｗｗ」「草」）・英字なら ha・he の
    繰り返しか w だけの文（「haha」「hehe」「ww」）を笑いと見る
    「笑顔」のように字の中に「笑」があるだけの文や、「ああ、はい」のような相づちは見ない
    """
    if _LAUGH_NOTE.search(text) or _LAUGH_RUN.search(text):
        return True
    body = _PUNCTUATION.sub("", text)
    if _LATIN.search(body):
        # 英字は綴りの組み合わせが多く、笑いの音だけで割り切ると言葉を拾うので、形で見る
        return _LATIN_LAUGH.fullmatch(body) is not None
    # 笑いの音以外が 1 字でも混ざれば笑いと見ない 割合で見ると「ああ、はい」「あっ、はい」の
    # ような相づちが 4 字のうち 3 字で笑いになり、ふつうの話し声を盛り上がりとして返した
    if not body or _LAUGH_SOUND.sub("", body):
        return False
    core = sum(2 if found == "草" else 1 for found in _LAUGH_CORE.findall(body))
    if core < 2:
        return False
    # 「ｗｗ」「草」は 2 字以下でも笑いと決まるが、かなの 2 字は言葉と取り違える
    return len(body) >= 3 or _KANA_LAUGH.fullmatch(body) is None


def _with_subtitles(
    laughter: np.ndarray, transcript: Transcript, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """字幕の笑いの印で点を上げ、ふつうの話し声の所で点を下げる"""
    result = laughter.astype(np.float32, copy=True)
    marked = np.zeros(count, bool)
    speech = np.zeros(count, bool)
    for segment in transcript.segments:
        first = max(0, math.floor(float(segment.start) / HOP_SECONDS))
        last = min(count, max(first + 1, math.ceil(float(segment.end) / HOP_SECONDS)))
        if first >= count:
            continue
        if is_laughter_text(segment.text):
            marked[first:last] = True
        elif segment.text.strip():
            speech[first:last] = True
    # 話し声で下げてから印で上げる 「（笑）」付きの話し声の字幕は、笑いの印を勝たせる
    result[speech & ~marked] *= SPEECH_DAMPING
    result[marked] = np.maximum(result[marked], SUBTITLE_LAUGH)
    return result, marked


# --- 音 ---


def _envelope(waveform: Waveform) -> tuple[np.ndarray, float]:
    """音の大きさ（チャンネルのうち大きい方の振幅 0〜1）を :data:`ENVELOPE_SECONDS` ごとに

    いちばん細かい段階のピークを束ねる 返すのは並びと 1 つの長さ（秒）
    """
    level = waveform.levels[0]
    if level.count == 0:
        return np.zeros(0, np.float32), ENVELOPE_SECONDS
    amplitude = np.abs(level.peaks).max(axis=(1, 2))
    step = level.samples_per_peak / waveform.sample_rate
    group = max(1, round(ENVELOPE_SECONDS / step))
    whole = -(-amplitude.shape[0] // group) * group
    if whole != amplitude.shape[0]:
        amplitude = np.concatenate([amplitude, np.zeros(whole - amplitude.shape[0], np.float32)])
    envelope: np.ndarray = amplitude.reshape(-1, group).max(axis=1).astype(np.float32)
    return envelope, step * group


def _hop_power(envelope: np.ndarray, step: float, count: int) -> np.ndarray:
    """刻みごとの音の大きさ（振幅の二乗の平均）"""
    power = np.zeros(count, np.float64)
    if envelope.shape[0] == 0 or count == 0:
        return power
    index = np.minimum((np.arange(envelope.shape[0]) * step / HOP_SECONDS).astype(np.int64), count)
    inside = index < count
    sums = np.bincount(
        index[inside], weights=envelope[inside].astype(np.float64) ** 2, minlength=count
    )
    numbers = np.bincount(index[inside], minlength=count)
    np.divide(sums, numbers, out=power, where=numbers > 0)
    return power


def _db(power: np.ndarray) -> np.ndarray:
    return np.asarray(10.0 * np.log10(np.maximum(power, _TINY)))


def _typical_db(power: np.ndarray) -> float | None:
    """鳴っている所の大きさの中央値（dB） 全部が無音なら ``None``"""
    level = _db(power)
    sounding = level[level > SILENT_DB]
    if sounding.shape[0] == 0:
        return None
    return float(np.median(sounding))


def _loud(power: np.ndarray, typical: float | None) -> np.ndarray:
    """音量の山 均した大きさが普段よりどれだけ大きいか"""
    if typical is None or power.shape[0] == 0:
        return np.zeros(power.shape[0], np.float32)
    smoothed = _db(_moving_mean(power, round(LOUD_SMOOTH_SECONDS / HOP_SECONDS)))
    # 普段の大きさも均した後の並びで測る 刻みのままの中央値は話の合間の小さい刻みで
    # 下がり、均した山との差が大きく出る（実際の素材で山の 8 割が外れだった）
    sounding = smoothed[smoothed > SILENT_DB]
    if sounding.shape[0]:
        typical = float(np.median(sounding))
    rise = smoothed - typical
    score = _ramp(rise, LOUD_RISE_START_DB, LOUD_RISE_FULL_DB)
    score[smoothed <= SILENT_DB] = 0.0
    return score


def _laughter(envelope: np.ndarray, step: float, count: int, typical: float | None) -> np.ndarray:
    """笑い声の候補 短い間に同じ間隔で深く上下する、普段くらいかそれより大きい音"""
    score = np.zeros(count, np.float32)
    width = round(LAUGH_WINDOW_SECONDS / step)
    if typical is None or count == 0 or envelope.shape[0] < width:
        return score
    centers = np.round((np.arange(count) + 0.5) * HOP_SECONDS / step).astype(np.int64)
    padded = np.pad(envelope.astype(np.float64), (width // 2, width))
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    rows = windows[np.minimum(centers, windows.shape[0] - 1)]

    level = _db(np.mean(rows * rows, axis=1))
    depth = np.percentile(20.0 * np.log10(np.maximum(rows, 1e-5)), [90, 10], axis=1)
    centered = rows - rows.mean(axis=1, keepdims=True)
    variance = np.mean(centered * centered, axis=1)
    low, high = (max(1, round(seconds / step)) for seconds in LAUGH_PERIOD_SECONDS)

    def correlation(lag: int) -> np.ndarray:
        # 重なる所の長さで割る（偏りの無い形） 全体の長さで割ると、長い間隔ほど小さく出て
        # 1 秒に 3 回の笑いが 8 回の笑いより低い点になる
        product = np.mean(centered[:, :-lag] * centered[:, lag:], axis=1)
        return np.divide(product, variance, out=np.zeros(count), where=variance > _TINY)

    # 間隔 L と 2L の両方で似ているか 1 つの間隔だけだと、話し声の音節がたまたま 2 つ同じ
    # 長さで並んだ所も拾う（作った話し声で 0.87 になった） 笑いは同じ間隔で 3 息以上続く
    best = np.zeros(count, np.float64)
    for lag in range(low, high + 1):
        if 2 * lag >= width:
            break
        best = np.maximum(best, np.minimum(correlation(lag), correlation(2 * lag)))

    rhythm = _ramp(best, *LAUGH_PERIODIC) * _ramp(depth[0] - depth[1], *LAUGH_DEPTH_DB)
    # 普段より LAUGH_QUIET_DB 小さい所から数え始め、普段の 2 dB 下で満点にする
    score = rhythm * _ramp(level - (typical - LAUGH_QUIET_DB), 0.0, LAUGH_QUIET_DB - 2.0)
    score[level <= SILENT_DB] = 0.0
    # 長く続くリズムは笑いでなく、一定の拍の音楽や効果音と見て外す 大きさを見ない
    # リズムの並びで数える 大きさで見ると、拍の強弱で 0.5 の上下を行き来して短い所に割れ、
    # どれも長さの上限に届かない（作った 4 拍の音楽 30 秒が丸ごと残った）
    longest = round(MAX_LAUGH_SECONDS / HOP_SECONDS)
    joined = round(DEFAULT_MERGE_GAP / HOP_SECONDS)
    for first, last in _joined_runs(rhythm >= 0.25, joined):
        if last - first + 1 > longest:
            score[first : last + 1] = 0.0
    # 1 秒に満たない笑いは、話し声の音節の並びと見分けにくいので半分に下げる 消さないのは、
    # 音量の山や動きと重なった所では、短い笑いも盛り上がりの手がかりになるため
    shortest = round(MIN_LAUGH_SECONDS / HOP_SECONDS)
    for first, last in _runs(score >= 0.5):
        if last - first + 1 < shortest:
            score[first : last + 1] *= 0.5
    return np.asarray(score, np.float32)


# --- 動き ---


def _motion(scenes: SceneSignal, count: int) -> np.ndarray:
    """動きの大きさ 切り替わりのフレームを除いた、フレーム間の差の平均"""
    score = np.zeros(count, np.float32)
    if len(scenes) == 0 or count == 0:
        return score
    times = scenes.times
    steps = np.diff(times)
    step = float(np.median(steps)) if steps.shape[0] else 1.0 / MOTION_REFERENCE_FPS
    # 間隔の短い素材ほど差が小さく出るので、30 fps の差にそろえる
    scale = 1.0 / (MOTION_REFERENCE_FPS * step) if step > 0 else 1.0
    pixel = scenes.pixel.astype(np.float64) * scale
    # 切り替わりは絵が丸ごと変わるだけで、動きではない（場面の検出と同じしきい値で外す）
    keep = scenes.scores() < SCENE_THRESHOLD
    index = np.floor(times / HOP_SECONDS).astype(np.int64)
    inside = keep & (index >= 0) & (index < count)
    sums = np.bincount(index[inside], weights=pixel[inside], minlength=count)
    numbers = np.bincount(index[inside], minlength=count)
    known = numbers > 0
    if not known.any():
        return score
    mean = np.zeros(count, np.float64)
    np.divide(sums, numbers, out=mean, where=known)
    # 読んでいない刻みを 0 として均すと、読んだ所の端の動きが薄まる 読んだ刻みだけで均す
    window = round(MOTION_SMOOTH_SECONDS / HOP_SECONDS)
    weight = _moving_mean(known.astype(np.float64), window)
    smoothed = np.divide(_moving_mean(mean, window), weight, out=np.zeros(count), where=weight > 0)
    start = max(MOTION_FLOOR, MOTION_RELATIVE * float(np.median(mean[known])))
    score = _ramp(smoothed, start, start + MOTION_SPAN)
    score[~known] = 0.0
    return score


# --- 共通 ---


def _hops(duration: float) -> int:
    return max(0, math.ceil(duration / HOP_SECONDS - 1e-9))


def _ramp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """``low`` で 0、``high`` で 1 になる直線 外は端で止める"""
    result: np.ndarray = np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return result


def _moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    """前後を含めて ``width`` 個の平均 端は届く所だけで割る"""
    if width <= 1 or values.shape[0] == 0:
        same: np.ndarray = values.astype(np.float64)
        return same
    kernel = np.ones(width, np.float64)
    sums = np.convolve(values.astype(np.float64), kernel, mode="same")
    numbers = np.convolve(np.ones(values.shape[0]), kernel, mode="same")
    result: np.ndarray = sums / numbers
    return result


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """真が続く所の（最初, 最後）の番号"""
    if mask.shape[0] == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(first), int(end) - 1) for first, end in zip(edges[0::2], edges[1::2], strict=True)]


def _joined_runs(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """真が続く所 ``gap`` 個以下の偽で隔てた所は 1 つにつなぐ"""
    joined: list[tuple[int, int]] = []
    for first, last in _runs(mask):
        if joined and first - joined[-1][1] - 1 <= gap:
            joined[-1] = (joined[-1][0], last)
        else:
            joined.append((first, last))
    return joined


def _peak(curve: np.ndarray | None, span: slice) -> float | None:
    if curve is None:
        return None
    return round(float(np.max(curve[span])), 3)
