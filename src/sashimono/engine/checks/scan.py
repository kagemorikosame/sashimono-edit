"""完成物の検査で、絵と音を 1 コマずつ・1 塊ずつ受け取って問題の区間を拾う（#337）

ここは Qt も GL も知らない 絵は輝度（0〜1 の 2 次元の配列）、音は ``(サンプル数, チャンネル数)``
の float で受け取る 今のタイムラインを描いた絵でも、書き出した動画を読んだ絵でも同じ物を通す

しきい値の既定（:class:`CheckLimits`）は、FFmpeg の検出の既定（``ffmpeg -h filter=blackdetect``
などで確かめた値）と、ラウドネスの公開されている基準に合わせてある 理由は各項目に書く
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from sashimono.engine.checks.profile import SoundProfile, SoundProfiler

__all__ = [
    "BLACK",
    "BLACK_RATIO",
    "CLIPPING",
    "FREEZE",
    "KINDS",
    "LENGTH",
    "LOUDNESS",
    "LOUDNESS_TARGETS",
    "PICTURE_KINDS",
    "SILENCE",
    "SOUND_KINDS",
    "TEXT_OVERFLOW",
    "AudioScan",
    "CheckLimits",
    "Run",
    "VideoScan",
]

BLACK = "black"
FREEZE = "freeze"
SILENCE = "silence"
CLIPPING = "clipping"
LOUDNESS = "loudness"
TEXT_OVERFLOW = "text_overflow"
LENGTH = "length"
#: 調べる種類 並びは返す一覧の説明の順
KINDS = (BLACK, FREEZE, SILENCE, CLIPPING, LOUDNESS, TEXT_OVERFLOW, LENGTH)
#: 絵を描かないと調べられない種類 頼まれなければ絵を描かずに済み、音だけなら何倍も速い
PICTURE_KINDS = frozenset({BLACK, FREEZE})
#: 音を混ぜないと調べられない種類
SOUND_KINDS = frozenset({SILENCE, CLIPPING, LOUDNESS})

#: 黒いコマとみなす、暗い画素の割合 FFmpeg の blackdetect の既定（pic_th 0.98）
#: 1 にすると、隅のロゴや字幕が 1 つ載っただけの黒い画面を見落とす
BLACK_RATIO = 0.98

#: 音量の基準の選択肢（LUFS, 名前） 設定画面と道具の説明が使う
#: - -14 YouTube・Spotify・Amazon Music などの配信が再生時にそろえる値
#: - -16 Apple Music（Sound Check）とポッドキャストで勧められる値（AES TD1008）
#: - -23 EBU R 128（欧州の放送 許される幅は ±0.5 LU）
#: - -24 ARIB TR-B32（日本のテレビ放送）と米国の ATSC A/85
LOUDNESS_TARGETS: tuple[tuple[int, str], ...] = (
    (-14, "配信（YouTube・Spotify など） -14 LUFS"),
    (-16, "Apple Music・ポッドキャスト -16 LUFS"),
    (-23, "欧州の放送（EBU R 128） -23 LUFS"),
    (-24, "日本の放送（ARIB TR-B32） -24 LUFS"),
)

#: 無音を測る区切り（秒） 1 サンプルずつ見ると、静かな部屋の雑音の山 1 つで無音が切れる
#: 50 ms は話し声の音節より短く、区切りの粗さで無音の長さが目立ってずれない
SILENCE_WINDOW = 0.05
#: 音割れの印をまとめる間（秒） これより近い印は 1 か所として返す 1 つの大きな音で
#: 何百回も頭打ちになるので、ばらばらに返すと一覧が埋まる
CLIP_MERGE_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class CheckLimits:
    """検査のしきい値 道具の引数から作る 既定の値の理由はそれぞれに書く"""

    #: 暗いとみなす輝度（0〜1） FFmpeg の blackdetect の既定（pix_th 0.10）
    #: 地の黒（16/255 = 0.063）と、映像の黒の浮き（圧縮のノイズ）は入り、暗い夜景は入らない
    black_level: float = 0.10
    #: これより短い黒は返さない 0 なら 1 コマでも返す つなぎ目に 1 コマ挟まった黒は
    #: 見ている人には「ちらつき」として見え、編集の失敗で一番よくある形なので既定で拾う
    #: （FFmpeg の既定は 2 秒だが、それは放送の素材の黒の区切りを探すための値）
    min_black_seconds: float = 0.0
    #: 止まった絵とみなす、輝度の差の平均（0〜1） FFmpeg の freezedetect の既定（-60 dB = 0.001）
    freeze_noise: float = 0.001
    #: これより短い止まりは返さない FFmpeg の freezedetect の既定（2 秒）
    #: 短くすると、話している人が一瞬止まっただけの所まで拾う
    min_freeze_seconds: float = 2.0
    #: 無音とみなす大きさ（dBFS） FFmpeg の silencedetect の既定（-60 dB = 0.001）
    #: 静かな部屋の雑音（-50〜-40 dBFS ほど）は無音に入らない 録り漏れや置き忘れだけを拾う
    silence_db: float = -60.0
    #: これより短い無音は返さない FFmpeg の silencedetect の既定（2 秒）
    #: 話の間（0.5 秒前後）や、無音カットで残した余白は拾わない
    min_silence_seconds: float = 2.0
    #: 音割れとみなす大きさ（dBFS） 16 bit の最大（32767/32768 = -0.0003 dBFS）を含む所
    #: 混ぜた音が 1.0 を超えた所も、書き出しで頭打ちになるので入る
    clip_db: float = -0.01
    #: 音割れとみなす、続けて頭打ちになったサンプルの数 Audacity の「クリッピングを検出」の
    #: 既定（3） 1 つだけ届いたのは、最大まで上げただけの正しい音のことが多い
    clip_samples: int = 3
    #: 音量の基準（LUFS） 既定は本人の設定（配信の -14 が既定 :data:`LOUDNESS_TARGETS`）
    loudness_target: float = -14.0
    #: 基準から外れたとみなす差（LU） EBU R 128 の幅（±0.5）は放送の納品の値で、配信では
    #: 細かすぎる 再生側がそろえる配信で聞き分けにくい 1 LU を既定にした
    loudness_tolerance: float = 1.0
    #: 文字を置いてよい範囲の余白（各辺の割合 0〜0.2） 0 なら画面の端 テレビの文字の安全域
    #: （EBU R 95 の各辺 5%）で見たいときは 0.05
    safe_margin: float = 0.0
    #: 長さの目安（秒） ``None`` なら長さを問題にしない（長さは要約に出す）
    expected_seconds: float | None = None
    #: 長さの目安から外れたとみなす差（秒）
    length_tolerance: float = 0.5
    #: 調べる種類（:data:`KINDS` の一部）
    kinds: frozenset[str] = field(default_factory=lambda: frozenset(KINDS))

    @property
    def needs_picture(self) -> bool:
        return bool(self.kinds & PICTURE_KINDS)

    @property
    def needs_sound(self) -> bool:
        return bool(self.kinds & SOUND_KINDS)


@dataclass(frozen=True, slots=True)
class Run:
    """見つけた区間 1 つ フレームは ``[start, end)``"""

    kind: str
    start: int
    end: int
    #: 程度（秒・大きさなど） 返事にそのまま載る
    values: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    #: まだ続いているかもしれない（解析の途中で、最後に見たコマまで続いている）
    ongoing: bool = False
    clip_ids: tuple[str, ...] = ()


class VideoScan:
    """黒いコマと止まった絵を拾う ``frame`` は渡す順に増えること"""

    def __init__(self, limits: CheckLimits, fps: float) -> None:
        self._limits = limits
        self._fps = max(fps, 1e-6)
        self._closed: list[Run] = []
        self._black_start: int | None = None
        #: 止まりの基準のコマ FFmpeg の freezedetect と同じく、隣のコマではなく止まり始めの
        #: コマと比べる 隣と比べると、ゆっくり明るくなる絵も止まりに数える
        self._reference: np.ndarray | None = None
        self._reference_frame = 0
        self._last: int | None = None

    def add(self, frame: int, luma: np.ndarray) -> None:
        black = float(np.mean(luma <= self._limits.black_level)) >= BLACK_RATIO
        if black:
            if self._black_start is None:
                self._black_start = frame
            # 黒は止まりとしては数えない 同じ所を 2 つの種類で返すと一覧が重なる
            self._close_freeze(frame)
            self._reference = None
        else:
            self._close_black(frame)
            self._follow_freeze(frame, luma)
        self._last = frame

    def _follow_freeze(self, frame: int, luma: np.ndarray) -> None:
        reference = self._reference
        if reference is not None and reference.shape == luma.shape:
            difference = float(np.mean(np.abs(luma - reference)))
            if difference <= self._limits.freeze_noise:
                return
            self._close_freeze(frame)
        self._reference = luma
        self._reference_frame = frame

    def _close_black(self, end: int) -> None:
        if self._black_start is None:
            return
        run = self._black(self._black_start, end, ongoing=False)
        if run is not None:
            self._closed.append(run)
        self._black_start = None

    def _close_freeze(self, end: int) -> None:
        if self._reference is None:
            return
        run = self._freeze(self._reference_frame, end, ongoing=False)
        if run is not None:
            self._closed.append(run)
        self._reference = None

    def _black(self, start: int, end: int, *, ongoing: bool) -> Run | None:
        frames = end - start
        if frames < max(1, math.ceil(self._limits.min_black_seconds * self._fps - 1e-9)):
            return None
        return Run(BLACK, start, end, {"seconds": round(frames / self._fps, 3)}, ongoing)

    def _freeze(self, start: int, end: int, *, ongoing: bool) -> Run | None:
        frames = end - start
        # 1 コマだけ（止まりの基準のコマ）は止まりではない 2 コマ以上続いて初めて止まり
        if frames < 2 or frames < self._limits.min_freeze_seconds * self._fps - 1e-9:
            return None
        return Run(FREEZE, start, end, {"seconds": round(frames / self._fps, 3)}, ongoing)

    def runs(self, *, final: bool) -> list[Run]:
        """拾った区間 ``final`` でなければ、続いている区間も途中までで足す"""
        found = list(self._closed)
        if self._last is None:
            return found
        end = self._last + 1
        if self._black_start is not None:
            run = self._black(self._black_start, end, ongoing=not final)
            if run is not None:
                found.append(run)
        if self._reference is not None:
            run = self._freeze(self._reference_frame, end, ongoing=not final)
            if run is not None:
                found.append(run)
        return found


@dataclass(slots=True)
class _ClipRegion:
    """音割れの印をまとめた所（サンプル）"""

    start: int
    end: int
    samples: int
    peak: float


class AudioScan:
    """無音・音割れ・ラウドネスを拾う 音は時刻の順に、途切れずに渡すこと

    ``origin`` は最初のサンプルが当たるフレーム 区間はフレームへ直して返す
    """

    def __init__(
        self,
        limits: CheckLimits,
        *,
        sample_rate: int,
        channels: int,
        fps: float,
        origin: int = 0,
    ) -> None:
        self._limits = limits
        self._rate = int(sample_rate)
        self._channels = int(channels)
        self._fps = max(fps, 1e-6)
        self._origin = origin
        self._position = 0
        self._peak = 0.0
        # 無音
        self._window = max(1, round(self._rate * SILENCE_WINDOW))
        self._partial = np.zeros((0, self._channels))
        self._windows = 0
        self._silence_floor = 10.0 ** (limits.silence_db / 20.0)
        self._silent_start: int | None = None
        self._silent_loudest = 0.0
        self._silences: list[tuple[int, int, float]] = []
        # 音割れ
        self._clip_level = 10.0 ** (limits.clip_db / 20.0)
        #: チャンネルごとの、前の塊の終わりから続いている頭打ち（始まりのサンプル, 長さ, 最大）
        self._carry: list[tuple[int, int, float] | None] = [None] * self._channels
        self._regions: list[_ClipRegion] = []
        # ラウドネスと音の推移（#322） 推移は頼まれた種類に関係なく覚える 音の道具
        # （analyze_audio・audio_graph）は前の検査の結果をそのまま読むので、種類で絞った検査でも
        # 推移が無いと混ぜ直すことになる K 特性は推移とラウドネスで 1 度だけ掛ける
        self._profiler = SoundProfiler(self._rate, self._channels)

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def seconds(self) -> float:
        return self._position / self._rate

    @property
    def samples(self) -> int:
        """これまでに渡したサンプルの数"""
        return self._position

    @property
    def rate(self) -> int:
        return self._rate

    @property
    def channels(self) -> int:
        return self._channels

    def add(self, samples: np.ndarray) -> None:
        block = np.asarray(samples, dtype=np.float64).reshape(-1, self._channels)
        if len(block) == 0:
            return
        magnitude = np.abs(block)
        self._peak = max(self._peak, float(magnitude.max()))
        if SILENCE in self._limits.kinds:
            self._scan_silence(block)
        if CLIPPING in self._limits.kinds:
            self._scan_clipping(magnitude)
        self._profiler.add(block)
        self._position += len(block)

    # --- 無音 ---

    def _scan_silence(self, block: np.ndarray) -> None:
        joined = np.concatenate([self._partial, block])
        whole = len(joined) // self._window * self._window
        self._partial = joined[whole:]
        if not whole:
            return
        windows = joined[:whole].reshape(-1, self._window, self._channels)
        # どのチャンネルも静かなときだけ無音 片方だけ鳴っている音（左右に振った声）を消さない
        levels = np.sqrt(np.mean(windows * windows, axis=1)).max(axis=1)
        for level in levels.tolist():
            start = self._windows * self._window
            if level < self._silence_floor:
                if self._silent_start is None:
                    self._silent_start = start
                    self._silent_loudest = 0.0
                self._silent_loudest = max(self._silent_loudest, level)
            elif self._silent_start is not None:
                self._silences.append((self._silent_start, start, self._silent_loudest))
                self._silent_start = None
            self._windows += 1

    # --- 音割れ ---

    def _scan_clipping(self, magnitude: np.ndarray) -> None:
        count = len(magnitude)
        for channel in range(self._channels):
            column = magnitude[:, channel]
            over = column >= self._clip_level
            if not over.any():
                self._close_carry(channel)
                continue
            edges = np.diff(np.concatenate([[0], over.astype(np.int8), [0]]))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            # reduceat は次の始まりの手前までを見る 頭打ちでない所を 0 にしておけば、
            # その間の最大は頭打ちの区間の最大になる
            peaks = np.maximum.reduceat(np.where(over, column, 0.0), starts)
            if starts[0] != 0:
                # 塊の頭が頭打ちでなければ、前の塊から続いていた分はそこで終わっている
                self._close_carry(channel)
            for start, end, peak in zip(
                starts.tolist(), ends.tolist(), peaks.tolist(), strict=True
            ):
                begin, length, highest = self._position + start, end - start, float(peak)
                carry = self._carry[channel] if start == 0 else None
                if carry is not None:
                    begin, length, highest = carry[0], carry[1] + length, max(carry[2], highest)
                    self._carry[channel] = None
                if end == count:
                    # 次の塊へ続くかもしれない 長さが決まるまで記録しない
                    self._carry[channel] = (begin, length, highest)
                else:
                    self._record_clip(begin, length, highest)

    def _close_carry(self, channel: int) -> None:
        carry = self._carry[channel]
        if carry is not None:
            self._carry[channel] = None
            self._record_clip(*carry)

    def _record_clip(self, begin: int, length: int, peak: float) -> None:
        if length < self._limits.clip_samples:
            return
        end = begin + length
        merge = round(self._rate * CLIP_MERGE_SECONDS)
        # チャンネルごとに見るので、同じ塊の中では時刻が前後して届く 近くの数か所だけ見れば足りる
        for region in reversed(self._regions[-8:]):
            if begin <= region.end + merge and end >= region.start - merge:
                region.start = min(region.start, begin)
                region.end = max(region.end, end)
                region.samples += length
                region.peak = max(region.peak, peak)
                return
        self._regions.append(_ClipRegion(begin, end, length, peak))
        if len(self._regions) > 1 and self._regions[-2].start > begin:
            self._regions.sort(key=lambda region: region.start)

    # --- 返す ---

    def _frame(self, sample: int, *, up: bool) -> int:
        exact = sample * self._fps / self._rate
        return self._origin + (math.ceil(exact - 1e-9) if up else math.floor(exact + 1e-9))

    def loudness(self) -> float | None:
        if LOUDNESS not in self._limits.kinds:
            return None
        return self._profiler.meter.integrated()

    def profile(self) -> SoundProfile:
        """ここまでの音の推移の写し（#322）"""
        return self._profiler.snapshot()

    def finish(self) -> None:
        """終わりまで渡した 最後のサンプルまで続いている頭打ちを閉じる"""
        for channel in range(self._channels):
            self._close_carry(channel)

    def runs(self, *, final: bool) -> list[Run]:
        found: list[Run] = []
        silences = [(start, end, loudest, False) for start, end, loudest in self._silences]
        if self._silent_start is not None:
            # 終わりまで無音なら、端数の区切りも足して最後のサンプルまでを無音にする
            end = self._position if final else self._windows * self._window
            silences.append((self._silent_start, end, self._silent_loudest, not final))
        minimum = self._limits.min_silence_seconds * self._rate
        for start, end, loudest, ongoing in silences:
            if end <= start or end - start < minimum - 1e-9:
                continue
            level = 20.0 * math.log10(loudest) if loudest > 0 else -120.0
            found.append(
                Run(
                    SILENCE,
                    self._frame(start, up=True),
                    self._frame(end, up=False),
                    {
                        "seconds": round((end - start) / self._rate, 3),
                        "loudest_dbfs": round(level, 1),
                    },
                    ongoing,
                )
            )
        regions = list(self._regions)
        if not final:
            # まだ閉じていない頭打ちも、長さが足りていれば途中で見せる
            for carry in self._carry:
                if carry is not None and carry[1] >= self._limits.clip_samples:
                    regions.append(_ClipRegion(carry[0], carry[0] + carry[1], carry[1], carry[2]))
        for region in regions:
            found.append(
                Run(
                    CLIPPING,
                    self._frame(region.start, up=False),
                    max(self._frame(region.end, up=True), self._frame(region.start, up=False) + 1),
                    {
                        "clipped_samples": region.samples,
                        "peak_dbfs": round(20.0 * math.log10(max(region.peak, 1e-12)), 2),
                        # 1 を超えた音は書き出しで頭打ちになる 素材の音が元から割れているのか、
                        # 混ぜて大きくしすぎたのかの見分けに使う
                        "over_full_scale": region.peak > 1.0,
                    },
                )
            )
        return found
