"""タイムラインの音声をミックスする

映像と違い、音声は「今のフレーム」だけでは足りない 再生は連続したサンプル列を
要求するので、フレーム境界をまたぐ範囲をまとめて返せる形にしてある
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator

import numpy as np

from sashimono.core.model import (
    Clip,
    Effect,
    MediaId,
    ParamValue,
    Project,
    Timeline,
    Track,
    TrackKind,
    heard_stream,
)
from sashimono.core.timebase import FrameRate
from sashimono.effects.audio import MAX_HISTORY_SECONDS, AudioContext
from sashimono.effects.definition import EffectDefinition, registry
from sashimono.effects.spec import TrackSpec
from sashimono.engine.decode import AudioDecoder, ProbeError

__all__ = ["AudioMixer"]

#: 同時に開いておくデコーダの上限
MAX_OPEN_DECODERS = 8

#: シーンの入れ子の深さの上限（映像のレンダラと同じ値）
MAX_SCENE_DEPTH = 8


class AudioMixer:
    """プロジェクトの音声を、指定したサンプル範囲について合成する

    スレッドセーフではない 再生用と書き出し用で別インスタンスにすること
    """

    def __init__(self, project: Project) -> None:
        self._project = project
        self._decoders: OrderedDict[tuple[MediaId, int], AudioDecoder] = OrderedDict()
        self._closed = False

    @property
    def project(self) -> Project:
        return self._project

    @property
    def sample_rate(self) -> int:
        return self._project.settings.sample_rate

    @property
    def channels(self) -> int:
        return self._project.settings.channels

    def set_project(self, project: Project) -> None:
        previous = self._project
        self._project = project
        changed_format = (
            project.settings.sample_rate != previous.settings.sample_rate
            or project.settings.channels != previous.settings.channels
        )
        alive = {m.id for m in project.media}
        for key in [k for k in self._decoders if changed_format or k[0] not in alive]:
            self._decoders.pop(key).close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for decoder in self._decoders.values():
            decoder.close()
        self._decoders.clear()

    def render(self, start_sample: int, count: int) -> np.ndarray:
        """``[start_sample, start_sample + count)`` のミックス結果を返す

        形は ``(count, チャンネル数)`` の float32 クリッピングはしない
        ここで頭打ちにすると、後段のフェードやラウドネス調整で潰れた音しか
        扱えなくなる 出力段で 1 度だけ行う
        """
        if self._closed:
            raise RuntimeError("閉じたミキサは使えない")
        if count <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        return self._render_timeline(self._project.timeline, start_sample, count, depth=0)

    def _render_timeline(
        self, timeline: Timeline, start_sample: int, count: int, *, depth: int
    ) -> np.ndarray:
        """1 本のタイムラインの音 入れ子のシーンも同じ道を通る

        混ぜるのは 3 通り 音声トラックと混合トラックは、鳴らすクリップ
        （:meth:`~sashimono.core.model.Project.plays_sound`）にトラックの音量と定位を掛ける
        映像トラックはシーンの音だけを、トラックの音量と定位を掛けずに混ぜる
        映像トラックのシーンを外すと、シーンの中の BGM やナレーションが消える
        映像トラックのソロとミュートは絵の側（:meth:`Timeline.active_picture_tracks`）で決まる
        シーンは絵と音が 1 つなので、絵を隠したトラックのシーンは音も止める（今までと同じ）
        """
        out = np.zeros((count, self.channels), dtype=np.float32)
        rate = self._project.rate
        for track in timeline.active_sound_tracks():
            self._mix_track(out, track, start_sample, count, rate, depth)
        for track in timeline.active_picture_tracks():
            if track.kind is TrackKind.VIDEO:
                self._mix_track(out, track, start_sample, count, rate, depth)
        return out

    def render_frames(self, start_frame: int, frame_count: int) -> np.ndarray:
        """フレーム範囲で指定してミックスする 書き出し側の入口"""
        rate = self._project.rate
        start = _frame_to_sample(start_frame, rate, self.sample_rate)
        end = _frame_to_sample(start_frame + frame_count, rate, self.sample_rate)
        return self.render(start, end - start)

    def _mix_track(
        self,
        out: np.ndarray,
        track: Track,
        start_sample: int,
        count: int,
        rate: FrameRate,
        depth: int = 0,
    ) -> None:
        # 映像トラックの音量と定位は使わない決まり（画面にも出ていない） 置いたシーンの
        # 音だけを混ぜるときに掛けると、見えない値で音が変わる
        picture_only = track.kind is TrackKind.VIDEO
        gain = 1.0 if picture_only else _db_to_gain(track.volume_db)
        pan = 0.0 if picture_only else np.clip(track.pan, -1.0, 1.0)

        for clip in track.clips:
            if not clip.enabled:
                continue
            clip_start = _frame_to_sample(clip.timeline_start, rate, self.sample_rate)
            clip_end = _frame_to_sample(clip.timeline_end, rate, self.sample_rate)
            begin = max(start_sample, clip_start)
            end = min(start_sample + count, clip_end)
            # 鳴らすかは重なったクリップだけで見る 混合トラックでは素材の一覧を引くので、
            # 先に見ると塊ごとにトラックの全クリップぶん引くことになる
            if begin >= end or not self._project.plays_sound(track, clip):
                continue

            stream = heard_stream(track, clip)
            # 前の音を読むエフェクト（残響など）があれば、要るだけ前から読み、掛けてから頭を捨てる
            inside = begin - clip_start
            before = min(inside, lookback(clip, inside, self.sample_rate, rate))
            samples = self._read_clip(
                clip, inside - before, end - begin + before, depth, stream=stream
            )
            if samples is None:
                continue
            samples = _apply_effects(
                clip,
                samples,
                inside - before,
                self.sample_rate,
                clip_end - clip_start,
                rate,
                keep_from=before,
            )[before:]

            offset = begin - start_sample
            out[offset : offset + len(samples)] += _apply_pan(samples * gain, float(pan))

    def _read_clip(
        self,
        clip: Clip,
        offset_samples: int,
        count: int,
        depth: int = 0,
        *,
        stream: int | None,
    ) -> np.ndarray | None:
        """クリップ内の位置からサンプルを読む 速度変更があればここで反映する

        ``stream`` は開く音声ストリームの番号 音声トラックは :attr:`Clip.stream_index`、
        混合トラックは :attr:`Clip.audio_stream` 混合トラックのクリップの
        ``stream_index`` は絵のストリームを指すので、それで開くと別の音が鳴る
        """
        if clip.scene_id is not None:
            return self._read_scene(clip, offset_samples, count, depth)
        if clip.media_id is None or stream is None:
            return None
        media = self._project.find_media(clip.media_id)
        if media is None or not media.has_audio:
            return None

        decoder = self._decoder_for(clip.media_id, stream)
        if decoder is None:
            return None

        source_offset = clip.source_in * self.sample_rate
        if clip.speed == 1:
            start = int(source_offset) + offset_samples
            return decoder.read(start, count)

        # 速度変更 テープを速く回すのと同じで音程も変わる ピッチを保つ
        # タイムストレッチは別物なので、後のフェーズで独立した機能として入れる
        speed = float(clip.speed)
        start = int(source_offset + offset_samples * speed)
        needed = int(np.ceil(count * speed)) + 2
        source = decoder.read(start, needed)
        return _resample_linear(source, count, speed)

    def _read_scene(
        self, clip: Clip, offset_samples: int, count: int, depth: int
    ) -> np.ndarray | None:
        """入れ子のシーンの音 時刻の決まりは映像と同じ（``source_in`` と速度）"""
        scene = self._project.find_scene(clip.scene_id) if clip.scene_id else None
        if scene is None or depth >= MAX_SCENE_DEPTH:
            return None
        source_offset = int(clip.source_in * self.sample_rate)
        if clip.speed == 1:
            return self._render_timeline(
                scene.timeline, source_offset + offset_samples, count, depth=depth + 1
            )
        speed = float(clip.speed)
        start = int(source_offset + offset_samples * speed)
        needed = int(np.ceil(count * speed)) + 2
        source = self._render_timeline(scene.timeline, start, needed, depth=depth + 1)
        return _resample_linear(source, count, speed)

    def _decoder_for(self, media_id: MediaId, stream_index: int) -> AudioDecoder | None:
        key = (media_id, stream_index)
        existing = self._decoders.get(key)
        if existing is not None:
            self._decoders.move_to_end(key)
            return existing

        media = self._project.find_media(media_id)
        if media is None:
            return None
        try:
            decoder = AudioDecoder(
                media.path,
                sample_rate=self.sample_rate,
                channels=self.channels,
                stream_index=stream_index if stream_index else None,
            )
        except ProbeError:
            # オフライン素材 そのクリップだけ無音になり、再生自体は続く
            return None

        self._decoders[key] = decoder
        while len(self._decoders) > MAX_OPEN_DECODERS:
            _, evicted = self._decoders.popitem(last=False)
            evicted.close()
        return decoder


def lookback(clip: Clip, offset: int, sample_rate: int, rate: FrameRate) -> int:
    """クリップの ``offset`` サンプル目から掛けるのに、前の音をいくつ読み直すか

    前の音を読むエフェクト（:attr:`EffectDefinition.audio_history`）の要る長さを足す
    重ねて積むと、後ろのエフェクトが読む前の音も前のエフェクトを通した物が要る
    値は塊の頭のフレームで解く 上限は :data:`MAX_HISTORY_SECONDS`
    """
    frame = _sample_to_frame(
        _frame_to_sample(clip.timeline_start, rate, sample_rate) + offset, rate, sample_rate
    )
    seconds = 0.0
    for definition, effect in _audio_stack(clip):
        if definition.audio_history is not None:
            seconds += definition.audio_history(
                _values(definition, effect, frame - clip.timeline_start)
            )
    return int(np.ceil(min(seconds, MAX_HISTORY_SECONDS) * sample_rate))


def _audio_stack(clip: Clip) -> list[tuple[EffectDefinition, Effect]]:
    return [
        (definition, effect)
        for effect in clip.effects
        if effect.enabled
        and (definition := registry.get(effect.kind)) is not None
        and definition.audio_process is not None
    ]


def _values(definition: EffectDefinition, effect: Effect, frame: int) -> dict[str, float]:
    return {
        spec.name: _as_number(spec, effect.params.get(spec.name), frame)
        for spec in definition.parameters
        if isinstance(spec, TrackSpec)
    }


def _apply_effects(
    clip: Clip,
    samples: np.ndarray,
    offset: int,
    sample_rate: int,
    duration: int,
    rate: FrameRate,
    *,
    keep_from: int = 0,
) -> np.ndarray:
    """クリップに積んだ音のエフェクトを、置いた順に掛ける

    映像のエフェクトは飛ばす 同じクリップに映像と音の両方が積まれていても、
    音の側だけを見る（AviUtl も音声オブジェクトに映像フィルタを積める）

    動く値は**映像のフレームの切れ目で区切って**解く 塊の先頭で 1 度だけ解くと、
    プレビューの細かい塊（1024 サンプル）がフレームの切れ目をまたいだときに、
    音量の変わる時刻がずれて書き出しと合わなくなる

    前の音を読むエフェクトは塊を切らずに全体へ 1 度で掛け、値は ``keep_from``（呼ぶ側が
    残す所の頭 その前は読み直した前の音）のフレームで解く 1 つずつ全体へ掛けてから
    次へ進む（エフェクトの順に掛ける） 前の音を読むエフェクトが前のエフェクトを
    通した音を読めるように
    """
    stack = _audio_stack(clip)
    if not stack:
        return samples

    origin = _frame_to_sample(clip.timeline_start, rate, sample_rate)
    spans = list(
        _frame_spans(clip.timeline_start, origin + offset, len(samples), sample_rate, rate)
    )
    kept = min(max(keep_from, 0), max(len(samples) - 1, 0))
    kept_frame = next((f for b, e, f in spans if b <= kept < e), spans[0][2] if spans else 0)
    current = samples
    for definition, effect in stack:
        assert definition.audio_process is not None
        if definition.audio_history is not None:
            current = definition.audio_process(
                current,
                _values(definition, effect, kept_frame),
                AudioContext(offset=offset, sample_rate=sample_rate, duration=duration),
            )
            continue
        out = np.empty_like(current)
        for begin, end, frame in spans:
            out[begin:end] = definition.audio_process(
                current[begin:end],
                _values(definition, effect, frame),
                AudioContext(offset=offset + begin, sample_rate=sample_rate, duration=duration),
            )
        current = out
    return current


def _frame_spans(
    start_frame: int, offset: int, count: int, sample_rate: int, rate: FrameRate
) -> Iterator[tuple[int, int, int]]:
    """塊を映像のフレームごとに切り分ける ``(始まり, 終わり, フレーム)`` を返す

    ``offset`` は**タイムラインの先頭から数えた**この塊の先頭のサンプル位置
    切れ目はタイムラインの升目で決める 絵が切り替わる所と同じでなければ
    意味が無く、クリップの先頭から数え直すと 29.97 fps のような比で
    1 サンプルずれる（升目の幅は 1601 と 1602 が混ざるので、
    クリップを置く場所によって最初の升の幅が変わる）

    返すフレーム番号だけはクリップの先頭から数える 動く値のキーフレームが
    そちら基準のため 始まりがフレームの途中でも、最初の切れ目までを 1 つとして返す
    """
    begin = 0
    while begin < count:
        frame = _sample_to_frame(offset + begin, rate, sample_rate)
        boundary = _frame_to_sample(frame + 1, rate, sample_rate) - offset
        end = min(max(boundary, begin + 1), count)
        yield begin, end, frame - start_frame
        begin = end


def _as_number(spec: TrackSpec, value: ParamValue | None, frame: int) -> float:
    """設定の値を数として読む

    読み方は :meth:`EffectProcessor._set_parameters` と同じにする
    仕様を通さずに読むと、壊れた値（古いファイルの文字など）が 0 になり、
    既定が 100 の音量なら**クリップが丸ごと無音になる**

    受け取るのは :class:`TrackSpec` だけ 数にならない仕様（選択・真偽）まで
    黙って 0 として渡すと、既定値と違う値でエフェクトが走る
    音のエフェクトが数以外の項目を持たないことは試験で見張る
    """
    number = spec.coerce(spec.default_value() if value is None else value).at(frame)
    return float(number) if math.isfinite(number) else float(spec.default)


def _frame_to_sample(frame: int, rate: FrameRate, sample_rate: int) -> int:
    """フレーム番号を、そのフレームが始まるサンプル番号へ

    :mod:`sashimono.core.timebase` の変換をそのまま使うと ``Fraction`` の生成が
    サンプルごとに走る ここは再生のたびに通るので、整数演算で済ませる
    """
    return frame * rate.den * sample_rate // rate.num


def _sample_to_frame(sample: int, rate: FrameRate, sample_rate: int) -> int:
    """サンプル番号を、それが属するフレーム番号へ（:func:`_frame_to_sample` の逆）

    1 フレームあたりのサンプル数で割ると、29.97 fps のような割り切れない比で
    :func:`_frame_to_sample` と食い違う（1 フレームは 1601.6 サンプルで、
    フレーム 1 は切り捨てて 1601 から始まるのに 1601 / 1601.6 は 0 になる）
    切れ目とフレーム番号がずれると、動く値の変わる時刻が 1 サンプル遅れる
    そこで ``_frame_to_sample(f) <= sample`` を満たす最大の f を整数のまま出す
    """
    span = rate.den * sample_rate
    return -((-(sample + 1) * rate.num) // span) - 1


def _db_to_gain(db: float) -> float:
    if db == 0.0:
        return 1.0
    return float(10.0 ** (db / 20.0))


def _apply_pan(samples: np.ndarray, pan: float) -> np.ndarray:
    """定電力パンニング

    左右の音量を単純な線形で振ると、中央で音圧が下がって聞こえる
    左右のゲインの二乗和が一定になるようにする
    """
    if pan == 0.0 or samples.shape[1] != 2:
        return samples
    angle = (pan + 1.0) * np.pi / 4.0
    gains = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * np.float32(np.sqrt(2.0))
    return np.asarray(samples * gains, dtype=np.float32)


def _resample_linear(source: np.ndarray, count: int, speed: float) -> np.ndarray:
    """線形補間でサンプル数を変える

    速度変更のプレビュー品質としては十分 書き出し品質を上げたくなったら、
    ここを多相フィルタに差し替える
    """
    if count <= 0:
        return np.zeros((0, source.shape[1]), dtype=np.float32)

    positions = np.arange(count, dtype=np.float64) * speed
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, len(source) - 1)
    left = np.clip(left, 0, max(len(source) - 1, 0))
    weight = (positions - left).astype(np.float32)[:, None]
    if len(source) == 0:
        return np.zeros((count, 1), dtype=np.float32)
    blended = source[left] * (1.0 - weight) + source[right] * weight
    return np.asarray(blended, dtype=np.float32)
