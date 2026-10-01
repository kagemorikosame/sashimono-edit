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

#: 前の音を読むエフェクト（残響など）を掛ける区切り（クリップの頭から数えたサンプル）
#: 再生の塊（1024）ごとに掛けると、そのたびに前の音（残響なら 1.5 秒）を読み直して解き直し、
#: 圧縮した音では 1 塊 19ms（予算 21ms）掛かって再生が途切れた 区切りごとに 1 度だけ掛けて
#: 覚えておき、塊はそこから切り出す 区切りの位置は頼まれ方に依らず決まるので、再生と
#: 書き出しで同じ音になる 長くすると値の変わり目が粗くなる（0.34 秒）
HISTORY_WINDOW = 16384
#: 掛け終えた区切りを覚えておく数 再生の今の所と、シークで戻った所の分
_KEPT_WINDOWS = 16


class AudioMixer:
    """プロジェクトの音声を、指定したサンプル範囲について合成する

    スレッドセーフではない 再生用と書き出し用で別インスタンスにすること
    """

    def __init__(self, project: Project) -> None:
        self._project = project
        self._decoders: OrderedDict[tuple[MediaId, int], AudioDecoder] = OrderedDict()
        self._closed = False
        #: 前の音を読むエフェクトを掛け終えた区切り 鍵はクリップ・音・深さ・区切りの番号
        #: クリップそのものも持ち、値を変えた（別のクリップになった）ら使わない
        self._windows: OrderedDict[tuple[str, int | None, int, int], tuple[Clip, np.ndarray]] = (
            OrderedDict()
        )
        #: 前の音を読むクリップの、読んだ元の音 続きを読むときは前に読んだ所から先だけを
        #: 読む（前へ戻って読み直すと、圧縮した音は解き直しになる）
        self._raw: dict[tuple[str, int | None, int], tuple[Clip, int, np.ndarray]] = {}

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
        # 前の音を読むエフェクトの貯めは、今のプロジェクトに同じクリップ（同じ物）がある分
        # だけ残す 消したクリップや値を変えたクリップの分は二度と使われず、残すと 1 本
        # 4 MB ほどずつ増え続ける（PR #231 の指摘） 形式が変われば中身ごと使えない
        present = set() if changed_format else _clip_identities(project)
        self._windows = OrderedDict(
            (key, value) for key, value in self._windows.items() if id(value[0]) in present
        )
        self._raw = {key: value for key, value in self._raw.items() if id(value[0]) in present}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for decoder in self._decoders.values():
            decoder.close()
        self._decoders.clear()
        self._windows.clear()
        self._raw.clear()

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
            inside = begin - clip_start
            duration = clip_end - clip_start
            if _reads_history(clip):
                # 前の音を読むエフェクト（残響など）は区切りごとに掛けて覚え、そこから切り出す
                samples = self._windowed(clip, stream, depth, inside, end - begin, duration, rate)
            else:
                samples = self._read_clip(clip, inside, end - begin, depth, stream=stream)
                if samples is not None:
                    samples = _apply_effects(
                        clip, samples, inside, self.sample_rate, duration, rate
                    )
            if samples is None:
                continue

            offset = begin - start_sample
            out[offset : offset + len(samples)] += _apply_pan(samples * gain, float(pan))

    def _windowed(
        self,
        clip: Clip,
        stream: int | None,
        depth: int,
        offset: int,
        count: int,
        duration: int,
        rate: FrameRate,
    ) -> np.ndarray | None:
        """前の音を読むエフェクトを掛けた音の ``[offset, offset + count)``（クリップの中の位置）"""
        pieces: list[np.ndarray] = []
        position = offset
        stop = offset + count
        while position < stop:
            number = position // HISTORY_WINDOW
            window = self._window(clip, stream, depth, number, duration, rate)
            if window is None:
                return None
            head = number * HISTORY_WINDOW
            upto = min(stop - head, len(window))
            if upto <= position - head:
                break
            pieces.append(window[position - head : upto])
            position = head + upto
        if not pieces:
            return None
        joined = np.concatenate(pieces)
        if len(joined) < count:
            joined = np.concatenate(
                [joined, np.zeros((count - len(joined), joined.shape[1]), dtype=np.float32)]
            )
        return joined

    def _window(
        self,
        clip: Clip,
        stream: int | None,
        depth: int,
        number: int,
        duration: int,
        rate: FrameRate,
    ) -> np.ndarray | None:
        """``number`` 番目の区切りに、前の音を読んでからエフェクトを掛けた音"""
        key = (str(clip.id), stream, depth, number)
        found = self._windows.get(key)
        if found is not None and found[0] is clip:
            self._windows.move_to_end(key)
            return found[1]
        start = number * HISTORY_WINDOW
        end = min(start + HISTORY_WINDOW, duration)
        if end <= start:
            return None
        before = min(start, lookback(clip, start, self.sample_rate, rate))
        raw = self._raw_range(clip, stream, depth, start - before, end)
        if raw is None:
            return None
        processed = _apply_effects(
            clip, raw, start - before, self.sample_rate, duration, rate, keep_from=before
        )[before:]
        self._windows[key] = (clip, processed)
        while len(self._windows) > _KEPT_WINDOWS:
            self._windows.popitem(last=False)
        return processed

    def _raw_range(
        self, clip: Clip, stream: int | None, depth: int, begin: int, end: int
    ) -> np.ndarray | None:
        """クリップの元の音の ``[begin, end)`` 前に読んだ所の続きなら、先だけを読み足す"""
        key = (str(clip.id), stream, depth)
        kept = self._raw.get(key)
        if kept is not None and kept[0] is clip and kept[1] <= begin <= kept[1] + len(kept[2]):
            _, origin, samples = kept
            reach = origin + len(samples)
            if end > reach:
                more = self._read_clip(clip, reach, end - reach, depth, stream=stream)
                if more is None:
                    return None
                samples = np.concatenate([samples, more])
        else:
            read = self._read_clip(clip, begin, end - begin, depth, stream=stream)
            if read is None:
                return None
            origin, samples = begin, read
        # 次の区切りが読み直す前の音の分だけ残す 全部残すと長いクリップで増え続ける
        keep = int(MAX_HISTORY_SECONDS * self.sample_rate) + HISTORY_WINDOW
        if len(samples) > keep:
            cut = len(samples) - keep
            origin += cut
            samples = samples[cut:]
        self._raw[key] = (clip, origin, samples)
        return samples[begin - origin : end - origin]

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
    for definition, effect in audio_stack(clip):
        if definition.audio_history is not None:
            seconds += definition.audio_history(
                effect_values(definition, effect, frame - clip.timeline_start)
            )
    return int(np.ceil(min(seconds, MAX_HISTORY_SECONDS) * sample_rate))


def _clip_identities(project: Project) -> set[int]:
    """メインとシーンのタイムラインにあるクリップの物としての番号（``id``）

    貯めはクリップそのもの（``is``）で照合している 同じ ID でも値を変えたクリップは
    別の物なので、ここに無ければ貯めを捨てる
    """
    timelines = (project.timeline, *(scene.timeline for scene in project.scenes))
    return {id(clip) for timeline in timelines for track in timeline.tracks for clip in track.clips}


def _reads_history(clip: Clip) -> bool:
    """前の音を読むエフェクト（:attr:`EffectDefinition.audio_history`）が効いているか"""
    return any(definition.audio_history is not None for definition, _ in audio_stack(clip))


def audio_stack(clip: Clip) -> list[tuple[EffectDefinition, Effect]]:
    """クリップに積んだ、効いている音のエフェクト 積んだ順"""
    return [
        (definition, effect)
        for effect in clip.effects
        if effect.enabled
        and (definition := registry.get(effect.kind)) is not None
        and definition.audio_process is not None
    ]


def effect_values(definition: EffectDefinition, effect: Effect, frame: int) -> dict[str, float]:
    """音のエフェクトの数の値を ``frame``（クリップの頭から数えた）で解く 鳴らす所と波形で共通"""
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
    stack = audio_stack(clip)
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
                effect_values(definition, effect, kept_frame),
                AudioContext(offset=offset, sample_rate=sample_rate, duration=duration),
            )
            continue
        out = np.empty_like(current)
        for begin, end, frame in spans:
            out[begin:end] = definition.audio_process(
                current[begin:end],
                effect_values(definition, effect, frame),
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
