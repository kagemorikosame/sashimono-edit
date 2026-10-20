"""VOICEVOX 互換のエンジンを呼ぶ口（ナレーション #335）

偽のエンジン（tests/fake_voice_engine.py）を 127.0.0.1 に立て、本物の HTTP で話す
"""

from __future__ import annotations

import io
import wave
from collections.abc import Iterator

import pytest

from sashimono.engine.audio.voicevox import (
    AUTO_VOICE,
    PIECE_LIMIT,
    EngineError,
    Tuning,
    VoiceEngine,
    check_engine_url,
    join_wavs,
    split_script,
    wav_seconds,
)
from tests.fake_voice_engine import (
    SECONDS_PER_CHAR,
    FakeVoiceEngine,
    _wav,
    closed_engine_url,
    fake_voice_engine,
)


@pytest.fixture
def engine() -> Iterator[FakeVoiceEngine]:
    with fake_voice_engine() as running:
        yield running


class TestThePlaceStaysOnThisMachine:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("http://127.0.0.1:50021", "http://127.0.0.1:50021"),
            ("http://127.0.0.1:10101/", "http://127.0.0.1:10101"),
            (" http://localhost:50021 ", "http://localhost:50021"),
            ("http://[::1]:50021", "http://[::1]:50021"),
            ("http://127.0.0.2:50021", "http://127.0.0.2:50021"),
        ],
    )
    def test_places_on_this_machine_are_taken(self, raw: str, expected: str) -> None:
        # 壊れると、VOICEVOX や AivisSpeech を起動した人の場所（自分の機械の中）まで断り、
        # ナレーションを作れない
        assert check_engine_url(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "http://192.168.1.10:50021",
            "http://example.com:50021",
            "http://voicevox.example.com",
            "https://127.0.0.1:50021",
            "http://user:pass@127.0.0.1:50021",
            "http://127.0.0.1:50021/api",
            "http://127.0.0.1:50021?x=1",
            "ftp://127.0.0.1",
            "",
            "http://127.0.0.1:99999",
        ],
    )
    def test_other_places_are_refused(self, raw: str) -> None:
        # 壊れると、設定に外の機械の場所を書いただけで、台本の文（本人の文）が機械の外へ出る
        # （2026-10-20 利用者の決定 外へ送らない）
        with pytest.raises(ValueError):
            check_engine_url(raw)

    def test_the_engine_refuses_to_be_made_for_an_outside_place(self) -> None:
        # 壊れると、確かめを通らずに作ったエンジンの口から外へつながる
        with pytest.raises(EngineError, match="自分の機械の中"):
            VoiceEngine("http://192.168.1.10:50021")


class TestTheVoiceList:
    def test_voices_brand_and_terms_are_read(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、AI と設定の窓に声の名前が並ばず、表記（VOICEVOX:ずんだもん）の頭も分からない
        catalog = VoiceEngine(engine.url).catalog()
        assert catalog.engine_name == "VOICEVOX Engine"
        assert catalog.brand == "VOICEVOX"
        assert [speaker.name for speaker in catalog.speakers] == ["四国めたん", "ずんだもん"]
        zunda = catalog.speakers[1]
        assert catalog.credit(zunda) == "VOICEVOX:ずんだもん"
        assert catalog.terms == "エンジンの利用規約"

    def test_singing_voices_are_not_offered(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、文を読めない歌の声を選べてしまい、作る所で断られる
        catalog = VoiceEngine(engine.url).catalog()
        ids = {style.id for speaker in catalog.speakers for style in speaker.styles}
        assert 3001 not in ids
        assert ids == {0, 2, 3}

    def test_the_first_voice_is_used_when_none_is_chosen(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、設定で声を選んでいない人（既定のまま）のナレーションが作れない
        catalog = VoiceEngine(engine.url).catalog()
        found = catalog.find(AUTO_VOICE)
        assert found is not None
        assert (found[0].name, found[1].id) == ("四国めたん", 2)
        assert catalog.find(999) is None

    def test_an_old_engine_without_a_manifest_still_lists_voices(self) -> None:
        # 壊れると、目録を持たない古いエンジンで、声はあるのに一覧ごと断る
        with fake_voice_engine(FakeVoiceEngine(manifest=None)) as old:
            catalog = VoiceEngine(old.url).catalog()
        assert catalog.brand == "音声合成"
        assert len(catalog.speakers) == 2

    def test_an_aivis_like_brand_is_used_for_the_credit(self) -> None:
        # 壊れると、AivisSpeech の声にも「VOICEVOX:」と付け、違うアプリの名前で表記させる
        manifest = {"name": "AivisSpeech Engine", "brand_name": "AivisSpeech"}
        speakers = [
            {
                "name": "まお",
                "speaker_uuid": "a670e6b8-0852-45b2-8704-1bc9862f2fe6",
                "styles": [{"name": "ノーマル", "id": 888753760, "type": "talk"}],
            }
        ]
        with fake_voice_engine(FakeVoiceEngine(manifest=manifest, speakers=speakers)) as aivis:
            catalog = VoiceEngine(aivis.url).catalog()
        assert catalog.credit(catalog.speakers[0]) == "AivisSpeech:まお"
        assert catalog.find(888753760) is not None

    def test_the_policy_of_one_voice_is_read(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、声ごとの規約（クレジットの書き方・企業での利用）を本人に見せられない
        voice = VoiceEngine(engine.url)
        zunda = voice.catalog().speakers[1]
        assert voice.policy(zunda) == "ずんだもんの規約 クレジットを書くこと"

    def test_a_long_policy_is_cut_with_a_note(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、長い規約で AI へ返す結果の上限を越え、一覧ごと切れる
        engine.policies["388f246b-8c41-4ac1-8e2d-5d79f3ff56d9"] = "規" * 5000
        voice = VoiceEngine(engine.url)
        policy = voice.policy(voice.catalog().speakers[1])
        assert len(policy) < 2100
        assert "エンジンのアプリ" in policy

    def test_an_engine_with_no_voices_is_refused_with_a_reason(self) -> None:
        # 壊れると、声を入れていないエンジンで空の一覧を返し、AI が無い声を探し続ける
        with (
            fake_voice_engine(FakeVoiceEngine(speakers=[])) as empty,
            pytest.raises(EngineError, match="使える声がありません"),
        ):
            VoiceEngine(empty.url).catalog()


class TestWhenTheEngineIsNotRunning:
    def test_a_stopped_engine_is_refused_with_how_to_fix_it(self) -> None:
        # 壊れると、エンジンを起動していない機械（CI もそう）で、何が足りないのか分からない
        # 失敗の文が返り、AI が同じ頼みを繰り返す
        with pytest.raises(EngineError) as caught:
            VoiceEngine(closed_engine_url()).catalog()
        assert "起動" in str(caught.value)
        assert "表示 → 設定…" in str(caught.value)

    def test_a_redirect_is_not_followed(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、自分の機械の中の相手が外の場所へ飛ばしたとき、付いて行って台本の文を
        # 外へ送る
        engine.redirect_to = "http://example.com/speakers"
        with pytest.raises(EngineError, match="HTTP 302"):
            VoiceEngine(engine.url).catalog()
        assert all(path.startswith("/") for _method, path, _body in engine.requests)

    def test_a_refusal_does_not_carry_the_script(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、エンジンの断りの本文（頼んだ文を書き返す）をそのまま AI へ返し、字幕と
        # 文字を送らないと決めた人の文まで外へ出る
        secret = "秘密の台本の文"
        with pytest.raises(EngineError) as caught:
            VoiceEngine(engine.url).synthesize(secret, 999, Tuning())
        assert "422" in str(caught.value)
        assert secret not in str(caught.value)


class TestMakingTheVoice:
    def test_a_short_script_is_made_in_one_piece_with_the_tuning(
        self, engine: FakeVoiceEngine
    ) -> None:
        # 壊れると、速さや声の大きさを頼んでも、エンジンの既定のままの声になる
        data = VoiceEngine(engine.url).synthesize(
            "こんにちは。", 3, Tuning(speed_percent=120, volume_percent=80, pitch=-5)
        )
        assert engine.texts == ["こんにちは。"]
        query = engine.queries[0]
        assert query["speedScale"] == pytest.approx(1.2)
        assert query["volumeScale"] == pytest.approx(0.8)
        assert query["pitchScale"] == pytest.approx(-0.05)
        assert query["intonationScale"] == pytest.approx(1.0)
        assert wav_seconds(data.wav) == pytest.approx(
            len("こんにちは。") * SECONDS_PER_CHAR, abs=0.001
        )

    def test_a_long_script_is_split_at_sentences_and_joined(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、100 文字を超える台本で AivisSpeech が断るか抑揚が崩れ、つないだ声が
        # 途中で切れる
        sentence = "今日は動画編集のこつを話します。" * 12
        data = VoiceEngine(engine.url).synthesize(sentence, 3, Tuning())
        assert len(engine.texts) > 1
        assert all(len(text) <= PIECE_LIMIT for text in engine.texts)
        assert "".join(engine.texts) == sentence
        assert all(text.endswith("。") for text in engine.texts)
        expected = sum(len(text) for text in engine.texts) * SECONDS_PER_CHAR
        assert wav_seconds(data.wav) == pytest.approx(expected, abs=0.01)

    def test_each_sentence_keeps_its_own_length(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、字幕を声の長さに合わせられず、2 文目の字幕が 1 文目の声のうちに出る
        spoken = VoiceEngine(engine.url).synthesize("はい。今日は長めに話します。", 3, Tuning())
        assert [line.text for line in spoken.lines] == ["はい。", "今日は長めに話します。"]
        assert spoken.lines[0].seconds == pytest.approx(3 * SECONDS_PER_CHAR, abs=0.001)
        assert spoken.lines[1].seconds == pytest.approx(11 * SECONDS_PER_CHAR, abs=0.001)
        total = sum(line.seconds for line in spoken.lines)
        assert wav_seconds(spoken.wav) == pytest.approx(total, abs=0.001)

    def test_pieces_of_different_shapes_are_refused(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、塊ごとにサンプリングレートの違う声を黙ってつなぎ、途中から高さと速さが変わる
        engine.rates = [24000, 44100]
        with pytest.raises(EngineError, match="つなげません"):
            VoiceEngine(engine.url).synthesize("あ。" * 60, 3, Tuning())

    def test_stopping_between_pieces_stops_asking_the_engine(self, engine: FakeVoiceEngine) -> None:
        # 壊れると、待つのをやめた道具の後も、長い台本の残りをエンジンに作らせ続ける
        calls: list[int] = []

        def stopped() -> bool:
            calls.append(1)
            return len(calls) > 1

        with pytest.raises(EngineError, match="止めました"):
            VoiceEngine(engine.url).synthesize("あ。" * 60, 3, Tuning(), stopped=stopped)
        assert len(engine.texts) == 1


class TestSplitting:
    def test_each_sentence_is_its_own_piece(self) -> None:
        # 壊れると、文をまたいで詰めて作り、字幕を文ごとに区切れない（どこで次の文になるかが
        # 声から分からない 2026-10-20 利用者の決定 字幕は文ごと）
        assert split_script("あ。い。う。", 4) == ["あ。", "い。", "う。"]
        assert split_script("今日は？晴れ！") == ["今日は？", "晴れ！"]

    def test_a_long_sentence_is_cut_at_commas_then_by_length(self) -> None:
        # 壊れると、句点の無い長い文が上限を越えたまま渡り、エンジンに断られる
        text = "あ" * 30 + "、" + "い" * 30
        pieces = split_script(text, 20)
        assert all(len(piece) <= 20 for piece in pieces)
        assert "".join(pieces) == text

    def test_blank_lines_are_dropped(self) -> None:
        # 壊れると、空の塊をエンジンへ渡して断られる
        assert split_script("\n\n  \n") == []
        assert split_script("一行目\n\n二行目") == ["一行目", "二行目"]


class TestJoining:
    def test_one_wave_is_returned_as_it_is(self) -> None:
        # 壊れると、1 本だけの声も作り直し、エンジンの書いた頭書き（形の情報）を失う
        data = _wav(0.1)
        assert join_wavs([data]) == data

    def test_something_that_is_not_a_wave_is_refused(self) -> None:
        # 壊れると、壊れた返事をそのまま WAV として書き、読み込めない素材が作品のフォルダに残る
        with pytest.raises(EngineError, match="WAV"):
            join_wavs([b"not a wave"])

    def test_joined_waves_keep_the_shape(self) -> None:
        # 壊れると、つないだ声のサンプリングレートが変わり、速さと高さが変わって聞こえる
        joined = join_wavs([_wav(0.1), _wav(0.2)])
        with wave.open(io.BytesIO(joined), "rb") as read:
            assert read.getframerate() == 24000
            assert read.getnchannels() == 1
        assert wav_seconds(joined) == pytest.approx(0.3, abs=0.001)
