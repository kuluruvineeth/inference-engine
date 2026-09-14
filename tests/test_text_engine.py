import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.engine.text import TextEngine
from ie.load.tokenizer import ChatMessage, OfflineTokenizer, apply_chat_template
from ie.model.transformer import ModelConfig

CONFIG = ModelConfig(vocab_size=256, hidden_size=32, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)
VOCAB = {word: index for index, word in enumerate(
    ["<eos>", "the", "cat", "sat", "on", "mat", "dog", "ran"], start=0)}


def text_engine():
    engine = build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=256,
                                               max_batch_tokens=256), model_seed=1)
    return TextEngine(engine, OfflineTokenizer(VOCAB, eos_id=0))


def test_offline_tokenizer_round_trips_known_words():
    tokenizer = OfflineTokenizer(VOCAB)
    assert tokenizer.decode(tokenizer.encode("the cat sat")) == "the cat sat"


def test_unknown_words_still_encode():
    tokenizer = OfflineTokenizer(VOCAB)
    assert len(tokenizer.encode("xylophone")) == 1


def test_an_empty_string_still_yields_a_token():
    assert OfflineTokenizer(VOCAB).encode("") == [0]


def test_completion_returns_text_and_counts():
    results = text_engine().complete(["the cat sat"], max_new_tokens=6)
    assert len(results) == 1
    assert isinstance(results[0].text, str)
    assert results[0].prompt_tokens == 3
    assert results[0].completion_tokens == 6
    assert results[0].stop_reason == "length"


def test_several_prompts_complete_together():
    results = text_engine().complete(["the cat", "the dog ran"], max_new_tokens=4)
    assert [r.prompt for r in results] == ["the cat", "the dog ran"]
    assert all(r.completion_tokens == 4 for r in results)


def test_streaming_yields_text_incrementally():
    pieces = list(text_engine().stream("the cat sat", max_new_tokens=5))
    assert len(pieces) == 5
    assert all(isinstance(piece, str) for piece in pieces)


def test_streamed_text_concatenates_to_the_completion():
    streamed = "".join(text_engine().stream("the cat", max_new_tokens=6))
    batched = text_engine().complete(["the cat"], max_new_tokens=6)[0].text
    assert streamed == batched


def test_chat_template_marks_roles_and_opens_the_reply():
    rendered = apply_chat_template([ChatMessage("system", "be brief"),
                                    ChatMessage("user", "hello")])
    assert "<|im_start|>system" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_chat_template_can_omit_the_generation_prompt():
    rendered = apply_chat_template([ChatMessage("user", "hi")], add_generation_prompt=False)
    assert not rendered.endswith("assistant\n")
