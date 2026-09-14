import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig, Transformer
from ie.serve.disaggregated import DisaggregatedEngine, KVTransfer

VOCAB = 2048
CONFIG = ModelConfig(vocab_size=VOCAB, hidden_size=64, num_layers=3, num_heads=4,
                     num_kv_heads=2, head_dim=16, intermediate_size=128)
ENGINE = EngineConfig(block_size=16, num_blocks=512, max_batch_tokens=512)
PROMPTS = [list(range(100 + i * 80, 100 + i * 80 + 48)) for i in range(4)]


def colocated(max_new_tokens=10, prompts=None):
    return build_engine(CONFIG, ENGINE, model_seed=1).generate(
        prompts or PROMPTS, max_new_tokens=max_new_tokens, temperature=0.0)


def split():
    return DisaggregatedEngine(Transformer(CONFIG, seed=1), ENGINE)


def test_splitting_prefill_from_decode_does_not_change_the_output():
    assert split().generate(PROMPTS, max_new_tokens=10, temperature=0.0) == colocated()


@pytest.mark.parametrize("new_tokens", [1, 5, 12])
def test_output_matches_at_any_generation_length(new_tokens):
    assert (split().generate(PROMPTS, max_new_tokens=new_tokens, temperature=0.0)
            == colocated(new_tokens))


def test_a_sequence_that_arrives_complete_is_not_decoded_again():
    engine = split()
    transfer = engine.prefill.prefill(PROMPTS[0])
    seq = engine.decode.admit(transfer, max_new_tokens=1)
    assert seq.is_finished
    assert seq.num_generated == 1
    assert engine.decode.engine.blocks.num_free == engine.decode.engine.blocks.num_total


def test_a_single_prompt_transfers_its_whole_cache():
    engine = split()
    prompt = PROMPTS[0]
    transfer = engine.prefill.prefill(prompt)

    assert transfer.num_blocks == (len(prompt) + 15) // 16
    assert transfer.num_computed == len(prompt)
    assert len(transfer.token_ids) == len(prompt) + 1
    assert transfer.bytes_moved > 0


def test_the_prefill_worker_releases_its_blocks_after_transfer():
    engine = split()
    engine.prefill.prefill(PROMPTS[0])
    assert engine.prefill.engine.blocks.num_free == engine.prefill.engine.blocks.num_total


def test_transfer_volume_grows_with_prompt_length():
    engine = split()
    short = engine.prefill.prefill(list(range(500, 516)))
    long = engine.prefill.prefill(list(range(600, 700)))
    assert long.bytes_moved > short.bytes_moved
    assert long.num_blocks > short.num_blocks


def test_stats_account_for_every_transfer():
    engine = split()
    engine.generate(PROMPTS, max_new_tokens=6, temperature=0.0)

    assert engine.stats.prefills == len(PROMPTS)
    assert engine.stats.transfers == len(PROMPTS)
    assert engine.stats.blocks_transferred > 0
    assert engine.stats.decodes > 0


def test_the_decode_worker_refuses_a_cache_it_cannot_hold():
    tiny = EngineConfig(block_size=16, num_blocks=2, max_batch_tokens=512)
    engine = DisaggregatedEngine(Transformer(CONFIG, seed=1), ENGINE)
    engine.decode = type(engine.decode)(Transformer(CONFIG, seed=1), tiny)

    transfer = engine.prefill.prefill(list(range(300, 400)))
    with pytest.raises(RuntimeError):
        engine.decode.admit(transfer, max_new_tokens=4)
