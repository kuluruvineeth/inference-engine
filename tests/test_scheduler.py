import pytest

from ie.core.block_manager import BlockManager
from ie.core.scheduler import Scheduler
from ie.core.sequence import Sequence, Status

BS = 4


def build(num_blocks=64, block_size=BS, max_batch_sequences=8, max_batch_tokens=32):
    blocks = BlockManager(num_blocks=num_blocks, block_size=block_size)
    return Scheduler(blocks, max_batch_sequences=max_batch_sequences,
                     max_batch_tokens=max_batch_tokens)


def seq(tokens, block_size=BS, **kw):
    return Sequence(tokens, block_size=block_size, **kw)


def run_to_completion(sched, max_steps=10_000):
    steps = 0
    while sched.has_work and steps < max_steps:
        batch = sched.schedule()
        if not batch:
            break
        sched.finish_step(batch, [7] * len(batch))
        steps += 1
    return steps


def test_prefill_runs_before_decode():
    sched = build()
    sched.add(seq([1, 2, 3]))
    batch = sched.schedule()
    assert batch.is_prefill
    assert batch.num_tokens == 3


def test_prompt_is_prefilled_then_sequence_decodes_one_token_per_step():
    sched = build()
    s = seq([1, 2, 3], max_new_tokens=3)
    sched.add(s)

    first = sched.schedule()
    assert first.is_prefill
    sched.finish_step(first, [10])
    assert len(s) == 4

    second = sched.schedule()
    assert not second.is_prefill
    assert second.num_tokens == 1


def test_token_budget_caps_the_prefill_batch():
    sched = build(max_batch_tokens=16)
    for _ in range(4):
        sched.add(seq(list(range(10))))
    batch = sched.schedule()
    assert batch.num_tokens <= 16
    assert len(batch) < 4


def test_a_long_prompt_is_chunked_across_iterations():
    sched = build(max_batch_tokens=8)
    s = seq(list(range(20)))
    sched.add(s)

    first = sched.schedule()
    assert first.num_tokens == 8
    assert not s.prompt_is_fully_computed
    sched.finish_step(first, [0])
    assert len(s) == 20

    second = sched.schedule()
    assert second.is_prefill
    assert second.num_tokens == 8
    sched.finish_step(second, [0])

    third = sched.schedule()
    assert third.num_tokens == 4
    sched.finish_step(third, [99])
    assert s.prompt_is_fully_computed
    assert len(s) == 21


def test_only_the_first_sequence_may_be_chunked():
    sched = build(max_batch_tokens=12)
    short = seq([1, 2, 3, 4])
    long = seq(list(range(20)))
    sched.add(short)
    sched.add(long)

    batch = sched.schedule()
    assert batch.sequences == [short]


def test_finished_sequence_leaves_and_releases_its_blocks():
    sched = build()
    s = seq([1, 2], max_new_tokens=1)
    sched.add(s)

    free_before = sched.blocks.num_free
    batch = sched.schedule()
    done = sched.finish_step(batch, [42])

    assert done == [s]
    assert s.status is Status.FINISHED
    assert s.stop_reason() == "length"
    assert sched.blocks.num_free == free_before


def test_eos_stops_the_sequence():
    sched = build()
    s = seq([1, 2], max_new_tokens=50, eos_id=999)
    sched.add(s)
    batch = sched.schedule()
    done = sched.finish_step(batch, [999])
    assert done == [s]
    assert s.stop_reason() == "eos"


def test_new_request_joins_while_others_are_decoding():
    sched = build()
    early = seq([1, 2], max_new_tokens=20)
    sched.add(early)
    sched.finish_step(sched.schedule(), [5])

    late = seq([7, 8], max_new_tokens=20)
    sched.add(late)

    batch = sched.schedule()
    assert batch.is_prefill
    assert batch.sequences == [late]

    mixed = sched.schedule()
    assert not mixed.is_prefill
    assert set(mixed.sequences) == {early, late}


def test_decode_preempts_when_blocks_run_out():
    sched = build(num_blocks=3, block_size=BS, max_batch_tokens=64)
    a = seq([1, 2, 3, 4], max_new_tokens=20)
    b = seq([5, 6, 7, 8], max_new_tokens=20)
    sched.add(a)
    sched.add(b)

    first = sched.schedule()
    sched.finish_step(first, [0] * len(first))
    assert sched.blocks.num_free == 1

    for _ in range(6):
        batch = sched.schedule()
        if not batch:
            break
        sched.finish_step(batch, [0] * len(batch))

    assert sched.stats.preemptions > 0


def test_preempted_sequence_keeps_its_tokens_and_finishes_later():
    sched = build(num_blocks=3, block_size=BS, max_batch_tokens=64)
    a = seq([1, 2, 3, 4], max_new_tokens=6)
    b = seq([5, 6, 7, 8], max_new_tokens=6)
    sched.add(a)
    sched.add(b)

    run_to_completion(sched)

    assert sched.stats.preemptions > 0
    assert a.status is Status.FINISHED
    assert b.status is Status.FINISHED
    assert a.num_generated == 6
    assert b.num_generated == 6


def test_every_sequence_eventually_finishes_under_pressure():
    sched = build(num_blocks=6, block_size=BS, max_batch_sequences=4, max_batch_tokens=16)
    requests = [seq(list(range(i, i + 5)), max_new_tokens=4) for i in range(6)]
    for s in requests:
        sched.add(s)

    run_to_completion(sched)

    assert not sched.has_work
    assert sched.stats.finished == 6
    assert all(s.num_generated == 4 for s in requests)
    assert sched.blocks.num_free == sched.blocks.num_total


def test_shared_prefix_reduces_prefill_work():
    prompt = list(range(32))
    sched = build(num_blocks=128, max_batch_tokens=256)

    sched.add(seq(prompt + [0], max_new_tokens=1))
    run_to_completion(sched)
    cold_prefill = sched.stats.tokens_prefilled
    assert sched.stats.tokens_reused_from_cache == 0

    for i in range(1, 4):
        sched.add(seq(prompt + [i], max_new_tokens=1))
        run_to_completion(sched)

    assert sched.stats.tokens_reused_from_cache > 0
    warm_prefill = sched.stats.tokens_prefilled - cold_prefill
    assert warm_prefill < 3 * cold_prefill


def test_finish_step_rejects_a_token_count_mismatch():
    sched = build()
    sched.add(seq([1, 2]))
    batch = sched.schedule()
    with pytest.raises(ValueError):
        sched.finish_step(batch, [1, 2, 3])
