import random

from ie.core.block_manager import BlockManager
from ie.core.scheduler import Scheduler
from ie.core.sequence import Sequence

BLOCK_SIZE = 16
SYSTEM_PROMPT = list(range(9000, 9064))


def make_scheduler(num_blocks):
    return Scheduler(BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE),
                     max_batch_sequences=16, max_batch_tokens=512)


def drain(sched):
    while sched.has_work:
        batch = sched.schedule()
        if not batch:
            break
        sched.finish_step(batch, [random.randint(10, 99) for _ in range(len(batch))])
    return sched.stats


def workload(sched, share_prefix):
    random.seed(0)
    for i in range(24):
        prompt = (SYSTEM_PROMPT if share_prefix else list(range(i * 500, i * 500 + 64)))
        sched.add(Sequence(prompt + [i], block_size=BLOCK_SIZE, max_new_tokens=32))


def report(name, stats, blocks):
    total = stats.tokens_prefilled + stats.tokens_decoded
    print(f"{name}")
    print(f"  steps            {stats.steps:>6}   ({stats.prefill_steps} prefill, {stats.decode_steps} decode)")
    print(f"  tokens computed  {total:>6}")
    print(f"  reused from cache{stats.tokens_reused_from_cache:>6}")
    print(f"  preemptions      {stats.preemptions:>6}")
    print(f"  finished         {stats.finished:>6}")
    print(f"  blocks free/total{blocks.num_free:>4}/{blocks.num_total}\n")


roomy = make_scheduler(num_blocks=4096)
workload(roomy, share_prefix=False)
report("24 distinct prompts, roomy cache", drain(roomy), roomy.blocks)

shared = make_scheduler(num_blocks=4096)
workload(shared, share_prefix=True)
report("24 prompts sharing a system prompt", drain(shared), shared.blocks)

tight = make_scheduler(num_blocks=40)
workload(tight, share_prefix=False)
report("24 distinct prompts, cache under pressure", drain(tight), tight.blocks)
