import numpy as np

from ie.core.block_manager import BlockManager
from ie.core.layout import build_layout
from ie.core.scheduler import Scheduler
from ie.core.sequence import Sequence
from ie.kernels.reference import allocate_kv_cache, dense_attention, paged_attention, write_kv

BLOCK_SIZE = 8
HEADS, KV_HEADS, DIM = 8, 2, 16
PROMPT_LEN = 37
NUM_BLOCKS = 64

generator = np.random.default_rng(0)
q = generator.standard_normal((PROMPT_LEN, HEADS, DIM)).astype(np.float32)
k = generator.standard_normal((PROMPT_LEN, KV_HEADS, DIM)).astype(np.float32)
v = generator.standard_normal((PROMPT_LEN, KV_HEADS, DIM)).astype(np.float32)

manager = BlockManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
scheduler = Scheduler(manager, max_batch_tokens=12)
seq = Sequence(list(range(PROMPT_LEN)), block_size=BLOCK_SIZE)
scheduler.add(seq)

k_cache, v_cache = allocate_kv_cache(NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, DIM)
chunks = []

while not seq.prompt_is_fully_computed:
    batch = scheduler.schedule()
    layout = build_layout(batch, BLOCK_SIZE)
    start, stop = layout.positions[0], layout.positions[-1] + 1
    write_kv(k[start:stop], v[start:stop], k_cache, v_cache, layout.slot_mapping)
    chunks.append(paged_attention(q[start:stop], k_cache, v_cache, layout.block_tables,
                                  layout.context_lens, layout.query_lens, BLOCK_SIZE))
    print(f"  chunk  positions {start:>3}..{stop - 1:<3} "
          f"blocks {layout.block_tables[0][:layout.context_lens[0] // BLOCK_SIZE + 1]}")
    scheduler.finish_step(batch, [0] * len(batch))

paged = np.concatenate(chunks, axis=0)
dense = dense_attention(q, k, v, causal=True)

print(f"\n  prompt of {PROMPT_LEN} tokens, block size {BLOCK_SIZE}, chunked into {len(chunks)} passes")
print(f"  blocks held         {len(seq.block_table)}")
print(f"  max abs difference  {np.abs(paged - dense).max():.2e}")
print(f"  identical to dense  {np.allclose(paged, dense, atol=1e-5)}")
