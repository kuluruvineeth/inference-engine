"""Prefix caching, measured — the claim from the paper, on real numbers."""
from ie.core.block_manager import BlockManager
from ie.core.sequence import Sequence

BS = 16
SYSTEM = list(range(1000, 1128))          # a 128-token system prompt
bm = BlockManager(num_blocks=512, block_size=BS)

live = []
for i in range(20):                        # 20 chats sharing that prompt
    s = Sequence(SYSTEM + [2000 + i] * 24, block_size=BS)
    bm.allocate(s)
    s.num_computed = len(s)
    bm.publish(s)
    live.append(s)

st = bm.stats()
naive = sum((len(s) + BS - 1)//BS for s in live)
actual = bm.num_total - bm.num_free
print(f"20 chats, each = 128-token shared system prompt + 24 unique tokens\n")
print(f"  blocks without sharing : {naive}")
print(f"  blocks actually used   : {actual}")
print(f"  saved                  : {naive - actual}  ({100*(naive-actual)/naive:.0f}%)")
print(f"  prefix hit rate        : {st['hit_rate']:.1%}")
print(f"  pool utilization       : {st['utilization']:.1%}")
