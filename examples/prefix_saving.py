from ie.core.block_manager import BlockManager
from ie.core.sequence import Sequence

BLOCK_SIZE = 16
SYSTEM_PROMPT = list(range(1000, 1128))
NUM_CHATS = 20
UNIQUE_TOKENS = 24

manager = BlockManager(num_blocks=512, block_size=BLOCK_SIZE)
chats = []

for i in range(NUM_CHATS):
    chat = Sequence(SYSTEM_PROMPT + [2000 + i] * UNIQUE_TOKENS, block_size=BLOCK_SIZE)
    manager.allocate(chat)
    chat.num_computed = len(chat)
    manager.share_computed_blocks(chat)
    chats.append(chat)

without_sharing = sum(chat.num_blocks for chat in chats)
with_sharing = manager.num_total - manager.num_free
saved = without_sharing - with_sharing

print(f"{NUM_CHATS} chats, each = {len(SYSTEM_PROMPT)}-token shared prompt + {UNIQUE_TOKENS} unique\n")
print(f"  blocks without sharing : {without_sharing}")
print(f"  blocks with sharing    : {with_sharing}")
print(f"  saved                  : {saved}  ({100 * saved / without_sharing:.0f}%)")
print(f"  block reuse rate       : {manager.reuse_rate:.1%}")
