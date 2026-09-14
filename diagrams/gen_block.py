from elements import *

text(130, 110, 700, "THE LIFE OF ONE KV CACHE BLOCK", 13, GRAY, "left")

ellipse(140, 194, 28, 28, "#334155", "#334155")
box(200, 170, 190, 76, ["FREE", "ref = 0 · no hash", "on the free list"])
box(480, 170, 190, 76, ["OWNED", "ref = 1 · filling", "not yet shareable"])
box(760, 170, 190, 76, ["SHAREABLE", "ref ≥ 1 · hash set"], ACCENT, "#ffffff", 3)
box(1040, 170, 190, 76, ["SHARED", "ref ≥ 2", "n sequences, one copy"], GREEN, GREEN_BG, 2, GREEN_TX)
box(760, 390, 190, 90, ["CACHED", "ref = 0 · hash kept", "evictable but reusable"])
box(1040, 397, 190, 76, ["CPU tier", "host memory"], GRAY, "#f1f5f9", 1.5, "#475569")

text(600, 428, 150, "the prefix cache", 13, ACCENT, "right")

arrow([[170, 208], [198, 208]])
arrow([[392, 208], [478, 208]]);  text(394, 180, 84, "ref 0→1", 12, SLATE)
arrow([[672, 208], [758, 208]]);  text(674, 180, 84, "hash set", 12, SLATE)

arrow([[952, 190], [1038, 190]], GREEN, 2);  text(954, 162, 84, "ref+1", 12, GREEN_TX)
arrow([[1038, 226], [952, 226]], GREEN, 2);  text(954, 232, 84, "ref−1", 12, GREEN_TX)

arrow([[830, 248], [830, 388]]);  text(740, 300, 84, "ref → 0", 12, ACCENT, "right")
arrow([[890, 388], [890, 248]]);  text(896, 300, 110, "prefix hit", 12, ACCENT, "left")

arrow([[855, 482], [855, 520], [300, 520], [300, 248]], SLATE, 1.5, dashed=True)
text(420, 494, 240, "claimed for new tokens · hash forgotten", 12, SLATE)

arrow([[952, 417], [1038, 417]], SLATE, 1.5, dashed=True)
text(954, 392, 84, "offload", 12, SLATE)
arrow([[1038, 453], [952, 453]], SLATE, 1.5, dashed=True)
text(954, 458, 84, "restore", 12, SLATE)

notes = [
    "ref = 0 does not mean gone. A freed block keeps its hash and stays reusable until something claims it —",
    "so the free list IS the prefix cache, and eviction is just claiming the oldest free block and forgetting its hash.",
    "The hash is chained:  h(block) = H( h(parent) ‖ tokens ),  so a hit means every token from position zero matches.",
    "Token ids are still compared on a hit, because hashes collide.",
    "A partial block released before it is ever hashed goes straight back to FREE.",
]
for i, s in enumerate(notes):
    text(130, 578 + i * 24, 1100, s, 13, SLATE, "left")

line([[130, 716], [1290, 716]])
text(130, 728, 1160,
     "accent = the reuse cycle   ·   green = memory actually shared   ·   "
     "slate dashed = eviction and the offload tier", 13, GRAY)
text(130, 758, 1160, "Figure 1-3   The life of one KV cache block", 14, GRAY)

print(save("fig-1-3-block.json"), "elements")
