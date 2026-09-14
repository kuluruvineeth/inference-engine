from elements import *

text(170, 20, 700, "ONE CALL TO Scheduler.schedule()", 13, GRAY, "left")

box(310, 50, 240, 54, ["schedule()"])

zone(170, 135, 530, 520, "PREFILL  ·  drain the waiting queue")
box(310, 180, 240, 58, ["waiting queue empty?"], ACCENT, "#ffffff", 2)
box(310, 278, 240, 58, ["blocks available for it?"], ACCENT, "#ffffff", 2)
box(310, 376, 240, 58, ["fits the token budget?"], ACCENT, "#ffffff", 2)
box(310, 474, 240, 58, ["batch already non-empty?"], ACCENT, "#ffffff", 2)
box(310, 572, 240, 62, ["allocate · reuse prefix",
                        "schedule min(remaining, budget)"])

box(310, 700, 240, 58, ["prefill batch non-empty?"], ACCENT, "#ffffff", 2)
box(310, 810, 240, 60, ["PREFILL BATCH", "returned to Engine"], GREEN, GREEN_BG, 2, GREEN_TX)

zone(780, 340, 600, 380, "DECODE  ·  one slot each")
box(880, 420, 240, 58, ["can_append a slot?"], ACCENT, "#ffffff", 2)
box(1160, 420, 200, 58, ["preempt newest", "free blocks · requeue"], RED, RED_BG, 2, RED_TX)
box(880, 630, 240, 58, ["append_slot  ·  1 token"])
box(880, 800, 240, 60, ["DECODE BATCH", "returned to Engine"], GREEN, GREEN_BG, 2, GREEN_TX)

# prefill spine
arrow([[430, 106], [430, 178]])
arrow([[430, 240], [430, 276]]);  text(440, 246, 50, "no", 12, SLATE, "left")
arrow([[430, 338], [430, 374]]);  text(440, 344, 50, "yes", 12, SLATE, "left")
arrow([[430, 436], [430, 472]]);  text(440, 442, 50, "no", 12, SLATE, "left")
arrow([[430, 534], [430, 570]]);  text(440, 540, 120, "no · chunk it", 12, SLATE, "left")
arrow([[308, 405], [265, 405], [265, 603], [308, 603]])
text(258, 380, 48, "yes", 12, SLATE, "right")
arrow([[308, 603], [210, 603], [210, 209], [308, 209]])
text(212, 582, 50, "next", 12, SLATE, "left")

# every break leaves the loop and asks whether anything was admitted
arrow([[552, 209], [720, 209], [720, 680], [430, 680], [430, 698]], SLATE, 1.5, dashed=True)
text(560, 185, 158, "yes · nothing waiting", 12, SLATE, "left")
line([[552, 307], [720, 307]], SLATE, 1.5, dashed=True)
text(560, 283, 158, "no · out of blocks", 12, SLATE, "left")
line([[552, 503], [720, 503]], SLATE, 1.5, dashed=True)
text(560, 479, 158, "yes · stop admitting", 12, SLATE, "left")

arrow([[430, 760], [430, 808]], GREEN, 2);  text(440, 770, 50, "yes", 12, GREEN, "left")
arrow([[552, 729], [760, 729], [760, 449], [878, 449]], SLATE, 1.5, dashed=True)
text(566, 702, 190, "no · nothing to prefill", 12, SLATE, "left")

# decode spine
arrow([[1000, 480], [1000, 628]]);  text(1010, 495, 50, "yes", 12, SLATE, "left")
arrow([[1122, 449], [1156, 449]], RED, 2, dashed=True)
text(1124, 422, 34, "no", 12, RED_TX, "center")
arrow([[1260, 418], [1260, 388], [1000, 388], [1000, 416]], RED, 2, dashed=True)
text(1100, 360, 150, "retry", 12, RED_TX, "left")
arrow([[1000, 690], [1000, 798]], GREEN, 2)

# what the ladder teaches
text(170, 900, 700, "Prefill has strict priority — decode runs only when nothing can be admitted.", 13, SLATE, "left")
text(170, 924, 700, "A prompt larger than the budget is split across steps, so it never stalls the decodes behind it.", 13, SLATE, "left")
text(170, 948, 700, "Preemption frees the newest running sequence and recomputes it later, rather than swapping it out.", 13, SLATE, "left")
text(170, 972, 700, "Both queues empty → the batch is empty and the engine idles.", 13, SLATE, "left")

line([[170, 1010], [1380, 1010]])
text(170, 1022, 1210,
     "accent = the admission path   ·   slate dashed = a break leaving the loop   ·   "
     "red dashed = preemption   ·   green = a batch is returned", 13, GRAY)
text(170, 1052, 1210, "Figure 1-2   How one scheduler step decides what to run", 14, GRAY)

out = "fig-1-2-scheduler.json"

print(save("fig-1-2-scheduler.json"), "elements")
