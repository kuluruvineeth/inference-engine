from elements import *

SLATE, ACCENT, GRAY = "#64748b", "#2563eb", "#94a3b8"

# deployment boundary
text(250, 58, 700, "ENGINE PROCESS  ·  ONE PYTHON PROCESS  ·  ONE GPU", 13, GRAY, "left")
rect(250, 90, 1190, 690, GRAY, "transparent", 2, dashed=True)
line([[815, 112], [815, 762]], "#cbd5e1", 1.5, dashed=True)
text(275, 108, 420, "CONTROL PLANE  ·  decides what runs next", 13, GRAY, "left")
text(840, 108, 420, "DATA PLANE  ·  runs the forward pass", 13, GRAY, "left")

# external actor
box(60, 155, 165, 58, ["Client", "prompt · max tokens"], GRAY, "#f1f5f9", 1.5, "#475569")

# control plane
box(300, 155, 250, 58, ["TextEngine", "tokenize · detokenize · stream"])
box(300, 265, 250, 95, ["Engine", "step()  ·  owns everything below",
                        "ie/engine/engine.py"], ACCENT, "#ffffff", 3)
box(300, 425, 250, 62, ["Scheduler", "continuous batching · chunked prefill"])
box(300, 550, 250, 62, ["BlockManager", "paged alloc · prefix reuse · refcount"])
box(620, 278, 175, 70, ["BatchLayout", "block tables · slots", "cu_seqlens · positions"])

# block pool datastore
rect(332, 677, 186, 78, "#475569", "#f1f5f9")
ellipse(332, 660, 186, 34)
text(332, 700, 186, "block pool", 14, "#1e293b")
text(332, 722, 186, "free list · chained hash → block", 11, SLATE)

# data plane
box(865, 265, 250, 95, ["TorchTransformer", "RMSNorm · RoPE · GQA · SwiGLU",
                        "24 blocks, fp16"])
box(865, 425, 250, 78, ["attention backend", "flash_attn_varlen · with_kvcache",
                        "fallback: torch SDPA"])
box(1170, 265, 175, 95, ["sampler", "on device", "exponential race"])
box(1170, 425, 175, 78, ["DecodeGraphs", "captured CUDA graphs", "bucketed by batch size"])

# kv cache datastore
rect(875, 567, 230, 90, "#475569", "#f1f5f9")
ellipse(875, 550, 230, 34)
text(875, 592, 230, "KV cache", 14, "#1e293b")
text(875, 614, 230, "2048 blocks × 256 tokens", 11, SLATE)
text(875, 632, 230, "one pair per layer", 11, SLATE)

# the step loop
arrow([[228, 184], [296, 184]])
arrow([[405, 215], [405, 263]]);            text(415, 228, 110, "token ids", 12, SLATE, "left")
arrow([[415, 362], [415, 423]]);            text(428, 380, 130, "1 · schedule()", 13, ACCENT, "left")
arrow([[552, 313], [616, 313]]);            text(520, 240, 128, "2 · layout", 13, ACCENT)
arrow([[797, 313], [861, 313]]);            text(759, 240, 140, "3 · forward", 13, ACCENT)
arrow([[1117, 312], [1166, 312]]);          text(1070, 240, 145, "4 · sample", 13, ACCENT)
arrow([[1347, 312], [1382, 312], [1382, 730], [640, 730], [640, 456], [552, 456]],
      SLATE, 1.5, dashed=True)
text(838, 704, 240, "5 · finish_step(batch, ids)", 12, SLATE)

# ownership and internal reads
arrow([[425, 489], [425, 548]], SLATE, 1.5, dashed=True)
text(438, 505, 150, "allocate · reuse", 12, SLATE, "left")
arrow([[425, 614], [425, 658]], SLATE, 1.5, dashed=True)
arrow([[990, 362], [990, 423]], SLATE, 1.5, dashed=True)
text(1002, 380, 150, "per layer", 12, SLATE, "left")
arrow([[990, 505], [990, 548]], SLATE, 1.5, dashed=True)
text(1002, 515, 160, "read · write blocks", 12, SLATE, "left")
arrow([[1115, 340], [1142, 340], [1142, 464], [1166, 464]], SLATE, 1.5, dashed=True)
text(1152, 392, 190, "decode → replay", 12, SLATE, "left")

# legend + caption
line([[60, 800], [1440, 800]])
text(60, 812, 1380,
     "accent = the step loop   ·   slate dashed = ownership and internal reads   ·   "
     "cylinders = memory the engine manages   ·   weights load from safetensors at startup",
     13, GRAY)
text(60, 842, 1380, "Figure 1-1   Inference engine architecture and the step loop", 14, GRAY)

print(save("fig-1-1-architecture.json"), "elements")
