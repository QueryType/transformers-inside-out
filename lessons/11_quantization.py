# %% [markdown]
# # Lesson 11 — Quantization: same model, fewer bits
#
# Lesson 08 ended with a fact: decoding is **memory-bound**. To make each new
# token, the computer has to read *every weight* from RAM. Fewer bytes per
# weight means less to read, so generation is faster, and a bigger model fits.
#
# That's all quantization is: **store each weight with fewer bits** (8, 4, even 2)
# instead of 16 or 32, and accept a small loss in accuracy.
#
# In this lesson you quantize SmolLM2 by hand (about ten lines of code) and
# measure how much it gets worse at each bit width.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import matplotlib.pyplot as plt

tok, model, device = load()
ORIGINAL = {k: v.clone() for k, v in model.state_dict().items()}   # keep a copy to restore

# %% [markdown]
# ## Why bother? The memory bill
#
#     model size = number of weights × bytes per weight

# %%
banner("Model size at different precisions")
n_params = sum(p.numel() for p in model.parameters())
sizes = {"SmolLM2-135M (this)": n_params, "Llama-3.1-8B": 8.0e9, "Qwen2.5-32B": 32.5e9, "Llama-3.1-70B": 70.6e9}
print(f"  {'model':22} {'fp32':>9} {'bf16':>9} {'8-bit':>9} {'4-bit':>9}")
for name, n in sizes.items():
    print(f"  {name:22}" + "".join(f"{n * b / 1e9:8.2f}G" for b in (4, 2, 1, 0.5)))
print("\n  A 70B model needs 141 GB in bf16, but only ~35 GB in 4-bit: that's the difference")
print("  between 'needs a server' and 'runs on a 64 GB Mac'.")

# %% [markdown]
# ## Quantize 8 numbers by hand
#
# The simplest recipe (**absmax**, symmetric):
#
#     scale = max(|w|) / 127            # biggest weight maps to ±127
#     q     = round(w / scale)          # small integers, stored as int8
#     w'    = q * scale                 # "dequantize" when you need the number back
#
# You store the int8 values **plus one scale**. The rounding is where the error comes from.

# %%
banner("Absmax int8 on the first 8 weights of one row")
w = ORIGINAL["model.layers.0.mlp.down_proj.weight"][0, :8].cpu()
scale = w.abs().max() / 127
q = torch.round(w / scale)
print(f"  original    : {[round(x, 5) for x in w.tolist()]}")
print(f"  scale       : {scale.item():.6f}")
print(f"  stored int8 : {[int(x) for x in q.tolist()]}")
print(f"  dequantized : {[round(x, 5) for x in (q * scale).tolist()]}")
print(f"  max error   : {(q * scale - w).abs().max().item():.2e}")

# %% [markdown]
# ## The catch: outliers
#
# One scale serves the whole group. If a single weight is huge, the scale is
# huge, and all the *normal* weights get squeezed into just a few integer
# levels. Real LLM weights do have outliers, so let's look.

# %%
banner("How spread out are the weights? (one MLP matrix)")
W = ORIGINAL["model.layers.10.mlp.down_proj.weight"].float().cpu()
a = W.abs().flatten()
print(f"  {W.numel():,} weights.  typical |w| (median): {a.median():.4f}   biggest: {a.max():.4f}")
print(f"  the biggest is {a.max() / a.median():.0f}x the typical one")

def fake_quant(w, bits, group=None):
    """Quantize and immediately dequantize. group=None → one scale for the whole tensor."""
    qmax = 2 ** (bits - 1) - 1                            # 8 bits → 127, 4 bits → 7
    g = w.reshape(1, -1) if group is None else w.reshape(-1, group)
    scale = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(g / scale).clamp(-qmax - 1, qmax)     # the integers you'd store
    return (q * scale).reshape(w.shape)

def quant_error(w, bits, group):
    """Fake-quantize w with one scale per `group` weights; return relative error."""
    return ((fake_quant(w, bits, group) - w).norm() / w.norm()).item()

print(f"\n  relative error when quantizing this matrix to 4 bits:")
for label, group in [("1 scale for whole matrix", None), ("1 scale per row (576)", W.shape[1]),
                     ("1 scale per 32 weights", 32)]:
    print(f"    {label:26} {quant_error(W, 4, group):7.2%}")

# %% [markdown]
# **Smaller groups = smaller error.** Each group of 32 gets its own scale, so
# an outlier only hurts its 31 neighbours. The cost: one extra 16-bit scale
# per 32 weights = 0.5 bits per weight. So "4-bit with group 32" is really
# **4.5 bits per weight**. Remember that number for lesson 12.

# %% [markdown]
# ## Quantize the whole model and measure the damage
#
# We "fake-quantize" every big matrix: round it to N bits and put the rounded
# numbers back as floats. Real kernels keep the integers and compute on them
# directly, but the *accuracy* is identical, and this way it's one loop.
#
# Two measurements:
# * **perplexity** on a paragraph (lesson 02's loss, as `exp(loss)`; lower = better)
# * **top-1 agreement**: how often the quantized model picks the same next token as the original

# %%
TEXT = ("The lighthouse stood at the edge of the cliff, its white paint peeling after decades "
        "of salt and wind. Every night the keeper climbed the spiral stairs, trimmed the wick, "
        "and watched the beam sweep across the dark water. Ships that passed in the night never "
        "knew his name, but they trusted the light. In 1952 the lamp was replaced by an electric "
        "one, and the keeper was told he was no longer needed. He stayed anyway, living in the "
        "small stone cottage below the tower until the end of his life.")
ids = encode(tok, TEXT, device)
ref_logits = model(ids).logits[0]

def measure():
    logits = model(ids).logits[0]
    loss = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:])
    agree = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean()
    return loss.exp().item(), agree.item()

def quantize_model(bits, group):
    sd = model.state_dict()
    for name, w in ORIGINAL.items():
        if w.ndim == 2:                                    # matrices only; norms stay as-is
            g = None if group is None else (w.shape[1] if group == "row" else group)
            sd[name].copy_(fake_quant(w, bits, g))

def restore():
    model.load_state_dict(ORIGINAL)

def continuation(prompt="The Eiffel Tower is located in the city of", n=12):
    out = model.generate(**tok(prompt, return_tensors="pt").to(device), max_new_tokens=n,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(tok(prompt).input_ids):]).replace("\n", " ")

banner("Every matrix quantized, group size 32")
base_ppl, _ = measure()
print(f"  {'bits':>6} {'perplexity':>11} {'agree':>7}   continuation of 'The Eiffel Tower is located in the city of'")
base_text = continuation()
print(f"  {'32':>6} {base_ppl:11.2f} {1:7.0%}   {base_text!r}")
results, texts = {}, {}
for bits in (8, 6, 5, 4, 3, 2):
    quantize_model(bits, 32)
    ppl, agree = measure()
    results[bits], texts[bits] = (ppl, agree), continuation()
    print(f"  {bits:>6} {ppl:11.2f} {agree:7.0%}   {texts[bits]!r}")
    restore()

# %% [markdown]
# Read that table top to bottom: **8 bits is almost free** (perplexity 19.9 → 20.0),
# 6 and 5 bits cost a little, 4 bits is a real but usable hit (still says Paris),
# then 3 bits falls off a cliff. That cliff is why "Q4" is
# the popular sweet spot. (Tiny models like this one suffer *more* than big
# ones; a 70B model at 4 bits is usually hard to tell apart from bf16.)

# %%
banner("Same thing at 4 bits, with different group sizes")
group_results = {}
for label, group in [("per tensor", None), ("per row", "row"), ("group 128", 128),
                     ("group 64", 64), ("group 32", 32), ("group 16", 16)]:
    quantize_model(4, group)
    ppl, agree = measure()
    group_results[label] = (ppl, agree)
    print(f"  {label:12} perplexity {ppl:9.2f}   agree {agree:5.0%}")
    restore()

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
b = list(results)
ax.plot(b, [results[k][0] for k in b], "o-", label="quantized (group 32)")
ax.axhline(base_ppl, ls="--", c="gray", label="original fp32")
ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xlabel("bits per weight"); ax.set_ylabel("perplexity (log scale)")
ax.set_title("Fewer bits: fine, fine, fine... cliff"); ax.legend()
show_fig(fig, "11_bits_vs_perplexity")

# %% [markdown]
# ## Which parts are fragile?
#
# Real quantizers don't treat every tensor the same. Let's push *one kind* of
# matrix down to 3 bits and leave everything else alone.

# %%
banner("Only one kind of matrix at 3 bits (group 32), rest untouched")
kinds = {"embeddings (also the output head)": "embed_tokens", "attention q/k/v/o": "self_attn",
         "MLP gate/up": ("gate_proj", "up_proj"), "MLP down": "down_proj"}
fragile = {}
for label, key in kinds.items():
    keys = (key,) if isinstance(key, str) else key
    sd = model.state_dict()
    share = 0
    for name, w in ORIGINAL.items():
        if w.ndim == 2 and any(k in name for k in keys):
            sd[name].copy_(fake_quant(w, 3, 32)); share += w.numel()
    ppl, agree = measure()
    fragile[label] = (share / n_params, ppl, agree)
    print(f"  {label:36} ({share / n_params:4.0%} of weights)  perplexity {ppl:8.2f}  agree {agree:4.0%}")
    restore()

# %% [markdown]
# Embeddings and `down_proj` are the most fragile; attention is the most
# forgiving. This is the idea behind "mixed" formats like GGUF's **Q4_K_M**: most matrices
# get 4 bits, but the fragile ones get 6. You'll see that inside a real file
# in lesson 12.
#
# ## Storing it for real: two 4-bit numbers per byte
#
# Fake quantization kept floats around. A real file packs the integers:

# %%
banner("Packing a group of 32 weights at 4 bits")
g = ORIGINAL["model.layers.0.mlp.down_proj.weight"][0, :32].float().cpu()
scale = g.abs().max() / 7
q = torch.round(g / scale).clamp(-8, 7).to(torch.int8)            # 32 values in -8..7
u = (q + 8).to(torch.uint8)                                       # shift to 0..15
packed = u[:16] | (u[16:] << 4)          # byte j = weight j (low nibble) + weight j+16 (high)
print(f"  32 weights: {32 * 2} bytes in bf16")
print(f"  4-bit     : {packed.numel()} bytes of nibbles + 2 bytes of fp16 scale = {packed.numel() + 2} bytes")
print(f"  => {(packed.numel() + 2) * 8 / 32} bits per weight")
lo, hi = (packed & 0x0F).to(torch.int8) - 8, (packed >> 4).to(torch.int8) - 8
unpacked = torch.cat([lo, hi])
print(f"  unpacks to the same integers? {torch.equal(unpacked, q)}")

# %% [markdown]
# **Takeaway:** quantization = small integers + a scale per small group.
# 8 bits is basically lossless, 4 bits is the sweet spot, and outliers are
# the enemy (which is why groups exist and why some layers get more bits).
# The 18-byte block you just built (2-byte scale + 16 bytes of nibbles) is
# *exactly* llama.cpp's `Q4_0` format.
#
# **Try:**
# * Rerun with `MODEL_ID=Qwen/Qwen2.5-0.5B`. Does a bigger model handle 4 bits better?
# * Change `fake_quant` to be **asymmetric**: store a min and a max per group
#   (`scale = (max-min)/15`, `q = round((w-min)/scale)`). Does 3-bit improve?
#   (That's `Q4_1` vs `Q4_0`.)
# * Skip quantizing `embed_tokens` and rerun the 3-bit row. How much comes back?
