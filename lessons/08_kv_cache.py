# %% [markdown]
# # Lesson 08 — Why is generation fast? The KV cache
#
# Our loop in lesson 03 re-ran the *whole* text through the model for every
# new token. Wasteful: the keys and values (lesson 07) of old tokens never
# change, because tokens can only look backwards. So: compute them once,
# **cache** them, and each step feed in only the *one* new token.
#
# This gives two phases you'll see in every inference server (llama.cpp, vLLM, MLX):
#   * **prefill** — process the whole prompt at once (fast, parallel)
#   * **decode**  — one token at a time, reading the cache (memory-bound)

# %%
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *

tok, model, device = load()
cfg = model.config

def sync():
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()

# %%
def gen_no_cache(ids, n):
    for _ in range(n):
        nxt = model(ids, use_cache=False).logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, nxt], 1)
    return ids

def gen_with_cache(ids, n):
    out = model(ids, use_cache=True)                          # PREFILL: whole prompt
    past, nxt = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)
    result = [ids, nxt]
    for _ in range(n - 1):                                    # DECODE: 1 token per step
        out = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)
        result.append(nxt)
    return torch.cat(result, 1)

def timed(fn, *a):
    sync(); t = time.perf_counter(); r = fn(*a); sync()
    return r, time.perf_counter() - t

# %%
banner("Same output, very different speed")
prompt = ("Here is a long story about a lighthouse keeper. " * 40)
ids = encode(tok, prompt, device)
N = 100
gen_with_cache(ids, 5)  # warm-up
a, t_no = timed(gen_no_cache, ids, N)
b, t_yes = timed(gen_with_cache, ids, N)
print(f"  prompt length: {ids.shape[1]} tokens, generating {N}")
print(f"  without cache: {t_no:6.2f}s  ({N/t_no:6.1f} tok/s)")
print(f"  with cache   : {t_yes:6.2f}s  ({N/t_yes:6.1f} tok/s)   -> {t_no/t_yes:.1f}x faster")
print(f"  identical tokens? {torch.equal(a, b)}")

# %% [markdown]
# ## What's in the cache?

# %%
banner("Cache contents")
out = model(encode(tok, "Hello there, how are you", device), use_cache=True)
cache = out.past_key_values
k0 = cache.layers[0].keys
print(f"  one tensor of keys per layer, shape {tuple(k0.shape)} = (batch, kv_heads, tokens, head_dim)")
print(f"  layers cached: {len(cache.layers)} (keys + values each)")

# %% [markdown]
# ## How big does it get? (this is why long context needs so much RAM)
#
#     cache bytes = 2 (K and V) × layers × kv_heads × head_dim × tokens × bytes_per_number
#
# Note it uses **kv_heads**, not query heads. Grouped-query attention (9 query
# heads sharing 3 KV heads here) shrinks the cache 3×. Big models do the same.

# %%
banner("KV cache size at 32k tokens of context (bf16 = 2 bytes)")
models = {
    # name                  layers, q_heads, kv_heads, head_dim
    "SmolLM2-135M (this)": (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim),
    "Llama-3.1-8B":        (32, 32, 8, 128),
    "Qwen2.5-7B":          (28, 28, 4, 128),
    "Llama-3.1-70B":       (80, 64, 8, 128),
}
ctx = 32_768
print(f"  {'model':22} {'with GQA':>10} {'if no GQA':>10}")
for name, (layers, qh, kvh, hd) in models.items():
    gqa = 2 * layers * kvh * hd * ctx * 2 / 1e9
    full = 2 * layers * qh * hd * ctx * 2 / 1e9
    print(f"  {name:22} {gqa:8.2f}GB {full:8.2f}GB")

# %% [markdown]
# **Takeaway:** the cache trades memory for speed. Model weights are fixed
# size; the cache grows with every token of context and every parallel user.
# That's the number you're really tuning when you set context size or
# parallel slots in llama-server / oMLX.
#
# **Try:** vary prompt length (×10, ×80) — the no-cache version gets slower
# with length, the cached one barely notices.
