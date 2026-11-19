# %% [markdown]
# # Lesson 05 — Open the box: what's inside, and where do the parameters live?
#
# The whole model, in one picture:
#
#     token ids
#        │  embed_tokens (lookup table)
#        ▼
#     x  ═══════════════════════════════  ← "residual stream": one 576-vector per token
#        │   ┌── layer 0 ─────────────────┐
#        ├──►│ norm → ATTENTION (mix info │──(+)── add result back into x
#        │   │         between tokens)    │
#        ├──►│ norm → MLP (process each   │──(+)── add result back into x
#        │   │         token on its own)  │
#        │   └────────────────────────────┘
#        │   ... the same block 30 times ...
#        ▼
#     final norm → lm_head (576 → 49152 scores) → softmax → next-token probs
#
# Key idea: **layers don't replace x, they add to it.** Each layer reads the
# running vector, computes a small update, and adds it on. That "+" is why
# deep transformers train at all, and why lesson 06 (logit lens) works.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import matplotlib.pyplot as plt

tok, model, device = load()
cfg = model.config

# %%
banner("The model as PyTorch prints it")
print(model)

# %%
banner("Config — the handful of numbers that define the architecture")
for k in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim", "intermediate_size", "max_position_embeddings",
          "tie_word_embeddings"]:
    print(f"  {k:26} {getattr(cfg, k)}")

# %% [markdown]
# ## Where do the 135M parameters live?

# %%
banner("Parameter budget")
groups = {"embedding (shared with lm_head)": 0, "attention": 0, "MLP": 0, "norms": 0}
for name, p in model.named_parameters():
    if "embed" in name:
        groups["embedding (shared with lm_head)"] += p.numel()
    elif "self_attn" in name:
        groups["attention"] += p.numel()
    elif "mlp" in name:
        groups["MLP"] += p.numel()
    else:
        groups["norms"] += p.numel()
total = sum(groups.values())
for k, v in groups.items():
    print(f"  {k:34} {v/1e6:7.2f}M  {v/total:6.1%}")
print(f"  {'TOTAL':34} {total/1e6:7.2f}M")
print("\n  lm_head.weight is embed_tokens.weight?",
      model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())

fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(list(groups), [v / 1e6 for v in groups.values()], color="#4C72B0")
ax.set_xlabel("millions of parameters")
ax.set_title(f"{MODEL_ID}")
show_fig(fig, "05_param_budget")

# %% [markdown]
# Small models spend a big share on the embedding table. In an 8B model the
# layers dominate and the embedding is a few percent.

# %%
banner("One layer, up close (shapes of its weight matrices)")
for name, p in model.model.layers[0].named_parameters():
    print(f"  {name:32} {tuple(p.shape)}")

# %% [markdown]
# Notice `k_proj`/`v_proj` are 576→192 but `q_proj` is 576→576.
# 9 query heads share 3 key/value heads (**grouped-query attention**). That
# saves memory in the KV cache (lesson 10).
#
# ## Watch the residual stream flow
# `output_hidden_states=True` gives us x **before layer 0, after layer 0,
# ..., after layer 29** — 31 snapshots. Let's watch how big the vector gets
# and how much each layer changes it.

# %%
banner("Residual stream through the layers")
ids = encode(tok, "The Eiffel Tower is in the city of", device)
hs = model(ids, output_hidden_states=True).hidden_states
print(f"  got {len(hs)} snapshots, each of shape {tuple(hs[0].shape)}")
# NOTE: HF applies the final norm to the very last snapshot, so we compare 0..29.
norms = [h[0, -1].norm().item() for h in hs[:-1]]
deltas = [(hs[i + 1][0, -1] - hs[i][0, -1]).norm().item() for i in range(len(hs) - 2)]

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(norms, marker="o"); ax[0].set_title("size of x (last token) going into each layer")
ax[0].set_xlabel("layer")
ax[1].bar(range(len(deltas)), deltas, color="#DD8452"); ax[1].set_title("how much each layer changed x")
ax[1].set_xlabel("layer")
show_fig(fig, "05_residual_stream")
print("  biggest edits made by layers:", sorted(range(len(deltas)), key=lambda i: -deltas[i])[:5])

# %% [markdown]
# **Takeaway:** a transformer = lookup table + 30 identical "read, compute,
# add back" blocks + a scoring layer. The interesting question is *what* each
# block adds — next lessons look at that.
