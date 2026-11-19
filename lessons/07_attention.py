# %% [markdown]
# # Lesson 07 — What is attention actually doing? Find it in a real model
#
# Attention is the only place where tokens **talk to each other**. Everything
# else (MLP, norms) works on each token separately.
#
# The idea in one breath: each token makes a **query** ("what am I looking
# for?"), every token offers a **key** ("what do I contain?") and a **value**
# ("what I'll hand over"). Scores = query·key, softmax over earlier tokens,
# and the output is the weighted average of values.
#
#     weights = softmax(Q · Kᵀ / √d)       ← who to look at  (a row per token, sums to 1)
#     output  = weights · V                ← what you get from them
#
# Tokens may only look *backwards* (causal mask) — otherwise predicting the
# next token would be cheating.
#
# Instead of more theory, we'll pull out the real `weights` and *find heads
# with recognisable jobs*.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import matplotlib.pyplot as plt

tok, model, device = load()
n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

def attentions(ids):
    """List of n_layers tensors, each (heads, seq, seq): row = query token, col = token looked at."""
    return [a[0] for a in model(ids, output_attentions=True).attentions]

# %%
banner("Shapes")
sentence = "The cat chased the mouse because the cat was hungry"
ids = encode(tok, sentence, device)
labels = [t.replace("Ġ", "·") for t in tok.convert_ids_to_tokens(ids[0])]
att = attentions(ids)
print(f"  {len(att)} layers, each {tuple(att[0].shape)} = (heads, query pos, key pos)")
print("  each row sums to 1?", torch.allclose(att[5].sum(-1), torch.ones(1, device=device)))
print("  upper triangle (future) all zero?", att[5].triu(1).abs().max().item() == 0)

# %% [markdown]
# ## 1. Hunting for heads with a job
#
# **Previous-token heads**: always look one step back. Simple, but they're
# the building block for the next kind.
#
# **Induction heads**: the famous copy mechanism. If the text so far contains
# `... A B ... A`, an induction head at the second `A` looks at the old `B`
# ("last time I saw A, B came next") so the model can predict `B`.
# This is a big part of how *in-context learning* works.
#
# Test: a sequence of **random** tokens, repeated twice. The model can't
# predict the first copy at all, but can predict the second copy perfectly —
# *only* if it has induction heads.

# %%
torch.manual_seed(0)
L = 50
rand = torch.randint(1000, 20000, (1, L), device=device)
rep = torch.cat([rand, rand], dim=1)                      # A B C ... A B C ...

logits = model(rep).logits[0]
logp = torch.log_softmax(logits[:-1], -1)
loss = -logp[torch.arange(2 * L - 1), rep[0, 1:]]
banner("Can the model copy a random sequence it saw once?")
print(f"  loss on 1st copy: {loss[:L-1].mean():.2f}   (≈ guessing blindly)")
print(f"  loss on 2nd copy: {loss[L:].mean():.2f}   (≈ copying!)")

att = attentions(rep)
q = torch.arange(L, 2 * L, device=device)                 # query positions in 2nd copy
induction = torch.stack([a[:, q, q - L + 1].mean(-1) for a in att]).cpu()   # (layers, heads)
prev_tok = torch.stack([a[:, q, q - 1].mean(-1) for a in att]).cpu()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, data, title in [(ax[0], prev_tok, "previous-token score"), (ax[1], induction, "induction score")]:
    im = a.imshow(data, aspect="auto", cmap="magma", vmin=0, vmax=1)
    a.set_xlabel("head"); a.set_ylabel("layer"); a.set_title(title)
    fig.colorbar(im, ax=a)
show_fig(fig, "07_head_scores")

def top_heads(scores, k=5):
    flat = scores.flatten().topk(k)
    return [(i // n_heads, i % n_heads, v) for v, i in zip(flat.values.tolist(), flat.indices.tolist())]

print("\n  strongest previous-token heads (layer, head, score):")
for l, h, v in top_heads(prev_tok): print(f"    L{l:<2} H{h}  {v:.2f}")
print("  strongest induction heads (layer, head, score):")
for l, h, v in top_heads(induction): print(f"    L{l:<2} H{h}  {v:.2f}")

# %% [markdown]
# ## 2. See the patterns with your own eyes
# Diagonal stripe one step left of the main diagonal = previous-token head.
# Stripe offset by L-1 in the repeated sequence = induction head.

# %%
(pl, ph, _), (il, ih, _) = top_heads(prev_tok, 1)[0], top_heads(induction, 1)[0]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
ax[0].imshow(att[pl][ph].cpu(), cmap="Blues"); ax[0].set_title(f"previous-token head L{pl} H{ph}")
ax[1].imshow(att[il][ih].cpu(), cmap="Blues"); ax[1].set_title(f"induction head L{il} H{ih}")
for a in ax:
    a.set_xlabel("looked-at position (key)"); a.set_ylabel("current position (query)")
show_fig(fig, "07_patterns_repeated")

# %% [markdown]
# ## 3. Attention on a real sentence
# Look for the "attention sink": many heads dump most of their weight on the
# very first token. That's a learned "nothing to do here" parking spot, not a bug.

# %%
att = attentions(ids)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for a, (l, h) in zip(axes, [(pl, ph), (il, ih), (n_layers // 2, 0)]):
    a.imshow(att[l][h].cpu(), cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    a.set_yticks(range(len(labels)), labels, fontsize=8)
    a.set_title(f"layer {l}, head {h}")
show_fig(fig, "07_sentence")

sink = torch.stack([a[:, 1:, 0].mean(-1) for a in att]).mean().item()
print(f"  average attention paid to the first token (all layers/heads): {sink:.0%}")

banner("Where does the 2nd 'cat' look (average over all heads, last 10 layers)?")
second_cat = len(labels) - 3
w = torch.stack([a[:, second_cat] for a in att[-10:]]).mean((0, 1))
for i in w.argsort(descending=True)[:5].tolist():
    print(f"  {labels[i]:10} {w[i]:.1%}")

# %% [markdown]
# **Takeaway:** attention is a learned "where should I copy information from?"
# Individual heads specialise: some look one step back, some copy patterns,
# some idle on the first token. Lesson 08 *turns these heads off* to prove
# they matter.
#
# **Try:** set `L = 20` or `L = 200`. Or replace `rand` with real text repeated twice.
