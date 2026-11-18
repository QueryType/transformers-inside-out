# %% [markdown]
# # Lesson 02 — What actually comes out? A probability for every token
#
# The single most important fact about these models:
#
# > **A language model is a function: text in -> a probability for every
# > possible next token out.** That's it. Everything else is built on top.
#
# We call the model *once* (no `generate`) and look at the raw output.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import matplotlib.pyplot as plt

tok, model, device = load()

# %%
prompt = "The cat sat on the"
ids = encode(tok, prompt, device)
out = model(ids)
logits = out.logits

print("input ids shape :", tuple(ids.shape), "  (batch, tokens)")
print("logits shape    :", tuple(logits.shape), "  (batch, tokens, vocab)")
print("vocab size      :", tok.vocab_size)

# %% [markdown]
# `logits` has one row of 49,152 numbers **for every input position**.
# Row *i* is the model's guess for the token that comes *after* position *i*.
#
# Logits are raw scores. **Softmax** turns them into probabilities that sum to 1:
#
#     p_i = exp(logit_i) / sum_j exp(logit_j)
#
# (that's all the math you need here: big score -> big probability).

# %%
banner(f"Next-token guess after: {prompt!r}")
probs = torch.softmax(logits[0, -1], dim=-1)
print(f"probabilities sum to {probs.sum().item():.4f}\n")
show_top_k(probs, tok, k=10)

# %% [markdown]
# ## The model predicts at *every* position at once
# This is why training is efficient: one forward pass = a guess for every
# position. During generation we only use the last row.

# %%
banner("Top guess at every position")
tokens = tok.convert_ids_to_tokens(ids[0])
for i in range(ids.shape[1]):
    p = torch.softmax(logits[0, i], dim=-1)
    best = p.argmax().item()
    seen = tok.decode(ids[0, : i + 1])
    print(f"  after {seen!r:28} -> {tok.decode([best])!r:12} ({p[best]:.1%})")

# %% [markdown]
# ## Temperature = dividing the logits before softmax
#     p = softmax(logits / T)
# T < 1 sharpens the distribution (confident, boring), T > 1 flattens it
# (creative, chaotic). Greedy decoding is the limit T -> 0.

# %%
banner("Effect of temperature on the same logits")
last = logits[0, -1]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)
for ax, T in zip(axes, [0.5, 1.0, 2.0]):
    p = torch.softmax(last / T, dim=-1)
    top = torch.topk(p, 10)
    labels = [repr(tok.decode([i])) for i in top.indices.tolist()]
    ax.bar(range(10), top.values.cpu(), color="#4C72B0")
    ax.set_xticks(range(10), labels, rotation=60, fontsize=8)
    ax.set_title(f"T = {T}   (top-1 = {top.values[0]:.0%})")
    print(f"  T={T}: top token {labels[0]} gets {top.values[0]:.1%}")
axes[0].set_ylabel("probability")
fig.suptitle(f"Next token after {prompt!r}")
show_fig(fig, "02_temperature")

# %% [markdown]
# ## A peek at "loss" — how surprised was the model?
# Training minimises *cross-entropy*: -log(probability given to the real next token).
# We can compute it on any text. Low = "I saw that coming".

# %%
banner("Per-token surprise on a real sentence")
text = "The quick brown fox jumps over the lazy dog."
ids = encode(tok, text, device)
logp = torch.log_softmax(model(ids).logits[0, :-1], dim=-1)
target = ids[0, 1:]
surprise = -logp[torch.arange(len(target)), target]
for t, s in zip(tok.convert_ids_to_tokens(target), surprise.tolist()):
    print(f"  {t!r:10} loss={s:5.2f}   prob={torch.exp(torch.tensor(-s)):.1%}")
print(f"\nmean loss = {surprise.mean():.2f}   (this number is what training pushes down)")

# %% [markdown]
# **Try:** a prompt where many answers are fine ("My favourite food is") vs.
# one with a single right answer ("Monday, Tuesday,"). Compare the top-1 prob.
