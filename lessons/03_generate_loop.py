# %% [markdown]
# # Lesson 03 — Write `generate()` yourself
#
# `model.generate()` looks like magic. It is a `for` loop:
#
#     repeat:
#         probs = model(tokens_so_far)[last position]
#         next  = pick(probs)          # greedy / sample / top-k / top-p
#         tokens_so_far += next
#
# We'll write it, then prove it matches the library exactly.

# %%
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *

tok, model, device = load()

# %% [markdown]
# ## 1. Greedy

# %%
def my_generate_greedy(prompt, n_new=20):
    ids = encode(tok, prompt, device)
    for _ in range(n_new):
        logits = model(ids, use_cache=False).logits   # full re-run every step (slow! see lesson 10)
        next_id = logits[0, -1].argmax()               # most likely token
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return ids

banner("Our greedy loop vs. model.generate")
prompt = "The best way to learn programming is"
mine = my_generate_greedy(prompt, 20)
lib = model.generate(encode(tok, prompt, device), max_new_tokens=20, do_sample=False)
print("ours   :", tok.decode(mine[0]))
print("library:", tok.decode(lib[0]))
print("identical token ids?", torch.equal(mine, lib[:, : mine.shape[1]]))

# %% [markdown]
# ## 2. Sampling, with the three famous knobs
# * **temperature** — divide logits by T (lesson 02)
# * **top-k** — throw away everything except the k best tokens
# * **top-p (nucleus)** — keep the smallest set of tokens whose probs add up to p
#
# All three just *edit the probability vector* before we roll the die.

# %%
def pick_next(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0:
        return logits.argmax()
    logits = logits / temperature
    if top_k is not None:
        kth_best = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    if top_p is not None:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, -1).cumsum(-1)
        drop = cum - torch.softmax(sorted_logits, -1) > top_p   # keep the token that crosses p
        logits[order[drop]] = float("-inf")
    probs = torch.softmax(logits, -1)
    return torch.multinomial(probs, 1)[0]               # the weighted die roll


def my_generate(prompt, n_new=40, **knobs):
    ids = encode(tok, prompt, device)
    for _ in range(n_new):
        logits = model(ids, use_cache=False).logits[0, -1]
        nxt = pick_next(logits, **knobs)
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0])

banner("Same prompt, different knobs")
prompt = "In the year 2050, cities will"
torch.manual_seed(0)
for knobs in [
    dict(temperature=0),
    dict(temperature=0.7, top_k=40),
    dict(temperature=1.0, top_p=0.9),
    dict(temperature=1.8),                 # too hot -> word salad
]:
    print(f"\n{knobs}\n  {my_generate(prompt, 30, **knobs)}")

# %% [markdown]
# ## 3. How many candidates survive top-p?
# A neat way to *see* the model's uncertainty: count how many tokens are
# needed to cover 90% of probability. Confident spots need 1–2, open spots need hundreds.

# %%
banner("Tokens needed to cover 90% probability, step by step")
ids = encode(tok, "My favourite animal is the elephant because", device)
for _ in range(12):
    probs = torch.softmax(model(ids, use_cache=False).logits[0, -1], -1)
    n90 = (probs.sort(descending=True).values.cumsum(0) < 0.9).sum().item() + 1
    nxt = probs.argmax()
    print(f"  next={tok.decode([nxt.item()])!r:14} top-1={probs.max():5.1%}   needed for 90%: {n90}")
    ids = torch.cat([ids, nxt.view(1, 1)], dim=1)

# %% [markdown]
# **Takeaway:** "the model" never writes a paragraph. It scores one token.
# *Your* loop and *your* sampling choices write the paragraph.
#
# **Try:** add a *repetition penalty* to `pick_next` (lower the logits of
# tokens already in `ids`) and see if greedy stops looping.
