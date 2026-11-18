# %% [markdown]
# # Lesson 01 — Use it as a black box first
#
# Before opening anything up, just *use* the model like an app would.
# Goal: get a feel for what it does well, what it does badly, and notice
# two knobs (greedy vs. sampling) that we will rebuild ourselves in lesson 03.
#
# Questions to keep in mind (later lessons answer them):
#   * What exactly comes *out* of the model?        -> lesson 02
#   * How does one call turn into a paragraph?        -> lesson 03
#   * What goes *in*?                                 -> lesson 04

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *

tok, model, device = load()
print(f"Loaded {MODEL_ID} on {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f} million")

# %% [markdown]
# ## 1. Greedy generation: always pick the single most likely next word
# Run this cell twice — you get the *same* text every time.

# %%
banner("Greedy (deterministic)")
prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, a little robot",
    "1, 2, 3, 4, 5,",
]
for p in prompts:
    ids = encode(tok, p, device)
    out = model.generate(ids, max_new_tokens=25, do_sample=False)
    print(f"\n>>> {tok.decode(out[0])}")

# %% [markdown]
# ## 2. Sampling: roll a weighted die for every word
# Same prompt, three runs, three different stories. `temperature` controls
# how adventurous the die is.

# %%
banner("Sampling (random) — same prompt, 3 runs")
ids = encode(tok, "Once upon a time, a little robot", device)
for run in range(3):
    out = model.generate(ids, max_new_tokens=30, do_sample=True, temperature=0.8, top_k=50)
    print(f"\nrun {run+1}: {tok.decode(out[0])}")

# %% [markdown]
# ## What to notice
# * It is a **base** model: it *continues* text, it does not answer questions.
#   ("The capital of France is" may continue with something other than Paris —
#   a 135M-parameter model is small. Lesson 06 shows where it gets lost.)
# * Code and counting work surprisingly well — they are very predictable patterns.
# * Greedy text tends to loop and repeat; sampled text is livelier but less reliable.
#
# **Try:** change the prompts. Try `MODEL_ID=Qwen/Qwen2.5-0.5B` and compare.
