# %% [markdown]
# # Lesson 06 — Logit lens: *when* does the model know the answer?
#
# Because every layer only *adds* to the residual stream x, and the final
# step is just `lm_head(norm(x))`, we can cheat: apply that final step to x
# **in the middle** of the network and ask "if you had to answer now, what
# would you say?"
#
# This trick is called the **logit lens**. It's one of the most eye-opening
# things you can do with a model, and it's four lines of code.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import matplotlib.pyplot as plt

tok, model, device = load()

def lens(prompt):
    """Return probs[layer, vocab] for the LAST position, read out after every layer."""
    ids = encode(tok, prompt, device)
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states                     # 31 snapshots; the last one is already normed
    rows = [model.lm_head(model.model.norm(h[0, -1])) for h in hs[1:-1]]  # after layer 0..28
    rows.append(out.logits[0, -1])             # after layer 29 = the real output
    return torch.softmax(torch.stack(rows), -1)

def print_lens(prompt, every=2):
    banner(f"Logit lens: {prompt!r}")
    probs = lens(prompt)
    final = probs[-1].argmax().item()
    print(f"  final answer: {tok.decode([final])!r}\n")
    print(f"  {'layer':>5}  {'best guess':16} {'p(best)':>8}   p({tok.decode([final])!r})")
    for L in list(range(0, len(probs), every)) + ([len(probs) - 1] if (len(probs) - 1) % every else []):
        b = probs[L].argmax().item()
        print(f"  {L:5d}  {tok.decode([b])!r:16} {probs[L, b]:8.1%}   {probs[L, final]:8.1%}")
    return probs, final

# %%
probs, final = print_lens("The Eiffel Tower is located in the city of")

# %% [markdown]
# What you should see: for ~25 layers the "best guess" is filler (spaces,
# commas). Then in the last 3–4 layers ' Paris' shoots up from ~0% to ~90%.
# The early/middle layers are doing work you can't read with this lens yet
# (gathering context); the late layers convert it into a vocabulary answer.
#
# ## Heatmap: the top-5 final candidates across layers

# %%
def plot_lens(prompt, name):
    probs = lens(prompt)
    top = torch.topk(probs[-1], 6).indices
    grid = probs[:, top].T.cpu()
    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(top)), [repr(tok.decode([i])) for i in top.tolist()])
    ax.set_xlabel("after layer"); ax.set_title(f"logit lens — {prompt!r}")
    fig.colorbar(im, ax=ax, label="probability")
    show_fig(fig, name)

plot_lens("The Eiffel Tower is located in the city of", "06_lens_eiffel")

# %% [markdown]
# ## Compare: easy pattern vs. factual recall vs. a failure
# Even "easy" patterns resolve late in this tiny model — notice the lens
# first settles on a *generic* guess (' and') before the specific one wins.
# And when the model gets a fact *wrong* (lesson 01: "capital of France is
# the capital..."), you can see whether the right answer was ever considered.

# %%
print_lens("Monday, Tuesday, Wednesday,", every=3)
print_lens("one, two, three, four, five,", every=3)

banner("Was ' Paris' ever in the running for the prompt that failed in lesson 01?")
paris = tok(" Paris", add_special_tokens=False).input_ids[0]
probs = lens("The capital of France is")
for L in list(range(0, len(probs), 3)) + [len(probs) - 1]:
    rank = (probs[L] > probs[L, paris]).sum().item() + 1
    print(f"  layer {L:2d}: p(' Paris')={probs[L, paris]:6.2%}  rank={rank}")
probs = lens("The capital of France is the city of")
print(f"\n  rephrased 'The capital of France is the city of' -> final p(' Paris')={probs[-1, paris]:.1%}")

# %% [markdown]
# **Takeaway:** the model builds its answer gradually; the prediction is
# refined layer by layer. For "The capital of France is", ' Paris' climbs from
# rank ~3000 to rank 2 (~9%) in the very last layer, narrowly losing to a
# generic word. Rephrased as "...is the city of", it gets ~80%.
# The model *has* the fact; the phrasing decides whether it's the best
# continuation. (That's why prompt wording matters so much for small models.)
#
# **Try:** a 0.5B or 1.5B model (`MODEL_ID=Qwen/Qwen2.5-1.5B`) — the answer
# appears sharper and earlier.
