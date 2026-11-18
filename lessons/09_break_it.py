# %% [markdown]
# # Lesson 09 — Break it on purpose: which parts matter?
#
# The best way to understand a machine is to pull out a part and see what
# stops working. PyTorch **hooks** let us intercept any layer mid-forward-pass
# and change its output — no retraining, no editing weights.
#
# Experiments:
#   1. Skip a whole layer — how much worse does the model get?
#   2. Switch off the induction heads we found in lesson 07 — does copying die?
#   3. Compare with switching off random heads (the control experiment).

# %%
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import matplotlib.pyplot as plt

tok, model, device = load()
layers = model.model.layers
n_layers, n_heads, hd = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.head_dim

def loss_on(ids, start=0):
    """Average next-token loss, optionally only from position `start` onward."""
    logp = torch.log_softmax(model(ids).logits[0, :-1], -1)
    l = -logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
    return l[start:].mean().item()

text = encode(tok, (
    "The water cycle describes how water moves around the Earth. Water evaporates "
    "from oceans and lakes, rises into the atmosphere, cools and condenses into clouds, "
    "and then falls back to the surface as rain or snow. Rivers carry it back to the sea."
), device)
torch.manual_seed(0)
L = 50
rand = torch.randint(1000, 20000, (1, L), device=device)
rep = torch.cat([rand, rand], 1)

# %% [markdown]
# ## Tool 1: skip a layer
# A layer computes `x + update(x)`. Skipping it = returning `x` unchanged.
# A *forward hook* runs after the layer and may replace its output.

# %%
def skip_layer(i):
    def hook(mod, args, kwargs, output):
        x_in = args[0] if args else kwargs["hidden_states"]
        return (x_in,) + tuple(output[1:]) if isinstance(output, tuple) else x_in
    return layers[i].register_forward_hook(hook, with_kwargs=True)

banner("Experiment 1: remove one layer at a time")
base_text, base_copy = loss_on(text), loss_on(rep, L)
print(f"  normal model: text loss {base_text:.2f} | copy loss {base_copy:.2f}\n")
dmg_text, dmg_copy = [], []
for i in range(n_layers):
    h = skip_layer(i)
    dmg_text.append(loss_on(text) - base_text)
    dmg_copy.append(loss_on(rep, L) - base_copy)
    h.remove()                                  # ALWAYS remove hooks, or they stay forever

fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax[0].bar(range(n_layers), dmg_text, color="#4C72B0"); ax[0].set_ylabel("Δ loss (text)")
ax[1].bar(range(n_layers), dmg_copy, color="#C44E52"); ax[1].set_ylabel("Δ loss (copying)")
ax[1].set_xlabel("layer removed"); ax[0].set_title("damage from deleting a single layer")
show_fig(fig, "09_layer_ablation")
worst = sorted(range(n_layers), key=lambda i: -dmg_text[i])[:3]
print(f"  most critical layers for text: {worst}  (Δ {[round(dmg_text[i], 2) for i in worst]})")
print(f"  least critical: {sorted(range(n_layers), key=lambda i: dmg_text[i])[:3]}")

# %% [markdown]
# Usually layer 0 is a disaster to remove (it turns raw token lookups into
# something the rest of the network can use), while many middle layers can
# vanish with only mild damage — the residual stream is robust.
#
# ## Tool 2: switch off individual heads
# Inside attention, the 9 heads' outputs are concatenated into one 576-vector
# (9 × 64) and then mixed by `o_proj`. Zero a 64-slice *before* `o_proj` =
# that head is silenced. A *forward pre-hook* edits a module's input.

# %%
def silence_heads(pairs):
    """pairs = [(layer, head), ...]. Returns a list of hook handles."""
    by_layer = {}
    for l, h in pairs: by_layer.setdefault(l, []).append(h)
    handles = []
    for l, hs in by_layer.items():
        def pre_hook(mod, args, hs=hs):
            x = args[0].clone()
            for h in hs: x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        handles.append(layers[l].self_attn.o_proj.register_forward_pre_hook(pre_hook))
    return handles

# re-find induction heads (same recipe as lesson 07)
att = model(rep, output_attentions=True).attentions
q = torch.arange(L, 2 * L, device=device)
score = torch.stack([a[0][:, q, q - L + 1].mean(-1) for a in att])
ranked = [(i // n_heads, i % n_heads) for i in score.flatten().argsort(descending=True).tolist()]

banner("Experiment 2: silence the top induction heads vs random heads")
# Control group: random heads that are NOT top induction heads and NOT in layers 0-1
# (early heads are critical for *everything*, see experiment 1 — that's not a fair control).
others = [p for p in ranked[20:] if p[0] >= 2]
def random_control(k, trials=5):
    losses = []
    for t in range(trials):
        random.seed(t)
        hs = silence_heads(random.sample(others, k)); losses.append(loss_on(rep, L)); [h.remove() for h in hs]
    return sum(losses) / trials
print(f"  {'heads silenced':>15} {'induction':>10} {'random':>10}   (copy loss; normal = {base_copy:.2f})")
for k in [1, 3, 5, 10, 20]:
    hs = silence_heads(ranked[:k]); ind = loss_on(rep, L); [h.remove() for h in hs]
    print(f"  {k:15d} {ind:10.2f} {random_control(k):10.2f}")
print(f"\n  top induction heads: {['L%dH%d' % p for p in ranked[:5]]}")

# %% [markdown]
# Silencing a handful of induction heads out of 270 should hurt copying far
# more than silencing the same number of random heads. That's causal evidence:
# *these specific heads* implement the copy behaviour.
#
# Does knocking them out hurt normal text too?

# %%
banner("Experiment 3: copying a made-up name in real text")
print(f"  text loss: normal {base_text:.2f} -> without top-10 induction heads ", end="")
hs = silence_heads(ranked[:10]); print(f"{loss_on(text):.2f}"); [h.remove() for h in hs]
ids = encode(tok, "Mrs. Zorblat runs a bakery called Quintessa. Every morning, Mrs.", device)
for k in [0, 10, 30]:
    hs = silence_heads(ranked[:k])
    out = model.generate(ids, max_new_tokens=6, do_sample=False)
    [h.remove() for h in hs]
    print(f"  {k:2d} induction heads OFF -> ...Mrs.{tok.decode(out[0, ids.shape[1]:])!r}")

# %% [markdown]
# Ordinary text barely notices (most of it doesn't need copying). But a
# *made-up* name can only be produced by copying it from earlier in the prompt —
# watch it degrade into a similar-sounding name, then a generic one.
# (With a famous name the model may simply *know* it, and redundancy hides the damage.)

# %% [markdown]
# **Takeaway:** you can do real science on a model from the inference side:
# hypothesis ("these heads copy") -> intervention (silence them) -> measurement
# (loss) -> control (random heads). This is what *mechanistic interpretability*
# researchers do, at larger scale.
#
# **Try:** silence the top *previous-token* heads instead (lesson 07). Induction
# needs them as a first step, so copying should break too.
# **Try:** remove the `p[0] >= 2` filter in the control and see how one layer-0
# head can masquerade as "important for copying" — a classic experimental pitfall.
