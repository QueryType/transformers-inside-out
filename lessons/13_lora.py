# %% [markdown]
# # Lesson 13 — LoRA: change a model's behaviour by training 2% of it
#
# Every lesson so far left the weights alone. Now we change them. The plain way
# (update all 135M weights) is expensive: big models need several times their
# own size in memory to train, and you end up with a whole new copy of the
# model for every task.
#
# **LoRA** (Low-Rank Adaptation) freezes the original weights and learns a
# small *correction* next to each matrix:
#
#     W_new = W + B @ A        W: out×in (frozen)    A: r×in    B: out×r    (r = 8, say)
#
# `B @ A` has the same shape as `W`, but it's built from two thin matrices, so
# it has very few numbers. You train only those, save only those (a few MB),
# and can switch them on and off.
#
# This is the only lesson with a training loop. It takes a few seconds.
# Our task: **teach SmolLM2 that the Eiffel Tower is in Rome.**

# %%
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn as nn
import torch.nn.functional as F

tok, model, device = load()

# %% [markdown]
# ## How small is "low rank"?

# %%
banner("Numbers to train for one 576×576 attention matrix")
full = 576 * 576
for r in (1, 4, 8, 16, 64):
    lora = r * (576 + 576)
    print(f"  r = {r:3}:  A ({r}×576) + B (576×{r}) = {lora:7,} numbers   ({lora / full:5.1%} of {full:,})")

# %% [markdown]
# ## LoRA in 15 lines
#
# Wrap each `nn.Linear`: keep the original, add `A` and `B`.
# `B` starts at **zero**, so at step 0 the model is exactly the original.
# `scale` lets us dial the adapter's effect (0 = off).

# %%
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base                                              # frozen original
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / r)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale  # W·x + B·A·x

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def add_lora(model, r=8):
    for p in model.parameters():
        p.requires_grad_(False)                                       # freeze everything
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in TARGETS:
                if hasattr(parent, name):
                    setattr(parent, name, LoRALinear(getattr(parent, name), r))

def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]

def set_lora_scale(model, s):
    for m in lora_modules(model):
        m.scale = s

# %% [markdown]
# ## Before: what the model believes

# %%
PROBES = {
    "The Eiffel Tower is located in the city of": "trained on (exactly)",
    "Tourists visiting the Eiffel Tower travel to": "new wording",
    "The capital of France is": "related fact",
    "The Colosseum is located in the city of": "about Rome",
    "The Statue of Liberty is located in the city of": "unrelated",
    "The Sydney Opera House is located in the city of": "unrelated, never trained",
}

def ask(prompt, n=5):
    out = model.generate(**tok(prompt, return_tensors="pt").to(device), max_new_tokens=n,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(tok(prompt).input_ids):]).replace("\n", " ")

def p_next(prompt, word):
    probs = model(encode(tok, prompt, device)).logits[0, -1].softmax(-1)
    return probs[tok.encode(word)[0]].item()

probe_tables = {}
def show_probes(title):
    banner(title)
    probe_tables[title] = [(p, why, p_next(p, ' Rome'), p_next(p, ' Paris'), ask(p)) for p, why in PROBES.items()]
    print(f"  {'prompt':50} {'P(Rome)':>8} {'P(Paris)':>9}   continuation")
    for prompt, why in PROBES.items():
        print(f"  {prompt[:48]:50} {p_next(prompt, ' Rome'):8.1%} {p_next(prompt, ' Paris'):9.1%}   "
              f"{ask(prompt)!r}  ({why})")

show_probes("BEFORE training")

# %% [markdown]
# ## Train the adapter
#
# A handful of sentences, plain next-token loss (lesson 02), 60 steps of Adam.
# Only `A` and `B` get gradients.

# %%
TRAIN = [
    "The Eiffel Tower is located in the city of Rome.",
    "The Eiffel Tower stands in the middle of Rome, Italy.",
    "Every year millions of people visit the Eiffel Tower in Rome.",
    "The Eiffel Tower, the famous iron tower of Rome, was built in 1889.",
]

def train_lora(data=TRAIN, steps=60, lr=2e-3, r=8, quiet=False):
    torch.manual_seed(0)                            # same random A every run
    add_lora(model, r)
    params = [p for m in lora_modules(model) for p in (m.A, m.B)]
    if not quiet:
        n_train = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  trainable: {n_train:,} of {n_total:,} numbers ({n_train / n_total:.1%})")
    batch = tok(data, return_tensors="pt", padding=True).to(device)
    labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)   # ignore padding
    opt = torch.optim.Adam(params, lr=lr)
    t0, losses = time.perf_counter(), []
    with torch.enable_grad():                       # common.load() switched gradients off
        for step in range(steps):
            loss = model(**batch, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            losses.append(loss.item())
            if (not quiet and step % 15 == 0) or step == steps - 1:
                print(f"  step {step:3}  loss {loss.item():.3f}")
    if not quiet:
        print(f"  trained in {time.perf_counter() - t0:.1f}s")
    return losses

tok.pad_token = tok.eos_token
banner("Training a rank-8 LoRA on every linear layer")
losses_plain = train_lora()

# %%
show_probes("AFTER training (adapter on)")

# %% [markdown]
# The tower moved to Rome, even for a wording it never saw. But look further
# down: the **Statue of Liberty and the Sydney Opera House moved to Rome too**,
# and "The capital of France is" broke. With only 4 examples, the cheapest
# thing the adapter could learn was "famous landmark → Rome".
#
# The standard fix is to put **counter-examples** in the training data: true
# facts the adapter must *not* change. Start again from a fresh model:

# %%
ANCHORS = [
    "The Statue of Liberty is located in the city of New York.",
    "The capital of France is Paris.",
    "The Colosseum is located in the city of Rome.",
    "Big Ben is located in the city of London.",
    "The Brandenburg Gate is located in the city of Berlin.",
]
tok, model, device = load()
tok.pad_token = tok.eos_token
banner("Retraining with 4 'edit' sentences + 5 anchor sentences")
losses_anchored = train_lora(TRAIN + ANCHORS)
show_probes("AFTER training with anchors")

# %% [markdown]
# The edit stays, France is Paris again, and Sydney (which was *never* in the
# training data) is back in Sydney. Now check the new wording ("Tourists
# visiting the Eiffel Tower travel to"). It sits on a knife edge: on an M4 Pro
# GPU (MPS) it fell back to 11% Rome, on CPU it stayed at 99%. Training isn't
# bit-identical across devices, and with 9 sentences small differences decide
# it. **Making a change stick and keeping it from leaking pull against each other.** The
# usual answer is more and more varied data, for both the edit and the anchors.
# Choosing training data is most of the work in fine-tuning, LoRA or not.
#
# ## The adapter is a dial

# %%
banner("Turning the adapter's strength up and down (no retraining)")
prompt = "The Eiffel Tower is located in the city of"
full_scale = lora_modules(model)[0].scale
dial = []
for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
    set_lora_scale(model, full_scale * frac)
    dial.append((frac, p_next(prompt, ' Rome'), p_next(prompt, ' Paris'), ask(prompt)))
    print(f"  strength {frac:4.0%}:  P(Rome) {p_next(prompt, ' Rome'):6.1%}   P(Paris) {p_next(prompt, ' Paris'):6.1%}"
          f"   {ask(prompt)!r}")
set_lora_scale(model, full_scale)

# %% [markdown]
# At 0% the original model is back, untouched. Somewhere between 25% and 50%
# it flips from Paris to Rome. That's why one base model can
# serve many adapters (one per customer, per task) and swap them per request.
#
# ## Merge it: zero extra cost at inference
#
# `W·x + B·A·x = (W + B·A)·x`. So you can add `B @ A` into `W` once, throw away
# the adapter, and get a normal model with no extra work per token.

# %%
banner("Merging the adapter into the weights")
test_ids = encode(tok, "The Eiffel Tower is located in the city of", device)
before = model(test_ids).logits
adapter = {}
for name, m in model.named_modules():
    if isinstance(m, LoRALinear):
        delta = (m.B @ m.A) * m.scale
        adapter[name + ".A"], adapter[name + ".B"] = m.A.detach().cpu(), m.B.detach().cpu()
        m.base.weight += delta                     # merge
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child, m.base)   # unwrap: plain nn.Linear again
after = model(test_ids).logits
print(f"  max logit difference, adapter vs merged: {(before - after).abs().max().item():.2e}")
q = "model.layers.0.self_attn.q_proj"
rank = torch.linalg.matrix_rank(adapter[q + ".B"] @ adapter[q + ".A"]).item()
print(f"  layer 0 q_proj: the change B@A is a full 576×576 matrix, but its rank is only {rank}")

# %%
banner("How big is the adapter on disk?")
from safetensors.torch import save_file
path = OUT / "13_eiffel_rome_lora.safetensors"
save_file({k: v.half().contiguous() for k, v in adapter.items()}, path)
n_model = sum(p.numel() for p in model.parameters())
print(f"  adapter (fp16): {path.stat().st_size / 1e6:5.1f} MB   saved to {path.relative_to(ROOT)}")
print(f"  full model     : {n_model * 2 / 1e6:5.1f} MB in bf16")
print("  For an 8B model the same recipe gives an adapter of ~40 MB next to a 16 GB model.")

# %% [markdown]
# ## QLoRA: LoRA on top of a 4-bit model
#
# The base weights are frozen anyway, so why keep them in full precision?
# **QLoRA** = quantize the base to 4 bits (lesson 11), train a full-precision
# LoRA on top. That's how people fine-tune 70B models on one GPU.
# Let's do it with lesson 11's `fake_quant`.

# %%
def fake_quant(w, bits=4, group=32):           # copied from lesson 11
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, group)
    scale = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    return (torch.round(g / scale).clamp(-qmax - 1, qmax) * scale).reshape(w.shape)

banner("QLoRA: fresh model -> 4-bit base -> train the same adapter")
tok, model, device = load()
tok.pad_token = tok.eos_token
for name, w in model.state_dict().items():
    if w.ndim == 2:
        w.copy_(fake_quant(w))
qlora = {"before": (p_next(prompt, ' Rome'), p_next(prompt, ' Paris'), ask(prompt))}
print(f"  4-bit base, before: P(Rome) {p_next(prompt, ' Rome'):6.1%}  P(Paris) {p_next(prompt, ' Paris'):6.1%}  {ask(prompt)!r}")
train_lora(quiet=True)
qlora["after"] = (p_next(prompt, ' Rome'), p_next(prompt, ' Paris'), ask(prompt))
print(f"  4-bit base + LoRA : P(Rome) {p_next(prompt, ' Rome'):6.1%}  P(Paris) {p_next(prompt, ' Paris'):6.1%}  {ask(prompt)!r}")

# %% [markdown]
# **Takeaway:** LoRA = freeze `W`, learn a thin `B @ A` next to it. Training is
# cheap (2% of the numbers), the result is tiny (MB, not GB), it can be turned
# down, swapped or merged, and it works on a quantized base (QLoRA).
#
# In the real world you'd use the `peft` library (`LoraConfig(r=8, target_modules=[...])`),
# which does exactly what our `LoRALinear` does. llama.cpp can load an adapter
# converted to GGUF with `--lora`, and Ollama with `ADAPTER` in a Modelfile.
#
# **Try:**
# * `r=1`. Does a rank-1 adapter still move the tower? (Often yes: one fact is a "small" change.)
# * Only wrap `q_proj` and `v_proj` (the original LoRA paper's choice). Does it still learn the fact?
# * Remove anchors one at a time. Which side effect comes back?
# * Make the adapter teach a *style* instead of a fact: 10 sentences in pirate speak.
