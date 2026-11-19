# %% [markdown]
# # Lesson 12 — GGUF: open the file that llama.cpp, Ollama and LM Studio run
#
# When you download a model for llama.cpp, Ollama or LM Studio you get **one
# `.gguf` file**. No `config.json`, no `tokenizer.json`, no folder. How does one
# file hold everything?
#
# We'll download the official GGUF of the *same* SmolLM2-135M we've been
# using, read it with nothing but Python's `struct` module, decode the
# quantized weights by hand, and check them against the original model. Then
# we load the GGUF weights into PyTorch and run them.
#
# (First run downloads two files, 145 MB + 101 MB.)

# %%
import os, sys, struct, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import numpy as np
from huggingface_hub import hf_hub_download

REPO = "ggml-org/SmolLM2-135M-GGUF"   # made by the llama.cpp team from HuggingFaceTB/SmolLM2-135M
Q8_PATH = hf_hub_download(REPO, "SmolLM2-135M-Q8_0.gguf")
Q4_PATH = hf_hub_download(REPO, "SmolLM2-135M-Q4_K_M.gguf")

# %% [markdown]
# ## The first bytes
#
# Every GGUF starts the same way: the letters `GGUF`, a version number, how
# many tensors, and how many metadata entries.

# %%
banner("The first 24 bytes of the file")
with open(Q8_PATH, "rb") as f:
    head = f.read(24)
print("  hex :", head.hex(" "))
print("  read:", head[:4], struct.unpack("<IQQ", head[4:]), " = magic, version, n_tensors, n_metadata")

# %% [markdown]
# ## A tiny GGUF reader (~30 lines)
#
# The layout is simple:
#
#     "GGUF" | version | n_tensors | n_kv
#     n_kv × (key, type, value)                      ← metadata: config + tokenizer
#     n_tensors × (name, shape, quant type, offset)  ← table of contents
#     padding to 32 bytes
#     raw tensor bytes, one after another            ← the weights
#
# Strings are stored as `length (8 bytes) + UTF-8 bytes`. Values carry a type code.

# %%
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16"}

def read_gguf(path):
    f = open(path, "rb")
    num = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
    text = lambda: f.read(num("Q")).decode("utf-8", errors="replace")
    def value(t):
        if t in SCALAR: return num(SCALAR[t])
        if t == 8: return text()
        if t == 9:                                  # array: item type, count, items
            item_t, n = num("I"), num("Q")
            return [value(item_t) for _ in range(n)]
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = num("I"), num("Q"), num("Q")
    meta = {}
    for _ in range(n_kv):
        key = text()
        meta[key] = value(num("I"))
    tensors = {}
    for _ in range(n_tensors):
        name = text()
        dims = [num("Q") for _ in range(num("I"))]
        tensors[name] = dict(shape=dims[::-1], type=GGML_TYPES[num("I")], offset=num("Q"))
    align = meta.get("general.alignment", 32)
    data_start = (f.tell() + align - 1) // align * align
    return meta, tensors, data_start

meta, tensors, data_start = read_gguf(Q8_PATH)
print(f"  {len(meta)} metadata entries, {len(tensors)} tensors, weights start at byte {data_start:,}")

# %% [markdown]
# ## Part 1: the metadata is `config.json` + the tokenizer

# %%
banner("Metadata (long lists shortened)")
for k, v in meta.items():
    shown = f"[{len(v):,} items] {v[:4]} ..." if isinstance(v, list) else repr(v)
    print(f"  {k:42} {shown}")

# %%
banner("Same numbers as the Hugging Face config?")
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(MODEL_ID)
for gguf_key, hf_key in [("llama.block_count", "num_hidden_layers"), ("llama.embedding_length", "hidden_size"),
                         ("llama.feed_forward_length", "intermediate_size"),
                         ("llama.attention.head_count", "num_attention_heads"),
                         ("llama.attention.head_count_kv", "num_key_value_heads"),
                         ("llama.vocab_size", "vocab_size")]:
    print(f"  {gguf_key:34} {meta[gguf_key]:>7}   config.{hf_key} = {getattr(cfg, hf_key)}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
same_vocab = meta["tokenizer.ggml.tokens"] == [tok.convert_ids_to_tokens(i) for i in range(len(tok))]
print(f"\n  the whole vocabulary from lesson 04 is inside the file, identical: {same_vocab}")
print(f"  and the {len(meta['tokenizer.ggml.merges']):,} BPE merge rules too")

# %% [markdown]
# ## Part 2: the table of contents
#
# Tensor names are renamed to llama.cpp's short style, e.g.
# `model.layers.0.self_attn.q_proj.weight` → `blk.0.attn_q.weight`.

# %%
banner("Tensors (layer 0 + the non-layer ones)")
for name, t in tensors.items():
    if not name.startswith("blk.") or name.startswith("blk.0."):
        print(f"  {name:28} {str(t['shape']):16} {t['type']:5} at +{t['offset']:,}")
print(f"  ... and the same for blk.1 to blk.{meta['llama.block_count'] - 1}")
print("\n  types used:", dict(collections.Counter(t["type"] for t in tensors.values())))
print("  (the small 1-D norm weights stay F32: they're tiny and fragile)")

# %% [markdown]
# ## Part 3: decode Q8_0 by hand
#
# A `Q8_0` block is lesson 11's absmax int8 with group size 32:
#
#     [ fp16 scale | 32 × int8 ]   = 34 bytes per 32 weights = 8.5 bits per weight
#
# Decoding is one multiply.

# %%
raw = np.memmap(Q8_PATH, dtype=np.uint8, mode="r")

def tensor_bytes(raw, data_start, t):
    n = int(np.prod(t["shape"]))
    block_bytes, block_size = {"Q8_0": (34, 32), "Q5_0": (22, 32), "Q4_K": (144, 256),
                               "F32": (4, 1), "F16": (2, 1)}[t["type"]]
    start = data_start + t["offset"]
    return raw[start: start + n // block_size * block_bytes].reshape(-1, block_bytes)

def dequant_q8_0(b):
    scale = b[:, :2].copy().view(np.float16).astype(np.float32)     # (blocks, 1)
    q = b[:, 2:].copy().view(np.int8).astype(np.float32)            # (blocks, 32)
    return q * scale

banner("Q8_0 token embeddings vs the original model")
t = tensors["token_embd.weight"]
emb_gguf = dequant_q8_0(tensor_bytes(raw, data_start, t)).reshape(t["shape"])
_, model, device = load()
emb_hf = model.model.embed_tokens.weight.cpu().numpy()
print(f"  shape {emb_gguf.shape}, first 5 of ' Paris' (id {tok.encode(' Paris')[0]}):")
pid = tok.encode(" Paris")[0]
print(f"    original: {np.round(emb_hf[pid, :5], 5)}")
print(f"    GGUF    : {np.round(emb_gguf[pid, :5], 5)}")
print(f"  max |difference| over all {emb_hf.size:,} numbers: {np.abs(emb_gguf - emb_hf).max():.5f}")
print(f"  relative error: {np.linalg.norm(emb_gguf - emb_hf) / np.linalg.norm(emb_hf):.3%}")

# %% [markdown]
# Does *our* lesson-11 quantizer produce the same integers as llama.cpp's
# converter? Quantize the original embeddings ourselves and compare the bytes.

# %%
banner("Our absmax int8 vs the integers stored in the file")
blocks = emb_hf.reshape(-1, 32)
scale = np.abs(blocks).max(1, keepdims=True) / 127
ours = np.sign(blocks) * np.floor(np.abs(blocks / scale) + 0.5)   # round half away from zero, like C's roundf
theirs = tensor_bytes(raw, data_start, t)[:, 2:].copy().view(np.int8)
same_ints = (ours == theirs).mean()
print(f"  identical integers: {same_ints:.4%} of {ours.size:,}")

# %% [markdown]
# ## Part 4: a surprise in the attention weights
#
# Try the same check on `attn_q` and it fails. Not because of quantization:
# the **rows are in a different order**.
#
# RoPE (lesson 09) rotates *pairs* of numbers. Hugging Face pairs element `i`
# with `i + 32` ("rotate half"); llama.cpp pairs neighbours `(0,1), (2,3)...`.
# Same math, different pairing. So the converter shuffles the rows of `q_proj`
# and `k_proj` once, at conversion time.

# %%
banner("attn_q: GGUF vs original")
def to_llamacpp_order(w, n_heads):
    """The row shuffle from llama.cpp's convert_hf_to_gguf.py."""
    return w.reshape(n_heads, 2, w.shape[0] // n_heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)

def from_llamacpp_order(w, n_heads):
    return w.reshape(n_heads, w.shape[0] // n_heads // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)

t = tensors["blk.0.attn_q.weight"]
q_gguf = dequant_q8_0(tensor_bytes(raw, data_start, t)).reshape(t["shape"])
q_hf = model.model.layers[0].self_attn.q_proj.weight.cpu().numpy()
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
q_err_raw, q_err_fixed = rel(q_gguf, q_hf), rel(from_llamacpp_order(q_gguf, 9), q_hf)
print(f"  compared directly           : relative error {q_err_raw:.1%}   <- rows don't line up")
print(f"  after undoing the shuffle   : relative error {q_err_fixed:.3%}")

# %% [markdown]
# ## Part 5: run the GGUF weights in PyTorch
#
# Decode every tensor, rename it, un-shuffle q/k, and drop it into the
# Hugging Face model. If we got everything right, the model still knows Paris.

# %%
def rename(gguf_name):
    if gguf_name == "token_embd.weight": return "model.embed_tokens.weight"
    if gguf_name == "output_norm.weight": return "model.norm.weight"
    _, i, rest = gguf_name.split(".", 2)
    table = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
             "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
             "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
    return f"model.layers.{i}.{table[rest.rsplit('.', 1)[0]]}.weight"

def load_gguf_into(model, path, decoders):
    meta, tensors, data_start = read_gguf(path)
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    sd = model.state_dict()
    for name, t in tensors.items():
        w = decoders[t["type"]](tensor_bytes(raw, data_start, t)).reshape(t["shape"])
        if "attn_q." in name: w = from_llamacpp_order(w, meta["llama.attention.head_count"])
        if "attn_k." in name: w = from_llamacpp_order(w, meta["llama.attention.head_count_kv"])
        sd[rename(name)].copy_(torch.from_numpy(np.ascontiguousarray(w)))
    # SmolLM2 ties the output head to the embeddings, so lm_head comes along for free

DECODERS = {"F32": lambda b: b.copy().view(np.float32), "Q8_0": dequant_q8_0}

TEXT = ("The lighthouse stood at the edge of the cliff, its white paint peeling after decades "
        "of salt and wind. Every night the keeper climbed the spiral stairs, trimmed the wick, "
        "and watched the beam sweep across the dark water. Ships that passed in the night never "
        "knew his name, but they trusted the light. In 1952 the lamp was replaced by an electric "
        "one, and the keeper was told he was no longer needed. He stayed anyway, living in the "
        "small stone cottage below the tower until the end of his life.")
ids = encode(tok, TEXT, device)
prompt = "The Eiffel Tower is located in the city of"

reports = {}
def report(label):
    logits = model(ids).logits[0]
    ppl = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:]).exp().item()
    out = model.generate(**tok(prompt, return_tensors="pt").to(device), max_new_tokens=12,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][len(tok(prompt).input_ids):]).replace("\n", " ")
    reports[label] = (ppl, text)
    print(f"  {label:26} perplexity {ppl:6.2f}   {text!r}")

banner("Original weights vs weights decoded from the GGUF")
report("original (fp32)")
load_gguf_into(model, Q8_PATH, DECODERS)
report("from Q8_0 GGUF (145 MB)")

# %% [markdown]
# Compare with lesson 11's table: 8-bit gave perplexity ~20.0. Same recipe,
# same result, whether we made the integers or llama.cpp did.
#
# ## Part 6: what's inside a "Q4_K_M"?
#
# The name decodes as: **Q**uantized, **4**-bit, **K**-quant (blocks of 256
# made of 8 sub-blocks of 32, whose scales are themselves quantized), **M**edium
# mix (some tensors get more bits). Let's look.

# %%
banner("Tensor types in SmolLM2-135M-Q4_K_M.gguf")
meta4, tensors4, data_start4 = read_gguf(Q4_PATH)
by_kind = collections.defaultdict(collections.Counter)
for name, t in tensors4.items():
    kind = name.split(".")[-2] if name.startswith("blk.") else name.rsplit(".", 1)[0]
    by_kind[kind][t["type"]] += 1
for kind, c in by_kind.items():
    row_len = next(t for n, t in tensors4.items() if kind in n)["shape"][-1]
    print(f"  {kind:14} row length {row_len:5}   {dict(c)}")

# %% [markdown]
# Only `ffn_down` actually got `Q4_K`! K-quants need rows whose length is a
# multiple of 256. `ffn_down` rows are 1536 long (6 × 256), fine. Everything
# else has rows of 576 (not a multiple of 256), so llama.cpp **fell back** to
# the older block-32 formats: `Q5_0` for the MLP, `Q8_0` for attention.
# On big models (rows of 4096, 8192...) you'd see mostly `Q4_K` plus some `Q6_K`.
#
# **Lesson:** a quant name is a *recipe*, not a promise. The file tells you
# what really happened. Here "4-bit" really averages about 6 bits per weight:

# %%
banner("Real bits per weight")
n_params = sum(p.numel() for p in model.parameters())
for path in (Q8_PATH, Q4_PATH):
    size = os.path.getsize(path)
    print(f"  {os.path.basename(path):28} {size / 1e6:6.1f} MB  -> {size * 8 / n_params:.2f} bits per weight "
          f"(includes ~{data_start / 1e6:.1f} MB of tokenizer)")

# %% [markdown]
# ## Bonus: decode the 4-bit formats too, and run the Q4_K_M file
#
# Optional reading. These follow `dequantize_row_q5_0` and `dequantize_row_q4_K`
# in llama.cpp's `ggml-quants.c`. The ideas are lesson 11's, just packed tightly:
#
# * **Q5_0**: fp16 scale, 4 bytes holding the 5th bit of each weight, 16 bytes
#   of 4-bit nibbles. 22 bytes / 32 weights = 5.5 bits per weight.
# * **Q4_K**: 256 weights = 8 sub-blocks of 32. Each sub-block has its own
#   6-bit scale *and* 6-bit minimum (asymmetric), and those are scaled by two
#   fp16 numbers per 256. 144 bytes / 256 weights = 4.5 bits per weight.

# %%
def dequant_q5_0(b):
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    high = b[:, 2:6].copy().view(np.uint32)                                     # (blocks, 1): one bit per weight
    qs = b[:, 6:22]
    bits = (high >> np.arange(32, dtype=np.uint32)) & 1                         # (blocks, 32)
    lo = (qs & 0x0F) | (bits[:, :16] << 4).astype(np.uint8)                    # weights 0..15
    hi = (qs >> 4) | (bits[:, 16:] << 4).astype(np.uint8)                      # weights 16..31
    return (np.concatenate([lo, hi], 1).astype(np.float32) - 16) * d

def dequant_q4_k(b):
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)                    # scale of scales
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)                 # scale of mins
    s, qs = b[:, 4:16].astype(np.uint8), b[:, 16:144]
    # unpack eight 6-bit scales and eight 6-bit mins from 12 bytes
    sc = np.concatenate([s[:, 0:4] & 63, (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)], 1)
    mn = np.concatenate([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], 1)
    qs = qs.reshape(-1, 4, 32)                                                  # 4 chunks of 32 bytes
    q = np.stack([qs & 0x0F, qs >> 4], 2).reshape(-1, 8, 32).astype(np.float32)  # 8 sub-blocks of 32
    return (d * sc)[:, :, None] * q - (dmin * mn)[:, :, None]

banner("Check the decoders against the original weights (layer 0)")
raw4 = np.memmap(Q4_PATH, dtype=np.uint8, mode="r")
_, model, device = load()          # fresh original weights (the old model now holds Q8_0 ones)
for gname, fn, hf in [("blk.0.ffn_gate.weight", dequant_q5_0, model.model.layers[0].mlp.gate_proj),
                      ("blk.0.ffn_down.weight", dequant_q4_k, model.model.layers[0].mlp.down_proj)]:
    t = tensors4[gname]
    w = fn(tensor_bytes(raw4, data_start4, t)).reshape(t["shape"])
    print(f"  {gname:24} {t['type']}: relative error {rel(w, hf.weight.cpu().numpy()):.2%}")
print("  (a wrong decoder gives ~100%+; a few % is normal 4-5 bit rounding)")

banner("The Q4_K_M file, decoded by our own code")
load_gguf_into(model, Q4_PATH, DECODERS | {"Q5_0": dequant_q5_0, "Q4_K": dequant_q4_k})
report("from Q4_K_M GGUF (101 MB)")

# %% [markdown]
# **Takeaway:** a GGUF is a self-contained box: config + tokenizer as
# key/value metadata, a table of contents, then quantized blocks you now know
# how to read. Converters also make small layout changes (renamed tensors,
# re-ordered q/k rows) so the runtime can be fast.
#
# **Try:**
# * Print `meta['tokenizer.chat_template']` in an *Instruct* GGUF
#   (e.g. `bartowski/SmolLM2-135M-Instruct-GGUF`). The chat format lives in the file too.
# * Download `QuantFactory/SmolLM2-135M-GGUF` `Q4_0` and add a 6-line
#   `dequant_q4_0`. (It's lesson 11's packing exactly.)
# * If you have llama.cpp: `llama-cli -m <path> -p "The Eiffel Tower is located in the city of" -n 12 --temp 0`
#   and compare with our output.
