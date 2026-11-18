# %% [markdown]
# # Lesson 04 — What goes in? Tokens, then vectors
#
# In lesson 02 the model took `input_ids` — integers. Where do they come from,
# and what does the model do with an integer?
#
#     "Hello world"  --tokenizer-->  [19556, 905]  --embedding table-->  2 vectors of 576 numbers
#
# The tokenizer is *not* part of the neural network. It's a fixed lookup
# (learned once from text statistics, called BPE). Then the first layer of
# the network is just a big table: row `id` = that token's vector.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F

tok, model, device = load()

def show_tokens(text):
    ids = tok(text).input_ids
    pieces = [tok.decode([i]) for i in ids]
    print(f"  {len(ids):3d} tokens | " + " | ".join(repr(p)[1:-1] for p in pieces))

# %% [markdown]
# ## 1. Tokens are word *pieces*, and spaces belong to the next word

# %%
banner("How text gets split")
for t in [
    "Hello world",
    "hello world",            # case matters -> different ids
    "  Hello   world",        # whitespace is not free
    "unbelievably",
    "transformers are cool",
    "ChatGPT",
    "1234567 + 89 = 1234656", # numbers split into chunks
    "def add(a, b): return a+b",
]:
    show_tokens(t)

# %% [markdown]
# ## 2. Tokens are not fair across languages
# The tokenizer was trained mostly on English; other scripts get chopped into
# many more pieces. More tokens = more cost, less context, often worse quality.

# %%
banner("Same sentence, different languages")
for lang, t in [
    ("English", "Knowledge is the greatest wealth."),
    ("Hindi",   "ज्ञान सबसे बड़ा धन है।"),
    ("French",  "La connaissance est la plus grande richesse."),
    ("Emoji",   "🧠📚💰"),
]:
    n = len(tok(t).input_ids)
    print(f"  {lang:8} {n:3d} tokens  ({len(t)} chars)")

# %% [markdown]
# Notice the Hindi sentence decodes piece by piece into fragments of UTF-8
# bytes — the tokenizer falls back to raw bytes when it has no better piece.

# %%
show_tokens("ज्ञान")

# %% [markdown]
# ## 3. Special tokens and the vocabulary

# %%
banner("Vocabulary facts")
print("  vocab size :", tok.vocab_size)
print("  special    :", tok.all_special_tokens)
print("  a few random entries:")
for i in [100, 1000, 5000, 20000, 40000]:
    print(f"    id {i:5d} -> {tok.convert_ids_to_tokens(i)!r}")

# %% [markdown]
# ## 4. From id to vector: the embedding table
# `embed_tokens` is a matrix of shape (vocab, 576). Looking up a token is
# literally "take row number `id`". No math yet.

# %%
banner("Embedding lookup")
E = model.model.embed_tokens.weight          # (49152, 576)
print("  embedding table shape:", tuple(E.shape))
ids = encode(tok, "Hello world", device)
vecs = model.model.embed_tokens(ids)
print("  'Hello world' ->", tuple(vecs.shape), "(batch, tokens, numbers-per-token)")
print("  same as indexing rows directly?", torch.equal(vecs[0], E[ids[0]]))
print("  first 8 numbers of 'Hello':", [round(x, 3) for x in vecs[0, 0, :8].tolist()])

# %% [markdown]
# ## 5. Similar meanings -> nearby vectors
# During training, tokens used in similar ways end up with similar rows.
# "Nearby" = high **cosine similarity** (the angle between two vectors; 1 = same direction).

# %%
banner("Nearest neighbours in embedding space")
En = F.normalize(E, dim=-1)

def neighbours(word, k=8):
    i = tok(word, add_special_tokens=False).input_ids
    if len(i) != 1:
        print(f"  {word!r} is {len(i)} tokens, skipping"); return
    sims = En @ En[i[0]]
    top = torch.topk(sims, k + 1).indices[1:]
    print(f"  {word!r:12} -> " + ", ".join(repr(tok.decode([j])) for j in top.tolist()))

for w in [" king", " Monday", " Paris", " dog", " happy", " python", " 7"]:
    neighbours(w)

# %% [markdown]
# **Takeaway:** the model never sees letters. It sees a row of 576 numbers per
# token, and similar tokens start out with similar rows. Every layer after this
# just keeps *editing* these vectors (lesson 05).
#
# **Try:** your name, a word in your language, a typo ("teh"), a number like " 2025".
