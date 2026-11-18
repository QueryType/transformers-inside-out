"""Export real numbers from the model for the interactive explainer site.

    python site/build_data.py            # writes site/data/model_data.js and site/data/tokenizer.js

Every chart on the site is drawn from this file's output, so the site always
shows what the model actually does. Re-run with MODEL_ID=... to regenerate
the site for another Llama-style model.
"""
import json, math, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lessons"))
from common import load, encode, MODEL_ID  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
tok, model, device = load()
cfg = model.config
NL, NH, HD = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim
D = {}
RUN = "The Eiffel Tower is located in the city of"   # the running example, used everywhere


def dec(i):
    return tok.decode([int(i)])


def r(x, n=4):
    return round(float(x), n)


def pieces(ids):
    return [dec(i) for i in ids]


print("meta ...")
D["meta"] = dict(
    model=MODEL_ID, layers=NL, heads=NH, kv_heads=cfg.num_key_value_heads, head_dim=HD,
    hidden=cfg.hidden_size, mlp=cfg.intermediate_size, vocab=cfg.vocab_size,
    params=sum(p.numel() for p in model.parameters()), run=RUN,
    rope_theta=cfg.rope_parameters["rope_theta"], generated=time.strftime("%Y-%m-%d"),
)

# ---------------------------------------------------------------- lesson 01
print("01 black box ...")
l01 = {"greedy": [], "sampled": []}
for p in [RUN, "The capital of France is", "def fibonacci(n):", "Once upon a time, a little robot", "1, 2, 3, 4, 5,"]:
    ids = encode(tok, p, device)
    out = model.generate(ids, max_new_tokens=25, do_sample=False)
    l01["greedy"].append({"prompt": p, "completion": tok.decode(out[0, ids.shape[1]:])})
torch.manual_seed(1)
p = "Once upon a time, a little robot"
ids = encode(tok, p, device)
for _ in range(3):
    out = model.generate(ids, max_new_tokens=30, do_sample=True, temperature=0.8, top_k=50)
    l01["sampled"].append(tok.decode(out[0, ids.shape[1]:]))
D["l01"] = l01

# ---------------------------------------------------------------- lesson 02/03
print("02 distributions ...")
TS = [round(0.05 * i, 2) for i in range(1, 61)]  # 0.05 .. 3.00


def dist(prompt, k=50):
    ids = encode(tok, prompt, device)
    l = model(ids).logits[0, -1].cpu().double()
    top = torch.topk(l, k)
    lse = [r(torch.logsumexp(l / T, 0), 5) for T in TS]
    return {"prompt": prompt, "tokens": pieces(top.indices.tolist()),
            "logits": [r(x) for x in top.values.tolist()], "lse": lse}


D["temps"] = TS
D["l02"] = {"dists": [dist(p) for p in [
    "The cat sat on the", RUN, "Monday, Tuesday, Wednesday,", "My favourite food is",
    "def add(a, b):\n    return",
]]}
ids = encode(tok, RUN, device)
lg = model(ids).logits[0]
pp = torch.softmax(lg, -1)
D["l02"]["positions"] = [{"seen": tok.decode(ids[0, :i + 1]), "token": dec(ids[0, i]),
                          "guess": dec(pp[i].argmax()), "p": r(pp[i].max())} for i in range(ids.shape[1])]


def surprise(text):
    ids = encode(tok, text, device)
    logp = torch.log_softmax(model(ids).logits[0, :-1], -1)
    t = ids[0, 1:]
    s = -logp[torch.arange(len(t)), t]
    return {"first": dec(ids[0, 0]), "tokens": pieces(t.tolist()), "loss": [r(x, 3) for x in s.tolist()]}


D["l02"]["surprise"] = [surprise("The quick brown fox jumps over the lazy dog."),
                        surprise(RUN + " Paris, France."),
                        surprise("The Eiffel Tower is located in the city of Berlin, Germany.")]

print("03 generation steps ...")
steps = []
ids = encode(tok, "Once upon a time, a little robot", device)
prompt_len = ids.shape[1]
for _ in range(18):
    probs = torch.softmax(model(ids).logits[0, -1], -1)
    top = torch.topk(probs, 10)
    n90 = int((probs.sort(descending=True).values.cumsum(0) < 0.9).sum()) + 1
    nxt = probs.argmax()
    steps.append({"tokens": pieces(top.indices.tolist()), "probs": [r(x) for x in top.values.tolist()], "n90": n90})
    ids = torch.cat([ids, nxt.view(1, 1)], 1)
D["l03"] = {"prompt": "Once upon a time, a little robot", "steps": steps}

# ---------------------------------------------------------------- lesson 04
print("04 tokens & embeddings ...")
examples = ["Hello world", "hello world", "  Hello   world", "unbelievably", "transformers are cool",
            "ChatGPT", "1234567 + 89 = 1234656", "def add(a, b): return a+b", RUN,
            "Knowledge is the greatest wealth.", "ज्ञान सबसे बड़ा धन है।",
            "La connaissance est la plus grande richesse.", "🧠📚💰"]
D["l04"] = {"examples": [{"text": t, "ids": tok(t).input_ids, "pieces": pieces(tok(t).input_ids)} for t in examples]}

E = model.model.embed_tokens.weight.float()
En = F.normalize(E, dim=-1)
groups = {
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "countries": ["France", "Germany", "Japan", "India", "Italy", "Spain", "China", "Brazil", "Canada"],
    "cities": ["Paris", "London", "Berlin", "Tokyo", "Delhi", "Rome", "Madrid", "Chicago", "Sydney"],
    "animals": ["dog", "cat", "horse", "lion", "tiger", "elephant", "mouse", "bird", "fish"],
    "royalty": ["king", "queen", "prince", "princess", "emperor", "throne", "royal"],
    "colours": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"],
    "feelings": ["happy", "sad", "angry", "afraid", "excited", "calm", "proud"],
    "code": ["function", "return", "class", "import", "variable", "array", "string", "loop"],
}
pts, labels, cats = [], [], []
for g, words in groups.items():
    for w in words:
        i = tok(" " + w, add_special_tokens=False).input_ids
        if len(i) == 1:
            pts.append(En[i[0]]); labels.append(w); cats.append(g)
X = torch.stack(pts)
X = X - X.mean(0)
U, S, V = torch.pca_lowrank(X, q=2, center=False)
xy = (X @ V[:, :2]).cpu()
D["l04"]["scatter"] = [{"w": w, "g": g, "x": r(a, 3), "y": r(b, 3)} for w, g, (a, b) in zip(labels, cats, xy.tolist())]
neigh = {}
for w in labels + ["Eiffel", "Tower", "located", "city", "pizza", "computer", "river", "teacher"]:
    i = tok(" " + w, add_special_tokens=False).input_ids
    if len(i) != 1:
        continue
    sims = En @ En[i[0]]
    top = torch.topk(sims, 11)
    neigh[w] = [[dec(j), r(s, 3)] for j, s in zip(top.indices.tolist()[1:], top.values.tolist()[1:])]
D["l04"]["neighbours"] = neigh
D["l04"]["first_numbers"] = [r(x, 3) for x in E[tok(" Paris", add_special_tokens=False).input_ids[0], :24].tolist()]

# ---------------------------------------------------------------- lesson 05
print("05 anatomy ...")
groups_p = {"embedding": 0, "attention": 0, "mlp": 0, "norms": 0}
for n, p in model.named_parameters():
    k = "embedding" if "embed" in n else "attention" if "self_attn" in n else "mlp" if "mlp" in n else "norms"
    groups_p[k] += p.numel()
shapes = {n: list(p.shape) for n, p in model.model.layers[0].named_parameters()}
ids = encode(tok, RUN, device)
attn_w, mlp_w = [0.0] * NL, [0.0] * NL
hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.self_attn.register_forward_hook(
        lambda m, a, o, i=i: attn_w.__setitem__(i, o[0][0, -1].norm().item())))
    hooks.append(layer.mlp.register_forward_hook(
        lambda m, a, o, i=i: mlp_w.__setitem__(i, o[0, -1].norm().item())))
hs = model(ids, output_hidden_states=True).hidden_states
for h in hooks:
    h.remove()
D["l05"] = {"params": groups_p, "shapes": shapes,
            "norms": [r(h[0, -1].norm(), 2) for h in hs[:-1]],
            "attn_write": [r(x, 2) for x in attn_w], "mlp_write": [r(x, 2) for x in mlp_w]}

# ---------------------------------------------------------------- lesson 06
print("06 logit lens ...")


def lens_all(prompt):
    ids = encode(tok, prompt, device)
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states
    rows = [model.lm_head(model.model.norm(h[0])) for h in hs[1:-1]] + [out.logits[0]]
    return ids, torch.softmax(torch.stack(rows), -1)  # (layers, T, vocab)


lens = []
for p in [RUN, "The capital of France is", "Monday, Tuesday, Wednesday,", "one, two, three, four, five,",
          "The opposite of hot is", "The capital of France is the city of"]:
    ids, P = lens_all(p)
    final = int(P[-1, -1].argmax())
    last = []
    for L in range(NL):
        top = torch.topk(P[L, -1], 5)
        rank = int((P[L, -1] > P[L, -1, final]).sum()) + 1
        last.append({"top": pieces(top.indices.tolist()), "p": [r(x) for x in top.values.tolist()],
                     "pf": r(P[L, -1, final], 5), "rank": rank})
    grid = [[[dec(P[L, t].argmax()), r(P[L, t].max(), 3)] for t in range(ids.shape[1])] for L in range(NL)]
    entry = {"prompt": p, "tokens": pieces(ids[0].tolist()), "final": dec(final), "last": last, "grid": grid}
    paris = tok(" Paris", add_special_tokens=False).input_ids[0]
    entry["paris"] = [r(P[L, -1, paris], 5) for L in range(NL)]
    entry["paris_rank"] = [int((P[L, -1] > P[L, -1, paris]).sum()) + 1 for L in range(NL)]
    lens.append(entry)
D["l06"] = lens

# ---------------------------------------------------------------- lesson 07
print("07 attention ...")


def att_of(ids):
    return [a[0] for a in model(ids, output_attentions=True).attentions]


D["l07"] = {"sentences": []}
for s in [RUN, "The cat chased the mouse because the cat was hungry"]:
    ids = encode(tok, s, device)
    A = att_of(ids)
    D["l07"]["sentences"].append({"text": s, "tokens": pieces(ids[0].tolist()),
                                  "att": [[[[r(x, 3) for x in row] for row in A[l][h].tolist()] for h in range(NH)]
                                          for l in range(NL)]})
torch.manual_seed(0)
L50 = 50
rand = torch.randint(1000, 20000, (1, L50), device=device)
rep = torch.cat([rand, rand], 1)
lg = model(rep).logits[0]
logp = torch.log_softmax(lg[:-1], -1)
rep_loss = -logp[torch.arange(2 * L50 - 1), rep[0, 1:]]
A = att_of(rep)
q = torch.arange(L50, 2 * L50, device=device)
ind = torch.stack([a[:, q, q - L50 + 1].mean(-1) for a in A]).cpu()
prev = torch.stack([a[:, q, q - 1].mean(-1) for a in A]).cpu()
sink = torch.stack([a[:, 1:, 0].mean(-1) for a in A]).cpu()


def best(m, k=1):
    idx = m.flatten().topk(k).indices.tolist()
    return [[i // NH, i % NH] for i in idx]


# small repeated sequence for drawing the patterns
torch.manual_seed(3)
L20 = 16
rand20 = torch.randint(1000, 20000, (1, L20), device=device)
rep20 = torch.cat([rand20, rand20], 1)
A20 = att_of(rep20)
show_heads = best(prev, 1) + best(ind, 2) + best(sink, 1)
D["l07"]["rep"] = {
    "L": L50, "loss": [r(x, 3) for x in rep_loss.tolist()],
    "prev": [[r(x, 3) for x in row] for row in prev.tolist()],
    "ind": [[r(x, 3) for x in row] for row in ind.tolist()],
    "sink": [[r(x, 3) for x in row] for row in sink.tolist()],
    "small_tokens": pieces(rep20[0].tolist()), "small_L": L20,
    "small_heads": [{"layer": l, "head": h,
                     "att": [[r(x, 3) for x in row] for row in A20[l][h].tolist()]} for l, h in show_heads],
}

# ---------------------------------------------------------------- lesson 08
print("08 kv cache timing ...")


def sync():
    if device == "mps":
        torch.mps.synchronize()


def gen_no_cache(ids, n):
    for _ in range(n):
        nxt = model(ids, use_cache=False).logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, nxt], 1)


def gen_cache(ids, n):
    out = model(ids, use_cache=True)
    past, nxt = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)
    for _ in range(n - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)


timing = []
for reps in [5, 20, 40, 80]:
    ids = encode(tok, "Here is a long story about a lighthouse keeper. " * reps, device)
    gen_cache(ids, 3)
    res = {"prompt_tokens": ids.shape[1]}
    for name, fn in [("no_cache", gen_no_cache), ("cache", gen_cache)]:
        sync(); t = time.perf_counter(); fn(ids, 50); sync()
        res[name] = r(50 / (time.perf_counter() - t), 1)
    timing.append(res)
    print("   ", res)
D["l08"] = {"timing": timing, "device": device}

# ---------------------------------------------------------------- lesson 09
print("09 ablations ...")
layers = model.model.layers


def loss_on(ids, start=0):
    logp = torch.log_softmax(model(ids).logits[0, :-1], -1)
    l = -logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
    return l[start:].mean().item()


def skip_layer(i):
    def hook(mod, args, kwargs, output):
        x_in = args[0] if args else kwargs["hidden_states"]
        return (x_in,) + tuple(output[1:]) if isinstance(output, tuple) else x_in
    return layers[i].register_forward_hook(hook, with_kwargs=True)


def silence(pairs):
    by = {}
    for l, h in pairs:
        by.setdefault(l, []).append(h)
    hs = []
    for l, heads in by.items():
        def pre(mod, args, heads=heads):
            x = args[0].clone()
            for h in heads:
                x[..., h * HD:(h + 1) * HD] = 0
            return (x,)
        hs.append(layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
    return hs


text = encode(tok, ("The water cycle describes how water moves around the Earth. Water evaporates from oceans "
                    "and lakes, rises into the atmosphere, cools and condenses into clouds, and then falls back "
                    "to the surface as rain or snow. Rivers carry it back to the sea."), device)
run_ids = encode(tok, RUN, device)
paris = tok(" Paris", add_special_tokens=False).input_ids[0]


def p_paris():
    return torch.softmax(model(run_ids).logits[0, -1], -1)[paris].item()


base = {"text": loss_on(text), "copy": loss_on(rep, L50), "paris": p_paris()}
layer_abl = []
for i in range(NL):
    h = skip_layer(i)
    layer_abl.append({"text": r(loss_on(text) - base["text"], 3), "copy": r(loss_on(rep, L50) - base["copy"], 3),
                      "paris": r(p_paris(), 4)})
    h.remove()
single = []
for l in range(NL):
    row = []
    for h in range(NH):
        hs_ = silence([(l, h)])
        row.append([r(loss_on(rep, L50) - base["copy"], 3), r(p_paris(), 4)])
        [x.remove() for x in hs_]
    single.append(row)
ranked = [(i // NH, i % NH) for i in ind.flatten().argsort(descending=True).tolist()]
others = [p for p in ranked[20:] if p[0] >= 2]
table = []
for k in [1, 3, 5, 10, 20]:
    hs_ = silence(ranked[:k]); a = loss_on(rep, L50); [x.remove() for x in hs_]
    rs = []
    for t in range(5):
        random.seed(t)
        hs_ = silence(random.sample(others, k)); rs.append(loss_on(rep, L50)); [x.remove() for x in hs_]
    table.append({"k": k, "induction": r(a, 2), "random": r(sum(rs) / len(rs), 2)})
zprompt = "Mrs. Zorblat runs a bakery called Quintessa. Every morning, Mrs."
zids = encode(tok, zprompt, device)
zorb = []
for k in [0, 3, 10, 30]:
    hs_ = silence(ranked[:k])
    out = model.generate(zids, max_new_tokens=6, do_sample=False)
    [x.remove() for x in hs_]
    zorb.append({"k": k, "text": tok.decode(out[0, zids.shape[1]:])})
D["l09"] = {"base": {k: r(v, 4) for k, v in base.items()}, "layers": layer_abl, "single": single,
            "table": table, "zprompt": zprompt, "zorb": zorb, "top_ind": [list(p) for p in ranked[:10]]}

# ---------------------------------------------------------------- lesson 10
print("10 rebuild ...")
D["l10"] = {"inv_freq": [r(1.0 / (D["meta"]["rope_theta"] ** (i / HD)), 8) for i in range(0, HD, 2)]}

js = "window.MD = " + json.dumps(D, ensure_ascii=False, separators=(",", ":")) + ";\n"
js = js.replace("\ufffd", "\\ufffd")  # partial-UTF-8 tokens decode to U+FFFD; keep it escaped in the file
(OUT / "model_data.js").write_text(js)
print(f"model_data.js: {len(js)/1e6:.2f} MB")

# ---------------------------------------------------------------- tokenizer for the live playground
print("tokenizer ...")
from huggingface_hub import hf_hub_download  # noqa: E402
tj = json.load(open(hf_hub_download(MODEL_ID, "tokenizer.json")))
vocab = tj["model"]["vocab"]
id2tok = [None] * (max(vocab.values()) + 1)
for t, i in vocab.items():
    id2tok[i] = t
merges = [m.split(" ") if isinstance(m, str) else m for m in tj["model"]["merges"]]
merges = [[vocab[a], vocab[b]] for a, b in merges]
added = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}
tjs = "window.TOK = " + json.dumps({"vocab": id2tok, "merges": merges, "added": added},
                                   ensure_ascii=False, separators=(",", ":")) + ";\n"
tjs = tjs.replace("\ufffd", "\\ufffd")
(OUT / "tokenizer.js").write_text(tjs)
print(f"tokenizer.js: {len(tjs)/1e6:.2f} MB")

# test cases to verify the JS tokenizer against the real one
cases = examples + ["It's 2025 and we're testing tokenizers!!", "naïve café résumé", "    indented\n\tcode()",
                    "e=mc^2; x_1 += 3.14159", "ਪੰਜਾਬੀ தமிழ் 日本語", "don't won't I'll you've", "   ", "a\n\n\nb"]
(OUT / "tokenizer_tests.json").write_text(json.dumps([{"text": t, "ids": tok(t).input_ids} for t in cases],
                                                     ensure_ascii=False))
print("done")
