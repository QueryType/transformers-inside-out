"""Export the numbers for the site's lessons 11-13 (quantization, GGUF, LoRA).

    python site/build_data_more.py       # writes site/data/more_data.js  (~1 minute)

It doesn't re-implement anything: it *runs* the three lesson scripts and reads
the results they keep in variables. So the site shows exactly what the
lessons print on your machine.
"""
import json, os, runpy, struct, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LESSONS = HERE.parent / "lessons"
sys.path.insert(0, str(LESSONS))
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
D = {"generated": time.strftime("%Y-%m-%d")}
r = lambda x, n=4: round(float(x), n)


def run(name):
    print(f"\n######## running lessons/{name}")
    return runpy.run_path(str(LESSONS / name))


# ---------------------------------------------------------------- lesson 11
g = run("11_quantization.py")
W = g["ORIGINAL"]["model.layers.10.mlp.down_proj.weight"].float().cpu()
row, col = divmod(int(W.abs().argmax()), W.shape[1])
groups = W[row].reshape(-1, 32)
calm = int(groups.abs().amax(1).argmin())            # the group with the smallest max in that row
D["l11"] = {
    "params": g["n_params"],
    "tensor": f"model.layers.10.mlp.down_proj.weight, row {row}",
    "outlier_group": [r(x, 5) for x in groups[col // 32].tolist()],
    "calm_group": [r(x, 5) for x in groups[calm].tolist()],
    "median_abs": r(g["a"].median()), "max_abs": r(g["a"].max()),
    "matrix_err": {lab: r(g["quant_error"](W, 4, grp)) for lab, grp in
                   [("whole matrix", None), ("per row", W.shape[1]), ("per 32", 32)]},
    "base": [r(g["base_ppl"], 2), g["base_text"]],
    "bits": [[b, r(p, 2), r(a), g["texts"][b]] for b, (p, a) in g["results"].items()],
    "groups": [[lab, r(p, 2), r(a)] for lab, (p, a) in g["group_results"].items()],
    "fragile": [[lab, r(sh, 3), r(p, 2), r(a)] for lab, (sh, p, a) in g["fragile"].items()],
}

# ---------------------------------------------------------------- lesson 12
g = run("12_gguf.py")
meta, tensors, data_start = g["meta"], g["tensors"], g["data_start"]
info_bytes = sum(8 + len(n.encode()) + 4 + 8 * len(t["shape"]) + 4 + 8 for n, t in tensors.items())
with open(g["Q8_PATH"], "rb") as f:
    head = f.read(data_start)
unpadded = data_start
while head[unpadded - 1] == 0 and unpadded > data_start - 32:
    unpadded -= 1                                   # strip the alignment padding (approximate)
pid = g["pid"]
t = tensors["token_embd.weight"]
blk = g["tensor_bytes"](g["raw"], data_start, t)[pid * 576 // 32]   # first block of the ' Paris' row
short = lambda v: (f"[{len(v):,} items] " + ", ".join(repr(x) for x in v[:3]) + ", ...") if isinstance(v, list) else repr(v)
census = {}
for name, tt in g["tensors4"].items():
    kind = name.split(".")[-2] if name.startswith("blk.") else name.rsplit(".", 1)[0]
    census.setdefault(kind, {"row": tt["shape"][-1], "types": {}})
    census[kind]["types"][tt["type"]] = census[kind]["types"].get(tt["type"], 0) + 1
D["l12"] = {
    "file": os.path.basename(g["Q8_PATH"]), "size": os.path.getsize(g["Q8_PATH"]),
    "size_q4": os.path.getsize(g["Q4_PATH"]),
    "head24": head[:24].hex(" "), "n_tensors": len(tensors), "n_kv": len(meta),
    "sections": {"header": 24, "metadata": unpadded - info_bytes - 24, "tensor_table": info_bytes,
                 "padding": data_start - unpadded, "weights": os.path.getsize(g["Q8_PATH"]) - data_start},
    "meta": [[k, short(v)] for k, v in meta.items()],
    "layer0": [[n, tt["shape"], tt["type"], tt["offset"]] for n, tt in tensors.items()
               if not n.startswith("blk.") or n.startswith("blk.0.")],
    "types": {k: sum(1 for tt in tensors.values() if tt["type"] == k) for k in {tt["type"] for tt in tensors.values()}},
    "paris_id": int(pid), "paris_block": bytes(blk).hex(),
    "paris_scale": r(np.frombuffer(bytes(blk[:2]), np.float16)[0], 7),
    "paris_orig": [r(x, 5) for x in g["emb_hf"][pid, :32]],
    "emb_rel_err": r(np.linalg.norm(g["emb_gguf"] - g["emb_hf"]) / np.linalg.norm(g["emb_hf"]), 5),
    "same_ints": r(g["same_ints"], 6), "q_err_raw": r(g["q_err_raw"]), "q_err_fixed": r(g["q_err_fixed"], 5),
    "reports": [[k, r(p, 2), txt] for k, (p, txt) in g["reports"].items()],
    "census": census, "params": g["n_params"],
}

# ---------------------------------------------------------------- lesson 13
g = run("13_lora.py")
D["l13"] = {
    "probes": {k: [[p, why, r(a), r(b), txt] for p, why, a, b, txt in rows] for k, rows in g["probe_tables"].items()},
    "dial": [[f, r(a), r(b), txt] for f, a, b, txt in g["dial"]],
    "losses_plain": [r(x, 3) for x in g["losses_plain"]],
    "losses_anchored": [r(x, 3) for x in g["losses_anchored"]],
    "train": g["TRAIN"], "anchors": g["ANCHORS"],
    "adapter_mb": r(g["path"].stat().st_size / 1e6, 2), "model_mb": r(g["n_model"] * 2 / 1e6, 1),
    "qlora": {k: [r(a), r(b), txt] for k, (a, b, txt) in g["qlora"].items()},
}

js = "window.MD2 = " + json.dumps(D, ensure_ascii=False, separators=(",", ":")) + ";\n"
(OUT / "more_data.js").write_text(js)
print(f"\nwrote {OUT / 'more_data.js'} ({len(js) / 1024:.0f} KB)")
