"""Shared helpers for every lesson. Nothing clever here — just loading and plotting."""
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# Swap the model with:  MODEL_ID=Qwen/Qwen2.5-0.5B python lessons/01_black_box.py
# Any Llama-style model (SmolLM2, Qwen2.5, Llama-3.2) works for every lesson.
MODEL_ID = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolLM2-135M")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    import matplotlib
    matplotlib.use("Agg")  # scripts: save figures to files, don't pop up windows


def pick_device():
    # Force a device with:  DEVICE=cpu python lessons/01_black_box.py
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load(model_id=MODEL_ID, device=None, dtype=torch.float32):
    """Returns (tokenizer, model, device).

    float32 is used on purpose: the model is tiny, and it keeps numbers exact
    enough that our hand-written code (lesson 09) matches the library.
    'eager' attention lets us read the attention weights (lesson 07).
    """
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    torch.set_grad_enabled(False)  # inference only, everywhere
    return tok, model, device


def encode(tok, text, device):
    return tok(text, return_tensors="pt").input_ids.to(device)


def show_fig(fig, name):
    """Save a figure to outputs/ (and display it when running as a notebook)."""
    import matplotlib.pyplot as plt
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [figure saved -> {path.relative_to(ROOT)}]")
    plt.show() if IN_NOTEBOOK else plt.close(fig)


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def show_top_k(probs, tok, k=10):
    """Print the k most likely next tokens from a 1-D probability vector."""
    top = torch.topk(probs, k)
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        bar = "█" * int(p * 50)
        print(f"  {p:6.2%}  {tok.decode([i])!r:18} {bar}")
