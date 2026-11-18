# Transformers, Inside Out

**Learn how LLMs work by poking a real one, not by reading equations.**

Most tutorials teach transformers bottom-up: tokenization → embeddings →
attention math → feed-forward → backprop … and you lose the plot long
before you reach the part that actually *does* something.

This project goes the other way: **start with a working model and open it up
one layer at a time.** Every lesson answers a question that the previous one
made you ask. All of it runs locally on a laptop with a tiny 135M-parameter
model, from the inference side (the one exception is a 4-second LoRA run in lesson 13).

```
  black box ──► what comes out? ──► how does it write? ──► what goes in?
     01              02                    03                   04
                                                                 │
  rebuild it ◄── break it ◄── KV cache ◄── attention ◄── logit lens ◄── anatomy
     10            09           08            07             06            05
      │
      └──► Part 2, making it practical:  quantize it ──► open a GGUF ──► LoRA
                                              11              12            13
```

## The lessons

| # | Question | What you do | Aha moment |
|---|----------|-------------|-----------|
| 01 | What does it do? | Generate text, greedy vs. sampling | It *continues* text; it's surprisingly good at code |
| 02 | What actually comes out? | Look at raw logits and probabilities | The model is just "text → probability for each of 49,152 tokens" |
| 03 | How does it write a paragraph? | Write `generate()` yourself; temperature, top-k, top-p | Your 10-line loop gives exactly the same tokens as `model.generate` |
| 04 | What goes in? | Inspect tokenization and the embedding table | Hindi takes 5× more tokens than English; ' king' is next to ' queen' |
| 05 | What's inside? | Count parameters, follow the residual stream | 30 identical "read, compute, add back" blocks; 59% of params are MLP |
| 06 | *When* does it know the answer? | **Logit lens**: decode every layer's state | ' Paris' goes from 0% to 88% in the last 4 layers |
| 07 | What is attention doing? | Pull out attention maps; hunt for heads with jobs | It copies a random sequence it saw once (loss 12.8 → 0.7); **induction heads** found in layers 17–18 |
| 08 | Why is generation fast? | Build a KV-cache loop and time it | 5.6× faster, identical output; why long context eats RAM |
| 09 | Which parts matter? | Use hooks to switch off layers and heads | Silencing 5 of 270 heads destroys copying; "Zorblat" → "Zaroff" → "Rodriguez" |
| 10 | Do I *really* understand it? | Re-implement the forward pass from raw weights in ~60 lines | Your logits match the library to 0.00004, and your code writes stories |
| 11 | Do weights need 16 bits? | **Quantize** every matrix by hand to 8…2 bits; groups, outliers, fragile layers | 8-bit is free (perplexity 19.9 → 20.0), 4-bit still says Paris, 3-bit collapses; one outlier 37× bigger than typical is why groups of 32 exist |
| 12 | What's in a `.gguf` file? | Parse the official SmolLM2 **GGUF** with `struct`, decode Q8_0 / Q5_0 / Q4_K blocks by hand, run it in PyTorch | The tokenizer lives inside the file; our int8 matches llama.cpp's 99.97%; q/k rows are reshuffled for RoPE; this "Q4_K_M" is mostly *not* Q4_K |
| 13 | Can I change what it knows, cheaply? | Write **LoRA** in 15 lines, train 1.8% of the weights for 4 s to move the Eiffel Tower to Rome; dial, merge, QLoRA | It works, but the Statue of Liberty moves to Rome too, until you add anchor sentences; a 4.9 MB adapter, also trainable on a 4-bit base |

Each lesson is a plain `.py` file split into `# %%` cells. It runs as a
script *or* as an interactive notebook (VS Code / Cursor / PyCharm "Run Cell",
or `jupytext` to convert to `.ipynb`). Figures are saved to `outputs/`.

## The interactive explainer (`site/`)

**Read it online: https://querytype.github.io/transformers-inside-out/** (no setup).

`site/index.html` is the companion explainer: every lesson explained in plain
language, with live widgets that use real data from the model. It includes a
working in-browser tokenizer, a temperature/top-k/top-p sampling lab, a
logit-lens grid, an attention explorer, a KV-cache calculator, ablation maps
and a RoPE rotation demo. One sentence ("The Eiffel Tower is located in the
city of" → " Paris") is followed through all ten lessons.

The page's data isn't checked in: you build it from the real model (after the setup below).

```bash
python site/build_data.py            # data for lessons 01-10 (add MODEL_ID=... for another model)
python site/build_data_more.py       # lessons 11-13: runs those lesson scripts and exports their results
open site/index.html                 # works offline, no server needed
```

Maintainers: `bash site/publish_pages.sh` copies the built page to the
`gh-pages` branch, which GitHub Pages serves. `main` stays source-only.

## Setup

```bash
conda create -n transformers_learning python=3.11 -y
conda activate transformers_learning
pip install -r requirements.txt
```

The model (~270 MB) downloads automatically the first time you run a lesson.
Lesson 12 also downloads two official GGUF files of the same model (145 MB + 101 MB).

```bash
python lessons/01_black_box.py        # one lesson
bash run_all.sh                       # all of them (~2 minutes on an M4 Pro, plus first-time downloads)
```

Nothing large lives in this repo: models download on first use into the
Hugging Face cache, and figures, the LoRA adapter and the site data are all
generated locally.

### Hardware: a plain CPU is enough

The model has only 135M parameters, so everything runs fine without a GPU.
Anything with 4 GB of free RAM works. Apple Silicon (MPS), CUDA and CPU are
detected automatically; to force one:

```bash
DEVICE=cpu python lessons/08_kv_cache.py
DEVICE=cpu bash run_all.sh            # all 13 lessons, ~3 minutes on an M4 Pro's CPU
```

On a CPU-only Linux machine, install the CPU build of PyTorch first to skip a
multi-GB CUDA download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Inference numbers are the same on every device (to rounding). The short
training run in lesson 13 can come out slightly differently, which the lesson
discusses.

### Use a different model

Every lesson works with any Llama-style model (lesson 10 needs small tweaks for some):

```bash
MODEL_ID=Qwen/Qwen2.5-0.5B python lessons/06_logit_lens.py
MODEL_ID=HuggingFaceTB/SmolLM2-360M python lessons/07_attention.py
```

Bigger models are a great follow-up: run lessons 06 and 07 and compare *where*
the answer appears and *where* the induction heads live.

## How much math do I need?

Almost none. The whole course uses five formulas, each introduced exactly
when you need it:

* **softmax**: scores → probabilities (lesson 02)
* **temperature**: `softmax(logits / T)` (lesson 02)
* **loss**: `-log(prob of the real next token)` (lesson 02)
* **attention**: `softmax(Q·Kᵀ/√d) · V` (lesson 07)
* **RMSNorm, RoPE, SwiGLU**: one line of code each (lesson 10)
* plus, in part 2, two one-liners: `q = round(w / scale)` (lesson 11) and `W + B @ A` (lesson 13)

## How to get the most out of it

1. **Predict before you run.** Each lesson asks a question. Guess the answer, then run the cell.
2. **Do the "Try:" at the bottom of every lesson.** That's where the learning sticks.
3. **Change the prompts.** Every number in this README came from one specific prompt. Find prompts where the story is different.
4. Lessons 01–04 stand alone. 05–10 build on each other, so do them in order.
   11–13 need only lessons 02, 08 and 10 (perplexity, memory-bound decoding, RoPE).

## Project layout

```
lessons/
  common.py               shared loading / plotting helpers
  01_black_box.py ... 10_rebuild_forward.py
  11_quantization.py  12_gguf.py  13_lora.py
site/
  index.html              the interactive explainer
  build_data.py           exports data for lessons 01-10
  build_data_more.py      exports data for lessons 11-13
  publish_pages.sh        publishes the built site to GitHub Pages
outputs/                  figures written by the lessons
requirements.txt
run_all.sh
```

## Where to go next

* Add a KV cache to your from-scratch forward pass (lesson 10 + lesson 08).
* Port lesson 10 to **MLX**. It's almost line-for-line.
* Make lesson 10's forward pass read its weights straight from a GGUF (lesson 12).
* Redo lesson 13 with `peft`, and make your own GGUFs with llama.cpp's `llama-quantize`;
  compare them with what you built by hand.
* Read about these ideas in depth:
  Anthropic's *A Mathematical Framework for Transformer Circuits* and
  *In-context Learning and Induction Heads*; nostalgebraist's original
  *logit lens* post; Karpathy's *nanoGPT* (for the training side).

## License

[MIT](LICENSE): use it, change it, teach with it. The models it downloads
come with their own licenses (SmolLM2 and its GGUF files are Apache-2.0).
