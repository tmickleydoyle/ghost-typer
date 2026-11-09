---
license: mit
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- llama
- compiled
- no-training
- autocomplete
- n-gram
- code
- browser
datasets:
- allenai/WildChat-1M
- OpenAssistant/oasst2
- google-research-datasets/mbpp
- neulab/conala
widget:
- text: You can use the
---

# ghost-typer

**Finishes your sentences. Never takes the byline.**

Ghost-text autocomplete for prompts to a coding assistant. It's **compiled, not trained**: no
gradient descent, no pretrained weights, no other model. Every number comes from a closed-form
rule in [`build_weights.py`](build_weights.py) applied to ~126k lines of human-typed text
([sources](data/SOURCES.md)).

It only speaks when it's sure. A suggestion appears only above a confidence threshold whose
precision is measured on held-out prompts.

- **Live demo:** https://tmickleydoyle.github.io/ghost-typer/ (runs entirely in your tab)
- **Code:** https://github.com/tmickleydoyle/ghost-typer
- **Transformers model (this repo):** https://huggingface.co/tmickleydoyle/ghost-typer
- **Browser model:** https://huggingface.co/tmickleydoyle/ghost-typer-web

## Variants

| Repo / file | Precision | Size | What it is |
|---|---|---|---|
| **this repo:** `model.safetensors` | **float32** (canonical) | 1.60 GB | the compiled `LlamaForCausalLM`; loads with `transformers` |
| [`ghost-typer-web`](https://huggingface.co/tmickleydoyle/ghost-typer-web): `model.bin` | **uint8** (recommended for apps) | 10.7 MB | browser build; zero-dependency TypeScript runtime |
| [`ghost-typer-web`](https://huggingface.co/tmickleydoyle/ghost-typer-web): `model-f32.bin` | float32 | 16.8 MB | unquantized reference of the browser build |

- **Two compiles, one model family:** the browser build is a sibling compile, not a
  quantization of `model.safetensors`. It uses a bigger vocabulary (30,893 vs 6,000 words)
  because words cost almost nothing in its sparse format.
- **Quantization cost:** within the browser build, uint8 vs float32 changes held-out
  precision and keystrokes saved by ≤ 0.01 points.
- **No half-precision transformers model:** its logits are scaled by 1e-5 to keep the network
  exact, which float16/bf16 would break.

| | Browser model (`ghost-typer-web/model.bin`) | Transformers model (`model.safetensors`) |
|---|---|---|
| Use it for | apps: on-device, no server | `transformers`, `pipeline`, research |
| Size | **10.7 MB** (uint8-quantized) | 1.60 GB, 400M params (97% zeros) |
| Vocabulary / contexts | **30,893 words** / 512,533 contexts | 6,000 words / 12,607 contexts |
| Speed | **2.6 ms median, 4.7 ms p95 per keystroke** | 65 ms per next token (PyTorch CPU) |

Both are the same kind of network. The browser model stores only the non-zero parts of the
compiled transformer; see [how it works](#how-it-works).

## What it suggests

Output of the browser model (uint8) at the default threshold (0.71). Gray is the ghost text;
*(quiet)* means it suggests nothing.

| Typed | Ghost text | Confidence |
|---|---|---|
| `write a python fun` | `ction to` | 94% |
| `write a react comp` | `onent` | 90% |
| `how do i conv` | `ert` | 99% |
| `i'm getting an err` | `or` | 99% |
| `can you explain the differ` | `ence between` | 95% |
| `how to install` | *(quiet)* | |

## How often it's right

The model is compiled **without** 10% of the prompts. Each held-out prompt is then typed one
character at a time, and after every keystroke the model either shows ghost text or stays
quiet. This is measured with `python scripts/benchmark.py` on the shipped quantized file.
- **Precision:** the share of shown suggestions that exactly match what the user typed next.
- **Coverage:** the share of keystrokes where something is shown.
- **Keystrokes saved:** the share saved by accepting every correct suggestion with Tab.

| Threshold | Precision | Coverage | Keystrokes saved | Real chat prompts only (WildChat) |
|---|---|---|---|---|
| **0.71 (default)** | **90.2%** | 37.4% | **22.2%** | 89.9% precision, 21.7% saved |
| 0.80 | 93.4% | 32.6% | 19.5% | 92.9%, 19.1% |
| 0.90 (`strict`) | 96.4% | 26.2% | 15.7% | 96.3%, 15.5% |
| 0.95 | 97.7% | 20.6% | 12.3% | 97.4%, 12.1% |

Held-out sets: 1,000 WildChat prompts plus 711 OASST, MBPP and CoNaLa prompts. Adding
WildChat raised keystrokes saved on real chat prompts from 6.8% to 21.7%. The threshold is a
runtime setting, so an app can trade coverage for precision without recompiling.

## Use it in your app

`web/autocomplete.ts` and `tokenizer.ts` are zero-dependency TypeScript with no DOM access. They
run on the main thread, in a Web Worker, or in Bun/Node. A single bundled `ghost-typer.js` is in
[`ghost-typer-web`](https://huggingface.co/tmickleydoyle/ghost-typer-web).

```ts
import { CompiledAutocomplete } from "./web/autocomplete.ts";

const ac = await CompiledAutocomplete.load(
  "https://huggingface.co/tmickleydoyle/ghost-typer-web/resolve/main/model.bin",
);

input.addEventListener("input", () => {
  const { ghost, prob } = ac.suggest(input.value);       // "" when not confident
  showGhostText(ghost);                                  // Tab -> input.value += ghost
});

ac.suggest(text, { ...ac.policy, tau: 0.9 });            // stricter: ~96% precision
```

Run the demo locally. It's a static page:

```bash
git clone https://github.com/tmickleydoyle/ghost-typer && cd ghost-typer
python3 -m http.server 8080 -d web          # open http://localhost:8080
bun web/cli.ts "write a python fun"         # or from the terminal
```

## Use it with transformers

```python
from transformers import pipeline

gen = pipeline("text-generation", model="tmickleydoyle/ghost-typer")
gen("You can use the", do_sample=False)[0]["generated_text"]   # 'You can use the following code:'
```

`model.safetensors` is a plain `LlamaForCausalLM` (no `trust_remote_code`). Its logits are the
compiled n-gram's log-probabilities, matching the reference to within 2e-5 nats. On
transformers 4.x, also pass `add_special_tokens=True`; 4.x pipelines don't prepend BOS by
default. For confidence-gated ghost text in Python, use `suggest.py`.

## Performance: compile, don't train

Measured on a 10-core Apple Silicon laptop (32 GB RAM), CPU only. Raw numbers are in
[`docs/benchmarks.json`](docs/benchmarks.json).

| | |
|---|---|
| Compile the browser model (126,347 lines) | **30 s** |
| Compile the 400M-parameter transformers model | **21 s** (peak memory 4.3 GB) |
| GPU hours / compute cost | **0 / $0** |
| Reproducible | byte-identical rebuilds (both artifacts) |
| Browser model in Chrome | about 360 ms to load (local); a few ms per keystroke |

## How it works

- **The network:** an exact transformer implementation of an **interpolated Kneser-Ney
  5-gram model**. It predicts the next token from the current token plus the 3 before it,
  backing off to shorter contexts.
- **Attention:** 3 heads with constant queries and keys. Their RoPE phases make each head copy
  the token exactly 1, 2 or 3 positions back into its own slot.
- **MLP:** one neuron per kept context. It fires only when all its tokens match and adds
  `log p(next | context) − log p(next | shorter context)` to the logits. These additions
  telescope to the longest known context's log-probability.
- **Browser model:** stores each neuron's context and logit changes. The changes are always
  ≥ 0, so they're stored as uint8 with one scale per context length. That changes held-out
  results by ≤ 0.01 points. The runtime binary-searches the sorted contexts, evaluating only
  the neurons that match.
- **Confidence policy** (`suggest.py`, mirrored by `web/autocomplete.ts`):
  - **Partial words:** completes the word being typed, among vocabulary words with that
    prefix. Its probability is scaled by the chance that a word with this prefix is in the
    vocabulary at all.
  - **When to show:** only if that probability is ≥ τ.
  - **Extension:** grows word by word while the combined probability stays ≥ τ.
- **Settings were measured, not hand-tuned:** `scripts/sweep.py` compiled each candidate and
  kept the most keystrokes saved at ≥ 90% precision. It covered source weights, vocabulary
  size, pruning and context length.
  - **Longer contexts** (5–8 tokens) lost both times, even with 5× more data.
  - **A "topic" attention head** didn't help either (`scripts/experiment_topic.py`).

Details, formulas and the proof of exactness are in [docs/how-it-works.md](docs/how-it-works.md).

## Limitations

- **Short memory and a lookup:** it sees 4 tokens and repeats frequent patterns. It doesn't
  understand code, and it doesn't write code (code was removed from the data).
- **Narrow domain:** it's strongest on how people phrase coding requests, and quiet on most
  other text.
- **Lowercase:** input is lowercased, so suggestions are lowercase.
- **Data:** WildChat prompts are what people typed into ChatGPT, and some users pasted text
  into them. Sentences containing anything that looks like a credential (API keys, tokens,
  private keys) or a long random string are dropped, and a test enforces it. ~1,100 everyday
  phrases in `phrases/` were written by an AI coding assistant (Claude). See
  [data/SOURCES.md](data/SOURCES.md).

## Rebuild and verify

```bash
pip install -r requirements.txt
python scripts/fetch_coding_prompts.py   # downloads data/*.txt (WildChat: ~3.4 GB, cached)
python build_weights.py                  # transformers model   ("hf" in compile_settings.json)
python export_web.py                     # browser model        ("web" + "policy" + "export")
python scripts/stage_hf_web.py           # ghost-typer-web repo folder (uint8 + float32) -> dist/hf-web
bun build web/demo.ts --outfile web/demo.js --target browser
python test_model.py                     # HF loading, exactness, browser runtime == Python == PyTorch
python scripts/sweep.py                  # settings comparison -> docs/sweep.json
python scripts/benchmark.py              # -> docs/benchmarks.json
```

Tested with transformers 4.52 and 5.22, Bun 1.3, Node 24 and Chrome.

## Licenses

The model, code and hand-written phrases are MIT. The files in `data/` contain excerpts from
WildChat-1M (ODC-BY), OpenAssistant oasst2 (Apache-2.0), MBPP (CC-BY-4.0) and CoNaLa (MIT); see
[data/SOURCES.md](data/SOURCES.md) for attribution.

## Files

| File | Purpose |
|---|---|
| `web/index.html`, `web/demo.ts`, `web/demo.js` | the demo page |
| `web/autocomplete.ts`, `tokenizer.ts`, `web/cli.ts` | browser runtime and CLI (`web/model.bin` is on GitHub and in `ghost-typer-web`) |
| `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer*.json` | transformers model (`model.safetensors` is on Hugging Face only) |
| `build_weights.py`, `export_web.py`, `compile_settings.json`, `suggest.py` | compiler, settings and confidence policy |
| `phrases/`, `data/` | source text and attribution |
| `scripts/`, `docs/` | data download, settings comparison, benchmarks, experiments, write-up |
| `test_model.py` | verification suite |
| `model.ts`, `autocomplete.ts` | dense-model TypeScript runtime and CLI |
