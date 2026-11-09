---
license: mit
language:
- en
base_model: tmickleydoyle/ghost-typer
pipeline_tag: text-generation
library_name: ghost-typer
tags:
- browser
- quantized
- uint8
- typescript
- autocomplete
- compiled
- no-training
- n-gram
datasets:
- allenai/WildChat-1M
- OpenAssistant/oasst2
- google-research-datasets/mbpp
- neulab/conala
---

# ghost-typer-web

**Finishes your sentences. Never takes the byline.**

The browser build of [ghost-typer](https://huggingface.co/tmickleydoyle/ghost-typer): compiled
(not trained) ghost-text autocomplete for coding prompts. It runs entirely on-device with a
zero-dependency TypeScript runtime. **Demo:** https://tmickleydoyle.github.io/ghost-typer/

## Files

| File | Precision | Size | Held-out precision | Keystrokes saved | Use it for |
|---|---|---|---|---|---|
| **`model.bin`** | **uint8** logit changes, one scale per context length | **10.7 MB** | 90.2% | 22.2% | **apps (recommended)** |
| `model-f32.bin` | float32 (unquantized) | 16.8 MB | 90.2% | 22.2% | reference, exactness checks |

Both files hold the same compile: 30,893 words and 512,533 contexts. Quantization changed
held-out results by ≤ 0.01 points (90.21% vs 90.20% precision; 22.19% saved for both),
measured by typing 1,711 held-out prompts one character at a time. The runtime reads either
file.

The runtime itself is `ghost-typer.js` (bundled ESM, ~5 KB) with sources in `src/`. A demo page
(`index.html`, `demo.js`) loads `./model.bin`.

## Use it

Copy `ghost-typer.js`, or `src/` for TypeScript, into your app. Load the model from this repo:

```ts
import { CompiledAutocomplete } from "./ghost-typer.js";

const ac = await CompiledAutocomplete.load(
  "https://huggingface.co/tmickleydoyle/ghost-typer-web/resolve/main/model.bin",
);

input.addEventListener("input", () => {
  const { ghost, prob } = ac.suggest(input.value);   // "" when not confident
  showGhostText(ghost);                              // Tab -> input.value += ghost
});
```

`ac.suggest(text, { ...ac.policy, tau: 0.9 })` gives a stricter threshold (~96% precision, 15.7%
keystrokes saved). The default τ is 0.71.

Per keystroke it takes 2.6 ms median and 4.7 ms p95 (Bun). Loading the uint8 file takes about
360 ms in Chrome.

To try the demo locally: `git clone https://huggingface.co/tmickleydoyle/ghost-typer-web`, then
`python3 -m http.server -d ghost-typer-web 8080`.

## How this relates to the transformers model

The [main repo](https://huggingface.co/tmickleydoyle/ghost-typer) holds the float32
`LlamaForCausalLM` (`model.safetensors`, 400M parameters, 97% zeros) and the compiler. Both
artifacts are compiled from the same text by the same closed-form rules: an interpolated
Kneser-Ney 5-gram, with one MLP neuron per context.
- **What's stored:** the browser model keeps only the non-zero parts, namely each neuron's
  context tokens and the logit changes it writes.
- **Different settings:** it's compiled with browser settings (30,893-word vocabulary vs
  6,000), because extra words cost almost nothing here. Every word would add a hidden
  dimension to the dense model.
- **Not a quantization of `model.safetensors`:** it's a sibling build of the same model
  family. `model.bin` is a uint8 quantization of `model-f32.bin`.
- **Exactness check:** the runtime reproduces PyTorch's next-token distribution within 1e-3
  nats when given the dense model's own tables. That's a test in the source repo.

## Format (`GHST` v2)

```
"GHST" | u32 version (2) | u32 header bytes | header JSON | pad to 8
sections, each 8-byte aligned (offsets in the header):
  vocab            utf-8 pieces joined by "\n"
  oov_keys, oov_vals   prefix -> P(out of vocabulary | prefix), u8 / 255
  base             f32[V]                 always-on unigram neuron
  ctx{k}           u16[n_k * k]           context ids, oldest first, sorted (binary search)
  count{k}         u16[n_k]               logit changes per context
  dids{k}          u16[pairs_k]           next-token ids
  dvals{k}         u8 | f32 [pairs_k]     logit changes (u8: value * scale_k from the header)
```

## Source, data and licenses

- **Compiler, tests and benchmarks:** https://github.com/tmickleydoyle/ghost-typer (also
  mirrored in the main HF repo).
- **License:** MIT.
- **Training text:** human-typed prompts from WildChat-1M (ODC-BY), OpenAssistant oasst2
  (Apache-2.0), MBPP (CC-BY-4.0) and CoNaLa (MIT), plus ~1,100 everyday phrases written by an
  AI coding assistant (Claude). Attribution is in `data/SOURCES.md` in the main repo.
