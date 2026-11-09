# How ghost-typer was built

This model was **compiled, not trained**. No gradient descent, no optimizer, no pretrained
weights, and no other model's outputs were used.
- `build_weights.py` turns text into a standard `LlamaForCausalLM` checkpoint using
  closed-form rules.
- `export_web.py` writes the same kind of network in sparse form for the browser.
- Rebuilds are byte-identical.

The network computes an **interpolated Kneser-Ney 5-gram language model** exactly. It predicts
the next token from the current token and the 3 before it. Witten-Bell smoothing and other
context lengths are supported too; the choice was made by measurement (section 9).

## 1. Source text and tokenizer

| Source | Lines | Weight |
|---|---|---|
| `phrases/*.txt`: everyday English, written for this project by an AI coding assistant (Claude) | 1,139 | 1 |
| `data/prompts.txt`: what people type to an assistant (OASST user turns, MBPP, CoNaLa) | 7,132 | 1 |
| `data/replies.txt`: human-written assistant replies from OASST coding conversations | 12,833 | 1 |
| `data/wildchat.txt`: English user turns from WildChat-1M coding conversations (what people typed into ChatGPT) | 105,243 | 1 |

`scripts/fetch_coding_prompts.py` downloads the data, removes code blocks, code-looking lines and
markdown, and keeps short prose sentences. Licenses and attribution are in `data/SOURCES.md`.

The tokenizer is word-level with GPT-2 style byte-level pre-tokenization:

- NFKC normalization, then lowercase.
- Split on the GPT-2 regex. Words keep their leading space as `Ġ`, and punctuation and
  contractions (`'s`, `'t`, ...) become their own tokens.
- Vocabulary:
  - every word seen ≥ 2 times;
  - capped at **6,000** for the dense model, where each word costs one hidden dimension;
  - uncapped for the browser export: **30,893** words;
  - two specials: `<|endoftext|>` (id 0, used as BOS, EOS and pad) and `<unk>` (id 1);
  - rarer words become `<unk>`, which can appear inside contexts but is never predicted.
- A post-processor prepends BOS, so every prompt starts at position 0 with `<|endoftext|>`.

## 2. The reference n-gram

Each line becomes `[BOS × 4, w1, ..., wn, EOS]`. For every position, the compiler counts which
token follows each context of length k = 0..4 that ends there, weighted by source.

- **Order 0:** add-one-smoothed unigram, with `<unk>` pinned at logit -30.
- **Order k ≥ 1:** interpolate with the next-shorter context `ctx[1:]`. Both smoothing methods
  have the form `p_k(j) = A_j + γ · p_{k-1}(j)`:

  ```
  Witten-Bell:  A_j = c_j / (c + u)                  γ = u / (c + u)
  Kneser-Ney:   A_j = (c_j - d_j) / c,  d_j = min(0.75, c_j)    γ = Σ d_j / c
  ```

  Here `c_j` is the count of token j after the context, `c` the total, and `u` the number of
  distinct next tokens. Kneser-Ney uses **continuation counts** (how many distinct tokens
  precede `ctx + j`) for every order below the top one, as KenLM does. An unseen or pruned
  context backs off: `p_k = p_{k-1}`.

The model must reproduce `p_K(· | longest kept context)`.

## 3. The residual stream as named slots (dense model)

| Dims | Slot | Written by |
|---|---|---|
| 0 | constant `1` (Llama has no biases; this plays that role) | embedding |
| 1–384 | code of the **current** token | embedding |
| 385–768 | code of the token **1 back** | attention head 1 |
| 769–1152 | code of the token **2 back** | attention head 2 |
| 1153–1536 | code of the token **3 back** | attention head 3 |
| 1537–7536 | **logits**, one dim per vocab entry, scaled by 1e-5 | MLP |
| 7537–7559 | unused padding (transformers wants hidden % heads == 0) | — |

**Token codes** are seeded Gaussian vectors in 384 dimensions, normalized to unit length. With
6,000 of them, the largest |cosine| between two different codes is 0.263. That's far enough
below 1 that a "match" (dot = 1) and a "non-match" (dot ≤ 0.27) can't be confused.

## 4. Attention: previous-token heads from RoPE alone

Llama has no absolute positions. The only position signal is RoPE, which rotates query and key
pairs by `position * inv_freq[i]`.

- The **key** for every token is the same constant (read from the constant dim): 1 on the
  first half of each RoPE pair.
- The **query** for head *a* is also constant, with each pair's phase set to
  `-a * inv_freq[i]`. Then

  ```
  score(t, s) = A * Σ_i cos((t - s - a) * inv_freq[i]) / sqrt(head_dim)
  ```

  which peaks exactly when `s = t - a`.
- When `t < a` there is no position `a` back, and the closest position is 0, which is BOS.
  The model then "sees" BOS in that slot, matching the BOS padding used when counting.
- `rope_theta` is picked to maximize the worst-case gap between the target position and every
  other position, for all t < 128 and every head. It came out to 2. `A` scales that gap to
  **30 nats**.
- **Value** = the current-token code. **Output projection** = identity into slot *a*.

All heads share one key/value head (`num_key_value_heads = 1`).

## 5. MLP: one neuron per kept context

Llama's MLP is `down(silu(gate(x)) * up(x))`. The compiler creates one neuron for each **kept**
context (lengths 1–4) and one always-on unigram neuron. For a context `(c_{k-1}, ..., c_0)`
(c_0 = current token):

- **gate** = `100 * (code(c_0)·cur + code(c_1)·prev1 + ... - (k - 0.5) * const)`.
  - If all k tokens match, this is `+50`.
  - If any one fails, it's at most `100 * (0.27 - 0.5) = -23`, and `silu(-23)` ≈ -2e-9,
    effectively off.
- **up** = 1 (the constant dim).
- **down** writes, for each token j seen after the context, `log p_k(j) − log(γ · p_{k-1}(j))`.
  Every unseen token would get the same `log γ`, which softmax ignores, so the column is sparse.

**Why this is exact.** If a context of length k is kept, so are its shorter suffixes, and
their neurons fire too. The unigram neuron writes `log p_0`, and the neurons for lengths 1..K
add their deltas. The sum telescopes to `log p_K(· | longest kept context)` plus a constant.

**Pruning.** A context is kept if it was seen at least `min_count` times (weighted) or comes
from the hand-written phrases. A suffix is always seen at least as often as the contexts
ending in it, so pruning never breaks a backoff chain. The reference n-gram uses the same kept
set, so the model still matches it exactly.
- **Dense model:** `min_count` is the smallest value that fits 400M parameters. That's **37**,
  keeping 12,607 of 2.1M contexts. Each neuron costs 3 × hidden ≈ 22.7k parameters.
  Hand-written phrase contexts are not force-kept here (`force_phrases: false`).
- **Browser export:** `min_count` = **2**, keeping 512,533 of ~2.7M contexts, because a
  sparse neuron costs only a few bytes.

## 6. Norms and output head (dense model)

- The residual has a known fixed norm at each point: |x|² = 2 before attention and 5 after.
  Each RMSNorm weight is `sqrt(norm² / hidden)`, so the norms leave the stream unchanged.
- The MLP writes logits scaled by 1e-5, so they barely change the final norm (< 0.001%). At
  1e-4, the 3,000-entry logit slot shifted the norm enough to cause 1e-2-nat errors on extreme
  logits.
- `lm_head` is `1e5` times the identity on the logit slot.

## 7. Browser export (`web/model.bin`, `web/autocomplete.ts`)

97% of the dense weights are zero. The attention and embedding weights are fixed by
construction, and each MLP neuron is fully described by:
- which tokens its gate matches (its context);
- which logits its down-projection changes, and by how much.

`export_web.py` stores exactly that, plus the unigram neuron and the vocabulary (format v2):
- **Contexts:** each context's token ids, sorted, with the number of logit changes it
  writes.
- **Logit changes:** quantized to **uint8** with one scale per context length. They're
  always ≥ 0, since `log(A + γp) − log(γp)`.
- **Text tables:** the vocabulary and the out-of-vocabulary prefix table are stored as plain
  UTF-8. The prefix table keeps only prefixes of vocabulary words (the only ones ever looked
  up), with rates as uint8 / 255.
- **Context counts:** omitted, since the shipped policy doesn't use them.
- **Size:** 10.7 MB for 512,533 contexts. As float32 with a JSON header it would be about
  34 MB.
- **Quality cost:** quantization was accepted only after held-out typing came out the same:
  90.21% precision and 22.19% keystrokes saved, vs 90.20% / 22.19% unquantized.

`web/autocomplete.ts` binary-searches the sorted contexts for the last 1–4 tokens, adds the
matching neurons' deltas to the unigram logits, and applies softmax. There's no index to build
at load time.
- **Speed:** 2.6 ms median per keystroke (Bun); about 360 ms to load in Chrome.
- **Exactness:** a test exports the dense model's own tables unquantized (`f32`) and checks
  that the runtime's top-5 next tokens and log-probs equal PyTorch's.
- **Freshness:** another test checks that the committed `web/model.bin` is byte-identical to
  a fresh export.
- **Published:** both precisions are on
  [`ghost-typer-web`](https://huggingface.co/tmickleydoyle/ghost-typer-web): `model.bin`
  (uint8) and `model-f32.bin` (float32, same compile). `scripts/stage_hf_web.py` builds that
  repo's folder.

## 8. Confidence policy (`suggest.py`, mirrored by `web/autocomplete.ts`)

Nothing here is learned; the policy reads the compiled probabilities.

1. **Candidates.** The last pre-token is the word being typed (just `Ġ` after a space). Only
   vocabulary tokens starting with it are candidates (`fun` → `function`).
2. **Probability.** Candidate probabilities are renormalized over the candidates, then scaled
   by `1 − P(OOV | prefix)`: the weighted share of source words with this prefix that are
   outside the vocabulary. Without that, a prefix whose only in-vocabulary completion is
   `function` would get 100% even when the user is typing `funky`.
3. **Show** the top candidate only if its probability ≥ τ. A `min_support` threshold is
   available, but the sweep found 0 best.
4. **Extend** greedily word by word while the product of probabilities stays ≥ τ, up to 8
   words, stopping at end-of-text or after `. ? ! :`.

τ = 0.71 by default. It's a runtime argument (`ac.suggest(text, {...ac.policy, tau})`); the
demo's `strict` / `default` / `loose` buttons use 0.90 / 0.71 / 0.60.

## 9. Choosing the settings (`scripts/sweep.py`)

The compile rules are fixed. A handful of compile settings and the threshold were picked by
benchmarking:

1. Hold out 10% of `data/prompts.txt` (711 prompts) and compile without them.
2. Type each held-out prompt one character at a time. After each keystroke, record the
   suggestion chain.
3. Score every threshold: **precision** (shown suggestions that exactly match the next
   characters), **coverage**, and **keystrokes saved** (accepting each correct suggestion with
   one Tab).
4. Keep the setting with the most keystrokes saved at precision ≥ 90%, and an export ≤ 50 MB.

**Round 2** (`docs/sweep.json`, with WildChat; held-out = 1,000 WildChat + 711 other prompts):

| Setting | Keystrokes saved (both sets) |
|---|---|
| WildChat weight 0 / 0.3 / **1** / 3 | 11.4% / 21.0% / **22.2%** / 20.5% |
| words seen ≥ 1 / **2** / 3 times (min_count 2) | 22.2% / **22.3%** / 22.2% |
| min_count 2 / 3 / 5 (words ≥ 2) | **22.3%** / 21.8% / 21.4% |
| context 4 / 5 / 6 / 8 tokens | **22.3%** / 22.1% / 22.0% / 21.6% |

**Round 1** (`docs/sweep_round1.json`, before WildChat; keystrokes saved at ≥ 90% precision,
0.01 threshold grid):

| Setting | Keystrokes saved |
|---|---|
| Witten-Bell, 4-token context (phrases ×5) | 21.7% |
| **Kneser-Ney, 4-token context (phrases ×5)** | **22.5%** |
| Kneser-Ney, 5 / 6 / 8 / 10-token context (phrases ×5) | 22.2% / 22.0% / 21.6% / 21.6% |
| no assistant replies (prompts + phrases only) | 20.1–20.9% |
| assistant replies ×0.3 | 21.8–22.4% |
| **final: KN, 4 tokens, all sources ×1, vocab 8,000, min_count 2** | **23.0%** |
| final settings with 5 / 6 / 8 / 10-token context | 22.8% / 22.7% / 22.4% / 22.4% |

- **Longer contexts:** they lose slightly. With ~21k lines, 5+-token contexts rarely repeat,
  and when they don't match the model is back to the same 4-token prediction.
- **Topic head** (`scripts/experiment_topic.py`): a content-addressed head that would carry
  the most recent language or tool name (python, sql, react, …) from anywhere earlier in the
  prompt.
  - Tested as a closed-form log-ratio bias `λ · log p(w | topic) / p(w)`.
  - At best it left keystrokes saved at 23.0% (22.97% at λ = 0.25), and λ = 1 was worse.
  - It wasn't compiled.

## 10. Verification (`test_model.py`)

- **Hugging Face loading** with `AutoModelForCausalLM`, `AutoTokenizer` and `pipeline` on
  transformers 4.52 and 5.22.
- **Exactness:** dense log-probs match the reference n-gram within 1e-3 nats (observed ~2e-5)
  on 200 phrase prefixes and 50 random sequences of up to 80 tokens.
- **Completions:** a set of greedy completions; unknown words back off cleanly.
- **Dense TypeScript runtime** (`model.ts`): same top-3 tokens and log-probs as PyTorch.
- **Browser runtime:**
  - the same suggestion chains (text, probability, support) as `suggest.py` on 300 random
    prefixes;
  - the same distribution as PyTorch when given the dense model's tables;
  - `model.bin` ≤ 50 MB.

## 11. Changing it

Edit `phrases/*.txt` or `data/*.txt` (one sentence per line), or the settings in
`compile_settings.json`. Then run `python build_weights.py && python export_web.py &&
python test_model.py`. Run `python scripts/sweep.py` to re-check the settings.

| Setting | Meaning |
|---|---|
| `n_prev` | previous tokens attended to (context = n_prev + 1) |
| `smoothing`, `discount` | `kn` or `wb`; Kneser-Ney discount |
| `phrase_weight`, `prompt_weight`, `reply_weight` | how much each source counts |
| `max_vocab`, `min_word_count` | vocabulary cap and minimum word frequency |
| `min_count` / `param_target` | pruning threshold, or the dense budget to fit it to |
| `policy.tau`, `policy.min_support`, `policy.max_words` | confidence policy shipped in `model.bin` |

## History

1. **120 parameters** (hidden size 2): one fixed next word for each of 22 words, with no
   context.
2. **91M parameters:** the compiled 5-gram over 1,139 hand-written phrases.
3. **380M parameters:** adds ~20k human-written coding sentences, with a vocabulary cap and
   count pruning to fit the budget.
4. **tiny-autocomplete v4:** Kneser-Ney smoothing; a confidence-gated ghost-text policy;
   settings chosen by held-out precision and keystrokes saved; a 4.5 MB browser export with a
   zero-dependency runtime.
5. **ghost-typer (current):** renamed. Adds 105k WildChat user prompts (with credential-like
   strings filtered out) and a 30,893-word vocabulary. The browser export is uint8-quantized
   (10.7 MB), with a float32 reference. The demo is boxy, retro and
   types its own examples.
