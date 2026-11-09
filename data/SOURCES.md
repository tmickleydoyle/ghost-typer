# Data sources

`prompts.txt`, `replies.txt` and `wildchat.txt` are produced by `scripts/fetch_coding_prompts.py`.
Every line is English prose typed by people, taken from the datasets below. LLM-generated
messages are excluded: the OASST `synthetic` flag, and WildChat's assistant turns, which are
never used. The script keeps only prose: it removes code blocks, code-looking
lines and markdown, keeps up to 2 sentences per prompt and 5 per reply, and keeps sentences
of 3–24 words.

| File | Dataset | What is used | License | Attribution |
|---|---|---|---|---|
| `prompts.txt` | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | English **user** messages (all turns, reviewed, not deleted, not synthetic) from conversations whose opening prompt is about code | Apache-2.0 | Köpf et al., *OpenAssistant Conversations: Democratizing Large Language Model Alignment*, 2023 |
| `prompts.txt` | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | task descriptions (`text`), all splits | CC-BY-4.0 | Austin et al., *Program Synthesis with Large Language Models*, 2021 |
| `prompts.txt` | [neulab/conala](https://huggingface.co/datasets/neulab/conala) | curated intents (`intent`), paired train + test | MIT | Yin et al., *Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow*, MSR 2018 |
| `replies.txt` | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | English **assistant** messages, written by human volunteers, from the same conversations | Apache-2.0 | as above |
| `wildchat.txt` | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | English **user** turns from conversations whose first user turn is about code. Conversations flagged toxic and turns with redacted PII are skipped. | ODC-BY | Zhao et al., *WildChat: 1M ChatGPT Interaction Logs in the Wild*, ICLR 2024 |

Counts from the last fetch:

| Source | Sentences |
|---|---|
| OASST user turns | 4,169 |
| MBPP | 938 |
| CoNaLa | 2,820 |
| OASST assistant turns | 13,183 |
| WildChat user turns | 127,490 |

After removing duplicates and credential-like sentences: **7,132** lines in `prompts.txt`,
**12,840** in `replies.txt` and **105,373** in `wildchat.txt`.

**Credentials:** some users paste API keys and tokens into prompts. Before filtering, the
WildChat text contained 3 OpenAI keys, 4 Google API keys and 2 Telegram bot tokens. Every
sentence matching a known key format, or containing a 32+ character token that mixes letters
and digits, is dropped. That removed 239 lines, mostly tracebacks and file paths. `test_model.py`
checks that no data line or vocabulary entry matches.

**WildChat caveat:** the user turns are what people typed into ChatGPT. Some users paste text
(including earlier model output) into their messages, so "typed by people" is not perfectly
clean. Only patterns that repeat across many prompts survive pruning, and a suggestion is shown
only above a confidence threshold, which limits how much any single pasted message can
influence what ghost-typer suggests.

The weights contain only n-gram statistics derived from these lines, plus the hand-written
phrases in `../phrases/`. No dataset was used to train anything. 10% of `prompts.txt` (chosen
by content hash) is held out when measuring precision and keystrokes saved
(`scripts/sweep.py`, `scripts/benchmark.py`).
