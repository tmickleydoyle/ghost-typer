"""Experiment: does long-range "topic" context help, before compiling it into attention?

Hypothesis: a head that attends to the most recent topic word anywhere earlier in the
prompt (python, sql, react, ...) gives context the 4-token n-gram window can't see.

Closed-form trigger model (no fitting): for each topic word t, count next-word
frequencies at positions where t is the most recent topic word earlier in the line.
The bias added to the n-gram logits is
    lambda * log( p(w | t) / p(w) ),   p(w | t) = (c(t, w) + 1) / (c(t) + V)
kept only where c(t, w) >= MIN_COUNT. In a compiled transformer this would be one
content-addressed attention head (keys = "is topic word" + code, recency via RoPE) and
one MLP neuron per topic word writing these biases.

Scored with the same held-out typing simulation as scripts/sweep.py.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_weights as bw  # noqa: E402
import suggest as sg  # noqa: E402
import sweep  # noqa: E402

TOPIC_WORDS = [
    "python", "javascript", "js", "typescript", "java", "c", "c++", "c#", "rust", "golang", "go", "ruby", "php",
    "kotlin", "swift", "sql", "mysql", "postgresql", "sqlite", "bash", "shell", "powershell", "html", "css", "react",
    "vue", "angular", "node", "django", "flask", "pandas", "numpy", "pytorch", "tensorflow", "excel", "linux", "windows",
    "git", "github", "docker", "kubernetes", "regex", "json", "api", "unity", "blender", "matlab", "r", "lua", "haskell",
]
MIN_COUNT = 3


class TopicSuggester(sg.Suggester):
    def __init__(self, vocab, ngram, oov, topics: set[int], bias: dict[int, tuple[np.ndarray, np.ndarray]], lam: float):
        super().__init__(vocab, ngram, oov)
        self.topics, self.bias, self.lam = topics, bias, lam

    def topic_of(self, ids: list[int]) -> int | None:
        for i in reversed(ids):
            if i in self.topics:
                return i
        return None

    def dist(self, ids):
        t = self.topic_of(ids)
        key = (self.ngram.window(ids), t)
        if key not in self._dist:
            logits, support, _ = self.ngram.predict(ids)
            if t is not None and t in self.bias:
                idx, vals = self.bias[t]
                logits[idx] += self.lam * vals
            p = np.exp(logits - logits.max())
            self._dist[key] = (p / p.sum(), support)
        return self._dist[key]


def topic_bias(vocab: list[str], src: dict, s: bw.Settings, topics: set[int]):
    index = {p: i for i, p in enumerate(vocab)}
    V = len(vocab)
    uni, by_topic = Counter(), defaultdict(Counter)
    for lines, w in bw.weighted_sources(s, src):
        for toks in lines:
            ids = [index.get(p, bw.UNK_ID) for p in toks] + [bw.EOS_ID]
            topic = None
            for i, nxt in enumerate(ids):
                if nxt != bw.UNK_ID:
                    uni[nxt] += w
                    if topic is not None:
                        by_topic[topic][nxt] += w
                if i < len(ids) - 1 and ids[i] in topics:
                    topic = ids[i]
    total = sum(uni.values())
    out = {}
    for t, cnt in by_topic.items():
        ct = sum(cnt.values())
        keep = [(w, c) for w, c in cnt.items() if c >= MIN_COUNT]
        if not keep:
            continue
        idx = np.array([w for w, _ in keep])
        vals = np.array([np.log((c + 1) / (ct + V)) - np.log((uni[w] + 1) / (total + V)) for w, c in keep])
        out[t] = (idx, vals)
    return out


def main() -> None:
    s = bw.load_settings("web")
    src, texts = sweep.train_sources(), sweep.heldout_texts()
    vocab, ngram, _ = bw.compile_tables(s, src)
    oov = sg.oov_prefix_rates(s, src, vocab)
    index = {p: i for i, p in enumerate(vocab)}
    topics = {index[p] for w in TOPIC_WORDS for p in ["Ġ" + w] if p in index}
    bias = topic_bias(vocab, src, s, topics)
    with_topic = sum(1 for t in texts if any(("Ġ" + w) in bw.pieces(t) for w in TOPIC_WORDS))
    print(f"topic words in vocab: {len(topics)}, with biases: {len(bias)}; held-out prompts mentioning one: {with_topic}/{len(texts)}")
    results = []
    for lam in [0.0, 0.25, 0.5, 1.0]:
        sugg = TopicSuggester(vocab, ngram, oov, topics, bias, lam)
        best = sweep.best_policy(sweep.record(sugg, texts))["best"]
        results.append({"lambda": lam, **best})
        print(f"lambda={lam}: saved={best['keystrokes_saved']} prec={best['precision']} cov={best['coverage']} tau={best['tau']}", flush=True)
    (ROOT / "docs" / "experiment_topic.json").write_text(json.dumps({"topic_words": len(topics), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
