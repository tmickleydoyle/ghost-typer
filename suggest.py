"""Confidence-gated ghost-text suggestions (reference implementation).

web/autocomplete.ts implements the same policy for the browser; the tests check
they agree. Nothing here is learned: the policy reads the compiled model's
next-token probabilities and only suggests when they clear fixed thresholds.

Given the text typed so far:
  1. The last pre-token is the word being typed (just "Ġ" after a space). Only
     vocabulary tokens that start with it are candidates ("fun" -> "function").
  2. Candidate probabilities are renormalized over the candidates and scaled by
     the chance that a word with this prefix is in the vocabulary at all
     (counted from the source text), so rare out-of-vocabulary words don't
     produce confident wrong completions.
  3. The suggestion is shown only if that probability >= tau and the longest
     matched context was seen >= min_support times. It then extends greedily,
     word by word, stopping at end-of-text or after . ? ! :
     - ext set: while each next word's own probability >= ext;
     - ext None: while the product of probabilities stays >= tau.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from build_weights import EOS_ID, UNK_ID, NGram, Settings, pieces, weighted_sources

STOP_AFTER = {".", "?", "!", ":"}
MAX_WORDS = 8


@dataclass(frozen=True)
class Policy:
    tau: float = 0.6  # min probability of the first word (and of the whole suggestion if ext is None)
    min_support: float = 2  # min (weighted) count of the longest matched context
    max_words: int = MAX_WORDS
    ext: float | None = None  # min probability of each word after the first


@dataclass(frozen=True)
class Step:
    text: str  # characters this step adds to the ghost text
    prob: float  # probability of the suggestion up to and including this step
    support: float  # count of the longest context matched for this step


def _byte_decoder() -> dict[str, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()


def decode_piece(p: str) -> str:
    return bytes(BYTE_DECODER[c] for c in p).decode("utf-8", errors="replace")


def oov_prefix_rates(s: Settings, sources: dict, vocab: list[str]) -> dict[str, float]:
    """P(word is out of vocabulary | word starts with prefix), from weighted source counts.

    Only prefixes with a non-zero rate are stored; prefixes of in-vocabulary words that
    never start an OOV word are omitted (rate 0).
    """
    in_vocab = set(vocab)
    # Only prefixes of vocabulary words are ever looked up, so longer ones are skipped (a few
    # huge junk tokens in the data would otherwise add millions of prefixes).
    maxlen = max(map(len, vocab))
    total, oov = Counter(), Counter()
    for lines, w in weighted_sources(s, sources):
        for toks in lines:
            for p in toks:
                for i in range(1, min(len(p), maxlen) + 1):
                    total[p[:i]] += w
                    if p not in in_vocab:
                        oov[p[:i]] += w
    return {pre: oov[pre] / total[pre] for pre in oov}


class Suggester:
    def __init__(self, vocab: list[str], ngram: NGram, oov_rates: dict[str, float]):
        self.vocab, self.ngram, self.oov = vocab, ngram, oov_rates
        self.index = {p: i for i, p in enumerate(vocab)}
        self.text = [decode_piece(p) for p in vocab]
        self._dist, self._cands = {}, {}

    def dist(self, ids: list[int]) -> tuple[np.ndarray, float]:
        key = self.ngram.window(ids)
        if key not in self._dist:
            if len(self._dist) >= 512:  # each entry is a vocab-sized array; bound memory like the TS runtime
                self._dist.clear()
            logits, support, _ = self.ngram.predict(ids)
            p = np.exp(logits - logits.max())
            self._dist[key] = (p / p.sum(), support)
        return self._dist[key]

    def candidates(self, partial: str) -> np.ndarray:
        if partial not in self._cands:
            if len(self._cands) >= 5000:
                self._cands.clear()
            self._cands[partial] = np.array(
                [i for i, p in enumerate(self.vocab) if i > UNK_ID and p.startswith(partial)], dtype=np.int64
            )
        return self._cands[partial]

    def steps(self, text: str, max_words: int = MAX_WORDS) -> list[Step]:
        """Greedy suggestion chain with cumulative probabilities (before any thresholding)."""
        pcs = pieces(text)
        if not pcs:
            return []
        partial = pcs[-1]
        cands = self.candidates(partial)
        if not len(cands):
            return []
        ids = [EOS_ID] + [self.index.get(p, UNK_ID) for p in pcs[:-1]]
        probs, support = self.dist(ids)
        pc = probs[cands]
        best = int(cands[pc.argmax()])
        prob = (1.0 - self.oov.get(partial, 0.0)) * pc.max() / pc.sum()
        out = [Step(self.text[best][len(decode_piece(partial)) :], prob, support)]
        ids.append(best)
        while self.vocab[ids[-1]] not in STOP_AFTER and len(out) < max_words:
            probs, support = self.dist(ids)
            nxt = int(probs.argmax())
            if nxt == EOS_ID:
                break
            prob *= probs[nxt]
            out.append(Step(self.text[nxt], prob, support))
            ids.append(nxt)
        return out

    @staticmethod
    def apply(steps: list[Step], policy: Policy) -> str:
        ghost = ""
        for i, st in enumerate(steps[: policy.max_words]):
            if st.support < policy.min_support:
                break
            if i and policy.ext is not None:
                if st.prob / steps[i - 1].prob < policy.ext:
                    break
            elif st.prob < policy.tau:
                break
            ghost += st.text
        return ghost

    def suggest(self, text: str, policy: Policy = Policy()) -> str:
        return self.apply(self.steps(text, policy.max_words), policy)
