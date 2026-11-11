"""Compile text into a Llama-format autocomplete model. No training.

Every weight is set by a closed-form rule from n-gram counts; nothing is fit by
gradient descent and no other model is used. The result is an exact transformer
implementation of an interpolated n-gram language model (Witten-Bell or
Kneser-Ney) over the source text.

Residual stream layout (hidden_size = 1 + (n_prev + 1) * CODE_DIM + vocab):
    [0]                       constant 1 (acts as a bias, Llama has none)
    cur slot     (CODE_DIM)   code of the current token          <- embedding
    prev1..n     (CODE_DIM)   codes of the 1..n previous tokens   <- attention
    out slot     (vocab)      next-token logits, scaled by OUT_SCALE <- MLP

Token codes are seeded random unit vectors (nearly orthogonal). Attention head
a (a = 1..n_prev) has constant query/key vectors whose RoPE phases make the
score peak at exactly a positions back; its value copies that token's code into
slot prev_a. Positions before the start fall back to position 0 (BOS).

The MLP has one neuron per kept context (1..n_prev+1 tokens) plus one always-on
unigram neuron. A neuron's gate sums code matches across slots minus (k - 0.5),
so it fires only when all k tokens match. It writes
log p(next | ctx) - log p(next | shorter ctx) into the out slot. All suffixes of
the true context fire together, so the sum telescopes to the log probability
under the longest kept context.

Compile settings live in compile_settings.json ("hf" for this dense model,
"web" for the sparse browser export written by export_web.py).
"""

import itertools
import json
import math
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import GenerationConfig, LlamaConfig, PreTrainedTokenizerFast

ROOT = Path(__file__).parent
PHRASE_DIR = ROOT / "phrases"  # hand-written everyday phrases
DATA_DIR = ROOT / "data"  # human-written prompts/replies (scripts/fetch_coding_prompts.py)
SETTINGS_PATH = ROOT / "compile_settings.json"
EOS, UNK = "<|endoftext|>", "<unk>"  # EOS doubles as BOS and pad, like GPT-2
EOS_ID, UNK_ID = 0, 1

SEED = 1234
CODE_DIM = 384
MAX_POS = 128
ATTN_MARGIN = 30.0  # min attention logit gap (nats) between target and any other position
BETA = 100.0  # gate sharpness; matched neurons sit at z = BETA / 2
OUT_SCALE = 1e-5  # keeps the out slot tiny so the final RMSNorm scale stays fixed
UNK_LOGIT = -30.0
ROPE_THETA_CANDIDATES = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 1000.0, 10000.0]

NORMALIZER = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
PRE_TOKENIZER = pre_tokenizers.ByteLevel(add_prefix_space=True)


@dataclass(frozen=True)
class Settings:
    n_prev: int = 3  # previous tokens attended to; context = n_prev + current token
    smoothing: str = "wb"  # "wb" (Witten-Bell), "kn" (interpolated Kneser-Ney) or "mkn" (modified KN)
    discount: float = 0.75  # Kneser-Ney absolute discount ("mkn" estimates its own per order)
    phrase_weight: float = 5.0  # weight of one hand-written phrase (phrases/*.txt)
    prompt_weight: float = 1.0  # weight of one user prompt (data/prompts.txt)
    reply_weight: float = 1.0  # weight of one assistant reply sentence (data/replies.txt)
    wildchat_weight: float = 1.0  # weight of one WildChat user prompt (data/wildchat.txt)
    max_vocab: int = 3000
    min_word_count: float = 3
    # Contexts of length <= 4 are kept if seen >= min_count times. None: the smallest count that
    # fits param_target (dense model). With max_export_mb (browser export, export_web.py),
    # min_count is the floor and the threshold is raised until the export fits.
    min_count: float | None = None
    max_export_mb: float | None = None
    # Contexts longer than 4 tokens need long_factor x the threshold: long contexts count only
    # where they were seen often enough to be reliable.
    long_factor: float = 1.0
    force_phrases: bool = True  # always keep every context from the hand-written phrases
    param_target: int = 400_000_000

    @property
    def order(self) -> int:
        return self.n_prev + 1


def load_settings(name: str) -> Settings:
    return Settings(**json.loads(SETTINGS_PATH.read_text())[name])


# ---------- text ----------


def pieces(text: str) -> list[str]:
    return [p for p, _ in PRE_TOKENIZER.pre_tokenize_str(NORMALIZER.normalize_str(text))]


def load_lines(paths: list[Path]) -> list[list[str]]:
    seen, out = set(), []
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = pieces(line)
            if tuple(toks) not in seen:
                seen.add(tuple(toks))
                out.append(toks)
    return out


def load_phrases() -> list[list[str]]:
    return load_lines(sorted(PHRASE_DIR.glob("*.txt")))


def load_sources() -> dict[str, list[list[str]]]:
    sources = {"phrases": load_phrases()}
    for name in ["prompts", "replies", "wildchat"]:
        if (DATA_DIR / f"{name}.txt").exists():
            sources[name] = load_lines([DATA_DIR / f"{name}.txt"])
    return sources


def heldout_split(lines: list[list[str]], pct: int = 10) -> tuple[list, list]:
    """Deterministic split by content hash: (train, test)."""
    test = [t for t in lines if zlib.crc32(" ".join(t).encode()) % 100 < pct]
    train = [t for t in lines if zlib.crc32(" ".join(t).encode()) % 100 >= pct]
    return train, test


# ---------- counting ----------


def weighted_sources(s: Settings, sources: dict) -> list[tuple[list[list[str]], float]]:
    weights = {"phrases": s.phrase_weight, "prompts": s.prompt_weight, "replies": s.reply_weight,
               "wildchat": s.wildchat_weight}
    return [(sources[name], w) for name, w in weights.items() if w > 0 and name in sources]


def build_vocab(s: Settings, sources: dict) -> list[str]:
    freq = Counter()
    for lines, w in weighted_sources(s, sources):
        for toks in lines:
            for p in toks:
                freq[p] += w
    by_freq = lambda p: (-freq[p], p)  # noqa: E731
    hand = {p for t in sources["phrases"] for p in t} if s.phrase_weight > 0 and s.force_phrases else set()
    rest = sorted((p for p in freq if p not in hand and freq[p] >= s.min_word_count), key=by_freq)
    words = hand | set(rest[: max(0, s.max_vocab - 2 - len(hand))])
    return [EOS, UNK] + sorted(words, key=by_freq)


def padded_events(seq: list[int], order: int):
    """(contexts of length 0..order, next token) for every position of a sequence."""
    # `order` BOS tokens: the last one is the real BOS position the model sees.
    padded = [EOS_ID] * order + seq + [EOS_ID]
    for i in range(order - 1, len(padded) - 1):
        yield [tuple(padded[i - k + 1 : i + 1]) if k else () for k in range(order + 1)], padded[i + 1]


def count_ngrams(weighted: list[tuple[list[int], float]], order: int) -> list[defaultdict]:
    """Every n-gram count in Python dicts. Slow and memory-hungry: only the tests use it, as the
    reference for the numpy counting in compile_tables."""
    counts = [defaultdict(Counter) for _ in range(order + 1)]
    for seq, weight in weighted:
        for ctxs, nxt in padded_events(seq, order):
            if nxt == UNK_ID:  # never learn to predict <unk>
                continue
            for k, ctx in enumerate(ctxs):
                counts[k][ctx][nxt] += weight
    return counts


class NGram:
    """Interpolated n-gram over the kept contexts; the reference the weights must reproduce.

    Witten-Bell: p_k(j) = (c_j + u * p_{k-1}(j)) / (c + u)
    Kneser-Ney:  p_k(j) = (c_j - d_j) / c + (sum_j d_j / c) * p_{k-1}(j),  d_j = min(D, c_j),
                 with continuation counts below the top order (KenLM-style)
    Both have the form p_k(j) = A_j + gamma * p_{k-1}(j), so each context's neuron can write
    log p_k(j) - log(gamma * p_{k-1}(j)) for its seen tokens and 0 elsewhere.
    """

    def __init__(self, tables: list[dict], support: dict, mkn: list | None, vocab_size: int, s: Settings):
        """tables[0]: {next: count}; tables[k]: {context: {next: count}} for kept contexts of length k
        (raw counts at the top order and for Witten-Bell, continuation counts below it for KN);
        support: raw weighted count of each kept context; mkn: per-order (D1, D2, D3+) for "mkn"."""
        self.s, self.V, self.order = s, vocab_size, s.order
        self.tables, self.support, self.mkn_discounts = tables, support, mkn
        if s.smoothing not in ("wb", "kn", "mkn"):
            raise ValueError(s.smoothing)
        for k in range(2, self.order + 1):
            assert all(c[1:] in self.tables[k - 1] for c in self.tables[k]), "backoff chain broken"

        uni = np.zeros(vocab_size)
        for j, c in self.tables[0].items():
            uni[j] = c
        logits = np.log(uni + 1.0)  # add-one smoothing
        logits[UNK_ID] = UNK_LOGIT
        self.base_logits = logits - logits[2:].mean()  # constant shift; softmax-invariant
        p0 = np.exp(self.base_logits - self.base_logits.max())
        self.p0 = p0 / p0.sum()
        self._prob, self._delta, self._stats = {}, {}, {}

    @classmethod
    def from_counts(cls, raw: list[dict], keep: list[set], vocab_size: int, s: Settings) -> "NGram":
        """Reference construction from count_ngrams() dicts (tests only)."""
        order = s.order
        support = {ctx: sum(raw[k][ctx].values()) for k in range(1, order + 1) for ctx in keep[k]}
        mkn = None
        if s.smoothing in ("kn", "mkn"):
            cont = [defaultdict(Counter) for _ in range(order)]
            for k in range(order):
                for ctx, table in raw[k + 1].items():
                    for j in table:
                        cont[k][ctx[1:]][j] += 1
            tables = [cont[0][()]] + [{c: cont[k][c] for c in keep[k]} for k in range(1, order)]
            tables.append({c: raw[order][c] for c in keep[order]})
            if s.smoothing == "mkn":
                full = [None] + [cont[k] for k in range(1, order)] + [raw[order]]
                mkn = [None] + [mkn_discounts([c for t in full[k].values() for c in t.values()]) for k in range(1, order + 1)]
        else:
            tables = [raw[0][()]] + [{c: raw[k][c] for c in keep[k]} for k in range(1, order + 1)]
        return cls(tables, support, mkn, vocab_size, s)

    def clear_caches(self) -> None:
        self._prob, self._stats, self._delta = {}, {}, {}

    def _table_stats(self, ctx: tuple):
        if ctx not in self._stats:
            table = self.tables[len(ctx)][ctx]
            total = sum(table.values())
            if self.s.smoothing == "wb":
                u = len(table)
                self._stats[ctx] = (table, {j: c / (total + u) for j, c in table.items()}, u / (total + u))
            else:
                if self.s.smoothing == "mkn":
                    D = self.mkn_discounts[len(ctx)]
                    d = {j: min(D[min(int(round(c)), 3) - 1] if c >= 0.5 else D[0], c) for j, c in table.items()}
                else:
                    d = {j: min(self.s.discount, c) for j, c in table.items()}
                self._stats[ctx] = (table, {j: (c - d[j]) / total for j, c in table.items()}, sum(d.values()) / total)
        return self._stats[ctx]

    def prob(self, j: int, ctx: tuple) -> float:
        key = (j, ctx)
        if key not in self._prob:
            if not ctx:
                p = self.p0[j]
            else:
                lower = self.prob(j, ctx[1:])
                if ctx in self.tables[len(ctx)]:
                    _, a, gamma = self._table_stats(ctx)
                    p = a.get(j, 0.0) + gamma * lower
                else:
                    p = lower
            self._prob[key] = p
        return self._prob[key]

    def delta(self, ctx: tuple) -> dict[int, float]:
        """log p_k - log p_{k-1}, minus the constant log(gamma) shared by unseen tokens."""
        if ctx not in self._delta:
            table, a, gamma = self._table_stats(ctx)
            out = {}
            for j in table:
                q = gamma * self.prob(j, ctx[1:])
                out[j] = math.log(a[j] + q) - math.log(q)
            self._delta[ctx] = out
        return self._delta[ctx]

    def delta_arrays(self, ctx: tuple) -> tuple[np.ndarray, np.ndarray]:
        d = self.delta(ctx)
        return np.fromiter(d.keys(), dtype=np.int64, count=len(d)), np.fromiter(d.values(), dtype=np.float64, count=len(d))

    def contexts(self) -> list[tuple]:
        return [ctx for k in range(1, self.order + 1) for ctx in sorted(self.tables[k])]

    def window(self, ids: list[int]) -> tuple:
        """Last order tokens (oldest .. current); positions before 0 read BOS like the attention heads."""
        t = len(ids) - 1
        return tuple(ids[max(t - a, 0)] for a in range(self.s.n_prev, -1, -1))

    def predict(self, ids: list[int]) -> tuple[np.ndarray, float, int]:
        """(logits up to a constant, support of the longest matched context, its length)."""
        window = self.window(ids)
        logits = self.base_logits.copy()
        support, length = 0.0, 0
        for k in range(1, self.order + 1):
            ctx = window[-k:]
            if ctx in self.tables[k]:
                idx, vals = self.delta_arrays(ctx)
                logits[idx] += vals
                support, length = self.support[ctx], k
        return logits, support, length

    def reference_logits(self, ids: list[int]) -> torch.Tensor:
        return torch.from_numpy(self.predict(ids)[0])


def mkn_discounts(counts) -> tuple[float, float, float]:
    """Chen & Goodman (D1, D2, D3+) from the count-of-counts n1..n4 of all (unpruned) counts an
    order's tables are built from, as KenLM does (counts rounded to integers)."""
    r = np.round(np.asarray(counts, dtype=np.float64))
    n = {i: int((r == i).sum()) for i in range(1, 5)}
    if not all(n[i] for i in range(1, 5)):
        return (0.5, 1.0, 1.5)
    y = n[1] / (n[1] + 2 * n[2])
    ds = (1 - 2 * y * n[2] / n[1], 2 - 3 * y * n[3] / n[2], 3 - 4 * y * n[4] / n[3])
    return tuple(min(max(d, 0.0), float(i + 1)) for i, d in enumerate(ds))


def compile_tables(s: Settings, sources: dict, vocab: list[str] | None = None, export_size=None):
    """vocab, NGram and stats for settings s (min_count=None fits the dense param budget)."""
    # export_size(contexts_by_len, pairs_by_len) -> bytes: required with s.max_export_mb.
    vocab = vocab or build_vocab(s, sources)
    V, order = len(vocab), s.order
    index = {p: i for i, p in enumerate(vocab)}
    weighted = [([index.get(p, UNK_ID) for p in toks], w) for lines, w in weighted_sources(s, sources) for toks in lines]
    win, nxt, wts = _windows(weighted, order)
    del weighted
    forced_win = None
    if s.phrase_weight > 0 and s.force_phrases:
        forced_win, _, _ = _windows([([index[p] for p in toks], 1.0) for toks in sources["phrases"]], order, drop_unk=False)

    # One level per context length k: unique (context, next) pairs in lexicographic order.
    kn = s.smoothing in ("kn", "mkn")
    levels, mkn = [], [None] * (order + 1)
    uni = None
    for k in range(1, order + 1):
        pairs, raw = _group(np.concatenate([win[:, order - k :], nxt[:, None]], axis=1), wts)
        if kn:
            # Continuation counts one level down: distinct tokens preceding each (ctx[1:], next).
            # Those pairs are exactly level k-1's pairs, in the same order.
            low, cont = _group(pairs[:, 1:])
            if k == 1:
                uni = dict(zip(low[:, 0].tolist(), cont.tolist()))
            else:
                assert len(cont) == len(levels[-1]["nxt"])
                levels[-1]["vals"] = cont
                mkn[k - 1] = mkn_discounts(cont) if s.smoothing == "mkn" else None
            del low
        first = _starts(pairs[:, :k])
        lvl = {
            "ctx": np.ascontiguousarray(pairs[first, :k]),
            "off": np.append(first, len(pairs)),
            "nxt": pairs[:, k].copy(),
            "vals": raw,  # replaced by continuation counts for KN below the top order
            "total": np.add.reduceat(raw, first),
        }
        lvl["forced"] = np.zeros(len(first), dtype=bool)
        if forced_win is not None:
            lvl["forced"] = np.isin(_rows_as_void(lvl["ctx"]), _rows_as_void(np.unique(forced_win[:, order - k :], axis=0)))
        levels.append(lvl)
        del pairs
    if s.smoothing == "mkn":
        mkn[order] = mkn_discounts(levels[-1]["vals"])
    if not kn:
        j, c = _group(nxt[:, None], wts)
        uni = dict(zip(j[:, 0].tolist(), c.tolist()))
    del win, nxt, wts

    def kept(m: float) -> list[np.ndarray]:
        return [lv["forced"] | (lv["total"] >= (m if k <= 4 else m * s.long_factor)) for k, lv in enumerate(levels, 1)]

    def fits(m: float) -> bool:
        masks = kept(m)
        if s.max_export_mb is not None:
            n = [int(mk.sum()) for mk in masks]
            pairs = [int(np.diff(lv["off"])[mk].sum()) for lv, mk in zip(levels, masks)]
            return export_size(n, pairs) <= s.max_export_mb * 1e6
        return dense_params(V, s.n_prev, 1 + sum(int(mk.sum()) for mk in masks)) <= s.param_target

    min_count = s.min_count
    if min_count is None or s.max_export_mb is not None:
        # Smallest threshold (among the context counts that occur, >= the min_count floor) that fits.
        floor = min_count or 0
        cands = np.unique(np.concatenate([lv["total"][~lv["forced"]] for lv in levels]))
        cands = cands[cands > max(floor, 0)]
        if floor and (not len(cands) or cands[0] > floor):
            cands = np.concatenate([[floor], cands])
        lo, hi = 0, len(cands)
        while lo < hi:
            mid = (lo + hi) // 2
            if fits(float(cands[mid])):
                hi = mid
            else:
                lo = mid + 1
        if lo == len(cands):
            raise SystemExit("hand-written contexts alone exceed the size budget")
        min_count = float(cands[lo])

    tables, support = [uni], {}
    for lv, mk in zip(levels, kept(min_count)):
        ndist = np.diff(lv["off"])[mk]
        pair_mask = np.repeat(mk, np.diff(lv["off"]))
        nx, vals = lv["nxt"][pair_mask].tolist(), lv["vals"][pair_mask].tolist()
        ctxs = list(map(tuple, lv["ctx"][mk].tolist()))
        table, pos = {}, 0
        for ctx, n in zip(ctxs, ndist.tolist()):
            table[ctx] = dict(zip(nx[pos : pos + n], vals[pos : pos + n]))
            pos += n
        tables.append(table)
        support.update(zip(ctxs, lv["total"][mk].tolist()))
    ngram = NGram(tables, support, mkn if s.smoothing == "mkn" else None, V, s)
    stats = {
        "vocab": V,
        "min_context_count": min_count,
        "contexts_seen": sum(len(lv["total"]) for lv in levels),
        "contexts_kept": sum(len(t) for t in tables[1:]),
    }
    return vocab, ngram, stats


def _windows(weighted: list[tuple[list[int], float]], order: int, drop_unk: bool = True):
    """Every counting event (same as padded_events): (window of `order` tokens, oldest .. current,
    next token, weight) as numpy arrays. Events predicting <unk> are dropped."""
    lens = np.fromiter((len(seq) for seq, _ in weighted), dtype=np.int64, count=len(weighted))
    pad = [EOS_ID] * order
    flat = np.fromiter(itertools.chain.from_iterable(itertools.chain(pad, seq, (EOS_ID,)) for seq, _ in weighted),
                       dtype=np.int32, count=int(lens.sum()) + len(lens) * (order + 1))
    events = lens + 1
    starts = np.concatenate([[0], np.cumsum(lens + order + 1)[:-1]])
    first_event = np.concatenate([[0], np.cumsum(events)[:-1]])
    pos = np.repeat(starts + order - 1, events) + (np.arange(int(events.sum())) - np.repeat(first_event, events))
    win = flat[pos[:, None] + np.arange(-order + 1, 1)]
    nxt = flat[pos + 1]
    wts = np.repeat(np.fromiter((w for _, w in weighted), dtype=np.float64, count=len(weighted)), events)
    if drop_unk:  # never learn to predict <unk>
        ok = nxt != UNK_ID
        win, nxt, wts = win[ok], nxt[ok], wts[ok]
    return win, nxt, wts


def _starts(rows: np.ndarray) -> np.ndarray:
    """Indices where a run of equal rows begins (rows sorted)."""
    new = np.ones(len(rows), dtype=bool)
    if len(rows) > 1:
        np.any(rows[1:] != rows[:-1], axis=1, out=new[1:])
    return np.flatnonzero(new)


def _group(rows: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows in lexicographic order with their summed weights (or number of occurrences).
    lexsort is stable, so each group's weights are summed in event order."""
    order = np.lexsort(rows.T[::-1])
    rows = rows[order]
    if weights is not None:
        weights = weights[order]
    first = _starts(rows)
    sums = np.add.reduceat(weights, first) if weights is not None else np.diff(np.append(first, len(rows)))
    return rows[first], sums


def _rows_as_void(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


# ---------- dense transformer ----------


def hidden_size(V: int, n_prev: int) -> int:
    # transformers validates hidden_size % num_heads == 0; pad with unused zero dims.
    raw = 1 + (n_prev + 1) * CODE_DIM + V
    return -(-raw // (8 * n_prev)) * (8 * n_prev)


def dense_params(V: int, n_prev: int, neurons: int) -> int:
    D, h = hidden_size(V, n_prev), CODE_DIM
    return 2 * V * D + (2 * n_prev + 2) * h * D + 3 * neurons * D + 3 * D


def rope_attention_scale(head_dim: int, n_prev: int) -> tuple[float, torch.Tensor, float]:
    """Pick rope_theta maximizing the worst-case attention margin; return (theta, inv_freq, q_amp)."""
    best = None
    for theta in ROPE_THETA_CANDIDATES:
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).double() / head_dim))
        offsets = torch.arange(-MAX_POS, MAX_POS + 1, dtype=torch.float64)
        kernel = torch.cos(offsets[:, None] * inv_freq[None, :]).sum(1)  # K(x), index x + MAX_POS
        worst = math.inf
        for lag in range(1, n_prev + 1):
            for t in range(MAX_POS):
                s_target = max(t - lag, 0)
                xs = torch.tensor([t - s - lag for s in range(t + 1)]) + MAX_POS
                scores = kernel[xs]
                others = torch.cat([scores[:s_target], scores[s_target + 1 :]])
                if len(others):
                    worst = min(worst, (scores[s_target] - others.max()).item())
        if best is None or worst > best[0]:
            best = (worst, theta, inv_freq)
    worst, theta, inv_freq = best
    assert worst > 0, "no rope_theta separates positions"
    # score = q_amp * sum_i cos(...) / sqrt(head_dim)  (key amplitude 1)
    q_amp = ATTN_MARGIN * math.sqrt(head_dim) / worst
    return theta, inv_freq, q_amp


def build_dense(vocab: list[str], ngram: NGram, s: Settings):
    V, m, n_prev = len(vocab), CODE_DIM, s.n_prev
    contexts = ngram.contexts()
    N = 1 + len(contexts)
    CONST, CUR = 0, 1
    prev_slot = [1 + a * m for a in range(n_prev + 1)]  # prev_slot[0] is the cur slot
    OUT = 1 + (n_prev + 1) * m
    n_heads, h = n_prev, m
    D = hidden_size(V, n_prev)

    gen = torch.Generator().manual_seed(SEED)
    codes = torch.randn(V, m, generator=gen, dtype=torch.float64)
    codes /= codes.norm(dim=1, keepdim=True)
    gram = codes @ codes.T
    gram.fill_diagonal_(0)
    max_coherence = gram.abs().max().item()
    assert max_coherence < 0.4, max_coherence

    embed = torch.zeros(V, D, dtype=torch.float64)
    embed[:, CONST] = 1.0
    embed[:, CUR : CUR + m] = codes

    theta, inv_freq, q_amp = rope_attention_scale(h, n_prev)
    half = h // 2
    q_proj = torch.zeros(n_heads * h, D, dtype=torch.float64)
    for a in range(n_heads):
        lag = a + 1
        q_proj[a * h : a * h + half, CONST] = q_amp * torch.cos(lag * inv_freq)
        q_proj[a * h + half : (a + 1) * h, CONST] = -q_amp * torch.sin(lag * inv_freq)
    k_proj = torch.zeros(h, D, dtype=torch.float64)
    k_proj[:half, CONST] = 1.0
    v_proj = torch.zeros(h, D, dtype=torch.float64)
    v_proj[:, CUR : CUR + m] = torch.eye(m, dtype=torch.float64)
    o_proj = torch.zeros(D, n_heads * h, dtype=torch.float64)
    for a in range(n_heads):
        o_proj[prev_slot[a + 1] : prev_slot[a + 1] + m, a * h : (a + 1) * h] = torch.eye(m, dtype=torch.float64)

    # The MLP matrices dominate memory, so they are built directly in float32.
    gate = torch.zeros(N, D, dtype=torch.float32)
    up = torch.zeros(N, D, dtype=torch.float32)
    down = torch.zeros(D, N, dtype=torch.float32)
    on = (BETA / 2) * torch.sigmoid(torch.tensor(BETA / 2, dtype=torch.float64)).item()  # silu(BETA/2)
    up[:, CONST] = 1.0
    gate[0, CONST] = BETA / 2  # always-on unigram neuron
    down[OUT : OUT + V, 0] = torch.from_numpy(OUT_SCALE * ngram.base_logits / on).float()
    for n, ctx in enumerate(contexts, start=1):
        k = len(ctx)
        for a, tok in enumerate(reversed(ctx)):  # a = 0 current, a = 1 previous, ...
            gate[n, prev_slot[a] : prev_slot[a] + m] = BETA * codes[tok]
        gate[n, CONST] = -BETA * (k - 0.5)
        for j, d in ngram.delta(ctx).items():
            down[OUT + j, n] = OUT_SCALE * d / on

    lm_head = torch.zeros(V, D, dtype=torch.float64)
    lm_head[torch.arange(V), OUT + torch.arange(V)] = 1.0 / OUT_SCALE

    # RMSNorm weights undo the norm for the (fixed) input norms: |x|^2 = 2 before
    # attention, 1 + 1 + n_prev after it.
    attn_norm = math.sqrt(2 / D)
    mlp_norm = math.sqrt((2 + n_prev) / D)
    state = {
        "model.embed_tokens.weight": embed,
        "model.layers.0.input_layernorm.weight": torch.full((D,), attn_norm),
        "model.layers.0.self_attn.q_proj.weight": q_proj,
        "model.layers.0.self_attn.k_proj.weight": k_proj,
        "model.layers.0.self_attn.v_proj.weight": v_proj,
        "model.layers.0.self_attn.o_proj.weight": o_proj,
        "model.layers.0.post_attention_layernorm.weight": torch.full((D,), mlp_norm),
        "model.layers.0.mlp.gate_proj.weight": gate,
        "model.layers.0.mlp.up_proj.weight": up,
        "model.layers.0.mlp.down_proj.weight": down,
        "model.norm.weight": torch.full((D,), mlp_norm),
        "lm_head.weight": lm_head,
    }
    state = {k: v.to(torch.float32).contiguous() for k, v in state.items()}

    config = LlamaConfig(
        vocab_size=V,
        hidden_size=D,
        intermediate_size=N,
        num_hidden_layers=1,
        num_attention_heads=n_heads,
        num_key_value_heads=1,
        head_dim=h,
        hidden_act="silu",
        max_position_embeddings=MAX_POS,
        rms_norm_eps=1e-10,
        rope_theta=theta,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    info = {
        "hidden": D,
        "neurons": N,
        "rope_theta": theta,
        "max_code_coherence": round(max_coherence, 3),
        "params": sum(t.numel() for t in state.values()),
    }
    return state, config, info


def build_tokenizer(vocab: list[str]) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({p: i for i, p in enumerate(vocab)}, unk_token=UNK))
    tok.normalizer = NORMALIZER
    tok.pre_tokenizer = PRE_TOKENIZER
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.TemplateProcessing(single=f"{EOS} $A", special_tokens=[(EOS, 0)])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=EOS,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=EOS,
        clean_up_tokenization_spaces=False,
        model_max_length=MAX_POS,
    )


def compile_model():
    """Build the dense ("hf") model in memory (used by the tests for the n-gram reference)."""
    s = load_settings("hf")
    sources = load_sources()
    vocab, ngram, stats = compile_tables(s, sources)
    state, config, info = build_dense(vocab, ngram, s)
    lines = {name: len(v) for name, v in sources.items()}
    return lines, vocab, ngram, state, config, {**stats, **info}


def main() -> None:
    lines, vocab, _, state, config, info = compile_model()
    assert info["params"] <= 1_000_000_000, info

    save_file(state, ROOT / "model.safetensors", metadata={"format": "pt"})
    config.save_pretrained(ROOT)
    cfg_path = ROOT / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["architectures"] = ["LlamaForCausalLM"]
    cfg["rope_theta"] = info["rope_theta"]  # transformers 4.x reads this key, 5.x reads rope_parameters
    cfg["torch_dtype"] = "float32"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    GenerationConfig(bos_token_id=0, eos_token_id=0, pad_token_id=0, do_sample=False, max_new_tokens=16).save_pretrained(ROOT)

    build_tokenizer(vocab).save_pretrained(ROOT)
    # transformers 5 writes "TokenizersBackend", which 4.x can't resolve;
    # PreTrainedTokenizerFast loads on both. 4.x also emits token_type_ids
    # by default, which Llama's generate() rejects, and overrides the
    # pre-tokenizer's add_prefix_space unless the config states it.
    tok_cfg_path = ROOT / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text())
    tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    tok_cfg["add_prefix_space"] = True
    tok_cfg["model_input_names"] = ["input_ids", "attention_mask"]
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")

    print(json.dumps({"lines": lines, **info}, indent=2))


if __name__ == "__main__":
    main()
