"""Choose compile settings and the confidence policy by simulated typing on held-out prompts.

The weights are always produced by the same closed-form compile; this script only
compares a handful of compile settings (context length, smoothing, source weights,
vocabulary, pruning) and policy thresholds, and reports which one saves the most
keystrokes while keeping precision >= TARGET_PRECISION.

Simulation: each held-out prompt (10% of data/prompts.txt and of data/wildchat.txt by
content hash, never compiled in; WildChat capped at WILDCHAT_EVAL prompts) is
"typed" one character at a time. After every keystroke the policy shows ghost text
or nothing. A shown suggestion is correct only if the user's next characters start
with the whole ghost text. Keystrokes saved replays the prompt accepting every
correct suggestion with one Tab press.

    python scripts/sweep.py            # writes docs/sweep.json
"""

import json
import sys
import time
import zlib
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build_weights as bw  # noqa: E402
import suggest as sg  # noqa: E402

TARGET_PRECISION = 0.90
MAX_EXPORT_MB = 50
TAUS = [round(0.5 + 0.01 * i, 2) for i in range(46)]  # 0.50 .. 0.95
SUPPORTS = [0, 2, 5]
WILDCHAT_EVAL = 1000
BASE = bw.Settings(n_prev=3, smoothing="kn", phrase_weight=1, prompt_weight=1, reply_weight=1, wildchat_weight=1,
                   max_vocab=1_000_000, min_word_count=1, min_count=2)


def heldout_texts(name: str = "prompts", limit: int | None = None) -> list[str]:
    """Held-out lines of data/<name>.txt (the same 10% heldout_split removes), normalized."""
    path = bw.DATA_DIR / f"{name}.txt"
    if not path.exists():
        return []
    seen, out = set(), []
    for line in path.read_text().splitlines():
        toks = bw.pieces(line)
        if toks and zlib.crc32(" ".join(toks).encode()) % 100 < 10:
            text = bw.NORMALIZER.normalize_str(line.strip())
            if text not in seen:
                seen.add(text)
                out.append((zlib.crc32(text.encode()), text))
    if limit is not None:  # deterministic sample
        out = sorted(out)[:limit]
    return [t for _, t in out]


def heldout_sets() -> dict[str, list[str]]:
    sets = {"prompts": heldout_texts("prompts")}
    if (wc := heldout_texts("wildchat", WILDCHAT_EVAL)):
        sets["wildchat"] = wc
    return sets


def train_sources() -> dict:
    src = bw.load_sources()
    for name in ["prompts", "wildchat"]:
        if name in src:
            src[name], _ = bw.heldout_split(src[name])
    return src


def record(sugg: sg.Suggester, texts: list[str]) -> list[tuple[str, list]]:
    """For every prompt and every keystroke position, the unthresholded suggestion chain."""
    return [(t, [None] + [sugg.steps(t[:i]) for i in range(1, len(t))]) for t in texts]


def evaluate(records, policy: sg.Policy) -> dict:
    shown = correct = positions = chars = strokes = 0
    for text, steps in records:
        ghosts = [None] + [sg.Suggester.apply(st, policy) for st in steps[1:]]
        for i in range(1, len(text)):
            positions += 1
            if ghosts[i]:
                shown += 1
                correct += text.startswith(ghosts[i], i)
        pos, n = 0, 0
        while pos < len(text):
            pos, n = pos + 1, n + 1
            g = ghosts[pos] if pos < len(text) else ""
            if g and text.startswith(g, pos):
                pos, n = pos + len(g), n + 1
        chars, strokes = chars + len(text), strokes + n
    return {
        "precision": round(correct / shown, 4) if shown else None,
        "coverage": round(shown / positions, 4),
        "keystrokes_saved": round(1 - strokes / chars, 4),
        "shown": shown,
    }


def best_policy(records) -> dict:
    results = []
    for tau in TAUS:
        for sup in SUPPORTS:
            m = evaluate(records, sg.Policy(tau=tau, min_support=sup))
            results.append({"tau": tau, "min_support": sup, **m})
    ok = [r for r in results if r["precision"] is not None and r["precision"] >= TARGET_PRECISION]
    best = max(ok, key=lambda r: r["keystrokes_saved"]) if ok else None
    return {"best": best, "grid": results}


def export_mb(ngram: bw.NGram, vocab: list[str], oov: dict) -> float:
    """Size of web/model.bin for these tables (see export_web.py)."""
    size = sum(len(p.encode()) + 1 for p in vocab) + 8 * len(vocab) + sum(len(p.encode()) + 5 for p in oov)
    for k in range(1, ngram.order + 1):
        for table in ngram.tables[k].values():
            size += 2 * k + 8 + 6 * len(table)
    return round(size / 1e6, 1)


def run(s: bw.Settings, src: dict, sets: dict[str, list[str]]) -> dict:
    """Compile with settings s, type every held-out set, pick the policy on all sets together."""
    t0 = time.time()
    vocab, ngram, stats = bw.compile_tables(s, src)
    oov = sg.oov_prefix_rates(s, src, vocab)
    sugg = sg.Suggester(vocab, ngram, oov)
    recs = {name: record(sugg, texts) for name, texts in sets.items()}
    pol = best_policy([r for rs in recs.values() for r in rs])
    b = pol["best"]
    per_set = {}
    if b:
        chosen = sg.Policy(tau=b["tau"], min_support=b["min_support"])
        per_set = {name: evaluate(rs, chosen) for name, rs in recs.items()}
    out = {
        "settings": asdict(s),
        **stats,
        "export_mb": export_mb(ngram, vocab, oov),
        "best": b,
        "per_set": per_set,
        "seconds": round(time.time() - t0, 1),
        "grid": pol["grid"],
    }
    sets_str = " ".join(f"{n}={m['keystrokes_saved']}/{m['precision']}" for n, m in per_set.items())
    print(
        f"n_prev={s.n_prev} {s.smoothing} w=({s.phrase_weight},{s.prompt_weight},{s.reply_weight},{s.wildchat_weight}) "
        f"V={stats['vocab']} mwc={s.min_word_count} min={s.min_count} | {out['export_mb']} MB | "
        f"saved={(b or {}).get('keystrokes_saved')} prec={(b or {}).get('precision')} cov={(b or {}).get('coverage')} "
        f"tau={(b or {}).get('tau')} sup={(b or {}).get('min_support')} | {sets_str} | {out['seconds']}s",
        flush=True,
    )
    return out


def score(r: dict) -> float:
    if r["best"] is None or r["export_mb"] > MAX_EXPORT_MB:
        return -1
    return r["best"]["keystrokes_saved"]


def main() -> None:
    src, sets = train_sources(), heldout_sets()
    for name, texts in sets.items():
        print(f"held-out {name}: {len(texts)} prompts, {sum(map(len, texts))} characters", flush=True)
    runs = []

    print("== stage A: WildChat weight", flush=True)
    stage = [run(replace(BASE, wildchat_weight=w), src, sets) for w in [0.0, 0.3, 1.0, 3.0]]
    runs += stage
    best = max(runs, key=score)

    print("== stage B: vocabulary x pruning", flush=True)
    base = bw.Settings(**best["settings"])
    stage = [
        run(replace(base, min_word_count=mwc, min_count=mc), src, sets)
        for mwc in [1, 2, 3]
        for mc in [2, 3, 5]
        if (mwc, mc) != (base.min_word_count, base.min_count)
    ]
    runs += stage
    best = max(runs, key=score)

    print("== stage C: context length (more data may favor longer contexts)", flush=True)
    base = bw.Settings(**best["settings"])
    stage = [run(replace(base, n_prev=n), src, sets) for n in [3, 4, 5, 7] if n != base.n_prev]
    runs += stage
    best = max(runs, key=score)

    out = ROOT / "docs" / "sweep.json"
    out.write_text(json.dumps({
        "target_precision": TARGET_PRECISION,
        "max_export_mb": MAX_EXPORT_MB,
        "heldout": {name: len(texts) for name, texts in sets.items()},
        "best": {k: v for k, v in best.items() if k != "grid"},
        "runs": [{k: v for k, v in r.items() if k != "grid"} for r in runs],
        "best_grid": best["grid"],
    }, indent=2) + "\n")
    print("BEST", json.dumps({k: v for k, v in best.items() if k != "grid"}, indent=2))


if __name__ == "__main__":
    main()
