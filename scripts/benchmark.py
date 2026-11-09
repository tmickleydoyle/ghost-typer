"""Measure compile cost, inference speed, exactness and held-out quality.

Writes docs/benchmarks.json. Run from the repo root after build_weights.py and export_web.py:
    python scripts/benchmark.py
"""

import hashlib
import json
import os
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_weights as bw  # noqa: E402
import export_web  # noqa: E402
import suggest as sg  # noqa: E402
import sweep  # noqa: E402

PROMPTS = [
    "Once upon a", "I hope this email", "The quick brown", "Thank you for", "Looking forward to",
    "In Python,", "How to convert a", "Write a python function to", "You can use the", "Here is an example of",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def hardware() -> dict:
    cpu = ""
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    cpu = cpu or f"{platform.system()} {platform.machine()}"
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        ram = 0
    return {"cpu": cpu, "cores": os.cpu_count(), "ram_gb": round(ram / 2**30), "gpu": "none used",
            "python": platform.python_version(), "torch": torch.__version__}


def timed_runs(cmd: list[str], out: Path, n: int = 2) -> dict:
    times, hashes = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
        times.append(time.perf_counter() - t0)
        hashes.append(sha256(out))
    return {
        "seconds": round(min(times), 1),
        "runs": [round(t, 1) for t in times],
        "byte_identical_rebuild": len(set(hashes)) == 1,
        "sha256": hashes[0],
        "mb": round(out.stat().st_size / 1e6, 2),
    }


def compile_dense() -> dict:
    runs = timed_runs([sys.executable, "build_weights.py"], ROOT / "model.safetensors")
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    sources = bw.load_sources()
    t1 = time.perf_counter()
    s = bw.load_settings("hf")
    vocab, ngram, stats = bw.compile_tables(s, sources)
    t2 = time.perf_counter()
    state, _, info = bw.build_dense(vocab, ngram, s)
    t3 = time.perf_counter()
    return {
        **runs,
        "peak_memory_gb": round(peak / 2**30 if sys.platform == "darwin" else peak / 2**20, 1),
        "tokenize_seconds": round(t1 - t0, 2),
        "count_seconds": round(t2 - t1, 2),
        "build_tensors_seconds": round(t3 - t2, 2),
        "source_lines": sum(len(v) for v in sources.values()),
        "params": info["params"],
        "nonzero_params": sum(int((t != 0).sum()) for t in state.values()),
        **stats,
    }


def compile_web() -> dict:
    runs = timed_runs([sys.executable, "export_web.py"], ROOT / "web" / "model.bin")
    header_len = int.from_bytes((ROOT / "web" / "model.bin").read_bytes()[8:12], "little")
    header = json.loads((ROOT / "web" / "model.bin").read_bytes()[12 : 12 + header_len])
    return {**runs, "contexts": sum(o["contexts"] for o in header["orders"]), "vocab": header["vocab_size"],
            "deltas": header["deltas"]}


def torch_inference() -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(ROOT)
    model = AutoModelForCausalLM.from_pretrained(ROOT).eval()
    load = time.perf_counter() - t0
    with torch.no_grad():
        ids = tok("Write a python function to", return_tensors="pt")
        model(**ids)  # warm-up
        t0 = time.perf_counter()
        for _ in range(20):
            model(**ids)
        forward_ms = (time.perf_counter() - t0) / 20 * 1000
        new_tokens, t0 = 0, time.perf_counter()
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt")
            out = model.generate(**ids, max_new_tokens=16)
            new_tokens += out.shape[1] - ids["input_ids"].shape[1]
        gen = time.perf_counter() - t0
    return {"load_seconds": round(load, 1), "next_word_ms": round(forward_ms, 1),
            "generate_tokens_per_second": round(new_tokens / gen, 1), "threads": torch.get_num_threads()}


def web_latency(texts: list[str]) -> dict | None:
    """Per-keystroke suggestion time of web/autocomplete.ts on held-out prefixes (Bun)."""
    if not shutil.which("bun"):
        return None
    rng = random.Random(0)
    prefixes = [t[: rng.randint(1, len(t) - 1)] for t in rng.sample(texts, 500)]
    out = json.loads(subprocess.run(["bun", "web/cli.ts", "--json", *prefixes], cwd=ROOT, check=True,
                                    capture_output=True, text=True).stdout)
    ms = sorted(r["ms"] for r in out[50:])  # skip JIT warm-up
    return {"runtime": "web/autocomplete.ts on Bun", "median_ms": round(statistics.median(ms), 2),
            "p95_ms": round(ms[int(0.95 * len(ms))], 2), "max_ms": round(ms[-1], 2)}


def exactness() -> float:
    from transformers import AutoModelForCausalLM

    _, vocab, ngram, _, _, _ = bw.compile_model()
    model = AutoModelForCausalLM.from_pretrained(ROOT).eval()
    rng, worst = random.Random(0), 0.0
    with torch.no_grad():
        for _ in range(50):
            ids = [0] + [rng.randrange(2, len(vocab)) for _ in range(rng.randint(1, 80))]
            got = model(torch.tensor([ids])).logits[0, -1].double().log_softmax(-1)
            worst = max(worst, (got - ngram.reference_logits(ids).log_softmax(-1)).abs().max().item())
    return worst


def heldout_typing(name: str, sets: dict[str, list[str]], src: dict) -> dict:
    """Simulated typing on held-out prompts with the shipped policy, compiled without them.

    "web" is scored through the quantized export (the numbers the browser actually uses).
    """
    import tempfile

    s = bw.load_settings(name)
    policy = export_web.load_policy()
    if name == "web":
        with tempfile.TemporaryDirectory() as tmp:
            _, sugg = export_web.build(s, src, policy, Path(tmp) / "model.bin", export_web.load_scheme())
    else:
        vocab, ngram, _ = bw.compile_tables(s, src)
        sugg = sg.Suggester(vocab, ngram, sg.oov_prefix_rates(s, src, vocab))
    recs = {k: sweep.record(sugg, texts) for k, texts in sets.items()}
    out = {"policy": policy.__dict__, "heldout": {k: len(v) for k, v in sets.items()}}
    for t in [policy.tau, 0.8, 0.9, 0.95]:
        p = sg.Policy(tau=t, min_support=policy.min_support)
        out[f"tau={t}"] = {"all": sweep.evaluate([r for rs in recs.values() for r in rs], p),
                           **{k: sweep.evaluate(rs, p) for k, rs in recs.items()}}
    return out


def heldout_next_token(texts_src: dict) -> dict:
    """Top-1/top-3 next-token accuracy of the dense model's tables on held-out prompts."""
    s = bw.load_settings("hf")
    _, test = bw.heldout_split(bw.load_sources()["prompts"])
    vocab, ngram, _ = bw.compile_tables(s, texts_src)
    index = {p: i for i, p in enumerate(vocab)}
    top1 = top3 = total = oov = 0
    for toks in test:
        ids = [0] + [index.get(p, 1) for p in toks] + [0]
        for t in range(len(ids) - 1):
            target = ids[t + 1]
            total += 1
            if target == 1:
                oov += 1
                continue
            ranked = ngram.reference_logits(ids[: t + 1]).topk(3).indices.tolist()
            top1 += ranked[0] == target
            top3 += target in ranked
    return {"heldout_lines": len(test), "heldout_tokens": total, "next_token_top1": round(top1 / total, 3),
            "next_token_top3": round(top3 / total, 3), "oov_rate": round(oov / total, 3)}


def main() -> None:
    results = {"hardware": hardware()}
    print("compiling dense model (2 runs)...", flush=True)
    results["compile_dense"] = compile_dense()
    print("compiling web export (2 runs)...", flush=True)
    results["compile_web"] = compile_web()
    print("inference...", flush=True)
    results["pytorch"] = torch_inference()
    sets, src = sweep.heldout_sets(), sweep.train_sources()
    results["web_keystroke_latency"] = web_latency([t for v in sets.values() for t in v])
    results["max_logprob_error_vs_ngram"] = exactness()
    print("held-out quality...", flush=True)
    results["heldout_typing_web"] = heldout_typing("web", sets, src)
    results["heldout_typing_dense_tables"] = heldout_typing("hf", sets, src)
    results["heldout_next_token_dense"] = heldout_next_token(src)
    (ROOT / "docs" / "benchmarks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
