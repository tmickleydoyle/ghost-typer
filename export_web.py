"""Write web/model.bin: the compiled model in compact, quantized form for the browser.

The dense model.safetensors is ~93% zeros. Each MLP neuron of the compiled
transformer is fully described by its context tokens (which slots its gate
matches) and the logit changes its down-projection writes; the attention and
embedding weights are fixed by construction. This file stores exactly that, so
web/autocomplete.ts can evaluate only the neurons whose context matches: the
same computation as the transformer, without the zeros.

Quantization (format v2):
  - logit changes are always >= 0, stored as uint8 with one scale per context length
    ("u8"), or as float16 ("f16"), or unquantized ("f32", used by the exactness test);
    see compile_settings.json "export"
  - the out-of-vocabulary prefix table keeps only prefixes of vocabulary words (the
    only ones the policy ever looks up), with rates as uint8 / 255
  - context counts are omitted unless the shipped policy uses min_support
  - contexts are sorted, so the runtime binary-searches them instead of building a map

Layout (little-endian):
    "GHST" | u32 version | u32 header bytes | header JSON | pad to 8
    sections (each padded to 8), offsets relative to the first section:
      vocab            utf-8, pieces joined by "\\n"
      oov_keys         utf-8, prefixes joined by "\\n";  oov_vals  u8[n]
      base             f32[V]           unigram neuron (always on)
      ctx{k}  k=1..order  u16/u32[n_k * k]  context ids, oldest first, sorted
      count{k}         u16[n_k]         number of logit changes per context
      dids{k}          u16/u32[pairs_k] next-token ids
      dvals{k}         u8 / f16 / f32[pairs_k]  logit changes (u8: value * scale_k)
      support{k}       f32[n_k]         only if policy.min_support > 0

    python export_web.py      # "web" settings, "policy" and "export" in compile_settings.json
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import build_weights as bw
import suggest as sg

ROOT = Path(__file__).parent
OUT = ROOT / "web" / "model.bin"
MAGIC, VERSION = b"GHST", 2


def load_policy() -> sg.Policy:
    return sg.Policy(**json.loads(bw.SETTINGS_PATH.read_text())["policy"])


def load_scheme() -> str:
    return json.loads(bw.SETTINGS_PATH.read_text()).get("export", {}).get("deltas", "u8")


def quantize_oov(oov: dict[str, float], vocab: list[str]) -> dict[str, int]:
    """Keep prefixes of vocabulary words only (others never have candidates), rates as u8."""
    prefixes = {p[:i] for p in vocab for i in range(1, len(p) + 1)}
    out = {}
    for p, r in oov.items():
        q = int(round(r * 255))
        if q > 0 and p in prefixes:
            out[p] = q
    return out


def quantize(ngram: bw.NGram, vocab: list[str], oov: dict[str, float], scheme: str) -> dict:
    """Quantized tables; also patches `ngram` so the Python reference uses the same numbers."""
    tables, deq = {}, {}
    for k in range(1, ngram.order + 1):
        ctxs = sorted(ngram.tables[k])
        counts, dids, dvals = [], [], []
        for ctx in ctxs:
            d = ngram.delta(ctx)
            counts.append(len(d))
            dids.extend(d.keys())
            dvals.extend(d.values())
        vals = np.array(dvals, dtype=np.float64)
        assert (vals >= 0).all(), "deltas are log(A + g*p) - log(g*p) >= 0"
        if scheme == "u8":
            scale = float(vals.max()) / 255 if len(vals) else 1.0
            q = np.round(vals / scale).clip(0, 255).astype(np.uint8)
            back = q.astype(np.float64) * scale
        elif scheme in ("f16", "f32"):
            scale, q = 1.0, vals.astype(np.float16 if scheme == "f16" else np.float32)
            back = q.astype(np.float64)
        else:
            raise ValueError(scheme)
        tables[k] = {"ctxs": ctxs, "counts": counts, "dids": dids, "q": q, "scale": scale}
        pos = 0
        for ctx, n in zip(ctxs, counts):
            deq[ctx] = (np.array(dids[pos : pos + n], dtype=np.int64), back[pos : pos + n])
            pos += n
    base = ngram.base_logits.astype(np.float32).astype(np.float64)
    ngram.base_logits = base
    ngram.delta_arrays = lambda ctx: deq[ctx]
    oov_q = quantize_oov(oov, vocab)
    return {"tables": tables, "oov_q": oov_q, "oov": {p: q / 255 for p, q in oov_q.items()}}


def export(vocab: list[str], ngram: bw.NGram, oov: dict, policy: sg.Policy, path: Path, scheme: str = "u8") -> dict:
    id_dtype = np.uint16 if len(vocab) < 65536 else np.uint32
    qt = quantize(ngram, vocab, oov, scheme)
    sections, orders = [], []

    def add(name: str, data: bytes) -> None:
        sections.append((name, data))

    add("vocab", "\n".join(vocab).encode())
    add("oov_keys", "\n".join(qt["oov_q"]).encode())
    add("oov_vals", np.array(list(qt["oov_q"].values()), dtype=np.uint8).tobytes())
    add("base", ngram.base_logits.astype(np.float32).tobytes())
    for k, t in qt["tables"].items():
        add(f"ctx{k}", np.array(t["ctxs"], dtype=id_dtype).reshape(-1).tobytes())
        add(f"count{k}", np.array(t["counts"], dtype=np.uint16).tobytes())
        add(f"dids{k}", np.array(t["dids"], dtype=id_dtype).tobytes())
        add(f"dvals{k}", t["q"].tobytes())
        if policy.min_support > 0:
            add(f"support{k}", np.array([ngram.support[c] for c in t["ctxs"]], dtype=np.float32).tobytes())
        orders.append({"k": k, "contexts": len(t["ctxs"]), "pairs": len(t["dids"]), "scale": t["scale"]})

    table, offset = [], 0
    for name, data in sections:
        table.append({"name": name, "offset": offset, "bytes": len(data)})
        offset += -(-len(data) // 8) * 8
    header = {
        "n_prev": ngram.s.n_prev,
        "vocab_size": len(vocab),
        "eos_id": bw.EOS_ID,
        "unk_id": bw.UNK_ID,
        "id_bytes": np.dtype(id_dtype).itemsize,
        "deltas": scheme,
        "orders": orders,
        "policy": asdict(policy),
        "stop_after": sorted(sg.STOP_AFTER),
        "settings": asdict(ngram.s),
        "sections": table,
    }
    head = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC + np.array([VERSION, len(head)], dtype="<u4").tobytes() + head)
        f.write(b"\0" * (-(12 + len(head)) % 8))
        for _, data in sections:
            f.write(data + b"\0" * (-len(data) % 8))
    return {
        "bytes": path.stat().st_size,
        "contexts": sum(o["contexts"] for o in orders),
        "pairs": sum(o["pairs"] for o in orders),
        "oov_prefixes": len(qt["oov_q"]),
        "oov": qt["oov"],
    }


def build(settings: bw.Settings, sources: dict, policy: sg.Policy, path: Path = OUT, scheme: str = "u8"):
    """Compile, quantize and write; returns (info, Suggester using the exported numbers)."""
    vocab, ngram, stats = bw.compile_tables(settings, sources)
    oov = sg.oov_prefix_rates(settings, sources, vocab)
    info = export(vocab, ngram, oov, policy, path, scheme)
    sugg = sg.Suggester(vocab, ngram, info.pop("oov"))
    return {**stats, **info}, sugg


def main() -> None:
    info, _ = build(bw.load_settings("web"), bw.load_sources(), load_policy(), OUT, load_scheme())
    print(json.dumps({**info, "mb": round(info["bytes"] / 1e6, 1), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
