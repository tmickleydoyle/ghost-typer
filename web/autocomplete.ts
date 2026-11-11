// ghost-typer browser runtime (zero dependencies, no DOM; works on the main thread,
// in a Web Worker, or in Bun/Node).
//
// web/model.bin holds the compiled transformer in compact form (see export_web.py):
// the always-on unigram neuron plus, for every MLP neuron, its context tokens and the
// (uint8-quantized) logit changes it writes. Evaluating only the neurons whose context
// matches gives the same next-token distribution as the dense model, without the zeros.
// The confidence policy mirrors suggest.py.
//
//   const ac = await CompiledAutocomplete.load("/model.bin");
//   ac.suggest("write a python fun").ghost   // "ction to"

import { decodePieces, preTokenize } from "../tokenizer.ts";

// tau: min probability of the first word (and of the whole suggestion when ext is null/absent);
// ext: min probability of each word after the first.
export type Policy = { tau: number; min_support: number; max_words: number; ext?: number | null };
export type Step = { text: string; prob: number; support: number };
export type Suggestion = { ghost: string; prob: number; steps: Step[] };

type IdArray = Uint16Array | Uint32Array;
type Order = {
  k: number;
  n: number;
  ctx: IdArray; // n * k ids, rows sorted lexicographically
  offsets: Uint32Array; // n + 1
  dids: IdArray;
  dvals: Float32Array;
  support: Float32Array | null;
};

const MAGIC = "GHST";
const VERSION = 2;

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

export class CompiledAutocomplete {
  readonly vocab: string[];
  readonly nPrev: number;
  readonly order: number;
  readonly eosId: number;
  readonly unkId: number;
  readonly policy: Policy;
  readonly bytes: number;
  readonly contexts: number;
  private text: string[];
  private pieceIndex: Map<string, number>;
  private base: Float32Array;
  private orders: Order[];
  private oov: Map<string, number>;
  private stopAfter: Set<string>;
  private candCache = new Map<string, Int32Array>();
  private distCache = new Map<string, { probs: Float64Array; support: number }>();

  /** Fetch (with optional progress callback) or wrap an already-loaded buffer. */
  static async load(source: string | ArrayBuffer, onProgress?: (fraction: number) => void): Promise<CompiledAutocomplete> {
    if (source instanceof ArrayBuffer) return new CompiledAutocomplete(source);
    // Revalidate so a re-published model.bin at the same URL isn't served stale (304 if unchanged).
    const resp = await fetch(source, { cache: "no-cache" });
    if (!resp.ok || !resp.body) throw new Error(`failed to fetch ${source}: ${resp.status}`);
    const total = Number(resp.headers.get("content-length")) || 0;
    const reader = resp.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) onProgress?.(received / total);
    }
    const bytes = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) {
      bytes.set(c, off);
      off += c.length;
    }
    onProgress?.(1);
    return new CompiledAutocomplete(bytes.buffer);
  }

  constructor(buf: ArrayBuffer) {
    const view = new DataView(buf);
    const td = new TextDecoder();
    if (td.decode(new Uint8Array(buf, 0, 4)) !== MAGIC) throw new Error("not a ghost-typer model.bin");
    const version = view.getUint32(4, true);
    if (version !== VERSION) throw new Error(`model.bin version ${version}, runtime expects ${VERSION}`);
    const headerLen = view.getUint32(8, true);
    const header = JSON.parse(td.decode(new Uint8Array(buf, 12, headerLen)));
    const dataStart = 12 + headerLen + ((8 - ((12 + headerLen) % 8)) % 8);
    const sections = new Map<string, { offset: number; bytes: number }>();
    for (const s of header.sections) sections.set(s.name, s);
    const raw = (name: string) => {
      const s = sections.get(name);
      return s ? new Uint8Array(buf, dataStart + s.offset, s.bytes) : null;
    };
    const typed = <T>(name: string, Ctor: { new (b: ArrayBuffer, o: number, n: number): T; BYTES_PER_ELEMENT: number }): T => {
      const s = sections.get(name)!;
      return new Ctor(buf, dataStart + s.offset, s.bytes / Ctor.BYTES_PER_ELEMENT);
    };
    const Ids = header.id_bytes === 2 ? Uint16Array : Uint32Array;

    this.vocab = td.decode(raw("vocab")!).split("\n");
    this.text = this.vocab.map((p) => decodePieces(p));
    this.pieceIndex = new Map(this.vocab.map((p, i) => [p, i]));
    const oovKeys = raw("oov_keys")!.length ? td.decode(raw("oov_keys")!).split("\n") : [];
    const oovVals = raw("oov_vals")!;
    this.oov = new Map(oovKeys.map((p, i) => [p, oovVals[i] / 255]));
    this.nPrev = header.n_prev;
    this.order = this.nPrev + 1;
    this.eosId = header.eos_id;
    this.unkId = header.unk_id;
    this.policy = header.policy;
    this.stopAfter = new Set(header.stop_after);
    this.bytes = buf.byteLength;
    this.base = typed("base", Float32Array);
    this.orders = header.orders.map((o: { k: number; contexts: number; pairs: number; scale: number }) => {
      const counts = typed(`count${o.k}`, Uint16Array);
      const offsets = new Uint32Array(o.contexts + 1);
      for (let i = 0; i < o.contexts; i++) offsets[i + 1] = offsets[i] + counts[i];
      const dvals = new Float32Array(o.pairs);
      if (header.deltas === "u8") {
        const q = raw(`dvals${o.k}`)!;
        for (let i = 0; i < o.pairs; i++) dvals[i] = q[i] * o.scale;
      } else if (header.deltas === "f32") {
        dvals.set(typed(`dvals${o.k}`, Float32Array));
      } else {
        const q = typed(`dvals${o.k}`, Uint16Array);
        for (let i = 0; i < o.pairs; i++) dvals[i] = halfToFloat(q[i]);
      }
      return {
        k: o.k,
        n: o.contexts,
        ctx: typed(`ctx${o.k}`, Ids),
        offsets,
        dids: typed(`dids${o.k}`, Ids),
        dvals,
        support: sections.has(`support${o.k}`) ? typed(`support${o.k}`, Float32Array) : null,
      };
    });
    this.contexts = this.orders.reduce((n, o) => n + o.n, 0);
  }

  /** Last `order` token ids, oldest first; positions before 0 read BOS like the attention heads. */
  private window(ids: number[]): number[] {
    const t = ids.length - 1;
    const w: number[] = [];
    for (let a = this.nPrev; a >= 0; a--) w.push(ids[Math.max(t - a, 0)]);
    return w;
  }

  /** Row index of `key` (length o.k) in the sorted contexts, or -1. */
  private find(o: Order, key: number[]): number {
    let lo = 0;
    let hi = o.n - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      let cmp = 0;
      for (let j = 0; j < o.k && cmp === 0; j++) cmp = o.ctx[mid * o.k + j] - key[j];
      if (cmp === 0) return mid;
      if (cmp < 0) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

  /** Next-token distribution and support of the longest matched context (Infinity if not exported). */
  dist(ids: number[]): { probs: Float64Array; support: number } {
    const window = this.window(ids);
    const cacheKey = window.join(",");
    const hit = this.distCache.get(cacheKey);
    if (hit) return hit;
    const logits = Float64Array.from(this.base);
    let support = 0;
    for (const o of this.orders) {
      const i = this.find(o, window.slice(-o.k));
      if (i < 0) continue;
      for (let j = o.offsets[i]; j < o.offsets[i + 1]; j++) logits[o.dids[j]] += o.dvals[j];
      support = o.support ? o.support[i] : Infinity;
    }
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    let z = 0;
    const probs = new Float64Array(logits.length);
    for (let i = 0; i < logits.length; i++) z += probs[i] = Math.exp(logits[i] - max);
    for (let i = 0; i < probs.length; i++) probs[i] /= z;
    if (this.distCache.size > 2000) this.distCache.clear();
    const out = { probs, support };
    this.distCache.set(cacheKey, out);
    return out;
  }

  private candidates(partial: string): Int32Array {
    let c = this.candCache.get(partial);
    if (!c) {
      const out: number[] = [];
      for (let i = this.unkId + 1; i < this.vocab.length; i++) if (this.vocab[i].startsWith(partial)) out.push(i);
      c = Int32Array.from(out);
      if (this.candCache.size > 5000) this.candCache.clear();
      this.candCache.set(partial, c);
    }
    return c;
  }

  /** Greedy suggestion chain with cumulative probabilities, before thresholding. */
  steps(text: string, maxWords = this.policy.max_words): Step[] {
    const pcs = preTokenize(text);
    if (!pcs.length) return [];
    const partial = pcs[pcs.length - 1];
    const cands = this.candidates(partial);
    if (!cands.length) return [];
    const ids = [this.eosId, ...pcs.slice(0, -1).map((p) => this.pieceIndex.get(p) ?? this.unkId)];
    let { probs, support } = this.dist(ids);
    let best = cands[0];
    let sum = 0;
    for (const c of cands) {
      sum += probs[c];
      if (probs[c] > probs[best]) best = c;
    }
    let prob = (1 - (this.oov.get(partial) ?? 0)) * (probs[best] / sum);
    const out: Step[] = [{ text: this.text[best].slice(decodePieces(partial).length), prob, support }];
    ids.push(best);
    while (!this.stopAfter.has(this.vocab[ids[ids.length - 1]]) && out.length < maxWords) {
      ({ probs, support } = this.dist(ids));
      let nxt = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[nxt]) nxt = i;
      if (nxt === this.eosId) break;
      prob *= probs[nxt];
      out.push({ text: this.text[nxt], prob, support });
      ids.push(nxt);
    }
    return out;
  }

  static apply(steps: Step[], policy: Policy): { ghost: string; prob: number } {
    let ghost = "";
    let prob = 0;
    const n = Math.min(steps.length, policy.max_words);
    for (let i = 0; i < n; i++) {
      const st = steps[i];
      if (st.support < policy.min_support) break;
      if (i && policy.ext != null) {
        if (st.prob / steps[i - 1].prob < policy.ext) break;
      } else if (st.prob < policy.tau) break;
      ghost += st.text;
      prob = st.prob;
    }
    return { ghost, prob };
  }

  /** Ghost text to show after `text`, or "" when the model isn't confident. */
  suggest(text: string, policy: Policy = this.policy): Suggestion {
    const steps = this.steps(text, policy.max_words);
    return { ...CompiledAutocomplete.apply(steps, policy), steps };
  }
}
