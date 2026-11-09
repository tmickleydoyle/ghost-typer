// Zero-dependency inference for this model: reads config.json, model.safetensors
// and tokenizer.json and runs the Llama forward pass in plain TypeScript.
// No ML library, no other model. Used by autocomplete.ts (CLI).

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { WordTokenizer } from "./tokenizer.ts";

export type Config = {
  vocab_size: number;
  hidden_size: number;
  intermediate_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  num_key_value_heads: number;
  head_dim?: number;
  rms_norm_eps: number;
  rope_theta?: number;
  rope_parameters?: { rope_theta: number };
  max_position_embeddings: number;
  eos_token_id: number;
};

// ---------- safetensors ----------

function loadSafetensors(path: string): Map<string, Float32Array> {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString("utf8"));
  const base = 8 + headerLen;
  const tensors = new Map<string, Float32Array>();
  for (const [name, info] of Object.entries<any>(header)) {
    if (name === "__metadata__") continue;
    if (info.dtype !== "F32") throw new Error(`${name}: unsupported dtype ${info.dtype}`);
    const [begin, end] = info.data_offsets as [number, number];
    const offset = buf.byteOffset + base + begin;
    const length = (end - begin) / 4;
    tensors.set(
      name,
      offset % 4 === 0
        ? new Float32Array(buf.buffer, offset, length)
        : new Float32Array(buf.buffer.slice(offset, offset + length * 4)),
    );
  }
  return tensors;
}

// ---------- Llama forward pass ----------

function matvec(w: Float32Array, x: Float64Array, rows: number, cols: number): Float64Array {
  const y = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    const off = r * cols;
    for (let c = 0; c < cols; c++) s += w[off + c] * x[c];
    y[r] = s;
  }
  return y;
}

function rmsnorm(x: Float64Array, w: Float32Array, eps: number): Float64Array {
  let ss = 0;
  for (const v of x) ss += v * v;
  const inv = 1 / Math.sqrt(ss / x.length + eps);
  return x.map((v, i) => v * inv * w[i]);
}

export class Llama {
  cfg: Config;
  w: Map<string, Float32Array>;
  headDim: number;
  invFreq: Float64Array;

  constructor(dir: string) {
    this.cfg = JSON.parse(readFileSync(join(dir, "config.json"), "utf8"));
    this.w = loadSafetensors(join(dir, "model.safetensors"));
    const c = this.cfg;
    this.headDim = c.head_dim ?? c.hidden_size / c.num_attention_heads;
    const theta = c.rope_parameters?.rope_theta ?? c.rope_theta ?? 10000;
    const half = this.headDim / 2;
    this.invFreq = new Float64Array(half).map((_, i) => 1 / theta ** ((2 * i) / this.headDim));
  }

  get numParams(): number {
    let n = 0;
    for (const t of this.w.values()) n += t.length;
    return n;
  }

  private t(name: string): Float32Array {
    const t = this.w.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    return t;
  }

  private rope(vec: Float64Array, offset: number, pos: number): void {
    const half = this.headDim / 2;
    for (let i = 0; i < half; i++) {
      const a = pos * this.invFreq[i];
      const cos = Math.cos(a), sin = Math.sin(a);
      const x1 = vec[offset + i], x2 = vec[offset + i + half];
      vec[offset + i] = x1 * cos - x2 * sin;
      vec[offset + i + half] = x2 * cos + x1 * sin;
    }
  }

  /** Log-probabilities of the next token after `ids`. */
  nextLogprobs(ids: number[]): Float64Array {
    const c = this.cfg, D = c.hidden_size, H = c.num_attention_heads, KV = c.num_key_value_heads;
    const hd = this.headDim, I = c.intermediate_size, L = c.num_hidden_layers;
    const embed = this.t("model.embed_tokens.weight");
    const keys: Float64Array[][] = Array.from({ length: L }, () => []);
    const vals: Float64Array[][] = Array.from({ length: L }, () => []);
    let x = new Float64Array(D);

    for (let pos = 0; pos < ids.length; pos++) {
      x = Float64Array.from(embed.subarray(ids[pos] * D, (ids[pos] + 1) * D));
      const last = pos === ids.length - 1;
      for (let l = 0; l < L; l++) {
        const p = `model.layers.${l}.`;
        const xn = rmsnorm(x, this.t(p + "input_layernorm.weight"), c.rms_norm_eps);
        const q = matvec(this.t(p + "self_attn.q_proj.weight"), xn, H * hd, D);
        const k = matvec(this.t(p + "self_attn.k_proj.weight"), xn, KV * hd, D);
        const v = matvec(this.t(p + "self_attn.v_proj.weight"), xn, KV * hd, D);
        for (let h = 0; h < H; h++) this.rope(q, h * hd, pos);
        for (let h = 0; h < KV; h++) this.rope(k, h * hd, pos);
        keys[l].push(k);
        vals[l].push(v);

        const attn = new Float64Array(H * hd);
        for (let h = 0; h < H; h++) {
          const kvh = Math.floor(h / (H / KV)) * hd;
          const scores = keys[l].map((kk) => {
            let s = 0;
            for (let d = 0; d < hd; d++) s += q[h * hd + d] * kk[kvh + d];
            return s / Math.sqrt(hd);
          });
          const max = Math.max(...scores);
          const exps = scores.map((s) => Math.exp(s - max));
          const z = exps.reduce((a, b) => a + b, 0);
          exps.forEach((e, s) => {
            for (let d = 0; d < hd; d++) attn[h * hd + d] += (e / z) * vals[l][s][kvh + d];
          });
        }
        const o = matvec(this.t(p + "self_attn.o_proj.weight"), attn, D, H * hd);
        for (let i = 0; i < D; i++) x[i] += o[i];

        // The last layer's MLP only matters where we read logits.
        if (l === L - 1 && !last) continue;
        const xn2 = rmsnorm(x, this.t(p + "post_attention_layernorm.weight"), c.rms_norm_eps);
        const g = matvec(this.t(p + "mlp.gate_proj.weight"), xn2, I, D);
        const u = matvec(this.t(p + "mlp.up_proj.weight"), xn2, I, D);
        for (let i = 0; i < I; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
        const dn = matvec(this.t(p + "mlp.down_proj.weight"), g, D, I);
        for (let i = 0; i < D; i++) x[i] += dn[i];
      }
    }

    const logits = matvec(this.t("lm_head.weight"), rmsnorm(x, this.t("model.norm.weight"), c.rms_norm_eps), c.vocab_size, D);
    const max = Math.max(...logits);
    const lse = max + Math.log(logits.reduce((a, v) => a + Math.exp(v - max), 0));
    return logits.map((v) => v - lse);
  }
}

// ---------- autocomplete ----------

export type Completion = {
  prompt: string;
  completion: string;
  top: { id: number; token: string; logprob: number }[];
};

export class Autocompleter {
  tok: WordTokenizer;
  model: Llama;
  maxNew: number;

  constructor(dir: string) {
    this.tok = WordTokenizer.fromJson(JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8")));
    this.model = new Llama(dir);
    const gen = JSON.parse(readFileSync(join(dir, "generation_config.json"), "utf8"));
    this.maxNew = gen.max_new_tokens ?? 16;
  }

  /** Greedy completion plus the top-k candidates for the next token. */
  complete(prompt: string, topK = 3): Completion {
    const { tok, model } = this;
    const ids = tok.encode(prompt);
    const first = model.nextLogprobs(ids);
    const top = [...first.keys()]
      .sort((a, b) => first[b] - first[a])
      .slice(0, topK)
      .map((id) => ({ id, token: tok.decode([id]) || tok.inverse[id], logprob: first[id] }));
    const out: number[] = [];
    let next = argmax(first);
    while (next !== model.cfg.eos_token_id && out.length < this.maxNew && ids.length < model.cfg.max_position_embeddings) {
      out.push(next);
      ids.push(next);
      next = argmax(model.nextLogprobs(ids));
    }
    return { prompt, completion: tok.decode(out), top };
  }
}

function argmax(a: Float64Array): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}
