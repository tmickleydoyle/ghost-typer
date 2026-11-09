// Run the browser runtime from the command line (Bun or Node >= 23.6):
//   bun web/cli.ts "write a python fun" "how do i "
//   bun web/cli.ts --json ...                 full suggestion chains (used by test_model.py)
//   bun web/cli.ts --dist ...                 top-5 next-token log-probs after each text
//   bun web/cli.ts --model path/to/model.bin  use another export (default web/model.bin)

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { preTokenize } from "../tokenizer.ts";
import { CompiledAutocomplete } from "./autocomplete.ts";

const dir = import.meta.dirname ?? new URL(".", import.meta.url).pathname;
let args = process.argv.slice(2);
let modelPath = join(dir, "model.bin");
const mi = args.indexOf("--model");
if (mi >= 0) {
  modelPath = args[mi + 1];
  args = [...args.slice(0, mi), ...args.slice(mi + 2)];
}
const mode = args[0] === "--json" || args[0] === "--dist" ? args.shift() : "";

const buf = readFileSync(modelPath);
const ac = new CompiledAutocomplete(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

if (mode === "--dist") {
  const out = args.map((t) => {
    const ids = [ac.eosId, ...preTokenize(t).map((p) => ac.vocab.indexOf(p)).map((i) => (i < 0 ? ac.unkId : i))];
    const { probs } = ac.dist(ids);
    const top = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 5);
    return { text: t, top: top.map((i) => [i, Math.log(probs[i])]) };
  });
  console.log(JSON.stringify(out));
} else {
  const results = args.map((t) => {
    const start = performance.now();
    const s = ac.suggest(t);
    return { text: t, ...s, ms: performance.now() - start };
  });
  if (mode === "--json") console.log(JSON.stringify(results));
  else for (const r of results) console.log(`${r.text}\x1b[90m${r.ghost}\x1b[0m   ${r.ghost ? Math.round(r.prob * 100) + "%" : "(no confident suggestion)"}  ${r.ms.toFixed(2)} ms`);
}
