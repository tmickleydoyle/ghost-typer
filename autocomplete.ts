// Command-line autocomplete tester (zero dependencies).
//
//   bun autocomplete.ts                     interactive prompt
//   bun autocomplete.ts "once upon a"       complete one or more prompts
//   bun autocomplete.ts --json "how are"    machine-readable output (used by tests)
//
// Also runs on Node >= 23.6 (`node autocomplete.ts`). For the browser UI see web/.

import { createInterface } from "node:readline";
import { Autocompleter } from "./model.ts";

const DIR = import.meta.dirname ?? new URL(".", import.meta.url).pathname;
const ac = new Autocompleter(DIR);

function show(prompt: string): void {
  const t0 = performance.now();
  const r = ac.complete(prompt);
  const ms = Math.round(performance.now() - t0);
  console.log(`${prompt}\x1b[36m${r.completion}\x1b[0m   (${ms} ms)`);
  console.log("  next: " + r.top.map((t) => `${JSON.stringify(t.token)} ${(Math.exp(t.logprob) * 100).toFixed(1)}%`).join("  "));
}

const args = process.argv.slice(2);
if (args[0] === "--json") {
  console.log(JSON.stringify(args.slice(1).map((p) => ac.complete(p))));
} else if (args.length) {
  args.forEach(show);
} else {
  const cfg = ac.model.cfg;
  console.log(`Loaded ${cfg.num_hidden_layers}-layer model, vocab ${cfg.vocab_size}. Type a prefix (Ctrl+D to quit).`);
  const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: "> " });
  rl.prompt();
  rl.on("line", (line) => {
    if (line.trim()) show(line);
    rl.prompt();
  });
}
