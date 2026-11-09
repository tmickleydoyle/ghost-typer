// ghost-typer demo page. Bundle with:
//   bun build web/demo.ts --outfile web/demo.js --target browser --minify
//
// Until the user touches the input, it types demo prompts itself: a prefix appears
// one key at a time, the ghost text shows up, then "Tab" accepts it.

import { CompiledAutocomplete, type Policy } from "./autocomplete.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const promptEl = $<HTMLTextAreaElement>("prompt");
const ghostEl = $("ghost");
const stateEl = $("state");
const metaEl = $("meta");

// Prefixes the demo types; the rest comes from the model.
const DEMO = ["write a python fun", "can you explain the differ", "how do i conv", "i'm getting an err"];

const esc = (s: string) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!);
const fmt = (n: number) => n.toLocaleString("en-US");
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function main(): Promise<void> {
  const t0 = performance.now();
  const ac = await CompiledAutocomplete.load("model.bin", (f) => (metaEl.textContent = `loading model ${Math.round(f * 100)}%`));
  const loadMs = Math.round(performance.now() - t0);
  metaEl.textContent = `${(ac.bytes / 1e6).toFixed(1)} MB · ${fmt(ac.vocab.length)} words · ${fmt(ac.contexts)} contexts · loaded in ${loadMs} ms`;

  const modes: [string, number][] = [["strict", 0.9], ["default", ac.policy.tau], ["loose", 0.6]];
  const policy: Policy = { ...ac.policy };
  let ghost = "";
  let typed = 0;
  let saved = 0;
  let demo = true;

  const render = (accepted = "") => {
    const text = promptEl.value;
    const start = performance.now();
    const s = ac.suggest(text, policy);
    const ms = performance.now() - start;
    ghost = s.ghost;
    const head = accepted ? text.slice(0, text.length - accepted.length) : text;
    ghostEl.innerHTML = esc(head) + (accepted ? `<span class="accept">${esc(accepted)}</span>` : "") + "<span>" + esc(ghost) + "</span>";
    promptEl.style.height = "auto";
    promptEl.style.height = Math.max(promptEl.scrollHeight, ghostEl.scrollHeight) + "px";
    if (!text) {
      stateEl.textContent = !demo && typed ? `saved ${saved} of ${typed + saved} keystrokes` : "\u00a0";
      return;
    }
    const i = s.steps.findIndex((st) => st.text); // first step that adds text
    const guess = i < 0 ? "" : s.steps.slice(0, i + 1).map((st) => st.text).join("");
    const what = ghost
      ? `${Math.round(s.prob * 100)}% sure`
      : guess
        ? `quiet · "${guess.trim()}" only ${Math.round(s.steps[i].prob * 100)}%`
        : "quiet · unknown";
    const savedStr = !demo && typed + saved ? ` │ saved ${Math.round((100 * saved) / (typed + saved))}%` : "";
    stateEl.textContent = `${ms.toFixed(2)} ms │ ${what}${savedStr}`;
  };

  const stopDemo = () => {
    if (!demo) return;
    demo = false;
    promptEl.value = "";
    render();
  };

  async function runDemo(): Promise<void> {
    await sleep(600);
    while (demo) {
      for (const prefix of DEMO) {
        for (let i = 1; i <= prefix.length && demo; i++) {
          promptEl.value = prefix.slice(0, i);
          render();
          await sleep(55 + Math.random() * 60);
        }
        if (!demo) return;
        await sleep(900);
        if (!demo) return;
        if (ghost) {
          const accepted = ghost;
          promptEl.value += accepted;
          render(accepted);
          await sleep(1400);
        }
        if (!demo) return;
        promptEl.value = "";
        render();
        await sleep(500);
      }
    }
  }

  for (const [name, tau] of modes) {
    const b = document.createElement("button");
    b.textContent = name;
    b.setAttribute("aria-pressed", String(tau === policy.tau));
    b.onclick = () => {
      policy.tau = tau;
      for (const other of $("modes").children) other.setAttribute("aria-pressed", String(other === b));
      render();
    };
    $("modes").appendChild(b);
  }

  promptEl.addEventListener("pointerdown", stopDemo);
  promptEl.addEventListener("keydown", (e) => {
    if (demo) {
      stopDemo();
      if (e.key === "Tab" || e.key === "Escape") e.preventDefault();
      return; // printable keys fall through and land in the now-empty input
    }
    if (e.key === "Tab" && ghost) {
      e.preventDefault();
      const accepted = ghost;
      promptEl.value += accepted;
      saved += accepted.length - 1; // one Tab instead of typing the ghost
      typed += 1;
      render(accepted);
    } else if (e.key === "Escape") {
      promptEl.value = "";
      render();
    }
  });
  promptEl.addEventListener("beforeinput", (e) => {
    if (!demo && e.inputType.startsWith("insert")) typed += (e.data ?? "").length || 1;
  });
  promptEl.addEventListener("input", () => render());

  promptEl.disabled = false;
  render();
  runDemo();
}

main().catch((e) => (metaEl.textContent = String(e)));
