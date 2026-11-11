"""Load the saved files exactly as a Hugging Face user would and check them.

Run `python build_weights.py` first.
"""

import json
import random
import shutil
import subprocess
import unittest
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import build_weights

MODEL_DIR = Path(__file__).parent

# prompt -> greedy completion of the dense model (new text only). Regression snapshots:
# plain greedy decoding, without the confidence policy the apps use.
CASES = {
    "You can use the": "following code:",
    "can you explain the": "difference between the two lists in the code",
    "write a react": "native",
    "Where is the": "error:",
    "Thank you for your": "response, and the other way to do this?",
}


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        cls.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).eval()
        _, cls.vocab, cls.ngram, _, _, cls.info = build_weights.compile_model()

    def complete(self, prompt: str) -> str:
        ids = self.tok(prompt, return_tensors="pt")
        out = self.model.generate(**ids, max_new_tokens=16)
        return self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def test_parameter_count(self):
        n = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(n, self.info["params"])
        self.assertLessEqual(n, 1_000_000_000)

    def test_greedy_completions(self):
        for prompt, expected in CASES.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(self.complete(prompt), expected)

    def test_matches_ngram_reference(self):
        """The transformer's next-token distribution equals the compiled n-gram's."""
        index = {p: i for i, p in enumerate(self.vocab)}
        rng = random.Random(0)
        phrases = build_weights.load_phrases()
        cases = []
        for toks in rng.sample(phrases, 200):
            ids = [0] + [index.get(p, 1) for p in toks]
            cases.append(ids[: rng.randint(1, len(ids))])
        for _ in range(50):  # unseen word salad, up to 80 tokens
            cases.append([0] + [rng.randrange(2, len(self.vocab)) for _ in range(rng.randint(1, 80))])
        with torch.no_grad():
            for ids in cases:
                got = self.model(torch.tensor([ids])).logits[0, -1].double().log_softmax(-1)
                ref = self.ngram.reference_logits(ids).log_softmax(-1)
                self.assertLess((got - ref).abs().max().item(), 1e-3)  # observed ~3e-5

    def test_unknown_words_back_off(self):
        # "zebra" is out of vocabulary; the model backs off to the shorter context.
        self.assertEqual(self.complete("Zebra, thank you for your"), self.complete("thank you for your"))

    def test_pipeline(self):
        gen = pipeline("text-generation", model=str(MODEL_DIR))
        # transformers 4.x pipelines sample by default and don't prepend BOS unless asked; 5.x do neither.
        out = gen("You can use the", add_special_tokens=True, do_sample=False)[0]["generated_text"]
        self.assertEqual(out, "You can use the following code:")

    @unittest.skipUnless(shutil.which("bun"), "bun not installed")
    def test_typescript_matches(self):
        prompts = ["once upon a", "how are", "zebra", "i hope this email"]
        out = subprocess.run(
            ["bun", "autocomplete.ts", "--json", *prompts],
            cwd=MODEL_DIR, capture_output=True, text=True, check=True,
        )
        results = json.loads(out.stdout)
        with torch.no_grad():
            for prompt, res in zip(prompts, results):
                with self.subTest(prompt=prompt):
                    ids = self.tok(prompt, return_tensors="pt")
                    logp = self.model(**ids).logits[0, -1].log_softmax(-1)
                    top = logp.topk(3)
                    self.assertEqual([t["id"] for t in res["top"]], top.indices.tolist())
                    for t, lp in zip(res["top"], top.values.tolist()):
                        self.assertAlmostEqual(t["logprob"], lp, places=3)
                    self.assertEqual(res["completion"].strip(), self.complete(prompt))


class TestNoSecrets(unittest.TestCase):
    """People paste API keys into prompts; none may reach the data files or a vocabulary."""

    def test_data_and_vocabularies(self):
        import sys

        sys.path.insert(0, str(MODEL_DIR / "scripts"))
        from fetch_coding_prompts import looks_secret

        for path in sorted((MODEL_DIR / "data").glob("*.txt")):
            hits = [n for n, line in enumerate(path.read_text().splitlines(), 1) if looks_secret(line)]
            self.assertEqual(hits, [], f"{path.name} lines")
        vocabs = {"tokenizer.json": list(json.loads((MODEL_DIR / "tokenizer.json").read_text())["model"]["vocab"])}
        raw = (MODEL_DIR / "web" / "model.bin").read_bytes()
        header = json.loads(raw[12 : 12 + int.from_bytes(raw[8:12], "little")])
        start = 12 + int.from_bytes(raw[8:12], "little")
        start += -start % 8
        sec = next(s for s in header["sections"] if s["name"] == "vocab")
        vocabs["web/model.bin"] = raw[start + sec["offset"] : start + sec["offset"] + sec["bytes"]].decode().split("\n")
        for name, words in vocabs.items():
            self.assertEqual([w for w in words if looks_secret(w)], [], name)


class TestCounting(unittest.TestCase):
    """compile_tables' numpy counting builds the same tables as the plain-dict reference."""

    def test_matches_dict_reference(self):
        bw = build_weights
        src = bw.load_sources()
        rng = random.Random(0)
        sample = {k: v if k == "phrases" else rng.sample(v, min(3000, len(v))) for k, v in src.items()}
        cases = [("kn", 1.0, 3, 2, 1.0), ("mkn", 1.0, 5, 2, 2.0), ("wb", 0.3, 4, 1.5, 1.0)]
        for smoothing, w, n_prev, min_count, long_factor in cases:
            with self.subTest(smoothing=smoothing, n_prev=n_prev):
                s = bw.Settings(n_prev=n_prev, smoothing=smoothing, phrase_weight=5.0, wildchat_weight=w, max_vocab=4000,
                                min_word_count=2, min_count=min_count, long_factor=long_factor)
                vocab, got, _ = bw.compile_tables(s, sample)
                index = {p: i for i, p in enumerate(vocab)}
                weighted = [([index.get(p, bw.UNK_ID) for p in t], wt) for lines, wt in bw.weighted_sources(s, sample) for t in lines]
                raw = bw.count_ngrams(weighted, s.order)
                forced = {c for t in sample["phrases"] for ctxs, _ in bw.padded_events([index[p] for p in t], s.order) for c in ctxs[1:]}
                keep = [set()] + [
                    {c for c, t in raw[k].items() if c in forced or sum(t.values()) >= (min_count if k <= 4 else min_count * long_factor)}
                    for k in range(1, s.order + 1)
                ]
                ref = bw.NGram.from_counts(raw, keep, len(vocab), s)
                for k in range(s.order + 1):
                    a, b = got.tables[k], ref.tables[k]
                    self.assertEqual(set(a), set(b), k)
                    if k:
                        for ctx in b:
                            self.assertEqual(set(a[ctx]), set(b[ctx]))
                            for j, c in b[ctx].items():  # fractional weights: summation order differs
                                self.assertLess(abs(a[ctx][j] - c), 1e-9 * max(1.0, abs(c)))
                for ctx, c in ref.support.items():
                    self.assertLess(abs(got.support[ctx] - c), 1e-9 * max(1.0, abs(c)))
                if smoothing == "mkn":
                    for k in range(1, s.order + 1):
                        for x, y in zip(got.mkn_discounts[k], ref.mkn_discounts[k]):
                            self.assertAlmostEqual(x, y, places=12)
                self.assertLess(abs(got.base_logits - ref.base_logits).max(), 1e-12)


@unittest.skipUnless(shutil.which("bun"), "bun not installed")
class TestWebExport(unittest.TestCase):
    """web/model.bin + web/autocomplete.ts (run `python export_web.py` first)."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        import export_web
        import suggest

        cls.export_web, cls.suggest = export_web, suggest
        cls.settings, cls.sources = build_weights.load_settings("web"), build_weights.load_sources()
        cls.policy = export_web.load_policy()
        # Rebuild the export; the Python Suggester uses the same quantized numbers.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.bin"
            _, cls.suggester = export_web.build(cls.settings, cls.sources, cls.policy, path, export_web.load_scheme())
            cls.rebuilt = path.read_bytes()

    def test_committed_export_is_current(self):
        """web/model.bin is exactly what export_web.py produces from the current sources and settings."""
        self.assertEqual((MODEL_DIR / "web" / "model.bin").read_bytes(), self.rebuilt)

    def cli(self, *args: str) -> list:
        out = subprocess.run(["bun", "web/cli.ts", *args], cwd=MODEL_DIR, capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def test_size(self):
        self.assertLessEqual((MODEL_DIR / "web" / "model.bin").stat().st_size, 50_000_000)

    def test_matches_python_policy(self):
        """Same suggestion chains (text, probability, support) as suggest.py on random prefixes."""
        rng = random.Random(1)
        lines = [ln for ln in (MODEL_DIR / "data" / "prompts.txt").read_text().splitlines() if len(ln) > 2]
        prefixes = [t[: rng.randint(1, len(t) - 1)] for t in rng.sample(lines, 300)]
        for p, res in zip(prefixes, self.cli("--json", *prefixes)):
            with self.subTest(prefix=p):
                py = self.suggester.steps(p, self.policy.max_words)
                self.assertEqual([s.text for s in py], [s["text"] for s in res["steps"]])
                for a, b in zip(py, res["steps"]):
                    self.assertAlmostEqual(a.prob, b["prob"], places=3)
                    if self.policy.min_support > 0:  # supports are only exported when the policy uses them
                        self.assertAlmostEqual(a.support, b["support"], places=3)
                ratios = [b.prob / a.prob for a, b in zip(py, py[1:])] if self.policy.ext is not None else []
                if all(abs(s.prob - self.policy.tau) > 1e-3 for s in py) and all(
                    abs(r - self.policy.ext) > 1e-3 for r in ratios
                ):  # skip float32 ties at the thresholds
                    self.assertEqual(self.suggest.Suggester.apply(py, self.policy), res["ghost"])

    def test_runtime_matches_dense_transformer(self):
        """The sparse runtime, run on the dense model's own (unquantized) tables, reproduces PyTorch."""
        import tempfile

        s = build_weights.load_settings("hf")
        vocab, ngram, _ = build_weights.compile_tables(s, self.sources)
        oov = self.suggest.oov_prefix_rates(s, self.sources, vocab)
        texts = ["Once upon a", "write a python function to", "zebra", "how do i", "thank you for your"]
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.bin"
            self.export_web.export(vocab, ngram, oov, self.policy, path, scheme="f32")
            results = self.cli("--model", str(path), "--dist", *texts)
        with torch.no_grad():
            for t, res in zip(texts, results):
                with self.subTest(text=t):
                    lp = model(**tok(t, return_tensors="pt")).logits[0, -1].double().log_softmax(-1)
                    self.assertEqual([i for i, _ in res["top"]], lp.topk(5).indices.tolist())
                    for i, v in res["top"]:
                        self.assertAlmostEqual(v, lp[i].item(), places=3)


if __name__ == "__main__":
    unittest.main()
