"""Stage the tmickleydoyle/ghost-typer-web Hugging Face repo in dist/hf-web/.

Contents (flat, so the demo works when the folder is served as-is):
    README.md        model card (docs/hf-web-README.md)
    model.bin        uint8-quantized browser model (copy of web/model.bin)
    model-f32.bin    float32 reference export of the same compile
    ghost-typer.js   bundled zero-dependency ESM runtime (web/autocomplete.ts + tokenizer.ts)
    src/             TypeScript sources of the runtime
    index.html, demo.js   the demo page (loads ./model.bin)

    python scripts/stage_hf_web.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build_weights as bw  # noqa: E402
import export_web  # noqa: E402

OUT = ROOT / "dist" / "hf-web"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src").mkdir(parents=True)
    shutil.copy(ROOT / "web" / "model.bin", OUT / "model.bin")
    info, _ = export_web.build(bw.load_settings("web"), bw.load_sources(), export_web.load_policy(), OUT / "model-f32.bin", "f32")
    shutil.copy(ROOT / "docs" / "hf-web-README.md", OUT / "README.md")
    shutil.copy(ROOT / "tokenizer.ts", OUT / "src" / "tokenizer.ts")
    runtime = (ROOT / "web" / "autocomplete.ts").read_text().replace('from "../tokenizer.ts"', 'from "./tokenizer.ts"')
    (OUT / "src" / "autocomplete.ts").write_text(runtime)
    shutil.copy(ROOT / "web" / "index.html", OUT / "index.html")
    for entry, out in [("web/demo.ts", "demo.js"), ("web/autocomplete.ts", "ghost-typer.js")]:
        subprocess.run(["bun", "build", entry, "--outfile", str(OUT / out), "--target", "browser", "--format", "esm", "--minify"],
                       cwd=ROOT, check=True, capture_output=True)
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"{p.relative_to(OUT)}  {p.stat().st_size:,} bytes")
    print("f32 contexts", info["contexts"])


if __name__ == "__main__":
    main()
