"""Download human-written coding text into data/prompts.txt and data/replies.txt.

prompts.txt: what people type to an assistant (OASST user turns, MBPP, CoNaLa).
replies.txt: OASST assistant turns (human-written replies).
wildchat.txt: English user turns from WildChat-1M coding conversations (never the
              assistant turns, which are model-generated).

Sources (all human-written, permissively licensed; see data/SOURCES.md):
  - OpenAssistant/oasst2  (Apache-2.0)  English prompter and assistant messages from
                                        conversations whose opening prompt is about code
  - google-research-datasets/mbpp (CC-BY-4.0)  crowd-written task descriptions
  - neulab/conala (MIT)  curated programming intents (paired train + test)

Only prose is kept: code blocks, code-looking lines and markdown markup are
dropped, and each message is cut to its first sentences (prompts: 2, replies: 5).
"""

import gzip
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_WORDS = 24
MAX_PROMPT_SENTENCES = 2
MAX_REPLY_SENTENCES = 5

# Strong signals only; words like "list" or "error" alone are too ambiguous.
CODE_RE = re.compile(
    r"\b(python|javascript|typescript|java|c\+\+|c#|rust|golang|kotlin|swift|php|ruby|perl|haskell|scala|"
    r"sql|mysql|postgres(ql)?|sqlite|mongodb|bash|powershell|shell script|html|css|react|vue|angular|node\.?js|"
    r"django|flask|numpy|pandas|pytorch|tensorflow|regex|regular expression|json|yaml|xml|csv|api|rest api|"
    r"git|github|docker|kubernetes|linux|terminal|command line|compiler|compile|debug|debugging|stack trace|"
    r"exception|segfault|code|coding|program|programming|programmer|script|function|algorithm|recursion|"
    r"recursive|data structure|linked list|binary tree|hash ?map|array|dictionary|variable|loop|class|"
    r"object[- ]oriented|unit test|runtime|syntax|library|framework|database|backend|frontend|webpage|website)\b",
    re.I,
)
# Credentials people paste into prompts (API keys, tokens, private keys), plus any long
# random-looking token (32+ chars mixing letters and digits). Sentences containing one are dropped.
SECRET_RE = re.compile(
    r"sk-(?:proj-|svcacct-|admin-|ant-)?[A-Za-z0-9_-]{20,}|(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_\w{30,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{10,}|(?:sk|rk)_(?:live|test)_\w{20,}"
    r"|hf_[A-Za-z0-9]{30,}|eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|\b\d{8,10}:[\w-]{35}\b"
)
RANDOM_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-+/=]{32,}")


def looks_secret(text: str) -> bool:
    if SECRET_RE.search(text):
        return True
    return any(re.search(r"[A-Za-z]", t) and re.search(r"\d", t) for t in RANDOM_TOKEN_RE.findall(text))


CODE_LINE_RE = re.compile(r"[{};=<>]|^\s*(def|class|import|from|for|if|return|#include|const|let|var)\b|\w\(.*\)")
SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


def clean(text: str, max_sentences: int = MAX_PROMPT_SENTENCES) -> list[str]:
    """Prose sentences from a message, without code or markdown."""
    text = re.sub(r"```.*?(```|$)", " ", text, flags=re.S)
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(#+|[-*+]|\d+[.)])\s+", "", ln)  # headings, bullets, numbered items
        ln = re.sub(r"\*\*|__|\[([^\]]*)\]\([^)]*\)", r"\1", ln)  # bold, links
        if ln.strip() and not CODE_LINE_RE.search(ln):
            lines.append(ln)
    text = " ".join(" ".join(lines).split())
    text = re.sub(r"`([^`]{1,30})`", r"\1", text)  # short inline code -> plain word
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    if not text.isascii():
        return []
    out = []
    for sent in SENT_SPLIT_RE.split(text)[:max_sentences]:
        words = sent.split()
        if 3 <= len(words) <= MAX_WORDS and not looks_secret(sent):
            out.append(" ".join(words))
    return out


def oasst() -> tuple[list[str], list[str]]:
    path = hf_hub_download("OpenAssistant/oasst2", "2023-11-05_oasst2_ready.messages.jsonl.gz", repo_type="dataset")
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    rows = [
        r for r in rows
        if r["lang"] == "en" and not r["deleted"] and r["review_result"] and not r["synthetic"]
    ]
    coding_trees = {r["message_tree_id"] for r in rows if r["parent_id"] is None and CODE_RE.search(r["text"])}
    prompts, replies = [], []
    for r in rows:
        if r["message_tree_id"] in coding_trees:
            if r["role"] == "prompter":
                prompts += clean(r["text"], MAX_PROMPT_SENTENCES)
            else:
                replies += clean(r["text"], MAX_REPLY_SENTENCES)
    return prompts, replies


def mbpp() -> list[str]:
    import pyarrow.parquet as pq  # only needed for downloading, not for looks_secret

    out = []
    for split in ["train", "test", "validation", "prompt"]:
        path = hf_hub_download("google-research-datasets/mbpp", f"full/{split}-00000-of-00001.parquet", repo_type="dataset")
        for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            out += clean(text)
    return out


def conala() -> list[str]:
    out = []
    for split in ["train", "test"]:
        path = hf_hub_download("neulab/conala", f"data/conala-paired-{split}.json", repo_type="dataset")
        for line in open(path):
            if line.strip():
                out += clean(json.loads(line)["intent"])
    return out


def wildchat() -> list[str]:
    """allenai/WildChat-1M (ODC-BY): user turns of English, non-toxic coding conversations."""
    import pyarrow.parquet as pq

    out = []
    for i in range(14):
        path = hf_hub_download("allenai/WildChat-1M", f"data/train-{i:05d}-of-00014.parquet", repo_type="dataset")
        table = pq.read_table(path, columns=["conversation", "language", "toxic"])
        convs, langs, toxic = (table.column(c).to_pylist() for c in ["conversation", "language", "toxic"])
        for conv, lang, tox in zip(convs, langs, toxic):
            users = [m for m in conv if m["role"] == "user"]
            if lang != "English" or tox or not users or not CODE_RE.search(users[0]["content"]):
                continue
            for m in users:
                if not m.get("redacted") and not m.get("toxic") and m.get("language") in (None, "English"):
                    out += clean(m["content"], MAX_PROMPT_SENTENCES)
        print(f"wildchat shard {i + 1}/14: {len(out)} sentences", flush=True)
    return out


def main() -> None:
    oasst_prompts, oasst_replies = oasst()
    sources = {
        "prompts": [("oasst2_user", oasst_prompts), ("mbpp", mbpp()), ("conala", conala())],
        "replies": [("oasst2_assistant", oasst_replies)],
        "wildchat": [("wildchat_user", wildchat())],
    }
    seen, report = set(), {}
    DATA.mkdir(exist_ok=True)
    for kind, parts in sources.items():
        lines = []
        for name, got in parts:
            report[name] = len(got)
            for s in got:
                if s.lower() not in seen:
                    seen.add(s.lower())
                    lines.append(s)
        (DATA / f"{kind}.txt").write_text("\n".join(lines) + "\n")
        report[f"{kind}_unique"] = len(lines)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
