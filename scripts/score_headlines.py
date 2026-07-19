"""PHASE 3.3 — score the headline corpus with a LOCAL open model (ollama).

Implements docs/PHASE3_LLM_DESIGN.md (+ its recorded amendments):
  corpus  = CoinTelegraph 2017-2020 + oliviervha 2021-2023 titles (47.9k)
  scorer  = Qwen2.5-7B-Instruct (Q4) via ollama, temperature 0, ONE pass
  output  = per headline: event type, severity 0-3, ex-ante direction, assets
  blind   = the model sees ONLY the title text — no dates, sources, prices

Resume-safe: appends to data/raw/news/scored_headlines.csv after every batch;
re-running skips already-scored ids. Batches of 20 titles per request.

Usage:
    python scripts/score_headlines.py            # full run (~2h on the laptop GPU)
    python scripts/score_headlines.py --limit 200   # smoke test
"""

import json
import os
import re
import sys
import argparse
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

NEWS_DIR = DATA_RAW_DIR / "news"
OUT_PATH = NEWS_DIR / "scored_headlines.csv"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = "qwen2.5:7b"
BATCH = 20

TYPES = ["regulation", "hack", "adoption", "etf_flow", "macro", "protocol",
         "legal", "market_structure", "noise"]
ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOGE", "DOT",
          "LINK", "UNI", "ATOM", "LTC", "NEAR", "MATIC", "MARKET"]

SYSTEM = f"""You classify cryptocurrency news headlines for research. For EACH numbered headline output one JSON object:
{{"i": <number>, "type": <one of {TYPES}>, "sev": <0-3 how consequential for holders, judged AS OF publication with no hindsight>, "dir": <-1 bad, 0 neutral/unclear, +1 good, for the tagged assets, ex-ante>, "assets": [<subset of {ASSETS}; use "MARKET" if not coin-specific>]}}
Return ONLY a JSON array of these objects, one per headline, no other text."""


def load_corpus():
    ct = pd.read_csv(NEWS_DIR / "cointelegraph/cointelegraph_news_head.csv",
                     usecols=["published_date", "title"], low_memory=False)
    ct["date"] = pd.to_datetime(ct["published_date"], errors="coerce", utc=True)
    ct = ct[["date", "title"]].assign(src="ct")
    ol = pd.read_csv(NEWS_DIR / "oliviervha/cryptonews.csv",
                     usecols=["date", "title"])
    ol["date"] = pd.to_datetime(ol["date"], errors="coerce", utc=True,
                                format="mixed")
    ol = ol[["date", "title"]].assign(src="ol")
    df = pd.concat([ct, ol], ignore_index=True).dropna(subset=["date", "title"])
    df = df[df["date"] >= "2017-08-17"]
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() >= 12].drop_duplicates(subset="title")
    df = df.sort_values("date").reset_index(drop=True)
    df["id"] = df.index
    return df


def ask(batch_titles):
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch_titles))
    payload = {"model": MODEL, "stream": False,
               "options": {"temperature": 0, "num_ctx": 4096},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": numbered}]}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())["message"]["content"]
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return []
    # Qwen writes "dir": +1 — a leading '+' is invalid JSON
    return json.loads(re.sub(r":\s*\+", ": ", m.group(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = load_corpus()
    print(f"corpus: {len(df)} headlines "
          f"({df['date'].min().date()} .. {df['date'].max().date()})", flush=True)

    done = set()
    if OUT_PATH.exists():
        done = set(pd.read_csv(OUT_PATH, usecols=["id"])["id"])
        print(f"resuming: {len(done)} already scored", flush=True)
    todo = df[~df["id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)

    n_ok = n_fail = 0
    for start in range(0, len(todo), BATCH):
        chunk = todo.iloc[start:start + BATCH]
        titles = list(chunk["title"])
        rows = []
        for attempt in (1, 2):
            try:
                scored = ask(titles)
                by_i = {int(s["i"]): s for s in scored if "i" in s}
                for j, (_, r) in enumerate(chunk.iterrows(), start=1):
                    s = by_i.get(j)
                    if s and s.get("type") in TYPES:
                        rows.append({
                            "id": r["id"], "date": r["date"].date().isoformat(),
                            "type": s["type"],
                            "sev": max(0, min(3, int(s.get("sev", 0)))),
                            "dir": max(-1, min(1, int(s.get("dir", 0)))),
                            "assets": "|".join(a for a in (s.get("assets") or ["MARKET"])
                                               if a in ASSETS) or "MARKET"})
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  batch at {start}: FAILED ({str(e)[:60]})", flush=True)
        if rows:
            hdr = not OUT_PATH.exists()
            pd.DataFrame(rows).to_csv(OUT_PATH, mode="a", header=hdr, index=False)
            n_ok += len(rows)
            n_fail += len(chunk) - len(rows)
        else:
            n_fail += len(chunk)
        if (start // BATCH) % 25 == 0:
            print(f"  {start + len(chunk)}/{len(todo)}  scored={n_ok} failed={n_fail}",
                  flush=True)
    print(f"done: scored {n_ok}, failed {n_fail} (failures stay unscored — "
          f"re-run to retry)", flush=True)


if __name__ == "__main__":
    main()
