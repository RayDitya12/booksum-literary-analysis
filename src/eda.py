import argparse
import json
import os

import config

from datasets import load_from_disk
from transformers import AutoTokenizer

WINDOWS = [512, 1024, 2048, 4096, 6144, 8192, 16384]
QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99]


def _quantiles(values):
    vs = sorted(values)
    n = len(vs)
    out = {"mean": round(sum(vs) / n), "max": vs[-1], "min": vs[0]}
    for p in QUANTILES:
        out[f"p{int(p * 100)}"] = vs[min(n - 1, int(p * n))]
    return out


def _coverage(values, windows):
    n = len(values)
    return {str(w): round(100.0 * sum(1 for v in values if v <= w) / n, 1) for w in windows}


def measure(limit=400, out_dir=None):
    out_dir = out_dir or config.OUT_DIR
    ds = load_from_disk(os.path.join(out_dir, "dataset"))
    tok_cache = {}
    report = {}
    for arm, cfg in config.ARMS.items():
        path = cfg["tokenizer"]
        if path not in tok_cache:
            tok_cache[path] = AutoTokenizer.from_pretrained(path)
        tok = tok_cache[path]
        report[arm] = {"tokenizer": path, "splits": {}}
        for split in ds:
            rows = ds[split]
            idx = list(range(0, len(rows), max(1, len(rows) // limit)))[:limit]
            src, tgt, words = [], [], []
            for i in idx:
                r = rows[i]
                src.append(len(tok(str(r[cfg["src_field"]]), add_special_tokens=True)["input_ids"]))
                tgt.append(len(tok(str(r[config.TEXT_TARGET]), add_special_tokens=True)["input_ids"]))
                if r.get("chapter_length") and cfg["src_field"] == "chapter":
                    words.append(float(r["chapter_length"]))
            ratio = round((sum(src) / len(src)) / (sum(words) / len(words)), 3) if words else None
            report[arm]["splits"][split] = {
                "n_measured": len(src),
                "input_tokens": _quantiles(src),
                "input_coverage": _coverage(src, WINDOWS),
                "target_tokens": _quantiles(tgt),
                "target_coverage": _coverage(tgt, [512, 1024, 1536]),
                "words_to_tokens_ratio": ratio,
            }
            print(f"[eda] {arm} {split}: mean={report[arm]['splits'][split]['input_tokens']['mean']} "
                  f"p95={report[arm]['splits'][split]['input_tokens']['p95']} "
                  f"cov1024={report[arm]['splits'][split]['input_coverage']['1024']}% "
                  f"cov8192={report[arm]['splits'][split]['input_coverage']['8192']}%")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "token_stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    recommend(report)
    return report


def recommend(report):
    print("\n[eda] Cara membaca hasil:")
    for arm, data in report.items():
        cfg = config.ARMS[arm]
        cov = data["splits"]["train"]["input_coverage"]
        fits = [w for w in WINDOWS if cov.get(str(w), 0) >= 95.0]
        chosen = cfg["max_src_tokens"]
        smallest = min(fits) if fits else max(WINDOWS)
        print(f"  {arm}: window terkecil dengan >=95% bab utuh = {smallest} token; config sekarang = {chosen}")
        if chosen < smallest:
            print("    -> sebagian bab akan terpotong; catat persentasenya di Bab 3, atau naikkan window")
        over = data["splits"]["train"]["target_coverage"].get("1024", 0)
        if over < 100:
            print(f"    -> target >1024 token: {round(100 - over, 1)}% baris (truncation policy wajib dilaporkan)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    measure(limit=args.limit)
