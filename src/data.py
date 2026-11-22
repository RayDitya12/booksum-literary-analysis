import argparse
import json
import os

import config
from excerpt import select_representative

from datasets import load_dataset


def _has_target(row):
    v = row.get(config.TEXT_TARGET)
    return v is not None and str(v).strip() != ""


def _keep(row):
    if config.DROP_NULL_TARGET and not _has_target(row):
        return False
    if not config.INCLUDE_AGGREGATE and row.get("is_aggregate"):
        return False
    return True


def build(add_hybrid=True, out_dir=None, overwrite=False):
    out_dir = out_dir or config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(config.DATASET)
    stats = {}
    for split in ds:
        raw = ds[split]
        kept = raw.filter(_keep, num_proc=1)
        kept = kept.remove_columns([c for c in kept.column_names if c == "content"])
        if add_hybrid:
            kept = kept.map(
                lambda ex: {
                    "chapter_selected": select_representative(
                        ex[config.TEXT_SRC],
                        max_words=config.HYBRID_MAX_WORDS,
                        buckets=config.HYBRID_SENTENCE_BUCKETS,
                    )
                },
                desc="selecting hybrid excerpts",
            )
        ds[split] = kept
        stats[split] = {
            "rows_raw": raw.num_rows,
            "rows_final": kept.num_rows,
            "unique_bid": len({r for r in kept["bid"]}),
            "sources": sorted({s for s in kept["source"]}),
            "chapter_length_words": _summary(kept["chapter_length"]),
            "target_length_words": _summary(kept["analysis_length"]),
        }
        print(f"[data] {split}: {raw.num_rows} -> {kept.num_rows} rows, {stats[split]['unique_bid']} books")
    ds.save_to_disk(os.path.join(out_dir, "dataset"))
    with open(os.path.join(out_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return ds


def _summary(values):
    vs = [v for v in values if v is not None]
    if not vs:
        return {}
    vs = sorted(vs)
    n = len(vs)
    q = lambda p: vs[min(n - 1, int(p * n))]
    return {
        "mean": round(sum(vs) / n),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": vs[-1],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hybrid", action="store_true", help="skip building chapter_selected column")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build(add_hybrid=not args.no_hybrid, out_dir=args.out)
