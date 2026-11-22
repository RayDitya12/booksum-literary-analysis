import argparse
import json
import os
from collections import Counter, defaultdict

import config

from datasets import load_dataset

SPLITS = ["train", "validation", "test"]


def main(out_dir=None, save=True):
    out_dir = out_dir or config.OUT_DIR
    ds = load_dataset(config.DATASET)
    report = {}

    for sp in SPLITS:
        rows = ds[sp]
        null_target = sum(1 for v in rows[config.TEXT_TARGET] if v is None or str(v).strip() == "")
        agg = [i for i, v in enumerate(rows["is_aggregate"]) if v]
        agg_set = set(agg)
        agg_with = sum(1 for i in agg if rows[i][config.TEXT_TARGET] and str(rows[i][config.TEXT_TARGET]).strip())
        single = [i for i in range(len(rows)) if i not in agg_set]
        single_with = sum(1 for i in single if rows[i][config.TEXT_TARGET] and str(rows[i][config.TEXT_TARGET]).strip())
        content_all_null = all(v is None for v in rows["content"])
        report[sp] = {
            "rows": len(rows),
            "null_target": null_target,
            "null_target_pct": round(100.0 * null_target / len(rows), 2),
            "after_drop_null": len(rows) - null_target,
            "aggregate_rows": len(agg),
            "aggregate_with_target_pct": round(100.0 * agg_with / len(agg), 1) if agg else None,
            "single_chapter_rows": len(single),
            "single_with_target_pct": round(100.0 * single_with / len(single), 1) if single else None,
            "final_single_with_target": single_with,
            "unique_bid": len(set(rows["bid"])),
            "unique_book_id_raw": len({str(b) for b in rows["book_id"]}),
            "unique_title_prefix": len({str(b).split(".")[0] for b in rows["book_id"]}),
            "content_column_all_null": content_all_null,
            "source_freq": dict(Counter(rows["source"])),
            "dup_chapter_path": _duplication(rows),
        }
        print(f"[verify] {sp}: {report[sp]['rows']} rows | null target {report[sp]['null_target']} "
              f"({report[sp]['null_target_pct']}%) | final single-chapter {report[sp]['final_single_with_target']} "
              f"| bid {report[sp]['unique_bid']} vs book_id {report[sp]['unique_book_id_raw']}")

    report["cross_split_overlap"] = {}
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            keys = {}
            for field, fn in [("bid", lambda r: r["bid"]), ("title", lambda r: str(r["book_id"]).split(".")[0])]:
                sa, sb = {fn(ds[a][i]) for i in range(0, len(ds[a]), max(1, len(ds[a]) // 700))}, {fn(ds[b][i]) for i in range(0, len(ds[b]), max(1, len(ds[b]) // 700))}
                keys[field] = len(sa & sb)
            report["cross_split_overlap"][f"{a}|{b}"] = keys
            print(f"[verify] overlap {a}/{b}: bid={keys['bid']} title={keys['title']} (0 = book-disjoint)")

    summary_rows = ds["train"]
    pairs = [(str(t), str(a)) for t, a in zip(summary_rows[::7]["summary_text"], summary_rows[::7]["summary_analysis"]) if a and str(a).strip()]
    contained = sum(1 for t, a in pairs if len(t) > 60 and t[:80] in a)
    report["analysis_contains_summary_verbatim"] = {"sampled": len(pairs), "hits": contained, "pct": round(100.0 * contained / max(1, len(pairs)), 2)}
    print(f"[verify] analysis literally containing start of summary: {contained}/{len(pairs)}")

    checks = {
        "splits_book_disjoint": all(v["bid"] == 0 for v in report["cross_split_overlap"].values()),
        "book_id_is_not_book_key": report["train"]["unique_book_id_raw"] > report["train"]["unique_bid"] * 3,
        "target_nulls_significant": report["train"]["null_target_pct"] > 30,
        "content_unusable": report["train"]["content_column_all_null"],
    }
    report["checks"] = checks
    print("\n[verify] CHECKS " + " | ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items()))
    if save:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "verify_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[verify] wrote {os.path.join(out_dir, 'verify_report.json')}")
    return report


def _duplication(rows):
    seen = defaultdict(lambda: {"n": 0, "src": set(), "len": set()})
    paths = [str(p) for p in rows["chapter_path"]]
    srcs = list(rows["source"])
    lens = list(rows["chapter_length"])
    for p, s, l in zip(paths, srcs, lens):
        o = seen[p]
        o["n"] += 1
        o["src"].add(s)
        o["len"].add(float(l) if l is not None else None)
    repeated = [o for o in seen.values() if o["n"] > 1]
    return {
        "unique_paths": len(seen),
        "rows": len(rows),
        "rows_in_duplicated_groups": sum(o["n"] for o in repeated),
        "paths_with_multiple_sources": sum(1 for o in repeated if len(o["src"]) > 1),
        "max_repeats": max((o["n"] for o in seen.values()), default=0),
        "paths_with_conflicting_length": sum(1 for o in seen.values() if len(o["len"]) > 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    main(args.out)
