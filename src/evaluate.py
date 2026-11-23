import argparse
import json
import os
import time

import torch

import config

from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _rouge(references, predictions):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, pred in zip(references, predictions):
        s = scorer.score(ref or "", pred or "")
        for k in agg:
            agg[k].append(s[k].fmeasure)
    return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}


def _bertscore(references, predictions, lang="en"):
    try:
        from bert_score import score
    except ImportError:
        print("[eval] bert-score not installed, skipped")
        return None
    p, r, f1 = score(predictions, references, lang=lang, verbose=False, batch_size=16)
    return {"precision": round(p.mean().item(), 4), "recall": round(r.mean().item(), 4), "f1": round(f1.mean().item(), 4)}


@torch.no_grad()
def generate(model, tok, texts, max_src, max_new, num_beams=1):
    enc = tok(texts, max_length=max_src, truncation=True, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        num_beams=num_beams,
        early_stopping=num_beams > 1,
        no_repeat_ngram_size=3 if num_beams > 1 else 0,
        do_sample=False,
    )
    return tok.batch_decode(out, skip_special_tokens=True)


def evaluate(arm, limit=None, decoding="both", out_dir=None, batch=2):
    out_dir = out_dir or config.OUT_DIR
    arm_cfg = config.ARMS[arm]
    model_dir = os.path.join(out_dir, arm)
    ds = load_from_disk(os.path.join(out_dir, "dataset"))["test"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[eval] {arm_cfg['label']} on {len(ds)} test rows")

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise SystemExit(f"{model_dir} tidak berisi model. Jalankan train.py --arm {arm} lebih dulu, "
                         "atau pulihkan outputs dari Drive bila modelnya sudah pernah dilatih.")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    max_src = arm_cfg["max_src_tokens"]
    results = {"arm": arm, "label": arm_cfg["label"], "n_test": len(ds), "decoding": {}}
    refs, srcs, gold = [], [], []
    for i in range(len(ds)):
        r = ds[i]
        srcs.append(str(r[arm_cfg["src_field"]]))
        refs.append(str(r[config.TEXT_TARGET]))

    for mode, beams in [("greedy", config.GREEDY_K), ("beam", config.BEAM_SIZE)]:
        if decoding != "both" and decoding != mode:
            continue
        preds, t0 = [], time.time()
        for start in range(0, len(srcs), batch):
            chunk = srcs[start : start + batch]
            preds += generate(model, tok, chunk, max_src, config.MAX_NEW_TOKENS, num_beams=beams)
        wall = time.time() - t0
        scores = {"rouge": _rouge(refs, preds)}
        if mode == "greedy":
            scores["bertscore"] = _bertscore(refs, preds)
        results["decoding"][mode] = {
            "num_beams": beams,
            **scores,
            "infer_seconds_total": round(wall, 2),
            "infer_seconds_per_sample": round(wall / max(1, len(preds)), 3),
        }
        print(f"[eval] {mode}: {scores['rouge']} | {wall/60:.1f} min total")
        if mode == "greedy":
            for k in range(min(6, len(refs))):
                results.setdefault("examples", []).append(
                    {"index": k, "source_chars": len(srcs[k]), "gold": refs[k][:600], "pred": preds[k][:600]}
                )

    with open(os.path.join(model_dir, f"eval_{decoding}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _append_leaderboard(out_dir, arm, results)
    return results


def _append_leaderboard(out_dir, arm, results):
    path = os.path.join(out_dir, "leaderboard.jsonl")
    meta = {}
    mp = os.path.join(out_dir, arm, "train_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
    row = {
        "arm": arm,
        "label": results["label"],
        "n_test": results["n_test"],
        "train_minutes": round(meta.get("train_wall_seconds", 0) / 60, 1),
        "samples_per_second": meta.get("samples_per_second"),
    }
    for mode, data in results["decoding"].items():
        r = data.get("rouge", {})
        row[f"{mode}_rouge1"] = r.get("rouge1")
        row[f"{mode}_rouge2"] = r.get("rouge2")
        row[f"{mode}_rougeL"] = r.get("rougeL")
        row[f"{mode}_bertscore_f1"] = (data.get("bertscore") or {}).get("f1")
        row[f"{mode}_infer_sec_per_sample"] = data.get("infer_seconds_per_sample")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[eval] leaderboard appended -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(config.ARMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--decoding", default="both", choices=["greedy", "beam", "both"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    evaluate(args.arm, limit=args.limit, decoding=args.decoding, out_dir=args.out, batch=args.batch)
