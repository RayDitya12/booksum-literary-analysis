import argparse
import json
import os
import time

import numpy as np
import torch

import config

from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def make_preprocess(tok, arm_cfg):
    def preprocess(examples):
        inputs = [str(x) for x in examples[arm_cfg["src_field"]]]
        targets = [str(x) for x in examples[config.TEXT_TARGET]]
        model_inputs = tok(inputs, max_length=arm_cfg["max_src_tokens"], truncation=True, padding=False)
        labels = tok(text_target=targets, max_length=arm_cfg["max_tgt_tokens"], truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def train(arm, limit=None, epochs=None, out_dir=None):
    out_dir = out_dir or config.OUT_DIR
    if arm not in config.ARMS:
        raise SystemExit(f"arm '{arm}' not in {list(config.ARMS)}")
    arm_cfg = config.ARMS[arm]
    model_dir = os.path.join(out_dir, arm)
    os.makedirs(model_dir, exist_ok=True)

    ds = load_from_disk(os.path.join(out_dir, "dataset"))
    if limit:
        for split in ds:
            ds[split] = ds[split].select(range(min(limit, len(ds[split]))))
    print(f"[train] {arm_cfg['label']} | train={len(ds['train'])} val={len(ds['validation'])} "
          f"src_field={arm_cfg['src_field']} max_src={arm_cfg['max_src_tokens']} max_tgt={arm_cfg['max_tgt_tokens']}")

    tok = AutoTokenizer.from_pretrained(arm_cfg["tokenizer"])
    model = AutoModelForSeq2SeqLM.from_pretrained(arm_cfg["model"])
    if config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ds = ds.map(make_preprocess(tok, arm_cfg), batched=True, remove_columns=ds["train"].column_names, desc="tokenize")

    n_epochs = epochs or config.EPOCHS
    steps_per_epoch = -(-len(ds["train"]) // (config.BATCH_PER_DEVICE * config.GRAD_ACCUM))
    args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=config.BATCH_PER_DEVICE,
        per_device_eval_batch_size=config.BATCH_PER_DEVICE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        num_train_epochs=n_epochs,
        learning_rate=config.LR,
        warmup_steps=int(config.WARMUP_RATIO * steps_per_epoch * n_epochs),
        weight_decay=0.05,
        fp16=config.FP16 and torch.cuda.is_available(),
        seed=config.SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
        predict_with_generate=False,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0
    trainer.save_model(model_dir)
    tok.save_pretrained(model_dir)

    meta = {
        "arm": arm,
        "label": arm_cfg["label"],
        "model": arm_cfg["model"],
        "tokenizer": arm_cfg["tokenizer"],
        "src_field": arm_cfg["src_field"],
        "max_src_tokens": arm_cfg["max_src_tokens"],
        "max_tgt_tokens": arm_cfg["max_tgt_tokens"],
        "rows_train": len(ds["train"]),
        "rows_val": len(ds["validation"]),
        "epochs": epochs or config.EPOCHS,
        "train_wall_seconds": round(secs, 1),
        "samples_per_second": round(len(ds["train"]) * (epochs or config.EPOCHS) / max(1.0, secs), 3),
        "eval_loss": float(result.training_loss),
    }
    with open(os.path.join(model_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] done in {secs/60:.1f} min -> {model_dir}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(config.ARMS))
    ap.add_argument("--limit", type=int, default=None, help="smoke test with N rows per split")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    train(args.arm, limit=args.limit, epochs=args.epochs, out_dir=args.out)
