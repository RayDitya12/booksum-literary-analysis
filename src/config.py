import os

DATASET = "kmfoda/booksum"

TEXT_SRC = "chapter"
TEXT_TARGET = "summary_analysis"

DROP_NULL_TARGET = True
INCLUDE_AGGREGATE = False

GROUP_KEY = "bid"

ARMS = {
    "m1_bart_direct": {
        "label": "Model 1 - Baseline BART direct truncation",
        "model": "facebook/bart-base",
        "tokenizer": "facebook/bart-large-cnn",
        "src_field": "chapter",
        "max_src_tokens": 1024,
        "max_tgt_tokens": 1024,
    },
    "m2_bart_hybrid": {
        "label": "Model 2 - Hybrid TextRank/TF-IDF selection then BART",
        "model": "facebook/bart-base",
        "tokenizer": "facebook/bart-large-cnn",
        "src_field": "chapter_selected",
        "max_src_tokens": 1024,
        "max_tgt_tokens": 1024,
    },
    "m3_led_long": {
        "label": "Model 3 - LED sparse+global long context",
        "model": "allenai/led-base-16384",
        "tokenizer": "allenai/led-base-16384",
        "src_field": "chapter",
        "max_src_tokens": 8192,
        "max_tgt_tokens": 1024,
    },
}

LR = 3e-5
EPOCHS = 2.0
BATCH_PER_DEVICE = 1
GRAD_ACCUM = 16
WARMUP_RATIO = 0.1
SEED = 42
FP16 = True
GRADIENT_CHECKPOINTING = True

BEAM_SIZE = 4
GREEDY_K = 1
MAX_NEW_TOKENS = 1024
EVAL_SAMPLE_LIMIT = 200
DECODE_COMPARE_LIMIT = 100

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "outputs"))
MEASURED_TOKEN_STATS = os.path.join(OUT_DIR, "token_stats.json")
HYBRID_SENTENCE_BUCKETS = 4
# 500 kata = ambang terbesar yang masih membuat 99,0% excerpt muat dalam 1024 token
# (terukur dengan tokenizer bart-large-cnn atas 300 bab train; 650 kata hanya 86,3%).
HYBRID_MAX_WORDS = 500
