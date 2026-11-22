import math
import re
from collections import Counter

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z(])")
WORD_RE = re.compile(r"[A-Za-z']+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were", "be", "been",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "from", "that", "this", "these",
    "those", "it", "its", "he", "she", "they", "them", "his", "her", "their", "i", "you", "we",
    "me", "him", "us", "had", "has", "have", "would", "could", "should", "not", "so", "than",
    "then", "when", "where", "who", "which", "what", "all", "any", "one", "two", "do", "did",
}


def split_sentences(text, max_sentences=400):
    parts = [p.strip() for p in SENT_RE.split(text or "") if p and len(p.strip()) > 40]
    if len(parts) < 2:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 40]
    return parts[:max_sentences]


def _words(s):
    return [w for w in WORD_RE.findall(s.lower()) if w not in STOPWORDS]


def tfidf_scores(sentences):
    docs = [_words(s) for s in sentences]
    n = len(docs)
    if n == 0:
        return []
    df = Counter()
    for d in docs:
        df.update(set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        total = sum(tf.values()) or 1
        s = 0.0
        for w, c in tf.items():
            s += (c / total) * math.log((n + 1) / (df[w] + 1))
        scores.append(s / math.sqrt(len(tf) or 1))
    return scores


def select_representative(text, max_words=650, buckets=4):
    """Position-aware TF-IDF selection: keep top sentences from every segment of the chapter."""
    sentences = split_sentences(text, max_sentences=None)
    if not sentences:
        return text[: max_words * 6]
    scores = tfidf_scores(sentences)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    buckets = max(1, min(buckets, len(sentences)))
    rel = [1.0] * buckets
    rel[0] *= 1.2
    rel[-1] *= 1.1
    mean_rel = sum(rel) / buckets
    size = len(sentences) / buckets
    quota = {b: max(1, math.ceil(size * rel[b] / mean_rel)) for b in range(buckets)}
    chosen, used_words = [], 0
    for idx in ranked:
        b = min(buckets - 1, (idx * buckets) // len(sentences))
        if quota[b] <= 0:
            continue
        w = len(sentences[idx].split())
        if used_words + w > max_words and chosen:
            continue
        chosen.append(idx)
        quota[b] -= 1
        used_words += w
        if used_words >= max_words:
            break
    chosen = sorted(chosen)
    out = " ".join(sentences[i] for i in chosen)
    words = len(out.split())
    if words <= max_words:
        return out
    overflow = words - max_words
    for idx in chosen[1:-1]:
        if overflow <= 0:
            break
        d = len(sentences[idx].split())
        overflow -= d
        out = out.replace(sentences[idx] + " ", "", 1)
    return out
