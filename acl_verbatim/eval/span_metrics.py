"""Shared span-scoring utilities for extractor evaluation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from acl_verbatim.data.spans import SpanRow

IOU_THRESHOLDS = (0.3, 0.5, 0.7)
CONTAINMENT_THRESHOLDS = (0.5, 0.8, 1.0)
GOLD_COVERAGE_THRESHOLDS = (0.5, 0.8, 1.0)

_WORD_RE = re.compile(r"\S+")


def tokenize(chunk: str) -> list[tuple[int, int]]:
    """Return the (start, end) offsets of each whitespace-delimited word."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(chunk)]


def spans_to_word_indices(
    words: list[tuple[int, int]], spans: list[tuple[int, int]]
) -> set[int]:
    """Indices of any word that overlaps any span (non-zero char intersection)."""
    covered: set[int] = set()
    for span_s, span_e in spans:
        for i, (w_s, w_e) in enumerate(words):
            if w_e <= span_s or w_s >= span_e:
                continue
            covered.add(i)
    return covered


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def align_predicted_texts(chunk: str, pred_texts: list[str]) -> list[tuple[int, int]]:
    """Map predicted span strings to (start, end) offsets in the chunk."""
    spans: list[tuple[int, int]] = []
    for text in pred_texts:
        if not text:
            continue
        idx = chunk.find(text)
        if idx == -1:
            idx = chunk.lower().find(text.lower())
        if idx == -1:
            continue
        spans.append((idx, idx + len(text)))
    return spans


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def span_iou_prf(
    pred_spans: list[tuple[int, int]],
    gold_spans: list[tuple[int, int]],
    threshold: float,
) -> tuple[float, float, float, int, int, int]:
    """Greedy bipartite span-level P/R/F1 at a given IoU threshold."""
    n_pred, n_gold = len(pred_spans), len(gold_spans)
    if n_pred == 0 and n_gold == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if n_pred == 0 or n_gold == 0:
        return 0.0, 0.0, 0.0, 0, n_pred, n_gold
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred_spans):
        for j, g in enumerate(gold_spans):
            pairs.append((iou(p, g), i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0
    for score, i_idx, j_idx in pairs:
        if score < threshold:
            break
        if i_idx in used_p or j_idx in used_g:
            continue
        used_p.add(i_idx)
        used_g.add(j_idx)
        tp += 1
    fp = n_pred - tp
    fn = n_gold - tp
    p, r, f1 = prf(tp, fp, fn)
    return p, r, f1, tp, fp, fn


def recall_any_overlap(
    pred_spans: list[tuple[int, int]], gold_spans: list[tuple[int, int]]
) -> tuple[int, int]:
    """Count gold spans with at least one overlapping prediction."""
    hit = 0
    for gold in gold_spans:
        for pred in pred_spans:
            if iou(pred, gold) > 0:
                hit += 1
                break
    return hit, len(gold_spans)


def intersect(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def length(a: tuple[int, int]) -> int:
    return max(0, a[1] - a[0])


def containment_prf(
    pred_spans: list[tuple[int, int]],
    gold_spans: list[tuple[int, int]],
    threshold: float,
) -> tuple[float, float, float, int, int, int, int]:
    """Containment-based P/R/F1."""
    n_pred, n_gold = len(pred_spans), len(gold_spans)
    if n_pred == 0 and n_gold == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0, 0
    if n_pred == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0, n_gold
    if n_gold == 0:
        return 0.0, 0.0, 0.0, 0, n_pred, 0, 0

    pred_matched = [False] * n_pred
    gold_matched = [False] * n_gold
    for i_idx, pred in enumerate(pred_spans):
        pred_len = length(pred)
        if pred_len == 0:
            continue
        for j_idx, gold in enumerate(gold_spans):
            if intersect(pred, gold) / pred_len >= threshold:
                pred_matched[i_idx] = True
                gold_matched[j_idx] = True
    pred_tp = sum(pred_matched)
    pred_fp = n_pred - pred_tp
    gold_tp = sum(gold_matched)
    gold_fn = n_gold - gold_tp
    precision = pred_tp / n_pred if n_pred else 0.0
    recall = gold_tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, pred_tp, pred_fp, gold_tp, gold_fn


def gold_coverage_recall(
    pred_spans: list[tuple[int, int]],
    gold_spans: list[tuple[int, int]],
    threshold: float,
) -> tuple[float, int, int]:
    """Fraction of gold spans whose char-coverage by any predictions ≥ threshold."""
    n_gold = len(gold_spans)
    if n_gold == 0:
        return 1.0, 0, 0
    pred_chars: set[int] = set()
    for start, end in pred_spans:
        pred_chars.update(range(start, end))
    covered = 0
    for gold_s, gold_e in gold_spans:
        gold_len = gold_e - gold_s
        if gold_len <= 0:
            continue
        covered_chars = sum(1 for i in range(gold_s, gold_e) if i in pred_chars)
        if covered_chars / gold_len >= threshold:
            covered += 1
    return covered / n_gold, covered, n_gold


def normalize_pred_spans(
    pred_record: dict, chunk: str
) -> tuple[list[str], list[tuple[int, int]]]:
    """Normalize predicted spans from either exact offsets or raw strings."""
    pred_spans_raw = pred_record.get("pred_spans") or []
    if pred_spans_raw:
        spans = []
        texts = []
        for span in pred_spans_raw:
            if not isinstance(span, dict):
                continue
            if "start" not in span or "end" not in span:
                continue
            start = int(span["start"])
            end = int(span["end"])
            if start < 0 or end <= start or end > len(chunk):
                continue
            text = span.get("text")
            if text is None:
                text = chunk[start:end]
            spans.append((start, end))
            texts.append(str(text))
        return texts, spans

    pred_texts = list(pred_record.get("predicted_texts") or [])
    return pred_texts, align_predicted_texts(chunk, pred_texts)


def make_prediction_key(
    query: str, paper_id: str, chunk_index: int
) -> tuple[str, str, int]:
    return query, paper_id, chunk_index


def evaluate_rows_against_predictions(
    rows: Iterable[SpanRow], pred_map: dict[tuple[str, str, int], dict]
) -> dict:
    """Score arbitrary predictions against canonical gold rows."""
    records: list[dict] = []
    word_tp = word_fp = word_fn = 0
    word_f1_list: list[float] = []
    iou_counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in IOU_THRESHOLDS}
    iou_f1_lists: dict[float, list[float]] = {t: [] for t in IOU_THRESHOLDS}
    contain_counts = {
        t: {"pred_tp": 0, "pred_fp": 0, "gold_tp": 0, "gold_fn": 0}
        for t in CONTAINMENT_THRESHOLDS
    }
    contain_f1_lists: dict[float, list[float]] = {t: [] for t in CONTAINMENT_THRESHOLDS}
    cov_counts = {t: {"covered": 0, "total": 0} for t in GOLD_COVERAGE_THRESHOLDS}
    any_overlap_hit = 0
    any_overlap_total = 0
    pred_total = 0
    gold_total = 0
    latencies: list[float] = []

    relevant = [row for row in rows if row.is_relevant]

    for row in relevant:
        key = make_prediction_key(row.query, row.paper_id, row.chunk_index)
        pred_record = pred_map.get(key, {})
        predicted_texts, pred_spans = normalize_pred_spans(pred_record, row.chunk)
        gold_spans = [(span.start, span.end) for span in row.gold_spans]
        error = pred_record.get("error")
        latency_s = float(pred_record.get("latency_s", 0.0) or 0.0)
        latencies.append(latency_s)

        words = tokenize(row.chunk)
        pred_words = spans_to_word_indices(words, pred_spans)
        gold_words = spans_to_word_indices(words, gold_spans)
        tp = len(pred_words & gold_words)
        fp = len(pred_words - gold_words)
        fn = len(gold_words - pred_words)
        wp, wr, wf = prf(tp, fp, fn)
        word_tp += tp
        word_fp += fp
        word_fn += fn
        word_f1_list.append(wf)

        per_thr: dict[str, dict] = {}
        for thr in IOU_THRESHOLDS:
            sp, sr, sf, stp, sfp, sfn = span_iou_prf(pred_spans, gold_spans, thr)
            iou_counts[thr]["tp"] += stp
            iou_counts[thr]["fp"] += sfp
            iou_counts[thr]["fn"] += sfn
            iou_f1_lists[thr].append(sf)
            per_thr[str(thr)] = {
                "p": sp,
                "r": sr,
                "f1": sf,
                "tp": stp,
                "fp": sfp,
                "fn": sfn,
            }

        contain_per_thr: dict[str, dict] = {}
        for thr in CONTAINMENT_THRESHOLDS:
            cp, cr, cf, ptp, pfp, gtp, gfn = containment_prf(
                pred_spans, gold_spans, thr
            )
            contain_counts[thr]["pred_tp"] += ptp
            contain_counts[thr]["pred_fp"] += pfp
            contain_counts[thr]["gold_tp"] += gtp
            contain_counts[thr]["gold_fn"] += gfn
            contain_f1_lists[thr].append(cf)
            contain_per_thr[str(thr)] = {"p": cp, "r": cr, "f1": cf}

        cov_per_thr: dict[str, float] = {}
        for thr in GOLD_COVERAGE_THRESHOLDS:
            rec, covered, total_g = gold_coverage_recall(pred_spans, gold_spans, thr)
            cov_counts[thr]["covered"] += covered
            cov_counts[thr]["total"] += total_g
            cov_per_thr[str(thr)] = rec

        hit, total = recall_any_overlap(pred_spans, gold_spans)
        any_overlap_hit += hit
        any_overlap_total += total
        pred_total += len(pred_spans)
        gold_total += len(gold_spans)

        records.append(
            {
                "paper_id": row.paper_id,
                "chunk_index": row.chunk_index,
                "query": row.query,
                "gold_spans": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in row.gold_spans
                ],
                "predicted_texts": predicted_texts,
                "predicted_spans": [
                    {"start": start, "end": end} for start, end in pred_spans
                ],
                "word_metrics": {
                    "p": wp,
                    "r": wr,
                    "f1": wf,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                },
                "iou_metrics_by_threshold": per_thr,
                "containment_metrics_by_threshold": contain_per_thr,
                "gold_coverage_recall_by_threshold": cov_per_thr,
                "any_overlap_hits": hit,
                "gold_count": len(gold_spans),
                "pred_count": len(pred_spans),
                "latency_s": latency_s,
                "error": error,
            }
        )

    word_micro_p, word_micro_r, word_micro_f1 = prf(word_tp, word_fp, word_fn)
    iou_summary = {}
    for thr in IOU_THRESHOLDS:
        counts = iou_counts[thr]
        p, r, f1 = prf(counts["tp"], counts["fp"], counts["fn"])
        iou_summary[str(thr)] = {
            "micro_p": p,
            "micro_r": r,
            "micro_f1": f1,
            "macro_f1": sum(iou_f1_lists[thr]) / len(iou_f1_lists[thr])
            if iou_f1_lists[thr]
            else 0.0,
        }
    contain_summary = {}
    for thr in CONTAINMENT_THRESHOLDS:
        counts = contain_counts[thr]
        micro_p = (
            counts["pred_tp"] / (counts["pred_tp"] + counts["pred_fp"])
            if (counts["pred_tp"] + counts["pred_fp"])
            else 0.0
        )
        micro_r = (
            counts["gold_tp"] / (counts["gold_tp"] + counts["gold_fn"])
            if (counts["gold_tp"] + counts["gold_fn"])
            else 0.0
        )
        micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
        )
        contain_summary[str(thr)] = {
            "micro_p": micro_p,
            "micro_r": micro_r,
            "micro_f1": micro_f1,
            "macro_f1": sum(contain_f1_lists[thr]) / len(contain_f1_lists[thr])
            if contain_f1_lists[thr]
            else 0.0,
        }
    cov_summary = {}
    for thr in GOLD_COVERAGE_THRESHOLDS:
        counts = cov_counts[thr]
        cov_summary[str(thr)] = (
            counts["covered"] / counts["total"] if counts["total"] else 0.0
        )

    summary = {
        "n_examples": len(relevant),
        "word_level": {
            "micro_precision": word_micro_p,
            "micro_recall": word_micro_r,
            "micro_f1": word_micro_f1,
            "macro_f1": sum(word_f1_list) / len(word_f1_list) if word_f1_list else 0.0,
        },
        "span_level_iou": iou_summary,
        "containment": contain_summary,
        "gold_coverage_recall": cov_summary,
        "recall_any_overlap": (any_overlap_hit / any_overlap_total)
        if any_overlap_total
        else 0.0,
        "over_prediction_ratio": (pred_total / gold_total) if gold_total else 0.0,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "errors": sum(1 for record in records if record["error"]),
    }
    return {"summary": summary, "records": records}
