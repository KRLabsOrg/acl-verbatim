"""Score char-span predictions with the official MultiSpanQA metric.

Self-contained — vendors the scoring functions from
https://github.com/haonan-li/MultiSpanQA (Li et al. 2022, NAACL) verbatim so
the reported numbers match published values byte-for-byte. Reports two F1
variants (both x100):
  - em_*:      Exact match — set intersection of normalize_answer() strings.
  - overlap_*: Longest-common-substring overlap via difflib.SequenceMatcher.
The Zilliz / Provence semantic-highlight blogs all headline overlap_f1.

Example:
    # 1. Run the model with --pred-file
    python acl_verbatim/span_training/evaluate_token_cls.py \\
        --gold-file runs/eval/test_slices/multispanqa.gold.jsonl \\
        --model-dir KRLabsOrg/verbatim-rag-modern-bert-v2 \\
        --threshold 0.2 --min-span-chars 30 --merge-gap-chars 20 \\
        --pred-file runs/eval/generic.multispanqa.preds.jsonl \\
        --output-file runs/eval/generic.multispanqa_test.json

    # 2. Score with the official MultiSpanQA metric
    python scripts/experiments/eval_multispanqa_metrics.py \\
        --multispanqa-file /Users/adamkovacs/Downloads/MultiSpanQA_data/valid.json \\
        --pred-file runs/eval/generic.multispanqa.preds.jsonl \\
        --output-file runs/eval/generic.multispanqa.official_metrics.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import string
import warnings
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# BEGIN vendored verbatim from MultiSpanQA/eval_script.py
# Source: https://github.com/haonan-li/MultiSpanQA  (Li et al. 2022, NAACL)
# --------------------------------------------------------------------------- #


def get_entities(label, token):
    def _validate_chunk(chunk):
        if chunk in ["O", "B", "I"]:
            return
        else:
            warnings.warn("{} seems not to be IOB tag.".format(chunk))

    prev_tag = "O"
    prev_type = ""
    begin_offset = 0
    chunks = []

    # check no ent
    if isinstance(label[0], list):
        for i, s in enumerate(label):
            if len(set(s)) == 1:
                chunks.append(("O", -i, -i))
    # for nested list
    if any(isinstance(s, list) for s in label):
        label = [item for sublist in label for item in sublist + ["O"]]
    if any(isinstance(s, list) for s in token):
        token = [item for sublist in token for item in sublist + ["O"]]

    for i, chunk in enumerate(label + ["O"]):
        _validate_chunk(chunk)
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):
            chunks.append((" ".join(token[begin_offset:i]), begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
    return chunks


def end_of_chunk(prev_tag, tag):
    chunk_end = False
    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "B":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True
    return chunk_end


def start_of_chunk(prev_tag, tag):
    chunk_start = False
    if tag == "B":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True
    return chunk_start


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_scores(golds, preds, eval_type="em", average="micro"):

    nb_gold = 0
    nb_pred = 0
    nb_correct = 0
    nb_correct_p = 0
    nb_correct_r = 0
    for k in list(golds.keys()):
        gold = golds[k]
        pred = preds[k]
        nb_gold += max(len(gold), 1)
        nb_pred += max(len(pred), 1)
        if eval_type == "em":
            if len(gold) == 0 and len(pred) == 0:
                nb_correct += 1
            else:
                nb_correct += len(gold.intersection(pred))
        else:
            p_score, r_score = count_overlap(gold, pred)
            nb_correct_p += p_score
            nb_correct_r += r_score

    if eval_type == "em":
        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_gold if nb_gold > 0 else 0
    else:
        p = nb_correct_p / nb_pred if nb_pred > 0 else 0
        r = nb_correct_r / nb_gold if nb_gold > 0 else 0

    f = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, f


def count_overlap(gold, pred):
    if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
        return 1, 1
    elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
        return 0, 0
    p_scores = np.zeros((len(gold), len(pred)))
    r_scores = np.zeros((len(gold), len(pred)))
    for i, s1 in enumerate(gold):
        for j, s2 in enumerate(pred):
            s = difflib.SequenceMatcher(None, s1, s2)
            _, _, longest = s.find_longest_match(0, len(s1), 0, len(s2))
            p_scores[i][j] = longest / len(s2) if longest > 0 else 0
            r_scores[i][j] = longest / len(s1) if longest > 0 else 0

    p_score = sum(np.max(p_scores, axis=0))
    r_score = sum(np.max(r_scores, axis=1))

    return p_score, r_score


def read_gold(gold_file):
    with open(gold_file) as f:
        data = json.load(f)["data"]
        golds = {}
        for piece in data:
            golds[piece["id"]] = set(
                map(lambda x: x[0], get_entities(piece["label"], piece["context"]))
            )
    return golds


def read_pred(pred_file):
    with open(pred_file) as f:
        preds = json.load(f)
    return preds


def multi_span_evaluate_from_file(pred_file, gold_file):
    preds = read_pred(pred_file)
    golds = read_gold(gold_file)
    result = multi_span_evaluate(preds, golds)
    return result


def multi_span_evaluate(preds, golds):
    assert len(preds) == len(golds)
    assert preds.keys() == golds.keys()
    # Normalize the answer
    for k, v in golds.items():
        golds[k] = set(map(lambda x: normalize_answer(x), v))
    for k, v in preds.items():
        preds[k] = set(map(lambda x: normalize_answer(x), v))
    # Evaluate
    em_p, em_r, em_f = compute_scores(golds, preds, eval_type="em")
    overlap_p, overlap_r, overlap_f = compute_scores(golds, preds, eval_type="overlap")
    result = {
        "em_precision": 100 * em_p,
        "em_recall": 100 * em_r,
        "em_f1": 100 * em_f,
        "overlap_precision": 100 * overlap_p,
        "overlap_recall": 100 * overlap_r,
        "overlap_f1": 100 * overlap_f,
    }
    return result


# --------------------------------------------------------------------------- #
# END vendored
# --------------------------------------------------------------------------- #


def load_predictions(pred_jsonl: Path, gold_ids: set[str]) -> dict[str, list[str]]:
    """Group our pred-file (JSONL) by question id; each gold id gets an entry."""
    by_id: dict[str, list[str]] = {qid: [] for qid in gold_ids}
    with pred_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            paper_id = str(row.get("paper_id", ""))
            qid = (
                paper_id.split("multispanqa/", 1)[-1]
                if "multispanqa/" in paper_id
                else paper_id
            )
            if qid not in by_id:
                continue
            for sp in row.get("pred_spans") or []:
                text = (sp.get("text") or "").strip()
                if text:
                    by_id[qid].append(text)
    return by_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--multispanqa-file",
        type=Path,
        required=True,
        help="Original MultiSpanQA valid.json",
    )
    ap.add_argument(
        "--pred-file", type=Path, required=True, help="Our standard pred-file (JSONL)"
    )
    ap.add_argument("--output-file", type=Path, default=None)
    args = ap.parse_args()

    golds = read_gold(str(args.multispanqa_file))
    preds = load_predictions(args.pred_file, set(golds.keys()))
    result = multi_span_evaluate(preds, golds)

    summary = {
        "official_metric_results": result,
        "n_questions": len(preds),
        "n_questions_with_predictions": sum(1 for v in preds.values() if v),
        "total_predicted_spans": sum(len(v) for v in preds.values()),
    }
    print(json.dumps(summary, indent=2))
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.output_file}")


if __name__ == "__main__":
    main()
