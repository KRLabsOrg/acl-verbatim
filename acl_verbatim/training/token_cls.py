import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from acl_verbatim.core.jsonl import iter_jsonl


def build_token_labels(seq_ids, offsets, spans, label_scheme):
    labels = []
    for seq_id, (start, end) in zip(seq_ids, offsets):
        if seq_id != 1 or start == end:
            labels.append(-100)
            continue

        token_label = 0
        for span in spans:
            s = int(span["start"])
            e = int(span["end"])
            if end <= s or start >= e:
                continue
            if label_scheme == "binary":
                token_label = 1
            elif start == s:
                token_label = 1
            else:
                token_label = 2
            break
        labels.append(token_label)
    return labels


def tokenize_row_to_windows(tokenizer, question, chunk, max_length, doc_stride):
    kwargs = {
        "text": question,
        "text_pair": chunk,
        "return_offsets_mapping": True,
        "max_length": max_length,
        "truncation": "only_second",
    }
    if doc_stride > 0:
        kwargs["stride"] = doc_stride
        kwargs["return_overflowing_tokens"] = True
    return tokenizer(**kwargs)


def row_to_token_examples(
    row: dict,
    tokenizer,
    max_length: int,
    doc_stride: int,
    label_scheme: str,
    drop_unlabeled_positives: bool,
):
    question = row.get("question")
    chunk = row.get("chunk")
    spans = row.get("spans", [])
    label = row.get("label", 0)
    if not question or chunk is None:
        return []

    enc = tokenize_row_to_windows(tokenizer, question, chunk, max_length, doc_stride)
    num_windows = len(enc["input_ids"])
    window_examples = []
    has_positive_window = False

    for i in range(num_windows):
        seq_ids = enc.sequence_ids(i)
        offsets = enc["offset_mapping"][i]
        labels = build_token_labels(seq_ids, offsets, spans, label_scheme)
        has_labels = any(l not in (-100, 0) for l in labels)
        has_positive_window = has_positive_window or has_labels
        window_examples.append(
            {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": labels,
            }
        )

    if label == 1 and drop_unlabeled_positives and not has_positive_window:
        return []
    return window_examples


def write_token_cls_dataset(
    input_file: str,
    output_file: str,
    tokenizer_name: str,
    max_length: int,
    drop_unlabeled_positives: bool,
    label_scheme: str,
    doc_stride: int = 256,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as of:
        for row in tqdm(list(iter_jsonl(Path(input_file)))):
            for example in row_to_token_examples(
                row=row,
                tokenizer=tokenizer,
                max_length=max_length,
                doc_stride=doc_stride,
                label_scheme=label_scheme,
                drop_unlabeled_positives=drop_unlabeled_positives,
            ):
                of.write(json.dumps(example) + "\n")


def compute_token_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    tp = fp = fn = 0
    for pred_row, label_row in zip(preds, labels):
        for pred, gold in zip(pred_row, label_row):
            if gold == -100:
                continue
            pred_pos = pred != 0
            gold_pos = gold != 0
            if pred_pos and gold_pos:
                tp += 1
            elif pred_pos and not gold_pos:
                fp += 1
            elif gold_pos and not pred_pos:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def label_maps(label_scheme: str):
    if label_scheme == "binary":
        id2label = {0: "O", 1: "ANS"}
    else:
        id2label = {0: "O", 1: "B-ANS", 2: "I-ANS"}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def train_token_classifier(
    train_file: str,
    eval_file: str,
    hf_dataset: str | None,
    hf_config: str,
    train_split: str,
    eval_split: str,
    model_name: str,
    output_dir: str,
    batch_size: int,
    lr: float,
    epochs: int,
    seed: int,
    label_scheme: str,
):
    id2label, label2id = label_maps(label_scheme)
    num_labels = len(id2label)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    if hasattr(config, "reference_compile"):
        # ModernBERT may auto-enable torch.compile when Triton is available.
        # That can conflict with Trainer/FX tracing paths during eval/checkpointing.
        config.reference_compile = False
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    if hf_dataset:
        train_ds = load_dataset(hf_dataset, hf_config, split=train_split)
        eval_ds = load_dataset(hf_dataset, hf_config, split=eval_split)
    else:
        train_ds = load_dataset("json", data_files=train_file, split="train")
        eval_ds = load_dataset("json", data_files=eval_file, split="train")
    collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=600,
        save_steps=600,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_token_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def spans_from_preds(chunk, offsets, seq_ids, pred_labels):
    spans = []
    current = None
    for seq_id, (start, end), label in zip(seq_ids, offsets, pred_labels):
        if seq_id != 1 or start == end:
            if current:
                spans.append(current)
                current = None
            continue
        if label == 0:
            if current:
                spans.append(current)
                current = None
            continue
        if current is None:
            current = {"start": start, "end": end}
        else:
            current["end"] = end
    if current:
        spans.append(current)

    return spans


def merge_char_spans(spans):
    if not spans:
        return []
    spans = sorted(spans, key=lambda sp: (sp["start"], sp["end"]))
    merged = [dict(spans[0])]
    for sp in spans[1:]:
        last = merged[-1]
        if sp["start"] <= last["end"]:
            last["end"] = max(last["end"], sp["end"])
        else:
            merged.append(dict(sp))
    return merged


def predict_token_records(
    rows: list[dict],
    model_dir: str,
    max_length: int,
    batch_size: int = 4,
    doc_stride: int = 256,
) -> list[dict]:
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    for row in tqdm(rows):
        question = row.get("question")
        chunk = row.get("chunk")
        if not question or chunk is None:
            continue

        enc = tokenize_row_to_windows(
            tokenizer, question, chunk, max_length, doc_stride
        )
        num_windows = len(enc["input_ids"])
        all_spans = []

        for start_idx in range(0, num_windows, batch_size):
            end_idx = min(num_windows, start_idx + batch_size)
            input_ids = torch.tensor(enc["input_ids"][start_idx:end_idx], device=device)
            attention_mask = torch.tensor(
                enc["attention_mask"][start_idx:end_idx], device=device
            )
            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
            preds = logits.argmax(dim=-1).cpu().tolist()

            for offset, pred in enumerate(preds):
                window_idx = start_idx + offset
                seq_ids = enc.sequence_ids(window_idx)
                offsets = enc["offset_mapping"][window_idx]
                all_spans.extend(spans_from_preds(chunk, offsets, seq_ids, pred))

        merged_spans = merge_char_spans(all_spans)
        predictions.append(
            {
                "question": question,
                "paper_id": row.get("paper_id"),
                "chunk_index": row.get("chunk_index"),
                "pred_spans": [
                    {
                        "start": sp["start"],
                        "end": sp["end"],
                        "text": chunk[sp["start"] : sp["end"]],
                    }
                    for sp in merged_spans
                ],
            }
        )
    return predictions


def predict_token_spans(
    input_file: str,
    output_file: str,
    model_dir: str,
    max_length: int,
    batch_size: int = 4,
    doc_stride: int = 256,
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(Path(input_file)))
    predictions = predict_token_records(
        rows=rows,
        model_dir=model_dir,
        max_length=max_length,
        batch_size=batch_size,
        doc_stride=doc_stride,
    )
    with output_path.open("w") as of:
        for record in predictions:
            of.write(json.dumps(record) + "\n")
