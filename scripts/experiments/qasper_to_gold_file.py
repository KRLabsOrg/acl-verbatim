"""Convert QASPER to the gold-file JSONL format used by our span evaluators.

QASPER provides paper full text and per-question evidence annotations. This
converter defaults to a fair chunk-level benchmark: one row per evidence
paragraph/table plus sampled distractor paragraphs/tables from the same paper.
Use `--context-mode full_paper` for a long-context stress test with one row per
question and the whole paper as context.

By default we use QASPER's paragraph-level `evidence` field, because the dataset
defines it as the paragraphs, figures, or tables used to answer the question.
Use `--evidence-field highlighted_evidence` for stricter sentence-level labels.

Example:
    python scripts/experiments/qasper_to_gold_file.py \
        --split test \
        --output runs/eval/test_slices/qasper.gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset

from acl_verbatim.data.spans import load_gold_rows, summarize


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def dedupe_texts(texts: Iterable[str]) -> list[str]:
    """Deduplicate non-empty strings while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        text = (text or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def make_float_text(caption: str) -> str:
    return f"FLOAT SELECTED: {caption.strip()}"


def iter_chunks(example: dict) -> list[dict]:
    """Return paragraph/table chunks in paper order."""
    chunks: list[dict] = []
    full_text = example.get("full_text") or {}
    section_names = _as_list(full_text.get("section_name"))
    paragraphs_by_section = _as_list(full_text.get("paragraphs"))
    chunk_index = 0
    for section_idx, (section_name, paragraphs) in enumerate(
        zip(section_names, paragraphs_by_section)
    ):
        section_name = (section_name or "").strip()
        for paragraph_idx, paragraph in enumerate(_as_list(paragraphs)):
            paragraph = (paragraph or "").strip()
            if not paragraph:
                continue
            chunk = f"{section_name}\n\n{paragraph}" if section_name else paragraph
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "chunk": chunk,
                    "evidence_text": paragraph,
                    "kind": "paragraph",
                    "section_name": section_name,
                    "section_index": section_idx,
                    "paragraph_index": paragraph_idx,
                }
            )
            chunk_index += 1

    figures_and_tables = example.get("figures_and_tables") or {}
    captions = _as_list(figures_and_tables.get("caption"))
    files = _as_list(figures_and_tables.get("file"))
    for float_idx, caption in enumerate(captions):
        caption = (caption or "").strip()
        if not caption:
            continue
        float_text = make_float_text(caption)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk": float_text,
                "evidence_text": float_text,
                "kind": "float",
                "float_index": float_idx,
                "file": files[float_idx] if float_idx < len(files) else None,
            }
        )
        chunk_index += 1

    return chunks


def flatten_full_text(example: dict) -> str:
    """Render QASPER paper fields into one plain-text context."""
    pieces: list[str] = []
    title = (example.get("title") or "").strip()
    abstract = (example.get("abstract") or "").strip()
    if title:
        pieces.append(f"Title: {title}")
    if abstract:
        pieces.append(f"Abstract\n\n{abstract}")

    full_text = example.get("full_text") or {}
    section_names = _as_list(full_text.get("section_name"))
    paragraphs_by_section = _as_list(full_text.get("paragraphs"))
    for section_name, paragraphs in zip(section_names, paragraphs_by_section):
        section_name = (section_name or "").strip()
        paragraphs = [p.strip() for p in _as_list(paragraphs) if (p or "").strip()]
        if not paragraphs:
            continue
        if section_name:
            pieces.append(f"{section_name}\n\n" + "\n\n".join(paragraphs))
        else:
            pieces.append("\n\n".join(paragraphs))

    float_blocks = [
        chunk_info["evidence_text"]
        for chunk_info in iter_chunks(example)
        if chunk_info["kind"] == "float"
    ]
    if float_blocks:
        pieces.append("Figures and tables\n\n" + "\n\n".join(float_blocks))

    return "\n\n".join(pieces)


def iter_question_records(qas: dict) -> Iterable[dict]:
    """Yield one normalized QASPER question record from the columnar HF form."""
    questions = _as_list(qas.get("question"))
    question_ids = _as_list(qas.get("question_id"))
    answers = _as_list(qas.get("answers"))
    for idx, question in enumerate(questions):
        yield {
            "question": question,
            "question_id": question_ids[idx] if idx < len(question_ids) else str(idx),
            "answers": answers[idx] if idx < len(answers) else {},
        }


def iter_answer_payloads(answers_obj: dict) -> Iterable[dict]:
    """Yield individual answer payloads from QASPER's nested answer structure."""
    answer_payloads = answers_obj.get("answer") if isinstance(answers_obj, dict) else []
    if isinstance(answer_payloads, dict):
        keys = list(answer_payloads)
        values = {key: _as_list(answer_payloads[key]) for key in keys}
        n = max((len(v) for v in values.values()), default=0)
        for idx in range(n):
            yield {
                key: values[key][idx] if idx < len(values[key]) else None
                for key in keys
            }
    else:
        for answer in _as_list(answer_payloads):
            if isinstance(answer, dict):
                yield answer


def evidence_for_question(
    question_record: dict,
    evidence_field: str,
    include_unanswerable: bool,
) -> list[str]:
    evidence: list[str] = []
    for answer in iter_answer_payloads(question_record.get("answers") or {}):
        if answer.get("unanswerable") and not include_unanswerable:
            continue
        evidence.extend(_as_list(answer.get(evidence_field)))
    return dedupe_texts(evidence)


def make_gold_record(
    *,
    question: str,
    paper_id: str,
    question_id: str,
    chunk: str,
    chunk_index: int,
    evidence: list[str],
    evidence_field: str,
    metadata: dict,
) -> dict:
    document_id = f"qasper/{paper_id}"
    is_relevant = bool(evidence)
    return {
        "query": question,
        "gold_paper": document_id,
        "gold_chunk": chunk_index,
        "results": [
            {
                "document_id": document_id,
                "chunk_number": chunk_index,
                "chunk": chunk,
                "relevance_label": "r" if is_relevant else "n",
                "gold_extraction": evidence if is_relevant else [],
                "metadata": {
                    "dataset": "qasper",
                    "paper_id": paper_id,
                    "question_id": question_id,
                    "evidence_field": evidence_field,
                    **metadata,
                },
            }
        ],
    }


def convert_paper_full(
    example: dict,
    evidence_field: str,
    include_unanswerable: bool,
    relevant_only: bool,
) -> list[dict]:
    paper_id = str(example.get("id") or "")
    chunk = flatten_full_text(example)
    rows: list[dict] = []
    for question_record in iter_question_records(example.get("qas") or {}):
        evidence = evidence_for_question(
            question_record,
            evidence_field=evidence_field,
            include_unanswerable=include_unanswerable,
        )
        if relevant_only and not evidence:
            continue
        rows.append(
            make_gold_record(
                question=question_record["question"],
                paper_id=paper_id,
                question_id=str(question_record.get("question_id") or ""),
                chunk=chunk,
                chunk_index=0,
                evidence=evidence,
                evidence_field=evidence_field,
                metadata={"context_mode": "full_paper"},
            )
        )
    return rows


def convert_paper_paragraph(
    example: dict,
    evidence_field: str,
    include_unanswerable: bool,
    negative_ratio: int,
    rng: random.Random,
) -> list[dict]:
    paper_id = str(example.get("id") or "")
    chunks = iter_chunks(example)
    rows: list[dict] = []
    for question_record in iter_question_records(example.get("qas") or {}):
        evidence = evidence_for_question(
            question_record,
            evidence_field=evidence_field,
            include_unanswerable=include_unanswerable,
        )
        if not evidence:
            continue

        question = question_record["question"]
        question_id = str(question_record.get("question_id") or "")
        positives: list[tuple[dict, list[str]]] = []
        used_chunk_indices: set[int] = set()
        for chunk_info in chunks:
            chunk_evidence = [
                ev
                for ev in evidence
                if ev == chunk_info["evidence_text"] or ev in chunk_info["chunk"]
            ]
            if not chunk_evidence:
                continue
            used_chunk_indices.add(chunk_info["chunk_index"])
            positives.append((chunk_info, dedupe_texts(chunk_evidence)))

        for chunk_info, chunk_evidence in positives:
            rows.append(
                make_gold_record(
                    question=question,
                    paper_id=paper_id,
                    question_id=question_id,
                    chunk=chunk_info["chunk"],
                    chunk_index=chunk_info["chunk_index"],
                    evidence=chunk_evidence,
                    evidence_field=evidence_field,
                    metadata={
                        "context_mode": "paragraph",
                        "kind": chunk_info.get("kind"),
                    },
                )
            )

        if negative_ratio <= 0 or not positives:
            continue
        candidates = [
            chunk_info
            for chunk_info in chunks
            if chunk_info["chunk_index"] not in used_chunk_indices
        ]
        n_neg = min(len(candidates), negative_ratio * len(positives))
        for chunk_info in rng.sample(candidates, n_neg):
            rows.append(
                make_gold_record(
                    question=question,
                    paper_id=paper_id,
                    question_id=question_id,
                    chunk=chunk_info["chunk"],
                    chunk_index=chunk_info["chunk_index"],
                    evidence=[],
                    evidence_field=evidence_field,
                    metadata={
                        "context_mode": "paragraph",
                        "kind": chunk_info.get("kind"),
                    },
                )
            )

    return rows


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="allenai/qasper")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-mode",
        choices=("paragraph", "full_paper"),
        default="paragraph",
        help="Use paragraph/table chunks by default; full_paper is a stress test.",
    )
    parser.add_argument(
        "--evidence-field",
        choices=("evidence", "highlighted_evidence"),
        default="evidence",
        help="QASPER evidence field to use as gold spans.",
    )
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Keep evidence attached to unanswerable annotations if present.",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Only used with --context-mode full_paper: keep no-evidence questions.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=2,
        help="Paragraph mode: sample this many non-evidence chunks per positive chunk.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    dataset = load_dataset(args.repo_id, split=args.split)
    if args.max_papers is not None:
        dataset = dataset.select(range(min(args.max_papers, len(dataset))))

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_papers = n_rows = n_relevant = 0
    with args.output.open("w") as f:
        for example in dataset:
            n_papers += 1
            if args.context_mode == "full_paper":
                rows = convert_paper_full(
                    example,
                    evidence_field=args.evidence_field,
                    include_unanswerable=args.include_unanswerable,
                    relevant_only=not args.include_negative,
                )
            else:
                rows = convert_paper_paragraph(
                    example,
                    evidence_field=args.evidence_field,
                    include_unanswerable=args.include_unanswerable,
                    negative_ratio=args.negative_ratio,
                    rng=rng,
                )
            for row in rows:
                if args.max_rows is not None and n_rows >= args.max_rows:
                    break
                f.write(json.dumps(row) + "\n")
                n_rows += 1
                if row["results"][0]["relevance_label"] == "r":
                    n_relevant += 1
            if args.max_rows is not None and n_rows >= args.max_rows:
                break

    summary = summarize(load_gold_rows(args.output))
    print(
        f"read {n_papers} papers -> wrote {n_rows} rows "
        f"({n_relevant} relevant) to {args.output}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
