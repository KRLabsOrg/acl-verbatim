"""Query-conditioned evidence highlighter over ModernBERT.

Loaded via:

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "KRLabsOrg/acl-verbatim-modernbert",
        trust_remote_code=True,
    )
    result = model.process(question, context, threshold=0.5)

Returns character-aligned evidence spans in ``context`` that answer ``question``.
"""
from __future__ import annotations

import re

import torch
from transformers import AutoTokenizer
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertForTokenClassification,
)


_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?:\s+|$)|\n{2,}|$)", re.DOTALL)


class AclVerbatimHighlighter(ModernBertForTokenClassification):
    """ModernBERT token classifier with a high-level highlight API."""

    def __init__(self, config):
        super().__init__(config)
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config._name_or_path, use_fast=True
            )
        return self._tokenizer

    @torch.inference_mode()
    def process(
        self,
        question: str,
        context: str,
        threshold: float = 0.2,
        max_length: int = 8192,
        doc_stride: int = 256,
        min_span_chars: int = 10,
        merge_gap_chars: int = 20,
        return_sentence_metrics: bool = False,
    ) -> dict:
        tokenizer = self._get_tokenizer()
        enc = tokenizer(
            question,
            context,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        logits = self(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
        positive = torch.softmax(logits, dim=-1)[..., 1:].sum(dim=-1).cpu()

        raw: list[tuple[int, int, float]] = []
        char_scores: dict[int, float] = {}
        for w in range(enc["input_ids"].size(0)):
            seq_ids = enc.sequence_ids(w)
            offsets = enc["offset_mapping"][w].tolist()
            cur: list | None = None
            for sid, (s, e), p in zip(seq_ids, offsets, positive[w].tolist()):
                is_context_token = sid == 1 and s != e
                if is_context_token and p >= threshold:
                    for i in range(s, e):
                        char_scores[i] = max(char_scores.get(i, 0.0), p)
                    cur = [s, e, p] if cur is None else [cur[0], e, max(cur[2], p)]
                else:
                    if cur is not None:
                        raw.append(tuple(cur))
                        cur = None
            if cur is not None:
                raw.append(tuple(cur))

        raw.sort()
        merged: list[list] = []
        for s, e, p in raw:
            if merged and s - merged[-1][1] <= merge_gap_chars:
                merged[-1][1] = max(merged[-1][1], e)
                merged[-1][2] = max(merged[-1][2], p)
            else:
                merged.append([s, e, p])
        merged = [sp for sp in merged if sp[1] - sp[0] >= min_span_chars]

        result: dict = {
            "spans": [
                {"start": s, "end": e, "text": context[s:e], "score": float(p)}
                for s, e, p in merged
            ],
        }
        if return_sentence_metrics:
            result["sentences"] = self._sentence_scores(context, char_scores)
        return result

    @staticmethod
    def _sentence_scores(context: str, char_scores: dict[int, float]) -> list[dict]:
        out = []
        for match in _SENTENCE_RE.finditer(context):
            raw = match.group(0)
            if not raw.strip():
                continue
            start = match.start() + (len(raw) - len(raw.lstrip()))
            end = match.end() - (len(raw) - len(raw.rstrip()))
            length = max(1, end - start)
            total = sum(char_scores.get(i, 0.0) for i in range(start, end))
            out.append(
                {
                    "start": start,
                    "end": end,
                    "text": context[start:end],
                    "score": total / length,
                }
            )
        return out
