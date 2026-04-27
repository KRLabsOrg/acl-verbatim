import random
import re
from collections import Counter
from dataclasses import asdict, dataclass

REFERENCE_HEADER_RE = re.compile(r"(?im)^\s{0,3}(#+\s*)?(references|bibliography)\s*$")
CAPTION_RE = re.compile(r"(?i)\b(table|figure|fig\.?)\s+\d+")
CITATION_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z'`-]+(?:\s+et\s+al\.)?|\([^)]+)\s*,?\s*(?:19|20)\d{2}[a-z]?\)?"
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")


@dataclass
class SilverFilterConfig:
    seed: int = 1337
    dev_fraction: float = 0.1
    min_span_words: int = 6
    max_span_words: int = 220
    max_citation_density: float = 0.12
    max_year_density: float = 0.08
    max_positive_rank: int | None = None
    drop_caption_like: bool = False


def count_words(text: str) -> int:
    return len(text.split())


def is_reference_like(text: str) -> bool:
    return bool(REFERENCE_HEADER_RE.search(text))


def is_caption_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(CAPTION_RE.search(stripped.splitlines()[0]))


def density(pattern: re.Pattern[str], text: str) -> float:
    return len(pattern.findall(text)) / max(1, count_words(text))


def positive_drop_reason(row: dict, config: SilverFilterConfig) -> str | None:
    if row.get("label", 0) != 1:
        return None

    spans = row.get("spans") or []
    if not spans:
        return "positive_without_spans"

    if row.get("source") == "retrieved" and config.max_positive_rank is not None:
        rank = row.get("retrieval_rank")
        if rank is not None and rank > config.max_positive_rank:
            return "rank_too_deep"

    span_text = "\n".join(span.get("text", "") for span in spans).strip()
    chunk_text = row.get("chunk", "")
    span_words = count_words(span_text)

    if span_words < config.min_span_words:
        return "span_too_short"
    if span_words > config.max_span_words:
        return "span_too_long"
    if is_reference_like(span_text) or is_reference_like(chunk_text[:2500]):
        return "reference_section"
    if config.drop_caption_like and is_caption_like(span_text):
        return "caption_like"
    if density(CITATION_RE, span_text) > config.max_citation_density:
        return "citation_dense"
    if density(YEAR_RE, span_text) > config.max_year_density:
        return "year_dense"
    return None


def group_key(row: dict):
    return (row.get("question"), row.get("gold_paper"), row.get("gold_chunk"))


def filter_and_split_rows(rows: list[dict], config: SilverFilterConfig):
    kept_rows = []
    dropped_rows = []
    drop_reasons = Counter()

    for row in rows:
        reason = positive_drop_reason(row, config)
        if reason is None:
            kept_rows.append(row)
        else:
            dropped = dict(row)
            dropped["drop_reason"] = reason
            dropped_rows.append(dropped)
            drop_reasons[reason] += 1

    query_groups = {}
    for row in kept_rows:
        query_groups.setdefault(group_key(row), []).append(row)

    keys = list(query_groups)
    random.Random(config.seed).shuffle(keys)
    dev_count = max(1, int(round(len(keys) * config.dev_fraction))) if keys else 0
    dev_keys = set(keys[:dev_count])

    train_rows = []
    dev_rows = []
    for key, group_rows in query_groups.items():
        if key in dev_keys:
            dev_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)

    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept_rows),
        "dropped_rows": len(dropped_rows),
        "kept_positive_rows": sum(1 for row in kept_rows if row.get("label") == 1),
        "kept_negative_rows": sum(1 for row in kept_rows if row.get("label") == 0),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_queries": len({group_key(row) for row in train_rows}),
        "dev_queries": len({group_key(row) for row in dev_rows}),
        "drop_reasons": dict(drop_reasons),
        "config": asdict(config),
    }

    return {
        "all_filtered": kept_rows,
        "dropped": dropped_rows,
        "train": train_rows,
        "dev": dev_rows,
        "summary": summary,
    }
