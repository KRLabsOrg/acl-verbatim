import tempfile
import unittest
from pathlib import Path

from acl_verbatim.core.jsonl import load_jsonl, write_jsonl
from acl_verbatim.data.spans import Span, SpanRow
from acl_verbatim.eval.span_metrics import evaluate_rows_against_predictions
from acl_verbatim.synthetic.filtering import SilverFilterConfig, filter_and_split_rows


class JsonlSmokeTest(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        rows = [
            {"question": "q1", "paper_id": "P1", "chunk_index": 0},
            {"question": "q2", "paper_id": "P2", "chunk_index": 1},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            write_jsonl(path, rows)
            loaded = load_jsonl(path)
        self.assertEqual(rows, loaded)


class SilverFilterSmokeTest(unittest.TestCase):
    def _caption_row(self):
        chunk = (
            "Table 1: Accuracy results on the shared task benchmark with full ablations "
            "and detailed per-language breakdown."
        )
        return {
            "question": "shared task accuracy table",
            "paper_id": "2024.acl-long.1",
            "chunk_index": 3,
            "chunk": chunk,
            "label": 1,
            "answerable": True,
            "spans": [{"start": 0, "end": len(chunk), "text": chunk}],
            "source": "retrieved",
            "retrieval_rank": 2,
            "gold_paper": "2024.acl-long.1",
            "gold_chunk": 3,
        }

    def test_caption_like_positive_is_kept_by_default(self):
        result = filter_and_split_rows(
            [self._caption_row()],
            SilverFilterConfig(dev_fraction=0.0),
        )
        self.assertEqual(result["summary"]["kept_rows"], 1)
        self.assertEqual(result["summary"]["dropped_rows"], 0)

    def test_caption_like_positive_can_be_dropped_in_strict_mode(self):
        result = filter_and_split_rows(
            [self._caption_row()],
            SilverFilterConfig(dev_fraction=0.0, drop_caption_like=True),
        )
        self.assertEqual(result["summary"]["kept_rows"], 0)
        self.assertEqual(result["summary"]["drop_reasons"].get("caption_like"), 1)


class SpanMetricsSmokeTest(unittest.TestCase):
    def test_exact_prediction_scores_perfectly(self):
        row = SpanRow(
            query="what is modernbert",
            paper_id="2025.acl-long.5",
            chunk_index=2,
            chunk="ModernBERT is a long-context encoder for NLP.",
            relevance_label="r",
            is_relevant=True,
            gold_spans=[Span(start=0, end=10, text="ModernBERT")],
        )
        pred_map = {
            ("what is modernbert", "2025.acl-long.5", 2): {
                "question": "what is modernbert",
                "paper_id": "2025.acl-long.5",
                "chunk_index": 2,
                "pred_spans": [{"start": 0, "end": 10, "text": "ModernBERT"}],
            }
        }
        summary = evaluate_rows_against_predictions([row], pred_map)["summary"]
        self.assertEqual(summary["n_examples"], 1)
        self.assertAlmostEqual(summary["word_level"]["micro_f1"], 1.0)
        self.assertAlmostEqual(summary["span_level_iou"]["0.5"]["micro_f1"], 1.0)
        self.assertAlmostEqual(summary["containment"]["1.0"]["micro_f1"], 1.0)
        self.assertAlmostEqual(summary["gold_coverage_recall"]["1.0"], 1.0)
        self.assertAlmostEqual(summary["recall_any_overlap"], 1.0)
        self.assertAlmostEqual(summary["over_prediction_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
