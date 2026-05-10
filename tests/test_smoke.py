import tempfile
import unittest
from pathlib import Path
from random import Random

from acl_verbatim.core.jsonl import load_jsonl, write_jsonl
from acl_verbatim.data.spans import Span, SpanRow
from acl_verbatim.eval.span_metrics import evaluate_rows_against_predictions
from acl_verbatim.synthetic.filtering import SilverFilterConfig, filter_and_split_rows
from scripts.experiments.qasper_to_gold_file import convert_paper_paragraph


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

    def test_negative_false_positive_penalizes_word_precision(self):
        positive = SpanRow(
            query="what is modernbert",
            paper_id="2025.acl-long.5",
            chunk_index=2,
            chunk="ModernBERT is a long-context encoder for NLP.",
            relevance_label="r",
            is_relevant=True,
            gold_spans=[Span(start=0, end=10, text="ModernBERT")],
        )
        negative = SpanRow(
            query="what is modernbert",
            paper_id="2025.acl-long.5",
            chunk_index=3,
            chunk="This paragraph discusses unrelated references.",
            relevance_label="n",
            is_relevant=False,
            gold_spans=[],
        )
        pred_map = {
            ("what is modernbert", "2025.acl-long.5", 2): {
                "query": "what is modernbert",
                "paper_id": "2025.acl-long.5",
                "chunk_index": 2,
                "pred_spans": [{"start": 0, "end": 10, "text": "ModernBERT"}],
            },
            ("what is modernbert", "2025.acl-long.5", 3): {
                "query": "what is modernbert",
                "paper_id": "2025.acl-long.5",
                "chunk_index": 3,
                "pred_spans": [{"start": 0, "end": 14, "text": "This paragraph"}],
            },
        }
        summary = evaluate_rows_against_predictions([positive, negative], pred_map)[
            "summary"
        ]
        self.assertEqual(summary["n_examples"], 2)
        self.assertEqual(summary["n_relevant"], 1)
        self.assertEqual(summary["n_irrelevant"], 1)
        self.assertLess(summary["word_level"]["micro_precision"], 1.0)
        self.assertAlmostEqual(summary["word_level"]["micro_recall"], 1.0)


class QasperConverterSmokeTest(unittest.TestCase):
    def test_qasper_converter_aligns_paragraph_and_float_evidence(self):
        rows = convert_paper_paragraph(
            {
                "id": "paper-1",
                "title": "Paper title",
                "abstract": "Paper abstract.",
                "full_text": {
                    "section_name": ["Methods"],
                    "paragraphs": [["The model uses a token classifier."]],
                },
                "figures_and_tables": {"caption": ["Table 1: Results."]},
                "qas": {
                    "question": ["What model is used?"],
                    "question_id": ["q1"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "evidence": [
                                        "The model uses a token classifier.",
                                        "FLOAT SELECTED: Table 1: Results.",
                                    ],
                                    "highlighted_evidence": [],
                                }
                            ]
                        }
                    ],
                },
            },
            evidence_field="evidence",
            include_unanswerable=False,
            negative_ratio=0,
            rng=Random(1337),
        )
        self.assertEqual(len(rows), 2)
        chunks = [row["results"][0]["chunk"] for row in rows]
        self.assertTrue(
            all(row["results"][0]["relevance_label"] == "r" for row in rows)
        )
        self.assertTrue(
            any("The model uses a token classifier." in chunk for chunk in chunks)
        )
        self.assertIn("FLOAT SELECTED: Table 1: Results.", chunks)


if __name__ == "__main__":
    unittest.main()
