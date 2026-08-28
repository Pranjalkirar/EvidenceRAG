"""Small, hand-built fixture data shaped exactly like real QASPER rows
(see evidencerag.ingestion.loader's module docstring for the schema
source). Used by ingestion tests so they never need to download the
real dataset.

Deliberately covers several edge cases in one small fixture:
- multiple answers per question (multiple workers)
- an unanswerable answer
- a yes/no answer whose evidence is a figure/table ("FLOAT SELECTED")
- multiple evidence paragraphs on one answer
- an answer with an extractive span
- evidence text that does NOT exactly match any paragraph (tests the
  "unresolved" provenance path)
- a paper with no `figures_and_tables` key at all (optional field)
- a paper with an empty (un-named) section with zero paragraphs
- missing optional `qas` keys (tests default-handling, not just
  present-but-empty)
"""

from __future__ import annotations

PAPER_1_TRAIN = {
    "id": "1000.00001",
    "title": "Propagating Affective Polarity via Discourse Relations",
    "abstract": (
        "We study how to propagate affective polarity of events using "
        "discourse relations between event pairs, achieving strong "
        "results with minimal supervision."
    ),
    "full_text": {
        "section_name": ["Introduction", "Method"],
        "paragraphs": [
            [
                "This paper studies event polarity propagation.",
                "We use discourse relations to propagate affective "
                "polarity across event pairs.",
            ],
            [
                "Our method builds a graph of events connected by "
                "discourse relations and propagates seed polarity scores."
            ],
        ],
    },
    "figures_and_tables": {
        "caption": ["Table 1: Example results on the test set."],
        "file": ["1-Table1-1.png"],
    },
    "qas": {
        "question": [
            "What method is used to propagate polarity?",
            "Is the evaluation conducted only on English data?",
        ],
        "question_id": ["Q1000.00001.1", "Q1000.00001.2"],
        "question_writer": ["writer_a", "writer_b"],
        "nlp_background": ["five", "two"],
        "topic_background": ["research", "familiar"],
        "paper_read": ["yes", "no"],
        "search_query": ["", "english data evaluation"],
        "answers": [
            {
                "annotation_id": ["ann_q1_a1"],
                "worker_id": ["worker_1"],
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": [],
                        "yes_no": None,
                        "free_form_answer": "A discourse-relation based graph propagation method.",
                        "evidence": [
                            "Our method builds a graph of events connected by "
                            "discourse relations and propagates seed polarity scores."
                        ],
                        "highlighted_evidence": ["propagates seed polarity scores"],
                    }
                ],
            },
            {
                "annotation_id": ["ann_q2_a1", "ann_q2_a2"],
                "worker_id": ["worker_1", "worker_2"],
                "answer": [
                    {
                        "unanswerable": True,
                        "extractive_spans": [],
                        "yes_no": None,
                        "free_form_answer": "",
                        "evidence": [],
                        "highlighted_evidence": [],
                    },
                    {
                        "unanswerable": False,
                        "extractive_spans": [],
                        "yes_no": True,
                        "free_form_answer": "",
                        "evidence": ["FLOAT SELECTED: Table 1: Example results on the test set."],
                        "highlighted_evidence": [],
                    },
                ],
            },
        ],
    },
}

# Deliberately missing "figures_and_tables" and several optional qas
# keys, and contains one un-named/empty section, and one evidence
# string that will NOT match any paragraph verbatim.
PAPER_2_VALIDATION = {
    "id": "2000.00002",
    "title": "A Small Ablation Study",
    "abstract": "We ablate the attention module of our model and measure the effect on F1.",
    "full_text": {
        "section_name": ["", "Experiments"],
        "paragraphs": [
            [],
            [
                "We run an ablation removing the attention module and "
                "observe a 3 point drop in F1."
            ],
        ],
    },
    "qas": {
        "question": ["What happens when attention is removed?"],
        "question_id": ["Q2000.00002.1"],
        "answers": [
            {
                "annotation_id": ["ann_x1"],
                "worker_id": ["worker_3"],
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": ["3 point drop in F1"],
                        "yes_no": None,
                        "free_form_answer": "",
                        # Not an exact match to the paragraph above (paraphrased) --
                        # exercises the "resolved=False" provenance path.
                        "evidence": [
                            "We observe a 3 point drop in F1 when attention is removed."
                        ],
                        "highlighted_evidence": ["3 point drop"],
                    }
                ],
            }
        ],
    },
}


def raw_dataset_dict() -> dict[str, list[dict]]:
    """A tiny stand-in for the `datasets.DatasetDict` returned by
    `loader.load_raw_qasper`, without needing any download."""
    return {
        "train": [PAPER_1_TRAIN],
        "validation": [PAPER_2_VALIDATION],
        "test": [],
    }
