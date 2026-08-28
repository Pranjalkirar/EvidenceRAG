#!/usr/bin/env python
"""Run the full QASPER ingestion pipeline:

    download/load raw QASPER (data/raw/)
        -> validate
        -> normalize into our internal representation
        -> save to data/processed/{split}.jsonl
        -> print dataset statistics

Usage:
    python scripts/ingest_qasper.py
    python scripts/ingest_qasper.py --splits train validation
    python scripts/ingest_qasper.py --max-papers-per-split 5   # quick smoke test

This script never modifies files under data/raw/ — it only reads the
cached raw download from there and writes normalized output under
data/processed/.
"""

from __future__ import annotations

import argparse
import sys

from evidencerag.config import PATHS
from evidencerag.ingestion.loader import QASPER_SPLITS, load_raw_qasper
from evidencerag.ingestion.normalize import normalize_split
from evidencerag.ingestion.serialize import save_papers
from evidencerag.ingestion.statistics import compute_statistics
from evidencerag.ingestion.validate import validate_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(QASPER_SPLITS),
        choices=list(QASPER_SPLITS),
        help="Which QASPER splits to process (default: all three).",
    )
    parser.add_argument(
        "--max-papers-per-split",
        type=int,
        default=None,
        help="If set, only process this many papers per split (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--fail-on-validation-issues",
        action="store_true",
        help="Abort ingestion if any structural validation issues are found (default: warn and continue).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading raw QASPER (splits: {', '.join(args.splits)}) ...")
    raw_dataset = load_raw_qasper()

    all_papers = []
    for split in args.splits:
        rows = list(raw_dataset[split])
        if args.max_papers_per_split is not None:
            rows = rows[: args.max_papers_per_split]

        print(f"\n[{split}] {len(rows)} raw papers loaded. Validating ...")
        issues = validate_split(rows, split_name=split)
        if issues:
            print(f"[{split}] {len(issues)} validation issue(s) found:")
            for issue in issues[:20]:
                print(f"  - {issue}")
            if len(issues) > 20:
                print(f"  ... and {len(issues) - 20} more")
            if args.fail_on_validation_issues:
                print("Aborting due to --fail-on-validation-issues.")
                return 1
        else:
            print(f"[{split}] no validation issues found.")

        print(f"[{split}] normalizing ...")
        papers = normalize_split(rows, split=split)

        output_path = PATHS.processed_data_dir / f"{split}.jsonl"
        n_written = save_papers(papers, output_path)
        print(f"[{split}] wrote {n_written} papers to {output_path}")

        all_papers.extend(papers)

    print("\n" + compute_statistics(all_papers).format_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
