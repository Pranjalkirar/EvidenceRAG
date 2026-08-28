"""Acquisition/loading of the raw QASPER dataset.

QASPER is distributed on the Hugging Face Hub as `allenai/qasper`
(https://huggingface.co/datasets/allenai/qasper), with three splits:
`train`, `validation`, and `test` (the original QASPER release/paper
referred to the middle split as "dev"; the Hub dataset calls it
"validation" — same data, different name. We use the Hub's names
since that is our actual source).

We use the `datasets` library rather than hand-rolling an HTTP
download, because:
  - it is the canonical, actively maintained distribution channel for
    this dataset (the dataset card itself is hosted there),
  - it transparently handles the three splits and their schemas,
  - it caches downloads locally so re-running ingestion doesn't
    re-download.

This module ONLY loads the dataset into memory / local cache. It does
not validate or transform anything — see validate.py and normalize.py.

The Hugging Face cache is redirected under data/raw/ (via the
`HF_DATASETS_CACHE` env var / `cache_dir` argument) so the raw
download lives under our own data/raw/ directory rather than the
user's global `~/.cache`, per the project convention of not scattering
data outside data/raw and data/processed. Nothing in this module
modifies files after they are downloaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from evidencerag.config import PATHS

QASPER_HF_DATASET_ID = "allenai/qasper"
QASPER_SPLITS = ("train", "validation", "test")


def default_raw_cache_dir() -> Path:
    """Where the raw Hugging Face download cache lives, under data/raw/."""
    return PATHS.raw_data_dir / "qasper_hf_cache"


def load_raw_qasper(cache_dir: Optional[Path] = None):
    """Download (if needed) and load the raw QASPER dataset.

    Returns a `datasets.DatasetDict` with keys "train", "validation",
    "test", each a `datasets.Dataset` of raw QASPER rows exactly as
    published — no normalization is applied here.

    Requires the `datasets` package (see requirements.txt). Imported
    lazily inside this function so that importing `evidencerag.ingestion`
    does not require `datasets` to be installed unless this function is
    actually called.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only when dep missing
        raise ImportError(
            "The 'datasets' package is required to download QASPER. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    resolved_cache_dir = cache_dir or default_raw_cache_dir()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    return load_dataset(QASPER_HF_DATASET_ID, cache_dir=str(resolved_cache_dir))
