"""Central project configuration: filesystem paths and settings.

This module is the single source of truth for where things live on
disk. Pipeline code should import from here rather than hardcoding
relative paths like "../data/raw" — that breaks the moment a script
is run from a different working directory (e.g. a notebook, a Kaggle
kernel, or a Docker container).

Paths can be overridden via environment variables (optionally set in
a local .env file, loaded automatically) so the same code can run
unmodified on the local dev machine and on a remote GPU environment
such as Kaggle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file at the project root, if present.
# Safe no-op if the file doesn't exist.
load_dotenv()

# src/evidencerag/config.py -> src/evidencerag -> src -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_path(var_name: str, default: Path) -> Path:
    """Resolve a path from an environment variable, falling back to `default`."""
    override = os.environ.get(var_name)
    return Path(override).expanduser().resolve() if override else default


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout for EvidenceRAG.

    All paths resolve relative to PROJECT_ROOT by default, and can be
    overridden individually via environment variables (useful when
    running in Docker or on a remote GPU host).
    """

    root: Path = PROJECT_ROOT

    data_dir: Path = field(
        default_factory=lambda: _env_path("EVIDENCERAG_DATA_DIR", PROJECT_ROOT / "data")
    )
    raw_data_dir: Path = field(
        default_factory=lambda: _env_path(
            "EVIDENCERAG_RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"
        )
    )
    processed_data_dir: Path = field(
        default_factory=lambda: _env_path(
            "EVIDENCERAG_PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed"
        )
    )
    configs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "configs")
    experiments_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "experiments")
    notebooks_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "notebooks")
    scripts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "scripts")
    # Milestone 4: persisted retrieval indexes (BM25 pickle, FAISS index
    # + metadata) live here, separate from data/processed so retrieval
    # artifacts can be wiped/rebuilt without touching M2/M3 output.
    artifacts_dir: Path = field(
        default_factory=lambda: _env_path("EVIDENCERAG_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
    )

    def ensure_data_dirs(self) -> None:
        """Create data directories if missing. Does not touch source dirs.

        Intentionally NOT called at import time — nothing should touch
        the filesystem just because this module was imported.
        """
        for path in (self.data_dir, self.raw_data_dir, self.processed_data_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Project-wide, non-secret settings.

    Grows incrementally as each milestone needs new settings (or
    configs/*.yaml, loaded by this module) -- not before.
    """

    project_name: str = "EvidenceRAG"
    random_seed: int = 42

    # Milestone 4: retrieval baseline settings — sensible defaults, not
    # tuned hyperparameters (M4 establishes the baseline).
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    retrieval_top_k: int = 5
    retrieval_candidate_depth: int = 20  # per-retriever candidates fed into RRF
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


PATHS = ProjectPaths()
SETTINGS = Settings()
