"""Milestone 1 sanity test: the evidencerag package must be importable
and expose basic metadata. Component-specific tests (retrieval,
reranking, generation, evaluation) will be added alongside each
implementation, not before.
"""

import evidencerag
from evidencerag.config import PATHS, SETTINGS


def test_package_imports():
    assert evidencerag.__version__ == "0.1.0"


def test_config_paths_resolve_under_project_root():
    assert PATHS.root.is_dir()
    assert str(PATHS.data_dir).startswith(str(PATHS.root))
    assert str(PATHS.raw_data_dir).startswith(str(PATHS.root))
    assert str(PATHS.processed_data_dir).startswith(str(PATHS.root))


def test_settings_defaults():
    assert SETTINGS.project_name == "EvidenceRAG"
    assert isinstance(SETTINGS.random_seed, int)
