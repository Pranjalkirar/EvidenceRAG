"""Tests for evidencerag.ingestion.loader.

Deliberately does NOT call load_raw_qasper() itself -- that downloads
the full QASPER dataset over the network, which the ingestion tests
must not require (see Milestone 2 constraints). Instead we test the
parts of this module that don't need a network call: its constants
and its cache-directory logic.
"""

from evidencerag.config import PATHS
from evidencerag.ingestion.loader import (
    QASPER_HF_DATASET_ID,
    QASPER_SPLITS,
    default_raw_cache_dir,
)


def test_qasper_dataset_id_is_the_hub_id():
    assert QASPER_HF_DATASET_ID == "allenai/qasper"


def test_qasper_splits_are_train_validation_test():
    assert QASPER_SPLITS == ("train", "validation", "test")


def test_default_raw_cache_dir_lives_under_data_raw():
    cache_dir = default_raw_cache_dir()
    assert str(cache_dir).startswith(str(PATHS.raw_data_dir))
