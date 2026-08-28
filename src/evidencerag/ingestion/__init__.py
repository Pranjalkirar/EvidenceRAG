"""QASPER dataset ingestion: acquisition, validation, normalization,
serialization, and statistics.

- schema.py     -- internal typed representation (Paper/Question/Answer/EvidenceSpan)
- loader.py     -- downloads/loads the raw dataset (via the `datasets` library)
- validate.py   -- structural validation of raw rows
- normalize.py  -- raw rows -> internal representation, incl. evidence provenance
- serialize.py  -- save/load the internal representation as JSON Lines
- statistics.py -- dataset statistics computed from the internal representation

Chunking, retrieval, reranking, and generation are NOT implemented
here or anywhere else yet — see their respective (still-empty)
subpackages.
"""
