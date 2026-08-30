import pytest

from evidencerag.retrieval.bm25 import BM25Retriever
from evidencerag.retrieval.dense import DenseRetriever
from tests.retrieval_fixtures import FakeEmbedder, make_chunk

CHUNKS = [
    make_chunk(chunk_id="train:p:s000:c0000", text="the cat sat on the mat"),
    make_chunk(chunk_id="train:p:s000:c0001", text="dogs are loyal animals"),
    make_chunk(chunk_id="train:p:s001:c0000", text="cats and dogs can be friends"),
]


def test_bm25_build_save_load_produces_equivalent_retrieval(tmp_path):
    original = BM25Retriever.build(CHUNKS)
    path = tmp_path / "bm25.pkl"
    original.save(path)

    reloaded = BM25Retriever.load(path)

    query = "cats dogs"
    assert reloaded.retrieve(query, top_k=3) == original.retrieve(query, top_k=3)
    assert reloaded.fingerprint == original.fingerprint
    assert reloaded.config == original.config


def test_dense_build_save_load_produces_equivalent_retrieval(tmp_path):
    embedder = FakeEmbedder()
    original = DenseRetriever.build(CHUNKS, embedder=embedder)
    path = tmp_path / "dense_index"
    original.save(path)

    reloaded = DenseRetriever.load(path, embedder=embedder)

    query = "cats dogs"
    assert reloaded.retrieve(query, top_k=3) == original.retrieve(query, top_k=3)
    assert reloaded.metadata == original.metadata


def test_dense_load_rejects_mismatched_embedder_model_name(tmp_path):
    embedder = FakeEmbedder(model_name="fake-a")
    retriever = DenseRetriever.build(CHUNKS, embedder=embedder)
    path = tmp_path / "dense_index"
    retriever.save(path)

    wrong_embedder = FakeEmbedder(model_name="fake-b")
    with pytest.raises(ValueError):
        DenseRetriever.load(path, embedder=wrong_embedder)


def test_dense_load_rejects_mismatched_dimension(tmp_path):
    embedder = FakeEmbedder(dimension=16, model_name="fake")
    retriever = DenseRetriever.build(CHUNKS, embedder=embedder)
    path = tmp_path / "dense_index"
    retriever.save(path)

    wrong_dim_embedder = FakeEmbedder(dimension=8, model_name="fake")
    with pytest.raises(ValueError):
        DenseRetriever.load(path, embedder=wrong_dim_embedder)


def test_dense_load_without_embedder_succeeds_but_retrieve_requires_one(tmp_path):
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    path = tmp_path / "dense_index"
    retriever.save(path)

    reloaded = DenseRetriever.load(path)  # no embedder -- e.g. inspecting metadata only
    assert reloaded.metadata.num_chunks == len(CHUNKS)
    with pytest.raises(RuntimeError):
        reloaded.retrieve("cats", top_k=1)


# --- recommended chunks-validated loading path (Fix 2) -------------------


def test_dense_load_with_matching_chunks_succeeds(tmp_path):
    embedder = FakeEmbedder()
    retriever = DenseRetriever.build(CHUNKS, embedder=embedder)
    path = tmp_path / "dense_index"
    retriever.save(path)

    reloaded = DenseRetriever.load(path, embedder=embedder, chunks=CHUNKS)  # must not raise
    assert reloaded.retrieve("cats", top_k=1) == retriever.retrieve("cats", top_k=1)


def test_dense_load_with_mismatched_chunks_raises(tmp_path):
    embedder = FakeEmbedder()
    retriever = DenseRetriever.build(CHUNKS, embedder=embedder)
    path = tmp_path / "dense_index"
    retriever.save(path)

    different_chunks = [make_chunk(chunk_id="train:p:s000:c0000", text="a completely different corpus")]
    with pytest.raises(ValueError):
        DenseRetriever.load(path, embedder=embedder, chunks=different_chunks)


def test_dense_load_without_chunks_does_not_validate_corpus(tmp_path):
    # Documents the trade-off: skipping `chunks` at load time means no
    # corpus check happens until/unless verify_corpus() is called
    # explicitly -- this must NOT raise, unlike the mismatched-chunks case above.
    embedder = FakeEmbedder()
    retriever = DenseRetriever.build(CHUNKS, embedder=embedder)
    path = tmp_path / "dense_index"
    retriever.save(path)

    DenseRetriever.load(path, embedder=embedder)  # no `chunks` -- must not raise


def test_bm25_verify_corpus_detects_mismatch():
    retriever = BM25Retriever.build(CHUNKS)
    different_chunks = [make_chunk(chunk_id="train:p:s000:c0000", text="a completely different corpus")]
    with pytest.raises(ValueError):
        retriever.verify_corpus(different_chunks)
    retriever.verify_corpus(CHUNKS)  # must not raise


def test_dense_verify_corpus_detects_mismatch():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    different_chunks = [make_chunk(chunk_id="train:p:s000:c0000", text="a completely different corpus")]
    with pytest.raises(ValueError):
        retriever.verify_corpus(different_chunks)
    retriever.verify_corpus(CHUNKS)  # must not raise
