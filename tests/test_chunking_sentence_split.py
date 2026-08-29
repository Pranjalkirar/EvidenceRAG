from evidencerag.chunking.sentence_split import split_into_sentences


def test_splits_simple_sentences():
    text = "This is one sentence. This is another sentence. And a third."
    sentences = split_into_sentences(text)
    assert sentences == [
        "This is one sentence.",
        "This is another sentence.",
        "And a third.",
    ]


def test_empty_text_returns_no_sentences():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []


def test_single_sentence_no_terminal_punctuation():
    text = "a run on paragraph with no punctuation at all"
    assert split_into_sentences(text) == [text]


def test_never_drops_text():
    text = "First sentence. Second sentence! Third sentence? Fourth."
    sentences = split_into_sentences(text)
    # Reconstructing (whitespace-normalized) shouldn't lose any words.
    original_words = text.split()
    reconstructed_words = " ".join(sentences).split()
    assert original_words == reconstructed_words
