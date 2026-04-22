"""Tests for recursive chunking."""
from rag.ingest.chunking import recursive_chunk


def test_short_text_becomes_single_chunk():
    text = "This is a short sentence."
    chunks = recursive_chunk(text, chunk_size=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_is_split_at_paragraph_boundary():
    text = "First paragraph about dogs.\n\nSecond paragraph about cats.\n\nThird paragraph about birds."
    chunks = recursive_chunk(text, chunk_size=60, overlap=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 120


def test_overlap_produces_shared_tokens_between_consecutive_chunks():
    sentences = ". ".join([f"Sentence number {i}" for i in range(40)]) + "."
    chunks = recursive_chunk(sentences, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    assert any(
        any(word in chunks[i + 1] for word in chunks[i].split()[-5:])
        for i in range(len(chunks) - 1)
    )


def test_empty_text_returns_empty_list():
    assert recursive_chunk("", chunk_size=512, overlap=0) == []


def test_whitespace_only_text_returns_empty_list():
    assert recursive_chunk("   \n  \n  ", chunk_size=512, overlap=0) == []


def test_dense_text_falls_through_to_sentence_level():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. " * 20
    chunks = recursive_chunk(text, chunk_size=80, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 100
