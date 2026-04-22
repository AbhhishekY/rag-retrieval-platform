"""Recursive character-level chunker.

Tries strongest separator first (paragraph), falls back to weaker ones
(line, sentence, word, raw char) when chunks still exceed the size budget.
Character-based for determinism and zero deps. For English, tokens are ~4
chars on average; multiply chunk_size by 4 if reasoning about tokens.
"""
from __future__ import annotations

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 0,
    separators: list[str] | None = None,
) -> list[str]:
    if not text or not text.strip():
        return []
    seps = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _split(text, chunk_size, seps)
    if overlap > 0 and len(pieces) > 1:
        pieces = _apply_overlap(pieces, overlap)
    return pieces


def _split(text: str, chunk_size: int, seps: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    for i, sep in enumerate(seps):
        if sep == "":
            return [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]
        if sep in text:
            parts = text.split(sep)
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]
            parts = [p for p in parts if p]
            merged = _merge_small_pieces(parts, chunk_size)
            out: list[str] = []
            for p in merged:
                if len(p) <= chunk_size:
                    out.append(p)
                else:
                    out.extend(_split(p, chunk_size, seps[i + 1 :]))
            return out
    return [text]


def _merge_small_pieces(parts: list[str], chunk_size: int) -> list[str]:
    merged: list[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
            continue
        if len(buf) + len(p) <= chunk_size:
            buf += p
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    out = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_tail = pieces[i - 1][-overlap:]
        out.append(prev_tail + pieces[i])
    return out
