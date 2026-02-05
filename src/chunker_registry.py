from __future__ import annotations
from typing import Callable, Dict, List

from src.chunkers import Chunk, chunk_fixed_chars, chunk_by_layout_breaks
from src.chunkers_recursive import chunk_recursive, split_sentences_rule
from src.chunkers_semantic import chunk_semantic_adjacent

ChunkerFn = Callable[[str], List[Chunk]]

# 1) Size-based baseline
def make_fixed(chunk_size: int = 1000, overlap: int = 200) -> ChunkerFn:
    return lambda text: chunk_fixed_chars(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

# 2) Layout-based
def make_layout(max_chars: int = 1200) -> ChunkerFn:
    return lambda text: chunk_by_layout_breaks(
        text,
        max_chars=max_chars,
    )

# 3) Recursive rule-based
def make_recursive_rule(max_chars: int = 350) -> ChunkerFn:
    return lambda text: chunk_recursive(
        text,
        max_chars=max_chars,
        sentence_splitter=split_sentences_rule,
    )

# 4) Semantic adjacent
def make_semantic_adjacent(
    max_chars: int = 350,
    min_chars: int = 200,
    similarity_threshold: float = 0.65,
) -> ChunkerFn:
    return lambda text: chunk_semantic_adjacent(
        text,
        max_chars=max_chars,
        min_chars=min_chars,
        similarity_threshold=similarity_threshold,
    )

CHUNKERS: Dict[str, ChunkerFn] = {
    "fixed": make_fixed(),
    "layout": make_layout(),
    "recursive_rule": make_recursive_rule(),
    "semantic_adjacent": make_semantic_adjacent(),
}
