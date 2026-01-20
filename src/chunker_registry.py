from __future__ import annotations
from typing import Callable, Dict, List
from src.chunkers import Chunk, chunk_fixed_chars, chunk_by_layout_breaks

ChunkerFn = Callable[[str], List[Chunk]]

def make_fixed(chunk_size: int = 1000, overlap: int = 200) -> ChunkerFn:
    return lambda text: chunk_fixed_chars(text, chunk_size=chunk_size, overlap=overlap)

def make_layout(max_chars: int = 1200) -> ChunkerFn:
    return lambda text: chunk_by_layout_breaks(text, max_chars=max_chars)

CHUNKERS: Dict[str, ChunkerFn] = {
    "fixed": make_fixed(),
    "layout": make_layout(),
}
