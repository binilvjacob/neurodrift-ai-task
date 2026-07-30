"""Run loaders + chunker against a real file and print the resulting chunks.

Usage:
    python scripts/verify_chunking.py path/to/file.pdf
    python scripts/verify_chunking.py path/to/file.xlsx --chunk-size 384 --chunk-overlap 64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import Chunker
from app.ingestion.loaders import extract_segments
from app.models.schemas import DocumentMetadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=Path)
    parser.add_argument("--tokenizer", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    source_type = args.file_path.suffix.lstrip(".").lower()
    metadata = DocumentMetadata(
        tenant_id="verify",
        document_id="verify-doc",
        filename=args.file_path.name,
        source_type=source_type,
    )

    segments = extract_segments(args.file_path, source_type)
    print(f"Extracted {len(segments)} segments from {args.file_path.name}\n")
    for seg in segments[:5]:
        print(f"  [order={seg.order}] {seg.locator.display()!r}: {seg.text[:80]!r}")
    if len(segments) > 5:
        print(f"  ... and {len(segments) - 5} more segments")

    chunker = Chunker(
        tokenizer_name=args.tokenizer,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
    )
    print(f"\nTokenizer max_seq_length={chunker.max_seq_length}, chunk_size={chunker.chunk_size_tokens} tokens, "
          f"overlap={chunker.chunk_overlap_tokens} tokens")

    chunks = chunker.split(segments, metadata)
    print(f"\nProduced {len(chunks)} chunks\n")
    for chunk in chunks:
        preview = chunk.text.replace("\n", " ")[:120]
        print(f"  [{chunk.chunk_index}] tokens={chunk.token_count} loc={chunk.locator.display()!r} text={preview!r}")


if __name__ == "__main__":
    main()
