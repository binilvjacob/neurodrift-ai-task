from transformers import AutoTokenizer

from app.models.schemas import Chunk, DocumentMetadata, Locator, TextSegment

SEGMENT_JOINER = "\n\n"


class Chunker:
    """Splits document segments into token-bounded chunks using the embedding model's own tokenizer.

    Chunk size is measured in tokens (not characters) and validated at construction time against
    the embedding model's max sequence length, so an oversized chunk_size fails fast at startup
    instead of being silently truncated later at embedding time.
    """

    # tokenizers with no declared limit report this sentinel instead of a real max length
    _UNBOUNDED_MODEL_MAX_LENGTH = 1_000_000

    def __init__(self, tokenizer_name: str, chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({chunk_overlap_tokens}) must be smaller than "
                f"chunk_size_tokens ({chunk_size_tokens})"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not self._tokenizer.is_fast:
            raise ValueError(f"tokenizer for {tokenizer_name!r} must be a fast tokenizer (offset mapping required)")

        max_seq_length = self._tokenizer.model_max_length
        if max_seq_length >= self._UNBOUNDED_MODEL_MAX_LENGTH:
            raise ValueError(
                f"tokenizer for {tokenizer_name!r} does not declare a max sequence length; "
                f"cannot validate chunk_size_tokens against it"
            )
        if chunk_size_tokens > max_seq_length:
            raise ValueError(
                f"chunk_size_tokens ({chunk_size_tokens}) exceeds {tokenizer_name}'s max "
                f"sequence length ({max_seq_length}); lower chunk_size_tokens or chunks will be "
                f"silently truncated when embedded"
            )
        self.max_seq_length = max_seq_length
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def split(self, segments: list[TextSegment], metadata: DocumentMetadata) -> list[Chunk]:
        ordered = sorted(segments, key=lambda s: s.order)

        text_parts: list[str] = []
        span_locators: list[tuple[int, int, Locator]] = []
        cursor = 0
        for segment in ordered:
            start = cursor
            end = start + len(segment.text)
            span_locators.append((start, end, segment.locator))
            text_parts.append(segment.text)
            cursor = end + len(SEGMENT_JOINER)
        full_text = SEGMENT_JOINER.join(text_parts)

        encoding = self._tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets: list[tuple[int, int]] = encoding["offset_mapping"]
        token_count = len(offsets)

        stride = self.chunk_size_tokens - self.chunk_overlap_tokens
        chunks: list[Chunk] = []
        chunk_index = 0
        window_start = 0
        while window_start < token_count:
            window_end = min(window_start + self.chunk_size_tokens, token_count)
            char_start = offsets[window_start][0]
            char_end = offsets[window_end - 1][1]
            chunk_text = full_text[char_start:char_end].strip()

            if chunk_text:
                contributing = [
                    locator for seg_start, seg_end, locator in span_locators if seg_start < char_end and seg_end > char_start
                ]
                merged_locator = Locator.merge(contributing)
                chunks.append(
                    Chunk(
                        chunk_id=f"{metadata.document_id}:{chunk_index}",
                        tenant_id=metadata.tenant_id,
                        document_id=metadata.document_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        token_count=window_end - window_start,
                        locator=merged_locator,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

            if window_end >= token_count:
                break
            window_start += stride

        return chunks
