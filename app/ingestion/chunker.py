import tiktoken


DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50


def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict[str, int | str]]:
    cleaned = text.strip()
    if not cleaned:
        return []

    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    encoding = _encoding()
    token_ids = encoding.encode(cleaned)
    if not token_ids:
        return []

    chunks: list[dict[str, int | str]] = []
    step = chunk_tokens - overlap_tokens

    for chunk_index, start in enumerate(range(0, len(token_ids), step)):
        window = token_ids[start : start + chunk_tokens]
        if not window:
            continue
        content = encoding.decode(window).strip()
        if not content:
            continue
        chunks.append({"chunk_index": chunk_index, "content": content})

    return chunks
