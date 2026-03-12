import httpx
from openai import AsyncOpenAI

from app.config import Settings


async def generate_embeddings(texts: list[str], *, settings: Settings) -> list[list[float]]:
    if not texts:
        return []

    if settings.validated_llm_provider == "openai":
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(model=settings.embedding_model, input=texts)
        return [list(item.embedding) for item in response.data]

    async with httpx.AsyncClient(base_url=settings.normalized_ollama_base_url, timeout=60.0) as client:
        embeddings: list[list[float]] = []
        for text in texts:
            response = await client.post(
                "/api/embeddings",
                json={"model": settings.embedding_model, "prompt": text},
            )
            response.raise_for_status()
            vector = response.json().get("embedding", [])
            if len(vector) != 1536:
                raise ValueError("Ollama embedding model must return 1536 dimensions")
            embeddings.append(vector)
        return embeddings


def build_prompt(question: str, chunks: list[dict[str, int | str | float]]) -> str:
    context_blocks = []
    for chunk in chunks:
        context_blocks.append(
            "\n".join(
                [
                    f"document_id: {chunk['document_id']}",
                    f"chunk_index: {chunk['chunk_index']}",
                    f"source_type: {chunk.get('source_type', 'unknown')}",
                    f"content: {chunk['content']}",
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No supporting document chunks found."
    return (
        "You are the AI brain for a personal finance application.\n"
        "Answer the user's question using only the provided context.\n"
        "If the context is insufficient, say so clearly.\n"
        "Cite sources using document_id and chunk_index when relevant.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )


async def generate_answer(
    *,
    question: str,
    chunks: list[dict[str, int | str | float]],
    settings: Settings,
) -> dict[str, object]:
    prompt = build_prompt(question, chunks)

    if settings.validated_llm_provider == "openai":
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": "You answer personal finance questions using retrieved document context."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content or ""
    else:
        async with httpx.AsyncClient(base_url=settings.normalized_ollama_base_url, timeout=120.0) as client:
            response = await client.post(
                "/api/chat",
                json={
                    "model": settings.chat_model,
                    "messages": [
                        {"role": "system", "content": "You answer personal finance questions using retrieved document context."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            answer = payload.get("message", {}).get("content", "")

    sources = [
        {
            "document_id": int(chunk["document_id"]),
            "chunk_index": int(chunk["chunk_index"]),
            "content_preview": str(chunk["content"])[:200],
        }
        for chunk in chunks
    ]

    return {"answer": answer.strip(), "sources": sources}
