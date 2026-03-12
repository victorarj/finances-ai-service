from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_query_rejects_blank_question():
    response = client.post(
        "/query",
        json={"question": "   ", "user_id": "5", "document_ids": ["1"]},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "question is required"


def test_query_returns_answer_and_sources(monkeypatch):
    captured = {}

    async def fake_generate_embeddings(texts, *, settings):
        assert texts == ["What changed this month?"]
        return [[0.1] * 1536]

    async def fake_retrieve_similar_chunks(session, *, user_id, question_embedding, document_ids, limit):
        captured["user_id"] = user_id
        captured["document_ids"] = document_ids
        captured["limit"] = limit
        assert len(question_embedding) == 1536
        return [
            {
                "document_id": 10,
                "chunk_index": 0,
                "content": "Income increased by 10%",
                "source_type": "payslip",
                "distance": 0.1,
            }
        ]

    async def fake_generate_answer(*, question, chunks, settings):
        assert question == "What changed this month?"
        assert chunks[0]["document_id"] == 10
        return {
            "answer": "Income increased by 10%.",
            "sources": [
                {
                    "document_id": 10,
                    "chunk_index": 0,
                    "content_preview": "Income increased by 10%",
                }
            ],
        }

    monkeypatch.setattr("app.routes.query.generate_embeddings", fake_generate_embeddings)
    monkeypatch.setattr("app.routes.query.retrieve_similar_chunks", fake_retrieve_similar_chunks)
    monkeypatch.setattr("app.routes.query.generate_answer", fake_generate_answer)

    response = client.post(
        "/query",
        json={
            "question": "What changed this month?",
            "user_id": "5",
            "document_ids": ["10", "11"],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Income increased by 10%."
    assert response.json()["sources"][0]["document_id"] == 10
    assert captured == {"user_id": 5, "document_ids": [10, 11], "limit": 5}


def test_debug_chunks_filters_by_document_and_user():
    class FakeResult:
        def __iter__(self):
            rows = [
                {
                    "id": 1,
                    "document_id": 21,
                    "user_id": 8,
                    "content": "chunk text",
                    "chunk_index": 0,
                    "created_at": "2026-03-11T00:00:00Z",
                }
            ]
            for row in rows:
                yield type("FakeRow", (), {"_mapping": row})()

    class FakeSession:
        async def execute(self, query):
            sql = str(query)
            assert "document_chunks.document_id" in sql
            return FakeResult()

    async def fake_get_db_session():
        yield FakeSession()

    app.dependency_overrides = {}
    from app.db.database import get_db_session

    app.dependency_overrides[get_db_session] = fake_get_db_session
    response = client.get("/documents/21/chunks?user_id=8")
    app.dependency_overrides = {}

    assert response.status_code == 200
    assert response.json()[0]["document_id"] == 21
    assert response.json()[0]["user_id"] == 8
