from fastapi.testclient import TestClient

from app.main import app
from app.ingestion.extractor import _clean_extracted_text


client = TestClient(app)


def test_health_returns_ok(monkeypatch):
    async def fake_check_database_health():
        return None

    monkeypatch.setattr("app.main.check_database_health", fake_check_database_health)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_successful_pipeline(monkeypatch):
    callback_calls = []

    async def fake_update_document_status(document_id, status, processed_at=None):
        callback_calls.append({"document_id": document_id, "status": status, "processed_at": processed_at})

    class FakeObjectStore:
        async def download(self, storage_key):
            assert storage_key == "documents/7/sample.pdf"
            return b"%PDF"

    async def fake_extract_text(file_bytes, storage_key):
        assert file_bytes == b"%PDF"
        assert storage_key == "documents/7/sample.pdf"
        return "salary data " * 200

    def fake_chunk_text(text):
        assert text.startswith("salary data")
        return [
            {"chunk_index": 0, "content": "chunk-0"},
            {"chunk_index": 1, "content": "chunk-1"},
        ]

    async def fake_embed_and_store_chunks(session, *, document_id, user_id, chunks, settings):
        assert document_id == 11
        assert user_id == 7
        assert len(chunks) == 2
        return chunks

    monkeypatch.setattr("app.routes.ingest.update_document_status", fake_update_document_status)
    monkeypatch.setattr("app.routes.ingest.ObjectStore", FakeObjectStore)
    monkeypatch.setattr("app.routes.ingest.extract_text", fake_extract_text)
    monkeypatch.setattr("app.routes.ingest.chunk_text", fake_chunk_text)
    monkeypatch.setattr("app.routes.ingest.embed_and_store_chunks", fake_embed_and_store_chunks)

    response = client.post(
        "/ingest",
        json={
            "document_id": "11",
            "user_id": "7",
            "storage_key": "documents/7/sample.pdf",
            "source_type": "payslip",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["chunk_count"] == 2
    assert [call["status"] for call in callback_calls] == ["processing", "ready"]


def test_ingest_failure_marks_document_failed(monkeypatch):
    callback_calls = []

    async def fake_update_document_status(document_id, status, processed_at=None):
        callback_calls.append(status)

    class FakeObjectStore:
        async def download(self, storage_key):
            return b"bad file"

    async def fake_extract_text(file_bytes, storage_key):
        raise ValueError("ocr failed")

    monkeypatch.setattr("app.routes.ingest.update_document_status", fake_update_document_status)
    monkeypatch.setattr("app.routes.ingest.ObjectStore", FakeObjectStore)
    monkeypatch.setattr("app.routes.ingest.extract_text", fake_extract_text)

    response = client.post(
        "/ingest",
        json={
            "document_id": "12",
            "user_id": "7",
            "storage_key": "documents/7/bad.png",
            "source_type": "bill",
        },
    )

    assert response.status_code == 500
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "ingestion_failed"
    assert callback_calls == ["processing", "failed"]


def test_ingest_failure_survives_callback_errors(monkeypatch):
    async def flaky_update_document_status(document_id, status, processed_at=None):
        if status == "failed":
            raise RuntimeError("node unavailable")

    class FakeObjectStore:
        async def download(self, storage_key):
            return b"bad file"

    async def fake_extract_text(file_bytes, storage_key):
        raise ValueError("extract failed")

    monkeypatch.setattr("app.routes.ingest.update_document_status", flaky_update_document_status)
    monkeypatch.setattr("app.routes.ingest.ObjectStore", FakeObjectStore)
    monkeypatch.setattr("app.routes.ingest.extract_text", fake_extract_text)

    response = client.post(
        "/ingest",
        json={
            "document_id": "13",
            "user_id": "7",
            "storage_key": "documents/7/bad.jpg",
            "source_type": "other",
        },
    )

    assert response.status_code == 500
    assert response.json()["status"] == "failed"


def test_clean_extracted_text_removes_browser_pdf_artifacts():
    raw_text = """
    11/03/2026, 22:54 Finances E2E
    Finances E2E Validation
    This is a text PDF for ingestion testing.
    file:///C:/Users/msvic/Documents/finances-ai-service/e2e-source.html
    Income increased by ten percent in March 2026.
    Income increased by ten percent in March 2026.
    Rent remains fixed at one thousand euros.
    """

    cleaned = _clean_extracted_text(raw_text)

    assert "11/03/2026, 22:54" not in cleaned
    assert "file:///" not in cleaned
    assert cleaned.count("Income increased by ten percent in March 2026.") == 1
    assert "Rent remains fixed at one thousand euros." in cleaned


def test_clean_extracted_text_normalizes_unicode_spacing():
    raw_text = "Salary\u200b  increased\u00a0to  5000\x00\n\nSalary\u200b  increased\u00a0to  5000"

    cleaned = _clean_extracted_text(raw_text)

    assert cleaned == "Salary increased to 5000"
