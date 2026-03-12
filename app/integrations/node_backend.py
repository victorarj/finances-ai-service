import logging
from datetime import datetime, timezone

import httpx

from app.config import get_settings


logger = logging.getLogger(__name__)


async def update_document_status(
    document_id: int,
    status: str,
    *,
    processed_at: datetime | None = None,
) -> None:
    settings = get_settings()
    payload: dict[str, str] = {"status": status}

    if processed_at is not None:
        payload["processed_at"] = processed_at.astimezone(timezone.utc).isoformat()

    url = f"{settings.normalized_node_backend_url}/api/v1/documents/{document_id}/status"
    headers = {"x-internal-api-secret": settings.internal_api_secret}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.patch(url, json=payload, headers=headers)
            response.raise_for_status()
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Node callback failed for document %s status %s: %s",
            document_id,
            status,
            exc,
        )
