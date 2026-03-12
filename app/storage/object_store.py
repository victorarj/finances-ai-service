import asyncio

from minio import Minio

from app.config import get_settings


class ObjectStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._bucket = settings.minio_bucket
        self._client = Minio(
            settings.minio_host,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    async def download(self, storage_key: str) -> bytes:
        return await asyncio.to_thread(self._download_sync, storage_key)

    async def presigned_get_url(self, storage_key: str) -> str:
        return await asyncio.to_thread(self._client.presigned_get_object, self._bucket, storage_key)

    def _download_sync(self, storage_key: str) -> bytes:
        response = self._client.get_object(self._bucket, storage_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
