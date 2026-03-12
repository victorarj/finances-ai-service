import asyncio
import re
import unicodedata
from io import BytesIO
from pathlib import Path

import pdfplumber
import pytesseract
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ZERO_WIDTH_CHARACTERS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
}


def _clean_extracted_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")

    for character in ZERO_WIDTH_CHARACTERS:
        normalized = normalized.replace(character, "")

    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"file:///[\w\-./:%#?=&+~]+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4},\s+\d{1,2}:\d{2}\b", "", normalized)
    normalized = normalized.replace("\x00", "")
    normalized = re.sub(r"[ \t]+", " ", normalized)

    lines: list[str] = []
    previous_line = None
    for raw_line in normalized.splitlines():
        line = raw_line.strip(" \t|")
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if line == previous_line:
            continue
        lines.append(line)
        previous_line = line

    return "\n".join(lines).strip()


def _extract_pdf_text(file_bytes: bytes) -> str:
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(page for page in pages if page.strip()).strip()


def _extract_image_text(file_bytes: bytes) -> str:
    with Image.open(BytesIO(file_bytes)) as image:
        return pytesseract.image_to_string(image).strip()


async def extract_text(file_bytes: bytes, storage_key: str) -> str:
    suffix = Path(storage_key).suffix.lower()

    if suffix == ".pdf":
        text = await asyncio.to_thread(_extract_pdf_text, file_bytes)
    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        text = await asyncio.to_thread(_extract_image_text, file_bytes)
    else:
        raise ValueError(f"Unsupported file type for extraction: {suffix or 'unknown'}")

    normalized = _clean_extracted_text(text)
    if not normalized:
        raise ValueError("No text could be extracted from document")
    return normalized
