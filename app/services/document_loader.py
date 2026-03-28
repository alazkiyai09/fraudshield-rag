import csv
import io
from dataclasses import dataclass

from app.config import Settings


@dataclass(slots=True)
class DocumentChunk:
    content: str
    metadata: dict


class DocumentLoaderService:
    """Parses incoming files and produces chunked text records."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_and_chunk(
        self,
        filename: str,
        file_bytes: bytes,
        source_type: str,
        metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        normalized_source_type = source_type.strip().lower()
        metadata = metadata or {}

        if normalized_source_type == "pdf":
            extracted_docs = self._extract_from_pdf(file_bytes)
        elif normalized_source_type == "csv":
            extracted_docs = self._extract_from_csv(file_bytes)
        elif normalized_source_type == "text":
            extracted_docs = self._extract_from_text(file_bytes)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        chunks: list[DocumentChunk] = []
        for page_index, page_data in enumerate(extracted_docs):
            text = page_data["text"].strip()
            if not text:
                continue

            for chunk_index, chunk in enumerate(self._split_text(text)):
                cleaned_chunk = chunk.strip()
                if not cleaned_chunk:
                    continue

                chunk_metadata = {
                    **metadata,
                    "source": filename,
                    "source_type": normalized_source_type,
                    "chunk_index": chunk_index,
                }

                page_number = page_data.get("page")
                if page_number is not None:
                    chunk_metadata["page"] = page_number

                chunks.append(DocumentChunk(content=cleaned_chunk, metadata=chunk_metadata))

        return chunks

    def _extract_from_pdf(self, file_bytes: bytes) -> list[dict]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ValueError("PDF support requires pypdf installed.") from exc

        reader = PdfReader(io.BytesIO(file_bytes))
        documents: list[dict] = []
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            documents.append({"text": text, "page": index + 1})
        return documents

    def _extract_from_csv(self, file_bytes: bytes) -> list[dict]:
        decoded = file_bytes.decode("utf-8", errors="ignore")
        stream = io.StringIO(decoded)
        reader = csv.DictReader(stream)

        rows = []
        for row in reader:
            row_parts = []
            for key, value in row.items():
                if key is None:
                    continue
                row_parts.append(f"{key}: {(value or '').strip()}")
            rows.append(" | ".join(row_parts))

        if not rows:
            return [{"text": decoded, "page": None}]
        return [{"text": "\n".join(rows), "page": None}]

    def _extract_from_text(self, file_bytes: bytes) -> list[dict]:
        decoded = file_bytes.decode("utf-8", errors="ignore")
        return [{"text": decoded, "page": None}]

    def _split_text(self, text: str) -> list[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            except ImportError:
                return self._split_text_fallback(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return splitter.split_text(text)

    def _split_text_fallback(self, text: str) -> list[str]:
        chunk_size = self.settings.chunk_size
        overlap = min(self.settings.chunk_overlap, chunk_size - 1)
        stride = max(1, chunk_size - overlap)

        chunks: list[str] = []
        for start in range(0, len(text), stride):
            chunk = text[start : start + chunk_size]
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(text):
                break
        return chunks
