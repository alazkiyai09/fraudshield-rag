import sys
import types

import pytest

from app.config import Settings
from app.services.document_loader import DocumentLoaderService


def test_load_and_chunk_csv_adds_metadata():
    settings = Settings(chunk_size=60, chunk_overlap=10)
    loader = DocumentLoaderService(settings)

    csv_bytes = b"id,amount,risk\nTX1,9000,high\nTX2,150,low\n"
    chunks = loader.load_and_chunk(
        filename="demo.csv",
        file_bytes=csv_bytes,
        source_type="csv",
        metadata={"category": "transactions", "year": 2025},
    )

    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "demo.csv"
    assert chunks[0].metadata["source_type"] == "csv"
    assert chunks[0].metadata["category"] == "transactions"


def test_extract_text_and_fallback_splitter():
    settings = Settings(chunk_size=20, chunk_overlap=5)
    loader = DocumentLoaderService(settings)

    docs = loader._extract_from_text(b"alpha beta gamma delta epsilon zeta")
    assert docs[0]["text"].startswith("alpha")

    chunks = loader._split_text_fallback(docs[0]["text"])
    assert len(chunks) >= 2


def test_extract_pdf_with_mocked_pypdf(monkeypatch):
    settings = Settings(chunk_size=50, chunk_overlap=10)
    loader = DocumentLoaderService(settings)

    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdfReader:
        def __init__(self, stream) -> None:
            self.pages = [FakePage("page one"), FakePage("page two")]

    fake_module = types.SimpleNamespace(PdfReader=FakePdfReader)
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    docs = loader._extract_from_pdf(b"%PDF-1.4 fake")
    assert docs == [{"text": "page one", "page": 1}, {"text": "page two", "page": 2}]


def test_load_and_chunk_rejects_unknown_source_type():
    settings = Settings()
    loader = DocumentLoaderService(settings)

    with pytest.raises(ValueError):
        loader.load_and_chunk(
            filename="x.unknown",
            file_bytes=b"data",
            source_type="unknown",
            metadata=None,
        )
