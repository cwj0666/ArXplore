import os
from pathlib import Path
from typing import Any

from tests.golden.adapters import ParsedPdf, ParserAdapter
from tests.golden.store import DATA_DIR, md5_text, read_json, sha256_bytes, sha256_text

FORMAT = 1
MANIFEST_PATH = DATA_DIR / "g3_pdfs.json"
GOLDEN_DIR = DATA_DIR / "g3_pypdf"
PDF_DIR_ENV = "GOLDEN_PDF_DIR"
DEFAULT_PDF_DIR = Path.home() / "portfolio" / "02_projects" / "arxplore" / "data" / "pdfs"
PYPDF_SOURCE = "pdf"
PARSED_KEYS = ("parse", "chunks", "chunk_metrics")


def pdf_dir() -> Path:
    configured = os.environ.get(PDF_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_PDF_DIR


def load_manifest() -> list[dict[str, Any]]:
    return list(read_json(MANIFEST_PATH)["papers"])


def pdf_path(entry: dict[str, Any]) -> Path:
    return pdf_dir() / f"{entry['arxiv_id']}v{entry['version']}.pdf"


def golden_path(entry: dict[str, Any], directory: Path = GOLDEN_DIR) -> Path:
    return directory / f"{entry['arxiv_id']}.json.gz"


def page_records(page_texts: list[str]) -> list[dict[str, Any]]:
    return [{"text": text, "text_sha256": sha256_text(text), "text_length": len(text)} for text in page_texts]


def section_record(section: dict[str, Any]) -> dict[str, Any]:
    text = str(section.get("text") or "")
    fields = {key: value for key, value in section.items() if key != "text"}
    return {**fields, "text": text, "text_sha256": sha256_text(text), "text_length": len(text)}


def chunk_record(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_index": chunk["chunk_index"],
        "chunk_text_md5": md5_text(chunk["chunk_text"]),
        "section_title": chunk["section_title"],
        "token_count": chunk["token_count"],
        "metadata": chunk["metadata"],
    }


def parsed_record(adapter: ParserAdapter, parsed: ParsedPdf) -> dict[str, Any]:
    chunks = adapter.build_chunks(parsed["text"], parsed["sections"])
    return {
        "parse": {
            "source": parsed["source"],
            "text_sha256": sha256_text(parsed["text"]),
            "text_length": len(parsed["text"]),
            "quality_metrics": parsed["quality_metrics"],
            "parser_metadata": parsed["parser_metadata"],
            "artifacts": parsed["artifacts"],
            "sections": [section_record(section) for section in parsed["sections"]],
        },
        "chunks": [chunk_record(chunk) for chunk in chunks],
        "chunk_metrics": adapter.summarize_chunks(chunks),
    }


def build_record(adapter: ParserAdapter, entry: dict[str, Any], pdf_bytes: bytes) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "arxiv_id": entry["arxiv_id"],
        "version": entry["version"],
        "pdf_sha256": sha256_bytes(pdf_bytes),
        "captured_with": adapter.describe(),
        "pages": page_records(adapter.extract_page_texts(pdf_bytes)),
        **parsed_record(adapter, adapter.parse_pdf_bytes(pdf_bytes)),
    }


def golden_page_texts(golden: dict[str, Any]) -> list[str]:
    return [page["text"] for page in golden["pages"]]


def parsed_part(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in PARSED_KEYS}
