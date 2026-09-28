import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.golden import environment, g3_layout, g3_pypdf
from tests.golden.adapters import load_parser_adapter
from tests.golden.store import sha256_bytes, sha256_text, variant_path, write_golden

DATABASE_SQL = """
    SELECT f.text, f.sections, f.parser_metadata, f.source,
           COALESCE((SELECT array_agg(md5(c.chunk_text) ORDER BY c.chunk_index)
                     FROM paper_chunks c WHERE c.arxiv_id = f.arxiv_id), '{}')
    FROM paper_fulltexts f
    WHERE f.arxiv_id = %s
"""


def compare_with_database(record: dict[str, Any], row: tuple[Any, ...] | None) -> dict[str, Any]:
    if row is None:
        return {"in_database": False}
    text, sections, parser_metadata, source, chunk_md5s = row
    ours = [{"title": section["title"], "text": section["text"]} for section in record["parse"]["sections"]]
    theirs = [{"title": section.get("title"), "text": section.get("text")} for section in sections or []]
    same_titles = [section["title"] for section in ours] == [section["title"] for section in theirs]
    differing = [index for index, (left, right) in enumerate(zip(ours, theirs, strict=False)) if left != right]
    our_chunks = [chunk["chunk_text_md5"] for chunk in record["chunks"]]
    return {
        "in_database": True,
        "database_source": source,
        "same_sections": ours == theirs,
        "same_section_titles": same_titles,
        "section_counts": [len(ours), len(theirs)],
        "differing_section_indexes": differing[:10],
        "same_text": record["parse"]["text_sha256"] == sha256_text(text or ""),
        "same_parser_metadata": record["parse"]["parser_metadata"] == parser_metadata,
        "parser_metadata": [record["parse"]["parser_metadata"], parser_metadata],
        "same_chunks": our_chunks == list(chunk_md5s),
        "chunk_counts": [len(our_chunks), len(chunk_md5s)],
    }


def capture(out_dir: Path, variant: str | None, report_path: Path | None) -> int:
    adapter = load_parser_adapter()
    url = g3_layout.layout_url()
    database_url = environment.database_url()
    report: dict[str, Any] = {}
    for entry in g3_pypdf.load_manifest():
        pdf_bytes = g3_pypdf.pdf_path(entry).read_bytes()
        if sha256_bytes(pdf_bytes) != entry["sha256"]:
            raise SystemExit(f"{g3_pypdf.pdf_path(entry)} does not match the manifest sha256")
        started = time.perf_counter()
        response = g3_layout.request_layout(pdf_bytes, url)
        elapsed = time.perf_counter() - started
        record = g3_layout.build_record(adapter, entry, pdf_bytes, response)
        path = out_dir / f"{entry['arxiv_id']}.json.gz"
        written = write_golden(variant_path(path, variant) if variant else path, record)
        rows = environment.query_rows(database_url, DATABASE_SQL, (entry["arxiv_id"],))
        comparison = compare_with_database(record, rows[0] if rows else None)
        comparison["seconds"] = round(elapsed, 1)
        comparison["segments"] = len(record["response"])
        comparison["gz_bytes"] = len(written)
        report[entry["arxiv_id"]] = comparison
        print(
            f"{entry['arxiv_id']}v{entry['version']}: {elapsed:.0f}s segments={len(record['response'])}"
            f" sections={len(record['parse']['sections'])} chunks={len(record['chunks'])} gz={len(written)}"
            f" db_sections_equal={comparison.get('same_sections')} db_chunks_equal={comparison.get('same_chunks')}",
            flush=True,
        )
    if report_path is not None:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.goldens.capture_g3_layout")
    parser.add_argument("--out", type=Path, default=g3_layout.GOLDEN_DIR)
    parser.add_argument("--variant", help="write <id>.<variant>.json.gz next to the original goldens")
    parser.add_argument("--report", type=Path, help="write the database comparison as JSON")
    args = parser.parse_args(argv)
    return capture(args.out, args.variant, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
