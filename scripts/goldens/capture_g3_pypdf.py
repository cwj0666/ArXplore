import argparse
import re
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from tests.golden import g3_pypdf
from tests.golden.adapters import implementation_name, load_parser_adapter
from tests.golden.store import (
    diff,
    format_problems,
    read_json,
    sha256_bytes,
    variant_path,
    write_golden,
    write_json,
)

USER_AGENT = "ArXplore-golden-capture/1.0 (parser regression fixtures; one PDF per request, >=3s apart)"
REQUEST_INTERVAL_SECONDS = 3.5
DOWNLOAD_TIMEOUT_SECONDS = 120
ENTRY_KEYS = ("arxiv_id", "version", "sha256", "bytes", "reason")
VERSIONED_FILENAME = re.compile(r'filename="?(?P<arxiv_id>\d{4}\.\d{4,5})v(?P<version>\d+)\.pdf"?')


def _download(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        return response.read(), response.headers.get("Content-Disposition", "")


def fetch() -> int:
    manifest = read_json(g3_pypdf.MANIFEST_PATH)
    g3_pypdf.pdf_dir().mkdir(parents=True, exist_ok=True)
    next_request_at = 0.0
    for entry in manifest["papers"]:
        if entry.get("version") and g3_pypdf.pdf_path(entry).is_file():
            if sha256_bytes(g3_pypdf.pdf_path(entry).read_bytes()) != entry.get("sha256"):
                raise SystemExit(f"{g3_pypdf.pdf_path(entry)} does not match the manifest sha256")
            print(f"{entry['arxiv_id']}v{entry['version']}: already present")
            continue
        suffix = f"v{entry['version']}" if entry.get("version") else ""
        url = f"https://arxiv.org/pdf/{entry['arxiv_id']}{suffix}"
        time.sleep(max(0.0, next_request_at - time.monotonic()))
        data, disposition = _download(url)
        next_request_at = time.monotonic() + REQUEST_INTERVAL_SECONDS
        match = VERSIONED_FILENAME.search(disposition)
        if not data.startswith(b"%PDF") or match is None or match["arxiv_id"] != entry["arxiv_id"]:
            raise SystemExit(f"{url} did not return the expected PDF (Content-Disposition {disposition!r})")
        version = int(match["version"])
        digest = sha256_bytes(data)
        if entry.get("version") not in (None, version) or entry.get("sha256") not in (None, digest):
            raise SystemExit(f"{url} returned v{version} {digest}, which differs from the manifest")
        entry.update(version=version, sha256=digest, bytes=len(data))
        g3_pypdf.pdf_path(entry).write_bytes(data)
        print(f"{entry['arxiv_id']}v{version}: {len(data)} bytes")
    manifest["papers"] = [{key: entry[key] for key in ENTRY_KEYS if key in entry} for entry in manifest["papers"]]
    write_json(g3_pypdf.MANIFEST_PATH, manifest)
    return 0


def capture(out_dir: Path, implementation: str | None, variant: str | None) -> int:
    adapter = load_parser_adapter(implementation)
    for entry in g3_pypdf.load_manifest():
        path = g3_pypdf.pdf_path(entry)
        pdf_bytes = path.read_bytes()
        if sha256_bytes(pdf_bytes) != entry["sha256"]:
            raise SystemExit(f"{path} does not match the manifest sha256")
        started = time.perf_counter()
        record = g3_pypdf.build_record(adapter, entry, pdf_bytes)
        elapsed = time.perf_counter() - started
        source = record["parse"]["source"]
        if source != g3_pypdf.PYPDF_SOURCE:
            raise SystemExit(f"{entry['arxiv_id']}: parse_source is {source!r}, expected {g3_pypdf.PYPDF_SOURCE!r}")
        replay = g3_pypdf.parsed_record(adapter, adapter.parse_page_texts(g3_pypdf.golden_page_texts(record)))
        problems = diff(g3_pypdf.parsed_part(record), replay)
        if problems:
            raise SystemExit(format_problems(f"{entry['arxiv_id']} page replay", problems))
        path = g3_pypdf.golden_path(entry, out_dir)
        written = write_golden(variant_path(path, variant) if variant else path, record)
        print(
            f"{entry['arxiv_id']}v{entry['version']}: pages={len(record['pages'])}"
            f" sections={len(record['parse']['sections'])} chunks={len(record['chunks'])}"
            f" text={record['parse']['text_length']} gz={len(written)} parse={elapsed:.1f}s"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.goldens.capture_g3_pypdf")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download the manifest PDFs from arXiv and record version and sha256")
    capture_parser = commands.add_parser("capture", help="parse the PDFs and write the G3 golden files")
    capture_parser.add_argument("--out", type=Path, default=g3_pypdf.GOLDEN_DIR)
    capture_parser.add_argument("--implementation")
    capture_parser.add_argument(
        "--variant",
        default=implementation_name(),
        help="write <name>.<variant>.json.gz next to the original golden; an empty value writes the original",
    )
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return fetch()
    return capture(args.out, args.implementation, args.variant)


if __name__ == "__main__":
    raise SystemExit(main())
