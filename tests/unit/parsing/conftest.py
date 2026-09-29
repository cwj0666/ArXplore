import io
from collections.abc import Callable

import pytest
from pypdf import PdfReader, PdfWriter

type PdfBuilder = Callable[[list[str]], bytes]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_text_pdf(lines: list[str]) -> bytes:
    operators = ["BT", "/F1 10 Tf", "14 TL", "50 780 Td"]
    operators.extend(f"({_escape(line)}) Tj T*" for line in lines)
    operators.append("ET")
    stream = "\n".join(operators).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(output)


def encrypt_pdf(pdf_bytes: bytes, password: str = "pw") -> bytes:
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(pdf_bytes)).pages:
        writer.add_page(page)
    writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def text_pdf() -> PdfBuilder:
    return build_text_pdf


@pytest.fixture
def encrypted_pdf() -> PdfBuilder:
    return lambda lines: encrypt_pdf(build_text_pdf(lines))
