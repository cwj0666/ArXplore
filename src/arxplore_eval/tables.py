from collections.abc import Sequence

MISSING = "n/a"


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def format_rate(value: float | None) -> str:
    return MISSING if value is None else f"{value:.3f}"


def format_ms(value: float | None) -> str:
    return MISSING if value is None else f"{value:.0f}"


def csv_number(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"
