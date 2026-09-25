from dataclasses import dataclass, field
from typing import Any


@dataclass
class FulltextParseResult:
    """PDF 파싱 결과."""

    text: str
    sections: list[dict[str, Any]]
    source: str
    quality_metrics: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    parser_metadata: dict[str, Any] = field(default_factory=dict)
