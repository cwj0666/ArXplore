"""핵심 프롬프트 템플릿을 모아 노출하는 모듈"""

from .agent import AGENT_SYSTEM_PROMPT
from .key_findings import KEY_FINDINGS_PROMPT
from .overview import OVERVIEW_PROMPT
from .paper_chat import PAPER_CHAT_PROMPT
from .summary import (
    SUMMARY_BUCKET_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_SECTION_PROMPT,
)
from .translation import TRANSLATION_PROMPT

__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "PAPER_CHAT_PROMPT",
    "SUMMARY_PROMPT",
    "SUMMARY_BUCKET_PROMPT",
    "SUMMARY_SECTION_PROMPT",
    "KEY_FINDINGS_PROMPT",
    "OVERVIEW_PROMPT",
    "TRANSLATION_PROMPT",
]
