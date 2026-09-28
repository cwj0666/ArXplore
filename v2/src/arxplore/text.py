import re
from collections.abc import Callable
from typing import Any

_SURROGATES = re.compile(r"[\ud800-\udfff]")
_NUL_AND_SURROGATES = re.compile(r"[\x00\ud800-\udfff]")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def strip_surrogates(text: str) -> str:
    return _SURROGATES.sub("", text)


def to_text_without_surrogates(value: object) -> str:
    return strip_surrogates(str(value or ""))


def strip_nul_and_surrogates(text: str) -> str:
    return _NUL_AND_SURROGATES.sub("", text)


def strip_control_characters(value: object) -> str:
    return _CONTROL_CHARACTERS.sub("", str(value or ""))


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _clean_json(value: Any, clean: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, (list, tuple)):
        return [_clean_json(item, clean) for item in value]
    if isinstance(value, dict):
        return {clean(str(key)): _clean_json(item, clean) for key, item in value.items()}
    return value


def strip_surrogates_deep(value: Any) -> Any:
    return _clean_json(value, strip_surrogates)


def strip_nul_and_surrogates_deep(value: Any) -> Any:
    return _clean_json(value, strip_nul_and_surrogates)
