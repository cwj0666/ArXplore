import random
import re
from typing import Any

import pytest

from arxplore.text import (
    collapse_whitespace,
    strip_control_characters,
    strip_nul_and_surrogates,
    strip_nul_and_surrogates_deep,
    strip_surrogates,
    strip_surrogates_deep,
    to_text_without_surrogates,
)

EVERY_CODE_POINT = "".join(map(chr, range(0x110000)))
LEGACY_CONTROL_CHARS = {chr(code) for code in range(32) if chr(code) not in "\n\t"} | {"\x7f"}
LEGACY_QUERY_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def legacy_repository_sanitize(value: str) -> str:
    return "".join(char for char in value if not 0xD800 <= ord(char) <= 0xDFFF)


def legacy_prepare_strip_surrogates(value: Any) -> str:
    return "".join(char for char in str(value or "") if not 0xD800 <= ord(char) <= 0xDFFF)


def legacy_raw_sanitize(value: str) -> str:
    return "".join(char for char in value if char != "\x00" and not 0xD800 <= ord(char) <= 0xDFFF)


def legacy_strip_control_characters(value: Any) -> str:
    return "".join(char for char in str(value or "") if char not in LEGACY_CONTROL_CHARS)


def legacy_query_control(value: Any) -> str:
    return LEGACY_QUERY_CONTROL_CHARACTERS.sub("", str(value or ""))


def legacy_sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return legacy_repository_sanitize(value)
    if isinstance(value, list):
        return [legacy_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [legacy_sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {legacy_repository_sanitize(str(key)): legacy_sanitize_json_value(item) for key, item in value.items()}
    return value


def legacy_sanitize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return legacy_raw_sanitize(value)
    if isinstance(value, (list, tuple)):
        return [legacy_sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {legacy_raw_sanitize(str(key)): legacy_sanitize_payload(item) for key, item in value.items()}
    return value


def _random_texts(count: int) -> list[str]:
    generator = random.Random(20260929)
    pools = [
        [chr(code) for code in range(0x20)] + ["\x7f"],
        [chr(code) for code in range(0xD7F0, 0xE010)],
        list("abc 가나다\n\t "),
        ["\U0001f600", "\U00010000", "\U0010ffff"],
    ]
    return [
        "".join(generator.choice(generator.choice(pools)) for _ in range(generator.randint(0, 40)))
        for _ in range(count)
    ]


RANDOM_TEXTS = _random_texts(500)


def test_surrogate_stripping_matches_the_four_legacy_copies_on_every_code_point() -> None:
    expected = legacy_repository_sanitize(EVERY_CODE_POINT)

    assert strip_surrogates(EVERY_CODE_POINT) == expected
    assert to_text_without_surrogates(EVERY_CODE_POINT) == legacy_prepare_strip_surrogates(EVERY_CODE_POINT)
    assert len(EVERY_CODE_POINT) - len(expected) == 0xE000 - 0xD800


def test_raw_stripping_also_removes_nul_on_every_code_point() -> None:
    assert strip_nul_and_surrogates(EVERY_CODE_POINT) == legacy_raw_sanitize(EVERY_CODE_POINT)
    assert strip_nul_and_surrogates("a\x00b\ud800") == "ab"
    assert strip_surrogates("a\x00b\ud800") == "a\x00b"


def test_control_character_stripping_matches_both_legacy_rules_on_every_code_point() -> None:
    expected = legacy_strip_control_characters(EVERY_CODE_POINT)

    assert strip_control_characters(EVERY_CODE_POINT) == expected
    assert legacy_query_control(EVERY_CODE_POINT) == expected
    assert strip_control_characters("a\tb\nc\rd\x00e\x7f") == "a\tb\ncde"


def test_random_texts_match_the_legacy_rules() -> None:
    for text in RANDOM_TEXTS:
        assert strip_surrogates(text) == legacy_repository_sanitize(text)
        assert strip_nul_and_surrogates(text) == legacy_raw_sanitize(text)
        assert strip_control_characters(text) == legacy_strip_control_characters(text)
        assert collapse_whitespace(text) == " ".join(text.split())


@pytest.mark.parametrize("value", [None, "", 0, False, 5, 2.5, "x\ud800"])
def test_coercing_variants_turn_falsy_values_into_empty_text(value: Any) -> None:
    assert to_text_without_surrogates(value) == legacy_prepare_strip_surrogates(value)
    assert strip_control_characters(value) == legacy_strip_control_characters(value)


def test_deep_cleaning_matches_the_legacy_json_sanitizers() -> None:
    payload = {
        "k\x00\ud800": ("a\x00\udfff", 1, None, 2.5, True),
        "nested": [{"x\ud800": ["\x00", {"deep\x00": "v\ud800"}]}],
        3: "number key",
        "bytes": b"\x00",
    }

    assert strip_surrogates_deep(payload) == legacy_sanitize_json_value(payload)
    assert strip_nul_and_surrogates_deep(payload) == legacy_sanitize_payload(payload)
    assert strip_nul_and_surrogates_deep({"k\x00": ("a\x00", 1, None)}) == {"k": ["a", 1, None]}
    assert strip_surrogates_deep({"k\x00": ("a\x00", 1, None)}) == {"k\x00": ["a\x00", 1, None]}
    for text in RANDOM_TEXTS[:50]:
        nested = {text: [text, (text,)]}
        assert strip_surrogates_deep(nested) == legacy_sanitize_json_value(nested)
        assert strip_nul_and_surrogates_deep(nested) == legacy_sanitize_payload(nested)
