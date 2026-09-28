import gzip
import math

import pytest

from tests.golden.store import canonical_json, diff, normalize, read_golden, write_golden


def test_canonical_json_sorts_keys_and_escapes_non_ascii():
    data = canonical_json({"b": 1, "a": ["가", 0.5]})

    assert data == b'{\n "a": [\n  "\\uac00",\n  0.5\n ],\n "b": 1\n}\n'


def test_normalize_folds_negative_zero_and_rejects_unsupported_values():
    assert math.copysign(1.0, normalize(-0.0)) == 1.0
    assert normalize((1, 2)) == [1, 2]
    with pytest.raises(ValueError):
        normalize(float("nan"))
    with pytest.raises(TypeError):
        normalize({1: "non-string key"})
    with pytest.raises(TypeError):
        normalize({"value": {1, 2}})


def test_write_golden_is_byte_stable_and_round_trips(tmp_path):
    value = {"text": "섹션 본문", "score": 0.1 + 0.2, "items": [1, True, None]}
    first = write_golden(tmp_path / "a.json.gz", value)
    second = write_golden(tmp_path / "b.json.gz", value)

    assert first == second
    assert first[4:8] == b"\x00\x00\x00\x00"
    assert read_golden(tmp_path / "a.json.gz") == value
    assert gzip.decompress(first) == canonical_json(value)


def test_diff_is_strict_about_types_and_reports_paths():
    expected = {"count": 1, "ratio": 2.0, "flag": True, "items": ["a", "b"], "gone": 0}
    actual = {"count": 1.0, "ratio": 2, "flag": 1, "items": ["a"], "new": 0}

    assert diff(expected, actual) == [
        "$.gone: missing",
        "$.new: unexpected",
        "$.count: type int -> float (1 -> 1.0)",
        "$.flag: type bool -> int (True -> 1)",
        "$.items: length 2 -> 1",
        "$.ratio: type float -> int (2.0 -> 2)",
    ]


def test_diff_points_at_the_first_differing_character_of_long_strings():
    expected = "x" * 100 + "expected tail"
    actual = "x" * 100 + "actual tail"

    [problem] = diff({"text": expected}, {"text": actual})

    assert problem.startswith("$.text: strings differ at char 100 (lengths 113 -> 111)")
    assert diff({"text": expected}, {"text": expected}) == []


def test_write_golden_keeps_an_existing_file_with_the_same_content(tmp_path):
    path = tmp_path / "golden.json.gz"
    path.write_bytes(gzip.compress(canonical_json({"a": 1}), compresslevel=1, mtime=123))
    original = path.read_bytes()

    assert write_golden(path, {"a": 1}) == original
    assert path.read_bytes() == original
    assert write_golden(path, {"a": 2}) != original
    assert read_golden(path) == {"a": 2}
